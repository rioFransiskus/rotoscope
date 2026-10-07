"""Pengukuran Tahap 3 T-402 (alat sekali pakai; semua pada SALINAN klip di folder `--s3`, klip asli tidak diubah):

    python scripts/t402_measure.py --s3 <dir> regress       # amplitudo 0 byte-identik T-401 (hash strokes/ klip asli sebelum T-402)
    python scripts/t402_measure.py --s3 <dir> cost          # 3 run bersih per klip: waktu per frame + rincian, ukuran, memori puncak
    python scripts/t402_measure.py --s3 <dir> determinism   # dua run dari nol, --limit, --from, resume, basi, salinan SVG
    python scripts/t402_measure.py --s3 <dir> safety        # invarian keselamatan seluruh klip + statistik getar
    python scripts/t402_measure.py --s3 <dir> stats         # variasi akibat WAKTU vs GERAK per tipe + loncatan G saat track_id berganti
    python scripts/t402_measure.py --s3 <dir> cost_amp0     # pembanding A/B waktu: jitter mati pada kondisi mesin yang sama

<dir> berisi: config.yaml (paths), eval.yaml / amp0.yaml / amp0_weird.yaml (style), work/clips/<klip>/ (salinan), out/. Hasil JSON ke
work/t402/. Perintah stage dipanggil lewat subprocess (CLI nyata)."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "src"))

import jitter_metrics as jm  # noqa: E402

from rotoscope import stylize as sty  # noqa: E402
from rotoscope.config import load_style  # noqa: E402

OUT = ROOT / "work" / "t402"
CLIPS = ("test_short", "test")
PY = sys.executable


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def strokes_hashes(d: Path) -> dict[str, str]:
    return {p.name: sha(p) for p in sorted(d.glob("frame_*")) if p.suffix in (".svg", ".png")}


def before_hashes(clip: str) -> dict[str, str]:
    out = {}
    for line in (OUT / f"before_strokes_{clip}.sha256").read_text(encoding="utf-8").splitlines():
        h, name = line.split(maxsplit=1)
        out[name.lstrip("*").strip()] = h
    return out


def stylize(s3: Path, clip: str, style: str, *extra: str) -> subprocess.CompletedProcess:
    cmd = [PY, "-m", "rotoscope.stylize", "--work-dir", str(s3 / "work" / "clips" / clip), "--style", str(s3 / style), *extra]
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")


def rm_strokes(s3: Path, clip: str) -> None:
    import shutil
    shutil.rmtree(s3 / "work" / "clips" / clip / "strokes", ignore_errors=True)


def write(name: str, obj) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"  work/t402/{name}")


# ── regress ────────────────────────────────────────
def cmd_regress(s3: Path) -> None:
    res = {}
    for clip in CLIPS:
        ref = before_hashes(clip)
        for style in ("amp0.yaml", "amp0_weird.yaml"):
            rm_strokes(s3, clip)
            r = stylize(s3, clip, style)
            got = strokes_hashes(s3 / "work" / "clips" / clip / "strokes")
            diff = sorted(k for k in set(ref) | set(got) if ref.get(k) != got.get(k))
            res[f"{clip}/{style}"] = {"rc": r.returncode, "files": len(got), "reference_files": len(ref), "identical": not diff,
                                      "n_diff": len(diff), "first_diff": diff[:5]}
            print(clip, style, res[f"{clip}/{style}"], flush=True)
    write("s3_regress_amplitude0.json", res)


# ── cost ───────────────────────────────────────────
def frame_records(strokes_dir: Path) -> list[dict]:
    return [r for r in (json.loads(x) for x in (strokes_dir / "frames.jsonl").read_text(encoding="utf-8").splitlines())
            if r.get("event") == "frame"]


PEAK_WRAPPER = ("import sys, ctypes; from ctypes import wintypes; from rotoscope import stylize\n"
                "rc = stylize.main(sys.argv[1:])\n"
                "class P(ctypes.Structure):\n"
                "    _fields_ = [('cb', wintypes.DWORD), ('pf', wintypes.DWORD), ('peak', ctypes.c_size_t)] + [(f'x{i}', ctypes.c_size_t) for i in range(7)]\n"
                "p = P(); p.cb = ctypes.sizeof(P)\n"
                "h = ctypes.WinDLL('kernel32').GetCurrentProcess()\n"
                "ctypes.WinDLL('psapi').GetProcessMemoryInfo(ctypes.c_void_p(h), ctypes.byref(p), p.cb)\n"
                "print('PEAK_MIB', p.peak / 2**20); sys.exit(rc)\n")


def cmd_cost(s3: Path, style: str = "eval.yaml", tag: str = "s3_cost") -> None:
    res = {}
    for clip in CLIPS:
        runs = []
        for k in range(3):
            rm_strokes(s3, clip)
            args = ["--work-dir", str(s3 / "work" / "clips" / clip), "--style", str(s3 / style)]
            t0 = time.perf_counter()
            pr = subprocess.run([PY, "-c", PEAK_WRAPPER, *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
            rc, wall = pr.returncode, time.perf_counter() - t0
            peak = next((float(x.split()[1]) for x in pr.stdout.splitlines() if x.startswith("PEAK_MIB")), float("nan"))
            sd = s3 / "work" / "clips" / clip / "strokes"
            recs = frame_records(sd)
            t = np.array([r["total_s"] for r in recs]) * 1000
            row = {"run": k + 1, "rc": rc, "wall_s": round(wall, 1), "peak_mib": round(peak, 1), "n": len(recs),
                   "total_mean_ms": float(t.mean()), "total_p95_ms": float(np.percentile(t, 95)), "total_max_ms": float(t.max()),
                   "n_gt_400": int((t > 400).sum()), "n_gt_250": int((t > 250).sum())}
            for key in ("geom_s", "raster_png_s", "svg_s", "write_s", "jitter_s"):
                row[key.replace("_s", "_mean_ms")] = float(np.mean([r.get(key, 0.0) for r in recs]) * 1000)
            row["png_mean_kib"] = float(np.mean([r["png_bytes"] for r in recs]) / 1024)
            row["svg_mean_kib"] = float(np.mean([r["svg_bytes"] for r in recs]) / 1024)
            row["png_total_mib"] = float(sum(r["png_bytes"] for r in recs) / 2**20)
            row["svg_total_mib"] = float(sum(r["svg_bytes"] for r in recs) / 2**20)
            runs.append(row)
            print(clip, row, flush=True)
        res[clip] = runs
    write(f"{tag}.json", res)


# ── determinism / resume / limit / from / stale ────
def cmd_determinism(s3: Path) -> None:
    res = {}
    for clip in CLIPS:
        d = s3 / "work" / "clips" / clip / "strokes"
        rm_strokes(s3, clip)
        stylize(s3, clip, "eval.yaml")
        full = strokes_hashes(d)
        out = {"files": len(full)}
        rm_strokes(s3, clip)
        stylize(s3, clip, "eval.yaml")
        out["two_runs_identical"] = strokes_hashes(d) == full
        rm_strokes(s3, clip)
        stylize(s3, clip, "eval.yaml", "--limit", "20")
        part = strokes_hashes(d)
        out["limit20_is_prefix"] = len(part) == 40 and all(full[k] == v for k, v in part.items())
        stylize(s3, clip, "eval.yaml")
        out["limit_then_full_identical"] = strokes_hashes(d) == full
        for p in list(d.glob("frame_0006[0-4].*")):
            p.unlink()                                                                    # resume: hanya frame hilang dihitung
        r = stylize(s3, clip, "eval.yaml")
        out["resume_identical"] = strokes_hashes(d) == full and "5 diproses" in r.stdout
        rm_strokes(s3, clip)
        stylize(s3, clip, "eval.yaml", "--from", "100", "--limit", "10")
        win = strokes_hashes(d)
        out["from100_limit10_equals_full"] = len(win) == 20 and all(full[k] == v for k, v in win.items())
        # basi: hold 3 → strokes dihitung ulang, lalu kembali ke eval = identik run awal
        (s3 / "hold3.yaml").write_text("jitter:\n  amplitude: 4.0\n  frequency: 0.053\n  temporal_drift: 0.35\n  hold_frames: 3\n", encoding="utf-8")
        r = stylize(s3, clip, "hold3.yaml")
        out["stale_hold3_recomputed"] = "basi" in r.stdout and strokes_hashes(d) != full
        stylize(s3, clip, "eval.yaml")
        out["back_to_eval_identical"] = strokes_hashes(d) == full
        res[clip] = out
        print(clip, out, flush=True)
    write("s3_determinism.json", res)


# ── safety / stats ─────────────────────────────────
def load_doc(s3: Path, clip: str, i: int) -> dict:
    return json.loads((s3 / "work" / "clips" / clip / "contours" / f"frame_{i:05d}.json").read_text(encoding="utf-8"))


def pct(a, q):
    a = np.asarray(a, float)
    return float(np.percentile(a, q)) if len(a) else float("nan")


def cmd_safety(s3: Path) -> None:
    """Invarian keselamatan di SELURUH frame kedua klip (s = 0, nilai evaluasi awal) + varian s dan amplitudo (informasi)."""
    configs = {"eval_s0": {}, "amp2_f0.053": {"jitter.amplitude": 2.0}, "amp3_f0.053_r0.159": {"jitter.amplitude": 3.0},
               "amp3.4_f0.053_r0.18": {"jitter.amplitude": 3.4}, "amp6_f0.035": {"jitter.amplitude": 6.0, "jitter.frequency": 0.035},
               "amp8_f0.0265": {"jitter.amplitude": 8.0, "jitter.frequency": 0.0265},
               "amp8_f0.053_LIPATAN": {"jitter.amplitude": 8.0, "jitter.frequency": 0.053}}
    res = {}
    for clip in CLIPS:
        meta = json.loads((s3 / "work" / "clips" / clip / "meta.json").read_text(encoding="utf-8"))
        n = int(meta["frame_count"])
        for cname, ov in configs.items():
            style = load_style(s3 / "eval.yaml", overrides=ov or None)
            g = sty.make_geometry(style, int(meta["working_width"]), int(meta["working_height"]))
            tol, bound = jm.joint_tolerance(g), jm.displacement_bound(g)
            acc = {"max_disp": [], "joint_max": [], "joint_n": 0, "joint_over_tol": 0, "new_cross": 0, "frames_new_cross": 0,
                   "min_det": [], "edge_ink": 0, "intrusion_over_base": 0, "rms": [], "frames": 0}
            for i in range(n):
                doc = load_doc(s3, clip, i)
                base, _ = sty.base_pieces(doc, g)
                jit = sty.jitter_pieces(base, g, int(doc["frame_index"]))
                acc["max_disp"].append(jm.max_displacement(base, jit))
                ch = jm.joint_changes(base, jit, g)
                acc["joint_n"] += len(ch)
                acc["joint_over_tol"] += int((ch > tol).sum())
                if len(ch):
                    acc["joint_max"].append(float(ch.max()))
                nc = jm.new_crossings(base, jit)
                acc["new_cross"] += nc
                acc["frames_new_cross"] += int(nc > 0)
                acc["min_det"].append(jm.pieces_jacobian_min_det(jit, g, int(doc["frame_index"])))
                if i % 4 == 0:
                    acc["edge_ink"] += jm.edge_ink_changed(base, jit, g)
                acc["intrusion_over_base"] += int(jm.extension_intrusion(jit, g) > jm.extension_intrusion(base, g) + 1e-9)
                pts = np.vstack([pc.points for pc in base]) if base else np.zeros((1, 2))
                acc["rms"].append(jm.jitter_stats(g, int(doc["frame_index"]), pts)["rms_per_channel_over_amp"])
                acc["frames"] += 1
            row = {"amp_px": g.jitter_amp, "r": sty.jitter_fold_r(style.jitter.amplitude, style.jitter.frequency, 0.0),
                   "bound_px": bound, "tol_px": tol, "max_disp_max": float(max(acc["max_disp"])),
                   "bound_ok": bool(max(acc["max_disp"]) <= bound + 0.02), "joints": acc["joint_n"],
                   "joint_change_p95": pct(acc["joint_max"], 95), "joint_change_max": float(max(acc["joint_max"] or [0])),
                   "joint_over_tol": acc["joint_over_tol"], "new_crossings": acc["new_cross"], "frames_with_new_crossings": acc["frames_new_cross"],
                   "min_det_min": float(min(acc["min_det"])), "frames_min_det_le_0.05": int(sum(d <= sty.JACOBIAN_MIN_DET for d in acc["min_det"])),
                   "edge_ink_changed_px_every4th": acc["edge_ink"], "intrusion_frames": acc["intrusion_over_base"],
                   "rms_over_amp_mean": float(np.mean(acc["rms"])), "frames": acc["frames"]}
            row["pass"] = bool(row["bound_ok"] and row["joint_over_tol"] == 0 and row["new_crossings"] == 0 and
                               row["min_det_min"] > sty.JACOBIAN_MIN_DET and row["edge_ink_changed_px_every4th"] == 0 and
                               row["intrusion_frames"] == 0)
            res[f"{clip}/{cname}"] = row
            print(clip, cname, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    write("s3_safety.json", res)


# ── stats: variasi akibat WAKTU vs GERAK (item 7), loncatan id (item 8) ─
MATCH_PX = 3.0                    # padanan titik antar frame (px output, per track_id + tipe)
STATIC_MOVE_PX, MOVING_MIN_PX = 0.5, 2.0


def piece_groups(pieces) -> dict:
    out: dict = {}
    for pc in pieces:
        out.setdefault((pc.track_id, pc.type), []).append(pc.points)
    return {k: np.vstack(v) for k, v in out.items()}


def cmd_stats(s3: Path) -> None:
    import dataclasses

    from scipy.spatial import cKDTree
    res: dict = {}
    for clip in CLIPS:
        meta = json.loads((s3 / "work" / "clips" / clip / "meta.json").read_text(encoding="utf-8"))
        n = int(meta["frame_count"])
        w, h = int(meta["working_width"]), int(meta["working_height"])
        modes = {"frame_hold1": {"jitter.hold_frames": 1}, "frame_hold2": {"jitter.hold_frames": 2},
                 "fixed": {"jitter.temporal_seed_mode": "fixed"}}
        gs = {m: sty.make_geometry(load_style(s3 / "eval.yaml", overrides=ov), w, h) for m, ov in modes.items()}
        amp = gs["frame_hold1"].jitter_amp
        acc = {m: {} for m in modes}
        new_ids = {t: [0, 0] for t in sty.TYPE_ORDER}
        id_jump = {s: [] for s in (0.25, 0.5, 1.0)}
        prev_base = None
        for i in range(n):
            doc = load_doc(s3, clip, i)
            base, _ = sty.base_pieces(doc, gs["frame_hold1"])
            cur_groups = piece_groups(base)
            if prev_base is not None:
                prev_groups = piece_groups(prev_base)
                prev_ids = set(prev_groups)
                for pc in base:
                    new_ids[pc.type][0] += int((pc.track_id, pc.type) not in prev_ids)
                    new_ids[pc.type][1] += 1
                pairs = []                                                   # (tipe, p_cur, p_prev, move)
                for key, pts in cur_groups.items():
                    if key in prev_groups:
                        d, j = cKDTree(prev_groups[key]).query(pts)
                        ok = d <= MATCH_PX
                        pairs.append((key[1], pts[ok], prev_groups[key][j[ok]], d[ok]))
                for m, g in gs.items():
                    for typ, pc_, pp_, mv in pairs:
                        if not len(pc_):
                            continue
                        v = np.hypot(*(jm.displacement_at(g, i, pc_) - jm.displacement_at(g, i - 1, pp_)).T) / amp
                        for cls, sel in (("diam", mv <= STATIC_MOVE_PX), ("bergerak", mv >= MOVING_MIN_PX)):
                            acc[m].setdefault((typ, cls), []).extend(v[sel].tolist())
                # loncatan G saat id berganti: strok id baru vs strok terdekat frame sebelumnya (id lama)
                if len(prev_base):
                    allp = np.vstack([pc.points for pc in prev_base])
                    owner = np.concatenate([np.full(len(pc.points), k) for k, pc in enumerate(prev_base)])
                    tree = cKDTree(allp)
                    for pc in base:
                        if (pc.track_id, pc.type) in prev_ids:
                            continue
                        d, j = tree.query(pc.points)
                        ok = d <= MATCH_PX
                        if ok.sum() < 5:
                            continue
                        old = prev_base[int(np.bincount(owner[j[ok]]).argmax())].track_id
                        for s in id_jump:
                            gj = dataclasses.replace(gs["frame_hold1"], jitter_s=s)
                            dn = jm.displacement_at(gj, i, pc.points[ok], pc.track_id)
                            do = jm.displacement_at(gj, i, pc.points[ok], old)
                            id_jump[s].extend((np.hypot(*(dn - do).T) / amp).tolist())
            prev_base = base
        out = {"amp_px": amp, "time_vs_motion": {}, "new_id_per_stroke_frame": {t: (a / b if b else None) for t, (a, b) in new_ids.items()},
               "id_jump_over_amp": {str(s): {"p50": pct(v, 50), "p95": pct(v, 95), "max": float(max(v)) if v else None, "n": len(v)}
                                    for s, v in id_jump.items()}}
        for m in modes:
            out["time_vs_motion"][m] = {f"{t}/{c}": {"p50": pct(v, 50), "p95": pct(v, 95), "max": float(max(v)), "n": len(v)}
                                        for (t, c), v in sorted(acc[m].items())}
        res[clip] = out
        print(clip, json.dumps(out["new_id_per_stroke_frame"]), json.dumps(out["id_jump_over_amp"]), flush=True)
        for m, rows in out["time_vs_motion"].items():
            print("  ", m, {k: (round(v["p50"], 3), round(v["p95"], 3), round(v["max"], 3)) for k, v in rows.items()}, flush=True)
    write("s3_motion_time_idjump.json", res)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--s3", type=Path, required=True)
    p.add_argument("what", choices=["regress", "cost", "cost_amp0", "determinism", "safety", "stats"])
    a = p.parse_args(argv)
    if a.what == "cost_amp0":                  # pembanding A/B pada kondisi mesin yang sama: jitter mati (= T-401)
        cmd_cost(a.s3, "amp0.yaml", "s3_cost_amp0")
        return 0
    {"regress": cmd_regress, "cost": cmd_cost, "determinism": cmd_determinism, "safety": cmd_safety, "stats": cmd_stats}[a.what](a.s3)
    return 0


if __name__ == "__main__":
    sys.exit(main())
