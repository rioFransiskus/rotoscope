"""Pengukuran Tahap 3 T-403 (alat sekali pakai; semua pada SALINAN klip di folder `--s3`, klip asli tidak diubah):

    python scripts/t403_measure.py --s3 <dir> regress       # default style = T-402 byte-identik (hash strokes/ klip asli sebelum T-403) + 1 pass opacity 1,0 varian
    python scripts/t403_measure.py --s3 <dir> cost          # 3 run bersih per klip untuk passes 1 / 2 / 3: waktu per frame + rincian, ukuran PNG / SVG
    python scripts/t403_measure.py --s3 <dir> determinism   # dua run dari nol, --limit, --from, resume, basi
    python scripts/t403_measure.py --s3 <dir> safety        # invarian per pass di SELURUH frame kedua klip (offset 2,7 / 5,5 / 8 / 12) + pemisahan
    python scripts/t403_measure.py --s3 <dir> alpha         # IoU SVG-PNG alpha + massa / tebal tampak vs T-402 (frame sampel)
    python scripts/t403_measure.py --s3 <dir> mp4           # ukuran MP4 (export nyata) T-402 vs varian

<dir> berisi: work/clips/<klip>/{contours/, meta.json} (salinan), YAML style (dibuat otomatis: mp2.yaml, mp3.yaml, eval.yaml). Hasil JSON ke
work/t403/. Perintah stage dipanggil lewat subprocess (CLI nyata)."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "src"))

import jitter_metrics as jm  # noqa: E402
import multipass_metrics as mm  # noqa: E402
import stylize_metrics as sm  # noqa: E402

from rotoscope import stylize as sty  # noqa: E402
from rotoscope.config import load_style  # noqa: E402

OUT = ROOT / "work" / "t403"
CLIPS = ("test_short", "test")
PY = sys.executable
STYLES = {                                        # nilai evaluasi (HANYA papan / ukur; bukan default)
    "mp2.yaml": "stroke:\n  opacity: 0.92\nmultipass:\n  passes: 2\n  offset: 8.0\n  opacity_falloff: 0.55\n",
    "mp3.yaml": "stroke:\n  opacity: 0.92\nmultipass:\n  passes: 3\n  offset: 8.0\n  opacity_falloff: 0.55\n",
    "mp2_hold.yaml": "stroke:\n  opacity: 0.92\nmultipass:\n  passes: 2\n  offset: 8.0\n  opacity_falloff: 0.55\n  temporal_mode: frame\n",
    "mp3_off2p7.yaml": "stroke:\n  opacity: 0.92\nmultipass:\n  passes: 3\n  offset: 2.7\n  opacity_falloff: 0.55\n",
    "default.yaml": "",
}


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


def prepare(s3: Path) -> None:
    for name, text in STYLES.items():
        (s3 / name).write_text(text, encoding="utf-8")


def stylize(s3: Path, clip: str, style: str, *extra: str) -> subprocess.CompletedProcess:
    cmd = [PY, "-m", "rotoscope.stylize", "--work-dir", str(s3 / "work" / "clips" / clip), "--style", str(s3 / style), *extra]
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")


def rm_strokes(s3: Path, clip: str) -> None:
    shutil.rmtree(s3 / "work" / "clips" / clip / "strokes", ignore_errors=True)


def write(name: str, obj) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"  work/t403/{name}")


def frame_records(strokes_dir: Path) -> list[dict]:
    return [r for r in (json.loads(x) for x in (strokes_dir / "frames.jsonl").read_text(encoding="utf-8").splitlines())
            if r.get("event") == "frame"]


# ── regress ────────────────────────────────────────
def cmd_regress(s3: Path) -> None:
    res = {}
    weird = "multipass:\n  passes: 1\n  offset: 9.0\n  opacity_falloff: 0.1\n  temporal_mode: frame\nstroke:\n  opacity: 1.0\n"
    off = "multipass:\n  enabled: false\n  passes: 5\n  offset: 20.0\nstroke:\n  opacity: 1.0\n"
    (s3 / "weird.yaml").write_text(weird, encoding="utf-8")
    (s3 / "disabled.yaml").write_text(off, encoding="utf-8")
    for clip in CLIPS:
        ref = before_hashes(clip)
        for style in ("default.yaml", "weird.yaml", "disabled.yaml"):
            rm_strokes(s3, clip)
            r = stylize(s3, clip, style)
            got = strokes_hashes(s3 / "work" / "clips" / clip / "strokes")
            diff = sorted(k for k in set(ref) | set(got) if ref.get(k) != got.get(k))
            res[f"{clip}/{style}"] = {"rc": r.returncode, "files": len(got), "reference_files": len(ref), "identical": not diff,
                                      "n_diff": len(diff), "first_diff": diff[:5]}
            print(clip, style, res[f"{clip}/{style}"], flush=True)
    write("s3_regress.json", res)


# ── cost ───────────────────────────────────────────
PEAK_WRAPPER = ("import sys, ctypes; from ctypes import wintypes; from rotoscope import stylize\n"
                "rc = stylize.main(sys.argv[1:])\n"
                "class P(ctypes.Structure):\n"
                "    _fields_ = [('cb', wintypes.DWORD), ('pf', wintypes.DWORD), ('peak', ctypes.c_size_t)] + [(f'x{i}', ctypes.c_size_t) for i in range(7)]\n"
                "p = P(); p.cb = ctypes.sizeof(P)\n"
                "h = ctypes.WinDLL('kernel32').GetCurrentProcess()\n"
                "ctypes.WinDLL('psapi').GetProcessMemoryInfo(ctypes.c_void_p(h), ctypes.byref(p), p.cb)\n"
                "print('PEAK_MIB', p.peak / 2**20); sys.exit(rc)\n")


def cmd_cost(s3: Path) -> None:
    res = {}
    for label, style in (("passes1_default", "default.yaml"), ("passes2", "mp2.yaml"), ("passes3", "mp3.yaml")):
        for clip in CLIPS:
            runs = []
            for k in range(3):
                rm_strokes(s3, clip)
                args = ["--work-dir", str(s3 / "work" / "clips" / clip), "--style", str(s3 / style)]
                t0 = time.perf_counter()
                pr = subprocess.run([PY, "-c", PEAK_WRAPPER, *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
                rc, wall = pr.returncode, time.perf_counter() - t0
                peak = next((float(x.split()[1]) for x in pr.stdout.splitlines() if x.startswith("PEAK_MIB")), float("nan"))
                recs = frame_records(s3 / "work" / "clips" / clip / "strokes")
                t = np.array([r["total_s"] for r in recs]) * 1000
                row = {"run": k + 1, "rc": rc, "wall_s": round(wall, 1), "peak_mib": round(peak, 1), "n": len(recs),
                       "total_mean_ms": float(t.mean()), "total_p95_ms": float(np.percentile(t, 95)), "total_max_ms": float(t.max()),
                       "n_gt_400": int((t > 400).sum())}
                for key in ("geom_s", "raster_png_s", "svg_s", "write_s", "jitter_s", "multipass_s"):
                    row[key.replace("_s", "_mean_ms")] = float(np.mean([r.get(key, 0.0) for r in recs]) * 1000)
                row["png_mean_kib"] = float(np.mean([r["png_bytes"] for r in recs]) / 1024)
                row["svg_mean_kib"] = float(np.mean([r["svg_bytes"] for r in recs]) / 1024)
                row["png_total_mib"] = float(sum(r["png_bytes"] for r in recs) / 2**20)
                row["svg_total_mib"] = float(sum(r["svg_bytes"] for r in recs) / 2**20)
                runs.append(row)
                print(label, clip, row, flush=True)
            res[f"{label}/{clip}"] = runs
    write("s3_cost.json", res)


# ── determinism ────────────────────────────────────
def cmd_determinism(s3: Path) -> None:
    res = {}
    for clip in CLIPS:
        d = s3 / "work" / "clips" / clip / "strokes"
        rm_strokes(s3, clip)
        stylize(s3, clip, "mp3.yaml")
        full = strokes_hashes(d)
        out = {"files": len(full)}
        rm_strokes(s3, clip)
        stylize(s3, clip, "mp3.yaml")
        out["two_runs_identical"] = strokes_hashes(d) == full
        rm_strokes(s3, clip)
        stylize(s3, clip, "mp3.yaml", "--limit", "20")
        part = strokes_hashes(d)
        out["limit20_is_prefix"] = len(part) == 40 and all(full[k] == v for k, v in part.items())
        stylize(s3, clip, "mp3.yaml")
        out["limit_then_full_identical"] = strokes_hashes(d) == full
        for p in list(d.glob("frame_0006[0-4].*")):
            p.unlink()
        r = stylize(s3, clip, "mp3.yaml")
        out["resume_identical"] = strokes_hashes(d) == full and "5 diproses" in r.stdout
        rm_strokes(s3, clip)
        stylize(s3, clip, "mp3.yaml", "--from", "100", "--limit", "10")
        win = strokes_hashes(d)
        out["from100_limit10_equals_full"] = len(win) == 20 and all(full[k] == v for k, v in win.items())
        r = stylize(s3, clip, "mp2.yaml")
        out["stale_passes2_recomputed"] = "basi" in r.stdout and strokes_hashes(d) != full
        stylize(s3, clip, "mp3.yaml")
        out["back_to_mp3_identical"] = strokes_hashes(d) == full
        # mode frame: jendela == run penuh
        rm_strokes(s3, clip)
        stylize(s3, clip, "mp2_hold.yaml")
        fullh = strokes_hashes(d)
        rm_strokes(s3, clip)
        stylize(s3, clip, "mp2_hold.yaml", "--from", "101", "--limit", "7")
        winh = strokes_hashes(d)
        out["frame_mode_from101_limit7_equals_full"] = len(winh) == 14 and all(fullh[k] == v for k, v in winh.items())
        res[clip] = out
        print(clip, out, flush=True)
    write("s3_determinism.json", res)


# ── safety ─────────────────────────────────────────
def load_doc(s3: Path, clip: str, i: int) -> dict:
    return json.loads((s3 / "work" / "clips" / clip / "contours" / f"frame_{i:05d}.json").read_text(encoding="utf-8"))


def pct(a, q):
    a = np.asarray(a, float)
    return float(np.percentile(a, q)) if len(a) else float("nan")


def cmd_safety(s3: Path) -> None:
    """Invarian per pass (pass 1 dan 2) di SELURUH frame kedua klip; offset {2,7; 5,5; 8; 12}; tanpa jitter (default)."""
    res = {}
    for clip in CLIPS:
        meta = json.loads((s3 / "work" / "clips" / clip / "meta.json").read_text(encoding="utf-8"))
        n = int(meta["frame_count"])
        for off in (2.7, 5.5, 8.0, 12.0):
            style = load_style(None, overrides={"multipass.passes": 3, "multipass.offset": off})
            g = sty.make_geometry(style, int(meta["working_width"]), int(meta["working_height"]))
            acc = {"joint_ratio": [], "joint_n": 0, "joint_over_tol": 0, "new_cross": 0, "frames_new_cross": 0, "min_det": [],
                   "edge_ink": 0, "edge_ink_frames": 0, "intrusion_frames": 0, "seam_ratio": [], "seam_rel_fail": 0, "seam_fail": 0, "max_disp_ratio": [], "frames": 0}
            sep = []
            for i in range(n):
                doc = load_doc(s3, clip, i)
                passes, _ = sty.frame_passes(doc, g)
                for r in mm.pass_invariants(passes, g, int(doc["frame_index"])):
                    acc["joint_n"] += r["joint_n"]
                    acc["joint_ratio"].append(r["joint_ratio"])
                    acc["joint_over_tol"] += int(r["joint_ratio"] > 1.0)
                    acc["new_cross"] += r["new_crossings"]
                    acc["frames_new_cross"] += int(r["new_crossings"] > 0)
                    acc["min_det"].append(r["min_det"])
                    acc["edge_ink"] += r["edge_ink_changed"]
                    acc["edge_ink_frames"] += int(r["edge_ink_changed"] > 0)
                    acc["intrusion_frames"] += int(r["intrusion_excess"] > 1e-9)
                    acc["seam_ratio"].append(r["seam_ratio_max"])
                    acc["seam_rel_fail"] += r["seam_rel_fail"]
                    acc["seam_fail"] += r["seam_fail"]
                    acc["max_disp_ratio"].append(r["max_displacement_over_bound"])
                if i % 4 == 0 and passes[0]:
                    sep.append(mm.separation_stats(passes[0], g, int(doc["frame_index"])))
                acc["frames"] += 1
            row = {"offset": off, "A_ref": off / sty.MULTIPASS_SEP_MEDIAN, "cell_ref": off / sty.MULTIPASS_SEP_MEDIAN / sty.MULTIPASS_FOLD_R,
                   "joints": acc["joint_n"], "joint_ratio_p95": pct(acc["joint_ratio"], 95), "joint_ratio_max": float(max(acc["joint_ratio"] or [0])),
                   "joint_over_tol": acc["joint_over_tol"], "new_crossings": acc["new_cross"], "frames_with_new_crossings": acc["frames_new_cross"],
                   "min_det_min": float(min(acc["min_det"])), "frames_min_det_le_0.05": int(sum(d <= sty.JACOBIAN_MIN_DET for d in acc["min_det"])),
                   "edge_ink_px_total": acc["edge_ink"], "edge_ink_pass_frames_nonzero": acc["edge_ink_frames"],
                   "intrusion_pass_frames": acc["intrusion_frames"], "seam_ratio_p95": pct(acc["seam_ratio"], 95),
                   "seam_ratio_max": float(max(acc["seam_ratio"])), "seam_rel_fail_strokes": acc["seam_rel_fail"],
                   "seam_fail_strokes": acc["seam_fail"],"max_disp_over_bound_max": float(max(acc["max_disp_ratio"])),
                   "separation_p50_ref_mean": float(np.mean([s["p50"] for s in sep])), "separation_rms_ref_mean": float(np.mean([s["rms"] for s in sep])),
                   "separation_p95_ref_mean": float(np.mean([s["p95"] for s in sep])), "separation_max_ref": float(max(s["max"] for s in sep)),
                   "pass_frames": 2 * acc["frames"], "frames": acc["frames"]}
            res[f"{clip}/off{off:g}"] = row
            print(clip, off, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    write("s3_safety.json", res)


# ── alpha ──────────────────────────────────────────
def cmd_alpha(s3: Path) -> None:
    """IoU SVG-PNG alpha + massa / tebal tampak vs T-402 pada frame sampel (konfigurasi papan)."""
    res = []
    samples = {"test_short": [5, 20, 40, 60, 80, 100, 115], "test": [25, 30, 50, 80, 100, 130, 150, 183, 192, 200, 233, 260]}
    configs = {"2p_off2.7_f0.55_op1.0": {"multipass.passes": 2, "multipass.offset": 2.7},
               "3p_off8_f0.55_op0.92": {"multipass.passes": 3, "multipass.offset": 8.0, "stroke.opacity": 0.92},
               "2p_off12_f0.55_op0.85": {"multipass.passes": 2, "multipass.offset": 12.0, "stroke.opacity": 0.85},
               "1p_op0.92": {"stroke.opacity": 0.92}}
    for clip in CLIPS:
        meta = json.loads((s3 / "work" / "clips" / clip / "meta.json").read_text(encoding="utf-8"))
        w, h = int(meta["working_width"]), int(meta["working_height"])
        g0 = sty.make_geometry(load_style(None), w, h)
        for cname, ov in configs.items():
            st = load_style(None, overrides=ov)
            g = sty.make_geometry(st, w, h)
            for pos in samples[clip]:
                doc = load_doc(s3, clip, pos)
                svg, png, _ = sty.render_frame(doc, g, st)
                f_png = mm.png_ink_fraction(sm.decode_png(png), g)
                rep = mm.alpha_report(mm.svg_ink_fraction(svg, g), f_png)
                ref_svg, ref_png, _ = sty.render_frame(doc, g0, load_style(None))
                f_ref = mm.png_ink_fraction(sm.decode_png(ref_png), g0)
                L = sum(float(np.hypot(*np.diff(pc.points, axis=0).T).sum()) for pc in sty.frame_pieces(doc, g0)[0])
                res.append({"clip": clip, "config": cname, "frame": pos, **rep, "mass_vs_t402": float(f_png.sum() / f_ref.sum()),
                            "px_gt_0.5_vs_t402": float((f_png > 0.5).sum() / max((f_ref > 0.5).sum(), 1)),
                            "eff_width_px": float(f_png.sum() / L), "vis_width_px": float((f_png > 0.5).sum() / L)})
    agg = {}
    for r in res:
        agg.setdefault(r["config"], []).append(r)
    summ = {k: {m: {"min": float(min(x[m] for x in v)), "mean": float(np.mean([x[m] for x in v])), "max": float(max(x[m] for x in v))}
                for m in ("iou", "l1", "big_diff_frac", "mass_ratio", "mass_vs_t402", "px_gt_0.5_vs_t402", "eff_width_px", "vis_width_px")}
            for k, v in agg.items()}
    for k, v in summ.items():
        print(k, {m: {a: round(b, 4) for a, b in s.items()} for m, s in v.items()}, flush=True)
    write("s3_alpha.json", {"summary": summ, "rows": res})


# ── mp4 (export nyata pada salinan) ────────────────
def cmd_mp4(s3: Path) -> None:
    """Export nyata (stage [6]) pada salinan klip: perlu strokes/ + frames/ + meta; hanya dipakai bila salinan lengkap. Diganti ukuran MP4 prototipe
    scripts: encode PNG strokes/ langsung (crf 18, sama dengan export) untuk T-402 vs varian."""
    from rotoscope import export as ex
    import cv2
    res = {}
    for clip in CLIPS:
        meta = json.loads((s3 / "work" / "clips" / clip / "meta.json").read_text(encoding="utf-8"))
        fps = float(meta["target_fps"])
        sizes = {}
        for label, style in (("t402_default", "default.yaml"), ("mp2_off8_op0.92", "mp2.yaml"), ("mp3_off8_op0.92", "mp3.yaml"),
                             ("mp3_off2.7_op0.92", "mp3_off2p7.yaml"), ("mp2_hold_off8_op0.92", "mp2_hold.yaml")):
            rm_strokes(s3, clip)
            stylize(s3, clip, style)
            d = s3 / "work" / "clips" / clip / "strokes"
            pngs = sorted(d.glob("frame_*.png"))
            first = cv2.imread(str(pngs[0]))
            h, w = first.shape[:2]
            tmp = OUT / f"mp4_{clip}_{label}.mp4"
            OUT.mkdir(parents=True, exist_ok=True)
            cmd = ex.build_ffmpeg_cmd("ffmpeg", (w, h), fps, 18, "medium", None, tmp, color_tags=True)
            pr = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            for p in pngs:
                pr.stdin.write(np.ascontiguousarray(cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)).tobytes())
            pr.stdin.close()
            pr.wait()
            sizes[label] = tmp.stat().st_size
            print(clip, label, sizes[label], flush=True)
            tmp.unlink()
        base = sizes["t402_default"]
        res[clip] = {"bytes": sizes, "ratio": {k: v / base for k, v in sizes.items()}}
    write("s3_mp4.json", res)


COMMANDS = {"regress": cmd_regress, "cost": cmd_cost, "determinism": cmd_determinism, "safety": cmd_safety, "alpha": cmd_alpha, "mp4": cmd_mp4}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--s3", type=Path, required=True)
    p.add_argument("what", choices=COMMANDS)
    a = p.parse_args()
    prepare(a.s3)
    COMMANDS[a.what](a.s3)


if __name__ == "__main__":
    main()
