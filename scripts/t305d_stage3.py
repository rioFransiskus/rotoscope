"""T-305d Tahap 3: angka aturan QC baru di SALINAN scratch (klip asli dan out/ hanya DIBACA). Keluaran: work/t305d_scratch/t3/
(+ bahan penilaian work/t305d/nilai/). Salinan = berkas disalin penuh (bukan hard link: stage menambah baris ke *.jsonl).

    python scripts/t305d_stage3.py prepare    # salinan: 4 klip meta.json + seg/; klip3 juga frames/ depth/ stable/ contours/ strokes/
    python scripts/t305d_stage3.py qc         # segment --qc-only per klip: legacy, report, transient (terakhir = default)
    python scripts/t305d_stage3.py run        # klip3: stabilize -> vectorize -> stylize -> export (config default, out/ scratch)
    python scripts/t305d_stage3.py metrics    # stable/ (frame berubah, IoU fg / grup), [4] (oklusi, kelahiran, id baru), [5], [6]
    python scripts/t305d_stage3.py video      # work/t305d/nilai/klip3_sebelum_sesudah.mp4 (kiri = sekarang, kanan = baru)

Tahap 4 (klip ASLI; butuh salinan scratch Tahap 3 untuk `verify`):
    python scripts/t305d_stage3.py backup     # qc_report.json lama 4 klip -> work/t305d/qc_report_sebelum/ (dicek terhadap hash F1)
    python scripts/t305d_stage3.py real       # python -m rotoscope run samples/<klip>.mp4 (4 klip) -> work/t305d/run_t4/*.log
    python scripts/t305d_stage3.py verify     # klip asli vs hash F1 + qc_report lama; klip3 stable/ contours/ strokes/ out/ vs scratch
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import temporal_metrics as tm  # noqa: E402
import track_metrics as trm  # noqa: E402
from rotoscope import stabilize as stb  # noqa: E402

CLIPS_DIR = ROOT / "work" / "clips"
T3 = ROOT / "work" / "t305d_scratch" / "t3"
NILAI = ROOT / "work" / "t305d" / "nilai"
CLIPS = ("test", "test_short", "Klip2", "klip3")
FULL = "klip3"
FULL_SUBDIRS = ("frames", "seg", "depth", "stable", "contours", "strokes")
RULES = ("legacy", "report", "transient")          # transient terakhir: qc_report salinan = default
CFG = {r: T3 / f"cfg_{r}.yaml" for r in RULES}
STAGES = ("stabilize", "vectorize", "stylize", "export")
OCC = "occlusion"
VIDEO_SCALE = 0.5
LABEL_H = 110
FPS = 24


def sdir(clip: str) -> Path:
    return T3 / "clips" / clip


def jload(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def jsave(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def runs_str(idx: list[int]) -> str:
    out, s = [], None
    for k, i in enumerate(idx):
        if s is None:
            s = i
        if k == len(idx) - 1 or idx[k + 1] != i + 1:
            out.append(f"{s}" if s == i else f"{s}–{i}")
            s = None
    return ", ".join(out) or "-"


# ── prepare ────────────────────────────────────────
def cmd_prepare() -> None:
    for c in CLIPS:
        src, dst = CLIPS_DIR / c, sdir(c)
        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir(parents=True)
        shutil.copy2(src / "meta.json", dst / "meta.json")
        shutil.copy2(src / "qc_report.json", dst / "qc_report.json")
        for d in (FULL_SUBDIRS if c == FULL else ("seg",)):
            shutil.copytree(src / d, dst / d, copy_function=shutil.copy2)
        n_src = sum(1 for d in (FULL_SUBDIRS if c == FULL else ("seg",)) for p in (src / d).rglob("*") if p.is_file())
        n_dst = sum(1 for p in dst.rglob("*") if p.is_file()) - 2
        print(f"{c}: {n_dst} berkas disalin (sumber {n_src}) -> {dst.relative_to(ROOT).as_posix()}", flush=True)


# ── QC ─────────────────────────────────────────────
def stage(name: str, clip: str, cfg: Path, *extra: str) -> tuple[int, float, str]:
    t0 = time.perf_counter()
    r = subprocess.run([sys.executable, "-m", f"rotoscope.{name}", "--work-dir", str(sdir(clip)), "--config", str(cfg), *extra],
                       cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return r.returncode, time.perf_counter() - t0, r.stdout + r.stderr


def cmd_qc() -> None:
    res: dict = {}
    for c in CLIPS:
        orig = jload(CLIPS_DIR / c / "qc_report.json")
        res[c] = {}
        for rule in RULES:
            rc, secs, out = stage("segment", c, CFG[rule], "--qc-only")
            if rc != 0:
                raise SystemExit(f"{c} {rule}: exit {rc}\n{out}")
            rep = jload(sdir(c) / "qc_report.json")
            (T3 / "qc").mkdir(parents=True, exist_ok=True)
            shutil.copy2(sdir(c) / "qc_report.json", T3 / "qc" / f"{c}.{rule}.json")
            s = rep["summary"]
            ent = {"n_fail": s["n_fail"], "fail_frames": s["fail_frames"], "fail_counts": s["fail_counts"],
                   "frames_identical_to_current": json.dumps(rep["frames"]) == json.dumps(orig["frames"]),
                   "summary_identical_to_current": json.dumps(s) == json.dumps(orig["summary"]),
                   "rows_identical_except_fail_reasons": all(
                       {k: v for k, v in a.items() if k != "fail_reasons"} == {k: v for k, v in b.items() if k != "fail_reasons"}
                       for a, b in zip(rep["frames"], orig["frames"])) and len(rep["frames"]) == len(orig["frames"]),
                   "qc_version": rep.get("qc_version"), "rule": rep["qc"]["big_blobs_rule"], "qc_hash": rep["qc_hash"][:12],
                   "wall_s": round(secs, 2), "log": [ln for ln in out.splitlines() if ln.startswith("QC")]}
            res[c][rule] = ent
            print(f"{c:10s} {rule:9s} gagal {ent['n_fail']:3d}/{s['n_frames']} frames[]=sekarang {ent['frames_identical_to_current']!s:5s} "
                  f"summary=sekarang {ent['summary_identical_to_current']!s:5s} baris-selain-fail_reasons {ent['rows_identical_except_fail_reasons']} "
                  f"v{ent['qc_version']} {secs:.1f} s  gagal: {runs_str(s['fail_frames'])}", flush=True)
        print(f"  {c} qc_report sekarang: {orig['summary']['n_fail']} gagal, qc_hash {orig['qc_hash'][:12]}, "
              f"qc_version {orig.get('qc_version')}", flush=True)
    jsave(T3 / "qc_results.json", res)


# ── run [3]+ klip3 ─────────────────────────────────
def cmd_run() -> None:
    timing: dict = {}
    (T3 / "logs").mkdir(parents=True, exist_ok=True)
    for st in STAGES:
        rc, secs, out = stage(st, FULL, CFG["transient"])
        (T3 / "logs" / f"{st}.log").write_text(out, encoding="utf-8")
        timing[st] ={"rc": rc, "wall_s": round(secs, 2), "head": [ln[:220] for ln in out.splitlines()[:6]],
                      "tail": [ln[:220] for ln in out.splitlines()[-3:]]}
        print(f"--- {st}: exit {rc}, {secs:.1f} s", flush=True)
        for ln in (timing[st]["head"] + ["…"] + timing[st]["tail"]):
            print("   ", ln, flush=True)
        if rc != 0:
            raise SystemExit(f"{st} gagal (exit {rc}); log: {T3 / 'logs' / f'{st}.log'}")
    jsave(T3 / "timing.json", timing)


# ── metrik ─────────────────────────────────────────
def iou(a: np.ndarray, b: np.ndarray) -> float:
    u = np.logical_or(a, b).sum()
    return 1.0 if u == 0 else float(np.logical_and(a, b).sum() / u)


def group_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Rata-rata IoU per grup (1..G) yang muncul di salah satu peta."""
    gs = [g for g in np.union1d(np.unique(a), np.unique(b)) if g != 0]
    return float(np.mean([iou(a == g, b == g) for g in gs])) if gs else 1.0


def stats(v: list[float], names: list[str]) -> dict:
    a = np.asarray(v)
    k = int(np.argmin(a))
    return {"mean": round(float(a.mean()), 6), "min": round(float(a.min()), 6), "worst_frame": names[k]}


def geometry(strokes: list[dict], only: str | None = None) -> list:
    """Geometri strok tanpa track_id, bentuk kanonik track_metrics (titik awal / arah dinormalkan, urutan strok diurutkan):
    penomoran ulang id dan rotasi anchor (T-202, berantai dari frame sebelumnya) tidak dihitung beda."""
    return sorted(json.dumps(trm.canonical_stroke(s)) for s in strokes if only is None or s["type"] == only)


def births(docs: list[dict]) -> float:
    ids = [{s["track_id"] for s in d["strokes"] if s["type"] == OCC} for d in docs]
    return float(np.mean([len(ids[t] - ids[t - 1]) for t in range(1, len(ids))]))


def contour_metrics(d: Path) -> dict:
    docs = [jload(p) for p in sorted(d.glob("frame_*.json"))]
    pop = tm.pop_energy_by_type(tm.stroke_lengths_from_dir(d), ("silhouette", "silhouette_hole", "group_boundary", OCC))
    occ = np.array([sum(s["type"] == OCC for s in x["strokes"]) for x in docs])
    allst = np.array([len(x["strokes"]) for x in docs])
    return {"occ_per_frame": round(float(occ.mean()), 4), "occ_total": int(occ.sum()),
            "births_per_frame": round(births(docs), 4),
            "id_new_per_stroke_frame_occ": round(pop["types"][OCC]["id_new_per_stroke_frame"], 4),
            "age_mean_occ": round(pop["types"][OCC]["age_mean"], 4),
            "strokes_per_frame_all": round(float(allst.mean()), 4),
            "id_new_per_stroke_frame_all": round(float(np.mean([pop["types"][t]["id_new_per_stroke_frame"] for t in pop["types"]])), 4)}


def cmd_metrics() -> None:
    oc, nc = stb.load_clip(CLIPS_DIR / FULL), stb.load_clip(sdir(FULL))
    names = list(oc.names)
    g_changed, d_changed, ious, gious, frac = [], [], [], [], []
    for i, n in enumerate(names):
        ga, gb = stb.read_groups(oc.groups_path(n)), stb.read_groups(nc.groups_path(n))
        if sha(oc.groups_path(n)) != sha(nc.groups_path(n)):
            g_changed.append(i)
        if sha(oc.depth_smooth_path(n)) != sha(nc.depth_smooth_path(n)):
            d_changed.append(i)
        ious.append(iou(ga != 0, gb != 0))
        gious.append(group_iou(ga, gb))
        frac.append(float((ga != gb).mean()))
    changed = sorted(set(g_changed) | set(d_changed))
    qc_old = jload(CLIPS_DIR / FULL / "qc_report.json")["summary"]["fail_frames"]
    qc_new = jload(sdir(FULL) / "qc_report.json")["summary"]["fail_frames"]
    q_changed = sorted(set(qc_old) ^ set(qc_new))
    m_old = jload(CLIPS_DIR / FULL / "stable" / "manifest.json")["temporal"]
    m_new = jload(sdir(FULL) / "stable" / "manifest.json")["temporal"]
    res = {"stable": {"frames": len(names), "groups_changed": len(g_changed), "depth_smooth_changed": len(d_changed),
                      "frames_changed": len(changed), "frames_changed_runs": runs_str(changed),
                      "q_changed_frames": len(q_changed), "q_changed_runs": runs_str(q_changed),
                      "manifest_qc_fail_frames": [len(m_old["qc_fail_frames"]), len(m_new["qc_fail_frames"])],
                      "manifest_cut_frames_equal": m_old["cut_frames"] == m_new["cut_frames"],
                      "iou_fg": stats(ious, names), "iou_group": stats(gious, names),
                      "group_px_changed_pct": {"mean": round(100 * float(np.mean(frac)), 4), "max": round(100 * float(np.max(frac)), 4)}}}
    occ_o, occ_n = contour_metrics(CLIPS_DIR / FULL / "contours"), contour_metrics(sdir(FULL) / "contours")
    co = sorted((CLIPS_DIR / FULL / "contours").glob("frame_*.json"))
    strokes_diff, geom_diff, occ_geom_diff = [], [], []
    for i, p in enumerate(co):
        a, b = jload(p)["strokes"], jload(sdir(FULL) / "contours" / p.name)["strokes"]
        if a != b:
            strokes_diff.append(i)
        if geometry(a) != geometry(b):
            geom_diff.append(i)
        if geometry(a, OCC) != geometry(b, OCC):
            occ_geom_diff.append(i)
    res["contours"] = {"sekarang": occ_o, "baru": occ_n, "frames_strokes_differ": len(strokes_diff),
                       "frames_strokes_differ_runs": runs_str(strokes_diff),
                       "frames_geometry_differ": len(geom_diff), "frames_geometry_differ_runs": runs_str(geom_diff),
                       "frames_occlusion_geometry_differ": len(occ_geom_diff),
                       "frames_occlusion_geometry_differ_runs": runs_str(occ_geom_diff),
                       "clip_stats": {k: [jload(CLIPS_DIR / FULL / "contours" / "clip_stats.json")[k],
                                          jload(sdir(FULL) / "contours" / "clip_stats.json")[k]] for k in ("t_high", "t_low")}}
    res["worst_frames"] = {}
    gnames = ["background"] + [g for g, _ in jload(CLIPS_DIR / FULL / "stable" / "manifest.json")["groups"]]
    for key in ("iou_fg", "iou_group"):
        n = res["stable"][key]["worst_frame"]
        ga, gb = stb.read_groups(oc.groups_path(n)), stb.read_groups(nc.groups_path(n))
        res["worst_frames"][f"{key}:{n}"] = {
            gnames[g]: {"px_sekarang": int((ga == g).sum()), "px_baru": int((gb == g).sum()), "iou": round(iou(ga == g, gb == g), 4)}
            for g in np.union1d(np.unique(ga), np.unique(gb)) if g != 0}
    png = [i for i, p in enumerate(sorted((CLIPS_DIR / FULL / "strokes").glob("frame_*.png")))
           if sha(p) != sha(sdir(FULL) / "strokes" / p.name)]
    svg = [i for i, p in enumerate(sorted((CLIPS_DIR / FULL / "strokes").glob("frame_*.svg")))
           if sha(p) != sha(sdir(FULL) / "strokes" / p.name)]
    mp4_o, mp4_n = ROOT / "out" / f"{FULL}.mp4", T3 / "out" / f"{FULL}.mp4"
    res["strokes"] = {"png_differ": len(png), "svg_differ": len(svg), "png_differ_runs": runs_str(png)}
    res["export"] = {"mp4_new_exists": mp4_n.is_file(), "mp4_identical": mp4_n.is_file() and sha(mp4_o) == sha(mp4_n)}
    jsave(T3 / "metrics.json", res)
    print(json.dumps(res, indent=1, ensure_ascii=False))


# ── video berdampingan ─────────────────────────────
def label_bar(width: int, text: str) -> np.ndarray:
    bar = np.full((LABEL_H, width, 3), 255, np.uint8)
    cv2.putText(bar, text, (14, LABEL_H - 28), cv2.FONT_HERSHEY_DUPLEX, 1.8, (0, 0, 0), 4, cv2.LINE_AA)
    return bar


def read_frames(path: Path):
    cap = cv2.VideoCapture(str(path))
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        yield fr
    cap.release()


def cmd_video() -> None:
    NILAI.mkdir(parents=True, exist_ok=True)
    left, right = ROOT / "out" / f"{FULL}.mp4", T3 / "out" / f"{FULL}.mp4"
    tmp = T3 / "_tmp_berdampingan.mp4"
    writer, n = None, 0
    for a, b in zip(read_frames(left), read_frames(right)):
        panels = []
        for fr, title in ((a, "SEKARANG"), (b, "BARU")):
            fr = cv2.resize(fr, None, fx=VIDEO_SCALE, fy=VIDEO_SCALE, interpolation=cv2.INTER_AREA)
            panels.append(np.vstack([label_bar(fr.shape[1], title), fr]))
        img = np.hstack(panels)
        if writer is None:
            writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (img.shape[1], img.shape[0]))
        writer.write(img)
        n += 1
    writer.release()
    final = NILAI / f"{FULL}_sebelum_sesudah.mp4"
    r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp), "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p",
                        "-r", str(FPS), str(final)], capture_output=True, text=True)
    if r.returncode == 0:
        tmp.unlink()
    else:
        shutil.move(tmp, final)
        print("PERINGATAN: ffmpeg gagal, berkas berdampingan = mp4v:", r.stderr[-200:])
    print(f"{final.relative_to(ROOT).as_posix()}: {n} frame")


# ── Tahap 4: klip asli ─────────────────────────────
T305D = ROOT / "work" / "t305d"
QC_BEFORE = T305D / "qc_report_sebelum"
RUN_LOGS = T305D / "run_t4"
F1_BASELINE = T305D / "before_t23_sha256.txt"
REAL_ORDER = ("test_short", "test", "Klip2", "klip3")
GPU_INPUTS = ("frames", "seg", "depth")
DOWNSTREAM = ("stable", "contours", "strokes")


def f1_baseline() -> dict[str, str]:
    out = {}
    for line in F1_BASELINE.read_text(encoding="utf-8").splitlines():
        h, p = line.split("  ", 1)
        out[p] = h
    return out


def cmd_backup() -> None:
    base = f1_baseline()
    QC_BEFORE.mkdir(parents=True, exist_ok=True)
    for c in CLIPS:
        src = CLIPS_DIR / c / "qc_report.json"
        rel = src.relative_to(ROOT).as_posix()
        if sha(src) != base[rel]:
            raise SystemExit(f"{rel} sudah berbeda dari hash F1 — backup dibatalkan")
        shutil.copy2(src, QC_BEFORE / f"{c}.json")
        print(f"{rel} -> {(QC_BEFORE / f'{c}.json').relative_to(ROOT).as_posix()} (sha256 = F1)", flush=True)


def cmd_real() -> None:
    RUN_LOGS.mkdir(parents=True, exist_ok=True)
    res = {}
    for c in REAL_ORDER:
        t0 = time.perf_counter()
        r = subprocess.run([sys.executable, "-m", "rotoscope", "run", f"samples/{c}.mp4"], cwd=ROOT, capture_output=True,
                           text=True, encoding="utf-8", errors="replace")
        secs = time.perf_counter() - t0
        out = r.stdout + r.stderr
        (RUN_LOGS / f"{c}.log").write_text(out, encoding="utf-8")
        times = [ln for ln in out.splitlines() if ln.startswith("Waktu per stage")]
        res[c] = {"rc": r.returncode, "wall_s": round(secs, 1), "waktu_per_stage": times[-1] if times else None}
        print(f"{c}: exit {r.returncode}, {secs:.1f} s — {res[c]['waktu_per_stage']}", flush=True)
        if r.returncode != 0:
            raise SystemExit(f"run {c} gagal; log {RUN_LOGS / f'{c}.log'}")
    jsave(RUN_LOGS / "timing.json", res)


def stage_lines(log: str) -> dict[str, list[str]]:
    """Baris ringkas per stage dari log `run` (bagian '=== ... ===')."""
    out, cur = {}, None
    for ln in log.splitlines():
        if ln.startswith("=== ") and ln.endswith(" ==="):
            cur = ln.strip("= ").strip()
            out[cur] = []
        elif cur and any(k in ln for k in ("diproses", "basi", "up-to-date", "selesai", "PERINGATAN", "QC:", "dilewati")):
            out[cur].append(ln.strip()[:200])
    return out


def cmd_verify() -> None:
    base = f1_baseline()
    res: dict = {}
    for c in CLIPS:
        prefix = f"work/clips/{c}/"
        files = {p: h for p, h in base.items() if p.startswith(prefix)}
        cur = {p.relative_to(ROOT).as_posix(): p for d in (*GPU_INPUTS, *DOWNSTREAM) for p in (CLIPS_DIR / c / d).rglob("*")
               if p.is_file()}
        cur[f"{prefix}qc_report.json"] = CLIPS_DIR / c / "qc_report.json"
        missing = sorted(set(files) - set(cur))
        added = sorted(set(cur) - set(files))
        changed = sorted(p for p in set(files) & set(cur) if sha(cur[p]) != files[p])
        by_dir: dict[str, int] = {}
        for p in changed + added + missing:
            k = p[len(prefix):].split("/")[0]
            by_dir[k] = by_dir.get(k, 0) + 1
        old, new = jload(QC_BEFORE / f"{c}.json"), jload(CLIPS_DIR / c / "qc_report.json")
        ent = {"files_f1": len(files), "changed": len(changed), "added": len(added), "missing": len(missing),
               "changed_by_dir": by_dir, "changed_examples": (changed + added + missing)[:8],
               "qc_frames_identical": json.dumps(new["frames"]) == json.dumps(old["frames"]),
               "qc_summary_identical": json.dumps(new["summary"]) == json.dumps(old["summary"]),
               "qc_keys_changed": sorted(k for k in set(old) | set(new) if old.get(k) != new.get(k)),
               "qc_n_fail": [old["summary"]["n_fail"], new["summary"]["n_fail"]], "qc_fail_frames_new": new["summary"]["fail_frames"],
               "run_log": stage_lines((RUN_LOGS / f"{c}.log").read_text(encoding="utf-8"))}
        if c == FULL:
            ent["gpu_inputs_identical_f1"] = not any(p[len(prefix):].split("/")[0] in GPU_INPUTS for p in changed + added + missing)
            vs: dict = {}
            for d in DOWNSTREAM:
                a = {p.relative_to(CLIPS_DIR / c / d).as_posix(): p for p in (CLIPS_DIR / c / d).rglob("*") if p.is_file()}
                b = {p.relative_to(sdir(c) / d).as_posix(): p for p in (sdir(c) / d).rglob("*") if p.is_file()}
                diff = sorted(k for k in set(a) & set(b) if sha(a[k]) != sha(b[k]))
                vs[d] = {"files": [len(a), len(b)], "only_real": sorted(set(a) - set(b))[:5], "only_scratch": sorted(set(b) - set(a))[:5],
                         "differ": len(diff), "differ_names": diff[:10]}
            ra = {p.relative_to(ROOT / "out").as_posix(): p for p in (ROOT / "out").rglob("*") if p.is_file()
                  and (p.name.startswith(f"{FULL}.") or f"svg/{FULL}/" in p.as_posix())}
            rb = {p.relative_to(T3 / "out").as_posix(): p for p in (T3 / "out").rglob("*") if p.is_file()}
            diff = sorted(k for k in set(ra) & set(rb) if sha(ra[k]) != sha(rb[k]))
            vs["out"] = {"files": [len(ra), len(rb)], "only_real": sorted(set(ra) - set(rb))[:5], "only_scratch": sorted(set(rb) - set(ra))[:5],
                         "differ": len(diff), "differ_names": diff[:10],
                         "mp4_identical": sha(ROOT / "out" / f"{FULL}.mp4") == sha(T3 / "out" / f"{FULL}.mp4")}
            ent["vs_scratch_t3"] = vs
        res[c] = ent
        print(c, json.dumps(ent, indent=1, ensure_ascii=False), flush=True)
    others = {p: h for p, h in base.items() if p.startswith("out/")}
    out_changed = sorted(p for p, h in others.items() if not (ROOT / p).is_file() or sha(ROOT / p) != h)
    out_added = sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "out").rglob("*") if p.is_file()
                       and p.relative_to(ROOT).as_posix() not in others)
    res["out"] = {"files_f1": len(others), "changed": out_changed, "added": out_added}
    print("out/", json.dumps(res["out"], indent=1, ensure_ascii=False))
    jsave(T305D / "verify_t4.json", res)


def main(argv: list[str]) -> int:
    cmds = {"prepare": cmd_prepare, "qc": cmd_qc, "run": cmd_run, "metrics": cmd_metrics, "video": cmd_video,
            "backup": cmd_backup, "real": cmd_real, "verify": cmd_verify}
    if len(argv) == 2 and argv[1] in cmds:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        cmds[argv[1]]()
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
