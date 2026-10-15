"""T-305 induk Tahap 1: uji klip baru dengan pipeline default [1]–[6] (docs/04 "Keputusan T-305 induk Tahap 1 (uji klip baru)").

    python scripts/t305_newclips.py hash before|after   # F1: sha256 4 klip lama + 4 qc_report.json + out/ (after: out/ baru hanya milik klip baru)
    python scripts/t305_newclips.py run KLIP            # python -m rotoscope run samples/KLIP.mp4 --yes + log + sampel nvidia-smi -> work/t305/run/
    python scripts/t305_newclips.py metrics             # metrik klip baru vs test / Klip2 / klip3 (data yang ada, tanpa run ulang) -> work/t305/metrics.json
    python scripts/t305_newclips.py media KLIP          # work/t305/nilai/KLIP_berdampingan.mp4, KLIP_lambat_0.5x.mp4, KLIP_hasil.mp4

Hanya MEMBACA work/clips/* dan out/ (run menulis klip baru lewat pipeline); keluaran skrip ke work/t305/ (di-ignore git).
Memakai ulang: hash + wajah V1 (`scripts/t701_study.py`), kelahiran oklusi (`scripts/t305d_stage3.py`), panjang strok per
(tipe, track_id) (`tests/temporal_metrics.py`), label + pembaca video (`scripts/t305d_stage3.py`).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import temporal_metrics as tm  # noqa: E402
import t701_study as t701  # noqa: E402
from t305d_stage3 import births as occ_births, label_bar, read_frames, runs_str  # noqa: E402

from rotoscope import stabilize as stb  # noqa: E402
from rotoscope.config import load_pipeline  # noqa: E402

CLIPS_DIR = ROOT / "work" / "clips"
OUT = ROOT / "work" / "t305"
RUN = OUT / "run"
NILAI = OUT / "nilai"
OLD = ("test", "Klip2", "klip3")                 # pembanding (test_short hanya di F1)
NEW = ("klip4", "klip5")
OCC = "occlusion"
STAGE_LOGS = (("[2] segment", "seg"), ("[2c] depth", "depth"), ("[3] stabilize", "stable"),
              ("[4] vectorize", "contours"), ("[5] stylize", "strokes"))
VRAM_SAMPLE_S = 2.0
FPS = 24
PANEL_H = 960                                     # tinggi panel foto / hasil (hasil 1080 × 1920 → 540 × 960)
TIME_H = 76


def jload(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def jsave(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"  -> {p.relative_to(ROOT).as_posix()}")


# ── F1 ─────────────────────────────────────────────
def hash_map() -> dict[str, str]:
    return {p.relative_to(ROOT).as_posix(): t701.file_sha(p) for p in t701.hash_files()}


def is_new_out(rel: str) -> bool:
    return any(rel.startswith(f"out/{k}.") or rel.startswith(f"out/svg/{k}/") for k in NEW)


def cmd_hash(which: str) -> None:
    cur = hash_map()
    path = OUT / f"{which}_sha256.txt"
    if which == "before":
        bad = [r for r in cur if is_new_out(r)]
        if bad:
            raise SystemExit(f"out/ sudah memuat berkas klip baru sebelum run: {bad[:5]}")
        OUT.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(f"{h}  {r}\n" for r, h in cur.items()), encoding="utf-8")
        print(f"{path.relative_to(ROOT).as_posix()}: {len(cur)} berkas")
        return
    before = {}
    for line in (OUT / "before_sha256.txt").read_text(encoding="utf-8").splitlines():
        h, r = line.split("  ", 1)
        before[r] = h
    old = {r: h for r, h in cur.items() if r in before}
    extra = sorted(r for r in cur if r not in before)
    missing = sorted(r for r in before if r not in cur)
    changed = sorted(r for r in old if old[r] != before[r])
    foreign = [r for r in extra if not is_new_out(r)]
    path.write_text("".join(f"{h}  {r}\n" for r, h in old.items()), encoding="utf-8")
    res = {"before": len(before), "after_old": len(old), "missing": missing, "changed": changed,
           "identical": not missing and not changed, "new_files": len(extra),
           "new_by_clip": {k: sum(r.startswith(f"out/{k}.") or r.startswith(f"out/svg/{k}/") for r in extra) for k in NEW},
           "new_not_new_clip": foreign}
    jsave(OUT / "f1_check.json", res)
    print(json.dumps({k: (v if not isinstance(v, list) else v[:10]) for k, v in res.items()}, ensure_ascii=False))


# ── run (GPU) ──────────────────────────────────────
def gpu_used_mib() -> tuple[float, float] | None:
    r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True)
    try:
        used, total = (float(v) for v in r.stdout.strip().splitlines()[0].split(","))
        return used, total
    except (ValueError, IndexError):
        return None


def cmd_run(klip: str) -> None:
    RUN.mkdir(parents=True, exist_ok=True)
    video = ROOT / "samples" / f"{klip}.mp4"
    if not video.is_file():
        raise SystemExit(f"tidak ada {video}")
    samples: list[tuple[float, float]] = []
    stop = threading.Event()
    t0 = time.perf_counter()

    def sampler() -> None:
        while not stop.is_set():
            v = gpu_used_mib()
            if v is not None:
                samples.append((round(time.perf_counter() - t0, 1), v[0]))
            stop.wait(VRAM_SAMPLE_S)

    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines: list[str] = []
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    with (RUN / f"{klip}.log").open("a", encoding="utf-8") as fh:
        fh.write(f"\n##### percobaan {started} #####\n")
        p = subprocess.Popen([sys.executable, "-m", "rotoscope", "run", f"samples/{klip}.mp4", "--yes"], cwd=ROOT,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", env=env)
        for line in p.stdout:
            fh.write(line)
            fh.flush()
            lines.append(line.rstrip("\n"))
            print(line, end="", flush=True)
        rc = p.wait()
    wall = time.perf_counter() - t0
    stop.set()
    th.join()
    with (RUN / f"{klip}_vram.csv").open("a", encoding="utf-8") as fh:
        fh.write(f"# percobaan {started}: detik_sejak_mulai,memory.used_MiB (nvidia-smi, total semua proses)\n")
        fh.writelines(f"{t},{u}\n" for t, u in samples)
    times = [ln for ln in lines if ln.startswith("Waktu per stage")]
    rec = {"started_utc": started, "rc": rc, "wall_s": round(wall, 1), "waktu_per_stage": times[-1] if times else None,
           "nvidia_smi_used_peak_mib": max((u for _, u in samples), default=None),
           "nvidia_smi_used_first_mib": samples[0][1] if samples else None, "n_samples": len(samples),
           "vram_errors": [ln for ln in lines if "VRAM bebas" in ln]}
    jpath = RUN / f"{klip}.json"
    hist = jload(jpath) if jpath.is_file() else []
    jsave(jpath, hist + [rec])
    print(json.dumps(rec, ensure_ascii=False))
    if rc != 0:
        raise SystemExit(rc)


# ── metrik ─────────────────────────────────────────
def last_frame_records(path: Path) -> tuple[dict[str, dict], list[dict]]:
    recs, starts = {}, []
    if not path.is_file():
        return recs, starts
    for ln in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(ln)
        if r.get("event") == "frame":
            recs[r["frame"]] = r
        elif r.get("event") == "run_start":
            starts.append(r)
    return recs, starts


def pct(a, q) -> float:
    return round(float(np.percentile(np.asarray(a, float), q)), 4)


def qc_metrics(work: Path) -> dict:
    q = jload(work / "qc_report.json")
    fails = [f for f in q["frames"] if f["fail_reasons"]]
    reasons = Counter(r for f in fails for r in f["fail_reasons"])
    bb = [f["big_blobs"] for f in q["frames"]]
    return {"n_frames": len(q["frames"]), "n_fail": len(fails), "fail_frames": runs_str([f["index"] for f in fails]),
            "fail_reasons": dict(reasons), "qc_version": q.get("qc_version"), "big_blobs_rule": q["qc"].get("big_blobs_rule"),
            "big_blobs_median": float(np.median(bb)), "big_blobs_hist": dict(sorted(Counter(bb).items())),
            "area_vs_median_min": round(q["summary"]["stats"]["area_vs_median"]["min"], 4),
            "iou_prev_min": round(q["summary"]["stats"]["iou_prev"]["min"], 4)}


def fg_metrics(work: Path) -> dict:
    clip = stb.load_clip(work)
    area = clip.width * clip.height
    fg = np.array([100.0 * np.count_nonzero(stb.read_groups(clip.groups_path(n))) / area for n in clip.names])
    return {"fg_pct_median": round(float(np.median(fg)), 2), "fg_pct_p5": pct(fg, 5), "fg_pct_p95": pct(fg, 95),
            "fg_pct_min": round(float(fg.min()), 2), "fg_pct_max": round(float(fg.max()), 2),
            "size": [clip.width, clip.height], "n": len(clip.names)}


def face_metrics(name: str, cfg) -> dict:
    """Wajah V1 terdaftar T-701 1c (`faces_v1_of`: komponen wajah 1b dengan centroid di setengah atas bbox blob tubuhnya;
    wajah tanpa blob tubuh tetap dihitung). Komponen wajah 1b mentah dilaporkan sebagai pembanding."""
    cd = t701.ClipData(name, cfg)
    v1, raw, merged, no_blob = [], [], 0, 0
    for f in range(cd.n):
        blobs, fc = t701.frame_people(cd, f)
        f1 = t701.faces_v1_of(fc, blobs)
        raw.append(len(fc))
        v1.append(len(f1))
        merged += any(sum(x["blob"] == k for x in f1) >= 2 for k in range(len(blobs)))
        no_blob += any(x["blob"] is None for x in f1)
    return {"faces_v1_median": float(np.median(v1)), "faces_v1_hist": dict(sorted(Counter(v1).items())),
            "faces_1b_median": float(np.median(raw)), "frames_v1_merged_blob": merged, "frames_v1_face_without_blob": no_blob,
            "cuts": cd.cuts}


def cut_metrics(work: Path, cfg) -> dict:
    """Skor cut stage [3] (`stb.cut_scores`, sama dengan stabilize) + panjang shot (frame di shot 1 frame = tanpa penghalusan temporal)."""
    clip = stb.load_clip(work)
    scores = stb.cut_scores([stb.frame_thumb(work / "frames" / n) for n in clip.names])
    thr = cfg.stabilize.temporal.cut_diff
    cut = stb.detect_cuts(scores, thr)
    start, end = stb.shot_bounds(cut)
    lens = [e - s + 1 for s, e in sorted(set(zip(start, end)))]
    non = sorted((s for k, s in enumerate(scores) if not cut[k]), reverse=True)
    return {"cut_diff": thr, "scores_at_cuts": {k: round(s, 4) for k, s in enumerate(scores) if cut[k]},
            "max_non_cut": round(non[0], 4) if non else None, "score_p95": pct(scores, 95), "score_median": pct(scores, 50),
            "shot_lengths": lens, "frames_in_1frame_shots": sum(x for x in lens if x == 1)}


def contour_metrics(work: Path) -> dict:
    d = work / "contours"
    docs = [jload(p) for p in sorted(d.glob("frame_*.json"))]
    occ = np.array([sum(s["type"] == OCC for s in x["strokes"]) for x in docs])
    allst = np.array([len(x["strokes"]) for x in docs])
    lens = tm.stroke_lengths_from_dir(d)                       # {(tipe, track_id): panjang} per frame
    b_all = [len(set(lens[t]) - set(lens[t - 1])) for t in range(1, len(lens))]   # definisi t701_proto (semua tipe)
    by_type = {ty: round(float(np.mean([sum(s["type"] == ty for s in x["strokes"]) for x in docs])), 3) for ty in tm.STROKE_TYPES}
    stats = jload(d / "clip_stats.json")
    return {"occ_per_frame": round(float(occ.mean()), 3), "occ_min": int(occ.min()), "occ_max": int(occ.max()),
            "occ_total": int(occ.sum()), "frames_without_occ_pct": round(float(100 * (occ == 0).mean()), 1),
            "frames_without_occ": int((occ == 0).sum()),
            "births_occ_per_frame": round(occ_births(docs), 3), "births_all_per_frame": round(float(np.mean(b_all)), 3),
            "strokes_per_frame": round(float(allst.mean()), 2), "strokes_by_type_per_frame": by_type,
            "t_high": stats["t_high"], "t_low": stats["t_low"], "contract": stats.get("contract")}


def time_metrics(work: Path) -> dict:
    out = {}
    for label, sub in STAGE_LOGS:
        recs, _ = last_frame_records(work / sub / "frames.jsonl")
        tot = [r["total_s"] for r in recs.values()]
        out[label] = {"sum_s": round(float(sum(tot)), 1), "per_frame_s": round(float(np.mean(tot)), 3) if tot else None,
                      "n": len(tot)}
    vram = {}
    for sub in ("seg", "depth"):
        recs, starts = last_frame_records(work / sub / "frames.jsonl")
        vram[sub] = {"peak_reserved_mib": max((r.get("peak_reserved_mib") or 0 for r in recs.values()), default=None),
                     "peak_allocated_mib": max((r.get("peak_allocated_mib") or 0 for r in recs.values()), default=None),
                     "free_before_load_mib": [s.get("vram_free_before_load_mib") for s in starts],
                     "model": sorted({r.get("model") for r in recs.values() if r.get("model")})}
    return {"per_stage": out, "vram": vram}


def run_metrics(klip: str) -> dict | None:
    p = RUN / f"{klip}.json"
    if not p.is_file():
        return None
    hist = jload(p)
    stages = {}
    for rec in hist:
        line = rec.get("waktu_per_stage") or ""
        for m in re.finditer(r"(\[\w+\] \w+) ([\d.]+) s", line):
            stages.setdefault(m.group(1), []).append(float(m.group(2)))
    return {"attempts": len(hist), "rc": [r["rc"] for r in hist], "wall_s": [r["wall_s"] for r in hist],
            "stage_wall_s": stages, "nvidia_smi_used_peak_mib": max((r["nvidia_smi_used_peak_mib"] or 0 for r in hist), default=None),
            "nvidia_smi_used_first_mib": [r["nvidia_smi_used_first_mib"] for r in hist]}


def cmd_metrics() -> None:
    cfg = load_pipeline()
    res = {}
    for name in OLD + NEW:
        work = CLIPS_DIR / name
        if not (work / "contours" / "clip_stats.json").is_file() or not (work / "strokes" / "manifest.json").is_file():
            print(f"{name}: belum lengkap, dilewati", flush=True)
            continue
        t = time.perf_counter()
        meta = jload(work / "meta.json")
        r = {"meta": {k: meta.get(k) for k in ("source_fps", "source_duration_s", "source_width", "source_height", "frame_count",
                                                "working_width", "working_height", "has_audio")},
             "qc": qc_metrics(work), "fg": fg_metrics(work), "people": face_metrics(name, cfg), "cut": cut_metrics(work, cfg),
             "contours": contour_metrics(work), "time": time_metrics(work), "run": run_metrics(name)}
        res[name] = r
        print(f"{name}: {time.perf_counter() - t:.1f} s", flush=True)
    jsave(OUT / "metrics.json", res)
    for name, r in res.items():
        c, q, fg, pp = r["contours"], r["qc"], r["fg"], r["people"]
        print(f"| {name} | {q['n_frames']} | {q['n_fail']} {q['fail_reasons']} ({q['fail_frames']}) | {len(pp['cuts'])} {pp['cuts']} | "
              f"{fg['fg_pct_median']} [{fg['fg_pct_p5']}–{fg['fg_pct_p95']}] | {q['big_blobs_median']} / {pp['faces_v1_median']} | "
              f"{c['t_high']:.5f} / {c['t_low']:.5f} | {c['occ_per_frame']} ({c['occ_min']}–{c['occ_max']}) | "
              f"{c['births_occ_per_frame']} / {c['births_all_per_frame']} | {c['frames_without_occ_pct']} | {c['strokes_per_frame']} |")


# ── bahan penilaian ────────────────────────────────
def fit_h(img: np.ndarray, h: int) -> np.ndarray:
    return cv2.resize(img, (round(img.shape[1] * h / img.shape[0] / 2) * 2, h), interpolation=cv2.INTER_AREA)


def time_bar(width: int, f: int, n: int) -> np.ndarray:
    bar = np.full((TIME_H, width, 3), 235, np.uint8)
    text = f"detik {f / FPS:.2f}".replace(".", ",") + f"   (frame {f} / {n - 1})"
    cv2.putText(bar, text, (14, TIME_H - 22), cv2.FONT_HERSHEY_DUPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
    return bar


def encoder(path: Path, size: tuple[int, int]) -> subprocess.Popen:
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{size[0]}x{size[1]}",
           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", "-r", str(FPS), str(path)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def cmd_media(klip: str) -> None:
    """Berdampingan (kiri foto frames/, kanan hasil out/KLIP.mp4) normal + lambat 0,5× (tiap frame dua kali), dialirkan ke ffmpeg."""
    NILAI.mkdir(parents=True, exist_ok=True)
    clip = stb.load_clip(CLIPS_DIR / klip)
    src_mp4 = ROOT / "out" / f"{klip}.mp4"
    n = len(clip.names)
    enc, size, k = None, None, 0
    for f, res in enumerate(read_frames(src_mp4)):
        if f >= n:
            raise SystemExit(f"{klip}: MP4 hasil lebih panjang dari frames/ ({n})")
        foto = cv2.imdecode(np.fromfile(clip.work_dir / "frames" / clip.names[f], np.uint8), cv2.IMREAD_COLOR)
        a, b = fit_h(foto, PANEL_H), fit_h(res, PANEL_H)
        top = np.hstack([label_bar(a.shape[1], "FOTO"), label_bar(b.shape[1], "HASIL")])
        img = np.vstack([top, np.hstack([a, b])])
        img = np.ascontiguousarray(np.vstack([img, time_bar(img.shape[1], f, n)]))
        if enc is None:
            size = (img.shape[1], img.shape[0])
            enc = (encoder(NILAI / f"{klip}_berdampingan.mp4", size), encoder(NILAI / f"{klip}_lambat_0.5x.mp4", size))
        raw = img.tobytes()
        enc[0].stdin.write(raw)
        enc[1].stdin.write(raw)
        enc[1].stdin.write(raw)
        k += 1
    for p in enc:
        p.stdin.close()
        if p.wait() != 0:
            raise SystemExit(f"{klip}: ffmpeg gagal")
    if k != n:
        raise SystemExit(f"{klip}: MP4 hasil {k} frame ≠ frames/ {n}")
    shutil.copy2(src_mp4, NILAI / f"{klip}_hasil.mp4")
    print(f"{klip}: berdampingan {k} frame, lambat {2 * k} frame, ukuran {size[0]}×{size[1]}, hasil disalin dari out/{klip}.mp4")


def main(argv: list[str]) -> None:
    if argv[:1] == ["hash"] and argv[1:2] in (["before"], ["after"]):
        cmd_hash(argv[1])
    elif argv[:1] == ["run"] and len(argv) == 2 and argv[1] in NEW:
        cmd_run(argv[1])
    elif argv[:1] == ["metrics"]:
        cmd_metrics()
    elif argv[:1] == ["media"] and len(argv) == 2 and argv[1] in NEW:
        cmd_media(argv[1])
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
