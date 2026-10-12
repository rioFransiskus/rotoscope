"""T-305c Tahap 3: run [4] (+ [5] [6]) pada SALINAN scratch untuk hi_floor 0 / 0,02 / 0,025 di 4 klip; regresi netral, metrik, biaya.
Klip asli dan out/ hanya DIBACA (stable/ di-hard-link ke salinan; stage menulis ke salinan). Keluaran: work/t305c_scratch/t3/.

    python scripts/t305c_stage3.py prepare            # salinan scratch (stable/ hard link + meta.json + qc_report.json)
    python scripts/t305c_stage3.py run                # [4] semua lantai x klip; [5] + [6] untuk lantai 0 (4 klip) dan klip3 (semua lantai)
    python scripts/t305c_stage3.py regress            # lantai 0 vs produksi: contours (field demi field), PNG, SVG, MP4
    python scripts/t305c_stage3.py metrics            # tabel Tahap 3 + selisih terhadap simulasi 1b (A_f0.02_pct)
    python scripts/t305c_stage3.py timing             # biaya [4] per frame, berurutan, median 3 pengulangan
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "tests"))
import t305c_eval as ev  # noqa: E402
import t305c_study as st  # noqa: E402
import temporal_metrics as tm  # noqa: E402

ROOT = st.ROOT
T3 = st.SCRATCH / "t3"
FLOORS = ("0", "0.02", "0.025")
CLIPS = st.HASH_CLIPS
SOURCE_NAME = {"test": "test", "test_short": "test_short", "Klip2": "Klip2", "klip3": "klip3"}


def sdir(floor: str, clip: str) -> Path:
    return T3 / f"f{floor}" / clip


def cfg(floor: str) -> Path:
    return T3 / f"cfg_f{floor}.yaml"


def link_or_copy(a: str, b: str) -> None:
    try:
        os.link(a, b)
    except OSError:
        shutil.copy2(a, b)


def cmd_prepare() -> None:
    for fl in FLOORS:
        for c in CLIPS:
            d, src = sdir(fl, c), st.CLIPS_DIR / c
            if (d / "stable").is_dir():
                continue
            d.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src / "stable", d / "stable", copy_function=link_or_copy)
            for f in ("meta.json", "qc_report.json"):
                shutil.copyfile(src / f, d / f)
            print("siap", d.relative_to(ROOT).as_posix())


def stage(floor: str, clip: str, name: str, restart: bool = False) -> tuple[int, float, str]:
    cmd = [sys.executable, "-m", f"rotoscope.{name}", "--config", str(cfg(floor)), "--work-dir", str(sdir(floor, clip))]
    if restart:
        cmd.append("--restart")
    t0 = time.perf_counter()
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    log = T3 / "logs" / f"{name}_f{floor}_{clip}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(r.stdout + "\n--- stderr ---\n" + r.stderr, encoding="utf-8")
    return r.returncode, time.perf_counter() - t0, r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr.strip()[-300:]


def cmd_run() -> None:
    cmd_prepare()
    jobs = [(fl, c, ["vectorize"] + (["stylize", "export"] if fl == "0" or c == "klip3" else [])) for fl in FLOORS for c in CLIPS]
    jobs.sort(key=lambda j: j[1] != "klip3")

    def go(job):
        fl, c, stages = job
        out = []
        for s in stages:
            rc, dt, last = stage(fl, c, s)
            out.append(f"{s} rc={rc} {dt:.1f}s {last[:120]}")
            if rc:
                break
        return f"f{fl} {c}: " + " | ".join(out)

    with ThreadPoolExecutor(max_workers=6) as ex:
        for line in ex.map(go, jobs):
            print(line, flush=True)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def cmd_regress() -> None:
    """Lantai 0 (salinan) vs produksi."""
    res: dict = {}
    for c in CLIPS:
        prod, new = st.CLIPS_DIR / c, sdir("0", c)
        pf = sorted((prod / "contours").glob("frame_*.json"))
        nf = sorted((new / "contours").glob("frame_*.json"))
        diff_keys: dict = {}
        strokes_eq = 0
        for a, b in zip(pf, nf):
            da, db = json.loads(a.read_text(encoding="utf-8")), json.loads(b.read_text(encoding="utf-8"))
            strokes_eq += da["strokes"] == db["strokes"]
            keys = {k for k in set(da) | set(db) if k != "source" and da.get(k) != db.get(k)}
            keys |= {f"source.{k}" for k in set(da["source"]) | set(db["source"]) if da["source"].get(k) != db["source"].get(k)}
            for k in keys:
                diff_keys[k] = diff_keys.get(k, 0) + 1
        sa, sb = st.jload(prod / "contours" / "clip_stats.json"), st.jload(new / "contours" / "clip_stats.json")
        png = [sha(a) == sha(new / "strokes" / a.name) for a in sorted((prod / "strokes").glob("frame_*.png"))]
        svg = [sha(a) == sha(new / "strokes" / a.name) for a in sorted((prod / "strokes").glob("frame_*.svg"))]
        mp4_p, mp4_n = ROOT / "out" / f"{SOURCE_NAME[c]}.mp4", T3 / "f0" / "out" / f"{SOURCE_NAME[c]}.mp4"
        svgdir_p, svgdir_n = ROOT / "out" / "svg" / SOURCE_NAME[c], T3 / "f0" / "out" / "svg" / SOURCE_NAME[c]
        svg_out = [sha(a) == sha(svgdir_n / a.name) for a in sorted(svgdir_p.glob("frame_*.svg"))] if svgdir_n.is_dir() else []
        res[c] = {"frames": [len(pf), len(nf)], "strokes_equal_frames": strokes_eq, "differing_keys_count_frames": diff_keys,
                  "t_high_equal": sa["t_high"] == sb["t_high"], "t_low_equal": sa["t_low"] == sb["t_low"],
                  "t_high_pct_new": sb.get("t_high_pct"), "png_identical": [sum(png), len(png)], "svg_identical": [sum(svg), len(svg)],
                  "mp4_identical": (mp4_p.is_file() and mp4_n.is_file() and sha(mp4_p) == sha(mp4_n)),
                  "mp4_sizes": [mp4_p.stat().st_size if mp4_p.is_file() else None, mp4_n.stat().st_size if mp4_n.is_file() else None],
                  "out_svg_identical": [sum(svg_out), len(svg_out)]}
        print(c, json.dumps(res[c], ensure_ascii=False), flush=True)
    st.jsave(T3 / "regress.json", res)


def births(frames: list[dict]) -> dict:
    """Kelahiran strok oklusi: track_id oklusi di frame t yang tidak ada di t-1 (t >= 1)."""
    ids = [{s["track_id"] for s in d["strokes"] if s["type"] == "occlusion"} for d in frames]
    b = [len(ids[t] - ids[t - 1]) for t in range(1, len(ids))]
    return {"births_per_frame": round(float(np.mean(b)), 4), "births_total": int(sum(b)), "frames_counted": len(b)}


def clip_metrics(floor: str, clip: str, units, idx_of, M) -> dict:
    d = sdir(floor, clip) / "contours"
    docs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(d.glob("frame_*.json"))]
    pop = tm.pop_energy_by_type(tm.stroke_lengths_from_dir(d), ("silhouette", "silhouette_hole", "group_boundary", "occlusion"))
    o = pop["types"]["occlusion"]
    occ_n = np.array([sum(s["type"] == "occlusion" for s in x["strokes"]) for x in docs])
    stats = st.jload(d / "clip_stats.json")
    res = {"occ_total": int(occ_n.sum()), "occ_per_frame": round(float(occ_n.mean()), 4),
           "frames_without_occ_pct": round(float(100 * (occ_n == 0).mean()), 2), "id_new_per_stroke_frame": round(o["id_new_per_stroke_frame"], 4),
           **births(docs), "age_mean": round(o["age_mean"], 4), "age_median": o["age_median"], "pop_occ": round(o["pop_energy_mean"], 5),
           "t_high": stats["t_high"], "t_high_pct": stats.get("t_high_pct"), "t_low": stats["t_low"]}
    prod = st.ClipData(clip)
    if clip in ("test", "test_short"):
        by = {i: [s for s in docs[i]["strokes"] if s["type"] == "occlusion"] for i in st.LEG_FRAMES}
        okf = ev.lower_clothing_frames(prod, by)
        res["done_when_kaki"] = {"frames_ok": okf, "n_ok": len(okf), "of": 5}
    if clip in ("test", "Klip2", "klip3"):
        tallies = []
        for u in units:
            if u["clip"] != clip:
                continue
            kept = 0
            for r in u["rows"]:
                base = next(s for s in prod.docs[r["frame"]]["strokes"] if s["type"] == "occlusion" and s["track_id"] == r["track_id"])
                cand = [s for s in docs[r["frame"]]["strokes"] if s["type"] == "occlusion"]
                if not cand:
                    continue
                tree = cKDTree(np.concatenate([np.asarray(s["points"], float) for s in cand]))
                dist = tree.query(np.asarray(base["points"], float))[0]
                kept += int(np.mean(dist <= ev.MATCH_RADIUS) >= ev.MATCH_FRACTION)
            tallies.append({"key": list(u["key"]), "label": u["label"], "w": u["w"], "cell": u["cell"], "n": len(u["rows"]), "kept": kept})
        res["unit_tallies"] = tallies
    return res


def cmd_metrics() -> None:
    units, _ = ev.load_units()
    idx_of = {u["key"]: u["idx"] for u in units}
    M = ev.boot_mult(units)
    out: dict = {}
    for fl in FLOORS:
        out[fl] = {c: clip_metrics(fl, c, units, idx_of, M) for c in CLIPS}
        tl = [x for c in ("test", "klip3") for x in out[fl][c]["unit_tallies"]]
        out[fl]["labels_gabungan"] = ev.tally_stats(tl, idx_of, M, ("test", "klip3"))
        out[fl]["labels_klip3"] = ev.tally_stats(out[fl]["klip3"]["unit_tallies"], idx_of, M, ("klip3",))
        print("lantai", fl, json.dumps({c: {k: v for k, v in out[fl][c].items() if k != "unit_tallies"} for c in CLIPS}, ensure_ascii=False)[:1800], flush=True)
        print("   label", json.dumps(out[fl]["labels_gabungan"]), flush=True)
    # selisih terhadap simulasi 1b: lantai 0 = base, 0,02 = A_f0.02_pct, 0,025 = A_f0.025_pct
    cmp_: dict = {}
    for fl, var in (("0", "base"), ("0.02", "A_f0.02_pct"), ("0.025", "A_f0.025_pct")):
        cmp_[fl] = {}
        for c in CLIPS:
            sim = st.jload(st.SCRATCH / "sim" / f"{var}__{c}.json")
            m, n = sim["metrics"], out[fl][c]
            d = {k: (n[k] - m[k]) for k in ("occ_total", "occ_per_frame", "frames_without_occ_pct", "age_mean", "pop_occ", "id_new_per_stroke_frame")}
            if "unit_tallies" in sim:
                d["kept_diff_units"] = sum(a["kept"] != b["kept"] for a, b in zip(sim["unit_tallies"], n["unit_tallies"]))
            if "done_when_kaki" in sim:
                d["done_when_same"] = sim["done_when_kaki"]["frames_ok"] == n["done_when_kaki"]["frames_ok"]
            cmp_[fl][c] = d
    out["selisih_vs_simulasi_1b"] = cmp_
    print(json.dumps(cmp_, indent=0))
    st.jsave(T3 / "metrics.json", out)


def cmd_timing() -> None:
    """Biaya [4] per frame (total_s dari frames.jsonl: baca + vectorize + pelacak + tulis), berurutan, median 3 pengulangan."""
    res = {}
    for c in ("test", "klip3"):
        for fl in FLOORS:
            reps = []
            for _ in range(3):
                rc, dt, _ = stage(fl, c, "vectorize", restart=True)
                assert rc == 0
                recs = [json.loads(x) for x in (sdir(fl, c) / "contours" / "frames.jsonl").read_text(encoding="utf-8").splitlines()]
                tot = np.array([r["total_s"] for r in recs if r.get("event") == "frame"]) * 1000
                vec_ = np.array([r["vectorize_s"] for r in recs if r.get("event") == "frame"]) * 1000
                reps.append({"total_mean": float(tot.mean()), "total_p95": float(np.percentile(tot, 95)),
                             "vec_mean": float(vec_.mean()), "vec_p95": float(np.percentile(vec_, 95)), "wall_s": dt})
            res[f"f{fl}__{c}"] = {k: round(float(np.median([r[k] for r in reps])), 2) for k in reps[0]}
            print(c, fl, res[f"f{fl}__{c}"], flush=True)
    st.jsave(T3 / "timing.json", res)


if __name__ == "__main__":
    cmds = {"prepare": cmd_prepare, "run": cmd_run, "regress": cmd_regress, "metrics": cmd_metrics, "timing": cmd_timing}
    if len(sys.argv) < 2 or sys.argv[1] not in cmds:
        raise SystemExit(__doc__)
    cmds[sys.argv[1]]()
