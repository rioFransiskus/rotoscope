"""T-304 Tahap 1 (alat, F3): kalibrasi `boil_preserve` b — varian di salinan scratch, tanpa mengubah work/clips/ atau out/.

    python scripts/t304_tahap1.py run VAR KLIP        # [3] + [4] in-process di work/t304_scratch/VAR/KLIP/ -> work/t304/run/VAR_KLIP.json
    python scripts/t304_tahap1.py metrics             # metrik per varian x klip -> work/t304/tahap1_metrics.json
    python scripts/t304_tahap1.py porsi               # verifikasi hitungan porsi bobot frame t (Konteks T304-prompt) -> work/t304/porsi.json
    python scripts/t304_tahap1.py bsehat              # B-sehat vs b 0,3 (frame gagal QC + tetangga, [4] hilir) -> work/t304/bsehat.json

Varian: b0 / b0.3 / b0.6 / b1.0 = `stabilize.temporal.boil_preserve`; `bs` = prototipe B-sehat (scratch saja, b nominal 0,3):
b = 0 di frame yang SENDIRI gagal QC (q < 1), b nominal di frame lain. Parameter lain = configs/default.yaml (produksi).

Input (meta.json, qc_report.json, frames/, seg/, depth/, isolate/, target/) di-HARD-LINK dari work/clips/KLIP (tanpa salinan
byte, tanpa ruang tambahan); stable/ + contours/ ditulis baru di scratch. Klip asli dan out/ hanya dibaca. CPU saja.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "scripts"))

import stable_preview as sp  # noqa: E402  (load_raw: argmax mentah seg, tanpa temporal)
import temporal_metrics as tm  # noqa: E402
from rotoscope import stabilize as stb  # noqa: E402
from rotoscope import vectorize as vz  # noqa: E402
from rotoscope.config import load_pipeline  # noqa: E402
from rotoscope.stage_common import reconfigure_stdio, work_dir_overrides  # noqa: E402

CLIPS_DIR = ROOT / "work" / "clips"
SCR = ROOT / "work" / "t304_scratch"
OUT = ROOT / "work" / "t304"
CFG = ROOT / "configs" / "default.yaml"
CLIPS = ("test", "test_short", "Klip2", "klip3", "klip4", "klip5")
VARIANTS = {"b0": 0.0, "b0.3": 0.3, "b0.6": 0.6, "b1.0": 1.0, "bs": 0.3}
INPUT_FILES = ("meta.json", "qc_report.json")
INPUT_DIRS = ("frames", "seg", "depth", "isolate", "target")
FPS_DEFAULT = 24
WIN_QC_HALF = 10                  # jendela gagal QC = 21 frame berpusat di frame gagal pertama


class BSehatCache(stb.FrameCache):
    """Prototipe B-sehat (scratch): sama dengan FrameCache.groups, tetapi b = 0 bila frame t sendiri gagal QC (q[t] < 1)."""

    def groups(self, k: int):
        if k not in self._groups:
            w = self.plan.weights(k, self.plan.radius)
            sums, tie, _ = self.raw(k)
            t0 = time.perf_counter()
            if len(w) == 1:
                p = sums
            else:
                acc = np.zeros(sums.shape, np.float32)
                total = 0.0
                for j, wt in w:
                    acc += np.float32(wt) * self.raw(j)[0]
                    total += wt
                p = acc / np.float32(total)
                b = 0.0 if self.plan.q[k] < 1.0 else self.plan.boil
                if b > 0:
                    p = np.float32(1.0 - b) * p + np.float32(b) * sums
            st = self.cfg.stabilize
            self._groups[k] = stb.clean_groups(p, tie, st.island_min_px, st.mode_k)
            self.timers["groups"] += time.perf_counter() - t0
        return self._groups[k]


def jsave(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"  -> {p.relative_to(ROOT).as_posix()}")


def _link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.link(src, dst)


def scratch_dir(var: str, clip: str) -> Path:
    """Salinan scratch: input di-hard-link dari klip asli; stable/ dan contours/ tidak ada (ditulis run)."""
    src, dst = CLIPS_DIR / clip, SCR / var / clip
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for f in INPUT_FILES:
        _link(src / f, dst / f)
    for d in INPUT_DIRS:
        s = src / d
        if s.is_dir():
            for f in sorted(s.rglob("*")):
                if f.is_file():
                    _link(f, dst / d / f.relative_to(s))
    return dst


def cfg_for(var: str, work: Path):
    return load_pipeline(CFG, overrides={**work_dir_overrides(work), "stabilize.temporal.boil_preserve": VARIANTS[var]})


# ── run ────────────────────────────────────────────
def cmd_run(var: str, clip: str) -> None:
    if var not in VARIANTS:
        raise SystemExit(f"varian tidak dikenal: {var}")
    work = scratch_dir(var, clip)
    cfg = cfg_for(var, work)
    if var == "bs":
        stb.FrameCache = BSehatCache                         # run_stabilize memakai nama modul ini (proses sendiri)
    t0 = time.perf_counter()
    r3 = stb.run_stabilize(cfg, log=print)
    t3 = time.perf_counter() - t0
    t1 = time.perf_counter()
    r4 = vz.run_vectorize(cfg, log=print)
    t4 = time.perf_counter() - t1
    jsave(OUT / "run" / f"{var}_{clip}.json", {
        "variant": var, "clip": clip, "b": VARIANTS[var], "t_stage3_s": round(t3, 2), "t_stage4_s": round(t4, 2),
        "stage3": {k: r3[k] for k in ("selected", "skipped", "processed", "deferred", "stale")},
        "stage4": {k: r4[k] for k in r4 if k in ("selected", "skipped", "processed")}})


# ── bahan bersama metrik ───────────────────────────
def clip_obj(clip: str) -> stb.Clip:
    return stb.load_clip(CLIPS_DIR / clip)


def raw_groups(clip: str, cfg) -> np.ndarray:
    cache = OUT / "raw" / f"{clip}.npy"
    if not cache.is_file():
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, sp.load_raw(clip_obj(clip), cfg))
    return np.load(cache)


def fps_of(clip: str) -> int:
    try:
        return int(json.loads((CLIPS_DIR / clip / "meta.json").read_text(encoding="utf-8")).get("fps", FPS_DEFAULT))
    except (OSError, ValueError, TypeError):
        return FPS_DEFAULT


def windows(clip: str, raw: np.ndarray, cfg) -> dict:
    """Jendela papan (aturan T-302) dari argmax mentah + 1 jendela gagal QC (21 frame berpusat di frame gagal pertama)."""
    n = len(raw)
    fails = [k for k, v in enumerate(stb.qc_fail_flags(clip_obj(clip)).values()) if v is True]
    qc = []
    if fails:
        c = fails[0]
        qc = [(max(0, c - WIN_QC_HALF), min(n - 1, c + WIN_QC_HALF))]
    return {"static": tm.static_windows(raw), "fast": tm.fast_windows(raw), "qc": qc, "qc_fail_frames": fails, "n": n,
            "fps": fps_of(clip)}


def load_groups(work: Path, clip: stb.Clip) -> np.ndarray:
    return np.stack([stb.read_groups(clip.groups_path(n)) for n in clip.names])


def contour_frames(work: Path) -> list[dict]:
    return tm.stroke_lengths_from_dir(work / "contours")


def strokes_geom(work: Path) -> list[Counter]:
    """Per frame: multiset (tipe, points) — geometri saja, TANPA track_id (id ikut berubah bila frame sebelumnya berubah)."""
    out = []
    for f in sorted((work / "contours").glob("frame_*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        out.append(Counter((s["type"], json.dumps(s["points"])) for s in d["strokes"]))
    return out


def geom_diff(a: Counter, b: Counter) -> tuple[int, set[str]]:
    """Jumlah strok yang tidak ada di pasangan lain (simetris, multiset) + tipe yang terlibat."""
    d_ab, d_ba = a - b, b - a
    types = {k[0] for k in (*d_ab, *d_ba)}
    return sum(d_ab.values()) + sum(d_ba.values()), types


def wmask(n: int, runs) -> np.ndarray:
    return tm.window_mask(n, runs)


def measure(raw: np.ndarray, g: np.ndarray, win: dict, frames: list[dict]) -> dict:
    n = len(raw)
    st, fa, qc = wmask(n, win["static"]), wmask(n, win["fast"]), wmask(n, win["qc"])
    cen_err = tm.centroid_error(g, raw)
    lr = tm.limb_ratio(g, raw)
    gi = tm.group_iou(g, raw)
    fgi = tm.fg_iou(g, raw)
    pop = tm.pop_energy_by_type(frames)
    keep = ("pop_energy_mean", "pop_sum", "born_per_frame", "died_per_frame", "length_share_mean", "id_new_per_stroke_frame",
            "age_mean", "age_median", "frames_without", "pop_share_pct")
    return {
        "flipflop_per10k": {"all": tm.flipflop_per10k(g), "static": tm.flipflop_per10k(g, st) if st.any() else None,
                            "fast": tm.flipflop_per10k(g, fa) if fa.any() else None,
                            "qc": tm.flipflop_per10k(g, qc) if qc.any() else None},
        "fidelity": {"fg_iou_min": float(fgi.min()), "fg_iou_median": float(np.median(fgi)),
                     "group_iou_min": float(gi.min()) if gi.size else None,
                     "group_iou_median": float(np.median(gi)) if gi.size else None,
                     "centroid_err_median_px": float(np.nanmedian(cen_err)), "centroid_err_p95_px": float(np.nanpercentile(cen_err, 95)),
                     "limb_ratio_min": float(lr.min()), "limb_ratio_median": float(np.median(lr)),
                     "area_lag_frames": tm.area_lag(g, raw)},
        "pop": {"total_pop_energy_mean": pop["total_pop_energy_mean"],
                "types": {ty: {k: pop["types"][ty][k] for k in keep} for ty in tm.STROKE_TYPES}},
    }


def cmd_metrics() -> None:
    res = {}
    for clip in CLIPS:
        cfg = load_pipeline(CFG)
        raw = raw_groups(clip, cfg)
        win = windows(clip, raw, cfg)
        ref_frames = None
        rows = {}
        for var in VARIANTS:
            work = SCR / var / clip
            if not (work / "stable" / "manifest.json").is_file():
                rows[var] = "BELUM DIHITUNG"
                continue
            c = stb.load_clip(work)
            g = load_groups(work, c)
            frames = contour_frames(work)
            rows[var] = measure(raw, g, win, frames)
        res[clip] = {"windows": win, "variants": rows}
        print(f"{clip}: {len(raw)} frame, jendela {win}")
    jsave(OUT / "tahap1_metrics.json", res)


# ── porsi bobot (verifikasi hitungan Konteks T304-prompt) ─
def cmd_porsi() -> None:
    cfg = load_pipeline(CFG)
    res = {}
    for clip in CLIPS:
        c = clip_obj(clip)
        plan = stb.build_plan(cfg, c, log=lambda *_: None)
        healthy = [k for k in range(plan.n) if plan.q[k] == 1.0 and plan.window(k, plan.radius)[0] < k < plan.window(k, plan.radius)[1]]
        failing = [k for k in range(plan.n) if plan.q[k] < 1.0]
        rows = {}
        for label, ks in (("sehat", healthy[len(healthy) // 2:len(healthy) // 2 + 1]), ("gagal_qc", failing[:1])):
            for k in ks:
                w = plan.weights(k, plan.radius)
                W = sum(x for _, x in w)
                own = dict(w).get(k, 0.0)
                share = {f"b{b}": round((1 - b) * own / W + b, 4) for b in (0.0, 0.3, 0.6, 1.0)}
                rows[f"{label}_frame{k}"] = {"own_w": round(own, 5), "total_w": round(W, 5), "share": share,
                                             "neighbors_fail": [j for j, _ in w if plan.q[j] < 1.0]}
        res[clip] = {"n": plan.n, "radius": plan.radius, "rho": round(plan.rho, 5), "qc_fail": list(plan.qc_fail_frames),
                     "cut": list(plan.cut_frames), "frames": rows}
        print(clip, json.dumps(rows, ensure_ascii=False))
    jsave(OUT / "porsi.json", res)


# ── B-sehat vs b 0,3 ───────────────────────────────
def cmd_bsehat() -> None:
    res = {}
    for clip in CLIPS:
        c = clip_obj(clip)
        cfg = load_pipeline(CFG)
        raw = raw_groups(clip, cfg)
        fails = [k for k, v in enumerate(stb.qc_fail_flags(c).values()) if v is True]
        if not fails:
            res[clip] = {"qc_fail": []}
            continue
        wa, wb = SCR / "b0.3" / clip, SCR / "bs" / clip
        ca, cb = stb.load_clip(wa), stb.load_clip(wb)
        ga, gb = load_groups(wa, ca), load_groups(wb, cb)
        sa, sb = strokes_geom(wa), strokes_geom(wb)
        diffs = [geom_diff(sa[f], sb[f]) for f in range(len(raw))]
        per = {}
        for f in sorted({j for k in fails for j in range(max(0, k - 2), min(len(raw), k + 3))}):
            ch = int((ga[f] != gb[f]).sum())
            fa, fb = ga[f] != 0, gb[f] != 0
            iou = float((fa & fb).sum() / max((fa | fb).sum(), 1))
            per[f] = {"name": c.names[f], "fail": f in fails, "px_changed": ch, "fg_iou": round(iou, 5),
                      "strokes_diff": diffs[f][0], "strokes_diff_types": sorted(diffs[f][1])}
        total = {"px_changed_all": int(sum(int((ga[f] != gb[f]).sum()) for f in range(len(raw)))),
                 "frames_groups_changed": int(sum(int((ga[f] != gb[f]).any()) for f in range(len(raw)))),
                 "frames_strokes_changed": int(sum(1 for d in diffs if d[0]))}
        res[clip] = {"qc_fail": [c.names[k] for k in fails], "frames": per, "total": total}
        print(clip, json.dumps(total))
    jsave(OUT / "bsehat.json", res)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("var")
    r.add_argument("clip")
    sub.add_parser("metrics")
    sub.add_parser("porsi")
    sub.add_parser("bsehat")
    a = ap.parse_args(argv)
    reconfigure_stdio()
    if a.cmd == "run":
        cmd_run(a.var, a.clip)
    elif a.cmd == "metrics":
        cmd_metrics()
    elif a.cmd == "porsi":
        cmd_porsi()
    else:
        cmd_bsehat()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
