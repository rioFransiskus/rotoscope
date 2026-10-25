"""T-304 data E (alat, F3): data klip5 tanpa render — per varian: frame kesetiaan terburuk, status QC, jarak ke cut, zoom f63–f77;
kesetiaan ulang tanpa frame gagal QC; T_high / T_low [4]; uji dugaan "strok berubah karena ambang per klip" (b 0,3 vs B-sehat).

    python scripts/t304_data_e.py  -> scripts/t304_data_e.json (ter-track; angka saja, tanpa kunci huruf)

Input: work/t304_scratch/<var>/klip5/ (stable/, contours/; Tahap 1), work/t304/raw/klip5.npy (argmax seg mentah, Tahap 1).
Hanya membaca. Kunci huruf (scripts/t304_kunci.json) TIDAK dibaca di sini: data E memakai nama varian (ini bukan papan buta).
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import temporal_metrics as tm  # noqa: E402
from rotoscope import stabilize as stb  # noqa: E402

SCR = ROOT / "work" / "t304_scratch"
OUT = ROOT / "scripts" / "t304_data_e.json"
VARS = ("b0", "b0.3", "b0.6", "b1.0", "bs")
CLIP = "klip5"
FPS = 24
ZOOM = (63, 77)                     # zoom f63–f77 (decision Rio, T304 keputusan D/E)


def per_frame_fidelity(g: np.ndarray, raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """fg-IoU per frame dan IoU grup MINIMUM per frame (grup ≥ GROUP_MIN_PX di mentah). Frame tanpa grup → 1."""
    fgi = np.array([((g[t] != 0) & (raw[t] != 0)).sum() / max(((g[t] != 0) | (raw[t] != 0)).sum(), 1) for t in range(len(g))])
    gmin = np.ones(len(g))
    for t in range(len(g)):
        vals = []
        for gid in np.unique(raw[t]):
            if gid == 0 or (raw[t] == gid).sum() < tm.GROUP_MIN_PX:
                continue
            a, b = g[t] == gid, raw[t] == gid
            vals.append((a & b).sum() / (a | b).sum())
        if vals:
            gmin[t] = min(vals)
    return fgi, gmin


def canon(points: list) -> tuple:
    """Bentuk kanonik strok tertutup: rotasi + arah dinormalkan (anchor / orientasi tidak ikut dihitung)."""
    pts = [tuple(p) for p in points]
    n = len(pts)
    best = None
    for seq in (pts, pts[::-1]):
        for s in range(n):
            cand = tuple(seq[s:] + seq[:s])
            if best is None or cand < best:
                best = cand
    return best


def geom_key(stroke: dict) -> tuple:
    """Strok tertutup: rotasi + arah dinormalkan (anchor). Strok terbuka: arah dinormalkan (arah berkesinambungan T-202
    bisa membalik garis yang sama) — keduanya tidak mengubah geometri garis."""
    pts = [tuple(p) for p in stroke["points"]]
    if stroke["closed"]:
        return (stroke["type"], canon(pts))
    return (stroke["type"], min(tuple(pts), tuple(pts[::-1])))


def stroke_sets(work: Path) -> list[Counter]:
    out = []
    for f in sorted((work / "contours").glob("frame_*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        out.append(Counter(geom_key(s) for s in d["strokes"]))
    return out


def sym_diff(a: Counter, b: Counter) -> tuple[int, Counter, Counter]:
    d_ab, d_ba = a - b, b - a
    return sum(d_ab.values()) + sum(d_ba.values()), d_ab, d_ba


def main() -> None:
    raw = np.load(ROOT / "work" / "t304" / "raw" / f"{CLIP}.npy")
    n = len(raw)
    work0 = SCR / "b0.3" / CLIP
    c0 = stb.load_clip(work0)
    st0 = json.loads((work0 / "stable" / "manifest.json").read_text(encoding="utf-8"))["temporal"]
    cut = {int(Path(x).stem.split("_")[1]) for x in st0["cut_frames"]}
    fails = {int(Path(x).stem.split("_")[1]) for x in st0["qc_fail_frames"]}
    keep = np.array([t not in fails for t in range(n)])
    cut_arr = np.array(sorted(cut)) if cut else np.array([])

    def cut_dist(t: int) -> int | None:
        if not cut_arr.size:
            return None
        return int(np.min(np.abs(cut_arr - t)))

    res = {"clip": CLIP, "frames": n, "fps": FPS, "qc_fail_frames": sorted(fails), "cut_frames": sorted(cut),
           "zoom_range_frames": list(ZOOM), "variants": {}}
    fid = {}
    for var in VARS:
        work = SCR / var / CLIP
        c = stb.load_clip(work)
        g = np.stack([stb.read_groups(c.groups_path(nm)) for nm in c.names])
        fgi, gmin = per_frame_fidelity(g, raw)
        fid[var] = (fgi, gmin)
        worst_fg = sorted(range(n), key=lambda t: fgi[t])[:3]
        worst_gr = sorted(range(n), key=lambda t: gmin[t])[:3]

        def row(t: int) -> dict:
            return {"frame": t, "detik": round(t / FPS, 2), "fg_iou": round(float(fgi[t]), 4),
                    "group_iou_min": round(float(gmin[t]), 4), "qc_gagal": t in fails, "jarak_cut_frame": cut_dist(t),
                    "zoom": ZOOM[0] <= t <= ZOOM[1]}

        keep_fg, keep_gr = fgi[keep], gmin[keep]
        cl = json.loads((work / "contours" / "clip_stats.json").read_text(encoding="utf-8"))
        res["variants"][var] = {
            "t_high": cl["t_high"], "t_low": cl["t_low"],
            "frame_fg_iou_min": row(worst_fg[0]), "frame_fg_iou_terburuk3": [row(t) for t in worst_fg],
            "frame_group_iou_min": row(worst_gr[0]), "frame_group_iou_terburuk3": [row(t) for t in worst_gr],
            "fidelity_all": {"fg_iou_min": round(float(fgi.min()), 4), "group_iou_min": round(float(gmin.min()), 4)},
            "fidelity_tanpa_qc": {"frames": int(keep.sum()), "fg_iou_min": round(float(keep_fg.min()), 4),
                                  "group_iou_min": round(float(keep_gr.min()), 4),
                                  "fg_iou_median": round(float(np.median(keep_fg)), 4),
                                  "group_iou_median": round(float(np.median(keep_gr)), 4)},
        }
        print(var, res["variants"][var]["t_high"], res["variants"][var]["t_low"], res["variants"][var]["fidelity_tanpa_qc"])

    # Uji hipotesis: apakah strok berubah karena ambang [4] (T_high / T_low) berbeda?
    ref = SCR / "b0.3" / CLIP
    ref_sets = stroke_sets(ref)
    test = {}
    for var in ("bs",):
        other = stroke_sets(SCR / var / CLIP)
        n_changed_all, n_changed_occ_only, n_changed_no_occ, frames_changed = 0, 0, 0, []
        changed_no_px = []
        groups_same = np.array([np.array_equal(stb.read_groups(stb.load_clip(ref).groups_path(nm)),
                                               stb.read_groups(stb.load_clip(SCR / var / CLIP).groups_path(nm)))
                                for nm in c0.names])
        for t in range(n):
            k, d_ab, d_ba = sym_diff(ref_sets[t], other[t])
            if k:
                frames_changed.append(t)
                types = {x[0] for x in (*d_ab, *d_ba)}
                if types == {"occlusion"}:
                    n_changed_occ_only += 1
                if "occlusion" not in types:
                    n_changed_no_occ += 1
                if groups_same[t]:
                    changed_no_px.append(t)
            n_changed_all += k
        same_thr = (json.loads((ref / "contours" / "clip_stats.json").read_text(encoding="utf-8"))["t_high"]
                    == json.loads((SCR / var / CLIP / "contours" / "clip_stats.json").read_text(encoding="utf-8"))["t_high"])
        test[var] = {"vs": "b0.3", "t_high_sama": same_thr, "frames_strokes_berubah": len(frames_changed),
                     "strok_selisih_total_geometri": n_changed_all,
                     "frames_hanya_occlusion": n_changed_occ_only, "frames_tanpa_occlusion": n_changed_no_occ,
                     "frames_strok_berubah_peta_sama": len(changed_no_px),
                     "contoh_frames_peta_sama": changed_no_px[:12]}
    res["uji_hipotesis_ambang"] = test
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"  -> {OUT.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
