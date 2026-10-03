"""Metrik kualitas pelacak batas grup (T-201a), dipakai test_vectorize.py sebagai syarat lulus permanen.

- cakupan: persentase piksel skeleton pasca-thinning yang berada ≤ 1 px (Chebyshev) dari titik mana pun pada
  polyline group_boundary yang dipancarkan untuk pasangan grup itu;
- run tak-tercakup: komponen 8-arah piksel skeleton yang tidak tercakup (+ berapa bagian tercakup yang diapitnya);
- fragmentasi: jumlah strok per pasangan vs jumlah komponen 8-arah skeleton pasangan itu;
- integritas: titik berulang dalam satu strok, loncatan (jarak Chebyshev > 1 piksel antar titik berurutan).
"""

from __future__ import annotations

import cv2
import numpy as np

from rotoscope import vectorize as vec

K3 = np.ones((3, 3), np.uint8)


def pixels(stroke: dict) -> list[tuple[int, int]]:
    """Titik strok (pusat piksel i + 0.5) → indeks piksel (y, x)."""
    return [(int(y - 0.5), int(x - 0.5)) for x, y in stroke["points"]]


def integrity(strokes: list[dict]) -> dict:
    """Jumlah titik berulang (di luar penutup loop) dan loncatan (> 1 piksel) di semua strok."""
    repeats = jumps = 0
    for s in strokes:
        px = pixels(s)
        closed = len(px) > 3 and px[0] == px[-1]
        body = px[:-1] if closed else px
        repeats += len(body) - len(set(body))
        jumps += sum(max(abs(a[0] - b[0]), abs(a[1] - b[1])) > 1 for a, b in zip(px, px[1:]))
    return {"repeats": repeats, "jumps": jumps}


def measure(gmap: np.ndarray, names: tuple[str, ...], params: dict) -> dict:
    """Cakupan + run tak-tercakup + fragmentasi untuk satu peta grup (semua pasangan)."""
    strokes, _ = vec.group_boundary_strokes(gmap, names, params["line_min_px"], params["min_stroke_px"])
    per_pair: dict[tuple[str, str], list[dict]] = {}
    for s in strokes:
        per_pair.setdefault(tuple(s["groups"]), []).append(s)
    total = uncovered = 0
    runs: list[dict] = []
    pairs: list[dict] = []
    for a, b, skel, ox, oy in vec.pair_skeletons(gmap, params["line_min_px"]):
        mine = per_pair.get((names[a - 1], names[b - 1]), [])
        h, w = skel.shape
        cov = np.zeros((h, w), np.uint8)
        for s in mine:
            for y, x in pixels(s):
                if 0 <= y - oy < h and 0 <= x - ox < w:
                    cov[y - oy, x - ox] = 1
        covered = cv2.dilate(cov, K3).astype(bool)
        unc = skel & ~covered
        n_comp = cv2.connectedComponents(skel.astype(np.uint8), connectivity=8)[0] - 1
        pairs.append({"pair": (names[a - 1], names[b - 1]), "skeleton": int(skel.sum()),
                      "uncovered": int(unc.sum()), "components": n_comp, "strokes": len(mine)})
        total += int(skel.sum())
        uncovered += int(unc.sum())
        if unc.any():
            k, lab, st, cen = cv2.connectedComponentsWithStats(unc.astype(np.uint8), connectivity=8)
            _, clab = cv2.connectedComponents((skel & covered).astype(np.uint8), connectivity=8)
            for i in range(1, k):
                m = lab == i
                ring = cv2.dilate(m.astype(np.uint8), K3).astype(bool) & ~m
                runs.append({"pair": (names[a - 1], names[b - 1]), "px": int(st[i, cv2.CC_STAT_AREA]),
                             "xy": (int(cen[i][0]) + ox, int(cen[i][1]) + oy),
                             "flanks": len({int(v) for v in clab[ring] if v > 0})})
    return {"skeleton": total, "uncovered": uncovered,
            "coverage": 100.0 * (1 - uncovered / total) if total else 100.0,
            "runs": runs, "pairs": pairs, "integrity": integrity(strokes)}


# ── Garis oklusi (T-201b) ──────────────────────────
K5 = np.ones((5, 5), np.uint8)
SHADOW_TOLERANCE_PX = 0.5      # titik strok = pusat piksel; jarak D dihitung antar pusat piksel → selisih ≤ 0,5 px


def shadow_distance(occlusion: list[dict], others: list[dict]) -> float | None:
    """'Bayangan': jarak minimum (px) dari titik oklusi mana pun ke titik silhouette / silhouette_hole /
    group_boundary terdekat. Syarat lulus: ≥ min_dist_px − SHADOW_TOLERANCE_PX. None kalau salah satunya kosong."""
    from scipy.spatial import cKDTree

    a = [p for s in occlusion for p in s["points"]]
    b = [p for s in others for p in s["points"]]
    if not a or not b:
        return None
    return float(cKDTree(np.array(b, float)).query(np.array(a, float))[0].min())


def occlusion_measure(gmap: np.ndarray, depth: np.ndarray, names: tuple[str, ...], params: dict,
                      t_high: float, t_low: float) -> dict:
    """Metrik objektif garis oklusi satu frame (tanpa mata):
    - skeleton: % piksel skeleton (≥ L) yang berada ≤ 1 px dari titik strok (cakupan skeleton);
    - band: % piksel band (komponen mask pra-thinning yang skeleton-nya lolos L) ≤ 2 px dari titik strok — pelengkap,
      karena cakupan skeleton buta terhadap pengikisan thinning;
    - components / strokes: fragmentasi; integrity: titik berulang + loncatan; shadow: jarak minimum ke strok tipe lain;
    - per_group: jumlah strok oklusi per grup."""
    dp = vec.depth_params(params)
    strokes, stats = vec.occlusion_strokes(gmap, depth, names, params, t_high, t_low)
    masks = vec.occlusion_masks(gmap, depth, dp, t_high, t_low)
    cov = np.zeros(gmap.shape, np.uint8)
    for s in strokes:
        for y, x in pixels(s):
            cov[y, x] = 1
    near1, near2 = cv2.dilate(cov, K3).astype(bool), cv2.dilate(cov, K5).astype(bool)
    skel_n = skel_unc = band_n = band_unc = components = 0
    for gid in (int(g) for g in np.unique(gmap[masks["dist_ok"]])):
        mask = masks["dist_ok"] & (gmap == gid)
        thinned = vec.thin_band(mask, params["line_min_px"])
        if thinned is None:
            continue
        skel, ox, oy = thinned
        skel = vec.keep_long_components(skel, dp["min_len_px"])
        full = np.zeros(gmap.shape, bool)
        ys, xs = np.nonzero(skel)
        full[ys + oy, xs + ox] = True
        if not full.any():
            continue
        components += cv2.connectedComponents(full.astype(np.uint8), connectivity=8)[0] - 1
        _, lab = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
        ids = np.unique(lab[full])
        band = np.isin(lab, ids[ids > 0])
        skel_n, skel_unc = skel_n + int(full.sum()), skel_unc + int((full & ~near1).sum())
        band_n, band_unc = band_n + int(band.sum()), band_unc + int((band & ~near2).sum())
    others = vec.vectorize_gmap(gmap, names, params)[0]
    per_group: dict[str, int] = {}
    for s in strokes:
        per_group[s["groups"][0]] = per_group.get(s["groups"][0], 0) + 1
    return {"strokes": len(strokes), "components": components, "skeleton": skel_n, "band": band_n,
            "skeleton_coverage": 100.0 * (1 - skel_unc / skel_n) if skel_n else 100.0,
            "band_coverage": 100.0 * (1 - band_unc / band_n) if band_n else 100.0,
            "integrity": integrity(strokes), "shadow": shadow_distance(strokes, others),
            "per_group": per_group, "stats": stats}
