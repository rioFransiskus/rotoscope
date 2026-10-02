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
