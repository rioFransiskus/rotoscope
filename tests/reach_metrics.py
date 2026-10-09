"""Metrik keselamatan histeresis jarak garis oklusi (T-305b), dipakai test_vectorize_reach.py sebagai syarat lulus permanen.

- komponen baru: strok varian tanpa satu pun titik ≤ 1 px (Chebyshev) dari strok dasar (D_low = 0) — histeresis terjaga
  hanya MEMPERPANJANG strok yang sudah ada, tidak melahirkan strok;
- run zona D: runs titik strok dengan jarak ke strok tipe lain < D − toleransi. Run yang menyentuh ujung strok = ekstensi
  (panjang dibatasi), run di TENGAH strok = pelanggaran ('bayangan' sadar zona D), run seluruh strok = duplikat batas;
- batas panjang ekstensi: ceil((D − D_low) / sin θ) langkah (+ 1 langkah jalur) dikali √2 (diagonal) dalam px;
- jarak ujung: jarak ujung terbuka (bukan tepi frame) ke strok LAIN terdekat → histogram kelas T-305a.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree

from rotoscope import vectorize as vec

ZONE_TOL = 0.5                      # jarak D dihitung antar pusat piksel (selisih ≤ 0,5 px, sama dengan SHADOW_TOLERANCE_PX)
GAP_CLASSES = (("<=1", 0.0, 1.0), ("1-3", 1.0, 3.0), ("3-6", 3.0, 6.0), ("6-10", 6.0, 10.0), ("10-20", 10.0, 20.0),
               (">20", 20.0, math.inf))


def pts(stroke: dict) -> np.ndarray:
    return np.asarray(stroke["points"], float)


def runs(flags: np.ndarray) -> list[tuple[int, int]]:
    out, start = [], None
    for k, f in enumerate(flags):
        if f and start is None:
            start = k
        if not f and start is not None:
            out.append((start, k - 1))
            start = None
    if start is not None:
        out.append((start, len(flags) - 1))
    return out


def new_components(base: list[dict], var: list[dict]) -> int:
    """Jumlah strok varian yang tidak menyentuh strok dasar manapun (≤ 1 px Chebyshev)."""
    if not var:
        return 0
    bp = np.concatenate([pts(s) for s in base]) if base else np.zeros((0, 2))
    if not len(bp):
        return len(var)
    tree = cKDTree(bp)
    return sum(1 for s in var if tree.query(pts(s), p=np.inf)[0].min() > 1.0 + 1e-9)


def zone_runs(occ: list[dict], others: list[dict], d: float) -> list[dict]:
    """Run titik oklusi dengan jarak ke strok tipe lain < d − ZONE_TOL: {stroke, a, b, kind, length}; kind ∈
    {"end", "middle", "whole"} ("whole" = seluruh strok di zona: duplikat batas)."""
    op = np.concatenate([pts(s) for s in others]) if others else np.zeros((0, 2))
    if not len(op):
        return []
    tree = cKDTree(op)
    out = []
    for j, s in enumerate(occ):
        p = pts(s)
        for a, b in runs(tree.query(p)[0] < d - ZONE_TOL):
            kind = "whole" if (a == 0 and b == len(p) - 1) else "end" if (a == 0 or b == len(p) - 1) else "middle"
            seg = p[max(a - 1, 0):min(b + 2, len(p))]
            out.append({"stroke": j, "a": a, "b": b, "kind": kind,
                        "length": float(np.hypot(*np.diff(seg, axis=0).T).sum()) if len(seg) > 1 else 0.0})
    return out


def reach_bound_px(d: float, d_low: float) -> float:
    """Panjang jalur ekstensi maksimum (px) dari kontrak: ceil((D − D_low) / sin θ) langkah 8-arah, diagonal √2, + 1 langkah."""
    steps = math.ceil((d - d_low) / math.sin(math.radians(vec.OCC_REACH_MIN_ANGLE_DEG)))
    return (steps + 1) * math.sqrt(2)


def end_gaps(occ: list[dict], others: list[dict], width: int, height: int, edge_tol: float = 1.0) -> list[float]:
    """Jarak (px kerja) ujung terbuka oklusi (bukan tepi frame / loop) ke titik strok LAIN terdekat."""
    gaps = []
    for j, s in enumerate(occ):
        p = pts(s)
        if len(p) > 3 and (p[0] == p[-1]).all():
            continue
        rest = [pts(o) for k, o in enumerate(occ) if k != j] + [pts(o) for o in others]
        tree = cKDTree(np.concatenate(rest)) if rest else None
        for e in (0, -1):
            x, y = p[e]
            if x <= 0.5 + edge_tol or y <= 0.5 + edge_tol or x >= width - 0.5 - edge_tol or y >= height - 0.5 - edge_tol:
                continue
            gaps.append(float(tree.query(p[e])[0]) if tree is not None else math.inf)
    return gaps


def gap_histogram(gaps: list[float], scale: float = 1.0) -> dict[str, int]:
    h = {name: 0 for name, _, _ in GAP_CLASSES}
    for g in gaps:
        g *= scale
        for name, lo, hi in GAP_CLASSES:
            if (g <= hi if name == "<=1" else lo < g <= hi):
                h[name] += 1
                break
    return h
