"""Oracle VF (T-305f): salinan APA ADANYA `vf_fill` dari prototipe `scripts/t305_1b.py` (aturan yang dinilai Rio di T-305 1b).

Dipakai `tests/test_stabilize_hole_fill.py` untuk membuktikan `stabilize.fill_group_holes` = prototipe pada peta acak. Jangan diubah:
perubahan aturan VF = keputusan baru (docs/04).
"""

from __future__ import annotations

import cv2
import numpy as np

BACKGROUND_ID = 0
K3 = np.ones((3, 3), np.uint8)


def vf_fill(g: np.ndarray, face_id: int) -> tuple[np.ndarray, int, int]:
    """Komponen background 4-arah yang tidak menyentuh tepi frame dan seluruh cincin 1 px-nya (8-arah) = face → face."""
    H, W = g.shape
    n, lab, st, _ = cv2.connectedComponentsWithStats((g == BACKGROUND_ID).astype(np.uint8), connectivity=4)
    out, regions, px = g.copy(), 0, 0
    for i in range(1, n):
        x, y, w, h, a = (int(v) for v in st[i])
        if x == 0 or y == 0 or x + w >= W or y + h >= H:
            continue
        sl = (slice(y - 1, y + h + 1), slice(x - 1, x + w + 1))
        comp = lab[sl] == i
        ring = cv2.dilate(comp.astype(np.uint8), K3).astype(bool) & ~comp
        if (g[sl][ring] == face_id).all():
            out[sl][comp] = face_id
            regions += 1
            px += a
    return out, regions, px
