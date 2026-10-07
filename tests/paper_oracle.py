"""ORACLE T-404a (Opsi A, jalur [5] lama): merender kertas bertekstur LANGSUNG di stage [5] dari level tinta mentah (tanpa PNG datar sebagai perantara).
Dipertahankan HANYA di tests/ untuk membuktikan bahwa susunan export (`paper.compose_textured`, invers LUT dari PNG datar) setara: lihat
`tests/test_paper.py` dan `scripts/t404a_measure.py oracle`. Kode produksi [5] tidak lagi merender kertas bertekstur (docs/04 "Hasil T-404a")."""

from __future__ import annotations

import cv2
import numpy as np

from rotoscope import stylize as sty
from rotoscope.paper import PaperLayer


def ink_levels(passes, g: sty.Geometry) -> tuple[np.ndarray, tuple[int, int, int, int] | None]:
    """Level tinta uint8 (jendela) + jendela, persis seperti [5]: legacy = cakupan INTER_AREA seluruh kanvas; selain itu f → 256 level di `ink_box`."""
    if g.legacy:
        return cv2.resize(sty.render_mask(passes[0], g), (g.out_w, g.out_h), interpolation=cv2.INTER_AREA), (0, 0, g.out_w, g.out_h)
    box = sty.ink_box(passes, g)
    if box is None:
        return np.zeros((0, 0), np.uint8), None
    f = sty.ink_fraction_box(passes, g, box)
    return np.rint(f * (sty.MASK_LEVELS - 1)).astype(np.uint8), box


def blend_ink(paper: PaperLayer, levels: np.ndarray, box, ink) -> np.ndarray:
    """RGB uint8 seluruh kanvas: di luar `box` = kertas (dibulatkan); di dalam = rint(kertas_f32 × (1 − a) + tinta × a), a = level / 255 (float32)."""
    x0, y0, x1, y1 = box
    out = paper.u8.copy()
    a = (levels.astype(np.float32) / np.float32(sty.MASK_LEVELS - 1))[..., None]
    out[y0:y1, x0:x1] = np.rint(paper.f32[y0:y1, x0:x1] * (np.float32(1.0) - a) + np.array(ink, np.float32) * a).astype(np.uint8)
    return out


def render_rgb(passes, g: sty.Geometry, paper: PaperLayer) -> np.ndarray:
    """Frame RGB uint8 jalur lama (Opsi A) untuk `paper`."""
    levels, box = ink_levels(passes, g)
    return paper.u8.copy() if box is None else blend_ink(paper, levels, box, g.ink)


def flat_rgb(passes, g: sty.Geometry) -> np.ndarray:
    """Frame RGB datar = decode PNG [5] (jalur LUT)."""
    return cv2.cvtColor(cv2.imdecode(np.frombuffer(sty.render_png_passes(passes, g), np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
