"""T-606 (alat sekali pakai, F3): inti algoritma H (hibrida) — dipakai bersama `t606_tahap1.py` (ukur, M2-H, bahan) dan
`t606_m1.py`. Modul ini TIDAK menulis berkas apa pun; pemanggil yang menulis (hanya ke work/t606_scratch/ dan work/t606/).

Algoritma H per frame (docs/04 "Rancangan T-606 Tahap 1"; keputusan Rio T-606 no. 1, 3, 5):
  1. F = foreground seg MENTAH frame ini = argmax jumlah probabilitas grup ≠ 0 (langkah 1 + 3 stage [3] tanpa temporal /
     pulau / mode: `stabilize.group_sums` + `stabilize.group_argmax`, seri → grup classmap).
  2. M = peta label [1b] ≠ 0 (semua target). M_f = M + lubang terisi (lubang = komponen 4-arah bukan-M yang tidak menyentuh
     tepi gambar; `hole_max_px` None = semua lubang). M_d = M_f dilebarkan d px (elips (2d + 1) × (2d + 1)).
  3. Komponen 8-arah C dari F; r = |C ∩ M_f| / |C|:
       r ≥ r_keep → C dipertahankan UTUH;
       r ≤ r_drop → C dibuang UTUH;
       di antaranya → C ∩ M_d, dipecah menjadi potongan 8-arah P; P dipertahankan hanya bila |P ∩ M_f| ≥ piece_min_px
       (potongan kecil di cincin dilasi / sisa orang lain dibuang).
  4. R = F − yang dipertahankan. Piksel R → Background one-hot SEBELUM kernel temporal: probabilitas kelas = (255, 0, …, 0)
     (jumlah grup = (255, 0, …)), classmap = 0. Piksel di luar F TIDAK diubah (argmax-nya sudah background di frame ini).
  Frame `kosong` (M kosong): semua r = 0 ≤ r_drop → semua dibuang = tidak menggambar apa pun (keputusan no. 5).
"""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rotoscope import stabilize as stb  # noqa: E402

BACKGROUND_CLASS = 0
PROBS_ONE = 255


@dataclass(frozen=True)
class HParams:
    r_keep: float
    r_drop: float
    d_px: int
    hole_max_px: int | None = None      # None = semua lubang masker diisi
    piece_min_px: int = 1               # potongan C ∩ M_d dengan |P ∩ M_f| < ini dibuang

    def to_json(self) -> dict:
        return asdict(self)


def raw_groups(probs: np.ndarray, classmap: np.ndarray, lut: np.ndarray, n_groups: int) -> np.ndarray:
    """Peta grup mentah frame (langkah 1 + 3 stage [3], tanpa temporal / filter)."""
    sums = stb.group_sums(probs, lut, n_groups)
    return stb.group_argmax(sums, lut[classmap])[0]


def fill_holes(m: np.ndarray, hole_max_px: int | None) -> tuple[np.ndarray, list[int], np.ndarray]:
    """(M_f, luas tiap lubang, peta label lubang int32 (0 = bukan lubang)). Lubang = komponen 4-arah ~M yang tidak menyentuh
    tepi gambar (pasangan konektivitas 8-arah M / 4-arah latar)."""
    n, lab, st, _ = cv2.connectedComponentsWithStats((~m).astype(np.uint8), connectivity=4)
    h, w = m.shape
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])).tolist())
    holes = [i for i in range(1, n) if i not in border]
    areas = [int(st[i, cv2.CC_STAT_AREA]) for i in holes]
    hole_lab = np.zeros((h, w), np.int32)
    out = m.copy()
    for k, (i, a) in enumerate(zip(holes, areas), 1):
        sel = lab == i
        hole_lab[sel] = k
        if hole_max_px is None or a <= hole_max_px:
            out |= sel
    return out, areas, hole_lab


def dilate(m: np.ndarray, d: int) -> np.ndarray:
    if d <= 0:
        return m.copy()
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * d + 1, 2 * d + 1))
    return cv2.dilate(m.astype(np.uint8), ker) > 0


def classify(r: float, p: HParams) -> str:
    if r >= p.r_keep:
        return "utuh"
    if r <= p.r_drop:
        return "buang"
    return "potong"


def apply_h(fg: np.ndarray, mask: np.ndarray, p: HParams) -> dict:
    """F (bool) + M (bool) → {"keep": piksel F yang dipertahankan, "remove": R, "m_f", "m_d", "comps", "pieces", "holes"}."""
    m_f, hole_areas, _ = fill_holes(mask, p.hole_max_px)
    m_d = dilate(m_f, p.d_px)
    n, lab, st, _ = cv2.connectedComponentsWithStats(fg.astype(np.uint8), connectivity=8)
    inter = np.bincount(lab[m_f], minlength=n)
    keep = np.zeros(fg.shape, bool)
    comps, pieces = [], []
    for i in range(1, n):
        area = int(st[i, cv2.CC_STAT_AREA])
        r = float(inter[i]) / area
        cls = classify(r, p)
        x, y, bw, bh = (int(v) for v in st[i, :4])
        rec = {"label": i, "area": area, "inter_mf": int(inter[i]), "r": round(r, 4), "class": cls, "bbox": [x, y, bw, bh]}
        sl = (slice(y, y + bh), slice(x, x + bw))
        c = lab[sl] == i
        if cls == "utuh":
            keep[sl] |= c
            rec["kept_px"] = area
        elif cls == "potong":
            cp = c & m_d[sl]
            pn, plab, pst, _ = cv2.connectedComponentsWithStats(cp.astype(np.uint8), connectivity=8)
            pin = np.bincount(plab[m_f[sl] & cp], minlength=pn)
            kept = 0
            for j in range(1, pn):
                ok = int(pin[j]) >= p.piece_min_px
                pieces.append({"comp": i, "area": int(pst[j, cv2.CC_STAT_AREA]), "inter_mf": int(pin[j]), "kept": ok})
                if ok:
                    keep[sl] |= plab == j
                    kept += int(pst[j, cv2.CC_STAT_AREA])
            rec["kept_px"] = kept
        else:
            rec["kept_px"] = 0
        comps.append(rec)
    return {"keep": keep, "remove": fg & ~keep, "m_f": m_f, "m_d": m_d, "comps": comps, "pieces": pieces,
            "holes": hole_areas}


def mask_probs(probs: np.ndarray, classmap: np.ndarray, remove: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Piksel `remove` → Background one-hot (probabilitas kelas 255 / 0, classmap 0). Salinan; input tidak diubah."""
    pr, cm = probs.copy(), classmap.copy()
    pr[:, remove] = 0
    pr[BACKGROUND_CLASS, remove] = PROBS_ONE
    cm[remove] = BACKGROUND_CLASS
    return pr, cm
