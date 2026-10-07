"""Kertas bertekstur + vignette (T-404a, Opsi B: disusun saat EXPORT [6], BUKAN di [5]). CPU-only: torch tidak pernah di-import.

[5] menulis strokes/*.png di atas kertas DATAR `paper.color` (LUT 256 level warna dari `paper.color` ke `stroke.color`) dan strokes/*.svg
(kertas datar). Export (source strokes) memulihkan level tinta per piksel (`LevelMap`, invers LUT), menyusun kertas (`render_paper`) dan
mencampur ulang tinta di atas kertas itu (`compose_textured`) sebelum encode. Keputusan: docs/04 "Keputusan T-404" dan "Hasil T-404a".

Kertas = `paper.color` × ((1 − op) + op × t') × vignette; t = luminansi `paper.texture_image` (BT.601 pada sRGB, float32) → putar 90° berlawanan
jarum jam bila orientasi kanvas dan gambar berlawanan secara ketat (persegi tidak diputar) → potong "cover" di tengah TANPA resampling bila
gambar ≥ kanvas di kedua sumbu, selain itu skala seragam INTER_CUBIC (selalu memperbesar) → normalisasi rata-rata 1,0 (akumulasi float64) →
t' = max(1 + gain × (t − 1), PAPER_TEXTURE_FLOOR). Vignette v = 1 − `vignette` × d², d = jarak elips (dinormalisasi per sumbu, sudut = 1);
hanya mengenai kertas, tidak tinta. Statis antar frame; hasil = fungsi murni dari (ukuran, parameter, isi berkas tekstur).
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from rotoscope.config import ConfigError, StyleConfig, project_root
from rotoscope.stage_common import StageError
from rotoscope.stylize import MASK_LEVELS, ink_lut

PAPER_LUMA_BGR = (0.114, 0.587, 0.299)   # luminansi BT.601 pada nilai sRGB (urutan BGR cv2), float32
PAPER_TEXTURE_FLOOR = 0.05        # batas aman t' (> 0) setelah penguat gain: pengali kertas tidak pernah ≤ 0
PAPER_MIN_MEAN = 1.0              # rata-rata luminansi mentah (0–255) di bawah ini = gambar tekstur tidak layak (hampir hitam)
PAPER_CACHE_MAX = 4               # jumlah kertas (ukuran / parameter berbeda) yang disimpan di cache proses
PAPER_ROTATE_K = 1                # np.rot90 k: 90° berlawanan jarum jam
MAX_RGB = 255.0
KEY_SHIFT_R, KEY_SHIFT_G = 16, 8  # kunci warna = R << 16 | G << 8 | B


def hex_rgb(color: str) -> tuple[int, int, int]:
    c = color.lstrip("#")
    return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)


# ── Parameter → status ─────────────────────────────
def texture_active(style: StyleConfig) -> bool:
    """Tekstur dimuat / dipakai: paper.enabled dan texture_opacity > 0 dan texture_gain > 0."""
    p = style.paper
    return bool(p.enabled and p.texture_opacity > 0 and p.texture_gain > 0)


def paper_flat(style: StyleConfig) -> bool:
    """Kertas datar = PNG strokes dipakai apa adanya (MP4 byte-identik T-403): paper.enabled false, atau tekstur tidak aktif dan vignette 0."""
    p = style.paper
    return not (p.enabled and (texture_active(style) or p.vignette > 0))


def relative_to_root(path: Path) -> str:
    """Path aset relatif terhadap root project (posix) bila di dalamnya, selain itu apa adanya (hash tidak bergantung lokasi checkout)."""
    try:
        return path.resolve().relative_to(project_root().resolve()).as_posix()
    except (ValueError, ConfigError):
        return path.as_posix()


_FILE_SHA: dict = {}


def _file_sha256(path: Path) -> str:
    try:
        st = path.stat()
        key = (str(path), st.st_size, st.st_mtime_ns)
        if key not in _FILE_SHA:
            _FILE_SHA[key] = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as e:
        raise StageError(f"paper.texture_image {path} tidak bisa dibaca: {e}") from e
    return _FILE_SHA[key]


def texture_sha256(style: StyleConfig) -> str | None:
    return _file_sha256(style.paper.texture_image) if texture_active(style) else None


def paper_ref(style: StyleConfig) -> dict | None:
    """Parameter kertas yang memengaruhi isi MP4 (ikut hash / pencocokan basi export). None = kertas datar (tidak ada pengaruh)."""
    if paper_flat(style):
        return None
    p = style.paper
    tex = texture_active(style)
    return {"color": p.color, "texture_image": relative_to_root(p.texture_image) if tex else None, "texture_sha256": texture_sha256(style),
            "texture_opacity": float(p.texture_opacity) if tex else 0.0, "texture_gain": float(p.texture_gain) if tex else 1.0,
            "vignette": float(p.vignette)}


def validate_texture(style: StyleConfig) -> None:
    """Pre-flight: gambar tekstur ada + terdekode + tidak hampir hitam bila dipakai (StageError). Tidak melakukan apa pun pada kertas datar."""
    if texture_active(style):
        load_texture_luma(style.paper.texture_image)


# ── Tekstur ────────────────────────────────────────
def load_texture_luma(path: Path) -> np.ndarray:
    """Luminansi gambar tekstur (float32, 0–255, (h, w)). Gambar tidak terbaca / hampir hitam → StageError."""
    try:
        data = np.fromfile(path, np.uint8)
    except OSError as e:
        raise StageError(f"paper.texture_image {path} tidak bisa dibaca: {e}") from e
    im = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if im is None:
        raise StageError(f"paper.texture_image {path} bukan gambar yang bisa didekode")
    luma = im.astype(np.float32) @ np.array(PAPER_LUMA_BGR, np.float32)
    if float(luma.mean(dtype=np.float64)) < PAPER_MIN_MEAN:
        raise StageError(f"paper.texture_image {path} terlalu gelap (rata-rata luminansi < {PAPER_MIN_MEAN:g}): tidak bisa dinormalisasi")
    return luma


def orient_texture(tex: np.ndarray, w: int, h: int) -> tuple[np.ndarray, bool]:
    """Putar 90° berlawanan jarum jam bila kanvas portrait tetapi gambar landscape (atau sebaliknya); persegi tidak diputar."""
    ih, iw = tex.shape
    if (h > w and iw > ih) or (w > h and ih > iw):
        return np.ascontiguousarray(np.rot90(tex, PAPER_ROTATE_K)), True
    return tex, False


def fit_texture(tex: np.ndarray, w: int, h: int) -> tuple[np.ndarray, str, float]:
    """Potong "cover" di tengah (offset = (selisih) // 2): native bila gambar ≥ kanvas di kedua sumbu, selain itu diskalakan
    seragam (INTER_CUBIC, selalu memperbesar) lalu dipotong. → (potongan (h, w), "native" | "scaled", skala)."""
    ih, iw = tex.shape
    mode, s = "native", 1.0
    if iw < w or ih < h:
        s = max(w / iw, h / ih)
        tex = cv2.resize(tex, (max(w, math.ceil(iw * s)), max(h, math.ceil(ih * s))), interpolation=cv2.INTER_CUBIC)
        mode = "scaled"
        ih, iw = tex.shape
    x0, y0 = (iw - w) // 2, (ih - h) // 2
    return np.ascontiguousarray(tex[y0:y0 + h, x0:x0 + w]), mode, s


def normalize_texture(crop: np.ndarray) -> tuple[np.ndarray, float]:
    """Potongan luminansi → t dengan rata-rata 1,0 (akumulasi float64; rata-rata float32 pada jutaan elemen meleset sampai ~1,8%). → (t, rata-rata mentah)."""
    mean_raw = float(crop.mean(dtype=np.float64))
    return crop / np.float32(mean_raw), mean_raw


def texture_multiplier(t: np.ndarray, opacity: float, gain: float) -> tuple[np.ndarray, np.ndarray]:
    """t' = max(1 + gain × (t − 1), PAPER_TEXTURE_FLOOR); pengali kertas = (1 − op) + op × t'. → (pengali, t')."""
    tp = np.maximum(np.float32(1.0) + np.float32(gain) * (t - np.float32(1.0)), np.float32(PAPER_TEXTURE_FLOOR))
    return np.float32(1.0 - opacity) + np.float32(opacity) * tp, tp


def vignette_field(w: int, h: int, amount: float) -> np.ndarray:
    """Pengali vignette (float32, (h, w)): v = 1 − amount × d², d = √((x−cx)²/cx² + (y−cy)²/cy²) / √2 (jarak elips: sudut = 1,
    tengah sisi 0,707); pusat 1,0, simetris, monotonik."""
    cx, cy = (w - 1) / 2, (h - 1) / 2
    x = (np.arange(w, dtype=np.float32) - np.float32(cx)) / np.float32(cx)
    y = (np.arange(h, dtype=np.float32) - np.float32(cy)) / np.float32(cy)
    d2 = (x[None, :] ** 2 + y[:, None] ** 2) / np.float32(2.0)
    return (np.float32(1.0) - np.float32(amount) * np.minimum(d2, np.float32(1.0))).astype(np.float32)


# ── Kertas ─────────────────────────────────────────
@dataclass(frozen=True)
class PaperLayer:
    """Kertas statis satu ukuran kanvas: f32 = RGB float32 (h, w, 3), 0–255 (belum dibulatkan; dipakai pemulihan tinta di metrik),
    u8 = RGB dibulatkan (rint), info = ringkasan untuk manifest. Array read-only."""
    f32: np.ndarray
    u8: np.ndarray
    info: dict


_PAPER_CACHE: dict = {}


def render_paper(w: int, h: int, style: StyleConfig) -> PaperLayer:
    """Kertas ukuran (w, h) (px output) — fungsi MURNI dari (ukuran, parameter paper, isi berkas tekstur); di-cache per proses.
    `paper.enabled` false = kertas datar (parameter lain diabaikan)."""
    p = style.paper
    tex_on = texture_active(style)
    vig = float(p.vignette) if p.enabled else 0.0
    sha = texture_sha256(style)
    key = (w, h, p.color, bool(p.enabled), relative_to_root(p.texture_image) if tex_on else None, sha,
           float(p.texture_opacity) if tex_on else 0.0, float(p.texture_gain) if tex_on else 1.0, vig)
    if key in _PAPER_CACHE:
        return _PAPER_CACHE[key]
    info: dict = {"enabled": bool(p.enabled), "size": {"width": w, "height": h}, "texture_active": tex_on, "vignette": vig,
                  "texture_sha256": sha}
    mult = np.ones((h, w), np.float32)
    if tex_on:
        tex = load_texture_luma(p.texture_image)
        info["image_size"] = {"width": int(tex.shape[1]), "height": int(tex.shape[0])}
        tex, rotated = orient_texture(tex, w, h)
        crop, mode, scale = fit_texture(tex, w, h)
        t, mean_raw = normalize_texture(crop)
        mult, tp = texture_multiplier(t, p.texture_opacity, p.texture_gain)
        info.update({"rotated": rotated, "fit": mode, "scale": round(scale, 6), "texture_mean_raw": round(mean_raw, 4),
                     "texture_std": round(float(t.std(dtype=np.float64)), 6), "opacity": float(p.texture_opacity),
                     "gain": float(p.texture_gain), "clipped_fraction": float((tp <= PAPER_TEXTURE_FLOOR).mean(dtype=np.float64))})
    if vig > 0:
        mult = mult * vignette_field(w, h, vig)
    f32 = np.clip(mult[..., None] * np.array(hex_rgb(p.color), np.float32)[None, None, :], 0.0, MAX_RGB).astype(np.float32)
    u8 = np.rint(f32).astype(np.uint8)
    info["mean_rgb"] = [round(float(x), 4) for x in u8.reshape(-1, 3).mean(axis=0, dtype=np.float64)]
    f32.setflags(write=False)
    u8.setflags(write=False)
    layer = PaperLayer(f32=f32, u8=u8, info=info)
    if len(_PAPER_CACHE) >= PAPER_CACHE_MAX:
        _PAPER_CACHE.pop(next(iter(_PAPER_CACHE)))
    _PAPER_CACHE[key] = layer
    return layer


def paper_layer(style: StyleConfig, w: int, h: int) -> PaperLayer | None:
    """None = kertas datar (PNG strokes dipakai apa adanya); selain itu kertas bertekstur / bervignette ukuran output."""
    return None if paper_flat(style) else render_paper(w, h, style)


def paper_info(style: StyleConfig, w: int, h: int) -> dict:
    """Blok informasi `paper` manifest export (tidak ikut pencocokan basi; yang ikut = `paper_ref`)."""
    if paper_flat(style):
        return {"enabled": bool(style.paper.enabled), "flat": True, "color": style.paper.color}
    return {"flat": False, "color": style.paper.color, **render_paper(w, h, style).info}


# ── Pemulihan level tinta + susun ──────────────────
def _pack(rgb: np.ndarray) -> np.ndarray:
    a = rgb.astype(np.uint32)
    return (a[..., 0] << KEY_SHIFT_R) | (a[..., 1] << KEY_SHIFT_G) | a[..., 2]


class LevelMap:
    """Invers LUT [5]: PNG strokes datar → level tinta per piksel. LUT = `stylize.ink_lut` (256 level, `paper.color` → `stroke.color`).
    ATURAN TABRAKAN (tetap): level bertetangga yang menghasilkan warna sama (kontras < 256 level) → level terpulihkan = RATA-RATA level
    yang berbagi warna itu (pecahan; galat ≤ 0,5 level); warna kertas → level 0 (kertas datar mendominasi; level 1 yang bertabrakan dengan
    kertas ikut dipulihkan sebagai 0). Warna di luar LUT = StageError (PNG tidak dibuat dengan paper.color / stroke.color ini)."""

    def __init__(self, paper_rgb: tuple[int, int, int], ink_rgb: tuple[int, int, int]):
        lut = ink_lut(SimpleNamespace(paper=paper_rgb, ink=ink_rgb))
        keys = _pack(lut)
        self.keys, inv = np.unique(keys, return_inverse=True)
        self.levels = (np.bincount(inv, weights=np.arange(MASK_LEVELS)) / np.bincount(inv)).astype(np.float32)
        self.paper_key = np.uint32(_pack(np.array(paper_rgb, np.uint8)[None])[0])
        self.n_collisions = int(MASK_LEVELS - len(self.keys))

    def levels_of(self, img: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int] | None]:
        """(level float32 (h, w); 0 di kertas, bbox (x0, y0, x1, y1) piksel bukan-kertas atau None bila hanya kertas)."""
        k = _pack(img)
        mask = k != self.paper_key
        out = np.zeros(k.shape, np.float32)
        if not mask.any():
            return out, None
        kk = k[mask]
        idx = np.searchsorted(self.keys, kk)
        idx[idx == len(self.keys)] = len(self.keys) - 1
        bad = self.keys[idx] != kk
        if bad.any():
            c = img[mask][bad][0]
            raise StageError(f"{int(bad.sum())} piksel PNG strokes berwarna di luar LUT paper.color / stroke.color (mis. RGB {tuple(int(x) for x in c)}) — "
                             f"strokes/ tidak dibuat dengan warna ini; jalankan ulang stage [5] atau samakan paper.color / stroke.color")
        out[mask] = self.levels[idx]
        ys, xs = np.nonzero(mask.any(axis=1))[0], np.nonzero(mask.any(axis=0))[0]
        return out, (int(xs[0]), int(ys[0]), int(xs[-1]) + 1, int(ys[-1]) + 1)


def compose_textured(img_rgb: np.ndarray, levels: LevelMap, layer: PaperLayer, ink_rgb: tuple[int, int, int]) -> np.ndarray:
    """PNG strokes datar (RGB uint8) → RGB uint8 di atas kertas bertekstur / bervignette: di luar jendela tinta = kertas (dibulatkan); di dalam =
    rint(kertas_f32 × (1 − a) + tinta × a), a = level / 255 (float32)."""
    if img_rgb.shape[:2] != layer.u8.shape[:2]:
        raise StageError(f"PNG strokes {img_rgb.shape[1]}x{img_rgb.shape[0]} ≠ kertas {layer.u8.shape[1]}x{layer.u8.shape[0]}")
    lv, box = levels.levels_of(img_rgb)
    out = layer.u8.copy()
    if box is not None:
        x0, y0, x1, y1 = box
        a = (lv[y0:y1, x0:x1] / np.float32(MASK_LEVELS - 1))[..., None]
        out[y0:y1, x0:x1] = np.rint(layer.f32[y0:y1, x0:x1] * (np.float32(1.0) - a) + np.array(ink_rgb, np.float32) * a).astype(np.uint8)
    return out
