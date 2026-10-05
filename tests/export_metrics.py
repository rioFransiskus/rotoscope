"""Metrik objektif stage [6] export (T-203b): MP4 vs PNG sumber. Fungsi murni + konstanta ambang; dipakai oleh
tests/test_export_strokes.py (data sintetis + klip nyata, di-skip bila klip tidak ada) dan alat pratinjau Tahap 3.

Ambang — dasar: pengukuran Tahap 2 pada klip nyata test_short (26 frame) + test (59 frame), setiap 5 frame + pertama /
tengah / terakhir, setelan produksi (crf 18 + COLOR_TAGS) dibandingkan crf 23 (angka crf 18 → crf 23):
  PSNR seluruh frame   min 41,40 → 39,60   ⇒ PSNR_MIN_DB = 40,4 (minimum crf 18 − 1 dB; crf 23 gagal)
  tinta dilatasi 5 px  MAE maks 3,50 → 5,09 ⇒ INK_MAE_MAX = 4,2;  selisih maks 75 → 105 ⇒ INK_MAX_DIFF = 90
  kertas datar         selisih per kanal maks 2,00 (kuantisasi rentang tv, bukan matriks) ⇒ COLOR_TOLERANCE = 3
  tinta datar (nyata)  maks 3,71 (crf 23: 4,54) ⇒ INK_FLAT_TOLERANCE = 4,5 (garis tipis: sedikit area datar)
  warna jenuh sintetis (merah / hijau / biru, balok besar) ≤ COLOR_TOLERANCE: satu-satunya yang membedakan tag
  warna aktif / mati (palet netral kertas + tinta tidak diskriminatif).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

PSNR_MIN_DB = 40.4
INK_DILATE_PX = 5
INK_MAE_MAX = 4.2
INK_MAX_DIFF = 90
COLOR_TOLERANCE = 3
INK_FLAT_TOLERANCE = 4.5
PAPER_RGB = (0xF4, 0xF1, 0xEA)
INK_RGB = (0x1A, 0x1A, 0x1A)
PLAYER_MATRIX = "bt709"                # matriks yang diasumsikan pemutar untuk video HD


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_png_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def ffprobe_stream(mp4: Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-count_packets", "-print_format", "json", "-show_streams",
                          str(mp4)], capture_output=True, text=True, check=True).stdout
    return next(s for s in json.loads(out)["streams"] if s["codec_type"] == "video")


def decode_rgb(mp4: Path, index: int, matrix: str = PLAYER_MATRIX) -> np.ndarray:
    """Frame `index` sebagai RGB, didekode seperti pemutar yang mengasumsikan `matrix` (tv range → pc)."""
    st = ffprobe_stream(mp4)
    w, h = int(st["width"]), int(st["height"])
    vf = (f"select=eq(n\\,{index}),scale=in_color_matrix={matrix}:in_range=tv:out_color_matrix=bt709:out_range=pc,"
          f"format=rgb24")
    proc = subprocess.run(["ffmpeg", "-v", "error", "-i", str(mp4), "-vf", vf, "-frames:v", "1", "-f", "rawvideo",
                           "-pix_fmt", "rgb24", "-"], capture_output=True, check=True)
    return np.frombuffer(proc.stdout, np.uint8).reshape(h, w, 3)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse == 0 else 10 * float(np.log10(255.0 ** 2 / mse))


def ink_mask(ref: np.ndarray, dilate_px: int = INK_DILATE_PX) -> np.ndarray:
    """Piksel tinta / tepi garis (≠ kertas) didilatasi: wilayah tempat galat kompresi terkonsentrasi."""
    off_paper = np.any(ref.astype(int) != np.array(PAPER_RGB), axis=-1).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
    return cv2.dilate(off_paper, k).astype(bool)


def ink_metrics(ref: np.ndarray, dec: np.ndarray) -> dict:
    m = ink_mask(ref)
    if not m.any():
        return {"ink_px": 0, "mae": 0.0, "max_diff": 0}
    d = np.abs(ref.astype(int) - dec.astype(int))[m]
    return {"ink_px": int(m.sum()), "mae": float(d.mean()), "max_diff": int(d.max())}


def flat_color_delta(ref: np.ndarray, dec: np.ndarray, color: tuple[int, int, int], erode_px: int = 4) -> list[float] | None:
    """Selisih rata-rata kanal (dec − ref) pada area datar berwarna `color` (dierosi: tepi blok chroma tidak ikut)."""
    m = np.all(ref == np.array(color, np.uint8), axis=-1).astype(np.uint8)
    m = cv2.erode(m, np.ones((2 * erode_px + 1, 2 * erode_px + 1), np.uint8)).astype(bool)
    if not m.any():
        return None
    return [float(x) for x in (dec[m].astype(float).mean(0) - ref[m].astype(float).mean(0))]


def frame_report(ref: np.ndarray, dec: np.ndarray) -> dict:
    return {"psnr": psnr(ref, dec), **ink_metrics(ref, dec),
            "paper_delta": flat_color_delta(ref, dec, PAPER_RGB), "ink_delta": flat_color_delta(ref, dec, INK_RGB)}


def sample_indices(n: int, step: int = 10) -> list[int]:
    """Frame pertama, tengah, terakhir + setiap `step` frame."""
    return sorted({0, n // 2, n - 1, *range(0, n, step)})
