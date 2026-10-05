"""Metrik objektif T-204 (bukan test sendiri; dipakai test_preview dan scripts/t204_measure.py).

Definisi (sama dengan T-203b): PSNR dihitung pada frame RGB penuh terhadap PNG sumber (strokes/*.png); "tinta MAE" =
rata-rata |selisih| luminans pada piksel tinta (luminans PNG sumber < INK_LUMA)."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

PSNR_MIN_DB = 40.4          # ambang T-203b
INK_MAE_MAX = 4.2           # ambang T-203b
INK_LUMA = 200


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_hashes(root: Path, pattern: str = "**/*") -> dict[str, str]:
    """{path relatif: sha256} untuk semua berkas di bawah root (kosong bila root tidak ada)."""
    if not root.exists():
        return {}
    return {p.relative_to(root).as_posix(): sha256_file(p) for p in sorted(root.glob(pattern)) if p.is_file()}


def window_hashes(work_dir: Path, first: int, last: int) -> dict[str, str]:
    """sha256 berkas jendela: contours/frame_*.json dan strokes/frame_*.{svg,png}, frame first..last inklusif."""
    out = {}
    for i in range(first, last + 1):
        for rel in (f"contours/frame_{i:05d}.json", f"strokes/frame_{i:05d}.svg", f"strokes/frame_{i:05d}.png"):
            p = work_dir / rel
            out[rel] = sha256_file(p) if p.is_file() else "<tidak ada>"
    return out


def out_snapshot(out_dir: Path, stem: str) -> dict[str, object]:
    """Keadaan keluaran utama yang TIDAK boleh disentuh preview: <stem>.mp4, <stem>.export.json, out/svg/<stem>/."""
    files = {f"{stem}.mp4": out_dir / f"{stem}.mp4", f"{stem}.export.json": out_dir / f"{stem}.export.json"}
    snap: dict[str, object] = {k: (sha256_file(p) if p.is_file() else None) for k, p in files.items()}
    snap["svg"] = tree_hashes(out_dir / "svg" / stem)
    return snap


def decode_mp4(path: Path, width: int, height: int, first: int = 0, count: int | None = None) -> np.ndarray:
    """Frame RGB uint8 (n, h, w, 3) lewat ffmpeg (pipe rawvideo); `first`/`count` memilih potongan."""
    vf = f"select='gte(n\\,{first})'" if first else "null"
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-vf", vf, "-fps_mode", "passthrough", "-pix_fmt", "rgb24"]
    if count is not None:
        cmd += ["-frames:v", str(count)]
    cmd += ["-f", "rawvideo", "-"]
    raw = subprocess.run(cmd, check=True, capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, height, width, 3)


def read_png_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else 10.0 * np.log10(255.0 ** 2 / mse)


def ink_mae(decoded: np.ndarray, source: np.ndarray) -> float:
    luma = lambda x: x.astype(np.float64) @ np.array([0.299, 0.587, 0.114])      # noqa: E731
    ls, ld = luma(source), luma(decoded)
    mask = ls < INK_LUMA
    return float(np.abs(ld - ls)[mask].mean()) if mask.any() else 0.0


def compare_to_sources(mp4: Path, strokes_dir: Path, first: int, n: int) -> dict:
    """Bandingkan N frame MP4 dengan strokes/frame_<first..>.png: PSNR min dan tinta MAE maks."""
    src0 = read_png_rgb(strokes_dir / f"frame_{first:05d}.png")
    h, w = src0.shape[:2]
    dec = decode_mp4(mp4, w, h)
    psnrs, maes = [], []
    for k in range(n):
        s = read_png_rgb(strokes_dir / f"frame_{first + k:05d}.png")
        psnrs.append(psnr(dec[k], s))
        maes.append(ink_mae(dec[k], s))
    return {"n_decoded": int(dec.shape[0]), "psnr_min": min(psnrs), "ink_mae_max": max(maes),
            "ok": dec.shape[0] == n and min(psnrs) >= PSNR_MIN_DB and max(maes) <= INK_MAE_MAX}


def compare_preview_to_main(preview: Path, main: Path, first: int, n: int, width: int, height: int) -> dict:
    """Frame preview = potongan MP4 utama pada frame yang sama (PSNR antar-keduanya; encode terpisah → bukan byte-identik)."""
    p = decode_mp4(preview, width, height)
    m = decode_mp4(main, width, height, first=first, count=n)
    vals = [psnr(p[k], m[k]) for k in range(n)]
    return {"n_preview": int(p.shape[0]), "n_main": int(m.shape[0]), "psnr_min": min(vals)}


def ffprobe_stream(path: Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                          "stream=codec_name,pix_fmt,width,height,r_frame_rate,nb_read_frames,color_space,"
                          "color_transfer,color_primaries,color_range", "-of", "json", str(path)],
                         check=True, capture_output=True, text=True).stdout
    return json.loads(out)["streams"][0]
