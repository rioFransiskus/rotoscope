"""Smoke test T-003: cek environment, lisensi, ffmpeg, OpenCV, dan benchmark segmentasi MediaPipe (D-008).

rembg + onnxruntime sudah dihapus dari stack (D-008, T-107): cek import/lisensi/benchmark-nya dibuang.

Jalankan dari root repo:
    venv/Scripts/python.exe scripts/smoke_test.py [--image samples/person.jpg] [--runs 10]

Exit code 1 kalau ada cek yang FAIL.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata as md
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MP_MODEL = ROOT / "models" / "selfie_multiclass_256x256.tflite"

# Proyeksi: klip 15 detik x 24 fps
PROJ_SECONDS = 15
PROJ_FPS = 24
PROJ_FRAMES = PROJ_SECONDS * PROJ_FPS

SYNTH_W, SYNTH_H = 720, 1280  # ukuran gambar sintetis (portrait) kalau --image tidak ada
MASK_THRESHOLD = 0.5  # piksel dihitung "orang" kalau confidence > nilai ini

# (nama import, nama distribusi pip untuk metadata)
PACKAGES = [
    ("cv2", "opencv-contrib-python"),
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("svgwrite", "svgwrite"),
    ("PIL", "Pillow"),
    ("yaml", "PyYAML"),
    ("mediapipe", "mediapipe"),
]
LICENSE_RED_FLAGS = ["AGPL", "GPL", "NON-COMMERCIAL", "NONCOMMERCIAL", "NON COMMERCIAL", "CC-BY-NC"]
# --all-licenses: juga copyleft lemah (LGPL, MPL). Classifier memakai nama panjang
# ("GNU General Public License", "Mozilla Public License"), jadi dicek juga.
ALL_LICENSE_RED_FLAGS = LICENSE_RED_FLAGS + [
    "LGPL", "MPL", "GENERAL PUBLIC LICENSE", "MOZILLA PUBLIC",
]

results: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, note: str = "") -> None:
    results.append((name, ok, note))


def header(title: str) -> None:
    print(f"\n=== {title} ===")


# --- a. Python / venv ------------------------------------------------------
def check_python() -> None:
    header("a. Python")
    in_venv = sys.prefix != sys.base_prefix
    print(f"Python  : {sys.version.split()[0]}")
    print(f"exe     : {sys.executable}")
    print(f"in venv : {in_venv}")
    record("python venv", in_venv, sys.version.split()[0])


# --- b. Import + versi -----------------------------------------------------
def check_imports() -> None:
    header("b. Import + versi")
    for mod_name, dist in PACKAGES:
        try:
            mod = importlib.import_module(mod_name)
            ver = getattr(mod, "__version__", None) or md.version(dist)
            print(f"  {mod_name:<12} {ver}")
            record(f"import {mod_name}", True, str(ver))
        except Exception as e:  # noqa: BLE001
            print(f"  {mod_name:<12} GAGAL: {e}")
            record(f"import {mod_name}", False, str(e))


# --- c. Lisensi --------------------------------------------------------------
def license_text(dist: str) -> str:
    meta = md.metadata(dist)
    parts = []
    if meta.get("License-Expression"):
        parts.append(meta["License-Expression"])
    lic = meta.get("License")
    if lic:
        # Beberapa paket menaruh seluruh teks lisensi di field ini; ambil baris pertama saja
        parts.append(lic.strip().splitlines()[0][:80])
    for c in meta.get_all("Classifier") or []:
        if c.startswith("License ::"):
            parts.append(c.replace("License :: ", ""))
    return " | ".join(dict.fromkeys(parts)) or "(tidak ada metadata lisensi)"


def check_licenses() -> None:
    header("c. Lisensi (importlib.metadata)")
    any_warn = False
    for _, dist in PACKAGES:
        try:
            text = license_text(dist)
        except md.PackageNotFoundError:
            print(f"  {dist:<22} TIDAK TERINSTALL")
            record(f"license {dist}", False, "not installed")
            continue
        upper = text.upper()
        # "LGPL" juga mengandung "GPL" -> tetap ditandai supaya direview manual
        flag = any(f in upper for f in LICENSE_RED_FLAGS)
        any_warn |= flag
        print(f"  {'WARNING ' if flag else ''}{dist:<22} {text}")
    record("license scan", True, "ada WARNING, review manual" if any_warn else "tidak ada red flag")


def check_all_licenses() -> None:
    header("c2. Lisensi SEMUA paket terinstall (--all-licenses)")
    dists = sorted(md.distributions(), key=lambda d: d.metadata["Name"].lower())
    flagged = []
    for d in dists:
        name = d.metadata["Name"]
        text = license_text(name)
        upper = text.upper()
        hits = [f for f in ALL_LICENSE_RED_FLAGS if f in upper]
        if hits:
            flagged.append(name)
            print(f"  WARNING {name}=={d.version:<12} {text}")
    print(f"  {len(dists)} paket dipindai, {len(flagged)} ditandai (review manual, belum tentu dilarang)")
    record("license scan (all)", True, f"{len(flagged)} ditandai" if flagged else "tidak ada red flag")


# --- d. ffmpeg -------------------------------------------------------------
def check_ffmpeg() -> None:
    header("d. ffmpeg")
    try:
        out = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=15)
        first = (out.stdout or out.stderr).splitlines()[0] if (out.stdout or out.stderr) else ""
        print(f"  {first}")
        record("ffmpeg", out.returncode == 0, first[:50])
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f"  GAGAL: {e}")
        record("ffmpeg", False, str(e))


# --- e. OpenCV sanity ------------------------------------------------------
def check_opencv() -> None:
    header("e. OpenCV sanity")
    import cv2
    import numpy as np

    try:
        img = np.zeros((64, 64), np.uint8)
        cv2.circle(img, (32, 32), 15, 255, -1)
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        ok = len(contours) == 1 and len(contours[0]) > 10
        print(f"  findContours: {len(contours)} kontur, {len(contours[0])} titik")
        record("cv2.findContours", ok)
    except Exception as e:  # noqa: BLE001
        print(f"  findContours GAGAL: {e}")
        record("cv2.findContours", False, str(e))

    try:
        prev = np.zeros((64, 64), np.uint8)
        cv2.rectangle(prev, (20, 20), (40, 40), 255, -1)
        nxt = np.roll(prev, 2, axis=1)  # geser 2 px ke kanan
        flow = cv2.calcOpticalFlowFarneback(prev, nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        dx = float(flow[20:41, 20:41, 0].mean())
        print(f"  calcOpticalFlowFarneback: shape={flow.shape}, rata-rata dx di kotak={dx:.2f} (harapan ~2)")
        record("cv2.calcOpticalFlowFarneback", flow.shape == (64, 64, 2) and dx > 0)
    except Exception as e:  # noqa: BLE001
        print(f"  calcOpticalFlowFarneback GAGAL: {e}")
        record("cv2.calcOpticalFlowFarneback", False, str(e))


# --- f. Benchmark segmentasi ----------------------------------------------
def load_image(path: Path):
    """Return (RGB uint8 array, is_real)."""
    import numpy as np
    from PIL import Image

    if path.is_file():
        return np.asarray(Image.open(path).convert("RGB")), True
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (SYNTH_H, SYNTH_W, 3), dtype=np.uint8), False


def bench(fn, runs: int) -> tuple[float, float, object]:
    out = fn()  # warm-up
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        out = fn()
        times.append((time.perf_counter() - t0) * 1000)
    return sum(times) / len(times), min(times), out


def report_bench(name: str, avg: float, mn: float, mask, is_real: bool) -> None:
    import numpy as np

    proj = avg * PROJ_FRAMES / 1000
    print(f"  {name}: avg {avg:.1f} ms/frame, min {mn:.1f} ms/frame")
    print(f"    proyeksi {PROJ_FRAMES} frame ({PROJ_SECONDS}s x {PROJ_FPS}fps): {proj:.1f} s")
    if is_real:
        area = float((np.asarray(mask) > MASK_THRESHOLD).mean() * 100)
        print(f"    area mask: {area:.1f}% dari frame")


def bench_mediapipe(img, is_real: bool, runs: int) -> None:
    import mediapipe as mp
    import numpy as np
    from mediapipe.tasks.python import BaseOptions, vision

    try:
        if not MP_MODEL.is_file():
            raise FileNotFoundError(MP_MODEL)
        options = vision.ImageSegmenterOptions(
            base_options=BaseOptions(model_asset_path=str(MP_MODEL)),
            output_confidence_masks=True,
        )
        with vision.ImageSegmenter.create_from_options(options) as segmenter:
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(img))

            def run():
                res = segmenter.segment(mp_img)
                # kategori 0 = background -> mask orang = 1 - confidence background
                return 1.0 - res.confidence_masks[0].numpy_view()

            avg, mn, mask = bench(run, runs)
        report_bench("MediaPipe selfie_multiclass_256x256", avg, mn, mask, is_real)
        record("bench mediapipe selfie_multiclass", True, f"{avg:.0f} ms/frame")
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        record("bench mediapipe selfie_multiclass", False, str(e)[:60])


def check_segmentation(image: Path, runs: int) -> None:
    header("f. Benchmark segmentasi (D-008)")
    img, is_real = load_image(image)
    h, w = img.shape[:2]
    if is_real:
        print(f"  gambar: {image} ({w}x{h})")
    else:
        print(f"  {image} tidak ada -> gambar sintetis {w}x{h}: TIMING SAJA, BUKAN UJI KUALITAS")
    print(f"  1x warm-up + {runs} run per model (CPU)")
    bench_mediapipe(img, is_real, runs)


# --- g. Ringkasan ----------------------------------------------------------
def summary() -> int:
    header("g. Ringkasan")
    width = max(len(n) for n, _, _ in results)
    for name, ok, note in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<{width}}  {note}")
    n_fail = sum(1 for _, ok, _ in results if not ok)
    print(f"\n  {len(results) - n_fail} PASS, {n_fail} FAIL")
    return 1 if n_fail else 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--image", type=Path, default=ROOT / "samples" / "person.jpg")
    p.add_argument("--runs", type=int, default=10)
    p.add_argument("--all-licenses", action="store_true",
                   help="pindai lisensi semua paket terinstall, tampilkan yang ditandai")
    args = p.parse_args()

    check_python()
    check_imports()
    check_licenses()
    if args.all_licenses:
        check_all_licenses()
    check_ffmpeg()
    check_opencv()
    check_segmentation(args.image, max(1, args.runs))
    return summary()


if __name__ == "__main__":
    sys.exit(main())
