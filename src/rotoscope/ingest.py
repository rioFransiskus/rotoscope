"""Stage [1] ingest: video -> work/frames/frame_%05d.png + work/meta.json.

Decode, konversi fps, rotasi, dan resize dikerjakan ffmpeg via subprocess.
Filter `fps` ffmpeg bekerja berdasarkan timestamp, jadi aman untuk video VFR
(variable frame rate, umum di rekaman HP). Metadata dibaca dari ffprobe (JSON),
bukan CAP_PROP_* OpenCV yang tidak andal untuk VFR.

Frame index mulai 0 (frame_00000.png = index 0). Index ini dipakai sebagai seed
jitter di stage berikutnya (P-007), jadi harus konsisten.

meta.json ditulis PALING AKHIR sebagai penanda sukses: kalau ada meta.json,
frame di folder yang sama lengkap dan tervalidasi.

CLI final (T-104b): python -m rotoscope ingest <video> [--target-fps N] [--working-width N] (folder kerja =
<paths.work_dir>/clips/<nama video>/); atau seluruh pipeline: python -m rotoscope run <video>. Entry point stage:
    python -m rotoscope.ingest <video> [--work-dir work] [--target-fps N] [--working-width N]
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path

from PIL import Image

DEFAULT_TARGET_FPS = 24
DEFAULT_WORKING_WIDTH = 720
DEFAULT_WORK_DIR = "work"

FRAMES_DIRNAME = "frames"
FRAME_PATTERN = "frame_%05d.png"
FRAME_GLOB = "frame_*.png"
META_FILENAME = "meta.json"
FRAME_INDEX_START = 0

# Metode resampling ffmpeg; "area" menghasilkan downscale yang halus tanpa aliasing.
SCALE_FLAGS = "area"
# Toleransi tinggi output vs hitungan Python (pembulatan genap ffmpeg bisa beda sedikit).
HEIGHT_TOLERANCE_PX = 2
STDERR_TAIL_LINES = 20


class IngestError(RuntimeError):
    """Ingest gagal: binary hilang, video tidak valid, atau ffmpeg error."""


def _require_binary(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise IngestError(
            f"{name} tidak ditemukan di PATH. Install ffmpeg (gyan.dev essentials build) "
            f"dan pastikan folder bin-nya ada di PATH."
        )
    return path


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )


def _stderr_tail(stderr: str | None) -> str:
    lines = (stderr or "").strip().splitlines()
    return "\n".join(lines[-STDERR_TAIL_LINES:])


def _parse_rate(value: str | None) -> float | None:
    """'30000/1001' -> 29.97. None kalau kosong atau 0/0."""
    if not value:
        return None
    try:
        rate = Fraction(value)
    except (ValueError, ZeroDivisionError):
        return None
    return float(rate) if rate > 0 else None


def _parse_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _stream_rotation(stream: dict) -> int:
    """Rotasi display (derajat, dinormalisasi ke 0/90/180/270) dari metadata stream."""
    rotation = None
    for side_data in stream.get("side_data_list") or []:
        if "rotation" in side_data:
            rotation = _parse_float(side_data["rotation"])
            break
    if rotation is None:  # container lama menyimpannya sebagai tag
        rotation = _parse_float((stream.get("tags") or {}).get("rotate"))
    if rotation is None:
        return 0
    return int(round(rotation)) % 360


def probe_video(video_path: str | Path) -> dict:
    """Baca metadata video via ffprobe.

    Return dict: fps, duration_s, frame_count (None kalau tidak tersedia),
    width, height (SETELAH rotasi), rotation, has_audio.
    """
    video_path = Path(video_path)
    ffprobe = _require_binary("ffprobe")
    proc = _run([
        ffprobe, "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", str(video_path),
    ])
    if proc.returncode != 0:
        raise IngestError(
            f"ffprobe gagal membaca {video_path} (exit {proc.returncode}):\n"
            f"{_stderr_tail(proc.stderr)}"
        )
    try:
        info = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise IngestError(f"output ffprobe bukan JSON valid: {exc}") from exc

    streams = info.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise IngestError(f"tidak ada stream video di {video_path}")

    width = _parse_int(video.get("width"))
    height = _parse_int(video.get("height"))
    if not width or not height:
        raise IngestError(f"dimensi stream video tidak terbaca di {video_path}")

    rotation = _stream_rotation(video)
    if rotation in (90, 270):
        width, height = height, width

    fps = _parse_rate(video.get("avg_frame_rate")) or _parse_rate(video.get("r_frame_rate"))
    duration = _parse_float(video.get("duration"))
    if duration is None:
        duration = _parse_float((info.get("format") or {}).get("duration"))

    return {
        "fps": fps,
        "duration_s": duration,
        "frame_count": _parse_int(video.get("nb_frames")),
        "width": width,
        "height": height,
        "rotation": rotation,
        "has_audio": any(s.get("codec_type") == "audio" for s in streams),
    }


def _round_even(value: float) -> int:
    # half-up (bukan round() Python yang half-to-even)
    return max(2, int(value / 2 + 0.5) * 2)


def compute_output_size(src_w: int, src_h: int, working_width: int) -> tuple[int, int]:
    """Ukuran output: proporsional, tanpa upscale, lebar & tinggi genap.

    Genap diwajibkan libx264/yuv420p di stage export.
    """
    if src_w <= 0 or src_h <= 0:
        raise ValueError(f"dimensi sumber tidak valid: {src_w}x{src_h}")
    if working_width < 2:
        raise ValueError(f"working_width minimal 2, dapat {working_width}")
    width = min(src_w, working_width)
    width = max(2, width - width % 2)  # bulatkan ke bawah: tidak boleh melebihi sumber
    height = _round_even(src_h * width / src_w)
    return width, height


def _clear_previous_output(frames_dir: Path, meta_path: Path) -> None:
    meta_path.unlink(missing_ok=True)
    for old in frames_dir.glob(FRAME_GLOB):
        old.unlink()


def _write_json_atomic(path: Path, data: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def ingest(
    video_path: str | Path,
    work_dir: str | Path = DEFAULT_WORK_DIR,
    target_fps: float = DEFAULT_TARGET_FPS,
    working_width: int = DEFAULT_WORKING_WIDTH,
) -> dict:
    """Ekstrak frame video ke <work_dir>/frames/ dan tulis <work_dir>/meta.json.

    Return isi meta.json.
    """
    video_path = Path(video_path)
    work_dir = Path(work_dir)
    if not video_path.is_file():
        raise FileNotFoundError(f"file video tidak ditemukan: {video_path}")
    if target_fps <= 0:
        raise ValueError(f"target_fps harus > 0, dapat {target_fps}")

    ffmpeg = _require_binary("ffmpeg")
    src = probe_video(video_path)
    out_w, out_h = compute_output_size(src["width"], src["height"], working_width)

    frames_dir = work_dir / FRAMES_DIRNAME
    meta_path = work_dir / META_FILENAME
    frames_dir.mkdir(parents=True, exist_ok=True)
    _clear_previous_output(frames_dir, meta_path)

    # Autorotate ffmpeg aktif default -> frame keluar tegak. Tinggi (-2) dihitung
    # ffmpeg dari frame yang SUDAH dirotasi, lalu divalidasi terhadap hitungan Python.
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-i", str(video_path),
        "-map", "0:v:0", "-an",
        "-vf", f"fps={target_fps},scale={out_w}:-2:flags={SCALE_FLAGS}",
        "-fps_mode", "passthrough",
        "-pix_fmt", "rgb24",
        "-start_number", str(FRAME_INDEX_START),
        str(frames_dir / FRAME_PATTERN),
    ]
    proc = _run(cmd)
    if proc.returncode != 0:
        raise IngestError(
            f"ffmpeg gagal (exit {proc.returncode}) untuk {video_path}:\n"
            f"{_stderr_tail(proc.stderr)}"
        )

    frames = sorted(frames_dir.glob(FRAME_GLOB))
    if not frames:
        raise IngestError(f"ffmpeg tidak menghasilkan frame dari {video_path}")

    first = frames_dir / (FRAME_PATTERN % FRAME_INDEX_START)
    with Image.open(first) as img:
        actual_w, actual_h = img.size
    if actual_w != out_w or abs(actual_h - out_h) > HEIGHT_TOLERANCE_PX:
        raise IngestError(
            f"dimensi output tidak cocok — cek deteksi rotasi/SAR: "
            f"ffmpeg menghasilkan {actual_w}x{actual_h}, diharapkan {out_w}x{out_h} "
            f"(sumber {src['width']}x{src['height']}, rotasi {src['rotation']})"
        )

    meta = {
        "source_path": str(video_path.resolve()),
        "source_fps": src["fps"],
        "source_duration_s": src["duration_s"],
        "source_frame_count": src["frame_count"],
        "source_width": src["width"],
        "source_height": src["height"],
        "source_rotation": src["rotation"],
        "target_fps": target_fps,
        "frame_count": len(frames),
        "working_width": actual_w,
        "working_height": actual_h,
        "has_audio": src["has_audio"],
        "frame_index_start": FRAME_INDEX_START,
    }
    _write_json_atomic(meta_path, meta)
    return meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m rotoscope.ingest",
        description="Stage [1]: video -> frames/*.png + meta.json",
    )
    parser.add_argument("video", type=Path)
    parser.add_argument("--work-dir", type=Path, default=Path(DEFAULT_WORK_DIR))
    parser.add_argument("--target-fps", type=float, default=DEFAULT_TARGET_FPS)
    parser.add_argument("--working-width", type=int, default=DEFAULT_WORKING_WIDTH)
    args = parser.parse_args(argv)

    start = time.perf_counter()
    try:
        meta = ingest(args.video, args.work_dir, args.target_fps, args.working_width)
    except (IngestError, FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    elapsed = time.perf_counter() - start

    print(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"{meta['frame_count']} frame -> {args.work_dir / FRAMES_DIRNAME} ({elapsed:.2f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
