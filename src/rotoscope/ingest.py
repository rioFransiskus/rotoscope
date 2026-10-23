"""Stage [1] ingest: video -> work/frames/frame_%05d.png + work/meta.json.

Decode, konversi fps, rotasi, dan resize dikerjakan ffmpeg via subprocess.
Filter `fps` ffmpeg bekerja berdasarkan timestamp, jadi aman untuk video VFR
(variable frame rate, umum di rekaman HP). Metadata dibaca dari ffprobe (JSON),
bukan CAP_PROP_* OpenCV yang tidak andal untuk VFR.

Frame index mulai 0 (frame_00000.png = index 0). Index ini dipakai sebagai seed
jitter di stage berikutnya (P-007), jadi harus konsisten.

meta.json ditulis PALING AKHIR sebagai penanda sukses: kalau ada meta.json,
frame di folder yang sama lengkap dan tervalidasi.

CLI final (T-104b): python -m rotoscope ingest <video> [--start S] [--end E] [--target-fps N] [--working-width N] (folder kerja =
<paths.work_dir>/clips/<nama video>[@S-E]/); atau seluruh pipeline: python -m rotoscope run <video>. Entry point stage:
    python -m rotoscope.ingest <video> [--work-dir work] [--config PATH] [--crop-dir DIR] [--start S] [--end E]
                                       [--target-fps N] [--working-width N]

T-607 (D-013): trim SESUDAH `fps`, berdasarkan indeks frame (`trim=start_frame=a:end_frame=b`; a = ceil(S·fps − 1e-6),
b = ceil(E·fps − 1e-6), akhir eksklusif; frame dinomori ulang dari 0). Crop: ingest TIDAK pernah memotong sendiri — kotak hanya dari
<crop-dir>/kotak.json (alat `python -m rotoscope crop`); tanpa kotak, deteksi R2 pada rentang penuh sumber → saran + pratinjau
(rotoscope.crop). Tanpa --crop-dir: tanpa saran / kotak. Key `trim` / `crop` di meta.json HANYA bila dipakai (klip lain byte-identik).
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path

from PIL import Image

from rotoscope.config import ConfigError, CropConfig, load_pipeline

DEFAULT_TARGET_FPS = 24
DEFAULT_WORKING_WIDTH = 720
DEFAULT_WORK_DIR = "work"
DEFAULT_CONFIG = Path("configs") / "default.yaml"
TRIM_EPS = 1e-6                 # toleransi galat float konversi detik → indeks (persetujuan rancangan T-607 no. 1)

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


def _canonical_s(frames: int, fps: float) -> float:
    """Detik frame ke-n dibulatkan KE BAWAH ke milidetik: dibaca ulang dengan ceil(S·fps − 1e-6) memberi n yang sama."""
    return math.floor(frames * 1000 / fps + 1e-9) / 1000


def make_trim(start_s: float | None, end_s: float | None, fps: float) -> dict | None:
    """--start / --end (detik dari frame pertama ingest) → key `trim` meta.json: {start_s, end_s, start_frame a, end_frame b}
    (detik KANONIK, sama dengan nama folder <nama>@S-E). None = tanpa trim (a = 0, tanpa akhir). ValueError = masukan salah."""
    if start_s is None and end_s is None:
        return None
    s = 0.0 if start_s is None else float(start_s)
    if not math.isfinite(s) or s < 0:
        raise ValueError(f"--start harus ≥ 0 detik, dapat {start_s}")
    a = math.ceil(s * fps - TRIM_EPS)
    b = None
    if end_s is not None:
        e = float(end_s)
        if not math.isfinite(e) or e <= s:
            raise ValueError(f"--end ({end_s}) harus > --start ({s:g}) detik")
        b = math.ceil(e * fps - TRIM_EPS)
        if b <= a:
            raise ValueError(f"--start {s:g} --end {e:g} tidak memuat satu frame pun (frame {a} … {b - 1} pada {fps:g} fps)")
    if a == 0 and b is None:
        return None
    return {"start_s": _canonical_s(a, fps), "end_s": _canonical_s(b, fps) if b is not None else None,
            "start_frame": a, "end_frame": b}


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
    *,
    crop_dir: str | Path | None = None,
    crop_cfg: CropConfig | None = None,
    start_s: float | None = None,
    end_s: float | None = None,
    log: Callable[[str], None] = print,
) -> dict:
    """Ekstrak frame video ke <work_dir>/frames/ dan tulis <work_dir>/meta.json.

    `crop_dir` (T-607) = <paths.work_dir>/crop/<nama>/: kotak.json di sana dipakai (crop SEBELUM scale); tanpa kotak + `crop.suggest`
    → deteksi R2 rentang penuh sumber → saran + pratinjau + pesan (TIDAK memotong). `start_s` / `end_s` = trim berbasis indeks
    sesudah fps. Return isi meta.json.
    """
    video_path = Path(video_path)
    work_dir = Path(work_dir)
    if not video_path.is_file():
        raise FileNotFoundError(f"file video tidak ditemukan: {video_path}")
    if target_fps <= 0:
        raise ValueError(f"target_fps harus > 0, dapat {target_fps}")
    trim = make_trim(start_s, end_s, target_fps)

    ffmpeg = _require_binary("ffmpeg")
    src = probe_video(video_path)
    kotak = saran = None
    if crop_dir is not None:
        from rotoscope import crop as crop_mod          # malas: crop → stage_common → ingest (impor melingkar di level modul)

        crop_dir = Path(crop_dir)
        cfg_c = crop_cfg or CropConfig()
        identity = crop_mod.source_identity(video_path, src)
        video_arg = crop_mod.quote_arg(identity["path"])
        try:
            kotak = crop_mod.kotak_for_ingest(crop_dir, identity, video_arg)
        except crop_mod.CropError as e:
            raise IngestError(str(e)) from None
        if kotak is None and cfg_c.suggest:
            w0, _ = compute_output_size(src["width"], src["height"], working_width)
            try:
                saran = crop_mod.suggest(
                    ffmpeg=ffmpeg, video=video_path, identity=identity, src_wh=(src["width"], src["height"]),
                    detect_vf=f"fps={target_fps},scale={w0}:-2:flags={SCALE_FLAGS}", target_fps=target_fps,
                    working_width=working_width, cfg=cfg_c, crop_dir=crop_dir, duration_s=src["duration_s"], log=log)
            except crop_mod.CropError as e:
                log(f"PERINGATAN: deteksi crop gagal ({e}) — ingest tanpa saran")
    box = kotak["box_source"] if kotak is not None and kotak["mode"] != "none" else None
    base_w, base_h = (box[2], box[3]) if box else (src["width"], src["height"])
    out_w, out_h = compute_output_size(base_w, base_h, working_width)

    frames_dir = work_dir / FRAMES_DIRNAME
    meta_path = work_dir / META_FILENAME
    frames_dir.mkdir(parents=True, exist_ok=True)
    _clear_previous_output(frames_dir, meta_path)

    # Autorotate ffmpeg aktif default -> frame keluar tegak. Tinggi (-2) dihitung
    # ffmpeg dari frame yang SUDAH dirotasi, lalu divalidasi terhadap hitungan Python.
    # T-607: trim (indeks frame sesudah fps) dan crop (px sumber, SEBELUM scale) hanya bila dipakai → tanpa keduanya perintah sama.
    vf = [f"fps={target_fps}"]
    if trim is not None:
        vf.append(f"trim=start_frame={trim['start_frame']}"
                  + (f":end_frame={trim['end_frame']}" if trim["end_frame"] is not None else ""))
    if box:
        vf.append(f"crop={box[2]}:{box[3]}:{box[0]}:{box[1]}:exact=1")
    vf.append(f"scale={out_w}:-2:flags={SCALE_FLAGS}")
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-i", str(video_path),
        "-map", "0:v:0", "-an",
        "-vf", ",".join(vf),
        "-fps_mode", "passthrough",
        "-pix_fmt", "rgb24",
        "-start_number", str(FRAME_INDEX_START),
        str(frames_dir / FRAME_PATTERN),
    ]
    proc = _run(cmd)
    frames = sorted(frames_dir.glob(FRAME_GLOB))
    if proc.returncode != 0:
        if trim is not None and not frames:            # --start melewati akhir klip: ffmpeg bisa gagal karena 0 frame
            _check_trim(trim, 0, target_fps, src, frames_dir, meta_path)
        raise IngestError(
            f"ffmpeg gagal (exit {proc.returncode}) untuk {video_path}:\n"
            f"{_stderr_tail(proc.stderr)}"
        )

    if trim is not None:
        _check_trim(trim, len(frames), target_fps, src, frames_dir, meta_path)
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
    if trim is not None:
        meta["trim"] = trim
    crop_key = crop_mod.meta_crop_key(kotak) if crop_dir is not None else None
    if crop_key is not None:
        meta["crop"] = crop_key
    _write_json_atomic(meta_path, meta)
    if saran is not None:
        for line in crop_mod.saran_lines(saran, crop_dir, video_arg):
            log(line)
    return meta


def _check_trim(trim: dict, n_out: int, fps: float, src: dict, frames_dir: Path, meta_path: Path) -> None:
    """Rentang trim harus berada di dalam klip: --start melewati akhir → 0 frame; --end melewati akhir → frame kurang."""
    a, b = trim["start_frame"], trim["end_frame"]
    if n_out == 0 or (b is not None and n_out < b - a):
        _clear_previous_output(frames_dir, meta_path)
        dur = src.get("duration_s")
        length = (f"klip punya {a + n_out} frame ({_canonical_s(a + n_out, fps):g} s pada {fps:g} fps)" if n_out
                  else f"klip ±{dur:.3f} s ≈ {round(dur * fps)} frame pada {fps:g} fps" if dur else "panjang klip tidak diketahui")
        what = (f"--start {trim['start_s']:g} (frame {a})" if n_out == 0 else f"--end {trim['end_s']:g} (frame {b})")
        raise IngestError(f"trim {what} melewati akhir klip: {length}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m rotoscope.ingest",
        description="Stage [1]: video -> frames/*.png + meta.json",
    )
    parser.add_argument("video", type=Path)
    parser.add_argument("--work-dir", type=Path, default=Path(DEFAULT_WORK_DIR))
    parser.add_argument("--config", type=Path, default=None,
                        help=f"YAML pipeline, section ingest (default: {DEFAULT_CONFIG.as_posix()} kalau ada, selain itu default kode)")
    parser.add_argument("--crop-dir", type=Path, default=None,
                        help="<work_dir>/crop/<nama>/ (cli): kotak.json dipakai, tanpa kotak → saran; tanpa flag: tanpa saran / kotak")
    parser.add_argument("--start", dest="trim_start", type=float, default=None, help="trim: detik awal (indeks frame sesudah fps)")
    parser.add_argument("--end", dest="trim_end", type=float, default=None, help="trim: detik akhir (eksklusif)")
    parser.add_argument("--target-fps", type=float, default=DEFAULT_TARGET_FPS)
    parser.add_argument("--working-width", type=int, default=DEFAULT_WORKING_WIDTH)
    args = parser.parse_args(argv)
    from rotoscope.stage_common import set_message_config      # malas: stage_common meng-import modul ini

    set_message_config(args.config)                             # T-607: --config non-default ikut di perintah siap tempel

    start = time.perf_counter()
    notes: list[str] = []                       # PERINGATAN + saran crop: dicetak SESUDAH meta supaya tidak tertimbun
    try:
        path = args.config if args.config is not None else (DEFAULT_CONFIG if DEFAULT_CONFIG.is_file() else None)
        cfg = load_pipeline(path)
        meta = ingest(args.video, args.work_dir, args.target_fps, args.working_width, crop_dir=args.crop_dir,
                      crop_cfg=cfg.ingest.crop, start_s=args.trim_start, end_s=args.trim_end, log=notes.append)
    except (IngestError, FileNotFoundError, ValueError, ConfigError) as exc:
        for line in notes:
            print(line, flush=True)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    elapsed = time.perf_counter() - start

    print(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"{meta['frame_count']} frame -> {args.work_dir / FRAMES_DIRNAME} ({elapsed:.2f} s)")
    for line in notes:
        print(line, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
