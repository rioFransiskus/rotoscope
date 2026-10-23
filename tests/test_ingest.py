"""Test stage [1] ingest. Fixture video dibuat sintetis via ffmpeg lavfi (testsrc)."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from PIL import Image

from rotoscope import ingest as ingest_mod
from rotoscope.ingest import IngestError, compute_output_size, ingest, probe_video

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe tidak ada di PATH",
)

REQUIRED_META_FIELDS = {
    "source_path", "source_fps", "source_duration_s", "source_frame_count",
    "source_width", "source_height", "target_fps", "frame_count",
    "working_width", "working_height", "has_audio", "frame_index_start",
}


def _ffmpeg(*args: str) -> None:
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y", *args],
        check=True, capture_output=True,
    )


def make_video(
    path: Path, width: int, height: int, fps: int = 30, duration: float = 1.0,
    codec: tuple[str, ...] = ("-c:v", "libx264", "-pix_fmt", "yuv420p"),
) -> Path:
    _ffmpeg(
        "-f", "lavfi", "-i", f"testsrc=size={width}x{height}:rate={fps}:duration={duration}",
        *codec, str(path),
    )
    return path


def frame_files(work_dir: Path) -> list[Path]:
    return sorted((work_dir / "frames").glob("frame_*.png"))


# --- compute_output_size (tanpa ffmpeg) -------------------------------------

@pytest.mark.parametrize(
    ("src", "working_width", "expected"),
    [
        ((1080, 1920), 720, (720, 1280)),
        ((480, 854), 720, (480, 854)),    # tidak upscale
        ((481, 271), 720, (480, 270)),    # lebar ganjil <= working_width
        ((1001, 563), 720, (720, 404)),   # tinggi hasil skala ganjil -> genap
        ((1920, 1080), 721, (720, 406)),  # working_width ganjil
    ],
)
def test_compute_output_size(src, working_width, expected):
    w, h = compute_output_size(*src, working_width)
    assert (w, h) == expected
    assert w % 2 == 0 and h % 2 == 0


# --- ingest end-to-end ------------------------------------------------------

def test_fps_conversion_30fps_2s(tmp_path):
    video = make_video(tmp_path / "in.mp4", 320, 240, fps=30, duration=2)
    meta = ingest(video, tmp_path / "work")
    assert abs(meta["frame_count"] - 48) <= 1
    assert len(frame_files(tmp_path / "work")) == meta["frame_count"]
    assert (tmp_path / "work" / "frames" / "frame_00000.png").exists()


def test_no_upscale(tmp_path):
    video = make_video(tmp_path / "in.mp4", 480, 270, duration=0.5)
    meta = ingest(video, tmp_path / "work", working_width=720)
    assert (meta["working_width"], meta["working_height"]) == (480, 270)


def test_portrait_downscale(tmp_path):
    video = make_video(tmp_path / "in.mp4", 1080, 1920, duration=0.5)
    meta = ingest(video, tmp_path / "work")
    assert (meta["working_width"], meta["working_height"]) == (720, 1280)
    with Image.open(frame_files(tmp_path / "work")[0]) as img:
        assert img.size == (720, 1280)
        assert img.mode == "RGB"


def test_odd_dimensions_rounded_to_even(tmp_path):
    # libx264 yuv420p tidak bisa dimensi ganjil -> fixture pakai ffv1 (lossless) di .mkv
    video = make_video(tmp_path / "in.mkv", 481, 271, duration=0.5, codec=("-c:v", "ffv1"))
    meta = ingest(video, tmp_path / "work")
    assert (meta["source_width"], meta["source_height"]) == (481, 271)
    assert (meta["working_width"], meta["working_height"]) == (480, 270)


def test_rerun_leaves_no_stale_frames(tmp_path):
    work = tmp_path / "work"
    long_meta = ingest(make_video(tmp_path / "long.mp4", 320, 240, duration=2), work)
    short_meta = ingest(make_video(tmp_path / "short.mp4", 320, 240, duration=1), work)
    assert short_meta["frame_count"] < long_meta["frame_count"]
    files = frame_files(work)
    assert len(files) == short_meta["frame_count"]
    assert files[-1].name == f"frame_{short_meta['frame_count'] - 1:05d}.png"


def test_meta_json_has_required_fields(tmp_path):
    video = make_video(tmp_path / "in.mp4", 320, 240, duration=1)
    meta = ingest(video, tmp_path / "work")
    on_disk = json.loads((tmp_path / "work" / "meta.json").read_text(encoding="utf-8"))
    assert on_disk == meta
    assert REQUIRED_META_FIELDS <= on_disk.keys()
    assert on_disk["frame_index_start"] == 0
    assert on_disk["target_fps"] == 24
    assert on_disk["has_audio"] is False
    assert on_disk["source_fps"] == pytest.approx(30)
    assert on_disk["source_duration_s"] == pytest.approx(1, abs=0.1)
    assert not (tmp_path / "work" / "meta.json.tmp").exists()


def test_meta_json_is_byte_deterministic(tmp_path):
    """Dasar identitas klip (T-108): ingest ulang video + parameter yang sama → meta.json byte-identik
    (juga di work_dir lain); parameter ingest berbeda → byte berbeda."""
    video = make_video(tmp_path / "in.mp4", 320, 240, duration=1)
    ingest(video, tmp_path / "w1")
    ingest(video, tmp_path / "w1")                      # ingest ulang di work_dir yang sama
    ingest(video, tmp_path / "w2")                      # work_dir lain
    first = (tmp_path / "w1" / "meta.json").read_bytes()
    assert first == (tmp_path / "w2" / "meta.json").read_bytes()
    ingest(video, tmp_path / "w3", target_fps=12)
    assert (tmp_path / "w3" / "meta.json").read_bytes() != first
    ingest(video, tmp_path / "w4", working_width=160)
    assert (tmp_path / "w4" / "meta.json").read_bytes() != first


def test_has_audio_detected(tmp_path):
    video = tmp_path / "av.mp4"
    _ffmpeg(
        "-f", "lavfi", "-i", "testsrc=size=320x240:rate=30:duration=0.5",
        "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
        "-t", "0.5", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(video),
    )
    assert ingest(video, tmp_path / "work")["has_audio"] is True


@pytest.fixture
def rotated_video(tmp_path) -> Path:
    """Video 320x180 dengan display matrix rotasi 90 derajat (seperti video HP)."""
    src = make_video(tmp_path / "src.mp4", 320, 180, duration=0.5)
    rotated = tmp_path / "rot.mp4"
    _ffmpeg("-display_rotation", "90", "-i", str(src), "-c", "copy", str(rotated))
    return rotated


def test_rotation_metadata_output_upright(rotated_video, tmp_path):
    probe = probe_video(rotated_video)
    assert probe["rotation"] in (90, 270)
    meta = ingest(rotated_video, tmp_path / "work")
    assert (meta["source_width"], meta["source_height"]) == (180, 320)
    assert (meta["working_width"], meta["working_height"]) == (180, 320)
    with Image.open(frame_files(tmp_path / "work")[0]) as img:
        assert img.size == (180, 320)


def test_wrong_rotation_detection_raises(rotated_video, tmp_path, monkeypatch):
    real_probe = ingest_mod.probe_video

    def probe_without_rotation(path):
        info = real_probe(path)
        # simulasikan deteksi rotasi gagal: dimensi mentah, rotasi 0
        return {**info, "width": info["height"], "height": info["width"], "rotation": 0}

    monkeypatch.setattr(ingest_mod, "probe_video", probe_without_rotation)
    with pytest.raises(IngestError, match="dimensi output tidak cocok"):
        ingest(rotated_video, tmp_path / "work")
    assert not (tmp_path / "work" / "meta.json").exists()


# --- trim (T-607: indeks frame sesudah fps) ---------------------------------

@pytest.mark.parametrize(("start", "end", "expected"), [
    (0, 3.8333, (0, 92, 0.0, 3.833)),           # K5: klip5 f0–f91
    (0, 92 / 24, (0, 92, 0.0, 3.833)),          # E tepat di batas frame (galat float) → tetap 92
    (1 / 24, 92 / 24, (1, 92, 0.041, 3.833)),
    (0.5, 1.5, (12, 36, 0.5, 1.5)),
    (0.51, 1.0, (13, 24, 0.541, 1.0)),          # ceil, bukan round (0.51 · 24 = 12.24 → frame 13)
    (2, None, (48, None, 2.0, None)),
    (None, 1.0, (0, 24, 0.0, 1.0)),
])
def test_make_trim_ceil_end_exclusive_canonical(start, end, expected):
    t = ingest_mod.make_trim(start, end, 24)
    assert (t["start_frame"], t["end_frame"], t["start_s"], t["end_s"]) == expected
    again = ingest_mod.make_trim(t["start_s"], t["end_s"], 24)      # detik kanonik dibaca ulang → indeks sama (nama folder)
    assert (again["start_frame"], again["end_frame"]) == (t["start_frame"], t["end_frame"])


def test_make_trim_float_tolerance():
    """0.28 · 25 = 7.000000000000001 dan 0.56 · 25 = 14.000000000000002 di float: tanpa toleransi 1e-6 frame awal / akhir
    bergeser satu (ceil)."""
    assert 0.28 * 25 > 7 and 0.56 * 25 > 14
    t = ingest_mod.make_trim(0.28, 0.56, 25)
    assert (t["start_frame"], t["end_frame"]) == (7, 14)


@pytest.mark.parametrize(("start", "end"), [(None, None), (0, None), (0.0, None)])
def test_make_trim_untrimmed_is_none(start, end):
    assert ingest_mod.make_trim(start, end, 24) is None


@pytest.mark.parametrize(("start", "end", "match"), [
    (-1, None, "≥ 0"), (2, 2, "harus >"), (2, 1, "harus >"), (0.01, 0.02, "tidak memuat satu frame"),
])
def test_make_trim_rejects_bad_ranges(start, end, match):
    with pytest.raises(ValueError, match=match):
        ingest_mod.make_trim(start, end, 24)


def test_trim_frames_are_byte_identical_subset_renumbered(tmp_path):
    video = make_video(tmp_path / "in.mp4", 320, 240, fps=30, duration=2)
    full = ingest(video, tmp_path / "full")
    meta = ingest(video, tmp_path / "trim", start_s=0.5, end_s=1.5)
    assert meta["frame_count"] == 24 and meta["trim"] == {"start_s": 0.5, "end_s": 1.5, "start_frame": 12, "end_frame": 36}
    assert list(meta)[-1] == "trim" and meta["source_frame_count"] == full["source_frame_count"]
    a, b = frame_files(tmp_path / "full"), frame_files(tmp_path / "trim")
    assert b[0].name == "frame_00000.png" and len(b) == 24
    assert all(x.read_bytes() == y.read_bytes() for x, y in zip(a[12:36], b))
    tail = ingest(video, tmp_path / "tail", start_s=1.0)
    assert tail["frame_count"] == full["frame_count"] - 24 and tail["trim"]["end_frame"] is None
    assert all(x.read_bytes() == y.read_bytes() for x, y in zip(a[24:], frame_files(tmp_path / "tail")))


def test_trim_zero_start_without_end_is_byte_identical(tmp_path):
    video = make_video(tmp_path / "in.mp4", 320, 240, duration=1)
    ingest(video, tmp_path / "a")
    ingest(video, tmp_path / "b", start_s=0)
    assert (tmp_path / "a" / "meta.json").read_bytes() == (tmp_path / "b" / "meta.json").read_bytes()


@pytest.mark.parametrize(("start", "end", "match"), [(0.5, 9.0, r"--end 9 \(frame 216\).*klip punya 24 frame"),
                                                      (5.0, None, r"--start 5 \(frame 120\)")])
def test_trim_beyond_clip_end_fails_without_output(tmp_path, start, end, match):
    video = make_video(tmp_path / "in.mp4", 320, 240, duration=1)
    with pytest.raises(IngestError, match=match):
        ingest(video, tmp_path / "w", start_s=start, end_s=end)
    assert not (tmp_path / "w" / "meta.json").exists() and frame_files(tmp_path / "w") == []


# --- error handling ---------------------------------------------------------

def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        ingest(tmp_path / "tidak_ada.mp4", tmp_path / "work")


def test_not_a_video(tmp_path):
    bogus = tmp_path / "bogus.mp4"
    bogus.write_text("bukan video", encoding="utf-8")
    with pytest.raises(IngestError, match="ffprobe gagal"):
        ingest(bogus, tmp_path / "work")


def test_audio_only_has_no_video_stream(tmp_path):
    audio = tmp_path / "audio.wav"
    _ffmpeg("-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono", "-t", "0.5", str(audio))
    with pytest.raises(IngestError, match="tidak ada stream video"):
        ingest(audio, tmp_path / "work")


def test_ffmpeg_not_found(tmp_path, monkeypatch):
    video = make_video(tmp_path / "in.mp4", 320, 240, duration=0.5)
    monkeypatch.setattr(ingest_mod.shutil, "which", lambda name: None)
    with pytest.raises(IngestError, match="tidak ditemukan di PATH"):
        ingest(video, tmp_path / "work")


def test_ffmpeg_failure_leaves_no_meta(tmp_path, monkeypatch):
    video = make_video(tmp_path / "in.mp4", 320, 240, duration=0.5)
    work = tmp_path / "work"
    ingest(video, work)  # run sukses dulu -> ada meta.json + frame lama
    assert (work / "meta.json").exists()

    real_run = ingest_mod._run

    def failing_run(cmd):
        if Path(cmd[0]).stem.lower() == "ffmpeg":
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="boom: encoder rusak")
        return real_run(cmd)

    monkeypatch.setattr(ingest_mod, "_run", failing_run)
    with pytest.raises(IngestError, match="boom: encoder rusak"):
        ingest(video, work)
    assert not (work / "meta.json").exists()
    assert frame_files(work) == []
