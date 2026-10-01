"""Test stage [6] export (T-103): peta grup sintetis kecil → MP4 lewat ffmpeg / ffprobe nyata."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from rotoscope import export as ex
from rotoscope import stabilize as stab
from rotoscope.config import load_pipeline, section_hash
from rotoscope.stage_common import EXIT_OK, EXIT_PRECONDITION, StageError

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe tidak ada di PATH",
)

W, H, N, FPS = 64, 48, 6, 24


def _ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y", *args],
                   check=True, capture_output=True)


def make_source(path: Path, audio: bool) -> Path:
    cmd = ["-f", "lavfi", "-i", f"testsrc=size=64x48:rate={FPS}:duration=1"]
    if audio:
        cmd += ["-f", "lavfi", "-i", "sine=frequency=440:duration=1", "-c:a", "aac"]
    _ffmpeg(*cmd, "-pix_fmt", "yuv420p", str(path))
    return path


def gmap_for(i: int, w: int = W, h: int = H) -> np.ndarray:
    """Persegi grup 1 yang bergeser per frame + kotak grup 2."""
    g = np.zeros((h, w), np.uint8)
    g[10:30, 5 + i * 3:25 + i * 3] = 1
    g[35:45, 40:55] = 2
    return g


def make_clip(tmp_path: Path, *, w: int = W, h: int = H, n: int = N, source_name: str = "meme clip.mp4",
              has_audio: bool = False, source: Path | None = None, **overrides):
    """work_dir sintetis lengkap (meta + stable) → (cfg, work_dir, out_dir)."""
    work, out = tmp_path / "work", tmp_path / "out"
    cfg = load_pipeline(overrides={"paths.work_dir": str(work), "paths.out_dir": str(out), **overrides})
    (work / "stable" / "groups").mkdir(parents=True, exist_ok=True)
    src = source or (tmp_path / source_name)
    (work / "meta.json").write_text(json.dumps({
        "source_path": str(src), "target_fps": FPS, "frame_count": n, "working_width": w,
        "working_height": h, "has_audio": has_audio, "frame_index_start": 0}), encoding="utf-8")
    (work / "stable" / "manifest.json").write_text(json.dumps({
        "stabilize_hash": "s" * 64, "groups_hash": section_hash(cfg, "groups"),
        "frame_size": {"width": w, "height": h}, "created_utc": "2026-10-01T00:00:00+00:00"}), encoding="utf-8")
    for i in range(n):
        stab.write_groups(work / "stable" / "groups" / f"frame_{i:05d}.png", gmap_for(i, w, h))
    return cfg, work, out


def decode_frames(path: Path, w: int, h: int) -> np.ndarray:
    proc = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                          capture_output=True, check=True)
    return np.frombuffer(proc.stdout, np.uint8).reshape(-1, h, w)


# ── tanpa ffmpeg ───────────────────────────────────
@pytest.mark.parametrize("src, expected", [
    ("C:/v/meme clip.mp4", "meme_clip"), ("/a/b/Lucu!!(1).final.mov", "Lucu_1_.final"),
    ("/a/...mp4", "clip"), ("D:\\x\\kucing-lucu.mp4", "kucing-lucu"),
])
def test_sanitize_source_name(src, expected):
    assert ex.sanitize_source_name(src) == expected


def test_resolve_filename():
    assert ex.resolve_filename("{source}.mp4", "/a/b c.mp4") == "b_c.mp4"
    assert ex.resolve_filename("animation.mp4", "/a/b.mp4") == "animation.mp4"
    assert ex.resolve_filename("{source}.mp4", "/a/b.mp4", limit=5) == "b.limit5.mp4"
    assert ex.manifest_path_for(Path("out/b.mp4")) == Path("out/b.export.json")


def test_silhouette_maps_groups_to_colors(tmp_path):
    cfg, work, _ = make_clip(tmp_path, **{"export.foreground_color": "#102030", "export.background_color": "#a0b0c0"})
    src = ex.SilhouetteSource(cfg, stab.load_clip(work))
    img = src.render("frame_00000.png")
    assert img.shape == (H, W, 3) and img.dtype == np.uint8
    assert tuple(img[0, 0]) == (0xA0, 0xB0, 0xC0)           # background
    assert tuple(img[20, 10]) == (0x10, 0x20, 0x30)         # grup 1
    assert tuple(img[40, 45]) == (0x10, 0x20, 0x30)         # grup 2 juga foreground (grup ≠ 0)


def test_pad_even():
    bg = np.array([9, 9, 9], np.uint8)
    img = np.zeros((5, 7, 3), np.uint8)
    out = ex.pad_even(img, bg)
    assert out.shape == (6, 8, 3) and tuple(out[5, 7]) == (9, 9, 9) and out[4, 6].sum() == 0
    even = np.zeros((4, 6, 3), np.uint8)
    assert ex.pad_even(even, bg) is even


def test_verification_errors_lists_each_mismatch():
    probe = {"frames": 5, "fps": 25.0, "codec": "mpeg4", "pix_fmt": "yuv444p", "width": 10, "height": 10,
             "duration_s": 9.0, "audio_streams": 1}
    errs = ex.verification_errors(probe, frames=6, fps=24, size=(64, 48), audio=False)
    assert len(errs) == 7


# ── encode end-to-end ──────────────────────────────
def test_export_end_to_end(tmp_path):
    cfg, _, out = make_clip(tmp_path)
    run = ex.run_export(cfg, log=lambda m: None)
    mp4 = out / "meme_clip.mp4"
    assert run["output"] == mp4 and not run["skipped"] and mp4.is_file()
    p = ex.probe_output(mp4)
    assert (p["frames"], p["codec"], p["pix_fmt"], p["width"], p["height"], p["audio_streams"]) == \
           (N, "h264", "yuv420p", W, H, 0)
    assert p["fps"] == pytest.approx(FPS) and p["duration_s"] == pytest.approx(N / FPS, abs=1 / FPS)
    frames = decode_frames(mp4, W, H)
    assert len(frames) == N
    assert frames[0][20, 10] < 40 and frames[0][0, 0] > 215          # siluet hitam, latar putih
    # siluet bergerak: kolom yang hitam bergeser antar frame
    assert frames[0][20, 28] > 215 and frames[N - 1][20, 28] < 40
    assert not list(out.glob("*.tmp"))
    m = json.loads((out / "meme_clip.export.json").read_text(encoding="utf-8"))
    assert m["clip"]["source_path"] == str(tmp_path / "meme clip.mp4") and len(m["clip"]["meta_sha256"]) == 64
    assert m["output"]["frames"] == N and m["frame_count"] == N


def test_odd_dimensions_padded_to_even(tmp_path):
    cfg, _, out = make_clip(tmp_path, w=63, h=47)
    ex.run_export(cfg, log=lambda m: None)
    p = ex.probe_output(out / "meme_clip.mp4")
    assert (p["width"], p["height"], p["frames"]) == (64, 48, N)


def test_limit_writes_separate_preview_without_manifest(tmp_path):
    cfg, _, out = make_clip(tmp_path)
    run = ex.run_export(cfg, limit=3, log=lambda m: None)
    assert run["output"].name == "meme_clip.limit3.mp4"
    assert ex.probe_output(run["output"])["frames"] == 3
    assert not (out / "meme_clip.mp4").exists() and not list(out.glob("*.export.json"))
    with pytest.raises(StageError, match="--limit"):
        ex.run_export(cfg, limit=0, log=lambda m: None)


# ── manifest, basi, pengaman klip lain ─────────────
def test_second_run_skipped_then_stale_on_param_change(tmp_path):
    cfg, work, out = make_clip(tmp_path)
    ex.run_export(cfg, log=lambda m: None)
    mp4 = out / "meme_clip.mp4"
    mtime = mp4.stat().st_mtime_ns
    assert ex.run_export(cfg, log=lambda m: None)["skipped"] is True
    assert mp4.stat().st_mtime_ns == mtime

    logs: list[str] = []
    cfg2, _, _ = make_clip(tmp_path, **{"export.crf": 30})
    run = ex.run_export(cfg2, log=logs.append)
    assert not run["skipped"] and any("export_hash" in s for s in run["stale"])
    assert any("PERINGATAN" in m and "basi" in m for m in logs)
    assert ex.run_export(cfg2, log=lambda m: None)["skipped"] is True


def test_filename_change_alone_does_not_make_stale(tmp_path):
    cfg, _, out = make_clip(tmp_path)
    ex.run_export(cfg, log=lambda m: None)
    assert ex.export_hash(cfg) == ex.export_hash(load_pipeline(overrides={"export.filename": "lain.mp4"}))


def test_stale_when_meta_changes_same_source(tmp_path):
    cfg, work, out = make_clip(tmp_path)
    ex.run_export(cfg, log=lambda m: None)
    meta = json.loads((work / "meta.json").read_text(encoding="utf-8"))
    meta["working_width"] = W  # isi sama → hash sama → dilewati
    (work / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    assert ex.run_export(cfg, log=lambda m: None)["skipped"] is True
    meta["extra"] = 1  # isi berbeda → identitas klip beda → basi
    (work / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    run = ex.run_export(cfg, log=lambda m: None)
    assert not run["skipped"] and any("clip.meta_sha256" in s for s in run["stale"])


def test_refuses_to_overwrite_other_clip_result(tmp_path):
    cfg, work, out = make_clip(tmp_path, **{"export.filename": "animation.mp4"})
    ex.run_export(cfg, log=lambda m: None)
    before = (out / "animation.mp4").read_bytes()
    meta = json.loads((work / "meta.json").read_text(encoding="utf-8"))
    meta["source_path"] = str(tmp_path / "klip_lain.mp4")
    (work / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    for restart in (False, True):
        with pytest.raises(StageError, match=r"video sumber LAIN.*export\.filename"):
            ex.run_export(cfg, restart=restart, log=lambda m: None)
    assert (out / "animation.mp4").read_bytes() == before


def test_existing_file_without_manifest_refused_unless_restart(tmp_path):
    cfg, _, out = make_clip(tmp_path)
    out.mkdir()
    (out / "meme_clip.mp4").write_bytes(b"bukan video")
    with pytest.raises(StageError, match="tanpa meme_clip.export.json.*--restart"):
        ex.run_export(cfg, log=lambda m: None)
    assert (out / "meme_clip.mp4").read_bytes() == b"bukan video"
    ex.run_export(cfg, restart=True, log=lambda m: None)
    assert ex.probe_output(out / "meme_clip.mp4")["frames"] == N


def test_corrupt_existing_output_is_reencoded(tmp_path):
    cfg, _, out = make_clip(tmp_path)
    ex.run_export(cfg, log=lambda m: None)
    (out / "meme_clip.mp4").write_bytes(b"rusak")
    run = ex.run_export(cfg, log=lambda m: None)
    assert not run["skipped"] and ex.probe_output(out / "meme_clip.mp4")["frames"] == N


def test_restart_forces_reencode(tmp_path):
    cfg, _, _ = make_clip(tmp_path)
    ex.run_export(cfg, log=lambda m: None)
    assert ex.run_export(cfg, restart=True, log=lambda m: None)["skipped"] is False


# ── tulis atomik ───────────────────────────────────
def test_failed_encode_keeps_old_mp4_and_leaves_no_tmp(tmp_path):
    cfg, work, out = make_clip(tmp_path)
    ex.run_export(cfg, log=lambda m: None)
    before = (out / "meme_clip.mp4").read_bytes()
    (work / "stable" / "groups" / "frame_00004.png").write_bytes(b"rusak")  # frame rusak di tengah encode
    with pytest.raises(StageError, match="peta grup frame_00004 rusak"):
        ex.run_export(cfg, restart=True, log=lambda m: None)
    assert (out / "meme_clip.mp4").read_bytes() == before and not list(out.glob("*.tmp"))


def test_ffmpeg_failure_raises_with_tail(tmp_path, monkeypatch):
    cfg, _, out = make_clip(tmp_path)
    real = ex.build_ffmpeg_cmd
    monkeypatch.setattr(ex, "build_ffmpeg_cmd",
                        lambda *a, **k: real(*a, **k)[:-1] + [str(tmp_path / "tidak" / "ada" / "x.mp4")])
    with pytest.raises(StageError, match="ffmpeg gagal"):
        ex.run_export(cfg, log=lambda m: None)
    assert not list(out.glob("*.tmp")) and not (out / "meme_clip.mp4").exists()


# ── audio ──────────────────────────────────────────
def test_audio_false_has_no_audio_stream_even_if_source_has_audio(tmp_path):
    src = make_source(tmp_path / "a.mp4", audio=True)
    cfg, _, out = make_clip(tmp_path, source=src, has_audio=True)
    ex.run_export(cfg, log=lambda m: None)
    assert ex.probe_output(out / "a.mp4")["audio_streams"] == 0


def test_audio_true_adds_aac_and_matches_duration(tmp_path):
    src = make_source(tmp_path / "a.mp4", audio=True)
    cfg, _, out = make_clip(tmp_path, source=src, has_audio=True, **{"export.audio": True})
    ex.run_export(cfg, log=lambda m: None)
    p = ex.probe_output(out / "a.mp4")
    assert p["audio_streams"] == 1 and p["audio_codec"] == "aac" and p["frames"] == N
    assert p["duration_s"] == pytest.approx(N / FPS, abs=1 / FPS + 0.05)
    # audio berubah dari tanpa → ada: basi
    cfg_off, _, _ = make_clip(tmp_path, source=src, has_audio=True)
    run = ex.run_export(cfg_off, log=lambda m: None)
    assert any("audio" in s for s in run["stale"]) and ex.probe_output(out / "a.mp4")["audio_streams"] == 0


def test_audio_true_without_source_audio_is_error(tmp_path):
    src = make_source(tmp_path / "v.mp4", audio=False)
    cfg, _, out = make_clip(tmp_path, source=src, has_audio=False, **{"export.audio": True})
    with pytest.raises(StageError, match="tidak punya audio"):
        ex.run_export(cfg, log=lambda m: None)
    assert not out.exists() or not list(out.glob("*.mp4"))


def test_audio_true_missing_source_file_is_error(tmp_path):
    cfg, _, _ = make_clip(tmp_path, has_audio=True, **{"export.audio": True})
    with pytest.raises(StageError, match="tidak ditemukan"):
        ex.run_export(cfg, log=lambda m: None)


# ── prasyarat + entry point ────────────────────────
def test_missing_meta(tmp_path):
    cfg = load_pipeline(overrides={"paths.work_dir": str(tmp_path / "work"), "paths.out_dir": str(tmp_path / "out")})
    with pytest.raises(StageError, match="meta.json"):
        ex.run_export(cfg, log=lambda m: None)


def test_missing_stable_manifest(tmp_path):
    cfg, work, _ = make_clip(tmp_path)
    (work / "stable" / "manifest.json").unlink()
    with pytest.raises(StageError, match=r"stabilize"):
        ex.run_export(cfg, log=lambda m: None)


def test_missing_group_frames(tmp_path):
    cfg, work, _ = make_clip(tmp_path)
    (work / "stable" / "groups" / "frame_00002.png").unlink()
    with pytest.raises(StageError, match=r"1 dari 6 frame hilang.*frame_00002"):
        ex.run_export(cfg, log=lambda m: None)


def test_groups_hash_mismatch(tmp_path):
    cfg, work, _ = make_clip(tmp_path)
    m = json.loads((work / "stable" / "manifest.json").read_text(encoding="utf-8"))
    m["groups_hash"] = "x"
    (work / "stable" / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(StageError, match="groups"):
        ex.run_export(cfg, log=lambda m: None)


def test_group_id_above_n_groups_rejected(tmp_path):
    cfg, work, _ = make_clip(tmp_path)
    bad = gmap_for(0)
    bad[0, 0] = len(cfg.groups) + 1
    stab.write_groups(work / "stable" / "groups" / "frame_00000.png", bad)
    with pytest.raises(StageError, match="rusak"):
        ex.run_export(cfg, log=lambda m: None)


def test_strokes_source_not_implemented(tmp_path):
    cfg, _, _ = make_clip(tmp_path, **{"export.source": "strokes"})
    with pytest.raises(StageError, match="belum diimplementasi"):
        ex.run_export(cfg, log=lambda m: None)


def test_main_exit_codes(tmp_path, monkeypatch, capsys):
    cfg, work, out = make_clip(tmp_path)
    conf = tmp_path / "c.yaml"
    conf.write_text(f"paths:\n  work_dir: '{work}'\n  out_dir: '{out}'\n", encoding="utf-8")
    assert ex.main(["--config", str(conf)]) == EXIT_OK
    assert (out / "meme_clip.mp4").is_file()
    assert ex.main(["--config", str(conf), "--limit", "2"]) == EXIT_OK
    assert ex.main(["--config", str(tmp_path / "tidak-ada.yaml")]) == EXIT_PRECONDITION
    (work / "meta.json").unlink()
    assert ex.main(["--config", str(conf)]) == EXIT_PRECONDITION
    assert "ERROR:" in capsys.readouterr().err
