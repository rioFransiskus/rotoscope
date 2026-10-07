"""Stage [6] export — source "strokes" (T-203b): validasi masukan, manifest, tag warna, verifikasi ffprobe, salinan
SVG + perlindungan suntingan pengguna, determinisme, metrik objektif (sintetis + klip nyata bila ada)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

import export_metrics as em
import test_export as te
from rotoscope import export as ex
from rotoscope.config import load_pipeline
from rotoscope.stage_common import StageError, clip_identity

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe tidak ada di PATH",
)

SCALE = 2
STROKES = {"export.source": "strokes"}
RED, GREEN, BLUE = (200, 30, 30), (30, 170, 60), (30, 60, 200)
REAL_CLIPS = Path("work") / "clips"
quiet = lambda m: None      # noqa: E731


# ── data sintetis ──────────────────────────────────
def strokes_png(i: int, w: int = te.W, h: int = te.H) -> np.ndarray:
    """RGB (h·2, w·2): kertas + balok jenuh (merah, hijau, biru) + batang tinta yang bergeser per frame."""
    img = np.empty((h * SCALE, w * SCALE, 3), np.uint8)
    img[:] = em.PAPER_RGB
    for x, c in ((8, RED), (48, GREEN), (88, BLUE)):
        img[8:40, x:x + 32] = c
    img[60:76, 10 + 4 * i:90 + 4 * i] = em.INK_RGB
    return img


def svg_text(i: int, w: int, h: int, tag: str = "") -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">'
            f'<path d="M0 0 L{i} {i}"/>{tag}</svg>\n')


def make_strokes(work: Path, cfg, *, n: int = te.N, w: int = te.W, h: int = te.H, tag: str = "",
                 created_utc: str = "2026-10-03T00:00:00+00:00") -> None:
    """contours/ + strokes/ sintetis yang konsisten satu sama lain (manifest [4] dan [5] minimal)."""
    ident = clip_identity(work)
    cdir, sdir = work / "contours", work / "strokes"
    cdir.mkdir(parents=True, exist_ok=True)
    sdir.mkdir(parents=True, exist_ok=True)
    cm = {"stage": "vectorize", "contract": "T-202", "algo_rev": 2, "pending": [], "clip": ident,
          "frame_size": {"width": w, "height": h}, "vectorize_hash": "c" * 64, "created_utc": created_utc}
    (cdir / "manifest.json").write_text(json.dumps(cm), encoding="utf-8")
    ow, oh = w * SCALE, h * SCALE
    sm = {"stage": "stylize", "contract": "T-402", "algo_rev": 1, "style": "t", "style_hash": "d" * 64,
          "contours": {k: cm[k] for k in ("contract", "vectorize_hash", "algo_rev", "created_utc")},
          "frame_size": {"width": w, "height": h}, "clip": ident, "output_width": ow,
          "output_size": {"width": ow, "height": oh}, "created_utc": "2026-10-03T01:00:00+00:00"}
    (sdir / "manifest.json").write_text(json.dumps(sm), encoding="utf-8")
    for i in range(n):
        ok, buf = cv2.imencode(".png", cv2.cvtColor(strokes_png(i, w, h), cv2.COLOR_RGB2BGR))
        assert ok
        (sdir / f"frame_{i:05d}.png").write_bytes(buf.tobytes())
        (sdir / f"frame_{i:05d}.svg").write_text(svg_text(i, ow, oh, tag), encoding="utf-8")


def clip_strokes(tmp_path: Path, **kw):
    cfg, work, out = te.make_clip(tmp_path, **{**STROKES, **kw})
    make_strokes(work, cfg)
    return cfg, work, out


def edit_json(path: Path, **changes) -> None:
    m = json.loads(path.read_text(encoding="utf-8"))
    m.update(changes)
    path.write_text(json.dumps(m), encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# ── happy path ─────────────────────────────────────
def test_strokes_end_to_end(tmp_path):
    cfg, work, out = clip_strokes(tmp_path)
    run = ex.run_export(cfg, log=quiet)
    mp4 = out / "meme_clip.mp4"
    assert run["output"] == mp4 and not run["skipped"]
    p = ex.probe_output(mp4)
    assert (p["width"], p["height"], p["frames"], p["codec"], p["pix_fmt"]) == \
        (te.W * SCALE, te.H * SCALE, te.N, "h264", "yuv420p")       # ukuran OUTPUT, bukan resolusi kerja
    assert p["color"] == {"color_space": "bt709", "color_primaries": "bt709", "color_transfer": "bt709",
                          "color_range": "tv"}
    m = read_json(out / "meme_clip.export.json")
    assert m["source"] == "strokes" and "stable" not in m
    assert m["strokes"] == {"contract": "T-402", "style_hash": "d" * 64, "created_utc": "2026-10-03T01:00:00+00:00",
                            "output_size": {"width": te.W * SCALE, "height": te.H * SCALE}}
    assert m["encoded_size"] == m["strokes"]["output_size"] and m["frame_size"] == {"width": te.W, "height": te.H}
    assert m["color_tags"] == ex.COLOR_TAGS
    svg = out / "svg" / "meme_clip"
    assert run["svg"]["count"] == te.N and run["svg"]["copied"] == te.N
    for i in range(te.N):
        name = f"frame_{i:05d}.svg"
        assert em.sha256_file(svg / name) == em.sha256_file(work / "strokes" / name)
    marker = read_json(svg / ex.SVG_MARKER)
    assert marker["clip"] == clip_identity(work) and set(marker["files"]) == {f"frame_{i:05d}.svg" for i in range(te.N)}


def test_color_tags_are_discriminating_saturated_colors(tmp_path):
    """Palet netral (kertas, tinta) tidak membedakan bt601 / bt709 — warna jenuh yang membedakan (terukur)."""
    cfg, _, out = clip_strokes(tmp_path)
    ex.run_export(cfg, log=quiet)
    ref, dec = strokes_png(0), em.decode_rgb(out / "meme_clip.mp4", 0, "bt709")
    for name, color in (("merah", RED), ("hijau", GREEN), ("biru", BLUE), ("kertas", em.PAPER_RGB),
                        ("tinta", em.INK_RGB)):
        d = em.flat_color_delta(ref, dec, color)
        assert d is not None and max(abs(x) for x in d) <= em.COLOR_TOLERANCE, (name, d)


def test_color_tags_only_for_strokes_not_silhouette(tmp_path):
    cfg, _, out = te.make_clip(tmp_path)
    ex.run_export(cfg, log=quiet)
    assert ex.probe_output(out / "meme_clip.mp4")["color"]["color_space"] != "bt709"
    assert "-colorspace" not in ex.build_ffmpeg_cmd("ffmpeg", (64, 48), 24, 18, "medium", None, Path("o.mp4"))
    cmd = ex.build_ffmpeg_cmd("ffmpeg", (64, 48), 24, 18, "medium", None, Path("o.mp4"), color_tags=True)
    for flag, value in ex.COLOR_TAGS.items():
        assert cmd[cmd.index(f"-{flag}") + 1] == value
    vf = cmd[cmd.index("-vf") + 1]
    assert vf.startswith("setparams=") and all(f"{ex.SETPARAMS_KEYS[f]}={v}" in vf for f, v in ex.COLOR_TAGS.items())
    assert set(ex.COLOR_TAGS) == {"colorspace", "color_primaries", "color_trc", "color_range"}


def test_verification_checks_all_color_tags():
    probe = {"frames": 6, "fps": 24.0, "codec": "h264", "pix_fmt": "yuv420p", "width": 8, "height": 8,
             "duration_s": 0.25, "audio_streams": 0, "color": dict.fromkeys(
                 ("color_space", "color_primaries", "color_transfer", "color_range"))}
    kw = dict(frames=6, fps=24.0, size=(8, 8), audio=False)
    assert ex.verification_errors(probe, **kw) == []
    assert len(ex.verification_errors(probe, **kw, color_tags=True)) == 4
    for field_name in ("color_space", "color_primaries", "color_transfer", "color_range"):
        good = {"color_space": "bt709", "color_primaries": "bt709", "color_transfer": "bt709", "color_range": "tv"}
        probe["color"] = {**good, field_name: "unknown"}
        errs = ex.verification_errors(probe, **kw, color_tags=True)
        assert len(errs) == 1 and field_name in errs[0]
    probe["color"] = good
    assert ex.verification_errors(probe, **kw, color_tags=True) == []


def test_strokes_verification_uses_output_size_not_working_size(tmp_path):
    cfg, _, out = clip_strokes(tmp_path)
    ex.run_export(cfg, log=quiet)
    p = ex.probe_output(out / "meme_clip.mp4")
    with_working = ex.verification_errors(p, frames=te.N, fps=te.FPS, size=(te.W, te.H), audio=False, color_tags=True)
    assert any("ukuran" in e for e in with_working)


# ── validasi masukan ───────────────────────────────
def test_missing_strokes_manifest(tmp_path):
    cfg, work, _ = clip_strokes(tmp_path)
    (work / "strokes" / "manifest.json").unlink()
    with pytest.raises(StageError, match=r"stylize .*meme clip\.mp4") as e:
        ex.run_export(cfg, log=quiet)
    assert "manifest.json" in str(e.value)


def test_unsupported_strokes_contract(tmp_path):
    cfg, work, _ = clip_strokes(tmp_path)
    edit_json(work / "strokes" / "manifest.json", contract="T-999")
    with pytest.raises(StageError, match=r"contract 'T-999'.*stylize"):
        ex.run_export(cfg, log=quiet)


def test_strokes_manifest_of_other_clip(tmp_path):
    cfg, work, _ = clip_strokes(tmp_path)
    edit_json(work / "strokes" / "manifest.json", clip={"meta_sha256": "0" * 64, "source_path": "lain.mp4"})
    with pytest.raises(StageError, match=r"klip lain.*stylize"):
        ex.run_export(cfg, log=quiet)


def test_strokes_manifest_frame_size_mismatch(tmp_path):
    cfg, work, _ = clip_strokes(tmp_path)
    edit_json(work / "strokes" / "manifest.json", frame_size={"width": 1, "height": 1})
    with pytest.raises(StageError, match=r"frame_size.*stylize"):
        ex.run_export(cfg, log=quiet)


@pytest.mark.parametrize("osz", [{"width": 127, "height": 96}, {"width": 128}, "x"])
def test_strokes_manifest_bad_output_size(tmp_path, osz):
    cfg, work, _ = clip_strokes(tmp_path)
    edit_json(work / "strokes" / "manifest.json", output_size=osz)
    with pytest.raises(StageError, match="output_size"):
        ex.run_export(cfg, log=quiet)


def test_strokes_reference_to_other_contours_run(tmp_path):
    cfg, work, _ = clip_strokes(tmp_path)
    edit_json(work / "contours" / "manifest.json", vectorize_hash="e" * 64)
    with pytest.raises(StageError, match=r"contours yang berbeda.*stylize"):
        ex.run_export(cfg, log=quiet)
    edit_json(work / "contours" / "manifest.json", vectorize_hash="c" * 64, created_utc="2027-01-01T00:00:00+00:00")
    with pytest.raises(StageError, match="contours yang berbeda"):
        ex.run_export(cfg, log=quiet)


def test_contours_manifest_missing_points_to_vectorize(tmp_path):
    cfg, work, _ = clip_strokes(tmp_path)
    (work / "contours" / "manifest.json").unlink()
    with pytest.raises(StageError, match="vectorize"):
        ex.run_export(cfg, log=quiet)


@pytest.mark.parametrize("damage", ["missing", "corrupt", "wrong_size"])
def test_strokes_png_problems(tmp_path, damage):
    cfg, work, out = clip_strokes(tmp_path)
    png = work / "strokes" / "frame_00002.png"
    if damage == "missing":
        png.unlink()
    elif damage == "corrupt":
        png.write_bytes(png.read_bytes()[:50])
    else:
        ok, buf = cv2.imencode(".png", np.zeros((10, 10, 3), np.uint8))
        png.write_bytes(buf.tobytes())
    with pytest.raises(StageError, match=r"strokes/\*\.png.*frame_00002.*stylize"):
        ex.run_export(cfg, log=quiet)
    assert not (out / "meme_clip.mp4").exists() and not (out / "svg").exists()


def test_silhouette_still_does_not_need_strokes(tmp_path):
    cfg, work, out = te.make_clip(tmp_path)
    assert not (work / "strokes").exists()
    assert ex.run_export(cfg, log=quiet)["svg"] is None and not (out / "svg").exists()


# ── manifest, stale, ffmpeg ────────────────────────
def test_second_run_skipped_and_stale_cases(tmp_path):
    cfg, work, out = clip_strokes(tmp_path)
    ex.run_export(cfg, log=quiet)
    assert ex.run_export(cfg, log=quiet)["skipped"]
    # strokes dihitung ulang ([5] mengubah created_utc) → export basi → di-encode ulang otomatis
    edit_json(work / "strokes" / "manifest.json", created_utc="2026-10-04T00:00:00+00:00")
    run = ex.run_export(cfg, log=quiet)
    assert not run["skipped"] and any("strokes.created_utc" in s for s in run["stale"])
    edit_json(work / "strokes" / "manifest.json", style_hash="f" * 64)
    assert any("style_hash" in s for s in ex.run_export(cfg, log=quiet)["stale"])
    cfg2 = load_pipeline(overrides={"paths.work_dir": str(work), "paths.out_dir": str(out), **STROKES,
                                    "export.crf": 23})
    run = ex.run_export(cfg2, log=quiet)
    assert not run["skipped"] and any("export_hash" in s for s in run["stale"])


def test_changing_source_reencodes_and_silhouette_colors_do_not_stale_strokes(tmp_path):
    cfg, work, out = clip_strokes(tmp_path)
    ex.run_export(cfg, log=quiet)
    colors = load_pipeline(overrides={"paths.work_dir": str(work), "paths.out_dir": str(out), **STROKES,
                                      "export.foreground_color": "#112233"})
    assert ex.run_export(colors, log=quiet)["skipped"]        # warna siluet tidak dipakai jalur strokes
    sil = load_pipeline(overrides={"paths.work_dir": str(work), "paths.out_dir": str(out),
                                   "export.source": "silhouette"})
    run = ex.run_export(sil, log=quiet)
    assert not run["skipped"] and any("export_hash" in s or "strokes" in s for s in run["stale"])


def test_manifest_records_ffmpeg_version_but_it_is_not_in_the_hash(tmp_path):
    cfg, work, out = clip_strokes(tmp_path)
    ex.run_export(cfg, log=quiet)
    mpath = out / "meme_clip.export.json"
    m = read_json(mpath)
    assert m["ffmpeg"].lower().startswith("ffmpeg version") and m["ffmpeg"] == ex.ffmpeg_version()
    assert "ffmpeg" not in ex.MANIFEST_MATCH_KEYS
    edit_json(mpath, ffmpeg="ffmpeg version 0.0.1-lama")
    assert ex.run_export(cfg, log=quiet)["skipped"]              # versi lain ≠ basi (hanya penjelas)


def test_old_silhouette_manifest_is_not_stale_without_new_keys(tmp_path):
    cfg, _, out = te.make_clip(tmp_path)
    ex.run_export(cfg, log=quiet)
    mpath = out / "meme_clip.export.json"
    m = read_json(mpath)
    for k in ("ffmpeg", "strokes", "color_tags"):
        m.pop(k, None)
    m["output"].pop("color", None)
    mpath.write_text(json.dumps(m), encoding="utf-8")            # = bentuk manifest Phase 1
    assert ex.run_export(cfg, log=quiet)["skipped"]


def test_limit_writes_preview_without_svg_or_manifest_and_leaves_main(tmp_path):
    cfg, work, out = clip_strokes(tmp_path)
    ex.run_export(cfg, log=quiet)
    before = (em.sha256_file(out / "meme_clip.mp4"), em.sha256_file(out / "meme_clip.export.json"),
              sorted(p.name for p in (out / "svg" / "meme_clip").iterdir()))
    run = ex.run_export(cfg, limit=2, log=quiet)
    assert run["svg"] is None and run["output"].name == "meme_clip.limit2.mp4"
    assert ex.probe_output(run["output"])["frames"] == 2
    assert before == (em.sha256_file(out / "meme_clip.mp4"), em.sha256_file(out / "meme_clip.export.json"),
                      sorted(p.name for p in (out / "svg" / "meme_clip").iterdir()))
    assert not (out / "meme_clip.limit2.export.json").exists()


def test_limit_without_prior_run_creates_no_svg_folder(tmp_path):
    cfg, _, out = clip_strokes(tmp_path)
    ex.run_export(cfg, limit=2, log=quiet)
    assert not (out / "svg").exists()


# ── salinan SVG + perlindungan suntingan ───────────
def svg_dir(out: Path) -> Path:
    return out / "svg" / "meme_clip"


def test_svg_still_copied_when_mp4_is_up_to_date_and_restored_if_missing(tmp_path):
    cfg, work, out = clip_strokes(tmp_path)
    ex.run_export(cfg, log=quiet)
    (svg_dir(out) / "frame_00003.svg").unlink()
    run = ex.run_export(cfg, log=quiet)
    assert run["skipped"] and run["svg"]["copied"] == 1
    assert (svg_dir(out) / "frame_00003.svg").read_bytes() == (work / "strokes" / "frame_00003.svg").read_bytes()


def test_user_edited_svg_is_not_overwritten_only_restart_does(tmp_path):
    cfg, work, out = clip_strokes(tmp_path)
    ex.run_export(cfg, log=quiet)
    edited = svg_dir(out) / "frame_00001.svg"
    edited.write_text("<svg><!-- suntingan Rio --></svg>", encoding="utf-8")
    mp4_before = em.sha256_file(out / "meme_clip.mp4")
    with pytest.raises(StageError, match=r"frame_00001\.svg.*--restart") as e:
        ex.run_export(cfg, log=quiet)
    assert "disunting" in str(e.value) and edited.read_text(encoding="utf-8") == "<svg><!-- suntingan Rio --></svg>"
    assert em.sha256_file(out / "meme_clip.mp4") == mp4_before
    # strokes juga berubah: suntingan tetap dilindungi (beda dari yang dicatat)
    (work / "strokes" / "frame_00001.svg").write_text(svg_text(1, 128, 96, "<g/>"), encoding="utf-8")
    with pytest.raises(StageError, match="frame_00001"):
        ex.run_export(cfg, log=quiet)
    run = ex.run_export(cfg, restart=True, log=quiet)
    assert run["svg"]["copied"] >= 1
    assert edited.read_bytes() == (work / "strokes" / "frame_00001.svg").read_bytes()


def test_edit_check_runs_before_encode(tmp_path):
    cfg, work, out = clip_strokes(tmp_path)
    ex.run_export(cfg, log=quiet)
    (svg_dir(out) / "frame_00000.svg").write_text("edit", encoding="utf-8")
    edit_json(work / "strokes" / "manifest.json", created_utc="2026-10-05T00:00:00+00:00")     # → MP4 basi
    mtime = (out / "meme_clip.mp4").stat().st_mtime_ns
    with pytest.raises(StageError, match="frame_00000"):
        ex.run_export(cfg, log=quiet)
    assert (out / "meme_clip.mp4").stat().st_mtime_ns == mtime     # gagal cepat, MP4 tidak di-encode


def test_changed_strokes_unedited_svgs_are_recopied(tmp_path):
    cfg, work, out = clip_strokes(tmp_path)
    ex.run_export(cfg, log=quiet)
    make_strokes(work, cfg, tag="<g id='baru'/>", created_utc="2026-10-03T00:00:00+00:00")
    run = ex.run_export(cfg, log=quiet)
    assert run["svg"]["stale"] == te.N and run["svg"]["copied"] == te.N
    for i in range(te.N):
        name = f"frame_{i:05d}.svg"
        assert (svg_dir(out) / name).read_bytes() == (work / "strokes" / name).read_bytes()
    assert ex.run_export(cfg, log=quiet)["svg"]["copied"] == 0


def test_old_contract_strokes_rejected_then_new_svgs_recopied_as_stale_not_as_user_edit(tmp_path):
    """T-402: strokes T-203a / T-401 ditolak ("jalankan stylize"); setelah dihitung ulang (contract T-402, SVG berubah) salinan SVG lama di
    out/svg/<nama>/ disalin ulang sebagai BASI — bukan dianggap suntingan pengguna (tidak ada StageError, tanpa --restart)."""
    cfg, work, out = clip_strokes(tmp_path)
    ex.run_export(cfg, log=quiet)
    old_copy = {p.name: p.read_bytes() for p in svg_dir(out).glob("frame_*.svg")}
    for old in ("T-203a", "T-401"):
        edit_json(work / "strokes" / "manifest.json", contract=old)
        with pytest.raises(StageError, match=rf"contract '{old}'.*T-402.*stylize"):
            ex.run_export(cfg, log=quiet)
        assert {p.name: p.read_bytes() for p in svg_dir(out).glob("frame_*.svg")} == old_copy       # ditolak sebelum menyentuh SVG
    make_strokes(work, cfg, tag="<path d='M0 0 1 1 Z'/>", created_utc="2026-10-06T00:00:00+00:00")   # = stylize T-402 baru
    run = ex.run_export(cfg, log=quiet)
    assert run["svg"]["stale"] == te.N and run["svg"]["copied"] == te.N
    for i in range(te.N):
        name = f"frame_{i:05d}.svg"
        assert (svg_dir(out) / name).read_bytes() == (work / "strokes" / name).read_bytes() != old_copy[name]


def test_foreign_files_are_never_deleted_but_recorded_missing_ones_are_cleaned(tmp_path):
    cfg, work, out = clip_strokes(tmp_path)
    ex.run_export(cfg, log=quiet)
    d = svg_dir(out)
    (d / "catatan-rio.txt").write_text("milik pengguna", encoding="utf-8")
    (d / "frame_99999.svg").write_text("<svg>asing</svg>", encoding="utf-8")        # tidak dicatat penanda
    for p in work.glob("strokes/frame_0000[45].*"):                                   # klip menjadi lebih pendek
        p.unlink()
    meta = json.loads((work / "meta.json").read_text(encoding="utf-8"))
    meta["frame_count"] = te.N - 2
    (work / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    te.resync_stable_clip(work)
    make_strokes(work, cfg, n=te.N - 2)
    run = ex.run_export(cfg, log=quiet)
    assert run["svg"]["removed"] == 2 and not (d / "frame_00004.svg").exists() and not (d / "frame_00005.svg").exists()
    assert (d / "catatan-rio.txt").read_text(encoding="utf-8") == "milik pengguna"
    assert (d / "frame_99999.svg").exists()
    assert set(read_json(d / ex.SVG_MARKER)["files"]) == {f"frame_{i:05d}.svg" for i in range(te.N - 2)}


def test_edited_svg_beyond_shorter_clip_is_kept(tmp_path):
    cfg, work, out = clip_strokes(tmp_path)
    ex.run_export(cfg, log=quiet)
    d = svg_dir(out)
    (d / "frame_00005.svg").write_text("<svg>suntingan</svg>", encoding="utf-8")
    meta = json.loads((work / "meta.json").read_text(encoding="utf-8"))
    meta["frame_count"] = te.N - 1
    (work / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    te.resync_stable_clip(work)
    make_strokes(work, cfg, n=te.N - 1)
    run = ex.run_export(cfg, log=quiet)
    assert run["svg"]["kept"] == ["frame_00005.svg"] and (d / "frame_00005.svg").read_text(encoding="utf-8") \
        == "<svg>suntingan</svg>"


def test_svg_folder_without_marker_refused_unless_restart(tmp_path):
    cfg, work, out = clip_strokes(tmp_path)
    d = svg_dir(out)
    d.mkdir(parents=True)
    (d / "frame_00000.svg").write_text("<svg>tanpa penanda</svg>", encoding="utf-8")
    with pytest.raises(StageError, match=r"tanpa \.rotoscope-clip\.json.*--restart"):
        ex.run_export(cfg, log=quiet)
    assert not (out / "meme_clip.mp4").exists()
    ex.run_export(cfg, restart=True, log=quiet)
    assert (d / "frame_00000.svg").read_bytes() == (work / "strokes" / "frame_00000.svg").read_bytes()
    assert (d / ex.SVG_MARKER).is_file()


def test_svg_folder_of_other_clip_refused_even_with_restart(tmp_path):
    cfg, work, out = clip_strokes(tmp_path)
    d = svg_dir(out)
    d.mkdir(parents=True)
    (d / ex.SVG_MARKER).write_text(json.dumps({"clip": {"meta_sha256": "0" * 64, "source_path": "lain.mp4"},
                                               "files": {}}), encoding="utf-8")
    for restart in (False, True):
        with pytest.raises(StageError, match=r"video sumber LAIN.*lain\.mp4"):
            ex.run_export(cfg, restart=restart, log=quiet)
    assert not (out / "meme_clip.mp4").exists()


BOM = b"\xef\xbb\xbf"
BAD_MARKERS = {
    "bukan JSON": b"ini bukan json {",
    "bukan objek": b"[1, 2]",
    "tanpa clip": b'{"files": {}}',
    "files salah": b'{"clip": {"source_path": "x.mp4"}, "files": [1]}',
    "bukan UTF-8": b"\xff\xfe\x00\x80",
}


def test_marker_is_written_without_bom_and_read_with_bom(tmp_path):
    cfg, work, out = clip_strokes(tmp_path)
    ex.run_export(cfg, log=quiet)
    marker = svg_dir(out) / ex.SVG_MARKER
    raw = marker.read_bytes()
    assert not raw.startswith(BOM)
    marker.write_bytes(BOM + raw)                       # editor / PowerShell menambah BOM
    run = ex.run_export(cfg, log=quiet)
    assert run["skipped"] and run["svg"]["copied"] == 0   # penanda ber-BOM tetap dikenali
    (svg_dir(out) / "frame_00001.svg").write_text("suntingan", encoding="utf-8")
    with pytest.raises(StageError, match=r"frame_00001\.svg.*--restart"):   # perlindungan suntingan tetap bekerja
        ex.run_export(cfg, log=quiet)


@pytest.mark.parametrize("bad", list(BAD_MARKERS))
def test_corrupt_marker_is_not_the_same_as_no_marker(tmp_path, bad):
    cfg, work, out = clip_strokes(tmp_path)
    ex.run_export(cfg, log=quiet)
    d = svg_dir(out)
    (d / ex.SVG_MARKER).write_bytes(BAD_MARKERS[bad])
    mp4_before = em.sha256_file(out / "meme_clip.mp4")
    with pytest.raises(StageError, match=r"ada tetapi (tidak terbaca|bukan|skema salah)") as e:
        ex.run_export(cfg, log=quiet)
    assert "berisi berkas tanpa" not in str(e.value)       # bukan pesan 'folder tanpa penanda'
    assert (d / ex.SVG_MARKER).read_bytes() == BAD_MARKERS[bad]
    assert em.sha256_file(out / "meme_clip.mp4") == mp4_before


def test_corrupt_marker_restart_allowed_only_if_every_svg_is_identical(tmp_path):
    cfg, work, out = clip_strokes(tmp_path)
    ex.run_export(cfg, log=quiet)
    d = svg_dir(out)
    marker = d / ex.SVG_MARKER
    marker.write_bytes(b"rusak")
    (d / "catatan.txt").write_text("bukan svg: tidak dihitung", encoding="utf-8")
    ex.run_export(cfg, restart=True, log=quiet)          # semua SVG identik → aman, penanda diperbaiki
    assert ex.read_svg_marker(d) == (json.loads(marker.read_text(encoding="utf-8")), None)
    assert not marker.read_bytes().startswith(BOM)
    assert (d / "catatan.txt").exists()
    # satu SVG disunting → ditolak walau --restart, pesan menyebut berkas itu; tidak ada yang ditimpa
    marker.write_bytes(b"rusak")
    (d / "frame_00002.svg").write_text("suntingan Rio", encoding="utf-8")
    with pytest.raises(StageError, match=r"frame_00002\.svg.*walau --restart"):
        ex.run_export(cfg, restart=True, log=quiet)
    assert (d / "frame_00002.svg").read_text(encoding="utf-8") == "suntingan Rio" and marker.read_bytes() == b"rusak"
    # SVG asing (tidak ada di sumber) juga dianggap berbeda
    (d / "frame_00002.svg").write_bytes((work / "strokes" / "frame_00002.svg").read_bytes())
    (d / "gambar-rio.svg").write_text("<svg/>", encoding="utf-8")
    with pytest.raises(StageError, match=r"gambar-rio\.svg"):
        ex.run_export(cfg, restart=True, log=quiet)


def test_corrupt_marker_without_restart_refused_even_if_all_identical(tmp_path):
    cfg, work, out = clip_strokes(tmp_path)
    ex.run_export(cfg, log=quiet)
    (svg_dir(out) / ex.SVG_MARKER).write_bytes(b"rusak")
    with pytest.raises(StageError, match=r"ada tetapi bukan JSON valid.*--restart"):
        ex.run_export(cfg, log=quiet)


def test_stale_tmp_files_are_cleaned_but_user_tmp_kept(tmp_path):
    cfg, work, out = clip_strokes(tmp_path)
    ex.run_export(cfg, log=quiet)
    d = svg_dir(out)
    (d / "frame_00002.svg.tmp").write_text("setengah", encoding="utf-8")
    (d / "punya-rio.tmp").write_text("milik pengguna", encoding="utf-8")
    ex.run_export(cfg, log=quiet)
    assert not (d / "frame_00002.svg.tmp").exists() and (d / "punya-rio.tmp").exists()


def test_svg_copy_verification_failure_is_reported(tmp_path, monkeypatch):
    cfg, work, out = clip_strokes(tmp_path)
    monkeypatch.setattr(ex, "write_bytes_atomic", lambda path, data: path.write_bytes(data[:-3]))
    with pytest.raises(StageError, match=r"verifikasi salinan SVG"):
        ex.run_export(cfg, log=quiet)


def test_missing_source_svg(tmp_path):
    cfg, work, out = clip_strokes(tmp_path)
    (work / "strokes" / "frame_00001.svg").unlink()
    with pytest.raises(StageError, match=r"frame_00001\.svg.*stylize"):
        ex.run_export(cfg, log=quiet)


# ── determinisme ───────────────────────────────────
def test_mp4_and_svg_identical_across_runs_from_scratch(tmp_path):
    cfg, work, out = clip_strokes(tmp_path)
    ex.run_export(cfg, log=quiet)
    first = {p.name: em.sha256_file(p) for p in [out / "meme_clip.mp4", *sorted(svg_dir(out).glob("*.svg"))]}
    shutil.rmtree(out)
    ex.run_export(cfg, log=quiet)
    second = {p.name: em.sha256_file(p) for p in [out / "meme_clip.mp4", *sorted(svg_dir(out).glob("*.svg"))]}
    assert first == second and len(first) == te.N + 1


def test_main_cli_strokes_exit_codes(tmp_path, capsys):
    cfg, work, out = clip_strokes(tmp_path)
    conf = tmp_path / "c.yaml"
    conf.write_text(f"paths:\n  work_dir: '{work}'\n  out_dir: '{out}'\nexport:\n  source: strokes\n", encoding="utf-8")
    assert ex.main(["--config", str(conf)]) == 0
    (work / "strokes" / "manifest.json").unlink()
    assert ex.main(["--config", str(conf)]) == 1 and "stylize" in capsys.readouterr().err


# ── metrik objektif pada klip nyata (dilewati bila tidak ada) ──
def _real(clip: str):
    d = REAL_CLIPS / clip
    m = d / "strokes" / "manifest.json"
    if not (m.is_file() and (d / "meta.json").is_file()):
        return None
    # strokes klip nyata dengan contract lama (belum dihitung ulang oleh `run`) = basi → dilewati (export menolaknya, lihat test lain)
    return d if read_json(m).get("contract") in ex.SUPPORTED_STROKES_CONTRACTS else None


@pytest.mark.parametrize("clip", ["test_short", "test"])
def test_real_clip_objective_metrics(tmp_path, clip):
    work = _real(clip)
    if work is None:
        pytest.skip(f"klip nyata {clip} tidak ada")
    out = tmp_path / "out"
    cfg = load_pipeline(overrides={"paths.work_dir": str(work), "paths.out_dir": str(out), **STROKES})
    sm = read_json(work / "strokes" / "manifest.json")
    n = read_json(work / "meta.json")["frame_count"]
    run = ex.run_export(cfg, log=quiet)
    mp4 = run["output"]
    st = em.ffprobe_stream(mp4)
    osz = sm["output_size"]
    assert (int(st["width"]), int(st["height"])) == (osz["width"], osz["height"])        # (a)
    assert int(st["nb_read_packets"]) == n
    worst_psnr, worst_mae, worst_max, worst_paper, worst_ink = 99.0, 0.0, 0, 0.0, 0.0
    for i in em.sample_indices(n, step=20):                                                # (b)(c)
        rep = em.frame_report(em.read_png_rgb(work / "strokes" / f"frame_{i:05d}.png"), em.decode_rgb(mp4, i))
        worst_psnr, worst_mae, worst_max = min(worst_psnr, rep["psnr"]), max(worst_mae, rep["mae"]), \
            max(worst_max, rep["max_diff"])
        if rep["paper_delta"] is not None:
            worst_paper = max(worst_paper, max(abs(x) for x in rep["paper_delta"]))
        if rep["ink_delta"] is not None:
            worst_ink = max(worst_ink, max(abs(x) for x in rep["ink_delta"]))
    assert worst_psnr >= em.PSNR_MIN_DB and worst_mae <= em.INK_MAE_MAX and worst_max <= em.INK_MAX_DIFF
    assert worst_paper <= em.COLOR_TOLERANCE and worst_ink <= em.INK_FLAT_TOLERANCE
    svg = ex.svg_dir_for(out, read_json(work / "meta.json")["source_path"])               # (d)
    files = sorted(svg.glob("frame_*.svg"))
    assert len(files) == n
    assert all(em.sha256_file(f) == em.sha256_file(work / "strokes" / f.name) for f in files)
