"""Export [6] source strokes dengan kertas bertekstur (T-404a Opsi B): PNG [5] (kertas datar) → invers LUT → kertas bertekstur → encode. Regresi byte-identik
MP4 T-403 untuk kertas datar, manifest / basi / hash, error, SVG tidak berubah, jendela (--limit / --from), `--style` (export.main + cli), pre-flight run.
Unit paper.py: tests/test_paper.py. Metrik MP4 bertekstur: tests/export_metrics.py."""

from __future__ import annotations

import dataclasses
import shutil
from pathlib import Path

import cv2
import export_metrics as em
import numpy as np
import pytest
import stylize_metrics as sm
import test_export as te
from test_export_strokes import SCALE, STROKES, edit_json, make_strokes, quiet, read_json

from rotoscope import cli
from rotoscope import export as ex
from rotoscope import paper as pap
from rotoscope import stylize as sty
from rotoscope.config import ConfigError, load_style
from rotoscope.stage_common import StageError

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="ffmpeg/ffprobe tidak ada di PATH")

PAPER, INK = "#f4f1ea", "#1a1a1a"


def levels_for(i: int, w: int, h: int) -> np.ndarray:
    """Level tinta uint8 sintetis: batang berundak 1..255 yang bergeser + kotak level tengah + garis tipis (level rendah)."""
    lv = np.zeros((h, w), np.uint8)
    lv[60:76, 10 + 4 * i:90 + 4 * i] = np.linspace(1, 255, 80).astype(np.uint8)[None, :]
    lv[8:40, 20:52] = 128
    lv[90, 5:120] = 7
    return lv


def lut_png(lv: np.ndarray) -> np.ndarray:
    g = dataclasses.make_dataclass("G", ["paper", "ink"])(pap.hex_rgb(PAPER), pap.hex_rgb(INK))
    return sty.ink_lut(g)[lv]


def make_lut_strokes(work: Path, cfg, n: int = te.N) -> None:
    """strokes/ sintetis yang dibuat dengan LUT (seperti [5]) + style_params paper.color / stroke.color di manifest."""
    make_strokes(work, cfg, n=n)
    sdir = work / "strokes"
    ow, oh = te.W * SCALE, te.H * SCALE
    for i in range(n):
        ok, buf = cv2.imencode(".png", cv2.cvtColor(lut_png(levels_for(i, ow, oh)), cv2.COLOR_RGB2BGR))
        assert ok
        (sdir / f"frame_{i:05d}.png").write_bytes(buf.tobytes())
    edit_json(sdir / "manifest.json", style_params={"paper.color": PAPER, "stroke.color": INK})


def write_tex(path: Path, seed: int = 0) -> Path:
    rng = np.random.RandomState(seed)
    assert cv2.imwrite(str(path), np.clip(np.rint(rng.normal(180, 12, (200, 300))), 1, 255).astype(np.uint8))
    return path


def pstyle(tex: Path, **ov):
    return load_style(None, overrides={"paper.texture_image": str(tex), "paper.texture_opacity": 0.5, "paper.texture_gain": 3.0,
                                       "paper.vignette": 0.1, **ov})


@pytest.fixture
def clip(tmp_path):
    cfg, work, out = te.make_clip(tmp_path, **STROKES)
    make_lut_strokes(work, cfg)
    return cfg, work, out, write_tex(tmp_path / "tex.png")


def mp4_of(out: Path) -> Path:
    return out / "meme_clip.mp4"


# ── Susunan + metrik ───────────────────────────────
def test_textured_export_frames_equal_composition_and_pass_metrics(clip):
    cfg, work, out, tex = clip
    st = pstyle(tex)
    run = ex.run_export(cfg, style=st, log=quiet)
    assert not run["skipped"] and mp4_of(out).is_file()
    size = (te.W * SCALE, te.H * SCALE)
    layer = pap.render_paper(*size, st)
    lm = pap.LevelMap(pap.hex_rgb(PAPER), pap.hex_rgb(INK))
    for i in (0, te.N // 2, te.N - 1):
        exp = pap.compose_textured(em.read_png_rgb(work / "strokes" / f"frame_{i:05d}.png"), lm, layer, pap.hex_rgb(INK))
        f = sm.recover_f(exp, layer.f32, pap.hex_rgb(INK))
        rep = em.textured_report(exp, em.decode_rgb(mp4_of(out), i), f)
        assert rep["psnr"] >= 30 and max(abs(x) for x in rep["paper_bias"]) <= em.PAPER_BIAS_MAX, rep
    tex_bytes = mp4_of(out).read_bytes()
    ex.run_export(cfg, restart=True, log=quiet)                         # tanpa style = kertas datar
    assert mp4_of(out).read_bytes() != tex_bytes


def test_flat_variants_give_mp4_byte_identical_to_no_style(clip):
    """Regresi T-403: kertas datar (style None, enabled false, opacity 0 + vignette 0, gain 0 + vignette 0) → MP4 byte-identik."""
    cfg, work, out, tex = clip
    ex.run_export(cfg, log=quiet)
    ref = mp4_of(out).read_bytes()
    m_ref = read_json(out / "meme_clip.export.json")
    assert m_ref["paper"] is None and "paper_info" not in m_ref
    for ov in ({"paper.enabled": False, "paper.vignette": 0.3}, {"paper.texture_opacity": 0.0, "paper.vignette": 0.0},
               {"paper.texture_gain": 0.0, "paper.vignette": 0.0}):
        st = pstyle(tex, **ov)
        assert pap.paper_flat(st)
        run = ex.run_export(cfg, style=st, log=quiet)
        assert run["skipped"] and mp4_of(out).read_bytes() == ref                        # manifest sama → dilewati
        ex.run_export(cfg, restart=True, style=st, log=quiet)
        assert mp4_of(out).read_bytes() == ref                                           # di-encode ulang → byte sama
        assert read_json(out / "meme_clip.export.json")["paper"] is None


def test_strokes_and_svg_untouched_by_textured_export(clip):
    cfg, work, out, tex = clip
    before = {p.name: p.read_bytes() for p in sorted((work / "strokes").iterdir())}
    ex.run_export(cfg, style=pstyle(tex), log=quiet)
    assert {p.name: p.read_bytes() for p in sorted((work / "strokes").iterdir())} == before
    for i in range(te.N):
        name = f"frame_{i:05d}.svg"
        assert (out / "svg" / "meme_clip" / name).read_bytes() == (work / "strokes" / name).read_bytes()      # SVG tetap kertas datar


# ── Manifest / basi ────────────────────────────────
def test_manifest_paper_block_and_staleness(clip):
    cfg, work, out, tex = clip
    st = pstyle(tex)
    ex.run_export(cfg, style=st, log=quiet)
    m = read_json(out / "meme_clip.export.json")
    sha = pap.paper_ref(st)["texture_sha256"]
    assert m["paper"]["texture_sha256"] == sha and m["paper"]["texture_opacity"] == 0.5 and m["paper"]["texture_gain"] == 3.0
    assert m["paper_info"]["flat"] is False and m["paper_info"]["fit"] == "native" and m["paper_info"]["rotated"] is False and m["paper_info"]["texture_sha256"] == sha
    assert "paper" in ex.MANIFEST_MATCH_KEYS and "paper_info" not in ex.MANIFEST_MATCH_KEYS
    strokes_before = {p.name: p.read_bytes() for p in (work / "strokes").iterdir()}
    assert ex.run_export(cfg, style=st, log=quiet)["skipped"]                            # sama → dilewati
    logs: list[str] = []
    run = ex.run_export(cfg, style=pstyle(tex, **{"paper.texture_opacity": 0.4}), log=logs.append)
    assert not run["skipped"] and any("paper.texture_opacity" in x for x in run["stale"])        # parameter kertas berubah → MP4 basi
    run = ex.run_export(cfg, style=pstyle(tex, **{"paper.vignette": 0.0}), log=logs.append)
    assert not run["skipped"] and any("paper.vignette" in x for x in run["stale"])
    write_tex(tex, seed=5)                                                               # isi berkas tekstur berubah (path sama)
    run = ex.run_export(cfg, style=pstyle(tex, **{"paper.vignette": 0.0}), log=logs.append)
    assert not run["skipped"] and any("texture_sha256" in x for x in run["stale"])
    run = ex.run_export(cfg, log=logs.append)                                            # kembali datar → basi
    assert not run["skipped"] and read_json(out / "meme_clip.export.json")["paper"] is None
    assert {p.name: p.read_bytes() for p in (work / "strokes").iterdir()} == strokes_before          # strokes tidak pernah disentuh


def test_color_mismatch_and_foreign_png_colors_rejected(clip):
    cfg, work, out, tex = clip
    with pytest.raises(StageError, match=r"paper\.color style .* ≠ paper\.color strokes"):
        ex.run_export(cfg, style=pstyle(tex, **{"paper.color": "#ffffff"}), log=quiet)
    edit_json(work / "strokes" / "manifest.json", style_params={})
    with pytest.raises(StageError, match="style_params paper.color / stroke.color"):
        ex.run_export(cfg, style=pstyle(tex), log=quiet)
    edit_json(work / "strokes" / "manifest.json", style_params={"paper.color": PAPER, "stroke.color": INK})
    img = lut_png(levels_for(0, te.W * SCALE, te.H * SCALE)).copy()
    img[3, 3] = (200, 30, 30)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    assert ok
    (work / "strokes" / "frame_00000.png").write_bytes(buf.tobytes())
    with pytest.raises(StageError, match="di luar LUT"):
        ex.run_export(cfg, style=pstyle(tex), log=quiet)
    assert not mp4_of(out).exists()                                                      # gagal → tidak ada MP4 setengah jadi


def test_missing_texture_file_is_stage_error_at_export(clip, tmp_path):
    cfg, work, out, tex = clip
    st = pstyle(tex)
    gone = dataclasses.replace(st, paper=dataclasses.replace(st.paper, texture_image=tmp_path / "hilang.png"))
    with pytest.raises(StageError, match="hilang.png"):
        ex.run_export(cfg, style=gone, log=quiet)


# ── Jendela + main + cli ───────────────────────────
def test_window_preview_uses_same_path_and_does_not_touch_main_mp4(clip):
    cfg, work, out, tex = clip
    st = pstyle(tex)
    ex.run_export(cfg, style=st, log=quiet)
    main_bytes, man = mp4_of(out).read_bytes(), (out / "meme_clip.export.json").read_bytes()
    run = ex.run_export(cfg, limit=3, start=1, style=st, log=quiet)
    assert run["output"].name == "meme_clip.preview_1-3.mp4" and run["probe"]["frames"] == 3 and run["svg"] is None
    assert mp4_of(out).read_bytes() == main_bytes and (out / "meme_clip.export.json").read_bytes() == man
    layer = pap.render_paper(te.W * SCALE, te.H * SCALE, st)
    lm = pap.LevelMap(pap.hex_rgb(PAPER), pap.hex_rgb(INK))
    exp = pap.compose_textured(em.read_png_rgb(work / "strokes" / "frame_00002.png"), lm, layer, pap.hex_rgb(INK))
    assert em.psnr(exp, em.decode_rgb(run["output"], 1)) >= 30                           # frame ke-2 jendela = frame 2 klip


def style_file(tmp_path: Path, tex: Path) -> Path:
    p = tmp_path / "style.yaml"
    p.write_text(f'paper:\n  texture_image: "{tex.as_posix()}"\n  texture_opacity: 0.5\n  texture_gain: 3.0\n  vignette: 0.1\n', encoding="utf-8")
    return p


def test_export_main_with_style_flag(clip, tmp_path):
    cfg, work, out, tex = clip
    args = ["--work-dir", str(work)]
    cfg_path = tmp_path / "pipe.yaml"
    cfg_path.write_text(f'paths:\n  out_dir: "{out.as_posix()}"\nexport:\n  source: "strokes"\n', encoding="utf-8")
    assert ex.main([*args, "--config", str(cfg_path), "--style", str(style_file(tmp_path, tex))]) == 0
    assert read_json(out / "meme_clip.export.json")["paper"]["texture_opacity"] == 0.5
    assert ex.main([*args, "--config", str(cfg_path), "--style", str(tmp_path / "tidak_ada.yaml")]) == 1


def test_cli_passes_style_to_export_and_preflight_checks_texture(clip, tmp_path):
    cfg, work, out, tex = clip
    ctx = cli.Ctx(video=Path("v.mp4"), cfg=cfg, work_dir=work, style=Path("s.yaml"))
    a = cli.stage_args("export", ctx)
    assert a[a.index("--style") + 1] == "s.yaml"
    assert "--style" in cli.stage_args("stylize", ctx) and "--style" not in cli.stage_args("vectorize", ctx)
    assert "--style" in cli.preview_args("export", ctx, 0, 5) and "--style" in cli.preview_args("stylize", ctx, 0, 5)
    assert "--style" not in cli.stage_args("export", cli.Ctx(video=Path("v.mp4"), cfg=cfg, work_dir=work))
    cli.preflight_style(style_file(tmp_path, tex))                                        # ada + terdekode → lolos
    missing = tmp_path / "m.yaml"
    missing.write_text('paper:\n  texture_image: "hilang.jpg"\n  texture_opacity: 0.35\n', encoding="utf-8")
    with pytest.raises(cli.CliError, match="texture_image"):
        cli.preflight_style(missing)                                                      # berkas hilang → berhenti sebelum stage mana pun
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"bukan gambar")
    corrupt = tmp_path / "c.yaml"
    corrupt.write_text(f'paper:\n  texture_image: "{bad.as_posix()}"\n  texture_opacity: 0.35\n', encoding="utf-8")
    with pytest.raises(cli.CliError, match="bukan gambar"):
        cli.preflight_style(corrupt)
    off = tmp_path / "o.yaml"
    off.write_text('paper:\n  texture_image: "hilang.jpg"\n  texture_opacity: 0.0\n', encoding="utf-8")
    cli.preflight_style(off)                                                              # kertas datar: berkas tidak dibutuhkan
    assert pap.paper_ref(load_style(off)) is None
    with pytest.raises(ConfigError):
        load_style(missing)
