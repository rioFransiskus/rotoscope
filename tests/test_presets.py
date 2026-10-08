"""T-406 preset library: resolver nama / path, kunci config `style`, `{style}`, peringatan penimpaan, kelengkapan preset,
zona mati tepi pass k (tinta tepi = 0). Tanpa GPU / torch."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

import export_metrics as em
import jitter_metrics as jm
import test_cli as tc
import test_export as te
import test_export_strokes as tes
from rotoscope import cli
from rotoscope import export as ex
from rotoscope import stylize as sty
from rotoscope.config import (
    DEFAULT_STYLE_NAME, ConfigError, load_pipeline, load_style, project_root, resolve_style, style_name,
)

ROOT = Path(__file__).resolve().parents[1]
PRESET_DIR = ROOT / "configs" / "styles"
PRESETS = ("rough-sketch", "clean-line", "heavy-marker", "pencil-light")
NEW_PRESETS = PRESETS[1:]
rec = tc.rec                                                    # fixture subproses palsu test_cli
quiet = lambda m: None                                          # noqa: E731


# ── Resolver ───────────────────────────────────────
def test_name_resolves_under_project_root_not_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in PRESETS:
        assert resolve_style(name) == project_root() / "configs" / "styles" / f"{name}.yaml"
    (tmp_path / "configs" / "styles").mkdir(parents=True)
    (tmp_path / "configs" / "styles" / "clean-line.yaml").write_text("stroke:\n  width_base: 1\n", encoding="utf-8")   # tiruan di cwd
    assert resolve_style("clean-line") == project_root() / "configs" / "styles" / "clean-line.yaml"                       # tidak membajak


def test_default_and_name_equals_default_path_equals_default_code(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    p = resolve_style(None)
    assert p == resolve_style(DEFAULT_STYLE_NAME) == resolve_style("rough-sketch") == PRESET_DIR / "rough-sketch.yaml"
    assert load_style(p) == load_style(str(PRESET_DIR / "rough-sketch.yaml")) == load_style(None)
    assert style_name(p) == "rough-sketch" and style_name(None) == DEFAULT_STYLE_NAME


def test_unknown_name_lists_presets_and_is_case_sensitive():
    with pytest.raises(ConfigError) as e:
        resolve_style("neon-glow")
    msg = str(e.value)
    assert all(n in msg for n in PRESETS) and "neon-glow" in msg
    with pytest.raises(ConfigError, match=r"maksudnya 'clean-line'"):
        resolve_style("Clean-Line")                       # Windows: berkas ada, tetapi nama peka huruf besar-kecil
    for bad in ("clean.line", "clean line", "CLEAN-LINE"):
        with pytest.raises(ConfigError, match="bukan preset yang dikenal"):
            resolve_style(bad)


def test_path_forms(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    own = tmp_path / "mine.yaml"
    own.write_text("stroke:\n  width_base: 3\n", encoding="utf-8")
    assert resolve_style("mine.yaml") == Path("mine.yaml")                 # berakhiran .yaml = path (relatif ke cwd)
    assert resolve_style("./mine.yaml") == Path("./mine.yaml") and resolve_style(str(own)) == own and resolve_style(own) == own
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "clean-line.yaml").write_text("", encoding="utf-8")
    assert resolve_style("sub/clean-line.yaml") == Path("sub/clean-line.yaml")   # ada pemisah = path, bukan nama
    assert style_name(resolve_style("sub/clean-line.yaml")) == "clean-line"
    with pytest.raises(ConfigError, match="tidak ditemukan"):
        resolve_style("nope.yaml")
    with pytest.raises(ConfigError, match="tidak ditemukan"):
        resolve_style("sub/tidak-ada.yaml")
    with pytest.raises(ConfigError, match="kosong"):
        resolve_style("  ")


def test_default_name_without_file_falls_back_to_code_default(tmp_path):
    (tmp_path / "configs" / "styles").mkdir(parents=True)
    assert resolve_style("rough-sketch", root=tmp_path) is None              # pipeline jalan tanpa YAML
    with pytest.raises(ConfigError, match="bukan preset"):
        resolve_style("clean-line", root=tmp_path)


# ── Preset: lengkap, valid, berbeda ────────────────
def key_paths(d, prefix=""):
    out = set()
    for k, v in d.items():
        if isinstance(v, dict) and k != "by_type" and not set(v) <= {"width_scale", "opacity_scale", "taper_ends"}:
            out |= key_paths(v, f"{prefix}{k}.")
        elif isinstance(v, dict):
            for kk, vv in v.items():
                out |= key_paths(vv, f"{prefix}{k}.{kk}.") if isinstance(vv, dict) else {f"{prefix}{k}.{kk}"}
        else:
            out.add(prefix + k)
    return out


def load_raw(name: str) -> dict:
    return yaml.safe_load((PRESET_DIR / f"{name}.yaml").read_text(encoding="utf-8"))


def test_presets_exist_and_are_listed():
    from rotoscope.config import list_presets
    assert set(PRESETS) <= set(list_presets())


@pytest.mark.parametrize("name", NEW_PRESETS)
def test_every_preset_has_exactly_the_keys_of_rough_sketch(name):
    ref = key_paths(load_raw("rough-sketch"))
    assert len(ref) > 40 and key_paths(load_raw(name)) == ref               # eksplisit, lengkap, tanpa kunci ekstra
    load_style(PRESET_DIR / f"{name}.yaml")                                  # lolos validasi


def test_style_hashes_differ_and_rough_sketch_is_default_code():
    hashes = {n: sty.params_hash(sty.style_params(load_style(resolve_style(n)))[0]) for n in PRESETS}
    assert len(set(hashes.values())) == 4
    assert hashes["rough-sketch"] == sty.params_hash(sty.style_params(load_style(None))[0])


@pytest.mark.parametrize("name", PRESETS)
def test_preset_rules_jitter_off_fold_and_offset_ratio(name):
    s = load_style(resolve_style(name))
    assert s.jitter.amplitude == 0.0                                          # keputusan Rio 6
    assert sty.fold_warning(s) is None
    if name != "rough-sketch" and s.multipass.enabled:
        assert s.multipass.offset / s.stroke.width_base <= 0.6 + 1e-9         # pegangan Rio (T-406 butir 2)
    g = sty.make_geometry(s, 480, 854)
    assert g.out_w == 1080 and g.passes == (s.multipass.passes if s.multipass.enabled else 1)


def test_no_new_parameters_beyond_rough_sketch():
    from rotoscope.config import StyleConfig, _defaults
    assert key_paths(_defaults(StyleConfig)) == key_paths(load_raw("rough-sketch"))


# ── Kunci config `style` + pemilihan ───────────────
def test_config_style_key_default_and_validation(tmp_path):
    assert load_pipeline().style == "rough-sketch"
    assert load_pipeline(overrides={"style": "clean-line"}).style == "clean-line"
    with pytest.raises(ConfigError, match="style"):
        load_pipeline(overrides={"style": ""})
    with pytest.raises(ConfigError, match="kontrol"):
        load_pipeline(overrides={"style": "a\rb"})
    with pytest.raises(ConfigError, match="style"):
        load_pipeline(overrides={"style": 3})


def test_flag_beats_config_and_config_beats_default(tmp_path):
    cfg = load_pipeline(overrides={"style": "clean-line"})
    assert cli.pick_style(None, cfg) == (PRESET_DIR / "clean-line.yaml", "clean-line")
    assert cli.pick_style("heavy-marker", cfg) == (PRESET_DIR / "heavy-marker.yaml", "heavy-marker")      # flag mengalahkan config
    assert cli.pick_style(None, load_pipeline()) == (PRESET_DIR / "rough-sketch.yaml", "rough-sketch")
    with pytest.raises(cli.CliError, match="bukan preset"):
        cli.pick_style("neon", cfg)
    with pytest.raises(cli.CliError, match="bukan preset"):
        cli.pick_style(None, load_pipeline(overrides={"style": "neon"}))


def test_stage_mains_use_the_same_resolver(monkeypatch, tmp_path):
    """stylize.main dan export.main memilih style lewat resolve_style (satu jalur): nama preset diterima, nama tak dikenal = exit 1."""
    seen = []
    monkeypatch.setattr(sty, "resolve_style", lambda ref: seen.append(ref) or resolve_style(ref))
    monkeypatch.setattr(ex, "resolve_style", lambda ref: seen.append(ref) or resolve_style(ref))
    assert sty.main(["--work-dir", str(tmp_path), "--style", "neon"]) == 1
    assert ex.main(["--work-dir", str(tmp_path), "--style", "neon"]) == 1
    assert seen == ["neon", "neon"]


# ── {style} di export.filename ─────────────────────
def test_style_placeholder_filename_and_svg_dir():
    src = "/a/b c.mp4"
    assert ex.resolve_filename("{source}.mp4", src) == "b_c.mp4" == ex.resolve_filename("{source}.mp4", src, style="clean-line")
    assert ex.resolve_filename("{source}_{style}.mp4", src, style="clean-line") == "b_c_clean-line.mp4"
    assert ex.resolve_filename("{style}.mp4", src, style="my style!") == "my_style.mp4"            # disanitasi
    assert ex.resolve_filename("{source}_{style}.mp4", src, 5, style="pencil-light") == "b_c_pencil-light.limit5.mp4"
    assert ex.resolve_filename("{source}_{style}.mp4", src, 10, 73, "pencil-light") == "b_c_pencil-light.preview_73-82.mp4"
    out = Path("out")
    assert ex.svg_dir_for(out, src) == out / "svg" / "b_c"                                          # default tidak berubah
    assert ex.svg_dir_for(out, src, "{source}.mp4", "heavy-marker") == out / "svg" / "b_c"
    assert ex.svg_dir_for(out, src, "{source}_{style}.mp4", "heavy-marker") == out / "svg" / "b_c_heavy-marker"
    assert ex.svg_dir_for(out, src, "final.mp4", "heavy-marker") == out / "svg" / "final"            # stem hasil


def test_style_placeholder_validation():
    load_pipeline(overrides={"export.filename": "{source}_{style}.mp4"})
    load_pipeline(overrides={"export.filename": "{style}.mp4"})
    with pytest.raises(ConfigError, match="export.filename"):
        load_pipeline(overrides={"export.filename": "{preset}.mp4"})


# ── Pre-flight + run (CLI palsu) ───────────────────
def test_run_forwards_resolved_name_and_preflight_unknown_runs_nothing(rec, tmp_path, capsys):
    video = tc.make_video(tmp_path)
    assert tc.run_cli("run", str(video), "--style", "clean-line") == 0
    for stage in ("stylize", "export"):
        a = rec.args(stage)
        assert Path(a[a.index("--style") + 1]) == PRESET_DIR / "clean-line.yaml"
    rec.calls.clear()
    assert tc.run_cli("run", str(video), "--style", "neon") == 1
    assert rec.calls == []
    err = capsys.readouterr().err
    assert all(n in err for n in PRESETS)


def test_run_without_flag_does_not_forward_style(rec, tmp_path):
    assert tc.run_cli("run", str(tc.make_video(tmp_path))) == 0
    assert all("--style" not in rec.args(s) for s in rec.stages)


def test_preflight_uses_config_style_and_style_in_target(rec, tmp_path, capsys):
    conf = tc.write_conf(tmp_path, 'style: "neon"\n')
    assert tc.run_cli("run", str(tc.make_video(tmp_path)), "--config", str(conf)) == 1 and rec.calls == []
    # target export per style: MP4 video LAIN dengan nama {source}_{style} diperiksa dengan style terpilih
    conf = tc.write_conf(tmp_path, 'export:\n  filename: "{source}_{style}.mp4"\n')
    video = tc.make_video(tmp_path / "b")
    out = tmp_path / "out"
    out.mkdir()
    (out / "clip_heavy-marker.mp4").write_bytes(b"x")
    (out / "clip_heavy-marker.export.json").write_text(json.dumps({"clip": {"source_path": str(tmp_path / "a" / "clip.mp4")}}), encoding="utf-8")
    assert tc.run_cli("run", str(video), "--config", str(conf), "--style", "clean-line") == 0       # nama lain → tidak bentrok
    rec.calls.clear()
    assert tc.run_cli("run", str(video), "--config", str(conf), "--style", "heavy-marker") == 1       # nama sama, video lain
    assert rec.calls == [] and "sudah ada" in capsys.readouterr().err


def test_preview_target_name_includes_style(rec, tmp_path, capsys):
    conf = tc.write_conf(tmp_path, 'export:\n  filename: "{source}_{style}.mp4"\n')
    video = tc.make_video(tmp_path)
    try:
        tc.run_cli("run", str(video), "--config", str(conf), "--style", "pencil-light", "--preview", "5", "--from", "3")
    except Exception:                                                          # seg/ depth/ sintetis tidak ada: gagal SESUDAH baris pembuka
        pass
    assert "clip_pencil-light.preview_3-7.mp4" in capsys.readouterr().out


def test_torch_not_imported_by_config_stylize_export_cli():
    code = ("import sys, rotoscope.config, rotoscope.stylize, rotoscope.export, rotoscope.cli; "
            "rotoscope.config.resolve_style('clean-line'); sys.exit(1 if 'torch' in sys.modules else 0)")
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


# ── Export nyata (sintetis): tabrakan nama, peringatan, SVG terpisah ──
needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="ffmpeg/ffprobe tidak ada")


def warnings_of(lines):
    return [m for m in lines if m.startswith("PERINGATAN") and "export.filename" in m]


@needs_ffmpeg
def test_overwrite_warning_once_without_style_placeholder(tmp_path):
    cfg, work, out = tes.clip_strokes(tmp_path)
    log = []
    ex.run_export(cfg, style_label="clean-line", log=log.append)
    assert warnings_of(log) == []                                                  # pertama kali: tidak ada yang ditimpa
    manifest = json.loads((out / "meme_clip.export.json").read_text(encoding="utf-8"))
    assert manifest["style"] == "clean-line"
    log.clear()
    ex.run_export(cfg, style_label="clean-line", log=log.append)
    assert warnings_of(log) == []                                                  # style sama: tidak ada peringatan
    log.clear()
    first = (out / "meme_clip.mp4").read_bytes()
    ex.run_export(cfg, style_label="heavy-marker", restart=True, log=log.append)
    w = warnings_of(log)
    assert len(w) == 1 and "{style}" in w[0] and "clean-line" in w[0] and "heavy-marker" in w[0]
    assert (out / "meme_clip.mp4").is_file()                                       # perilaku tidak berubah: tetap menimpa (bukan error)
    assert first == (out / "meme_clip.mp4").read_bytes()                           # isi sama (strokes sama) — hanya nama style berubah


@needs_ffmpeg
def test_no_warning_and_no_collision_with_style_in_filename(tmp_path):
    cfg, work, out = tes.clip_strokes(tmp_path, **{"export.filename": "{source}_{style}.mp4"})
    log = []
    ex.run_export(cfg, style_label="clean-line", log=log.append)
    ex.run_export(cfg, style_label="heavy-marker", log=log.append)
    assert warnings_of(log) == []
    assert (out / "meme_clip_clean-line.mp4").is_file() and (out / "meme_clip_heavy-marker.mp4").is_file()
    assert (out / "svg" / "meme_clip_clean-line" / ex.SVG_MARKER).is_file() and (out / "svg" / "meme_clip_heavy-marker" / ex.SVG_MARKER).is_file()
    assert not (out / "svg" / "meme_clip").exists()
    # MP4 preset pertama tidak tersentuh oleh render preset kedua
    m1 = json.loads((out / "meme_clip_clean-line.export.json").read_text(encoding="utf-8"))
    assert m1["style"] == "clean-line" and m1["output"]["file"] == "meme_clip_clean-line.mp4"


@needs_ffmpeg
def test_default_filename_unchanged_and_old_manifest_without_style_gives_no_warning(tmp_path):
    cfg, work, out = tes.clip_strokes(tmp_path)
    ex.run_export(cfg, log=quiet)
    mp = out / "meme_clip.export.json"
    m = json.loads(mp.read_text(encoding="utf-8"))
    m.pop("style")                                                                 # manifest T-405 dan sebelumnya: tanpa nama style
    mp.write_text(json.dumps(m), encoding="utf-8")
    log = []
    ex.run_export(cfg, style_label="pencil-light", restart=True, log=log.append)
    assert warnings_of(log) == [] and (out / "svg" / "meme_clip").is_dir()


# ── Zona mati tepi pass k ──────────────────────────
def edge_geometry(name: str = "pencil-light"):
    return sty.make_geometry(load_style(resolve_style(name)), 480, 854)


def vertical_piece(g, d_edge: float, width: float):
    ys = np.arange(600.0, 1000.0, 2.0)
    pts = np.column_stack([np.full_like(ys, g.out_w - d_edge), ys])
    return sty.Piece("group_boundary", False, pts, 1, widths=np.full(len(ys), width))


def test_extra_pass_points_inside_local_guard_zone_do_not_move():
    """Titik pass k >= 1 yang ujung tintanya (jarak tepi − setengah tebal LOKAL) < EDGE_INK_GUARD_PX tidak bergeser sama sekali; titik di
    luar zona itu bergeser. Zona mati lama (margin tetap 2 px) akan menggeser titik di jarak 5,6–6,7 px dengan garis selebar 6,4 px."""
    g = sty.make_geometry(load_style(resolve_style("pencil-light"), overrides={"multipass.offset": 5.5}), 480, 854)
    gp = sty.pass_geometry(g, 1)
    hw = 3.2
    ys = np.arange(300.0, 1700.0, 1.0)
    d = np.linspace(5.65, 30.0, len(ys))                                    # jarak tepi kanan naik pelan
    pts = np.round(np.column_stack([g.out_w - d, ys]), sty.SVG_DECIMALS)      # titik [5] sudah dibulatkan (jitter_pieces membulatkan lagi)
    d = g.out_w - pts[:, 0]
    base = [sty.Piece("group_boundary", False, pts, 1, widths=np.full(len(ys), 2 * hw))]
    pk = sty.jitter_pieces(base, gp, 0, ink_guard=True)[0]
    moved = np.abs(pk.points - base[0].points).max(axis=1) > 0
    assert gp.edge_dead < hw + sty.EDGE_INK_GUARD_PX                         # lebar ini memang melampaui zona mati lama
    assert not moved[d <= hw + sty.EDGE_INK_GUARD_PX].any()
    assert moved[d > hw + sty.EDGE_INK_GUARD_PX + 12.0].any()               # jauh dari tepi: pass k tetap bergeser (offset berlaku)
    old = sty.jitter_pieces(base, gp, 0, ink_guard=False)[0]
    assert (np.abs(old.points - base[0].points).max(axis=1) > 0)[d <= hw + sty.EDGE_INK_GUARD_PX].any()    # tanpa guard: bergeser


REAL_TEST = ROOT / "work" / "clips" / "test"


@pytest.mark.skipif(not (REAL_TEST / "contours" / "frame_00194.json").is_file(), reason="klip test tidak ada")
def test_real_frame_194_pencil_edge_ink_is_zero_with_guard_and_leaks_without():
    """Kasus nyata T-406 (pencil-light offset 4, frame 194, pass 1): tinta 3 baris terluar berubah 1 piksel (85 → 57) tanpa guard."""
    st = load_style(resolve_style("pencil-light"), overrides={"multipass.offset": 4.0})
    g = sty.make_geometry(st, 480, 854)
    gp = sty.pass_geometry(g, 1)
    doc = json.loads((REAL_TEST / "contours" / "frame_00194.json").read_text(encoding="utf-8"))
    base, _ = sty.frame_pieces(doc, g)
    assert jm.edge_ink_changed(base, sty.jitter_pieces(base, gp, 194, ink_guard=True), gp) == 0
    assert jm.edge_ink_changed(base, sty.jitter_pieces(base, gp, 194, ink_guard=False), gp) >= 1


def test_edge_guard_is_continuous_and_only_enlarges_dead_zone():
    g = edge_geometry()
    gp = sty.pass_geometry(g, 1)
    d = np.linspace(0, 80, 4001)
    pts = np.column_stack([g.out_w - d, np.full_like(d, 700.0)])
    hw = np.full_like(d, 3.2)
    old, new = sty.edge_fade(pts, gp), sty.edge_fade(pts, gp, hw)
    assert (new <= old + 1e-12).all() and (new[d <= 3.2 + sty.EDGE_INK_GUARD_PX - 1e-9] < 1e-12).all()
    step = (d[1] - d[0]) * 1.5 / g.edge_fade_len                              # kemiringan maks smoothstep = 1,5 / panjang pelunakan
    assert np.abs(np.diff(new)).max() <= step * 1.01                          # kontinu (smoothstep), bukan pemotongan keras
    thin = sty.edge_fade(pts, gp, np.full_like(d, 0.5))
    assert np.array_equal(thin, old)                                          # tebal kecil: zona mati lama sudah cukup → tidak berubah
    assert np.array_equal(sty.edge_fade(pts, g), sty.edge_fade(pts, g, None))


def test_single_pass_preset_has_no_extra_pass_and_pass_zero_untouched():
    g = edge_geometry("clean-line")
    assert g.passes == 1
    base = [vertical_piece(g, 6.0, 7.0)]
    assert len(sty.multipass_pieces(base, g, 0)) == 1 and sty.multipass_pieces(base, g, 0)[0] is base


def test_algo_rev_bumped():
    assert sty.ALGO_REV == 2 and sty.EDGE_INK_GUARD_PX >= 3.5
