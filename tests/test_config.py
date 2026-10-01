"""Test config.py (T-104a): default, sinkron kode ↔ configs/*.yaml ↔ docs/02, merge, validasi, hash."""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import re
from pathlib import Path

import pytest
import yaml

from rotoscope import config as cfg_mod
from rotoscope.config import (
    ConfigError, PipelineConfig, StyleConfig, ensure_dir, load_class_names, load_pipeline, load_style,
    project_root, section_hash, to_dict,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_YAML = ROOT / "configs" / "default.yaml"
STYLE_YAML = ROOT / "configs" / "styles" / "rough-sketch.yaml"
DOCS_02 = ROOT / "docs" / "02-STYLE-PARAMS.md"
DOCS_PIPELINE_HEADING = "## `configs/default.yaml`"
DOCS_STYLE_HEADING = "## `configs/styles/rough-sketch.yaml`"
PROBE_SCRIPT = ROOT / "scripts" / "sapiens2_probe.py"
EXP_SCRIPT = ROOT / "scripts" / "sapiens2_exp.py"


def docs_block(heading: str) -> dict:
    text = DOCS_02.read_text(encoding="utf-8")
    m = re.search(re.escape(heading) + r".*?```yaml\n(.*?)```", text, re.S)
    assert m, f"blok YAML di bawah {heading} tidak ditemukan di docs/02"
    return yaml.safe_load(m.group(1))


def write_yaml(tmp_path: Path, text: str, name: str = "c.yaml") -> Path:
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def default_groups() -> dict[str, list[str]]:
    return {name: list(classes) for name, classes in load_pipeline().groups}


# ── Default + sinkron (opsi 2a: kode = configs/*.yaml = docs/02) ─
def test_defaults_without_yaml():
    p = load_pipeline()
    assert p.segment.model == "0.8b" and p.segment.precision == "fp16"
    assert dict(p.segment.revision) == {"0.8b": "196a627b928676c4429b738ed76f78a21d96c4eb",
                                        "0.4b": "449b3c5335e6722bb94990abdd1aa6e612432f22"}
    assert p.depth.revision == "5426e4f0f36572d16453bbda7a8389317b1bef99"
    assert dict(p.segment.vram_min_free_mib) == {"0.8b": 3300, "0.4b": 2300}
    assert p.depth.model_id == "depth-anything/Depth-Anything-V2-Small-hf" and p.depth.precision == "fp32"
    assert p.qc.area_median_window == 49 and p.stabilize.mode_k == 3 and p.stabilize.temporal.enabled is False
    assert (p.vectorize.depth_lines.lo_pct, p.vectorize.depth_lines.hi_pct) == (90, 95)
    assert [n for n, _ in p.groups] == ["hair", "face", "torso", "left_arm", "right_arm", "left_leg", "right_leg"]
    assert p.paths.work_dir == Path.cwd() / "work"
    s = load_style()
    assert s.stroke.width_base == 3.2 and s.stroke.color == "#1a1a1a" and s.render.ss == 3
    assert s.texture.brush_image == ROOT / "assets" / "brushes" / "pencil_01.png"


@pytest.mark.parametrize("cls, heading, path, loader", [
    (PipelineConfig, DOCS_PIPELINE_HEADING, DEFAULT_YAML, load_pipeline),
    (StyleConfig, DOCS_STYLE_HEADING, STYLE_YAML, load_style),
])
def test_code_defaults_match_config_files_and_docs(cls, heading, path, loader):
    code = cfg_mod._defaults(cls)
    docs = docs_block(heading)
    file = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert docs == code, "blok YAML docs/02 ≠ default kode"
    assert file == docs, f"{path.name} ≠ blok YAML docs/02"
    if "groups" in code:  # urutan grup = id; dict == mengabaikan urutan
        assert list(docs["groups"]) == list(code["groups"]) == list(file["groups"])
    assert loader(path) == loader()


def test_config_is_immutable():
    p = load_pipeline()
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.qc.area_min = 0.1
    with pytest.raises(TypeError):
        p.segment.model_ids["0.8b"] = "x"
    with pytest.raises(TypeError):
        load_style().stroke.by_type["occlusion"] = None


def test_to_dict_is_json_serializable():
    json.dumps(to_dict(load_pipeline()))
    json.dumps(to_dict(load_style()))


# ── Merge / overrides ──────────────────────────────
def test_partial_yaml_keeps_other_defaults(tmp_path):
    p = load_pipeline(write_yaml(tmp_path, "qc:\n  iou_min: 0.4\nvectorize:\n  depth_lines:\n    hi_pct: 97\n"))
    d = load_pipeline()
    assert p.qc.iou_min == 0.4 and p.qc.area_min == d.qc.area_min
    assert p.vectorize.depth_lines.hi_pct == 97 and p.vectorize.depth_lines.lo_pct == 90
    assert p.segment == d.segment and p.groups == d.groups and p.stabilize == d.stabilize


def test_partial_mapping_merges_per_key(tmp_path):
    p = load_pipeline(write_yaml(tmp_path, 'segment:\n  vram_min_free_mib:\n    "0.4b": 2500\n'))
    assert dict(p.segment.vram_min_free_mib) == {"0.8b": 3300, "0.4b": 2500}
    s = load_style(write_yaml(tmp_path, "stroke:\n  by_type:\n    occlusion: {width_scale: 0.6}\n", "s.yaml"))
    assert s.stroke.by_type["occlusion"].width_scale == 0.6
    assert s.stroke.by_type["occlusion"].opacity_scale == 1.0
    assert s.stroke.by_type["silhouette"] == load_style().stroke.by_type["silhouette"]


@pytest.mark.parametrize("text", ["", "# hanya komentar\n", "stroke:\n"])
def test_empty_yaml_or_section_is_default(tmp_path, text):
    assert load_style(write_yaml(tmp_path, text)) == load_style()


def test_overrides_dotted_keys_applied_after_yaml(tmp_path):
    path = write_yaml(tmp_path, 'segment:\n  model: "0.4b"\n')
    p = load_pipeline(path, overrides={"segment.model": "0.8b", "stabilize.temporal.enabled": True,
                                       "segment.vram_min_free_mib": {"0.8b": 3000}})
    assert p.segment.model == "0.8b" and p.stabilize.temporal.enabled is True
    assert dict(p.segment.vram_min_free_mib) == {"0.8b": 3000, "0.4b": 2300}
    assert load_pipeline(path).segment.model == "0.4b"


def test_groups_replaced_whole():
    rest = [c for c in load_class_names()[1:] if c != "Hair"]
    p = load_pipeline(overrides={"groups": {"body": rest, "hair": ["Hair"]}})
    assert [n for n, _ in p.groups] == ["body", "hair"]


def test_revision_string_or_null_allowed(tmp_path):
    p = load_pipeline(write_yaml(tmp_path, 'segment:\n  revision:\n    "0.4b": null\ndepth:\n  revision: abc123\n'))
    assert p.segment.revision["0.4b"] is None and p.depth.revision == "abc123"
    assert p.segment.revision["0.8b"] == load_pipeline().segment.revision["0.8b"]  # merge per key


# ── Validasi pipeline (tabel docs/02) ──────────────
PIPELINE_INVALID = [
    ({"paths.work_dir": ""}, r"paths\.work_dir"),
    ({"paths.out_dir": 5}, r"paths\.out_dir"),
    ({"segment.model": "1b"}, r"segment\.model \('1b'\)"),
    ({"segment.model_ids": {"0.8b": "facebook/sapiens-seg-0.8b"}}, r"segment\.model_ids\.0\.8b.*sapiens2-seg-"),
    ({"segment.model_ids": {"0.4b": "facebook/sapiens-seg-0.4b-torchscript"}}, r"CC-BY-NC"),
    ({"segment.model_ids": {"1b": "facebook/sapiens2-seg-1b"}}, r"segment\.model_ids\.1b.*tidak dikenal"),
    ({"segment.revision": "abc123"}, r"segment\.revision.*mapping"),
    ({"segment.revision": {"0.8b": ""}}, r"segment\.revision\.0\.8b"),
    ({"segment.revision": {"0.4b": 123}}, r"segment\.revision\.0\.4b.*string"),
    ({"segment.revision": {"1b": "abc"}}, r"segment\.revision\.1b.*tidak dikenal"),
    ({"segment.precision": "bf16"}, r"segment\.precision.*P-005"),
    ({"segment.precision": "int8"}, r"segment\.precision"),
    ({"segment.vram_min_free_mib": {"0.4b": 5000}}, r"segment\.vram_min_free_mib\.0\.4b \(5000\)"),
    ({"segment.vram_min_free_mib": {"0.8b": -1}}, r"segment\.vram_min_free_mib\.0\.8b"),
    ({"segment.vram_min_free_mib": {"0.8b": 3300.5}}, r"segment\.vram_min_free_mib\.0\.8b.*int"),
    ({"segment.vram_min_free_mib": {"1b": 4000}}, r"tidak dikenal"),
    ({"segment.probs_dtype": "float16"}, r"segment\.probs_dtype"),
    ({"depth.model_id": "depth-anything/Depth-Anything-V2-Base-hf"}, r"depth\.model_id.*Small"),
    ({"depth.model_id": "depth-anything/Depth-Anything-V2-Large-hf"}, r"CC-BY-NC"),
    ({"depth.revision": "  "}, r"depth\.revision"),
    ({"depth.precision": "bf16"}, r"depth\.precision.*P-005"),
    ({"depth.vram_min_free_mib": 4097}, r"depth\.vram_min_free_mib"),
    ({"qc.area_min": 0.8}, r"qc\.area_min \(0\.8\) harus < qc\.area_max \(0\.7\)"),
    ({"qc.area_min": -0.1}, r"qc\.area_min"),
    ({"qc.area_max": 1.5}, r"qc\.area_max"),
    ({"qc.iou_min": 1.2}, r"qc\.iou_min"),
    ({"qc.iou_min": float("nan")}, r"qc\.iou_min"),
    ({"qc.blob_min": -0.1}, r"qc\.blob_min"),
    ({"qc.area_drop_min": 2}, r"qc\.area_drop_min"),
    ({"qc.max_big_blobs": 0}, r"qc\.max_big_blobs"),
    ({"qc.max_big_blobs": True}, r"qc\.max_big_blobs.*int, bukan bool"),
    ({"qc.area_median_window": 48}, r"qc\.area_median_window \(48\).*ganjil"),
    ({"qc.area_median_window": 1}, r"qc\.area_median_window"),
    ({"stabilize.temporal.enabled": 1}, r"stabilize\.temporal\.enabled.*bool"),
    ({"stabilize.temporal.mask_ema_alpha": 0}, r"stabilize\.temporal\.mask_ema_alpha"),
    ({"stabilize.temporal.mask_ema_alpha": 1.01}, r"stabilize\.temporal\.mask_ema_alpha"),
    ({"stabilize.temporal.optical_flow_blend": 1.1}, r"optical_flow_blend"),
    ({"stabilize.temporal.boil_preserve": -0.1}, r"boil_preserve"),
    ({"stabilize.temporal.qc_fail_weight": 2}, r"qc_fail_weight"),
    ({"stabilize.island_min_px": -1}, r"stabilize\.island_min_px"),
    ({"stabilize.mode_k": 4}, r"stabilize\.mode_k \(4\).*ganjil"),
    ({"stabilize.mode_k": 0}, r"stabilize\.mode_k"),
    ({"stabilize.depth.normalize": "affine"}, r"stabilize\.depth\.normalize.*log_median_iqr"),
    ({"stabilize.depth.temporal": "true"}, r"stabilize\.depth\.temporal.*bool"),
    ({"stabilize.depth.log_eps": 0}, r"stabilize\.depth\.log_eps"),
    ({"stabilize.depth.log_eps": 0.02}, r"stabilize\.depth\.log_eps"),
    ({"stabilize.depth.iqr_min": 1e-4}, r"stabilize\.depth\.iqr_min"),
    ({"stabilize.depth.iqr_min": 1.5}, r"stabilize\.depth\.iqr_min"),
    ({"vectorize.min_region_area": -1}, r"vectorize\.min_region_area"),
    ({"vectorize.min_hole_area": 1.5}, r"vectorize\.min_hole_area.*int"),
    ({"vectorize.line_min_px": 0}, r"vectorize\.line_min_px"),
    ({"vectorize.min_stroke_px": 0}, r"vectorize\.min_stroke_px"),
    ({"vectorize.depth_lines.blur_sigma": -1}, r"blur_sigma"),
    ({"vectorize.depth_lines.lo_pct": 95}, r"vectorize\.depth_lines\.lo_pct \(95\) harus < hi_pct \(95\)"),
    ({"vectorize.depth_lines.lo_pct": 96}, r"vectorize\.depth_lines\.lo_pct \(96\) harus < hi_pct \(95\)"),
    ({"vectorize.depth_lines.lo_pct": 0}, r"vectorize\.depth_lines\.lo_pct"),
    ({"vectorize.depth_lines.hi_pct": 100}, r"vectorize\.depth_lines\.hi_pct"),
    ({"vectorize.depth_lines.erode_px": 4}, r"vectorize\.depth_lines\.erode_px.*ganjil"),
    ({"vectorize.depth_lines.min_dist_px": -1}, r"min_dist_px"),
    ({"vectorize.depth_lines.min_len_px": -1}, r"min_len_px"),
    ({"vectorize.track.max_match_dist_px": 0}, r"vectorize\.track\.max_match_dist_px"),
    ({"export.source": "svg"}, r"export\.source \('svg'\)"),
    ({"export.crf": 52}, r"export\.crf"),
    ({"export.crf": -1}, r"export\.crf"),
    ({"export.crf": 18.5}, r"export\.crf.*int"),
    ({"export.preset": "turbo"}, r"export\.preset"),
    ({"export.foreground_color": "black"}, r"export\.foreground_color.*#rrggbb"),
    ({"export.background_color": "#fff"}, r"export\.background_color.*#rrggbb"),
    ({"export.background_color": "#000000"}, r"export\.foreground_color.*≠.*export\.background_color"),
    ({"export.background_color": "#000000", "export.foreground_color": "#000000"}, r"siluet tidak akan terlihat"),
    ({"export.audio": "false"}, r"export\.audio.*bool"),
    ({"export.filename": "clip.avi"}, r"export\.filename.*\.mp4"),
    ({"export.filename": "sub/{source}.mp4"}, r"export\.filename.*path"),
    ({"export.filename": "{name}.mp4"}, r"export\.filename.*placeholder"),
    ({"export.filename": ".mp4"}, r"export\.filename"),
]


@pytest.mark.parametrize("overrides, match", PIPELINE_INVALID)
def test_pipeline_validation_rejects(overrides, match):
    with pytest.raises(ConfigError, match=match):
        load_pipeline(overrides=overrides)


# ── Validasi style (tabel docs/02) ─────────────────
STYLE_INVALID = [
    ({"shape.simplify_epsilon": -1}, r"shape\.simplify_epsilon"),
    ({"shape.resample_points": 3}, r"shape\.resample_points"),
    ({"shape.smooth_tension": 1.5}, r"shape\.smooth_tension"),
    ({"shape.spline_steps": 0}, r"shape\.spline_steps"),
    ({"stroke.width_base": 0}, r"stroke\.width_base"),
    ({"stroke.width_variation": 1.1}, r"stroke\.width_variation"),
    ({"stroke.opacity": -0.1}, r"stroke\.opacity"),
    ({"stroke.taper_min": 2}, r"stroke\.taper_min"),
    ({"stroke.width_noise_scale": -1}, r"stroke\.width_noise_scale"),
    ({"stroke.color": "1a1a1a"}, r"stroke\.color.*#rrggbb"),
    ({"stroke.cap": "flat"}, r"stroke\.cap"),
    ({"stroke.taper_ends": "yes"}, r"stroke\.taper_ends.*bool"),
    ({"stroke.taper_px": -1}, r"stroke\.taper_px"),
    ({"stroke.by_type": {"occlusion": {"width_scale": 0}}}, r"stroke\.by_type\.occlusion\.width_scale"),
    ({"stroke.by_type": {"silhouette": {"opacity_scale": 1.5}}}, r"stroke\.by_type\.silhouette\.opacity_scale"),
    ({"stroke.by_type": {"hatch": {"width_scale": 1.0}}}, r"stroke\.by_type\.hatch.*tidak dikenal"),
    ({"jitter.amplitude": -1}, r"jitter\.amplitude"),
    ({"jitter.frequency": -1}, r"jitter\.frequency"),
    ({"jitter.temporal_seed_mode": "random"}, r"jitter\.temporal_seed_mode"),
    ({"jitter.temporal_drift": 1.5}, r"jitter\.temporal_drift"),
    ({"jitter.param_seed": 1.5}, r"jitter\.param_seed.*int"),
    ({"multipass.enabled": 1}, r"multipass\.enabled.*bool"),
    ({"multipass.passes": 0}, r"multipass\.passes"),
    ({"multipass.offset": -1}, r"multipass\.offset"),
    ({"multipass.opacity_falloff": 2}, r"multipass\.opacity_falloff"),
    ({"texture.mode": "pencil"}, r"texture\.mode"),
    ({"texture.stamp_spacing": 0}, r"texture\.stamp_spacing"),
    ({"texture.pressure_noise": 2}, r"texture\.pressure_noise"),
    ({"texture.grain_strength": -1}, r"texture\.grain_strength"),
    ({"texture.brush_image": "assets/brushes/tidak_ada.png"}, r"texture\.brush_image.*tidak ada"),
    ({"paper.enabled": "no"}, r"paper\.enabled.*bool"),
    ({"paper.color": "#fff"}, r"paper\.color"),
    ({"paper.texture_image": "assets/paper/tidak_ada.jpg"}, r"paper\.texture_image.*tidak ada"),
    ({"paper.texture_opacity": 2}, r"paper\.texture_opacity"),
    ({"paper.vignette": -1}, r"paper\.vignette"),
    ({"render.ss": 0}, r"render\.ss"),
    ({"render.ss": 9}, r"render\.ss"),
]


@pytest.mark.parametrize("overrides, match", STYLE_INVALID)
def test_style_validation_rejects(overrides, match):
    with pytest.raises(ConfigError, match=match):
        load_style(overrides=overrides)


def test_unused_assets_not_checked():
    load_style(overrides={"texture.mode": "none", "texture.brush_image": "tidak_ada.png"})
    load_style(overrides={"paper.enabled": False, "paper.texture_image": "tidak_ada.jpg"})


# ── Path ───────────────────────────────────────────
def test_work_dir_relative_to_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert load_pipeline().paths.work_dir == Path.cwd() / "work"


def test_asset_paths_relative_to_project_root_not_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert project_root() == ROOT
    s = load_style()
    assert s.texture.brush_image == ROOT / "assets" / "brushes" / "pencil_01.png"
    assert s.paper.texture_image == ROOT / "assets" / "paper" / "rough_01.jpg"


def test_absolute_asset_path_allowed(tmp_path):
    brush = tmp_path / "brush.png"
    brush.write_bytes(b"")
    assert load_style(overrides={"texture.brush_image": str(brush)}).texture.brush_image == brush


def test_project_root_not_found(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg_mod, "_PACKAGE_DIR", tmp_path / "pkg")
    with pytest.raises(ConfigError, match=r"texture\.brush_image.*root project"):
        load_style()
    brush, paper = tmp_path / "b.png", tmp_path / "p.jpg"
    brush.write_bytes(b"")
    paper.write_bytes(b"")
    load_style(overrides={"texture.brush_image": str(brush), "paper.texture_image": str(paper)})


def test_windows_path_single_quotes_ok(tmp_path):
    p = load_pipeline(write_yaml(tmp_path, "paths:\n  work_dir: 'D:\\rotoscope\\work'\n"))
    assert p.paths.work_dir == Path("D:/rotoscope/work")


def test_path_with_control_char_rejected(tmp_path):
    path = write_yaml(tmp_path, 'paths:\n  work_dir: "D:\\rotoscope\\new"\n')
    with pytest.raises(ConfigError, match=r"paths\.work_dir.*karakter kontrol.*kutip"):
        load_pipeline(path)


def test_invalid_yaml_escape_gives_hint(tmp_path):
    path = write_yaml(tmp_path, 'paths:\n  work_dir: "D:\\rotoscope\\work"\n')  # "\w" escape tidak valid
    with pytest.raises(ConfigError, match=r"(?s)YAML tidak valid.*kutip tunggal"):
        load_pipeline(path)


def test_load_has_no_side_effect_and_ensure_dir_creates(tmp_path):
    work = tmp_path / "a" / "b"
    p = load_pipeline(overrides={"paths.work_dir": str(work)})
    assert not work.exists()
    assert ensure_dir(p.paths.work_dir) == work and work.is_dir()


# ── YAML: null, tipe, file ─────────────────────────
def test_unquoted_hex_color_hint(tmp_path):
    with pytest.raises(ConfigError, match=r"stroke\.color \(null\).*komentar"):
        load_style(write_yaml(tmp_path, "stroke:\n  color: #1a1a1a\n"))


def test_null_rejected_outside_revision(tmp_path):
    with pytest.raises(ConfigError, match=r"qc\.iou_min \(null\)"):
        load_pipeline(write_yaml(tmp_path, "qc:\n  iou_min: null\n"))


def test_duplicate_yaml_key_rejected(tmp_path):
    with pytest.raises(ConfigError, match=r"duplikat qc\.iou_min"):
        load_pipeline(write_yaml(tmp_path, "qc:\n  iou_min: 0.5\n  iou_min: 0.6\n"))


def test_top_level_must_be_mapping(tmp_path):
    with pytest.raises(ConfigError, match=r"mapping"):
        load_pipeline(write_yaml(tmp_path, "- a\n- b\n"))


def test_section_must_be_mapping(tmp_path):
    with pytest.raises(ConfigError, match=r"qc \(5\) harus mapping"):
        load_pipeline(write_yaml(tmp_path, "qc: 5\n"))


def test_missing_file(tmp_path):
    with pytest.raises(ConfigError, match=r"tidak ditemukan"):
        load_pipeline(tmp_path / "tidak_ada.yaml")


# ── Key tidak dikenal ──────────────────────────────
@pytest.mark.parametrize("loader, text, match", [
    (load_pipeline, "segment:\n  presicion: fp16\n", r"segment\.presicion.*maksudnya 'precision'"),
    (load_pipeline, "qcc:\n  iou_min: 0.5\n", r"qcc.*maksudnya 'qc'"),
    (load_pipeline, "vectorize:\n  xyz: 1\n", r"vectorize\.xyz.*Key yang valid di vectorize"),
    (load_pipeline, "stroke:\n  width_base: 2\n", r"stroke.*configs/styles"),
    (load_style, "temporal:\n  mask_ema_alpha: 0.5\n", r"temporal.*stabilize\.temporal"),
    (load_style, "shape:\n  min_contour_area: 800\n", r"vectorize\.min_region_area"),
])
def test_unknown_key_with_suggestion(tmp_path, loader, text, match):
    with pytest.raises(ConfigError, match=match):
        loader(write_yaml(tmp_path, text))


def test_unknown_override_key_with_suggestion():
    with pytest.raises(ConfigError, match=r"overrides: qc\.iou.*'iou_min'"):
        load_pipeline(overrides={"qc.iou": 0.5})


# ── Aturan grup ────────────────────────────────────
def _groups_with(**changes) -> dict:
    g = default_groups()
    g.update(changes)
    return g


def test_group_unknown_class_with_suggestion():
    with pytest.raises(ConfigError, match=r"groups\.hair.*'Hiar'.*'Hair'"):
        load_pipeline(overrides={"groups": _groups_with(hair=["Hiar"])})


def test_group_unknown_class():
    with pytest.raises(ConfigError, match=r"groups\.hair.*'Hat'.*tidak ada"):
        load_pipeline(overrides={"groups": _groups_with(hair=["Hair", "Hat"])})


@pytest.mark.parametrize("hair", [["Hair", "Hair"], ["Hair", "Tongue"]])
def test_group_class_twice(hair):
    with pytest.raises(ConfigError, match=r"sudah ada di groups\."):
        load_pipeline(overrides={"groups": _groups_with(hair=hair)})


def test_group_class_not_listed():
    face = [c for c in default_groups()["face"] if c != "Tongue"]
    with pytest.raises(ConfigError, match=r"tidak tercantum.*Tongue"):
        load_pipeline(overrides={"groups": _groups_with(face=face)})


def test_group_background_class_listed():
    with pytest.raises(ConfigError, match=r"groups\.hair.*'Background'"):
        load_pipeline(overrides={"groups": _groups_with(hair=["Hair", "Background"])})


def test_group_name_background_reserved():
    g = default_groups()
    g["background"] = g.pop("hair")
    with pytest.raises(ConfigError, match=r"groups\.background.*dicadangkan"):
        load_pipeline(overrides={"groups": g})


@pytest.mark.parametrize("name", ["Hair", "left-arm", "2arm", "rambut panjang"])
def test_group_name_must_be_identifier(name):
    g = default_groups()
    g[name] = g.pop("hair")
    with pytest.raises(ConfigError, match=r"identifier"):
        load_pipeline(overrides={"groups": g})


def test_group_name_duplicate_in_yaml(tmp_path):
    text = DEFAULT_YAML.read_text(encoding="utf-8").replace("  hair: [Hair]\n", "  hair: [Hair]\n  hair: [Hair]\n")
    with pytest.raises(ConfigError, match=r"duplikat groups\.hair"):
        load_pipeline(write_yaml(tmp_path, text))


def test_too_many_groups():
    with pytest.raises(ConfigError, match=r"256 grup.*maksimal 255"):
        load_pipeline(overrides={"groups": {f"g{i}": ["Hair"] for i in range(256)}})


@pytest.mark.parametrize("groups", [{}, ["hair"], {"hair": []}, {"hair": "Hair"}])
def test_groups_bad_shape(groups):
    with pytest.raises(ConfigError, match=r"groups"):
        load_pipeline(overrides={"groups": groups})


# ── Hash ───────────────────────────────────────────
def test_hash_ignores_yaml_key_order_and_int_float(tmp_path):
    a = write_yaml(tmp_path, "vectorize:\n  min_region_area: 800\n  depth_lines:\n    hi_pct: 95\n    lo_pct: 90\n"
                             "qc:\n  iou_min: 0.5\n", "a.yaml")
    b = write_yaml(tmp_path, "qc:\n  iou_min: 0.5\nvectorize:\n  depth_lines:\n    lo_pct: 90.0\n    hi_pct: 95.0\n"
                             "  min_region_area: 800\n", "b.yaml")
    pa, pb = load_pipeline(a), load_pipeline(b)
    for section in ("vectorize", "qc", "segment", "groups", "stabilize", "depth"):
        assert section_hash(pa, section) == section_hash(pb, section)
    h = section_hash(pa, "vectorize")
    assert re.fullmatch(r"[0-9a-f]{64}", h)


def test_hash_changes_when_value_changes():
    base = load_pipeline()
    changed = load_pipeline(overrides={"vectorize.depth_lines.min_len_px": 31})
    assert section_hash(base, "vectorize") != section_hash(changed, "vectorize")
    assert section_hash(base, "qc") == section_hash(changed, "qc")
    s1, s2 = load_style(), load_style(overrides={"stroke.color": "#000000"})
    assert section_hash(s1, "stroke") != section_hash(s2, "stroke")


def test_groups_hash_class_order_vs_group_order():
    g = default_groups()
    base = section_hash(load_pipeline(), "groups")
    g["face"] = list(reversed(g["face"]))  # urutan kelas di dalam grup: tidak berpengaruh
    assert section_hash(load_pipeline(overrides={"groups": g}), "groups") == base
    swapped = dict(reversed(list(g.items())))  # urutan grup = id grup: berpengaruh
    assert section_hash(load_pipeline(overrides={"groups": swapped}), "groups") != base


def test_hash_unknown_section():
    with pytest.raises(ConfigError, match=r"section_hash"):
        section_hash(load_pipeline(), "groupz")


# ── Nama kelas (package data) + script T-102c ──────
def test_class_names_package_data():
    classes = load_class_names()
    assert len(classes) == 29 and classes[0] == "Background" and "Tongue" in classes


def test_t102c_scripts_read_class_list_from_new_path():
    assert not (ROOT / "scripts" / "sapiens2_classes.json").exists()
    spec = importlib.util.spec_from_file_location("sapiens2_probe_t104a", PROBE_SCRIPT)
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)  # hanya import: konstanta + fungsi, tanpa model
    assert probe.CLASSES_FILE.is_file()
    names = probe.resolve_id2label({i: f"LABEL_{i}" for i in range(29)})
    assert tuple(names.values()) == load_class_names()
    # sapiens2_exp.py membaca lewat probe.CLASSES_FILE, tanpa path sendiri
    exp_src = EXP_SCRIPT.read_text(encoding="utf-8")
    assert "probe.CLASSES_FILE" in exp_src and '"sapiens2_classes.json"' not in exp_src
