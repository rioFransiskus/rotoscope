"""Test kertas bertekstur + vignette (T-404a Opsi B; modul rotoscope.paper): orientasi / potong / skala, normalisasi rata-rata, gain, vignette (elips,
kuadrat), invers LUT (`LevelMap`, aturan tabrakan), susunan export vs ORACLE jalur [5] lama (tests/paper_oracle.py; selisih ≤ 1 level), tinta tidak
terkena vignette, statis antar frame, kertas datar tidak memuat tekstur, [5] mengabaikan paper.* (strokes tetap byte-identik, tidak basi), error,
determinisme / cache, tanpa torch. Integrasi export / cli: tests/test_export_paper.py."""

from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path

import cv2
import numpy as np
import paper_oracle as po
import pytest
import stylize_metrics as sm
from test_stylize import H, OW, STYLE_YAML, W, cfg_for, doc_for, frame_strokes, make_stage_work, manifest, out_hashes, run_main, stroke

from rotoscope import paper as pap
from rotoscope import stylize as sty
from rotoscope.config import load_style
from rotoscope.stage_common import StageError

NO_JITTER = {"jitter.amplitude": 0.0, "multipass.passes": 1, "stroke.opacity": 1.0}
BASE = {"render.output_width": OW, "stroke.width_base": 25.3, **NO_JITTER}
ASSET_SHA = "1ec1da7815c3e53e02f7f6f4c68b091dddd5d089ba9aa40ccebb7685e3032804"


def write_tex(path: Path, h: int = 300, w: int = 400, seed: int = 0, mean: float = 180.0, std: float = 12.0) -> Path:
    rng = np.random.RandomState(seed)
    arr = np.clip(np.rint(rng.normal(mean, std, (h, w))), 1, 255).astype(np.uint8)
    assert cv2.imwrite(str(path), arr)
    return path


def pstyle(tex: Path | None, **ov):
    base = {**BASE, "paper.color": "#808080", "paper.texture_opacity": 0.5, "paper.texture_gain": 1.0, "paper.vignette": 0.0}
    if tex is not None:
        base["paper.texture_image"] = str(tex)
    return load_style(None, overrides={**base, **ov})


@pytest.fixture
def tex(tmp_path):
    return write_tex(tmp_path / "t.png")


def flat_frame(st, doc=None, w=W, h=H):
    g = sty.make_geometry(st, w, h)
    doc = doc or doc_for(frame_strokes(1), 1, w, h)
    passes, _ = sty.frame_passes(doc, g)
    return g, passes, po.flat_rgb(passes, g)


# ── Default ────────────────────────────────────────
def test_defaults_are_rio_final_choice():
    s = load_style(None)
    p = s.paper
    assert p.enabled and p.color == "#f4f1ea" and p.texture_opacity == 0.35 and p.texture_gain == 3.0 and p.vignette == 0.0
    assert pap.texture_active(s) and not pap.paper_flat(s)


def test_flat_conditions():
    base = load_style(None)
    ov = lambda **k: dataclasses.replace(base, paper=dataclasses.replace(base.paper, **k))      # noqa: E731
    assert pap.paper_flat(ov(enabled=False)) and pap.paper_flat(ov(enabled=False, vignette=0.3))
    assert pap.paper_flat(ov(texture_opacity=0.0)) and pap.paper_flat(ov(texture_gain=0.0))
    assert not pap.paper_flat(ov(texture_opacity=0.0, vignette=0.1)) and not pap.paper_flat(ov(texture_gain=0.0, vignette=0.1))
    assert pap.paper_ref(ov(enabled=False)) is None and pap.paper_ref(ov(texture_opacity=0.0)) is None


def test_paper_ref_contains_params_and_content_hash(tex):
    r = pap.paper_ref(pstyle(tex))
    assert r["texture_opacity"] == 0.5 and r["texture_gain"] == 1.0 and r["vignette"] == 0.0 and r["color"] == "#808080"
    assert r["texture_sha256"] == hashlib.sha256(Path(tex).read_bytes()).hexdigest()
    assert pap.paper_ref(pstyle(tex, **{"paper.vignette": 0.1}))["vignette"] == 0.1
    vig_only = pap.paper_ref(pstyle(tex, **{"paper.texture_opacity": 0.0, "paper.vignette": 0.1}))
    assert vig_only["texture_sha256"] is None and vig_only["texture_image"] is None and vig_only["vignette"] == 0.1
    d = load_style(None)
    assert pap.paper_ref(d)["texture_image"] == "assets/paper/rough_01.jpg"           # relatif root, bukan absolut


# ── Orientasi / potong / skala ─────────────────────
def test_orient_rotates_ccw_only_when_orientation_differs():
    t = np.arange(40 * 60, dtype=np.float32).reshape(40, 60)          # landscape
    r, rot = pap.orient_texture(t, 20, 30)                            # kanvas portrait
    assert rot and r.shape == (60, 40) and np.array_equal(r, np.rot90(t, 1))
    assert r[0, 0] == t[0, -1]                                        # berlawanan jarum jam: pojok kanan-atas → kiri-atas
    for w, h in ((30, 20), (25, 25)):                                 # landscape / persegi: tidak diputar
        r, rot = pap.orient_texture(t, w, h)
        assert not rot and r is t
    p = np.ascontiguousarray(t.T)                                     # gambar portrait + kanvas landscape → putar
    assert pap.orient_texture(p, 30, 20)[1] and not pap.orient_texture(p, 20, 30)[1]


def test_fit_native_crop_is_exact_center_subarray_without_resampling():
    t = np.arange(40 * 60, dtype=np.float32).reshape(40, 60)
    rot, _ = pap.orient_texture(t, 20, 30)
    crop, mode, s = pap.fit_texture(rot, 20, 30)
    assert mode == "native" and s == 1.0 and np.array_equal(crop, rot[15:45, 10:30])     # offset (60−30)//2, (40−20)//2
    crop, mode, _ = pap.fit_texture(t, 60, 40)                                            # tepat sama ukuran
    assert mode == "native" and np.array_equal(crop, t)


def test_fit_scaled_when_canvas_larger_keeps_size_mean_range_and_is_cubic():
    rng = np.random.RandomState(3)
    t = (rng.rand(40, 60).astype(np.float32) * 20 + 100)
    crop, mode, s = pap.fit_texture(t, 90, 80)
    assert mode == "scaled" and crop.shape == (80, 90) and s == max(90 / 60, 80 / 40)
    assert abs(float(crop.mean()) - float(t.mean())) < 1.0 and crop.min() > 80 and crop.max() < 140
    big = cv2.resize(t, (120, 80), interpolation=cv2.INTER_CUBIC)       # s = max(90/60, 80/40) = 2 → 120 × 80; potong tengah (x 15, y 0)
    assert np.array_equal(crop, big[0:80, 15:105])                       # INTER_CUBIC (bukan nearest / linear: std tekstur terjaga)


# ── Normalisasi rata-rata + gain ───────────────────
@pytest.mark.parametrize("mean,std,seed", [(60.0, 8.0, 1), (180.0, 12.0, 2), (230.0, 10.0, 3)])
def test_paper_mean_equals_color_for_any_texture_brightness(tmp_path, mean, std, seed):
    t = write_tex(tmp_path / "t.png", mean=mean, std=std, seed=seed)
    layer = pap.render_paper(OW, OW, pstyle(t))
    m = layer.f32.reshape(-1, 3).mean(axis=0, dtype=np.float64)
    assert np.allclose(m, 128.0, rtol=2e-4), m
    assert layer.f32.std() > 0.3                                       # tekstur benar-benar ada (bukan datar)


def test_texture_gain_scales_contrast_and_floor_keeps_positive(tex):
    s1 = pap.render_paper(OW, OW, pstyle(tex)).f32[..., 1].std(dtype=np.float64)
    s2 = pap.render_paper(OW, OW, pstyle(tex, **{"paper.texture_gain": 2.0})).f32[..., 1].std(dtype=np.float64)
    assert s2 / s1 == pytest.approx(2.0, rel=0.03)
    big = pap.render_paper(OW, OW, pstyle(tex, **{"paper.texture_gain": 100.0, "paper.texture_opacity": 1.0}))
    assert float(big.f32.min()) > 0 and big.info["clipped_fraction"] > 0
    flat = pap.render_paper(OW, OW, pstyle(tex, **{"paper.texture_gain": 0.0}))     # gain 0 = datar, berkas tidak dibaca
    assert np.all(flat.f32 == 128.0) and not flat.info["texture_active"]


# ── Vignette ───────────────────────────────────────
def test_vignette_field_center_corner_symmetry_monotone_and_elliptic():
    f = pap.vignette_field(101, 201, 0.3)                              # ganjil: pusat tepat
    assert f.dtype == np.float32 and f[100, 50] == 1.0
    assert f[0, 0] == pytest.approx(0.7, abs=1e-6) and f[-1, -1] == pytest.approx(0.7, abs=1e-6)
    assert np.array_equal(f, f[::-1, ::-1]) and np.array_equal(f, f[:, ::-1]) and np.array_equal(f, f[::-1, :])
    assert (np.diff(f[100, 50:]) <= 0).all() and (np.diff(f[100:, 50]) <= 0).all()
    assert f[100, 0] == pytest.approx(1 - 0.3 / 2, abs=1e-5) and f[0, 50] == pytest.approx(1 - 0.3 / 2, abs=1e-5)   # elips: tengah sisi sama di portrait
    assert (pap.vignette_field(64, 64, 0.0) == 1.0).all()


def test_vignette_applies_only_to_paper_multiplier(tex):
    layer = pap.render_paper(OW, OW, pstyle(tex, **{"paper.vignette": 0.5, "paper.texture_opacity": 0.0}))
    assert layer.f32[0, 0, 1] == pytest.approx(128 * 0.5, abs=1e-4) and layer.f32[OW // 2, OW // 2, 1] > 128 * 0.99
    assert not layer.info["texture_active"] and layer.info["vignette"] == 0.5


# ── Invers LUT ─────────────────────────────────────
def lut_image(paper, ink):
    lut = sty.ink_lut(dataclasses.make_dataclass("G", ["paper", "ink"])(paper, ink))
    return lut[None, :, :]                                            # (1, 256, 3): piksel k = level k


def test_levelmap_inverts_lut_exactly_for_production_palette():
    """Palet produksi (#f4f1ea → #1a1a1a): tiga kanal punya kontras berbeda (218 / 215 / 208 langkah) → warna ketiga kanal sekaligus unik
    per level: TANPA tabrakan, pemulihan eksak untuk 256 level."""
    paper, ink = (0xF4, 0xF1, 0xEA), (0x1A, 0x1A, 0x1A)
    lm = pap.LevelMap(paper, ink)
    lv, box = lm.levels_of(lut_image(paper, ink))
    assert lm.n_collisions == 0 and box is not None
    assert np.array_equal(lv[0], np.arange(256, dtype=np.float32))
    lm2 = pap.LevelMap((255, 255, 255), (0, 0, 0))
    lv2, _ = lm2.levels_of(lut_image((255, 255, 255), (0, 0, 0)))
    assert lm2.n_collisions == 0 and np.array_equal(lv2[0], np.arange(256, dtype=np.float32))


def test_levelmap_collision_rule_is_mean_of_sharing_levels_and_paper_is_zero():
    """Kontras rendah (abu-abu 100 → 90: 10 langkah per 256 level) → banyak tabrakan: level = RATA-RATA level yang berbagi warna (galat ≤ 0,5·lebar
    kelompok); warna kertas → level 0 walau level 1.. berbagi warna itu."""
    paper, ink = (100, 100, 100), (90, 90, 90)
    lm = pap.LevelMap(paper, ink)
    img = lut_image(paper, ink)
    lv, _ = lm.levels_of(img)
    lut = img[0]
    assert lm.n_collisions == 256 - 11 and lv[0, 0] == 0.0
    for k in range(256):
        same = [j for j in range(256) if (lut[j] == lut[k]).all()]
        want = 0.0 if (lut[k] == np.array(paper)).all() else float(np.mean(same))
        assert lv[0, k] == pytest.approx(want), (k, same)
        assert abs(lv[0, k] - k) <= len(same) / 2 or want == 0.0


def test_levelmap_rejects_colors_outside_lut_and_handles_paper_only():
    lm = pap.LevelMap((0xF4, 0xF1, 0xEA), (0x1A, 0x1A, 0x1A))
    img = np.full((4, 4, 3), (0xF4, 0xF1, 0xEA), np.uint8)
    lv, box = lm.levels_of(img)
    assert box is None and not lv.any()
    img[1, 1] = (200, 30, 30)                                          # merah jenuh: bukan hasil LUT
    with pytest.raises(StageError, match="di luar LUT"):
        lm.levels_of(img)


# ── Susunan export vs ORACLE jalur [5] lama ────────
@pytest.mark.parametrize("color", ["#808080", "#f4f1ea"])
@pytest.mark.parametrize("legacy", [True, False])
@pytest.mark.parametrize("vig", [0.0, 0.25])
def test_compose_matches_oracle_within_one_level(tex, legacy, color, vig):
    ov = {} if legacy else {"multipass.passes": 2, "stroke.opacity": 0.92}
    st = pstyle(tex, **{**ov, "paper.color": color, "paper.vignette": vig, "paper.texture_gain": 3.0})
    g, passes, flat = flat_frame(st)
    assert g.legacy == legacy
    layer = pap.render_paper(g.out_w, g.out_h, st)
    oracle = po.render_rgb(passes, g, layer)
    comp = pap.compose_textured(flat, pap.LevelMap(g.paper, g.ink), layer, g.ink)
    d = np.abs(comp.astype(int) - oracle.astype(int))
    assert int(d.max()) <= 1, int(d.max())
    ink_px = int((np.abs(flat.astype(int) - np.array(g.paper)).max(-1) > 0).sum())
    assert ink_px > 500
    if color == "#f4f1ea":
        assert int(d.max()) == 0                                         # palet produksi tanpa tabrakan LUT → susunan export = oracle PERSIS
    else:
        assert int((d.max(-1) > 0).sum()) < 0.6 * ink_px                        # kontras rendah (102 langkah): selisih 1 level hanya pada piksel tabrakan
    # oracle sendiri = rumus tangan
    lv, box = po.ink_levels(passes, g)
    x0, y0, x1, y1 = box
    a = (lv.astype(np.float32) / np.float32(255))[..., None]
    hand = np.rint(layer.f32[y0:y1, x0:x1] * (np.float32(1) - a) + np.array(g.ink, np.float32) * a).astype(np.uint8)
    assert np.array_equal(oracle[y0:y1, x0:x1], hand) and np.array_equal(oracle[:y0], layer.u8[:y0])


def test_compose_exact_when_lut_has_no_collisions(tex):
    st = pstyle(tex, **{"paper.color": "#ffffff", "stroke.color": "#000000", "paper.texture_gain": 3.0, "paper.vignette": 0.2})
    g, passes, flat = flat_frame(st)
    layer = pap.render_paper(g.out_w, g.out_h, st)
    lm = pap.LevelMap(g.paper, g.ink)
    assert lm.n_collisions == 0
    assert np.array_equal(pap.compose_textured(flat, lm, layer, g.ink), po.render_rgb(passes, g, layer))


def test_recovery_from_composed_frame_within_bound(tex):
    st = pstyle(tex, **{"paper.color": "#f4f1ea", "paper.texture_gain": 3.0, "paper.vignette": 0.25, "multipass.passes": 2, "stroke.opacity": 0.92})
    g, passes, flat = flat_frame(st)
    layer = pap.render_paper(g.out_w, g.out_h, st)
    comp = pap.compose_textured(flat, pap.LevelMap(g.paper, g.ink), layer, g.ink)
    lv, box = po.ink_levels(passes, g)
    truth = np.zeros((g.out_h, g.out_w), np.float32)
    truth[box[1]:box[3], box[0]:box[2]] = lv.astype(np.float32) / 255
    err = np.abs(sm.coverage(comp, g, layer.f32) - truth)
    assert float(err.max()) <= sm.recover_f_bound(layer.f32, g.ink) + 1 / 255
    wrong = np.abs(sm.coverage(po.flat_rgb(passes, g), g, layer.f32) - truth)        # PNG DATAR dibaca sebagai bertekstur → salah besar
    assert float(wrong.max()) > 4 * float(err.max())


def test_ink_core_untouched_and_vignette_only_on_paper(tex):
    ink_doc = doc_for([stroke([(15.5, 15.5), (40.5, 15.5)], closed=False, typ="group_boundary")], 0)
    st = pstyle(tex, **{"paper.vignette": 0.5, "stroke.width_base": 60.0, "stroke.width_variation": 0.0, "stroke.taper_ends": False})
    g, passes, flat = flat_frame(st, ink_doc)
    layer = pap.render_paper(g.out_w, g.out_h, st)
    comp = pap.compose_textured(flat, pap.LevelMap(g.paper, g.ink), layer, g.ink)
    y, x = int(15.5 * g.scale), int(28 * g.scale)
    assert tuple(comp[y, x]) == g.ink                                  # tinta penuh di dekat sudut: TIDAK ikut digelapkan
    assert layer.f32[0, 0, 1] < 128 * 0.6 and layer.f32[g.out_h // 2, g.out_w // 2, 1] > 128 * 0.97


def test_empty_frame_is_paper_and_paper_static_between_frames(tex):
    st = pstyle(tex, **{"paper.vignette": 0.2})
    g = sty.make_geometry(st, W, H)
    layer = pap.render_paper(g.out_w, g.out_h, st)
    lm = pap.LevelMap(g.paper, g.ink)
    empty = pap.compose_textured(po.flat_rgb([[]], g), lm, layer, g.ink)
    assert np.array_equal(empty, layer.u8)
    outs = []
    for pts, i in (([(10.5, 10.5), (30.5, 12.5)], 0), ([(70.5, 88.5), (90.5, 80.5)], 7)):
        d = doc_for([stroke(pts, closed=False, typ="group_boundary")], i)
        passes, _ = sty.frame_passes(d, g)
        outs.append(pap.compose_textured(po.flat_rgb(passes, g), lm, layer, g.ink))
    q = (slice(g.out_h // 2, None), slice(0, g.out_w // 2))           # kuadran kiri-bawah: tanpa strok di kedua frame
    assert np.array_equal(outs[0][q], layer.u8[q]) and np.array_equal(outs[1][q], layer.u8[q]) and not np.array_equal(outs[0], layer.u8)


def test_compose_rejects_size_mismatch(tex):
    layer = pap.render_paper(OW, OW, pstyle(tex))
    lm = pap.LevelMap((128, 128, 128), (26, 26, 26))
    with pytest.raises(StageError, match="≠ kertas"):
        pap.compose_textured(np.full((10, 10, 3), 128, np.uint8), lm, layer, (26, 26, 26))


# ── [5] mengabaikan paper.* ────────────────────────
def paper_style_yaml(extra: str) -> str:
    return STYLE_YAML + extra


def test_stage5_ignores_paper_params_strokes_identical_and_not_stale(tmp_path, tex):
    work = make_stage_work(tmp_path, n=3)
    plain = tmp_path / "plain.yaml"
    plain.write_text(STYLE_YAML, encoding="utf-8")
    assert run_main(work, plain) == 0
    base_hash, m0 = out_hashes(work), manifest(work)
    assert m0["contract"] == "T-403" and "paper" not in m0 and not any(k.startswith("paper.") and k != "paper.color" for k in m0["style_params"])
    variants = [f'paper:\n  texture_image: "{Path(tex).as_posix()}"\n  texture_opacity: 0.9\n  texture_gain: 7.0\n  vignette: 0.4\n',
                "paper:\n  enabled: false\n", "paper:\n  texture_opacity: 0.0\n  vignette: 0.2\n"]
    for i, extra in enumerate(variants):
        style = tmp_path / f"v{i}.yaml"
        style.write_text(paper_style_yaml(extra), encoding="utf-8")
        logs: list[str] = []
        run = sty.run_stylize(cfg_for(work), load_style(style), "plain", log=logs.append)
        assert run["processed"] == 0 and run["stale"] == [] and out_hashes(work) == base_hash       # strokes tidak basi, byte-identik
        assert manifest(work)["style_hash"] == m0["style_hash"]
    ign = sty.style_params(load_style(tmp_path / "v0.yaml"))[1]
    assert {"paper.enabled", "paper.texture_image", "paper.texture_opacity", "paper.texture_gain", "paper.vignette"} <= set(ign)
    assert "paper.color" in sty.style_params(load_style(None))[0]
    changed = tmp_path / "color.yaml"                                  # paper.color TETAP aktif di [5] (LUT)
    changed.write_text(paper_style_yaml('paper:\n  color: "#ffffff"\n'), encoding="utf-8")
    assert sty.run_stylize(cfg_for(work), load_style(changed), "plain", log=lambda m: None)["processed"] == 3


def test_render_frame_svg_and_png_independent_of_paper_params(tmp_path, tex):
    """SVG DAN PNG [5] tidak bergantung pada paper.* selain paper.color (kertas datar; kertas bertekstur hanya di export)."""
    doc = doc_for(frame_strokes(2), 2)
    outs = []
    for ov in ({}, {"paper.texture_opacity": 0.9, "paper.texture_gain": 7.0, "paper.vignette": 0.4},
               {"paper.enabled": False}, {"paper.texture_opacity": 0.0, "paper.vignette": 0.2}):
        st = load_style(None, overrides={**BASE, **ov, "paper.texture_image": str(tex)})
        g = sty.make_geometry(st, W, H)
        svg, png, _ = sty.render_frame(doc, g, st)
        outs.append((svg, png))
    assert all(o == outs[0] for o in outs)
    assert b"<image" not in outs[0][0] and outs[0][0].count(b"<rect") == 1


# ── Error ──────────────────────────────────────────
def test_missing_corrupt_and_black_texture_raise_stage_error(tmp_path, tex):
    st = pstyle(tex)
    missing = dataclasses.replace(st, paper=dataclasses.replace(st.paper, texture_image=tmp_path / "tidak_ada.png"))
    with pytest.raises(StageError, match="tidak_ada"):
        pap.render_paper(OW, OW, missing)
    with pytest.raises(StageError, match="tidak_ada"):
        pap.validate_texture(missing)
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"bukan gambar")
    with pytest.raises(StageError, match="bukan gambar"):
        pap.render_paper(OW, OW, dataclasses.replace(st, paper=dataclasses.replace(st.paper, texture_image=bad)))
    black = tmp_path / "black.png"
    cv2.imwrite(str(black), np.zeros((20, 20), np.uint8))
    with pytest.raises(StageError, match="terlalu gelap"):
        pap.render_paper(OW, OW, dataclasses.replace(st, paper=dataclasses.replace(st.paper, texture_image=black)))
    flat = dataclasses.replace(st, paper=dataclasses.replace(st.paper, texture_opacity=0.0, texture_image=tmp_path / "tidak_ada.png"))
    pap.validate_texture(flat)                                         # kertas datar: berkas tidak dibutuhkan


# ── Determinisme / cache / ukuran ──────────────────
def test_paper_deterministic_cached_and_cache_independent(tex):
    st = pstyle(tex, **{"paper.vignette": 0.12})
    pap._PAPER_CACHE.clear()
    l1 = pap.render_paper(256, 456, st)
    assert pap.render_paper(256, 456, st) is l1 and not l1.f32.flags.writeable and not l1.u8.flags.writeable
    h1 = hashlib.sha256(l1.f32.tobytes() + l1.u8.tobytes()).hexdigest()
    pap._PAPER_CACHE.clear()
    l2 = pap.render_paper(256, 456, st)
    assert l2 is not l1 and hashlib.sha256(l2.f32.tobytes() + l2.u8.tobytes()).hexdigest() == h1
    assert l1.f32.shape == (456, 256, 3) and l1.u8.shape == (456, 256, 3) and l1.info["rotated"] is True


@pytest.mark.parametrize("w,h", [(256, 456), (456, 256), (300, 300), (500, 700)])
def test_other_canvas_sizes_orientation_and_fit(tmp_path, w, h):
    t = write_tex(tmp_path / "t.png", h=300, w=400)                    # landscape 400×300
    layer = pap.render_paper(w, h, pstyle(t))
    assert layer.f32.shape == (h, w, 3)
    assert layer.info["rotated"] == (h > w)
    assert layer.info["fit"] == ("native" if (w <= 400 and h <= 300) or (h > w and w <= 300 and h <= 400) else "scaled")


def test_real_asset_native_crop_1080x1922_stats():
    st = load_style(None)
    layer = pap.render_paper(1080, 1922, load_style(None, overrides={"paper.texture_gain": 1.0}))
    i = layer.info
    assert i["texture_sha256"] == ASSET_SHA and i["rotated"] is True and i["fit"] == "native" and i["image_size"] == {"width": 2048, "height": 1201}
    assert i["texture_std"] == pytest.approx(0.0137, abs=0.001)
    m = layer.u8.reshape(-1, 3).mean(axis=0, dtype=np.float64)
    assert np.allclose(m, [244, 241, 234], rtol=0.005)
    d = pap.render_paper(1080, 1922, st)                               # default final: gain 3
    assert d.f32[..., 1].std() == pytest.approx(3 * layer.f32[..., 1].std(), rel=0.05)


def test_paper_does_not_import_torch():
    import subprocess
    import sys
    code = ("import sys; from rotoscope import paper, export; from rotoscope.config import load_style; "
            "l = paper.render_paper(256, 456, load_style(None)); assert l.u8.shape == (456, 256, 3); "
            "assert 'torch' not in sys.modules, 'torch di-import'")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
