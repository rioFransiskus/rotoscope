"""Test multipass stage [5] (T-403): pass tambahan = medan koheren (mesin jitter T-402, salt per pass), statis / hold, opasitas "over"
(alpha tepat), SVG per pass, byte-identik T-402 pada 1 pass + opacity 1,0, invarian per pass (sambungan, seam, persilangan, tepi),
penjaga lipatan, manifest / basi / resume / jendela, data nyata. Metrik: tests/multipass_metrics.py + jitter_metrics.py."""

from __future__ import annotations

import dataclasses
import hashlib
import math
import xml.etree.ElementTree as ET

import cv2
import jitter_metrics as jm
import multipass_metrics as mm
import numpy as np
import pytest
import stylize_metrics as sm
from test_stylize import (H, N_FRAMES, OW, STYLE_YAML, W, cfg_for, circle, doc_for, make_stage_work, manifest, out_hashes, quiet,
                          run_main, stroke)
from test_stylize_jitter import REAL_PINNED, edge_doc, junction_doc, real_doc

from rotoscope import noise
from rotoscope import stylize as sty
from rotoscope.config import ConfigError, load_style, resolve_style

BASE = {"render.output_width": 1080, "stroke.width_base": 9.0, "shape.resample_points": 4}
MP = {"multipass.passes": 3, "multipass.offset": 2.7, "multipass.opacity_falloff": 0.55, "stroke.opacity": 1.0}   # pin: default Tahap 4 = 0,92


def st_mp(**ov):
    return load_style(None, overrides={**BASE, **MP, **ov})


def geo(**ov):
    return sty.make_geometry(st_mp(**ov), W, H)


def passes_of(doc, **ov):
    g = geo(**ov)
    return g, sty.frame_passes(doc, g)[0]


def render_bytes(doc, **ov):
    st = st_mp(**ov)
    g = sty.make_geometry(st, W, H)
    svg, png, _ = sty.render_frame(doc, g, st)
    return svg, png


def thick_line_doc(width_ref=60.0, index=0):
    return doc_for([stroke([(15.5, 50.5), (85.5, 50.5)], closed=False, typ="group_boundary", tid=1)], index)


THICK = {"stroke.width_base": 60.0, "stroke.width_variation": 0.0, "stroke.taper_ends": False}


def center_pixel(png: bytes, g) -> np.ndarray:
    return sm.decode_png(png)[g.out_h // 2, g.out_w // 2]


def lut_color(g, f: float) -> np.ndarray:
    return sty.ink_lut(g)[int(np.rint(np.float32(f) * (sty.MASK_LEVELS - 1)))]


# ── Default netral + geometri ──────────────────────
def test_defaults_are_rio_final_choice_and_neutral_override_is_legacy():
    """Default Tahap 4 (penilaian visual Rio, docs/04): 2 pass, offset 5,5, falloff 0,35, opacity 0,92, pass tambahan statis, jitter mati."""
    s = load_style(None)
    assert s.multipass.enabled and s.multipass.passes == 2 and s.multipass.offset == 5.5 and s.multipass.opacity_falloff == 0.35
    assert s.multipass.temporal_mode == "fixed" and s.stroke.opacity == 0.92 and s.jitter.amplitude == 0.0
    assert load_style(resolve_style("rough-sketch")).multipass == s.multipass and load_style(resolve_style("rough-sketch")).stroke.opacity == 0.92
    g = sty.make_geometry(s, W, H)
    assert not g.legacy and g.passes == 2 and [round(g.pass_alpha(k), 4) for k in range(2)] == [0.92, round(0.92 * 0.35, 4)]
    assert sty.make_geometry(load_style(None, overrides={"multipass.passes": 1, "stroke.opacity": 1.0}), W, H).legacy


def test_geometry_constants_follow_offset_and_unit():
    g = geo()
    assert math.isclose(g.mp_amp, 2.7 / sty.MULTIPASS_SEP_MEDIAN) and math.isclose(g.mp_cell, g.mp_amp / sty.MULTIPASS_FOLD_R)
    g2 = geo(**{"multipass.offset": 5.4})
    assert math.isclose(g2.mp_cell / g.mp_cell, 2.0) and math.isclose(g2.mp_amp / g2.mp_cell, sty.MULTIPASS_FOLD_R)      # skala medan MENGIKUTI offset
    gh = geo(**{"render.output_width": 540})
    assert math.isclose(gh.mp_amp, g.mp_amp / 2) and math.isclose(gh.mp_cell, g.mp_cell / 2)
    assert g.passes == 3 and len(g.mp_seeds) == 2 and geo(**{"multipass.enabled": False}).passes == 1
    assert sty.MULTIPASS_FOLD_R < sty.JITTER_FOLD_R_WARN


def test_pass_seeds_are_distinct_salted_streams():
    g = geo(**{"multipass.passes": 4})
    flat = [int(s) for pair in g.mp_seeds for s in pair]
    assert len(set(flat)) == len(flat) == 6
    assert not set(flat) & ({int(s) for s in g.jitter_seeds} | {g.noise_seed})
    assert g.mp_seeds[0] == tuple(noise.seed_of(0, sty.MULTIPASS_SALT_FIELD, 1, c) for c in range(sty.JITTER_CHANNELS))
    assert sty.MULTIPASS_SALT_FIELD not in (sty.JITTER_SALT_FIELD, sty.JITTER_SALT_TRACK)


def test_median_separation_equals_offset_and_scale_grows_with_it():
    xs, ys = np.meshgrid(np.linspace(80, 1000, 160), np.linspace(80, 1000, 160))
    pts = np.column_stack([xs.ravel(), ys.ravel()])
    for off in (2.7, 8.0, 12.0):
        g = geo(**{"multipass.offset": off})
        d = np.hypot(*sty.jitter_displacement(pts, [(0, len(pts))], [1], sty.pass_geometry(g, 1), 0).T) / g.unit
        assert 0.75 * off <= float(np.median(d)) <= 1.25 * off, (off, np.median(d))
        assert float(d.max()) <= off / sty.MULTIPASS_SEP_MEDIAN * math.sqrt(2) + 1e-6


def test_pass_field_is_2d_vector_coherent_and_independent_between_passes():
    g = geo()
    xs, ys = np.meshgrid(np.linspace(100, 900, 60), np.linspace(100, 900, 60))
    pts = np.column_stack([xs.ravel(), ys.ravel()])
    d1 = sty.jitter_displacement(pts, [(0, len(pts))], [1], sty.pass_geometry(g, 1), 0)
    d2 = sty.jitter_displacement(pts, [(0, len(pts))], [1], sty.pass_geometry(g, 2), 0)
    assert abs(float(np.corrcoef(d1[:, 0], d1[:, 1])[0, 1])) < 0.15                     # vektor 2D, bukan sepanjang normal
    assert abs(float(np.corrcoef(d1[:, 0], d2[:, 0])[0, 1])) < 0.15 and not np.array_equal(d1, d2)
    other = sty.jitter_displacement(pts[:5], [(0, 5)], [99], sty.pass_geometry(g, 1), 0)
    assert np.array_equal(d1[:5], other)                                                # lokasi sama → D sama, track_id tidak berpengaruh


# ── Byte-identik T-402 ─────────────────────────────
def reference_t402(doc, **ov):
    """Jalur T-402 ditulis ulang tanpa multipass: jitter → mask → LUT lama → PNG; SVG legacy."""
    st = st_mp(**{"multipass.passes": 1, **ov})
    g = sty.make_geometry(st, W, H)
    pieces, _ = sty.frame_pieces(doc, g)
    cov = cv2.resize(sty.render_mask(pieces, g), (g.out_w, g.out_h), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(sty.ink_lut(g)[cov], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, sty.PNG_COMPRESSION])
    return sty.render_svg(pieces, g, st), buf.tobytes()


def mixed_doc(index=3):
    doc = edge_doc(index)
    doc["strokes"] += junction_doc(index)["strokes"] + [stroke(circle(50.5, 50.5, 12), typ="silhouette", tid=40)]
    return doc


def test_one_pass_opacity_one_is_byte_identical_to_t402_for_any_other_multipass_param():
    doc = mixed_doc()
    for jit in ({}, {"jitter.amplitude": 3.0, "jitter.frequency": 0.04}):
        ref = reference_t402(doc, **jit)
        assert b'<g id="pass_' not in ref[0] and b'<g id="silhouette"' in ref[0]
        for ov in ({"multipass.passes": 1}, {"multipass.passes": 1, "multipass.offset": 9.0, "multipass.opacity_falloff": 0.1,
                                              "multipass.temporal_mode": "frame"},
                   {"multipass.enabled": False}, {"multipass.enabled": False, "multipass.passes": 5, "multipass.offset": 20.0},
                   {"multipass.passes": 1, "stroke.by_type.occlusion.opacity_scale": 1.0}):
            assert render_bytes(doc, **{**jit, "stroke.opacity": 1.0, **ov}) == ref, (jit, ov)


def test_real_neutral_style_is_t402_pinned_hashes():
    for clip, pinned in REAL_PINNED.items():
        doc = real_doc(clip, 80)
        st = load_style(resolve_style("rough-sketch"), overrides={"multipass.passes": 1, "stroke.opacity": 1.0})
        g = sty.make_geometry(st, 480, 854)
        svg, png, _ = sty.render_frame(doc, g, st)
        assert hashlib.sha256(svg).hexdigest() == pinned["frame_00080.svg"] and hashlib.sha256(png).hexdigest() == pinned["frame_00080.png"]


def test_opacity_below_one_or_extra_pass_changes_output():
    doc = mixed_doc()
    ref = reference_t402(doc)
    assert render_bytes(doc, **{"multipass.passes": 1, "stroke.opacity": 0.9}) != ref
    assert render_bytes(doc, **{"stroke.opacity": 1.0}) != ref
    svg1 = render_bytes(doc, **{"multipass.passes": 1, "stroke.opacity": 0.9})[0]
    assert b'<g id="pass_1" opacity="0.9000">' in svg1 and b'<g id="pass_1_silhouette"' in svg1


# ── Pass: isi dan waktu ────────────────────────────
def test_pass_zero_is_the_t402_line_with_or_without_jitter():
    doc = mixed_doc()
    for jit in ({}, {"jitter.amplitude": 3.0, "jitter.frequency": 0.04}):
        g, passes = passes_of(doc, **jit)
        ref, _ = sty.frame_pieces(doc, g)
        assert len(passes) == 3 and all(np.array_equal(a.points, b.points) for a, b in zip(passes[0], ref))
        for k in (1, 2):
            assert len(passes[k]) == len(ref)
            for a, b in zip(passes[k], ref):                                          # tebal / taper / flag disalin
                assert np.array_equal(a.widths, b.widths) and a.edge == b.edge and a.closed == b.closed and a.type == b.type


def test_offset_zero_passes_are_copies_of_pass_zero():
    g, passes = passes_of(mixed_doc(), **{"multipass.offset": 0.0})
    assert all(np.array_equal(a.points, b.points) for pk in passes[1:] for a, b in zip(passes[0], pk))


def test_fixed_mode_is_static_frame_mode_holds_in_pairs():
    pts = np.random.default_rng(3).uniform(100, 900, (300, 2))

    def d(g, f, k=1):
        return sty.jitter_displacement(pts, [(0, 300)], [1], sty.pass_geometry(g, k), f)
    fixed, move = geo(), geo(**{"multipass.temporal_mode": "frame", "jitter.hold_frames": 2})
    assert np.array_equal(d(fixed, 0), d(fixed, 17))
    assert np.array_equal(d(move, 0), d(move, 1)) and np.array_equal(d(move, 2), d(move, 3))
    assert not np.array_equal(d(move, 1), d(move, 2)) and np.array_equal(d(move, 0), d(fixed, 0))     # "frame" pada gambar 0 = irisan statis
    static_doc = [sty.frame_passes(mixed_doc(i), fixed)[0][1] for i in (0, 5, 40)]
    assert all(np.array_equal(a.points, b.points) for p in static_doc[1:] for a, b in zip(static_doc[0], p))


def test_frame_result_depends_only_on_absolute_frame_index_not_call_history():
    doc = mixed_doc(9)
    g = geo(**{"multipass.temporal_mode": "frame"})
    a = sty.frame_passes(doc, g)[0]
    for k in (0, 3, 5):
        sty.frame_passes(mixed_doc(k), g)
    b = sty.frame_passes(doc, g)[0]
    assert all(np.array_equal(x.points, y.points) for pa, pb in zip(a, b) for x, y in zip(pa, pb))
    assert render_bytes(doc) == render_bytes(doc)                                      # determinisme byte


# ── Opasitas "over": alpha tepat ───────────────────
def test_stacked_alpha_is_exact_in_png_center():
    doc = thick_line_doc()
    for passes, op, fall in ((3, 0.9, 0.5), (2, 0.85, 0.35), (3, 1.0, 0.8)):
        ov = {**THICK, "multipass.passes": passes, "multipass.offset": 0.0, "stroke.opacity": op, "multipass.opacity_falloff": fall}
        g = geo(**ov)
        _, png = render_bytes(doc, **ov)
        keep = 1.0
        for k in range(passes):
            keep *= 1.0 - op * fall ** k
        assert np.abs(center_pixel(png, g).astype(int) - lut_color(g, 1.0 - keep).astype(int)).max() <= 1, (passes, op, fall)


def test_over_is_not_sum_and_falloff_is_used():
    doc = thick_line_doc()
    ov = {**THICK, "multipass.passes": 2, "multipass.offset": 0.0, "stroke.opacity": 0.8}
    cols = {}
    for fall in (0.2, 0.6, 1.0):
        g = geo(**ov, **{"multipass.opacity_falloff": fall})
        f = 1 - (1 - 0.8) * (1 - 0.8 * fall)
        cols[fall] = center_pixel(render_bytes(doc, **ov, **{"multipass.opacity_falloff": fall})[1], g)
        assert np.abs(cols[fall].astype(int) - lut_color(g, f).astype(int)).max() <= 1 and f <= 1.0     # penjumlahan 0,8 + 0,8 f akan > 1
    assert not np.array_equal(cols[0.2], cols[0.6]) and not np.array_equal(cols[0.6], cols[1.0])


def test_single_pass_opacity_scales_alpha_and_type_scale_applies():
    doc = thick_line_doc()
    ov = {**THICK, "multipass.passes": 1, "stroke.opacity": 0.6}
    g = geo(**ov)
    assert np.abs(center_pixel(render_bytes(doc, **ov)[1], g).astype(int) - lut_color(g, 0.6).astype(int)).max() <= 1
    ov2 = {**ov, "stroke.opacity": 1.0, "stroke.by_type.group_boundary.opacity_scale": 0.5}
    assert np.abs(center_pixel(render_bytes(doc, **ov2)[1], g).astype(int) - lut_color(g, 0.5).astype(int)).max() <= 1
    ov3 = {**THICK, "multipass.passes": 2, "multipass.offset": 0.0, "stroke.opacity": 0.8, "multipass.opacity_falloff": 0.5,
           "stroke.by_type.group_boundary.opacity_scale": 0.5}                         # a_k × scale: 0,8 × 0,5 dan 0,4 × 0,5
    g3 = geo(**ov3)
    f = 1 - (1 - 0.8 * 0.5) * (1 - 0.4 * 0.5)
    assert np.abs(center_pixel(render_bytes(doc, **ov3)[1], g3).astype(int) - lut_color(g3, f).astype(int)).max() <= 1


def test_union_inside_a_pass_junction_not_darker():
    """Tipe berbeda ber-scale 1,0 yang bertumpuk dalam SATU pass = union (tidak lebih gelap di sambungan), antar pass "over"."""
    a = stroke([(15.5, 50.5), (85.5, 50.5)], closed=False, typ="group_boundary", tid=1)
    b = stroke([(15.5, 50.5), (85.5, 50.5)], closed=False, typ="silhouette", tid=2)
    ov = {**THICK, "multipass.passes": 1, "stroke.opacity": 0.7}
    g = geo(**ov)
    both = sm.decode_png(sty.render_frame(doc_for([a, b]), g, st_mp(**ov))[1])[g.out_h // 2, g.out_w // 2]
    assert np.abs(both.astype(int) - lut_color(g, 0.7).astype(int)).max() <= 1


def test_over_is_commutative_in_pass_order():
    g, passes = passes_of(mixed_doc(), **{"stroke.opacity": 0.9, "multipass.offset": 5.5})
    f = sty.ink_fraction(passes, g)
    covs = [cv2.resize(sty.render_mask(p, g), (g.out_w, g.out_h), interpolation=cv2.INTER_AREA).astype(np.float64) / 255 for p in passes]
    alphas = [g.pass_alpha(k) for k in range(len(passes))]
    assert np.abs(f - mm.over(covs, alphas)).max() < 1e-5
    assert np.abs(f - mm.over(covs[::-1], alphas[::-1])).max() < 1e-5               # urutan pass tidak berpengaruh (pasangan cov-alpha tetap)
    swapped = sty.ink_fraction(passes[::-1], g)                                      # alpha menempel indeks: BERBEDA bila alpha berbeda → peran pass memang ditentukan indeks
    assert float(np.abs(swapped - f).max()) > 0.0


@pytest.mark.parametrize("ov", [{"stroke.opacity": 0.92}, {"stroke.opacity": 1.0, "multipass.passes": 2, "multipass.offset": 12.0},
                                {"stroke.opacity": 0.9, "stroke.by_type.group_boundary.opacity_scale": 0.6, "stroke.by_type.silhouette.opacity_scale": 0.8}])
def test_windowed_ink_fraction_is_bit_identical_to_full_frame_reference(ov):
    """Optimasi biaya (jendela tinta + tabel 256 entri) TIDAK mengubah hasil: sama dengan implementasi acuan penuh-frame, bit demi bit."""
    for doc in (mixed_doc(), edge_doc(2), doc_for([], 1)):
        g, passes = passes_of(doc, **ov)
        assert np.array_equal(sty.ink_fraction(passes, g), mm.ink_fraction_reference(passes, g))


def test_render_mask_window_equals_slice_of_full_mask_and_box_covers_all_ink():
    g, passes = passes_of(mixed_doc(), **{"stroke.opacity": 0.92})
    full = sty.render_mask(passes[1], g)
    box = sty.ink_box(passes, g)
    x0, y0, x1, y1 = box
    win = sty.render_mask(passes[1], g, box)
    assert np.array_equal(win, full[y0 * g.ss:y1 * g.ss, x0 * g.ss:x1 * g.ss])
    outside = full.copy()
    outside[y0 * g.ss:y1 * g.ss, x0 * g.ss:x1 * g.ss] = 0
    assert not outside.any()                                                         # tidak ada tinta di luar jendela
    assert sty.ink_box([[]], g) is None and sty.ink_box([], g) is None


def test_extra_passes_use_wider_edge_dead_zone():
    g = geo()
    gp = sty.pass_geometry(g, 1)
    assert math.isclose(gp.edge_dead, g.edge_dead + sty.MULTIPASS_EDGE_EXTRA_PX) and gp.edge_dead > g.widths["silhouette"] / 2 + sty.EDGE_MARGIN_PX
    inside_dead = np.array([[gp.edge_dead - 0.01, 400.0], [400.0, g.out_h - gp.edge_dead + 0.01]])
    assert not sty.jitter_displacement(inside_dead, [(0, 2)], [1], gp, 0).any()      # zona mati: tidak bergeser sama sekali


# ── SVG ────────────────────────────────────────────
def test_svg_structure_pass_groups_unique_ids_and_opacity():
    doc = mixed_doc()
    svg, _ = render_bytes(doc, **{"stroke.opacity": 0.92})
    root = ET.fromstring(svg)
    ids = [e.get("id") for e in root.iter() if e.get("id")]
    assert len(ids) == len(set(ids))
    assert [e.get("id") for e in root if e.tag.endswith("g")] == ["pass_1", "pass_2", "pass_3"]
    ops = [e.get("opacity") for e in root if e.tag.endswith("g")]
    assert ops == [f"{0.92 * 0.55 ** k:.4f}" for k in range(3)]
    for k in (1, 2, 3):
        assert [c.get("id") for c in root.find(f"{{*}}g[@id='pass_{k}']")] == [f"pass_{k}_{t}" for t in sty.TYPE_ORDER]
    svg2, _ = render_bytes(doc, **{"stroke.by_type.occlusion.opacity_scale": 0.5, "stroke.opacity": 0.9})
    assert svg2.count(b'id="pass_1_occlusion" fill="#1a1a1a" fill-rule="nonzero" stroke="none" opacity="0.5000"') == 1
    assert b'opacity="0.5000">' not in svg2.split(b'id="pass_1_silhouette"')[1].split(b"</g>")[0]
    assert svg2 == render_bytes(doc, **{"stroke.by_type.occlusion.opacity_scale": 0.5, "stroke.opacity": 0.9})[0]


@pytest.mark.parametrize("ov", [{"stroke.opacity": 1.0}, {"stroke.opacity": 0.92, "multipass.opacity_falloff": 0.55},
                                {"stroke.opacity": 0.85, "multipass.offset": 8.0, "multipass.opacity_falloff": 0.8},
                                {"multipass.passes": 2, "multipass.offset": 12.0, "stroke.opacity": 0.9},
                                {"stroke.opacity": 0.9, "stroke.by_type.group_boundary.opacity_scale": 0.6}])
def test_svg_png_alpha_equivalence(ov):
    doc = junction_doc(6)
    doc["strokes"] += edge_doc(6)["strokes"]
    st = st_mp(**ov)
    g = sty.make_geometry(st, W, H)
    svg, png, _ = sty.render_frame(doc, g, st)
    rep = mm.alpha_report(mm.svg_ink_fraction(svg, g), mm.png_ink_fraction(sm.decode_png(png), g))
    assert rep["iou"] >= sm.SVG_PNG_IOU_MIN and rep["l1"] <= sm.SVG_PNG_L1_MAX, rep
    assert rep["big_diff_frac"] <= sm.SVG_PNG_BIG_DIFF_MAX and abs(rep["mass_ratio"] - 1) <= sm.INK_MASS_TOL, rep


# ── Invarian per pass (sintetis) ───────────────────
OFFSETS = [2.7, 5.5, 8.0, 12.0]


def check_invariants(doc, ov, index):
    g, passes = passes_of(doc, **ov)
    for r in mm.pass_invariants(passes, g, index):
        assert r["joint_ratio"] <= 1.0, r
        assert r["new_crossings"] == 0 and r["edge_ink_changed"] == 0, r
        assert r["seam_fail"] == 0 and r["intrusion_excess"] <= 1e-9, r
        assert r["min_det"] > sty.JACOBIAN_MIN_DET and r["max_displacement_over_bound"] <= 1.0 + 1e-9, r
    return g, passes


@pytest.mark.parametrize("off", OFFSETS)
def test_t_junctions_stay_joined_per_pass(off):
    g, passes = check_invariants(junction_doc(2), {"multipass.offset": off}, 2)
    ch = jm.joint_changes(passes[0], passes[1], sty.pass_geometry(g, 1))
    assert len(ch) >= 11 and jm.max_displacement(passes[0], passes[1]) > 0.5               # pass benar-benar bergeser


@pytest.mark.parametrize("off", OFFSETS)
def test_closed_seam_and_parallel_lines_per_pass(off):
    ring = doc_for([stroke(circle(50.5, 50.5, 30), typ="silhouette", tid=1)], 4)
    check_invariants(ring, {"multipass.offset": off}, 4)
    lines = [stroke([(15.5, 30.5 + 0.9 * k), (85.5, 30.5 + 0.9 * k)], closed=False, typ="group_boundary", tid=1 + k) for k in range(3)]
    check_invariants(doc_for(lines, 2), {"multipass.offset": off}, 2)


@pytest.mark.parametrize("off", OFFSETS)
def test_canvas_edges_and_corners_per_pass(off):
    for index in (0, 2, 5):
        g, passes = check_invariants(edge_doc(index), {"multipass.offset": off}, index)
        assert sum(pc.edge[0] or pc.edge[1] for pc in passes[0]) >= 6
        assert max(jm.max_displacement(passes[0], p) for p in passes[1:]) > 0.5


def test_short_stroke_empty_frame_and_tight_bend_survive_multipass():
    g = geo(**{"multipass.offset": 8.0})
    tiny = stroke([(50.5, 50.5), (50.8, 50.6)], closed=False, typ="occlusion", tid=3)
    for strokes in ([], [tiny]):
        passes, stats = sty.frame_passes(doc_for(strokes, 2), g)
        assert len(passes) == 3 and all(len(p) == len(strokes) for p in passes) and stats["passes"] == 3
        st = st_mp(**{"multipass.offset": 8.0})
        svg, png, _ = sty.render_frame(doc_for(strokes, 2), g, st)
        assert ET.fromstring(svg) is not None and png.startswith(b"\x89PNG")
    bend = stroke([(30.5, 30.5), (50.5, 31.5), (31.5, 33.5), (60.5, 35.5)], closed=False, typ="occlusion", tid=4)
    check_invariants(doc_for([bend], 1), {"multipass.offset": 8.0}, 1)


def test_separation_stats_report_median_close_to_offset():
    doc = mixed_doc()
    g, passes = passes_of(doc, **{"multipass.offset": 8.0})
    st = mm.separation_stats(passes[0], g, 3)
    assert st["p50"] < st["p95"] <= st["max"] and 4.0 <= st["p50"] <= 12.0


# ── Penjaga lipatan ────────────────────────────────
def test_fold_warning_adds_multipass_r_to_jitter_r():
    assert sty.fold_warning(st_mp()) is None                                          # r multipass 0,15 saja ≤ 0,19
    w = sty.fold_warning(st_mp(**{"jitter.amplitude": 2.0}))                          # 0,106 + 0,15 = 0,256
    assert w is not None and "multipass" in w and "0.256" in w
    assert sty.fold_warning(st_mp(**{"multipass.passes": 1, "jitter.amplitude": 2.0})) is None       # tanpa pass efektif: hanya r jitter 0,106
    assert sty.fold_warning(st_mp(**{"multipass.enabled": False, "jitter.amplitude": 2.0})) is None
    assert sty.fold_warning(st_mp(**{"multipass.offset": 0.0, "jitter.amplitude": 2.0})) is None
    assert sty.fold_warning(st_mp(**{"multipass.passes": 1, "jitter.amplitude": 4.0})) is not None   # peringatan jitter lama (r 0,212)


def test_fold_warning_printed_once_and_manifest_block(tmp_path, capsys):
    work = make_stage_work(tmp_path)
    st = st_mp(**{"jitter.amplitude": 2.0, "stroke.opacity": 0.92})
    run = sty.run_stylize(cfg_for(work), st, "t", log=quiet)
    assert run["processed"] == N_FRAMES and capsys.readouterr().err.count("PERINGATAN: r jitter") == 1
    m = manifest(work)["multipass"]
    assert m["passes"] == 3 and m["alphas"] == [0.92, 0.506, 0.2783] and m["temporal_mode"] == "fixed" and m["fold_r"] == 0.15
    assert manifest(work)["style_params"]["multipass.offset"] == 2.7 and manifest(work)["style_params"]["stroke.opacity"] == 0.92


# ── Config ─────────────────────────────────────────
@pytest.mark.parametrize("key,bad", [("multipass.temporal_mode", "x"), ("multipass.passes", 0), ("multipass.offset", -1),
                                     ("multipass.opacity_falloff", 1.5), ("stroke.opacity", 1.2),
                                     ("stroke.by_type.silhouette.opacity_scale", -0.1)])
def test_multipass_params_validated(key, bad):
    with pytest.raises(ConfigError):
        load_style(None, overrides={key: bad})


# ── Run: manifest, basi, resume, jendela ───────────
MP_YAML = (STYLE_YAML.replace("opacity: 1.0", "opacity: 0.92").replace("multipass:\n  passes: 1\n", "")
           + "multipass:\n  passes: 3\n  offset: 8.0\n  opacity_falloff: 0.55\n")


@pytest.fixture
def mp_style(tmp_path):
    p = tmp_path / "mp-style.yaml"
    p.write_text(MP_YAML, encoding="utf-8")
    return p


def test_run_multipass_deterministic_resume_limit_from_identical(tmp_path, mp_style):
    work = make_stage_work(tmp_path, n=6)
    assert run_main(work, mp_style) == 0
    full = out_hashes(work)
    assert len(full) == 12 and manifest(work)["contract"] == "T-403" and manifest(work)["multipass"]["passes"] == 3
    assert run_main(work, mp_style, "--restart") == 0 and out_hashes(work) == full
    assert run_main(work, mp_style, "--restart", "--limit", "3") == 0
    assert out_hashes(work) == {k: v for k, v in full.items() if int(k[6:11]) < 3}
    assert run_main(work, mp_style) == 0 and out_hashes(work) == full
    for p in list(work.glob("strokes/frame_0000[34].*")):
        p.unlink()
    assert run_main(work, mp_style, "--from", "3", "--limit", "2") == 0 and out_hashes(work) == full


def test_window_equals_full_run_in_frame_mode(tmp_path, mp_style):
    fm = tmp_path / "frame-style.yaml"
    fm.write_text(MP_YAML + "  temporal_mode: frame\n", encoding="utf-8")
    full_work, win_work = make_stage_work(tmp_path / "f", n=6), make_stage_work(tmp_path / "w", n=6)
    assert run_main(full_work, fm) == 0 and run_main(win_work, fm, "--from", "3", "--limit", "2") == 0
    full, win = out_hashes(full_work), out_hashes(win_work)
    assert set(win) == {"frame_00003.png", "frame_00003.svg", "frame_00004.png", "frame_00004.svg"} and all(win[k] == full[k] for k in win)
    assert out_hashes(full_work) != {}


@pytest.mark.parametrize("ov", [{"multipass.passes": 2}, {"multipass.offset": 5.5}, {"multipass.opacity_falloff": 0.3},
                                {"multipass.temporal_mode": "frame"}, {"multipass.enabled": False}, {"stroke.opacity": 0.8},
                                {"stroke.by_type.occlusion.opacity_scale": 0.5}])
def test_new_active_params_make_strokes_stale(tmp_path, ov):
    work = make_stage_work(tmp_path)
    sty.run_stylize(cfg_for(work), st_mp(), "t", log=quiet)
    logs: list[str] = []
    run = sty.run_stylize(cfg_for(work), st_mp(**ov), "t", log=logs.append)
    assert run["processed"] == N_FRAMES and any("style_params." in x for x in logs), ov


def test_old_t402_strokes_are_stale_and_recomputed(tmp_path, mp_style):
    import json
    work = make_stage_work(tmp_path, n=3)
    assert run_main(work, mp_style) == 0
    p = work / "strokes" / "manifest.json"
    m = json.loads(p.read_text(encoding="utf-8"))
    m["contract"] = "T-402"
    p.write_text(json.dumps(m), encoding="utf-8")
    logs: list[str] = []
    run = sty.run_stylize(cfg_for(work), load_style(mp_style), "t", log=logs.append)
    assert run["processed"] == 3 and any("contract" in x and "T-403" in x for x in logs)


# ── Data nyata ─────────────────────────────────────
@pytest.mark.parametrize("clip", ["test_short", "test"])
@pytest.mark.parametrize("off", [2.7, 8.0, 12.0])
def test_real_invariants_per_pass_and_alpha_equivalence(clip, off):
    doc = real_doc(clip, 80)
    st = load_style(None, overrides={"multipass.passes": 3, "multipass.offset": off, "stroke.opacity": 0.92})
    g = sty.make_geometry(st, 480, 854)
    passes = sty.frame_passes(doc, g)[0]
    for r in mm.pass_invariants(passes, g, 80):
        assert r["joint_ratio"] <= 1.0 and r["new_crossings"] == 0 and r["edge_ink_changed"] == 0, r
        assert r["seam_fail"] == 0 and r["intrusion_excess"] <= 1e-9 and r["min_det"] > sty.JACOBIAN_MIN_DET, r
    svg, png, _ = sty.render_frame(doc, g, st)
    rep = mm.alpha_report(mm.svg_ink_fraction(svg, g), mm.png_ink_fraction(sm.decode_png(png), g))
    assert rep["iou"] >= sm.SVG_PNG_IOU_MIN and rep["l1"] <= sm.SVG_PNG_L1_MAX, rep
    assert rep["big_diff_frac"] <= sm.SVG_PNG_BIG_DIFF_MAX and abs(rep["mass_ratio"] - 1) <= sm.INK_MASS_TOL, rep


@pytest.mark.parametrize("off", [5.5, 8.0, 12.0])
def test_real_edge_ink_unchanged_on_frame_that_failed_before_wider_dead_zone(off):
    """Klip `test` frame 207 (offset 5,5 / 8): 3 piksel tinta di 3 baris / kolom terluar berubah dengan zona mati T-402 (Tahap 3)."""
    doc = real_doc("test", 207)
    g = sty.make_geometry(load_style(None, overrides={"multipass.passes": 3, "multipass.offset": off}), 480, 854)
    passes = sty.frame_passes(doc, g)[0]
    for r in mm.pass_invariants(passes, g, 207):
        assert r["edge_ink_changed"] == 0 and r["seam_fail"] == 0, r


def test_stylize_multipass_does_not_import_torch():
    import subprocess
    import sys
    code = ("import sys; from rotoscope import stylize; from rotoscope.config import load_style; "
            "st = load_style(None, overrides={'multipass.passes': 3, 'stroke.opacity': 0.9}); g = stylize.make_geometry(st, 100, 100); "
            "s = {'track_id': 1, 'type': 'group_boundary', 'closed': False, 'groups': ['a', 'b'], 'points': [[10.5, 50.5], [80.5, 50.5]]}; "
            "stylize.render_frame({'frame_index': 0, 'width': 100, 'height': 100, 'source': {}, 'prev_sha256': None, 'strokes': [s]}, g, st); "
            "assert 'torch' not in sys.modules, 'torch di-import'")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
