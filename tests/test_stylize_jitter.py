"""Test jitter stage [5] (T-402): medan perpindahan koheren (vektor 2D), noise 3D + indeks gambar / hold / fixed / drift, komponen independen
`stroke_independence`, pelunakan tepi, amplitudo 0 = T-401 byte-identik, penjaga lipatan, manifest / basi / resume / jendela, data nyata.
Metrik: tests/jitter_metrics.py (ambang = konstanta bernama di stylize.py)."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from pathlib import Path

import jitter_metrics as jm
import numpy as np
import pytest
import stylize_metrics as sm
from test_stylize import (H, N_FRAMES, W, circle, cfg_for, doc_for, lattice, make_stage_work, manifest, out_hashes, quiet,
                          run_main, stroke, style_file)

from rotoscope import noise
from rotoscope import stylize as sty
from rotoscope.config import ConfigError, load_style, resolve_style

BASE = {"render.output_width": 1080, "stroke.width_base": 9.0, "shape.resample_points": 4}
SAFE = {"jitter.amplitude": 4.0, "jitter.frequency": 0.053,         # r = 0,21 (aman)
        "multipass.passes": 1, "stroke.opacity": 1.0}               # test jitter T-402: garis tunggal solid (default T-403 = 2 pass, opacity 0,92)
ROOT = Path(__file__).resolve().parents[1]
CLIPS = ROOT / "work" / "clips"


def st_for(**ov):
    return load_style(None, overrides={**BASE, **SAFE, **ov})


def geo(**ov):
    return sty.make_geometry(st_for(**ov), W, H)


def disp(g, frame_index, pts, tid=1):
    return jm.displacement_at(g, frame_index, np.asarray(pts, float), tid)


def junction_doc(index=0, n=11, gap=0.3):
    """Garis tuan rumah horizontal (silhouette terbuka, tipe berbeda) + n garis batas grup yang berakhir `gap` px kerja di
    bawahnya (sambungan T dua tipe)."""
    rect = stroke([(15.5, 20.5), (85.5, 20.5)], closed=False, typ="silhouette", tid=1)
    lines = [stroke([(x + 0.5, 60.5), (x + 0.5, 20.5 + gap)], closed=False, typ="group_boundary", tid=2 + k)
             for k, x in enumerate(np.linspace(25, 75, n).round())]
    return doc_for([rect, *lines], index)


def base_and_jit(doc, **ov):
    g = geo(**ov)
    base, _ = sty.base_pieces(doc, g)
    return g, base, sty.jitter_pieces(base, g, int(doc["frame_index"]))


# ── noise 3D ───────────────────────────────────────
def grid3(n=120, span=40.0):
    x, y = np.meshgrid(np.linspace(0, span, n), np.linspace(0, span, n))
    return x.ravel(), y.ravel()


def test_noise3d_range_zero_mean_and_deterministic():
    x, y = grid3()
    sd = noise.seed_of(0, sty.JITTER_SALT_FIELD, 0)
    for t in (0.0, 0.35, 0.5, 1.7, 12.9):
        v = noise.value_noise_3d(sd, x, y, t)
        assert float(v.min()) >= -1.0 and float(v.max()) <= 1.0 and abs(float(v.mean())) < 0.06
        assert np.array_equal(v, noise.value_noise_3d(sd, x, y, t))


def test_noise3d_integer_time_is_a_pure_slice_and_continuous():
    x, y = grid3(40, 9.0)
    sd = noise.seed_of(0, sty.JITTER_SALT_FIELD, 0)
    a = noise.value_noise_3d(sd, x, y, 2.0)
    assert np.abs(noise.value_noise_3d(sd, x, y, 2.0 + 1e-9) - a).max() < 1e-6
    assert np.abs(noise.value_noise_3d(sd, x, y, 3.0 - 1e-9) - noise.value_noise_3d(sd, x, y, 3.0)).max() < 1e-6


def test_noise3d_rms_does_not_breathe_over_time():
    """Equal-power: RMS tetap antar gambar (trilinear turun ±30% di tengah sel waktu)."""
    x, y = grid3(150, 60.0)
    sd = noise.seed_of(0, sty.JITTER_SALT_FIELD, 0)
    rms = np.array([float(np.sqrt((noise.value_noise_3d(sd, x, y, 0.35 * k) ** 2).mean())) for k in range(24)])
    assert rms.min() / rms.max() > 0.9, rms


def test_noise3d_time_correlation_decreases_with_drift():
    x, y = grid3(150, 60.0)
    sd = noise.seed_of(0, sty.JITTER_SALT_FIELD, 0)

    def corr(drift):
        c = []
        for k in range(20):
            a, b = noise.value_noise_3d(sd, x, y, k * drift), noise.value_noise_3d(sd, x, y, (k + 1) * drift)
            c.append(float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum())))
        return float(np.mean(c))
    cs = [corr(d) for d in (0.15, 0.35, 0.7, 1.0)]
    assert cs[0] > 0.9 > cs[1] > cs[2] > cs[3] and abs(cs[3]) < 0.1 and 0.7 < cs[1] < 0.9, cs


def test_noise3d_channels_independent_and_streams_separate_from_width_noise():
    x, y = grid3(150, 60.0)
    s0, s1 = (noise.seed_of(0, sty.JITTER_SALT_FIELD, c) for c in (0, 1))
    a, b = noise.value_noise_3d(s0, x, y, 0.3), noise.value_noise_3d(s1, x, y, 0.3)
    assert abs(float(np.corrcoef(a, b)[0, 1])) < 0.1
    wseed = noise.seed_of(0)                                              # stream tebal T-401: tanpa salt
    assert wseed not in (s0, s1)
    w = noise.value_noise_2d(wseed, x, y)
    assert abs(float(np.corrcoef(a, w)[0, 1])) < 0.1 and abs(float(np.corrcoef(b, w)[0, 1])) < 0.1
    assert noise.seed_of(0, sty.JITTER_SALT_FIELD, 0) != noise.seed_of(0, sty.JITTER_SALT_TRACK, 1, 0)


def test_geometry_jitter_seeds_come_from_salted_streams_distinct_from_width_stream():
    for ps in (0, 7):
        g = geo(**{"jitter.param_seed": ps, "stroke.width_variation": 0.5})
        expect = tuple(noise.seed_of(ps, sty.JITTER_SALT_FIELD, c) for c in range(sty.JITTER_CHANNELS))
        assert g.jitter_seeds == expect and len(set(g.jitter_seeds)) == sty.JITTER_CHANNELS
        assert g.noise_seed == int(noise.seed_of(ps)) and g.noise_seed not in {int(s) for s in g.jitter_seeds}


def test_amplitude_and_cell_scale_with_output_unit():
    g, g2 = geo(), geo(**{"render.output_width": 540})
    assert math.isclose(g.jitter_amp, 4.0) and math.isclose(g.jitter_cell, 1.0 / 0.053)
    assert math.isclose(g2.jitter_amp, 2.0) and math.isclose(g2.jitter_cell, 0.5 / 0.053)             # unit = 540 / 1080
    assert math.isclose(g2.edge_fade_len, 20.0)


def test_value_noise_2d_unchanged_by_t402():
    """Stream tebal T-401 TIDAK berubah (nilai emas 2D); amplitudo 0 identik T-401 bergantung padanya."""
    x, y = grid3(40, 9.0)
    h = hashlib.sha256(np.round(noise.value_noise_2d(noise.seed_of(0), x, y), 12).tobytes()).hexdigest()
    assert h == hashlib.sha256(np.round(noise.value_noise_2d(noise.seed_of(0), x, y), 12).tobytes()).hexdigest()
    assert abs(float(noise.value_noise_2d(noise.seed_of(0), np.array([3.25]), np.array([-1.5]))[0]) - 0.2941145434226272) < 1e-12


# ── Medan: batas, koheren, vektor ──────────────────
def test_displacement_bounded_rms_and_geometry_constants():
    g = geo(**{"jitter.amplitude": 6.0, "jitter.frequency": 0.03})
    x, y = np.meshgrid(np.linspace(200, 880, 120), np.linspace(200, 880, 120))
    pts = np.column_stack([x.ravel(), y.ravel()])
    st = jm.jitter_stats(g, 3, pts)
    assert st["max_norm_over_amp"] <= math.sqrt(2) + 1e-9 and st["max_abs_over_amp"] <= 1.0 + 1e-9
    assert 0.3 < st["rms_per_channel_over_amp"] < 0.55                       # terukur 0,40–0,47 × amplitudo
    assert math.isclose(jm.displacement_bound(g), 6.0 * math.sqrt(2))


def test_coherent_same_location_same_displacement_regardless_of_stroke_and_track():
    g = geo()
    p = np.array([[400.0, 500.0], [403.7, 512.2]])
    assert np.array_equal(disp(g, 5, p, tid=1), disp(g, 5, p, tid=99))
    assert np.array_equal(disp(g, 5, p[:1], tid=1), disp(g, 5, p, tid=1)[:1])        # titik yang sama, daftar berbeda


def test_displacement_is_a_2d_vector_not_along_stroke_normal():
    g = geo()
    x, y = np.meshgrid(np.linspace(100, 900, 60), np.linspace(100, 900, 60))
    d = disp(g, 2, np.column_stack([x.ravel(), y.ravel()]))
    assert abs(float(np.corrcoef(d[:, 0], d[:, 1])[0, 1])) < 0.15 and d[:, 0].std() > 0.5 and d[:, 1].std() > 0.5


def test_t_junctions_stay_joined_under_coherent_field():
    doc = junction_doc()
    g, base, jit = base_and_jit(doc, **{"jitter.amplitude": 6.0, "jitter.frequency": 0.026})
    ch = jm.joint_changes(base, jit, g)
    assert len(ch) >= 11
    assert float(ch.max()) <= jm.joint_tolerance(g), (ch.max(), jm.joint_tolerance(g))
    assert jm.max_displacement(base, jit) > 1.0                                       # jitter benar-benar bergerak


def test_independent_per_stroke_opens_junctions_and_metric_detects_it():
    """s = 1 (semua independen per track_id) membuka sambungan melewati toleransi koheren: metrik membedakan keduanya."""
    doc = junction_doc()
    g, base, jit = base_and_jit(doc, **{"jitter.amplitude": 6.0, "jitter.frequency": 0.026, "jitter.stroke_independence": 1.0})
    assert float(jm.joint_changes(base, jit, g).max()) > jm.joint_tolerance(g)


def test_closed_stroke_seam_is_continuous():
    doc = doc_for([stroke(circle(50.5, 50.5, 30), typ="silhouette", tid=1)], 4)
    g, base, jit = base_and_jit(doc, **{"jitter.amplitude": 8.0, "jitter.frequency": 0.026})
    b, j = base[0], jit[0]
    assert j.closed and jm.seam_ok(b, j, g) and jm.seam_ratio(b, j) <= jm.SEAM_REL_TOL


def test_two_close_parallel_lines_do_not_cross_under_coherent_field():
    lines = [stroke([(15.5, 30.5 + 0.9 * k), (85.5, 30.5 + 0.9 * k)], closed=False, typ="group_boundary", tid=1 + k) for k in range(3)]
    g, base, jit = base_and_jit(doc_for(lines, 2), **{"jitter.amplitude": 6.0, "jitter.frequency": 0.026})
    assert jm.new_crossings(base, jit) == 0 and jm.pieces_jacobian_min_det(jit, g, 2) > sty.JACOBIAN_MIN_DET


def test_jacobian_detects_folds_at_high_r_and_passes_at_safe_r():
    pts = np.column_stack([np.linspace(100, 900, 4000), np.linspace(150, 850, 4000)])
    xs, ys = np.meshgrid(np.linspace(100, 900, 70), np.linspace(100, 900, 70))
    grid = np.column_stack([xs.ravel(), ys.ravel()])
    safe = geo(**{"jitter.amplitude": 2.0, "jitter.frequency": 0.053})
    fold = geo(**{"jitter.amplitude": 8.0, "jitter.frequency": 0.053})
    assert float(jm.jacobian_dets(safe, 1, grid).min()) > 0.3
    assert float(jm.jacobian_dets(fold, 1, grid).min()) < sty.JACOBIAN_MIN_DET
    assert pts.shape == (4000, 2)


# ── stroke_independence ────────────────────────────
@pytest.mark.parametrize("s", [0.0, 0.25, 0.5, 1.0])
def test_independence_keeps_variance_and_respects_bound(s):
    tracks = list(range(1, 25))
    g = geo(**{"jitter.amplitude": 6.0, "jitter.frequency": 0.03, "jitter.stroke_independence": s})
    rng = np.random.default_rng(0)
    ds = []
    for tid in tracks:
        pts = rng.uniform(100, 900, (600, 2))
        ds.append(disp(g, 7, pts, tid))
    d = np.vstack(ds)
    rms = float(math.sqrt((d ** 2).mean()))
    ref = geo(**{"jitter.amplitude": 6.0, "jitter.frequency": 0.03})
    rms0 = float(math.sqrt((np.vstack([disp(ref, 7, rng.uniform(100, 900, (600, 2)), t) for t in tracks]) ** 2).mean()))
    assert abs(rms / rms0 - 1) < 0.1, (s, rms, rms0)                              # varians dijaga
    assert float(np.hypot(*d.T).max()) <= jm.displacement_bound(g) + 1e-9


def test_independence_zero_equals_field_only_and_nonzero_differs_by_track():
    pts = np.random.default_rng(1).uniform(100, 900, (200, 2))
    g0 = geo()
    g0b = geo(**{"jitter.stroke_independence": 0.0})
    assert np.array_equal(disp(g0, 3, pts, 1), disp(g0b, 3, pts, 77))              # s = 0: track_id tidak berpengaruh
    g5 = geo(**{"jitter.stroke_independence": 0.5})
    assert not np.array_equal(disp(g5, 3, pts, 1), disp(g5, 3, pts, 2))
    assert np.array_equal(disp(g5, 3, pts, 1), disp(g5, 3, pts, 1))


# ── Waktu: gambar, hold, fixed, drift ──────────────
def test_image_index_hold_fixed_and_absolute_frame_index():
    g = geo(**{"jitter.hold_frames": 2})
    assert [sty.jitter_image_index(f, g) for f in range(6)] == [0, 0, 1, 1, 2, 2]
    assert [sty.jitter_image_index(f, geo(**{"jitter.hold_frames": 3})) for f in range(7)] == [0, 0, 0, 1, 1, 1, 2]
    assert all(sty.jitter_image_index(f, geo(**{"jitter.temporal_seed_mode": "fixed"})) == 0 for f in range(9))


def test_hold_two_repeats_displacement_in_pairs_and_changes_between_pairs():
    pts = np.random.default_rng(2).uniform(100, 900, (300, 2))
    g = geo(**{"jitter.hold_frames": 2})
    d = [disp(g, f, pts) for f in range(5)]
    assert np.array_equal(d[0], d[1]) and np.array_equal(d[2], d[3])
    assert not np.array_equal(d[1], d[2]) and not np.array_equal(d[3], d[4])


def test_fixed_mode_and_zero_drift_are_static_frame_mode_moves():
    pts = np.random.default_rng(3).uniform(100, 900, (300, 2))
    fixed, still, move = (geo(**{"jitter.temporal_seed_mode": "fixed"}), geo(**{"jitter.temporal_drift": 0.0}), geo())
    assert np.array_equal(disp(fixed, 0, pts), disp(fixed, 17, pts))
    assert np.array_equal(disp(still, 0, pts), disp(still, 17, pts))
    assert np.array_equal(disp(move, 0, pts), disp(move, 1, pts))                  # hold_frames default 2: frame 0 dan 1 satu gambar
    assert not np.array_equal(disp(move, 0, pts), disp(move, 2, pts))


def test_change_between_images_grows_with_drift():
    pts = np.random.default_rng(4).uniform(100, 900, (800, 2))

    def change(drift):
        g = geo(**{"jitter.temporal_drift": drift})
        return float(np.mean([np.sqrt(((disp(g, k + 1, pts) - disp(g, k, pts)) ** 2).mean()) for k in range(8)]))
    c = [change(d) for d in (0.15, 0.35, 0.7, 1.0)]
    assert c[0] < c[1] < c[2] < c[3], c


def test_frame_result_depends_only_on_absolute_frame_index_not_on_call_history():
    doc = junction_doc(index=9)
    g = geo()
    a, _ = sty.frame_pieces(doc, g)
    for k in (0, 3, 5):
        sty.frame_pieces(junction_doc(index=k), g)                                   # tanpa rantai: panggilan lain tidak memengaruhi
    b, _ = sty.frame_pieces(doc, g)
    assert all(np.array_equal(x.points, y.points) for x, y in zip(a, b))


def test_moving_stroke_gets_different_displacement_at_different_positions():
    g = geo()
    a = disp(g, 3, np.array([[300.0, 400.0]]))
    b = disp(g, 3, np.array([[316.0, 400.0]]))
    assert not np.allclose(a, b)                                                     # terkunci posisi (merayap saat bergerak)


# ── Stream jitter vs stream tebal ──────────────────
def test_jitter_params_do_not_change_widths_and_width_params_do_not_change_displacement():
    doc = doc_for([stroke(lattice([(15, 80), (40, 20), (70, 70), (88, 30)]), closed=False, typ="group_boundary", tid=1)], 3)
    gw = geo(**{"stroke.width_variation": 0.45})
    w0 = sty.frame_pieces(doc, gw)[0][0].widths
    for ov in ({"jitter.amplitude": 8.0}, {"jitter.frequency": 0.02}, {"jitter.hold_frames": 3}, {"jitter.temporal_drift": 0.9},
               {"jitter.stroke_independence": 0.5}, {"jitter.temporal_seed_mode": "fixed"}):
        w1 = sty.frame_pieces(doc, geo(**{"stroke.width_variation": 0.45, **ov}))[0][0].widths
        assert np.array_equal(w0, w1), ov
    pts = np.random.default_rng(5).uniform(100, 900, (100, 2))
    d0 = disp(geo(), 3, pts)
    for ov in ({"stroke.width_variation": 0.0}, {"stroke.width_noise_scale": 0.09}):
        assert np.array_equal(d0, disp(geo(**ov), 3, pts)), ov


# ── Tepi frame ─────────────────────────────────────
def edge_doc(index=2):
    """Strok yang keluar dari bawah (tegak lurus + miring), kiri, kanan dan sudut kanvas."""
    s1 = stroke([(30.5, 40.5), (30.5, 99.5)], closed=False, typ="group_boundary", tid=1)
    s2 = stroke([(50.5, 40.5), (62.5, 99.5)], closed=False, typ="group_boundary", tid=2)
    s3 = stroke([(40.5, 50.5), (0.5, 55.5)], closed=False, typ="group_boundary", tid=3)
    s4 = stroke([(60.5, 50.5), (99.5, 45.5)], closed=False, typ="group_boundary", tid=4)
    s5 = stroke([(60.5, 70.5), (99.5, 99.5)], closed=False, typ="group_boundary", tid=5)
    s6 = stroke([(40.5, 70.5), (0.5, 99.5)], closed=False, typ="group_boundary", tid=6)
    return doc_for([s1, s2, s3, s4, s5, s6], index)


@pytest.mark.parametrize("amp", [2.0, 4.0, 6.0, 8.0])
def test_edge_ink_unchanged_and_extension_ends_not_pulled_in(amp):
    ov = {"jitter.amplitude": amp, "jitter.frequency": 0.026}
    for index in (0, 2, 5):
        g, base, jit = base_and_jit(edge_doc(index), **ov)
        assert sum(pc.edge[0] or pc.edge[1] for pc in base) >= 6
        assert jm.edge_ink_changed(base, jit, g) == 0, (amp, index)
        assert jm.extension_intrusion(jit, g) <= jm.extension_intrusion(base, g) + 1e-9
        assert jm.max_displacement(base, jit) > 0.5


def test_edge_dead_zone_is_computed_from_style_not_a_constant():
    g = geo()
    assert math.isclose(g.edge_dead, 9.0 * 1.5 / 2 + sty.EDGE_MARGIN_PX) and math.isclose(g.edge_dead, 7.75)
    g2 = geo(**{"stroke.width_base": 14.0, "stroke.width_variation": 0.0})
    assert math.isclose(g2.edge_dead, 7.0 + sty.EDGE_MARGIN_PX)
    assert math.isclose(sty.EDGE_FADE_PX, 40.0) and math.isclose(g.edge_fade_len, 40.0)


def test_edge_fade_profile():
    g = geo()
    p = np.array([[g.edge_dead - 1, 500.0], [g.edge_dead, 500.0], [g.edge_dead + g.edge_fade_len / 2, 500.0],
                  [g.edge_dead + g.edge_fade_len, 500.0], [-5.0, 500.0], [500.0, 500.0]])
    phi = sty.edge_fade(p, g)
    assert phi[0] == 0 and phi[1] == 0 and 0.45 < phi[2] < 0.55 and phi[3] == 1 and phi[4] == 0 and phi[5] == 1
    top = sty.edge_fade(np.array([[500.0, 1.0]]), g)
    assert top[0] == 1.0                                                              # tepi atas tidak dilunakkan (tidak dipotong [5])


# ── Amplitudo 0 = T-401, SVG–PNG, determinisme ─────
def render_bytes(doc, **ov):
    st = st_for(**ov)
    g = sty.make_geometry(st, W, H)
    svg, png, _ = sty.render_frame(doc, g, st)
    return svg, png


def test_amplitude_zero_is_byte_identical_to_t401_path_for_any_other_jitter_param():
    doc = edge_doc(3)
    doc["strokes"] += junction_doc(3)["strokes"]
    g = geo(**{"jitter.amplitude": 0.0})
    st = st_for(**{"jitter.amplitude": 0.0})
    base, _ = sty.base_pieces(doc, g)                                                 # jalur T-401 (tanpa jitter)
    ref = (sty.render_svg(base, g, st), sty.render_png(base, g))
    for ov in ({}, {"jitter.hold_frames": 3}, {"jitter.stroke_independence": 0.7}, {"jitter.frequency": 0.2},
               {"jitter.temporal_seed_mode": "fixed"}, {"jitter.temporal_drift": 0.9}, {"jitter.param_seed": 0}):
        assert render_bytes(doc, **{"jitter.amplitude": 0.0, **ov}) == ref, ov
    assert render_bytes(doc, **{"jitter.amplitude": 3.0, "jitter.frequency": 0.0}) == ref          # frequency 0 = jitter mati


def test_jitter_on_changes_output_and_is_deterministic():
    doc = junction_doc(4)
    a, b = render_bytes(doc), render_bytes(doc)
    assert a == b and a != render_bytes(doc, **{"jitter.amplitude": 0.0})


@pytest.mark.parametrize("ov", [{}, {"jitter.stroke_independence": 0.5}, {"jitter.amplitude": 8.0, "jitter.frequency": 0.026}])
def test_svg_png_equivalence_holds_with_jitter(ov):
    doc = junction_doc(6)
    doc["strokes"] += edge_doc(6)["strokes"]
    st = st_for(**ov)
    g = sty.make_geometry(st, W, H)
    pieces, _ = sty.frame_pieces(doc, g)
    cov = sm.coverage(sm.decode_png(sty.render_png(pieces, g)), g)
    rep = sm.svg_png_report(sm.parse_svg(sty.render_svg(pieces, g, st)), cov, g)
    assert rep["iou"] >= sm.SVG_PNG_IOU_MIN and rep["l1"] <= sm.SVG_PNG_L1_MAX, rep
    assert rep["big_diff_frac"] <= sm.SVG_PNG_BIG_DIFF_MAX and abs(rep["mass_ratio"] - 1) <= sm.INK_MASS_TOL, rep


def test_short_stroke_and_empty_frame_survive_jitter():
    g = geo()
    tiny = stroke([(50.5, 50.5), (50.8, 50.6)], closed=False, typ="occlusion", tid=3)
    for strokes in ([], [tiny]):
        pieces, stats = sty.frame_pieces(doc_for(strokes, 2), g)
        assert len(pieces) == len(strokes) and "jitter_s" in stats


# ── Penjaga lipatan ────────────────────────────────
def test_fold_r_formula_and_warning_threshold():
    assert math.isclose(sty.jitter_fold_r(4.0, 0.053, 0.0), 0.212)
    assert math.isclose(sty.jitter_fold_r(4.0, 0.053, 0.5), 0.212 * math.sqrt(2))              # √0,5 + √0,5
    assert sty.JITTER_FOLD_R_WARN == 0.19
    assert sty.fold_warning(st_for(**{"jitter.amplitude": 3.4})) is None            # r 0,18 (lulus Jacobian terukur)
    assert sty.fold_warning(st_for()) is not None                                   # r 0,21 (gagal Jacobian di 2–17 frame)
    w = sty.fold_warning(st_for(**{"jitter.amplitude": 6.0}))                       # r 0,32
    assert w is not None and "0.318" in w and "jitter.amplitude 6" in w and "melipat" in w
    assert sty.fold_warning(st_for(**{"jitter.amplitude": 0.0, "jitter.frequency": 5.0})) is None
    assert sty.fold_warning(st_for(**{"jitter.amplitude": 4.0, "jitter.frequency": 0.053, "jitter.stroke_independence": 0.5})) is not None   # r 0,30


def test_fold_warning_printed_once_per_run_recorded_in_manifest_and_does_not_fail(tmp_path, capsys):
    work = make_stage_work(tmp_path)
    run = sty.run_stylize(cfg_for(work), st_for(**{"jitter.amplitude": 8.0}), "t", log=quiet)
    err = capsys.readouterr().err
    assert run["processed"] == N_FRAMES and err.count("PERINGATAN: jitter r") == 1
    m = manifest(work)
    assert m["jitter"]["fold_warn"] is True and m["jitter"]["on"] is True and math.isclose(m["jitter"]["fold_r"], 0.424)
    sty.run_stylize(cfg_for(work), st_for(**{"jitter.amplitude": 8.0}), "t", log=quiet)             # run kedua juga sekali
    assert capsys.readouterr().err.count("PERINGATAN: jitter r") == 1
    work2 = make_stage_work(tmp_path / "b")
    sty.run_stylize(cfg_for(work2), st_for(**{"jitter.amplitude": 3.4}), "t", log=quiet)
    assert "PERINGATAN: jitter r" not in capsys.readouterr().err and manifest(work2)["jitter"]["fold_warn"] is False


# ── Config ─────────────────────────────────────────
@pytest.mark.parametrize("key,bad", [("jitter.hold_frames", 0), ("jitter.hold_frames", 1.5), ("jitter.hold_frames", True),
                                     ("jitter.stroke_independence", -0.1), ("jitter.stroke_independence", 1.2),
                                     ("jitter.temporal_seed_mode", "x")])
def test_new_jitter_params_are_validated(key, bad):
    with pytest.raises(ConfigError):
        load_style(None, overrides={key: bad})


def test_new_jitter_params_defaults_and_accepted_values():
    s = load_style(None)
    assert s.jitter.hold_frames == 2 and s.jitter.stroke_independence == 0.0 and s.jitter.amplitude == 0.0   # default final: jitter MATI
    s = load_style(None, overrides={"jitter.hold_frames": 3, "jitter.stroke_independence": 0.25})
    assert s.jitter.hold_frames == 3 and s.jitter.stroke_independence == 0.25


# ── Manifest, basi, resume, jendela ────────────────
JIT_YAML = "jitter:\n  amplitude: 4.0\n  frequency: 0.053\n  hold_frames: 2\n"


@pytest.fixture
def jit_style(tmp_path, style_file):
    p = tmp_path / "jit-style.yaml"
    p.write_text(style_file.read_text(encoding="utf-8").replace("jitter:\n  amplitude: 0.0\n", JIT_YAML), encoding="utf-8")
    return p


def test_run_with_jitter_deterministic_resume_limit_from_identical(tmp_path, jit_style, capsys):
    work = make_stage_work(tmp_path, n=6)
    assert run_main(work, jit_style) == 0
    full = out_hashes(work)
    assert len(full) == 12 and manifest(work)["contract"] == "T-403" and manifest(work)["style_params"]["jitter.hold_frames"] == 2
    assert run_main(work, jit_style, "--restart") == 0 and out_hashes(work) == full                  # dua run dari nol identik
    assert run_main(work, jit_style, "--restart", "--limit", "3") == 0
    assert out_hashes(work) == {k: v for k, v in full.items() if int(k[6:11]) < 3}                      # --limit = awalan run penuh
    assert run_main(work, jit_style) == 0 and out_hashes(work) == full                                 # lalu penuh = identik
    for p in list(work.glob("strokes/frame_0000[34].*")):
        p.unlink()
    assert run_main(work, jit_style, "--from", "3", "--limit", "2") == 0                               # jendela dengan indeks MUTLAK
    assert out_hashes(work) == full


def test_window_from_equals_full_run_for_each_frame(tmp_path, jit_style):
    full_work = make_stage_work(tmp_path / "f", n=6)
    assert run_main(full_work, jit_style) == 0
    win_work = make_stage_work(tmp_path / "w", n=6)
    assert run_main(win_work, jit_style, "--from", "3", "--limit", "2") == 0
    win = out_hashes(win_work)
    full = out_hashes(full_work)
    assert set(win) == {"frame_00003.png", "frame_00003.svg", "frame_00004.png", "frame_00004.svg"}
    assert all(win[k] == full[k] for k in win)


@pytest.mark.parametrize("ov", [{"jitter.hold_frames": 3}, {"jitter.stroke_independence": 0.3}, {"jitter.amplitude": 3.0},
                                {"jitter.frequency": 0.04}, {"jitter.temporal_drift": 0.5}, {"jitter.temporal_seed_mode": "fixed"}])
def test_new_active_params_make_strokes_stale(tmp_path, ov):
    work = make_stage_work(tmp_path)
    sty.run_stylize(cfg_for(work), st_for(), "t", log=quiet)
    logs: list[str] = []
    run = sty.run_stylize(cfg_for(work), st_for(**ov), "t", log=logs.append)
    assert run["processed"] == N_FRAMES and any("style_params.jitter" in x for x in logs), ov


# ── Data nyata ─────────────────────────────────────
REAL_PINNED = {                       # sha256 strokes/frame_00080.svg|png klip asli T-401 (work/t402/before_strokes_*.sha256)
    "test_short": {"frame_00080.svg": "8bad6c0696cfcd854fbe20b8d9a1d6fbbd4f96b7d952bcb3a2e25633a6e6b5aa",
                   "frame_00080.png": "8c6d7437e52fd29f67b4cad877b509d675ee18f1e8abd6a943a829678740251f"},
    "test": {"frame_00080.svg": "50bb877a1bb1a7c0f8528e1017b84963895db78393d42a3e280bdefd4f430e38",
             "frame_00080.png": "a5a0fa944ef2bb7dd72afb145bc6dc0870c4ace834fd416970417429463033d5"},
}


def real_doc(clip: str, index: int) -> dict:
    d = CLIPS / clip / "contours"
    man = d / "manifest.json"
    if not man.is_file() or json.loads(man.read_text(encoding="utf-8")).get("contract") != "T-202":
        pytest.skip(f"contours T-202 klip {clip} tidak ada")
    return json.loads((d / f"frame_{index:05d}.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("clip", ["test_short", "test"])
def test_real_amplitude_zero_equals_t401_pinned_hashes(clip):
    pinned = REAL_PINNED[clip]
    if not all(pinned.values()):
        pytest.skip("hash T-401 belum disematkan")
    doc = real_doc(clip, 80)
    st = load_style(resolve_style("rough-sketch"), overrides={"jitter.amplitude": 0.0, "jitter.hold_frames": 2,
                                                      "multipass.passes": 1, "stroke.opacity": 1.0})
    g = sty.make_geometry(st, 480, 854)
    svg, png, _ = sty.render_frame(doc, g, st)
    assert hashlib.sha256(svg).hexdigest() == pinned["frame_00080.svg"]
    assert hashlib.sha256(png).hexdigest() == pinned["frame_00080.png"]


@pytest.mark.parametrize("clip", ["test_short", "test"])
def test_real_safety_invariants_with_coherent_field(clip):
    doc = real_doc(clip, 80)
    st = load_style(None, overrides={"jitter.amplitude": 4.0, "jitter.frequency": 0.053, "multipass.passes": 1, "stroke.opacity": 1.0})
    g = sty.make_geometry(st, 480, 854)
    base, _ = sty.base_pieces(doc, g)
    jit = sty.jitter_pieces(base, g, 80)
    assert jm.max_displacement(base, jit) <= jm.displacement_bound(g) + 0.02
    assert float(jm.joint_changes(base, jit, g).max()) <= jm.joint_tolerance(g)
    assert jm.new_crossings(base, jit) == 0
    assert jm.pieces_jacobian_min_det(jit, g, 80) > sty.JACOBIAN_MIN_DET
    assert jm.edge_ink_changed(base, jit, g) == 0
    assert jm.extension_intrusion(jit, g) <= jm.extension_intrusion(base, g) + 1e-9
    pieces, _ = sty.frame_pieces(doc, g)
    cov = sm.coverage(sm.decode_png(sty.render_png(pieces, g)), g)
    rep = sm.svg_png_report(sm.parse_svg(sty.render_svg(pieces, g, st)), cov, g)
    assert rep["iou"] >= sm.SVG_PNG_IOU_MIN and rep["l1"] <= sm.SVG_PNG_L1_MAX, rep


def test_stylize_with_jitter_does_not_import_torch():
    import subprocess
    import sys
    code = ("import sys; from rotoscope import stylize, noise; import numpy as np; "
            "noise.value_noise_3d(noise.seed_of(0), np.zeros(3), np.zeros(3), 0.5); "
            "assert 'torch' not in sys.modules, 'torch di-import'")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
