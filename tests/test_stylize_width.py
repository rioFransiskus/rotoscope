"""Test tebal variabel + taper + resample (T-401): profil tebal, noise terkunci posisi, taper hanya di ujung bebas (bertemu / tepi /
loop / tertutup / oklusi), lantai tebal, tikungan rapat, cincin, seam, resample, SVG poligon kontur, kesetaraan SVG ↔ PNG dan
regresi tebal konstan (data nyata dilewati bila klip / contours tidak ada). Sintetis di 1080 px (satu kerja px = 10,8 px output)."""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pytest
import stylize_metrics as sm
from test_stylize import W, H, circle, doc_for, lattice, stroke

from rotoscope import stylize as sty
from rotoscope.config import load_style

BASE = 9.0
S = 10.8                              # px output per px kerja (1080 / 100)


def mk(**ov):
    """Style 1080: tebal konstan, taper hidup (taper_min 0,2), resample minimum 4 — override per test."""
    base = {"render.output_width": 1080, "stroke.width_base": BASE, "stroke.width_variation": 0.0, "stroke.taper_ends": True,
            "stroke.taper_px": 45.0, "stroke.taper_min": 0.2, "shape.resample_points": 4, "jitter.amplitude": 0.0,
            "multipass.passes": 1, "stroke.opacity": 1.0}          # garis tunggal solid (default T-403 = 2 pass, opacity 0,92)
    return load_style(None, overrides={**base, **ov})


def pcs(strokes, st=None, index=0):
    st = st or mk()
    g = sty.make_geometry(st, W, H)
    pieces, stats = sty.frame_pieces(doc_for(strokes, index), g)
    return g, pieces


def line(x0, y0, x1, y1, typ="group_boundary", tid=1):
    return stroke([(x0, y0), (x1, y1)], closed=False, typ=typ, tid=tid)


def w_at(pc, g, s_px):
    """Tebal pada busur s (px output) dari titik awal jalur."""
    return float(np.interp(s_px, sm.piece_arclength(pc), sty.piece_widths(pc, g)))


# ── Profil tebal ───────────────────────────────────
def test_constant_when_variation_zero_and_taper_off():
    g, ps = pcs([line(20.5, 50.5, 80.5, 50.5)], mk(**{"stroke.taper_ends": False}))
    assert np.allclose(sty.piece_widths(ps[0], g), BASE)


def test_width_bounded_by_variation_mean_ratio_near_one():
    pts = lattice([(10, 85), (25, 20), (45, 80), (65, 15), (90, 70)])
    g, ps = pcs([stroke(pts, closed=False, typ="group_boundary")], mk(**{"stroke.width_variation": 0.45, "stroke.taper_ends": False}))
    w = sty.piece_widths(ps[0], g)
    assert float(w.min()) >= BASE * 0.55 - 1e-6 and float(w.max()) <= BASE * 1.45 + 1e-6
    assert 0.9 <= float(w.mean()) / BASE <= 1.1 and 0.08 <= float(w.std()) / BASE <= 0.3          # amplitudo terpakai, rata-rata ≈ dasar


def test_noise_is_position_locked_not_arclength_and_independent_of_track_frame_seed_time():
    """Tebal fungsi POSISI: jalur yang sama dibalik / track_id lain / frame_index lain → tebal di titik yang sama identik."""
    st = mk(**{"stroke.width_variation": 0.45, "stroke.taper_ends": False})
    g = sty.make_geometry(st, W, H)
    pts = np.array(lattice([(10, 85), (25, 20), (45, 80), (65, 15), (90, 70)]), float) * S
    fwd = sty.width_profile(pts, False, "group_boundary", (False, False), g)
    rev = sty.width_profile(pts[::-1], False, "group_boundary", (False, False), g)[::-1]
    assert np.array_equal(fwd, rev)
    shifted = sty.width_profile(pts[40:], False, "group_boundary", (False, False), g)             # titik awal lain
    assert np.array_equal(fwd[40:], shifted)
    a = sty.frame_pieces(doc_for([stroke(lattice([(10, 85), (50, 20), (90, 70)]), closed=False, typ="group_boundary", tid=1)], 0), g)[0]
    b = sty.frame_pieces(doc_for([stroke(lattice([(10, 85), (50, 20), (90, 70)]), closed=False, typ="group_boundary", tid=99)], 57), g)[0]
    assert np.array_equal(sty.piece_widths(a[0], g), sty.piece_widths(b[0], g))               # track_id / frame_index tidak masuk seed


def test_param_seed_changes_widths_only_when_noise_is_on():
    pts = np.array(lattice([(10, 85), (25, 20), (45, 80), (65, 15), (90, 70)]), float) * S
    on = [sty.make_geometry(mk(**{"stroke.width_variation": 0.45, "jitter.param_seed": s}), W, H) for s in (0, 1)]
    wa, wb = (sty.width_profile(pts, False, "silhouette", (False, False), g) for g in on)
    assert float(np.abs(wa - wb).mean()) > 0.5
    off = [sty.make_geometry(mk(**{"stroke.width_variation": 0.0, "jitter.param_seed": s}), W, H) for s in (0, 1)]
    assert np.array_equal(*(sty.width_profile(pts, False, "silhouette", (False, False), g) for g in off))


def test_noise_scale_zero_disables_noise_and_scale_sets_correlation_length():
    g0 = sty.make_geometry(mk(**{"stroke.width_variation": 0.45, "stroke.width_noise_scale": 0.0}), W, H)
    pts = np.array(lattice([(10, 50), (90, 50)]), float) * S
    assert np.allclose(sty.width_profile(pts, False, "silhouette", (False, False), g0), BASE)
    g1 = sty.make_geometry(mk(**{"stroke.width_variation": 0.45, "stroke.width_noise_scale": 0.036}), W, H)
    g2 = sty.make_geometry(mk(**{"stroke.width_variation": 0.45, "stroke.width_noise_scale": 0.072}), W, H)
    assert g1.noise_cell == pytest.approx(1 / 0.036) and g2.noise_cell == pytest.approx(1 / 0.072)
    ys = np.column_stack([np.linspace(50, 950, 4000), np.full(4000, 500.0)])
    zc = [int((np.diff(np.sign(sty.width_profile(ys, False, "silhouette", (False, False), g) - BASE)) != 0).sum()) for g in (g1, g2)]
    assert zc[1] > 1.5 * zc[0]                                                  # skala ×2 → ±2× lebih banyak osilasi


def test_type_hierarchy_scales_widths():
    st = mk(**{"stroke.by_type.occlusion.width_scale": 0.5, "stroke.by_type.group_boundary.width_scale": 0.6,
               "stroke.by_type.silhouette_hole.width_scale": 0.8, "stroke.taper_ends": False})
    strokes = [line(20.5, 20.5, 80.5, 20.5, "silhouette", 1), line(20.5, 40.5, 80.5, 40.5, "silhouette_hole", 2),
               line(20.5, 60.5, 80.5, 60.5, "group_boundary", 3), line(20.5, 80.5, 80.5, 80.5, "occlusion", 4)]
    g, ps = pcs(strokes, st)
    got = {p.type: float(np.median(sty.piece_widths(p, g))) for p in ps}
    assert got == pytest.approx({"silhouette": 9.0, "silhouette_hole": 7.2, "group_boundary": 5.4, "occlusion": 4.5})


def test_floor_width_enforced_and_line_has_no_gaps():
    st = mk(**{"stroke.width_base": 0.2, "stroke.taper_ends": False})
    g, ps = pcs([line(20.5, 50.5, 80.5, 50.5), line(20.5, 70.5, 80.5, 85.5, tid=2)], st)
    assert all(float(sty.piece_widths(p, g).min()) == sty.WIDTH_FLOOR_PX for p in ps)
    cov = sm.coverage(sm.decode_png(sty.render_png(ps, g)), g)
    for p in ps:                                                                # tiap kolom tinta menerus (tidak bolong)
        x0, x1 = int(p.points[0, 0]) + 8, int(p.points[-1, 0]) - 8
        col = cov[:, x0:x1].sum(0)
        assert float(col.min()) >= 0.85


def test_minimum_width_is_at_least_floor_with_taper_zero_min():
    st = mk(**{"stroke.taper_min": 0.0, "stroke.width_variation": 0.5})
    g, ps = pcs([line(20.5, 50.5, 80.5, 50.5)], st)
    assert float(sty.piece_widths(ps[0], g).min()) >= sty.WIDTH_FLOOR_PX


# ── Taper ──────────────────────────────────────────
def test_taper_profile_at_free_ends_and_clamp_on_short_strokes():
    g, ps = pcs([line(20.5, 50.5, 80.5, 50.5)])
    pc = ps[0]
    L = sm.piece_arclength(pc)[-1]
    tp = 45.0                                                                   # px ref = px output
    assert w_at(pc, g, 0) == pytest.approx(0.2 * BASE, rel=1e-3) and w_at(pc, g, L) == pytest.approx(0.2 * BASE, rel=1e-3)
    assert w_at(pc, g, tp / 2) == pytest.approx((0.2 + 0.8 * 0.5) * BASE, rel=1e-2)       # smoothstep(0,5) = 0,5
    assert w_at(pc, g, tp) == pytest.approx(BASE, rel=1e-3) and w_at(pc, g, L / 2) == pytest.approx(BASE)
    # strok pendek: L = 43 px < 2 × taper_px → zona = L / 2, puncak tebal penuh hanya di tengah
    g, ps = pcs([line(45.5, 50.5, 49.5, 50.5)])
    pc = ps[0]
    L = sm.piece_arclength(pc)[-1]
    assert L < 2 * tp and w_at(pc, g, L / 2) == pytest.approx(BASE, rel=1e-3)
    assert w_at(pc, g, L / 4) == pytest.approx((0.2 + 0.8 * 0.5) * BASE, rel=1e-2)          # zona = L/2 bukan taper_px
    assert w_at(pc, g, 0) == pytest.approx(0.2 * BASE, rel=1e-3)


def test_taper_off_when_disabled_or_taper_px_zero():
    for ov in ({"stroke.taper_ends": False}, {"stroke.taper_px": 0.0}):
        g, ps = pcs([line(20.5, 50.5, 80.5, 50.5)], mk(**ov))
        assert np.allclose(sty.piece_widths(ps[0], g), BASE)


def test_end_meeting_another_stroke_is_not_tapered():
    """T: strok vertikal berujung di strok horizontal → ujung bertemu penuh; ujung lain menipis."""
    g, ps = pcs([line(20.5, 50.5, 80.5, 50.5, tid=1), line(50.5, 20.5, 50.5, 50.5, tid=2)])
    b = ps[1]
    assert w_at(b, g, 0) == pytest.approx(0.2 * BASE, rel=1e-3)                 # ujung atas bebas
    assert w_at(b, g, sm.piece_arclength(b)[-1]) == pytest.approx(BASE)         # ujung bawah bertemu
    a = ps[0]
    assert w_at(a, g, 0) == pytest.approx(0.2 * BASE, rel=1e-3)                 # ujung strok horizontal sendiri bebas


@pytest.mark.parametrize("gap_px, tapered", [(6.0, False), (7.9, False), (9.0, True)])
def test_join_distance_threshold_for_non_occlusion_ends(gap_px, tapered):
    """JOIN_DIST_PX = 8 px: ujung ≤ 8 px dari strok lain bertemu (px output; unit = 1)."""
    y_end = 50.5 - gap_px / S
    g, ps = pcs([line(20.5, 50.5, 80.5, 50.5, tid=1), line(50.5, 20.5, 50.5, y_end, tid=2)])
    end = w_at(ps[1], g, sm.piece_arclength(ps[1])[-1])
    assert (end < BASE - 0.5) == tapered


def test_end_at_frame_edge_is_not_tapered_other_end_is():
    g, ps = pcs([line(50.5, 60.5, 50.5, 99.5)])                                 # ujung bawah = tepi frame (hide: diperpanjang)
    pc = ps[0]
    assert pc.edge == (False, True)
    assert w_at(pc, g, 0) == pytest.approx(0.2 * BASE, rel=1e-3)
    L = sm.piece_arclength(pc)[-1]
    assert w_at(pc, g, L) == pytest.approx(BASE) and w_at(pc, g, L - 20) == pytest.approx(BASE)
    assert pc.points[-1, 1] > g.out_h                                           # tetap diperpanjang keluar kanvas


def test_edge_end_in_draw_mode_also_not_tapered():
    g, ps = pcs([line(50.5, 60.5, 50.5, 99.5)], mk(**{"shape.edge_mode": "draw", "stroke.width_base": 12.0}))
    pc = ps[0]
    assert any(pc.edge) and w_at(pc, g, sm.piece_arclength(pc)[-1]) == pytest.approx(12.0)


def test_loop_and_closed_have_no_taper_and_no_notch():
    ring = circle(50.5, 50.5, 25)
    for s in (stroke(ring, closed=True, typ="silhouette_hole"), stroke(ring + [ring[0]], closed=False, typ="group_boundary")):
        g, ps = pcs([s], mk(**{"stroke.width_variation": 0.0}))
        assert len(ps) == 1 and ps[0].closed
        assert np.allclose(sty.piece_widths(ps[0], g), BASE)                    # tidak ada takik di sambungan / ujung


def test_seam_of_closed_stroke_is_continuous_with_noise():
    st = mk(**{"stroke.width_variation": 0.5})
    g, ps = pcs([stroke(circle(50.5, 50.5, 25), closed=True, typ="silhouette_hole")], st)
    jump, step = sm.seam_jump(ps[0], g)
    assert jump <= step + 1e-9 and step < 0.15 * BASE                           # sambungan = langkah biasa antar titik (B terkunci posisi)


def occ(x0, y0, x1, y1, tid):
    return stroke([(x0, y0), (x1, y1)], closed=False, typ="occlusion", tid=tid)


def test_occlusion_free_end_is_tapered_and_joined_occlusions_are_not():
    g, ps = pcs([occ(20.5, 50.5, 50.5, 50.5, 1), occ(50.5 + 1.5 / S, 50.5, 80.5, 50.5, 2)])      # celah 1,5 px ≤ 2 → bersambung
    a, b = ps
    assert w_at(a, g, 0) < BASE - 0.5 and w_at(b, g, sm.piece_arclength(b)[-1]) < BASE - 0.5       # ujung luar bebas
    assert w_at(a, g, sm.piece_arclength(a)[-1]) == pytest.approx(BASE)         # sambungan tanpa taper (tanpa takik)
    assert w_at(b, g, 0) == pytest.approx(BASE)


def test_two_occlusions_with_gap_over_two_px_both_taper():
    g, ps = pcs([occ(20.5, 50.5, 50.5, 50.5, 1), occ(50.5 + 3.0 / S, 50.5, 80.5, 50.5, 2)])         # celah 3 px > 2
    assert w_at(ps[0], g, sm.piece_arclength(ps[0])[-1]) < BASE - 0.5 and w_at(ps[1], g, 0) < BASE - 0.5


@pytest.mark.parametrize("other_type, gap_px, tapered", [("group_boundary", 6.0, False), ("silhouette_hole", 7.0, False),
                                                           ("group_boundary", 9.0, True)])
def test_occlusion_end_near_non_occlusion_follows_join_dist_px(other_type, gap_px, tapered):
    other = line(20.5, 50.5, 80.5, 50.5, other_type, 1)
    g, ps = pcs([other, occ(50.5, 20.5, 50.5, 50.5 - gap_px / S, 2)])
    end = w_at(ps[1], g, sm.piece_arclength(ps[1])[-1])
    assert (end < BASE - 0.5) == tapered


def test_gb_end_near_occlusion_uses_join_dist_px():
    g, ps = pcs([occ(20.5, 50.5, 80.5, 50.5, 1), line(50.5, 20.5, 50.5, 50.5 - 6.0 / S, tid=2)])
    assert w_at(ps[1], g, sm.piece_arclength(ps[1])[-1]) == pytest.approx(BASE)


def test_per_type_taper_override_and_inheritance():
    strokes = [line(20.5, 30.5, 80.5, 30.5, tid=1), occ(20.5, 60.5, 80.5, 60.5, 2)]
    for ov, gb_taper, oc_taper in (({}, True, True), ({"stroke.by_type.occlusion.taper_ends": False}, True, False),
                                   ({"stroke.taper_ends": False}, False, False),
                                   ({"stroke.taper_ends": False, "stroke.by_type.occlusion.taper_ends": True}, False, True),
                                   ({"stroke.by_type.group_boundary.taper_ends": False}, False, True)):
        g, ps = pcs(strokes, mk(**ov))
        assert (w_at(ps[0], g, 0) < BASE - 0.5) == gb_taper and (w_at(ps[1], g, 0) < BASE - 0.5) == oc_taper, ov


def test_taper_profile_is_smoothstep():
    g, ps = pcs([line(20.5, 50.5, 80.5, 50.5)])
    t = 0.25                                                                    # s = taper_px / 4
    assert w_at(ps[0], g, 45 / 4) == pytest.approx((0.2 + 0.8 * t * t * (3 - 2 * t)) * BASE, rel=1e-2)
    assert w_at(ps[0], g, 0) == pytest.approx(0.2 * BASE, rel=1e-3) and not hasattr(sty, "TAPER_PROFILE")


# ── Resample ───────────────────────────────────────
def test_resample_gap_lower_bound_and_ends():
    g, ps = pcs([line(20.5, 50.5, 80.5, 50.5)], mk(**{"shape.resample_points": 4}))
    p = ps[0].points
    gaps = np.hypot(*np.diff(p, axis=0).T)
    assert float(gaps.max()) <= sty.RESAMPLE_MAX_GAP_REF * g.unit + 0.02
    g2, ps2 = pcs([line(20.5, 50.5, 80.5, 50.5)], mk(**{"shape.resample_points": 500}))
    assert len(ps2[0].points) == 500 and len(p) < 500                           # resample_points = batas bawah N
    assert np.allclose(p[0], np.array([20.5, 50.5]) * S, atol=0.02) and np.allclose(p[-1], np.array([80.5, 50.5]) * S, atol=0.02)


def test_resample_closed_has_no_duplicate_point_and_even_spacing():
    g, ps = pcs([stroke(circle(50.5, 50.5, 25), closed=True, typ="silhouette_hole")])
    p = ps[0].points
    gaps = np.hypot(*np.diff(np.vstack([p, p[:1]]), axis=0).T)
    assert not np.array_equal(p[0], p[-1]) and float(gaps.max()) <= 2.0 + 0.05 and float(gaps.min()) > 0.5 * float(gaps.max())


def test_resample_keeps_centerline_fidelity():
    s = stroke(lattice([(12, 85), (30, 45), (55, 30), (75, 45), (90, 85)]), closed=False, typ="group_boundary")
    st = mk()
    g = sty.make_geometry(st, W, H)
    raw = sty.stroke_pieces(s, g, sty.new_stats(), 0)[0]
    res = sty.frame_pieces(doc_for([s]), g)[0][0]
    d = sm.point_polyline_distance(res.points, raw.points)
    assert float(d.max()) <= 0.1                                                # titik resample ≤ 0,1 px dari spline


# ── Renderer: tikungan rapat, cincin, SVG ──────────
def u_turn(rho: float, y0=120.0, x=300.0, n=60):
    th = np.linspace(0, np.pi, n)
    arc = np.column_stack([x + rho * np.cos(th), y0 - rho * np.sin(th)])
    leg1 = np.column_stack([np.full(60, x + rho), np.linspace(y0 + 100, y0, 60, endpoint=False)])
    leg2 = np.column_stack([np.full(59, x - rho), np.linspace(y0, y0 + 100, 60)[1:]])
    return sty.arc_resample(np.vstack([leg1, arc, leg2]), False, 1.0)


@pytest.mark.parametrize("rho", [1.5, 3.0, 4.5, 8.0])
def test_tight_bend_union_raster_and_nonzero_svg_agree_with_ideal(rho):
    """Jari-jari < tebal/2 (lipatan offset): raster union dan SVG nonzero sama dengan tinta ideal (ss tinggi); tidak bolong."""
    st = mk()
    g = sty.make_geometry(st, W, H)
    pc = sty.Piece("silhouette", False, np.round(u_turn(rho), 2), 1, 0, (False, False), np.full(len(u_turn(rho)), BASE))
    ideal_g = dataclasses.replace(g, ss=12)
    ideal_cov = np.clip(sm.cv2.resize(sty.render_mask([pc], ideal_g), (g.out_w, g.out_h), interpolation=sm.cv2.INTER_AREA).astype(float) / 255, 0, 1)
    ras = sm.cv2.resize(sty.render_mask([pc], g), (g.out_w, g.out_h), interpolation=sm.cv2.INTER_AREA).astype(float) / 255
    svg = sm.rasterize_svg([sm.SvgPath("silhouette", sty.piece_outline(pc, g))], g)
    for cov in (ras, svg):
        assert sm.cov_report(cov, ideal_cov)["l1"] <= 0.06
        assert sm.centerline_gaps(cov, [pc], g, step=0.5, min_cov=0.9) == 0     # tidak ada celah di garis tengah (tikungan)


def test_closed_ring_is_hollow_in_raster_and_svg():
    g = sty.make_geometry(mk(), W, H)
    th = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    pts = np.round(np.column_stack([540 + 200 * np.cos(th), 540 + 200 * np.sin(th)]), 2)
    pc = sty.Piece("silhouette_hole", True, pts, 1, 0, (False, False), np.full(len(pts), BASE))
    cov = sm.cv2.resize(sty.render_mask([pc], g), (g.out_w, g.out_h), interpolation=sm.cv2.INTER_AREA).astype(float) / 255
    polys = sty.piece_outline(pc, g)
    assert len(polys) == 2                                                      # luar + dalam
    svg = sm.rasterize_svg([sm.SvgPath("silhouette_hole", polys)], g)
    for c in (cov, svg):
        assert float(c[540, 540]) == 0.0 and float(c[540, 540 + 200]) > 0.95 and float(c[540, 540 + 200 + 7]) < 0.3
    assert sm.cov_report(svg, cov)["iou"] >= 0.98


def test_svg_png_metric_detects_polygon_shifted_one_pixel():
    """Mutasi 'SVG memakai titik berbeda dari raster': geser 1 px pada garis 9 px → metrik kesetaraan GAGAL (IoU turun jauh)."""
    pts = lattice([(12, 85), (30, 45), (55, 30), (75, 45), (90, 85)])
    st = mk(**{"stroke.width_variation": 0.3})
    g = sty.make_geometry(st, W, H)
    pieces, _ = sty.frame_pieces(doc_for([stroke(pts, closed=False, typ="group_boundary")]), g)
    cov = sm.cv2.resize(sty.render_mask(pieces, g), (g.out_w, g.out_h), interpolation=sm.cv2.INTER_AREA).astype(float) / 255
    good = sm.cov_report(sm.rasterize_svg([sm.SvgPath(p.type, sty.piece_outline(p, g)) for p in pieces], g), cov)
    shifted = sm.cov_report(sm.rasterize_svg([sm.SvgPath(p.type, [q + 1.0 for q in sty.piece_outline(p, g)]) for p in pieces], g), cov)
    assert good["iou"] >= sm.SYNTH_SVG_PNG_IOU_MIN and shifted["iou"] < sm.SYNTH_SVG_PNG_IOU_MIN - 0.05
    assert shifted["l1"] > 3 * good["l1"]


def test_svg_is_filled_nonzero_polygon_one_path_per_piece():
    g, ps = pcs([stroke(circle(50.5, 50.5, 25), closed=True, typ="silhouette_hole", tid=1),
                 line(20.5, 20.5, 80.5, 20.5, tid=2)])
    st = mk()
    svg = sty.render_svg(ps, g, st).decode()
    assert svg.count("<path ") == 2 and 'fill-rule="nonzero"' in svg and " L" not in svg
    parsed = sm.parse_svg(svg.encode())
    assert [len(p.polys) for p in parsed] == [1, 2] or [len(p.polys) for p in parsed] == [2, 1]       # terbuka 1 sub-path, tertutup 2


def test_variable_width_render_is_deterministic_and_frame_index_free():
    st = mk(**{"stroke.width_variation": 0.45})
    strokes = [stroke(circle(50.5, 50.5, 25), closed=True, typ="silhouette_hole", tid=3), line(20.5, 20.5, 80.5, 20.5, tid=2)]
    g = sty.make_geometry(st, W, H)
    a = sty.render_frame(doc_for(strokes, 0), g, st)[:2]
    b = sty.render_frame(doc_for(strokes, 0), g, st)[:2]
    c = sty.render_frame(doc_for(strokes, 123), g, st)[:2]
    assert a == b and a == c                                                    # tanpa waktu: frame_index tidak berpengaruh


def test_render_mask_variable_equals_constant_when_widths_equal():
    """Renderer: widths per titik konstan = jalur tanpa widths (T-203a)."""
    g = sty.make_geometry(mk(), W, H)
    pts = np.round(sty.arc_resample(np.array([[100.0, 100.0], [400, 300], [700, 150.0]]), False, 2.0), 2)
    a = sty.Piece("group_boundary", False, pts, 1)
    b = dataclasses.replace(a, widths=np.full(len(pts), g.widths["group_boundary"]))
    assert np.array_equal(sty.render_mask([a], g), sty.render_mask([b], g))


# ── Manifest / basi / hash ─────────────────────────
def test_new_active_params_change_style_hash():
    base = load_style(None)
    h0 = sty.params_hash(sty.style_params(base)[0])
    for ov in ({"stroke.width_variation": 0.3}, {"stroke.width_noise_scale": 0.05}, {"stroke.taper_px": 30.0}, {"stroke.taper_min": 0.3},
               {"stroke.taper_ends": False}, {"shape.resample_points": 16}, {"jitter.param_seed": 3},
               {"stroke.by_type.occlusion.taper_ends": False}):
        assert sty.params_hash(sty.style_params(load_style(None, overrides=ov))[0]) != h0, ov
    for ov in ({"jitter.amplitude": 9.0}, {"jitter.frequency": 0.1}, {"jitter.temporal_seed_mode": "fixed"}, {"jitter.temporal_drift": 0.5},
               {"jitter.hold_frames": 3}, {"jitter.stroke_independence": 0.5}):
        assert sty.params_hash(sty.style_params(load_style(None, overrides=ov))[0]) != h0, ov          # aktif sejak T-402
    assert sty.params_hash(sty.style_params(load_style(None, overrides={"paper.vignette": 0.3}))[0]) == h0     # tak aktif
    for ov in ({"multipass.passes": 5}, {"multipass.offset": 5.0}, {"multipass.opacity_falloff": 0.3}, {"multipass.temporal_mode": "frame"},
               {"multipass.enabled": False}, {"stroke.opacity": 0.9}, {"stroke.by_type.occlusion.opacity_scale": 0.5}):
        assert sty.params_hash(sty.style_params(load_style(None, overrides=ov))[0]) != h0, ov          # aktif sejak T-403


def test_old_contract_strokes_are_stale_and_recomputed(tmp_path):
    from test_stylize import cfg_for, make_stage_work, quiet, style_for
    work = make_stage_work(tmp_path)
    sty.run_stylize(cfg_for(work), style_for(), "t", log=quiet)
    p = work / "strokes" / "manifest.json"
    m = json.loads(p.read_text(encoding="utf-8"))
    m["contract"] = "T-203a"
    p.write_text(json.dumps(m), encoding="utf-8")
    logs: list[str] = []
    run = sty.run_stylize(cfg_for(work), style_for(), "t", log=logs.append)
    assert run["processed"] == 4 and any("contract" in x and "T-403" in x for x in logs)
    assert json.loads(p.read_text(encoding="utf-8"))["contract"] == "T-403"


def test_by_type_taper_ends_is_validated_as_optional_bool():
    from rotoscope.config import ConfigError
    assert load_style(None).stroke.by_type["occlusion"].taper_ends is None
    assert load_style(None, overrides={"stroke.by_type.occlusion.taper_ends": False}).stroke.by_type["occlusion"].taper_ends is False
    with pytest.raises(ConfigError, match="taper_ends"):
        load_style(None, overrides={"stroke.by_type.occlusion.taper_ends": "ya"})


# ── Data nyata (dilewati bila klip / contours T-202 tidak ada) ──
CLIPS = Path(__file__).resolve().parents[1] / "work" / "clips"
REAL_STEP = 40


def real_docs(clip: str, step: int = REAL_STEP, start: int = 0):
    d = CLIPS / clip / "contours"
    man = d / "manifest.json"
    if not man.is_file() or json.loads(man.read_text(encoding="utf-8")).get("contract") != "T-305b":
        pytest.skip(f"contours T-305b klip {clip} tidak ada")
    return [json.loads(f.read_text(encoding="utf-8")) for f in sorted(d.glob("frame_*.json"))[start::step]]


def _cov(mask, g):
    return sm.cv2.resize(mask, (g.out_w, g.out_h), interpolation=sm.cv2.INTER_AREA).astype(float) / 255.0


@pytest.mark.parametrize("clip", ["test_short", "test"])
def test_real_svg_png_equivalence_and_width_bounds(clip):
    docs = real_docs(clip, 60)
    st = load_style(None, overrides={"multipass.passes": 1, "stroke.opacity": 1.0})       # garis tunggal solid (default T-403 = 2 pass)
    g = sty.make_geometry(st, 480, 854)
    for d in docs:
        pieces, _ = sty.frame_pieces(d, g)
        svg = sty.render_svg(pieces, g, st)
        cov = sm.coverage(sm.decode_png(sty.render_png(pieces, g)), g)
        rep = sm.svg_png_report(sm.parse_svg(svg), cov, g)
        assert rep["iou"] >= sm.SVG_PNG_IOU_MIN and rep["l1"] <= sm.SVG_PNG_L1_MAX, (clip, d["frame_index"], rep)
        assert rep["big_diff_frac"] <= sm.SVG_PNG_BIG_DIFF_MAX and abs(rep["mass_ratio"] - 1) <= sm.INK_MASS_TOL, (clip, rep)
        for pc in pieces:
            w = sty.piece_widths(pc, g)
            assert float(w.min()) >= sty.WIDTH_FLOOR_PX and float(w.max()) <= g.widths[pc.type] * (1 + g.variation) + 1e-6


@pytest.mark.parametrize("clip", ["test_short", "test"])
def test_real_constant_width_regression_vs_t203a_and_fidelity(clip):
    """Tebal konstan (variasi 0, taper mati): renderer baru ≈ T-203a pada strok TANPA resample (IoU, L1, massa) dan deviasi centerline
    tidak bertambah (batas dari ukuran Tahap 1: ≤ +0,1 px)."""
    docs = real_docs(clip, 60)
    st = load_style(None, overrides={"stroke.width_variation": 0.0, "stroke.taper_ends": False, "jitter.amplitude": 0.0})
    g = sty.make_geometry(st, 480, 854)
    for d in docs:
        raw = []
        for i, s in enumerate(d["strokes"]):
            raw += sty.stroke_pieces(s, g, sty.new_stats(), i)
        old = _cov(sty.render_mask(raw, g), g)
        new_pieces, _ = sty.frame_pieces(d, g)
        rep = sm.cov_report(_cov(sty.render_mask(new_pieces, g), g), old)
        assert rep["iou"] >= sm.REGRESSION_IOU_MIN and rep["l1"] <= sm.SVG_PNG_L1_MAX and abs(rep["mass_ratio"] - 1) <= sm.INK_MASS_TOL, rep
        for s in d["strokes"]:
            dev_old = sm.stroke_deviation(s, g)
            if not dev_old.size:
                continue
            pcs_new = [p for p in new_pieces if p.track_id == s["track_id"] and p.type == s["type"]]
            ref = np.asarray(s["points"], float) * g.scale
            keep = [not x for x in sty.edge_sides(np.asarray(s["points"], float), g.width, g.height)]
            dn = np.min([sm.point_polyline_distance(ref[keep], sm.piece_curve(p)) for p in pcs_new], axis=0)
            assert float(dn.max()) <= float(dev_old.max()) + 0.1, (clip, d["frame_index"], s["type"])


@pytest.mark.parametrize("clip", ["test_short", "test"])
def test_real_end_rules_and_pop_bounds(clip):
    """Aturan ujung pada data nyata + 'pop tebal' B (taper mati): diam p95 ≤ 0,03, bergerak p95 ≤ 0,10 × width_base (ukur Tahap 1:
    0,019 / 0,065) untuk silhouette, lubang, batas grup; juga dilaporkan relatif terhadap tebal tipe sendiri."""
    docs = real_docs(clip, 1)[60:100]
    st = load_style(None, overrides={"stroke.taper_ends": False, "jitter.amplitude": 0.0})
    g = sty.make_geometry(st, 480, 854)
    base = st.stroke.width_base * g.unit
    acc = None
    prev = None
    for d in docs:
        cur, _ = sty.frame_pieces(d, g)
        if prev is not None:
            r = sm.pop_by_type(prev, cur, g, base)
            acc = r if acc is None else {t: {k: acc[t][k] + r[t][k] for k in acc[t]} for t in acc}
        prev = cur
    res = sm.summarize_pop(acc)
    for t in ("silhouette", "silhouette_hole", "group_boundary"):
        assert res[t]["base_still"]["p95"] <= 0.03 and res[t]["base_move"]["p95"] <= 0.10, (clip, t, res[t])
        assert res[t]["own_move"]["p95"] <= 0.10 + 1e-9
    # aturan ujung bebas dengan taper hidup (semua tipe)
    st2 = load_style(None, overrides={"jitter.amplitude": 0.0})
    g2 = sty.make_geometry(st2, 480, 854)
    for d in docs[::8]:
        pieces, _ = sty.frame_pieces(d, g2)
        flags = sty.free_ends(pieces, g2)
        for pc, fl in zip(pieces, flags):
            for e in (0, 1):
                if pc.closed or pc.edge[e]:
                    assert not fl[e]                                              # tertutup / tepi tidak pernah bebas
        others = [(pc, fl) for pc, fl in zip(pieces, flags)]
        for pc, fl in others:
            for e in (0, 1):
                if fl[e]:
                    end = pc.points[0 if e == 0 else -1]
                    for q in pieces:
                        if q.stroke_idx == pc.stroke_idx:
                            continue
                        dist = float(sm.point_polyline_distance(end[None, :], sm.piece_curve(q))[0])
                        thr = sty.JOIN_DIST_OCC_PX if (pc.type == "occlusion" and q.type == "occlusion") else sty.JOIN_DIST_PX
                        assert dist > thr - 0.3, (clip, d["frame_index"], pc.type, q.type, dist)    # ujung bebas tidak dekat strok lain


@pytest.mark.parametrize("clip", ["test_short", "test"])
def test_real_frame_render_deterministic(clip):
    d = real_docs(clip, 120)[0]
    st = load_style(None)
    g = sty.make_geometry(st, 480, 854)
    a = sty.render_frame(d, g, st)[:2]
    b = sty.render_frame(d, g, st)[:2]
    assert a == b
