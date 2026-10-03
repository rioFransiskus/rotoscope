"""Test penghalusan Gaussian sebelum approxPolyDP (T-203a, keputusan Rio): garis miring / busur tidak menyusut, ujung tetap, sudut
tajam tetap, sambungan tertutup tanpa takik, derau tepi berkurang, determinisme, `smooth_px` 0 = mati. Sintetis non-sumbu-sejajar."""

from __future__ import annotations

import math

import numpy as np
import pytest
import stylize_metrics as sm
from test_stylize import (OW, W, H, circle, geom_for, lattice, outer_stroke, render, stroke, style_for, TRAPEZOID)

from rotoscope import stylize as sty

STEP = 1.0                     # px output (unit 1 pada 1080)
SIGMA = 3.0


def noisy_line(angle_deg: float, length: float = 400.0, noise: float = 0.7, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(0, length, 1.0)
    d = np.array([math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))])
    n = np.array([-d[1], d[0]])
    base = np.array([50.0, 50.0]) + t[:, None] * d
    return base + (rng.uniform(-noise, noise, len(t)) * 1.0)[:, None] * n, base, n


def second_diff_energy(q: np.ndarray) -> float:
    r = sty.arc_resample(q, False, STEP)
    return float((np.diff(r, 2, axis=0) ** 2).sum(1).mean())


@pytest.mark.parametrize("angle", [30, 45, 60])
def test_oblique_noisy_line_is_denoised_without_drift_and_ends_fixed(angle):
    pts, base, n = noisy_line(angle)
    out = sty.smooth_polyline(pts, False, SIGMA, STEP)
    assert np.array_equal(out[0], pts[0]) and np.array_equal(out[-1], pts[-1])                   # ujung tidak bergeser
    off = (out - np.array([50.0, 50.0])) @ n                                                      # jarak tegak lurus ke garis ideal
    inner = off[int(2 * SIGMA / STEP) + 2:-int(2 * SIGMA / STEP) - 2]
    assert abs(float(inner.mean())) < 0.1 and float(inner.std()) < 0.5 * float(((pts - base) @ n).std())
    assert second_diff_energy(out) < 0.4 * second_diff_energy(pts)                               # derau berkurang


@pytest.mark.parametrize("angle", [30, 45, 60])
def test_exact_line_is_unchanged_including_near_the_ends(angle):
    """Garis lurus tanpa derau tidak berubah sama sekali (pantulan ganjil di ujung): titik dekat ujung tidak tertarik ke dalam
    (tanpa penguncian ujung titik-titik di dekat ujung bergeser menuju ujung)."""
    d = np.array([math.cos(math.radians(angle)), math.sin(math.radians(angle))])
    line = np.array([50.0, 50.0]) + np.arange(0, 300, 1.0)[:, None] * d
    out = sty.smooth_polyline(line, False, SIGMA, STEP)
    assert len(out) == len(line) and float(np.abs(out - line).max()) < 1e-6


def test_noise_reduction_is_strong_on_noisy_circle_and_circle_does_not_shrink():
    r0 = 300.0
    rng = np.random.default_rng(3)
    th = np.linspace(0, 2 * np.pi, 1900, endpoint=False)
    c = np.column_stack([400 + r0 * np.cos(th), 400 + r0 * np.sin(th)]) + rng.uniform(-0.8, 0.8, (1900, 2))
    out = sty.smooth_polyline(c, True, SIGMA, STEP)
    rad = np.hypot(out[:, 0] - 400, out[:, 1] - 400)
    assert rad.std() < 0.5 * np.hypot(c[:, 0] - 400, c[:, 1] - 400).std()                       # derau berkurang
    assert r0 - rad.mean() < 0.1 + SIGMA**2 / (2 * r0)                                           # penyusutan ≤ σ²/(2R) + margin
    assert not np.allclose(out[0], out[-1])                                                      # tertutup: tanpa titik ganda


def test_arc_open_keeps_ends_and_does_not_shrink():
    th = np.radians(np.linspace(200, 340, 500))
    arc = np.column_stack([400 + 200 * np.cos(th), 400 + 200 * np.sin(th)])
    arc += np.random.default_rng(5).uniform(-0.5, 0.5, arc.shape)
    out = sty.smooth_polyline(arc, False, SIGMA, STEP)
    assert np.array_equal(out[0], arc[0]) and np.array_equal(out[-1], arc[-1])
    mid = out[len(out) // 4: -len(out) // 4]
    assert 200 - np.hypot(mid[:, 0] - 400, mid[:, 1] - 400).mean() < 0.1 + SIGMA**2 / 400


@pytest.mark.parametrize("deg", [90, 60, 40])
def test_sharp_corners_are_preserved(deg):
    """Sudut tajam tetap tajam walau sigma besar: jarak sudut asli ke hasil ≤ setengah langkah re-sample (titik kunci = sampel
    terdekat; 0,6 langkah dengan margin). Tanpa penguncian sudut membulat > 2 px."""
    a = math.radians(deg)
    v = np.array([[200.0, 200.0], [600.0, 200.0], [200.0 + 400 * math.cos(a), 200.0 + 400 * math.sin(a)]])
    poly = sty.arc_resample(v, True, STEP)
    out = sty.smooth_polyline(poly, True, 6.0, STEP)
    for corner in v:
        assert float(np.hypot(*(out - corner).T).min()) <= 0.6 * STEP, (deg, corner)


def test_shallow_bend_below_threshold_is_smoothed_not_pinned():
    v = np.array([[100.0, 100.0], [300.0, 100.0], [500.0, 160.0]])                              # belok ±17°: bukan sudut tajam
    path = sty.arc_resample(v, False, STEP)
    out = sty.smooth_polyline(path, False, 8.0, STEP)
    assert float(np.hypot(*(out - v[1]).T).min()) > 0.5                                         # tikungan dibulatkan


def test_closed_seam_has_no_notch_and_loop_equals_closed():
    pts = np.asarray(circle(200.5, 200.5, 120), float) * 1.0
    rng = np.random.default_rng(9)
    noisy = pts + rng.uniform(-0.6, 0.6, pts.shape)
    out = sty.smooth_polyline(noisy, True, SIGMA, STEP)
    rad = np.hypot(out[:, 0] - 200.5, out[:, 1] - 200.5)
    n = len(rad)
    seam = np.r_[rad[:12], rad[-12:]]
    rest = rad[40:-40]
    assert float(np.abs(seam - rad.mean()).max()) < float(np.abs(rest - rad.mean()).max()) * 1.5 + 0.2
    step_seam = float(np.hypot(*(out[0] - out[-1])))
    assert step_seam < 1.5 * STEP                                                                # sambungan menyatu (tanpa celah)
    # geometri penuh: strok tertutup vs loop terbuka (titik akhir = titik awal) → polyline sama
    g = geom_for()
    pc = sty.stroke_pieces(stroke(circle(50.5, 50.5, 30)), g, sty.new_stats())
    loop = sty.stroke_pieces(stroke(circle(50.5, 50.5, 30) + [circle(50.5, 50.5, 30)[0]], closed=False, typ="group_boundary"),
                             g, sty.new_stats())
    assert len(pc) == len(loop) == 1 and pc[0].closed and loop[0].closed
    assert np.allclose(pc[0].points, loop[0].points)


def test_determinism_and_off_switch():
    pts, _, _ = noisy_line(37)
    a, b = sty.smooth_polyline(pts, False, SIGMA, STEP), sty.smooth_polyline(pts, False, SIGMA, STEP)
    assert np.array_equal(a, b)
    assert sty.smooth_polyline(pts, False, 0.0, STEP) is pts                                    # smooth_px 0 = mati
    s = outer_stroke(TRAPEZOID)
    off, on = style_for(**{"shape.smooth_px": 0}), style_for()
    po = sty.stroke_pieces(s, sty.make_geometry(off, W, H), sty.new_stats())
    pn = sty.stroke_pieces(s, sty.make_geometry(on, W, H), sty.new_stats())
    assert not np.array_equal(po[0].points, pn[0].points)
    assert sty.render_frame({"frame_index": 0, "width": W, "height": H, "source": {}, "prev_sha256": None,
                             "strokes": [s]}, sty.make_geometry(on, W, H), on)[1] == \
        sty.render_frame({"frame_index": 0, "width": W, "height": H, "source": {}, "prev_sha256": None,
                          "strokes": [s]}, sty.make_geometry(on, W, H), on)[1]


def test_edge_crossing_ends_do_not_move_in_pieces():
    """Titik silang tepi (ujung jalur terpotong) tetap di koordinat kontur terskala walau dihaluskan."""
    s = outer_stroke(TRAPEZOID)
    g = geom_for()
    pieces = sty.stroke_pieces(s, g, sty.new_stats())
    p = np.asarray(s["points"], float)
    on = [bool(x) for x in sty.edge_sides(p, W, H)]
    crossings = [p[i] * g.scale for i in range(len(p)) if on[i] and not (on[i - 1] and on[(i + 1) % len(p)])]
    pts = np.vstack([pc.points for pc in pieces])
    for c in crossings:
        assert float(np.hypot(*(pts - c).T).min()) < 0.02 + 1e-9                                  # ada titik jalur tepat di titik silang


def test_smoothing_leaves_edge_tests_intact_and_reduces_vertices():
    """Pada kontur kisi berderau (tangga piksel) penghalusan mengurangi jumlah titik approx dibanding tanpa."""
    pts = lattice([(15, 85), (35, 30), (60, 70), (85, 25), (90, 90)])
    s = stroke(pts)
    n_on = len(sty.stroke_pieces(s, geom_for(**{"render.output_width": 1080}), sty.new_stats())[0].points)
    n_off = len(sty.stroke_pieces(s, geom_for(**{"render.output_width": 1080, "shape.smooth_px": 0}), sty.new_stats())[0].points)
    assert n_on <= n_off
