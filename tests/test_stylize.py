"""Test stage [5] stylize (T-203a): geometri, raster, SVG, tepi, stage (resume / stale / identitas / determinisme) dan data
nyata (dilewati bila klip / contours T-202 tidak ada). Fungsi metrik: tests/stylize_metrics.py. Sintetis sengaja TIDAK hanya
sumbu-sejajar (bug junction T-201a lolos karena itu): garis miring, busur, kontur non-konveks, kontak tepi miring, loop."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import stylize_metrics as sm
from test_vectorize import cfg_for, quiet

from rotoscope import stylize as sty
from rotoscope import vectorize as vec
from rotoscope.config import load_style
from rotoscope.stage_common import clip_identity

W = H = 100                  # frame kerja sintetis
OW = 256                     # lebar output sintetis (minimum yang diizinkan) → s = 2,56
WIDTH_REF = 25.3             # px ref → 25.3 × (256 / 1080) = 6.0 px output
FIDELITY_TOL_WIDTH_FRACTION = 0.35   # toleransi kesetiaan geometri sintetis = bagian dari tebal garis (bukan epsilon)


def style_for(**ov):
    base = {"render.output_width": OW, "stroke.width_base": WIDTH_REF}
    return load_style(None, overrides={**base, **ov})


def geom_for(**ov):
    return sty.make_geometry(style_for(**ov), W, H)


def stroke(points, closed=True, typ="silhouette", tid=1):
    pts = [[float(x), float(y)] for x, y in points]
    s = {"track_id": tid, "type": typ, "closed": closed, "groups": ["background"] if closed else ["a", "b"], "points": pts}
    if typ in ("silhouette", "silhouette_hole"):
        s["anchor"] = 0
    if typ == "occlusion":
        s.update(groups=["torso"], strength=0.1)
    return s


def doc_for(strokes, index=0, w=W, h=H):
    return {"frame_index": index, "width": w, "height": h, "source": {"vectorize_hash": "h" * 8},
            "prev_sha256": None, "strokes": strokes}


def lattice(vertices, closed=True, w=W, h=H, step=0.25):
    """Poligon kontinu → titik kisi pusat piksel (k + 0,5) seperti kontur [4]; di luar frame di-clamp ke tepi (run tepi)."""
    v = [np.array(p, float) for p in vertices]
    if closed:
        v.append(v[0])
    pts: list[tuple[float, float]] = []
    for a, b in zip(v[:-1], v[1:]):
        n = max(int(np.hypot(*(b - a)) / step), 1)
        for t in np.arange(n) / n:
            x, y = a + (b - a) * t
            q = (min(max(math.floor(x) + 0.5, 0.5), w - 0.5), min(max(math.floor(y) + 0.5, 0.5), h - 0.5))
            if not pts or pts[-1] != q:
                pts.append(q)
    while closed and len(pts) > 1 and pts[0] == pts[-1]:
        pts.pop()
    return pts


def circle(cx, cy, r, n=360):
    return lattice([(cx + r * math.cos(2 * math.pi * k / n), cy + r * math.sin(2 * math.pi * k / n)) for k in range(n)])


def render(strokes, g=None, st=None):
    st = st or style_for()
    g = g or sty.make_geometry(st, W, H)
    doc = doc_for(strokes)
    svg, png, stats = sty.render_frame(doc, g, st)
    pieces, _ = sty.frame_pieces(doc, g)
    return g, svg, png, stats, pieces, sm.coverage(sm.decode_png(png), g)


# ── Ukuran + satuan ────────────────────────────────
def test_output_size_even_and_rounding():
    assert sty.output_size(480, 854, 1080) == (1080, 1922)         # 1921,5 → 1922
    assert sty.output_size(480, 854, 720) == (720, 1282)           # 1281 → genap 1282
    assert sty.output_size(100, 100, 256) == (256, 256)
    assert sty.output_size(100, 99, 256) == (256, 254)             # 253,44 → 253 → genap 254
    w, h = sty.output_size(7, 11, 256)
    assert h % 2 == 0 and abs(h - 11 * 256 / 7) <= 1


def test_geometry_units_scale_with_output_width():
    g1, g2 = geom_for(**{"render.output_width": 1080}), geom_for(**{"render.output_width": 540})
    assert g1.unit == 1.0 and g2.unit == 0.5
    assert g1.widths["silhouette"] == pytest.approx(WIDTH_REF) and g2.widths["silhouette"] == pytest.approx(WIDTH_REF / 2)
    assert g1.epsilon == pytest.approx(2.8) and g2.epsilon == pytest.approx(1.4)
    assert g1.smooth == pytest.approx(5.0) and g2.smooth == pytest.approx(2.5)
    assert g1.scale == pytest.approx(10.8) and g2.scale == pytest.approx(5.4)


def test_output_width_does_not_change_look():
    """Tampilan identik (setara) pada 2 output_width: gambar besar diperkecil ≈ gambar kecil."""
    pts = lattice([(20, 70), (45, 25), (80, 60), (60, 85)])
    big = render([stroke(pts)], geom_for(**{"render.output_width": 400}), style_for(**{"render.output_width": 400}))
    small = render([stroke(pts)])
    import cv2
    shrunk = cv2.resize(big[5], (OW, OW), interpolation=cv2.INTER_AREA)
    diff = np.abs(shrunk - small[5])
    assert float(diff.mean()) < 0.02 and float(np.percentile(diff, 99.9)) < 0.5       # selisih cakupan (0–1)


def test_cap_other_than_round_rejected():
    with pytest.raises(sty.StageError, match="cap"):
        sty.make_geometry(load_style(None, overrides={"stroke.cap": "butt"}), W, H)


# ── Penyelarasan raster (±0,25 px output) + tebal ──
@pytest.mark.parametrize("ss", [3, 4])
@pytest.mark.parametrize("angle", [0, 30, 45, 60])
def test_line_alignment_and_width(angle, ss):
    a = np.array([20.5, 30.5])
    b = a + 50 * np.array([math.cos(math.radians(angle)), math.sin(math.radians(angle))])
    g, _, _, _, pieces, cov = render([stroke([a, b], closed=False, typ="group_boundary")], geom_for(**{"render.ss": ss}),
                                     style_for(**{"render.ss": ss}))
    A, B = a * g.scale, b * g.scale
    xs = np.sort([A[0], B[0]])
    pos, width = sm.line_alignment_error(cov, A, B, xs[0] + 4 * g.scale, xs[1] - 4 * g.scale)
    assert abs(pos) <= 0.3, (angle, ss, pos)                      # ±0,25 px output (keputusan Rio) + kuantisasi grid ss
    assert abs(width - g.widths["group_boundary"]) <= 0.4, (angle, ss, width)


@pytest.mark.parametrize("ss", [3, 4])
def test_no_systematic_offset_over_subpixel_positions(ss):
    """Galat posisi RATA-RATA atas posisi sub-piksel acak ≈ 0: menangkap offset koordinat 0,5 piksel ss yang hilang
    (0,5 / ss = 0,12–0,17 px output pada garis horizontal) yang tersembunyi di bawah kuantisasi per kasus (±0,25)."""
    rng = np.random.default_rng(7)
    pos_all, width_all = [], []
    for _ in range(16):
        off = rng.uniform(0, 1, 2)
        a, b = np.array([20.5, 40.5]) + off, np.array([80.5, 40.5]) + off
        g, _, _, _, _, cov = render([stroke([a, b], closed=False, typ="group_boundary")], geom_for(**{"render.ss": ss}),
                                    style_for(**{"render.ss": ss}))
        pos, width = sm.line_alignment_error(cov, a * g.scale, b * g.scale, 60.0, 190.0)
        pos_all.append(pos)
        width_all.append(width - g.widths["group_boundary"])
    assert abs(np.mean(pos_all)) <= 0.05 and abs(np.mean(width_all)) <= 0.1, (np.mean(pos_all), np.mean(width_all))


# ── Tanpa celah, tebal seragam, sambungan ──────────
SHAPES = {
    "circle": lambda: (circle(50.5, 50.5, 30), True),
    "star_nonconvex": lambda: (lattice([(50 + (34 if k % 2 == 0 else 14) * math.cos(math.pi * k / 5 - 1.2),
                                         50 + (34 if k % 2 == 0 else 14) * math.sin(math.pi * k / 5 - 1.2))
                                        for k in range(10)]), True),
    "L_shape": lambda: (lattice([(20, 20), (45, 20), (45, 60), (80, 60), (80, 80), (20, 80)]), True),
    "arc_open": lambda: (lattice([(50 + 35 * math.cos(math.radians(a)), 60 + 35 * math.sin(math.radians(a)))
                                  for a in range(200, 341, 2)], closed=False), False),
}


@pytest.mark.parametrize("name", SHAPES)
def test_no_gaps_on_centerline_and_no_stray_ink(name):
    pts, closed = SHAPES[name]()
    g, _, _, _, pieces, cov = render([stroke(pts, closed=closed, typ="silhouette" if closed else "group_boundary")])
    assert pieces and sm.centerline_gaps(cov, pieces, g) == 0
    ys, xs = np.nonzero(cov > 0.05)
    ink = np.column_stack([xs + 0.5, ys + 0.5])[::7]
    dist = np.min([sm.point_polyline_distance(ink, sm.piece_curve(p)) for p in pieces], axis=0)
    assert float(dist.max()) <= g.widths["silhouette"] / 2 + 1.0       # tidak ada tinta jauh dari garis tengah


def test_circle_fidelity_and_loop_has_no_seam_notch():
    pts, _ = SHAPES["circle"]()
    g = geom_for()
    s = stroke(pts)
    dev = sm.stroke_deviation(s, g)
    # Toleransi = 0,35 × tebal garis (garis tengah tetap jauh di dalam pita tinta ±tebal/2), BUKAN epsilon: approxPolyDP
    # tidak menjamin ≤ epsilon (terukur melebihi sampai +31% pada ±5% strok panjang). Terukur lingkaran ini: 1,56 px.
    assert float(dev.max()) <= FIDELITY_TOL_WIDTH_FRACTION * g.widths["silhouette"]
    # loop terbuka (titik akhir = titik awal) digambar sama dengan tertutup, tanpa takik di sambungan
    loop = stroke(pts + [pts[0]], closed=False, typ="group_boundary")
    stats = sty.new_stats()
    pcs = sty.stroke_pieces(loop, g, stats)
    assert len(pcs) == 1 and pcs[0].closed
    _, _, _, _, pieces, cov = render([loop])
    assert sm.centerline_gaps(cov, pieces, g) == 0


def test_spline_is_close_to_dense_reference():
    """Polyline spline (adaptif) vs referensi sangat rapat: ≤ ~0,1 px output."""
    g = geom_for()
    p = np.array([[10, 10], [40, 90], [60, 20], [130, 100], [150, 40]], float)
    fine = sty.catmull_rom(p, g.tension, 512, 1e-6, False)
    for closed in (False, True):
        coarse = sty.catmull_rom(p, g.tension, g.steps, g.spline_tol, closed)
        ref = sty.catmull_rom(p, g.tension, 512, 1e-6, closed)
        curve = sm.closed_polyline(coarse) if closed else coarse
        assert float(sm.point_polyline_distance(ref, curve).max()) <= g.spline_tol * 1.5 + 1e-6
    assert len(fine) > len(sty.catmull_rom(p, g.tension, g.steps, g.spline_tol, False))


def test_catmull_closed_is_periodic_and_open_hits_endpoints():
    p = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)
    closed = sty.catmull_rom(p, 0.5, 8, 0.01, True)
    assert np.allclose(closed[0], p[0]) and not np.allclose(closed[-1], closed[0])
    assert any(np.allclose(q, p[2]) for q in closed)
    op = sty.catmull_rom(p, 0.5, 8, 0.01, False)
    assert np.allclose(op[0], p[0]) and np.allclose(op[-1], p[-1])
    assert np.allclose(sty.catmull_rom(p[:2], 0.5, 8, 0.01, False), p[:2])


# ── Tepi frame ─────────────────────────────────────
def outer_stroke(vertices):
    return stroke(lattice(vertices))


TRAPEZOID = [(30, 130), (45, 60), (55, 60), (70, 130)]
ARCH = [(20, 130), (20, 50), (50, 25), (80, 50), (80, 130), (68, 130), (68, 55), (50, 40), (32, 55), (32, 130)]
RIGHT_BLOB = [(70, 30), (130, 40), (130, 80), (75, 85), (60, 55)]
SHALLOW = [(5, 60), (5, 92), (120, 104), (120, 60)]


def tails(pieces, g):
    """Titik tempat ekor ekstensi meninggalkan kanvas: [(edge, koordinat sepanjang tepi)]."""
    out = []
    for pc in pieces:
        for inner, outer in ((pc.points[1], pc.points[0]), (pc.points[-2], pc.points[-1])):
            if 0 <= outer[0] <= g.out_w and 0 <= outer[1] <= g.out_h:
                continue
            d = outer - inner
            ts = []
            if outer[1] > g.out_h:
                ts.append(((g.out_h - inner[1]) / d[1], "bottom"))
            if outer[0] > g.out_w:
                ts.append(((g.out_w - inner[0]) / d[0], "right"))
            if outer[0] < 0:
                ts.append((-inner[0] / d[0], "left"))
            t, edge = min(ts)
            q = inner + t * d
            out.append((edge, float(q[0] if edge == "bottom" else q[1])))
    return out


def allowed_columns(pieces, g, edge, n):
    """Indeks piksel di tepi `edge` yang boleh bertinta (dekat titik keluar ekor)."""
    ok = np.zeros(n, bool)
    slack = max(g.widths.values()) + 2.0
    for e, c in tails(pieces, g):
        if e == edge:
            ok[max(int(c - slack), 0):int(c + slack) + 1] = True
    return ok


@pytest.mark.parametrize("name,verts", [("trapezoid", TRAPEZOID), ("arch", ARCH)])
def test_hide_bottom_ink_only_at_crossings_and_reaches_edge(name, verts):
    s = outer_stroke(verts)
    g, _, _, stats, pieces, cov = render([s])
    assert stats["edge_cuts"] == (1 if name == "trapezoid" else 2)       # 1 / 2 run di dasar → jalur
    assert len(pieces) == stats["edge_cuts"] and all(not p.closed for p in pieces)
    ink = np.zeros(g.out_w, bool)
    ink[sm.border_ink(cov)["bottom"]] = True
    ok = allowed_columns(pieces, g, "bottom", g.out_w)
    assert ink.any() and not (ink & ~ok).any()                           # tinta di baris terakhir hanya di titik silang
    for _, c in tails(pieces, g):
        assert cov[-1, int(c)] >= 0.99                                   # mencapai tepi kanvas tanpa celah
    base = cov[-2:, int(0.4 * g.out_w):int(0.6 * g.out_w)]
    assert float(base.max()) < 0.05                                      # dasar tidak digambar
    assert sm.centerline_gaps(cov, pieces, g) == 0


def test_hide_right_run_and_corner_stroke():
    g, _, _, stats, pieces, cov = render([outer_stroke(RIGHT_BLOB)])
    assert stats["edge_cuts"] >= 1
    ink = np.zeros(g.out_h, bool)
    ink[sm.border_ink(cov)["right"]] = True
    ok = allowed_columns(pieces, g, "right", g.out_h)
    assert ink.any() and not (ink & ~ok).any()
    # strok terbuka yang berakhir TEPAT di sudut kanvas (dua tepi sekaligus) menyentuh piksel sudut
    corner = stroke([(60.5, 60.5), (80.5, 80.5), (99.5, 99.5)], closed=False, typ="group_boundary")
    g, _, _, stats, pieces, cov = render([corner])
    assert stats["corner_ends"] == 1 and cov[-1, -1] >= 0.99
    assert sm.centerline_gaps(cov, pieces, g) == 0


def test_shallow_contact_uses_perpendicular_extension():
    g, _, _, stats, pieces, cov = render([outer_stroke(SHALLOW)])
    assert stats["shallow_ends"] >= 1 and stats["min_contact_deg"] <= sty.SHALLOW_CONTACT_DEG
    mids = sm.edge_run_midpoints(doc_for([outer_stroke(SHALLOW)]), g)
    assert len(mids) and float(sm.min_distance_to_pieces(mids, pieces).min()) >= 1.0    # run tepi tidak digambar / disusuri
    bottom_tails = [p for p in pieces for q in (p.points[0], p.points[-1]) if q[1] > g.out_h]
    assert bottom_tails


def test_extension_point_rule_unit():
    g = geom_for()
    end = np.array([60.0, g.out_h - 1.0])
    w = g.widths["silhouette"]
    q, info = sty.extension_point(end, np.array([0.3, 1.0]), ["bottom"], g, w)     # curam → searah tangen
    assert not info["shallow"] and q[1] >= g.out_h + w / 2 + sty.EDGE_MARGIN_PX - 1e-9
    assert (q - end)[0] / (q - end)[1] == pytest.approx(0.3)
    q, info = sty.extension_point(end, np.array([1.0, 0.2]), ["bottom"], g, w)     # dangkal → tegak lurus
    assert info["shallow"] and q[0] == pytest.approx(end[0]) and q[1] > g.out_h + w / 2
    q, info = sty.extension_point(end, np.array([1.0, -0.5]), ["bottom"], g, w)    # arah ke dalam → tegak lurus
    assert info["shallow"] and q[0] == pytest.approx(end[0])


def test_draw_mode_keeps_edge_run_and_hide_removes_it():
    s = outer_stroke(TRAPEZOID)
    g, _, _, _, pieces, cov = render([s], geom_for(**{"shape.edge_mode": "draw"}), style_for(**{"shape.edge_mode": "draw"}))
    assert float(cov[-2:, 110:150].max()) > 0.5                                  # dasar digambar (inset 0,5·s dari tepi)
    assert any(not p.closed for p in pieces) and not any(
        (q[1] > g.out_h or q[0] < 0 or q[0] > g.out_w) for p in pieces for q in p.points)     # tanpa ekstensi
    g2, _, _, _, _, cov2 = render([s])
    assert float(cov2[-2:, 110:150].max()) < 0.05


def test_open_stroke_end_reaches_right_edge_without_gap():
    s = stroke([(40.5, 40.5), (70.5, 48.5), (99.5, 50.5)], closed=False, typ="group_boundary")
    for mode in ("hide", "draw"):
        g, _, _, _, pieces, cov = render([s], geom_for(**{"shape.edge_mode": mode}), style_for(**{"shape.edge_mode": mode}))
        row = int(50.5 * g.scale)
        assert cov[row - 1:row + 2, -1].max() >= 0.9, mode
    # syarat minimum di mode draw: tebal ≥ s (jarak titik terakhir ke tepi kanvas 0,5·s, jari-jari ujung w/2)
    thin = geom_for(**{"shape.edge_mode": "draw", "stroke.width_base": 2.0})
    assert thin.widths["group_boundary"] < sty.draw_mode_min_width(thin)
    _, _, _, _, _, cov = render([s], thin, style_for(**{"shape.edge_mode": "draw", "stroke.width_base": 2.0}))
    assert cov[row - 1:row + 2, -1].max() < 0.9


def test_draw_mode_thin_width_warns_once_on_stderr_not_error(tmp_path, capsys):
    work = make_stage_work(tmp_path)
    thin = {"shape.edge_mode": "draw", "stroke.width_base": 2.0}                      # 0,47 px < s = 2,56
    run = sty.run_stylize(cfg_for(work), style_for(**thin), "t", log=quiet)
    err = capsys.readouterr().err
    assert run["processed"] == N_FRAMES and err.count("PERINGATAN") == 1                # satu kali per run, bukan per frame
    g = geom_for(**thin)
    assert f"{g.widths['silhouette']:.2f} px" in err and f"{sty.draw_mode_min_width(g):.2f} px" in err and "draw" in err
    for ok in ({"shape.edge_mode": "draw"}, {"shape.edge_mode": "hide", "stroke.width_base": 2.0}):   # tidak berlaku
        sty.run_stylize(cfg_for(work), style_for(**ok), "t", restart=True, log=quiet)
        assert capsys.readouterr().err == ""
    assert sty.draw_width_warning(geom_for(**{"shape.edge_mode": "draw"})) is None


def test_contour_fully_on_edge_is_dropped_in_hide_and_counted():
    s = stroke([(0.5, 99.5), (50.5, 99.5), (99.5, 99.5)], closed=False)
    _, _, _, stats, pieces, cov = render([s])
    assert stats["dropped_all_edge"] == 1 and not pieces and float(cov.max()) == 0.0


def test_empty_frame_is_paper_only():
    g, svg, png, stats, pieces, cov = render([])
    img = sm.decode_png(png)
    assert (img == np.array(g.paper, np.uint8)).all() and stats["n_pieces"] == 0
    root = sty.ET.fromstring(svg)
    assert root.get("viewBox") == f"0 0 {g.out_w} {g.out_h}" and not list(root.iter("{http://www.w3.org/2000/svg}path"))


# ── Jumlah strok, SVG ↔ PNG, union ────────────────
def test_all_strokes_drawn_one_path_each_without_edges():
    strokes = [stroke(circle(30.5, 30.5, 15), tid=1), stroke(circle(70.5, 70.5, 12), typ="silhouette_hole", tid=2),
               stroke([(10.5, 80.5), (30.5, 90.5)], closed=False, typ="group_boundary", tid=3),
               stroke([(60.5, 20.5), (90.5, 30.5)], closed=False, typ="occlusion", tid=4)]
    _, svg, _, stats, pieces, _ = render(strokes)
    assert stats["n_strokes"] == 4 and stats["n_pieces"] == 4 == len(sm.parse_svg(svg))
    assert [p.type for p in sm.parse_svg(svg)] == list(sty.TYPE_ORDER)                # urutan <g> tetap


def test_svg_points_equal_raster_points():
    pts = lattice([(15, 85), (35, 30), (60, 70), (85, 25), (90, 90)])
    st = style_for()
    g, svg, png, _, pieces, _ = render([stroke(pts)], None, st)
    parsed = sm.parse_svg(svg)
    assert len(parsed) == len(pieces)
    for a, b in zip(parsed, pieces):
        assert a.closed == b.closed and np.allclose(a.points, b.points, atol=1e-9)
    assert sty.render_png(parsed, g) == png                                          # raster dari titik SVG = PNG


def test_svg_structure_deterministic_and_wellformed():
    pts = lattice(TRAPEZOID)
    a = render([stroke(pts)])[1]
    b = render([stroke(pts)])[1]
    assert a == b and b"id=\"" in a and b"timestamp" not in a
    root = sty.ET.fromstring(a)
    ns = "{http://www.w3.org/2000/svg}"
    assert root.get("width") == str(OW) and root.get("height") == str(OW)
    groups = [e.get("id") for e in root.iter(ns + "g")]
    assert groups == list(sty.TYPE_ORDER)
    g0 = next(root.iter(ns + "g"))
    assert g0.get("stroke-linecap") == "round" and g0.get("stroke-linejoin") == "round" and g0.get("fill") == "none"
    assert next(root.iter(ns + "rect")).get("fill") == "#f4f1ea"


def test_overlap_not_darker_union_composition():
    """Dua strok bersilangan: piksel di tepi anti-alias perpotongan tidak lebih gelap daripada cakupan satu mask (union)."""
    a = stroke([(20.5, 20.5), (80.5, 70.5)], closed=False, typ="group_boundary", tid=1)
    b = stroke([(20.5, 60.5), (80.5, 30.5)], closed=False, typ="group_boundary", tid=2)
    g, _, _, _, pieces, cov = render([a, b])
    mask = sty.render_mask(pieces, g)
    ref = cv2_resize(mask, g)
    assert float(np.abs(cov - ref).max()) < 0.01 and float(cov.max()) <= 1.0 + 1e-9
    # strok yang tumpang tindih persis (anti-alias setengah di tepi yang sama) tidak boleh lebih gelap: union, bukan
    # komposisi per-strok (1 − (1 − a)² = 0,75 untuk a = 0,5)
    dup = [stroke([(20.5, 20.5), (80.5, 70.5)], closed=False, typ="group_boundary", tid=t) for t in (1, 2)]
    single = render(dup[:1])[5]
    twice = render(dup)[5]
    assert float(np.abs(twice - single).max()) < 0.01 and 0.2 < float(single[(single > 0.2) & (single < 0.8)].mean()) < 0.8


def cv2_resize(mask, g):
    import cv2
    return cv2.resize(mask, (g.out_w, g.out_h), interpolation=cv2.INTER_AREA).astype(float) / 255.0


# ── Stage ──────────────────────────────────────────
N_FRAMES = 4
STYLE_YAML = f"render:\n  output_width: {OW}\nstroke:\n  width_base: {WIDTH_REF}\n"


def frame_strokes(i: int) -> list[dict]:
    d = i * 2
    return [stroke(lattice([(25 + d, 90), (35 + d, 40), (60 + d, 35), (75 + d, 130)]), tid=1),
            stroke(circle(60.5, 30.5 + d, 8), typ="silhouette_hole", tid=2),
            stroke([(10.5, 50.5 + d), (30.5, 70.5)], closed=False, typ="group_boundary", tid=3)]


def make_stage_work(tmp_path: Path, n: int = N_FRAMES, contract: str = "T-202") -> Path:
    work = tmp_path / "work"
    (work / "contours").mkdir(parents=True)
    (work / "meta.json").write_text(json.dumps({"source_path": "C:/clips/a.mp4", "frame_count": n, "working_width": W,
                                                "working_height": H, "frame_index_start": 0}), encoding="utf-8")
    (work / "contours" / "manifest.json").write_text(json.dumps(
        {"stage": "vectorize", "contract": contract, "algo_rev": 2, "pending": [], "vectorize_hash": "h" * 8,
         "created_utc": "t0", "frame_size": {"width": W, "height": H}, "clip": clip_identity(work)}), encoding="utf-8")
    for i in range(n):
        (work / "contours" / f"frame_{i:05d}.json").write_text(json.dumps(doc_for(frame_strokes(i), i)), encoding="utf-8")
    return work


@pytest.fixture
def style_file(tmp_path):
    p = tmp_path / "test-style.yaml"
    p.write_text(STYLE_YAML, encoding="utf-8")
    return p


def run_main(work: Path, style_file: Path, *extra: str) -> int:
    return sty.main(["--work-dir", str(work), "--style", str(style_file), *extra])


def out_hashes(work: Path) -> dict[str, str]:
    d = work / "strokes"
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.glob("frame_*")) if p.suffix in (".svg", ".png")}


def manifest(work: Path) -> dict:
    return json.loads((work / "strokes" / "manifest.json").read_text(encoding="utf-8"))


def test_run_writes_outputs_manifest_and_log(tmp_path, style_file, capsys):
    work = make_stage_work(tmp_path)
    assert run_main(work, style_file) == 0
    h = out_hashes(work)
    assert len(h) == 2 * N_FRAMES
    m = manifest(work)
    assert m["contract"] == "T-203a" and m["algo_rev"] == sty.ALGO_REV and m["style"] == "test-style"
    assert m["output_size"] == {"width": OW, "height": OW} and m["scale"] == OW / W and m["edge_mode"] == "hide"
    assert m["contours"]["contract"] == "T-202" and m["contours"]["vectorize_hash"] == "h" * 8
    assert "jitter.amplitude" in m["ignored_params"] and "shape.resample_points" in m["ignored_params"]
    assert set(m["style_params"]) == set(sty.active_param_names()) and m["style_hash"]
    recs = [json.loads(x) for x in (work / "strokes" / "frames.jsonl").read_text(encoding="utf-8").splitlines()]
    frames = [r for r in recs if r["event"] == "frame"]
    assert len(frames) == N_FRAMES and all({"geom_s", "raster_png_s", "svg_bytes", "png_bytes", "edge_cuts"} <= set(r) for r in frames)
    for i in range(N_FRAMES):
        assert sty.frame_valid(sty.load_clip(work), f"frame_{i:05d}.png", (OW, OW))


def test_determinism_from_scratch_and_resume_limit(tmp_path, style_file):
    work = make_stage_work(tmp_path)
    assert run_main(work, style_file) == 0
    full = out_hashes(work)
    assert run_main(work, style_file, "--restart") == 0
    assert out_hashes(work) == full                                                   # dua run dari nol identik
    assert run_main(work, style_file, "--restart", "--limit", "2") == 0
    assert len(out_hashes(work)) == 4
    run = sty.run_stylize(cfg_for(work), style_for(), "t", limit=None, log=quiet)
    assert run["skipped"] == 2 and run["processed"] == 2
    assert out_hashes(work) == full                                                   # --limit lalu penuh = run penuh
    run = sty.run_stylize(cfg_for(work), style_for(), "t", log=quiet)
    assert run["processed"] == 0 and run["skipped"] == N_FRAMES


def test_damaged_frame_is_recomputed_only(tmp_path, style_file):
    work = make_stage_work(tmp_path)
    run_main(work, style_file)
    full = out_hashes(work)
    (work / "strokes" / "frame_00001.png").write_bytes(b"x")
    (work / "strokes" / "frame_00002.svg").write_text("<svg", encoding="utf-8")
    run = sty.run_stylize(cfg_for(work), style_for(), "t", log=quiet)
    assert run["processed"] == 2 and out_hashes(work) == full


@pytest.mark.parametrize("override", [{"render.output_width": 400}, {"shape.edge_mode": "draw"},
                                      {"shape.simplify_epsilon": 9.0}, {"shape.smooth_px": 0.0}, {"stroke.width_base": 20.0}, {"stroke.color": "#112233"}])
def test_stale_style_change_warns_and_recomputes(tmp_path, override):
    work = make_stage_work(tmp_path)
    logs: list[str] = []
    sty.run_stylize(cfg_for(work), style_for(), "t", log=logs.append)
    first = out_hashes(work)
    logs.clear()
    run = sty.run_stylize(cfg_for(work), style_for(**override), "t", log=logs.append)
    key = next(iter(override))
    assert run["processed"] == N_FRAMES and any("basi" in m and key in m for m in logs)
    assert out_hashes(work) != first
    sty.run_stylize(cfg_for(work), style_for(), "t", log=quiet)                       # kembali → hash identik
    assert out_hashes(work) == first


def test_inactive_param_change_does_not_recompute(tmp_path):
    work = make_stage_work(tmp_path)
    sty.run_stylize(cfg_for(work), style_for(), "t", log=quiet)
    logs: list[str] = []
    run = sty.run_stylize(cfg_for(work), style_for(**{"jitter.amplitude": 9.0}), "t", log=logs.append)
    assert run["processed"] == 0 and run["stale"] == [] and any("jitter.amplitude" in m for m in logs)


def test_stale_when_contours_manifest_changes(tmp_path):
    work = make_stage_work(tmp_path)
    sty.run_stylize(cfg_for(work), style_for(), "t", log=quiet)
    p = work / "contours" / "manifest.json"
    m = json.loads(p.read_text(encoding="utf-8"))
    m["algo_rev"] = 3
    m["created_utc"] = "t1"
    p.write_text(json.dumps(m), encoding="utf-8")
    logs: list[str] = []
    run = sty.run_stylize(cfg_for(work), style_for(), "t", log=logs.append)
    assert run["processed"] == N_FRAMES and any("contours.algo_rev" in x for x in logs)


def test_output_without_manifest_is_refused(tmp_path):
    work = make_stage_work(tmp_path)
    (work / "strokes").mkdir()
    (work / "strokes" / "frame_00000.svg").write_text("<svg/>", encoding="utf-8")
    with pytest.raises(sty.StageError, match="tanpa manifest"):
        sty.run_stylize(cfg_for(work), style_for(), "t", log=quiet)


def edit_contours_manifest(work: Path, **changes):
    p = work / "contours" / "manifest.json"
    m = json.loads(p.read_text(encoding="utf-8"))
    m.update(changes)
    p.write_text(json.dumps(m), encoding="utf-8")


def test_input_validation_exit_codes(tmp_path, style_file, capsys):
    cmd = "vectorize"
    work = make_stage_work(tmp_path / "a", contract="T-201b")                          # contract tidak didukung
    assert run_main(work, style_file) == 1
    err = capsys.readouterr().err
    assert "T-201b" in err and "T-202" in err and cmd in err
    work = make_stage_work(tmp_path / "b")
    edit_contours_manifest(work, pending=["anchor"])
    assert run_main(work, style_file) == 1 and "pending" in capsys.readouterr().err
    work = make_stage_work(tmp_path / "c")
    edit_contours_manifest(work, clip={"meta_sha256": "0" * 64, "source_path": "C:/lain.mp4"})
    assert run_main(work, style_file) == 1 and "LAIN" in capsys.readouterr().err
    work = make_stage_work(tmp_path / "d")
    m = json.loads((work / "contours" / "manifest.json").read_text(encoding="utf-8"))
    m.pop("clip")
    (work / "contours" / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
    assert run_main(work, style_file) == 1 and "identitas klip" in capsys.readouterr().err
    work = make_stage_work(tmp_path / "e")
    (work / "contours" / "manifest.json").unlink()
    assert run_main(work, style_file) == 1 and "stage [4]" in capsys.readouterr().err
    work = make_stage_work(tmp_path / "f")
    edit_contours_manifest(work, frame_size={"width": 50, "height": 50})
    assert run_main(work, style_file) == 1 and "frame_size" in capsys.readouterr().err
    work = make_stage_work(tmp_path / "g")
    (work / "contours" / "frame_00002.json").write_text("{", encoding="utf-8")
    assert run_main(work, style_file) == 1 and "frame_00002" in capsys.readouterr().err
    work = make_stage_work(tmp_path / "h")
    d = json.loads((work / "contours" / "frame_00001.json").read_text(encoding="utf-8"))
    d["source"]["vectorize_hash"] = "other"
    (work / "contours" / "frame_00001.json").write_text(json.dumps(d), encoding="utf-8")
    assert run_main(work, style_file) == 1
    assert not (work / "strokes").exists()                                             # gagal sebelum menulis apa pun


def test_limit_checks_only_selected_frames_and_validation(tmp_path, style_file, capsys):
    work = make_stage_work(tmp_path)
    (work / "contours" / "frame_00003.json").unlink()
    assert run_main(work, style_file, "--limit", "3") == 0
    assert run_main(work, style_file) == 1
    assert run_main(work, style_file, "--limit", "0") == 1


def test_restart_removes_only_strokes_dir(tmp_path, style_file):
    work = make_stage_work(tmp_path)
    run_main(work, style_file)
    before = {p.name: p.read_bytes() for p in (work / "contours").iterdir()}
    run_main(work, style_file, "--restart")
    assert {p.name: p.read_bytes() for p in (work / "contours").iterdir()} == before


def test_cli_subcommand_registered():
    from rotoscope import cli
    assert cli.CPU_MAINS["stylize"] is sty.main and "stylize" in cli.STAGES      # masuk `run` sejak T-203b
    args = cli.build_parser().parse_args(["stylize", "x.mp4", "--limit", "3"])
    assert args.stage == "stylize" and args.rest == ["--limit", "3"]


def test_stage_does_not_import_torch():
    code = ("import sys; import rotoscope.stylize, rotoscope.cli; "
            "assert 'torch' not in sys.modules, 'torch di-import'")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_default_style_values_converted_from_look_test():
    s = load_style(None)
    assert s.stroke.width_base == 9.0 and s.shape.simplify_epsilon == 2.8 and s.shape.smooth_px == 5.0 and s.render.output_width == 1080
    assert s.shape.edge_mode == "hide"
    active, ignored, _ = sty.style_params(s)
    assert not set(active) & set(ignored) and "shape.resample_points" in ignored


# ── Data nyata (dilewati bila klip / contours T-202 tidak ada) ──
CLIPS = Path(__file__).resolve().parents[1] / "work" / "clips"
REAL_STEP = 12


def real_docs(clip: str):
    d = CLIPS / clip / "contours"
    man = d / "manifest.json"
    if not man.is_file() or json.loads(man.read_text(encoding="utf-8")).get("contract") != "T-202":
        pytest.skip(f"contours T-202 klip {clip} tidak ada")
    return [json.loads(f.read_text(encoding="utf-8")) for f in sorted(d.glob("frame_*.json"))[::REAL_STEP]]


@pytest.mark.parametrize("clip", ["test_short", "test"])
def test_real_clip_fidelity_edges_and_counts(clip):
    docs = real_docs(clip)
    st = load_style(None)
    g = sty.make_geometry(st, 480, 854)
    assert (g.out_w, g.out_h) == (1080, 1922)
    per: dict[str, list[float]] = {t: [] for t in sty.TYPE_ORDER}
    for d in docs:
        svg, png, stats = sty.render_frame(d, g, st)
        pieces, _ = sty.frame_pieces(d, g)
        assert stats["n_pieces"] >= stats["n_strokes"] - stats["dropped_all_edge"] and stats["dropped_all_edge"] == 0
        cov = sm.coverage(sm.decode_png(png), g)
        mids = sm.edge_run_midpoints(d, g)
        assert len(mids) and float(sm.min_distance_to_pieces(mids, pieces).min()) >= 1.0, (clip, d["frame_index"])
        for edge, c in tails(pieces, g):                                              # ekor mencapai tepi kanvas tanpa celah
            assert cov[-1, int(c)] >= 0.99 if edge == "bottom" else cov[int(c), -1 if edge == "right" else 0] >= 0.99
        for s in d["strokes"]:
            dev = sm.stroke_deviation(s, g)
            if dev.size:
                per[s["type"]].append(float(dev.max()))
    # ambang regresi dari pengukuran Tahap 3 (epsilon 5,6 ref): deviasi maks per strok, px output
    bound = {"silhouette": 30.0, "silhouette_hole": 22.0, "group_boundary": 22.0, "occlusion": 12.0}
    for t, v in per.items():
        if v:
            assert max(v) <= bound[t], (clip, t, max(v))
