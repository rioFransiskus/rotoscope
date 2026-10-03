"""Test track.py (T-202): orientasi, anchor, arah garis terbuka, track_id. Sintetis, TIDAK hanya sumbu-sejajar (kontur
miring / non-konveks / berputar, garis kemiringan berganti tanda, dua garis sejajar berdekatan, split / merge,
kelahiran / kematian, seri jarak, loop). Tiap pengaman punya test yang gagal bila pengaman dimatikan (mutation check
dengan plugin pytest di luar repo: orientasi, kesinambungan arah, pencocokan optimal, tanda sumbu)."""

from __future__ import annotations

import math
import subprocess
import sys

import numpy as np
import pytest

from rotoscope import track as trk

MAX_DIST = 12.0
GROUPS = ["left_arm", "torso"]


# ── Pembangun bentuk ───────────────────────────────
def densify(vertices, step: float = 1.0) -> np.ndarray:
    """Poligon tertutup → titik rapat (jarak ≤ step), titik awal tidak diulang, koordinat 3 desimal."""
    out = []
    for (x0, y0), (x1, y1) in zip(vertices, vertices[1:] + vertices[:1]):
        n = max(1, int(math.hypot(x1 - x0, y1 - y0) / step))
        out += [(x0 + (x1 - x0) * k / n, y0 + (y1 - y0) * k / n) for k in range(n)]
    return np.round(np.array(out), 3)


def blob(center, rot_deg: float = 0.0, n: int = 240) -> np.ndarray:
    """Kontur tidak-konveks, tidak simetris (radius berombak), searah jarum jam di layar (θ naik, y ke bawah)."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = 30 + 9 * np.sin(3 * t) + 5 * np.cos(5 * t + 1)
    a = np.radians(rot_deg)
    x, y = r * np.cos(t + a), r * np.sin(t + a)
    return np.round(np.stack([x + center[0], y + center[1]], axis=1), 3)


def line(p0, p1, n: int = 81) -> np.ndarray:
    return np.round(np.linspace(p0, p1, n), 3)


def stroke(kind: str, pts, closed: bool | None = None, groups=None, **extra) -> dict:
    closed = kind in ("silhouette", "silhouette_hole") if closed is None else closed
    return {"type": kind, "closed": closed, "groups": groups or (["background"] if closed else list(GROUPS)),
            "points": np.asarray(pts).tolist(), **extra}


def loop_pts(center, r: float = 20.0, n: int = 80, ccw: bool = True) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n, endpoint=False) * (-1 if ccw else 1)
    ring = np.round(np.stack([center[0] + r * np.cos(t), center[1] + r * np.sin(t)], axis=1), 3)
    return np.vstack([ring, ring[:1]])


def step(tr: trk.Tracker, *strokes: dict, hair_face=None) -> list[dict]:
    out = list(strokes)
    tr.step(out, hair_face)
    return out


def rotate_start(pts: np.ndarray, k: int) -> np.ndarray:
    return np.roll(pts, -k, axis=0)


def sorted_points(pts) -> list:
    return sorted(map(tuple, np.asarray(pts).tolist()))


# ── Orientasi ──────────────────────────────────────
def test_signed_area_positive_means_clockwise_on_screen():
    """Persegi (0,0) → (10,0) → (10,10) → (0,10): kanan di sisi atas, lalu turun (y ke bawah) = searah jarum jam."""
    assert trk.signed_area(np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)) == 100.0
    assert trk.signed_area(np.array([[0, 0], [0, 10], [10, 10], [10, 0]], float)) == -100.0


CW_SHAPES = {
    "L_nonconvex": [(0, 0), (30, 0), (30, 10), (10, 10), (10, 30), (0, 30)],
    "tilted_square": [(20, 5), (35, 20), (20, 35), (5, 20)],
    "star": [(math.cos(a) * r + 50, math.sin(a) * r + 50)
             for a, r in ((k * math.pi / 5 - math.pi / 2, 30 if k % 2 == 0 else 12) for k in range(10))],
    "comb": [(0, 0), (40, 0), (40, 20), (32, 20), (32, 8), (24, 8), (24, 20), (16, 20), (16, 8), (8, 8), (8, 20),
             (0, 20)],
}


@pytest.mark.parametrize("name", CW_SHAPES)
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("start", [0, 7, 31])
def test_silhouette_becomes_clockwise_hole_counter_clockwise(name, reverse, start):
    """Daftar titik CW didefinisikan lewat tampilan (kanan di atas, turun di kanan) — bukan lewat luas bertanda."""
    cw = densify(CW_SHAPES[name])
    raw = rotate_start(cw[::-1] if reverse else cw, start)
    (sil,) = step(trk.Tracker(MAX_DIST), stroke("silhouette", raw))
    (hole,) = step(trk.Tracker(MAX_DIST), stroke("silhouette_hole", raw))
    ring = lambda pts: sorted_points(pts)  # noqa: E731
    assert ring(sil["points"]) == ring(cw) == ring(hole["points"])
    assert trk.signed_area(np.array(sil["points"])) > 0 > trk.signed_area(np.array(hole["points"]))
    # urutan = rotasi dari daftar CW (silhouette) / kebalikannya (lubang), bukan sekadar urutan acak
    s = [tuple(p) for p in sil["points"]]
    i = [tuple(p) for p in cw.tolist()].index(s[0])
    assert s == [tuple(p) for p in np.roll(cw, -i, axis=0).tolist()]
    h = [tuple(p) for p in hole["points"]]
    rev = cw[::-1]
    j = [tuple(p) for p in rev.tolist()].index(h[0])
    assert h == [tuple(p) for p in np.roll(rev, -j, axis=0).tolist()]


def test_shape_with_hole_each_oriented_by_own_type():
    outer = densify([(0, 0), (60, 0), (60, 60), (0, 60)])
    inner = densify([(20, 20), (20, 40), (40, 40), (40, 20)])            # CCW di layar
    tr = trk.Tracker(MAX_DIST)
    a, b = step(tr, stroke("silhouette", outer[::-1]), stroke("silhouette_hole", inner[::-1]))
    assert trk.signed_area(np.array(a["points"])) > 0 and trk.signed_area(np.array(b["points"])) < 0


def test_points_set_count_and_stroke_order_unchanged():
    tr = trk.Tracker(MAX_DIST)
    raw = [stroke("silhouette", rotate_start(blob((80, 80))[::-1], 17)),
           stroke("silhouette_hole", blob((80, 80))[::3] * 0.2 + 50),
           stroke("group_boundary", line((5, 5), (60, 40))), stroke("occlusion", line((5, 50), (60, 70)),
                                                                     closed=False, groups=["torso"], strength=0.12)]
    before = [(s["type"], sorted_points(s["points"])) for s in raw]
    out = step(tr, *raw)
    assert [(s["type"], sorted_points(s["points"])) for s in out] == before
    assert [s["track_id"] for s in out] == [1, 2, 3, 4]
    assert [tuple(s) for s in out] == [("track_id", "type", "closed", "groups", "points", "anchor")] * 2 + [
        ("track_id", "type", "closed", "groups", "points"), ("track_id", "type", "closed", "groups", "points",
                                                              "strength")]
    assert out[3]["strength"] == 0.12 and all(s["anchor"] == 0 for s in out[:2])


# ── Anchor ─────────────────────────────────────────
def test_anchor_follows_translation_and_small_rotation():
    """Kontur miring, tak-konveks, bergeser (1,3; 0,7) dan berputar 2° per frame; titik awal MENTAH acak + arah acak.
    Anchor melompat ≤ 4 px per frame (titik awal mentah sendiri melompat jauh lebih banyak)."""
    tr = trk.Tracker(MAX_DIST)
    jumps, raw_jumps, prev_raw, prev = [], [], None, None
    for f in range(14):
        pts = blob((80 + 1.3 * f, 90 + 0.7 * f), rot_deg=2.0 * f)
        raw = rotate_start(pts[::-1] if f % 3 == 0 else pts, (f * 53) % len(pts))
        (out,) = step(tr, stroke("silhouette", raw))
        a = np.array(out["points"][0])
        if prev is not None:
            jumps.append(float(np.linalg.norm(a - prev)))
            raw_jumps.append(float(np.linalg.norm(raw[0] - prev_raw)))
        prev, prev_raw = a, raw[0]
        assert out["track_id"] == 1
    assert max(jumps) <= 4.0, jumps
    assert max(raw_jumps) > 10.0


def test_anchor_stays_put_when_topmost_point_alternates_between_two_peaks():
    """Dua puncak 30 px berselisih; puncak tertinggi berganti tiap frame (selisih 1 px). Anchor mengikuti padanan,
    bukan titik tertinggi: tidak melompat antar puncak."""
    tr = trk.Tracker(MAX_DIST)
    prev, jumps = None, []
    for f in range(8):
        delta = 1.0 if f % 2 == 0 else -1.0
        poly = [(0, 60), (5, 10), (20, 30), (35, 10 + delta), (40, 60)]
        pts = densify(poly)
        (s,) = step(tr, stroke("silhouette", rotate_start(pts[::-1] if f % 3 == 0 else pts, 5 * f)))
        a = np.array(s["points"][0])
        if prev is not None:
            jumps.append(float(np.linalg.norm(a - prev)))
        prev = a
    assert max(jumps) <= 3.0, jumps


def test_new_silhouette_anchor_is_contour_point_nearest_topmost_hair_face_pixel_inside():
    rect = densify([(10, 10), (50, 10), (50, 60), (10, 60)])
    island = densify([(70, 20), (90, 20), (90, 40), (70, 40)])
    hf = np.zeros((80, 100), bool)
    hf[14, 25:30] = True                  # piksel tertinggi hair ∪ face: (x = 25, y = 14) → pusat (25.5, 14.5)
    hf[30:34, 40:44] = True
    hf[22, 75] = True                     # milik pulau → pulau memakai pikselnya sendiri
    a, b = step(trk.Tracker(MAX_DIST), stroke("silhouette", rect), stroke("silhouette", island), hair_face=hf)
    assert a["points"][0] == [25.0, 10.0]          # (25, 10) dan (26, 10) sama jauh dari (25.5, 14.5): x terkecil
    d =[np.hypot(x - 25.5, y - 14.5) for x, y in a["points"]]
    assert math.isclose(d[0], min(d), abs_tol=1e-9)
    # pulau: kontur terdekat ke (75.5, 22.5)
    d2 = [np.hypot(x - 75.5, y - 22.5) for x, y in b["points"]]
    assert math.isclose(d2[0], min(d2), abs_tol=1e-9)


def test_new_silhouette_without_hair_face_uses_topmost_point():
    rect = densify([(10, 10), (50, 10), (50, 60), (10, 60)])
    for mask in (None, np.zeros((80, 100), bool)):
        (s,) = step(trk.Tracker(MAX_DIST), stroke("silhouette", rotate_start(rect, 40)), hair_face=mask)
        assert s["points"][0] == [10.0, 10.0]          # y terkecil, seri → x terkecil


def test_new_hole_anchor_is_topmost_point():
    ring = densify([(20, 20), (20, 40), (40, 40), (40, 20)])
    (h,) = step(trk.Tracker(MAX_DIST), stroke("silhouette_hole", rotate_start(ring, 25)))
    assert h["points"][0] == [20.0, 20.0]


def test_nearest_index_tie_is_lowest_y_then_x_regardless_of_order():
    pts = np.array([[0.5, 4.5], [4.5, 2.5], [2.5, 8.5]])
    ref = np.array([2.5, 3.5])             # (0.5, 4.5) dan (4.5, 2.5) sama jauh (√5): y terkecil menang
    for perm in ([0, 1, 2], [1, 0, 2], [2, 1, 0]):
        sel = pts[perm]
        assert tuple(sel[trk.nearest_index(sel, ref)]) == (4.5, 2.5)
    row = np.array([[4.5, 2.5], [0.5, 2.5], [9.5, 9.5]])
    assert tuple(row[trk.nearest_index(row, np.array([2.5, 2.5]))]) == (0.5, 2.5)   # y sama → x terkecil


# ── Garis terbuka ──────────────────────────────────
def seg(center, length: float, angle_deg: float, n: int = 81) -> np.ndarray:
    a = math.radians(angle_deg)
    d = np.array([math.cos(a), math.sin(a)]) * length / 2
    c = np.asarray(center, float)
    return line(c - d, c + d, n)


def reversals(frames_out: list[np.ndarray]) -> int:
    n = 0
    for a, b in zip(frames_out, frames_out[1:]):
        same = np.linalg.norm(b[0] - a[0]) + np.linalg.norm(b[-1] - a[-1])
        flip = np.linalg.norm(b[0] - a[-1]) + np.linalg.norm(b[-1] - a[0])
        n += flip < same
    return n


@pytest.mark.parametrize("angles", [[44.0, 46.0] * 6, [-46.0, -44.0] * 6, [3.0, -3.0] * 6, [86.0, 94.0, 88.0, 92.0] * 3,
                                    [-2.0, 2.0, -1.0, 1.0] * 3])
def test_open_line_direction_never_flips_when_slope_changes_sign(angles):
    """Garis hampir horizontal (kemiringan ± berganti), dan garis di sekitar sudut pemotongan aturan statis (45° /
    vertikal): arah tidak membalik; aturan statis sendiri membalik di sekitar pemotongan."""
    tr = trk.Tracker(MAX_DIST)
    out, static = [], []
    for k, ang in enumerate(angles):
        pts = seg((60, 60), 80, ang)
        raw = pts[::-1] if k % 2 else pts                       # arah mentah acak
        (s,) = step(tr, stroke("group_boundary", raw))
        out.append(np.array(s["points"]))
        static.append(trk.static_open(raw.copy())[0])
    assert reversals(out) == 0
    assert max(np.linalg.norm(o[0] - out[0][0]) for o in out) < 8.0       # titik awal tetap di ujung yang sama
    if angles[0] in (-46.0, -44.0):       # garis naik ke kanan di sekitar -45°: ujung kiri ↔ ujung atas (kanan)
        starts = np.array(static)
        assert np.ptp(starts, axis=0).max() > 40.0              # aturan statis memang melompat ke ujung seberang


@pytest.mark.parametrize("angle", [-80, -60, -30, -5, 0, 5, 30, 60, 80])
@pytest.mark.parametrize("flip", [False, True])
def test_static_rule_axis_sign_locked_dominant_component_positive(angle, flip):
    """Track baru: |sudut| < 45° → titik awal = ujung kiri (x terkecil); selain itu ujung atas (y terkecil), apa pun
    arah mentah dan apa pun tanda vektor eigen."""
    pts = seg((60, 60), 80, angle)
    (s,) = step(trk.Tracker(MAX_DIST), stroke("group_boundary", pts[::-1] if flip else pts))
    first = np.array(s["points"][0])
    expect = pts[np.argmin(pts[:, 0])] if abs(angle) < 45 else pts[np.argmin(pts[:, 1])]
    assert np.allclose(first, expect)


def test_two_parallel_close_lines_keep_their_ids():
    tr = trk.Tracker(MAX_DIST)
    ids = {}
    for f in range(20):
        a = seg((50 + 2 * f, 50 + 2 * f), 80, 0.0) + 0 * f
        b = a + [0, 6]
        raw = [stroke("group_boundary", a), stroke("group_boundary", b)]
        out = step(tr, *raw)
        ids.setdefault("a", out[0]["track_id"])
        ids.setdefault("b", out[1]["track_id"])
        assert (out[0]["track_id"], out[1]["track_id"]) == (ids["a"], ids["b"]), f
    assert ids["a"] != ids["b"]


# ── track_id ───────────────────────────────────────
def hline(y: float, n: int = 101) -> np.ndarray:
    return line((0.0, y), (100.0, y), n)


def test_optimal_assignment_beats_greedy():
    """Biaya: A–P 2, A–Q 3, B–P 2,5, B–Q 7,5 (> ambang 6). Rakus: A–P dulu, B tanpa padanan. Optimal: A–Q + B–P."""
    tr = trk.Tracker(6.0)
    p, q = step(tr, stroke("group_boundary", hline(0.0)), stroke("group_boundary", hline(5.0)))
    a, b = step(tr, stroke("group_boundary", hline(2.0)), stroke("group_boundary", hline(-2.5)))
    assert (a["track_id"], b["track_id"]) == (q["track_id"], p["track_id"])
    assert tr.next_id == 3                                       # tidak ada id baru


def test_birth_death_ids_unique_integers_from_one_no_gap_tolerance():
    tr = trk.Tracker(MAX_DIST)
    f0 = step(tr, stroke("group_boundary", hline(10.0)))
    f1 = step(tr, stroke("group_boundary", hline(11.0)), stroke("group_boundary", hline(80.0)))
    f2 = step(tr, stroke("group_boundary", hline(80.5)))                     # strok pertama mati
    f3 = step(tr, stroke("group_boundary", hline(10.0)), stroke("group_boundary", hline(81.0)))   # kembali = id BARU
    assert [s["track_id"] for s in f0 + f1 + f2 + f3] == [1, 1, 2, 2, 3, 2]
    assert all(type(s["track_id"]) is int for s in f0 + f1 + f2 + f3)


def test_ids_continue_across_empty_frame_and_frame_without_strokes_is_valid():
    tr = trk.Tracker(MAX_DIST)
    assert step(tr) == []
    a = step(tr, stroke("group_boundary", hline(10.0)), stroke("group_boundary", hline(60.0)))
    assert step(tr) == []
    b = step(tr, stroke("group_boundary", hline(10.0)))
    assert [s["track_id"] for s in a] == [1, 2] and b[0]["track_id"] == 3


def test_different_type_or_groups_never_match():
    tr = trk.Tracker(MAX_DIST)
    step(tr, stroke("group_boundary", hline(10.0), groups=["left_arm", "torso"]))
    (s,) = step(tr, stroke("group_boundary", hline(10.0), groups=["torso", "right_arm"]))
    (o,) = step(tr, stroke("occlusion", hline(10.0), closed=False, groups=["torso"], strength=0.1))
    assert s["track_id"] == 2 and o["track_id"] == 3


def test_distance_at_threshold_is_not_a_match():
    for gap, matched in ((11.9, True), (12.0, False)):
        tr = trk.Tracker(12.0)
        step(tr, stroke("group_boundary", hline(10.0)))
        (s,) = step(tr, stroke("group_boundary", hline(10.0 + gap)))
        assert (s["track_id"] == 1) is matched


def test_split_one_part_keeps_old_id_tie_goes_to_first_stroke():
    tr = trk.Tracker(MAX_DIST)
    (whole,) = step(tr, stroke("group_boundary", hline(50.0, 101)))
    left, right = step(tr, stroke("group_boundary", line((0, 50), (49, 50), 50)),
                       stroke("group_boundary", line((51, 50), (100, 50), 50)))
    assert left["track_id"] == whole["track_id"] and right["track_id"] != whole["track_id"]


def test_merge_one_old_id_survives_tie_goes_to_first_previous():
    tr = trk.Tracker(MAX_DIST)
    left, right = step(tr, stroke("group_boundary", line((0, 50), (49, 50), 50)),
                       stroke("group_boundary", line((51, 50), (100, 50), 50)))
    (whole,) = step(tr, stroke("group_boundary", hline(50.0, 101)))
    assert whole["track_id"] == left["track_id"] != right["track_id"]


def test_assignment_is_deterministic_under_repeated_runs():
    def run():
        tr = trk.Tracker(MAX_DIST)
        out = []
        for f in range(6):
            out.append([s["track_id"] for s in step(
                tr, stroke("group_boundary", hline(20.0 + f)), stroke("group_boundary", hline(24.0 + f)),
                stroke("group_boundary", hline(28.0 + f)))])
        return out
    assert run() == run()


def test_assign_rounding_makes_equal_costs_tie_to_lowest_indices():
    cost = np.array([[3.0, 3.0], [3.0, 3.0]])
    assert trk.assign(cost, 12.0) == {0: 0, 1: 1}
    assert trk.assign(np.array([[3.0000001, 3.0]]), 12.0) == {0: 0}      # dibulatkan 6 desimal → seri → indeks kecil
    assert trk.assign(np.zeros((0, 3)), 12.0) == {} and trk.assign(np.zeros((2, 0)), 12.0) == {}


def test_load_state_sets_counter_from_max_id_and_matches_next_frame():
    tr = trk.Tracker(MAX_DIST)
    first = step(tr, stroke("group_boundary", hline(10.0)), stroke("group_boundary", hline(60.0)))
    tr2 = trk.Tracker(MAX_DIST)
    tr2.load(first)
    assert tr2.next_id == 3
    (s,) = step(tr2, stroke("group_boundary", hline(61.0)))
    assert s["track_id"] == 2


# ── Loop ───────────────────────────────────────────
@pytest.mark.parametrize("kind, groups, extra", [("group_boundary", ["hair", "face"], {}),
                                                 ("occlusion", ["torso"], {"strength": 0.2})])
def test_loop_keeps_end_equal_start_after_orientation_and_rotation(kind, groups, extra):
    tr = trk.Tracker(MAX_DIST)
    prev_anchor = None
    for f in range(6):
        ring = loop_pts((60 + 1.5 * f, 60 + f), ccw=(f % 2 == 0))               # arah mentah berganti
        body = rotate_start(ring[:-1], (f * 17) % (len(ring) - 1))
        raw = np.vstack([body, body[:1]])                                       # titik awal mentah berpindah
        (s,) = step(tr, stroke(kind, raw, closed=False, groups=groups, **extra))
        pts = np.array(s["points"])
        assert s["closed"] is False and s["points"][0] == s["points"][-1] and len(pts) == len(raw)
        assert sorted_points(pts[:-1]) == sorted_points(raw[:-1])
        assert trk.signed_area(pts[:-1]) > 0                                    # loop searah jarum jam
        assert "anchor" not in s and s["track_id"] == 1
        if prev_anchor is not None:
            assert np.linalg.norm(pts[0] - prev_anchor) <= 3.0
        prev_anchor = pts[0]


def test_loop_matching_open_previous_and_open_matching_loop_do_not_crash():
    tr = trk.Tracker(MAX_DIST)
    ring = loop_pts((60, 60))
    step(tr, stroke("group_boundary", ring[:-3], closed=False, groups=["hair", "face"]))     # hampir loop, terbuka
    (a,) = step(tr, stroke("group_boundary", ring, closed=False, groups=["hair", "face"]))
    (b,) = step(tr, stroke("group_boundary", ring[:-3], closed=False, groups=["hair", "face"]))
    assert a["points"][0] == a["points"][-1] and b["points"][0] != b["points"][-1]
    assert a["track_id"] == b["track_id"] == 1


# ── Silhouette terbesar mewarisi id (pendekatan Y, belum dipakai produksi) ──
def test_inherit_main_silhouette_function_replaces_conflicting_pairs():
    big = trk.make_item(stroke("silhouette", densify([(0, 0), (100, 0), (100, 100), (0, 100)])), 1)
    small = trk.make_item(stroke("silhouette", densify([(300, 0), (320, 0), (320, 20), (300, 20)])), 2)
    cur_small = trk.make_item(stroke("silhouette", densify([(300, 0), (320, 0), (320, 20), (300, 20)])))
    cur_big = trk.make_item(stroke("silhouette", densify([(0, 0), (60, 0), (60, 100), (0, 100)])))
    assert trk.inherit_main_silhouette({}, [cur_small, cur_big], [big, small]) == {1: 0}
    assert trk.inherit_main_silhouette({0: 1, 1: 0}, [cur_small, cur_big], [big, small]) == {1: 0, 0: 1}
    assert trk.inherit_main_silhouette({0: 0}, [cur_small], [big]) == {0: 0}
    assert trk.inherit_main_silhouette({}, [], [big]) == {}


def test_tracker_inherit_main_keeps_id_when_arm_detaches_beyond_threshold():
    body = densify([(0, 0), (100, 0), (100, 100), (0, 100)])
    shrunk = densify([(0, 0), (30, 0), (30, 100), (0, 100)])             # tinggal 30% → chamfer jauh > 12
    arm = densify([(60, 0), (100, 0), (100, 100), (60, 100)])
    for inherit, expect in ((False, False), (True, True)):
        tr = trk.Tracker(MAX_DIST, inherit_main=inherit)
        (a,) = step(tr, stroke("silhouette", body))
        b, c = step(tr, stroke("silhouette", shrunk), stroke("silhouette", arm))
        assert ((b["track_id"] == a["track_id"]) or (c["track_id"] == a["track_id"])) is expect


def test_track_module_does_not_import_torch():
    code = "import sys, rotoscope.track; sys.exit(1 if 'torch' in sys.modules else 0)"
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0
