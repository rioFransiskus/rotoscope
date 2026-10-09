"""Test vectorize.py (T-201a): peta grup sintetis, tanpa model/GPU. Satu test memakai data nyata (ditandai)."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import skeleton_metrics as sm

from rotoscope import stabilize as stb
from rotoscope import stage_common
from rotoscope import vectorize as vec
from rotoscope.config import load_pipeline, section_hash
from rotoscope.stage_common import StageError

H, W = 24, 32
N_FRAMES = 4
HAIR, FACE, TORSO, LARM = 1, 2, 3, 4          # id grup default (urutan YAML)
NAMES = tuple(g for g, _ in load_pipeline().groups)
PARAMS = {"min_region_area": 800, "min_hole_area": 200, "line_min_px": 5, "min_stroke_px": 6}
DEPTH_DEFAULTS = {"depth_lines.blur_sigma": 1.0, "depth_lines.hi_pct": 95.0, "depth_lines.lo_pct": 90.0,
                  "depth_lines.erode_px": 5, "depth_lines.min_dist_px": 7.0, "depth_lines.min_len_px": 30.0,
                  "depth_lines.min_dist_low_px": 0.0, "depth_lines.exclude_groups": []}
SMALL = {**PARAMS, "min_region_area": 1, "min_hole_area": 1}
REAL_CLIP = Path("work/clips/test_short")


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setattr(stage_common, "REPLACE_DELAY_S", 0)


def blank(h: int = H, w: int = W) -> np.ndarray:
    return np.zeros((h, w), np.uint8)


def pts_of(stroke: dict) -> set[tuple[float, float]]:
    return {tuple(p) for p in stroke["points"]}


def by_type(strokes: list[dict], kind: str) -> list[dict]:
    return [s for s in strokes if s["type"] == kind]


def run_gmap(g: np.ndarray, **over) -> tuple[list[dict], dict]:
    return vec.vectorize_gmap(g, NAMES, {**PARAMS, **over})


# ── Siluet + lubang ────────────────────────────────
def test_square_with_hole_one_silhouette_one_hole():
    g = blank(30, 30)
    g[5:25, 5:25] = TORSO               # 20×20
    g[12:17, 12:17] = 0                 # lubang 5×5
    strokes, st = run_gmap(g, min_region_area=1, min_hole_area=1)
    sil, hole = by_type(strokes, "silhouette"), by_type(strokes, "silhouette_hole")
    assert (len(sil), len(hole), len(by_type(strokes, "group_boundary"))) == (1, 1, 0)
    assert sil[0]["closed"] is True and sil[0]["groups"] == ["background"]
    assert len(sil[0]["points"]) == 4 * 19          # keliling piksel kontur 20×20; titik awal tidak diulang
    assert len(hole[0]["points"]) == 4 * 5          # piksel foreground tetangga-4 lubang 5×5 (4 sudut tidak ikut)
    assert min(p[0] for p in sil[0]["points"]) == 5.5 and max(p[0] for p in sil[0]["points"]) == 24.5
    assert min(p[1] for p in hole[0]["points"]) == 11.5 and max(p[1] for p in hole[0]["points"]) == 17.5
    assert (st["regions_raw"], st["holes_raw"]) == (1, 1)


@pytest.mark.parametrize("h, kept", [(40, True), (39, False)])
def test_region_area_is_pixel_count(h, kept):
    """20×40 = 800 px lolos min_region_area=800 (cv2.contourArea = 741 akan membuangnya); 20×39 = 780 dibuang."""
    g = blank(60, 60)
    g[5:5 + h, 5:25] = TORSO
    strokes, st = run_gmap(g)
    assert len(by_type(strokes, "silhouette")) == (1 if kept else 0)
    assert st["regions_dropped"] == (0 if kept else 1)


@pytest.mark.parametrize("hole_w, kept", [(15, True), (14, False)])
def test_hole_area_is_pixel_count(hole_w, kept):
    """Lubang 15×14 = 210 lolos min_hole_area=200; 14×14 = 196 dibuang (dilaporkan di statistik)."""
    g = blank(60, 70)
    g[2:52, 2:62] = TORSO
    g[10:24, 10:10 + hole_w] = 0
    strokes, st = run_gmap(g)
    assert len(by_type(strokes, "silhouette_hole")) == (1 if kept else 0)
    assert st["holes_dropped_areas"] == ([] if kept else [196])


def test_hole_of_dropped_region_is_dropped_too():
    g = blank(30, 30)
    g[5:15, 5:15] = TORSO                # 100 px − lubang 16 = 84 px foreground
    g[8:12, 8:12] = 0
    strokes, st = run_gmap(g, min_region_area=100, min_hole_area=1)
    assert strokes == [] and st["regions_dropped"] == 1 and st["holes_dropped_parent"] == 1


def test_nested_ring_hole_area_exact():
    """Pulau yang punya lubang di dalam lubang: luas lubang luar = piksel background-nya saja."""
    g = blank(60, 60)
    g[2:46, 2:46] = TORSO                # 44×44
    g[5:43, 5:43] = 0                    # lubang 38×38 = 1444
    g[15:25, 15:25] = FACE               # pulau 10×10 = 100 px
    g[18:22, 18:22] = 0                  # lubang pulau 4×4 = 16
    area_outer_hole = 38 * 38 - 100
    strokes, st = run_gmap(g, min_region_area=1, min_hole_area=area_outer_hole)
    assert len(by_type(strokes, "silhouette")) == 2 and len(by_type(strokes, "silhouette_hole")) == 1
    assert st["holes_dropped_areas"] == [16]
    strokes, st = run_gmap(g, min_region_area=1, min_hole_area=area_outer_hole + 1)
    assert len(by_type(strokes, "silhouette_hole")) == 0 and sorted(st["holes_dropped_areas"]) == [16, area_outer_hole]


def test_foreground_at_frame_edge_contour_stays_closed():
    g = blank(20, 20)
    g[10:20, 4:14] = TORSO               # menempel tepi bawah
    strokes, st = run_gmap(g, min_region_area=1)
    (sil,) = by_type(strokes, "silhouette")
    assert sil["closed"] is True and sil["points"][0] != sil["points"][-1]
    ys = [p[1] for p in sil["points"]]
    assert max(ys) == 19.5 and all(0 <= p[0] <= 20 and 0 <= p[1] <= 20 for p in sil["points"])
    assert sum(y == 19.5 for y in ys) == 10          # run titik di baris tepi bawah (10 piksel)
    assert st["edge_points"] == 10


def test_frame_filling_corner_edges_counted():
    g = blank(10, 10)
    g[0:10, 0:10] = TORSO                # seluruh frame: kontur = bingkai
    strokes, st = run_gmap(g, min_region_area=1)
    assert len(strokes) == 1 and len(strokes[0]["points"]) == 36 and st["edge_points"] == 36


def test_empty_frame_valid_document():
    strokes, st = run_gmap(blank())
    assert strokes == [] and st["n_silhouette"] == 0
    doc = json.loads(vec.frame_document(3, W, H, {"seg_model": "0.8b"}, strokes))
    assert doc["strokes"] == [] and doc["frame_index"] == 3


def test_diagonal_contact_is_one_region_and_short_band_dropped():
    g = blank(20, 20)
    g[0:10, 0:10] = HAIR
    g[10:20, 10:20] = FACE               # hanya bersentuhan diagonal → 8-arah: 1 komponen
    strokes, st = run_gmap(g, min_region_area=1)
    assert len(by_type(strokes, "silhouette")) == 1 and st["regions_raw"] == 1
    assert by_type(strokes, "group_boundary") == []     # band 2 px < line_min_px


def test_stroke_keys_exact_no_track_id_or_anchor():
    g = blank()
    g[4:20, 4:12] = TORSO
    g[4:20, 12:22] = LARM
    g[8:12, 6:10] = 0
    strokes, _ = run_gmap(g, min_region_area=1, min_hole_area=1)
    assert {s["type"] for s in strokes} == {"silhouette", "silhouette_hole", "group_boundary"}
    for s in strokes:
        assert set(s) == set(vec.STROKE_KEYS)
        assert "track_id" not in s and "anchor" not in s and "strength" not in s


# ── Batas grup ─────────────────────────────────────
def test_two_adjacent_groups_one_boundary():
    g = blank(20, 30)
    g[5:15, 5:15] = HAIR
    g[5:15, 15:25] = FACE
    strokes, st = run_gmap(g, min_region_area=1)
    (b,) = by_type(strokes, "group_boundary")
    assert b["closed"] is False and b["groups"] == [NAMES[HAIR - 1], NAMES[FACE - 1]]
    xs = {p[0] for p in b["points"]}
    assert xs <= {14.5, 15.5}                        # skeleton band 2 px: ≤ 0.5 px dari batas (x = 15.0)
    ys = [p[1] for p in b["points"]]
    assert ys[0] == min(ys) and min(ys) <= 6.5 and max(ys) >= 13.5   # awal = ujung (y, x) terkecil; seluruh batas
    assert len(b["points"]) >= 8 and st["loops"] == 0


def test_three_groups_meeting_three_boundaries_ends_near_junction():
    g = blank(20, 20)
    g[0:10, 0:10] = HAIR
    g[0:10, 10:20] = FACE
    g[10:20, :] = TORSO
    strokes, _ = run_gmap(g, min_region_area=1)
    bnd = by_type(strokes, "group_boundary")
    assert [b["groups"] for b in bnd] == [[NAMES[0], NAMES[1]], [NAMES[0], NAMES[2]], [NAMES[1], NAMES[2]]]
    for b in bnd:
        ends = [b["points"][0], b["points"][-1]]
        assert any(abs(p[0] - 10.0) <= 2.5 and abs(p[1] - 10.0) <= 2.5 for p in ends)   # satu ujung di titik temu


def test_enclosed_group_boundary_is_open_loop_with_end_equal_start():
    g = blank(40, 40)
    g[2:38, 2:38] = TORSO
    g[12:26, 12:26] = FACE               # dikelilingi torso: tidak menyentuh background
    strokes, st = run_gmap(g, min_region_area=1)
    (b,) = by_type(strokes, "group_boundary")
    assert b["closed"] is False and len(b["points"]) > 20
    assert b["points"][0] == b["points"][-1] and st["loops"] == 1
    assert (b["points"][0][1], b["points"][0][0]) == min((p[1], p[0]) for p in b["points"])   # mulai (y, x) terkecil


def test_boundary_short_component_removed_before_thinning():
    g = blank(20, 20)
    g[5:15, 5:15] = TORSO
    g[5:7, 15:17] = LARM                 # kontak 2 px → band < line_min_px
    strokes, _ = run_gmap(g, min_region_area=1)
    assert by_type(strokes, "group_boundary") == []


def _line_with_spur(n: int = 30, y: int = 10, x: int = 15, spur: int = 2) -> np.ndarray:
    sk = np.zeros((20, n), np.uint8)
    sk[y, :] = 1
    sk[y - spur:y, x] = 1
    return sk


def test_spur_dropped_and_line_rejoined():
    polys, st = vec.trace_skeleton(_line_with_spur(), min_stroke=6)
    assert len(polys) == 1 and st["spurs_dropped"] == 1 and st["loops"] == 0
    assert polys[0][0] == (10, 0) and polys[0][-1] == (10, 29) and len(polys[0]) >= 29


def test_staircase_is_not_junction_single_polyline():
    """Tangga 4-arah (bentuk skeleton Zhang-Suen): piksel bertetangga 3 BUKAN junction → satu polyline utuh."""
    sk = np.zeros((40, 40), np.uint8)
    for i in range(12):
        sk[5 + i, 5 + i] = 1             # diagonal
        sk[5 + i, 6 + i] = 1             # + sambungan 4-arah: tiap piksel punya 3 tetangga
    assert not vec.junction_mask(sk).any()
    polys, st = vec.trace_skeleton(sk, min_stroke=6)
    assert len(polys) == 1 and st["spurs_dropped"] == 0 and st["short_dropped"] == 0
    assert len(polys[0]) >= 12           # panjang diagonal 12 piksel utuh (sudut siku redundan dihapus)


def test_t_junction_is_junction():
    sk = np.zeros((20, 30), np.uint8)
    sk[10, 2:28] = 1
    sk[3:10, 15] = 1
    assert vec.junction_mask(sk).sum() == 1
    polys, _ = vec.trace_skeleton(sk, min_stroke=6)
    assert len(polys) == 3               # tiga cabang panjang tetap terpisah


def test_long_branch_kept_as_separate_stroke():
    polys, st = vec.trace_skeleton(_line_with_spur(spur=8), min_stroke=6)
    assert len(polys) == 3 and st["spurs_dropped"] == 0


def test_short_path_dropped_and_ring_traced():
    sk = np.zeros((30, 30), np.uint8)
    sk[2, 2:5] = 1                       # 3 px
    sk[10:20, 10] = 1                    # cincin 10×10 (kontur piksel)
    sk[10:20, 19] = 1
    sk[10, 10:20] = 1
    sk[19, 10:20] = 1
    polys, st = vec.trace_skeleton(sk, min_stroke=6)
    assert st["short_dropped"] == 1 and st["loops"] == 1 and len(polys) == 1
    assert polys[0][0] == polys[0][-1] == min(polys[0])      # loop mulai dari titik (y, x) terkecil


# ── Pelacak skeleton: bentuk dasar (semua gagal pada detektor lama "≥ 3 tetangga") ──
def thin(mask: np.ndarray) -> np.ndarray:
    """Skeleton Zhang-Suen nyata (bertangga) dari bentuk tebal — sama dengan yang dihasilkan stage [4]."""
    return vec.thin_mask(mask)


def thick(draw) -> np.ndarray:
    img = np.zeros((70, 90), np.uint8)
    draw(img)
    return thin(img > 0)


def shared_points(polys: list[list[tuple[int, int]]]) -> dict[tuple[int, int], int]:
    """Piksel yang dipakai > 1 strok → jumlah strok yang memakainya (penutup loop tidak dihitung dua kali)."""
    use: dict[tuple[int, int], int] = {}
    for p in polys:
        body = p[:-1] if len(p) > 3 and p[0] == p[-1] else p
        for q in set(body):
            use[q] = use.get(q, 0) + 1
    return {q: n for q, n in use.items() if n > 1}


def junction_region(sk: np.ndarray) -> set[tuple[int, int]]:
    """Piksel klaster junction: crossing number ≥ 3 + tetangganya di skeleton yang sudah dirapikan."""
    s = vec.prune_redundant(sk)
    region = cv2.dilate(vec.junction_mask(s).astype(np.uint8), vec.KERNEL_3X3).astype(bool) & s
    return {tuple(p) for p in np.argwhere(region).tolist()}


def assert_shared_only_at_junctions(polys, sk, *, meeting: dict[int, int]) -> None:
    """Titik yang dipakai bersama hanya di klaster junction; `meeting` = {jumlah strok: jumlah titik temu} —
    mis. {3: 1} = tepat satu titik yang dipakai ketiga strok sebuah T."""
    shared = shared_points(polys)
    assert set(shared) <= junction_region(sk), "titik bersama di luar klaster junction"
    counts: dict[int, int] = {}
    for n in shared.values():
        counts[n] = counts.get(n, 0) + 1
    for n, want in meeting.items():
        assert counts.get(n, 0) == want, f"titik temu {n} strok: {counts.get(n, 0)} ≠ {want}"


def assert_clean(polys: list[list[tuple[int, int]]]) -> None:
    """Tanpa titik ganda dalam satu strok dan tanpa loncatan (> 1 piksel antar titik berurutan)."""
    for p in polys:
        body = p[:-1] if len(p) > 3 and p[0] == p[-1] else p
        assert len(body) == len(set(body)), "titik berulang dalam satu strok"
        assert all(max(abs(a[0] - b[0]), abs(a[1] - b[1])) <= 1 for a, b in zip(p, p[1:])), "loncatan"


def monotone(values: list[float]) -> bool:
    d = np.diff(values)
    return bool((d >= 0).all() or (d <= 0).all())


def single_boundary(g: np.ndarray) -> dict:
    strokes, st = run_gmap(g, min_region_area=1)
    bnd = by_type(strokes, "group_boundary")
    assert len(bnd) == 1 and st["loops"] == 0, [len(s["points"]) for s in bnd]
    assert sm.integrity(bnd) == {"repeats": 0, "jumps": 0}
    xs, ys = [p[0] for p in bnd[0]["points"]], [p[1] for p in bnd[0]["points"]]
    assert monotone(xs) and monotone(ys)
    return bnd[0]


def test_diagonal_45_degree_boundary_is_one_monotone_stroke():
    yy, xx = np.mgrid[0:50, 0:50]
    g = np.where(xx < yy, HAIR, FACE).astype(np.uint8)
    b = single_boundary(g)
    assert 45 <= len(b["points"]) <= 52


def test_four_connected_staircase_boundary_is_one_monotone_stroke():
    yy, xx = np.mgrid[0:50, 0:70]
    g = np.where(yy > xx // 3 + 4, HAIR, FACE).astype(np.uint8)              # tangga landai: 3 piksel per anak tangga
    b = single_boundary(g)
    assert len(b["points"]) >= 55


def stair(p0: tuple[int, int], p1: tuple[int, int]) -> list[tuple[int, int]]:
    """Garis 4-arah (bertangga; tiap langkah satu piksel ortogonal) dari p0 ke p1 — bentuk skeleton Zhang-Suen
    yang membuat detektor lama (≥ 3 tetangga) menganggap hampir semua piksel junction."""
    (y, x), (y1, x1) = p0, p1
    ay, ax, sy, sx = abs(y1 - y), abs(x1 - x), np.sign(y1 - y), np.sign(x1 - x)
    ix = iy = 0
    pts = [(y, x)]
    while ix < ax or iy < ay:
        if iy >= ay or (ix < ax and (1 + 2 * ix) * ay < (1 + 2 * iy) * ax):
            x, ix = x + int(sx), ix + 1
        else:
            y, iy = y + int(sy), iy + 1
        pts.append((y, x))
    return pts


def stairs(*segments: tuple[tuple[int, int], tuple[int, int]], shape: tuple[int, int] = (70, 90)) -> np.ndarray:
    sk = np.zeros(shape, bool)
    for p0, p1 in segments:
        for y, x in stair(p0, p1):
            sk[y, x] = True
    return sk


def test_stair_helper_is_four_connected():
    pts = stair((5, 5), (20, 35))
    assert all(abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1 for a, b in zip(pts, pts[1:])) and len(pts) == 46


def test_circle_arc_boundary_is_one_monotone_stroke():
    yy, xx = np.mgrid[0:60, 0:60]
    g = np.where(xx ** 2 + yy ** 2 < 40 ** 2, FACE, HAIR).astype(np.uint8)   # busur seperempat lingkaran r = 40
    b = single_boundary(g)
    assert 55 <= len(b["points"]) <= 75
    # busur 4-arah bertangga (bentuk skeleton nyata): satu strok, monoton, tanpa titik ganda / loncatan
    ang = np.linspace(0.5, 1.07, 15)           # 29°–61°: anak tangga ≤ 2 piksel di sepanjang busur
    ring = [(int(round(60 - 45 * np.sin(t))), int(round(10 + 45 * np.cos(t)))) for t in ang]
    arc = stairs(*zip(ring, ring[1:]), shape=(70, 70))
    polys, st = vec.trace_skeleton(arc, min_stroke=6)
    assert len(polys) == 1 and st["loops"] == 0 and st["spurs_dropped"] == 0
    assert_clean(polys)
    assert monotone([p[0] for p in polys[0]]) and monotone([p[1] for p in polys[0]])
    assert len(polys[0]) >= 20                 # panjang sumbu busur ≈ 21 piksel setelah sudut siku dipangkas


def test_t_junction_three_strokes_share_one_point():
    thinned = thick(lambda im: (cv2.line(im, (8, 40), (80, 40), 255, 5), cv2.line(im, (44, 40), (44, 8), 255, 5)))
    stepped = stairs(((35, 45), (8, 15)), ((35, 45), (8, 75)), ((35, 45), (65, 45)))
    for sk in (thinned, stepped):
        polys, st = vec.trace_skeleton(sk, min_stroke=6)
        assert len(polys) == 3 and st["loops"] == 0
        assert_clean(polys)
        assert_shared_only_at_junctions(polys, sk, meeting={3: 1})   # satu titik temu yang dipakai ketiga strok


def test_x_crossing_four_strokes_share_one_point():
    thinned = thick(lambda im: (cv2.line(im, (10, 10), (80, 60), 255, 5), cv2.line(im, (10, 60), (80, 10), 255, 5)))
    stepped = stairs(((10, 15), (35, 45)), ((35, 45), (60, 75)), ((10, 75), (35, 45)), ((35, 45), (60, 15)))
    for sk in (thinned, stepped):
        polys, _ = vec.trace_skeleton(sk, min_stroke=6)
        assert len(polys) == 4
        assert_clean(polys)
        assert_shared_only_at_junctions(polys, sk, meeting={4: 1})   # persilangan: satu titik temu 4 strok


def test_closed_ring_is_one_loop_without_repeats():
    thinned = thick(lambda im: cv2.circle(im, (45, 35), 25, 255, 5))
    top, right, bottom, left = (8, 45), (35, 75), (62, 45), (35, 15)
    diamond = stairs((top, right), (right, bottom), (bottom, left), (left, top))
    for sk, min_len in ((thinned, 140), (diamond, 100)):
        polys, st = vec.trace_skeleton(sk, min_stroke=6)
        assert len(polys) == 1 and st["loops"] == 1 and polys[0][0] == polys[0][-1]
        assert_clean(polys)
        assert shared_points(polys) == {}
        assert len(polys[0]) >= min_len


def test_two_close_t_junctions_five_strokes_two_shared_points():
    def draw(im):
        cv2.line(im, (5, 45), (85, 45), 255, 5)
        cv2.line(im, (30, 45), (30, 8), 255, 5)
        cv2.line(im, (52, 45), (52, 8), 255, 5)
    stepped = stairs(((45, 5), (45, 85)), ((45, 30), (8, 20)), ((45, 52), (8, 62)))
    for sk in (thick(draw), stepped):
        polys, st = vec.trace_skeleton(sk, min_stroke=6)
        assert len(polys) == 5 and st["loops"] == 0
        assert_clean(polys)
        assert_shared_only_at_junctions(polys, sk, meeting={3: 2})   # dua titik temu, masing-masing 3 strok


def test_prune_keeps_endpoints_and_plus_center():
    stairs = np.zeros((30, 30), np.uint8)
    for i in range(10):
        stairs[5 + i, 5 + i] = stairs[5 + i, 6 + i] = 1
    pruned = vec.prune_redundant(stairs)
    assert int(pruned.sum()) == 11 and pruned[5, 5] and pruned[14, 15]       # 20 → 11: sudut siku dihapus, ujung utuh
    assert (vec.prune_redundant(pruned) == pruned).all()                      # idempoten
    plus = np.zeros((9, 9), np.uint8)
    plus[4, :] = plus[:, 4] = 1
    assert vec.prune_redundant(plus)[4, 4]                           # pusat '+' (crossing number 4) tidak dihapus


# ── Konvensi koordinat + urutan ────────────────────
def test_points_are_pixel_centers_one_decimal():
    g = blank()
    g[4:20, 4:12] = TORSO
    g[4:20, 12:22] = LARM
    strokes, _ = run_gmap(g, min_region_area=1)
    for s in strokes:
        for x, y in s["points"]:
            assert x % 1 == 0.5 and y % 1 == 0.5
            assert round(x, vec.COORD_DECIMALS) == x and round(y, vec.COORD_DECIMALS) == y


def test_stroke_order_fixed_by_type_pair_position():
    g = blank(50, 50)
    g[2:22, 2:22] = TORSO
    g[26:46, 26:46] = LARM
    g[30:34, 30:34] = 0
    g[2:22, 22:30] = FACE
    strokes, _ = run_gmap(g, min_region_area=1, min_hole_area=1)
    kinds = [s["type"] for s in strokes]
    assert kinds == sorted(kinds, key=vec.STROKE_TYPES.index)
    sil = by_type(strokes, "silhouette")
    assert [min((p[1], p[0]) for p in s["points"]) for s in sil] == sorted(min((p[1], p[0]) for p in s["points"]) for s in sil)
    assert vec.vectorize_gmap(g, NAMES, SMALL)[0] == strokes


# ── Klip sintetis ──────────────────────────────────
def subject_groups(i: int) -> np.ndarray:
    g = blank()
    g[6:20, 4 + i % 2:12 + i % 2] = TORSO
    g[8:16, 12 + i % 2:16 + i % 2] = LARM
    g[2:6, 6:10] = FACE
    return g


def make_work(tmp_path: Path, n: int = N_FRAMES, h: int = H, w: int = W) -> Path:
    work = tmp_path / "work"
    (work / "stable" / "groups").mkdir(parents=True)
    (work / "stable" / "depth_smooth").mkdir(parents=True)
    (work / "meta.json").write_text(json.dumps({"source_path": "C:/clips/a.mp4", "frame_count": n,
                                                "working_width": w, "working_height": h,
                                                "frame_index_start": 0}), encoding="utf-8")
    cfg = cfg_for(work)
    (work / "stable" / "manifest.json").write_text(json.dumps(
        {"stage": "stabilize", "stabilize_hash": "s" * 64, "groups_hash": section_hash(cfg, "groups"),
         "seg": {"model": "0.8b"}, "frame_size": {"width": w, "height": h},
         "clip": stage_common.clip_identity(work), "created_utc": "t0"}), encoding="utf-8")
    for i in range(n):
        stb.write_groups(work / "stable" / "groups" / f"frame_{i:05d}.png", subject_groups(i))
        stb.write_depth_smooth(work / "stable" / "depth_smooth" / f"frame_{i:05d}.npy", subject_depth(i, h, w))
    return work


def subject_depth(i: int, h: int = H, w: int = W) -> np.ndarray:
    """depth_smooth sintetis float16: tangga kedalaman + sedikit variasi per frame (ambang klip bukan nol)."""
    xx = np.arange(w)[None, :] + 0 * np.arange(h)[:, None]
    return (np.where(xx < 10 + i, 0.0, 1.5) + 0.01 * i * (xx % 3)).astype(np.float16)


TRACK_DEFAULTS = {"track.max_match_dist_px": 16.0}
CLIP_PARAMS = {**PARAMS, "min_region_area": 50, **DEPTH_DEFAULTS, **TRACK_DEFAULTS,   # subjek sintetis 24×32 << 800 px
               "depth_lines.min_dist_low_px": 2.0, "depth_lines.exclude_groups": ["hair"]}      # default T-305b


def cfg_for(work: Path, **overrides):
    if "groups" in overrides:                           # grup tanpa hair → default exclude_groups [hair] (T-305b) tidak berlaku
        overrides.setdefault("vectorize.depth_lines.exclude_groups", [])
    return load_pipeline(overrides={"paths.work_dir": str(work), "vectorize.min_region_area": 50, **overrides})


def quiet(_msg: str) -> None:
    pass


def merged_groups() -> dict:
    """Grup default dengan hair digabung ke face (hash grup berbeda; semua kelas tetap tercantum)."""
    groups = {name: list(cls) for name, cls in load_pipeline().groups}
    groups["face"] = groups["face"] + groups.pop("hair")
    return groups


def frame_hashes(work: Path) -> dict[str, str]:
    """sha256 contours/frame_*.json SAJA (manifest.json + frames.jsonl memuat waktu → tidak deterministik)."""
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((work / "contours").glob("frame_*.json"))}


def edit_stable_manifest(work: Path, **changes) -> None:
    p = work / "stable" / "manifest.json"
    m = json.loads(p.read_text(encoding="utf-8"))
    m.update(changes)
    p.write_text(json.dumps(m), encoding="utf-8")


def test_run_writes_outputs_manifest_and_log(tmp_path):
    work = make_work(tmp_path)
    run = vec.run_vectorize(cfg_for(work), log=quiet)
    assert run["processed"] == N_FRAMES and run["skipped"] == 0
    assert len(frame_hashes(work)) == N_FRAMES
    m = json.loads((work / "contours" / "manifest.json").read_text(encoding="utf-8"))
    assert m["contract"] == "T-305b" and m["stroke_types"] == list(vec.STROKE_TYPES)
    assert m["stroke_types"][-1] == "occlusion" and m["pending"] == []
    stats = json.loads((work / "contours" / "clip_stats.json").read_text(encoding="utf-8"))
    assert m["depth_thresholds"] == {"t_high": stats["t_high"], "t_low": stats["t_low"]} and m["clip_stats"]
    assert m["vectorize"] == CLIP_PARAMS and m["seg_model"] == "0.8b" and m["stable_created_utc"] == "t0"
    assert m["clip"] == stage_common.clip_identity(work) and m["frame_size"] == {"width": W, "height": H}
    d = json.loads((work / "contours" / "frame_00002.json").read_text(encoding="utf-8"))
    assert d["frame_index"] == 2 and (d["width"], d["height"]) == (W, H)
    assert d["source"] == {"seg_model": "0.8b", "groups_hash": m["groups_hash"], "stabilize_hash": "s" * 64,
                           "vectorize_hash": m["vectorize_hash"]}
    assert "created_utc" not in d and "time_utc" not in json.dumps(d)
    assert {s["type"] for s in d["strokes"]} >= {"silhouette", "group_boundary"}
    assert len(stage_common.last_frame_records(work / "contours" / "frames.jsonl")) == N_FRAMES


def test_rerun_from_scratch_byte_identical(tmp_path):
    work = make_work(tmp_path)
    vec.run_vectorize(cfg_for(work), log=quiet)
    first = frame_hashes(work)
    vec.run_vectorize(cfg_for(work), restart=True, log=quiet)
    assert frame_hashes(work) == first and len(first) == N_FRAMES


def test_resume_limit_then_full_skips_done(tmp_path):
    work = make_work(tmp_path)
    assert vec.run_vectorize(cfg_for(work), limit=2, log=quiet)["processed"] == 2
    before = {p.name: p.stat().st_mtime_ns for p in (work / "contours").glob("frame_*.json")}
    run = vec.run_vectorize(cfg_for(work), log=quiet)
    assert (run["skipped"], run["processed"]) == (2, N_FRAMES - 2)
    assert all((work / "contours" / n).stat().st_mtime_ns == t for n, t in before.items())
    again = vec.run_vectorize(cfg_for(work), log=quiet)
    assert (again["skipped"], again["processed"]) == (N_FRAMES, 0)


def test_corrupt_frame_redone(tmp_path):
    work = make_work(tmp_path)
    vec.run_vectorize(cfg_for(work), log=quiet)
    good = frame_hashes(work)
    (work / "contours" / "frame_00001.json").write_text("{potong", encoding="utf-8")
    run = vec.run_vectorize(cfg_for(work), log=quiet)
    assert run["processed"] == 1 and frame_hashes(work) == good


def test_frame_valid_rejects_other_source_and_index(tmp_path):
    work = make_work(tmp_path)
    vec.run_vectorize(cfg_for(work), log=quiet)
    clip = vec.load_clip(work)
    path = work / "contours" / "frame_00000.json"
    src = json.loads(path.read_text(encoding="utf-8"))["source"]
    assert vec.frame_valid(path, 0, clip, src)
    assert not vec.frame_valid(path, 1, clip, src)
    assert not vec.frame_valid(path, 0, clip, {**src, "vectorize_hash": "x"})
    assert not vec.frame_valid(work / "contours" / "tidak_ada.json", 0, clip, src)


@pytest.mark.parametrize("change", ["param", "groups", "stable_recomputed", "stabilize_hash", "seg_model"])
def test_stale_manifest_recomputed_with_warning(tmp_path, change):
    work = make_work(tmp_path)
    vec.run_vectorize(cfg_for(work), log=quiet)
    cfg, expect = cfg_for(work), None
    if change == "param":
        cfg, expect = cfg_for(work, **{"vectorize.min_region_area": 60}), "vectorize.min_region_area: 50 → 60"
    elif change == "groups":
        cfg = cfg_for(work, groups=merged_groups())
        edit_stable_manifest(work, groups_hash=section_hash(cfg, "groups"))
        expect = "groups_hash"
    elif change == "stable_recomputed":
        edit_stable_manifest(work, created_utc="t1")          # [3] dihitung ulang (mis. setelah --restart [2])
        expect = "stable_created_utc: 't0' → 't1'"
    elif change == "stabilize_hash":
        edit_stable_manifest(work, stabilize_hash="z" * 64)
        expect = "stabilize_hash"
    else:
        edit_stable_manifest(work, seg={"model": "0.4b"})
        expect = "seg_model: '0.8b' → '0.4b'"
    logs: list[str] = []
    run = vec.run_vectorize(cfg, log=logs.append)
    assert run["processed"] == N_FRAMES and run["skipped"] == 0 and expect in run["stale"][0] + "".join(run["stale"])
    assert any("PERINGATAN" in m and expect in m for m in logs)


def test_track_and_depth_line_params_make_stale(tmp_path):
    """Hash = 4 parameter T-201a + depth_lines.* (T-201b) + track.max_match_dist_px (T-202)."""
    work = make_work(tmp_path)
    vec.run_vectorize(cfg_for(work), log=quiet)
    run = vec.run_vectorize(cfg_for(work, **{"vectorize.track.max_match_dist_px": 20.0}), log=quiet)
    assert any("vectorize.track.max_match_dist_px: 16.0 → 20.0" in s for s in run["stale"])
    assert run["processed"] == N_FRAMES and run["skipped"] == 0
    run = vec.run_vectorize(cfg_for(work, **{"vectorize.track.max_match_dist_px": 20.0,
                                             "vectorize.depth_lines.min_len_px": 40.0}), log=quiet)
    assert any("vectorize.depth_lines.min_len_px: 30.0 → 40.0" in s for s in run["stale"])
    assert run["processed"] == N_FRAMES and run["skipped"] == 0


def test_contract_change_marks_stale(tmp_path):
    work = make_work(tmp_path)
    vec.run_vectorize(cfg_for(work), log=quiet)
    p = work / "contours" / "manifest.json"
    m = json.loads(p.read_text(encoding="utf-8"))
    m["contract"] = "T-201b"            # manifest lama (sebelum anchor / orientasi / track_id)
    p.write_text(json.dumps(m), encoding="utf-8")
    run = vec.run_vectorize(cfg_for(work), log=quiet)
    assert run["stale"] == ["contract: 'T-201b' → 'T-305b'"] and run["processed"] == N_FRAMES


@pytest.mark.parametrize("old", [1, None])
def test_algo_rev_change_marks_stale_with_warning(tmp_path, old):
    """Perbaikan perilaku tanpa perubahan parameter → algo_rev naik → output lama basi, dihitung ulang."""
    assert vec.ALGO_REV == 2
    work = make_work(tmp_path)
    vec.run_vectorize(cfg_for(work), log=quiet)
    p = work / "contours" / "manifest.json"
    m = json.loads(p.read_text(encoding="utf-8"))
    if old is None:
        del m["algo_rev"]
    else:
        m["algo_rev"] = old
    p.write_text(json.dumps(m), encoding="utf-8")
    logs: list[str] = []
    run = vec.run_vectorize(cfg_for(work), log=logs.append)
    assert run["stale"] == [f"algo_rev: {old!r} → 2"] and run["processed"] == N_FRAMES and run["skipped"] == 0
    assert any("PERINGATAN" in m and f"algo_rev: {old!r} → 2" in m for m in logs)
    assert json.loads(p.read_text(encoding="utf-8"))["algo_rev"] == 2


def test_same_manifest_resumes_without_delete(tmp_path):
    work = make_work(tmp_path)
    vec.run_vectorize(cfg_for(work), log=quiet)
    marker = work / "contours" / "frame_00000.json"
    t = marker.stat().st_mtime_ns
    assert vec.run_vectorize(cfg_for(work), log=quiet)["stale"] == [] and marker.stat().st_mtime_ns == t


def test_outputs_without_manifest_rejected(tmp_path):
    work = make_work(tmp_path)
    vec.run_vectorize(cfg_for(work), log=quiet)
    (work / "contours" / "manifest.json").unlink()
    with pytest.raises(StageError, match="tanpa manifest.json"):
        vec.run_vectorize(cfg_for(work), log=quiet)
    assert len(frame_hashes(work)) == N_FRAMES            # tidak dihapus
    assert vec.run_vectorize(cfg_for(work), restart=True, log=quiet)["processed"] == N_FRAMES


def test_restart_removes_contours_only_and_tmp_cleaned(tmp_path):
    work = make_work(tmp_path)
    vec.run_vectorize(cfg_for(work), limit=1, log=quiet)
    leftover = work / "contours" / "frame_00001.json.tmp"
    leftover.write_text("x", encoding="utf-8")
    vec.run_vectorize(cfg_for(work), log=quiet)
    assert not leftover.exists()
    assert (work / "stable" / "manifest.json").is_file()
    vec.run_vectorize(cfg_for(work), restart=True, limit=1, log=quiet)
    assert len(frame_hashes(work)) == 1 and (work / "stable" / "groups" / "frame_00003.png").is_file()


# ── Input [3] ──────────────────────────────────────
def test_stable_manifest_missing_rejected(tmp_path):
    work = make_work(tmp_path)
    (work / "stable" / "manifest.json").unlink()
    with pytest.raises(StageError, match="python -m rotoscope stabilize"):
        vec.run_vectorize(cfg_for(work), log=quiet)


def test_stable_manifest_without_clip_rejected(tmp_path):
    work = make_work(tmp_path)
    p = work / "stable" / "manifest.json"
    m = json.loads(p.read_text(encoding="utf-8"))
    del m["clip"]
    p.write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(StageError, match="identitas klip"):
        vec.run_vectorize(cfg_for(work), log=quiet)
    assert not (work / "contours").exists()


def test_stable_manifest_other_clip_rejected(tmp_path):
    work = make_work(tmp_path)
    edit_stable_manifest(work, clip={"meta_sha256": "f" * 64, "source_path": "C:/clips/lain.mp4"})
    with pytest.raises(StageError, match="klip LAIN"):
        vec.run_vectorize(cfg_for(work), log=quiet)
    assert not (work / "contours").exists()


def test_frame_size_mismatch_rejected(tmp_path):
    work = make_work(tmp_path)
    edit_stable_manifest(work, frame_size={"width": W + 1, "height": H})
    with pytest.raises(StageError, match="frame_size"):
        vec.run_vectorize(cfg_for(work), log=quiet)


def test_groups_config_differs_from_stable_rejected(tmp_path):
    work = make_work(tmp_path)
    cfg = cfg_for(work, groups=merged_groups())
    with pytest.raises(StageError, match="grup di config"):
        vec.run_vectorize(cfg, log=quiet)


def test_missing_group_or_depth_frame_rejected_even_with_limit(tmp_path):
    """Ambang per klip dihitung dari SELURUH klip → frame hilang menghentikan run, juga dengan --limit."""
    work = make_work(tmp_path)
    (work / "stable" / "groups" / "frame_00003.png").unlink()
    for limit in (None, 3):
        with pytest.raises(StageError, match="stable/groups belum lengkap"):
            vec.run_vectorize(cfg_for(work), limit=limit, log=quiet)
    stb.write_groups(work / "stable" / "groups" / "frame_00003.png", subject_groups(3))
    (work / "stable" / "depth_smooth" / "frame_00003.npy").unlink()
    for limit in (None, 3):
        with pytest.raises(StageError, match="stable/depth_smooth belum lengkap"):
            vec.run_vectorize(cfg_for(work), limit=limit, log=quiet)


def test_corrupt_group_frame_rejected(tmp_path):
    work = make_work(tmp_path)
    (work / "stable" / "groups" / "frame_00001.png").write_bytes(b"bukan png")
    with pytest.raises(StageError, match="rusak"):
        vec.run_vectorize(cfg_for(work), log=quiet)


def test_corrupt_depth_frame_rejected(tmp_path):
    work = make_work(tmp_path)
    (work / "stable" / "depth_smooth" / "frame_00001.npy").write_bytes(b"bukan npy")
    with pytest.raises(StageError, match="depth_smooth frame_00001 rusak"):
        vec.run_vectorize(cfg_for(work), log=quiet)


def test_limit_must_be_positive(tmp_path):
    with pytest.raises(StageError, match="--limit"):
        vec.run_vectorize(cfg_for(make_work(tmp_path)), limit=0, log=quiet)


# ── Entry point + impor ────────────────────────────
def _write_cfg(tmp_path: Path, work: Path) -> Path:
    p = tmp_path / "cfg.yaml"
    p.write_text(f"paths:\n  work_dir: '{work.as_posix()}'\n", encoding="utf-8")
    return p


def test_main_exit_ok(tmp_path, capsys):
    work = make_work(tmp_path)
    assert vec.main(["--config", str(_write_cfg(tmp_path, work)), "--limit", "1"]) == stage_common.EXIT_OK
    assert "selesai: 1 diproses" in capsys.readouterr().out


def test_main_work_dir_wins_over_config(tmp_path):
    work, wrong = make_work(tmp_path), tmp_path / "salah"
    assert vec.main(["--config", str(_write_cfg(tmp_path, wrong)), "--work-dir", str(work), "--limit", "1"]) == 0
    assert (work / "contours" / "manifest.json").is_file() and not wrong.exists()


def test_main_exit_precondition(tmp_path, capsys):
    work = make_work(tmp_path)
    (work / "stable" / "manifest.json").unlink()
    assert vec.main(["--config", str(_write_cfg(tmp_path, work))]) == stage_common.EXIT_PRECONDITION
    assert "ERROR" in capsys.readouterr().err


def test_torch_not_imported_by_vectorize():
    code = "import sys, rotoscope.vectorize; sys.exit(1 if 'torch' in sys.modules else 0)"
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


# ── Data nyata (ditandai; dilewati kalau klip tidak ada) ──
# 9, 19, 20, 21 = kasus terburuk sebelum perbaikan junction. Frame 38 / 40 (96,8–97,0%) sengaja tidak di sini:
# satu komponen skeleton terisolasi 15–17 px (bbox 5×7, semua cabang < min_stroke_px) dibuang sesuai aturan.
# Frame 64 dikeluarkan di T-302 Tahap 4: dengan `stable/` baru (temporal + log_median) cakupan 97,62% (< 98%) karena dua komponen skeleton
# terisolasi (11 px hair|torso di (279, 303) dan 4 px face|torso di (277, 301), flanks 0; semua cabang < min_stroke_px) dibuang — aturan
# yang sama dengan frame 38 / 40 di atas. Aturan tracer tidak diubah; keputusan melonggarkan / mengganti frame ada di Rio.
REAL_FRAMES = (0, 9, 19, 20, 21, 100)
needs_real = pytest.mark.skipif(not (REAL_CLIP / "stable" / "groups" / "frame_00000.png").is_file(),
                                reason="data nyata work/clips/test_short/stable tidak ada")


@needs_real
@pytest.mark.parametrize("index", REAL_FRAMES)
def test_real_frame_boundary_coverage_at_least_98_percent(index):
    """Syarat lulus permanen: ≥ 98% piksel skeleton batas grup tercakup polyline (≤ 1 px) + pelacak utuh."""
    g = stb.read_groups(REAL_CLIP / "stable" / "groups" / f"frame_{index:05d}.png")
    m = sm.measure(g, NAMES, PARAMS)
    assert m["skeleton"] > 0
    assert m["coverage"] >= 98.0, (index, m["coverage"], sorted(m["runs"], key=lambda r: -r["px"])[:3])
    assert m["integrity"] == {"repeats": 0, "jumps": 0}
    strokes = sum(p["strokes"] for p in m["pairs"])
    components = sum(p["components"] for p in m["pairs"])  # satu komponen bercabang boleh jadi beberapa strok
    assert strokes <= 2 * components, (index, strokes, components)


@needs_real
def test_real_frame_basic_sanity():
    g = stb.read_groups(REAL_CLIP / "stable" / "groups" / "frame_00000.png")
    strokes, st = vec.vectorize_gmap(g, NAMES, PARAMS)
    assert len(by_type(strokes, "silhouette")) >= 1 and st["edge_points"] > 0
    for s in strokes:
        assert set(s) == set(vec.STROKE_KEYS) and len(s["points"]) >= 1
