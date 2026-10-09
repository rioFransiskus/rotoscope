"""Test T-305b: `exclude_groups` + histeresis jarak terjaga (`min_dist_low_px`) pada garis oklusi, stage [4].

Sintetis non-trivial (tegak lurus ke batas, sejajar menempel, seluruhnya di zona D, benih tidak lolos L, dua benih,
tepi frame, sudut landai vs curam, grup dikecualikan, group_boundary tidak terpengaruh) + default netral identik +
invarian pada data nyata (dilewati bila `stable/` klip tidak ada). Mutation-sensitive: tiap pengaman punya test yang
gagal bila pengaman itu dimatikan (dicek lewat scripts/t305b_mutations.py, monkeypatch dari luar berkas ini).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import reach_metrics as rm
from test_vectorize import DEPTH_DEFAULTS, HAIR, N_FRAMES, NAMES, PARAMS, TORSO, blank, cfg_for, make_work, quiet
from test_vectorize_occlusion import FH, FW, T_HIGH, T_LOW, XX, YY, big_group, step

from rotoscope import vectorize as vec
from rotoscope.config import ConfigError, load_pipeline

P = {**PARAMS, **DEPTH_DEFAULTS}
D = 7.0


def occ(g, z, d_low=0.0, exclude=(), **over):
    params = {**P, "depth_lines.min_dist_low_px": float(d_low), "depth_lines.exclude_groups": list(exclude), **over}
    return vec.occlusion_strokes(g, z, NAMES, params, T_HIGH, T_LOW)[0]


def xs(strokes):
    return np.array([p[0] for s in strokes for p in s["points"]])


def ys(strokes):
    return np.array([p[1] for s in strokes for p in s["points"]])


# ── Tegak lurus ke batas: memanjang ────────────────
def test_perpendicular_ridge_extends_toward_boundary_both_ends():
    g, z = big_group(), step(YY >= 90)                      # garis horizontal y = 90, batas kiri x = 20, kanan x = 139
    base, new = occ(g, z), occ(g, z, 2)
    assert len(base) == len(new) == 1
    assert xs(new).min() < xs(base).min() - 3 and xs(new).max() > xs(base).max() + 3
    assert xs(new).min() >= 20 + 2 - 1 and xs(new).max() <= 139 - 2 + 1          # tidak melewati D_low
    assert rm.new_components(base, new) == 0
    assert not [r for r in rm.zone_runs(new, vec.vectorize_gmap(g, NAMES, P)[0], D) if r["kind"] != "end"]


def test_extension_length_is_bounded_by_contract():
    g, z = big_group(), step(YY >= 90)
    for d_low in (5, 4, 3, 2):
        base, new = occ(g, z), occ(g, z, d_low)
        grow = (xs(base).min() - xs(new).min())
        assert 0 <= grow <= rm.reach_bound_px(D, d_low)


def test_default_neutral_is_identical():
    g, z = big_group(), step(YY >= 90)
    assert occ(g, z, 0) == occ(g, z, 0, ()) == vec.occlusion_strokes(g, z, NAMES, P, T_HIGH, T_LOW)[0]
    assert occ(g, z, D) == occ(g, z, 0)                       # D_low >= D = mati (fungsi; config menolak nilai ini)


# ── Sejajar menempel / seluruhnya di zona D: tidak boleh ──
def test_ridge_parallel_inside_zone_d_is_not_created():
    g, z = big_group(), step(XX >= 25)                         # jarak ke batas kiri 5 < D: tanpa benih
    assert occ(g, z) == [] and occ(g, z, 2) == []


def test_shallow_approach_is_not_extended_but_steep_is():
    def slanted(angle_deg):
        t = math.tan(math.radians(angle_deg))
        return step((XX >= 20 + np.maximum(YY - 90, 0) * t) & (XX >= 20))
    g = big_group()
    for angle, grows in ((15, False), (70, True)):          # sudut terhadap batas kiri (vertikal): landai vs curam
        z = slanted(angle)
        base, new = occ(g, z), occ(g, z, 2)
        assert base, angle
        moved = ys(base).min() - ys(new).min() if angle == 70 else xs(base).min() - xs(new).min()
        assert (moved >= 1.5) == grows, (angle, moved)
        assert rm.new_components(base, new) == 0


# ── Benih harus lolos L ────────────────────────────
def test_seed_must_pass_L_before_extension():
    g, z = big_group(), step(YY >= 90)
    assert len(occ(g, z, 0, **{"depth_lines.min_len_px": 100.0})) == 1
    # benih ±106 px < L = 110: dibuang; dengan ekstensi (±116) NAIVE akan lolos L, terjaga tidak
    assert occ(g, z, 0, **{"depth_lines.min_len_px": 110.0}) == []
    assert occ(g, z, 2, **{"depth_lines.min_len_px": 110.0}) == []


def test_two_seeds_extend_independently():
    g, z = big_group(), step((YY >= 60) & (YY < 120))
    base, new = occ(g, z), occ(g, z, 2)
    assert len(base) == len(new) == 2
    assert all(xs([n]).min() < xs([b]).min() - 3 for b, n in zip(base, new))


# ── Tepi frame = batas ─────────────────────────────
def test_frame_edge_is_a_boundary_not_crossed():
    g, z = big_group(), step(XX >= 80)                         # garis vertikal sampai tepi bawah frame (baris 169)
    base, new = occ(g, z), occ(g, z, 2)
    assert ys(new).max() > ys(base).max() + 3
    assert ys(new).max() <= FH - 1 - 2 + 0.5                   # jarak ≥ D_low dari baris tepi (tepi frame = batas)


# ── exclude_groups ─────────────────────────────────
def two_group_frame():
    g = blank(FH, FW)
    g[10:90, 20:140] = HAIR
    g[90:FH, 20:140] = TORSO
    return g, step(XX >= 80)


def test_exclude_groups_removes_only_listed_group():
    g, z = two_group_frame()
    allg = occ(g, z)
    assert {s["groups"][0] for s in allg} == {"hair", "torso"}
    no_hair = occ(g, z, exclude=["hair"])
    assert [s for s in allg if s["groups"][0] != "hair"] == no_hair
    assert occ(g, z, exclude=["hair", "torso"]) == []
    assert occ(g, z, exclude=["face"]) == allg                 # grup tak punya garis: tidak berubah


def test_exclude_groups_leaves_boundary_and_silhouette_untouched():
    g, z = two_group_frame()
    params_a = {**P, "depth_lines.exclude_groups": []}
    params_b = {**P, "depth_lines.exclude_groups": ["hair"]}
    a = vec.vectorize_frame(g, z, NAMES, params_a, T_HIGH, T_LOW)[0]
    b = vec.vectorize_frame(g, z, NAMES, params_b, T_HIGH, T_LOW)[0]
    other = lambda ss: [s for s in ss if s["type"] != "occlusion"]
    assert other(a) == other(b) and len(other(a)) > 0
    assert any(s["type"] == "group_boundary" for s in b)
    assert not any(s["type"] == "occlusion" and s["groups"] == ["hair"] for s in b)


def test_exclude_combined_with_extension():
    g, z = two_group_frame()
    both = occ(g, z, 2, exclude=["hair"])
    assert {s["groups"][0] for s in both} == {"torso"}
    assert both == [s for s in occ(g, z, 2) if s["groups"][0] == "torso"]


# ── Konfigurasi, hash, clip_stats ──────────────────
def test_config_normalizes_exclude_groups():
    c = load_pipeline(overrides={"vectorize.depth_lines.exclude_groups": ["torso", "hair", "hair"]})
    assert c.vectorize.depth_lines.exclude_groups == ("hair", "torso")
    assert vec.vectorize_params(c)["depth_lines.exclude_groups"] == ["hair", "torso"]
    d = load_pipeline()                                   # default produksi (keputusan Rio, T-305b)
    assert d.vectorize.depth_lines.exclude_groups == ("hair",) and d.vectorize.depth_lines.min_dist_low_px == 2
    off = load_pipeline(overrides={"vectorize.depth_lines.exclude_groups": [], "vectorize.depth_lines.min_dist_low_px": 0})
    assert off.vectorize.depth_lines.exclude_groups == () and off.vectorize.depth_lines.min_dist_low_px == 0


def test_config_unknown_group_lists_valid_names():
    with pytest.raises(ConfigError, match="hair, face, torso, left_arm"):
        load_pipeline(overrides={"vectorize.depth_lines.exclude_groups": ["tail"]})


def test_params_hash_changes_with_new_params_and_not_with_order():
    base = vec.params_hash(vec.vectorize_params(load_pipeline()))
    a = vec.params_hash(vec.vectorize_params(load_pipeline(overrides={"vectorize.depth_lines.min_dist_low_px": 0})))
    b = vec.params_hash(vec.vectorize_params(load_pipeline(overrides={"vectorize.depth_lines.exclude_groups": []})))
    o1 = vec.params_hash(vec.vectorize_params(load_pipeline(overrides={"vectorize.depth_lines.exclude_groups": ["hair", "torso"]})))
    o2 = vec.params_hash(vec.vectorize_params(load_pipeline(overrides={"vectorize.depth_lines.exclude_groups": ["torso", "hair", "torso"]})))
    assert len({base, a, b}) == 3 and o1 == o2


def test_clip_stats_thresholds_independent_of_d_low_and_exclude(tmp_path):
    work = make_work(tmp_path)
    vec.run_vectorize(cfg_for(work), log=quiet)
    s0 = json.loads((work / "contours" / "clip_stats.json").read_text(encoding="utf-8"))
    cfg2 = cfg_for(work, **{"vectorize.depth_lines.min_dist_low_px": 0, "vectorize.depth_lines.exclude_groups": []})
    run = vec.run_vectorize(cfg2, log=quiet)
    s1 = json.loads((work / "contours" / "clip_stats.json").read_text(encoding="utf-8"))
    assert s0 == s1                                   # masukan DAN hasil identik: file dipakai ulang, tidak dihitung ulang
    assert run["stale"] and any("min_dist_low_px" in s for s in run["stale"])        # contours/ basi, dihitung ulang


def test_neutral_contours_restart_identical(tmp_path):
    """Nilai netral (D_low 0, tanpa pengecualian): `restart` menghasilkan byte yang sama (kesetaraan dengan jalur lama T-202
    dibuktikan pada klip nyata di Tahap 4 T-305b dan oleh test_vectorize_occlusion.py yang memakai jalur dasar)."""
    neutral = {"vectorize.depth_lines.min_dist_low_px": 0, "vectorize.depth_lines.exclude_groups": []}
    work = make_work(tmp_path)
    vec.run_vectorize(cfg_for(work, **neutral), log=quiet)
    first = {p.name: p.read_bytes() for p in (work / "contours").glob("frame_*.json")}
    vec.run_vectorize(cfg_for(work, **neutral), restart=True, log=quiet)
    assert first == {p.name: p.read_bytes() for p in (work / "contours").glob("frame_*.json")}


def test_limit_run_is_prefix_of_full_run_with_new_params(tmp_path):
    over = {"vectorize.depth_lines.min_dist_low_px": 2, "vectorize.depth_lines.exclude_groups": ["hair"]}
    w1, w2 = make_work(tmp_path / "a"), make_work(tmp_path / "b")
    vec.run_vectorize(cfg_for(w1, **over), log=quiet)
    vec.run_vectorize(cfg_for(w2, **over), limit=2, log=quiet)
    for p in sorted((w2 / "contours").glob("frame_*.json")):
        assert p.read_bytes() == (w1 / "contours" / p.name).read_bytes()


# ── Data nyata ─────────────────────────────────────
REAL = {c: Path("work/clips") / c for c in ("test_short", "test")}


def real_ready(clip: str) -> bool:
    w = REAL[clip]
    return (w / "stable" / "manifest.json").is_file() and (w / "contours" / "clip_stats.json").is_file()


@pytest.mark.parametrize("clip", list(REAL))
def test_real_invariants_d_low_2(clip):
    """Pada frame nyata (setiap 7): tidak ada komponen baru, tidak ada run zona D di tengah / seluruh strok, ekstensi ≤ batas,
    persilangan baru 0; exclude [hair] menghapus tepat strok hair."""
    if not real_ready(clip):
        pytest.skip("data nyata tidak ada")
    cl = vec.load_clip(REAL[clip])
    cs = json.loads((REAL[clip] / "contours" / "clip_stats.json").read_text(encoding="utf-8"))
    th, tl = cs["t_high"], cs["t_low"]
    names = tuple(g for g, _ in load_pipeline().groups)
    params = {**P, "min_region_area": 800}
    ext_total = 0
    for i in range(0, len(cl.names), 7):
        gmap, depth = vec.load_frame_inputs(cl, cl.names[i], len(names))
        base = vec.occlusion_strokes(gmap, depth, names, params, th, tl)[0]
        var = vec.occlusion_strokes(gmap, depth, names, {**params, "depth_lines.min_dist_low_px": 2.0}, th, tl)[0]
        others = vec.vectorize_gmap(gmap, names, params)[0]
        assert rm.new_components(base, var) == 0, i
        bad = [r for r in rm.zone_runs(var, others, D) if r["kind"] != "end"]
        assert not bad, (i, bad)
        assert all(r["length"] <= rm.reach_bound_px(D, 2) + 1 for r in rm.zone_runs(var, others, D)), i
        assert min((np.hypot(*(rm.pts(s)[:, None, :] - rm.pts(o)[None, :, :]).T).min() for s in var for o in others
                    if len(rm.pts(o))), default=9) >= 0.99, i                                   # tidak menyentuh / memotong
        ext_total += sum(len(s["points"]) for s in var) - sum(len(s["points"]) for s in base)
        excl = vec.occlusion_strokes(gmap, depth, names, {**params, "depth_lines.exclude_groups": ["hair"]}, th, tl)[0]
        assert excl == [s for s in base if s["groups"][0] != "hair"], i
    assert ext_total > 0
