"""Test garis oklusi vectorize.py (T-201b): depth_smooth + peta grup sintetis, tanpa model/GPU. Bentuk TIDAK hanya
sumbu-sejajar (diagonal 45°, busur, X, T), plus mutation-sensitive: tiap pengaman (NMS, syarat D, hysteresis, L,
pelacak junction) punya test yang gagal bila pengaman itu dimatikan (dicek dengan plugin pytest di luar repo).
Data nyata (ditandai) = metrik objektif permanen: bayangan, cakupan, integritas, kriteria wilayah Lower_Clothing."""

from __future__ import annotations

import functools
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import skeleton_metrics as sm
from test_vectorize import (DEPTH_DEFAULTS, N_FRAMES, NAMES, PARAMS, TORSO, LARM, blank, by_type, cfg_for,
                            frame_hashes, make_work, quiet)

from rotoscope import stabilize as stb
from rotoscope import stage_common
from rotoscope import vectorize as vec
from rotoscope.config import load_class_names, load_pipeline
from rotoscope.stage_common import StageError

FH, FW = 170, 160
P = {**PARAMS, **DEPTH_DEFAULTS}
T_HIGH, T_LOW = 0.5, 0.2
YY, XX = np.mgrid[0:FH, 0:FW]


def big_group() -> np.ndarray:
    """Foreground persegi panjang 160 × 120 yang menempel tepi bawah frame (seperti klip nyata)."""
    g = blank(FH, FW)
    g[10:FH, 20:140] = TORSO
    return g


def step(mask: np.ndarray, amp: float = 3.0) -> np.ndarray:
    return (mask * amp).astype(np.float32)


def occ(g: np.ndarray, z: np.ndarray, **over) -> tuple[list[dict], dict]:
    return vec.occlusion_strokes(g, z, NAMES, {**P, **over}, T_HIGH, T_LOW)


def all_points(strokes: list[dict]) -> np.ndarray:
    return np.array([p for s in strokes for p in s["points"]], float).reshape(-1, 2)


def clean(strokes: list[dict]) -> None:
    assert sm.integrity(strokes) == {"repeats": 0, "jumps": 0}


# ── Bentuk tepi ────────────────────────────────────
def test_vertical_and_horizontal_step_one_stroke_each():
    g = big_group()
    for z, axis, at in ((step(XX >= 80), 0, 80.0), (step(YY >= 90), 1, 90.0)):
        strokes, st = occ(g, z)
        assert len(strokes) == 1 and st["n_occlusion"] == 1
        pts = all_points(strokes)
        assert abs(np.median(pts[:, axis]) - at) <= 1.0 and np.ptp(pts[:, axis]) <= 2.0
        assert np.ptp(pts[:, 1 - axis]) >= 80
        clean(strokes)


def test_diagonal_45_degree_step_is_one_continuous_stroke():
    g = big_group()
    strokes, _ = occ(g, step(XX - YY >= 0))
    pts = all_points(strokes)
    assert len(strokes) == 1 and len(pts) >= 80
    assert np.abs(pts[:, 0] - pts[:, 1]).max() <= 3.0          # tetap di garis x = y
    clean(strokes)


def test_arc_circle_is_one_loop_with_end_equal_start():
    g = big_group()
    strokes, st = occ(g, step((XX - 80) ** 2 + (YY - 90) ** 2 < 40 ** 2))
    assert len(strokes) == 1 and st["occ_loops"] == 1
    s = strokes[0]
    assert s["closed"] is False and s["points"][0] == s["points"][-1]
    r = np.hypot(all_points(strokes)[:, 0] - 80, all_points(strokes)[:, 1] - 90)
    assert abs(np.median(r) - 40) <= 1.5 and r.min() >= 37 and r.max() <= 43
    clean(strokes)


def test_quarter_arc_is_one_stroke():
    g = big_group()
    strokes, _ = occ(g, step((XX - 40) ** 2 + (YY - 20) ** 2 < 70 ** 2))      # busur: pusat di luar foreground
    assert len(strokes) == 1 and len(strokes[0]["points"]) >= 60
    clean(strokes)


def test_x_crossing_all_four_arms_traced_without_repeats_or_jumps():
    """Dua tepi tipis yang bersilang: NMS menghasilkan persilangan dua garis 1 px (dua T berdekatan di skeleton), jadi
    pelacak bisa menggabung dua lengan berseberangan → 3 atau 4 strok; yang dijamin: keempat lengan tercakup."""
    g = big_group()
    z = step(XX - YY >= 0) + step(XX + YY >= 170)
    strokes, _ = occ(g, z)
    assert 3 <= len(strokes) <= 5
    clean(strokes)
    m = sm.occlusion_measure(g, z, NAMES, P, T_HIGH, T_LOW)
    assert m["skeleton_coverage"] >= 98.0 and m["band_coverage"] >= 98.0 and m["skeleton"] >= 200
    pts = all_points(strokes)
    for arm in ((40, 40), (40, 130), (125, 45), (125, 125)):     # titik tengah tiap lengan
        assert np.hypot(pts[:, 0] - arm[0], pts[:, 1] - arm[1]).min() <= 2.0, arm


def test_t_junction_three_arms_traced_and_covered():
    g = big_group()
    z = 3 * step(YY >= 90, 1.0) + step((XX >= 80) & (YY < 90), 6.0)      # kiri 0 | atas-kanan 6 | bawah 3
    strokes, _ = occ(g, z)
    assert 2 <= len(strokes) <= 4          # dua lengan horizontal boleh tersambung jadi satu garis lurus
    clean(strokes)
    m = sm.occlusion_measure(g, z, NAMES, P, T_HIGH, T_LOW)
    assert m["skeleton_coverage"] >= 98.0 and m["band_coverage"] >= 98.0
    pts = all_points(strokes)
    for arm in ((45, 90), (125, 90), (80, 45)):
        assert np.hypot(pts[:, 0] - arm[0], pts[:, 1] - arm[1]).min() <= 2.0, arm


# ── Tanpa tepi / lemah / noise ─────────────────────
def test_smooth_ramp_has_no_line():
    """Ramp mulus (gradien konstan 0,05 ≪ T_low) dan ramp diagonal: tanpa tepi → tanpa garis."""
    g = big_group()
    for z in ((0.05 * XX).astype(np.float32), (0.03 * (XX + YY)).astype(np.float32)):
        strokes, st = occ(g, z)
        assert strokes == [] and st["occ_px_hyst"] == 0


def test_low_noise_is_not_a_line():
    g = big_group()
    z = np.random.default_rng(1).normal(0, 0.02, (FH, FW)).astype(np.float32)
    assert occ(g, z)[0] == []


def test_flat_depth_frame_is_valid_without_occlusion():
    strokes, st = occ(big_group(), np.zeros((FH, FW), np.float32))
    assert strokes == [] and st["n_occlusion"] == 0 and st["occ_px_hyst"] == 0
    assert vec.occlusion_strokes(big_group(), np.zeros((FH, FW), np.float32), NAMES, P, None, None)[0] == []


def test_hysteresis_weak_connected_kept_isolated_weak_dropped():
    g = big_group()
    # garis horizontal y = 60 yang amplitudonya menurun mulus (amp 3 → 1,1; gradien horizontal 0,06/px ≪ T_low, jadi
    # tidak ada tepi vertikal): kiri kuat (|grad| ≈ 0,32 × amp), kanan lemah (antara T_low dan T_high) tetapi
    # bersambung dengan yang kuat
    amp = np.clip(3.0 - (XX - 40) * 0.06, 1.1, 3.0)
    z = (amp * (YY >= 60)).astype(np.float32)
    # kotak lemah terisolasi (amp 1,1) di bawah: semua tepinya lemah, tidak ada piksel kuat
    z = z + 1.1 * ((YY >= 130) & (XX >= 40) & (XX < 120)).astype(np.float32)
    masks = vec.occlusion_masks(g, z, vec.depth_params(P), T_HIGH, T_LOW)
    weak = masks["mag"][131:160, 41:119].max()                  # interior kotak: gradien tinggi hanya di tepinya
    iso = masks["mag"][130, 60:100].max()
    assert T_LOW < iso < T_HIGH, iso                            # prasyarat: tepi terisolasi memang lemah
    assert T_LOW < masks["mag"][60, 120:130].max() < T_HIGH     # dan ekor garis atas lemah
    h = masks["hyst"]
    assert h[55:66, 100:130].sum() >= 25 and h[55:66, 40:70].sum() >= 25     # ekor lemah dipertahankan (+ bagian kuat)
    assert not h[125:170, 30:130].any() and weak < T_HIGH       # lemah terisolasi → dibuang
    strokes, _ = occ(g, z)
    ys = all_points(strokes)[:, 1]
    assert len(strokes) == 1 and (np.abs(ys - 60) <= 2).all()


def test_nms_makes_pre_thinning_mask_thin():
    """Mask sesudah NMS + hysteresis ≤ 2 px lebar (tanpa NMS: pita gradien 5+ px). Mutation: NMS dimatikan → gagal."""
    g = big_group()
    for z, length in ((step(XX >= 80), 150), (step(XX - YY >= 0), 150)):
        h = vec.occlusion_masks(g, z, vec.depth_params(P), T_HIGH, T_LOW)["hyst"]
        assert 0 < h.sum() <= 2.2 * length, h.sum()


# ── Syarat D, foreground, L ────────────────────────
def test_edge_closer_than_d_to_group_boundary_dropped():
    g = big_group()                                              # batas kiri foreground di x = 20
    assert occ(g, step(XX >= 25))[0] == []                       # 5 px dari batas (< D = 7)
    assert len(occ(g, step(XX >= 35))[0]) == 1                   # 15 px: tetap
    assert occ(g, step(XX >= 25), **{"depth_lines.min_dist_px": 3.0})[0] != []   # D lebih kecil → muncul


def test_frame_edge_counts_as_boundary_for_d():
    """Tepi kedalaman dalam D px dari tepi bawah frame dibuang (foreground menempel tepi); padding tidak membuat
    garis palsu di tepi frame."""
    g = big_group()
    assert occ(g, step(YY >= FH - 4))[0] == []
    assert len(occ(g, step(YY >= FH - 20))[0]) == 1
    assert occ(g, np.zeros((FH, FW), np.float32))[0] == []       # depth datar: tidak ada garis di tepi frame
    d = vec.boundary_distance(g)
    assert d[FH - 1, 80] == 0 and d[FH - 8, 80] == 7 and d[FH - 1 - 7, 80] >= 7


def test_edge_outside_foreground_dropped():
    g = big_group()
    assert occ(g, step((XX < 15) & (YY >= 60) & (YY < 120)))[0] == []      # kotak di background kiri (foreground x ≥ 20)
    assert occ(g, step(YY < 6))[0] == []                                   # tepi y = 6 di luar foreground (mulai y = 10)
    assert occ(g, step((XX >= 60) & (XX < 90) & (YY >= 70) & (YY < 100)))[0] != []   # kotak yang sama di dalam: ada


def test_component_shorter_than_l_dropped():
    g = big_group()
    short = step((XX >= 60) & (XX < 90) & (YY >= 70) & (YY < 100))             # kotak 30 × 30: kontur ≈ 120 px
    assert occ(g, short)[0] != []
    assert occ(g, short, **{"depth_lines.min_len_px": 200.0})[0] == []


def test_l_threshold_compares_skeleton_pixels():
    g = big_group()
    z = step(XX - YY >= 0)
    assert len(occ(g, z, **{"depth_lines.min_len_px": 60.0})[0]) == 1
    assert occ(g, z, **{"depth_lines.min_len_px": 300.0})[0] == []


# ── Skema, grup, strength ──────────────────────────
def test_stroke_schema_strength_rounding_and_single_group():
    g = big_group()
    strokes, st = vec.vectorize_frame(g, step(XX - YY >= 0), NAMES, P, T_HIGH, T_LOW)
    (o,) = by_type(strokes, "occlusion")
    assert set(o) == set(vec.OCCLUSION_KEYS) and o["closed"] is False and o["groups"] == [NAMES[TORSO - 1]]
    assert isinstance(o["strength"], float) and round(o["strength"], vec.STRENGTH_DECIMALS) == o["strength"]
    assert T_HIGH < o["strength"] < 3.0
    assert all(set(s) == set(vec.STROKE_KEYS) for s in strokes if s["type"] != "occlusion")
    assert [s["type"] for s in strokes] == sorted((s["type"] for s in strokes), key=vec.STROKE_TYPES.index)
    assert st["n_occlusion"] == 1 and "_pair" not in o
    assert all(round(x, 1) == x and x % 1 == 0.5 for p in o["points"] for x in p)


def test_each_stroke_single_group_even_when_d_is_zero():
    g = big_group()
    g[:, 80:140] = LARM                                          # dua grup bersebelahan: batas vertikal x = 80
    z = step(XX - YY >= 0) + step(XX + YY >= 200)
    strokes, _ = occ(g, z, **{"depth_lines.min_dist_px": 0.0})
    assert strokes
    for s in strokes:
        ids = {int(g[int(y), int(x)]) for x, y in s["points"]}
        assert len(ids) == 1 and NAMES[ids.pop() - 1] == s["groups"][0] and len(s["groups"]) == 1


def test_two_groups_get_separate_strokes_with_own_group():
    g = big_group()
    g[:, 80:140] = LARM
    strokes, _ = occ(g, step(YY >= 90))                          # satu tepi horizontal melewati dua grup
    assert {s["groups"][0] for s in strokes} == {NAMES[TORSO - 1], NAMES[LARM - 1]}


def test_deterministic_two_calls_identical():
    g = big_group()
    z = step(XX - YY >= 0) + step((XX - 80) ** 2 + (YY - 90) ** 2 < 900)
    a, b = occ(g, z), occ(g, z)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_percentile_linear_matches_numpy_and_is_deterministic():
    v = np.sort(np.random.default_rng(3).random(10_001).astype(np.float32))
    for p in (50.0, 90.0, 95.0, 99.0, 0.5):
        assert vec.percentile_linear(v, p) == pytest.approx(float(np.percentile(v.astype(np.float64), p)), rel=1e-12)
    assert vec.percentile_linear(np.array([2.0], np.float32), 95.0) == 2.0


# ── Metrik objektif (sintetis) ─────────────────────
def test_metrics_synthetic_shadow_coverage_integrity():
    g = big_group()
    z = step(XX - YY >= 0) + step((XX - 110) ** 2 + (YY - 60) ** 2 < 15 ** 2)
    m = sm.occlusion_measure(g, z, NAMES, P, T_HIGH, T_LOW)
    assert m["strokes"] >= 2 and m["integrity"] == {"repeats": 0, "jumps": 0}
    assert m["shadow"] >= P["depth_lines.min_dist_px"] - sm.SHADOW_TOLERANCE_PX
    assert m["skeleton_coverage"] >= 98.0 and m["band_coverage"] >= 98.0
    assert m["strokes"] <= 2 * m["components"]


def test_shadow_metric_detects_a_duplicate_of_the_silhouette():
    sil = [{"points": [[20.5, 30.5], [20.5, 31.5]]}]
    assert sm.shadow_distance([{"points": [[21.5, 30.5]]}], sil) == 1.0
    assert sm.shadow_distance([{"points": [[40.5, 30.5]]}], sil) == 20.0
    assert sm.shadow_distance([], sil) is None


# ── Klip sintetis: clip_stats, resume, basi ────────
def stats_path(work: Path) -> Path:
    return work / "contours" / "clip_stats.json"


def read_stats(work: Path) -> dict:
    return json.loads(stats_path(work).read_text(encoding="utf-8"))


def test_clip_stats_content_and_exact_percentiles(tmp_path):
    work = make_work(tmp_path)
    vec.run_vectorize(cfg_for(work), log=quiet)
    s = read_stats(work)
    assert s["contract"] == vec.CONTRACT and s["algo_rev"] == vec.ALGO_REV and s["n_frames"] == N_FRAMES
    assert (s["hi_pct"], s["lo_pct"], s["blur_sigma"], s["erode_px"]) == (95.0, 90.0, 1.0, 5)
    assert s["sobel_norm"] == vec.SOBEL_NORM and s["percentile_method"] == "linear"
    assert s["stable"]["groups_hash"] and s["stable"]["created_utc"] == "t0" and s["clip"] == stage_common.clip_identity(work)
    vals = []
    for i in range(N_FRAMES):
        gmap = stb.read_groups(work / "stable" / "groups" / f"frame_{i:05d}.png")
        z = vec.read_depth_smooth(work / "stable" / "depth_smooth" / f"frame_{i:05d}.npy", 24, 32)
        vals.append(vec.depth_gradient(z, 1.0)[0][vec.inner_region(gmap, 5)])
    v = np.concatenate(vals).astype(np.float64)
    assert s["n_values"] == v.size
    assert s["t_high"] == pytest.approx(float(np.percentile(v, 95)), rel=1e-9)
    assert s["t_low"] == pytest.approx(float(np.percentile(v, 90)), rel=1e-9)
    assert s["t_low"] < s["t_high"]
    m = json.loads((work / "contours" / "manifest.json").read_text(encoding="utf-8"))
    assert m["depth_thresholds"] == {"t_high": s["t_high"], "t_low": s["t_low"]}


def test_clip_stats_byte_identical_across_fresh_runs_and_reused_on_resume(tmp_path, monkeypatch):
    work = make_work(tmp_path)
    vec.run_vectorize(cfg_for(work), log=quiet)
    first = stats_path(work).read_bytes()
    vec.run_vectorize(cfg_for(work), restart=True, log=quiet)
    assert stats_path(work).read_bytes() == first
    monkeypatch.setattr(vec, "compute_clip_stats", lambda *a, **k: pytest.fail("pass 1 harus dilewati (masukan sama)"))
    assert vec.run_vectorize(cfg_for(work), log=quiet)["processed"] == 0
    stats_path(work).unlink()                                    # hilang: dihitung ulang + ditulis ulang (kalau ada kerja)
    monkeypatch.undo()
    (work / "contours" / "frame_00001.json").unlink()
    vec.run_vectorize(cfg_for(work), log=quiet)
    assert stats_path(work).read_bytes() == first


def test_limit_uses_full_clip_thresholds_and_is_prefix_of_full_run(tmp_path):
    full = make_work(tmp_path / "a")
    vec.run_vectorize(cfg_for(full), log=quiet)
    part = make_work(tmp_path / "b")
    vec.run_vectorize(cfg_for(part), limit=2, log=quiet)
    assert read_stats(part)["t_high"] == read_stats(full)["t_high"] and read_stats(part)["n_frames"] == N_FRAMES
    all_full = frame_hashes(full)
    assert frame_hashes(part) == {k: v for k, v in all_full.items() if k in frame_hashes(part)} and len(frame_hashes(part)) == 2
    run = vec.run_vectorize(cfg_for(part), log=quiet)            # resume: 2 dilewati, hasil akhir = run penuh
    assert (run["skipped"], run["processed"]) == (2, N_FRAMES - 2)
    assert frame_hashes(part) == all_full


def test_hi_pct_change_is_stale_and_restoring_default_reproduces_hashes(tmp_path):
    work = make_work(tmp_path)
    vec.run_vectorize(cfg_for(work), log=quiet)
    base, base_stats = frame_hashes(work), stats_path(work).read_bytes()
    logs: list[str] = []
    run = vec.run_vectorize(cfg_for(work, **{"vectorize.depth_lines.hi_pct": 93.0}), log=logs.append)
    assert run["processed"] == N_FRAMES and any("PERINGATAN" in m and "depth_lines.hi_pct: 95.0 → 93.0" in m for m in logs)
    assert read_stats(work)["hi_pct"] == 93.0 and read_stats(work)["t_high"] != json.loads(base_stats)["t_high"]
    run = vec.run_vectorize(cfg_for(work), log=quiet)
    assert run["processed"] == N_FRAMES and frame_hashes(work) == base and stats_path(work).read_bytes() == base_stats


def test_thresholds_in_manifest_are_part_of_staleness(tmp_path):
    work = make_work(tmp_path)
    vec.run_vectorize(cfg_for(work), log=quiet)
    p = work / "contours" / "manifest.json"
    m = json.loads(p.read_text(encoding="utf-8"))
    m["depth_thresholds"]["t_high"] += 0.1
    p.write_text(json.dumps(m), encoding="utf-8")
    run = vec.run_vectorize(cfg_for(work), log=quiet)
    assert any(s.startswith("depth_thresholds.t_high") for s in run["stale"]) and run["processed"] == N_FRAMES


def test_clip_stats_recomputed_when_stable_changes(tmp_path):
    work = make_work(tmp_path)
    vec.run_vectorize(cfg_for(work), log=quiet)
    from test_vectorize import edit_stable_manifest
    edit_stable_manifest(work, created_utc="t1")
    vec.run_vectorize(cfg_for(work), log=quiet)
    assert read_stats(work)["stable"]["created_utc"] == "t1"


def test_old_t201a_manifest_is_stale_and_gets_occlusion(tmp_path):
    work = make_work(tmp_path)
    vec.run_vectorize(cfg_for(work), log=quiet)
    p = work / "contours" / "manifest.json"
    m = json.loads(p.read_text(encoding="utf-8"))
    for k in ("depth_thresholds", "clip_stats"):
        m.pop(k)
    m["contract"] = "T-201a"
    m["stroke_types"] = ["silhouette", "silhouette_hole", "group_boundary"]
    m["vectorize"] = {k: v for k, v in m["vectorize"].items() if not k.startswith("depth_lines.")}
    p.write_text(json.dumps(m), encoding="utf-8")
    stats_path(work).unlink()
    logs: list[str] = []
    run = vec.run_vectorize(cfg_for(work), log=logs.append)
    assert run["processed"] == N_FRAMES and "contract: 'T-201a' → 'T-201b'" in "".join(run["stale"])
    assert stats_path(work).is_file()


def test_run_frames_log_has_occlusion_counts_and_old_types_unchanged(tmp_path):
    work = make_work(tmp_path)
    vec.run_vectorize(cfg_for(work), log=quiet)
    recs = stage_common.last_frame_records(work / "contours" / "frames.jsonl")
    assert len(recs) == N_FRAMES
    for r in recs.values():
        assert {"occ_px_hyst", "occ_px_dist", "occ_px_len", "n_occlusion"} <= set(r)
        assert r["occ_px_hyst"] >= r["occ_px_dist"] >= 0
    # tipe lama byte-identik dengan vectorize_gmap murni (regresi: oklusi tidak menyentuh siluet / lubang / batas)
    for i in range(N_FRAMES):
        d = json.loads((work / "contours" / f"frame_{i:05d}.json").read_text(encoding="utf-8"))
        old = [s for s in d["strokes"] if s["type"] != "occlusion"]
        g = stb.read_groups(work / "stable" / "groups" / f"frame_{i:05d}.png")
        assert old == vec.vectorize_gmap(g, NAMES, {**PARAMS, "min_region_area": 50})[0]


def test_missing_depth_for_stage_is_precondition_error(tmp_path):
    work = make_work(tmp_path)
    for f in (work / "stable" / "depth_smooth").glob("*.npy"):
        f.unlink()
    with pytest.raises(StageError, match="depth_smooth belum lengkap"):
        vec.run_vectorize(cfg_for(work), log=quiet)
    assert not (work / "contours").exists()


# ── Data nyata (ditandai; dilewati kalau klip tidak ada) ──
CLIPS = ("test_short", "test")
# Frame per klip (test_short hanya 119 frame; 150 / 200 hanya ada di klip test): tanpa kasus yang selalu skip.
FRAMES_REAL = {"test_short": (0, 40, 73, 78, 82, 87, 90, 92), "test": (0, 40, 73, 78, 82, 87, 90, 92, 150, 200)}
REAL_CASES = [(clip, i) for clip in CLIPS for i in FRAMES_REAL[clip]]
LEG_FRAMES = (73, 78, 82, 87, 92)
REAL_PARAMS = vec.vectorize_params(load_pipeline())


def real(clip: str) -> Path:
    return Path("work/clips") / clip


def n_frames(clip: str) -> int:
    return json.loads((real(clip) / "meta.json").read_text(encoding="utf-8"))["frame_count"]


def has_real(clip: str) -> bool:
    return (real(clip) / "stable" / "depth_smooth" / "frame_00000.npy").is_file() and (
        real(clip) / "seg" / "classmap" / "frame_00000.png").is_file()


@functools.lru_cache(maxsize=None)
def clip_thresholds(clip: str) -> tuple[float, float]:
    meta = json.loads((real(clip) / "meta.json").read_text(encoding="utf-8"))
    h, w, n = meta["working_height"], meta["working_width"], meta["frame_count"]
    chunks = []
    for i in range(n):
        g = stb.read_groups(real(clip) / "stable" / "groups" / f"frame_{i:05d}.png")
        z = vec.read_depth_smooth(real(clip) / "stable" / "depth_smooth" / f"frame_{i:05d}.npy", h, w)
        chunks.append(vec.depth_gradient(z, REAL_PARAMS["depth_lines.blur_sigma"])[0][vec.inner_region(g, 5)])
    v = np.sort(np.concatenate(chunks))
    return (vec.percentile_linear(v, REAL_PARAMS["depth_lines.hi_pct"]),
            vec.percentile_linear(v, REAL_PARAMS["depth_lines.lo_pct"]))


def real_frame(clip: str, i: int) -> tuple[np.ndarray, np.ndarray]:
    g = stb.read_groups(real(clip) / "stable" / "groups" / f"frame_{i:05d}.png")
    z = vec.read_depth_smooth(real(clip) / "stable" / "depth_smooth" / f"frame_{i:05d}.npy", *g.shape)
    return g, z


@pytest.mark.parametrize("clip, index", REAL_CASES)
def test_real_no_shadow_integrity_and_coverage(clip, index):
    if not has_real(clip):
        pytest.skip("data nyata tidak ada")
    th, tl = clip_thresholds(clip)
    g, z = real_frame(clip, index)
    m = sm.occlusion_measure(g, z, NAMES, REAL_PARAMS, th, tl)
    assert m["integrity"] == {"repeats": 0, "jumps": 0}
    if m["shadow"] is not None:                                  # bayangan: ≥ D − toleransi dari strok tipe lain
        assert m["shadow"] >= REAL_PARAMS["depth_lines.min_dist_px"] - sm.SHADOW_TOLERANCE_PX, m["shadow"]
    assert m["skeleton_coverage"] >= COVERAGE_MIN and m["band_coverage"] >= BAND_COVERAGE_MIN, m
    assert m["strokes"] <= 3 * max(m["components"], 1), (m["strokes"], m["components"])


# Per frame longgar (70%) karena komponen skeleton 30–34 px (tepat di atas L) kehilangan cabang < min_stroke_px
# (spur, 2 per frame) → cakupan turun sampai 73,5% (test frame 148; frame 174: 85,3%, 207: 83,3%). Terbukti: dengan
# min_stroke_px = 1 ketiganya 100%. Distribusi (frame ber-skeleton): test_short 85 frame, p5 97,65%, min 94,94%, 0 frame
# < 90%; test 123 frame, p5 96,26%, min 73,53%, 3 frame < 90%. Batas agregat seluruh klip ketat (≥ 97%; terukur ≥ 99,5%).
COVERAGE_MIN = 70.0
BAND_COVERAGE_MIN = 70.0
COVERAGE_AGGREGATE_MIN = 97.0


@pytest.mark.parametrize("clip", CLIPS)
def test_real_all_frames_aggregate_coverage_no_shadow_no_integrity_errors(clip):
    """Seluruh frame klip: bayangan 0 (jarak ≥ D − toleransi), titik berulang 0, loncatan 0, cakupan agregat ≥ 97%."""
    if not has_real(clip):
        pytest.skip("data nyata tidak ada")
    th, tl = clip_thresholds(clip)
    sk = sk_unc = band = band_unc = bad = 0
    shadow = []
    for i in range(n_frames(clip)):
        g, z = real_frame(clip, i)
        m = sm.occlusion_measure(g, z, NAMES, REAL_PARAMS, th, tl)
        sk, band = sk + m["skeleton"], band + m["band"]
        sk_unc += round(m["skeleton"] * (100 - m["skeleton_coverage"]) / 100)
        band_unc += round(m["band"] * (100 - m["band_coverage"]) / 100)
        bad += m["integrity"]["repeats"] + m["integrity"]["jumps"]
        if m["shadow"] is not None:
            shadow.append(m["shadow"])
    assert bad == 0 and shadow
    assert min(shadow) >= REAL_PARAMS["depth_lines.min_dist_px"] - sm.SHADOW_TOLERANCE_PX, min(shadow)
    assert 100.0 * (1 - sk_unc / sk) >= COVERAGE_AGGREGATE_MIN and 100.0 * (1 - band_unc / band) >= COVERAGE_AGGREGATE_MIN


def lower_clothing_fraction(clip: str, index: int, stroke: dict) -> float:
    cm = cv2.imread(str(real(clip) / "seg" / "classmap" / f"frame_{index:05d}.png"), cv2.IMREAD_UNCHANGED)
    lc = load_class_names().index("Lower_Clothing")
    pts = [(int(y), int(x)) for x, y in stroke["points"]]
    return sum(cm[y, x] == lc for y, x in pts) / len(pts)


@pytest.mark.parametrize("clip", CLIPS)
def test_real_done_when_leg_crossing_has_occlusion_in_lower_clothing(clip):
    """Done-when T-201b (kriteria wilayah, disetujui Rio): pada frame 73 / 78 / 82 / 87 / 92 ada strok oklusi dengan
    ≥ 80% titik di piksel Lower_Clothing (seg/classmap); lulus bila ≥ 4 dari 5 frame. Kaki ada di grup torso."""
    if not has_real(clip):
        pytest.skip("data nyata tidak ada")
    th, tl = clip_thresholds(clip)
    ok = []
    for i in LEG_FRAMES:
        g, z = real_frame(clip, i)
        strokes, _ = vec.occlusion_strokes(g, z, NAMES, REAL_PARAMS, th, tl)
        if any(lower_clothing_fraction(clip, i, s) >= 0.8 for s in strokes):
            ok.append(i)
    assert len(ok) >= 4, ok


@pytest.mark.parametrize("clip", CLIPS)
def test_real_leg_groups_empty_so_legs_only_via_torso(clip):
    """Catatan data (D-009): left_leg / right_leg 0 piksel — Lower_Clothing ada di grup torso."""
    if not has_real(clip):
        pytest.skip("data nyata tidak ada")
    g, _ = real_frame(clip, 80)
    assert not (g >= NAMES.index("left_leg") + 1).any()
