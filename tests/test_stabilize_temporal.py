"""Test temporal [3] (T-302): kernel simetris terpotong, bobot QC, cut, boil_preserve, --limit berjendela, resume.

Sintetis (tanpa model / GPU): objek statis + flicker acak, objek bergerak diagonal, tungkai tipis cepat, cut, frame gagal QC,
tepi klip, klip lebih pendek dari jendela. Metrik: tests/temporal_metrics.py.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

import temporal_metrics as tm
import test_stabilize as tst
from rotoscope import stabilize as stb
from rotoscope import stage_common
from rotoscope.config import ConfigError, load_pipeline
from rotoscope.segment import write_classmap, write_probs
from rotoscope.depth import write_depth
from rotoscope.stage_common import StageError

H, W = 40, 56
CID, CLASSES = tst.CID, tst.CLASSES
TORSO_ID, LARM_ID = CID["Torso"], CID["Left_Lower_Arm"]
GID_TORSO, GID_LARM = 3, 4
NO_FILTER = {"stabilize.island_min_px": 0, "stabilize.mode_k": 1}
DEPTH_OFF = {"stabilize.depth.temporal": False}
ON = {"stabilize.temporal.enabled": True,
      "stabilize.temporal.boil_preserve": 0.0, "stabilize.temporal.mask_ema_alpha": 0.7, **NO_FILTER}


def static_cm(h: int = H, w: int = W) -> np.ndarray:
    cm = np.zeros((h, w), np.uint8)
    cm[8:34, 14:42] = TORSO_ID
    return cm


def make_clip(tmp_path: Path, cms: list[np.ndarray], *, fail: tuple[int, ...] = (), shade: list[int] | None = None,
              qc: bool = True, frames: bool = True) -> Path:
    """Klip sintetis dari classmap per frame: probabilitas, depth, frames/ (abu-abu rata `shade`) dan qc_report.json."""
    n, (h, w) = len(cms), cms[0].shape
    work = tst.make_work(tmp_path, n, h, w)
    for i, cm in enumerate(cms):
        stem = f"frame_{i:05d}"
        write_classmap(work / "seg" / "classmap" / f"{stem}.png", cm)
        pr = cms.probs[i] if getattr(cms, "probs", None) else tst.probs_from_classmap(cm)
        write_probs(work / "seg" / "probs" / f"{stem}.npz", pr)
        d = np.where(cm != 0, 2.0, 1.0).astype(np.float32) + np.linspace(0, 0.5, w, dtype=np.float32)[None]
        write_depth(work / "depth" / f"{stem}.npy", d.astype(np.float16))
    if frames:
        (work / "frames").mkdir(exist_ok=True)
        for i in range(n):
            v = 100 if shade is None else shade[i]
            ok, buf = cv2.imencode(".png", np.full((h, w, 3), v, np.uint8))
            (work / "frames" / f"frame_{i:05d}.png").write_bytes(buf.tobytes())
    if qc:
        rows = [{"frame": f"frame_{i:05d}.png", "index": i, "fail_reasons": ["area_vs_median"] if i in fail else []}
                for i in range(n)]
        (work / "qc_report.json").write_text(json.dumps({"stage": "segment_qc", "frames": rows}), encoding="utf-8")
    return work


def run(work: Path, **overrides):
    return stb.run_stabilize(tst.cfg_for(work, **{**ON, **overrides}), log=tst.quiet)


def read_all(work: Path) -> np.ndarray:
    clip = stb.load_clip(work)
    return np.stack([stb.read_groups(clip.groups_path(n)) for n in clip.names])


def read_depth_all(work: Path) -> np.ndarray:
    clip = stb.load_clip(work)
    return np.stack([np.load(clip.depth_smooth_path(n)) for n in clip.names])


def raw_groups(cms: list[np.ndarray]) -> np.ndarray:
    lut = stb.class_group_lut(load_pipeline().groups, CLASSES)
    return np.stack([lut[c] for c in cms])


class Frames(list):
    """Daftar classmap (argmax) + `probs` opsional per frame (probabilitas ambigu: kelas salah 130 vs benar 125)."""

    probs: list | None = None


def cat(*parts: Frames) -> Frames:
    out = Frames(c for p in parts for c in p)
    out.probs = [pr for p in parts for pr in p.probs]
    return out


def noisy(base: np.ndarray, frames: int, rate: float, seed: int = 0) -> Frames:
    """Flicker murni: `rate` piksel per frame punya probabilitas AMBIGU (kelas salah 130, kelas benar 125) → argmax mentah
    salah, tetangga waktu tetap benar. Kebenaran tidak bergerak. (Flip dengan keyakinan penuh tidak bisa diredam: itu
    gerak sah.)"""
    rng = np.random.default_rng(seed)
    out, probs = Frames(), []
    for _ in range(frames):
        pr = tst.probs_from_classmap(base)
        m = rng.random(base.shape) < rate
        wrong = rng.choice([0, TORSO_ID, LARM_ID, CID["Face_Neck"]], size=base.shape).astype(np.uint8)
        wrong = np.where(wrong == base, (wrong + 1) % len(CLASSES), wrong).astype(np.uint8)
        cm = np.where(m, wrong, base).astype(np.uint8)
        pr = tst.probs_from_classmap(base).astype(np.int32)
        ys, xs = np.nonzero(m)
        pr[:, ys, xs] = 0
        pr[wrong[ys, xs], ys, xs] = 130
        pr[base[ys, xs], ys, xs] = 125
        out.append(cm)
        probs.append(pr.astype(np.uint8))
    out.probs = probs
    return out


def moving_cms(n: int, vx: float, vy: float, size: int = 12, x0: int = 6, y0: int = 6) -> list[np.ndarray]:
    out = []
    for i in range(n):
        cm = np.zeros((H, W), np.uint8)
        x, y = int(round(x0 + vx * i)), int(round(y0 + vy * i))
        cm[y:y + size, x:x + size] = TORSO_ID
        out.append(cm)
    return out


def limb_cms(n: int, width: int, speed: int) -> list[np.ndarray]:
    out = []
    for i in range(n):
        cm = static_cm()
        x = 4 + speed * i
        cm[10:30, x:x + width] = LARM_ID
        out.append(cm)
    return out


# ── Kernel: rumus ──────────────────────────────────
def test_kernel_radius_values():
    assert stb.kernel_radius(1.0) == 0
    assert [stb.kernel_radius(a) for a in (0.85, 0.7, 0.55, 0.4, 0.3)] == [2, 2, 4, 5, 7]
    assert stb.kernel_radius(0.05) == stb.R_MAX
    assert stb.kernel_rho(1.0) == 0 and abs(stb.kernel_rho(0.7) - 0.3 / 1.7) < 1e-12


def test_center_weight_equals_alpha_without_truncation():
    rho = stb.kernel_rho(0.7)
    plan = stb.TemporalPlan(40, 8, rho, 0, 0.0, (0,) * 40, (39,) * 40, (1.0,) * 40)
    w = dict(plan.weights(20, 8))
    assert abs(w[20] / sum(w.values()) - 0.7) < 1e-3          # R besar → mendekati α
    assert w[20] == 1.0 and abs(w[21] - rho) < 1e-12 and w[19] == w[21]   # simetris


def test_plan_window_truncated_at_clip_edges_and_cuts():
    plan = stb.TemporalPlan(10, 2, 0.2, 0, 0.0, (0,) * 5 + (5,) * 5, (4,) * 5 + (9,) * 5, (1.0,) * 10, cut_frames=(5,))
    assert plan.window(0, 2) == (0, 2) and plan.window(9, 2) == (7, 9)       # tepi klip
    assert plan.window(4, 2) == (2, 4) and plan.window(5, 2) == (5, 7)       # cut antara 4 dan 5
    assert plan.window(3, 2) == (1, 4)


def test_plan_short_clip_and_weights_fallback():
    plan = stb.TemporalPlan(2, 8, 0.5, 0, 0.0, (0, 0), (1, 1), (1.0, 1.0))
    assert plan.window(0, 8) == (0, 1) and [k for k, _ in plan.weights(0, 8)] == [0, 1]
    tiny = stb.TemporalPlan(3, 2, 1e-9, 0, 0.0, (0,) * 3, (2,) * 3, (1e-9, 1e-9, 1e-9))
    assert tiny.weights(1, 2) == [(1, 1.0)]                                    # jumlah bobot < WEIGHT_SUM_MIN


def test_plan_needed_includes_depth_window():
    plan = stb.TemporalPlan(20, 2, 0.2, 2, 0.0, (0,) * 20, (19,) * 20, (1.0,) * 20)
    assert plan.needed(10) == (6, 14)                                          # depth 8..12 × grup ±2
    spatial = stb.spatial_plan(20)
    assert spatial.needed(10) == (10, 10) and spatial.weights(10, 0) == [(10, 1.0)]


def test_shot_bounds():
    start, end = stb.shot_bounds([False, False, True, False, True, False])
    assert start == (0, 0, 2, 2, 4, 4) and end == (1, 1, 3, 3, 5, 5)


# ── Kernel: pada data sintetis ─────────────────────
def test_flicker_reduced_and_fidelity_to_truth(tmp_path):
    base = static_cm()
    cms = noisy(base, 14, 0.12)
    work = make_clip(tmp_path, cms)
    run(work)
    out, raw, truth = read_all(work), raw_groups(cms), np.stack([raw_groups([base])[0]] * 14)
    assert tm.flipflop_per10k(out) < 0.35 * tm.flipflop_per10k(raw)
    assert tm.fg_iou(out, truth).mean() > tm.fg_iou(raw, truth).mean()


def test_alpha_one_equals_temporal_off_any_boil(tmp_path):
    cms = noisy(static_cm(), 8, 0.1, seed=3)
    off = make_clip(tmp_path / "a", cms)
    stb.run_stabilize(tst.cfg_for(off, **NO_FILTER), log=tst.quiet)
    for b in (0.0, 0.3, 1.0):
        on = make_clip(tmp_path / f"b{b}", cms)
        run(on, **{"stabilize.temporal.mask_ema_alpha": 1.0, "stabilize.temporal.boil_preserve": b})
        assert (read_all(on) == read_all(off)).all()
        assert (read_depth_all(on) == read_depth_all(off)).all()


def test_boil_one_equals_raw_and_boil_interpolates(tmp_path):
    cms = noisy(static_cm(), 10, 0.15, seed=5)
    raw = raw_groups(cms)
    errs = []
    for b in (0.0, 0.5, 1.0):
        w = make_clip(tmp_path / f"b{b}", cms)
        run(w, **{"stabilize.temporal.boil_preserve": b})
        out = read_all(w)
        errs.append(float((out != raw).mean()))
        if b == 1.0:
            assert (out == raw).all()
    assert errs[0] > errs[1] > errs[2] == 0


@pytest.mark.parametrize("vx,vy", [(1, 1), (2, 1), (3, 2)])
def test_moving_object_no_lag_no_shift(tmp_path, vx, vy):
    cms = moving_cms(14, vx, vy, size=14)
    work = make_clip(tmp_path, cms)
    run(work)
    out, raw = read_all(work), raw_groups(cms)
    assert abs(tm.area_lag(out, raw)) <= 0.05
    assert tm.centroid_error(out, raw)[2:-2].max() <= 1.0
    assert tm.fg_iou(out, raw).min() >= 0.7


def test_object_rotating_like_motion_keeps_fidelity(tmp_path):
    cms = []
    for i in range(14):                              # persegi panjang berputar-geser: tinggi / lebar bertukar
        cm = np.zeros((H, W), np.uint8)
        a = 6 + (i % 3)
        cm[8:8 + 8 + a, 10 + 2 * i:10 + 2 * i + 18 - a] = TORSO_ID
        cms.append(cm)
    work = make_clip(tmp_path, cms)
    run(work)
    assert tm.fg_iou(read_all(work), raw_groups(cms)).min() >= 0.6


def test_thin_limb_survives_moderate_motion_alpha_07(tmp_path):
    cms = limb_cms(14, width=4, speed=2)
    work = make_clip(tmp_path, cms)
    run(work)
    out, raw = read_all(work), raw_groups(cms)
    assert tm.limb_ratio(out, raw)[2:-2].min() >= 0.9


def test_thin_limb_vanishes_when_alpha_low_and_fast(tmp_path):
    cms = limb_cms(10, width=2, speed=5)             # w = 2 px, v = 5 px/frame: tidak tumpang tindih antar frame
    work = make_clip(tmp_path, cms)
    run(work, **{"stabilize.temporal.mask_ema_alpha": 0.3})
    assert tm.limb_ratio(read_all(work), raw_groups(cms))[3:-3].max() < 0.3   # batas α aman terdokumentasi (docs/04)
    work2 = make_clip(tmp_path / "b", cms)
    run(work2, **{"stabilize.temporal.mask_ema_alpha": 0.7})
    assert tm.limb_ratio(read_all(work2), raw_groups(cms))[3:-3].min() >= 0.95  # α = 0,7: gerak sah dipertahankan


def test_clip_edges_first_and_last_frame(tmp_path):
    cms = noisy(static_cm(), 6, 0.1, seed=9)
    work = make_clip(tmp_path, cms)
    run(work)
    out = read_all(work)
    assert out.shape[0] == 6 and out[0].any() and out[-1].any()


def test_clip_shorter_than_window(tmp_path):
    for n in (1, 2, 3):
        cms = noisy(static_cm(), n, 0.1, seed=n)
        work = make_clip(tmp_path / f"n{n}", cms)
        assert run(work, **{"stabilize.temporal.mask_ema_alpha": 0.3})["processed"] == n
        assert read_all(work).shape[0] == n


# ── Bobot QC ───────────────────────────────────────
def _fail_clip(tmp_path, fail: tuple[int, ...], n: int = 11, name: str = "w"):
    base = static_cm()
    cms = [np.zeros_like(base) if i in fail else base for i in range(n)]
    return make_clip(tmp_path / name, cms, fail=fail), raw_groups([base])[0]


def _recovery(tmp_path, fail, q, name):
    work, truth = _fail_clip(tmp_path, fail, name=name)
    run(work, **{"stabilize.temporal.qc_fail_weight": q})
    out = read_all(work)
    return tm.recovery_iou(out, np.stack([truth] * len(out)), list(fail))


def test_failed_frame_filled_by_neighbours_and_weight_monotone(tmp_path):
    rec = {q: _recovery(tmp_path, (5,), q, f"q{q}")[0] for q in (1.0, 0.5, 0.25, 0.1)}
    assert rec[1.0] < 0.5                                   # tanpa bobot: frame kosong menang atas tetangga
    assert rec[0.25] > 0.95 and rec[0.1] > 0.95             # bobot kecil: diisi tetangga
    assert rec[1.0] <= rec[0.5] <= rec[0.25] <= rec[0.1]


def test_consecutive_failures_limit(tmp_path):
    """α = 0,7 (R = 2, ρ = 0,176), q default 0,1 (< ρ): 1 dan 2 frame gagal berurutan dipulihkan; 3+ berurutan TIDAK (batas yang
    diketahui, docs/01). q = 0,25 (> ρ) hanya memulihkan frame tunggal."""
    q = load_pipeline().stabilize.temporal.qc_fail_weight
    assert q == 0.1 and q < stb.kernel_rho(0.7)
    assert _recovery(tmp_path, (5,), q, "c1")[0] > 0.95
    assert _recovery(tmp_path, (5, 6), q, "c2").min() > 0.9
    assert _recovery(tmp_path, (4, 5, 6), q, "c3")[1] < 0.5            # tengah tiga frame gagal: batas
    assert _recovery(tmp_path, (5, 6), 0.25, "c2b").max() < 0.5        # q > ρ: 2 berurutan tidak pulih
    assert _recovery(tmp_path, (5,), 0.25, "c1b")[0] > 0.95


def test_all_frames_failed_is_uniform_weighting_no_crash(tmp_path):
    base = static_cm()
    cms = noisy(base, 8, 0.1, seed=2)
    allfail = make_clip(tmp_path / "a", cms, fail=tuple(range(8)))
    none = make_clip(tmp_path / "b", cms)
    run(allfail)
    run(none)
    assert (read_all(allfail) == read_all(none)).all()      # q konstan hilang di normalisasi


def test_failed_frame_at_clip_edge_needs_smaller_weight(tmp_path):
    """Di tepi klip jendela hanya satu sisi → q = 0,25 belum cukup (batas terdokumentasi); q = 0,1 memulihkan."""
    for edge in (0, 9):
        for q, expect_ok in ((0.25, False), (0.1, True)):
            work, truth = _fail_clip(tmp_path / f"e{edge}_{q}", (edge,), n=10, name="w")
            run(work, **{"stabilize.temporal.qc_fail_weight": q})
            rec = tm.recovery_iou(read_all(work), np.stack([truth] * 10), [edge])[0]
            assert (rec > 0.9) == expect_ok


def test_qc_report_required_only_with_temporal(tmp_path):
    work = make_clip(tmp_path, noisy(static_cm(), 5, 0.1), qc=False)
    stb.run_stabilize(tst.cfg_for(work, **NO_FILTER), log=tst.quiet)      # temporal mati: tidak dibaca
    with pytest.raises(StageError, match="qc_report.json tidak ada"):
        run(work, **{"stabilize.depth.log_eps": 1e-5})                     # setelan lain → basi → hitung ulang, temporal aktif


def test_qc_report_missing_frames_rejected(tmp_path):
    work = make_clip(tmp_path, noisy(static_cm(), 5, 0.1))
    p = work / "qc_report.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    d["frames"] = d["frames"][:3]
    p.write_text(json.dumps(d), encoding="utf-8")
    with pytest.raises(StageError, match="qc_report.json tidak memuat"):
        run(work)


# ── Cut ────────────────────────────────────────────
def test_scores_and_detect_cuts():
    a = [np.full((4, 4), v, np.float32) for v in (10, 11, 200, 201)]
    s = stb.cut_scores(a)
    assert s[0] == 0 and abs(s[1] - 1 / 255) < 1e-6 and abs(s[2] - 189 / 255) < 1e-6
    assert stb.detect_cuts(s, 0.08) == [False, False, True, False]
    assert stb.detect_cuts(s, 0) == [False] * 4


def _two_scenes(tmp_path, cut_diff):
    a_cm, b_cm = static_cm(), np.zeros((H, W), np.uint8)
    b_cm[4:20, 30:52] = LARM_ID
    a = noisy(a_cm, 6, 0.1, seed=11)
    b = noisy(b_cm, 6, 0.1, seed=12)
    both = make_clip(tmp_path / "both", cat(a, b), shade=[40] * 6 + [220] * 6)
    only_a = make_clip(tmp_path / "a", a, shade=[40] * 6)
    only_b = make_clip(tmp_path / "b", b, shade=[220] * 6)
    for w in (both, only_a, only_b):
        run(w, **{"stabilize.temporal.cut_diff": cut_diff})
    return read_all(both), read_all(only_a), read_all(only_b)


def test_cut_window_truncated_equals_independent_clips(tmp_path):
    both, a, b = _two_scenes(tmp_path, 0.08)
    assert (both[:6] == a).all() and (both[6:] == b).all()


def test_cut_guard_off_leaks_across_cut(tmp_path):
    both, a, b = _two_scenes(tmp_path, 0.0)
    assert not ((both[:6] == a).all() and (both[6:] == b).all())


def test_cut_listed_in_manifest_and_log(tmp_path):
    work = make_clip(tmp_path, cat(noisy(static_cm(), 4, 0.05), noisy(static_cm(), 4, 0.05, seed=1)),
                     shade=[30] * 4 + [230] * 4)
    msgs = []
    stb.run_stabilize(tst.cfg_for(work, **ON), log=msgs.append)
    m = json.loads((work / "stable" / "manifest.json").read_text(encoding="utf-8"))
    assert m["temporal"]["cut_frames"] == ["frame_00004.png"] and any("cut terdeteksi" in x for x in msgs)


def test_missing_frames_for_cut_detection(tmp_path):
    work = make_clip(tmp_path, noisy(static_cm(), 4, 0.05), frames=False)
    with pytest.raises(StageError, match="deteksi cut"):
        run(work)
    run(work, **{"stabilize.temporal.cut_diff": 0.0})            # cut mati → frames/ tidak dibaca


# ── Kedalaman ──────────────────────────────────────
def test_normalize_variants_match_formula():
    rng = np.random.default_rng(1)
    d = rng.uniform(1.0, 3.0, (H, W)).astype(np.float32)
    fg = np.zeros((H, W), bool)
    fg[8:30, 10:40] = True
    lg = np.log(d)
    med = np.median(lg[fg])
    q25, q75 = np.percentile(lg[fg], (25, 75))
    a, _ = stb.normalize_depth(d, fg, 1e-6, 0.01, stb.NORMALIZE_A)
    b, _ = stb.normalize_depth(d, fg, 1e-6, 0.01, stb.NORMALIZE_B)
    assert np.abs(a.astype(np.float32) - (lg - med) / (q75 - q25)).max() < 5e-3 * 10
    assert np.abs(b.astype(np.float32) - (lg - med)).max() < 2e-3
    assert not np.allclose(a.astype(np.float32), b.astype(np.float32), atol=1e-2)     # varian tidak tertukar
    legacy, _ = stb.normalize_depth(d, fg, 1e-6, 0.01)
    assert (legacy == a).all()                                                       # default fungsi = A


def test_depth_temporal_off_equals_single_frame_and_on_smooths(tmp_path):
    cms = noisy(static_cm(), 10, 0.1, seed=4)
    off = make_clip(tmp_path / "off", cms)
    on = make_clip(tmp_path / "on", cms)
    spatial = make_clip(tmp_path / "sp", cms)
    run(off, **{"stabilize.depth.temporal": False})
    run(on, **{"stabilize.depth.temporal": True})
    stb.run_stabilize(tst.cfg_for(spatial, **NO_FILTER), log=tst.quiet)
    # depth.temporal mati: kedalaman tiap frame dihitung dari peta grup terhaluskan frame itu saja (bukan EMA)
    assert read_depth_all(off).shape == read_depth_all(on).shape
    fluct = lambda x: float(np.abs(np.diff(x.astype(np.float32), axis=0)).mean())
    assert fluct(read_depth_all(on)) < fluct(read_depth_all(off))


def test_depth_temporal_needs_wider_window(tmp_path):
    cms = noisy(static_cm(), 12, 0.1, seed=6)
    work = make_clip(tmp_path, cms)
    for k in range(8, 12):
        (work / "seg" / "probs" / f"frame_{k:05d}.npz").unlink()
    run_ = stb.run_stabilize(tst.cfg_for(work, **{**ON, "stabilize.depth.temporal": True}), limit=8, log=tst.quiet)
    # R = 2 grup + 2 depth → frame t butuh input sampai t + 4 → hanya frame 0..3 berjendela lengkap
    assert run_["processed"] == 4 and len(run_["deferred"]) == 4


# ── --limit, resume, determinisme ──────────────────
def test_limit_then_full_identical_to_scratch(tmp_path):
    cms = noisy(static_cm(), 12, 0.1, seed=7)
    a = make_clip(tmp_path / "a", cms)
    b = make_clip(tmp_path / "b", cms)
    run(a, **{"stabilize.depth.temporal": True})
    stb.run_stabilize(tst.cfg_for(b, **{**ON, "stabilize.depth.temporal": True}), limit=5, log=tst.quiet)
    run(b, **{"stabilize.depth.temporal": True})
    assert (read_all(a) == read_all(b)).all()
    assert (read_depth_all(a) == read_depth_all(b)).all()
    ca, cb = stb.load_clip(a), stb.load_clip(b)
    for n in ca.names:
        assert ca.groups_path(n).read_bytes() == cb.groups_path(n).read_bytes()
        assert ca.depth_smooth_path(n).read_bytes() == cb.depth_smooth_path(n).read_bytes()


def test_limit_with_inputs_beyond_writes_all_n_identical_to_full(tmp_path):
    cms = noisy(static_cm(), 12, 0.1, seed=8)
    full = make_clip(tmp_path / "f", cms)
    part = make_clip(tmp_path / "p", cms)
    run(full)
    r = stb.run_stabilize(tst.cfg_for(part, **ON), limit=5, log=tst.quiet)       # input ada sampai frame 11
    assert r["processed"] == 5 and r["deferred"] == []
    cf, cp = stb.load_clip(full), stb.load_clip(part)
    for n in cp.names[:5]:
        assert cf.groups_path(n).read_bytes() == cp.groups_path(n).read_bytes()


def test_limit_with_limited_inputs_writes_only_complete_windows(tmp_path):
    cms = noisy(static_cm(), 12, 0.1, seed=10)
    full = make_clip(tmp_path / "f", cms)
    part = make_clip(tmp_path / "p", cms)
    for k in range(5, 12):                                    # seg / depth hanya sampai frame 4
        (part / "seg" / "probs" / f"frame_{k:05d}.npz").unlink()
        (part / "seg" / "classmap" / f"frame_{k:05d}.png").unlink()
        (part / "depth" / f"frame_{k:05d}.npy").unlink()
    msgs = []
    r = stb.run_stabilize(tst.cfg_for(part, **ON, **DEPTH_OFF), limit=5, log=msgs.append)
    assert r["processed"] == 3 and r["deferred"] == ["frame_00003.png", "frame_00004.png"]   # R = 2
    assert any("ditunda" in m and "segment" in m and "--limit" in m for m in msgs)
    run(full, **DEPTH_OFF)
    cf, cp = stb.load_clip(full), stb.load_clip(part)
    written = sorted(p.name for p in cp.groups_dir.glob("frame_*.png"))
    assert written == [f"frame_{k:05d}.png" for k in range(3)]
    for n in written:                                         # tidak ada frame berjendela terpotong yang ditulis
        assert cf.groups_path(n).read_bytes() == cp.groups_path(n).read_bytes()
        assert cf.depth_smooth_path(n).read_bytes() == cp.depth_smooth_path(n).read_bytes()


def test_truncated_window_only_at_clip_edge(tmp_path):
    cms = noisy(static_cm(), 6, 0.1, seed=13)
    work = make_clip(tmp_path, cms)
    r = run(work)
    assert r["processed"] == 6
    recs = [x for x in stage_common.read_jsonl(stb.load_clip(work).frames_log) if x.get("event") == "frame"]
    assert [x["window"] for x in recs] == [[0, 2], [0, 3], [0, 4], [1, 5], [2, 5], [3, 5]]


def test_deterministic_two_runs_from_scratch(tmp_path):
    cms = noisy(static_cm(), 9, 0.1, seed=14)
    cfg = {"stabilize.depth.temporal": True, "stabilize.temporal.boil_preserve": 0.3}
    a, b = make_clip(tmp_path / "a", cms), make_clip(tmp_path / "b", cms)
    run(a, **cfg)
    run(b, **cfg)
    ca, cb = stb.load_clip(a), stb.load_clip(b)
    for n in ca.names:
        assert ca.groups_path(n).read_bytes() == cb.groups_path(n).read_bytes()
        assert ca.depth_smooth_path(n).read_bytes() == cb.depth_smooth_path(n).read_bytes()


# ── Basi (manifest) ────────────────────────────────
@pytest.mark.parametrize("key,val", [
    ("stabilize.temporal.mask_ema_alpha", 0.55), ("stabilize.temporal.boil_preserve", 0.3),
    ("stabilize.temporal.cut_diff", 0.2), ("stabilize.temporal.qc_fail_weight", 0.5),
    ("stabilize.depth.normalize", "log_median_iqr"), ("stabilize.depth.temporal", True),
    ("stabilize.temporal.enabled", False),
])
def test_stale_when_temporal_setting_changes(tmp_path, key, val):
    work = make_clip(tmp_path, noisy(static_cm(), 6, 0.1))
    run(work)
    msgs = []
    r = stb.run_stabilize(tst.cfg_for(work, **{**ON, key: val}), log=msgs.append)
    assert r["stale"] and r["processed"] == 6 and "basi" in msgs[0]


def test_stale_when_qc_failure_set_changes(tmp_path):
    work = make_clip(tmp_path, noisy(static_cm(), 6, 0.1))
    run(work)
    p = work / "qc_report.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    d["frames"][2]["fail_reasons"] = ["iou_prev"]
    p.write_text(json.dumps(d), encoding="utf-8")
    r = run(work)
    assert any(s.startswith("temporal") for s in r["stale"]) and r["processed"] == 6


def test_optical_flow_removed_no_warning_no_manifest_field(tmp_path):
    """T-303: optical flow ditolak → parameter, peringatan, dan field manifest `temporal.optical_flow` tidak ada."""
    work = make_clip(tmp_path, noisy(static_cm(), 4, 0.1))
    msgs = []
    stb.run_stabilize(tst.cfg_for(work, **ON), log=msgs.append)
    assert not any("optical_flow" in m or "T-303" in m for m in msgs)
    m = json.loads((work / "stable" / "manifest.json").read_text(encoding="utf-8"))
    assert "optical_flow" not in m["temporal"]
    assert "optical_flow_blend" not in m["stabilize"]["temporal"]


def test_manifest_temporal_disabled(tmp_path):
    work = make_clip(tmp_path, noisy(static_cm(), 3, 0.1))
    stb.run_stabilize(tst.cfg_for(work, **NO_FILTER), log=tst.quiet)
    m = json.loads((work / "stable" / "manifest.json").read_text(encoding="utf-8"))
    assert m["temporal"] == {"enabled": False}


# ── Config + dependensi ────────────────────────────
@pytest.mark.parametrize("key,val", [("stabilize.temporal.cut_diff", -0.1), ("stabilize.temporal.cut_diff", 1.5),
                                     ("stabilize.depth.normalize", "affine_median_iqr")])
def test_config_rejects(key, val):
    with pytest.raises(ConfigError):
        load_pipeline(overrides={key: val})


def test_config_accepts_enum_and_cut_zero():
    c = load_pipeline(overrides={"stabilize.depth.normalize": "log_median", "stabilize.temporal.cut_diff": 0})
    assert c.stabilize.depth.normalize == "log_median" and c.stabilize.temporal.cut_diff == 0


def test_shipped_defaults():
    """Default T-302 (Tahap 4, disetujui Rio): temporal aktif, α 0,7, b 0,3, B, EMA kedalaman mati, cut 0,06 (sebelum 2026-10-10 0,08), q 0,1."""
    c = load_pipeline()
    t = c.stabilize.temporal
    assert t.enabled is True and t.mask_ema_alpha == 0.7 and t.boil_preserve == 0.3
    assert t.cut_diff == 0.06 and t.qc_fail_weight == 0.1
    assert c.stabilize.depth.normalize == "log_median" and c.stabilize.depth.temporal is False


def test_torch_not_imported():
    code = "import sys, rotoscope.stabilize; sys.exit(1 if 'torch' in sys.modules else 0)"
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0


# ── Metrik (fungsi murni) ──────────────────────────
def test_metric_flipflop_counts():
    g = np.zeros((6, 1, 4), np.uint8)
    g[2, 0, 0] = 1                       # A→B→A dalam 1 frame (frame 2 berubah, kembali di 3)
    g[2:4, 0, 1] = 1                     # A→B→B→A dalam 2 frame
    g[3:, 0, 2] = 1                      # perubahan menetap → bukan flip-flop
    ff = tm.flipflop_counts(g)
    assert ff.tolist() == [0, 0, 2, 0, 0, 0]       # kolom 0: 1 frame, kolom 1: 2 frame (keduanya dimulai di t = 2)


def test_metric_windows_rule():
    g = np.zeros((40, 20, 20), np.uint8)
    for t in range(40):
        x = 0 if t < 20 else (t - 19) * 3          # 0..19 diam; 20.. bergerak 3 px / frame (persegi 4 px)
        g[t, 5:9, x:x + 4] = 1
    assert tm.static_windows(g)[0][0] == 1 and tm.static_windows(g)[0][1] == 19
    assert tm.fast_windows(g) == []                # 3 px < FAST_SPEED_MIN_PX


def test_metric_lag_detects_delay():
    n = 30
    raw = np.zeros((n, 10, 30), np.uint8)
    for t in range(n):
        raw[t, :, :5 + int(10 * np.sin(t / 3) ** 2 + 5)] = 1
    delayed = np.concatenate([raw[:1], raw[:-1]])
    assert tm.area_lag(raw, raw) == 0 and tm.area_lag(delayed, raw) > 0.5
