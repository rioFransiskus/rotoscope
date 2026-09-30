"""Test stabilize.py (T-106): probabilitas sintetis, tanpa model/GPU."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rotoscope import stabilize as stb
from rotoscope import stage_common
from rotoscope.config import load_class_names, load_pipeline
from rotoscope.depth import write_depth
from rotoscope.segment import write_classmap, write_probs
from rotoscope.stage_common import StageError

H, W = 20, 16
N_FRAMES = 4
CLASSES = load_class_names()
CID = {c: i for i, c in enumerate(CLASSES)}
# Grup default: 1 hair, 2 face, 3 torso, 4 left_arm, 5 right_arm, 6 left_leg, 7 right_leg
BG, HAIR, FACE, TORSO, LARM = 0, 1, 2, 3, 4


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setattr(stage_common, "REPLACE_DELAY_S", 0)


def probs_from_classmap(cm: np.ndarray, p: int = 230) -> np.ndarray:
    """Probabilitas uint8: kelas classmap = p, sisanya dibagi rata ke Background."""
    probs = np.zeros((len(CLASSES), *cm.shape), np.uint8)
    np.put_along_axis(probs, cm[None].astype(np.intp), p, axis=0)
    probs[0] += np.where(cm == 0, 0, 255 - p).astype(np.uint8)
    return probs


def subject_classmap(i: int, h: int = H, w: int = W) -> np.ndarray:
    """Badan Torso + lengan kiri + kepala Face_Neck, bergeser 1 px per frame."""
    cm = np.zeros((h, w), np.uint8)
    cm[6:18, 4 + i % 2:12 + i % 2] = CID["Torso"]
    cm[8:16, 2 + i % 2:4 + i % 2] = CID["Left_Lower_Arm"]
    cm[2:6, 6:10] = CID["Face_Neck"]
    return cm


def make_work(tmp_path: Path, n: int = N_FRAMES, h: int = H, w: int = W) -> Path:
    work = tmp_path / "work"
    (work / "seg" / "classmap").mkdir(parents=True)
    (work / "seg" / "probs").mkdir(parents=True)
    (work / "depth").mkdir(parents=True)
    (work / "meta.json").write_text(json.dumps({"frame_count": n, "working_width": w, "working_height": h,
                                                "frame_index_start": 0}), encoding="utf-8")
    size = {"width": w, "height": h}
    (work / "seg" / "manifest.json").write_text(json.dumps(
        {"stage": "segment", "model": "0.8b", "model_id": "facebook/sapiens2-seg-0.8b", "revision": "a" * 40,
         "precision": "fp16", "processor": {}, "num_labels": len(CLASSES), "frame_size": size,
         "classes": list(CLASSES), "created_utc": "t0"}), encoding="utf-8")
    (work / "depth" / "manifest.json").write_text(json.dumps(
        {"stage": "depth", "model_id": "depth-anything/Depth-Anything-V2-Small-hf", "revision": "b" * 40,
         "license": "apache-2.0", "precision": "fp32", "processor": {}, "input_size": {}, "output": {},
         "frame_size": size, "created_utc": "t0"}), encoding="utf-8")
    for i in range(n):
        name = f"frame_{i:05d}"
        cm = subject_classmap(i, h, w)
        write_classmap(work / "seg" / "classmap" / f"{name}.png", cm)
        write_probs(work / "seg" / "probs" / f"{name}.npz", probs_from_classmap(cm))
        d = np.where(cm != 0, 2.0, 1.0).astype(np.float32) + np.linspace(0, 0.5, w, dtype=np.float32)[None]
        write_depth(work / "depth" / f"{name}.npy", d.astype(np.float16))
        stage_common.append_jsonl(work / "depth" / "frames.jsonl",
                                  {"event": "frame", "frame": f"{name}.png", "finite": True})
    return work


def cfg_for(work: Path, **overrides):
    return load_pipeline(overrides={"paths.work_dir": str(work), **overrides})


def quiet(_msg: str) -> None:
    pass


# ── Langkah 1 + 3 ──────────────────────────────────
def test_class_group_lut_default_groups():
    cfg = load_pipeline()
    lut = stb.class_group_lut(cfg.groups, CLASSES)
    assert lut[CID["Background"]] == BG
    assert lut[CID["Hair"]] == HAIR
    assert lut[CID["Lower_Clothing"]] == TORSO
    assert lut[CID["Left_Hand"]] == LARM
    assert int(lut.max()) == len(cfg.groups)


def test_group_sums_and_argmax():
    lut = np.array([0, 1, 1, 2], np.uint8)
    probs = np.zeros((4, 1, 3), np.uint8)
    probs[:, 0, 0] = [100, 80, 75, 0]     # grup 1 = 155 menang atas background 100
    probs[:, 0, 1] = [0, 0, 5, 250]
    probs[:, 0, 2] = [200, 30, 0, 25]
    sums = stb.group_sums(probs, lut, 2)
    assert sums.dtype == np.uint16
    assert sums[:, 0, 0].tolist() == [100, 155, 0]
    gmap, n_tie = stb.group_argmax(sums, np.zeros((1, 3), np.uint8))
    assert gmap.tolist() == [[1, 2, 0]] and n_tie == 0


def test_group_sums_exact_at_max():
    lut = np.ones(29, np.uint8)
    lut[0] = 0
    probs = np.full((29, 1, 1), 255, np.uint8)
    assert stb.group_sums(probs, lut, 1)[1, 0, 0] == 28 * 255


def test_tie_uses_classmap_group():
    sums = np.array([[[127, 127, 100]], [[127, 127, 50]], [[1, 1, 100]]], np.uint16)  # (3, 1, 3)
    tie_groups = np.array([[1, 2, 2]], np.uint8)
    gmap, n_tie = stb.group_argmax(sums, tie_groups)
    # px0: seri 0/1, classmap 1 → 1. px1: seri 0/1, classmap 2 tidak ikut seri → id terkecil 0.
    # px2: seri 0/2, classmap 2 → 2.
    assert gmap.tolist() == [[1, 0, 2]]
    assert n_tie == 3


# ── Langkah 4: filter pulau ────────────────────────
def test_island_below_n_replaced_by_ring_majority():
    g = np.full((10, 10), TORSO, np.uint8)
    g[4:6, 4:6] = LARM                     # 4 px < 5
    out = stb.island_filter(g, 5)
    assert (out == TORSO).all()


def test_island_at_n_kept():
    g = np.full((10, 10), TORSO, np.uint8)
    g[4:6, 4:7] = LARM                     # 6 px = N
    assert (stb.island_filter(g, 6) == g).all()


def test_small_background_hole_filled():
    g = np.zeros((12, 12), np.uint8)
    g[2:10, 2:10] = TORSO
    g[5, 5] = BG                           # lubang 1 px di badan
    out = stb.island_filter(g, 3)
    assert out[5, 5] == TORSO
    assert (out[0] == BG).all()            # background besar tidak disentuh


def test_island_ring_tie_lowest_id():
    g = np.zeros((3, 4), np.uint8)
    g[:, :2] = TORSO                       # 6 px
    g[:, 2:] = LARM                        # 6 px
    g[1, 1] = FACE                         # pulau 1 px; cincin: torso 5, left_arm 3 → torso
    assert stb.island_filter(g, 2)[1, 1] == TORSO
    g3 = np.array([[TORSO, FACE, LARM]], np.uint8)   # cincin FACE: torso 1, left_arm 1 → seri → id kecil
    assert stb.island_filter(g3, 2)[0, 1] == TORSO


def test_island_touching_edge_uses_clipped_ring():
    g = np.full((8, 8), TORSO, np.uint8)
    g[0, 0] = LARM                         # di pojok: cincin hanya 3 piksel di dalam gambar
    assert stb.island_filter(g, 2)[0, 0] == TORSO


def test_island_single_pass_neighbours():
    g = np.full((8, 8), TORSO, np.uint8)
    g[3, 3] = LARM
    g[3, 4] = FACE                         # dua pulau 1 px bertetangga — cincin dari peta ASLI
    out = stb.island_filter(g, 2)
    assert out[3, 3] == TORSO and out[3, 4] == TORSO


def test_island_whole_image_and_off():
    g = np.full((4, 4), TORSO, np.uint8)
    assert (stb.island_filter(g, 100) == g).all()   # cincin kosong → tidak diubah
    g[1, 1] = LARM
    assert (stb.island_filter(g, 0) == g).all()


# ── Langkah 5: mode filter ─────────────────────────
def test_mode_filter_removes_one_px_spur():
    g = np.zeros((9, 9), np.uint8)
    g[2:7, 2:7] = TORSO
    g[4, 7] = TORSO                        # tonjolan 1 px
    out = stb.mode_filter(g, 3)
    assert out[4, 7] == BG
    assert out[4, 4] == TORSO


def test_mode_filter_tie_keeps_original():
    # jendela tengah: torso 4, left_arm 4 (termasuk asli), bg 1 → seri → grup asli (bukan id terkecil)
    g = np.array([[TORSO, TORSO, LARM],
                  [TORSO, LARM, LARM],
                  [TORSO, BG, LARM]], np.uint8)
    assert stb.mode_filter(g, 3)[1, 1] == LARM


def test_mode_filter_tie_between_others_lowest_id():
    # piksel tengah FACE; jendela: torso 4, left_arm 4, face 1 → seri torso/left_arm → id terkecil
    g = np.array([[TORSO, TORSO, TORSO],
                  [TORSO, FACE, LARM],
                  [LARM, LARM, LARM]], np.uint8)
    assert stb.mode_filter(g, 3)[1, 1] == TORSO


def test_mode_filter_k1_identity():
    g = np.random.default_rng(0).integers(0, 8, (6, 7)).astype(np.uint8)
    assert (stb.mode_filter(g, 1) == g).all()


# ── Langkah 6: kedalaman ───────────────────────────
def _fg_mask():
    fg = np.zeros((H, W), bool)
    fg[4:16, 4:12] = True
    return fg


def test_normalize_median_zero_iqr_one_on_foreground():
    rng = np.random.default_rng(1)
    d = rng.uniform(1.0, 3.0, (H, W)).astype(np.float32)
    fg = _fg_mask()
    out, st = stb.normalize_depth(d, fg, 1e-6, 0.01)
    assert out.dtype == np.float16 and np.isfinite(out).all()
    v = out.astype(np.float64)[fg]
    assert abs(np.median(v)) < 1e-2
    assert abs(np.percentile(v, 75) - np.percentile(v, 25) - 1) < 1e-2
    assert st["region"] == "foreground" and not st["iqr_clamped"]


def test_normalize_scale_invariant():
    rng = np.random.default_rng(2)
    d = rng.uniform(1.0, 3.0, (H, W)).astype(np.float32)
    fg = _fg_mask()
    a, _ = stb.normalize_depth(d, fg, 1e-6, 0.01)
    b, _ = stb.normalize_depth(d * 7.5, fg, 1e-6, 0.01)
    assert np.abs(a.astype(np.float32) - b.astype(np.float32)).max() < 1e-2


def test_normalize_background_same_transform():
    d = np.full((H, W), 1.0, np.float32)
    fg = _fg_mask()
    d[fg] = np.linspace(2.0, 4.0, int(fg.sum()))
    out, st = stb.normalize_depth(d, fg, 1e-6, 0.01)
    expected_bg = (np.log(1.0) - st["log_median"]) / st["log_iqr"]
    assert np.allclose(out[~fg].astype(np.float32), expected_bg, atol=2e-2)


def test_normalize_nonpositive_uses_eps_and_clamp():
    d = np.full((H, W), 2.0, np.float32)
    d[0, 0] = 0.0                          # frame NaN dari [2c] diganti 0
    fg = _fg_mask()
    out, st = stb.normalize_depth(d, fg, 1e-6, 0.01)
    assert st["iqr_clamped"] and st["log_iqr"] == 0
    assert np.isfinite(out).all()
    assert out[0, 0] == np.float16((np.log(1e-6) - np.log(2.0)) / 0.01)


def test_normalize_empty_foreground_uses_frame():
    d = np.random.default_rng(3).uniform(1.0, 2.0, (H, W)).astype(np.float32)
    out, st = stb.normalize_depth(d, np.zeros((H, W), bool), 1e-6, 0.01)
    assert st["region"] == "frame"
    assert abs(np.median(out.astype(np.float64))) < 1e-2


def test_normalize_extreme_values_fit_float16():
    d = np.full((H, W), 65504.0, np.float32)
    d[_fg_mask()] = 0.0
    d[0, :] = 1e-12
    out, _ = stb.normalize_depth(d, _fg_mask(), 1e-12, 1e-3)
    assert np.isfinite(out).all()


# ── Run end-to-end ─────────────────────────────────
def test_run_writes_outputs_manifest_and_log(tmp_path):
    work = make_work(tmp_path)
    cfg = cfg_for(work)
    run = stb.run_stabilize(cfg, log=quiet)
    assert run["processed"] == N_FRAMES and run["skipped"] == 0
    clip = stb.load_clip(work)
    for n in clip.names:
        g = stb.read_groups(clip.groups_path(n))
        assert g.shape == (H, W) and set(np.unique(g)) <= {BG, FACE, TORSO, LARM}
        assert stb.depth_smooth_valid(clip.depth_smooth_path(n), H, W)
    m = json.loads(clip.manifest_path.read_text(encoding="utf-8"))
    assert m["stage"] == "stabilize" and len(m["groups_hash"]) == 64 and len(m["stabilize_hash"]) == 64
    assert m["seg"]["model"] == "0.8b" and m["depth"]["license"] == "apache-2.0"
    assert m["stabilize"]["depth"]["log_eps"] == 1e-6
    recs = [r for r in stage_common.read_jsonl(clip.frames_log) if r.get("event") == "frame"]
    assert len(recs) == N_FRAMES
    assert {"tie_px", "island_changed_px", "mode_changed_px", "fg_px", "depth_smooth"} <= set(recs[0])
    assert recs[0]["depth_finite"] is True


def test_groups_match_classmap_for_clean_input(tmp_path):
    work = make_work(tmp_path)
    stb.run_stabilize(cfg_for(work, **{"stabilize.island_min_px": 0, "stabilize.mode_k": 1}), log=quiet)
    clip = stb.load_clip(work)
    lut = stb.class_group_lut(load_pipeline().groups, CLASSES)
    g = stb.read_groups(clip.groups_path(clip.names[0]))
    assert (g == lut[subject_classmap(0)]).all()


def test_resume_skips_valid_and_redoes_corrupt(tmp_path):
    work = make_work(tmp_path)
    cfg = cfg_for(work)
    assert stb.run_stabilize(cfg, limit=2, log=quiet)["processed"] == 2
    clip = stb.load_clip(work)
    clip.depth_smooth_path(clip.names[1]).write_bytes(b"rusak")
    run = stb.run_stabilize(cfg, log=quiet)
    assert run["skipped"] == 1 and run["processed"] == N_FRAMES - 1
    assert [r["frame"] for r in run["frames"]][0] == clip.names[1]


def test_rerun_deterministic(tmp_path):
    work = make_work(tmp_path)
    cfg = cfg_for(work)
    stb.run_stabilize(cfg, log=quiet)
    clip = stb.load_clip(work)
    a = [clip.groups_path(n).read_bytes() + clip.depth_smooth_path(n).read_bytes() for n in clip.names]
    stb.run_stabilize(cfg, restart=True, log=quiet)
    b = [clip.groups_path(n).read_bytes() + clip.depth_smooth_path(n).read_bytes() for n in clip.names]
    assert a == b


@pytest.mark.parametrize("change", ["param", "groups", "seg_input", "depth_input"])
def test_stale_manifest_recomputed_with_warning(tmp_path, change):
    work = make_work(tmp_path)
    stb.run_stabilize(cfg_for(work), log=quiet)
    overrides = {}
    if change == "param":
        overrides = {"stabilize.island_min_px": 10}
    elif change == "groups":
        groups = {name: list(cls) for name, cls in load_pipeline().groups}
        groups["face"] = groups["face"] + groups.pop("hair")
        overrides = {"groups": groups}
    else:
        p = work / ("seg" if change == "seg_input" else "depth") / "manifest.json"
        m = json.loads(p.read_text(encoding="utf-8"))
        m["created_utc"] = "t1"
        p.write_text(json.dumps(m), encoding="utf-8")
    msgs = []
    run = stb.run_stabilize(cfg_for(work, **overrides), log=msgs.append)
    assert run["processed"] == N_FRAMES and run["stale"]
    assert "basi" in msgs[0]
    expect = {"param": "stabilize_hash", "groups": "groups_hash", "seg_input": "seg.created_utc",
              "depth_input": "depth.created_utc"}[change]
    assert any(s.startswith(expect) for s in run["stale"])
    m = json.loads((work / "stable" / "manifest.json").read_text(encoding="utf-8"))
    if change == "param":
        assert m["stabilize"]["island_min_px"] == 10


def test_stale_diff_shows_short_hashes():
    old = {"stabilize_hash": "a" * 64, "seg": {"model": "0.8b", "revision": "x"}}
    new = {"stabilize_hash": "b" * 64, "seg": {"model": "0.4b", "revision": "x"}}
    diff = stb.manifest_diff(old, new)
    assert diff == ["stabilize_hash: " + "a" * 12 + " → " + "b" * 12, "seg.model: '0.8b' → '0.4b'"]


def test_same_manifest_resumes_without_delete(tmp_path):
    work = make_work(tmp_path)
    stb.run_stabilize(cfg_for(work), log=quiet)
    run = stb.run_stabilize(cfg_for(work, **{"vectorize.min_region_area": 5}), log=quiet)  # section lain
    assert run["processed"] == 0 and run["skipped"] == N_FRAMES and run["stale"] == []


def test_outputs_without_manifest_rejected(tmp_path):
    work = make_work(tmp_path)
    stb.run_stabilize(cfg_for(work), log=quiet)
    (work / "stable" / "manifest.json").unlink()
    with pytest.raises(StageError, match="tanpa manifest.json"):
        stb.run_stabilize(cfg_for(work), log=quiet)
    assert stb.run_stabilize(cfg_for(work), restart=True, log=quiet)["processed"] == N_FRAMES


def test_restart_removes_stable_and_tmp_cleaned(tmp_path):
    work = make_work(tmp_path)
    stb.run_stabilize(cfg_for(work), log=quiet)
    junk = work / "stable" / "groups" / "frame_00000.png.tmp"
    junk.write_bytes(b"x")
    stb.run_stabilize(cfg_for(work), log=quiet)
    assert not junk.exists()
    run = stb.run_stabilize(cfg_for(work), restart=True, log=quiet)
    assert run["processed"] == N_FRAMES


# ── Input kurang / salah ───────────────────────────
@pytest.mark.parametrize("missing, match", [
    ("seg/probs/frame_00002.npz", r"seg/probs belum lengkap.*frame_00002.*\[2\] segment"),
    ("seg/classmap/frame_00001.png", r"seg/classmap belum lengkap"),
    ("depth/frame_00003.npy", r"depth belum lengkap.*\[2c\] depth"),
    ("seg/manifest.json", r"seg.manifest\.json tidak ada.*\[2\] segment"),
    ("depth/manifest.json", r"depth.manifest\.json tidak ada.*\[2c\] depth"),
])
def test_missing_inputs_rejected(tmp_path, missing, match):
    work = make_work(tmp_path)
    (work / missing).unlink()
    with pytest.raises(StageError, match=match):
        stb.run_stabilize(cfg_for(work), log=quiet)


def test_limit_only_needs_selected_inputs(tmp_path):
    work = make_work(tmp_path)
    (work / "depth" / "frame_00003.npy").unlink()
    assert stb.run_stabilize(cfg_for(work), limit=2, log=quiet)["processed"] == 2


def test_frame_size_mismatch_rejected(tmp_path):
    work = make_work(tmp_path)
    p = work / "depth" / "manifest.json"
    m = json.loads(p.read_text(encoding="utf-8"))
    m["frame_size"] = {"width": 99, "height": H}
    p.write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(StageError, match="frame_size"):
        stb.run_stabilize(cfg_for(work), log=quiet)


def test_class_list_mismatch_rejected(tmp_path):
    work = make_work(tmp_path)
    p = work / "seg" / "manifest.json"
    m = json.loads(p.read_text(encoding="utf-8"))
    m["classes"] = list(reversed(m["classes"]))
    p.write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(StageError, match="daftar kelas"):
        stb.run_stabilize(cfg_for(work), log=quiet)


def test_corrupt_probs_rejected(tmp_path):
    work = make_work(tmp_path)
    (work / "seg" / "probs" / "frame_00001.npz").write_bytes(b"rusak")
    with pytest.raises(StageError, match=r"seg/probs frame_00001 rusak"):
        stb.run_stabilize(cfg_for(work), log=quiet)


def test_temporal_enabled_rejected(tmp_path):
    work = make_work(tmp_path)
    with pytest.raises(StageError, match="T-302"):
        stb.run_stabilize(cfg_for(work, **{"stabilize.temporal.enabled": True}), log=quiet)


def test_limit_must_be_positive(tmp_path):
    with pytest.raises(StageError, match="--limit"):
        stb.run_stabilize(cfg_for(make_work(tmp_path)), limit=0, log=quiet)


# ── Entry point ────────────────────────────────────
def _write_cfg(tmp_path: Path, work: Path) -> Path:
    p = tmp_path / "cfg.yaml"
    p.write_text(f"paths:\n  work_dir: '{work.as_posix()}'\n", encoding="utf-8")
    return p


def test_main_exit_ok(tmp_path, capsys):
    work = make_work(tmp_path)
    assert stb.main(["--config", str(_write_cfg(tmp_path, work)), "--limit", "1"]) == stage_common.EXIT_OK
    assert "selesai: 1 diproses" in capsys.readouterr().out


def test_main_exit_precondition(tmp_path, capsys):
    work = make_work(tmp_path)
    (work / "seg" / "manifest.json").unlink()
    assert stb.main(["--config", str(_write_cfg(tmp_path, work))]) == stage_common.EXIT_PRECONDITION
    assert "ERROR" in capsys.readouterr().err
