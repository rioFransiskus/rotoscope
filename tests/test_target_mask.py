"""Test target_mask.py + integrasi [3] (T-606, D-012): algoritma H, MaskSet, FrameCache.raw, basi, peringatan, config.

Sintetis, tanpa model / GPU / torch: masker [1b] dibuat oleh stage [1b] ASLI dengan SAM tiruan (`test_isolate.FakeBackend`,
lingkaran berjari-jari r di sekitar klik), jadi kunci basi [1b] sungguhan. Klip: 10 frame, dua adegan (cut sebelum f6). Adegan A
(f0–f5): target T + orang lain N tersambung lengan → SATU blob seg (komponen dipotong); adegan B (f6–f9): T dan N terpisah.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

import test_isolate as ti
import test_stabilize as tst
from rotoscope import isolate as iso
from rotoscope import stabilize as stb
from rotoscope import stage_common
from rotoscope import target_mask as tmk
from rotoscope.config import ConfigError, TargetMaskConfig, load_pipeline, load_style
from rotoscope.depth import write_depth
from rotoscope.ingest import FRAME_PATTERN
from rotoscope.segment import write_classmap, write_probs
from rotoscope.stage_common import StageError

H, W = 40, 60
N, CUT = 10, 6
CID, CLASSES = tst.CID, tst.CLASSES
CLICK = (17, 19)
RADIUS = 16.0
P = TargetMaskConfig()


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setattr(stage_common, "REPLACE_DELAY_S", 0)
    monkeypatch.setenv(stage_common.HF_OFFLINE_ENV, "1")


# ── H murni ────────────────────────────────────────
def test_fill_holes_enclosed_border_diagonal_and_limit():
    m = np.zeros((12, 12), bool)
    m[1:8, 1:8] = True
    m[3:5, 3:6] = False                                   # lubang tertutup 6 px
    m[0:3, 9:12] = True
    m[0:2, 10:12] = False                                 # cekungan menyentuh tepi atas → bukan lubang
    ring = np.zeros((6, 6), bool)
    ring[1:5, 1:5] = True
    ring[2, 2] = ring[3, 3] = False                       # dua piksel latar yang hanya bersentuhan diagonal = 2 lubang 4-arah
    out = tmk.fill_holes(m)
    assert out[3:5, 3:6].all() and not out[0:2, 10:12].any() and (out & ~m).sum() == 6
    assert tmk.fill_holes(ring)[2, 2] and tmk.fill_holes(ring)[3, 3]
    assert tmk.fill_holes(m, 6)[3:5, 3:6].all() and not tmk.fill_holes(m, 5)[3:5, 3:6].any()
    assert tmk.fill_holes(m, 0).sum() == out.sum()        # 0 = tanpa batas


def test_dilate_zero_copy_and_ellipse():
    m = np.zeros((11, 11), bool)
    m[5, 5] = True
    z = tmk.dilate(m, 0)
    assert (z == m).all() and z is not m
    d = tmk.dilate(m, 2)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)) > 0
    assert (d[3:8, 3:8] == ker).all() and d.sum() == ker.sum()
    assert d[5, 7] and not d[7, 7]                        # elips: (2, 0) masuk, sudut (2, 2) tidak


def test_component_class_boundaries():
    p = TargetMaskConfig(r_keep=0.9, r_drop=0.1)
    assert tmk.component_class(0.9, p) == tmk.CLASS_KEEP and tmk.component_class(0.8999, p) == tmk.CLASS_CUT
    assert tmk.component_class(0.1, p) == tmk.CLASS_DROP and tmk.component_class(0.1001, p) == tmk.CLASS_CUT
    assert tmk.component_class(0.0, p) == tmk.CLASS_DROP and tmk.component_class(1.0, p) == tmk.CLASS_KEEP


def _blocks():
    fg = np.zeros((30, 60), bool)
    mask = np.zeros((30, 60), bool)
    fg[2:12, 2:12] = True                                 # A: 100 px, 95 di masker → utuh (termasuk 5 px di luar M_d)
    mask[2:12, 2:12] = True
    mask[2:7, 11] = False                                 # 5 px kolom kanan A di luar masker
    fg[2:12, 20:30] = True                                # B: di luar masker → dibuang
    mask[2:4, 20:23] = True                               # 6 px B di dalam masker (r 0,06 ≤ 0,1)
    fg[16:26, 2:12] = True                                # C: 100 px dalam masker + 100 px di luar, tersambung
    fg[16:26, 12:22] = True
    mask[16:26, 2:12] = True
    return fg, mask


def test_apply_h_keep_whole_drop_whole_cut():
    fg, mask = _blocks()
    p = TargetMaskConfig(dilate_px=0, piece_min_px=1)
    keep, st = tmk.apply_h(fg, mask, p)
    assert st["komponen"] == {"utuh": 1, "potong": 1, "buang": 1}
    assert keep[2:12, 2:12].all()                        # utuh: piksel di luar masker ikut
    assert not keep[2:12, 20:30].any()                   # buang: piksel di dalam masker ikut dibuang
    assert keep[16:26, 2:12].all() and not keep[16:26, 12:22].any()       # potong = C ∩ M_d (d 0)
    keep2, _ = tmk.apply_h(fg, mask, TargetMaskConfig(dilate_px=3, piece_min_px=1))
    assert keep2[16:26, 12:15].all() and not keep2[16:26, 15:22].any()    # d 3: tiga kolom ikut
    assert st["fg_px"] == int(fg.sum()) and st["dibuang_px"] == int((fg & ~keep).sum())


@pytest.mark.parametrize("inside, kept", [(29, False), (30, True)])
def test_apply_h_piece_rule_boundary(inside, kept):
    fg = np.zeros((30, 60), bool)
    mask = np.zeros((30, 60), bool)
    fg[10:20, 2:12] = mask[10:20, 2:12] = True           # bagian target (100 px di masker)
    fg[10:20, 30:50] = True                              # orang lain, tersambung lewat bawah (di luar masker)
    fg[22:24, 2:50] = True
    fg[20:22, 2] = fg[20:22, 49] = True
    fg[2, 20:20 + inside] = mask[2, 20:20 + inside] = True   # potongan kecil: hanya `inside` px di masker
    fg[3:10, 40] = True                                  # tersambung ke orang lain (di luar masker)
    keep, st = tmk.apply_h(fg, mask, TargetMaskConfig(dilate_px=0, piece_min_px=30))
    assert st["komponen"]["potong"] == 1
    assert bool(keep[2, 20:20 + inside].all()) is kept and bool(keep[2, 20:20 + inside].any()) is kept
    assert keep[10:20, 2:12].all() and st["potongan_dibuang"] == (0 if kept else 1)


def test_apply_h_mask_holes_filled_before_r_and_cut():
    fg = np.zeros((30, 60), bool)
    mask = np.zeros((30, 60), bool)
    fg[5:15, 5:15] = mask[5:15, 5:15] = True
    mask[8:12, 8:12] = False                              # lubang masker 16 px, tertutup seg
    fg[5:15, 15:35] = True                                # orang lain tersambung → r 0,33 → dipotong
    keep, st = tmk.apply_h(fg, mask, TargetMaskConfig(dilate_px=0))
    assert st["komponen"]["potong"] == 1 and keep[8:12, 8:12].all() and not keep[5:15, 15:35].any()
    keep_lim, _ = tmk.apply_h(fg, mask, TargetMaskConfig(dilate_px=0, hole_fill_max_px=15))    # 16 px > 15: tidak diisi
    assert not keep_lim[8:12, 8:12].any() and keep_lim[5:8, 5:15].all()
    fg2 = np.zeros((30, 60), bool)
    fg2[5:15, 5:16] = True                                # 100 px di M_f (84 di M) + 10 px di luar
    keep2, st2 = tmk.apply_h(fg2, mask, TargetMaskConfig(dilate_px=0))
    assert st2["komponen"]["utuh"] == 1 and keep2[5:15, 15].all()          # r dari M_f: 100 / 110 ≥ 0,9


def test_apply_h_components_are_8_connected():
    fg = np.zeros((20, 20), bool)
    mask = np.zeros((20, 20), bool)
    fg[2:12, 2:12] = mask[2:12, 2:12] = True
    fg[12, 12] = True                                     # hanya bersentuhan diagonal, di luar masker
    keep, st = tmk.apply_h(fg, mask, P)
    assert st["komponen"] == {"utuh": 1, "potong": 0, "buang": 0} and keep[12, 12]


def test_apply_h_empty_mask_and_empty_fg():
    fg, mask = _blocks()
    keep, st = tmk.apply_h(fg, np.zeros_like(mask), P)
    assert not keep.any() and st["komponen"]["buang"] == 3 and st["dibuang_px"] == int(fg.sum())
    keep, st = tmk.apply_h(np.zeros_like(fg), mask, P)
    assert not keep.any() and st["fg_px"] == 0 and sum(st["komponen"].values()) == 0


def test_to_background_matches_one_hot_probs():
    rng = np.random.default_rng(1)
    cfg = load_pipeline()
    lut = stb.class_group_lut(cfg.groups, CLASSES)
    probs = rng.integers(0, 9, (len(CLASSES), 8, 9)).astype(np.uint8)
    cm = rng.integers(0, len(CLASSES), (8, 9)).astype(np.uint8)
    remove = rng.random((8, 9)) < 0.4
    sums, tie = stb.group_sums(probs, lut, len(cfg.groups)), lut[cm]
    keep_sums, keep_tie = sums.copy(), tie.copy()
    tmk.to_background(sums, tie, remove)
    pr2, cm2 = probs.copy(), cm.copy()
    pr2[:, remove] = 0
    pr2[0, remove] = 255
    cm2[remove] = 0
    assert (sums == stb.group_sums(pr2, lut, len(cfg.groups))).all() and (tie == lut[cm2]).all()
    assert (sums[:, ~remove] == keep_sums[:, ~remove]).all() and (tie[~remove] == keep_tie[~remove]).all()


def test_statuses_match_isolate():
    assert (tmk.STATUS_TARGET, tmk.STATUS_EMPTY, tmk.STATUS_NO_PROMPT) == (iso.STATUS_TARGET, iso.STATUS_EMPTY,
                                                                          iso.STATUS_NO_PROMPT)


# ── MaskSet ────────────────────────────────────────
def _maskset(tmp_path, status, no_prompt="keep", write=True):
    d = tmp_path / "labels"
    d.mkdir(exist_ok=True)
    if write:
        lab = np.zeros((10, 10), np.uint8)
        lab[2:8, 2:8] = 1
        cv2.imwrite(str(d / (FRAME_PATTERN % 0)), lab)
        cv2.imwrite(str(d / (FRAME_PATTERN % 1)), np.zeros((10, 10), np.uint8))
    return tmk.MaskSet(TargetMaskConfig(no_prompt=no_prompt), status, d, {})


def test_needs_and_no_prompt_modes(tmp_path):
    st = {0: "target", 1: "kosong", 2: "tanpa_prompt"}
    keep_ms, clear_ms = _maskset(tmp_path, st), _maskset(tmp_path, st, "clear")
    assert [keep_ms.needs(i) for i in range(4)] == [True, True, False, False]
    assert [clear_ms.needs(i) for i in range(4)] == [True, True, True, False]
    fg = np.zeros((10, 10), bool)
    fg[1:9, 1:9] = True
    sums = np.zeros((8, 10, 10), np.uint16)
    sums[3][fg] = 255
    sums[0][~fg] = 255
    tie = np.where(fg, 3, 0).astype(np.uint8)
    s, t = sums.copy(), tie.copy()
    info = clear_ms.apply(2, s, t, fg)
    assert info["status"] == "tanpa_prompt" and (s[0] == 255).all() and not s[1:].any() and not t.any()
    s, t = sums.copy(), tie.copy()
    info = keep_ms.apply(1, s, t, fg)                     # kosong → semua dibuang
    assert info["status"] == "kosong" and (s[0] == 255).all() and info["dibuang_px"] == 64
    s, t = sums.copy(), tie.copy()
    info = keep_ms.apply(0, s, t, fg)                     # komponen 64 px, 36 di masker → r 0,56 → dipotong (d 4 = semua)
    assert info["komponen"]["potong"] == 1 and info["dibuang_px"] == 0 and (s == sums).all()


def test_missing_label_is_error(tmp_path):
    ms = _maskset(tmp_path, {5: "target"}, write=False)
    with pytest.raises(StageError, match="isolate/labels/frame_00005.png"):
        ms.apply(5, np.zeros((8, 10, 10), np.uint16), np.zeros((10, 10), np.uint8), np.zeros((10, 10), bool))


# ── Klip sintetis + [1b] tiruan ────────────────────
def classmap(i: int) -> np.ndarray:
    cm = np.zeros((H, W), np.uint8)
    s = i % 2                                            # target bergeser 1 px tiap frame
    cm[8:32, 10 + s:26 + s] = CID["Torso"]               # target T (di dalam lingkaran masker)
    cm[8:32, 36:52] = CID["Torso"]                       # orang lain N
    cm[8:13, 38:46] = CID["Face_Neck"]
    if i < CUT:                                          # adegan A: lengan menyambung T dan N → satu blob
        cm[18:22, 26 + s:36] = CID["Left_Lower_Arm"]
    return cm


def probs_for(i: int) -> np.ndarray:
    """Probabilitas dari classmap + pita AMBIGU di bawah N: classmap Background, tetapi jumlah grup Torso 135 > Background 120
    → F (argmax jumlah grup) memuatnya, classmap tidak."""
    pr = tst.probs_from_classmap(classmap(i))
    pr[:, 33:37, 44:56] = 0
    pr[0, 33:37, 44:56] = 120
    pr[CID["Torso"], 33:37, 44:56] = 135
    return pr


def make_clip(tmp_path: Path, name: str = "w", fail: tuple[int, ...] = ()) -> Path:
    work = tst.make_work(tmp_path / name, N, H, W)
    (work / "frames").mkdir()
    for i in range(N):
        stem = f"frame_{i:05d}"
        cm = classmap(i)
        write_classmap(work / "seg" / "classmap" / f"{stem}.png", cm)
        write_probs(work / "seg" / "probs" / f"{stem}.npz", probs_for(i))
        d = np.where(cm != 0, 2.0, 1.0).astype(np.float32) + np.linspace(0, 0.5, W, dtype=np.float32)[None]
        write_depth(work / "depth" / f"{stem}.npy", d.astype(np.float16))
        base = 60 if i < CUT else 200
        img = np.full((H, W, 3), base, np.uint8)
        img[0, i] = base + 9
        cv2.imwrite(str(work / "frames" / f"{stem}.png"), img)
    rows = [{"frame": f"frame_{i:05d}.png", "index": i, "fail_reasons": ["area_vs_median"] if i in fail else []}
            for i in range(N)]
    (work / "qc_report.json").write_text(json.dumps({"stage": "segment_qc", "frames": rows}), encoding="utf-8")
    return work


def cfg_for(work: Path, **over):
    return load_pipeline(overrides={"paths.work_dir": str(work), **over})


def isolate(work: Path, clicks=(0,), empty=(), **kw) -> dict:
    ti.write_points(work, {f: [(1, [(*CLICK, 1)], None)] for f in clicks})
    return iso.run_isolate(cfg_for(work), yes=True, backend_factory=ti.factory(radius={1: RADIUS}, empty=empty),
                           log=ti.quiet, ram_probe=lambda: None, **kw)


def run(work: Path, logs: list | None = None, limit: int | None = None, **over) -> dict:
    return stb.run_stabilize(cfg_for(work, **over), limit=limit, log=(logs.append if logs is not None else tst.quiet))


def groups(work: Path) -> np.ndarray:
    clip = stb.load_clip(work)
    return np.stack([stb.read_groups(clip.groups_path(n)) for n in clip.names])


def stable_bytes(work: Path) -> dict[str, bytes]:
    return {p.relative_to(work).as_posix(): p.read_bytes() for d in ("groups", "depth_smooth")
            for p in sorted((work / "stable" / d).iterdir())}


def manifest(work: Path) -> dict:
    return json.loads((work / "stable" / "manifest.json").read_text(encoding="utf-8"))


@pytest.fixture
def pair(tmp_path):
    """(klip TANPA titik — sudah di-stabilize, klip sama DENGAN titik + masker [1b] berlaku)."""
    base = make_clip(tmp_path, "base")
    run(base)
    work = make_clip(tmp_path, "mask")
    isolate(work)
    return base, work


def test_masks_valid_after_isolate(pair):
    _, work = pair
    ms = tmk.load_masks(cfg_for(work), work, tst.quiet)
    assert ms is not None and ms.status == {i: ("target" if i < CUT else "tanpa_prompt") for i in range(N)}
    assert ms.block["frames"] == {"target": CUT, "kosong": 0, "tanpa_prompt": N - CUT} and ms.block["shots"] == [[0, CUT - 1]]


def test_no_points_no_isolate_key_no_masks_and_not_recomputed(tmp_path, monkeypatch):
    work = make_clip(tmp_path)
    called = []
    monkeypatch.setattr(iso, "frames_fingerprint", lambda *a: called.append(1))     # tanpa titik: [1b] tidak disentuh
    assert tmk.load_masks(cfg_for(work), work) is None and not called
    r = run(work)
    m = manifest(work)
    assert r["processed"] == N and tmk.MANIFEST_KEY not in m
    lines = (work / "stable" / "frames.jsonl").read_text(encoding="utf-8")
    assert '"isolate"' not in lines
    before = stable_bytes(work)
    r2 = run(work, **{"target_mask.dilate_px": 9, "target_mask.no_prompt": "clear"})   # parameter target_mask berubah
    assert r2["processed"] == 0 and not r2["stale"] and stable_bytes(work) == before


def test_masks_only_target_drawn_in_clicked_scene_other_scene_unchanged(pair):
    base, work = pair
    logs = []
    run(work, logs)
    g, g0 = groups(work), groups(base)
    assert any("masker target [1b] dipakai" in x for x in logs)
    assert (g[CUT:] == g0[CUT:]).all()                    # adegan tanpa klik (keep): sama dengan tanpa masker
    assert g0[:CUT, 20, 44].all() and not g[:CUT, :, 42:].any()       # orang lain N hilang di adegan berklik
    assert g0[:, 34, 50].all() and g[CUT:, 34, 50].all()  # pita ambigu (F dari jumlah grup) digambar tanpa masker
    assert (g[:CUT, 9:31, 12:25] == g0[:CUT, 9:31, 12:25]).all()       # badan target tidak berubah
    m = manifest(work)
    assert m[tmk.MANIFEST_KEY]["target_mask"] == {"enabled": True, "r_keep": 0.9, "r_drop": 0.1, "dilate_px": 4,
                                                  "hole_fill_max_px": 0, "piece_min_px": 30, "no_prompt": "keep"}
    assert m[tmk.MANIFEST_KEY]["key_sha256"] == ti.manifest(work)["key_sha256"]


@pytest.mark.parametrize("over", [{}, {"stabilize.temporal.mask_ema_alpha": 0.3, "stabilize.temporal.boil_preserve": 0.0}])
def test_masking_before_kernel_equals_one_hot_inputs(pair, tmp_path, over):
    """Setara prototipe M2-H: seg frame bermasker ditulis sebagai one-hot background lalu [3] TANPA masker → byte-identik
    (stable/ groups + depth_smooth). α 0,3 + boil 0: bobot tetangga > frame tengah — masker yang tidak sampai ke SEMUA frame
    jendela kernel (hanya frame tengah) gagal."""
    _, work = pair
    run(work, **over)
    cfg = cfg_for(work)
    lut = stb.class_group_lut(cfg.groups, CLASSES)
    proto = tmp_path / "proto"
    shutil.copytree(work, proto)
    for d in ("stable", "target", "isolate"):
        shutil.rmtree(proto / d)
    for i in range(CUT):
        stem = f"frame_{i:05d}"
        pr = probs_for(i)
        cm = classmap(i)
        fg = stb.group_argmax(stb.group_sums(pr, lut, len(cfg.groups)), lut[cm])[0] != 0
        keep, _ = tmk.apply_h(fg, ti.label(work, i) != 0, cfg.target_mask)
        rm = fg & ~keep
        pr[:, rm] = 0
        pr[0, rm] = 255
        cm[rm] = 0
        write_probs(proto / "seg" / "probs" / f"{stem}.npz", pr)
        write_classmap(proto / "seg" / "classmap" / f"{stem}.png", cm)
    run(proto, **over)
    assert stable_bytes(proto) == stable_bytes(work)


def test_kosong_frame_draws_nothing(tmp_path):
    work = make_clip(tmp_path)
    isolate(work, empty=(3,))
    assert ti.rows(work)[3]["status"] == "kosong"
    run(work)
    g = groups(work)
    assert not g[3].any() and g[2].any() and g[4].any()


def test_no_prompt_clear_empties_unclicked_scene(pair):
    base, work = pair
    run(work, **{"target_mask.no_prompt": "clear"})
    g = groups(work)
    assert not g[CUT:].any() and g[:CUT].any() and groups(base)[CUT:].any()
    assert manifest(work)[tmk.MANIFEST_KEY]["target_mask"]["no_prompt"] == "clear"


def _change_points(work):
    p = iso.points_path(work)
    p.write_bytes(p.read_bytes() + b"\n")


def _change_frame(work):
    p = work / "frames" / "frame_00009.png"
    img = cv2.imread(str(p))
    img[5, 5] = (1, 2, 3)
    cv2.imwrite(str(p), img)


@pytest.mark.parametrize("breaker, why", [
    (_change_points, "points_sha256"),
    (_change_frame, "frames_pixel_sha256"),
    (lambda w: (w / "isolate" / "manifest.json").unlink(), "manifest.json"),
    (lambda w: (w / "isolate" / "labels" / "frame_00002.png").unlink(), "labels/"),
    (lambda w: shutil.rmtree(w / "isolate"), "isolate/ belum ada"),
])
def test_invalid_masks_warn_and_run_without(pair, breaker, why):
    base, work = pair
    breaker(work)
    logs = []
    run(work, logs)
    warn = [x for x in logs if x.startswith("PERINGATAN: target/points.json ada")]
    assert len(warn) == 1 and why in warn[0] and "isolate" in warn[0] and "--yes" in warn[0]
    assert tmk.MANIFEST_KEY not in manifest(work)
    if breaker is not _change_frame:                      # frame berubah = masukan cut berbeda; cukup tanpa masker
        assert stable_bytes(work) == stable_bytes(base)


def test_target_mask_disabled(pair):
    base, work = pair
    logs = []
    run(work, logs, **{"target_mask.enabled": False})
    assert any("target_mask.enabled = false" in x for x in logs) and not any(x.startswith("PERINGATAN") for x in logs)
    assert tmk.MANIFEST_KEY not in manifest(work) and stable_bytes(work) == stable_bytes(base)


def test_stale_when_params_change_and_when_masks_appear_or_vanish(pair):
    base, work = pair
    run(work, **{"target_mask.enabled": False})
    r = run(work)                                          # masker muncul → basi
    assert r["processed"] == N and any(s.startswith("isolate.") for s in r["stale"])
    r = run(work)
    assert r["processed"] == 0
    r = run(work, **{"target_mask.dilate_px": 6})
    assert r["processed"] == N and any("dilate_px" in s for s in r["stale"])
    r = run(work, **{"target_mask.dilate_px": 6, "target_mask.enabled": False})   # masker hilang → basi
    assert r["processed"] == N and stable_bytes(work) == stable_bytes(base)


def test_stale_when_isolate_labels_change_not_when_recomputed_identical(pair):
    _, work = pair
    run(work)
    isolate(work, restart=True)                            # [1b] dihitung ulang, hasil sama (deterministik) → tidak basi
    assert run(work)["processed"] == 0
    ti.write_points(work, {0: [(1, [(*CLICK, 1)], None)]})
    iso.run_isolate(cfg_for(work), yes=True, restart=True, backend_factory=ti.factory(radius={1: RADIUS - 3}),
                    log=ti.quiet, ram_probe=lambda: None)  # kunci [1b] sama, isi label lain
    r = run(work)
    assert r["processed"] == N and any("isolate.labels_sha256" in s for s in r["stale"])


def test_qc_weights_unchanged(tmp_path):
    base = make_clip(tmp_path, "b", fail=(2,))
    work = make_clip(tmp_path, "m", fail=(2,))
    isolate(work)
    run(base)
    run(work)
    assert manifest(work)["temporal"] == manifest(base)["temporal"]
    assert manifest(work)["temporal"]["qc_fail_frames"] == ["frame_00002.png"]


def test_frames_log_records_isolate_info(pair):
    _, work = pair
    run(work)
    recs = [json.loads(x) for x in (work / "stable" / "frames.jsonl").read_text(encoding="utf-8").splitlines()]
    fr = {r["index"]: r for r in recs if r.get("event") == "frame"}
    assert all(fr[i][tmk.MANIFEST_KEY]["status"] == "target" for i in range(CUT))
    assert all(tmk.MANIFEST_KEY not in fr[i] for i in range(CUT, N))
    k = fr[0][tmk.MANIFEST_KEY]
    assert k["komponen"]["potong"] == 1 and k["dibuang_px"] > 0
    assert fr[CUT - 1][tmk.MANIFEST_KEY]["komponen"]["potong"] == 1


def test_scene_b_click_keeps_target_whole(tmp_path):
    work = make_clip(tmp_path)
    isolate(work, clicks=(0, CUT))
    run(work)
    recs = [json.loads(x) for x in (work / "stable" / "frames.jsonl").read_text(encoding="utf-8").splitlines()]
    fr = {r["index"]: r for r in recs if r.get("event") == "frame"}
    assert fr[CUT][tmk.MANIFEST_KEY]["komponen"] == {"utuh": 1, "potong": 0, "buang": 2}      # N + pita ambigu
    g = groups(work)
    assert g[CUT:, 20, 18].all() and not g[CUT:, :, 34:].any()


def test_limit_prefix_identical_with_masks(pair, tmp_path):
    _, work = pair
    other = tmp_path / "other"
    shutil.copytree(work, other)
    run(work)
    run(other, limit=4)
    run(other)
    assert stable_bytes(other) == stable_bytes(work)


# ── Config ─────────────────────────────────────────
def test_defaults_are_rio_decision():
    assert load_pipeline().target_mask == TargetMaskConfig(True, 0.9, 0.1, 4, 0, 30, "keep")


@pytest.mark.parametrize("over, match", [
    ({"target_mask.r_keep": 0.0}, "r_keep"), ({"target_mask.r_keep": 1.01}, "r_keep"),
    ({"target_mask.r_drop": -0.01}, "r_drop"), ({"target_mask.r_drop": 1.0}, "r_drop"),
    ({"target_mask.r_drop": 0.5, "target_mask.r_keep": 0.5}, r"r_drop.*harus <"),
    ({"target_mask.dilate_px": -1}, "dilate_px"), ({"target_mask.dilate_px": 33}, "dilate_px"),
    ({"target_mask.dilate_px": 2.5}, "dilate_px"), ({"target_mask.hole_fill_max_px": -1}, "hole_fill_max_px"),
    ({"target_mask.piece_min_px": 0}, "piece_min_px"), ({"target_mask.no_prompt": "hapus"}, "no_prompt"),
    ({"target_mask.enabled": "ya"}, "enabled"),
])
def test_config_rejects(over, match):
    with pytest.raises(ConfigError, match=match):
        load_pipeline(overrides=over)


def test_config_accepts_edges_and_style_rejects_section():
    c = load_pipeline(overrides={"target_mask.r_keep": 1.0, "target_mask.r_drop": 0.0, "target_mask.dilate_px": 0,
                                 "target_mask.no_prompt": "clear"}).target_mask
    assert (c.r_keep, c.r_drop, c.dilate_px, c.no_prompt) == (1.0, 0.0, 0, "clear")
    with pytest.raises(ConfigError, match="parameter pipeline"):
        load_style(overrides={"target_mask.r_keep": 0.5})


def test_no_torch_import():
    code = "import sys, rotoscope.target_mask, rotoscope.stabilize; sys.exit(1 if 'torch' in sys.modules else 0)"
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0
