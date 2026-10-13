"""Test QC stage [2] (T-305d): aturan `big_blobs` (transient / legacy / report), QC_VERSION, dan rantai basi.

Backend palsu (test_segment / test_depth), tanpa GPU/model. Fungsi aturan dipanggil lewat modul (`seg.…`) supaya
mutation check (scripts/t305d_mutations.py) yang mem-monkeypatch modul ikut terdeteksi.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import qc_oracle
import test_depth as td
import test_segment as ts
import test_stabilize as tst
import test_stabilize_temporal as tsm
from rotoscope import cli
from rotoscope import segment as seg
from rotoscope import stabilize as stb
from rotoscope.config import ConfigError, load_pipeline
from rotoscope.depth import run_depth
from rotoscope.segment import run_qc, run_segment


def gpu_cfg(work: Path, **overrides):
    """Config untuk [2] + [2c] di work sintetis (revisi palsu kedua model) + override qc.*."""
    ov = {"paths.work_dir": str(work), "segment.revision": dict(ts.REV), "depth.revision": td.REV}
    ov.update(overrides)
    return load_pipeline(overrides=ov)


def qc_cfg(**over):
    return load_pipeline(overrides={f"qc.{k}": v for k, v in over.items()}).qc


def tree_sha(root: Path) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def transient(n, min_run=4):
    return seg.transient_big_blobs(list(n), min_run)


def failing(flags) -> list[int]:
    return [i for i, f in enumerate(flags) if f]


def reference_transient(n: list[int], min_run: int) -> list[bool]:
    """Definisi T305d-prompt 5.2 kandidat C, ditulis ulang langsung (O(R²)) sebagai pembanding acak."""
    runs, s = [], 0
    for i in range(1, len(n) + 1):
        if i == len(n) or n[i] != n[s]:
            runs.append((s, i, n[s]))
            s = i
    long_k = [k for k, (a, b, _) in enumerate(runs) if b - a >= min_run]
    out = [False] * len(n)
    for k, (a, b, v) in enumerate(runs):
        if b - a >= min_run or not long_k:
            continue
        before = [runs[j][2] for j in long_k if j < k]
        after = [runs[j][2] for j in long_k if j > k]
        ref = ([before[-1]] if before else []) + ([after[0]] if after else [])
        if v > max(ref):
            out[a:b] = [True] * (b - a)
    return out


# ── Aturan transient pada deret n sintetis ─────────
@pytest.mark.parametrize("k", [1, 2, 3])
def test_spike_shorter_than_n_fails(k):
    n = [1] * 10 + [2] * k + [1] * 10
    assert failing(transient(n)) == list(range(10, 10 + k))


@pytest.mark.parametrize("k", [4, 5, 9])
def test_run_of_at_least_n_is_long_and_passes(k):
    assert failing(transient([1] * 10 + [2] * k + [1] * 10)) == []


def test_run_exactly_n_counts_as_long_reference():
    # run n=1 tepat 4 frame di kedua sisi = run panjang (≥ N, bukan > N) → lonjakan di antaranya gagal
    assert failing(transient([1] * 4 + [3] * 2 + [1] * 4)) == [4, 5]


def test_compares_with_max_of_both_sides():
    # 2 di antara run panjang 3 (sebelum) dan 1 (sesudah): 2 > max(3, 1)? tidak → lolos (min akan menggagalkannya)
    assert failing(transient([3] * 6 + [2] * 2 + [1] * 6)) == []
    assert failing(transient([1] * 6 + [2] * 2 + [3] * 6)) == []
    assert failing(transient([1] * 6 + [4] * 2 + [3] * 6)) == [6, 7]


def test_nearest_long_run_not_farther_one():
    # sebelum frame 14: run panjang terdekat = 1 (f6–13), bukan 3 (f0–5); sesudah: 1 (f16–23)
    n = [3] * 6 + [1] * 8 + [2] * 2 + [1] * 8
    assert failing(transient(n)) == [14, 15]


def test_only_one_side_at_clip_start_and_end():
    assert failing(transient([2] * 2 + [1] * 10)) == [0, 1]           # hanya run panjang sesudah
    assert failing(transient([1] * 10 + [2] * 3)) == [10, 11, 12]     # hanya run panjang sebelum
    assert failing(transient([2] + [1] * 8 + [3])) == [0, 9]          # tepi klip, satu frame


def test_one_side_only_value_is_used_even_if_higher():
    # awal klip: lonjakan 2 lalu run panjang 3 → 2 > 3? tidak → lolos
    assert failing(transient([2] * 2 + [3] * 10)) == []


def test_no_long_run_passes_everything():
    assert failing(transient([1, 2, 1, 2, 3, 1, 3, 2])) == []
    assert failing(transient([])) == []


def test_transient_drop_does_not_fail():
    assert failing(transient([2] * 6 + [1] * 2 + [2] * 6)) == []      # dua orang bersentuhan sesaat
    assert failing(transient([3] * 6 + [0] * 1 + [3] * 6)) == []


def test_several_short_runs_between_long_runs():
    assert failing(transient([1] * 5 + [2, 3, 2] + [1] * 5)) == [5, 6, 7]
    assert failing(transient([1] * 5 + [2, 2, 1, 2] + [1] * 5)) == [5, 6, 8]


def test_min_run_parameter():
    n = [1] * 10 + [2] * 5 + [1] * 10
    assert failing(transient(n, 6)) == list(range(10, 15)) and failing(transient(n, 5)) == []
    assert failing(transient([1] * 3 + [2] + [1] * 3, 2)) == [3]


def test_matches_reference_definition_on_random_sequences():
    rng = np.random.default_rng(305402)
    for _ in range(400):
        n = []
        while len(n) < 40:
            n += [int(rng.integers(0, 4))] * int(rng.integers(1, 8))
        for m in (2, 3, 4, 6):
            assert transient(n, m) == reference_transient(n, m), (n, m)


def test_count_runs():
    assert seg.count_runs([1, 1, 2, 2, 2, 1]) == [(0, 2, 1), (2, 5, 2), (5, 6, 1)]
    assert seg.count_runs([]) == []


# ── compute_qc per aturan ──────────────────────────
def blobs_cm(k: int, h: int = 100, w: int = 100) -> np.ndarray:
    """k blob besar terpisah, masing-masing 60 × 10 px = 6% frame 100 × 100 (> blob_min 5%)."""
    cm = np.zeros((h, w), np.uint8)
    for j in range(k):
        cm[5:65, 5 + j * 30:15 + j * 30] = 1
    return cm


def rows_for(ks: list[int], **qc_over) -> list[dict]:
    cms = [blobs_cm(k) for k in ks]
    names = [f"frame_{i:05d}.png" for i in range(len(cms))]
    over = {"area_median_window": 3, "iou_min": 0.0, "area_min": 0.0, "area_drop_min": 0.0, **qc_over}
    return seg.compute_qc(cms, qc_cfg(**over), names, list(range(len(cms))))


def test_default_rule_is_transient_n4():
    q = load_pipeline().qc
    assert q.big_blobs_rule == "transient" and q.transient_min_run == 4 and q.max_big_blobs == 1


def test_default_two_people_long_run_passes_spike_fails():
    ks = [2] * 8 + [3] + [2] * 8                 # dua orang sepanjang klip, satu frame lonjakan
    rows = rows_for(ks)
    assert [r["big_blobs"] for r in rows] == ks
    assert [i for i, r in enumerate(rows) if r["fail_reasons"]] == [8]
    assert rows[8]["fail_reasons"] == ["big_blobs"]


def test_legacy_rule_is_old_behaviour():
    rows = rows_for([2] * 8 + [3] + [2] * 8, big_blobs_rule="legacy")
    assert all(r["fail_reasons"] == ["big_blobs"] for r in rows)
    rows = rows_for([1, 2, 1], big_blobs_rule="legacy", max_big_blobs=2)
    assert all(r["fail_reasons"] == [] for r in rows)


def test_report_rule_never_fails_on_big_blobs_but_reports_n():
    ks = [1] * 6 + [3] + [1] * 6 + [2] * 2
    rows = rows_for(ks, big_blobs_rule="report")
    assert [r["big_blobs"] for r in rows] == ks
    assert all("big_blobs" not in r["fail_reasons"] for r in rows)
    assert seg.summarize_qc(rows)["fail_counts"]["big_blobs"] == 0


def test_other_metrics_unaffected_by_rule():
    cms = [blobs_cm(1)] * 4 + [np.zeros((100, 100), np.uint8)] + [blobs_cm(1)] * 4
    names = [f"frame_{i:05d}.png" for i in range(len(cms))]
    out = {}
    for rule in ("transient", "legacy", "report"):
        out[rule] = seg.compute_qc(cms, qc_cfg(big_blobs_rule=rule, area_median_window=3), names, list(range(9)))
    assert out["transient"] == out["legacy"] == out["report"]
    assert out["report"][4]["fail_reasons"] == ["area_ratio", "iou_prev", "area_vs_median"]


# ── legacy = perilaku sebelum T-305d (oracle) ──────
def random_clip(seed: int, n: int = 40, h: int = 60, w: int = 80) -> list[np.ndarray]:
    """Klip acak: 0–3 blob besar / kecil, frame kosong, kelas campur — memicu semua metrik QC."""
    rng = np.random.default_rng(seed)
    cms = []
    for _ in range(n):
        cm = np.zeros((h, w), np.uint8)
        for _ in range(int(rng.integers(0, 4))):
            bh, bw = int(rng.integers(2, 30)), int(rng.integers(2, 25))
            y, x = int(rng.integers(0, h - bh)), int(rng.integers(0, w - bw))
            cm[y:y + bh, x:x + bw] = int(rng.integers(1, 29))
        cms.append(cm)
    return cms


@pytest.mark.parametrize("seed", [1, 2, 3])
@pytest.mark.parametrize("window", [3, 49])
def test_legacy_rows_and_summary_identical_to_pre_t305d(seed, window):
    cms = random_clip(seed)
    names = [f"frame_{i:05d}.png" for i in range(len(cms))]
    log = {names[5]: {"finite": False, "argmax_mismatch_pct": 0.5}, names[6]: {"finite": True}}
    q = qc_cfg(big_blobs_rule="legacy", area_median_window=window)
    new = seg.compute_qc(cms, q, names, list(range(len(cms))), log)
    old = qc_oracle.compute_qc(cms, q, names, list(range(len(cms))), log)
    assert any("big_blobs" in r["fail_reasons"] for r in old)            # oracle memang menguji big_blobs
    assert json.dumps(new) == json.dumps(old)                              # byte-identik (urutan key + nilai)
    assert json.dumps(seg.summarize_qc(new)) == json.dumps(qc_oracle.summarize_qc(old))


def test_transient_differs_from_legacy_only_in_big_blobs_reason():
    cms = random_clip(4)
    names = [f"frame_{i:05d}.png" for i in range(len(cms))]
    old = qc_oracle.compute_qc(cms, qc_cfg(), names, list(range(len(cms))))
    new = seg.compute_qc(cms, qc_cfg(), names, list(range(len(cms))))
    for a, b in zip(old, new):
        assert {k: v for k, v in a.items() if k != "fail_reasons"} == {k: v for k, v in b.items() if k != "fail_reasons"}
        assert [x for x in a["fail_reasons"] if x != "big_blobs"] == [x for x in b["fail_reasons"] if x != "big_blobs"]


def test_run_qc_report_version_rule_and_legacy_identity(tmp_path, monkeypatch):
    monkeypatch.setenv(seg.HF_OFFLINE_ENV, "1")
    work = ts.make_work(tmp_path)
    run_segment(gpu_cfg(work), backend_factory=ts.factory(), log=ts.quiet)
    msgs = []
    rep = run_qc(gpu_cfg(work), log=msgs.append)
    assert rep["qc_version"] == seg.QC_VERSION == 2
    assert rep["qc"]["big_blobs_rule"] == "transient" and rep["qc"]["transient_min_run"] == 4
    assert rep["definitions"]["big_blobs"] == seg.BIG_BLOBS_DEFINITIONS["transient"]
    assert "aturan big_blobs: transient" in msgs[-1]
    on_disk = json.loads((work / "qc_report.json").read_text(encoding="utf-8"))
    assert on_disk["qc_version"] == 2

    leg = run_qc(gpu_cfg(work, **{"qc.big_blobs_rule": "legacy"}), log=ts.quiet)
    clip = seg.load_clip(work)
    cms = [seg.read_classmap(clip.classmap_path(n)) for n in clip.names]
    old = qc_oracle.compute_qc(cms, gpu_cfg(work).qc, clip.names, clip.indices,
                               seg.last_frame_records(clip.frames_log))
    assert json.dumps(leg["frames"]) == json.dumps(old)
    assert json.dumps(leg["summary"]) == json.dumps(qc_oracle.summarize_qc(old))
    assert leg["definitions"]["big_blobs"] == seg.BIG_BLOBS_DEFINITIONS["legacy"]


# ── Validasi parameter ─────────────────────────────
@pytest.mark.parametrize("over,match", [
    ({"qc.big_blobs_rule": "auto"}, r"qc\.big_blobs_rule.*'transient', 'legacy', 'report'"),
    ({"qc.big_blobs_rule": "Transient"}, r"qc\.big_blobs_rule"),
    ({"qc.big_blobs_rule": 1}, r"qc\.big_blobs_rule.*str"),
    ({"qc.transient_min_run": 1}, r"qc\.transient_min_run.*≥ 2"),
    ({"qc.transient_min_run": 0}, r"qc\.transient_min_run"),
    ({"qc.transient_min_run": 4.5}, r"qc\.transient_min_run.*int"),
    ({"qc.transient_min_run": True}, r"qc\.transient_min_run.*int, bukan bool"),
])
def test_invalid_qc_rule_params(over, match):
    with pytest.raises(ConfigError, match=match):
        load_pipeline(overrides=over)


def test_valid_qc_rule_params():
    for rule in ("transient", "legacy", "report"):
        assert load_pipeline(overrides={"qc.big_blobs_rule": rule}).qc.big_blobs_rule == rule
    assert load_pipeline(overrides={"qc.transient_min_run": 2}).qc.transient_min_run == 2


# ── qc.* tidak membuat seg/ atau depth/ basi ───────
QC_CHANGES = [
    ("qc.area_min", 0.2), ("qc.area_max", 0.9), ("qc.iou_min", 0.95), ("qc.blob_min", 0.01),
    ("qc.max_big_blobs", 3), ("qc.area_median_window", 3), ("qc.area_drop_min", 0.99),
    ("qc.big_blobs_rule", "legacy"), ("qc.big_blobs_rule", "report"), ("qc.transient_min_run", 6),
]


@pytest.mark.parametrize("key,val", QC_CHANGES)
def test_qc_change_does_not_make_seg_or_depth_stale(tmp_path, monkeypatch, key, val):
    monkeypatch.setenv(seg.HF_OFFLINE_ENV, "1")
    work = ts.make_work(tmp_path)
    cfg = gpu_cfg(work)
    run_segment(cfg, backend_factory=ts.factory(), log=ts.quiet)
    run_depth(cfg, backend_factory=td.factory(), log=ts.quiet)
    seg_before, dep_before = tree_sha(work / "seg"), tree_sha(work / "depth")
    report_before = json.loads((work / "qc_report.json").read_text(encoding="utf-8"))

    cfg2 = gpu_cfg(work, **{key: val})
    fs, fd = ts.factory(), td.factory()
    r_seg = run_segment(cfg2, backend_factory=fs, log=ts.quiet)          # tidak melempar "setelan berbeda"
    r_dep = run_depth(cfg2, backend_factory=fd, log=ts.quiet)
    assert r_seg["processed"] == 0 and r_seg["skipped"] == ts.N_FRAMES
    assert r_dep["processed"] == 0 and r_dep["skipped"] == td.N_FRAMES
    assert fs.made[0].opened == 0 and fs.made[0].calls == 0                 # GPU tidak disentuh
    assert fd.made[0].opened == 0 and fd.made[0].calls == 0
    assert tree_sha(work / "seg") == seg_before and tree_sha(work / "depth") == dep_before   # byte-identik
    report = json.loads((work / "qc_report.json").read_text(encoding="utf-8"))
    assert report["qc"][key.split(".", 1)[1]] == val                       # QC dihitung ulang dengan setelan baru
    assert report["qc_hash"] != report_before["qc_hash"]


def test_qc_only_does_not_touch_seg(tmp_path, monkeypatch):
    monkeypatch.setenv(seg.HF_OFFLINE_ENV, "1")
    work = ts.make_work(tmp_path)
    run_segment(gpu_cfg(work), backend_factory=ts.factory(), log=ts.quiet)
    before = tree_sha(work / "seg")
    run_qc(gpu_cfg(work, **{"qc.big_blobs_rule": "legacy"}), log=ts.quiet)
    assert tree_sha(work / "seg") == before


# ── [3] basi bila himpunan frame gagal QC berubah ──
def two_people_clip(tmp_path: Path) -> Path:
    """12 frame: f0–7 dua orang (n=2), f8–11 satu orang (n=1) → legacy: 8 gagal; transient / report: 0 gagal."""
    one = tsm.static_cm()
    two = one.copy()
    two[8:34, 2:10] = tst.CID["Torso"]          # 26 × 8 = 208 px > 5% dari 40 × 56; terpisah 4 px dari torso
    return tsm.make_clip(tmp_path, [two] * 8 + [one] * 4, qc=False)


def stab_cfg(work: Path, **qc_over):
    ov = {"paths.work_dir": str(work), **tsm.ON, **{f"qc.{k}": v for k, v in qc_over.items()}}
    return load_pipeline(overrides=ov)


def test_stage3_recomputes_when_qc_fail_frames_change(tmp_path):
    work = two_people_clip(tmp_path)
    rep = run_qc(stab_cfg(work, big_blobs_rule="legacy"), log=ts.quiet)
    assert rep["summary"]["fail_frames"] == list(range(8))
    r = stb.run_stabilize(stab_cfg(work), log=ts.quiet)
    assert r["processed"] == 12
    m = json.loads((work / "stable" / "manifest.json").read_text(encoding="utf-8"))
    assert m["temporal"]["qc_fail_frames"] == [f"frame_{i:05d}.png" for i in range(8)]

    rep = run_qc(stab_cfg(work), log=ts.quiet)                              # default transient → 0 gagal
    assert rep["summary"]["n_fail"] == 0
    msgs = []
    r = stb.run_stabilize(stab_cfg(work), log=msgs.append)
    assert r["processed"] == 12 and "basi" in msgs[0]
    assert any(s.startswith("temporal.qc_fail_frames") for s in r["stale"])

    run_qc(stab_cfg(work, big_blobs_rule="report"), log=ts.quiet)          # himpunan gagal sama ([]) → tidak basi
    r = stb.run_stabilize(stab_cfg(work), log=ts.quiet)
    assert r["processed"] == 0 and r["stale"] == []


# ── Peringatan CLI menyebut aturan ─────────────────
def test_cli_qc_warning_names_rule(tmp_path):
    rep = {"qc": {"big_blobs_rule": "transient"}, "summary": {"n_frames": 182, "n_fail": 1, "fail_frames": [121]}}
    (tmp_path / "qc_report.json").write_text(json.dumps(rep), encoding="utf-8")
    w = cli.qc_warning(tmp_path)
    assert w.startswith("PERINGATAN: 1/182 frame gagal QC") and "aturan big_blobs: transient" in w and "121" in w
    rep.pop("qc")
    (tmp_path / "qc_report.json").write_text(json.dumps(rep), encoding="utf-8")
    assert "aturan big_blobs: legacy (laporan sebelum T-305d)" in cli.qc_warning(tmp_path)


def test_qc_config_is_frozen_dataclass_with_new_fields():
    names = [f.name for f in dataclasses.fields(load_pipeline().qc)]
    assert {"big_blobs_rule", "transient_min_run"} <= set(names)
