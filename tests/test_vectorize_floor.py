"""Test T-305c: `vectorize.depth_lines.hi_floor` (lantai absolut T_high) pada garis oklusi, stage [4].

T_high dipakai = max(T_high persentil, lantai); T_low TIDAK disentuh; lantai 0 = perilaku T-305b. Sintetis (fungsi murni,
config, hash, clip_stats, info log, basi) + klip nyata (dilewati bila `stable/` tidak ada). Mutation-sensitive: tiap pengaman
punya test yang gagal bila pengaman itu dimatikan (scripts/t305c_mutations.py, monkeypatch dari luar berkas ini).
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest
from test_vectorize import DEPTH_DEFAULTS, N_FRAMES, NAMES, PARAMS, cfg_for, make_work, quiet
from test_vectorize_occlusion import LEG_FRAMES, T_HIGH, T_LOW, YY, big_group, has_real, lower_clothing_fraction, step

from rotoscope import vectorize as vec
from rotoscope.config import ConfigError, load_pipeline

FLOOR = "vectorize.depth_lines.hi_floor"
HUGE = 100.0                                   # jauh di atas T_high persentil sintetis


def read_stats(work: Path) -> dict:
    return json.loads((work / "contours" / "clip_stats.json").read_text(encoding="utf-8"))


def read_strokes(work: Path) -> list:
    return [json.loads(p.read_text(encoding="utf-8"))["strokes"] for p in sorted((work / "contours").glob("frame_*.json"))]


# ── Fungsi murni ───────────────────────────────────
def test_effective_t_high_cases():
    assert vec.effective_t_high(0.0140, 0.0) == 0.0140                # lantai 0 = persentil (netral)
    assert vec.effective_t_high(0.0140, 0.0100) == 0.0140             # lantai < T_high: tidak berubah
    assert vec.effective_t_high(0.0140, 0.0140) == 0.0140             # sama
    assert vec.effective_t_high(0.0140, 0.0200) == 0.0200             # lantai > T_high: lantai dipakai
    assert vec.effective_t_high(0.0140, 0.0250) == 0.0250


# ── Config ─────────────────────────────────────────
def test_config_default_zero_and_negative_rejected():
    assert load_pipeline().vectorize.depth_lines.hi_floor == 0
    assert load_pipeline(overrides={FLOOR: 0.02}).vectorize.depth_lines.hi_floor == 0.02
    assert load_pipeline(overrides={FLOOR: 0}).vectorize.depth_lines.hi_floor == 0
    with pytest.raises(ConfigError, match="hi_floor"):
        load_pipeline(overrides={FLOOR: -0.01})


def test_hi_floor_in_params_and_hash():
    p0 = vec.vectorize_params(load_pipeline())
    p1 = vec.vectorize_params(load_pipeline(overrides={FLOOR: 0.02}))
    assert p0["depth_lines.hi_floor"] == 0 and p1["depth_lines.hi_floor"] == 0.02
    assert vec.params_hash(p0) != vec.params_hash(p1)


def test_contract_is_t305c_and_algo_rev_unchanged():
    assert vec.CONTRACT == "T-305c" and vec.ALGO_REV == 2


# ── clip_stats.json, manifest, info, basi ──────────
def test_neutral_floor_records_percentile_equal_to_used(tmp_path):
    work = make_work(tmp_path)
    vec.run_vectorize(cfg_for(work), log=quiet)
    s = read_stats(work)
    assert s["hi_floor"] == 0 and s["t_high_pct"] == s["t_high"] and s["t_high"] > 0 and s["t_low"] > 0


def test_floor_above_t_high_is_used_and_t_low_untouched(tmp_path):
    w0, w1 = make_work(tmp_path / "a"), make_work(tmp_path / "b")
    vec.run_vectorize(cfg_for(w0), log=quiet)
    vec.run_vectorize(cfg_for(w1, **{FLOOR: HUGE}), log=quiet)
    s0, s1 = read_stats(w0), read_stats(w1)
    assert s1["t_high"] == HUGE and s1["t_high_pct"] == s0["t_high"]          # dipakai = lantai; persentil dicatat
    assert s1["t_low"] == s0["t_low"] and s1["t_low"] < HUGE                  # T_low = persentil, TIDAK dinaikkan
    m = json.loads((w1 / "contours" / "manifest.json").read_text(encoding="utf-8"))
    assert m["depth_thresholds"] == {"t_high": HUGE, "t_low": s0["t_low"]}
    assert sum(s["type"] == "occlusion" for fr in read_strokes(w1) for s in fr) == 0      # T_high di atas semua |grad| → tanpa benih


def test_floor_acts_on_t_high_only_in_occlusion_strokes():
    g, z = big_group(), step(YY >= 90)
    p = {**PARAMS, **DEPTH_DEFAULTS}
    base = vec.occlusion_strokes(g, z, NAMES, p, T_HIGH, T_LOW)[0]
    assert base
    assert vec.occlusion_strokes(g, z, NAMES, p, vec.effective_t_high(T_HIGH, T_HIGH / 2), T_LOW)[0] == base
    assert vec.occlusion_strokes(g, z, NAMES, p, vec.effective_t_high(T_HIGH, 1e6), T_LOW)[0] == []


def test_floor_below_t_high_leaves_strokes_unchanged(tmp_path):
    w0, w1 = make_work(tmp_path / "a"), make_work(tmp_path / "b")
    vec.run_vectorize(cfg_for(w0), log=quiet)
    th = read_stats(w0)["t_high"]
    vec.run_vectorize(cfg_for(w1, **{FLOOR: th / 2}), log=quiet)
    assert read_strokes(w0) == read_strokes(w1)
    s1 = read_stats(w1)
    assert s1["t_high"] == th == s1["t_high_pct"]


def test_info_line_only_when_floor_active(tmp_path):
    logs: list[str] = []
    w0 = make_work(tmp_path / "a")
    vec.run_vectorize(cfg_for(w0), log=logs.append)
    assert not [m for m in logs if "hi_floor" in m]
    th = read_stats(w0)["t_high"]
    logs.clear()
    w1 = make_work(tmp_path / "b")
    vec.run_vectorize(cfg_for(w1, **{FLOOR: HUGE}), log=logs.append)
    info = [m for m in logs if "hi_floor" in m]
    assert len(info) == 1 and f"{th:.6g}" in info[0] and f"{HUGE:.6g}" in info[0] and "persentil" in info[0]
    logs.clear()
    w2 = make_work(tmp_path / "c")
    vec.run_vectorize(cfg_for(w2, **{FLOOR: th / 2}), log=logs.append)           # lantai di bawah T_high: tidak aktif
    assert not [m for m in logs if "hi_floor" in m]


def test_changing_hi_floor_makes_contours_stale_and_recomputes(tmp_path):
    work = make_work(tmp_path)
    vec.run_vectorize(cfg_for(work), log=quiet)
    run = vec.run_vectorize(cfg_for(work, **{FLOOR: HUGE}), log=quiet)
    assert run["processed"] == N_FRAMES and any("hi_floor" in s for s in run["stale"])
    again = vec.run_vectorize(cfg_for(work, **{FLOOR: HUGE}), log=quiet)
    assert again["processed"] == 0 and again["stale"] == []                      # nilai sama: dipakai ulang


# ── Klip nyata (dilewati bila tidak ada) ───────────
REAL_LIMIT = {"test_short": 93, "test": 93, "Klip2": 30}


def scratch_copy(clip: str, dst: Path) -> Path:
    """Salinan scratch: stable/ (hard link, hanya dibaca oleh [4]) + meta.json + qc_report.json. Klip asli tidak disentuh."""
    src = Path("work/clips") / clip
    shutil.copytree(src / "stable", dst / "stable", copy_function=lambda a, b: _link_or_copy(a, b))
    for f in ("meta.json", "qc_report.json"):
        shutil.copyfile(src / f, dst / f)
    return dst


def _link_or_copy(a: str, b: str) -> None:
    try:
        os.link(a, b)
    except OSError:
        shutil.copy2(a, b)


def real_run(clip: str, dst: Path, floor: float) -> Path:
    cfg = load_pipeline(overrides={"paths.work_dir": str(dst), FLOOR: floor})
    vec.run_vectorize(cfg, limit=REAL_LIMIT[clip], log=quiet)
    return dst


@pytest.mark.parametrize("clip", ["test_short", "test", "Klip2"])
def test_real_floor_002_identical_to_zero_and_leg_done_when(tmp_path, clip):
    """Lantai 0,02 ≤ T_high persentil ketiga klip (0,0257 / ... / 0,0733) → strok identik dengan lantai 0; Done-when kaki T-201b
    (test / test_short): ≥ 80% titik di piksel Lower_Clothing pada f73 / 78 / 82 / 87 / 92, 5 dari 5 frame."""
    if not has_real(clip) or not (Path("work/clips") / clip / "meta.json").is_file():
        pytest.skip("data nyata tidak ada")
    a = real_run(clip, scratch_copy(clip, tmp_path / "a"), 0.0)
    b = real_run(clip, scratch_copy(clip, tmp_path / "b"), 0.02)
    sa, sb = read_stats(a), read_stats(b)
    assert sa["t_high"] >= 0.02, "klip ini tidak lagi di atas lantai: test tidak bermakna"
    assert sb["t_high"] == sa["t_high"] == sb["t_high_pct"] and sb["t_low"] == sa["t_low"]
    assert read_strokes(a) == read_strokes(b) and read_strokes(a)
    if clip in ("test", "test_short"):
        strokes = read_strokes(b)
        ok = [i for i in LEG_FRAMES
              if any(s["type"] == "occlusion" and lower_clothing_fraction(clip, i, s) >= 0.8 for s in strokes[i])]
        assert ok == list(LEG_FRAMES), ok


def test_real_klip3_floor_is_active_and_raises_t_high(tmp_path):
    """klip3 (T_high persentil 0,0140 < 0,02): lantai aktif → T_high dipakai 0,02, T_low tetap persentil, strok oklusi berkurang."""
    if not has_real("klip3") or not (Path("work/clips/klip3/meta.json")).is_file():
        pytest.skip("data nyata klip3 tidak ada")
    REAL_LIMIT["klip3"] = 12
    a = real_run("klip3", scratch_copy("klip3", tmp_path / "a"), 0.0)
    b = real_run("klip3", scratch_copy("klip3", tmp_path / "b"), 0.02)
    sa, sb = read_stats(a), read_stats(b)
    assert sa["t_high"] < 0.02 and sb["t_high"] == 0.02 and sb["t_high_pct"] == sa["t_high"] and sb["t_low"] == sa["t_low"]
    n = lambda w: sum(s["type"] == "occlusion" for fr in read_strokes(w) for s in fr)   # noqa: E731
    assert 0 < n(b) < n(a)
