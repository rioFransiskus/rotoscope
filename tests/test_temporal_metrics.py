"""Test metrik permanen T-303 di tests/temporal_metrics.py: pop energy per tipe strok, umur track, galat alignment warp,
konsistensi maju-mundur. Nilai dihitung tangan dari kasus kecil; satu test memakai klip nyata (di-skip bila tidak ada).
Optical flow produksi TIDAK ada (ditolak, docs/04 "Hasil T-303"): fungsi ini hanya mengukur flow yang diberikan pemanggil."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import temporal_metrics as tm

SIL, OCC, HOLE = ("silhouette", 1), ("occlusion", 1), ("silhouette_hole", 2)
REPO = Path(__file__).resolve().parents[1]


def test_stroke_length_open_and_closed():
    pts = [[0, 0], [3, 0], [3, 4]]
    assert tm.stroke_length(pts, False) == pytest.approx(7.0)
    assert tm.stroke_length(pts, True) == pytest.approx(12.0)          # + penutup (3,4)→(0,0) = 5
    assert tm.stroke_length([[1, 1]], True) == 0.0


def test_pop_energy_born_and_died_hand_computed():
    frames = [{SIL: 10.0}, {SIL: 10.0, OCC: 5.0}, {SIL: 10.0}]
    r = tm.pop_energy_by_type(frames)
    occ = r["types"]["occlusion"]
    # t=1: lahir 5 / total 15; t=2: mati 5 / total 10 → rata-rata (1/3 + 1/2) / 2
    assert occ["pop_energy_mean"] == pytest.approx((1 / 3 + 1 / 2) / 2)
    assert occ["born_per_frame"] == pytest.approx(0.5) and occ["died_per_frame"] == pytest.approx(0.5)
    assert occ["length_share_mean"] == pytest.approx((5 / 15 + 0) / 2)
    assert occ["id_new_per_stroke_frame"] == pytest.approx(1 / 1)       # satu strok-frame (t=1), satu id baru
    assert occ["frames_without"] == 2 and occ["age_mean"] == 1.0
    assert r["types"]["silhouette"]["pop_energy_mean"] == 0.0
    assert r["total_pop_energy_mean"] == pytest.approx(occ["pop_energy_mean"])
    assert occ["pop_share_pct"] == pytest.approx(100.0)                  # satu-satunya sumber pop


def test_pop_energy_shares_sum_to_100_and_identity_swap_counts():
    # id berganti (OCC id 1 → id 3) = satu mati + satu lahir, walau bentuk sama
    frames = [{SIL: 10.0, HOLE: 4.0, OCC: 6.0}, {SIL: 10.0, HOLE: 4.0, ("occlusion", 3): 6.0}]
    r = tm.pop_energy_by_type(frames)
    assert r["types"]["occlusion"]["pop_energy_mean"] == pytest.approx((6 + 6) / 20)
    assert sum(v["pop_share_pct"] for v in r["types"].values()) == pytest.approx(100.0)
    assert r["types"]["silhouette_hole"]["pop_energy_mean"] == 0.0


def test_pop_energy_no_pop_gives_none_share():
    r = tm.pop_energy_by_type([{SIL: 5.0}, {SIL: 5.0}])
    assert r["total_pop_energy_mean"] == 0.0
    assert all(v["pop_share_pct"] is None for v in r["types"].values())


def test_track_ages_and_frames_without():
    frames = [{OCC: 1.0}, {OCC: 1.0}, {}, {OCC: 1.0}, {HOLE: 1.0}]
    assert sorted(tm.track_ages(frames, "occlusion")) == [1, 2]          # run putus = dua run
    assert tm.frames_without_type(frames, "occlusion") == 2
    assert tm.track_ages(frames, "silhouette") == []


def test_stroke_lengths_from_dir(tmp_path):
    for i, strokes in enumerate([[{"type": "silhouette", "track_id": 1, "closed": True, "points": [[0, 0], [4, 0], [4, 3]]}], []]):
        (tmp_path / f"frame_{i:05d}.json").write_text(json.dumps({"strokes": strokes}), encoding="utf-8")
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")        # bukan frame_*.json → diabaikan
    fr = tm.stroke_lengths_from_dir(tmp_path)
    assert fr == [{("silhouette", 1): pytest.approx(12.0)}, {}]


def test_fb_error_consistent_and_inconsistent():
    h, w = 20, 30
    f = np.zeros((h, w, 2), np.float32)
    f[..., 0] = 2.0
    b = -f
    assert float(tm.fb_error(f, b)[:, 2:-2].max()) < 1e-5              # tepi kanan membaca b yang direplikasi → tetap 0 (konstan)
    b_bad = np.zeros_like(f)
    b_bad[..., 0] = -1.0                                                  # kembali hanya 1 px dari 2
    e = tm.fb_error(f, b_bad)
    assert np.allclose(e, 1.0, atol=1e-5)
    frac = tm.inconsistent_fraction(e)
    assert frac[0.5] == 1.0 and frac[1.0] == 0.0 and frac[2.0] == 0.0     # 1,0 > 1,0 salah; > 0,5 benar
    mask = np.zeros((h, w), bool)
    assert tm.inconsistent_fraction(e, mask) == {0.5: 0.0, 1.0: 0.0, 2.0: 0.0}   # mask kosong → 0


def test_alignment_error_perfect_warp_vs_no_warp():
    h, w = 40, 60
    raw_t = np.zeros((h, w), np.uint8)
    raw_t[10:30, 10:20] = 1
    raw_tk = np.zeros_like(raw_t)
    raw_tk[10:30, 14:24] = 1                                              # bergeser 4 px
    perfect = tm.alignment_error(raw_t, raw_t.copy(), raw_tk)
    assert perfect["err_warp"] == 0.0 and perfect["err_nowarp"] > 0.2 and perfect["moving_px"] > 0
    useless = tm.alignment_error(raw_t, raw_tk, raw_tk)                   # "warp" yang tidak menggeser = tanpa warp
    assert useless["err_warp"] == useless["err_nowarp"]
    # valid membuang piksel: tanpa piksel valid → 0 piksel, galat 0 (tidak membagi nol)
    none = tm.alignment_error(raw_t, raw_t, raw_tk, valid=np.zeros((h, w), bool))
    assert none == {"moving_px": 0, "err_warp": 0.0, "err_nowarp": 0.0}


def test_moving_mask_dilates_xor():
    a = np.zeros((20, 20), np.uint8)
    b = a.copy()
    b[10, 10] = 1
    m = tm.moving_mask(a, b)
    assert int(m.sum()) == tm.ALIGN_DILATE_PX ** 2 and m[10, 10]


@pytest.mark.parametrize("clip,expected", [("test_short", 0.0488), ("test", 0.0633)])
def test_real_clip_pop_energy_t302_default(clip, expected):
    """Regresi alat ukur pada klip nyata: nilai terukur T-303 Tahap 1 untuk contours/ default T-302 (hanya bila klip ada)."""
    d = REPO / "work" / "clips" / clip / "contours"
    if not d.is_dir() or not list(d.glob("frame_*.json")):
        pytest.skip(f"klip {clip} tidak ada")
    r = tm.pop_energy_by_type(tm.stroke_lengths_from_dir(d))
    assert r["total_pop_energy_mean"] == pytest.approx(expected, abs=5e-4)
    shares = [v["pop_share_pct"] for v in r["types"].values()]
    assert sum(shares) == pytest.approx(100.0)
