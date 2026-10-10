"""Test kalibrasi cut_diff (2026-10-10, keputusan Rio: 0,08 -> 0,06): skor selisih frame fungsi produksi stabilize pada klip nyata
(label Rio: klip3 punya 7 cut f85 / f126 / f134 / f146 / f154 / f166 / f174; test, test_short, Klip2 tanpa cut) + sintetis di sekitar ambang.
Klip nyata tidak ada -> skip (pola test real-clip)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rotoscope import stabilize as stb
from rotoscope.config import load_pipeline

CLIPS = ("test", "test_short", "Klip2", "klip3")
POSITIVES = {85, 126, 134, 146, 154, 166, 174}          # label Rio, klip3
KNOWN_FALSE_POSITIVES = {49}                              # klip3 f49: skor 0,0652 (negatif tertinggi), FP yang diketahui; BOLEH terdeteksi atau tidak (tidak dikunci)
_CACHE: dict[str, list[float]] = {}


def frames_dir(clip: str) -> Path:
    return Path("work/clips") / clip / "frames"


def has_real(clip: str) -> bool:
    return frames_dir(clip).is_dir() and any(frames_dir(clip).glob("frame_*.png"))


def real_scores(clip: str) -> list[float]:
    if clip not in _CACHE:
        names = sorted(p.name for p in frames_dir(clip).glob("frame_*.png"))
        _CACHE[clip] = stb.cut_scores([stb.frame_thumb(frames_dir(clip) / n) for n in names])
    return _CACHE[clip]


def detected(clip: str, cut_diff: float) -> set[int]:
    return {k for k, c in enumerate(stb.detect_cuts(real_scores(clip), cut_diff)) if c}


def check_real_clips(cut_diff: float) -> None:
    """Pernyataan kalibrasi; dipanggil test dengan default produksi dan skrip mutasi dengan nilai lain."""
    got = detected("klip3", cut_diff)
    assert POSITIVES <= got, f"positif tidak terdeteksi: {sorted(POSITIVES - got)}"                      # (a) ketujuh label terdeteksi
    assert got - POSITIVES <= KNOWN_FALSE_POSITIVES, f"FP selain f49: {sorted(got - POSITIVES - KNOWN_FALSE_POSITIVES)}"   # (b) f49 boleh ada / tidak
    for c in ("test", "test_short", "Klip2"):                                                              # (c) klip tanpa cut
        assert detected(c, cut_diff) == set(), (c, sorted(detected(c, cut_diff)))


def test_default_cut_diff_is_calibrated_value():
    assert load_pipeline().stabilize.temporal.cut_diff == 0.06


@pytest.mark.skipif(not all(has_real(c) for c in CLIPS), reason="klip nyata tidak ada")
def test_real_clips_seven_positives_no_false_positive_except_f49_and_no_cut_elsewhere():
    check_real_clips(load_pipeline().stabilize.temporal.cut_diff)


@pytest.mark.skipif(not has_real("klip3"), reason="klip nyata tidak ada")
def test_real_positive_scores_have_margin_over_default_threshold():
    """Hanya tentang POSITIF: skor terendah (f85, 0,0718) >= 1,15 x ambang yang dipakai. Jarak negatif ke ambang sengaja tidak dikunci."""
    s = real_scores("klip3")
    assert min(s[f] for f in POSITIVES) >= 1.15 * load_pipeline().stabilize.temporal.cut_diff


def _thumbs_with_difference(d: int) -> list[np.ndarray]:
    """Dua thumbnail abu-abu seragam yang berbeda tepat d tingkat (0-255): skor = d / 255."""
    a = np.full((8, 8), 100, np.float32)
    return [a, a + d]


def test_synthetic_score_just_above_and_below_threshold():
    t = 0.06
    below = stb.cut_scores(_thumbs_with_difference(15))          # 15 / 255 = 0,0588
    above = stb.cut_scores(_thumbs_with_difference(16))          # 16 / 255 = 0,0627
    assert below[1] == pytest.approx(15 / 255) and above[1] == pytest.approx(16 / 255)
    assert stb.detect_cuts(below, t) == [False, False]
    assert stb.detect_cuts(above, t) == [False, True]


def test_detect_cuts_is_strictly_greater_and_zero_disables():
    assert stb.detect_cuts([0.0, 0.06, 0.0600001], 0.06) == [False, False, True]
    assert stb.detect_cuts([0.0, 0.9], 0) == [False, False]
