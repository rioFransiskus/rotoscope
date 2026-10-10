"""Kalibrasi cut_diff dengan label Rio (alat sekali pakai; CPU, baca-saja pada work/clips/*/frames; hasil JSON ke work/cut_scratch/):

    python scripts/cut_calibrate.py scores    # skor produksi (stabilize.frame_thumb + cut_scores), positif / negatif, sweep, rasio
    python scripts/cut_calibrate.py alt       # ukuran alternatif (rasio ke median lokal, selisih histogram) bila pemisahan absolut kurang bersih

Label (Rio, 2026-10-10): klip3 punya 7 cut: f85, f126, f134, f146, f154, f166, f174; test, test_short, Klip2 tidak punya cut."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rotoscope import stabilize as stb  # noqa: E402

CLIPS = ("test", "test_short", "Klip2", "klip3")
LABELS = {"klip3": {85, 126, 134, 146, 154, 166, 174}}
OUT = ROOT / "work" / "cut_scratch"
SWEEP = [round(0.04 + 0.005 * k, 3) for k in range(11)]          # 0,04 .. 0,09


def names_of(clip: str) -> list[str]:
    return sorted(p.name for p in (ROOT / "work" / "clips" / clip / "frames").glob("frame_*.png"))


def thumbs(clip: str) -> list[np.ndarray]:
    return [stb.frame_thumb(ROOT / "work" / "clips" / clip / "frames" / n) for n in names_of(clip)]


def cmd_scores() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    scores = {c: stb.cut_scores(thumbs(c)) for c in CLIPS}          # indeks k = skor pasangan (k-1, k); skor[0] = 0
    pos = {f: scores["klip3"][f] for f in sorted(LABELS["klip3"])}
    neg = {}
    for c in CLIPS:
        lab = LABELS.get(c, set())
        neg[c] = sorted(((k, s) for k, s in enumerate(scores[c]) if k > 0 and k not in lab), key=lambda x: -x[1])
    all_neg = max(s for c in CLIPS for _, s in neg[c][:1])
    res: dict = {"thumb_width": stb.CUT_THUMB_WIDTH, "n_frames": {c: len(scores[c]) for c in CLIPS}, "positives": pos,
                 "positive_min": min(pos.values()), "negative_top10": {c: neg[c][:10] for c in CLIPS},
                 "negative_max_per_clip": {c: neg[c][0] for c in CLIPS}, "negative_max_overall": all_neg,
                 "ratio_min_pos_over_max_neg": min(pos.values()) / all_neg}
    sweep = {}
    for t in SWEEP:
        tp = sum(s > t for s in pos.values())
        fp = {c: [k for k, s in neg[c] if s > t] for c in CLIPS}
        sweep[str(t)] = {"TP": tp, "FN": len(pos) - tp, "FP": sum(len(v) for v in fp.values()), "FP_frames": {c: v for c, v in fp.items() if v}}
    res["sweep"] = sweep
    res["geometric_mean_threshold"] = math.sqrt(min(pos.values()) * all_neg)
    (OUT / "scores.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    (OUT / "scores_all.json").write_text(json.dumps({c: scores[c] for c in CLIPS}), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "sweep"}, indent=1, ensure_ascii=False))
    for t, v in sweep.items():
        print(t, v)


def local_ratio(s: list[float], half: int = 5) -> np.ndarray:
    """skor[k] / median skor tetangga k-half..k+half (tanpa k); frame 0 = 0."""
    a = np.asarray(s, float)
    out = np.zeros(len(a))
    for k in range(1, len(a)):
        nb = np.concatenate([a[max(1, k - half):k], a[k + 1:k + half + 1]])
        out[k] = a[k] / max(float(np.median(nb)), 1e-6) if len(nb) else 0.0
    return out


def spike_ratio(s: list[float]) -> np.ndarray:
    """skor[k] / max(skor[k-1], skor[k+1]) -- cut = lonjakan satu frame; frame tepi memakai satu tetangga."""
    a = np.asarray(s, float)
    out = np.zeros(len(a))
    for k in range(1, len(a)):
        nb = [a[j] for j in (k - 1, k + 1) if 1 <= j < len(a)]
        out[k] = a[k] / max(max(nb), 1e-6) if nb else 0.0
    return out


def hist_scores(clip: str, bins: int = 16) -> list[float]:
    """Selisih histogram warna: 0,5 x L1 antar histogram BGR (bins per kanal, dinormalisasi) thumbnail lebar 48; skor frame 0 = 0."""
    hs = []
    for n in names_of(clip):
        im = cv2.imdecode(np.fromfile(ROOT / "work" / "clips" / clip / "frames" / n, np.uint8), cv2.IMREAD_COLOR)
        h, w = im.shape[:2]
        im = cv2.resize(im, (stb.CUT_THUMB_WIDTH, max(1, round(h * stb.CUT_THUMB_WIDTH / w))), interpolation=cv2.INTER_AREA)
        hs.append(np.concatenate([np.bincount(im[..., c].ravel().astype(np.int32) * bins // 256, minlength=bins) for c in range(3)]).astype(float) / im[..., 0].size / 3)
    return [0.0] + [0.5 * float(np.abs(b - a).sum()) for a, b in zip(hs, hs[1:])]


def separation(meas: dict[str, np.ndarray]) -> dict:
    pos = {f: float(meas["klip3"][f]) for f in sorted(LABELS["klip3"])}
    neg = {}
    for c in CLIPS:
        lab = LABELS.get(c, set())
        neg[c] = sorted(((k, float(v)) for k, v in enumerate(meas[c]) if k > 0 and k not in lab), key=lambda x: -x[1])
    mx = max(neg[c][0][1] for c in CLIPS)
    worst = max(CLIPS, key=lambda c: neg[c][0][1])
    return {"positives": pos, "pos_min": min(pos.values()), "neg_max_overall": mx, "neg_max_clip_frame": [worst, neg[worst][0][0]],
            "neg_max_per_clip": {c: neg[c][0] for c in CLIPS}, "neg_top5_overall": sorted(((c, k, v) for c in CLIPS for k, v in neg[c][:5]), key=lambda x: -x[2])[:5],
            "ratio": min(pos.values()) / mx if mx else None}


def cmd_alt() -> None:
    base = json.loads((OUT / "scores_all.json").read_text(encoding="utf-8"))
    res = {"absolute": separation({c: np.asarray(base[c]) for c in CLIPS}),
           "local_ratio_median_pm5": separation({c: local_ratio(base[c]) for c in CLIPS}),
           "spike_ratio_neighbors": separation({c: spike_ratio(base[c]) for c in CLIPS})}
    hs = {c: hist_scores(c) for c in CLIPS}
    res["histogram_colour_16bin_L1"] = separation({c: np.asarray(hs[c]) for c in CLIPS})
    res["histogram_local_ratio_pm5"] = separation({c: local_ratio(hs[c]) for c in CLIPS})
    (OUT / "alt.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    for k, v in res.items():
        print(k, "pos_min", round(v["pos_min"], 4), "neg_max", round(v["neg_max_overall"], 4), v["neg_max_clip_frame"], "ratio", round(v["ratio"], 3))
        print("   pos", {f: round(x, 3) for f, x in v["positives"].items()})
        print("   neg_top5", [(c, k2, round(x, 3)) for c, k2, x in v["neg_top5_overall"]])


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "scores":
        cmd_scores()
    elif len(sys.argv) > 1 and sys.argv[1] == "alt":
        cmd_alt()
    else:
        sys.exit(__doc__)
