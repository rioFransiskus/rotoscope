"""T-605 Tahap 1 (alat sekali pakai, F3): ukur deteksi adegan untuk stage [1b] dengan FUNGSI CUT [3] APA ADANYA.

    python scripts/t605_scenes.py

Per klip (6 klip): thumbnail abu-abu `stabilize.frame_thumb` → `cut_scores` → `detect_cuts(cut_diff)` → `shot_bounds`
(ambang = `stabilize.temporal.cut_diff` dari configs/default.yaml), dibandingkan dengan (a) cut yang tercatat di
`stable/manifest.json` (`temporal.cut_frames`, hasil [3] produksi) dan (b) label Rio (docs/04: "Kalibrasi cut_diff" klip3,
"Penilaian Rio Tahap 1 + keputusan Tahap 1b" klip4 / klip5). Lalu frame yang diklik di `target/points.json` (klip3, klip5)
dipetakan ke adegan → adegan yang akan dilacak [1b] dan jumlah frame "tanpa_prompt".

Hanya MEMBACA work/clips/* (frames/, stable/manifest.json, target/points.json, meta.json); menulis hanya work/t605/scenes.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rotoscope import stabilize as stb  # noqa: E402
from rotoscope.config import load_pipeline  # noqa: E402
from rotoscope.ingest import FRAMES_DIRNAME  # noqa: E402
from rotoscope.stage_common import load_frame_list  # noqa: E402

CLIPS_DIR = ROOT / "work" / "clips"
OUT = ROOT / "work" / "t605" / "scenes.json"
CLIPS = ("test", "test_short", "Klip2", "klip3", "klip4", "klip5")

# Label Rio (docs/04). true = cut sungguhan; fp = BUKAN cut (positif palsu yang diketahui). Klip tanpa entri `true` /
# `fp` dan dengan `no_cuts` = Rio menyatakan tanpa cut.
LABELS = {
    "test": {"true": [], "fp": [], "no_cuts": True, "src": "Kalibrasi cut_diff (2026-10-10)"},
    "test_short": {"true": [], "fp": [], "no_cuts": True, "src": "Kalibrasi cut_diff (2026-10-10)"},
    "Klip2": {"true": [], "fp": [], "no_cuts": True, "src": "Kalibrasi cut_diff (2026-10-10)"},
    "klip3": {"true": [85, 126, 134, 146, 154, 166, 174], "fp": [49], "no_cuts": False,
              "src": "Kalibrasi cut_diff (2026-10-10): 7 cut; f49 = FP diketahui"},
    "klip4": {"true": [], "fp": [209], "no_cuts": False,
              "src": "Penilaian Rio Tahap 1 + keputusan Tahap 1b (2026-10-15): f209 BUKAN cut"},
    # klip5: f92–f104 = cut / transisi sungguhan ke objek lain; deteksi di f92–96 dan f100–104 BUKAN positif palsu.
    "klip5": {"true_range": [92, 104], "fp": [], "no_cuts": False,
              "src": "Penilaian Rio Tahap 1 + keputusan Tahap 1b (2026-10-15): f92–f104 cut / transisi sungguhan"},
}


def clicked_frames(work_dir: Path) -> list[int] | None:
    p = work_dir / "target" / "points.json"
    if not p.is_file():
        return None
    doc = json.loads(p.read_text(encoding="utf-8"))
    return sorted(int(e["frame_index"]) for e in doc["frames"])


def manifest_cuts(work_dir: Path) -> list[int] | None:
    p = work_dir / stb.STABLE_DIRNAME / stb.MANIFEST_FILENAME
    if not p.is_file():
        return None
    t = json.loads(p.read_text(encoding="utf-8")).get("temporal", {})
    if not t.get("enabled"):
        return None
    return [int(Path(n).stem.split("_")[-1]) for n in t.get("cut_frames", [])]


def shots_of(cut_idx: list[int], indices: tuple[int, ...]) -> list[list[int]]:
    """[awal, akhir] inklusif (indeks frame) tiap adegan, dari daftar indeks frame yang punya cut SEBELUM-nya."""
    cut_before = [i in set(cut_idx) for i in indices]
    start, end = stb.shot_bounds(cut_before)
    seen, out = set(), []
    for k in range(len(indices)):
        if start[k] not in seen:
            seen.add(start[k])
            out.append([indices[start[k]], indices[end[k]]])
    return out


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    cfg = load_pipeline(ROOT / "configs" / "default.yaml")
    cut_diff = cfg.stabilize.temporal.cut_diff
    res = {"cut_diff": cut_diff, "thumb_width": stb.CUT_THUMB_WIDTH,
           "functions": "stabilize.frame_thumb + cut_scores + detect_cuts + shot_bounds (tanpa perubahan)", "clips": {}}
    print(f"cut_diff = {cut_diff} (configs/default.yaml), thumbnail lebar {stb.CUT_THUMB_WIDTH} px abu-abu")
    for clip in CLIPS:
        wd = CLIPS_DIR / clip
        names, indices, w, h = load_frame_list(wd)
        thumbs = [stb.frame_thumb(wd / FRAMES_DIRNAME / n) for n in names]
        scores = stb.cut_scores(thumbs)
        cut_before = stb.detect_cuts(scores, cut_diff)
        det = [indices[k] for k in range(len(indices)) if cut_before[k]]
        man = manifest_cuts(wd)
        lab = LABELS[clip]
        if "true_range" in lab:
            a, b = lab["true_range"]
            tp = [f for f in det if a <= f <= b]
            fp_unknown = [f for f in det if not a <= f <= b and f not in lab["fp"]]
            fp_known = [f for f in det if f in lab["fp"]]
            fn = [] if any(f == a for f in det) else [a]          # awal transisi wajib terdeteksi (batas adegan berklik)
            true_desc = f"f{a}–f{b} (rentang transisi)"
        else:
            tp = [f for f in det if f in lab["true"]]
            fp_known = [f for f in det if f in lab["fp"]]
            fp_unknown = [f for f in det if f not in lab["true"] and f not in lab["fp"]]
            fn = [f for f in lab["true"] if f not in det]
            true_desc = ", ".join(f"f{f}" for f in lab["true"]) or "tanpa cut"
        missed_fp = [f for f in lab["fp"] if f not in det]
        shots = shots_of(det, indices)
        clicks = clicked_frames(wd)
        tracked, n_tanpa = None, None
        if clicks is not None:
            tracked = [s for s in shots if any(s[0] <= c <= s[1] for c in clicks)]
            n_tanpa = len(indices) - sum(s[1] - s[0] + 1 for s in tracked)
        scored = sorted(((scores[k], indices[k]) for k in range(len(indices))), reverse=True)
        top_neg = [(f, round(s, 5)) for s, f in scored
                   if f not in det][:3]
        row = {"n_frames": len(indices), "frame_size": [w, h], "detected": det,
               "detected_scores": {str(indices[k]): round(scores[k], 5) for k in range(len(indices)) if cut_before[k]},
               "stage3_manifest_cuts": man, "same_as_stage3": (man == det) if man is not None else None,
               "labels": lab, "tp": tp, "fn": fn, "fp_known": fp_known, "fp_unknown": fp_unknown,
               "known_fp_not_detected": missed_fp, "top_scores_below_threshold": top_neg,
               "shots": shots, "n_shots": len(shots), "clicked_frames": clicks, "tracked_shots": tracked,
               "n_frames_tanpa_prompt": n_tanpa}
        res["clips"][clip] = row
        print(f"\n{clip}: {len(indices)} frame {w}×{h}; label: {true_desc}")
        print(f"  terdeteksi ({len(det)}): {', '.join(f'f{f}' for f in det) or '-'}")
        print(f"  = cut [3] di stable/manifest.json: {row['same_as_stage3']}  ({man})")
        print(f"  cocok label: TP {len(tp)}, FN {fn or '-'}, FP diketahui {fp_known or '-'}, FP lain {fp_unknown or '-'}"
              + (f", FP diketahui TIDAK terdeteksi {missed_fp}" if missed_fp else ""))
        print(f"  skor tertinggi di bawah ambang: {top_neg}")
        print(f"  adegan ({len(shots)}): {shots}")
        if clicks is not None:
            print(f"  klik di frame {clicks} → adegan dilacak {tracked}; frame tanpa_prompt {n_tanpa}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\n-> {OUT.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
