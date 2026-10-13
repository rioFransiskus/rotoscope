"""Oracle QC stage [2] SEBELUM T-305d (salinan verbatim `segment.py` di commit aff72be): aturan `big_blobs` lama
(n_t > max_big_blobs) + ringkasan. Dipakai test `legacy` byte-identik (tests/test_segment_qc.py). Jangan diubah.

Metrik dasar (iou, big_blob_count, label_agreement, area_vs_median) diambil dari `segment.py` — tidak diubah T-305d.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from typing import Any

import numpy as np

from rotoscope.segment import area_vs_median, big_blob_count, iou, label_agreement

QC_METRICS = ("area_ratio", "iou_prev", "big_blobs", "area_vs_median", "finite")


def fail_reasons(row: dict, q) -> list[str]:
    reasons = []
    if row["area_ratio"] < q.area_min or row["area_ratio"] > q.area_max:
        reasons.append("area_ratio")
    if row["iou_prev"] is not None and row["iou_prev"] < q.iou_min:
        reasons.append("iou_prev")
    if row["big_blobs"] > q.max_big_blobs:
        reasons.append("big_blobs")
    if row["area_vs_median"] is not None and row["area_vs_median"] < q.area_drop_min:
        reasons.append("area_vs_median")
    if row["finite"] is False:
        reasons.append("finite")
    return reasons


def compute_qc(classmaps: Sequence[np.ndarray] | Any, q, names: Sequence[str], indices: Sequence[int],
               frame_log: dict[str, dict] | None = None) -> list[dict]:
    """Metrik QC per frame dari classmap berurutan (iterable). Foreground = classmap ≠ 0."""
    frame_log = frame_log or {}
    rows, prev_cm = [], None
    for name, index, cm in zip(names, indices, classmaps):
        fg = cm != 0
        prev_fg = prev_cm != 0 if prev_cm is not None else None
        rec = frame_log.get(name, {})
        rows.append({
            "frame": name, "index": index,
            "area_px": int(fg.sum()),
            "area_ratio": float(fg.mean()),
            "iou_prev": iou(fg, prev_fg) if prev_fg is not None else None,
            "big_blobs": big_blob_count(fg, q.blob_min),
            "label_agreement_prev": label_agreement(cm, prev_cm) if prev_cm is not None else None,
            "finite": rec.get("finite"),
            "argmax_mismatch_pct": rec.get("argmax_mismatch_pct"),
        })
        prev_cm = cm
    for row, (med, ratio) in zip(rows, area_vs_median([r["area_px"] for r in rows], q.area_median_window)):
        row["area_median_px"] = med
        row["area_vs_median"] = ratio
    for row in rows:
        row["fail_reasons"] = fail_reasons(row, q)
    return rows


def _stats(values: list) -> dict | None:
    v = [x for x in values if x is not None]
    if not v:
        return None
    return {"mean": statistics.fmean(v), "median": statistics.median(v), "min": min(v), "max": max(v)}


def summarize_qc(rows: list[dict]) -> dict:
    return {
        "n_frames": len(rows),
        "n_fail": sum(bool(r["fail_reasons"]) for r in rows),
        "fail_counts": {m: sum(m in r["fail_reasons"] for r in rows) for m in QC_METRICS},
        "fail_frames": [r["index"] for r in rows if r["fail_reasons"]],
        "finite_false": sum(r["finite"] is False for r in rows),
        "finite_unknown": sum(r["finite"] is None for r in rows),
        "stats": {k: _stats([r[k] for r in rows]) for k in
                  ("area_ratio", "iou_prev", "big_blobs", "area_vs_median", "label_agreement_prev",
                   "argmax_mismatch_pct")},
    }
