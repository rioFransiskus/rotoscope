"""T-305c Tahap 4: simpan 65 label Rio (papan v2) + peta lengkap sebagai ground truth ter-track -> scripts/t305c_labels.json.

Butuh data scratch (work/t305c_scratch/: labels.json, board_map.json, features.json) dan contours klip di work/clips/.
Dijalankan SEKALI sebelum scratch dihapus; berkas hasil mandiri (tidak bergantung pada scratch).

    python scripts/t305c_export_labels.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "tests"))
import t305c_eval as ev  # noqa: E402
import t305c_study as st  # noqa: E402

OUT = HERE / "t305c_labels.json"


def main() -> None:
    lab = st.jload(st.SCRATCH / "labels.json")
    bm = st.jload(st.SCRATCH / "board_map.json")
    units, _ = ev.load_units()
    unit_of = {u["key"]: u for u in units}
    clips = {c: st.ClipData(c) for c in st.STUDY_CLIPS}
    contract = {c: st.jload(clips[c].dir / "contours" / "manifest.json").get("contract") for c in clips}
    items = {}
    for num, m in sorted(lab["labels"].items()):
        mp = bm["mapping"][num]
        clip = clips[m["clip"]]
        frame = mp["frame"]
        stroke = next(s for s in clip.docs[frame]["strokes"] if s["type"] == "occlusion" and s["track_id"] == m["track_id"])
        pts = np.asarray(stroke["points"], float)
        u = unit_of.get((m["clip"], m["track_id"], m["run_start"]))
        frames = sorted(r["frame"] for r in u["rows"]) if u else [frame]
        items[num] = {
            "board_number": num, "label": m["label"], "clip": m["clip"], "cell": m["cell"], "donewhen": m["donewhen"],
            "rep_frame": frame, "track_id": m["track_id"], "run_start": m["run_start"],
            "frame_range": [frames[0], frames[-1]], "n_frames": len(frames),
            "rep_stroke_bbox_xyxy": [round(float(v), 2) for v in (pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max())],
            "rep_stroke_centroid_xy": [round(float(v), 2) for v in pts.mean(axis=0)],
            "rep_stroke_n_points": int(len(pts)), "group": mp.get("group"),
            "weight": (round(u["w"], 6) if u else None),
        }
    doc = {
        "source": lab["source"], "board": "papan v2", "board_order_seed": bm["order_seed"], "board_sampling_seed": bm["seed"],
        "board_salt_sample": bm["salt_sample"],
        "contours_contract_when_exported": contract,
        "contract_note": ("label dibuat pada contours 'T-305b'; strok identik di 'T-305c' dengan hi_floor 0 (hanya hash turunan berbeda). "
                          "track_id adalah id milik run yang diberi label; bila algoritma pelacak berubah, kenali unit lewat clip + "
                          "frame_range + bbox / centroid strok wakil pada rep_frame (koordinat px kerja contours)."),
        "label_meaning": {"O": "garis oklusi anggota tubuh yang diinginkan", "L": "garis tak diinginkan (lipatan / tepi lapisan kain)",
                          "?": "ragu; dikeluarkan dari analisis"},
        "counts_by_cell": lab["counts_by_cell"], "counts_by_clip": lab["counts_by_clip"], "donewhen_labels": lab["donewhen_labels"],
        "sampling": bm["sampling"], "units": items,
    }
    OUT.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{OUT.name}: {len(items)} unit, contract {contract}")


if __name__ == "__main__":
    main()
