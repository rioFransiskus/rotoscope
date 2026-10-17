"""T-605 Tahap 1 (alat sekali pakai, F3): cek salinan titik Rio `scripts/t605_points_rio.json` terhadap berkas asli.

    python scripts/t605_points_check.py

Per klip (klip3, klip5): (1) `json.dumps(doc, indent=2, ensure_ascii=False)` dalam UTF-8 = byte berkas asli
`work/clips/<klip>/target/points.json` (cara tulis `pick` = `stage_common.write_json_atomic`); (2) `bytes` + `sha256` di
salinan = berkas asli; (3) `doc` lolos `pick.validate_doc`; (4) status tiap frame berklik = `ok` terhadap frames/ sekarang
(`pick.frame_statuses`, sidik piksel). Hanya MEMBACA; tidak menulis berkas apa pun.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rotoscope import pick  # noqa: E402
from rotoscope.stage_common import clip_identity, load_frame_list  # noqa: E402

COPY = ROOT / "scripts" / "t605_points_rio.json"


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    arc = json.loads(COPY.read_text(encoding="utf-8"))
    ok_all = arc.get("format") == "t605-points-rio/1"
    print(f"{COPY.relative_to(ROOT).as_posix()}: format {arc.get('format')} ({COPY.stat().st_size} B)")
    for clip, entry in arc["clips"].items():
        src = ROOT / entry["source"]
        raw = src.read_bytes()
        regen = json.dumps(entry["doc"], indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8")
        same_bytes = regen == raw
        meta_ok = entry["bytes"] == len(raw) and entry["sha256"] == hashlib.sha256(raw).hexdigest()
        try:
            pick.validate_doc(entry["doc"])
            valid = True
        except pick.PointsError as e:
            valid = f"tidak valid: {e}"
        wd = src.parent.parent
        names, indices, w, h = load_frame_list(wd)
        pick.check_doc_matches_clip(entry["doc"], clip_identity(wd), w, h, src)
        statuses = pick.frame_statuses(entry["doc"], wd, set(indices), {})
        ok = same_bytes and meta_ok and valid is True and all(s == pick.STATUS_OK for s in statuses.values())
        ok_all &= ok
        print(f"  {clip}: byte-identik {same_bytes} ({len(regen)} / {len(raw)} B), bytes + sha256 {meta_ok}, "
              f"validate_doc {valid}, status frame {statuses} -> {'LOLOS' if ok else 'GAGAL'}")
    print("SEMUA LOLOS" if ok_all else "ADA YANG GAGAL")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
