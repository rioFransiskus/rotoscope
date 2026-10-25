"""T-304 (alat, F3): kemas data Tahap 1–3 yang berharga dari work/t304/ (ter-ignore) ke scripts/t304_tahap1_data.json (ter-track)
sebelum scratch dihapus. Hanya membaca work/t304/; menulis satu berkas di scripts/.

    python scripts/t304_pack.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "work" / "t304"
OUT = ROOT / "scripts" / "t304_tahap1_data.json"


def jload(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> int:
    pack = {
        "versi": 1,
        "catatan": "Data Tahap 1–3 T-304 (lihat docs/04 'Hasil T-304 Tahap 1' dan 'Hasil T-304 Tahap 2–3'). Skrip penghasil "
                   "(scripts/t304_tahap1.py, t304_tahap2.py, t304_board.py) TIDAK bisa dijalankan ulang: scratch "
                   "work/t304_scratch/ dan raw cache sudah dihapus.",
        "tahap1_metrics": jload(DATA / "tahap1_metrics.json"),
        "porsi": jload(DATA / "porsi.json"),
        "bsehat": jload(DATA / "bsehat.json"),
        "run_stage34": [jload(p) for p in sorted((DATA / "run").glob("*.json"))],
        "render_stage56": [jload(p) for p in sorted((DATA / "render").glob("*.json"))],
        "nilai_meta": jload(DATA / "nilai" / "meta.json"),
        "f1": {p.stem: jload(p) for p in sorted(DATA.glob("f1_*.json"))},
    }
    OUT.write_text(json.dumps(pack, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"  -> {OUT.relative_to(ROOT).as_posix()} ({OUT.stat().st_size} B)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
