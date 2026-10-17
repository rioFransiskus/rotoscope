"""T-605 Tahap 4 (alat sekali pakai, F3): arsipkan titik klip5 lalu hapus `target/` klip5 (keputusan akhir T-605 no. 4).

    python scripts/t605_tutup.py arsip

1. Salin `work/clips/klip5/target/points.json` + `points.prev.json` ke `work/t605/arsip_klip5_target/` (shutil.copy2).
2. Verifikasi sha256 + ukuran salinan = sumber; tulis `work/t605/arsip_klip5_target/arsip.json` (sha256, ukuran, isi ringkas).
3. HANYA bila semua cocok: hapus `work/clips/klip5/target/` (berisi tepat dua berkas itu; berkas lain → berhenti, tidak dihapus).
`isolate/` klip5 TIDAK dihapus di sini — [1b] menghapusnya sendiri saat `run` (titik tidak ada → masker lama dihapus).
Arsip yang sudah ada dan cocok → tidak disalin ulang (aman diulang).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "work" / "clips" / "klip5" / "target"
DST = ROOT / "work" / "t605" / "arsip_klip5_target"
FILES = ("points.json", "points.prev.json")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def summary(p: Path) -> list[str]:
    doc = json.loads(p.read_text(encoding="utf-8"))
    return [f"f{e['frame_index']}: " + ", ".join(
        f"target {t['id']} {sum(q['label'] == 1 for q in t['points'])} kiri / {sum(q['label'] == 0 for q in t['points'])} kanan"
        + (" + kotak" if t["box"] else "") for t in e["targets"]) for e in doc["frames"]]


def cmd_arsip() -> int:
    if not SRC.is_dir():
        done = DST / "arsip.json"
        print(f"{SRC} tidak ada" + (f" (sudah diarsipkan: {done.relative_to(ROOT).as_posix()})" if done.is_file() else ""))
        return 0 if done.is_file() else 1
    present = sorted(p.name for p in SRC.iterdir())
    if present != sorted(FILES):
        print(f"BERHENTI: isi {SRC} = {present}, diharapkan {sorted(FILES)} — tidak ada yang dihapus")
        return 1
    DST.mkdir(parents=True, exist_ok=True)
    rec = {}
    for name in FILES:
        s, d = SRC / name, DST / name
        if not d.is_file() or sha(d) != sha(s):
            shutil.copy2(s, d)
        rec[name] = {"sumber": s.relative_to(ROOT).as_posix(), "bytes": s.stat().st_size, "sha256": sha(s),
                     "salinan_bytes": d.stat().st_size, "salinan_sha256": sha(d), "isi": summary(d)}
        ok = rec[name]["sha256"] == rec[name]["salinan_sha256"] and rec[name]["bytes"] == rec[name]["salinan_bytes"]
        rec[name]["cocok"] = ok
        print(f"{name}: {rec[name]['bytes']} B, sha256 {rec[name]['sha256'][:12]}… salinan {rec[name]['salinan_sha256'][:12]}… "
              f"-> {'COCOK' if ok else 'BEDA'}; {'; '.join(rec[name]['isi'])}")
    if not all(r["cocok"] for r in rec.values()):
        print("BERHENTI: salinan tidak cocok — target/ klip5 TIDAK dihapus")
        return 1
    (DST / "arsip.json").write_text(json.dumps({"keputusan": "T-605 keputusan akhir no. 4", "berkas": rec}, indent=1,
                                               ensure_ascii=False), encoding="utf-8")
    shutil.rmtree(SRC)
    print(f"arsip -> {DST.relative_to(ROOT).as_posix()}; {SRC.relative_to(ROOT).as_posix()} DIHAPUS")
    return 0


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if sys.argv[1:] != ["arsip"]:
        print(__doc__)
        return 2
    return cmd_arsip()


if __name__ == "__main__":
    sys.exit(main())
