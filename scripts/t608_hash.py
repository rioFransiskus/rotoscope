"""T-608 F1 (alat, F3): sha256 SEMUA berkas di work/clips/*/ + folder crop sumber work/crop/*/ + out/ -> work/t608/.

    python scripts/t608_hash.py sebelum_tahap1 | sesudah_tahap1 | sebelum_<x> | sesudah_<x> [sebelum_<y>]

Format daftar sama dengan F1 T-304 / T-607 (`<sha256>  <path relatif root, posix>` per baris, urutan `sorted(rglob)`), ditulis ke
`work/t608/f1_<label>.txt` (+ ringkasan `f1_<label>.json`). Label `sesudah_<x>` dibandingkan dengan `sebelum_<x>` atau dengan label
pembanding kedua bila diberikan (snapshot baru tanpa menimpa yang lama, mis. `sesudah_iso1b1 sebelum_iso`): harus identik
(`qc_report.json` yang berubah HANYA pada `created_utc` dicatat terpisah). Hanya MEMBACA work/clips/*, work/crop/*, dan out/;
menulis hanya ke work/t608/ (di-ignore git).
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLIPS_DIR = ROOT / "work" / "clips"
CROP_DIR = ROOT / "work" / "crop"
OUT = ROOT / "work" / "t608"
LABEL_RE = re.compile(r"^(sebelum|sesudah)_[A-Za-z0-9]+$")
QC_NAME = "qc_report.json"
CREATED_RE = re.compile(rb'("created_utc"\s*:\s*)"[^"]*"')
CREATED_MARK = rb'\1"<created_utc>"'


def file_sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def qc_normalized_sha(p: Path) -> str:
    return hashlib.sha256(CREATED_RE.sub(CREATED_MARK, p.read_bytes())).hexdigest()


def hash_files() -> list[Path]:
    files: list[Path] = []
    for base in (CLIPS_DIR, CROP_DIR):
        for c in sorted(p for p in base.iterdir() if p.is_dir()):
            files += [p for p in sorted(c.rglob("*")) if p.is_file()]
    files += [p for p in sorted((ROOT / "out").rglob("*")) if p.is_file()]
    return files


def read_list(p: Path) -> dict[str, str]:
    return {r: h for h, r in (line.split("  ", 1) for line in p.read_text(encoding="utf-8").splitlines() if line)}


def compare(ref: dict[str, str], cur: dict[str, str], ref_norm: dict[str, str], cur_norm: dict[str, str]) -> dict:
    changed = sorted(r for r in ref if r in cur and cur[r] != ref[r])
    ts_only = [r for r in changed if r.endswith("/" + QC_NAME) and ref_norm.get(r) is not None
               and ref_norm.get(r) == cur_norm.get(r)]
    res = {"ref_n": len(ref), "missing": sorted(set(ref) - set(cur)), "extra": sorted(set(cur) - set(ref)),
           "changed": [r for r in changed if r not in ts_only], "qc_created_utc_only": ts_only}
    res["identical"] = not (res["missing"] or res["changed"] or res["extra"])
    return res


def main(argv: list[str]) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if (len(argv) not in (1, 2) or not all(LABEL_RE.match(a) for a in argv)
            or (len(argv) == 2 and not (argv[0].startswith("sesudah_") and argv[1].startswith("sebelum_")))):
        print(__doc__)
        return 2
    which = argv[0]
    files = hash_files()
    cur = {p.relative_to(ROOT).as_posix(): file_sha(p) for p in files}
    norm = {p.relative_to(ROOT).as_posix(): qc_normalized_sha(p) for p in files if p.name == QC_NAME}
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"f1_{which}.txt"
    if path.exists():
        print(f"{path.relative_to(ROOT).as_posix()} sudah ada — snapshot F1 selalu nama baru")
        return 1
    path.write_text("".join(f"{h}  {r}\n" for r, h in cur.items()), encoding="utf-8")
    print(f"{path.relative_to(ROOT).as_posix()}: {len(cur)} berkas; out/ {sum(r.startswith('out/') for r in cur)}; "
          f"crop {sum(r.startswith('work/crop/') for r in cur)}; qc_report.json {len(norm)}")
    res = {"which": which, "n": len(cur), "list_sha256": file_sha(path), "qc_report_normalized_sha256": norm}
    if which.startswith("sesudah_"):
        ref_label = argv[1] if len(argv) == 2 else "sebelum_" + which.split("_", 1)[1]
        ref_path = OUT / f"f1_{ref_label}.txt"
        ref_json = OUT / f"f1_{ref_label}.json"
        if ref_path.is_file() and ref_json.is_file():
            ref_norm = json.loads(ref_json.read_text(encoding="utf-8"))["qc_report_normalized_sha256"]
            v = compare(read_list(ref_path), cur, ref_norm, norm)
            res["vs_" + ref_label] = v
            print(f"  vs {ref_label} ({v['ref_n']} berkas): hilang {len(v['missing'])}, berubah {len(v['changed'])}, "
                  f"tambahan {len(v['extra'])}, qc_report hanya created_utc {len(v['qc_created_utc_only'])} "
                  f"-> identik = {v['identical']}")
            for r in (v["missing"] + v["changed"] + v["extra"])[:20]:
                print(f"    ! {r}")
        else:
            print(f"  (pembanding work/t608/f1_{ref_label}.txt / .json tidak ada)")
    out_json = OUT / f"f1_{which}.json"
    out_json.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"  -> {out_json.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
