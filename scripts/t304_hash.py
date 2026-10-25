"""T-304 F1 (alat, F3): sha256 SEMUA berkas di work/clips/*/ (frames/ seg/ depth/ isolate/ target/ stable/ contours/ strokes/ +
meta.json + qc_report.json, apa pun yang ada) + folder crop sumber work/crop/*/ + out/ -> work/t304/.

    python scripts/t304_hash.py before | after_1 | after_2 | after_3 | ...

Format daftar sama dengan F1 T-607 (`<sha256>  <path relatif root, posix>` per baris, urutan `sorted(rglob)`). Setiap label
selain `before` dibandingkan dengan `before`: sebelum Tahap 4 klip asli, crop, dan out/ TIDAK boleh berubah. `qc_report.json` yang
berubah HANYA pada nilai `created_utc` dicatat terpisah (perilaku [2] yang diketahui).
Hanya MEMBACA work/clips/*, work/crop/*, dan out/; menulis hanya ke work/t304/ (di-ignore git).
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
OUT = ROOT / "work" / "t304"
LABEL_RE = re.compile(r"^(before|after_[A-Za-z0-9]+)$")
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
    for c in sorted(p for p in CLIPS_DIR.iterdir() if p.is_dir()):
        files += [p for p in sorted(c.rglob("*")) if p.is_file()]
    for c in sorted(p for p in CROP_DIR.iterdir() if p.is_dir()):
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
    if len(argv) != 1 or not LABEL_RE.match(argv[0]):
        print(__doc__)
        return 2
    which = argv[0]
    files = hash_files()
    cur = {p.relative_to(ROOT).as_posix(): file_sha(p) for p in files}
    norm = {p.relative_to(ROOT).as_posix(): qc_normalized_sha(p) for p in files if p.name == QC_NAME}
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{which}_sha256.txt"
    path.write_text("".join(f"{h}  {r}\n" for r, h in cur.items()), encoding="utf-8")
    print(f"{path.relative_to(ROOT).as_posix()}: {len(cur)} berkas; out/ {sum(r.startswith('out/') for r in cur)}; "
          f"qc_report.json {len(norm)}")
    res = {"which": which, "n": len(cur), "list_sha256": file_sha(path), "qc_report_normalized_sha256": norm}
    if which != "before":
        ref_path = OUT / "before_sha256.txt"
        ref_json = OUT / "f1_before.json"
        if ref_path.is_file() and ref_json.is_file():
            ref_norm = json.loads(ref_json.read_text(encoding="utf-8"))["qc_report_normalized_sha256"]
            v = compare(read_list(ref_path), cur, ref_norm, norm)
            res["vs_before"] = v
            print(f"  vs before ({v['ref_n']} berkas): hilang {len(v['missing'])}, berubah {len(v['changed'])}, tambahan "
                  f"{len(v['extra'])}, qc_report hanya created_utc {len(v['qc_created_utc_only'])} -> identik = {v['identical']}")
            for r in (v["missing"] + v["changed"] + v["extra"])[:20]:
                print(f"    ! {r}")
        else:
            print("  (pembanding work/t304/before_sha256.txt / f1_before.json tidak ada)")
    out_json = OUT / f"f1_{which}.json"
    out_json.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"  -> {out_json.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
