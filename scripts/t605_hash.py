"""T-605 F1 (alat sekali pakai, F3): sha256 seluruh berkas yang ada di work/clips/{test,test_short,Klip2,klip3,klip4,klip5}/
(6 `qc_report.json` + `target/points.json` klip3/klip5 termasuk) + out/ -> work/t605/.

    python scripts/t605_hash.py before|after|before_<x>|after_<x>

Format daftar sama dengan F1 T-602/T-604 (`<sha256>  <path relatif root, posix>` per baris, urutan `sorted(rglob)` per klip
lalu out/), jadi bisa dibandingkan byte per byte dengan `work/t604/after_t4_sha256.txt` (F1 terakhir, penutupan T-604).
`after*` dibandingkan dengan `before*` berlabel sama. Berkas baru di `work/clips/<klip>/isolate/` (keluaran stage [1b],
mulai Tahap 3) DIIZINKAN: dilaporkan terpisah, tidak membuat F1 gagal. Di Tahap 1 tidak boleh ada tambahan apa pun. Tahap 4:
berkas yang hilang di `work/clips/klip5/target/` dan `work/clips/klip5/isolate/` (diarsipkan / dihapus sesuai keputusan akhir
T-605 no. 4) dilaporkan terpisah; berkas lain yang hilang tetap membuat F1 gagal.
Hanya MEMBACA work/clips/* dan out/; menulis hanya ke work/t605/ (di-ignore git).
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLIPS_DIR = ROOT / "work" / "clips"
OUT = ROOT / "work" / "t605"
HASH_CLIPS = ("test", "test_short", "Klip2", "klip3", "klip4", "klip5")
PREV_F1 = ROOT / "work" / "t604" / "after_t4_sha256.txt"         # F1 terakhir (penutupan T-604)
LABEL_RE = re.compile(r"^(before|after)(_[A-Za-z0-9]+)?$")
ALLOWED_NEW_RE = re.compile(r"^work/clips/[^/]+/isolate/")         # folder baru `isolate/` boleh muncul (Tahap 3)
# Tahap 4 (keputusan akhir T-605 no. 4): titik klip5 diarsipkan lalu target/ klip5 dihapus; isolate/ klip5 ikut terhapus.
ALLOWED_MISSING_RE = re.compile(r"^work/clips/klip5/(target|isolate)/")
POINTS = ("work/clips/klip3/target/points.json", "work/clips/klip5/target/points.json")


def file_sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_files() -> list[Path]:
    files: list[Path] = []
    for c in HASH_CLIPS:
        files += [p for p in sorted((CLIPS_DIR / c).rglob("*")) if p.is_file()]
    files += [p for p in sorted((ROOT / "out").rglob("*")) if p.is_file()]
    return files


def read_list(p: Path) -> dict[str, str]:
    return {r: h for h, r in (line.split("  ", 1) for line in p.read_text(encoding="utf-8").splitlines() if line)}


def compare(ref: dict[str, str], cur: dict[str, str]) -> dict:
    extra = sorted(set(cur) - set(ref))
    missing = sorted(set(ref) - set(cur))
    res = {"ref_n": len(ref), "missing": [r for r in missing if not ALLOWED_MISSING_RE.match(r)],
           "missing_allowed_klip5": [r for r in missing if ALLOWED_MISSING_RE.match(r)],
           "changed": sorted(r for r in ref if r in cur and cur[r] != ref[r]),
           "extra_allowed_isolate": [r for r in extra if ALLOWED_NEW_RE.match(r)],
           "extra_other": [r for r in extra if not ALLOWED_NEW_RE.match(r)]}
    res["identical_existing"] = not (res["missing"] or res["changed"] or res["extra_other"])
    return res


def report(label: str, v: dict) -> None:
    print(f"  vs {label} ({v['ref_n']} berkas): hilang {len(v['missing'])} (+ dihapus sesuai keputusan, klip5 target/ + "
          f"isolate/: {len(v['missing_allowed_klip5'])}), berubah {len(v['changed'])}, "
          f"tambahan isolate/ {len(v['extra_allowed_isolate'])}, tambahan lain {len(v['extra_other'])} "
          f"-> identik (berkas lama) = {v['identical_existing']}")
    for r in v["missing"] + v["changed"] + v["extra_other"]:
        print(f"    ! {r}")


def main(argv: list[str]) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if len(argv) != 1 or not LABEL_RE.match(argv[0]):
        print(__doc__)
        return 2
    which = argv[0]
    missing_clips = [c for c in HASH_CLIPS if not (CLIPS_DIR / c).is_dir()]
    if missing_clips:
        print(f"folder klip tidak ada: {missing_clips}")
        return 1
    cur = {p.relative_to(ROOT).as_posix(): file_sha(p) for p in hash_files()}
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{which}_sha256.txt"
    path.write_text("".join(f"{h}  {r}\n" for r, h in cur.items()), encoding="utf-8")
    n_qc = sum(r.endswith("/qc_report.json") for r in cur)
    print(f"{path.relative_to(ROOT).as_posix()}: {len(cur)} berkas (qc_report.json: {n_qc})")
    res = {"which": which, "n": len(cur), "n_qc_report": n_qc, "list_sha256": file_sha(path),
           "points_sha256": {r: cur.get(r) for r in POINTS},
           "qc_report_sha256": {r: h for r, h in cur.items() if r.endswith("/qc_report.json")}}
    for r, h in res["points_sha256"].items():
        print(f"  {r}: {h}")
    if PREV_F1.is_file():
        res["vs_t604_after_t4"] = compare(read_list(PREV_F1), cur)
        report("F1 penutupan T-604", res["vs_t604_after_t4"])
    m = LABEL_RE.match(which)
    if m.group(1) == "after":
        ref_path = OUT / f"before{m.group(2) or ''}_sha256.txt"
        if ref_path.is_file():
            res["vs_before"] = {"ref": ref_path.relative_to(ROOT).as_posix(), **compare(read_list(ref_path), cur)}
            report(res["vs_before"]["ref"], res["vs_before"])
        else:
            print(f"  (pembanding {ref_path.relative_to(ROOT).as_posix()} tidak ada)")
    out_json = OUT / f"f1_{which}.json"
    out_json.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"  -> {out_json.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
