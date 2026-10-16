"""T-604 F1 (alat sekali pakai, F3): sha256 seluruh berkas yang ada di work/clips/{test,test_short,Klip2,klip3,klip4,klip5}/
(6 `qc_report.json` termasuk) + out/ -> work/t604/.

    python scripts/t604_hash.py before|after|before_<x>|after_<x>

Format daftar sama dengan F1 T-602 (`<sha256>  <path relatif root, posix>` per baris, urutan `sorted(rglob)` per klip lalu out/),
jadi bisa dibandingkan byte per byte dengan `work/t602/after_tutup_sha256.txt` (F1 terakhir). `after*` dibandingkan dengan
`before*` berlabel sama. Berkas baru di `work/clips/<klip>/target/` (keluaran `pick`, T-604) DIIZINKAN: dilaporkan terpisah,
tidak membuat F1 gagal; begitu juga berkas `target/pick_dryrun/` yang hilang (dihapus di Tahap 4). Hanya MEMBACA work/clips/*
dan out/; menulis hanya ke work/t604/ (di-ignore git).
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLIPS_DIR = ROOT / "work" / "clips"
OUT = ROOT / "work" / "t604"
HASH_CLIPS = ("test", "test_short", "Klip2", "klip3", "klip4", "klip5")
PREV_F1 = ROOT / "work" / "t602" / "after_tutup_sha256.txt"      # F1 terakhir (penutupan T-602)
LABEL_RE = re.compile(r"^(before|after)(_[A-Za-z0-9]+)?$")
ALLOWED_NEW_RE = re.compile(r"^work/clips/[^/]+/target/")          # folder baru `target/` boleh muncul (T-604)
DRYRUN_RE = re.compile(r"^work/clips/[^/]+/target/pick_dryrun/")   # Tahap 2–3: target/ hanya boleh berisi pick_dryrun/
POINTS_RE = re.compile(r"^work/clips/[^/]+/target/points(\.prev)?\.json$")   # Tahap 4: pick_dryrun/ dihapus, titik Rio tetap


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
    res = {"ref_n": len(ref), "missing": sorted(set(ref) - set(cur)),
           "changed": sorted(r for r in ref if r in cur and cur[r] != ref[r]),
           "extra_allowed_target": [r for r in extra if ALLOWED_NEW_RE.match(r)],
           "extra_other": [r for r in extra if not ALLOWED_NEW_RE.match(r)]}
    res["target_not_dryrun"] = [r for r in res["extra_allowed_target"] if not DRYRUN_RE.match(r)]
    res["target_not_points"] = [r for r in res["extra_allowed_target"] if not POINTS_RE.match(r)]
    res["missing_dryrun"] = [r for r in res["missing"] if DRYRUN_RE.match(r)]      # dihapus di Tahap 4 (diizinkan)
    missing_other = [r for r in res["missing"] if not DRYRUN_RE.match(r)]
    res["identical_existing"] = not (missing_other or res["changed"] or res["extra_other"])
    return res


def print_target(v: dict) -> None:
    print(f"    target/: di luar pick_dryrun/ {len(v['target_not_dryrun'])}, selain points(.prev).json "
          f"{len(v['target_not_points'])}, pick_dryrun/ yang hilang {len(v['missing_dryrun'])}")
    for r in v["extra_allowed_target"]:
        print(f"    + {r}")
    for r in v["missing_dryrun"]:
        print(f"    - {r}")


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
    res = {"which": which, "n": len(cur), "n_qc_report": n_qc, "list_sha256": file_sha(path)}
    if PREV_F1.is_file():
        res["vs_t602_after_tutup"] = compare(read_list(PREV_F1), cur)
        v = res["vs_t602_after_tutup"]
        print(f"  vs F1 penutupan T-602 ({v['ref_n']} berkas): hilang {len(v['missing'])}, berubah {len(v['changed'])}, "
              f"tambahan target/ {len(v['extra_allowed_target'])}, tambahan lain {len(v['extra_other'])} "
              f"-> identik (berkas lama) = {v['identical_existing']}")
        print_target(v)
    m = LABEL_RE.match(which)
    if m.group(1) == "after":
        ref_path = OUT / f"before{m.group(2) or ''}_sha256.txt"
        if ref_path.is_file():
            res["vs_before"] = {"ref": ref_path.relative_to(ROOT).as_posix(), **compare(read_list(ref_path), cur)}
            v = res["vs_before"]
            print(f"  vs {v['ref']} ({v['ref_n']} berkas): hilang {len(v['missing'])}, berubah {len(v['changed'])}, "
                  f"tambahan target/ {len(v['extra_allowed_target'])}, tambahan lain {len(v['extra_other'])} "
                  f"-> identik (berkas lama) = {v['identical_existing']}")
            print_target(v)
        else:
            print(f"  (pembanding {ref_path.relative_to(ROOT).as_posix()} tidak ada)")
    out_json = OUT / f"f1_{which}.json"
    out_json.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"  -> {out_json.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
