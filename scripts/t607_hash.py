"""T-607 F1 (alat, F3): sha256 SEMUA berkas di work/clips/*/ (frames/ seg/ depth/ isolate/ target/ stable/ contours/ strokes/ +
meta.json + qc_report.json, apa pun yang ada) + out/ -> work/t607/.

    python scripts/t607_hash.py before | after_1a | after_1b | after_<x>

Format daftar sama dengan F1 T-602 ... T-305f (`<sha256>  <path relatif root, posix>` per baris, urutan `sorted(rglob)` per klip lalu
out/). `before` dibandingkan dengan F1 terakhir (penutupan T-305f, `work/t305f/after_t4_sha256.txt`); setiap label lain dibandingkan
dengan `before`. Sebelum Tahap 4 klip asli + out/ TIDAK boleh berubah, dengan SATU pengecualian yang dibuktikan otomatis:
`qc_report.json` yang berubah HANYA pada nilai `created_utc` (sha256 "ternormalisasi" sama; perilaku [2] yang diketahui).
Hanya MEMBACA work/clips/* dan out/; menulis hanya ke work/t607/ (di-ignore git).
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLIPS_DIR = ROOT / "work" / "clips"
OUT = ROOT / "work" / "t607"
PREV_F1 = ROOT / "work" / "t305f" / "after_t4_sha256.txt"        # F1 terakhir (penutupan T-305f)
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
    files += [p for p in sorted((ROOT / "out").rglob("*")) if p.is_file()]
    return files


def read_list(p: Path) -> dict[str, str]:
    return {r: h for h, r in (line.split("  ", 1) for line in p.read_text(encoding="utf-8").splitlines() if line)}


def folder_of(r: str) -> str:
    parts = r.split("/")
    if parts[0] == "work" and len(parts) > 4:
        return "/".join(parts[:4]) + "/"
    if parts[0] == "out" and len(parts) > 3:
        return "/".join(parts[:3]) + "/"
    return r


def compare(ref: dict[str, str], cur: dict[str, str], ref_norm: dict[str, str] | None, cur_norm: dict[str, str]) -> dict:
    changed = sorted(r for r in ref if r in cur and cur[r] != ref[r])
    ts_only = [r for r in changed if r.endswith("/" + QC_NAME) and ref_norm is not None
               and ref_norm.get(r) is not None and ref_norm.get(r) == cur_norm.get(r)]
    res = {"ref_n": len(ref), "missing": sorted(set(ref) - set(cur)), "extra": sorted(set(cur) - set(ref)),
           "changed": [r for r in changed if r not in ts_only], "qc_created_utc_only": ts_only}
    by: dict[str, dict[str, int]] = {}
    for kind in ("changed", "missing", "extra"):
        for r in res[kind]:
            by.setdefault(folder_of(r), {"changed": 0, "missing": 0, "extra": 0})[kind] += 1
    res["by_folder"] = dict(sorted(by.items()))
    res["identical"] = not (res["missing"] or res["changed"] or res["extra"])
    return res


def report(label: str, v: dict) -> None:
    print(f"  vs {label} ({v['ref_n']} berkas): hilang {len(v['missing'])}, berubah {len(v['changed'])}, tambahan "
          f"{len(v['extra'])}, qc_report hanya created_utc {len(v['qc_created_utc_only'])} -> identik = {v['identical']}")
    for r in (v["missing"] + v["changed"] + v["extra"])[:20]:
        print(f"    ! {r}")
    for f, c in v["by_folder"].items():
        print(f"    per folder {f}: berubah {c['changed']}, hilang {c['missing']}, tambahan {c['extra']}")


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
    clips = sorted({r.split("/")[2] for r in cur if r.startswith("work/clips/")})
    per_clip = {c: sum(r.startswith(f"work/clips/{c}/") for r in cur) for c in clips}
    print(f"{path.relative_to(ROOT).as_posix()}: {len(cur)} berkas; per klip {per_clip}; out/ "
          f"{sum(r.startswith('out/') for r in cur)}; qc_report.json {len(norm)}")
    res = {"which": which, "n": len(cur), "list_sha256": file_sha(path), "per_clip": per_clip,
           "n_out": sum(r.startswith("out/") for r in cur), "qc_report_normalized_sha256": norm}
    if which == "before":
        if PREV_F1.is_file():
            res["vs_t305f_after_t4"] = compare(read_list(PREV_F1), cur, None, norm)
            report("F1 penutupan T-305f", res["vs_t305f_after_t4"])
    else:
        ref_path, ref_json = OUT / "before_sha256.txt", OUT / "f1_before.json"
        if ref_path.is_file():
            ref_norm = json.loads(ref_json.read_text(encoding="utf-8")).get("qc_report_normalized_sha256") if ref_json.is_file() else None
            res["vs_before"] = compare(read_list(ref_path), cur, ref_norm, norm)
            report("before", res["vs_before"])
        else:
            print("  (pembanding work/t607/before_sha256.txt tidak ada)")
    out_json = OUT / f"f1_{which}.json"
    out_json.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"  -> {out_json.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
