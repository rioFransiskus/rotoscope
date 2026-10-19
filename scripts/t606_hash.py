"""T-606 F1 (alat sekali pakai, F3): sha256 seluruh berkas yang ada di work/clips/{test,test_short,Klip2,klip3,klip4,klip5}/
(termasuk `target/` + `isolate/` klip3 dan 6 `qc_report.json`) + out/ -> work/t606/.

    python scripts/t606_hash.py before|after|before_<x>|after_<x>

Format daftar sama dengan F1 T-602/T-604/T-605 (`<sha256>  <path relatif root, posix>` per baris, urutan `sorted(rglob)` per
klip lalu out/), jadi bisa dibandingkan byte per byte dengan `work/t605/after_t4_sha256.txt` (F1 terakhir, penutupan T-605).
`after*` dibandingkan dengan `before*` berlabel sama. Di T-606 Tahap 1 TIDAK ada berkas yang boleh hilang / berubah / bertambah,
dengan SATU pengecualian: `qc_report.json` boleh berubah HANYA pada nilai `created_utc` (perilaku [2] yang diketahui, penutupan
T-605). Buktinya otomatis: sha256 "ternormalisasi" = sha256 byte berkas dengan nilai string `created_utc` diganti penanda tetap
(format / spasi / urutan lain tidak disentuh); berubah mentah + ternormalisasi sama = hanya cap waktu.
Hanya MEMBACA work/clips/* dan out/; menulis hanya ke work/t606/ (di-ignore git).
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLIPS_DIR = ROOT / "work" / "clips"
OUT = ROOT / "work" / "t606"
HASH_CLIPS = ("test", "test_short", "Klip2", "klip3", "klip4", "klip5")
PREV_F1 = ROOT / "work" / "t605" / "after_t4_sha256.txt"         # F1 terakhir (penutupan T-605)
LABEL_RE = re.compile(r"^(before|after)(_[A-Za-z0-9]+)?$")
QC_NAME = "qc_report.json"
# Tahap 3 (label berakhiran `_t3`): run produksi klip3 dengan masker [3] BOLEH mengubah / menambah / menghapus berkas klip3 stable/,
# contours/, strokes/ dan out/klip3* + out/svg/klip3/ (keputusan T-606 Tahap 2–3 no. 5); lainnya tetap harus identik.
T3_ALLOWED_RE = re.compile(r"^(work/clips/klip3/(stable|contours|strokes)/|out/klip3\.|out/svg/klip3/)")
CREATED_RE = re.compile(rb'("created_utc"\s*:\s*)"[^"]*"')
CREATED_MARK = rb'\1"<created_utc>"'
POINTS = ("work/clips/klip3/target/points.json",)


def file_sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def qc_normalized_sha(p: Path) -> tuple[str, int]:
    """(sha256 dengan nilai created_utc diganti penanda, jumlah penggantian)."""
    data, n = CREATED_RE.subn(CREATED_MARK, p.read_bytes())
    return hashlib.sha256(data).hexdigest(), n


def hash_files() -> list[Path]:
    files: list[Path] = []
    for c in HASH_CLIPS:
        files += [p for p in sorted((CLIPS_DIR / c).rglob("*")) if p.is_file()]
    files += [p for p in sorted((ROOT / "out").rglob("*")) if p.is_file()]
    return files


def read_list(p: Path) -> dict[str, str]:
    return {r: h for h, r in (line.split("  ", 1) for line in p.read_text(encoding="utf-8").splitlines() if line)}


def folder_of(r: str) -> str:
    """Kelompok untuk ringkasan: work/clips/<klip>/<folder>/ , out/svg/<klip>/ , atau berkas di akar klip / out/."""
    parts = r.split("/")
    if parts[0] == "work" and len(parts) > 4:
        return "/".join(parts[:4]) + "/"
    if parts[0] == "out" and len(parts) > 3:
        return "/".join(parts[:3]) + "/"
    return r


def compare(ref: dict[str, str], cur: dict[str, str], ref_qc_norm: dict[str, str] | None, cur_qc_norm: dict[str, str],
            allowed: re.Pattern | None = None) -> dict:
    changed = sorted(r for r in ref if r in cur and cur[r] != ref[r])
    qc_ts_only, qc_unproven = [], []
    for r in [r for r in changed if r.endswith("/" + QC_NAME)]:
        if ref_qc_norm is not None and ref_qc_norm.get(r) is not None and ref_qc_norm.get(r) == cur_qc_norm.get(r):
            qc_ts_only.append(r)
        else:
            qc_unproven.append(r)
    res = {"ref_n": len(ref), "missing": sorted(set(ref) - set(cur)), "extra": sorted(set(cur) - set(ref)),
           "changed": [r for r in changed if r not in qc_ts_only], "qc_created_utc_only": qc_ts_only,
           "qc_not_proven": qc_unproven}
    by = {}
    for kind in ("changed", "missing", "extra"):
        for r in res[kind]:
            by.setdefault(folder_of(r), {"changed": 0, "missing": 0, "extra": 0})[kind] += 1
    res["by_folder"] = dict(sorted(by.items()))
    if allowed is not None:
        res["unexpected"] = [r for k in ("changed", "missing", "extra") for r in res[k] if not allowed.match(r)]
        res["identical_except_allowed"] = not res["unexpected"]
    res["identical_existing"] = not (res["missing"] or res["changed"] or res["extra"])
    return res


def report(label: str, v: dict) -> None:
    print(f"  vs {label} ({v['ref_n']} berkas): hilang {len(v['missing'])}, berubah {len(v['changed'])}, tambahan "
          f"{len(v['extra'])}, qc_report hanya created_utc {len(v['qc_created_utc_only'])} -> identik = {v['identical_existing']}")
    shown = v["missing"] + v["changed"] + v["extra"]
    if "unexpected" in v:
        shown = v["unexpected"]                              # Tahap 3: yang diizinkan diringkas per folder saja
    for r in shown:
        print(f"    ! {r}")
    for r in v["qc_created_utc_only"]:
        print(f"    (hanya created_utc) {r}")
    for f, c in v.get("by_folder", {}).items():
        print(f"    per folder {f}: berubah {c['changed']}, hilang {c['missing']}, tambahan {c['extra']}")
    if "unexpected" in v:
        print(f"    di luar yang diizinkan Tahap 3: {len(v['unexpected'])} -> identik kecuali yang diizinkan = "
              f"{v['identical_except_allowed']}")


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
    files = hash_files()
    cur = {p.relative_to(ROOT).as_posix(): file_sha(p) for p in files}
    qc_norm = {}
    for p in files:
        if p.name == QC_NAME:
            sha, n = qc_normalized_sha(p)
            qc_norm[p.relative_to(ROOT).as_posix()] = sha
            if n != 1:
                print(f"  PERINGATAN: {p.relative_to(ROOT).as_posix()} memuat {n} created_utc (diharapkan 1)")
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{which}_sha256.txt"
    path.write_text("".join(f"{h}  {r}\n" for r, h in cur.items()), encoding="utf-8")
    print(f"{path.relative_to(ROOT).as_posix()}: {len(cur)} berkas (qc_report.json: {len(qc_norm)})")
    res = {"which": which, "n": len(cur), "list_sha256": file_sha(path),
           "points_sha256": {r: cur.get(r) for r in POINTS},
           "n_isolate_klip3": sum(r.startswith("work/clips/klip3/isolate/") for r in cur),
           "n_target_klip3": sum(r.startswith("work/clips/klip3/target/") for r in cur),
           "n_out": sum(r.startswith("out/") for r in cur),
           "qc_report_sha256": {r: h for r, h in cur.items() if r.endswith("/" + QC_NAME)},
           "qc_report_normalized_sha256": qc_norm}
    for r, h in res["points_sha256"].items():
        print(f"  {r}: {h}")
    print(f"  klip3 target/ {res['n_target_klip3']} + isolate/ {res['n_isolate_klip3']} berkas; out/ {res['n_out']} berkas")
    if PREV_F1.is_file():
        res["vs_t605_after_t4"] = compare(read_list(PREV_F1), cur, None, qc_norm)
        report("F1 penutupan T-605", res["vs_t605_after_t4"])
    m = LABEL_RE.match(which)
    if m.group(1) == "after":
        ref_path = OUT / f"before{m.group(2) or ''}_sha256.txt"
        ref_json = OUT / f"f1_before{m.group(2) or ''}.json"
        if ref_path.is_file():
            ref_norm = json.loads(ref_json.read_text(encoding="utf-8")).get("qc_report_normalized_sha256") if ref_json.is_file() else None
            res["vs_before"] = {"ref": ref_path.relative_to(ROOT).as_posix(),
                                **compare(read_list(ref_path), cur, ref_norm, qc_norm,
                                          T3_ALLOWED_RE if which.endswith("_t3") else None)}
            report(res["vs_before"]["ref"], res["vs_before"])
        else:
            print(f"  (pembanding {ref_path.relative_to(ROOT).as_posix()} tidak ada)")
    out_json = OUT / f"f1_{which}.json"
    out_json.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"  -> {out_json.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
