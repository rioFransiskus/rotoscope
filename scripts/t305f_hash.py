"""T-305f F1 (alat sekali pakai, F3): sha256 seluruh berkas work/clips/{test,test_short,Klip2,klip3,klip4,klip5}/ + out/ -> work/t305f/.

    python scripts/t305f_hash.py before|after|before_<x>|after_<x>

Memakai ulang `scripts/t606_hash.py` (daftar berkas, format `<sha256>  <path>`, pembanding, bukti `qc_report.json` hanya `created_utc`
lewat sha256 ternormalisasi). Pembanding tambahan: F1 terakhir `work/t606/after_t4_sha256.txt` (penutupan T-606). Di Tahap 1 TIDAK ada
berkas yang boleh hilang / berubah / bertambah; satu-satunya pengecualian = nilai `created_utc` di `qc_report.json` (dibuktikan).
Hanya MEMBACA work/clips/* dan out/; menulis hanya ke work/t305f/ (di-ignore git).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import t606_hash as h  # noqa: E402

OUT = ROOT / "work" / "t305f"
PREV_F1 = ROOT / "work" / "t606" / "after_t4_sha256.txt"         # F1 terakhir (penutupan T-606)


def main(argv: list[str]) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if len(argv) != 1 or not h.LABEL_RE.match(argv[0]):
        print(__doc__)
        return 2
    which = argv[0]
    missing_clips = [c for c in h.HASH_CLIPS if not (h.CLIPS_DIR / c).is_dir()]
    if missing_clips:
        print(f"folder klip tidak ada: {missing_clips}")
        return 1
    files = h.hash_files()
    cur = {p.relative_to(ROOT).as_posix(): h.file_sha(p) for p in files}
    qc_norm = {}
    for p in files:
        if p.name == h.QC_NAME:
            sha, n = h.qc_normalized_sha(p)
            qc_norm[p.relative_to(ROOT).as_posix()] = sha
            if n != 1:
                print(f"  PERINGATAN: {p.relative_to(ROOT).as_posix()} memuat {n} created_utc (diharapkan 1)")
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{which}_sha256.txt"
    path.write_text("".join(f"{v}  {r}\n" for r, v in cur.items()), encoding="utf-8")
    print(f"{path.relative_to(ROOT).as_posix()}: {len(cur)} berkas (qc_report.json: {len(qc_norm)})")
    res = {"which": which, "n": len(cur), "list_sha256": h.file_sha(path),
           "n_out": sum(r.startswith("out/") for r in cur),
           "qc_report_sha256": {r: v for r, v in cur.items() if r.endswith("/" + h.QC_NAME)},
           "qc_report_normalized_sha256": qc_norm}
    if PREV_F1.is_file():
        res["vs_t606_after_t4"] = h.compare(h.read_list(PREV_F1), cur, None, qc_norm)
        h.report("F1 penutupan T-606", res["vs_t606_after_t4"])
    m = h.LABEL_RE.match(which)
    if m.group(1) == "after":
        ref_path = OUT / f"before{m.group(2) or ''}_sha256.txt"
        ref_json = OUT / f"f1_before{m.group(2) or ''}.json"
        if ref_path.is_file():
            ref_norm = json.loads(ref_json.read_text(encoding="utf-8")).get("qc_report_normalized_sha256") if ref_json.is_file() else None
            res["vs_before"] = {"ref": ref_path.relative_to(ROOT).as_posix(),
                                **h.compare(h.read_list(ref_path), cur, ref_norm, qc_norm)}
            h.report(res["vs_before"]["ref"], res["vs_before"])
        else:
            print(f"  (pembanding {ref_path.relative_to(ROOT).as_posix()} tidak ada)")
    out_json = OUT / f"f1_{which}.json"
    out_json.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"  -> {out_json.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
