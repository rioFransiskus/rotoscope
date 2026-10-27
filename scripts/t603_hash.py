"""T-603 F1 (alat, F3): sha256 SEMUA berkas di work/clips/*/ + work/crop/ + out/ -> work/t603/.

    python scripts/t603_hash.py before | after_1 | after_<x>
    python scripts/t603_hash.py k5_snap<n>                   -> snapshot scratch K5 (work/t603_scratch/k5/) → work/t603/k5_snap<n>_sha256.txt
    python scripts/t603_hash.py k5_diff k5_snap4 k5_snap6    -> seg/ + depth/ yang ada di kedua snapshot harus identik (K3)
    python scripts/t603_hash.py t304_cmp <label>             -> BACA-SAJA: produksi sekarang vs F1 akhir T-304 (work/t304/after_3_sha256.txt);
                                                                qc_report.json tanpa created_utc; menulis work/t603/<label>.json + <label>_list.txt

Format daftar sama dengan scripts/t607_hash.py (`<sha256>  <path relatif root, posix>` per baris). Bedanya: work/crop/
ikut dihitung (F1 T-603). `before` dibandingkan dengan F1 terakhir (penutupan T-607, `work/t607/after_4f_sha256.txt`);
setiap label lain dibandingkan dengan `before`. `qc_report.json` yang berubah HANYA pada `created_utc` dinormalisasi
(perilaku [2] yang sudah diketahui, sama dengan t607_hash.py).
Hanya MEMBACA work/ dan out/; menulis hanya ke work/t603/ (di-ignore git).
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
OUT = ROOT / "work" / "t603"
PREV_F1 = ROOT / "work" / "t607" / "after_4f_sha256.txt"          # F1 terakhir (penutupan T-607)
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
    files += [p for p in sorted(CROP_DIR.rglob("*")) if p.is_file()]
    files += [p for p in sorted((ROOT / "out").rglob("*")) if p.is_file()]
    return files


def read_list(p: Path) -> dict[str, str]:
    return {r: h for h, r in (line.split("  ", 1) for line in p.read_text(encoding="utf-8").splitlines() if line)}


def folder_of(r: str) -> str:
    parts = r.split("/")
    if parts[0] == "work" and parts[1] == "clips" and len(parts) > 4:
        return "/".join(parts[:4]) + "/"
    if parts[0] == "work" and len(parts) > 3:
        return "/".join(parts[:3]) + "/"
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


K5_ROOT = ROOT / "work" / "t603_scratch" / "k5"
K5_SEG_DEPTH = ("/seg/", "/depth/")


def k5_snapshot(label: str) -> int:
    """Snapshot scratch K5 (work/t603_scratch/k5/, seluruh isi berkas) → work/t603/<label>_sha256.txt."""
    files = sorted(p for p in K5_ROOT.rglob("*") if p.is_file())
    cur = {p.relative_to(ROOT).as_posix(): file_sha(p) for p in files}
    # qc_report.json: sha ternormalisasi (tanpa created_utc) ikut dicatat sebagai baris "<sha>  <path>#norm" (butir C)
    norm = {p.relative_to(ROOT).as_posix() + "#norm": qc_normalized_sha(p) for p in files if p.name == QC_NAME}
    OUT.mkdir(parents=True, exist_ok=True)
    lines = [f"{h}  {r}\n" for r, h in {**cur, **norm}.items()]
    (OUT / f"{label}_sha256.txt").write_text("".join(lines), encoding="utf-8")
    print(f"{label}: {len(cur)} berkas di {K5_ROOT.relative_to(ROOT).as_posix()}; seg/depth {sum(any(s in r for s in K5_SEG_DEPTH) for r in cur)}")
    return 0


def jsonl_prefix(path: Path, want_sha: str) -> tuple[int, int] | None:
    """Cari batas BARIS pada berkas sekarang yang sha256 awalannya = want_sha (snapshot). Return (baris, byte) atau None.
    Bukti tanpa menyimpan isi snapshot: sha256 yang sama = byte yang sama (tanpa tebakan)."""
    h = hashlib.sha256()
    data = path.read_bytes()
    pos = 0
    n_lines = 0
    for line in data.splitlines(keepends=True):
        h.update(line)
        pos += len(line)
        n_lines += 1
        if h.hexdigest() == want_sha:
            return n_lines, pos
    return None


def k5_diff(a: str, b: str) -> int:
    """Bandingkan dua snapshot K5 (keputusan Rio no. 4 + 5):
    - berkas biasa (frame seg/depth, manifest, …) harus IDENTIK di kedua snapshot;
    - `*.jsonl` (log append-only): snapshot a harus AWALAN dari berkas snapshot b (dibuktikan dengan sha256 awalan per batas baris)
      dan snapshot b harus sama dengan berkas sekarang; bila tidak terbukti → dilaporkan TIDAK TERBUKTI, bukan dianggap sama;
    - qc_report.json: beda hanya created_utc bila sha ternormalisasi sama (butir C).
    Berkas hanya di b (baru) dilaporkan, tidak gagal. Exit 0 bila semua syarat terpenuhi."""
    ra = read_list(OUT / f"{a}_sha256.txt")
    rb = read_list(OUT / f"{b}_sha256.txt")
    fail, rep = [], []
    for r in sorted(r for r in ra if not r.endswith("#norm")):
        if r not in rb:
            fail.append(r)
            rep.append(f"HILANG  {r}")
            continue
        if ra[r] == rb[r]:
            continue
        path = ROOT / r
        if r.endswith(".jsonl"):
            if file_sha(path) != rb[r]:
                fail.append(r)
                rep.append(f"TIDAK TERBUKTI (berkas berubah sesudah snapshot {b})  {r}")
                continue
            proof = jsonl_prefix(path, ra[r])
            if proof is None:
                fail.append(r)
                rep.append(f"TIDAK TERBUKTI awalan  {r}")
            else:
                rep.append(f"awalan terbukti ({proof[0]} baris, {proof[1]} byte dari {b})  {r}")
        elif r.endswith("qc_report.json") and ra.get(r + "#norm") is not None and rb.get(r + "#norm") == ra[r + "#norm"]:
            rep.append(f"beda hanya created_utc  {r}")
        else:
            fail.append(r)
            rep.append(f"BERBEDA  {r}")
    new = [r for r in rb if not r.endswith("#norm") and r not in ra]
    n_common = sum(1 for r in ra if r in rb and not r.endswith("#norm"))
    for line in rep:
        print("  " + line)
    print(f"{a} vs {b}: {n_common} berkas bersama; syarat gagal {len(fail)}; berkas baru di {b}: {len(new)}")
    print(f"identik (kecuali bukti awalan / created_utc di atas) = {not fail}")
    return 0 if not fail else 1


T304_LIST = ROOT / "work" / "t304" / "after_3_sha256.txt"          # F1 AKHIR T-304 (docs/04 "Keputusan akhir T-304", butir 7)
T304_REC = ROOT / "work" / "t304" / "f1_after_3.json"             # berisi qc_report_normalized_sha256 (tanpa created_utc)


def t304_cmp(label: str) -> int:
    """Baca-saja: bandingkan produksi SEKARANG (work/clips/*/ + work/crop/ + out/) dengan F1 akhir T-304 (10 719 berkas).
    Berkas biasa: sha256 sama. qc_report.json: sha ternormalisasi (tanpa created_utc) dari referensi vs sekarang.
    Menulis HANYA berkas baru: work/t603/<label>_list.txt dan work/t603/<label>.json (tidak menimpa referensi apa pun)."""
    ref = read_list(T304_LIST)
    ref_norm = json.loads(T304_REC.read_text(encoding="utf-8"))["qc_report_normalized_sha256"]
    files = hash_files()
    cur = {p.relative_to(ROOT).as_posix(): file_sha(p) for p in files}
    cur_norm = {p.relative_to(ROOT).as_posix(): qc_normalized_sha(p) for p in files if p.name == QC_NAME}
    missing = sorted(set(ref) - set(cur))
    extra = sorted(set(cur) - set(ref))
    changed, qc_ok = [], []
    for r in sorted(set(ref) & set(cur)):
        if r.endswith("/" + QC_NAME):
            (qc_ok if ref_norm.get(r) == cur_norm.get(r) else changed).append(r)
        elif ref[r] != cur[r]:
            changed.append(r)
    res = {"referensi": T304_LIST.relative_to(ROOT).as_posix(), "rekaman_qc": T304_REC.relative_to(ROOT).as_posix(),
           "n_referensi": len(ref), "n_sekarang": len(cur), "hilang": missing, "berubah": changed, "tambahan": extra,
           "qc_report_sha_ternormalisasi_sama": len(qc_ok), "identik": not (missing or changed or extra)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{label}_list.txt").write_text("".join(f"{h}  {r}\n" for r, h in cur.items()), encoding="utf-8")
    (OUT / f"{label}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"referensi {res['referensi']}: {len(ref)} berkas; sekarang {len(cur)} berkas")
    print(f"hilang {len(missing)}, berubah {len(changed)}, tambahan {len(extra)}; qc_report (tanpa created_utc) sama {len(qc_ok)}/{len(qc_ok) + sum(r.endswith('/' + QC_NAME) for r in changed)}")
    for r in (missing + changed + extra)[:50]:
        print(f"    ! {r}")
    print(f"identik = {res['identik']}  -> work/t603/{label}.json")
    return 0 if res["identik"] else 1


def main(argv: list[str]) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if len(argv) == 2 and argv[0] == "t304_cmp":
        return t304_cmp(argv[1])
    if len(argv) == 3 and argv[0] == "k5_diff":
        return k5_diff(argv[1], argv[2])
    if len(argv) == 1 and argv[0].startswith("k5_snap"):
        return k5_snapshot(argv[0])
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
    n_crop = sum(r.startswith("work/crop/") for r in cur)
    print(f"{path.relative_to(ROOT).as_posix()}: {len(cur)} berkas; per klip {per_clip}; work/crop/ {n_crop}; out/ "
          f"{sum(r.startswith('out/') for r in cur)}; qc_report.json {len(norm)}")
    res = {"which": which, "n": len(cur), "list_sha256": file_sha(path), "per_clip": per_clip, "n_crop": n_crop,
           "n_out": sum(r.startswith("out/") for r in cur), "qc_report_normalized_sha256": norm}
    if which == "before":
        if PREV_F1.is_file():
            res["vs_t607_after_4f"] = compare(read_list(PREV_F1), cur, None, norm)
            report("F1 penutupan T-607", res["vs_t607_after_4f"])
    else:
        ref_path, ref_json = OUT / "before_sha256.txt", OUT / "f1_before.json"
        if ref_path.is_file():
            ref_norm = json.loads(ref_json.read_text(encoding="utf-8")).get("qc_report_normalized_sha256") if ref_json.is_file() else None
            res["vs_before"] = compare(read_list(ref_path), cur, ref_norm, norm)
            report("before", res["vs_before"])
        else:
            print("  (pembanding work/t603/before_sha256.txt tidak ada)")
    out_json = OUT / f"f1_{which}.json"
    out_json.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"  -> {out_json.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
