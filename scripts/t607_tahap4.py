"""T-607 Tahap 4 (alat, F3; CPU, GPU DISEMBUNYIKAN di setiap CLI yang dijalankan): Klip2 + klip5 asli diganti hasil scratch t3
(persetujuan rancangan no. 7 + rancangan 1a (h): salin → cek sha256 → ganti, HANYA bila `meta.json` + config sama) + bukti `out/`
test, test_short, klip3, klip4 byte-identik dengan kode export baru (keputusan Tahap 3b no. 5).

    python scripts/t607_tahap4.py config                  # config t3 = configs/default.yaml kecuali `paths`
    python scripts/t607_tahap4.py simpan                  # out/ lama Klip2 + klip5 -> work/t607/out_lama/ (+ cek sha256)
    python scripts/t607_tahap4.py jalankan LABEL ARGV...  # python -m rotoscope ARGV (CUDA_VISIBLE_DEVICES="") -> work/t607/tahap4/LABEL.log
    python scripts/t607_tahap4.py cek KLIP                # klip asli sesudah ingest ber-kotak vs t3: meta, frames, manifest, crop/
    python scripts/t607_tahap4.py ganti KLIP              # hanya bila `cek` lolos: t3 -> <item>.t607baru -> cek sha256 -> ganti
    python scripts/t607_tahap4.py banding KLIP            # out/ asli vs t3/out: MP4, SVG, export.json (tanpa waktu)
    python scripts/t607_tahap4.py ekspor4                 # export 4 klip lain ke work/t607_scratch/t4/out (klip asli dibaca) vs out/
    python scripts/t607_tahap4.py data                    # F4: ringkasan angka work/t607/*.json -> scripts/t607_data.json (ter-track)

Klip asli HANYA ditulis oleh `ganti` dan oleh CLI produksi lewat `jalankan`. Hasil -> work/t607/tahap4.json (digabung per subperintah).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CLIPS = ROOT / "work" / "clips"
CROP = ROOT / "work" / "crop"
OUTD = ROOT / "out"
T3 = ROOT / "work" / "t607_scratch" / "t3"
T3_CFG = T3 / "config.yaml"
T4_CFG = ROOT / "work" / "t607_scratch" / "t4" / "config.yaml"
T4_OUT = ROOT / "work" / "t607_scratch" / "t4" / "out"
DATA = ROOT / "work" / "t607"
RES = DATA / "tahap4.json"
LOGS = DATA / "tahap4"
LAMA = DATA / "out_lama"
CROPPED = ("Klip2", "klip5")
UNCHANGED = ("test", "test_short", "klip3", "klip4")
ITEMS = ("seg", "depth", "stable", "contours", "strokes", "qc_report.json")    # hasil t3 yang dipindah (frames/ + meta.json dari ingest)
NEW_SUFFIX = ".t607baru"
TIME_KEYS = ("utc", "time", "elapsed", "seconds")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def tree(p: Path) -> dict[str, str]:
    """{path relatif posix: sha256} semua berkas di bawah p (atau satu berkas)."""
    if p.is_file():
        return {p.name: sha(p)}
    return {f.relative_to(p).as_posix(): sha(f) for f in sorted(p.rglob("*")) if f.is_file()}


def jload(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def save(section: str, key: str | None, value) -> None:
    res = jload(RES) if RES.is_file() else {}
    if key is None:
        res[section] = value
    else:
        res.setdefault(section, {})[key] = value
    RES.parent.mkdir(parents=True, exist_ok=True)
    RES.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"  -> {RES.relative_to(ROOT).as_posix()} [{section}{'/' + key if key else ''}]")


def strip_time(o):
    if isinstance(o, dict):
        return {k: strip_time(v) for k, v in o.items() if not any(t in k.lower() for t in TIME_KEYS)}
    if isinstance(o, list):
        return [strip_time(v) for v in o]
    return o


def diff_keys(a, b, path: str = "") -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            out += diff_keys(a.get(k), b.get(k), f"{path}.{k}" if path else k) if k in a and k in b else [f"{path}.{k}"]
        return out
    return [] if a == b else [path or "<root>"]


def plain(o):
    """Config → dict / list / skalar (dataclasses.asdict gagal pada mappingproxy)."""
    if dataclasses.is_dataclass(o):
        return {f.name: plain(getattr(o, f.name)) for f in dataclasses.fields(o)}
    if hasattr(o, "items"):
        return {str(k): plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [plain(v) for v in o]
    return str(o) if isinstance(o, Path) else o


def cmd_config() -> None:
    from rotoscope.config import load_pipeline

    d = plain(load_pipeline(ROOT / "configs" / "default.yaml"))
    s = plain(load_pipeline(T3_CFG))
    keys = diff_keys({k: v for k, v in d.items() if k != "paths"}, {k: v for k, v in s.items() if k != "paths"})
    paths = diff_keys(d["paths"], s["paths"])
    print(f"config t3 vs default: beda di luar paths {keys or 'TIDAK ADA'}; beda di paths {paths}")
    save("config", None, {"beda_selain_paths": keys, "beda_paths": paths, "sama": not keys})


def out_files(clip: str) -> list[Path]:
    """Berkas out/ milik klip: <klip>.* (MP4, export.json, preview) + out/svg/<klip>/."""
    return sorted(p for p in OUTD.iterdir() if p.is_file() and p.name.split(".")[0] == clip) + [OUTD / "svg" / clip]


def cmd_simpan() -> None:
    res = {}
    for clip in CROPPED:
        for src in out_files(clip):
            dst = LAMA / src.relative_to(OUTD)
            if dst.exists():
                shutil.rmtree(dst) if dst.is_dir() else dst.unlink()
            dst.parent.mkdir(parents=True, exist_ok=True)
            (shutil.copytree if src.is_dir() else shutil.copy2)(src, dst)
            a, b = tree(src), tree(dst)
            res[src.relative_to(OUTD).as_posix()] = {"berkas": len(a), "bytes": sum(f.stat().st_size for f in
                                                                                   ([src] if src.is_file() else src.rglob("*"))
                                                                                   if f.is_file()), "sha256_sama": a == b}
            print(f"{src.relative_to(ROOT).as_posix()} -> {dst.relative_to(ROOT).as_posix()}: {len(a)} berkas, sama {a == b}")
    ok = all(v["sha256_sama"] for v in res.values())
    print(f"semua sama: {ok}")
    save("simpan", None, {"berkas": res, "semua_sama": ok})


def cmd_jalankan(label: str, argv: list[str]) -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONIOENCODING": "utf-8"}
    t0 = time.perf_counter()
    p = subprocess.run([sys.executable, "-m", "rotoscope", *argv], cwd=ROOT, env=env, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    dt = round(time.perf_counter() - t0, 1)
    log = LOGS / f"{label}.log"
    log.write_text(f"$ python -m rotoscope {' '.join(argv)}  (CUDA_VISIBLE_DEVICES='')\n--- stdout ---\n{p.stdout}\n--- stderr ---\n"
                   f"{p.stderr}\n--- rc {p.returncode}, {dt} s ---\n", encoding="utf-8")
    lines = (p.stdout + p.stderr).splitlines()
    print("\n".join(lines[-30:]))
    print(f"--- {label}: rc {p.returncode}, {dt} s, log {log.relative_to(ROOT).as_posix()} ({len(lines)} baris)")
    save("jalankan", label, {"argv": argv, "rc": p.returncode, "detik": dt})


def manifest_meta_sha(clip_dir: Path) -> dict[str, str | None]:
    out = {}
    for m in sorted(clip_dir.glob("*/manifest.json")):
        out[m.parent.name] = (jload(m).get("clip") or {}).get("meta_sha256")
    return out


def cmd_cek(clip: str) -> None:
    real, t3c = CLIPS / clip, T3 / "work" / "clips" / clip
    meta_real, meta_t3 = (real / "meta.json").read_bytes(), (t3c / "meta.json").read_bytes()
    m_sha = hashlib.sha256(meta_real).hexdigest()
    t3_manifest = manifest_meta_sha(t3c)
    t3_export = (jload(T3 / "out" / f"{clip}.export.json").get("clip") or {}).get("meta_sha256")
    fr, ft = tree(real / "frames"), tree(t3c / "frames")
    cr, ct = tree(CROP / clip), tree(T3 / "work" / "crop" / clip)
    meta = json.loads(meta_real)
    cfg = jload(RES).get("config", {}).get("sama") if RES.is_file() else None
    r = {"meta_sama": meta_real == meta_t3, "meta_sha256": m_sha,
         "manifest_t3_meta_sha256_cocok": {k: v == m_sha for k, v in t3_manifest.items()}, "export_t3_cocok": t3_export == m_sha,
         "frames": {"asli": len(fr), "t3": len(ft), "sama": fr == ft}, "kerja": [meta["working_width"], meta["working_height"]],
         "key_crop": meta.get("crop", {}).get("box_source"), "crop_dir": {"asli": sorted(cr), "sama": cr == ct,
                                                                        "beda": sorted(k for k in set(cr) | set(ct) if cr.get(k) != ct.get(k))},
         "config_sama": cfg, "isolate_target_ada": (real / "isolate").exists() or (real / "target").exists()}
    r["lolos"] = (r["meta_sama"] and all(r["manifest_t3_meta_sha256_cocok"].values()) and r["export_t3_cocok"] and r["frames"]["sama"]
                  and r["crop_dir"]["sama"] and cfg is True and not r["isolate_target_ada"] and r["key_crop"] is not None)
    print(json.dumps(r, indent=1, ensure_ascii=False))
    save("cek", clip, r)


def cmd_ganti(clip: str) -> None:
    c = (jload(RES).get("cek") or {}).get(clip) if RES.is_file() else None
    if not c or not c.get("lolos"):
        raise SystemExit(f"{clip}: `cek` belum lolos — tidak ada yang dipindah")
    real, t3c = CLIPS / clip, T3 / "work" / "clips" / clip
    res = {}
    for item in ITEMS:
        src, new, old = t3c / item, real / (item + NEW_SUFFIX), real / item
        content = (lambda p: tree(p)) if src.is_dir() else (lambda p: sha(p))   # berkas tunggal: isi saja (nama .t607baru beda)
        if old.exists() and content(old) == content(src):
            new.unlink(missing_ok=True) if not src.is_dir() else None
            res[item] = {"sudah_sama": True}
            print(f"{clip}/{item}: sudah sama dengan t3 — dilewati")
            continue
        if new.exists():
            shutil.rmtree(new) if new.is_dir() else new.unlink()
        (shutil.copytree if src.is_dir() else shutil.copy2)(src, new)          # 1. salin
        a, b = content(src), content(new)
        if a != b:                                                              # 2. cek sha256
            raise SystemExit(f"{clip}/{item}: salinan berbeda dari t3 — berhenti, {item} asli belum diganti")
        n_old = len(tree(old)) if old.exists() else 0
        if old.exists():                                                        # 3. ganti
            shutil.rmtree(old) if old.is_dir() else old.unlink()
        new.rename(old)
        n_new = len(a) if src.is_dir() else 1
        res[item] = {"berkas_baru": n_new, "berkas_lama_dihapus": n_old, "sha256_sama": content(old) == a}
        print(f"{clip}/{item}: {n_old} berkas lama -> {n_new} berkas t3, sama {res[item]['sha256_sama']}")
    res["semua_sama_t3"] = all(content_eq for content_eq in
                               ((tree(real / i) == tree(t3c / i)) if (t3c / i).is_dir() else sha(real / i) == sha(t3c / i) for i in ITEMS))
    print(f"{clip}: semua item = t3: {res['semua_sama_t3']}")
    save("ganti", clip, res)


SVG_MARKER = ".rotoscope-clip.json"


def compare_out(name: str, a_root: Path, b_root: Path) -> dict:
    ma, mb = a_root / f"{name}.mp4", b_root / f"{name}.mp4"
    da, db = a_root / "svg" / name, b_root / "svg" / name
    sa, sb = tree(da), tree(db)
    beda = sorted(k for k in set(sa) | set(sb) if sa.get(k) != sb.get(k))
    marker = (diff_keys(jload(da / SVG_MARKER), jload(db / SVG_MARKER)) if SVG_MARKER in beda and SVG_MARKER in sa and SVG_MARKER in sb
              else [])
    ea, eb = strip_time(jload(a_root / f"{name}.export.json")), strip_time(jload(b_root / f"{name}.export.json"))
    return {"mp4_sama": sha(ma) == sha(mb), "mp4_bytes": [ma.stat().st_size, mb.stat().st_size],
            "svg": {"berkas": [len(sa), len(sb)], "svg_sama": all(k == SVG_MARKER for k in beda), "beda": beda,
                    "penanda_beda_kunci": marker, "penanda_hanya_waktu": all(any(t in k.lower() for t in TIME_KEYS) for k in marker)},
            "export_json_beda_selain_waktu": diff_keys(ea, eb)}


def cmd_banding(clip: str) -> None:
    r = compare_out(clip, OUTD, T3 / "out")
    print(json.dumps(r, indent=1, ensure_ascii=False))
    save("banding", clip, r)


def cmd_ekspor4() -> None:
    res = {}
    for clip in UNCHANGED:
        video = Path(jload(CLIPS / clip / "meta.json")["source_path"])
        cmd_jalankan(f"ekspor4_{clip}", ["export", str(video), "--config", str(T4_CFG)])
        res[clip] = compare_out(clip, OUTD, T4_OUT)
        print(f"{clip}: {res[clip]}")
    save("ekspor4", None, res)


DATA_OUT = ROOT / "scripts" / "t607_data.json"
PENILAIAN_K6 = DATA / "nilai" / "PENILAIAN_K6.md"
SUITE = {"lolos": 1656, "skip": 4, "gagal": 0, "berkas": 35, "catatan": "per berkas, berurutan, sesudah usulan A + B (Tahap 4)"}


def pick_keys(d: dict, keys) -> dict:
    return {k: d[k] for k in keys if k in d}


def read_penilaian(path: Path) -> dict:
    """Bagian `## <klip>` + baris `kunci: nilai` sesudah garis `---` (editor Rio menulis `\\_` untuk `_` dan `\\---` untuk `---`)."""
    text = path.read_text(encoding="utf-8").replace("\\_", "_").replace("\n\\---", "\n---")
    out, sec = {}, None
    for line in text.split("\n---", 1)[1].splitlines():
        if line.startswith("## "):
            sec = line[3:].strip()
            out[sec] = {}
        elif sec and ":" in line and not line.startswith(" "):
            k, v = line.split(":", 1)
            out[sec][k.strip()] = v.strip()
    return out


def cmd_data() -> None:
    w = {f: jload(DATA / f"{f}.json") for f in ("ukur", "cropdetect_lama", "probe_lama", "probe_baru", "probe_k2", "baru", "baru_k2",
                                                 "trimframe", "tahap3", "tahap3_cek", "k6_angka", "k6_video", "mutations", "tahap4")}
    clip_keys = ("source_sha256", "n_frames", "frame_wh", "label", "full", "r2_cropped", "r1_equals_r2_full", "K1", "K2_angka",
                 "K3_fp_vs_r2_full", "K3_fp_vs_frame")
    muts = w["mutations"]
    doc = {
        "format": "t607-data/1", "task": "T-607 potong letterbox / pillarbox / durasi, stage [1] ingest (D-013)",
        "catatan": "Ringkasan angka work/t607/*.json (F4, 2026-10-23) sebelum work/t607_scratch/ dihapus; rincian per jendela / berkas "
                   "tetap di work/t607/ (tidak di-commit). Keputusan + tafsir: docs/04 entri T-607.",
        "r2": w["baru"]["r2"], "window_k3": w["baru"]["window"], "cropdetect_params": w["baru"]["cropdetect_params"],
        "ffmpeg": w["ukur"]["ffmpeg"],
        "lama_tahap1a": {c: pick_keys(v, ("n_clip", "ingest_s", "detect_png_s", "detect_pipe_s", "pipe_frames", "pipe_stats_equal_png",
                                          "box_pipe", "box_equals_r2_lama", "meta_bytes_equal", "first_pts_after_fps"))
                         for c, v in w["ukur"]["clips"].items()},
        "cropdetect_lama": {c: pick_keys(v, ("r2_xyxy_incl_work", "diff_cd_minus_r2_work_px", "same_as_r2"))
                            for c, v in w["cropdetect_lama"]["clips"].items()},
        "probe": {"lama": w["probe_lama"], "baru": w["probe_baru"], "k2": w["probe_k2"]},
        "baru_tahap1b": {"labels_sha256": w["baru"]["labels_sha256"],
                         "clips": {c: pick_keys(v, clip_keys) for c, v in w["baru"]["clips"].items()}},
        "k2_klip10_11": {"labels_sha256": w["baru_k2"]["labels_sha256"],
                         "clips": {c: pick_keys(v, clip_keys) for c, v in w["baru_k2"]["clips"].items()}},
        "trimframe": w["trimframe"],
        "tahap3a": {c: pick_keys(v, ("rc", "s", "frame_count", "working_wh", "meta_keys_extra", "deteksi", "saran", "kotak"))
                    for c, v in w["tahap3"]["clips"].items()},
        "tahap3a_cek": {c: pick_keys(v, ("kotak_box_source", "kotak_mode", "working_wh", "frame_count", "meta_crop",
                                         "frames_vs_studi_psrc")) for c, v in w["tahap3_cek"]["clips"].items()},
        "k6_angka": w["k6_angka"],
        "k6_video": {n: pick_keys(v, ("hasil", "bytes")) for n, v in w["k6_video"]["videos"].items()},
        "penilaian_k6": {"sha256": sha(PENILAIAN_K6), "jawaban": read_penilaian(PENILAIAN_K6), "hasil": "K6 lolos (Klip2, klip5); "
                         "K2' klip11 lolos"},
        "tahap4": w["tahap4"],
        "mutasi": {"n": len(muts) - 1, "terbunuh": sum(v["rc"] == 1 for k, v in muts.items() if not k.startswith("kontrol")),
                   "kontrol": next((v["ringkas"] for k, v in muts.items() if k.startswith("kontrol")), None),
                   "daftar": {k: v["rc"] for k, v in muts.items()}},
        "suite": SUITE,
    }
    DATA_OUT.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{DATA_OUT.relative_to(ROOT).as_posix()}: {DATA_OUT.stat().st_size} byte, sha256 {sha(DATA_OUT)[:12]}…")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "jalankan":
        cmd_jalankan(args[0], args[1:])
    elif cmd in ("cek", "ganti", "banding"):
        {"cek": cmd_cek, "ganti": cmd_ganti, "banding": cmd_banding}[cmd](args[0])
    else:
        {"config": cmd_config, "simpan": cmd_simpan, "ekspor4": cmd_ekspor4, "data": cmd_data}[cmd]()
