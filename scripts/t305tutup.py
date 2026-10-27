"""T-305 induk · penutupan, Tahap 1 (docs/04 "Keputusan penutupan T-305 induk (2026-10-27)").

    python scripts/t305tutup.py hash before|after1|after_final  # F1: sha256 + ukuran 6 klip (work/clips/<klip>/) + out/ -> work/t305tutup/f1_<label>.json
    python scripts/t305tutup.py cek                              # BACA-SAJA: out/<klip>.mp4 + export manifest vs work/clips/ (basi?) -> work/t305tutup/cek.json
    python scripts/t305tutup.py metrics                          # metrik deskriptif per klip + per tipe strok -> work/t305tutup/metrics.json
    python scripts/t305tutup.py bahan                            # work/t305tutup/nilai/<klip>_{hasil,berdampingan,lambat_0.5x}.mp4 + cek ffprobe

Hanya MEMBACA work/clips/* dan out/ (kecuali menulis work/t305tutup/). Tanpa GPU, tanpa run / batch.
Memakai ulang: metrik + bahan dari `scripts/t305_newclips.py` (import; NILAI diarahkan ke work/t305tutup/nilai/),
pop energy / track age dari `tests/temporal_metrics.py`, hash qc_report tanpa created_utc seperti `scripts/t603_hash.py`.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import temporal_metrics as tm  # noqa: E402
import t305_newclips as t305n  # noqa: E402

from rotoscope import export as ex  # noqa: E402
from rotoscope import stabilize as stb  # noqa: E402
from rotoscope.config import load_pipeline  # noqa: E402
from rotoscope.stage_common import CLIP_KEY, StageError, clip_identity_from_bytes  # noqa: E402

CLIPS_DIR = ROOT / "work" / "clips"
OUT = ROOT / "work" / "t305tutup"
NILAI = OUT / "nilai"
F1_CLIPS = ("test", "test_short", "Klip2", "klip3", "klip4", "klip5")   # produksi: 6 klip (F1)
LINGKUP = ("test", "Klip2", "klip3", "klip4")                             # penilaian Rio
PEMBANDING = ("klip5",)                                                    # di luar lingkup; hanya pembanding
FPS = 24
CREATED_RE = re.compile(rb'("created_utc"\s*:\s*)"[^"]*"')
CREATED_MARK = rb'\1"<created_utc>"'
QC_NAME = "qc_report.json"
LABELS = ("before", "after1", "after_final")
t305n.NILAI = NILAI                                                       # bahan ke folder baru (fungsi dipakai ulang)


def jload(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def jsave(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"  -> {p.relative_to(ROOT).as_posix()}")


def ffprobe(path: Path) -> dict:
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
                        "-show_entries", "stream=codec_name,pix_fmt,width,height,nb_read_frames,r_frame_rate",
                        "-of", "json", str(path)], capture_output=True, text=True, check=True)
    s = json.loads(r.stdout)["streams"][0]
    return {"codec": s["codec_name"], "pix_fmt": s["pix_fmt"], "width": int(s["width"]), "height": int(s["height"]),
            "frames": int(s["nb_read_frames"]), "fps": s["r_frame_rate"]}


# ── F1 ─────────────────────────────────────────────
def sha_of(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def entry(p: Path) -> dict:
    rel = p.relative_to(ROOT).as_posix()
    if p.name == QC_NAME:
        sha = hashlib.sha256(CREATED_RE.sub(CREATED_MARK, p.read_bytes())).hexdigest()
        return {"sha256": sha, "size": p.stat().st_size, "norm": "created_utc"}
    return {"sha256": sha_of(p), "size": p.stat().st_size}


def snapshot() -> dict[str, dict]:
    files: list[Path] = []
    for k in F1_CLIPS:
        d = CLIPS_DIR / k
        if not d.is_dir():
            raise SystemExit(f"klip F1 tidak ada: {d}")
        files += [p for p in sorted(d.rglob("*")) if p.is_file()]
    files += [p for p in sorted((ROOT / "out").rglob("*")) if p.is_file()]
    return {p.relative_to(ROOT).as_posix(): entry(p) for p in files}


def compare(ref: dict[str, dict], cur: dict[str, dict]) -> dict:
    missing = sorted(set(ref) - set(cur))
    extra = sorted(set(cur) - set(ref))
    changed = sorted(r for r in ref if r in cur and ref[r]["sha256"] != cur[r]["sha256"])
    return {"ref_n": len(ref), "cur_n": len(cur), "missing": missing, "extra": extra, "changed": changed,
            "identical": not (missing or extra or changed)}


def cmd_hash(label: str) -> None:
    if label not in LABELS:
        raise SystemExit(f"label salah: {label} (pilih {', '.join(LABELS)})")
    path = OUT / f"f1_{label}.json"
    if path.exists():
        raise SystemExit(f"{path.relative_to(ROOT).as_posix()} sudah ada — snapshot lama tidak ditimpa")
    cur = snapshot()
    res = {"label": label, "clips": list(F1_CLIPS), "files": cur}
    if label != "before":
        res["vs_before"] = compare(jload(OUT / "f1_before.json")["files"], cur)
        v = res["vs_before"]
        print(f"vs before: {v['ref_n']} berkas, hilang {len(v['missing'])}, tambahan {len(v['extra'])}, "
              f"berubah {len(v['changed'])} -> identik = {v['identical']}")
        for r in (v["missing"] + v["extra"] + v["changed"])[:20]:
            print(f"  ! {r}")
    jsave(path, res)
    print(f"{label}: {len(cur)} berkas")


# ── cek keluaran (baca-saja) ──────────────────────────
def cmd_cek() -> None:
    cfg = load_pipeline()
    h = ex.export_hash(cfg)
    rows = {}
    for k in LINGKUP + PEMBANDING:
        work = CLIPS_DIR / k
        identity = clip_identity_from_bytes((work / "meta.json").read_bytes())
        clip = stb.load_clip(work)
        n = len(clip.names)
        mp4 = ROOT / "out" / f"{k}.mp4"
        m = ex.read_manifest(ex.manifest_path_for(mp4)) or {}
        pr = ffprobe(mp4)
        try:
            ref = ex.load_strokes_ref(work, clip, identity)
            strokes_ok, strokes_msg = m.get("strokes") == ref, "sama dengan strokes/manifest.json"
        except StageError as e:
            strokes_ok, strokes_msg = False, str(e)[:160]
        checks = {"frames_mp4_eq_frames": pr["frames"] == n, "manifest_frame_count_eq": m.get("frame_count") == n,
                  "manifest_clip_identity_eq": m.get(CLIP_KEY) == identity, "manifest_export_hash_eq": m.get("export_hash") == h,
                  "manifest_strokes_eq": strokes_ok}
        rows[k] = {"frames_dir": n, "mp4": pr, "style": m.get("style"), "checks": checks, "strokes_msg": strokes_msg,
                   "fresh": all(checks.values())}
        print(f"{k}: frames/ {n}, mp4 {pr['frames']} f {pr['width']}x{pr['height']} {pr['codec']} {pr['fps']}, "
              f"style {m.get('style')}, {'SEGAR' if all(checks.values()) else 'BASI: ' + ', '.join(c for c, v in checks.items() if not v)}")
        if not strokes_ok:
            print(f"  strokes: {strokes_msg}")
    jsave(OUT / "cek.json", rows)


# ── metrik deskriptif ─────────────────────────────────
def per_second_series(frames: list[dict]) -> dict:
    """Kelahiran + kematian (tipe, track_id) per detik (frame t dihitung ke detik floor(t / FPS)); per tipe + semua tipe.
    Definisi kelahiran / kematian sama dengan `tm.pop_energy_by_type` (pembanding frame t vs t − 1)."""
    n = len(frames)
    secs = (n + FPS - 1) // FPS
    born = {ty: np.zeros(secs, int) for ty in tm.STROKE_TYPES}
    died = {ty: np.zeros(secs, int) for ty in tm.STROKE_TYPES}
    for t in range(1, n):
        s = t // FPS
        for ty in tm.STROKE_TYPES:
            cur = {k for k in frames[t] if k[0] == ty}
            prv = {k for k in frames[t - 1] if k[0] == ty}
            born[ty][s] += len(cur - prv)
            died[ty][s] += len(prv - cur)
    tot_b = sum(born.values())
    tot_d = sum(died.values())
    out = {}
    for name, b, d in [(ty, born[ty], died[ty]) for ty in tm.STROKE_TYPES] + [("semua", tot_b, tot_d)]:
        out[name] = {"born_per_s": b.tolist(), "died_per_s": d.tolist(),
                     "born_mean": round(float(b.mean()), 3), "died_mean": round(float(d.mean()), 3),
                     "born_p95": float(np.percentile(b, 95)), "died_p95": float(np.percentile(d, 95))}
    return out


def clip_metrics(name: str, cfg) -> dict:
    work = CLIPS_DIR / name
    meta = jload(work / "meta.json")
    contours = work / "contours"
    frames = tm.stroke_lengths_from_dir(contours)
    pop = tm.pop_energy_by_type(frames)
    all_ages = [a for ty in tm.STROKE_TYPES for a in tm.track_ages(frames, ty)]
    return {"meta": {k: meta.get(k) for k in ("source_fps", "source_duration_s", "source_width", "source_height", "frame_count",
                                                "working_width", "working_height", "has_audio")},
            "qc": t305n.qc_metrics(work), "fg": t305n.fg_metrics(work), "people": t305n.face_metrics(name, cfg),
            "cut": t305n.cut_metrics(work, cfg), "contours": t305n.contour_metrics(work),
            "pop": pop, "track_age_median_semua": float(np.median(all_ages)) if all_ages else None,
            "per_second": per_second_series(frames)}


def cmd_metrics() -> None:
    cfg = load_pipeline()
    res = {}
    for name in LINGKUP + PEMBANDING:
        res[name] = clip_metrics(name, cfg)
        print(f"{name}: selesai", flush=True)
    jsave(OUT / "metrics.json", res)
    for name, r in res.items():
        p, c, q = r["pop"], r["contours"], r["qc"]
        print(f"| {name} | {q['n_fail']} | {r['cut']['cut_diff']} | {len(r['cut']['scores_at_cuts'])} | "
              f"{c['strokes_per_frame']} | {r['track_age_median_semua']} | "
              f"{ {ty: round(p['types'][ty]['pop_share_pct'] or 0, 1) for ty in tm.STROKE_TYPES} } |")


# ── bahan penilaian ───────────────────────────────────
def cmd_bahan() -> None:
    NILAI.mkdir(parents=True, exist_ok=True)
    for k in LINGKUP:
        t305n.cmd_media(k)
    for k in LINGKUP:
        n = len(stb.load_clip(CLIPS_DIR / k).names)
        got = {f: ffprobe(NILAI / f) for f in (f"{k}_hasil.mp4", f"{k}_berdampingan.mp4", f"{k}_lambat_0.5x.mp4")}
        ok = (got[f"{k}_hasil.mp4"]["frames"] == n and got[f"{k}_berdampingan.mp4"]["frames"] == n
              and got[f"{k}_lambat_0.5x.mp4"]["frames"] == 2 * n)
        print(f"{k}: frames/ {n} -> " + ", ".join(f"{f} {v['frames']} f {v['width']}x{v['height']}" for f, v in got.items())
              + f" | cek = {ok}")


# ── Tahap 4: data penutupan (baca-saja + tulis scripts/t305tutup_data.json) ──
DATA_OUT = ROOT / "scripts" / "t305tutup_data.json"
DOCS05 = ROOT / "docs" / "05-TASK-BOARD.md"
PRE_CROP_METRICS = ROOT / "work" / "t305" / "metrics.json"       # "Hasil T-305 induk Tahap 1" (sebelum crop T-607)
LABEL_RIO = (92, 104)                                              # "f92–104 = cut sungguhan" (label Rio, 2026-10-15)
PAPAN_INDUK_EXCLUDE = ("T-404",)                                   # induk, dipecah a / b (preseden: tidak dihitung)


def cut5() -> dict:
    """Cut klip5 sesudah crop (stabilize.cut_scores + detect_cuts, sama dengan cut_metrics) vs sebelum crop (work/t305/metrics.json)."""
    cfg = load_pipeline()
    thr = cfg.stabilize.temporal.cut_diff
    work = CLIPS_DIR / "klip5"
    clip = stb.load_clip(work)
    scores = stb.cut_scores([stb.frame_thumb(work / "frames" / n) for n in clip.names])
    cut = stb.detect_cuts(scores, thr)
    now = {k: round(float(s), 4) for k, s in enumerate(scores) if cut[k]}
    non_cut = [float(s) for k, s in enumerate(scores) if not cut[k]]
    prev_all = jload(PRE_CROP_METRICS)["klip5"]
    prev = {int(k): float(v) for k, v in prev_all["cut"]["scores_at_cuts"].items()}
    lo, hi = LABEL_RIO
    inside = {k: v for k, v in now.items() if lo <= k <= hi}
    outside = {k: v for k, v in now.items() if not lo <= k <= hi}
    meta_now = jload(work / "meta.json")
    same_frames = meta_now.get("frame_count") == prev_all["meta"].get("frame_count")
    return {"cut_diff": thr,
            "trim_sekarang": "trim" in meta_now, "crop_sekarang": "crop" in meta_now,
            "frame_count_sebelum": prev_all["meta"].get("frame_count"), "frame_count_sesudah": meta_now.get("frame_count"),
            "pemetaan_indeks": "identik (tanpa trim, jumlah frame sama)" if same_frames and "trim" not in meta_now
            else "BERGESER: periksa manual",
            "sebelum_crop": {"n": len(prev), "cuts": {k: round(v, 4) for k, v in sorted(prev.items())},
                             "skor_min": round(min(prev.values()), 4), "skor_max": round(max(prev.values()), 4)},
            "sesudah_crop": {"n": len(now), "cuts": now, "max_non_cut": round(max(non_cut), 4),
                             "skor_min": round(min(now.values()), 4), "skor_max": round(max(now.values()), 4)},
            "label_rio": {"rentang": [lo, hi], "terdeteksi": sorted(inside), "n": len(inside),
                          "skor_min": round(min(inside.values()), 4), "skor_max": round(max(inside.values()), 4)},
            "belum_berlabel": {"n": len(outside), "cuts": outside,
                               "skor_min": round(min(outside.values()), 4) if outside else None,
                               "skor_max": round(max(outside.values()), 4) if outside else None},
            "hanya_sesudah_crop": sorted(set(now) - set(prev)), "hanya_sebelum_crop": sorted(set(prev) - set(now))}


def papan() -> dict:
    """Hitungan papan dari docs/05: judul `### T-…` per `# PHASE n`, status di akhir judul. T-404 induk tidak dihitung (dipecah a / b)."""
    phase, rows = None, []
    for line in DOCS05.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^# PHASE (\d+)", line)
        if m:
            phase = int(m.group(1))
            continue
        m = re.match(r"^### (T-\S+) · .* · `(TODO|WIP|DONE|SKIP|BLOCKED)`", line)
        if m and phase is not None and m.group(1) not in PAPAN_INDUK_EXCLUDE:
            rows.append((phase, m.group(1), m.group(2)))
    per = {}
    for ph in sorted({p for p, _, _ in rows}):
        st = [s for p, _, s in rows if p == ph]
        per[f"Phase {ph}"] = {"done": st.count("DONE"), "tasks": len(st)}
    return {"per_phase": per, "done": sum(1 for _, _, s in rows if s == "DONE"), "total": len(rows),
            "status": {t: s for _, t, s in rows if s != "DONE"}}


def f1_summary() -> dict:
    out = {}
    for label in LABELS:
        p = OUT / f"f1_{label}.json"
        if p.is_file():
            d = jload(p)
            out[label] = {"berkas": len(d["files"]), **({"vs_before": {k: v for k, v in d["vs_before"].items()
                                                                  if k not in ("missing", "extra", "changed")},
                                                       "hilang": d["vs_before"]["missing"], "tambahan": d["vs_before"]["extra"],
                                                       "berubah": d["vs_before"]["changed"]} if "vs_before" in d else {})}
    return out


def cmd_data() -> None:
    cek = jload(OUT / "cek.json")
    metrics = jload(OUT / "metrics.json")
    res = {"tanggal": "2026-10-27", "sumber": ["work/t305tutup/metrics.json", "work/t305tutup/cek.json", "work/t305tutup/f1_*.json",
                                              "work/t305/metrics.json (sebelum crop)", "docs/05 (hitungan papan)"],
           "f1": f1_summary(),
           "cek": {k: {"segar": v["fresh"], "frames_dir": v["frames_dir"], "style": v["style"], "checks": v["checks"]}
                   for k, v in cek.items()},
           "cut_klip5": cut5(),
           "cut_per_klip": {k: {"n": len(v["cut"]["scores_at_cuts"]), "cuts": v["cut"]["scores_at_cuts"],
                                "max_non_cut": v["cut"]["max_non_cut"], "cut_diff": v["cut"]["cut_diff"]}
                            for k, v in metrics.items()},
           "metrik": metrics,
           "papan": papan()}
    jsave(DATA_OUT, res)


def main(argv: list[str]) -> None:
    if argv[:1] == ["hash"] and len(argv) == 2:
        cmd_hash(argv[1])
    elif argv == ["cek"]:
        cmd_cek()
    elif argv == ["metrics"]:
        cmd_metrics()
    elif argv == ["bahan"]:
        cmd_bahan()
    elif argv == ["data"]:
        cmd_data()
    elif argv == ["papan"]:
        print(json.dumps(papan(), ensure_ascii=False, indent=1))
    elif argv == ["cut5"]:
        print(json.dumps(cut5(), ensure_ascii=False, indent=1))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
