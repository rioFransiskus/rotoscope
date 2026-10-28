"""T-608 Tahap 4b bagian 1 (alat, F3; CPU): verifikasi run produksi klip5@0-2.25, banding per stage vs scratch iso2, metrik, bahan buta.
docs/04 "Hasil run produksi klip5@0-2.25 (2026-10-28)".

    python scripts/t608_tahap4b.py verifikasi  # B1: mp4 (ffprobe) + SVG, masker [3], produksi lama vs f1_sesudah_4a2 -> t4b_verifikasi.json
    python scripts/t608_tahap4b.py banding     # B2: per stage f0–53 produksi trim vs scratch iso (sha256 + sidik isi) + bukti
                                               #     penyebab (manifest / jendela [3] / QC / uji kontrafaktual ambang [4]) -> t4b_banding.json
    python scripts/t608_tahap4b.py metrik      # B3: metrik t608_1b2 (definisi sama) SEBELUM / SESUDAH / BARU -> t4b_metrik.json
    python scripts/t608_tahap4b.py bahan       # B4: work/t608/nilai_4b/ (X / Y acak; kunci + seed -> kunci.json, TIDAK dicetak)
    python scripts/t608_tahap4b.py cek         # B4: ffprobe + sha256 bahan -> nilai_4b/bahan.json
    python scripts/t608_tahap4b.py f1          # B6: f1_sesudah_4b1 vs f1_sesudah_4a2, perubahan dikelompokkan -> t4b_f1.json
    python scripts/t608_tahap4b.py data        # 4b (h): data berharga T-608 -> work/t608/t608_data.json (salinan ter-track
                                               #     scripts/t608_data.json dibuat dengan Write, BUKAN skrip)
    python scripts/t608_tahap4b.py cek_data    # 4b (h): bangun ulang di memori; byte = work/t608/t608_data.json = scripts/t608_data.json?

SEBELUM = produksi klip5 (133 frame), SESUDAH = scratch iso2 (versi yang dinilai Rio), BARU = produksi `klip5@0-2.25` (run GPU Rio).
Hanya MEMBACA work/clips/*, work/t608_scratch/iso/, out/; menulis hanya ke work/t608/. Tanpa GPU (CUDA_VISIBLE_DEVICES=-1; torch tidak
diimpor). Frame / video dibaca sebagai angka (cv2), tidak ditampilkan.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import secrets
import subprocess
import sys
from collections import Counter
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"                 # pengaman: tanpa GPU

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import t608_1b2 as m  # noqa: E402
from rotoscope import vectorize as vec  # noqa: E402
from rotoscope.config import load_pipeline  # noqa: E402

OUT = m.OUT
NILAI = OUT / "nilai_4b"
NEW = ROOT / "work" / "clips" / "klip5@0-2.25"
SCR = m.SCR
PROD = m.PROD
VERS = {"sebelum": PROD, "sesudah": SCR, "baru": NEW}
MP4_NEW = ROOT / "out" / "klip5@0-2.25.mp4"
SVG_NEW = ROOT / "out" / "svg" / "klip5@0-2.25"
MP4_SCR = m.MP4["SESUDAH"]
N = 54
LAST = N - 1
SHOTS = ((0, 21, "A"), (22, 53, "B"))                     # adegan dalam lingkup f0–53 (cut f22)
F1_REF = OUT / "f1_sesudah_4a2.txt"
F1_NEW = OUT / "f1_sesudah_4b1.txt"
OLD_PROD = ("out/klip5.mp4", "out/klip5.export.json", "out/svg/klip5/", "work/clips/klip5/")
RUN_WRITES = ("work/clips/klip5@0-2.25/", "out/klip5@0-2.25.mp4", "out/svg/klip5@0-2.25/", "out/klip5@0-2.25.export.json")
STROKE_TYPES = ("silhouette", "silhouette_hole", "group_boundary", "occlusion")
TYPE_ID = {"group_boundary": "batas grup", "occlusion": "oklusi", "silhouette": "siluet", "silhouette_hole": "lubang"}
SVG_GROUP_RE = re.compile(r'<g id="(?:pass_(\d+)_)?(' + "|".join(STROKE_TYPES) + r')"[^>]*>(.*?)</g>', re.S)
PANEL_H, GUTTER = 880, 16                                  # 1080 × 1002 → 948 × 880; 948 + 16 + 948 = 1912 ≤ 1920
FOCUS_MAX = 6
FOCUS_PCT = 75                                             # frame dengan beda ≥ persentil ini = kandidat fokus


def rel(p: Path) -> str:
    return p.relative_to(ROOT).as_posix()


def name(i: int, suffix: str) -> str:
    return f"frame_{i:05d}{suffix}"


def ranges(xs) -> str:
    xs = sorted(xs)
    if not xs:
        return "-"
    out, a, b = [], xs[0], xs[0]
    for x in xs[1:]:
        if x == b + 1:
            b = x
            continue
        out.append(f"f{a}" if a == b else f"f{a}–{b}")
        a = b = x
    out.append(f"f{a}" if a == b else f"f{a}–{b}")
    return ", ".join(out)


def arr_sha(a: np.ndarray) -> str:
    a = np.ascontiguousarray(a)
    return hashlib.sha256(f"{a.dtype.str}|{a.shape}|".encode() + a.tobytes()).hexdigest()


def content_sha(p: Path) -> str:
    """Sidik isi: array hasil decode (PNG / npy / npz per key), bukan byte berkas."""
    if p.suffix == ".png":
        return arr_sha(m.read_img(p, cv2.IMREAD_UNCHANGED))
    if p.suffix == ".npy":
        return arr_sha(np.load(p, allow_pickle=False))
    if p.suffix == ".npz":
        with np.load(p, allow_pickle=False) as z:
            return hashlib.sha256("".join(f"{k}:{arr_sha(z[k])};" for k in sorted(z.files)).encode()).hexdigest()
    raise ValueError(p)


def jsonl_last(p: Path) -> dict[int, dict]:
    rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    return {r["index"]: r for r in rows if r.get("event") == "frame"}


def ffprobe(p: Path) -> dict:
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
           "stream=codec_name,pix_fmt,width,height,avg_frame_rate,nb_read_frames:format=duration,size", "-of", "json", str(p)]
    d = json.loads(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout)
    return {**d["streams"][0], **d["format"]}


def f1_list(p: Path) -> dict[str, str]:
    return {r: h for h, r in (ln.split("  ", 1) for ln in p.read_text(encoding="utf-8").splitlines() if ln)}


# ── B1 ─────────────────────────────────────────────
def cmd_verifikasi() -> None:
    res: dict = {"mp4": {"berkas": rel(MP4_NEW), "sha256": m.file_sha(MP4_NEW), **ffprobe(MP4_NEW)}}
    svgs = sorted(p.name for p in SVG_NEW.glob("*.svg"))
    res["svg"] = {"folder": rel(SVG_NEW), "n": len(svgs), "nama_sesuai_f0_53": svgs == [name(i, ".svg") for i in range(N)],
                  "berkas_lain": sorted(p.name for p in SVG_NEW.iterdir() if p.suffix != ".svg")}
    st = json.loads((NEW / "stable" / "manifest.json").read_text(encoding="utf-8"))
    blk = st.get("isolate")
    iso_m = json.loads((NEW / "isolate" / "manifest.json").read_text(encoding="utf-8"))
    iso_rows = jsonl_last(NEW / "isolate" / "frames.jsonl")
    st_rows = jsonl_last(NEW / "stable" / "frames.jsonl")
    res["masker_3"] = {"manifest_key_isolate_ada": blk is not None,
                       "frames": blk and blk["frames"], "shots": blk and blk["shots"],
                       "key_sha256_sama_isolate_manifest": bool(blk) and blk["key_sha256"] == iso_m.get("key_sha256"),
                       "target_mask": blk and blk["target_mask"],
                       "isolate_frames_jsonl_status": dict(Counter(iso_rows[i]["status"] for i in range(N) if i in iso_rows)),
                       "stable_frames_jsonl_n": len(st_rows),
                       "stable_frame_dengan_key_isolate": sum("isolate" in st_rows.get(i, {}) for i in range(N)),
                       "contoh_f0": st_rows.get(0, {}).get("isolate")}
    ref = f1_list(F1_REF)
    old = {r: h for r, h in ref.items() if r.startswith(OLD_PROD)}
    now = {}
    for prefix in OLD_PROD:
        p = ROOT / prefix
        files = [p] if p.is_file() else sorted(x for x in p.rglob("*") if x.is_file())
        now.update({rel(x): m.file_sha(x) for x in files})
    res["produksi_lama"] = {"pembanding": rel(F1_REF), "n_ref": len(old), "n_sekarang": len(now),
                            "berubah": sorted(r for r in old if r in now and now[r] != old[r]),
                            "hilang": sorted(set(old) - set(now)), "tambahan": sorted(set(now) - set(old)),
                            "per_awalan": {x: sum(r.startswith(x) for r in old) for x in OLD_PROD}}
    res["produksi_lama"]["identik"] = not (res["produksi_lama"]["berubah"] or res["produksi_lama"]["hilang"]
                                           or res["produksi_lama"]["tambahan"])
    exp = m.MP4["SEBELUM"].parent / "klip5@0-2.25.export.json"
    if exp.is_file():
        e = json.loads(exp.read_text(encoding="utf-8"))
        res["export_manifest"] = {"berkas": rel(exp), "key": sorted(e)}
    mp = res["mp4"]
    print(f"mp4 {mp['berkas']}: {mp['nb_read_frames']} frame, {mp['avg_frame_rate']}, {mp['width']}x{mp['height']}, "
          f"{mp['duration']} s, {int(mp['size']) / 1024:.1f} KiB, {mp['codec_name']} {mp['pix_fmt']}; sha256 {mp['sha256'][:12]}")
    print(f"svg: {res['svg']}")
    print(f"masker [3]: {json.dumps(res['masker_3'], ensure_ascii=False)}")
    pl = res["produksi_lama"]
    print(f"produksi lama vs {pl['pembanding']}: {pl['n_ref']} berkas {pl['per_awalan']}; berubah {len(pl['berubah'])}, hilang "
          f"{len(pl['hilang'])}, tambahan {len(pl['tambahan'])} -> identik {pl['identik']}")
    res["lolos"] = (int(mp["nb_read_frames"]) == N and res["svg"]["n"] == N and res["svg"]["nama_sesuai_f0_53"]
                    and bool(blk) and blk["frames"] == {"target": N, "kosong": 0, "tanpa_prompt": 0}
                    and res["masker_3"]["stable_frame_dengan_key_isolate"] == N and pl["identik"])
    print(f"B1 lolos: {res['lolos']}")
    m.jsave(OUT / "t4b_verifikasi.json", res)


# ── B2 ─────────────────────────────────────────────
STAGE_FILES = (   # (stage, sub-berkas, pola relatif folder klip)
    ("isolate", "labels", "isolate/labels/{}", ".png"),
    ("seg", "classmap", "seg/classmap/{}", ".png"),
    ("seg", "probs", "seg/probs/{}", ".npz"),
    ("depth", "depth", "depth/{}", ".npy"),
    ("stable", "groups", "stable/groups/{}", ".png"),
    ("stable", "depth_smooth", "stable/depth_smooth/{}", ".npy"),
    ("strokes", "png", "strokes/{}", ".png"),
)


def stroke_key(s: dict) -> tuple:
    """Strok tanpa track_id, urutan / arah titik + urutan `groups` diabaikan (anchor, orientasi, arah = normalisasi T-202 saja;
    membalik garis terbuka ikut membalik pasangan grup)."""
    return (s["type"], bool(s["closed"]), tuple(sorted(s["groups"])), tuple(sorted(map(tuple, s["points"]))), s.get("strength"))


def contour_doc(base: Path, i: int) -> dict:
    return m.jload(base / "contours" / name(i, ".json"))


def by_type_keys(strokes: list[dict]) -> dict[str, Counter]:
    out = {t: Counter() for t in STROKE_TYPES}
    for s in strokes:
        out[s["type"]][stroke_key(s)] += 1
    return out


def svg_groups(p: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, t, body in SVG_GROUP_RE.findall(p.read_text(encoding="utf-8")):
        out[f"{t}@pass{k or 0}"] = body
    return out


def qc_rows(base: Path) -> dict[int, dict]:
    return {r["index"]: r for r in m.jload(base / "qc_report.json")["frames"]}


def cmd_banding() -> None:
    res: dict = {"lingkup": [0, LAST], "produksi_trim": rel(NEW), "scratch_iso2": rel(SCR), "stage": {}}
    # berkas per stage
    for stage, sub, pat, suf in STAGE_FILES:
        rows = []
        for i in range(N):
            a, b = NEW / pat.format(name(i, suf)), SCR / pat.format(name(i, suf))
            same_b = m.file_sha(a) == m.file_sha(b)
            rows.append({"i": i, "bytes": same_b, "isi": same_b or content_sha(a) == content_sha(b)})
        res["stage"][f"{stage}/{sub}"] = {"bytes_sama": sum(r["bytes"] for r in rows), "isi_sama": sum(r["isi"] for r in rows),
                                          "beda_isi": [r["i"] for r in rows if not r["isi"]],
                                          "beda_bytes_saja": [r["i"] for r in rows if r["isi"] and not r["bytes"]]}
    # isolate status + skor
    ia, ib = jsonl_last(NEW / "isolate" / "frames.jsonl"), jsonl_last(SCR / "isolate" / "frames.jsonl")
    st_same = [i for i in range(N) if ia[i]["status"] == ib[i]["status"]]
    sc_same = [i for i in range(N) if ia[i]["targets"] == ib[i]["targets"]]
    res["stage"]["isolate/frames.jsonl"] = {"status_sama": len(st_same), "targets_sama": len(sc_same),
                                            "beda_targets": [i for i in range(N) if i not in sc_same],
                                            "status_baru": dict(Counter(ia[i]["status"] for i in range(N)))}
    # QC per frame
    qa, qb = qc_rows(NEW), qc_rows(SCR)
    metrics = ("area_px", "area_ratio", "iou_prev", "big_blobs", "label_agreement_prev", "finite", "argmax_mismatch_pct",
               "area_median_px", "area_vs_median", "fail_reasons")
    qc_diff = {i: [k for k in metrics if qa[i].get(k) != qb[i].get(k)] for i in range(N)}
    fail_a = [i for i in range(N) if qa[i]["fail_reasons"]]
    fail_b = [i for i in range(N) if qb[i]["fail_reasons"]]
    res["stage"]["qc"] = {"status_sama": sum(bool(qa[i]["fail_reasons"]) == bool(qb[i]["fail_reasons"]) for i in range(N)),
                          "fail_reasons_sama": sum(qa[i]["fail_reasons"] == qb[i]["fail_reasons"] for i in range(N)),
                          "baris_sama_semua_metrik": sum(not v for v in qc_diff.values()),
                          "gagal_baru": {i: qa[i]["fail_reasons"] for i in fail_a},
                          "gagal_scratch": {i: qb[i]["fail_reasons"] for i in fail_b},
                          "metrik_beda": {i: v for i, v in qc_diff.items() if v},
                          "n_frame_qc": {"baru": len(qa), "scratch": len(qb)},
                          "area_median_window": m.jload(NEW / "qc_report.json")["qc"]["area_median_window"],
                          "transient_min_run": m.jload(NEW / "qc_report.json")["qc"]["transient_min_run"]}
    # bukti QC: big_blobs n per frame, run panjang di sekitar ujung klip
    res["stage"]["qc"]["big_blobs_n"] = {"baru": [qa[i]["big_blobs"] for i in range(N)],
                                         "scratch_f0_63": [qb[i]["big_blobs"] for i in range(min(64, len(qb)))]}
    # stable: jendela kernel + key isolate per frame + manifest temporal
    sa, sb = jsonl_last(NEW / "stable" / "frames.jsonl"), jsonl_last(SCR / "stable" / "frames.jsonl")
    ma = m.jload(NEW / "stable" / "manifest.json")
    mb = m.jload(SCR / "stable" / "manifest.json")
    res["stable_manifest"] = {k: {"baru": ma.get(k), "scratch": mb.get(k)} for k in ("temporal",)}
    res["stable_manifest"]["sama"] = {k: ma.get(k) == mb.get(k) for k in ("stabilize_hash", "groups_hash", "seg", "depth")}
    res["stable_manifest"]["isolate"] = {"baru": {k: v for k, v in ma["isolate"].items() if k != "target_mask"},
                                         "scratch": {k: v for k, v in mb["isolate"].items() if k != "target_mask"},
                                         "target_mask_sama": ma["isolate"]["target_mask"] == mb["isolate"]["target_mask"]}
    win = {i: (sa[i]["window"], sb[i]["window"]) for i in range(N)}
    res["stable_jendela_beda"] = {i: {"baru": a, "scratch": b} for i, (a, b) in win.items() if a != b}
    res["stable_key_isolate_beda"] = [i for i in range(N) if sa[i].get("isolate") != sb[i].get("isolate")]
    # penyebab terbukti per frame stable yang beda
    stage = res["stage"]
    seg_bad = set(stage["seg/probs"]["beda_isi"]) | set(stage["seg/classmap"]["beda_isi"])
    iso_bad = set(stage["isolate/labels"]["beda_isi"])
    dep_bad = set(stage["depth/depth"]["beda_isi"])
    qfail_bad = {i for i in range(N) if bool(qa[i]["fail_reasons"]) != bool(qb[i]["fail_reasons"])}
    stable_bad = sorted(set(stage["stable/groups"]["beda_isi"]) | set(stage["stable/depth_smooth"]["beda_isi"]))
    cause = {}
    for i in stable_bad:
        lo, hi = sb[i]["window"]
        lo2, hi2 = sa[i]["window"]
        span = set(range(min(lo, lo2), max(hi, hi2) + 1))
        c = []
        if win[i][0] != win[i][1]:
            c.append(f"jendela kernel [3] beda (baru {win[i][0]}, scratch {win[i][1]}): dipotong di tepi klip f{LAST}")
        if span & qfail_bad:
            c.append(f"status QC gagal beda di jendela ({ranges(span & qfail_bad)}): bobot q (qc_fail_weight) beda")
        for lab, bad in (("seg", seg_bad), ("isolate", iso_bad), ("depth", dep_bad)):
            if span & bad:
                c.append(f"input {lab} beda di jendela ({ranges(span & bad)})")
        cause[i] = c or ["belum terbukti"]
    res["stable_penyebab"] = cause
    gdiff = {}
    for i in stable_bad:
        ga = m.read_img(NEW / "stable" / "groups" / name(i, ".png"), cv2.IMREAD_UNCHANGED)
        gb = m.read_img(SCR / "stable" / "groups" / name(i, ".png"), cv2.IMREAD_UNCHANGED)
        gdiff[i] = {"piksel_grup_beda": int((ga != gb).sum()), "piksel_fg_beda": int(((ga > 0) != (gb > 0)).sum()),
                    "fg_baru": int((ga > 0).sum()), "fg_scratch": int((gb > 0).sum())}
    res["stable_groups_beda_piksel"] = gdiff
    # contours
    cs_a, cs_b = m.jload(NEW / "contours" / "clip_stats.json"), m.jload(SCR / "contours" / "clip_stats.json")
    res["clip_stats"] = {k: {"baru": cs_a.get(k), "scratch": cs_b.get(k)}
                         for k in ("n_frames", "n_values", "t_high", "t_low", "t_high_pct", "hi_pct", "lo_pct")}
    res["clip_stats"]["masukan_lain_sama"] = {k: cs_a.get(k) == cs_b.get(k) for k in ("blur_sigma", "erode_px", "hi_floor",
                                                                                     "sobel_norm", "percentile_method")}
    crow = []
    for i in range(N):
        da, db = contour_doc(NEW, i), contour_doc(SCR, i)
        ka, kb = by_type_keys(da["strokes"]), by_type_keys(db["strokes"])
        strip = [[{k: v for k, v in s.items() if k != "track_id"} for s in d["strokes"]] for d in (da, db)]
        crow.append({"i": i, "bytes": (NEW / "contours" / name(i, ".json")).read_bytes()
                     == (SCR / "contours" / name(i, ".json")).read_bytes(),
                     "strok_persis_tanpa_track_id": strip[0] == strip[1],
                     "isi": all(ka[t] == kb[t] for t in STROKE_TYPES), "per_tipe": {t: ka[t] == kb[t] for t in STROKE_TYPES},
                     "source_sama": da["source"] == db["source"], "n": [len(da["strokes"]), len(db["strokes"])]})
    res["stage"]["contours"] = {"bytes_sama": sum(r["bytes"] for r in crow),
                                "strok_persis_tanpa_track_id": sum(r["strok_persis_tanpa_track_id"] for r in crow),
                                "isi_sama": sum(r["isi"] for r in crow),
                                "per_tipe_sama": {t: sum(r["per_tipe"][t] for r in crow) for t in STROKE_TYPES},
                                "per_tipe_beda": {t: [r["i"] for r in crow if not r["per_tipe"][t]] for t in STROKE_TYPES},
                                "source_sama": sum(r["source_sama"] for r in crow)}
    # uji kontrafaktual [4]: input [3] PRODUKSI TRIM + ambang SCRATCH → strok = scratch? (dan + ambang trim → = produksi trim)
    cfg = load_pipeline(m.ROOT / "configs" / "default.yaml")
    params = vec.vectorize_params(cfg)
    names = tuple(g for g, _ in cfg.groups)
    clip = vec.load_clip(NEW)
    cf = []
    for i in range(N):
        gmap, depth = vec.load_frame_inputs(clip, name(i, ".png"), len(names))
        row = {"i": i}
        for lab, cs, base in (("ambang_scratch", cs_b, SCR), ("ambang_trim", cs_a, NEW)):
            raw, _ = vec.vectorize_frame(gmap, depth, names, params, cs["t_high"], cs["t_low"])
            kr = by_type_keys(raw)
            kd = by_type_keys(contour_doc(base, i)["strokes"])
            row[lab] = {t: kr[t] == kd[t] for t in STROKE_TYPES}
        cf.append(row)
    res["kontrafaktual_4"] = {
        "definisi": "peta grup + depth_smooth PRODUKSI TRIM, vectorize_frame (kode [4]) dengan T_high / T_low scratch; strok mentah "
                    "dibanding (tanpa track_id, urutan / arah titik diabaikan) dengan contours scratch. Kontrol: ambang trim vs "
                    "contours produksi trim.",
        "ambang_scratch_sama_scratch": {t: sum(r["ambang_scratch"][t] for r in cf) for t in STROKE_TYPES},
        "ambang_scratch_frame_beda": {t: [r["i"] for r in cf if not r["ambang_scratch"][t]] for t in STROKE_TYPES},
        "kontrol_ambang_trim_sama_trim": {t: sum(r["ambang_trim"][t] for r in cf) for t in STROKE_TYPES}}
    # strokes SVG per tipe + piksel beda PNG
    srow = []
    for i in range(N):
        a, b = NEW / "strokes" / name(i, ".svg"), SCR / "strokes" / name(i, ".svg")
        ga, gb = svg_groups(a), svg_groups(b)
        pa = m.read_img(NEW / "strokes" / name(i, ".png"))
        pb = m.read_img(SCR / "strokes" / name(i, ".png"))
        diff = int(np.any(pa != pb, axis=2).sum())
        srow.append({"i": i, "svg_bytes": a.read_bytes() == b.read_bytes(),
                     "svg_per_tipe": {t: all(ga.get(k) == gb.get(k) for k in set(ga) | set(gb) if k.startswith(t + "@"))
                                      for t in STROKE_TYPES},
                     "png_piksel_beda": diff})
    res["stage"]["strokes/svg"] = {"bytes_sama": sum(r["svg_bytes"] for r in srow),
                                   "per_tipe_sama": {t: sum(r["svg_per_tipe"][t] for r in srow) for t in STROKE_TYPES},
                                   "per_tipe_beda": {t: [r["i"] for r in srow if not r["svg_per_tipe"][t]] for t in STROKE_TYPES}}
    res["strokes_png_piksel_beda"] = [r["png_piksel_beda"] for r in srow]
    # penyebab contours / strokes per frame
    stable_bad_set = set(stable_bad)
    cc = {}
    for r in crow:
        if r["isi"]:
            continue
        i = r["i"]
        bad_types = [t for t in STROKE_TYPES if not r["per_tipe"][t]]
        c = []
        for t in bad_types:     # per tipe: kontrafaktual cocok (+ kontrol) = ambang; selain itu input [3] beda; selain itu belum terbukti
            if cf[i]["ambang_scratch"][t] and cf[i]["ambang_trim"][t]:
                c.append(f"{t}: ambang per klip [4] (kontrafaktual ambang scratch = scratch; kontrol ambang trim = trim)")
            elif i in stable_bad_set:
                c.append(f"{t}: input [3] beda (stable_penyebab)")
            else:
                c.append(f"{t}: belum terbukti")
        cc[i] = {"tipe_beda": bad_types, "penyebab": c}
    res["contours_penyebab"] = cc
    sc = {}
    for r in srow:
        if r["svg_bytes"]:
            continue
        i = r["i"]
        bad_t = [t for t in STROKE_TYPES if not r["svg_per_tipe"][t]]
        c = []
        if not crow[i]["isi"]:
            c.append(f"contours beda (tipe {', '.join(t for t in STROKE_TYPES if not crow[i]['per_tipe'][t])})")
        elif not crow[i]["strok_persis_tanpa_track_id"] or not crow[i]["bytes"]:
            c.append("contours sama isi, beda urutan / arah titik / track_id / source")
        sc[i] = {"tipe_svg_beda": bad_t, "penyebab": c or ["belum terbukti"]}
    res["strokes_penyebab"] = sc
    m.jsave(OUT / "t4b_banding.json", res)
    # ringkas
    for k, v in res["stage"].items():
        print(f"{k:22s} {json.dumps({x: (ranges(y) if isinstance(y, list) and y and isinstance(y[0], int) else y) for x, y in v.items() if x not in ('metrik_beda', 'big_blobs_n', 'gagal_baru', 'gagal_scratch')}, ensure_ascii=False)}")
    q = res["stage"]["qc"]
    print(f"QC gagal baru {ranges(q['gagal_baru'])} {q['gagal_baru']}; scratch {ranges(q['gagal_scratch'])} {q['gagal_scratch']}")
    print(f"QC metrik beda: {q['metrik_beda']}")
    print(f"big_blobs n baru: {q['big_blobs_n']['baru']}")
    print(f"big_blobs n scratch f0–63: {q['big_blobs_n']['scratch_f0_63']}")
    print(f"stable manifest temporal: {json.dumps(res['stable_manifest']['temporal'], ensure_ascii=False)}")
    print(f"stable manifest sama: {res['stable_manifest']['sama']}; isolate: {json.dumps(res['stable_manifest']['isolate'], ensure_ascii=False)}")
    print(f"jendela [3] beda: {res['stable_jendela_beda']}; key isolate frames.jsonl beda: {res['stable_key_isolate_beda']}")
    print(f"stable penyebab: {json.dumps(cause, ensure_ascii=False)}")
    print(f"stable groups piksel beda: {json.dumps(gdiff, ensure_ascii=False)}")
    print(f"clip_stats: {json.dumps(res['clip_stats'], ensure_ascii=False)}")
    k4 = res["kontrafaktual_4"]
    print(f"kontrafaktual [4]: ambang scratch = scratch {k4['ambang_scratch_sama_scratch']}; beda "
          f"{ {t: ranges(v) for t, v in k4['ambang_scratch_frame_beda'].items()} }; kontrol {k4['kontrol_ambang_trim_sama_trim']}")
    for lab, d, tk in (("contours", cc, "tipe_beda"), ("strokes", sc, "tipe_svg_beda")):
        grp: dict[str, list[int]] = {}
        for i, v in d.items():
            grp.setdefault(f"{v[tk]} :: {' | '.join(v['penyebab'])}", []).append(i)
        print(f"{lab} penyebab:")
        for k, v in grp.items():
            print(f"  {ranges(v)} ({len(v)}): {k}")
    print(f"strokes PNG piksel beda per frame: {res['strokes_png_piksel_beda']}")


# ── B3 ─────────────────────────────────────────────
def type_lengths(doc: dict) -> dict[str, float]:
    out = {t: 0.0 for t in STROKE_TYPES}
    for s in doc["strokes"]:
        p = np.asarray(s["points"], np.float64)
        if len(p) < 2:
            continue
        seg = np.vstack([p, p[:1]]) if s.get("closed") else p
        out[s["type"]] += float(np.hypot(*(seg[1:] - seg[:-1]).T).sum())
    return out


def cmd_metrik() -> None:
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * m.DILATE_PX + 1, 2 * m.DILATE_PX + 1))
    per = []
    for i in range(N):
        near = cv2.dilate(m.mask(i).astype(np.uint8), kern) > 0
        rec = {"index": i}
        for lab, base in VERS.items():
            doc = m.jload(base / "contours" / name(i, ".json"))
            g = m.read_img(base / "stable" / "groups" / name(i, ".png"), cv2.IMREAD_UNCHANGED)
            rec[lab] = {**m.stroke_metrics(doc, near), **m.group_components(g), "panjang_per_tipe": type_lengths(doc)}
        per.append(rec)
    keys = ("strok", "panjang_px", "porsi_panjang_luar_masker", "komponen", "komponen_kecil", "fg_porsi")

    def block(sel: list[dict]) -> dict:
        out = {}
        for lab in VERS:
            out[lab] = {k: m.stats([r[lab][k] for r in sel if r[lab][k] is not None]) for k in keys}
            out[lab]["jumlah_per_tipe_per_frame"] = {t: round(sum(r[lab]["per_tipe"].get(t, 0) for r in sel) / len(sel), 2)
                                                     for t in STROKE_TYPES}
            out[lab]["panjang_per_tipe_per_frame"] = {t: round(sum(r[lab]["panjang_per_tipe"][t] for r in sel) / len(sel), 1)
                                                      for t in STROKE_TYPES}
        return out

    res = {"definisi": "sama dengan scripts/t608_1b2.py metrik (masker = isolate/labels scratch iso, +4 px, untuk SEMUA versi); "
                       "tambahan: panjang per tipe per frame (rata-rata, px)",
           "lingkup": [0, LAST], "versi": {k: rel(v) for k, v in VERS.items()},
           "semua": block(per), "per_adegan": {}, "per_frame": per}
    for lo, hi, nm in SHOTS:
        res["per_adegan"][nm] = {"frame": [lo, hi], **block([r for r in per if lo <= r["index"] <= hi])}
    old = m.jload(OUT / "iso2_metrik.json")["banding"]
    res["cek_ulang_iso2_metrik"] = {lab: {k: old[lab][k]["median"] == res["semua"][lab][k]["median"] for k in keys}
                                    for lab in ("sebelum", "sesudah")}
    m.jsave(OUT / "t4b_metrik.json", res)
    print(f"cek ulang vs iso2_metrik.json (median sama): {res['cek_ulang_iso2_metrik']}")
    for scope, blk in [("f0–53", res["semua"])] + [(f"adegan {k}", v) for k, v in res["per_adegan"].items()]:
        print(f"== {scope}")
        for k in keys:
            print(f"  {k:28s} " + " | ".join(f"{lab} med {blk[lab][k]['median']:>10}" for lab in VERS))
        for lab in VERS:
            print(f"  jumlah/tipe/frame {lab:8s} {blk[lab]['jumlah_per_tipe_per_frame']}")
        for lab in VERS:
            print(f"  panjang/tipe/frame {lab:7s} {blk[lab]['panjang_per_tipe_per_frame']}")


# ── B4 ─────────────────────────────────────────────
def focus_ranges(diff: list[int]) -> list[dict]:
    """Rentang fokus: frame dengan piksel beda strok (PNG [5], simetris) ≥ persentil FOCUS_PCT, frame bersebelahan (selisih ≤ 1)
    digabung, diurutkan menurut jumlah beda, maks FOCUS_MAX, lalu diurutkan menurut frame."""
    thr = float(np.percentile(diff, FOCUS_PCT))
    sel = [i for i, d in enumerate(diff) if d >= thr and d > 0]
    groups: list[list[int]] = []
    for i in sel:
        if groups and i - groups[-1][-1] <= 1:
            groups[-1].append(i)
        else:
            groups.append([i])
    groups.sort(key=lambda g: -sum(diff[i] for i in g))
    keep = sorted(groups[:FOCUS_MAX], key=lambda g: g[0])
    return [{"frame": [g[0], g[-1]], "detik": [round(g[0] / m.FPS, 2), round((g[-1] + 1) / m.FPS, 2)],
             "adegan": next(n for lo, hi, n in SHOTS if lo <= g[0] <= hi)} for g in keep]


def scene_of(i: int) -> str:
    lo, hi, n = next(s for s in SHOTS if s[0] <= i <= s[1])
    return f"adegan {n} (f{lo}–f{hi})"


def info_bar(width: int, i: int) -> np.ndarray:
    sec = f"{i / m.FPS:.2f}".replace(".", ",")
    return m.text_bar(width, m.INFO_H, [f"frame {i}  /  {LAST}        detik {sec}", scene_of(i)], 1.25, 3, bg=225)


def cmd_bahan() -> None:
    if NILAI.exists():
        raise SystemExit(f"{rel(NILAI)} sudah ada — bahan tidak ditimpa")
    b2 = m.jload(OUT / "t4b_banding.json")
    if b2["stage"]["strokes/svg"]["bytes_sama"] == N:
        raise SystemExit("strokes 54 / 54 byte-identik — tidak perlu penilaian")
    NILAI.mkdir(parents=True)
    seed = secrets.randbits(32)
    x_is_scratch = random.Random(seed).random() < 0.5
    src = {"scratch_iso2": MP4_SCR, "produksi_trim": MP4_NEW}
    order = ["scratch_iso2", "produksi_trim"] if x_is_scratch else ["produksi_trim", "scratch_iso2"]
    vids = {k: m.read_video(p, N) for k, p in src.items()}
    fokus = focus_ranges(b2["strokes_png_piksel_beda"])

    def frames():
        for i in range(N):
            row = [m.fit_h(vids[k][i], PANEL_H) for k in order]
            gap = np.full((PANEL_H, GUTTER, 3), 128, np.uint8)
            body = np.hstack([row[0], gap, row[1]])
            top = np.hstack([m.text_bar(row[0].shape[1], m.TITLE_H, ["X"], 1.4, 3), np.full((m.TITLE_H, GUTTER, 3), 128, np.uint8),
                             m.text_bar(row[1].shape[1], m.TITLE_H, ["Y"], 1.4, 3)])
            yield np.vstack([top, body, info_bar(body.shape[1], i)])

    stem = "XY_f0-53"
    enc, size, k = None, None, 0
    for img in frames():
        img = np.ascontiguousarray(img)
        if enc is None:
            size = (img.shape[1], img.shape[0])
            enc = (m.encoder(NILAI / f"{stem}.mp4", size), m.encoder(NILAI / f"{stem}_lambat_0.5x.mp4", size))
        raw = img.tobytes()
        enc[0].stdin.write(raw)
        enc[1].stdin.write(raw)
        enc[1].stdin.write(raw)
        k += 1
    for p in enc:
        p.stdin.close()
        if p.wait() != 0:
            raise SystemExit("ffmpeg gagal")
    kunci = {"seed": seed, "aturan": "random.Random(seed).random() < 0.5 → X = scratch_iso2, selain itu X = produksi_trim",
             "X": order[0], "Y": order[1],
             "sumber": {kk: {"berkas": rel(p), "sha256": m.file_sha(p), "frame": [0, LAST]} for kk, p in src.items()}}
    (NILAI / "kunci.json").write_text(json.dumps(kunci, indent=1, ensure_ascii=False), encoding="utf-8")
    pub = {"video": {"normal": f"{stem}.mp4", "lambat": f"{stem}_lambat_0.5x.mp4"}, "frame": k, "ukuran": list(size),
           "fokus": fokus, "fokus_dasar": f"piksel beda PNG strokes [5] per frame (simetris), ≥ persentil {FOCUS_PCT}, gabung "
                                          f"frame bersebelahan, maks {FOCUS_MAX} rentang"}
    m.jsave(NILAI / "bahan_buat.json", pub)
    print(f"bahan: {k} frame, ukuran {size}; kunci -> {rel(NILAI / 'kunci.json')} (tidak dicetak)")
    for f in fokus:
        print(f"  fokus f{f['frame'][0]}–f{f['frame'][1]} detik {f['detik']} adegan {f['adegan']}")


def cmd_cek() -> None:
    res = {}
    for p in sorted(NILAI.glob("*.mp4")):
        res[p.name] = {**ffprobe(p), "sha256": m.file_sha(p)}
        print(p.name, {k: v for k, v in res[p.name].items() if k != "sha256"})
    m.jsave(NILAI / "bahan.json", res)


# ── B6 ─────────────────────────────────────────────
def cmd_f1() -> None:
    ref, cur = f1_list(F1_REF), f1_list(F1_NEW)
    changed = sorted(r for r in ref if r in cur and cur[r] != ref[r])
    missing = sorted(set(ref) - set(cur))
    extra = sorted(set(cur) - set(ref))

    def split(xs):
        return [x for x in xs if x.startswith(RUN_WRITES)], [x for x in xs if not x.startswith(RUN_WRITES)]

    res = {"ref": rel(F1_REF), "baru": rel(F1_NEW), "n_ref": len(ref), "n_baru": len(cur)}
    for lab, xs in (("berubah", changed), ("hilang", missing), ("tambahan", extra)):
        ok, bad = split(xs)
        res[lab] = {"tulisan_run": len(ok), "di_luar": bad,
                    "per_folder_run": dict(Counter(next(w for w in RUN_WRITES if x.startswith(w)) + (
                        x[len("work/clips/klip5@0-2.25/"):].split("/", 1)[0] if x.startswith("work/clips/klip5@0-2.25/") else "")
                        for x in ok))}
        if lab == "berubah":
            res[lab]["daftar_tulisan_run"] = ok
    fr = [x for x in changed if x.startswith("work/clips/klip5@0-2.25/frames/")]
    res["frame_trim_berubah"] = fr
    res["identik_kecuali_tulisan_run"] = not (res["berubah"]["di_luar"] or res["hilang"]["di_luar"] or res["tambahan"]["di_luar"])
    res["lama_identik"] = sum(1 for r in ref if r in cur and cur[r] == ref[r])
    m.jsave(OUT / "t4b_f1.json", res)
    print(json.dumps({k: v for k, v in res.items() if k != "berubah"}, ensure_ascii=False, indent=1))
    print("berubah:", json.dumps({k: v for k, v in res["berubah"].items()}, ensure_ascii=False))


# ── 4b (h): data berharga ──────────────────────────
DATA_OUT = OUT / "t608_data.json"
DATA_TRACKED = ROOT / "scripts" / "t608_data.json"
ISO1 = OUT / "arsip_iso1"
R0_JSON = m.ROOT / "work" / "t608_scratch" / "r0" / "r0.json"
LINE_MAX = 130
RUN_PASTE = [   # tempelan Rio sesudah run produksi (2026-10-28), apa adanya
    "run selesai. Exit code 0. Baris terakhir keluaran:",
    "Waktu per stage: [1] ingest 0.5 s, [1b] isolate 58.0 s, [2] segment 913.8 s, [2c] depth 18.0 s, [3] stabilize 12.2 s, "
    "[4] vectorize 5.5 s, [5] stylize 6.3 s, [6] export 2.8 s",
    "Angka lain dari keluaran run (catat apa adanya): sebelum run nvidia-smi memory.free 3488 MiB, RAM bebas 6973 MiB; [1b] target 54, "
    "kosong 0, tanpa_prompt 0, SAM 55.6 s, skor f0 18.40; [2] 16.24–16.65 s/frame (frame pertama 22.20 s), peak reserved 3276.0 MiB; "
    "QC 8/54 gagal big_blobs (indeks 0, 6, 7, 9, 10, 14, 15, 16); [4] pass 1 T_high 0.024896337650716277, T_low 0.010551941022276882; "
    "[6] ffprobe 54 frame, 24 fps, 1080x1002, 2.250 s, 730.4 KiB; SVG 54 berkas.",
]
R1_PIN = {   # docs/04 "Hasil + rancangan T-608 Tahap 1" no. 5 (dibaca dari web 2026-10-27; TIDAK diunduh)
    "model": "caidas/swin2SR-compressed-sr-x4-48", "commit": "1079c3c5fcea8a2c047b270ac8f4c6c18adb27c9",
    "berkas": {"model.safetensors": {"bytes": 49073568,
                                     "sha256": "3fc6baf86cda09a42e5f8167ca4b29fc043137f3c05b8d362717dd1bfef143b4"},
               "config.json": {"bytes": 776}, "preprocessor_config.json": {"bytes": 152}},
    "lisensi": "kode transformers Apache-2.0; repo mv-lab/swin2sr Apache-2.0 (commit 6f25d768); model card apache-2.0",
    "status": "tidak dijalankan, tidak diunduh (keputusan akhir T-608 no. 2)",
}


def pj(x, ind: int = 0, lead: int = 0) -> str:
    """JSON ber-indentasi 1; objek / larik yang muat satu baris (≤ LINE_MAX) ditulis satu baris."""
    s = json.dumps(x, ensure_ascii=False)
    if not isinstance(x, (dict, list)) or not x or ind + lead + len(s) <= LINE_MAX:
        return s
    pad = " " * (ind + 1)
    if isinstance(x, dict):
        rows = []
        for k, v in x.items():
            ks = json.dumps(k, ensure_ascii=False)
            rows.append(f"{pad}{ks}: {pj(v, ind + 1, len(ks) + 2)}")
        return "{\n" + ",\n".join(rows) + "\n" + " " * ind + "}"
    if all(not isinstance(v, (dict, list)) for v in x):  # larik skalar: beberapa elemen per baris
        rows, cur = [], ""
        for v in (json.dumps(v, ensure_ascii=False) for v in x):
            if cur and len(pad) + len(cur) + 2 + len(v) > LINE_MAX:
                rows.append(cur)
                cur = v
            else:
                cur = f"{cur}, {v}" if cur else v
        rows.append(cur)
        return "[\n" + ",\n".join(pad + r for r in rows) + "\n" + " " * ind + "]"
    return "[\n" + ",\n".join(pad + pj(v, ind + 1) for v in x) + "\n" + " " * ind + "]"


def med_p90(d: dict) -> dict:
    """Statistik {n, median, p90, ...} → [median, p90]; nilai lain apa adanya."""
    out = {}
    for k, v in d.items():
        if isinstance(v, dict) and "median" in v and "p90" in v:
            out[k] = [v["median"], v["p90"]]
        elif isinstance(v, dict):
            out[k] = med_p90(v)
        else:
            out[k] = v
    return out


def iso_summary(iso_dir: Path) -> dict:
    man = m.jload(iso_dir / "manifest.json")
    rows = jsonl_last(iso_dir / "frames.jsonl")
    jobs = [{k: j[k] for k in ("shot", "prompt_frames", "prompts", "n_infer", "infer_s")} for j in man["jobs"]]
    idx = [i for i, r in sorted(rows.items()) if r["status"] != "tanpa_prompt"]
    per = {"frame": ranges(idx), "score_logit": [rows[i]["targets"].get("1", {}).get("score_logit") for i in idx],
           "area_px": [rows[i]["targets"].get("1", {}).get("area_px") for i in idx]}
    return {"manifest_sha256": m.file_sha(iso_dir / "manifest.json"), "key_sha256": man["key_sha256"],
            "points_sha256": man["key"]["points_sha256"], "frames_pixel_sha256": man["key"]["frames_pixel_sha256"],
            "model": man["model"], "frame_count": man["frame_count"], "frames": man["frames"], "jobs": jobs, "run": man["run"],
            "per_frame": per}


def build_data() -> tuple[bytes, list[Path]]:
    srcs: list[Path] = []

    def use(p: Path):
        srcs.append(p)
        return m.jload(p)

    rk = use(OUT / "ringkas.json")
    r0 = use(R0_JSON)
    cut = use(OUT / "cut.json")
    dg = use(OUT / "iso_diag_prompt.json")
    ver = use(OUT / "t4b_verifikasi.json")
    b2 = use(OUT / "t4b_banding.json")
    b3 = use(OUT / "t4b_metrik.json")
    pts_p = NEW / "target" / "points.json"
    pts = use(pts_p)
    qa, qb = qc_rows(NEW), qc_rows(SCR)
    srcs += [NEW / "qc_report.json", SCR / "qc_report.json"]
    diag = {"device": dg["device"], "mask_threshold": dg["mask_threshold"],
            "varian_kolom": ["n_kiri", "n_kanan", "kotak", "skor_objek", "luas_px", "porsi_di_dalam_kotak"], "frame": {}}
    for f, d in dg["frame"].items():
        diag["frame"][f] = {k: d[k] for k in ("klik_kiri", "klik_kanan", "kanan_di_dalam_kotak", "kotak", "skor_gpu_run_rio")}
        diag["frame"][f]["varian"] = {n: [v["n_kiri"], v["n_kanan"], v["kotak"], v["skor_objek"], v["luas_px"], v["porsi_di_dalam_kotak"]]
                                      for n, v in d["varian"].items()}
    clicks = [{"frame": e["frame_index"], "frame_pixel_sha256_di_titik": e["frame_pixel_sha256"],
               "frame_sha256_berkas": m.file_sha(NEW / "frames" / e["frame_file"])} for e in pts["frames"]]
    st = b2["stage"]
    trim_iso = iso_summary(NEW / "isolate")
    iso2_per = iso_summary(SCR / "isolate")["per_frame"]
    same = all(trim_iso["per_frame"][k] == iso2_per[k][:N] for k in ("score_logit", "area_px"))
    trim_iso["per_frame"] = ("= iso2_scratch per_frame f0–53 (dicek sama)" if same else trim_iso["per_frame"])
    metrics_keys =("strok", "panjang_px", "porsi_panjang_luar_masker", "komponen", "komponen_kecil", "fg_porsi")

    def b3_block(blk: dict) -> dict:
        return {lab: {**{k: blk[lab][k]["median"] for k in metrics_keys},
                      "jumlah_per_tipe_per_frame": blk[lab]["jumlah_per_tipe_per_frame"],
                      "panjang_per_tipe_per_frame": blk[lab]["panjang_per_tipe_per_frame"]} for lab in VERS}

    def group_causes(d: dict, key: str) -> dict:
        grp: dict[str, list[int]] = {}
        for i, v in d.items():
            grp.setdefault(" | ".join(v["penyebab"]) + f" [tipe {', '.join(v[key])}]", []).append(int(i))
        return {k: ranges(v) for k, v in grp.items()}

    data = {
        "format": "t608-data/1",
        "dibuat": "python scripts/t608_tahap4b.py data -> work/t608/t608_data.json; scripts/t608_data.json = salinan Write, byte-identik "
                  "(cek: python scripts/t608_tahap4b.py cek_data, sebelum work/t608_scratch/ dihapus)",
        "docs04": ["Hasil + rancangan T-608 Tahap 1", "Hasil T-608 1b-2 langkah 1", "Hasil T-608 1b-2 percobaan kedua",
                   "Keputusan akhir T-608 (2026-10-28)", "Hasil run produksi klip5@0-2.25 (2026-10-28)", "Penutupan T-608"],
        "tahap1": {
            "definisi": "scripts/t608_tahap1.py (docstring); statistik per klip = [median, p90] (n / mean / min / max di work/t608/ringkas.json)",
            "kualitas": med_p90(rk["kualitas"]), "ambigu": med_p90(rk["ambigu"]), "tepi": med_p90(rk["tepi"]),
            "korelasi_klip5": rk["korelasi_klip5"], "korelasi_klip5_f0_62": rk["korelasi_klip5_f0_62"],
            "klip5_per_segmen": med_p90(rk["klip5_per_segmen"]), "klip5_per_jendela24": med_p90(rk["klip5_per_jendela24"]),
            "r0": {k: r0[k] for k in ("param", "cpu", "frame_ukuran", "rata2", "catatan")}, "cut_klip5": cut},
        "diagnosis_klik_iso1": diag,
        "isolate": {"iso1": iso_summary(ISO1 / "isolate"), "iso2_scratch": iso_summary(SCR / "isolate"),
                    "produksi_trim": trim_iso},
        "titik_iso2": {"berkas": rel(pts_p), "sha256": m.file_sha(pts_p), "bytes": pts_p.stat().st_size,
                       "catatan": "isi = JSON yang di-parse (bukan byte asli); salinan asli = work/clips/klip5@0-2.25/target/points.json",
                       "frame_klik": clicks, "isi": pts},
        "r1_pin_usulan": R1_PIN,
        "run_produksi": {"perintah": r"python -m rotoscope run samples\klip5.mp4 --end 2.25 --yes", "exit_code": 0,
                         "tempelan_rio": RUN_PASTE},
        "b1": {"mp4": ver["mp4"], "svg": ver["svg"], "masker_3": {k: v for k, v in ver["masker_3"].items() if k != "contoh_f0"},
               "produksi_lama": {k: ver["produksi_lama"][k] for k in ("pembanding", "n_ref", "per_awalan", "identik")},
               "lolos": ver["lolos"]},
        "b2": {
            "definisi": "f0–53 produksi trim vs scratch iso2; sidik isi = array decode PNG / npy / npz; contours = strok tanpa track_id, "
                        "urutan / arah titik + urutan groups diabaikan",
            "per_stage": {k: {x: (ranges(y) if isinstance(y, list) else y) for x, y in v.items()
                              if x not in ("metrik_beda", "big_blobs_n", "gagal_baru", "gagal_scratch", "per_tipe_beda")}
                          for k, v in st.items()},
            "per_tipe_beda": {k: {t: ranges(v) for t, v in st[k]["per_tipe_beda"].items()} for k in ("contours", "strokes/svg")},
            "qc": {"gagal_baru": ranges(map(int, st["qc"]["gagal_baru"])), "gagal_scratch": ranges(map(int, st["qc"]["gagal_scratch"])),
                   "metrik_beda": ranges(map(int, st["qc"]["metrik_beda"])),
                   "f47": {"baru": {k: qa[47][k] for k in ("area_px", "area_median_px", "area_vs_median", "fail_reasons")},
                           "scratch": {k: qb[47][k] for k in ("area_px", "area_median_px", "area_vs_median", "fail_reasons")},
                           "area_drop_min": m.jload(NEW / "qc_report.json")["qc"]["area_drop_min"],
                           "area_median_window": st["qc"]["area_median_window"]},
                   "big_blobs_n_sama_f0_53": st["qc"]["big_blobs_n"]["baru"] == st["qc"]["big_blobs_n"]["scratch_f0_63"][:N]},
            "stable_jendela_beda": b2["stable_jendela_beda"], "stable_penyebab": b2["stable_penyebab"],
            "stable_groups_beda_piksel": b2["stable_groups_beda_piksel"], "clip_stats": b2["clip_stats"],
            "kontrafaktual_4": b2["kontrafaktual_4"] | {"ambang_scratch_frame_beda": {t: ranges(v) for t, v in
                                                                                     b2["kontrafaktual_4"]["ambang_scratch_frame_beda"].items()}},
            "contours_penyebab": group_causes(b2["contours_penyebab"], "tipe_beda"),
            "strokes_penyebab": group_causes(b2["strokes_penyebab"], "tipe_svg_beda"),
            "strokes_png_piksel_beda": b2["strokes_png_piksel_beda"]},
        "b3": {"definisi": b3["definisi"], "median_per_frame_dan_per_tipe": {"f0-53": b3_block(b3["semua"]),
                                                                          **{f"adegan {k}": b3_block(v) for k, v in b3["per_adegan"].items()}}},
        "penilaian": ["scripts/t608_penilaian.json", "scripts/t608_penilaian_1b2.json", "scripts/t608_penilaian_4b.json"],
    }
    data["sumber"] = {rel(p): m.file_sha(p) for p in srcs + [ISO1 / "isolate" / "manifest.json", SCR / "isolate" / "manifest.json",
                                                              NEW / "isolate" / "manifest.json"]}
    return (pj(data) + "\n").encode("utf-8"), srcs


def cmd_data() -> None:
    raw, _ = build_data()
    lines = raw.decode("utf-8").splitlines()
    json.loads(raw.decode("utf-8"))                        # JSON valid
    DATA_OUT.write_bytes(raw)
    print(f"-> {rel(DATA_OUT)}: {len(raw)} B, sha256 {hashlib.sha256(raw).hexdigest()}, {len(lines)} baris, baris terpanjang "
          f"{max(len(x) for x in lines)} karakter")


def cmd_cek_data() -> None:
    raw, _ = build_data()
    a, b = DATA_OUT.read_bytes(), DATA_TRACKED.read_bytes() if DATA_TRACKED.is_file() else None
    res = {"bangun_ulang_sha256": hashlib.sha256(raw).hexdigest(), "work_sha256": hashlib.sha256(a).hexdigest(),
           "scripts_sha256": b and hashlib.sha256(b).hexdigest(), "bangun_ulang_sama_work": raw == a, "scripts_sama_work": b == a,
           "bytes": [len(raw), len(a), b and len(b)], "json_valid": json.loads(b.decode("utf-8")) == json.loads(a.decode("utf-8")) if b else None}
    print(json.dumps(res, ensure_ascii=False, indent=1))
    m.jsave(OUT / "t4b_cek_data.json", res)


def main(argv: list[str]) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    cmds = {"verifikasi": cmd_verifikasi, "banding": cmd_banding, "metrik": cmd_metrik, "bahan": cmd_bahan, "cek": cmd_cek,
            "f1": cmd_f1, "data": cmd_data, "cek_data": cmd_cek_data}
    if len(argv) != 1 or argv[0] not in cmds:
        print(__doc__)
        return 2
    cmds[argv[0]]()
    if "torch" in sys.modules:
        raise SystemExit("torch ikut terimpor — tidak boleh")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
