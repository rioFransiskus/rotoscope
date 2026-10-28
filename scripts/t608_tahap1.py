"""T-608 Tahap 1 (alat, F3; CPU, BACA-SAJA terhadap produksi). docs/04 "Keputusan T-608 (restorasi input klip5), 2026-10-27".

    python scripts/t608_tahap1.py cut      # skor cut klip5 (fungsi [3]) -> 19 cut belum berlabel dicocokkan -> work/t608/cut.json
    python scripts/t608_tahap1.py ukur     # kualitas input 5 klip: ffprobe + σ derau, blok 8×8, ketajaman -> work/t608/kualitas.json
    python scripts/t608_tahap1.py ambigu   # seg/probs: ambigu lengan/tangan vs Background + entropi pita tepi -> work/t608/ambigu.json
    python scripts/t608_tahap1.py r0       # waktu CPU R0 (NLM temporal + CLAHE) pada 3 frame klip5 -> work/t608_scratch/r0/r0.json
    python scripts/t608_tahap1.py bahan    # work/t608/nilai_1/: klip5_tandai.mp4 + _lambat_0.5x.mp4 + papan_cut_<i>.png
    python scripts/t608_tahap1.py cek      # ffprobe / ukuran bahan -> work/t608/bahan.json
    python scripts/t608_tahap1.py tepi     # kontras tepi orang (Sobel di batas foreground seg) + ukuran subjek -> work/t608/tepi.json
    python scripts/t608_tahap1.py ringkas  # gabung kualitas + ambigu + tepi + r0 (+ korelasi) -> work/t608/ringkas.json + tabel di layar

Hanya MEMBACA samples/, work/clips/*, out/; menulis hanya ke work/t608/ dan work/t608_scratch/. Tanpa GPU (torch tidak diimpor),
tanpa run / batch / ingest. Frame dibaca sebagai angka (cv2), tidak ditampilkan.

Definisi metrik (dicatat sebelum dihitung):
- Y = 0,299 R + 0,587 G + 0,114 B dari `frames/` (faktor skala 5 klip = 1,0: `frames/` = area aktif sumber tanpa resize).
- σ derau Immerkaer (1996): σ = √(π/2) · mean|Y ∗ M| / 6, M = [[1,−2,1],[−2,4,−2],[1,−2,1]] (tanpa baris / kolom tepi). `sigma_flat`
  (utama) = sama, tetapi piksel tepi dibuang (10% gradien Sobel tertinggi; Tai & Yang 2008) → tekstur / garis tidak terhitung derau.
- Blok 8×8: B = mean|ΔY| melintasi batas grid 8 px / mean|ΔY| di dalam blok (horizontal dan vertikal dirata-rata). Grid = koordinat
  SUMBER (offset kotak crop), karena kodek membagi blok dari pojok frame sumber. 1,0 = tanpa blok; makin besar makin berblok. 16 px dicatat.
- Ketajaman: VoL = var(Laplacian 3×3 Y); `vol_norm` = VoL / var(Y) (bebas kontras). Derau ikut menaikkan VoL, jadi dibaca bersama σ.
- PSNR bolak-balik ×2: Y → ½ (INTER_AREA) → ×2 (INTER_CUBIC) vs Y. Tinggi = sedikit isi di atas ½ Nyquist (resolusi efektif rendah).
- Ambigu (seg/probs, uint8 /255): top-2 kelas = {Background, lengan/tangan} dan selisih p1 − p2 < 0,2 → per luas foreground (argmax ≠
  Background). `amb_person` = sama untuk {Background, kelas orang apa pun}. Entropi (bit, 29 kelas, p dinormalkan) rata-rata di pita
  tepi orang ±6 px (jarak L2 ke batas foreground ≤ 6, di dalam maupun di luar).
- Tepi (ditambahkan sesudah hasil `ukur` / `ambigu` terlihat, untuk memisahkan "lembut" dari "isi kerumunan"): batas foreground seg
  (`seg/classmap` ≠ Background; piksel fg dengan tetangga 4-arah bukan fg). `edge_grad` = rata-rata |Sobel Y| di batas itu;
  `edge_ratio` = `edge_grad` / p90 |Sobel Y| seluruh frame (1 = batas orang setajam tepi terkuat gambar; kecil = batas lembut / kontras
  rendah). `fg_frac` = luas foreground / luas frame.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")      # pengaman: Tahap 1 tanpa GPU

import cv2  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rotoscope import stabilize as stb  # noqa: E402
from rotoscope.config import load_pipeline  # noqa: E402

CLIPS_DIR = ROOT / "work" / "clips"
OUT = ROOT / "work" / "t608"
NILAI = OUT / "nilai_1"
SCRATCH = ROOT / "work" / "t608_scratch"
CLIPS = ("test", "Klip2", "klip3", "klip4", "klip5")
K5 = "klip5"
NEW_SOURCE = ROOT / "samples" / "kilp5_new.mp4"           # berkas baru Rio: HANYA ffprobe (tidak di-ingest)
FPS = 24

# klip5: jendela peristiwa (docs/04 diagnosis T-305 1b / T-605 / T-305 induk) + jendela 24 frame (screening 1b ≤ 24 frame)
K5_SEGMENTS = (("f0-62 kerumunan", 0, 62), ("f63-77 zoom", 63, 77), ("f78-91", 78, 91), ("f92-104 cut", 92, 104),
               ("f105-132 objek lain", 105, 132))
WIN = 24
LABEL_RIO_CUT = (92, 104)                                  # cut sungguhan (label Rio)
UNLABELED_EXPECTED = (22, 60, 61, 62, 63, 64, 65, 66, 67, 68, 72, 73, 74, 88, 89, 90, 91, 106, 108)   # docs/05 "Peninjauan cut_diff"

IMMERKAER = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
EDGE_FRAC = 0.10
BLOCK = 8
BLOCK_ALT = 16
AMB_MARGIN = 0.2
BAND_PX = 6
PROB_SCALE = 255.0
ARM_WORDS = ("Hand", "Arm")
BG = "Background"

# R0 (OpenCV): 3 frame klip5 tanpa cut di jendela ±2 (cut terdeteksi 22, 60–68, 72–74, 88–91)
R0_FRAMES = (12, 45, 80)
R0_TEMPORAL = 5
R0_TEMPLATE = 7
R0_SEARCH = 21
R0_CLAHE_CLIP = 2.0
R0_CLAHE_TILE = 8

# bahan
PANEL_H = 880                                              # FOTO 720 × 668 → 948 × 880; hasil 1080 × 1002 → 948 × 880
TITLE_H = 56
INFO_H = 110
BOARD_W, BOARD_H = 1920, 1080
BOARD_HEADER_H = 54
BOARD_ROWS, BOARD_COLS = 2, 2
CELL_LABEL_H = 44
CELL_PAD = 12


def jload(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def jsave(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"  -> {p.relative_to(ROOT).as_posix()}")


def rnd(x, n=4):
    return None if x is None else round(float(x), n)


def stats(vals) -> dict:
    a = np.asarray([v for v in vals if v is not None and np.isfinite(v)], np.float64)
    if a.size == 0:
        return {"n": 0, "median": None, "p90": None, "mean": None, "min": None, "max": None}
    return {"n": int(a.size), "median": rnd(np.median(a)), "p90": rnd(np.percentile(a, 90)), "mean": rnd(a.mean()),
            "min": rnd(a.min()), "max": rnd(a.max())}


def read_bgr(p: Path) -> np.ndarray:
    im = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)
    if im is None:
        raise SystemExit(f"{p} tidak terbaca")
    return im


def luma(bgr: np.ndarray) -> np.ndarray:
    f = bgr.astype(np.float32)
    return 0.114 * f[..., 0] + 0.587 * f[..., 1] + 0.299 * f[..., 2]


def frame_names(klip: str) -> list[str]:
    return sorted(p.name for p in (CLIPS_DIR / klip / "frames").glob("frame_*.png"))


def crop_offset(meta: dict) -> tuple[int, int]:
    c = meta.get("crop")
    return (int(c["box_source"][0]), int(c["box_source"][1])) if c else (0, 0)


def windows(n: int) -> list[tuple[str, int, int]]:
    return [(f"f{a}-{min(a + WIN, n) - 1}", a, min(a + WIN, n) - 1) for a in range(0, n, WIN)]


def by_ranges(per_frame: list[dict], keys: tuple[str, ...], ranges) -> dict:
    return {lab: {k: stats([r[k] for r in per_frame if a <= r["index"] <= b]) for k in keys} for lab, a, b in ranges}


# ── cut ────────────────────────────────────────────
def cut_info() -> dict:
    cfg = load_pipeline()
    thr = cfg.stabilize.temporal.cut_diff
    work = CLIPS_DIR / K5
    clip = stb.load_clip(work)
    scores = stb.cut_scores([stb.frame_thumb(work / "frames" / n) for n in clip.names])
    cut = stb.detect_cuts(scores, thr)
    detected = [k for k in range(len(scores)) if cut[k]]
    lo, hi = LABEL_RIO_CUT
    unlabeled = [k for k in detected if not lo <= k <= hi]
    return {"cut_diff": thr, "n_frames": len(scores), "detected": detected, "n_detected": len(detected),
            "label_rio": [k for k in detected if lo <= k <= hi], "unlabeled": unlabeled,
            "unlabeled_scores": {k: rnd(scores[k]) for k in unlabeled},
            "unlabeled_matches_docs05": tuple(unlabeled) == UNLABELED_EXPECTED,
            "definisi": "cut di frame k = skor(k−1 → k) > cut_diff; papan: kiri f(k−1) SEBELUM, kanan f(k) SESUDAH"}


def cmd_cut() -> None:
    d = cut_info()
    print(f"cut_diff {d['cut_diff']}: {d['n_detected']} cut; label Rio {len(d['label_rio'])}; belum berlabel {len(d['unlabeled'])} "
          f"= docs/05 {d['unlabeled_matches_docs05']}: {d['unlabeled']}")
    jsave(OUT / "cut.json", d)
    if not d["unlabeled_matches_docs05"]:
        raise SystemExit("cut belum berlabel TIDAK sama dengan docs/05 — berhenti")


# ── kualitas input ─────────────────────────────────
def ffprobe(path: Path) -> dict:
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
           "stream=codec_name,profile,width,height,avg_frame_rate,bit_rate,pix_fmt,nb_frames,color_space,color_range:"
           "format=bit_rate,duration,size:format_tags=encoder", "-of", "json", str(path)]
    out = json.loads(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout)
    s, f = out["streams"][0], out["format"]
    num, den = (int(x) for x in s["avg_frame_rate"].split("/"))
    fps = num / den if den else None
    vb = int(s["bit_rate"]) if s.get("bit_rate", "N/A") != "N/A" else None
    w, h = int(s["width"]), int(s["height"])
    return {"codec": s["codec_name"], "profile": s.get("profile"), "width": w, "height": h, "fps": rnd(fps, 3),
            "pix_fmt": s.get("pix_fmt"), "color_space": s.get("color_space"), "video_bitrate": vb,
            "format_bitrate": int(f["bit_rate"]) if f.get("bit_rate") else None, "duration_s": rnd(float(f["duration"]), 3),
            "size_bytes": int(f["size"]), "encoder": f.get("tags", {}).get("encoder"),
            "bpp": rnd(vb / (w * h * fps), 4) if vb and fps else None}


def file_sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def noise_sigma(y: np.ndarray) -> tuple[float, float]:
    r = np.abs(cv2.filter2D(y, cv2.CV_32F, IMMERKAER, borderType=cv2.BORDER_REFLECT))[1:-1, 1:-1]
    g = cv2.magnitude(cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3))[1:-1, 1:-1]
    c = math.sqrt(math.pi / 2.0) / 6.0
    thr = np.percentile(g, 100.0 * (1.0 - EDGE_FRAC))
    return c * float(r.mean()), c * float(r[g < thr].mean())


def blockiness(y: np.ndarray, period: int, off_x: int, off_y: int) -> float:
    dx = np.abs(np.diff(y, axis=1))
    dy = np.abs(np.diff(y, axis=0))
    bx = (np.arange(dx.shape[1]) + 1 + off_x) % period == 0
    by = (np.arange(dy.shape[0]) + 1 + off_y) % period == 0
    bh = dx[:, bx].mean() / max(dx[:, ~bx].mean(), 1e-6)
    bv = dy[by, :].mean() / max(dy[~by, :].mean(), 1e-6)
    return float((bh + bv) / 2.0)


def sharpness(y: np.ndarray) -> tuple[float, float]:
    vol = float(cv2.Laplacian(y, cv2.CV_32F, ksize=1).var())
    return vol, vol / max(float(y.var()), 1e-6)


def psnr_roundtrip(y: np.ndarray, s: int = 2) -> float:
    h, w = y.shape
    small = cv2.resize(y, (round(w / s), round(h / s)), interpolation=cv2.INTER_AREA)
    back = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    mse = float(((back - y) ** 2).mean())
    return 99.0 if mse <= 0 else 10.0 * math.log10(255.0 ** 2 / mse)


QUALITY_KEYS = ("sigma_full", "sigma_flat", "block8", "block16", "vol", "vol_norm", "psnr_rt2")


def frame_quality(bgr: np.ndarray, off: tuple[int, int]) -> dict:
    y = luma(bgr)
    s_full, s_flat = noise_sigma(y)
    vol, vol_n = sharpness(y)
    return {"sigma_full": s_full, "sigma_flat": s_flat, "block8": blockiness(y, BLOCK, *off),
            "block16": blockiness(y, BLOCK_ALT, *off), "vol": vol, "vol_norm": vol_n, "psnr_rt2": psnr_roundtrip(y)}


def cmd_ukur() -> None:
    res = {"definisi": __doc__.split("Definisi metrik (dicatat sebelum dihitung):", 1)[1].strip(), "klip": {}}
    for klip in CLIPS:
        t0 = time.perf_counter()
        meta = jload(CLIPS_DIR / klip / "meta.json")
        src = Path(meta["source_path"])
        probe = ffprobe(src)
        off = crop_offset(meta)
        box = meta["crop"]["box_source"] if meta.get("crop") else [0, 0, meta["source_width"], meta["source_height"]]
        names = frame_names(klip)
        per = []
        for i, n in enumerate(names):
            r = frame_quality(read_bgr(CLIPS_DIR / klip / "frames" / n), off)
            per.append({"index": i, **{k: rnd(v, 5) for k, v in r.items()}})
        entry = {
            "source": {"path": src.relative_to(ROOT).as_posix(), "sha256": file_sha(src),
                       "sha256_cocok_meta": (meta.get("crop") or {}).get("source_sha256") in (None, file_sha(src)), **probe},
            "area_aktif_sumber": {"box": box, "w": box[2], "h": box[3], "porsi_kanvas": rnd(box[2] * box[3] / (probe["width"] * probe["height"]))},
            "ukuran_kerja": [meta["working_width"], meta["working_height"]],
            "faktor_skala": rnd(meta["working_width"] / box[2]),
            "n_frames": len(names), "grid_offset_sumber": list(off),
            "ringkas": {k: stats([r[k] for r in per]) for k in QUALITY_KEYS},
            "per_frame": per,
        }
        if klip == K5:
            entry["per_segmen"] = by_ranges(per, QUALITY_KEYS, K5_SEGMENTS)
            entry["per_jendela24"] = by_ranges(per, QUALITY_KEYS, windows(len(per)))
        res["klip"][klip] = entry
        s = entry["ringkas"]
        print(f"{klip}: {len(names)} frame, σ_flat {s['sigma_flat']['median']} (p90 {s['sigma_flat']['p90']}), "
              f"blok8 {s['block8']['median']}, vol_norm {s['vol_norm']['median']}, psnr×2 {s['psnr_rt2']['median']} "
              f"[{time.perf_counter() - t0:.1f} s]")
    if NEW_SOURCE.is_file():
        res["kilp5_new_info"] = {"catatan": "berkas baru di samples/ — HANYA ffprobe + sha256, tidak di-ingest (larangan Rio)",
                                 "path": NEW_SOURCE.relative_to(ROOT).as_posix(), "sha256": file_sha(NEW_SOURCE),
                                 **ffprobe(NEW_SOURCE)}
    jsave(OUT / "kualitas.json", res)


# ── ambiguitas seg/probs ───────────────────────────
def seg_classes(klip: str) -> list[str]:
    return jload(CLIPS_DIR / klip / "seg" / "manifest.json")["classes"]


def frame_ambiguity(probs: np.ndarray, arm_ids: np.ndarray) -> dict:
    p = probs
    i1 = p.argmax(0)
    p1 = np.take_along_axis(p, i1[None], 0)[0].astype(np.int16)
    q = p.copy()
    np.put_along_axis(q, i1[None], 0, 0)
    i2 = q.argmax(0)
    p2 = np.take_along_axis(q, i2[None], 0)[0].astype(np.int16)
    close = (p1 - p2) < AMB_MARGIN * PROB_SCALE
    is_arm1, is_arm2 = np.isin(i1, arm_ids), np.isin(i2, arm_ids)
    bg_pair = (i1 == 0) | (i2 == 0)
    amb_arm = close & (((i1 == 0) & is_arm2) | ((i2 == 0) & is_arm1))
    amb_person = close & bg_pair & (i1 != i2)
    fg = (i1 != 0).astype(np.uint8)
    fg_px = int(fg.sum())
    arm_zone = int((is_arm1 | is_arm2).sum())
    if fg_px:
        d_in = cv2.distanceTransform(fg, cv2.DIST_L2, 5)
        d_out = cv2.distanceTransform(1 - fg, cv2.DIST_L2, 5)
        band = ((fg == 1) & (d_in <= BAND_PX)) | ((fg == 0) & (d_out <= BAND_PX))
        pb = p[:, band].astype(np.float32)
        pb /= np.maximum(pb.sum(0, keepdims=True), 1e-6)
        ent = float((-(pb * np.log2(np.where(pb > 0, pb, 1.0)))).sum(0).mean())
        band_px = int(band.sum())
    else:
        ent, band_px = None, 0
    return {"fg_px": fg_px, "amb_arm_px": int(amb_arm.sum()), "amb_person_px": int(amb_person.sum()), "arm_zone_px": arm_zone,
            "amb_arm_frac": rnd(amb_arm.sum() / fg_px, 5) if fg_px else None,
            "amb_person_frac": rnd(amb_person.sum() / fg_px, 5) if fg_px else None,
            "amb_arm_of_zone": rnd(amb_arm.sum() / arm_zone, 5) if arm_zone else None,
            "ent_band": rnd(ent, 5), "band_px": band_px}


AMB_KEYS = ("amb_arm_frac", "amb_person_frac", "amb_arm_of_zone", "ent_band")


def cmd_ambigu() -> None:
    res = {"definisi": {"ambigu": f"top-2 = {{{BG}, lengan/tangan}} dan p1 − p2 < {AMB_MARGIN} (probs uint8 / 255); per luas "
                                  f"foreground argmax ≠ {BG}",
                        "amb_person": f"top-2 = {{{BG}, kelas orang apa pun}} dan p1 − p2 < {AMB_MARGIN}",
                        "amb_arm_of_zone": "ambigu lengan / piksel dengan top-1 atau top-2 = lengan/tangan",
                        "ent_band": f"entropi (bit, 29 kelas, dinormalkan) rata-rata di pita ±{BAND_PX} px batas foreground"},
           "klip": {}}
    for klip in CLIPS:
        t0 = time.perf_counter()
        classes = seg_classes(klip)
        if classes[0] != BG:
            raise SystemExit(f"{klip}: kelas 0 bukan {BG}")
        arm_ids = np.array([i for i, c in enumerate(classes) if any(w in c for w in ARM_WORDS)])
        names = frame_names(klip)
        qc = {f["index"]: bool(f.get("fail_reasons")) for f in jload(CLIPS_DIR / klip / "qc_report.json")["frames"]}
        per = []
        for i, n in enumerate(names):
            with np.load(CLIPS_DIR / klip / "seg" / "probs" / (Path(n).stem + ".npz")) as z:
                probs = z["probs"]
            per.append({"index": i, "qc_fail": qc.get(i), **frame_ambiguity(probs, arm_ids)})
        entry = {"n_frames": len(per), "kelas_lengan": [classes[i] for i in arm_ids], "n_kelas": len(classes),
                 "qc_fail": sum(1 for r in per if r["qc_fail"]),
                 "ringkas": {k: stats([r[k] for r in per]) for k in AMB_KEYS}, "per_frame": per}
        if klip == K5:
            entry["per_segmen"] = by_ranges(per, AMB_KEYS, K5_SEGMENTS)
            entry["per_jendela24"] = by_ranges(per, AMB_KEYS, windows(len(per)))
        res["klip"][klip] = entry
        s = entry["ringkas"]
        print(f"{klip}: ambigu lengan/fg {s['amb_arm_frac']['median']} (p90 {s['amb_arm_frac']['p90']}), orang/fg "
              f"{s['amb_person_frac']['median']}, entropi pita {s['ent_band']['median']} [{time.perf_counter() - t0:.1f} s]")
    jsave(OUT / "ambigu.json", res)


# ── kontras tepi orang ─────────────────────────────
EDGE_KEYS = ("edge_grad", "edge_ratio", "fg_frac")
CROSS4 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))


def frame_edge(bgr: np.ndarray, classmap: np.ndarray) -> dict:
    y = luma(bgr)
    g = cv2.magnitude(cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3))
    fg = (classmap != 0).astype(np.uint8)
    edge = (fg == 1) & (cv2.erode(fg, CROSS4, borderType=cv2.BORDER_REPLICATE) == 0)
    p90 = float(np.percentile(g, 90))
    eg = float(g[edge].mean()) if edge.any() else None
    return {"edge_grad": eg, "edge_ratio": eg / p90 if eg is not None and p90 > 0 else None,
            "fg_frac": float(fg.mean()), "edge_px": int(edge.sum())}


def cmd_tepi() -> None:
    res = {"definisi": "lihat docstring: Tepi", "klip": {}}
    for klip in CLIPS:
        names = frame_names(klip)
        per = []
        for i, n in enumerate(names):
            cm = cv2.imdecode(np.fromfile(CLIPS_DIR / klip / "seg" / "classmap" / n, np.uint8), cv2.IMREAD_UNCHANGED)
            r = frame_edge(read_bgr(CLIPS_DIR / klip / "frames" / n), cm)
            per.append({"index": i, **{k: (rnd(v, 5) if isinstance(v, float) else v) for k, v in r.items()}})
        entry = {"n_frames": len(per), "ringkas": {k: stats([r[k] for r in per]) for k in EDGE_KEYS}, "per_frame": per}
        if klip == K5:
            entry["per_segmen"] = by_ranges(per, EDGE_KEYS, K5_SEGMENTS)
            entry["per_jendela24"] = by_ranges(per, EDGE_KEYS, windows(len(per)))
        res["klip"][klip] = entry
        s = entry["ringkas"]
        print(f"{klip}: edge_grad {s['edge_grad']['median']}, edge_ratio {s['edge_ratio']['median']}, fg_frac {s['fg_frac']['median']}")
    jsave(OUT / "tepi.json", res)


# ── R0: waktu CPU ──────────────────────────────────
def cmd_r0() -> None:
    q = jload(OUT / "kualitas.json")["klip"][K5]
    sigma = q["ringkas"]["sigma_flat"]["median"]
    h = round(sigma, 1)
    names = frame_names(K5)
    out_dir = SCRATCH / "r0"
    out_dir.mkdir(parents=True, exist_ok=True)
    half = R0_TEMPORAL // 2
    clahe = cv2.createCLAHE(clipLimit=R0_CLAHE_CLIP, tileGridSize=(R0_CLAHE_TILE, R0_CLAHE_TILE))
    rows = []
    for k in R0_FRAMES:
        imgs = [read_bgr(CLIPS_DIR / K5 / "frames" / names[j]) for j in range(k - half, k + half + 1)]
        t0 = time.perf_counter()
        multi = cv2.fastNlMeansDenoisingColoredMulti(imgs, half, R0_TEMPORAL, None, h, h, R0_TEMPLATE, R0_SEARCH)
        t_multi = time.perf_counter() - t0
        t0 = time.perf_counter()
        single = cv2.fastNlMeansDenoisingColored(imgs[half], None, h, h, R0_TEMPLATE, R0_SEARCH)
        t_single = time.perf_counter() - t0
        t0 = time.perf_counter()
        lab = cv2.cvtColor(multi, cv2.COLOR_BGR2LAB)
        lab[..., 0] = clahe.apply(lab[..., 0])
        cl = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        t_clahe = time.perf_counter() - t0
        before, after_m, after_s, after_c = (frame_quality(x, crop_offset(jload(CLIPS_DIR / K5 / "meta.json")))
                                             for x in (imgs[half], multi, single, cl))
        rows.append({"frame": k, "t_multi_s": rnd(t_multi, 3), "t_single_s": rnd(t_single, 3), "t_clahe_s": rnd(t_clahe, 4),
                     "sebelum": {kk: rnd(v) for kk, v in before.items()},
                     "sesudah_multi": {kk: rnd(v) for kk, v in after_m.items()},
                     "sesudah_single": {kk: rnd(v) for kk, v in after_s.items()},
                     "sesudah_multi_clahe": {kk: rnd(v) for kk, v in after_c.items()},
                     "beda_multi_mean_abs": rnd(np.abs(multi.astype(np.int16) - imgs[half].astype(np.int16)).mean(), 3)})
        print(f"f{k}: NLM multi {t_multi:.2f} s, single {t_single:.2f} s, CLAHE {t_clahe * 1000:.1f} ms; σ_flat "
              f"{before['sigma_flat']:.2f} → {after_m['sigma_flat']:.2f}")
    res = {"param": {"h": h, "hColor": h, "h_dari": f"median sigma_flat klip5 {sigma}", "temporalWindowSize": R0_TEMPORAL,
                     "templateWindowSize": R0_TEMPLATE, "searchWindowSize": R0_SEARCH,
                     "clahe": {"clipLimit": R0_CLAHE_CLIP, "tile": R0_CLAHE_TILE, "kanal": "L (Lab), sesudah NLM multi"}},
           "cpu": {"os_cpu_count": os.cpu_count(), "cv2_threads": cv2.getNumThreads(), "cv2": cv2.__version__},
           "frame_ukuran": [int(imgs[0].shape[1]), int(imgs[0].shape[0])], "per_frame": rows,
           "rata2": {"t_multi_s": rnd(np.mean([r["t_multi_s"] for r in rows]), 3),
                     "t_single_s": rnd(np.mean([r["t_single_s"] for r in rows]), 3),
                     "t_clahe_s": rnd(np.mean([r["t_clahe_s"] for r in rows]), 4)},
           "catatan": "angka saja; gambar hasil TIDAK disimpan"}
    jsave(out_dir / "r0.json", res)


# ── bahan penandaan ────────────────────────────────
def fit_h(img: np.ndarray, h: int) -> np.ndarray:
    return cv2.resize(img, (round(img.shape[1] * h / img.shape[0] / 2) * 2, h), interpolation=cv2.INTER_AREA)


def text_bar(width: int, height: int, text: str, scale: float, thick: int, bg: int = 255) -> np.ndarray:
    bar = np.full((height, width, 3), bg, np.uint8)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, thick)
    cv2.putText(bar, text, (max(8, (width - tw) // 2), (height + th) // 2), cv2.FONT_HERSHEY_DUPLEX, scale, (0, 0, 0), thick,
                cv2.LINE_AA)
    return bar


def info_bar(width: int, f: int, n: int) -> np.ndarray:
    sec = f"{f / FPS:.2f}".replace(".", ",")
    return text_bar(width, INFO_H, f"frame {f}  /  {n - 1}        detik {sec}", 2.4, 5, bg=225)


def read_video(path: Path):
    cap = cv2.VideoCapture(str(path))
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        yield fr
    cap.release()


def encoder(path: Path, size: tuple[int, int]) -> subprocess.Popen:
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{size[0]}x{size[1]}",
           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", "-r", str(FPS), str(path)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def make_video() -> dict:
    names = frame_names(K5)
    n = len(names)
    src = ROOT / "out" / f"{K5}.mp4"
    enc, size, k = None, None, 0
    for f, res in enumerate(read_video(src)):
        if f >= n:
            raise SystemExit("out/klip5.mp4 lebih panjang dari frames/")
        a, b = fit_h(read_bgr(CLIPS_DIR / K5 / "frames" / names[f]), PANEL_H), fit_h(res, PANEL_H)
        top = np.hstack([text_bar(a.shape[1], TITLE_H, "FOTO (asli)", 1.5, 3), text_bar(b.shape[1], TITLE_H, "HASIL (sekarang)", 1.5, 3)])
        img = np.vstack([top, np.hstack([a, b])])
        img = np.ascontiguousarray(np.vstack([img, info_bar(img.shape[1], f, n)]))
        if enc is None:
            size = (img.shape[1], img.shape[0])
            enc = (encoder(NILAI / "klip5_tandai.mp4", size), encoder(NILAI / "klip5_tandai_lambat_0.5x.mp4", size))
        raw = img.tobytes()
        enc[0].stdin.write(raw)
        enc[1].stdin.write(raw)
        enc[1].stdin.write(raw)
        k += 1
    for p in enc:
        p.stdin.close()
        if p.wait() != 0:
            raise SystemExit("ffmpeg gagal")
    if k != n:
        raise SystemExit(f"out/klip5.mp4 {k} frame ≠ frames/ {n}")
    return {"frames": k, "size": list(size), "sumber_hasil": f"out/{K5}.mp4", "sha256_sumber_hasil": file_sha(src)}


def make_boards(cuts: list[int]) -> list[dict]:
    names = frame_names(K5)
    per_board = BOARD_ROWS * BOARD_COLS
    cell_w = (BOARD_W - CELL_PAD * (BOARD_COLS + 1)) // BOARD_COLS
    cell_h = (BOARD_H - BOARD_HEADER_H - CELL_PAD * (BOARD_ROWS + 1)) // BOARD_ROWS
    probe = read_bgr(CLIPS_DIR / K5 / "frames" / names[0])
    fh = min(cell_h - CELL_LABEL_H, int((cell_w - CELL_PAD) / 2 * probe.shape[0] / probe.shape[1]))
    fw = round(fh * probe.shape[1] / probe.shape[0])
    boards = []
    n_boards = math.ceil(len(cuts) / per_board)
    for b in range(n_boards):
        img = np.full((BOARD_H, BOARD_W, 3), 255, np.uint8)
        head = text_bar(BOARD_W, BOARD_HEADER_H, f"Papan cut klip5 - bagian {b + 1} / {n_boards}   (kiri = SEBELUM, kanan = SESUDAH)",
                        1.3, 3, bg=225)
        img[:BOARD_HEADER_H] = head
        items = []
        for j, k in enumerate(cuts[b * per_board:(b + 1) * per_board]):
            no = b * per_board + j + 1
            r, c = divmod(j, BOARD_COLS)
            x0 = CELL_PAD + c * (cell_w + CELL_PAD)
            y0 = BOARD_HEADER_H + CELL_PAD + r * (cell_h + CELL_PAD)
            sec = f"{k / FPS:.2f}".replace(".", ",")
            lab = text_bar(cell_w, CELL_LABEL_H, f"No. {no}:  frame {k - 1}  ->  frame {k}   (detik {sec})", 1.1, 2, bg=245)
            img[y0:y0 + CELL_LABEL_H, x0:x0 + cell_w] = lab
            pair = [cv2.resize(read_bgr(CLIPS_DIR / K5 / "frames" / names[i]), (fw, fh), interpolation=cv2.INTER_AREA)
                    for i in (k - 1, k)]
            px = x0 + (cell_w - (2 * fw + CELL_PAD)) // 2
            py = y0 + CELL_LABEL_H
            img[py:py + fh, px:px + fw] = pair[0]
            img[py:py + fh, px + fw + CELL_PAD:px + 2 * fw + CELL_PAD] = pair[1]
            items.append({"no": no, "sebelum": k - 1, "sesudah": k, "detik": round(k / FPS, 3)})
        path = NILAI / f"papan_cut_{b + 1}.png"
        ok, buf = cv2.imencode(".png", img)
        if not ok:
            raise SystemExit("imencode gagal")
        path.write_bytes(buf.tobytes())
        boards.append({"file": path.name, "size": [BOARD_W, BOARD_H], "frame_panel": [fw, fh], "isi": items})
    return boards


def cmd_bahan() -> None:
    cut = jload(OUT / "cut.json")
    if not cut["unlabeled_matches_docs05"]:
        raise SystemExit("cut.json tidak cocok docs/05 — jalankan `cut` dulu")
    NILAI.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    video = make_video()
    boards = make_boards(cut["unlabeled"])
    jsave(OUT / "bahan_buat.json", {"video": video, "papan": boards, "waktu_s": rnd(time.perf_counter() - t0, 1)})


def cmd_cek() -> None:
    res = {}
    for p in sorted(NILAI.iterdir()):
        if p.suffix == ".mp4":
            cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
                   "stream=codec_name,pix_fmt,width,height,avg_frame_rate,nb_read_frames:format=duration,size", "-of", "json", str(p)]
            d = json.loads(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout)
            res[p.name] = {**d["streams"][0], **d["format"]}
        elif p.suffix == ".png":
            im = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_UNCHANGED)
            res[p.name] = {"width": int(im.shape[1]), "height": int(im.shape[0]), "size": p.stat().st_size}
        else:
            res[p.name] = {"size": p.stat().st_size}
        res[p.name]["sha256"] = file_sha(p)
        print(p.name, {k: v for k, v in res[p.name].items() if k != "sha256"})
    jsave(OUT / "bahan.json", res)


# ── ringkas ────────────────────────────────────────
def spearman(a, b) -> float | None:
    from scipy.stats import spearmanr
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if len(pairs) < 5:
        return None
    return rnd(spearmanr([p[0] for p in pairs], [p[1] for p in pairs]).statistic, 3)


def cmd_ringkas() -> None:
    q, a = jload(OUT / "kualitas.json")["klip"], jload(OUT / "ambigu.json")["klip"]
    r0p = SCRATCH / "r0" / "r0.json"
    res = {"kualitas": {}, "ambigu": {}, "korelasi_klip5": {}, "r0": jload(r0p)["rata2"] if r0p.is_file() else None}
    print("| klip | area aktif sumber | kerja | skala | bitrate video kb/s | bpp | σ flat med / p90 | σ full med | blok8 med / p90 | "
          "blok16 med | vol_norm med / p90 | PSNR ×2 med |")
    for k in CLIPS:
        e, s = q[k], q[k]["ringkas"]
        res["kualitas"][k] = {"area": e["area_aktif_sumber"], "kerja": e["ukuran_kerja"], "skala": e["faktor_skala"],
                              "bitrate": e["source"]["video_bitrate"], "bpp": e["source"]["bpp"], "profile": e["source"]["profile"],
                              "fps_sumber": e["source"]["fps"], **{kk: s[kk] for kk in QUALITY_KEYS}}
        print(f"| {k} | {e['area_aktif_sumber']['w']}×{e['area_aktif_sumber']['h']} | {e['ukuran_kerja'][0]}×{e['ukuran_kerja'][1]} | "
              f"{e['faktor_skala']} | {round(e['source']['video_bitrate'] / 1000)} | {e['source']['bpp']} | "
              f"{s['sigma_flat']['median']} / {s['sigma_flat']['p90']} | {s['sigma_full']['median']} | "
              f"{s['block8']['median']} / {s['block8']['p90']} | {s['block16']['median']} | "
              f"{s['vol_norm']['median']} / {s['vol_norm']['p90']} | {s['psnr_rt2']['median']} |")
    print("\n| klip | ambigu lengan/fg med / p90 | ambigu lengan / zona lengan med | ambigu orang/fg med / p90 | entropi pita med / p90 | QC gagal |")
    for k in CLIPS:
        s = a[k]["ringkas"]
        res["ambigu"][k] = {**{kk: s[kk] for kk in AMB_KEYS}, "qc_fail": a[k]["qc_fail"], "n": a[k]["n_frames"]}
        print(f"| {k} | {s['amb_arm_frac']['median']} / {s['amb_arm_frac']['p90']} | {s['amb_arm_of_zone']['median']} | "
              f"{s['amb_person_frac']['median']} / {s['amb_person_frac']['p90']} | {s['ent_band']['median']} / {s['ent_band']['p90']} | "
              f"{a[k]['qc_fail']} / {a[k]['n_frames']} |")
    tp = OUT / "tepi.json"
    t = jload(tp)["klip"] if tp.is_file() else None
    if t:
        res["tepi"] = {}
        print("\n| klip | edge_grad med | edge_ratio med / p10 | fg_frac med |")
        for k in CLIPS:
            s = t[k]["ringkas"]
            p10 = rnd(np.percentile([r["edge_ratio"] for r in t[k]["per_frame"] if r["edge_ratio"] is not None], 10))
            res["tepi"][k] = {**{kk: s[kk] for kk in EDGE_KEYS}, "edge_ratio_p10": p10}
            print(f"| {k} | {s['edge_grad']['median']} | {s['edge_ratio']['median']} / {p10} | {s['fg_frac']['median']} |")
    qa, aa = q[K5]["per_frame"], a[K5]["per_frame"]
    for qk in ("sigma_flat", "block8", "vol_norm", "psnr_rt2"):
        for ak in ("amb_arm_frac", "amb_person_frac", "ent_band"):
            res["korelasi_klip5"][f"{qk}~{ak}"] = spearman([r[qk] for r in qa], [r[ak] for r in aa])
    if t:
        ta = t[K5]["per_frame"]
        for tk in EDGE_KEYS:
            for ak in ("amb_arm_frac", "amb_person_frac", "ent_band"):
                res["korelasi_klip5"][f"{tk}~{ak}"] = spearman([r[tk] for r in ta], [r[ak] for r in aa])
        # dalam segmen f0–62 saja (tanpa pengaruh pergantian isi antar segmen)
        lo, hi = K5_SEGMENTS[0][1], K5_SEGMENTS[0][2]
        res["korelasi_klip5_f0_62"] = {}
        for qk, src in (("sigma_flat", qa), ("vol_norm", qa), ("psnr_rt2", qa), ("block8", qa), ("edge_ratio", ta), ("fg_frac", ta)):
            for ak in ("amb_person_frac", "ent_band"):
                res["korelasi_klip5_f0_62"][f"{qk}~{ak}"] = spearman([r[qk] for r in src[lo:hi + 1]],
                                                                     [r[ak] for r in aa[lo:hi + 1]])
        print("\nSpearman klip5 f0–62 saja:", res["korelasi_klip5_f0_62"])
    print("\nSpearman klip5 per frame (133):", res["korelasi_klip5"])
    print("\nklip5 per segmen:")
    for lab, _, _ in K5_SEGMENTS:
        qs, as_ = q[K5]["per_segmen"][lab], a[K5]["per_segmen"][lab]
        ts = t[K5]["per_segmen"][lab] if t else None
        print(f"| {lab} | σ flat {qs['sigma_flat']['median']} | blok8 {qs['block8']['median']} | vol_norm {qs['vol_norm']['median']} | "
              f"PSNR×2 {qs['psnr_rt2']['median']} | ambigu lengan {as_['amb_arm_frac']['median']} | ambigu orang "
              f"{as_['amb_person_frac']['median']} | entropi {as_['ent_band']['median']} |"
              + (f" edge_ratio {ts['edge_ratio']['median']} | fg {ts['fg_frac']['median']} |" if ts else ""))
    print("\nklip5 per jendela 24:")
    for lab in q[K5]["per_jendela24"]:
        qs, as_ = q[K5]["per_jendela24"][lab], a[K5]["per_jendela24"][lab]
        print(f"| {lab} | σ flat {qs['sigma_flat']['median']} | blok8 {qs['block8']['median']} | vol_norm {qs['vol_norm']['median']} | "
              f"ambigu lengan {as_['amb_arm_frac']['median']} | ambigu orang {as_['amb_person_frac']['median']} | entropi "
              f"{as_['ent_band']['median']} |")
    res["klip5_per_segmen"] = {lab: {"kualitas": q[K5]["per_segmen"][lab], "ambigu": a[K5]["per_segmen"][lab],
                                     **({"tepi": t[K5]["per_segmen"][lab]} if t else {})}
                               for lab, _, _ in K5_SEGMENTS}
    res["klip5_per_jendela24"] = {lab: {"kualitas": q[K5]["per_jendela24"][lab], "ambigu": a[K5]["per_jendela24"][lab]}
                                  for lab in q[K5]["per_jendela24"]}
    jsave(OUT / "ringkas.json", res)


def main(argv: list[str]) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    cmds = {"cut": cmd_cut, "ukur": cmd_ukur, "ambigu": cmd_ambigu, "tepi": cmd_tepi, "r0": cmd_r0, "bahan": cmd_bahan,
            "cek": cmd_cek, "ringkas": cmd_ringkas}
    if len(argv) != 1 or argv[0] not in cmds:
        print(__doc__)
        return 2
    cmds[argv[0]]()
    if "torch" in sys.modules:
        raise SystemExit("torch ikut terimpor — tidak boleh di Tahap 1")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
