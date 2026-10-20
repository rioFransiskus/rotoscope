"""T-607 Tahap 1a (CPU, baca-saja terhadap klip; TANPA GPU / unduhan): potong letterbox / pillarbox / durasi di stage [1].
Keputusan + kriteria: docs/04 "Keputusan T-607" + D-013.

    python scripts/t607_study.py r2           # reproduksi R1 / R2 (aturan T-602) di 6 klip lama dari frames/ yang ada: rentang penuh
                                              # (+ rentang studi T-602) + jendela 24 frame langkah 12; dibandingkan dengan
                                              # scripts/t602_data.json dan arsip work/t602/letterbox.json -> work/t607/r2_lama.json
    python scripts/t607_study.py cropdetect   # ffmpeg cropdetect (PEMBANDING, bukan aturan) di 6 sumber: resolusi kerja (rantai
                                              # fps + scale ingest) dan resolusi sumber -> work/t607/cropdetect_lama.json
    python scripts/t607_study.py probe        # ffprobe 6 sumber + kotak R2 di koordinat sumber (pembulatan keluar / ke dalam,
                                              # genap) + ukuran kerja sesudah crop (P-src vs P-frm) -> work/t607/probe_lama.json
    python scripts/t607_study.py ukur         # di scratch work/t607_scratch/: ingest biasa (= frames/ klip?), waktu deteksi dari PNG
                                              # vs pipa rawvideo, ingest ulang ber-crop (Klip2, klip5), trim sebelum / sesudah fps
                                              # -> work/t607/ukur.json
    python scripts/t607_study.py trimframe    # trim berbasis indeks frame (fps -> trim=start_frame:end_frame) = frames/ a..b-1?
                                              # + trim + crop klip5 -> work/t607/trimframe.json
    python scripts/t607_study.py probe-baru   # HANYA ffprobe (resolusi, SAR / DAR, rotasi, fps, durasi) klip validasi samples/klip6..klip9
                                              # (nama asli, tidak disalin; persetujuan Rio no. 10) -> work/t607/probe_baru.json
    python scripts/t607_study.py label        # LABEL.md Rio -> scripts/t607_labels.json (SEBELUM deteksi; isian salah -> exit 1)
    python scripts/t607_study.py baru         # Tahap 1b: HANYA bila R2 = beku + label = sha256 docs/04: ingest klip6..klip9 ke
                                              # work/t607_scratch/baru/, R2 (+ R1 info) rentang penuh + jendela, cropdetect
                                              # -> K1 / K3 vs label -> work/t607/baru.json
    python scripts/t607_study.py papan        # papan K2 work/t607/nilai/*.png (berpita / dipotong: awal, tengah, akhir; lainnya 1
                                              # frame) + papan.json
    python scripts/t607_study.py penilaian    # PENILAIAN.md Rio (+ klarifikasi) -> scripts/t607_penilaian.json (SESUDAH deteksi)
    python scripts/t607_study.py probe-k2     # HANYA ffprobe klip10, klip11 (K2) -> work/t607/probe_k2.json

R2 di sini = salinan PERSIS `detect_crop(rule="R2")` scripts/t602_study.py (aritmetika sama: akumulasi float64, hasil float32), parameter
DIBEKUKAN di R2_PARAMS (sidik `r2_fingerprint()` dicatat di docs/04 "R2 dibekukan" sebelum klip baru diukur). Hanya MEMBACA
work/clips/*, out/, samples/ dan scripts/t602_data.json; menulis ke work/t607/ dan work/t607_scratch/ (keduanya di-ignore git).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rotoscope import ingest as ing  # noqa: E402

CLIPS_DIR = ROOT / "work" / "clips"
SAMPLES = ROOT / "samples"
OUT = ROOT / "work" / "t607"
SCR = ROOT / "work" / "t607_scratch"
T602_DATA = ROOT / "scripts" / "t602_data.json"
T602_LETTERBOX = ROOT / "work" / "t602" / "letterbox.json"
OLD_CLIPS = ("test", "test_short", "Klip2", "klip3", "klip4", "klip5")
T602_CLIPS = ("test", "Klip2", "klip3", "klip4", "klip5")
STUDY_RANGES = {"klip5": [(0, 91)], "klip3": [(0, 23)]}            # rentang studi T-602 (selain rentang penuh)
NEW_CLIPS = ("klip6", "klip7", "klip8", "klip9")                   # klip validasi T-607 (samples/, label Rio dulu)
K2_CLIPS = ("klip10", "klip11")                                     # klip berpita untuk K2 (keputusan sesudah K3 no. 4)
LABEL_MD = OUT / "label" / "LABEL.md"
LABELS_JSON = ROOT / "scripts" / "t607_labels.json"                # ter-track; ditulis SEBELUM deteksi klip baru (papan buta)
R2_FROZEN_FINGERPRINT = "d40c3dfdd270a715a7bbc9bfbec96ce69ce613a1dac7480c1bcd607ec6eb1da0"   # docs/04 "R2 DIBEKUKAN"
LABELS_JSON_SHA256 = "4c71a8132c6e4137d1f2c12031214f2a477e05580cca513072dfab1f30ca9071"      # docs/04 "Label Rio T-607"
BARU_DIR = SCR / "baru"                                             # ingest klip baru (scratch; work/clips/ tidak disentuh)
NILAI_DIR = OUT / "nilai"                                           # papan K2 + PENILAIAN.md

# ── R2 BEKU (T-602 Tahap 1a; docs/04 "Hasil T-602 Tahap 1a" butir e) ──
R2_RULE = "R2"
R2_VERSION = 1
R2_PARAMS = {
    "STATIC_STD_MAX": 3.0,      # piksel statis: simpangan baku temporal (maks antar kanal, 0–255) <= 3
    "STATIC_LINE_FRAC": 0.99,   # baris / kolom statis: >= 99% pikselnya statis (di dalam persegi aktif saat itu)
    "UNIFORM_STD_MAX": 6.0,     # baris / kolom seragam: simpangan baku spasial warna rata-rata temporal (maks antar kanal) <= 6
    "BAND_MIN_PX": 8,           # pita minimum (dan syarat pasangan R2) agar kotak teks di sebelah dalamnya ikut dikupas
    "MIN_ACTIVE_FRAC": 0.10,    # area aktif < 10% frame -> tidak dipotong (pengaman)
}
WIN_LEN, WIN_STEP, WIN_TOL_PX = 24, 12, 4      # prosedur jendela T-602 (K3): potong berlebih bila > acuan + 4 px di salah satu sisi
EDGES = ("top", "bottom", "left", "right")
OPP = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}
RULES = ("R1", "R2")

TARGET_FPS = ing.DEFAULT_TARGET_FPS
WORKING_WIDTH = ing.DEFAULT_WORKING_WIDTH
CROPDETECT = {"limit": 24, "round": 2, "reset": 0}   # pembanding; default ffmpeg round = 16 (diganti 2 agar sebanding per piksel)
CROPDETECT_RE = re.compile(r"crop=(\d+):(\d+):(\d+):(\d+)")
SHOWINFO_PTS_RE = re.compile(r"n:\s*(\d+)\s+pts:\s*(-?\d+)\s+pts_time:([-\d.]+)")


def r2_fingerprint() -> str:
    """sha256 JSON kanonik {rule, version, params} — penanda pembekuan R2 (docs/04)."""
    doc = {"rule": R2_RULE, "version": R2_VERSION, "params": R2_PARAMS}
    return hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def jsave(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"  -> {p.relative_to(ROOT).as_posix()}")


def file_sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ── Deteksi (salinan persis t602_study: temporal_stats / line_ok / shrink / peel_bands / detect_crop) ──
class Stats:
    """Akumulator rata-rata + simpangan baku temporal per piksel (BGR 0–255, float64), hasil float32 seperti T-602."""

    def __init__(self) -> None:
        self.s = self.s2 = None
        self.k = 0

    def add(self, bgr: np.ndarray) -> None:
        a = bgr.astype(np.float64)
        if self.s is None:
            self.s, self.s2 = np.zeros_like(a), np.zeros_like(a)
        self.s += a
        self.s2 += a * a
        self.k += 1

    def result(self) -> tuple[np.ndarray, np.ndarray]:
        m = self.s / self.k
        sd = np.sqrt(np.maximum(self.s2 / self.k - m * m, 0.0)).max(axis=2)
        return m.astype(np.float32), sd.astype(np.float32)


def read_png(p: Path) -> np.ndarray:
    return cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)


def stats_of_pngs(paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    st = Stats()
    for p in paths:
        st.add(read_png(p))
    return st.result()


def line_ok(m, sd, rect, edge, need_uniform, P=R2_PARAMS) -> bool:
    y0, y1, x0, x1 = rect
    if y1 - y0 < 1 or x1 - x0 < 1:
        return False
    if edge in ("top", "bottom"):
        y = y0 if edge == "top" else y1 - 1
        sdl, ml = sd[y, x0:x1], m[y, x0:x1]
    else:
        x = x0 if edge == "left" else x1 - 1
        sdl, ml = sd[y0:y1, x], m[y0:y1, x]
    if (sdl <= P["STATIC_STD_MAX"]).mean() < P["STATIC_LINE_FRAC"]:
        return False
    return (not need_uniform) or float(ml.std(axis=0).max()) <= P["UNIFORM_STD_MAX"]


def shrink(rect, edge) -> None:
    i = {"top": 0, "bottom": 1, "left": 2, "right": 3}[edge]
    rect[i] += 1 if edge in ("top", "left") else -1


def peel_bands(m, sd, edges, P=R2_PARAMS):
    H, W = sd.shape
    rect = [0, H, 0, W]
    band = dict.fromkeys(EDGES, 0)
    changed = True
    while changed:
        changed = False
        for e in EDGES:
            if e not in edges:
                continue
            while line_ok(m, sd, rect, e, True, P):
                shrink(rect, e)
                band[e] += 1
                changed = True
    return rect, band


def detect_crop(m, sd, rule: str = R2_RULE, P=R2_PARAMS) -> dict:
    H, W = sd.shape
    allowed = set(EDGES)
    while True:
        rect, band = peel_bands(m, sd, allowed, P)
        if rule == "R1":
            break
        keep = {e for e in allowed if band[e] >= P["BAND_MIN_PX"] and band[OPP[e]] >= P["BAND_MIN_PX"]}
        if keep == allowed:
            break
        allowed = keep
    overlay = dict.fromkeys(EDGES, 0)
    for e in EDGES:
        if rule == "R2" and e not in ("top", "bottom"):
            continue
        if band[e] >= P["BAND_MIN_PX"]:
            while line_ok(m, sd, rect, e, False, P):
                shrink(rect, e)
                overlay[e] += 1
    y0, y1, x0, x1 = rect
    area = max(0, y1 - y0) * max(0, x1 - x0)
    applied = area >= P["MIN_ACTIVE_FRAC"] * H * W
    return {"rule": rule, "frame_wh": [W, H], "band_px": band, "overlay_px": overlay, "applied": bool(applied),
            "active_xyxy_incl": [x0, y0, x1 - 1, y1 - 1] if applied else [0, 0, W - 1, H - 1],
            "active_wh": [x1 - x0, y1 - y0] if applied else [W, H],
            "active_frac": round(area / (H * W), 4) if applied else 1.0}


def windows_of(n: int) -> list[tuple[int, int]]:
    return [(a, min(n - 1, a + WIN_LEN - 1)) for a in range(0, max(1, n - WIN_LEN + 1), WIN_STEP)]


def window_eval(det: dict, ref: list[int]) -> dict:
    x0, y0, x1, y1 = det["active_xyxy_incl"]
    excess = {"left": x0 - ref[0], "top": y0 - ref[1], "right": ref[2] - x1, "bottom": ref[3] - y1}
    return {"active_xyxy_incl": det["active_xyxy_incl"], "band_px": det["band_px"], "overlay_px": det["overlay_px"],
            "excess_vs_full_px": excess, "fp": any(v > WIN_TOL_PX for v in excess.values()),
            "fn": any(v < -WIN_TOL_PX for v in excess.values())}


def frame_paths(clip: str) -> list[Path]:
    return sorted((CLIPS_DIR / clip / "frames").glob("frame_*.png"))


# ── r2 ─────────────────────────────────────────────
DET_KEYS = ("band_px", "overlay_px", "active_xyxy_incl", "active_wh", "active_frac", "applied")
SUMM_KEYS = ("n", "fp", "fn", "fp_ranges", "fn_ranges", "max_excess_px", "min_excess_px")


def cmd_r2() -> None:
    t602 = json.loads(T602_DATA.read_text(encoding="utf-8"))
    arch = json.loads(T602_LETTERBOX.read_text(encoding="utf-8")) if T602_LETTERBOX.is_file() else None
    res = {"r2": {"rule": R2_RULE, "version": R2_VERSION, "params": R2_PARAMS, "fingerprint": r2_fingerprint()},
           "window": {"len": WIN_LEN, "step": WIN_STEP, "tol_px": WIN_TOL_PX}, "clips": {}, "checks": {}}
    tot = {r: {"n": 0, "fp": 0, "fn": 0} for r in RULES}
    tot602 = {r: {"n": 0, "fp": 0, "fn": 0} for r in RULES}
    for clip in OLD_CLIPS:
        paths = frame_paths(clip)
        n = len(paths)
        t0 = time.perf_counter()
        m, sd = stats_of_pngs(paths)
        t_stats = time.perf_counter() - t0
        t0 = time.perf_counter()
        full = {r: detect_crop(m, sd, r) for r in RULES}
        t_det = time.perf_counter() - t0
        ranges = {f"f0-{n - 1}": {r: {k: full[r][k] for k in DET_KEYS} for r in RULES}}
        for a, b in STUDY_RANGES.get(clip, []):
            ms, sds = stats_of_pngs(paths[a:b + 1])
            ranges[f"f{a}-{b}"] = {r: {k: v for k, v in detect_crop(ms, sds, r).items() if k in DET_KEYS} for r in RULES}
        ref = full["R2"]["active_xyxy_incl"]
        wins = {r: [] for r in RULES}
        for a, b in windows_of(n):
            ms, sds = stats_of_pngs(paths[a:b + 1])
            for r in RULES:
                wins[r].append({"range": [a, b], **window_eval(detect_crop(ms, sds, r), ref)})
        summ = {}
        for r in RULES:
            ws = wins[r]
            summ[r] = {"n": len(ws), "fp": sum(w["fp"] for w in ws), "fn": sum(w["fn"] for w in ws),
                       "fp_ranges": [w["range"] for w in ws if w["fp"]], "fn_ranges": [w["range"] for w in ws if w["fn"]],
                       "max_excess_px": {e: max(w["excess_vs_full_px"][e] for w in ws) for e in EDGES},
                       "min_excess_px": {e: min(w["excess_vs_full_px"][e] for w in ws) for e in EDGES}}
            for k in ("n", "fp", "fn"):
                tot[r][k] += summ[r][k]
                if clip in T602_CLIPS:
                    tot602[r][k] += summ[r][k]
        row = {"n_frames": n, "frame_wh": full["R2"]["frame_wh"], "ranges": ranges,
               "r1_equals_r2_full": full["R1"]["active_xyxy_incl"] == full["R2"]["active_xyxy_incl"],
               "windows_summary": summ, "windows": wins,
               "time_s": {"read_stats_full": round(t_stats, 3), "detect_R1_R2_full": round(t_det, 4),
                          "per_frame_read_stats_ms": round(1000 * t_stats / n, 2)}}
        chk = {}
        if clip in T602_CLIPS:
            c602 = t602["crop"]["clips"][clip]
            chk["crop_vs_t602_data"] = (full["R2"]["active_xyxy_incl"] == c602["crop_xyxy_incl"]
                                        and full["R2"]["active_wh"] == c602["crop_wh"]
                                        and full["R2"]["active_frac"] == c602["active_frac"]
                                        and full["R2"]["band_px"] == c602["band_px"]
                                        and full["R2"]["overlay_px"] == c602["overlay_px"]
                                        and full["R2"]["applied"] == c602["applied"] and n == c602["n_frames"])
            l602 = t602["letterbox"]["clips"][clip]
            chk["ranges_vs_t602_data"] = all(ranges[k][r] == l602["rentang"][r][k] for r in RULES for k in l602["rentang"][r])
            chk["windows_summary_vs_t602_data"] = all({k: summ[r][k] for k in SUMM_KEYS} == {k: l602["jendela_24"][r][k] for k in SUMM_KEYS}
                                                      for r in RULES)
            if arch is not None:
                aw = arch["clips"][clip]["windows"]
                chk["windows_each_vs_t602_archive"] = all(
                    len(aw[r]) == len(wins[r]) and all(
                        {k: x[k] for k in ("range", "active_xyxy_incl", "band_px", "overlay_px", "excess_vs_full_px", "fp", "fn")}
                        == {k: y[k] for k in ("range", "active_xyxy_incl", "band_px", "overlay_px", "excess_vs_full_px", "fp", "fn")}
                        for x, y in zip(wins[r], aw[r])) for r in RULES)
        else:
            chk["note"] = "data baru (tidak ada di T-602)"
        row["checks"] = chk
        res["clips"][clip] = row
        print(f"{clip} ({n} f, {full['R2']['frame_wh']}): R2 {full['R2']['active_xyxy_incl']} {full['R2']['active_wh']} "
              f"pita {full['R2']['band_px']} overlay {full['R2']['overlay_px']}; R1=R2 {row['r1_equals_r2_full']}; jendela R1 "
              f"{summ['R1']['fp']}/{summ['R1']['n']} R2 {summ['R2']['fp']}/{summ['R2']['n']} (kurang {summ['R2']['fn']}); "
              f"baca+stat {t_stats:.2f} s, deteksi {t_det * 1000:.1f} ms; cek {chk}", flush=True)
    res["windows_total"] = tot
    res["windows_total_t602_clips"] = tot602
    res["checks"]["windows_total_t602_clips_vs_t602_data"] = tot602 == t602["letterbox"]["jendela_total"]
    res["checks"]["all_true"] = res["checks"]["windows_total_t602_clips_vs_t602_data"] and all(
        v for c in res["clips"].values() for k, v in c["checks"].items() if k != "note")
    print(f"total jendela 5 klip T-602: {tot602} (t602_data {t602['letterbox']['jendela_total']}); 6 klip: {tot}")
    print(f"semua cek = {res['checks']['all_true']}; sidik R2 {r2_fingerprint()}")
    jsave(OUT / "r2_lama.json", res)


# ── ffprobe ────────────────────────────────────────
def ffprobe_json(src: Path) -> dict:
    p = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(src)],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0:
        raise SystemExit(f"ffprobe gagal untuk {src}: {p.stderr[-500:]}")
    return json.loads(p.stdout)


def probe_summary(src: Path) -> dict:
    info = ffprobe_json(src)
    v = next(s for s in info["streams"] if s.get("codec_type") == "video")
    side = [d for d in v.get("side_data_list") or [] if "rotation" in d]
    fmt = info.get("format") or {}
    return {"file": src.name, "bytes": src.stat().st_size, "codec": v.get("codec_name"), "pix_fmt": v.get("pix_fmt"),
            "coded_wh": [v.get("width"), v.get("height")], "sample_aspect_ratio": v.get("sample_aspect_ratio"),
            "display_aspect_ratio": v.get("display_aspect_ratio"),
            "rotation_side_data": side[0]["rotation"] if side else None, "rotate_tag": (v.get("tags") or {}).get("rotate"),
            "r_frame_rate": v.get("r_frame_rate"), "avg_frame_rate": v.get("avg_frame_rate"), "nb_frames": v.get("nb_frames"),
            "stream_start_time": v.get("start_time"), "stream_duration": v.get("duration"),
            "format_start_time": fmt.get("start_time"), "format_duration": fmt.get("duration"),
            "field_order": v.get("field_order"), "color_range": v.get("color_range"), "color_space": v.get("color_space"),
            "has_audio": any(s.get("codec_type") == "audio" for s in info["streams"]),
            "ingest_probe": ing.probe_video(src)}


def even_floor(v: int) -> int:
    return v - v % 2


def map_box_to_source(box: list[int], work_wh: tuple[int, int], src_wh: tuple[int, int], outward: bool = True) -> dict:
    """Kotak kerja inklusif (x0, y0, x1, y1) -> kotak sumber (x, y, w, h) genap. outward: floor awal / ceil akhir (konten tidak
    pernah terpotong; sisa pita ≤ pembulatan); ke dalam: ceil awal / floor akhir (pita tidak tersisa; konten bisa terpotong ≤ 1 px kerja)."""
    (Ww, Hw), (Ws, Hs) = work_wh, src_wh
    sx, sy = Ws / Ww, Hs / Hw
    x0, y0, x1, y1 = box
    if outward:
        xs0, ys0 = math.floor(x0 * sx), math.floor(y0 * sy)
        xs1, ys1 = math.ceil((x1 + 1) * sx), math.ceil((y1 + 1) * sy)
        xs0, ys0 = even_floor(xs0), even_floor(ys0)
        w, h = xs1 - xs0, ys1 - ys0
        w, h = min(w + w % 2, Ws - xs0), min(h + h % 2, Hs - ys0)
    else:
        xs0, ys0 = math.ceil(x0 * sx), math.ceil(y0 * sy)
        xs1, ys1 = math.floor((x1 + 1) * sx), math.floor((y1 + 1) * sy)
        xs0, ys0 = xs0 + xs0 % 2, ys0 + ys0 % 2
        w, h = even_floor(xs1 - xs0), even_floor(ys1 - ys0)
    return {"x": xs0, "y": ys0, "w": w, "h": h, "xyxy_incl": [xs0, ys0, xs0 + w - 1, ys0 + h - 1], "scale_src_per_work": [sx, sy]}


def cmd_probe() -> None:
    r2 = json.loads((OUT / "r2_lama.json").read_text(encoding="utf-8"))
    res = {"clips": {}}
    for clip in OLD_CLIPS:
        meta = json.loads((CLIPS_DIR / clip / "meta.json").read_text(encoding="utf-8"))
        src = Path(meta["source_path"])
        pr = probe_summary(src)
        work_wh = (meta["working_width"], meta["working_height"])
        src_wh = (meta["source_width"], meta["source_height"])
        full = r2["clips"][clip]["ranges"][f"f0-{meta['frame_count'] - 1}"]["R2"]
        box = full["active_xyxy_incl"]
        row = {"probe": pr, "meta": {k: meta[k] for k in ("source_width", "source_height", "source_rotation", "source_fps",
                                                         "working_width", "working_height", "frame_count")},
               "r2_box_work": box, "r2_wh_work": full["active_wh"], "cropped": full["active_wh"] != list(work_wh)}
        if row["cropped"]:
            out_b, in_b = map_box_to_source(box, work_wh, src_wh, True), map_box_to_source(box, work_wh, src_wh, False)
            p_src = ing.compute_output_size(out_b["w"], out_b["h"], WORKING_WIDTH)
            row.update({"source_box_outward": out_b, "source_box_inward": in_b,
                        "outward_minus_inward_px": {"left": in_b["x"] - out_b["x"], "top": in_b["y"] - out_b["y"],
                                                    "right": out_b["xyxy_incl"][2] - in_b["xyxy_incl"][2],
                                                    "bottom": out_b["xyxy_incl"][3] - in_b["xyxy_incl"][3]},
                        "active_width_source": out_b["w"], "working_width_cfg": WORKING_WIDTH,
                        "p_src_working_wh": list(p_src), "p_frm_working_wh": full["active_wh"],
                        "p_src_gain_linear": round(p_src[0] / full["active_wh"][0], 3),
                        "p_src_equals_p_frm_geometry": list(p_src) == full["active_wh"]})
        res["clips"][clip] = row
        print(f"{clip}: {pr['coded_wh']} SAR {pr['sample_aspect_ratio']} DAR {pr['display_aspect_ratio']} rot "
              f"{pr['rotation_side_data']}/{pr['rotate_tag']} fps {pr['r_frame_rate']}/{pr['avg_frame_rate']} dur "
              f"{pr['stream_duration']}/{pr['format_duration']} start {pr['stream_start_time']}/{pr['format_start_time']} "
              f"{pr['pix_fmt']} {pr['color_range']}; kerja {work_wh}; R2 {box}"
              + (f" -> sumber keluar {row['source_box_outward']['xyxy_incl']} ke dalam {row['source_box_inward']['xyxy_incl']}; "
                 f"P-src {row['p_src_working_wh']} vs P-frm {row['p_frm_working_wh']} (x{row['p_src_gain_linear']})"
                 if row["cropped"] else " (tanpa potongan)"), flush=True)
    jsave(OUT / "probe_lama.json", res)


# ── cropdetect (pembanding) ────────────────────────
def run_cropdetect(src: Path, prefix_vf: str) -> dict:
    cd = ":".join(f"{k}={v}" for k, v in CROPDETECT.items())
    vf = f"{prefix_vf},cropdetect={cd}" if prefix_vf else f"cropdetect={cd}"
    p = subprocess.run(["ffmpeg", "-hide_banner", "-nostdin", "-i", str(src), "-map", "0:v:0", "-an", "-vf", vf, "-f", "null", "-"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    hits = CROPDETECT_RE.findall(p.stderr)
    if p.returncode != 0 or not hits:
        return {"ok": False, "rc": p.returncode, "stderr_tail": p.stderr[-400:]}
    w, h, x, y = (int(v) for v in hits[-1])
    return {"ok": True, "vf": vf, "n_lines": len(hits), "last_crop_whxy": [w, h, x, y], "xyxy_incl": [x, y, x + w - 1, y + h - 1],
            "distinct": len(set(hits))}


def cmd_cropdetect() -> None:
    r2 = json.loads((OUT / "r2_lama.json").read_text(encoding="utf-8"))
    res = {"params": CROPDETECT, "catatan": "reset=0 -> kotak kumulatif (gabungan area tidak hitam) seluruh klip; skip default 2 frame; "
                                            "hanya mendeteksi pita GELAP (kotak teks terang tidak)", "clips": {}}
    for clip in OLD_CLIPS:
        meta = json.loads((CLIPS_DIR / clip / "meta.json").read_text(encoding="utf-8"))
        src = Path(meta["source_path"])
        Ww = meta["working_width"]
        work = run_cropdetect(src, f"fps={TARGET_FPS},scale={Ww}:-2:flags={ing.SCALE_FLAGS}")
        srcres = run_cropdetect(src, f"fps={TARGET_FPS}")
        box = r2["clips"][clip]["ranges"][f"f0-{meta['frame_count'] - 1}"]["R2"]["active_xyxy_incl"]
        row = {"r2_xyxy_incl_work": box, "cropdetect_work": work, "cropdetect_source": srcres}
        if work["ok"]:
            c = work["xyxy_incl"]
            row["diff_cd_minus_r2_work_px"] = {"left": c[0] - box[0], "top": c[1] - box[1], "right": box[2] - c[2], "bottom": box[3] - c[3]}
            row["same_as_r2"] = c == box
        res["clips"][clip] = row
        print(f"{clip}: R2 {box} | cropdetect kerja {work.get('xyxy_incl')} (beda {row.get('diff_cd_minus_r2_work_px')}) | "
              f"sumber {srcres.get('xyxy_incl')}", flush=True)
    jsave(OUT / "cropdetect_lama.json", res)


# ── ukur (scratch) ─────────────────────────────────
def ffmpeg_png(src: Path, vf: str, out_dir: Path) -> tuple[float, int]:
    """Perintah ingest apa adanya (ingest.py) dengan -vf lain -> PNG di out_dir. (detik, jumlah frame)."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-i", str(src), "-map", "0:v:0", "-an", "-vf", vf,
           "-fps_mode", "passthrough", "-pix_fmt", "rgb24", "-start_number", str(ing.FRAME_INDEX_START), str(out_dir / ing.FRAME_PATTERN)]
    t0 = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    dt = time.perf_counter() - t0
    if p.returncode != 0:
        raise SystemExit(f"ffmpeg gagal: {' '.join(cmd)}\n{p.stderr[-800:]}")
    return dt, len(list(out_dir.glob(ing.FRAME_GLOB)))


def pipe_stats(src: Path, vf: str, w: int, h: int) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Frame rgb24 lewat pipa (tanpa PNG) -> Stats (dikonversi ke BGR seperti cv2.imdecode)."""
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-i", str(src), "-map", "0:v:0", "-an", "-vf", vf,
           "-fps_mode", "passthrough", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    st, nbytes = Stats(), w * h * 3
    while True:
        buf = proc.stdout.read(nbytes)
        if len(buf) < nbytes:
            break
        st.add(np.frombuffer(buf, np.uint8).reshape(h, w, 3)[:, :, ::-1])
    proc.wait()
    m, sd = st.result()
    return m, sd, st.k, time.perf_counter() - t0


def compare_dirs(a: Path, b: Path, offset_b: int = 0, n: int | None = None) -> dict:
    """Bandingkan frame a[k] dengan b[k + offset_b]: byte identik, piksel identik, beda maks."""
    fa = sorted(a.glob(ing.FRAME_GLOB))[: n]
    res = {"n": len(fa), "bytes_equal": 0, "pixels_equal": 0, "max_abs_diff": 0, "mean_abs_diff_max": 0.0, "missing_b": 0}
    for i, p in enumerate(fa):
        q = b / (ing.FRAME_PATTERN % (i + offset_b))
        if not q.is_file():
            res["missing_b"] += 1
            continue
        if p.read_bytes() == q.read_bytes():
            res["bytes_equal"] += 1
            res["pixels_equal"] += 1
            continue
        x, y = read_png(p).astype(np.int16), read_png(q).astype(np.int16)
        if x.shape != y.shape:
            res.setdefault("shape_mismatch", []).append(i)
            continue
        d = np.abs(x - y)
        res["pixels_equal"] += int(d.max() == 0)
        res["max_abs_diff"] = max(res["max_abs_diff"], int(d.max()))
        res["mean_abs_diff_max"] = max(res["mean_abs_diff_max"], round(float(d.mean()), 4))
    return res


def first_pts(src: Path, vf: str, k: int = 3) -> list[str]:
    p = subprocess.run(["ffmpeg", "-hide_banner", "-nostdin", "-i", str(src), "-map", "0:v:0", "-an", "-vf", f"{vf},showinfo",
                        "-frames:v", str(k), "-f", "null", "-"], capture_output=True, text=True, encoding="utf-8", errors="replace")
    return [f"n{n} pts {pts} t {t}" for n, pts, t in SHOWINFO_PTS_RE.findall(p.stderr)][:k]


def cmd_ukur() -> None:
    r2 = json.loads((OUT / "r2_lama.json").read_text(encoding="utf-8"))
    probe = json.loads((OUT / "probe_lama.json").read_text(encoding="utf-8"))
    res = {"ffmpeg": subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout.splitlines()[0], "clips": {},
           "crop_pass": {}, "trim": {}}
    # (a) ingest biasa ke scratch (modul produksi) = frames/ + meta.json klip?  (b) deteksi PNG vs pipa
    for clip in OLD_CLIPS:
        meta_b = (CLIPS_DIR / clip / "meta.json").read_bytes()
        meta = json.loads(meta_b)
        src = Path(meta["source_path"])
        wd = SCR / "ingest" / clip
        if wd.exists():
            shutil.rmtree(wd)
        t0 = time.perf_counter()
        ing.ingest(src, wd, TARGET_FPS, WORKING_WIDTH)
        t_ing = time.perf_counter() - t0
        same_meta = (wd / "meta.json").read_bytes() == meta_b
        cmp = compare_dirs(wd / "frames", CLIPS_DIR / clip / "frames")
        n_clip = len(frame_paths(clip))
        paths = sorted((wd / "frames").glob(ing.FRAME_GLOB))
        t0 = time.perf_counter()
        m1, sd1 = stats_of_pngs(paths)
        det1 = detect_crop(m1, sd1)
        t_png = time.perf_counter() - t0
        W, H = meta["working_width"], meta["working_height"]
        vf = f"fps={TARGET_FPS},scale={W}:-2:flags={ing.SCALE_FLAGS}"
        m2, sd2, k2, t_pipe_read = pipe_stats(src, vf, W, H)
        t0 = time.perf_counter()
        det2 = detect_crop(m2, sd2)
        t_pipe = t_pipe_read + time.perf_counter() - t0
        box = r2["clips"][clip]["ranges"][f"f0-{n_clip - 1}"]["R2"]["active_xyxy_incl"]
        row = {"ingest_s": round(t_ing, 2), "meta_bytes_equal": same_meta, "frames_vs_clip": cmp, "n_clip": n_clip,
               "detect_png_s": round(t_png, 2), "detect_pipe_s": round(t_pipe, 2), "pipe_frames": k2,
               "pipe_stats_equal_png": bool(np.array_equal(m1, m2) and np.array_equal(sd1, sd2)),
               "box_png": det1["active_xyxy_incl"], "box_pipe": det2["active_xyxy_incl"], "box_equals_r2_lama": det1["active_xyxy_incl"] == box,
               "first_pts_after_fps": first_pts(src, f"fps={TARGET_FPS}")}
        res["clips"][clip] = row
        print(f"{clip}: ingest {t_ing:.2f} s, meta identik {same_meta}, frames byte identik {cmp['bytes_equal']}/{cmp['n']} "
              f"(klip {n_clip}); deteksi PNG {t_png:.2f} s / pipa {t_pipe:.2f} s, stat sama {row['pipe_stats_equal_png']}, "
              f"kotak {det1['active_xyxy_incl']} = lama {row['box_equals_r2_lama']}; pts awal {row['first_pts_after_fps']}", flush=True)
    # (c) ingest ulang ber-crop (P-src) untuk klip terpotong + pembanding P-frm
    for clip in ("Klip2", "klip5"):
        meta = json.loads((CLIPS_DIR / clip / "meta.json").read_text(encoding="utf-8"))
        src = Path(meta["source_path"])
        pr = probe["clips"][clip]
        out = {}
        for mode in ("outward", "inward"):
            b = pr[f"source_box_{mode}"]
            ow, oh = ing.compute_output_size(b["w"], b["h"], WORKING_WIDTH)
            vf = f"fps={TARGET_FPS},crop={b['w']}:{b['h']}:{b['x']}:{b['y']}:exact=1,scale={ow}:-2:flags={ing.SCALE_FLAGS}"
            d = SCR / "crop" / f"{clip}_{mode}"
            dt, n = ffmpeg_png(src, vf, d)
            with Image_open(d / (ing.FRAME_PATTERN % 0)) as wh:
                size = list(wh)
            out[mode] = {"vf": vf, "s": round(dt, 2), "frames": n, "size": size, "expected_size": [ow, oh]}
            # residu pita di tepi hasil (kolom / baris terluar): rata-rata kecerahan + std temporal median
            ms, sds = stats_of_pngs(sorted(d.glob(ing.FRAME_GLOB)))
            det = detect_crop(ms, sds)
            edge = {"left": ms[:, 0].mean(), "right": ms[:, -1].mean(), "top": ms[0].mean(), "bottom": ms[-1].mean()}
            out[mode]["r2_on_result"] = {k: det[k] for k in ("active_xyxy_incl", "band_px", "overlay_px")}
            out[mode]["edge_mean_bgr_avg"] = {k: round(float(v), 1) for k, v in edge.items()}
            out[mode]["edge_static_frac"] = {"left": round(float((sds[:, 0] <= 3).mean()), 3), "right": round(float((sds[:, -1] <= 3).mean()), 3),
                                             "top": round(float((sds[0] <= 3).mean()), 3), "bottom": round(float((sds[-1] <= 3).mean()), 3)}
        # P-frm: potong PNG frames/ klip dengan kotak kerja R2 -> bandingkan dengan P-src (hanya bila geometri sama)
        if pr["p_src_equals_p_frm_geometry"]:
            x0, y0, x1, y1 = pr["r2_box_work"]
            dfrm = SCR / "crop" / f"{clip}_pfrm"
            if dfrm.exists():
                shutil.rmtree(dfrm)
            dfrm.mkdir(parents=True)
            for p in frame_paths(clip):
                ok, buf = cv2.imencode(".png", read_png(p)[y0:y1 + 1, x0:x1 + 1])
                (dfrm / p.name).write_bytes(buf.tobytes())
            out["pfrm_vs_psrc_outward"] = compare_dirs(SCR / "crop" / f"{clip}_outward", dfrm)
        res["crop_pass"][clip] = out
        print(f"{clip}: P-src keluar {out['outward']['size']} {out['outward']['s']} s {out['outward']['frames']} f, R2 pada hasil "
              f"{out['outward']['r2_on_result']}, tepi {out['outward']['edge_mean_bgr_avg']}; ke dalam {out['inward']['size']} "
              f"R2 {out['inward']['r2_on_result']}; P-frm vs P-src {out.get('pfrm_vs_psrc_outward')}", flush=True)
    # (d) trim: sebelum fps (desain T-602 f) vs sesudah fps (grid frame ingest)
    cases = [("klip5", 0.0, 3.8333), ("klip5", 0.0, 92 / 24), ("klip5", 1.0, 3.0), ("klip3", 1.0, 3.0), ("test", 1.0, 3.0),
             ("Klip2", 0.5, 2.5)]
    for clip, s, e in cases:
        meta = json.loads((CLIPS_DIR / clip / "meta.json").read_text(encoding="utf-8"))
        src = Path(meta["source_path"])
        W = meta["working_width"]
        scale = f"scale={W}:-2:flags={ing.SCALE_FLAGS}"
        key = f"{clip}_{s:g}-{e:.6g}"
        row = {}
        for kind, vf in (("pre_fps", f"trim=start={s}:end={e},setpts=PTS-STARTPTS,fps={TARGET_FPS},{scale}"),
                         ("post_fps", f"fps={TARGET_FPS},trim=start={s}:end={e},{scale}")):
            d = SCR / "trim" / f"{key}_{kind}"
            dt, n = ffmpeg_png(src, vf, d)
            # pembanding: frame klip dengan offset = frame pertama yang waktunya >= s (dicari: offset terbaik 0..48)
            best = None
            for off in range(0, min(48, len(frame_paths(clip)))):
                c = compare_dirs(d, CLIPS_DIR / clip / "frames", off, n=3)
                if best is None or c["pixels_equal"] > best[1]["pixels_equal"]:
                    best = (off, c)
                if c["bytes_equal"] == c["n"]:
                    break
            off = best[0]
            full_cmp = compare_dirs(d, CLIPS_DIR / clip / "frames", off)
            row[kind] = {"vf": vf, "s": round(dt, 2), "frames": n, "best_offset": off, "vs_clip_frames": full_cmp,
                         "expected_frames_ceil": math.ceil(e * TARGET_FPS - 1e-9) - math.ceil(s * TARGET_FPS - 1e-9)}
        res["trim"][key] = row
        print(f"trim {key}: pre_fps {row['pre_fps']['frames']} f (offset {row['pre_fps']['best_offset']}, byte identik "
              f"{row['pre_fps']['vs_clip_frames']['bytes_equal']}/{row['pre_fps']['vs_clip_frames']['n']}, beda maks "
              f"{row['pre_fps']['vs_clip_frames']['max_abs_diff']}); post_fps {row['post_fps']['frames']} f (offset "
              f"{row['post_fps']['best_offset']}, byte identik {row['post_fps']['vs_clip_frames']['bytes_equal']}/"
              f"{row['post_fps']['vs_clip_frames']['n']})", flush=True)
    res["scratch_bytes"] = sum(p.stat().st_size for p in SCR.rglob("*") if p.is_file())
    jsave(OUT / "ukur.json", res)


class Image_open:  # noqa: N801 — ukuran PNG tanpa memuat piksel (PIL lazy)
    def __init__(self, p: Path):
        from PIL import Image
        self.img = Image.open(p)

    def __enter__(self):
        return self.img.size

    def __exit__(self, *a):
        self.img.close()


def cmd_trimframe() -> None:
    """trim berbasis indeks frame keluaran fps (start_frame / end_frame): frame = frames/ klip a..b-1? + trim + crop (klip5)."""
    res = {"trim_frame": {}, "trim_crop": {}}
    for clip, a, b in (("klip5", 0, 92), ("klip3", 24, 72), ("test", 24, 72), ("Klip2", 12, 60), ("klip4", 200, 217)):
        meta = json.loads((CLIPS_DIR / clip / "meta.json").read_text(encoding="utf-8"))
        src, W = Path(meta["source_path"]), meta["working_width"]
        vf = f"fps={TARGET_FPS},trim=start_frame={a}:end_frame={b},scale={W}:-2:flags={ing.SCALE_FLAGS}"
        d = SCR / "trimframe" / f"{clip}_{a}-{b}"
        dt, n = ffmpeg_png(src, vf, d)
        c = compare_dirs(d, CLIPS_DIR / clip / "frames", a)
        res["trim_frame"][f"{clip}_{a}-{b}"] = {"vf": vf, "s": round(dt, 2), "frames": n, "expected": b - a, "vs_clip_frames_offset_a": c}
        print(f"{clip} start_frame {a} end_frame {b}: {n} f (harap {b - a}), byte identik dengan frames/ f{a}.. {c['bytes_equal']}/{c['n']}", flush=True)
    probe = json.loads((OUT / "probe_lama.json").read_text(encoding="utf-8"))
    bx = probe["clips"]["klip5"]["source_box_outward"]
    src = Path(json.loads((CLIPS_DIR / "klip5" / "meta.json").read_text(encoding="utf-8"))["source_path"])
    ow, _ = ing.compute_output_size(bx["w"], bx["h"], WORKING_WIDTH)
    for a, b in ((0, 92), (40, 92)):
        vf = (f"fps={TARGET_FPS},trim=start_frame={a}:end_frame={b},crop={bx['w']}:{bx['h']}:{bx['x']}:{bx['y']}:exact=1,"
              f"scale={ow}:-2:flags={ing.SCALE_FLAGS}")
        d = SCR / "trimcrop" / f"klip5_{a}-{b}"
        dt, n = ffmpeg_png(src, vf, d)
        c = compare_dirs(d, SCR / "crop" / "klip5_outward", a)
        res["trim_crop"][f"klip5_{a}-{b}"] = {"vf": vf, "s": round(dt, 2), "frames": n, "vs_crop_full_offset_a": c}
        print(f"klip5 trim {a}-{b} + crop: {n} f, byte identik dengan crop rentang penuh f{a}.. {c['bytes_equal']}/{c['n']}", flush=True)
    jsave(OUT / "trimframe.json", res)


# ── klip validasi: HANYA ffprobe (sebelum label) ───
def new_source(stem: str) -> Path:
    found = sorted(p for p in SAMPLES.glob(f"{stem}.*") if p.is_file())
    if len(found) != 1:
        raise SystemExit(f"samples/{stem}.*: {len(found)} berkas (harus tepat 1)")
    return found[0]


def cmd_probe_baru() -> None:
    probe_clips(NEW_CLIPS, "probe_baru.json")


def cmd_probe_k2() -> None:
    probe_clips(K2_CLIPS, "probe_k2.json")


def probe_clips(stems: tuple[str, ...], out_name: str) -> None:
    res = {}
    for stem in stems:
        pr = probe_summary(new_source(stem))
        res[stem] = {k: pr[k] for k in ("file", "bytes", "codec", "pix_fmt", "coded_wh", "sample_aspect_ratio",
                                        "display_aspect_ratio", "rotation_side_data", "rotate_tag", "r_frame_rate",
                                        "avg_frame_rate", "nb_frames", "stream_start_time", "stream_duration", "format_duration",
                                        "has_audio", "ingest_probe")}
        print(f"{stem}: {res[stem]}", flush=True)
    jsave(OUT / out_name, res)


# ── label Rio (Tahap 1b langkah 1; SEBELUM deteksi) ─
LABEL_KEYS = ("pita_atas", "pita_bawah", "pita_kiri", "pita_kanan", "kotak_teks", "kotak_teks_posisi", "tata_letak_berubah",
              "tata_letak_berubah_detik", "catatan")
YA_TIDAK = {"ya": True, "tidak": False}
POSISI = ("atas", "bawah", "kiri", "kanan", "tengah")
KOSONG = ("-", "tidak", "")                    # isian "tidak ada" pada posisi / detik (template: "-")
MD_ESCAPE_RE = re.compile(r"\\([\\`*_{}\[\]()#+\-.!|])")   # escape Markdown yang ditambahkan editor (mis. `pita\_atas`)


def parse_label_md(text: str, keys: tuple[str, ...] = LABEL_KEYS) -> tuple[dict[str, dict[str, str]], int]:
    """LABEL.md / PENILAIAN.md -> {bagian: {kunci: nilai mentah}} (hanya `keys`) + jumlah baris yang escape-nya dibuang."""
    secs: dict[str, dict[str, str]] = {}
    cur, n_esc = None, 0
    for raw in text.splitlines():
        line = MD_ESCAPE_RE.sub(r"\1", raw).strip()
        n_esc += line != raw.strip()
        if line.startswith("## "):
            cur = line[3:].strip()
            secs.setdefault(cur, {})
            continue
        if cur is None or ":" not in line:
            continue
        k, v = (s.strip() for s in line.split(":", 1))
        if k in keys:
            if k in secs[cur]:
                raise SystemExit(f"bagian {cur} memuat {k} dua kali")
            secs[cur][k] = v
    return secs, n_esc


def normalize_label(clip: str, raw: dict[str, str], problems: list[str]) -> dict:
    for k in LABEL_KEYS:
        if k not in raw:
            problems.append(f"{clip}: {k} tidak ada")
        elif k != "catatan" and raw[k] in ("", "?"):
            problems.append(f"{clip}: {k} kosong / '?'")

    def yn(k: str) -> bool | None:
        v = raw.get(k, "").lower()
        if v not in YA_TIDAK:
            problems.append(f"{clip}: {k} = {raw.get(k)!r} (harus ya / tidak)")
            return None
        return YA_TIDAK[v]

    pita = {e: yn(f"pita_{e}") for e in ("atas", "bawah", "kiri", "kanan")}
    kotak = yn("kotak_teks")
    pos_raw = raw.get("kotak_teks_posisi", "").lower()
    pos = [p for p in re.split(r"[,\s]+", pos_raw) if p] if kotak else []    # koma ATAU spasi (mis. "kanan atas")
    if kotak and (not pos or any(p not in POSISI for p in pos)):
        problems.append(f"{clip}: kotak_teks ya tetapi posisi {raw.get('kotak_teks_posisi')!r} (harus {', '.join(POSISI)})")
    if kotak is False and pos_raw not in KOSONG:
        problems.append(f"{clip}: kotak_teks tidak tetapi posisi {raw.get('kotak_teks_posisi')!r}")
    berubah = yn("tata_letak_berubah")
    det_raw = raw.get("tata_letak_berubah_detik", "")
    if berubah and det_raw.lower() in KOSONG:
        problems.append(f"{clip}: tata_letak_berubah ya tetapi detik kosong")
    if berubah is False and det_raw.lower() not in KOSONG:
        problems.append(f"{clip}: tata_letak_berubah tidak tetapi detik {det_raw!r}")
    known = None not in pita.values()
    return {"mentah": {k: raw.get(k) for k in LABEL_KEYS},
            "pita": pita, "berpita": any(pita.values()) if known else None,
            "pita_berpasangan": ((pita["atas"] and pita["bawah"]) or (pita["kiri"] and pita["kanan"])) if known else None,
            "kotak_teks": kotak, "kotak_teks_posisi": pos,
            "tata_letak_berubah": berubah, "tata_letak_berubah_detik_mentah": det_raw if berubah else None,
            "catatan": raw.get("catatan", "")}


def cmd_label() -> None:
    """LABEL.md -> scripts/t607_labels.json. Isian kosong / tidak konsisten -> exit 1 TANPA menulis (tanya Rio)."""
    if r2_fingerprint() != R2_FROZEN_FINGERPRINT:
        raise SystemExit(f"sidik R2 {r2_fingerprint()} != beku {R2_FROZEN_FINGERPRINT}")
    data = LABEL_MD.read_bytes()
    secs, n_esc = parse_label_md(data.decode("utf-8"))
    problems: list[str] = []
    extra = sorted(s for s, kv in secs.items() if kv and s not in NEW_CLIPS)
    if extra:
        problems.append(f"bagian berlabel di luar {NEW_CLIPS}: {extra}")
    clips = {}
    for c in NEW_CLIPS:
        if c not in secs:
            problems.append(f"bagian ## {c} tidak ada")
            continue
        clips[c] = normalize_label(c, secs[c], problems)
    if problems:
        print("LABEL.md belum bisa disimpan:\n  " + "\n  ".join(problems))
        raise SystemExit(1)
    notes = []
    if n_esc:
        notes.append(f"escape Markdown dari editor dibuang di {n_esc} baris (mis. 'pita\\_atas' -> 'pita_atas'); isi tidak berubah")
    for c, v in clips.items():
        for k in ("kotak_teks_posisi", "tata_letak_berubah_detik"):
            if v["mentah"][k].lower() == "tidak":
                notes.append(f"{c}: {k} 'tidak' dibaca sebagai kosong (template: '-')")
    res = {"format": "t607-labels/1", "sumber": LABEL_MD.relative_to(ROOT).as_posix(), "label_md_sha256": hashlib.sha256(data).hexdigest(),
           "label_md_bytes": len(data), "r2_fingerprint": R2_FROZEN_FINGERPRINT,
           "disimpan_sebelum_deteksi": True, "normalisasi": notes, "clips": clips,
           "ringkasan": {"berpita": [c for c, v in clips.items() if v["berpita"]],
                         "tanpa_pita": [c for c, v in clips.items() if not v["berpita"]],
                         "kotak_teks": {c: v["kotak_teks_posisi"] for c, v in clips.items() if v["kotak_teks"]},
                         "tata_letak_berubah": [c for c, v in clips.items() if v["tata_letak_berubah"]]}}
    for c, v in clips.items():
        print(f"{c}: pita {v['pita']} (berpita {v['berpita']}, berpasangan {v['pita_berpasangan']}); kotak teks {v['kotak_teks']} "
              f"{v['kotak_teks_posisi']}; tata letak berubah {v['tata_letak_berubah']}; catatan {v['catatan']!r}")
    print(f"ringkasan {res['ringkasan']}\nnormalisasi {notes}\nLABEL.md sha256 {res['label_md_sha256']}")
    jsave(LABELS_JSON, res)
    print(f"  t607_labels.json sha256 {file_sha(LABELS_JSON)}")


# ── label K2 klip10–klip11 (keputusan sesudah K3 no. 4; SEBELUM deteksi) ─
LABELS_K2_JSON = ROOT / "scripts" / "t607_labels_k2.json"           # berkas BARU; t607_labels.json tidak diubah
K2_KEYS = LABEL_KEYS + ("jenis_pita",)
JENIS_PITA = ("hitam polos", "putih polos", "buram bergerak")
# Isian di luar pilihan / letak kotak teks -> ditanyakan sekali SEBELUM deteksi (AskUserQuestion, 2026-10-20); jawaban Rio apa adanya.
KLARIFIKASI_K2 = {
    "klip10": {"jenis_pita": "hitam polos", "kotak_teks_letak": "di sebelah pita bagian kanan",
               "harapan_k2": "pita kiri + kanan terbuang; kotak teks (kanan atas, di sebelah pita kanan) ikut terbuang (K2); konten utuh",
               "catatan": "jawaban letak tidak memilih opsi; tafsir Claude Code: menempel pada pita kanan (bukan caption di atas adegan) "
                          "-> K2 'kotak teks terbuang'; tidak ditanyakan ulang (sudah sekali); dinilai Rio di papan"},
    "klip11": {"jenis_pita": "hitam polos", "kotak_teks_letak": "di pita atas / antara pita dan adegan",
               "harapan_k2": "pita atas + bawah + kotak teks atas terbuang (K2); konten utuh", "catatan": ""},
}


def cmd_label_k2() -> None:
    """Bagian klip10–klip11 LABEL.md + klarifikasi -> scripts/t607_labels_k2.json. Isian salah -> exit 1 tanpa menulis."""
    require_blind_labels()                     # sidik R2 + label klip6–klip9 tidak berubah
    if LABELS_K2_JSON.exists():
        raise SystemExit(f"{LABELS_K2_JSON.name} sudah ada - tidak ditimpa (label K2 disimpan sekali)")
    data = LABEL_MD.read_bytes()
    secs, n_esc = parse_label_md(data.decode("utf-8"), K2_KEYS)
    problems: list[str] = []
    clips = {}
    for c in K2_CLIPS:
        if c not in secs:
            problems.append(f"bagian ## {c} tidak ada")
            continue
        raw = secs[c]
        v = normalize_label(c, raw, problems)
        jp_raw = raw.get("jenis_pita", "")
        if jp_raw in ("", "?"):
            problems.append(f"{c}: jenis_pita kosong / '?'")
        if not v["pita_berpasangan"]:
            problems.append(f"{c}: menurut label TIDAK berpita berpasangan ({v['pita']}) -> K2 tidak bisa diuji (lapor + berhenti)")
        k = KLARIFIKASI_K2[c]
        v["mentah"]["jenis_pita"] = jp_raw
        v.update({"jenis_pita_mentah": jp_raw, "jenis_pita": k["jenis_pita"] if jp_raw not in JENIS_PITA else jp_raw,
                  "kotak_teks_letak": k["kotak_teks_letak"], "harapan_k2": k["harapan_k2"], "klarifikasi_catatan": k["catatan"]})
        clips[c] = v
    if problems:
        print("label K2 belum bisa disimpan:\n  " + "\n  ".join(problems))
        raise SystemExit(1)
    notes = [f"escape Markdown dari editor dibuang di {n_esc} baris"] if n_esc else []
    for c, v in clips.items():
        if v["jenis_pita_mentah"] not in JENIS_PITA:
            notes.append(f"{c}: jenis_pita {v['jenis_pita_mentah']!r} -> {v['jenis_pita']!r} (klarifikasi Rio)")
        if "," not in v["mentah"]["kotak_teks_posisi"] and len(v["kotak_teks_posisi"]) > 1:
            notes.append(f"{c}: kotak_teks_posisi {v['mentah']['kotak_teks_posisi']!r} dipisah spasi -> {v['kotak_teks_posisi']}")
        for key in ("kotak_teks_posisi", "tata_letak_berubah_detik"):
            if v["mentah"][key].lower() == "tidak":
                notes.append(f"{c}: {key} 'tidak' dibaca sebagai kosong (template: '-')")
    res = {"format": "t607-labels-k2/1", "sumber": LABEL_MD.relative_to(ROOT).as_posix(),
           "label_md_sha256": hashlib.sha256(data).hexdigest(), "label_md_bytes": len(data),
           "labels_klip6_9_sha256": LABELS_JSON_SHA256, "r2_fingerprint": R2_FROZEN_FINGERPRINT, "disimpan_sebelum_deteksi": True,
           "klarifikasi_sebelum_deteksi": KLARIFIKASI_K2, "normalisasi": notes, "clips": clips,
           "ringkasan": {"berpita_berpasangan": [c for c, v in clips.items() if v["pita_berpasangan"]],
                         "sisi_pita": {c: [e for e, y in v["pita"].items() if y] for c, v in clips.items()},
                         "kotak_teks": {c: v["kotak_teks_posisi"] for c, v in clips.items() if v["kotak_teks"]}}}
    for c, v in clips.items():
        print(f"{c}: pita {v['pita']} (berpasangan {v['pita_berpasangan']}), jenis {v['jenis_pita']!r}; kotak teks {v['kotak_teks']} "
              f"{v['kotak_teks_posisi']} ({v['kotak_teks_letak']}); tata letak berubah {v['tata_letak_berubah']}; harapan: {v['harapan_k2']}")
    print(f"normalisasi {notes}\nLABEL.md sha256 {res['label_md_sha256']}")
    jsave(LABELS_K2_JSON, res)
    print(f"  t607_labels_k2.json sha256 {file_sha(LABELS_K2_JSON)}")


# ── penilaian Rio atas papan 1b (SESUDAH deteksi; tidak buta) ─
PENILAIAN_MD = NILAI_DIR / "PENILAIAN.md"
PENILAIAN_JSON = ROOT / "scripts" / "t607_penilaian.json"           # ter-track; label asli (t607_labels.json) TIDAK diubah
PENILAIAN_KEYS = ("papan", "penilaian", "komentar", "pita_hitam_tipis_kiri_kanan")
PENILAIAN_OPTS = ("pas", "konten terpotong", "sisa pita", "berpita tetapi tidak dipotong")
PITA_TIPIS_SEC = "klip6 — pita tipis"
# Jawaban tidak konsisten -> ditanyakan sekali (AskUserQuestion, 2026-10-20): papan 02 "pas" padahal pita = "tidak" (tepi pintu hitam).
KLARIFIKASI = {"02": {"penilaian": "konten terpotong", "komentar": "tidak mengganggu",
                      "alasan": "Rio: pita_hitam_tipis_kiri_kanan = tidak ('tepi pintu berwarna hitam'); papan 02 membuang tepi itu -> "
                                "definisi 'konten terpotong'; ditanyakan sekali, Rio memilih 'Konten terpotong, tidak mengganggu'"}}


def cmd_penilaian() -> None:
    """PENILAIAN.md + klarifikasi Rio -> scripts/t607_penilaian.json (jawaban mentah DAN final; papan dicek sha256 = papan.json)."""
    data = PENILAIAN_MD.read_bytes()
    secs, n_esc = parse_label_md(data.decode("utf-8"), PENILAIAN_KEYS)
    boards = {b["file"]: b for b in json.loads((NILAI_DIR / "papan.json").read_text(encoding="utf-8"))}
    res = json.loads((OUT / "baru.json").read_text(encoding="utf-8"))
    problems, out = [], {}
    for b in boards.values():
        sec = b["file"][:2]
        raw = secs.get(sec)
        if raw is None or raw.get("penilaian", "?") in ("", "?"):
            problems.append(f"papan {sec}: penilaian kosong")
            continue
        if not raw.get("papan", "").startswith(b["file"]):
            problems.append(f"papan {sec}: baris papan {raw.get('papan')!r} != {b['file']}")
        if raw["penilaian"] not in PENILAIAN_OPTS:
            problems.append(f"papan {sec}: penilaian {raw['penilaian']!r} bukan salah satu {PENILAIAN_OPTS}")
        png_same = file_sha(NILAI_DIR / b["file"]) == b["sha256"]
        if not png_same:
            problems.append(f"papan {sec}: {b['file']} berubah sesudah papan.json ditulis")
        k = KLARIFIKASI.get(sec)
        row = res["clips"][b["clip"]]
        out[sec] = {"papan": b["file"], "clip": b["clip"], "frame": b["frame"], "jenis": b["kind"], "alat_memotong": b["cropped"],
                    "kotak": b["box"], "penilaian_mentah": raw["penilaian"], "komentar_mentah": raw.get("komentar", ""),
                    "penilaian": k["penilaian"] if k else raw["penilaian"], "komentar": k["komentar"] if k else raw.get("komentar", ""),
                    "klarifikasi": k["alasan"] if k else None, "png_sha256_sama_papan_json": png_same,
                    "r2_rentang_penuh": row["full"]["R2"]["active_xyxy_incl"],
                    "cropdetect_kerja": row["cropdetect_work"].get("xyxy_incl")}
    pita = secs.get(PITA_TIPIS_SEC, {})
    if pita.get("pita_hitam_tipis_kiri_kanan", "").lower() not in YA_TIDAK:
        problems.append(f"{PITA_TIPIS_SEC}: jawaban {pita.get('pita_hitam_tipis_kiri_kanan')!r} (harus ya / tidak)")
    if problems:
        print("PENILAIAN.md belum bisa disimpan:\n  " + "\n  ".join(problems))
        raise SystemExit(1)
    doc = {"format": "t607-penilaian/1", "sumber": PENILAIAN_MD.relative_to(ROOT).as_posix(),
           "penilaian_md_sha256": hashlib.sha256(data).hexdigest(), "penilaian_md_bytes": len(data),
           "labels_sha256": LABELS_JSON_SHA256, "r2_fingerprint": R2_FROZEN_FINGERPRINT, "buta": False,
           "catatan": "SESUDAH deteksi (tidak buta); label asli scripts/t607_labels.json tidak diubah",
           "normalisasi": [f"escape Markdown dari editor dibuang di {n_esc} baris"] if n_esc else [],
           "papan": out,
           "klip6_pita_tipis": {"jawaban": YA_TIDAK[pita["pita_hitam_tipis_kiri_kanan"].lower()], "komentar": pita.get("komentar", ""),
                                "label_asli_berpita": False, "koreksi_label": False}}
    for sec, v in out.items():
        print(f"{sec} {v['clip']} f{v['frame']} ({v['jenis']}): alat memotong {v['alat_memotong']} {v['kotak']} -> Rio "
              f"{v['penilaian_mentah']!r}" + (f" -> final {v['penilaian']!r} ({v['komentar']})" if v["klarifikasi"] else ""))
    print(f"klip6 pita tipis: {doc['klip6_pita_tipis']}\nPENILAIAN.md sha256 {doc['penilaian_md_sha256']}")
    jsave(PENILAIAN_JSON, doc)
    print(f"  t607_penilaian.json sha256 {file_sha(PENILAIAN_JSON)}")


# ── penilaian Rio K2 (klip10, klip11, papan 12; SESUDAH deteksi) ─
PENILAIAN_K2_JSON = ROOT / "scripts" / "t607_penilaian_k2.json"    # berkas BARU; t607_penilaian.json tidak diubah
PENILAIAN_K2_KEYS = ("papan", "penilaian", "kotak_teks", "komentar")
KOTAK_TEKS_OPTS = ("terbuang", "masih ada", "sebagian")
# Jawaban vs angka tidak konsisten -> ditanyakan sekali (AskUserQuestion, 2026-10-20); jawaban Rio apa adanya.
KLARIFIKASI_PENILAIAN_K2 = {
    "klip10": {"kotak_teks": "masih ada",
               "alasan": "angka: pita kiri + kanan hitam murni di semua frame (tidak ada piksel > 16) -> kotak teks kanan atas tidak di pita; "
                         "Rio: 'Masih ada di hasil' (di dalam area video; di luar lingkup K2 menurut 'Klarifikasi lingkup K2')"},
    "klip11": {"kotak_teks": "sebagian",
               "alasan": "angka: hasil rentang penuh menyisakan strip diam 74 baris (388–461; 390–461 putih 243–253, sama di semua frame; "
                         "adegan mulai baris 462); papan 12 membuangnya. Rio: strip = 'Bagian kotak teks' -> sisa kotak teks > 4 px"},
}


def cmd_penilaian_k2() -> None:
    """Bagian K2 PENILAIAN.md + klarifikasi -> scripts/t607_penilaian_k2.json + hasil K2 menurut aturan hasil Rio (docs/04
    'Klarifikasi lingkup K2'). Isian salah -> exit 1 tanpa menulis."""
    require_blind_labels_k2()
    if PENILAIAN_K2_JSON.exists():
        raise SystemExit(f"{PENILAIAN_K2_JSON.name} sudah ada - tidak ditimpa")
    data = PENILAIAN_MD.read_bytes()
    secs, n_esc = parse_label_md(data.decode("utf-8"), PENILAIAN_K2_KEYS)
    boards = json.loads((NILAI_DIR / "papan_k2.json").read_text(encoding="utf-8"))
    res = json.loads((OUT / "baru_k2.json").read_text(encoding="utf-8"))
    problems, out = [], {}
    for sec in K2_CLIPS + ("12",):
        raw = secs.get(sec)
        if raw is None or raw.get("penilaian", "?") in ("", "?"):
            problems.append(f"{sec}: penilaian kosong")
            continue
        if raw["penilaian"] not in PENILAIAN_OPTS:
            problems.append(f"{sec}: penilaian {raw['penilaian']!r} bukan salah satu {PENILAIAN_OPTS}")
        if sec in K2_CLIPS and raw.get("kotak_teks") not in KOTAK_TEKS_OPTS:
            problems.append(f"{sec}: kotak_teks {raw.get('kotak_teks')!r} bukan salah satu {KOTAK_TEKS_OPTS}")
        files = [b for b in boards if (b["clip"] == sec) or (sec == "12" and b["file"].startswith("12_"))]
        if sec in K2_CLIPS:
            files = [b for b in files if b["kind"].startswith("rentang penuh")]
        bad_png = [b["file"] for b in files if file_sha(NILAI_DIR / b["file"]) != b["sha256"]]
        if not files or bad_png:
            problems.append(f"{sec}: papan {bad_png or 'tidak ada'} berubah / hilang")
        k = KLARIFIKASI_PENILAIAN_K2.get(sec, {})
        clip = files[0]["clip"] if files else sec
        out[sec] = {"papan": [b["file"] for b in files], "clip": clip, "kotak": files[0]["box"] if files else None,
                    "penilaian_mentah": raw["penilaian"], "kotak_teks_mentah": raw.get("kotak_teks"),
                    "komentar_mentah": raw.get("komentar", ""), "penilaian": raw["penilaian"],
                    "kotak_teks": k.get("kotak_teks", raw.get("kotak_teks")), "klarifikasi": k.get("alasan"),
                    "r2_rentang_penuh": res["clips"][clip]["full"]["R2"]["active_xyxy_incl"]}
    if problems:
        print("PENILAIAN.md (K2) belum bisa disimpan:\n  " + "\n  ".join(problems))
        raise SystemExit(1)
    # aturan hasil Rio (sebelum penilaian): lolos = pita terbuang + konten utuh (penilaian pas) + kotak teks DALAM LINGKUP terbuang;
    # klip10: kotak teks sudut (tidak selebar area aktif) di luar lingkup = pengamatan; klip11: kotak teks selebar area aktif = dalam lingkup
    in_scope = {"klip10": False, "klip11": True}
    hasil = {c: bool(out[c]["penilaian"] == "pas" and (not in_scope[c] or out[c]["kotak_teks"] == "terbuang")) for c in K2_CLIPS}
    doc = {"format": "t607-penilaian-k2/1", "sumber": PENILAIAN_MD.relative_to(ROOT).as_posix(),
           "penilaian_md_sha256": hashlib.sha256(data).hexdigest(), "penilaian_md_bytes": len(data),
           "labels_k2_sha256": LABELS_K2_SHA256, "r2_fingerprint": R2_FROZEN_FINGERPRINT, "buta": False,
           "catatan": "SESUDAH deteksi (tidak buta); t607_penilaian.json + berkas label tidak diubah",
           "normalisasi": [f"escape Markdown dari editor dibuang di {n_esc} baris"] if n_esc else [],
           "penilaian": out, "kotak_teks_dalam_lingkup": in_scope, "k2_per_klip_lolos": hasil, "k2_lolos": all(hasil.values()),
           "papan_12": "pas -> strip 74 baris = kotak teks; jendela benar, rentang penuh potong kurang (bukan potong berlebih nyata)"}
    for sec, v in out.items():
        print(f"{sec}: penilaian {v['penilaian_mentah']!r}, kotak_teks {v['kotak_teks_mentah']!r}"
              + (f" -> {v['kotak_teks']!r}" if v["klarifikasi"] else ""))
    print(f"K2 per klip {hasil} -> K2 lolos {doc['k2_lolos']}\nPENILAIAN.md sha256 {doc['penilaian_md_sha256']}")
    jsave(PENILAIAN_K2_JSON, doc)
    print(f"  t607_penilaian_k2.json sha256 {file_sha(PENILAIAN_K2_JSON)}")


# ── klip baru: R2 beku + cropdetect (Tahap 1b langkah 2–3) ─
def require_blind_labels() -> dict:
    """Papan buta: deteksi klip baru hanya bila R2 = beku DAN label tersimpan = sha256 yang dicatat di docs/04 DAN LABEL.md tidak
    berubah sesudahnya."""
    if r2_fingerprint() != R2_FROZEN_FINGERPRINT:
        raise SystemExit(f"sidik R2 {r2_fingerprint()} != beku {R2_FROZEN_FINGERPRINT}")
    if not LABELS_JSON.is_file() or file_sha(LABELS_JSON) != LABELS_JSON_SHA256:
        raise SystemExit(f"{LABELS_JSON.name} tidak ada / berbeda dari sha256 yang dicatat di docs/04 - simpan label dulu")
    labels = json.loads(LABELS_JSON.read_text(encoding="utf-8"))
    # LABEL.md ditambah bagian klip10–klip11 (keputusan sesudah K3 no. 4) -> sha256 berkas berubah; catatan ter-track = t607_labels.json
    # (sha di atas). Bagian klip6–klip9 di LABEL.md harus tetap memberi nilai mentah yang SAMA (label lama tidak disunting).
    secs, _ = parse_label_md(LABEL_MD.read_text(encoding="utf-8"))
    edited = [c for c in NEW_CLIPS if {k: secs.get(c, {}).get(k) for k in LABEL_KEYS} != labels["clips"][c]["mentah"]]
    if edited:
        raise SystemExit(f"label {edited} di LABEL.md berubah sesudah disimpan")
    return labels


def window_summary(ws: list[dict]) -> dict:
    return {"n": len(ws), "fp": sum(w["fp"] for w in ws), "fn": sum(w["fn"] for w in ws),
            "fp_ranges": [w["range"] for w in ws if w["fp"]], "fn_ranges": [w["range"] for w in ws if w["fn"]],
            "max_excess_px": {e: max(w["excess_vs_full_px"][e] for w in ws) for e in EDGES},
            "min_excess_px": {e: min(w["excess_vs_full_px"][e] for w in ws) for e in EDGES}}


LABELS_K2_SHA256 = "2f0e5156817af80bc13dece13e85a81cd8569539e385830faba77eb774869ef3"   # docs/04 "Label Rio K2 klip10–klip11"
SISI = {"atas": "top", "bawah": "bottom", "kiri": "left", "kanan": "right"}


def require_blind_labels_k2() -> dict:
    """Seperti require_blind_labels + t607_labels_k2.json = sha256 yang dicatat + bagian klip10–klip11 LABEL.md tidak disunting."""
    require_blind_labels()
    if not LABELS_K2_JSON.is_file() or file_sha(LABELS_K2_JSON) != LABELS_K2_SHA256:
        raise SystemExit(f"{LABELS_K2_JSON.name} tidak ada / berbeda dari sha256 yang dicatat di docs/04 - simpan label K2 dulu")
    labels = json.loads(LABELS_K2_JSON.read_text(encoding="utf-8"))
    secs, _ = parse_label_md(LABEL_MD.read_text(encoding="utf-8"), K2_KEYS)
    edited = [c for c in K2_CLIPS if {k: secs.get(c, {}).get(k) for k in K2_KEYS} != labels["clips"][c]["mentah"]]
    if edited:
        raise SystemExit(f"label {edited} di LABEL.md berubah sesudah disimpan")
    return labels


def cut_px(box: list[int], W: int, H: int) -> dict[str, int]:
    """Kotak inklusif -> px yang dibuang per sisi (top / bottom / left / right)."""
    x0, y0, x1, y1 = box
    return {"top": y0, "bottom": H - 1 - y1, "left": x0, "right": W - 1 - x1}


def cmd_baru() -> None:
    detect_clips(NEW_CLIPS, require_blind_labels()["clips"], LABELS_JSON_SHA256, "baru.json")


def cmd_baru_k2() -> None:
    detect_clips(K2_CLIPS, require_blind_labels_k2()["clips"], LABELS_K2_SHA256, "baru_k2.json")


def detect_clips(clips: tuple[str, ...], label_of: dict, labels_sha: str, out_name: str) -> None:
    res = {"r2": {"rule": R2_RULE, "version": R2_VERSION, "params": R2_PARAMS, "fingerprint": r2_fingerprint()},
           "window": {"len": WIN_LEN, "step": WIN_STEP, "tol_px": WIN_TOL_PX}, "labels_sha256": labels_sha,
           "ffmpeg": subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout.splitlines()[0],
           "cropdetect_params": CROPDETECT, "clips": {}}
    tot = {r: {"n": 0, "fp": 0, "fn": 0} for r in RULES}
    tot_frame = {"n": 0, "fp": 0}
    for clip in clips:
        src = new_source(clip)
        wd = BARU_DIR / clip
        if wd.exists():
            shutil.rmtree(wd)
        t0 = time.perf_counter()
        ing.ingest(src, wd, TARGET_FPS, WORKING_WIDTH)
        t_ing = time.perf_counter() - t0
        meta = json.loads((wd / "meta.json").read_text(encoding="utf-8"))
        paths = sorted((wd / "frames").glob(ing.FRAME_GLOB))
        n = len(paths)
        if n != meta["frame_count"]:
            raise SystemExit(f"{clip}: {n} PNG != meta frame_count {meta['frame_count']}")
        t0 = time.perf_counter()
        m, sd = stats_of_pngs(paths)
        full = {r: detect_crop(m, sd, r) for r in RULES}
        t_full = time.perf_counter() - t0
        W, H = full["R2"]["frame_wh"]
        frame_box = [0, 0, W - 1, H - 1]
        ref = full["R2"]["active_xyxy_incl"]
        wins = {r: [] for r in RULES}
        t0 = time.perf_counter()
        for a, b in windows_of(n):
            ms, sds = stats_of_pngs(paths[a:b + 1])
            for r in RULES:
                det = detect_crop(ms, sds, r)
                w = {"range": [a, b], **window_eval(det, ref)}
                if r == "R2":                    # kebenaran label (tanpa pita) = frame penuh
                    vf = window_eval(det, frame_box)
                    w["excess_vs_frame_px"], w["fp_vs_frame"] = vf["excess_vs_full_px"], vf["fp"]
                wins[r].append(w)
        t_win = time.perf_counter() - t0
        summ = {r: window_summary(wins[r]) for r in RULES}
        summ["R2"]["fp_vs_frame"] = sum(w["fp_vs_frame"] for w in wins["R2"])
        summ["R2"]["fp_vs_frame_ranges"] = [w["range"] for w in wins["R2"] if w["fp_vs_frame"]]
        for r in RULES:
            for k in ("n", "fp", "fn"):
                tot[r][k] += summ[r][k]
        tot_frame["n"] += summ["R2"]["n"]
        tot_frame["fp"] += summ["R2"]["fp_vs_frame"]
        lab = label_of[clip]
        cropped = ref != frame_box
        cd_work = run_cropdetect(src, f"fps={TARGET_FPS},scale={meta['working_width']}:-2:flags={ing.SCALE_FLAGS}")
        cd_src = run_cropdetect(src, f"fps={TARGET_FPS}")
        r2_cut = cut_px(ref, W, H)
        label_sides = sorted(SISI[s] for s, y in lab["pita"].items() if y)
        k2 = None
        if lab["berpita"]:                       # angka K2 (kotak vs label); K2 final = penilaian Rio di papan
            k2 = {"label_sides": label_sides, "r2_cut_px": r2_cut, "r2_band_px": full["R2"]["band_px"],
                  "r2_overlay_px": full["R2"]["overlay_px"],
                  "r2_sides_cut": sorted(e for e, v in r2_cut.items() if v > 0),
                  "sides_match_label": sorted(e for e, v in r2_cut.items() if v > 0) == label_sides,
                  "cropdetect_cut_px": cut_px(cd_work["xyxy_incl"], W, H) if cd_work.get("ok") else None,
                  "kotak_teks_label": lab["kotak_teks_posisi"]}
        row = {"source": src.name, "source_sha256": file_sha(src), "meta": meta, "n_frames": n, "frame_wh": [W, H],
               "label": {k: lab[k] for k in ("pita", "berpita", "kotak_teks", "kotak_teks_posisi", "tata_letak_berubah")},
               "full": {r: {k: full[r][k] for k in DET_KEYS} for r in RULES}, "r2_cropped": cropped,
               "r1_equals_r2_full": full["R1"]["active_xyxy_incl"] == ref,
               "K1": None if lab["berpita"] else (not cropped), "K2_angka": k2,
               "K3_fp_vs_r2_full": summ["R2"]["fp"], "K3_fp_vs_frame": summ["R2"]["fp_vs_frame"],
               "windows_summary": summ, "windows": wins,
               "cropdetect_work": cd_work, "cropdetect_source": cd_src,
               "cropdetect_work_is_full": cd_work.get("xyxy_incl") == frame_box if cd_work.get("ok") else None,
               "time_s": {"ingest": round(t_ing, 2), "read_stats_detect_full": round(t_full, 2), "windows": round(t_win, 2),
                          "per_frame_read_stats_ms": round(1000 * t_full / n, 2)}}
        res["clips"][clip] = row
        print(f"{clip} ({n} f, {W}x{H}; label berpita {lab['berpita']}, kotak teks {lab['kotak_teks_posisi']}): R2 {ref} pita "
              f"{full['R2']['band_px']} overlay {full['R2']['overlay_px']} dipotong {cropped} -> K1 {row['K1']}; jendela R2 "
              f"{summ['R2']['fp']}/{summ['R2']['n']} potong berlebih (vs frame penuh {summ['R2']['fp_vs_frame']}, kurang "
              f"{summ['R2']['fn']}); R1 rentang penuh {full['R1']['active_xyxy_incl']}, jendela R1 {summ['R1']['fp']}/{summ['R1']['n']}; "
              f"cropdetect kerja {cd_work.get('xyxy_incl')} sumber {cd_src.get('xyxy_incl')}; ingest {t_ing:.1f} s, rentang penuh "
              f"{t_full:.1f} s, jendela {t_win:.1f} s" + (f"\n  K2 angka: {k2}" if k2 else ""), flush=True)
    res["windows_total"] = tot
    res["windows_total_r2_vs_frame"] = tot_frame
    k1 = [c["K1"] for c in res["clips"].values() if c["K1"] is not None]
    res["K1_all"] = all(k1) if k1 else None            # None = tidak ada klip tanpa pita di himpunan ini
    res["K3_all"] = tot["R2"]["fp"] == 0
    res["scratch_baru_bytes"] = sum(p.stat().st_size for p in BARU_DIR.rglob("*") if p.is_file())
    print(f"total jendela {tot}; R2 vs frame penuh {tot_frame}; K1 semua {res['K1_all']}; K3 {res['K3_all']}; scratch baru "
          f"{res['scratch_baru_bytes'] / 2**30:.2f} GiB")
    jsave(OUT / out_name, res)


# ── kasus K3: statistik kolom tepi (aritmetika R2 per kolom; R2 TIDAK diubah) ─
EDGE_COLS = 16
DARK_MAX = 24                                    # = cropdetect limit (pembanding): kolom "gelap" bila maks kanal <= 24 di frame itu


def column_stats(cols: np.ndarray, P=R2_PARAMS) -> dict:
    """cols (k, H, C, 3) uint8 -> per kolom: fraksi piksel statis (sd <= STATIC_STD_MAX), simpangan baku spasial warna rata-rata
    (maks antar kanal; syarat seragam), kecerahan rata-rata, fraksi frame gelap. Rumus = Stats + line_ok (jumlah bilangan bulat di
    float64 eksak -> sama dengan akumulasi per frame)."""
    a = cols.astype(np.float64)
    k = a.shape[0]
    m = a.sum(axis=0) / k
    sd = np.sqrt(np.maximum((a * a).sum(axis=0) / k - m * m, 0.0)).max(axis=2).astype(np.float32)
    m = m.astype(np.float32)
    static = (sd <= P["STATIC_STD_MAX"]).mean(axis=0)
    uniform = m.std(axis=0).max(axis=1)
    dark = (cols.max(axis=(1, 3)) <= DARK_MAX).mean(axis=0)
    return {"static_frac": [round(float(v), 4) for v in static], "uniform_std": [round(float(v), 2) for v in uniform],
            "mean_level": [round(float(v), 1) for v in m.mean(axis=(0, 2))], "dark_frames_frac": [round(float(v), 4) for v in dark],
            "mean_bgr": [[round(float(c), 1) for c in v] for v in m.mean(axis=0)],
            "temporal_sd_median": [round(float(v), 2) for v in np.median(sd, axis=0)],
            "line_ok_static_uniform": [bool(s >= P["STATIC_LINE_FRAC"] and u <= P["UNIFORM_STD_MAX"]) for s, u in zip(static, uniform)]}


def cmd_tepi() -> None:
    require_blind_labels()
    res = json.loads((OUT / "baru.json").read_text(encoding="utf-8"))
    out = {"edge_cols": EDGE_COLS, "dark_max": DARK_MAX, "catatan": "kolom 0..15 = kiri, lalu W-16..W-1 = kanan; statistik pada kolom "
                                                                   "penuh (tinggi frame), bukan di dalam persegi aktif yang sudah dikupas", "clips": {}}
    for clip in NEW_CLIPS:
        row = res["clips"][clip]
        if row["windows_summary"]["R2"]["fp"] == 0 and row["cropdetect_work_is_full"] is not False:
            continue                             # hanya klip dengan kasus K3 atau cropdetect memotong
        paths = sorted((BARU_DIR / clip / "frames").glob(ing.FRAME_GLOB))
        W = row["frame_wh"][0]
        idx = list(range(EDGE_COLS)) + list(range(W - EDGE_COLS, W))
        cols = np.stack([read_png(p)[:, idx] for p in paths])
        full = column_stats(cols)
        wins = []
        for w in row["windows"]["R2"]:
            a, b = w["range"]
            st = column_stats(cols[a:b + 1])
            wins.append({"range": [a, b], "fp": w["fp"], "band_px": w["band_px"], **st})
        out["clips"][clip] = {"columns": idx, "full": full, "windows": wins}
        print(f"{clip}: kolom {idx}")
        first_fp = next((w for w in wins if w["fp"]), None)
        for name, st in [("rentang penuh", full)] + [(f"jendela {w['range']} fp {w['fp']}", w) for w in wins
                                                     if not w["fp"] or w is first_fp]:
            print(f"  {name}: statis {st['static_frac']}\n    seragam {st['uniform_std']}\n    terang {st['mean_level']}\n"
                  f"    gelap {st['dark_frames_frac']}\n    line_ok {''.join('1' if v else '.' for v in st['line_ok_static_uniform'])}")
    jsave(OUT / "tepi.json", out)


def cmd_tepi_k2() -> None:
    """klip11: kolom tepi kiri 16 + kanan 16 (strip satu sisi kanan: R1 4 px, cropdetect 6 px) — setinggi frame dan HANYA di baris
    yang disimpan kotak R2 rentang penuh (yang terlihat di hasil). Baca-saja dari scratch."""
    require_blind_labels_k2()
    res = json.loads((OUT / "baru_k2.json").read_text(encoding="utf-8"))
    row = res["clips"]["klip11"]
    W, H = row["frame_wh"]
    x0, y0, x1, y1 = row["full"]["R2"]["active_xyxy_incl"]
    paths = sorted((BARU_DIR / "klip11" / "frames").glob(ing.FRAME_GLOB))
    idx = list(range(16)) + list(range(W - 16, W))
    cols = np.stack([read_png(p)[:, idx] for p in paths])
    out = {"clip": "klip11", "columns": idx, "r2_box": [x0, y0, x1, y1], "r1_full": row["full"]["R1"],
           "cropdetect_work": row["cropdetect_work"].get("xyxy_incl"),
           "full_height": column_stats(cols), "rows_kept": column_stats(cols[:, y0:y1 + 1]), "rows_kept_range": [y0, y1]}
    for name in ("full_height", "rows_kept"):
        st = out[name]
        print(f"{name} (baris {'0-' + str(H - 1) if name == 'full_height' else f'{y0}-{y1}'}):")
        for i, x in enumerate(idx):
            if x >= W - 12 or x < 4:
                print(f"  kolom {x}: terang {st['mean_level'][i]} BGR {st['mean_bgr'][i]} sd-temporal median {st['temporal_sd_median'][i]} "
                      f"statis {st['static_frac'][i]} seragam {st['uniform_std'][i]} gelap(<=24) {st['dark_frames_frac'][i]} "
                      f"line_ok {st['line_ok_static_uniform'][i]}")
    jsave(OUT / "tepi_k2.json", out)


def cmd_cek_k2() -> None:
    """Cek angka untuk penilaian K2 (baca-saja): (a) klip10 — ada piksel tidak hitam di pita kiri / kanan (kotak teks redup di pita
    ikut terkupas?) dan daerah statis kontras di sudut kanan atas area aktif; (b) klip11 — profil baris 380–470 (rentang penuh vs
    jendela f12–f35, dan per frame papan)."""
    require_blind_labels_k2()
    res = json.loads((OUT / "baru_k2.json").read_text(encoding="utf-8"))
    out = {}
    r10 = res["clips"]["klip10"]
    x0, y0, x1, y1 = r10["full"]["R2"]["active_xyxy_incl"]
    m, sd = stats_of_pngs(sorted((BARU_DIR / "klip10" / "frames").glob(ing.FRAME_GLOB)))
    lum = m.max(axis=2)
    bands = {}
    for name, sl in (("left", np.s_[:, :x0]), ("right", np.s_[:, x1 + 1:])):
        b = lum[sl]
        ys, xs = np.nonzero(b > 16)
        bands[name] = {"max_mean_level": round(float(b.max()), 1), "px_gt16": int(b.size and (b > 16).sum()), "px": int(b.size),
                       "rows_gt16": [int(ys.min()), int(ys.max())] if ys.size else None,
                       "cols_gt16": [int(xs.min()) + (0 if name == "left" else x1 + 1), int(xs.max()) + (0 if name == "left" else x1 + 1)]
                       if xs.size else None, "max_line_std": round(float(m[sl].std(axis=0).max()), 2)}
    act = sd[y0:y1 + 1, x0:x1 + 1]
    corner = {}
    for name, (ya, yb, xa, xb) in {"kanan_atas_100x135": (0, 100, act.shape[1] - 135, act.shape[1]),
                                   "kiri_atas_100x135": (0, 100, 0, 135)}.items():
        s, l = act[ya:yb, xa:xb], lum[y0:y1 + 1, x0:x1 + 1][ya:yb, xa:xb]
        corner[name] = {"static_frac": round(float((s <= R2_PARAMS["STATIC_STD_MAX"]).mean()), 3),
                        "static_bright_frac": round(float(((s <= R2_PARAMS["STATIC_STD_MAX"]) & (l > 200)).mean()), 3),
                        "mean_level": round(float(l.mean()), 1)}
    out["klip10"] = {"box": [x0, y0, x1, y1], "bands": bands, "active_corners": corner,
                     "active_static_frac": round(float((act <= R2_PARAMS["STATIC_STD_MAX"]).mean()), 3)}
    print(f"klip10 pita: {bands}\n  sudut area aktif: {corner}; area aktif statis {out['klip10']['active_static_frac']}")
    r11 = res["clips"]["klip11"]
    paths = sorted((BARU_DIR / "klip11" / "frames").glob(ing.FRAME_GLOB))
    rows = list(range(380, 472))
    def prof(ps: list[Path]) -> dict:
        mm, ss = stats_of_pngs(ps)
        return {"static_frac": [round(float((ss[y] <= R2_PARAMS["STATIC_STD_MAX"]).mean()), 3) for y in rows],
                "mean_level": [round(float(mm[y].max(axis=1).mean()), 1) for y in rows]}
    out["klip11"] = {"rows": rows, "full": prof(paths), "win_f12_35": prof(paths[12:36]), "win_f0_23": prof(paths[0:24]),
                     "frame_level": {f: [round(float(read_png(paths[f])[y].max(axis=1).mean()), 1) for y in rows] for f in (0, 11, 23, 39, 77)}}
    k = out["klip11"]
    for i in range(0, len(rows), 6):
        y = rows[i]
        print(f"klip11 baris {y}: statis penuh {k['full']['static_frac'][i]} / f12-35 {k['win_f12_35']['static_frac'][i]} / f0-23 "
              f"{k['win_f0_23']['static_frac'][i]}; terang rata-rata penuh {k['full']['mean_level'][i]}; per frame "
              + " ".join(f"f{f}={v[i]}" for f, v in k["frame_level"].items()))
    jsave(OUT / "cek_k2.json", out)


# ── papan K2 (Tahap 1b langkah 4) ──────────────────
BOARD_HEADER_PX, BOARD_GAP_PX = 92, 16
BOARD_BOX_BGR, BOARD_GAP_BGR = (255, 0, 255), (128, 128, 128)
ZOOM_COLS, ZOOM_X = 48, 4                        # panel tepi: 48 kolom terluar kiri / kanan, diperlebar 4x (horizontal saja)


def edge_zoom(img: np.ndarray, side: str, cut_x: int | None) -> np.ndarray:
    """48 kolom terluar diperlebar 4x (nearest, tinggi tetap) + garis potong magenta (cut_x = kolom pertama yang DIBUANG di
    kanan / kolom pertama yang DISIMPAN di kiri, koordinat frame)."""
    W = img.shape[1]
    x_start = 0 if side == "left" else W - ZOOM_COLS
    z = cv2.resize(img[:, x_start:x_start + ZOOM_COLS], None, fx=ZOOM_X, fy=1, interpolation=cv2.INTER_NEAREST)
    if cut_x is not None:
        xz = (cut_x - x_start) * ZOOM_X
        cv2.line(z, (xz, 0), (xz, z.shape[0] - 1), BOARD_BOX_BGR, 2)
    return z


def make_board(img: np.ndarray, box: list[int], cropped: bool, title: str, status: str | None = None) -> np.ndarray:
    """Frame asli (+ kotak magenta bila memotong) | tepi kiri 4x | tepi kanan 4x | hasil potong (hanya bila memotong)."""
    H, W = img.shape[:2]
    gap = np.full((H, BOARD_GAP_PX, 3), BOARD_GAP_BGR, np.uint8)
    left = img.copy()
    x0, y0, x1, y1 = box
    if cropped:
        cv2.rectangle(left, (x0, y0), (x1, y1), BOARD_BOX_BGR, 3)
    panels = [left, gap, edge_zoom(img, "left", x0 if cropped else None), gap,
              edge_zoom(img, "right", x1 + 1 if cropped else None)]
    if cropped:
        crop = img[y0:y1 + 1, x0:x1 + 1]
        res = np.full((H, crop.shape[1], 3), 255, np.uint8)
        oy = (H - crop.shape[0]) // 2
        res[oy:oy + crop.shape[0]] = crop
        panels += [gap, res]
        status = status or f"R2 memotong: kotak {box} ({x1 - x0 + 1}x{y1 - y0 + 1})"
        layout = "panel: asli + kotak | tepi kiri 48 px 4x | tepi kanan 48 px 4x (garis = potong) | hasil potong"
    else:
        status = status or f"R2: tidak dipotong (kotak = frame penuh {W}x{H})"
        layout = "panel: asli | tepi kiri 48 px diperlebar 4x | tepi kanan 48 px diperlebar 4x"
    body = np.hstack(panels)
    header = np.full((BOARD_HEADER_PX, body.shape[1], 3), 255, np.uint8)
    for i, text in enumerate((title, status, layout)):
        cv2.putText(header, text, (8, 24 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([header, body])


def zoom_strip(img: np.ndarray, side: str, cut: int | None) -> np.ndarray:
    """48 px di sisi `side`, diperbesar 4x tegak lurus tepi (nearest). cut = koordinat garis potong (kolom / baris pertama yang
    DISIMPAN di kiri / atas, pertama yang DIBUANG di kanan / bawah): jendela berpusat di garis itu + garis magenta; None = 48 px terluar."""
    H, W = img.shape[:2]
    L = W if side in ("left", "right") else H
    if cut is None:
        s = 0 if side in ("left", "top") else L - ZOOM_COLS
    else:
        s = min(max(cut - ZOOM_COLS // 2, 0), L - ZOOM_COLS)
    if side in ("left", "right"):
        z = cv2.resize(img[:, s:s + ZOOM_COLS], None, fx=ZOOM_X, fy=1, interpolation=cv2.INTER_NEAREST)
        if cut is not None:
            cv2.line(z, ((cut - s) * ZOOM_X, 0), ((cut - s) * ZOOM_X, z.shape[0] - 1), BOARD_BOX_BGR, 2)
    else:
        z = cv2.resize(img[s:s + ZOOM_COLS], None, fx=1, fy=ZOOM_X, interpolation=cv2.INTER_NEAREST)
        if cut is not None:
            cv2.line(z, (0, (cut - s) * ZOOM_X), (z.shape[1] - 1, (cut - s) * ZOOM_X), BOARD_BOX_BGR, 2)
    return z


def make_board_k2(img: np.ndarray, box: list[int], title: str, status: str, with_tb: bool) -> np.ndarray:
    """Asli + kotak | zoom kiri | zoom kanan | (zoom atas + bawah, bila with_tb) | hasil potong. Zoom berpusat di garis potong."""
    H, W = img.shape[:2]
    x0, y0, x1, y1 = box
    cut = cut_px(box, W, H)
    gap = np.full((H, BOARD_GAP_PX, 3), BOARD_GAP_BGR, np.uint8)
    left = img.copy()
    cv2.rectangle(left, (x0, y0), (x1, y1), BOARD_BOX_BGR, 3)
    panels = [left, gap, zoom_strip(img, "left", x0 if cut["left"] else None), gap,
              zoom_strip(img, "right", x1 + 1 if cut["right"] else None)]
    layout = "asli + kotak | kiri 48 px 4x | kanan 48 px 4x"
    if with_tb:
        col = np.full((H, W, 3), 255, np.uint8)
        top = zoom_strip(img, "top", y0 if cut["top"] else None)
        bot = zoom_strip(img, "bottom", y1 + 1 if cut["bottom"] else None)
        col[:top.shape[0]] = top
        col[top.shape[0] + BOARD_GAP_PX:top.shape[0] + BOARD_GAP_PX + bot.shape[0]] = bot
        col[top.shape[0]:top.shape[0] + BOARD_GAP_PX] = BOARD_GAP_BGR
        panels += [gap, col]
        layout += " | atas 48 px 4x (atas) + bawah 48 px 4x (di bawahnya)"
    crop = img[y0:y1 + 1, x0:x1 + 1]
    res = np.full((H, crop.shape[1], 3), 255, np.uint8)
    oy = (H - crop.shape[0]) // 2
    res[oy:oy + crop.shape[0]] = crop
    panels += [gap, res]
    layout += " | hasil potong; zoom berpusat di garis potong (magenta)"
    body = np.hstack(panels)
    header = np.full((BOARD_HEADER_PX, body.shape[1], 3), 255, np.uint8)
    for i, text in enumerate((title, status, layout)):
        cv2.putText(header, text, (8, 24 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([header, body])


def cmd_papan_k2() -> None:
    """Papan K2 klip10–klip11 (nomor lanjut dari papan 1b; papan 01–05 tidak disentuh): per klip awal / tengah / akhir dengan kotak
    R2 rentang penuh + kasus K3 (kotak jendela potong berlebih paling sering)."""
    labels = require_blind_labels_k2()
    res = json.loads((OUT / "baru_k2.json").read_text(encoding="utf-8"))
    old_listing = NILAI_DIR / "papan_k2.json"
    if old_listing.is_file():
        for b in json.loads(old_listing.read_text(encoding="utf-8")):
            (NILAI_DIR / b["file"]).unlink(missing_ok=True)
    k = len(json.loads((NILAI_DIR / "papan.json").read_text(encoding="utf-8")))
    listing = []
    for clip in K2_CLIPS:
        row = res["clips"][clip]
        n, W, H = row["n_frames"], *row["frame_wh"]
        paths = sorted((BARU_DIR / clip / "frames").glob(ing.FRAME_GLOB))
        lab = labels["clips"][clip]
        tb = lab["pita"]["atas"] or lab["pita"]["bawah"] or any(cut_px(row["full"]["R2"]["active_xyxy_incl"], W, H)[e] for e in ("top", "bottom"))
        jobs = [(f, row["full"]["R2"]["active_xyxy_incl"], "rentang penuh (kotak produksi)", None) for f in (0, n // 2, n - 1)]
        fps = [w for w in row["windows"]["R2"] if w["fp"]]
        if fps:
            boxes = [tuple(w["active_xyxy_incl"]) for w in fps]
            top = max(set(boxes), key=lambda bx: (boxes.count(bx), -boxes.index(bx)))
            w = fps[boxes.index(top)]
            a, b = w["range"]
            e = {s: v for s, v in w["excess_vs_full_px"].items() if v}
            jobs.append(((a + b) // 2, w["active_xyxy_incl"], f"kasus K3: jendela f{a}-f{b}",
                         f"R2 jendela memotong {w['active_xyxy_incl']} (lebih dalam {e} px; kotak ini di {boxes.count(top)} / "
                         f"{len(row['windows']['R2'])} jendela)"))
        for f, box, kind, status in jobs:
            k += 1
            c = cut_px(box, W, H)
            status = status or f"R2 memotong {box} = atas {c['top']} / bawah {c['bottom']} / kiri {c['left']} / kanan {c['right']} px"
            board = make_board_k2(read_png(paths[f]), box, f"{k:02d} {clip} f{f} / {n - 1} - {kind}", status, tb)
            name = f"{k:02d}_{clip}_f{f:05d}.png"
            ok, buf = cv2.imencode(".png", board)
            if not ok:
                raise SystemExit(f"encode gagal {name}")
            (NILAI_DIR / name).write_bytes(buf.tobytes())
            listing.append({"file": name, "clip": clip, "frame": f, "n_frames": n, "kind": kind, "cropped": True, "box": box,
                            "board_wh": [board.shape[1], board.shape[0]], "sha256": file_sha(NILAI_DIR / name)})
            print(f"{name}: {kind}, {board.shape[1]}x{board.shape[0]}, kotak {box}", flush=True)
    jsave(old_listing, listing)


def cmd_papan() -> None:
    labels = require_blind_labels()
    res = json.loads((OUT / "baru.json").read_text(encoding="utf-8"))
    NILAI_DIR.mkdir(parents=True, exist_ok=True)
    for old in NILAI_DIR.glob("*.png"):       # hanya papan buatan alat ini; PENILAIAN.md tidak disentuh
        old.unlink()
    listing, k = [], 0

    def save(clip: str, f: int, n: int, box: list[int], cropped: bool, kind: str, status: str | None = None) -> None:
        nonlocal k
        k += 1
        img = read_png(sorted((BARU_DIR / clip / "frames").glob(ing.FRAME_GLOB))[f])
        board = make_board(img, box, cropped, f"{k:02d} {clip} f{f} / {n - 1} - {kind}", status)
        name = f"{k:02d}_{clip}_f{f:05d}.png"
        ok, buf = cv2.imencode(".png", board)
        if not ok:
            raise SystemExit(f"encode gagal {name}")
        (NILAI_DIR / name).write_bytes(buf.tobytes())
        listing.append({"file": name, "clip": clip, "frame": f, "n_frames": n, "kind": kind, "cropped": cropped, "box": box,
                        "board_wh": [board.shape[1], board.shape[0]], "sha256": file_sha(NILAI_DIR / name)})
        print(f"{name}: {kind}, {board.shape[1]}x{board.shape[0]}, dipotong {cropped} {box}", flush=True)

    for clip in NEW_CLIPS:
        row = res["clips"][clip]
        n, box, cropped = row["n_frames"], row["full"]["R2"]["active_xyxy_incl"], row["r2_cropped"]
        three = labels["clips"][clip]["berpita"] or cropped       # berpita (label) ATAU dipotong R2 -> awal / tengah / akhir
        for f in ([0, n // 2, n - 1] if three else [n // 2]):
            save(clip, f, n, box, cropped, "rentang penuh (kotak produksi)")
        fps = [w for w in row["windows"]["R2"] if w["fp"]]          # kasus K3: kotak potong berlebih yang PALING SERING
        if fps:
            boxes = [tuple(w["active_xyxy_incl"]) for w in fps]
            top = max(set(boxes), key=lambda bx: (boxes.count(bx), -boxes.index(bx)))
            fp = fps[boxes.index(top)]
            a, b = fp["range"]
            e = {s: v for s, v in fp["excess_vs_full_px"].items() if v}
            save(clip, (a + b) // 2, n, fp["active_xyxy_incl"], True, f"kasus K3: jendela f{a}-f{b}",
                 f"R2 jendela memotong {fp['active_xyxy_incl']} (lebih dalam {e} px; kotak ini di {boxes.count(top)} / "
                 f"{len(row['windows']['R2'])} jendela)")
    jsave(NILAI_DIR / "papan.json", listing)


def main(argv: list[str]) -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    cmds = {"r2": cmd_r2, "probe": cmd_probe, "cropdetect": cmd_cropdetect, "ukur": cmd_ukur, "trimframe": cmd_trimframe,
            "probe-baru": cmd_probe_baru, "label": cmd_label, "baru": cmd_baru, "tepi": cmd_tepi, "papan": cmd_papan,
            "penilaian": cmd_penilaian, "probe-k2": cmd_probe_k2, "label-k2": cmd_label_k2, "baru-k2": cmd_baru_k2,
            "papan-k2": cmd_papan_k2, "tepi-k2": cmd_tepi_k2, "cek-k2": cmd_cek_k2,
            "penilaian-k2": cmd_penilaian_k2}
    if len(argv) != 1 or argv[0] not in cmds:
        raise SystemExit(__doc__)
    cmds[argv[0]]()


if __name__ == "__main__":
    main(sys.argv[1:])
