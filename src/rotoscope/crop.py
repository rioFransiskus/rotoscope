"""Saran crop [1] (T-607, D-013): deteksi pita aturan R2 BEKU pada RENTANG PENUH SUMBER → saran + pratinjau. Kotak dipakai ingest
HANYA bila diterima lewat alat CLI `python -m rotoscope crop <video>` (BUKAN stage, CPU). Modul ini tidak pernah memotong sendiri.

Berkas per SUMBER video (bukan per folder klip; dipakai bersama semua folder <nama>@S-E), di <paths.work_dir>/crop/<nama>/ — di
luar folder klip, jadi tidak tersentuh `run --restart-from ingest` maupun penghapusan folder klip:
- `deteksi.json` (`rotoscope-crop-deteksi/1`): kunci D1 + hasil (juga "tanpa pita"). Kunci sama → deteksi dilewati.
- `saran.json` (`rotoscope-crop-saran/1`): HANYA bila pita terdeteksi — kotak kerja + sumber, pita / kotak teks per sisi, petunjuk.
- `pratinjau/` 3 PNG (awal / tengah / akhir rentang penuh) + `pratinjau.json`.
- `kotak.json` (`rotoscope-crop-kotak/1`): keputusan `saran` | `manual` | `none`, ditulis HANYA oleh alat `crop`; isi lama →
  `kotak.prev.json`.
Identitas sumber = sha256 byte + ukuran + dimensi (sesudah rotasi); path hanya dicatat sebagai info (docs/04 persetujuan Tahap 1c b).

Aturan R2 = salinan persis `scripts/t607_study.py detect_crop(rule="R2")` (docs/04 "R2 DIBEKUKAN"): statistik temporal per piksel
(akumulasi float64, hasil float32), kupas pita statis + seragam dari tepi, syarat pasangan ≥ `band_min_px`, kotak teks hanya atas /
bawah, pengaman area aktif. Parameter dari `ingest.crop.detect`; selain nilai beku → PERINGATAN + sidik lain.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import cv2
import numpy as np

from rotoscope.config import CropConfig, CropDetectConfig
from rotoscope.stage_common import StageError, config_flag, quote_arg, write_bytes_atomic, write_json_atomic

CROP_DIRNAME = "crop"
DETEKSI_FILENAME = "deteksi.json"
SARAN_FILENAME = "saran.json"
KOTAK_FILENAME = "kotak.json"
KOTAK_PREV_FILENAME = "kotak.prev.json"
PREVIEW_DIRNAME = "pratinjau"
PREVIEW_JSON = "pratinjau.json"
FORMAT_DETEKSI = "rotoscope-crop-deteksi/1"
FORMAT_SARAN = "rotoscope-crop-saran/1"
FORMAT_KOTAK = "rotoscope-crop-kotak/1"
FORMAT_PREVIEW = "rotoscope-crop-pratinjau/1"
CLI_PROG = "python -m rotoscope"

# ── Aturan R2 beku (docs/04 "R2 DIBEKUKAN") ────────
RULE = "R2"
RULE_VERSION = 1
FROZEN_PARAMS = {"STATIC_STD_MAX": 3.0, "STATIC_LINE_FRAC": 0.99, "UNIFORM_STD_MAX": 6.0, "BAND_MIN_PX": 8, "MIN_ACTIVE_FRAC": 0.10}
FROZEN_FINGERPRINT = "d40c3dfdd270a715a7bbc9bfbec96ce69ce613a1dac7480c1bcd607ec6eb1da0"
EDGES = ("top", "bottom", "left", "right")
OPP = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}
OVERLAY_EDGES = ("top", "bottom")          # kotak teks hanya di tepi atas / bawah (keputusan T-607 no. 4)
SIDE_ID = {"top": "atas", "bottom": "bawah", "left": "kiri", "right": "kanan"}

# ── Kotak ──────────────────────────────────────────
MODES = ("saran", "manual", "none")
ROUNDING = "outward-even"                  # awal floor, akhir ceil, lalu genap (yuv420); konten tidak pernah terpotong
MANUAL_MIN_PX = 16                         # kotak manual (px sumber) minimal 16 × 16
IDENTITY_KEYS = ("sha256", "bytes", "width", "height")
SHA_CHUNK = 1 << 20

# ── Pratinjau (konstanta tata letak di modul, preseden pick) ─
PREVIEW_FRAMES = ("1_awal", "2_tengah", "3_akhir")
ZOOM_PX, ZOOM_X = 48, 4                    # panel zoom: 48 px di garis potong, diperbesar 4× tegak lurus tepi
HEADER_LINE_PX, HEADER_PAD_PX = 28, 10
GAP_PX = 16
BOX_BGR = (255, 0, 255)                    # magenta = kotak / garis potong
GAP_BGR = (128, 128, 128)
HINT_BGR = (0, 140, 255)                   # oranye = baris / kolom petunjuk
HINT_ALPHA = 0.45
FONT_SCALE = 0.55
BOX_THICKNESS = 3
LINE_THICKNESS = 2


class CropError(StageError):
    """Kotak / saran crop tidak bisa dipakai (exit 1)."""


# ── R2 (salinan persis t607_study / t602_study) ────
def r2_params(d: CropDetectConfig) -> dict:
    return {"STATIC_STD_MAX": float(d.static_std_max), "STATIC_LINE_FRAC": float(d.static_line_frac),
            "UNIFORM_STD_MAX": float(d.uniform_std_max), "BAND_MIN_PX": int(d.band_min_px),
            "MIN_ACTIVE_FRAC": float(d.min_active_frac)}


def rule_fingerprint(params: Mapping) -> str:
    """sha256 JSON kanonik {rule, version, params} — penanda aturan (docs/04: nilai beku = FROZEN_FINGERPRINT)."""
    doc = {"rule": RULE, "version": RULE_VERSION, "params": dict(params)}
    return hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


class Stats:
    """Akumulator rata-rata + simpangan baku temporal per piksel (BGR 0–255, float64), hasil float32."""

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


def _line_ok(m, sd, rect, edge, need_uniform, p) -> bool:
    y0, y1, x0, x1 = rect
    if y1 - y0 < 1 or x1 - x0 < 1:
        return False
    if edge in ("top", "bottom"):
        y = y0 if edge == "top" else y1 - 1
        sdl, ml = sd[y, x0:x1], m[y, x0:x1]
    else:
        x = x0 if edge == "left" else x1 - 1
        sdl, ml = sd[y0:y1, x], m[y0:y1, x]
    if (sdl <= p["STATIC_STD_MAX"]).mean() < p["STATIC_LINE_FRAC"]:
        return False
    return (not need_uniform) or float(ml.std(axis=0).max()) <= p["UNIFORM_STD_MAX"]


def _shrink(rect, edge) -> None:
    i = {"top": 0, "bottom": 1, "left": 2, "right": 3}[edge]
    rect[i] += 1 if edge in ("top", "left") else -1


def _peel_bands(m, sd, edges, p):
    H, W = sd.shape
    rect = [0, H, 0, W]
    band = dict.fromkeys(EDGES, 0)
    changed = True
    while changed:
        changed = False
        for e in EDGES:
            if e not in edges:
                continue
            while _line_ok(m, sd, rect, e, True, p):
                _shrink(rect, e)
                band[e] += 1
                changed = True
    return rect, band


def detect_crop(m: np.ndarray, sd: np.ndarray, p: Mapping) -> dict:
    """R2: pita berpasangan ≥ BAND_MIN_PX, lalu kotak teks (baris statis) hanya di atas / bawah; area < MIN_ACTIVE_FRAC → penuh."""
    H, W = sd.shape
    allowed = set(EDGES)
    while True:
        rect, band = _peel_bands(m, sd, allowed, p)
        keep = {e for e in allowed if band[e] >= p["BAND_MIN_PX"] and band[OPP[e]] >= p["BAND_MIN_PX"]}
        if keep == allowed:
            break
        allowed = keep
    overlay = dict.fromkeys(EDGES, 0)
    for e in EDGES:
        if e not in OVERLAY_EDGES:
            continue
        if band[e] >= p["BAND_MIN_PX"]:
            while _line_ok(m, sd, rect, e, False, p):
                _shrink(rect, e)
                overlay[e] += 1
    y0, y1, x0, x1 = rect
    area = max(0, y1 - y0) * max(0, x1 - x0)
    applied = area >= p["MIN_ACTIVE_FRAC"] * H * W
    return {"frame_wh": [W, H], "band_px": band, "overlay_px": overlay, "applied": bool(applied),
            "box_work": [x0, y0, x1 - 1, y1 - 1] if applied else [0, 0, W - 1, H - 1]}


def cut_px(box: Sequence[int], W: int, H: int) -> dict[str, int]:
    """Kotak inklusif (x0, y0, x1, y1) → px yang dibuang per sisi."""
    x0, y0, x1, y1 = box
    return {"top": y0, "bottom": H - 1 - y1, "left": x0, "right": W - 1 - x1}


def _even_floor(v: int) -> int:
    return v - v % 2


def map_box_to_source(box: Sequence[int], work_wh: Sequence[int], src_wh: Sequence[int]) -> list[int]:
    """Kotak kerja inklusif → kotak sumber [x, y, w, h], pembulatan KELUAR + genap (konten tidak pernah terpotong)."""
    (Ww, Hw), (Ws, Hs) = work_wh, src_wh
    sx, sy = Ws / Ww, Hs / Hw
    x0, y0, x1, y1 = box
    xs0, ys0 = _even_floor(math.floor(x0 * sx)), _even_floor(math.floor(y0 * sy))
    xs1, ys1 = math.ceil((x1 + 1) * sx), math.ceil((y1 + 1) * sy)
    w, h = xs1 - xs0, ys1 - ys0
    return [xs0, ys0, min(w + w % 2, Ws - xs0), min(h + h % 2, Hs - ys0)]


def source_box_to_work(box: Sequence[int], work_wh: Sequence[int], src_wh: Sequence[int]) -> list[int]:
    """Kotak sumber [x, y, w, h] → kotak kerja inklusif (hanya untuk menggambar pratinjau kotak manual)."""
    (Ww, Hw), (Ws, Hs) = work_wh, src_wh
    x, y, w, h = box
    x0, y0 = round(x * Ww / Ws), round(y * Hw / Hs)
    return [x0, y0, max(x0, round((x + w) * Ww / Ws) - 1), max(y0, round((y + h) * Hw / Hs) - 1)]


def round_manual_box(box: Sequence[int], src_wh: Sequence[int]) -> list[int]:
    """`crop --box X Y W H` (px sumber sesudah rotasi) → [x, y, w, h] dibulatkan keluar + genap, divalidasi."""
    Ws, Hs = src_wh
    x, y, w, h = (int(v) for v in box)
    if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > Ws or y + h > Hs:
        raise CropError(f"--box {x} {y} {w} {h} di luar frame sumber {Ws} × {Hs} (x, y ≥ 0; W, H > 0; x + W ≤ {Ws}; y + H ≤ {Hs})")
    x0, y0 = _even_floor(x), _even_floor(y)
    x1, y1 = min(x + w + (x + w) % 2, Ws), min(y + h + (y + h) % 2, Hs)
    out = [x0, y0, x1 - x0, y1 - y0]
    if out[2] < MANUAL_MIN_PX or out[3] < MANUAL_MIN_PX:
        raise CropError(f"--box {x} {y} {w} {h} → {out[2]} × {out[3]} px: minimal {MANUAL_MIN_PX} × {MANUAL_MIN_PX}")
    if out == [0, 0, Ws, Hs]:
        raise CropError(f"--box {x} {y} {w} {h} = frame penuh {Ws} × {Hs} — untuk tanpa potong pakai --none")
    return out


# ── Identitas sumber + berkas ──────────────────────
def crop_dir_for(work_root: Path, name: str) -> Path:
    return Path(work_root) / CROP_DIRNAME / name


def source_identity(video: Path, probe: Mapping) -> dict:
    """sha256 byte + ukuran + dimensi (sesudah rotasi) + path (info, TIDAK dicek)."""
    h, size = hashlib.sha256(), 0
    with Path(video).open("rb") as f:
        for chunk in iter(lambda: f.read(SHA_CHUNK), b""):
            h.update(chunk)
            size += len(chunk)
    return {"sha256": h.hexdigest(), "bytes": size, "width": int(probe["width"]), "height": int(probe["height"]),
            "path": str(Path(video).resolve())}


def same_source(a: Mapping | None, b: Mapping) -> bool:
    return isinstance(a, Mapping) and all(a.get(k) == b.get(k) for k in IDENTITY_KEYS)


def _read_json(path: Path) -> dict | None:
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return d if isinstance(d, dict) else None


def _short(sha: str | None) -> str:
    return (sha or "?")[:12]


def ffmpeg_version(ffmpeg: str) -> str:
    p = subprocess.run([ffmpeg, "-version"], capture_output=True, text=True, encoding="utf-8", errors="replace")
    return (p.stdout or "").splitlines()[0].strip() if p.returncode == 0 and p.stdout else "?"


def detection_key(identity: Mapping, target_fps: float, working_width: int, params: Mapping, ffmpeg: str) -> dict:
    """Kunci D1: deteksi dilewati bila deteksi.json memuat kunci yang sama."""
    return {"source": {k: identity[k] for k in IDENTITY_KEYS}, "target_fps": target_fps, "working_width": working_width,
            "rule": RULE, "rule_version": RULE_VERSION, "params": dict(params), "rule_fingerprint": rule_fingerprint(params),
            "ffmpeg": ffmpeg}


def crop_cmd(video_arg: str, flags: str) -> str:
    """Perintah `crop` siap tempel (T-607): `--config` non-default proses ini ikut (stage_common.set_message_config)."""
    return f"{CLI_PROG} crop {video_arg} {flags}{config_flag()}"


def read_kotak(crop_dir: Path, video_arg: str = "<video>") -> dict | None:
    """kotak.json (keputusan) atau None. Rusak → CropError (keputusan tidak boleh diabaikan diam-diam)."""
    path = crop_dir / KOTAK_FILENAME
    if not path.is_file():
        return None
    d = _read_json(path)
    ok = (d is not None and d.get("format") == FORMAT_KOTAK and d.get("mode") in MODES
          and isinstance(d.get("source"), dict) and (d["mode"] == "none" or _is_box(d.get("box_source"))))
    if not ok:
        raise CropError(f"{path} rusak / format tidak dikenal — periksa lalu: {crop_cmd(video_arg, '--clear')} "
                        f"(atau --accept / --box / --none lagi)")
    return d


def _is_box(v) -> bool:
    return isinstance(v, list) and len(v) == 4 and all(isinstance(i, int) and i >= 0 for i in v) and v[2] > 0 and v[3] > 0


def kotak_for_ingest(crop_dir: Path, identity: Mapping, video_arg: str) -> dict | None:
    """Keputusan untuk sumber INI: None = belum ada; beda sumber (sha256 / ukuran / dimensi) → CropError (ingest exit 1)."""
    d = read_kotak(crop_dir, video_arg)
    if d is None:
        return None
    if not same_source(d["source"], identity):
        old = d["source"]
        raise CropError(
            f"kotak crop di {crop_dir / KOTAK_FILENAME} dibuat untuk sumber LAIN (sha256 {_short(old.get('sha256'))}…, "
            f"{old.get('width')} × {old.get('height')}, {old.get('bytes')} byte; sekarang {_short(identity['sha256'])}…, "
            f"{identity['width']} × {identity['height']}, {identity['bytes']} byte) — periksa lalu: "
            f"{crop_cmd(video_arg, '--clear')} (atau --accept / --box / --none lagi)")
    if d["mode"] != "none":
        x, y, w, h = d["box_source"]
        if x + w > identity["width"] or y + h > identity["height"]:
            raise CropError(f"{crop_dir / KOTAK_FILENAME}: kotak {d['box_source']} di luar frame sumber "
                            f"{identity['width']} × {identity['height']} — {crop_cmd(video_arg, '--clear')}")
    return d


def meta_crop_key(kotak: Mapping | None) -> dict | None:
    """Key `crop` meta.json HANYA bila kotak dipakai (mode none / tanpa kotak → None: meta byte-identik)."""
    if kotak is None or kotak["mode"] == "none":
        return None
    out = {"mode": kotak["mode"], "box_source": list(kotak["box_source"]), "rounding": ROUNDING,
           "source_sha256": kotak["source"]["sha256"]}
    if kotak["mode"] == "saran":
        s = kotak["saran"]
        out.update({"rule": s["rule"], "rule_version": s["rule_version"], "rule_fingerprint": s["rule_fingerprint"],
                    "detect_frame_size": s["detect_frame_size"], "box_work": s["box_work"], "band_px": s["band_px"],
                    "overlay_px": s["overlay_px"]})
    else:
        out["box_requested"] = list(kotak["box_requested"])
    return out


def pending_saran(crop_dir: Path) -> dict | None:
    """Saran yang BELUM diputuskan (saran.json ada, kotak.json tidak) — dipakai `run` untuk berhenti sesudah [1] (exit 4)."""
    if (crop_dir / KOTAK_FILENAME).is_file():
        return None
    s = _read_json(crop_dir / SARAN_FILENAME)
    return s if s is not None and s.get("format") == FORMAT_SARAN else None


def _clear_saran(crop_dir: Path) -> None:
    (crop_dir / SARAN_FILENAME).unlink(missing_ok=True)
    pdir = crop_dir / PREVIEW_DIRNAME
    if pdir.is_dir():
        shutil.rmtree(pdir)


# ── Membaca frame lewat pipa ffmpeg (tanpa PNG) ────
def _base_cmd(ffmpeg: str, video: Path, vf: str) -> list[str]:
    return [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-i", str(video), "-map", "0:v:0", "-an", "-vf", vf,
            "-fps_mode", "passthrough", "-pix_fmt", "rgb24"]


def frame_size(ffmpeg: str, video: Path, vf: str) -> tuple[int, int]:
    """Ukuran frame keluaran `vf` (dari satu frame PNG di memori) — tinggi `-2` dihitung ffmpeg, bukan ditebak."""
    p = subprocess.run([*_base_cmd(ffmpeg, video, vf), "-frames:v", "1", "-f", "image2pipe", "-c:v", "png", "-"],
                       capture_output=True)
    img = cv2.imdecode(np.frombuffer(p.stdout, np.uint8), cv2.IMREAD_COLOR) if p.returncode == 0 and p.stdout else None
    if img is None:
        raise CropError(f"ffmpeg gagal membaca frame pertama {video} (vf {vf}): "
                        f"{p.stderr.decode('utf-8', 'replace').strip()[-300:]}")
    return img.shape[1], img.shape[0]


def scan(ffmpeg: str, video: Path, vf: str, sample_at: set[int], with_stats: bool = True):
    """Semua frame keluaran `vf` lewat pipa rawvideo → (m, sd, n, contoh {indeks: BGR}); frame terakhir selalu disimpan.
    with_stats False = hanya menghitung frame + contoh (pratinjau kotak manual)."""
    w, h = frame_size(ffmpeg, video, vf)
    proc = subprocess.Popen([*_base_cmd(ffmpeg, video, vf), "-f", "rawvideo", "-"], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    st, nbytes, samples, last, k = Stats(), w * h * 3, {}, None, 0
    try:
        while True:
            buf = proc.stdout.read(nbytes)
            if len(buf) < nbytes:
                break
            bgr = np.frombuffer(buf, np.uint8).reshape(h, w, 3)[:, :, ::-1]
            if with_stats:
                st.add(bgr)
            if k in sample_at:
                samples[k] = np.ascontiguousarray(bgr)
            last, k = bgr, k + 1
        err = proc.stderr.read().decode("utf-8", "replace")
    finally:
        proc.wait()
    if proc.returncode != 0 or k == 0:
        raise CropError(f"ffmpeg gagal mendekode {video} untuk deteksi crop (exit {proc.returncode}): {err.strip()[-300:]}")
    samples[k - 1] = np.ascontiguousarray(last)
    m, sd = st.result() if with_stats else (None, None)
    return m, sd, k, samples


def grab_frame(ffmpeg: str, video: Path, vf: str, index: int, wh: tuple[int, int]) -> np.ndarray:
    """Satu frame keluaran `vf` (indeks dari 0) — dipakai bila frame tengah tidak tertangkap saat scan."""
    w, h = wh
    p = subprocess.run([*_base_cmd(ffmpeg, video, f"{vf},select=eq(n\\,{index})"), "-frames:v", "1", "-f", "rawvideo", "-"],
                       capture_output=True)
    if p.returncode != 0 or len(p.stdout) < w * h * 3:
        raise CropError(f"ffmpeg gagal mengambil frame {index} dari {video}: {p.stderr.decode('utf-8', 'replace').strip()[-300:]}")
    return np.ascontiguousarray(np.frombuffer(p.stdout[: w * h * 3], np.uint8).reshape(h, w, 3)[:, :, ::-1])


def preview_samples(ffmpeg: str, video: Path, vf: str, n: int, samples: dict, wh: tuple[int, int]) -> list[tuple[str, int, np.ndarray]]:
    idx = (0, n // 2, n - 1)
    out = []
    for label, i in zip(PREVIEW_FRAMES, idx):
        img = samples.get(i)
        out.append((label, i, img if img is not None else grab_frame(ffmpeg, video, vf, i, wh)))
    return out


# ── Petunjuk (informasi; kotak tidak diubah) ───────
def hints(sd: np.ndarray, box: Sequence[int], cut: Mapping[str, int], frac_min: float, p: Mapping) -> dict | None:
    """Baris / kolom di DALAM kotak, dari garis potong ke dalam, yang ≥ frac_min piksel diam (sd ≤ STATIC_STD_MAX), hanya di sisi
    yang DIPOTONG. Kotak alternatif = kotak tanpa baris itu (None bila < MIN_ACTIVE_FRAC frame)."""
    H, W = sd.shape
    x0, y0, x1, y1 = box
    static = sd <= p["STATIC_STD_MAX"]

    def run(lines) -> int:
        k = 0
        for line in lines:
            if float(line.mean()) < frac_min:
                break
            k += 1
        return k

    n = {"top": run(static[y, x0:x1 + 1] for y in range(y0, y1 + 1)) if cut["top"] else 0,
         "bottom": run(static[y, x0:x1 + 1] for y in range(y1, y0 - 1, -1)) if cut["bottom"] else 0,
         "left": run(static[y0:y1 + 1, x] for x in range(x0, x1 + 1)) if cut["left"] else 0,
         "right": run(static[y0:y1 + 1, x] for x in range(x1, x0 - 1, -1)) if cut["right"] else 0}
    if not any(n.values()):
        return None
    lines = {}
    if n["top"]:
        lines["top"] = [y0, y0 + n["top"] - 1]
    if n["bottom"]:
        lines["bottom"] = [y1 - n["bottom"] + 1, y1]
    if n["left"]:
        lines["left"] = [x0, x0 + n["left"] - 1]
    if n["right"]:
        lines["right"] = [x1 - n["right"] + 1, x1]
    alt = [x0 + n["left"], y0 + n["top"], x1 - n["right"], y1 - n["bottom"]]
    ok = alt[0] <= alt[2] and alt[1] <= alt[3] and (alt[2] - alt[0] + 1) * (alt[3] - alt[1] + 1) >= p["MIN_ACTIVE_FRAC"] * H * W
    return {"lines": lines, "count": {e: v for e, v in n.items() if v}, "alt_box_work": alt if ok else None}


# ── Pratinjau ──────────────────────────────────────
def _shade(img: np.ndarray, box: Sequence[int], lines: Mapping[str, Sequence[int]]) -> np.ndarray:
    out = img.copy()
    x0, y0, x1, y1 = box
    for e, (a, b) in lines.items():
        region = (np.s_[a:b + 1, x0:x1 + 1] if e in ("top", "bottom") else np.s_[y0:y1 + 1, a:b + 1])
        patch = out[region].astype(np.float32)
        out[region] = np.clip(patch * (1 - HINT_ALPHA) + np.array(HINT_BGR, np.float32) * HINT_ALPHA + 0.5, 0, 255).astype(np.uint8)
    return out


def _zoom(img: np.ndarray, side: str, cut: int | None) -> np.ndarray:
    """ZOOM_PX px di sisi `side`, diperbesar ZOOM_X× tegak lurus tepi; cut = koordinat garis potong (kolom / baris pertama yang
    DISIMPAN di kiri / atas, pertama yang DIBUANG di kanan / bawah) → berpusat di garis itu + garis magenta; None = 48 px terluar."""
    H, W = img.shape[:2]
    L = W if side in ("left", "right") else H
    n = min(ZOOM_PX, L)
    s = (0 if side in ("left", "top") else L - n) if cut is None else min(max(cut - n // 2, 0), L - n)
    if side in ("left", "right"):
        z = cv2.resize(img[:, s:s + n], None, fx=ZOOM_X, fy=1, interpolation=cv2.INTER_NEAREST)
        if cut is not None:
            cv2.line(z, ((cut - s) * ZOOM_X, 0), ((cut - s) * ZOOM_X, z.shape[0] - 1), BOX_BGR, LINE_THICKNESS)
    else:
        z = cv2.resize(img[s:s + n], None, fx=1, fy=ZOOM_X, interpolation=cv2.INTER_NEAREST)
        if cut is not None:
            cv2.line(z, (0, (cut - s) * ZOOM_X), (z.shape[1] - 1, (cut - s) * ZOOM_X), BOX_BGR, LINE_THICKNESS)
    return z


def make_board(img: np.ndarray, box: Sequence[int], lines: Mapping[str, Sequence[int]], header: Sequence[str]) -> np.ndarray:
    """asli + kotak | zoom kiri | zoom kanan | (zoom atas + bawah, bila dipotong atas / bawah) | hasil potong. Petunjuk diarsir."""
    H, W = img.shape[:2]
    x0, y0, x1, y1 = box
    c = cut_px(box, W, H)
    shaded = _shade(img, box, lines)
    gap = np.full((H, GAP_PX, 3), GAP_BGR, np.uint8)
    main = shaded.copy()
    cv2.rectangle(main, (x0, y0), (x1, y1), BOX_BGR, BOX_THICKNESS)
    panels = [main, gap, _zoom(shaded, "left", x0 if c["left"] else None), gap,
              _zoom(shaded, "right", x1 + 1 if c["right"] else None)]
    if c["top"] or c["bottom"]:
        top, bot = _zoom(shaded, "top", y0 if c["top"] else None), _zoom(shaded, "bottom", y1 + 1 if c["bottom"] else None)
        col = np.full((max(H, top.shape[0] + GAP_PX + bot.shape[0]), W, 3), 255, np.uint8)
        col[:top.shape[0]] = top
        col[top.shape[0]:top.shape[0] + GAP_PX] = GAP_BGR
        col[top.shape[0] + GAP_PX:top.shape[0] + GAP_PX + bot.shape[0]] = bot
        panels += [gap, col[:H]]
    crop = shaded[y0:y1 + 1, x0:x1 + 1]
    res = np.full((H, crop.shape[1], 3), 255, np.uint8)
    oy = (H - crop.shape[0]) // 2
    res[oy:oy + crop.shape[0]] = crop
    panels += [gap, res]
    body = np.hstack(panels)
    head = np.full((HEADER_PAD_PX + HEADER_LINE_PX * len(header), body.shape[1], 3), 255, np.uint8)
    for i, text in enumerate(header):
        cv2.putText(head, text, (8, HEADER_LINE_PX * (i + 1) - 4), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([head, body])


def _ascii(s: str) -> str:
    """cv2.putText hanya ASCII."""
    return s.replace("×", "x").replace("≥", ">=").replace("→", "->").replace("–", "-").replace("—", "-")


def write_preview(crop_dir: Path, frames: Sequence[tuple[str, int, np.ndarray]], box: Sequence[int], hint: Mapping | None,
                  jenis: str, identity: Mapping, info: Sequence[str]) -> dict:
    """crop/pratinjau/<1_awal|2_tengah|3_akhir>_fNNNNN.png + pratinjau.json. PNG lama di folder itu dihapus dulu."""
    pdir = crop_dir / PREVIEW_DIRNAME
    if pdir.is_dir():
        shutil.rmtree(pdir)
    pdir.mkdir(parents=True)
    n_total = frames[-1][1] + 1
    lines = (hint or {}).get("lines", {})
    files = []
    layout = "panel: asli + kotak (magenta) | zoom kiri | zoom kanan | (zoom atas + bawah) | hasil potong; zoom 48 px 4x di garis potong"
    for label, idx, img in frames:
        header = [_ascii(f"{label} f{idx} / {n_total - 1} - {jenis}"), *(_ascii(t) for t in info), layout]
        ok, buf = cv2.imencode(".png", make_board(img, box, lines, header))
        if not ok:
            raise CropError(f"gagal meng-encode pratinjau {label}")
        name = f"{label}_f{idx:05d}.png"
        write_bytes_atomic(pdir / name, buf.tobytes())
        files.append({"file": name, "frame": idx, "sha256": hashlib.sha256(buf.tobytes()).hexdigest()})
    doc = {"format": FORMAT_PREVIEW, "jenis": jenis, "box_work": list(box), "source_sha256": identity["sha256"],
           "hint_lines": dict(lines), "files": files}
    write_json_atomic(pdir / PREVIEW_JSON, doc)
    return doc


def _preview_valid(crop_dir: Path, saran: Mapping) -> bool:
    doc = _read_json(crop_dir / PREVIEW_DIRNAME / PREVIEW_JSON)
    if doc is None or doc.get("format") != FORMAT_PREVIEW or doc.get("jenis") != "saran" or doc.get("box_work") != saran["box_work"]:
        return False
    return all((crop_dir / PREVIEW_DIRNAME / f["file"]).is_file() for f in doc.get("files", []))


# ── Deteksi + saran (dipanggil ingest) ─────────────
def side_text(saran: Mapping, e: str) -> str:
    band, ov = saran["band_px"][e], saran["overlay_px"][e]
    total = saran["cut_px"][e]
    if ov:
        return f"{SIDE_ID[e]} {total} (pita {band} + kotak teks {ov})"
    return f"{SIDE_ID[e]} {total}"


def hint_text(saran: Mapping) -> str | None:
    h = saran.get("hints")
    if not h:
        return None
    parts = []
    for e, (a, b) in h["lines"].items():
        unit = "baris" if e in ("top", "bottom") else "kolom"
        parts.append(f"{SIDE_ID[e]} {b - a + 1} {unit} ({a}–{b})")
    frac = round(saran["hint_static_frac"] * 100)
    return f"petunjuk (oranye di pratinjau): {', '.join(parts)} di dalam kotak ≥ {frac}% diam — mungkin sisa kotak teks / pita"


def saran_lines(saran: Mapping, crop_dir: Path, video_arg: str) -> list[str]:
    """Pesan A+ (persetujuan Tahap 1c e): kotak, path pratinjau, petunjuk + perintah alternatif, --accept / --box / --none."""
    W, H = saran["detect_frame_size"]
    x, y, w, h = saran["box_source"]
    out = [f"[1] saran crop (BELUM dipotong): {' / '.join(side_text(saran, e) for e in EDGES)} px di frame kerja {W} × {H} → "
           f"kotak sumber x {x}, y {y}, {w} × {h}.",
           f"  Periksa pratinjau: {crop_dir / PREVIEW_DIRNAME}"]
    ht = hint_text(saran)
    if ht:
        out.append(f"  {ht}.")
        alt = saran["hints"].get("alt_box_source")
        if alt == saran["box_source"]:
            out.append("  (kotak tanpa baris / kolom petunjuk = kotak saran sesudah pembulatan ke luar + genap — tidak ada perintah lain)")
        elif alt:
            out.append(f"  Kotak tanpa baris / kolom petunjuk: {crop_cmd(video_arg, f'--box {alt[0]} {alt[1]} {alt[2]} {alt[3]}')}")
        else:
            out.append("  (kotak tanpa baris petunjuk < batas area aktif — tidak disarankan)")
    out += [f"  Terima saran: {crop_cmd(video_arg, '--accept')}",
            f"  Kotak sendiri (px sumber; ganti X Y W H): {crop_cmd(video_arg, '--box X Y W H')}",
            f"  Tolak saran (tanpa potong): {crop_cmd(video_arg, '--none')}"]
    return out


def _preview_info(saran: Mapping) -> list[str]:
    x, y, w, h = saran["box_source"]
    info = [f"saran R2: {' / '.join(side_text(saran, e) for e in EDGES)} px; kotak sumber x {x}, y {y}, {w} x {h}"]
    ht = hint_text(saran)
    if ht:
        alt = saran["hints"].get("alt_box_source")
        if alt == saran["box_source"]:
            ht += "; tanpa itu = kotak saran (pembulatan genap)"
        elif alt:
            ht += f"; tanpa itu: --box {alt[0]} {alt[1]} {alt[2]} {alt[3]}"
        info.append(ht)
    return info


def suggest(*, ffmpeg: str, video: Path, identity: Mapping, src_wh: Sequence[int], detect_vf: str, target_fps: float,
            working_width: int, cfg: CropConfig, crop_dir: Path, duration_s: float | None,
            log: Callable[[str], None] = print) -> dict | None:
    """Deteksi R2 pada rentang penuh sumber (D1: dilewati bila deteksi.json berkunci sama). Pita → saran.json + pratinjau, return
    saran; tanpa pita → saran / pratinjau lama dihapus, return None. Tidak pernah menulis kotak.json."""
    params = r2_params(cfg.detect)
    fp = rule_fingerprint(params)
    if fp != FROZEN_FINGERPRINT:
        log(f"PERINGATAN: ingest.crop.detect ≠ R2 beku (sidik {fp[:12]}…, beku {FROZEN_FINGERPRINT[:12]}…) — aturan belum "
            f"divalidasi; saran dari aturan ini ditolak crop --accept")
    key = detection_key(identity, target_fps, working_width, params, ffmpeg_version(ffmpeg))
    old = _read_json(crop_dir / DETEKSI_FILENAME)
    if old is not None and old.get("format") == FORMAT_DETEKSI and old.get("key") == key:
        if not old.get("pita_terdeteksi"):
            _clear_saran(crop_dir)
            return None
        saran = _read_json(crop_dir / SARAN_FILENAME)
        if (saran is not None and saran.get("format") == FORMAT_SARAN and saran.get("key") == key
                and saran.get("hint_static_frac") == cfg.preview.hint_static_frac and _preview_valid(crop_dir, saran)):
            return saran
    n_est = round(duration_s * target_fps) if duration_s else None
    m, sd, n, samples = scan(ffmpeg, video, detect_vf, {0} | ({n_est // 2} if n_est else set()))
    det = detect_crop(m, sd, params)
    W, H = det["frame_wh"]
    crop_dir.mkdir(parents=True, exist_ok=True)
    deteksi = {"format": FORMAT_DETEKSI, "key": key, "n_frames": n, "detect_frame_size": [W, H],
               "pita_terdeteksi": det["box_work"] != [0, 0, W - 1, H - 1], "box_work": det["box_work"]}
    if not deteksi["pita_terdeteksi"]:
        _clear_saran(crop_dir)
        write_json_atomic(crop_dir / DETEKSI_FILENAME, deteksi)
        return None
    box = det["box_work"]
    cut = cut_px(box, W, H)
    hint = hints(sd, box, cut, cfg.preview.hint_static_frac, params)
    if hint is not None:
        hint["alt_box_source"] = (map_box_to_source(hint["alt_box_work"], (W, H), src_wh) if hint["alt_box_work"] else None)
    saran = {"format": FORMAT_SARAN, "key": key, "n_frames": n, "detect_frame_size": [W, H], "box_work": box,
             "box_source": map_box_to_source(box, (W, H), src_wh), "rounding": ROUNDING, "band_px": det["band_px"],
             "overlay_px": det["overlay_px"], "cut_px": cut, "rule": RULE, "rule_version": RULE_VERSION, "rule_fingerprint": fp,
             "hint_static_frac": cfg.preview.hint_static_frac, "hints": hint}
    frames = preview_samples(ffmpeg, video, detect_vf, n, samples, (W, H))
    write_preview(crop_dir, frames, box, hint, "saran", identity, _preview_info(saran))
    write_json_atomic(crop_dir / SARAN_FILENAME, saran)
    write_json_atomic(crop_dir / DETEKSI_FILENAME, deteksi)          # terakhir: deteksi.json valid ⇒ saran + pratinjau lengkap
    return saran


# ── Alat CLI `crop` (BUKAN stage) ──────────────────
def kotak_doc(mode: str, identity: Mapping, box_source: Sequence[int] | None = None, saran: Mapping | None = None,
              requested: Sequence[int] | None = None) -> dict:
    doc = {"format": FORMAT_KOTAK, "mode": mode, "source": dict(identity),
           "box_source": list(box_source) if box_source is not None else None, "rounding": ROUNDING if box_source else None}
    if mode == "saran":
        doc["saran"] = {k: saran[k] for k in ("rule", "rule_version", "rule_fingerprint", "detect_frame_size", "box_work",
                                              "band_px", "overlay_px")}
    if mode == "manual":
        doc["box_requested"] = list(requested)
    return doc


def write_kotak(crop_dir: Path, doc: Mapping) -> bool:
    """Tulis kotak.json; isi lama → kotak.prev.json. Return True bila ada isi lama."""
    crop_dir.mkdir(parents=True, exist_ok=True)
    path = crop_dir / KOTAK_FILENAME
    had = path.is_file()
    if had:
        write_bytes_atomic(crop_dir / KOTAK_PREV_FILENAME, path.read_bytes())
    write_json_atomic(path, doc)
    return had


def describe_kotak(d: Mapping) -> str:
    if d["mode"] == "none":
        return "none (saran ditolak, tanpa potong)"
    x, y, w, h = d["box_source"]
    return f"{d['mode']}: kotak sumber x {x}, y {y}, {w} × {h}"


def run_crop(*, crop_dir: Path, video: Path, cfg: CropConfig, action: str, box: Sequence[int] | None, target_fps: float,
             working_width: int, log: Callable[[str], None] = print) -> bool:
    """`crop <video> [--accept | --box X Y W H | --none | --clear]`; tanpa aksi = status. Return True bila kotak.json berubah."""
    from rotoscope import ingest as ing                    # malas: ingest meng-import modul ini saat dipakai

    probe = ing.probe_video(video)
    identity = source_identity(video, probe)
    src_wh = (identity["width"], identity["height"])
    video_arg = quote_arg(identity["path"])
    saran = _read_json(crop_dir / SARAN_FILENAME)
    if saran is not None and not (saran.get("format") == FORMAT_SARAN and same_source(saran.get("key", {}).get("source"), identity)):
        saran = None                                        # saran milik sumber lain / versi lain: tidak dipakai
    try:
        kotak = read_kotak(crop_dir, video_arg)
    except CropError as e:
        if action != "clear":
            raise
        log(f"PERINGATAN: {e}")
        kotak = None
    if action == "status":
        log(f"crop {video.name}: sumber {identity['width']} × {identity['height']}, sha256 {_short(identity['sha256'])}…; "
            f"berkas di {crop_dir}")
        if saran is not None:
            for line in saran_lines(saran, crop_dir, video_arg):
                log(line)
        else:
            log("  saran: tidak ada (tanpa pita, belum di-ingest, atau ingest.crop.suggest false)")
        if kotak is not None:
            log(f"  kotak.json: {describe_kotak(kotak)}" + ("" if same_source(kotak["source"], identity) else
                                                           " — MILIK SUMBER LAIN (ingest akan menolak; --clear)"))
        else:
            log("  kotak.json: belum ada (belum diputuskan)")
        return False
    if action == "clear":
        path = crop_dir / KOTAK_FILENAME
        if not path.is_file():
            log(f"crop --clear: {path} tidak ada — tidak ada yang diubah")
            return False
        write_bytes_atomic(crop_dir / KOTAK_PREV_FILENAME, path.read_bytes())
        path.unlink()
        log(f"crop --clear: kotak.json → {KOTAK_PREV_FILENAME}; ingest berikutnya TANPA potong"
            + (" (saran masih ada: run akan berhenti sesudah [1] sampai diputuskan)" if saran is not None else ""))
        return True
    if action == "accept":
        if saran is None:
            raise CropError(f"crop --accept: tidak ada saran untuk sumber ini di {crop_dir} — jalankan {CLI_PROG} ingest "
                            f"{video_arg}{config_flag()} dulu (atau pakai --box X Y W H)")
        if saran.get("rule_fingerprint") != FROZEN_FINGERPRINT:
            raise CropError(f"crop --accept: saran dibuat dengan aturan selain R2 beku (sidik {_short(saran.get('rule_fingerprint'))}…) "
                            f"— kembalikan ingest.crop.detect ke nilai beku dan ingest ulang, atau pakai --box X Y W H")
        doc = kotak_doc("saran", identity, saran["box_source"], saran=saran)
    elif action == "none":
        doc = kotak_doc("none", identity)
    elif action == "box":
        src_box = round_manual_box(box, src_wh)
        doc = kotak_doc("manual", identity, src_box, requested=box)
        _manual_preview(crop_dir, video, identity, src_box, box, target_fps, working_width, probe, log)
    else:
        raise ValueError(action)
    had = write_kotak(crop_dir, doc)
    log(f"crop: kotak.json = {describe_kotak(doc)}" + (f" (isi lama → {KOTAK_PREV_FILENAME})" if had else ""))
    return True


def _manual_preview(crop_dir: Path, video: Path, identity: Mapping, src_box: Sequence[int], requested: Sequence[int],
                    target_fps: float, working_width: int, probe: Mapping, log: Callable[[str], None]) -> None:
    from rotoscope import ingest as ing

    ffmpeg = ing._require_binary("ffmpeg")
    out_w, _ = ing.compute_output_size(identity["width"], identity["height"], working_width)
    vf = f"fps={target_fps},scale={out_w}:-2:flags={ing.SCALE_FLAGS}"
    det = _read_json(crop_dir / DETEKSI_FILENAME)
    n = det.get("n_frames") if det and same_source(det.get("key", {}).get("source"), identity) else None
    if n:
        wh = frame_size(ffmpeg, video, vf)
        samples = {}
    else:                                                   # jumlah frame belum diketahui → hitung (decode saja, tanpa statistik)
        _, _, n, samples = scan(ffmpeg, video, vf, {0}, with_stats=False)
        wh = (samples[0].shape[1], samples[0].shape[0])
    box_work = source_box_to_work(src_box, wh, (identity["width"], identity["height"]))
    x, y, w, h = src_box
    info = [f"kotak manual (px sumber) x {x}, y {y}, {w} x {h} (diminta {' '.join(map(str, requested))}; dibulatkan keluar + genap)"]
    frames = preview_samples(ffmpeg, video, vf, n, samples, wh)
    write_preview(crop_dir, frames, box_work, None, "manual", identity, info)
    log(f"  pratinjau kotak manual: {crop_dir / PREVIEW_DIRNAME}")
