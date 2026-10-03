"""Stage [5] stylize (T-203a, CPU): contours/*.json → strokes/frame_%05d.svg + .png (garis polos) + manifest.json.

    python -m rotoscope stylize <video> [--config PATH] [--style PATH] [--restart] [--limit N]

cli.py memanggil main(). TANPA GPU: torch tidak pernah di-import.

Garis polos (T-203a): latar `paper.color`, tinta `stroke.color`, tebal seragam per tipe, solid. Jitter, taper, width
modulation, multipass, tekstur, opasitas dan `shape.resample_points` BELUM aktif (Phase 4; `ignored_params` di manifest).

Satuan panjang style = px REFERENSI lebar 1080 × unit (unit = render.output_width / 1080). Geometri dihitung di px
OUTPUT: titik kontur (pusat piksel kerja) × s, s = output_width / width.

Pipeline geometri per strok (satu geometri untuk SVG dan raster):
  titik × s → [mode hide] run di tepi frame dibuang, strok dipecah jadi jalur terbuka → approxPolyDP → Catmull-Rom
  (uniform; periodik untuk strok tertutup) → ujung di tepi diperpanjang keluar KANVAS → titik dibulatkan 1 desimal.
SVG = string manual (byte-deterministik). Raster = satu mask supersampling (union semua strok) → INTER_AREA → kertas.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import shutil
import sys
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

from rotoscope import vectorize as vz
from rotoscope.config import ConfigError, StyleConfig, ensure_dir, load_pipeline, load_style
from rotoscope.stage_common import (
    CLIP_KEY, EXIT_OK, EXIT_PRECONDITION, StageError, add_work_dir_arg, append_jsonl, clean_tmp, cli_cmd,
    clip_identity, describe_identity, reconfigure_stdio, utc_now, work_dir_overrides, write_bytes_atomic,
    write_json_atomic,
)

# ── Layout output (docs/01 [5]) ────────────────────
STROKES_DIRNAME = "strokes"
MANIFEST_FILENAME = "manifest.json"
FRAMES_LOG_FILENAME = "frames.jsonl"
SVG_SUFFIX = ".svg"
PNG_SUFFIX = ".png"
CONTRACT = "T-203a"
# Naik 1 HANYA untuk perbaikan PERILAKU pada kode yang sudah dikontrak, tanpa perubahan parameter (docs/01). Fitur baru
# (jitter, taper, ...) menaikkan CONTRACT, bukan ALGO_REV.
ALGO_REV = 1
SUPPORTED_CONTOURS_CONTRACTS = frozenset({"T-202"})
DEFAULT_STYLE = Path("configs") / "styles" / "rough-sketch.yaml"
DEFAULT_PIPELINE = Path("configs") / "default.yaml"

# ── Konstanta struktural (bukan parameter style) ───
REF_WIDTH = 1080                  # lebar referensi satuan panjang style
PIXEL_CENTER = 0.5                # titik kontur = pusat piksel (k + 0.5); koordinat cv2 integer = pusat piksel
RASTER_OFFSET = 0.5               # koordinat cv2 = x' · ss − RASTER_OFFSET (titik cv2 integer = pusat piksel)
SHIFT_BITS = 4                    # cv2.polylines sub-piksel 1/16 (koordinat cv2 × 16)
SHIFT_SCALE = 1 << SHIFT_BITS
LINE_TYPE = cv2.LINE_8            # tanpa anti-alias cv2: anti-alias = supersampling + INTER_AREA
# cv2.polylines(thickness=t) hanya menghasilkan lebar ganjil (t genap → t+1, t ganjil → t+2; terukur) → tebal tidak bisa
# eksak. Garis digambar sebagai poligon terisi (kuad per segmen + cakram di tiap titik) dengan koordinat sub-piksel.
FILL_BIAS_SS = 0.5                # cv2.fillPoly mengisi tepi poligon (inklusif): memperlebar ≈ 0,5 piksel ss per sisi (terukur)
CIRCLE_VERTICES = 24              # poligon pendekatan cakram sambungan / ujung
CIRCLE = np.column_stack([np.cos(2 * np.pi * (np.arange(CIRCLE_VERTICES) + 0.5) / CIRCLE_VERTICES),
                          np.sin(2 * np.pi * (np.arange(CIRCLE_VERTICES) + 0.5) / CIRCLE_VERTICES)]
                         ) / np.cos(np.pi / CIRCLE_VERTICES)                         # sisi poligon menyinggung lingkaran jari-jari 1
PNG_COMPRESSION = 3               # cv2.IMWRITE_PNG_COMPRESSION (0–9); deterministik
SVG_DECIMALS = 2                  # titik SVG dan raster dibulatkan 0,01 px output (titik SAMA); galat ≤ 0,007 px
SVG_WIDTH_DECIMALS = 3
EDGE_MARGIN_PX = 1.0              # ekstensi di tepi: melewati kanvas sejauh tebal/2 + margin ini (px output)
SHALLOW_CONTACT_DEG = 45.0        # kontak ke tepi < sudut ini (terhadap garis tepi) → ekstensi tegak lurus tepi
SHALLOW_SIN = math.sin(math.radians(SHALLOW_CONTACT_DEG))
ANGLE_EPS = 1e-9                  # toleransi pembulatan: kontur kisi sering tepat 45°
SPLINE_TOL_REF = 0.03             # galat akor maks polyline vs spline (px ref); + pembulatan 1 desimal (≤ 0,07) ≈ 0,1 px
MAX_SPLINE_STEPS = 512            # batas penggandaan sampling per segmen
MIN_PIECE_POINTS = 2
# Penghalusan Gaussian arc-length sebelum approxPolyDP (keputusan Rio; konstanta struktural, bukan parameter)
SMOOTH_STEP_REF = 1.0             # jarak re-sample (px ref) sebelum penghalusan
CORNER_DEG = 60.0                 # sudut tajam: belok > ini pada jendela ±CORNER_WINDOW sampel → dikunci
CORNER_WINDOW = 6                 # sampel (SMOOTH_STEP_REF) tiap sisi untuk sudut belok
CORNER_DETECT_SIGMA_REF = 1.5     # penghalusan ringan HANYA untuk mendeteksi sudut (px ref)
DETECT_SIGMA_SAMPLES = CORNER_DETECT_SIGMA_REF / SMOOTH_STEP_REF
GAUSS_TRUNCATE = 3.0              # radius kernel Gaussian dalam sigma
MIN_SMOOTH_POINTS = 4
MASK_LEVELS = 256                 # mask / cakupan uint8
MASK_PAD_PX = 4                   # tambahan pad mask di luar kanvas (px output) selain tebal maks
ONLY_CAP = "round"                # T-203a hanya cap bulat
TYPE_ORDER = vz.STROKE_TYPES      # urutan <g> di SVG = urutan strok [4]

# Parameter style AKTIF di T-203a (masuk hash); sisanya = ignored_params.
ACTIVE_SCALARS = ("shape.simplify_epsilon", "shape.smooth_px", "shape.smooth_tension", "shape.spline_steps", "shape.edge_mode",
                  "stroke.width_base", "stroke.color", "stroke.cap", "paper.color", "render.ss", "render.output_width")
ACTIVE_BY_TYPE_PREFIX = "stroke.by_type."
ACTIVE_BY_TYPE_SUFFIX = ".width_scale"
MANIFEST_MATCH_KEYS = ("contract", "algo_rev", "style_hash", "style_params", "contours", "clip", "frame_size",
                       "output_size", "scale", "unit")
COORDS_INFO = {"space": "output pixels", "origin": "top-left corner of pixel (0, 0)",
               "mapping": "x' = s * x (x = pixel center k + 0.5 of the working frame), s = output_width / width",
               "raster": "cv2 coordinate = x' * ss - 0.5 (integer = pixel center), shift 4"}

BOTTOM, LEFT, RIGHT = "bottom", "left", "right"
OUTWARD = {BOTTOM: np.array([0.0, 1.0]), LEFT: np.array([-1.0, 0.0]), RIGHT: np.array([1.0, 0.0])}


# ── Parameter style ────────────────────────────────
def _flatten(obj, prefix: str = "") -> dict:
    """Style → {nama bertitik: nilai} (dataclass / Mapping dirata; Path → posix)."""
    out: dict = {}
    if dataclasses.is_dataclass(obj):
        for f in dataclasses.fields(obj):
            out.update(_flatten(getattr(obj, f.name), f"{prefix}{f.name}."))
    elif isinstance(obj, Mapping):
        for k, v in obj.items():
            out.update(_flatten(v, f"{prefix}{k}."))
    else:
        out[prefix[:-1]] = obj.as_posix() if isinstance(obj, Path) else obj
    return out


def active_param_names() -> tuple[str, ...]:
    return ACTIVE_SCALARS + tuple(f"{ACTIVE_BY_TYPE_PREFIX}{t}{ACTIVE_BY_TYPE_SUFFIX}" for t in TYPE_ORDER)


def style_params(style: StyleConfig) -> tuple[dict, list[str], dict]:
    """(parameter aktif, nama parameter yang diabaikan, nilai non-default dari yang diabaikan)."""
    flat, default = _flatten(style), _flatten(load_style(None))     # default ter-resolve (path aset absolut) seperti `style`
    active_names = active_param_names()
    active = {k: flat[k] for k in active_names}
    ignored = sorted(k for k in flat if k not in active)
    return active, ignored, {k: flat[k] for k in ignored if flat[k] != default.get(k)}


def params_hash(active: dict) -> str:
    return hashlib.sha256(json.dumps(active, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def hex_rgb(color: str) -> tuple[int, int, int]:
    c = color.lstrip("#")
    return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)


# ── Ukuran + geometri ──────────────────────────────
def output_size(width: int, height: int, output_width: int) -> tuple[int, int]:
    """(lebar, tinggi) output: tinggi = round(height × ow / width) (setengah naik, integer) dinaikkan ke genap."""
    h = (2 * height * output_width + width) // (2 * width)
    return output_width, h + (h % 2)


@dataclass(frozen=True)
class Geometry:
    """Semua parameter geometri + render dalam px output."""
    width: int                    # lebar kerja
    height: int                   # tinggi kerja
    out_w: int
    out_h: int
    scale: float                  # s = out_w / width
    unit: float                   # out_w / REF_WIDTH
    epsilon: float
    smooth: float                 # sigma penghalusan (px output); 0 = mati
    tension: float
    steps: int
    spline_tol: float             # galat akor spline (px output)
    edge_mode: str
    ss: int
    widths: Mapping[str, float]   # tebal per tipe (px output)
    ink: tuple[int, int, int]
    paper: tuple[int, int, int]


def make_geometry(style: StyleConfig, width: int, height: int) -> Geometry:
    if style.stroke.cap != ONLY_CAP:
        raise StageError(f"stroke.cap = {style.stroke.cap!r} belum didukung di T-203a (hanya {ONLY_CAP!r})")
    ow = style.render.output_width
    out_w, out_h = output_size(width, height, ow)
    unit = ow / REF_WIDTH
    widths = {t: style.stroke.width_base * style.stroke.by_type[t].width_scale * unit for t in TYPE_ORDER}
    return Geometry(width=width, height=height, out_w=out_w, out_h=out_h, scale=ow / width, unit=unit,
                    epsilon=style.shape.simplify_epsilon * unit, smooth=style.shape.smooth_px * unit, tension=style.shape.smooth_tension,
                    steps=style.shape.spline_steps, spline_tol=SPLINE_TOL_REF * unit,
                    edge_mode=style.shape.edge_mode, ss=style.render.ss, widths=widths,
                    ink=hex_rgb(style.stroke.color), paper=hex_rgb(style.paper.color))


@dataclass(frozen=True)
class Piece:
    """Satu jalur yang digambar (px output, sudah dibulatkan 1 desimal)."""
    type: str
    closed: bool
    points: np.ndarray            # (N, 2)
    track_id: int


def edge_sides(p: np.ndarray, width: int, height: int) -> list[list[str]]:
    """Sisi tepi yang disentuh tiap titik (koordinat PERSIS: y = H − 0,5, x = 0,5, x = W − 0,5), dari koordinat kerja."""
    out = []
    for x, y in p:
        s = []
        if y == height - PIXEL_CENTER:
            s.append(BOTTOM)
        if x == PIXEL_CENTER:
            s.append(LEFT)
        if x == width - PIXEL_CENTER:
            s.append(RIGHT)
        out.append(s)
    return out


def split_edge_runs(on: np.ndarray, closed: bool) -> tuple[list[tuple[list[int], bool, bool]], list[list[int]]]:
    """Pecah titik per run tepi. Return (jalur luar-tepi, run tepi).

    Jalur = (indeks, titik_pertama_di_tepi, titik_terakhir_di_tepi): run titik bukan-tepi maksimal ditambah titik
    tepi tetangga (titik silang) di tiap sisi bila ada. Strok tertutup ditelusuri siklik mulai dari awal sebuah run tepi."""
    n = len(on)
    if closed:
        start = next(i for i in range(n) if on[i] and not on[i - 1])
        order = [(start + i) % n for i in range(n)]
    else:
        order = list(range(n))
    off_runs: list[list[int]] = []
    on_runs: list[list[int]] = []
    cur: list[int] = []
    cur_on = None
    for i in order:
        if cur and bool(on[i]) != cur_on:
            (on_runs if cur_on else off_runs).append(cur)
            cur = []
        cur_on = bool(on[i])
        cur.append(i)
    if cur:
        (on_runs if cur_on else off_runs).append(cur)
    pieces = []
    for r in off_runs:
        before = (r[0] - 1) % n if (closed or r[0] > 0) else None
        after = (r[-1] + 1) % n if (closed or r[-1] < n - 1) else None
        idx = ([before] if before is not None else []) + r + ([after] if after is not None else [])
        pieces.append((idx, before is not None, after is not None))
    return pieces, on_runs


def hermite(p0: np.ndarray, p1: np.ndarray, m0: np.ndarray, m1: np.ndarray, t: np.ndarray) -> np.ndarray:
    t = t[:, None]
    return ((2 * t**3 - 3 * t**2 + 1) * p0 + (t**3 - 2 * t**2 + t) * m0 + (-2 * t**3 + 3 * t**2) * p1
            + (t**3 - t**2) * m1)


def chord_error(p0, p1, m0, m1, k: int) -> float:
    """Jarak maks titik tengah tiap sub-segmen (k sub-segmen seragam) ke akor-nya: ukuran galat polyline vs kurva."""
    t = np.arange(k + 1) / k
    pts = hermite(p0, p1, m0, m1, t)
    mid = hermite(p0, p1, m0, m1, (np.arange(k) + 0.5) / k)
    a, ab = pts[:-1], pts[1:] - pts[:-1]
    l2 = np.maximum((ab**2).sum(1), 1e-18)
    s = np.clip(((mid - a) * ab).sum(1) / l2, 0, 1)
    return float(np.sqrt(((a + ab * s[:, None] - mid) ** 2).sum(1)).max())


def catmull_rom(p: np.ndarray, tension: float, steps: int, tol: float, closed: bool) -> np.ndarray:
    """Catmull-Rom seragam: tangen m_i = tension × (p[i+1] − p[i−1]); ujung terbuka = titik ujung digandakan, tertutup =
    periodik (tanpa takik; titik akhir tidak mengulang titik awal). Tiap segmen disampel dengan `steps` titik, digandakan
    sampai galat akor (chord_error) ≤ `tol` (px output) atau MAX_SPLINE_STEPS."""
    n = len(p)
    if n < 3:
        return p.copy()
    if closed:
        m = tension * (np.roll(p, -1, axis=0) - np.roll(p, 1, axis=0))
        nxt = np.roll(p, -1, axis=0)
        mn = np.roll(m, -1, axis=0)
        count = n
    else:
        ext = np.vstack([p[:1], p, p[-1:]])
        m = tension * (ext[2:] - ext[:-2])
        nxt, mn = p[1:], m[1:]
        count = n - 1
    out = []
    for i in range(count):
        k = steps
        while k < MAX_SPLINE_STEPS and chord_error(p[i], nxt[i], m[i], mn[i], k) > tol:
            k *= 2
        out.append(hermite(p[i], nxt[i], m[i], mn[i], np.arange(k) / k))
    if not closed:
        out.append(p[-1:])
    return np.vstack(out)


# ── Penghalusan sebelum approxPolyDP ───────────────
def arc_resample(p: np.ndarray, closed: bool, step: float) -> np.ndarray:
    """Re-sample jarak sama (arc-length). Terbuka: titik pertama dan terakhir = titik asli; tertutup: tanpa titik ganda."""
    q = np.vstack([p, p[:1]]) if closed else p
    s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(q, axis=0).T))]
    length = float(s[-1])
    if length <= 0:
        return p.copy()
    n = max(int(round(length / step)), 3 if closed else 1)
    t = np.arange(n) * (length / n) if closed else np.linspace(0.0, length, n + 1)
    return np.column_stack([np.interp(t, s, q[:, 0]), np.interp(t, s, q[:, 1])])


def gauss_open_fixed(seg: np.ndarray, sigma_samples: float) -> np.ndarray:
    """Gaussian jalur terbuka; kedua ujung dikunci lewat pantulan ganjil (2·ujung − titik): ujung tidak bergeser."""
    n = len(seg)
    k = min(int(math.ceil(GAUSS_TRUNCATE * sigma_samples)), n - 1)
    if n < 3 or k < 1:
        return seg
    ext = np.vstack([2 * seg[0] - seg[k:0:-1], seg, 2 * seg[-1] - seg[-2:-k - 2:-1]])
    out = gaussian_filter1d(ext, sigma_samples, axis=0, mode="nearest", truncate=GAUSS_TRUNCATE)[k:k + n]
    out[0], out[-1] = seg[0], seg[-1]
    return out


def find_corners(r: np.ndarray, closed: bool, detect_sigma_samples: float) -> list[int]:
    """Indeks sudut tajam pada kurva ter-resample: sudut antara arah masuk dan keluar (±CORNER_WINDOW sampel, setelah
    penghalusan ringan) > CORNER_DEG dan maksimum lokal."""
    n, w = len(r), CORNER_WINDOW
    if n < 2 * w + 3:
        return []
    if closed:
        s = gaussian_filter1d(r, detect_sigma_samples, axis=0, mode="wrap", truncate=GAUSS_TRUNCATE)
        a, b, idx = s - np.roll(s, w, axis=0), np.roll(s, -w, axis=0) - s, np.arange(n)
    else:
        s = gauss_open_fixed(r, detect_sigma_samples)
        a, b, idx = s[w:-w] - s[:-2 * w], s[2 * w:] - s[w:-w], np.arange(w, n - w)
    cosang = np.clip((a * b).sum(1) / np.maximum(np.hypot(*a.T) * np.hypot(*b.T), 1e-12), -1, 1)
    ang = np.degrees(np.arccos(cosang))
    out: list[int] = []
    for j in np.flatnonzero(ang > CORNER_DEG):
        win = ang[np.arange(j - w, j + w + 1) % len(ang)] if closed else ang[max(j - w, 0):j + w + 1]
        if ang[j] >= win.max() and (not out or int(idx[j]) - out[-1] > w):
            out.append(int(idx[j]))
    return out


def smooth_polyline(p: np.ndarray, closed: bool, sigma: float, step: float) -> np.ndarray:
    """Gaussian arc-length (sigma, step dalam px output). Ujung jalur terbuka tidak bergeser; sudut tajam dikunci; tertutup
    periodik tanpa takik. Mengembalikan polyline rapat (jarak ≈ step); tertutup tidak mengulang titik awal."""
    if sigma <= 0 or len(p) < MIN_SMOOTH_POINTS:
        return p
    r = arc_resample(p, closed, step)
    if len(r) < MIN_SMOOTH_POINTS:
        return p
    sig = sigma / step
    corners = find_corners(r, closed, DETECT_SIGMA_SAMPLES)
    if not closed:
        pins = sorted({0, len(r) - 1, *corners})
        out = np.vstack([gauss_open_fixed(r[a:b + 1], sig) if k == 0 else gauss_open_fixed(r[a:b + 1], sig)[1:]
                         for k, (a, b) in enumerate(zip(pins[:-1], pins[1:]))])
        out[0], out[-1] = p[0], p[-1]
        return out
    if not corners:
        return gaussian_filter1d(r, sig, axis=0, mode="wrap", truncate=GAUSS_TRUNCATE)
    c0 = corners[0]
    rr = np.roll(r, -c0, axis=0)
    rr = np.vstack([rr, rr[:1]])
    pins = sorted({0, len(rr) - 1, *((c - c0) % len(r) for c in corners)})
    out = np.vstack([gauss_open_fixed(rr[a:b + 1], sig) if k == 0 else gauss_open_fixed(rr[a:b + 1], sig)[1:]
                     for k, (a, b) in enumerate(zip(pins[:-1], pins[1:]))])
    return out[:-1]


def simplify(p: np.ndarray, epsilon: float, closed: bool) -> np.ndarray:
    if len(p) < 3 or epsilon <= 0:
        return p
    a = cv2.approxPolyDP(p.astype(np.float32).reshape(-1, 1, 2), epsilon, closed).reshape(-1, 2).astype(np.float64)
    return a if len(a) >= (3 if closed else MIN_PIECE_POINTS) else p


def extension_point(end: np.ndarray, tangent: np.ndarray, sides: list[str], g: Geometry, width_px: float
                    ) -> tuple[np.ndarray, dict]:
    """Titik di LUAR kanvas untuk ujung jalur yang berhenti di tepi (titik silang `end`, px output).

    Arah = arah ujung spline (`tangent`, searah keluar jalur) bila membentuk ≥ SHALLOW_CONTACT_DEG terhadap setidaknya satu
    garis tepi yang disentuh (komponen keluar ≥ sin 45°); kalau tidak (kontak dangkal) → tegak lurus tepi (jumlah normal
    untuk sudut kanvas). Panjang: melewati kanvas (bukan bingkai konten) sejauh tebal/2 + EDGE_MARGIN_PX di setiap tepi yang
    disentuh dan dilintasi arah itu."""
    margin = width_px / 2 + EDGE_MARGIN_PX
    dist = {BOTTOM: g.out_h - end[1], LEFT: end[0], RIGHT: g.out_w - end[0]}
    norm = float(np.hypot(*tangent))
    d = tangent / norm if norm > 0 else None
    info = {"shallow": False, "corner": len(sides) > 1, "angle_deg": None}
    if d is not None and sides:
        info["angle_deg"] = min(math.degrees(math.asin(max(-1.0, min(1.0, float(d @ OUTWARD[s]))))) for s in sides)
    if d is not None and any(float(d @ OUTWARD[s]) >= SHALLOW_SIN - ANGLE_EPS for s in sides):
        direction = d
    else:
        direction = sum(OUTWARD[s] for s in sides)
        direction = direction / float(np.hypot(*direction))
        info["shallow"] = d is not None
    t = max((dist[s] + margin) / float(direction @ OUTWARD[s]) for s in sides if float(direction @ OUTWARD[s]) > 1e-9)
    return end + t * direction, info


def stroke_pieces(stroke: dict, g: Geometry, stats: dict) -> list[Piece]:
    """Satu strok kontur → jalur yang digambar (satu geometri untuk SVG dan raster)."""
    p = np.asarray(stroke["points"], dtype=np.float64)
    closed = bool(stroke["closed"])
    if not closed and len(p) > 3 and np.array_equal(p[0], p[-1]):     # loop: titik akhir = titik awal
        closed, p = True, p[:-1]
    typ, tid, wpx = stroke["type"], int(stroke["track_id"]), g.widths[stroke["type"]]
    smooth_step = SMOOTH_STEP_REF * g.unit
    sides = edge_sides(p, g.width, g.height)
    on = np.array([bool(s) for s in sides])
    raw: list[tuple[np.ndarray, bool, list[list[str]], bool, bool, str]] = []   # (titik px output, closed, sisi, ext_awal, ext_akhir, kind)
    if not on.any():
        raw.append((p * g.scale, closed, sides, False, False, "curve"))
    elif on.all():
        stats["dropped_all_edge"] += 1
        if g.edge_mode == "draw":
            raw.append((p * g.scale, closed, sides, False, False, "run"))
    else:
        pieces, on_runs = split_edge_runs(on, closed)
        stats["edge_cuts"] += len(pieces)
        for idx, c0, c1 in pieces:
            raw.append((p[idx] * g.scale, False, [sides[i] for i in idx], c0 and g.edge_mode == "hide",
                        c1 and g.edge_mode == "hide", "curve"))
        if g.edge_mode == "draw":
            for r in on_runs:
                raw.append((p[r] * g.scale, False, [sides[i] for i in r], False, False, "run"))
    out: list[Piece] = []
    for pts, cl, sd, ext0, ext1, kind in raw:
        if len(pts) < MIN_PIECE_POINTS:
            continue
        if kind == "curve":
            pts = smooth_polyline(pts, cl, g.smooth, smooth_step)     # ujung (titik silang tepi) tidak bergeser
        a = simplify(pts, g.epsilon, cl)
        # simplify tidak menyimpan sisi per titik → ujung (indeks 0 / −1) tetap titik tepi asli
        curve = a if kind == "run" else catmull_rom(a, g.tension, g.steps, g.spline_tol, cl)
        if not cl and (ext0 or ext1):
            # tangen ujung = akor terakhir polyline sederhana (turunan Hermite di ujung ∝ akor itu; tidak bergantung
            # pada kerapatan sampling spline)
            head, tail = [], []
            if ext0:
                q, info = extension_point(a[0], a[0] - a[1], sd[0], g, wpx)
                head.append(q)
                _count_contact(stats, info)
            if ext1:
                q, info = extension_point(a[-1], a[-1] - a[-2], sd[-1], g, wpx)
                tail.append(q)
                _count_contact(stats, info)
            curve = np.vstack(head + [curve] + tail)
        out.append(Piece(typ, cl, np.round(curve, SVG_DECIMALS) + 0.0, tid))
    return out


def _count_contact(stats: dict, info: dict) -> None:
    stats["edge_ends"] += 1
    stats["shallow_ends"] += int(info["shallow"])
    stats["corner_ends"] += int(info["corner"])
    if info["angle_deg"] is not None:
        stats["min_contact_deg"] = min(stats["min_contact_deg"], info["angle_deg"])


def draw_mode_min_width(g: Geometry) -> float:
    """Tebal minimum (px output) agar strok terbuka yang berakhir di tepi frame pada mode `draw` tetap menyentuh tepi kanvas:
    titik terakhir berjarak 0,5 · s dari tepi kanvas, jari-jari ujung bulat = w / 2 → w ≥ s. (Mode `hide` memperpanjang
    ujungnya keluar kanvas, jadi tidak bergantung pada tebal.)"""
    return g.scale


def draw_width_warning(g: Geometry) -> str | None:
    """Peringatan (bukan error) bila mode `draw` dan ada tipe bertebal < draw_mode_min_width: strok terbuka yang berakhir di
    tepi frame tidak akan menyentuh tepi kanvas. None bila tidak berlaku."""
    if g.edge_mode != "draw":
        return None
    need = draw_mode_min_width(g)
    thin = {t: w for t, w in g.widths.items() if w < need}
    if not thin:
        return None
    return ("PERINGATAN: shape.edge_mode = 'draw' dengan tebal terlalu kecil — "
            + ", ".join(f"{t} {w:.2f} px" for t, w in thin.items())
            + f" < syarat minimum {need:.2f} px (= s = output_width / lebar kerja; jari-jari ujung w / 2 harus ≥ 0,5 · s). "
              "Strok terbuka yang berakhir di tepi frame menyisakan celah ke tepi kanvas; naikkan stroke.width_base / "
              "width_scale atau pakai edge_mode 'hide'.")


def new_stats() -> dict:
    return {"dropped_all_edge": 0, "edge_cuts": 0, "edge_ends": 0, "shallow_ends": 0, "corner_ends": 0,
            "min_contact_deg": 90.0}


def frame_pieces(doc: dict, g: Geometry) -> tuple[list[Piece], dict]:
    stats = new_stats()
    pieces: list[Piece] = []
    for s in doc["strokes"]:
        pieces += stroke_pieces(s, g, stats)
    return pieces, stats


# ── Render ─────────────────────────────────────────
def render_mask(pieces: list[Piece], g: Geometry) -> np.ndarray:
    """Satu mask supersampling (union semua strok, tipe dan tebal berbeda). Koordinat cv2 = x' · ss − RASTER_OFFSET (titik
    cv2 integer = pusat piksel supersampling; titik kontinu x' · ss = k + 0,5 adalah pusatnya), sub-piksel 1/16."""
    pad = int(math.ceil(max(g.widths.values()))) + MASK_PAD_PX          # kanvas diperlebar: poligon tidak terpotong di tepi mask
    mask = np.zeros(((g.out_h + 2 * pad) * g.ss, (g.out_w + 2 * pad) * g.ss), np.uint8)
    quads: list[np.ndarray] = []
    discs: list[np.ndarray] = []
    for pc in pieces:
        p = (pc.points + pad) * g.ss - RASTER_OFFSET
        r = max(g.widths[pc.type] * g.ss / 2 - FILL_BIAS_SS, 0.0)
        q = np.vstack([p, p[:1]]) if pc.closed else p
        d = np.diff(q, axis=0)
        length = np.hypot(d[:, 0], d[:, 1])
        keep = length > 0
        n = np.column_stack([-d[:, 1], d[:, 0]])[keep] / length[keep, None] * r
        a, b = q[:-1][keep], q[1:][keep]
        quads.append(np.stack([a + n, b + n, b - n, a - n], axis=1))
        discs.append(p[:, None, :] + r * CIRCLE[None, :, :])                      # sambungan + ujung bulat (cap round)
    if quads:
        # fillConvexPoly per poligon: satu panggilan fillPoly memakai aturan genap-ganjil, jadi poligon yang tumpang
        # tindih (kuad bertetangga, cakram) menjadi LUBANG; union butuh panggilan terpisah.
        for polys in (np.concatenate(quads), np.concatenate(discs)):
            for poly in np.rint(polys * SHIFT_SCALE).astype(np.int32):
                cv2.fillConvexPoly(mask, poly, 255, LINE_TYPE, SHIFT_BITS)
    return np.ascontiguousarray(mask[pad * g.ss:(pad + g.out_h) * g.ss, pad * g.ss:(pad + g.out_w) * g.ss])


def compose(mask: np.ndarray, g: Geometry) -> np.ndarray:
    """mask → cakupan (INTER_AREA) → ink di atas kertas, RGB uint8 (out_h, out_w, 3)."""
    cov = cv2.resize(mask, (g.out_w, g.out_h), interpolation=cv2.INTER_AREA)
    a = (np.arange(MASK_LEVELS, dtype=np.float32) / (MASK_LEVELS - 1))[:, None]       # tabel: cakupan uint8 → warna
    lut = np.rint(np.array(g.paper, np.float32) * (1 - a) + np.array(g.ink, np.float32) * a).astype(np.uint8)
    return lut[cov]


def render_png(pieces: list[Piece], g: Geometry) -> bytes:
    ok, buf = cv2.imencode(PNG_SUFFIX, cv2.cvtColor(compose(render_mask(pieces, g), g), cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION])
    if not ok:
        raise StageError("cv2.imencode PNG gagal")
    return buf.tobytes()


def _num(v: float) -> str:
    return f"{v:.{SVG_DECIMALS}f}"


def render_svg(pieces: list[Piece], g: Geometry, style: StyleConfig) -> bytes:
    """SVG manual: <g id=tipe> per tipe (urutan tetap), satu <path> per jalur, titik yang SAMA dengan raster."""
    head = (f'<?xml version="1.0" encoding="UTF-8"?>\n<svg xmlns="http://www.w3.org/2000/svg" width="{g.out_w}" '
            f'height="{g.out_h}" viewBox="0 0 {g.out_w} {g.out_h}">\n'
            f'<rect width="{g.out_w}" height="{g.out_h}" fill="{style.paper.color}"/>\n')
    parts = [head]
    for t in TYPE_ORDER:
        parts.append(f'<g id="{t}" fill="none" stroke="{style.stroke.color}" stroke-linecap="round" '
                     f'stroke-linejoin="round" stroke-width="{g.widths[t]:.{SVG_WIDTH_DECIMALS}f}">\n')
        for pc in pieces:
            if pc.type != t:
                continue
            d = "M" + " L".join(f"{_num(x)} {_num(y)}" for x, y in pc.points) + (" Z" if pc.closed else "")
            parts.append(f'<path d="{d}"/>\n')
        parts.append("</g>\n")
    parts.append("</svg>\n")
    return "".join(parts).encode("utf-8")


def render_frame(doc: dict, g: Geometry, style: StyleConfig) -> tuple[bytes, bytes, dict]:
    t0 = time.perf_counter()
    pieces, stats = frame_pieces(doc, g)
    t1 = time.perf_counter()
    png = render_png(pieces, g)
    t2 = time.perf_counter()
    svg = render_svg(pieces, g, style)
    t3 = time.perf_counter()
    stats.update({"n_strokes": len(doc["strokes"]), "n_pieces": len(pieces),
                  "n_points": int(sum(len(p.points) for p in pieces)), "geom_s": round(t1 - t0, 4),
                  "raster_png_s": round(t2 - t1, 4), "svg_s": round(t3 - t2, 4),
                  "svg_bytes": len(svg), "png_bytes": len(png), "min_contact_deg": round(stats["min_contact_deg"], 1)})
    return svg, png, stats


# ── Layout klip ────────────────────────────────────
@dataclass(frozen=True)
class Clip:
    work_dir: Path
    names: tuple[str, ...]
    indices: tuple[int, ...]
    width: int
    height: int

    @property
    def contours(self) -> vz.Clip:
        return vz.Clip(self.work_dir, self.names, self.indices, self.width, self.height)

    @property
    def strokes_dir(self) -> Path:
        return self.work_dir / STROKES_DIRNAME

    @property
    def manifest_path(self) -> Path:
        return self.strokes_dir / MANIFEST_FILENAME

    @property
    def frames_log(self) -> Path:
        return self.strokes_dir / FRAMES_LOG_FILENAME

    def svg_path(self, name: str) -> Path:
        return self.strokes_dir / (Path(name).stem + SVG_SUFFIX)

    def png_path(self, name: str) -> Path:
        return self.strokes_dir / (Path(name).stem + PNG_SUFFIX)


def load_clip(work_dir: Path) -> Clip:
    c = vz.load_clip(work_dir)
    return Clip(c.work_dir, c.names, c.indices, c.width, c.height)


# ── Input [4] ──────────────────────────────────────
def load_contours_manifest(clip: Clip) -> dict:
    """contours/manifest.json dicek: contract didukung, pending kosong, identitas klip, frame_size. Gagal → StageError."""
    cmd = cli_cmd("vectorize", clip.work_dir)
    path = clip.contours.manifest_path
    if not path.is_file():
        raise StageError(f"{path} tidak ada — jalankan stage [4] dulu: {cmd}")
    try:
        m = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise StageError(f"gagal membaca {path} ({e}) — jalankan ulang stage [4]: {cmd}") from None
    if not isinstance(m, dict):
        raise StageError(f"{path} bukan objek JSON — jalankan ulang stage [4]: {cmd}")
    if m.get("contract") not in SUPPORTED_CONTOURS_CONTRACTS:
        raise StageError(f"contours/manifest.json: contract {m.get('contract')!r} tidak didukung (didukung: "
                         f"{', '.join(sorted(SUPPORTED_CONTOURS_CONTRACTS))}) — jalankan ulang stage [4]: {cmd} "
                         f"(output lama dihitung ulang otomatis)")
    if m.get("pending"):
        raise StageError(f"contours/manifest.json: pending tidak kosong ({m.get('pending')}) — output [4] belum lengkap; "
                         f"jalankan ulang stage [4]: {cmd}")
    current = clip_identity(clip.work_dir)
    old = m.get(CLIP_KEY)
    if not isinstance(old, dict) or not old.get("meta_sha256"):
        raise StageError(f"contours/manifest.json tidak memuat identitas klip (manifest lama) — jalankan ulang stage "
                         f"[4]: {cmd}")
    if old["meta_sha256"] != current["meta_sha256"]:
        raise StageError(f"contours/manifest.json milik klip LAIN: output [4] = {describe_identity(old)}, meta.json "
                         f"saat ini = {describe_identity(current)}. Jalankan ulang stage [4]: {cmd}")
    size = {"width": clip.width, "height": clip.height}
    if m.get("frame_size") != size:
        raise StageError(f"contours/manifest.json: frame_size {m.get('frame_size')} ≠ meta.json {size} — jalankan ulang "
                         f"stage [4]: {cmd}")
    for key in ("vectorize_hash", "created_utc"):
        if not isinstance(m.get(key), str):
            raise StageError(f"contours/manifest.json tidak memuat {key} — jalankan ulang stage [4]: {cmd}")
    if not isinstance(m.get("algo_rev"), int):
        raise StageError(f"contours/manifest.json tidak memuat algo_rev — jalankan ulang stage [4]: {cmd}")
    return m


def read_contours_frame(clip: Clip, name: str, index: int, vectorize_hash: str) -> dict | None:
    """Dokumen frame contours bila valid (frame_index, ukuran, source.vectorize_hash cocok, strok final), selain itu None."""
    try:
        d = json.loads(clip.contours.frame_path(name).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    ok = (isinstance(d, dict) and d.get("frame_index") == index and d.get("width") == clip.width
          and d.get("height") == clip.height and isinstance(d.get("strokes"), list)
          and isinstance(d.get("source"), dict) and d["source"].get("vectorize_hash") == vectorize_hash
          and vz.strokes_final(d["strokes"]))
    return d if ok else None


def require_contours(clip: Clip, names: tuple[str, ...], indices, vectorize_hash: str) -> None:
    bad = [n for n, i in zip(names, indices) if read_contours_frame(clip, n, i, vectorize_hash) is None]
    if bad:
        raise StageError(f"contours belum lengkap / rusak: {len(bad)} dari {len(names)} frame, mis. {Path(bad[0]).stem} "
                         f"— jalankan stage [4]: {cli_cmd('vectorize', clip.work_dir)}")


# ── Manifest ───────────────────────────────────────
def build_manifest(style: StyleConfig, style_name: str, g: Geometry, clip: Clip, cm: dict) -> dict:
    active, ignored, _ = style_params(style)
    return {"stage": "stylize", "contract": CONTRACT, "algo_rev": ALGO_REV, "style": style_name,
            "style_hash": params_hash(active), "style_params": active, "ignored_params": ignored,
            "contours": {"contract": cm["contract"], "vectorize_hash": cm["vectorize_hash"],
                         "algo_rev": cm["algo_rev"], "created_utc": cm["created_utc"]},
            "frame_size": {"width": clip.width, "height": clip.height}, CLIP_KEY: clip_identity(clip.work_dir),
            "output_width": g.out_w, "output_size": {"width": g.out_w, "height": g.out_h}, "scale": g.scale,
            "unit": g.unit, "edge_mode": g.edge_mode, "coords": dict(COORDS_INFO), "created_utc": utc_now()}


def _short(v) -> str:
    return v[:12] if isinstance(v, str) and len(v) > 12 else repr(v)


def manifest_diff(old: dict, new: dict) -> list[str]:
    out = []
    for k in MANIFEST_MATCH_KEYS:
        a, b = old.get(k), new.get(k)
        if a == b:
            continue
        if isinstance(b, dict) and not isinstance(a, dict):
            a = {}
        if isinstance(a, dict) and isinstance(b, dict):
            out += [f"{k}.{s}: {_short(a.get(s))} → {_short(b.get(s))}"
                    for s in sorted(set(a) | set(b)) if a.get(s) != b.get(s)]
        else:
            out.append(f"{k}: {_short(a)} → {_short(b)}")
    return out


def _has_outputs(clip: Clip) -> bool:
    return any(clip.strokes_dir.glob("frame_*" + SVG_SUFFIX)) or any(clip.strokes_dir.glob("frame_*" + PNG_SUFFIX))


def restart_outputs(clip: Clip) -> None:
    """Hapus output [5] (strokes/ saja)."""
    if clip.strokes_dir.exists():
        shutil.rmtree(clip.strokes_dir)


# ── Validitas frame ────────────────────────────────
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
PNG_IEND = b"IEND\xaeB`\x82"
PNG_HEADER_LEN = 24


def png_valid(path: Path, size: tuple[int, int]) -> bool:
    """Header PNG (signature + IHDR) dan trailer IEND utuh, ukuran = ukuran output (tanpa decode penuh)."""
    try:
        data = path.read_bytes()
    except OSError:
        return False
    if len(data) < PNG_HEADER_LEN + len(PNG_IEND) or not data.startswith(PNG_SIGNATURE) or not data.endswith(PNG_IEND):
        return False
    return (int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")) == size


def svg_valid(path: Path, size: tuple[int, int]) -> bool:
    """XML well-formed dengan viewBox / width / height = ukuran output."""
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError):
        return False
    return (root.tag.endswith("svg") and root.get("viewBox") == f"0 0 {size[0]} {size[1]}"
            and root.get("width") == str(size[0]) and root.get("height") == str(size[1]))


def frame_valid(clip: Clip, name: str, size: tuple[int, int]) -> bool:
    return svg_valid(clip.svg_path(name), size) and png_valid(clip.png_path(name), size)


# ── Run ────────────────────────────────────────────
def run_stylize(cfg, style: StyleConfig, style_name: str = "default", *, restart: bool = False,
                limit: int | None = None, log: Callable[[str], None] = print) -> dict:
    """Jalankan stage [5] (T-203a). Return ringkasan run."""
    if limit is not None and limit < 1:
        raise StageError(f"--limit harus ≥ 1, dapat {limit}")
    clip = load_clip(cfg.paths.work_dir)
    selected = clip.names[:limit] if limit else clip.names
    indices = clip.indices[:len(selected)]
    cm = load_contours_manifest(clip)
    require_contours(clip, selected, indices, cm["vectorize_hash"])
    g = make_geometry(style, clip.width, clip.height)
    manifest = build_manifest(style, style_name, g, clip, cm)
    if (warning := draw_width_warning(g)) is not None:
        print(warning, file=sys.stderr, flush=True)
    _, _, ignored_nondefault = style_params(style)
    if ignored_nondefault:
        log("catatan: parameter style belum aktif di T-203a (diabaikan): "
            + ", ".join(f"{k}={v!r}" for k, v in ignored_nondefault.items()))

    stale: list[str] = []
    if restart:
        restart_outputs(clip)
    elif clip.manifest_path.is_file():
        try:
            old = json.loads(clip.manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            old = {}
        stale = manifest_diff(old if isinstance(old, dict) else {}, manifest)
        if stale:
            log("PERINGATAN: output [5] basi (setelan / input berubah) — strokes/ dihapus dan dihitung ulang:\n  "
                + "\n  ".join(stale))
            restart_outputs(clip)
    elif _has_outputs(clip):
        raise StageError(f"{clip.strokes_dir} berisi output tanpa {MANIFEST_FILENAME} — asal output tidak diketahui. "
                         f"Jalankan {cli_cmd('stylize', clip.work_dir)} --restart.")
    clean_tmp(clip.strokes_dir)

    size = (g.out_w, g.out_h)
    todo = [(n, i) for n, i in zip(selected, indices) if not frame_valid(clip, n, size)]
    log(f"[5] stylize ({CONTRACT}: garis polos, edge_mode {g.edge_mode}): {len(selected)} frame dipilih, "
        f"{len(selected) - len(todo)} valid dilewati, {len(todo)} diproses; output {g.out_w}x{g.out_h} "
        f"(s {g.scale:.4f}, unit {g.unit:.4f})")
    run = {"selected": len(selected), "skipped": len(selected) - len(todo), "processed": 0, "stale": stale,
           "frames": [], "output_size": size}
    if not todo:
        return run
    ensure_dir(clip.strokes_dir)
    if not clip.manifest_path.is_file():
        write_json_atomic(clip.manifest_path, manifest)
    t_run = time.perf_counter()
    append_jsonl(clip.frames_log, {"event": "run_start", "time_utc": utc_now(), "n_todo": len(todo),
                                   "style_hash": manifest["style_hash"]})
    try:
        for name, index in todo:
            t0 = time.perf_counter()
            doc = read_contours_frame(clip, name, index, cm["vectorize_hash"])
            svg, png, stats = render_frame(doc, g, style)
            t1 = time.perf_counter()
            write_bytes_atomic(clip.svg_path(name), svg)
            write_bytes_atomic(clip.png_path(name), png)     # PNG terakhir: frame valid hanya bila keduanya ada
            t2 = time.perf_counter()
            rec = {"event": "frame", "frame": name, "index": index, "time_utc": utc_now(),
                   "write_s": round(t2 - t1, 4), "total_s": round(t2 - t0, 4), **stats}
            append_jsonl(clip.frames_log, rec)
            run["frames"].append(rec)
            run["processed"] += 1
            log(f"  [{run['processed']}/{len(todo)}] {name} {rec['total_s']:.3f} s, strok {stats['n_strokes']}, jalur "
                f"{stats['n_pieces']}, svg {len(svg) / 1024:.1f} KiB, png {len(png) / 1024:.1f} KiB")
    finally:
        append_jsonl(clip.frames_log, {"event": "run_end", "time_utc": utc_now(), "n_done": run["processed"],
                                       "wall_s": round(time.perf_counter() - t_run, 2)})
    return run


# ── Entry point stage (dipanggil cli.py) ───────────
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m rotoscope.stylize",
                                description="Stage [5]: contours/ → strokes/ (SVG + PNG garis polos)")
    p.add_argument("--config", type=Path, default=None,
                   help=f"YAML pipeline (default: {DEFAULT_PIPELINE.as_posix()} kalau ada, selain itu default kode)")
    p.add_argument("--style", type=Path, default=None,
                   help=f"YAML style (default: {DEFAULT_STYLE.as_posix()} kalau ada, selain itu default kode)")
    add_work_dir_arg(p)
    p.add_argument("--restart", action="store_true", help="hapus output [5] lama (strokes/) lalu hitung ulang")
    p.add_argument("--limit", type=int, default=None, help="hanya N frame pertama")
    args = p.parse_args(argv)
    reconfigure_stdio()

    def log(msg: str) -> None:
        print(msg, flush=True)

    try:
        cpath = args.config if args.config is not None else (DEFAULT_PIPELINE if DEFAULT_PIPELINE.is_file() else None)
        cfg = load_pipeline(cpath, overrides=work_dir_overrides(args.work_dir))
        spath = args.style if args.style is not None else (DEFAULT_STYLE if DEFAULT_STYLE.is_file() else None)
        style = load_style(spath)
        t0 = time.perf_counter()
        run = run_stylize(cfg, style, spath.stem if spath else "default", restart=args.restart, limit=args.limit,
                          log=log)
        log(f"selesai: {run['processed']} diproses, {run['skipped']} dilewati ({time.perf_counter() - t0:.1f} s)")
    except (StageError, ConfigError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return EXIT_PRECONDITION
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
