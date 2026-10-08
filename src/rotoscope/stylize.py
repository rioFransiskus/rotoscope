"""Stage [5] stylize (T-402, CPU): contours/*.json → strokes/frame_%05d.svg + .png + manifest.json.

    python -m rotoscope stylize <video> [--config PATH] [--style PATH] [--restart] [--limit N]

cli.py memanggil main(). TANPA GPU: torch tidak pernah di-import.

Garis bertebal variabel (T-401): latar `paper.color`, tinta `stroke.color`, solid. Tebal per titik = lantai(dasar × skala tipe ×
(1 + variasi × noise 2D terkunci posisi) × taper); taper hanya di ujung BEBAS. Jitter, multipass, tekstur, opasitas BELUM aktif
(`ignored_params` di manifest). T-404a (Opsi B): [5] tetap menulis kertas DATAR (PNG + SVG); kertas bertekstur / vignette disusun saat export (rotoscope.paper).

Satuan panjang style = px REFERENSI lebar 1080 × unit (unit = render.output_width / 1080). Geometri dihitung di px
OUTPUT: titik kontur (pusat piksel kerja) × s, s = output_width / width.

Pipeline geometri per strok (satu geometri untuk SVG dan raster):
  titik × s → [mode hide] run di tepi frame dibuang, strok dipecah jadi jalur terbuka → approxPolyDP → Catmull-Rom
  (uniform; periodik untuk strok tertutup) → ujung di tepi diperpanjang keluar KANVAS → titik dibulatkan 2 desimal
  → resample arc-length (jarak maks RESAMPLE_MAX_GAP_REF, N ≥ shape.resample_points) → tebal per titik.
Raster = satu mask supersampling (union trapesium + cakram per titik; union semua strok) → INTER_AREA → kertas.
SVG = string manual (byte-deterministik): SATU poligon kontur terisi per jalur (fill-rule nonzero) dari garis tengah + tebal yang SAMA.
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
from scipy.spatial import cKDTree

from rotoscope import noise
from rotoscope import vectorize as vz
from rotoscope.config import ConfigError, StyleConfig, ensure_dir, load_pipeline, load_style, resolve_style, style_name
from rotoscope.stage_common import (
    CLIP_KEY, EXIT_OK, EXIT_PRECONDITION, StageError, add_work_dir_arg, append_jsonl, clean_tmp, cli_cmd,
    clip_identity, describe_identity, reconfigure_stdio, utc_now, window_bounds, work_dir_overrides,
    write_bytes_atomic, write_json_atomic,
)

# ── Layout output (docs/01 [5]) ────────────────────
STROKES_DIRNAME = "strokes"
MANIFEST_FILENAME = "manifest.json"
FRAMES_LOG_FILENAME = "frames.jsonl"
SVG_SUFFIX = ".svg"
PNG_SUFFIX = ".png"
CONTRACT = "T-403"
# Naik 1 HANYA untuk perbaikan PERILAKU pada kode yang sudah dikontrak, tanpa perubahan parameter (docs/01). Fitur baru
# (jitter, taper, ...) menaikkan CONTRACT, bukan ALGO_REV.
ALGO_REV = 2    # 2 (T-406): zona mati tepi pass k >= 1 mengikuti tebal LOKAL (EDGE_INK_GUARD_PX); strokes ALGO_REV 1 basi
SUPPORTED_CONTOURS_CONTRACTS = frozenset({"T-202"})
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
# cv2.fillConvexPoly mengisi tepi poligon (inklusif): memperlebar ≈ 0,5 piksel ss per sisi pada garis horizontal, lebih pada garis
# miring (+0,4…0,7% pada tebal 9 px). 0,55 = kalibrasi T-401 terhadap luas poligon SVG pada frame nyata (rasio massa SVG/PNG
# 0,998–0,9995; 0,5 → 0,994–0,996); selisih terhadap T-203a: massa tinta −0,3%, IoU 0,993 (tebal konstan).
FILL_BIAS_SS = 0.55
CIRCLE_VERTICES = 24              # poligon pendekatan cakram sambungan / ujung
CIRCLE = np.column_stack([np.cos(2 * np.pi * (np.arange(CIRCLE_VERTICES) + 0.5) / CIRCLE_VERTICES),
                          np.sin(2 * np.pi * (np.arange(CIRCLE_VERTICES) + 0.5) / CIRCLE_VERTICES)]
                         ) / np.cos(np.pi / CIRCLE_VERTICES)                         # sisi poligon menyinggung lingkaran jari-jari 1
PNG_COMPRESSION = 3               # cv2.IMWRITE_PNG_COMPRESSION (0–9); deterministik
SVG_DECIMALS = 2                  # titik SVG dan raster dibulatkan 0,01 px output (titik SAMA); galat ≤ 0,007 px
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
ONLY_CAP = "round"                # hanya cap bulat
TYPE_ORDER = vz.STROKE_TYPES      # urutan <g> di SVG = urutan strok [4]

# ── T-401: resample, tebal, taper, noise (konstanta struktural; keputusan Rio, docs/04 "Keputusan T-401") ─
RESAMPLE_MAX_GAP_REF = 2.0        # jarak maks titik hasil resample (px ref)
WIDTH_FLOOR_PX = 1.0              # lantai tebal ABSOLUT (px output): < 1 px raster ss = 3 terkuantisasi per 1/3 px, tak monoton
JOIN_DIST_PX = 8.0                # ujung ≤ ini (px ref) dari strok lain = "bertemu" (tanpa taper); celah data di 6–8 px
JOIN_DIST_OCC_PX = 2.0            # ujung OKLUSI ≤ ini (px ref) dari strok OKLUSI lain = bertemu (pecahan satu garis)
JOIN_SAMPLE_STEP = 0.5            # jarak sampel (px output) polyline strok lain untuk pengukuran jarak ujung
OUTLINE_CAP_STEPS = 8             # segmen setengah lingkaran ujung bulat pada poligon kontur SVG
SVG_FILL_RULE = "nonzero"

# ── T-402: jitter (konstanta struktural; keputusan Rio, docs/04 "Keputusan T-402") ─
JITTER_SALT_FIELD = 0x4A17        # salt stream medan koheren F (seed_of(param_seed, salt, kanal)); stream tebal T-401 = seed_of(param_seed) tanpa salt
JITTER_SALT_TRACK = 0x7C0F        # salt stream komponen independen G (seed_of(param_seed, salt, track_id, kanal))
JITTER_CHANNELS = 2               # kanal noise independen: perpindahan x dan y (vektor 2D, bukan sepanjang normal)
EDGE_FADE_PX = 40.0               # panjang pelunakan jitter dari zona mati tepi sampai penuh (px ref)
JITTER_FOLD_R_WARN = 0.19         # r = amplitude × frequency × (√(1−s) + √s) > ini → peringatan lipatan (terukur: r ≤ 0,18 lulus Jacobian 0,05, r 0,21 gagal di 2–17 frame, 0,32 melipat)
JACOBIAN_MIN_DET = 0.05           # ambang keselamatan min det(I + ∇D) pada titik strok (tests/jitter_metrics.py)
JITTER_JOINT_TOL_GRAD = 3.2       # toleransi perubahan celah sambungan (s = 0) = ini × r × join_dist (p99 norma ∇D = 3,24 r × celah maks); terukur maks 2,4 r × join_dist

# ── T-403: multipass (konstanta struktural; keputusan Rio, docs/04 "Keputusan T-403") ─
MULTIPASS_SALT_FIELD = 0x5D2B     # salt medan koheren pass k >= 1 (seed_of(param_seed, salt, k, kanal)); beda dari salt jitter
MULTIPASS_FOLD_R = 0.15           # r = amplitudo / panjang gelombang pass tambahan (konstan; terukur min det Jacobian >= 0,39; batas peringatan 0,19)
MULTIPASS_SEP_MEDIAN = 0.59       # median |D| / A medan pass (terukur kedua klip: 0,590–0,591) -> A = offset / ini; offset = median |D|
PASS_ID_FMT = "pass_{}"           # id grup pass (1-based); id grup tipe = pass_K_<tipe>
INK_BOX_MARGIN_PX = 2             # margin jendela tinta (px output) di luar titik ± tebal/2: antialias INTER_AREA + pembulatan sub-piksel
MULTIPASS_EDGE_EXTRA_PX = 1.0    # tambahan zona mati tepi pass k >= 1 (px output): tinta 3 baris/kolom terluar tidak berubah (T-403, diukur)
EDGE_INK_GUARD_PX = 3.5          # T-406: ujung tinta pass k >= 1 (titik ± setengah tebal LOKAL) tak boleh lebih dekat ke tepi dari ini (px output): 3 baris terluar + 0,5 px antialias; zona mati lama (margin tetap 2 px) bocor pada garis lebar
OPACITY_DECIMALS = 4           # atribut opacity SVG (grup) 4 desimal tetap (deterministik)

# Parameter style AKTIF (masuk hash); sisanya = ignored_params.
ACTIVE_SCALARS = ("shape.simplify_epsilon", "shape.smooth_px", "shape.smooth_tension", "shape.spline_steps", "shape.edge_mode",
                  "shape.resample_points", "stroke.width_base", "stroke.width_variation", "stroke.width_noise_scale",
                  "stroke.taper_ends", "stroke.taper_px", "stroke.taper_min", "stroke.color", "stroke.cap", "jitter.param_seed",
                  "jitter.amplitude", "jitter.frequency", "jitter.temporal_seed_mode", "jitter.temporal_drift",
                  "jitter.hold_frames", "jitter.stroke_independence", "stroke.opacity", "multipass.enabled", "multipass.passes",
                  "multipass.offset", "multipass.opacity_falloff", "multipass.temporal_mode",
                  "paper.color", "render.ss", "render.output_width")
# T-404a (Opsi B): paper.* LAINNYA (enabled, texture_*, vignette) tidak dipakai [5] → ignored_params; kertas disusun saat export (rotoscope.paper)
ACTIVE_BY_TYPE_PREFIX = "stroke.by_type."
ACTIVE_BY_TYPE_SUFFIXES = (".width_scale", ".taper_ends", ".opacity_scale")
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
    return ACTIVE_SCALARS + tuple(f"{ACTIVE_BY_TYPE_PREFIX}{t}{s}" for t in TYPE_ORDER for s in ACTIVE_BY_TYPE_SUFFIXES)


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
    widths: Mapping[str, float]   # tebal dasar per tipe (px output) = width_base × width_scale × unit
    ink: tuple[int, int, int]
    paper: tuple[int, int, int]
    variation: float = 0.0        # amplitudo variasi tebal (0 = seragam)
    noise_cell: float = 0.0       # panjang gelombang noise (px output); 0 = noise mati
    noise_seed: int = 0
    taper_len: float = 0.0        # panjang zona taper (px output)
    taper_min: float = 1.0
    taper_on: Mapping[str, bool] = dataclasses.field(default_factory=dict)   # taper aktif per tipe
    resample_n: int = 0           # batas bawah N titik per jalur
    join_dist: float = 0.0        # px output
    join_dist_occ: float = 0.0
    floor: float = 0.0            # lantai tebal (px output)
    # ── T-402 jitter (medan perpindahan koheren) ──
    jitter_amp: float = 0.0       # amplitudo puncak per kanal (px output) = amplitude × unit; 0 = jitter mati
    jitter_cell: float = 0.0      # panjang sel noise (px output) = unit / frequency; 0 = jitter mati
    jitter_param_seed: int = 0
    jitter_seeds: tuple = ()      # seed kanal medan koheren F (x, y)
    jitter_drift: float = 0.0
    jitter_hold: int = 1
    jitter_fixed: bool = False    # temporal_seed_mode "fixed": indeks gambar selalu 0
    jitter_s: float = 0.0         # stroke_independence
    edge_dead: float = 0.0        # zona mati pelunakan tepi (px output): tebal maks / 2 + EDGE_MARGIN_PX
    edge_fade_len: float = 0.0    # panjang pelunakan (px output) = EDGE_FADE_PX × unit
    # ── T-403 multipass (pass efektif = passes bila enabled, selain itu 1) ──
    passes: int = 1               # jumlah pass efektif (pass 0 = garis asli)
    mp_amp: float = 0.0           # amplitudo puncak medan pass tambahan (px output) = offset / MULTIPASS_SEP_MEDIAN × unit
    mp_cell: float = 0.0          # panjang sel medan (px output) = mp_amp / MULTIPASS_FOLD_R
    mp_seeds: tuple = ()          # per pass k >= 1 (indeks k − 1): seed kanal (x, y)
    mp_fixed: bool = True         # temporal_mode "fixed": indeks gambar selalu 0
    opacity: float = 1.0          # stroke.opacity
    falloff: float = 1.0          # multipass.opacity_falloff
    type_scale: Mapping[str, float] = dataclasses.field(default_factory=dict)   # opacity_scale per tipe (kosong = semua 1,0)

    @property
    def jitter_on(self) -> bool:
        return self.jitter_amp > 0 and self.jitter_cell > 0

    def pass_alpha(self, k: int) -> float:
        """Alpha pass k (0-based) tanpa opacity_scale tipe: stroke.opacity × opacity_falloff^k."""
        return self.opacity * self.falloff ** k

    def scale_of(self, typ: str) -> float:
        return float(self.type_scale.get(typ, 1.0))

    @property
    def legacy(self) -> bool:
        """Satu pass, opacity 1,0, semua opacity_scale 1,0 → jalur T-402 persis (SVG tanpa pembungkus pass; PNG LUT lama)."""
        return self.passes == 1 and self.opacity == 1.0 and all(float(v) == 1.0 for v in self.type_scale.values())


def jitter_fold_r(amplitude: float, frequency: float, s: float) -> float:
    """r = amplitudo × frekuensi × (√(1 − s) + √s): gradien medan relatif (lipatan bila r besar, docs/04 T-402)."""
    return amplitude * frequency * (math.sqrt(1.0 - s) + math.sqrt(s))


def make_geometry(style: StyleConfig, width: int, height: int) -> Geometry:
    if style.stroke.cap != ONLY_CAP:
        raise StageError(f"stroke.cap = {style.stroke.cap!r} belum didukung (hanya {ONLY_CAP!r})")
    ow = style.render.output_width
    out_w, out_h = output_size(width, height, ow)
    unit = ow / REF_WIDTH
    st = style.stroke
    widths = {t: st.width_base * st.by_type[t].width_scale * unit for t in TYPE_ORDER}
    taper_on = {t: bool(st.taper_ends if st.by_type[t].taper_ends is None else st.by_type[t].taper_ends)
                and st.taper_px > 0 for t in TYPE_ORDER}
    mp = style.multipass
    n_passes = mp.passes if mp.enabled else 1
    mp_amp = mp.offset / MULTIPASS_SEP_MEDIAN * unit
    return Geometry(width=width, height=height, out_w=out_w, out_h=out_h, scale=ow / width, unit=unit,
                    epsilon=style.shape.simplify_epsilon * unit, smooth=style.shape.smooth_px * unit, tension=style.shape.smooth_tension,
                    steps=style.shape.spline_steps, spline_tol=SPLINE_TOL_REF * unit,
                    edge_mode=style.shape.edge_mode, ss=style.render.ss, widths=widths,
                    ink=hex_rgb(st.color), paper=hex_rgb(style.paper.color),
                    variation=st.width_variation if st.width_noise_scale > 0 else 0.0,
                    noise_cell=unit / st.width_noise_scale if st.width_noise_scale > 0 else 0.0,
                    noise_seed=int(noise.seed_of(style.jitter.param_seed)),
                    taper_len=st.taper_px * unit, taper_min=st.taper_min, taper_on=taper_on,
                    resample_n=style.shape.resample_points, join_dist=JOIN_DIST_PX * unit,
                    join_dist_occ=JOIN_DIST_OCC_PX * unit, floor=WIDTH_FLOOR_PX,
                    jitter_amp=style.jitter.amplitude * unit,
                    jitter_cell=unit / style.jitter.frequency if style.jitter.frequency > 0 else 0.0,
                    jitter_param_seed=int(style.jitter.param_seed),
                    jitter_seeds=tuple(noise.seed_of(style.jitter.param_seed, JITTER_SALT_FIELD, c) for c in range(JITTER_CHANNELS)),
                    jitter_drift=style.jitter.temporal_drift, jitter_hold=style.jitter.hold_frames,
                    jitter_fixed=style.jitter.temporal_seed_mode == "fixed", jitter_s=style.jitter.stroke_independence,
                    edge_dead=max(widths.values()) * (1.0 + (st.width_variation if st.width_noise_scale > 0 else 0.0)) / 2
                    + EDGE_MARGIN_PX, edge_fade_len=EDGE_FADE_PX * unit,
                    passes=n_passes, mp_amp=mp_amp, mp_cell=mp_amp / MULTIPASS_FOLD_R if mp_amp > 0 else 0.0,
                    mp_seeds=tuple(tuple(noise.seed_of(style.jitter.param_seed, MULTIPASS_SALT_FIELD, k, c)
                                         for c in range(JITTER_CHANNELS)) for k in range(1, n_passes)),
                    mp_fixed=mp.temporal_mode == "fixed", opacity=st.opacity, falloff=mp.opacity_falloff,
                    type_scale={t: st.by_type[t].opacity_scale for t in TYPE_ORDER})


@dataclass(frozen=True)
class Piece:
    """Satu jalur yang digambar (px output, sudah dibulatkan 2 desimal)."""
    type: str
    closed: bool
    points: np.ndarray            # (N, 2)
    track_id: int
    stroke_idx: int = -1          # indeks strok asal di frame (jalur hasil potong tepi berbagi indeks)
    edge: tuple[bool, bool] = (False, False)   # ujung awal / akhir menyentuh tepi frame (tanpa taper)
    widths: np.ndarray | None = None           # tebal per titik (px output); None = seragam g.widths[type]


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


def stroke_pieces(stroke: dict, g: Geometry, stats: dict, stroke_idx: int = -1) -> list[Piece]:
    """Satu strok kontur → jalur yang digambar (satu geometri untuk SVG dan raster); belum di-resample, tanpa tebal per titik."""
    p = np.asarray(stroke["points"], dtype=np.float64)
    closed = bool(stroke["closed"])
    if not closed and len(p) > 3 and np.array_equal(p[0], p[-1]):     # loop: titik akhir = titik awal
        closed, p = True, p[:-1]
    typ, tid = stroke["type"], int(stroke["track_id"])
    wpx = g.widths[typ] * (1.0 + g.variation)         # tebal maksimum yang mungkin: ekstensi tepi melewati kanvas dengan aman
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
        edge = (False, False) if cl else (bool(sd[0]), bool(sd[-1]))     # titik silang tepi / run tepi (kedua mode)
        out.append(Piece(typ, cl, np.round(curve, SVG_DECIMALS) + 0.0, tid, stroke_idx, edge))
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


def fold_warning(style: StyleConfig) -> str | None:
    """Peringatan (bukan error, tanpa clamp) bila jitter aktif dan r > JITTER_FOLD_R_WARN: garis dapat melipat / bersilang."""
    j, mp = style.jitter, style.multipass
    r = jitter_fold_r(j.amplitude, j.frequency, j.stroke_independence) if j.amplitude > 0 and j.frequency > 0 else 0.0
    mp_on = mp.enabled and mp.passes > 1 and mp.offset > 0
    if mp_on and r > 0 and r + MULTIPASS_FOLD_R > JITTER_FOLD_R_WARN:
        return (f"PERINGATAN: r jitter {r:.3f} + r multipass {MULTIPASS_FOLD_R} = {r + MULTIPASS_FOLD_R:.3f} > {JITTER_FOLD_R_WARN}: "
                f"pass tambahan (pass 0 sudah bergetar) dapat melipat atau bersilang. Turunkan jitter.amplitude / jitter.frequency "
                f"atau matikan multipass. Tidak ada clamp; render tetap jalan.")
    if r <= JITTER_FOLD_R_WARN:
        return None
    return (f"PERINGATAN: jitter r = amplitude × frequency × (√(1−s) + √s) = {r:.3f} > {JITTER_FOLD_R_WARN} "
            f"(jitter.amplitude {j.amplitude:g}, jitter.frequency {j.frequency:g}, jitter.stroke_independence {j.stroke_independence:g}): "
            f"medan perpindahan terlalu curam, garis dapat melipat atau bersilang. Turunkan amplitude atau frequency "
            f"(terukur: r 0,21 aman, 0,32 melipat). Tidak ada clamp; render tetap jalan.")


def new_stats() -> dict:
    return {"dropped_all_edge": 0, "edge_cuts": 0, "edge_ends": 0, "shallow_ends": 0, "corner_ends": 0,
            "min_contact_deg": 90.0, "free_ends": 0}


# ── Resample + tebal per titik (T-401) ─────────────
def resample_n(points: np.ndarray, closed: bool, n: int) -> np.ndarray:
    """n titik berjarak busur sama dari titik awal. Terbuka: ujung persis dipertahankan; tertutup: tanpa titik ganda."""
    q = np.vstack([points, points[:1]]) if closed else points
    s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(q, axis=0).T))]
    length = float(s[-1])
    if length <= 0:
        return points
    t = np.arange(n) * (length / n) if closed else np.linspace(0.0, length, n)
    return np.column_stack([np.interp(t, s, q[:, 0]), np.interp(t, s, q[:, 1])])


def resample_piece(pc: Piece, g: Geometry) -> np.ndarray:
    """N = max(shape.resample_points, ceil(panjang / (RESAMPLE_MAX_GAP_REF × unit)) [+ 1 bila terbuka]); dibulatkan 2 desimal."""
    q = np.vstack([pc.points, pc.points[:1]]) if pc.closed else pc.points
    length = float(np.hypot(*np.diff(q, axis=0).T).sum())
    n = max(g.resample_n, int(math.ceil(length / (RESAMPLE_MAX_GAP_REF * g.unit))) + (0 if pc.closed else 1))
    return np.round(resample_n(pc.points, pc.closed, n), SVG_DECIMALS) + 0.0


def taper_factor(s: np.ndarray, length: float, free: tuple[bool, bool], g: Geometry) -> np.ndarray:
    """Faktor taper per titik (busur s dari titik awal): taper_min di ujung bebas → 1 pada jarak min(taper_px, L/2)."""
    lt = min(g.taper_len, length / 2)
    if lt <= 0 or not (free[0] or free[1]):
        return np.ones_like(s)
    t = np.full_like(s, np.inf)
    if free[0]:
        t = np.minimum(t, s / lt)
    if free[1]:
        t = np.minimum(t, (length - s) / lt)
    t = np.clip(t, 0.0, 1.0)
    return g.taper_min + (1.0 - g.taper_min) * t * t * (3.0 - 2.0 * t)      # smoothstep (keputusan Rio: bukan linear)


def width_profile(points: np.ndarray, closed: bool, typ: str, free: tuple[bool, bool], g: Geometry) -> np.ndarray:
    """Tebal per titik (px output): lantai(dasar × (1 + variasi × noise 2D terkunci posisi) × taper). Statis terhadap waktu."""
    w = np.full(len(points), g.widths[typ])
    if g.variation > 0 and g.noise_cell > 0:
        n = noise.value_noise_2d(np.uint64(g.noise_seed), points[:, 0] / g.noise_cell, points[:, 1] / g.noise_cell)
        w = w * (1.0 + g.variation * n)
    if not closed and (free[0] or free[1]):
        s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(points, axis=0).T))]
        w = w * taper_factor(s, float(s[-1]), free, g)
    return np.maximum(w, g.floor)


def free_ends(pieces: list[Piece], g: Geometry) -> list[tuple[bool, bool]]:
    """Ujung BEBAS (kena taper) per jalur: terbuka, tipenya taper-aktif, tidak menyentuh tepi frame, dan tidak "bertemu" strok
    LAIN: jarak ≤ join_dist (JOIN_DIST_PX); ujung oklusi ke strok OKLUSI lain ≤ join_dist_occ (JOIN_DIST_OCC_PX)."""
    flags = [[False, False] for _ in pieces]
    cand = [(i, e) for i, pc in enumerate(pieces)
            if g.taper_on[pc.type] and not pc.closed and len(pc.points) >= 2 for e in (0, 1) if not pc.edge[e]]
    if not cand:
        return [(False, False)] * len(pieces)
    dense = [arc_resample(pc.points, pc.closed, JOIN_SAMPLE_STEP * g.unit) for pc in pieces]
    owner = np.concatenate([np.full(len(d), pc.stroke_idx if pc.stroke_idx >= 0 else -(k + 2))
                            for k, (pc, d) in enumerate(zip(pieces, dense))])
    occ = np.concatenate([np.full(len(d), pc.type == "occlusion") for pc, d in zip(pieces, dense)])
    samples = np.vstack(dense)
    ends = np.array([pieces[i].points[0 if e == 0 else -1] for i, e in cand])
    hits = cKDTree(samples).query_ball_point(ends, max(g.join_dist, g.join_dist_occ))
    for (i, e), end, idx in zip(cand, ends, hits):
        pc = pieces[i]
        idx = np.asarray(idx, dtype=int)
        meet = False
        if len(idx):
            own = pc.stroke_idx if pc.stroke_idx >= 0 else -(i + 2)
            idx = idx[owner[idx] != own]
            if len(idx):
                dist = np.hypot(*(samples[idx] - end).T)
                thr = np.where(occ[idx] & (pc.type == "occlusion"), g.join_dist_occ, g.join_dist)
                meet = bool((dist <= thr).any())
        flags[i][e] = not meet
    return [(f[0], f[1]) for f in flags]


# ── Jitter (T-402) ─────────────────────────────────
def jitter_image_index(frame_index: int, g: Geometry) -> int:
    """Indeks "gambar" k: 0 pada mode fixed; selain itu floor(frame_index / hold_frames) dengan frame_index ABSOLUT."""
    return 0 if g.jitter_fixed else int(frame_index) // g.jitter_hold


def edge_fade(p: np.ndarray, g: Geometry, half_width: np.ndarray | None = None) -> np.ndarray:
    """φ(d) pelunakan di tepi bawah / kiri / kanan: 0 untuk d ≤ edge_dead (termasuk di luar kanvas), smoothstep sampai 1 pada
    edge_dead + edge_fade_len. d = jarak (px output) ke tepi terdekat dari ketiganya. `half_width` (T-406, pass k >= 1): setengah tebal
    LOKAL per titik; zona mati titik = max(edge_dead, half_width + EDGE_INK_GUARD_PX) — kontinu (tebal berubah mulus sepanjang jalur),
    dan titik yang zona matinya sudah cukup tidak berubah."""
    d = np.minimum(np.minimum(p[:, 0], g.out_w - p[:, 0]), g.out_h - p[:, 1])
    dead = g.edge_dead if half_width is None else np.maximum(g.edge_dead, half_width + EDGE_INK_GUARD_PX)
    u = np.clip((d - dead) / max(g.edge_fade_len, 1e-9), 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def jitter_displacement(points: np.ndarray, piece_ends: list[tuple[int, int]], track_ids: list[int], g: Geometry,
                        frame_index: int, half_width: np.ndarray | None = None) -> np.ndarray:
    """D (N, 2) px output untuk titik-titik (semua jalur frame digabung; piece_ends = (awal, akhir) per jalur).

    D = amplitude × unit × [√(1 − s) F + √s G_track] × φ(d). F = medan koheren (dua kanal noise 3D independen, fungsi posisi +
    waktu SAJA: titik di lokasi sama → D sama); G_track = medan sama dengan seed per track_id (hanya bila s > 0)."""
    t = jitter_image_index(frame_index, g) * g.jitter_drift
    x, y = points[:, 0] / g.jitter_cell, points[:, 1] / g.jitter_cell
    d = np.column_stack([noise.value_noise_3d(sd, x, y, t) for sd in g.jitter_seeds])
    s = g.jitter_s
    if s > 0:
        d *= math.sqrt(1.0 - s)
        wg = math.sqrt(s)
        for (a, b), tid in zip(piece_ends, track_ids):
            seeds = [noise.seed_of(g.jitter_param_seed, JITTER_SALT_TRACK, tid, c) for c in range(JITTER_CHANNELS)]
            d[a:b] += wg * np.column_stack([noise.value_noise_3d(sd, x[a:b], y[a:b], t) for sd in seeds])
    return d * g.jitter_amp * edge_fade(points, g, half_width)[:, None]


def jitter_pieces(pieces: list[Piece], g: Geometry, frame_index: int, ink_guard: bool = False) -> list[Piece]:
    """Terapkan jitter ke titik tiap jalur (setelah resample + tebal; arc-length taper, ujung bebas, "bertemu" memakai geometri
    TANPA jitter). Titik dibulatkan SVG_DECIMALS lagi (titik SVG = titik raster). Mati (amplitude 0 / frequency 0) → pieces apa
    adanya (byte-identik T-401)."""
    if not g.jitter_on or not pieces:
        return pieces
    ends, off = [], 0
    for pc in pieces:
        ends.append((off, off + len(pc.points)))
        off += len(pc.points)
    pts = np.vstack([pc.points for pc in pieces])
    half = np.concatenate([piece_widths(pc, g) for pc in pieces]) / 2.0 if ink_guard else None
    d = jitter_displacement(pts, ends, [pc.track_id for pc in pieces], g, frame_index, half)
    return [dataclasses.replace(pc, points=np.round(pc.points + d[a:b], SVG_DECIMALS) + 0.0)
            for pc, (a, b) in zip(pieces, ends)]


def base_pieces(doc: dict, g: Geometry) -> tuple[list[Piece], dict]:
    """Jalur T-401 (tanpa jitter): geometri → resample → ujung bebas → tebal per titik."""
    stats = new_stats()
    raw: list[Piece] = []
    for i, s in enumerate(doc["strokes"]):
        raw += stroke_pieces(s, g, stats, i)
    res = [dataclasses.replace(pc, points=resample_piece(pc, g)) for pc in raw]
    flags = free_ends(res, g)
    pieces = []
    for pc, fl in zip(res, flags):
        stats["free_ends"] += int(fl[0]) + int(fl[1])
        pieces.append(dataclasses.replace(pc, widths=width_profile(pc.points, pc.closed, pc.type, fl, g)))
    return pieces, stats


def frame_pieces(doc: dict, g: Geometry) -> tuple[list[Piece], dict]:
    """Jalur yang digambar: jalur T-401 + jitter (indeks gambar dari doc["frame_index"] ABSOLUT, tanpa rantai antar frame)."""
    pieces, stats = base_pieces(doc, g)
    t0 = time.perf_counter()
    pieces = jitter_pieces(pieces, g, int(doc["frame_index"]))
    stats["jitter_s"] = round(time.perf_counter() - t0, 4)
    return pieces, stats


# ── Multipass (T-403) ──────────────────────────────
def pass_geometry(g: Geometry, k: int) -> Geometry:
    """Geometri medan pass k >= 1: mesin jitter T-402 (noise 3D koheren, penjaga tepi φ) dengan amplitudo / sel / seed pass, s = 0.
    Waktu: mode "fixed" → indeks gambar 0 (statis); "frame" → floor(frame_index / jitter.hold_frames) × jitter.temporal_drift.
    Zona mati tepi = edge_dead pass 0 + MULTIPASS_EDGE_EXTRA_PX (tinta 3 baris / kolom terluar tidak berubah)."""
    return dataclasses.replace(g, jitter_amp=g.mp_amp, jitter_cell=g.mp_cell, jitter_seeds=g.mp_seeds[k - 1],
                               jitter_fixed=g.mp_fixed, jitter_s=0.0, edge_dead=g.edge_dead + MULTIPASS_EDGE_EXTRA_PX)


def multipass_pieces(pieces: list[Piece], g: Geometry, frame_index: int) -> list[list[Piece]]:
    """[pass 0 = pieces (setelah jitter T-402)] + pass k >= 1: titik pass 0 + D_k. Tebal / taper / flag tepi disalin. Pass dengan
    offset 0 = salinan pass 0."""
    return [pieces] + [jitter_pieces(pieces, pass_geometry(g, k), frame_index, ink_guard=True) for k in range(1, g.passes)]


def frame_passes(doc: dict, g: Geometry) -> tuple[list[list[Piece]], dict]:
    """Semua pass yang digambar frame ini (indeks 0 = garis asli); frame_index ABSOLUT, tanpa rantai antar frame."""
    pieces, stats = frame_pieces(doc, g)
    t0 = time.perf_counter()
    passes = multipass_pieces(pieces, g, int(doc["frame_index"])) if g.passes > 1 else [pieces]
    stats["multipass_s"] = round(time.perf_counter() - t0, 4)
    stats["passes"] = len(passes)
    return passes, stats


# ── Render ─────────────────────────────────────────
def piece_widths(pc: Piece, g: Geometry) -> np.ndarray:
    return pc.widths if pc.widths is not None else np.full(len(pc.points), g.widths[pc.type])


def render_mask(pieces: list[Piece], g: Geometry, box: tuple[int, int, int, int] | None = None) -> np.ndarray:
    """Satu mask supersampling (union semua strok, tipe dan tebal berbeda). Koordinat cv2 = x' · ss − RASTER_OFFSET (titik
    cv2 integer = pusat piksel supersampling; titik kontinu x' · ss = k + 0,5 adalah pusatnya), sub-piksel 1/16.
    Tebal per titik: trapesium per segmen (jari-jari kedua titik) + cakram di tiap titik (union = nonzero dari poligon kontur).
    box = (x0, y0, x1, y1) piksel output (bulat): hanya jendela itu yang dirender (T-403, hemat biaya; hasil jendela = potongan mask penuh);
    None = seluruh kanvas."""
    x0, y0, x1, y1 = box if box is not None else (0, 0, g.out_w, g.out_h)
    wmax = max([float(piece_widths(pc, g).max()) for pc in pieces] or [0.0])
    pad = int(math.ceil(wmax)) + MASK_PAD_PX          # kanvas diperlebar: poligon tidak terpotong di tepi mask
    mask = np.zeros(((y1 - y0 + 2 * pad) * g.ss, (x1 - x0 + 2 * pad) * g.ss), np.uint8)
    shift = pad if x0 == 0 and y0 == 0 else np.array([pad - x0, pad - y0], np.float64)   # (0, 0) = expresi T-402 persis
    quads: list[np.ndarray] = []
    discs: list[np.ndarray] = []
    for pc in pieces:
        p = (pc.points + shift) * g.ss - RASTER_OFFSET
        r = np.maximum(piece_widths(pc, g) * g.ss / 2 - FILL_BIAS_SS, 0.0)
        q = np.vstack([p, p[:1]]) if pc.closed else p
        rq = np.r_[r, r[:1]] if pc.closed else r
        d = np.diff(q, axis=0)
        length = np.hypot(d[:, 0], d[:, 1])
        keep = length > 0
        n = np.column_stack([-d[:, 1], d[:, 0]])[keep] / length[keep, None]
        a, b = q[:-1][keep], q[1:][keep]
        ra, rb = rq[:-1][keep, None], rq[1:][keep, None]
        quads.append(np.stack([a + n * ra, b + n * rb, b - n * rb, a - n * ra], axis=1))
        discs.append(p[:, None, :] + r[:, None, None] * CIRCLE[None, :, :])       # sambungan + ujung bulat (cap round)
    if quads:
        # fillConvexPoly per poligon: satu panggilan fillPoly memakai aturan genap-ganjil, jadi poligon yang tumpang
        # tindih (kuad bertetangga, cakram) menjadi LUBANG; union butuh panggilan terpisah.
        for polys in (np.concatenate(quads), np.concatenate(discs)):
            for poly in np.rint(polys * SHIFT_SCALE).astype(np.int32):
                cv2.fillConvexPoly(mask, poly, 255, LINE_TYPE, SHIFT_BITS)
    return np.ascontiguousarray(mask[pad * g.ss:(pad + y1 - y0) * g.ss, pad * g.ss:(pad + x1 - x0) * g.ss])


def ink_lut(g: Geometry) -> np.ndarray:
    a = (np.arange(MASK_LEVELS, dtype=np.float32) / (MASK_LEVELS - 1))[:, None]       # tabel: cakupan uint8 → warna
    return np.rint(np.array(g.paper, np.float32) * (1 - a) + np.array(g.ink, np.float32) * a).astype(np.uint8)


def compose(mask: np.ndarray, g: Geometry) -> np.ndarray:
    """mask → cakupan (INTER_AREA) → ink di atas kertas, RGB uint8 (out_h, out_w, 3)."""
    cov = cv2.resize(mask, (g.out_w, g.out_h), interpolation=cv2.INTER_AREA)
    return ink_lut(g)[cov]


def pass_layers(pieces: list[Piece], g: Geometry) -> list[tuple[float, list[Piece]]]:
    """Lapisan satu pass: (opacity_scale, jalur). Tipe ber-scale 1,0 digabung (union, satu lapisan); tiap tipe ber-scale ≠ 1
    = lapisan sendiri (digabung "over" dengan lapisan lain, sama dengan grup bersarang di SVG). Lapisan kosong dibuang."""
    base = [p for p in pieces if g.scale_of(p.type) == 1.0]
    out = [(1.0, base)] if base else []
    for t in TYPE_ORDER:
        if g.scale_of(t) != 1.0:
            sub = [p for p in pieces if p.type == t]
            if sub:
                out.append((g.scale_of(t), sub))
    return out


def ink_box(passes: list[list[Piece]], g: Geometry) -> tuple[int, int, int, int] | None:
    """Jendela piksel (x0, y0, x1, y1; bulat, dalam kanvas) yang memuat SEMUA tinta semua pass: titik ± tebal/2 ± INK_BOX_MARGIN_PX.
    None = tidak ada jalur. Di luar jendela cakupan = 0 persis."""
    lo = np.full(2, np.inf)
    hi = np.full(2, -np.inf)
    for pcs in passes:
        for pc in pcs:
            r = float(piece_widths(pc, g).max()) / 2
            lo = np.minimum(lo, pc.points.min(axis=0) - r)
            hi = np.maximum(hi, pc.points.max(axis=0) + r)
    if not np.isfinite(lo).all():
        return None
    x0, y0 = (np.floor(lo) - INK_BOX_MARGIN_PX).astype(int)
    x1, y1 = (np.ceil(hi) + INK_BOX_MARGIN_PX).astype(int)
    x0, y0, x1, y1 = max(int(x0), 0), max(int(y0), 0), min(int(x1), g.out_w), min(int(y1), g.out_h)
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


def ink_fraction_box(passes: list[list[Piece]], g: Geometry, box: tuple[int, int, int, int]) -> np.ndarray:
    """f di jendela `box` (float32, 0–1): f = 1 − Π_k (1 − a_k × g_k), a_k = opacity × falloff^k, g_k = "over" lapisan pass k (union
    dalam lapisan). Komutatif. Tiap lapisan: cakupan uint8 (INTER_AREA) → tabel float32 256 entri (1 − scale × cakupan / 255): nilai
    per piksel identik dengan aritmetika float32 penuh-frame (operasi elementwise yang sama), tanpa konversi uint8 → float32 per piksel."""
    x0, y0, x1, y1 = box
    keep = np.ones((y1 - y0, x1 - x0), np.float32)
    levels = np.arange(MASK_LEVELS, dtype=np.float32) / (MASK_LEVELS - 1)
    for k, pcs in enumerate(passes):
        inner = None
        for scale, sub in pass_layers(pcs, g):
            cov = cv2.resize(render_mask(sub, g, box), (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA)
            t = (1.0 - scale * levels)[cov]
            inner = t if inner is None else np.multiply(inner, t, out=inner)
        if inner is None:
            continue                                               # pass tanpa jalur: 1 − a × (1 − 1) = 1
        np.subtract(1.0, inner, out=inner)
        inner *= np.float32(g.pass_alpha(k))
        np.subtract(1.0, inner, out=inner)
        keep *= inner
    return 1.0 - keep


def ink_fraction(passes: list[list[Piece]], g: Geometry) -> np.ndarray:
    """Fraksi tinta f seluruh kanvas (float32, (out_h, out_w)); nol di luar `ink_box`."""
    f = np.zeros((g.out_h, g.out_w), np.float32)
    box = ink_box(passes, g)
    if box is not None:
        f[box[1]:box[3], box[0]:box[2]] = ink_fraction_box(passes, g, box)
    return f


def render_png_passes(passes: list[list[Piece]], g: Geometry) -> bytes:
    """PNG semua pass. g.legacy (1 pass, opacity 1,0, scale 1,0) = jalur T-402 persis; selain itu f dikuantisasi ke 256 level → LUT.
    Kertas SELALU datar (`paper.color`); kertas bertekstur / vignette disusun saat export (rotoscope.paper, T-404a Opsi B)."""
    if g.legacy:
        bgr = cv2.cvtColor(compose(render_mask(passes[0], g), g), cv2.COLOR_RGB2BGR)
    else:
        lut = np.ascontiguousarray(ink_lut(g)[:, ::-1])       # tabel langsung BGR (tanpa cvtColor penuh-frame)
        bgr = np.empty((g.out_h, g.out_w, 3), np.uint8)
        bgr[:] = lut[0]                                       # di luar jendela tinta: kertas (f = 0)
        box = ink_box(passes, g)
        if box is not None:
            f = ink_fraction_box(passes, g, box)
            bgr[box[1]:box[3], box[0]:box[2]] = lut[np.rint(f * (MASK_LEVELS - 1)).astype(np.uint8)]
    ok, buf = cv2.imencode(PNG_SUFFIX, bgr, [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION])
    if not ok:
        raise StageError("cv2.imencode PNG gagal")
    return buf.tobytes()


def render_png(pieces: list[Piece], g: Geometry) -> bytes:
    return render_png_passes([pieces], g)


def piece_outline(pc: Piece, g: Geometry) -> list[np.ndarray]:
    """Poligon kontur jalur dari garis tengah + tebal per titik: terbuka = satu poligon (sisi kiri → ujung bulat → sisi kanan
    terbalik → ujung bulat awal); tertutup = dua sub-path berlawanan arah (luar + dalam; nonzero menyisakan tengah kosong).
    Titik dibulatkan SVG_DECIMALS (titik SVG). Tikungan rapat (jari-jari < tebal/2) menghasilkan lipatan; fill nonzero = union."""
    p, w = pc.points, piece_widths(pc, g)
    if pc.closed:
        d = np.roll(p, -1, axis=0) - np.roll(p, 1, axis=0)
    else:
        d = np.empty_like(p)
        d[1:-1] = p[2:] - p[:-2]
        d[0], d[-1] = p[1] - p[0], p[-1] - p[-2]
    norm = np.hypot(*d.T)
    t = d / np.where(norm > 0, norm, 1.0)[:, None]
    nrm = np.column_stack([-t[:, 1], t[:, 0]])
    r = (w / 2)[:, None]
    left, right = p + nrm * r, p - nrm * r
    if pc.closed:
        polys = [left, right[::-1]]
    else:
        th = np.linspace(0.0, math.pi, OUTLINE_CAP_STEPS + 1)[1:-1, None]
        end_cap = p[-1] + r[-1] * (np.cos(th) * nrm[-1] + np.sin(th) * t[-1])
        start_cap = p[0] - r[0] * (np.cos(th) * nrm[0] + np.sin(th) * t[0])
        polys = [np.vstack([left, end_cap, right[::-1], start_cap])]
    return [np.round(pl, SVG_DECIMALS) + 0.0 for pl in polys]


def _path_data(polys: list[np.ndarray]) -> str:
    """"M x y x y … Z" per sub-path (lineto implisit setelah M, SVG 1.1)."""
    return " ".join("M" + " ".join(f"{x:.{SVG_DECIMALS}f} {y:.{SVG_DECIMALS}f}" for x, y in pl) + " Z" for pl in polys)


def render_svg(pieces: list[Piece], g: Geometry, style: StyleConfig) -> bytes:
    """SVG manual: <g id=tipe> per tipe (urutan tetap), satu <path> terisi (poligon kontur) per jalur; garis tengah + tebal
    SAMA dengan raster."""
    return render_svg_passes([pieces], g, style)


def _svg_head(g: Geometry, style: StyleConfig) -> str:
    return (f'<?xml version="1.0" encoding="UTF-8"?>\n<svg xmlns="http://www.w3.org/2000/svg" width="{g.out_w}" '
            f'height="{g.out_h}" viewBox="0 0 {g.out_w} {g.out_h}">\n'
            f'<rect width="{g.out_w}" height="{g.out_h}" fill="{style.paper.color}"/>\n')


def _svg_type_groups(pieces: list[Piece], g: Geometry, style: StyleConfig, prefix: str, scaled: bool) -> list[str]:
    parts = []
    for t in TYPE_ORDER:
        op = f' opacity="{g.scale_of(t):.{OPACITY_DECIMALS}f}"' if scaled and g.scale_of(t) != 1.0 else ""
        parts.append(f'<g id="{prefix}{t}" fill="{style.stroke.color}" fill-rule="{SVG_FILL_RULE}" stroke="none"{op}>\n')
        for pc in pieces:
            if pc.type == t:
                parts.append(f'<path d="{_path_data(piece_outline(pc, g))}"/>\n')
        parts.append("</g>\n")
    return parts


def render_svg_passes(passes: list[list[Piece]], g: Geometry, style: StyleConfig) -> bytes:
    """g.legacy: <g id=tipe> langsung (byte-identik T-402). Selain itu satu <g id="pass_K" opacity=a_k> per pass (a_k = opacity ×
    falloff^k, 4 desimal) berisi <g id="pass_K_<tipe>"> (opacity = opacity_scale bila ≠ 1): SVG viewer mengomposisi grup = "over" antar pass,
    union dalam grup — sama dengan raster."""
    parts = [_svg_head(g, style)]
    if g.legacy:
        parts += _svg_type_groups(passes[0], g, style, "", False)
    else:
        for k, pcs in enumerate(passes):
            pid = PASS_ID_FMT.format(k + 1)
            parts.append(f'<g id="{pid}" opacity="{g.pass_alpha(k):.{OPACITY_DECIMALS}f}">\n')
            parts += _svg_type_groups(pcs, g, style, pid + "_", True)
            parts.append("</g>\n")
    parts.append("</svg>\n")
    return "".join(parts).encode("utf-8")


def render_frame(doc: dict, g: Geometry, style: StyleConfig) -> tuple[bytes, bytes, dict]:
    t0 = time.perf_counter()
    passes, stats = frame_passes(doc, g)
    pieces = passes[0]
    t1 = time.perf_counter()
    png = render_png_passes(passes, g)
    t2 = time.perf_counter()
    svg = render_svg_passes(passes, g, style)
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
            "unit": g.unit, "edge_mode": g.edge_mode, "coords": dict(COORDS_INFO),
            "jitter": {"on": g.jitter_on, "fold_r": round(jitter_fold_r(style.jitter.amplitude, style.jitter.frequency,
                                                                       style.jitter.stroke_independence), 4),
                       "fold_warn": fold_warning(style) is not None, "edge_dead_px": g.edge_dead,
                       "edge_fade_px": g.edge_fade_len},
            "multipass": {"passes": g.passes, "amplitude_px": g.mp_amp, "cell_px": g.mp_cell, "fold_r": MULTIPASS_FOLD_R if g.passes > 1 else 0.0,
                          "alphas": [round(g.pass_alpha(k), 6) for k in range(g.passes)], "temporal_mode": style.multipass.temporal_mode},
            "created_utc": utc_now()}


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
                limit: int | None = None, start: int | None = None,
                log: Callable[[str], None] = print) -> dict:
    """Jalankan stage [5] (T-402). `start` (--from, T-204) = jendela [start, start+limit): wajib bersama `limit`."""
    if limit is not None and limit < 1:
        raise StageError(f"--limit harus ≥ 1, dapat {limit}")
    clip = load_clip(cfg.paths.work_dir)
    lo, hi = window_bounds(len(clip.names), start, limit)
    selected = clip.names[lo:hi]
    indices = clip.indices[lo:hi]
    cm = load_contours_manifest(clip)
    require_contours(clip, selected, indices, cm["vectorize_hash"])
    g = make_geometry(style, clip.width, clip.height)
    manifest = build_manifest(style, style_name, g, clip, cm)
    if (warning := draw_width_warning(g)) is not None:
        print(warning, file=sys.stderr, flush=True)
    if (warning := fold_warning(style)) is not None:
        print(warning, file=sys.stderr, flush=True)
    _, _, ignored_nondefault = style_params(style)
    if ignored_nondefault:
        log("catatan: parameter style tidak dipakai stage [5] (diabaikan; kertas disusun saat export): "
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
            if start is None:
                log("PERINGATAN: output [5] basi (setelan / input berubah) — strokes/ dihapus dan dihitung ulang:\n  "
                    + "\n  ".join(stale))
            else:
                n_old = sum(1 for _ in clip.strokes_dir.glob("frame_*" + PNG_SUFFIX))
                log(f"PERINGATAN: output [5] basi (setelan / input berubah):\n  " + "\n  ".join(stale) +
                    f"\n  strokes/ DIHAPUS seluruhnya ({n_old} frame), hanya jendela {lo}-{hi - 1} yang dihitung; "
                    f"MP4 utama dan out/svg/ tetap versi lama sampai `run <video>` penuh berikutnya "
                    f"(export penuh sebelum itu gagal: jalankan {cli_cmd('stylize', clip.work_dir)})")
            restart_outputs(clip)
    elif _has_outputs(clip):
        raise StageError(f"{clip.strokes_dir} berisi output tanpa {MANIFEST_FILENAME} — asal output tidak diketahui. "
                         f"Jalankan {cli_cmd('stylize', clip.work_dir)} --restart.")
    clean_tmp(clip.strokes_dir)

    size = (g.out_w, g.out_h)
    todo = [(n, i) for n, i in zip(selected, indices) if not frame_valid(clip, n, size)]
    log(f"[5] stylize ({CONTRACT}: tebal variabel + taper + jitter {'aktif' if g.jitter_on else 'mati'}, passes {g.passes}, "
        f"edge_mode {g.edge_mode}): {len(selected)} frame dipilih, "
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
                                description="Stage [5]: contours/ → strokes/ (SVG + PNG, tebal variabel + taper)")
    p.add_argument("--config", type=Path, default=None,
                   help=f"YAML pipeline (default: {DEFAULT_PIPELINE.as_posix()} kalau ada, selain itu default kode)")
    p.add_argument("--style", type=str, default=None,
                   help="nama preset (configs/styles/<nama>.yaml) atau path YAML style (default: kunci `style` di config = rough-sketch)")
    add_work_dir_arg(p)
    p.add_argument("--restart", action="store_true", help="hapus output [5] lama (strokes/) lalu hitung ulang")
    p.add_argument("--limit", type=int, default=None, help="hanya N frame pertama (dengan --from: N frame sejak K)")
    p.add_argument("--from", dest="start", type=int, default=None,
                   help="jendela mulai frame K: frame K..K+N-1; WAJIB bersama --limit N")
    args = p.parse_args(argv)
    reconfigure_stdio()

    def log(msg: str) -> None:
        print(msg, flush=True)

    try:
        cpath = args.config if args.config is not None else (DEFAULT_PIPELINE if DEFAULT_PIPELINE.is_file() else None)
        cfg = load_pipeline(cpath, overrides=work_dir_overrides(args.work_dir))
        spath = resolve_style(args.style if args.style is not None else cfg.style)
        style = load_style(spath)
        t0 = time.perf_counter()
        run = run_stylize(cfg, style, style_name(spath), restart=args.restart, limit=args.limit,
                          start=args.start, log=log)
        log(f"selesai: {run['processed']} diproses, {run['skipped']} dilewati ({time.perf_counter() - t0:.1f} s)")
    except (StageError, ConfigError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return EXIT_PRECONDITION
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
