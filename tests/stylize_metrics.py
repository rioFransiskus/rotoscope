"""Fungsi metrik objektif stage [5] (T-203a): kesetiaan geometri, penyelarasan raster, tinta di tepi, celah garis tengah,
konsistensi SVG ↔ PNG. Dipakai tests/test_stylize.py (sintetis + data nyata) dan scripts/strokes_preview.py bukan —
fungsi di sini murni (numpy), tanpa efek samping."""

from __future__ import annotations

import re
from typing import NamedTuple

import cv2
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree

from rotoscope import stylize as sty

SVG_PATH_RE = re.compile(r'<path d="([^"]*)"/>')
SVG_GROUP_RE = re.compile(r'<g id="([a-z_]+)"[^>]*>(.*?)</g>', re.S)


# ── Kesetiaan geometri ─────────────────────────────
def point_polyline_distance(points: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Jarak tiap titik (N, 2) ke polyline TERBUKA poly (M, 2): jarak segmen eksak, vektor atas segmen."""
    if len(poly) == 1:
        return np.hypot(*(points - poly[0]).T)
    a, b = poly[:-1], poly[1:]
    ab = b - a
    l2 = np.maximum((ab**2).sum(1), 1e-12)
    out = np.empty(len(points))
    chunk = 512
    for s in range(0, len(points), chunk):
        q = points[s:s + chunk, None, :]
        t = np.clip(((q - a) * ab).sum(2) / l2, 0, 1)
        c = a + ab * t[..., None]
        out[s:s + chunk] = np.sqrt(((c - q) ** 2).sum(2)).min(1)
    return out


def closed_polyline(poly: np.ndarray) -> np.ndarray:
    return np.vstack([poly, poly[:1]])


def piece_curve(piece: sty.Piece) -> np.ndarray:
    return closed_polyline(piece.points) if piece.closed else piece.points


def stroke_deviation(stroke: dict, g: sty.Geometry, edge_mode: str | None = None) -> np.ndarray:
    """Jarak (px output) tiap titik kontur asli (terskala; titik di tepi dikecualikan pada mode hide) ke garis tengah
    akhir (gabungan jalur strok itu, tanpa ekor ekstensi di luar kanvas ikut dihitung — ekor tidak memengaruhi jarak titik
    kontur). Kosong bila strok tidak punya titik acuan."""
    mode = edge_mode or g.edge_mode
    stats = sty.new_stats()
    pieces = sty.stroke_pieces(stroke, g, stats)
    p = np.asarray(stroke["points"], float)
    ref = p * g.scale
    if mode == "hide":
        sides = sty.edge_sides(p, g.width, g.height)
        ref = ref[[not s for s in sides]]
    if len(ref) == 0 or not pieces:
        return np.zeros(0)
    d = np.min([point_polyline_distance(ref, piece_curve(pc)) for pc in pieces], axis=0)
    return d


# ── Raster ─────────────────────────────────────────
def coverage(img_rgb: np.ndarray, g: sty.Geometry) -> np.ndarray:
    """Cakupan tinta 0–1 per piksel dari PNG hasil render (kanal dengan kontras terbesar)."""
    paper, ink = np.array(g.paper, float), np.array(g.ink, float)
    ch = int(np.argmax(np.abs(paper - ink)))
    return (paper[ch] - img_rgb[..., ch].astype(float)) / (paper[ch] - ink[ch])


def decode_png(data: bytes) -> np.ndarray:
    bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def sample_polyline(poly: np.ndarray, step: float) -> np.ndarray:
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(poly, axis=0).T))]
    s = np.arange(0, d[-1], step) if d[-1] > 0 else np.zeros(1)
    return np.column_stack([np.interp(s, d, poly[:, 0]), np.interp(s, d, poly[:, 1])])


def centerline_gaps(cov: np.ndarray, pieces: list[sty.Piece], g: sty.Geometry, step: float = 0.5,
                    min_cov: float = 0.9) -> int:
    """Jumlah sampel garis tengah di DALAM kanvas yang piksel-nya bercakupan < min_cov (celah). Titik kontinu (x, y) →
    piksel (floor x, floor y)."""
    bad = 0
    for pc in pieces:
        for x, y in sample_polyline(piece_curve(pc), step):
            i, j = int(np.floor(x)), int(np.floor(y))
            if 0 <= i < g.out_w and 0 <= j < g.out_h and cov[j, i] < min_cov:
                bad += 1
    return bad


def line_alignment_error(cov: np.ndarray, p0: np.ndarray, p1: np.ndarray, x_lo: float, x_hi: float) -> tuple[float, float]:
    """Garis lurus p0 → p1 (px output, tidak vertikal) dengan tinta: (galat posisi, galat tebal) rata-rata kolom di
    x ∈ [x_lo, x_hi]. Posisi = centroid vertikal cakupan kolom − y garis di pusat kolom; tebal = jumlah cakupan kolom ×
    cos(sudut) − tebal nominal tidak dihitung di sini (dikembalikan tebal ukur vertikal rata-rata kolom)."""
    slope = (p1[1] - p0[1]) / (p1[0] - p0[0])
    errs, widths = [], []
    for i in range(int(np.ceil(x_lo - 0.5)), int(np.floor(x_hi - 0.5)) + 1):
        col = cov[:, i]
        if col.sum() <= 0:
            continue
        ys = np.arange(len(col)) + 0.5
        yc = float((col * ys).sum() / col.sum())
        errs.append(yc - (p0[1] + slope * (i + 0.5 - p0[0])))
        widths.append(float(col.sum()))
    return float(np.mean(errs)), float(np.mean(widths) / np.sqrt(1 + slope**2))


# ── Tepi ───────────────────────────────────────────
def border_ink(cov: np.ndarray, threshold: float = 0.05) -> dict:
    """Indeks (piksel) di baris / kolom terluar kanvas yang bertinta."""
    return {"bottom": np.flatnonzero(cov[-1, :] > threshold), "top": np.flatnonzero(cov[0, :] > threshold),
            "left": np.flatnonzero(cov[:, 0] > threshold), "right": np.flatnonzero(cov[:, -1] > threshold)}


def edge_run_midpoints(doc: dict, g: sty.Geometry, min_points: int = 10) -> np.ndarray:
    """Titik tengah (px output) tiap run tepi kontur sepanjang ≥ min_points titik. Pada mode hide tidak boleh ada garis
    tengah jalur di titik ini (jarak ≥ ~1 px; garis yang menyusur satu piksel di atas tepi berjarak 1,1·s); pada mode draw
    jaraknya ≈ 0."""
    out = []
    for s in doc["strokes"]:
        p = np.asarray(s["points"], float)
        closed = bool(s["closed"])
        if not closed and len(p) > 3 and np.array_equal(p[0], p[-1]):
            closed, p = True, p[:-1]
        on = np.array([bool(x) for x in sty.edge_sides(p, g.width, g.height)])
        if not on.any() or on.all():
            continue
        for run in sty.split_edge_runs(on, closed)[1]:
            if len(run) >= min_points:
                out.append(p[run[len(run) // 2]] * g.scale)
    return np.array(out).reshape(-1, 2)


def min_distance_to_pieces(points: np.ndarray, pieces: list[sty.Piece]) -> np.ndarray:
    if len(points) == 0 or not pieces:
        return np.zeros(0) if len(points) == 0 else np.full(len(points), np.inf)
    return np.min([point_polyline_distance(points, piece_curve(p)) for p in pieces], axis=0)


# ── SVG (T-401: poligon kontur terisi) ─────────────
SVG_SUBPATH_RE = re.compile(r"M([^MZ]*)Z")
SVG_PNG_IOU_MIN = 0.990           # keputusan Rio (docs/04 "Keputusan T-401" butir 3); terukur 0,9927–0,9947
SVG_PNG_L1_MAX = 0.010            # selisih cakupan absolut / tinta
SVG_PNG_BIG_DIFF_MAX = 0.002      # fraksi piksel tinta dengan |selisih cakupan| > 0,5 (hanya tikungan rapat / ujung)
REGRESSION_IOU_MIN = 0.990        # tebal konstan vs T-203a (lantai noise kuantisasi cv2 1/16 piksel ss)
INK_MASS_TOL = 0.005              # rasio massa tinta ±0,5% (frame nyata)
SYNTH_SVG_PNG_IOU_MIN = 0.980     # strok sintetis tunggal: rasio tepi / luas besar → IoU / L1 lebih peka (terukur 0,988 / 0,0117)
SYNTH_SVG_PNG_L1_MAX = 0.015
SYNTH_INK_MASS_TOL = 0.015


class SvgPath(NamedTuple):
    type: str
    polys: list[np.ndarray]       # sub-path (px output); terbuka = 1, tertutup = 2 (luar + dalam)


def parse_svg(svg: bytes) -> list[SvgPath]:
    """SVG (hasil render_svg) → daftar jalur (tipe dari id <g>; urutan dokumen)."""
    out = []
    for typ, body in SVG_GROUP_RE.findall(svg.decode("utf-8")):
        for d in SVG_PATH_RE.findall(body):
            polys = [np.array([float(v) for v in sub.split()]).reshape(-1, 2) for sub in SVG_SUBPATH_RE.findall(d)]
            out.append(SvgPath(typ, polys))
    return out


def winding_grid(polys: list[np.ndarray], shape: tuple[int, int], ss: int) -> np.ndarray:
    """Bilangan lilitan tiap piksel supersampling (pusat piksel ss) untuk poligon px output. Independen dari cv2 (fill nonzero
    SVG): tiap sisi menambah ±1 pada kolom perpotongan scanline; jumlah kumulatif sepanjang x = lilitan."""
    h, w = shape
    delta = np.zeros((h * ss, w * ss + 1), np.int32)
    for poly in polys:
        a = poly * ss
        b = np.roll(a, -1, axis=0)
        for (x0, y0), (x1, y1) in zip(a, b):
            if y0 == y1:
                continue
            sign = 1 if y1 > y0 else -1
            ylo, yhi = min(y0, y1), max(y0, y1)
            r0, r1 = max(int(np.ceil(ylo - 0.5)), 0), min(int(np.ceil(yhi - 0.5)), h * ss)
            if r1 <= r0:
                continue
            rows = np.arange(r0, r1)
            xs = x0 + (rows + 0.5 - y0) * (x1 - x0) / (y1 - y0)
            cols = np.clip(np.ceil(xs - 0.5).astype(int), 0, w * ss)
            np.add.at(delta, (rows, cols), sign)
    return np.cumsum(delta, axis=1)[:, :w * ss]


def rasterize_svg(paths: list[SvgPath], g: sty.Geometry) -> np.ndarray:
    """Cakupan 0–1 per piksel output dari SVG: tiap <path> diisi nonzero sendiri-sendiri, lalu union (INTER_AREA dari grid ss)."""
    mask = np.zeros((g.out_h * g.ss, g.out_w * g.ss), bool)
    for p in paths:
        mask |= winding_grid(p.polys, (g.out_h, g.out_w), g.ss) != 0
    return cv2.resize(mask.astype(np.uint8) * 255, (g.out_w, g.out_h), interpolation=cv2.INTER_AREA).astype(np.float64) / 255.0


def cov_report(a: np.ndarray, b: np.ndarray) -> dict:
    """Kesetaraan dua peta cakupan (a vs acuan b): IoU (cakupan > 0,5), L1 / tinta, fraksi piksel selisih besar, rasio massa."""
    ia, ib = a > 0.5, b > 0.5
    ink = max(int(ib.sum()), 1)
    return {"iou": float((ia & ib).sum() / max((ia | ib).sum(), 1)), "l1": float(np.abs(a - b).sum() / max(b.sum(), 1.0)),
            "big_diff_frac": float((np.abs(a - b) > 0.5).sum() / ink), "mass_ratio": float(a.sum() / max(b.sum(), 1e-9))}


def svg_png_report(paths: list[SvgPath], cov_png: np.ndarray, g: sty.Geometry) -> dict:
    return cov_report(rasterize_svg(paths, g), cov_png)


# ── Tebal ──────────────────────────────────────────
def ink_width_along_normal(cov: np.ndarray, p: np.ndarray, n: np.ndarray, half: float = 14.0, step: float = 0.25) -> float:
    """Tebal tinta terukur (px) di titik p sepanjang normal n: integral cakupan (bilinear). Hanya untuk strok terisolasi."""
    ts = np.arange(-half, half + 1e-9, step)
    xs, ys = p[0] + ts * n[0] - 0.5, p[1] + ts * n[1] - 0.5          # indeks piksel = koordinat − 0,5 (pusat piksel k + 0,5)
    vals = map_coordinates(cov, [ys, xs], order=1, mode="constant", cval=0.0)
    return float(vals.sum() * step)


def piece_arclength(pc: sty.Piece) -> np.ndarray:
    return np.r_[0.0, np.cumsum(np.hypot(*np.diff(pc.points, axis=0).T))]


def seam_jump(pc: sty.Piece, g: sty.Geometry) -> tuple[float, float]:
    """Strok tertutup: (selisih tebal titik awal ↔ titik terakhir, selisih maksimum antar titik bertetangga), px output."""
    w = sty.piece_widths(pc, g)
    return float(abs(w[0] - w[-1])), float(np.abs(np.diff(w)).max())


def pop_by_type(prev: list[sty.Piece], cur: list[sty.Piece], g: sty.Geometry, base: float, match: float = 3.0,
                still: float = 1.0) -> dict[str, dict[str, list[np.ndarray]]]:
    """"Pop tebal" antar frame berurutan: untuk tiap titik strok di t, titik terdekat di strok bertrack sama di t−1 (jarak ≤ match
    px ref); |Δtebal| dibagi `base` (= width_base × unit) dan dibagi tebal NOMINAL tipe itu sendiri (g.widths[tipe]).
    Dipisah titik DIAM (jarak ≤ still px ref) dan BERGERAK. Return {tipe: {"base_still", "base_move", "own_still", "own_move"}}."""
    out = {t: {"base_still": [], "base_move": [], "own_still": [], "own_move": []} for t in sty.TYPE_ORDER}
    by_prev: dict[tuple[int, str], list[sty.Piece]] = {}
    for pc in prev:
        by_prev.setdefault((pc.track_id, pc.type), []).append(pc)
    for pc in cur:
        olds = by_prev.get((pc.track_id, pc.type))
        if not olds:
            continue
        q = np.vstack([o.points for o in olds])
        qw = np.concatenate([sty.piece_widths(o, g) for o in olds])
        d, j = cKDTree(q).query(pc.points)
        ok = d <= match * g.unit
        if not ok.any():
            continue
        dw = np.abs(sty.piece_widths(pc, g)[ok] - qw[j[ok]])
        near = d[ok] <= still * g.unit
        for key, nominal in (("base", base), ("own", g.widths[pc.type])):
            out[pc.type][f"{key}_still"].append(dw[near] / nominal)
            out[pc.type][f"{key}_move"].append(dw[~near] / nominal)
    return out


def summarize_pop(acc: dict) -> dict:
    """Gabungkan daftar array pop_by_type → {tipe: {kunci: {n, p50, p95, max}}}."""
    res = {}
    for t, kv in acc.items():
        res[t] = {}
        for k, lst in kv.items():
            a = np.concatenate(lst) if lst else np.zeros(0)
            res[t][k] = {"n": int(len(a)), "p50": float(np.percentile(a, 50)) if len(a) else 0.0,
                         "p95": float(np.percentile(a, 95)) if len(a) else 0.0, "max": float(a.max()) if len(a) else 0.0}
    return res


def taper_depth(pieces: list[sty.Piece], flags: list[tuple[bool, bool]], g: sty.Geometry) -> dict[str, float]:
    """Kedalaman taper per tipe: median (tebal di ujung bebas / tebal nominal tipe). 1,0 = tanpa taper. Dipasangkan dengan pop
    supaya taper mati tidak tampak "menang"."""
    res: dict[str, list[float]] = {}
    for pc, fl in zip(pieces, flags):
        w = sty.piece_widths(pc, g)
        for e, free in enumerate(fl):
            if free:
                res.setdefault(pc.type, []).append(float(w[0 if e == 0 else -1] / g.widths[pc.type]))
    return {t: float(np.median(v)) for t, v in res.items()}
