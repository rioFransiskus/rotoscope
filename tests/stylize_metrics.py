"""Fungsi metrik objektif stage [5] (T-203a): kesetiaan geometri, penyelarasan raster, tinta di tepi, celah garis tengah,
konsistensi SVG ↔ PNG. Dipakai tests/test_stylize.py (sintetis + data nyata) dan scripts/strokes_preview.py bukan —
fungsi di sini murni (numpy), tanpa efek samping."""

from __future__ import annotations

import re

import cv2
import numpy as np

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


# ── SVG ────────────────────────────────────────────
def parse_svg(svg: bytes) -> list[sty.Piece]:
    """SVG (hasil render_svg) → daftar Piece (tipe dari id <g>; track_id tidak ada di SVG → 0)."""
    text = svg.decode("utf-8")
    out = []
    for typ, body in SVG_GROUP_RE.findall(text):
        for d in SVG_PATH_RE.findall(body):
            closed = d.endswith(" Z")
            nums = [float(v) for v in re.findall(r"-?\d+\.\d+", d)]
            out.append(sty.Piece(typ, closed, np.array(nums).reshape(-1, 2), 0))
    return out
