"""Fungsi metrik keselamatan jitter stage [5] (T-402): batas |D|, Jacobian, sambungan, persilangan baru, tinta tepi, ujung ekstensi,
statistik getar. Dipakai tests/test_stylize_jitter.py (sintetis + data nyata) dan scripts/ (papan, mutasi). Murni (numpy / scipy),
tanpa efek samping. Ambang = konstanta bernama di stylize.py (JACOBIAN_MIN_DET, JITTER_JOINT_TOL_GRAD)."""

from __future__ import annotations

import math

import cv2
import numpy as np
from scipy.spatial import cKDTree
from stylize_metrics import piece_curve, point_polyline_distance

from rotoscope import stylize as sty

JACOBIAN_STEP_PX = 0.25           # langkah beda hingga (px output) untuk ∇D
CROSS_EXCLUDE_PX = sty.JOIN_DIST_PX     # persilangan dalam jarak ini dari ujung strok terbuka = kontak sambungan, bukan "garis bersilang"
INTRUDE_TOL_PX = 0.25             # tutup bulat ujung ekstensi masuk kanvas > ini = tertarik masuk


# ── Batas dan Jacobian ─────────────────────────────
def displacement_at(g: sty.Geometry, frame_index: int, pts: np.ndarray, track_id: int = 1) -> np.ndarray:
    pts = np.asarray(pts, dtype=float)
    return sty.jitter_displacement(pts, [(0, len(pts))], [track_id], g, frame_index)


def displacement_bound(g: sty.Geometry) -> float:
    """|D| ≤ amplitude × unit × B(s), B(s) = √2 (√(1 − s) + √s) (px output)."""
    s = g.jitter_s
    return g.jitter_amp * math.sqrt(2.0) * (math.sqrt(1.0 - s) + math.sqrt(s))


def max_displacement(base: list[sty.Piece], jit: list[sty.Piece]) -> float:
    """Perpindahan titik terbesar (px output) antara jalur tanpa dan dengan jitter (titik berpasangan)."""
    return max([float(np.hypot(*(b.points - j.points).T).max()) for b, j in zip(base, jit)] or [0.0])


def jacobian_dets(g: sty.Geometry, frame_index: int, pts: np.ndarray, track_id: int = 1, h: float = JACOBIAN_STEP_PX) -> np.ndarray:
    """det(I + ∇D) pada titik-titik (beda hingga pusat; φ tepi ikut karena D fungsi posisi)."""
    pts = np.asarray(pts, dtype=float)
    ex, ey = np.array([h, 0.0]), np.array([0.0, h])
    dx = (displacement_at(g, frame_index, pts + ex, track_id) - displacement_at(g, frame_index, pts - ex, track_id)) / (2 * h)
    dy = (displacement_at(g, frame_index, pts + ey, track_id) - displacement_at(g, frame_index, pts - ey, track_id)) / (2 * h)
    return (1.0 + dx[:, 0]) * (1.0 + dy[:, 1]) - dy[:, 0] * dx[:, 1]


def pieces_jacobian_min_det(pieces: list[sty.Piece], g: sty.Geometry, frame_index: int) -> float:
    """min det(I + ∇D) pada titik strok yang digambar (ambang JACOBIAN_MIN_DET; medan dasar sebelum perpindahan)."""
    return min([float(jacobian_dets(g, frame_index, pc.points, pc.track_id).min()) for pc in pieces if len(pc.points)] or [1.0])


# ── Sambungan ──────────────────────────────────────
def end_gap(pieces: list[sty.Piece], i: int, e: int) -> float:
    """Jarak ujung e dari jalur i ke garis tengah jalur strok LAIN terdekat (px output); inf bila tidak ada."""
    pc = pieces[i]
    end = pc.points[0 if e == 0 else -1][None, :]
    d = [float(point_polyline_distance(end, piece_curve(q))[0]) for q in pieces if q.stroke_idx != pc.stroke_idx or pc.stroke_idx < 0 and q is not pc]
    return min(d) if d else math.inf


def joint_ends(pieces: list[sty.Piece], g: sty.Geometry) -> list[tuple[int, int]]:
    """Ujung terbuka bukan-tepi yang bertemu strok lain (celah ≤ join_dist): (indeks jalur, ujung)."""
    return [(i, e) for i, pc in enumerate(pieces) if not pc.closed for e in (0, 1)
            if not pc.edge[e] and end_gap(pieces, i, e) <= g.join_dist]


def joint_changes(base: list[sty.Piece], jit: list[sty.Piece], g: sty.Geometry) -> np.ndarray:
    """|perubahan celah| tiap sambungan (px output); sambungan ditentukan dari geometri TANPA jitter."""
    return np.array([abs(end_gap(jit, i, e) - end_gap(base, i, e)) for i, e in joint_ends(base, g)])


def joint_tolerance(g: sty.Geometry) -> float:
    """Δ celah ≤ |∇D| × celah ≤ JITTER_JOINT_TOL_GRAD × r × join_dist (px output), r = amplitude × frequency (s = 0)."""
    return sty.JITTER_JOINT_TOL_GRAD * (g.jitter_amp / g.jitter_cell) * g.join_dist


def seam_jump(base: sty.Piece, jit: sty.Piece) -> float:
    """Perubahan panjang segmen penutup strok tertutup (px) — retak seam bila jauh lebih besar dari segmen interior."""
    def closing(pc):
        return float(np.hypot(*(pc.points[0] - pc.points[-1])))
    return abs(closing(jit) - closing(base))


def interior_change_max(base: sty.Piece, jit: sty.Piece) -> float:
    lb = np.hypot(*np.diff(base.points, axis=0).T)
    lj = np.hypot(*np.diff(jit.points, axis=0).T)
    return float(np.abs(lj - lb).max())


# Seam strok tertutup (revisi metrik T-403, docs/04): medan koheren kontinu → perubahan panjang segmen penutup tidak bisa melebihi
# Lipschitz(D) × panjang segmen. Ambang absolut "+0,05 px" salah untuk amplitudo besar; diganti dua kriteria (lulus bila SALAH SATU):
#  (a) relatif: seam_jump ≤ SEAM_REL_TOL × interior_change_max (strok yang sama);
#  (b) batas teoretis: seam_jump ≤ batas Lipschitz × panjang segmen penutup.
SEAM_REL_TOL = 1.25
FIELD_LIPSCHITZ_R = 4.5     # σ_maks(∇F) / r, r = amplitudo / sel: terukur 4,25 (200 000 titik interior, maks), p99 3,55; margin ~6%
EDGE_PHI_SLOPE = 1.5        # maks |dφ/dd| × panjang pelunakan (smoothstep: 1,5)


def seam_ratio(base: sty.Piece, jit: sty.Piece) -> float:
    """seam_jump / interior_change_max (strok yang sama)."""
    return seam_jump(base, jit) / max(interior_change_max(base, jit), 1e-9)


def field_lipschitz(g: sty.Geometry, pts: np.ndarray) -> float:
    """Konstanta Lipschitz (tak berdimensi) medan D di sekitar titik `pts`: FIELD_LIPSCHITZ_R × r × (√(1−s)+√s) + |D|maks × maks|∇φ| bila
    ada titik di zona pelunakan tepi. g = geometri medan (jitter / pass)."""
    if g.jitter_cell <= 0 or g.jitter_amp <= 0:
        return 0.0
    s = g.jitter_s
    lip = FIELD_LIPSCHITZ_R * (g.jitter_amp / g.jitter_cell) * (math.sqrt(1.0 - s) + math.sqrt(s))
    if float(sty.edge_fade(pts, g).min()) < 1.0:
        lip += g.jitter_amp * 2.0 * EDGE_PHI_SLOPE / max(g.edge_fade_len, 1e-9)       # |D| ≤ amp × 2 (dua kanal √2 → ≤ 2): konservatif
    return lip


def seam_bound(base: sty.Piece, g: sty.Geometry) -> float:
    """Batas teoretis perubahan segmen penutup (px output) = Lipschitz × panjang segmen penutup pada geometri tanpa jitter."""
    return field_lipschitz(g, base.points[[0, -1]]) * float(np.hypot(*(base.points[0] - base.points[-1])))


def interior_bound(base: sty.Piece, g: sty.Geometry) -> float:
    """Batas teoretis perubahan segmen interior terbesar = Lipschitz × segmen interior terpanjang (menjaga kriteria relatif: retak
    di tengah strok ikut tertangkap, bukan hanya retak di seam)."""
    return field_lipschitz(g, base.points) * float(np.hypot(*np.diff(base.points, axis=0).T).max())


def seam_ok(base: sty.Piece, jit: sty.Piece, g: sty.Geometry) -> bool:
    """Lulus bila interior dalam batas Lipschitz DAN (seam ≤ 1,25 × interior ATAU seam ≤ batas Lipschitz seam)."""
    sj, ic = seam_jump(base, jit), interior_change_max(base, jit)
    return ic <= interior_bound(base, g) + 1e-9 and (sj <= SEAM_REL_TOL * ic + 1e-9 or sj <= seam_bound(base, g) + 1e-9)


# ── Persilangan ────────────────────────────────────
def crossings(pieces: list[sty.Piece]) -> list[tuple[int, int, np.ndarray]]:
    """Persilangan segmen strok-strok BERBEDA: (pemilik a, pemilik b, titik). Kandidat via KD-tree titik tengah segmen."""
    A, B, own = [], [], []
    for k, pc in enumerate(pieces):
        q = np.vstack([pc.points, pc.points[:1]]) if pc.closed else pc.points
        A.append(q[:-1])
        B.append(q[1:])
        own.append(np.full(len(q) - 1, pc.stroke_idx if pc.stroke_idx >= 0 else -(k + 2)))
    A, B, own = np.vstack(A), np.vstack(B), np.concatenate(own)
    if len(A) < 2:
        return []
    mid = (A + B) / 2
    r = float(np.hypot(*(B - A).T).max()) + 0.05
    pr = cKDTree(mid).query_pairs(r, output_type="ndarray")
    if len(pr) == 0:
        return []
    i, j = pr[:, 0], pr[:, 1]
    keep = own[i] != own[j]
    i, j = i[keep], j[keep]
    p1, p2, p3, p4 = A[i], B[i], A[j], B[j]

    def orient(a, b, c):
        return (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])

    o1, o2, o3, o4 = orient(p1, p2, p3), orient(p1, p2, p4), orient(p3, p4, p1), orient(p3, p4, p2)
    hit = (o1 * o2 < 0) & (o3 * o4 < 0)
    t = o1[hit] / (o1[hit] - o2[hit])
    pts = p3[hit] + (p4[hit] - p3[hit]) * t[:, None]
    return [(int(min(a, b)), int(max(a, b)), p) for a, b, p in zip(own[i][hit], own[j][hit], pts)]


def new_crossings(base: list[sty.Piece], jit: list[sty.Piece], exclude_px: float = CROSS_EXCLUDE_PX) -> int:
    """Jumlah persilangan BARU antar strok: hanya pasangan strok yang TIDAK bersilangan di geometri tanpa jitter (pasangan yang sudah
    bersilang / bertumpuk di ujung sambungan tidak dihitung; terukur: pertambahan 1 → 2 persilangan pada pasangan batas grup ↔ siluet
    yang sudah tumpang tindih); persilangan dalam `exclude_px` dari ujung strok terbuka mana pun juga tidak dihitung."""
    def count(pieces):
        ends = [pc.points[e] for pc in pieces if not pc.closed for e in (0, -1)]
        tree = cKDTree(np.array(ends)) if ends else None
        out: dict = {}
        for a, b, p in crossings(pieces):
            if tree is not None and tree.query(p)[0] <= exclude_px:
                continue
            out[(a, b)] = out.get((a, b), 0) + 1
        return out
    cb, cj = count(base), count(jit)
    already = {(a, b) for a, b, _ in crossings(base)}                    # pasangan yang bersilang / bertumpuk SEBELUM jitter (juga di dekat ujung)
    return sum(v for k, v in cj.items() if k not in already and k not in cb)


# ── Tepi ───────────────────────────────────────────
def coverage_u8(pieces: list[sty.Piece], g: sty.Geometry) -> np.ndarray:
    return cv2.resize(sty.render_mask(pieces, g), (g.out_w, g.out_h), interpolation=cv2.INTER_AREA)


def edge_ink_changed(base: list[sty.Piece], jit: list[sty.Piece], g: sty.Geometry, depth: int = 3) -> int:
    """Jumlah piksel berubah di `depth` baris / kolom terluar (bawah, kiri, kanan) antara T-401 dan jitter."""
    a, b = coverage_u8(base, g).astype(int), coverage_u8(jit, g).astype(int)
    n = 0
    for sl in (np.s_[-depth:, :], np.s_[:, :depth], np.s_[:, -depth:]):
        n += int((a[sl] != b[sl]).sum())
    return n


def rect_distance(q: np.ndarray, g: sty.Geometry) -> float:
    """Jarak titik ke kanvas [0, W] × [0, H] (positif di luar, negatif = kedalaman di dalam)."""
    dx = max(-q[0], q[0] - g.out_w, 0.0)
    dy = max(-q[1], q[1] - g.out_h, 0.0)
    if dx > 0 or dy > 0:
        return math.hypot(dx, dy)
    return -min(q[0], g.out_w - q[0], g.out_h - q[1])


def extension_intrusion(pieces: list[sty.Piece], g: sty.Geometry) -> float:
    """Kedalaman terbesar (px) tutup bulat ujung ekstensi yang masuk kanvas (jari-jari − jarak ke kanvas); ≤ 0 = tidak ada."""
    worst = -math.inf
    for pc in pieces:
        if pc.closed or pc.widths is None:
            continue
        for e in (0, 1):
            if pc.edge[e]:
                q = pc.points[0 if e == 0 else -1]
                worst = max(worst, float(pc.widths[0 if e == 0 else -1]) / 2.0 - rect_distance(q, g))
    return worst


# ── Statistik getar (informasi, BUKAN kriteria lulus) ──
def jitter_stats(g: sty.Geometry, frame_index: int, pts: np.ndarray) -> dict:
    d = displacement_at(g, frame_index, pts)
    a = max(g.jitter_amp, 1e-12)
    return {"rms_per_channel_over_amp": float(math.sqrt((d ** 2).mean())) / a,
            "max_abs_over_amp": float(np.abs(d).max()) / a, "max_norm_over_amp": float(np.hypot(*d.T).max()) / a}
