"""Fungsi metrik stage [5] multipass (T-403): "over", rasterisasi SVG per pass → alpha, invarian per pass (memakai ulang jitter_metrics),
statistik pemisahan. Murni (numpy), tanpa efek samping; dipakai tests/test_stylize_multipass.py dan scripts/ (papan, ukur, mutasi)."""

from __future__ import annotations

import re

import cv2
import jitter_metrics as jm
import numpy as np
from stylize_metrics import SvgPath, rasterize_svg

from rotoscope import stylize as sty

PASS_GROUP_RE = re.compile(r'<g id="(pass_\d+)" opacity="([0-9.]+)">')
TYPE_GROUP_RE = re.compile(r'<g id="pass_\d+_([a-z_]+)"([^>]*)>(.*?)</g>', re.S)
PATH_RE = re.compile(r'<path d="([^"]*)"/>')
SUBPATH_RE = re.compile(r"M([^MZ]*)Z")
OPACITY_ATTR_RE = re.compile(r'opacity="([0-9.]+)"')


def over(covs: list[np.ndarray], alphas: list[float]) -> np.ndarray:
    """f = 1 − Π(1 − a_k × cov_k) (komutatif)."""
    keep = np.ones_like(covs[0], dtype=np.float64)
    for c, a in zip(covs, alphas):
        keep *= 1.0 - a * c
    return 1.0 - keep


def svg_pass_layers(svg: bytes) -> list[tuple[float, list[tuple[float, SvgPath]]]]:
    """SVG multipass → [(opacity grup pass, [(opacity grup tipe (1,0 bila tanpa atribut), SvgPath)])] urut dokumen."""
    txt = svg.decode("utf-8")
    out = []
    for chunk in re.split(r'(?=<g id="pass_\d+" opacity=)', txt)[1:]:
        m = PASS_GROUP_RE.match(chunk)
        layers = []
        for typ, attrs, body in TYPE_GROUP_RE.findall(chunk):
            om = OPACITY_ATTR_RE.search(attrs)
            for d in PATH_RE.findall(body):
                polys = [np.array([float(v) for v in sub.split()]).reshape(-1, 2) for sub in SUBPATH_RE.findall(d)]
                layers.append((float(om.group(1)) if om else 1.0, SvgPath(typ, polys)))
        out.append((float(m.group(2)), layers))
    return out


def svg_ink_fraction(svg: bytes, g: sty.Geometry) -> np.ndarray:
    """f dari SVG: tiap pass: tipe dengan opacity sama = satu union (rasterize_svg nonzero), antar kelompok "over" (grup bersarang);
    lalu "over" antar pass dengan opacity grup. Independen dari kode raster."""
    keep = np.ones((g.out_h, g.out_w), np.float64)
    for pass_op, layers in svg_pass_layers(svg):
        inner = np.ones_like(keep)
        by_op: dict[float, list[SvgPath]] = {}
        for op, path in layers:
            by_op.setdefault(op, []).append(path)
        # tipe ber-scale 1,0 = satu union; tiap tipe ber-scale ≠ 1 = grup sendiri (id tipe berbeda → pisah per tipe)
        base = [p for op, p in layers if op == 1.0]
        if base:
            inner *= 1.0 - rasterize_svg(base, g)
        for typ in sorted({p.type for op, p in layers if op != 1.0}):
            ps = [(op, p) for op, p in layers if p.type == typ and op != 1.0]
            inner *= 1.0 - ps[0][0] * rasterize_svg([p for _, p in ps], g)
        keep *= 1.0 - pass_op * (1.0 - inner)
    return 1.0 - keep


def png_ink_fraction(png_rgb: np.ndarray, g: sty.Geometry) -> np.ndarray:
    paper, ink = np.array(g.paper, float), np.array(g.ink, float)
    ch = int(np.argmax(np.abs(paper - ink)))
    return (paper[ch] - png_rgb[..., ch].astype(float)) / (paper[ch] - ink[ch])


def alpha_report(a: np.ndarray, b: np.ndarray) -> dict:
    """Kesetaraan dua peta alpha (a vs acuan b): IoU (alpha > 0,5), L1 / massa, fraksi piksel |selisih| > 0,5 terhadap piksel tinta, rasio massa."""
    ia, ib = a > 0.5, b > 0.5
    ink = max(int(ib.sum()), 1)
    return {"iou": float((ia & ib).sum() / max((ia | ib).sum(), 1)), "l1": float(np.abs(a - b).sum() / max(b.sum(), 1.0)),
            "big_diff_frac": float((np.abs(a - b) > 0.5).sum() / ink), "mass_ratio": float(a.sum() / max(b.sum(), 1e-9))}


def ink_fraction_reference(passes: list[list[sty.Piece]], g: sty.Geometry) -> np.ndarray:
    """Implementasi ACUAN penuh-frame (sebelum optimasi jendela / tabel, Tahap 3): float32 per piksel. ink_fraction harus identik BIT."""
    keep = np.ones((g.out_h, g.out_w), np.float32)
    for k, pcs in enumerate(passes):
        inner = np.ones_like(keep)
        for scale, sub in sty.pass_layers(pcs, g):
            cov = cv2.resize(sty.render_mask(sub, g), (g.out_w, g.out_h), interpolation=cv2.INTER_AREA).astype(np.float32) / (sty.MASK_LEVELS - 1)
            inner *= 1.0 - scale * cov
        keep *= 1.0 - np.float32(g.pass_alpha(k)) * (1.0 - inner)
    return 1.0 - keep


# ── Invarian per pass ──────────────────────────────
def _excess(a: float, b: float) -> float:
    """a − b; −inf − (−inf) (tanpa ujung ekstensi) = 0."""
    return 0.0 if a == b else float(a - b)


def pass_invariants(passes: list[list[sty.Piece]], g: sty.Geometry, frame_index: int) -> list[dict]:
    """Untuk tiap pass k >= 1 terhadap pass 0: rasio Δ celah sambungan / toleransi, persilangan baru, seam strok tertutup (rasio
    seam_jump / interior_change_max terbesar; jumlah gagal kriteria relatif 1,25×; jumlah gagal KEDUANYA = relatif DAN batas Lipschitz,
    jm.seam_ok → harus 0), piksel tinta tepi berubah, intrusi ekstensi (≤ pass 0), min det Jacobian medan pass."""
    base = passes[0]
    out = []
    for k in range(1, len(passes)):
        gp = sty.pass_geometry(g, k)
        pk = passes[k]
        ch = jm.joint_changes(base, pk, gp)
        tol = jm.joint_tolerance(gp)
        closed = [(b, p) for b, p in zip(base, pk) if b.closed]
        out.append({"pass": k, "joint_n": len(ch), "joint_ratio": float(ch.max() / tol) if len(ch) else 0.0,
                    "new_crossings": jm.new_crossings(base, pk),
                    "seam_ratio_max": max([jm.seam_ratio(b, p) for b, p in closed] or [0.0]),
                    "seam_rel_fail": sum(jm.seam_ratio(b, p) > jm.SEAM_REL_TOL for b, p in closed),
                    "seam_fail": sum(not jm.seam_ok(b, p, gp) for b, p in closed),
                    "edge_ink_changed": jm.edge_ink_changed(base, pk, gp),
                    "intrusion_excess": _excess(jm.extension_intrusion(pk, gp), jm.extension_intrusion(base, gp)),
                    "min_det": float(jm.pieces_jacobian_min_det(base, gp, frame_index)),
                    "max_displacement_over_bound": float(jm.max_displacement(base, pk) / max(jm.displacement_bound(gp), 1e-12))})
    return out


def separation_stats(pieces0: list[sty.Piece], g: sty.Geometry, frame_index: int, k: int = 1) -> dict:
    """|D_k| (px ref) atas titik pass 0: rms, p05, p50, p95, maks (informasi; p50 ≈ offset pada zona interior)."""
    gp = sty.pass_geometry(g, k)
    pts = np.vstack([p.points for p in pieces0])
    d = np.hypot(*(sty.jitter_displacement(pts, [(0, len(pts))], [1], gp, frame_index) / g.unit).T)
    interior = sty.edge_fade(pts, g) > 0.99
    d = d[interior] if interior.any() else d
    return {"rms": float(np.sqrt((d ** 2).mean())), "p05": float(np.percentile(d, 5)), "p50": float(np.percentile(d, 50)),
            "p95": float(np.percentile(d, 95)), "max": float(d.max())}
