"""Look test T-102c (D-009): bisakah seg 0.8B + DA-V2 mencapai style target (docs/00, docs/02)?

Alat sekali pakai, BUKAN modul pipeline (bukan implementasi stage [4]/[5]). TANPA inferensi: memakai
class map seg 0.8B full run (work/t102c/exp/seg_seg_0.8b/) + kedalaman DA-V2-Small tersimpan
(work/t102c/exp/depth_da2s_gpu/). Jalankan dari root repo:
    venv/Scripts/python.exe scripts/look_test.py diag     # [garis seg | garis DA | gabungan] preview sekarang
    venv/Scripts/python.exe scripts/look_test.py stats    # data untuk memilih threshold rendah, D, L
    venv/Scripts/python.exe scripts/look_test.py render   # look_frames.png + look_f073-092.mp4 + look_f183-202.mp4
Output: work/t102c/look/

Peta grup = class map 0.8B -> (a) filter pulau N -> grup (scripts/sapiens2_groups.json) -> (b) mode filter K
(sama dengan filtered_full_0.8b). Preview "sekarang" = garis grup + garis DA relative_jump > p95 + (c) M.

Garis kedalaman BARU (tipis + selektif), dari disparity DA-V2:
  1. Besaran tepi = |grad log d| (Sobel 3x3 / 8 -> satuan: selisih log per px) setelah Gaussian blur
     sigma --depth-blur. log membuat gradien RELATIF (sama untuk Z dan 1/Z, seperti relative_jump).
  2. Non-maximum suppression searah gradien (arah dikuantisasi 4 bin, seperti Canny): piksel dipertahankan
     kalau besarnya >= kedua tetangga searah gradien.
  3. Hysteresis: piksel NMS > T_high = kuat; > T_low = lemah; komponen lemah (8-arah) dipertahankan kalau
     memuat >= 1 piksel kuat. T_high / T_low = persentil PER KLIP (283 frame) dari |grad log d| di foreground
     ter-erode (--lines-erode), sama dengan wilayah p95 preview sekarang.
  4. Hanya di DALAM satu grup: jarak ke batas grup terdekat (termasuk siluet luar) >= --depth-min-dist px.
  5. Skeleton (thinning), komponen 8-arah dengan panjang skeleton < --depth-min-len px dibuang.

Stroke sederhana (bukan stage 5 lengkap), default = preset rough-sketch docs/02:
  batas grup (morph. gradient 3x3 peta grup -> thinning) + garis kedalaman -> tracing skeleton jadi
  polyline (junction dilepas lalu disambung ke ujung) -> approxPolyDP(simplify_epsilon) -> spline
  Catmull-Rom (tangen = smooth_tension x (p[i+1] - p[i-1]); 0.5 = Catmull-Rom standar) -> resample ->
  tebal = width_base x (1 + width_variation x noise(s x width_noise_scale)) x taper ujung -> jitter searah
  normal = amplitude x noise(s x frequency) -> multipass -> digambar (supersampling, anti-alias) di atas
  warna kertas. Noise 1D = value noise (interpolasi smoothstep) dengan seed = crc32(frame_index,
  param_seed) (P-007) + indeks stroke + indeks pass.
Tidak ada tekstur kertas/brush dan TIDAK ada stabilisasi temporal (stage [3]).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import zlib
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ab_segment as ab  # noqa: E402
import sapiens2_exp as ex  # noqa: E402
import sapiens2_probe as probe  # noqa: E402

ROOT = probe.ROOT
LABEL_NOTE = "tanpa stabilisasi temporal: getaran antar frame BELUM representatif"


# --- data ------------------------------------------------------------------
class Clip:
    """Peta grup 0.8B (post a+b) + disparity DA per frame, dengan cache."""

    def __init__(self, args):
        meta = json.loads((args.work / "meta.json").read_text(encoding="utf-8"))
        self.names = ab.frame_names(meta)
        self.hw = (meta["working_height"], meta["working_width"])
        self.frames_dir = args.work / "frames"
        self.seg_dir = args.exp / f"seg_{args.model}"
        self.lut = ex.group_lut(args.t102c / "seg")
        self.args = args
        self._g: dict[str, np.ndarray] = {}
        bad = [n for n in self.names if not (ex.valid_seg(self.seg_dir / n, self.hw)
                                             and ex.valid_depth(ex.depth_path(args.exp, "da2s_gpu", n), self.hw))]
        if bad:
            ab.fail(f"{len(bad)} frame belum lengkap (seg {args.model} / DA), mis. {bad[0]}")

    def name(self, frame_no: int) -> str:
        return f"frame_{frame_no:05d}.png"

    def groups(self, n: str) -> np.ndarray:
        if n not in self._g:
            a = self.args
            seg = ex.read_class_map(self.seg_dir / n)
            self._g[n] = ex.mode_filter(self.lut[ex.island_filter(seg, a.island_min)], a.mode_k)
        return self._g[n]

    def depth(self, n: str) -> np.ndarray:
        return np.load(ex.depth_path(self.args.exp, "da2s_gpu", n)).astype(np.float32)

    def bgr(self, n: str) -> np.ndarray:
        return cv2.imread(str(self.frames_dir / n), cv2.IMREAD_COLOR)

    def inner(self, g: np.ndarray) -> np.ndarray:
        k = np.ones((self.args.lines_erode, self.args.lines_erode), np.uint8)
        return cv2.erode((g != 0).astype(np.uint8), k).astype(bool)


# --- preview sekarang (sama dengan filtered_full) --------------------------
def current_lines(clip: Clip, n: str, thr: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = clip.args
    g, z = clip.groups(n), clip.depth(n)
    s = ex.seg_line_mask(g, ex.IDENTITY_LUT, a.line_px)
    d = ex.depth_line_mask(g, z, a.lines_erode, a.lines_eps, thr)
    return s, d, ex.component_filter(s | d, a.line_min)


# --- garis kedalaman baru --------------------------------------------------
def log_gradient(z: np.ndarray, blur: float, eps: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lz = np.log(np.maximum(z, eps))
    if blur > 0:
        lz = cv2.GaussianBlur(lz, (0, 0), blur)
    gx = cv2.Sobel(lz, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(lz, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    return np.sqrt(gx * gx + gy * gy), gx, gy


def nms(mag: np.ndarray, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """True di piksel yang besarnya >= kedua tetangga searah gradien (4 bin arah: 0/45/90/135 derajat)."""
    ang = (np.rad2deg(np.arctan2(gy, gx)) + 180.0) % 180.0
    b = np.digitize(ang, [22.5, 67.5, 112.5, 157.5]) % 4  # 0: horizontal, 1: 45, 2: vertikal, 3: 135
    p = np.pad(mag, 1, mode="edge")
    h, w = mag.shape

    def sh(dy: int, dx: int) -> np.ndarray:
        return p[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]

    # gradien horizontal (bin 0) -> bandingkan kiri/kanan; bin 1 (45 derajat, y ke bawah) -> diagonal
    pairs = {0: ((0, -1), (0, 1)), 1: ((-1, -1), (1, 1)), 2: ((-1, 0), (1, 0)), 3: ((-1, 1), (1, -1))}
    keep = np.zeros(mag.shape, bool)
    for k, (q1, q2) in pairs.items():
        keep |= (b == k) & (mag >= sh(*q1)) & (mag >= sh(*q2))
    return keep & (mag > 0)


def hysteresis(cand: np.ndarray, mag: np.ndarray, lo: float, hi: float) -> np.ndarray:
    weak = cand & (mag > lo)
    k, lab = cv2.connectedComponents(weak.astype(np.uint8), connectivity=8)
    has_strong = np.zeros(k, bool)
    has_strong[np.unique(lab[weak & (mag > hi)])] = True
    has_strong[0] = False
    return has_strong[lab]


def group_boundary_dist(g: np.ndarray) -> np.ndarray:
    """Jarak (px) ke batas grup terdekat, termasuk siluet luar (batas ke grup 0)."""
    bnd = probe.boundary(g)
    return cv2.distanceTransform((~bnd).astype(np.uint8), cv2.DIST_L2, 3)


def skeleton_length_filter(mask: np.ndarray, min_len: int) -> tuple[np.ndarray, list[int]]:
    """Thinning -> buang komponen 8-arah dengan panjang skeleton (jumlah piksel) < min_len."""
    sk = cv2.ximgproc.thinning(mask.astype(np.uint8) * 255) > 0
    k, lab, st, _ = cv2.connectedComponentsWithStats(sk.astype(np.uint8), connectivity=8)
    lens = st[1:, cv2.CC_STAT_AREA]
    keep = np.zeros(k, bool)
    keep[1:] = lens >= min_len
    return keep[lab], lens.tolist()


def depth_edges(clip: Clip, n: str, lo: float, hi: float, stage: str = "final") -> np.ndarray:
    """Garis kedalaman baru; stage: 'nms' (NMS+hysteresis di foreground), 'group' (+ jarak grup), 'final'."""
    a = clip.args
    g = clip.groups(n)
    mag, gx, gy = log_gradient(clip.depth(n), a.depth_blur, a.lines_eps)
    region = clip.inner(g)
    e = hysteresis(nms(mag, gx, gy) & region, mag, lo, hi)
    if stage == "nms":
        return e
    e &= group_boundary_dist(g) >= a.depth_min_dist
    if stage == "group":
        return e
    return skeleton_length_filter(e, a.depth_min_len)[0]


def clip_thresholds(clip: Clip) -> dict:
    a = clip.args
    vals = []
    for n in clip.names:
        g = clip.groups(n)
        mag, _, _ = log_gradient(clip.depth(n), a.depth_blur, a.lines_eps)
        vals.append(mag[clip.inner(g)])
    v = np.concatenate(vals)
    return {"hi": float(np.percentile(v, a.depth_hi_pct)), "lo": float(np.percentile(v, a.depth_lo_pct)),
            "hi_pct": a.depth_hi_pct, "lo_pct": a.depth_lo_pct, "n_values": int(v.size),
            "pct_table": {f"p{p:g}": float(np.percentile(v, p)) for p in (50, 70, 75, 80, 85, 88, 90, 92, 95, 98)}}


# --- tracing polyline ------------------------------------------------------
_N8 = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]  # 4-arah dulu


def trace_polylines(skel: np.ndarray, min_px: int) -> list[np.ndarray]:
    """Skeleton 1 px -> daftar polyline (x, y). Junction (>= 3 tetangga) dilepas, jalur dilacak,
    lalu ujung jalur disambung ke piksel junction yang bersebelahan. Jalur < min_px piksel dibuang."""
    sk = skel.astype(bool)
    nb = cv2.filter2D(sk.astype(np.uint8), -1, np.ones((3, 3), np.float32), borderType=cv2.BORDER_CONSTANT) - sk
    junc = sk & (nb >= 3)
    rest = sk & ~junc
    k, lab, st, _ = cv2.connectedComponentsWithStats(rest.astype(np.uint8), connectivity=8)
    h, w = sk.shape
    polys = []
    for i in range(1, k):
        if st[i, cv2.CC_STAT_AREA] < min_px:
            continue
        x0, y0, bw, bh = st[i, :4]
        ys, xs = np.nonzero(lab[y0:y0 + bh, x0:x0 + bw] == i)
        pts = set(zip((ys + y0).tolist(), (xs + x0).tolist()))
        deg = {p: sum((p[0] + dy, p[1] + dx) in pts for dy, dx in _N8) for p in pts}
        visited: set = set()
        while len(visited) < len(pts):
            left = [p for p in pts if p not in visited]
            ends = [p for p in left if deg[p] <= 1]
            cur = min(ends) if ends else min(left)
            path = [cur]
            visited.add(cur)
            while True:
                nxt = next(((cur[0] + dy, cur[1] + dx) for dy, dx in _N8
                            if (cur[0] + dy, cur[1] + dx) in pts and (cur[0] + dy, cur[1] + dx) not in visited), None)
                if nxt is None:
                    break
                path.append(nxt)
                visited.add(nxt)
                cur = nxt
            if len(path) < min_px:
                continue
            for end, pos in ((path[0], 0), (path[-1], len(path))):
                j = next(((end[0] + dy, end[1] + dx) for dy, dx in _N8
                          if 0 <= end[0] + dy < h and 0 <= end[1] + dx < w and junc[end[0] + dy, end[1] + dx]), None)
                if j is not None:
                    path.insert(pos, j)
            if len(path) > 3 and max(abs(path[0][0] - path[-1][0]), abs(path[0][1] - path[-1][1])) <= 1:
                path.append(path[0])  # jalur tertutup
            polys.append(np.array([(x, y) for y, x in path], np.float32))
    return polys


# --- stroke ----------------------------------------------------------------
def value_noise(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Noise 1D halus di [-1, 1]: nilai acak di titik bulat, interpolasi smoothstep."""
    n = int(np.floor(x.max())) + 2 if x.size else 2
    lat = rng.uniform(-1.0, 1.0, n + 1)
    i = np.floor(x).astype(int)
    f = x - i
    t = f * f * (3 - 2 * f)
    return lat[i] * (1 - t) + lat[i + 1] * t


def catmull_rom(p: np.ndarray, tension: float, steps: int) -> np.ndarray:
    """Spline melalui titik p (N x 2); tangen m_i = tension x (p[i+1] - p[i-1])."""
    if len(p) < 3:
        return p
    ext = np.vstack([p[0], p, p[-1]])
    m = tension * (ext[2:] - ext[:-2])
    t = np.linspace(0, 1, steps, endpoint=False)[:, None]
    h00, h10, h01, h11 = 2 * t**3 - 3 * t**2 + 1, t**3 - 2 * t**2 + t, -2 * t**3 + 3 * t**2, t**3 - t**2
    segs = [h00 * p[i] + h10 * m[i] + h01 * p[i + 1] + h11 * m[i + 1] for i in range(len(p) - 1)]
    return np.vstack(segs + [p[-1:]])


def resample(p: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    s = np.linspace(0, d[-1], n)
    return np.column_stack([np.interp(s, d, p[:, 0]), np.interp(s, d, p[:, 1])]), s


def stroke_geometry(poly: np.ndarray, a, rng: np.random.Generator, offset: float) -> tuple[np.ndarray, np.ndarray]:
    """Polyline -> (titik N x 2, tebal N): simplify -> Catmull-Rom -> resample -> tebal + taper -> jitter."""
    closed = len(poly) > 3 and np.allclose(poly[0], poly[-1])
    src = poly[:-1] if closed else poly
    simp = cv2.approxPolyDP(src.reshape(-1, 1, 2), a.simplify_epsilon, closed).reshape(-1, 2).astype(np.float32)
    if closed:
        simp = np.vstack([simp, simp[:1]])
    if len(simp) < 2:
        simp = poly[[0, -1]]
    curve = catmull_rom(simp, a.smooth_tension, a.spline_steps)
    length = float(np.sum(np.linalg.norm(np.diff(curve, axis=0), axis=1)))
    # resample_points titik per stroke, tapi maksimal 1 titik / px untuk stroke pendek
    pts, s = resample(curve, int(max(4, min(a.resample_points, length))))
    tang = np.gradient(pts, axis=0)
    tang /= np.maximum(np.linalg.norm(tang, axis=1, keepdims=True), 1e-6)
    normal = np.column_stack([-tang[:, 1], tang[:, 0]])
    width = a.width_base * (1 + a.width_variation * value_noise(s * a.width_noise_scale, rng))
    if a.taper_ends:
        t = np.clip(np.minimum(s, length - s) / a.taper_px, 0, 1)
        width *= a.taper_min + (1 - a.taper_min) * np.sqrt(t)
    jit = a.jitter_amplitude * value_noise(s * a.jitter_frequency, rng) + offset
    return pts + normal * jit[:, None], np.maximum(width, 0.3)


def frame_seed(frame_index: int, param_seed: int) -> int:
    """seed = hash(frame_index, param_seed) (P-007), stabil antar proses (crc32, bukan hash() Python)."""
    return zlib.crc32(f"{frame_index}:{param_seed}".encode())


def hex_bgr(s: str) -> np.ndarray:
    s = s.lstrip("#")
    return np.array([int(s[4:6], 16), int(s[2:4], 16), int(s[0:2], 16)], np.float32)


def draw_strokes(polys: list[np.ndarray], hw: tuple[int, int], frame_index: int, a) -> np.ndarray:
    h, w = hw
    ss = a.ss
    canvas = np.empty((h, w, 3), np.float32)
    canvas[:] = hex_bgr(a.paper_color)
    ink = hex_bgr(a.color)
    base = frame_seed(frame_index, a.param_seed)
    for k in range(a.passes):
        mask = np.zeros((h * ss, w * ss), np.uint8)
        for si, poly in enumerate(polys):
            rng = np.random.default_rng([base, si, k])
            pts, wid = stroke_geometry(poly, a, rng, k * a.pass_offset)
            P = pts * ss
            r = wid * ss / 2
            nrm = np.diff(P, axis=0)
            nrm = np.column_stack([-nrm[:, 1], nrm[:, 0]]) / np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-6)
            for j in range(len(P) - 1):
                quad = np.array([P[j] + nrm[j] * r[j], P[j + 1] + nrm[j] * r[j + 1],
                                 P[j + 1] - nrm[j] * r[j + 1], P[j] - nrm[j] * r[j]])
                cv2.fillConvexPoly(mask, np.round(quad).astype(np.int32), 255)
            for j in range(len(P)):  # sambungan + ujung bulat (cap: round)
                cv2.circle(mask, (int(round(P[j, 0])), int(round(P[j, 1]))), max(int(round(r[j])), 1), 255, -1)
        cov = cv2.resize(mask, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)[..., None] / 255.0
        alpha = a.opacity * (a.opacity_falloff ** k) * cov
        canvas = canvas * (1 - alpha) + ink * alpha
    return np.clip(canvas, 0, 255).astype(np.uint8)


def look_frame(clip: Clip, n: str, thr: dict) -> tuple[np.ndarray, dict]:
    a = clip.args
    g = ex.island_filter(clip.groups(n), a.min_contour_area)  # min_contour_area pada peta grup
    group_skel = cv2.ximgproc.thinning(probe.boundary(g).astype(np.uint8) * 255) > 0
    dep = depth_edges(clip, n, thr["lo"], thr["hi"])
    dep_skel = cv2.ximgproc.thinning(dep.astype(np.uint8) * 255) > 0
    polys = trace_polylines(group_skel, a.min_stroke_px) + trace_polylines(dep_skel, a.min_stroke_px)
    img = draw_strokes(polys, clip.hw, probe.frame_no(n), a)
    return img, {"n_strokes": len(polys), "depth_px": int(dep_skel.sum())}


# --- perintah --------------------------------------------------------------
def old_threshold(args) -> float:
    return json.loads((args.exp / f"filtered_full_{args.model.replace('seg_', '')}.json").read_text(
        encoding="utf-8"))["da_threshold"]


def lines_img(mask: np.ndarray) -> np.ndarray:
    img = np.full(mask.shape + (3,), probe.LINE_BG, np.uint8)
    img[mask] = probe.LINE_FG
    return img


def parse_frames(s: str) -> list[int]:
    return [int(x) for x in s.split(",")]


def cmd_diag(args, clip: Clip) -> None:
    thr = old_threshold(args)
    rows, rep = [], []
    for f in parse_frames(args.frames):
        n = clip.name(f)
        s, d, comb = current_lines(clip, n, thr)
        seg_only, da_only = comb & s, comb & d & ~s
        r = {"frame": f, "seg_px": int(seg_only.sum()), "da_only_px": int(da_only.sum()), "total_px": int(comb.sum())}
        rep.append(r)
        panels = []
        for m, lab in ((seg_only, f"garis seg saja ({r['seg_px']} px)"),
                       (da_only, f"garis DA saja ({r['da_only_px']} px)"), (comb, f"gabungan ({r['total_px']} px)")):
            p = lines_img(m)
            ab.draw_label(p, [lab, f"DA p{args.old_pct:g} thr {thr:.5f}", Path(n).stem])
            panels.append(p)
        rows.append(cv2.resize(np.hstack(panels), None, fx=args.grid_scale, fy=args.grid_scale,
                               interpolation=cv2.INTER_AREA))
    cv2.imwrite(str(args.out / "diag_sources.png"), np.vstack(rows))
    ex.dump(args.out / "diag_sources.json", {"da_threshold": thr, "frames": rep})
    for r in rep:
        print(f"  frame {r['frame']}: seg {r['seg_px']} px, DA saja {r['da_only_px']} px "
              f"({r['da_only_px'] / max(r['total_px'], 1):.1%} dari gabungan)")


def cmd_stats(args, clip: Clip) -> None:
    """Data untuk memilih T_low, D, L (dicetak; tidak mengubah default)."""
    t0 = time.perf_counter()
    thr = clip_thresholds(clip)
    print(f"|grad log d| per klip ({thr['n_values']} nilai): " +
          ", ".join(f"{k}={v:.5f}" for k, v in thr["pct_table"].items()))
    print(f"  T_high p{args.depth_hi_pct:g} = {thr['hi']:.5f}; rasio persentil / T_high: " +
          ", ".join(f"{k}={v / thr['hi']:.2f}" for k, v in thr["pct_table"].items()))
    step = max(1, len(clip.names) // args.stats_frames)
    sample = clip.names[::step]
    dist_hist = np.zeros(21, np.int64)
    lens_all = []
    for n in sample:
        e = depth_edges(clip, n, thr["lo"], thr["hi"], stage="nms")
        dist = group_boundary_dist(clip.groups(n))[e]
        dist_hist += np.bincount(np.minimum(dist.astype(int), 20), minlength=21)
        _, lens = skeleton_length_filter(depth_edges(clip, n, thr["lo"], thr["hi"], stage="group"), 1)
        lens_all += lens
    print(f"jarak piksel tepi NMS+hysteresis ke batas grup ({len(sample)} frame), px 0..19, >=20:")
    print("  " + str(dist_hist.tolist()))
    la = np.array(lens_all)
    print(f"panjang skeleton komponen (setelah D={args.depth_min_dist}): n={la.size}, "
          + ", ".join(f"<={t}: {int((la <= t).sum())}" for t in (5, 10, 15, 20, 30, 40, 60, 80, 120)))
    print("  terpanjang 20:", sorted(la.tolist())[-20:])
    for f in parse_frames(args.stats_leg_frames):
        n = clip.name(f)
        _, lens = skeleton_length_filter(depth_edges(clip, n, thr["lo"], thr["hi"], stage="group"), 1)
        print(f"  frame {f} (kaki menyilang): panjang komponen {sorted(lens, reverse=True)[:8]}")
    print(f"selesai {time.perf_counter() - t0:.1f} s")


def cmd_render(args, clip: Clip) -> None:
    t0 = time.perf_counter()
    thr = clip_thresholds(clip)
    old = old_threshold(args)
    print(f"T_high p{args.depth_hi_pct:g} = {thr['hi']:.5f}, T_low p{args.depth_lo_pct:g} = {thr['lo']:.5f}")
    params = (f"hi p{args.depth_hi_pct:g} lo p{args.depth_lo_pct:g} D{args.depth_min_dist} L{args.depth_min_len} "
              f"w{args.width_base} j{args.jitter_amplitude}")
    stats = {}

    def look(n: str) -> np.ndarray:
        img, st = look_frame(clip, n, thr)
        stats[n] = st
        ab.draw_label(img, ["look test", params, Path(n).stem])
        return img

    rows = []
    for f in parse_frames(args.frames):
        n = clip.name(f)
        p0 = clip.bgr(n)
        ab.draw_label(p0, ["asli", Path(n).stem])
        p1 = lines_img(current_lines(clip, n, old)[2])
        ab.draw_label(p1, ["preview filtered_full sekarang", Path(n).stem])
        rows.append(cv2.resize(np.hstack([p0, p1, look(n)]), None, fx=args.grid_scale, fy=args.grid_scale,
                               interpolation=cv2.INTER_AREA))
    cv2.imwrite(str(args.out / "look_frames.png"), np.vstack(rows))
    print(f"  look_frames.png ({time.perf_counter() - t0:.0f} s)")

    for win in args.windows.split(","):
        a0, b0 = (int(x) for x in win.split("-"))
        names = [clip.name(f) for f in range(a0, b0 + 1)]

        def render(n: str) -> np.ndarray:
            p0 = clip.bgr(n)
            ab.draw_label(p0, ["asli", Path(n).stem])
            p1 = look(n)
            ab.draw_label(p1, ["look test", params, LABEL_NOTE, Path(n).stem])
            return np.hstack([p0, p1])

        ex.write_video(args.out / f"look_f{a0:03d}-{b0:03d}.mp4", names, render, args.fps)
        print(f"  look_f{a0:03d}-{b0:03d}.mp4 ({time.perf_counter() - t0:.0f} s)")
    ex.dump(args.out / "look_params.json", {"thresholds": thr, "old_da_threshold": old,
                                            "args": {k: str(v) if isinstance(v, Path) else v
                                                     for k, v in vars(args).items()},
                                            "per_frame": stats})


def self_check() -> None:
    mag = np.array([[0, 1, 3, 1, 0]] * 3, np.float32)
    keep = nms(mag, np.ones_like(mag), np.zeros_like(mag))  # gradien horizontal -> puncak di kolom 2
    if not (keep[:, 2].all() and not keep[:, [1, 3]].any()):
        ab.fail("self-check gagal: nms")
    cand = np.zeros((3, 6), bool)
    cand[1, :] = True
    m2 = np.array([[0] * 6, [0.5, 0.5, 2, 0.5, 0, 0.5], [0] * 6], np.float32)
    hy = hysteresis(cand, m2, 0.4, 1.0)
    if not (hy[1, :4].all() and not hy[1, 4:].any()):
        ab.fail("self-check gagal: hysteresis")
    sk = np.zeros((10, 10), bool)
    sk[5, 1:9] = True
    polys = trace_polylines(sk, 3)
    if len(polys) != 1 or len(polys[0]) != 8:
        ab.fail("self-check gagal: trace_polylines")
    if frame_seed(3, 0) != frame_seed(3, 0) or frame_seed(3, 0) == frame_seed(4, 0):
        ab.fail("self-check gagal: frame_seed")
    print("self-check OK")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("cmd", choices=["diag", "stats", "render"])
    p.add_argument("--work", type=Path, default=ROOT / "work")
    p.add_argument("--t102c", type=Path, default=ROOT / "work" / "t102c")
    p.add_argument("--exp", type=Path, default=ROOT / "work" / "t102c" / "exp")
    p.add_argument("--out", type=Path, default=ROOT / "work" / "t102c" / "look")
    p.add_argument("--model", default="seg_0.8b", choices=list(ex.SEG_MODELS))
    p.add_argument("--frames", default="1,87,120,183,236", help="frame untuk diag_sources.png / look_frames.png")
    p.add_argument("--windows", default="73-92,183-202", help="jendela video look_fAAA-BBB.mp4")
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--grid-scale", type=float, default=0.5)
    # Peta grup + preview sekarang: nilai sama dengan filtered_full_0.8b (update log T-102c docs/05)
    p.add_argument("--island-min", type=int, default=30, help="(a) N pulau kelas")
    p.add_argument("--mode-k", type=int, default=3, help="(b) K mode filter peta grup")
    p.add_argument("--line-min", type=int, default=5, help="(c) M komponen garis preview sekarang")
    p.add_argument("--line-px", type=int, default=2, help="tebal garis preview piksel")
    p.add_argument("--lines-erode", type=int, default=5, help="kernel erosi foreground untuk garis kedalaman")
    p.add_argument("--lines-eps", type=float, default=1e-6, help="kedalaman minimum yang dianggap valid")
    p.add_argument("--old-pct", type=float, default=95.0, help="label saja: persentil threshold preview sekarang")
    # Garis kedalaman baru (alasan default: update log T-102c docs/05, 2026-09-29 look test)
    p.add_argument("--depth-blur", type=float, default=1.0,
                   help="sigma Gaussian sebelum gradien log (px); meredam tangga kuantisasi float16 DA")
    p.add_argument("--depth-hi-pct", type=float, default=95.0, help="T_high = persentil per klip (= p95 sekarang)")
    p.add_argument("--depth-lo-pct", type=float, default=90.0,
                   help="T_low = persentil per klip; p90 = 0.44 x T_high (rasio Canny 1:2-1:3)")
    p.add_argument("--depth-min-dist", type=int, default=7,
                   help="D: jarak minimum ke batas grup (px); tepi DA menumpuk di 0-6 px dari batas seg, turun tajam di 7")
    p.add_argument("--depth-min-len", type=int, default=30,
                   help="L: panjang skeleton minimum (px); garis kaki di frame uji 73-92 >= 38 px")
    p.add_argument("--stats-frames", type=int, default=40, help="stats: jumlah frame sampel")
    p.add_argument("--stats-leg-frames", default="73,82,92", help="stats: frame kaki menyilang")
    # Stroke: default = preset rough-sketch docs/02 (shape / stroke / jitter / multipass / paper.color)
    p.add_argument("--simplify-epsilon", type=float, default=2.5)
    p.add_argument("--resample-points", type=int, default=200, help="titik per stroke (maks 1 titik / px)")
    p.add_argument("--smooth-tension", type=float, default=0.5)
    p.add_argument("--min-contour-area", type=int, default=800,
                   help="komponen peta grup < nilai ini (px^2) digabung ke grup mayoritas sekeliling")
    p.add_argument("--width-base", type=float, default=3.2)
    p.add_argument("--width-variation", type=float, default=0.45)
    p.add_argument("--width-noise-scale", type=float, default=0.08)
    p.add_argument("--color", default="#1a1a1a")
    p.add_argument("--opacity", type=float, default=0.92)
    p.add_argument("--no-taper", dest="taper_ends", action="store_false", help="taper_ends: true di preset")
    p.add_argument("--jitter-amplitude", type=float, default=1.8)
    p.add_argument("--jitter-frequency", type=float, default=0.12)
    p.add_argument("--passes", type=int, default=2, help="multipass.passes (1 = multipass mati)")
    p.add_argument("--pass-offset", type=float, default=1.2)
    p.add_argument("--opacity-falloff", type=float, default=0.55)
    p.add_argument("--paper-color", default="#f4f1ea")
    # Parameter baru di luar preset
    p.add_argument("--taper-px", type=float, default=20.0,
                   help="panjang taper tiap ujung (px); ~6x width_base supaya penipisan terlihat")
    p.add_argument("--taper-min", type=float, default=0.15, help="tebal di ujung sebagai fraksi tebal normal")
    p.add_argument("--spline-steps", type=int, default=8, help="titik spline per segmen sebelum resample")
    p.add_argument("--min-stroke-px", type=int, default=6,
                   help="jalur skeleton < nilai ini tidak digambar (cabang pendek sisa thinning di junction)")
    p.add_argument("--param-seed", type=int, default=0, help="param_seed P-007")
    p.add_argument("--ss", type=int, default=3, help="faktor supersampling untuk anti-alias")
    args = p.parse_args()

    self_check()
    args.out.mkdir(parents=True, exist_ok=True)
    clip = Clip(args)
    {"diag": cmd_diag, "stats": cmd_stats, "render": cmd_render}[args.cmd](args, clip)
    return 0


if __name__ == "__main__":
    sys.exit(main())
