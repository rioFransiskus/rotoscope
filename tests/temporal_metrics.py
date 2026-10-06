"""Metrik kualitas temporal [3] (T-302), dipakai test_stabilize_temporal.py dan scripts/stable_preview.py.

Semua fungsi murni numpy atas peta grup (T, H, W) uint8 (0 = background). Metrik yang bisa "dimenangkan" dengan merusak
(flip-flop turun karena gerak sah ikut diratakan) SELALU dipasangkan dengan metrik kesetiaan terhadap argmax mentah:

- flip-flop: piksel yang berubah grup di frame t lalu kembali ke grup frame t−1 pada t+1 atau t+2 (A→B→A ≤ 2 frame);
- kesetiaan: IoU grup / foreground vs argmax mentah, centroid error (px), rasio luas lengan, lag luas (korelasi silang);
- jendela otomatis: frame "statis" (derau murni) dan "gerak cepat" dipilih dari data, bukan nomor frame tetap.

T-303 (optical flow DITOLAK, docs/04) menambah alat ukur yang dipertahankan: pop energy per tipe strok dari contours/ (`pop_energy_by_type`,
`track_ages`), galat alignment sebuah warp (`alignment_error`) dan konsistensi maju-mundur (`fb_error`, `inconsistent_fraction`) atas flow yang
diberikan pemanggil — tidak ada kode flow produksi.
"""

from __future__ import annotations

import numpy as np

# Aturan jendela (T-302), dihitung dari argmax MENTAH (bukan hasil temporal): jendela = ≥ RUN_MIN_FRAMES frame berurutan yang SETIAP frame-nya
# memenuhi kecepatan centroid foreground (px / frame) DAN XOR foreground t vs t−1 (dibagi luas foreground). Batas dipilih dari distribusi gerak
# kedua klip uji (persentil kecepatan / XOR: test_short p25 1,06 / 0,037, p50 1,93 / 0,055, p75 3,21 / 0,088; test p25 1,83 / 0,059, p50 3,63 /
# 0,100, p75 5,96 / 0,131), BUKAN dari hasil flip-flop. Statis ≈ di bawah median kedua klip; cepat ≈ di atas p50–p75.
RUN_MIN_FRAMES = 10               # jendela = ≥ 10 frame berurutan
STATIC_SPEED_MAX_PX = 3.0         # kecepatan centroid foreground (px / frame) ≤ ini
STATIC_XOR_MAX = 0.06             # XOR foreground t vs t−1, dibagi luas foreground ≤ ini
FAST_SPEED_MIN_PX = 4.0
FAST_XOR_MIN = 0.08
GROUP_MIN_PX = 200                # grup lebih kecil dari ini di argmax mentah tidak ikut IoU grup
LAG_MAX_FRAMES = 3
ARM_GROUPS = (4, 5)               # left_arm, right_arm (urutan groups: default)


def fg_area(g: np.ndarray) -> np.ndarray:
    return (g != 0).sum(axis=(1, 2))


def flipflop_counts(g: np.ndarray) -> np.ndarray:
    """(T,) piksel flip-flop yang dimulai di frame t (berubah di t, kembali di t+1 atau t+2). Frame tepi = 0."""
    n = len(g)
    out = np.zeros(n, np.int64)
    for t in range(1, n - 1):
        ch = g[t] != g[t - 1]
        back1 = ch & (g[t + 1] == g[t - 1])
        total = int(back1.sum())
        if t + 2 < n:
            total += int((ch & ~back1 & (g[t + 1] != g[t - 1]) & (g[t + 2] == g[t - 1])).sum())
        out[t] = total
    return out


def flipflop_per10k(g: np.ndarray, frames: np.ndarray | None = None) -> float:
    """Rata-rata flip-flop per 10k piksel foreground pada frame terpilih (default: semua frame bertetangga)."""
    ff, area = flipflop_counts(g), np.maximum(fg_area(g), 1)
    sel = np.zeros(len(g), bool)
    sel[1:-1] = True
    if frames is not None:
        sel &= frames
    return float(1e4 * np.mean(ff[sel] / area[sel])) if sel.any() else float("nan")


def iou_prev(g: np.ndarray) -> np.ndarray:
    """IoU foreground t vs t−1 (laporan saja; didominasi gerak)."""
    f = g != 0
    return np.array([(f[t] & f[t - 1]).sum() / max((f[t] | f[t - 1]).sum(), 1) for t in range(1, len(g))])


def label_agreement_prev(g: np.ndarray) -> np.ndarray:
    """Fraksi piksel foreground-di-kedua-frame yang grupnya sama (laporan saja; didominasi gerak)."""
    out = []
    for t in range(1, len(g)):
        both = (g[t] != 0) & (g[t - 1] != 0)
        out.append(float((g[t][both] == g[t - 1][both]).mean()) if both.any() else 1.0)
    return np.array(out)


def centroids(g: np.ndarray) -> np.ndarray:
    """(T, 2) centroid (x, y) foreground, pusat piksel; foreground kosong → NaN."""
    h, w = g.shape[1:]
    ys, xs = np.mgrid[0:h, 0:w]
    return np.array([[xs[m].mean(), ys[m].mean()] if m.any() else [np.nan, np.nan] for m in (g != 0)])


def centroid_speed(g: np.ndarray) -> np.ndarray:
    c = centroids(g)
    return np.r_[0.0, np.hypot(*np.diff(c, axis=0).T)]


def xor_fraction(g: np.ndarray) -> np.ndarray:
    f = g != 0
    area = np.maximum(f.sum(axis=(1, 2)), 1)
    return np.r_[0.0, [(f[t] ^ f[t - 1]).sum() / area[t] for t in range(1, len(g))]]


def runs_of(mask: np.ndarray, min_len: int = RUN_MIN_FRAMES) -> list[tuple[int, int]]:
    """Run True berurutan sepanjang ≥ min_len → [(awal, akhir inklusif)]."""
    out, start = [], None
    for i, v in enumerate(list(mask) + [False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_len:
                out.append((start, i - 1))
            start = None
    return out


def window_mask(n: int, runs: list[tuple[int, int]]) -> np.ndarray:
    m = np.zeros(n, bool)
    for lo, hi in runs:
        m[lo:hi + 1] = True
    return m


def static_windows(g: np.ndarray, speed_max: float = STATIC_SPEED_MAX_PX, xor_max: float = STATIC_XOR_MAX,
                   min_len: int = RUN_MIN_FRAMES) -> list[tuple[int, int]]:
    """Run ≥ min_len frame dengan kecepatan centroid ≤ speed_max DAN XOR foreground ≤ xor_max (frame 0 dilewati)."""
    ok = (centroid_speed(g) <= speed_max) & (xor_fraction(g) <= xor_max)
    ok[0] = False
    return runs_of(ok, min_len)


def fast_windows(g: np.ndarray, speed_min: float = FAST_SPEED_MIN_PX, xor_min: float = FAST_XOR_MIN,
                 min_len: int = RUN_MIN_FRAMES) -> list[tuple[int, int]]:
    """Run ≥ min_len frame dengan kecepatan centroid ≥ speed_min DAN XOR foreground ≥ xor_min (aturan sama dengan statis)."""
    ok = (centroid_speed(g) >= speed_min) & (xor_fraction(g) >= xor_min)
    ok[0] = False
    return runs_of(ok, min_len)


def group_iou(g: np.ndarray, raw: np.ndarray, groups: range | tuple[int, ...] | None = None,
              min_px: int = GROUP_MIN_PX) -> np.ndarray:
    """IoU per grup per frame antara hasil dan argmax mentah, hanya grup dengan ≥ min_px di mentah. Return 1-D."""
    ids = groups if groups is not None else [i for i in np.unique(raw) if i != 0]
    out = []
    for t in range(len(g)):
        for gid in ids:
            b = raw[t] == gid
            if b.sum() >= min_px:
                a = g[t] == gid
                out.append((a & b).sum() / (a | b).sum())
    return np.array(out)


def fg_iou(g: np.ndarray, raw: np.ndarray) -> np.ndarray:
    a, b = g != 0, raw != 0
    return np.array([(a[t] & b[t]).sum() / max((a[t] | b[t]).sum(), 1) for t in range(len(g))])


def centroid_error(g: np.ndarray, raw: np.ndarray) -> np.ndarray:
    """Jarak (px) antara centroid foreground hasil dan mentah, per frame."""
    return np.hypot(*(centroids(g) - centroids(raw)).T)


def area_lag(g: np.ndarray, raw: np.ndarray, max_lag: int = LAG_MAX_FRAMES) -> float:
    """Lag (frame, subframe lewat parabola) dari korelasi silang luas foreground hasil vs mentah; 0 = tanpa lag.
    Positif = hasil tertinggal dari mentah."""
    a = fg_area(g).astype(float)
    b = fg_area(raw).astype(float)
    a, b = a - a.mean(), b - b.mean()
    n = len(a)
    if n <= 2 * max_lag + 1 or not a.any() or not b.any():
        return 0.0
    corr = {}
    for lag in range(-max_lag, max_lag + 1):
        x = a[max(0, lag):n + min(0, lag)]
        y = b[max(0, -lag):n - max(0, lag)]
        denom = np.sqrt((x * x).sum() * (y * y).sum())
        corr[lag] = float((x * y).sum() / denom) if denom else 0.0
    best = max(corr, key=corr.get)
    if abs(best) == max_lag:
        return float(best)
    y0, y1, y2 = corr[best - 1], corr[best], corr[best + 1]
    den = y0 - 2 * y1 + y2
    return float(best + (0.5 * (y0 - y2) / den if den else 0.0))


def limb_ratio(g: np.ndarray, raw: np.ndarray, groups: tuple[int, ...] = ARM_GROUPS) -> np.ndarray:
    """Luas grup lengan hasil / mentah per frame (frame tanpa lengan di mentah → 1)."""
    out = []
    for t in range(len(g)):
        b = np.isin(raw[t], groups).sum()
        out.append(np.isin(g[t], groups).sum() / b if b else 1.0)
    return np.array(out)


# ── T-303: pop garis akhir per tipe strok + galat alignment / konsistensi flow (alat ukur dipertahankan; flow produksi TIDAK ada) ──
# "Pop" = strok yang lahir / mati di frame t. Strok = (tipe, track_id) di contours/frame_*.json (T-202). Pop energy per tipe =
# panjang (px) strok tipe itu yang lahir ATAU mati di frame t, dibagi total panjang SEMUA strok frame t; dirata-ratakan atas frame 1..T−1.
STROKE_TYPES = ("silhouette", "silhouette_hole", "group_boundary", "occlusion")
ALIGN_DILATE_PX = 7               # piksel bergerak = XOR argmax t vs t+k, didilatasi kotak 7×7
FB_THRESHOLDS_PX = (0.5, 1.0, 2.0)  # ambang inkonsistensi maju-mundur (px)


def stroke_length(points, closed: bool) -> float:
    """Panjang polyline (px); closed → termasuk segmen penutup."""
    p = np.asarray(points, float)
    if len(p) < 2:
        return 0.0
    total = float(np.hypot(*np.diff(p, axis=0).T).sum())
    if closed:
        total += float(np.hypot(*(p[0] - p[-1])))
    return total


def stroke_lengths_from_dir(contours_dir) -> list[dict[tuple[str, int], float]]:
    """contours/frame_*.json terurut → per frame {(tipe, track_id): panjang px}."""
    import json
    from pathlib import Path

    out = []
    for f in sorted(Path(contours_dir).glob("frame_*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        out.append({(s["type"], s["track_id"]): stroke_length(s["points"], s["closed"]) for s in d["strokes"]})
    return out


def pop_energy_by_type(frames: list[dict], types: tuple[str, ...] = STROKE_TYPES) -> dict:
    """frames = keluaran stroke_lengths_from_dir. Per tipe: pop_energy_mean, pop_sum, born / died per frame, length_share_mean,
    id_new_per_stroke_frame, pop_per_length_share, pop_share_pct (porsi terhadap total semua tipe), age_mean / age_median, frames_without.
    `total_pop_energy_mean` = jumlah pop energy semua tipe (rata-rata atas frame 1..T−1)."""
    n = len(frames)
    per, total = {}, 0.0
    for ty in types:
        pop, born_n, died_n, share, n_str = [], [], [], [], []
        for t in range(1, n):
            cur = {k: v for k, v in frames[t].items() if k[0] == ty}
            prv = {k: v for k, v in frames[t - 1].items() if k[0] == ty}
            all_len = sum(frames[t].values()) or 1.0
            born = [v for k, v in cur.items() if k not in prv]
            died = [v for k, v in prv.items() if k not in cur]
            pop.append((sum(born) + sum(died)) / all_len)
            born_n.append(len(born)), died_n.append(len(died))
            share.append(sum(cur.values()) / all_len)
            n_str.append(len(cur))
        ages = track_ages(frames, ty)
        per[ty] = {"pop_energy_mean": float(np.mean(pop)) if pop else 0.0, "pop_sum": float(np.sum(pop)),
                   "born_per_frame": float(np.mean(born_n)) if pop else 0.0, "died_per_frame": float(np.mean(died_n)) if pop else 0.0,
                   "length_share_mean": float(np.mean(share)) if pop else 0.0,
                   "id_new_per_stroke_frame": float(np.sum(born_n) / max(np.sum(n_str), 1)) if pop else 0.0,
                   "age_mean": float(np.mean(ages)) if ages else None, "age_median": float(np.median(ages)) if ages else None,
                   "frames_without": frames_without_type(frames, ty)}
        total += per[ty]["pop_sum"]
    for ty in types:
        p = per[ty]
        p["pop_share_pct"] = 100 * p["pop_sum"] / total if total else None
        p["pop_per_length_share"] = p["pop_energy_mean"] / p["length_share_mean"] if p["length_share_mean"] else None
    return {"frames": n, "total_pop_energy_mean": total / max(n - 1, 1), "types": per}


def track_ages(frames: list[dict], ty: str) -> list[int]:
    """Panjang (frame) tiap run berurutan sebuah (tipe, track_id); track yang putus lalu muncul lagi = dua run."""
    ages = []
    for key in {k for f in frames for k in f if k[0] == ty}:
        run = 0
        for f in frames:
            if key in f:
                run += 1
            elif run:
                ages.append(run)
                run = 0
        if run:
            ages.append(run)
    return ages


def frames_without_type(frames: list[dict], ty: str) -> int:
    return sum(1 for f in frames if not any(k[0] == ty for k in f))


def fb_error(f_fwd: np.ndarray, f_bwd: np.ndarray) -> np.ndarray:
    """Galat konsistensi maju-mundur per piksel (px): |f(x) + b(x + f(x))|. f_fwd, f_bwd (H, W, 2) float32 (dx, dy); b dibaca
    bilinear di x + f(x) (tepi gambar direplikasi)."""
    import cv2

    h, w = f_fwd.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    mx, my = xx + f_fwd[..., 0], yy + f_fwd[..., 1]
    bx = cv2.remap(f_bwd[..., 0], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    by = cv2.remap(f_bwd[..., 1], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return np.hypot(f_fwd[..., 0] + bx, f_fwd[..., 1] + by)


def inconsistent_fraction(err: np.ndarray, mask: np.ndarray | None = None,
                          thresholds: tuple[float, ...] = FB_THRESHOLDS_PX) -> dict[float, float]:
    """Fraksi piksel (dalam mask; default semua) dengan galat maju-mundur > ambang."""
    sel = np.ones(err.shape, bool) if mask is None else mask
    if not sel.any():
        return {th: 0.0 for th in thresholds}
    return {th: float((err[sel] > th).mean()) for th in thresholds}


def moving_mask(raw_t: np.ndarray, raw_tk: np.ndarray, dilate_px: int = ALIGN_DILATE_PX) -> np.ndarray:
    """Piksel bergerak antara dua argmax: XOR (label berbeda) didilatasi kotak dilate_px × dilate_px."""
    import cv2

    return cv2.dilate((raw_t != raw_tk).astype(np.uint8), np.ones((dilate_px, dilate_px), np.uint8)).astype(bool)


def alignment_error(raw_t: np.ndarray, warped: np.ndarray, raw_tk: np.ndarray, valid: np.ndarray | None = None,
                    dilate_px: int = ALIGN_DILATE_PX) -> dict:
    """Galat alignment sebuah warp: pada piksel bergerak (moving_mask ∩ valid), fraksi label yang BEDA dari argmax frame t —
    `err_warp` = label hasil warp (argmax tetangga yang di-warp ke t) vs raw_t, `err_nowarp` = label tetangga mentah vs raw_t
    (pembanding tanpa warp). Warp berguna ⇔ err_warp < err_nowarp. `warped` = peta label (H, W) hasil warp."""
    m = moving_mask(raw_t, raw_tk, dilate_px)
    if valid is not None:
        m &= valid
    n = int(m.sum())
    if n == 0:
        return {"moving_px": 0, "err_warp": 0.0, "err_nowarp": 0.0}
    return {"moving_px": n, "err_warp": 1.0 - float((warped[m] == raw_t[m]).sum()) / n,
            "err_nowarp": 1.0 - float((raw_tk[m] == raw_t[m]).sum()) / n}


def recovery_iou(g: np.ndarray, ref: np.ndarray, frames: list[int]) -> np.ndarray:
    """IoU grup rata-rata (semua grup ≠ 0 yang ada di ref) hasil vs REF (frame asli sebelum disuntik) di frame terpilih."""
    out = []
    for t in frames:
        ious = []
        for gid in (int(i) for i in np.unique(ref[t]) if i != 0):
            a, b = g[t] == gid, ref[t] == gid
            ious.append((a & b).sum() / (a | b).sum())
        out.append(float(np.mean(ious)) if ious else 1.0)
    return np.array(out)
