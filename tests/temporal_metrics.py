"""Metrik kualitas temporal [3] (T-302), dipakai test_stabilize_temporal.py dan scripts/stable_preview.py.

Semua fungsi murni numpy atas peta grup (T, H, W) uint8 (0 = background). Metrik yang bisa "dimenangkan" dengan merusak
(flip-flop turun karena gerak sah ikut diratakan) SELALU dipasangkan dengan metrik kesetiaan terhadap argmax mentah:

- flip-flop: piksel yang berubah grup di frame t lalu kembali ke grup frame t−1 pada t+1 atau t+2 (A→B→A ≤ 2 frame);
- kesetiaan: IoU grup / foreground vs argmax mentah, centroid error (px), rasio luas lengan, lag luas (korelasi silang);
- jendela otomatis: frame "statis" (derau murni) dan "gerak cepat" dipilih dari data, bukan nomor frame tetap.
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
