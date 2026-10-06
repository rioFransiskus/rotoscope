"""Noise deterministik tanpa dependensi (T-401): value noise 2D dari hash bilangan bulat 64-bit, numpy murni.

Hasil identik antar run / proses / mesin (hanya aritmetika uint64 + float64). Keluaran DINORMALISASI [-1, 1] oleh konstruksi:
nilai lattice seragam di [-1, 1) lalu kombinasi konveks (fade kuintik + interpolasi bilinear), jadi tidak bisa keluar rentang.
Nol-rata-rata (lattice simetris). Panjang korelasi ≈ satu sel lattice."""

from __future__ import annotations

import numpy as np

U64 = np.uint64
MASK64 = (1 << 64) - 1
MIX_C1 = 0xBF58476D1CE4E5B9           # konstanta finalizer splitmix64
MIX_C2 = 0x94D049BB133111EB
GOLDEN = 0x9E3779B97F4A7C15
SEED_SALT = 0x2545F4914F6CDD1D
SHIFT_A, SHIFT_B, SHIFT_C = 30, 27, 31
MANTISSA_SHIFT = 11                   # 64 − 53 bit
MANTISSA_SCALE = float(1 << 53)
FADE_A, FADE_B, FADE_C = 6.0, 15.0, 10.0   # 6t⁵ − 15t⁴ + 10t³


def _mix_int(x: int) -> int:
    x &= MASK64
    x ^= x >> SHIFT_A
    x = (x * MIX_C1) & MASK64
    x ^= x >> SHIFT_B
    x = (x * MIX_C2) & MASK64
    return x ^ (x >> SHIFT_C)


def seed_of(*ints: int) -> np.uint64:
    """Seed 64-bit dari bilangan bulat (urutan berpengaruh). T-401: seed_of(jitter.param_seed) — tanpa frame_index."""
    h = SEED_SALT
    for v in ints:
        h = _mix_int(h ^ ((int(v) & MASK64) * GOLDEN & MASK64))
    return U64(h)


def _mix(x: np.ndarray) -> np.ndarray:
    x = x ^ (x >> U64(SHIFT_A))
    x = x * U64(MIX_C1)
    x = x ^ (x >> U64(SHIFT_B))
    x = x * U64(MIX_C2)
    return x ^ (x >> U64(SHIFT_C))


def lattice(seed: np.uint64, ix: np.ndarray, iy: np.ndarray) -> np.ndarray:
    """Nilai acak seragam [-1, 1) pada titik lattice bilangan bulat (ix, iy)."""
    with np.errstate(over="ignore"):
        a = np.asarray(ix, dtype=np.int64).astype(U64) * U64(GOLDEN)
        b = np.asarray(iy, dtype=np.int64).astype(U64) * U64(MIX_C1)
        h = _mix(_mix(a + seed) ^ b)
    return (h >> U64(MANTISSA_SHIFT)).astype(np.float64) / MANTISSA_SCALE * 2.0 - 1.0


def fade(t: np.ndarray) -> np.ndarray:
    return t * t * t * (t * (t * FADE_A - FADE_B) + FADE_C)


def value_noise_2d(seed: np.uint64, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Value noise 2D di [-1, 1]; (x, y) dalam satuan sel lattice (koordinat / panjang gelombang)."""
    x0, y0 = np.floor(x), np.floor(y)
    tx, ty = fade(x - x0), fade(y - y0)
    ix, iy = x0.astype(np.int64), y0.astype(np.int64)
    n00, n10 = lattice(seed, ix, iy), lattice(seed, ix + 1, iy)
    n01, n11 = lattice(seed, ix, iy + 1), lattice(seed, ix + 1, iy + 1)
    return (n00 * (1 - tx) + n10 * tx) * (1 - ty) + (n01 * (1 - tx) + n11 * tx) * ty
