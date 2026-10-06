"""Test noise.py (T-401): value noise 2D dari hash bilangan bulat — rentang [-1, 1], nol-rata-rata, determinisme (nilai emas),
seed, panjang korelasi, tanpa dependensi baru."""

from __future__ import annotations

import subprocess
import sys

import numpy as np

from rotoscope import noise

GOLDEN_LATTICE = [0.09200957022353329, -0.901923659960195, 0.10248061539381315, 0.6993286043370921]   # seed_of(0)
GOLDEN_NOISE = 0.2941145434226272                                                                     # (3,25; −1,5)


def grid(n=300, span=60.0):
    x, y = np.meshgrid(np.linspace(0, span, n), np.linspace(0, span, n))
    return x.ravel(), y.ravel()


def test_range_is_normalized_and_zero_mean():
    x, y = grid()
    v = noise.value_noise_2d(noise.seed_of(0), x, y)
    assert float(v.min()) >= -1.0 and float(v.max()) <= 1.0
    assert float(v.max()) > 0.8 and float(v.min()) < -0.8                 # rentang benar-benar dipakai
    assert abs(float(v.mean())) < 0.03
    assert 0.25 < float(v.std()) < 0.6


def test_lattice_values_uniform_in_range():
    ix, iy = np.meshgrid(np.arange(-200, 200), np.arange(-200, 200))
    v = noise.lattice(noise.seed_of(3), ix.ravel(), iy.ravel())
    assert float(v.min()) >= -1.0 and float(v.max()) < 1.0 and abs(float(v.mean())) < 0.02
    assert abs(float(v.std()) - 1 / np.sqrt(3)) < 0.02                      # seragam [-1, 1): std 0,577


def test_deterministic_and_golden_values():
    x, y = grid(40, 9.0)
    a = noise.value_noise_2d(noise.seed_of(0), x, y)
    b = noise.value_noise_2d(noise.seed_of(0), x.copy(), y.copy())
    assert np.array_equal(a, b)
    assert int(noise.seed_of(0)) == int(noise.seed_of(0)) and int(noise.seed_of(0)) != int(noise.seed_of(1))
    # nilai emas lintas run / proses: lattice di titik bilangan bulat tetap
    g = noise.lattice(noise.seed_of(0), np.array([0, 1, -1, 17]), np.array([0, 5, -3, 17]))
    r = subprocess.run([sys.executable, "-c",
                        "import numpy as np; from rotoscope import noise; "
                        "g = noise.lattice(noise.seed_of(0), np.array([0, 1, -1, 17]), np.array([0, 5, -3, 17])); "
                        "print(repr(g.tolist()))"], capture_output=True, text=True)
    assert r.returncode == 0 and eval(r.stdout) == g.tolist()                 # proses lain: identik
    assert g.tolist() == GOLDEN_LATTICE                                          # lintas versi: berubah = perilaku noise berubah (contract)
    assert float(noise.value_noise_2d(noise.seed_of(0), np.array([3.25]), np.array([-1.5]))[0]) == GOLDEN_NOISE


def test_seed_changes_field_and_negative_cells_work():
    x, y = grid(50, 20.0)
    a = noise.value_noise_2d(noise.seed_of(0), x - 25, y - 25)                   # sel negatif
    b = noise.value_noise_2d(noise.seed_of(1), x - 25, y - 25)
    assert float(np.abs(a - b).mean()) > 0.2 and float(np.abs(a).max()) <= 1.0
    assert noise.seed_of(0, 5) != noise.seed_of(5, 0)                            # urutan berpengaruh


def test_correlation_length_is_one_cell():
    """Skala = panjang korelasi: korelasi pada jarak 0,1 sel tinggi, pada jarak ≥ 2 sel rendah."""
    x = np.random.default_rng(0).uniform(0, 200, 20000)
    y = np.random.default_rng(1).uniform(0, 200, 20000)
    s = noise.seed_of(0)
    base = noise.value_noise_2d(s, x, y)

    def corr(dx):
        return float(np.corrcoef(base, noise.value_noise_2d(s, x + dx, y))[0, 1])
    assert corr(0.1) > 0.95 and abs(corr(2.0)) < 0.1 and abs(corr(5.0)) < 0.1
    # skala berubah (koordinat / panjang gelombang): panjang korelasi ikut
    assert corr(0.5) < corr(0.1)


def test_no_new_dependency_and_no_torch():
    code = "import sys; import rotoscope.noise; assert 'torch' not in sys.modules"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    src = open(noise.__file__, encoding="utf-8").read()
    assert "import random" not in src and "np.random" not in src                 # tanpa RNG berstatus
