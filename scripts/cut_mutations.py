"""Mutasi test kalibrasi cut_diff (alat sekali pakai; tidak mengubah kode produksi):

    python scripts/cut_mutations.py

Memanggil pernyataan kalibrasi `tests/test_cut_diff.py::check_real_clips` dengan nilai ambang lain dan melaporkan LULUS / GAGAL.
Harapan (asersi: 7 positif terdeteksi, terdeteksi - positif <= {f49}, 0 cut di klip lain): 0,06 / 0,065 / 0,07 lulus; 0,075 gagal (f85 hilang);
0,08 gagal (3 positif hilang); 0,05 gagal (banyak FP di klip3)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "src"))

import test_cut_diff as t  # noqa: E402

if __name__ == "__main__":
    import os
    os.chdir(ROOT)
    for thr in (0.05, 0.06, 0.065, 0.07, 0.075, 0.08):
        try:
            t.check_real_clips(thr)
            verdict = "LULUS"
        except AssertionError as e:
            got = sorted(t.detected("klip3", thr))
            verdict = f"GAGAL (klip3 terdeteksi {len(got)}: {got[:12]}{'...' if len(got) > 12 else ''}; positif hilang {sorted(t.POSITIVES - set(got))})"
        print(f"cut_diff {thr}: {verdict}")
