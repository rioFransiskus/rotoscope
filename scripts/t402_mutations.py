"""T-402 mutation check: monkeypatch DARI LUAR berkas test (kode produksi tidak diubah di disk). Tiap mutasi harus membuat test T-402
GAGAL. Pakai: python scripts/t402_mutations.py  → work/t402/mutations.json."""

from __future__ import annotations

import dataclasses
import json
import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402

from rotoscope import noise  # noqa: E402
from rotoscope import stylize as sty  # noqa: E402

TESTS = ["tests/test_stylize_jitter.py", "tests/test_stylize.py", "tests/test_stylize_width.py"]
ORIG = {"jitter_displacement": sty.jitter_displacement, "jitter_pieces": sty.jitter_pieces, "make_geometry": sty.make_geometry,
        "jitter_image_index": sty.jitter_image_index, "frame_pieces": sty.frame_pieces, "edge_fade": sty.edge_fade,
        "value_noise_3d": noise.value_noise_3d, "fold_warning": sty.fold_warning, "width_profile": sty.width_profile}


def mutate_independent_replaces_field():
    sty.jitter_displacement = lambda pts, ends, tids, g, fi: ORIG["jitter_displacement"](
        pts, ends, tids, dataclasses.replace(g, jitter_s=1.0), fi)


def mutate_displacement_along_normal():
    def jp(pieces, g, fi):
        if not g.jitter_on:
            return pieces
        out = []
        for pc in ORIG["jitter_pieces"]([dataclasses.replace(p) for p in pieces], g, fi):
            out.append(pc)
        res = []
        for base, pc in zip(pieces, out):
            d = pc.points - base.points
            p = base.points
            t = np.roll(p, -1, axis=0) - np.roll(p, 1, axis=0) if base.closed else np.gradient(p, axis=0)
            t = t / np.maximum(np.hypot(*t.T), 1e-12)[:, None]
            n = np.column_stack([-t[:, 1], t[:, 0]])
            res.append(dataclasses.replace(base, points=np.round(p + (d * n).sum(1)[:, None] * n, 2) + 0.0))
        return res
    sty.jitter_pieces = jp


def mutate_relative_window_frame_index():
    """Indeks gambar relatif terhadap frame pertama JENDELA run (bukan mutlak)."""
    first: dict = {}
    run = sty.run_stylize

    def fresh_run(*a, **k):
        first.clear()
        return run(*a, **k)

    def fp(doc, g):
        first.setdefault("v", int(doc["frame_index"]))
        return ORIG["frame_pieces"]({**doc, "frame_index": int(doc["frame_index"]) - first["v"]}, g)
    sty.run_stylize, sty.frame_pieces = fresh_run, fp


def mutate_hold_ignored():
    sty.jitter_image_index = lambda frame_index, g: 0 if g.jitter_fixed else int(frame_index)


def mutate_frame_index_leaks_into_seed():
    def jd(pts, ends, tids, g, fi):
        seeds = tuple(noise.seed_of(int(fi), int(s)) for s in g.jitter_seeds)
        return ORIG["jitter_displacement"](pts, ends, tids, dataclasses.replace(g, jitter_seeds=seeds), fi)
    sty.jitter_displacement = jd


def mutate_jitter_stream_equals_width_stream():
    def mg(style, width, height):
        g = ORIG["make_geometry"](style, width, height)
        return dataclasses.replace(g, jitter_seeds=(noise.seed_of(style.jitter.param_seed),) * 2)
    sty.make_geometry = mg


def mutate_amplitude_without_unit():
    def mg(style, width, height):
        g = ORIG["make_geometry"](style, width, height)
        return dataclasses.replace(g, jitter_amp=style.jitter.amplitude, jitter_cell=1.0 / style.jitter.frequency)
    sty.make_geometry = mg


def mutate_mix_does_not_preserve_variance():
    def jd(pts, ends, tids, g, fi):
        s = g.jitter_s
        f = ORIG["jitter_displacement"](pts, ends, tids, dataclasses.replace(g, jitter_s=0.0), fi)
        gg = ORIG["jitter_displacement"](pts, ends, tids, dataclasses.replace(g, jitter_s=1.0), fi)
        return (1 - s) * f + s * gg
    sty.jitter_displacement = jd


def mutate_edge_fade_off():
    sty.edge_fade = lambda p, g: np.ones(len(p))


def mutate_edge_dead_zone_zero():
    def mg(style, width, height):
        return dataclasses.replace(ORIG["make_geometry"](style, width, height), edge_dead=0.0)
    sty.make_geometry = mg


def mutate_amplitude_zero_not_identical():
    def mg(style, width, height):
        g = ORIG["make_geometry"](style, width, height)
        return dataclasses.replace(g, jitter_amp=max(g.jitter_amp, 0.5), jitter_cell=g.jitter_cell or 20.0)
    sty.make_geometry = mg


def mutate_time_blend_trilinear():
    def vn(seed, x, y, t):
        x0, y0 = np.floor(x), np.floor(y)
        tx, ty = noise.fade(x - x0), noise.fade(y - y0)
        ix, iy = x0.astype(np.int64), y0.astype(np.int64)
        z0 = int(np.floor(t))
        f = float(noise.fade(np.float64(t - z0)))
        a, b = noise._slab_2d(seed, ix, iy, tx, ty, z0), noise._slab_2d(seed, ix, iy, tx, ty, z0 + 1)
        return a * (1 - f) + b * f
    noise.value_noise_3d = vn


def mutate_fold_warning_off():
    sty.fold_warning = lambda style: None


def mutate_displacement_not_clamped():
    noise.value_noise_3d = lambda seed, x, y, t: 3.0 * ORIG["value_noise_3d"](seed, x, y, t)


MUTATIONS = {"jitter independen per strok menggantikan medan": mutate_independent_replaces_field,
             "perpindahan sepanjang normal": mutate_displacement_along_normal,
             "indeks gambar memakai frame_index relatif jendela": mutate_relative_window_frame_index,
             "hold diabaikan": mutate_hold_ignored, "frame_index bocor ke seed": mutate_frame_index_leaks_into_seed,
             "stream jitter = stream tebal": mutate_jitter_stream_equals_width_stream,
             "amplitudo tanpa unit": mutate_amplitude_without_unit,
             "bobot campuran tidak menjaga varians": mutate_mix_does_not_preserve_variance,
             "pelunakan tepi mati": mutate_edge_fade_off, "zona mati tepi nol": mutate_edge_dead_zone_zero,
             "amplitudo 0 tidak identik T-401": mutate_amplitude_zero_not_identical,
             "sumbu waktu trilinear (RMS bernapas)": mutate_time_blend_trilinear, "penjaga lipatan mati": mutate_fold_warning_off,
             "noise tidak dibatasi": mutate_displacement_not_clamped}


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else None
    if which is None:                       # induk: satu proses per mutasi (modul bersih)
        import subprocess
        out = {}
        for name in MUTATIONS:
            p = subprocess.run([sys.executable, __file__, name], cwd=REPO, capture_output=True, text=True,
                               encoding="utf-8", errors="replace")
            tail = [x for x in p.stdout.splitlines() if " failed" in x or " passed" in x][-1:]
            failed = [x for x in p.stdout.splitlines() if x.startswith("FAILED")][:3]
            out[name] = {"rc": p.returncode, "ringkas": tail, "test_gagal": p.returncode != 0, "contoh": failed}
            print(name, out[name], flush=True)
        (REPO / "work" / "t402").mkdir(parents=True, exist_ok=True)
        (REPO / "work" / "t402" / "mutations.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
        return
    MUTATIONS[which]()
    rc = pytest.main([*TESTS, "-q", "-p", "no:cacheprovider", "--tb=line", "-x", "-k", "not real"])
    sys.exit(int(rc))


if __name__ == "__main__":
    main()
