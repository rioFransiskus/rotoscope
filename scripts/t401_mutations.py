"""T-401 mutation check: monkeypatch DARI LUAR berkas test (kode produksi tidak diubah di disk). Tiap mutasi harus membuat
test T-401 GAGAL. Pakai: python scripts/t401_mutations.py  → work/t401/mutations.json (+ IoU SVG-PNG hasil mutasi geser 1 px)."""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402

from rotoscope import noise  # noqa: E402
from rotoscope import stylize as sty  # noqa: E402

TESTS = ["tests/test_stylize_width.py", "tests/test_noise.py", "tests/test_stylize.py"]
ORIG = {"free_ends": sty.free_ends, "stroke_pieces": sty.stroke_pieces, "taper_factor": sty.taper_factor,
        "make_geometry": sty.make_geometry, "piece_outline": sty.piece_outline, "frame_pieces": sty.frame_pieces,
        "value_noise_2d": noise.value_noise_2d, "width_profile": sty.width_profile}


def mutate_taper_on_meeting_ends():
    sty.free_ends = lambda pieces, g: [(not pc.closed and g.taper_on[pc.type] and not pc.edge[0],
                                        not pc.closed and g.taper_on[pc.type] and not pc.edge[1]) for pc in pieces]


def mutate_taper_on_edge_ends():
    def sp(stroke, g, stats, stroke_idx=-1):
        return [dataclasses.replace(p, edge=(False, False)) for p in ORIG["stroke_pieces"](stroke, g, stats, stroke_idx)]
    sty.stroke_pieces = sp


def mutate_floor_off():
    sty.WIDTH_FLOOR_PX = 0.0


def mutate_taper_clamp_off():
    def tf(s, length, free, g):
        lt = g.taper_len                                              # tanpa min(taper_px, L / 2)
        if lt <= 0 or not (free[0] or free[1]):
            return np.ones_like(s)
        t = np.full_like(s, np.inf)
        if free[0]:
            t = np.minimum(t, s / lt)
        if free[1]:
            t = np.minimum(t, (length - s) / lt)
        t = np.clip(t, 0.0, 1.0)
        return g.taper_min + (1.0 - g.taper_min) * t * t * (3.0 - 2.0 * t)
    sty.taper_factor = tf


def mutate_seed_leaks_frame_index():
    def fp(doc, g):
        g2 = dataclasses.replace(g, noise_seed=int(noise.seed_of(int(doc["frame_index"]), g.noise_seed)))
        return ORIG["frame_pieces"](doc, g2)
    sty.frame_pieces = fp


def mutate_svg_uses_other_polygon():
    sty.piece_outline = lambda pc, g: [q + 1.0 for q in ORIG["piece_outline"](pc, g)]       # geser 1 px (garis 9 px)


def mutate_type_hierarchy_ignored():
    def mg(style, width, height):
        g = ORIG["make_geometry"](style, width, height)
        base = style.stroke.width_base * g.unit
        return dataclasses.replace(g, widths={t: base for t in g.widths})
    sty.make_geometry = mg


def mutate_noise_not_normalized():
    noise.value_noise_2d = lambda seed, x, y: 3.0 * ORIG["value_noise_2d"](seed, x, y)


def mutate_noise_arclength_not_position():
    """Noise memakai busur dari titik awal (varian A) alih-alih posisi: gagal pada tes 'terkunci posisi'."""
    def wp(points, closed, typ, free, g):
        s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(points, axis=0).T))]
        w = np.full(len(points), g.widths[typ])
        if g.variation > 0 and g.noise_cell > 0:
            w = w * (1.0 + g.variation * ORIG["value_noise_2d"](np.uint64(g.noise_seed), s / g.noise_cell, np.zeros_like(s)))
        return np.maximum(w, g.floor)
    sty.width_profile = wp


MUTATIONS = {"taper pada ujung bertemu": mutate_taper_on_meeting_ends, "taper pada ujung tepi": mutate_taper_on_edge_ends,
             "lantai tebal mati": mutate_floor_off, "taper-clamp mati": mutate_taper_clamp_off,
             "seed bocor frame_index": mutate_seed_leaks_frame_index, "SVG memakai poligon berbeda dari raster (geser 1 px)":
             mutate_svg_uses_other_polygon, "hierarki tipe diabaikan": mutate_type_hierarchy_ignored,
             "noise tidak dinormalisasi": mutate_noise_not_normalized, "noise berbasis busur (bukan posisi)": mutate_noise_arclength_not_position}


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
        (REPO / "work" / "t401").mkdir(parents=True, exist_ok=True)
        (REPO / "work" / "t401" / "mutations.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
        return
    MUTATIONS[which]()
    rc = pytest.main([*TESTS, "-q", "-p", "no:cacheprovider", "--tb=line", "-x", "-k", "not real"])
    sys.exit(int(rc))


if __name__ == "__main__":
    main()
