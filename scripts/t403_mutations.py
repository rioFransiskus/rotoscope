"""T-403 mutation check: monkeypatch DARI LUAR berkas test (kode produksi tidak diubah di disk). Tiap mutasi harus membuat test multipass
GAGAL. Pakai: python scripts/t403_mutations.py  → work/t403/mutations.json."""

from __future__ import annotations

import dataclasses
import json
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))
sys.path.insert(0, str(REPO / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from rotoscope import noise  # noqa: E402
from rotoscope import stylize as sty  # noqa: E402

TESTS = ["tests/test_stylize_multipass.py"]
ORIG = {"multipass_pieces": sty.multipass_pieces, "pass_geometry": sty.pass_geometry, "make_geometry": sty.make_geometry,
        "ink_fraction": sty.ink_fraction, "ink_box": sty.ink_box, "ink_fraction_box": sty.ink_fraction_box,"render_svg_passes": sty.render_svg_passes, "edge_fade": sty.edge_fade,
        "pass_alpha": sty.Geometry.pass_alpha, "legacy": sty.Geometry.legacy, "scale_of": sty.Geometry.scale_of}


def mutate_offset_along_normal():
    def mp(pieces, g, fi):
        out = ORIG["multipass_pieces"](pieces, g, fi)
        res = [out[0]]
        for pk in out[1:]:
            cur = []
            for base, pc in zip(pieces, pk):
                d = pc.points - base.points
                p = base.points
                t = np.roll(p, -1, axis=0) - np.roll(p, 1, axis=0) if base.closed else np.gradient(p, axis=0)
                t = t / np.maximum(np.hypot(*t.T), 1e-12)[:, None]
                n = np.column_stack([-t[:, 1], t[:, 0]])
                cur.append(dataclasses.replace(base, points=np.round(p + (d * n).sum(1)[:, None] * n, 2) + 0.0))
            res.append(cur)
        return res
    sty.multipass_pieces = mp


def mutate_independent_field_per_stroke():
    sty.pass_geometry = lambda g, k: dataclasses.replace(ORIG["pass_geometry"](g, k), jitter_s=1.0)


def mutate_pass_without_edge_guard():
    def mp(pieces, g, fi):
        sty.edge_fade = lambda p, gg: np.ones(len(p))
        try:
            return ORIG["multipass_pieces"](pieces, g, fi)
        finally:
            sty.edge_fade = ORIG["edge_fade"]
    sty.multipass_pieces = mp


def mutate_over_is_sum():
    def inf(passes, g, box):
        x0, y0, x1, y1 = box
        f = np.zeros((y1 - y0, x1 - x0), np.float32)
        for k, pcs in enumerate(passes):
            cov = cv2.resize(sty.render_mask(pcs, g, box), (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
            f += np.float32(g.pass_alpha(k)) * cov
        return np.clip(f, 0.0, 1.0)
    sty.ink_fraction_box = inf


def mutate_falloff_ignored():
    sty.Geometry.pass_alpha = lambda self, k: self.opacity


def mutate_pass_order_changes_result():
    """Komposisi sekuensial "src over dst" dengan bobot menempel urutan (tidak komutatif)."""
    def inf(passes, g, box):
        x0, y0, x1, y1 = box
        f = np.zeros((y1 - y0, x1 - x0), np.float32)
        for k, pcs in enumerate(passes):
            cov = cv2.resize(sty.render_mask(pcs, g, box), (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
            f = f * np.float32(0.5) + np.float32(g.pass_alpha(k)) * cov * np.float32(0.5 + 0.5 * k / max(len(passes) - 1, 1))
        return f
    sty.ink_fraction_box = inf


def mutate_svg_without_group_opacity():
    def rs(passes, g, style):
        svg = ORIG["render_svg_passes"](passes, g, style)
        return re.sub(rb'(<g id="pass_\d+") opacity="[0-9.]+"', rb"\1", svg)
    sty.render_svg_passes = rs


def mutate_field_scale_not_following_offset():
    def mg(style, width, height):
        g = ORIG["make_geometry"](style, width, height)
        return dataclasses.replace(g, mp_cell=30.0 * g.unit if g.mp_amp > 0 else 0.0)
    sty.make_geometry = mg


def mutate_one_pass_not_identical_t402():
    sty.Geometry.legacy = property(lambda self: False)


def mutate_same_seed_for_all_passes():
    def mg(style, width, height):
        g = ORIG["make_geometry"](style, width, height)
        return dataclasses.replace(g, mp_seeds=tuple(g.mp_seeds[0] for _ in g.mp_seeds))
    sty.make_geometry = mg


def mutate_static_mode_moves():
    def mg(style, width, height):
        return dataclasses.replace(ORIG["make_geometry"](style, width, height), mp_fixed=False)
    sty.make_geometry = mg


def mutate_type_scale_ignored():
    sty.Geometry.scale_of = lambda self, typ: 1.0


def mutate_noise_unbounded_amplitude_without_unit():
    def mg(style, width, height):
        g = ORIG["make_geometry"](style, width, height)
        return dataclasses.replace(g, mp_amp=g.mp_amp / g.unit, mp_cell=g.mp_cell / g.unit)
    sty.make_geometry = mg


def mutate_pass_edge_dead_not_widened():
    """Perbaikan 2 (Tahap 4 revisi): pass k >= 1 memakai zona mati tepi pass 0 (tanpa tambahan)."""
    sty.pass_geometry = lambda g, k: dataclasses.replace(ORIG["pass_geometry"](g, k), edge_dead=g.edge_dead)


def mutate_window_too_small():
    """Optimasi jendela: margin jendela tinta 0 dan ujung jendela dipotong 3 px (tinta terpotong)."""
    def box(passes, g):
        b = ORIG["ink_box"](passes, g)
        return None if b is None else (b[0] + 3, b[1] + 3, b[2] - 3, b[3] - 3)
    sty.ink_box = box


def mutate_seam_cracked_at_middle():
    """Retak di tengah strok tertutup pass k >= 1 (separuh titik terakhir digeser 4 px): interior DAN seam retak — hanya batas Lipschitz
    interior (metrik revisi) yang menangkapnya; kriteria relatif saja lolos."""
    def mp(pieces, g, fi):
        out = ORIG["multipass_pieces"](pieces, g, fi)
        res = [out[0]]
        for pk in out[1:]:
            cur = []
            for pc in pk:
                if pc.closed and len(pc.points) > 8:
                    p = pc.points.copy()
                    p[len(p) // 2:] += 4.0
                    pc = dataclasses.replace(pc, points=np.round(p, 2) + 0.0)
                cur.append(pc)
            res.append(cur)
        return res
    sty.multipass_pieces = mp


MUTATIONS = {"offset sepanjang normal": mutate_offset_along_normal,
             "medan independen per strok": mutate_independent_field_per_stroke,
             "pass tanpa penjaga tepi": mutate_pass_without_edge_guard,
             '"over" diganti penjumlahan': mutate_over_is_sum,
             "opacity_falloff diabaikan": mutate_falloff_ignored,
             "urutan pass mengubah hasil": mutate_pass_order_changes_result,
             "SVG tanpa opacity grup": mutate_svg_without_group_opacity,
             "skala medan tidak mengikuti offset": mutate_field_scale_not_following_offset,
             "1 pass tidak identik T-402": mutate_one_pass_not_identical_t402,
             "seed sama untuk semua pass": mutate_same_seed_for_all_passes,
             "mode fixed ikut bergerak": mutate_static_mode_moves,
             "opacity_scale tipe diabaikan": mutate_type_scale_ignored,
             "amplitudo / sel pass tanpa unit": mutate_noise_unbounded_amplitude_without_unit,
             "zona mati tepi pass tidak diperlebar": mutate_pass_edge_dead_not_widened,
             "jendela tinta terlalu kecil": mutate_window_too_small,
             "retak di tengah strok tertutup": mutate_seam_cracked_at_middle}


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else None
    if which is None:                       # induk: satu proses per mutasi (modul bersih)
        import subprocess
        out = {}
        for name in MUTATIONS:
            p = subprocess.run([sys.executable, __file__, name], cwd=REPO, capture_output=True, text=True,
                               encoding="utf-8", errors="replace")
            tail = [x for x in p.stdout.splitlines() if " failed" in x or " passed" in x][-1:]
            failed = [x for x in p.stdout.splitlines() if x.startswith("FAILED") or ".py:" in x][:3]
            out[name] = {"rc": p.returncode, "ringkas": tail, "test_gagal": p.returncode != 0, "contoh": failed}
            print(name, out[name], flush=True)
        (REPO / "work" / "t403").mkdir(parents=True, exist_ok=True)
        (REPO / "work" / "t403" / "mutations.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"{sum(v['test_gagal'] for v in out.values())}/{len(out)} mutasi membuat test gagal")
        return
    MUTATIONS[which]()
    rc = pytest.main([*TESTS, "-q", "-p", "no:cacheprovider", "--tb=line", "-x", "-k", "not torch"])
    sys.exit(int(rc))


if __name__ == "__main__":
    main()
