"""ARSIP: studi T-404b; modul grain dihapus (tekstur garis ditolak, docs/04 "Hasil T-404b"). TIDAK BISA DIJALANKAN di repo ini: mengimpor
`rotoscope.grain` dan menjalankan tests/test_grain.py + tests/test_export_grain.py yang sudah dihapus. Tidak ada commit asal (tidak pernah di-commit);
sumber disimpan di `work/t404b/grain_source/` (ter-ignore, lokal). Hasil terakhir: `work/t404b/mutations.json` (15/15 mutasi membuat test gagal).

T-404b mutation check: monkeypatch DARI LUAR berkas test (kode produksi tidak diubah di disk). Tiap mutasi harus membuat test tekstur garis GAGAL.
Pakai: python scripts/t404b_mutations.py  → work/t404b/mutations.json. Satu proses per mutasi (modul bersih)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402

from rotoscope import export as ex  # noqa: E402
from rotoscope import grain as gr  # noqa: E402
from rotoscope import paper as pap  # noqa: E402
from rotoscope import stylize as sty  # noqa: E402

TESTS = ["tests/test_grain.py", "tests/test_export_grain.py", "tests/test_paper.py"]
ORIG_COMPOSE = pap.compose_textured
ORIG_BUILD = gr.build_line_texture
ORIG_RANK = gr.rank_uniform
ORIG_SVG_HEAD = sty._svg_head
ORIG_ASSET = gr._asset
ORIG_LEVELS = pap.LevelMap.levels_of


def _set_compose(fn):
    pap.compose_textured = fn
    ex.pap.compose_textured = fn


def mutate_texture_adds_ink():
    gr.m_from_g = lambda g, s: (np.float32(1.0) + np.float32(s) * g).astype(np.float32)          # m > 1


def mutate_texture_changes_per_frame():
    n = [0]

    def build(w, h, style):
        n[0] += 1
        lay = ORIG_BUILD(w, h, style)
        return gr.GrainLayer(m=np.roll(lay.m, n[0], axis=1), info=lay.info)                     # peta berbeda tiap panggilan
    gr.build_line_texture = build


def mutate_f_recovered_wrong():
    def lo(self, img):
        lv, box = ORIG_LEVELS(self, img)
        return np.where(lv > 0, np.minimum(lv + 3.0, 255.0), lv).astype(np.float32), box
    pap.LevelMap.levels_of = lo


def mutate_m_sticks_to_path():
    def comp(img, levels, layer, ink, grain=None):
        if grain is None:
            return ORIG_COMPOSE(img, levels, layer, ink)
        lv, _ = levels.levels_of(img)
        return ORIG_COMPOSE(img, levels, layer, ink, np.where(lv > 0, np.float32(0.7), np.float32(1.0)).astype(np.float32))     # m dari bentuk garis
    _set_compose(comp)


def mutate_mode_none_not_identical():
    def comp(img, levels, layer, ink, grain=None):
        return ORIG_COMPOSE(img, levels, layer, ink, np.full(layer.u8.shape[:2], 0.99, np.float32) if grain is None else grain)
    _set_compose(comp)


def mutate_preflight_assets_dead():
    gr.validate_grain = lambda style: None


def mutate_strength_ignored():
    gr.m_from_g = lambda g, s: (np.float32(1.0) - np.float32(0.35) * g).astype(np.float32)


def mutate_svg_textured():
    sty._svg_head = lambda g, style: ORIG_SVG_HEAD(g, style) + f"<!-- grain {style.texture.mode} {style.texture.grain_strength} -->\n"


def mutate_texture_not_in_match_keys():
    ex.MANIFEST_MATCH_KEYS = tuple(k for k in ex.MANIFEST_MATCH_KEYS if k != "texture")


def mutate_sha_not_in_ref():
    gr._asset = lambda style: (ORIG_ASSET(style)[0], None)


def mutate_brush_tiled_periodic():
    def gb(w, h, style):
        a = gr.load_brush_alpha(style.texture.brush_image)
        return np.tile(a, (h // a.shape[0] + 1, w // a.shape[1] + 1))[:h, :w].astype(np.float32)
    gr.g_brush = gb


def mutate_paper_sign_flipped():
    def gp(w, h, style):
        return np.float32(1.0) - ORIG_GP(w, h, style)
    ORIG_GP = gr.g_paper
    gr.g_paper = gp


def mutate_rank_ties_by_position():
    def rk(x):
        flat = x.ravel()
        order = np.argsort(flat, kind="stable")
        r = np.empty(flat.size, np.float32)
        r[order] = (np.arange(flat.size, dtype=np.float64) / max(flat.size - 1, 1)).astype(np.float32)
        return r.reshape(x.shape)
    gr.rank_uniform = rk


def mutate_flat_surface_ignores_grain():
    gr.surface_flat = lambda style: pap.paper_flat(style)
    ex.gr.surface_flat = gr.surface_flat


def mutate_stage5_uses_texture_params():
    sty.ACTIVE_SCALARS = (*sty.ACTIVE_SCALARS, "texture.grain_strength")


MUTATIONS = {"tekstur menambah tinta (m > 1)": mutate_texture_adds_ink, "tekstur berubah per frame": mutate_texture_changes_per_frame,
             "f dipulihkan salah": mutate_f_recovered_wrong, "m menempel ke jalur (bentuk garis)": mutate_m_sticks_to_path,
             "mode none tidak identik": mutate_mode_none_not_identical, "aset tidak divalidasi di pre-flight": mutate_preflight_assets_dead,
             "strength diabaikan": mutate_strength_ignored, "SVG ikut bertekstur": mutate_svg_textured,
             "texture tidak di MANIFEST_MATCH_KEYS": mutate_texture_not_in_match_keys, "sha256 aset tidak di ref": mutate_sha_not_in_ref,
             "kuas dipasang sebagai ubin periodik": mutate_brush_tiled_periodic, "tanda A dibalik (g = peringkat)": mutate_paper_sign_flipped,
             "peringkat nilai kembar by posisi": mutate_rank_ties_by_position, "butiran di kertas datar diabaikan": mutate_flat_surface_ignores_grain,
             "[5] memakai texture.* (strokes basi)": mutate_stage5_uses_texture_params}


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else None
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if which is None:
        import subprocess
        out = {}
        for name in MUTATIONS:
            p = subprocess.run([sys.executable, __file__, name], cwd=REPO, capture_output=True, text=True, encoding="utf-8", errors="replace")
            tail = [x for x in p.stdout.splitlines() if " failed" in x or " passed" in x][-1:]
            failed = [x for x in p.stdout.splitlines() if x.startswith("FAILED") or ".py:" in x][:3]
            out[name] = {"rc": p.returncode, "ringkas": tail, "test_gagal": p.returncode != 0, "contoh": failed}
            print(name, out[name], flush=True)
        (REPO / "work" / "t404b").mkdir(parents=True, exist_ok=True)
        (REPO / "work" / "t404b" / "mutations.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"{sum(v['test_gagal'] for v in out.values())}/{len(out)} mutasi membuat test gagal")
        return
    MUTATIONS[which]()
    sys.exit(int(pytest.main([*TESTS, "-q", "-p", "no:cacheprovider", "--tb=line", "-x", "-k", "not torch"])))


if __name__ == "__main__":
    main()
