"""T-404a (Opsi B) mutation check: monkeypatch DARI LUAR berkas test (kode produksi tidak diubah di disk). Tiap mutasi harus membuat test kertas / export GAGAL.
Pakai: python scripts/t404a_mutations.py  → work/t404a/mutations.json. Satu proses per mutasi (modul bersih)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))
sys.path.insert(0, str(REPO / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import stylize_metrics as sm  # noqa: E402

from rotoscope import export as ex  # noqa: E402
from rotoscope import paper as pap  # noqa: E402
from rotoscope import stylize as sty  # noqa: E402

TESTS = ["tests/test_paper.py", "tests/test_export_paper.py"]
ORIG = {n: getattr(pap, n) for n in ("normalize_texture", "texture_multiplier", "orient_texture", "fit_texture", "vignette_field",
                                      "compose_textured", "texture_sha256", "validate_texture", "paper_flat")}
ORIG["levels_of"] = pap.LevelMap.levels_of
ORIG["LevelMap_init"] = pap.LevelMap.__init__
ORIG_SVG_HEAD = sty._svg_head
ORIG_RECOVER = sm.recover_f


def mutate_no_mean_normalisation():
    pap.normalize_texture = lambda crop: (crop / np.float32(128.0), 128.0)          # pembagi tetap, bukan rata-rata potongan


def mutate_gain_ignored():
    pap.texture_multiplier = lambda t, op, gain: ORIG["texture_multiplier"](t, op, 1.0)


def mutate_vignette_hits_ink():
    def comp(img, levels, layer, ink):
        out = ORIG["compose_textured"](img, levels, layer, ink)
        dark = (layer.f32 / layer.f32.max()).astype(np.float32)                    # kertas relatif terang: sudut < 1 -> tinta ikut digelapkan
        return np.rint(out.astype(np.float32) * dark).astype(np.uint8)
    pap.compose_textured = comp
    ex.pap.compose_textured = comp


def mutate_paper_not_static_per_frame():
    counter = [0]

    def comp(img, levels, layer, ink):
        counter[0] += 1
        out = ORIG["compose_textured"](img, levels, layer, ink)
        return np.clip(out.astype(int) + counter[0] % 5, 0, 255).astype(np.uint8)
    pap.compose_textured = comp
    ex.pap.compose_textured = comp


def mutate_rotation_lost():
    pap.orient_texture = lambda tex, w, h: (tex, False)


def mutate_level_recovery_wrong():
    def lo(self, img):
        lv, box = ORIG["levels_of"](self, img)
        return np.where(lv > 0, np.minimum(lv + 3.0, 255.0), lv).astype(np.float32), box
    pap.LevelMap.levels_of = lo


def mutate_collision_rule_min_level():
    def init(self, paper_rgb, ink_rgb):
        ORIG["LevelMap_init"](self, paper_rgb, ink_rgb)
        lut = sty.ink_lut(type("G", (), {"paper": paper_rgb, "ink": ink_rgb}))
        keys = pap._pack(lut)
        self.levels = np.array([int(np.flatnonzero(keys == k)[0]) for k in self.keys], np.float32)       # level TERKECIL, bukan rata-rata
    pap.LevelMap.__init__ = init


def mutate_svg_textured():
    sty._svg_head = lambda g, style: ORIG_SVG_HEAD(g, style) + f"<!-- paper {style.paper.texture_opacity} {style.paper.vignette} -->\n"


def mutate_stage5_uses_paper_params():
    sty.ACTIVE_SCALARS = (*sty.ACTIVE_SCALARS, "paper.vignette", "paper.texture_opacity")


def mutate_preflight_texture_dead():
    pap.validate_texture = lambda style: None


def mutate_flat_not_detected():
    pap.paper_flat = lambda style: False
    ex.pap.paper_flat = lambda style: False


def mutate_recover_wrong_formula():
    def rec(img_rgb, paper_f32, ink_rgb):
        flat = paper_f32.mean(axis=(0, 1), keepdims=True)                           # mengandaikan latar konstan (rumus lama)
        ink = np.array(ink_rgb, np.float32)
        d = flat - ink
        ch = int(np.argmax(np.abs(d)))
        return ((flat[..., ch] - img_rgb[..., ch].astype(np.float32)) / d[..., ch]).astype(np.float32)
    sm.recover_f = rec


def mutate_vignette_circular():
    def vf(w, h, amount):
        cx, cy = (w - 1) / 2, (h - 1) / 2
        x = np.arange(w, dtype=np.float32) - np.float32(cx)
        y = np.arange(h, dtype=np.float32) - np.float32(cy)
        d = np.sqrt(x[None, :] ** 2 + y[:, None] ** 2) / np.float32(np.hypot(cx, cy))
        return (np.float32(1) - np.float32(amount) * np.minimum(d, 1) ** 2).astype(np.float32)
    pap.vignette_field = vf


def mutate_crop_not_centered():
    def fit(tex, w, h):
        crop, mode, s = ORIG["fit_texture"](tex, w, h)
        return (np.ascontiguousarray(tex[:h, :w]), mode, s) if mode == "native" else (crop, mode, s)
    pap.fit_texture = fit


def mutate_sha_not_in_ref():
    pap.texture_sha256 = lambda style: None


def mutate_nearest_not_cubic():
    def fit(tex, w, h):
        ih, iw = tex.shape
        if iw >= w and ih >= h:
            return ORIG["fit_texture"](tex, w, h)
        s = max(w / iw, h / ih)
        t2 = cv2.resize(tex, (max(w, int(np.ceil(iw * s))), max(h, int(np.ceil(ih * s)))), interpolation=cv2.INTER_NEAREST)
        x0, y0 = (t2.shape[1] - w) // 2, (t2.shape[0] - h) // 2
        return np.ascontiguousarray(t2[y0:y0 + h, x0:x0 + w]), "scaled", s
    pap.fit_texture = fit


MUTATIONS = {"tanpa normalisasi rata-rata": mutate_no_mean_normalisation, "gain diabaikan": mutate_gain_ignored,
             "vignette mengenai tinta": mutate_vignette_hits_ink, "kertas tidak statis per frame": mutate_paper_not_static_per_frame,
             "rotasi hilang": mutate_rotation_lost, "pemulihan level salah": mutate_level_recovery_wrong,
             "aturan tabrakan LUT = level terkecil": mutate_collision_rule_min_level, "SVG ikut bertekstur": mutate_svg_textured,
             "[5] memakai paper.* (strokes basi)": mutate_stage5_uses_paper_params, "pre-flight kertas mati": mutate_preflight_texture_dead,
             "kertas datar tidak terdeteksi": mutate_flat_not_detected, "pemulihan f rumus latar konstan": mutate_recover_wrong_formula,
             "vignette jarak lingkaran": mutate_vignette_circular, "potong tidak di tengah": mutate_crop_not_centered,
             "sha256 tekstur tidak di ref": mutate_sha_not_in_ref, "skala nearest (bukan cubic)": mutate_nearest_not_cubic}


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
        (REPO / "work" / "t404a").mkdir(parents=True, exist_ok=True)
        (REPO / "work" / "t404a" / "mutations.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"{sum(v['test_gagal'] for v in out.values())}/{len(out)} mutasi membuat test gagal")
        return
    MUTATIONS[which]()
    sys.exit(int(pytest.main([*TESTS, "-q", "-p", "no:cacheprovider", "--tb=line", "-x", "-k", "not torch"])))


if __name__ == "__main__":
    main()
