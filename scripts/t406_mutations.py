"""T-406 mutation check: monkeypatch DARI LUAR berkas test (kode produksi tidak diubah di disk). Tiap mutasi harus membuat tests/test_presets.py GAGAL.
Pakai: python scripts/t406_mutations.py  → work/t406/mutations.json. Satu proses per mutasi (modul bersih)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))
sys.path.insert(0, str(REPO / "src"))

from rotoscope import cli  # noqa: E402
from rotoscope import config as cfg_mod  # noqa: E402
from rotoscope import export as ex  # noqa: E402
from rotoscope import stylize as sty  # noqa: E402

TESTS = ["tests/test_presets.py"]
ORIG_RESOLVE, ORIG_FILENAME, ORIG_SVG_DIR, ORIG_RUN = cfg_mod.resolve_style, ex.resolve_filename, ex.svg_dir_for, ex.run_export
ORIG_EDGE_FADE, ORIG_PICK = sty.edge_fade, cli.pick_style


def rebind_resolver(fn):
    """Semua pemakai resolve_style (stage, cli, config) memakai fungsi mutan."""
    cfg_mod.resolve_style = cli.resolve_style = ex.resolve_style = sty.resolve_style = fn


def mutate_name_relative_to_cwd():
    cfg_mod.project_root = lambda: Path.cwd()                                  # nama dicari relatif ke cwd, bukan root project


def mutate_two_selection_paths():
    cli.pick_style = lambda flag, cfg: (lambda p: (p, cfg_mod.style_name(p)))(ORIG_RESOLVE(flag))   # cli mengabaikan kunci config `style`


def mutate_config_beats_flag():
    def pick(flag, cfg):
        p = ORIG_RESOLVE(cfg.style if cfg.style != cfg_mod.DEFAULT_STYLE_NAME else flag)
        return p, cfg_mod.style_name(p)
    cli.pick_style = pick


def mutate_style_placeholder_ignored():
    ex.resolve_filename = lambda template, src, limit=None, start=None, style=cfg_mod.DEFAULT_STYLE_NAME: \
        ORIG_FILENAME(template.replace("{style}", ""), src, limit, start, style)


def mutate_preset_overwrites_other_mp4():
    def run(cfg, **kw):
        kw["style_label"] = cfg_mod.DEFAULT_STYLE_NAME                         # nama style tidak pernah sampai ke nama file / manifest
        return ORIG_RUN(cfg, **kw)
    ex.run_export = run


def mutate_svg_dir_ignores_style():
    ex.svg_dir_for = lambda out, src, template=ex.DEFAULT_FILENAME_TEMPLATE, style=cfg_mod.DEFAULT_STYLE_NAME: ORIG_SVG_DIR(out, src)


def mutate_rough_sketch_changed():
    rebind_resolver(lambda ref, root=None: ORIG_RESOLVE("clean-line" if ref in (None, "rough-sketch") else ref, root))


def mutate_case_insensitive_names():
    def res(ref, root=None):
        try:
            return ORIG_RESOLVE(ref, root)
        except cfg_mod.ConfigError:
            return ORIG_RESOLVE(str(ref).lower(), root)
    rebind_resolver(res)


def mutate_unknown_name_without_listing():
    def res(ref, root=None):
        try:
            return ORIG_RESOLVE(ref, root)
        except cfg_mod.ConfigError as e:
            raise cfg_mod.ConfigError(f"style {ref!r} tidak dikenal") from e
    rebind_resolver(res)


def mutate_preset_missing_key_passes():
    import test_presets as tp
    orig = tp.load_raw

    def load_raw(name):
        d = orig(name)
        if name == "heavy-marker":
            d["multipass"].pop("temporal_mode")                                # preset kehilangan satu kunci
        return d
    tp.load_raw = load_raw


def mutate_old_dead_zone():
    sty.edge_fade = lambda p, g, half_width=None: ORIG_EDGE_FADE(p, g)         # zona mati lama: margin tetap 2 px


def mutate_hard_cut_guard():
    def fade(p, g, half_width=None):
        out = ORIG_EDGE_FADE(p, g)
        if half_width is not None:
            d = (g.out_w - p[:, 0])
            out = np.where(d < half_width + sty.EDGE_INK_GUARD_PX, 0.0, out)    # pemotongan keras per titik (diskontinu)
        return out
    import numpy as np
    sty.edge_fade = fade


def mutate_algo_rev_not_bumped():
    sty.ALGO_REV = 1


def mutate_no_overwrite_warning():
    def run(cfg, **kw):
        log = kw.get("log", print)
        kw["log"] = lambda m: None if "export.filename" in m else log(m)
        return ORIG_RUN(cfg, **kw)
    ex.run_export = run


def mutate_always_warn():
    def run(cfg, **kw):
        log = kw.get("log", print)
        out = ORIG_RUN(cfg, **kw)
        log("PERINGATAN: ditimpa — export.filename tidak memuat {style}")
        return out
    ex.run_export = run


MUTATIONS = {"nama diresolusi relatif ke cwd": mutate_name_relative_to_cwd, "dua jalur pemilihan (cli mengabaikan config)": mutate_two_selection_paths,
             "config mengalahkan flag": mutate_config_beats_flag, "{style} diabaikan": mutate_style_placeholder_ignored,
             "preset menimpa MP4 preset lain": mutate_preset_overwrites_other_mp4, "folder SVG tidak mengikuti style": mutate_svg_dir_ignores_style,
             "rough-sketch berubah": mutate_rough_sketch_changed, "nama tak peka huruf besar-kecil": mutate_case_insensitive_names,
             "error nama tanpa daftar preset": mutate_unknown_name_without_listing, "preset tanpa kunci lengkap lolos": mutate_preset_missing_key_passes,
             "zona mati lama (margin tetap 2 px)": mutate_old_dead_zone, "guard pemotongan keras (diskontinu)": mutate_hard_cut_guard,
             "ALGO_REV tidak naik": mutate_algo_rev_not_bumped, "peringatan penimpaan hilang": mutate_no_overwrite_warning,
             "peringatan penimpaan selalu muncul": mutate_always_warn}


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
        (REPO / "work" / "t406").mkdir(parents=True, exist_ok=True)
        (REPO / "work" / "t406" / "mutations.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"{sum(v['test_gagal'] for v in out.values())}/{len(out)} mutasi membuat test gagal")
        return
    MUTATIONS[which]()
    sys.exit(int(pytest.main([*TESTS, "-q", "-p", "no:cacheprovider", "--tb=line", "-x", "-k", "not torch"])))


if __name__ == "__main__":
    main()
