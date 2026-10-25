"""T-304 Tahap 2–3 (alat, F3): render [5] stylize + [6] export per varian di salinan scratch — tanpa menyentuh work/clips/ atau out/.

    python scripts/t304_tahap2.py render VAR KLIP     # [5] + [6] -> work/t304_scratch/out/VAR/KLIP/*.mp4 ; waktu -> work/t304/render/VAR_KLIP.json

Style = kunci `style` di config (default rough-sketch). Input scratch = Tahap 1 (`work/t304_scratch/VAR/KLIP/`, stable/ + contours/
sudah ada). Output [5] di scratch (strokes/), output [6] di folder out_dir scratch per varian (tidak menyentuh out/ produksi).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rotoscope import export as exp  # noqa: E402
from rotoscope import stylize as sty  # noqa: E402
from rotoscope.config import load_pipeline, load_style, resolve_style, style_name  # noqa: E402
from rotoscope.stage_common import reconfigure_stdio, work_dir_overrides  # noqa: E402

SCR = ROOT / "work" / "t304_scratch"
SCR_OUT = SCR / "out"
OUT = ROOT / "work" / "t304"
CFG = ROOT / "configs" / "default.yaml"
VARIANTS = ("b0", "b0.3", "b0.6", "b1.0", "bs")


def jsave(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"  -> {p.relative_to(ROOT).as_posix()}")


def cmd_render(var: str, clip: str) -> None:
    if var not in VARIANTS:
        raise SystemExit(f"varian tidak dikenal: {var}")
    work = SCR / var / clip
    if not (work / "contours" / "manifest.json").is_file():
        raise SystemExit(f"{work} belum punya contours/ (Tahap 1 belum dijalankan untuk ini)")
    out = SCR_OUT / var / clip
    cfg = load_pipeline(CFG, overrides={**work_dir_overrides(work), "paths.out_dir": str(out)})
    spath = resolve_style(cfg.style)
    style = load_style(spath)
    label = style_name(spath)
    t0 = time.perf_counter()
    r5 = sty.run_stylize(cfg, style, label, log=print)
    t5 = time.perf_counter() - t0
    t1 = time.perf_counter()
    r6 = exp.run_export(cfg, style=style, style_label=label, log=print)
    t6 = time.perf_counter() - t1
    mp4s = sorted(out.glob("*.mp4"))
    jsave(OUT / "render" / f"{var}_{clip}.json", {
        "variant": var, "clip": clip, "style": label, "t_stage5_s": round(t5, 2), "t_stage6_s": round(t6, 2),
        "stage5": {k: r5[k] for k in ("selected", "skipped", "processed") if k in r5},
        "stage6": {k: str(v) for k, v in r6.items() if isinstance(v, (int, float, str))},
        "mp4": [{"name": p.name, "bytes": p.stat().st_size} for p in mp4s]})


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render")
    r.add_argument("var")
    r.add_argument("clip")
    a = ap.parse_args(argv)
    reconfigure_stdio()
    cmd_render(a.var, a.clip)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
