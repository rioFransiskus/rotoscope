"""Alat ukur sekali pakai T-404a Opsi B (BUKAN bagian src/). Tanpa GPU. Bekerja pada SALINAN klip di work/t404a_scratch/s1/<klip> (meta.json + contours/);
klip asli (work/clips) dan out/ tidak disentuh. Keluaran JSON ke work/t404a/ (ter-ignore).

    python scripts/t404a_measure.py stage5    # [5] dengan style default -> strokes byte-identik T-403 (hash F1) + kertas tidak membuat basi
    python scripts/t404a_measure.py oracle    # susunan export (invers LUT dari PNG datar) vs ORACLE jalur [5] lama, SEMUA frame, kandidat op 0,35 gain 3 vig 0
    python scripts/t404a_measure.py cost      # waktu susun per frame (baca PNG + invers LUT + blend) pada kedua klip
    python scripts/t404a_measure.py final     # setelah `run samples/<klip>.mp4` nyata: MP4, strokes / hulu / out/svg tidak berubah, metrik MP4 (klip ASLI, hanya dibaca)
    python scripts/t404a_measure.py all       # stage5 + oracle + cost
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

import export_metrics as em  # noqa: E402
import paper_oracle as po  # noqa: E402

from rotoscope import paper as pap  # noqa: E402
from rotoscope import stylize as sty  # noqa: E402
from rotoscope.config import load_pipeline, load_style  # noqa: E402
from rotoscope.stage_common import work_dir_overrides  # noqa: E402

S1 = ROOT / "work" / "t404a_scratch" / "s1"
OUT = ROOT / "work" / "t404a"
STYLE_PATH = ROOT / "configs" / "styles" / "rough-sketch.yaml"
CLIPS = ("test_short", "test")


def save(name: str, obj) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=1), encoding="utf-8")
    print(json.dumps(obj, indent=1)[:5000], flush=True)


def before(clip: str) -> dict[str, str]:
    out = {}
    for line in (OUT / f"before_strokes_{clip}.sha256").read_text().splitlines():
        h, n = line.split(" *", 1)
        out[n] = h
    return out


def hashes(clip: str) -> dict[str, str]:
    d = S1 / clip / "strokes"
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.glob("frame_*")) if p.suffix in (".svg", ".png")}


def run_stage5(clip: str, style, restart: bool = False) -> dict:
    d = S1 / clip / "strokes"
    if restart and d.exists():
        shutil.rmtree(d)
    cfg = load_pipeline(None, overrides=work_dir_overrides(S1 / clip))
    return sty.run_stylize(cfg, style, "t404a", log=lambda m: None)


def cmd_stage5() -> None:
    res = {}
    default = load_style(STYLE_PATH)
    for clip in CLIPS:
        run = run_stage5(clip, default, restart=True)
        ref, h = before(clip), hashes(clip)
        diff = [k for k in ref if h.get(k) != ref[k]]
        m = json.loads((S1 / clip / "strokes" / "manifest.json").read_text())
        res[f"{clip}/default"] = {"processed": run["processed"], "n_files": len(ref), "n_identical": len(ref) - len(diff),
                                  "contract": m["contract"], "paper_keys_in_style_params": [k for k in m["style_params"] if k.startswith("paper.")],
                                  "style_hash": m["style_hash"][:16]}
        for tag, ov in (("op0.5_gain6_vig0.25", {"paper.texture_opacity": 0.5, "paper.texture_gain": 6.0, "paper.vignette": 0.25}),
                        ("enabled_false", {"paper.enabled": False}), ("flat", {"paper.texture_opacity": 0.0})):
            r = run_stage5(clip, load_style(STYLE_PATH, overrides=ov))
            m2 = json.loads((S1 / clip / "strokes" / "manifest.json").read_text())
            res[f"{clip}/{tag}"] = {"processed": r["processed"], "stale": r["stale"], "style_hash_same": m2["style_hash"] == m["style_hash"],
                                    "files_identical": hashes(clip) == h}
    save("t4_stage5.json", res)


def cmd_oracle() -> None:
    """Semua frame: susunan export (PNG datar -> LevelMap -> kertas) vs oracle (kertas dirender langsung dari level mentah)."""
    st = load_style(STYLE_PATH)
    res = {}
    for clip in CLIPS:
        m = json.loads((S1 / clip / "meta.json").read_text())
        g = sty.make_geometry(st, int(m["working_width"]), int(m["working_height"]))
        layer = pap.render_paper(g.out_w, g.out_h, st)
        lm = pap.LevelMap(g.paper, g.ink)
        n = int(m["frame_count"])
        max_d, diff1, diff_gt1, px_total, ink_px, frames_gt1 = 0, 0, 0, 0, 0, 0
        for i in range(n):
            doc = json.loads((S1 / clip / "contours" / f"frame_{i:05d}.json").read_text(encoding="utf-8"))
            passes, _ = sty.frame_passes(doc, g)
            flat_png = em.read_png_rgb(S1 / clip / "strokes" / f"frame_{i:05d}.png")
            assert np.array_equal(flat_png, po.flat_rgb(passes, g)), f"PNG [5] != render ulang pada frame {i}"
            comp = pap.compose_textured(flat_png, lm, layer, g.ink)
            orc = po.render_rgb(passes, g, layer)
            d = np.abs(comp.astype(int) - orc.astype(int)).max(-1)
            max_d = max(max_d, int(d.max()))
            diff1 += int((d == 1).sum())
            diff_gt1 += int((d > 1).sum())
            frames_gt1 += int(d.max() > 1)
            px_total += d.size
            ink_px += int((np.abs(flat_png.astype(int) - np.array(g.paper)).max(-1) > 0).sum())
        res[clip] = {"frames": n, "max_abs_diff": max_d, "px_diff_eq1": diff1, "px_diff_gt1": diff_gt1, "frames_with_diff_gt1": frames_gt1,
                     "pct_px_diff1_of_all": 100 * diff1 / px_total, "pct_px_diff1_of_ink_px": 100 * diff1 / max(ink_px, 1),
                     "lut_collisions": lm.n_collisions}
        print(clip, res[clip], flush=True)
    save("t4_oracle.json", res)


def cmd_cost() -> None:
    st = load_style(STYLE_PATH)
    res = {}
    for clip in CLIPS:
        m = json.loads((S1 / clip / "meta.json").read_text())
        g = sty.make_geometry(st, int(m["working_width"]), int(m["working_height"]))
        layer = pap.render_paper(g.out_w, g.out_h, st)
        lm = pap.LevelMap(g.paper, g.ink)
        ts_read, ts_comp = [], []
        for i in range(int(m["frame_count"])):
            t0 = time.perf_counter()
            img = em.read_png_rgb(S1 / clip / "strokes" / f"frame_{i:05d}.png")
            t1 = time.perf_counter()
            pap.compose_textured(img, lm, layer, g.ink)
            ts_comp.append((time.perf_counter() - t1) * 1000)
            ts_read.append((t1 - t0) * 1000)
        res[clip] = {"read_png_ms_mean": float(np.mean(ts_read)), "compose_ms_mean": float(np.mean(ts_comp)),
                     "compose_ms_p95": float(np.percentile(ts_comp, 95)), "compose_ms_max": float(max(ts_comp))}
    pap._PAPER_CACHE.clear()
    t0 = time.perf_counter()
    pap.render_paper(1080, 1922, st)
    res["render_paper_1080x1922_ms"] = (time.perf_counter() - t0) * 1000
    save("t4_cost.json", res)


def cmd_final() -> None:
    """Setelah `python -m rotoscope run samples/<klip>.mp4` (config + style default): klip ASLI work/clips + out/ (hanya dibaca).
    Hash hulu (frames / seg / depth / stable / contours) dibandingkan dengan `work/t404a/before_upstream.txt` lewat perintah shell yang sama
    dengan saat F1 dibuat (find | sort | xargs sha256sum | sha256sum), bukan di sini."""
    from rotoscope import export as ex
    import export_metrics as em2
    import stylize_metrics as sm

    st = load_style(STYLE_PATH)
    res: dict = {}
    for clip in CLIPS:
        work = ROOT / "work" / "clips" / clip
        meta = json.loads((work / "meta.json").read_text())
        n = int(meta["frame_count"])
        mp4 = ROOT / "out" / f"{clip}.mp4"
        bef = OUT / f"before_t403_{clip}.mp4"
        ref = before(clip)
        h = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((work / "strokes").glob("frame_*"))}
        probe = ex.probe_output(mp4)
        stroke_m = json.loads((work / "strokes" / "manifest.json").read_text())
        osz = stroke_m["output_size"]
        layer = pap.render_paper(osz["width"], osz["height"], st)
        ink = pap.hex_rgb(st.stroke.color)
        lm = pap.LevelMap(pap.hex_rgb(st.paper.color), ink)
        worst = {"psnr": 99.0, "mae": 0.0, "max_diff": 0, "bias": 0.0}
        for i in em2.sample_indices(n, step=10):
            png = em2.read_png_rgb(work / "strokes" / f"frame_{i:05d}.png")
            exp = pap.compose_textured(png, lm, layer, ink)
            r = em2.textured_report(exp, em2.decode_rgb(mp4, i), sm.recover_f(exp, layer.f32, ink))
            worst = {"psnr": min(worst["psnr"], r["psnr"]), "mae": max(worst["mae"], r["mae"]), "max_diff": max(worst["max_diff"], r["max_diff"]),
                     "bias": max(worst["bias"], max(abs(x) for x in r["paper_bias"]))}
        svg_dir = ROOT / "out" / "svg" / clip
        svg_same = all((svg_dir / f"frame_{i:05d}.svg").read_bytes() == (work / "strokes" / f"frame_{i:05d}.svg").read_bytes() for i in range(n))
        png_sizes = [p.stat().st_size for p in (work / "strokes").glob("frame_*.png")]
        svg_sizes = [p.stat().st_size for p in (work / "strokes").glob("frame_*.svg")]
        prev_svg80 = (OUT / f"before_t403_{clip}_frame_00080.svg").read_bytes()
        res[clip] = {"mp4_B": mp4.stat().st_size, "mp4_before_t403_B": bef.stat().st_size, "x_t403": mp4.stat().st_size / bef.stat().st_size,
                     "probe": {k: probe[k] for k in ("frames", "fps", "codec", "pix_fmt", "width", "height", "duration_s", "color")},
                     "strokes_files": len(ref), "strokes_identical_F1": sum(1 for k in ref if h.get(k) == ref[k]),
                     "out_svg_identical_to_strokes": svg_same,
                     "out_svg_frame80_equals_before_t403": (svg_dir / "frame_00080.svg").read_bytes() == prev_svg80,
                     "png_KiB_mean": float(np.mean(png_sizes)) / 1024, "svg_KiB_mean": float(np.mean(svg_sizes)) / 1024,
                     "metrics_worst_sampled_every10": worst, "thresholds": {"psnr_min": em2.PSNR_TEXTURED_MIN_DB, "mae_max": em2.INK_MAE_MAX,
                                                                            "max_diff": em2.INK_MAX_DIFF, "bias": em2.PAPER_BIAS_MAX}}
    save("t4_final.json", res)


def main() -> None:
    cmds = {"stage5": cmd_stage5, "oracle": cmd_oracle, "cost": cmd_cost, "final": cmd_final}
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name in (("stage5", "oracle", "cost") if which == "all" else [which]):     # "final" hanya setelah `run` nyata
        t0 = time.perf_counter()
        cmds[name]()
        print(f"[{name}] {time.perf_counter() - t0:.1f} s", flush=True)


if __name__ == "__main__":
    main()
