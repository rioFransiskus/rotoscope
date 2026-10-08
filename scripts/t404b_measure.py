"""ARSIP: studi T-404b; modul grain dihapus (tekstur garis ditolak, docs/04 "Hasil T-404b"). TIDAK BISA DIJALANKAN di repo ini: mengimpor
`rotoscope.grain`, `texture.*` dan `compose_textured(..., grain)` yang sudah dihapus. Tidak ada commit asal (Tahap 2–3 tidak pernah di-commit);
sumber lengkap disimpan di `work/t404b/grain_source/` (grain.py, test, patch perubahan berkas terlacak `tracked_changes_stage2-3.patch`; ter-ignore, lokal).
Hasil studi: `work/t404b/t3_*.json`, papan `work/t404b/nilai/`.

T-404b Tahap 3: angka. Semua membaca klip ASLI (work/clips/<klip>/strokes dst., hanya dibaca) dan menulis ke work/t404b/ + work/t404b_scratch/.
Subperintah: regress (mode none byte-identik F1), invariants (f' <= f, dukungan, massa tinta hilang, SEMUA frame kedua klip), determinism, cost, mp4 (data ambang)."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "src"))

import export_metrics as em  # noqa: E402
import grain_metrics as gm  # noqa: E402
import numpy as np  # noqa: E402
import stylize_metrics as sm  # noqa: E402

from rotoscope import export as ex  # noqa: E402
from rotoscope import grain as gr  # noqa: E402
from rotoscope import paper as pap  # noqa: E402
from rotoscope.config import load_pipeline, load_style  # noqa: E402
from rotoscope.stage_common import work_dir_overrides  # noqa: E402

OUT = ROOT / "work" / "t404b"
SCR = ROOT / "work" / "t404b_scratch"
STYLE_PATH = ROOT / "configs" / "styles" / "rough-sketch.yaml"
CLIPS = ("test_short", "test")
MODES = ("grain_paper", "grain_brush", "grain_noise")
STRENGTHS = (0.2, 0.35, 0.5)
INV_STRENGTHS = (0.2, 0.5)


def save(name: str, obj) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(obj, indent=1, ensure_ascii=False)[:6000], flush=True)


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def style_with(**ov):
    return load_style(STYLE_PATH, overrides=ov or None)


def clip_info(clip: str):
    work = ROOT / "work" / "clips" / clip
    sm_ = json.loads((work / "strokes" / "manifest.json").read_text())
    osz = sm_["output_size"]
    n = int(json.loads((work / "meta.json").read_text())["frame_count"])
    return work, (osz["width"], osz["height"]), n


def png_path(work: Path, i: int) -> Path:
    return work / "strokes" / f"frame_{i:05d}.png"


def cfg_for(work: Path, **ov):
    return load_pipeline(None, overrides=work_dir_overrides(work, {"paths.out_dir": str(SCR / "out"), **ov}))


# ── regresi ────────────────────────────────────────
def cmd_regress() -> None:
    """Export default (rough-sketch.yaml, texture.mode none) pada klip asli → out_dir scratch: MP4 sha256 == salinan F1 (before_t404a_*)."""
    res = {}
    for clip in CLIPS:
        work, size, n = clip_info(clip)
        cfg = cfg_for(work)
        st = style_with()
        assert st.texture.mode == "none"
        t0 = time.perf_counter()
        run = ex.run_export(cfg, restart=True, style=st, log=lambda m: None)
        wall = time.perf_counter() - t0
        mp4 = SCR / "out" / f"{clip}.mp4"
        svg_ok = all((SCR / "out" / "svg" / clip / f"frame_{i:05d}.svg").read_bytes() == (work / "strokes" / f"frame_{i:05d}.svg").read_bytes() for i in range(n))
        m = json.loads((SCR / "out" / f"{clip}.export.json").read_text(encoding="utf-8"))
        res[clip] = {"mp4_sha256": sha(mp4), "before_t404a_sha256": sha(OUT / f"before_t404a_{clip}.mp4"), "identical": sha(mp4) == sha(OUT / f"before_t404a_{clip}.mp4"),
                     "bytes": mp4.stat().st_size, "wall_s": round(wall, 1), "svg_copy_identical_to_strokes": svg_ok, "manifest_texture": m.get("texture"),
                     "frames": run["frames"]}
    save("t3_regress.json", res)


# ── invarian + massa tinta hilang ──────────────────
def cmd_invariants() -> None:
    res = {}
    for clip in CLIPS:
        work, size, n = clip_info(clip)
        st0 = style_with()
        layer = pap.render_paper(*size, st0)
        ink = pap.hex_rgb(st0.stroke.color)
        lm = pap.LevelMap(pap.hex_rgb(st0.paper.color), ink)
        maps = {}
        for mode in MODES:
            for s in STRENGTHS:
                maps[(mode, s)] = gr.build_line_texture(*size, style_with(**{"texture.mode": mode, "texture.grain_strength": s})).m
        gA =(np.float32(1.0) - maps[("grain_paper", 0.5)]) / np.float32(0.5)
        for s in STRENGTHS:
            maps[("A_tanda_sebaliknya", s)] = gr.m_from_g(np.float32(1.0) - gA, s)
        agg = {k: {"mass_f": 0.0, "mass_fm": 0.0, "per_frame_loss": []} for k in maps}
        inv = {(mode, s): {"max_excess": -1.0, "off_support_px": 0, "m_ok": True, "tol": 0.0, "frames": 0, "fail_frames": []} for mode in MODES for s in INV_STRENGTHS}
        t0 = time.perf_counter()
        for i in range(n):
            png = em.read_png_rgb(png_path(work, i))
            lv, _ = lm.levels_of(png)
            f = lv / np.float32(255)
            mass = float(f.sum(dtype=np.float64))
            for k, m in maps.items():
                fm = float((f * m).sum(dtype=np.float64))
                agg[k]["mass_f"] += mass
                agg[k]["mass_fm"] += fm
                agg[k]["per_frame_loss"].append(1 - fm / mass if mass > 0 else 0.0)
            for (mode, s), r in inv.items():
                m = maps[(mode, s)]
                comp = pap.compose_textured(png, lm, layer, ink, m)
                x = gm.grain_invariants(png, comp, lm, layer, ink, m)
                r["frames"] += 1
                r["max_excess"] = max(r["max_excess"], x["max_excess"])
                r["off_support_px"] += x["off_support_px"]
                r["m_ok"] = r["m_ok"] and x["m_ok"]
                r["tol"] = x["tol"]
                if not gm.invariants_pass(x):
                    r["fail_frames"].append(i)
        res[clip] = {"frames": n, "wall_s": round(time.perf_counter() - t0, 1),
                     "mass_lost_pct": {f"{k[0]}|{k[1]}": {"total": round(100 * (1 - v["mass_fm"] / v["mass_f"]), 2), "frame_min": round(100 * min(v["per_frame_loss"]), 2),
                                                          "frame_max": round(100 * max(v["per_frame_loss"]), 2)} for k, v in agg.items()},
                     "invariants": {f"{k[0]}|{k[1]}": {**v, "pass": v["max_excess"] <= v["tol"] and v["off_support_px"] == 0 and v["m_ok"] and not v["fail_frames"]}
                                    for k, v in inv.items()}}
    save("t3_invariants.json", res)


# ── determinisme ───────────────────────────────────
CODE = ("import sys, hashlib; sys.path.insert(0, r'{src}'); from rotoscope import grain; from rotoscope.config import load_style; "
        "import numpy as np; st = load_style(r'{style}', overrides={{'texture.mode': '{mode}'}}); "
        "m = grain.build_line_texture(1080, 1922, st).m; print(hashlib.sha256(np.ascontiguousarray(m).tobytes()).hexdigest())")


def cmd_determinism() -> None:
    res = {"maps_across_processes": {}, "mp4_window_twice": {}}
    for mode in MODES:
        hs = [subprocess.run([sys.executable, "-c", CODE.format(src=ROOT / "src", style=STYLE_PATH, mode=mode)], capture_output=True, text=True).stdout.strip()
              for _ in range(2)]
        res["maps_across_processes"][mode] = {"sha256": hs[0][:16], "identical": hs[0] == hs[1] and len(hs[0]) == 64}
    work, size, n = clip_info("test_short")
    for mode in MODES:
        shas = []
        for k in range(2):
            gr._GRAIN_CACHE.clear()
            pap._PAPER_CACHE.clear()
            cfg = cfg_for(work)
            ex.run_export(cfg, limit=12, start=73, style=style_with(**{"texture.mode": mode, "texture.grain_strength": 0.35}), log=lambda m: None)
            shas.append(sha(SCR / "out" / "test_short.preview_73-84.mp4"))
        res["mp4_window_twice"][mode] = {"sha256": shas[0][:16], "identical": shas[0] == shas[1]}
    save("t3_determinism.json", res)


# ── biaya ──────────────────────────────────────────
def cmd_cost() -> None:
    res = {}
    for mode in MODES:
        st = style_with(**{"texture.mode": mode})
        gr._GRAIN_CACHE.clear()
        tracemalloc.start()
        t0 = time.perf_counter()
        lay = gr.build_line_texture(1080, 1922, st)
        res[f"build_{mode}"] = {"s": round(time.perf_counter() - t0, 3), "peak_MiB": round(tracemalloc.get_traced_memory()[1] / 2**20, 1)}
        tracemalloc.stop()
        res[f"build_{mode}"]["cache_MiB"] = round(lay.m.nbytes / 2**20, 2)
    for clip in CLIPS:
        work, size, n = clip_info(clip)
        st0 = style_with()
        layer = pap.render_paper(*size, st0)
        ink = pap.hex_rgb(st0.stroke.color)
        lm = pap.LevelMap(pap.hex_rgb(st0.paper.color), ink)
        m = gr.build_line_texture(*size, style_with(**{"texture.mode": "grain_noise"})).m
        base, with_m = [], []
        for i in range(0, n, max(1, n // 40)):
            png = em.read_png_rgb(png_path(work, i))
            t = time.perf_counter(); pap.compose_textured(png, lm, layer, ink); base.append(time.perf_counter() - t)
            t = time.perf_counter(); pap.compose_textured(png, lm, layer, ink, m); with_m.append(time.perf_counter() - t)
        res[f"compose_{clip}_ms"] = {"base_mean": round(1000 * np.mean(base), 1), "with_m_mean": round(1000 * np.mean(with_m), 1),
                                     "extra_mean": round(1000 * (np.mean(with_m) - np.mean(base)), 2), "extra_p95_with_m": round(1000 * np.percentile(with_m, 95), 1)}
    save("t3_cost.json", res)


# ── data ambang MP4 ────────────────────────────────
def decode_many(mp4: Path, idx: list[int], size: tuple[int, int]) -> list[np.ndarray]:
    sel = "+".join(f"eq(n\\,{i})" for i in idx)
    vf = f"select='{sel}',scale=in_color_matrix=bt709:in_range=tv:out_color_matrix=bt709:out_range=pc,format=rgb24"
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", str(mp4), "-vf", vf, "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                       capture_output=True, check=True)
    a = np.frombuffer(p.stdout, np.uint8).reshape(-1, size[1], size[0], 3)
    assert len(a) == len(idx), (len(a), len(idx))
    return list(a)


class Src(ex.FrameSource):
    kind = "strokes"

    def __init__(self, work: Path, size, textured):
        self.s = ex.StrokesSource(work, size, textured)

    def check(self, names):
        return None

    def render(self, name):
        return self.s.render(name)


def cmd_mp4() -> None:
    res: dict = json.loads((OUT / "t3_mp4.json").read_text()) if (OUT / "t3_mp4.json").is_file() else {}
    mdir = SCR / "mp4"
    mdir.mkdir(exist_ok=True)
    cfg0 = load_pipeline(None)
    bg = np.array(ex.parse_hex(cfg0.export.background_color), np.uint8)
    variants = [("none", 0.0)] + [(m, s) for m in MODES for s in STRENGTHS]
    for clip in CLIPS:
        work, size, n = clip_info(clip)
        names = [f"frame_{i:05d}" for i in range(n)]
        idx = em.sample_indices(n, 10)
        st0 = style_with()
        ink = pap.hex_rgb(st0.stroke.color)
        lm = pap.LevelMap(pap.hex_rgb(st0.paper.color), ink)
        pngs = {i: em.read_png_rgb(png_path(work, i)) for i in idx}
        fs = {i: lm.levels_of(pngs[i])[0] / np.float32(255) for i in idx}
        fps = float(json.loads((work / "meta.json").read_text())["target_fps"])
        for mode, s in variants:
            st = style_with(**({"texture.mode": mode, "texture.grain_strength": s} if mode != "none" else {}))
            tp = ex.make_textured(work, st, size)
            refs = {i: tp(pngs[i]) for i in idx}
            for crf in (18, 23):
                key = f"{clip}|{mode}|{s}|crf{crf}"
                p = mdir / f"{clip}_{mode}_{s}_crf{crf}.mp4"
                p.unlink(missing_ok=True)
                ex.encode(Src(work, size, tp), names, size, bg, fps, crf, cfg0.export.preset, None, p)
                dec = decode_many(p, idx, size)
                reps = []
                for i, d in zip(idx, dec):
                    r = em.textured_report(refs[i], d, fs[i])
                    core = fs[i] > 0.5
                    diff = np.abs(refs[i].astype(int) - d.astype(int))
                    r["mae_core"] = float(diff[core].mean()) if core.any() else 0.0
                    reps.append(r)
                res[key] = {"bytes": p.stat().st_size, "psnr_min": round(min(r["psnr"] for r in reps), 3), "mae_max": round(max(r["mae"] for r in reps), 3),
                            "mae_core_max": round(max(r["mae_core"] for r in reps), 3), "maxdiff_max": max(r["max_diff"] for r in reps),
                            "bias_max": round(max(abs(b) for r in reps for b in r["paper_bias"]), 3)}
                print(key, res[key], flush=True)
                (OUT / "t3_mp4.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
                p.unlink()                                                    # hemat disk scratch; ukuran sudah tercatat
    save("t3_mp4.json", res)


COMMANDS = {"regress": cmd_regress, "invariants": cmd_invariants, "determinism": cmd_determinism, "cost": cmd_cost, "mp4": cmd_mp4}

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    for name in (list(COMMANDS) if sys.argv[1] == "all" else sys.argv[1:]):
        print(f"== {name}", flush=True)
        COMMANDS[name]()
