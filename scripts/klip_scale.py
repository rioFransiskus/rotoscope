"""Uji skala subjek (alat sekali pakai; HANYA membaca work/clips/<klip>; eksperimen hanya di SALINAN work/klip_scale_scratch/):

    python scripts/klip_scale.py table [klip ...]        # tabel skala (butir 4): default test Klip2 klip3 -> work/klip_scale/table.json
    python scripts/klip_scale.py experiment <klip> ...   # hitung ulang [3] dan [4] pada salinan dgn parameter piksel diskalakan s (butir 5)

Tanpa GPU. Kode produksi dan default tidak diubah: eksperimen memanggil stage lewat CLI dengan --config ke YAML di scratch."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "src"))

import reach_metrics as rm  # noqa: E402
import temporal_metrics as tm  # noqa: E402

from rotoscope import stabilize as stb  # noqa: E402
from rotoscope import vectorize as vec  # noqa: E402
from rotoscope.config import load_pipeline  # noqa: E402

CLIPS = ("test", "Klip2", "klip3")
OUT = ROOT / "work" / "klip_scale"
SCRATCH = ROOT / "work" / "klip_scale_scratch"
LIMB_MIN_AREA_PX = 30                # komponen lengan / kaki < ini (= island_min_px default) tidak dihitung lebarnya
S_SKIP_MAX = 1.2                     # s <= ini: ukuran subjek mirip, eksperimen tidak dijalankan
REF_WIDTH = 1080.0                   # render.output_width default -> px ref = px kerja x REF_WIDTH / lebar kerja
SRC = {"test": "test.mp4", "Klip2": "Klip2.mp4", "klip3": "klip3.mp4"}


def jload(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def q3(a) -> dict:
    a = np.asarray(a, float)
    if not len(a):
        return {"p5": None, "p50": None, "p95": None, "max": None}
    return {"p5": float(np.percentile(a, 5)), "p50": float(np.median(a)), "p95": float(np.percentile(a, 95)), "max": float(a.max())}


def load_maps(work: Path, cfg):
    clip = vec.load_clip(work)
    classes = jload(work / "seg" / "manifest.json")["classes"]
    lut = stb.class_group_lut(cfg.groups, classes)
    stable, raw = [], []
    for n in clip.names:
        stable.append(stb.read_groups(clip.stable_clip.groups_path(n)))
        cm = cv2.imdecode(np.fromfile(work / "seg" / "classmap" / n, np.uint8), cv2.IMREAD_UNCHANGED)
        raw.append(lut[cm])
    return clip, np.stack(stable), np.stack(raw)


def bbox_height(m: np.ndarray) -> int:
    rows = np.flatnonzero((m != 0).any(axis=1))
    return int(rows[-1] - rows[0] + 1) if len(rows) else 0


def limb_widths(g: np.ndarray, gids: list[int]) -> list[float]:
    """2 x nilai maks distance transform (L2) per komponen 8-arah grup; komponen < LIMB_MIN_AREA_PX dilewati."""
    out = []
    for gid in gids:
        m = (g == gid).astype(np.uint8)
        n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        for k in range(1, n):
            if st[k, cv2.CC_STAT_AREA] < LIMB_MIN_AREA_PX:
                continue
            comp = np.pad((lab == k).astype(np.uint8), 1)
            out.append(2.0 * float(cv2.distanceTransform(comp, cv2.DIST_L2, 5).max()))
    return out


def vec_metrics(work: Path, cfg) -> dict:
    """Metrik stage [4] atas contours/ di `work`: strok per tipe, frame tanpa oklusi, oklusi hair yang dibuang exclude_groups,
    ujung oklusi per kelas jarak (px ref), umur track, pop energy, invarian reach_metrics (komponen baru, run zona D di tengah /
    seluruh strok, ekstensi > batas) memakai parameter dari contours/manifest.json milik `work` itu sendiri."""
    clip = vec.load_clip(work)
    names = tuple(g for g, _ in cfg.groups)
    mv = jload(work / "contours" / "manifest.json")["vectorize"]
    cs = jload(work / "contours" / "clip_stats.json")
    th, tl = cs["t_high"], cs["t_low"]
    D, D_low = float(mv["depth_lines.min_dist_px"]), float(mv["depth_lines.min_dist_low_px"])
    types = ("silhouette", "silhouette_hole", "group_boundary", "occlusion")
    docs = [jload(work / "contours" / f"frame_{i:05d}.json") for i in range(len(clip.names))]
    per = {t: np.array([sum(s["type"] == t for s in d["strokes"]) for d in docs]) for t in types}
    hair_dropped, gaps, new_comp, mid_runs, whole_runs, ext_over, bad_frames = [], [], 0, 0, 0, 0, set()
    bound = rm.reach_bound_px(D, D_low) if D_low else 0.0
    for i, nme in enumerate(clip.names):
        gmap, depth = vec.load_frame_inputs(clip, nme, len(names))
        incl = vec.occlusion_strokes(gmap, depth, names, {**mv, "depth_lines.exclude_groups": []}, th, tl)[0]
        hair_dropped.append(sum(s["groups"][0] == "hair" for s in incl))
        others = vec.vectorize_gmap(gmap, names, mv)[0]
        kept = [s for s in docs[i]["strokes"] if s["type"] == "occlusion"]
        gaps += rm.end_gaps(kept, others, clip.width, clip.height)
        base = vec.occlusion_strokes(gmap, depth, names, {**mv, "depth_lines.min_dist_low_px": 0.0, "depth_lines.exclude_groups": []}, th, tl)[0]
        nc = rm.new_components(base, kept)
        runs = rm.zone_runs(kept, others, D)
        mid, whole = sum(r["kind"] == "middle" for r in runs), sum(r["kind"] == "whole" for r in runs)
        over = sum(r["length"] > bound + 1 for r in runs if r["kind"] == "end") if D_low else 0
        new_comp, mid_runs, whole_runs, ext_over = new_comp + nc, mid_runs + mid, whole_runs + whole, ext_over + over
        if nc or mid or whole or over:
            bad_frames.add(i)
    fl = tm.stroke_lengths_from_dir(work / "contours")
    pop = tm.pop_energy_by_type(fl, types)
    occ = per["occlusion"]
    return {"strokes_per_frame": {t: {**q3(per[t]), "mean": float(per[t].mean())} for t in types},
            "frames_without_occlusion": int((occ == 0).sum()), "frames_without_occlusion_pct": float(100 * (occ == 0).mean()),
            "occlusion_total": int(occ.sum()), "hair_occlusion_dropped_total": int(sum(hair_dropped)),
            "end_gap_hist_px_ref": rm.gap_histogram(gaps, scale=REF_WIDTH / clip.width), "n_open_ends": len(gaps),
            "track_age_mean_median": {t: [pop["types"][t]["age_mean"], pop["types"][t]["age_median"]] for t in types},
            "pop_energy_total": pop["total_pop_energy_mean"], "pop_energy_by_type": {t: pop["types"][t]["pop_energy_mean"] for t in types},
            "pop_share_pct": {t: pop["types"][t]["pop_share_pct"] for t in types},
            "occlusion_new_id_per_stroke_frame": pop["types"]["occlusion"]["id_new_per_stroke_frame"],
            "invariants": {"new_components": new_comp, "zone_runs_middle": mid_runs, "zone_runs_whole": whole_runs,
                           "extension_over_bound": ext_over, "frames_with_violation": sorted(bad_frames),
                           "new_crossings": "n/a di [4] (metrik persilangan jitter_metrics hidup di [5])"}}


def clip_table(work: Path, cfg) -> dict:
    clip, stable, raw = load_maps(work, cfg)
    names = tuple(g for g, _ in cfg.groups)
    gid = {n: i + 1 for i, n in enumerate(names)}
    n = len(clip.names)
    hw = stable.shape[1] * stable.shape[2]
    H, W = stable.shape[1:]
    cs = jload(work / "contours" / "clip_stats.json")
    mv = jload(work / "contours" / "manifest.json")["vectorize"]
    dp = {k.split(".", 1)[1]: v for k, v in mv.items() if k.startswith("depth_lines.")}
    fg = (stable != 0)
    bbox = np.array([bbox_height(s) for s in stable])
    arm_w, leg_w = [], []
    for s in stable:                                  # median lebar per frame (komponen), lalu median / p5 / p95 antar frame
        a = limb_widths(s, [gid["left_arm"], gid["right_arm"]])
        b = limb_widths(s, [gid["left_leg"], gid["right_leg"]])
        arm_w.append(np.median(a) if a else np.nan), leg_w.append(np.median(b) if b else np.nan)
    arm_w, leg_w = np.array(arm_w), np.array(leg_w)
    docs = [jload(work / "contours" / f"frame_{i:05d}.json") for i in range(n)]
    ref = REF_WIDTH / W
    occ_n = np.array([sum(s["type"] == "occlusion" for s in d["strokes"]) for d in docs])
    bnd_n = np.array([sum(s["type"] == "group_boundary" for s in d["strokes"]) for d in docs])
    occ_len = [ref * float(np.hypot(*np.diff(np.asarray(s["points"], float), axis=0).T).sum())
               for d in docs for s in d["strokes"] if s["type"] == "occlusion"]
    # (e) kaki kosong + piksel yang berubah dari mentah ke stabil per grup
    legs = {}
    lost, gained = {}, {}
    for nm in names:
        r, s = raw == gid[nm], stable == gid[nm]
        legs[nm] = {"empty_raw": int((r.sum(axis=(1, 2)) == 0).sum()), "empty_stable": int((s.sum(axis=(1, 2)) == 0).sum())}
        lo = (r & ~s).sum(axis=(1, 2)).astype(float)
        ga = (s & ~r).sum(axis=(1, 2)).astype(float)
        rs = np.maximum(r.sum(axis=(1, 2)), 1)
        lost[nm] = {**q3(lo), "frac_of_raw_p50": float(np.median(lo / rs)), "frac_of_raw_max": float((lo / rs).max())}
        gained[nm] = q3(ga)
    fr = [json.loads(x) for x in (work / "stable" / "frames.jsonl").read_text(encoding="utf-8").splitlines()]
    fr = [r for r in fr if r.get("event") == "frame"]
    island = np.array([r["island_changed_px"] for r in fr], float)
    mode = np.array([r["mode_changed_px"] for r in fr], float)
    # (f) disparity <= 0, (g) zona D / erosi
    d0, elig, elig_er, elig_nohair, er_only = [], [], [], [], []
    for i, nme in enumerate(clip.names):
        d = np.load(work / "depth" / nme.replace(".png", ".npy")).astype(np.float32)
        m = fg[i]
        d0.append(float((d[m] <= 0).mean()))
        g = stable[i]
        e = (g != 0) & (vec.boundary_distance(g) >= dp["min_dist_px"])
        inner = vec.inner_region(g, dp["erode_px"])
        nfg = max(int(m.sum()), 1)
        elig.append(e.sum() / nfg), elig_er.append((e & inner).sum() / nfg), er_only.append(inner.sum() / nfg)
        elig_nohair.append((e & inner & (g != gid["hair"])).sum() / nfg)
    return {
        "n_frames": n, "work_size": [W, H], "ref_scale": ref,
        "a_fg_fraction_pct": q3(100 * fg.sum(axis=(1, 2)) / hw),
        "b_bbox_height_px": q3(bbox), "b_bbox_height_rel": q3(bbox / H), "b_bbox_height_median_px": float(np.median(bbox)),
        "c_arm_width_px": q3(arm_w[~np.isnan(arm_w)]), "c_leg_width_px": q3(leg_w[~np.isnan(leg_w)]),
        "c_frames_without_leg_component": int(np.isnan(leg_w).sum()),
        "d_occlusion_per_frame": {**q3(occ_n), "mean": float(occ_n.mean())}, "d_frames_without_occlusion_pct": float(100 * (occ_n == 0).mean()),
        "d_group_boundary_per_frame": {**q3(bnd_n), "mean": float(bnd_n.mean())}, "d_occlusion_length_px_ref": q3(occ_len),
        "d_occlusion_strokes_total": int(occ_n.sum()),
        "e_legs": legs, "e_lost_raw_to_stable_px": lost, "e_gained_raw_to_stable_px": gained,
        "e_island_changed_px_fg_total": q3(island), "e_mode_changed_px_fg_total": q3(mode),
        "f_fg_disparity_le_0_fraction": q3(d0), "f_t_high": cs["t_high"], "f_t_low": cs["t_low"],
        "g_eligible_dist_ge_D_pct_of_fg": q3(100 * np.array(elig)), "g_eroded_region_pct_of_fg": q3(100 * np.array(er_only)),
        "g_eligible_and_eroded_pct_of_fg": q3(100 * np.array(elig_er)), "g_eligible_eroded_excl_hair_pct_of_fg": q3(100 * np.array(elig_nohair)),
        "params": {"min_dist_px": dp["min_dist_px"], "erode_px": dp["erode_px"]},
        "vectorize": vec_metrics(work, cfg),
    }


def cmd_table(names: list[str]) -> None:
    cfg = load_pipeline()
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    for nm in names or CLIPS:
        res[nm] = clip_table(ROOT / "work" / "clips" / nm, cfg)
        print(nm, "ok", flush=True)
    path = OUT / "table.json"
    old = jload(path) if path.is_file() else {}
    old.update(res)
    path.write_text(json.dumps(old, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: {"bbox_med": v["b_bbox_height_median_px"], "fg_pct_p50": v["a_fg_fraction_pct"]["p50"]} for k, v in old.items()}))


def odd_round(x: float) -> int:
    """Bilangan ganjil >= 1 terdekat."""
    return max(1, 2 * int(np.floor((x - 1.0) / 2.0 + 0.5)) + 1)


def scaled_params(f: float) -> list[tuple[str, float, float]]:
    """(kunci config, nilai default, nilai skala). f = tinggi bbox klip / tinggi bbox test (satu dimensi); area x f^2.
    Batas config: erode_px / mode_k ganjil >= 1; line_min_px / min_stroke_px >= 1; D_low 0 atau 1 <= D_low < D."""
    c = load_pipeline()
    v, dl = c.vectorize, c.vectorize.depth_lines
    d = dl.min_dist_px * f
    rows = [("stabilize.island_min_px", c.stabilize.island_min_px, max(0, round(c.stabilize.island_min_px * f * f))),
            ("stabilize.mode_k", c.stabilize.mode_k, odd_round(c.stabilize.mode_k * f)),
            ("vectorize.depth_lines.erode_px", dl.erode_px, odd_round(dl.erode_px * f)),
            ("vectorize.depth_lines.min_dist_px", dl.min_dist_px, d),
            ("vectorize.depth_lines.min_dist_low_px", dl.min_dist_low_px, min(max(1.0, dl.min_dist_low_px * f), d * 0.999)),
            ("vectorize.depth_lines.min_len_px", dl.min_len_px, dl.min_len_px * f),
            ("vectorize.depth_lines.blur_sigma", dl.blur_sigma, dl.blur_sigma * f),
            ("vectorize.line_min_px", v.line_min_px, max(1, round(v.line_min_px * f))),
            ("vectorize.min_stroke_px", v.min_stroke_px, max(1, round(v.min_stroke_px * f))),
            ("vectorize.min_region_area", v.min_region_area, max(0, round(v.min_region_area * f * f))),
            ("vectorize.min_hole_area", v.min_hole_area, max(0, round(v.min_hole_area * f * f)))]
    return rows


def yaml_for(rows) -> str:
    tree: dict = {}
    for key, _, val in rows:
        node = tree
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = val
    out: list[str] = []

    def emit(n: dict, ind: int) -> None:
        for k, v in n.items():
            if isinstance(v, dict):
                out.append(" " * ind + f"{k}:")
                emit(v, ind + 2)
            else:
                out.append(" " * ind + f"{k}: {v!r}")
    emit(tree, 0)
    return "\n".join(out) + "\n"


def run_stage(mod: str, cfgfile: Path, work: Path) -> str:
    r = subprocess.run([sys.executable, "-m", f"rotoscope.{mod}", "--config", str(cfgfile), "--work-dir", str(work), "--restart"],
                       cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise SystemExit(f"{mod} gagal (rc {r.returncode}):\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}")
    return r.stdout.strip().splitlines()[-1]


def cmd_experiment(names: list[str]) -> None:
    cfg = load_pipeline()
    tab = jload(OUT / "table.json")
    ref_h = tab["test"]["b_bbox_height_median_px"]
    for nm in names:
        s = ref_h / tab[nm]["b_bbox_height_median_px"]            # definisi instruksi: s = tinggi bbox test / tinggi bbox klip
        res: dict = {"clip": nm, "bbox_test_px": ref_h, "bbox_clip_px": tab[nm]["b_bbox_height_median_px"], "s": s}
        if s <= S_SKIP_MAX:
            res["skipped"] = f"s = {s:.3f} <= {S_SKIP_MAX}: ukuran subjek mirip / lebih besar dari test, eksperimen tidak dijalankan"
        else:
            f = 1.0 / s              # parameter px dikalikan 1/s = bbox_klip / bbox_test (subjek lebih kecil -> ambang lebih kecil)
            rows = scaled_params(f)
            res["factor_applied"] = f
            res["params"] = [{"key": k, "default": a, "scaled": b} for k, a, b in rows]
            work = SCRATCH / nm
            if work.exists():
                shutil.rmtree(work)
            work.mkdir(parents=True)
            src = ROOT / "work" / "clips" / nm
            for d in ("frames", "seg", "depth"):
                shutil.copytree(src / d, work / d)
            for fl in ("meta.json", "qc_report.json"):
                shutil.copyfile(src / fl, work / fl)
            cfgfile = SCRATCH / f"{nm}_scaled.yaml"
            cfgfile.write_text(yaml_for(rows), encoding="utf-8")
            res["stabilize"] = run_stage("stabilize", cfgfile, work)
            res["vectorize"] = run_stage("vectorize", cfgfile, work)
            res["table"] = clip_table(work, cfg)
        (OUT / f"experiment_{nm}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
        print(nm, "s =", round(s, 3), res.get("skipped", "selesai"), flush=True)


def read_png(path: Path):
    return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)


def label(img, text: str):
    im = img.copy()
    cv2.rectangle(im, (0, 0), (im.shape[1], 22), (255, 255, 255), -1)
    cv2.putText(im, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return im


def cmd_compare(name: str, windows: list[str], outdir: Path) -> None:
    """[hasil default | hasil skala-ekuivalen] untuk jendela frame `a-b` (video) + satu PNG per jendela (frame tengah).
    Stage [5] dijalankan (CPU) pada salinan scratch dengan style default; klip asli tidak disentuh."""
    work = SCRATCH / name
    if not (work / "contours" / "manifest.json").is_file():
        raise SystemExit(f"{work} belum ada — jalankan: experiment {name}")
    r = subprocess.run([sys.executable, "-m", "rotoscope.stylize", "--work-dir", str(work), "--restart"], cwd=ROOT,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise SystemExit(f"stylize gagal:\n{r.stdout[-1000:]}\n{r.stderr[-1000:]}")
    outdir.mkdir(parents=True, exist_ok=True)
    for w in windows:
        a, b = (int(x) for x in w.split("-"))
        pipe = None
        fn = outdir / f"C_{name}_default_vs_skala_f{a}-{b}.mp4"
        for i in range(a, b + 1):
            f = f"frame_{i:05d}.png"
            left = label(read_png(ROOT / "work" / "clips" / name / "strokes" / f), f"default f{i}")
            right = label(read_png(work / "strokes" / f), "skala-ekuivalen")
            h = 480
            rs = lambda im: cv2.resize(im, (round(im.shape[1] * h / im.shape[0]) // 2 * 2, h), interpolation=cv2.INTER_AREA)  # noqa: E731
            fr = np.hstack([rs(left), rs(right)])
            if i == (a + b) // 2:
                ok, buf = cv2.imencode(".png", fr)
                (outdir / f"C_{name}_default_vs_skala_f{i:03d}.png").write_bytes(buf.tobytes())
            if pipe is None:
                pipe = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{fr.shape[1]}x{fr.shape[0]}",
                                         "-r", "24", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(fn)], stdin=subprocess.PIPE)
            pipe.stdin.write(np.ascontiguousarray(fr).tobytes())
        pipe.stdin.close()
        pipe.wait()
        print(fn.name, flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "table":
        cmd_table(sys.argv[2:])
    elif cmd == "experiment":
        cmd_experiment(sys.argv[2:])
    elif cmd == "compare":                      # compare <klip> <a-b> [<a-b> ...] -> work/klip3_review/ (ter-ignore; bahan Rio)
        cmd_compare(sys.argv[2], sys.argv[3:], ROOT / "work" / "klip3_review")
    else:
        sys.exit(__doc__)
