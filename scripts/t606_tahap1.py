"""T-606 Tahap 1 (alat sekali pakai, F3): ukur data H di klip3, prototipe M2-H di scratch, metrik, bahan penilaian. CPU saja.

    python scripts/t606_tahap1.py ukur       sebaran r per komponen, komponen gabungan per frame, lubang masker, jarak tepi seg di
                                             luar masker sepanjang batas target (sinar normal), QC [2] pada classmap bermasker,
                                             evaluasi kisi parameter -> work/t606/ukur.json
    python scripts/t606_tahap1.py m2 [--varian V0s,M2H]
                                             salinan scratch klip3 PENUH (182 frame) di work/t606_scratch/<varian>/: V0s = kontrol
                                             tanpa masker (harus byte-identik dengan produksi), M2H = seg/probs + classmap f85–f125
                                             dibatasi H (parameter usulan) -> [3] stabilize -> [4] vectorize (klip penuh) ->
                                             [5] stylize + [6] export (jendela f85–f125, out_dir scratch)
    python scripts/t606_tahap1.py metrik [--varian M2H,M1]
                                             metrik varian vs V0 (produksi): strok / siluet per frame, kelahiran strok, garis di
                                             dalam badan, piksel target hilang / sisa orang lain, kebocoran ke frame di luar adegan
                                             (ambang klip [4], track_id) -> work/t606/metrik.json
    python scripts/t606_tahap1.py bahan [--m1]
                                             video berdampingan foto | V0 | M2-H (+ M1 bila --m1), normal + lambat 0,5x ->
                                             work/t606/nilai/
    python scripts/t606_tahap1.py m2 --varian M2H_d8,M2H_d12,M2H_d20
                                             Tahap 1c: M2-H dengan d lain (M2H = d 4 acuan; parameter lain = PROPOSED)
    python scripts/t606_tahap1.py metrik --varian M2H,M2H_d8,M2H_d12,M2H_d20 --out metrik_1c.json
    python scripts/t606_tahap1.py tepi       Tahap 1c: piksel seg asli target terpotong / orang lain ikut per d (4, 8, 12, 20)
                                             + grup / sisi terbanyak -> work/t606/tepi_1c.json
    python scripts/t606_tahap1.py bahan1c    Tahap 1c: video foto + tepi masker | d4 | d8 | d12 (label besar), normal + lambat
    python scripts/t606_tahap1.py m1cek      baca-saja: verifikasi run M1 Rio (langkah, seg per frame, NaN, VRAM, QC, keluaran)
                                             + analisis tambahan (cakupan masker / cincin, label grup seg M1 vs asli, sumber
                                             pergeseran T_high [4], strok non-oklusi beda di luar adegan) -> work/t606/m1_cek.json

Hanya MEMBACA work/clips/klip3/ (frames, seg, depth, qc_report, isolate, target, stable, contours, strokes); menulis hanya ke
work/t606_scratch/ dan work/t606/. Salinan scratch: frames/ + depth/ + seg yang tidak diubah di-HARD-LINK (stage menulis atomik
lewat berkas sementara + rename, jadi berkas sumber tidak pernah ditulis); seg bermasker = berkas baru.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from rotoscope import isolate as iso  # noqa: E402
from rotoscope import segment as seg  # noqa: E402
from rotoscope import stabilize as stb  # noqa: E402
from rotoscope.config import load_class_names, load_pipeline  # noqa: E402
from rotoscope.ingest import FRAME_PATTERN  # noqa: E402
from rotoscope.stage_common import clip_identity  # noqa: E402

import t606_h as hh  # noqa: E402

CLIP = "klip3"
SRC = ROOT / "work" / "clips" / CLIP
SCR = ROOT / "work" / "t606_scratch"
OUT = ROOT / "work" / "t606"
NILAI = OUT / "nilai"
CFG_PATH = ROOT / "configs" / "default.yaml"
SCENE = (85, 125)                     # adegan berklik klip3 (isolate/manifest.json jobs[0].shot; dicek saat jalan)

# ── Parameter USULAN (dari `ukur`, docs/04 "Rancangan T-606 Tahap 1"; BUKAN keputusan) ──
#   r_keep 0,90 / r_drop 0,10: tengah celah sebaran r (komponen r 0,04 → 0,37 dan 0,64 → 0,96 kosong); d 4 = p95 jarak tepi seg
#   di luar M_f (sinar normal, komponen gabungan); semua lubang diisi; potongan < 30 px (= stabilize.island_min_px) di M_f dibuang.
PROPOSED = hh.HParams(r_keep=0.90, r_drop=0.10, d_px=4, hole_max_px=None, piece_min_px=30)

# Pengukuran (bukan parameter pipeline)
RAY_MAX_PX = 40                       # sinar normal keluar dari batas M_f: lebih jauh = "menyatu / bersambung" (disensor)
RAY_SIGMA = 2.0                       # σ Gaussian masker untuk arah normal
GRID_R_KEEP = (0.80, 0.85, 0.90, 0.95)
GRID_R_DROP = (0.02, 0.05, 0.10)
GRID_D = (0, 2, 4, 6, 8, 10)
INTERIOR_ERODE_PX = 8                 # wilayah "di dalam badan" = M_f dierosi 8 px (metrik garis palsu)


def jsave(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")


def pct(v, ps=(0, 5, 25, 50, 75, 95, 99, 100)) -> dict | None:
    a = np.asarray(v, np.float64)
    if a.size == 0:
        return None
    return {f"p{p}": round(float(np.percentile(a, p)), 4) for p in ps} | {"n": int(a.size), "mean": round(float(a.mean()), 4)}


def stem(i: int) -> str:
    return Path(FRAME_PATTERN % i).stem


# ── Masukan ────────────────────────────────────────
class Ctx:
    def __init__(self) -> None:
        self.cfg = load_pipeline(CFG_PATH, overrides={"paths.work_dir": str(SRC)})
        self.classes = load_class_names()
        self.lut = stb.class_group_lut(self.cfg.groups, self.classes)
        self.n_groups = len(self.cfg.groups)
        self.sclip = stb.load_clip(SRC)
        self.iclip = iso.load_clip(SRC)

    def mask_state(self) -> dict:
        """Aturan keputusan no. 6: masker dipakai hanya bila kunci manifest [1b] = titik + frame SEKARANG (fungsi stage [1b])."""
        frames_sha, per = iso.frames_fingerprint(self.iclip)
        key = iso.build_key(self.cfg, hashlib.sha256(self.iclip.points_path.read_bytes()).hexdigest(), frames_sha)
        state, reasons = iso.existing_state(self.iclip, key, clip_identity(SRC))
        m = iso.read_manifest(SRC / iso.ISOLATE_DIRNAME / iso.MANIFEST_FILENAME) or {}
        rows = iso.read_frame_rows(SRC / iso.ISOLATE_DIRNAME / iso.FRAMES_LOG_FILENAME)
        return {"state": state, "reasons": reasons, "key_sha256": iso.canonical_sha256(key),
                "manifest_key_sha256": m.get("key_sha256"), "shots": [j["shot"] for j in m.get("jobs", [])],
                "status": {r["index"]: r["status"] for r in rows}}

    def seg_inputs(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        name = FRAME_PATTERN % i
        pr = seg.read_probs(self.sclip.seg_clip.probs_path(name))
        cm = seg.read_classmap(self.sclip.seg_clip.classmap_path(name))
        if pr is None or cm is None:
            raise SystemExit(f"seg {name} tidak terbaca")
        return pr, cm

    def mask(self, i: int) -> np.ndarray | None:
        lb = iso.read_label(iso.label_path(SRC / iso.ISOLATE_DIRNAME, i))
        return None if lb is None else lb != 0

    def fg(self, pr: np.ndarray, cm: np.ndarray) -> np.ndarray:
        return hh.raw_groups(pr, cm, self.lut, self.n_groups) != 0


def scene_frames(ctx: Ctx, ms: dict) -> list[int]:
    if ms["state"] != "ok":
        raise SystemExit(f"masker [1b] klip3 TIDAK berlaku ({ms['state']}: {ms['reasons']}) — keputusan no. 6: tidak dipakai")
    if [list(s) for s in ms["shots"]] != [list(SCENE)]:
        raise SystemExit(f"adegan berklik {ms['shots']} ≠ {list(SCENE)} (konstanta skrip)")
    return list(range(SCENE[0], SCENE[1] + 1))


# ── Sinar normal: jarak tepi seg di luar masker ────
CROSS = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))


def overshoot_rays(m_f: np.ndarray, fg: np.ndarray, lab: np.ndarray, l_max: int = RAY_MAX_PX, sigma: float = RAY_SIGMA,
                   record: bool = False) -> dict:
    """Untuk tiap piksel batas M_f (piksel M_f dengan tetangga 4-arah di luar): sinar sepanjang normal keluar (−∇ masker halus).
    Kategori: `mask_lebih` (F salah di piksel batas: masker melewati seg), `ends` (F berakhir di luar M_f setelah v langkah),
    `reenter` (sinar masuk M_f lagi sebelum F berakhir — celah antar bagian target), `edge` (keluar gambar), `censored`
    (F berlanjut ≥ l_max px: menyatu dengan orang lain / bagian seg panjang). Komponen = label F di piksel batas."""
    h, w = m_f.shape
    b = m_f & ~(cv2.erode(m_f.astype(np.uint8), CROSS, borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0)
    ys, xs = np.nonzero(b)
    sm = cv2.GaussianBlur(m_f.astype(np.float32), (0, 0), sigma)
    gy, gx = np.gradient(sm)
    nx, ny = -gx[ys, xs], -gy[ys, xs]
    nr = np.hypot(nx, ny)
    ok = nr > 1e-4
    nx, ny = np.where(ok, nx / np.maximum(nr, 1e-9), 0.0), np.where(ok, ny / np.maximum(nr, 1e-9), 0.0)
    n = ys.size
    cat = np.full(n, "", object)
    val = np.full(n, -1, np.int32)
    comp = lab[ys, xs]
    f_at_b = fg[ys, xs]
    cat[~ok] = "no_normal"
    cat[ok & ~f_at_b] = "mask_lebih"
    alive = ok & f_at_b
    seen_out = np.zeros(n, bool)
    cnt = np.zeros(n, np.int32)
    rec = np.full((n, l_max), -1, np.int64) if record else None   # indeks linear piksel F di luar M_f per langkah terhitung
    for k in range(1, 4 * l_max):
        if not alive.any():
            break
        px = np.rint(xs + k * nx).astype(np.int64)
        py = np.rint(ys + k * ny).astype(np.int64)
        edge = alive & ((px < 0) | (px >= w) | (py < 0) | (py >= h))
        cat[edge], alive = "edge", alive & ~edge
        pxc, pyc = np.clip(px, 0, w - 1), np.clip(py, 0, h - 1)
        in_m = m_f[pyc, pxc]
        f = fg[pyc, pxc]
        re = alive & in_m & seen_out
        cat[re], alive = "reenter", alive & ~re
        out = alive & ~in_m
        end = out & ~f
        cat[end], val[end] = "ends", cnt[end]
        alive &= ~end
        step = alive & ~in_m & f
        if rec is not None:
            idx = np.flatnonzero(step)
            rec[idx, cnt[idx]] = (pyc * w + pxc)[idx]
        cnt[step] += 1
        seen_out |= out
        cen = alive & (cnt >= l_max)
        cat[cen], alive = "censored", alive & ~cen
    cat[alive] = "censored"
    return {"cat": cat, "val": val, "comp": comp, "px": rec}


# ── ukur ───────────────────────────────────────────
def cmd_ukur() -> None:
    t0 = time.perf_counter()
    ctx = Ctx()
    ms = ctx.mask_state()
    frames = scene_frames(ctx, ms)
    print(f"masker [1b]: {ms['state']} (kunci {ms['key_sha256'][:12]} = manifest {str(ms['manifest_key_sha256'])[:12]}); "
          f"adegan {SCENE}; status {Counter(ms['status'][i] for i in frames)}")
    probe = hh.HParams(r_keep=2.0, r_drop=-1.0, d_px=0)          # semua "potong": hanya untuk r per komponen
    per_frame, comps_all, holes_all, rays_rows = [], [], [], []
    cache = {}
    for i in frames:
        pr, cm = ctx.seg_inputs(i)
        fg = ctx.fg(pr, cm)
        m = ctx.mask(i)
        res = hh.apply_h(fg, m, probe)
        m_f, hole_areas, hole_lab = hh.fill_holes(m, None)
        hole_rows = []
        for k, a in enumerate(hole_areas, 1):
            sel = hole_lab == k
            cov = float(fg[sel].mean())
            hole_rows.append({"frame": i, "area": a, "fg_cover": round(cov, 4)})
        holes_all += hole_rows
        n, lab, _, _ = cv2.connectedComponentsWithStats(fg.astype(np.uint8), connectivity=8)
        rays = overshoot_rays(m_f, fg, lab)
        for c in res["comps"]:
            c["frame"] = i
        comps_all += res["comps"]
        r_of = {c["label"]: c["r"] for c in res["comps"]}
        rays_rows.append({"frame": i, "cat": rays["cat"], "val": rays["val"],
                          "r": np.array([r_of.get(int(c), -1.0) for c in rays["comp"]])})
        per_frame.append({"frame": i, "mask_px": int(m.sum()), "mask_filled_px": int(m_f.sum()), "fg_px": int(fg.sum()),
                          "mask_in_fg": round(float(fg[m].mean()), 4) if m.any() else None,
                          "n_comps": len(res["comps"]), "n_holes": len(hole_areas), "holes_px": int(sum(hole_areas)),
                          "big_blobs_raw_classmap": seg.big_blob_count(cm != 0, ctx.cfg.qc.blob_min)})
        cache[i] = (fg, m, cm)
    # sebaran r
    rs = [c["r"] for c in comps_all]
    big = [c for c in comps_all if c["area"] >= ctx.cfg.stabilize.island_min_px]
    hist_edges = [0, 1e-9, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 1.0 + 1e-9]
    def hist(values, weights=None):
        h, _ = np.histogram(values, bins=hist_edges, weights=weights)
        return [{"r_lo": hist_edges[k], "r_hi": hist_edges[k + 1], "n": round(float(h[k]), 1)} for k in range(len(h))]
    r_info = {"n_comps": len(comps_all), "n_comps_ge_island": len(big),
              "hist_count_all": hist(rs), "hist_count_ge_island": hist([c["r"] for c in big]),
              "hist_area_px": hist(rs, [c["area"] for c in comps_all]),
              "area_by_r_band": {},
              "comps_mid": sorted(({k: c[k] for k in ("frame", "area", "r", "bbox")} for c in comps_all if 0.02 < c["r"] < 0.98),
                                  key=lambda c: (c["frame"], -c["area"]))}
    for lo, hi, name in ((-1, 1e-9, "r=0"), (1e-9, 0.05, "0<r<=0.05"), (0.05, 0.95, "0.05<r<0.95"), (0.95, 2, "r>=0.95")):
        sel = [c for c in comps_all if lo < c["r"] <= hi] if name != "r=0" else [c for c in comps_all if c["r"] == 0]
        r_info["area_by_r_band"][name] = {"n": len(sel), "area": pct([c["area"] for c in sel], (0, 50, 95, 100)),
                                          "frames_with": len({c["frame"] for c in sel})}
    # lubang
    ha = [h["area"] for h in holes_all]
    hole_info = {"n_total": len(holes_all), "per_frame": pct([p["n_holes"] for p in per_frame], (0, 50, 100)),
                 "area": pct(ha), "fg_cover": pct([h["fg_cover"] for h in holes_all]),
                 "n_fg_cover_ge_50pct": sum(h["fg_cover"] >= 0.5 for h in holes_all),
                 "largest": sorted(holes_all, key=lambda h: -h["area"])[:10]}
    # sinar
    cats = np.concatenate([r["cat"] for r in rays_rows])
    vals = np.concatenate([r["val"] for r in rays_rows])
    rr = np.concatenate([r["r"] for r in rays_rows])
    def ray_stats(sel):
        c = Counter(cats[sel].tolist())
        e = vals[sel & (cats == "ends")]
        tot = int(sel.sum())
        return {"n_boundary_px": tot, "categories": {k: [v, round(v / max(tot, 1), 4)] for k, v in sorted(c.items())},
                "ends_px": pct(e, (50, 75, 90, 95, 99, 100)),
                "ends_gt": {str(d): round(float((e > d).mean()), 4) if e.size else None for d in GRID_D}}
    ray_info = {"ray_max_px": RAY_MAX_PX, "sigma": RAY_SIGMA, "all": ray_stats(np.ones(cats.size, bool)),
                "r_mid_0.05_0.95": ray_stats((rr > 0.05) & (rr < 0.95)),
                "r_ge_0.95": ray_stats(rr >= 0.95)}
    # kisi parameter (proksi piksel; render hanya untuk PROPOSED)
    grid = []
    for rk in GRID_R_KEEP:
        for rd in GRID_R_DROP:
            for d in GRID_D:
                p = hh.HParams(rk, rd, d, None, PROPOSED.piece_min_px)
                agg = Counter()
                for i in frames:
                    fg, m, _ = cache[i]
                    res = hh.apply_h(fg, m, p)
                    keep, m_f, m_d = res["keep"], res["m_f"], res["m_d"]
                    cls = Counter(c["class"] for c in res["comps"] if c["area"] >= ctx.cfg.stabilize.island_min_px)
                    agg["frames_with_potong"] += cls["potong"] > 0
                    agg["potong"] += cls["potong"]
                    agg["utuh"] += cls["utuh"]
                    agg["keep_px"] += int(keep.sum())
                    agg["keep_outside_md_px"] += int((keep & ~m_d).sum())
                    agg["fg_in_mf_lost_px"] += int((fg & m_f & ~keep).sum())
                    agg["fg_in_md_lost_px"] += int((fg & m_d & ~keep).sum())
                    agg["pieces_dropped"] += sum(not q["kept"] for q in res["pieces"])
                    agg["pieces_dropped_px"] += sum(q["area"] for q in res["pieces"] if not q["kept"])
                grid.append({"r_keep": rk, "r_drop": rd, "d": d, **{k: int(v) for k, v in agg.items()}})
    # parameter usulan: detail per frame + QC [2] di classmap bermasker
    prop_rows, masked_cms = [], {}
    for i in frames:
        fg, m, cm = cache[i]
        res = hh.apply_h(fg, m, PROPOSED)
        keep, m_f, m_d = res["keep"], res["m_f"], res["m_d"]
        cls = Counter(c["class"] for c in res["comps"])
        cls_big = Counter(c["class"] for c in res["comps"] if c["area"] >= ctx.cfg.stabilize.island_min_px)
        cm2 = cm.copy()
        cm2[res["remove"]] = 0
        masked_cms[i] = cm2
        prop_rows.append({"frame": i, "classes": dict(cls), "classes_ge_island": dict(cls_big),
                          "potong_r": [c["r"] for c in res["comps"] if c["class"] == "potong"],
                          "potong_area": [c["area"] for c in res["comps"] if c["class"] == "potong"],
                          "pieces": [{k: q[k] for k in ("area", "inter_mf", "kept")} for q in res["pieces"]],
                          "keep_px": int(keep.sum()), "remove_px": int(res["remove"].sum()),
                          "keep_outside_md_px": int((keep & ~m_d).sum()), "fg_in_mf_lost_px": int((fg & m_f & ~keep).sum()),
                          "big_blobs_masked_classmap": seg.big_blob_count(cm2 != 0, ctx.cfg.qc.blob_min)})
    q_orig = json.loads(ctx.sclip.seg_clip.qc_report_path.read_text(encoding="utf-8"))
    cms = (masked_cms.get(i, None) if i in masked_cms else seg.read_classmap(ctx.sclip.seg_clip.classmap_path(FRAME_PATTERN % i))
           for i in ctx.sclip.indices)
    rows = seg.compute_qc(cms, ctx.cfg.qc, ctx.sclip.names, ctx.sclip.indices)
    qc_info = {"fail_frames_original": [r["index"] for r in q_orig["frames"] if r["fail_reasons"]],
               "fail_frames_masked_scene": [r["index"] for r in rows if r["fail_reasons"]],
               "fail_reasons_masked_scene": {r["index"]: r["fail_reasons"] for r in rows if r["fail_reasons"]},
               "catatan": "QC [2] dihitung ulang (segment.compute_qc, CPU) dengan classmap f85–f125 dibatasi H; HANYA ukuran — "
                          "prototipe M2-H memakai qc_report.json asli"}
    res = {"clip": CLIP, "scene": list(SCENE), "mask_state": {k: v for k, v in ms.items() if k != "status"},
           "proposed": PROPOSED.to_json(), "island_min_px": ctx.cfg.stabilize.island_min_px,
           "per_frame": per_frame, "r": r_info, "holes": hole_info, "rays": ray_info, "grid": grid,
           "proposed_per_frame": prop_rows, "qc": qc_info, "wall_s": round(time.perf_counter() - t0, 1)}
    jsave(OUT / "ukur.json", res)
    print_ukur(res)


def print_ukur(res: dict) -> None:
    pf = res["per_frame"]
    print(f"frame {len(pf)}; masker px median {int(np.median([p['mask_px'] for p in pf]))}, isi lubang +"
          f"{int(np.median([p['mask_filled_px'] - p['mask_px'] for p in pf]))} px median; masker ∩ F median "
          f"{np.median([p['mask_in_fg'] for p in pf]):.4f}; komponen F per frame {pct([p['n_comps'] for p in pf], (0, 50, 100))}")
    print("r (jumlah komponen, semua):", [(b["r_lo"], b["n"]) for b in res["r"]["hist_count_all"] if b["n"]])
    print("r (komponen ≥ N px):", [(b["r_lo"], b["n"]) for b in res["r"]["hist_count_ge_island"] if b["n"]])
    for k, v in res["r"]["area_by_r_band"].items():
        print(f"  {k}: {v}")
    print("komponen 0,02 < r < 0,98:")
    for c in res["r"]["comps_mid"]:
        print(f"   f{c['frame']} luas {c['area']} r {c['r']} bbox {c['bbox']}")
    print("lubang:", {k: v for k, v in res["holes"].items() if k != "largest"})
    print("  terbesar:", res["holes"]["largest"][:6])
    for k in ("all", "r_mid_0.05_0.95", "r_ge_0.95"):
        print(f"sinar {k}: {res['rays'][k]}")
    print("kisi (r_keep, r_drop, d): frame dgn potong / potong / keep_luar_Md / fg_Mf_hilang / fg_Md_hilang / potongan dibuang (px)")
    for g in res["grid"]:
        print(f"  {g['r_keep']:.2f} {g['r_drop']:.2f} {g['d']:2d}: {g['frames_with_potong']:2d} / {g['potong']:3d} / "
              f"{g['keep_outside_md_px']:7d} / {g['fg_in_mf_lost_px']:6d} / {g['fg_in_md_lost_px']:6d} / "
              f"{g['pieces_dropped']} ({g['pieces_dropped_px']})")
    print("usulan per frame (kelas ≥ N px | r potong | potongan | big_blobs bermasker):")
    for r in res["proposed_per_frame"]:
        print(f"  f{r['frame']}: {r['classes_ge_island']} r={r['potong_r']} luas={r['potong_area']} "
              f"potongan={[(q['area'], q['inter_mf'], q['kept']) for q in r['pieces']]} bb={r['big_blobs_masked_classmap']} "
              f"keep_luar_Md={r['keep_outside_md_px']} hilang_Mf={r['fg_in_mf_lost_px']}")
    print("QC:", res["qc"]["fail_frames_original"], "→", res["qc"]["fail_reasons_masked_scene"])
    print(f"({res['wall_s']} s) -> {(OUT / 'ukur.json').relative_to(ROOT).as_posix()}")


# ── salinan scratch klip penuh + render ────────────
SEG_FILES = (("seg", "manifest.json"), ("seg", "frames.jsonl"), ("depth", "manifest.json"), ("depth", "frames.jsonl"))
STAGE_WINDOW = ["--from", str(SCENE[0]), "--limit", str(SCENE[1] - SCENE[0] + 1)]


def vdir(var: str) -> Path:
    return SCR / var / CLIP


def link(src: Path, dst: Path) -> None:
    """Hard link (berkas sumber tidak pernah ditulis: stage menulis atomik lewat berkas baru + rename); gagal → salin."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def file_sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def prepare_copy(var: str, fingerprint: dict, seg_for: "callable", log=print) -> tuple[Path, dict]:
    """Salinan klip3 PENUH di work/t606_scratch/<var>/klip3/: meta.json + qc_report.json disalin; frames/, depth/*.npy, dan seg
    yang tidak diganti di-hard-link; `seg_for(i)` → None (pakai seg asli) atau (probs, classmap, info) untuk frame i.
    Sidik sama dengan prep.json yang ada → dipakai ulang (stage [3]–[6] melanjutkan sendiri)."""
    d = vdir(var)
    prep_path = d.parent / "prep.json"
    fp = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode("utf-8")).hexdigest()
    if prep_path.is_file() and d.is_dir():
        old = json.loads(prep_path.read_text(encoding="utf-8"))
        if old.get("fingerprint") == fp:
            log(f"  {var}: salinan scratch sudah siap (sidik {fp[:12]}) — dipakai ulang")
            return d, old
    if d.exists():
        shutil.rmtree(d)
    sclip = stb.load_clip(SRC)
    d.mkdir(parents=True, exist_ok=True)
    for f in ("meta.json", "qc_report.json"):
        shutil.copyfile(SRC / f, d / f)
    for a, b in SEG_FILES:
        (d / a).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SRC / a / b, d / a / b)
    info = {"var": var, "fingerprint": fp, "fingerprint_obj": fingerprint, "frames": {}, "created_utc": time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    for i, name in zip(sclip.indices, sclip.names):
        link(SRC / "frames" / name, d / "frames" / name)
        link(sclip.depth_clip.depth_path(name), d / "depth" / sclip.depth_clip.depth_path(name).name)
        got = seg_for(i)
        pp, cp = sclip.seg_clip.probs_path(name), sclip.seg_clip.classmap_path(name)
        if got is None:
            link(pp, d / "seg" / "probs" / pp.name)
            link(cp, d / "seg" / "classmap" / cp.name)
        else:
            pr, cm, fi = got
            (d / "seg" / "probs").mkdir(parents=True, exist_ok=True)
            (d / "seg" / "classmap").mkdir(parents=True, exist_ok=True)
            seg.write_probs(d / "seg" / "probs" / pp.name, pr)
            seg.write_classmap(d / "seg" / "classmap" / cp.name, cm)
            info["frames"][i] = fi
    jsave(prep_path, info)
    log(f"  {var}: salinan scratch dibuat ({len(sclip.indices)} frame; seg diganti di {len(info['frames'])} frame)")
    return d, info


def cfg_path(var: str) -> Path:
    """configs/default.yaml dengan paths.out_dir → work/t606_scratch/<var>/out (export tidak pernah menulis ke out/)."""
    p = SCR / var / "cfg.yaml"
    text = CFG_PATH.read_text(encoding="utf-8")
    line = '  out_dir: "out"'
    if text.count(line) != 1:
        raise SystemExit("configs/default.yaml: baris out_dir tidak ditemukan tepat sekali")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text.replace(line, f'  out_dir: "{(SCR / var / "out").as_posix()}"'), encoding="utf-8")
    return p


def run_logged(cmd: list[str], log_path: Path) -> tuple[int, float]:
    t0 = time.perf_counter()
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(r.stdout + "\n--- stderr ---\n" + r.stderr, encoding="utf-8")
    return r.returncode, time.perf_counter() - t0


def render(var: str, log=print) -> dict:
    """[3] stabilize + [4] vectorize (klip PENUH: ambang [4] per klip + rantai track_id) → [5] stylize + [6] export (jendela
    adegan), semuanya dengan --work-dir scratch + cfg scratch."""
    d, cfg = vdir(var), cfg_path(var)
    py = [sys.executable, "-m"]
    steps = [("stabilize", py + ["rotoscope.stabilize", "--config", str(cfg), "--work-dir", str(d)]),
             ("vectorize", py + ["rotoscope.vectorize", "--config", str(cfg), "--work-dir", str(d)]),
             ("stylize", py + ["rotoscope.stylize", "--config", str(cfg), "--work-dir", str(d), *STAGE_WINDOW]),
             ("export", py + ["rotoscope.export", "--config", str(cfg), "--work-dir", str(d), *STAGE_WINDOW])]
    out = {}
    for name, cmd in steps:
        rc, dt = run_logged(cmd, SCR / "logs" / f"{var}_{name}.log")
        out[name] = {"rc": rc, "s": round(dt, 1)}
        log(f"  {var} [{name}] rc {rc}, {dt:.1f} s")
        if rc:
            raise SystemExit(f"{name} gagal — lihat work/t606_scratch/logs/{var}_{name}.log")
    return out


def h_seg_for(ctx: Ctx, frames: list[int], p: hh.HParams, base: "callable | None" = None):
    """seg_for untuk H: frame adegan → seg (asli, atau `base(i)` mis. seg M1) dibatasi H; frame lain → None / base."""
    fs = set(frames)

    def seg_for(i: int):
        src = base(i) if base is not None else None
        if i not in fs:
            return src
        pr, cm = (src[0], src[1]) if src is not None else ctx.seg_inputs(i)
        fg = ctx.fg(pr, cm)
        m = ctx.mask(i)
        res = hh.apply_h(fg, m, p)
        pr2, cm2 = hh.mask_probs(pr, cm, res["remove"])
        cls = Counter(c["class"] for c in res["comps"])
        return pr2, cm2, {"classes": dict(cls), "remove_px": int(res["remove"].sum()), "keep_px": int(res["keep"].sum()),
                          "fg_px": int(fg.sum())}
    return seg_for


def control_check(var: str, ctx: Ctx) -> dict:
    """V0s (tanpa masker) vs produksi: stable/ (182 frame) byte-identik, strok contours/ (182) sama, strokes/ PNG + SVG adegan
    byte-identik. Bukti bahwa salinan scratch + cara menjalankan stage tidak mengubah apa pun selain masker."""
    d = vdir(var)
    sc = ctx.sclip
    bad = Counter()
    for name in sc.names:
        st = Path(name).stem
        for sub, suf in (("stable/groups", ".png"), ("stable/depth_smooth", ".npy")):
            if file_sha(SRC / sub / (st + suf)) != file_sha(d / sub / (st + suf)):
                bad[sub] += 1
        a = json.loads((SRC / "contours" / (st + ".json")).read_text(encoding="utf-8"))
        b = json.loads((d / "contours" / (st + ".json")).read_text(encoding="utf-8"))
        if a["strokes"] != b["strokes"]:
            bad["contours.strokes"] += 1
        if a["source"] != b["source"]:
            bad["contours.source (cap waktu manifest)"] += 1
    for i in range(SCENE[0], SCENE[1] + 1):
        for suf in (".png", ".svg"):
            if file_sha(SRC / "strokes" / (stem(i) + suf)) != file_sha(d / "strokes" / (stem(i) + suf)):
                bad["strokes" + suf] += 1
    a = json.loads((SRC / "contours" / "clip_stats.json").read_text(encoding="utf-8"))
    b = json.loads((d / "contours" / "clip_stats.json").read_text(encoding="utf-8"))
    res = {"n_frames": len(sc.names), "diff_counts": dict(bad),
           "clip_stats_same_thresholds": [a["t_high"], a["t_low"]] == [b["t_high"], b["t_low"]]}
    res["ok"] = not any(k for k in bad if not k.startswith("contours.source")) and res["clip_stats_same_thresholds"]
    return res


D_VAR_RE = re.compile(r"M2H_d(\d+)")


def h_params_for(var: str) -> hh.HParams | None:
    """M2H = PROPOSED (d 4); M2H_d<N> = PROPOSED dengan d = N (Tahap 1c); selain itu None."""
    if var == "M2H":
        return PROPOSED
    m = D_VAR_RE.fullmatch(var)
    return dataclasses.replace(PROPOSED, d_px=int(m.group(1))) if m else None


def cmd_m2(variants: list[str]) -> None:
    ctx = Ctx()
    ms = ctx.mask_state()
    frames = scene_frames(ctx, ms)
    base_fp = {"src_seg_manifest": file_sha(SRC / "seg" / "manifest.json"), "qc": file_sha(SRC / "qc_report.json"),
               "isolate_key": ms["key_sha256"], "meta": file_sha(SRC / "meta.json")}
    res_path = OUT / "m2_run.json"
    res = json.loads(res_path.read_text(encoding="utf-8")) if res_path.is_file() else {}
    for var in variants:
        t0 = time.perf_counter()
        p = h_params_for(var)
        if var == "V0s":
            _, prep = prepare_copy(var, {**base_fp, "var": var}, lambda i: None)
        elif p is not None:
            _, prep = prepare_copy(var, {**base_fp, "var": var, "h": p.to_json(), "scene": list(SCENE)},
                                   h_seg_for(ctx, frames, p))
        else:
            raise SystemExit(f"varian {var!r} tidak dikenal (V0s, M2H, M2H_d<N>; M1 lewat scripts/t606_m1.py)")
        steps = render(var)
        r = {"steps": steps, "prep_frames": prep.get("frames"), "wall_s": round(time.perf_counter() - t0, 1)}
        if var == "V0s":
            r["control"] = control_check(var, ctx)
            print(f"  kontrol V0s vs produksi: {r['control']}")
        res[var] = r
        jsave(res_path, res)
    print(f"-> {res_path.relative_to(ROOT).as_posix()}")


# ── metrik varian vs V0 ────────────────────────────
NEW_DIST_PX = 4.0                     # strok varian "baru" bila median jarak titiknya ke strok V0 terdekat > 4 px (px kerja)
INSIDE_FRAC = 0.5                     # strok "di dalam badan" bila ≥ 50% titiknya di wilayah I
TYPES = ("silhouette", "silhouette_hole", "group_boundary", "occlusion")
CLOSED_TYPES = ("silhouette", "silhouette_hole")


def contours_doc(d: Path, i: int) -> dict:
    return json.loads((d / "contours" / (stem(i) + ".json")).read_text(encoding="utf-8"))


def pts_of(s: dict) -> np.ndarray:
    return np.asarray(s["points"], np.float64).reshape(-1, 2)


def raster(strokes: list[dict], shape: tuple[int, int]) -> np.ndarray:
    img = np.zeros(shape, np.uint8)
    for s in strokes:
        p = np.rint(pts_of(s) - 0.5).astype(np.int32)
        if len(p):
            cv2.polylines(img, [p.reshape(-1, 1, 2)], bool(s["closed"]), 255, 1)
    return img


def inside_len(p: np.ndarray, closed: bool, inside: np.ndarray) -> float:
    if len(p) < 2:
        return 0.0
    q = np.vstack([p, p[:1]]) if closed else p
    h, w = inside.shape
    ix = np.clip(np.rint(q[:, 0] - 0.5).astype(int), 0, w - 1)
    iy = np.clip(np.rint(q[:, 1] - 0.5).astype(int), 0, h - 1)
    a = inside[iy, ix]
    seg_in = a[:-1] & a[1:]
    return float(np.hypot(*np.diff(q, axis=0).T)[seg_in].sum())


def frame_metrics(doc: dict, prev_doc: dict | None, interior: np.ndarray, v0_dist: np.ndarray | None,
                  m_d: np.ndarray) -> dict:
    h, w = interior.shape
    by = Counter(s["type"] for s in doc["strokes"])
    out = {"n": len(doc["strokes"]), "by_type": {t: by.get(t, 0) for t in TYPES}}
    def on_target(s: dict) -> bool:                                       # < 50% titik di luar M_d
        p = pts_of(s)
        ix = np.clip(np.rint(p[:, 0] - 0.5).astype(int), 0, w - 1)
        iy = np.clip(np.rint(p[:, 1] - 0.5).astype(int), 0, h - 1)
        return bool(len(p)) and float((~m_d[iy, ix]).mean()) < INSIDE_FRAC

    out["n_on_target"] = sum(on_target(s) for s in doc["strokes"])
    if prev_doc is not None:
        prev = {(s["type"], s["track_id"]) for s in prev_doc["strokes"]}
        born = [s for s in doc["strokes"] if (s["type"], s["track_id"]) not in prev]
        out["births"] = len(born)
        out["births_on_target"] = sum(on_target(s) for s in born)
        out["births_by_type"] = {t: sum(s["type"] == t for s in born) for t in TYPES}
    ins = {t: {"n": 0, "len": 0.0} for t in TYPES}
    new = {t: {"n": 0, "len": 0.0} for t in TYPES}
    outside = {"n": 0, "len": 0.0}
    for s in doc["strokes"]:
        p = pts_of(s)
        ix = np.clip(np.rint(p[:, 0] - 0.5).astype(int), 0, w - 1)
        iy = np.clip(np.rint(p[:, 1] - 0.5).astype(int), 0, h - 1)
        if len(p) and float((~m_d[iy, ix]).mean()) >= INSIDE_FRAC:      # strok di luar target (orang lain / sisa)
            outside["n"] += 1
            outside["len"] += _len(s)
        frac = float(interior[iy, ix].mean()) if len(p) else 0.0
        if frac < INSIDE_FRAC:
            continue
        L = inside_len(p, bool(s["closed"]), interior)
        ins[s["type"]]["n"] += 1
        ins[s["type"]]["len"] += L
        if v0_dist is not None and float(np.median(v0_dist[iy, ix])) > NEW_DIST_PX:
            new[s["type"]]["n"] += 1
            new[s["type"]]["len"] += L
    out["inside"] = {t: {"n": v["n"], "len": round(v["len"], 1)} for t, v in ins.items()}
    out["outside_md"] = {"n": outside["n"], "len": round(outside["len"], 1)}
    if v0_dist is not None:
        out["new_inside"] = {t: {"n": v["n"], "len": round(v["len"], 1)} for t, v in new.items()}
    return out


def strip_ids(strokes: list[dict]) -> list[dict]:
    return [{k: v for k, v in s.items() if k != "track_id"} for s in strokes]


def variant_dir(var: str) -> Path:
    return SRC if var == "V0" else vdir(var)


def cmd_metrik(variants: list[str], out_name: str = "metrik.json") -> None:
    ctx = Ctx()
    ms = ctx.mask_state()
    frames = scene_frames(ctx, ms)
    shape = (ctx.sclip.height, ctx.sclip.width)
    masks = {}
    for i in frames:
        res = hh.apply_h(np.zeros(shape, bool), ctx.mask(i), PROPOSED)
        er = cv2.erode(res["m_f"].astype(np.uint8), cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * INTERIOR_ERODE_PX + 1, 2 * INTERIOR_ERODE_PX + 1))) > 0
        masks[i] = (res["m_f"], res["m_d"], er)
    v0 = SRC
    v0_docs = {i: contours_doc(v0, i) for i in frames}
    v0_dist = {i: cv2.distanceTransform(255 - raster(v0_docs[i]["strokes"], shape), cv2.DIST_L2, 3) for i in frames}
    out = {"scene": list(SCENE), "params": PROPOSED.to_json(), "interior_erode_px": INTERIOR_ERODE_PX,
           "new_dist_px": NEW_DIST_PX, "inside_frac": INSIDE_FRAC, "variants": {}}
    for var in ["V0", *variants]:
        d = variant_dir(var)
        if not (d / "contours").is_dir():
            print(f"  {var}: contours/ belum ada ({d}) — dilewati")
            continue
        rows, prev = {}, None
        for i in frames:
            doc = contours_doc(d, i)
            m_f, m_d, interior = masks[i]
            r = frame_metrics(doc, prev, interior, None if var == "V0" else v0_dist[i], m_d)
            g_var = stb.read_groups(d / "stable" / "groups" / (stem(i) + ".png")) != 0
            g_v0 = stb.read_groups(v0 / "stable" / "groups" / (stem(i) + ".png")) != 0
            r["px"] = {"fg": int(g_var.sum()), "in_mf": int((g_var & m_f).sum()),
                       "lost_in_mf_vs_v0": int((g_v0 & m_f & ~g_var).sum()),
                       "lost_in_ring_vs_v0": int((g_v0 & m_d & ~m_f & ~g_var).sum()),
                       "outside_md": int((g_var & ~m_d).sum()), "mf": int(m_f.sum())}
            rows[i] = r
            prev = doc
        agg = {"strokes_per_frame": round(float(np.mean([r["n"] for r in rows.values()])), 2),
               "silhouette_per_frame": round(float(np.mean([r["by_type"]["silhouette"] for r in rows.values()])), 3),
               "by_type_per_frame": {t: round(float(np.mean([r["by_type"][t] for r in rows.values()])), 3) for t in TYPES},
               "births_per_frame": round(float(np.mean([r["births"] for i, r in rows.items() if i != frames[0]])), 3),
               "strokes_on_target_per_frame": round(float(np.mean([r["n_on_target"] for r in rows.values()])), 2),
               "births_on_target_per_frame": round(float(np.mean([r["births_on_target"] for i, r in rows.items()
                                                                  if i != frames[0]])), 3),
               "births_by_type_per_frame": {t: round(float(np.mean([r["births_by_type"][t] for i, r in rows.items()
                                                                     if i != frames[0]])), 3) for t in TYPES},
               "inside_per_frame": {t: {"n": round(float(np.mean([r["inside"][t]["n"] for r in rows.values()])), 3),
                                        "len": round(float(np.mean([r["inside"][t]["len"] for r in rows.values()])), 1)}
                                    for t in TYPES},
               "px_per_frame": {k: round(float(np.mean([r["px"][k] for r in rows.values()])), 1) for k in rows[frames[0]]["px"]},
               "strokes_outside_md_per_frame": {
                   "n": round(float(np.mean([r["outside_md"]["n"] for r in rows.values()])), 3),
                   "len": round(float(np.mean([r["outside_md"]["len"] for r in rows.values()])), 1)},
               "frames_strokes_outside_md": [i for i, r in rows.items() if r["outside_md"]["n"]],
               "frames_outside_md_gt_0": [i for i, r in rows.items() if r["px"]["outside_md"] > 0]}
        if var != "V0":
            agg["new_inside_total"] = {t: {"n": sum(r["new_inside"][t]["n"] for r in rows.values()),
                                           "len": round(sum(r["new_inside"][t]["len"] for r in rows.values()), 1)} for t in TYPES}
            agg["frames_new_inside_closed"] = [i for i, r in rows.items()
                                               if any(r["new_inside"][t]["n"] for t in CLOSED_TYPES)]
            agg["frames_new_inside_any"] = [i for i, r in rows.items() if any(r["new_inside"][t]["n"] for t in TYPES)]
            # kebocoran ke frame di luar adegan (ambang klip [4] + rantai track_id); stable/ seharusnya identik
            leak = Counter()
            leak_frames = {"stable_beda": [], "strok_geometri_beda": [], "strok_selain_oklusi_beda": [],
                           "hanya_track_id_beda": []}
            occ_len = {"V0": 0.0, var: 0.0}
            for i, name in zip(ctx.sclip.indices, ctx.sclip.names):
                if SCENE[0] <= i <= SCENE[1]:
                    continue
                st = Path(name).stem
                if any(file_sha(v0 / sub / (st + suf)) != file_sha(d / sub / (st + suf))
                       for sub, suf in (("stable/groups", ".png"), ("stable/depth_smooth", ".npy"))):
                    leak_frames["stable_beda"].append(i)
                a, b = contours_doc(v0, i)["strokes"], contours_doc(d, i)["strokes"]
                if strip_ids([s for s in a if s["type"] != "occlusion"]) != strip_ids([s for s in b if s["type"] != "occlusion"]):
                    leak_frames["strok_selain_oklusi_beda"].append(i)
                if strip_ids(a) != strip_ids(b):
                    leak_frames["strok_geometri_beda"].append(i)
                    ca, cb = Counter(s["type"] for s in a), Counter(s["type"] for s in b)
                    for t in TYPES:
                        leak[f"{t}_n_selisih"] += abs(ca.get(t, 0) - cb.get(t, 0))
                    occ_len["V0"] += sum(_len(s) for s in a if s["type"] == "occlusion")
                    occ_len[var] += sum(_len(s) for s in b if s["type"] == "occlusion")
                elif a != b:
                    leak_frames["hanya_track_id_beda"].append(i)
            ca = json.loads((v0 / "contours" / "clip_stats.json").read_text(encoding="utf-8"))
            cb = json.loads((d / "contours" / "clip_stats.json").read_text(encoding="utf-8"))
            agg["leak_outside_scene"] = {"n_frames_outside": len(ctx.sclip.names) - len(frames),
                                         **{k: {"n": len(v), "frames": v} for k, v in leak_frames.items()},
                                         "type_count_abs_diff": dict(leak),
                                         "occlusion_len_in_changed_frames": {k: round(v, 1) for k, v in occ_len.items()},
                                         "clip_stats": {"V0": {"t_high": ca["t_high"], "t_low": ca["t_low"], "n_values": ca["n_values"]},
                                                        var: {"t_high": cb["t_high"], "t_low": cb["t_low"], "n_values": cb["n_values"]}}}
        out["variants"][var] = {"agg": agg, "rows": rows}
        print(f"{var}: " + json.dumps({k: v for k, v in agg.items() if k not in ("frames_outside_md_gt_0",)}, ensure_ascii=False))
    jsave(OUT / out_name, out)
    print(f"-> {(OUT / out_name).relative_to(ROOT).as_posix()}")


def _len(s: dict) -> float:
    p = pts_of(s)
    if len(p) < 2:
        return 0.0
    q = np.vstack([p, p[:1]]) if s["closed"] else p
    return float(np.hypot(*np.diff(q, axis=0).T).sum())


# ── bahan penilaian ────────────────────────────────
PANEL_H = 960
BAR_H = 40
FONT = cv2.FONT_HERSHEY_SIMPLEX
MASK_COLOR = (255, 200, 0)            # BGR: tepi masker target (M) di panel foto


def labeled(img: np.ndarray, title: str, sub: str) -> np.ndarray:
    bar = np.full((BAR_H, img.shape[1], 3), 28, np.uint8)
    cv2.putText(bar, title, (8, 17), FONT, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.putText(bar, sub, (8, 35), FONT, 0.45, (0, 230, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def fit(img: np.ndarray) -> np.ndarray:
    s = PANEL_H / img.shape[0]
    return cv2.resize(img, (round(img.shape[1] * s), PANEL_H), interpolation=cv2.INTER_AREA)


def photo_panel(ctx: Ctx, i: int) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(SRC / "frames" / (FRAME_PATTERN % i), np.uint8), cv2.IMREAD_COLOR)
    m = ctx.mask(i)
    if m is not None:
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, cs, -1, MASK_COLOR, 2, cv2.LINE_AA)
    return fit(img)


def strokes_panel(d: Path, i: int) -> np.ndarray:
    p = d / "strokes" / (stem(i) + ".png")
    img = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"{p} tidak terbaca")
    return fit(img)


def even(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    return img[: h - h % 2, : w - w % 2]


def write_video(path: Path, frames, fps: float) -> int:
    proc, n = None, 0
    for img in frames:
        if proc is None:
            proc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s",
                                     f"{img.shape[1]}x{img.shape[0]}", "-r", f"{fps:g}", "-i", "-", "-c:v", "libx264",
                                     "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", str(path)], stdin=subprocess.PIPE)
        proc.stdin.write(np.ascontiguousarray(img).tobytes())
        n += 1
    if proc is not None:
        proc.stdin.close()
        if proc.wait() != 0:
            raise SystemExit(f"ffmpeg gagal menulis {path}")
    return n


def cmd_bahan(with_m1: bool) -> None:
    ctx = Ctx()
    frames = scene_frames(ctx, ctx.mask_state())
    fps = float(json.loads((SRC / "meta.json").read_text(encoding="utf-8"))["target_fps"])
    cols = [("V0 (sekarang)", SRC), ("M2-H", vdir("M2H"))] + ([("M1", vdir("M1"))] if with_m1 else [])
    for _, d in cols:
        missing = [i for i in frames if not (d / "strokes" / (stem(i) + ".png")).is_file()]
        if missing:
            raise SystemExit(f"{d}/strokes belum lengkap (mis. f{missing[0]}) — jalankan render dulu")
    NILAI.mkdir(parents=True, exist_ok=True)
    name = "klip3_banding_" + "_".join(["V0", "M2H"] + (["M1"] if with_m1 else [])) + f"_f{SCENE[0]:03d}-{SCENE[1]:03d}"
    made = []
    for suffix, f in (("", fps), ("_lambat", fps / 2)):
        def gen():
            for k, i in enumerate(frames, 1):
                sub = f"f{i:03d}  t={i / fps:.2f} s  ({k}/{len(frames)})"
                ps = [labeled(photo_panel(ctx, i), "foto + tepi masker SAM", sub)]
                ps += [labeled(strokes_panel(d, i), title, sub) for title, d in cols]
                yield even(np.hstack(ps))
        n = write_video(NILAI / f"{name}{suffix}.mp4", gen(), f)
        made.append({"video": f"{name}{suffix}.mp4", "frames": [frames[0], frames[-1]], "n": n, "fps": f,
                     "panel": ["foto"] + [t for t, _ in cols]})
        print(f"  {NILAI.relative_to(ROOT).as_posix()}/{name}{suffix}.mp4: {n} frame, {f:g} fps")
    idx = OUT / "bahan.json"
    old = json.loads(idx.read_text(encoding="utf-8")) if idx.is_file() else {}
    old["m1" if with_m1 else "m2"] = made
    jsave(idx, old)


# ── m1cek: verifikasi run M1 (Rio) + analisis tambahan (baca-saja) ──
M1_SEG = SCR / "M1_seg" / CLIP


def frame_set(strokes: list[dict]) -> Counter:
    """Strok tanpa urutan titik / arah / titik awal: (tipe, closed, himpunan titik) — beda di sini = beda bentuk."""
    return Counter((s["type"], bool(s["closed"]), frozenset(tuple(p) for p in s["points"])) for s in strokes)


def cmd_m1cek() -> None:
    from rotoscope import vectorize as vec
    ctx = Ctx()
    frames = scene_frames(ctx, ctx.mask_state())
    out: dict = {}
    # A. run M1
    steps = json.loads((SCR / "M1_steps.json").read_text(encoding="utf-8"))
    rows = [json.loads(x) for x in (M1_SEG / "seg" / "frames.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    fr = [r for r in rows if r.get("event") == "frame"]
    tt = [r["total_s"] for r in fr]
    qc = json.loads((M1_SEG / "qc_report.json").read_text(encoding="utf-8"))["summary"]
    d = vdir("M1")
    out["run"] = {
        "steps": {k: {x: v.get(x) for x in ("done", "rc", "wall_s", "started_utc", "finished_utc")} for k, v in steps.items()},
        "seg_events": dict(Counter(r.get("event") for r in rows)),
        "seg_run_start": next(({k: v for k, v in r.items() if k != "event"} for r in rows if r.get("event") == "run_start"), None),
        "seg_run_end": next(({k: v for k, v in r.items() if k != "event"} for r in rows if r.get("event") == "run_end"), None),
        "seg_frames": len(fr), "seg_unique_frames": len({r["frame"] for r in fr}),
        "seg_frame_range": [min(r["index"] for r in fr), max(r["index"] for r in fr)],
        "finite_false": sum(r.get("finite") is False for r in fr), "finite_unknown": sum(r.get("finite") is None for r in fr),
        "total_s": {"first": tt[0], "p50": round(float(np.median(tt)), 3), "max": max(tt), "sum": round(sum(tt), 1)},
        "infer_s_p50": round(float(np.median([r["infer_s"] for r in fr])), 3),
        "peak_reserved_mib_max": max(r["peak_reserved_mib"] for r in fr),
        "peak_allocated_mib_max": max(r["peak_allocated_mib"] for r in fr),
        "argmax_mismatch_pct_max": max(r["argmax_mismatch_pct"] for r in fr),
        "nvidia_smi": {k: steps["seg"].get(k) for k in ("vram_free_before_mib", "nvidia_smi_used_first_mib",
                                                        "nvidia_smi_used_peak_mib", "n_samples")},
        "qc_m1_seg_41": {k: qc[k] for k in ("n_frames", "n_fail", "finite_false", "finite_unknown", "fail_frames")},
        "outputs": {"stable_groups": len(list((d / "stable" / "groups").glob("*.png"))),
                    "stable_depth_smooth": len(list((d / "stable" / "depth_smooth").glob("*.npy"))),
                    "contours": len(list((d / "contours").glob("frame_*.json"))),
                    "strokes_png": len(list((d / "strokes").glob("*.png"))), "strokes_svg": len(list((d / "strokes").glob("*.svg"))),
                    "mp4": sorted(p.name for p in (SCR / "M1" / "out").glob("*.mp4"))}}
    # B + D. cakupan masker / cincin dan label grup di dalam M_f
    erode, sigma = ctx.cfg.vectorize.depth_lines.erode_px, ctx.cfg.vectorize.depth_lines.blur_sigma
    cs_v0 = json.loads((SRC / "contours" / "clip_stats.json").read_text(encoding="utf-8"))
    cov = {v: Counter() for v in ("V0", "M2H", "M1")}
    raw_cov = {v: Counter() for v in ("seg_asli", "seg_M1")}
    agree = Counter()
    grad = {v: Counter() for v in ("V0", "M2H", "M1")}
    extra = Counter()
    m1c = seg.load_clip(M1_SEG)
    for i in frames:
        res = hh.apply_h(np.zeros((ctx.sclip.height, ctx.sclip.width), bool), ctx.mask(i), PROPOSED)
        m_f, m_d = res["m_f"], res["m_d"]
        ring = m_d & ~m_f
        g, inner = {}, {}
        for v in ("V0", "M2H", "M1"):
            dv = variant_dir(v)
            g[v] = stb.read_groups(dv / "stable" / "groups" / (stem(i) + ".png"))
            fg = g[v] != 0
            c = cov[v]
            c["mf"] += int(m_f.sum()); c["ring"] += int(ring.sum())
            c["fg_in_mf"] += int((fg & m_f).sum()); c["fg_in_ring"] += int((fg & ring).sum()); c["fg_out_md"] += int((fg & ~m_d).sum())
            depth = vec.read_depth_smooth(dv / "stable" / "depth_smooth" / (stem(i) + ".npy"), ctx.sclip.height, ctx.sclip.width)
            mag = vec.depth_gradient(depth, sigma)[0]
            inner[v] = vec.inner_region(g[v], erode)
            vals = mag[inner[v]]
            grad[v]["n"] += int(vals.size)
            grad[v]["gt_t_high_v0"] += int((vals > cs_v0["t_high"]).sum())
            if v == "M1":
                e = inner["M1"] & ~inner["M2H"]
                extra["n"] += int(e.sum()); extra["gt_t_high_v0"] += int((mag[e] > cs_v0["t_high"]).sum())
                extra["in_ring"] += int((e & ring).sum()); extra["out_md"] += int((e & ~m_d).sum())
                extra["in_mf"] += int((e & m_f).sum())
                lost = inner["M2H"] & ~inner["M1"]
                extra["n_m2h_only"] += int(lost.sum()); extra["m2h_only_gt_t_high_v0"] += int((mag[lost] > cs_v0["t_high"]).sum())
        pr, cm = ctx.seg_inputs(i)
        ga = hh.raw_groups(pr, cm, ctx.lut, ctx.n_groups)
        p1, c1 = seg.read_probs(m1c.probs_path(FRAME_PATTERN % i)), seg.read_classmap(m1c.classmap_path(FRAME_PATTERN % i))
        gb = hh.raw_groups(p1, c1, ctx.lut, ctx.n_groups)
        for name, gr in (("seg_asli", ga), ("seg_M1", gb)):
            raw_cov[name]["fg_in_mf"] += int(((gr != 0) & m_f).sum()); raw_cov[name]["fg_in_ring"] += int(((gr != 0) & ring).sum())
            raw_cov[name]["fg_out_md"] += int(((gr != 0) & ~m_d).sum())
        both = m_f & (ga != 0) & (gb != 0)
        agree["both_fg_in_mf"] += int(both.sum()); agree["same_group"] += int((both & (ga == gb)).sum())
        for k, name in enumerate(["background", *[n for n, _ in ctx.cfg.groups]]):
            agree[f"area_asli.{name}"] += int(((ga == k) & m_f).sum())
            agree[f"area_M1.{name}"] += int(((gb == k) & m_f).sum())
    n = len(frames)
    out["coverage_stable"] = {v: {"fg_in_mf_pct": round(100 * c["fg_in_mf"] / c["mf"], 2),
                                  "fg_in_ring_pct": round(100 * c["fg_in_ring"] / c["ring"], 2),
                                  "fg_out_md_per_frame": round(c["fg_out_md"] / n, 1)} for v, c in cov.items()}
    out["coverage_raw_seg"] = {v: {"fg_in_mf_pct": round(100 * c["fg_in_mf"] / cov["V0"]["mf"], 2),
                                   "fg_in_ring_pct": round(100 * c["fg_in_ring"] / cov["V0"]["ring"], 2),
                                   "fg_out_md_per_frame": round(c["fg_out_md"] / n, 1)} for v, c in raw_cov.items()}
    out["ring_px_per_frame"] = round(cov["V0"]["ring"] / n, 1)
    out["group_agreement_in_mf"] = {"same_group_pct": round(100 * agree["same_group"] / agree["both_fg_in_mf"], 2),
                                    "area_per_frame": {k: round(v / n, 1) for k, v in agree.items() if k.startswith("area_")}}
    cs = {v: json.loads((variant_dir(v) / "contours" / "clip_stats.json").read_text(encoding="utf-8")) for v in ("V0", "M2H", "M1")}
    out["t_high_analysis"] = {"clip_stats": {v: {k: c[k] for k in ("t_high", "t_low", "n_values")} for v, c in cs.items()},
                              "scene_inner_values": {v: {"n": c["n"], "gt_t_high_v0": c["gt_t_high_v0"],
                                                         "gt_pct": round(100 * c["gt_t_high_v0"] / max(c["n"], 1), 2)}
                                                     for v, c in grad.items()},
                              "m1_extra_vs_m2h": dict(extra)}
    # E. frame di luar adegan dengan strok non-oklusi berbeda (M1)
    diff = []
    for i, name in zip(ctx.sclip.indices, ctx.sclip.names):
        if SCENE[0] <= i <= SCENE[1]:
            continue
        a = [s for s in contours_doc(SRC, i)["strokes"] if s["type"] != "occlusion"]
        b = [s for s in contours_doc(vdir("M1"), i)["strokes"] if s["type"] != "occlusion"]
        if strip_ids(a) != strip_ids(b):
            diff.append({"frame": i, "same_shape_ignoring_order": frame_set(a) == frame_set(b),
                         "types": sorted({s["type"] for s, t in zip(strip_ids(a), strip_ids(b)) if s != t})})
    out["non_occlusion_diff_outside_scene"] = diff
    jsave(OUT / "m1_cek.json", out)
    print(json.dumps(out, indent=1, ensure_ascii=False))


# ── Tahap 1c: variasi d ────────────────────────────
D_LIST = (4, 8, 12, 20)
D_VIDEO = (4, 8, 12)


def d_var(d: int) -> str:
    return "M2H" if d == PROPOSED.d_px else f"M2H_d{d}"


def cmd_tepi(ds: tuple[int, ...] = D_LIST) -> None:
    """Per d (seg mentah, sebelum [3]; parameter lain = PROPOSED):
      rim     = piksel F di luar M_f pada sinar normal yang BERAKHIR ≤ 40 px (`ends`) = tepi seg target yang melewati masker;
      contact = piksel F di luar M_f pada sinar yang DISENSOR (F berlanjut ≥ 40 px) = zona sentuh / bersambung orang lain;
      cut     = rim di komponen `potong` yang TIDAK dipertahankan H (di luar M_d / potongan dibuang) = "seg asli target terpotong";
      neighbor_kept = contact yang dipertahankan H (di cincin M_d − M_f) = "orang lain ikut";
      kept_not_rim = piksel dipertahankan di luar M_f, di komponen `potong`, bukan rim (batas atas "orang lain ikut").
    Komponen `utuh` (f121–f125) tidak dipotong pada d berapa pun (r tidak bergantung d)."""
    ctx = Ctx()
    frames = scene_frames(ctx, ctx.mask_state())
    names = ["background", *[n for n, _ in ctx.cfg.groups]]
    keys = ("rim_px", "cut_px", "neighbor_kept_px", "kept_not_rim_px", "keep_outside_md_px")
    per = {d: {} for d in ds}
    grp = {d: Counter() for d in ds}
    side = {d: Counter() for d in ds}
    ends_v, n_cen, n_rays = [], 0, 0
    for i in frames:
        pr, cm = ctx.seg_inputs(i)
        graw = hh.raw_groups(pr, cm, ctx.lut, ctx.n_groups)
        fg = graw != 0
        m = ctx.mask(i)
        m_f, _, _ = hh.fill_holes(m, PROPOSED.hole_max_px)
        _, lab, _, _ = cv2.connectedComponentsWithStats(fg.astype(np.uint8), connectivity=8)
        rays = overshoot_rays(m_f, fg, lab, record=True)
        px, cat = rays["px"], rays["cat"]

        def union(sel: np.ndarray) -> np.ndarray:
            v = px[sel].ravel()
            out = np.zeros(fg.size, bool)
            out[v[v >= 0]] = True
            return out.reshape(fg.shape)

        rim, contact = union(cat == "ends"), union(cat == "censored")
        cx = float(np.nonzero(m)[1].mean())
        xs = np.arange(fg.shape[1])[None, :]
        for d in ds:
            res = hh.apply_h(fg, m, dataclasses.replace(PROPOSED, d_px=d))
            pot_labels = [c["label"] for c in res["comps"] if c["class"] == "potong"]
            potong = np.isin(lab, pot_labels)
            keep, m_d = res["keep"], res["m_d"]
            cut = rim & potong & ~keep
            row = {"rim_px": int((rim & potong).sum()), "cut_px": int(cut.sum()),
                   "neighbor_kept_px": int((contact & keep & ~m_f).sum()),
                   "kept_not_rim_px": int((keep & potong & ~m_f & ~rim).sum()),
                   "keep_outside_md_px": int((keep & ~m_d).sum()),
                   "cut_by_group": {names[int(g)]: int(c) for g, c in zip(*np.unique(graw[cut], return_counts=True))},
                   "cut_left_right": [int((cut & (xs < cx)).sum()), int((cut & (xs >= cx)).sum())]}
            per[d][i] = row
            grp[d].update(row["cut_by_group"])
            side[d].update({"kiri": row["cut_left_right"][0], "kanan": row["cut_left_right"][1]})
        sel = np.isin(rays["comp"], pot_labels)                   # kelas komponen sama untuk semua d
        ends_v += rays["val"][sel & (cat == "ends")].tolist()
        n_cen += int((sel & (cat == "censored")).sum())
        n_rays += int(sel.sum())
    ev = np.asarray(ends_v)
    out = {"definitions": cmd_tepi.__doc__, "params_base": PROPOSED.to_json(), "frames": [frames[0], frames[-1]],
           "rays_potong": {"n_boundary_px": n_rays, "censored_px": n_cen, "ends_n": int(ev.size),
                           "ends_gt_d_pct": {str(d): round(100 * float((ev > d).mean()), 2) for d in ds}},
           "per_d": {}}
    for d in ds:
        rows = per[d]
        agg = {k: {"per_frame": round(float(np.mean([r[k] for r in rows.values()])), 1),
                   "max": max(r[k] for r in rows.values()),
                   "max_frame": max(rows, key=lambda i: rows[i][k]), "total": int(sum(r[k] for r in rows.values()))}
               for k in keys}
        tot_cut = sum(grp[d].values())
        agg["cut_pct_of_rim"] = round(100 * agg["cut_px"]["total"] / max(agg["rim_px"]["total"], 1), 2)
        agg["frames_cut_gt_0"] = [i for i, r in rows.items() if r["cut_px"] > 0]
        agg["frames_cut_ge_100"] = [i for i, r in rows.items() if r["cut_px"] >= 100]
        agg["cut_by_group_pct"] = {g: round(100 * c / max(tot_cut, 1), 1) for g, c in grp[d].most_common()}
        agg["cut_left_right_pct"] = {k: round(100 * v / max(tot_cut, 1), 1) for k, v in side[d].items()}
        out["per_d"][str(d)] = {"agg": agg, "rows": rows}
    jsave(OUT / "tepi_1c.json", out)
    print(json.dumps({"rays_potong": out["rays_potong"],
                      **{f"d{d}": {k: v for k, v in out["per_d"][str(d)]["agg"].items() if k != "frames_cut_gt_0"}
                         for d in ds}}, indent=1, ensure_ascii=False))
    print(f"-> {(OUT / 'tepi_1c.json').relative_to(ROOT).as_posix()}")


BIG_BAR_H = 88


def labeled_big(img: np.ndarray, title: str, sub: str) -> np.ndarray:
    bar = np.full((BIG_BAR_H, img.shape[1], 3), 28, np.uint8)
    cv2.putText(bar, title, (10, 38), FONT, 1.15, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(bar, sub, (10, 76), FONT, 0.8, (0, 230, 255), 2, cv2.LINE_AA)
    return np.vstack([bar, img])


def cmd_bahan_1c() -> None:
    ctx = Ctx()
    frames = scene_frames(ctx, ctx.mask_state())
    fps = float(json.loads((SRC / "meta.json").read_text(encoding="utf-8"))["target_fps"])
    cols = [(f"d = {d} px" + (" (acuan)" if d == PROPOSED.d_px else ""), vdir(d_var(d))) for d in D_VIDEO]
    for _, d in cols:
        missing = [i for i in frames if not (d / "strokes" / (stem(i) + ".png")).is_file()]
        if missing:
            raise SystemExit(f"{d}/strokes belum lengkap (mis. f{missing[0]}) — jalankan m2 --varian dulu")
    NILAI.mkdir(parents=True, exist_ok=True)
    name = "klip3_banding_" + "_".join(f"d{d}" for d in D_VIDEO) + f"_f{SCENE[0]:03d}-{SCENE[1]:03d}"
    made = []
    for suffix, f in (("", fps), ("_lambat", fps / 2)):
        def gen():
            for k, i in enumerate(frames, 1):
                sub = f"f{i:03d}  t={i / fps:.2f} s  ({k}/{len(frames)})"
                ps = [labeled_big(photo_panel(ctx, i), "foto + tepi masker", sub)]
                ps += [labeled_big(strokes_panel(d, i), title, sub) for title, d in cols]
                yield even(np.hstack(ps))
        n = write_video(NILAI / f"{name}{suffix}.mp4", gen(), f)
        made.append({"video": f"{name}{suffix}.mp4", "frames": [frames[0], frames[-1]], "n": n, "fps": f,
                     "panel": ["foto + tepi masker"] + [t for t, _ in cols]})
        print(f"  {NILAI.relative_to(ROOT).as_posix()}/{name}{suffix}.mp4: {n} frame, {f:g} fps")
    idx = OUT / "bahan.json"
    old = json.loads(idx.read_text(encoding="utf-8")) if idx.is_file() else {}
    old["1c"] = made
    jsave(idx, old)


def main(argv: list[str]) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(prog="t606_tahap1.py")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ukur")
    m2 = sub.add_parser("m2")
    m2.add_argument("--varian", default="V0s,M2H")
    mt = sub.add_parser("metrik")
    mt.add_argument("--varian", default="M2H")
    mt.add_argument("--out", default="metrik.json", help="nama berkas di work/t606/ (Tahap 1c: metrik_1c.json)")
    bh = sub.add_parser("bahan")
    bh.add_argument("--m1", action="store_true")
    sub.add_parser("m1cek")
    sub.add_parser("tepi")
    sub.add_parser("bahan1c")
    a = ap.parse_args(argv)
    if a.cmd == "ukur":
        cmd_ukur()
    elif a.cmd == "m2":
        cmd_m2([v for v in a.varian.split(",") if v])
    elif a.cmd == "metrik":
        cmd_metrik([v for v in a.varian.split(",") if v], a.out)
    elif a.cmd == "tepi":
        cmd_tepi()
    elif a.cmd == "bahan1c":
        cmd_bahan_1c()
    elif a.cmd == "bahan":
        cmd_bahan(a.m1)
    elif a.cmd == "m1cek":
        cmd_m1cek()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
