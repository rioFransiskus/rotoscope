"""Diagnosis CPU baca-saja klip3 (alat sekali pakai; HANYA membaca work/clips/klip3, hasil JSON ke work/klip3_diag_scratch/):

    python scripts/klip3_diag.py 1   # invarian reach: komponen baru di f41 (penyebab)
    python scripts/klip3_diag.py 2   # QC: run frame gagal, blob per frame, dampak ke temporal [3]
    python scripts/klip3_diag.py 3   # cut: campuran lintas cut, cut tak terdeteksi / palsu

Tanpa GPU; tidak ada berkas klip yang ditulis. Observasi: tidak memperbaiki apa pun."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "src"))

import reach_metrics as rm  # noqa: E402
import temporal_metrics as tm  # noqa: E402

from rotoscope import stabilize as stb  # noqa: E402
from rotoscope import vectorize as vec  # noqa: E402
from rotoscope.config import load_pipeline  # noqa: E402

WORK = ROOT / "work" / "clips" / "klip3"
OUT = ROOT / "work" / "klip3_diag_scratch"
FRAME_DIAG1 = 41
NEAR_PX = 10                       # blob kecil <= ini dari blob terbesar = "dekat" (potongan tubuh); selain itu "jauh"


def jload(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def save(name: str, obj) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"  work/klip3_diag_scratch/{name}")


def pix(stroke: dict, shape) -> np.ndarray:
    """Titik strok (koordinat tengah piksel + 0,5) -> mask piksel."""
    m = np.zeros(shape, bool)
    for x, y in stroke["points"]:
        m[int(np.floor(y)), int(np.floor(x))] = True
    return m


def describe(stroke: dict) -> dict:
    p = np.asarray(stroke["points"], float)
    return {"groups": stroke["groups"], "type": stroke["type"], "n_points": len(p), "closed": stroke["closed"],
            "length_px": float(np.hypot(*np.diff(p, axis=0).T).sum()) if len(p) > 1 else 0.0,
            "bbox_xyxy": [float(p[:, 0].min()), float(p[:, 1].min()), float(p[:, 0].max()), float(p[:, 1].max())],
            "first": p[0].tolist(), "last": p[-1].tolist()}


def min_dist(stroke: dict, others: list[dict]) -> tuple[float, float]:
    """(Chebyshev, Euclid) jarak terkecil titik strok ke titik strok lain."""
    op = np.concatenate([rm.pts(s) for s in others]) if others else np.zeros((0, 2))
    if not len(op):
        return float("inf"), float("inf")
    t = cKDTree(op)
    return float(t.query(rm.pts(stroke), p=np.inf)[0].min()), float(t.query(rm.pts(stroke))[0].min())


def diag1() -> None:
    cfg = load_pipeline()
    names = tuple(g for g, _ in cfg.groups)
    clip = vec.load_clip(WORK)
    mv = jload(WORK / "contours" / "manifest.json")["vectorize"]
    cs = jload(WORK / "contours" / "clip_stats.json")
    th, tl = cs["t_high"], cs["t_low"]
    nme = clip.names[FRAME_DIAG1]
    gmap, depth = vec.load_frame_inputs(clip, nme, len(names))
    H, W = gmap.shape

    def occ(**over):
        return vec.occlusion_strokes(gmap, depth, names, {**mv, **over}, th, tl)[0]

    variants = {"dasar (D_low 0, exclude [])": occ(**{"depth_lines.min_dist_low_px": 0.0, "depth_lines.exclude_groups": []}),
                "D_low 2, exclude []": occ(**{"depth_lines.exclude_groups": []}),
                "D_low 2, exclude [hair] (produksi)": occ()}
    base = variants["dasar (D_low 0, exclude [])"]
    res: dict = {"frame": FRAME_DIAG1, "t_high": th, "t_low": tl, "params": {k: v for k, v in mv.items() if k.startswith("depth_lines")},
                 "n_strokes": {k: len(v) for k, v in variants.items()}}
    # 1) komponen baru per varian terhadap dasar (definisi rm.new_components: titik tanpa tetangga <= 1 px Chebyshev)
    res["new_components"] = {k: rm.new_components(base, v) for k, v in variants.items()}
    prod_doc = [s for s in jload(WORK / "contours" / f"frame_{FRAME_DIAG1:05d}.json")["strokes"] if s["type"] == "occlusion"]
    res["new_components"]["contours/frame_00041.json (tersimpan)"] = rm.new_components(base, prod_doc)
    var = variants["D_low 2, exclude [hair] (produksi)"]
    bp = np.concatenate([rm.pts(s) for s in base]) if base else np.zeros((0, 2))
    tree = cKDTree(bp)
    offenders = [j for j, s in enumerate(var) if tree.query(rm.pts(s), p=np.inf)[0].min() > 1.0 + 1e-9]
    res["offenders"] = []
    # 2) masker tahap
    dp = vec.depth_params(mv)
    m = vec.occlusion_masks(gmap, depth, dp, th, tl)
    seed = m["dist_ok"]
    dist = vec.boundary_distance(gmap)
    skip = set()
    kept = vec.surviving_seeds(gmap, seed, mv, skip)
    ridge = m["hyst"] & (gmap != stb.BACKGROUND_ID) & (dist >= dp["min_dist_low_px"]) & ~seed
    ext = vec.reach_extension(kept, ridge, dist, dp["min_dist_px"], dp["min_dist_low_px"])
    mask = kept | ext
    n_seed, lab_seed = cv2.connectedComponents(seed.astype(np.uint8), connectivity=8)
    n_mask, lab_mask = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    res["mask_px"] = {"hyst": int(m["hyst"].sum()), "seed_dist_ok": int(seed.sum()), "kept_after_L": int(kept.sum()), "ext": int(ext.sum()),
                      "n_seed_components": n_seed - 1, "n_kept_plus_ext_components": n_mask - 1}
    for j in offenders:
        s = var[j]
        pm = pix(s, (H, W))
        gid = names.index(s["groups"][0]) + 1
        info = {**describe(s), "min_dist_to_base_cheb": min_dist(s, base)[0], "min_dist_to_base_euclid": min_dist(s, base)[1],
                "frac_pixels_in": {"hyst": float((pm & m["hyst"]).sum() / pm.sum()), "seed_dist_ok": float((pm & seed).sum() / pm.sum()),
                                   "kept": float((pm & kept).sum() / pm.sum()), "ext": float((pm & ext).sum() / pm.sum()),
                                   "in_group": float((pm & (gmap == gid)).sum() / pm.sum())}}
        # komponen mask (kept|ext) yang memuat strok dan benih (komponen dist_ok) di dalamnya
        labs = np.unique(lab_mask[pm & mask])
        labs = labs[labs > 0]
        comp = np.isin(lab_mask, labs)
        seeds_in = np.unique(lab_seed[comp & seed])
        seeds_in = seeds_in[seeds_in > 0]
        info["mask_component"] = {"px": int(comp.sum()), "px_seed": int((comp & seed).sum()), "px_kept": int((comp & kept).sum()),
                                  "px_ext": int((comp & ext).sum()), "n_seed_components_inside": len(seeds_in)}
        # panjang skeleton benih tiap komponen benih (per grup, seperti surviving_seeds) dan apakah ada di strok dasar
        sk_info = []
        for sid in seeds_in:
            sm = lab_seed == sid
            thinned = vec.thin_band(sm & (gmap == gid), mv["line_min_px"])
            if thinned is None:
                sk_info.append({"seed_id": int(sid), "px": int(sm.sum()), "skeleton": None})
                continue
            skel, ox, oy = thinned
            n_sk, lab_sk, st_sk, _ = cv2.connectedComponentsWithStats(skel.astype(np.uint8), connectivity=8)
            lens = [int(st_sk[k, cv2.CC_STAT_AREA]) for k in range(1, n_sk)]
            ys, xs = np.nonzero(skel)
            full = np.zeros((H, W), bool)
            full[ys + oy, xs + ox] = True
            sk_info.append({"seed_id": int(sid), "px": int(sm.sum()), "skeleton_component_lengths_px": lens, "L": dp["min_len_px"],
                            "survives_L": bool(max(lens, default=0) >= dp["min_len_px"]), "skeleton_px_in_kept": int((full & kept).sum())})
        info["seed_skeletons"] = sk_info
        # apakah ada strok dasar di dalam / dekat komponen yang sama (selain yang tidak menempel)
        info["base_strokes_in_component_dilated5"] = int(sum(
            1 for b in base if (pix(b, (H, W)) & cv2.dilate(comp.astype(np.uint8), np.ones((11, 11), np.uint8)).astype(bool)).any()))
        # panjang strok varian vs strok dasar terdekat
        nearest = min(range(len(base)), key=lambda k: min_dist(s, [base[k]])[1]) if base else None
        info["nearest_base_stroke"] = (describe(base[nearest]) | {"min_dist_euclid": min_dist(s, [base[nearest]])[1]}) if nearest is not None else None
        res["offenders"].append(info)
    # 3) dasar tanpa exclude vs dengan: apakah komponen baru juga muncul tanpa D_low? (varian "dasar" = pembanding itu sendiri);
    #    uji mandiri: D_low 0 + exclude [hair] terhadap dasar, dan D_low 1 / 2 / 3 tanpa exclude
    res["sweep_D_low_new_components"] = {}
    for dl in (0.0, 1.0, 2.0, 3.0, 4.0):
        for ex in ([], ["hair"]):
            v = occ(**{"depth_lines.min_dist_low_px": dl, "depth_lines.exclude_groups": ex})
            res["sweep_D_low_new_components"][f"D_low {dl:g}, exclude {ex}"] = {"n_strokes": len(v), "new_components": rm.new_components(base, v)}
    # 4) seluruh klip: berapa frame yang melanggar (konteks)
    viol = []
    for i, n in enumerate(clip.names):
        g2, d2 = vec.load_frame_inputs(clip, n, len(names))
        b2 = vec.occlusion_strokes(g2, d2, names, {**mv, "depth_lines.min_dist_low_px": 0.0, "depth_lines.exclude_groups": []}, th, tl)[0]
        k2 = [s for s in jload(WORK / "contours" / f"frame_{i:05d}.json")["strokes"] if s["type"] == "occlusion"]
        c = rm.new_components(b2, k2)
        if c:
            viol.append((i, c))
    res["violations_whole_clip"] = viol
    save("diag1.json", res)
    print(json.dumps(res, indent=1, ensure_ascii=False, default=str)[:6000])


def blobs_of(cm: np.ndarray, groups_raw: np.ndarray, names, blob_min: float):
    fg = (cm != 0).astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(fg, connectivity=8)
    fgpx = int(fg.sum())
    order = sorted(range(1, n), key=lambda i: -st[i, cv2.CC_STAT_AREA])
    out = []
    big_min = blob_min * fg.size
    main = order[0] if order else None
    dmain = cv2.distanceTransform((lab != main).astype(np.uint8), cv2.DIST_L2, 5) if main else None
    for i in order:
        a = int(st[i, cv2.CC_STAT_AREA])
        if a < 50:
            continue
        gm = groups_raw[lab == i]
        hist = np.bincount(gm, minlength=len(names) + 1)[1:]
        out.append({"area_px": a, "pct_of_fg": 100 * a / max(fgpx, 1), "pct_of_frame": 100 * a / fg.size, "big": bool(a > big_min),
                    "centroid_xy": [float(cen[i][0]), float(cen[i][1])],
                    "bbox_xywh": [int(st[i, cv2.CC_STAT_LEFT]), int(st[i, cv2.CC_STAT_TOP]), int(st[i, cv2.CC_STAT_WIDTH]), int(st[i, cv2.CC_STAT_HEIGHT])],
                    "min_dist_to_largest_px": 0.0 if i == main else float(dmain[lab == i].min()),
                    "dominant_group": names[int(hist.argmax())] if hist.sum() else None,
                    "group_px": {names[k]: int(hist[k]) for k in range(len(names)) if hist[k]}})
    return out, lab, main


def runs_of(flags) -> list[tuple[int, int]]:
    out, s = [], None
    for i, f in enumerate(flags):
        if f and s is None:
            s = i
        if not f and s is not None:
            out.append((s, i - 1))
            s = None
    if s is not None:
        out.append((s, len(flags) - 1))
    return out


def diag2() -> None:
    cfg = load_pipeline()
    names = tuple(g for g, _ in cfg.groups)
    q = jload(WORK / "qc_report.json")
    rows = q["frames"]
    blob_min, max_blobs = q["qc"]["blob_min"], q["qc"]["max_big_blobs"]
    fail = [bool(r["fail_reasons"]) for r in rows]
    n = len(rows)
    runs = runs_of(fail)
    lens = [b - a + 1 for a, b in runs]
    clip = stb.load_clip(WORK)
    classes = jload(WORK / "seg" / "manifest.json")["classes"]
    lut = stb.class_group_lut(cfg.groups, classes)
    res: dict = {"rule": f"big_blobs = jumlah komponen 8-arah foreground (classmap != 0) dengan luas > blob_min ({blob_min}) x luas frame; gagal bila > max_big_blobs ({max_blobs})",
                 "frame_px": rows[0] and 720 * 1280, "big_blob_threshold_px": blob_min * 720 * 1280, "n_fail": sum(fail),
                 "runs": [[a, b, b - a + 1] for a, b in runs], "n_runs": len(runs), "longest_run": max(lens), "runs_ge_3": sum(l >= 3 for l in lens),
                 "frames_in_runs_ge_3": int(sum(l for l in lens if l >= 3))}
    stable = [stb.read_groups(clip.groups_path(nm)) for nm in clip.names]
    per, rawm = {}, {}
    for i, nm in enumerate(clip.names):
        cm = cv2.imdecode(np.fromfile(WORK / "seg" / "classmap" / nm, np.uint8), cv2.IMREAD_UNCHANGED)
        rawm[i] = lut[cm]
        bl, lab, main = blobs_of(cm, rawm[i], names, blob_min)
        per[i] = {"fail": fail[i], "n_big_blobs_qc": rows[i]["big_blobs"], "blobs": bl[:6], "n_blobs_ge_50px": len(bl)}
    # subjek utama = blob terbesar frame itu; bandingkan IoU dengan blob terbesar frame tidak-gagal terdekat
    good = [i for i in range(n) if not fail[i]]
    main_mask = {}
    for i, nm in enumerate(clip.names):
        cm = cv2.imdecode(np.fromfile(WORK / "seg" / "classmap" / nm, np.uint8), cv2.IMREAD_UNCHANGED)
        _, lab, main = blobs_of(cm, rawm[i], names, blob_min)
        main_mask[i] = (lab == main) if main else np.zeros_like(cm, bool)
    sec_cls = {"dekat_<=%d" % NEAR_PX: 0, "jauh": 0}
    sec_group: dict[str, int] = {}
    sec_pct = []
    for i in range(n):
        if not fail[i]:
            continue
        g = min(good, key=lambda k: abs(k - i)) if good else None
        iou = float((main_mask[i] & main_mask[g]).sum() / max((main_mask[i] | main_mask[g]).sum(), 1)) if g is not None else None
        per[i]["largest_blob_iou_with_nearest_good_frame_largest"] = iou
        per[i]["nearest_good_frame"] = g
        for b in per[i]["blobs"][1:]:
            if b["big"]:
                key = "dekat_<=%d" % NEAR_PX if b["min_dist_to_largest_px"] <= NEAR_PX else "jauh"
                sec_cls[key] += 1
                sec_group[b["dominant_group"]] = sec_group.get(b["dominant_group"], 0) + 1
                sec_pct.append(b["pct_of_frame"])
    fr_main = [per[i]["largest_blob_iou_with_nearest_good_frame_largest"] for i in range(n) if fail[i]]
    res["secondary_big_blobs"] = {"by_distance_to_largest": sec_cls, "dominant_group_counts": sec_group,
                                  "pct_of_frame": {"min": float(min(sec_pct)), "p50": float(np.median(sec_pct)), "max": float(max(sec_pct))},
                                  "margin_over_threshold_pct_points": {"min": float(min(sec_pct) - 100 * blob_min), "p50": float(np.median(sec_pct) - 100 * blob_min)}}
    # kontinuitas blob terbesar antar frame berurutan (lebih bermakna daripada frame baik terdekat yang bisa berjarak puluhan frame)
    cont = {i: float((main_mask[i] & main_mask[i - 1]).sum() / max((main_mask[i] | main_mask[i - 1]).sum(), 1)) for i in range(1, n)}
    cuts_set = set(jload(WORK / "stable" / "manifest.json")["temporal"].get("cut_frames", []))
    cut_idx = {int(c.split("_")[1].split(".")[0]) for c in cuts_set}
    f_c = [cont[i] for i in range(1, n) if fail[i] and i not in cut_idx]
    o_c = [cont[i] for i in range(1, n) if not fail[i] and i not in cut_idx]
    sec_ratio = [per[i]["blobs"][1]["area_px"] / per[i]["blobs"][0]["area_px"] for i in range(n) if fail[i] and len(per[i]["blobs"]) > 1]
    big_pct_fg = [per[i]["blobs"][0]["pct_of_fg"] for i in range(n) if fail[i]]
    res["main_subject_continuity"] = {
        "largest_blob_iou_with_prev_frame_largest_FAIL": {"min": float(min(f_c)), "p5": float(np.percentile(f_c, 5)), "p50": float(np.median(f_c))},
        "largest_blob_iou_with_prev_frame_largest_OK": {"min": float(min(o_c)), "p5": float(np.percentile(o_c, 5)), "p50": float(np.median(o_c))},
        "largest_blob_pct_of_fg_FAIL": {"min": float(min(big_pct_fg)), "p50": float(np.median(big_pct_fg)), "max": float(max(big_pct_fg))},
        "second_over_largest_area_FAIL": {"min": float(min(sec_ratio)), "p50": float(np.median(sec_ratio)), "max": float(max(sec_ratio))},
        "fail_frames_where_second_blob_ge_largest_x0.8": int(sum(r >= 0.8 for r in sec_ratio)),
        "excluded_cut_frames": sorted(cut_idx)}
    res["largest_blob_iou_vs_nearest_good"] = {"min": float(min(fr_main)), "p50": float(np.median(fr_main)), "frames_iou_lt_0.5": int(sum(x < 0.5 for x in fr_main))}
    # dampak ke temporal [3]
    plan = stb.build_plan(cfg, clip, log=lambda *_: None)
    fail_idx = [i for i in range(n) if fail[i]]
    share, no_good = {}, []
    for i in fail_idx:
        w = plan.weights(i, plan.radius)
        tot = sum(x for _, x in w)
        good_w = sum(x for k, x in w if not fail[k])
        share[i] = good_w / tot if tot else 0.0
        if not any(not fail[k] for k, _ in w):
            no_good.append(i)
    in_ge3 = {i for a, b in runs if b - a + 1 >= 3 for i in range(a, b + 1)}
    iou_f, iou_g, chg = {}, {}, {}
    for i in fail_idx:
        r, s = rawm[i], stable[i]
        fr_, fs_ = r != 0, s != 0
        iou_f[i] = float((fr_ & fs_).sum() / max((fr_ | fs_).sum(), 1))
        ids = [g for g in np.unique(r) if g != 0]
        iou_g[i] = float(np.mean([((r == g) & (s == g)).sum() / max(((r == g) | (s == g)).sum(), 1) for g in ids])) if ids else 1.0
        chg[i] = int((r != s).sum())
    ok_idx = [i for i in range(n) if not fail[i]]
    iou_ok = [float(((rawm[i] != 0) & (stable[i] != 0)).sum() / max(((rawm[i] != 0) | (stable[i] != 0)).sum(), 1)) for i in ok_idx]
    chg_ok = [int((rawm[i] != stable[i]).sum()) for i in ok_idx]
    iou_g_ok = []
    for i in ok_idx:
        ids = [g for g in np.unique(rawm[i]) if g != 0]
        iou_g_ok.append(float(np.mean([((rawm[i] == g) & (stable[i] == g)).sum() / max(((rawm[i] == g) | (stable[i] == g)).sum(), 1) for g in ids])) if ids else 1.0)

    def st(a):
        a = np.asarray(a, float)
        return {"min": float(a.min()), "p50": float(np.median(a)), "p95": float(np.percentile(a, 95)), "max": float(a.max())}

    res["temporal_impact"] = {"radius_R": plan.radius, "rho": plan.rho, "q_fail": cfg.stabilize.temporal.qc_fail_weight,
                              "n_fail_frames": len(fail_idx), "fail_frames_in_run_ge_3": sum(i in in_ge3 for i in fail_idx),
                              "fail_frames_window_has_no_good_frame": len(no_good), "no_good_frames_list": no_good,
                              "good_weight_share_of_fail_frames": st(list(share.values())),
                              "fail_frames_good_weight_share_lt_0.5": int(sum(v < 0.5 for v in share.values())),
                              "fg_iou_stable_vs_raw_FAIL": st(list(iou_f.values())), "fg_iou_stable_vs_raw_OK": st(iou_ok),
                              "group_iou_stable_vs_raw_FAIL": st(list(iou_g.values())), "group_iou_stable_vs_raw_OK": st(iou_g_ok),
                              "px_changed_raw_vs_stable_FAIL": st(list(chg.values())), "px_changed_raw_vs_stable_OK": st(chg_ok)}
    res["per_frame_fail"] = {i: {k: per[i][k] for k in ("n_big_blobs_qc", "n_blobs_ge_50px", "largest_blob_iou_with_nearest_good_frame_largest")}
                             | {"blobs": [{k2: (round(v2, 3) if isinstance(v2, float) else v2) for k2, v2 in b.items() if k2 != "group_px"} for b in per[i]["blobs"][:4]]}
                             | {"good_weight_share": share[i], "fg_iou_stable_raw": iou_f[i], "px_changed": chg[i]} for i in fail_idx}
    save("diag2.json", res)
    brief = {k: v for k, v in res.items() if k != "per_frame_fail"}
    print(json.dumps(brief, indent=1, ensure_ascii=False, default=str)[:7000])


def diag3() -> None:
    cfg = load_pipeline()
    names = tuple(g for g, _ in cfg.groups)
    clip = stb.load_clip(WORK)
    n = len(clip.names)
    classes = jload(WORK / "seg" / "manifest.json")["classes"]
    lut = stb.class_group_lut(cfg.groups, classes)
    raw = np.stack([lut[cv2.imdecode(np.fromfile(WORK / "seg" / "classmap" / nm, np.uint8), cv2.IMREAD_UNCHANGED)] for nm in clip.names])
    stable = np.stack([stb.read_groups(clip.groups_path(nm)) for nm in clip.names])
    plan = stb.build_plan(cfg, clip, log=lambda *_: None)
    scores = stb.cut_scores([stb.frame_thumb(WORK / "frames" / nm) for nm in clip.names])
    cut_diff = cfg.stabilize.temporal.cut_diff
    cuts = list(plan.cut_frames)
    qc = {r["index"]: bool(r["fail_reasons"]) for r in jload(WORK / "qc_report.json")["frames"]}

    def fgiou(a, b):
        x, y = a != 0, b != 0
        return float((x & y).sum() / max((x | y).sum(), 1))

    def gmean(a, b):
        ids = [g for g in np.unique(a) if g != 0]
        return float(np.mean([((a == g) & (b == g)).sum() / max(((a == g) | (b == g)).sum(), 1) for g in ids])) if ids else 1.0

    self_fg = np.array([fgiou(stable[i], raw[i]) for i in range(n)])
    self_g = np.array([gmean(raw[i], stable[i]) for i in range(n)])
    near_cut = set()
    for c in cuts:
        near_cut |= {c - 1, c}
    noncut = [i for i in range(n) if i not in near_cut]
    res = {"cut_diff": cut_diff, "cuts_detected": cuts, "cut_scores_at_cuts": {c: scores[c] for c in cuts},
           "plan_shots": {c: {"window_t-1": plan.window(c - 1, plan.radius), "window_t": plan.window(c, plan.radius)} for c in cuts},
           "reference_self_iou_noncut": {"fg_p5": float(np.percentile(self_fg[noncut], 5)), "fg_p50": float(np.median(self_fg[noncut])),
                                         "grp_p5": float(np.percentile(self_g[noncut], 5)), "grp_p50": float(np.median(self_g[noncut]))}}
    per = {}
    for c in cuts:
        d = {}
        for tag, i in (("t-1", c - 1), ("t", c)):
            other = c if i == c - 1 else c - 1          # frame di seberang cut
            d[tag] = {"frame": i, "qc_fail": qc[i], "fg_iou_stable_vs_own_raw": float(self_fg[i]), "grp_iou_stable_vs_own_raw": float(self_g[i]),
                      "fg_iou_stable_vs_OTHER_side_raw": fgiou(stable[i], raw[other]), "grp_iou_stable_vs_OTHER_side_raw": gmean(raw[other], stable[i]),
                      "px_changed_vs_own_raw": int((stable[i] != raw[i]).sum()),
                      "own_over_other_fg_iou_ratio": float(self_fg[i] / max(fgiou(stable[i], raw[other]), 1e-9))}
        d["raw_fg_iou_across_cut"] = fgiou(raw[c - 1], raw[c])
        d["shot_start_of_t"], d["shot_end_of_t-1"] = plan.shot_start[c], plan.shot_end[c - 1]
        per[c] = d
    res["per_cut"] = per
    order = np.argsort(scores)[::-1]
    res["top_scores"] = [{"frame": int(i), "score": float(scores[i]), "detected": int(i) in cuts, "ratio_to_cut_diff": float(scores[i] / cut_diff),
                          "raw_fg_iou_with_prev": fgiou(raw[i - 1], raw[i]) if i else None, "qc_fail": qc[int(i)]} for i in order[:12]]
    nonc = [float(scores[i]) for i in range(1, n) if i not in cuts]
    res["non_cut_scores"] = {"max": float(max(nonc)), "p95": float(np.percentile(nonc, 95)), "p50": float(np.median(nonc))}
    res["undetected_candidates_score_ge_0.5x_threshold"] = [{"frame": i, "score": float(scores[i])} for i in range(1, n) if i not in cuts and scores[i] >= 0.5 * cut_diff]
    chk = {}
    for f in [int(i) for i in order if int(i) not in cuts][:4]:          # 4 skor tertinggi di luar cut terdeteksi
        chk[f] = {"score": float(scores[f]), "neighbor_scores": [float(scores[f - 1]), float(scores[f + 1])] if 0 < f < n - 1 else None,
                  "fg_iou_stable_t_vs_own_raw": float(self_fg[f]), "fg_iou_stable_t_vs_raw_prev": fgiou(stable[f], raw[f - 1]),
                  "grp_iou_stable_t_vs_own_raw": float(self_g[f]), "grp_iou_stable_t_vs_raw_prev": gmean(raw[f - 1], stable[f]),
                  "fg_iou_stable_prev_vs_raw_t": fgiou(stable[f - 1], raw[f]), "raw_fg_iou_with_prev": fgiou(raw[f - 1], raw[f]),
                  "window_t": plan.window(f, plan.radius)}
    res["undetected_top4_leak_check"] = chk
    res["true_cut_neighbor_scores"] = {c: [float(scores[c - 1]), float(scores[c]), float(scores[c + 1]) if c + 1 < n else None] for c in cuts}
    sp = tm.centroid_speed(raw)
    res["centroid_speed_raw_at_cuts"] = {c: float(sp[c]) for c in cuts}
    res["centroid_speed_raw_top5_noncut"] = sorted(((int(i), float(sp[i])) for i in range(1, n) if i not in cuts), key=lambda x: -x[1])[:5]
    save("diag3.json", res)
    print(json.dumps(res, indent=1, ensure_ascii=False, default=str)[:7000])


# ── Diagnosis 2 (lengan, kaki, lipatan, cek cut) ───
ARM_FRAMES = (19, 24, 30, 36, 42, 60)       # f19 (pengamatan Rio) + 5 frame 3-blob non-cut
BLOB_MIN = 0.05
ASSIGN_MIN_PX = 30                          # komponen foreground >= ini ikut orang terdekat


def class_groups(cfg, classes):
    mp = {c: g for g, members in cfg.groups for c in members}
    return mp, {g: [c for c in classes if mp.get(c) == g] for g, _ in cfg.groups}


def persons_of(cm: np.ndarray, H: int, W: int) -> list[dict]:
    """Orang = komponen foreground besar (> BLOB_MIN x frame); komponen kecil >= ASSIGN_MIN_PX ikut yang terdekat. Urut kiri -> kanan."""
    fg = (cm != 0).astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(fg, connectivity=8)
    big = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] > BLOB_MIN * H * W]
    big.sort(key=lambda i: cen[i][0])
    if not big:
        return []
    dists = [cv2.distanceTransform((lab != i).astype(np.uint8), cv2.DIST_L2, 5) for i in big]
    masks = [lab == i for i in big]
    small = []
    for i in range(1, n):
        if i in big or st[i, cv2.CC_STAT_AREA] < ASSIGN_MIN_PX:
            continue
        comp = lab == i
        d = [float(dd[comp].min()) for dd in dists]
        k = int(np.argmin(d))
        masks[k] |= comp
        small.append({"person": k, "area": int(st[i, cv2.CC_STAT_AREA]), "dist_to_person_px": d[k], "bbox_xywh": [int(v) for v in st[i, :4]]})
    names3 = ["kiri", "tengah", "kanan"] if len(big) == 3 else [f"orang{k + 1}_dari_kiri" for k in range(len(big))]
    return [{"label": names3[k], "mask": masks[k], "big_area": int(st[big[k], cv2.CC_STAT_AREA]), "centroid_xy": [float(cen[big[k]][0]), float(cen[big[k]][1])],
             "big_bbox_xywh": [int(v) for v in st[big[k], :4]], "small": [s for s in small if s["person"] == k]} for k in range(len(big))]


def diag4() -> None:
    cfg = load_pipeline()
    names = tuple(g for g, _ in cfg.groups)
    clip = stb.load_clip(WORK)
    classes = jload(WORK / "seg" / "manifest.json")["classes"]
    lut = stb.class_group_lut(cfg.groups, classes)
    cmap, by_group = class_groups(cfg, classes)
    arm_cls = by_group["left_arm"] + by_group["right_arm"]
    cloth_cls = ["Upper_Clothing", "Apparel", "Torso", "Lower_Clothing"]
    leg_cls = by_group["left_leg"] + by_group["right_leg"]
    st_cfg = cfg.stabilize
    H, W = 1280, 720
    res: dict = {"mapping_class_to_group": {c: cmap.get(c, "background") for c in classes},
                 "stabilize_params": {"island_min_px": st_cfg.island_min_px, "mode_k": st_cfg.mode_k},
                 "frames": {}}
    for f in ARM_FRAMES:
        nm = clip.names[f]
        cm = cv2.imdecode(np.fromfile(WORK / "seg" / "classmap" / nm, np.uint8), cv2.IMREAD_UNCHANGED)
        R = lut[cm]
        ISL = stb.island_filter(R, st_cfg.island_min_px)
        MOD = stb.mode_filter(ISL, st_cfg.mode_k)
        S = stb.read_groups(clip.groups_path(nm))
        out = []
        for p in persons_of(cm, H, W):
            pm, region = p["mask"], cv2.dilate(p["mask"].astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
            h = np.bincount(cm[pm], minlength=len(classes))
            cls = lambda lst: {c: int(h[classes.index(c)]) for c in lst if h[classes.index(c)]}  # noqa: E731
            grp = {}
            for gi, g in enumerate(names, 1):
                grp[g] = {"raw": int((R[pm] == gi).sum()), "after_island": int((ISL[pm] == gi).sum()), "after_mode": int((MOD[pm] == gi).sum()),
                          "stable": int((S[region] == gi).sum())}
            ys, xs = np.nonzero(pm)
            legc = []
            for gname in ("left_leg", "right_leg"):
                gi = names.index(gname) + 1
                n, lab, st, _ = cv2.connectedComponentsWithStats(((R == gi) & pm).astype(np.uint8), connectivity=8)
                for i in range(1, n):
                    x, y, w_, h_ = (int(v) for v in st[i, :4])
                    legc.append({"group": gname, "area": int(st[i, cv2.CC_STAT_AREA]), "bbox_xywh": [x, y, w_, h_],
                                 "touches_frame_edge": bool(x == 0 or y == 0 or x + w_ >= W or y + h_ >= H)})
            out.append({"label": p["label"], "centroid_xy": p["centroid_xy"], "big_blob_bbox_xywh": p["big_bbox_xywh"], "big_blob_area": p["big_area"],
                        "assigned_area": int(pm.sum()), "assigned_bbox_xywh": [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)],
                        "small_components": p["small"], "raw_arm_class_px": cls(arm_cls), "raw_cloth_class_px": cls(cloth_cls), "raw_leg_class_px": cls(leg_cls),
                        "groups_px": grp, "leg_components_raw": legc,
                        "touches_frame_edge_left_right": bool(xs.min() == 0 or xs.max() >= W - 1)})
        res["frames"][f] = out
    # filter pulau / mode / temporal per grup di SELURUH klip (piksel yang berubah dari grup g)
    lost = {g: {"island": [], "mode": [], "temporal": [], "raw": []} for g in names}
    for i, nm in enumerate(clip.names):
        cm = cv2.imdecode(np.fromfile(WORK / "seg" / "classmap" / nm, np.uint8), cv2.IMREAD_UNCHANGED)
        R = lut[cm]
        ISL = stb.island_filter(R, st_cfg.island_min_px)
        MOD = stb.mode_filter(ISL, st_cfg.mode_k)
        S = stb.read_groups(clip.groups_path(nm))
        for gi, g in enumerate(names, 1):
            lost[g]["raw"].append(int((R == gi).sum()))
            lost[g]["island"].append(int(((R == gi) & (ISL != gi)).sum()))
            lost[g]["mode"].append(int(((ISL == gi) & (MOD != gi)).sum()))
            lost[g]["temporal"].append(int(((MOD == gi) & (S != gi)).sum()))

    def sm(a):
        a = np.asarray(a, float)
        return {"p50": float(np.median(a)), "p95": float(np.percentile(a, 95)), "max": float(a.max()), "frames_nonzero": int((a > 0).sum())}

    res["clip_removed_px_per_group"] = {g: {k: sm(v) for k, v in d.items()} for g, d in lost.items()}
    save("diag4.json", res)
    print(json.dumps({"mapping": res["mapping_class_to_group"]}, ensure_ascii=False)[:1500])
    for f, ps in res["frames"].items():
        for p in ps:
            print(f, p["label"], "bbox", p["assigned_bbox_xywh"], "arm", p["raw_arm_class_px"], "cloth", p["raw_cloth_class_px"], "leg", p["raw_leg_class_px"])
            print("    grp", {g: v for g, v in p["groups_px"].items() if g in ("left_arm", "right_arm", "left_leg", "right_leg")}, "legcomps", p["leg_components_raw"], "small", len(p["small_components"]))
    print(json.dumps(res["clip_removed_px_per_group"], indent=0)[:3500])


def run_stage_vec(cfgfile: Path, work: Path) -> str:
    import subprocess
    r = subprocess.run([sys.executable, "-m", "rotoscope.vectorize", "--config", str(cfgfile), "--work-dir", str(work), "--restart"], cwd=ROOT,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise SystemExit(f"vectorize gagal:\n{r.stdout[-1200:]}\n{r.stderr[-1200:]}")
    return r.stdout.strip().splitlines()[-1]


def lower_clothing_ok(clip_work: Path, docs_dir: Path, frames=(73, 78, 82, 87, 92), classes=None) -> list[int]:
    lc = classes.index("Lower_Clothing")
    ok = []
    for i in frames:
        cm = cv2.imdecode(np.fromfile(clip_work / "seg" / "classmap" / f"frame_{i:05d}.png", np.uint8), cv2.IMREAD_UNCHANGED)
        d = jload(docs_dir / f"frame_{i:05d}.json")
        for s in d["strokes"]:
            if s["type"] != "occlusion":
                continue
            pts = [(int(y), int(x)) for x, y in s["points"]]
            if sum(cm[y, x] == lc for y, x in pts) / len(pts) >= 0.8:
                ok.append(i)
                break
    return ok


def sim_metrics(contours: Path, clip_work: Path, classes, leg_check: bool) -> dict:
    fl = tm.stroke_lengths_from_dir(contours)
    pop = tm.pop_energy_by_type(fl, ("silhouette", "silhouette_hole", "group_boundary", "occlusion"))
    docs = [jload(p) for p in sorted(contours.glob("frame_*.json"))]
    occ = np.array([sum(s["type"] == "occlusion" for s in d["strokes"]) for d in docs])
    o = pop["types"]["occlusion"]
    cs = jload(contours / "clip_stats.json")
    out = {"occ_per_frame_mean": float(occ.mean()), "occ_per_frame_p50": float(np.median(occ)), "occ_total": int(occ.sum()),
           "frames_without_occ_pct": float(100 * (occ == 0).mean()), "age_mean": o["age_mean"], "age_median": o["age_median"],
           "pop_energy_occ": o["pop_energy_mean"], "id_new_per_stroke_frame": o["id_new_per_stroke_frame"],
           "pop_energy_total": pop["total_pop_energy_mean"], "t_high": cs["t_high"], "t_low": cs["t_low"]}
    if leg_check:
        ok = lower_clothing_ok(clip_work, contours, classes=classes)
        out["done_when_kaki"] = {"frames_ok": ok, "n_ok": len(ok), "of": 5}
    return out


SIM_GRID = [(95, 90), (97, 94), (98, 96)]
SIM_L = [30.0, 40.0, 50.0]


def diag5(args: list[str]) -> None:
    """diag5 folds -> statistik strok oklusi klip3; diag5 sim <klip> [indeks varian awal-akhir] -> simulasi hi/lo x L (scratch)."""
    import os
    cfg = load_pipeline()
    if args and args[0] == "sim":
        clip = args[1]
        lo_i, hi_i = (int(x) for x in args[2].split("-")) if len(args) > 2 else (0, 8)
        src = ROOT / "work" / "clips" / clip
        work = OUT / "sim" / clip
        if not (work / "stable").is_dir():
            (work).mkdir(parents=True, exist_ok=True)
            shutil.copytree(src / "stable", work / "stable", copy_function=os.link)      # hard link, hanya dibaca oleh [4]
            for fl in ("meta.json", "qc_report.json"):
                shutil.copyfile(src / fl, work / fl)
        classes = jload(src / "seg" / "manifest.json")["classes"]
        path = OUT / f"diag5_sim_{clip}.json"
        results = jload(path) if path.is_file() else {}
        variants = [(h, l, L) for (h, l) in SIM_GRID for L in SIM_L]
        for k, (h, l, L) in enumerate(variants):
            if not (lo_i <= k <= hi_i):
                continue
            key = f"hi{h}_lo{l}_L{int(L)}"
            cfgfile = OUT / "sim" / f"{clip}_{key}.yaml"
            cfgfile.write_text(f"vectorize:\n  depth_lines:\n    hi_pct: {h}\n    lo_pct: {l}\n    min_len_px: {L}\n", encoding="utf-8")
            run_stage_vec(cfgfile, work)
            results[key] = sim_metrics(work / "contours", src, classes, leg_check=(clip == "test"))
            print(clip, key, {a: (round(b, 4) if isinstance(b, float) else b) for a, b in results[key].items()}, flush=True)
            path.write_text(json.dumps(results, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
        return
    # folds: strok oklusi klip3 (produksi) -- kekuatan relatif T_high / T_low, panjang relatif L, umur track
    cs = jload(WORK / "contours" / "clip_stats.json")
    th, tl = cs["t_high"], cs["t_low"]
    mv = jload(WORK / "contours" / "manifest.json")["vectorize"]
    L = mv["depth_lines.min_len_px"]
    rows = []
    docs = [jload(p) for p in sorted((WORK / "contours").glob("frame_*.json"))]
    for i, d in enumerate(docs):
        for s in d["strokes"]:
            if s["type"] == "occlusion":
                p = np.asarray(s["points"], float)
                rows.append({"frame": i, "track": s["track_id"], "group": s["groups"][0], "strength": s["strength"], "n_points": len(p),
                             "length_px": float(np.hypot(*np.diff(p, axis=0).T).sum())})
    # umur = panjang run frame berurutan sebuah track_id
    by_track: dict[int, list[int]] = {}
    for r in rows:
        by_track.setdefault(r["track"], []).append(r["frame"])
    age_of = {}
    for t, fr in by_track.items():
        fr = sorted(set(fr))
        s0 = fr[0]
        for a, b in zip(fr, fr[1:] + [None]):
            if b is None or b != a + 1:
                for x in range(s0, a + 1):
                    age_of[(t, x)] = a - s0 + 1
                s0 = b
    for r in rows:
        r["age"] = age_of[(r["track"], r["frame"])]

    def q(a):
        a = np.asarray(a, float)
        return {"n": len(a), "p5": float(np.percentile(a, 5)), "p25": float(np.percentile(a, 25)), "p50": float(np.median(a)),
                "p75": float(np.percentile(a, 75)), "p95": float(np.percentile(a, 95)), "min": float(a.min()), "max": float(a.max())}

    young, old = [r for r in rows if r["age"] == 1], [r for r in rows if r["age"] >= 2]
    res = {"t_high": th, "t_low": tl, "L": L, "n_occlusion_strokes": len(rows), "groups": {g: sum(r["group"] == g for r in rows) for g in names_of(cfg)},
           "strength_over_t_high_all": q([r["strength"] / th for r in rows]), "strength_over_t_low_all": q([r["strength"] / tl for r in rows]),
           "n_points_over_L_all": q([r["n_points"] / L for r in rows]), "length_px_all": q([r["length_px"] for r in rows]),
           "age1": {"n": len(young), "strength_over_t_high": q([r["strength"] / th for r in young]), "n_points_over_L": q([r["n_points"] / L for r in young]),
                    "length_px": q([r["length_px"] for r in young])},
           "age_ge2": {"n": len(old), "strength_over_t_high": q([r["strength"] / th for r in old]), "n_points_over_L": q([r["n_points"] / L for r in old]),
                       "length_px": q([r["length_px"] for r in old])},
           "age_distribution_stroke_frames": {str(a): int(sum(r["age"] == a for r in rows)) for a in (1, 2, 3, 4, 5)} | {">=6": int(sum(r["age"] >= 6 for r in rows))},
           "share_n_points_lt_1.5L_age1": float(np.mean([r["n_points"] < 1.5 * L for r in young])), "share_n_points_lt_1.5L_age_ge2": float(np.mean([r["n_points"] < 1.5 * L for r in old])),
           "share_strength_lt_2_t_high_age1": float(np.mean([r["strength"] < 2 * th for r in young])), "share_strength_lt_2_t_high_age_ge2": float(np.mean([r["strength"] < 2 * th for r in old]))}
    from scipy.stats import mannwhitneyu
    res["mannwhitney_age1_vs_ge2"] = {"strength_p": float(mannwhitneyu([r["strength"] for r in young], [r["strength"] for r in old]).pvalue),
                                      "n_points_p": float(mannwhitneyu([r["n_points"] for r in young], [r["n_points"] for r in old]).pvalue)}
    save("diag5_folds.json", res)
    print(json.dumps(res, indent=1, ensure_ascii=False)[:5000])


def names_of(cfg):
    return [g for g, _ in cfg.groups]


def diag6() -> None:
    """Gambar cek cut: untuk tiap kandidat / cut: frame sumber t-1, t, t+1 berdampingan + skor selisih / ambang."""
    cfg = load_pipeline()
    clip = stb.load_clip(WORK)
    scores = stb.cut_scores([stb.frame_thumb(WORK / "frames" / nm) for nm in clip.names])
    cd = cfg.stabilize.temporal.cut_diff
    cands = [("tak_terdeteksi", 126), ("tak_terdeteksi", 154), ("tak_terdeteksi", 85), ("terdeteksi", 134), ("terdeteksi", 146), ("terdeteksi", 166), ("terdeteksi", 174)]
    outdir = ROOT / "work" / "klip3_review"
    outdir.mkdir(parents=True, exist_ok=True)
    pw = 520
    for k, (kind, f) in enumerate(cands, 1):
        panels = []
        for j in (f - 1, f, f + 1):
            im = cv2.imdecode(np.fromfile(WORK / "frames" / clip.names[j], np.uint8), cv2.IMREAD_COLOR)
            im = cv2.resize(im, (pw, round(im.shape[0] * pw / im.shape[1])), interpolation=cv2.INTER_AREA)
            sc = scores[j]
            tag = f"f{j}  skor {sc:.4f} = {sc / cd:.2f}x ambang"
            cv2.rectangle(im, (0, 0), (pw, 24), (255, 255, 255), -1)
            cv2.putText(im, tag, (4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            if j == f:
                cv2.rectangle(im, (0, 0), (pw - 1, im.shape[0] - 1), (0, 0, 255), 4)
            panels.append(im)
        img = np.hstack(panels)
        cv2.rectangle(img, (0, img.shape[0] - 26), (img.shape[1], img.shape[0]), (255, 255, 255), -1)
        cv2.putText(img, f"{kind.replace('_', ' ')}: kandidat f{f} (bingkai merah); ambang cut_diff {cd}", (6, img.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        ok, buf = cv2.imencode(".png", img)
        p = outdir / f"cek_cut_{k}_f{f}.png"
        p.write_bytes(buf.tobytes())
        print(p.name, img.shape[1], "x", img.shape[0])


if __name__ == "__main__":
    a = sys.argv[1:]
    cmds = {"1": diag1, "2": diag2, "3": diag3, "4": diag4, "6": diag6}
    if a and a[0] in cmds:
        cmds[a[0]]()
    elif a and a[0] == "5":
        diag5(a[1:])
    else:
        sys.exit(__doc__)
