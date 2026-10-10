"""Observasi Klip2 (alat sekali pakai; HANYA membaca work/clips/Klip2, tidak mengubah kode / default / klip lain):

    python scripts/klip2_review.py measure    # angka butir 3 (a)-(g) -> work/klip2_review/metrics.json + ringkasan di layar
    python scripts/klip2_review.py material   # bahan tontonan Rio: PNG frame kunci, video berdampingan, PANDUAN.md -> work/klip2_review/

Metrik memakai ulang tests/temporal_metrics.py, reach_metrics.py, multipass_metrics.py. Hasil = observasi, tanpa klaim visual."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "src"))

import multipass_metrics as mm  # noqa: E402
import reach_metrics as rm  # noqa: E402
import temporal_metrics as tm  # noqa: E402

from rotoscope import stabilize as stb  # noqa: E402
from rotoscope import stylize as sty  # noqa: E402
from rotoscope import vectorize as vec  # noqa: E402
from rotoscope.config import load_pipeline, load_style  # noqa: E402

CLIP = os.environ.get("KLIP", "Klip2")            # nama folder klip (peka huruf besar-kecil), mis. KLIP=klip3
WORK = ROOT / "work" / "clips" / CLIP
OUT = ROOT / "work" / f"{CLIP.lower()}_review"
D_PX = 7.0                                        # min_dist_px default (zona D), hanya untuk run zona di laporan
FAST_FALLBACK = 20                                # panjang jendela gerak tercepat bila aturan otomatis tak menemukan
STATIC_FALLBACK = 12


def jload(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def stats(a) -> dict:
    a = np.asarray(a, float)
    if not len(a):
        return {"min": None, "p50": None, "p95": None, "max": None}
    return {"min": float(a.min()), "p50": float(np.median(a)), "p95": float(np.percentile(a, 95)), "max": float(a.max())}


def names_and_groups():
    cfg = load_pipeline()
    return cfg, tuple(g for g, _ in cfg.groups)


def load_stable(clip, n_groups):
    return np.stack([stb.read_groups(clip.stable_clip.groups_path(n)) for n in clip.names])


def load_raw(clip, cfg):
    import cv2
    classes = jload(WORK / "seg" / "manifest.json")["classes"]
    lut = stb.class_group_lut(cfg.groups, classes)
    out = []
    for n in clip.names:
        cm = cv2.imdecode(np.fromfile(WORK / "seg" / "classmap" / n, np.uint8), cv2.IMREAD_UNCHANGED)
        out.append(lut[cm])
    return np.stack(out)


def pick_windows(raw: np.ndarray) -> dict:
    """Jendela otomatis aturan T-302 (dari argmax mentah); kalau kosong -> jendela fallback dari data (kecepatan centroid)."""
    sp = tm.centroid_speed(raw)
    cut_names = jload(WORK / "stable" / "manifest.json")["temporal"].get("cut_frames", [])
    cuts = [int(c.split("_")[1].split(".")[0]) for c in cut_names]         # cut antara k-1 dan k: lompatan centroid bukan gerak
    clean = lambda w: not any(w[0] < k <= w[1] for k in cuts)               # noqa: E731
    fast, static = [w for w in tm.fast_windows(raw) if clean(w)], [w for w in tm.static_windows(raw) if clean(w)]
    info = {"fast_auto": [list(w) for w in fast], "static_auto": [list(w) for w in static], "cut_frames": cuts}
    n = len(raw)
    if fast:
        info["fast"] = list(max(fast, key=lambda w: sp[w[0]:w[1] + 1].mean()))
        info["fast_source"] = "otomatis T-302"
    else:
        L = min(FAST_FALLBACK, n)
        s = int(np.argmax([sp[i:i + L].mean() for i in range(n - L + 1)]))
        info["fast"], info["fast_source"] = [s, s + L - 1], f"fallback: {L} frame dgn rata-rata kecepatan tertinggi"
    if static and max(w[1] - w[0] + 1 for w in static) >= STATIC_FALLBACK:
        a, b = max(static, key=lambda w: w[1] - w[0])
        s = min(range(a, b - STATIC_FALLBACK + 2), key=lambda i: sp[max(i, 1):i + STATIC_FALLBACK].mean())
        info["static"] = [s, s + STATIC_FALLBACK - 1]
        info["static_source"] = f"otomatis T-302 (jendela terpanjang {a}-{b}), sub-jendela {STATIC_FALLBACK} frame dgn kecepatan terendah"
    elif static:
        info["static"] = list(max(static, key=lambda w: w[1] - w[0]))
        info["static_source"] = "otomatis T-302 (jendela terpanjang < 12 frame)"
    else:
        L = min(STATIC_FALLBACK, n)
        s = int(np.argmin([sp[max(i, 1):i + L].mean() for i in range(n - L + 1)]))
        info["static"], info["static_source"] = [s, s + L - 1], f"fallback: {L} frame dgn rata-rata kecepatan terendah"
    return info


def cmd_measure() -> None:
    cfg, names = names_and_groups()
    clip = vec.load_clip(WORK)
    n = len(clip.names)
    res: dict = {"n_frames": n}

    # (a) QC
    q = jload(WORK / "qc_report.json")
    fr = q["frames"]
    avm = np.array([f["area_vs_median"] for f in fr])
    iou = np.array([f["iou_prev"] for f in fr[1:]], float)
    blobs = np.array([f["big_blobs"] for f in fr])
    area = np.array([f["area_ratio"] for f in fr])
    res["qc"] = {"n_fail": q["summary"]["n_fail"], "fail_counts": q["summary"]["fail_counts"],
                 "area_vs_median": stats(avm), "area_vs_median_margin_min": float(avm.min() - q["qc"]["area_drop_min"]),
                 "worst_area_vs_median_frame": int(avm.argmin()), "iou_prev": stats(iou),
                 "iou_prev_margin_min": float(iou.min() - q["qc"]["iou_min"]), "worst_iou_prev_frame": int(iou.argmin()) + 1,
                 "big_blobs_max": int(blobs.max()), "area_ratio": stats(area), "area_ratio_limits": [q["qc"]["area_min"], q["qc"]["area_max"]]}

    # (b) segmentasi: piksel per grup per frame (peta grup stabil + argmax mentah)
    stable, raw = load_stable(clip, len(names)), load_raw(clip, cfg)
    hw = stable.shape[1] * stable.shape[2]
    for label, g in (("stable", stable), ("raw", raw)):
        per = {}
        for gid, nm in enumerate(names, 1):
            c = (g == gid).sum(axis=(1, 2))
            per[nm] = {**stats(c), "frames_empty": int((c == 0).sum())}
        fg = (g != 0).sum(axis=(1, 2)) / hw
        res[f"groups_{label}"] = {"px_per_group": per, "fg_fraction": stats(fg)}
    hair = (stable == names.index("hair") + 1).sum(axis=(1, 2))
    res["hair_max_frame"] = int(hair.argmax())

    # (c) kedalaman
    dsm = []
    for i, nme in enumerate(clip.names):
        d = np.load(clip.stable_clip.depth_smooth_path(nme)).astype(np.float32)
        v = d[stable[i] != 0]
        dsm.append([float(v.min()), float(np.percentile(v, 5)), float(np.median(v)), float(np.percentile(v, 95)), float(v.max()), float(v.std())])
    dsm = np.array(dsm)
    cs = jload(WORK / "contours" / "clip_stats.json")
    res["depth"] = {"depth_smooth_fg_per_frame": {k: stats(dsm[:, j]) for j, k in enumerate(["min", "p5", "median", "p95", "max", "std"])},
                    "t_high": cs["t_high"], "t_low": cs["t_low"], "test_ref": {"t_high": 0.1389, "t_low": 0.0641}}

    # (d) stabilize
    scores = stb.cut_scores([stb.frame_thumb(WORK / "frames" / nme) for nme in clip.names])
    sp_raw, sp_st = tm.centroid_speed(raw), tm.centroid_speed(stable)
    win = pick_windows(raw)
    res["stabilize"] = {"cut_score_max": float(max(scores)), "cut_score_max_frame": int(np.argmax(scores)), "cut_diff": cfg.stabilize.temporal.cut_diff,
                        "n_cuts": int(sum(s > cfg.stabilize.temporal.cut_diff for s in scores)),
                        "centroid_speed_raw": {**stats(sp_raw[1:]), "p50": float(np.median(sp_raw[1:])), "argmax_frame": int(sp_raw.argmax())},
                        "centroid_speed_stable": stats(sp_st[1:]), "test_ref_fast_speed": 7.3,
                        "iou_prev_stable": {**stats(tm.iou_prev(stable)[1:]), "argmin_frame": int(np.argmin(tm.iou_prev(stable)[1:])) + 1},
                        "iou_prev_raw": stats(tm.iou_prev(raw)[1:]), "windows": win,
                        "flipflop_per10k": {"raw_all": tm.flipflop_per10k(raw), "stable_all": tm.flipflop_per10k(stable)}}
    for nm in ("fast", "static"):
        a, b = win[nm]
        idx = tm.window_mask(n, [(a, b)])
        res["stabilize"]["flipflop_per10k"][f"raw_{nm}"] = tm.flipflop_per10k(raw, idx)
        res["stabilize"]["flipflop_per10k"][f"stable_{nm}"] = tm.flipflop_per10k(stable, idx)
    res["stabilize"]["fg_iou_stable_vs_raw"] = stats(tm.fg_iou(stable, raw))
    res["stabilize"]["limb_ratio_arms"] = stats(tm.limb_ratio(stable, raw))

    # (e) vectorize
    docs = [jload(WORK / "contours" / f"frame_{i:05d}.json") for i in range(n)]
    types = ("silhouette", "silhouette_hole", "group_boundary", "occlusion")
    per_type = {t: np.array([sum(s["type"] == t for s in d["strokes"]) for d in docs]) for t in types}
    occ_n = per_type["occlusion"]
    mv = jload(WORK / "contours" / "manifest.json")["vectorize"]
    th, tl = cs["t_high"], cs["t_low"]
    hair_dropped, gaps_all = [], []
    occ_by_group: dict[str, int] = {}
    for i, nme in enumerate(clip.names):
        gmap, depth = vec.load_frame_inputs(clip, nme, len(names))
        incl = vec.occlusion_strokes(gmap, depth, names, {**mv, "depth_lines.exclude_groups": []}, th, tl)[0]
        hair_dropped.append(sum(s["groups"][0] == "hair" for s in incl))
        for s in incl:
            occ_by_group[s["groups"][0]] = occ_by_group.get(s["groups"][0], 0) + 1
        others = vec.vectorize_gmap(gmap, names, mv)[0]
        kept = [s for s in docs[i]["strokes"] if s["type"] == "occlusion"]
        gaps_all += rm.end_gaps(kept, others, clip.width, clip.height)
    frames_len = tm.stroke_lengths_from_dir(WORK / "contours")
    pop = tm.pop_energy_by_type(frames_len, types)
    res["vectorize"] = {"strokes_per_type_per_frame": {t: {**stats(per_type[t]), "mean": float(per_type[t].mean())} for t in types},
                        "frames_without_occlusion": int((occ_n == 0).sum()),
                        "occlusion_strokes_total": int(occ_n.sum()), "occlusion_frame_most": int(occ_n.argmax()),
                        "hair_occlusion_dropped_by_exclude": {"total": int(sum(hair_dropped)), "per_frame": stats(hair_dropped),
                                                              "frames_with_any": int((np.array(hair_dropped) > 0).sum())},
                        "occlusion_before_exclude_by_group": occ_by_group,
                        "end_gap_histogram_px_ref": rm.gap_histogram(gaps_all, scale=1080 / clip.width),
                        "end_gap_histogram_px_work": rm.gap_histogram(gaps_all), "n_open_ends": len(gaps_all),
                        "pop_energy": pop, "legs_nonempty": {nm: bool((stable == names.index(nm) + 1).any()) for nm in ("left_leg", "right_leg")}}

    # (f) stylize / export
    style = load_style(None)
    meta = jload(WORK / "meta.json")
    g = sty.make_geometry(style, int(meta["working_width"]), int(meta["working_height"]))
    acc = {"joint_ratio": [], "joint_over_tol": 0, "new_cross": [], "seam_fail": [], "seam_rel_fail": [], "edge_ink": [], "min_det": [],
           "intrusion": [], "disp": []}
    for d in docs:
        passes, _ = sty.frame_passes(d, g)
        for r in mm.pass_invariants(passes, g, int(d["frame_index"])):
            fi = d["frame_index"]
            acc["joint_ratio"].append((fi, r["joint_ratio"])), acc["new_cross"].append((fi, r["new_crossings"]))
            acc["seam_fail"].append((fi, r["seam_fail"])), acc["seam_rel_fail"].append((fi, r["seam_rel_fail"]))
            acc["edge_ink"].append((fi, r["edge_ink_changed"])), acc["min_det"].append((fi, r["min_det"]))
            acc["intrusion"].append((fi, r["intrusion_excess"])), acc["disp"].append((fi, r["max_displacement_over_bound"]))
    sr = [json.loads(x) for x in (WORK / "strokes" / "frames.jsonl").read_text(encoding="utf-8").splitlines()]
    sr = [r for r in sr if r.get("event") == "frame"]
    tt = np.array([r["total_s"] for r in sr])
    sdir = WORK / "strokes"
    pngs = [p.stat().st_size for p in sorted(sdir.glob("frame_*.png"))]
    svgs = [p.stat().st_size for p in sorted(sdir.glob("frame_*.svg"))]
    mp4 = ROOT / "out" / f"{CLIP}.mp4"
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_name,width,height,r_frame_rate,nb_frames,pix_fmt,color_space:format=duration,size",
                            "-of", "json", str(mp4)], capture_output=True, text=True).stdout
    mx = lambda k: max(v for _, v in acc[k])  # noqa: E731
    bad = lambda k, pred: [fi for fi, v in acc[k] if pred(v)]  # noqa: E731
    res["stylize"] = {"time_per_frame_s": {"mean": float(tt.mean()), "p95": float(np.percentile(tt, 95)), "max": float(tt.max())},
                      "png_kib": {"mean": float(np.mean(pngs) / 1024), "max": float(max(pngs) / 1024)}, "svg_kib": {"mean": float(np.mean(svgs) / 1024), "max": float(max(svgs) / 1024)},
                      "mp4_bytes": mp4.stat().st_size, "ffprobe": json.loads(probe) if probe else None,
                      "invariants_pass_frames": len(acc["joint_ratio"]),
                      "joint_ratio_max": mx("joint_ratio"), "frames_joint_ratio_gt_1": bad("joint_ratio", lambda v: v > 1.0),
                      "new_crossings_total": int(sum(v for _, v in acc["new_cross"])), "frames_new_crossings": bad("new_cross", lambda v: v > 0),
                      "seam_fail_total": int(sum(v for _, v in acc["seam_fail"])), "frames_seam_fail": bad("seam_fail", lambda v: v > 0),
                      "seam_rel_fail_total": int(sum(v for _, v in acc["seam_rel_fail"])), "frames_seam_rel_fail": bad("seam_rel_fail", lambda v: v > 0),
                      "edge_ink_total": int(sum(v for _, v in acc["edge_ink"])), "frames_edge_ink": bad("edge_ink", lambda v: v > 0),
                      "intrusion_frames": bad("intrusion", lambda v: v > 1e-9),
                      "min_det_min": min(v for _, v in acc["min_det"]), "frames_min_det_le_0.05": bad("min_det", lambda v: v <= sty.JACOBIAN_MIN_DET),
                      "max_disp_over_bound_max": mx("disp"),
                      "n_strokes_per_frame": stats([r["n_strokes"] for r in sr]), "n_points_per_frame": stats([r["n_points"] for r in sr])}
    # (g) waktu stage dari log run
    OUT.mkdir(parents=True, exist_ok=True)
    times = OUT / "stage_times.json"                  # ditulis tangan dari baris "Waktu per stage" log run
    res["stage_times_s"] = jload(times) if times.is_file() else {}
    fd = []                                           # fraksi piksel foreground dengan disparity mentah <= 0 (stage [2c])
    for i, nme in enumerate(clip.names):
        d = np.load(WORK / "depth" / nme.replace(".png", ".npy")).astype(np.float32)
        fd.append(float((d[stable[i] != 0] <= 0).mean()))
    res["depth"]["fg_disparity_le_0_fraction"] = stats(fd)
    res["depth"]["fg_disparity_le_0_frames_nonzero"] = int(sum(f > 0 for f in fd))
    (OUT / "metrics.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(res, indent=1, ensure_ascii=False, default=str))


GROUP_COLORS_BGR = {0: (235, 235, 235), 1: (60, 60, 200), 2: (80, 170, 240), 3: (90, 160, 60), 4: (200, 120, 40), 5: (170, 60, 170),
                    6: (200, 200, 40), 7: (40, 120, 120)}      # background, hair, face, torso, left_arm, right_arm, left_leg, right_leg
PANEL_W = 540


def read_png(path: Path):
    import cv2
    return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)


def panel(img, title: str):
    import cv2
    h = round(img.shape[0] * PANEL_W / img.shape[1])
    im = cv2.resize(img, (PANEL_W, h), interpolation=cv2.INTER_AREA).copy()
    cv2.rectangle(im, (0, 0), (PANEL_W, 20), (255, 255, 255), -1)
    cv2.putText(im, title, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return im


def color_groups(g: np.ndarray):
    out = np.zeros(g.shape + (3,), np.uint8)
    for gid, bgr in GROUP_COLORS_BGR.items():
        out[g == gid] = bgr
    return out


def triple(i: int, clip, stable, label: str):
    import cv2
    nme = clip.names[i]
    src, res = read_png(WORK / "frames" / nme), read_png(WORK / "strokes" / nme)
    pa, pb, pc = panel(src, f"sumber f{i}"), panel(color_groups(stable[i]), "grup (hair merah, face oranye, torso hijau, L-arm biru, R-arm ungu)"), panel(res, label)
    h = max(p.shape[0] for p in (pa, pb, pc))
    pad = lambda p: cv2.copyMakeBorder(p, 0, h - p.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))  # noqa: E731
    return np.hstack([pad(pa), pad(pb), pad(pc)])


def side_by_side_video(clip, a: int, b: int, path: Path) -> None:
    import cv2
    pipe = None
    for i in range(a, b + 1):
        nme = clip.names[i]
        left, right = read_png(WORK / "frames" / nme), read_png(WORK / "strokes" / nme)
        h = 480
        left = cv2.resize(left, (round(left.shape[1] * h / left.shape[0]) // 2 * 2, h), interpolation=cv2.INTER_AREA)
        right = cv2.resize(right, (round(right.shape[1] * h / right.shape[0]) // 2 * 2, h), interpolation=cv2.INTER_AREA)
        cv2.putText(left, f"f{i}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        fr = np.hstack([left, right])
        if pipe is None:
            pipe = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{fr.shape[1]}x{fr.shape[0]}", "-r", "24",
                                     "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(path)], stdin=subprocess.PIPE)
        pipe.stdin.write(np.ascontiguousarray(fr).tobytes())
    pipe.stdin.close()
    pipe.wait()


def cmd_material() -> None:
    import shutil

    import cv2
    cfg, names = names_and_groups()
    clip = vec.load_clip(WORK)
    m = jload(OUT / "metrics.json")
    stable = load_stable(clip, len(names))
    raw = load_raw(clip, cfg)
    sp = tm.centroid_speed(raw)
    q = jload(WORK / "qc_report.json")["frames"]
    docs = [jload(WORK / "contours" / f"frame_{i:05d}.json") for i in range(len(clip.names))]
    occ = [sum(s["type"] == "occlusion" for s in d["strokes"]) for d in docs]
    hair = (stable == names.index("hair") + 1).sum(axis=(1, 2))
    win = m["stabilize"]["windows"]
    a_s, b_s = win["static"]
    rng = np.random.default_rng(7)
    rnd = sorted(int(x) for x in rng.choice(np.arange(a_s, b_s + 1), 2, replace=False))
    keys = [("A1_gerak_tercepat", int(sp.argmax()), "kecepatan centroid tertinggi"),
            ("A2_qc_terburuk", int(np.argmin([f["area_vs_median"] for f in q])), "area_vs_median terendah"),
            ("A3_oklusi_terbanyak", int(np.argmax(occ)), "strok oklusi terbanyak"),
            ("A4_hair_terbesar", int(hair.argmax()), "grup hair terbesar"),
            ("A5_statis_acak1", rnd[0], "frame statis acak"), ("A6_statis_acak2", rnd[1], "frame statis acak")]
    failing = [f for f in q if f["fail_reasons"]]
    if failing:
        top = max(failing, key=lambda f: f["big_blobs"])
        keys.append(("A7_qc_gagal", int(top["index"]), f"frame gagal QC dengan blob terbesar terbanyak ({top['big_blobs']}); alasan {top['fail_reasons']}"))
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for tag, i, why in keys:
        fn = f"{tag}_f{i:03d}.png"
        ok, buf = cv2.imencode(".png", triple(i, clip, stable, f"hasil strokes default f{i}"))
        (OUT / fn).write_bytes(buf.tobytes())
        rows.append((tag, fn, i, why))
    fa, fb = win["fast"]
    vids = [("B1_gerak_cepat", f"B1_gerak_cepat_f{fa}-{fb}.mp4", fa, fb), ("B2_statis", f"B2_statis_f{a_s}-{b_s}.mp4", a_s, b_s)]
    for _, fn, a, b in vids:
        side_by_side_video(clip, a, b, OUT / fn)
    shutil.copyfile(ROOT / "out" / f"{CLIP}.mp4", OUT / f"B3_hasil_penuh_{CLIP}.mp4")
    lines = [f"# {CLIP} — panduan tontonan", "", "Semua berkas di folder ini. Nomor frame = indeks 0-based (frame_00098.png = f98). "
             f"Klip: {len(clip.names)} frame kerja {clip.width}x{clip.height}, 24 fps.", "",
             "## A. Frame kunci (kiri: sumber | tengah: peta grup | kanan: hasil strokes default)", "",
             "| No | Berkas | Frame | Yang dilihat |", "|---|---|---|---|"]
    look = {"A1": "garis menempel pada tepi tubuh saat gerak paling cepat; lengan / kaki hilang atau bocor",
            "A2": "frame dengan luas foreground paling menyimpang dari median (status QC: metrics.json); tubuh terpotong?",
            "A3": "garis oklusi (hanya torso): menempel atau melayang / putus",
            "A4": f"rambut: hair {int(hair.max())} px; apakah rambut punya garis (exclude_groups [hair] membuang garis OKLUSI di rambut; siluet / batas grup tetap)",
            "A5": "frame statis: kualitas garis dasar", "A6": "frame statis: kualitas garis dasar",
            "A7": "frame gagal QC: peta grup (tengah) apakah ada objek / orang kedua; bentuk tubuh utama"}
    for tag, fn, i, why in rows:
        lines.append(f"| {tag.split('_')[0]} | `{fn}` | {i} | {look[tag.split('_')[0]]} — {why} |")
    lines += ["", "## B. Video", "", "| No | Berkas | Isi | Yang dilihat |", "|---|---|---|---|"]
    lines.append(f"| B1 | `{vids[0][1]}` | [sumber | hasil], {fb - fa + 1} frame, gerak tercepat ({win['fast_source']}) | kedipan, garis putus saat gerak, tubuh terpotong |")
    lines.append(f"| B2 | `{vids[1][1]}` | [sumber | hasil], {b_s - a_s + 1} frame, statis ({win['static_source']}) | getar / kedipan saat diam |")
    lines.append(f"| B3 | `B3_hasil_penuh_{CLIP}.mp4` | salinan out/{CLIP}.mp4 (hasil penuh, {len(clip.names)} frame, tanpa audio) | kesan keseluruhan |")
    cfiles = sorted(p.name for p in OUT.glob("C_*"))
    if cfiles:                                        # dibuat scripts/klip_scale.py compare (uji skala)
        lines += ["", "## C. Default vs skala-ekuivalen (uji skala; kiri: hasil default, kanan: [3]+[4] dengan parameter px diskalakan)", "",
                  "Kandidat, BUKAN keputusan; klip3 tidak punya pasangan (s = 0,671 ≤ 1,2, eksperimen dilewati).", "",
                  "| Berkas | Yang dilihat |", "|---|---|"] + [f"| `{n}` | garis dalam (oklusi) lebih banyak / wajar atau berantakan; kaki / lengan tetap ada; kedipan |" for n in cfiles]
    lines += ["", "## D. Template penilaian Rio", "", "Skala saran: baik / cukup / buruk, + catatan nomor frame.", "",
              "| Aspek | Nilai | Catatan (nomor frame) |", "|---|---|---|",
              "| Bentuk tubuh (siluet utuh, orang kedua / objek lain) | | |", "| Garis menempel pada tepi tubuh | | |",
              "| Garis putus / terpotong | | |", "| Garis dalam terlalu sedikit | | |", "| Rambut (garis, hilang, bocor) | | |",
              "| Lengan hilang atau bocor | | |", "| Kaki hilang atau bocor | | |", "| Tubuh terpotong | | |",
              "| Kedipan (pop garis antar frame) | | |", "| Subjek kecil (Klip2) | | |",
              "| Default vs skala-ekuivalen (C): mana lebih baik, kenapa | | |",
              "| Kesan keseluruhan (gesture drawing, kasar, bukan edge detection) | | |", ""]
    (OUT / "PANDUAN.md").write_text("\n".join(lines), encoding="utf-8")
    for r in rows:
        print(r)
    print("fast", win["fast"], "static", win["static"])


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "measure":
        cmd_measure()
    elif cmd == "material":
        cmd_material()
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
