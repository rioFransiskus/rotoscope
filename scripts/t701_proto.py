"""T-701 Tahap 3: prototipe I1 (kapsul pose → relabel Upper_Clothing → group_boundary) di SALINAN scratch.
docs/04 "Keputusan T-701 setelah Tahap 2" + "Desain prototipe I1 (Tahap 3)". Klip asli dan out/ hanya DIBACA: salinan per varian
= meta.json + qc_report.json + stable/ (groups DISALIN, depth_smooth di-hard-link); stage [4]–[6] menulis ke salinan.

    python scripts/t701_proto.py windows          # verifikasi / pilih jendela klip3 (f0–f47) dan test (12 frame) -> proto/windows.json
    python scripts/t701_proto.py pose [fp16]      # P2 crop per orang (desain 1c) di jendela; fp32 / fp16 otomatis -> proto/pose.json
    python scripts/t701_proto.py lr               # konvensi kiri / kanan pose vs seg (pergelangan vs kulit lengan bawah / tangan) -> proto/lr.json
    python scripts/t701_proto.py arm              # lacak + haluskan + kapsul R1 / R2 + relabel + masker tutup bahu -> proto/<var>/<klip>/
    python scripts/t701_proto.py render           # [4] (R1 / R2: patch tutup bahu) + [5] + [6] --from K --limit N per varian x klip
    python scripts/t701_proto.py metrics          # metrik per varian -> proto/metrics.json
    python scripts/t701_proto.py media            # MP4 berdampingan (normal 3x, lambat 0,5x) + papan 12 frame -> work/t701/proto/
    python scripts/t701_proto.py vectorize VAR KLIP   # (dipanggil `render`) [4] in-process; VAR != base -> patch tutup bahu
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import t701_study as st  # noqa: E402  (menambah tests/ + src/ + scripts/ ke sys.path)
import temporal_metrics as tm  # noqa: E402

from rotoscope import stabilize as stb  # noqa: E402
from rotoscope.config import load_pipeline  # noqa: E402

ROOT = st.ROOT
SCR = st.SCRATCH / "proto"
OUTP = st.OUT / "proto"
CLIPS = ("klip3", "test")
VARIANTS = ("base", "R1", "R2")

# ── nilai desain (docs/04 "Desain prototipe I1 (Tahap 3)", ditetapkan sebelum run) ──
KLIP3_PROPOSAL = (24, 47)          # usulan Rio; dipakai bila lolos verifikasi `windows`
KLIP3_RANGE = (0, 47)              # jendela klip3 di dalam f0–f47 (f49 = cut [3])
KLIP3_LEN = 24
TEST_LEN = 12
SCORE_MIN = st.P2_SCORE_MIN        # titik valid = skor >= 0,3
ARM_MIN_OK = st.ARM_MIN_OK         # orang = crop dengan >= 4 / 6 titik lengan valid
HOLD_MAX = 2                       # titik tidak valid: pakai titik valid terakhir maks 2 frame, sesudah itu segmen dihapus
TRACK_MATCH_FRAC = st.JITTER_MATCH_FRAC   # cocok antar frame: jarak titik tengah bahu <= 0,15 x tinggi kotak orang
R1_FRAC = 0.17                     # R1: r = 0,17 x jarak bahu kiri–kanan (titik halus)
R2_DMAX_FRAC = 3.0                 # R2: sinar tegak lurus sampai 3 x r_R1
R2_STEP_PX = 4.0                   # jarak sampel sepanjang sumbu segmen
R2_T_RANGE = (0.15, 0.85)          # bagian segmen yang disampel (dekat sendi tidak ikut)
R2_MIN_HITS = 3                    # < 3 sinar mengenai background -> R2 tidak terukur -> R1
CAP_MARGIN_PX = 2                  # tutup bahu = cakram (r + 2 px) di sekitar titik bahu halus
LR_RADIUS_FRAC = 0.02              # konvensi kiri / kanan: cakram 0,02 x tinggi frame di sekitar pergelangan
LR_CONSISTENT = 0.9                # >= 90% suara yang menentukan searah = konvensi konsisten
NEW_STROKE_DIST_PX = 2.0           # strok group_boundary "baru" = >= 50% titik berjarak <= 2 px dari piksel relabel
NEW_STROKE_FRAC = 0.5
BOARD_PER_WINDOW = 6
SEED_BOARD_PROTO = 701004
ARM_KP = (5, 6, 7, 8, 62, 41)
SEGMENTS = (("L_atas", 5, 7), ("L_bawah", 7, 62), ("R_atas", 6, 8), ("R_bawah", 8, 41))   # kiri / kanan = penamaan pose
PANEL_W, PANEL_H, GAP, HEAD = 540, 960, 12, 90


def jload(p: Path):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def jsave(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"  -> {p.relative_to(ROOT).as_posix()}")


def vdir(var: str, clip: str) -> Path:
    return SCR / var / clip


def window(clip: str) -> tuple[int, int]:
    a, b = jload(SCR / "windows.json")[clip]["window"]
    return int(a), int(b)


def group_ids() -> dict:
    names = tuple(g for g, _ in load_pipeline().groups)
    return {g: names.index(g) + 1 for g in ("torso", "left_arm", "right_arm")}


# ── jendela ────────────────────────────────────────
def cmd_windows() -> None:
    cfg = load_pipeline()
    k3, kt = st.ClipData("klip3", cfg), st.ClipData("test", cfg)
    res: dict = {"params": {"klip3_proposal": KLIP3_PROPOSAL, "klip3_range": KLIP3_RANGE, "klip3_len": KLIP3_LEN,
                            "test_len": TEST_LEN, "arm_front_min": st.ARM_FRONT_MIN}}
    rows = {}
    for f in range(KLIP3_RANGE[0], KLIP3_RANGE[1] + 1):
        blobs, faces, faces1, parts = st.frame_parts(k3, f)
        sl = st.sleeve_proxy(k3, k3.classmap(f), blobs)
        rows[f] = {"v1": len(faces1), "blobs": len(blobs), "split": sum(p["split"] for p in parts),
                   "sleeve": sum(s["sleeve_stuck_proxy"] for s in sl)}
    cands = []
    for s in range(KLIP3_RANGE[0], KLIP3_RANGE[1] - KLIP3_LEN + 2):
        e = s + KLIP3_LEN - 1
        fr = [rows[f] for f in range(s, e + 1)]
        cands.append({"window": [s, e], "cuts_inside": [c for c in k3.cuts if s < c <= e],
                      "frames_v1_3": sum(r["v1"] == 3 for r in fr), "frames_split": sum(r["split"] > 0 for r in fr),
                      "frames_sleeve": sum(r["sleeve"] >= 1 for r in fr), "sleeve_sum": sum(r["sleeve"] for r in fr)})
    prop = next(c for c in cands if c["window"] == list(KLIP3_PROPOSAL))
    ok = not prop["cuts_inside"] and prop["frames_v1_3"] == KLIP3_LEN and prop["frames_sleeve"] == KLIP3_LEN
    pick = prop if ok else max((c for c in cands if not c["cuts_inside"]),
                               key=lambda c: (c["frames_v1_3"], c["frames_sleeve"], c["sleeve_sum"], -c["window"][0]))
    res["klip3"] = {"window": pick["window"], "proposal_ok": ok, "proposal": prop, "cuts": k3.cuts, "f49_is_cut": 49 in k3.cuts,
                    "rows": rows, "candidates": cands}
    tsamp = jload(st.SCRATCH / "samples_1b.json")["samples"]["test"]
    scores = {f: st.arm_front_score(kt, kt.lut[kt.classmap(f)])["score"] for f in range(kt.n)}
    tc = []
    for s in range(0, kt.n - TEST_LEN + 1):
        e = s + TEST_LEN - 1
        inc = [x for x in tsamp if s <= x <= e]
        if not inc or any(s < c <= e for c in kt.cuts):
            continue
        sc = [scores[f] for f in range(s, e + 1)]
        tc.append({"window": [s, e], "samples": inc, "frames_ge_min": sum(v >= st.ARM_FRONT_MIN for v in sc),
                   "mean": float(np.mean(sc)), "min": float(np.min(sc))})
    tp = max(tc, key=lambda c: (c["frames_ge_min"], c["mean"], -c["window"][0]))
    res["test"] = {"window": tp["window"], "pick": tp, "cuts": kt.cuts,
                   "scores": {f: round(scores[f], 4) for f in range(tp["window"][0], tp["window"][1] + 1)},
                   "best_per_sample": {x: max((c for c in tc if x in c["samples"]), key=lambda c: (c["frames_ge_min"], c["mean"]))
                                       for x in tsamp if any(x in c["samples"] for c in tc)}}
    jsave(SCR / "windows.json", res)
    print(f"klip3 cut {k3.cuts}; f49 cut: {49 in k3.cuts}; usulan {KLIP3_PROPOSAL}: {prop} -> lolos {ok}; dipakai {pick['window']}")
    for f in range(pick["window"][0], pick["window"][1] + 1):
        print(f"  klip3 f{f}: {rows[f]}")
    print(f"test cut {kt.cuts}; dipakai {tp}")
    for x, c in res["test"]["best_per_sample"].items():
        print(f"  sampel test f{x}: terbaik {c}")


# ── pose ───────────────────────────────────────────
def cmd_pose(force: str | None) -> None:
    import torch

    cfg = load_pipeline()
    cds = {c: st.ClipData(c, cfg) for c in CLIPS}
    if not torch.cuda.is_available():
        raise SystemExit("CUDA tidak tersedia — butuh GPU NVIDIA lewat PyTorch cu118")
    free = torch.cuda.mem_get_info()[0] / st.MIB
    prec = force or ("fp32" if free >= st.VRAM_NEED_MIB["p2_fp32"] else "fp16" if free >= st.VRAM_NEED_MIB["p2_fp16"] else None)
    if prec is None or free < st.VRAM_NEED_MIB[f"p2_{prec}"]:
        raise SystemExit(f"VRAM bebas {free:.0f} MiB < {st.VRAM_NEED_MIB['p2_fp16']} MiB — tutup aplikasi lain yang memakai GPU "
                         f"(model belum dimuat)")
    dtype = torch.float32 if prec == "fp32" else torch.float16
    t0 = time.perf_counter()
    proc, model = st.hf_load("pose", "Sapiens2ForPoseEstimation", dtype, "cuda")
    res: dict = {"precision": prec, "vram_free_before_load_mib": round(free, 1), "load_s": round(time.perf_counter() - t0, 2),
                 "vram_free_after_load_mib": round(torch.cuda.mem_get_info()[0] / st.MIB, 1), "kept_indices": list(st.P2_KEEP),
                 "oom": None, "frames": {}}
    keep = torch.tensor(st.P2_KEEP, device="cuda")
    for c in CLIPS:
        cd = cds[c]
        a, b = window(c)
        for f in range(a, b + 1):
            rgb = st.read_rgb(cd, f)
            _, _, faces1, parts = st.frame_parts(cd, f)
            torch.cuda.reset_peak_memory_stats()
            rows = []
            try:
                for i, p in enumerate(parts):
                    row = {"part": i, **{k: v for k, v in p.items() if k != "mask"}}
                    if p["bbox_xywh"] is not None:
                        row.update(st.p2_infer(proc, model, rgb, [float(v) for v in p["bbox_xywh"]], dtype, keep))
                    rows.append(row)
            except torch.cuda.OutOfMemoryError:
                res["oom"] = {"frame": f"{c}:{f}", "reserved_mib": round(torch.cuda.memory_reserved() / st.MIB, 1)}
                jsave(SCR / f"pose_{prec}_oom.json", res)
                raise SystemExit(f"OOM {prec} di {c}:{f} — keputusan 5: SELURUH run diulang dengan `pose fp16`")
            tf = sum(r.get("time_s", 0.0) for r in rows)
            res["frames"][f"{c}:{f}"] = {"time_s": tf, "n_v1": len(faces1), "parts": rows,
                                         "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / st.MIB, 1)}
            print(f"  {c}:{f} kotak {len(parts)} V1 {len(faces1)} {tf:.2f} s, puncak {res['frames'][f'{c}:{f}']['peak_reserved_mib']} MiB, "
                  f"NaN {sum(r.get('nan', 0) for r in rows)}", flush=True)
    jsave(SCR / "pose.json", res)


def persons_at(cd: st.ClipData, f: int, prow: dict) -> tuple[list[dict], list[dict]]:
    """(orang, bagian crop) frame f: orang = bagian crop (wajah V1 atau tidak) yang posenya >= 4 / 6 titik lengan skor >= 0,3."""
    _, _, _, parts = st.frame_parts(cd, f)
    if [p["bbox_xywh"] for p in parts] != [r["bbox_xywh"] for r in prow["parts"]]:
        raise SystemExit(f"{cd.name} f{f}: kotak crop dihitung ulang != pose.json")
    out = []
    for i, r in enumerate(prow["parts"]):
        if "kp" not in r:
            continue
        d = {k[0]: k for k in r["kp"]}
        if sum(d[j][3] >= SCORE_MIN for j in ARM_KP) < ARM_MIN_OK:
            continue
        out.append({"part": i, "kp": d, "mid": [(d[5][1] + d[6][1]) / 2, (d[5][2] + d[6][2]) / 2],
                    "height": float(r["bbox_xywh"][3]), "face": r["face"]})
    return out, parts


# ── konvensi kiri / kanan ──────────────────────────
def lr_votes(cd: st.ClipData, f: int, prow: dict) -> list[dict]:
    persons, parts = persons_at(cd, f, prow)
    cm = cd.classmap(f)
    left = np.isin(cm, [cd.cls("Left_Lower_Arm"), cd.cls("Left_Hand")])
    right = np.isin(cm, [cd.cls("Right_Lower_Arm"), cd.cls("Right_Hand")])
    rad = int(round(LR_RADIUS_FRAC * cd.H))
    out = []
    for p in persons:
        for j, side in ((62, "L"), (41, "R")):
            x, y, s = p["kp"][j][1], p["kp"][j][2], p["kp"][j][3]
            if s < SCORE_MIN:
                continue
            disk = np.zeros((cd.H, cd.W), np.uint8)
            cv2.circle(disk, (int(round(x)), int(round(y))), rad, 1, -1)
            m = disk.astype(bool) & parts[p["part"]]["mask"]
            nl, nr = int((left & m).sum()), int((right & m).sum())
            vote = None if nl == nr else ("sama" if (nl > nr) == (side == "L") else "terbalik")
            out.append({"frame": f, "part": p["part"], "pose_side": side, "left_px": nl, "right_px": nr, "vote": vote})
    return out


def cmd_lr() -> None:
    cfg = load_pipeline()
    cds = {c: st.ClipData(c, cfg) for c in CLIPS}
    pose = jload(SCR / "pose.json")
    p1c = jload(st.SCRATCH / "p2c_fp32.json")
    res: dict = {"radius_frac": LR_RADIUS_FRAC, "consistent_min": LR_CONSISTENT, "sets": {}}
    sets = {f"jendela_{c}": [(c, f, pose["frames"][f"{c}:{f}"]) for f in range(window(c)[0], window(c)[1] + 1)] for c in CLIPS}
    sets["klip3_1c_20"] = [("klip3", int(f), {"parts": v["parts"]}) for f, v in p1c["frames"].items()]
    allv = []
    for name, items in sets.items():
        votes = [v for c, f, prow in items for v in lr_votes(cds[c], f, prow)]
        allv += votes
        res["sets"][name] = summarize_lr(votes)
        print(f"{name}: {res['sets'][name]}")
    res["gabungan"] = summarize_lr(allv)
    res["convention"] = res["gabungan"]["convention"]
    jsave(SCR / "lr.json", res)
    print(f"gabungan: {res['gabungan']}")


def summarize_lr(votes: list[dict]) -> dict:
    same = sum(v["vote"] == "sama" for v in votes)
    flip = sum(v["vote"] == "terbalik" for v in votes)
    dec = same + flip
    conv = ("sama" if dec and same / dec >= LR_CONSISTENT else "terbalik" if dec and flip / dec >= LR_CONSISTENT
            else "tidak konsisten")
    return {"n_wrists": len(votes), "sama": same, "terbalik": flip, "tak_tentu": len(votes) - dec,
            "frac_sama": round(same / dec, 4) if dec else None, "convention": conv,
            "median_px_side_match": float(np.median([max(v["left_px"], v["right_px"]) for v in votes if v["vote"]])) if dec else None}


# ── lacak + haluskan + kapsul + relabel ────────────
def track(seq: list[list[dict]]) -> list[dict]:
    """seq[t] = orang di frame jendela ke-t. Hungarian pada jarak titik tengah bahu (<= 0,15 x tinggi kotak) terhadap trek yang
    terakhir terlihat <= HOLD_MAX + 1 frame lalu; tak cocok -> trek baru. Return trek {t: orang}."""
    tracks: list[dict] = []
    for t, people in enumerate(seq):
        act = [k for k, tr in enumerate(tracks) if t - tr["last_t"] <= HOLD_MAX + 1]
        used = set()
        if act and people:
            cost = np.array([[float(np.hypot(p["mid"][0] - tracks[k]["mid"][0], p["mid"][1] - tracks[k]["mid"][1]))
                              for k in act] for p in people])
            ri, ci = linear_sum_assignment(cost)
            for i, j in zip(ri, ci):
                if cost[i, j] <= TRACK_MATCH_FRAC * people[i]["height"]:
                    tr = tracks[act[j]]
                    tr["at"][t], tr["mid"], tr["last_t"] = people[i], people[i]["mid"], t
                    used.add(i)
        for i, p in enumerate(people):
            if i not in used:
                tracks.append({"at": {t: p}, "mid": p["mid"], "last_t": t})
    return tracks


def smooth_points(tr: dict, rho: float, radius: int) -> dict:
    """Per titik lengan: isi (valid = skor >= 0,3; tidak valid -> titik valid terakhir bila <= HOLD_MAX frame) lalu kernel
    simetris rho^|k| (R = radius) atas frame trek yang punya nilai terisi. Return {t: {j: (x, y, held)}}."""
    ts = sorted(tr["at"])
    filled: dict = {t: {} for t in ts}
    for j in ARM_KP:
        last = None
        for t in ts:
            k = tr["at"][t]["kp"][j]
            if k[3] >= SCORE_MIN:
                last = (t, k[1], k[2])
                filled[t][j] = (k[1], k[2], False)
            elif last is not None and t - last[0] <= HOLD_MAX:
                filled[t][j] = (last[1], last[2], True)
    out: dict = {t: {} for t in ts}
    for t in ts:
        for j, (_, _, held) in filled[t].items():
            sw = sx = sy = 0.0
            for k in range(t - radius, t + radius + 1):
                v = filled.get(k, {}).get(j)
                if v is None:
                    continue
                w = rho ** abs(k - t)
                sw, sx, sy = sw + w, sx + w * v[0], sy + w * v[1]
            out[t][j] = (sx / sw, sy / sw, held)
    return out


def smooth_series(vals: dict, rho: float, radius: int) -> dict:
    return {t: sum(rho ** abs(k - t) * vals[k] for k in range(t - radius, t + radius + 1) if k in vals)
            / sum(rho ** abs(k - t) for k in range(t - radius, t + radius + 1) if k in vals) for t in vals}


def r2_measure(fg: np.ndarray, A: np.ndarray, B: np.ndarray, r1: float) -> tuple[float | None, int]:
    """Median jarak sumbu segmen ke tepi siluet di sisi yang berbatasan background (sinar tegak lurus, kedua sisi, sampai
    3 x r1; hanya sinar yang mengenai background dihitung). Tidak terukur (< R2_MIN_HITS sinar) -> None."""
    H, W = fg.shape
    v = B - A
    L = float(np.hypot(*v))
    if L < 1.0:
        return None, 0
    u = v / L
    n = np.array([-u[1], u[0]])
    dmax = int(np.ceil(R2_DMAX_FRAC * r1))
    ds = np.arange(1, dmax + 1, dtype=float)
    hits = []
    for s in np.arange(R2_T_RANGE[0] * L, R2_T_RANGE[1] * L + 1e-9, R2_STEP_PX):
        P = A + u * s
        px, py = int(np.floor(P[0])), int(np.floor(P[1]))
        if not (0 <= px < W and 0 <= py < H) or not fg[py, px]:
            continue
        for sign in (1.0, -1.0):
            xs = np.floor(P[0] + sign * n[0] * ds).astype(int)
            ys = np.floor(P[1] + sign * n[1] * ds).astype(int)
            inside = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
            stop = np.argmin(inside) if not inside.all() else len(ds)      # sinar keluar frame -> tidak dihitung sesudahnya
            bg = ~fg[ys[:stop], xs[:stop]]
            if bg.any():
                hits.append(float(ds[int(np.argmax(bg))]) - 0.5)
    return (float(np.median(hits)) if len(hits) >= R2_MIN_HITS else None), len(hits)


def capsule(A: np.ndarray, B: np.ndarray, r: float, H: int, W: int) -> tuple[np.ndarray, np.ndarray]:
    """(masker kapsul, jarak ke sumbu) seluruh frame; piksel (x, y) = indeks kolom / baris (konvensi titik P2)."""
    x0, x1 = int(max(0, np.floor(min(A[0], B[0]) - r - 1))), int(min(W, np.ceil(max(A[0], B[0]) + r + 2)))
    y0, y1 = int(max(0, np.floor(min(A[1], B[1]) - r - 1))), int(min(H, np.ceil(max(A[1], B[1]) + r + 2)))
    dist = np.full((H, W), np.inf, np.float32)
    if x0 >= x1 or y0 >= y1:
        return np.zeros((H, W), bool), dist
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    v = B - A
    vv = float(v @ v)
    t = np.clip(((xx - A[0]) * v[0] + (yy - A[1]) * v[1]) / vv, 0.0, 1.0) if vv > 0 else np.zeros_like(xx)
    d = np.hypot(xx - (A[0] + t * v[0]), yy - (A[1] + t * v[1]))
    dist[y0:y1, x0:x1] = d
    return dist <= r, dist


def prepare(var: str, clip: str) -> Path:
    """Salinan scratch: meta.json + qc_report.json + stable/ (groups disalin, depth_smooth di-hard-link). Dibuat ulang tiap kali."""
    src, d = st.CLIPS_DIR / clip, vdir(var, clip)
    if d.exists():
        shutil.rmtree(d)
    (d / "stable" / "groups").mkdir(parents=True)
    (d / "stable" / "depth_smooth").mkdir(parents=True)
    for f in ("meta.json", "qc_report.json"):
        shutil.copyfile(src / f, d / f)
    for f in ("manifest.json", "frames.jsonl"):
        if (src / "stable" / f).is_file():
            shutil.copyfile(src / "stable" / f, d / "stable" / f)
    for p in sorted((src / "stable" / "groups").glob("*.png")):
        shutil.copyfile(p, d / "stable" / "groups" / p.name)
    for p in sorted((src / "stable" / "depth_smooth").glob("*.npy")):
        try:
            os.link(p, d / "stable" / "depth_smooth" / p.name)
        except OSError:
            shutil.copyfile(p, d / "stable" / "depth_smooth" / p.name)
    return d


def write_mask(path: Path, m: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", m.astype(np.uint8))
    buf.tofile(path)


def read_mask(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_UNCHANGED).astype(bool)


def cmd_arm() -> None:
    cfg = load_pipeline()
    cds = {c: st.ClipData(c, cfg) for c in CLIPS}
    pose = jload(SCR / "pose.json")
    lr = jload(SCR / "lr.json")
    if lr["convention"] == "tidak konsisten":
        raise SystemExit(f"konvensi kiri / kanan tidak konsisten ({lr['gabungan']}) — berhenti, laporkan ke Rio")
    flip = lr["convention"] == "terbalik"
    gid = group_ids()
    side_group = {"L": gid["right_arm" if flip else "left_arm"], "R": gid["left_arm" if flip else "right_arm"]}
    rho, radius = stb.kernel_rho(cfg.stabilize.temporal.mask_ema_alpha), stb.kernel_radius(cfg.stabilize.temporal.mask_ema_alpha)
    res: dict = {"convention": lr["convention"], "side_group": side_group, "rho": rho, "radius": radius, "clips": {}}
    for c in CLIPS:
        cd = cds[c]
        a, b = window(c)
        frames = list(range(a, b + 1))
        for var in VARIANTS:
            prepare(var, c)
        t0 = time.perf_counter()
        seq, parts_of = [], {}
        for f in frames:
            persons, parts = persons_at(cd, f, pose["frames"][f"{c}:{f}"])
            seq.append(persons)
            parts_of[f] = parts
        t_io = time.perf_counter() - t0
        t1 = time.perf_counter()
        tracks = track(seq)
        sm = [smooth_points(tr, rho, radius) for tr in tracks]
        # jarak bahu per trek (R1), segmen yang ada
        geo: list[dict] = []
        for k, (tr, pts) in enumerate(zip(tracks, sm)):
            sd = {t: float(np.hypot(p[5][0] - p[6][0], p[5][1] - p[6][1])) for t, p in pts.items() if 5 in p and 6 in p}
            sd_med = float(np.median(list(sd.values()))) if sd else None
            g = {"track": k, "frames": sorted(tr["at"]), "sd": sd, "sd_med": sd_med, "seg": {}}
            for name, ja, jb in SEGMENTS:
                g["seg"][name] = {t: (np.array(pts[t][ja][:2]), np.array(pts[t][jb][:2]),
                                      R1_FRAC * (sd.get(t, sd_med) or 0.0), pts[t][ja][2] or pts[t][jb][2])
                                  for t in pts if ja in pts[t] and jb in pts[t] and (sd.get(t) or sd_med)}
            geo.append(g)
        t_track = time.perf_counter() - t1
        # R2: ukur per segmen-frame di peta grup base, R1 bila tidak terukur, lalu haluskan per trek-segmen
        base_g = {f: stb.read_groups(st.CLIPS_DIR / c / "stable" / "groups" / f"frame_{f:05d}.png") for f in frames}
        t_r2 = {f: 0.0 for f in frames}
        r2_info = []
        for g in geo:
            for name, segs in g["seg"].items():
                raw = {}
                for t, (A, B, r1, held) in segs.items():
                    t2 = time.perf_counter()
                    r2, nh = r2_measure(base_g[frames[t]] != stb.BACKGROUND_ID, A, B, r1)
                    t_r2[frames[t]] += time.perf_counter() - t2
                    raw[t] = r2 if r2 is not None else r1
                    r2_info.append({"track": g["track"], "seg": name, "frame": frames[t], "r1": round(r1, 2),
                                    "r2_raw": None if r2 is None else round(r2, 2), "hits": nh, "fallback_r1": r2 is None})
                g.setdefault("r2", {})[name] = smooth_series(raw, rho, radius) if raw else {}
        per_frame = {var: {} for var in ("R1", "R2")}
        timing = {f: {} for f in frames}
        for t, f in enumerate(frames):
            gm = base_g[f]
            cm = cd.classmap(f)
            t3 = time.perf_counter()
            cand_base = (cm == cd.cls("Upper_Clothing")) & (gm == gid["torso"])
            for var in ("R1", "R2"):
                best_d = np.full(gm.shape, np.inf, np.float32)
                best_g = np.zeros(gm.shape, np.uint8)
                own = np.full(gm.shape, -1, np.int16)
                caps = []
                for g, tr in zip(geo, tracks):
                    if t not in tr["at"]:
                        continue
                    pmask = parts_of[f][tr["at"][t]["part"]]["mask"]
                    for name, segs in g["seg"].items():
                        if t not in segs:
                            continue
                        A, B, r1, _ = segs[t]
                        r = r1 if var == "R1" else g["r2"][name][t]
                        cap, dist = capsule(A, B, r, cd.H, cd.W)
                        cand = cap & pmask & cand_base
                        upd = cand & (dist < best_d)
                        best_d[upd] = dist[upd]
                        best_g[upd] = side_group[name[0]]
                        own[upd] = g["track"]
                        if name.endswith("atas"):
                            caps.append((g["track"], A, r))
                rel = best_g > 0
                new = gm.copy()
                new[rel] = best_g[rel]
                sup = np.zeros(gm.shape, bool)
                for k, S, r in caps:
                    disk = np.zeros(gm.shape, np.uint8)
                    cv2.circle(disk, (int(round(S[0])), int(round(S[1]))), int(np.ceil(r + CAP_MARGIN_PX)), 1, -1)
                    sup |= disk.astype(bool) & cv2.dilate((own == k).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
                timing[f][var] = time.perf_counter() - t3
                d = vdir(var, c)
                stb.write_groups(d / "stable" / "groups" / f"frame_{f:05d}.png", new)
                write_mask(d / "suppress" / f"frame_{f:05d}.png", sup)
                write_mask(d / "relabel" / f"frame_{f:05d}.png", rel)
                per_frame[var][f] = {"relabel_px": int(rel.sum()), "suppress_px": int(sup.sum()),
                                     "relabel_left_px": int((best_g == gid["left_arm"]).sum()),
                                     "relabel_right_px": int((best_g == gid["right_arm"]).sum())}
                t3 = time.perf_counter()
        n = len(frames)
        present = []
        for g, tr in zip(geo, tracks):
            mx = float(np.mean([tr["at"][t]["mid"][0] for t in tr["at"]]))
            present.append({"track": g["track"], "mid_x_mean": round(mx, 1), "frames_present": len(tr["at"]),
                            "pct_frames_present": round(100 * len(tr["at"]) / n, 1),
                            "seg_pct": {name: round(100 * len(segs) / n, 1) for name, segs in g["seg"].items()},
                            "held_points": sum(v[2] for p in sm[g["track"]].values() for v in p.values()),
                            "sd_med": g["sd_med"]})
        fb = [x for x in r2_info]
        res["clips"][c] = {
            "window": [a, b], "n_tracks": len(tracks), "persons_per_frame": [len(p) for p in seq], "tracks": present,
            "r2_fallback": {"segment_frames": len(fb), "fallback_r1": sum(x["fallback_r1"] for x in fb),
                            "pct": round(100 * sum(x["fallback_r1"] for x in fb) / max(len(fb), 1), 1),
                            "per_seg": {name: {"n": sum(x["seg"] == name for x in fb),
                                               "fallback": sum(x["seg"] == name and x["fallback_r1"] for x in fb)} for name, _, _ in SEGMENTS}},
            "r_values": {"r1_median": float(np.median([x["r1"] for x in fb])) if fb else None,
                         "r2_raw_median": float(np.median([x["r2_raw"] for x in fb if x["r2_raw"] is not None]))
                         if any(x["r2_raw"] is not None for x in fb) else None},
            "r2_detail": r2_info, "per_frame": per_frame,
            "timing": {"read_inputs_s": t_io, "track_smooth_s": t_track, "track_smooth_per_frame_s": t_track / n,
                       "r2_measure_per_frame_s": {f: t_r2[f] for f in frames},
                       "relabel_per_frame_s": {var: {f: timing[f][var] for f in frames} for var in ("R1", "R2")}}}
        r = res["clips"][c]
        print(f"{c} {a}-{b}: trek {len(tracks)}, orang / frame {r['persons_per_frame']}")
        for p in present:
            print(f"   trek {p}")
        print(f"   R2 jatuh ke R1: {r['r2_fallback']}; r1 median {r['r_values']['r1_median']}, r2 mentah median {r['r_values']['r2_raw_median']}")
        for var in ("R1", "R2"):
            px = [v["relabel_px"] for v in per_frame[var].values()]
            print(f"   {var}: relabel px / frame mean {np.mean(px):.0f} min {min(px)} maks {max(px)}")
    jsave(SCR / "arm.json", res)


# ── render [4]–[6] ─────────────────────────────────
def cfg_path(var: str) -> Path:
    p = SCR / var / "cfg.yaml"
    text = (ROOT / "configs" / "default.yaml").read_text(encoding="utf-8")
    line = '  out_dir: "out"'
    if text.count(line) != 1:
        raise SystemExit("configs/default.yaml: baris out_dir tidak ditemukan tepat sekali")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text.replace(line, f'  out_dir: "{(SCR / var / "out").as_posix()}"'), encoding="utf-8")
    return p


def cmd_vectorize(var: str, clip: str) -> int:
    from rotoscope import vectorize as vz

    d = vdir(var, clip)
    a, b = window(clip)
    if var != "base":
        gid = group_ids()
        arms = {gid["left_arm"], gid["right_arm"]}
        cur: dict = {"name": None}
        orig_load = vz.load_frame_inputs

        def load_frame_inputs(clip_obj, name, n_groups):
            cur["name"] = name
            return orig_load(clip_obj, name, n_groups)

        def pair_skeletons(gmap, line_min_px):
            """= vz.pair_skeletons, kecuali band pasangan torso | lengan dikurangi masker tutup bahu frame ini."""
            sup = read_mask(d / "suppress" / (Path(cur["name"]).stem + ".png")) if cur["name"] else None
            ids = [int(g) for g in np.unique(gmap) if g != stb.BACKGROUND_ID]
            dil = {g: cv2.dilate((gmap == g).astype(np.uint8), vz.KERNEL_3X3).astype(bool) for g in ids}
            out = []
            for ai, ga in enumerate(ids):
                ma = gmap == ga
                for gb in ids[ai + 1:]:
                    band = (ma & dil[gb]) | ((gmap == gb) & dil[ga])
                    if sup is not None and gid["torso"] in (ga, gb) and ({ga, gb} & arms):
                        band &= ~sup
                    thinned = vz.thin_band(band, line_min_px)
                    if thinned is not None:
                        out.append((ga, gb, *thinned))
            return out

        vz.load_frame_inputs = load_frame_inputs
        vz.pair_skeletons = pair_skeletons
    return vz.main(["--config", str(cfg_path(var)), "--work-dir", str(d), "--limit", str(b + 1)])


def run_logged(cmd: list[str], log: Path) -> tuple[int, float]:
    t0 = time.perf_counter()
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(r.stdout + "\n--- stderr ---\n" + r.stderr, encoding="utf-8")
    return r.returncode, time.perf_counter() - t0


def cmd_render() -> None:
    for c in CLIPS:
        a, b = window(c)
        for var in VARIANTS:
            d, cfg = vdir(var, c), cfg_path(var)
            steps = [("vectorize", [sys.executable, str(Path(__file__)), "vectorize", var, c]),
                     ("stylize", [sys.executable, "-m", "rotoscope.stylize", "--config", str(cfg), "--work-dir", str(d),
                                  "--from", str(a), "--limit", str(b - a + 1)]),
                     ("export", [sys.executable, "-m", "rotoscope.export", "--config", str(cfg), "--work-dir", str(d),
                                 "--from", str(a), "--limit", str(b - a + 1)])]
            for name, cmd in steps:
                rc, dt = run_logged(cmd, SCR / "logs" / f"{name}_{var}_{c}.log")
                print(f"  {c} {var} {name}: rc {rc}, {dt:.1f} s", flush=True)
                if rc:
                    raise SystemExit(f"{name} gagal — lihat work/t701_scratch/proto/logs/{name}_{var}_{c}.log")


# ── metrik ─────────────────────────────────────────
def contour_docs(var: str, clip: str, frames: list[int]) -> list[dict]:
    return [jload(vdir(var, clip) / "contours" / f"frame_{f:05d}.json") for f in frames]


def new_keys(doc: dict, rel: np.ndarray | None, gid: dict) -> set:
    if rel is None or not rel.any():
        return set()
    dist = cv2.distanceTransform((~rel).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    H, W = rel.shape
    out = set()
    for s in doc["strokes"]:
        if s["type"] != "group_boundary" or "torso" not in s["groups"] or not ({"left_arm", "right_arm"} & set(s["groups"])):
            continue
        p = np.asarray(s["points"], float)
        xs = np.clip(np.floor(p[:, 0]).astype(int), 0, W - 1)
        ys = np.clip(np.floor(p[:, 1]).astype(int), 0, H - 1)
        if np.mean(dist[ys, xs] <= NEW_STROKE_DIST_PX) >= NEW_STROKE_FRAC:
            out.add((s["type"], s["track_id"]))
    return out


def cmd_metrics() -> None:
    gid = group_ids()
    arm = jload(SCR / "arm.json")
    pose = jload(SCR / "pose.json")
    res: dict = {"lr": jload(SCR / "lr.json")["gabungan"], "pose_precision": pose["precision"], "clips": {}}
    for c in CLIPS:
        a, b = window(c)
        frames = list(range(a, b + 1))
        rc: dict = {"window": [a, b]}
        # base scratch = produksi (sanity jalur scratch)
        same_json = sum((vdir("base", c) / "contours" / f"frame_{f:05d}.json").read_bytes()
                        == (st.CLIPS_DIR / c / "contours" / f"frame_{f:05d}.json").read_bytes() for f in frames)
        same_png = sum(st.file_sha(vdir("base", c) / "strokes" / f"frame_{f:05d}.png")
                       == st.file_sha(st.CLIPS_DIR / c / "strokes" / f"frame_{f:05d}.png") for f in frames)
        rc["base_vs_produksi"] = {"contours_json_identik": same_json, "strokes_png_identik": same_png, "dari": len(frames)}
        for var in VARIANTS:
            docs = contour_docs(var, c, frames)
            rels = [read_mask(vdir(var, c) / "relabel" / f"frame_{f:05d}.png") if var != "base" else None for f in frames]
            nk = [new_keys(dd, r, gid) for dd, r in zip(docs, rels)]
            lens = [{(s["type"], s["track_id"]): tm.stroke_length(s["points"], s["closed"]) for s in dd["strokes"]} for dd in docs]
            pop = tm.pop_energy_by_type(lens)
            births = [len(set(lens[t]) - set(lens[t - 1])) for t in range(1, len(lens))]
            births_new = [len(nk[t] - set(lens[t - 1])) for t in range(1, len(lens))]
            pop_new = []
            for t in range(1, len(lens)):
                tot = sum(lens[t].values()) or 1.0
                born = sum(v for k, v in lens[t].items() if k in nk[t] and k not in lens[t - 1])
                died = sum(v for k, v in lens[t - 1].items() if k in nk[t - 1] and k not in lens[t])
                pop_new.append((born + died) / tot)
            gb = [sum(s["type"] == "group_boundary" for s in dd["strokes"]) for dd in docs]
            occ = [sum(s["type"] == "occlusion" for s in dd["strokes"]) for dd in docs]
            new_len_share = [sum(v for k, v in lens[t].items() if k in nk[t]) / (sum(lens[t].values()) or 1.0) for t in range(len(lens))]
            new_lens = [lens[t][k] for t in range(len(lens)) for k in nk[t]]
            v = {"group_boundary_per_frame": float(np.mean(gb)), "occlusion_per_frame": float(np.mean(occ)),
                 "new_gb_per_frame": float(np.mean([len(x) for x in nk])), "new_gb_min_max": [min(len(x) for x in nk), max(len(x) for x in nk)],
                 "new_gb_len_px": ({"p50": float(np.median(new_lens)), "p10": float(np.percentile(new_lens, 10)),
                                    "p90": float(np.percentile(new_lens, 90)), "pct_lt_30px": float(100 * np.mean(np.array(new_lens) < 30))}
                                   if new_lens else None),
                 "new_len_share_mean": float(np.mean(new_len_share)),
                 "births_per_frame": float(np.mean(births)), "births_new_per_frame": float(np.mean(births_new)),
                 "pop_total_mean": pop["total_pop_energy_mean"], "pop_new_mean": float(np.mean(pop_new)),
                 "pop_by_type": {ty: round(p["pop_energy_mean"], 5) for ty, p in pop["types"].items()},
                 "age_median_gb": pop["types"]["group_boundary"]["age_median"]}
            if var != "base":
                pf = arm["clips"][c]["per_frame"][var]
                px = [pf[str(f)]["relabel_px"] if str(f) in pf else pf[f]["relabel_px"] for f in frames]
                v["relabel_px_per_frame"] = {"mean": float(np.mean(px)), "min": int(min(px)), "max": int(max(px))}
                tt = arm["clips"][c]["timing"]
                per = [tt["relabel_per_frame_s"][var][str(f)] if str(f) in tt["relabel_per_frame_s"][var] else tt["relabel_per_frame_s"][var][f]
                       for f in frames]
                r2t = [tt["r2_measure_per_frame_s"][str(f)] if str(f) in tt["r2_measure_per_frame_s"] else tt["r2_measure_per_frame_s"][f]
                       for f in frames] if var == "R2" else [0.0] * len(frames)
                v["arm_hint_s_per_frame"] = {"mean": float(np.mean(np.array(per) + np.array(r2t)) + tt["track_smooth_per_frame_s"]),
                                             "p95": float(np.percentile(np.array(per) + np.array(r2t), 95) + tt["track_smooth_per_frame_s"])}
            rc[var] = v
        pt = [pose["frames"][f"{c}:{f}"]["time_s"] for f in frames]
        rc["p2_s_per_frame"] = {"mean": float(np.mean(pt)), "p50": float(np.median(pt)), "p95": float(np.percentile(pt, 95)),
                                "per_crop_p50": float(np.median([p["time_s"] for f in frames for p in pose["frames"][f"{c}:{f}"]["parts"]
                                                                 if "time_s" in p]))}
        rc["peak_reserved_mib"] = max(pose["frames"][f"{c}:{f}"]["peak_reserved_mib"] for f in frames)
        rc["tracks"] = arm["clips"][c]["tracks"]
        rc["r2_fallback"] = arm["clips"][c]["r2_fallback"]
        rc["r_values"] = arm["clips"][c]["r_values"]
        res["clips"][c] = rc
    jsave(SCR / "metrics.json", res)
    print(json.dumps(res, indent=1, ensure_ascii=False))


# ── bahan penilaian ────────────────────────────────
def preview_mp4(var: str, clip: str) -> Path:
    a, b = window(clip)
    got = sorted((SCR / var / "out").glob(f"*.preview_{a}-{b}.mp4"))
    if len(got) != 1:
        raise SystemExit(f"preview {var} {clip}: {len(got)} berkas")
    return got[0]


def read_video(p: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(p))
    out = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        out.append(fr)
    cap.release()
    return out


def header(img_w: int, labels: list[str], num: str | None = None) -> np.ndarray:
    h = np.full((HEAD, img_w, 3), 255, np.uint8)
    for k, lbl in enumerate(labels):
        cv2.putText(h, lbl, (GAP + k * (PANEL_W + GAP) + 10, HEAD - 22), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 5, cv2.LINE_AA)
    if num:
        cv2.putText(h, num, (img_w - 150, HEAD - 18), cv2.FONT_HERSHEY_SIMPLEX, 2.4, (0, 0, 200), 6, cv2.LINE_AA)
    return h


def row(panels: list[np.ndarray], labels: list[str], num: str | None = None) -> np.ndarray:
    W = len(panels) * PANEL_W + (len(panels) + 1) * GAP
    body = np.full((PANEL_H + GAP, W, 3), 255, np.uint8)
    for k, p in enumerate(panels):
        x = GAP + k * (PANEL_W + GAP)
        body[:PANEL_H, x:x + PANEL_W] = cv2.resize(p, (PANEL_W, PANEL_H), interpolation=cv2.INTER_AREA)
    return np.vstack([header(W, labels, num), body])


def encode(frames: list[np.ndarray], path: Path, fps: int = 24) -> None:
    h, w = frames[0].shape[:2]
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
           "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fr in frames:
        p.stdin.write(np.ascontiguousarray(fr).tobytes())
    p.stdin.close()
    if p.wait():
        raise SystemExit(f"ffmpeg gagal untuk {path}")


def cmd_media() -> None:
    cfg = load_pipeline()
    OUTP.mkdir(parents=True, exist_ok=True)
    info: dict = {"videos": {}, "board": {}}
    for c in CLIPS:
        vids = {var: read_video(preview_mp4(var, c)) for var in VARIANTS}
        n = {var: len(v) for var, v in vids.items()}
        if len(set(n.values())) != 1:
            raise SystemExit(f"{c}: jumlah frame preview beda {n}")
        side = [row([vids[var][t] for var in VARIANTS], list(VARIANTS)) for t in range(n["base"])]
        encode(side * 3, OUTP / f"{c}_normal_3x.mp4")
        encode([fr for fr in side for _ in range(2)], OUTP / f"{c}_lambat_0.5x.mp4")
        info["videos"][c] = {"frames": n["base"], "size": list(side[0].shape[1::-1]), "normal_3x_frames": 3 * n["base"],
                             "slow_frames": 2 * n["base"]}
        print(f"  {c}: {n['base']} frame -> {c}_normal_3x.mp4, {c}_lambat_0.5x.mp4 ({side[0].shape[1]}x{side[0].shape[0]})")
    picks = []
    for c in CLIPS:
        a, b = window(c)
        picks += [(c, a + int(i)) for i in np.round(np.linspace(0, b - a, BOARD_PER_WINDOW)).astype(int)]
    order = np.random.default_rng(SEED_BOARD_PROTO).permutation(len(picks))
    pdir = OUTP / "papan"
    pdir.mkdir(parents=True, exist_ok=True)
    cds = {c: st.ClipData(c, cfg) for c in CLIPS}
    mapping = {}
    for k, idx in enumerate(order):
        num = f"{k + 1:02d}"
        c, f = picks[int(idx)]
        mapping[num] = {"clip": c, "frame": f}
        foto = cv2.cvtColor(st.read_rgb(cds[c], f), cv2.COLOR_RGB2BGR)
        pans = [foto] + [cv2.imdecode(np.fromfile(vdir(var, c) / "strokes" / f"frame_{f:05d}.png", np.uint8), cv2.IMREAD_COLOR)
                         for var in VARIANTS]
        img = row(pans, ["foto", *VARIANTS], num)
        cv2.imencode(".png", img)[1].tofile(pdir / f"{num}.png")
        diff = [int((cv2.absdiff(pans[1], p) > 40).any(axis=2).sum()) for p in pans[2:]]
        print(f"  {num}.png {img.shape[1]}x{img.shape[0]} piksel beda vs base (R1, R2): {diff}")
    info["board"] = {"seed": SEED_BOARD_PROTO, "map": mapping}
    jsave(SCR / "board_map_proto.json", info)


def main(argv: list[str]) -> None:
    cmds = {"windows": cmd_windows, "lr": cmd_lr, "arm": cmd_arm, "render": cmd_render, "metrics": cmd_metrics, "media": cmd_media}
    if argv[:1] == ["pose"]:
        cmd_pose(argv[1] if len(argv) > 1 and argv[1] in ("fp16", "fp32") else None)
    elif argv[:1] == ["vectorize"] and len(argv) == 3:
        sys.exit(cmd_vectorize(argv[1], argv[2]))
    elif argv[:1] and argv[0] in cmds:
        cmds[argv[0]]()
    else:
        print(__doc__)
        sys.exit(2)


if __name__ == "__main__":
    main(sys.argv[1:])
