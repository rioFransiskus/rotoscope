"""T-305c Tahap 1b: label Rio -> AUC / retensi (bootstrap per TRACK) dan simulasi [4] di salinan scratch. Dipanggil dari
`scripts/t305c_study.py` (subperintah di bawah); baca-saja terhadap klip, keluaran hanya ke work/t305c_scratch/.

    python scripts/t305c_study.py labels                  # labels.json dari daftar label Rio (papan v2) + hitungan
    python scripts/t305c_study.py eval                    # AUC + CI, retensi, thresholds.json -> results/eval.json
    python scripts/t305c_study.py sim VARIAN KLIP [KLIP]  # satu varian x klip -> sim/VARIAN__KLIP.json
    python scripts/t305c_study.py simall [VARIAN ...]     # semua varian (paralel, subproses)
    python scripts/t305c_study.py simtable                # tabel markdown dari sim/*.json
    python scripts/t305c_study.py timing                  # biaya [4] per frame (berurutan, tanpa paralel)

Seed bootstrap tetap (SEED_BOOT), 2000 ulangan, resampling TRACK di dalam sel sampling (desain papan dipertahankan).
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
import t305c_study as st  # noqa: E402
from rotoscope import stabilize as stb  # noqa: E402
from rotoscope import track as trk  # noqa: E402
from rotoscope import vectorize as vec  # noqa: E402

ROOT, SCRATCH = st.ROOT, st.SCRATCH
RESULTS = SCRATCH / "results"
SIM_DIR = SCRATCH / "sim"
SEED_BOOT = 305900
N_BOOT = 2000
SEP_LOWER = 0.6                          # "tidak memisahkan" bila batas bawah CI AUC <= 0,6
RETENTIONS = (0.90, 0.95)
MATCH_RADIUS = 1.5                       # px: titik strok berlabel dianggap ada di varian bila <= ini dari titik strok oklusi varian
MATCH_FRACTION = 0.5

LABELS_RIO = {
    "O": [1, 4, 7, 8, 13, 14, 16, 18, 19, 22, 23, 24, 25, 26, 27, 28, 29, 31, 33, 35, 39, 40, 41, 42, 43, 45, 46, 47, 48, 49, 52,
          53, 56, 57, 59, 60, 61, 62, 63, 64, 65],
    "L": [3, 6, 10, 12, 15, 17, 20, 21, 30, 32, 34, 36, 37, 38, 44, 50, 51, 54, 55, 58],
    "?": [2, 5, 9, 11],
}

# fitur: (arah yang dideklarasikan SEBELUM label dilihat: +1 = nilai besar -> O, -1 = nilai kecil -> O, kelompok kandidat)
FEATS = {
    "strength_abs": (+1, "A"),
    "J_2": (+1, "B"), "J_4": (+1, "B"), "J_6": (+1, "B"),
    "R_2": (+1, "B"), "R_4": (+1, "B"), "R_6": (+1, "B"),
    "S": (-1, "B"),
    "G_2": (-1, "B"), "G_4": (-1, "B"), "G_6": (-1, "B"),
    "absC_2": (-1, "B*"), "absC_4": (-1, "B*"), "absC_6": (-1, "B*"),     # B*: arah tidak didaftarkan -> eksploratif
    "path_px": (+1, "C"),
}
SCOPES = ("test", "klip3", "gabungan", "Klip2")


# ── label ──────────────────────────────────────────
def cmd_labels() -> None:
    allnum = [n for v in LABELS_RIO.values() for n in v]
    assert len(allnum) == len(set(allnum)) == 65 and sorted(allnum) == list(range(1, 66)), "label: harus 65 nomor unik 1..65"
    bm = st.jload(SCRATCH / "board_map.json")
    labels = {}
    for lab, nums in LABELS_RIO.items():
        for n in nums:
            m = bm["mapping"][f"{n:02d}"]
            labels[f"{n:02d}"] = {"label": lab, "clip": m["clip"], "track_id": m["track_id"], "run_start": m["run_start"],
                                  "cell": m["cell"], "donewhen": bool(m.get("donewhen_frames"))}
    counts: dict = {}
    for v in labels.values():
        counts.setdefault(v["cell"], {"O": 0, "L": 0, "?": 0})[v["label"]] += 1
    per_clip: dict = {}
    for v in labels.values():
        per_clip.setdefault(v["clip"], {"O": 0, "L": 0, "?": 0})[v["label"]] += 1
    dw = {"O": 0, "L": 0, "?": 0}
    for v in labels.values():
        if v["donewhen"]:
            dw[v["label"]] += 1
    st.jsave(SCRATCH / "labels.json", {"source": "Rio 2026-10-12, papan v2 seed 311001", "labels": labels, "counts_by_cell": counts,
                                       "counts_by_clip": per_clip, "donewhen_labels": dw})
    print(json.dumps({"by_clip": per_clip, "by_cell": counts, "donewhen": dw}, ensure_ascii=False, indent=1))


# ── data analisis ──────────────────────────────────
def load_units() -> tuple[list[dict], list[dict]]:
    """(units, strokes). unit = track berlabel (O / L; '?' dikeluarkan) dengan bobot invers peluang sampling per sel."""
    feats = st.jload(SCRATCH / "features.json")
    bm = st.jload(SCRATCH / "board_map.json")
    lab = st.jload(SCRATCH / "labels.json")["labels"]
    cells = bm["sampling"]["klip3"]["cells"]
    t_high = {c: i["t_high"] for c, i in feats["clips"].items()}
    units: dict[tuple, dict] = {}
    for num, m in lab.items():
        key = (m["clip"], m["track_id"], m["run_start"])
        if m["cell"].startswith("klip3:"):
            c = cells[m["cell"].split(":")[1]]
            w = c["jauh_dari_cut"] / c["dipilih"]
        else:
            w = 1.0
        units[key] = {"key": key, "number": num, "label": m["label"], "cell": m["cell"], "clip": m["clip"], "w": w,
                      "donewhen": m["donewhen"], "rows": []}
    for r in feats["rows"]:
        k = (r["clip"], r["track_id"], r["run_start"])
        if k in units:
            r = dict(r)
            r["strength_abs"] = r["strength"] * t_high[r["clip"]]
            units[k]["rows"].append(r)
    ulist = [u for u in units.values()]
    for i, u in enumerate(ulist):
        u["idx"] = i
    return ulist, feats["rows"]


def feat_value(r: dict, f: str):
    return r.get(f)


def unit_scores(units: list[dict], f: str, direction: int) -> np.ndarray:
    """Skor tingkat track = median nilai fitur atas strok track (dikali arah); NaN bila tak ada."""
    out = np.full(len(units), np.nan)
    for u in units:
        v = [r[f] for r in u["rows"] if r.get(f) is not None]
        if v:
            out[u["idx"]] = direction * float(np.median(v))
    return out


def stroke_table(units: list[dict], f: str, direction: int):
    sc, uid = [], []
    for u in units:
        for r in u["rows"]:
            if r.get(f) is not None:
                sc.append(direction * r[f])
                uid.append(u["idx"])
    return np.array(sc), np.array(uid, int)


def wauc(s: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """AUC berbobot: P(skor O > skor L) + 0,5 P(sama); y = 1 untuk O. NaN bila salah satu kelas berbobot 0."""
    pos, neg = (y == 1) & (w > 0), (y == 0) & (w > 0)
    if not pos.any() or not neg.any():
        return float("nan")
    sn, wn = s[neg], w[neg]
    order = np.argsort(sn)
    sn, cw = sn[order], np.concatenate([[0.0], np.cumsum(wn[order])])
    lo, hi = np.searchsorted(sn, s[pos], "left"), np.searchsorted(sn, s[pos], "right")
    num = (w[pos] * (cw[lo] + 0.5 * (cw[hi] - cw[lo]))).sum()
    return float(num / (w[pos].sum() * wn.sum()))


def boot_mult(units: list[dict]) -> np.ndarray:
    """Matriks multiplisitas (N_BOOT x n_unit): resampling track dengan pengembalian DI DALAM tiap sel sampling."""
    rng = np.random.default_rng(SEED_BOOT)
    M = np.zeros((N_BOOT, len(units)), np.float64)
    cells: dict[str, list[int]] = {}
    for u in units:
        cells.setdefault(u["cell"], []).append(u["idx"])
    for c, idx in sorted(cells.items()):
        idx = np.array(idx)
        d = rng.integers(0, len(idx), (N_BOOT, len(idx)))
        for r in range(N_BOOT):
            M[r, :] += np.bincount(idx[d[r]], minlength=len(units))
    return M


def scope_mask(units: list[dict], scope: str) -> np.ndarray:
    clips = {"test": ("test",), "klip3": ("klip3",), "gabungan": ("test", "klip3"), "Klip2": ("Klip2",)}[scope]
    return np.array([u["clip"] in clips for u in units])


def ci(a: np.ndarray) -> list[float | None]:
    a = a[~np.isnan(a)]
    return [None, None] if len(a) < 100 else [round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4)]


def auc_block(units, M, f, direction) -> dict:
    y_u = np.array([1 if u["label"] == "O" else 0 for u in units])
    w_u = np.array([u["w"] for u in units])
    su = unit_scores(units, f, direction)
    ss, uid = stroke_table(units, f, direction)
    res = {}
    for scope in SCOPES:
        sm = scope_mask(units, scope)
        out = {}
        for level in ("track", "stroke"):
            for wt in ("unweighted", "weighted"):
                if level == "track":
                    s, y, base_w, mult_idx = su, y_u, (w_u if wt == "weighted" else np.ones(len(units))), np.arange(len(units))
                    ok = sm & ~np.isnan(su)
                    s_, y_, bw_, mi_ = s[ok], y[ok], base_w[ok], mult_idx[ok]
                else:
                    ok = sm[uid]
                    s_, y_, mi_ = ss[ok], y_u[uid][ok], uid[ok]
                    bw_ = (w_u if wt == "weighted" else np.ones(len(units)))[mi_]
                point = wauc(s_, y_, bw_)
                if scope == "Klip2":
                    out[f"{level}_{wt}"] = {"auc": None if np.isnan(point) else round(point, 4), "ci": None}
                    continue
                reps = np.array([wauc(s_, y_, bw_ * M[r, mi_]) for r in range(N_BOOT)])
                out[f"{level}_{wt}"] = {"auc": round(point, 4), "ci": ci(reps)}
        res[scope] = out
    return res


# ── retensi ────────────────────────────────────────
def tau_for(s: np.ndarray, y: np.ndarray, w: np.ndarray, ret: float) -> float:
    """Ambang terbesar sehingga fraksi berbobot strok O dengan skor >= tau adalah >= ret."""
    so, wo = s[y == 1], w[y == 1]
    order = np.argsort(-so)
    cum = np.cumsum(wo[order]) / wo.sum()
    return float(so[order][int(np.searchsorted(cum, ret - 1e-12))])


def l_removed(s, y, w, tau) -> float:
    m = (y == 0) & (w > 0)
    return float(w[m & (s < tau)].sum() / w[m].sum())


def o_kept(s, y, w, tau) -> float:
    m = (y == 1) & (w > 0)
    return float(w[m & (s >= tau)].sum() / w[m].sum())


def retention_block(units, M, f, direction, scope="gabungan") -> dict:
    y_u = np.array([1 if u["label"] == "O" else 0 for u in units])
    w_u = np.array([u["w"] for u in units])
    sm = scope_mask(units, scope)
    ss, uid = stroke_table(units, f, direction)
    ok = sm[uid]
    s, y, mi = ss[ok], y_u[uid][ok], uid[ok]
    out = {}
    for ret in RETENTIONS:
        w = w_u[mi]
        tau = tau_for(s, y, w, ret)
        res = {"tau": tau, "scope": scope, "weighted": {"O_kept": round(o_kept(s, y, w, tau), 4), "L_removed": round(l_removed(s, y, w, tau), 4)},
               "unweighted": {"O_kept": round(o_kept(s, y, np.ones(len(s)), tau), 4), "L_removed": round(l_removed(s, y, np.ones(len(s)), tau), 4)}}
        lr, ok_ = [], []
        for r in range(N_BOOT):
            wb = w * M[r, mi]
            if not ((y == 1) & (wb > 0)).any() or not ((y == 0) & (wb > 0)).any():
                continue
            tb = tau_for(s, y, wb, ret)
            lr.append(l_removed(s, y, wb, tb))
            ok_.append(o_kept(s, y, wb, tau))
        res["ci_L_removed_tau_rederived"] = ci(np.array(lr))
        res["ci_O_kept_at_fixed_tau"] = ci(np.array(ok_))
        per = {}
        for c in ("test", "klip3", "Klip2"):
            cm = np.array([units[i]["clip"] == c for i in mi])
            if cm.any() and (y[cm] == 1).any() and (y[cm] == 0).any():
                per[c] = {"O_kept": round(o_kept(s[cm], y[cm], w[cm], tau), 4), "L_removed": round(l_removed(s[cm], y[cm], w[cm], tau), 4)}
        res["per_clip_weighted"] = per
        out[f"r{int(ret * 100)}"] = res
    return out


def bc_grid(units, f, direction, ret) -> dict:
    """B+C eksploratif: aturan DAN (skor B >= tau_B dan panjang jalur >= tau_C); grid ambang dari nilai strok O; pilih L terbuang
    terbesar dengan retensi O berbobot >= ret (dipilih pada data yang sama -> optimistis, tanpa CI)."""
    y_u = np.array([1 if u["label"] == "O" else 0 for u in units])
    w_u = np.array([u["w"] for u in units])
    sm = scope_mask(units, "gabungan")
    rows = [(direction * r[f], r["path_px"], y_u[u["idx"]], w_u[u["idx"]]) for u in units if sm[u["idx"]] for r in u["rows"] if r.get(f) is not None]
    sb, sc, y, w = (np.array(c, float) for c in zip(*rows))
    best = None
    for tb in np.unique(np.quantile(sb[y == 1], np.linspace(0, 0.3, 31))):
        for tc in np.unique(np.concatenate([[0.0], np.quantile(sc[y == 1], np.linspace(0, 0.3, 31))])):
            keep = (sb >= tb) & (sc >= tc)
            ko = w[(y == 1) & keep].sum() / w[y == 1].sum()
            if ko + 1e-12 < ret:
                continue
            lr = w[(y == 0) & ~keep].sum() / w[y == 0].sum()
            if best is None or lr > best["L_removed"]:
                best = {"tau_B": float(tb), "tau_C_path_px": float(tc), "O_kept": round(float(ko), 4), "L_removed": round(float(lr), 4)}
    return best or {}


def cmd_eval() -> None:
    units, _ = load_units()
    nO, nL = sum(u["label"] == "O" for u in units), sum(u["label"] == "L" for u in units)
    print(f"unit berlabel O {nO}, L {nL} (gerbang: O >= 10 dan L >= 10)")
    if nO < 10 or nL < 10:
        raise SystemExit("GERBANG: label O < 10 atau L < 10 -> berhenti sebelum memilih kandidat")
    M = boot_mult(units)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out: dict = {"seed_boot": SEED_BOOT, "n_boot": N_BOOT, "n_units": {"O": nO, "L": nL}, "auc": {}, "retention": {}}
    for f, (d, grp) in FEATS.items():
        out["auc"][f] = {"direction": d, "group": grp, **auc_block(units, M, f, d)}
        print(f, {sc: out["auc"][f][sc]["track_weighted"] for sc in ("gabungan",)}, flush=True)
    # kandidat memisahkan: batas bawah CI AUC tingkat track (gabungan) > 0,6, berbobot DAN tanpa bobot
    sep = {}
    for f in FEATS:
        g = out["auc"][f]["gabungan"]
        lo_w, lo_u = g["track_weighted"]["ci"][0], g["track_unweighted"]["ci"][0]
        sep[f] = bool(lo_w is not None and lo_u is not None and lo_w > SEP_LOWER and lo_u > SEP_LOWER)
    out["separates_gabungan_track"] = sep
    for f in FEATS:
        out["retention"][f] = retention_block(units, M, f, FEATS[f][0])
    # B terbaik (hanya fitur B yang didaftarkan, tanpa B*): batas bawah CI berbobot terbesar
    bf = [f for f in FEATS if FEATS[f][1] == "B"]
    best_b = max(bf, key=lambda f: (out["auc"][f]["gabungan"]["track_weighted"]["ci"][0] or 0))
    out["best_B"] = best_b
    out["BC"] = {f"r{int(r * 100)}": bc_grid(units, best_b, FEATS[best_b][0], r) for r in RETENTIONS}
    # B+C skor rank-rata-rata (AUC tingkat strok, gabungan, berbobot, tanpa CI: eksploratif)
    thr = {}
    for f in FEATS:
        for r in RETENTIONS:
            thr[f"{f}_r{int(r * 100)}"] = {"feature": f, "direction": FEATS[f][0], "tau": out["retention"][f][f"r{int(r * 100)}"]["tau"]}
    for k, v in out["BC"].items():
        thr[f"BC_{k}"] = {"feature": best_b, "direction": FEATS[best_b][0], **v}
    st.jsave(SCRATCH / "thresholds.json", thr)
    st.jsave(RESULTS / "eval.json", out)
    print("best_B", best_b, "BC", out["BC"])
    print("separates:", [f for f, v in sep.items() if v])
    print("results/eval.json, thresholds.json ditulis")


# ── simulasi [4] ───────────────────────────────────
def variant_spec(name: str) -> dict:
    """Nama varian -> spesifikasi. base | A_f<lantai>_<ratio|pct> | C_L<n> | D_L<n> | B_<fitur>_r<90|95> | BC_r<90|95>."""
    v: dict = {"name": name}
    if name == "base":
        return v
    kind, *rest = name.split("_", 1)
    arg = rest[0] if rest else ""
    if kind == "A":
        f, rule = arg.split("_")
        v.update(floor=float(f[1:]), rule=rule)
    elif kind == "C":
        L = float(arg[1:])
        v.update(L_path=L, L_pre=float(math.floor(L / math.sqrt(2))))
    elif kind == "D":
        v.update(D_low=float(arg[1:]))
    elif kind in ("B", "BC"):
        thr = st.jload(SCRATCH / "thresholds.json")
        key = name if kind == "BC" else name
        if kind == "B":
            key = arg
            if key not in thr:
                raise SystemExit(f"ambang tidak ada: {key}")
            v.update(bfeat=thr[key]["feature"], bdir=thr[key]["direction"], btau=thr[key]["tau"])
        else:
            t = thr[name]
            v.update(bfeat=t["feature"], bdir=t["direction"], btau=t["tau_B"], path_min=t["tau_C_path_px"])
    else:
        raise SystemExit(f"varian tidak dikenal: {name}")
    return v


def path_len(pts: list) -> float:
    p = np.asarray(pts, float)
    return float(np.hypot(*np.diff(p, axis=0).T).sum()) if len(p) > 1 else 0.0


def keep_components_D(skel: np.ndarray, L: float, L_low: float, prev_mask: np.ndarray | None, ox: int, oy: int) -> np.ndarray:
    """Pembanding D (histeresis temporal, docs/04 "Tahap 3b"): komponen skeleton lolos bila >= L ATAU (>= L_low dan beririsan
    dengan strok oklusi frame t-1 yang didilatasi 2 px)."""
    n, lab, stt, _ = cv2.connectedComponentsWithStats(skel.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    area = stt[:, cv2.CC_STAT_AREA]
    keep[1:] = area[1:] >= L
    if prev_mask is not None:
        h, w = skel.shape
        crop = prev_mask[oy:oy + h, ox:ox + w]
        if crop.shape == skel.shape:
            hit = np.zeros(n, bool)
            hit[np.unique(lab[crop & (lab > 0)])] = True
            keep[1:] |= (area[1:] >= L_low) & hit[1:]
    return keep[lab]


def surviving_seeds_D(gmap, seed, params, skip, L_low, prev_mask):
    """Salinan vec.surviving_seeds dengan aturan D (komponen >= L atau >= L_low beririsan strok t-1)."""
    dp = vec.depth_params(params)
    surv = np.zeros(gmap.shape, bool)
    for gid in (int(g) for g in np.unique(gmap[seed]) if int(g) not in skip):
        thinned = vec.thin_band(seed & (gmap == gid), params["line_min_px"])
        if thinned is None:
            continue
        skel, ox, oy = thinned
        ys, xs = np.nonzero(keep_components_D(skel, dp["min_len_px"], L_low, prev_mask, ox, oy))
        surv[ys + oy, xs + ox] = True
    n, lab = cv2.connectedComponents(seed.astype(np.uint8), connectivity=8)
    ok = np.zeros(n, bool)
    ok[np.unique(lab[surv])] = True
    ok[0] = False
    return ok[lab]


def sim_occlusion(gmap, depth, names, params, t_high, t_low, v, prev_mask) -> list[dict]:
    """Salinan occlusion_strokes (src/rotoscope/vectorize.py) dengan kait kandidat A (ambang di luar), C (L jalur), D. Varian
    netral menghasilkan keluaran identik dengan produksi (dicek terhadap contours/)."""
    p = dict(params)
    if v.get("L_pre") is not None:
        p[f"{vec.DEPTH_PREFIX}min_len_px"] = v["L_pre"]
    dp = vec.depth_params(p)
    m = vec.occlusion_masks(gmap, depth, dp, t_high, t_low)
    skip = {names.index(g) + 1 for g in dp["exclude_groups"] if g in names}
    mask = m["dist_ok"]
    if 0 < dp["min_dist_low_px"] < dp["min_dist_px"]:
        dist = vec.boundary_distance(gmap)
        kept = (surviving_seeds_D(gmap, mask, p, skip, v["D_low"], prev_mask) if v.get("D_low") is not None
                else vec.surviving_seeds(gmap, mask, p, skip))
        ridge = m["hyst"] & (gmap != stb.BACKGROUND_ID) & (dist >= dp["min_dist_low_px"]) & ~mask
        ext = vec.reach_extension(kept, ridge, dist, dp["min_dist_px"], dp["min_dist_low_px"])
        mask = kept | ext
    out = []
    for gid in (int(g) for g in np.unique(gmap[mask]) if int(g) not in skip):
        thinned = vec.thin_band(mask & (gmap == gid), p["line_min_px"])
        if thinned is None:
            continue
        skel, ox, oy = thinned
        if v.get("D_low") is not None:
            skel = keep_components_D(skel, dp["min_len_px"], v["D_low"], prev_mask, ox, oy)
        else:
            skel = vec.keep_long_components(skel, dp["min_len_px"])
        polys, _ = vec.trace_skeleton(skel, p["min_stroke_px"])
        for poly in polys:
            body = poly[:-1] if vec._is_loop(poly) else poly
            strength = float(np.mean([m["mag"][py + oy, px + ox] for py, px in body], dtype=np.float64))
            pts = [vec._point(px + ox, py + oy) for py, px in poly]
            if v.get("L_path") is not None and path_len(pts) < v["L_path"]:
                continue
            out.append({"type": vec.TYPE_OCCLUSION, "closed": False, "groups": [names[gid - 1]], "points": pts,
                        "strength": round(strength, vec.STRENGTH_DECIMALS), "_pair": (gid, 0)})
    return out


def apply_b_filter(clip: st.ClipData, i: int, occ: list[dict], gmap, depth, mag, v) -> list[dict]:
    keep = []
    for s in occ:
        gid = clip.names.index(s["groups"][0]) + 1
        r = st.stroke_row(clip, i, s, gmap, depth, mag, gid)
        val = r.get(v["bfeat"] if v["bfeat"] != "strength_abs" else "strength")
        if v["bfeat"] == "strength_abs" and val is not None:
            val = val * clip.t_high
        ok = True if val is None else (v["bdir"] * val >= v["btau"])
        if ok and v.get("path_min") is not None:
            ok = r["path_px"] >= v["path_min"]
        if ok:
            keep.append(s)
    return keep


def lower_clothing_frames(clip: st.ClipData, frames_strokes: dict[int, list[dict]]) -> list[int]:
    from rotoscope.config import load_class_names
    lc = list(load_class_names()).index("Lower_Clothing")
    ok = []
    for i in st.LEG_FRAMES:
        cm = cv2.imread(str(clip.dir / "seg" / "classmap" / f"frame_{i:05d}.png"), cv2.IMREAD_UNCHANGED)
        for s in frames_strokes.get(i, []):
            if s["type"] != "occlusion":
                continue
            pts = [(int(y), int(x)) for x, y in s["points"]]
            if sum(cm[y, x] == lc for y, x in pts) / len(pts) >= 0.8:
                ok.append(i)
                break
    return ok


def run_sim(clip_name: str, name: str, timing_only: bool = False) -> dict:
    import temporal_metrics as tm

    v = variant_spec(name)
    clip = st.ClipData(clip_name)
    man = st.jload(clip.dir / "contours" / "manifest.json")["vectorize"]
    params = dict(man)
    names = tuple(clip.names)
    th, tl = clip.t_high, clip.t_low
    if v.get("floor") is not None:
        f = v["floor"]
        if f > th:
            tl = max(tl, f * clip.t_low / clip.t_high) if v["rule"] == "ratio" else tl
            th = f
    tracker = trk.Tracker(params[vec.TRACK_PREFIX + "max_match_dist_px"], inherit_main=vec.INHERIT_MAIN_SILHOUETTE)
    prev_mask = None
    fl, final_by_frame, t_total, t_occ = [], {}, [], []
    mismatches = 0
    sigma = clip.sigma
    for i in range(len(clip.docs)):
        gmap = stb.read_groups(clip.dir / "stable" / "groups" / f"{clip.frame_name(i)}.png")
        depth = vec.read_depth_smooth(clip.dir / "stable" / "depth_smooth" / f"{clip.frame_name(i)}.npy", clip.height, clip.width)
        t0 = time.perf_counter()
        strokes, _ = vec.vectorize_gmap(gmap, names, params)
        t1 = time.perf_counter()
        occ = sim_occlusion(gmap, depth, names, params, th, tl, v, prev_mask)
        occ.sort(key=vec._sort_key)
        for s in occ:
            s.pop("_pair", None)
        if v.get("bfeat"):
            mag = vec.depth_gradient(depth, sigma)[0]
            occ = apply_b_filter(clip, i, occ, gmap, depth, mag, v)
        t2 = time.perf_counter()
        allst = strokes + occ
        tracker.step(allst, vec.hair_face_mask(gmap, names))
        t3 = time.perf_counter()
        t_total.append(t3 - t0)
        t_occ.append(t2 - t1)
        fl.append({(s["type"], s["track_id"]): tm.stroke_length(s["points"], s["closed"]) for s in allst})
        final_by_frame[i] = [s for s in allst if s["type"] == vec.TYPE_OCCLUSION]
        if v.get("D_low") is not None:
            prev_mask = np.zeros(gmap.shape, np.uint8)
            for s in final_by_frame[i]:
                pp = np.floor(np.asarray(s["points"], float)).astype(int)
                prev_mask[pp[:, 1], pp[:, 0]] = 1
            prev_mask = cv2.dilate(prev_mask, np.ones((5, 5), np.uint8)).astype(bool)
        if name == "base" and json.dumps(allst) != json.dumps(clip.docs[i]["strokes"]):
            mismatches += 1
    res: dict = {"variant": name, "clip": clip_name, "spec": v, "t_high_used": th, "t_low_used": tl,
                 "time_ms": {"total_mean": round(1000 * float(np.mean(t_total)), 3), "total_p95": round(1000 * float(np.percentile(t_total, 95)), 3),
                             "occ_mean": round(1000 * float(np.mean(t_occ)), 3), "occ_p95": round(1000 * float(np.percentile(t_occ, 95)), 3)}}
    if name == "base":
        res["baseline_mismatch_frames"] = mismatches
    if timing_only:
        return res
    pop = tm.pop_energy_by_type(fl, ("silhouette", "silhouette_hole", "group_boundary", "occlusion"))
    o = pop["types"]["occlusion"]
    occ_n = np.array([len(final_by_frame[i]) for i in range(len(clip.docs))])
    res["metrics"] = {"occ_total": int(occ_n.sum()), "occ_per_frame": round(float(occ_n.mean()), 4),
                      "frames_without_occ_pct": round(float(100 * (occ_n == 0).mean()), 2), "age_mean": round(o["age_mean"], 4),
                      "age_median": o["age_median"], "pop_occ": round(o["pop_energy_mean"], 5),
                      "id_new_per_stroke_frame": round(o["id_new_per_stroke_frame"], 4), "pop_total": round(pop["total_pop_energy_mean"], 5)}
    if clip_name in ("test", "test_short"):
        okf = lower_clothing_frames(clip, final_by_frame)
        res["done_when_kaki"] = {"frames_ok": okf, "n_ok": len(okf), "of": 5}
    # kecocokan dengan strok berlabel (unit track berlabel; label diperluas ke semua strok dalam unit)
    if clip_name in ("test", "Klip2", "klip3"):
        units, _ = load_units()
        tallies = []
        for u in units:
            if u["clip"] != clip_name:
                continue
            kept = 0
            for r in u["rows"]:
                base = next(s for s in clip.docs[r["frame"]]["strokes"] if s["type"] == "occlusion" and s["track_id"] == r["track_id"])
                cand = final_by_frame[r["frame"]]
                if not cand:
                    continue
                tree = cKDTree(np.concatenate([np.asarray(s["points"], float) for s in cand]))
                d = tree.query(np.asarray(base["points"], float))[0]
                kept += int(np.mean(d <= MATCH_RADIUS) >= MATCH_FRACTION)
            tallies.append({"key": list(u["key"]), "label": u["label"], "w": u["w"], "cell": u["cell"], "n": len(u["rows"]), "kept": kept})
        res["unit_tallies"] = tallies
    return res


def cmd_sim(args: list[str]) -> None:
    name, clips = args[0], args[1:] or list(st.HASH_CLIPS)
    SIM_DIR.mkdir(parents=True, exist_ok=True)
    for c in clips:
        r = run_sim(c, name)
        st.jsave(SIM_DIR / f"{name}__{c}.json", r)
        print(name, c, r.get("metrics"), r.get("done_when_kaki"), r.get("baseline_mismatch_frames"), r["time_ms"], flush=True)


ALL_VARIANTS_FIXED = (["base"] + [f"A_f{f}_{rule}" for f in (0.02, 0.025, 0.03, 0.04, 0.05) for rule in ("ratio", "pct")]
                      + [f"C_L{L}" for L in (30, 40, 50, 60)] + ["D_L15", "D_L20"])


def cmd_simall(args: list[str]) -> None:
    variants = args or ALL_VARIANTS_FIXED
    jobs = [(v, c) for v in variants for c in st.HASH_CLIPS]
    jobs.sort(key=lambda j: j[1] != "klip3")          # klip3 dulu (terpanjang)

    def go(job):
        v, c = job
        if (SIM_DIR / f"{v}__{c}.json").exists():
            return f"lewati {v} {c}"
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "t305c_study.py"), "sim", v, c], capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        return f"{v} {c} rc={r.returncode} {r.stdout.strip()[-200:]} {r.stderr.strip()[-300:]}"

    with ThreadPoolExecutor(max_workers=8) as ex:
        for line in ex.map(go, jobs):
            print(line, flush=True)


def cmd_timing(args: list[str]) -> None:
    """Biaya [4] per frame, BERURUTAN (tanpa proses paralel). Ulangi tiap varian 3 kali, ambil median dari mean."""
    variants = args or ["base"]
    out = {}
    for v in variants:
        for c in ("test", "klip3"):
            runs = [run_sim(c, v, timing_only=True)["time_ms"] for _ in range(3)]
            out[f"{v}__{c}"] = {k: round(float(np.median([r[k] for r in runs])), 3) for k in runs[0]}
            print(v, c, out[f"{v}__{c}"], flush=True)
    path = SCRATCH / "results" / "timing.json"
    old = st.jload(path) if path.exists() else {}
    st.jsave(path, {**old, **out})


def cmd_explore(args: list[str]) -> None:
    """EKSPLORATIF (ditambahkan SETELAH label dilihat; bukan fitur terdaftar): maks |grad| sepanjang strok (kriteria benih hysteresis
    yang sebenarnya dipakai lantai A), AUC + retensi dengan mesin yang sama."""
    units, _ = load_units()
    M = boot_mult(units)
    cache: dict = {}
    for u in units:
        c = st.ClipData(u["clip"]) if u["clip"] not in cache else cache[u["clip"]]
        cache[u["clip"]] = c
        for r in u["rows"]:
            depth = c.depth(r["frame"])
            mag = vec.depth_gradient(depth, c.sigma)[0]
            s = next(x for x in c.docs[r["frame"]]["strokes"] if x["type"] == "occlusion" and x["track_id"] == r["track_id"])
            p = np.floor(np.asarray(s["points"], float)).astype(int)
            r["maxgrad_abs"] = float(mag[p[:, 1], p[:, 0]].max())
    out = {"maxgrad_abs": {"auc": auc_block(units, M, "maxgrad_abs", +1), "retention": retention_block(units, M, "maxgrad_abs", +1)}}
    st.jsave(RESULTS / "explore.json", out)
    for sc in ("gabungan", "test", "klip3"):
        print(sc, {k: v for k, v in out["maxgrad_abs"]["auc"][sc].items()})
    print(json.dumps(out["maxgrad_abs"]["retention"], indent=0)[:1800])


def tally_stats(tallies: list[dict], idx_of: dict, M: np.ndarray, clips: tuple[str, ...]) -> dict:
    """L terbuang / O tersisa berbobot (strok berlabel, label diperluas ke track) + CI bootstrap per track."""
    t = [x for x in tallies if x["key"][0] in clips and x["label"] in ("O", "L")]
    idx = np.array([idx_of[tuple(x["key"])] for x in t])
    n, kept = np.array([x["n"] for x in t], float), np.array([x["kept"] for x in t], float)
    w = np.array([x["w"] for x in t])
    isL = np.array([x["label"] == "L" for x in t])

    def calc(mult):
        ww = w * mult
        lr = (ww[isL] * (n - kept)[isL]).sum() / max((ww[isL] * n[isL]).sum(), 1e-12)
        ok = (ww[~isL] * kept[~isL]).sum() / max((ww[~isL] * n[~isL]).sum(), 1e-12)
        return lr, ok

    lr, ok = calc(np.ones(len(t)))
    reps = np.array([calc(M[r, idx]) for r in range(0, N_BOOT)])
    return {"L_removed": round(float(lr), 4), "L_removed_ci": ci(reps[:, 0]), "O_kept": round(float(ok), 4), "O_kept_ci": ci(reps[:, 1])}


def cmd_simtable(args: list[str]) -> None:
    units, _ = load_units()
    idx_of = {u["key"]: u["idx"] for u in units}
    M = boot_mult(units)
    names = sorted({p.name.split("__")[0] for p in SIM_DIR.glob("*__*.json")}, key=lambda n: (ALL_VARIANTS_FIXED + [n]).index(n))
    out = {}
    lines = ["| varian | test: strok/f · tanpa oklusi % · id baru · umur · pop · Done-when | test_short: strok/f · Done-when | Klip2: strok · tanpa oklusi % · id baru · umur | klip3: strok/f · tanpa oklusi % · id baru · umur · pop | L terbuang (gab., berbobot) | O tersisa (gab., berbobot) |",
             "|---|---|---|---|---|---|---|"]
    for n in names:
        R = {c: st.jload(SIM_DIR / f"{n}__{c}.json") for c in st.HASH_CLIPS if (SIM_DIR / f"{n}__{c}.json").exists()}
        if len(R) < 4:
            continue
        m = {c: R[c]["metrics"] for c in R}
        tl = [x for c in ("test", "klip3") for x in R[c]["unit_tallies"]]
        lab = tally_stats(tl, idx_of, M, ("test", "klip3"))
        lab_clip = {c: tally_stats(R[c]["unit_tallies"], idx_of, M, (c,)) if any(x["label"] == "L" for x in R[c]["unit_tallies"]) else None for c in ("test", "klip3")}
        out[n] = {"metrics": m, "done_when": {c: R[c]["done_when_kaki"]["n_ok"] for c in ("test", "test_short")}, "labels_gabungan": lab,
                  "labels_per_clip": lab_clip, "spec": R["test"]["spec"], "t_used": {c: [R[c]["t_high_used"], R[c]["t_low_used"]] for c in R}}

        def fmt(c, extra=True):
            x = m[c]
            return f"{x['occ_per_frame']:.3f} · {x['frames_without_occ_pct']:.1f} · {x['id_new_per_stroke_frame']:.3f} · {x['age_mean']:.2f} · {x['pop_occ']:.4f}"

        def ci_s(v, c):
            return f"{100 * v:.1f}% [{100 * c[0]:.0f}–{100 * c[1]:.0f}]" if c[0] is not None else f"{100 * v:.1f}%"

        lines.append(f"| {n} | {fmt('test')} · {R['test']['done_when_kaki']['n_ok']}/5 | {m['test_short']['occ_per_frame']:.3f} · {R['test_short']['done_when_kaki']['n_ok']}/5 | "
                     f"{m['Klip2']['occ_total']} · {m['Klip2']['frames_without_occ_pct']:.1f} · {m['Klip2']['id_new_per_stroke_frame']:.3f} · {m['Klip2']['age_mean']:.2f} | {fmt('klip3')} | "
                     f"{ci_s(lab['L_removed'], lab['L_removed_ci'])} | {ci_s(lab['O_kept'], lab['O_kept_ci'])} |")
    st.jsave(RESULTS / "simtable.json", out)
    (RESULTS / "simtable.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
