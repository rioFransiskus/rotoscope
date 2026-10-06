"""T-303 studi optical flow + diagnostik pop (SEKALI PAKAI, bukan bagian src/). Hasil studi: optical flow DITOLAK berdasarkan data
(docs/04 "Hasil T-303 (ditolak)"); skrip ini dipertahankan supaya studi bisa diulang pada klip kedua (T-305). Semua baca-saja atas
work/clips/<klip>/ (frames, seg, stable, contours); bukti JSON + tabel ditulis ke --out (default work/t303/). Tanpa GPU, tanpa torch.
Kode flow di sini HANYA untuk studi (tidak ada flow di src/).

Subperintah (semua menerima --clips, default: test_short test):
  pop           pop energy per tipe strok dari contours/ (--contours label=<folder, {clip} diganti nama klip> bisa diulang)
  holes         poin 6 (a)-(c): luas lubang siluet saat lahir / mati vs min_hole_area, sweep min_hole_area, histeresis temporal
  occl          poin 6 (d): strok oklusi vs L (min_len_px) dan ambang persentil; sweep L, histeresis temporal pada L
  flowq         kualitas flow (galat alignment warp, inkonsistensi maju-mundur, ms/flow) per metode: --methods dis:medium fb:base ...
  determinism   hash flow antar run dan jumlah benang (--methods)
  fail          kasus gagal flow: overlay statis, area datar, blur gerak
  adaptive      pembanding alpha adaptif per frame vs T-302 vs alpha seragam (metrik T-302)
  grid          prototipe kernel + warp + konsistensi (alpha x R_flow x blend), --method, --preset fin|grid

Catatan: prototipe KEDALAMAN ber-flow (warp kedalaman blend 1,0, EMA, varian gate / edge) tidak dikonsolidasikan di sini; parameternya
tercatat di docs/04 (DIS MEDIUM, bilinear, alpha 0,7, R_flow 2, blend 1,0 vs 0,4).

Pakai: python scripts/t303_flow_study.py holes --clips test_short test
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tests"))

import temporal_metrics as tm  # noqa: E402
from rotoscope import depth as dep  # noqa: E402
from rotoscope import stabilize as stb  # noqa: E402
from rotoscope import track as trk  # noqa: E402
from rotoscope import vectorize as vec  # noqa: E402
from rotoscope.config import load_class_names, load_pipeline  # noqa: E402

WORK_ROOT = REPO / "work" / "clips"
DEFAULT_OUT = REPO / "work" / "t303"
DEFAULT_CLIPS = ("test_short", "test")
CFG = load_pipeline(REPO / "configs" / "default.yaml")
CLASSES = list(load_class_names())
LUT = stb.class_group_lut(CFG.groups, CLASSES)
NG = len(CFG.groups) + 1
GROUP_NAMES = tuple(g for g, _ in CFG.groups)

FLOW_KS = (1, 2, 3, 4)               # selisih frame yang diuji untuk kualitas flow
THREAD_COUNTS = (1, 4, 12)
SUN_REL, SUN_ABS = 0.01, 0.5         # aturan konsistensi Sundaram: |f+b|^2 <= 0.01 (|f|^2 + |b|^2) + 0.5
SIZE_BINS = (0.5, 0.8, 1.2, 2.0)     # tepi bin rasio relatif ambang
SIZE_BIN_LABELS = ("<0.5x", "0.5-0.8x", "0.8-1.2x", "1.2-2x", ">2x")
STRENGTH_BINS = (1.0, 1.2, 1.5, 2.0)             # kekuatan strok relatif T_high
STRENGTH_BIN_LABELS = ("<1.0x", "1.0-1.2x", "1.2-1.5x", "1.5-2x", ">2x")
HOLE_THETAS = (50, 100, 150, 200, 300)           # sweep min_hole_area (px)
HOLE_T_HIGH = 200                                # histeresis: muncul bila luas >= T_high
HOLE_T_LOWS = (100, 140, 170)
OCC_LS = (20, 30, 40, 50, 70)                    # sweep L (min_len_px, px skeleton)
OCC_L_HIGH = 30
OCC_L_LOWS = (15, 20, 25)
FOOT_FRAMES = (73, 92)


# ── Utilitas ───────────────────────────────────────
def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")


def md_table(header: list[str], rows: list[list]) -> str:
    def cell(v):
        return f"{v:.4f}" if isinstance(v, float) else str(v)

    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(cell(v) for v in r) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def bin_counts(values, ref: float, edges=SIZE_BINS, labels=SIZE_BIN_LABELS) -> dict[str, int]:
    out = {lab: 0 for lab in labels}
    for v in values:
        out[labels[int(np.searchsorted(edges, v / ref, side="right"))]] += 1
    return out


def md5(a) -> str:
    return hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()


class ClipData:
    """Akses baca-saja klip nyata: frames abu-abu, jumlah grup (lazy, cache kecil), argmax mentah."""

    def __init__(self, name: str):
        self.name = name
        self.work = WORK_ROOT / name
        self.clip = stb.load_clip(self.work)
        self.names = self.clip.names
        self.T = len(self.names)
        self.H, self.W = self.clip.height, self.clip.width
        self._gray, self._sums, self.raw = {}, collections.OrderedDict(), None
        qc = stb.qc_fail_flags(self.clip)
        self.fail = [bool(qc[n]) for n in self.names]

    def gray(self, k):
        if k not in self._gray:
            self._gray[k] = cv2.imdecode(np.fromfile(self.work / "frames" / self.names[k], np.uint8), cv2.IMREAD_GRAYSCALE)
        return self._gray[k]

    def sums(self, k):
        if k in self._sums:
            self._sums.move_to_end(k)
            return self._sums[k][0]
        probs, cm, _ = stb._read_inputs(self.clip, self.names[k], len(CLASSES))
        self._sums[k] = (stb.group_sums(probs, LUT, NG - 1), LUT[cm])
        if len(self._sums) > 14:
            self._sums.popitem(last=False)
        return self._sums[k][0]

    def tiemap(self, k):
        self.sums(k)
        return self._sums[k][1]

    def raw_maps(self):
        if self.raw is None:
            self.raw = np.stack([stb.group_argmax(self.sums(k), self.tiemap(k))[0] for k in range(self.T)])
        return self.raw

    def windows(self):
        raw = self.raw_maps()
        return {"static": tm.window_mask(self.T, tm.static_windows(raw)), "fast": tm.window_mask(self.T, tm.fast_windows(raw)),
                "foot": np.isin(np.arange(self.T), np.arange(FOOT_FRAMES[0], FOOT_FRAMES[1] + 1)), "all": np.ones(self.T, bool)}


# ── Flow (HANYA studi) + warp + konsistensi ────────
def fb_flow(a, b, variant="base"):
    if variant == "base":
        return cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    if variant == "big":
        return cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 4, 25, 3, 7, 1.5, 0)
    raise ValueError(variant)


_DIS: dict = {}


def dis_flow(a, b, preset="fast"):
    if preset not in _DIS:
        _DIS[preset] = cv2.DISOpticalFlow_create({"fast": cv2.DISOPTICAL_FLOW_PRESET_FAST, "medium": cv2.DISOPTICAL_FLOW_PRESET_MEDIUM,
                                                  "ultrafast": cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST}[preset])
    return _DIS[preset].calc(a, b, None)


def flow_fn(method: str):
    kind, _, var = method.partition(":")
    if kind == "fb":
        return lambda a, b: fb_flow(a, b, var or "base")
    return lambda a, b: dis_flow(a, b, var or "fast")


def remap_xy(flow):
    h, w = flow.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return xx + flow[..., 0], yy + flow[..., 1]


def warp(q, flow):
    """q (C,H,W) → (C,H,W) float32 disampel di x+flow (bilinear; di luar gambar = 0), + valid (H,W)."""
    mx, my = remap_xy(flow)
    out = np.stack([cv2.remap(q[c].astype(np.float32), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                    for c in range(q.shape[0])])
    h, w = flow.shape[:2]
    return out, (mx >= 0) & (mx <= w - 1) & (my >= 0) & (my <= h - 1)


def fb_err2(f, g):
    mx, my = remap_xy(f)
    bx = cv2.remap(g[..., 0], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    by = cv2.remap(g[..., 1], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return (f[..., 0] + bx) ** 2 + (f[..., 1] + by) ** 2, f[..., 0] ** 2 + f[..., 1] ** 2 + bx ** 2 + by ** 2


def cons_weight(e2, m2, rule):
    """none | fix<px> | sun | soft<px> → bobot konsistensi per piksel (float32 0..1)."""
    if rule == "none":
        return np.ones(e2.shape, np.float32)
    if rule.startswith("fix"):
        return (e2 <= float(rule[3:]) ** 2).astype(np.float32)
    if rule == "sun":
        return (e2 <= SUN_REL * m2 + SUN_ABS).astype(np.float32)
    if rule.startswith("soft"):
        return np.clip(2.0 - np.sqrt(e2) / float(rule[4:]), 0.0, 1.0).astype(np.float32)
    raise ValueError(rule)


class Neigh:
    __slots__ = ("k", "raw", "W", "valid", "e2", "m2", "qk")


def build_neigh(cd, t, k, fn):
    f, g = fn(cd.gray(t), cd.gray(t + k)), fn(cd.gray(t + k), cd.gray(t))
    n = Neigh()
    n.k, n.raw = k, cd.sums(t + k)
    n.e2, n.m2 = fb_err2(f, g)
    n.W, n.valid = warp(n.raw, f)
    n.qk = CFG.stabilize.temporal.qc_fail_weight if cd.fail[t + k] else 1.0
    return n


def assemble(S, tie, center_w, neighs, rho, blend, rule, boil):
    """p = Σ w s / Σ w; s_k = b·W + (1−b)·raw; w_k = ρ^|k|·q·cons·valid (k≠0); lalu boil_preserve + filter pulau / mode."""
    acc = np.float32(center_w) * S.astype(np.float32)
    tot = np.full(S.shape[1:], center_w, np.float32)
    for n in sorted(neighs, key=lambda n: n.k):
        w = np.float32(rho ** abs(n.k) * n.qk) * cons_weight(n.e2, n.m2, rule) * n.valid
        acc += w * (np.float32(blend) * n.W + np.float32(1 - blend) * n.raw)
        tot += w
    p = acc / tot
    if boil > 0:
        p = np.float32(1 - boil) * p + np.float32(boil) * S
    return stb.clean_groups(p, tie, CFG.stabilize.island_min_px, CFG.stabilize.mode_k)[0]


def run_flow_sweep(cd, method, configs, boil, log=print):
    fn = flow_fn(method)
    out = {c["name"]: np.zeros((cd.T, cd.H, cd.W), np.uint8) for c in configs}
    rmax = max(c["rflow"] for c in configs)
    t_flow = t_asm = 0.0
    qw = CFG.stabilize.temporal.qc_fail_weight
    for t in range(cd.T):
        S, tie = cd.sums(t), cd.tiemap(t)
        t0 = time.perf_counter()
        neigh = {k: build_neigh(cd, t, k, fn) for k in range(-rmax, rmax + 1) if k != 0 and 0 <= t + k < cd.T}
        t1 = time.perf_counter()
        for c in configs:
            reff = min(stb.kernel_radius(c["alpha"]), c["rflow"])
            out[c["name"]][t] = assemble(S, tie, qw if cd.fail[t] else 1.0, [n for k, n in neigh.items() if abs(k) <= reff],
                                         stb.kernel_rho(c["alpha"]), c["blend"], c.get("rule", "sun"), boil)
        t_flow += t1 - t0
        t_asm += time.perf_counter() - t1
        if t % 40 == 0:
            log(f"  {method} t={t}/{cd.T} flow {t_flow / (t + 1):.3f}s asm {t_asm / (t + 1):.3f}s per frame")
    return out, {"flow_s_per_frame": t_flow / cd.T, "asm_s_per_frame": t_asm / cd.T}


def run_plain(cd, alpha_fn, boil):
    """Kernel simetris terpotong T-302 tanpa flow; alpha_fn(t) → alpha frame t (R, rho dari alpha itu)."""
    qw = CFG.stabilize.temporal.qc_fail_weight
    out = np.zeros((cd.T, cd.H, cd.W), np.uint8)
    for t in range(cd.T):
        S, tie = cd.sums(t), cd.tiemap(t)
        a = alpha_fn(t)
        radius, rho = stb.kernel_radius(a), stb.kernel_rho(a)
        if radius == 0:
            out[t] = stb.clean_groups(S, tie, CFG.stabilize.island_min_px, CFG.stabilize.mode_k)[0]
            continue
        acc, tot = np.zeros(S.shape, np.float32), 0.0
        for j in range(max(t - radius, 0), min(t + radius, cd.T - 1) + 1):
            w = rho ** abs(j - t) * (qw if cd.fail[j] else 1.0)
            acc += np.float32(w) * cd.sums(j)
            tot += w
        p = acc / np.float32(tot)
        if boil > 0:
            p = np.float32(1 - boil) * p + np.float32(boil) * S
        out[t] = stb.clean_groups(p, tie, CFG.stabilize.island_min_px, CFG.stabilize.mode_k)[0]
    return out


def motion_u(cd):
    """u_t ∈ [0,1] dari argmax mentah: u_j = rata-rata(clip((kecepatan − S_stat)/(S_fast − S_stat)), clip((XOR − X_stat)/(X_fast − X_stat)));
    u_t = max(u_t, u_{t+1}) (perubahan yang menyentuh frame t). Ambang = aturan jendela T-302 (temporal_metrics)."""
    raw = cd.raw_maps()
    us = np.clip((tm.centroid_speed(raw) - tm.STATIC_SPEED_MAX_PX) / (tm.FAST_SPEED_MIN_PX - tm.STATIC_SPEED_MAX_PX), 0, 1)
    ux = np.clip((tm.xor_fraction(raw) - tm.STATIC_XOR_MAX) / (tm.FAST_XOR_MIN - tm.STATIC_XOR_MAX), 0, 1)
    u = (us + ux) / 2
    u[0] = u[1] if len(u) > 1 else 0
    u2 = u.copy()
    u2[:-1] = np.maximum(u[:-1], u[1:])
    return u2


def adaptive_alpha_fn(cd, a_static, a_fast):
    u = motion_u(cd)
    return lambda t: float(a_static + (a_fast - a_static) * u[t])


def fidelity(g, raw, wins):
    gi = tm.group_iou(g, raw)
    lr = tm.limb_ratio(g, raw)
    f = {"group_iou_mean": float(gi.mean()), "group_iou_min": float(gi.min()),
         "centroid_err_max": float(np.nanmax(tm.centroid_error(g, raw))), "limb_ratio_min_all": float(lr.min()),
         "limb_ratio_min_fast": float(lr[wins["fast"]].min()) if wins["fast"].any() else None,
         "fg_iou_mean": float(tm.fg_iou(g, raw).mean())}
    return f


def metrics(g, raw, wins, base_ff):
    r = {}
    for wn in ("static", "fast", "all"):
        ff = tm.flipflop_per10k(g, wins[wn])
        r[f"ff_{wn}"], r[f"ff_{wn}_vs_a1"] = ff, ff / base_ff[wn] - 1.0
    return r | fidelity(g, raw, wins)


def base_flipflop(cd, wins, boil):
    g1 = run_plain(cd, lambda t: 1.0, boil)
    return {wn: tm.flipflop_per10k(g1, wins[wn]) for wn in ("static", "fast", "all")}


# ── pop ────────────────────────────────────────────
def cmd_pop(args) -> None:
    specs = dict(s.split("=", 1) for s in args.contours) or {"t302_default": str(WORK_ROOT / "{clip}" / "contours")}
    res, rows = {}, []
    for clip in args.clips:
        for label, pattern in specs.items():
            r = tm.pop_energy_by_type(tm.stroke_lengths_from_dir(Path(pattern.format(clip=clip))))
            res[f"{clip}/{label}"] = r
            for ty, v in r["types"].items():
                rows.append([clip, label, ty, v["pop_energy_mean"], v["pop_share_pct"], v["pop_per_length_share"],
                             v["id_new_per_stroke_frame"], v["age_mean"], v["frames_without"]])
            rows.append([clip, label, "TOTAL", r["total_pop_energy_mean"], 100.0, None, None, None, None])
    write_json(args.out / "pop.json", res)
    (args.out / "pop.md").write_text(md_table(["klip", "label", "tipe", "pop", "porsi %", "pop/porsi panjang", "id baru/strok-frame", "umur",
                                               "frame tanpa"], rows), encoding="utf-8")
    print((args.out / "pop.md").read_text(encoding="utf-8"))


# ── poin 6: lubang siluet ──────────────────────────
def hole_candidates(gmap: np.ndarray, min_region_area: int) -> list[tuple[dict, int]]:
    """Seperti vectorize.silhouette_strokes dengan min_hole_area = 0, tetapi mengembalikan (strok lubang, luas px) — luas = komponen
    background 4-arah yang dilingkupi (sama dengan yang dibandingkan produksi dengan min_hole_area). Terurut seperti produksi."""
    pad = vec.PAD_PX
    padded = cv2.copyMakeBorder((gmap != stb.BACKGROUND_ID).astype(np.uint8), pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    contours, hier = cv2.findContours(padded, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return []
    parent = [int(h[3]) for h in hier[0]]
    _, lab, st, _ = cv2.connectedComponentsWithStats(padded, connectivity=8)
    _, blab, bst, _ = cv2.connectedComponentsWithStats(1 - padded, connectivity=4)
    keep = {}
    for i, cnt in enumerate(contours):
        if parent[i] < 0:
            x, y = cnt[0, 0]
            keep[i] = int(st[lab[y, x], cv2.CC_STAT_AREA]) >= min_region_area
    out = []
    for i, cnt in enumerate(contours):
        if parent[i] < 0 or not keep[parent[i]]:
            continue
        label = vec._hole_label(padded, blab, cnt)
        if label >= 0:
            area = int(bst[label, cv2.CC_STAT_AREA])
        else:
            x, y, w, h = cv2.boundingRect(cnt)
            m = np.zeros((h, w), np.uint8)
            cv2.drawContours(m, [cnt], -1, 1, cv2.FILLED, offset=(-x, -y))
            area = int((m.astype(bool) & (padded[y:y + h, x:x + w] == 0)).sum())
        out.append((vec._closed_stroke(vec.TYPE_HOLE, cnt), area))
    out.sort(key=lambda p: vec._sort_key(p[0]))
    return out


def track_series(items_per_frame, max_dist: float):
    """items_per_frame: per frame list (strok mentah, atribut dict). Lacak dengan trk.Tracker → per frame {track_id: atribut + panjang}."""
    tracker = trk.Tracker(max_dist)
    series = []
    for items in items_per_frame:
        strokes = [s for s, _ in items]
        tracker.step(strokes)
        series.append({s["track_id"]: {**attr, "len": tm.stroke_length(s["points"], s["closed"])} for s, (_, attr) in zip(strokes, items)})
    return series


def shown_plain(series, key, thr):
    return [{tid for tid, a in f.items() if a[key] >= thr} for f in series]


def shown_hysteresis(series, key, t_high, t_low):
    """Ada bila nilai >= t_high; bertahan selama track hidup sampai nilai < t_low."""
    out, prev = [], set()
    for f in series:
        cur = {tid for tid, a in f.items() if a[key] >= (t_low if tid in prev else t_high)}
        out.append(cur)
        prev = cur
    return out


def pop_of(shown, series, type_name: str, other_len: list[float]) -> dict:
    """Pop energy tipe `type_name` dari himpunan track tampil; penyebut = panjang strok tipe lain (produksi) + yang tampil."""
    frames = []
    for sh, f, ol in zip(shown, series, other_len):
        d = {(type_name, tid): f[tid]["len"] for tid in sh}
        d[("rest", 0)] = ol
        frames.append(d)
    r = tm.pop_energy_by_type(frames, types=(type_name,))["types"][type_name]
    return {"pop_energy_mean": r["pop_energy_mean"], "born_per_frame": r["born_per_frame"], "died_per_frame": r["died_per_frame"],
            "shown_per_frame": float(np.mean([len(s) for s in shown])), "id_new_per_stroke_frame": r["id_new_per_stroke_frame"],
            "age_mean": r["age_mean"]}


def events(shown, series, key):
    """Lahir / mati pada deret tampil. Sebab: 'crossing' = track masih ada di frame yang lain tetapi di bawah ambang; 'new' / 'vanished'
    = track mentah baru muncul / hilang. Luas = nilai `key` saat lahir (frame t) / saat terakhir tampil (frame t−1)."""
    births, deaths = [], []
    for t in range(1, len(shown)):
        for tid in shown[t] - shown[t - 1]:
            a, before = series[t][tid], series[t - 1].get(tid)
            births.append({"v": a[key], "len": a["len"], "cause": "crossing" if before is not None else "new",
                           "v_other": before[key] if before is not None else None})
        for tid in shown[t - 1] - shown[t]:
            a, after = series[t - 1][tid], series[t].get(tid)
            deaths.append({"v": a[key], "len": a["len"], "cause": "crossing" if after is not None else "vanished",
                           "v_other": after[key] if after is not None else None})
    return births, deaths


def event_summary(births, deaths, ref: float, edges=SIZE_BINS, labels=SIZE_BIN_LABELS) -> dict:
    def part(ev, causes):
        total_len = sum(e["len"] for e in ev) or 1.0
        out = {"n": len(ev), "hist_v_over_ref": bin_counts([e["v"] for e in ev], ref, edges, labels),
               "len_share_by_cause": {c: sum(e["len"] for e in ev if e["cause"] == c) / total_len for c in causes},
               "n_by_cause": {c: sum(1 for e in ev if e["cause"] == c) for c in causes}}
        cross = [e["v_other"] for e in ev if e["cause"] == "crossing"]
        out["hist_other_side_over_ref(crossing)"] = bin_counts(cross, ref, edges, labels) if cross else {}
        return out

    return {"births": part(births, ("crossing", "new")), "deaths": part(deaths, ("crossing", "vanished"))}


def raw_track_events(series, key, ref: float) -> dict:
    """Lahir / mati track MENTAH (tanpa ambang): nilai di frame pertama / terakhir track, relatif ref."""
    born, died = [], []
    for t in range(1, len(series)):
        born += [series[t][tid][key] for tid in series[t].keys() - series[t - 1].keys()]
        died += [series[t - 1][tid][key] for tid in series[t - 1].keys() - series[t].keys()]
    return {"n_born": len(born), "n_died": len(died), "hist_born_over_ref": bin_counts(born, ref), "hist_died_over_ref": bin_counts(died, ref),
            "frac_born_below_ref": float(np.mean([v < ref for v in born])) if born else None,
            "frac_died_below_ref": float(np.mean([v < ref for v in died])) if died else None}


def cmd_holes(args) -> None:
    params = vec.vectorize_params(CFG)
    theta0 = params["min_hole_area"]
    max_dist = params[vec.TRACK_PREFIX + "max_match_dist_px"]
    allres = {}
    for clipname in args.clips:
        clip = vec.load_clip(WORK_ROOT / clipname)
        real = tm.stroke_lengths_from_dir(clip.contours_dir)
        other_len = [sum(v for k, v in f.items() if k[0] != "silhouette_hole") for f in real]
        t0 = time.perf_counter()
        items = []
        for name in clip.names:
            gmap = stb.read_groups(clip.stable_clip.groups_path(name))
            items.append([(s, {"area": a}) for s, a in hole_candidates(gmap, params["min_region_area"])])
        series = track_series(items, max_dist)
        print(f"{clipname}: lubang mentah dilacak ({time.perf_counter() - t0:.1f} s), rata-rata {np.mean([len(f) for f in series]):.1f}/frame")
        res = {"clip": clipname, "min_hole_area_prod": theta0, "t_high_hysteresis": HOLE_T_HIGH,
               "raw_holes_per_frame_mean": float(np.mean([len(f) for f in series])),
               "prod_hole_pop_energy": tm.pop_energy_by_type(real)["types"]["silhouette_hole"]["pop_energy_mean"]}
        res["raw_tracks"] = raw_track_events(series, "area", theta0)
        plain = shown_plain(series, "area", theta0)
        res["sim_theta_prod"] = pop_of(plain, series, "silhouette_hole", other_len)
        res["events_theta_prod"] = event_summary(*events(plain, series, "area"), theta0)
        res["sweep_min_hole_area"] = {str(th): pop_of(shown_plain(series, "area", th), series, "silhouette_hole", other_len)
                                      for th in HOLE_THETAS}
        hyst = {}
        for tl in HOLE_T_LOWS:
            sh = shown_hysteresis(series, "area", HOLE_T_HIGH, tl)
            extra = [len(h - p) for h, p in zip(sh, plain)]
            ex_area = [series[t][tid]["area"] for t in range(len(sh)) for tid in sh[t] - plain[t]]
            hyst[str(tl)] = pop_of(sh, series, "silhouette_hole", other_len) | {
                "extra_vs_plain_per_frame_mean": float(np.mean(extra)), "extra_max": int(max(extra)),
                "frames_with_extra": int(sum(1 for e in extra if e)), "extra_area_mean": float(np.mean(ex_area)) if ex_area else None,
                "events": event_summary(*events(sh, series, "area"), theta0)}
        res["hysteresis"] = hyst
        allres[clipname] = res
        write_json(args.out / f"holes_{clipname}.json", res)
    rows = []
    for c, r in allres.items():
        rows.append([c, "produksi (contours/)", r["prod_hole_pop_energy"], None, None])
        rows.append([c, f"simulasi θ={theta0}", r["sim_theta_prod"]["pop_energy_mean"], r["sim_theta_prod"]["shown_per_frame"], 0])
        for th, v in r["sweep_min_hole_area"].items():
            rows.append([c, f"θ={th}", v["pop_energy_mean"], v["shown_per_frame"], None])
        for tl, v in r["hysteresis"].items():
            rows.append([c, f"hist {HOLE_T_HIGH}/{tl}", v["pop_energy_mean"], v["shown_per_frame"], v["extra_vs_plain_per_frame_mean"]])
    (args.out / "holes.md").write_text(md_table(["klip", "konfigurasi", "pop energy lubang", "lubang tampil/frame", "ekstra vs θ prod/frame"], rows),
                                        encoding="utf-8")
    print((args.out / "holes.md").read_text(encoding="utf-8"))


# ── poin 6: strok oklusi ───────────────────────────
def occlusion_candidates(gmap, depth, params, t_high, t_low) -> list[tuple[dict, dict]]:
    """Seperti vectorize.occlusion_strokes dengan L = 0; tiap strok diberi ukuran komponen skeleton asalnya (px, = yang dibandingkan L)."""
    dp = vec.depth_params(params)
    m = vec.occlusion_masks(gmap, depth, dp, t_high, t_low)
    out = []
    for gid in (int(g) for g in np.unique(gmap[m["dist_ok"]])):
        thinned = vec.thin_band(m["dist_ok"] & (gmap == gid), params["line_min_px"])
        if thinned is None:
            continue
        skel, ox, oy = thinned
        _, lab, st, _ = cv2.connectedComponentsWithStats(skel.astype(np.uint8), connectivity=8)
        polys, _ = vec.trace_skeleton(skel, params["min_stroke_px"])
        for poly in polys:
            body = poly[:-1] if vec._is_loop(poly) else poly
            strength = float(np.mean([m["mag"][py + oy, px + ox] for py, px in body], dtype=np.float64))
            comp = int(st[lab[body[0][0], body[0][1]], cv2.CC_STAT_AREA])
            s = {"type": vec.TYPE_OCCLUSION, "closed": False, "groups": [GROUP_NAMES[gid - 1]],
                 "points": [vec._point(px + ox, py + oy) for py, px in poly], "strength": round(strength, vec.STRENGTH_DECIMALS),
                 "_pair": (gid, 0)}
            out.append((s, {"size": comp, "strength": strength}))
    out.sort(key=lambda p: vec._sort_key(p[0]))
    for s, _ in out:
        s.pop("_pair", None)
    return out


def cmd_occl(args) -> None:
    params = vec.vectorize_params(CFG)
    L0 = vec.depth_params(params)["min_len_px"]
    max_dist = params[vec.TRACK_PREFIX + "max_match_dist_px"]
    allres = {}
    for clipname in args.clips:
        clip = vec.load_clip(WORK_ROOT / clipname)
        stats = json.loads(clip.clip_stats_path.read_text(encoding="utf-8"))
        t_high, t_low = stats["t_high"], stats["t_low"]
        real = tm.stroke_lengths_from_dir(clip.contours_dir)
        other_len = [sum(v for k, v in f.items() if k[0] != "occlusion") for f in real]
        t0 = time.perf_counter()
        items = []
        for name in clip.names:
            gmap, depth = vec.load_frame_inputs(clip, name, len(GROUP_NAMES))
            items.append(occlusion_candidates(gmap, depth, params, t_high, t_low))
        series = track_series(items, max_dist)
        print(f"{clipname}: oklusi L=0 dilacak ({time.perf_counter() - t0:.1f} s), rata-rata {np.mean([len(f) for f in series]):.1f}/frame")
        res = {"clip": clipname, "L_prod": L0, "t_high": t_high, "t_low": t_low,
               "prod_occlusion_pop_energy": tm.pop_energy_by_type(real)["types"]["occlusion"]["pop_energy_mean"],
               "raw_tracks_size": raw_track_events(series, "size", L0)}
        plain = shown_plain(series, "size", L0)
        res["sim_L_prod"] = pop_of(plain, series, "occlusion", other_len)
        res["events_L_prod_size_over_L"] = event_summary(*events(plain, series, "size"), L0)
        res["events_L_prod_strength_over_T_high"] = event_summary(*events(plain, series, "strength"), t_high, STRENGTH_BINS, STRENGTH_BIN_LABELS)
        res["sweep_L"] = {str(L): pop_of(shown_plain(series, "size", L), series, "occlusion", other_len) for L in OCC_LS}
        hyst = {}
        for ll in OCC_L_LOWS:
            sh = shown_hysteresis(series, "size", OCC_L_HIGH, ll)
            extra = [len(h - p) for h, p in zip(sh, plain)]
            hyst[str(ll)] = pop_of(sh, series, "occlusion", other_len) | {"extra_vs_plain_per_frame_mean": float(np.mean(extra)),
                                                                          "extra_max": int(max(extra))}
        res["hysteresis_L"] = hyst
        allres[clipname] = res
        write_json(args.out / f"occl_{clipname}.json", res)
    rows = []
    for c, r in allres.items():
        rows.append([c, "produksi (contours/)", r["prod_occlusion_pop_energy"], None, None])
        rows.append([c, f"simulasi L={r['L_prod']}", r["sim_L_prod"]["pop_energy_mean"], r["sim_L_prod"]["shown_per_frame"], 0])
        for L, v in r["sweep_L"].items():
            rows.append([c, f"L={L}", v["pop_energy_mean"], v["shown_per_frame"], None])
        for ll, v in r["hysteresis_L"].items():
            rows.append([c, f"hist {OCC_L_HIGH}/{ll}", v["pop_energy_mean"], v["shown_per_frame"], v["extra_vs_plain_per_frame_mean"]])
    (args.out / "occl.md").write_text(md_table(["klip", "konfigurasi", "pop energy oklusi", "strok tampil/frame", "ekstra vs L prod/frame"], rows),
                                      encoding="utf-8")
    print((args.out / "occl.md").read_text(encoding="utf-8"))


# ── flow: kualitas, determinisme, kasus gagal ──────
def cmd_flowq(args) -> None:
    for clipname in args.clips:
        cd = ClipData(clipname)
        raw, wins, T = cd.raw_maps(), cd.windows(), cd.T
        fg_masks = raw != 0
        result = {"clip": clipname, "T": T, "windows": {k: int(v.sum()) for k, v in wins.items()}, "methods": {}}
        for method in args.methods:
            fn = flow_fn(method)
            rows = {k: [] for k in FLOW_KS}
            times = []
            wall0 = time.perf_counter()
            for k in FLOW_KS:
                for t in range(T - k):
                    t0 = time.perf_counter()
                    f = fn(cd.gray(t), cd.gray(t + k))
                    t1 = time.perf_counter()
                    g = fn(cd.gray(t + k), cd.gray(t))
                    times += [t1 - t0, time.perf_counter() - t1]
                    err = tm.fb_error(f, g)
                    q, valid = warp(cd.sums(t + k), f)
                    r = {"t": t} | tm.alignment_error(raw[t], q.argmax(axis=0).astype(np.uint8), raw[t + k], valid)
                    fg = fg_masks[t]
                    for th, v in tm.inconsistent_fraction(err).items():
                        r[f"inc_all_{th}"] = v
                    for th, v in tm.inconsistent_fraction(err, fg).items():
                        r[f"inc_fg_{th}"] = v
                    r["oob_frac"] = float((~valid).mean())
                    rows[k].append(r)
            agg = {}
            for k in FLOW_KS:
                for wn, wm in wins.items():
                    sel = [r for r in rows[k] if wm[r["t"]] and wm[min(r["t"] + k, T - 1)]]
                    if sel:
                        agg[f"k{k}_{wn}"] = {key: float(np.mean([r[key] for r in sel])) for key in sel[0] if key != "t"} | {"n": len(sel)}
            result["methods"][method] = {"agg": agg, "ms_per_flow_median": 1000 * float(np.median(times)), "wall_s": time.perf_counter() - wall0}
            print(clipname, method, "ms/flow", round(result["methods"][method]["ms_per_flow_median"], 1), flush=True)
            write_json(args.out / f"flowq_{clipname}.json", result)
        rows_md = [[clipname, m, v["ms_per_flow_median"], v["agg"].get("k2_all", {}).get("err_warp"), v["agg"].get("k2_all", {}).get("err_nowarp"),
                    v["agg"].get("k2_static", {}).get("inc_fg_1.0"), v["agg"].get("k2_fast", {}).get("inc_fg_1.0")]
                   for m, v in result["methods"].items()]
        (args.out / f"flowq_{clipname}.md").write_text(md_table(["klip", "metode", "ms/flow", "galat warp k2", "tanpa warp k2",
                                                                 "inkons fg>1px statis", "inkons fg>1px cepat"], rows_md), encoding="utf-8")


def cmd_determinism(args) -> None:
    res = {}
    for clipname in args.clips:
        cd = ClipData(clipname)
        pairs = [(t, t + 2) for t in np.linspace(5, cd.T - 8, 4).astype(int)]
        for method in args.methods:
            fn = flow_fn(method)
            hashes = collections.defaultdict(set)
            for nthreads in THREAD_COUNTS:
                cv2.setNumThreads(nthreads)
                for _ in range(2):
                    for a, b in pairs:
                        hashes[(a, b)].add(md5(fn(cd.gray(a), cd.gray(b))))
            res[f"{clipname}/{method}"] = {"pairs": [list(map(int, p)) for p in pairs], "threads": list(THREAD_COUNTS), "runs_per_thread": 2,
                                           "identik": all(len(h) == 1 for h in hashes.values())}
            print(clipname, method, "identik" if res[f"{clipname}/{method}"]["identik"] else "BEDA", flush=True)
    write_json(args.out / "determinism.json", res)


def cmd_fail(args) -> None:
    res = {}
    for clipname in args.clips:
        cd = ClipData(clipname)
        raw, T = cd.raw_maps(), cd.T
        stack = np.stack([cd.gray(t) for t in range(T)]).astype(np.float32)
        std, mean = stack.std(axis=0), stack.mean(axis=0)
        ever_fg = cv2.dilate((raw != 0).any(axis=0).astype(np.uint8), np.ones((31, 31), np.uint8)).astype(bool)
        grad = np.hypot(cv2.Sobel(mean, cv2.CV_32F, 1, 0), cv2.Sobel(mean, cv2.CV_32F, 0, 1))
        static = std < 2.0
        overlay = static & (grad > np.percentile(grad, 90)) & ~ever_fg
        flat = (grad < np.percentile(grad, 20)) & ~ever_fg
        out = {"static_frac_of_image": float(static.mean()), "overlay_px_frac": float(overlay.mean()), "flat_px_frac": float(flat.mean())}
        fn = flow_fn("dis:medium")
        mags = {"overlay": [], "flat": [], "bg_all": []}
        errs = {"overlay": [], "flat": [], "bg_all": []}
        for t in range(0, T - 2, max(1, T // 40)):
            f, g = fn(cd.gray(t), cd.gray(t + 2)), fn(cd.gray(t + 2), cd.gray(t))
            mag, e = np.hypot(f[..., 0], f[..., 1]), tm.fb_error(f, g)
            for nm, mask in (("overlay", overlay), ("flat", flat), ("bg_all", ~ever_fg)):
                if mask.any():
                    mags[nm].append(float(mag[mask].mean()))
                    errs[nm].append(float((e[mask] > 1.0).mean()))
        out["dismed_k2_mean_flow_mag_px"] = {k: float(np.mean(v)) if v else None for k, v in mags.items()}
        out["dismed_k2_inconsistent_frac_1px"] = {k: float(np.mean(v)) if v else None for k, v in errs.items()}
        lv = []
        for t in range(T):
            lap = cv2.Laplacian(cd.gray(t), cv2.CV_32F)
            fg = raw[t] != 0
            lv.append(float(lap[fg].var()) if fg.any() else np.nan)
        lv = np.array(lv)
        sp = tm.centroid_speed(raw)
        med = float(np.nanmedian(lv))
        out["blur_frames_lt_half_median"] = [int(i) for i in np.flatnonzero(lv < 0.5 * med)]
        out["blur_frames_corr_speed"] = float(np.corrcoef(np.nan_to_num(lv, nan=med), sp)[0, 1])
        res[clipname] = out
        print(clipname, json.dumps({k: v for k, v in out.items() if k != "blur_frames_lt_half_median"}), flush=True)
    write_json(args.out / "fail.json", res)


# ── alpha adaptif + grid flow ──────────────────────
def cmd_adaptive(args) -> None:
    boil = CFG.stabilize.temporal.boil_preserve
    allres = {}
    for clipname in args.clips:
        cd = ClipData(clipname)
        raw, wins = cd.raw_maps(), cd.windows()
        base = base_flipflop(cd, wins, boil)
        u = motion_u(cd)
        res = {"clip": clipname, "windows": {k: int(v.sum()) for k, v in wins.items()}, "base_ff": base,
               "motion_u": {"mean": float(u.mean()), "frac_static": float((u == 0).mean()), "frac_fast": float((u == 1).mean())}, "rows": {}}
        cfgs = {"a0.7(T-302)": lambda t: 0.7, "a0.55": lambda t: 0.55, "a0.4": lambda t: 0.4}
        for a_s, a_f in ((0.55, 0.85), (0.55, 1.0), (0.4, 0.85), (0.4, 1.0), (0.55, 0.7), (0.6, 0.85)):
            cfgs[f"adapt_{a_s}_{a_f}"] = adaptive_alpha_fn(cd, a_s, a_f)
        for name, fn in cfgs.items():
            t0 = time.perf_counter()
            g = run_plain(cd, fn, boil)
            res["rows"][name] = metrics(g, raw, wins, base) | {"s_per_frame": (time.perf_counter() - t0) / cd.T}
            r = res["rows"][name]
            print(clipname, name, {k: round(r[k], 4) for k in ("ff_static_vs_a1", "ff_fast_vs_a1", "group_iou_mean", "group_iou_min",
                                                               "centroid_err_max", "limb_ratio_min_all")}, flush=True)
        allres[clipname] = res
        write_json(args.out / f"adaptive_{clipname}.json", res)


def cmd_grid(args) -> None:
    boil = CFG.stabilize.temporal.boil_preserve
    for clipname in args.clips:
        cd = ClipData(clipname)
        raw, wins = cd.raw_maps(), cd.windows()
        base = base_flipflop(cd, wins, boil)
        if args.preset == "fin":
            configs = [{"name": f"a{a}_rf{rf}_b{b}", "alpha": a, "rflow": rf, "blend": b, "rule": "sun"}
                       for a, rf, b in ((0.55, 2, 0.4), (0.4, 3, 0.4), (0.55, 2, 1.0))]
        else:
            configs = [{"name": f"a{a}_rf{rf}_b{b}", "alpha": a, "rflow": rf, "blend": b, "rule": "sun"}
                       for a, rfs in ((0.7, (1, 2)), (0.55, (1, 2, 3)), (0.4, (1, 2, 3))) for rf in rfs for b in (0.4, 1.0)]
        t0 = time.perf_counter()
        out, timing = run_flow_sweep(cd, args.method, configs, boil)
        res = {"clip": clipname, "method": args.method, "preset": args.preset, "base_ff": base, "timing": timing | {"wall_s": time.perf_counter() - t0},
               "rows": {c["name"]: metrics(out[c["name"]], raw, wins, base) | {"cfg": c} for c in configs}}
        for n, r in res["rows"].items():
            print(clipname, args.method, n, {k: round(r[k], 4) for k in ("ff_static_vs_a1", "ff_fast_vs_a1", "group_iou_mean", "group_iou_min",
                                                                         "centroid_err_max", "limb_ratio_min_all")}, flush=True)
        write_json(args.out / f"grid_{clipname}_{args.method.replace(':', '-')}_{args.preset}.json", res)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (("pop", cmd_pop), ("holes", cmd_holes), ("occl", cmd_occl), ("flowq", cmd_flowq), ("determinism", cmd_determinism),
                     ("fail", cmd_fail), ("adaptive", cmd_adaptive), ("grid", cmd_grid)):
        sp = sub.add_parser(name)
        sp.add_argument("--clips", nargs="+", default=list(DEFAULT_CLIPS))
        sp.add_argument("--out", type=Path, default=DEFAULT_OUT)
        sp.set_defaults(fn=fn)
        if name == "pop":
            sp.add_argument("--contours", nargs="*", default=[], help="label=<folder contours, {clip} diganti nama klip>")
        if name in ("flowq", "determinism"):
            sp.add_argument("--methods", nargs="+", default=["dis:medium", "dis:fast", "fb:base", "fb:big"])
        if name == "grid":
            sp.add_argument("--method", default="dis:medium")
            sp.add_argument("--preset", choices=("fin", "grid"), default="fin")
    args = p.parse_args(argv)
    args.fn(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
