"""T-305c Tahap 1a (CPU, baca-saja terhadap klip): papan label BUTA + fitur buta untuk strok oklusi `test`, Klip2, klip3.

    python scripts/t305c_study.py hash before|after   # sha256 frames/seg/depth/stable/contours/strokes 4 klip + out/ -> work/t305c/
    python scripts/t305c_study.py features            # work/t305c_scratch/features.json (SEMUA strok oklusi) + ringkasan
    python scripts/t305c_study.py board               # work/t305c/papan/*.png + PANDUAN.md; peta nomor -> sel hanya di scratch

`eval` (Tahap 1b: ukur fitur terhadap label Rio) menyusul. Hanya membaca work/clips/*; keluaran ke work/t305c/ dan
work/t305c_scratch/ (di-ignore git). Deterministik: SEED tetap (sub-seed per keperluan di bawah).
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rotoscope import vectorize as vec  # noqa: E402
from rotoscope.config import load_class_names  # noqa: E402

CLIPS_DIR = ROOT / "work" / "clips"
OUT = ROOT / "work" / "t305c"
SCRATCH = ROOT / "work" / "t305c_scratch"
BOARD_DIR = OUT / "papan"
HASH_CLIPS = ("test", "test_short", "Klip2", "klip3")
HASH_SUBDIRS = ("frames", "seg", "depth", "stable", "contours", "strokes")
STUDY_CLIPS = ("test", "Klip2", "klip3")

SEED = 305003                       # seed tetap: sampel papan, urutan nomor (sub-seed = SEED + salt)
SALT_SAMPLE = 1
ORDER_SEED = 311001                 # urutan nomor papan v2 (v1: SEED + 2 = 305005)
LEG_FRAMES = (73, 78, 82, 87, 92)   # Done-when kaki T-201b

# fitur B
KS = (2, 4, 6)                      # px kerja
EPS = 1e-6                          # R_k = J / (J + |C| + EPS); satuan depth_smooth (log - median), jadi EPS << J tipikal
TANGENT_HALF = 3                    # tangen = p[i+3] - p[i-3] (polyline skeleton berundak 8 arah; tangen 1 langkah terlalu kasar)
GRAD_MIN = 1e-9                     # G_k: titik dengan |grad| <= ini dilewati (pembagi nol)
CLAMP_DEPTH = -10.0                # depth_smooth < ini = piksel disparity <= 0 yang di-clamp (log_eps 1e-6 -> sekitar -13,8)

# sampel papan
TEST_WINDOW = (73, 92)
TEST_MAX, KLIP2_MAX, KLIP3_PER_CELL = 16, 8, 9
CUT_FRAMES = (49, 85, 126, 134, 146, 154, 166, 174)
CUT_MARGIN = 2
AGE_LONG = 6
STRENGTH_LOW_PCT, STRENGTH_HIGH_PCT = 50, 75

# papan
PAGE_COLS, PAGE_ROWS = 3, 4
SLOT = 300                          # sisi slot panel (px)
GAP = 10
HEADER = 64
CROP_SCALE = 2.0                    # bbox strok x 2
CROP_MIN = 160                      # px kerja
MAGENTA = (255, 0, 255)             # BGR
GRAY = (150, 150, 150)
SHIFT = 4                           # polylines subpiksel


# ── util ───────────────────────────────────────────
def jload(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def jsave(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")


def q3(a) -> list | None:
    a = np.asarray([x for x in a if x is not None and np.isfinite(x)], float)
    return None if not len(a) else [round(float(np.percentile(a, p)), 5) for p in (5, 50, 95)]


# ── hash ───────────────────────────────────────────
def hash_files() -> list[Path]:
    files: list[Path] = []
    for c in HASH_CLIPS:
        for d in HASH_SUBDIRS:
            files += [p for p in sorted((CLIPS_DIR / c / d).rglob("*")) if p.is_file()]
    files += [p for p in sorted((ROOT / "out").rglob("*")) if p.is_file()]
    return files


def file_sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cmd_hash(which: str) -> None:
    lines = [f"{file_sha(p)}  {p.relative_to(ROOT).as_posix()}" for p in hash_files()]
    path = OUT / f"{which}_sha256.txt"
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{path.relative_to(ROOT).as_posix()}: {len(lines)} berkas")
    if which == "after":
        before = (OUT / "before_sha256.txt").read_text(encoding="utf-8").splitlines()
        same = before == lines
        diff = len(set(before) ^ set(lines))
        print(f"identik dengan before: {same} (before {len(before)} baris, after {len(lines)}, selisih himpunan {diff})")


# ── data klip ──────────────────────────────────────
class ClipData:
    def __init__(self, name: str) -> None:
        self.name = name
        self.dir = CLIPS_DIR / name
        self.docs = [jload(p) for p in sorted((self.dir / "contours").glob("frame_*.json"))]
        self.stats = jload(self.dir / "contours" / "clip_stats.json")
        man = jload(self.dir / "contours" / "manifest.json")["vectorize"]
        self.L = float(man["depth_lines.min_len_px"])
        self.sigma = float(man["depth_lines.blur_sigma"])
        self.names = [g for g, _ in jload(self.dir / "stable" / "manifest.json")["groups"]]
        self.t_high, self.t_low = self.stats["t_high"], self.stats["t_low"]
        self.width, self.height = self.docs[0]["width"], self.docs[0]["height"]

    def frame_name(self, i: int) -> str:
        return f"frame_{i:05d}"

    def gmap(self, i: int) -> np.ndarray:
        g = cv2.imread(str(self.dir / "stable" / "groups" / f"{self.frame_name(i)}.png"), cv2.IMREAD_UNCHANGED)
        return g

    def depth(self, i: int) -> np.ndarray:
        d = vec.read_depth_smooth(self.dir / "stable" / "depth_smooth" / f"{self.frame_name(i)}.npy", self.height, self.width)
        if d is None:
            raise SystemExit(f"depth_smooth rusak: {self.name} {i}")
        return d

    def frame_image(self, i: int) -> np.ndarray:
        return cv2.imread(str(self.dir / "frames" / f"{self.frame_name(i)}.png"), cv2.IMREAD_COLOR)


def track_runs(rows: list[dict]) -> None:
    """Isi `run_start` / `run_len` / `age` (umur track = panjang run frame berurutan sebuah track_id; hanya untuk stratifikasi,
    BUKAN fitur kandidat: memakai frame setelah t)."""
    by: dict[int, list[int]] = {}
    for r in rows:
        by.setdefault(r["track_id"], []).append(r["frame"])
    run_of: dict[tuple[int, int], tuple[int, int]] = {}
    for t, fr in by.items():
        fr = sorted(set(fr))
        s = fr[0]
        for a, b in zip(fr, fr[1:] + [None]):
            if b is None or b != a + 1:
                for x in range(s, a + 1):
                    run_of[(t, x)] = (s, a - s + 1)
                s = b
    for r in rows:
        s, n = run_of[(r["track_id"], r["frame"])]
        r["run_start"], r["run_len"], r["age"] = s, n, n


# ── fitur ──────────────────────────────────────────
def tangents(p: np.ndarray) -> np.ndarray:
    n = len(p)
    t = np.zeros_like(p)
    for i in range(n):
        a, b = max(i - TANGENT_HALF, 0), min(i + TANGENT_HALF, n - 1)
        v = p[b] - p[a]
        if not np.hypot(*v):                       # loop tertutup: ujung = awal
            a, b = max(i - 1, 0), min(i + 1, n - 1)
            v = p[b] - p[a]
        t[i] = v / (np.hypot(*v) or 1.0)
    return t


def sample_depth(depth: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Bilinear di koordinat kontinu (pusat piksel = k + 0,5); di luar frame -> nilai tepi (divalidasi terpisah)."""
    m = (xy - 0.5).astype(np.float32)
    return cv2.remap(depth, m[:, 0].reshape(1, -1), m[:, 1].reshape(1, -1), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)[0].astype(np.float64)


def fg_group(gmap: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Id grup di piksel yang memuat titik (nearest); -1 di luar frame."""
    h, w = gmap.shape
    ix, iy = np.floor(xy[:, 0]).astype(int), np.floor(xy[:, 1]).astype(int)
    ok = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
    out = np.full(len(xy), -1, int)
    out[ok] = gmap[iy[ok], ix[ok]]
    return out


def stroke_row(clip: ClipData, i: int, s: dict, gmap: np.ndarray, depth: np.ndarray, mag: np.ndarray, gid: int) -> dict:
    p = np.asarray(s["points"], float)
    loop = len(p) > 2 and np.allclose(p[0], p[-1])
    body = p[:-1] if loop else p
    seg = np.hypot(*np.diff(p, axis=0).T) if len(p) > 1 else np.zeros(0)
    path = float(seg.sum())
    ix, iy = np.floor(body[:, 0]).astype(int), np.floor(body[:, 1]).astype(int)
    strength = float(np.mean(mag[iy, ix], dtype=np.float64))
    row = {"clip": clip.name, "frame": i, "track_id": s.get("track_id"), "group": s["groups"][0], "n_points": len(p),
           "path_px": round(path, 3), "points_over_L": round(len(p) / clip.L, 4), "path_over_L": round(path / clip.L, 4),
           "strength_json": s["strength"], "strength": round(strength / clip.t_high, 5), "loop": bool(loop)}
    t = tangents(p)
    nrm = np.stack([-t[:, 1], t[:, 0]], axis=1)
    d0 = sample_depth(depth, p)
    for k in KS:
        a, b = p + k * nrm, p - k * nrm
        ga, gb = fg_group(gmap, a), fg_group(gmap, b)
        g0 = fg_group(gmap, p)
        valid = (ga > 0) & (gb > 0) & (g0 > 0)
        da, db = sample_depth(depth, a), sample_depth(depth, b)
        J = np.abs(da - db)
        C = d0 - (da + db) / 2.0
        nv = int(valid.sum())
        side_valid = np.concatenate([ga[valid], gb[valid]]) if nv else np.zeros(0, int)
        row[f"n_valid_{k}"] = nv
        row[f"ignored_share_{k}"] = round(float(1.0 - np.mean(np.concatenate([ga > 0, gb > 0]))), 5)
        row[f"other_group_share_{k}"] = round(float(np.mean(side_valid != gid)), 5) if nv else None
        row[f"clamp_share_{k}"] = round(float(np.mean((da[valid] < CLAMP_DEPTH) | (db[valid] < CLAMP_DEPTH))), 5) if nv else None
        if nv:
            Jm, Cm = float(np.median(J[valid])), float(np.median(np.abs(C[valid])))
            row[f"J_{k}"], row[f"absC_{k}"] = round(Jm, 6), round(Cm, 6)
            row[f"signedC_{k}"] = round(float(np.median(C[valid])), 6)
            row[f"R_{k}"] = round(Jm / (Jm + Cm + EPS), 5)
            gm = mag[np.floor(p[:, 1]).astype(int), np.floor(p[:, 0]).astype(int)].astype(np.float64)
            use = valid & (gm > GRAD_MIN)
            row[f"G_{k}"] = round(float(np.median(J[use] / (2 * k * gm[use]))), 5) if use.any() else None
        else:
            row[f"J_{k}"] = row[f"absC_{k}"] = row[f"signedC_{k}"] = row[f"R_{k}"] = row[f"G_{k}"] = None
    row["S"] = round(row["J_6"] / row["J_2"], 5) if row["J_2"] and row["J_6"] is not None else None
    return row


def compute_features(name: str) -> tuple[list[dict], dict]:
    clip = ClipData(name)
    rows: list[dict] = []
    times: list[float] = []
    mismatch = 0
    for i, doc in enumerate(clip.docs):
        occ = [s for s in doc["strokes"] if s["type"] == "occlusion"]
        t0 = time.perf_counter()
        gmap, depth = clip.gmap(i), clip.depth(i)
        mag = vec.depth_gradient(depth, clip.sigma)[0]
        for s in occ:
            gid = clip.names.index(s["groups"][0]) + 1
            r = stroke_row(clip, i, s, gmap, depth, mag, gid)
            if abs(r["strength"] * clip.t_high - s["strength"]) > 0.5 * 10 ** -vec.STRENGTH_DECIMALS + 1e-9:
                mismatch += 1
            rows.append(r)
        times.append(time.perf_counter() - t0)
    track_runs(rows)
    info = {"clip": name, "n_frames": len(clip.docs), "n_strokes": len(rows), "t_high": clip.t_high, "t_low": clip.t_low, "L": clip.L,
            "frame_size": [clip.width, clip.height], "depth_smooth_shape_hw": [clip.height, clip.width],
            "strength_recompute_mismatch_vs_json": mismatch,
            "time_per_frame_s": {"mean": round(float(np.mean(times)), 4), "p95": round(float(np.percentile(times, 95)), 4)}}
    return rows, info


FEATURES = ("n_points", "path_px", "points_over_L", "path_over_L", "strength", "J_2", "J_4", "J_6", "absC_2", "absC_4", "absC_6",
            "R_2", "R_4", "R_6", "S", "G_2", "G_4", "G_6", "ignored_share_2", "ignored_share_4", "ignored_share_6", "other_group_share_2",
            "other_group_share_4", "other_group_share_6", "clamp_share_2", "clamp_share_4", "clamp_share_6")


def cmd_features() -> None:
    all_rows: list[dict] = []
    infos: dict[str, dict] = {}
    for name in STUDY_CLIPS:
        rows, info = compute_features(name)
        all_rows += rows
        info["distribution_p5_p50_p95"] = {f: q3([r[f] for r in rows]) for f in FEATURES}
        info["age_p5_p50_p95"] = q3([r["age"] for r in rows])
        info["invalid_strokes_k"] = {str(k): sum(r[f"n_valid_{k}"] == 0 for r in rows) for k in KS}
        infos[name] = info
        print(f"{name}: {len(rows)} strok, waktu/frame mean {info['time_per_frame_s']['mean']} s p95 {info['time_per_frame_s']['p95']} s, "
              f"selisih strength vs json {info['strength_recompute_mismatch_vs_json']}")
    meta = {"seed": SEED, "ks": list(KS), "eps": EPS, "tangent_half_window_points": TANGENT_HALF, "clamp_depth": CLAMP_DEPTH,
            "definitions": {"J_k": "median |d(p+kn) - d(p-kn)|", "absC_k": "median |d(p) - (d(p+kn)+d(p-kn))/2|",
                            "R_k": "J_k / (J_k + absC_k + eps) dari median stroke",
                            "S": "J_6 / J_2 (saturasi lompatan; median stroke)",
                            "G_k": "median atas titik valid dari J_k(p) / (2k * |grad d(p)|); |grad| = Gaussian sigma [4] + Sobel (mag [4]) di piksel titik", "strength": "mean |grad| (sigma [4]) di titik strok / T_high klip",
                            "age": "panjang run frame berurutan track_id (HANYA stratifikasi; memakai t+1)"}}
    jsave(SCRATCH / "features.json", {"meta": meta, "clips": infos, "rows": all_rows})
    print(f"work/t305c_scratch/features.json: {len(all_rows)} strok")


# ── papan ──────────────────────────────────────────
def load_rows() -> list[dict]:
    return jload(SCRATCH / "features.json")["rows"]


def near_cut(frame: int) -> bool:
    return any(abs(frame - c) <= CUT_MARGIN for c in CUT_FRAMES)


def tracks_of(rows: list[dict]) -> list[dict]:
    """Satu unit = (clip, track_id, run_start); wakil = strok di frame tengah run (indeks len // 2)."""
    by: dict[tuple, list[dict]] = {}
    for r in rows:
        by.setdefault((r["clip"], r["track_id"], r["run_start"]), []).append(r)
    out = []
    for key, rs in by.items():
        rs.sort(key=lambda r: r["frame"])
        rep = dict(rs[len(rs) // 2])
        rep["run_frames"] = [rs[0]["frame"], rs[-1]["frame"]]
        rep["in_test_window"] = any(TEST_WINDOW[0] <= r["frame"] <= TEST_WINDOW[1] for r in rs)
        out.append(rep)
    out.sort(key=lambda r: (r["clip"], r["run_start"], r["track_id"]))
    return out


def sample_tracks(rows: list[dict]) -> tuple[list[dict], dict]:
    rng = np.random.default_rng(SEED + SALT_SAMPLE)
    units = tracks_of(rows)
    report: dict = {}
    chosen: list[dict] = []

    def pick(pool: list[dict], n: int) -> list[dict]:
        idx = rng.permutation(len(pool))[:n]
        return [pool[j] for j in sorted(idx)]

    t_units = [u for u in units if u["clip"] == "test"]
    inwin = [u for u in t_units if u["in_test_window"]]
    take = pick(inwin, TEST_MAX)
    extra = []
    if len(take) < TEST_MAX:
        rest = [u for u in t_units if not u["in_test_window"]]
        extra = pick(rest, TEST_MAX - len(take))
    for u in take:
        u["cell"] = "test:jendela"
    for u in extra:
        u["cell"] = "test:luar-jendela"
    chosen += take + extra
    report["test"] = {"tracks_total": len(t_units), "tracks_in_window": len(inwin), "dipilih_jendela": len(take), "dipilih_luar": len(extra)}

    k_units = [u for u in units if u["clip"] == "Klip2"]
    take = pick(k_units, KLIP2_MAX)
    for u in take:
        u["cell"] = "Klip2"
    chosen += take
    report["Klip2"] = {"tracks_total": len(k_units), "dipilih": len(take)}

    k3_rows = [r for r in rows if r["clip"] == "klip3"]
    st = np.array([r["strength"] for r in k3_rows])
    lo, hi = float(np.percentile(st, STRENGTH_LOW_PCT)), float(np.percentile(st, STRENGTH_HIGH_PCT))
    k3 = [u for u in units if u["clip"] == "klip3"]
    cells = {"umur1_lemah": lambda u: u["run_len"] == 1 and u["strength"] <= lo,
             "umur1_kuat": lambda u: u["run_len"] == 1 and u["strength"] >= hi,
             "umurPanjang_lemah": lambda u: u["run_len"] >= AGE_LONG and u["strength"] <= lo,
             "umurPanjang_kuat": lambda u: u["run_len"] >= AGE_LONG and u["strength"] >= hi}
    rep3: dict = {"strength_median_over_T_high": round(lo, 4), "strength_p75_over_T_high": round(hi, 4), "cells": {}}
    for cname, fn in cells.items():
        pool = [u for u in k3 if fn(u)]
        far = [u for u in pool if not near_cut(u["frame"])]
        near = [u for u in pool if near_cut(u["frame"])]
        take = pick(far, KLIP3_PER_CELL)
        fill = pick(near, KLIP3_PER_CELL - len(take)) if len(take) < KLIP3_PER_CELL else []
        for u in take + fill:
            u["cell"] = f"klip3:{cname}"
        chosen += take + fill
        rep3["cells"][cname] = {"kandidat": len(pool), "jauh_dari_cut": len(far), "dekat_cut": len(near),
                                "dipilih": len(take) + len(fill), "dipilih_dekat_cut": len(fill)}
    report["klip3"] = rep3
    # Follow-up: `test` = SEMUA track yang punya strok di f73–92. Pemilihan acak di atas tetap dijalankan agar urutan rng (dan
    # karena itu sampel Klip2 / klip3) identik dengan papan v1; hasilnya dicek terhadap board_map_v1.json.
    chosen = [u for u in chosen if u["clip"] != "test"]
    for u in inwin:
        u["cell"] = "test:jendela"
    chosen += inwin
    report["test"] = {"tracks_total": len(t_units), "tracks_in_window": len(inwin), "dipilih_jendela": len(inwin), "dipilih_luar": 0}
    v1_path = SCRATCH / "board_map_v1.json"
    if v1_path.exists():
        v1 = {(m["clip"], m["track_id"], m["run_start"]) for m in jload(v1_path)["mapping"].values() if m["clip"] != "test"}
        now = {(u["clip"], u["track_id"], u["run_start"]) for u in chosen if u["clip"] != "test"}
        report["klip2_klip3_sama_dengan_v1"] = v1 == now
        if v1 != now:
            raise SystemExit("sampel Klip2 / klip3 berbeda dari papan v1")
    return chosen, report


def done_when_tracks(clips: dict[str, ClipData]) -> dict[tuple, list[int]]:
    """Definisi Done-when kaki T-201b (tests/test_vectorize_occlusion.py): pada frame 73 / 78 / 82 / 87 / 92 strok oklusi dengan
    >= 80% titik di piksel Lower_Clothing (seg/classmap). Return {(track_id, frame): ...} -> {track_id: [frame yang dipenuhi]}."""
    c = clips["test"]
    lc = list(load_class_names()).index("Lower_Clothing")
    out: dict[int, list[int]] = {}
    for i in LEG_FRAMES:
        cm = cv2.imread(str(c.dir / "seg" / "classmap" / f"frame_{i:05d}.png"), cv2.IMREAD_UNCHANGED)
        for s in c.docs[i]["strokes"]:
            if s["type"] != "occlusion":
                continue
            pts = [(int(y), int(x)) for x, y in s["points"]]
            if sum(cm[y, x] == lc for y, x in pts) / len(pts) >= 0.8:
                out.setdefault(s["track_id"], []).append(i)
    return out


def clip_boxes(clips: dict[str, ClipData], u: dict) -> tuple[int, int, int, int]:
    """Kotak crop (x0, y0, x1, y1) bilangan bulat: bbox strok x CROP_SCALE, min CROP_MIN, digeser masuk frame."""
    c = clips[u["clip"]]
    p = np.asarray(c.docs[u["frame"]]["strokes"][u["_stroke_idx"]]["points"], float)
    x0, y0, x1, y1 = p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w, h = min(max((x1 - x0) * CROP_SCALE, CROP_MIN), c.width), min(max((y1 - y0) * CROP_SCALE, CROP_MIN), c.height)
    ax = int(np.clip(round(cx - w / 2), 0, c.width - int(round(w))))
    ay = int(np.clip(round(cy - h / 2), 0, c.height - int(round(h))))
    return ax, ay, ax + int(round(w)), ay + int(round(h))


def fit(img: np.ndarray, size: int) -> tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    s = min(size / h, size / w)
    out = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    return out, s


def draw_poly(img: np.ndarray, p: np.ndarray, org: tuple[int, int], s: float, color, thick: int) -> None:
    """p = koordinat kontinu px kerja; org = sudut kiri-atas crop (px). Pemetaan: indeks piksel hasil = (x - org) * s - 0,5."""
    q = ((p - np.asarray(org, float)) * s - 0.5) * (1 << SHIFT)
    cv2.polylines(img, [np.round(q).astype(np.int32).reshape(-1, 1, 2)], False, color, thick, cv2.LINE_AA, SHIFT)


def slot(img: np.ndarray) -> np.ndarray:
    canvas = np.full((SLOT, SLOT, 3), 255, np.uint8)
    h, w = img.shape[:2]
    canvas[(SLOT - h) // 2:(SLOT - h) // 2 + h, (SLOT - w) // 2:(SLOT - w) // 2 + w] = img
    return canvas


def make_cell(clips: dict[str, ClipData], u: dict, number: int) -> tuple[np.ndarray, dict]:
    c = clips[u["clip"]]
    i = u["frame"]
    img = c.frame_image(i)
    box = clip_boxes(clips, u)
    x0, y0, x1, y1 = box
    strokes = c.docs[i]["strokes"]
    # (i) thumbnail utuh + kotak lokasi
    th, ts = fit(img, SLOT)
    cv2.rectangle(th, (round(x0 * ts), round(y0 * ts)), (round(x1 * ts), round(y1 * ts)), (0, 0, 255), 2, cv2.LINE_AA)
    # (ii) crop tanpa garis; (iii) crop + garis (digambar SETELAH penskalaan)
    crop, s = fit(img[y0:y1, x0:x1], SLOT)
    plain = crop.copy()
    lined = crop.copy()
    for j, st in enumerate(strokes):
        if j == u["_stroke_idx"]:
            continue
        draw_poly(lined, np.asarray(st["points"], float), (x0, y0), s, GRAY, 1)
    draw_poly(lined, np.asarray(strokes[u["_stroke_idx"]]["points"], float), (x0, y0), s, MAGENTA, 3)
    check = {"number": number, "magenta_px_plain": int((np.all(plain == MAGENTA, axis=2)).sum()),
             "magenta_px_lined": int((np.all(lined == MAGENTA, axis=2)).sum()), "scale": round(s, 4), "crop_xyxy": list(box)}
    cell = np.full((HEADER + SLOT + GAP, 3 * SLOT + 4 * GAP, 3), 255, np.uint8)
    cv2.putText(cell, f"{number:02d}", (GAP, HEADER - 12), cv2.FONT_HERSHEY_DUPLEX, 2.2, (0, 0, 0), 4, cv2.LINE_AA)
    for k, panel in enumerate((th, plain, lined)):
        sl = slot(panel)
        cv2.rectangle(sl, (0, 0), (SLOT - 1, SLOT - 1), (200, 200, 200), 1)
        cell[HEADER:HEADER + SLOT, GAP + k * (SLOT + GAP):GAP + k * (SLOT + GAP) + SLOT] = sl
    cv2.rectangle(cell, (0, 0), (cell.shape[1] - 1, cell.shape[0] - 1), (120, 120, 120), 2)
    return cell, check


def stroke_index(clips: dict[str, ClipData], u: dict) -> int:
    c = clips[u["clip"]]
    for j, st in enumerate(c.docs[u["frame"]]["strokes"]):
        if st["type"] == "occlusion" and st["track_id"] == u["track_id"]:
            return j
    raise SystemExit(f"strok wakil tidak ditemukan: {u['clip']} f{u['frame']} t{u['track_id']}")


def verify_mapping(clips: dict[str, ClipData]) -> dict:
    """Skala px kerja -> piksel frames/ dan cek geometri: titik silhouette harus berada pada batas foreground peta grup stabil."""
    res = {}
    for name, c in clips.items():
        img = c.frame_image(0)
        fh, fw = img.shape[:2]
        mid = len(c.docs) // 2
        g = c.gmap(mid)
        fgm = (g > 0).astype(np.uint8)
        edge = (cv2.dilate(fgm, np.ones((3, 3), np.uint8)) != cv2.erode(fgm, np.ones((3, 3), np.uint8)))
        near = cv2.dilate(edge.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        pts = np.concatenate([np.asarray(s["points"], float) for s in c.docs[mid]["strokes"] if s["type"] == "silhouette"])
        ix, iy = np.clip(np.floor(pts[:, 0]).astype(int), 0, fw - 1), np.clip(np.floor(pts[:, 1]).astype(int), 0, fh - 1)
        fp = cv2.copyMakeBorder(fgm, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)            # seperti [4]: luar frame = background
        edge_p = (cv2.dilate(fp, np.ones((3, 3), np.uint8)) != cv2.erode(fp, np.ones((3, 3), np.uint8)))[1:-1, 1:-1]
        near_p = cv2.dilate(edge_p.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        off = ~near[iy, ix]
        border = (pts[:, 0] < 2) | (pts[:, 1] < 2) | (pts[:, 0] > fw - 2) | (pts[:, 1] > fh - 2)
        res[name] = {"silhouette_near_edge_share_padded_like_stage4": round(float(near_p[iy, ix].mean()), 4),
                     "off_edge_points": int(off.sum()), "off_edge_within_2px_of_frame_border": int((off & border).sum()),
                     "frames_wh": [fw, fh], "contours_wh": [c.width, c.height], "scale_x": fw / c.width, "scale_y": fh / c.height,
                     "groups_shape_hw": list(g.shape), "silhouette_points_on_fg_edge_frame": mid,
                     "silhouette_points_near_fg_edge_share": round(float(near[iy, ix].mean()), 4)}
    return res


PANDUAN = """# Papan label — panduan

Tiap kotak bernomor punya tiga gambar, kiri ke kanan:
1. frame utuh, kotak merah = lokasi;
2. potongan frame yang sama TANPA garis;
3. potongan yang sama dengan SATU garis magenta (garis yang dinilai) dan garis lain abu-abu tipis (hanya konteks).

Nilai hanya garis MAGENTA. Pilih satu:

- **O — oklusi asli**: garis menelusuri tepi anggota tubuh / benda yang berada DI DEPAN bagian lain
  (termasuk lengan berbaju di depan badan, dan kaki menyilang di balik celana).
- **L — lipatan / detail tak diinginkan**: lipatan atau kerutan kain, dan garis detail lain yang tidak diinginkan
  (motif kain, wajah).
- **?** — ragu / tidak bisa dipastikan dari gambar.

Format balasan di chat (setiap nomor muncul TEPAT satu kali):

```
O: 01, 04, ...
L: 02, ...
?: ...
```

Catatan: jangan menilai bagus-jeleknya garis atau panjangnya; hanya jenisnya (O / L / ?).
"""


def cmd_board() -> None:
    rows = load_rows()
    chosen, report = sample_tracks(rows)
    clips = {n: ClipData(n) for n in STUDY_CLIPS}
    for u in chosen:
        u["_stroke_idx"] = stroke_index(clips, u)
    dw = done_when_tracks(clips)
    for u in chosen:
        if u["clip"] == "test":
            u["donewhen_frames"] = dw.get(u["track_id"], []) if u["in_test_window"] else []
    report["donewhen_tracks_in_test"] = len(dw)
    report["donewhen_tracks_in_board"] = sum(1 for u in chosen if u.get("donewhen_frames"))
    order = np.random.default_rng(ORDER_SEED).permutation(len(chosen))
    numbered = {int(n) + 1: chosen[j] for n, j in enumerate(order)}          # nomor 1.. -> unit (acak lintas klip)
    per_page = PAGE_COLS * PAGE_ROWS
    n_pages = -(-len(chosen) // per_page)
    BOARD_DIR.mkdir(parents=True, exist_ok=True)
    for old in BOARD_DIR.glob("*"):
        old.unlink()
    checks, page_of = [], {}
    cell_h, cell_w = HEADER + SLOT + GAP, 3 * SLOT + 4 * GAP
    for pg in range(n_pages):
        page = np.full((PAGE_ROWS * (cell_h + GAP) + GAP, PAGE_COLS * (cell_w + GAP) + GAP, 3), 255, np.uint8)
        for pos in range(per_page):
            n = pg * per_page + pos + 1
            if n not in numbered:
                break
            cell, chk = make_cell(clips, numbered[n], n)
            r, cc = divmod(pos, PAGE_COLS)
            y, x = GAP + r * (cell_h + GAP), GAP + cc * (cell_w + GAP)
            page[y:y + cell_h, x:x + cell_w] = cell
            checks.append(chk)
            page_of[n] = pg + 1
        cv2.imwrite(str(BOARD_DIR / f"papan_{pg + 1:02d}.png"), page)
    (BOARD_DIR / "PANDUAN.md").write_text(PANDUAN, encoding="utf-8")
    mapping = {f"{n:02d}": {k: v for k, v in u.items() if not k.startswith("_")} | {"page": page_of[n]} for n, u in numbered.items()}
    jsave(SCRATCH / "board_map.json", {"seed": SEED, "salt_sample": SALT_SAMPLE, "order_seed": ORDER_SEED, "sampling": report,
                                       "mapping": mapping, "checks": checks, "coordinate_mapping": verify_mapping(clips)})
    bad_plain = [c["number"] for c in checks if c["magenta_px_plain"]]
    bad_lined = [c["number"] for c in checks if not c["magenta_px_lined"]]
    print(f"{len(chosen)} sel, {n_pages} halaman -> work/t305c/papan/; magenta di panel tanpa garis: {bad_plain}; panel bergaris tanpa magenta: {bad_lined}")
    print(json.dumps(report, ensure_ascii=False))
    print(json.dumps(jload(SCRATCH / "board_map.json")["coordinate_mapping"], ensure_ascii=False))


if __name__ == "__main__":
    def _ev(name: str):                                 # Tahap 1b: scripts/t305c_eval.py (dimuat malas)
        import t305c_eval as ev
        return lambda: getattr(ev, name)(*([sys.argv[2:]] if name in ("cmd_sim", "cmd_simall", "cmd_timing", "cmd_simtable", "cmd_explore") else []))

    cmds = {"hash": lambda: cmd_hash(sys.argv[2]), "features": cmd_features, "board": cmd_board,
            "labels": _ev("cmd_labels"), "eval": _ev("cmd_eval"), "sim": _ev("cmd_sim"), "simall": _ev("cmd_simall"),
            "timing": _ev("cmd_timing"), "simtable": _ev("cmd_simtable"), "explore": _ev("cmd_explore")}
    if len(sys.argv) < 2 or sys.argv[1] not in cmds:
        raise SystemExit(__doc__)
    cmds[sys.argv[1]]()
