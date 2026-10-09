"""Alat sekali pakai T-201a / T-201b (seperti look_test.py; BUKAN bagian src/): overlay kontur [4] di atas frame asli.

    python scripts/contour_overlay.py --work-dir work/clips/test [--out-dir work/t201b]
        [--ranges 73-92,183-202,225-240,90] [--hole-frames 20] [--scale 2] [--samples 80,190,233] [--suffix _fix1]
        [--grad-panel] [--worst]

Membaca <work-dir>/contours/frame_*.json + frames/ (TIDAK menulis apa pun di folder klip). Menulis ke --out-dir
(default work/t201b, ter-ignore):
  overlay_<klip>.mp4   frame di --ranges + sampel frame ber-lubang + 3 frame tanpa oklusi, skala --scale (default 2×),
                       H.264 lewat export.encode; label "temporal belum aktif - getaran antar frame BELUM representatif"
  sample_<klip>_fNNNNN.png   frame --samples (+ satu frame ber-lubang)
  summary_<klip>.json  angka stage [4] dari contours/frames.jsonl + JSON frame (waktu, ukuran, strok per tipe, oklusi:
                       linimasa piksel hysteresis / D / L, run frame kosong, kriteria wilayah Lower_Clothing frame 73-92)

--grad-panel: panel pendamping di kanan = |grad| depth_smooth dipotong di T_high (gelap = kuat), area yang memenuhi
syarat D (jarak >= D dari batas grup / siluet / tepi frame) diarsir teal; garis oklusi digambar di atasnya. Untuk menilai
apakah garis yang hilang memang lemah gradiennya atau terbuang syarat D / L.
--worst: PNG kasus terburuk worst_<klip>_<jenis>_fNNNNN.png (+ panel): oklusi terdekat ke batas (kandidat bayangan),
strok oklusi terpendek yang lolos L, frame dengan strok oklusi terbanyak, frame 90 (komponen DA +-94 px di area tangan).

Warna: silhouette merah, silhouette_hole cyan, group_boundary hijau, occlusion oranye. Run titik yang menempel TEPI FRAME
(koordinat tepat di baris/kolom tepi) digambar magenta putus-putus supaya penilaian tertuju pada kontur sebenarnya.
Koordinat strok = pusat piksel (i + 0.5) → digambar di (x · skala − 0.5).

--track (T-202): mode terpisah → work/t202 (default): video overlay_track_<klip>.mp4 (warna per track_id, titik anchor
besar untuk silhouette / lubang / loop, lingkaran titik awal + panah arah untuk garis terbuka, label #id, label
"temporal belum aktif"), PNG kasus terburuk worst_<klip>_<jenis>_fNNNNN.png (lompatan anchor terbesar, pembalikan arah
bila ada, id baru terbanyak, padanan paling ambigu), dan arm_<klip>_fNNNNN.png untuk frame lengan terlepas (klip test:
213, 220, 228, 232-236, 249, 257-266) dengan panel berdampingan per --compare (default ambang 12 pendekatan X | ambang
16 pendekatan Y; keduanya dihitung ulang dari himpunan titik, tanpa menulis ke contours/).

--dropped: video TAMBAHAN overlay_<klip>_dropped.mp4 (frame yang sama; tanpa PNG / ringkasan, tidak menimpa berkas
lain) dengan kontur yang DIBUANG filter ukuran digambar abu-abu + label luas px: lubang < min_hole_area (abu-abu
terang) dan komponen luar < min_region_area (abu-abu gelap). Ambang dibaca dari contours/manifest.json; kontur
dihitung ulang dari stable/groups dengan fungsi yang sama dengan vectorize.silhouette_strokes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

from rotoscope import export as ex
from rotoscope import stabilize as stb
from rotoscope import vectorize as vec
from rotoscope.config import load_class_names
from rotoscope.stage_common import StageError, read_jsonl

SHIFT = 4                                    # bit subpiksel cv2.polylines
COLORS = {"silhouette": (0, 0, 255), "silhouette_hole": (255, 255, 0), "group_boundary": (0, 200, 0),
          "occlusion": (0, 140, 255)}   # BGR; oklusi oranye
EDGE_COLOR = (255, 0, 255)
DASH_ON, DASH_OFF = 3, 3                     # panjang dash (segmen) untuk run tepi
FADE = 0.45                                  # frame asli dipucatkan ke putih supaya garis menonjol
DROP_HOLE_COLOR, DROP_REGION_COLOR = (170, 170, 170), (80, 80, 80)    # abu-abu terang / gelap
LABEL = "temporal belum aktif - getaran antar frame BELUM representatif"
DEFAULT_RANGES = "73-92,183-202,225-240,90"
DEFAULT_SAMPLES = "80,190,233"
DEFAULT_OUT_DIR = "work/t201b"
N_NO_OCCLUSION_FRAMES = 3                    # frame tanpa oklusi yang ikut di video
LEG_FRAMES = (73, 78, 82, 87, 92)            # Done-when T-201b (kriteria wilayah Lower_Clothing)
LEG_CLASS = "Lower_Clothing"
LEG_FRACTION = 0.8                           # strok lolos bila >= 80% titiknya di piksel Lower_Clothing
REF_LEN_T102C = {73: 64, 78: 73, 82: 89, 87: 154, 92: 38}   # docs/05 log T-102c (|grad log d| mentah): orde besaran saja
DEBUG_FRAME = 90
HATCH_PERIOD, HATCH_WIDTH = 6, 2             # arsir diagonal area yang memenuhi syarat D (piksel frame)
HATCH_COLOR = (160, 140, 0)                  # BGR teal gelap


def parse_ranges(text: str) -> list[int]:
    out: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def on_edge(p, w: int, h: int) -> int:
    """Sisi bingkai (bitmask) tempat titik berada persis di baris/kolom tepi; 0 = bukan tepi."""
    x, y = p
    return (1 if x == 0.5 else 0) | (2 if x == w - 0.5 else 0) | (4 if y == 0.5 else 0) | (8 if y == h - 0.5 else 0)


def px(p, scale: int) -> tuple[int, int]:
    return round((p[0] * scale - 0.5) * (1 << SHIFT)), round((p[1] * scale - 0.5) * (1 << SHIFT))


def draw_strokes(img: np.ndarray, strokes: list[dict], w: int, h: int, scale: int) -> dict:
    """Gambar semua strok pada img (in-place). Return jumlah segmen biasa vs segmen tepi."""
    n_edge = n_norm = 0
    for s in strokes:
        pts = s["points"]
        color = COLORS[s["type"]]
        n = len(pts)
        closed = bool(s["closed"])
        pairs = [(i, i + 1) for i in range(n - 1)] + ([(n - 1, 0)] if closed and n > 1 else [])
        run = []                                     # titik berurutan non-tepi → satu polyline
        dash = 0
        for i, j in pairs:
            shared = on_edge(pts[i], w, h) & on_edge(pts[j], w, h)
            if shared:                               # segmen sepanjang tepi: putus-putus, magenta
                if run:
                    _poly(img, run, color, scale)
                    run = []
                if (dash // DASH_ON) % 2 == 0:
                    cv2.line(img, px(pts[i], scale), px(pts[j], scale), EDGE_COLOR, 2, cv2.LINE_AA, SHIFT)
                dash += 1
                n_edge += 1
            else:
                dash = 0
                if not run:
                    run = [pts[i]]
                run.append(pts[j])
                n_norm += 1
        if run:
            _poly(img, run, color, scale)
        if n == 1:
            cv2.circle(img, px(pts[0], scale), 2, color, -1, cv2.LINE_AA, SHIFT)
    return {"segments": n_norm, "edge_segments": n_edge}


def _poly(img: np.ndarray, pts: list, color: tuple, scale: int) -> None:
    arr = np.array([px(p, scale) for p in pts], np.int32)
    cv2.polylines(img, [arr.reshape(-1, 1, 2)], False, color, 2, cv2.LINE_AA, SHIFT)


def text_box(img: np.ndarray, lines: list[tuple[str, tuple]], x: int, y: int, scale: float = 0.55) -> None:
    font, th = cv2.FONT_HERSHEY_SIMPLEX, 1
    sizes = [cv2.getTextSize(t, font, scale, th)[0] for t, _ in lines]
    bw, lh = max(s[0] for s in sizes) + 12, max(s[1] for s in sizes) + 8
    cv2.rectangle(img, (x, y), (x + bw, y + lh * len(lines) + 6), (30, 30, 30), -1)
    for k, (t, c) in enumerate(lines):
        cv2.putText(img, t, (x + 6, y + lh * (k + 1)), font, scale, c, th, cv2.LINE_AA)


def dropped_contours(gmap: np.ndarray, min_region: int, min_hole: int) -> list[dict]:
    """Kontur luar < min_region dan lubang < min_hole (induk lolos): [{kind, area, points (pusat piksel)}].
    Luas = jumlah piksel, cara yang sama dengan vectorize.silhouette_strokes."""
    pad = vec.PAD_PX
    padded = cv2.copyMakeBorder((gmap != stb.BACKGROUND_ID).astype(np.uint8), pad, pad, pad, pad,
                                cv2.BORDER_CONSTANT, value=0)
    contours, hier = cv2.findContours(padded, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return []
    parent = [int(h[3]) for h in hier[0]]
    _, lab, st, _ = cv2.connectedComponentsWithStats(padded, connectivity=8)
    _, blab, bst, _ = cv2.connectedComponentsWithStats(1 - padded, connectivity=4)
    out, kept_outer = [], {}
    for i, cnt in enumerate(contours):
        if parent[i] < 0:
            x, y = cnt[0, 0]
            area = int(st[lab[y, x], cv2.CC_STAT_AREA])
            kept_outer[i] = area >= min_region
            if not kept_outer[i]:
                out.append({"kind": "region", "area": area, "points": cnt.reshape(-1, 2)})
    for i, cnt in enumerate(contours):
        if parent[i] >= 0 and kept_outer[parent[i]]:
            label = vec._hole_label(padded, blab, cnt)
            area = int(bst[label, cv2.CC_STAT_AREA]) if label >= 0 else 0
            if area < min_hole:
                out.append({"kind": "hole", "area": area, "points": cnt.reshape(-1, 2)})
    for d in out:
        d["points"] = [[x - pad + vec.PIXEL_CENTER, y - pad + vec.PIXEL_CENTER] for x, y in d["points"].tolist()]
    return out


def draw_dropped(img: np.ndarray, drops: list[dict], scale: int) -> None:
    for d in drops:
        color = DROP_HOLE_COLOR if d["kind"] == "hole" else DROP_REGION_COLOR
        arr = np.array([px(p, scale) for p in d["points"]], np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [arr], True, color, 2, cv2.LINE_AA, SHIFT)
        cx, cy = np.mean([[p[0], p[1]] for p in d["points"]], axis=0) * scale
        cv2.putText(img, f"{d['area']}px", (int(cx) - 14, int(cy) + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (20, 20, 20), 1, cv2.LINE_AA)


def panel_context(clip: Path) -> dict:
    """Parameter + ambang untuk panel |grad| (dari contours/manifest.json + clip_stats.json)."""
    params = read_json(clip / "contours" / "manifest.json")["vectorize"]
    stats = read_json(clip / "contours" / "clip_stats.json")
    return {"params": params, "t_high": stats["t_high"], "t_low": stats["t_low"]}


def grad_panel(clip: Path, index: int, scale: int, ctx: dict, strokes: list[dict]) -> np.ndarray:
    """Panel pendamping BGR: |grad| depth_smooth dipotong di T_high (gelap = kuat), area foreground yang memenuhi
    syarat D diarsir teal, garis oklusi (oranye) di atasnya."""
    gmap = stb.read_groups(clip / "stable" / "groups" / f"frame_{index:05d}.png")
    h, w = gmap.shape
    depth = vec.read_depth_smooth(clip / "stable" / "depth_smooth" / f"frame_{index:05d}.npy", h, w)
    if gmap is None or depth is None:
        raise StageError(f"input panel frame {index} tidak terbaca di {clip / 'stable'}")
    dp = vec.depth_params(ctx["params"])
    mag = vec.depth_gradient(depth, dp["blur_sigma"])[0]
    gray = (255 - np.clip(mag / ctx["t_high"], 0, 1) * 255).astype(np.uint8)
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    img[gmap == stb.BACKGROUND_ID] = (235, 235, 235)
    eligible = (gmap != stb.BACKGROUND_ID) & (vec.boundary_distance(gmap) >= dp["min_dist_px"])
    yy, xx = np.mgrid[0:h, 0:w]
    img[eligible & ((xx + yy) % HATCH_PERIOD < HATCH_WIDTH)] = HATCH_COLOR
    big = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
    draw_strokes(big, [s for s in strokes if s["type"] == "occlusion"], w, h, scale)
    text_box(big, [(f"|grad| depth_smooth, dipotong di T_high {ctx['t_high']:.4f} (gelap = kuat)", (255, 255, 255)),
                   (f"T_low {ctx['t_low']:.4f}; arsir teal = memenuhi syarat D >= {dp['min_dist_px']:g} px", HATCH_COLOR),
                   ("oranye = garis oklusi yang dipancarkan", COLORS["occlusion"])], 6, 6)
    return big


def render(clip: Path, index: int, scale: int, tag: str, dropped: dict | None = None,
           panel: dict | None = None) -> np.ndarray:
    """Frame asli (dipucatkan) × skala + strok + legenda → RGB uint8. `dropped` = {min_region_area, min_hole_area}
    → kontur yang dibuang filter ukuran ikut digambar (abu-abu + luas px). `panel` = panel_context → panel |grad|
    di kanan (lebar ×2)."""
    doc = read_json(clip / "contours" / f"frame_{index:05d}.json")
    w, h = doc["width"], doc["height"]
    bgr = cv2.imdecode(np.fromfile(clip / "frames" / f"frame_{index:05d}.png", np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise StageError(f"frame {index} tidak terbaca di {clip / 'frames'}")
    big = cv2.resize(bgr, (w * scale, h * scale), interpolation=cv2.INTER_LINEAR)
    big = cv2.addWeighted(big, 1 - FADE, np.full_like(big, 255), FADE, 0)
    st = draw_strokes(big, doc["strokes"], w, h, scale)
    kinds = {k: sum(s["type"] == k for s in doc["strokes"]) for k in COLORS}
    legend = [(LABEL, (255, 255, 255)),
              (f"frame {index}  [{tag}]", (255, 255, 255)),
              (f"silhouette {kinds['silhouette']} (merah)", COLORS["silhouette"]),
              (f"silhouette_hole {kinds['silhouette_hole']} (cyan)", COLORS["silhouette_hole"]),
              (f"group_boundary {kinds['group_boundary']} (hijau)", COLORS["group_boundary"]),
              (f"occlusion {kinds['occlusion']} (oranye)", COLORS["occlusion"]),
              (f"tepi frame {st['edge_segments']} seg (magenta putus-putus)", EDGE_COLOR)]
    if dropped is not None:
        gmap = stb.read_groups(clip / "stable" / "groups" / f"frame_{index:05d}.png")
        if gmap is None:
            raise StageError(f"peta grup frame {index} tidak terbaca di {clip / 'stable' / 'groups'}")
        drops = dropped_contours(gmap, dropped["min_region_area"], dropped["min_hole_area"])
        draw_dropped(big, drops, scale)
        n_hole = sum(d["kind"] == "hole" for d in drops)
        legend += [(f"DIBUANG lubang < {dropped['min_hole_area']} px: {n_hole} (abu-abu terang, label luas)",
                    DROP_HOLE_COLOR),
                   (f"DIBUANG komponen luar < {dropped['min_region_area']} px: {len(drops) - n_hole} "
                    f"(abu-abu gelap)", (150, 150, 150))]
    text_box(big, legend, 6, 6)
    if panel is not None:
        big = np.hstack([big, grad_panel(clip, index, scale, panel, doc["strokes"])])
    return cv2.cvtColor(big, cv2.COLOR_BGR2RGB)


class OverlaySource(ex.FrameSource):
    kind = "overlay"

    def __init__(self, clip: Path, plan: dict[str, tuple[int, str]], scale: int, dropped: dict | None = None,
                 panel: dict | None = None):
        self.clip, self.plan, self.scale, self.dropped, self.panel = clip, plan, scale, dropped, panel

    def check(self, names: list[str]) -> None:
        pass

    def render(self, name: str) -> np.ndarray:
        index, tag = self.plan[name]
        return render(self.clip, index, self.scale, tag, self.dropped, self.panel)


def hole_frames(records: dict[str, dict], exclude: set[int], n: int) -> list[int]:
    """Frame ber-lubang (n_hole ≥ 1) di luar `exclude`, n frame yang tersebar merata."""
    cand = sorted(r["index"] for r in records.values() if r.get("n_hole", 0) >= 1 and r["index"] not in exclude)
    if len(cand) <= n:
        return cand
    return [cand[round(i * (len(cand) - 1) / (n - 1))] for i in range(n)] if n > 1 else cand[:1]


def pct(vals: list[float], q: float) -> float:
    return float(np.percentile(vals, q)) if vals else 0.0


def empty_runs(indices: list[int], empty: list[bool]) -> list[dict]:
    """Run frame berurutan (indeks naik 1) dengan empty=True → [{start, length}]."""
    runs: list[dict] = []
    for i, e in zip(indices, empty):
        if not e:
            continue
        if runs and runs[-1]["start"] + runs[-1]["length"] == i:
            runs[-1]["length"] += 1
        else:
            runs.append({"start": i, "length": 1})
    return runs


def lower_clothing_region(clip: Path, index: int, strokes: list[dict]) -> list[dict]:
    """Per strok: jumlah titik, titik di piksel Lower_Clothing (seg/classmap) dan fraksinya. Hanya dibaca alat ini."""
    cm = cv2.imdecode(np.fromfile(clip / "seg" / "classmap" / f"frame_{index:05d}.png", np.uint8), cv2.IMREAD_UNCHANGED)
    lc = load_class_names().index(LEG_CLASS)
    out = []
    for s in strokes:
        inside = sum(int(cm[int(y), int(x)]) == lc for x, y in s["points"])
        out.append({"points": len(s["points"]), "inside": inside, "fraction": inside / len(s["points"]),
                    "group": s["groups"][0]})
    return out


def occlusion_summary(clip: Path, rs: list[dict]) -> dict:
    """Blok `occlusion` ringkasan: linimasa per frame, run frame kosong, kriteria wilayah Lower_Clothing (Done-when)."""
    idx = [r["index"] for r in rs]
    n_occ = [r["n_occlusion"] for r in rs]
    runs = empty_runs(idx, [n == 0 for n in n_occ])
    lengths = sorted(r["length"] for r in runs)
    stats = read_json(clip / "contours" / "clip_stats.json")
    legs, per_group = {}, {}
    for i in range(73, 93):
        if i not in idx:
            continue
        occ = [s for s in read_json(clip / "contours" / f"frame_{i:05d}.json")["strokes"] if s["type"] == "occlusion"]
        if 73 <= i <= 92:
            per_group[str(i)] = {g: sum(s["groups"][0] == g for s in occ) for g in sorted({s["groups"][0] for s in occ})}
        if i in LEG_FRAMES:
            reg = lower_clothing_region(clip, i, occ)
            legs[str(i)] = {"strokes": len(occ), "pass_80pct": any(r["fraction"] >= LEG_FRACTION for r in reg),
                            "best_fraction": max((r["fraction"] for r in reg), default=0.0),
                            "longest_px_in_region": max((r["inside"] for r in reg), default=0),
                            "total_px_in_region": sum(r["inside"] for r in reg), "ref_T102c_px": REF_LEN_T102C[i]}
    return {
        "thresholds": {k: stats[k] for k in ("t_high", "t_low", "hi_pct", "lo_pct", "n_values")},
        "strokes_per_frame": {"min": min(n_occ), "median": float(np.median(n_occ)), "max": max(n_occ)},
        "frames_without_occlusion": sum(n == 0 for n in n_occ),
        "empty_runs": {"count": len(runs), "longest": max(lengths, default=0),
                       "longest_at": next((r["start"] for r in runs if r["length"] == max(lengths)), None),
                       "length_distribution": {str(k): lengths.count(k) for k in sorted(set(lengths))}},
        "pixels_per_frame": {k: {"min": min(r[k] for r in rs), "median": float(np.median([r[k] for r in rs])),
                                 "max": max(r[k] for r in rs)} for k in ("occ_px_hyst", "occ_px_dist", "occ_px_len")},
        "timeline": [[r["index"], r["occ_px_hyst"], r["occ_px_dist"], r["occ_px_len"], r["n_occlusion"]] for r in rs],
        "done_when_legs": {"frames": legs, "frames_passing": sum(v["pass_80pct"] for v in legs.values()),
                           "needed": 4},
        "occlusion_per_group_73_92": per_group,
        "loops_total": sum(r["occ_loops"] for r in rs),
        "spurs_dropped_total": sum(r["occ_spurs_dropped"] for r in rs),
        "short_dropped_total": sum(r["occ_short_dropped"] for r in rs)}


def worst_cases(clip: Path, indices: list[int]) -> list[tuple[str, int, str]]:
    """Kasus terburuk (jenis, frame, keterangan): oklusi terdekat ke strok tipe lain (kandidat bayangan), strok oklusi
    terpendek yang lolos L, frame dengan strok oklusi terbanyak, dan frame DEBUG_FRAME."""
    from scipy.spatial import cKDTree

    nearest = (None, None)
    shortest = (None, None)
    most = (-1, None)
    for i in indices:
        strokes = read_json(clip / "contours" / f"frame_{i:05d}.json")["strokes"]
        occ = [s for s in strokes if s["type"] == "occlusion"]
        oth = [p for s in strokes if s["type"] != "occlusion" for p in s["points"]]
        if occ and oth:
            d = float(cKDTree(np.array(oth, float)).query(np.array([p for s in occ for p in s["points"]], float))[0].min())
            if nearest[0] is None or d < nearest[0]:
                nearest = (d, i)
        for s in occ:
            if shortest[0] is None or len(s["points"]) < shortest[0]:
                shortest = (len(s["points"]), i)
        if len(occ) > most[0]:
            most = (len(occ), i)
    out = []
    if nearest[1] is not None:
        out.append(("nearest_boundary", nearest[1], f"oklusi terdekat ke batas: {nearest[0]:.2f} px"))
    if shortest[1] is not None:
        out.append(("shortest_stroke", shortest[1], f"strok oklusi terpendek: {shortest[0]} titik"))
    if most[1] is not None:
        out.append(("most_strokes", most[1], f"strok oklusi terbanyak: {most[0]}"))
    if DEBUG_FRAME in indices:
        out.append(("frame90", DEBUG_FRAME, "komponen DA +-94 px area tangan (T-102c)"))
    return out


def summarize(clip: Path) -> dict:
    """Ringkasan [4]: bagian T-201a + blok `occlusion` (T-201b) bila contours/ memuatnya."""
    out = summarize_base(clip)
    recs = {r["frame"]: r for r in read_jsonl(clip / "contours" / "frames.jsonl") if r.get("event") == "frame"}
    rs = [recs[k] for k in sorted(recs)]
    if rs and "n_occlusion" in rs[0] and (clip / "contours" / "clip_stats.json").is_file():
        out["occlusion"] = occlusion_summary(clip, rs)
    return out


def summarize_base(clip: Path) -> dict:
    """Angka [4] dari contours/frames.jsonl (record frame terakhir per frame) + manifest."""
    cdir = clip / "contours"
    recs = {r["frame"]: r for r in read_jsonl(cdir / "frames.jsonl") if r.get("event") == "frame"}
    rs = [recs[k] for k in sorted(recs)]
    if not rs:
        raise StageError(f"{cdir / 'frames.jsonl'} tidak memuat record frame")
    t = [r["total_s"] for r in rs]

    def mmm(key: str) -> dict:
        v = [r[key] for r in rs]
        return {"min": min(v), "median": float(np.median(v)), "max": max(v)}

    dropped = sorted(a for r in rs for a in r["holes_dropped_areas"])
    json_bytes = sum(p.stat().st_size for p in cdir.glob("frame_*.json"))
    return {
        "clip": clip.name, "frames": len(rs), "contract": read_json(cdir / "manifest.json").get("contract"),
        "time_s": {"mean": float(np.mean(t)), "p95": pct(t, 95), "max": max(t), "over_1s": sum(x > 1 for x in t)},
        "json_bytes": {"total": json_bytes, "per_frame_mean": json_bytes / len(rs),
                       "per_frame_max": max(r["bytes"] for r in rs)},
        "strokes_per_frame": {"silhouette": mmm("n_silhouette"), "silhouette_hole": mmm("n_hole"),
                              "group_boundary": mmm("n_boundary")},
        "frames_without_silhouette": sum(r["n_silhouette"] == 0 for r in rs),
        "frames_with_holes": sum(r["n_hole"] >= 1 for r in rs),
        "edge_points": {"frames_touching": sum(r["edge_points"] > 0 for r in rs),
                        "mean": float(np.mean([r["edge_points"] for r in rs])),
                        "max": max(r["edge_points"] for r in rs)},
        "calibration_T305": {
            "holes_raw_total": sum(r["holes_raw"] for r in rs),
            "holes_kept_total": sum(r["n_hole"] for r in rs),
            "holes_dropped_by_area_total": len(dropped),
            "holes_dropped_by_parent_total": sum(r["holes_dropped_parent"] for r in rs),
            "holes_dropped_area_px": {"min": dropped[0] if dropped else None,
                                      "median": float(np.median(dropped)) if dropped else None,
                                      "p90": pct(dropped, 90) if dropped else None,
                                      "max": dropped[-1] if dropped else None},
            "regions_raw_total": sum(r["regions_raw"] for r in rs),
            "regions_dropped_total": sum(r["regions_dropped"] for r in rs),
            "group_boundary_loops_total": sum(r["loops"] for r in rs),
            "spurs_dropped_total": sum(r["spurs_dropped"] for r in rs),
            "spurs_dropped_by_length_px": {str(k): sum(r["spur_lengths"].count(k) for r in rs) for k in range(1, 6)},
            "short_fragments_dropped_total": sum(r["short_dropped"] for r in rs)},
    }


def write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


# ── Mode --track (T-202): warna per track_id, anchor, arah, kasus terburuk ──
TRACK_LABEL = "temporal belum aktif - kedip / lahir-mati strok BELUM representatif"
TRACK_RANGES = "73-92,183-202,225-240"
TRACK_OUT_DIR = "work/t202"
PALETTE = ((0, 0, 230), (230, 120, 0), (0, 170, 0), (200, 0, 200), (0, 170, 230), (150, 90, 0), (0, 90, 200),
           (110, 160, 0), (180, 0, 90), (0, 130, 130), (90, 0, 200), (60, 60, 60))   # BGR, dipilih track_id % len
ANCHOR_RADIUS, START_RADIUS, ARROW_SPAN = 6, 4, 10   # piksel layar; ARROW_SPAN = indeks titik ujung panah
ARM_FRAMES = (213, 220, 228, 232, 233, 234, 235, 236, 249, *range(257, 267))   # lengan terlepas di klip test
COMPARE_DEFAULT = "12:X,16:Y"                # ambang:pendekatan (X = global, Y = silhouette terbesar mewarisi id)
TRACK_TAG = [""]                             # label konfigurasi di legenda (--label), mis. "Y@16 (usulan)"


def track_color(track_id: int) -> tuple:
    return PALETTE[track_id % len(PALETTE)]


def retrack(clip: Path, thr: float, inherit: bool) -> list[list[dict]]:
    """Jalankan ulang pelacak (track.Tracker) atas himpunan titik strok di contours/ (tanpa menulis apa pun)."""
    from rotoscope import track as trk
    from rotoscope.config import load_pipeline
    names = tuple(g for g, _ in load_pipeline().groups)
    tr = trk.Tracker(thr, inherit_main=inherit)
    out = []
    for p in sorted((clip / "contours").glob("frame_*.json")):
        i = int(p.stem.split("_")[1])
        strokes = [dict(s) for s in read_json(p)["strokes"]]
        gm = stb.read_groups(clip / "stable" / "groups" / f"frame_{i:05d}.png")
        tr.step(strokes, vec.hair_face_mask(gm, names))
        out.append(strokes)
    return out


def stored_frames(clip: Path) -> list[list[dict]]:
    return [read_json(p)["strokes"] for p in sorted((clip / "contours").glob("frame_*.json"))]


def main_silhouette(strokes: list[dict]) -> dict | None:
    from rotoscope import track as trk
    sil = [s for s in strokes if s["type"] == "silhouette"]
    return max(sil, key=lambda s: abs(trk.signed_area(np.asarray(s["points"], float)))) if sil else None


def draw_tracks(img: np.ndarray, strokes: list[dict], scale: int) -> None:
    """Warna per track_id; tertutup / loop = titik anchor besar (points[0]); garis terbuka = lingkaran titik awal +
    panah arah; label #id kecil di tengah strok."""
    for s in strokes:
        pts, color = s["points"], track_color(s["track_id"])
        closed = bool(s["closed"])
        arr = np.array([px(p, scale) for p in pts], np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [arr], closed, color, 2, cv2.LINE_AA, SHIFT)
        loop = (not closed) and len(pts) > 3 and pts[0] == pts[-1]
        a = px(pts[0], scale)
        a = (a[0] >> SHIFT, a[1] >> SHIFT)
        if closed or loop:
            cv2.circle(img, a, ANCHOR_RADIUS, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(img, a, ANCHOR_RADIUS - 2, color, -1, cv2.LINE_AA)
        else:
            cv2.circle(img, a, START_RADIUS, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(img, a, START_RADIUS - 1, color, 1, cv2.LINE_AA)
            b = px(pts[min(ARROW_SPAN, len(pts) - 1)], scale)
            cv2.arrowedLine(img, a, (b[0] >> SHIFT, b[1] >> SHIFT), color, 2, cv2.LINE_AA, tipLength=0.5)
        mid = px(pts[len(pts) // 2], scale)
        cv2.putText(img, f"#{s['track_id']}", (mid[0] >> SHIFT, mid[1] >> SHIFT), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    color, 1, cv2.LINE_AA)


def render_track(clip: Path, index: int, scale: int, tag: str, strokes: list[dict], extra: list[tuple[str, tuple]]
                 ) -> np.ndarray:
    """Frame asli (dipucatkan) × skala + strok berwarna per track_id + legenda → BGR uint8."""
    doc = read_json(clip / "contours" / f"frame_{index:05d}.json")
    w, h = doc["width"], doc["height"]
    bgr = cv2.imdecode(np.fromfile(clip / "frames" / f"frame_{index:05d}.png", np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise StageError(f"frame {index} tidak terbaca di {clip / 'frames'}")
    big = cv2.addWeighted(cv2.resize(bgr, (w * scale, h * scale), interpolation=cv2.INTER_LINEAR), 1 - FADE,
                          np.full((h * scale, w * scale, 3), 255, np.uint8), FADE, 0)
    draw_tracks(big, strokes, scale)
    main = main_silhouette(strokes)
    legend = [(TRACK_LABEL, (255, 255, 255)), (f"frame {index}  [{tag}]", (255, 255, 255)),
              *([(TRACK_TAG[0], (0, 255, 255))] if TRACK_TAG[0] else []),
              ("warna = track_id; titik besar = anchor (tertutup / loop); lingkaran + panah = titik awal garis terbuka",
               (255, 255, 255))]
    if main is not None:
        legend.append((f"silhouette utama #{main['track_id']}  anchor {tuple(main['points'][0])}",
                       track_color(main["track_id"])))
    for s in strokes:
        if s["type"] == "silhouette" and s is not main:
            legend.append((f"silhouette lain #{s['track_id']} ({len(s['points'])} titik)  anchor {tuple(s['points'][0])}",
                           track_color(s["track_id"])))
    text_box(big, legend + extra, 6, 6, 0.45)
    return big


def pick_track_worst(frames: list[list[dict]]) -> list[tuple[str, int, str]]:
    """Kasus terburuk: lompatan anchor terbesar, pembalikan arah (bila ada), id baru terbanyak, padanan paling ambigu."""
    from rotoscope import track as trk
    seen: set[int] = set()
    jump = (-1.0, None)
    newest = (-1, None)
    rev = None
    ambig = (None, None)
    prev_main, prev_items = None, None
    for k, fr in enumerate(frames):
        m = main_silhouette(fr)
        if m is not None and prev_main is not None:
            d = float(np.linalg.norm(np.array(m["points"][0]) - np.array(prev_main["points"][0])))
            if d > jump[0]:
                jump = (d, k)
        n_new = sum(s["track_id"] not in seen for s in fr) if k else 0
        if n_new > newest[0]:
            newest = (n_new, k)
        items = [trk.make_item(s, s["track_id"]) for s in fr]
        if prev_items is not None and items and prev_items:
            cost = trk.cost_matrix(items, prev_items)
            pid = {it.track_id: j for j, it in enumerate(prev_items)}
            for i, it in enumerate(items):
                if it.track_id not in pid:
                    continue
                row = np.sort(cost[i][np.isfinite(cost[i])])
                if len(row) >= 2 and row[0] > 0 and (ambig[0] is None or row[1] / row[0] < ambig[0]):
                    ambig = (float(row[1] / row[0]), k)
                if (not it.cyclic) and rev is None:
                    a, b = prev_items[pid[it.track_id]].pts, it.pts
                    if (np.linalg.norm(b[0] - a[-1]) + np.linalg.norm(b[-1] - a[0])
                            < np.linalg.norm(b[0] - a[0]) + np.linalg.norm(b[-1] - a[-1])):
                        rev = k
        seen.update(s["track_id"] for s in fr)
        prev_main, prev_items = m, items
    out = []
    if jump[1] is not None:
        out.append(("max_anchor_jump", jump[1], f"lompatan anchor silhouette utama terbesar: {jump[0]:.1f} px"))
    if rev is not None:
        out.append(("direction_reversal", rev, "pembalikan arah garis terbuka"))
    if newest[1] is not None:
        out.append(("most_new_ids", newest[1], f"id baru terbanyak: {newest[0]}"))
    if ambig[1] is not None:
        out.append(("most_ambiguous", ambig[1], f"padanan paling ambigu: rasio kandidat ke-2 / terbaik {ambig[0]:.2f}"))
    return out


def parse_compare(text: str) -> list[tuple[float, bool]]:
    out = []
    for part in text.split(","):
        thr, _, mode = part.strip().partition(":")
        out.append((float(thr), mode.upper() == "Y"))
    return out


def write_png(path: Path, bgr: np.ndarray) -> None:
    ok, buf = cv2.imencode(".png", bgr)
    if ok:
        write_atomic(path, buf.tobytes())


def main_track(a: argparse.Namespace) -> int:
    """PNG kasus terburuk + sampel + video overlay T-202 ke --out-dir (default work/t202)."""
    clip = a.work_dir
    meta = read_json(clip / "meta.json")
    manifest = read_json(clip / "contours" / "manifest.json")
    if manifest.get("contract") != "T-305b":
        print("ERROR: contours/ bukan kontrak T-305b — jalankan `python -m rotoscope vectorize` dulu", file=sys.stderr)
        return 1
    thr0 = manifest["vectorize"]["track.max_match_dist_px"]
    stored = stored_frames(clip)
    out = a.out_dir
    out.mkdir(parents=True, exist_ok=True)
    present = set(range(len(stored)))
    worst = pick_track_worst(stored)
    summary = {"clip": clip.name, "threshold_stored": thr0, "worst": [{"kind": k, "frame": i, "note": n} for k, i, n in worst]}
    for kind, i, note in worst:
        write_png(out / f"worst_{clip.name}_{kind}_f{i:05d}.png",
                  render_track(clip, i, a.scale, f"terburuk: {note}", stored[i], [(f"tersimpan: ambang {thr0}", (255, 255, 255))]))
        print(f"kasus terburuk [{kind}] frame {i}: {note}")
    arm = [i for i in ARM_FRAMES if i in present]
    if arm:
        variants = [(thr, inh, retrack(clip, thr, inh)) for thr, inh in parse_compare(a.compare)]
        for i in arm:
            panels = []
            for thr, inh, fr in variants:
                main = main_silhouette(fr[i])
                others = [s["track_id"] for s in fr[i] if s["type"] == "silhouette" and s is not main]
                extra = [(f"ambang {thr:g} px, pendekatan {'Y (terbesar mewarisi id)' if inh else 'X (global)'}: "
                          f"utama #{main['track_id']}, komponen lain {['#' + str(t) for t in others]}", (255, 255, 255))]
                panels.append(render_track(clip, i, a.scale, "lengan terlepas", fr[i], extra))
            write_png(out / f"arm_{clip.name}_f{i:05d}.png", np.hstack(panels))
        summary["arm_frames"] = arm
        summary["arm_compare"] = [{"threshold": t, "inherit_main": inh, "main_ids": sorted({main_silhouette(f)["track_id"]
                                  for f in fr if main_silhouette(f)})} for t, inh, fr in variants]
        print(f"lengan terlepas: {len(arm)} frame (panel kiri-kanan = {a.compare}) → arm_{clip.name}_fNNNNN.png")
    wanted = [i for i in parse_ranges(a.ranges) if i in present]
    plan = {f"frame_{i:05d}": (i, "rentang") for i in wanted}
    for _, i, _ in worst:
        plan.setdefault(f"frame_{i:05d}", (i, "terburuk"))
    write_atomic(out / f"summary_t202_{clip.name}.json", (json.dumps(summary, indent=2) + "\n").encode("utf-8"))
    if a.no_video or not plan:
        return 0
    first = render_track(clip, next(iter(plan.values()))[0], a.scale, "x", stored[next(iter(plan.values()))[0]], [])
    size = (first.shape[1], first.shape[0])

    class TrackSource(OverlaySource):
        def render(self, name: str) -> np.ndarray:
            i, tag = self.plan[name]
            return cv2.cvtColor(render_track(clip, i, a.scale, tag, stored[i],
                                             [(f"tersimpan: ambang {thr0}", (255, 255, 255))]), cv2.COLOR_BGR2RGB)

    video = out / f"overlay_track_{clip.name}.mp4"
    tmp = video.with_name(video.name + ".tmp")
    ex.encode(TrackSource(clip, plan, a.scale), list(plan), size, np.array([255, 255, 255], np.uint8),
              float(meta.get("target_fps", 24)), a.crf, "medium", None, tmp)
    os.replace(tmp, video)
    print(f"video → {video} ({len(plan)} frame, {video.stat().st_size / 2**20:.2f} MiB)")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--work-dir", type=Path, required=True, help="folder klip, mis. work/clips/test")
    ap.add_argument("--out-dir", type=Path, default=Path(DEFAULT_OUT_DIR))
    ap.add_argument("--ranges", default=DEFAULT_RANGES, help="indeks frame, mis. 73-92,183-202")
    ap.add_argument("--hole-frames", type=int, default=20, help="jumlah frame ber-lubang tambahan")
    ap.add_argument("--samples", default=DEFAULT_SAMPLES, help="indeks frame untuk PNG sampel")
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--crf", type=int, default=20)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--suffix", default="", help="akhiran nama keluaran (mis. _fix1) supaya tidak menimpa berkas lama")
    ap.add_argument("--dropped", action="store_true",
                    help="HANYA tulis overlay_<klip>_dropped.mp4 (kontur yang dibuang filter ukuran, abu-abu)")
    ap.add_argument("--grad-panel", action="store_true", help="panel pendamping |grad| di kanan (video + PNG)")
    ap.add_argument("--worst", action="store_true", help="tulis PNG kasus terburuk worst_<klip>_<jenis>_fNNNNN.png")
    ap.add_argument("--track", action="store_true",
                    help=f"mode T-202: warna per track_id, anchor / arah, PNG kasus terburuk + lengan terlepas (bandingkan "
                         f"--compare), video; default --out-dir {TRACK_OUT_DIR}, --ranges {TRACK_RANGES}")
    ap.add_argument("--compare", default=COMPARE_DEFAULT,
                    help="mode --track, frame lengan terlepas: ambang:pendekatan dipisah koma, X = global, Y = silhouette "
                         f"terbesar mewarisi id (default {COMPARE_DEFAULT})")
    ap.add_argument("--label", default="", help="mode --track: label konfigurasi di legenda, mis. 'Y@16 (usulan)'")
    a = ap.parse_args(argv)
    TRACK_TAG[0] = a.label
    if a.track:
        if a.out_dir == Path(DEFAULT_OUT_DIR):
            a.out_dir = Path(TRACK_OUT_DIR)
        if a.ranges == DEFAULT_RANGES:
            a.ranges = TRACK_RANGES
        return main_track(a)
    clip = a.work_dir
    meta = read_json(clip / "meta.json")
    present = {int(p.stem.split("_")[1]) for p in (clip / "contours").glob("frame_*.json")}
    if not present:
        print(f"ERROR: {clip / 'contours'} kosong — jalankan `python -m rotoscope vectorize` dulu", file=sys.stderr)
        return 1
    a.out_dir.mkdir(parents=True, exist_ok=True)
    if not a.dropped:
        summary = summarize(clip)
        summary_path = a.out_dir / f"summary_{clip.name}{a.suffix}.json"
        write_atomic(summary_path, (json.dumps(summary, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
        print(f"ringkasan → {summary_path}")

    wanted = [i for i in parse_ranges(a.ranges) if i in present]
    skipped = [i for i in parse_ranges(a.ranges) if i not in present]
    if skipped:
        print(f"catatan: {len(skipped)} frame di --ranges tidak ada di klip ini (mis. {skipped[0]}) — dilewati")
    recs = {r["frame"]: r for r in read_jsonl(clip / "contours" / "frames.jsonl") if r.get("event") == "frame"}
    holes = hole_frames(recs, set(wanted), a.hole_frames)
    plan: dict[str, tuple[int, str]] = {}
    for i in wanted:
        plan[f"frame_{i:05d}"] = (i, "rentang")
    for i in holes:
        plan.setdefault(f"frame_{i:05d}", (i, "ber-lubang"))
    no_occ = sorted(r["index"] for r in recs.values() if r.get("n_occlusion") == 0 and r["index"] not in wanted)
    if len(no_occ) > N_NO_OCCLUSION_FRAMES:
        no_occ = [no_occ[round(k * (len(no_occ) - 1) / (N_NO_OCCLUSION_FRAMES - 1))] for k in range(N_NO_OCCLUSION_FRAMES)]
    for i in no_occ:
        plan.setdefault(f"frame_{i:05d}", (i, "tanpa oklusi"))

    dropped = None
    panel = panel_context(clip) if a.grad_panel and not a.dropped else None
    if a.dropped:
        params = read_json(clip / "contours" / "manifest.json")["vectorize"]
        dropped = {k: params[k] for k in ("min_region_area", "min_hole_area")}
    else:
        for i in [x for x in parse_ranges(a.samples) if x in present] + holes[:1]:
            img = render(clip, i, a.scale, "sampel", None, panel)
            ok, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            if ok:
                write_atomic(a.out_dir / f"sample_{clip.name}_f{i:05d}{a.suffix}.png", buf.tobytes())
        if a.worst:
            for kind, i, note in worst_cases(clip, sorted(present)):
                img = render(clip, i, a.scale, f"terburuk: {note}", None, panel_context(clip))
                ok, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                if ok:
                    write_atomic(a.out_dir / f"worst_{clip.name}_{kind}_f{i:05d}{a.suffix}.png", buf.tobytes())
                    print(f"kasus terburuk [{kind}] frame {i}: {note}")
    if a.no_video or not plan:
        return 0
    first = render(clip, next(iter(plan.values()))[0], a.scale, "x", dropped, panel)
    size = (first.shape[1], first.shape[0])
    out = a.out_dir / f"overlay_{clip.name}{a.suffix}{'_dropped' if a.dropped else ''}.mp4"
    tmp = out.with_name(out.name + ".tmp")
    names = list(plan)
    ex.encode(OverlaySource(clip, plan, a.scale, dropped, panel), names, size, np.array([255, 255, 255], np.uint8),
              float(meta.get("target_fps", 24)), a.crf, "medium", None, tmp)
    os.replace(tmp, out)
    print(f"video → {out} ({len(names)} frame, {out.stat().st_size / 2**20:.2f} MiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
