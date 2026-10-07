"""Alat pratinjau sekali pakai T-203a (BUKAN bagian src/): keluaran visual stage [5] ke work/t203a/ (ter-ignore).

    python scripts/strokes_preview.py videos      # video H.264 PNG hasil stage (rentang 73-92, 183-202, 225-240 + ...)
    python scripts/strokes_preview.py edge        # hide vs draw (frame 233 + frame tepi kanan terbanyak)
    python scripts/strokes_preview.py widths      # width_base {3.6, 5.4, 7.2, 9.0} px ref, frame 80 dan 233
    python scripts/strokes_preview.py epsilon     # simplify_epsilon {2.8, 5.6} px ref, frame 80 dan 233
    python scripts/strokes_preview.py worst       # PNG kasus terburuk (deviasi, terpanjang, run tepi, strok terbanyak,
                                                  # kontak paling dangkal, sudut kanvas) + crop zoom 3x
    python scripts/strokes_preview.py zoom        # crop zoom 3x: sambungan strok tertutup, ujung terbuka, tepi bawah
    python scripts/strokes_preview.py all

T-401 (keluaran ke work/t401/; render LANGSUNG dari contours/ klip, strokes/ dan out/ tidak dipakai / tidak diubah):
    python scripts/strokes_preview.py boards      # papan frame 80 + 233: amplitudo, skala noise, taper, taper oklusi, hierarki tipe (profil linear dihapus di Tahap 4)
    python scripts/strokes_preview.py variants    # varian A / B / C berdampingan + zoom 3x (A dan C hanya studi, tidak ada di produksi)
    python scripts/strokes_preview.py videos401   # video jendela 73-92 dan 183-202: skala x1 vs x2, B vs A, taper oklusi hidup vs mati, peta tebal berubah
    python scripts/strokes_preview.py changemap   # PNG peta tebal berubah (x1 dan x2 berdampingan)
    python scripts/strokes_preview.py worst401    # PNG kasus terburuk T-401 + zoom 3x + ringkasan JSON
    python scripts/strokes_preview.py svgsample   # SVG sampel frame 80 + info struktur
    python scripts/strokes_preview.py all401

T-402 (keluaran ke work/t402/; render LANGSUNG dari contours/ SALINAN klip: --clips-root <scratchpad>/work/clips; klip asli tidak dipakai):
    python scripts/strokes_preview.py videos402 --clips-root R   # video papan [T-401 | varian] ukuran penuh: amplitudo, frequency, hold x drift,
                                                                 # stroke_independence (+ peta id baru), jendela statis otomatis + gerak cepat, panel fixed
    python scripts/strokes_preview.py helpers402 --clips-root R  # strip 3 frame ditumpuk, zoom 3x sambungan T / seam / garis sejajar / tepi / tikungan
    python scripts/strokes_preview.py worst402 --clips-root R    # PNG kasus terburuk (sambungan, Jacobian, ujung tepi, persilangan) + ringkasan JSON
    python scripts/strokes_preview.py all402 --clips-root R

Membaca contours/ + strokes/ klip (tidak mengubahnya). Label alat T-203a: "garis polos T-203a - belum ada jitter / taper / tekstur".
Fungsi metrik dipakai ulang dari tests/stylize_metrics.py; encode dipakai ulang dari rotoscope.export.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

import jitter_metrics as jm  # noqa: E402
import stylize_metrics as sm  # noqa: E402

from rotoscope import export as ex  # noqa: E402
from rotoscope import noise  # noqa: E402
from rotoscope import stylize as sty  # noqa: E402
from rotoscope.config import load_style  # noqa: E402

CLIPS = ROOT / "work" / "clips"
OUT = ROOT / "work" / "t203a"
STYLE_PATH = ROOT / "configs" / "styles" / "rough-sketch.yaml"
LABEL = "garis polos T-203a - belum ada jitter / taper / tekstur"
CLIP_NAMES = ("test_short", "test")
RANGES = ((73, 92), (183, 202), (225, 240))
WIDTHS_REF = (3.6, 5.4, 7.2, 9.0)
EPSILONS_REF = (2.8, 5.6)
COMPARE_FRAMES = (80, 233)
ZOOM = 3
ZOOM_HALF = 90                       # setengah sisi crop (px output) → 180 px × 3 = 540 px
VIDEO_CRF, VIDEO_PRESET = 18, "medium"


# ── I/O ────────────────────────────────────────────
def read_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def meta(clip: str) -> dict:
    return read_json(CLIPS / clip / "meta.json")


def frame_count(clip: str) -> int:
    return int(meta(clip)["frame_count"])


def load_doc(clip: str, i: int) -> dict:
    return read_json(CLIPS / clip / "contours" / f"frame_{i:05d}.json")


def read_png_rgb(p: Path) -> np.ndarray:
    bgr = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise SystemExit(f"gagal membaca {p} — jalankan dulu: python -m rotoscope stylize <video>")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def write_png(p: Path, rgb: np.ndarray) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise SystemExit(f"encode PNG gagal: {p}")
    buf.tofile(str(p))
    print(f"  {p.relative_to(ROOT)}")


def label(img: np.ndarray, lines: list[str]) -> np.ndarray:
    out = np.ascontiguousarray(img.copy())
    for k, text in enumerate(lines):
        y = 26 + 26 * k
        cv2.putText(out, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(out, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 30, 30), 1, cv2.LINE_AA)
    return out


def style_with(**overrides):
    return load_style(STYLE_PATH, overrides=overrides or None)


def render_rgb(clip: str, doc: dict, style) -> tuple[np.ndarray, sty.Geometry, list[sty.Piece], dict]:
    m = meta(clip)
    g = sty.make_geometry(style, int(m["working_width"]), int(m["working_height"]))
    _, png, stats = sty.render_frame(doc, g, style)
    pieces, _ = sty.frame_pieces(doc, g)
    return sm.decode_png(png), g, pieces, stats


def hstack(images: list[np.ndarray], scale: float = 1.0, gap: int = 8) -> np.ndarray:
    if scale != 1.0:
        images = [cv2.resize(i, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) for i in images]
    h = max(i.shape[0] for i in images)
    parts = []
    for i in images:
        pad = np.full((h - i.shape[0], i.shape[1], 3), 255, np.uint8)
        parts += [np.vstack([i, pad]), np.full((h, gap, 3), 255, np.uint8)]
    return np.hstack(parts[:-1])


def crop_zoom(img: np.ndarray, cx: float, cy: float, half: int = ZOOM_HALF, zoom: int = ZOOM) -> tuple[np.ndarray, tuple[int, int]]:
    h, w = img.shape[:2]
    x0 = int(min(max(cx - half, 0), max(w - 2 * half, 0)))
    y0 = int(min(max(cy - half, 0), max(h - 2 * half, 0)))
    c = img[y0:y0 + 2 * half, x0:x0 + 2 * half]
    return cv2.resize(c, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST), (x0, y0)


def overlay_contour(zoomed: np.ndarray, doc: dict, g: sty.Geometry, origin: tuple[int, int], zoom: int = ZOOM) -> np.ndarray:
    """Titik kontur asli (terskala) sebagai titik merah di atas crop yang diperbesar."""
    out = zoomed.copy()
    x0, y0 = origin
    for s in doc["strokes"]:
        for x, y in np.asarray(s["points"], float) * g.scale:
            px, py = int((x - x0) * zoom), int((y - y0) * zoom)
            if 0 <= px < out.shape[1] and 0 <= py < out.shape[0]:
                cv2.circle(out, (px, py), 2, (230, 40, 40), -1)
    return out


# ── videos ─────────────────────────────────────────
class PngSource(ex.FrameSource):
    kind = "png"

    def __init__(self, clip: str):
        self.clip = clip
        self.dir = CLIPS / clip / "strokes"

    def check(self, names: list[str]) -> None:
        missing = [n for n in names if not (self.dir / n).is_file()]
        if missing:
            raise SystemExit(f"{len(missing)} PNG strokes hilang, mis. {missing[0]} — jalankan stage [5]")

    def render(self, name: str) -> np.ndarray:
        return label(read_png_rgb(self.dir / name), [LABEL, f"{self.clip} {Path(name).stem}"])


def make_video(clip: str, frames: list[int], name: str) -> None:
    src = PngSource(clip)
    names = [f"frame_{i:05d}.png" for i in frames]
    src.check(names)
    h, w = read_png_rgb(src.dir / names[0]).shape[:2]
    bg = np.array(ex.parse_hex(sty.load_style(STYLE_PATH).paper.color), np.uint8)
    OUT.mkdir(parents=True, exist_ok=True)
    tmp, final = OUT / (name + ".tmp"), OUT / name
    ex.encode(src, names, (w, h), bg, float(meta(clip)["target_fps"]), VIDEO_CRF, VIDEO_PRESET, None, tmp)
    os.replace(tmp, final)
    print(f"  {final.relative_to(ROOT)} ({final.stat().st_size / 1024:.0f} KiB, {len(names)} frame)")


def busiest_frames(clip: str) -> dict[str, int]:
    recs = [r for r in (json.loads(x) for x in (CLIPS / clip / "strokes" / "frames.jsonl").read_text(encoding="utf-8").splitlines())
            if r.get("event") == "frame"]
    last: dict[int, dict] = {r["index"]: r for r in recs}
    return {"most_strokes": max(last.values(), key=lambda r: r["n_strokes"])["index"],
            "most_points": max(last.values(), key=lambda r: r["n_points"])["index"]}


def cmd_videos() -> None:
    for clip in CLIP_NAMES:
        n = frame_count(clip)
        for a, b in RANGES:
            frames = [i for i in range(a, b + 1) if i < n]
            if frames:
                make_video(clip, frames, f"{clip}_f{frames[0]:03d}-{frames[-1]:03d}.mp4")
        for tag, idx in busiest_frames(clip).items():
            lo = max(idx - 8, 0)
            make_video(clip, list(range(lo, min(lo + 17, n))), f"{clip}_{tag}_f{idx:03d}.mp4")


# ── edge ───────────────────────────────────────────
def right_edge_frame(clip: str) -> int:
    best, best_n = 0, -1
    for f in sorted((CLIPS / clip / "contours").glob("frame_*.json")):
        d = read_json(f)
        n = sum(1 for s in d["strokes"] for x, _ in s["points"] if x == d["width"] - 0.5)
        if n > best_n:
            best, best_n = d["frame_index"], n
    return best


def cmd_edge() -> None:
    clip = "test"
    for tag, idx in (("bottom233", 233), ("right_max", right_edge_frame(clip))):
        doc = load_doc(clip, idx)
        panels, g = [], None
        for mode in ("hide", "draw"):
            img, g, pieces, stats = render_rgb(clip, doc, style_with(**{"shape.edge_mode": mode}))
            panels.append(label(img, [LABEL, f"{clip} frame {idx} edge_mode={mode}"]))
        write_png(OUT / f"edge_{tag}_hide_vs_draw.png", hstack(panels, 0.5))
        # crop zoom di tepi (bawah / kanan): kiri hide, kanan draw
        cx, cy = (g.out_w * 0.45, g.out_h - ZOOM_HALF) if tag == "bottom233" else (g.out_w - ZOOM_HALF, g.out_h * 0.5)
        crops = []
        for mode in ("hide", "draw"):
            img, g, pieces, _ = render_rgb(clip, doc, style_with(**{"shape.edge_mode": mode}))
            z, org = crop_zoom(img, cx, cy)
            crops.append(label(z, [f"{mode} 3x"]))
        write_png(OUT / f"edge_{tag}_zoom3x.png", hstack(crops))


# ── widths / epsilon ───────────────────────────────
def cmd_widths() -> None:
    clip = "test"
    for idx in COMPARE_FRAMES:
        doc = load_doc(clip, idx)
        panels, crops = [], []
        for w in WIDTHS_REF:
            img, g, _, _ = render_rgb(clip, doc, style_with(**{"stroke.width_base": w}))
            panels.append(label(img, [LABEL, f"{clip} frame {idx} width_base={w} px ref"]))
            z, _ = crop_zoom(img, g.out_w * 0.5, g.out_h * 0.38)
            crops.append(label(z, [f"width {w}"]))
        write_png(OUT / f"widths_f{idx:03d}.png", hstack(panels, 0.35))
        write_png(OUT / f"widths_f{idx:03d}_zoom3x.png", hstack(crops))


def cmd_epsilon() -> None:
    clip = "test"
    for idx in COMPARE_FRAMES:
        doc = load_doc(clip, idx)
        panels, crops = [], []
        for e in EPSILONS_REF:
            img, g, _, _ = render_rgb(clip, doc, style_with(**{"shape.simplify_epsilon": e}))
            panels.append(label(img, [LABEL, f"{clip} frame {idx} simplify_epsilon={e} px ref"]))
            z, _ = crop_zoom(img, g.out_w * 0.5, g.out_h * 0.38)
            crops.append(label(z, [f"epsilon {e}"]))
        write_png(OUT / f"epsilon_f{idx:03d}.png", hstack(panels, 0.5))
        write_png(OUT / f"epsilon_f{idx:03d}_zoom3x.png", hstack(crops))


# ── worst cases ────────────────────────────────────
def scan_worst(clip: str, style) -> dict:
    """Satu pass atas semua frame: deviasi terbesar, strok terpanjang, run tepi terpanjang, strok terbanyak, kontak paling
    dangkal, run sudut (dua tepi)."""
    m = meta(clip)
    g = sty.make_geometry(style, int(m["working_width"]), int(m["working_height"]))
    worst = {"dev": (-1.0, None), "longest": (-1.0, None), "edge_run": (-1, None), "most": (-1, None),
             "shallow": (999.0, None), "corner": (-1, None)}
    for f in sorted((CLIPS / clip / "contours").glob("frame_*.json")):
        d = read_json(f)
        i = d["frame_index"]
        if len(d["strokes"]) > worst["most"][0]:
            worst["most"] = (len(d["strokes"]), (i, None))
        _, st = sty.frame_pieces(d, g)
        if st["min_contact_deg"] < worst["shallow"][0]:
            worst["shallow"] = (st["min_contact_deg"], (i, None))
        for k, s in enumerate(d["strokes"]):
            p = np.asarray(s["points"], float)
            length = float(np.hypot(*np.diff(p, axis=0).T).sum()) * g.scale
            if length > worst["longest"][0]:
                worst["longest"] = (length, (i, k))
            dev = sm.stroke_deviation(s, g)
            if dev.size and dev.max() > worst["dev"][0]:
                ref = p * g.scale
                sides = sty.edge_sides(p, g.width, g.height)
                ref = ref[[not x for x in sides]]
                worst["dev"] = (float(dev.max()), (i, k, ref[int(np.argmax(dev))].tolist()))
            on = np.array([bool(x) for x in sty.edge_sides(p, g.width, g.height)])
            if on.any() and not on.all():
                closed = bool(s["closed"])
                for run in sty.split_edge_runs(on, closed)[1]:
                    if len(run) > worst["edge_run"][0]:
                        worst["edge_run"] = (len(run), (i, k))
                    sd = sty.edge_sides(p[run], g.width, g.height)
                    if any(len(x) > 1 for x in sd) or (any("bottom" in x for x in sd) and any("right" in x for x in sd)):
                        if len(run) > worst["corner"][0]:
                            worst["corner"] = (len(run), (i, k))
    return worst


def cmd_worst() -> None:
    style = style_with()
    summary = {}
    for clip in CLIP_NAMES:
        w = scan_worst(clip, style)
        summary[clip] = {k: {"value": v[0], "where": v[1]} for k, v in w.items()}
        for tag, (val, where) in w.items():
            if where is None:
                continue
            i = where[0]
            doc = load_doc(clip, i)
            img, g, pieces, stats = render_rgb(clip, doc, style)
            write_png(OUT / f"worst_{clip}_{tag}_f{i:03d}.png",
                      label(img, [LABEL, f"{clip} frame {i} kasus terburuk: {tag} = {val:.2f}"]))
            cx, cy = g.out_w / 2, g.out_h / 2
            if tag == "dev":
                cx, cy = where[2]
            elif tag in ("edge_run", "shallow", "corner", "longest"):
                s = doc["strokes"][where[1]] if where[1] is not None else None
                if tag == "corner":
                    cx, cy = g.out_w - ZOOM_HALF, g.out_h - ZOOM_HALF
                elif s is not None:
                    p = np.asarray(s["points"], float)
                    sides = sty.edge_sides(p, g.width, g.height)
                    on = [k for k, x in enumerate(sides) if x]
                    q = p[on[len(on) // 2]] if on else p[len(p) // 2]
                    cx, cy = q[0] * g.scale, q[1] * g.scale
                if tag == "shallow":
                    ends = sm_tails(pieces, g)
                    if ends:
                        cx, cy = ends[0]
            z, org = crop_zoom(img, cx, cy)
            z = overlay_contour(z, doc, g, org)
            write_png(OUT / f"worst_{clip}_{tag}_f{i:03d}_zoom3x.png", label(z, [f"{tag} 3x (titik merah = kontur asli)"]))
    (OUT / "worst_summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    print(f"  {(OUT / 'worst_summary.json').relative_to(ROOT)}")


def sm_tails(pieces: list[sty.Piece], g: sty.Geometry) -> list[tuple[float, float]]:
    """Titik silang (titik terakhir di dalam kanvas) ujung jalur yang diperpanjang keluar kanvas."""
    out = []
    for pc in pieces:
        for inner, outer in ((pc.points[1], pc.points[0]), (pc.points[-2], pc.points[-1])):
            if not (0 <= outer[0] <= g.out_w and 0 <= outer[1] <= g.out_h):
                out.append((float(inner[0]), float(inner[1])))
    return out


# ── zoom ───────────────────────────────────────────
def cmd_zoom() -> None:
    clip = "test"
    style = style_with()
    for idx in COMPARE_FRAMES:
        doc = load_doc(clip, idx)
        img, g, pieces, _ = render_rgb(clip, doc, style)
        targets = []
        sil = next((s for s in doc["strokes"] if s["type"] == "silhouette"), None)
        if sil is not None:                                    # sambungan strok tertutup = anchor points[0]
            x, y = sil["points"][0]
            targets.append(("seam_closed_anchor", x * g.scale, y * g.scale))
        gb = next((s for s in doc["strokes"] if s["type"] == "group_boundary" and len(s["points"]) > 20), None)
        if gb is not None:                                     # ujung terbuka
            x, y = gb["points"][-1]
            targets.append(("open_end", x * g.scale, y * g.scale))
        ends = sm_tails(pieces, g)
        if ends:                                               # tepi bawah / kanan: titik silang
            targets.append(("edge_crossing", ends[0][0], ends[0][1]))
        crops = []
        for tag, cx, cy in targets:
            z, org = crop_zoom(img, cx, cy)
            crops.append(label(overlay_contour(z, doc, g, org), [f"f{idx} {tag}"]))
        if crops:
            write_png(OUT / f"zoom_f{idx:03d}.png", hstack(crops))


# ── final (default produksi: hide, width 9,0, epsilon 2,8, smooth_px 5,0) ──
FINAL_VIDEOS = (("test", 73, 92), ("test", 225, 240))


def worst_deviation_point(doc: dict, g: sty.Geometry) -> tuple[float, float]:
    """Lokasi (px output) titik kontur dengan deviasi terbesar = takik cekung / sudut tajam terburuk frame ini."""
    best, where = -1.0, (g.out_w / 2, g.out_h / 2)
    for s in doc["strokes"]:
        dev = sm.stroke_deviation(s, g)
        if dev.size and dev.max() > best:
            p = np.asarray(s["points"], float)
            ref = p * g.scale
            ref = ref[[not x for x in sty.edge_sides(p, g.width, g.height)]]
            best, where = float(dev.max()), tuple(ref[int(np.argmax(dev))])
    return where


def cmd_final() -> None:
    clip, style = "test", style_with()
    for idx in COMPARE_FRAMES:
        doc = load_doc(clip, idx)
        img, g, pieces, stats = render_rgb(clip, doc, style)
        write_png(OUT / f"final_f{idx:03d}.png", label(img, [LABEL, f"FINAL {clip} frame {idx}: hide, width 9,0, eps 2,8, smooth 5,0"]))
        targets = [("takik cekung terburuk", *worst_deviation_point(doc, g))]
        sil = next((s for s in doc["strokes"] if s["type"] == "silhouette"), None)
        if sil is not None:
            x, y = sil["points"][0]
            targets.append(("sambungan tertutup (anchor)", x * g.scale, y * g.scale))
        gb = next((s for s in doc["strokes"] if s["type"] == "group_boundary" and len(s["points"]) > 20), None)
        if gb is not None:
            x, y = gb["points"][-1]
            targets.append(("ujung terbuka", x * g.scale, y * g.scale))
        ends = sm_tails(pieces, g)
        if ends:
            targets.append(("titik silang tepi", ends[0][0], ends[0][1]))
        crops = []
        for tag, cx, cy in targets:
            z, org = crop_zoom(img, cx, cy)
            crops.append(label(overlay_contour(z, doc, g, org), [f"f{idx} {tag} 3x"]))
        write_png(OUT / f"final_f{idx:03d}_zoom3x.png", hstack(crops))
    for c, a, b in FINAL_VIDEOS:
        make_video(c, [i for i in range(a, b + 1) if i < frame_count(c)], f"final_{c}_f{a:03d}-{b:03d}.mp4")


# ══ T-401: tebal variabel + taper (keluaran ke work/t401/; semua render LANGSUNG dari contours/, strokes/ klip TIDAK dipakai) ══
OUT401 = ROOT / "work" / "t401"
LABEL401 = "T-401 tebal variabel + taper"
WINDOWS = ((73, 92), (183, 202))                 # jendela gerak cepat klip `test`
BOARD_CLIP = "test"
TYPE_SETS = {"1,0/0,8/0,6/0,5": (1.0, 0.8, 0.6, 0.5), "1,0/0,7/0,7/0,7": (1.0, 0.7, 0.7, 0.7),
             "1,0/0,9/0,5/0,35": (1.0, 0.9, 0.5, 0.35), "semua 1,0": (1.0, 1.0, 1.0, 1.0)}


def pieces_for(clip: str, doc: dict, style, variant: str = "B") -> tuple[sty.Geometry, list[sty.Piece]]:
    """Piece produksi (varian B). Varian A / C = hanya studi: noise 1D berbasis busur dari points[0] (A), atau A untuk strok
    asal-tertutup dan B untuk strok terbuka (C) — tidak ada di kode produksi (keputusan Rio, docs/04)."""
    m = meta(clip)
    g = sty.make_geometry(style, int(m["working_width"]), int(m["working_height"]))
    pieces, _ = sty.frame_pieces(doc, g)
    if variant == "B":
        return g, pieces
    flags = sty.free_ends(pieces, g)
    out = []
    for pc, fl in zip(pieces, flags):
        s = doc["strokes"][pc.stroke_idx]
        orig = np.asarray(s["points"], float) * g.scale
        o_closed = bool(s["closed"])
        if variant == "C" and not o_closed:
            out.append(pc)
            continue
        cum = np.r_[0.0, np.cumsum(np.hypot(*np.diff(np.vstack([orig, orig[:1]]) if o_closed else orig, axis=0).T))]
        L_orig = float(cum[-1])
        pcum = np.r_[0.0, np.cumsum(np.hypot(*np.diff(pc.points, axis=0).T))]
        Lp = float(pcum[-1]) if not pc.closed else float(pcum[-1] + np.hypot(*(pc.points[0] - pc.points[-1])))
        s_arc = pcum
        k0 = int(np.argmin(np.hypot(*(orig - pc.points[0]).T)))
        s_off = 0.0 if pc.closed else float(cum[k0])
        lam = g.noise_cell
        sd = noise.seed_of(0, pc.track_id)
        if pc.closed:
            kk = max(2, int(round(Lp / lam)))
            u, period = s_arc / Lp * kk, kk
        elif o_closed:
            kk = max(2, int(round(L_orig / lam)))
            u, period = ((s_arc + s_off) % L_orig) / L_orig * kk, kk
        else:
            u, period = (s_arc + s_off) / lam, None
        i0 = np.floor(u).astype(np.int64)
        t = noise.fade(u - np.floor(u))
        i1 = i0 + 1
        if period:
            i0, i1 = i0 % period, i1 % period
        zeros = np.zeros_like(i0)
        n = noise.lattice(sd, i0, zeros) * (1 - t) + noise.lattice(sd, i1, zeros) * t
        w = np.full(len(pc.points), g.widths[pc.type]) * (1 + g.variation * n)
        if not pc.closed and (fl[0] or fl[1]):
            w = w * sty.taper_factor(pcum, float(pcum[-1]), fl, g)
        out.append(dataclasses.replace(pc, widths=np.maximum(w, g.floor)))
    return g, out


def rgb_of(pieces: list[sty.Piece], g: sty.Geometry) -> np.ndarray:
    return sm.decode_png(sty.render_png(pieces, g))


def pop_window(clip: str, style, variant: str = "B", windows=WINDOWS) -> dict:
    """Pop tebal (p95 titik bergerak, relatif width_base dan relatif tebal nominal tipe sendiri) + kedalaman taper, jendela cepat."""
    acc, depth = None, {}
    g = None
    for a, b in windows:
        prev = None
        for i in range(a, b + 1):
            doc = load_doc(clip, i)
            g, pcs = pieces_for(clip, doc, style, variant)
            for t, v in sm.taper_depth(pcs, sty.free_ends(pcs, g), g).items():
                depth.setdefault(t, []).append(v)
            if prev is not None:
                r = sm.pop_by_type(prev, pcs, g, style.stroke.width_base * g.unit)
                acc = r if acc is None else {t: {k: acc[t][k] + r[t][k] for k in acc[t]} for t in acc}
            prev = pcs
    s = sm.summarize_pop(acc)
    return {"pop": {t: {"base_move_p95": s[t]["base_move"]["p95"], "own_move_p95": s[t]["own_move"]["p95"],
                        "base_still_p95": s[t]["base_still"]["p95"], "base_move_max": s[t]["base_move"]["max"]} for t in s},
            "taper_depth_median": {t: float(np.median(v)) for t, v in depth.items()}}


def pop_label(res: dict) -> list[str]:
    p, d = res["pop"], res["taper_depth_median"]
    return [f"pop p95 bergerak (x width_base) sil {p['silhouette']['base_move_p95']:.3f} hole {p['silhouette_hole']['base_move_p95']:.3f} "
            f"gb {p['group_boundary']['base_move_p95']:.3f} okl {p['occlusion']['base_move_p95']:.3f}",
            f"pop okl relatif tebal sendiri {p['occlusion']['own_move_p95']:.3f} | kedalaman taper okl {d.get('occlusion', 1.0):.2f} "
            f"gb {d.get('group_boundary', 1.0):.2f} (1,0 = tanpa taper)"]


def board(tag: str, clip: str, frames: tuple[int, ...], configs: list[tuple[str, dict, str]], scale: float = 0.5) -> dict:
    """Papan: tiap konfigurasi (judul, override style, varian) → satu panel per frame; label memuat pop + kedalaman taper."""
    summary = {}
    infos = []
    for title, ov, variant in configs:
        style = style_with(**ov)
        res = pop_window(clip, style, variant)
        summary[title] = {"override": ov, "variant": variant, **res}
        infos.append((title, style, variant, res))
    for idx in frames:
        doc = load_doc(clip, idx)
        panels, crops = [], []
        for title, style, variant, res in infos:
            g, pcs = pieces_for(clip, doc, style, variant)
            img = rgb_of(pcs, g)
            panels.append(label(img, [LABEL401, f"{clip} f{idx} [{variant}] {title}"] + pop_label(res)))
            z, _ = crop_zoom(img, g.out_w * 0.5, g.out_h * 0.38)
            crops.append(label(z, [title]))
        write_png(OUT401 / f"board_{tag}_f{idx:03d}.png", hstack(panels, scale))
        write_png(OUT401 / f"board_{tag}_f{idx:03d}_zoom3x.png", hstack(crops))
    return summary


def cmd_boards() -> None:
    """Papan perbandingan frame 80 dan 233 `test` (semua dengan pop + kedalaman taper di label)."""
    frames = COMPARE_FRAMES
    sm_all = {}
    sm_all["amplitudo"] = board("amplitudo", BOARD_CLIP, frames, [(f"variasi {v}", {"stroke.width_variation": v}, "B") for v in (0.0, 0.15, 0.3, 0.5)])
    sm_all["skala"] = board("skala", BOARD_CLIP, frames, [(f"skala noise x{k} ({0.036 * k:.3f})", {"stroke.width_noise_scale": 0.036 * k}, "B")
                                                          for k in (0.5, 1.0, 2.0)])
    sm_all["taper"] = board("taper", BOARD_CLIP, frames, [("taper mati", {"stroke.taper_ends": False}, "B")]
                            + [(f"taper {px}", {"stroke.taper_px": float(px)}, "B") for px in (20, 45, 70)])
    sm_all["taper_oklusi"] = board("taper_oklusi", BOARD_CLIP, frames,
                                   [("oklusi taper MATI (tipe lain hidup)", {"stroke.by_type.occlusion.taper_ends": False}, "B")]
                                   + [(f"oklusi taper_min {tm}", {"stroke.taper_min": tm}, "B") for tm in (0.15, 0.3, 0.5)])
    sm_all["hierarki"] = board("hierarki", BOARD_CLIP, frames,
                               [(f"tipe {k}", {f"stroke.by_type.{t}.width_scale": v for t, v in zip(sty.TYPE_ORDER, vals)}, "B")
                                for k, vals in TYPE_SETS.items()])
    (OUT401 / "boards_summary.json").write_text(json.dumps(sm_all, indent=1, default=str), encoding="utf-8")
    print(f"  {(OUT401 / 'boards_summary.json').relative_to(ROOT)}")


def cmd_variants() -> None:
    """A vs B vs C berdampingan (frame 80, 233) + zoom 3x pada lengan (oklusi), silhouette, dan lubang; pop per varian di label."""
    sm_v = {}
    style = style_with()
    res = {v: pop_window(BOARD_CLIP, style, v) for v in ("A", "B", "C")}
    sm_v["pop"] = res
    for idx in COMPARE_FRAMES:
        doc = load_doc(BOARD_CLIP, idx)
        panels, zooms = [], {k: [] for k in ("lengan_oklusi", "silhouette", "lubang")}
        for v in ("A", "B", "C"):
            g, pcs = pieces_for(BOARD_CLIP, doc, style, v)
            img = rgb_of(pcs, g)
            panels.append(label(img, [LABEL401, f"{BOARD_CLIP} f{idx} varian {v}"] + pop_label(res[v])))
            occ = [p for p in pcs if p.type == "occlusion"]
            sil = [p for p in pcs if p.type == "silhouette"]
            hol = [p for p in pcs if p.type == "silhouette_hole"]
            for key, grp in (("lengan_oklusi", occ), ("silhouette", sil), ("lubang", hol)):
                if grp:
                    pc = max(grp, key=lambda q: len(q.points))
                    c = pc.points[len(pc.points) // 2]
                    z, _ = crop_zoom(img, c[0], c[1])
                    zooms[key].append(label(z, [f"{key} varian {v} 3x"]))
        write_png(OUT401 / f"variants_f{idx:03d}.png", hstack(panels, 0.4))
        for key, lst in zooms.items():
            if lst:
                write_png(OUT401 / f"variants_f{idx:03d}_{key}_zoom3x.png", hstack(lst))
    (OUT401 / "variants_summary.json").write_text(json.dumps(sm_v, indent=1, default=str), encoding="utf-8")
    print(f"  {(OUT401 / 'variants_summary.json').relative_to(ROOT)}")


class LiveSource(ex.FrameSource):
    """Sumber frame video dari render langsung (tanpa strokes/ klip): satu atau beberapa panel berdampingan."""
    kind = "live"

    def __init__(self, clip: str, panels: list[tuple[str, object, str]], overlay: bool = False):
        self.clip, self.panels, self.overlay = clip, panels, overlay

    def check(self, names: list[str]) -> None:
        return None

    def render(self, name: str) -> np.ndarray:
        idx = int(Path(name).stem.split("_")[1])
        doc = load_doc(self.clip, idx)
        outs = []
        for title, style, variant in self.panels:
            g, pcs = pieces_for(self.clip, doc, style, variant)
            img = rgb_of(pcs, g)
            if self.overlay:
                prev_doc = load_doc(self.clip, max(idx - 1, 0))
                _, prev = pieces_for(self.clip, prev_doc, style, variant)
                img = change_map(img, prev, pcs, g, style.stroke.width_base * g.unit)
            outs.append(label(img, [LABEL401, f"{self.clip} f{idx:03d} [{variant}] {title}"]))
        return hstack(outs, 0.5) if len(outs) > 1 else outs[0]


CHANGE_THRESHOLD = 0.15                         # |Δtebal| / width_base di atas ini = "berubah" (peta tebal berubah)


def change_map(img: np.ndarray, prev: list[sty.Piece], cur: list[sty.Piece], g: sty.Geometry, base: float) -> np.ndarray:
    """Titik yang tebalnya berubah > CHANGE_THRESHOLD antar frame ditandai (merah = berubah, hijau muda = tetap) di atas gambar."""
    out = (img.astype(np.float32) * 0.45 + 255 * 0.55).astype(np.uint8)
    by_prev: dict = {}
    for pc in prev:
        by_prev.setdefault((pc.track_id, pc.type), []).append(pc)
    for pc in cur:
        olds = by_prev.get((pc.track_id, pc.type))
        if not olds:
            continue
        q = np.vstack([o.points for o in olds])
        qw = np.concatenate([sty.piece_widths(o, g) for o in olds])
        d, j = sm.cKDTree(q).query(pc.points)
        w = sty.piece_widths(pc, g)
        for k in range(0, len(pc.points), 2):
            if d[k] > 3 * g.unit:
                continue
            bad = abs(w[k] - qw[j[k]]) / base > CHANGE_THRESHOLD
            cv2.circle(out, (int(pc.points[k, 0]), int(pc.points[k, 1])), 5 if bad else 2, (220, 30, 30) if bad else (120, 200, 140), -1)
    return out


def make_live_video(clip: str, frames: list[int], name: str, panels, overlay: bool = False) -> None:
    src = LiveSource(clip, panels, overlay)
    m = meta(clip)
    first = src.render(f"frame_{frames[0]:05d}.png")        # ukuran encode diukur dari frame nyata (hstack skala 0,5 + jarak)
    h, w = first.shape[:2]
    h, w = h + h % 2, w + w % 2                              # yuv420p butuh genap (encode memad 1 px)
    bg = np.array(ex.parse_hex(style_with().paper.color), np.uint8)
    OUT401.mkdir(parents=True, exist_ok=True)
    tmp, final = OUT401 / (name + ".tmp"), OUT401 / name
    ex.encode(src, [f"frame_{i:05d}.png" for i in frames], (w, h), bg, float(m["target_fps"]), VIDEO_CRF, VIDEO_PRESET, None, tmp)
    os.replace(tmp, final)
    print(f"  {final.relative_to(ROOT)} ({final.stat().st_size / 1024:.0f} KiB, {len(frames)} frame)")


def cmd_videos401() -> None:
    """Video jendela gerak cepat (73-92, 183-202): skala noise x1 vs x2; varian B vs A; taper oklusi mati vs hidup; peta tebal berubah."""
    s1, s2 = style_with(), style_with(**{"stroke.width_noise_scale": 0.072})
    s_occ_off = style_with(**{"stroke.by_type.occlusion.taper_ends": False})
    for a, b in WINDOWS:
        fr = list(range(a, b + 1))
        tag = f"f{a:03d}-{b:03d}"
        make_live_video(BOARD_CLIP, fr, f"skala_x1_vs_x2_{tag}.mp4", [("skala x1 (0,036)", s1, "B"), ("skala x2 (0,072)", s2, "B")])
        make_live_video(BOARD_CLIP, fr, f"varianB_vs_varianA_{tag}.mp4", [("B terkunci posisi (terpilih)", s1, "B"), ("A busur dari points[0]", s1, "A")])
        make_live_video(BOARD_CLIP, fr, f"taper_oklusi_hidup_vs_mati_{tag}.mp4", [("taper oklusi hidup", s1, "B"), ("taper oklusi MATI", s_occ_off, "B")])
        make_live_video(BOARD_CLIP, fr[1:], f"peta_tebal_berubah_x1_{tag}.mp4", [("peta tebal berubah x1", s1, "B")], overlay=True)
        make_live_video(BOARD_CLIP, fr[1:], f"peta_tebal_berubah_x2_{tag}.mp4", [("peta tebal berubah x2", s2, "B")], overlay=True)


def cmd_changemap() -> None:
    """PNG peta tebal berubah untuk beberapa frame jendela cepat (x1 dan x2 berdampingan)."""
    s1, s2 = style_with(), style_with(**{"stroke.width_noise_scale": 0.072})
    for idx in (78, 87, 190, 195):
        doc, prev_doc = load_doc(BOARD_CLIP, idx), load_doc(BOARD_CLIP, idx - 1)
        panels = []
        for title, st in (("skala x1", s1), ("skala x2", s2)):
            g, cur = pieces_for(BOARD_CLIP, doc, st)
            _, prev = pieces_for(BOARD_CLIP, prev_doc, st)
            panels.append(label(change_map(rgb_of(cur, g), prev, cur, g, st.stroke.width_base * g.unit),
                                [LABEL401, f"{BOARD_CLIP} f{idx} peta tebal berubah > {CHANGE_THRESHOLD} x width_base (merah) {title}"]))
        write_png(OUT401 / f"changemap_f{idx:03d}.png", hstack(panels, 0.5))


def worst401_scan(clip: str, style) -> dict:
    """Satu pass atas semua frame: kasus terburuk T-401 (pop terbesar, strok terpendek ber-taper-clamp, ujung bertemu terdekat ke
    ambang, ujung di tepi dengan kontak paling dangkal, tikungan rapat, sambungan tertutup, tebal minimum)."""
    m = meta(clip)
    n = frame_count(clip)
    g = sty.make_geometry(style, int(m["working_width"]), int(m["working_height"]))
    base = style.stroke.width_base * g.unit
    best = {"pop_terbesar": (-1.0, None), "terpendek_clamp": (1e9, None), "bertemu_dekat_ambang": (1e9, None),
            "tepi_dangkal": (999.0, None), "tikungan_rapat": (1e9, None), "sambungan_tertutup": (-1.0, None), "tebal_minimum": (1e9, None)}
    prev = None
    for i in range(n):
        d = load_doc(clip, i)
        pcs, stats = sty.frame_pieces(d, g)
        flags = sty.free_ends(pcs, g)
        if stats["min_contact_deg"] < best["tepi_dangkal"][0]:
            best["tepi_dangkal"] = (stats["min_contact_deg"], (i, None))
        if prev is not None:
            acc = sm.pop_by_type(prev, pcs, g, base)
            for t, kv in acc.items():
                for arr in kv["base_move"]:
                    if len(arr) and float(arr.max()) > best["pop_terbesar"][0]:
                        best["pop_terbesar"] = (float(arr.max()), (i, t))
        prev = pcs
        for pc, fl in zip(pcs, flags):
            w = sty.piece_widths(pc, g)
            L = float(np.hypot(*np.diff(pc.points, axis=0).T).sum())
            if (fl[0] or fl[1]) and L < 2 * g.taper_len and L < best["terpendek_clamp"][0]:
                best["terpendek_clamp"] = (L, (i, pc.stroke_idx))
            if float(w.min()) < best["tebal_minimum"][0]:
                best["tebal_minimum"] = (float(w.min()), (i, pc.stroke_idx))
            if pc.closed:
                jump, step = sm.seam_jump(pc, g)
                if jump / max(step, 1e-9) > best["sambungan_tertutup"][0]:
                    best["sambungan_tertutup"] = (jump / max(step, 1e-9), (i, pc.stroke_idx))
            if len(pc.points) > 6:
                p = pc.points
                a, b, c = (p[:-2], p[1:-1], p[2:]) if not pc.closed else (np.roll(p, 1, 0), p, np.roll(p, -1, 0))
                ab, bc, ca = (np.hypot(*(b - a).T), np.hypot(*(c - b).T), np.hypot(*(a - c).T))
                cross = np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
                with np.errstate(divide="ignore", invalid="ignore"):
                    r = ab * bc * ca / (2 * cross)
                ww = w[1:-1] if not pc.closed else w
                ratio = np.where(np.isfinite(r), r / np.maximum(ww / 2, 1e-9), np.inf)
                if float(ratio.min()) < best["tikungan_rapat"][0]:
                    best["tikungan_rapat"] = (float(ratio.min()), (i, pc.stroke_idx, p[1:-1][int(np.argmin(ratio))].tolist() if not pc.closed else p[int(np.argmin(ratio))].tolist()))
        # ujung bertemu terdekat ke ambang (JOIN 8): jarak ujung non-bebas terbesar yang masih "bertemu"
        for pc, fl in zip(pcs, flags):
            if pc.closed:
                continue
            for e in (0, 1):
                if pc.edge[e] or fl[e] or not g.taper_on[pc.type]:
                    continue
                end = pc.points[0 if e == 0 else -1]
                dmin = min((float(sm.point_polyline_distance(end[None, :], sm.piece_curve(q))[0]) for q in pcs if q.stroke_idx != pc.stroke_idx),
                           default=1e9)
                gap = abs(g.join_dist - dmin)
                if gap < best["bertemu_dekat_ambang"][0]:
                    best["bertemu_dekat_ambang"] = (gap, (i, pc.stroke_idx, end.tolist(), dmin))
    return best


def cmd_worst401() -> None:
    style = style_with()
    summary = {}
    for clip in CLIP_NAMES:
        best = worst401_scan(clip, style)
        summary[clip] = {k: {"value": v[0], "where": v[1]} for k, v in best.items()}
        for tag, (val, where) in best.items():
            if where is None:
                continue
            i = where[0]
            doc = load_doc(clip, i)
            g, pcs = pieces_for(clip, doc, style)
            img = rgb_of(pcs, g)
            cx, cy = g.out_w / 2, g.out_h / 2
            if tag in ("tikungan_rapat", "bertemu_dekat_ambang") and len(where) > 2:
                cx, cy = where[2]
            elif where[1] is not None and tag != "pop_terbesar":
                cand = [p for p in pcs if p.stroke_idx == where[1]]
                if cand:
                    c = cand[0].points[len(cand[0].points) // 2]
                    cx, cy = float(c[0]), float(c[1])
            elif tag == "pop_terbesar" and where[1]:
                cand = [p for p in pcs if p.type == where[1]]
                if cand:
                    c = max(cand, key=lambda q: len(q.points)).points[0]
                    cx, cy = float(c[0]), float(c[1])
            write_png(OUT401 / f"worst401_{clip}_{tag}_f{i:03d}.png", label(img, [LABEL401, f"{clip} f{i} terburuk: {tag} = {val:.3f}"]))
            z, _ = crop_zoom(img, cx, cy)
            write_png(OUT401 / f"worst401_{clip}_{tag}_f{i:03d}_zoom3x.png", label(z, [f"{tag} 3x"]))
    (OUT401 / "worst401_summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    print(f"  {(OUT401 / 'worst401_summary.json').relative_to(ROOT)}")


def cmd_svgsample() -> None:
    """SVG sampel (frame 80 `test`) + pemeriksaan struktur (XML well-formed, viewBox, jumlah path / sub-path, ukuran)."""
    import xml.etree.ElementTree as ET
    style = style_with()
    doc = load_doc(BOARD_CLIP, 80)
    g = sty.make_geometry(style, int(meta(BOARD_CLIP)["working_width"]), int(meta(BOARD_CLIP)["working_height"]))
    svg, png, stats = sty.render_frame(doc, g, style)
    OUT401.mkdir(parents=True, exist_ok=True)
    p = OUT401 / "after_t401_test_frame_00080.svg"
    p.write_bytes(svg)
    root = ET.fromstring(svg)
    paths = list(root.iter("{http://www.w3.org/2000/svg}path"))
    info = {"bytes": len(svg), "viewBox": root.get("viewBox"), "paths": len(paths),
            "subpaths": sum(d.get("d").count("M") for d in paths), "png_bytes": len(png)}
    (OUT401 / "after_t401_svg_info.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print(f"  {p.relative_to(ROOT)} {info}")


# ── T-402: jitter (video papan + gambar bantu; keluaran ke work/t402/; klip dari --clips-root, default work/clips) ─────
# Python 3.11, tanpa GPU. Semua pada SALINAN klip (scratchpad); strokes/ dan out/ klip tidak dipakai (render langsung dari contours/).
OUT402 = ROOT / "work" / "t402"
LABEL402 = "T-402 jitter"
EVAL402 = {"jitter.amplitude": 4.0, "jitter.frequency": 0.053, "jitter.temporal_drift": 0.35, "jitter.hold_frames": 2,
           "jitter.stroke_independence": 0.0, "jitter.temporal_seed_mode": "frame"}     # nilai evaluasi awal (HANYA papan; bukan default)
BASELINE402 = {"jitter.amplitude": 0.0}                                                  # T-401 tanpa jitter (byte-identik)
STATIC_SPEED_PX, STATIC_XOR, STATIC_MIN_FRAMES, WINDOW_LEN = 2.0, 0.04, 10, 20           # aturan jendela statis T-302
ZOOM_HALF402 = 90


def style402(**ov):
    return style_with(**{**EVAL402, **ov})


def r_of(style) -> float:
    j = style.jitter
    return sty.jitter_fold_r(j.amplitude, j.frequency, j.stroke_independence)


def param_line(style, baseline: bool = False) -> str:
    j = style.jitter
    if baseline:
        return "T-401 tanpa jitter (amplitudo 0)"
    return (f"amp {j.amplitude:g} freq {j.frequency:g} drift {j.temporal_drift:g} hold {j.hold_frames} s {j.stroke_independence:g} "
            f"mode {j.temporal_seed_mode} | r = {r_of(style):.3f}")


def find_static_window(clip: str) -> tuple[int, int]:
    """Jendela statis OTOMATIS (aturan T-302): ≥ 10 frame berurutan, kecepatan centroid ≤ 2 px/frame dan XOR foreground ≤ 4%
    (foreground = isi siluet dari contours/; piksel kerja). Terpanjang dipilih; panjang jendela dibatasi 20."""
    n = frame_count(clip)
    m = meta(clip)
    h, w = int(m["working_height"]), int(m["working_width"])
    masks, cents = [], []
    for i in range(n):
        d = load_doc(clip, i)
        mk = np.zeros((h, w), np.uint8)
        for s in d["strokes"]:
            if s["type"] == "silhouette":
                cv2.fillPoly(mk, [np.rint(np.asarray(s["points"], float) - 0.5).astype(np.int32)], 1)
        for s in d["strokes"]:
            if s["type"] == "silhouette_hole":
                cv2.fillPoly(mk, [np.rint(np.asarray(s["points"], float) - 0.5).astype(np.int32)], 0)
        mo = cv2.moments(mk, binaryImage=True)
        masks.append(mk)
        cents.append(np.array([mo["m10"], mo["m01"]]) / max(mo["m00"], 1.0))
    ok = [float(np.hypot(*(cents[i] - cents[i - 1]))) <= STATIC_SPEED_PX and
          float((masks[i] ^ masks[i - 1]).sum()) / max(float(masks[i].sum()), 1.0) <= STATIC_XOR for i in range(1, n)]
    best, cur = (0, 0), 0
    for i, v in enumerate(ok):
        cur = cur + 1 if v else 0
        if cur + 1 > best[1] - best[0] + 1 and cur + 1 >= STATIC_MIN_FRAMES:
            best = (i + 1 - cur, i + 1)
    if best == (0, 0):
        raise SystemExit(f"jendela statis tidak ditemukan di {clip} (aturan T-302)")
    return best[0], min(best[1], best[0] + WINDOW_LEN - 1)


def mark_new_ids(img: np.ndarray, pieces: list[sty.Piece], g: sty.Geometry, prev_ids: set) -> tuple[np.ndarray, int]:
    """Strok yang (type, track_id)-nya baru dibanding frame sebelumnya diwarnai merah (peta loncatan G saat s > 0)."""
    new = [pc for pc in pieces if (pc.type, pc.track_id) not in prev_ids]
    if not new:
        return img, 0
    cov = cv2.resize(sty.render_mask(new, g), (g.out_w, g.out_h), interpolation=cv2.INTER_AREA).astype(np.float32)[..., None] / 255.0
    red = np.array([230, 30, 30], np.float32)
    return (img * (1 - cov) + red * cov).astype(np.uint8), len(new)


class JitterSource(ex.FrameSource):
    """Frame video T-402: panel pertama = T-401 tanpa jitter (di-cache), panel berikutnya = varian jitter; berdampingan, ukuran penuh."""
    kind = "live"

    def __init__(self, clip: str, panels: list[tuple[str, object]], mark_ids: bool = False):
        self.clip, self.panels, self.mark_ids = clip, panels, mark_ids
        self.cache: dict = {}

    def check(self, names: list[str]) -> None:
        return None

    def panel_img(self, idx: int, title: str, style, doc: dict, baseline: bool) -> np.ndarray:
        key = (idx, baseline)
        if baseline and key in self.cache:
            img = self.cache[key]
        else:
            g, pcs = pieces_for(self.clip, doc, style, "B")
            img = rgb_of(pcs, g)
            if baseline:
                self.cache[key] = img
        lines = [f"{LABEL402} | {self.clip} f{idx:03d} | {title}", param_line(style, baseline)]
        if not baseline and self.mark_ids and style.jitter.stroke_independence > 0 and idx > 0:
            g, pcs = pieces_for(self.clip, doc, style, "B")
            prev = {(s["type"], s["track_id"]) for s in load_doc(self.clip, idx - 1)["strokes"]}
            img, k = mark_new_ids(img, pcs, g, prev)
            lines.append(f"merah = strok dengan track_id BARU di frame ini ({k} dari {len(pcs)} jalur): pola G loncat")
        return label(img, lines)

    def render(self, name: str) -> np.ndarray:
        idx = int(Path(name).stem.split("_")[1])
        doc = load_doc(self.clip, idx)
        outs = [self.panel_img(idx, t, s, doc, baseline=(k == 0)) for k, (t, s) in enumerate(self.panels)]
        return hstack(outs, 1.0)


def make_video402(clip: str, frames: list[int], name: str, panels, mark_ids: bool = False) -> None:
    src = JitterSource(clip, panels, mark_ids)
    first = src.render(f"frame_{frames[0]:05d}.png")
    h, w = first.shape[:2]
    h, w = h + h % 2, w + w % 2
    bg = np.array(ex.parse_hex(style_with().paper.color), np.uint8)
    OUT402.mkdir(parents=True, exist_ok=True)
    tmp, final = OUT402 / (name + ".tmp"), OUT402 / name
    ex.encode(src, [f"frame_{i:05d}.png" for i in frames], (w, h), bg, float(meta(clip)["target_fps"]), VIDEO_CRF, VIDEO_PRESET, None, tmp)
    os.replace(tmp, final)
    print(f"  {final.relative_to(ROOT)} ({final.stat().st_size / 1024:.0f} KiB, {len(frames)} frame, {w}x{h})", flush=True)


def video_sets402() -> dict[str, list[tuple[str, dict]]]:
    """Papan keputusan Rio (item 6): amplitudo dipasangkan dengan frequency yang menjaga r ≤ 0,21; satu panel LIPATAN (informasi)."""
    amp = [(f"amplitudo {a:g} @ freq {f:g}", {"jitter.amplitude": a, "jitter.frequency": f})
           for a, f in ((2, 0.053), (4, 0.053), (6, 0.035), (8, 0.0265))]
    amp.append((f"LIPATAN (informasi): r = {8 * 0.053:.3f} amplitudo 8 @ freq 0.053", {"jitter.amplitude": 8.0, "jitter.frequency": 0.053}))
    freq = [("frequency x0.5 (0.0265) amp 4", {"jitter.frequency": 0.0265, "jitter.amplitude": 4.0}),
            ("frequency x1 (0.053) amp 4", {"jitter.frequency": 0.053, "jitter.amplitude": 4.0}),
            ("frequency x2 (0.106) amp 2", {"jitter.frequency": 0.106, "jitter.amplitude": 2.0})]
    hold = [(f"hold {h} x drift {d:g}", {"jitter.hold_frames": h, "jitter.temporal_drift": d}) for h in (1, 2, 3) for d in (0.15, 0.35, 1.0)]
    ind = [(f"stroke_independence {s:g}", {"jitter.stroke_independence": s, "jitter.frequency": 0.037}) for s in (0.0, 0.25, 0.5, 1.0)]
    return {"amplitudo": amp, "frequency": freq, "hold_drift": hold, "independence": ind}


def slug(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in text.replace(".", "p")).strip("_")[:60]


def cmd_videos402() -> None:
    """Video papan: tiap video = [T-401 tanpa jitter | varian] ukuran penuh, jendela statis otomatis + gerak cepat 73-92 / 183-202.
    Plus panel `temporal_seed_mode: fixed` pada jendela cepat (memisahkan efek GERAK dari efek WAKTU)."""
    static = find_static_window(BOARD_CLIP)
    windows = {"statis": static, "cepat1": WINDOWS[0], "cepat2": WINDOWS[1]}
    OUT402.mkdir(parents=True, exist_ok=True)
    (OUT402 / "windows.json").write_text(json.dumps(windows, indent=1), encoding="utf-8")
    print(f"  jendela: {windows}")
    base = style402(**BASELINE402)
    for wname, (a, b) in windows.items():
        fr = list(range(a, b + 1))
        for group, items in video_sets402().items():
            for title, ov in items:
                st = style402(**ov)
                make_video402(BOARD_CLIP, fr, f"v402_{group}_{slug(title)}_{wname}.mp4",
                              [("T-401", base), (title, st)], mark_ids=(group == "independence"))
        if wname != "statis":
            make_video402(BOARD_CLIP, fr, f"v402_fixed_vs_frame_{wname}.mp4",
                          [("T-401", base), ("mode frame (hold 2)", style402()), ("mode FIXED (efek gerak saja)", style402(**{"jitter.temporal_seed_mode": "fixed"}))])


def overlay3(covs: list[np.ndarray]) -> np.ndarray:
    """Tiga cakupan tinta (0–1) ditumpuk subtraktif: frame 1 = cyan, 2 = magenta, 3 = kuning (tumpang tindih = gelap)."""
    out = np.full(covs[0].shape + (3,), 255.0)
    for cov, tint in zip(covs, ((0, 255, 255), (255, 0, 255), (255, 255, 0))):
        out -= cov[..., None] * np.array(tint)
    return np.clip(out, 0, 255).astype(np.uint8)


def cov_of(clip: str, idx: int, style) -> tuple[np.ndarray, sty.Geometry, list[sty.Piece]]:
    g, pcs = pieces_for(clip, load_doc(clip, idx), style, "B")
    return cv2.resize(sty.render_mask(pcs, g), (g.out_w, g.out_h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0, g, pcs


def body_center(clip: str, idx: int) -> tuple[float, float]:
    g = sty.make_geometry(style402(), int(meta(clip)["working_width"]), int(meta(clip)["working_height"]))
    pts = np.vstack([np.asarray(s["points"], float) for s in load_doc(clip, idx)["strokes"] if s["type"] == "silhouette"]) * g.scale
    return float(pts[:, 0].mean()), float(pts[:, 1].mean())


def crop_panels(images: list[np.ndarray], titles: list[str], cx: float, cy: float, half: int = ZOOM_HALF402, zoom: int = ZOOM) -> np.ndarray:
    crops = []
    for img, t in zip(images, titles):
        z, _ = crop_zoom(img, cx, cy, half, zoom)
        crops.append(label(z, [t]))
    return hstack(crops)


def cmd_helpers402() -> None:
    """Gambar bantu: strip 3 frame ditumpuk (besar getar), zoom 3x sambungan T / seam tertutup / garis sejajar dekat / tepi bawah + kanan /
    tikungan rapat — masing-masing [T-401 | koheren s = 0 | independen s = 1] atau [T-401 | amplitudo]."""
    clip = BOARD_CLIP
    a, b = find_static_window(clip)
    base = style402(**BASELINE402)
    # 1. strip 3 frame berurutan (hold 1) per amplitudo: besar getar pada jendela statis
    cx, cy = body_center(clip, a)
    strips = []
    for amp, f in ((2, 0.053), (4, 0.053), (6, 0.035), (8, 0.0265)):
        st = style402(**{"jitter.amplitude": float(amp), "jitter.frequency": f, "jitter.hold_frames": 1})
        covs = [cov_of(clip, a + k, st)[0] for k in range(3)]
        z, _ = crop_zoom(overlay3(covs), cx, cy, 130, 2)
        strips.append(label(z, [f"{LABEL402} strip 3 frame (f{a}-{a + 2}, jendela statis): cyan / magenta / kuning", param_line(st)]))
    write_png(OUT402 / "helper_strip_3frame.png", hstack(strips))
    # 2-7. kasus nyata pada frame 80 / 233: [T-401 | s=0 | s=1]
    s0 = style402(**{"jitter.amplitude": 6.0, "jitter.frequency": 0.035, "jitter.hold_frames": 1})
    s1 = style402(**{"jitter.amplitude": 6.0, "jitter.frequency": 0.035, "jitter.hold_frames": 1, "jitter.stroke_independence": 1.0})
    trio = [("T-401", base), ("koheren s = 0", s0), ("INDEPENDEN s = 1", s1)]

    def trio_png(name: str, idx: int, cx: float, cy: float, note: str, half: int = ZOOM_HALF402) -> None:
        imgs = [rgb_of(pieces_for(clip, load_doc(clip, idx), st, "B")[1], pieces_for(clip, load_doc(clip, idx), st, "B")[0]) for _, st in trio]
        write_png(OUT402 / f"helper_{name}_f{idx:03d}.png", crop_panels(imgs, [f"{t} | {note} f{idx}" for t, _ in trio], cx, cy, half))
    g = sty.make_geometry(s0, int(meta(clip)["working_width"]), int(meta(clip)["working_height"]))
    for idx in (80, 183):
        base_p, _ = sty.base_pieces(load_doc(clip, idx), g)
        ends = jm.joint_ends(base_p, g)
        if ends:
            i, e = ends[len(ends) // 2]
            q = base_p[i].points[0 if e == 0 else -1]
            trio_png("sambungan_T", idx, q[0], q[1], "sambungan T (silhouette - batas grup)")
        closed = [p for p in base_p if p.closed]            # siluet terpotong tepi menjadi terbuka (hide): lubang / batas grup tertutup
        if closed:
            pc = max(closed, key=lambda p: len(p.points))
            trio_png("seam_tertutup", idx, pc.points[0][0], pc.points[0][1], "seam strok tertutup (points[0])")
        # tikungan rapat: radius lokal terkecil
        best = None
        for pc in base_p:
            if len(pc.points) < 12:
                continue
            p = pc.points
            t = np.arctan2(*np.diff(p, axis=0).T[::-1])
            dth = np.abs(np.angle(np.exp(1j * np.diff(t))))
            ds = np.hypot(*np.diff(p, axis=0).T)[1:]
            rad = ds / np.maximum(dth, 1e-6)
            k = int(np.argmin(rad))
            if best is None or rad[k] < best[0]:
                best = (float(rad[k]), p[k + 1])
        if best:
            trio_png("tikungan_rapat", idx, best[1][0], best[1][1], f"tikungan rapat (radius {best[0]:.1f} px)")
        # garis sejajar dekat: dua strok berbeda berjarak < 2 x amplitudo (menjauhi ujung)
        pts, own = [], []
        for pc in base_p:
            q = pc.points
            far = np.ones(len(q), bool)
            if not pc.closed:
                far = np.minimum(np.hypot(*(q - q[0]).T), np.hypot(*(q - q[-1]).T)) > 12.0
            pts.append(q[far])
            own.append(np.full(int(far.sum()), pc.stroke_idx))
        pts, own = np.vstack(pts), np.concatenate(own)
        pairs = sm.cKDTree(pts).query_pairs(2 * s0.jitter.amplitude * g.unit, output_type="ndarray")
        pairs = pairs[own[pairs[:, 0]] != own[pairs[:, 1]]] if len(pairs) else pairs
        if len(pairs):
            dd = np.hypot(*(pts[pairs[:, 0]] - pts[pairs[:, 1]]).T)
            c = (pts[pairs[int(np.argmin(dd))][0]] + pts[pairs[int(np.argmin(dd))][1]]) / 2
            trio_png("garis_sejajar_dekat", idx, c[0], c[1], f"garis berdekatan (min {dd.min():.1f} px)")
    # tepi bawah (frame 233) + tepi kanan (frame dengan ujung ekstensi kanan terbanyak): [T-401 | amplitudo 8 | amplitudo 8 tanpa pelunakan tepi dihitung = T-401 vs jitter]
    st8 = style402(**{"jitter.amplitude": 8.0, "jitter.frequency": 0.0265, "jitter.hold_frames": 1})
    gg = sty.make_geometry(st8, int(meta(clip)["working_width"]), int(meta(clip)["working_height"]))
    best_r, right_f = -1, 80
    for i in range(0, frame_count(clip), 3):
        bp, _ = sty.base_pieces(load_doc(clip, i), gg)
        k = sum(1 for p in bp for e in (0, 1) if p.edge[e] and p.points[0 if e == 0 else -1][0] > gg.out_w)
        if k > best_r:
            best_r, right_f = k, i
    for idx, cx, cy, note in ((233, gg.out_w * 0.5, gg.out_h - 150, "tepi bawah"), (right_f, gg.out_w - 120, gg.out_h * 0.45, f"tepi kanan ({best_r} ujung)")):
        imgs = [rgb_of(pieces_for(clip, load_doc(clip, idx), st, "B")[1], gg) for st in (base, st8)]
        write_png(OUT402 / f"helper_tepi_{note.split()[0]}_{note.split()[1] if len(note.split()) > 1 else ''}_f{idx:03d}.png".replace("__", "_"),
                  crop_panels(imgs, [f"T-401 | {note} f{idx}", f"amplitudo 8 @ 0.0265 | {note} f{idx} (ujung tepi tidak berubah)"], cx, cy, 220, 2))


def worst402_scan(clip: str, style, every: int = 3) -> dict:
    """Kasus terburuk (informasi, dikirim SEBELUM Rio menilai): perubahan sambungan terbesar, min det Jacobian, ujung tepi terdekat, persilangan."""
    n = frame_count(clip)
    m = meta(clip)
    g = sty.make_geometry(style, int(m["working_width"]), int(m["working_height"]))
    worst = {"joint_change": (0.0, None, None), "min_det": (9.0, None, None), "intrusion": (-9e9, None, None), "new_cross": (0, None, None)}
    for i in range(0, n, every):
        doc = load_doc(clip, i)
        base, _ = sty.base_pieces(doc, g)
        jit = sty.jitter_pieces(base, g, int(doc["frame_index"]))
        for ii, e in jm.joint_ends(base, g):
            d = abs(jm.end_gap(jit, ii, e) - jm.end_gap(base, ii, e))
            if d > worst["joint_change"][0]:
                worst["joint_change"] = (d, i, base[ii].points[0 if e == 0 else -1])
        for pc in jit:
            dets = jm.jacobian_dets(g, int(doc["frame_index"]), pc.points, pc.track_id)
            k = int(np.argmin(dets))
            if dets[k] < worst["min_det"][0]:
                worst["min_det"] = (float(dets[k]), i, pc.points[k])
        for pc in jit:
            for e in (0, 1):
                if pc.edge[e] and not pc.closed and pc.widths is not None:
                    q = pc.points[0 if e == 0 else -1]
                    v = float(pc.widths[0 if e == 0 else -1]) / 2 - jm.rect_distance(q, g)
                    if v > worst["intrusion"][0]:
                        worst["intrusion"] = (v, i, q)
        nc = jm.new_crossings(base, jit)
        if nc > worst["new_cross"][0]:
            cr = [p for a, b, p in jm.crossings(jit)]
            worst["new_cross"] = (nc, i, cr[0] if cr else None)
    return worst


def cmd_worst402() -> None:
    clip = BOARD_CLIP
    base = style402(**BASELINE402)
    summary = {}
    for tag, ov in (("koheren_amp4", {"jitter.amplitude": 4.0, "jitter.hold_frames": 1}),
                    ("independen_amp4", {"jitter.amplitude": 4.0, "jitter.hold_frames": 1, "jitter.stroke_independence": 1.0}),
                    ("koheren_amp8_f0.0265", {"jitter.amplitude": 8.0, "jitter.frequency": 0.0265, "jitter.hold_frames": 1})):
        st = style402(**ov)
        w = worst402_scan(clip, st)
        summary[tag] = {k: {"value": float(v[0]), "frame": v[1], "point": None if v[2] is None else [float(v[2][0]), float(v[2][1])]}
                        for k, v in w.items()}
        for k, (val, idx, pt) in w.items():
            if idx is None or pt is None:
                continue
            imgs = [rgb_of(pieces_for(clip, load_doc(clip, idx), s, "B")[1], sty.make_geometry(s, 480, 854)) for s in (base, st)]
            write_png(OUT402 / f"worst402_{tag}_{k}_f{idx:03d}.png",
                      crop_panels(imgs, [f"T-401 | {tag} {k} = {val:.3f} f{idx}", f"{param_line(st)}"], pt[0], pt[1]))
    (OUT402 / "worst402_summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    print(f"  {(OUT402 / 'worst402_summary.json').relative_to(ROOT)}")


T402_COMMANDS = ("videos402", "helpers402", "worst402")


COMMANDS = {"final": cmd_final, "videos": cmd_videos, "edge": cmd_edge, "widths": cmd_widths, "epsilon": cmd_epsilon, "worst": cmd_worst,
            "zoom": cmd_zoom, "boards": cmd_boards, "variants": cmd_variants, "videos401": cmd_videos401, "changemap": cmd_changemap,
            "worst401": cmd_worst401, "svgsample": cmd_svgsample, "videos402": cmd_videos402, "helpers402": cmd_helpers402,
            "worst402": cmd_worst402}
T203_ONLY = ("final", "videos", "edge", "widths", "epsilon", "worst", "zoom")          # alat T-203a (strokes/ klip, work/t203a/)
T401_COMMANDS = ("boards", "variants", "videos401", "changemap", "worst401", "svgsample")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("what", choices=[*COMMANDS, "all", "all401", "all402"])
    p.add_argument("--clips-root", type=Path, default=None,
                   help="folder berisi <klip>/{meta.json,contours/} (default work/clips); T-402: SALINAN scratchpad")
    args = p.parse_args(argv)
    if args.clips_root is not None:
        global CLIPS
        CLIPS = args.clips_root
    if args.what in T402_COMMANDS or args.what == "all402":
        OUT402.mkdir(parents=True, exist_ok=True)
    else:
        OUT.mkdir(parents=True, exist_ok=True)
        OUT401.mkdir(parents=True, exist_ok=True)
    names = (list(COMMANDS) if args.what == "all" else list(T401_COMMANDS) if args.what == "all401"
             else list(T402_COMMANDS) if args.what == "all402" else [args.what])
    for name in names:
        print(f"== {name}")
        COMMANDS[name]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
