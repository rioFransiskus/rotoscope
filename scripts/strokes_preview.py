"""Alat pratinjau sekali pakai T-203a (BUKAN bagian src/): keluaran visual stage [5] ke work/t203a/ (ter-ignore).

    python scripts/strokes_preview.py videos      # video H.264 PNG hasil stage (rentang 73-92, 183-202, 225-240 + ...)
    python scripts/strokes_preview.py edge        # hide vs draw (frame 233 + frame tepi kanan terbanyak)
    python scripts/strokes_preview.py widths      # width_base {3.6, 5.4, 7.2, 9.0} px ref, frame 80 dan 233
    python scripts/strokes_preview.py epsilon     # simplify_epsilon {2.8, 5.6} px ref, frame 80 dan 233
    python scripts/strokes_preview.py worst       # PNG kasus terburuk (deviasi, terpanjang, run tepi, strok terbanyak,
                                                  # kontak paling dangkal, sudut kanvas) + crop zoom 3x
    python scripts/strokes_preview.py zoom        # crop zoom 3x: sambungan strok tertutup, ujung terbuka, tepi bawah
    python scripts/strokes_preview.py all

Membaca contours/ + strokes/ klip (tidak mengubahnya). Label: "garis polos T-203a - belum ada jitter / taper / tekstur".
Fungsi metrik dipakai ulang dari tests/stylize_metrics.py; encode dipakai ulang dari rotoscope.export.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

import stylize_metrics as sm  # noqa: E402

from rotoscope import export as ex  # noqa: E402
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


COMMANDS = {"final": cmd_final, "videos": cmd_videos, "edge": cmd_edge, "widths": cmd_widths, "epsilon": cmd_epsilon, "worst": cmd_worst,
            "zoom": cmd_zoom}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("what", choices=[*COMMANDS, "all"])
    args = p.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    for name in (COMMANDS if args.what == "all" else [args.what]):
        print(f"== {name}")
        COMMANDS[name]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
