"""A/B test backend segmentasi T-102a (D-008): MediaPipe vs MediaPipe+pad vs rembg u2net_human_seg.

Alat sekali pakai, BUKAN modul pipeline. Jalankan dari root repo setelah T-101 (ingest):
    venv/Scripts/python.exe scripts/ab_segment.py [--threshold 0.5] [--force]

Input : work/meta.json + work/frames/frame_%05d.png
Output: work/ab_t102a/ -> masks_<backend>/, infer_<backend>.json, side_by_side.mp4,
        disagreement_top12.png, metrics_per_frame.csv, metrics_summary.json

Backend (semua running per frame independen, CPU, setting thread default):
- mediapipe     : SelfieMulticlass 256x256, Tasks API ImageSegmenter, running mode IMAGE.
                  prob orang = 1 - confidence kelas 0 (background) -> rambut/aksesori ikut siluet.
- mediapipe_pad : sama, tapi frame di-pad jadi persegi (abu-abu netral, di tengah) sebelum
                  inferensi, lalu mask di-crop balik. Diagnostik aspect ratio: cross_iou
                  mediapipe vs mediapipe_pad ~1 berarti MediaPipe sudah letterbox sendiri.
- u2net         : rembg u2net_human_seg, model EKSPLISIT (default rembg = bria-rmbg, D-003).

PENYIMPANGAN dari rembg (disetujui di T-102a): jalur u2net TIDAK memakai remove()/predict().
predict() rembg (dan normPRED di kode test U-2-Net asli) melakukan min-max normalization per
frame: (pred - min) / (max - min). Akibatnya threshold 0.5 jadi relatif per frame: frame yang
modelnya ragu (max rendah) tetap "dipaksa" punya piksel bernilai 1, noise ikut terangkat.
Script ini memakai jalur raw: session.normalize() -> inner_session.run() -> channel 0 (sudah
sigmoid, 0..1) -> resize bilinear ke ukuran frame. Min/max raw per frame dicatat di CSV supaya
terukur seberapa besar efek min-max itu. Preprocessing input tetap milik rembg (resize LANCZOS
ke 320x320 = STRETCH, bagi max piksel, mean/std ImageNet).
KALAU u2net TERPILIH: T-102b WAJIB memakai jalur raw yang sama.

Metrik: mask dibinarisasi dengan threshold yang sama (prob > threshold), tanpa morfologi /
largest component (itu T-102b). Waktu = hanya panggilan infer() (RGB array -> prob HxW, termasuk
pre/post-processing library dan pad/crop A2), bukan baca/tulis file.

Definisi IoU: |A n B| / |A u B|. Kalau kedua mask kosong (union = 0) IoU = 1.0 (dua mask
identik); frame kosong tetap tertangkap oleh area_ratio < area_min.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MP_MODEL = ROOT / "models" / "selfie_multiclass_256x256.tflite"
REMBG_MODEL = "u2net_human_seg"  # WAJIB eksplisit: default rembg = bria-rmbg (D-003)

# Konstanta model u2net (sama dengan U2netHumanSegSession.predict di rembg 2.0.85)
U2NET_MEAN = (0.485, 0.456, 0.406)
U2NET_STD = (0.229, 0.224, 0.225)
U2NET_SIZE = (320, 320)

BACKENDS = ["mediapipe", "mediapipe_pad", "u2net"]
# Pasangan cross_iou: (kolom CSV, backend a, backend b)
CROSS_PAIRS = [
    ("cross_iou_mp_u2", "mediapipe", "u2net"),
    ("cross_iou_mppad_u2", "mediapipe_pad", "u2net"),
    ("cross_iou_mp_mppad", "mediapipe", "mediapipe_pad"),
]
DISAGREE_PAIR = "cross_iou_mp_u2"  # dasar pemilihan disagreement_top12

# Visualisasi (bukan parameter pipeline)
PAD_VALUE = 128  # warna pad abu-abu netral untuk mediapipe_pad
OVERLAY_BGR = {"mediapipe": (60, 60, 230), "mediapipe_pad": (60, 180, 60), "u2net": (230, 120, 40)}
OVERLAY_ALPHA = 0.45
CONTOUR_BGR = (20, 20, 20)
CONTOUR_THICKNESS = 4
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_THICKNESS = 1
LABEL_BG = (255, 255, 255)
LABEL_FG = (0, 0, 0)
LABEL_LINE_H = 22
LABEL_PAD = 6


# --- util ------------------------------------------------------------------
def fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU dua mask boolean. Keduanya kosong -> 1.0 (lihat docstring modul)."""
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(a, b).sum() / union)


def area_ratio(m: np.ndarray) -> float:
    return float(m.mean())


def big_blob_count(m: np.ndarray, blob_min: float) -> int:
    """Jumlah connected component dengan luas > blob_min x luas frame."""
    n, _, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    min_px = blob_min * m.size
    return int(sum(1 for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] > min_px))


def pad_square(img: np.ndarray, value: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Pad di tengah jadi persegi. Return (img_persegi, (offset_y, offset_x))."""
    h, w = img.shape[:2]
    s = max(h, w)
    oy, ox = (s - h) // 2, (s - w) // 2
    out = np.full((s, s) + img.shape[2:], value, dtype=img.dtype)
    out[oy:oy + h, ox:ox + w] = img
    return out, (oy, ox)


def crop_back(img: np.ndarray, offset: tuple[int, int], h: int, w: int) -> np.ndarray:
    oy, ox = offset
    return img[oy:oy + h, ox:ox + w]


def self_check() -> None:
    """Cek fungsi metrik sebelum menjalankan apa pun. Gagal -> berhenti dengan pesan jelas."""
    def check(cond: bool, msg: str) -> None:
        if not cond:
            fail(f"self-check gagal: {msg}")

    rng = np.random.default_rng(0)
    m = rng.random((40, 30)) > 0.5
    check(iou(m, m) == 1.0, "iou(m, m) != 1")
    empty = np.zeros((40, 30), bool)
    v = iou(empty, empty)
    check(v == 1.0 and not np.isnan(v), f"iou(kosong, kosong) = {v}, harapan 1.0")
    check(iou(empty, ~empty) == 0.0, "iou(kosong, penuh) != 0")
    check(area_ratio(np.ones((40, 30), bool)) == 1.0, "area_ratio(mask penuh) != 1")
    for h, w in [(854, 480), (480, 854), (256, 256)]:  # portrait, landscape, persegi
        img = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        sq, off = pad_square(img, PAD_VALUE)
        check(sq.shape[0] == sq.shape[1], f"pad_square {h}x{w} tidak persegi")
        check(np.array_equal(crop_back(sq, off, h, w), img), f"pad->crop round-trip {h}x{w} tidak identik")
    print("self-check OK")


# --- backend ---------------------------------------------------------------
class MediaPipeBackend:
    def __init__(self, pad: bool):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        if not MP_MODEL.is_file():
            fail(f"model MediaPipe tidak ada: {MP_MODEL}")
        self.mp = mp
        self.pad = pad
        options = vision.ImageSegmenterOptions(
            base_options=BaseOptions(model_asset_path=str(MP_MODEL)),
            running_mode=vision.RunningMode.IMAGE,
            output_confidence_masks=True,
        )
        self.segmenter = vision.ImageSegmenter.create_from_options(options)
        self.info = {"model": MP_MODEL.name, "mediapipe": mp.__version__, "pad": pad}

    def infer(self, rgb: np.ndarray) -> tuple[np.ndarray, dict]:
        h, w = rgb.shape[:2]
        src, off = pad_square(rgb, PAD_VALUE) if self.pad else (rgb, (0, 0))
        mp_img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(src))
        res = self.segmenter.segment(mp_img)
        prob = 1.0 - np.squeeze(res.confidence_masks[0].numpy_view()).astype(np.float32)
        if prob.shape != src.shape[:2]:
            prob = cv2.resize(prob, (src.shape[1], src.shape[0]), interpolation=cv2.INTER_LINEAR)
        if self.pad:
            prob = crop_back(prob, off, h, w)
        return prob, {}

    def close(self) -> None:
        self.segmenter.close()


class U2netBackend:
    def __init__(self):
        import onnxruntime as ort
        import rembg
        from PIL import Image
        from rembg import new_session
        from rembg.sessions.u2net_human_seg import U2netHumanSegSession

        # Pastikan TIDAK ada unduhan: file model harus sudah ada di lokal
        existing = U2netHumanSegSession.resolve_existing(f"{REMBG_MODEL}.onnx")
        if existing is None:
            fail(f"model {REMBG_MODEL}.onnx tidak ditemukan di {U2netHumanSegSession.model_dir()} "
                 "- script ini tidak mengunduh model")
        self.Image = Image
        # Model eksplisit + provider CPU eksplisit (aturan #6)
        self.session = new_session(REMBG_MODEL, providers=["CPUExecutionProvider"])
        providers = self.session.inner_session.get_providers()
        print(f"  u2net providers: {providers}")
        if providers != ["CPUExecutionProvider"]:
            fail(f"u2net harus hanya CPUExecutionProvider, didapat {providers}")
        self.info = {"model": REMBG_MODEL, "model_path": existing, "rembg": rembg.__version__,
                     "onnxruntime": ort.__version__, "providers": providers,
                     "path": "raw (normalize -> inner_session.run -> ch0), TANPA min-max predict()"}

    def infer(self, rgb: np.ndarray) -> tuple[np.ndarray, dict]:
        h, w = rgb.shape[:2]
        feed = self.session.normalize(self.Image.fromarray(rgb), U2NET_MEAN, U2NET_STD, U2NET_SIZE)
        out = self.session.inner_session.run(None, feed)
        raw = out[0][0, 0].astype(np.float32)  # d1, sudah sigmoid (0..1)
        prob = cv2.resize(raw, (w, h), interpolation=cv2.INTER_LINEAR)
        return prob, {"raw_min": float(raw.min()), "raw_max": float(raw.max())}

    def close(self) -> None:
        pass


def make_backend(name: str):
    if name == "mediapipe":
        return MediaPipeBackend(pad=False)
    if name == "mediapipe_pad":
        return MediaPipeBackend(pad=True)
    if name == "u2net":
        return U2netBackend()
    fail(f"backend tidak dikenal: {name}")


# --- I/O -------------------------------------------------------------------
def frame_names(meta: dict) -> list[str]:
    start = meta.get("frame_index_start", 0)
    return [f"frame_{i:05d}.png" for i in range(start, start + meta["frame_count"])]


def read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        fail(f"gagal membaca frame: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def read_mask(path: Path) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        fail(f"gagal membaca mask: {path}")
    return m > 127


def backend_complete(out: Path, name: str, names: list[str]) -> bool:
    mdir = out / f"masks_{name}"
    if not (out / f"infer_{name}.json").is_file() or not mdir.is_dir():
        return False
    return all((mdir / n).is_file() for n in names)


def run_backend(name: str, frames_dir: Path, out: Path, names: list[str], threshold: float) -> None:
    print(f"\n[{name}] inferensi {len(names)} frame ...")
    backend = make_backend(name)
    mdir = out / f"masks_{name}"
    mdir.mkdir(parents=True, exist_ok=True)
    ms, extra = [], []
    try:
        for i, n in enumerate(names):
            rgb = read_rgb(frames_dir / n)
            t0 = time.perf_counter()
            prob, ex = backend.infer(rgb)
            ms.append((time.perf_counter() - t0) * 1000)
            extra.append(ex)
            cv2.imwrite(str(mdir / n), np.where(prob > threshold, 255, 0).astype(np.uint8))
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(names)}")
    finally:
        backend.close()
    record = {"backend": name, "threshold": threshold, "info": backend.info, "ms": ms, "extra": extra}
    (out / f"infer_{name}.json").write_text(json.dumps(record, indent=1), encoding="utf-8")


# --- metrik ----------------------------------------------------------------
def compute_metrics(out: Path, names: list[str], backends: list[str], args) -> list[dict]:
    infer = {b: json.loads((out / f"infer_{b}.json").read_text(encoding="utf-8")) for b in backends}
    for b in backends:
        if infer[b]["threshold"] != args.threshold:
            print(f"  PERINGATAN: mask {b} dibuat dengan threshold {infer[b]['threshold']}, "
                  f"bukan {args.threshold} -> pakai --force untuk samakan")
    rows, prev = [], {}
    for i, n in enumerate(names):
        masks = {b: read_mask(out / f"masks_{b}" / n) for b in backends}
        row = {"frame": n}
        for b in backends:
            m = masks[b]
            row[f"{b}_ms"] = round(infer[b]["ms"][i], 3)
            row[f"{b}_area_ratio"] = round(area_ratio(m), 5)
            row[f"{b}_iou_prev"] = round(iou(m, prev[b]), 5) if b in prev else ""
            row[f"{b}_big_blobs"] = big_blob_count(m, args.blob_min)
        if "u2net" in backends:
            ex = infer["u2net"]["extra"][i]
            row["u2net_raw_min"] = round(ex["raw_min"], 5)
            row["u2net_raw_max"] = round(ex["raw_max"], 5)
        for col, a, b in CROSS_PAIRS:
            if a in masks and b in masks:
                row[col] = round(iou(masks[a], masks[b]), 5)
        rows.append(row)
        prev = masks
    return rows


def stats_block(values: list[float]) -> dict:
    if not values:
        return {}
    return {"mean": statistics.fmean(values), "median": statistics.median(values),
            "min": min(values), "max": max(values)}


def summarize(rows: list[dict], out: Path, backends: list[str], args) -> dict:
    infer = {b: json.loads((out / f"infer_{b}.json").read_text(encoding="utf-8")) for b in backends}
    summary = {
        "task": "T-102a", "frame_count": len(rows), "threshold": args.threshold,
        "qc": {"iou_min": args.iou_min, "area_min": args.area_min, "area_max": args.area_max,
               "blob_min": args.blob_min},
        "warmup_frames": args.warmup, "cpu_cores": os.cpu_count(),
        "backends": {}, "cross_iou": {},
    }
    for b in backends:
        ms = infer[b]["ms"]
        timed = ms[args.warmup:]
        ious = [r[f"{b}_iou_prev"] for r in rows if r[f"{b}_iou_prev"] != ""]
        areas = [r[f"{b}_area_ratio"] for r in rows]
        blobs = [r[f"{b}_big_blobs"] for r in rows]
        s = {
            "info": infer[b]["info"],
            "ms_warmup": [round(x, 1) for x in ms[:args.warmup]],
            "ms_mean": statistics.fmean(timed), "ms_median": statistics.median(timed),
            "ms_p95": float(np.percentile(timed, 95)),
            "iou_prev_mean": statistics.fmean(ious), "iou_prev_median": statistics.median(ious),
            "iou_prev_min": min(ious), "n_iou_prev_below_min": sum(v < args.iou_min for v in ious),
            "n_area_below_min": sum(v < args.area_min for v in areas),
            "n_area_above_max": sum(v > args.area_max for v in areas),
            "n_multi_big_blob": sum(v > 1 for v in blobs),
        }
        if b == "u2net":
            s["raw_min"] = stats_block([r["u2net_raw_min"] for r in rows])
            s["raw_max"] = stats_block([r["u2net_raw_max"] for r in rows])
        summary["backends"][b] = s
    for col, a, b in CROSS_PAIRS:
        vals = [r[col] for r in rows if col in r]
        if vals:
            worst = min(range(len(rows)), key=lambda i: rows[i].get(col, 2.0))
            summary["cross_iou"][col] = {"a": a, "b": b, "mean": statistics.fmean(vals),
                                         "median": statistics.median(vals), "min": min(vals),
                                         "min_frame": rows[worst]["frame"]}
    return summary


def write_csv(rows: list[dict], path: Path) -> None:
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def print_table(summary: dict) -> None:
    cols = [("ms mean", "ms_mean", "{:.1f}"), ("ms med", "ms_median", "{:.1f}"),
            ("ms p95", "ms_p95", "{:.1f}"), ("iou mean", "iou_prev_mean", "{:.3f}"),
            ("iou med", "iou_prev_median", "{:.3f}"), ("iou min", "iou_prev_min", "{:.3f}"),
            (f"iou<{summary['qc']['iou_min']}", "n_iou_prev_below_min", "{}"),
            (f"area<{summary['qc']['area_min']:.0%}", "n_area_below_min", "{}"),
            (f"area>{summary['qc']['area_max']:.0%}", "n_area_above_max", "{}"),
            (">1 blob", "n_multi_big_blob", "{}")]
    head = f"{'backend':<14}" + "".join(f"{c[0]:>10}" for c in cols)
    print("\n" + head + "\n" + "-" * len(head))
    for b, s in summary["backends"].items():
        print(f"{b:<14}" + "".join(f"{c[2].format(s[c[1]]):>10}" for c in cols))
    print(f"\n(n = {summary['frame_count']} frame; ms tanpa {summary['warmup_frames']} frame warm-up; "
          f"threshold {summary['threshold']}; {summary['cpu_cores']} CPU core)")
    for b, s in summary["backends"].items():
        print(f"  warm-up {b}: {s['ms_warmup']} ms")
    for col, c in summary["cross_iou"].items():
        print(f"  {c['a']} vs {c['b']}: cross_iou mean {c['mean']:.3f}, median {c['median']:.3f}, "
              f"min {c['min']:.3f} ({c['min_frame']})")
    if "u2net" in summary["backends"]:
        s = summary["backends"]["u2net"]
        print(f"  u2net raw max: mean {s['raw_max']['mean']:.3f}, min {s['raw_max']['min']:.3f}; "
              f"raw min: mean {s['raw_min']['mean']:.4f}, max {s['raw_min']['max']:.4f}")


# --- visualisasi -----------------------------------------------------------
def draw_label(img: np.ndarray, lines: list[str]) -> None:
    w = max(cv2.getTextSize(t, FONT, FONT_SCALE, FONT_THICKNESS)[0][0] for t in lines) + 2 * LABEL_PAD
    cv2.rectangle(img, (0, 0), (w, LABEL_LINE_H * len(lines) + LABEL_PAD), LABEL_BG, -1)
    for k, t in enumerate(lines):
        cv2.putText(img, t, (LABEL_PAD, LABEL_LINE_H * (k + 1)), FONT, FONT_SCALE, LABEL_FG,
                    FONT_THICKNESS, cv2.LINE_AA)


def render_panel(bgr: np.ndarray, mask: np.ndarray | None, name: str, frame: str, iou_prev) -> np.ndarray:
    img = bgr.copy()
    if mask is not None:
        color = np.zeros_like(img)
        color[:] = OVERLAY_BGR[name]
        blend = cv2.addWeighted(img, 1 - OVERLAY_ALPHA, color, OVERLAY_ALPHA, 0)
        img[mask] = blend[mask]
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, contours, -1, CONTOUR_BGR, CONTOUR_THICKNESS, cv2.LINE_AA)
        iou_txt = f"iou_prev {iou_prev:.3f}" if iou_prev != "" else "iou_prev -"
        draw_label(img, [name, frame, iou_txt])
    else:
        draw_label(img, ["asli", frame])
    return img


def render_row(frames_dir: Path, out: Path, backends: list[str], row: dict) -> np.ndarray:
    n = row["frame"]
    bgr = cv2.imread(str(frames_dir / n), cv2.IMREAD_COLOR)
    stem = Path(n).stem
    panels = [render_panel(bgr, None, "asli", stem, "")]
    for b in backends:
        panels.append(render_panel(bgr, read_mask(out / f"masks_{b}" / n), b, stem, row[f"{b}_iou_prev"]))
    return np.hstack(panels)


def even(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    return cv2.copyMakeBorder(img, 0, h % 2, 0, w % 2, cv2.BORDER_CONSTANT, value=(0, 0, 0))


def write_video(frames_dir: Path, out: Path, backends: list[str], rows: list[dict], fps: int) -> Path:
    path = out / "side_by_side.mp4"
    first = even(render_row(frames_dir, out, backends, rows[0]))
    h, w = first.shape[:2]
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
           str(path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    proc.stdin.write(first.tobytes())
    for row in rows[1:]:
        proc.stdin.write(even(render_row(frames_dir, out, backends, row)).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        fail("ffmpeg gagal meng-encode side_by_side.mp4")
    return path


def write_disagreement_grid(frames_dir: Path, out: Path, backends: list[str], rows: list[dict],
                            top_k: int, cols: int, scale: float) -> Path | None:
    if not all(DISAGREE_PAIR in r for r in rows):
        print(f"  lewati disagreement grid: {DISAGREE_PAIR} tidak tersedia")
        return None
    worst = sorted(rows, key=lambda r: r[DISAGREE_PAIR])[:top_k]
    tiles = []
    for r in worst:
        t = render_row(frames_dir, out, backends, r)
        t = cv2.resize(t, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        draw_label(t, [f"{DISAGREE_PAIR} {r[DISAGREE_PAIR]:.3f}"])
        tiles.append(t)
    blank = np.zeros_like(tiles[0])
    while len(tiles) % cols:
        tiles.append(blank)
    grid = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    path = out / f"disagreement_top{top_k}.png"
    cv2.imwrite(str(path), grid)
    return path


# --- main ------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--work", type=Path, default=ROOT / "work")
    p.add_argument("--out", type=Path, default=ROOT / "work" / "ab_t102a")
    p.add_argument("--backends", default=",".join(BACKENDS), help="subset dari: " + ",".join(BACKENDS))
    p.add_argument("--threshold", type=float, default=0.5, help="binarisasi mask: prob > threshold")
    # Threshold QC dari tabel QC docs/01 stage [2]
    p.add_argument("--iou-min", type=float, default=0.55, help="QC iou_prev gagal kalau < nilai ini")
    p.add_argument("--area-min", type=float, default=0.03, help="QC area_ratio gagal kalau < nilai ini")
    p.add_argument("--area-max", type=float, default=0.70, help="QC area_ratio gagal kalau > nilai ini")
    p.add_argument("--blob-min", type=float, default=0.05, help="blob besar = luas > nilai ini x frame")
    p.add_argument("--warmup", type=int, default=3, help="frame awal yang dibuang dari statistik waktu")
    p.add_argument("--fps", type=int, default=24, help="fps side_by_side.mp4")
    p.add_argument("--top-k", type=int, default=12, help="jumlah frame di disagreement grid")
    p.add_argument("--grid-cols", type=int, default=3)
    p.add_argument("--grid-scale", type=float, default=0.4)
    p.add_argument("--force", action="store_true", help="inferensi ulang walau mask sudah lengkap")
    args = p.parse_args()

    t_start = time.perf_counter()
    self_check()

    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    for b in backends:
        if b not in BACKENDS:
            fail(f"backend tidak dikenal: {b}")
    meta_path = args.work / "meta.json"
    if not meta_path.is_file():
        fail(f"{meta_path} tidak ada - jalankan ingest (T-101) dulu")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    frames_dir = args.work / "frames"
    names = frame_names(meta)
    missing = [n for n in names if not (frames_dir / n).is_file()]
    if missing:
        fail(f"{len(missing)} frame hilang di {frames_dir}, mis. {missing[0]}")
    if len(names) <= args.warmup:
        fail(f"frame_count {len(names)} <= warm-up {args.warmup}")
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"{len(names)} frame {meta['working_width']}x{meta['working_height']}, "
          f"backend: {', '.join(backends)}, {os.cpu_count()} CPU core")

    for b in backends:
        if backend_complete(args.out, b, names) and not args.force:
            print(f"\n[{b}] mask lengkap -> inferensi dilewati (--force untuk ulang)")
        else:
            run_backend(b, frames_dir, args.out, names, args.threshold)

    print("\nmenghitung metrik ...")
    rows = compute_metrics(args.out, names, backends, args)
    summary = summarize(rows, args.out, backends, args)
    write_csv(rows, args.out / "metrics_per_frame.csv")

    print("render side_by_side.mp4 ...")
    video = write_video(frames_dir, args.out, backends, rows, args.fps)
    grid = write_disagreement_grid(frames_dir, args.out, backends, rows,
                                   args.top_k, args.grid_cols, args.grid_scale)

    summary["total_wall_s"] = time.perf_counter() - t_start
    (args.out / "metrics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print_table(summary)
    print(f"\nOutput di {args.out}:")
    for f in ["metrics_per_frame.csv", "metrics_summary.json", video.name] + ([grid.name] if grid else []):
        print(f"  {f}")
    for b in backends:
        print(f"  masks_{b}/, infer_{b}.json")
    print(f"\nTotal waktu: {summary['total_wall_s']:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
