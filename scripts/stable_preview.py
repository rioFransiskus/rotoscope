"""T-302 alat visual (sekali pakai, bukan bagian src/): bandingkan peta grup argmax MENTAH vs hasil temporal [3].

Membaca (hanya baca): seg/ dari --raw-work (untuk argmax mentah), stable/groups dari --stable-work. Menulis ke --out (default work/t302/):
  (a) cmp_<nama>_<awal>-<akhir>.mp4: 4 panel berdampingan = argmax mentah | temporal | peta selisih (piksel hasil ≠ mentah) |
      peta flip-flop (A→B→A ≤ 2 frame di hasil), rentang 73-92, 183-202, 225-240 + jendela flip-flop terbanyak;
  (b) worst_<nama>_<jenis>.png: kasus terburuk (kesetiaan grup terendah, rasio luas lengan terendah, flip-flop tertinggi).
Label di tiap frame: "T-302 tanpa optical flow; alpha / b / normalisasi = ...". Tanpa GPU, tanpa torch.

Pakai: python scripts/stable_preview.py --raw-work work/clips/test --stable-work <salinan> --label "alpha=0,7 b=0 log_median"
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tests"))

import temporal_metrics as tm  # noqa: E402
from rotoscope import stabilize as stb  # noqa: E402
from rotoscope.config import load_class_names, load_pipeline  # noqa: E402

PALETTE = np.array([(245, 245, 245), (40, 70, 130), (120, 170, 230), (200, 120, 60), (80, 170, 80), (40, 140, 240),
                    (170, 70, 150), (60, 60, 200)], np.uint8)      # BGR: bg, hair, face, torso, left_arm, right_arm, legs
DIFF_COLOR, FLIP_COLOR, GRAY = (0, 0, 220), (200, 0, 200), (225, 225, 225)
DEFAULT_RANGES = ((73, 92), (183, 202), (225, 240))
FPS = 24
LABEL_H = 34


def colorize(g: np.ndarray) -> np.ndarray:
    return PALETTE[np.minimum(g, len(PALETTE) - 1)]


def diff_panel(raw: np.ndarray, out: np.ndarray) -> np.ndarray:
    img = np.full((*raw.shape, 3), GRAY, np.uint8)
    img[(raw != 0) | (out != 0)] = (190, 190, 190)
    img[raw != out] = DIFF_COLOR
    return img


def flip_panel(out: np.ndarray, flip: np.ndarray) -> np.ndarray:
    img = colorize(out) // 2 + 120
    img[flip] = FLIP_COLOR
    return img.astype(np.uint8)


def flip_mask(g: np.ndarray, t: int) -> np.ndarray:
    """Piksel yang berubah di frame t dan kembali ke grup t−1 pada t+1 atau t+2."""
    m = np.zeros(g.shape[1:], bool)
    if t < 1 or t + 1 >= len(g):
        return m
    ch = g[t] != g[t - 1]
    back1 = ch & (g[t + 1] == g[t - 1])
    m |= back1
    if t + 2 < len(g):
        m |= ch & ~back1 & (g[t + 1] != g[t - 1]) & (g[t + 2] == g[t - 1])
    return m


def frame_panels(raw, out, t, label: str) -> np.ndarray:
    row = np.hstack([colorize(raw[t]), colorize(out[t]), diff_panel(raw[t], out[t]), flip_panel(out[t], flip_mask(out, t))])
    bar = np.full((LABEL_H, row.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, f"frame {t} | mentah | temporal | selisih (merah) | flip-flop (magenta) | {label}", (6, 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([bar, row])


def write_video(path: Path, frames: list[np.ndarray]) -> None:
    h, w = frames[0].shape[:2]
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(FPS),
           "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(path)]
    p = subprocess.run(cmd, input=b"".join(f.tobytes() for f in frames), capture_output=True)
    if p.returncode != 0:
        raise SystemExit(f"ffmpeg gagal: {p.stderr.decode(errors='replace')[:400]}")


def load_raw(clip: stb.Clip, cfg) -> np.ndarray:
    classes = load_class_names()
    lut = stb.class_group_lut(cfg.groups, classes)
    out = []
    for n in clip.names:
        probs, cm, _ = stb._read_inputs(clip, n, len(classes))
        g, _ = stb.group_argmax(stb.group_sums(probs, lut, len(cfg.groups)), lut[cm])
        out.append(g)
    return np.stack(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-work", type=Path, required=True, help="folder klip dengan seg/ (argmax mentah)")
    ap.add_argument("--stable-work", type=Path, required=True, help="folder klip dengan stable/groups hasil temporal")
    ap.add_argument("--label", default="alpha / b / normalisasi = ?", help="teks label (alpha, b, normalisasi)")
    ap.add_argument("--out", type=Path, default=REPO / "work" / "t302")
    ap.add_argument("--name", default=None, help="nama berkas keluaran (default: nama folder klip)")
    ap.add_argument("--ranges", default=None, help="mis. 73-92,183-202 (default: tiga rentang standar)")
    args = ap.parse_args()
    cfg = load_pipeline(REPO / "configs" / "default.yaml")
    raw_clip, st_clip = stb.load_clip(args.raw_work), stb.load_clip(args.stable_work)
    name = args.name or args.raw_work.name
    args.out.mkdir(parents=True, exist_ok=True)
    label = f"T-302 tanpa optical flow; {args.label}"
    raw = load_raw(raw_clip, cfg)
    out = np.stack([stb.read_groups(st_clip.groups_path(n)) for n in st_clip.names])
    n = len(out)
    ranges = ([tuple(int(x) for x in r.split("-")) for r in args.ranges.split(",")] if args.ranges else list(DEFAULT_RANGES))
    ff = tm.flipflop_counts(out)
    peak = int(np.argmax(ff / np.maximum(tm.fg_area(out), 1)))
    ranges.append((max(0, peak - 10), min(n - 1, peak + 9)))
    summary = {"label": label, "clip": name, "frames": n, "flipflop_peak_frame": peak, "videos": [], "png": []}
    for lo, hi in ranges:
        if lo >= n:
            continue
        hi = min(hi, n - 1)
        path = args.out / f"cmp_{name}_{lo}-{hi}.mp4"
        write_video(path, [frame_panels(raw, out, t, label) for t in range(lo, hi + 1)])
        summary["videos"].append(path.name)
    # (b) kasus terburuk
    worst = {"kesetiaan_grup_terendah": None, "rasio_lengan_terendah": int(np.argmin(tm.limb_ratio(out, raw))),
             "flipflop_tertinggi": peak}
    gi = []
    for t in range(n):
        v = tm.group_iou(out[t:t + 1], raw[t:t + 1])
        gi.append(v.min() if v.size else 1.0)
    worst["kesetiaan_grup_terendah"] = int(np.argmin(gi))
    for kind, t in worst.items():
        path = args.out / f"worst_{name}_{kind}_f{t}.png"
        ok, buf = cv2.imencode(".png", frame_panels(raw, out, t, label))
        path.write_bytes(buf.tobytes())
        summary["png"].append(path.name)
    summary["worst_frames"] = worst
    (args.out / f"preview_{name}.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
