"""[DIARSIPKAN] Sumber data (work/t305b_scratch/arch) dihapus di Tahap 4 T-305b; skrip tidak dapat dijalankan lagi tanpa membangun ulang arsip
varian (config: D_low 0 / 2 dan exclude_groups [] / [hair]). Hasil papan: work/t305b/nilai/11–15.

T-305b Tahap 3: papan sebelum / sesudah dari keluaran PRODUKSI yang sudah dirender di arsip scratch
(work/t305b_scratch/arch/<varian>/<klip>/strokes/*.png, preset default). Keluaran ke work/t305b/nilai/.

    python scripts/t305b_board.py boards   # papan 4 panel [sekarang | hair saja | D_low 2 saja | kombinasi final], frame 80 + 233 + 2 lain
    python scripts/t305b_board.py videos   # video [sekarang | kombinasi final]: jendela statis 25-36 dan cepat 73-92 (klip test)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARCH = ROOT / "work" / "t305b_scratch" / "arch"
OUT = ROOT / "work" / "t305b" / "nilai"
PANELS = (("neutral", "sekarang"), ("hair", "exclude hair"), ("dlow", "D_low 2"), ("final", "D_low 2 + hair"))
SCALE_BOARD = 0.55
SCALE_VIDEO = 0.5
MARGIN = 40


def frame_png(variant: str, clip: str, i: int) -> np.ndarray:
    img = cv2.imread(str(ARCH / variant / clip / "strokes" / f"frame_{i:05d}.png"), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"PNG tidak ada: {variant}/{clip}/{i}")
    return img


def body_box(img: np.ndarray) -> tuple[int, int, int, int]:
    dark = (img.min(axis=2) < 200)
    ys, xs = np.nonzero(dark)
    h, w = dark.shape
    return (max(int(xs.min()) - MARGIN, 0), max(int(ys.min()) - MARGIN, 0), min(int(xs.max()) + MARGIN, w), min(int(ys.max()) + MARGIN, h))


def label(img: np.ndarray, text: str) -> np.ndarray:
    bar = np.full((34, img.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, text, (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def cmd_boards() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    info = json.loads((OUT.parent / "nilai_info.json").read_text(encoding="utf-8"))["meta"]
    jobs = [("test", 80), ("test", 233), ("test", info["test_ext_frame"]), ("test_short", info["test_short_frame"])]
    for n, (clip, i) in enumerate(jobs, 11):
        imgs = {v: frame_png(v, clip, i) for v, _ in PANELS}
        x0, y0, x1, y1 = body_box(np.minimum.reduce(list(imgs.values())))      # kotak yang memuat tinta SEMUA varian
        panels = []
        for v, name in PANELS:
            crop = imgs[v][y0:y1, x0:x1]
            crop = cv2.resize(crop, None, fx=SCALE_BOARD, fy=SCALE_BOARD, interpolation=cv2.INTER_AREA)
            panels.append(label(crop, f"{clip} f{i:03d}: {name}"))
        path = OUT / f"{n}_papan_{clip}_f{i:03d}.png"
        cv2.imwrite(str(path), np.hstack(panels))
        print(path.name)


def cmd_videos() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for tag, lo, hi in (("statis_25-36", 25, 36), ("cepat_73-92", 73, 92)):
        path = OUT / f"15_video_{tag}.mp4"
        vw = None
        for i in range(lo, hi + 1):
            a, b = frame_png("neutral", "test", i), frame_png("final", "test", i)
            panels = [label(cv2.resize(x, None, fx=SCALE_VIDEO, fy=SCALE_VIDEO, interpolation=cv2.INTER_AREA), f"f{i:03d} {t}")
                      for x, t in ((a, "sekarang"), (b, "D_low 2 + hair"))]
            frame = np.hstack(panels)
            if vw is None:
                vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 6, (frame.shape[1], frame.shape[0]))
            vw.write(frame)
        vw.release()
        print(path.name)


if __name__ == "__main__":
    {"boards": cmd_boards, "videos": cmd_videos}[sys.argv[1]]()
