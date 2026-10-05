"""Alat pratinjau sekali pakai T-203b (Tahap 3): MP4 hasil export vs PNG sumber strokes → work/t203b/ (F1).

Untuk tiap klip: frame kasus terburuk (PSNR terendah, selisih warna tinta terbesar), frame pertama / tengah / terakhir.
Keluaran per frame: <klip>_f<idx>_mp4.png (frame MP4 didekode bt709), <klip>_f<idx>_compare_zoom3x.png (kiri PNG sumber,
tengah MP4, kanan |selisih|×4; crop 3× di sekitar galat lokal terbesar pada area garis) + summary.json.
Pakai: python scripts/t203b_preview.py   (tidak membuka / menilai gambar; hanya menulis berkas).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))
import export_metrics as em  # noqa: E402

OUT = ROOT / "work" / "t203b"
CROP = 160
ZOOM = 3
DIFF_GAIN = 4
STEP = 5


def label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 22), (255, 255, 255), -1)
    cv2.putText(out, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def worst_spot(ref: np.ndarray, dec: np.ndarray) -> tuple[int, int]:
    err = np.abs(ref.astype(int) - dec.astype(int)).sum(-1).astype(np.float32)
    err *= em.ink_mask(ref)
    blur = cv2.blur(err, (31, 31))
    y, x = np.unravel_index(int(np.argmax(blur)), blur.shape)
    return int(y), int(x)


def save_rgb(path: Path, rgb: np.ndarray) -> None:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    assert ok
    path.write_bytes(buf.tobytes())


def compare(clip: str, idx: int, why: str, ref: np.ndarray, dec: np.ndarray) -> dict:
    stem = f"{clip}_f{idx:05d}"
    save_rgb(OUT / f"{stem}_mp4.png", dec)
    y, x = worst_spot(ref, dec)
    y0 = min(max(y - CROP // 2, 0), ref.shape[0] - CROP)
    x0 = min(max(x - CROP // 2, 0), ref.shape[1] - CROP)
    sl = (slice(y0, y0 + CROP), slice(x0, x0 + CROP))
    diff = np.clip(np.abs(ref.astype(int) - dec.astype(int)) * DIFF_GAIN, 0, 255).astype(np.uint8)
    panels = [label(cv2.resize(a[sl], None, fx=ZOOM, fy=ZOOM, interpolation=cv2.INTER_NEAREST), t)
              for a, t in ((ref, "PNG sumber (strokes)"), (dec, "MP4 (bt709)"), (255 - diff, f"|selisih| x{DIFF_GAIN}"))]
    save_rgb(OUT / f"{stem}_compare_zoom3x.png", np.hstack(panels))
    rep = em.frame_report(ref, dec)
    return {"frame": idx, "why": why, "crop_xy": [x0, y0], "crop_px": CROP, **rep}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary: dict = {}
    for clip in ("test_short", "test"):
        work = ROOT / "work" / "clips" / clip
        mp4 = ROOT / "out" / f"{clip}.mp4"
        n = json.loads((work / "meta.json").read_text(encoding="utf-8"))["frame_count"]
        rows = []
        for i in em.sample_indices(n, step=STEP):
            ref = em.read_png_rgb(work / "strokes" / f"frame_{i:05d}.png")
            rep = em.frame_report(ref, em.decode_rgb(mp4, i))
            rows.append({"frame": i, **rep})
        worst_psnr = min(rows, key=lambda r: r["psnr"])["frame"]
        worst_ink = max(rows, key=lambda r: max(abs(v) for v in (r["ink_delta"] or [0])))["frame"]
        worst_paper = max(rows, key=lambda r: max(abs(v) for v in (r["paper_delta"] or [0])))["frame"]
        picks = {worst_psnr: "PSNR terendah", worst_ink: "selisih warna tinta terbesar",
                 worst_paper: "selisih warna kertas terbesar", 0: "pertama", n // 2: "tengah", n - 1: "terakhir"}
        out = []
        for idx in sorted(picks):
            ref = em.read_png_rgb(work / "strokes" / f"frame_{idx:05d}.png")
            out.append(compare(clip, idx, picks[idx], ref, em.decode_rgb(mp4, idx)))
        summary[clip] = {"mp4": str(mp4), "frames_sampled": len(rows), "picked": out}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"ditulis ke {OUT}: {len(list(OUT.glob('*.png')))} PNG + summary.json")


if __name__ == "__main__":
    main()
