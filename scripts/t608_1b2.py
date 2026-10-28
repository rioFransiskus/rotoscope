"""T-608 1b-2 langkah 3–4 (alat, F3; CPU): metrik + bahan penilaian isolasi klip5, HANYA f0–53 (keputusan lingkup 2026-10-28).

    python scripts/t608_1b2.py metrik   # SEBELUM (produksi) vs SESUDAH (scratch iso, masker [1b] di [3]) -> work/t608/iso2_metrik.json
    python scripts/t608_1b2.py bahan    # work/t608/nilai_1b/: masker_f0-53(.mp4, _lambat_0.5x.mp4) + berdampingan_f0-53(.mp4, _lambat_0.5x.mp4)
    python scripts/t608_1b2.py cek      # ffprobe / sha256 bahan -> work/t608/iso2_bahan.json

Hanya MEMBACA work/clips/klip5 + out/klip5.mp4 (SEBELUM) dan work/t608_scratch/iso/ (SESUDAH + masker); menulis hanya ke work/t608/.
Tanpa GPU (torch tidak diimpor). Frame / video dibaca sebagai angka (cv2), tidak ditampilkan.

Definisi metrik (deskriptif, BUKAN kriteria):
- masker = `isolate/labels` (id 1): luas / luas frame, komponen 8-arah, porsi komponen terbesar, skor objek (`frames.jsonl`).
- strok dari `contours/frame_%05d.json`: jumlah per tipe; panjang = jumlah panjang segmen polyline (+ segmen penutup bila `closed`).
  Porsi panjang DI LUAR masker = panjang segmen yang titik tengahnya di luar masker yang dilebarkan `target_mask.dilate_px` (4 px; masker
  yang sama dipakai untuk SEBELUM dan SESUDAH).
- komponen peta grup (`stable/groups` ≠ 0, 8-arah) ≥ `vectorize.min_region_area` (800 px); "kecil" = < 5% luas frame (pengganti kasar Q2
  T-305 1b TANPA deteksi wajah).
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"                 # pengaman: tanpa GPU

import cv2  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "work" / "t608"
NILAI = OUT / "nilai_1b"
PROD = ROOT / "work" / "clips" / "klip5"
SCR = ROOT / "work" / "t608_scratch" / "iso" / "work" / "clips" / "klip5"
MP4 = {"SEBELUM": ROOT / "out" / "klip5.mp4", "SESUDAH": ROOT / "work" / "t608_scratch" / "iso" / "out" / "klip5.mp4"}
LAST = 53                                                  # lingkup f0–53
FPS = 24
DILATE_PX = 4                                              # = target_mask.dilate_px (config scratch)
MIN_REGION = 800                                           # = vectorize.min_region_area
SMALL_FRAC = 0.05
CLICK_FRAMES = (0, 22, 40)
SHOTS = ((0, 21, "adegan A (klik f0)"), (22, 59, "adegan B (klik f22, f40)"))
CUT_FALSE = 22                                             # cut palsu menurut label Rio (adegan dipotong di sana)
CUT_NOTE_FRAMES = 3                                        # f22–f24 diberi tanda "ADEGAN BARU"
MASK_BGR = (255, 160, 0)
MASK_ALPHA = 0.45
TITLE_H, INFO_H = 56, 110
MASK_PANEL_H = 880                                         # 720 × 668 → 948 × 880
SIDE_PANEL_H = 580                                         # FOTO 626 × 580 | SEBELUM 626 × 580 | SESUDAH 626 × 580


def jload(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def jsave(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"  -> {p.relative_to(ROOT).as_posix()}")


def file_sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_img(p: Path, flag=cv2.IMREAD_COLOR) -> np.ndarray:
    im = cv2.imdecode(np.fromfile(p, np.uint8), flag)
    if im is None:
        raise SystemExit(f"{p} tidak terbaca")
    return im


def mask(i: int) -> np.ndarray:
    return read_img(SCR / "isolate" / "labels" / f"frame_{i:05d}.png", cv2.IMREAD_UNCHANGED) == 1


def iso_rows() -> dict[int, dict]:
    rows = [json.loads(line) for line in (SCR / "isolate" / "frames.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    return {r["index"]: r for r in rows if r.get("event") == "frame"}


def stats(vals) -> dict:
    a = np.asarray(vals, np.float64)
    return {"n": int(a.size), "median": round(float(np.median(a)), 4), "mean": round(float(a.mean()), 4),
            "min": round(float(a.min()), 4), "max": round(float(a.max()), 4), "jumlah": round(float(a.sum()), 2)}


# ── metrik ─────────────────────────────────────────
def stroke_metrics(doc: dict, near: np.ndarray) -> dict:
    h, w = near.shape
    by_type: dict[str, int] = {}
    total = outside = 0.0
    for s in doc["strokes"]:
        by_type[s["type"]] = by_type.get(s["type"], 0) + 1
        p = np.asarray(s["points"], np.float64)
        if len(p) < 2:
            continue
        seg = np.vstack([p, p[:1]]) if s.get("closed") else p
        a, b = seg[:-1], seg[1:]
        ln = np.hypot(*(b - a).T)
        mid = (a + b) / 2
        xi = np.clip(np.round(mid[:, 0]).astype(int), 0, w - 1)
        yi = np.clip(np.round(mid[:, 1]).astype(int), 0, h - 1)
        total += ln.sum()
        outside += ln[~near[yi, xi]].sum()
    return {"strok": len(doc["strokes"]), "per_tipe": by_type, "panjang_px": round(total, 1),
            "porsi_panjang_luar_masker": round(outside / total, 4) if total else None}


def group_components(gmap: np.ndarray) -> dict:
    fg = (gmap > 0).astype(np.uint8)
    n, _, st, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    areas = st[1:, cv2.CC_STAT_AREA]
    big = areas[areas >= MIN_REGION]
    return {"komponen": int(big.size), "komponen_kecil": int((big < SMALL_FRAC * fg.size).sum()), "fg_porsi": round(float(fg.mean()), 4)}


def cmd_metrik() -> None:
    rows = iso_rows()
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * DILATE_PX + 1, 2 * DILATE_PX + 1))
    per = []
    for i in range(LAST + 1):
        m = mask(i)
        near = cv2.dilate(m.astype(np.uint8), kern) > 0
        n_lab, _, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
        t = rows[i]["targets"]["1"]
        rec = {"index": i, "status": rows[i]["status"], "kind": t["kind"], "skor_objek": t["score_logit"],
               "masker_porsi": round(float(m.mean()), 4), "masker_komponen": int(n_lab - 1),
               "masker_terbesar": round(float(st[1:, cv2.CC_STAT_AREA].max() / m.sum()), 4) if m.any() else None}
        for lab, base in (("sebelum", PROD), ("sesudah", SCR)):
            doc = jload(base / "contours" / f"frame_{i:05d}.json")
            g = read_img(base / "stable" / "groups" / f"frame_{i:05d}.png", cv2.IMREAD_UNCHANGED)
            rec[lab] = {**stroke_metrics(doc, near), **group_components(g)}
        per.append(rec)

    def agg(lab: str, key: str):
        return stats([r[lab][key] for r in per if r[lab][key] is not None])

    types = sorted({t for r in per for lab in ("sebelum", "sesudah") for t in r[lab]["per_tipe"]})
    res = {"definisi": __doc__.split("Definisi metrik (deskriptif, BUKAN kriteria):", 1)[1].strip(), "lingkup": [0, LAST],
           "masker": {"status": {s: sum(r["status"] == s for r in per) for s in sorted({r["status"] for r in per})},
                      "skor_objek": stats([r["skor_objek"] for r in per]), "porsi_frame": stats([r["masker_porsi"] for r in per]),
                      "komponen": stats([r["masker_komponen"] for r in per]),
                      "terbesar": stats([r["masker_terbesar"] for r in per if r["masker_terbesar"] is not None])},
           "banding": {lab: {k: agg(lab, k) for k in ("strok", "panjang_px", "porsi_panjang_luar_masker", "komponen",
                                                       "komponen_kecil", "fg_porsi")}
                       | {"per_tipe_per_frame": {t: round(sum(r[lab]["per_tipe"].get(t, 0) for r in per) / len(per), 2)
                                                 for t in types}}
                       for lab in ("sebelum", "sesudah")},
           "per_adegan": {}, "per_frame": per}
    for lo, hi, name in SHOTS:
        sel = [r for r in per if lo <= r["index"] <= min(hi, LAST)]
        res["per_adegan"][name] = {"frame": [lo, min(hi, LAST)],
                                   "skor_objek": stats([r["skor_objek"] for r in sel]),
                                   "masker_porsi": stats([r["masker_porsi"] for r in sel]),
                                   **{f"{lab}_{k}": stats([r[lab][k] for r in sel if r[lab][k] is not None])
                                      for lab in ("sebelum", "sesudah") for k in ("strok", "panjang_px", "porsi_panjang_luar_masker",
                                                                                  "komponen", "komponen_kecil")}}
    jumps = [(r["index"], round(per[k - 1]["masker_porsi"], 4), round(r["masker_porsi"], 4)) for k, r in enumerate(per)
             if k > 0 and per[k - 1]["masker_porsi"] > 0 and abs(r["masker_porsi"] / per[k - 1]["masker_porsi"] - 1) > 0.15]
    res["lompatan_luas_masker_gt_15pct"] = jumps
    jsave(OUT / "iso2_metrik.json", res)
    b = res["banding"]
    print(f"masker f0–53: {res['masker']['status']}, skor {res['masker']['skor_objek']}, porsi {res['masker']['porsi_frame']}")
    for k in ("strok", "panjang_px", "porsi_panjang_luar_masker", "komponen", "komponen_kecil", "fg_porsi"):
        print(f"  {k:28s} SEBELUM med {b['sebelum'][k]['median']:>10} | SESUDAH med {b['sesudah'][k]['median']:>10}")
    print("  per tipe per frame SEBELUM", b["sebelum"]["per_tipe_per_frame"])
    print("  per tipe per frame SESUDAH", b["sesudah"]["per_tipe_per_frame"])
    print("  lompatan luas masker > 15%:", jumps)


# ── bahan ──────────────────────────────────────────
def fit_h(img: np.ndarray, h: int) -> np.ndarray:
    return cv2.resize(img, (round(img.shape[1] * h / img.shape[0] / 2) * 2, h), interpolation=cv2.INTER_AREA)


def text_bar(width: int, height: int, lines: list[str], scale: float, thick: int, bg: int = 255,
             color=(0, 0, 0)) -> np.ndarray:
    bar = np.full((height, width, 3), bg, np.uint8)
    step = height // (len(lines) + 1)
    for k, t in enumerate(lines, 1):
        (tw, th), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_DUPLEX, scale, thick)
        cv2.putText(bar, t, (max(8, (width - tw) // 2), k * step + th // 2), cv2.FONT_HERSHEY_DUPLEX, scale, color, thick, cv2.LINE_AA)
    return bar


def scene_text(i: int) -> str:
    name = next(n for lo, hi, n in SHOTS if lo <= i <= hi)
    tag = "  [FRAME KLIK]" if i in CLICK_FRAMES else ""
    if CUT_FALSE <= i < CUT_FALSE + CUT_NOTE_FRAMES:
        tag += "  ADEGAN BARU (cut palsu)"
    return name + tag


def info_bar(width: int, i: int, n_total: int) -> np.ndarray:
    sec = f"{i / FPS:.2f}".replace(".", ",")
    return text_bar(width, INFO_H, [f"frame {i}  /  {LAST}        detik {sec}", scene_text(i)], 1.25, 3, bg=225)


def overlay(bgr: np.ndarray, m: np.ndarray) -> np.ndarray:
    img = bgr.copy()
    if m.any():
        col = np.zeros_like(img)
        col[:] = MASK_BGR
        img[m] = (img[m] * (1 - MASK_ALPHA) + col[m] * MASK_ALPHA).astype(np.uint8)
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, cnts, -1, MASK_BGR, 2, cv2.LINE_AA)
    return img


def read_video(path: Path, n: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    out = []
    while len(out) < n:
        ok, fr = cap.read()
        if not ok:
            break
        out.append(fr)
    cap.release()
    if len(out) != n:
        raise SystemExit(f"{path}: hanya {len(out)} frame terbaca (butuh {n})")
    return out


def encoder(path: Path, size: tuple[int, int]) -> subprocess.Popen:
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{size[0]}x{size[1]}",
           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", "-r", str(FPS), str(path)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def write_pair(stem: str, frames) -> dict:
    enc, size, k = None, None, 0
    for img in frames:
        img = np.ascontiguousarray(img)
        if enc is None:
            size = (img.shape[1], img.shape[0])
            enc = (encoder(NILAI / f"{stem}.mp4", size), encoder(NILAI / f"{stem}_lambat_0.5x.mp4", size))
        raw = img.tobytes()
        enc[0].stdin.write(raw)
        enc[1].stdin.write(raw)
        enc[1].stdin.write(raw)
        k += 1
    for p in enc:
        p.stdin.close()
        if p.wait() != 0:
            raise SystemExit("ffmpeg gagal")
    return {"frames": k, "size": list(size)}


def cmd_bahan() -> None:
    NILAI.mkdir(parents=True, exist_ok=True)
    n = LAST + 1
    photos = [read_img(SCR / "frames" / f"frame_{i:05d}.png") for i in range(n)]
    prod_photo0 = read_img(PROD / "frames" / "frame_00000.png")
    if not np.array_equal(prod_photo0, photos[0]):
        raise SystemExit("frame scratch ≠ produksi")

    def masker_frames():
        for i in range(n):
            a = fit_h(overlay(photos[i], mask(i)), MASK_PANEL_H)
            yield np.vstack([text_bar(a.shape[1], TITLE_H, ["FOTO + MASKER SAM (biru = target)"], 1.3, 3),
                             a, info_bar(a.shape[1], i, n)])

    vids = {k: read_video(p, n) for k, p in MP4.items()}

    def side_frames():
        for i in range(n):
            panels = [("FOTO", photos[i]), ("SEBELUM (produksi)", vids["SEBELUM"][i]), ("SESUDAH (masker)", vids["SESUDAH"][i])]
            row = [fit_h(img, SIDE_PANEL_H) for _, img in panels]
            top = np.hstack([text_bar(r.shape[1], TITLE_H, [t], 1.1, 2) for (t, _), r in zip(panels, row)])
            body = np.hstack(row)
            yield np.vstack([top, body, info_bar(body.shape[1], i, n)])

    res = {"masker_f0-53": write_pair("masker_f0-53", masker_frames()),
           "berdampingan_f0-53": write_pair("berdampingan_f0-53", side_frames()),
           "sumber": {k: {"path": p.relative_to(ROOT).as_posix(), "sha256": file_sha(p)} for k, p in MP4.items()}}
    print(res)
    jsave(OUT / "iso2_bahan_buat.json", res)


def cmd_cek() -> None:
    res = {}
    for p in sorted(NILAI.glob("*_f0-53*.mp4")):
        cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
               "stream=codec_name,pix_fmt,width,height,avg_frame_rate,nb_read_frames:format=duration,size", "-of", "json", str(p)]
        d = json.loads(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout)
        res[p.name] = {**d["streams"][0], **d["format"], "sha256": file_sha(p)}
        print(p.name, {k: v for k, v in res[p.name].items() if k != "sha256"})
    jsave(OUT / "iso2_bahan.json", res)


def main(argv: list[str]) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    cmds = {"metrik": cmd_metrik, "bahan": cmd_bahan, "cek": cmd_cek}
    if len(argv) != 1 or argv[0] not in cmds:
        print(__doc__)
        return 2
    cmds[argv[0]]()
    if "torch" in sys.modules:
        raise SystemExit("torch ikut terimpor — tidak boleh")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
