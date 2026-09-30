"""Uji kelayakan Sapiens2 lokal T-102c (D-009): seg 0.4B + pointmap 0.4B di GTX 1650 Ti.

Alat sekali pakai, BUKAN modul pipeline. Jalankan dari root repo setelah T-101 (ingest):
    venv/Scripts/python.exe scripts/sapiens2_probe.py [--force] [--force-bench]

Input : work/meta.json + work/frames/frame_%05d.png (+ work/ab_t102a/masks_mediapipe/ untuk cross_iou)
Output: work/t102c/ -> bench.json, seg/frame_%05d.png (class map uint8 0-28),
        pointmap/frame_%05d.npy (Z float16), infer_seg.json, infer_pointmap.json,
        metrics_per_frame.csv, metrics_summary.json, side_by_side.mp4, frames_grid.png

Alur:
1. VRAM terpakai di luar proses dicatat lewat nvidia-smi SEBELUM CUDA diinisialisasi.
2. Smoke benchmark: N frame (tersebar merata) per model x {fp16, fp32}, hanya satu model di GPU
   pada satu waktu. OOM dicatat sebagai hasil, kombinasi lain tetap jalan. + 3 frame seg di CPU.
3. Precision terpilih: fp16 kalau semua run fp16 bebas NaN/inf dan kesamaan kelas seg
   fp16 vs fp32 >= --agree-min; selain itu fp32. Kalau precision itu OOM untuk sebuah model,
   dipakai precision lain yang lolos (bebas NaN/inf).
4. Full run semua frame, lalu metrik + visual.

Preprocessing = image processor default checkpoint (1024x768): seg STRETCH (do_pad=false),
pointmap resize-jaga-rasio + PAD (do_pad=true); nilai aktual dicatat di bench.json.
Waktu "infer" = forward + post-process di GPU (torch.cuda.synchronize sebelum ukur);
preprocess (CPU) dicatat terpisah; "total" = baca frame s/d tulis output.

Definisi metrik foreground (kelas != 0) diimpor dari scripts/ab_segment.py supaya SAMA persis.
label_agreement_prev = % piksel yang foreground di frame ini DAN frame sebelumnya yang kelasnya
sama (irisan saja; perubahan siluet sudah diukur iou_prev).
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ab_segment as ab  # noqa: E402  (definisi metrik & util yang sama dengan T-102a)

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "seg": "facebook/sapiens2-seg-0.4b",
    "pointmap": "facebook/sapiens2-pointmap-0.4b",
}
PRECISIONS = ["fp16", "fp32"]  # tanpa bf16: Turing (P-005)
NUM_CLASSES = 29
# config.json checkpoint HF hanya berisi LABEL_0..28. Nama kelas resmi (facebookresearch/sapiens2
# docs/SEG.md, commit tercatat di file) disimpan sekali di sini; dipakai kalau id2label generik.
CLASSES_FILE = ROOT / "src" / "rotoscope" / "data" / "sapiens2_classes.json"
ZONES = [(1, 7), (236, 241)]  # zona gagal T-102a (nomor frame)

# Visualisasi (bukan parameter pipeline)
BG_BGR = (40, 40, 40)
LINE_BG = 255
LINE_FG = 0
DEPTH_OUTSIDE = 225  # abu-abu terang di luar foreground pada panel gradien kedalaman
DEPTH_NORM_PCT = 99  # normalisasi gradien per frame: persentil di dalam foreground


# --- util ------------------------------------------------------------------
def nvidia_smi() -> dict:
    cmd = ["nvidia-smi", "--query-gpu=name,driver_version,memory.used,memory.total",
           "--format=csv,noheader,nounits"]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as e:
        return {"error": str(e)}
    name, driver, used, total = [s.strip() for s in out.splitlines()[0].split(",")]
    return {"name": name, "driver": driver, "used_mib": int(used), "total_mib": int(total)}


def palette(n: int) -> np.ndarray:
    """Warna BGR deterministik per kelas (golden-ratio hue). Kelas 0 = latar gelap."""
    hsv = np.zeros((n, 1, 3), np.uint8)
    for i in range(n):
        hsv[i, 0] = (int((i * 0.618034 % 1.0) * 179), 200, 230)
    pal = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[:, 0]
    pal[0] = BG_BGR
    return pal


def group_lut(id2label: dict[int, str], groups: dict[str, list[str]]) -> tuple[np.ndarray, dict]:
    """LUT kelas -> id grup. 0 = background. Kelas di luar grup mana pun jadi grup sendiri."""
    name2id = {v: k for k, v in id2label.items()}
    unknown = [n for names in groups.values() for n in names if n not in name2id]
    if unknown:
        ab.fail(f"nama kelas di file grup tidak ada di id2label: {unknown}. "
                f"Tersedia: {[id2label[i] for i in sorted(id2label)]}")
    lut = np.zeros(256, np.uint8)
    info = {}
    gid = 1
    for g, names in groups.items():
        for n in names:
            lut[name2id[n]] = gid
        info[g] = gid
        gid += 1
    for c in sorted(id2label):
        if c != 0 and not any(id2label[c] in names for names in groups.values()):
            print(f"  PERINGATAN: kelas {c} {id2label[c]} tidak masuk grup -> grup sendiri")
            lut[c] = gid
            info[f"_{id2label[c]}"] = gid
            gid += 1
    return lut, info


def boundary(label_map: np.ndarray) -> np.ndarray:
    """True di piksel yang punya tetangga 3x3 berlabel beda (termasuk batas ke background)."""
    k = np.ones((3, 3), np.uint8)
    lm = label_map.astype(np.uint8)
    return cv2.dilate(lm, k) != cv2.erode(lm, k)


def resolve_id2label(id2label: dict[int, str], path: Path = CLASSES_FILE) -> dict[int, str]:
    """id2label model; kalau generik (LABEL_n) -> nama dari file kelas lokal (tanpa jaringan).

    Self-check: jumlah kelas file = num_labels model.
    """
    if not all(v == f"LABEL_{k}" for k, v in id2label.items()):
        return id2label
    classes = json.loads(path.read_text(encoding="utf-8"))["classes"]
    if len(classes) != len(id2label):
        ab.fail(f"self-check gagal: {path.name} punya {len(classes)} kelas, num_labels model {len(id2label)}")
    print(f"id2label checkpoint generik (LABEL_n) -> nama kelas dari {path.name} ({len(classes)} = num_labels)")
    return dict(enumerate(classes))


def self_check() -> None:
    def check(cond: bool, msg: str) -> None:
        if not cond:
            ab.fail(f"self-check gagal: {msg}")

    m = np.zeros((20, 20), np.uint8)
    m[5:15, 5:10] = 1   # kelas 1 dan 2 -> grup A, kelas 3 -> grup B
    m[5:15, 10:12] = 2
    m[5:15, 12:15] = 3
    lut = np.zeros(256, np.uint8)
    lut[[1, 2]] = 1
    lut[3] = 2
    b = boundary(lut[m])
    check(not b[10, 10], "batas di dalam satu grup ikut tergambar")
    check(b[10, 11] or b[10, 12], "batas antar grup tidak tergambar")
    check(b[4, 7] or b[5, 7], "siluet luar tidak tergambar")
    check(not b[0, 0], "background jauh dari subjek ikut tergambar")
    check(ab.iou(m > 0, m > 0) == 1.0, "iou(m, m) != 1")
    print("self-check OK")


def pct_stats(values: list[float]) -> dict:
    return {"mean": statistics.fmean(values), "median": statistics.median(values),
            "p95": float(np.percentile(values, 95)), "min": min(values), "max": max(values)}


# --- model -----------------------------------------------------------------
class Runner:
    """Satu model Sapiens2 pada satu device + dtype."""

    def __init__(self, kind: str, precision: str, device: str):
        import torch
        import transformers
        from transformers import AutoImageProcessor

        cls = (transformers.Sapiens2ForSemanticSegmentation if kind == "seg"
               else transformers.Sapiens2ForPointmapEstimation)
        self.torch = torch
        self.kind, self.precision, self.device = kind, precision, device
        self.dtype = torch.float16 if precision == "fp16" else torch.float32
        self.processor = AutoImageProcessor.from_pretrained(MODELS[kind])
        t0 = time.perf_counter()
        model = cls.from_pretrained(MODELS[kind], dtype=self.dtype)
        self.model = model.to(device).eval()
        self.load_s = time.perf_counter() - t0
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()} if kind == "seg" else {}

    def sync(self) -> None:
        if self.device == "cuda":
            self.torch.cuda.synchronize()

    def infer(self, rgb: np.ndarray) -> tuple[np.ndarray, dict]:
        """RGB HxWx3 -> (class map uint8 HxW | Z float32 HxW, info waktu + finite)."""
        torch = self.torch
        h, w = rgb.shape[:2]
        t0 = time.perf_counter()
        inputs = self.processor(images=rgb, return_tensors="pt")
        pixel = inputs["pixel_values"].to(self.device, self.dtype)
        self.sync()
        t1 = time.perf_counter()
        with torch.inference_mode():
            out = self.model(pixel_values=pixel)
            if self.kind == "seg":
                finite = bool(torch.isfinite(out.logits).all())
                res = self.processor.post_process_semantic_segmentation(out, target_sizes=[(h, w)])[0]
                res = res.to(torch.uint8)
            else:
                finite = bool(torch.isfinite(out.pointmaps).all())
                if out.scales is not None:
                    finite = finite and bool(torch.isfinite(out.scales).all())
                pm = self.processor.post_process_pointmap_estimation(out, source_sizes=[(h, w)])[0]["pointmap"]
                res = pm[2].float()
            self.sync()
        t2 = time.perf_counter()
        arr = res.cpu().numpy()
        return arr, {"pre_s": t1 - t0, "infer_s": t2 - t1, "finite": finite,
                     "input_hw": list(pixel.shape[-2:])}

    def close(self) -> None:
        del self.model
        gc.collect()
        if self.device == "cuda":
            self.torch.cuda.empty_cache()


def is_oom(e: BaseException) -> bool:
    import torch
    return isinstance(e, torch.OutOfMemoryError) or "out of memory" in str(e).lower()


def processor_info(kind: str) -> dict:
    from transformers import AutoImageProcessor
    p = AutoImageProcessor.from_pretrained(MODELS[kind])
    return {"size": dict(p.size), "do_pad": bool(p.do_pad), "resample": str(p.resample),
            "resize": "pad (jaga rasio)" if p.do_pad else "stretch"}


# --- benchmark -------------------------------------------------------------
def bench_one(kind: str, precision: str, device: str, frames: list[np.ndarray],
              warmup: int) -> tuple[dict, list[np.ndarray]]:
    import torch
    rec = {"model": MODELS[kind], "precision": precision, "device": device, "n_frames": len(frames)}
    runner = None
    outputs = []
    try:
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        runner = Runner(kind, precision, device)
        rec["load_s"] = runner.load_s
        rec["attn_implementation"] = runner.model.config._attn_implementation
        infos = []
        for rgb in frames:
            arr, info = runner.infer(rgb)
            outputs.append(arr)
            infos.append(info)
        timed = [i["infer_s"] for i in infos[warmup:]] or [i["infer_s"] for i in infos]
        rec.update({
            "status": "ok",
            "all_finite": all(i["finite"] for i in infos),
            "n_nonfinite_frames": sum(not i["finite"] for i in infos),
            "input_hw": infos[0]["input_hw"],
            "infer_s": pct_stats(timed),
            "pre_s_mean": statistics.fmean(i["pre_s"] for i in infos),
            "warmup_infer_s": [round(i["infer_s"], 3) for i in infos[:warmup]],
        })
    except Exception as e:  # noqa: BLE001 - OOM adalah hasil, bukan crash
        if device == "cuda" and is_oom(e):
            rec.update({"status": "oom", "error": str(e).splitlines()[0][:300]})
            outputs = []
        else:
            raise
    finally:
        if device == "cuda":
            rec["peak_alloc_mib"] = torch.cuda.max_memory_allocated() / 2**20
            rec["peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 2**20
            rec["nvidia_smi_after"] = nvidia_smi()
        if runner is not None:
            runner.close()
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    status = rec["status"]
    extra = (f"infer mean {rec['infer_s']['mean']:.3f} s, peak alloc {rec.get('peak_alloc_mib', 0):.0f} MiB, "
             f"finite {rec['all_finite']}" if status == "ok" else rec["error"])
    print(f"  {kind} {precision} {device}: {status} - {extra}")
    return rec, outputs


def run_bench(frames_dir: Path, names: list[str], args, external: dict) -> dict:
    idx = np.linspace(0, len(names) - 1, args.bench_frames).round().astype(int).tolist()
    frames = [ab.read_rgb(frames_dir / names[i]) for i in idx]
    bench = {"external_vram_before": external, "bench_frames": [names[i] for i in idx],
             "warmup": args.warmup, "processor": {k: processor_info(k) for k in MODELS}, "runs": {}}
    outs = {}
    for kind in MODELS:
        for prec in PRECISIONS:
            key = f"{kind}_{prec}"
            print(f"\n[bench] {key} ...")
            bench["runs"][key], outs[key] = bench_one(kind, prec, "cuda", frames, args.warmup)

    # fp16 vs fp32
    cmp = {}
    if outs.get("seg_fp16") and outs.get("seg_fp32"):
        agree = [float((a == b).mean()) for a, b in zip(outs["seg_fp16"], outs["seg_fp32"])]
        cmp["seg_class_agreement"] = {"mean": statistics.fmean(agree), "min": min(agree)}
    if outs.get("pointmap_fp16") and outs.get("pointmap_fp32"):
        ref_seg = outs.get("seg_fp32") or outs.get("seg_fp16")
        rel = []
        for k, (z16, z32) in enumerate(zip(outs["pointmap_fp16"], outs["pointmap_fp32"])):
            fg = ref_seg[k] != 0 if ref_seg else np.ones_like(z32, bool)
            fg &= np.isfinite(z16) & np.isfinite(z32) & (np.abs(z32) > 1e-6)
            if fg.any():
                rel.append(float(np.mean(np.abs(z16[fg] - z32[fg]) / np.abs(z32[fg]))))
        if rel:
            cmp["pointmap_rel_z_diff"] = {"mean": statistics.fmean(rel), "max": max(rel),
                                          "mask": "foreground seg fp32" if outs.get("seg_fp32") else "seg fp16"}
    bench["fp16_vs_fp32"] = cmp

    # CPU cadangan: seg fp32
    print(f"\n[bench] seg fp32 CPU ({args.cpu_frames} frame) ...")
    cpu, _ = bench_one("seg", "fp32", "cpu", frames[:args.cpu_frames], warmup=0)
    bench["runs"]["seg_fp32_cpu"] = cpu

    bench["choice"] = choose_precision(bench, args.agree_min)
    return bench


def choose_precision(bench: dict, agree_min: float) -> dict:
    runs = bench["runs"]

    def ok(key: str) -> bool:
        r = runs.get(key, {})
        return r.get("status") == "ok" and r.get("all_finite", False)

    reasons = []
    fp16_runs = [k for k in ("seg_fp16", "pointmap_fp16") if runs.get(k, {}).get("status") == "ok"]
    fp16_finite = all(runs[k]["all_finite"] for k in fp16_runs) and bool(fp16_runs)
    agree = bench["fp16_vs_fp32"].get("seg_class_agreement", {}).get("mean")
    if fp16_finite and agree is not None and agree >= agree_min:
        base = "fp16"
        reasons.append(f"fp16 bebas NaN/inf dan kesamaan kelas seg {agree:.4%} >= {agree_min:.0%}")
    else:
        base = "fp32"
        if not fp16_finite:
            reasons.append("fp16 menghasilkan NaN/inf atau tidak ada run fp16 yang lolos")
        elif agree is None:
            reasons.append("kesamaan kelas tidak bisa dihitung (seg fp16/fp32 tidak lengkap)")
        else:
            reasons.append(f"kesamaan kelas seg {agree:.4%} < {agree_min:.0%}")
    choice = {"base": base}
    for kind in MODELS:
        other = "fp32" if base == "fp16" else "fp16"
        if ok(f"{kind}_{base}"):
            choice[kind] = base
        elif ok(f"{kind}_{other}"):
            choice[kind] = other
            reasons.append(f"{kind}: {base} tidak lolos ({runs[f'{kind}_{base}'].get('status')}) -> {other}")
        else:
            choice[kind] = None
            reasons.append(f"{kind}: tidak ada precision yang lolos -> full run dilewati")
    choice["reasons"] = reasons
    return choice


# --- full run --------------------------------------------------------------
def out_path(out: Path, kind: str, name: str) -> Path:
    stem = Path(name).stem
    return out / kind / (f"{stem}.png" if kind == "seg" else f"{stem}.npy")


def complete(out: Path, kind: str, names: list[str]) -> bool:
    return (out / f"infer_{kind}.json").is_file() and all(out_path(out, kind, n).is_file() for n in names)


def full_run(kind: str, precision: str, frames_dir: Path, out: Path, names: list[str]) -> None:
    print(f"\n[full] {kind} {precision} - {len(names)} frame ...")
    (out / kind).mkdir(parents=True, exist_ok=True)
    runner = Runner(kind, precision, "cuda")
    recs = []
    try:
        for i, n in enumerate(names):
            t0 = time.perf_counter()
            arr, info = runner.infer(ab.read_rgb(frames_dir / n))
            p = out_path(out, kind, n)
            if kind == "seg":
                cv2.imwrite(str(p), arr)
            else:
                np.save(p, arr.astype(np.float16))
            info["total_s"] = time.perf_counter() - t0
            recs.append(info)
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(names)}")
        id2label = runner.id2label
    finally:
        runner.close()
    record = {"model": MODELS[kind], "precision": precision, "load_s": runner.load_s,
              "frames": recs, "id2label": id2label}
    (out / f"infer_{kind}.json").write_text(json.dumps(record, indent=1), encoding="utf-8")


# --- metrik ----------------------------------------------------------------
def frame_no(name: str) -> int:
    return int(Path(name).stem.split("_")[-1])


def read_seg(out: Path, name: str) -> np.ndarray:
    m = cv2.imread(str(out_path(out, "seg", name)), cv2.IMREAD_UNCHANGED)
    if m is None:
        ab.fail(f"gagal membaca class map {name}")
    return m


def compute_metrics(out: Path, names: list[str], mp_dir: Path, has_pm: bool, args,
                    seg_dir: Path | None = None) -> tuple[list[dict], np.ndarray]:
    """seg_dir: folder class map lain (mis. model lain) dengan definisi metrik yang sama; default out/seg."""
    rows, prev = [], None
    counts = np.zeros(NUM_CLASSES, np.int64)
    for n in names:
        seg = read_seg(out, n) if seg_dir is None else cv2.imread(str(seg_dir / n), cv2.IMREAD_UNCHANGED)
        if seg is None:
            ab.fail(f"gagal membaca class map {n}")
        fg = seg != 0
        counts += np.bincount(seg.ravel(), minlength=NUM_CLASSES)[:NUM_CLASSES]
        row = {"frame": n, "area_ratio": round(ab.area_ratio(fg), 5),
               "iou_prev": round(ab.iou(fg, prev != 0), 5) if prev is not None else "",
               "big_blobs": ab.big_blob_count(fg, args.blob_min)}
        if prev is not None:
            both = fg & (prev != 0)
            row["label_agreement_prev"] = round(float((seg[both] == prev[both]).mean()), 5) if both.any() else ""
        else:
            row["label_agreement_prev"] = ""
        mp = mp_dir / n
        row["cross_iou_mp"] = round(ab.iou(fg, ab.read_mask(mp)), 5) if mp.is_file() else ""
        if has_pm:
            z = np.load(out_path(out, "pointmap", n)).astype(np.float32)
            row["z_nonfinite"] = int((~np.isfinite(z)).sum())
            zf = z[fg & np.isfinite(z)]
            row["z_fg_median"] = round(float(np.median(zf)), 4) if zf.size else ""
        rows.append(row)
        prev = seg
    return rows, counts


def summarize_metrics(rows: list[dict], counts: np.ndarray, id2label: dict, args) -> dict:
    ious = [r["iou_prev"] for r in rows if r["iou_prev"] != ""]
    areas = [r["area_ratio"] for r in rows]
    agree = [r["label_agreement_prev"] for r in rows if r["label_agreement_prev"] != ""]
    cross = [r["cross_iou_mp"] for r in rows if r["cross_iou_mp"] != ""]
    fg_total = counts[1:].sum()
    s = {
        "iou_prev_mean": statistics.fmean(ious), "iou_prev_median": statistics.median(ious),
        "iou_prev_min": min(ious), "n_iou_prev_below_min": sum(v < args.iou_min for v in ious),
        "n_area_below_min": sum(v < args.area_min for v in areas),
        "n_area_above_max": sum(v > args.area_max for v in areas),
        "n_multi_big_blob": sum(r["big_blobs"] > 1 for r in rows),
        "label_agreement_prev": ab.stats_block(agree),
        "cross_iou_mp": ab.stats_block(cross),
        "cross_iou_zones": {},
        "class_share": [],
    }
    for a, b in ZONES:
        s["cross_iou_zones"][f"{a}-{b}"] = {r["frame"]: r["cross_iou_mp"] for r in rows
                                           if a <= frame_no(r["frame"]) <= b}
    for c in range(NUM_CLASSES):
        s["class_share"].append({"id": c, "name": id2label.get(c, str(c)),
                                 "share_all": float(counts[c] / counts.sum()),
                                 "share_fg": float(counts[c] / fg_total) if c and fg_total else None})
    if "z_nonfinite" in rows[0]:
        s["z_nonfinite_total"] = sum(r["z_nonfinite"] for r in rows)
    return s


def timing_summary(out: Path, kind: str, warmup: int) -> dict:
    rec = json.loads((out / f"infer_{kind}.json").read_text(encoding="utf-8"))
    fr = rec["frames"][warmup:] or rec["frames"]
    return {"precision": rec["precision"], "infer_s": pct_stats([f["infer_s"] for f in fr]),
            "total_s": pct_stats([f["total_s"] for f in fr]),
            "n_nonfinite_frames": sum(not f["finite"] for f in rec["frames"])}


# --- visualisasi -----------------------------------------------------------
def depth_gradient(z: np.ndarray, fg: np.ndarray) -> np.ndarray:
    """|grad Z| di foreground ter-erode, dinormalisasi per frame (persentil), TANPA threshold."""
    inner = cv2.erode(fg.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    z = np.where(np.isfinite(z), z, 0).astype(np.float32)
    gx = cv2.Sobel(z, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(z, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    img = np.full(z.shape, DEPTH_OUTSIDE, np.uint8)
    if inner.any():
        ref = np.percentile(mag[inner], DEPTH_NORM_PCT)
        v = np.clip(mag / ref, 0, 1) if ref > 0 else np.zeros_like(mag)
        img[inner] = (255 * (1 - v[inner])).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def render_row(frames_dir: Path, out: Path, name: str, pal: np.ndarray, lut: np.ndarray,
               has_pm: bool, line_px: int) -> np.ndarray:
    stem = Path(name).stem
    bgr = cv2.imread(str(frames_dir / name), cv2.IMREAD_COLOR)
    seg = read_seg(out, name)
    fg = seg != 0

    p0 = bgr.copy()
    ab.draw_label(p0, ["asli", stem])
    p1 = pal[seg]
    ab.draw_label(p1, ["kelas sapiens2-seg", stem])
    lines = boundary(lut[seg])
    if line_px > 1:
        lines = cv2.dilate(lines.astype(np.uint8), np.ones((line_px, line_px), np.uint8)).astype(bool)
    p2 = np.full_like(bgr, LINE_BG)
    p2[lines] = LINE_FG
    ab.draw_label(p2, ["garis grup", stem])
    if has_pm:
        p3 = depth_gradient(np.load(out_path(out, "pointmap", name)).astype(np.float32), fg)
        ab.draw_label(p3, ["|grad Z| pointmap", stem])
    else:
        p3 = np.full_like(bgr, DEPTH_OUTSIDE)
        ab.draw_label(p3, ["pointmap tidak tersedia", stem])
    return np.hstack([p0, p1, p2, p3])


def write_video(frames_dir: Path, out: Path, names: list[str], render, fps: int) -> Path:
    path = out / "side_by_side.mp4"
    first = ab.even(render(names[0]))
    h, w = first.shape[:2]
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    proc.stdin.write(first.tobytes())
    for n in names[1:]:
        proc.stdin.write(ab.even(render(n)).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        ab.fail("ffmpeg gagal meng-encode side_by_side.mp4")
    return path


def write_grid(out: Path, names: list[str], grid_frames: list[int], render, scale: float) -> Path:
    by_no = {frame_no(n): n for n in names}
    missing = [f for f in grid_frames if f not in by_no]
    if missing:
        print(f"  PERINGATAN: frame grid tidak ada: {missing}")
    tiles = [cv2.resize(render(by_no[f]), None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
             for f in grid_frames if f in by_no]
    path = out / "frames_grid.png"
    cv2.imwrite(str(path), np.vstack(tiles))
    return path


# --- laporan ---------------------------------------------------------------
def print_report(summary: dict, mp_ref: dict | None) -> None:
    b = summary["bench"]
    print("\n=== Setup ===")
    for k, v in summary["setup"].items():
        print(f"  {k}: {v}")
    ext = b["external_vram_before"]
    print(f"  VRAM terpakai di luar proses (sebelum CUDA init): {ext.get('used_mib')} / {ext.get('total_mib')} MiB")

    print("\n=== Benchmark (s/frame = forward + post-process, tanpa warm-up) ===")
    head = f"{'run':<20}{'status':>7}{'mean':>8}{'med':>8}{'p95':>8}{'alloc MiB':>11}{'resv MiB':>10}{'finite':>8}"
    print(head + "\n" + "-" * len(head))
    for key, r in b["runs"].items():
        if r["status"] == "ok":
            t = r["infer_s"]
            print(f"{key:<20}{'ok':>7}{t['mean']:>8.3f}{t['median']:>8.3f}{t['p95']:>8.3f}"
                  f"{r.get('peak_alloc_mib', float('nan')):>11.0f}{r.get('peak_reserved_mib', float('nan')):>10.0f}"
                  f"{str(r['all_finite']):>8}")
        else:
            print(f"{key:<20}{r['status']:>7}  {r.get('error', '')[:80]}")
    for k, v in b["fp16_vs_fp32"].items():
        print(f"  {k}: {v}")
    print(f"  precision: {b['choice']}")

    print("\n=== Full run ===")
    for kind, t in summary["timing"].items():
        print(f"  {kind} {t['precision']}: infer mean {t['infer_s']['mean']:.3f} s, "
              f"total mean {t['total_s']['mean']:.3f} s/frame, non-finite {t['n_nonfinite_frames']}")
    for k, v in summary["estimate_360_frames_s"].items():
        print(f"  estimasi 360 frame ({k}): {v:.0f} s")

    m = summary["metrics"]
    print("\n=== Foreground (kelas != 0) vs MediaPipe T-102a ===")
    cols = ["iou_prev_mean", "iou_prev_median", "iou_prev_min", "n_iou_prev_below_min",
            "n_area_below_min", "n_area_above_max", "n_multi_big_blob"]
    print(f"{'backend':<12}" + "".join(f"{c.replace('n_', '').replace('iou_prev_', 'iou_'):>16}" for c in cols))
    rows = [("sapiens2", m)] + ([("mediapipe", mp_ref)] if mp_ref else [])
    for name, s in rows:
        print(f"{name:<12}" + "".join(f"{s[c]:>16.3f}" if isinstance(s[c], float) else f"{s[c]:>16}" for c in cols))
    print(f"  label_agreement_prev: {m['label_agreement_prev']}")
    print(f"  cross_iou vs mediapipe: {m['cross_iou_mp']}")
    for z, vals in m["cross_iou_zones"].items():
        print(f"  zona {z}: " + ", ".join(f"{Path(k).stem[-5:]}={v}" for k, v in vals.items()))

    print("\n=== Pangsa piksel foreground per kelas ===")
    for c in sorted(m["class_share"][1:], key=lambda c: -(c["share_fg"] or 0)):
        print(f"  {c['id']:>2} {c['name']:<18} {c['share_fg']:.2%}")
    print(f"\nTotal waktu: {summary['total_wall_s']:.1f} s")


# --- main ------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--work", type=Path, default=ROOT / "work")
    p.add_argument("--out", type=Path, default=ROOT / "work" / "t102c")
    p.add_argument("--mp-masks", type=Path, default=ROOT / "work" / "ab_t102a" / "masks_mediapipe")
    p.add_argument("--groups", type=Path, default=ROOT / "scripts" / "sapiens2_groups.json")
    p.add_argument("--bench-frames", type=int, default=20)
    p.add_argument("--cpu-frames", type=int, default=3)
    p.add_argument("--warmup", type=int, default=3, help="frame awal yang dibuang dari statistik waktu")
    p.add_argument("--agree-min", type=float, default=0.99, help="kesamaan kelas minimum fp16 vs fp32")
    # Threshold QC sama dengan ab_segment.py (tabel QC docs/01 stage [2])
    p.add_argument("--iou-min", type=float, default=0.55)
    p.add_argument("--area-min", type=float, default=0.03)
    p.add_argument("--area-max", type=float, default=0.70)
    p.add_argument("--blob-min", type=float, default=0.05)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--grid-frames", default="1,4,73,78,82,87,92,100,167,236,240")
    p.add_argument("--grid-scale", type=float, default=0.5)
    p.add_argument("--line-px", type=int, default=2, help="tebal garis preview")
    p.add_argument("--force", action="store_true", help="full run ulang walau output sudah lengkap")
    p.add_argument("--force-bench", action="store_true", help="benchmark ulang walau bench.json ada")
    args = p.parse_args()

    t_start = time.perf_counter()
    self_check()
    external = nvidia_smi()  # SEBELUM import torch / CUDA init
    print(f"nvidia-smi sebelum CUDA init: {external}")

    meta_path = args.work / "meta.json"
    if not meta_path.is_file():
        ab.fail(f"{meta_path} tidak ada - jalankan ingest (T-101) dulu")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    frames_dir = args.work / "frames"
    names = ab.frame_names(meta)
    missing = [n for n in names if not (frames_dir / n).is_file()]
    if missing:
        ab.fail(f"{len(missing)} frame hilang di {frames_dir}, mis. {missing[0]}")
    groups = {k: v for k, v in json.loads(args.groups.read_text(encoding="utf-8")).items()
              if not k.startswith("_")}
    args.out.mkdir(parents=True, exist_ok=True)

    import torch
    import transformers
    if not torch.cuda.is_available():
        ab.fail("CUDA tidak tersedia")
    setup = {"gpu": torch.cuda.get_device_name(0), "driver": external.get("driver"),
             "torch": torch.__version__, "cuda": torch.version.cuda,
             "transformers": transformers.__version__, "arch_list": torch.cuda.get_arch_list()}

    bench_path = args.out / "bench.json"
    if bench_path.is_file() and not args.force_bench:
        print("\n[bench] bench.json ada -> dilewati (--force-bench untuk ulang)")
        bench = json.loads(bench_path.read_text(encoding="utf-8"))
    else:
        bench = run_bench(frames_dir, names, args, external)
        bench_path.write_text(json.dumps(bench, indent=2), encoding="utf-8")
    choice = bench["choice"]
    print(f"precision: {choice}")
    if choice["seg"] is None:
        ab.fail("seg tidak lolos di precision mana pun - full run tidak mungkin")

    for kind in MODELS:
        prec = choice[kind]
        if prec is None:
            continue
        if complete(args.out, kind, names) and not args.force:
            old = json.loads((args.out / f"infer_{kind}.json").read_text(encoding="utf-8"))["precision"]
            print(f"\n[full] {kind} lengkap ({old}) -> dilewati (--force untuk ulang)")
            if old != prec:
                print(f"  PERINGATAN: output {kind} dibuat dengan {old}, bukan {prec}")
        else:
            full_run(kind, prec, frames_dir, args.out, names)
    has_pm = complete(args.out, "pointmap", names)

    id2label = {int(k): v for k, v in json.loads(
        (args.out / "infer_seg.json").read_text(encoding="utf-8"))["id2label"].items()}
    id2label = resolve_id2label(id2label)
    print(f"kelas 0 = {id2label.get(0)}")
    lut, group_ids = group_lut(id2label, groups)

    print("menghitung metrik ...")
    rows, counts = compute_metrics(args.out, names, args.mp_masks, has_pm, args)
    ab.write_csv(rows, args.out / "metrics_per_frame.csv")
    metrics = summarize_metrics(rows, counts, id2label, args)

    timing = {k: timing_summary(args.out, k, args.warmup) for k in MODELS if complete(args.out, k, names)}
    est = {"seg": 360 * timing["seg"]["total_s"]["mean"]}
    if "pointmap" in timing:
        est["seg+pointmap"] = est["seg"] + 360 * timing["pointmap"]["total_s"]["mean"]

    pal = palette(NUM_CLASSES)

    def render(n: str) -> np.ndarray:
        return render_row(frames_dir, args.out, n, pal, lut, has_pm, args.line_px)

    print("render side_by_side.mp4 + frames_grid.png ...")
    video = write_video(frames_dir, args.out, names, render, args.fps)
    grid = write_grid(args.out, names, [int(x) for x in args.grid_frames.split(",")], render, args.grid_scale)

    mp_ref = None
    mp_summary = args.mp_masks.parent / "metrics_summary.json"
    if mp_summary.is_file():
        mp_ref = json.loads(mp_summary.read_text(encoding="utf-8"))["backends"].get("mediapipe")

    summary = {"task": "T-102c", "frame_count": len(names), "setup": setup, "bench": bench,
               "timing": timing, "estimate_360_frames_s": est, "groups": group_ids,
               "qc": {"iou_min": args.iou_min, "area_min": args.area_min, "area_max": args.area_max,
                      "blob_min": args.blob_min},
               "metrics": metrics, "total_wall_s": time.perf_counter() - t_start}
    (args.out / "metrics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print_report(summary, mp_ref)
    print(f"\nOutput di {args.out}: {video.name}, {grid.name}, metrics_per_frame.csv, metrics_summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
