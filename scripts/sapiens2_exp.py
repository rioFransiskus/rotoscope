"""Eksperimen lanjutan T-102c (D-009): E1 sumber kedalaman + E2 Sapiens2-seg 0.8B / 1B.

Alat sekali pakai, BUKAN modul pipeline. Tiap langkah = satu proses terpisah, supaya peak VRAM/RAM
terukur bersih dan OOM di satu eksperimen tidak mematikan eksperimen lain:
    venv/Scripts/python.exe scripts/sapiens2_exp.py bench  <sumber>   # E1: 20 frame tersebar
    venv/Scripts/python.exe scripts/sapiens2_exp.py full   <sumber>   # E1: semua frame (kalau bench lolos;
                                                                       #     frame yang sudah ada & valid dilewati)
    venv/Scripts/python.exe scripts/sapiens2_exp.py render <sumber>   # E1: side_by_side + grid + klip 73-92
    venv/Scripts/python.exe scripts/sapiens2_exp.py compare           # E1: DA vs pointmap CPU, tanpa inferensi
    venv/Scripts/python.exe scripts/sapiens2_exp.py lines             # E1: garis dari lompatan kedalaman relatif
    venv/Scripts/python.exe scripts/sapiens2_exp.py seglines          # E2: garis seg + DA p95, [0.4B | 0.8B]
    venv/Scripts/python.exe scripts/sapiens2_exp.py diag-hands        # diagnosis komponen garis kecil di tangan
    venv/Scripts/python.exe scripts/sapiens2_exp.py filtered          # post-processing a->b->garis->c, 3 panel
    venv/Scripts/python.exe scripts/sapiens2_exp.py seg-full <model>  # full run seg, resume per frame
    venv/Scripts/python.exe scripts/sapiens2_exp.py seg-metrics <model>    # metrik full klip vs 0.4B
    venv/Scripts/python.exe scripts/sapiens2_exp.py filtered-full <model>  # render full klip + post
    venv/Scripts/python.exe scripts/sapiens2_exp.py seg    <model>    # E2: 20 frame berurutan
    venv/Scripts/python.exe scripts/sapiens2_exp.py report
sumber: pm_autocast_gpu | pm_fp32_cpu | da2s_gpu      model: seg_0.8b | seg_1b

Input : work/meta.json + work/frames/, full run seg 0.4B di work/t102c/seg/ + metrics_per_frame.csv
        (hanya dibaca, TIDAK ditimpa).
Output: work/t102c/exp/ -> bench_<sumber>.json, depth_<sumber>/frame_%05d.npy (float16),
        infer_<sumber>.json, side_by_side_<sumber>.mp4, frames_grid_<sumber>.png,
        side_by_side_<sumber>_f073-092.mp4, compare_f073-092.mp4, frames_grid_compare.png,
        lines_<da2|pointmap>_f073-092.mp4, frames_grid_lines_<da2|pointmap>.png, lines_thresholds.json, e2_<model>.json, seg_<model>/frame_%05d.png,
        side_by_side_<model>_fAAA-BBB.mp4

Sumber kedalaman:
- pm_autocast_gpu: pointmap 0.4B, bobot fp32, forward di bawah torch.autocast fp16 (bukan model.half()).
  requires_grad dimatikan supaya autocast tidak menyimpan cache salinan fp16 bobot.
- pm_fp32_cpu: pointmap 0.4B fp32 di CPU, resolusi default image processor.
- da2s_gpu: Depth Anything V2 Small (Apache-2.0) fp32 GPU. Keluarannya kedalaman RELATIF terbalik
  (disparity, besar = dekat); panel 4 memakai |gradien| jadi arah skala tidak berpengaruh.
Lolos = tanpa OOM/error dan semua keluaran finite.

E2: GPU fp16 kalau muat. Kalau OOM: CPU fp32 hanya kalau RAM bebas - perkiraan kebutuhan
(ukuran bobot x --cpu-mem-factor) masih >= --ram-reserve x RAM total; CPU dijalankan di proses baru.

Peak RAM = PeakWorkingSetSize proses (Windows, lewat ctypes; tanpa dependency baru), termasuk load model.
Checkpoint dibaca dari cache Hugging Face dengan HF_HUB_OFFLINE=1 (tanpa jaringan saat runtime).
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import gc
import json
import os
import statistics
import subprocess
import sys
import time
import traceback
from ctypes import wintypes
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ab_segment as ab  # noqa: E402
import sapiens2_probe as probe  # noqa: E402

ROOT = probe.ROOT
POINTMAP_ID = "facebook/sapiens2-pointmap-0.4b"
DEPTH_SOURCES = {
    "pm_autocast_gpu": {"model": POINTMAP_ID, "kind": "pointmap", "device": "cuda", "autocast": True,
                        "label": "pointmap 0.4B autocast fp16"},
    "pm_fp32_cpu": {"model": POINTMAP_ID, "kind": "pointmap", "device": "cpu", "autocast": False,
                    "label": "pointmap 0.4B fp32 CPU"},
    "da2s_gpu": {"model": "depth-anything/Depth-Anything-V2-Small-hf", "kind": "da", "device": "cuda",
                 "autocast": False, "label": "Depth Anything V2 Small"},
}
SEG_MODELS = {"seg_0.8b": "facebook/sapiens2-seg-0.8b", "seg_1b": "facebook/sapiens2-seg-1b"}
# Keputusan Rio 2026-09-29: tidak dijalankan
SEG_SKIPPED = {"seg_1b": "dilewati: tidak layak di hardware ini (VRAM tidak muat, RAM CPU tidak memenuhi syarat)"}
GIB = 2 ** 30
MIB = 2 ** 20


# --- memori (Windows, ctypes) ----------------------------------------------
class _PMC(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]


class _MSX(ctypes.Structure):
    _fields_ = [("dwLength", wintypes.DWORD), ("dwMemoryLoad", wintypes.DWORD),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def proc_mem() -> dict:
    """Peak working set + peak private commit proses ini (MiB)."""
    k32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PMC), wintypes.DWORD]
    c = _PMC()
    c.cb = ctypes.sizeof(c)
    if not psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
        return {}
    return {"peak_ws_mib": c.PeakWorkingSetSize / MIB, "peak_private_mib": c.PeakPagefileUsage / MIB}


def sys_mem() -> dict:
    m = _MSX()
    m.dwLength = ctypes.sizeof(m)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return {"total_gib": m.ullTotalPhys / GIB, "avail_gib": m.ullAvailPhys / GIB}


def weights_gib(model_id: str) -> float | None:
    from huggingface_hub import try_to_load_from_cache
    p = try_to_load_from_cache(model_id, "model.safetensors")
    return Path(p).stat().st_size / GIB if isinstance(p, str) else None


# --- model -----------------------------------------------------------------
class Model:
    """Satu model (seg / pointmap / Depth Anything) di satu device + dtype, opsional autocast fp16."""

    def __init__(self, kind: str, model_id: str, device: str, precision: str, autocast: bool = False):
        import torch
        import transformers
        from transformers import AutoImageProcessor

        cls = {"seg": transformers.Sapiens2ForSemanticSegmentation,
               "pointmap": transformers.Sapiens2ForPointmapEstimation,
               "da": transformers.DepthAnythingForDepthEstimation}[kind]
        self.torch, self.kind, self.device, self.autocast = torch, kind, device, autocast
        self.dtype = torch.float16 if precision == "fp16" else torch.float32
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        t0 = time.perf_counter()
        model = cls.from_pretrained(model_id, dtype=self.dtype)
        model.requires_grad_(False)
        self.model = model.to(device).eval()
        self.load_s = time.perf_counter() - t0
        self.num_labels = model.config.num_labels if kind == "seg" else None

    def sync(self) -> None:
        if self.device == "cuda":
            self.torch.cuda.synchronize()

    def infer(self, rgb: np.ndarray) -> tuple[np.ndarray, dict]:
        """RGB HxWx3 -> (class map uint8 | Z / disparity float32 HxW, info waktu + finite)."""
        torch = self.torch
        h, w = rgb.shape[:2]
        t0 = time.perf_counter()
        pixel = self.processor(images=rgb, return_tensors="pt")["pixel_values"].to(self.device, self.dtype)
        self.sync()
        t1 = time.perf_counter()
        with torch.inference_mode():
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.autocast):
                out = self.model(pixel_values=pixel)
            if self.kind == "seg":
                finite = bool(torch.isfinite(out.logits).all())
                res = self.processor.post_process_semantic_segmentation(out, target_sizes=[(h, w)])[0]
                res = res.to(torch.uint8)
            elif self.kind == "pointmap":
                finite = bool(torch.isfinite(out.pointmaps).all())
                if out.scales is not None:
                    finite = finite and bool(torch.isfinite(out.scales).all())
                pm = self.processor.post_process_pointmap_estimation(out, source_sizes=[(h, w)])[0]["pointmap"]
                res = pm[2].float()
            else:
                finite = bool(torch.isfinite(out.predicted_depth).all())
                res = self.processor.post_process_depth_estimation(out, target_sizes=[(h, w)])[0]
                res = res["predicted_depth"].float()
            self.sync()
        t2 = time.perf_counter()
        return res.cpu().numpy(), {"pre_s": t1 - t0, "infer_s": t2 - t1, "finite": finite,
                                   "input_hw": list(pixel.shape[-2:])}

    def close(self) -> None:
        del self.model
        gc.collect()
        if self.device == "cuda":
            self.torch.cuda.empty_cache()


def run_frames(kind: str, model_id: str, device: str, precision: str, autocast: bool,
               frames_dir: Path, names: list[str], warmup: int, save=None) -> tuple[dict, list[np.ndarray]]:
    """Jalankan model pada `names`. OOM / error lain dicatat sebagai hasil (timebox), bukan crash."""
    import torch
    rec = {"model": model_id, "device": device, "precision": precision, "autocast_fp16": autocast,
           "n_frames": len(names), "sys_mem_before": sys_mem(),
           "cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF")}
    if device == "cuda":
        rec["nvidia_smi_before"] = probe.nvidia_smi()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    outputs, infos, runner = [], [], None
    try:
        runner = Model(kind, model_id, device, precision, autocast)
        rec["load_s"] = runner.load_s
        rec["num_labels"] = runner.num_labels
        for i, n in enumerate(names):
            t0 = time.perf_counter()
            arr, info = runner.infer(ab.read_rgb(frames_dir / n))
            if save is not None:
                save(n, arr)
            else:
                outputs.append(arr)
            info["total_s"] = time.perf_counter() - t0
            info["frame"] = n
            infos.append(info)
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(names)}", flush=True)
        timed = infos[warmup:] or infos
        rec.update({
            "status": "ok", "all_finite": all(i["finite"] for i in infos),
            "n_nonfinite_frames": sum(not i["finite"] for i in infos), "input_hw": infos[0]["input_hw"],
            "infer_s": probe.pct_stats([i["infer_s"] for i in timed]),
            "total_s": probe.pct_stats([i["total_s"] for i in timed]),
            "warmup_infer_s": [round(i["infer_s"], 3) for i in infos[:warmup]],
        })
    except Exception as e:  # noqa: BLE001 - OOM/setup gagal = hasil eksperimen
        oom = probe.is_oom(e) or isinstance(e, MemoryError)
        rec.update({"status": "oom" if oom else "error", "error": str(e).splitlines()[0][:300] if str(e) else repr(e),
                    "frames_done": len(infos)})
        if not oom:
            rec["traceback"] = traceback.format_exc()[-2000:]
        outputs = []
    finally:
        if device == "cuda":
            rec["peak_alloc_mib"] = torch.cuda.max_memory_allocated() / MIB
            rec["peak_reserved_mib"] = torch.cuda.max_memory_reserved() / MIB
        if runner is not None:
            runner.close()
        rec.update(proc_mem())
    rec["frames"] = infos
    rec["passed"] = rec["status"] == "ok" and rec["all_finite"]
    extra = (f"infer mean {rec['infer_s']['mean']:.3f} s, finite {rec['all_finite']}" if rec["status"] == "ok"
             else rec["error"])
    print(f"  {model_id} {device} {precision}{' autocast' if autocast else ''}: {rec['status']} - {extra}"
          f" | peak alloc {rec.get('peak_alloc_mib', float('nan')):.0f} MiB,"
          f" peak RAM {rec.get('peak_ws_mib', float('nan')):.0f} MiB", flush=True)
    return rec, outputs


# --- util ------------------------------------------------------------------
def load_ctx(args) -> dict:
    meta = json.loads((args.work / "meta.json").read_text(encoding="utf-8"))
    names = ab.frame_names(meta)
    frames_dir = args.work / "frames"
    missing = [n for n in names if not (frames_dir / n).is_file()]
    if missing:
        ab.fail(f"{len(missing)} frame hilang di {frames_dir}, mis. {missing[0]}")
    args.exp.mkdir(parents=True, exist_ok=True)
    return {"names": names, "frames_dir": frames_dir, "hw": (meta["working_height"], meta["working_width"])}


def dump(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=1), encoding="utf-8")


def depth_path(exp: Path, src: str, name: str) -> Path:
    return exp / f"depth_{src}" / f"{Path(name).stem}.npy"


def valid_depth(path: Path, hw: tuple[int, int]) -> bool:
    """Output kedalaman terbaca, berukuran sama dengan frame, dan semua finite (file terpotong = tidak valid)."""
    try:
        z = np.load(path)
    except (OSError, ValueError):
        return False
    return z.shape == hw and bool(np.isfinite(z).all())


def group_lut(seg_dir: Path) -> np.ndarray:
    id2label = {int(k): v for k, v in json.loads(
        (seg_dir.parent / "infer_seg.json").read_text(encoding="utf-8"))["id2label"].items()}
    groups = {k: v for k, v in json.loads((ROOT / "scripts" / "sapiens2_groups.json").read_text(
        encoding="utf-8")).items() if not k.startswith("_")}
    lut, _ = probe.group_lut(probe.resolve_id2label(id2label), groups)
    return lut


def seg_line_mask(seg: np.ndarray, lut: np.ndarray, line_px: int) -> np.ndarray:
    """Garis grup seg (siluet luar + batas antar grup), ditebalkan line_px."""
    lines = probe.boundary(lut[seg])
    if line_px > 1:
        lines = cv2.dilate(lines.astype(np.uint8), np.ones((line_px, line_px), np.uint8)).astype(bool)
    return lines


def line_panel(seg: np.ndarray, lut: np.ndarray, line_px: int) -> np.ndarray:
    img = np.full(seg.shape + (3,), probe.LINE_BG, np.uint8)
    img[seg_line_mask(seg, lut, line_px)] = probe.LINE_FG
    return img


def depth_line_mask(seg: np.ndarray, z: np.ndarray, erode_px: int, eps: float, thr: float) -> np.ndarray:
    """Garis kedalaman: J > thr di foreground seg ter-erode (lihat relative_jump)."""
    inner = cv2.erode((seg != 0).astype(np.uint8), np.ones((erode_px, erode_px), np.uint8)).astype(bool)
    return inner & (relative_jump(z, inner, eps) > thr)


def read_class_map(path: Path) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if m is None:
        ab.fail(f"gagal membaca class map {path}")
    return m


def write_video(path: Path, names: list[str], render, fps: int) -> Path:
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
        ab.fail(f"ffmpeg gagal meng-encode {path.name}")
    return path


def frame_range(names: list[str], a: int, b: int) -> list[str]:
    return [n for n in names if a <= probe.frame_no(n) <= b]


def label_agreement(cur: np.ndarray, prev: np.ndarray) -> float | None:
    both = (cur != 0) & (prev != 0)
    return float((cur[both] == prev[both]).mean()) if both.any() else None


def self_check() -> None:
    m = proc_mem()
    s = sys_mem()
    if not m.get("peak_ws_mib", 0) > 0 or not s["total_gib"] > 0:
        ab.fail(f"self-check gagal: pengukuran memori {m} {s}")
    a = np.array([[0, 1, 2], [0, 1, 1]], np.uint8)
    b = np.array([[0, 1, 1], [0, 0, 1]], np.uint8)
    if label_agreement(a, b) != 2 / 3:
        ab.fail("self-check gagal: label_agreement")
    z = np.ones((4, 4), np.float32)
    z[:, 2:] = 2.0  # lompatan Z 1 -> 2 = relatif 1.0; harus sama untuk disparity 1/Z
    ok = np.ones_like(z, bool)
    jz, jd = relative_jump(z, ok, 1e-6), relative_jump(1 / z, ok, 1e-6)
    if not (np.allclose(jz[:, 1:3], 1.0) and jz[:, 0].max() == 0 and np.allclose(jz, jd)):
        ab.fail("self-check gagal: relative_jump")
    s = np.ones((12, 12), np.uint8)
    s[5:7, 5:7] = 2  # pulau 4 px kelas 2 di dalam kelas 1
    s[:, 9:] = 3     # kelas 3 besar, harus tetap
    f = island_filter(s, 5)
    if not ((f[5:7, 5:7] == 1).all() and (f[:, 9:] == 3).all()):
        ab.fail("self-check gagal: island_filter")
    g = np.zeros((15, 15), np.uint8)
    g[3:12, 3:12] = 1
    g[7, 12] = 1  # tonjolan 1 px -> hilang; blok 9x9 tetap
    mf = mode_filter(g, 5)
    if mf[7, 12] != 0 or not (mf[5:10, 5:10] == 1).all():
        ab.fail("self-check gagal: mode_filter")
    diag = np.eye(5, dtype=bool)  # satu komponen 5 px (8-arah)
    if component_filter(diag, 6).any() or not (component_filter(diag, 5) == diag).all():
        ab.fail("self-check gagal: component_filter")
    names = [f"frame_{i:05d}.png" for i in range(10)]
    if frame_range(names, 3, 5) != names[3:6]:
        ab.fail("self-check gagal: frame_range")
    print("self-check OK")


# --- E1 --------------------------------------------------------------------
def cmd_bench(args, ctx) -> None:
    src = DEPTH_SOURCES[args.target]
    names = ctx["names"]
    idx = np.linspace(0, len(names) - 1, args.bench_frames).round().astype(int).tolist()
    sel = [names[i] for i in idx]
    print(f"[E1 bench] {args.target}: {len(sel)} frame")
    rec, _ = run_frames(src["kind"], src["model"], src["device"], "fp32", src["autocast"],
                        ctx["frames_dir"], sel, args.warmup)
    rec["source"] = args.target
    dump(args.exp / f"bench_{args.target}.json", rec)


def cmd_full(args, ctx) -> None:
    src = DEPTH_SOURCES[args.target]
    names = ctx["names"]
    bench_path = args.exp / f"bench_{args.target}.json"
    if not bench_path.is_file():
        ab.fail(f"{bench_path.name} tidak ada - jalankan bench dulu")
    if not json.loads(bench_path.read_text(encoding="utf-8"))["passed"]:
        ab.fail(f"{args.target} tidak lolos bench -> full run dilewati")
    info_path = args.exp / f"infer_{args.target}.json"
    # Skip per-frame: output yang sudah ada DAN valid (terbaca, ukuran = frame, semua finite) tidak diulang,
    # supaya run yang terhenti bisa dilanjutkan.
    todo = [n for n in names if args.force or not valid_depth(depth_path(args.exp, args.target, n), ctx["hw"])]
    if not todo:
        print(f"[E1 full] {args.target} lengkap -> dilewati (--force untuk ulang)")
        return
    (args.exp / f"depth_{args.target}").mkdir(parents=True, exist_ok=True)

    def save(n: str, arr: np.ndarray) -> None:
        np.save(depth_path(args.exp, args.target, n), arr.astype(np.float16))

    print(f"[E1 full] {args.target}: {len(todo)} frame ({len(names) - len(todo)} sudah ada, dilewati)")
    t0 = time.perf_counter()
    rec, _ = run_frames(src["kind"], src["model"], src["device"], "fp32", src["autocast"],
                        ctx["frames_dir"], todo, args.warmup, save=save)
    rec["source"] = args.target
    rec["n_skipped_existing"] = len(names) - len(todo)
    rec["wall_s"] = time.perf_counter() - t0
    dump(info_path, rec)


def cmd_render(args, ctx) -> None:
    src = DEPTH_SOURCES[args.target]
    names, frames_dir = ctx["names"], ctx["frames_dir"]
    if not all(depth_path(args.exp, args.target, n).is_file() for n in names):
        ab.fail(f"depth_{args.target} belum lengkap - jalankan full dulu")
    seg_dir = args.t102c / "seg"
    lut = group_lut(seg_dir)
    pal = probe.palette(probe.NUM_CLASSES)

    def render(n: str) -> np.ndarray:
        stem = Path(n).stem
        bgr = cv2.imread(str(frames_dir / n), cv2.IMREAD_COLOR)
        seg = read_class_map(seg_dir / n)
        p0 = bgr.copy()
        ab.draw_label(p0, ["asli", stem])
        p1 = pal[seg]
        ab.draw_label(p1, ["kelas seg 0.4B", stem])
        p2 = line_panel(seg, lut, args.line_px)
        ab.draw_label(p2, ["garis grup seg 0.4B", stem])
        z = np.load(depth_path(args.exp, args.target, n)).astype(np.float32)
        p3 = probe.depth_gradient(z, seg != 0)
        ab.draw_label(p3, [f"|grad| {src['label']}", stem])
        return np.hstack([p0, p1, p2, p3])

    t0 = time.perf_counter()
    print(f"[E1 render] {args.target}")
    write_video(args.exp / f"side_by_side_{args.target}.mp4", names, render, args.fps)
    a, b = args.leg_frames
    short = frame_range(names, a, b)
    write_video(args.exp / f"side_by_side_{args.target}_f{a:03d}-{b:03d}.mp4", short, render, args.fps)
    by_no = {probe.frame_no(n): n for n in names}
    grid = [int(x) for x in args.grid_frames.split(",")]
    missing = [f for f in grid if f not in by_no]
    if missing:
        print(f"  PERINGATAN: frame grid tidak ada: {missing}")
    tiles = [cv2.resize(render(by_no[f]), None, fx=args.grid_scale, fy=args.grid_scale,
                        interpolation=cv2.INTER_AREA) for f in grid if f in by_no]
    cv2.imwrite(str(args.exp / f"frames_grid_{args.target}.png"), np.vstack(tiles))
    print(f"  selesai {time.perf_counter() - t0:.1f} s")


def cmd_compare(args, ctx) -> None:
    """Tanpa inferensi: [asli | garis grup seg 0.4B | |grad| DA-V2-Small | |grad| pointmap CPU] dari output tersimpan.

    Hanya frame yang output kedua sumbernya ada dan valid yang dirender.
    """
    names, frames_dir = ctx["names"], ctx["frames_dir"]
    a, b = args.leg_frames
    srcs = ["da2s_gpu", "pm_fp32_cpu"]
    wanted = frame_range(names, a, b)
    have = [n for n in wanted if all(valid_depth(depth_path(args.exp, s, n), ctx["hw"]) for s in srcs)]
    lost = [probe.frame_no(n) for n in wanted if n not in have]
    print(f"[compare] frame {a}-{b}: {len(have)}/{len(wanted)} tersedia" + (f", tidak ada: {lost}" if lost else ""))
    if not have:
        ab.fail("tidak ada frame dengan output kedalaman lengkap")
    seg_dir = args.t102c / "seg"
    lut = group_lut(seg_dir)

    def render(n: str) -> np.ndarray:
        stem = Path(n).stem
        bgr = cv2.imread(str(frames_dir / n), cv2.IMREAD_COLOR)
        seg = read_class_map(seg_dir / n)
        p0 = bgr.copy()
        ab.draw_label(p0, ["asli", stem])
        p1 = line_panel(seg, lut, args.line_px)
        ab.draw_label(p1, ["garis grup seg 0.4B", stem])
        panels = [p0, p1]
        for s in srcs:
            z = np.load(depth_path(args.exp, s, n)).astype(np.float32)
            p = probe.depth_gradient(z, seg != 0)
            ab.draw_label(p, [f"|grad| {DEPTH_SOURCES[s]['label']}", stem])
            panels.append(p)
        return np.hstack(panels)

    write_video(args.exp / f"compare_f{a:03d}-{b:03d}.mp4", have, render, args.fps)
    by_no = {probe.frame_no(n): n for n in have}
    grid = [int(x) for x in args.compare_grid.split(",")]
    tiles = [cv2.resize(render(by_no[f]), None, fx=args.grid_scale, fy=args.grid_scale,
                        interpolation=cv2.INTER_AREA) for f in grid if f in by_no]
    cv2.imwrite(str(args.exp / "frames_grid_compare.png"), np.vstack(tiles))
    print(f"  grid: {[f for f in grid if f in by_no]}" + (f", tidak ada: {[f for f in grid if f not in by_no]}"
                                                          if len(tiles) < len(grid) else ""))


LINE_SOURCES = {"da2s_gpu": "da2", "pm_fp32_cpu": "pointmap"}  # sumber -> nama file lines_<nama>_*.mp4
LINE_PCTS = [95.0, 98.0, 99.5]


def relative_jump(d: np.ndarray, valid: np.ndarray, eps: float) -> np.ndarray:
    """Lompatan kedalaman RELATIF per piksel ke tetangga 4-arah.

    Untuk pasangan tetangga p, q yang keduanya valid (di dalam foreground ter-erode, finite, > eps):
        J(p, q) = max(d_p, d_q) / min(d_p, d_q) - 1
    J(p) = maksimum J(p, q) atas tetangga atas/bawah/kiri/kanan; 0 kalau tidak ada pasangan valid.
    Satu lompatan menandai piksel di KEDUA sisinya (garis ±2 px).

    Rasio ini sama untuk Z dan 1/Z: kalau Z_p < Z_q, (1/Z_p) / (1/Z_q) - 1 = Z_q / Z_p - 1. Jadi pointmap
    (Z, meter) dan Depth-Anything-V2 (disparity relatif) diukur dengan definisi yang sama. Catatan: disparity
    DA hanya benar sampai skala + offset, jadi nilai J DA tidak bisa dibandingkan langsung dengan J pointmap;
    karena itu threshold diambil dari persentil per sumber.
    """
    d = d.astype(np.float32)
    ok = valid & np.isfinite(d) & (d > eps)
    dv = np.where(ok, d, 1.0)
    jump = np.zeros(d.shape, np.float32)
    for axis in (0, 1):
        n = d.shape[axis] - 1
        p, q = np.take(dv, range(n), axis), np.take(dv, range(1, n + 1), axis)
        pair_ok = np.take(ok, range(n), axis) & np.take(ok, range(1, n + 1), axis)
        r = np.where(pair_ok, np.maximum(p, q) / np.minimum(p, q) - 1.0, 0.0).astype(np.float32)
        if axis == 0:
            jump[:-1] = np.maximum(jump[:-1], r)
            jump[1:] = np.maximum(jump[1:], r)
        else:
            jump[:, :-1] = np.maximum(jump[:, :-1], r)
            jump[:, 1:] = np.maximum(jump[:, 1:], r)
    return jump


def cmd_lines(args, ctx) -> None:
    """Tanpa inferensi: garis dari lompatan kedalaman relatif (+ garis grup seg) di frame --leg-frames.

    Threshold = persentil J di foreground ter-erode, dihitung PER KLIP (semua frame digabung) per sumber.
    """
    names, frames_dir = ctx["names"], ctx["frames_dir"]
    a, b = args.leg_frames
    seg_dir = args.t102c / "seg"
    lut = group_lut(seg_dir)
    kernel = np.ones((args.lines_erode, args.lines_erode), np.uint8)
    wanted = frame_range(names, a, b)
    report = {"frames": [a, b], "erode_px": args.lines_erode, "eps": args.lines_eps,
              "percentiles": LINE_PCTS, "sources": {}}
    for src, tag in LINE_SOURCES.items():
        have = [n for n in wanted if valid_depth(depth_path(args.exp, src, n), ctx["hw"])]
        lost = [probe.frame_no(n) for n in wanted if n not in have]
        print(f"[lines] {src}: {len(have)}/{len(wanted)} frame" + (f", tidak ada: {lost}" if lost else ""))
        if not have:
            continue
        cache = {}
        for n in have:
            seg = read_class_map(seg_dir / n)
            inner = cv2.erode((seg != 0).astype(np.uint8), kernel).astype(bool)
            z = np.load(depth_path(args.exp, src, n)).astype(np.float32)
            cache[n] = (seg, inner, relative_jump(z, inner, args.lines_eps))
        vals = np.concatenate([j[inner] for _, inner, j in cache.values()])
        thr = [float(np.percentile(vals, p)) for p in LINE_PCTS]
        report["sources"][src] = {"n_frames": len(have), "missing": lost, "n_values": int(vals.size),
                                  "thresholds": dict(zip([f"p{p:g}" for p in LINE_PCTS], thr)),
                                  "line_px_share": {f"p{p:g}": float((vals > t).mean())
                                                    for p, t in zip(LINE_PCTS, thr)}}
        print(f"  threshold {dict(zip(LINE_PCTS, [round(t, 5) for t in thr]))}")

        def render(n: str, src=src, thr=thr, cache=cache) -> np.ndarray:
            stem = Path(n).stem
            seg, inner, jump = cache[n]
            p0 = cv2.imread(str(frames_dir / n), cv2.IMREAD_COLOR)
            ab.draw_label(p0, ["asli", stem])
            panels = [p0]
            for p, t in zip(LINE_PCTS, thr):
                img = line_panel(seg, lut, args.line_px)
                img[inner & (jump > t)] = probe.LINE_FG
                ab.draw_label(img, [f"{DEPTH_SOURCES[src]['label']}", f"p{p:g} thr {t:.5f}", stem])
                panels.append(img)
            return np.hstack(panels)

        write_video(args.exp / f"lines_{tag}_f{a:03d}-{b:03d}.mp4", have, render, args.fps)
        by_no = {probe.frame_no(n): n for n in have}
        grid = [int(x) for x in args.lines_grid.split(",")]
        tiles = [cv2.resize(render(by_no[f]), None, fx=args.grid_scale, fy=args.grid_scale,
                            interpolation=cv2.INTER_AREA) for f in grid if f in by_no]
        cv2.imwrite(str(args.exp / f"frames_grid_lines_{tag}.png"), np.vstack(tiles))
    dump(args.exp / "lines_thresholds.json", report)


def cmd_seglines(args, ctx) -> None:
    """Tanpa inferensi: garis grup seg + garis DA-V2 p<--seglines-pct>, jendela E2, panel [0.4B | 0.8B].

    Threshold dihitung PER KLIP pada jendela ini dari J di foreground seg 0.4B ter-erode, lalu dipakai
    SAMA untuk kedua panel (beda panel = beda seg saja: garis grup + foreground tempat garis DA dihitung).
    """
    names, frames_dir = ctx["names"], ctx["frames_dir"]
    window, _, _ = e2_window(args, names)
    src, model = "da2s_gpu", "seg_0.8b"
    seg_dirs = {"0.4B": args.t102c / "seg", "0.8B": args.exp / f"seg_{model}"}
    lut = group_lut(args.t102c / "seg")
    kernel = np.ones((args.lines_erode, args.lines_erode), np.uint8)
    vals = []
    for n in window:
        seg = read_class_map(seg_dirs["0.4B"] / n)
        inner = cv2.erode((seg != 0).astype(np.uint8), kernel).astype(bool)
        z = np.load(depth_path(args.exp, src, n)).astype(np.float32)
        vals.append(relative_jump(z, inner, args.lines_eps)[inner])
    thr = float(np.percentile(np.concatenate(vals), args.seglines_pct))
    print(f"[seglines] frame {window[0]} .. {window[-1]}, DA p{args.seglines_pct:g} thr = {thr:.5f}")

    def render(n: str) -> np.ndarray:
        stem = Path(n).stem
        z = np.load(depth_path(args.exp, src, n)).astype(np.float32)
        panels = []
        for tag, d in seg_dirs.items():
            seg = read_class_map(d / n)
            img = line_panel(seg, lut, args.line_px)
            img[depth_line_mask(seg, z, args.lines_erode, args.lines_eps, thr)] = probe.LINE_FG
            ab.draw_label(img, [f"seg {tag} + DA-V2 p{args.seglines_pct:g}", f"thr {thr:.5f}", stem])
            panels.append(img)
        return np.hstack(panels)

    a, b = probe.frame_no(window[0]), probe.frame_no(window[-1])
    write_video(args.exp / f"seglines_da2_f{a:03d}-{b:03d}.mp4", window, render, args.fps)
    dump(args.exp / f"seglines_da2_f{a:03d}-{b:03d}.json",
         {"frames": [a, b], "percentile": args.seglines_pct, "threshold": thr, "threshold_fg": "seg 0.4B",
          "erode_px": args.lines_erode, "groups": str(ROOT / "scripts" / "sapiens2_groups.json")})


def cmd_diag_hands(args, ctx) -> None:
    """Diagnosis glitch tangan (tanpa perbaikan): garis gabungan seg 0.4B + DA-V2 p<pct> di --leg-frames.

    Zona tangan = kelas Left_Hand/Right_Hand di-dilate --hand-dilate px. Komponen = komponen terhubung
    (8-arah) garis gabungan yang menyentuh zona tangan; "kecil" = ukuran < --small-comp-px. Frame dengan
    komponen kecil terbanyak dilaporkan: piksel garis di zona dari seg saja / DA saja / keduanya, dan
    ukuran + asal piksel tiap komponen. Threshold DA = persentil per klip dari lines_thresholds.json.
    """
    names = ctx["names"]
    a, b = args.leg_frames
    thr_file = json.loads((args.exp / "lines_thresholds.json").read_text(encoding="utf-8"))
    thr = thr_file["sources"]["da2s_gpu"]["thresholds"][f"p{args.seglines_pct:g}"]
    seg_dir = args.t102c / "seg"
    lut = group_lut(seg_dir)
    classes = json.loads(probe.CLASSES_FILE.read_text(encoding="utf-8"))["classes"]
    hand_ids = [classes.index(c) for c in ("Left_Hand", "Right_Hand")]
    dk = np.ones((2 * args.hand_dilate + 1, 2 * args.hand_dilate + 1), np.uint8)

    per_frame = []
    for n in frame_range(names, a, b):
        seg = read_class_map(seg_dir / n)
        zone = cv2.dilate(np.isin(seg, hand_ids).astype(np.uint8), dk).astype(bool)
        z = np.load(depth_path(args.exp, "da2s_gpu", n)).astype(np.float32)
        s_lines = seg_line_mask(seg, lut, args.line_px)
        d_lines = depth_line_mask(seg, z, args.lines_erode, args.lines_eps, thr)
        both = s_lines | d_lines
        k, lab, stats, _ = cv2.connectedComponentsWithStats(both.astype(np.uint8), connectivity=8)
        comps = []
        for c in range(1, k):
            m = lab == c
            if not (m & zone).any():
                continue
            comps.append({"size_px": int(stats[c, cv2.CC_STAT_AREA]), "in_zone_px": int((m & zone).sum()),
                          "seg_only_px": int((m & s_lines & ~d_lines).sum()),
                          "da_only_px": int((m & d_lines & ~s_lines).sum()),
                          "seg_and_da_px": int((m & s_lines & d_lines).sum()),
                          "bbox_xywh": [int(v) for v in stats[c, :4]]})
        comps.sort(key=lambda c: c["size_px"])
        per_frame.append({"frame": n, "zone_px": int(zone.sum()),
                          "n_small": sum(c["size_px"] < args.small_comp_px for c in comps),
                          "n_comps": len(comps),
                          "zone_seg_only_px": int((zone & s_lines & ~d_lines).sum()),
                          "zone_da_only_px": int((zone & d_lines & ~s_lines).sum()),
                          "zone_seg_and_da_px": int((zone & s_lines & d_lines).sum()),
                          "components": comps})
    worst = max(per_frame, key=lambda r: (r["n_small"], r["n_comps"]))
    out = {"frames": [a, b], "da_threshold": thr, "percentile": args.seglines_pct, "hand_dilate_px": args.hand_dilate,
           "small_comp_px": args.small_comp_px, "line_px": args.line_px, "erode_px": args.lines_erode,
           "worst_frame": worst["frame"], "per_frame": per_frame}
    dump(args.exp / "diag_hands_da2.json", out)

    sys.stdout.reconfigure(encoding="utf-8")
    print(f"[diag-hands] DA p{args.seglines_pct:g} thr {thr:.5f}, zona tangan dilate {args.hand_dilate} px, "
          f"kecil < {args.small_comp_px} px")
    print("frame: komponen kecil / total di zona; piksel zona seg saja / DA saja / keduanya")
    for r in per_frame:
        print(f"  {Path(r['frame']).stem}: {r['n_small']} / {r['n_comps']}; "
              f"{r['zone_seg_only_px']} / {r['zone_da_only_px']} / {r['zone_seg_and_da_px']}")
    print(f"\nFrame terburuk: {worst['frame']}")
    print("| # | ukuran px | di zona px | seg saja | DA saja | seg+DA | bbox x,y,w,h |")
    print("|---|---|---|---|---|---|---|")
    for i, c in enumerate(worst["components"], 1):
        print(f"| {i} | {c['size_px']} | {c['in_zone_px']} | {c['seg_only_px']} | {c['da_only_px']} | "
              f"{c['seg_and_da_px']} | {c['bbox_xywh']} |")


# --- post-processing (tanpa mengubah model) ---------------------------------
IDENTITY_LUT = np.arange(256, dtype=np.uint8)


def island_filter(seg: np.ndarray, n_min: int) -> np.ndarray:
    """(a) Komponen terhubung (8-arah) sebuah kelas < n_min piksel -> kelas mayoritas di cincin 1 px sekelilingnya.

    Semua kelas termasuk background (lubang kecil di badan / bercak kecil di latar). Satu lintasan;
    cincin dibaca dari peta ASLI, jadi urutan pemrosesan tidak berpengaruh.
    """
    out = seg.copy()
    ring_k = np.ones((3, 3), np.uint8)
    h, w = seg.shape
    for c in np.unique(seg):
        k, lab, st, _ = cv2.connectedComponentsWithStats((seg == c).astype(np.uint8), connectivity=8)
        for i in np.nonzero(st[1:, cv2.CC_STAT_AREA] < n_min)[0] + 1:
            x, y, bw, bh = st[i, :4]
            y0, y1, x0, x1 = max(y - 1, 0), min(y + bh + 1, h), max(x - 1, 0), min(x + bw + 1, w)
            m = lab[y0:y1, x0:x1] == i
            ring = cv2.dilate(m.astype(np.uint8), ring_k).astype(bool) & ~m
            vals = seg[y0:y1, x0:x1][ring]
            if vals.size:
                out[y0:y1, x0:x1][m] = np.bincount(vals).argmax()
    return out


def mode_filter(gmap: np.ndarray, k: int) -> np.ndarray:
    """(b) Penghalusan tepi per grup: tiap piksel diberi grup yang paling sering muncul di jendela k x k.

    Hitungan per grup = box filter (tanpa normalisasi) atas mask grup itu; seri dimenangkan grup asli
    piksel (+0.5). Tonjolan / lekukan tepi yang lebih sempit dari ~k/2 hilang; batas antar grup dan
    siluet luar diperlakukan sama (background = grup 0).
    """
    out = gmap.copy()
    best = np.full(gmap.shape, -1.0, np.float32)
    for g in np.unique(gmap):
        m = (gmap == g).astype(np.float32)
        cnt = cv2.boxFilter(m, -1, (k, k), normalize=False, borderType=cv2.BORDER_REPLICATE) + 0.5 * m
        upd = cnt > best
        out[upd] = g
        best[upd] = cnt[upd]
    return out


def component_filter(lines: np.ndarray, m_min: int) -> np.ndarray:
    """(c) Buang komponen garis (8-arah) < m_min piksel."""
    k, lab, st, _ = cv2.connectedComponentsWithStats(lines.astype(np.uint8), connectivity=8)
    keep = np.zeros(k, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= m_min
    return keep[lab]


def build_lines(seg: np.ndarray, z: np.ndarray, lut: np.ndarray, thr: float, args, post: bool) -> dict:
    """Garis grup seg + garis DA. post=True: a (pulau) -> b (mode) -> garis -> c (komponen)."""
    cls_map = island_filter(seg, args.island_min) if post else seg
    gmap = lut[cls_map]
    if post:
        gmap = mode_filter(gmap, args.mode_k)
    s = seg_line_mask(gmap, IDENTITY_LUT, args.line_px)
    d = depth_line_mask(gmap, z, args.lines_erode, args.lines_eps, thr)
    pre_c = s | d
    lines = component_filter(pre_c, args.line_min) if post else pre_c
    return {"cls": cls_map, "lines": lines, "pre_c": pre_c}


def line_stats(lines: np.ndarray, cls_map: np.ndarray, hand_ids: list[int], args) -> dict:
    """Komponen kecil (< --small-comp-px) di zona tangan (definisi = diag-hands) + di seluruh frame, garis utama."""
    dk = np.ones((2 * args.hand_dilate + 1, 2 * args.hand_dilate + 1), np.uint8)
    zone = cv2.dilate(np.isin(cls_map, hand_ids).astype(np.uint8), dk).astype(bool)
    k, lab, st, _ = cv2.connectedComponentsWithStats(lines.astype(np.uint8), connectivity=8)
    areas = st[1:, cv2.CC_STAT_AREA]
    in_zone = np.zeros(k, bool)
    in_zone[np.unique(lab[zone & lines])] = True
    small = areas < args.small_comp_px
    return {"n_small_hand": int((small & in_zone[1:]).sum()), "n_small_all": int(small.sum()),
            "n_comps": int(k - 1), "main_px": int(areas.max()) if areas.size else 0,
            "line_px": int(lines.sum())}


def cmd_filtered(args, ctx) -> None:
    """Tanpa inferensi: [0.4B mentah | 0.4B + post | 0.8B + post], garis grup seg + DA-V2 p95, per jendela.

    Threshold DA per jendela = nilai per klip yang sudah dipakai (lines_thresholds.json untuk 73-92,
    seglines_da2_f183-202.json untuk 183-202); sama untuk semua panel.
    """
    names, frames_dir = ctx["names"], ctx["frames_dir"]
    lut = group_lut(args.t102c / "seg")
    classes = json.loads(probe.CLASSES_FILE.read_text(encoding="utf-8"))["classes"]
    hand_ids = [classes.index(c) for c in ("Left_Hand", "Right_Hand")]
    thr_73 = json.loads((args.exp / "lines_thresholds.json").read_text(encoding="utf-8"))
    thr_183 = json.loads((args.exp / "seglines_da2_f183-202.json").read_text(encoding="utf-8"))
    windows = {(73, 92): thr_73["sources"]["da2s_gpu"]["thresholds"]["p95"], (183, 202): thr_183["threshold"]}
    seg_dirs = {"0.4B": args.t102c / "seg", "0.8B": args.exp / "seg_seg_0.8b"}
    report = {"island_min": args.island_min, "mode_k": args.mode_k, "line_min": args.line_min,
              "small_comp_px": args.small_comp_px, "hand_dilate": args.hand_dilate, "windows": {}}
    variants = [("0.4B", False), ("0.4B", True), ("0.8B", False), ("0.8B", True)]
    for (a, b), thr in windows.items():
        win = frame_range(names, a, b)
        rows = []
        cache = {}
        for n in win:
            z = np.load(depth_path(args.exp, "da2s_gpu", n)).astype(np.float32)
            row = {"frame": n}
            for tag, post in variants:
                r = build_lines(read_class_map(seg_dirs[tag] / n), z, lut, thr, args, post)
                key = f"{tag}_{'post' if post else 'raw'}"
                row[key] = line_stats(r["lines"], r["cls"], hand_ids, args)
                if post:
                    row[key]["before_c"] = line_stats(r["pre_c"], r["cls"], hand_ids, args)
                cache[(n, key)] = r["lines"]
            rows.append(row)

        def render(n: str) -> np.ndarray:
            stem = Path(n).stem
            panels = []
            for key, label in (("0.4B_raw", "0.4B mentah"), ("0.4B_post", "0.4B + post"),
                               ("0.8B_post", "0.8B + post")):
                img = np.full(ctx["hw"] + (3,), probe.LINE_BG, np.uint8)
                img[cache[(n, key)]] = probe.LINE_FG
                ab.draw_label(img, [label, f"N{args.island_min} K{args.mode_k} M{args.line_min} thr {thr:.5f}", stem])
                panels.append(img)
            return np.hstack(panels)

        write_video(args.exp / f"filtered_f{a:03d}-{b:03d}.mp4", win, render, args.fps)
        report["windows"][f"{a}-{b}"] = {"da_threshold": thr, "frames": rows}
        print(f"[filtered] {a}-{b}: thr {thr:.5f}")
    dump(args.exp / "filtered_metrics.json", report)


# --- full run seg 0.8B + metrik + render full klip -------------------------
def valid_seg(path: Path, hw: tuple[int, int]) -> bool:
    """Class map terbaca, uint8, ukuran = frame, id kelas < NUM_CLASSES (file terpotong = tidak valid)."""
    m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    return m is not None and m.dtype == np.uint8 and m.shape == hw and int(m.max()) < probe.NUM_CLASSES


def cuda_mem(torch) -> dict:
    free, total = torch.cuda.mem_get_info()
    return {"free_mib": free / MIB, "total_mib": total / MIB,
            "reserved_mib": torch.cuda.memory_reserved() / MIB, "allocated_mib": torch.cuda.memory_allocated() / MIB}


def cmd_seg_full(args, ctx) -> None:
    """Full run seg semua frame, resume per frame (frame dengan output valid dilewati).

    Log per frame di-append ke full_<model>_frames.jsonl (waktu, peak VRAM reserved/allocated per frame,
    finite), jadi tetap tersimpan kalau run terhenti. OOM: frame + kondisi VRAM dicatat, lalu BERHENTI
    (tanpa retry otomatis).
    """
    import torch
    model_id = SEG_MODELS[args.target]
    names, frames_dir = ctx["names"], ctx["frames_dir"]
    seg_out = args.exp / f"seg_{args.target}"
    seg_out.mkdir(parents=True, exist_ok=True)
    log_path = args.exp / f"full_{args.target}_frames.jsonl"
    todo = [n for n in names if args.force or not valid_seg(seg_out / n, ctx["hw"])]
    rec = {"model": model_id, "precision": "fp16", "device": "cuda", "n_frames_clip": len(names),
           "n_skipped_existing": len(names) - len(todo), "n_todo": len(todo), "sys_mem_before": sys_mem(),
           "nvidia_smi_before": probe.nvidia_smi(),  # SEBELUM CUDA init
           "cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF")}
    print(f"[seg-full] {args.target}: {len(todo)} frame ({rec['n_skipped_existing']} sudah ada & valid, dilewati)")
    print(f"  nvidia-smi sebelum CUDA init: {rec['nvidia_smi_before']}")
    if not todo:
        return
    torch.cuda.init()
    rec["cuda_before_load"] = cuda_mem(torch)
    print(f"  torch.cuda.mem_get_info sebelum load: bebas {rec['cuda_before_load']['free_mib']:.0f} / "
          f"{rec['cuda_before_load']['total_mib']:.0f} MiB")
    t0 = time.perf_counter()
    runner = None
    done = 0
    try:
        runner = Model("seg", model_id, "cuda", "fp16")
        rec["load_s"] = runner.load_s
        rec["cuda_after_load"] = cuda_mem(torch)
        with log_path.open("a", encoding="utf-8") as log:
            for n in todo:
                torch.cuda.reset_peak_memory_stats()
                ts = time.perf_counter()
                arr, info = runner.infer(ab.read_rgb(frames_dir / n))
                cv2.imwrite(str(seg_out / n), arr)
                info.update({"frame": n, "total_s": time.perf_counter() - ts,
                             "peak_reserved_mib": torch.cuda.max_memory_reserved() / MIB,
                             "peak_alloc_mib": torch.cuda.max_memory_allocated() / MIB})
                log.write(json.dumps(info) + "\n")
                log.flush()
                done += 1
                if done % 25 == 0:
                    print(f"  {done}/{len(todo)}", flush=True)
        rec["status"] = "ok"
    except Exception as e:  # noqa: BLE001 - OOM = hasil; berhenti, tanpa retry
        if not probe.is_oom(e):
            raise
        rec.update({"status": "oom", "oom_frame": todo[done] if runner is not None else "load model",
                    "error": str(e).splitlines()[0][:300], "cuda_at_oom": cuda_mem(torch)})
        print(f"  OOM di {rec['oom_frame']}: {rec['cuda_at_oom']} -> berhenti")
    finally:
        if runner is not None:
            runner.close()
        rec.update(proc_mem())
    rec["frames_done_this_run"] = done
    rec["wall_s"] = time.perf_counter() - t0
    frames = {}
    for line in log_path.read_text(encoding="utf-8").splitlines():
        f = json.loads(line)
        frames[f["frame"]] = f  # run terakhir menang
    fr = [frames[n] for n in names if n in frames]
    timed = [f for f in fr if f["frame"] not in todo[:args.warmup]] or fr
    rec["log"] = {"n_frames_logged": len(fr), "n_nonfinite": sum(not f["finite"] for f in fr),
                  "infer_s": probe.pct_stats([f["infer_s"] for f in timed]),
                  "total_s": probe.pct_stats([f["total_s"] for f in timed]),
                  "peak_reserved_mib_max": max(f["peak_reserved_mib"] for f in fr),
                  "peak_alloc_mib_max": max(f["peak_alloc_mib"] for f in fr)} if fr else {}
    rec["n_valid_outputs"] = sum(valid_seg(seg_out / n, ctx["hw"]) for n in names)
    dump(args.exp / f"full_{args.target}.json", rec)
    print(f"  status {rec['status']}, output valid {rec['n_valid_outputs']}/{len(names)}, log {rec['log']}")


def qc_args(args):
    """Threshold QC sama dengan sapiens2_probe.py / ab_segment.py (tabel QC docs/01 stage [2])."""
    import types
    return types.SimpleNamespace(iou_min=args.iou_min, area_min=args.area_min, area_max=args.area_max,
                                 blob_min=args.blob_min)


def cmd_seg_metrics(args, ctx) -> None:
    """Metrik foreground + label_agreement_prev full klip, 0.8B vs 0.4B (fungsi sapiens2_probe.py yang sama)."""
    sys.stdout.reconfigure(encoding="utf-8")
    names = ctx["names"]
    seg08 = args.exp / f"seg_{args.target}"
    bad = [n for n in names if not valid_seg(seg08 / n, ctx["hw"])]
    if bad:
        ab.fail(f"{len(bad)} frame {args.target} belum valid, mis. {bad[0]}")
    id2label = probe.resolve_id2label({int(k): v for k, v in json.loads(
        (args.t102c / "infer_seg.json").read_text(encoding="utf-8"))["id2label"].items()})
    qa = qc_args(args)
    res, rows_by = {}, {}
    for tag, d in (("0.4B", args.t102c / "seg"), (args.target, seg08)):
        rows, counts = probe.compute_metrics(args.t102c, names, args.mp_masks, False, qa, seg_dir=d)
        res[tag] = probe.summarize_metrics(rows, counts, id2label, qa)
        rows_by[tag] = rows
    same = [float((read_class_map(args.t102c / "seg" / n) == read_class_map(seg08 / n)).mean()) for n in names]
    per_frame = []
    for n, r4, r8, s in zip(names, rows_by["0.4B"], rows_by[args.target], same):
        per_frame.append({"frame": n, "same_class_vs_04b": round(s, 5),
                          **{f"{k}_04b": r4[k] for k in ("iou_prev", "label_agreement_prev", "area_ratio")},
                          **{f"{k}_08b": r8[k] for k in ("iou_prev", "label_agreement_prev", "area_ratio",
                                                         "big_blobs", "cross_iou_mp")}})
    ab.write_csv(per_frame, args.exp / f"metrics_full_{args.target}.csv")
    out = {"qc": vars(qa), "same_class_vs_04b": ab.stats_block(same), "metrics": res}
    dump(args.exp / f"metrics_full_{args.target}.json", out)

    cols = ["iou_prev_mean", "iou_prev_median", "iou_prev_min", "n_iou_prev_below_min", "n_area_below_min",
            "n_area_above_max", "n_multi_big_blob"]
    print("| model | " + " | ".join(cols) + " | label_agr mean | median | min | cross_iou_mp mean |")
    print("|---" * (len(cols) + 5) + "|")
    for tag, s in res.items():
        la, cm = s["label_agreement_prev"], s["cross_iou_mp"]
        print(f"| {tag} | " + " | ".join(f"{s[c]:.4f}" if isinstance(s[c], float) else str(s[c]) for c in cols)
              + f" | {la['mean']:.4f} | {la['median']:.4f} | {la['min']:.4f} | {cm.get('mean', float('nan')):.4f} |")
    print(f"kelas sama 0.8B vs 0.4B: {out['same_class_vs_04b']}")
    for tag in res:
        worst = sorted((r for r in rows_by[tag] if r["label_agreement_prev"] != ""),
                       key=lambda r: r["label_agreement_prev"])[:5]
        print(f"{tag} label_agreement_prev terendah: " +
              ", ".join(f"{Path(r['frame']).stem[-5:]}={r['label_agreement_prev']:.4f}" for r in worst))


def cmd_filtered_full(args, ctx) -> None:
    """Render full klip [asli | garis 0.8B + post (N, K, M) + DA-V2 p<pct>].

    Threshold DA = persentil J per klip (semua frame), dihitung di foreground peta grup 0.8B SETELAH
    post-processing a+b, ter-erode (--lines-erode).
    """
    names, frames_dir = ctx["names"], ctx["frames_dir"]
    seg08 = args.exp / f"seg_{args.target}"
    bad = [n for n in names if not (valid_seg(seg08 / n, ctx["hw"])
                                    and valid_depth(depth_path(args.exp, "da2s_gpu", n), ctx["hw"]))]
    if bad:
        ab.fail(f"{len(bad)} frame belum lengkap (seg {args.target} / DA), mis. {bad[0]}")
    lut = group_lut(args.t102c / "seg")
    kernel = np.ones((args.lines_erode, args.lines_erode), np.uint8)
    gmaps, vals = {}, []
    for n in names:
        g = mode_filter(lut[island_filter(read_class_map(seg08 / n), args.island_min)], args.mode_k)
        gmaps[n] = g
        inner = cv2.erode((g != 0).astype(np.uint8), kernel).astype(bool)
        z = np.load(depth_path(args.exp, "da2s_gpu", n)).astype(np.float32)
        vals.append(relative_jump(z, inner, args.lines_eps)[inner])
    thr = float(np.percentile(np.concatenate(vals), args.seglines_pct))
    print(f"[filtered-full] {args.target}: DA p{args.seglines_pct:g} per klip = {thr:.5f}")
    stats = []

    def render(n: str) -> np.ndarray:
        stem = Path(n).stem
        g = gmaps[n]
        z = np.load(depth_path(args.exp, "da2s_gpu", n)).astype(np.float32)
        lines = component_filter(seg_line_mask(g, IDENTITY_LUT, args.line_px)
                                 | depth_line_mask(g, z, args.lines_erode, args.lines_eps, thr), args.line_min)
        stats.append({"frame": n, "line_px": int(lines.sum())})
        p0 = cv2.imread(str(frames_dir / n), cv2.IMREAD_COLOR)
        ab.draw_label(p0, ["asli", stem])
        p1 = np.full(ctx["hw"] + (3,), probe.LINE_BG, np.uint8)
        p1[lines] = probe.LINE_FG
        ab.draw_label(p1, [f"{args.target} + post N{args.island_min} K{args.mode_k} M{args.line_min}",
                           f"DA-V2 p{args.seglines_pct:g} thr {thr:.5f}", stem])
        return np.hstack([p0, p1])

    write_video(args.exp / f"filtered_full_{args.target.replace('seg_', '')}.mp4", names, render, args.fps)
    dump(args.exp / f"filtered_full_{args.target.replace('seg_', '')}.json",
         {"island_min": args.island_min, "mode_k": args.mode_k, "line_min": args.line_min,
          "percentile": args.seglines_pct, "da_threshold": thr, "threshold_fg": f"{args.target} + post a+b",
          "frames": stats})


# --- E2 --------------------------------------------------------------------
def e2_window(args, names: list[str]) -> tuple[list[str], str, float]:
    rows = [r for r in csv.DictReader((args.t102c / "metrics_per_frame.csv").open(encoding="utf-8"))
            if r["label_agreement_prev"] != ""]
    worst = min(rows, key=lambda r: float(r["label_agreement_prev"]))
    i = names.index(worst["frame"])
    start = min(max(i - args.window // 2, 0), len(names) - args.window)
    return names[start:start + args.window], worst["frame"], float(worst["label_agreement_prev"])


def cmd_seg(args, ctx) -> None:
    model_id = SEG_MODELS[args.target]
    names, frames_dir = ctx["names"], ctx["frames_dir"]
    window, worst, worst_val = e2_window(args, names)
    out_json = args.exp / f"e2_{args.target}.json"
    if args.seg_frames:  # jendela eksplisit, JSON terpisah (jendela default tidak ditimpa)
        window = frame_range(names, *args.seg_frames)
        out_json = args.exp / f"e2_{args.target}_f{args.seg_frames[0]:03d}-{args.seg_frames[1]:03d}.json"
    seg_out = args.exp / f"seg_{args.target}"
    device = "cpu" if args.device == "cpu" else "cuda"
    precision = "fp32" if device == "cpu" else "fp16"
    print(f"[E2] {args.target} {device} {precision}: frame {window[0]} .. {window[-1]} (terendah {worst} {worst_val})")

    attempts = json.loads(out_json.read_text(encoding="utf-8")).get("attempts", []) if (
        out_json.is_file() and device == "cpu") else []
    seg_out.mkdir(parents=True, exist_ok=True)

    def save(n: str, arr: np.ndarray) -> None:
        cv2.imwrite(str(seg_out / n), arr)

    rec, _ = run_frames("seg", model_id, device, precision, False, frames_dir, window, args.warmup, save=save)
    rec["weights_gib"] = weights_gib(model_id)
    n_classes = len(json.loads(probe.CLASSES_FILE.read_text(encoding="utf-8"))["classes"])
    if rec.get("num_labels") not in (None, n_classes):
        ab.fail(f"self-check gagal: num_labels {model_id} = {rec['num_labels']}, "
                f"{probe.CLASSES_FILE.name} punya {n_classes} kelas")
    attempts.append({k: v for k, v in rec.items() if k != "frames"})
    result = {"model": args.target, "model_id": model_id, "window": [window[0], window[-1]],
              "worst_frame_04b": worst, "worst_label_agreement_prev_04b": worst_val,
              "attempts": attempts, "run": rec}

    if rec["status"] == "oom" and device == "cuda":
        mem = sys_mem()
        need = (rec["weights_gib"] or 0) * args.cpu_mem_factor
        left = mem["avail_gib"] - need
        ram_ok = left >= args.ram_reserve * mem["total_gib"]
        result["cpu_fallback_check"] = {**mem, "need_gib_est": need, "left_gib_est": left,
                                        "reserve_gib": args.ram_reserve * mem["total_gib"], "ok": ram_ok}
        dump(out_json, result)
        print(f"  GPU OOM -> cek RAM CPU fp32: {result['cpu_fallback_check']}")
        if ram_ok and args.no_cpu_fallback:
            print("  --no-cpu-fallback -> berhenti (OOM dicatat sebagai hasil)")
        elif ram_ok:
            subprocess.run([sys.executable, __file__, "seg", args.target, "--device", "cpu"], check=False)
        return
    if rec["passed"]:
        result["frames"] = e2_metrics(args, window, seg_out)
        result["summary"] = e2_summary(result["frames"])
        render_e2(args, window, seg_out, frames_dir)
    dump(out_json, result)


def e2_metrics(args, window: list[str], seg_out: Path) -> list[dict]:
    ref_dir = args.t102c / "seg"
    rows, prev_m, prev_r = [], None, None
    for n in window:
        m, r = read_class_map(seg_out / n), read_class_map(ref_dir / n)
        row = {"frame": n, "same_class_vs_04b": float((m == r).mean()),
               "fg_iou_vs_04b": ab.iou(m != 0, r != 0),
               "label_agreement_prev": label_agreement(m, prev_m) if prev_m is not None else None,
               "label_agreement_prev_04b": label_agreement(r, prev_r) if prev_r is not None else None}
        rows.append(row)
        prev_m, prev_r = m, r
    return rows


def e2_summary(rows: list[dict]) -> dict:
    s = {}
    for k in ("same_class_vs_04b", "fg_iou_vs_04b", "label_agreement_prev", "label_agreement_prev_04b"):
        s[k] = ab.stats_block([r[k] for r in rows if r[k] is not None])
    return s


def render_e2(args, window: list[str], seg_out: Path, frames_dir: Path) -> None:
    ref_dir = args.t102c / "seg"
    lut = group_lut(ref_dir)
    pal = probe.palette(probe.NUM_CLASSES)
    tag = args.target.replace("seg_", "")

    def render(n: str) -> np.ndarray:
        stem = Path(n).stem
        bgr = cv2.imread(str(frames_dir / n), cv2.IMREAD_COLOR)
        m, r = read_class_map(seg_out / n), read_class_map(ref_dir / n)
        p0 = bgr.copy()
        ab.draw_label(p0, ["asli", stem])
        p1 = pal[m]
        ab.draw_label(p1, [f"kelas seg {tag}", stem])
        p2 = line_panel(m, lut, args.line_px)
        ab.draw_label(p2, [f"garis grup seg {tag}", stem])
        p3 = line_panel(r, lut, args.line_px)
        ab.draw_label(p3, ["garis grup seg 0.4B", stem])
        return np.hstack([p0, p1, p2, p3])

    a, b = probe.frame_no(window[0]), probe.frame_no(window[-1])
    write_video(args.exp / f"side_by_side_{args.target}_f{a:03d}-{b:03d}.mp4", window, render, args.fps)


# --- report ----------------------------------------------------------------
def fmt(v, spec: str = ".3f") -> str:
    return "–" if v is None or v == "" else format(v, spec)


def cmd_report(args, ctx) -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # tabel markdown memakai "–" / "↳"; konsol Windows cp1252
    print("\n### E1 benchmark (20 frame, s/frame = forward + post-process, tanpa 3 warm-up)\n")
    print("| sumber | status | NaN/inf | s mean | s med | s p95 | peak alloc MiB | peak reserved MiB | peak RAM MiB | input |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for src in DEPTH_SOURCES:
        p = args.exp / f"bench_{src}.json"
        if not p.is_file():
            print(f"| {src} | belum jalan |||||||||")
            continue
        r = json.loads(p.read_text(encoding="utf-8"))
        t = r.get("infer_s", {})
        nan = (f"{r['n_nonfinite_frames']}/{r['n_frames']}" if r["status"] == "ok" else "–")
        print(f"| {src} | {r['status']} | {nan} | {fmt(t.get('mean'))} | {fmt(t.get('median'))} | "
              f"{fmt(t.get('p95'))} | {fmt(r.get('peak_alloc_mib'), '.0f')} | {fmt(r.get('peak_reserved_mib'), '.0f')} | "
              f"{fmt(r.get('peak_ws_mib'), '.0f')} | {r.get('input_hw', '–')} |")
        if r["status"] != "ok":
            print(f"|  ↳ error: {r.get('error')} |||||||||")

    print("\n### E1 full run\n")
    print("| sumber | frame | NaN/inf frame | infer mean s | total mean s/frame | wall mnt | peak alloc MiB | peak RAM MiB |")
    print("|---|---|---|---|---|---|---|---|")
    for src in DEPTH_SOURCES:
        p = args.exp / f"infer_{src}.json"
        if not p.is_file():
            continue
        r = json.loads(p.read_text(encoding="utf-8"))
        print(f"| {src} | {r['n_frames']} | {r.get('n_nonfinite_frames', '–')} | {fmt(r.get('infer_s', {}).get('mean'))} | "
              f"{fmt(r.get('total_s', {}).get('mean'))} | {fmt(r['wall_s'] / 60, '.1f')} | "
              f"{fmt(r.get('peak_alloc_mib'), '.0f')} | {fmt(r.get('peak_ws_mib'), '.0f')} |")

    print("\n### E2 seg lebih besar\n")
    print("| model | percobaan | status | s mean | s med | peak alloc MiB | peak reserved MiB | peak RAM MiB |"
          " sama kelas vs 0.4B | fg IoU vs 0.4B | label_agr_prev | label_agr_prev 0.4B |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for m in SEG_MODELS:
        p = args.exp / f"e2_{m}.json"
        if not p.is_file():
            print(f"| {m} | {SEG_SKIPPED.get(m, 'belum jalan')} |||||||||||")
            continue
        r = json.loads(p.read_text(encoding="utf-8"))
        s = r.get("summary", {})
        for k, a in enumerate(r["attempts"]):
            t = a.get("infer_s", {})
            last = k == len(r["attempts"]) - 1 and a["status"] == "ok"
            print(f"| {m} | {a['device']} {a['precision']} | {a['status']} | {fmt(t.get('mean'))} | {fmt(t.get('median'))} | "
                  f"{fmt(a.get('peak_alloc_mib'), '.0f')} | {fmt(a.get('peak_reserved_mib'), '.0f')} | "
                  f"{fmt(a.get('peak_ws_mib'), '.0f')} | "
                  + (" | ".join(f"{fmt(s.get(c, {}).get('mean'), '.4f')} (min {fmt(s.get(c, {}).get('min'), '.4f')})"
                                for c in ("same_class_vs_04b", "fg_iou_vs_04b", "label_agreement_prev",
                                          "label_agreement_prev_04b")) if last else "– | – | – | –") + " |")
        if "cpu_fallback_check" in r:
            print(f"|  ↳ cek RAM CPU: {json.dumps({k: round(v, 2) if isinstance(v, float) else v for k, v in r['cpu_fallback_check'].items()})} |||||||||||")
        if r.get("frames"):
            print(f"\n{m} per frame (label_agreement_prev model / 0.4B, sama kelas vs 0.4B):")
            for f in r["frames"]:
                print(f"  {Path(f['frame']).stem}: {fmt(f['label_agreement_prev'], '.4f')} / "
                      f"{fmt(f['label_agreement_prev_04b'], '.4f')}, {f['same_class_vs_04b']:.4f}")


# --- main ------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("cmd", choices=["bench", "full", "render", "compare", "lines", "seglines", "diag-hands",
                                   "filtered", "seg", "seg-full", "seg-metrics", "filtered-full", "report"])
    p.add_argument("target", nargs="?", default="")
    p.add_argument("--work", type=Path, default=ROOT / "work")
    p.add_argument("--t102c", type=Path, default=ROOT / "work" / "t102c", help="full run 0.4B (hanya dibaca)")
    p.add_argument("--exp", type=Path, default=ROOT / "work" / "t102c" / "exp")
    p.add_argument("--bench-frames", type=int, default=20)
    p.add_argument("--warmup", type=int, default=3, help="frame awal yang dibuang dari statistik waktu")
    p.add_argument("--window", type=int, default=20, help="E2: jumlah frame berurutan")
    p.add_argument("--mp-masks", type=Path, default=ROOT / "work" / "ab_t102a" / "masks_mediapipe")
    # Threshold QC sama dengan sapiens2_probe.py / ab_segment.py (tabel QC docs/01 stage [2])
    p.add_argument("--iou-min", type=float, default=0.55)
    p.add_argument("--area-min", type=float, default=0.03)
    p.add_argument("--area-max", type=float, default=0.70)
    p.add_argument("--blob-min", type=float, default=0.05)
    p.add_argument("--seg-frames", type=int, nargs=2, default=None,
                   help="E2: jendela frame eksplisit A B (default: sekitar label_agreement_prev terendah)")
    p.add_argument("--device", choices=["auto", "cpu"], default="auto", help="E2: auto = GPU fp16")
    p.add_argument("--no-cpu-fallback", action="store_true", help="E2: GPU OOM -> catat dan berhenti")
    p.add_argument("--cpu-mem-factor", type=float, default=1.3, help="E2: perkiraan RAM CPU = bobot x faktor")
    p.add_argument("--ram-reserve", type=float, default=0.25, help="E2: sisa RAM minimum (fraksi total)")
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--grid-frames", default="1,4,73,78,82,87,92,100,167,236,240")
    p.add_argument("--leg-frames", type=int, nargs=2, default=[73, 92], help="E1: klip pendek kaki menyilang")
    p.add_argument("--compare-grid", default="73,78,82,87,92", help="frame untuk frames_grid_compare.png")
    p.add_argument("--lines-grid", default="73,82,92", help="frame untuk frames_grid_lines_<sumber>.png")
    p.add_argument("--lines-erode", type=int, default=5, help="kernel erosi foreground (px) untuk garis kedalaman")
    p.add_argument("--lines-eps", type=float, default=1e-6, help="nilai kedalaman minimum yang dianggap valid")
    p.add_argument("--seglines-pct", type=float, default=95.0, help="persentil threshold DA untuk seglines/diag-hands")
    p.add_argument("--hand-dilate", type=int, default=15, help="diag-hands: radius dilate kelas tangan (px)")
    p.add_argument("--small-comp-px", type=int, default=100, help="diag-hands: komponen garis 'kecil' (px)")
    # Post-processing (filtered); alasan nilai: update log T-102c docs/05 (2026-09-29)
    p.add_argument("--island-min", type=int, default=30, help="(a) N: pulau kelas < N px diganti mayoritas sekitar")
    p.add_argument("--mode-k", type=int, default=3, help="(b) K: kernel mode filter peta grup")
    p.add_argument("--line-min", type=int, default=5, help="(c) M: komponen garis < M px dibuang")
    p.add_argument("--grid-scale", type=float, default=0.5)
    p.add_argument("--line-px", type=int, default=2)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    valid = {"bench": DEPTH_SOURCES, "full": DEPTH_SOURCES, "render": DEPTH_SOURCES, "seg": SEG_MODELS,
             "seg-full": SEG_MODELS, "seg-metrics": SEG_MODELS, "filtered-full": SEG_MODELS}
    if args.cmd in valid and args.target not in valid[args.cmd]:
        ab.fail(f"target untuk {args.cmd} harus salah satu dari {list(valid[args.cmd])}")
    self_check()
    ctx = load_ctx(args)
    {"bench": cmd_bench, "full": cmd_full, "render": cmd_render, "compare": cmd_compare, "lines": cmd_lines,
     "seglines": cmd_seglines, "diag-hands": cmd_diag_hands, "filtered": cmd_filtered, "seg": cmd_seg,
     "seg-full": cmd_seg_full, "seg-metrics": cmd_seg_metrics, "filtered-full": cmd_filtered_full,
     "report": cmd_report}[args.cmd](args, ctx)
    return 0


if __name__ == "__main__":
    sys.exit(main())
