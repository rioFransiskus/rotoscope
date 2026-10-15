"""T-602 Tahap 1a (CPU, baca-saja terhadap klip, TANPA unduhan bobot / GPU): studi pra-proses, isolasi target dengan SAM 2.1 tiny.

    python scripts/t602_study.py hash before|after   # F1: sha256 seluruh work/clips/{test,test_short,Klip2,klip3,klip4,klip5}/ + out/ -> work/t602/
    python scripts/t602_study.py meta                # metadata HF facebook/sam2.1-hiera-tiny (API + HEAD, tanpa isi bobot) + jumlah parameter
                                                     # (model di device meta) + lingkungan (GUI OpenCV, tkinter, layar, RAM, disk) -> work/t602/meta.json
    python scripts/t602_study.py letterbox           # potong letterbox / kotak teks statis: 5 klip (rentang penuh, klip5 juga f0–f91)
                                                     # + jendela geser 24 frame (uji positif palsu) -> work/t602/letterbox.json
    python scripts/t602_study.py window              # klip3: frame dengan blob gabungan (>= 2 wajah V1 dalam satu blob) + pilihan
                                                     # jendela 24 frame tanpa cut -> work/t602/window.json

Tahap 1b-1 (docs/04 "Keputusan T-602 Tahap 1b (2026-10-15)", desain D1–D8):
    python scripts/t602_study.py hash before_1b1|after_1b1
    python scripts/t602_study.py download            # facebook/sam2.1-hiera-tiny @ pin -> cache HF + cek sha256 -> work/t602/download.json
    python scripts/t602_study.py crop                # kotak area aktif R2 (rentang penuh) 5 klip -> work/t602/crop.json
    python scripts/t602_study.py pick KLIP --frame N [--dry-run]
                                                     # jendela klik target (Rio) -> work/t602/points/KLIP.json;
                                                     # --dry-run: render tampilan + petunjuk ke PNG (work/t602_scratch/pick_dryrun/)
    python scripts/t602_study.py run KLIP [--until LANGKAH] [--redo LANGKAH ...] [--dry-run] [--points PATH]
                                                     # GPU (Rio, PowerShell terpisah): sam (fp32 + pembanding fp16) -> pratinjau -> C (klip5)
                                                     # -> M1; dapat dilanjutkan; keluaran work/t602_scratch/KLIP/
    (internal) python scripts/t602_study.py _sam KLIP --points P --out DIR [--device cuda|cpu] [--max-frames N] [--no-fp16]
    (uji, CPU) python scripts/t602_study.py _uji KLIP --points P [--frames 3]   # -> work/t602_scratch/uji/

Penutupan (docs/04 "Keputusan akhir T-602"; 1b-2 tidak dijalankan):
    python scripts/t602_study.py hash before_tutup|after_tutup   # F1, keduanya dibandingkan dengan after_1b1

Keputusan dan lingkup: docs/04 "Keputusan T-602 (studi pra-proses)". Hanya MEMBACA work/clips/* dan out/; keluaran ke work/t602/ (di-ignore git).
`meta` hanya meminta metadata (model_info, HEAD per berkas) dan membaca config.json kecil ke memori (tidak disimpan, tidak masuk cache HF);
tidak ada badan berkas bobot yang diunduh. CUDA tidak disentuh (tanpa torch.cuda.*).
Memakai ulang: hash (`scripts/t701_study.py`), wajah V1 + blob tubuh (`t701_study.frame_people` / `faces_v1_of`).
"""

from __future__ import annotations

import ctypes
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import t701_study as t701  # noqa: E402
from t305d_stage3 import runs_str  # noqa: E402

from rotoscope import depth as dep  # noqa: E402
from rotoscope.config import load_pipeline  # noqa: E402
from rotoscope.ingest import FRAME_PATTERN  # noqa: E402
from rotoscope.stage_common import CLIP_KEY, MIB, clip_identity, write_bytes_atomic, write_json_atomic  # noqa: E402

CLIPS_DIR = ROOT / "work" / "clips"
OUT = ROOT / "work" / "t602"
HASH_CLIPS = ("test", "test_short", "Klip2", "klip3", "klip4", "klip5")
PREV_F1 = ROOT / "work" / "t305" / "1b" / "after_sha256.txt"     # F1 terakhir (T-305 induk 1b), pembanding subset
HASH_LABELS = ("before", "after", "before_1b1", "after_1b1", "before_1b2", "after_1b2", "before_tutup", "after_tutup")
HASH_REF = {"before_tutup": "after_1b1", "after_tutup": "after_1b1"}   # penutupan (1b-2 tidak dijalankan): pembanding = hash 1b-1

# ── SAM 2.1 tiny (metadata saja) ───────────────────
HF_REPO = "facebook/sam2.1-hiera-tiny"
HF_FILES_TRANSFORMERS = ("model.safetensors", "config.json", "preprocessor_config.json", "processor_config.json",
                         "video_preprocessor_config.json", "README.md")
HF_FILES_SAM2_PKG = ("sam2.1_hiera_tiny.pt", "sam2.1_hiera_t.yaml")
SAM2_PKG_REQUIRES = ("torch", "torchvision", "numpy", "tqdm", "hydra-core", "iopath", "pillow")   # setup.py facebookresearch/sam2
HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"

# ── potong letterbox / kotak teks (ditetapkan saat eksplorasi profil, SEBELUM jendela geser diukur) ──
STATIC_STD_MAX = 3.0        # piksel statis: simpangan baku temporal (maks antar kanal BGR, 0–255) <= 3
STATIC_LINE_FRAC = 0.99     # baris / kolom statis: >= 99% pikselnya statis (di dalam persegi aktif saat itu)
UNIFORM_STD_MAX = 6.0       # baris / kolom seragam: simpangan baku spasial warna rata-rata temporal (maks antar kanal) <= 6
BAND_MIN_PX = 8             # pita letterbox minimum agar baris statis TIDAK seragam (kotak teks) di sebelah dalamnya ikut dipotong
MIN_ACTIVE_FRAC = 0.10      # area aktif < 10% frame -> tidak dipotong (pengaman)
WIN_LEN, WIN_STEP = 24, 12  # jendela geser (uji positif palsu untuk klip pendek setelah potong durasi)
WIN_TOL_PX = 4              # jendela "positif palsu" bila potongannya > acuan + 4 px di salah satu sisi
LB_CLIPS = ("test", "Klip2", "klip3", "klip4", "klip5")
EXPECTED = {"klip5": {"y": [348, 1009], "sumber": "docs/04 'Hasil Tahap 1b' (profil baris f0–f91)"}}
STUDY_RANGES = {"klip5": [(0, 91)], "klip3": [(0, 23)]}
EDGES = ("top", "bottom", "left", "right")
OPP = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}
RULES = ("R1", "R2")

# ── jendela klip3 ──────────────────────────────────
WINDOW_CLIP = "klip3"
WINDOW_LEN = 24

# ── Tahap 1b-1 (docs/04 "Keputusan T-602 Tahap 1b", D1–D8) ──
HF_PIN = "de431c4043854a71d8101e17995dfe596bf101a5"
HF_PIN_SHA256 = "48c14467e5cf9e51870511feb72c89688e82dd74523142c0538b663e193ac2a7"   # model.safetensors (LFS)
SCR = ROOT / "work" / "t602_scratch"
CROP_JSON = OUT / "crop.json"
POINTS_DIR = OUT / "points"
PREVIEW_DIR = OUT / "nilai_1b" / "pratinjau"
POINTS_FORMAT = "t602-points/1"
WINDOWS = {"klip5": (0, 91), "klip3": (85, 108)}          # inklusif; titik diklik di frame awal jendela (D2)
STEPS = {"klip5": ("sam", "pratinjau", "C", "M1"), "klip3": ("sam", "pratinjau", "M1")}   # D7
SAM_MIN_FREE_RAM_GIB = 4.0
SAM_MIN_FREE_VRAM_MIB = 2000
FP16_COMPARE_FRAMES = 8
M1_BG = (128, 128, 128)                                    # keputusan 5
MASK_DILATE_PX = 2                                         # keputusan 5 (elips 5 x 5)
SEG_MODEL = "0.8b"
VRAM_SAMPLE_S = 2.0
MIN_DISK_FREE_GIB = 2.0
EST_SEG_S_PER_FRAME = 16.8                                 # seg 0.8B klip5 T-305 (s / frame)
PICK_PANEL_W = 400
PICK_MAX_H_FRAC = 0.80                                     # tinggi kanvas <= 80% tinggi layar fisik (judul + taskbar)
PICK_MAX_W_FRAC = 0.95
PICK_PANEL_MIN_H = 600
TARGET_COLORS = ((0, 200, 0), (255, 160, 0), (0, 215, 255), (255, 0, 255), (0, 140, 255), (255, 255, 0))  # BGR
TARGET_COLOR_NAMES = ("hijau", "biru", "kuning", "ungu", "oranye", "toska")
NEG_COLOR = (0, 0, 255)
KEY_ENTER, KEY_ESC = (13, 10), 27


def jsave(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"  -> {p.relative_to(ROOT).as_posix()}")


# ── F1 ─────────────────────────────────────────────
def hash_files() -> list[Path]:
    files: list[Path] = []
    for c in HASH_CLIPS:
        files += [p for p in sorted((CLIPS_DIR / c).rglob("*")) if p.is_file()]
    files += [p for p in sorted((ROOT / "out").rglob("*")) if p.is_file()]
    return files


def read_list(p: Path) -> dict[str, str]:
    return {r: h for h, r in (line.split("  ", 1) for line in p.read_text(encoding="utf-8").splitlines() if line)}


def cmd_hash(which: str) -> None:
    cur = {p.relative_to(ROOT).as_posix(): t701.file_sha(p) for p in hash_files()}
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{which}_sha256.txt"
    path.write_text("".join(f"{h}  {r}\n" for r, h in cur.items()), encoding="utf-8")
    n_qc = sum(r.endswith("/qc_report.json") for r in cur)
    print(f"{path.relative_to(ROOT).as_posix()}: {len(cur)} berkas (qc_report.json: {n_qc})")
    prev = read_list(PREV_F1)
    sub = {"prev_list": PREV_F1.relative_to(ROOT).as_posix(), "prev_n": len(prev),
           "missing_now": sorted(set(prev) - set(cur)), "changed_vs_prev": sorted(r for r in prev if r in cur and cur[r] != prev[r]),
           "not_in_prev": sorted(set(cur) - set(prev))}
    print(f"  vs F1 T-305 1b ({len(prev)} berkas): hilang {len(sub['missing_now'])}, berubah {len(sub['changed_vs_prev'])}, "
          f"tambahan (di luar daftar lama) {len(sub['not_in_prev'])}: {sub['not_in_prev'][:8]}")
    res = {"which": which, "n": len(cur), "n_qc_report": n_qc, "vs_t305_1b": sub}
    ref = HASH_REF.get(which) or (which.replace("after", "before", 1) if which.startswith("after") else None)
    if ref:
        before = read_list(OUT / f"{ref}_sha256.txt")
        res.update({"ref": ref, "before": len(before), "missing": sorted(set(before) - set(cur)), "extra": sorted(set(cur) - set(before)),
                    "changed": sorted(r for r in before if r in cur and cur[r] != before[r])})
        res["identical"] = not (res["missing"] or res["extra"] or res["changed"])
        print(f"  {ref} {len(before)} / {which} {len(cur)}: identik = {res['identical']}")
    jsave(OUT / f"f1_{which}.json", res)


# ── meta ───────────────────────────────────────────
def mem_status() -> dict:
    class MS(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    m = MS()
    m.dwLength = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    gib = 1 << 30
    return {"ram_total_gib": round(m.ullTotalPhys / gib, 2), "ram_free_gib": round(m.ullAvailPhys / gib, 2), "load_pct": m.dwMemoryLoad,
            "commit_total_gib": round(m.ullTotalPageFile / gib, 2), "commit_free_gib": round(m.ullAvailPageFile / gib, 2)}


def dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.exists() else 0


def gui_env() -> dict:
    bi = cv2.getBuildInformation()
    i = bi.find("GUI:")
    gui = [ln.strip() for ln in bi[i:].splitlines()[:3]] if i >= 0 else []
    try:
        import tkinter
        tk = {"ok": True, "tk_version": tkinter.TkVersion}
    except Exception as e:  # noqa: BLE001
        tk = {"ok": False, "error": repr(e)}
    u = ctypes.windll.user32
    logical = [u.GetSystemMetrics(0), u.GetSystemMetrics(1)]
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except OSError:
        pass
    physical = [u.GetSystemMetrics(0), u.GetSystemMetrics(1)]
    from importlib import metadata as md
    pk = {d.metadata["Name"].lower(): d.version for d in md.distributions()}
    return {"cv2_version": cv2.__version__, "cv2_gui": gui,
            "cv2_has": {n: hasattr(cv2, n) for n in ("imshow", "setMouseCallback", "EVENT_LBUTTONDOWN", "EVENT_RBUTTONDOWN", "waitKey")},
            "opencv_dists": {k: v for k, v in pk.items() if "opencv" in k}, "tkinter": tk,
            "screen_logical_wh": logical, "screen_physical_wh": physical,
            "windows_scale_pct": round(100 * physical[0] / logical[0]) if logical[0] else None,
            "sam2_pkg_requires_installed": {r: pk.get(r) for r in SAM2_PKG_REQUIRES}}


def cmd_meta() -> None:
    from huggingface_hub import get_hf_file_metadata, get_token, hf_hub_url, model_info
    import urllib.request

    import torch
    import transformers
    from transformers import Sam2VideoConfig, Sam2VideoModel, Sam2VideoProcessor  # noqa: F401  (kelas ada di 5.17.0?)

    info = model_info(HF_REPO, files_metadata=True)
    commit = info.sha
    sib = {s.rfilename: {"size": s.size, "lfs_sha256": s.lfs.sha256 if s.lfs else None, "blob_id": s.blob_id} for s in info.siblings}
    heads = {}
    for f in HF_FILES_TRANSFORMERS + HF_FILES_SAM2_PKG:
        m = get_hf_file_metadata(hf_hub_url(HF_REPO, f, revision=commit))
        heads[f] = {"size": m.size, "etag": m.etag, "commit": m.commit_hash, "location_host": m.location.split("/")[2]}
    # config.json (5,7 KB) dibaca ke MEMORI saja untuk dibandingkan dengan konfigurasi bawaan transformers (tidak disimpan)
    with urllib.request.urlopen(hf_hub_url(HF_REPO, "config.json", revision=commit), timeout=30) as r:
        remote_cfg = json.loads(r.read().decode("utf-8"))
    cfg = Sam2VideoConfig(**{k: v for k, v in remote_cfg.items() if k not in ("architectures", "transformers_version", "torch_dtype")})
    with torch.device("meta"):
        model = Sam2VideoModel(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    n_buffers = sum(b.numel() for b in model.buffers())
    groups = Counter()
    for name, p in model.named_parameters():
        groups[name.split(".")[0]] += p.numel()
    vc = remote_cfg.get("vision_config", {})
    bc = vc.get("backbone_config", {})
    res = {
        "repo": HF_REPO, "commit": commit, "last_modified": str(info.last_modified), "gated": info.gated, "private": info.private,
        "card_license": (info.card_data or {}).get("license") if info.card_data else None, "tags": info.tags,
        "library_name": info.library_name, "pipeline_tag": info.pipeline_tag, "hf_token_present": bool(get_token()),
        "siblings": sib, "head_at_commit": heads,
        "download_transformers_bytes": sum(sib[f]["size"] for f in HF_FILES_TRANSFORMERS),
        "download_sam2_pkg_bytes": sum(sib[f]["size"] for f in HF_FILES_SAM2_PKG),
        "remote_config_summary": {k: remote_cfg.get(k) for k in ("architectures", "model_type", "torch_dtype", "transformers_version",
                                                                  "image_size", "num_maskmem", "memory_attention_num_layers",
                                                                  "memory_attention_hidden_size", "memory_encoder_output_channels",
                                                                  "max_object_pointers_in_encoder")},
        "remote_backbone": {k: bc.get(k) for k in ("hidden_size", "embed_dim_per_stage", "blocks_per_stage", "num_attention_heads_per_stage",
                                                   "global_attention_blocks", "window_size_per_stage")},
        "params_meta_device": {"total": int(n_params), "buffers": int(n_buffers), "per_top_module": dict(groups.most_common()),
                               "fp32_bytes": int(n_params) * 4, "fp16_bytes": int(n_params) * 2,
                               "safetensors_bytes": sib["model.safetensors"]["size"]},
        "versions": {"torch": torch.__version__, "torch_cuda_build": torch.version.cuda, "transformers": transformers.__version__,
                     "python": sys.version.split()[0]},
        "transformers_classes": ["Sam2VideoModel", "Sam2VideoProcessor", "Sam2VideoInferenceSession", "Sam2VideoConfig"],
        "env": gui_env(), "mem": mem_status(),
        "disk_c_free_gib": round(shutil.disk_usage("C:/").free / (1 << 30), 2),
        "hf_cache": {"path": str(HF_CACHE), "total_gib": round(dir_size(HF_CACHE) / (1 << 30), 3),
                     "repos": {p.name: round(dir_size(p) / (1 << 30), 3) for p in sorted(HF_CACHE.glob("models--*"))}},
        "sam2_in_cache": any("sam2" in p.name.lower() for p in HF_CACHE.glob("models--*")),
    }
    jsave(OUT / "meta.json", res)
    print(json.dumps({k: res[k] for k in ("commit", "last_modified", "gated", "card_license", "download_transformers_bytes",
                                          "download_sam2_pkg_bytes", "params_meta_device", "mem", "disk_c_free_gib")}, ensure_ascii=False))
    print(json.dumps(res["env"], ensure_ascii=False))


# ── potong letterbox / kotak teks ──────────────────
def frame_names(clip: str) -> list[str]:
    return sorted(p.name for p in (CLIPS_DIR / clip / "frames").glob("frame_*.png"))


def temporal_stats(clip: str, names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Rata-rata temporal (H, W, 3) dan simpangan baku temporal maks antar kanal (H, W), dari frames/ (BGR 0–255)."""
    s = s2 = None
    for n in names:
        a = cv2.imdecode(np.fromfile(CLIPS_DIR / clip / "frames" / n, np.uint8), cv2.IMREAD_COLOR).astype(np.float64)
        if s is None:
            s, s2 = np.zeros_like(a), np.zeros_like(a)
        s += a
        s2 += a * a
    k = len(names)
    m = s / k
    sd = np.sqrt(np.maximum(s2 / k - m * m, 0.0)).max(axis=2)
    return m.astype(np.float32), sd.astype(np.float32)


def line_ok(m: np.ndarray, sd: np.ndarray, rect: list[int], edge: str, need_uniform: bool) -> bool:
    y0, y1, x0, x1 = rect
    if y1 - y0 < 1 or x1 - x0 < 1:
        return False
    if edge in ("top", "bottom"):
        y = y0 if edge == "top" else y1 - 1
        sdl, ml = sd[y, x0:x1], m[y, x0:x1]
    else:
        x = x0 if edge == "left" else x1 - 1
        sdl, ml = sd[y0:y1, x], m[y0:y1, x]
    if (sdl <= STATIC_STD_MAX).mean() < STATIC_LINE_FRAC:
        return False
    return (not need_uniform) or float(ml.std(axis=0).max()) <= UNIFORM_STD_MAX


def shrink(rect: list[int], edge: str) -> None:
    i = {"top": 0, "bottom": 1, "left": 2, "right": 3}[edge]
    rect[i] += 1 if edge in ("top", "left") else -1


def peel_bands(m: np.ndarray, sd: np.ndarray, edges) -> tuple[list[int], dict]:
    H, W = sd.shape
    rect = [0, H, 0, W]
    band = dict.fromkeys(EDGES, 0)
    changed = True
    while changed:
        changed = False
        for e in EDGES:
            if e not in edges:
                continue
            while line_ok(m, sd, rect, e, need_uniform=True):
                shrink(rect, e)
                band[e] += 1
                changed = True
    return rect, band


def detect_crop(m: np.ndarray, sd: np.ndarray, rule: str = "R1") -> dict:
    """R1 (aturan awal): 1) kupas pita (statis DAN seragam) dari keempat tepi, berulang sampai stabil; 2) di tepi dengan pita
    >= BAND_MIN_PX, lanjut kupas baris / kolom statis yang tidak seragam (kotak teks / overlay yang menempel di pita) sampai baris
    tidak statis.
    R2 (POST-HOC, disusun setelah positif palsu R1 di jendela pendek terlihat): pita hanya sah bila BERPASANGAN (atas DAN bawah, atau
    kiri DAN kanan, masing-masing >= BAND_MIN_PX; letterbox / pillarbox), dan kotak teks hanya dikupas di tepi atas / bawah."""
    H, W = sd.shape
    allowed = set(EDGES)
    while True:
        rect, band = peel_bands(m, sd, allowed)
        if rule == "R1":
            break
        keep = {e for e in allowed if band[e] >= BAND_MIN_PX and band[OPP[e]] >= BAND_MIN_PX}
        if keep == allowed:
            break
        allowed = keep
    overlay = dict.fromkeys(EDGES, 0)
    for e in EDGES:
        if rule == "R2" and e not in ("top", "bottom"):
            continue
        if band[e] >= BAND_MIN_PX:
            while line_ok(m, sd, rect, e, need_uniform=False):
                shrink(rect, e)
                overlay[e] += 1
    y0, y1, x0, x1 = rect
    area = max(0, y1 - y0) * max(0, x1 - x0)
    applied = area >= MIN_ACTIVE_FRAC * H * W
    out = {"rule": rule, "frame_wh": [W, H], "band_px": band, "overlay_px": overlay, "applied": bool(applied),
           "active_xyxy_incl": [x0, y0, x1 - 1, y1 - 1] if applied else [0, 0, W - 1, H - 1],
           "active_wh": [x1 - x0, y1 - y0] if applied else [W, H],
           "active_frac": round(area / (H * W), 4) if applied else 1.0}
    if not applied:
        out["note"] = f"area aktif {area / (H * W):.3f} < {MIN_ACTIVE_FRAC} -> tidak dipotong"
    return out


def zone_colors(m: np.ndarray, sd: np.ndarray, det: dict) -> dict:
    """Warna rata-rata (BGR) + simpangan baku temporal median per zona terpotong (pita / overlay) per tepi."""
    H, W = sd.shape
    b, o = det["band_px"], det["overlay_px"]
    spans = {"top": [(0, b["top"]), (b["top"], b["top"] + o["top"])],
             "bottom": [(H - b["bottom"], H), (H - b["bottom"] - o["bottom"], H - b["bottom"])],
             "left": [(0, b["left"]), (b["left"], b["left"] + o["left"])],
             "right": [(W - b["right"], W), (W - b["right"] - o["right"], W - b["right"])]}
    res = {}
    for e, (bs, os_) in spans.items():
        for kind, (a0, a1) in (("pita", bs), ("overlay", os_)):
            if a1 <= a0:
                continue
            sl = (slice(a0, a1), slice(None)) if e in ("top", "bottom") else (slice(None), slice(a0, a1))
            res[f"{e}_{kind}"] = {"range": [a0, a1 - 1], "mean_bgr": [round(float(v), 1) for v in m[sl].reshape(-1, 3).mean(0)],
                                  "temporal_std_median": round(float(np.median(sd[sl])), 3),
                                  "temporal_std_p99": round(float(np.percentile(sd[sl], 99)), 3)}
    return res


def inner_static(sd: np.ndarray, det: dict) -> dict:
    """Informasi saja (TIDAK dipotong): piksel statis di dalam area aktif + komponen statis terbesar (calon overlay / latar diam)."""
    x0, y0, x1, y1 = det["active_xyxy_incl"]
    st = (sd[y0:y1 + 1, x0:x1 + 1] <= STATIC_STD_MAX).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(st, connectivity=8)
    big = max(range(1, n), key=lambda i: stats[i, cv2.CC_STAT_AREA]) if n > 1 else None
    tot = st.size
    return {"static_frac": round(float(st.mean()), 4),
            "largest_static_component": None if big is None else
            {"area_frac": round(float(stats[big, cv2.CC_STAT_AREA]) / tot, 4),
             "bbox_xywh_frame": [int(stats[big, 0]) + x0, int(stats[big, 1]) + y0, int(stats[big, 2]), int(stats[big, 3])]}}


def row_profile(sd: np.ndarray, m: np.ndarray, det: dict) -> dict:
    """Ringkasan profil baris di sekitar batas atas / bawah area aktif (untuk membandingkan dengan acuan klip5)."""
    x0, y0, x1, y1 = det["active_xyxy_incl"]
    out = {}
    for name, y in (("atas", y0), ("bawah", y1)):
        rows = range(max(0, y - 6), min(sd.shape[0], y + 7))
        out[name] = [{"y": r, "static_frac": round(float((sd[r, x0:x1 + 1] <= STATIC_STD_MAX).mean()), 3),
                      "uniform_std": round(float(m[r, x0:x1 + 1].std(axis=0).max()), 2)} for r in rows]
    return out


def cmd_letterbox() -> None:
    t0 = time.perf_counter()
    res = {"rule": {"STATIC_STD_MAX": STATIC_STD_MAX, "STATIC_LINE_FRAC": STATIC_LINE_FRAC, "UNIFORM_STD_MAX": UNIFORM_STD_MAX,
                    "BAND_MIN_PX": BAND_MIN_PX, "MIN_ACTIVE_FRAC": MIN_ACTIVE_FRAC, "WIN_LEN": WIN_LEN, "WIN_STEP": WIN_STEP,
                    "WIN_TOL_PX": WIN_TOL_PX}, "expected": EXPECTED, "clips": {}}
    for clip in LB_CLIPS:
        names = frame_names(clip)
        n = len(names)
        ranges = [(0, n - 1)] + STUDY_RANGES.get(clip, [])
        per_range = {r: {} for r in RULES}
        for a, b in ranges:
            m, sd = temporal_stats(clip, names[a:b + 1])
            for rule in RULES:
                det = detect_crop(m, sd, rule)
                det.update({"zones": zone_colors(m, sd, det), "inside_active": inner_static(sd, det), "edge_rows": row_profile(sd, m, det)})
                per_range[rule][f"f{a}-{b}"] = det
                print(f"{clip} {rule} f{a}-{b}: aktif {det['active_xyxy_incl']} ({det['active_wh']}, {det['active_frac']}) pita {det['band_px']} "
                      f"overlay {det['overlay_px']} statis di dalam {det['inside_active']['static_frac']}", flush=True)
        # acuan jendela = hasil R1 rentang penuh (R1 dan R2 sama di rentang penuh kelima klip — dicek di bawah)
        ref = per_range["R1"][f"f0-{n - 1}"]["active_xyxy_incl"]
        wins = {r: [] for r in RULES}
        for a in range(0, max(1, n - WIN_LEN + 1), WIN_STEP):
            b = min(n - 1, a + WIN_LEN - 1)
            m, sd = temporal_stats(clip, names[a:b + 1])
            for rule in RULES:
                det = detect_crop(m, sd, rule)
                x0, y0, x1, y1 = det["active_xyxy_incl"]
                excess = {"left": x0 - ref[0], "top": y0 - ref[1], "right": ref[2] - x1, "bottom": ref[3] - y1}
                wins[rule].append({"range": [a, b], "active_xyxy_incl": det["active_xyxy_incl"], "band_px": det["band_px"],
                                   "overlay_px": det["overlay_px"], "excess_vs_full_px": excess,
                                   "fp": any(v > WIN_TOL_PX for v in excess.values()), "fn": any(v < -WIN_TOL_PX for v in excess.values())})
        summ = {}
        for rule in RULES:
            ws = wins[rule]
            fp = [w for w in ws if w["fp"]]
            fn = [w for w in ws if w["fn"]]
            summ[rule] = {"n": len(ws), "fp": len(fp), "fn": len(fn), "fp_ranges": [w["range"] for w in fp], "fn_ranges": [w["range"] for w in fn],
                          "max_excess_px": {e: max(w["excess_vs_full_px"][e] for w in ws) for e in EDGES},
                          "min_excess_px": {e: min(w["excess_vs_full_px"][e] for w in ws) for e in EDGES}}
            print(f"  {rule} jendela {WIN_LEN} f / langkah {WIN_STEP}: {len(ws)}; potong berlebih (FP) {len(fp)} {[w['range'] for w in fp][:6]}; "
                  f"potong kurang {len(fn)} {[w['range'] for w in fn][:6]}; maks lebih {summ[rule]['max_excess_px']}", flush=True)
        same_full = all(per_range["R1"][k]["active_xyxy_incl"] == per_range["R2"][k]["active_xyxy_incl"] for k in per_range["R1"])
        res["clips"][clip] = {"n_frames": n, "ranges": per_range, "r1_equals_r2_on_ranges": same_full, "windows": wins, "windows_summary": summ}
    res["elapsed_s"] = round(time.perf_counter() - t0, 1)
    jsave(OUT / "letterbox.json", res)


# ── jendela klip3 (orang bersentuhan) ──────────────
def cmd_window() -> None:
    cfg = load_pipeline()
    cd = t701.ClipData(WINDOW_CLIP, cfg)
    rows = []
    for f in range(cd.n):
        blobs, faces = t701.frame_people(cd, f)
        f1 = t701.faces_v1_of(faces, blobs)
        per_blob = [sum(x["blob"] == k for x in f1) for k in range(len(blobs))]
        rows.append({"frame": f, "n_blobs": len(blobs), "n_faces_v1": len(f1), "faces_v1_per_blob": per_blob,
                     "merged": any(c >= 2 for c in per_blob)})
    merged = [r["frame"] for r in rows if r["merged"]]
    cands = []
    for a, b in cd.segments():
        for s in range(a, b - WINDOW_LEN + 2):
            w = rows[s:s + WINDOW_LEN]
            cands.append({"range": [s, s + WINDOW_LEN - 1], "merged": sum(r["merged"] for r in w),
                          "n_blobs_hist": dict(Counter(r["n_blobs"] for r in w)), "faces_v1_hist": dict(Counter(r["n_faces_v1"] for r in w))})
    best = max(c["merged"] for c in cands) if cands else 0
    top = [c for c in cands if c["merged"] == best]
    f0 = next((c for c in cands if c["range"][0] == 0), None)
    res = {"clip": WINDOW_CLIP, "n": cd.n, "cuts": cd.cuts, "segments": cd.segments(), "window_len": WINDOW_LEN,
           "merged_frames": runs_str(merged), "n_merged": len(merged), "best_merged": best,
           "best_windows": [c["range"] for c in top], "window_f0": f0, "rows": rows}
    jsave(OUT / "window.json", res)
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, ensure_ascii=False))


# ══ Tahap 1b-1 ═════════════════════════════════════
def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def rel(p: Path) -> str:
    try:
        return Path(p).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(p)


def dir_size_real(p: Path) -> int:
    """Ukuran nyata di disk: symlink tidak diikuti (cache HF bisa memakai symlink snapshots -> blobs)."""
    return sum(f.lstat().st_size for f in p.rglob("*") if not f.is_symlink() and f.is_file()) if p.exists() else 0


def git_blob_sha1(p: Path) -> str:
    data = p.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def frame_file(clip: str, idx: int) -> Path:
    return CLIPS_DIR / clip / "frames" / (FRAME_PATTERN % idx)


def read_frame(clip: str, idx: int) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(frame_file(clip, idx), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"frame tidak terbaca: {rel(frame_file(clip, idx))}")
    return img


def write_png(path: Path, img: np.ndarray) -> None:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise SystemExit(f"gagal encode PNG {rel(path)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    write_bytes_atomic(path, buf.tobytes())


def save_masks(path: Path, obj_ids: list[int], masks: np.ndarray) -> None:
    """Masker per target (n, H, W) bool, geometri frames/ penuh, dikemas packbits."""
    buf = io.BytesIO()
    np.savez_compressed(buf, obj_ids=np.asarray(obj_ids, np.int32), shape=np.asarray(masks.shape[1:], np.int32),
                        packed=np.packbits(masks, axis=-1))
    path.parent.mkdir(parents=True, exist_ok=True)
    write_bytes_atomic(path, buf.getvalue())


def load_masks(path: Path) -> tuple[list[int], np.ndarray]:
    with np.load(path) as z:
        h, w = (int(v) for v in z["shape"])
        return [int(v) for v in z["obj_ids"]], np.unpackbits(z["packed"], axis=-1, count=w).astype(bool).reshape(-1, h, w)


def proc_peak_ram_gib() -> float:
    class PMC(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong), ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
    k32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
    k32.GetCurrentProcess.restype = ctypes.c_void_p
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(PMC), ctypes.c_ulong]
    pmc = PMC()
    pmc.cb = ctypes.sizeof(PMC)
    psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
    return round(pmc.PeakWorkingSetSize / (1 << 30), 3)


# ── unduh (keputusan 1) ────────────────────────────
def cached_model_file() -> Path | None:
    from huggingface_hub import try_to_load_from_cache
    p = try_to_load_from_cache(HF_REPO, "model.safetensors", revision=HF_PIN)
    return Path(p) if isinstance(p, str) else None


def cmd_download() -> None:
    from huggingface_hub import model_info, snapshot_download
    t0 = time.perf_counter()
    snap = Path(snapshot_download(HF_REPO, revision=HF_PIN, allow_patterns=list(HF_FILES_TRANSFORMERS)))
    dt = time.perf_counter() - t0
    sib = {s.rfilename: s for s in model_info(HF_REPO, revision=HF_PIN, files_metadata=True).siblings}
    files, ok = {}, snap.name == HF_PIN
    for f in HF_FILES_TRANSFORMERS:
        p = snap / f
        r = {"size": p.stat().st_size, "expected_size": sib[f].size, "symlink": p.is_symlink(), "resolved": str(p.resolve())}
        if sib[f].lfs:
            r.update(sha256=t701.file_sha(p), expected_sha256=sib[f].lfs.sha256)
            r["ok"] = r["sha256"] == r["expected_sha256"] and r["size"] == r["expected_size"]
            if f == "model.safetensors":
                r["ok"] = r["ok"] and r["sha256"] == HF_PIN_SHA256
        else:
            r.update(git_blob_sha1=git_blob_sha1(p), expected_blob_id=sib[f].blob_id)
            r["ok"] = r["git_blob_sha1"] == r["expected_blob_id"] and r["size"] == r["expected_size"]
        ok = ok and r["ok"]
        files[f] = r
    extra = sorted(p.relative_to(snap).as_posix() for p in snap.rglob("*") if p.is_file() and p.name not in HF_FILES_TRANSFORMERS)
    readme = (snap / "README.md").read_text(encoding="utf-8")
    front = readme.split("---")[1] if readme.startswith("---") else ""
    lic = "license: apache-2.0" in front
    arch = json.loads((snap / "config.json").read_text(encoding="utf-8")).get("architectures")
    repo_dir = HF_CACHE / ("models--" + HF_REPO.replace("/", "--"))
    res = {"repo": HF_REPO, "commit": HF_PIN, "snapshot": str(snap), "elapsed_s": round(dt, 1), "files": files, "extra_files": extra,
           "readme_license_apache2": lic, "config_architectures": arch, "repo_cache_bytes_real": dir_size_real(repo_dir),
           "files_total_bytes": sum(r["size"] for r in files.values()),
           "hf_cache_total_gib": round(dir_size_real(HF_CACHE) / (1 << 30), 3), "disk_c_free_gib": round(shutil.disk_usage("C:/").free / (1 << 30), 2)}
    res["all_ok"] = bool(ok and lic and not extra and arch == ["Sam2VideoModel"])
    jsave(OUT / "download.json", res)
    print(json.dumps({k: v for k, v in res.items() if k != "files"}, ensure_ascii=False))
    for f, r in files.items():
        print(f"  {f}: {r['size']} B, {'sha256 ' + r['sha256'][:16] if 'sha256' in r else 'blob ' + r['git_blob_sha1'][:16]}…, ok={r['ok']}, symlink={r['symlink']}")
    if not res["all_ok"]:
        raise SystemExit("VERIFIKASI UNDUHAN GAGAL — lihat work/t602/download.json")


# ── crop (keputusan 3: R2, rentang penuh, POST-HOC) ─
def cmd_crop() -> None:
    res = {"rule": "R2", "post_hoc": True,
           "catatan": "aturan R2 disusun post-hoc (T-602 1a); wajib diuji di klip baru sebelum produksi (keputusan 1b no. 3)",
           "params": {"STATIC_STD_MAX": STATIC_STD_MAX, "STATIC_LINE_FRAC": STATIC_LINE_FRAC, "UNIFORM_STD_MAX": UNIFORM_STD_MAX,
                      "BAND_MIN_PX": BAND_MIN_PX, "MIN_ACTIVE_FRAC": MIN_ACTIVE_FRAC}, "clips": {}}
    for clip in LB_CLIPS:
        names = frame_names(clip)
        m, sd = temporal_stats(clip, names)
        det = detect_crop(m, sd, "R2")
        res["clips"][clip] = {"n_frames": len(names), "range": [0, len(names) - 1], "crop_xyxy_incl": det["active_xyxy_incl"],
                              "crop_wh": det["active_wh"], "active_frac": det["active_frac"], "band_px": det["band_px"],
                              "overlay_px": det["overlay_px"], "applied": det["applied"]}
        print(f"{clip}: {det['active_xyxy_incl']} ({det['active_wh'][0]} x {det['active_wh'][1]}, {det['active_frac']}) "
              f"pita {det['band_px']} overlay {det['overlay_px']}", flush=True)
    k5, k3 = res["clips"]["klip5"]["crop_xyxy_incl"], res["clips"]["klip3"]
    res["checks"] = {"klip5_y_344_1011": k5[1] == 344 and k5[3] == 1011 and k5[0] == 0 and k5[2] == 719,
                     "klip3_tanpa_potongan": k3["crop_wh"] == [720, 1280] and not any(k3["band_px"].values())}
    jsave(CROP_JSON, res)
    print(json.dumps(res["checks"], ensure_ascii=False))


def load_crop(clip: str) -> tuple[int, int, int, int]:
    if not CROP_JSON.is_file():
        raise SystemExit(f"{rel(CROP_JSON)} belum ada — jalankan dulu: python scripts/t602_study.py crop")
    c = json.loads(CROP_JSON.read_text(encoding="utf-8"))["clips"].get(clip)
    if c is None:
        raise SystemExit(f"{clip} tidak ada di {rel(CROP_JSON)}")
    x0, y0, x1, y1 = (int(v) for v in c["crop_xyxy_incl"])
    return x0, y0, x1, y1


# ── pick (UI klik, D1 / D2 / keputusan 9) ──────────
FONT = cv2.FONT_HERSHEY_SIMPLEX


def screen_physical() -> tuple[int, int]:
    """Ukuran layar dalam piksel FISIK (proses dibuat DPI-aware supaya Windows tidak memperbesar jendela 125%)."""
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except (OSError, AttributeError):
        pass
    u = ctypes.windll.user32
    return int(u.GetSystemMetrics(0)), int(u.GetSystemMetrics(1))


def target_color(obj: int) -> tuple[int, int, int]:
    return TARGET_COLORS[(obj - 1) % len(TARGET_COLORS)]


def target_color_name(obj: int) -> str:
    return TARGET_COLOR_NAMES[(obj - 1) % len(TARGET_COLOR_NAMES)]


class Picker:
    """Tampilan = frame yang dicrop R2, diskalakan s; titik disimpan dalam koordinat frames/ penuh."""

    def __init__(self, clip: str, frame: int, crop: tuple[int, int, int, int], img: np.ndarray, screen: tuple[int, int]) -> None:
        self.clip, self.frame, self.crop, self.screen = clip, frame, crop, screen
        x0, y0, x1, y1 = crop
        sub = img[y0:y1 + 1, x0:x1 + 1]
        h, w = sub.shape[:2]
        sw, sh = screen
        self.s = min(1.0, PICK_MAX_H_FRAC * sh / h, (PICK_MAX_W_FRAC * sw - PICK_PANEL_W) / w)
        self.dw, self.dh = max(1, round(w * self.s)), max(1, round(h * self.s))
        self.disp = cv2.resize(sub, (self.dw, self.dh), interpolation=cv2.INTER_AREA if self.s < 1 else cv2.INTER_LINEAR)
        self.H, self.W = max(self.dh, PICK_PANEL_MIN_H), self.dw + PICK_PANEL_W
        self.points: list[dict] = []
        self.obj = 1
        self.msg = ""

    def to_full(self, dx: float, dy: float) -> tuple[float, float]:
        return round(dx / self.s + self.crop[0], 1), round(dy / self.s + self.crop[1], 1)

    def to_disp(self, x: float, y: float) -> tuple[int, int]:
        return int(round((x - self.crop[0]) * self.s)), int(round((y - self.crop[1]) * self.s))

    def click(self, dx: int, dy: int, label: int) -> bool:
        if not (0 <= dx < self.dw and 0 <= dy < self.dh):
            return False
        fx, fy = self.to_full(dx, dy)
        self.points.append({"obj_id": self.obj, "label": label, "x": fx, "y": fy})
        self.msg = ""
        return True

    def on_mouse(self, event, x, y, flags, param) -> None:  # noqa: ARG002 (tanda tangan callback OpenCV)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.click(x, y, 1)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.click(x, y, 0)

    def counts(self) -> dict[int, list[int]]:
        c: dict[int, list[int]] = {}
        for p in self.points:
            c.setdefault(p["obj_id"], [0, 0])[0 if p["label"] == 1 else 1] += 1
        return c

    def validate(self) -> str | None:
        c = self.counts()
        if not any(v[0] for v in c.values()):
            return "Belum ada klik kiri (target)."
        bad = [k for k, v in c.items() if v[0] == 0]
        return f"Target {bad} belum punya klik kiri." if bad else None

    def key(self, k: int) -> str | None:
        if k in KEY_ENTER or k in (ord("s"), ord("S")):
            return "save"
        if k == KEY_ESC:
            return "cancel"
        if k in (ord("u"), ord("U")):
            self.msg = "Klik terakhir dibatalkan." if self.points else "Tidak ada klik."
            if self.points:
                self.points.pop()
        elif k in (ord("r"), ord("R")):
            self.points, self.obj, self.msg = [], 1, "Semua klik dihapus."
        elif k in (ord("n"), ord("N")):
            if self.counts().get(self.obj, [0, 0])[0] == 0:
                self.msg = f"Target {self.obj} belum punya klik kiri."
            else:
                self.obj += 1
                self.msg = f"Sekarang target {self.obj} ({target_color_name(self.obj)})."
        return None

    def panel_lines(self) -> list[tuple[str, tuple[int, int, int], float]]:
        x0, y0, x1, y1 = self.crop
        white, grey, yellow = (235, 235, 235), (170, 170, 170), (0, 230, 255)
        lines = [("T-602  PILIH TARGET", white, 0.65), (f"{self.clip}  frame {self.frame}", white, 0.5),
                 (f"crop {x1 - x0 + 1}x{y1 - y0 + 1}, skala {self.s:.2f}", grey, 0.45), ("", white, 0.5),
                 (f"Target aktif: {self.obj} ({target_color_name(self.obj)})", target_color(self.obj), 0.55), ("", white, 0.5),
                 ("Klik KIRI  = bagian target", white, 0.5), ("             (titik warna target)", grey, 0.45),
                 ("Klik KANAN = BUKAN target (merah)", white, 0.5), ("N = target berikutnya (warna beda)", white, 0.5),
                 ("U = batal klik terakhir", white, 0.5), ("R = ulang semua", white, 0.5), ("Enter / S = simpan", white, 0.5),
                 ("Esc = batal (tidak disimpan)", white, 0.5), ("", white, 0.5),
                 ("Saran per target: 3-5 klik kiri", grey, 0.45), ("tersebar (kepala, dada, pinggul,", grey, 0.45),
                 ("lengan / kaki). Klik kanan di orang", grey, 0.45), ("lain yang menempel ke target.", grey, 0.45),
                 ("", white, 0.5), ("Titik:", white, 0.5)]
        c = self.counts()
        for k in sorted(set(c) | {self.obj}):
            pos, neg = c.get(k, [0, 0])
            lines.append((f"  target {k}: {pos} kiri, {neg} kanan", target_color(k), 0.5))
        if self.msg:
            lines += [("", white, 0.5), (self.msg, yellow, 0.5)]
        return lines

    def render(self) -> np.ndarray:
        c = np.full((self.H, self.W, 3), 32, np.uint8)
        c[:self.dh, :self.dw] = self.disp
        for p in self.points:
            dx, dy = self.to_disp(p["x"], p["y"])
            col = target_color(p["obj_id"])
            if p["label"] == 1:
                cv2.circle(c, (dx, dy), 8, (0, 0, 0), -1, cv2.LINE_AA)
                cv2.circle(c, (dx, dy), 6, col, -1, cv2.LINE_AA)
            else:
                cv2.circle(c, (dx, dy), 9, col, 2, cv2.LINE_AA)
                cv2.circle(c, (dx, dy), 6, NEG_COLOR, -1, cv2.LINE_AA)
            cv2.putText(c, str(p["obj_id"]), (dx + 9, dy - 9), FONT, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(c, str(p["obj_id"]), (dx + 9, dy - 9), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        y = 30
        for text, col, sc in self.panel_lines():
            if text:
                cv2.putText(c, text, (self.dw + 14, y), FONT, sc, col, 2 if sc > 0.6 else 1, cv2.LINE_AA)
            y += int(42 * sc)
        return c

    def widest_line_px(self) -> int:
        return max(cv2.getTextSize(t, FONT, sc, 2 if sc > 0.6 else 1)[0][0] for t, _, sc in self.panel_lines() if t)


def points_doc(pk: Picker, clip: str, frame: int) -> dict:
    a, b = WINDOWS[clip]
    objs = sorted({p["obj_id"] for p in pk.points})
    img = read_frame(clip, frame)
    return {"format": POINTS_FORMAT, "clip": clip, "window": [a, b], "frame_wh": [int(img.shape[1]), int(img.shape[0])],
            "crop_rule": "R2", "crop_xyxy_incl": list(pk.crop),
            "coords": "piksel frames/ penuh (x ke kanan, y ke bawah), sebelum crop; label 1 = target, 0 = bukan target (konvensi SAM)",
            "objects": [{"obj_id": k, "name": f"target {k}"} for k in objs],
            "prompts": [{"frame_index": frame, "frame_file": FRAME_PATTERN % frame, "frame_sha256": t701.file_sha(frame_file(clip, frame)),
                         "obj_id": k, "points": [{"x": p["x"], "y": p["y"], "label": p["label"]} for p in pk.points if p["obj_id"] == k]}
                        for k in objs],
            "display_scale": round(pk.s, 4), "screen_physical_wh": list(pk.screen), "created_utc": utc_now(),
            "tool": "scripts/t602_study.py pick"}


def cmd_pick(clip: str, frame: int, dry_run: bool) -> None:
    if clip not in WINDOWS:
        raise SystemExit(f"klip {clip!r} bukan klip uji T-602 ({', '.join(WINDOWS)})")
    a, b = WINDOWS[clip]
    if not dry_run and frame != a:
        raise SystemExit(f"titik harus diklik di frame AWAL jendela {clip} (f{a}; jendela f{a}–f{b}): pakai --frame {a}")
    if not frame_file(clip, frame).is_file():
        raise SystemExit(f"frame {frame} tidak ada: {rel(frame_file(clip, frame))}")
    crop = load_crop(clip)
    pk = Picker(clip, frame, crop, read_frame(clip, frame), screen_physical())
    if dry_run:
        out = SCR / "pick_dryrun"
        write_png(out / f"{clip}_f{frame:05d}.png", pk.render())
        # contoh titik: menguji gambar titik + tombol (tidak disimpan sebagai titik)
        cx, cy = pk.dw // 2, pk.dh // 2
        pk.click(cx, cy - pk.dh // 4, 1), pk.click(cx, cy, 1), pk.click(cx + 40, cy + pk.dh // 5, 1), pk.click(cx + pk.dw // 3, cy, 0)
        pk.key(ord("n")), pk.click(cx - pk.dw // 3, cy, 1), pk.key(ord("u")), pk.click(cx - pk.dw // 3, cy + 10, 1)
        write_png(out / f"{clip}_f{frame:05d}_contoh_titik.png", pk.render())
        rt = []
        for dx, dy in ((0, 0), (pk.dw - 1, pk.dh - 1), (cx, cy), (17, pk.dh - 23)):
            fx, fy = pk.to_full(dx, dy)
            bx, by = pk.to_disp(fx, fy)
            rt.append({"disp": [dx, dy], "full": [fx, fy], "back": [bx, by],
                       "inside_crop": crop[0] <= fx <= crop[2] + 1 and crop[1] <= fy <= crop[3] + 1, "err_px": max(abs(bx - dx), abs(by - dy))})
        sw, sh = pk.screen
        res = {"clip": clip, "frame": frame, "crop_xyxy_incl": list(crop), "screen_physical_wh": [sw, sh], "scale": round(pk.s, 4),
               "image_disp_wh": [pk.dw, pk.dh], "canvas_wh": [pk.W, pk.H], "max_canvas_wh": [int(PICK_MAX_W_FRAC * sw), int(PICK_MAX_H_FRAC * sh)],
               "fits_screen": pk.W <= PICK_MAX_W_FRAC * sw and pk.H <= PICK_MAX_H_FRAC * sh,
               "widest_text_px": pk.widest_line_px(), "panel_text_room_px": PICK_PANEL_W - 14, "roundtrip": rt,
               "roundtrip_ok": all(r["inside_crop"] and r["err_px"] <= 1 for r in rt),
               "demo_counts": pk.counts(), "demo_validate": pk.validate(),
               "demo_doc_keys": sorted(points_doc(pk, clip, frame)), "png": [rel(out / f"{clip}_f{frame:05d}.png"),
                                                                             rel(out / f"{clip}_f{frame:05d}_contoh_titik.png")]}
        res["text_fits"] = res["widest_text_px"] <= res["panel_text_room_px"]
        jsave(out / f"{clip}_f{frame:05d}.json", res)
        print(json.dumps({k: v for k, v in res.items() if k not in ("roundtrip", "demo_doc_keys")}, ensure_ascii=False))
        return
    win = f"T-602 pilih target - {clip} f{frame}"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, pk.on_mouse)
    print(f"Jendela '{win}' terbuka ({pk.W} x {pk.H} px, skala {pk.s:.2f}). Petunjuk ada di panel kanan.")
    saved = None
    while True:
        cv2.imshow(win, pk.render())
        k = cv2.waitKeyEx(30)
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            act = "cancel"
        else:
            act = pk.key(k) if k != -1 else None
        if act == "save":
            err = pk.validate()
            if err:
                pk.msg = err
                continue
            POINTS_DIR.mkdir(parents=True, exist_ok=True)
            saved = POINTS_DIR / f"{clip}.json"
            if saved.exists():
                old = saved.with_name(f"{clip}.{datetime.now().strftime('%Y%m%dT%H%M%S')}.json")
                saved.rename(old)
                print(f"Berkas titik lama disimpan sebagai {rel(old)}")
            saved.write_text(json.dumps(points_doc(pk, clip, frame), indent=1, ensure_ascii=False), encoding="utf-8")
            break
        if act == "cancel":
            break
    cv2.destroyAllWindows()
    if saved is None:
        print("Dibatalkan — titik TIDAK disimpan.")
        return
    c = pk.counts()
    print(f"Tersimpan: {rel(saved)} — " + "; ".join(f"target {k}: {v[0]} kiri, {v[1]} kanan" for k, v in sorted(c.items())))
    print(f"Berikutnya (PowerShell terpisah): .\\venv\\Scripts\\python.exe scripts\\t602_study.py run {clip}")


# ── validasi titik (dipakai run dan _sam) ──────────
def load_points(path: Path, clip: str) -> tuple[dict | None, list[str]]:
    a, b = WINDOWS[clip]
    if not path.is_file():
        return None, [f"titik belum ada: {rel(path)} — jalankan dulu: python scripts/t602_study.py pick {clip} --frame {a}"]
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        return None, [f"{rel(path)} tidak terbaca ({e})"]
    errs = []
    if doc.get("format") != POINTS_FORMAT:
        errs.append(f"format {doc.get('format')!r} ≠ {POINTS_FORMAT!r}")
    if doc.get("clip") != clip:
        errs.append(f"berkas titik milik klip {doc.get('clip')!r}, bukan {clip!r}")
    if doc.get("window") != [a, b]:
        errs.append(f"jendela {doc.get('window')} ≠ [{a}, {b}]")
    crop = list(load_crop(clip))
    if doc.get("crop_xyxy_incl") != crop:
        errs.append(f"kotak crop titik {doc.get('crop_xyxy_incl')} ≠ crop.json {crop} — klik ulang")
    prompts = doc.get("prompts") or []
    if not prompts:
        errs.append("tidak ada prompt")
    objs = {o.get("obj_id") for o in doc.get("objects", [])}
    for pr in prompts:
        f = pr.get("frame_index")
        if f != a:
            errs.append(f"prompt di frame {f}, harus frame awal jendela f{a}")
        elif frame_file(clip, a).is_file() and pr.get("frame_sha256") != t701.file_sha(frame_file(clip, a)):
            errs.append(f"sha256 frame {a} berbeda dengan saat diklik (frames/ berubah?) — klik ulang")
        if pr.get("obj_id") not in objs:
            errs.append(f"obj_id {pr.get('obj_id')} tidak ada di objects")
        pts = pr.get("points") or []
        if not any(p.get("label") == 1 for p in pts):
            errs.append(f"target {pr.get('obj_id')} tanpa klik kiri")
        for p in pts:
            if p.get("label") not in (0, 1) or not (crop[0] <= p.get("x", -1) <= crop[2] + 1 and crop[1] <= p.get("y", -1) <= crop[3] + 1):
                errs.append(f"titik di luar kotak crop / label salah: {p}")
    return doc, errs


def prompt_objects(doc: dict, crop: tuple[int, int, int, int]) -> list[tuple[int, list[list[float]], list[int]]]:
    """[(obj_id, titik [x, y] dalam koordinat frame CROP, label)] per target."""
    out = []
    for pr in sorted(doc["prompts"], key=lambda r: r["obj_id"]):
        out.append((int(pr["obj_id"]), [[p["x"] - crop[0], p["y"] - crop[1]] for p in pr["points"]], [int(p["label"]) for p in pr["points"]]))
    return out


# ── _sam (subproses GPU; D4 / D5) ──────────────────
def sam_track(model, processor, clip: str, idxs: list[int], crop, objs, dtype, device: str, save_dir: Path | None,
              keep: int, log) -> dict:
    import torch
    x0, y0, x1, y1 = crop
    H, W = read_frame(clip, idxs[0]).shape[:2]
    session = processor.init_video_session(inference_device=device, inference_state_device="cpu", processing_device="cpu",
                                           video_storage_device="cpu", dtype=dtype)
    times, nan_frames, kept, rows = [], [], {}, []
    for i, f in enumerate(idxs):
        t0 = time.perf_counter()
        rgb = cv2.cvtColor(read_frame(clip, f)[y0:y1 + 1, x0:x1 + 1], cv2.COLOR_BGR2RGB)
        inputs = processor(images=rgb, return_tensors="pt")
        if i == 0:
            # SATU panggilan untuk semua target: `obj_with_new_inputs` DITIMPA tiap panggilan (transformers 5.17.0), jadi panggilan
            # terpisah per target membuat target selain yang terakhir tanpa frame kondisi. Jumlah titik berbeda antar target
            # dipadding processor dengan label -10 (diabaikan prompt encoder).
            processor.add_inputs_to_inference_session(inference_session=session, frame_idx=0, obj_ids=[o for o, _, _ in objs],
                                                      input_points=[[pts for _, pts, _ in objs]], input_labels=[[labs for _, _, labs in objs]],
                                                      original_size=inputs.original_sizes[0])
        with torch.inference_mode():
            out = model(inference_session=session, frame=inputs.pixel_values[0])
            logits = processor.post_process_masks([out.pred_masks], original_sizes=inputs.original_sizes, binarize=False)[0]
        if device == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        lg = logits.float().cpu().numpy().reshape(-1, y1 - y0 + 1, x1 - x0 + 1)
        ids = [int(v) for v in out.object_ids]
        finite = bool(np.isfinite(lg).all())
        full = np.zeros((len(ids), H, W), bool)
        full[:, y0:y1 + 1, x0:x1 + 1] = lg > 0
        scores = out.object_score_logits.float().cpu().numpy().reshape(-1).tolist() if out.object_score_logits is not None else []
        if save_dir is not None:
            save_masks(save_dir / "masks" / f"{Path(FRAME_PATTERN % f).stem}.npz", ids, full)
        if i < keep:
            kept[f] = (ids, full)
        times.append(dt)
        if not finite:
            nan_frames.append(f)
        rows.append({"frame": f, "time_s": round(dt, 4), "finite": finite, "obj_ids": ids,
                     "area_px": [int(m.sum()) for m in full], "score_logit": [round(float(s), 3) for s in scores]})
        if i == 0 or (i + 1) % 10 == 0 or i == len(idxs) - 1:
            log(f"  {dtype}: frame {f} ({i + 1}/{len(idxs)}) {dt:.2f} s; luas {[r for r in rows[-1]['area_px']]} px; skor {rows[-1]['score_logit']}")
    res = {"dtype": str(dtype), "n_frames": len(idxs), "time_total_s": round(sum(times), 2),
           "time_per_frame_s": {"mean": round(float(np.mean(times)), 3), "p50": round(float(np.median(times)), 3),
                                "max": round(float(np.max(times)), 3), "first": round(times[0], 3)},
           "nan_frames": nan_frames, "rows": rows, "kept": kept}
    if device == "cuda":
        res["vram_max_allocated_mib"] = round(torch.cuda.max_memory_allocated() / MIB, 1)
        res["vram_max_reserved_mib"] = round(torch.cuda.max_memory_reserved() / MIB, 1)
    return res


def mask_iou(a: np.ndarray, b: np.ndarray) -> float | None:
    u = int((a | b).sum())
    return None if u == 0 else round(int((a & b).sum()) / u, 5)


def cmd_sam(clip: str, points: Path, out: Path, device: str, max_frames: int | None, fp16: bool) -> None:
    import torch
    from transformers import Sam2VideoModel, Sam2VideoProcessor

    def log(msg: str) -> None:
        print(msg, flush=True)

    doc, errs = load_points(points, clip)
    if errs:
        raise SystemExit("titik tidak valid:\n  " + "\n  ".join(errs))
    crop = load_crop(clip)
    a, b = WINDOWS[clip]
    idxs = list(range(a, b + 1))[:max_frames] if max_frames else list(range(a, b + 1))
    mem = mem_status()
    info = {"clip": clip, "device": device, "window": [a, b], "frames": [idxs[0], idxs[-1]], "n_frames": len(idxs), "crop_xyxy_incl": list(crop),
            "points": rel(points), "points_sha256": t701.file_sha(points), "pin": HF_PIN, "ram_free_before_gib": mem["ram_free_gib"],
            "started_utc": utc_now(), "torch": torch.__version__}
    if mem["ram_free_gib"] < SAM_MIN_FREE_RAM_GIB:
        raise SystemExit(f"RAM bebas {mem['ram_free_gib']} GiB < {SAM_MIN_FREE_RAM_GIB} GiB — tutup aplikasi lain (browser, dll.) lalu ulangi")
    mf = cached_model_file()
    if mf is None or t701.file_sha(mf) != HF_PIN_SHA256:
        raise SystemExit("model SAM 2.1 tiny (pin) tidak ada di cache / sha256 salah — jalankan: python scripts/t602_study.py download")
    if device == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA tidak tersedia")
        free, total = torch.cuda.mem_get_info()
        info.update(vram_free_before_mib=round(free / MIB, 1), vram_total_mib=round(total / MIB, 1), gpu=torch.cuda.get_device_name(0))
        log(f"VRAM bebas {free / MIB:.0f} / {total / MIB:.0f} MiB (syarat ≥ {SAM_MIN_FREE_VRAM_MIB})")
        if free / MIB < SAM_MIN_FREE_VRAM_MIB:
            raise SystemExit(f"VRAM bebas {free / MIB:.0f} MiB < {SAM_MIN_FREE_VRAM_MIB} MiB — tutup aplikasi lain yang memakai GPU lalu ulangi")
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    model = Sam2VideoModel.from_pretrained(HF_REPO, revision=HF_PIN, local_files_only=True, dtype=torch.float32).to(device).eval()
    processor = Sam2VideoProcessor.from_pretrained(HF_REPO, revision=HF_PIN, local_files_only=True)
    info["load_s"] = round(time.perf_counter() - t0, 2)
    objs = prompt_objects(doc, crop)
    info["objects"] = [{"obj_id": o, "n_pos": labs.count(1), "n_neg": labs.count(0)} for o, _, labs in objs]
    log(f"model dimuat {info['load_s']} s; target {[o for o, _, _ in objs]}; {len(idxs)} frame f{idxs[0]}–f{idxs[-1]}, crop {crop}")
    out.mkdir(parents=True, exist_ok=True)
    r32 = sam_track(model, processor, clip, idxs, crop, objs, torch.float32, device, out / "fp32", FP16_COMPARE_FRAMES, log)
    kept32 = r32.pop("kept")
    rows = r32.pop("rows")
    (out / "fp32").mkdir(parents=True, exist_ok=True)
    (out / "fp32" / "frames.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    info["fp32"] = r32
    if fp16 and device == "cuda":
        n16 = min(FP16_COMPARE_FRAMES, len(idxs))
        try:
            model.to(torch.float16)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            r16 = sam_track(model, processor, clip, idxs[:n16], crop, objs, torch.float16, device, None, n16, log)
            kept16 = r16.pop("kept")
            r16.pop("rows")
            cmp = []
            for f in idxs[:n16]:
                ids32, m32 = kept32[f]
                ids16, m16 = kept16[f]
                per = {str(o): mask_iou(m32[ids32.index(o)], m16[ids16.index(o)]) for o in ids32 if o in ids16}
                cmp.append({"frame": f, "iou_per_target": per, "iou_union": mask_iou(m32.any(0), m16.any(0))})
            ious = [c["iou_union"] for c in cmp if c["iou_union"] is not None]
            r16.update(compare=cmp, iou_union_min=min(ious) if ious else None, iou_union_mean=round(float(np.mean(ious)), 5) if ious else None)
            info["fp16"] = r16
        except Exception as e:  # noqa: BLE001 — pembanding gagal (OOM / lain) dicatat, run fp32 tetap sah (D4)
            info["fp16"] = {"error": repr(e)}
            log(f"pembanding fp16 gagal: {e!r}")
    info["ram_peak_process_gib"] = proc_peak_ram_gib()
    info["finished_utc"] = utc_now()
    write_json_atomic(out / "summary.json", info)
    log(json.dumps({k: info[k] for k in ("n_frames", "load_s", "ram_peak_process_gib") if k in info}, ensure_ascii=False))
    log(json.dumps({"fp32": {k: v for k, v in info["fp32"].items()}, "fp16": {k: v for k, v in info.get("fp16", {}).items() if k != "compare"}},
                   ensure_ascii=False))


# ── run (D5 / D6 / D7 / D8) ────────────────────────
def short_fp(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def load_state(clip: str) -> dict:
    p = SCR / clip / "steps.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}


def save_state(clip: str, state: dict) -> None:
    (SCR / clip).mkdir(parents=True, exist_ok=True)
    write_json_atomic(SCR / clip / "steps.json", state)


def step_fp(step: str, clip: str, points_sha: str | None, crop, state: dict) -> str:
    base = {"step": step, "clip": clip, "window": list(WINDOWS[clip]), "crop": list(crop)}
    if step == "sam":
        base.update(points=points_sha, pin=HF_PIN, fp16_frames=FP16_COMPARE_FRAMES)
    if step in ("pratinjau", "M1"):
        base["sam_run"] = state.get("sam", {}).get("finished_utc")
    if step in ("C", "M1"):
        base["seg"] = SEG_MODEL
    if step == "M1":
        base.update(bg=list(M1_BG), dilate=MASK_DILATE_PX)
    return short_fp(base)


def prepare_variant(clip: str, var: str, fp: str, log, root: Path = SCR, n_limit: int | None = None) -> Path:
    """Klip scratch C / M1 (D2 / D3 / D6): frame crop (M1: latar abu-abu di luar masker + 2 px), depth asli dicrop, manifest depth scratch.
    `root` / `n_limit` hanya untuk uji (`_uji`): folder lain dan n frame pertama jendela."""
    d = root / clip / var
    prep = d / "prep.json"
    if prep.is_file() and json.loads(prep.read_text(encoding="utf-8")).get("fingerprint") == fp:
        log(f"  {var}: salinan scratch sudah siap (dilanjutkan)")
        return d
    if d.exists():
        shutil.rmtree(d)
    src = CLIPS_DIR / clip
    x0, y0, x1, y1 = load_crop(clip)
    a, b = WINDOWS[clip]
    if n_limit:
        b = a + n_limit - 1
    cw, ch = x1 - x0 + 1, y1 - y0 + 1
    meta = json.loads((src / "meta.json").read_text(encoding="utf-8"))
    meta.update(frame_count=b - a + 1, frame_index_start=a, working_width=cw, working_height=ch)
    meta["t602"] = {"variant": var, "source_clip": clip, "window": [a, b], "crop_xyxy_incl": [x0, y0, x1, y1], "fingerprint": fp,
                    "catatan": "salinan scratch studi T-602 (docs/04 'Keputusan T-602 Tahap 1b' D2 / D3 / D6)"}
    (d / "frames").mkdir(parents=True, exist_ok=True)
    (d / "depth").mkdir(parents=True, exist_ok=True)
    write_json_atomic(d / "meta.json", meta)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * MASK_DILATE_PX + 1, 2 * MASK_DILATE_PX + 1))
    bg_px = []
    for f in range(a, b + 1):
        stem = Path(FRAME_PATTERN % f).stem
        img = read_frame(clip, f)[y0:y1 + 1, x0:x1 + 1].copy()
        if var == "M1":
            _, m = load_masks(root / clip / "sam" / "fp32" / "masks" / f"{stem}.npz")
            u = cv2.dilate(m.any(0)[y0:y1 + 1, x0:x1 + 1].astype(np.uint8), ker) > 0
            img[~u] = M1_BG
            bg_px.append(int((~u).sum()))
        write_png(d / "frames" / (FRAME_PATTERN % f), img)
        dd = dep.read_depth(src / "depth" / f"{stem}.npy")
        if dd is None or dd.dtype != dep.DEPTH_DTYPE or dd.shape[:2] != read_frame(clip, f).shape[:2]:
            raise SystemExit(f"depth asli {stem} tidak terbaca / dtype / ukuran salah")
        dep.write_depth(d / "depth" / f"{stem}.npy", np.ascontiguousarray(dd[y0:y1 + 1, x0:x1 + 1]))
    dm = json.loads((src / "depth" / "manifest.json").read_text(encoding="utf-8"))
    dm[CLIP_KEY] = clip_identity(d)
    dm["frame_size"] = {"width": cw, "height": ch}
    dm["t602"] = {"catatan": "depth ASLI dicrop (bukan run [2c]); hanya scratch T-602 (D3)", "crop_xyxy_incl": [x0, y0, x1, y1],
                  "source_manifest_sha256": t701.file_sha(src / "depth" / "manifest.json")}
    write_json_atomic(d / "depth" / "manifest.json", dm)
    info = {"fingerprint": fp, "variant": var, "n_frames": b - a + 1, "frame_wh": [cw, ch], "created_utc": utc_now()}
    if bg_px:
        info["bg_frac_mean"] = round(float(np.mean(bg_px)) / (cw * ch), 4)
    write_json_atomic(prep, info)
    log(f"  {var}: salinan scratch dibuat ({b - a + 1} frame {cw} x {ch}{', latar abu-abu rata-rata ' + str(info['bg_frac_mean']) if bg_px else ''})")
    return d


def run_sub(cmd: list[str], log, logfh) -> tuple[int, float, dict]:
    """Subproses dengan keluaran ke layar + run.log dan sampel nvidia-smi tiap 2 s."""
    from t305_newclips import gpu_used_mib
    samples: list[float] = []
    stop = threading.Event()

    def sampler() -> None:
        while not stop.is_set():
            v = gpu_used_mib()
            if v is not None:
                samples.append(v[0])
            stop.wait(VRAM_SAMPLE_S)

    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    t0 = time.perf_counter()
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", env=env)
    try:
        for line in p.stdout:
            print(line, end="", flush=True)
            logfh.write(line)
            logfh.flush()
        rc = p.wait()
    except KeyboardInterrupt:
        p.terminate()
        p.wait()
        stop.set()
        raise
    finally:
        stop.set()
        th.join()
    return rc, time.perf_counter() - t0, {"nvidia_smi_used_first_mib": samples[0] if samples else None,
                                          "nvidia_smi_used_peak_mib": max(samples) if samples else None, "n_samples": len(samples)}


def preview(clip: str, log, root: Path = SCR, out_dir: Path = PREVIEW_DIR, n_limit: int | None = None) -> list[str]:
    x0, y0, x1, y1 = load_crop(clip)
    a, b = WINDOWS[clip]
    if n_limit:
        b = a + n_limit - 1
    paths = []
    for f in (a, (a + b) // 2, b):
        ids, m = load_masks(root / clip / "sam" / "fp32" / "masks" / f"{Path(FRAME_PATTERN % f).stem}.npz")
        img = read_frame(clip, f)[y0:y1 + 1, x0:x1 + 1].copy()
        for k, o in enumerate(ids):
            cs, _ = cv2.findContours(m[k, y0:y1 + 1, x0:x1 + 1].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(img, cs, -1, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.drawContours(img, cs, -1, target_color(o), 2, cv2.LINE_AA)
        label = f"{clip} f{f}  target {ids}  luas {[int(v.sum()) for v in m]} px"
        cv2.putText(img, label, (12, 30), FONT, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, label, (12, 30), FONT, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        p = out_dir / f"{clip}_f{f:05d}.png"
        write_png(p, img)
        paths.append(rel(p))
    log("  pratinjau: " + ", ".join(paths))
    return paths


def run_checks(clip: str, points: Path, steps: list[str]) -> tuple[list[str], list[str], dict | None]:
    """(galat, catatan, dokumen titik). Tanpa GPU / CUDA."""
    errs, notes = [], []
    a, b = WINDOWS[clip]
    if not CROP_JSON.is_file():
        return [f"{rel(CROP_JSON)} belum ada — jalankan: python scripts/t602_study.py crop"], notes, None
    doc, perr = (None, []) if "sam" not in steps else load_points(points, clip)
    errs += perr
    missing = [f for f in range(a, b + 1) if not frame_file(clip, f).is_file()]
    if missing:
        errs.append(f"{len(missing)} frame jendela hilang (mis. f{missing[0]})")
    if any(s in steps for s in ("C", "M1")):
        dmiss = [f for f in range(a, b + 1) if not (CLIPS_DIR / clip / "depth" / f"{Path(FRAME_PATTERN % f).stem}.npy").is_file()]
        if dmiss:
            errs.append(f"{len(dmiss)} depth asli hilang (mis. f{dmiss[0]})")
    if "sam" in steps:
        mf = cached_model_file()
        if mf is None:
            errs.append("model SAM 2.1 tiny belum diunduh — jalankan: python scripts/t602_study.py download")
        elif t701.file_sha(mf) != HF_PIN_SHA256:
            errs.append(f"sha256 model di cache ≠ pin ({mf})")
    free = shutil.disk_usage("C:/").free / (1 << 30)
    if free < MIN_DISK_FREE_GIB:
        errs.append(f"disk C: bebas {free:.1f} GiB < {MIN_DISK_FREE_GIB} GiB")
    if shutil.which("nvidia-smi") is None:
        notes.append("nvidia-smi tidak ditemukan di PATH — VRAM tidak disampel (run tetap jalan)")
    mem = mem_status()
    notes.append(f"RAM bebas sekarang {mem['ram_free_gib']} GiB (syarat langkah sam ≥ {SAM_MIN_FREE_RAM_GIB} GiB, dicek saat mulai)")
    notes.append(f"disk C: bebas {free:.1f} GiB")
    return errs, notes, doc


def cmd_run(clip: str, until: str | None, redo: list[str], dry_run: bool, points: Path | None) -> None:
    if clip not in WINDOWS:
        raise SystemExit(f"klip {clip!r} bukan klip uji T-602 ({', '.join(WINDOWS)})")
    all_steps = list(STEPS[clip])
    for s in [until, *redo]:
        if s is not None and s not in all_steps:
            raise SystemExit(f"langkah {s!r} tidak ada untuk {clip} (langkah: {', '.join(all_steps)})")
    steps = all_steps[:all_steps.index(until) + 1] if until else all_steps
    points = points or POINTS_DIR / f"{clip}.json"
    a, b = WINDOWS[clip]
    n = b - a + 1
    errs, notes, _ = run_checks(clip, points, all_steps)
    state = load_state(clip)
    crop = load_crop(clip) if CROP_JSON.is_file() else (0, 0, 0, 0)
    psha = t701.file_sha(points) if points.is_file() else None
    est = {"sam": f"≈ {n * 0.5 / 60 + 0.5:.1f}–{n * 1.5 / 60 + 1:.1f} menit (perkiraan)", "pratinjau": "< 10 s",
           "C": f"≈ {n * EST_SEG_S_PER_FRAME / 60:.0f} menit (seg 0.8B {EST_SEG_S_PER_FRAME} s / frame)",
           "M1": f"≈ {n * EST_SEG_S_PER_FRAME / 60:.0f} menit (seg 0.8B {EST_SEG_S_PER_FRAME} s / frame)"}
    print(f"T-602 run {clip}: jendela f{a}–f{b} ({n} frame), crop {list(crop)}, titik {rel(points)}")
    sim = dict(state)
    for s in steps:
        fp = step_fp(s, clip, psha, crop, sim)
        done = sim.get(s, {}).get("done") and sim[s].get("fingerprint") == fp and s not in redo
        print(f"  [{'selesai' if done else 'akan jalan'}] {s}: {est[s]}" + (" (--redo)" if s in redo else ""))
        if not done:
            sim[s] = {"done": True, "fingerprint": fp, "finished_utc": "(nanti)"}
    for t in notes:
        print(f"  catatan: {t}")
    if errs:
        print("GALAT (run tidak dimulai):\n  " + "\n  ".join(errs))
        raise SystemExit(1)
    if dry_run:
        print("--dry-run: validasi lolos; tidak ada yang dijalankan (tanpa GPU).")
        return
    (SCR / clip).mkdir(parents=True, exist_ok=True)
    with (SCR / clip / "run.log").open("a", encoding="utf-8") as logfh:
        def log(msg: str) -> None:
            print(msg, flush=True)
            logfh.write(msg + "\n")
            logfh.flush()

        log(f"\n##### run {clip} {utc_now()} langkah {steps} redo {redo} #####")
        for s in steps:
            fp = step_fp(s, clip, psha, crop, state)
            if state.get(s, {}).get("done") and state[s].get("fingerprint") == fp and s not in redo:
                log(f"[{s}] sudah selesai ({state[s].get('finished_utc')}) — dilewati")
                continue
            rec = {"done": False, "fingerprint": fp, "started_utc": utc_now()}
            log(f"[{s}] mulai {rec['started_utc']} — {est[s]}")
            if s == "sam":
                out = SCR / clip / "sam"
                if out.exists():
                    shutil.rmtree(out)
                rc, wall, gpu = run_sub([sys.executable, str(Path(__file__).resolve()), "_sam", clip, "--points", str(points),
                                         "--out", str(out)], log, logfh)
                rec.update(rc=rc, wall_s=round(wall, 1), **gpu)
                if rc == 0:
                    sm = json.loads((out / "summary.json").read_text(encoding="utf-8"))
                    rec["summary"] = {k: sm.get(k) for k in ("vram_free_before_mib", "load_s", "ram_peak_process_gib")}
                    rec["summary"]["fp32"] = {k: sm["fp32"].get(k) for k in ("time_total_s", "time_per_frame_s", "nan_frames",
                                                                              "vram_max_allocated_mib", "vram_max_reserved_mib")}
                    f16 = sm.get("fp16", {})
                    rec["summary"]["fp16"] = {k: f16.get(k) for k in ("time_per_frame_s", "nan_frames", "vram_max_reserved_mib",
                                                                       "iou_union_min", "iou_union_mean", "error") if k in f16}
            elif s == "pratinjau":
                rec.update(rc=0, paths=preview(clip, log))
                log("  PERIKSA pratinjau sekarang. Bila target salah: tekan Ctrl+C sebelum M1, klik ulang (pick), lalu run --redo sam.")
            else:
                d = prepare_variant(clip, s, fp, log)
                rc, wall, gpu = run_sub([sys.executable, "-m", "rotoscope.segment", "--work-dir", str(d), "--seg-model", SEG_MODEL], log, logfh)
                rec.update(rc=rc, wall_s=round(wall, 1), **gpu, work_dir=rel(d), qc_report=(d / "qc_report.json").is_file())
            rec["finished_utc"] = utc_now()
            rec["done"] = rec.get("rc") == 0
            state[s] = rec
            save_state(clip, state)
            log(f"[{s}] {'SELESAI' if rec['done'] else 'GAGAL'} rc {rec.get('rc')}, {rec.get('wall_s', 0)} s, "
                f"nvidia-smi puncak {rec.get('nvidia_smi_used_peak_mib')} MiB")
            if not rec["done"]:
                log(f"Berhenti. Perbaiki penyebabnya lalu jalankan lagi perintah yang sama (langkah selesai dilewati).")
                raise SystemExit(rec.get("rc") or 1)
        log(f"Semua langkah {steps} selesai untuk {clip}. Kabari Claude Code untuk Tahap 1b-2.")


def cmd_uji(clip: str, points: Path, n: int) -> None:
    """Uji alat TANPA GPU: _sam di CPU (n frame, fp32) + salinan C / M1 + pratinjau di work/t602_scratch/uji/ (bukan folder run),
    lalu cek terhadap pemeriksaan pipeline: daftar frame [2], ukuran frame, depth_valid, identitas klip di manifest depth."""
    from rotoscope import segment as seg_mod
    from rotoscope.stage_common import require_frames
    root = SCR / "uji"
    if (root / clip).exists():
        shutil.rmtree(root / clip)
    t0 = time.perf_counter()
    cmd_sam(clip, points, root / clip / "sam", "cpu", n, False)
    t_sam = time.perf_counter() - t0
    res = {"clip": clip, "n_frames": n, "sam_cpu_s": round(t_sam, 1), "variants": {}}
    sm = json.loads((root / clip / "sam" / "summary.json").read_text(encoding="utf-8"))
    res["sam_summary"] = {k: sm.get(k) for k in ("load_s", "objects", "ram_peak_process_gib")}
    res["sam_summary"]["fp32"] = {k: sm["fp32"].get(k) for k in ("time_per_frame_s", "nan_frames")}
    rows = [json.loads(ln) for ln in (root / clip / "sam" / "fp32" / "frames.jsonl").read_text(encoding="utf-8").splitlines()]
    res["sam_rows"] = [{k: r[k] for k in ("frame", "obj_ids", "area_px", "score_logit", "finite")} for r in rows]
    for var in [v for v in ("C", "M1") if v in STEPS[clip]]:
        d = prepare_variant(clip, var, f"uji-{var}", print, root=root, n_limit=n)
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        sc = seg_mod.load_clip(d)
        require_frames(d, sc.names)
        h, w = int(meta["working_height"]), int(meta["working_width"])
        shapes = {cv2.imdecode(np.fromfile(d / "frames" / nm, np.uint8), cv2.IMREAD_COLOR).shape[:2] for nm in sc.names}
        dm = json.loads((d / "depth" / "manifest.json").read_text(encoding="utf-8"))
        chk = {"names": [sc.names[0], sc.names[-1]], "n": len(sc.names), "frame_shapes": sorted(list(s) for s in shapes),
               "frame_shape_ok": shapes == {(h, w)},
               "depth_valid": all(dep.depth_valid(d / "depth" / f"{Path(nm).stem}.npy", h, w) for nm in sc.names),
               "depth_manifest_identity_ok": dm[CLIP_KEY] == clip_identity(d), "depth_manifest_size_ok": dm["frame_size"] == {"width": w, "height": h},
               "prep": json.loads((d / "prep.json").read_text(encoding="utf-8"))}
        if var == "M1":
            img = cv2.imdecode(np.fromfile(d / "frames" / sc.names[0], np.uint8), cv2.IMREAD_COLOR)
            chk["gray_frac_frame0"] = round(float((img == np.array(M1_BG, np.uint8)).all(axis=2).mean()), 4)
        res["variants"][var] = chk
    res["preview"] = preview(clip, print, root=root, out_dir=root / clip / "pratinjau", n_limit=n)
    res["ok"] = all(v["frame_shape_ok"] and v["depth_valid"] and v["depth_manifest_identity_ok"] and v["depth_manifest_size_ok"]
                    for v in res["variants"].values()) and not sm["fp32"]["nan_frames"]
    jsave(root / f"uji_{clip}.json", res)
    print(json.dumps({k: v for k, v in res.items() if k not in ("sam_rows",)}, ensure_ascii=False, default=str))


def main(argv: list[str]) -> None:
    import argparse
    for stream in (sys.stdout, sys.stderr):      # keluaran di-pipe (run.log / Tee) di Windows = cp1252 -> karakter ≈ – ≥ gagal
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    if argv[:1] == ["hash"] and len(argv) == 2 and argv[1] in HASH_LABELS:
        cmd_hash(argv[1])
    elif argv == ["meta"]:
        cmd_meta()
    elif argv == ["letterbox"]:
        cmd_letterbox()
    elif argv == ["window"]:
        cmd_window()
    elif argv == ["download"]:
        cmd_download()
    elif argv == ["crop"]:
        cmd_crop()
    elif argv[:1] == ["pick"]:
        p = argparse.ArgumentParser(prog="t602_study.py pick")
        p.add_argument("clip")
        p.add_argument("--frame", type=int, required=True)
        p.add_argument("--dry-run", action="store_true")
        a = p.parse_args(argv[1:])
        cmd_pick(a.clip, a.frame, a.dry_run)
    elif argv[:1] == ["run"]:
        p = argparse.ArgumentParser(prog="t602_study.py run")
        p.add_argument("clip")
        p.add_argument("--until", default=None)
        p.add_argument("--redo", nargs="+", default=[])
        p.add_argument("--dry-run", action="store_true")
        p.add_argument("--points", type=Path, default=None)
        a = p.parse_args(argv[1:])
        cmd_run(a.clip, a.until, a.redo, a.dry_run, a.points)
    elif argv[:1] == ["_sam"]:
        p = argparse.ArgumentParser(prog="t602_study.py _sam")
        p.add_argument("clip", choices=sorted(WINDOWS))
        p.add_argument("--points", type=Path, required=True)
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
        p.add_argument("--max-frames", type=int, default=None)
        p.add_argument("--no-fp16", action="store_true")
        a = p.parse_args(argv[1:])
        cmd_sam(a.clip, a.points, a.out, a.device, a.max_frames, not a.no_fp16)
    elif argv[:1] == ["_uji"]:
        p = argparse.ArgumentParser(prog="t602_study.py _uji")
        p.add_argument("clip", choices=sorted(WINDOWS))
        p.add_argument("--points", type=Path, required=True)
        p.add_argument("--frames", type=int, default=3)
        a = p.parse_args(argv[1:])
        cmd_uji(a.clip, a.points, a.frames)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
