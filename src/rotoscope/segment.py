"""Stage [2] segment: frames/*.png → seg/classmap + seg/probs + seg/manifest.json + seg/frames.jsonl
+ qc_report.json (Sapiens2-seg, D-009, D-010). Kontrak lengkap: docs/01 [2].

Dua lapis:
- Backend model (TorchSegBackend) — satu-satunya bagian yang meng-import torch/transformers (import
  malas di dalam method). describe() tanpa GPU (cek cache + config), open() = CUDA + cek VRAM + load,
  infer() = logits yang sudah diinterpolasi ke resolusi kerja, di CPU.
- Logika stage (run_segment, run_qc) — numpy/cv2 saja: resume, manifest, tulis atomik, QC. Unit test
  memakai backend palsu lewat `backend_factory`, tanpa GPU/model.

Per frame: logits (GPU, fp16) → interpolasi bilinear ke resolusi kerja di GPU (sama dengan
post_process_semantic_segmentation) → CPU → softmax float32 di CPU (sisa VRAM run 0.8B ±42 MiB) →
probs uint8 = round(p × 255) + classmap = argmax logits. Softmax = mengubah skor mentah (logits) jadi
probabilitas yang jumlahnya 1 per piksel; argmax = kelas dengan skor tertinggi.

Run berjalan OFFLINE (HF_HUB_OFFLINE=1 + local_files_only): revision yang di-pin harus sudah ada di
cache HF. Unduhan = langkah terpisah: python -m rotoscope.segment --download [--seg-model 0.4b].

Uji manual (entry point sementara sampai cli.py, T-104b):
    python -m rotoscope.segment [--config PATH] [--seg-model 0.8b|0.4b] [--restart] [--limit N]
    python -m rotoscope.segment --qc-only
    python -m rotoscope.segment --download [--seg-model 0.4b]
Exit code: 0 sukses, 1 prasyarat gagal, 3 OOM di tengah run.
"""

from __future__ import annotations

import argparse
import gc
import io
import json
import os
import re
import shutil
import statistics
import sys
import time
import zipfile
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from rotoscope.config import (
    ConfigError, PipelineConfig, ensure_dir, load_class_names, load_pipeline, section_hash, to_dict,
)
from rotoscope.ingest import FRAME_PATTERN, FRAMES_DIRNAME, META_FILENAME

# ── Layout output (docs/01 [2]) ────────────────────
SEG_DIRNAME = "seg"
CLASSMAP_DIRNAME = "classmap"
PROBS_DIRNAME = "probs"
MANIFEST_FILENAME = "manifest.json"
FRAMES_LOG_FILENAME = "frames.jsonl"
QC_REPORT_FILENAME = "qc_report.json"
PROBS_KEY = "probs"                  # nama array di npz
PROBS_SCALE = 255                    # probs uint8 = round(p × 255)
TMP_SUFFIX = ".tmp"
CLASSMAP_SUFFIX = ".png"
PROBS_SUFFIX = ".npz"

# Kunci manifest yang harus sama persis untuk resume (beda apa pun → tolak, docs/01 [2]).
MANIFEST_MATCH_KEYS = ("model", "model_id", "revision", "precision", "processor", "num_labels", "frame_size")

# ── Checkpoint HF ──────────────────────────────────
COMMIT_HASH_RE = re.compile(r"[0-9a-f]{40}")
REQUIRED_FILES = ("config.json", "preprocessor_config.json", "model.safetensors")
HF_OFFLINE_ENV = "HF_HUB_OFFLINE"
DOWNLOAD_REVISION_FALLBACK = "main"  # hanya untuk --download kalau revision null

# ── Windows: os.replace bisa gagal sesaat (file dikunci antivirus/indexer, WinError 5/32) ─
REPLACE_RETRIES = 5
REPLACE_DELAY_S = 0.2

# ── Exit code (dipakai cli.py, T-104b) ─────────────
EXIT_OK = 0
EXIT_PRECONDITION = 1
EXIT_OOM = 3

MIB = 2**20
DEFAULT_CONFIG = Path("configs") / "default.yaml"
QC_METRICS = ("area_ratio", "iou_prev", "big_blobs", "area_vs_median", "finite")


class SegmentError(RuntimeError):
    """Prasyarat stage [2] gagal (exit 1): config/revision/cache/CUDA/VRAM/manifest/frame/tulis file."""


class SegmentOOMError(SegmentError):
    """GPU kehabisan memori di tengah run (exit 3). Tidak ada ganti model otomatis."""


class BackendOOM(RuntimeError):
    """Dilempar backend saat OOM; `vram` = kondisi VRAM saat gagal (MiB)."""

    def __init__(self, vram: dict):
        super().__init__("CUDA out of memory")
        self.vram = vram


# ── Backend ────────────────────────────────────────
@dataclass(frozen=True)
class ModelInfo:
    model: str                 # "0.8b" | "0.4b"
    model_id: str
    revision: str
    precision: str
    processor: dict            # {"size": {"height", "width"}, "do_pad": bool}
    num_labels: int


class SegBackend(Protocol):
    def describe(self) -> ModelInfo: ...
    def open(self) -> dict: ...
    def infer(self, rgb: np.ndarray) -> tuple[np.ndarray, dict]: ...
    def close(self) -> None: ...


def set_offline() -> None:
    """HF_HUB_OFFLINE=1 di environment proses — dipanggil SEBELUM transformers/huggingface_hub di-import.

    huggingface_hub membaca variabel ini saat di-import; kalau sudah ter-import lebih dulu,
    local_files_only=True di setiap from_pretrained tetap menjamin tidak ada akses jaringan.
    """
    os.environ[HF_OFFLINE_ENV] = "1"


def resolve_revision(cfg: PipelineConfig) -> str:
    """Revision model yang dipakai; wajib commit hash 40-hex (null / nama branch ditolak)."""
    s = cfg.segment
    rev = s.revision[s.model]
    if rev is None:
        raise SegmentError(
            f"segment.revision.{s.model} = null — run offline butuh commit hash checkpoint yang di-pin. "
            f"Isi di configs/default.yaml, atau jalankan dulu (online): "
            f"python -m rotoscope.segment --download --seg-model {s.model} (mencetak hash untuk di-pin)")
    if not COMMIT_HASH_RE.fullmatch(rev):
        raise SegmentError(f"segment.revision.{s.model} ({rev!r}) harus commit hash 40 karakter hex, "
                           f"bukan nama branch/tag — nama branch tidak mem-pin checkpoint")
    return rev


def ensure_cached(model_id: str, revision: str, model_key: str, cache_dir: str | Path | None = None) -> None:
    """Semua file checkpoint revision ini ada di cache HF — kalau tidak, berhenti + perintah unduh."""
    from huggingface_hub import try_to_load_from_cache

    missing = [f for f in REQUIRED_FILES
               if not isinstance(try_to_load_from_cache(model_id, f, cache_dir=cache_dir, revision=revision), str)]
    if missing:
        raise SegmentError(
            f"checkpoint {model_id} revision {revision} tidak ada di cache HF (hilang: {', '.join(missing)}). "
            f"Run berjalan offline — unduh dulu (online, sekali jalan):\n"
            f"    python -m rotoscope.segment --download --seg-model {model_key}")


def check_vram(free_mib: float, total_mib: float, need_mib: int, model_key: str) -> None:
    if free_mib >= need_mib:
        return
    alt = " atau jalankan ulang dengan --seg-model 0.4b (untuk SELURUH klip, pakai --restart)" \
        if model_key == "0.8b" else ""
    raise SegmentError(
        f"VRAM bebas {free_mib:.0f} MiB < dibutuhkan {need_mib} MiB untuk Sapiens2-seg {model_key} "
        f"(total {total_mib:.0f} MiB, segment.vram_min_free_mib.{model_key}). Tutup aplikasi lain yang "
        f"memakai GPU (browser, game, editor video){alt}.")


def _is_oom(torch, e: BaseException) -> bool:
    return isinstance(e, torch.OutOfMemoryError) or "out of memory" in str(e).lower()


class TorchSegBackend:
    """Sapiens2-seg lewat Transformers, satu model di GPU (proses sendiri)."""

    def __init__(self, cfg: PipelineConfig, cache_dir: str | Path | None = None):
        s = cfg.segment
        self.model_key = s.model
        self.model_id = s.model_ids[s.model]
        self.revision = resolve_revision(cfg)
        self.precision = s.precision
        self.vram_need_mib = s.vram_min_free_mib[s.model]
        self.cache_dir = cache_dir
        self._torch = None
        self._processor = None
        self._model = None
        self._dtype = None

    def _hf_kwargs(self) -> dict:
        return {"revision": self.revision, "local_files_only": True, "cache_dir": self.cache_dir}

    def describe(self) -> ModelInfo:
        set_offline()
        ensure_cached(self.model_id, self.revision, self.model_key, self.cache_dir)
        from transformers import AutoConfig, AutoImageProcessor

        conf = AutoConfig.from_pretrained(self.model_id, **self._hf_kwargs())
        self._processor = AutoImageProcessor.from_pretrained(self.model_id, **self._hf_kwargs())
        size = self._processor.size
        return ModelInfo(
            model=self.model_key, model_id=self.model_id, revision=self.revision, precision=self.precision,
            processor={"size": {"height": int(size["height"]), "width": int(size["width"])},
                       "do_pad": bool(self._processor.do_pad)},
            num_labels=int(conf.num_labels))

    def vram_state(self) -> dict:
        torch = self._torch
        free, total = torch.cuda.mem_get_info()
        return {"free_mib": round(free / MIB, 1), "total_mib": round(total / MIB, 1),
                "reserved_mib": round(torch.cuda.memory_reserved() / MIB, 1),
                "allocated_mib": round(torch.cuda.memory_allocated() / MIB, 1),
                "max_reserved_mib": round(torch.cuda.max_memory_reserved() / MIB, 1)}

    def _load_model(self):
        from transformers import Sapiens2ForSemanticSegmentation

        model = Sapiens2ForSemanticSegmentation.from_pretrained(self.model_id, dtype=self._dtype,
                                                                **self._hf_kwargs())
        return model.to("cuda").eval()

    def open(self) -> dict:
        set_offline()
        import torch

        self._torch = torch
        if not torch.cuda.is_available():
            raise SegmentError("CUDA tidak tersedia — stage [2] butuh GPU NVIDIA lewat PyTorch cu118. Cek driver "
                               "dan bahwa torch terinstall versi +cu118 (bukan CPU).")
        free, total = torch.cuda.mem_get_info()  # setelah CUDA init, sebelum load
        check_vram(free / MIB, total / MIB, self.vram_need_mib, self.model_key)
        if self._processor is None:
            self.describe()
        self._dtype = torch.float16 if self.precision == "fp16" else torch.float32
        t0 = time.perf_counter()
        try:
            self._model = self._load_model()
        except Exception as e:
            if _is_oom(torch, e):
                raise BackendOOM(self.vram_state()) from e
            raise
        free_after, _ = torch.cuda.mem_get_info()
        return {"vram_free_before_load_mib": round(free / MIB, 1), "vram_total_mib": round(total / MIB, 1),
                "vram_free_after_load_mib": round(free_after / MIB, 1),
                "load_s": round(time.perf_counter() - t0, 2),
                "attn_implementation": getattr(self._model.config, "_attn_implementation", None)}

    def infer(self, rgb: np.ndarray) -> tuple[np.ndarray, dict]:
        """RGB H×W×3 → logits float32 (num_labels, H, W) di CPU + peak VRAM frame ini."""
        torch = self._torch
        h, w = rgb.shape[:2]
        torch.cuda.reset_peak_memory_stats()
        try:
            inputs = self._processor(images=rgb, return_tensors="pt")
            pixel = inputs["pixel_values"].to("cuda", self._dtype)
            with torch.inference_mode():
                logits = self._model(pixel_values=pixel).logits
                logits = torch.nn.functional.interpolate(logits, size=(h, w), mode="bilinear",
                                                         align_corners=False)
                out = logits[0].cpu()  # pindah dulu, baru float32 di CPU (hemat VRAM)
            del inputs, pixel, logits
        except Exception as e:
            if _is_oom(torch, e):
                raise BackendOOM(self.vram_state()) from e
            raise
        stats = {"peak_reserved_mib": round(torch.cuda.max_memory_reserved() / MIB, 1),
                 "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / MIB, 1)}
        return out.float().numpy(), stats

    def close(self) -> None:
        self._model = None
        gc.collect()
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()


# ── Tulis atomik ───────────────────────────────────
def _replace_with_retry(tmp: Path, dst: Path) -> None:
    last: OSError | None = None
    for attempt in range(REPLACE_RETRIES):
        try:
            os.replace(tmp, dst)
            return
        except PermissionError as e:  # WinError 5/32: dikunci antivirus/indexer sesaat
            last = e
            if attempt + 1 < REPLACE_RETRIES:
                time.sleep(REPLACE_DELAY_S)
    tmp.unlink(missing_ok=True)
    raise SegmentError(f"gagal mengganti {dst} setelah {REPLACE_RETRIES} percobaan ({last}). File mungkin "
                       f"dikunci antivirus/indexer/aplikasi lain — tutup yang membuka file itu lalu jalankan "
                       f"ulang (frame lain tetap di-resume).")


def write_bytes_atomic(path: Path, data: bytes) -> None:
    """Tulis ke <nama>.tmp (fsync) lalu os.replace — pembaca tidak pernah melihat file setengah jadi."""
    tmp = path.with_name(path.name + TMP_SUFFIX)
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    _replace_with_retry(tmp, path)


def write_json_atomic(path: Path, data: Any) -> None:
    write_bytes_atomic(path, json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8"))


def write_classmap(path: Path, classmap: np.ndarray) -> None:
    ok, buf = cv2.imencode(CLASSMAP_SUFFIX, classmap)
    if not ok:
        raise SegmentError(f"gagal meng-encode classmap {path.name}")
    write_bytes_atomic(path, buf.tobytes())


def write_probs(path: Path, probs: np.ndarray) -> None:
    buf = io.BytesIO()
    np.savez_compressed(buf, **{PROBS_KEY: probs})
    write_bytes_atomic(path, buf.getvalue())


def append_jsonl(path: Path, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def read_jsonl(path: Path) -> list[dict]:
    """Baris rusak/terpotong (mis. proses mati saat menulis) dilewati."""
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict):
            out.append(rec)
    return out


# ── Baca + validasi ────────────────────────────────
def read_classmap(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    m = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_UNCHANGED)
    return m if m is not None and m.dtype == np.uint8 and m.ndim == 2 else None


def read_probs(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    try:
        with np.load(path) as z:
            return z[PROBS_KEY]
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
        return None


def classmap_valid(path: Path, h: int, w: int, num_classes: int) -> bool:
    m = read_classmap(path)
    return m is not None and m.shape == (h, w) and int(m.max()) < num_classes


def probs_valid(path: Path, h: int, w: int, num_classes: int) -> bool:
    p = read_probs(path)
    return p is not None and p.dtype == np.uint8 and p.shape == (num_classes, h, w)


def read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR) if path.is_file() else None
    if bgr is None:
        raise SegmentError(f"gagal membaca frame {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ── Inferensi → output ─────────────────────────────
def logits_to_outputs(logits: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    """logits (C, H, W) → (probs uint8 (C, H, W) = round(p × 255), classmap uint8 = argmax, finite).

    NaN/inf diganti 0 dulu (output tetap ditulis, frame gagal QC `finite`). Softmax stabil: kurangi max.
    """
    x = np.array(logits, dtype=np.float32)  # salinan: operasi in-place di bawah
    finite = bool(np.isfinite(x).all())
    if not finite:
        np.nan_to_num(x, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    classmap = x.argmax(axis=0).astype(np.uint8)
    x -= x.max(axis=0, keepdims=True)
    np.exp(x, out=x)
    x /= x.sum(axis=0, keepdims=True)
    x *= PROBS_SCALE
    probs = np.rint(x).astype(np.uint8)
    return probs, classmap, finite


def argmax_mismatch_pct(probs: np.ndarray, classmap: np.ndarray) -> float:
    """% piksel dengan argmax(probs) ≠ classmap (kuantisasi uint8 menggeser piksel yang nyaris seri)."""
    return float((probs.argmax(axis=0) != classmap).mean() * 100.0)


# ── Layout klip ────────────────────────────────────
@dataclass(frozen=True)
class Clip:
    work_dir: Path
    names: tuple[str, ...]       # frame_%05d.png berurutan
    indices: tuple[int, ...]
    width: int
    height: int

    @property
    def seg_dir(self) -> Path:
        return self.work_dir / SEG_DIRNAME

    @property
    def classmap_dir(self) -> Path:
        return self.seg_dir / CLASSMAP_DIRNAME

    @property
    def probs_dir(self) -> Path:
        return self.seg_dir / PROBS_DIRNAME

    @property
    def manifest_path(self) -> Path:
        return self.seg_dir / MANIFEST_FILENAME

    @property
    def frames_log(self) -> Path:
        return self.seg_dir / FRAMES_LOG_FILENAME

    @property
    def qc_report_path(self) -> Path:
        return self.work_dir / QC_REPORT_FILENAME

    def classmap_path(self, name: str) -> Path:
        return self.classmap_dir / (Path(name).stem + CLASSMAP_SUFFIX)

    def probs_path(self, name: str) -> Path:
        return self.probs_dir / (Path(name).stem + PROBS_SUFFIX)

    def frame_valid(self, name: str, num_classes: int) -> bool:
        return (classmap_valid(self.classmap_path(name), self.height, self.width, num_classes)
                and probs_valid(self.probs_path(name), self.height, self.width, num_classes))


def load_clip(work_dir: Path) -> Clip:
    meta_path = work_dir / META_FILENAME
    if not meta_path.is_file():
        raise SegmentError(f"{meta_path} tidak ada — jalankan stage [1] ingest dulu")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    start = int(meta.get("frame_index_start", 0))
    indices = tuple(range(start, start + int(meta["frame_count"])))
    names = tuple(FRAME_PATTERN % i for i in indices)
    return Clip(work_dir, names, indices, int(meta["working_width"]), int(meta["working_height"]))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build_manifest(info: ModelInfo, clip: Clip, classes: Sequence[str]) -> dict:
    return {"stage": "segment", **asdict(info),
            "frame_size": {"width": clip.width, "height": clip.height},
            "classes": list(classes),
            "probs": {"key": PROBS_KEY, "dtype": "uint8", "scale": PROBS_SCALE, "shape": "(num_labels, H, W)",
                      "format": "npz deflate (np.savez_compressed)"},
            "classmap": "argmax logits, uint8 PNG",
            "created_utc": _utc_now()}


def manifest_diff(old: dict, new: dict) -> list[str]:
    return [f"{k}: {old.get(k)!r} → {new.get(k)!r}" for k in MANIFEST_MATCH_KEYS if old.get(k) != new.get(k)]


def _has_outputs(clip: Clip) -> bool:
    return any(clip.classmap_dir.glob("frame_*" + CLASSMAP_SUFFIX)) or any(clip.probs_dir.glob("frame_*" + PROBS_SUFFIX))


def _clean_tmp(clip: Clip) -> None:
    if clip.seg_dir.is_dir():
        for p in clip.seg_dir.rglob("*" + TMP_SUFFIX):
            p.unlink(missing_ok=True)
    clip.qc_report_path.with_name(clip.qc_report_path.name + TMP_SUFFIX).unlink(missing_ok=True)


def restart_outputs(clip: Clip) -> None:
    """--restart: hapus output [2] lama (seg/ + qc_report.json)."""
    if clip.seg_dir.exists():
        shutil.rmtree(clip.seg_dir)
    clip.qc_report_path.unlink(missing_ok=True)


# ── Run ────────────────────────────────────────────
BackendFactory = Callable[[PipelineConfig], SegBackend]


def run_segment(cfg: PipelineConfig, *, restart: bool = False, limit: int | None = None,
                backend_factory: BackendFactory | None = None,
                log: Callable[[str], None] = print) -> dict:
    """Jalankan stage [2]. Return ringkasan run (+ ringkasan QC kalau semua frame klip valid)."""
    set_offline()  # sebelum transformers/huggingface_hub di-import (import malas di backend)
    if limit is not None and limit < 1:
        raise SegmentError(f"--limit harus ≥ 1, dapat {limit}")
    resolve_revision(cfg)
    clip = load_clip(cfg.paths.work_dir)
    selected = clip.names[:limit] if limit else clip.names
    missing = [n for n in selected if not (clip.work_dir / FRAMES_DIRNAME / n).is_file()]
    if missing:
        raise SegmentError(f"{len(missing)} frame hilang di {clip.work_dir / FRAMES_DIRNAME}, mis. {missing[0]} "
                           f"— jalankan ulang stage [1] ingest")

    backend = (backend_factory or TorchSegBackend)(cfg)
    info = backend.describe()
    classes = load_class_names()
    if info.num_labels != len(classes):
        raise SegmentError(f"self-check gagal: num_labels model {info.model_id} = {info.num_labels}, "
                           f"daftar kelas paket = {len(classes)} kelas")

    if restart:
        restart_outputs(clip)
    manifest = build_manifest(info, clip, classes)
    if clip.manifest_path.is_file():
        old = json.loads(clip.manifest_path.read_text(encoding="utf-8"))
        diff = manifest_diff(old, manifest)
        if diff:
            raise SegmentError("output [2] di " + str(clip.seg_dir) + " dibuat dengan setelan berbeda "
                               "(model tidak pernah dicampur dalam satu klip):\n  " + "\n  ".join(diff) +
                               "\nJalankan dengan --restart untuk menghapus output lama dan mulai dari awal.")
    elif _has_outputs(clip):
        raise SegmentError(f"{clip.seg_dir} berisi output tanpa {MANIFEST_FILENAME} — asal output tidak "
                           f"diketahui. Jalankan dengan --restart.")
    _clean_tmp(clip)

    todo = [(i, n) for i, n in zip(clip.indices[:len(selected)], selected)
            if not clip.frame_valid(n, info.num_labels)]
    n_skip = len(selected) - len(todo)
    log(f"[2] segment {info.model} ({info.model_id}@{info.revision[:8]}, {info.precision}): "
        f"{len(selected)} frame dipilih, {n_skip} valid dilewati, {len(todo)} diproses")

    run = {"model": info.model, "selected": len(selected), "skipped": n_skip, "processed": 0,
           "frames": [], "open": None}
    if todo:
        ensure_dir(clip.classmap_dir)
        ensure_dir(clip.probs_dir)
        if not clip.manifest_path.is_file():
            write_json_atomic(clip.manifest_path, manifest)
        clip.qc_report_path.unlink(missing_ok=True)  # basi begitu ada frame baru
        _process(clip, backend, info, todo, run, log)

    whole_clip = len(selected) == len(clip.names)  # _process melempar error kalau ada frame gagal
    run["qc"] = run_qc(cfg, log=log)["summary"] if whole_clip else None
    if not whole_clip:
        log(f"QC dilewati: baru {len(selected)}/{len(clip.names)} frame (--limit); QC butuh semua frame klip")
    return run


def _process(clip: Clip, backend: SegBackend, info: ModelInfo, todo: list[tuple[int, str]],
             run: dict, log: Callable[[str], None]) -> None:
    t_run = time.perf_counter()
    try:
        opened = backend.open()
    except BackendOOM as e:
        append_jsonl(clip.frames_log, {"event": "oom", "frame": None, "stage": "load",
                                       "time_utc": _utc_now(), **e.vram})
        raise SegmentOOMError(f"OOM saat memuat model {info.model}: {e.vram}") from None
    run["open"] = opened
    append_jsonl(clip.frames_log, {"event": "run_start", "time_utc": _utc_now(), "model": info.model,
                                   "revision": info.revision, "n_todo": len(todo), **(opened or {})})
    try:
        for k, (index, name) in enumerate(todo, 1):
            t0 = time.perf_counter()
            rgb = read_rgb(clip.work_dir / FRAMES_DIRNAME / name)
            if rgb.shape[:2] != (clip.height, clip.width):
                raise SegmentError(f"{name}: ukuran {rgb.shape[1]}×{rgb.shape[0]} ≠ meta.json "
                                   f"{clip.width}×{clip.height} — jalankan ulang ingest")
            t1 = time.perf_counter()
            try:
                logits, stats = backend.infer(rgb)
            except BackendOOM as e:
                append_jsonl(clip.frames_log, {"event": "oom", "frame": name, "index": index,
                                               "time_utc": _utc_now(), **e.vram})
                raise SegmentOOMError(
                    f"OOM di frame {name} ({k}/{len(todo)}): {e.vram}. Frame sebelumnya tersimpan; jalankan "
                    f"ulang untuk melanjutkan (tutup aplikasi lain yang memakai GPU). Model TIDAK diganti "
                    f"otomatis.") from None
            if logits.shape != (info.num_labels, clip.height, clip.width):
                raise SegmentError(f"{name}: shape logits {logits.shape} ≠ "
                                   f"{(info.num_labels, clip.height, clip.width)}")
            t2 = time.perf_counter()
            probs, classmap, finite = logits_to_outputs(logits)
            mismatch = argmax_mismatch_pct(probs, classmap)
            t3 = time.perf_counter()
            write_probs(clip.probs_path(name), probs)
            write_classmap(clip.classmap_path(name), classmap)
            t4 = time.perf_counter()
            rec = {"event": "frame", "frame": name, "index": index, "time_utc": _utc_now(), "model": info.model,
                   "read_s": round(t1 - t0, 4), "infer_s": round(t2 - t1, 4), "post_s": round(t3 - t2, 4),
                   "write_s": round(t4 - t3, 4), "total_s": round(t4 - t0, 4), **stats,
                   "finite": finite, "argmax_mismatch_pct": round(mismatch, 5)}
            append_jsonl(clip.frames_log, rec)
            run["frames"].append(rec)
            run["processed"] += 1
            log(f"  [{k}/{len(todo)}] {name} {rec['total_s']:.2f} s, peak reserved "
                f"{stats.get('peak_reserved_mib', '-')} MiB, beda argmax {mismatch:.4f}%"
                + ("" if finite else ", NaN/inf!"))
    finally:
        backend.close()
        append_jsonl(clip.frames_log, {"event": "run_end", "time_utc": _utc_now(),
                                       "n_done": run["processed"], "wall_s": round(time.perf_counter() - t_run, 2)})


# ── QC ─────────────────────────────────────────────
def iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU dua mask boolean (irisan / gabungan). Keduanya kosong → 1.0."""
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else float(np.logical_and(a, b).sum() / union)


def big_blob_count(fg: np.ndarray, blob_min: float) -> int:
    """Jumlah komponen 8-arah foreground dengan luas > blob_min × luas frame."""
    n, _, stats, _ = cv2.connectedComponentsWithStats(fg.astype(np.uint8), connectivity=8)
    min_px = blob_min * fg.size
    return int(sum(1 for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] > min_px))


def label_agreement(cm: np.ndarray, prev: np.ndarray) -> float | None:
    """Fraksi piksel foreground di kedua frame yang kelasnya sama; irisan kosong → None."""
    both = (cm != 0) & (prev != 0)
    return float((cm[both] == prev[both]).mean()) if both.any() else None


def shifted_window(i: int, n: int, window: int) -> tuple[int, int]:
    """Jendela [start, end) berisi selalu W sampel dan memuat i; digeser (bukan dipotong) di tepi klip.

    Tengah klip: berpusat di i. n < W → seluruh klip.
    """
    if n <= window:
        return 0, n
    half = (window - 1) // 2
    start = min(max(i - half, 0), n - window)
    return start, start + window


def area_vs_median(areas: Sequence[float], window: int) -> list[tuple[float | None, float | None]]:
    """Per frame: (median area jendela digeser, area / median). Median 0 → (0.0, None)."""
    a = np.asarray(areas, dtype=np.float64)
    out = []
    for i in range(len(a)):
        s, e = shifted_window(i, len(a), window)
        med = float(np.median(a[s:e]))
        out.append((med, float(a[i] / med) if med > 0 else None))
    return out


def fail_reasons(row: dict, q) -> list[str]:
    reasons = []
    if row["area_ratio"] < q.area_min or row["area_ratio"] > q.area_max:
        reasons.append("area_ratio")
    if row["iou_prev"] is not None and row["iou_prev"] < q.iou_min:
        reasons.append("iou_prev")
    if row["big_blobs"] > q.max_big_blobs:
        reasons.append("big_blobs")
    if row["area_vs_median"] is not None and row["area_vs_median"] < q.area_drop_min:
        reasons.append("area_vs_median")
    if row["finite"] is False:
        reasons.append("finite")
    return reasons


def compute_qc(classmaps: Sequence[np.ndarray] | Any, q, names: Sequence[str], indices: Sequence[int],
               frame_log: dict[str, dict] | None = None) -> list[dict]:
    """Metrik QC per frame dari classmap berurutan (iterable). Foreground = classmap ≠ 0."""
    frame_log = frame_log or {}
    rows, prev_cm = [], None
    for name, index, cm in zip(names, indices, classmaps):
        fg = cm != 0
        prev_fg = prev_cm != 0 if prev_cm is not None else None
        rec = frame_log.get(name, {})
        rows.append({
            "frame": name, "index": index,
            "area_px": int(fg.sum()),
            "area_ratio": float(fg.mean()),
            "iou_prev": iou(fg, prev_fg) if prev_fg is not None else None,
            "big_blobs": big_blob_count(fg, q.blob_min),
            "label_agreement_prev": label_agreement(cm, prev_cm) if prev_cm is not None else None,
            "finite": rec.get("finite"),
            "argmax_mismatch_pct": rec.get("argmax_mismatch_pct"),
        })
        prev_cm = cm
    for row, (med, ratio) in zip(rows, area_vs_median([r["area_px"] for r in rows], q.area_median_window)):
        row["area_median_px"] = med
        row["area_vs_median"] = ratio
    for row in rows:
        row["fail_reasons"] = fail_reasons(row, q)
    return rows


def _stats(values: list) -> dict | None:
    v = [x for x in values if x is not None]
    if not v:
        return None
    return {"mean": statistics.fmean(v), "median": statistics.median(v), "min": min(v), "max": max(v)}


def summarize_qc(rows: list[dict]) -> dict:
    return {
        "n_frames": len(rows),
        "n_fail": sum(bool(r["fail_reasons"]) for r in rows),
        "fail_counts": {m: sum(m in r["fail_reasons"] for r in rows) for m in QC_METRICS},
        "fail_frames": [r["index"] for r in rows if r["fail_reasons"]],
        "finite_false": sum(r["finite"] is False for r in rows),
        "finite_unknown": sum(r["finite"] is None for r in rows),
        "stats": {k: _stats([r[k] for r in rows]) for k in
                  ("area_ratio", "iou_prev", "big_blobs", "area_vs_median", "label_agreement_prev",
                   "argmax_mismatch_pct")},
    }


def last_frame_records(frames_log: Path) -> dict[str, dict]:
    return {r["frame"]: r for r in read_jsonl(frames_log) if r.get("event") == "frame" and "frame" in r}


def run_qc(cfg: PipelineConfig, *, log: Callable[[str], None] = print) -> dict:
    """QC per klip (butuh semua frame valid). Tanpa GPU/torch. Tulis qc_report.json (atomik)."""
    clip = load_clip(cfg.paths.work_dir)
    if not clip.manifest_path.is_file():
        raise SegmentError(f"{clip.manifest_path} tidak ada — jalankan stage [2] dulu")
    manifest = json.loads(clip.manifest_path.read_text(encoding="utf-8"))
    size = {"width": clip.width, "height": clip.height}
    if manifest.get("frame_size") != size:
        raise SegmentError(f"ukuran frame manifest {manifest.get('frame_size')} ≠ meta.json {size} — "
                           f"jalankan stage [2] dengan --restart")
    num = int(manifest["num_labels"])
    invalid = [n for n in clip.names if not clip.frame_valid(n, num)]
    if invalid:
        raise SegmentError(f"QC butuh semua frame valid: {len(invalid)}/{len(clip.names)} belum/rusak, "
                           f"mis. {invalid[0]} — jalankan stage [2] (tanpa --limit) dulu")

    classmaps = (read_classmap(clip.classmap_path(n)) for n in clip.names)
    rows = compute_qc(classmaps, cfg.qc, clip.names, clip.indices, last_frame_records(clip.frames_log))
    summary = summarize_qc(rows)
    report = {"stage": "segment_qc", "created_utc": _utc_now(),
              "seg_manifest": {k: manifest.get(k) for k in ("model", "model_id", "revision", "precision")},
              "qc": to_dict(cfg.qc), "qc_hash": section_hash(cfg, "qc"),
              "definitions": {"foreground": "classmap != 0",
                              "area_vs_median": "area / median area jendela digeser W frame (selalu W sampel, "
                                                "memuat frame ini; n < W → seluruh klip)"},
              "summary": summary, "frames": rows}
    write_json_atomic(clip.qc_report_path, report)
    log(f"QC: {summary['n_fail']}/{summary['n_frames']} frame gagal {summary['fail_counts']} → "
        f"{clip.qc_report_path}")
    return report


# ── Download (online, terpisah dari run) ───────────
def download(cfg: PipelineConfig, *, log: Callable[[str], None] = print) -> str:
    """Unduh checkpoint model terpilih ke cache HF. Return commit hash snapshot."""
    s = cfg.segment
    model_id, rev = s.model_ids[s.model], s.revision[s.model]
    if rev is not None and not COMMIT_HASH_RE.fullmatch(rev):
        raise SegmentError(f"segment.revision.{s.model} ({rev!r}) harus commit hash 40 karakter hex")
    if os.environ.get(HF_OFFLINE_ENV, "").strip().lower() in ("1", "true", "yes", "on"):
        raise SegmentError(f"{HF_OFFLINE_ENV} aktif di environment — --download butuh online; hapus variabel itu")
    from huggingface_hub import snapshot_download

    path = Path(snapshot_download(model_id, revision=rev or DOWNLOAD_REVISION_FALLBACK,
                                  allow_patterns=list(REQUIRED_FILES)))
    commit = path.name
    log(f"{model_id} snapshot {commit} → {path}")
    if rev is None:
        log(f"revision belum di-pin — isi segment.revision.\"{s.model}\": \"{commit}\" di configs/default.yaml")
    return commit


# ── Entry point sementara (cli.py = T-104b) ────────
def main(argv: list[str] | None = None, backend_factory: BackendFactory | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m rotoscope.segment",
                                description="Stage [2]: frames → seg/classmap + seg/probs + qc_report.json")
    p.add_argument("--config", type=Path, default=None,
                   help=f"YAML pipeline (default: {DEFAULT_CONFIG.as_posix()} kalau ada, selain itu default kode)")
    p.add_argument("--seg-model", choices=("0.8b", "0.4b"), default=None,
                   help="override segment.model (untuk SELURUH klip)")
    p.add_argument("--restart", action="store_true", help="hapus output [2] lama lalu mulai dari awal")
    p.add_argument("--limit", type=int, default=None, help="hanya N frame pertama")
    p.add_argument("--qc-only", action="store_true", help="hanya QC (tanpa GPU), butuh semua frame valid")
    p.add_argument("--download", action="store_true", help="unduh checkpoint (online) lalu keluar")
    args = p.parse_args(argv)
    # stdout/stderr yang diarahkan ke file di Windows = cp1252: karakter seperti "→" / "≠" di pesan
    # membuat print crash (UnicodeEncodeError) → ganti dengan "?" daripada gagal.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    def log(msg: str) -> None:
        print(msg, flush=True)  # progress tetap terlihat walau stdout diarahkan ke file

    if args.qc_only and (args.restart or args.limit is not None or args.download or args.seg_model):
        p.error("--qc-only tidak bisa digabung dengan --restart/--limit/--download/--seg-model")
    if args.download and (args.restart or args.limit is not None):
        p.error("--download tidak bisa digabung dengan --restart/--limit")

    try:
        path = args.config if args.config is not None else (DEFAULT_CONFIG if DEFAULT_CONFIG.is_file() else None)
        cfg = load_pipeline(path, overrides={"segment.model": args.seg_model} if args.seg_model else None)
        if args.download:
            download(cfg, log=log)
        elif args.qc_only:
            run_qc(cfg, log=log)
        else:
            t0 = time.perf_counter()
            run = run_segment(cfg, restart=args.restart, limit=args.limit, backend_factory=backend_factory,
                              log=log)
            log(f"selesai: {run['processed']} diproses, {run['skipped']} dilewati "
                  f"({time.perf_counter() - t0:.1f} s)")
    except SegmentOOMError as e:
        print(f"ERROR (OOM): {e}", file=sys.stderr)
        return EXIT_OOM
    except (SegmentError, ConfigError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return EXIT_PRECONDITION
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
