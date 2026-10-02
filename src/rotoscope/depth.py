"""Stage [2c] depth: frames/*.png → depth/frame_%05d.npy + depth/manifest.json + depth/frames.jsonl
(Depth Anything V2 Small, D-009, D-010). Kontrak lengkap: docs/01 [2c].

Output = disparity RELATIF mentah (besar = dekat), hanya benar sampai skala + offset per frame —
normalisasi per frame ada di [3]. Disparity = kebalikan jarak (1/Z) versi model, tanpa satuan.

Dua lapis (pola sama dengan segment.py):
- Backend model (TorchDepthBackend) — satu-satunya bagian yang meng-import torch/transformers.
  describe() tanpa GPU (cek cache + lisensi + varian Small + ukuran input processor), open() = CUDA +
  cek VRAM + load, infer() = disparity float32 resolusi kerja di CPU.
- Logika stage (run_depth) — numpy saja: resume, manifest, tulis atomik. Unit test memakai backend
  palsu lewat `backend_factory`.

Per frame: image processor default (DPT: sisi pendek 518, rasio dipertahankan, kelipatan 14 → frame
480×854 masuk sebagai 518×924) → forward fp32 → post_process_depth_estimation ke resolusi kerja (GPU)
→ CPU → float16 → tulis atomik. NaN/inf (termasuk nilai di luar jangkauan float16) diganti 0, file
tetap ditulis, `finite: false` di frames.jsonl — frame itu tidak diproses ulang otomatis.

Lisensi: hanya varian Small (Apache-2.0); Base/Large CC-BY-NC DILARANG. Dicek tiga lapis: nama model
id (config + backend), backbone hidden_size 384 (ViT-S), dan front-matter `license` model card
(README.md) di cache = apache-2.0.

CLI final (T-104b): python -m rotoscope depth <video> [--config PATH] [--restart --yes] [--limit N] [--adopt];
atau seluruh pipeline: python -m rotoscope run <video>. cli.py menjalankan modul ini sebagai subprocess
(--work-dir <folder klip>); --yes hanya ada di cli (penghapusan hasil GPU wajib --yes, dibuang sebelum diteruskan).
Flag main() stage:
    python -m rotoscope.depth [--config PATH] [--work-dir DIR] [--restart] [--limit N]
    python -m rotoscope.depth --adopt            (catat identitas klip ke manifest lama, tanpa inferensi)
    python -m rotoscope.depth --download         (lewat cli: python -m rotoscope download)
Exit code: 0 sukses, 1 prasyarat gagal, 3 OOM di tengah run.

Identitas klip (T-108): manifest memuat `clip` (sha256 meta.json + source_path). Output klip lain / manifest
tanpa identitas → ditolak SEBELUM model dimuat (juga untuk --limit); jalan: --restart (identitas berbeda)
atau --adopt (manifest lama tanpa identitas).
"""

from __future__ import annotations

import argparse
import gc
import io
import json
import shutil
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import yaml

from rotoscope.config import DEPTH_MODEL_ID_REQUIRED, ConfigError, PipelineConfig, ensure_dir, load_pipeline
from rotoscope.ingest import FRAMES_DIRNAME
from rotoscope.stage_common import (
    COMMIT_HASH_RE, DOWNLOAD_REVISION_FALLBACK, EXIT_OK, EXIT_OOM, EXIT_PRECONDITION, MIB, BackendOOM,
    StageError, StageOOMError, add_work_dir_arg, adopt_identity, append_jsonl, cached_file, clean_tmp, cli_cmd,
    clip_identity, ensure_cached, hf_offline_active, is_oom, load_frame_list, read_rgb, reconfigure_stdio,
    require_frames, require_same_clip, set_offline, utc_now, vram_state, work_dir_overrides, write_bytes_atomic,
    write_json_atomic,
)

# ── Layout output (docs/01 [2c]) ───────────────────
DEPTH_DIRNAME = "depth"
MANIFEST_FILENAME = "manifest.json"
FRAMES_LOG_FILENAME = "frames.jsonl"
DEPTH_SUFFIX = ".npy"
DEPTH_DTYPE = np.float16
NONFINITE_FILL = 0.0                 # pengganti NaN/inf (sama dengan segment.py)
OUTPUT_INFO = {"kind": "disparity relatif mentah (besar = dekat), benar sampai skala + offset per frame",
               "dtype": "float16", "shape": "(H, W) resolusi kerja",
               "resize": "post_process_depth_estimation (GPU)", "nonfinite": "diganti 0, finite=false di frames.jsonl"}

# Kunci manifest yang harus sama persis untuk resume (beda apa pun → tolak).
MANIFEST_MATCH_KEYS = ("model_id", "revision", "license", "precision", "processor", "input_size", "output",
                       "frame_size")

# ── Checkpoint HF + lisensi ────────────────────────
MODEL_CARD = "README.md"
REQUIRED_FILES = ("config.json", "preprocessor_config.json", "model.safetensors", MODEL_CARD)
REQUIRED_LICENSE = "apache-2.0"
SMALL_HIDDEN_SIZE = 384              # backbone DINOv2 ViT-S (Small); Base 768, Large 1024
DOWNLOAD_CMD = "python -m rotoscope download"
LICENSE_HINT = "Base/Large berlisensi CC-BY-NC (non-komersial) — DILARANG; hanya Small (Apache-2.0)"

DEFAULT_CONFIG = Path("configs") / "default.yaml"

# Alias error bersama (stage_common): prasyarat gagal → exit 1, OOM di tengah run → exit 3.
DepthError = StageError
DepthOOMError = StageOOMError


def stage_cmd(work_dir: Path | None = None) -> str:
    """Perintah CLI stage [2c] untuk pesan error (path video dari meta.json)."""
    return cli_cmd("depth", work_dir)


# ── Backend ────────────────────────────────────────
@dataclass(frozen=True)
class ModelInfo:
    model_id: str
    revision: str
    license: str
    precision: str
    processor: dict            # class, size, keep_aspect_ratio, ensure_multiple_of, do_pad, resample
    input_size: dict           # {"height", "width"} tensor masuk model untuk frame klip ini


class DepthBackend(Protocol):
    def describe(self, height: int, width: int) -> ModelInfo: ...
    def open(self) -> dict: ...
    def infer(self, rgb: np.ndarray) -> tuple[np.ndarray, dict]: ...
    def close(self) -> None: ...


def require_small(model_id: str) -> None:
    """Lapis 1: nama model id wajib varian Small (dicek ulang di sini, bukan hanya di config)."""
    if DEPTH_MODEL_ID_REQUIRED not in model_id:
        raise DepthError(f"depth.model_id {model_id!r} bukan varian {DEPTH_MODEL_ID_REQUIRED} — {LICENSE_HINT}")


def require_small_backbone(hidden_size: int | None, model_id: str) -> None:
    """Lapis 2: arsitektur backbone = ViT-S (hidden_size 384), walau namanya mengandung "Small"."""
    if hidden_size != SMALL_HIDDEN_SIZE:
        raise DepthError(f"{model_id}: backbone hidden_size {hidden_size} ≠ {SMALL_HIDDEN_SIZE} (ViT-S) — "
                         f"checkpoint ini bukan Depth Anything V2 Small. {LICENSE_HINT}")


def card_license(text: str) -> str | None:
    """Nilai `license` di front-matter YAML model card (blok di antara '---' di awal file)."""
    lines = text.lstrip("﻿").splitlines()
    if not lines or lines[0].strip() != "---":
        return None
    try:
        end = next(i for i in range(1, len(lines)) if lines[i].strip() == "---")
        meta = yaml.safe_load("\n".join(lines[1:end]))
    except (StopIteration, yaml.YAMLError):
        return None
    lic = meta.get("license") if isinstance(meta, dict) else None
    return str(lic).strip().lower() if lic is not None else None


def require_license(text: str, model_id: str) -> str:
    """Lapis 3: front-matter model card = apache-2.0."""
    lic = card_license(text)
    if lic != REQUIRED_LICENSE:
        raise DepthError(f"lisensi model card {model_id} = {lic!r}, dibutuhkan {REQUIRED_LICENSE!r} — {LICENSE_HINT}")
    return lic


def resolve_revision(cfg: PipelineConfig) -> str:
    """Revision DA-V2 Small; wajib commit hash 40-hex (null / nama branch ditolak)."""
    rev = cfg.depth.revision
    if rev is None:
        raise DepthError(
            f"depth.revision = null — run offline butuh commit hash checkpoint yang di-pin. Isi di "
            f"configs/default.yaml, atau jalankan dulu (online): {DOWNLOAD_CMD} (mencetak hash untuk di-pin)")
    if not COMMIT_HASH_RE.fullmatch(rev):
        raise DepthError(f"depth.revision ({rev!r}) harus commit hash 40 karakter hex, bukan nama branch/tag — "
                         f"nama branch tidak mem-pin checkpoint")
    return rev


def check_vram(free_mib: float, total_mib: float, need_mib: int) -> None:
    if free_mib >= need_mib:
        return
    raise DepthError(
        f"VRAM bebas {free_mib:.0f} MiB < dibutuhkan {need_mib} MiB untuk Depth Anything V2 Small "
        f"(total {total_mib:.0f} MiB, depth.vram_min_free_mib). Tutup aplikasi lain yang memakai GPU "
        f"(browser, game, editor video), dan pastikan stage [2] tidak sedang jalan.")


def _processor_info(proc) -> dict:
    size = proc.size
    return {"class": type(proc).__name__,
            "size": {"height": int(size["height"]), "width": int(size["width"])},
            "keep_aspect_ratio": bool(getattr(proc, "keep_aspect_ratio", False)),
            "ensure_multiple_of": int(getattr(proc, "ensure_multiple_of", 1) or 1),
            "do_pad": bool(getattr(proc, "do_pad", False)),
            "resample": int(getattr(proc, "resample", -1))}


class TorchDepthBackend:
    """Depth Anything V2 Small lewat Transformers, satu model di GPU (proses sendiri)."""

    def __init__(self, cfg: PipelineConfig, cache_dir: str | Path | None = None):
        d = cfg.depth
        self.model_id = d.model_id
        require_small(self.model_id)
        self.revision = resolve_revision(cfg)
        self.precision = d.precision
        self.vram_need_mib = d.vram_min_free_mib
        self.cache_dir = cache_dir
        self._torch = None
        self._processor = None
        self._model = None
        self._dtype = None

    def _hf_kwargs(self) -> dict:
        return {"revision": self.revision, "local_files_only": True, "cache_dir": self.cache_dir}

    def describe(self, height: int, width: int) -> ModelInfo:
        set_offline()
        ensure_cached(self.model_id, self.revision, REQUIRED_FILES, DOWNLOAD_CMD, self.cache_dir)
        card = cached_file(self.model_id, MODEL_CARD, self.revision, self.cache_dir)
        lic = require_license(card.read_text(encoding="utf-8", errors="replace"), self.model_id)
        from transformers import AutoConfig, AutoImageProcessor

        conf = AutoConfig.from_pretrained(self.model_id, **self._hf_kwargs())
        backbone = getattr(conf, "backbone_config", None)
        require_small_backbone(getattr(backbone, "hidden_size", None), self.model_id)
        self._processor = AutoImageProcessor.from_pretrained(self.model_id, **self._hf_kwargs())
        # Ukuran input nyata untuk frame klip ini (processor saja, CPU).
        pixel = self._processor(images=np.zeros((height, width, 3), np.uint8), return_tensors="pt")["pixel_values"]
        return ModelInfo(model_id=self.model_id, revision=self.revision, license=lic, precision=self.precision,
                         processor=_processor_info(self._processor),
                         input_size={"height": int(pixel.shape[-2]), "width": int(pixel.shape[-1])})

    def _load_model(self):
        from transformers import DepthAnythingForDepthEstimation

        model = DepthAnythingForDepthEstimation.from_pretrained(self.model_id, dtype=self._dtype,
                                                                **self._hf_kwargs())
        require_small_backbone(getattr(model.config.backbone_config, "hidden_size", None), self.model_id)
        model.requires_grad_(False)
        return model.to("cuda").eval()

    def open(self) -> dict:
        set_offline()
        import torch

        self._torch = torch
        if not torch.cuda.is_available():
            raise DepthError("CUDA tidak tersedia — stage [2c] butuh GPU NVIDIA lewat PyTorch cu118. Cek driver "
                             "dan bahwa torch terinstall versi +cu118 (bukan CPU).")
        free, total = torch.cuda.mem_get_info()  # setelah CUDA init, sebelum load
        check_vram(free / MIB, total / MIB, self.vram_need_mib)
        if self._processor is None:
            raise DepthError("describe() harus dipanggil sebelum open()")
        self._dtype = torch.float16 if self.precision == "fp16" else torch.float32
        t0 = time.perf_counter()
        try:
            self._model = self._load_model()
        except Exception as e:
            if is_oom(torch, e):
                raise BackendOOM(vram_state(torch)) from e
            raise
        free_after, _ = torch.cuda.mem_get_info()
        return {"vram_free_before_load_mib": round(free / MIB, 1), "vram_total_mib": round(total / MIB, 1),
                "vram_free_after_load_mib": round(free_after / MIB, 1),
                "load_s": round(time.perf_counter() - t0, 2)}

    def infer(self, rgb: np.ndarray) -> tuple[np.ndarray, dict]:
        """RGB H×W×3 → disparity float32 (H, W) di CPU + peak VRAM frame ini."""
        torch = self._torch
        h, w = rgb.shape[:2]
        torch.cuda.reset_peak_memory_stats()
        try:
            pixel = self._processor(images=rgb, return_tensors="pt")["pixel_values"].to("cuda", self._dtype)
            with torch.inference_mode():
                out = self._model(pixel_values=pixel)
                res = self._processor.post_process_depth_estimation(out, target_sizes=[(h, w)])[0]
                disp = res["predicted_depth"].float().cpu()
            del pixel, out, res
        except Exception as e:
            if is_oom(torch, e):
                raise BackendOOM(vram_state(torch)) from e
            raise
        stats = {"peak_reserved_mib": round(torch.cuda.max_memory_reserved() / MIB, 1),
                 "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / MIB, 1)}
        return disp.numpy(), stats

    def close(self) -> None:
        self._model = None
        gc.collect()
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()


# ── Disparity → output ─────────────────────────────
def to_output(disparity: np.ndarray) -> tuple[np.ndarray, bool, int]:
    """float32 (H, W) → (float16, finite, n piksel tidak finite).

    NaN/inf dari model DAN nilai yang meluap saat dijadikan float16 (|x| > 65504) diganti 0.
    """
    x = np.asarray(disparity, dtype=np.float32)
    with np.errstate(over="ignore", invalid="ignore"):
        y = x.astype(DEPTH_DTYPE)
    bad = ~np.isfinite(y)
    n_bad = int(bad.sum())
    if n_bad:
        y[bad] = NONFINITE_FILL
    return y, n_bad == 0, n_bad


def disparity_stats(y: np.ndarray) -> dict:
    """min / median / max nilai yang TERSIMPAN (seluruh frame; [2c] tidak tahu foreground)."""
    v = y.astype(np.float32)
    return {"min": round(float(v.min()), 5), "median": round(float(np.median(v)), 5), "max": round(float(v.max()), 5)}


def write_depth(path: Path, y: np.ndarray) -> None:
    buf = io.BytesIO()
    np.save(buf, y, allow_pickle=False)
    write_bytes_atomic(path, buf.getvalue())


def read_depth(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    try:
        return np.load(path, allow_pickle=False)
    except (OSError, ValueError, EOFError):
        return None


def depth_valid(path: Path, h: int, w: int) -> bool:
    """Terbaca, float16, ukuran = frame, semua finite (NaN/inf di disk = rusak → diproses ulang)."""
    d = read_depth(path)
    return d is not None and d.dtype == DEPTH_DTYPE and d.shape == (h, w) and bool(np.isfinite(d).all())


# ── Layout klip ────────────────────────────────────
@dataclass(frozen=True)
class Clip:
    work_dir: Path
    names: tuple[str, ...]
    indices: tuple[int, ...]
    width: int
    height: int

    @property
    def depth_dir(self) -> Path:
        return self.work_dir / DEPTH_DIRNAME

    @property
    def manifest_path(self) -> Path:
        return self.depth_dir / MANIFEST_FILENAME

    @property
    def frames_log(self) -> Path:
        return self.depth_dir / FRAMES_LOG_FILENAME

    def depth_path(self, name: str) -> Path:
        return self.depth_dir / (Path(name).stem + DEPTH_SUFFIX)

    def frame_valid(self, name: str) -> bool:
        return depth_valid(self.depth_path(name), self.height, self.width)


def load_clip(work_dir: Path) -> Clip:
    return Clip(work_dir, *load_frame_list(work_dir))


def build_manifest(info: ModelInfo, clip: Clip) -> dict:
    return {"stage": "depth", **asdict(info), "output": dict(OUTPUT_INFO),
            "frame_size": {"width": clip.width, "height": clip.height},
            "clip": clip_identity(clip.work_dir), "created_utc": utc_now()}


def _require_same_clip(clip: Clip) -> None:
    """Manifest [2c] yang ada harus milik klip ini (meta.json saat ini). Tanpa manifest: tidak ada yang dicek."""
    if clip.manifest_path.is_file():
        manifest = json.loads(clip.manifest_path.read_text(encoding="utf-8"))
        require_same_clip(manifest, clip_identity(clip.work_dir), stage="[2c]", out_dir=clip.depth_dir,
                          cmd=stage_cmd(clip.work_dir))


def adopt_depth(cfg: PipelineConfig, *, log: Callable[[str], None] = print) -> dict:
    """--adopt: catat identitas klip saat ini ke depth/manifest.json yang ada. Tanpa GPU/model/inferensi."""
    clip = load_clip(cfg.paths.work_dir)
    stems = {Path(n).stem for n in clip.names}
    orphans = ([p for p in sorted(clip.depth_dir.glob("frame_*" + DEPTH_SUFFIX)) if p.stem not in stems]
               if clip.depth_dir.is_dir() else [])
    return adopt_identity(stage="[2c]", manifest_path=clip.manifest_path, frames_log=clip.frames_log,
                          work_dir=clip.work_dir, size={"width": clip.width, "height": clip.height},
                          names=clip.names, frame_valid=clip.frame_valid, orphans=orphans,
                          cmd=stage_cmd(clip.work_dir), log=log)


def manifest_diff(old: dict, new: dict) -> list[str]:
    return [f"{k}: {old.get(k)!r} → {new.get(k)!r}" for k in MANIFEST_MATCH_KEYS if old.get(k) != new.get(k)]


def _has_outputs(clip: Clip) -> bool:
    return clip.depth_dir.is_dir() and any(clip.depth_dir.glob("frame_*" + DEPTH_SUFFIX))


def restart_outputs(clip: Clip) -> None:
    """--restart: hapus output [2c] lama (depth/ saja)."""
    if clip.depth_dir.exists():
        shutil.rmtree(clip.depth_dir)


# ── Run ────────────────────────────────────────────
BackendFactory = Callable[[PipelineConfig], DepthBackend]


def run_depth(cfg: PipelineConfig, *, restart: bool = False, limit: int | None = None,
              backend_factory: BackendFactory | None = None,
              log: Callable[[str], None] = print) -> dict:
    """Jalankan stage [2c]. Return ringkasan run."""
    set_offline()  # sebelum transformers/huggingface_hub di-import (import malas di backend)
    if limit is not None and limit < 1:
        raise DepthError(f"--limit harus ≥ 1, dapat {limit}")
    clip = load_clip(cfg.paths.work_dir)
    if not restart:  # identitas dicek SEBELUM require_small / resolve_revision / load backend; --limit ikut dicek
        _require_same_clip(clip)
    require_small(cfg.depth.model_id)
    resolve_revision(cfg)
    selected = clip.names[:limit] if limit else clip.names
    require_frames(clip.work_dir, selected)

    backend = (backend_factory or TorchDepthBackend)(cfg)
    info = backend.describe(clip.height, clip.width)
    require_small(info.model_id)

    if restart:
        restart_outputs(clip)
    manifest = build_manifest(info, clip)
    if clip.manifest_path.is_file():
        old = json.loads(clip.manifest_path.read_text(encoding="utf-8"))
        diff = manifest_diff(old, manifest)
        if diff:
            raise DepthError("output [2c] di " + str(clip.depth_dir) + " dibuat dengan setelan berbeda:\n  " +
                             "\n  ".join(diff) +
                             f"\nJalankan {stage_cmd(clip.work_dir)} --restart --yes untuk menghapus output "
                             f"lama dan mulai dari awal.")
    elif _has_outputs(clip):
        raise DepthError(f"{clip.depth_dir} berisi output tanpa {MANIFEST_FILENAME} — asal output tidak "
                         f"diketahui. Jalankan {stage_cmd(clip.work_dir)} --restart --yes.")
    clean_tmp(clip.depth_dir)

    todo = [(i, n) for i, n in zip(clip.indices[:len(selected)], selected) if not clip.frame_valid(n)]
    n_skip = len(selected) - len(todo)
    log(f"[2c] depth ({info.model_id}@{info.revision[:8]}, {info.precision}, input "
        f"{info.input_size['height']}×{info.input_size['width']}): {len(selected)} frame dipilih, "
        f"{n_skip} valid dilewati, {len(todo)} diproses")

    run = {"selected": len(selected), "skipped": n_skip, "processed": 0, "nonfinite_frames": 0,
           "frames": [], "open": None}
    if todo:
        ensure_dir(clip.depth_dir)
        if not clip.manifest_path.is_file():
            write_json_atomic(clip.manifest_path, manifest)
        _process(clip, backend, info, todo, run, log)
    return run


def _process(clip: Clip, backend: DepthBackend, info: ModelInfo, todo: list[tuple[int, str]],
             run: dict, log: Callable[[str], None]) -> None:
    t_run = time.perf_counter()
    try:
        opened = backend.open()
    except BackendOOM as e:
        append_jsonl(clip.frames_log, {"event": "oom", "frame": None, "stage": "load",
                                       "time_utc": utc_now(), **e.vram})
        raise DepthOOMError(f"OOM saat memuat Depth Anything V2 Small: {e.vram}") from None
    run["open"] = opened
    append_jsonl(clip.frames_log, {"event": "run_start", "time_utc": utc_now(), "model_id": info.model_id,
                                   "revision": info.revision, "n_todo": len(todo), **(opened or {})})
    try:
        for k, (index, name) in enumerate(todo, 1):
            t0 = time.perf_counter()
            rgb = read_rgb(clip.work_dir / FRAMES_DIRNAME / name)
            if rgb.shape[:2] != (clip.height, clip.width):
                raise DepthError(f"{name}: ukuran {rgb.shape[1]}×{rgb.shape[0]} ≠ meta.json "
                                 f"{clip.width}×{clip.height} — jalankan ulang ingest")
            t1 = time.perf_counter()
            try:
                disparity, stats = backend.infer(rgb)
            except BackendOOM as e:
                append_jsonl(clip.frames_log, {"event": "oom", "frame": name, "index": index,
                                               "time_utc": utc_now(), **e.vram})
                raise DepthOOMError(
                    f"OOM di frame {name} ({k}/{len(todo)}): {e.vram}. Frame sebelumnya tersimpan; jalankan "
                    f"ulang untuk melanjutkan (tutup aplikasi lain yang memakai GPU).") from None
            if disparity.shape != (clip.height, clip.width):
                raise DepthError(f"{name}: shape disparity {disparity.shape} ≠ {(clip.height, clip.width)}")
            t2 = time.perf_counter()
            y, finite, n_bad = to_output(disparity)
            dstats = disparity_stats(y)
            t3 = time.perf_counter()
            write_depth(clip.depth_path(name), y)
            t4 = time.perf_counter()
            rec = {"event": "frame", "frame": name, "index": index, "time_utc": utc_now(),
                   "read_s": round(t1 - t0, 4), "infer_s": round(t2 - t1, 4), "post_s": round(t3 - t2, 4),
                   "write_s": round(t4 - t3, 4), "total_s": round(t4 - t0, 4), **stats,
                   "finite": finite, "n_nonfinite": n_bad, "disparity": dstats}
            append_jsonl(clip.frames_log, rec)
            run["frames"].append(rec)
            run["processed"] += 1
            run["nonfinite_frames"] += not finite
            log(f"  [{k}/{len(todo)}] {name} {rec['total_s']:.3f} s, peak reserved "
                f"{stats.get('peak_reserved_mib', '-')} MiB, disparity {dstats['min']:.3g}–{dstats['max']:.3g}"
                + ("" if finite else f", NaN/inf {n_bad} px → 0!"))
    finally:
        backend.close()
        append_jsonl(clip.frames_log, {"event": "run_end", "time_utc": utc_now(),
                                       "n_done": run["processed"], "wall_s": round(time.perf_counter() - t_run, 2)})


# ── Download (online, terpisah dari run) ───────────
def download(cfg: PipelineConfig, *, log: Callable[[str], None] = print) -> str:
    """Unduh checkpoint + model card DA-V2 Small ke cache HF, cek lisensi. Return commit hash snapshot."""
    model_id, rev = cfg.depth.model_id, cfg.depth.revision
    require_small(model_id)
    if rev is not None and not COMMIT_HASH_RE.fullmatch(rev):
        raise DepthError(f"depth.revision ({rev!r}) harus commit hash 40 karakter hex")
    if hf_offline_active():
        raise DepthError("HF_HUB_OFFLINE aktif di environment — --download butuh online; hapus variabel itu")
    from huggingface_hub import snapshot_download

    path = Path(snapshot_download(model_id, revision=rev or DOWNLOAD_REVISION_FALLBACK,
                                  allow_patterns=list(REQUIRED_FILES)))
    lic = require_license((path / MODEL_CARD).read_text(encoding="utf-8", errors="replace"), model_id)
    commit = path.name
    log(f"{model_id} snapshot {commit} (lisensi {lic}) → {path}")
    if rev is None:
        log(f"revision belum di-pin — isi depth.revision: \"{commit}\" di configs/default.yaml")
    return commit


# ── Entry point stage (dipanggil cli.py, T-104b) ───
def main(argv: list[str] | None = None, backend_factory: BackendFactory | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m rotoscope.depth",
                                description="Stage [2c]: frames → depth/*.npy (Depth Anything V2 Small)")
    p.add_argument("--config", type=Path, default=None,
                   help=f"YAML pipeline (default: {DEFAULT_CONFIG.as_posix()} kalau ada, selain itu default kode)")
    add_work_dir_arg(p)
    p.add_argument("--restart", action="store_true", help="hapus output [2c] lama (depth/) lalu mulai dari awal")
    p.add_argument("--limit", type=int, default=None, help="hanya N frame pertama")
    p.add_argument("--download", action="store_true", help="unduh checkpoint + model card (online) lalu keluar")
    p.add_argument("--adopt", action="store_true",
                   help="catat identitas klip saat ini ke manifest [2c] yang ada (tanpa inferensi) lalu keluar; "
                        "pernyataan Anda bahwa output itu milik klip ini")
    args = p.parse_args(argv)
    reconfigure_stdio()  # stdout cp1252 (diarahkan ke file di Windows) tidak boleh membuat print crash

    def log(msg: str) -> None:
        print(msg, flush=True)  # progress tetap terlihat walau stdout diarahkan ke file

    if args.download and (args.restart or args.limit is not None):
        p.error("--download tidak bisa digabung dengan --restart/--limit")

    try:
        if args.adopt and (args.restart or args.limit is not None or args.download):
            raise DepthError("--adopt tidak bisa digabung dengan --restart/--limit/--download")
        path = args.config if args.config is not None else (DEFAULT_CONFIG if DEFAULT_CONFIG.is_file() else None)
        cfg = load_pipeline(path, overrides=work_dir_overrides(args.work_dir))
        if args.adopt:
            t0 = time.perf_counter()
            adopt_depth(cfg, log=log)
            log(f"selesai: --adopt ({time.perf_counter() - t0:.2f} s)")
        elif args.download:
            download(cfg, log=log)
        else:
            t0 = time.perf_counter()
            run = run_depth(cfg, restart=args.restart, limit=args.limit, backend_factory=backend_factory, log=log)
            log(f"selesai: {run['processed']} diproses, {run['skipped']} dilewati, NaN/inf "
                f"{run['nonfinite_frames']} frame ({time.perf_counter() - t0:.1f} s)")
    except DepthOOMError as e:
        print(f"ERROR (OOM): {e}", file=sys.stderr)
        return EXIT_OOM
    except (DepthError, ConfigError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return EXIT_PRECONDITION
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
