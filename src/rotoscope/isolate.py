"""Stage [1b] isolate (T-605, D-011): frames/ + target/points.json → isolate/labels/frame_%05d.png + isolate/frames.jsonl
+ isolate/manifest.json (SAM 2.1 hiera tiny lewat `transformers`). Kontrak lengkap: docs/01 [1b].

OPT-IN: tanpa `target/points.json` → dilewati (cli `run` tidak membuat subprocess; tidak ada berkas yang ditulis). Bila
`isolate/` lama masih ada, folder itu DIHAPUS dengan pesan (keputusan Rio: T-606 tidak boleh memakai masker usang). [1b] hanya
MENGHASILKAN masker; pemakainya = stage [3] (T-606, D-012, `target_mask.py`: hanya bila kunci di manifest ini = titik + frame
sekarang).

Alur run:
  1. titik divalidasi (`pick.load_points` + `check_doc_matches_clip`; setiap frame klik harus `ok`, id target ≤ 255);
  2. kunci basi = byte `points.json` + sidik piksel SEMUA frame + model / revision + precision + `mask_threshold` + `cut_diff`
     + versi torch / transformers + kontrak. Hasil lama dengan kunci + identitas klip sama dan PNG lengkap = up-to-date →
     dilewati tanpa torch / GPU;
  3. adegan = cut [3] APA ADANYA (`stabilize.frame_thumb` / `cut_scores` / `detect_cuts` / `shot_bounds`, ambang
     `stabilize.temporal.cut_diff`); adegan tanpa klik = `tanpa_prompt` (tanpa PNG);
  4. satu sesi SAM per (adegan berklik, target): frame klik target itu (urut naik) dimasukkan lalu LANGSUNG dihitung (frame
     kondisi), lalu maju dari klik paling awal sampai akhir adegan, lalu mundur sampai awal adegan; tiap frame tepat sekali;
  5. target k bila logit_k > `isolate.mask_threshold` (float32); tumpang tindih → logit terbesar, seri → id terkecil →
     PNG uint8 per frame di adegan berklik; status `target` / `kosong` / `tanpa_prompt` di frames.jsonl; manifest terakhir;
  6. semua ditulis ke `isolate.partial/` lalu ditukar dengan `isolate/` (terputus → `isolate/` lama utuh; diulang per KLIP,
     bukan per frame, karena memori SAM berantai antar frame).

`--yes` WAJIB setiap kali GPU dipakai (hitung baru maupun basi; keputusan Rio setelah Tahap 1): tanpa `--yes` → berhenti dengan
pesan (exit 1), tidak ada yang dihitung / dihapus. `frames: []` → hasil tanpa GPU (semua `tanpa_prompt`). Tanpa `--limit`:
masker selalu seluruh klip.

Dua lapis (pola depth.py): backend model (`TorchSamBackend` / `TorchEngine`) — satu-satunya bagian yang meng-import torch /
transformers; logika stage (`run_isolate`, `track_job`, `compose_labels`, kunci basi, …) — numpy saja, diuji dengan backend
palsu lewat `backend_factory`. Urutan pemanggilan sesi (tambah input → hitung frame kondisi → maju → mundur) ada di `track_job`
(logika murni), bukan di backend.

CLI final: python -m rotoscope isolate <video> [--config PATH] [--yes] [--restart] [--dry-run]; di `run` = subprocess setelah
[1] ingest (hanya bila `points.json` ada). Modul:
    python -m rotoscope.isolate [--config PATH] [--work-dir DIR] [--yes] [--restart] [--dry-run]
    python -m rotoscope.isolate --download        (lewat cli: python -m rotoscope download)
Exit code: 0 sukses / dilewati / dry-run, 1 prasyarat gagal (termasuk butuh --yes), 3 OOM, 130 dihentikan.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import io
import json
import os
import shutil
import sys
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from rotoscope import depth as dep
from rotoscope import pick
from rotoscope import stabilize as stb
from rotoscope.config import ConfigError, PipelineConfig, ensure_dir, load_pipeline
from rotoscope.ingest import FRAME_PATTERN, FRAMES_DIRNAME
from rotoscope.stage_common import (
    CLIP_KEY, COMMIT_HASH_RE, DOWNLOAD_REVISION_FALLBACK, EXIT_INTERRUPTED, EXIT_OK, EXIT_OOM, EXIT_PRECONDITION, MIB,
    REPLACE_DELAY_S, REPLACE_RETRIES, BackendOOM, StageError, StageInterrupted, StageOOMError, add_work_dir_arg,
    close_quietly, interrupted_line, cached_file, cli_cmd, clip_identity,
    ensure_cached, hf_offline_active, is_oom, load_frame_list, read_rgb, reconfigure_stdio, require_frames, set_message_config,
    set_offline, utc_now, vram_state, work_dir_overrides, write_bytes_atomic, write_json_atomic,
)

EXIT_INTERRUPTED = 130

# ── Layout output (docs/01 [1b]) ───────────────────
ISOLATE_DIRNAME = "isolate"
PARTIAL_DIRNAME = "isolate.partial"     # run yang sedang berjalan; ditukar dengan isolate/ saat selesai
OLD_DIRNAME = "isolate.old"             # isolate/ lama sesaat sebelum dihapus
LABELS_DIRNAME = "labels"
TMP_DIRNAME = "_tmp"                    # masker + logit per target per frame, hanya selama satu adegan
MANIFEST_FILENAME = "manifest.json"
FRAMES_LOG_FILENAME = "frames.jsonl"
FORMAT = "rotoscope-isolate/1"
CONTRACT = "T-605"
STATUS_TARGET, STATUS_EMPTY, STATUS_NO_PROMPT = "target", "kosong", "tanpa_prompt"
KIND_PROMPT, KIND_FWD, KIND_REV = "klik", "maju", "mundur"
MAX_TARGET_ID = 255                     # peta label uint8
LIBS = ("torch", "transformers")
MODE_DOC = ("offline; satu sesi per (adegan, target); frame klik dimasukkan dulu (urut naik, langsung dihitung), lalu maju "
            "dari klik paling awal sampai akhir adegan, lalu mundur sampai awal adegan")
OUTPUT_INFO = {
    "labels": "labels/frame_%05d.png: uint8 1 kanal, 0 = bukan target, k = id target; HANYA frame di adegan berklik "
              "(status target / kosong)",
    "rule": "target k bila logit_k > mask_threshold (float32); tumpang tindih -> logit terbesar, seri -> id terkecil",
    "status": "frames.jsonl: target | kosong | tanpa_prompt (tanpa PNG)",
}
# Hanya untuk pesan (bukan parameter): p50 fp32 terukur T-602 (GTX 1650 Ti), muat model + import ±.
SECONDS_PER_FRAME_ESTIMATE = 0.67
LOAD_SECONDS_ESTIMATE = 15.0
LOG_EVERY = 10

# ── Checkpoint HF + lisensi ────────────────────────
MODEL_CARD = "README.md"
REQUIRED_FILES = ("config.json", "preprocessor_config.json", "processor_config.json", "video_preprocessor_config.json",
                  "model.safetensors", MODEL_CARD)
REQUIRED_LICENSE = "apache-2.0"
DOWNLOAD_CMD = "python -m rotoscope download"

DEFAULT_CONFIG = Path("configs") / "default.yaml"

FrameOut = tuple[np.ndarray, float]     # (logit float32 (H, W) ukuran frame kerja, logit skor objek)


def stage_cmd(work_dir: Path | None = None) -> str:
    return cli_cmd("isolate", work_dir)


# ── Layout klip ────────────────────────────────────
@dataclass(frozen=True)
class Clip:
    work_dir: Path
    names: tuple[str, ...]
    indices: tuple[int, ...]
    width: int
    height: int

    @property
    def out_dir(self) -> Path:
        return self.work_dir / ISOLATE_DIRNAME

    @property
    def partial_dir(self) -> Path:
        return self.work_dir / PARTIAL_DIRNAME

    @property
    def old_dir(self) -> Path:
        return self.work_dir / OLD_DIRNAME

    @property
    def points_path(self) -> Path:
        return points_path(self.work_dir)

    def frame_path(self, index: int) -> Path:
        return self.work_dir / FRAMES_DIRNAME / (FRAME_PATTERN % index)


def load_clip(work_dir: Path) -> Clip:
    return Clip(work_dir, *load_frame_list(work_dir))


def points_path(work_dir: Path) -> Path:
    return work_dir / pick.TARGET_DIRNAME / pick.POINTS_FILENAME


def points_present(work_dir: Path) -> bool:
    """Opt-in [1b]: `run` hanya membuat subprocess bila berkas titik ada (cek murah, tanpa membaca isinya)."""
    return points_path(work_dir).is_file()


def label_path(d: Path, index: int) -> Path:
    return d / LABELS_DIRNAME / (FRAME_PATTERN % index)


# ── Prompt + job ───────────────────────────────────
@dataclass(frozen=True)
class Prompt:
    """Klik satu target di satu frame. Koordinat = piksel frame kerja apa adanya (konvensi indeks piksel = konvensi klik SAM)."""
    index: int
    points: tuple[tuple[float, float], ...]
    labels: tuple[int, ...]
    box: tuple[float, float, float, float] | None


@dataclass(frozen=True)
class Job:
    """Satu sesi SAM: satu target di satu adegan [lo, hi] (indeks frame inklusif); prompts urut naik."""
    shot: tuple[int, int]
    target: int
    prompts: tuple[Prompt, ...]

    @property
    def start(self) -> int:
        return self.prompts[0].index


def shot_of(index: int, shots: Sequence[tuple[int, int]]) -> tuple[int, int] | None:
    return next((s for s in shots if s[0] <= index <= s[1]), None)


def build_jobs(doc: dict, shots: Sequence[tuple[int, int]]) -> list[Job]:
    """points.json → job per (adegan, target), urut (awal adegan, id). Klik di adegan lain = job lain."""
    groups: dict[tuple[tuple[int, int], int], list[Prompt]] = {}
    for e in doc["frames"]:
        f = int(e["frame_index"])
        shot = shot_of(f, shots)
        if shot is None:
            raise StageError(f"frame klik {f} di luar klip — pick ulang")
        for t in e["targets"]:
            box = t.get("box")
            p = Prompt(f, tuple((float(q["x"]), float(q["y"])) for q in t["points"]),
                       tuple(int(q["label"]) for q in t["points"]),
                       tuple(float(v) for v in box) if box is not None else None)
            groups.setdefault((shot, int(t["id"])), []).append(p)
    return [Job(shot, tid, tuple(sorted(ps, key=lambda p: p.index))) for (shot, tid), ps in sorted(groups.items())]


# ── Adegan (= cut [3], fungsi + ambang yang sama) ──
def detect_scenes(clip: Clip, cut_diff: float,
                  thumbs: Callable[[Path], np.ndarray] = stb.frame_thumb) -> tuple[list[int], list[tuple[int, int]]]:
    """(indeks frame yang punya cut SEBELUM-nya, adegan [lo, hi] urut). cut_diff 0 → satu adegan."""
    scores = stb.cut_scores([thumbs(clip.frame_path(i)) for i in clip.indices])
    cut_before = stb.detect_cuts(scores, cut_diff)
    start, end = stb.shot_bounds(cut_before)
    shots = sorted({(clip.indices[start[k]], clip.indices[end[k]]) for k in range(len(clip.indices))})
    return [clip.indices[k] for k in range(len(clip.indices)) if cut_before[k]], shots


# ── Sesi SAM: urutan pemanggilan (logika murni) ────
class Engine(Protocol):
    """Satu sesi SAM untuk satu job; indeks = posisi LOKAL di adegan (0 = frame pertama adegan)."""
    def add_prompt(self, k: int, prompt: Prompt) -> None: ...
    def run_frame(self, k: int) -> FrameOut: ...
    def propagate(self, start: int, reverse: bool) -> Iterator[tuple[int, FrameOut]]: ...
    def release(self) -> None: ...


def track_job(engine: Engine, job: Job) -> Iterator[tuple[int, str, FrameOut]]:
    """Yield (indeks frame GLOBAL, jenis, keluaran) untuk tiap frame adegan TEPAT sekali.

    1. Tiap frame klik (urut naik): input ditambahkan lalu frame itu LANGSUNG dihitung — di `transformers` 5.17.0 penanda
       "objek punya input baru" ditimpa tiap panggilan dan dikosongkan saat frame kondisi dihitung, jadi input frame kedua
       yang belum dihitung akan diabaikan diam-diam.
    2. Maju dari klik paling awal sampai akhir adegan (frame kondisi lain dipakai dari cache = koreksi).
    3. Mundur dari klik paling awal − 1 sampai awal adegan (klik paling awal = awal adegan → tanpa mundur).
    """
    lo, hi = job.shot
    seen: set[int] = set()
    for p in job.prompts:
        k = p.index - lo
        engine.add_prompt(k, p)
        out = engine.run_frame(k)
        seen.add(k)
        yield p.index, KIND_PROMPT, out
    start = job.start - lo
    for k, out in engine.propagate(start, reverse=False):
        if k not in seen:
            seen.add(k)
            yield lo + k, KIND_FWD, out
    if start > 0:
        for k, out in engine.propagate(start, reverse=True):
            if k not in seen:
                seen.add(k)
                yield lo + k, KIND_REV, out
    missing = sorted(set(range(hi - lo + 1)) - seen)
    if missing:
        raise StageError(f"[1b] sesi target {job.target} adegan f{lo}–f{hi}: {len(missing)} frame tidak dihasilkan "
                         f"(mis. f{lo + missing[0]}) — galat internal backend")


# ── Masker → peta label ────────────────────────────
def target_mask(logits: np.ndarray, threshold: float) -> np.ndarray:
    """Logit float32 > ambang (ketat); NaN → bukan target."""
    return np.asarray(logits, np.float32) > np.float32(threshold)


def compose_labels(per_target: Mapping[int, tuple[np.ndarray, np.ndarray]], shape: tuple[int, int]) -> np.ndarray:
    """{id: (masker bool, logit)} → uint8 (H, W): 0 = bukan target, id = target. Tumpang tindih → logit terbesar; seri → id
    terkecil (id diproses naik, pembanding ketat)."""
    label = np.zeros(shape, np.uint8)
    best = np.full(shape, -np.inf, np.float32)
    for tid in sorted(per_target):
        if not 1 <= tid <= MAX_TARGET_ID:
            raise StageError(f"id target {tid} di luar 1–{MAX_TARGET_ID} (peta label uint8)")
        mask, logit = per_target[tid]
        lg = np.asarray(logit, np.float32)
        take = mask & (lg > best)
        label[take] = tid
        best[take] = lg[take]
    return label


def frame_status(label: np.ndarray | None) -> str:
    if label is None:
        return STATUS_NO_PROMPT
    return STATUS_TARGET if bool(label.any()) else STATUS_EMPTY


def write_label(path: Path, label: np.ndarray) -> None:
    ok, buf = cv2.imencode(".png", label)
    if not ok:
        raise StageError(f"gagal meng-encode peta label {path.name}")
    write_bytes_atomic(path, buf.tobytes())


def read_label(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    m = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_UNCHANGED)
    return m if m is not None and m.dtype == np.uint8 and m.ndim == 2 else None


def _tmp_path(partial: Path, tid: int, index: int) -> Path:
    return partial / TMP_DIRNAME / f"t{tid:03d}" / (Path(FRAME_PATTERN % index).stem + ".npz")


def save_target_frame(path: Path, mask: np.ndarray, logits: np.ndarray) -> None:
    """Masker (dibandingkan di float32) + logit float16 (hanya untuk tumpang tindih) satu target satu frame."""
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    with np.errstate(over="ignore", invalid="ignore"):
        lg16 = np.asarray(logits, np.float32).astype(np.float16)
    np.savez(buf, mask=np.packbits(mask, axis=-1), logit=lg16)
    path.write_bytes(buf.getvalue())


def load_target_frame(path: Path, width: int) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return np.unpackbits(z["mask"], axis=-1, count=width).astype(bool), z["logit"]


def ranges(indices: Sequence[int]) -> list[list[int]]:
    """[3, 4, 5, 9] → [[3, 5], [9, 9]]."""
    out: list[list[int]] = []
    for i in sorted(indices):
        if out and i == out[-1][1] + 1:
            out[-1][1] = i
        else:
            out.append([i, i])
    return out


# ── Kunci basi ─────────────────────────────────────
def frames_fingerprint(clip: Clip) -> tuple[str, dict[int, str]]:
    """Sidik piksel (`pick.frame_pixel_sha256`) SEMUA frame → (sha256 gabungan urut indeks, {indeks: sidik})."""
    per = {i: pick.frame_pixel_sha256(clip.frame_path(i)) for i in clip.indices}
    agg = hashlib.sha256("".join(per[i] + "\n" for i in clip.indices).encode("ascii")).hexdigest()
    return agg, per


def lib_versions() -> dict[str, str | None]:
    """Versi paket TANPA meng-import-nya (importlib.metadata)."""
    out: dict[str, str | None] = {}
    for name in LIBS:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def build_key(cfg: PipelineConfig, points_sha256: str, frames_sha256: str) -> dict:
    """Semua yang menentukan hasil [1b]. BUKAN kunci: vram_min_free_mib / ram_min_free_mib (tidak mengubah hasil)."""
    i = cfg.isolate
    return {"contract": CONTRACT, "points_sha256": points_sha256, "frames_pixel_sha256": frames_sha256,
            "model": {"model_id": i.model_id, "revision": i.revision}, "precision": i.precision,
            "params": {"mask_threshold": i.mask_threshold, "cut_diff": cfg.stabilize.temporal.cut_diff},
            "libs": lib_versions()}


def canonical_sha256(obj) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _flat(d, prefix: str = "") -> dict[str, object]:
    if not isinstance(d, dict):
        return {prefix: d}
    out: dict[str, object] = {}
    for k, v in d.items():
        out.update(_flat(v, f"{prefix}.{k}" if prefix else str(k)))
    return out


def key_diff(old: dict | None, new: dict) -> list[str]:
    a, b = _flat(old or {}), _flat(new)
    short = lambda v: v[:12] if isinstance(v, str) and len(v) == 64 else v  # noqa: E731 — sha256 dipendekkan di pesan
    return [f"{k}: {short(a.get(k))!r} → {short(b.get(k))!r}" for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]


def read_manifest(path: Path) -> dict | None:
    try:
        m = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return m if isinstance(m, dict) else None


def expected_labels(manifest: dict) -> set[int]:
    """Frame yang WAJIB punya PNG = gabungan adegan semua job di manifest."""
    out: set[int] = set()
    for j in manifest.get("jobs", []):
        lo, hi = j["shot"]
        out.update(range(int(lo), int(hi) + 1))
    return out


def existing_state(clip: Clip, key: dict, identity: dict) -> tuple[str, list[str]]:
    """("ok", []) = up-to-date | ("basi", alasan) | ("tidak_ada", [])."""
    d = clip.out_dir
    if not d.exists():
        return "tidak_ada", []
    m = read_manifest(d / MANIFEST_FILENAME)
    if m is None:
        return "basi", [f"{MANIFEST_FILENAME} tidak ada / rusak (run terputus sebelum T-605 selesai?)"]
    reasons = []
    if m.get("format") != FORMAT:
        reasons.append(f"format: {m.get('format')!r} → {FORMAT!r}")
    if m.get(CLIP_KEY) != identity:
        reasons.append(f"clip (meta.json): {str((m.get(CLIP_KEY) or {}).get('meta_sha256'))[:12]} → "
                       f"{identity['meta_sha256'][:12]}")
    if m.get("frame_count") != len(clip.indices):
        reasons.append(f"frame_count: {m.get('frame_count')} → {len(clip.indices)}")
    reasons += [f"key.{r}" for r in key_diff(m.get("key"), key)]
    if reasons:
        return "basi", reasons
    try:
        expected = expected_labels(m)
    except (KeyError, TypeError, ValueError):
        return "basi", ["manifest.jobs rusak"]
    problems = output_problems(clip, expected)
    return ("basi", problems) if problems else ("ok", [])


def output_problems(clip: Clip, expected: set[int]) -> list[str]:
    """PNG wajib ada + terbaca + uint8 ukuran frame; tanpa PNG lebih; frames.jsonl = satu baris per frame klip."""
    d = clip.out_dir
    out = []
    bad = [i for i in sorted(expected) if (lb := read_label(label_path(d, i))) is None or lb.shape != (clip.height, clip.width)]
    if bad:
        out.append(f"labels/: {len(bad)} PNG hilang / rusak (mis. {FRAME_PATTERN % bad[0]})")
    want = {FRAME_PATTERN % i for i in expected}
    extra = sorted(p.name for p in (d / LABELS_DIRNAME).glob("*.png") if p.name not in want) if (d / LABELS_DIRNAME).is_dir() else []
    if extra:
        out.append(f"labels/: {len(extra)} PNG di luar adegan berklik (mis. {extra[0]})")
    rows = read_frame_rows(d / FRAMES_LOG_FILENAME)
    if [r.get("frame") for r in rows] != [FRAME_PATTERN % i for i in clip.indices]:
        out.append(f"{FRAMES_LOG_FILENAME}: baris frame tidak lengkap")
    return out


def read_frame_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            return []
        if isinstance(r, dict) and r.get("event") == "frame":
            rows.append(r)
    return rows


# ── Titik ──────────────────────────────────────────
def load_valid_points(clip: Clip, per_frame_sha: Mapping[int, str]) -> dict:
    """points.json valid untuk klip + frame SEKARANG; galat = StageError dengan perintah perbaikan."""
    try:
        doc = pick.load_points(clip.points_path)
        pick.check_doc_matches_clip(doc, clip_identity(clip.work_dir), clip.width, clip.height, clip.points_path,
                                    cli_cmd("pick", clip.work_dir))
    except pick.PointsError as e:
        raise StageError(f"{e} — perbaiki dengan {cli_cmd('pick', clip.work_dir)}, atau pindahkan berkas itu bila isolasi "
                         f"tidak dipakai") from None
    except pick.PickError as e:
        raise StageError(str(e)) from None
    big = sorted({t["id"] for e in doc["frames"] for t in e["targets"] if t["id"] > MAX_TARGET_ID})
    if big:
        raise StageError(f"{clip.points_path}: id target {big} > {MAX_TARGET_ID} (peta label uint8) — pick ulang dengan id ≤ "
                         f"{MAX_TARGET_ID}")
    cache = dict(per_frame_sha)
    statuses = pick.frame_statuses(doc, clip.work_dir, set(clip.indices), cache)
    bad = {f: s for f, s in statuses.items() if s != pick.STATUS_OK}
    if bad:
        video = cli_cmd("pick", clip.work_dir)
        frames = sorted(bad)
        raise StageError(
            f"titik frame {', '.join(map(str, frames))} tidak bisa dipakai ({', '.join(f'f{f}: {s}' for f, s in sorted(bad.items()))}"
            f"; basi = isi frame berubah sejak diklik (mis. ingest diulang dengan crop / trim / video lain); hilang = di luar klip / "
            f"tidak terbaca). "
            f"Pick ulang: {video} --frame {frames[0]}" + (f" (lalu {', '.join(f'--frame {f}' for f in frames[1:])})"
                                                        if len(frames) > 1 else "")
            + f", atau pindahkan {clip.points_path} bila isolasi tidak dipakai. Tidak ada yang dihitung.")
    return doc


# ── Tanpa titik / bersih-bersih ────────────────────
def _rmtree(p: Path) -> None:
    if p.exists():
        shutil.rmtree(p)


def remove_outputs(work_dir: Path) -> list[str]:
    """Hapus isolate/ + sisa isolate.partial/ / isolate.old/; daftar nama folder yang dihapus."""
    gone = []
    for name in (ISOLATE_DIRNAME, PARTIAL_DIRNAME, OLD_DIRNAME):
        if (work_dir / name).exists():
            _rmtree(work_dir / name)
            gone.append(name + "/")
    return gone


def skip_without_points(work_dir: Path, *, log: Callable[[str], None] = print, dry_run: bool = False) -> dict:
    """Tanpa target/points.json: [1b] dilewati. isolate/ lama DIHAPUS (keputusan Rio: T-606 tidak boleh memakai masker
    usang). Tanpa isolate/ lama → tidak ada operasi berkas apa pun."""
    present = [n + "/" for n in (ISOLATE_DIRNAME, PARTIAL_DIRNAME, OLD_DIRNAME) if (work_dir / n).exists()]
    if not present:
        log(f"[1b] isolate: {pick.TARGET_DIRNAME}/{pick.POINTS_FILENAME} tidak ada — dilewati (opt-in; tanpa GPU, tanpa berkas)")
        return {"status": "dilewati", "removed": []}
    if dry_run:
        log(f"[1b] isolate --dry-run: {pick.TARGET_DIRNAME}/{pick.POINTS_FILENAME} tidak ada — {', '.join(present)} AKAN "
            f"DIHAPUS (masker dari titik lama tidak berlaku)")
        return {"status": "dry-run", "removed": []}
    gone = remove_outputs(work_dir)
    log(f"[1b] isolate: {pick.TARGET_DIRNAME}/{pick.POINTS_FILENAME} tidak ada — {', '.join(gone)} lama DIHAPUS (masker dari "
        f"titik lama tidak berlaku; T-606 tidak boleh memakainya)")
    return {"status": "dilewati", "removed": gone}


def clean_leftovers(clip: Clip, log: Callable[[str], None]) -> None:
    """isolate.partial/ (run terputus) dan isolate.old/ bukan hasil — dihapus di awal run."""
    for d in (clip.partial_dir, clip.old_dir):
        if d.exists():
            _rmtree(d)
            log(f"  sisa run sebelumnya dihapus: {d.name}/")


def _replace_dir(src: Path, dst: Path) -> None:
    last: OSError | None = None
    for attempt in range(REPLACE_RETRIES):
        try:
            os.replace(src, dst)
            return
        except PermissionError as e:                 # WinError 5/32: dikunci antivirus / indexer sesaat
            last = e
            if attempt + 1 < REPLACE_RETRIES:
                time.sleep(REPLACE_DELAY_S)
    raise StageError(f"gagal mengganti {dst} dengan {src} ({last}) — tutup aplikasi yang membuka berkas di sana lalu "
                     f"jalankan ulang")


def promote(clip: Clip) -> None:
    """isolate.partial/ (lengkap) → isolate/; isolate/ lama → isolate.old/ → dihapus."""
    _rmtree(clip.old_dir)
    if clip.out_dir.exists():
        _replace_dir(clip.out_dir, clip.old_dir)
    _replace_dir(clip.partial_dir, clip.out_dir)
    _rmtree(clip.old_dir)


# ── RAM / VRAM ─────────────────────────────────────
class _MemStatus(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


class _ProcMem(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong), ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t)]


def ram_free_mib() -> float | None:
    """RAM fisik bebas (GlobalMemoryStatusEx); None bila tidak bisa dibaca (bukan Windows)."""
    try:
        k32 = ctypes.WinDLL("kernel32")
        st = _MemStatus()
        st.dwLength = ctypes.sizeof(st)
        if not k32.GlobalMemoryStatusEx(ctypes.byref(st)):
            return None
        return round(st.ullAvailPhys / MIB, 1)
    except (AttributeError, OSError):
        return None


def process_peak_ram_mib() -> float | None:
    """Puncak working set proses ini (GetProcessMemoryInfo); hanya diagnosis."""
    try:
        k32, psapi = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(_ProcMem), ctypes.c_ulong]
        pmc = _ProcMem()
        pmc.cb = ctypes.sizeof(pmc)
        if not psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
            return None
        return round(pmc.PeakWorkingSetSize / MIB, 1)
    except (AttributeError, OSError):
        return None


def check_ram(free_mib: float | None, need_mib: int) -> None:
    if free_mib is None or free_mib >= need_mib:
        return
    raise StageError(f"RAM bebas {free_mib:.0f} MiB < dibutuhkan {need_mib} MiB untuk [1b] SAM 2.1 tiny "
                     f"(isolate.ram_min_free_mib). Tutup aplikasi lain (browser, editor video) lalu jalankan ulang.")


def check_vram(free_mib: float, total_mib: float, need_mib: int) -> None:
    if free_mib >= need_mib:
        return
    raise StageError(f"VRAM bebas {free_mib:.0f} MiB < dibutuhkan {need_mib} MiB untuk [1b] SAM 2.1 tiny "
                     f"(total {total_mib:.0f} MiB, isolate.vram_min_free_mib). Tutup aplikasi lain yang memakai GPU "
                     f"(browser, game, editor video), dan pastikan stage GPU lain tidak sedang jalan.")


# ── Backend ────────────────────────────────────────
@dataclass(frozen=True)
class ModelInfo:
    model_id: str
    revision: str
    license: str
    precision: str
    processor: dict


class SamBackend(Protocol):
    def describe(self) -> ModelInfo: ...
    def open(self) -> dict: ...
    def session(self, frames: Sequence[Path], size: tuple[int, int], obj_id: int) -> Engine: ...
    def stats(self) -> dict: ...
    def close(self) -> None: ...


def require_model_id(model_id: str) -> None:
    from rotoscope.config import ISOLATE_MODEL_ID_PREFIX
    if not model_id.startswith(ISOLATE_MODEL_ID_PREFIX):
        raise StageError(f"isolate.model_id {model_id!r} bukan SAM 2.1 hiera ({ISOLATE_MODEL_ID_PREFIX}…) — D-011")


def resolve_revision(cfg: PipelineConfig) -> str:
    """Revision SAM wajib commit hash 40-hex (null / nama branch ditolak)."""
    rev = cfg.isolate.revision
    if rev is None:
        raise StageError(f"isolate.revision = null — run offline butuh commit hash yang di-pin. Isi di configs/default.yaml, "
                         f"atau jalankan dulu (online): {DOWNLOAD_CMD} (mencetak hash untuk di-pin)")
    if not COMMIT_HASH_RE.fullmatch(rev):
        raise StageError(f"isolate.revision ({rev!r}) harus commit hash 40 karakter hex, bukan nama branch/tag")
    return rev


def require_license(text: str, model_id: str) -> str:
    lic = dep.card_license(text)
    if lic != REQUIRED_LICENSE:
        raise StageError(f"lisensi model card {model_id} = {lic!r}, dibutuhkan {REQUIRED_LICENSE!r} (D-011)")
    return lic


def _size_dict(size) -> dict | None:
    if size is None:
        return None
    try:
        return {"height": int(size["height"]), "width": int(size["width"])}
    except (KeyError, TypeError):
        return {"height": int(getattr(size, "height")), "width": int(getattr(size, "width"))}


def _processor_info(proc) -> dict:
    vp = getattr(proc, "video_processor", None)
    num = lambda v: [float(x) for x in v] if v is not None else None  # noqa: E731
    return {"class": type(proc).__name__, "size": _size_dict(getattr(vp, "size", None)),
            "resample": int(getattr(vp, "resample", -1)), "keep_aspect_ratio": False,
            "image_mean": num(getattr(vp, "image_mean", None)), "image_std": num(getattr(vp, "image_std", None))}


class LazyFrames(Mapping):
    """`processed_frames` sesi SAM yang membaca frame dari disk SAAT DIBUTUHKAN (bukan semua frame di RAM). Praproses
    `processor(images=rgb)` = `video_processor` = `init_video_session(video=…)` (dicek T-605 Tahap 1: selisih 0,0).
    Bergantung pada atribut dalam `Sam2VideoInferenceSession.processed_frames` (`transformers` di-pin 5.17.0; test kontrak)."""

    def __init__(self, paths: Sequence[Path], processor, dtype):
        self._paths, self._processor, self._dtype = list(paths), processor, dtype

    def __len__(self) -> int:
        return len(self._paths)

    def __iter__(self):
        return iter(range(len(self._paths)))

    def __getitem__(self, k):
        if not isinstance(k, int) or not 0 <= k < len(self._paths):
            raise KeyError(k)
        rgb = read_rgb(self._paths[k])
        return self._processor(images=rgb, return_tensors="pt").pixel_values[0].to(self._dtype)


class TorchEngine:
    """Satu sesi `Sam2VideoInferenceSession` (mode offline): frame + state di CPU, inferensi di `backend.device`."""

    def __init__(self, backend: "TorchSamBackend", frames: Sequence[Path], size: tuple[int, int], obj_id: int):
        self._b, (self._h, self._w), self._obj = backend, size, obj_id
        self._s = backend._processor.init_video_session(
            video=None, inference_device=backend.device, inference_state_device="cpu", processing_device="cpu",
            video_storage_device="cpu", dtype=backend._dtype)
        self._s.processed_frames = LazyFrames(frames, backend._processor, backend._dtype)

    def _out(self, out) -> FrameOut:
        logits = self._b._processor.post_process_masks([out.pred_masks], original_sizes=[[self._h, self._w]],
                                                       binarize=False)[0]
        arr = logits.float().cpu().numpy().reshape(-1, self._h, self._w)[0]
        sc = out.object_score_logits
        score = float(sc.float().reshape(-1)[0].item()) if sc is not None else float("nan")
        return np.ascontiguousarray(arr, dtype=np.float32), score

    def _guard(self, fn):
        try:
            return fn()
        except Exception as e:
            if self._b.device == "cuda" and is_oom(self._b._torch, e):
                raise BackendOOM(vram_state(self._b._torch)) from e
            raise

    def add_prompt(self, k: int, prompt: Prompt) -> None:
        kw = {}
        if prompt.points:
            kw["input_points"] = [[[list(p) for p in prompt.points]]]
            kw["input_labels"] = [[list(prompt.labels)]]
        if prompt.box is not None:
            kw["input_boxes"] = [[list(prompt.box)]]
        self._b._processor.add_inputs_to_inference_session(inference_session=self._s, frame_idx=k, obj_ids=[self._obj],
                                                           original_size=(self._h, self._w), **kw)

    def run_frame(self, k: int) -> FrameOut:
        torch = self._b._torch

        def go():
            with torch.inference_mode():
                return self._out(self._b._model(inference_session=self._s, frame_idx=k))
        return self._guard(go)

    def propagate(self, start: int, reverse: bool) -> Iterator[tuple[int, FrameOut]]:
        torch = self._b._torch
        it = self._b._model.propagate_in_video_iterator(self._s, start_frame_idx=start, reverse=reverse)
        while True:
            def step():
                with torch.inference_mode():
                    try:
                        out = next(it)
                    except StopIteration:
                        return None
                    return int(out.frame_idx), self._out(out)
            res = self._guard(step)
            if res is None:
                return
            yield res

    def release(self) -> None:
        self._s = None
        gc.collect()
        if self._b.device == "cuda" and self._b._torch is not None:
            self._b._torch.cuda.empty_cache()


class TorchSamBackend:
    """SAM 2.1 hiera tiny lewat `transformers` 5.17.0, satu model di GPU (proses sendiri). `device` "cpu" hanya untuk test
    kontrak (tanpa cek VRAM)."""

    def __init__(self, cfg: PipelineConfig, cache_dir: str | Path | None = None, device: str = "cuda"):
        i = cfg.isolate
        self.model_id = i.model_id
        require_model_id(self.model_id)
        self.revision = resolve_revision(cfg)
        self.precision = i.precision
        self.vram_need_mib = i.vram_min_free_mib
        self.cache_dir, self.device = cache_dir, device
        self._torch = self._processor = self._model = self._dtype = None

    def _hf_kwargs(self) -> dict:
        return {"revision": self.revision, "local_files_only": True, "cache_dir": self.cache_dir}

    def describe(self) -> ModelInfo:
        set_offline()
        ensure_cached(self.model_id, self.revision, REQUIRED_FILES, DOWNLOAD_CMD, self.cache_dir)
        card = cached_file(self.model_id, MODEL_CARD, self.revision, self.cache_dir)
        lic = require_license(card.read_text(encoding="utf-8", errors="replace"), self.model_id)
        from transformers import Sam2VideoProcessor

        self._processor = Sam2VideoProcessor.from_pretrained(self.model_id, **self._hf_kwargs())
        return ModelInfo(self.model_id, self.revision, lic, self.precision, _processor_info(self._processor))

    def open(self) -> dict:
        set_offline()
        import torch

        self._torch = torch
        info: dict = {"device": self.device}
        if self.device == "cuda":
            if not torch.cuda.is_available():
                raise StageError("CUDA tidak tersedia — stage [1b] butuh GPU NVIDIA lewat PyTorch cu118. Cek driver dan "
                                 "bahwa torch terinstall versi +cu118 (bukan CPU).")
            free, total = torch.cuda.mem_get_info()
            check_vram(free / MIB, total / MIB, self.vram_need_mib)
            info.update(vram_free_before_load_mib=round(free / MIB, 1), vram_total_mib=round(total / MIB, 1),
                        gpu=torch.cuda.get_device_name(0))
        if self._processor is None:
            raise StageError("describe() harus dipanggil sebelum open()")
        self._dtype = torch.float16 if self.precision == "fp16" else torch.float32
        from transformers import Sam2VideoModel

        t0 = time.perf_counter()
        try:
            model = Sam2VideoModel.from_pretrained(self.model_id, dtype=self._dtype, **self._hf_kwargs())
            model.requires_grad_(False)
            self._model = model.to(self.device).eval()
        except Exception as e:
            if self.device == "cuda" and is_oom(torch, e):
                raise BackendOOM(vram_state(torch)) from e
            raise
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        info["load_s"] = round(time.perf_counter() - t0, 2)
        return info

    def session(self, frames: Sequence[Path], size: tuple[int, int], obj_id: int) -> TorchEngine:
        return TorchEngine(self, frames, size, obj_id)

    def stats(self) -> dict:
        if self.device != "cuda" or self._torch is None:
            return {}
        t = self._torch
        return {"max_reserved_mib": round(t.cuda.max_memory_reserved() / MIB, 1),
                "max_allocated_mib": round(t.cuda.max_memory_allocated() / MIB, 1)}

    def close(self) -> None:
        self._model = None
        gc.collect()
        if self.device == "cuda" and self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()


BackendFactory = Callable[[PipelineConfig], SamBackend]


# ── Rencana + run ──────────────────────────────────
def describe_plan(clip: Clip, cuts: list[int], shots: list[tuple[int, int]], jobs: list[Job]) -> list[str]:
    berklik = sorted({j.shot for j in jobs})
    no_prompt = [i for i in clip.indices if shot_of(i, berklik) is None]
    lines = [f"  adegan: {len(shots)} (cut di frame {', '.join(map(str, cuts)) or '-'}); berklik "
             f"{', '.join(f'f{a}–f{b}' for a, b in berklik) or '-'}"]
    for j in jobs:
        lo, hi = j.shot
        rev = f"mundur f{j.start - 1}→f{lo}" if j.start > lo else "tanpa mundur"
        lines.append(f"  job target {j.target}, adegan f{lo}–f{hi} ({hi - lo + 1} frame): klik di "
                     f"{', '.join(f'f{p.index}' for p in j.prompts)}; maju f{j.start}→f{hi}, {rev}")
    span = lambda a, b: f"f{a}–f{b}" if a != b else f"f{a}"  # noqa: E731
    rng = ", ".join(span(a, b) for a, b in ranges(no_prompt))
    lines.append(f"  tanpa_prompt: {len(no_prompt)} frame" + (f" ({rng})" if no_prompt else ""))
    for a, b in berklik:                     # adegan tanpa klik tepat sesudah adegan berklik: mungkin cut positif palsu
        nxt = shot_of(b + 1, shots)
        if nxt is not None and nxt not in berklik:
            lines.append(f"  catatan: adegan {span(*nxt)} (sesudah adegan berklik) tanpa klik → tanpa target; bila masih "
                         f"adegan yang sama (cut positif palsu), tambah klik: {cli_cmd('pick', clip.work_dir)} --frame {nxt[0]}")
    return lines


def gpu_frames(jobs: list[Job]) -> int:
    return sum(j.shot[1] - j.shot[0] + 1 for j in jobs)


def run_isolate(cfg: PipelineConfig, *, yes: bool = False, restart: bool = False, dry_run: bool = False,
                backend_factory: BackendFactory | None = None, log: Callable[[str], None] = print,
                thumbs: Callable[[Path], np.ndarray] = stb.frame_thumb,
                ram_probe: Callable[[], float | None] = ram_free_mib) -> dict:
    """Jalankan stage [1b]. Return ringkasan {"status": dilewati | up-to-date | dry-run | dihitung, …}."""
    set_offline()
    clip = load_clip(cfg.paths.work_dir)
    if restart and not yes and not dry_run:
        raise StageError(f"--restart menghapus hasil GPU [1b] — butuh --yes: {stage_cmd(clip.work_dir)} --restart --yes")
    if not clip.points_path.is_file():
        return skip_without_points(clip.work_dir, log=log, dry_run=dry_run)
    if not dry_run:
        clean_leftovers(clip, log)
    require_frames(clip.work_dir, [FRAME_PATTERN % i for i in clip.indices])
    t0 = time.perf_counter()
    frames_sha, per_frame = frames_fingerprint(clip)
    doc = load_valid_points(clip, per_frame)
    points_bytes = clip.points_path.read_bytes()
    identity = clip_identity(clip.work_dir)
    key = build_key(cfg, hashlib.sha256(points_bytes).hexdigest(), frames_sha)
    key_s = time.perf_counter() - t0
    state, reasons = existing_state(clip, key, identity)
    if state == "ok" and not restart:
        log(f"[1b] isolate: up-to-date — dilewati (tanpa GPU; kunci {canonical_sha256(key)[:12]}, {key_s:.1f} s)")
        return {"status": "up-to-date"}
    t1 = time.perf_counter()
    cuts, shots = detect_scenes(clip, cfg.stabilize.temporal.cut_diff, thumbs)
    jobs = build_jobs(doc, shots)
    scenes_s = time.perf_counter() - t1
    n_gpu = gpu_frames(jobs)
    est = n_gpu * SECONDS_PER_FRAME_ESTIMATE + (LOAD_SECONDS_ESTIMATE if jobs else 0.0)
    why = ("--restart" if restart else {"tidak_ada": "belum ada hasil", "basi": "hasil lama basi"}.get(state, state))
    plan = describe_plan(clip, cuts, shots, jobs)
    if dry_run:
        log(f"[1b] isolate --dry-run: status {'up-to-date' if state == 'ok' else why}"
            + (f" ({'; '.join(reasons)})" if reasons else ""))
        for line in plan:
            log(line)
        log(f"  GPU: {'dibutuhkan (--yes wajib)' if jobs and (state != 'ok' or restart) else 'tidak dibutuhkan'}; "
            f"{n_gpu} frame dihitung SAM ≈ {est:.0f} s (≈ {SECONDS_PER_FRAME_ESTIMATE} s/frame + muat ±{LOAD_SECONDS_ESTIMATE:.0f} s)")
        log("  dry-run: tidak ada yang dihitung / ditulis / dihapus")
        return {"status": "dry-run", "state": state, "reasons": reasons, "cuts": cuts, "shots": shots,
                "jobs": [job_json(j) for j in jobs], "gpu_frames": n_gpu, "estimate_s": round(est, 1)}
    if jobs and not yes:
        raise StageError(
            f"[1b] isolate butuh GPU ({why}" + (f": {'; '.join(reasons[:4])}" if reasons else "") + f") — {n_gpu} frame × "
            f"≈ {SECONDS_PER_FRAME_ESTIMATE} s ≈ {est / 60:.1f} mnt. Tidak ada yang dihitung / dihapus. Jalankan ulang dengan "
            f"--yes: {stage_cmd(clip.work_dir)} --yes (atau {cli_cmd('run', clip.work_dir)} --yes). Rencana: "
            f"{stage_cmd(clip.work_dir)} --dry-run")
    if state == "basi":
        log(f"PERINGATAN: isolate/ basi — dihitung ulang seluruh klip:\n  " + "\n  ".join(reasons))
    if restart and clip.out_dir.exists():
        _rmtree(clip.out_dir)
        log("[1b] --restart: isolate/ lama dihapus")
    log(f"[1b] isolate ({cfg.isolate.model_id}@{str(cfg.isolate.revision)[:8]}, {cfg.isolate.precision}): {len(jobs)} job, "
        f"{n_gpu} frame SAM, {len(clip.indices) - len({i for j in jobs for i in range(j.shot[0], j.shot[1] + 1)})} frame "
        f"tanpa_prompt")
    for line in plan:
        log(line)
    return compute(cfg, clip, key, identity, cuts, shots, jobs, per_frame, backend_factory, log, ram_probe,
                   {"key_s": round(key_s, 2), "scenes_s": round(scenes_s, 2)})


def job_json(j: Job) -> dict:
    lo, hi = j.shot
    return {"shot": [lo, hi], "target": j.target, "prompt_frames": [p.index for p in j.prompts], "start": j.start,
            "forward": [j.start, hi], "reverse": [lo, j.start - 1] if j.start > lo else None,
            "prompts": [{"frame": p.index, "n_pos": p.labels.count(1), "n_neg": p.labels.count(0),
                         "box": list(p.box) if p.box is not None else None} for p in j.prompts]}


def compute(cfg: PipelineConfig, clip: Clip, key: dict, identity: dict, cuts: list[int], shots: list[tuple[int, int]],
            jobs: list[Job], per_frame: Mapping[int, str], backend_factory: BackendFactory | None,
            log: Callable[[str], None], ram_probe: Callable[[], float | None], timings: dict) -> dict:
    """Hitung SELURUH klip ke isolate.partial/ lalu tukar dengan isolate/. Terputus → isolate/ lama utuh."""
    partial = clip.partial_dir
    _rmtree(partial)
    ensure_dir(partial / LABELS_DIRNAME)
    started, t_run = utc_now(), time.perf_counter()
    thr = cfg.isolate.mask_threshold
    size = (clip.height, clip.width)
    per_target: dict[int, dict[str, dict]] = {i: {} for i in clip.indices}
    labels: dict[int, np.ndarray] = {}
    jobs_out, infer_times, nonfinite = [], [], set()
    run_info: dict = {"started_utc": started, **timings}
    model_info = None
    if jobs:
        ram_free = ram_probe()
        run_info["ram"] = {"free_before_mib": ram_free}
        check_ram(ram_free, cfg.isolate.ram_min_free_mib)
        backend = (backend_factory or TorchSamBackend)(cfg)
        info = backend.describe()
        model_info = {"model_id": info.model_id, "revision": info.revision, "license": info.license,
                      "precision": info.precision}
        try:
            opened = backend.open()
        except BackendOOM as e:
            raise StageOOMError(f"OOM saat memuat SAM 2.1 tiny: {e.vram}") from None
        run_info["open"] = opened
        log(f"  model dimuat ({opened.get('load_s', '-')} s; {opened.get('device', '-')})")
        n_total, n_done = gpu_frames(jobs), 0
        try:
            for shot in sorted({j.shot for j in jobs}):
                lo, hi = shot
                shot_jobs = [j for j in jobs if j.shot == shot]
                frames = [clip.frame_path(i) for i in range(lo, hi + 1)]
                for job in shot_jobs:
                    engine = backend.session(frames, size, job.target)
                    t_job, n_job = time.perf_counter(), 0
                    try:
                        it = track_job(engine, job)
                        while True:
                            t0 = time.perf_counter()
                            try:
                                index, kind, (logits, score) = next(it)
                            except StopIteration:
                                break
                            dt = time.perf_counter() - t0
                            if logits.shape != size:
                                raise StageError(f"[1b] logit frame {index} berukuran {logits.shape} ≠ {size}")
                            if not np.isfinite(logits).all():
                                nonfinite.add(index)
                            mask = target_mask(logits, thr)
                            save_target_frame(_tmp_path(partial, job.target, index), mask, logits)
                            per_target[index][str(job.target)] = {"kind": kind, "area_px": int(mask.sum()),
                                                                  "score_logit": round(score, 3) if np.isfinite(score) else None}
                            infer_times.append(dt)
                            n_job += 1
                            n_done += 1
                            if n_done == 1 or n_done % LOG_EVERY == 0 or n_done == n_total:
                                log(f"  [{n_done}/{n_total}] {FRAME_PATTERN % index} target {job.target} ({kind}) {dt:.2f} s, "
                                    f"luas {int(mask.sum())} px, skor {score:.2f}")
                    except BackendOOM as e:
                        raise StageOOMError(f"OOM di [1b] (target {job.target}, adegan f{lo}–f{hi}): {e.vram}. Tidak ada "
                                            f"yang diganti; isolate/ lama utuh. Tutup aplikasi lain yang memakai GPU lalu "
                                            f"jalankan ulang (seluruh klip dihitung ulang).") from None
                    finally:
                        engine.release()
                    jobs_out.append({**job_json(job), "n_infer": n_job, "infer_s": round(time.perf_counter() - t_job, 2)})
                for i in range(lo, hi + 1):
                    parts = {j.target: load_target_frame(_tmp_path(partial, j.target, i), clip.width) for j in shot_jobs}
                    labels[i] = compose_labels(parts, size)
                    write_label(label_path(partial, i), labels[i])
                _rmtree(partial / TMP_DIRNAME)
        except KeyboardInterrupt:                  # T-603: posisi untuk satu baris; isolate/ lama tetap utuh (hasil ditulis di akhir)
            raise StageInterrupted(f"[1b] frame {n_done}/{n_total}") from None
        finally:
            run_info["vram"] = {**{k: v for k, v in (run_info.get("open") or {}).items()
                                   if k in ("vram_free_before_load_mib", "vram_total_mib")}, **backend.stats()}
            close_quietly(backend)
        run_info["ram"]["peak_process_mib"] = process_peak_ram_mib()
    statuses = {i: frame_status(labels.get(i)) for i in clip.indices}
    shots_of = {i: shot_of(i, shots) for i in clip.indices}
    rows = [{"event": "frame", "frame": FRAME_PATTERN % i, "index": i, "shot": list(shots_of[i]), "status": statuses[i],
             "pixel_sha256": per_frame[i], "targets": per_target[i]} for i in clip.indices]
    write_bytes_atomic(partial / FRAMES_LOG_FILENAME,
                       "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8"))
    count = {s: sum(v == s for v in statuses.values()) for s in (STATUS_TARGET, STATUS_EMPTY, STATUS_NO_PROMPT)}
    ts = np.asarray(infer_times) if infer_times else None
    run_info.update(
        finished_utc=utc_now(), wall_s=round(time.perf_counter() - t_run, 2),
        infer_s={"n": len(infer_times), "p50": round(float(np.median(ts)), 3) if ts is not None else None,
                 "max": round(float(ts.max()), 3) if ts is not None else None,
                 "total": round(float(ts.sum()), 2) if ts is not None else 0.0},
        nonfinite_frames=sorted(nonfinite))
    manifest = {"stage": "isolate", "format": FORMAT, CLIP_KEY: identity,
                "frame_size": {"width": clip.width, "height": clip.height}, "frame_count": len(clip.indices),
                "key": key, "key_sha256": canonical_sha256(key), "model": model_info,
                "processor": info.processor if jobs else None, "mode": MODE_DOC,
                "scenes": {"cut_frames": cuts, "shots": [list(s) for s in shots]},
                "jobs": jobs_out, "targets": sorted({j.target for j in jobs}),
                "frames": {**count, "kosong_ranges": ranges([i for i, s in statuses.items() if s == STATUS_EMPTY]),
                           "tanpa_prompt_ranges": ranges([i for i, s in statuses.items() if s == STATUS_NO_PROMPT])},
                "output": dict(OUTPUT_INFO), "run": run_info, "created_utc": utc_now()}
    write_json_atomic(partial / MANIFEST_FILENAME, manifest)
    promote(clip)
    log(f"[1b] selesai: target {count[STATUS_TARGET]}, kosong {count[STATUS_EMPTY]}, tanpa_prompt {count[STATUS_NO_PROMPT]} "
        f"frame; SAM {len(infer_times)} frame, {run_info['wall_s']:.1f} s" + (f"; NaN/inf di {sorted(nonfinite)}" if nonfinite else ""))
    return {"status": "dihitung", "counts": count, "jobs": jobs_out, "manifest": manifest}


# ── Download (online, terpisah dari run) ───────────
def download(cfg: PipelineConfig, *, log: Callable[[str], None] = print) -> str:
    """Unduh SAM 2.1 tiny (berkas jalur `transformers` saja) ke cache HF + cek lisensi. Return commit hash snapshot."""
    model_id, rev = cfg.isolate.model_id, cfg.isolate.revision
    require_model_id(model_id)
    if rev is not None and not COMMIT_HASH_RE.fullmatch(rev):
        raise StageError(f"isolate.revision ({rev!r}) harus commit hash 40 karakter hex")
    if hf_offline_active():
        raise StageError("HF_HUB_OFFLINE aktif di environment — --download butuh online; hapus variabel itu")
    from huggingface_hub import snapshot_download

    path = Path(snapshot_download(model_id, revision=rev or DOWNLOAD_REVISION_FALLBACK, allow_patterns=list(REQUIRED_FILES)))
    lic = require_license((path / MODEL_CARD).read_text(encoding="utf-8", errors="replace"), model_id)
    commit = path.name
    log(f"{model_id} snapshot {commit} (lisensi {lic}) → {path}")
    if rev is None:
        log(f"revision belum di-pin — isi isolate.revision: \"{commit}\" di configs/default.yaml")
    return commit


# ── Entry point stage (dipanggil cli.py) ───────────
def main(argv: list[str] | None = None, backend_factory: BackendFactory | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m rotoscope.isolate",
                                description="Stage [1b]: frames + target/points.json → isolate/ (SAM 2.1 hiera tiny; opt-in)")
    p.add_argument("--config", type=Path, default=None,
                   help=f"YAML pipeline (default: {DEFAULT_CONFIG.as_posix()} kalau ada, selain itu default kode)")
    add_work_dir_arg(p)
    p.add_argument("--yes", action="store_true", help="izinkan memakai GPU (hitung baru / basi) dan menghapus hasil lama")
    p.add_argument("--restart", action="store_true", help="hapus isolate/ lalu hitung ulang (butuh --yes)")
    p.add_argument("--dry-run", action="store_true", help="tampilkan status + adegan + job + perkiraan; tanpa GPU / tulis")
    p.add_argument("--download", action="store_true", help="unduh SAM 2.1 tiny + model card (online) lalu keluar")
    args = p.parse_args(argv)
    set_message_config(args.config)          # T-607: --config non-default ikut di perintah siap tempel
    reconfigure_stdio()

    def log(msg: str) -> None:
        print(msg, flush=True)

    if args.download and (args.restart or args.dry_run or args.yes):
        p.error("--download tidak bisa digabung dengan --restart / --dry-run / --yes")
    try:
        path = args.config if args.config is not None else (DEFAULT_CONFIG if DEFAULT_CONFIG.is_file() else None)
        cfg = load_pipeline(path, overrides=work_dir_overrides(args.work_dir))
        if args.download:
            download(cfg, log=log)
        else:
            t0 = time.perf_counter()
            res = run_isolate(cfg, yes=args.yes, restart=args.restart, dry_run=args.dry_run,
                              backend_factory=backend_factory, log=log)
            if res["status"] == "dihitung":
                log(f"selesai ({time.perf_counter() - t0:.1f} s)")
    except StageOOMError as e:
        print(f"ERROR (OOM): {e}", file=sys.stderr)
        return EXIT_OOM
    except (StageError, ConfigError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return EXIT_PRECONDITION
    except StageInterrupted as e:                  # T-603: satu baris, tanpa traceback; isolate/ lama utuh
        print(interrupted_line(f"{e}; isolate/ lama utuh (seluruh klip dihitung ulang)"), file=sys.stderr, flush=True)
        return EXIT_INTERRUPTED
    except KeyboardInterrupt:                      # di luar loop SAM (mis. memuat model)
        print(interrupted_line("[1b] isolate; isolate/ lama utuh (seluruh klip dihitung ulang)"), file=sys.stderr, flush=True)
        return EXIT_INTERRUPTED
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
