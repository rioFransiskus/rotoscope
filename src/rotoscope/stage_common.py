"""Helper bersama stage pipeline (T-105): tulis atomik + retry Windows, frames.jsonl, daftar frame,
error + exit code, stdout cp1252. Bagian khusus stage GPU ([2] segment, [2c] depth): mode offline HF,
commit hash revision, cek cache checkpoint, OOM, kondisi VRAM.

Dipakai segment.py dan depth.py; stage CPU [3]/[4]/[5] memakai bagian umum (tulis atomik, manifest,
frames.jsonl, resume). Modul ini tidak meng-import torch/transformers/huggingface_hub di level atas —
import malas di dalam fungsi yang membutuhkannya.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from rotoscope.ingest import FRAME_PATTERN, FRAMES_DIRNAME, META_FILENAME

TMP_SUFFIX = ".tmp"

# ── Windows: os.replace bisa gagal sesaat (file dikunci antivirus/indexer, WinError 5/32) ─
REPLACE_RETRIES = 5
REPLACE_DELAY_S = 0.2

# ── Exit code (dipakai cli.py, T-104b) ─────────────
EXIT_OK = 0
EXIT_PRECONDITION = 1
EXIT_OOM = 3

MIB = 2**20

# ── Checkpoint HF ──────────────────────────────────
COMMIT_HASH_RE = re.compile(r"[0-9a-f]{40}")
HF_OFFLINE_ENV = "HF_HUB_OFFLINE"
DOWNLOAD_REVISION_FALLBACK = "main"  # hanya untuk --download kalau revision null
_TRUTHY = ("1", "true", "yes", "on")


class StageError(RuntimeError):
    """Prasyarat stage gagal (exit 1): config/revision/cache/CUDA/VRAM/manifest/frame/tulis file."""


class StageOOMError(StageError):
    """GPU kehabisan memori di tengah run (exit 3). Tidak ada ganti model otomatis."""


class BackendOOM(RuntimeError):
    """Dilempar backend saat OOM; `vram` = kondisi VRAM saat gagal (MiB)."""

    def __init__(self, vram: dict):
        super().__init__("CUDA out of memory")
        self.vram = vram


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
    raise StageError(f"gagal mengganti {dst} setelah {REPLACE_RETRIES} percobaan ({last}). File mungkin "
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


def clean_tmp(directory: Path) -> None:
    """File *.tmp tidak pernah dianggap output — dihapus di awal run."""
    if directory.is_dir():
        for p in directory.rglob("*" + TMP_SUFFIX):
            p.unlink(missing_ok=True)


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


def last_frame_records(frames_log: Path) -> dict[str, dict]:
    """Record `frame` terakhir per nama frame di frames.jsonl (run ulang menimpa yang lama)."""
    return {r["frame"]: r for r in read_jsonl(frames_log) if r.get("event") == "frame" and "frame" in r}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ── Identitas klip (T-108) ─────────────────────────
CLIP_KEY = "clip"
HASH_SHORT = 12


def clip_identity_from_bytes(meta_bytes: bytes) -> dict:
    """Identitas klip = sha256 byte meta.json + source_path. Ingest ulang video + parameter yang sama
    menghasilkan meta.json byte-identik (test_ingest); parameter ingest berbeda → hash berbeda.
    Batas: meta.json tidak memuat ukuran/hash file video — video lain yang menimpa path yang sama
    dengan fps/durasi/ukuran identik tidak terdeteksi."""
    meta = json.loads(meta_bytes.decode("utf-8"))
    return {"meta_sha256": hashlib.sha256(meta_bytes).hexdigest(), "source_path": str(meta["source_path"])}


def clip_identity(work_dir: Path) -> dict:
    meta_path = work_dir / META_FILENAME
    if not meta_path.is_file():
        raise StageError(f"{meta_path} tidak ada — jalankan stage [1] ingest dulu")
    return clip_identity_from_bytes(meta_path.read_bytes())


def describe_identity(ident: dict | None) -> str:
    if not ident:
        return "(tanpa identitas)"
    return f"{ident.get('source_path')} (meta {str(ident.get('meta_sha256'))[:HASH_SHORT]})"


def identity_diff(old: dict | None, new: dict) -> list[str]:
    """Baris 'lama → baru' per field identitas yang berbeda (hash dipendekkan)."""
    old = old or {}
    out = []
    for k in ("source_path", "meta_sha256"):
        a, b = old.get(k), new.get(k)
        if a != b:
            if k == "meta_sha256":
                a, b = (a[:HASH_SHORT] if a else a), (b[:HASH_SHORT] if b else b)
            out.append(f"{k}: {a} → {b}")
    return out


def require_same_clip(manifest: dict, current: dict, *, stage: str, out_dir: Path, cmd: str) -> None:
    """Tolak output stage GPU milik klip lain / tanpa identitas (stage [2]/[2c]; --qc-only juga).
    `cmd` = perintah stage, mis. "python -m rotoscope.segment"."""
    old = manifest.get(CLIP_KEY)
    if not isinstance(old, dict) or not old.get("meta_sha256"):
        raise StageError(
            f"output {stage} di {out_dir} tidak memuat identitas klip (manifest lama) — tidak diketahui milik "
            f"klip mana. Klip saat ini: {describe_identity(current)}.\n"
            f"  - kalau output itu memang milik klip ini: {cmd} --adopt (mencatat identitas tanpa inferensi ulang)\n"
            f"  - kalau bukan / ragu: {cmd} --restart")
    if old["meta_sha256"] != current["meta_sha256"]:
        raise StageError(
            f"output {stage} di {out_dir} milik klip LAIN:\n"
            f"  klip lama (output): {describe_identity(old)}\n"
            f"  klip baru (meta.json): {describe_identity(current)}\n"
            f"Satu-satunya jalan: {cmd} --restart (menghapus output lama). --adopt tidak berlaku untuk "
            f"identitas berbeda.")


def adopt_identity(*, stage: str, manifest_path: Path, frames_log: Path, work_dir: Path, size: dict,
                   names: tuple[str, ...], frame_valid: Callable[[str], bool], orphans: list[Path],
                   cmd: str, log: Callable[[str], None] = print) -> dict:
    """--adopt: catat identitas klip saat ini ke manifest yang ADA tanpa inferensi ulang.

    Ini PERNYATAAN pengguna bahwa output milik klip ini. Cek kewajaran (frame_size, setiap frame klip valid,
    tanpa file yatim) tidak bisa membedakan klip lain dengan frame_size dan frame_count yang sama.
    Ringkasan dicetak SEBELUM menulis. Return {"adopted": bool, "identity": dict}.
    """
    if not manifest_path.is_file():
        raise StageError(f"{manifest_path} tidak ada — tidak ada yang di-adopt. Jalankan stage {stage} dulu")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise StageError(f"gagal membaca {manifest_path} ({e}) — jalankan {cmd} --restart") from None
    current = clip_identity(work_dir)
    old = manifest.get(CLIP_KEY)
    if isinstance(old, dict) and old.get("meta_sha256"):
        if old["meta_sha256"] == current["meta_sha256"]:
            log(f"{stage} --adopt: manifest sudah memuat identitas klip ini — tidak ada yang diubah")
            return {"adopted": False, "identity": current}
        raise StageError(
            f"manifest {stage} sudah memuat identitas klip LAIN — --adopt ditolak:\n"
            f"  klip lama (output): {describe_identity(old)}\n  klip baru (meta.json): {describe_identity(current)}\n"
            f"Satu-satunya jalan: {cmd} --restart")
    if manifest.get("frame_size") != size:
        raise StageError(f"{stage} --adopt ditolak: frame_size manifest {manifest.get('frame_size')} ≠ "
                         f"meta.json {size}")
    invalid = [n for n in names if not frame_valid(n)]
    if invalid:
        raise StageError(f"{stage} --adopt ditolak: {len(invalid)}/{len(names)} frame belum/rusak, mis. "
                         f"{invalid[0]} — output tidak lengkap, gunakan {cmd} --restart")
    if orphans:
        raise StageError(f"{stage} --adopt ditolak: {len(orphans)} file output di luar daftar frame klip, mis. "
                         f"{orphans[0].name} — output milik klip lain? Gunakan {cmd} --restart")
    log(f"{stage} --adopt: mencatat identitas klip ini ke {manifest_path} (tanpa inferensi)\n"
        f"  source_path : {current['source_path']}\n"
        f"  meta_sha256 : {current['meta_sha256'][:HASH_SHORT]}\n"
        f"  frame_count : {len(names)} ({len(names)} frame valid, frame_size {size['width']}x{size['height']})\n"
        f"  PERNYATAAN Anda: output itu milik klip ini — cek kewajaran tidak bisa membedakan klip lain "
        f"dengan frame_size dan frame_count sama")
    adopted_utc = utc_now()
    write_json_atomic(manifest_path, {**manifest, CLIP_KEY: current, "adopted_utc": adopted_utc})
    append_jsonl(frames_log, {"event": "adopt", "time_utc": adopted_utc, "clip": current,
                              "n_frames": len(names)})
    return {"adopted": True, "identity": current}


# ── Frame input ────────────────────────────────────
def load_frame_list(work_dir: Path) -> tuple[tuple[str, ...], tuple[int, ...], int, int]:
    """meta.json stage [1] → (nama frame berurutan, indeks, working_width, working_height)."""
    meta_path = work_dir / META_FILENAME
    if not meta_path.is_file():
        raise StageError(f"{meta_path} tidak ada — jalankan stage [1] ingest dulu")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    start = int(meta.get("frame_index_start", 0))
    indices = tuple(range(start, start + int(meta["frame_count"])))
    names = tuple(FRAME_PATTERN % i for i in indices)
    return names, indices, int(meta["working_width"]), int(meta["working_height"])


def require_frames(work_dir: Path, names: tuple[str, ...] | list[str]) -> None:
    missing = [n for n in names if not (work_dir / FRAMES_DIRNAME / n).is_file()]
    if missing:
        raise StageError(f"{len(missing)} frame hilang di {work_dir / FRAMES_DIRNAME}, mis. {missing[0]} "
                         f"— jalankan ulang stage [1] ingest")


def read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR) if path.is_file() else None
    if bgr is None:
        raise StageError(f"gagal membaca frame {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ── Checkpoint HF (stage GPU) ──────────────────────
def set_offline() -> None:
    """HF_HUB_OFFLINE=1 di environment proses — dipanggil SEBELUM transformers/huggingface_hub di-import.

    huggingface_hub membaca variabel ini saat di-import; kalau sudah ter-import lebih dulu,
    local_files_only=True di setiap from_pretrained tetap menjamin tidak ada akses jaringan.
    """
    os.environ[HF_OFFLINE_ENV] = "1"


def hf_offline_active() -> bool:
    return os.environ.get(HF_OFFLINE_ENV, "").strip().lower() in _TRUTHY


def ensure_cached(model_id: str, revision: str, required_files: tuple[str, ...], download_cmd: str,
                  cache_dir: str | Path | None = None) -> None:
    """Semua file checkpoint revision ini ada di cache HF — kalau tidak, berhenti + perintah unduh."""
    from huggingface_hub import try_to_load_from_cache

    missing = [f for f in required_files
               if not isinstance(try_to_load_from_cache(model_id, f, cache_dir=cache_dir, revision=revision), str)]
    if missing:
        raise StageError(
            f"checkpoint {model_id} revision {revision} tidak ada di cache HF (hilang: {', '.join(missing)}). "
            f"Run berjalan offline — unduh dulu (online, sekali jalan):\n"
            f"    {download_cmd}")


def cached_file(model_id: str, filename: str, revision: str, cache_dir: str | Path | None = None) -> Path:
    """Path file di cache HF (panggil setelah ensure_cached)."""
    from huggingface_hub import try_to_load_from_cache

    p = try_to_load_from_cache(model_id, filename, cache_dir=cache_dir, revision=revision)
    if not isinstance(p, str):
        raise StageError(f"{filename} dari {model_id}@{revision} tidak ada di cache HF")
    return Path(p)


# ── GPU ────────────────────────────────────────────
def is_oom(torch, e: BaseException) -> bool:
    return isinstance(e, torch.OutOfMemoryError) or "out of memory" in str(e).lower()


def vram_state(torch) -> dict:
    free, total = torch.cuda.mem_get_info()
    return {"free_mib": round(free / MIB, 1), "total_mib": round(total / MIB, 1),
            "reserved_mib": round(torch.cuda.memory_reserved() / MIB, 1),
            "allocated_mib": round(torch.cuda.memory_allocated() / MIB, 1),
            "max_reserved_mib": round(torch.cuda.max_memory_reserved() / MIB, 1)}


# ── Entry point ────────────────────────────────────
def reconfigure_stdio() -> None:
    """stdout/stderr yang diarahkan ke file di Windows = cp1252: karakter seperti "→" / "≠" di pesan
    membuat print crash (UnicodeEncodeError) → ganti dengan "?" daripada gagal."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
