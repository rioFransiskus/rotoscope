"""Pratinjau masker [1b] di CPU (T-609): `python -m rotoscope pick <video> --mask-preview` → <klip>/target/pratinjau/.

Untuk SETIAP frame berklik di target/points.json dan setiap target di frame itu: satu sesi SAM 2.1 tiny SATU frame dengan
prompt apa adanya (titik + label + kotak), device CPU. Itu cara [1b] menghitung frame klik (frame kondisi awal, tanpa memori,
`isolate.track_job`); T-609 Tahap 1: skor + luas = [1b] produksi GPU di 4 / 4 frame klik. BUKAN hasil pelacakan video: frame di
antara klik tidak diperiksa.

Keluaran (BUKAN output pipeline: tidak dibaca stage mana pun, tidak masuk kunci [1b], tidak di isolate/):
  target/pratinjau/klik_f%05d_t%d.png   bar teks + FOTO + masker biru transparan + garis tepi + kotak kuning + titik kiri hijau /
                                        kanan merah (gaya pratinjau T-608)
  target/pratinjau/pratinjau.json       ringkasan `rotoscope-mask-preview/1`
Ditulis ke target/pratinjau.partial/ lalu ditukar (terputus / galat → folder lama utuh; sisa dihapus di awal run berikutnya).
Selalu dibuat ulang SELURUHNYA. pick menghapus target/pratinjau/ saat titik disimpan berubah (keputusan Rio T-609 no. 4).

CPU dipaksa: proses ini MENOLAK berjalan bila CUDA_VISIBLE_DEVICES ≠ "-1" (cli memasangnya; dicek sebelum torch di-import) dan
gagal (hasil tidak dipromosikan) bila torch.cuda terinisialisasi. Model = isolate.model_id / revision dari cache HF (offline,
tanpa unduhan; tidak di cache → perintah `download`). Cek RAM = isolate.ram_min_free_mib. Peringatan klik kanan =
isolate.neg_points_warn_max (sama dengan pick / isolate).

    python -m rotoscope.mask_preview --work-dir DIR [--config PATH]     (lewat cli: python -m rotoscope pick <video> --mask-preview)
Exit code: 0 selesai, 1 prasyarat gagal, 130 dihentikan.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from rotoscope import isolate as iso
from rotoscope import pick
from rotoscope.config import ConfigError, PipelineConfig, ensure_dir, load_pipeline
from rotoscope.ingest import DEFAULT_TARGET_FPS, META_FILENAME
from rotoscope.stage_common import (
    EXIT_OK, EXIT_PRECONDITION, StageError, StageInterrupted, add_work_dir_arg, cli_cmd, clip_identity, close_quietly,
    interrupted_line, reconfigure_stdio, set_message_config, utc_now, work_dir_overrides, write_bytes_atomic, write_json_atomic,
)

EXIT_INTERRUPTED = 130
CUDA_ENV, CUDA_OFF = "CUDA_VISIBLE_DEVICES", "-1"
DEVICE = "cpu"
PARTIAL_DIRNAME = pick.PREVIEW_DIRNAME + ".partial"
OLD_DIRNAME = pick.PREVIEW_DIRNAME + ".old"
SUMMARY_FILENAME = "pratinjau.json"
FORMAT = "rotoscope-mask-preview/1"
PNG_PATTERN = "klik_f%05d_t%d.png"
NOTE = ("satu frame per klik (sesi SAM 1 frame, prompt apa adanya, CPU) = masker [1b] di frame klik; BUKAN hasil pelacakan "
        "video (frame di antara klik tidak diperiksa)")
DEFAULT_CONFIG = iso.DEFAULT_CONFIG

# Tampilan = gaya pratinjau T-608 yang dipakai Rio. Konstanta UI, bukan YAML: tidak memengaruhi keluaran pipeline.
MASK_BGR, MASK_ALPHA = (255, 160, 0), 0.45          # biru muda transparan
EDGE_PX = 2
POS_BGR, NEG_BGR, BOX_BGR = (0, 200, 0), (0, 0, 255), (0, 215, 255)   # hijau, merah, kuning
DOT_R, RING_R = 5, 7
BAR_H, BAR_PAD, BAR_BG, BAR_FG = 44, 10, 235, (0, 0, 0)
FONT, FONT_SCALE, FONT_SCALE_MIN, FONT_STEP = cv2.FONT_HERSHEY_DUPLEX, 0.75, 0.4, 0.05
# Hanya untuk pesan: terukur T-609 Tahap 1 (CPU 6 thread): 1,8–2,2 s per (frame, target); import + muat model 4,5–14 s.
SECONDS_PER_TARGET_ESTIMATE = 2.0
LOAD_SECONDS_ESTIMATE = 12.0

CHECKLIST = (
    "Buka setiap PNG dan cek:",
    "  1. Warna biru menutup target SAJA (orang lain / latar tidak ikut).",
    "  2. Bagian target yang penting tidak bolong (kepala, tangan, badan di dalam kotak kuning).",
    "  3. Skor objek positif. Negatif atau masker 0% = SAM tidak menemukan target.",
)

BackendFactory = Callable[[PipelineConfig], iso.SamBackend]


def _dec(v: float, spec: str) -> str:
    """Angka untuk Rio: desimal koma."""
    return format(v, spec).replace(".", ",")


def require_cpu_env() -> None:
    """Dipanggil SEBELUM torch di-import: tanpa CUDA_VISIBLE_DEVICES=-1 proses ini tidak berjalan."""
    cur = os.environ.get(CUDA_ENV)
    if cur != CUDA_OFF:
        raise StageError(f"pratinjau masker hanya berjalan di CPU: {CUDA_ENV} harus \"{CUDA_OFF}\" (sekarang {cur!r}) — jalankan "
                         f"lewat python -m rotoscope pick <video> {pick.PREVIEW_FLAG}")


def torch_cuda_initialized() -> bool:
    t = sys.modules.get("torch")
    return bool(t is not None and t.cuda.is_initialized())


def torch_threads() -> int | None:
    t = sys.modules.get("torch")
    return int(t.get_num_threads()) if t is not None else None


def cpu_backend(cfg: PipelineConfig) -> iso.SamBackend:
    return iso.TorchSamBackend(cfg, device=DEVICE)


def mask_components(mask: np.ndarray) -> tuple[int, float]:
    """(jumlah komponen 8-arah, porsi komponen terbesar terhadap luas MASKER); masker kosong → (0, 0.0)."""
    if not mask.any():
        return 0, 0.0
    n, _lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return n - 1, float(st[1:, cv2.CC_STAT_AREA].max() / mask.sum())


def bar_scale(text: str, width: int) -> float:
    """Skala font terbesar ≤ FONT_SCALE yang membuat teks bar muat di lebar frame (minimum FONT_SCALE_MIN)."""
    sc = FONT_SCALE
    while sc > FONT_SCALE_MIN and cv2.getTextSize(text, FONT, sc, 2)[0][0] > width - 2 * BAR_PAD:
        sc = round(sc - FONT_STEP, 2)
    return sc


def render(bgr: np.ndarray, mask: np.ndarray, points: list[dict], box, text: str) -> np.ndarray:
    """Bar teks (BAR_H px) di atas FOTO + masker transparan + garis tepi + kotak + titik. Ukuran = W × (H + BAR_H)."""
    img = bgr.copy()
    if mask.any():
        col = np.empty_like(img)
        col[:] = MASK_BGR
        img[mask] = (img[mask] * (1 - MASK_ALPHA) + col[mask] * MASK_ALPHA).astype(np.uint8)
        cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, cnts, -1, MASK_BGR, EDGE_PX, cv2.LINE_AA)
    if box is not None:
        x0, y0, x1, y1 = (int(round(v)) for v in box)
        cv2.rectangle(img, (x0, y0), (x1, y1), BOX_BGR, EDGE_PX, cv2.LINE_AA)
    for p in points:
        c = (int(round(p["x"])), int(round(p["y"])))
        cv2.circle(img, c, RING_R, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img, c, DOT_R, POS_BGR if p["label"] == 1 else NEG_BGR, -1, cv2.LINE_AA)
    bar = np.full((BAR_H, img.shape[1], 3), BAR_BG, np.uint8)
    sc = bar_scale(text, img.shape[1])
    cv2.putText(bar, text, (BAR_PAD, BAR_H - 14), FONT, sc, BAR_FG, 2 if sc >= 0.6 else 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def prompt_of(frame: int, target: dict) -> iso.Prompt:
    """Prompt target APA ADANYA (sama dengan `isolate.build_jobs`)."""
    box = target.get("box")
    return iso.Prompt(frame, tuple((float(p["x"]), float(p["y"])) for p in target["points"]),
                      tuple(int(p["label"]) for p in target["points"]), tuple(float(v) for v in box) if box is not None else None)


def read_bgr(path: Path) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise StageError(f"{path} tidak terbaca")
    return img


def promote(partial: Path, final: Path, old: Path) -> None:
    """pratinjau.partial/ (lengkap) → pratinjau/; pratinjau/ lama → pratinjau.old/ → dihapus."""
    if old.exists():
        shutil.rmtree(old)
    if final.exists():
        iso._replace_dir(final, old)
    iso._replace_dir(partial, final)
    if old.exists():
        shutil.rmtree(old)


def run_preview(cfg: PipelineConfig, *, backend_factory: BackendFactory | None = None, log: Callable[[str], None] = print,
                ram_probe: Callable[[], float | None] | None = None,
                cuda_probe: Callable[[], bool] = torch_cuda_initialized) -> dict:
    """Pratinjau semua frame berklik → target/pratinjau/. Return ringkasan (isi pratinjau.json)."""
    require_cpu_env()
    work_dir = cfg.paths.work_dir
    clip = iso.load_clip(work_dir)
    if not clip.points_path.is_file():
        raise StageError(f"{clip.points_path} tidak ada — belum ada titik. Klik target dulu: {cli_cmd('pick', work_dir)}")
    doc = iso.load_valid_points(clip, {})
    if not doc["frames"]:
        raise StageError(f"{clip.points_path} tanpa frame berklik — klik target dulu: {cli_cmd('pick', work_dir)}")
    meta = json.loads((work_dir / META_FILENAME).read_text(encoding="utf-8"))
    fps = float(meta.get("target_fps", DEFAULT_TARGET_FPS))
    start = int((meta.get("trim") or {}).get("start_frame", 0))
    limit = cfg.isolate.neg_points_warn_max
    over = pick.doc_over_negative(doc, limit)
    over_set = {(f, t) for f, t, _ in over}
    n_jobs = sum(len(e["targets"]) for e in doc["frames"])
    log(f"pratinjau masker (CPU, {cfg.isolate.model_id}@{str(cfg.isolate.revision)[:8]}): {len(doc['frames'])} frame klik, "
        f"{n_jobs} (frame, target) ≈ {n_jobs * SECONDS_PER_TARGET_ESTIMATE + LOAD_SECONDS_ESTIMATE:.0f} s")
    if over:
        log(iso.neg_points_warning(clip, over, limit, preview_hint=False))
    ram_free = (ram_probe or iso.ram_free_mib)()
    iso.check_ram(ram_free, cfg.isolate.ram_min_free_mib)
    target_dir = work_dir / pick.TARGET_DIRNAME
    partial, final, old = target_dir / PARTIAL_DIRNAME, target_dir / pick.PREVIEW_DIRNAME, target_dir / OLD_DIRNAME
    for d in (partial, old):                         # sisa run yang terputus
        if d.exists():
            shutil.rmtree(d)
    ensure_dir(partial)
    started, t_run = utc_now(), time.perf_counter()
    thr = cfg.isolate.mask_threshold
    size = (clip.height, clip.width)
    backend = (backend_factory or cpu_backend)(cfg)
    frames_out, warnings = [], []
    try:
        info = backend.describe()
        opened = backend.open()
        if opened.get("device") not in (None, DEVICE):
            raise StageError(f"backend pratinjau berjalan di {opened.get('device')!r}, bukan {DEVICE!r}")
        load_s = round(time.perf_counter() - t_run, 2)
        log(f"  model dimuat ({load_s} s; {DEVICE})")
        for e in doc["frames"]:
            f = int(e["frame_index"])
            fpath = clip.frame_path(f)
            bgr = read_bgr(fpath)
            sec = (start + f) / fps
            rows = []
            for t in e["targets"]:
                tid = int(t["id"])
                t0 = time.perf_counter()
                eng = backend.session([fpath], size, tid)
                try:
                    eng.add_prompt(0, prompt_of(f, t))
                    logits, score = eng.run_frame(0)
                finally:
                    eng.release()
                if logits.shape != size:
                    raise StageError(f"logit frame {f} target {tid} berukuran {logits.shape} ≠ {size}")
                mask = iso.target_mask(logits, thr)
                n_comp, big = mask_components(mask)
                area = int(mask.sum())
                frac = float(mask.mean())
                text = (f"frame {f} (detik {_dec(sec, '.2f')})  target {tid}  skor objek {_dec(score, '+.2f')}  masker "
                        f"{_dec(frac * 100, '.1f')}% frame")
                img = render(bgr, mask, t["points"], t.get("box"), text)
                ok, buf = cv2.imencode(".png", img)
                if not ok:
                    raise StageError(f"gagal meng-encode PNG pratinjau frame {f} target {tid}")
                name = PNG_PATTERN % (f, tid)
                write_bytes_atomic(partial / name, buf.tobytes())
                notes = []
                if not np.isfinite(score) or score <= 0 or area == 0:
                    notes.append(f"masker kosong / skor objek tidak positif (skor {_dec(score, '+.2f')}, {area} px)")
                if (f, tid) in over_set:
                    notes.append(f"klik kanan > {limit}")
                warnings += [f"f{f} target {tid}: {n}" for n in notes]
                n_neg = sum(p["label"] == 0 for p in t["points"])
                rows.append({"id": tid, "kiri": len(t["points"]) - n_neg, "kanan": n_neg, "kotak": t.get("box") is not None,
                             "skor_objek": round(score, 3) if np.isfinite(score) else None, "luas_px": area,
                             "porsi_frame": round(frac, 4), "komponen_8arah": n_comp, "porsi_komponen_terbesar": round(big, 4),
                             "waktu_s": round(time.perf_counter() - t0, 3), "png": name, "peringatan": notes})
                log(f"  f{f} target {tid}: skor objek {_dec(score, '+.2f')}, masker {_dec(frac * 100, '.1f')}% frame, {n_comp} "
                    f"bagian (terbesar {_dec(big * 100, '.1f')}%) -> {name}")
            frames_out.append({"frame": f, "detik": round(sec, 3), "frame_pixel_sha256": e["frame_pixel_sha256"], "targets": rows})
    except KeyboardInterrupt:
        raise StageInterrupted(f"pratinjau masker ({len(frames_out)}/{len(doc['frames'])} frame); target/pratinjau/ lama utuh") from None
    finally:
        close_quietly(backend)
    if cuda_probe():
        raise StageError("torch.cuda terinisialisasi — pratinjau harus CPU saja; hasil TIDAK dipakai (target/pratinjau/ lama utuh)")
    summary = {"format": FORMAT, "clip": clip_identity(work_dir), "frame_size": {"width": clip.width, "height": clip.height},
               "points_sha256": hashlib.sha256(clip.points_path.read_bytes()).hexdigest(),
               "model": {"model_id": info.model_id, "revision": info.revision, "license": info.license,
                         "precision": info.precision},
               "device": DEVICE, "mask_threshold": thr, "neg_points_warn_max": limit, "libs": iso.lib_versions(), "note": NOTE,
               "frames": frames_out, "peringatan": warnings,
               "run": {"started_utc": started, "load_s": load_s, "wall_s": round(time.perf_counter() - t_run, 2),
                       "threads": torch_threads(), "ram_free_before_mib": ram_free,
                       "peak_process_mib": iso.process_peak_ram_mib(), "cuda_initialized": False},
               "created_utc": utc_now()}
    write_json_atomic(partial / SUMMARY_FILENAME, summary)
    promote(partial, final, old)
    log(f"Pratinjau masker (CPU; SATU frame per klik = masker [1b] di frame klik, BUKAN hasil pelacakan video): {final}")
    for w in warnings:
        log(f"PERINGATAN: {w}")
    for line in CHECKLIST:
        log(line)
    log(f"Masker salah di frame N -> {cli_cmd('pick', work_dir)} --frame N, lalu pratinjau lagi.")
    log("Frame di antara klik tidak diperiksa di sini; lihat hasil [1b] sesudah run.")
    return summary


def main(argv: list[str] | None = None, backend_factory: BackendFactory | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m rotoscope.mask_preview",
                                description="T-609: pratinjau masker SAM 2.1 tiny di CPU per frame klik → target/pratinjau/")
    p.add_argument("--config", type=Path, default=None,
                   help=f"YAML pipeline (default: {DEFAULT_CONFIG.as_posix()} kalau ada, selain itu default kode)")
    add_work_dir_arg(p)
    args = p.parse_args(argv)
    set_message_config(args.config)          # --config non-default ikut di perintah siap tempel
    reconfigure_stdio()
    try:
        path = args.config if args.config is not None else (DEFAULT_CONFIG if DEFAULT_CONFIG.is_file() else None)
        cfg = load_pipeline(path, overrides=work_dir_overrides(args.work_dir))
        run_preview(cfg, backend_factory=backend_factory, log=lambda m: print(m, flush=True))
    except (StageError, ConfigError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return EXIT_PRECONDITION
    except StageInterrupted as e:
        print(interrupted_line(str(e)), file=sys.stderr, flush=True)
        return EXIT_INTERRUPTED
    except KeyboardInterrupt:
        print(interrupted_line("pratinjau masker; target/pratinjau/ lama utuh"), file=sys.stderr, flush=True)
        return EXIT_INTERRUPTED
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
