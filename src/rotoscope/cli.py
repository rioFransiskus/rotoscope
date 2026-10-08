"""CLI final (T-104b): `python -m rotoscope <subperintah>`.

    python -m rotoscope run <video> [--config PATH] [--style PATH] [--seg-model 0.8b|0.4b] [--limit N]
                                    [--preview N [--from K]]
                                    [--restart-from ingest|segment|depth|stabilize|vectorize|stylize|export [--yes]]
    python -m rotoscope ingest|segment|depth|stabilize|vectorize|stylize|export <video> [flag stage ...]
    python -m rotoscope download [--config PATH] [--seg-model 0.8b|0.4b]

Satu video = satu folder kerja `<paths.work_dir>/clips/<nama video disanitasi>/` (nama disanitasi dengan fungsi
yang sama dengan `{source}` di export.filename); `paths.out_dir` tidak berubah. Folder itu diteruskan ke tiap stage
lewat `--work-dir` (menang atas config) — tanpa menulis YAML.

Prinsip #3 docs/01: stage GPU ([2] segment, [2c] depth) selalu proses sendiri (subprocess; stdout/stderr
diteruskan apa adanya) dan proses induk TIDAK meng-import torch / menyentuh CUDA. Stage CPU (ingest, stabilize,
vectorize, export) dipanggil in-process lewat main(argv) stage-nya, jadi flag stage tidak diparse ulang di sini.
[4] vectorize dan [5] stylize masuk urutan `run` (T-203b; --style diteruskan ke stylize dan, sejak T-404a, ke export untuk kertas bertekstur).

Exit code: 0 sukses | 1 prasyarat gagal (config, pre-flight, VRAM, identitas, ...) | 2 salah pakai argumen |
3 OOM di stage GPU | 130 dihentikan (Ctrl+C). Kode stage yang gagal dikembalikan apa adanya.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from rotoscope import depth as depth_stage
from rotoscope import export as export_stage
from rotoscope import ingest as ingest_stage
from rotoscope import paper
from rotoscope import segment as segment_stage
from rotoscope import stabilize as stabilize_stage
from rotoscope import stylize as stylize_stage
from rotoscope import vectorize as vectorize_stage
from rotoscope.config import (
    DEFAULT_STYLE_NAME, ConfigError, PipelineConfig, load_pipeline, load_style, resolve_style, style_name,
)
from rotoscope.export import (
    DEFAULT_CONFIG, check_svg_owner, manifest_path_for, read_manifest, resolve_filename,
    sanitize_source_name, svg_dir_for,
)
from rotoscope.ingest import META_FILENAME
from rotoscope.segment import QC_REPORT_FILENAME
from rotoscope.stage_common import EXIT_OK, EXIT_OOM, EXIT_PRECONDITION, StageError, reconfigure_stdio

EXIT_USAGE = 2
EXIT_INTERRUPTED = 130

CLIPS_DIRNAME = "clips"
PROG = "python -m rotoscope"
STAGES = ("ingest", "segment", "depth", "stabilize", "vectorize", "stylize", "export")      # urutan run
GPU_STAGES = ("segment", "depth")
LABELS = {"ingest": "[1] ingest", "segment": "[2] segment", "depth": "[2c] depth",
          "stabilize": "[3] stabilize", "vectorize": "[4] vectorize", "stylize": "[5] stylize",
          "export": "[6] export"}
MODULES = {"segment": "rotoscope.segment", "depth": "rotoscope.depth"}

# Graf dependensi: ingest → {segment, depth} → stabilize → vectorize → stylize → export; depth TIDAK bergantung
# pada segment. --restart-from X = stage X (+ semua stage hilir kalau X = ingest) dijalankan dengan --restart.
# Hilir CPU yang basi dihitung ulang otomatis oleh stage-nya (prinsip #4); cli tidak menghapusnya. ingest selalu
# menulis ulang.
RESTART_SCOPE = {"ingest": ("segment", "depth", "stabilize", "vectorize", "stylize", "export"),
                 "segment": ("segment",), "depth": ("depth",), "stabilize": ("stabilize",),
                 "vectorize": ("vectorize",), "stylize": ("stylize",), "export": ("export",)}

# Hanya untuk pesan konfirmasi (estimasi waktu GPU per frame; terukur T-102b / T-105, GTX 1650 Ti).
GPU_SECONDS_PER_FRAME = {"segment": {"0.8b": 16.5, "0.4b": 8.0}, "depth": 0.19}
QC_INDEX_LIMIT = 10
TERMINATE_TIMEOUT_S = 10

_popen = subprocess.Popen                       # seam untuk test (subprocess dipalsukan)
CPU_MAINS: dict[str, Callable[[list[str]], int]] = {
    "ingest": ingest_stage.main, "stabilize": stabilize_stage.main, "vectorize": vectorize_stage.main,
    "stylize": stylize_stage.main, "export": export_stage.main,
}


class CliError(Exception):
    """Gagal sebelum / di antara stage; `code` = exit code proses."""

    def __init__(self, message: str, code: int = EXIT_PRECONDITION):
        super().__init__(message)
        self.code = code


@dataclass
class Ctx:
    video: Path
    cfg: PipelineConfig
    work_dir: Path
    config: Path | None = None            # hanya kalau diberikan pengguna (stage memuat default sendiri)
    style: Path | None = None             # diteruskan ke stylize dan export (kertas, T-404a)
    seg_model: str | None = None
    limit: int | None = None
    restart: tuple[str, ...] = ()         # stage yang dijalankan dengan --restart
    state: dict = field(default_factory=dict)   # {"stage": nama stage yang sedang jalan}


# ── Path ───────────────────────────────────────────
def _same_path(a: str | Path, b: str | Path) -> bool:
    """Bandingkan dengan fungsi resolve yang sama dengan ingest (`Path.resolve`), bukan string mentah."""
    return os.path.normcase(str(Path(a).resolve())) == os.path.normcase(str(Path(b).resolve()))


def clip_work_dir(cfg: PipelineConfig, video: Path) -> Path:
    return cfg.paths.work_dir / CLIPS_DIRNAME / sanitize_source_name(str(video))


def load_cfg(config: Path | None, seg_model: str | None = None) -> PipelineConfig:
    path = config if config is not None else (DEFAULT_CONFIG if DEFAULT_CONFIG.is_file() else None)
    try:
        return load_pipeline(path, overrides={"segment.model": seg_model} if seg_model else None)
    except ConfigError as e:
        raise CliError(str(e)) from None


def _read_meta(work_dir: Path) -> dict | None:
    try:
        meta = json.loads((work_dir / META_FILENAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return meta if isinstance(meta, dict) else None


# ── Pre-flight (CPU saja, SEBELUM penghapusan / ingest / subprocess apa pun) ──
def preflight_video(video: Path, work_dir: Path) -> None:
    """(a) file video ada; (b) folder kerja tidak berisi klip lain (ingest akan menghapus frame-nya)."""
    if not video.is_file():
        raise CliError(f"file video tidak ditemukan: {video}")
    meta = _read_meta(work_dir)
    old = meta.get("source_path") if meta else None
    if old and not _same_path(old, video):
        raise CliError(
            f"{work_dir} sudah berisi klip LAIN:\n"
            f"  klip lama: {old}\n"
            f"  klip baru: {video.resolve()}\n"
            f"Satu folder kerja = satu klip, dan namanya diturunkan dari nama file video "
            f"('{sanitize_source_name(str(video))}'). Ganti nama salah satu video. "
            f"(Kalau klip lama memang tidak dipakai lagi, folder itu bisa Anda hapus sendiri.)")


def pick_style(flag: str | Path | None, cfg: PipelineConfig) -> tuple[Path | None, str]:
    """(path YAML style, nama) yang dipakai: --style (nama preset / path) mengalahkan kunci `style` di config (T-406). Satu jalur
    resolver (`config.resolve_style`) dengan stage. Path None = default kode (rough-sketch tanpa berkasnya)."""
    try:
        path = resolve_style(flag if flag is not None else cfg.style)
    except ConfigError as e:
        raise CliError(str(e)) from None
    return path, style_name(path)


def preflight_export_target(cfg: PipelineConfig, video: Path, limit: int | None, style_label: str = DEFAULT_STYLE_NAME) -> None:
    """(c) target export milik video LAIN → berhenti sekarang, bukan di stage terakhir setelah ±78 mnt GPU.
    Pengaman export tidak bisa dilewati --restart. `--limit` menulis <nama>.limitN.mp4 tanpa pengaman → dilewati."""
    if limit is not None:
        return
    target = cfg.paths.out_dir / resolve_filename(cfg.export.filename, str(video.resolve()), style=style_label)
    old = read_manifest(manifest_path_for(target))
    old_src = (old or {}).get("clip", {}).get("source_path")
    if old_src and target.is_file() and not _same_path(old_src, video):
        raise CliError(
            f"{target} sudah ada dan berasal dari video LAIN ({old_src}) — pengaman export tidak ditimpa dan "
            f"tidak bisa dilewati dengan restart. Ganti nama salah satu video, atau ubah export.filename di config.")


def preflight_style(path: Path | None) -> None:
    """(d) style terpilih (HASIL `pick_style`: path YAML, None = default kode) LOLOS validasi — kegagalan di [5] baru terlihat setelah
    ±78 mnt GPU. `render.output_width` divalidasi genap oleh load_style, jadi ukuran output (yuv420p) tidak perlu cek lain."""
    try:
        style = load_style(path)
        paper.validate_texture(style)             # T-404a: gambar kertas ada + terdekode bila dipakai (export di akhir run tidak boleh gagal karena ini)
    except (ConfigError, StageError) as e:
        raise CliError(str(e)) from None


def preflight_svg_target(cfg: PipelineConfig, video: Path, limit: int | None, work_dir: Path,
                         restart: bool, style_label: str = DEFAULT_STYLE_NAME) -> None:
    """(e) out/svg/<nama>/ milik video LAIN, atau penanda rusak (+ --restart tanpa semua SVG identik dengan strokes/)
    → berhenti sekarang (hanya source strokes menyalin SVG; --limit tidak). `restart` = export ikut ber-restart."""
    if limit is not None or cfg.export.source != "strokes":
        return
    try:
        check_svg_owner(svg_dir_for(cfg.paths.out_dir, str(video.resolve()), cfg.export.filename, style_label), str(video.resolve()),
                        work_dir / stylize_stage.STROKES_DIRNAME, restart)
    except StageError as e:
        raise CliError(str(e)) from None


def preflight_tools() -> None:
    """(f) ffmpeg + ffprobe ada (export di akhir run tidak boleh gagal karena ini setelah GPU)."""
    missing = [t for t in ("ffmpeg", "ffprobe") if shutil.which(t) is None]
    if missing:
        raise CliError(f"{', '.join(missing)} tidak ditemukan di PATH. Install ffmpeg (gyan.dev essentials build) "
                       f"dan pastikan folder bin-nya ada di PATH.")


# ── Penghapusan GPU: konfirmasi --yes ──────────────
def _tree_stats(paths: Sequence[Path]) -> tuple[int, int]:
    files = [f for p in paths for f in ([p] if p.is_file() else p.rglob("*") if p.is_dir() else []) if f.is_file()]
    return len(files), sum(f.stat().st_size for f in files)


def _targets(stage: str, ctx: Ctx) -> list[Path]:
    w = ctx.work_dir
    return {"segment": [w / "seg", w / QC_REPORT_FILENAME], "depth": [w / "depth"],
            "stabilize": [w / "stable"], "vectorize": [w / "contours"], "stylize": [w / "strokes"],
            "export": []}[stage]


def deletion_preview(stages: Sequence[str], ctx: Ctx) -> str:
    meta = _read_meta(ctx.work_dir)
    n_frames = meta.get("frame_count") if meta else None
    model = ctx.cfg.segment.model
    lines, total_s = [], 0.0
    for st in stages:
        n, size = _tree_stats(_targets(st, ctx))
        names = ", ".join(p.name for p in _targets(st, ctx))
        est = ""
        if st in GPU_STAGES and n_frames:
            per = GPU_SECONDS_PER_FRAME[st][model] if st == "segment" else GPU_SECONDS_PER_FRAME[st]
            total_s += per * n_frames
            est = f", ulang ≈ {per * n_frames / 60:.1f} mnt GPU ({n_frames} frame × {per} s)"
        lines.append(f"  - {LABELS[st]}: {names} — {n} file, {size / 2**20:.1f} MiB{est}")
    est_total = f"\nEstimasi total waktu GPU untuk mengulang: ≈ {total_s / 60:.1f} mnt." if total_s else ""
    return "\n".join(lines) + est_total


def require_yes(stages: Sequence[str], ctx: Ctx, yes: bool, what: str) -> None:
    """Setiap penghapusan GPU lewat cli wajib --yes. Tanpa --yes: tampilkan yang akan dihapus, exit 1."""
    gpu = [s for s in stages if s in GPU_STAGES]
    if not gpu or yes:
        return
    raise CliError(f"{what} akan MENGHAPUS hasil stage GPU:\n{deletion_preview(stages, ctx)}\n"
                   f"Tidak ada yang dihapus. Tambahkan --yes untuk melanjutkan.")


# ── Menjalankan stage ──────────────────────────────
def _stop(proc) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=TERMINATE_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def run_gpu_stage(stage: str, args: Sequence[str], env: dict | None = None) -> int:
    """Stage GPU = proses sendiri; stdout/stderr diwariskan (pesan VRAM tetap terbaca). Induk menerima
    KeyboardInterrupt / exception → anak di-terminate dan ditunggu selesai sebelum diteruskan."""
    proc = _popen([sys.executable, "-m", MODULES[stage], *args], env=env)
    try:
        return proc.wait()
    except BaseException:
        _stop(proc)
        raise


def run_cpu_stage(stage: str, args: Sequence[str]) -> int:
    """Stage CPU in-process; SystemExit dari argparse stage ditangkap dan menyebut nama stage."""
    try:
        rc = CPU_MAINS[stage](list(args))
    except SystemExit as e:
        rc = e.code if isinstance(e.code, int) else (EXIT_OK if e.code is None else EXIT_PRECONDITION)
        if rc != EXIT_OK:
            print(f"ERROR: stage {LABELS[stage]} menolak argumen (exit {rc}) — lihat pesan di atas", file=sys.stderr)
    return int(rc or 0)


def run_stage(stage: str, args: Sequence[str], ctx: Ctx | None = None) -> int:
    if ctx is not None:
        ctx.state["stage"] = stage
    return run_gpu_stage(stage, args) if stage in GPU_STAGES else run_cpu_stage(stage, args)


def stage_args(stage: str, ctx: Ctx) -> list[str]:
    """argv stage untuk `run`: folder kerja klip + flag yang relevan (tidak ada fallback model otomatis)."""
    a = ["--work-dir", str(ctx.work_dir)]
    if stage == "ingest":
        return [str(ctx.video), *a]
    if ctx.config is not None:
        a += ["--config", str(ctx.config)]
    if stage == "segment" and ctx.seg_model:
        a += ["--seg-model", ctx.seg_model]
    if stage in ("stylize", "export") and ctx.style is not None:
        a += ["--style", str(ctx.style)]
    if stage in ctx.restart:
        a.append("--restart")
    if ctx.limit is not None:
        a += ["--limit", str(ctx.limit)]
    return a


def qc_warning(work_dir: Path) -> str | None:
    """Satu peringatan kalau ada frame gagal QC (stage [3] menanganinya, D-010). run tetap lanjut."""
    try:
        s = json.loads((work_dir / QC_REPORT_FILENAME).read_text(encoding="utf-8"))["summary"]
        n_fail, n_frames, frames = int(s["n_fail"]), int(s["n_frames"]), list(s["fail_frames"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if n_fail == 0:
        return None
    return (f"PERINGATAN: {n_fail}/{n_frames} frame gagal QC "
            f"(indeks: {', '.join(map(str, frames[:QC_INDEX_LIMIT]))}, maks {QC_INDEX_LIMIT} ditampilkan) — "
            f"stage [3] memberinya bobot temporal kecil; run dilanjutkan")


# ── Subperintah ────────────────────────────────────
def _resume_hint(restart_from: str | None) -> str:
    if restart_from:
        return (f"Untuk resume jalankan ulang TANPA --restart-from {restart_from} dan --yes "
                f"(kalau tidak, hasil stage itu dihapus lagi).")
    return "Jalankan ulang perintah yang sama untuk resume."


def check_gpu_inputs(cfg: PipelineConfig, video: Path, work_dir: Path, lo: int, hi: int) -> None:
    """Preview (T-204): seg/ dan depth/ valid untuk jendela [lo, hi) — CPU-only, memakai fungsi validasi stage
    (segment / depth tidak meng-import torch di level modul). Preview TIDAK menjalankan stage GPU."""
    try:
        sclip = segment_stage.load_clip(work_dir)
        dclip = depth_stage.load_clip(work_dir)
    except StageError as e:
        raise CliError(str(e)) from None
    names = sclip.names[lo:hi]
    full = f"{PROG} run {video}"
    limit_note = f"{hi} (K+N)"
    # seg/
    if not sclip.manifest_path.is_file():
        raise CliError(f"preview butuh hasil [2] segment, tetapi {sclip.manifest_path} tidak ada. Preview tidak menjalankan "
                       f"stage GPU — jalankan {full} (penuh) atau {PROG} segment {video} --limit {limit_note}.")
    try:
        sm = json.loads(sclip.manifest_path.read_text(encoding="utf-8"))
        segment_stage._require_same_clip(sclip, sm)
        num_labels = int(sm["num_labels"])
        seg_key = sm["model"]
    except StageError as e:
        raise CliError(str(e)) from None
    except (OSError, ValueError, KeyError, TypeError):
        raise CliError(f"{sclip.manifest_path} rusak / tanpa num_labels, model — jalankan {PROG} segment {video} "
                       f"--restart --yes (MENJALANKAN GPU)") from None
    if seg_key != cfg.segment.model:
        raise CliError(f"seg/ dihitung dengan model {seg_key}, tetapi preview meminta model {cfg.segment.model} "
                       f"(--seg-model / segment.model). Model tidak pernah dicampur dalam satu klip dan preview tidak "
                       f"menjalankan GPU. Jalankan ulang tanpa --seg-model {cfg.segment.model} (atau ubah segment.model "
                       f"menjadi {seg_key}); untuk mengganti model: {PROG} segment {video} --seg-model "
                       f"{cfg.segment.model} --restart --yes (MENJALANKAN GPU).")
    bad = [n for n in names if not sclip.frame_valid(n, num_labels)]
    if bad:
        raise CliError(f"seg/ belum valid untuk jendela: {len(bad)} dari {len(names)} frame, mis. {Path(bad[0]).stem}. "
                       f"Preview tidak menjalankan stage GPU — jalankan {full} (penuh) atau {PROG} segment {video} "
                       f"--limit {limit_note}.")
    # depth/
    if not dclip.manifest_path.is_file():
        raise CliError(f"preview butuh hasil [2c] depth, tetapi {dclip.manifest_path} tidak ada. Preview tidak menjalankan "
                       f"stage GPU — jalankan {full} (penuh) atau {PROG} depth {video} --limit {limit_note}.")
    try:
        depth_stage._require_same_clip(dclip)
    except StageError as e:
        raise CliError(str(e)) from None
    bad = [n for n in dclip.names[lo:hi] if not dclip.frame_valid(n)]
    if bad:
        raise CliError(f"depth/ belum valid untuk jendela: {len(bad)} dari {hi - lo} frame, mis. {Path(bad[0]).stem}. "
                       f"Preview tidak menjalankan stage GPU — jalankan {full} (penuh) atau {PROG} depth {video} "
                       f"--limit {limit_note}.")


def preview_window(a: argparse.Namespace, work_dir: Path) -> tuple[int, int]:
    """(K, N) setelah dicek terhadap frame_count di meta.json (kalau sudah ada)."""
    k, n = a.start or 0, a.preview
    meta = _read_meta(work_dir)
    count = meta.get("frame_count") if meta else None
    if isinstance(count, int) and k + n > count:
        raise CliError(f"jendela preview {k}..{k + n - 1} melewati klip ({count} frame): --from + --preview harus ≤ {count}")
    return k, n


def preview_args(stage: str, ctx: Ctx, k: int, n: int) -> list[str]:
    """argv stage untuk preview (hibrida A): stabilize penuh (resume), vectorize --limit K+N, stylize / export --from K
    --limit N. Tidak ada --restart."""
    a = ["--work-dir", str(ctx.work_dir)]
    if stage == "ingest":
        return [str(ctx.video), *a]
    if ctx.config is not None:
        a += ["--config", str(ctx.config)]
    if stage in ("stylize", "export") and ctx.style is not None:
        a += ["--style", str(ctx.style)]
    if stage == "vectorize":
        a += ["--limit", str(k + n)]
    elif stage in ("stylize", "export"):
        a += ["--from", str(k), "--limit", str(n)]
    return a


PREVIEW_STAGES = ("ingest", "stabilize", "vectorize", "stylize", "export")
# Estimasi pesan rantai vectorize (bukan parameter): terukur T-204 pada klip test 480x854, [4] ±0,078 s/frame (18,4 s untuk
# 235 frame, Tahap 1) sampai ±0,097 s/frame (22,9 s, Tahap 3) — variasi mesin besar; dibulatkan ke 0,1.
VECTORIZE_S_PER_FRAME = 0.1


def vectorize_chain_todo(cfg: PipelineConfig, work_dir: Path, n_prefix: int) -> int | None:
    """Jumlah frame di prefiks 0..n_prefix-1 yang BELUM valid untuk [4] (contours basi / hilang / rantai putus), memakai
    fungsi validasi [4] yang sama dengan run_vectorize. `clip_stats.json` tidak valid → seluruh prefiks dihitung dingin.
    None kalau tidak bisa ditentukan (input [3] bermasalah: stage [4] sendiri yang akan melapor)."""
    vz = vectorize_stage
    try:
        clip = vz.load_clip(work_dir)
        stable = vz.load_stable(clip, cfg)
        params = vz.vectorize_params(cfg)
        stats = vz.load_clip_stats(clip, vz.clip_stats_inputs(params, clip, stable))
        if stats is None:
            return n_prefix
        manifest = vz.build_manifest(cfg, clip, stable, stats)
        if clip.manifest_path.is_file():
            old = json.loads(clip.manifest_path.read_text(encoding="utf-8"))
            if vz.manifest_diff(old, manifest):
                return n_prefix
        selected = clip.names[:n_prefix]
        return n_prefix - sum(vz.scan_chain(clip, selected, clip.indices[:n_prefix], vz.frame_source(manifest)))
    except (StageError, OSError, ValueError):
        return None


def cmd_preview(a: argparse.Namespace) -> int:
    """`run --preview N [--from K]` (T-204): jendela K..K+N-1 → out/<nama>.preview_K-<K+N-1>.mp4. Tanpa GPU / torch."""
    if a.preview < 1:
        raise CliError(f"--preview harus ≥ 1, dapat {a.preview}")
    if a.start is not None and a.start < 0:
        raise CliError(f"--from harus ≥ 0, dapat {a.start}")
    bad = [f for f, v in (("--limit", a.limit is not None), ("--restart-from", a.restart_from is not None),
                          ("--adopt", a.adopt), ("--qc-only", a.qc_only)) if v]
    if bad:
        raise CliError(f"--preview tidak bisa dipakai bersama {', '.join(bad)}")
    cfg = load_cfg(a.config, a.seg_model)
    video = a.video
    ctx = Ctx(video=video, cfg=cfg, work_dir=clip_work_dir(cfg, video), config=a.config, seg_model=a.seg_model)
    preflight_video(video, ctx.work_dir)
    spath, label = pick_style(a.style, cfg)
    ctx.style = spath if a.style is not None else None      # stage memilih sendiri lewat resolver yang sama (kunci `style` di config)
    preflight_style(spath)
    preflight_tools()
    k, n = preview_window(a, ctx.work_dir)
    if _read_meta(ctx.work_dir) is not None:        # gagal cepat sebelum ingest (diulang sesudahnya)
        check_gpu_inputs(cfg, video, ctx.work_dir, k, k + n)
    target = cfg.paths.out_dir / resolve_filename(cfg.export.filename, str(video.resolve()), n, k, label)

    print(f"run {video.name} --preview {n} --from {k}: folder kerja {ctx.work_dir}, keluaran {target}", flush=True)
    timings: list[tuple[str, float]] = []
    try:
        for stage in PREVIEW_STAGES:
            print(f"\n=== {LABELS[stage]} ===", flush=True)
            if stage == "vectorize":
                todo = vectorize_chain_todo(cfg, ctx.work_dir, k + n)
                if todo:
                    print(f"rantai vectorize {todo} frame belum valid, estimasi ±{todo * VECTORIZE_S_PER_FRAME:.1f} s "
                          f"(≈ {VECTORIZE_S_PER_FRAME} s/frame, terukur pada klip test)", flush=True)
            t0 = time.perf_counter()
            rc = run_stage(stage, preview_args(stage, ctx, k, n), ctx)
            timings.append((stage, time.perf_counter() - t0))
            if rc != EXIT_OK:
                print(f"\nERROR: preview berhenti di stage {LABELS[stage]} (exit {rc}). Jalankan ulang perintah yang "
                      f"sama untuk resume.", file=sys.stderr)
                return rc
            if stage == "ingest":                    # frame_count bisa berubah kalau video diganti
                k, n = preview_window(a, ctx.work_dir)
                check_gpu_inputs(cfg, video, ctx.work_dir, k, k + n)
                print("(seg/ dan depth/ valid untuk jendela — tidak ada stage GPU)", flush=True)
    except KeyboardInterrupt:
        print(f"\nERROR: dihentikan (Ctrl+C) di stage {LABELS[ctx.state.get('stage', 'ingest')]}; jalankan ulang "
              f"untuk resume.", file=sys.stderr)
        return EXIT_INTERRUPTED
    except CliError:
        raise
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        print(f"\nERROR: preview berhenti di stage {LABELS[ctx.state.get('stage', 'ingest')]} — "
              f"{type(e).__name__}: {e}. Jalankan ulang perintah yang sama untuk resume.", file=sys.stderr)
        return EXIT_PRECONDITION
    print(f"\nPreview jendela {k}..{k + n - 1} ({n} frame): {target}", flush=True)
    print("Waktu per stage: " + ", ".join(f"{LABELS[s]} {t:.1f} s" for s, t in timings), flush=True)
    return EXIT_OK


def cmd_run(a: argparse.Namespace) -> int:
    if a.preview is not None:
        return cmd_preview(a)
    if a.start is not None:
        raise CliError("--from hanya bersama --preview N (jendela K..K+N-1)")
    if a.adopt or a.qc_only:
        raise CliError("--adopt / --qc-only tidak tersedia di `run` — pakai subperintah stage: "
                       f"{PROG} segment <video> --qc-only|--adopt (depth: --adopt)")
    if a.limit is not None and a.limit < 1:
        raise CliError(f"--limit harus ≥ 1, dapat {a.limit}")
    cfg = load_cfg(a.config, a.seg_model)
    video = a.video
    ctx = Ctx(video=video, cfg=cfg, work_dir=clip_work_dir(cfg, video), config=a.config,
              seg_model=a.seg_model, limit=a.limit, restart=RESTART_SCOPE.get(a.restart_from, ()))
    # Pre-flight CPU-only — sebelum penghapusan, ingest, atau subprocess apa pun
    preflight_video(video, ctx.work_dir)
    spath, label = pick_style(a.style, cfg)
    ctx.style = spath if a.style is not None else None      # stage memilih sendiri lewat resolver yang sama (kunci `style` di config)
    preflight_style(spath)
    preflight_tools()
    preflight_export_target(cfg, video, a.limit, label)
    preflight_svg_target(cfg, video, a.limit, ctx.work_dir, "export" in ctx.restart, label)
    if a.restart_from:
        require_yes(ctx.restart, ctx, a.yes, f"run --restart-from {a.restart_from}")

    print(f"run {video.name}: folder kerja {ctx.work_dir}, output {cfg.paths.out_dir}", flush=True)
    timings: list[tuple[str, float]] = []
    try:
        for stage in STAGES:
            print(f"\n=== {LABELS[stage]} ===", flush=True)
            t0 = time.perf_counter()
            rc = run_stage(stage, stage_args(stage, ctx), ctx)
            timings.append((stage, time.perf_counter() - t0))
            if rc != EXIT_OK:
                print(f"\nERROR: run berhenti di stage {LABELS[stage]} (exit {rc}). {_resume_hint(a.restart_from)}",
                      file=sys.stderr)
                return rc
            if stage == "segment" and (w := qc_warning(ctx.work_dir)):
                print(w, file=sys.stderr, flush=True)
    except KeyboardInterrupt:
        print(f"\nERROR: dihentikan (Ctrl+C) di stage {LABELS[ctx.state.get('stage', 'ingest')]}; "
              f"{_resume_hint(a.restart_from)}", file=sys.stderr)
        return EXIT_INTERRUPTED
    except Exception as e:  # noqa: BLE001 — anak sudah di-terminate di run_gpu_stage
        traceback.print_exc()
        print(f"\nERROR: run berhenti di stage {LABELS[ctx.state.get('stage', 'ingest')]} — "
              f"{type(e).__name__}: {e}. {_resume_hint(a.restart_from)}", file=sys.stderr)
        return EXIT_PRECONDITION
    print("\nWaktu per stage: " + ", ".join(f"{LABELS[s]} {t:.1f} s" for s, t in timings), flush=True)
    return EXIT_OK


def cmd_stage(a: argparse.Namespace) -> int:
    """Subperintah stage: sisa argumen diteruskan apa adanya ke main() stage (flag tidak diparse ulang)."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=None)
    pre.add_argument("--seg-model", default=None)
    pre.add_argument("--restart", action="store_true")
    pre.add_argument("--yes", action="store_true")
    pre.add_argument("--work-dir", default=None)
    known, rest = pre.parse_known_args(a.rest)
    stage = a.stage
    if known.work_dir is not None:
        raise CliError("--work-dir ditentukan oleh cli dari nama video (<work_dir>/clips/<nama>/); "
                       "ubah paths.work_dir di config untuk memindahkan semuanya", EXIT_USAGE)
    if known.seg_model and stage != "segment":
        raise CliError("--seg-model hanya untuk subperintah segment", EXIT_USAGE)
    if known.restart and stage == "ingest":
        raise CliError("ingest selalu menulis ulang frames/ + meta.json — tidak ada --restart", EXIT_USAGE)

    cfg = load_cfg(known.config, known.seg_model)
    ctx = Ctx(video=a.video, cfg=cfg, work_dir=clip_work_dir(cfg, a.video), config=known.config,
              seg_model=known.seg_model)
    if stage == "ingest":
        preflight_video(a.video, ctx.work_dir)
    if known.restart:
        require_yes([stage], ctx, known.yes, f"{PROG} {stage} --restart")

    args = ["--work-dir", str(ctx.work_dir)]
    if stage == "ingest":
        args = [str(a.video), *args]
    elif known.config is not None:
        args += ["--config", str(known.config)]
    if known.seg_model:
        args += ["--seg-model", known.seg_model]
    if known.restart:
        args.append("--restart")
    rc = run_stage(stage, [*args, *rest], ctx)
    return rc


def cmd_download(a: argparse.Namespace) -> int:
    """Unduh checkpoint (online): Sapiens2-seg (default 0.8b; --seg-model 0.4b = fallback) + Depth Anything V2
    Small + model card-nya. Subprocess mewarisi environment apa adanya — TIDAK memaksa HF_HUB_OFFLINE=1
    (run biasa offline; unduhan butuh online)."""
    load_cfg(a.config, a.seg_model)  # config tidak valid → berhenti sebelum mengunduh
    common = ["--download"] + (["--config", str(a.config)] if a.config is not None else [])
    plan = {"segment": common + (["--seg-model", a.seg_model] if a.seg_model else []), "depth": common}
    for stage, args in plan.items():
        print(f"=== download {LABELS[stage]} ===", flush=True)
        rc = run_stage(stage, args)
        if rc != EXIT_OK:
            print(f"ERROR: download berhenti di {LABELS[stage]} (exit {rc})", file=sys.stderr)
            return rc
    return EXIT_OK


# ── Argparse ───────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=PROG, description="Rotoscope Animation Builder — video → animasi sketsa")
    sub = p.add_subparsers(dest="cmd", required=True, metavar="<subperintah>")

    r = sub.add_parser("run", help="ingest → segment → depth → stabilize → vectorize → stylize → export untuk satu video")
    r.add_argument("video", type=Path)
    r.add_argument("--config", type=Path, default=None)
    r.add_argument("--style", type=str, default=None,
                   help="nama preset (rough-sketch, clean-line, heavy-marker, pencil-light = configs/styles/<nama>.yaml) atau path YAML "
                        "style: stage [5] stylize (garis) dan [6] export (kertas bertekstur / vignette). Default: kunci `style` di config")
    r.add_argument("--seg-model", choices=("0.8b", "0.4b"), default=None,
                   help="model segmentasi untuk SELURUH klip (tidak pernah fallback otomatis)")
    r.add_argument("--limit", type=int, default=None, help="hanya N frame pertama (export → <nama>.limitN.mp4)")
    r.add_argument("--preview", type=int, default=None, metavar="N",
                   help="render jendela N frame (K..K+N-1, K = --from, default 0) → out/<nama>.preview_K-<K+N-1>.mp4; "
                        "tanpa GPU: seg/ dan depth/ harus sudah valid untuk jendela")
    r.add_argument("--from", dest="start", type=int, default=None, metavar="K",
                   help="frame awal jendela --preview (hanya bersama --preview)")
    r.add_argument("--restart-from", choices=STAGES, default=None,
                   help="hapus hasil stage ini lalu jalankan ulang (ingest = semua stage; segment / depth = hanya "
                        "stage itu — depth tidak bergantung pada segment). Stage GPU wajib --yes; vectorize / "
                        "stylize / export tanpa --yes")
    r.add_argument("--yes", action="store_true", help="setujui penghapusan hasil stage GPU")
    r.add_argument("--adopt", action="store_true", help=argparse.SUPPRESS)
    r.add_argument("--qc-only", action="store_true", help=argparse.SUPPRESS)
    r.set_defaults(func=cmd_run)

    for stage, text in (("ingest", "[1] video → frames/ + meta.json"),
                        ("segment", "[2] Sapiens2-seg (GPU, proses sendiri); flag: --seg-model --restart --yes "
                                    "--limit --qc-only --adopt"),
                        ("depth", "[2c] Depth Anything V2 Small (GPU, proses sendiri); flag: --restart --yes "
                                  "--limit --adopt"),
                        ("stabilize", "[3] peta grup + kedalaman ternormalisasi; flag: --restart --limit"),
                        ("vectorize", "[4] stable/groups → contours/ (siluet, lubang, batas grup, garis oklusi); "
                                      "flag: --restart --limit"),
                        ("stylize", "[5] contours/ → strokes/ (SVG + PNG garis polos); flag: --style --restart "
                                    "--limit"),
                        ("export", "[6] strokes/ (atau stable/groups) → out/<nama>.mp4 + out/svg/<nama>/; flag: "
                                   "--style --restart --limit")):
        s = sub.add_parser(stage, help=text, description=text)
        s.add_argument("video", type=Path, help="video sumber (menentukan folder kerja klip)")
        s.add_argument("rest", nargs=argparse.REMAINDER, help="flag stage, diteruskan apa adanya")
        s.set_defaults(func=cmd_stage, stage=stage)

    d = sub.add_parser("download", help="unduh checkpoint (online, sekali jalan)",
                       description="Mengunduh Sapiens2-seg (default 0.8b; --seg-model 0.4b untuk fallback) dan "
                                   "Depth Anything V2 Small + model card-nya ke cache Hugging Face.")
    d.add_argument("--config", type=Path, default=None)
    d.add_argument("--seg-model", choices=("0.8b", "0.4b"), default=None)
    d.set_defaults(func=cmd_download)
    return p


def main(argv: list[str] | None = None) -> int:
    reconfigure_stdio()
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except CliError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return e.code
    except KeyboardInterrupt:
        print("ERROR: dihentikan (Ctrl+C); jalankan ulang untuk resume", file=sys.stderr)
        return EXIT_INTERRUPTED


if __name__ == "__main__":
    sys.exit(main())
