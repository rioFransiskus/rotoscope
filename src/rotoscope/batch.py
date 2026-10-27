"""Subperintah `batch` (T-603; BUKAN stage): banyak video → `run` berurutan, satu proses per klip.

    python -m rotoscope batch <video|folder> ... [--list BERKAS] [--yes] [--skip-pending] [--config P] [--style P] [--seg-model 0.8b|0.4b]

Tanpa `--yes` = RENCANA: CPU saja. Ingest (sama dengan `run`, opsi A T-108) + status crop / target / identitas / stage GPU +
estimasi GPU dan disk. Rencana tidak memanggil stage GPU dan tidak menghapus apa pun: `isolate.run_isolate(dry_run=True)` hanya
dipanggil bila target/points.json ada (tanpa titik tidak ada pemanggilan sama sekali → tidak ada jalur hapus isolate/).
Dengan `--yes` = JALAN: rencana dicek ulang; klip "perlu tindakan" → berhenti sebelum GPU (exit 4) kecuali `--skip-pending`;
disk dicek; VRAM dicek SEKALI (nvidia-smi, ambang = segment.check_vram); lalu tiap klip = proses `run <video> ... --yes`.

Exit rencana: 0 siap | 4 ada tindakan | 1 gagal (termasuk disk kurang). Exit jalan: 0 semua selesai | 1 ada gagal |
4 ada dilewati tanpa gagal / masih perlu tindakan | 130 Ctrl+C. Gagal + dilewati sekaligus → 1. Ringkasan JSON (mode jalan) di
<paths.work_dir>/batch/ringkasan_<UTC>.json, juga saat Ctrl+C dan saat berhenti exit 4. Waktu hanya dicatat, tidak dipakai keputusan.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import io
import json
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from rotoscope import cli as cli_mod
from rotoscope import crop as crop_mod
from rotoscope import depth as depth_stage
from rotoscope import isolate as isolate_stage
from rotoscope import segment as segment_stage
from rotoscope.config import ConfigError, PipelineConfig, load_class_names
from rotoscope.export import resolve_filename
from rotoscope.ingest import DEFAULT_TARGET_FPS, META_FILENAME, make_trim
from rotoscope.stage_common import (
    EXIT_CROP_DECISION, EXIT_OK, EXIT_PRECONDITION, StageError, clip_identity, cli_cmd, quote_arg, reconfigure_stdio,
    set_message_config, trim_label,
)

EXIT_ACTION = EXIT_CROP_DECISION          # 4: ada klip perlu tindakan / dilewati (sama dengan exit 4 `run`, T-607)
EXIT_INTERRUPTED = cli_mod.EXIT_INTERRUPTED
FORBIDDEN_FLAGS = ("--restart-from", "--preview", "--from", "--adopt", "--qc-only", "--limit")
MIB = 2**20
GIB = 2**30
SUMMARY_FORMAT = "rotoscope-batch/1"
SUMMARY_DIRNAME = "batch"
DOWNSTREAM_MANIFESTS = ("stable", "contours", "strokes")
DETEKSI_FILENAME = "deteksi.json"
TRIM_FLAGS = ("--start", "--end")
MODE_LABEL = {"none": "tanpa crop (--none)", "saran": "saran diterima (crop ya)", "manual": "kotak manual"}
_popen = subprocess.Popen                 # seam untuk test


class BatchError(Exception):
    def __init__(self, message: str, code: int = EXIT_PRECONDITION):
        super().__init__(message)
        self.code = code


@dataclass
class Spec:
    path: str
    trim: dict | None
    line: int | None                       # nomor baris di --list; None = argumen CLI / isi folder


@dataclass
class Row:
    spec: Spec
    video: Path
    name: str
    work_dir: Path
    crop_dir: Path
    status: str = "siap"                    # siap | perlu tindakan | gagal | selesai | dilewati | dihentikan | belum
    frames: int | None = None
    wh: tuple[int, int] | None = None
    crop: str = "-"
    target: str = "-"
    identity: str = "-"
    stages: str = "-"
    gpu_s: float = 0.0
    disk_mib: float = 0.0
    problems: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    out_mp4: str | None = None
    exit_code: int | None = None
    seconds: float | None = None


def fmt(x: float, nd: int = 1) -> str:
    """Angka dengan koma desimal (keluaran untuk pengguna Indonesia)."""
    return f"{x:.{nd}f}".replace(".", ",")


# ── Parsing masukan ───────────────────────────────
def _at(line: int | None) -> str:
    return f" baris {line}" if line is not None else ""


def _unquote(tok: str, line: int | None) -> str:
    """Kutip ganda luar yang berpasangan dilepas. Kutip tidak berpasangan / di tengah path → BatchError (exit 1)."""
    if tok.startswith('"'):
        if len(tok) >= 2 and tok.endswith('"'):
            tok = tok[1:-1]
        else:
            raise BatchError(f"--list{_at(line)}: kutip ganda tidak berpasangan pada {tok!r} — tutup dengan kutip ganda yang "
                             f"sama, mis. \"D:\\Meme Lucu\\video 1.mp4\"")
    if '"' in tok:
        raise BatchError(f"--list{_at(line)}: kutip ganda di tengah path {tok!r} — bungkus seluruh path dengan kutip ganda")
    if not tok.strip():
        raise BatchError(f"--list{_at(line)}: path kosong")
    return tok


def parse_list(path: Path) -> list[Spec]:
    """`<path> [--start S --end E]` per baris; `#` = komentar; baris kosong dilewati. Path boleh berspasi (kutip ganda)."""
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError as e:
        raise BatchError(f"--list tidak bisa dibaca: {path} ({e.strerror or e})") from None
    specs: list[Spec] = []
    for no, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            toks = shlex.split(line, posix=False)      # posix=False: backslash Windows tetap; tanda kutip TETAP ada di token
        except ValueError as e:
            raise BatchError(f"{path} baris {no}: kutip tidak berpasangan ({e}) — periksa kutip ganda pada path") from None
        if not toks or toks[0].startswith("--"):
            raise BatchError(f"{path} baris {no}: path video harus di depan, mis. samples\\klip4.mp4 --start 0 --end 2")
        video = _unquote(toks[0], no)
        start = end = None
        i = 1
        while i < len(toks):
            flag = toks[i]
            if flag not in TRIM_FLAGS or i + 1 >= len(toks):
                raise BatchError(f"{path} baris {no}: argumen tidak dikenal {flag!r} — hanya --start S dan --end E "
                                 f"(path berspasi wajib di dalam kutip ganda)")
            if (flag == "--start" and start is not None) or (flag == "--end" and end is not None):
                raise BatchError(f"{path} baris {no}: {flag} dobel")
            try:
                value = float(toks[i + 1])
            except ValueError:
                raise BatchError(f"{path} baris {no}: {flag} butuh angka detik, dapat {toks[i + 1]!r}") from None
            if flag == "--start":
                start = value
            else:
                end = value
            i += 2
        try:
            trim = make_trim(start, end, DEFAULT_TARGET_FPS)
        except ValueError as e:
            raise BatchError(f"{path} baris {no}: {e}") from None
        specs.append(Spec(path=video, trim=trim, line=no))
    return specs


def expand_inputs(paths: list[str], exts: tuple[str, ...]) -> list[Spec]:
    """Argumen CLI: file (apa adanya) atau folder (tidak rekursif, urut nama, hanya ekstensi `video_exts`)."""
    specs: list[Spec] = []
    for p in paths:
        cand = Path(p)
        if cand.is_dir():
            files = sorted((f for f in cand.iterdir() if f.is_file() and f.suffix.lower() in exts), key=lambda f: f.name)
            specs += [Spec(path=str(f), trim=None, line=None) for f in files]
        elif cand.is_file():
            specs.append(Spec(path=str(cand), trim=None, line=None))
        else:
            raise BatchError(f"input tidak ditemukan: {p}")
    return specs


def build_rows(specs: list[Spec], cfg: PipelineConfig) -> list[Row]:
    """Baris rencana. Dua masukan dengan folder klip sama (stem sama dari folder berbeda, atau baris dobel) → ditolak SEBELUM apa pun."""
    rows: list[Row] = []
    seen: dict[str, Spec] = {}
    for sp in specs:
        video = Path(sp.path)
        work_dir = cli_mod.clip_work_dir(cfg, video, sp.trim)
        key = work_dir.name.casefold()
        if key in seen:
            prev = seen[key]
            raise BatchError(f"klip duplikat: {sp.path}{_at(sp.line)} dan {prev.path}{_at(prev.line)} memakai folder kerja yang sama "
                             f"({work_dir.name}). Satu klip = satu folder; ganti nama salah satu video atau hapus salah satu baris.")
        seen[key] = sp
        rows.append(Row(spec=sp, video=video, name=work_dir.name, work_dir=work_dir,
                        crop_dir=cli_mod.crop_dir_of(cfg, video)))
    return rows


# ── Rencana (CPU) ─────────────────────────────────
def gpu_memory_mib() -> tuple[float, float] | None:
    """(free, total) MiB dari nvidia-smi — SATU panggilan, tanpa torch. None = nvidia-smi tidak ada / gagal."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0 or not out.stdout.strip():
        return None
    try:
        free, total = (float(x) for x in out.stdout.strip().splitlines()[0].split(",")[:2])
    except ValueError:
        return None
    return free, total


def free_bytes(path: Path) -> int:
    p = path
    while not p.exists() and p.parent != p:
        p = p.parent
    return shutil.disk_usage(p).free


def _read_json(path: Path) -> dict | None:
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return d if isinstance(d, dict) else None


def _run_cmd(row: Row, args: argparse.Namespace, restart_from: str | None = None) -> str:
    """Perintah `run` siap tempel: trim + --config lewat cli_cmd; --style / --seg-model dari argumen batch; --yes bila restart."""
    cmd = cli_cmd("run", row.work_dir)
    if args.style:
        cmd += f" --style {quote_arg(args.style)}"
    if args.seg_model:
        cmd += f" --seg-model {args.seg_model}"
    return cmd + (f" --restart-from {restart_from} --yes" if restart_from else "")


def _ingest_quiet(row: Row, cfg: PipelineConfig, args: argparse.Namespace) -> None:
    """[1] ingest seperti `run` (argv sama dengan cli.ingest_args). Keluarannya disembunyikan; galat → BatchError."""
    ctx = cli_mod.Ctx(video=row.video, cfg=cfg, work_dir=row.work_dir, config=args.config, trim=row.spec.trim,
                      crop_dir=row.crop_dir)
    buf_out, buf_err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
        rc = cli_mod.run_cpu_stage("ingest", cli_mod.ingest_args(ctx))
    if rc != EXIT_OK:
        tail = [ln for ln in buf_err.getvalue().splitlines() if ln.strip()][-3:]
        raise BatchError(f"[1] ingest gagal (exit {rc}): " + " | ".join(tail))


def _crop_status(row: Row, cfg: PipelineConfig) -> bool:
    """Status crop. Return True bila perlu tindakan (saran belum diputuskan)."""
    kotak = crop_mod.read_kotak(row.crop_dir, quote_arg(row.video.resolve()))
    if kotak is not None:
        row.crop = f"diputuskan: {MODE_LABEL[kotak['mode']]}"
        return False
    if crop_mod.pending_saran(row.crop_dir) is not None:
        row.crop = "BELUM diputuskan"
        v = quote_arg(row.video.resolve())
        row.problems.append("saran crop belum diputuskan")
        row.actions += [f"crop: {crop_mod.crop_cmd(v, flag)}" for flag in ("--accept", "--none", "--box X Y W H")]
        return True
    if not cfg.ingest.crop.suggest:
        row.crop = "tanpa deteksi (crop.suggest false)"
        return False
    det = _read_json(row.crop_dir / DETEKSI_FILENAME)
    row.crop = "tanpa pita" if det is not None and not det.get("pita_terdeteksi") else "tanpa saran"
    return False


def _target_status(row: Row, cfg: PipelineConfig) -> tuple[float, bool]:
    """Target: tidak ada / ada / BASI. Return (estimasi GPU [1b] detik, perlu tindakan?). Tanpa titik → isolate tidak dipanggil."""
    if not isolate_stage.points_present(row.work_dir):
        row.target = "tidak ada"
        return 0.0, False
    clip_cfg = dataclasses.replace(cfg, paths=dataclasses.replace(cfg.paths, work_dir=row.work_dir))   # isolate baca work_dir dari cfg
    try:
        res = isolate_stage.run_isolate(clip_cfg, dry_run=True, log=lambda m: None)     # dry-run: tanpa tulis / hapus
    except StageError as e:
        row.target = "BASI"
        row.problems.append(f"target basi: {e}")
        row.actions.append(f"pick ulang: {cli_cmd('pick', row.work_dir)}")
        return 0.0, True
    if res["status"] == "up-to-date":
        row.target = "ada (up-to-date)"
        return 0.0, False
    if res["status"] == "dry-run" and res["state"] == "ok":
        row.target = "ada (up-to-date)"
        return 0.0, False
    row.target = "ada, [1b] perlu dihitung"
    return float(res.get("estimate_s", 0.0)), False


def _seg_depth_status(row: Row, cfg: PipelineConfig, args: argparse.Namespace) -> tuple[float, bool]:
    """Identitas (T-108) + model; lalu frame [2]/[2c] yang belum valid (CPU; tanpa torch). Return (GPU detik, perlu tindakan?)."""
    ident = clip_identity(row.work_dir)
    seg = segment_stage.load_clip(row.work_dir)
    dep = depth_stage.load_clip(row.work_dir)
    sm, dm = _read_json(seg.manifest_path), _read_json(dep.manifest_path)
    blocked: dict[str, str] = {}                       # restart → alasan (satu perintah per alasan)
    if sm is None and seg.seg_dir.is_dir() and any(seg.seg_dir.iterdir()):
        blocked["segment"] = "[2] seg/ tanpa manifest (asal tidak diketahui)"
    elif sm is not None and sm.get("clip") != ident:
        blocked["ingest"] = "[2] seg/ milik identitas lain (crop / trim berubah, T-108)"
    elif sm is not None and sm.get("model") != cfg.segment.model:
        blocked["segment"] = f"[2] seg/ dibuat dengan {sm.get('model')}, config {cfg.segment.model}"
    if dm is not None and dm.get("clip") != ident:
        blocked.setdefault("ingest", "[2c] depth/ milik identitas lain (crop / trim berubah, T-108)")
    if blocked:
        row.identity = "BASI"
        for restart, why in blocked.items():
            row.problems.append(why)
            row.actions.append(f"run ulang dari {restart} (dijalankan Rio sendiri): {_run_cmd(row, args, restart)}")
        return 0.0, True
    row.identity = "ok"
    num = int(sm["num_labels"]) if sm is not None else len(load_class_names())
    seg_todo = sum(not seg.frame_valid(nm, num) for nm in seg.names)
    dep_todo = sum(not dep.frame_valid(nm) for nm in dep.names)
    row.stages = f"[2] {seg_todo} f, [2c] {dep_todo} f"
    gpu = (seg_todo * cli_mod.GPU_SECONDS_PER_FRAME["segment"][cfg.segment.model]
           + dep_todo * cli_mod.GPU_SECONDS_PER_FRAME["depth"])
    return gpu, False


def plan_row(row: Row, cfg: PipelineConfig, args: argparse.Namespace, label: str) -> None:
    """Isi satu baris rencana. Galat per klip → status gagal; klip lain tetap diperiksa."""
    try:
        cli_mod.preflight_video(row.video, row.work_dir, row.spec.trim)
        cli_mod.preflight_export_target(cfg, row.video, None, label, row.spec.trim)
        cli_mod.preflight_svg_target(cfg, row.video, None, row.work_dir, False, label, row.spec.trim)
        _ingest_quiet(row, cfg, args)
        meta = _read_json(row.work_dir / META_FILENAME) or {}
        row.frames = int(meta["frame_count"])
        row.wh = (int(meta["working_width"]), int(meta["working_height"]))
        row.out_mp4 = str(cfg.paths.out_dir / resolve_filename(cfg.export.filename, str(row.video.resolve()), style=label,
                                                              trim=trim_label(row.spec.trim)))
        blocked = [_crop_status(row, cfg)]
        iso_s, b = _target_status(row, cfg)
        blocked.append(b)
        gpu_s, b = _seg_depth_status(row, cfg, args)
        blocked.append(b)
        row.gpu_s = gpu_s + iso_s
        if any(blocked):
            row.status = "perlu tindakan"
            return
        if row.gpu_s > 0 or not all((row.work_dir / d / "manifest.json").is_file() for d in DOWNSTREAM_MANIFESTS):
            row.disk_mib = float(cfg.batch.disk_mib_per_mpx) * (row.wh[0] * row.wh[1] / 1e6) * row.frames
    except (cli_mod.CliError, BatchError, StageError, ConfigError, OSError, ValueError, KeyError, TypeError) as e:
        row.status = "gagal"
        row.problems.append(str(e))
    except Exception as e:  # noqa: BLE001 — satu klip rusak tidak menghentikan rencana; pesan lengkap tetap tampil
        row.status = "gagal"
        row.problems.append(f"{type(e).__name__}: {e}")


def plan(rows: list[Row], cfg: PipelineConfig, args: argparse.Namespace) -> None:
    cli_mod.preflight_tools()
    spath, label = cli_mod.pick_style(args.style, cfg)
    cli_mod.preflight_style(spath)
    for row in rows:
        plan_row(row, cfg, args, label)


def disk_check(rows: list[Row], cfg: PipelineConfig) -> tuple[float, float, float]:
    """(butuh GiB, bebas GiB, cadangan GiB) untuk klip SIAP. OK bila bebas ≥ butuh + cadangan."""
    need = sum(r.disk_mib for r in rows if r.status == "siap") * MIB / GIB
    return need, free_bytes(cfg.paths.work_dir) / GIB, float(cfg.batch.disk_reserve_gib)


def disk_ok(need: tuple[float, float, float]) -> bool:
    return need[1] >= need[0] + need[2]


def gpu_text(gpu_s: float) -> str:
    return f"{fmt(gpu_s / 60)} mnt" if gpu_s > 0 else "0"


NOT_COUNTED = ("dilewati", "perlu tindakan", "gagal")      # tidak masuk total GPU / disk → kolomnya "—" (T-603 keputusan 2)
EXIT_MEANING = {EXIT_OK: "semua klip selesai", EXIT_PRECONDITION: "ada klip gagal (atau prasyarat batch gagal)",
                EXIT_ACTION: "ada klip dilewati (perlu tindakan / --skip-pending), tidak ada yang gagal",
                EXIT_INTERRUPTED: "dihentikan (Ctrl+C)"}


def render_plan(rows: list[Row], hint: str, need: tuple[float, float, float], code: int | None = None) -> str:
    """Tabel rencana. code None = mode --yes: tanpa baris "Keluar" / "Untuk menjalankan" (keputusan Rio no. 2)."""
    head = "RENCANA batch — {} klip · CPU saja · tidak ada GPU, tidak ada yang dihapus" if code is not None \
        else "RENCANA batch — {} klip (mode --yes; dicek ulang sebelum GPU)"
    out = [head.format(len(rows)), ""]
    out.append(f"{'#':>2}  {'klip':<22} {'frame':>5}  {'crop':<27} {'target':<24} {'seg/depth':<22} {'GPU':>9} {'disk baru':>10}  status")
    for i, r in enumerate(rows, 1):
        counted = r.status not in NOT_COUNTED
        disk = ("0" if not r.disk_mib else f"{fmt(r.disk_mib / 1024, 2)} GiB") if counted else "—"
        frames = "-" if r.frames is None else str(r.frames)
        gpu = gpu_text(r.gpu_s) if counted else "—"
        out.append(f"{i:>2}  {r.name[:22]:<22} {frames:>5}  {r.crop[:27]:<27} {r.target[:24]:<24} {r.stages[:22]:<22} "
                   f"{gpu:>9} {disk:>10}  {r.status}")
    acts = [(r, a) for r in rows if r.status != "gagal" for a in r.actions]
    if acts:
        out += ["", "Tindakan (urutan: crop dulu, lalu pick / run; yang memakai GPU memuat --yes; batch TIDAK menjalankannya):"]
        out += [f"  {n}. [{r.name}] {a}" for n, (r, a) in enumerate(acts, 1)]
    alasan = [r for r in rows if r.status == "perlu tindakan"]
    if alasan:
        out += ["", "Alasan perlu tindakan:"]
        out += [f"  - {r.name}: {p}" for r in alasan for p in r.problems]
    gagal = [r for r in rows if r.status == "gagal"]
    if gagal:
        out += ["", "Klip gagal (tidak dijalankan):"]
        out += [f"  - {r.name}: {p}" for r in gagal for p in r.problems]
    out += ["", f"Disk: butuh ≈ {fmt(need[0], 2)} GiB, bebas {fmt(need[1])} GiB, cadangan {fmt(need[2], 0)} GiB → "
                f"{'OK' if disk_ok(need) else 'KURANG'}"]
    gpu_total = sum(r.gpu_s for r in rows if r.status == "siap")
    frames_new = sum(r.frames or 0 for r in rows if r.status == "siap" and r.gpu_s > 0)
    out.append(f"GPU: ≈ {gpu_text(gpu_total)} (perkiraan, + muat model per klip); klip dengan frame GPU baru: {frames_new} f.")
    if code is None:
        return "\n".join(out)
    out.append(f"Keluar: exit {code}")
    if code == EXIT_OK:
        out.append(f"Untuk menjalankan: {hint} --yes")
    elif code == EXIT_ACTION:
        out.append(f"Setelah tindakan di atas, jalankan rencana ulang: {hint}")
    return "\n".join(out)


def render_summary(rows: list[Row], code: int, path: Path | None) -> str:
    """Ringkasan akhir mode --yes (keputusan Rio no. 1): tabel per klip, arti exit, alasan + tindakan klip dilewati, path JSON."""
    out = ["", f"RINGKASAN batch — {len(rows)} klip", ""]
    out.append(f"{'#':>2}  {'klip':<22} {'status':<16} {'exit':>4} {'waktu':>9}  output MP4")
    for i, r in enumerate(rows, 1):
        exit_s = "—" if r.exit_code is None else str(r.exit_code)
        waktu = "—" if r.seconds is None else f"{fmt(r.seconds / 60)} mnt"
        mp4 = r.out_mp4 if r.status == "selesai" and r.out_mp4 else "—"
        out.append(f"{i:>2}  {r.name[:22]:<22} {r.status:<16} {exit_s:>4} {waktu:>9}  {mp4}")
    out.append("")
    out.append(f"Exit {code} = {EXIT_MEANING.get(code, 'tidak dikenal')}")
    dilewati = [r for r in rows if r.status == "dilewati"]
    if dilewati:
        out += ["", "Klip dilewati (alasan + perintah tindakan; batch TIDAK menjalankannya):"]
        for r in dilewati:
            out += [f"  - {r.name}: {p}" for p in r.problems]
            out += [f"    tindakan: {a}" for a in r.actions]
    gagal = [r for r in rows if r.status == "gagal"]
    if gagal:
        out += ["", "Klip gagal:"]
        out += [f"  - {r.name}: {p}" for r in gagal for p in r.problems]
    if path is not None:
        out += ["", f"Ringkasan JSON: {path}"]
    return "\n".join(out)


def plan_exit(rows: list[Row], need: tuple[float, float, float]) -> int:
    if any(r.status == "gagal" for r in rows) or not disk_ok(need):
        return EXIT_PRECONDITION
    if any(r.status == "perlu tindakan" for r in rows):
        return EXIT_ACTION
    return EXIT_OK


# ── Jalan (GPU) ───────────────────────────────────
def child_argv(row: Row, args: argparse.Namespace) -> list[str]:
    a = [sys.executable, "-m", "rotoscope", "run", str(row.video), *cli_mod.trim_args(row.spec.trim)]
    if args.config:
        a += ["--config", str(args.config)]
    if args.style:
        a += ["--style", args.style]
    if args.seg_model:
        a += ["--seg-model", args.seg_model]
    return a + ["--yes"]


def check_vram(cfg: PipelineConfig) -> None:
    """Sekali sebelum klip GPU pertama. Ambang = segment.check_vram (model terpilih). nvidia-smi gagal → peringatan, lanjut."""
    got = gpu_memory_mib()
    if got is None:
        print("PERINGATAN: nvidia-smi tidak ada / gagal — cek VRAM dilewati (segment tetap mengecek sendiri).", flush=True)
        return
    free, total = got
    try:
        segment_stage.check_vram(free, total, cfg.segment.vram_min_free_mib[cfg.segment.model], cfg.segment.model)
    except StageError as e:
        raise BatchError(f"{e}\nTutup browser / aplikasi berat (game, editor video), lalu jalankan ulang perintah batch.") from None


def run_jobs(rows: list[Row], args: argparse.Namespace) -> int:
    """Klip SIAP berurutan, satu proses `run` per klip. Return 0 (selesai) atau 130 (Ctrl+C; klip berikutnya belum jalan)."""
    todo = [r for r in rows if r.status == "siap"]
    for i, row in enumerate(todo, 1):
        sisa = sum(r.gpu_s for r in todo[i - 1:]) / 60
        print(f"\n[klip {i}/{len(todo)}] {row.name} — mulai — perkiraan sisa batch ≈ {fmt(sisa)} mnt GPU", flush=True)
        t0 = time.perf_counter()
        proc = _popen(child_argv(row, args))
        try:
            rc = proc.wait()
        except KeyboardInterrupt:
            cli_mod._stop(proc)
            row.status, row.seconds = "dihentikan", round(time.perf_counter() - t0, 1)
            return EXIT_INTERRUPTED
        row.seconds, row.exit_code = round(time.perf_counter() - t0, 1), rc
        if rc == EXIT_INTERRUPTED:
            row.status = "dihentikan"
            return EXIT_INTERRUPTED
        row.status = "selesai" if rc == EXIT_OK else "gagal"
        print(f"[klip {i}/{len(todo)}] {row.name} — {row.status} (exit {rc}, {fmt(row.seconds)} s)", flush=True)
    return EXIT_OK


def write_summary(cfg: PipelineConfig, rows: list[Row], exit_code: int, started: str) -> Path:
    d = cfg.paths.work_dir / SUMMARY_DIRNAME
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"ringkasan_{started}.json"
    counts: dict[str, int] = {}
    for r in rows:
        counts[r.status] = counts.get(r.status, 0) + 1
    data = {"format": SUMMARY_FORMAT, "started_utc": started, "exit_code": exit_code, "counts": dict(sorted(counts.items())),
            "klip": [{"nama": r.name, "video": str(r.video), "trim": r.spec.trim, "status": r.status,
                      "exit_code": r.exit_code, "detik": r.seconds, "work_dir": str(r.work_dir), "out_mp4": r.out_mp4}
                     for r in rows]}
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    return path


def final_code(rows: list[Row]) -> int:
    """Gagal → 1 (juga bila ada yang dilewati); dilewati tanpa gagal → 4; selain itu 0 (keputusan Rio B)."""
    if any(r.status == "gagal" for r in rows):
        return EXIT_PRECONDITION
    if any(r.status == "dilewati" for r in rows):
        return EXIT_ACTION
    return EXIT_OK


def execute(rows: list[Row], cfg: PipelineConfig, args: argparse.Namespace, hint: str, started: str) -> int:
    """Jalan (--yes): rencana dicek ulang → perlu tindakan = dilewati (berhenti sebelum GPU bila tanpa --skip-pending) → disk →
    VRAM → klip berurutan. Ringkasan (tabel, arti exit, alasan + tindakan, path JSON) selalu dicetak di akhir."""
    code = EXIT_OK
    try:
        plan(rows, cfg, args)
        pending = [r for r in rows if r.status == "perlu tindakan"]
        if pending and not args.skip_pending:
            for r in pending:
                r.status = "dilewati"
            print(render_plan(rows, hint, disk_check(rows, cfg)), flush=True)
            print(f"\nBerhenti SEBELUM GPU: {len(pending)} klip perlu tindakan (lihat ringkasan). Jalankan tindakan, lalu rencana "
                  f"ulang; atau tambahkan --skip-pending.", flush=True)
            code = EXIT_ACTION
        else:
            for r in pending:
                r.status = "dilewati"
            need = disk_check(rows, cfg)
            print(render_plan(rows, hint, need), flush=True)
            if not disk_ok(need):
                raise BatchError(f"disk kurang: butuh ≈ {fmt(need[0], 2)} GiB + cadangan {fmt(need[2], 0)} GiB, bebas "
                                 f"{fmt(need[1])} GiB. Tidak ada yang dihitung.")
            if any(r.status == "siap" and r.gpu_s > 0 for r in rows):
                check_vram(cfg)
            n_run, n_skip = sum(r.status == "siap" for r in rows), sum(r.status == "dilewati" for r in rows)
            skip_note = " (--skip-pending)" if n_skip else ""      # hanya bila ada klip yang dilewati (perbaikan a, Rio)
            print(f"Mulai: {n_run} klip dijalankan, {n_skip} dilewati{skip_note}", flush=True)
            code = run_jobs(rows, args)
            code = EXIT_INTERRUPTED if code == EXIT_INTERRUPTED else final_code(rows)
    except BatchError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        code = e.code
    except KeyboardInterrupt:
        code = EXIT_INTERRUPTED
    for r in rows:
        if r.status == "siap":
            r.status = "belum dijalankan"
    path = write_summary(cfg, rows, code, started)
    print(render_summary(rows, code, path), flush=True)
    return code


# ── Argumen ───────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=f"{cli_mod.PROG} batch", description="T-603: banyak klip → run berurutan (tanpa --yes = rencana)")
    p.add_argument("inputs", nargs="*", metavar="video|folder", help="video atau folder (tidak rekursif)")
    p.add_argument("--list", dest="list_file", type=Path, default=None, help="berkas daftar: <path> [--start S --end E] per baris")
    p.add_argument("--yes", action="store_true", help="JALAN (GPU); tanpa flag ini hanya rencana")
    p.add_argument("--skip-pending", action="store_true", help="dengan --yes: klip perlu tindakan dilewati, sisanya jalan")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--style", default=None)
    p.add_argument("--seg-model", choices=("0.8b", "0.4b"), default=None)
    return p


def main(argv: list[str]) -> int:
    reconfigure_stdio()
    for tok in argv:
        if tok.split("=", 1)[0] in FORBIDDEN_FLAGS:
            print(f"ERROR: {tok} tidak diizinkan di batch — jalankan `run` per klip untuk itu.", file=sys.stderr)
            return EXIT_PRECONDITION
    args = build_parser().parse_args(argv)
    set_message_config(args.config)
    started = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    try:
        cfg = cli_mod.load_cfg(args.config, args.seg_model)
        if not args.inputs and args.list_file is None:
            raise BatchError("tidak ada masukan: beri video / folder atau --list BERKAS", cli_mod.EXIT_USAGE)
        if args.skip_pending and not args.yes:
            print("PERINGATAN: --skip-pending hanya berarti dengan --yes (rencana tidak menjalankan apa pun).", flush=True)
        specs = expand_inputs(args.inputs, cfg.batch.video_exts) if args.inputs else []
        if args.list_file is not None:
            specs += parse_list(args.list_file)
        if not specs:
            raise BatchError("tidak ada klip di masukan (folder kosong / daftar kosong)")
        rows = build_rows(specs, cfg)
        hint = f"{cli_mod.PROG} batch " + " ".join(quote_arg(a) for a in argv if a not in ("--yes", "--skip-pending"))
        if not args.yes:
            plan(rows, cfg, args)
            need = disk_check(rows, cfg)
            code = plan_exit(rows, need)
            print(render_plan(rows, hint, need, code), flush=True)
            return code
        return execute(rows, cfg, args, hint, started)
    except (BatchError, cli_mod.CliError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return e.code
    except KeyboardInterrupt:
        print("ERROR: dihentikan (Ctrl+C); jalankan ulang perintah yang sama untuk melanjutkan", file=sys.stderr)
        return EXIT_INTERRUPTED
