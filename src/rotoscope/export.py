"""Stage [6] export (NAIF, Phase 1 — T-103, D-010): frame → out/<nama>.mp4. Kontrak lengkap: docs/01 [6].

Satu jalur encode untuk semua sumber gambar (`FrameSource`):
  - "silhouette" (Phase 1): stable/groups/*.png, grup ≠ 0 = foreground `export.foreground_color` di atas
    `export.background_color`.
  - "strokes" (Phase 2, T-2xx): strokes/*.png dari [5] apa adanya — belum diimplementasi; cukup menambah
    satu kelas FrameSource, encode / manifest / verifikasi tidak berubah.

Encode: frame RGB mentah di-pipe ke stdin ffmpeg (tanpa PNG sementara) → libx264, -pix_fmt yuv420p,
-r = `target_fps` dari meta.json, dimensi dipad ke genap. Ditulis ke <nama>.mp4.tmp lalu diverifikasi
ffprobe (jumlah frame, fps, codec, pix_fmt, ukuran, durasi, stream audio) dan baru di-os.replace (retry
Windows dari stage_common) — MP4 lama tidak pernah rusak oleh run yang gagal.

Nama file = `export.filename` (default "{source}.mp4"; {source} = nama video sumber dari meta.json,
disanitasi). Manifest <nama>.export.json memuat identitas klip (sha256 meta.json + source_path):
  - file tujuan sudah ada + manifest-nya menunjuk video sumber LAIN → DITOLAK (ubah export.filename;
    hasil klip lain tidak pernah ditimpa);
  - file tujuan ada tanpa manifest → ditolak (asal tidak diketahui); --restart menimpa;
  - parameter / input berubah (termasuk meta.json klip yang sama) → basi: di-encode ulang otomatis dengan
    peringatan (stage CPU murah + deterministik, prinsip #4 docs/01);
  - sama + MP4 lolos ffprobe → dilewati.

Audio (`export.audio`, default false): audio meme hampir selalu milik pihak ketiga (musik) → default
TANPA audio; tambahkan audio dari library berlisensi di editor platform (TikTok/CapCut). Kalau true:
audio video sumber dimasukkan (aac, -shortest); sumber tanpa audio (meta.has_audio false) atau file
sumber hilang = ERROR, bukan diam-diam tanpa audio.

Uji manual (entry point sementara sampai cli.py, T-104b):
    python -m rotoscope.export [--config PATH] [--restart] [--limit N]
--limit N → <nama>.limitN.mp4 (preview N frame pertama; tanpa manifest, tanpa pengaman, tidak menyentuh
hasil utama). Exit code: 0 sukses, 1 prasyarat gagal (3 = OOM tidak dipakai di stage CPU).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path

import numpy as np

from rotoscope import stabilize as stab
from rotoscope.config import (
    ConfigError, PipelineConfig, ensure_dir, load_pipeline, section_hash, to_dict,
)
from rotoscope.ingest import META_FILENAME
from rotoscope.stage_common import (
    CLIP_KEY, EXIT_OK, EXIT_PRECONDITION, StageError, _replace_with_retry, clip_identity_from_bytes,
    describe_identity, reconfigure_stdio, utc_now, write_json_atomic,
)

# ── Layout output (docs/01 [6]) ────────────────────
MP4_SUFFIX = ".mp4"
MANIFEST_SUFFIX = ".export.json"
TMP_SUFFIX = ".tmp"
SOURCE_PLACEHOLDER = "{source}"
SOURCE_FALLBACK_NAME = "clip"
SOURCE_UNSAFE_RE = re.compile(r"[^\w.\-]+")
SOURCE_STRIP_CHARS = "._-"

# ── Encode ─────────────────────────────────────────
VIDEO_CODEC = "libx264"
PIX_FMT = "yuv420p"
AUDIO_CODEC = "aac"
PROBE_CODEC = "h264"
RAW_PIX_FMT = "rgb24"
BACKGROUND_ID = stab.BACKGROUND_ID
FPS_TOLERANCE = 1e-3
STDERR_TAIL_LINES = 20

# Kunci manifest yang harus sama agar export dilewati; beda → basi (di-encode ulang).
MANIFEST_MATCH_KEYS = ("export_hash", "clip", "stable", "frame_count", "frame_size", "fps", "audio")

DEFAULT_CONFIG = Path("configs") / "default.yaml"


# ── Nama file ──────────────────────────────────────
def sanitize_source_name(source_path: str) -> str:
    """Nama video sumber (tanpa ekstensi) yang aman untuk nama file Windows."""
    stem = SOURCE_UNSAFE_RE.sub("_", Path(source_path).stem).strip(SOURCE_STRIP_CHARS)
    return stem or SOURCE_FALLBACK_NAME


def resolve_filename(template: str, source_path: str, limit: int | None = None) -> str:
    name = template.replace(SOURCE_PLACEHOLDER, sanitize_source_name(source_path))
    if limit is not None:
        name = name[: -len(MP4_SUFFIX)] + f".limit{limit}" + MP4_SUFFIX
    return name


def manifest_path_for(mp4: Path) -> Path:
    return mp4.with_name(mp4.stem + MANIFEST_SUFFIX)


# ── Sumber gambar ──────────────────────────────────
def parse_hex(color: str) -> tuple[int, int, int]:
    return int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)


class FrameSource:
    """Sumber frame RGB uint8 (H, W, 3) untuk satu jalur encode. Subkelas: Phase 1 silhouette, Phase 2 strokes."""

    kind = ""

    def check(self, names: list[str]) -> None:
        """File input frame terpilih ada (isi dicek saat dibaca)."""
        raise NotImplementedError

    def render(self, name: str) -> np.ndarray:
        raise NotImplementedError


class SilhouetteSource(FrameSource):
    kind = "silhouette"

    def __init__(self, cfg: PipelineConfig, clip: stab.Clip):
        self.clip = clip
        self.n_groups = len(cfg.groups)
        self.fg = np.array(parse_hex(cfg.export.foreground_color), np.uint8)
        self.bg = np.array(parse_hex(cfg.export.background_color), np.uint8)

    def check(self, names: list[str]) -> None:
        missing = [n for n in names if not self.clip.groups_path(n).is_file()]
        if missing:
            raise StageError(f"stable/groups belum lengkap: {len(missing)} dari {len(names)} frame hilang, mis. "
                             f"{Path(missing[0]).stem} — jalankan stage [3] stabilize sampai selesai")

    def render(self, name: str) -> np.ndarray:
        gmap = stab.read_groups(self.clip.groups_path(name))
        if (gmap is None or gmap.shape != (self.clip.height, self.clip.width)
                or int(gmap.max()) > self.n_groups):
            raise StageError(f"peta grup {Path(name).stem} rusak / ukuran salah — jalankan ulang stage [3] "
                             f"stabilize (frame rusak dihitung ulang)")
        img = np.empty((*gmap.shape, 3), np.uint8)
        img[:] = self.bg
        img[gmap != BACKGROUND_ID] = self.fg
        return img


def make_source(cfg: PipelineConfig, clip: stab.Clip) -> FrameSource:
    if cfg.export.source == SilhouetteSource.kind:
        return SilhouetteSource(cfg, clip)
    raise StageError(f"export.source = {cfg.export.source!r} belum diimplementasi (Phase 2: strokes dari "
                     f"stage [5], T-2xx) — set 'silhouette'")


def pad_even(img: np.ndarray, bg: np.ndarray) -> np.ndarray:
    """Dimensi genap (syarat yuv420p): tambah 1 px warna latar di kanan / bawah kalau ganjil."""
    h, w = img.shape[:2]
    if h % 2 == 0 and w % 2 == 0:
        return img
    out = np.empty((h + h % 2, w + w % 2, 3), np.uint8)
    out[:] = bg
    out[:h, :w] = img
    return out


# ── ffmpeg / ffprobe ───────────────────────────────
def _binary(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise StageError(f"{name} tidak ditemukan di PATH. Install ffmpeg (gyan.dev essentials build) dan "
                         f"pastikan folder bin-nya ada di PATH.")
    return path


def _fps_arg(fps: float) -> str:
    return str(int(fps)) if float(fps).is_integer() else repr(float(fps))


def _tail(text: str) -> str:
    return "\n".join(text.strip().splitlines()[-STDERR_TAIL_LINES:])


def build_ffmpeg_cmd(ffmpeg: str, size: tuple[int, int], fps: float, crf: int, preset: str,
                     audio_src: Path | None, out_tmp: Path) -> list[str]:
    w, h = size
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", RAW_PIX_FMT, "-s", f"{w}x{h}", "-r", _fps_arg(fps), "-i", "-"]
    if audio_src is not None:
        cmd += ["-i", str(audio_src)]
    cmd += ["-map", "0:v:0"]
    if audio_src is not None:
        cmd += ["-map", "1:a:0", "-c:a", AUDIO_CODEC, "-shortest"]
    cmd += ["-c:v", VIDEO_CODEC, "-preset", preset, "-crf", str(crf), "-pix_fmt", PIX_FMT,
            "-r", _fps_arg(fps), "-movflags", "+faststart", "-f", "mp4", str(out_tmp)]
    return cmd


def encode(source: FrameSource, names: list[str], size: tuple[int, int], bg: np.ndarray, fps: float,
           crf: int, preset: str, audio_src: Path | None, out_tmp: Path) -> None:
    """Pipe frame ke ffmpeg → out_tmp. Gagal apa pun → StageError (out_tmp dihapus oleh pemanggil)."""
    cmd = build_ffmpeg_cmd(_binary("ffmpeg"), size, fps, crf, preset, audio_src, out_tmp)
    with tempfile.TemporaryFile() as err:  # stderr ke file: pipe stderr penuh bisa membuat ffmpeg deadlock
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=err)
        broken = False
        try:
            for name in names:
                frame = pad_even(source.render(name), bg)
                if frame.shape[:2] != (size[1], size[0]):
                    raise StageError(f"ukuran frame {Path(name).stem} {frame.shape[1]}x{frame.shape[0]} ≠ "
                                     f"{size[0]}x{size[1]} (frame pertama)")
                try:
                    proc.stdin.write(np.ascontiguousarray(frame).tobytes())
                except (BrokenPipeError, OSError):
                    broken = True
                    break
            try:
                proc.stdin.close()
            except (BrokenPipeError, OSError):
                broken = True
        except BaseException:
            proc.kill()
            raise
        finally:
            rc = proc.wait()
        if rc != 0 or broken:
            err.seek(0)
            raise StageError(f"ffmpeg gagal (exit {rc}) saat meng-encode:\n"
                             f"{_tail(err.read().decode('utf-8', 'replace'))}")


def probe_output(path: Path) -> dict:
    """ffprobe file MP4 → ringkasan (video + jumlah stream audio)."""
    proc = subprocess.run([_binary("ffprobe"), "-v", "error", "-count_packets", "-print_format", "json",
                           "-show_streams", "-show_format", str(path)],
                          capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise StageError(f"ffprobe gagal membaca {path.name} (exit {proc.returncode}):\n{_tail(proc.stderr)}")
    try:
        info = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise StageError(f"output ffprobe bukan JSON valid: {e}") from None
    streams = info.get("streams", [])
    videos = [s for s in streams if s.get("codec_type") == "video"]
    audios = [s for s in streams if s.get("codec_type") == "audio"]
    if len(videos) != 1:
        raise StageError(f"{path.name}: {len(videos)} stream video, harus 1")
    v = videos[0]
    try:
        fps = float(Fraction(v.get("r_frame_rate", "0/1")))
    except (ValueError, ZeroDivisionError):
        fps = 0.0
    duration = v.get("duration", info.get("format", {}).get("duration"))
    return {"codec": v.get("codec_name"), "pix_fmt": v.get("pix_fmt"), "width": v.get("width"),
            "height": v.get("height"), "fps": fps,
            "frames": int(v["nb_read_packets"]) if v.get("nb_read_packets") is not None else None,
            "duration_s": float(duration) if duration is not None else None,
            "audio_streams": len(audios), "audio_codec": audios[0].get("codec_name") if audios else None,
            "size_bytes": path.stat().st_size}


def verification_errors(probe: dict, *, frames: int, fps: float, size: tuple[int, int],
                        audio: bool) -> list[str]:
    errs = []
    if probe["frames"] != frames:
        errs.append(f"jumlah frame {probe['frames']} ≠ {frames}")
    if abs(probe["fps"] - fps) > FPS_TOLERANCE:
        errs.append(f"fps {probe['fps']:.4g} ≠ {fps:g}")
    if probe["codec"] != PROBE_CODEC:
        errs.append(f"codec {probe['codec']} ≠ {PROBE_CODEC}")
    if probe["pix_fmt"] != PIX_FMT:
        errs.append(f"pix_fmt {probe['pix_fmt']} ≠ {PIX_FMT}")
    if (probe["width"], probe["height"]) != size:
        errs.append(f"ukuran {probe['width']}x{probe['height']} ≠ {size[0]}x{size[1]}")
    expected = frames / fps
    if probe["duration_s"] is None or abs(probe["duration_s"] - expected) > 1 / fps + FPS_TOLERANCE:
        errs.append(f"durasi {probe['duration_s']} s ≠ {expected:.3f} s (±1 frame)")
    want_audio = 1 if audio else 0
    if probe["audio_streams"] != want_audio:
        errs.append(f"stream audio {probe['audio_streams']} ≠ {want_audio} (export.audio = {str(audio).lower()})")
    return errs


# ── Manifest ───────────────────────────────────────
def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


def _short(v) -> str:
    return v[:12] if isinstance(v, str) and len(v) > 12 else repr(v)


def manifest_diff(old: dict, new: dict) -> list[str]:
    a, b = _flatten({k: old.get(k) for k in MANIFEST_MATCH_KEYS}), _flatten({k: new.get(k) for k in MANIFEST_MATCH_KEYS})
    return [f"{k}: {_short(a.get(k))} → {_short(b.get(k))}" for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]


def export_hash(cfg: PipelineConfig) -> str:
    """Hash parameter export yang memengaruhi ISI video (`filename` tidak — itu hanya nama)."""
    data = {k: v for k, v in to_dict(cfg.export).items() if k != "filename"}
    blob = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def read_manifest(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        m = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return m if isinstance(m, dict) else None


def _audio_ref(cfg: PipelineConfig, source_path: Path) -> dict:
    if not cfg.export.audio:
        return {"enabled": False}
    st = source_path.stat()
    return {"enabled": True, "size": st.st_size, "mtime_ns": st.st_mtime_ns}


# ── Run ────────────────────────────────────────────
def run_export(cfg: PipelineConfig, *, restart: bool = False, limit: int | None = None,
               log: Callable[[str], None] = print) -> dict:
    """Jalankan stage [6] (naif). Return ringkasan run."""
    if limit is not None and limit < 1:
        raise StageError(f"--limit harus ≥ 1, dapat {limit}")
    ex = cfg.export
    work_dir = cfg.paths.work_dir
    meta_path = work_dir / META_FILENAME
    clip = stab.load_clip(work_dir)                       # StageError kalau meta.json tidak ada
    meta_bytes = meta_path.read_bytes()
    meta = json.loads(meta_bytes.decode("utf-8"))
    fps = float(meta["target_fps"])
    identity = clip_identity_from_bytes(meta_bytes)
    source_path = identity["source_path"]
    names = list(clip.names[:limit] if limit else clip.names)

    stable_m = stab._read_manifest(clip.manifest_path, "[3] stabilize")
    stable_clip = stable_m.get(CLIP_KEY)
    if not isinstance(stable_clip, dict) or stable_clip.get("meta_sha256") != identity["meta_sha256"]:
        raise StageError(f"stable/manifest.json milik klip lain / tanpa identitas klip "
                         f"({describe_identity(stable_clip)}; meta.json saat ini = {describe_identity(identity)}) — "
                         f"jalankan ulang stage [3] stabilize (output basi dihitung ulang otomatis)")
    if stable_m.get("frame_size") != {"width": clip.width, "height": clip.height}:
        raise StageError(f"stable/manifest.json: frame_size {stable_m.get('frame_size')} ≠ meta.json — "
                         f"jalankan ulang stage [3] stabilize dengan --restart")
    if stable_m.get("groups_hash") != section_hash(cfg, "groups"):
        raise StageError("definisi groups di config ≠ stable/manifest.json — jalankan stage [3] stabilize "
                         "(output basi dihitung ulang otomatis)")
    source = make_source(cfg, clip)
    source.check(names)

    audio_src: Path | None = None
    if ex.audio:
        if not meta.get("has_audio"):
            raise StageError("export.audio = true tetapi video sumber tidak punya audio (meta.json has_audio "
                             "false) — set export.audio: false, atau pakai video sumber beraudio")
        audio_src = Path(source_path)
        if not audio_src.is_file():
            raise StageError(f"export.audio = true tetapi video sumber {audio_src} tidak ditemukan "
                             f"(dibutuhkan untuk mengambil audio)")

    out_dir = cfg.paths.out_dir
    target = out_dir / resolve_filename(ex.filename, source_path, limit)
    manifest_path = manifest_path_for(target)
    tmp = target.with_name(target.name + TMP_SUFFIX)

    size = (clip.width + clip.width % 2, clip.height + clip.height % 2)
    manifest = {
        "stage": "export", "source": ex.source, "export": to_dict(ex), "export_hash": export_hash(cfg),
        CLIP_KEY: identity,
        "stable": {k: stable_m.get(k) for k in ("stabilize_hash", "groups_hash", "created_utc")},
        "frame_count": len(names), "frame_size": {"width": clip.width, "height": clip.height},
        "encoded_size": {"width": size[0], "height": size[1]}, "fps": fps,
        "audio": _audio_ref(cfg, audio_src) if audio_src else {"enabled": False},
    }

    stale: list[str] = []
    if limit is None:
        old = read_manifest(manifest_path)
        if old is not None and old.get("clip", {}).get("source_path") not in (None, source_path) \
                and target.is_file():
            raise StageError(f"{target} sudah ada dan berasal dari video sumber LAIN "
                             f"({old['clip']['source_path']}) — tidak ditimpa. Ubah export.filename "
                             f"(mis. \"{SOURCE_PLACEHOLDER}{MP4_SUFFIX}\") atau pindahkan / hapus file itu.")
        if target.is_file():
            if old is None and not restart:
                raise StageError(f"{target} sudah ada tanpa {manifest_path.name} — asal file tidak diketahui. "
                                 f"Jalankan dengan --restart untuk menimpa.")
            if old is not None and not restart:
                stale = manifest_diff(old, manifest)
                if not stale:
                    try:
                        probe = probe_output(target)
                        bad = verification_errors(probe, frames=len(names), fps=fps, size=size,
                                                  audio=bool(audio_src))
                    except StageError as e:
                        bad = [str(e)]
                    if not bad:
                        log(f"[6] export: {target.name} sudah up-to-date ({len(names)} frame) — dilewati")
                        return {"output": target, "skipped": True, "stale": [], "frames": len(names),
                                "probe": probe, "wall_s": 0.0}
                    stale = [f"file lama tidak valid: {b}" for b in bad]
                log(f"PERINGATAN: output [6] basi (setelan / input berubah) — {target.name} di-encode ulang:\n  "
                    + "\n  ".join(stale))
    ensure_dir(out_dir)
    tmp.unlink(missing_ok=True)

    log(f"[6] export ({ex.source}, crf {ex.crf}, {ex.preset}, {fps:g} fps, {size[0]}x{size[1]}, "
        f"audio {'ya' if audio_src else 'tidak'}): {len(names)} frame → {target.name}")
    t0 = time.perf_counter()
    try:
        encode(source, names, size, np.array(parse_hex(ex.background_color), np.uint8), fps, ex.crf, ex.preset,
               audio_src, tmp)
        probe = probe_output(tmp)
        errs = verification_errors(probe, frames=len(names), fps=fps, size=size, audio=bool(audio_src))
        if errs:
            raise StageError(f"verifikasi ffprobe gagal untuk {target.name}:\n  " + "\n  ".join(errs))
        _replace_with_retry(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)
    wall = time.perf_counter() - t0
    probe["size_bytes"] = target.stat().st_size
    if limit is None:
        write_json_atomic(manifest_path, {**manifest, "output": {"file": target.name, **probe},
                                          "created_utc": utc_now()})
    log(f"  ffprobe OK: {probe['frames']} frame, {probe['fps']:g} fps, {probe['codec']} {probe['pix_fmt']} "
        f"{probe['width']}x{probe['height']}, {probe['duration_s']:.3f} s, audio {probe['audio_streams']} stream, "
        f"{probe['size_bytes'] / 1024:.1f} KiB, {wall:.1f} s")
    return {"output": target, "skipped": False, "stale": stale, "frames": len(names), "probe": probe,
            "wall_s": wall}


# ── Entry point sementara (cli.py = T-104b) ────────
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m rotoscope.export",
                                description="Stage [6]: stable/groups → out/<nama>.mp4 (siluet, naif)")
    p.add_argument("--config", type=Path, default=None,
                   help=f"YAML pipeline (default: {DEFAULT_CONFIG.as_posix()} kalau ada, selain itu default kode)")
    p.add_argument("--restart", action="store_true", help="encode ulang walau up-to-date (menimpa file tanpa manifest)")
    p.add_argument("--limit", type=int, default=None, help="preview N frame pertama → <nama>.limitN.mp4")
    args = p.parse_args(argv)
    reconfigure_stdio()

    def log(msg: str) -> None:
        print(msg, flush=True)

    try:
        path = args.config if args.config is not None else (DEFAULT_CONFIG if DEFAULT_CONFIG.is_file() else None)
        cfg = load_pipeline(path)
        run = run_export(cfg, restart=args.restart, limit=args.limit, log=log)
        log(f"selesai: {run['output']}")
    except (StageError, ConfigError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return EXIT_PRECONDITION
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
