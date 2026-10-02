"""Stage [3] stabilize: seg/probs + seg/classmap + depth/ → stable/groups/*.png + stable/depth_smooth/*.npy
+ stable/manifest.json + stable/frames.jsonl (D-010). Kontrak lengkap: docs/01 [3].

T-106 = SPASIAL SAJA (`stabilize.temporal.enabled: false`); temporal (EMA + optical flow) = T-302/T-303.
CPU saja, tanpa torch. Per frame:
  1. probabilitas kelas uint8 (29, H, W) → probabilitas grup = jumlah per grup / 255. argmax dihitung
     dari jumlah integer (eksak, sama dengan argmax float32). Seri eksak → grup dari seg/classmap
     (argmax logits tanpa pembulatan) kalau grup itu ikut seri, selain itu id terkecil.
  3. argmax → peta grup (0 = background, 1..G = urutan `groups:`).
  4. Filter pulau: komponen 8-arah sebuah grup (termasuk background) < N px → grup mayoritas di cincin
     1 px sekelilingnya, dibaca dari peta sebelum filter (satu lintasan). Seri → id terkecil.
  5. Mode filter K×K: grup yang paling sering muncul di jendela; background ikut; seri → grup asli
     piksel, seri antara grup lain → id terkecil.
  6. Kedalaman log_median_iqr: (log max(d, eps) − median) / max(IQR, iqr_min), statistik dari foreground
     peta grup bersih (kosong → seluruh frame). Transformasi yang sama dipakai di background (nilai
     kontinu + finite, tanpa lompatan palsu di siluet).
(Langkah 2 = temporal, belum ada.)

Manifest berbeda (grup / parameter stabilize / input [2]/[2c] / identitas klip berubah) → output lama BASI:
dihapus dan dihitung ulang otomatis dengan peringatan (stage CPU murah + deterministik). Manifest lama tanpa
`clip` juga basi (dihitung ulang sekali). Output tanpa manifest → ditolak. --restart = paksa hitung ulang.
Input [2]/[2c] yang milik klip lain / tanpa identitas → ditolak (load_inputs); [3] tidak punya --adopt.

CLI final (T-104b): python -m rotoscope stabilize <video> [--config PATH] [--restart] [--limit N]; atau seluruh
pipeline: python -m rotoscope run <video>. cli.py memanggil main() ini in-process dengan --work-dir <folder klip>:
    python -m rotoscope.stabilize [--config PATH] [--work-dir DIR] [--restart] [--limit N]
Exit code: 0 sukses, 1 prasyarat gagal (3 = OOM tidak dipakai di stage CPU).
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from rotoscope import depth as dep
from rotoscope import segment as seg
from rotoscope.config import (
    ConfigError, PipelineConfig, ensure_dir, load_class_names, load_pipeline, section_hash, to_dict,
)
from rotoscope.stage_common import (
    CLIP_KEY, EXIT_OK, EXIT_PRECONDITION, StageError, add_work_dir_arg, append_jsonl, clean_tmp, cli_cmd,
    clip_identity, describe_identity, last_frame_records, load_frame_list, reconfigure_stdio, utc_now,
    work_dir_overrides, write_bytes_atomic, write_json_atomic,
)

# ── Layout output (docs/01 [3]) ────────────────────
STABLE_DIRNAME = "stable"
GROUPS_DIRNAME = "groups"
DEPTH_SMOOTH_DIRNAME = "depth_smooth"
MANIFEST_FILENAME = "manifest.json"
FRAMES_LOG_FILENAME = "frames.jsonl"
GROUPS_SUFFIX = ".png"
DEPTH_SUFFIX = ".npy"
DEPTH_DTYPE = np.float16
BACKGROUND_ID = 0
PROBS_SCALE = seg.PROBS_SCALE          # probabilitas grup = jumlah uint8 per grup / 255
RING_KERNEL = np.ones((3, 3), np.uint8)  # cincin 1 px (8-arah) di filter pulau
OUTPUT_INFO = {
    "groups": "id grup uint8 PNG: 0 = background, 1..G = urutan groups: di YAML",
    "depth_smooth": "float16 (H, W): (log max(d, log_eps) − median fg) / max(IQR fg, iqr_min), per frame",
}

# Kunci manifest yang harus sama untuk resume; beda → output basi (dihapus + dihitung ulang).
MANIFEST_MATCH_KEYS = ("stabilize_hash", "groups_hash", "seg", "depth", "frame_size", CLIP_KEY)
SEG_REF_KEYS = ("model", "model_id", "revision", "precision", "processor", "num_labels", "frame_size",
                "classes", "created_utc")
DEPTH_REF_KEYS = ("model_id", "revision", "license", "precision", "processor", "input_size", "output",
                  "frame_size", "created_utc")

DEFAULT_CONFIG = Path("configs") / "default.yaml"


# ── Langkah 1 + 3: probabilitas grup → peta grup ──
def class_group_lut(groups: Sequence[tuple[str, Sequence[str]]], classes: Sequence[str]) -> np.ndarray:
    """id kelas → id grup (uint8); kelas yang tidak tercantum (hanya Background, dijamin config) → 0."""
    index = {c: i for i, c in enumerate(classes)}
    lut = np.zeros(len(classes), np.uint8)
    for gid, (_, members) in enumerate(groups, 1):
        for c in members:
            lut[index[c]] = gid
    return lut


def group_sums(probs: np.ndarray, lut: np.ndarray, n_groups: int) -> np.ndarray:
    """probs uint8 (C, H, W) → jumlah per grup uint16 (G+1, H, W); maks 29 × 255 = 7395 (eksak)."""
    out = np.zeros((n_groups + 1,) + probs.shape[1:], np.uint16)
    for g in range(n_groups + 1):
        ids = np.flatnonzero(lut == g)
        if ids.size:
            probs[ids].sum(axis=0, dtype=np.uint16, out=out[g])
    return out


def group_argmax(sums: np.ndarray, tie_groups: np.ndarray) -> tuple[np.ndarray, int]:
    """argmax jumlah grup → (peta grup uint8, jumlah piksel seri).

    Seri eksak → `tie_groups` (grup classmap) kalau grup itu termasuk yang seri; selain itu id terkecil.
    """
    top = sums.max(axis=0)
    is_max = sums == top
    gmap = is_max.argmax(axis=0).astype(np.uint8)  # id terkecil di antara yang maksimum
    tie = is_max.sum(axis=0) > 1
    n_tie = int(tie.sum())
    if n_tie:
        cg_is_max = np.take_along_axis(is_max, tie_groups.astype(np.intp)[None], axis=0)[0]
        use = tie & cg_is_max
        gmap[use] = tie_groups[use]
    return gmap, n_tie


# ── Langkah 4: filter pulau ────────────────────────
def island_filter(gmap: np.ndarray, n_min: int) -> np.ndarray:
    """Komponen 8-arah sebuah grup (termasuk background) < n_min px → grup mayoritas di cincin 1 px.

    Satu lintasan: cincin dibaca dari peta ASLI, jadi urutan tidak berpengaruh. Cincin dipotong di tepi
    gambar (tanpa piksel virtual); cincin kosong (komponen = seluruh gambar) → tidak diubah. Seri
    mayoritas → id terkecil. Sama dengan island_filter T-102c (scripts/sapiens2_exp.py), di peta grup.
    """
    out = gmap.copy()
    if n_min <= 0:
        return out
    h, w = gmap.shape
    for g in np.unique(gmap):
        _, lab, st, _ = cv2.connectedComponentsWithStats((gmap == g).astype(np.uint8), connectivity=8)
        for i in np.flatnonzero(st[1:, cv2.CC_STAT_AREA] < n_min) + 1:
            x, y, bw, bh = st[i, :4]
            y0, y1, x0, x1 = max(y - 1, 0), min(y + bh + 1, h), max(x - 1, 0), min(x + bw + 1, w)
            m = lab[y0:y1, x0:x1] == i
            ring = cv2.dilate(m.astype(np.uint8), RING_KERNEL).astype(bool) & ~m
            vals = gmap[y0:y1, x0:x1][ring]
            if vals.size:
                out[y0:y1, x0:x1][m] = np.bincount(vals).argmax()
    return out


# ── Langkah 5: mode filter ─────────────────────────
def mode_filter(gmap: np.ndarray, k: int) -> np.ndarray:
    """Tiap piksel → grup yang paling sering di jendela k×k (background ikut).

    Hitungan per grup = box filter tanpa normalisasi atas mask grup (BORDER_REPLICATE) + 0.5 untuk grup
    asli piksel → seri dimenangkan grup asli; seri antara grup lain → id terkecil (loop naik, `>` ketat).
    Sama dengan mode_filter T-102c.
    """
    if k <= 1:
        return gmap.copy()
    out = gmap.copy()
    best = np.full(gmap.shape, -1.0, np.float32)
    for g in np.unique(gmap):
        m = (gmap == g).astype(np.float32)
        cnt = cv2.boxFilter(m, -1, (k, k), normalize=False, borderType=cv2.BORDER_REPLICATE) + 0.5 * m
        upd = cnt > best
        out[upd] = g
        best[upd] = cnt[upd]
    return out


def clean_groups(sums: np.ndarray, tie_groups: np.ndarray, island_min_px: int, mode_k: int) -> tuple[np.ndarray, dict]:
    """Langkah 3–5 → (peta grup bersih, statistik perubahan)."""
    raw, n_tie = group_argmax(sums, tie_groups)
    isl = island_filter(raw, island_min_px)
    out = mode_filter(isl, mode_k)
    return out, {"tie_px": n_tie, "island_changed_px": int((isl != raw).sum()),
                 "mode_changed_px": int((out != isl).sum()), "fg_px": int((out != BACKGROUND_ID).sum())}


# ── Langkah 6: kedalaman ───────────────────────────
def normalize_depth(disparity: np.ndarray, fg: np.ndarray, log_eps: float, iqr_min: float) -> tuple[np.ndarray, dict]:
    """log_median_iqr per frame → (float16 (H, W), statistik).

    Statistik (median, IQR = p75 − p25) dari foreground; foreground kosong → seluruh frame. IQR < iqr_min
    → pembagi = iqr_min. Background memakai transformasi yang sama.
    """
    lg = np.log(np.maximum(disparity.astype(np.float32), np.float32(log_eps)))
    region = "foreground" if fg.any() else "frame"
    vals = lg[fg] if region == "foreground" else lg.ravel()
    p25, med, p75 = (float(v) for v in np.percentile(vals, (25, 50, 75)))
    iqr = p75 - p25
    clamped = iqr < iqr_min
    out32 = (lg - np.float32(med)) / np.float32(iqr_min if clamped else iqr)
    with np.errstate(over="ignore"):
        out = out32.astype(DEPTH_DTYPE)
    if not np.isfinite(out).all():
        raise StageError(f"depth_smooth tidak finite / meluap float16 (median {med:.4g}, IQR {iqr:.4g}) — "
                         f"cek depth/ dan stabilize.depth.log_eps / iqr_min")
    o = out.astype(np.float32)
    return out, {"region": region, "log_median": round(med, 5), "log_iqr": round(iqr, 5),
                 "iqr_clamped": bool(clamped), "min": round(float(o.min()), 4),
                 "median": round(float(np.median(o)), 4), "max": round(float(o.max()), 4)}


# ── Tulis + baca ───────────────────────────────────
def write_groups(path: Path, gmap: np.ndarray) -> None:
    ok, buf = cv2.imencode(GROUPS_SUFFIX, gmap)
    if not ok:
        raise StageError(f"gagal meng-encode peta grup {path.name}")
    write_bytes_atomic(path, buf.tobytes())


def read_groups(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    m = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_UNCHANGED)
    return m if m is not None and m.dtype == np.uint8 and m.ndim == 2 else None


def groups_valid(path: Path, h: int, w: int, n_groups: int) -> bool:
    m = read_groups(path)
    return m is not None and m.shape == (h, w) and int(m.max()) <= n_groups


def write_depth_smooth(path: Path, y: np.ndarray) -> None:
    buf = io.BytesIO()
    np.save(buf, y, allow_pickle=False)
    write_bytes_atomic(path, buf.getvalue())


def depth_smooth_valid(path: Path, h: int, w: int) -> bool:
    """Terbaca, float16, ukuran = frame, semua finite (aturan sama dengan depth/ [2c])."""
    return dep.depth_valid(path, h, w)


# ── Layout klip ────────────────────────────────────
@dataclass(frozen=True)
class Clip:
    work_dir: Path
    names: tuple[str, ...]
    indices: tuple[int, ...]
    width: int
    height: int

    @property
    def seg_clip(self) -> seg.Clip:
        return seg.Clip(self.work_dir, self.names, self.indices, self.width, self.height)

    @property
    def depth_clip(self) -> dep.Clip:
        return dep.Clip(self.work_dir, self.names, self.indices, self.width, self.height)

    @property
    def stable_dir(self) -> Path:
        return self.work_dir / STABLE_DIRNAME

    @property
    def groups_dir(self) -> Path:
        return self.stable_dir / GROUPS_DIRNAME

    @property
    def depth_smooth_dir(self) -> Path:
        return self.stable_dir / DEPTH_SMOOTH_DIRNAME

    @property
    def manifest_path(self) -> Path:
        return self.stable_dir / MANIFEST_FILENAME

    @property
    def frames_log(self) -> Path:
        return self.stable_dir / FRAMES_LOG_FILENAME

    def groups_path(self, name: str) -> Path:
        return self.groups_dir / (Path(name).stem + GROUPS_SUFFIX)

    def depth_smooth_path(self, name: str) -> Path:
        return self.depth_smooth_dir / (Path(name).stem + DEPTH_SUFFIX)

    def frame_valid(self, name: str, n_groups: int) -> bool:
        return (groups_valid(self.groups_path(name), self.height, self.width, n_groups)
                and depth_smooth_valid(self.depth_smooth_path(name), self.height, self.width))


def load_clip(work_dir: Path) -> Clip:
    return Clip(work_dir, *load_frame_list(work_dir))


# ── Input [2] / [2c] ───────────────────────────────
def _read_manifest(path: Path, stage: str) -> dict:
    if not path.is_file():
        raise StageError(f"{path} tidak ada — jalankan stage {stage} dulu")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise StageError(f"gagal membaca {path} ({e}) — jalankan ulang stage {stage}") from None


def load_inputs(clip: Clip) -> tuple[dict, dict]:
    """Manifest [2] + [2c], dicek terhadap meta.json dan daftar kelas paket."""
    seg_m = _read_manifest(clip.seg_clip.manifest_path, "[2] segment")
    dep_m = _read_manifest(clip.depth_clip.manifest_path, "[2c] depth")
    size = {"width": clip.width, "height": clip.height}
    current = clip_identity(clip.work_dir)
    seg_cmd, dep_cmd = seg.stage_cmd(clip.work_dir), dep.stage_cmd(clip.work_dir)
    for label, m, stage, cmd in (("seg/manifest.json", seg_m, "[2]", seg_cmd),
                                 ("depth/manifest.json", dep_m, "[2c]", dep_cmd)):
        old = m.get(CLIP_KEY)
        if not isinstance(old, dict) or not old.get("meta_sha256"):
            raise StageError(f"{label} tidak memuat identitas klip (manifest lama) — tidak diketahui milik klip "
                             f"mana. Kalau output {stage} memang milik klip ini: {cmd} --adopt; "
                             f"kalau bukan / ragu: {cmd} --restart --yes")
        if old["meta_sha256"] != current["meta_sha256"]:
            raise StageError(f"{label} milik klip LAIN: output {stage} = {describe_identity(old)}, meta.json "
                             f"saat ini = {describe_identity(current)}. Jalankan {cmd} --restart --yes, atau "
                             f"ingest klip yang benar ke work_dir ini")
    for label, m, cmd in (("seg/manifest.json", seg_m, seg_cmd), ("depth/manifest.json", dep_m, dep_cmd)):
        if m.get("frame_size") != size:
            raise StageError(f"{label}: frame_size {m.get('frame_size')} ≠ meta.json {size} — jalankan ulang "
                             f"stage itu: {cmd} --restart --yes")
    classes = list(load_class_names())
    if seg_m.get("classes") != classes or seg_m.get("num_labels") != len(classes):
        raise StageError("seg/manifest.json: daftar kelas / num_labels ≠ src/rotoscope/data/sapiens2_classes.json "
                         f"— pemetaan kelas → grup akan salah. Jalankan ulang stage [2]: {seg_cmd} --restart --yes")
    return seg_m, dep_m


def require_inputs(clip: Clip, names: Sequence[str]) -> None:
    """File input tiap frame terpilih ada (validasi isi dilakukan saat dibaca)."""
    checks = (("seg/probs", clip.seg_clip.probs_path, "[2] segment"),
              ("seg/classmap", clip.seg_clip.classmap_path, "[2] segment"),
              ("depth", clip.depth_clip.depth_path, "[2c] depth"))
    for label, path_of, stage in checks:
        missing = [n for n in names if not path_of(n).is_file()]
        if missing:
            raise StageError(f"input {label} belum lengkap: {len(missing)} dari {len(names)} frame hilang, mis. "
                             f"{Path(missing[0]).stem} — jalankan stage {stage} sampai selesai")


def _read_inputs(clip: Clip, name: str, n_classes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = clip.height, clip.width
    probs = seg.read_probs(clip.seg_clip.probs_path(name))
    if probs is None or probs.dtype != np.uint8 or probs.shape != (n_classes, h, w):
        raise StageError(f"seg/probs {Path(name).stem} rusak / shape salah — jalankan ulang stage [2] segment "
                         f"(frame rusak diproses ulang)")
    cm = seg.read_classmap(clip.seg_clip.classmap_path(name))
    if cm is None or cm.shape != (h, w) or int(cm.max()) >= n_classes:
        raise StageError(f"seg/classmap {Path(name).stem} rusak — jalankan ulang stage [2] segment")
    d = dep.read_depth(clip.depth_clip.depth_path(name))
    if d is None or d.dtype != dep.DEPTH_DTYPE or d.shape != (h, w) or not np.isfinite(d).all():
        raise StageError(f"depth {Path(name).stem} rusak — jalankan ulang stage [2c] depth")
    return probs, cm, d


# ── Manifest ───────────────────────────────────────
def build_manifest(cfg: PipelineConfig, clip: Clip, seg_m: dict, dep_m: dict) -> dict:
    return {"stage": "stabilize",
            "stabilize": to_dict(cfg.stabilize), "stabilize_hash": section_hash(cfg, "stabilize"),
            "groups": to_dict(cfg.groups), "groups_hash": section_hash(cfg, "groups"),
            "seg": {k: seg_m.get(k) for k in SEG_REF_KEYS},
            "depth": {k: dep_m.get(k) for k in DEPTH_REF_KEYS},
            "frame_size": {"width": clip.width, "height": clip.height},
            CLIP_KEY: clip_identity(clip.work_dir),
            "outputs": dict(OUTPUT_INFO), "created_utc": utc_now()}


def _short(v) -> str:
    return v[:12] if isinstance(v, str) and len(v) > 12 else repr(v)


def manifest_diff(old: dict, new: dict) -> list[str]:
    """Field yang berubah; hash disingkat (lama → baru), referensi input per sub-field."""
    out = []
    for k in MANIFEST_MATCH_KEYS:
        a, b = old.get(k), new.get(k)
        if a == b:
            continue
        if isinstance(b, dict) and not isinstance(a, dict):
            a = {}  # field baru (mis. clip di manifest lama): tampilkan per sub-field, bukan repr dict
        if isinstance(a, dict) and isinstance(b, dict):
            out += [f"{k}.{s}: {_short(a.get(s))} → {_short(b.get(s))}"
                    for s in sorted(set(a) | set(b)) if a.get(s) != b.get(s)]
        else:
            out.append(f"{k}: {_short(a)} → {_short(b)}")
    return out


def _has_outputs(clip: Clip) -> bool:
    return (any(clip.groups_dir.glob("frame_*" + GROUPS_SUFFIX))
            or any(clip.depth_smooth_dir.glob("frame_*" + DEPTH_SUFFIX)))


def restart_outputs(clip: Clip) -> None:
    """Hapus output [3] (stable/ saja)."""
    if clip.stable_dir.exists():
        shutil.rmtree(clip.stable_dir)


# ── Run ────────────────────────────────────────────
def run_stabilize(cfg: PipelineConfig, *, restart: bool = False, limit: int | None = None,
                  log: Callable[[str], None] = print) -> dict:
    """Jalankan stage [3] (spasial). Return ringkasan run."""
    if limit is not None and limit < 1:
        raise StageError(f"--limit harus ≥ 1, dapat {limit}")
    st = cfg.stabilize
    if st.temporal.enabled:
        raise StageError("stabilize.temporal.enabled = true belum diimplementasi (T-302/T-303) — set false")
    clip = load_clip(cfg.paths.work_dir)
    selected = clip.names[:limit] if limit else clip.names
    seg_m, dep_m = load_inputs(clip)
    require_inputs(clip, selected)

    classes = load_class_names()
    lut = class_group_lut(cfg.groups, classes)
    n_groups = len(cfg.groups)

    manifest = build_manifest(cfg, clip, seg_m, dep_m)
    stale = []
    if restart:
        restart_outputs(clip)
    elif clip.manifest_path.is_file():
        old = json.loads(clip.manifest_path.read_text(encoding="utf-8"))
        stale = manifest_diff(old, manifest)
        if stale:
            log("PERINGATAN: output [3] basi (setelan / input berubah) — stable/ dihapus dan dihitung ulang:\n  "
                + "\n  ".join(stale))
            restart_outputs(clip)
    elif _has_outputs(clip):
        raise StageError(f"{clip.stable_dir} berisi output tanpa {MANIFEST_FILENAME} — asal output tidak "
                         f"diketahui. Jalankan {cli_cmd('stabilize', clip.work_dir)} --restart.")
    clean_tmp(clip.stable_dir)

    todo = [(i, n) for i, n in zip(clip.indices[:len(selected)], selected) if not clip.frame_valid(n, n_groups)]
    n_skip = len(selected) - len(todo)
    log(f"[3] stabilize (spasial, N={st.island_min_px}, K={st.mode_k}, {st.depth.normalize}): "
        f"{len(selected)} frame dipilih, {n_skip} valid dilewati, {len(todo)} diproses")

    run = {"selected": len(selected), "skipped": n_skip, "processed": 0, "stale": stale, "frames": []}
    if not todo:
        return run
    ensure_dir(clip.groups_dir)
    ensure_dir(clip.depth_smooth_dir)
    if not clip.manifest_path.is_file():
        write_json_atomic(clip.manifest_path, manifest)
    depth_log = last_frame_records(clip.depth_clip.frames_log)
    t_run = time.perf_counter()
    append_jsonl(clip.frames_log, {"event": "run_start", "time_utc": utc_now(), "n_todo": len(todo),
                                   "stabilize_hash": manifest["stabilize_hash"], "groups_hash": manifest["groups_hash"]})
    try:
        for k, (index, name) in enumerate(todo, 1):
            t0 = time.perf_counter()
            probs, cm, d = _read_inputs(clip, name, len(classes))
            t1 = time.perf_counter()
            gmap, gstats = clean_groups(group_sums(probs, lut, n_groups), lut[cm], st.island_min_px, st.mode_k)
            t2 = time.perf_counter()
            ds, dstats = normalize_depth(d, gmap != BACKGROUND_ID, st.depth.log_eps, st.depth.iqr_min)
            t3 = time.perf_counter()
            write_groups(clip.groups_path(name), gmap)
            write_depth_smooth(clip.depth_smooth_path(name), ds)
            t4 = time.perf_counter()
            rec = {"event": "frame", "frame": name, "index": index, "time_utc": utc_now(),
                   "read_s": round(t1 - t0, 4), "groups_s": round(t2 - t1, 4), "depth_s": round(t3 - t2, 4),
                   "write_s": round(t4 - t3, 4), "total_s": round(t4 - t0, 4), **gstats,
                   "depth_finite": depth_log.get(name, {}).get("finite"), "depth_smooth": dstats}
            append_jsonl(clip.frames_log, rec)
            run["frames"].append(rec)
            run["processed"] += 1
            log(f"  [{k}/{len(todo)}] {name} {rec['total_s']:.3f} s, seri {gstats['tie_px']} px, pulau "
                f"{gstats['island_changed_px']} px, mode {gstats['mode_changed_px']} px, depth_smooth "
                f"{dstats['min']:.3g}…{dstats['max']:.3g}" + (" (IQR di-clamp)" if dstats["iqr_clamped"] else ""))
    finally:
        append_jsonl(clip.frames_log, {"event": "run_end", "time_utc": utc_now(), "n_done": run["processed"],
                                       "wall_s": round(time.perf_counter() - t_run, 2)})
    return run


# ── Entry point stage (dipanggil cli.py, T-104b) ───
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m rotoscope.stabilize",
                                description="Stage [3]: seg/probs + depth → stable/ (peta grup + kedalaman ternormalisasi)")
    p.add_argument("--config", type=Path, default=None,
                   help=f"YAML pipeline (default: {DEFAULT_CONFIG.as_posix()} kalau ada, selain itu default kode)")
    add_work_dir_arg(p)
    p.add_argument("--restart", action="store_true", help="hapus output [3] lama (stable/) lalu hitung ulang")
    p.add_argument("--limit", type=int, default=None, help="hanya N frame pertama")
    args = p.parse_args(argv)
    reconfigure_stdio()

    def log(msg: str) -> None:
        print(msg, flush=True)

    try:
        path = args.config if args.config is not None else (DEFAULT_CONFIG if DEFAULT_CONFIG.is_file() else None)
        cfg = load_pipeline(path, overrides=work_dir_overrides(args.work_dir))
        t0 = time.perf_counter()
        run = run_stabilize(cfg, restart=args.restart, limit=args.limit, log=log)
        log(f"selesai: {run['processed']} diproses, {run['skipped']} dilewati ({time.perf_counter() - t0:.1f} s)")
    except (StageError, ConfigError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return EXIT_PRECONDITION
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
