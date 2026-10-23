"""Masker target [1b] dipakai di stage [3] (T-606, D-012): algoritma H (hibrida), per frame SEBELUM kernel temporal.
Kontrak lengkap: docs/01 [3] langkah 1b.

Dipanggil `stabilize.FrameCache.raw(k)` tepat sesudah `group_sums`, jadi setiap jendela kernel yang memuat frame k melihat
versi bermasker (masker ikut dihaluskan). Per frame adegan berklik (status [1b] `target` / `kosong`):
  1. F = foreground seg MENTAH frame itu (argmax jumlah grup ≠ 0, seri → grup classmap; dihitung pemanggil).
  2. M = peta label [1b] ≠ 0 (semua target). M_f = M + lubang terisi (lubang = komponen 4-arah ~M yang tidak menyentuh tepi
     gambar; `hole_fill_max_px` 0 = semua lubang, N = hanya lubang ≤ N px). M_d = M_f dilebarkan `dilate_px` (elips).
  3. Komponen 8-arah C dari F, r = |C ∩ M_f| / |C|: r ≥ r_keep → C utuh; r ≤ r_drop → C dibuang; di antaranya → C ∩ M_d,
     dipecah menjadi potongan 8-arah P; P dipertahankan hanya bila |P ∩ M_f| ≥ `piece_min_px`.
  4. R = F − yang dipertahankan → background one-hot: jumlah grup (255, 0, …, 0) dan grup seri 0. Piksel di luar F tidak
     diubah (argmax-nya sudah background). Frame `kosong` (M kosong): semua r = 0 → semua dibuang (tidak menggambar apa pun).
Frame `tanpa_prompt`: tidak disentuh (`no_prompt: keep`, default) atau R = F (`clear`).

Masker dipakai HANYA bila berlaku (keputusan T-606 no. 2 / 5 / 6): `target/points.json` + `isolate/` ada, `target_mask.enabled`, dan
kunci manifest [1b] = titik + sidik piksel SEMUA frame + model + parameter SEKARANG (fungsi [1b] `build_key` / `existing_state`
yang sama). Tidak berlaku → None + PERINGATAN (pemanggil jalan tanpa masker, byte-identik dengan klip tanpa titik).

numpy + cv2 saja (tanpa torch). `isolate` di-import di dalam `load_masks`: isolate meng-import stabilize, dan stabilize
meng-import modul ini — import tingkat modul akan melingkar.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from rotoscope.config import PipelineConfig, TargetMaskConfig, section_hash, to_dict
from rotoscope.stage_common import StageError, cli_cmd, clip_identity

BACKGROUND_ID = 0
PROBS_ONE = 255                       # = seg.PROBS_SCALE: jumlah grup background one-hot
CLASS_KEEP, CLASS_DROP, CLASS_CUT = "utuh", "buang", "potong"
NO_PROMPT_KEEP, NO_PROMPT_CLEAR = "keep", "clear"
STATUS_TARGET, STATUS_EMPTY, STATUS_NO_PROMPT = "target", "kosong", "tanpa_prompt"   # = isolate.STATUS_* (dicek test)
MASKED_STATUSES = (STATUS_TARGET, STATUS_EMPTY)
MANIFEST_KEY = "isolate"             # key manifest [3]; HANYA ditulis bila masker dipakai


# ── Algoritma H (murni) ────────────────────────────
def fill_holes(mask: np.ndarray, max_px: int = 0) -> np.ndarray:
    """M_f: M + lubangnya. Lubang = komponen 4-arah ~M yang tidak menyentuh tepi gambar (pasangan 8-arah M / 4-arah latar);
    max_px 0 = semua lubang diisi, N > 0 = hanya lubang ≤ N px."""
    n, lab, st, _ = cv2.connectedComponentsWithStats((~mask).astype(np.uint8), connectivity=4)
    border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    fill = np.zeros(n, bool)
    fill[1:] = True
    fill[border] = False
    if max_px > 0:
        fill &= st[:, cv2.CC_STAT_AREA] <= max_px
    fill[0] = False                       # label 0 = piksel M
    return mask | fill[lab]


def dilate(mask: np.ndarray, d: int) -> np.ndarray:
    """M_d: dilasi elips (2d + 1) × (2d + 1); d 0 = salinan."""
    if d <= 0:
        return mask.copy()
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * d + 1, 2 * d + 1))
    return cv2.dilate(mask.astype(np.uint8), ker) > 0


def component_class(r: float, p: TargetMaskConfig) -> str:
    if r >= p.r_keep:
        return CLASS_KEEP
    if r <= p.r_drop:
        return CLASS_DROP
    return CLASS_CUT


def apply_h(fg: np.ndarray, mask: np.ndarray, p: TargetMaskConfig) -> tuple[np.ndarray, dict]:
    """F (bool) + M (bool) → (piksel F yang DIPERTAHANKAN, statistik {kelas komponen, potongan dibuang, piksel})."""
    m_f = fill_holes(mask, p.hole_fill_max_px)
    m_d = dilate(m_f, p.dilate_px)
    n, lab, st, _ = cv2.connectedComponentsWithStats(fg.astype(np.uint8), connectivity=8)
    inter = np.bincount(lab[m_f], minlength=n)
    keep = np.zeros(fg.shape, bool)
    classes: Counter = Counter()
    pieces_dropped = 0
    for i in range(1, n):
        area = int(st[i, cv2.CC_STAT_AREA])
        cls = component_class(float(inter[i]) / area, p)
        classes[cls] += 1
        if cls == CLASS_DROP:
            continue
        x, y, w, h = (int(v) for v in st[i, :4])
        sl = (slice(y, y + h), slice(x, x + w))
        c = lab[sl] == i
        if cls == CLASS_KEEP:
            keep[sl] |= c
            continue
        cut = c & m_d[sl]
        pn, plab = cv2.connectedComponents(cut.astype(np.uint8), connectivity=8)
        pin = np.bincount(plab[cut & m_f[sl]], minlength=pn)
        ok = pin >= p.piece_min_px
        ok[0] = False
        pieces_dropped += int(pn - 1 - ok[1:].sum())
        keep[sl] |= ok[plab]
    stats = {"komponen": {k: classes.get(k, 0) for k in (CLASS_KEEP, CLASS_CUT, CLASS_DROP)},
             "potongan_dibuang": pieces_dropped, "fg_px": int(fg.sum()), "dibuang_px": int((fg & ~keep).sum())}
    return keep, stats


def to_background(sums: np.ndarray, tie: np.ndarray, remove: np.ndarray) -> None:
    """Piksel `remove` → background one-hot DI TEMPAT: jumlah grup (255, 0, …) dan grup seri 0."""
    sums[:, remove] = 0
    sums[BACKGROUND_ID, remove] = PROBS_ONE
    tie[remove] = BACKGROUND_ID


# ── Masker berlaku untuk satu klip ─────────────────
@dataclass(frozen=True)
class MaskSet:
    params: TargetMaskConfig
    status: dict[int, str]               # indeks frame → status [1b]
    labels_dir: Path
    block: dict                          # blok manifest [3] `isolate`

    def needs(self, index: int) -> bool:
        st = self.status.get(index)
        return st in MASKED_STATUSES or (st == STATUS_NO_PROMPT and self.params.no_prompt == NO_PROMPT_CLEAR)

    def apply(self, index: int, sums: np.ndarray, tie: np.ndarray, fg: np.ndarray) -> dict:
        """H di frame `index` (hanya bila `needs`): ubah `sums` / `tie` di tempat; return info untuk stable/frames.jsonl."""
        st = self.status[index]
        if st == STATUS_NO_PROMPT:                               # no_prompt: clear
            remove, info = fg, {"fg_px": int(fg.sum()), "dibuang_px": int(fg.sum())}
        else:
            from rotoscope.ingest import FRAME_PATTERN
            path = self.labels_dir / (FRAME_PATTERN % index)
            label = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_UNCHANGED) if path.is_file() else None
            if label is None or label.dtype != np.uint8 or label.shape != fg.shape:
                raise StageError(f"isolate/labels/{path.name} hilang / rusak padahal manifest [1b] berlaku — jalankan "
                                 f"ulang [1b]: {cli_cmd('isolate', self.labels_dir.parents[1])} --yes --restart")
            keep, info = apply_h(fg, label != 0, self.params)
            remove = fg & ~keep
        to_background(sums, tie, remove)
        return {"status": st, **info}


def labels_sha256(labels_dir: Path, status: dict[int, str]) -> str:
    """Sidik ISI masker yang dibaca [3]: sha256 gabungan (nama + sha256 byte PNG) frame `target` / `kosong`, urut indeks.
    Bukan cap waktu [1b]: [1b] diulang dengan hasil sama tidak membuat [3] basi; label berubah selalu basi."""
    from rotoscope.ingest import FRAME_PATTERN
    h = hashlib.sha256()
    for i in sorted(i for i, s in status.items() if s in MASKED_STATUSES):
        name = FRAME_PATTERN % i
        h.update(f"{name} {hashlib.sha256((labels_dir / name).read_bytes()).hexdigest()}\n".encode("ascii"))
    return h.hexdigest()


def manifest_block(cfg: PipelineConfig, iso_manifest: dict, status: dict[int, str], labels_dir: Path) -> dict:
    counts = Counter(status.values())
    return {"key_sha256": iso_manifest.get("key_sha256"), "labels_sha256": labels_sha256(labels_dir, status),
            "shots": [list(j["shot"]) for j in iso_manifest.get("jobs", [])],
            "frames": {s: counts.get(s, 0) for s in (STATUS_TARGET, STATUS_EMPTY, STATUS_NO_PROMPT)},
            "target_mask": to_dict(cfg.target_mask), "target_mask_hash": section_hash(cfg, "target_mask")}


def load_masks(cfg: PipelineConfig, work_dir: Path, log: Callable[[str], None] = print) -> MaskSet | None:
    """MaskSet bila masker [1b] berlaku untuk klip + frame SEKARANG; selain itu None (tanpa titik: diam; ada titik tetapi tidak
    berlaku / dimatikan: satu baris log / PERINGATAN). Biaya: sidik piksel semua frame (±4,7 s pada 182 frame 720 × 1280)."""
    from rotoscope import isolate as iso

    if not iso.points_present(work_dir):
        return None
    if not cfg.target_mask.enabled:
        log("[3] target_mask.enabled = false — masker target [1b] TIDAK dipakai (semua orang digambar)")
        return None
    clip = iso.load_clip(work_dir)
    cmd = f"{cli_cmd('isolate', work_dir)} --yes"
    try:
        if not clip.out_dir.is_dir():
            raise StageError(f"{iso.ISOLATE_DIRNAME}/ belum ada")
        frames_sha, _ = iso.frames_fingerprint(clip)
        key = iso.build_key(cfg, hashlib.sha256(clip.points_path.read_bytes()).hexdigest(), frames_sha)
        state, reasons = iso.existing_state(clip, key, clip_identity(work_dir))
        if state != "ok":
            raise StageError("; ".join(reasons[:4]) or state)
    except (StageError, OSError, ValueError) as e:
        log(f"PERINGATAN: target/points.json ada, tetapi masker [1b] TIDAK berlaku ({e}) — [3] dijalankan TANPA isolasi "
            f"target (semua orang digambar). Hitung ulang masker: {cmd}")
        return None
    m = iso.read_manifest(clip.out_dir / iso.MANIFEST_FILENAME) or {}
    status = {int(r["index"]): str(r["status"]) for r in iso.read_frame_rows(clip.out_dir / iso.FRAMES_LOG_FILENAME)}
    labels = clip.out_dir / iso.LABELS_DIRNAME
    return MaskSet(cfg.target_mask, status, labels, manifest_block(cfg, m, status, labels))
