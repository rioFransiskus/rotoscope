"""`python -m rotoscope pick <video> [--frame N] [--config P] [--dry-run]` (T-604): jendela klik target → target/points.json.

BUKAN stage pipeline: `run` tidak memanggilnya dan tidak ada stage yang membaca `target/` (konsumen pertama = T-605).
Bila `meta.json` belum ada, cli menjalankan [1] ingest dulu (CPU); modul ini tidak pernah menulis `frames/` / `meta.json`.

Dua lapis:
- logika murni (tanpa jendela, diuji di tests/test_pick.py): `View` (skala + pemetaan koordinat tampilan ↔ frame),
  `PickState` (state machine mouse / tombol), format + validasi + gabung `points.json`, `render_canvas`;
- lapis tipis GUI Windows: `ensure_dpi_aware`, `probe_display` (area kerja), `CvGui` (jendela OpenCV WIN32UI,
  `WINDOW_AUTOSIZE`, tanpa ubah ukuran oleh sistem).

Format `rotoscope-target-points/1` (docs/01 "`pick.py`"): koordinat = piksel frame kerja (pusat piksel kiri-atas = 0, 0);
tiap entri frame terikat `frame_pixel_sha256` = sha256(b"rgb8 <W>x<H>\\n" + piksel RGB uint8 hasil decode) — sidik PIKSEL,
bukan byte PNG (keputusan Rio 2026-10-16): ingest ulang yang hanya mengubah kompresi PNG tidak membuat titik basi.

Konstanta UI (lebar panel, ambang tarik, warna, ...) sengaja konstanta modul, bukan YAML: tidak memengaruhi keluaran
pipeline (keputusan Rio T-604 setelah Tahap 1 no. 5).
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import math
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from rotoscope.ingest import FRAME_PATTERN, FRAMES_DIRNAME
from rotoscope.stage_common import (
    EXIT_OK, EXIT_PRECONDITION, StageError, clip_identity, config_flag, load_frame_list, read_rgb, trim_flags, utc_now,
    write_bytes_atomic, write_json_atomic,
)

EXIT_INTERRUPTED = 130
PROG = "python -m rotoscope"

# ── Berkas ─────────────────────────────────────────
POINTS_FORMAT = "rotoscope-target-points/1"
FORMAT_RE = re.compile(r"^rotoscope-target-points/(\d+)$")
FORMAT_VERSION = 1
T602_FORMATS = ("t602-points/1", "t602-points-arsip/1")
CONVERT_SCRIPT = "scripts/t604_convert_t602.py"
TARGET_DIRNAME = "target"
POINTS_FILENAME = "points.json"
PREV_FILENAME = "points.prev.json"
DRYRUN_DIRNAME = "pick_dryrun"
TOOL = f"{PROG} pick"
COORDS_DOC = ("piksel frame kerja frames/ (x ke kanan, y ke bawah; pusat piksel kiri-atas = 0, 0); label 1 = target, "
              "0 = bukan target (konvensi SAM); box = [x0, y0, x1, y1] dengan x0 < x1, y0 < y1")
HASH_DOC = ("frame_pixel_sha256 = sha256(b'rgb8 <W>x<H>\\n' + piksel RGB uint8 hasil decode, baris demi baris) - "
            "sidik piksel, bukan byte PNG")
PIXEL_HASH_HEADER = "rgb8 {w}x{h}\n"
COORD_DECIMALS = 1
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
STATUS_OK, STATUS_STALE, STATUS_MISSING = "ok", "basi", "hilang"

# ── UI (konstanta modul, bukan YAML) ───────────────
PANEL_W = 400                    # lebar panel petunjuk di kanan gambar
PANEL_MIN_H = 600
PANEL_MARGIN = 14
WINDOW_RESERVE = (16, 56)        # bingkai jendela ±8 px per sisi + judul ±31 px (125%) + margin
DRAG_MIN_PX = 6                  # tarik ≥ 6 px tampilan = kotak; kurang = klik
MAX_SCALE = 1.0                  # tanpa upscale
WAIT_MS = 30
FALLBACK_WORK_AREA = (0, 0, 1280, 720)
TARGET_COLORS = ((0, 200, 0), (255, 160, 0), (0, 215, 255), (255, 0, 255), (0, 140, 255), (255, 255, 0))   # BGR
TARGET_COLOR_NAMES = ("hijau", "biru", "kuning", "ungu", "oranye", "toska")
NEG_COLOR = (0, 0, 255)
BG_GRAY = 32
TEXT_WHITE, TEXT_GREY, TEXT_YELLOW = (235, 235, 235), (170, 170, 170), (0, 230, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX
LINE_GAP_PX = 10
KEY_ENTER, KEY_ESC = (13, 10), 27

# ── Win32 ──────────────────────────────────────────
SPI_GETWORKAREA = 0x0030
PROCESS_PER_MONITOR_DPI_AWARE = 2
_windll = getattr(ctypes, "windll", None)        # seam test (ctypes palsu); None di luar Windows

# ── State machine ──────────────────────────────────
MODE_NORMAL, MODE_PRESS, MODE_DRAG = "normal", "tekan", "tarik"
MODE_CONFIRM_DELETE, MODE_CONFIRM_EXIT = "konfirmasi-hapus", "konfirmasi-keluar"
ACT_SAVE, ACT_DELETE, ACT_CLOSE, ACT_EXIT = "simpan", "hapus", "tutup", "keluar"


class PickError(Exception):
    """Prasyarat pick gagal (exit 1); pesan siap cetak."""


class PointsError(ValueError):
    """points.json tidak valid (pick menolak membuka / menimpanya)."""


# ── Sidik frame (sha256 piksel) ────────────────────
def pixel_sha256(rgb: np.ndarray) -> str:
    """sha256(b"rgb8 <W>x<H>\\n" + byte piksel RGB uint8 baris demi baris). Ukuran ikut di-hash: frame 2 × 6 dan 6 × 2 dengan
    byte piksel sama berbeda sidiknya."""
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"sidik piksel butuh array RGB uint8 H × W × 3, dapat {rgb.dtype} {rgb.shape}")
    h, w = rgb.shape[:2]
    d = hashlib.sha256(PIXEL_HASH_HEADER.format(w=w, h=h).encode("ascii"))
    d.update(np.ascontiguousarray(rgb).tobytes())
    return d.hexdigest()


def frame_pixel_sha256(path: Path) -> str:
    """Sidik piksel berkas frame (decode = `stage_common.read_rgb`, sama dengan stage pipeline). StageError bila tak terbaca."""
    return pixel_sha256(read_rgb(path))


# ── Tampilan: skala + pemetaan koordinat ──────────
def _half_up(v: float) -> int:
    return int(math.floor(v + 0.5))


@dataclass(frozen=True)
class View:
    """Gambar frame W × H ditampilkan dw × dh (skala s) di kiri kanvas; panel petunjuk PANEL_W px di kanan.
    Pemetaan memakai sx = dw / W, sy = dh / H (bukan s) supaya tepi gambar tepat; konvensi pusat piksel."""

    width: int
    height: int
    scale: float
    dw: int
    dh: int
    canvas_w: int
    canvas_h: int
    max_w: int
    max_h: int

    @property
    def sx(self) -> float:
        return self.dw / self.width

    @property
    def sy(self) -> float:
        return self.dh / self.height

    @property
    def fits(self) -> bool:
        return self.canvas_w <= self.max_w and self.canvas_h <= self.max_h

    def in_image(self, dx: int, dy: int) -> bool:
        return 0 <= dx < self.dw and 0 <= dy < self.dh

    def clamp_disp(self, dx: int, dy: int) -> tuple[int, int]:
        return min(max(int(dx), 0), self.dw - 1), min(max(int(dy), 0), self.dh - 1)

    def to_frame(self, dx: float, dy: float) -> tuple[float, float]:
        """Piksel tampilan → koordinat frame: x = (dx + 0,5) / sx − 0,5, clamp ke [0, W − 1], bulat 0,1."""
        x = (dx + 0.5) / self.sx - 0.5
        y = (dy + 0.5) / self.sy - 0.5
        x = min(max(x, 0.0), self.width - 1.0)
        y = min(max(y, 0.0), self.height - 1.0)
        return round(x, COORD_DECIMALS), round(y, COORD_DECIMALS)

    def to_disp(self, x: float, y: float) -> tuple[int, int]:
        """Koordinat frame → piksel tampilan: dx = round((x + 0,5) · sx − 0,5) (half-up) = floor((x + 0,5) · sx)."""
        dx = math.floor((x + 0.5) * self.sx)
        dy = math.floor((y + 0.5) * self.sy)
        return min(max(dx, 0), self.dw - 1), min(max(dy, 0), self.dh - 1)


def fit_view(width: int, height: int, avail_w: int, avail_h: int) -> View:
    """s = min(1, H_maks / H, (W_maks − panel) / W); batas kanvas = area kerja − cadangan bingkai jendela."""
    max_w, max_h = avail_w - WINDOW_RESERVE[0], avail_h - WINDOW_RESERVE[1]
    s = min(MAX_SCALE, max_h / height, (max_w - PANEL_W) / width)
    if s <= 0:
        raise PickError(f"layar terlalu kecil untuk pick (area kerja {avail_w} × {avail_h} px)")
    dw, dh = max(1, _half_up(width * s)), max(1, _half_up(height * s))
    return View(width, height, s, dw, dh, dw + PANEL_W, max(dh, PANEL_MIN_H), max_w, max_h)


def roundtrip_report(view: View) -> dict:
    """Tampilan → frame → tampilan = identitas untuk SETIAP piksel tampilan (sumbu x dan y independen, jadi cukup tiap kolom
    + tiap baris); galat frame → tampilan → frame maksimum untuk piksel frame bulat."""
    ok_x = all(view.to_disp(view.to_frame(dx, 0)[0], 0)[0] == dx for dx in range(view.dw))
    ok_y = all(view.to_disp(0, view.to_frame(0, dy)[1])[1] == dy for dy in range(view.dh))
    err_x = max(abs(view.to_frame(view.to_disp(x, 0)[0], 0)[0] - x) for x in range(view.width))
    err_y = max(abs(view.to_frame(0, view.to_disp(0, y)[1])[1] - y) for y in range(view.height))
    return {"disp_identity": ok_x and ok_y, "max_frame_err_px": round(max(err_x, err_y), 3),
            "bound_px": round(0.5 / min(view.sx, view.sy) + 0.05, 3)}


# ── Warna target ───────────────────────────────────
def target_color(tid: int) -> tuple[int, int, int]:
    return TARGET_COLORS[(tid - 1) % len(TARGET_COLORS)]


def target_color_name(tid: int) -> str:
    return TARGET_COLOR_NAMES[(tid - 1) % len(TARGET_COLOR_NAMES)]


def _ids(ids) -> str:
    return ", ".join(str(i) for i in ids)


# ── State machine (satu frame per sesi) ────────────
def key_kind(code: int) -> str:
    if code < 0 or code > 0xFFFF:                 # waitKeyEx: tombol khusus (panah, F1, ...) = kode << 16
        return "other"
    if code in KEY_ENTER or code in (ord("s"), ord("S")):
        return "enter"
    if code == KEY_ESC:
        return "esc"
    ch = chr(code).lower() if code < 0x110000 else ""
    return ch if ch in ("n", "u", "r") else "other"


class PickState:
    """Titik + kotak per target di SATU frame. `history` = aksi yang bisa di-U: ("point", id) | ("box", id, kotak_lama) |
    ("reset", snapshot). Titik lama yang dimuat masuk riwayat dengan urutan simpanan (id naik, titik urut klik, kotak terakhir)."""

    def __init__(self, view: View, *, loaded_targets: list[dict] | None = None, known_other_ids=(),
                 entry_exists: bool = False):
        self.view = view
        self.targets: dict[int, dict] = {}
        self.history: list[tuple] = []
        self.active = 1
        self.mode = MODE_NORMAL
        self.press: tuple[int, int] | None = None
        self.cur: tuple[int, int] | None = None
        self.msg = ""
        self.known_other = {int(i) for i in known_other_ids}
        self.entry_exists = entry_exists
        for t in sorted(loaded_targets or (), key=lambda t: int(t["id"])):
            tid = int(t["id"])
            tt = self._target(tid)
            for p in t.get("points") or ():
                tt["points"].append({"x": float(p["x"]), "y": float(p["y"]), "label": int(p["label"])})
                self.history.append(("point", tid))
            if t.get("box") is not None:
                tt["box"] = [float(v) for v in t["box"]]
                self.history.append(("box", tid, None))
        self.initial = self._signature()

    # ── isi ──
    def _target(self, tid: int) -> dict:
        return self.targets.setdefault(tid, {"points": [], "box": None})

    def content(self) -> dict[int, dict]:
        """Target berisi (titik atau kotak), urut id. Target kosong TIDAK disimpan (keputusan 2026-10-16 no. 4)."""
        return {tid: t for tid, t in sorted(self.targets.items()) if t["points"] or t["box"] is not None}

    def _signature(self) -> tuple:
        return tuple((tid, tuple((p["x"], p["y"], p["label"]) for p in t["points"]),
                      tuple(t["box"]) if t["box"] is not None else None) for tid, t in self.content().items())

    @property
    def dirty(self) -> bool:
        """Ada perubahan yang belum disimpan (isi berbeda dengan saat dibuka; titik lalu U = tidak berubah)."""
        return self._signature() != self.initial

    def known_ids(self) -> set[int]:
        return self.known_other | set(self.content())

    def no_negative(self) -> list[int]:
        return [tid for tid, t in self.content().items() if not any(p["label"] == 0 for p in t["points"])]

    def negative_only(self) -> list[int]:
        return [tid for tid, t in self.content().items()
                if t["box"] is None and not any(p["label"] == 1 for p in t["points"])]

    # ── aksi ──
    def add_point(self, dx: int, dy: int, label: int) -> None:
        x, y = self.view.to_frame(dx, dy)
        self._target(self.active)["points"].append({"x": x, "y": y, "label": label})
        self.history.append(("point", self.active))
        self.msg = ""

    def set_box(self, dx0: int, dy0: int, dx1: int, dy1: int) -> None:
        if min(abs(dx1 - dx0), abs(dy1 - dy0)) < DRAG_MIN_PX:
            self.msg = f"Kotak terlalu tipis (< {DRAG_MIN_PX} px) - diabaikan."
            return
        x0, y0 = self.view.to_frame(min(dx0, dx1), min(dy0, dy1))
        x1, y1 = self.view.to_frame(max(dx0, dx1), max(dy0, dy1))
        if not (x0 < x1 and y0 < y1):
            self.msg = "Kotak terlalu kecil - diabaikan."
            return
        t = self._target(self.active)
        old = t["box"]
        self.history.append(("box", self.active, old))
        t["box"] = [x0, y0, x1, y1]
        self.msg = f"Kotak target {self.active} dipasang" + (" (menggantikan yang lama)." if old is not None else ".")

    def undo(self) -> None:
        if not self.history:
            self.msg = "Tidak ada yang bisa dibatalkan."
            return
        act = self.history.pop()
        if act[0] == "point":
            p = self.targets[act[1]]["points"].pop()
            self.msg = f"Dibatalkan: titik {'kiri' if p['label'] == 1 else 'kanan'} target {act[1]}."
        elif act[0] == "box":
            self.targets[act[1]]["box"] = act[2]
            self.msg = f"Dibatalkan: kotak target {act[1]}" + (" (kotak lama kembali)." if act[2] is not None else ".")
        else:
            self.targets = act[1]
            self.msg = "Dibatalkan: R (titik frame ini kembali)."

    def reset(self) -> None:
        if not self.content():
            self.msg = "Frame ini sudah kosong."
            return
        self.history.append(("reset", self.targets))
        self.targets = {}
        self.msg = "Frame ini dikosongkan (U = batal)."

    def next_target(self) -> None:
        """N: berputar 1 … (id terbesar yang dikenal + 1) → 1; id dikenal = id di frame lain berkas + id berisi di frame ini."""
        top = max(self.known_ids(), default=0) + 1
        self.active = self.active + 1 if self.active + 1 <= top else 1
        self.msg = f"Sekarang target {self.active} ({target_color_name(self.active)})."

    def _leave_confirm(self) -> None:
        if self.mode in (MODE_CONFIRM_DELETE, MODE_CONFIRM_EXIT):
            self.mode = MODE_NORMAL
            self.msg = "Konfirmasi dibatalkan."

    def _drop_gesture(self) -> None:
        if self.mode in (MODE_PRESS, MODE_DRAG):
            self.mode, self.press, self.cur = MODE_NORMAL, None, None

    # ── mouse ──
    def mouse(self, event: int, dx: int, dy: int, flags: int = 0) -> None:
        v = self.view
        if event == cv2.EVENT_LBUTTONDOWN:
            if not v.in_image(dx, dy):            # klik di panel diabaikan
                return
            self._leave_confirm()
            self.mode, self.press, self.cur = MODE_PRESS, (dx, dy), (dx, dy)
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.mode not in (MODE_PRESS, MODE_DRAG):
                return
            if not flags & cv2.EVENT_FLAG_LBUTTON:  # tombol dilepas di luar jendela: selesai di posisi terakhir yang terlihat
                self._finish_left(*self.cur)
                return
            self.cur = v.clamp_disp(dx, dy)
            if self.mode == MODE_PRESS and max(abs(self.cur[0] - self.press[0]),
                                               abs(self.cur[1] - self.press[1])) >= DRAG_MIN_PX:
                self.mode = MODE_DRAG
        elif event == cv2.EVENT_LBUTTONUP:
            if self.mode in (MODE_PRESS, MODE_DRAG):
                self._finish_left(*v.clamp_disp(dx, dy))
        elif event == cv2.EVENT_RBUTTONDOWN:
            if not v.in_image(dx, dy) or self.mode in (MODE_PRESS, MODE_DRAG):
                return
            self._leave_confirm()
            self.add_point(dx, dy, 0)

    def _finish_left(self, dx: int, dy: int) -> None:
        mode, (px, py) = self.mode, self.press
        self.mode, self.press, self.cur = MODE_NORMAL, None, None
        if mode == MODE_DRAG:
            self.set_box(px, py, dx, dy)
        else:
            self.add_point(px, py, 1)

    # ── tombol ──
    def key(self, code: int) -> str | None:
        """Tombol → aksi sesi (ACT_*) atau None (jendela tetap terbuka)."""
        kind = key_kind(code)
        self._drop_gesture()
        if self.mode == MODE_CONFIRM_EXIT and kind != "esc":
            self.mode = MODE_NORMAL
            self.msg = "Keluar dibatalkan - lanjutkan."
            if kind == "other":
                return None
        if self.mode == MODE_CONFIRM_DELETE and kind != "enter":
            self.mode = MODE_NORMAL
            self.msg = "Hapus dibatalkan."
            if kind in ("other", "esc"):
                return None
        if kind == "esc":
            return self.request_exit()
        if kind == "enter":
            return self.request_save()
        if kind == "n":
            self.next_target()
        elif kind == "u":
            self.undo()
        elif kind == "r":
            self.reset()
        return None

    def request_exit(self) -> str | None:
        """Esc / tombol X jendela: tanpa perubahan → keluar; dengan perubahan → konfirmasi dulu (Esc / X kedua = keluar)."""
        self._drop_gesture()
        if self.mode == MODE_CONFIRM_EXIT or not self.dirty:
            return ACT_EXIT
        self.mode = MODE_CONFIRM_EXIT
        self.msg = ("Ada perubahan BELUM disimpan. Esc lagi (atau tutup jendela lagi) = keluar TANPA simpan; "
                    "Enter / S = simpan; tombol lain = lanjut.")
        return None

    def request_save(self) -> str | None:
        content = self.content()
        if not content:
            if self.mode == MODE_CONFIRM_DELETE:
                self.mode = MODE_NORMAL
                return ACT_DELETE
            if self.entry_exists:
                self.mode = MODE_CONFIRM_DELETE
                self.msg = "Frame ini kosong. Enter / S lagi = HAPUS frame ini dari points.json; tombol lain = batal."
                return None
            self.msg = "Belum ada titik - klik kiri di target (atau tarik kotak) dulu."
            return None
        self.mode = MODE_NORMAL
        bad = self.negative_only()
        if bad:
            self.msg = f"Target {_ids(bad)} hanya punya klik kanan - tambah klik kiri / kotak, atau U."
            return None
        return ACT_SAVE if self.dirty else ACT_CLOSE


# ── Validasi points.json ───────────────────────────
def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def validate_doc(doc: Any) -> None:
    """Skema `rotoscope-target-points/1`; PointsError dengan lokasi kesalahan. Frame tanpa target, target kosong, target hanya
    klik kanan tanpa kotak = tidak valid (pick tidak pernah menulisnya)."""
    if not isinstance(doc, dict):
        raise PointsError("isi bukan objek JSON")
    fmt = doc.get("format")
    if fmt != POINTS_FORMAT:
        if fmt in T602_FORMATS:
            raise PointsError(f"format {fmt} (studi T-602) - konversi dulu dengan {CONVERT_SCRIPT}")
        m = FORMAT_RE.match(fmt) if isinstance(fmt, str) else None
        if m and int(m.group(1)) > FORMAT_VERSION:
            raise PointsError(f"format {fmt} dibuat versi pick yang lebih baru (versi ini membaca {POINTS_FORMAT})")
        raise PointsError(f"format {fmt!r} bukan {POINTS_FORMAT}")
    clip = doc.get("clip")
    if not (isinstance(clip, dict) and isinstance(clip.get("meta_sha256"), str)
            and isinstance(clip.get("source_path"), str)):
        raise PointsError("clip harus {meta_sha256, source_path} (string)")
    fs = doc.get("frame_size")
    if not (isinstance(fs, dict) and _is_int(fs.get("width")) and _is_int(fs.get("height"))
            and fs["width"] > 0 and fs["height"] > 0):
        raise PointsError("frame_size harus {width, height} bulat > 0")
    w, h = fs["width"], fs["height"]
    frames = doc.get("frames")
    if not isinstance(frames, list):
        raise PointsError("frames harus daftar")
    seen: set[int] = set()
    for e in frames:
        if not (isinstance(e, dict) and _is_int(e.get("frame_index")) and e["frame_index"] >= 0):
            raise PointsError("tiap entri frames harus punya frame_index bulat >= 0")
        f = e["frame_index"]
        where = f"frame {f}"
        if f in seen:
            raise PointsError(f"{where} muncul dua kali")
        seen.add(f)
        if e.get("frame_file") != FRAME_PATTERN % f:
            raise PointsError(f"{where}: frame_file {e.get('frame_file')!r} != {FRAME_PATTERN % f}")
        sha = e.get("frame_pixel_sha256")
        if not (isinstance(sha, str) and SHA_RE.match(sha)):
            raise PointsError(f"{where}: frame_pixel_sha256 harus 64 hex kecil")
        targets = e.get("targets")
        if not isinstance(targets, list) or not targets:
            raise PointsError(f"{where}: tanpa target")
        ids: set[int] = set()
        for t in targets:
            if not (isinstance(t, dict) and _is_int(t.get("id")) and t["id"] >= 1):
                raise PointsError(f"{where}: id target harus bulat >= 1")
            tid = t["id"]
            tw = f"{where} target {tid}"
            if tid in ids:
                raise PointsError(f"{tw} muncul dua kali")
            ids.add(tid)
            pts = t.get("points")
            if not isinstance(pts, list):
                raise PointsError(f"{tw}: points harus daftar")
            for k, p in enumerate(pts, 1):
                pw = f"{tw} titik #{k}"
                if not isinstance(p, dict):
                    raise PointsError(f"{pw}: bukan objek")
                lab = p.get("label")
                if not _is_int(lab) or lab not in (0, 1):
                    raise PointsError(f"{pw}: label {lab!r} bukan 0 / 1")
                for ax, lim in (("x", w), ("y", h)):
                    v = p.get(ax)
                    if not _is_num(v):
                        raise PointsError(f"{pw}: {ax} {v!r} bukan bilangan hingga")
                    if not 0 <= v <= lim - 1:
                        raise PointsError(f"{pw}: {ax} {v} di luar frame (0-{lim - 1})")
            box = t.get("box")
            if box is not None:
                if not (isinstance(box, list) and len(box) == 4 and all(_is_num(v) for v in box)):
                    raise PointsError(f"{tw}: box harus [x0, y0, x1, y1] atau null")
                x0, y0, x1, y1 = box
                if not (0 <= x0 < x1 <= w - 1 and 0 <= y0 < y1 <= h - 1):
                    raise PointsError(f"{tw}: box {box} terbalik / di luar frame")
            if not pts and box is None:
                raise PointsError(f"{tw}: tanpa titik dan tanpa kotak")
            if box is None and not any(p["label"] == 1 for p in pts):
                raise PointsError(f"{tw}: hanya klik kanan (label 0) tanpa kotak")


def load_points(path: Path) -> dict | None:
    """None bila berkas belum ada; PointsError bila ada tetapi tidak valid (pesan memuat path)."""
    if not path.is_file():
        return None
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as e:
        raise PointsError(f"{path} tidak valid: bukan JSON yang terbaca ({e})") from None
    try:
        validate_doc(doc)
    except PointsError as e:
        raise PointsError(f"{path} tidak valid: {e}") from None
    return doc


def _same_source(a: str, b: str) -> bool:
    return os.path.normcase(os.path.normpath(a)) == os.path.normcase(os.path.normpath(b))


def check_doc_matches_clip(doc: dict, identity: dict, width: int, height: int, path: Path, pick_cmd: str | None = None) -> None:
    """Berkas milik klip lain / ukuran frame lain → PickError (pick tidak membukanya dan tidak menimpanya). `pick_cmd` = perintah
    `pick <video> [--start S --end E]` (tanpa --frame) untuk pesan ukuran beda (T-607, docs/04 persetujuan rancangan no. 9)."""
    old = doc["clip"]["source_path"]
    if not _same_source(old, identity["source_path"]):
        raise PickError(f"{path} milik klip lain ({old}; klip ini: {identity['source_path']}) - pindahkan berkas itu; "
                        f"pick tidak menimpanya")
    fs = doc["frame_size"]
    if (fs["width"], fs["height"]) != (width, height):
        first = min((e["frame_index"] for e in doc.get("frames", [])), default=0)
        cmd = pick_cmd or f"{PROG} pick <video>"
        raise PickError(f"{path}: titik dibuat untuk frame {fs['width']} x {fs['height']}, frames/ sekarang {width} x {height} - "
                        f"frame berubah sejak diklik (ingest memakai crop / trim / ukuran kerja lain). Semua titik di berkas itu "
                        f"basi dan tidak dipakai. Pindahkan berkas itu (mis. ganti nama menjadi points.old.json), lalu klik ulang "
                        f"target: {cmd} --frame {first}; pick tidak menimpanya")


def frame_status(entry: dict, work_dir: Path, indices: set[int], cache: dict[int, str | None]) -> str:
    """ok (sidik piksel sama) | basi (berbeda: isi frame berubah, mis. ingest dari video / parameter lain) | hilang."""
    f = entry["frame_index"]
    if f not in indices:
        return STATUS_MISSING
    if f not in cache:
        try:
            cache[f] = frame_pixel_sha256(work_dir / FRAMES_DIRNAME / (FRAME_PATTERN % f))
        except StageError:
            cache[f] = None
    if cache[f] is None:
        return STATUS_MISSING
    return STATUS_OK if cache[f] == entry["frame_pixel_sha256"] else STATUS_STALE


def frame_statuses(doc: dict | None, work_dir: Path, indices: set[int], cache: dict[int, str | None]) -> dict[int, str]:
    return {e["frame_index"]: frame_status(e, work_dir, indices, cache) for e in (doc or {}).get("frames", [])}


# ── Gabung + tulis ─────────────────────────────────
def new_doc(identity: dict, width: int, height: int, frames: list[dict]) -> dict:
    return {"format": POINTS_FORMAT,
            "clip": {"meta_sha256": identity["meta_sha256"], "source_path": identity["source_path"]},
            "frame_size": {"width": width, "height": height}, "coords": COORDS_DOC, "frame_hash": HASH_DOC,
            "frames": sorted(frames, key=lambda e: e["frame_index"]), "tool": TOOL}


def merge_entry(disk: dict | None, frame: int, entry: dict | None, identity: dict, width: int, height: int) -> dict:
    """Ganti / hapus SATU entri frame; entri frame lain apa adanya (dari berkas di disk saat simpan)."""
    frames = [e for e in (disk or {}).get("frames", []) if e["frame_index"] != frame]
    if entry is not None:
        frames.append(entry)
    return new_doc(identity, width, height, frames)


def targets_json(state: PickState) -> list[dict]:
    return [{"id": tid, "points": [dict(p) for p in t["points"]],
             "box": list(t["box"]) if t["box"] is not None else None}
            for tid, t in state.content().items()]


@dataclass(frozen=True)
class DisplayInfo:
    screen_wh: tuple[int, int]
    work_area: tuple[int, int, int, int]      # kiri, atas, kanan, bawah (ruang koordinat proses)
    dpi_aware: bool
    dpi: int | None = None
    notes: tuple[str, ...] = ()

    @property
    def work_wh(self) -> tuple[int, int]:
        return self.work_area[2] - self.work_area[0], self.work_area[3] - self.work_area[1]


def build_entry(frame: int, pixel_sha: str, state: PickState, display: DisplayInfo) -> dict:
    return {"frame_index": frame, "frame_file": FRAME_PATTERN % frame, "frame_pixel_sha256": pixel_sha,
            "updated_utc": utc_now(),
            "display": {"scale": round(state.view.scale, 4), "screen_wh": list(display.screen_wh),
                        "work_area_wh": list(display.work_wh), "dpi_aware": display.dpi_aware},
            "targets": targets_json(state)}


def write_points(path: Path, doc: dict) -> bool:
    """Tulis atomik; isi lama disalin dulu ke points.prev.json (cadangan satu tingkat). Return True bila ada cadangan."""
    validate_doc(doc)
    path.parent.mkdir(parents=True, exist_ok=True)
    backed_up = path.is_file()
    if backed_up:
        write_bytes_atomic(path.with_name(PREV_FILENAME), path.read_bytes())
    write_json_atomic(path, doc)
    return backed_up


# ── Sesi ───────────────────────────────────────────
def _quote(p: str) -> str:
    return f'"{p}"' if " " in p else p


@dataclass
class Session:
    work_dir: Path
    clip: str
    video: str
    frame: int
    indices: tuple[int, ...]
    width: int
    height: int
    frame_path: Path
    frame_rgb: np.ndarray
    pixel_sha: str
    identity: dict
    points_path: Path
    doc: dict | None
    status: dict[int, str]
    notes: list[str] = field(default_factory=list)

    @property
    def entry(self) -> dict | None:
        return next((e for e in (self.doc or {}).get("frames", []) if e["frame_index"] == self.frame), None)

    def other_ids(self) -> dict[int, list[int]]:
        """id target → frame LAIN di berkas yang memuatnya."""
        out: dict[int, list[int]] = {}
        for e in (self.doc or {}).get("frames", []):
            if e["frame_index"] != self.frame:
                for t in e["targets"]:
                    out.setdefault(t["id"], []).append(e["frame_index"])
        return dict(sorted(out.items()))

    def pick_cmd(self, frame: int) -> str:
        return f"{PROG} pick {_quote(self.video)}{trim_flags(self.work_dir)}{config_flag()} --frame {frame}"


def open_session(work_dir: Path, frame: int, video: str) -> Session:
    """Baca meta.json + frame + points.json (divalidasi) + status tiap frame berkas. PickError = exit 1."""
    clip = work_dir.name
    ingest_cmd = f"{PROG} ingest {_quote(video)}{trim_flags(work_dir)}{config_flag()}"   # T-607: siap tempel (trim + --config)
    try:
        _names, indices, width, height = load_frame_list(work_dir)
        identity = clip_identity(work_dir)
    except (StageError, OSError, ValueError, KeyError, TypeError) as e:
        raise PickError(f"meta.json di {work_dir} tidak terbaca ({e}) - jalankan {ingest_cmd}") from None
    if not indices:
        raise PickError(f"klip {clip} tanpa frame (meta.json frame_count 0) - jalankan {ingest_cmd}")
    if not indices[0] <= frame <= indices[-1]:
        raise PickError(f"frame {frame} di luar klip {clip} ({len(indices)} frame: {indices[0]}-{indices[-1]})")
    frames_dir = work_dir / FRAMES_DIRNAME
    frame_path = frames_dir / (FRAME_PATTERN % frame)
    try:
        rgb = read_rgb(frame_path)
    except StageError:
        raise PickError(f"{frame_path.name} hilang / tidak terbaca di {frames_dir} - jalankan {ingest_cmd} "
                        f"(menulis ulang frames/; titik pada frame yang isinya berubah menjadi basi)") from None
    if rgb.shape[:2] != (height, width):
        raise PickError(f"{frame_path.name} berukuran {rgb.shape[1]} x {rgb.shape[0]}, meta.json {width} x {height} - "
                        f"jalankan {ingest_cmd}")
    pixel_sha = pixel_sha256(rgb)
    points_path = work_dir / TARGET_DIRNAME / POINTS_FILENAME
    try:
        doc = load_points(points_path)
    except PointsError as e:
        raise PickError(f"{e} - perbaiki atau pindahkan berkas itu; pick tidak menimpanya") from None
    s = Session(work_dir, clip, video, frame, indices, width, height, frame_path, rgb, pixel_sha, identity, points_path,
                doc, {})
    if doc is None:
        return s
    check_doc_matches_clip(doc, identity, width, height, points_path,
                           f"{PROG} pick {_quote(video)}{trim_flags(work_dir)}{config_flag()}")
    if doc["clip"]["meta_sha256"] != identity["meta_sha256"]:
        s.notes.append("meta.json berubah sejak titik disimpan - frame dicek satu per satu lewat sidik piksel.")
    s.status = frame_statuses(doc, work_dir, set(indices), {frame: pixel_sha})
    if s.status.get(frame) == STATUS_STALE:
        s.notes.append(f"titik lama frame {frame} dibuat pada versi frame lain (sidik piksel berbeda) - TIDAK dimuat; "
                       f"klik ulang lalu simpan untuk menggantinya.")
    stale = [f for f, st in s.status.items() if st == STATUS_STALE and f != frame]
    if stale:
        s.notes.append(f"titik frame {_ids(stale)} basi (isi frame berubah, mis. ingest diulang dari video lain) - "
                       f"pick ulang: {s.pick_cmd(stale[0])}" + (" (lalu frame lain)" if len(stale) > 1 else ""))
    missing = [f for f, st in s.status.items() if st == STATUS_MISSING]
    if missing:
        s.notes.append(f"frame {_ids(missing)} di points.json tidak ada di frames/ (hilang).")
    return s


def new_state(session: Session, view: View) -> PickState:
    entry = session.entry
    loaded = entry["targets"] if entry is not None and session.status.get(session.frame) == STATUS_OK else None
    return PickState(view, loaded_targets=loaded, known_other_ids=session.other_ids(), entry_exists=entry is not None)


def commit(session: Session, state: PickState, action: str, display: DisplayInfo) -> tuple[dict, bool]:
    """Simpan / hapus entri frame ini: points.json DIBACA ULANG dari disk (perubahan sesi lain tetap ada), sidik frame dihitung
    ulang (berbeda dengan saat dibuka → tidak disimpan). Return (dokumen baru, ada cadangan)."""
    path = session.points_path
    try:
        disk = load_points(path)
    except PointsError as e:
        raise PickError(f"{e} - tidak disimpan") from None
    try:
        identity = clip_identity(session.work_dir)
    except (StageError, OSError, ValueError, KeyError) as e:
        raise PickError(f"meta.json tidak terbaca saat menyimpan ({e}) - tidak disimpan") from None
    if not _same_source(identity["source_path"], session.identity["source_path"]):
        raise PickError(f"meta.json berganti klip selama pick ({identity['source_path']}) - tidak disimpan")
    if disk is not None:
        check_doc_matches_clip(disk, identity, session.width, session.height, path,
                               f"{PROG} pick {_quote(session.video)}{trim_flags(session.work_dir)}{config_flag()}")
    try:
        now = frame_pixel_sha256(session.frame_path)
    except StageError:
        now = None
    if now != session.pixel_sha:
        raise PickError(f"frame {session.frame} berubah selama pick - tidak disimpan; jalankan pick lagi: "
                        f"{session.pick_cmd(session.frame)}")
    entry = None if action == ACT_DELETE else build_entry(session.frame, session.pixel_sha, state, display)
    doc = merge_entry(disk, session.frame, entry, identity, session.width, session.height)
    try:
        backed_up = write_points(path, doc)
    except StageError as e:
        raise PickError(f"gagal menulis {path}: {e}") from None
    return doc, backed_up


# ── Gambar kanvas ──────────────────────────────────
@dataclass
class PanelInfo:
    clip: str
    frame: int
    first: int
    last: int
    other_ids: dict[int, list[int]]
    notes: list[str]


def _text_w(text: str, scale: float) -> int:
    return cv2.getTextSize(text, FONT, scale, 2 if scale > 0.6 else 1)[0][0]


def wrap_text(text: str, scale: float, width: int) -> list[str]:
    lines, cur = [], ""
    for word in text.split():
        cand = f"{cur} {word}" if cur else word
        if cur and _text_w(cand, scale) > width:
            lines.append(cur)
            cur = word
        else:
            cur = cand
    return [*lines, cur] if cur else lines


def _counts(t: dict) -> tuple[int, int, bool]:
    pos = sum(p["label"] == 1 for p in t["points"])
    return pos, len(t["points"]) - pos, t["box"] is not None


def panel_lines(state: PickState, info: PanelInfo, *, with_tips: bool = True) -> list[tuple[str, tuple, float] | None]:
    """Baris panel (teks, warna BGR, skala font); None = jarak antar blok. ASCII saja (font Hershey)."""
    room = PANEL_W - 2 * PANEL_MARGIN
    v = state.view
    clip = info.clip if len(info.clip) <= 16 else info.clip[:15] + "~"
    out: list[tuple[str, tuple, float] | None] = [
        ("PILIH TARGET (pick)", TEXT_WHITE, 0.65),
        (f"{clip}  frame {info.frame} ({info.first}-{info.last})", TEXT_WHITE, 0.5),
        (f"skala {v.scale:.2f}, gambar {v.dw}x{v.dh}", TEXT_GREY, 0.45), None,
        (f"Target aktif: {state.active} ({target_color_name(state.active)})", target_color(state.active), 0.55), None,
        ("Klik KIRI  = bagian target", TEXT_WHITE, 0.5),
        ("Klik KANAN = BUKAN target", TEXT_WHITE, 0.5),
        ("Tarik kiri = kotak (ganti lama)", TEXT_WHITE, 0.5),
        ("N = target berikutnya", TEXT_WHITE, 0.5),
        ("U = batal aksi terakhir", TEXT_WHITE, 0.5),
        ("R = kosongkan frame ini", TEXT_WHITE, 0.5),
        ("Enter / S = simpan + tutup", TEXT_WHITE, 0.5),
        ("Esc = keluar tanpa simpan", TEXT_WHITE, 0.5), None]
    if with_tips:
        out += [("Saran: 3-5 klik kiri tersebar per", TEXT_GREY, 0.45),
                ("target + klik kanan di orang /", TEXT_GREY, 0.45),
                ("benda yang menempel ke target.", TEXT_GREY, 0.45), None]
    out.append(("Frame ini:", TEXT_WHITE, 0.5))
    content = state.content()
    for tid in sorted(set(content) | {state.active}):
        pos, neg, box = _counts(state.targets.get(tid, {"points": [], "box": None}))
        mark = "  (!)" if tid in content and neg == 0 else ""
        out.append((f" {tid} {target_color_name(tid)}: {pos} kiri, {neg} kanan{', kotak' if box else ''}{mark}",
                    target_color(tid), 0.5))
    if state.no_negative():
        out.append(("(!) = belum ada klik kanan", TEXT_GREY, 0.45))
    if info.other_ids:
        text = "Di frame lain: " + "  ".join(f"{tid} (f{', f'.join(map(str, fs))})" for tid, fs in info.other_ids.items())
        out += [(ln, TEXT_WHITE, 0.45) for ln in wrap_text(text, 0.45, room)]
    for note in info.notes:
        out += [None, *[(ln, TEXT_YELLOW, 0.45) for ln in wrap_text("PERINGATAN: " + note, 0.45, room)]]
    if state.msg:
        out += [None, *[(ln, TEXT_YELLOW, 0.5) for ln in wrap_text(state.msg, 0.5, room)]]
    out += [None, ("Tombol tidak bereaksi? klik judul", TEXT_GREY, 0.45), ("jendela sekali.", TEXT_GREY, 0.45)]
    return out


def _layout(lines) -> tuple[list[tuple[str, tuple, float, int]], int]:
    y, placed = 30, []
    for ln in lines:
        if ln is None:
            y += LINE_GAP_PX
            continue
        text, col, sc = ln
        placed.append((text, col, sc, y))
        y += int(42 * sc)
    return placed, y


def panel_layout(state: PickState, info: PanelInfo) -> tuple[list[tuple[str, tuple, float, int]], int]:
    """Baris + posisi y; blok saran dibuang bila panel tidak muat. Return (baris, y bawah)."""
    placed, bottom = _layout(panel_lines(state, info))
    if bottom > state.view.canvas_h:
        placed, bottom = _layout(panel_lines(state, info, with_tips=False))
    return placed, bottom


def widest_text_px(state: PickState, info: PanelInfo) -> int:
    return max(_text_w(t, sc) for t, _, sc, _ in panel_layout(state, info)[0])


def display_image(frame_rgb: np.ndarray, view: View) -> np.ndarray:
    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    if (view.dw, view.dh) == (view.width, view.height):
        return bgr
    return cv2.resize(bgr, (view.dw, view.dh), interpolation=cv2.INTER_AREA)


def render_canvas(view: View, disp_bgr: np.ndarray, state: PickState, info: PanelInfo) -> np.ndarray:
    c = np.full((view.canvas_h, view.canvas_w, 3), BG_GRAY, np.uint8)
    c[:view.dh, :view.dw] = disp_bgr
    for tid, t in sorted(state.targets.items()):
        col = target_color(tid)
        if t["box"] is not None:
            p0, p1 = view.to_disp(t["box"][0], t["box"][1]), view.to_disp(t["box"][2], t["box"][3])
            cv2.rectangle(c, p0, p1, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.rectangle(c, p0, p1, col, 2, cv2.LINE_AA)
            cv2.putText(c, str(tid), (p0[0] + 4, p0[1] + 18), FONT, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(c, str(tid), (p0[0] + 4, p0[1] + 18), FONT, 0.55, col, 1, cv2.LINE_AA)
        for p in t["points"]:
            dx, dy = view.to_disp(p["x"], p["y"])
            if p["label"] == 1:
                cv2.circle(c, (dx, dy), 8, (0, 0, 0), -1, cv2.LINE_AA)
                cv2.circle(c, (dx, dy), 6, col, -1, cv2.LINE_AA)
            else:
                cv2.circle(c, (dx, dy), 9, col, 2, cv2.LINE_AA)
                cv2.circle(c, (dx, dy), 6, NEG_COLOR, -1, cv2.LINE_AA)
            cv2.putText(c, str(tid), (dx + 9, dy - 9), FONT, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(c, str(tid), (dx + 9, dy - 9), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    if state.mode == MODE_DRAG and state.press is not None and state.cur is not None:
        cv2.rectangle(c, state.press, state.cur, target_color(state.active), 1, cv2.LINE_AA)
    c[:, view.dw:] = BG_GRAY                                  # penanda di tepi kanan gambar tidak masuk panel
    c[view.dh:, :view.dw] = BG_GRAY
    for text, col, sc, y in panel_layout(state, info)[0]:
        cv2.putText(c, text, (view.dw + PANEL_MARGIN, y), FONT, sc, col, 2 if sc > 0.6 else 1, cv2.LINE_AA)
    return c


# ── Windows: DPI + area kerja ──────────────────────
def ensure_dpi_aware() -> tuple[bool, str | None]:
    """Proses DPI-aware (per monitor) SEBELUM jendela apa pun; hasil DICEK lewat GetProcessDpiAwareness (panggilan kedua
    SetProcessDpiAwareness mengembalikan E_ACCESSDENIED tanpa exception). Gagal → lanjut di koordinat logis (klik tetap tepat
    dengan WINDOW_AUTOSIZE, gambar lebih buram)."""
    dll = _windll
    if dll is None:
        return False, "bukan Windows - tanpa pengaturan DPI"
    hr = 0
    try:
        shcore = dll.shcore
        aw = ctypes.c_int(-1)
        shcore.GetProcessDpiAwareness(None, ctypes.byref(aw))
        if aw.value >= 1:
            return True, None
        hr = int(shcore.SetProcessDpiAwareness(PROCESS_PER_MONITOR_DPI_AWARE)) & 0xFFFFFFFF
        aw = ctypes.c_int(-1)
        shcore.GetProcessDpiAwareness(None, ctypes.byref(aw))
    except (OSError, AttributeError) as e:
        return False, f"pengaturan DPI tidak tersedia ({e}) - koordinat logis (klik tetap tepat, gambar lebih buram)"
    if aw.value >= 1:
        return True, None
    return False, (f"SetProcessDpiAwareness gagal (HRESULT 0x{hr:08X}, awareness {aw.value}) - koordinat logis "
                   f"(klik tetap tepat, gambar lebih buram)")


def probe_display() -> DisplayInfo:
    aware, note = ensure_dpi_aware()
    notes = [note] if note else []
    screen = work = dpi = None
    dll = _windll
    if dll is not None:
        try:
            import ctypes.wintypes as wt

            u = dll.user32
            screen = (int(u.GetSystemMetrics(0)), int(u.GetSystemMetrics(1)))
            r = wt.RECT()
            if u.SystemParametersInfoW(SPI_GETWORKAREA, 0, ctypes.byref(r), 0):
                work = (int(r.left), int(r.top), int(r.right), int(r.bottom))
            dpi = int(u.GetDpiForSystem())
        except (OSError, AttributeError, ValueError):
            pass
    if work is None or work[2] <= work[0] or work[3] <= work[1]:
        work = (0, 0, *screen) if screen and min(screen) > 0 else FALLBACK_WORK_AREA
        notes.append(f"area kerja tidak terbaca - dipakai {work[2] - work[0]} x {work[3] - work[1]}")
    if screen is None:
        screen = (work[2] - work[0], work[3] - work[1])
    return DisplayInfo(screen, work, aware, dpi, tuple(notes))


# ── Jendela OpenCV ─────────────────────────────────
class CvGui:
    """Lapis tipis OpenCV HighGUI (WIN32UI). Diganti GUI palsu di test."""

    def __init__(self) -> None:
        self.title = ""

    def open(self, title: str, x: int, y: int, on_mouse) -> None:
        self.title = title
        cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow(title, x, y)
        cv2.setMouseCallback(title, on_mouse)

    def show(self, img: np.ndarray) -> None:
        cv2.imshow(self.title, img)

    def pump(self) -> None:
        cv2.waitKey(1)

    def wait_key(self, ms: int) -> int:
        return cv2.waitKeyEx(ms)

    def visible(self) -> bool:
        try:
            return cv2.getWindowProperty(self.title, cv2.WND_PROP_VISIBLE) >= 1
        except cv2.error:
            return False

    def image_rect(self) -> tuple[int, int, int, int]:
        x, y, w, h = cv2.getWindowImageRect(self.title)
        return int(x), int(y), int(w), int(h)

    def focus(self) -> None:
        if _windll is None:
            return
        try:
            hwnd = _windll.user32.FindWindowW(None, self.title)
            if hwnd:
                _windll.user32.SetForegroundWindow(hwnd)
        except (OSError, AttributeError):
            pass

    def close_all(self) -> None:
        cv2.destroyAllWindows()


def open_window(gui, title: str, display: DisplayInfo, view: View, on_mouse, canvas: np.ndarray) -> str | None:
    """Buka jendela di pojok kiri-atas area kerja + PENJAGA: ukuran area gambar jendela harus = kanvas (selain itu klik bisa
    meleset → berhenti sebelum klik apa pun). Return pesan galat atau None."""
    try:
        gui.open(title, display.work_area[0], display.work_area[1], on_mouse)
        gui.show(canvas)
        gui.pump()
        x, y, w, h = gui.image_rect()
    except cv2.error as e:
        return f"OpenCV tidak bisa membuka jendela ({str(e).strip()}) - pakai --dry-run untuk memeriksa tampilan"
    if (w, h) != (view.canvas_w, view.canvas_h):
        return (f"area gambar jendela {w} x {h} != kanvas {view.canvas_w} x {view.canvas_h} (diskalakan sistem?) - klik bisa "
                f"meleset; pick dihentikan, tidak ada yang disimpan")
    left, top, right, bottom = display.work_area
    if x < left or y < top or x + w > right or y + h > bottom:
        print(f"PERINGATAN: area gambar jendela ({x}, {y}, {w} x {h}) keluar dari area kerja {left},{top}-{right},{bottom} "
              f"- geser jendela bila sebagian tidak terlihat", flush=True)
    gui.focus()
    return None


# ── Ringkasan teks ─────────────────────────────────
def describe_targets(targets: list[dict]) -> str:
    parts = []
    for t in targets:
        pos = sum(p["label"] == 1 for p in t["points"])
        parts.append(f"target {t['id']}: {pos} kiri, {len(t['points']) - pos} kanan" + (", kotak" if t["box"] else ""))
    return "; ".join(parts)


def describe_status(session: Session) -> str:
    if session.doc is None:
        return "belum ada"
    if not session.status:
        return "0 frame"
    return f"{len(session.status)} frame: " + ", ".join(f"{f} ({st})" for f, st in sorted(session.status.items()))


def no_negative_warning(ids: list[int]) -> str:
    return (f"PERINGATAN: target {_ids(ids)} tanpa klik kanan (bukan target) - klik kanan di orang / benda yang menempel "
            f"membantu SAM memisahkan target (T-602 klip5: 0 klik kanan, masker rusak). Tetap disimpan.")


def print_open(session: Session, view: View, display: DisplayInfo) -> None:
    ww, wh = display.work_wh
    print(f"pick {session.clip} - frame {session.frame} (klip {len(session.indices)} frame: "
          f"{session.indices[0]}-{session.indices[-1]})", flush=True)
    print(f"  tampilan: skala {view.scale:.4f} -> gambar {view.dw} x {view.dh}, kanvas {view.canvas_w} x {view.canvas_h} "
          f"(batas {view.max_w} x {view.max_h}; layar {display.screen_wh[0]} x {display.screen_wh[1]}, area kerja {ww} x {wh}, "
          f"DPI-aware: {'ya' if display.dpi_aware else 'tidak'}"
          + (f", DPI {display.dpi}" if display.dpi else "") + ")", flush=True)
    print(f"  points.json: {describe_status(session)}", flush=True)
    for n in (*display.notes, *session.notes):
        print(f"PERINGATAN: {n}", flush=True)
    if not view.fits:
        print(f"PERINGATAN: kanvas {view.canvas_w} x {view.canvas_h} lebih besar dari batas {view.max_w} x {view.max_h} - "
              f"sebagian jendela bisa di luar layar", flush=True)


# ── Jalankan ───────────────────────────────────────
def dry_run(session: Session, state: PickState, view: View, display: DisplayInfo, info: PanelInfo, disp: np.ndarray) -> int:
    """Tanpa jendela: kanvas yang akan tampil → target/pick_dryrun/frame_%05d.png + tata letak. points.json tidak disentuh."""
    out = session.work_dir / TARGET_DIRNAME / DRYRUN_DIRNAME / (FRAME_PATTERN % session.frame)
    canvas = render_canvas(view, disp, state, info)
    ok, buf = cv2.imencode(".png", canvas)
    if not ok:
        print(f"ERROR: gagal mengodekan PNG dry-run {out}", file=sys.stderr)
        return EXIT_PRECONDITION
    out.parent.mkdir(parents=True, exist_ok=True)
    write_bytes_atomic(out, buf.tobytes())
    rt = roundtrip_report(view)
    _placed, bottom = panel_layout(state, info)
    widest, room = widest_text_px(state, info), PANEL_W - 2 * PANEL_MARGIN
    loaded = describe_targets(targets_json(state)) or "-"
    print(f"  dry-run: {out} ({canvas.shape[1]} x {canvas.shape[0]})", flush=True)
    print(f"  kanvas muat di batas: {'ya' if view.fits else 'TIDAK'}; teks panel terlebar {widest} px (ruang {room}), "
          f"bawah panel y {bottom} (tinggi {view.canvas_h}); bolak-balik koordinat: identitas tiap piksel tampilan "
          f"{'ya' if rt['disp_identity'] else 'TIDAK'}, galat frame maks {rt['max_frame_err_px']} px "
          f"(batas {rt['bound_px']})", flush=True)
    print(f"  titik lama frame ini: {loaded}; jendela tidak dibuka, points.json tidak disentuh", flush=True)
    return EXIT_OK


def gui_loop(session: Session, state: PickState, view: View, display: DisplayInfo, info: PanelInfo, disp: np.ndarray,
             gui) -> int:
    title = f"pick {session.clip} f{session.frame}"

    def on_mouse(event, x, y, flags, _param) -> None:
        state.mouse(event, x, y, flags)

    try:
        err = open_window(gui, title, display, view, on_mouse, render_canvas(view, disp, state, info))
        if err:
            print(f"ERROR: {err}", file=sys.stderr)
            return EXIT_PRECONDITION
        print(f"Jendela '{title}' terbuka - petunjuk di panel kanan.", flush=True)
        while True:
            gui.show(render_canvas(view, disp, state, info))
            k = gui.wait_key(WAIT_MS)
            if not gui.visible():                       # tombol X = Esc
                act = state.request_exit()
                if act is None:                         # ada perubahan: jendela dibuka lagi dengan konfirmasi
                    err = open_window(gui, title, display, view, on_mouse, render_canvas(view, disp, state, info))
                    if err:
                        print(f"ERROR: {err}", file=sys.stderr)
                        return EXIT_PRECONDITION
                    continue
            elif k == -1:
                continue
            else:
                act = state.key(k)
            if act is None:
                continue
            if act == ACT_EXIT:
                print("Dibatalkan - points.json tidak berubah.", flush=True)
                return EXIT_OK
            if act == ACT_CLOSE:
                print("Tidak ada perubahan - points.json tidak ditulis.", flush=True)
                return EXIT_OK
            gui.close_all()
            try:
                doc, backed_up = commit(session, state, act, display)
            except PickError as e:
                print(f"ERROR: {e}", file=sys.stderr)
                return EXIT_PRECONDITION
            print_saved(session, state, act, doc, backed_up)
            return EXIT_OK
    except KeyboardInterrupt:
        print("Dihentikan (Ctrl+C) - points.json tidak berubah.", file=sys.stderr)
        return EXIT_INTERRUPTED
    finally:
        gui.close_all()


def print_saved(session: Session, state: PickState, act: str, doc: dict, backed_up: bool) -> None:
    others = [e["frame_index"] for e in doc["frames"] if e["frame_index"] != session.frame]
    status = frame_statuses({"frames": [e for e in doc["frames"] if e["frame_index"] != session.frame]},
                            session.work_dir, set(session.indices), {})
    if act == ACT_DELETE:
        print(f"Frame {session.frame} dihapus dari {session.points_path} (frame tersisa: {_ids(others) or '-'}).", flush=True)
    else:
        print(f"Tersimpan: {session.points_path} - frame {session.frame}: {describe_targets(targets_json(state))}", flush=True)
    if others:
        print("  frame lain di berkas: " + ", ".join(f"{f} ({status[f]})" for f in others), flush=True)
    if backed_up:
        print(f"  isi lama disalin ke {session.points_path.with_name(PREV_FILENAME).name}", flush=True)
    if act != ACT_DELETE and state.no_negative():
        print(no_negative_warning(state.no_negative()), flush=True)
    stale = [f for f in others if status[f] == STATUS_STALE]
    if stale:
        print(f"PERINGATAN: titik frame {_ids(stale)} basi - pick ulang: {session.pick_cmd(stale[0])}", flush=True)


def run_pick(work_dir: Path, frame: int, *, video: str, dry_run_only: bool = False, gui=None,
             display: DisplayInfo | None = None) -> int:
    """Pintu masuk dari cli (meta.json sudah ada). Exit 0 (tersimpan / tanpa perubahan / dibatalkan / dry-run) | 1 | 130."""
    try:
        session = open_session(work_dir, frame, video)
        display = display or probe_display()
        view = fit_view(session.width, session.height, *display.work_wh)
    except PickError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return EXIT_PRECONDITION
    state = new_state(session, view)
    info = PanelInfo(session.clip, session.frame, session.indices[0], session.indices[-1], session.other_ids(),
                     list(session.notes))
    print_open(session, view, display)
    disp = display_image(session.frame_rgb, view)
    if dry_run_only:
        return dry_run(session, state, view, display, info, disp)
    return gui_loop(session, state, view, display, info, disp, gui if gui is not None else CvGui())
