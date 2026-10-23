"""Test pick.py + subperintah `pick` (T-604): tanpa jendela (GUI dipalsukan), tanpa GPU / model / torch.

Nomor kelompok (1)–(26) = daftar test rancangan Tahap 1 (docs/04 "Rancangan T-604 Tahap 1" R8); (27)–(29) = tambahan
keputusan Rio 2026-10-16 (sidik piksel walau byte PNG berbeda; Esc / X dengan dan tanpa perubahan; target kosong tidak disimpan).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from rotoscope import cli
from rotoscope import pick
from rotoscope.ingest import FRAME_PATTERN

W, H, N = 64, 48, 5                                   # klip sintetis kecil (s = 1 di layar 1920 × 1020)
DISPLAY = pick.DisplayInfo((1920, 1080), (0, 0, 1920, 1020), True, 120)
LDOWN, LUP, RDOWN, MOVE = cv2.EVENT_LBUTTONDOWN, cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONDOWN, cv2.EVENT_MOUSEMOVE
FLAG_L = cv2.EVENT_FLAG_LBUTTON
ENTER, ESC = 13, 27
needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
                                  reason="ffmpeg/ffprobe tidak ada di PATH")


@pytest.fixture(autouse=True)
def _no_real_display(monkeypatch):
    """Tanpa SetProcessDpiAwareness sungguhan di proses test; layar = 1920 × 1080 (area kerja 1920 × 1020)."""
    monkeypatch.setattr(pick, "probe_display", lambda: DISPLAY)


# ── Klip sintetis ──────────────────────────────────
def frame_rgb(i: int, w: int = W, h: int = H) -> np.ndarray:
    y, x = np.indices((h, w))
    return np.stack([(x * 3 + i * 17) % 256, (y * 5 + i * 31) % 256, np.full_like(x, (i * 40) % 256)], -1).astype(np.uint8)


def write_png(path: Path, rgb: np.ndarray, compression: int = 3) -> None:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, compression])
    assert ok
    path.write_bytes(buf.tobytes())


def build_clip(work: Path, video: Path, n: int = N, w: int = W, h: int = H) -> None:
    (work / "frames").mkdir(parents=True, exist_ok=True)
    for i in range(n):
        write_png(work / "frames" / (FRAME_PATTERN % i), frame_rgb(i, w, h))
    meta = {"source_path": str(video.resolve()), "frame_count": n, "working_width": w, "working_height": h,
            "frame_index_start": 0}
    (work / "meta.json").write_text(json.dumps(meta), encoding="utf-8")


def make_clip(tmp_path: Path, name: str = "klip", n: int = N) -> tuple[Path, Path]:
    video = tmp_path / f"{name}.mp4"
    video.write_bytes(b"bukan video sungguhan")
    work = tmp_path / "work" / "clips" / name
    build_clip(work, video, n)
    return work, video


def points_path(work: Path) -> Path:
    return work / "target" / "points.json"


def session_state(work: Path, frame: int) -> tuple[pick.Session, pick.PickState]:
    s = pick.open_session(work, frame, "v.mp4")
    return s, pick.new_state(s, pick.fit_view(s.width, s.height, *DISPLAY.work_wh))


def save_points(work: Path, frame: int, clicks) -> dict:
    """clicks: [(dx, dy, label)] | ("box", dx0, dy0, dx1, dy1) | ("n",) — disimpan lewat commit (tanpa GUI)."""
    s, st = session_state(work, frame)
    for c in clicks:
        if c[0] == "n":
            st.next_target()
        elif c[0] == "box":
            st.set_box(*c[1:])
        else:
            st.add_point(*c)
    return pick.commit(s, st, pick.ACT_SAVE, DISPLAY)[0]


class FakeGui:
    """GUI palsu: tiap wait_key menjalankan satu langkah skrip — ("mouse", event, x, y, flags) | ("key", kode) |
    ("close",) (tombol X) | ("call", fn) | ("idle",)."""

    def __init__(self, script, rect=None, fail_open: bool = False):
        self.script, self.rect, self.fail_open = list(script), rect, fail_open
        self.opened = self.shown = 0
        self.closed = False
        self._visible = True
        self.cb = None
        self.last_shape = None
        self.positions: list[tuple[int, int]] = []

    def open(self, title, x, y, on_mouse):
        if self.fail_open:
            raise cv2.error("tanpa desktop")
        self.opened += 1
        self.positions.append((x, y))
        self.cb, self._visible = on_mouse, True

    def show(self, img):
        self.shown += 1
        self.last_shape = img.shape

    def pump(self):
        pass

    def wait_key(self, ms):
        assert self.script, "skrip GUI habis (jendela tidak menutup)"
        step = self.script.pop(0)
        if step[0] == "mouse":
            self.cb(step[1], step[2], step[3], step[4], None)
        elif step[0] == "key":
            return step[1]
        elif step[0] == "close":
            self._visible = False
        elif step[0] == "call":
            step[1]()
        return -1

    def visible(self):
        return self._visible

    def image_rect(self):
        return self.rect or (8, 31, self.last_shape[1], self.last_shape[0])

    def focus(self):
        pass

    def close_all(self):
        self.closed = True


def click(x, y):
    return [("mouse", LDOWN, x, y, FLAG_L), ("mouse", LUP, x, y, 0)]


def rclick(x, y):
    return [("mouse", RDOWN, x, y, 0)]


def run_gui(work: Path, frame: int, script, **kw) -> tuple[int, FakeGui]:
    gui = FakeGui(script, **kw)
    return pick.run_pick(work, frame, video="v.mp4", gui=gui, display=DISPLAY), gui


def small_view() -> pick.View:
    return pick.fit_view(W, H, *DISPLAY.work_wh)


# ── (1) skala ──────────────────────────────────────
def test_fit_scale_screens_and_sizes():
    v = pick.fit_view(720, 1280, 1920, 1020)                         # layar Rio, DPI-aware
    assert (v.scale, v.dw, v.dh, v.canvas_w, v.canvas_h, v.max_w, v.max_h) == (0.753125, 542, 964, 942, 964, 1904, 964)
    assert v.fits
    u = pick.fit_view(720, 1280, 1536, 816)                          # cadangan DPI-unaware (koordinat logis)
    assert (round(u.scale, 5), u.dw, u.dh) == (0.59375, 428, 760) and u.fits
    land = pick.fit_view(1280, 720, 1920, 1020)                      # lanskap: dibatasi lebar (tanpa upscale)
    assert land.scale == 1.0 and (land.dw, land.dh, land.canvas_w) == (1280, 720, 1680) and land.fits
    small = pick.fit_view(W, H, 1920, 1020)                          # kecil: tanpa upscale, panel min 600
    assert (small.scale, small.dw, small.dh, small.canvas_h) == (1.0, W, H, pick.PANEL_MIN_H)
    wide = pick.fit_view(3000, 400, 1920, 1020)
    assert wide.canvas_w <= wide.max_w and wide.dw == 1504
    with pytest.raises(pick.PickError, match="layar terlalu kecil"):
        pick.fit_view(720, 1280, 400, 300)


# ── (2) bolak-balik koordinat ──────────────────────
@pytest.mark.parametrize("size, area", [((720, 1280), (1920, 1020)), ((720, 1280), (1536, 816)), ((721, 1279), (1000, 700)),
                                        ((333, 999), (800, 600)), ((W, H), (1920, 1020)), ((1279, 721), (1200, 900))])
def test_roundtrip_every_display_pixel(size, area):
    v = pick.fit_view(*size, *area)
    assert all(v.to_disp(*v.to_frame(dx, 0))[0] == dx for dx in range(v.dw))
    assert all(v.to_disp(*v.to_frame(0, dy))[1] == dy for dy in range(v.dh))
    bound = 0.5 / min(v.sx, v.sy) + 0.05 + 1e-9
    for x in range(0, v.width, 7):
        assert abs(v.to_frame(*v.to_disp(x, 0))[0] - x) <= bound
    rep = pick.roundtrip_report(v)
    assert rep["disp_identity"] and rep["max_frame_err_px"] <= rep["bound_px"]


# ── (3) pusat piksel, sx / sy, clamp, panel ────────
def test_pixel_center_values_and_clamp():
    v = pick.View(100, 100, 0.5, 50, 50, 450, 600, 9999, 9999)
    assert v.to_frame(0, 0) == (0.5, 0.5) and v.to_frame(1, 1) == (2.5, 2.5) and v.to_frame(49, 49) == (98.5, 98.5)
    assert v.to_disp(0, 0) == (0, 0) and v.to_disp(2.5, 2.5) == (1, 1) and v.to_disp(99, 99) == (49, 49)
    assert v.to_frame(-5, -7) == (0.0, 0.0) and v.to_frame(60, 75) == (99.0, 99.0)       # clamp ke [0, W − 1]


def test_mapping_uses_sx_sy_not_s():
    v = pick.fit_view(720, 1280, 1920, 1020)                         # s 0,753125 ≠ sx 0,75278 (dw dibulatkan)
    assert v.scale != v.sx
    assert v.to_frame(v.dw - 1, v.dh - 1) == (round(720 - 0.5 / v.sx - 0.5, 1), round(1280 - 0.5 / v.sy - 0.5, 1)) == (718.8, 1278.8)
    assert v.to_disp(719, 1279) == (v.dw - 1, v.dh - 1)


def test_clicks_in_panel_or_below_image_are_ignored():
    st = pick.PickState(small_view())
    for x, y in ((W + 5, 10), (W, 0), (5, H + 3)):
        st.mouse(LDOWN, x, y, FLAG_L)
        st.mouse(LUP, x, y, 0)
        st.mouse(RDOWN, x, y, 0)
    assert st.content() == {} and st.history == [] and st.mode == pick.MODE_NORMAL


# ── (4) kotak dari tarik ───────────────────────────
@pytest.mark.parametrize("p0, p1", [((10, 8), (40, 30)), ((40, 30), (10, 8)), ((40, 8), (10, 30)), ((10, 30), (40, 8))])
def test_drag_box_any_direction_is_normalized(p0, p1):
    st = pick.PickState(small_view())
    st.mouse(LDOWN, *p0, FLAG_L)
    st.mouse(MOVE, *p1, FLAG_L)
    assert st.mode == pick.MODE_DRAG
    st.mouse(LUP, *p1, 0)
    assert st.content() == {1: {"points": [], "box": [10.0, 8.0, 40.0, 30.0]}}


def test_drag_release_outside_is_clamped_and_small_drag_is_click():
    st = pick.PickState(small_view())
    st.mouse(LDOWN, 50, 40, FLAG_L)
    st.mouse(MOVE, W + 80, H + 90, FLAG_L)                           # kursor di panel / di luar jendela
    st.mouse(LUP, W + 80, H + 90, 0)
    assert st.targets[1]["box"] == [50.0, 40.0, W - 1.0, H - 1.0]
    st2 = pick.PickState(small_view())
    st2.mouse(LDOWN, 20, 20, FLAG_L)
    st2.mouse(MOVE, 23, 22, FLAG_L)                                  # < 6 px = klik di posisi tekan
    st2.mouse(LUP, 23, 22, 0)
    assert st2.content() == {1: {"points": [{"x": 20.0, "y": 20.0, "label": 1}], "box": None}}
    st3 = pick.PickState(small_view())
    st3.mouse(LDOWN, 10, 10, FLAG_L)
    st3.mouse(MOVE, 40, 12, FLAG_L)                                  # tarik panjang tapi tipis (2 px) → diabaikan
    st3.mouse(LUP, 40, 12, 0)
    assert st3.content() == {} and "tipis" in st3.msg


# ── (5) klik kiri / kanan ──────────────────────────
def test_left_right_clicks_go_to_active_target():
    st = pick.PickState(small_view())
    st.mouse(LDOWN, 5, 6, FLAG_L)
    st.mouse(LUP, 5, 6, 0)
    st.mouse(RDOWN, 7, 8, 0)
    st.next_target()
    st.mouse(LDOWN, 9, 10, FLAG_L)
    st.mouse(LUP, 9, 10, 0)
    assert st.content() == {1: {"points": [{"x": 5.0, "y": 6.0, "label": 1}, {"x": 7.0, "y": 8.0, "label": 0}], "box": None},
                            2: {"points": [{"x": 9.0, "y": 10.0, "label": 1}], "box": None}}


# ── (6) kotak pengganti + U ────────────────────────
def test_second_box_replaces_and_undo_restores_old():
    st = pick.PickState(small_view())
    st.set_box(5, 5, 20, 20)
    st.set_box(30, 10, 60, 40)
    assert st.targets[1]["box"] == [30.0, 10.0, 60.0, 40.0] and "menggantikan" in st.msg
    st.undo()
    assert st.targets[1]["box"] == [5.0, 5.0, 20.0, 20.0] and "kotak lama kembali" in st.msg
    st.undo()
    assert st.targets[1]["box"] is None and st.content() == {}


# ── (7) urutan U ───────────────────────────────────
def test_undo_order_mixed_and_empty_history():
    st = pick.PickState(small_view())
    st.add_point(1, 1, 1)
    st.set_box(2, 2, 30, 30)
    st.add_point(3, 3, 0)
    st.next_target()
    st.add_point(4, 4, 1)
    st.undo()
    assert 2 not in st.content()
    st.undo()
    assert st.targets[1]["points"] == [{"x": 1.0, "y": 1.0, "label": 1}] and st.targets[1]["box"] is not None
    st.undo()
    st.undo()
    assert st.content() == {}
    st.undo()
    assert "Tidak ada" in st.msg


# ── (8) R lalu U ───────────────────────────────────
def test_reset_then_undo_restores():
    st = pick.PickState(small_view())
    st.add_point(1, 1, 1)
    st.set_box(2, 2, 30, 30)
    before = json.dumps(st.content())
    st.reset()
    assert st.content() == {} and "U = batal" in st.msg
    st.reset()
    assert "sudah kosong" in st.msg and len(st.history) == 3
    st.undo()
    assert json.dumps(st.content()) == before
    st.undo()
    assert st.targets[1]["box"] is None


# ── (9) siklus N ───────────────────────────────────
def test_n_cycles_through_known_ids_and_back_to_one():
    st = pick.PickState(small_view(), known_other_ids={1, 3})
    seen = []
    for _ in range(5):
        st.next_target()
        seen.append(st.active)
    assert seen == [2, 3, 4, 1, 2]
    fresh = pick.PickState(small_view())
    fresh.next_target()
    assert fresh.active == 1                                         # target 1 masih kosong → tetap 1
    fresh.add_point(1, 1, 1)
    fresh.next_target()
    assert fresh.active == 2
    fresh.next_target()
    assert fresh.active == 1


# ── (10) validasi simpan ───────────────────────────
def test_save_requests():
    st = pick.PickState(small_view())
    assert st.key(ENTER) is None and "Belum ada titik" in st.msg     # kosong, tanpa entri lama
    st.add_point(5, 5, 0)
    assert st.key(ord("s")) is None and "hanya punya klik kanan" in st.msg
    st.set_box(1, 1, 30, 30)
    assert st.key(ord("S")) == pick.ACT_SAVE                         # klik kanan + kotak = boleh
    only_box = pick.PickState(small_view())
    only_box.set_box(1, 1, 30, 30)
    assert only_box.key(10) == pick.ACT_SAVE                         # hanya kotak = boleh; Enter (LF) juga simpan


def test_empty_frame_with_entry_needs_confirmation():
    loaded = [{"id": 1, "points": [{"x": 3.0, "y": 4.0, "label": 1}], "box": None}]
    st = pick.PickState(small_view(), loaded_targets=loaded, entry_exists=True)
    assert st.key(ENTER) == pick.ACT_CLOSE                           # tanpa perubahan → tutup tanpa menulis
    st.key(ord("r"))
    assert st.key(ENTER) is None and st.mode == pick.MODE_CONFIRM_DELETE and "HAPUS" in st.msg
    assert st.key(ord("x")) is None and st.mode == pick.MODE_NORMAL  # tombol lain membatalkan
    assert st.key(ENTER) is None
    assert st.key(ESC) is None and st.mode == pick.MODE_NORMAL and "Hapus dibatalkan" in st.msg
    assert st.key(ENTER) is None
    assert st.key(ENTER) == pick.ACT_DELETE


# ── (11) peringatan tanpa klik kanan ───────────────
def test_no_negative_warning_does_not_block(tmp_path, capsys):
    work, _ = make_clip(tmp_path)
    rc, _ = run_gui(work, 0, [*click(5, 5), ("key", ord("n")), *click(9, 9), *rclick(20, 20), ("key", ENTER)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "PERINGATAN: target 1 tanpa klik kanan" in out and "Tetap disimpan" in out
    doc = pick.load_points(points_path(work))
    assert [t["id"] for t in doc["frames"][0]["targets"]] == [1, 2]
    rc, _ = run_gui(work, 1, [*click(5, 5), *rclick(9, 9), ("key", ENTER)])
    assert rc == 0 and "tanpa klik kanan" not in capsys.readouterr().out


def test_no_negative_lists_only_targets_without_right_click():
    st = pick.PickState(small_view())
    st.add_point(1, 1, 1)
    st.next_target()
    st.add_point(2, 2, 1)
    st.add_point(3, 3, 0)
    assert st.no_negative() == [1]
    lines = [ln[0] for ln in pick.panel_lines(st, pick.PanelInfo("k", 0, 0, 4, {}, [])) if ln]
    assert any(ln.startswith(" 1 hijau") and "(!)" in ln for ln in lines)
    assert any(ln.startswith(" 2 biru") and "(!)" not in ln for ln in lines)


# ── (12) titik lama dimuat ─────────────────────────
def test_loaded_points_and_history_order(tmp_path):
    work, _ = make_clip(tmp_path)
    save_points(work, 0, [(1, 2, 1), (3, 4, 1), (5, 6, 0), ("box", 2, 2, 40, 40), ("n",), (7, 8, 1)])
    s, st = session_state(work, 0)
    assert st.content() == {1: {"points": [{"x": 1.0, "y": 2.0, "label": 1}, {"x": 3.0, "y": 4.0, "label": 1},
                                           {"x": 5.0, "y": 6.0, "label": 0}], "box": [2.0, 2.0, 40.0, 40.0]},
                            2: {"points": [{"x": 7.0, "y": 8.0, "label": 1}], "box": None}}
    assert st.history == [("point", 1)] * 3 + [("box", 1, None), ("point", 2)]
    assert not st.dirty and st.entry_exists
    st.undo()
    assert 2 not in st.content() and st.dirty
    st.undo()
    assert st.targets[1]["box"] is None and len(st.targets[1]["points"]) == 3


# ── (13) event mouse lewat callback ────────────────
def test_mouse_sequence_through_callback_and_release_outside_window(tmp_path):
    work, _ = make_clip(tmp_path)
    script = [("mouse", LDOWN, 10, 10, FLAG_L), ("mouse", MOVE, 20, 20, FLAG_L), ("mouse", MOVE, 30, 25, FLAG_L),
              ("mouse", RDOWN, 33, 33, FLAG_L),                       # klik kanan saat menarik diabaikan
              ("mouse", MOVE, 50, 40, 0),                              # tombol dilepas di luar jendela: gerak pertama tanpa tombol
              *click(12, 13), *rclick(44, 30), ("key", ENTER)]
    rc, gui = run_gui(work, 2, script)
    assert rc == 0 and gui.opened == 1 and gui.closed
    t = pick.load_points(points_path(work))["frames"][0]["targets"][0]
    assert t["box"] == [10.0, 10.0, 30.0, 25.0]
    assert t["points"] == [{"x": 12.0, "y": 13.0, "label": 1}, {"x": 44.0, "y": 30.0, "label": 0}]


# ── (14) tulis → baca ──────────────────────────────
def test_write_read_roundtrip_and_contract_keys(tmp_path):
    work, video = make_clip(tmp_path)
    doc = save_points(work, 3, [(1, 2, 1), (5, 6, 0), ("box", 2, 2, 40, 40)])
    on_disk = pick.load_points(points_path(work))
    assert on_disk == doc
    assert set(doc) == {"format", "clip", "frame_size", "coords", "frame_hash", "frames", "tool"}
    assert doc["format"] == "rotoscope-target-points/1" and doc["frame_size"] == {"width": W, "height": H}
    assert doc["clip"]["source_path"] == str(video.resolve())
    e = doc["frames"][0]
    assert set(e) == {"frame_index", "frame_file", "frame_pixel_sha256", "updated_utc", "display", "targets"}
    assert e["frame_file"] == "frame_00003.png" and "frame_sha256" not in e
    assert e["display"] == {"scale": 1.0, "screen_wh": [1920, 1080], "work_area_wh": [1920, 1020], "dpi_aware": True}
    v = pick.fit_view(720, 1280, 1920, 1020)
    for dx in range(0, v.dw, 37):                                     # pembulatan 0,1 px
        x, _ = v.to_frame(dx, 0)
        assert round(x * 10) == pytest.approx(x * 10)


# ── (15) galat validasi ────────────────────────────
def valid_doc() -> dict:
    t = {"id": 1, "points": [{"x": 1.0, "y": 2.0, "label": 1}, {"x": 3.0, "y": 4.0, "label": 0}], "box": [1.0, 1.0, 9.0, 9.0]}
    e = {"frame_index": 0, "frame_file": "frame_00000.png", "frame_pixel_sha256": "a" * 64, "updated_utc": "x",
         "display": {}, "targets": [t]}
    return {"format": pick.POINTS_FORMAT, "clip": {"meta_sha256": "m", "source_path": "/v.mp4"},
            "frame_size": {"width": W, "height": H}, "frames": [e]}


def _set(path, value):
    def f(d):
        obj = d
        for k in path[:-1]:
            obj = obj[k]
        if value is _DEL:
            del obj[path[-1]]
        else:
            obj[path[-1]] = value
        return d
    return f


_DEL = object()
T0 = ("frames", 0, "targets", 0)


@pytest.mark.parametrize("change, msg", [
    (_set(("format",), "t602-points/1"), "konversi dulu dengan scripts/t604_convert_t602.py"),
    (_set(("format",), "rotoscope-target-points/2"), "versi pick yang lebih baru"),
    (_set(("format",), "lain/1"), "bukan rotoscope-target-points/1"),
    (_set(("clip",), {"meta_sha256": 1}), "clip harus"),
    (_set(("frame_size", "width"), 0), "frame_size"),
    (_set(("frames",), {}), "frames harus daftar"),
    (_set((*T0, "points", 0, "label"), 2), "titik #1: label 2 bukan 0 / 1"),
    (_set((*T0, "points", 0, "label"), True), "label True"),
    (_set((*T0, "points", 0, "x"), W + 0.5), "x 64.5 di luar frame (0-63)"),
    (_set((*T0, "points", 0, "y"), -0.1), "y -0.1 di luar frame"),
    (_set((*T0, "points", 0, "x"), float("nan")), "bukan bilangan hingga"),
    (_set((*T0, "points", 0, "x"), "3"), "bukan bilangan hingga"),
    (_set((*T0, "box"), [9.0, 1.0, 1.0, 9.0]), "terbalik / di luar frame"),
    (_set((*T0, "box"), [1.0, 1.0, 9.0, H + 1.0]), "terbalik / di luar frame"),
    (_set((*T0, "box"), [1.0, 1.0, 9.0]), "box harus"),
    (_set((*T0, "id"), 0), "id target harus bulat >= 1"),
    (_set((*T0, "points"), []), None),                                # hanya kotak = valid
    (lambda d: _set((*T0, "box"), None)(_set((*T0, "points"), [])(d)), "tanpa titik dan tanpa kotak"),
    (lambda d: _set((*T0, "box"), None)(_set((*T0, "points"), [{"x": 1.0, "y": 1.0, "label": 0}])(d)), "hanya klik kanan"),
    (_set(("frames", 0, "targets"), []), "frame 0: tanpa target"),
    (lambda d: d["frames"][0]["targets"].append(dict(d["frames"][0]["targets"][0])) or d, "target 1 muncul dua kali"),
    (lambda d: d["frames"].append(json.loads(json.dumps(d["frames"][0]))) or d, "frame 0 muncul dua kali"),
    (_set(("frames", 0, "frame_file"), "frame_00001.png"), "frame_file"),
    (_set(("frames", 0, "frame_pixel_sha256"), "A" * 64), "frame_pixel_sha256 harus 64 hex"),
    (_set(("frames", 0, "frame_index"), -1), "frame_index bulat >= 0"),
])
def test_validation_errors(change, msg):
    doc = change(valid_doc())
    if msg is None:
        pick.validate_doc(doc)
        return
    with pytest.raises(pick.PointsError) as e:
        pick.validate_doc(doc)
    assert msg in str(e.value)


def test_invalid_file_is_refused_and_not_overwritten(tmp_path, capsys):
    work, _ = make_clip(tmp_path)
    p = points_path(work)
    p.parent.mkdir(parents=True)
    for body, msg in (("{ rusak", "bukan JSON"), (json.dumps({**valid_doc(), "format": "t602-points/1"}), "konversi")):
        p.write_text(body, encoding="utf-8")
        before = p.read_bytes()
        rc, gui = run_gui(work, 0, [])
        err = capsys.readouterr().err
        assert rc == 1 and msg in err and "pick tidak menimpanya" in err and gui.opened == 0
        assert p.read_bytes() == before


def test_other_clip_or_other_frame_size_is_refused(tmp_path, capsys):
    work, video = make_clip(tmp_path)
    doc = save_points(work, 0, [(1, 1, 1)])
    p = points_path(work)
    p.write_text(json.dumps({**doc, "clip": {**doc["clip"], "source_path": str(tmp_path / "lain.mp4")}}), encoding="utf-8")
    assert run_gui(work, 0, [])[0] == 1 and "milik klip lain" in capsys.readouterr().err
    p.write_text(json.dumps({**doc, "frame_size": {"width": W * 2, "height": H * 2}}), encoding="utf-8")
    assert run_gui(work, 0, [])[0] == 1
    err = capsys.readouterr().err
    assert "dibuat untuk frame 128 x 96" in err and "crop / trim" in err and "points.old.json" in err   # T-607 butir 5g
    assert "python -m rotoscope pick v.mp4 --frame 0" in err                   # frame pertama berkas titik


def test_pick_cmd_carries_trim_flags(tmp_path):
    """T-607: perintah pick ulang menuju folder <nama>@S-E yang sama."""
    work, _ = make_clip(tmp_path)
    meta = json.loads((work / "meta.json").read_text(encoding="utf-8"))
    meta["trim"] = {"start_s": 0.5, "end_s": 1.0, "start_frame": 12, "end_frame": 24}
    (work / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    s = pick.open_session(work, 0, "v.mp4")
    assert s.pick_cmd(3) == "python -m rotoscope pick v.mp4 --start 0.5 --end 1 --frame 3"
    from rotoscope import stage_common
    conf = tmp_path / "c.yaml"
    stage_common.set_message_config(conf)
    assert s.pick_cmd(3) == f"python -m rotoscope pick v.mp4 --start 0.5 --end 1 --config {conf.resolve()} --frame 3"
    (work / "frames" / "frame_00002.png").unlink()                       # perintah ingest ulang juga siap tempel
    with pytest.raises(pick.PickError, match=r"jalankan python -m rotoscope ingest v\.mp4 --start 0\.5 --end 1 --config "):
        pick.open_session(work, 2, "v.mp4")


# ── (16) status ok / basi / hilang ─────────────────
def test_frame_status_ok_stale_missing(tmp_path, capsys):
    work, _ = make_clip(tmp_path)
    save_points(work, 0, [(1, 1, 1)])
    save_points(work, 1, [(2, 2, 1)])
    save_points(work, 2, [(3, 3, 1)])
    doc = pick.load_points(points_path(work))
    doc["frames"].append({**doc["frames"][0], "frame_index": 9, "frame_file": "frame_00009.png"})
    points_path(work).write_text(json.dumps(doc), encoding="utf-8")
    write_png(work / "frames" / "frame_00001.png", frame_rgb(7))         # isi frame 1 berubah
    (work / "frames" / "frame_00002.png").unlink()                       # frame 2 hilang
    st = pick.frame_statuses(pick.load_points(points_path(work)), work, set(range(N)), {})
    assert st == {0: "ok", 1: "basi", 2: "hilang", 9: "hilang"}
    s, state = session_state(work, 1)                                    # frame basi: titik lama TIDAK dimuat
    assert state.content() == {} and state.entry_exists and any("TIDAK dimuat" in n for n in s.notes)
    s0, state0 = session_state(work, 0)
    assert state0.content()[1]["points"][0]["x"] == 1.0
    assert any("basi" in n and "--frame 1" in n for n in s0.notes) and any("hilang" in n for n in s0.notes)
    rc, _ = run_gui(work, 3, [*click(4, 4), ("key", ENTER)])
    out = capsys.readouterr().out
    assert rc == 0 and "PERINGATAN: titik frame 1 basi" in out and "0 (ok), 1 (basi), 2 (hilang)" in out


def test_status_uses_own_frame_file(tmp_path):
    work, _ = make_clip(tmp_path)
    for f in range(N):
        save_points(work, f, [(1, 1, 1)])
    assert pick.frame_statuses(pick.load_points(points_path(work)), work, set(range(N)), {}) == {f: "ok" for f in range(N)}


def test_meta_change_alone_is_only_a_warning(tmp_path):
    work, _ = make_clip(tmp_path)
    save_points(work, 0, [(1, 1, 1)])
    meta = json.loads((work / "meta.json").read_text(encoding="utf-8"))
    (work / "meta.json").write_text(json.dumps({**meta, "target_fps": 12}), encoding="utf-8")
    s, st = session_state(work, 0)
    assert any("meta.json berubah" in n for n in s.notes) and st.content()


# ── (17) gabung ────────────────────────────────────
def test_merge_keeps_other_frames_and_changes_from_other_session(tmp_path):
    work, _ = make_clip(tmp_path)
    save_points(work, 3, [(1, 1, 1)])
    f3 = pick.load_points(points_path(work))["frames"][0]
    other = lambda: save_points(work, 1, [(2, 2, 1)])                    # noqa: E731 — sesi lain menyimpan f1 di tengah sesi ini
    rc, _ = run_gui(work, 0, [*click(5, 5), ("call", other), *rclick(6, 6), ("key", ENTER)])
    assert rc == 0
    doc = pick.load_points(points_path(work))
    assert [e["frame_index"] for e in doc["frames"]] == [0, 1, 3]
    assert doc["frames"][2] == f3                                        # entri frame lain apa adanya


def test_delete_entry_and_last_entry(tmp_path, capsys):
    work, _ = make_clip(tmp_path)
    save_points(work, 0, [(1, 1, 1)])
    save_points(work, 2, [(2, 2, 1)])
    rc, _ = run_gui(work, 2, [("key", ord("r")), ("key", ENTER), ("key", ENTER)])
    assert rc == 0 and "Frame 2 dihapus" in capsys.readouterr().out
    assert [e["frame_index"] for e in pick.load_points(points_path(work))["frames"]] == [0]
    rc, _ = run_gui(work, 0, [("key", ord("r")), ("key", ENTER), ("key", ENTER)])
    assert rc == 0 and pick.load_points(points_path(work))["frames"] == []
    rc, _ = run_gui(work, 0, [("key", ENTER), ("key", ESC)])              # kosong tanpa entri: Enter tidak menulis apa pun
    assert rc == 0 and pick.load_points(points_path(work))["frames"] == []


def test_merge_entry_unit():
    ident = {"meta_sha256": "m", "source_path": "/v"}
    disk = pick.new_doc(ident, W, H, [{"frame_index": 5, "x": 1}, {"frame_index": 1, "x": 2}])
    out = pick.merge_entry(disk, 3, {"frame_index": 3, "x": 3}, ident, W, H)
    assert [e["frame_index"] for e in out["frames"]] == [1, 3, 5]
    out = pick.merge_entry(disk, 5, None, ident, W, H)
    assert [e["frame_index"] for e in out["frames"]] == [1]
    assert pick.merge_entry(None, 0, None, ident, W, H)["frames"] == []


# ── (18) id konsisten antar frame ──────────────────
def test_target_ids_are_global_across_frames(tmp_path):
    work, _ = make_clip(tmp_path)
    save_points(work, 0, [(1, 1, 1), ("n",), (2, 2, 1)])
    s, st = session_state(work, 3)
    assert s.other_ids() == {1: [0], 2: [0]} and st.known_ids() == {1, 2}
    seen = []
    for _ in range(3):
        st.next_target()
        seen.append(st.active)
    assert seen == [2, 3, 1]
    rc, _ = run_gui(work, 3, [("key", ord("n")), *click(4, 4), ("key", ENTER)])
    assert rc == 0
    doc = pick.load_points(points_path(work))
    assert [[t["id"] for t in e["targets"]] for e in doc["frames"]] == [[1, 2], [2]]
    assert pick.target_color(2) == (255, 160, 0) and pick.target_color_name(7) == "hijau"


# ── (19) frame berubah selama sesi ─────────────────
def test_frame_changed_during_session_is_not_saved(tmp_path, capsys):
    work, _ = make_clip(tmp_path)
    save_points(work, 0, [(1, 1, 1)])
    before = points_path(work).read_bytes()
    change = lambda: write_png(work / "frames" / "frame_00001.png", frame_rgb(9))   # noqa: E731
    rc, _ = run_gui(work, 1, [*click(5, 5), ("call", change), ("key", ENTER)])
    assert rc == 1 and "frame 1 berubah selama pick" in capsys.readouterr().err
    assert points_path(work).read_bytes() == before


# ── (20) points.prev.json ──────────────────────────
def test_prev_backup_holds_previous_content(tmp_path):
    work, _ = make_clip(tmp_path)
    save_points(work, 0, [(1, 1, 1)])
    assert not (work / "target" / "points.prev.json").exists()
    first = points_path(work).read_bytes()
    save_points(work, 1, [(2, 2, 1)])
    assert (work / "target" / "points.prev.json").read_bytes() == first


# ── (21) cli ───────────────────────────────────────
def fake_ingest(calls: list):
    def main(argv):
        calls.append(list(argv))
        build_clip(Path(argv[argv.index("--work-dir") + 1]), Path(argv[0]))
        return 0
    return main


def test_cli_ingests_only_when_meta_missing(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    calls: list = []
    monkeypatch.setattr(cli, "CPU_MAINS", {"ingest": fake_ingest(calls)})
    video = tmp_path / "klip.mp4"
    video.write_bytes(b"x")
    work = tmp_path / "work" / "clips" / "klip"
    assert cli.main(["pick", str(video), "--dry-run"]) == 0
    cfg = cli.load_cfg(None)
    assert calls == [[str(video), "--work-dir", str(cli.clip_work_dir(cfg, video)), "--crop-dir", str(cli.crop_dir_of(cfg, video))]]
    assert "ingest dijalankan dulu" in capsys.readouterr().out
    assert (work / "target" / "pick_dryrun" / "frame_00000.png").is_file()
    frames_before = {p.name: p.read_bytes() for p in (work / "frames").iterdir()}
    assert cli.main(["pick", str(video), "--frame", "4", "--dry-run"]) == 0
    assert len(calls) == 1 and "ingest dijalankan dulu" not in capsys.readouterr().out
    assert {p.name: p.read_bytes() for p in (work / "frames").iterdir()} == frames_before


def test_cli_other_clip_frame_range_and_missing_frame(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    calls: list = []
    monkeypatch.setattr(cli, "CPU_MAINS", {"ingest": fake_ingest(calls)})
    work, video = make_clip(tmp_path)
    assert cli.main(["pick", str(video), "--frame", str(N), "--dry-run"]) == 1
    assert f"frame {N} di luar klip klip ({N} frame: 0-{N - 1})" in capsys.readouterr().err
    assert cli.main(["pick", str(video), "--frame", "-1"]) == 2
    (work / "frames" / "frame_00002.png").unlink()
    assert cli.main(["pick", str(video), "--frame", "2", "--dry-run"]) == 1
    assert "frame_00002.png hilang" in capsys.readouterr().err
    other = tmp_path / "sub" / "klip.mp4"                                # nama sama, video lain → folder kerja sama
    other.parent.mkdir()
    other.write_bytes(b"y")
    assert cli.main(["pick", str(other), "--dry-run"]) == 1
    assert "klip LAIN" in capsys.readouterr().err
    assert calls == [] and not (work / "target").exists()
    assert cli.main(["pick", str(tmp_path / "tidak_ada.mp4")]) == 1


# ── (22) --dry-run ─────────────────────────────────
class NoGui:
    def __getattr__(self, name):
        raise AssertionError(f"dry-run memanggil GUI ({name})")


def test_dry_run_writes_png_only(tmp_path, capsys):
    work, _ = make_clip(tmp_path)
    save_points(work, 2, [(1, 1, 1), ("box", 2, 2, 40, 40)])
    before = points_path(work).read_bytes()
    assert pick.run_pick(work, 2, video="v.mp4", dry_run_only=True, gui=NoGui(), display=DISPLAY) == 0
    png = work / "target" / "pick_dryrun" / "frame_00002.png"
    img = cv2.imdecode(np.fromfile(png, np.uint8), cv2.IMREAD_COLOR)
    assert img.shape == (pick.PANEL_MIN_H, W + pick.PANEL_W, 3)
    assert points_path(work).read_bytes() == before and not (work / "target" / "points.prev.json").exists()
    out = capsys.readouterr().out
    assert "skala 1.0000" in out and "kanvas 464 x 600" in out and "identitas tiap piksel tampilan ya" in out
    assert "titik lama frame ini: target 1: 1 kiri, 0 kanan, kotak" in out and "points.json: 1 frame: 2 (ok)" in out


# ── (23) render ────────────────────────────────────
def test_render_canvas_size_and_panel_fits(tmp_path):
    v = pick.fit_view(720, 1280, 1920, 1020)
    st = pick.PickState(v, known_other_ids={1, 2, 3})
    for k in range(6):                                                   # 6 target, catatan panjang, pesan panjang
        st.add_point(10 + k, 10 + k, 1)
        st.next_target()
    st.set_box(100, 100, 300, 500)
    st.mode, st.press, st.cur = pick.MODE_DRAG, (5, 5), (60, 80)
    st.msg = "Ada perubahan BELUM disimpan. Esc lagi (atau tutup jendela lagi) = keluar TANPA simpan; Enter / S = simpan."
    info = pick.PanelInfo("nama_klip_yang_sangat_panjang_sekali", 120, 0, 181, {1: [0, 64], 2: [0], 3: [64]},
                          ["titik frame 0, 64 basi (isi frame berubah, mis. ingest diulang dari video lain) - pick ulang: "
                           "python -m rotoscope pick samples\\klip5.mp4 --frame 0 (lalu frame lain)"])
    disp = pick.display_image(np.zeros((1280, 720, 3), np.uint8), v)
    c = pick.render_canvas(v, disp, st, info)
    assert c.shape == (964, 942, 3) and c.dtype == np.uint8
    assert pick.widest_text_px(st, info) <= pick.PANEL_W - 2 * pick.PANEL_MARGIN
    assert pick.panel_layout(st, info)[1] <= v.canvas_h
    small = small_view()                                                 # kanvas 600 px (tinggi minimum panel)
    st2 = pick.PickState(small)
    for k in range(6):
        st2.add_point(k, k, 1)
        st2.next_target()
    st2.msg = st.msg
    placed, _ = pick.panel_layout(st2, info)
    assert not any(t.startswith("Saran") for t, *_ in placed)            # saran dibuang bila tidak muat
    assert pick.widest_text_px(st2, info) <= pick.PANEL_W - 2 * pick.PANEL_MARGIN
    st3 = pick.PickState(small)
    st3.add_point(1, 1, 1)
    st3.add_point(2, 2, 0)
    st3.msg = st.msg
    placed, bottom = pick.panel_layout(st3, pick.PanelInfo("klip5", 0, 0, 132, {}, []))
    assert bottom <= small.canvas_h and any(t.startswith("Saran") for t, *_ in placed)


# ── (24) DPI + penjaga jendela ─────────────────────
class FakeShcore:
    def __init__(self, aw: int, set_ok: bool):
        self.aw, self.set_ok, self.set_calls = aw, set_ok, 0

    def GetProcessDpiAwareness(self, _h, ref):          # noqa: N802 — nama API Windows
        ref._obj.value = self.aw
        return 0

    def SetProcessDpiAwareness(self, value):             # noqa: N802
        self.set_calls += 1
        if self.set_ok:
            self.aw = value
            return 0
        return -2147024891                               # E_ACCESSDENIED (0x80070005) — TANPA exception


class FakeUser32:
    def GetSystemMetrics(self, i):                       # noqa: N802
        return (1920, 1080)[i]

    def SystemParametersInfoW(self, _spi, _a, ref, _b):  # noqa: N802
        r = ref._obj
        r.left, r.top, r.right, r.bottom = 0, 0, 1920, 1020
        return 1

    def GetDpiForSystem(self):                           # noqa: N802
        return 120


class FakeWindll:
    def __init__(self, aw=0, set_ok=True):
        self.shcore, self.user32 = FakeShcore(aw, set_ok), FakeUser32()


def test_dpi_awareness_checked_not_assumed(monkeypatch):
    fake = FakeWindll(aw=0, set_ok=True)
    monkeypatch.setattr(pick, "_windll", fake)
    assert pick.ensure_dpi_aware() == (True, None) and fake.shcore.set_calls == 1
    already = FakeWindll(aw=2)
    monkeypatch.setattr(pick, "_windll", already)
    assert pick.ensure_dpi_aware() == (True, None) and already.shcore.set_calls == 0
    denied = FakeWindll(aw=0, set_ok=False)
    monkeypatch.setattr(pick, "_windll", denied)
    aware, note = pick.ensure_dpi_aware()
    assert not aware and "0x80070005" in note and "koordinat logis" in note
    monkeypatch.setattr(pick, "_windll", None)
    assert pick.ensure_dpi_aware()[0] is False

    class Broken:
        @property
        def shcore(self):
            raise AttributeError("shcore")
    monkeypatch.setattr(pick, "_windll", Broken())
    assert pick.ensure_dpi_aware()[0] is False


def test_probe_display_reads_work_area(monkeypatch):
    monkeypatch.undo()                                                   # probe_display asli (tanpa fixture autouse)
    monkeypatch.setattr(pick, "_windll", FakeWindll(aw=0, set_ok=True))
    d = pick.probe_display()
    assert (d.screen_wh, d.work_area, d.work_wh, d.dpi_aware, d.dpi, d.notes) == (
        (1920, 1080), (0, 0, 1920, 1020), (1920, 1020), True, 120, ())
    monkeypatch.setattr(pick, "_windll", None)
    d = pick.probe_display()
    assert d.work_area == pick.FALLBACK_WORK_AREA and not d.dpi_aware and len(d.notes) == 2


def test_window_guard_size_mismatch_stops_before_any_click(tmp_path, capsys):
    work, _ = make_clip(tmp_path)
    rc, gui = run_gui(work, 0, [*click(5, 5), ("key", ENTER)], rect=(8, 31, 580, 480))
    assert rc == 1 and "diskalakan sistem" in capsys.readouterr().err and gui.closed
    assert not points_path(work).exists() and len(gui.script) == 3          # tidak ada event yang diproses


def test_window_outside_work_area_warns_and_no_gui_hint(tmp_path, capsys):
    work, _ = make_clip(tmp_path)
    rc, gui = run_gui(work, 0, [("key", ESC)], rect=(1700, 31, 464, 600))
    assert rc == 0 and "keluar dari area kerja" in capsys.readouterr().out and gui.positions == [(0, 0)]
    rc, _ = run_gui(work, 0, [], fail_open=True)
    assert rc == 1 and "--dry-run" in capsys.readouterr().err


# ── (25) ingest ulang: sidik piksel stabil ─────────
@needs_ffmpeg
def test_reingest_keeps_pixel_fingerprint_and_points_ok(tmp_path):
    import test_ingest as ti
    from rotoscope.ingest import ingest

    video = ti.make_video(tmp_path / "v.mp4", 96, 64, fps=24, duration=0.5)
    w1, w2 = tmp_path / "work" / "clips" / "v", tmp_path / "w2"
    ingest(video, w1)
    save_points(w1, 3, [(5, 5, 1), (20, 20, 0)])
    ingest(video, w1)                                                     # seperti `run` (selalu ingest ulang)
    s, st = session_state(w1, 3)
    assert s.status == {3: "ok"} and st.content()[1]["points"][0] == {"x": 5.0, "y": 5.0, "label": 1}
    ingest(video, w2)
    names = sorted(p.name for p in (w1 / "frames").glob("frame_*.png"))
    assert names == sorted(p.name for p in (w2 / "frames").glob("frame_*.png"))
    for n in names:
        assert pick.frame_pixel_sha256(w1 / "frames" / n) == pick.frame_pixel_sha256(w2 / "frames" / n)


# ── (26) torch + run ───────────────────────────────
def test_pick_does_not_import_torch():
    code = "import sys, rotoscope.pick, rotoscope.cli; sys.exit(1 if 'torch' in sys.modules else 0)"
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


def test_run_does_not_create_target_and_ignores_points(tmp_path, monkeypatch):
    from test_cli import CPU_STAGES, Rec

    monkeypatch.chdir(tmp_path)
    r = Rec()
    monkeypatch.setattr(cli, "_popen", r.popen)
    monkeypatch.setattr(cli, "CPU_MAINS", {s: r.cpu(s) for s in CPU_STAGES})
    work, video = make_clip(tmp_path)
    assert cli.main(["run", str(video)]) == 0
    assert not (work / "target").exists()
    first = list(r.calls)
    save_points(work, 0, [(1, 1, 1)])
    before = points_path(work).read_bytes()
    r.calls.clear()
    assert cli.main(["run", str(video)]) == 0
    # T-605: dengan points.json, `run` menjalankan [1b] isolate (subprocess, setelah ingest); argv stage lain tetap sama
    assert [c for c in r.calls if c[1] != "isolate"] == first
    assert [c[1] for c in r.calls][:2] == ["ingest", "isolate"] and r.args("isolate") == r.args("segment")[:2]
    assert points_path(work).read_bytes() == before
    assert sorted(p.name for p in (work / "target").iterdir()) == ["points.json"]


def _drop_utc(obj):
    if isinstance(obj, dict):
        return {k: _drop_utc(v) for k, v in obj.items() if not k.endswith("_utc")}
    return [_drop_utc(v) for v in obj] if isinstance(obj, list) else obj


def _content_hashes(wd: Path, out: Path) -> dict[str, str]:
    """Keluaran isi (tanpa manifest / log berstempel waktu; clip_stats.json tanpa kunci *_utc)."""
    skip = ("manifest.json", "frames.jsonl", ".export.json", ".rotoscope-clip.json")
    files = [p for d in ("stable", "contours", "strokes") for p in (wd / d).rglob("*")] + list(out.rglob("*"))
    res = {}
    for p in sorted(files):
        if p.is_file() and not p.name.endswith(skip):
            data = p.read_bytes()
            if p.name == "clip_stats.json":
                data = json.dumps(_drop_utc(json.loads(data)), sort_keys=True).encode()
            res[p.relative_to(wd.parents[2]).as_posix()] = hashlib.sha256(data).hexdigest()
    return res


@needs_ffmpeg
def test_real_run_is_byte_identical_with_points_present(tmp_path, monkeypatch):
    """Rantai CPU nyata (stabilize → vectorize → stylize → export; ingest + GPU palsu, termasuk [1b] T-605 palsu): run tanpa
    target/ lalu, setelah points.json ditulis dan keluaran CPU dihapus, run lagi → keluaran isi byte-identik; target/ hanya
    points.json, tidak disentuh."""
    import test_stabilize as tst
    from test_cli import Rec

    from rotoscope import export as ex
    from rotoscope import stabilize as stb
    from rotoscope import stylize as sty
    from rotoscope import vectorize as vec

    monkeypatch.chdir(tmp_path)
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x")
    work = tst.make_work(tmp_path / "stage")
    meta = json.loads((work / "meta.json").read_text(encoding="utf-8"))
    meta.update(source_path=str(video.resolve()), target_fps=24, has_audio=False)
    (work / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    wd = tmp_path / "work" / "clips" / "clip"
    wd.parent.mkdir(parents=True)
    shutil.move(str(work), str(wd))
    (wd / "frames").mkdir()
    for i in range(tst.N_FRAMES):
        write_png(wd / "frames" / f"frame_{i:05d}.png", frame_rgb(i, tst.W, tst.H))
    (wd / "qc_report.json").write_text(json.dumps({"stage": "segment_qc", "frames": [
        {"frame": f"frame_{i:05d}.png", "index": i, "fail_reasons": []} for i in range(tst.N_FRAMES)]}), encoding="utf-8")
    from rotoscope.stage_common import clip_identity
    for d in ("seg", "depth"):
        p = wd / d / "manifest.json"
        p.write_text(json.dumps({**json.loads(p.read_text(encoding="utf-8")), "clip": clip_identity(wd)}), encoding="utf-8")
    r = Rec()
    monkeypatch.setattr(cli, "_popen", r.popen)                            # [2] / [2c] palsu (hasil sudah ada)
    monkeypatch.setattr(cli, "CPU_MAINS", {"ingest": r.cpu("ingest"), "stabilize": stb.main, "vectorize": vec.main,
                                           "stylize": sty.main, "export": ex.main})
    out = tmp_path / "out"
    assert cli.main(["run", str(video)]) == 0
    first = _content_hashes(wd, out)
    assert any("strokes/" in k for k in first) and any(k.endswith(".mp4") for k in first)
    assert not (wd / "target").exists()
    save_points(wd, 1, [(3, 4, 1), (8, 9, 0)])
    pts = points_path(wd).read_bytes()
    for d in ("stable", "contours", "strokes"):
        shutil.rmtree(wd / d)
    shutil.rmtree(out)
    assert cli.main(["run", str(video)]) == 0
    assert _content_hashes(wd, out) == first
    assert points_path(wd).read_bytes() == pts and sorted(p.name for p in (wd / "target").iterdir()) == ["points.json"]


# ── (27) sidik piksel walau byte PNG berbeda ───────
def test_pixel_fingerprint_survives_png_recompression(tmp_path):
    work, _ = make_clip(tmp_path)
    save_points(work, 1, [(5, 5, 1)])
    p = work / "frames" / "frame_00001.png"
    old_bytes = p.read_bytes()
    write_png(p, frame_rgb(1), compression=9)                             # piksel sama, kompresi lain
    assert p.read_bytes() != old_bytes
    write_png(work / "frames" / "frame_00002.png", frame_rgb(2), compression=0)
    assert pick.frame_pixel_sha256(p) == pick.pixel_sha256(frame_rgb(1))
    s, st = session_state(work, 1)
    assert s.status == {1: "ok"} and st.content()[1]["points"] == [{"x": 5.0, "y": 5.0, "label": 1}]
    assert pick.commit(s, st, pick.ACT_SAVE, DISPLAY)                       # simpan setelah kompresi ulang: tidak "berubah"
    changed = frame_rgb(1)
    changed[3, 4, 0] ^= 1                                                 # satu nilai piksel berubah → sidik berubah
    assert pick.pixel_sha256(changed) != pick.pixel_sha256(frame_rgb(1))
    a = np.arange(36, dtype=np.uint8).reshape(2, 6, 3)
    assert a.tobytes() == a.reshape(6, 2, 3).tobytes() and pick.pixel_sha256(a) != pick.pixel_sha256(a.reshape(6, 2, 3))
    header = b"rgb8 6x2\n"
    assert pick.pixel_sha256(a) == hashlib.sha256(header + a.tobytes()).hexdigest()


# ── (28) Esc / X dengan dan tanpa perubahan ────────
def test_esc_without_changes_exits_at_once(tmp_path, capsys):
    work, _ = make_clip(tmp_path)
    rc, gui = run_gui(work, 0, [("key", ESC)])
    assert rc == 0 and "Dibatalkan - points.json tidak berubah." in capsys.readouterr().out and not gui.script
    save_points(work, 0, [(1, 1, 1)])
    before = points_path(work).read_bytes()
    rc, _ = run_gui(work, 0, [*click(5, 5), ("key", ord("u")), ("key", ESC)])   # klik lalu U = tidak berubah
    assert rc == 0 and points_path(work).read_bytes() == before


def test_esc_with_changes_asks_first(tmp_path, capsys):
    work, _ = make_clip(tmp_path)
    st = pick.PickState(small_view())
    st.add_point(1, 1, 1)
    assert st.key(ESC) is None and st.mode == pick.MODE_CONFIRM_EXIT and "BELUM disimpan" in st.msg
    assert st.key(ord("x")) is None and st.mode == pick.MODE_NORMAL                  # tombol lain = lanjut
    assert st.key(ESC) is None
    assert st.key(ESC) == pick.ACT_EXIT
    rc, _ = run_gui(work, 0, [*click(5, 5), ("key", ESC), ("key", ESC)])
    assert rc == 0 and not points_path(work).exists()
    rc, _ = run_gui(work, 0, [*click(5, 5), ("key", ESC), ("key", ENTER)])            # Enter saat konfirmasi = simpan
    assert rc == 0 and points_path(work).exists()


def test_close_button_same_as_esc(tmp_path, capsys):
    work, _ = make_clip(tmp_path)
    rc, gui = run_gui(work, 0, [("close",)])                               # tanpa perubahan: X = keluar
    assert rc == 0 and gui.opened == 1
    rc, gui = run_gui(work, 0, [*click(5, 5), ("close",), ("close",)])     # dengan perubahan: dibuka lagi, X kedua = keluar
    assert rc == 0 and gui.opened == 2 and not points_path(work).exists()
    assert "Dibatalkan" in capsys.readouterr().out
    rc, gui = run_gui(work, 0, [*click(5, 5), ("close",), ("key", ENTER)])  # X lalu Enter = simpan
    assert rc == 0 and gui.opened == 2 and pick.load_points(points_path(work))["frames"][0]["targets"][0]["points"]
    st = pick.PickState(small_view())
    st.add_point(1, 1, 1)
    assert st.request_exit() is None and st.request_exit() == pick.ACT_EXIT


# ── (29) target kosong tidak disimpan ──────────────
def test_empty_targets_are_not_saved(tmp_path):
    work, _ = make_clip(tmp_path)
    script = [*click(5, 5), ("key", ord("n")), *click(9, 9), ("key", ord("u")),    # target 2: titik lalu U = kosong
              ("key", ENTER)]                                                     # target 2 masih aktif, kosong
    rc, _ = run_gui(work, 0, script)
    assert rc == 0
    targets = pick.load_points(points_path(work))["frames"][0]["targets"]
    assert [t["id"] for t in targets] == [1]
    st = pick.PickState(small_view())
    st.add_point(1, 1, 1)
    st.next_target()
    st.set_box(1, 1, 20, 20)
    st.undo()
    assert 2 in st.targets and [t["id"] for t in pick.targets_json(st)] == [1]
