"""T-603 keputusan Rio no. 3: Ctrl+C di stage GPU → satu baris "dihentikan (Ctrl+C) di …", exit 130, tanpa traceback; cleanup
backend yang ikut kena Ctrl+C tidak memunculkan traceback kedua. Backend palsu (tanpa GPU / torch)."""

from __future__ import annotations

import json

import pytest

import test_depth as td
import test_segment as ts
from rotoscope import cli
from rotoscope import depth as dep
from rotoscope import isolate as iso
from rotoscope import segment as seg
from rotoscope.stage_common import StageInterrupted

from test_cli import make_video, rec, run_cli  # noqa: F401  (fixture rec + helper cli dipakai test di bawah)


class CtrlCSegBackend(ts.FakeBackend):
    """infer() kena Ctrl+C pada panggilan ke-`at` (0-based); close() kena Ctrl+C lagi (seperti L3: empty_cache)."""

    def __init__(self, cfg, at: int, open_ctrlc: bool = False):
        super().__init__(cfg)
        self.at, self.open_ctrlc = at, open_ctrlc

    def open(self):
        if self.open_ctrlc:
            raise KeyboardInterrupt
        return super().open()

    def infer(self, rgb):
        if self.calls == self.at:
            raise KeyboardInterrupt
        return super().infer(rgb)

    def close(self):
        self.closed += 1
        raise KeyboardInterrupt


class CtrlCDepthBackend(td.FakeBackend):
    def __init__(self, cfg, at: int):
        super().__init__(cfg)
        self.at = at

    def infer(self, rgb):
        if self.calls == self.at:
            raise KeyboardInterrupt
        return super().infer(rgb)

    def close(self):
        self.closed += 1
        raise KeyboardInterrupt


def factory_of(cls, **kw):
    made = []

    def f(cfg):
        b = cls(cfg, **kw)
        made.append(b)
        return b
    f.made = made
    return f


def one_line(err: str, needle: str) -> str:
    lines = [ln for ln in err.splitlines() if needle in ln]
    assert len(lines) == 1, err
    return lines[0]


# ── Segment ───────────────────────────────────────
def test_segment_ctrl_c_in_frame_one_line_exit_130(tmp_path, capsys):
    work = ts.make_work(tmp_path)
    f = factory_of(CtrlCSegBackend, at=2)
    rc = ts.main_with(work, f)
    err = capsys.readouterr().err
    assert rc == seg.EXIT_INTERRUPTED == 130
    line = one_line(err, "dihentikan (Ctrl+C)")
    assert line == "dihentikan (Ctrl+C) di frame 3/6 (frame_00002.png); jalankan ulang untuk melanjutkan"
    assert "Traceback" not in err and "KeyboardInterrupt" not in err
    b = f.made[0]
    assert b.closed == 1                                       # close() dipanggil walau Ctrl+C kedua muncul di sana
    assert (work / "seg" / "probs" / "frame_00001.npz").exists() and not (work / "seg" / "probs" / "frame_00002.npz").exists()
    events = [json.loads(ln)["event"] for ln in (work / "seg" / "frames.jsonl").read_text(encoding="utf-8").splitlines()]
    assert events[-1] == "run_end"                             # log tetap tertutup rapi


def test_segment_ctrl_c_while_loading_model(tmp_path, capsys):
    work = ts.make_work(tmp_path)
    rc = ts.main_with(work, factory_of(CtrlCSegBackend, at=0, open_ctrlc=True))
    err = capsys.readouterr().err
    assert rc == 130 and "Traceback" not in err
    assert one_line(err, "dihentikan (Ctrl+C)") == "dihentikan (Ctrl+C) di [2] segment; jalankan ulang untuk melanjutkan"


# ── Depth ─────────────────────────────────────────
def test_depth_ctrl_c_in_frame_one_line_exit_130(tmp_path, capsys):
    work = td.make_work(tmp_path)
    f = factory_of(CtrlCDepthBackend, at=3)
    rc = td.main_with(work, f)
    err = capsys.readouterr().err
    assert rc == dep.EXIT_INTERRUPTED == 130
    assert one_line(err, "dihentikan (Ctrl+C)") == "dihentikan (Ctrl+C) di frame 4/6 (frame_00003.png); jalankan ulang untuk melanjutkan"
    assert "Traceback" not in err and f.made[0].closed == 1


# ── Isolate (main memetakan StageInterrupted; SAM tidak dimuat) ─────
def test_isolate_main_maps_stage_interrupted(tmp_path, capsys, monkeypatch):
    def raising(cfg, **kw):
        raise StageInterrupted("[1b] frame 3/6")
    monkeypatch.setattr(iso, "run_isolate", raising)
    rc = iso.main(["--work-dir", str(tmp_path / "work")])
    err = capsys.readouterr().err
    assert rc == 130 and "Traceback" not in err
    assert one_line(err, "dihentikan (Ctrl+C)").startswith("dihentikan (Ctrl+C) di [1b] frame 3/6; isolate/ lama utuh")


# ── Run (proses induk): anak Ctrl+C mencetak sendiri; CPU stage in-process ──
def test_run_gpu_child_reports_once(rec, tmp_path, capsys):
    rec.interrupt = "segment"
    assert run_cli("run", str(make_video(tmp_path))) == cli.EXIT_INTERRUPTED
    err = capsys.readouterr().err
    assert err.count("dihentikan") == 1 and "Traceback" not in err


def test_run_cpu_stage_ctrl_c_single_line(rec, tmp_path, capsys, monkeypatch):
    def raising(argv):
        raise KeyboardInterrupt
    monkeypatch.setattr(cli, "CPU_MAINS", {**cli.CPU_MAINS, "stabilize": raising})
    assert run_cli("run", str(make_video(tmp_path))) == cli.EXIT_INTERRUPTED
    err = capsys.readouterr().err
    assert one_line(err, "dihentikan (Ctrl+C)") == \
        "dihentikan (Ctrl+C) di stage [3] stabilize; jalankan ulang untuk melanjutkan"
    assert "Traceback" not in err


# ── close_quietly ────────────────────────────────
def test_close_quietly_swallows_only_keyboard_interrupt():
    from rotoscope.stage_common import close_quietly

    class Boom:
        def close(self):
            raise KeyboardInterrupt

    class Bad:
        def close(self):
            raise RuntimeError("rusak")
    close_quietly(Boom())                                      # tidak melempar
    with pytest.raises(RuntimeError):
        close_quietly(Bad())
