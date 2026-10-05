"""Test cli.py (T-104b): stage dan subprocess dipalsukan — tanpa GPU / model / torch."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import test_depth as td
import test_export as te
import test_segment as ts
import test_stabilize as tst
from rotoscope import cli
from rotoscope import depth as dep
from rotoscope import export as ex
from rotoscope import segment as seg
from rotoscope import stabilize as stb
from rotoscope.stage_common import HF_OFFLINE_ENV, StageError

CPU_STAGES = ("ingest", "stabilize", "vectorize", "stylize", "export")


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    # run_segment / run_depth (layer 2) men-set HF_HUB_OFFLINE; monkeypatch mengembalikan nilai asli.
    monkeypatch.setenv(HF_OFFLINE_ENV, "1")


# ── Palsu: subprocess + stage CPU ──────────────────
class FakeProc:
    def __init__(self, rec, stage, rc, interrupt=False, boom=False):
        self.rec, self.stage, self.rc, self.interrupt, self.boom = rec, stage, rc, interrupt, boom
        self.alive = True

    def wait(self, timeout=None):
        if timeout is None and self.interrupt:
            raise KeyboardInterrupt
        if timeout is None and self.boom:
            raise RuntimeError("boom")
        self.alive = False
        self.rec.waited.append(self.stage)
        return self.rc

    def poll(self):
        return None if self.alive else self.rc

    def terminate(self):
        self.rec.terminated.append(self.stage)


class Rec:
    def __init__(self):
        self.calls: list[tuple[str, str, list[str], object]] = []
        self.rc: dict[str, int] = {}
        self.interrupt = self.boom = None
        self.exit_stage = None
        self.terminated: list[str] = []
        self.waited: list[str] = []

    def popen(self, argv, env=None):
        assert argv[:2] == [sys.executable, "-m"]
        stage = argv[2].split(".")[-1]
        self.calls.append(("gpu", stage, list(argv[3:]), env))
        return FakeProc(self, stage, self.rc.get(stage, 0), self.interrupt == stage, self.boom == stage)

    def cpu(self, stage):
        def main(argv):
            self.calls.append(("cpu", stage, list(argv), None))
            if self.exit_stage == stage:
                raise SystemExit(2)
            return self.rc.get(stage, 0)
        return main

    @property
    def stages(self) -> list[str]:
        return [c[1] for c in self.calls]

    def args(self, stage: str) -> list[str]:
        return next(c[2] for c in self.calls if c[1] == stage)


@pytest.fixture
def rec(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    r = Rec()
    monkeypatch.setattr(cli, "_popen", r.popen)
    monkeypatch.setattr(cli, "CPU_MAINS", {s: r.cpu(s) for s in CPU_STAGES})
    return r


def make_video(tmp_path: Path, name: str = "clip.mp4") -> Path:
    p = tmp_path / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"bukan video sungguhan")
    return p


def clip_dir(tmp_path: Path, stem: str = "clip") -> Path:
    return tmp_path / "work" / "clips" / stem


def write_meta(work: Path, source: str | Path, n: int = 283) -> None:
    work.mkdir(parents=True, exist_ok=True)
    (work / "meta.json").write_text(json.dumps({"source_path": str(source), "frame_count": n}), encoding="utf-8")


def run_cli(*argv: str) -> int:
    return cli.main(list(argv))


# ── Urutan, subprocess, work_dir per klip ──────────
def test_run_order_and_clip_workdir(rec, tmp_path):
    video = make_video(tmp_path)
    assert run_cli("run", str(video)) == 0
    assert rec.stages == ["ingest", "segment", "depth", "stabilize", "vectorize", "stylize", "export"]
    want = str(clip_dir(tmp_path).resolve())
    for stage in rec.stages:
        a = rec.args(stage)
        assert Path(a[a.index("--work-dir") + 1]).resolve() == Path(want)
    assert rec.args("ingest")[0] == str(video)
    assert "--restart" not in sum((c[2] for c in rec.calls), [])


def test_gpu_stages_are_subprocess_with_inherited_streams(rec, tmp_path):
    run_cli("run", str(make_video(tmp_path)))
    gpu = [c[0] for c in rec.calls]
    assert gpu == ["cpu", "gpu", "gpu", "cpu", "cpu", "cpu", "cpu"]     # popen palsu tidak menerima stdout/stderr = diwariskan
    assert all(c[3] is None for c in rec.calls)             # environment tidak diubah


def test_config_workdir_and_unsafe_stem(rec, tmp_path):
    other = tmp_path / "drive_lain"
    conf = tmp_path / "c.yaml"
    conf.write_text(f'paths:\n  work_dir: "{other.as_posix()}"\n', encoding="utf-8")
    video = make_video(tmp_path, "Lucu!! (1).final.mp4")
    assert run_cli("run", str(video), "--config", str(conf)) == 0
    a = rec.args("segment")
    assert Path(a[a.index("--work-dir") + 1]) == other / "clips" / ex.sanitize_source_name(str(video))
    assert "--config" in a and a[a.index("--config") + 1] == str(conf)
    assert "--config" not in rec.args("ingest")             # ingest tidak membaca config
    assert all("--out-dir" not in c[2] for c in rec.calls)  # paths.out_dir tidak berubah


def test_flags_forwarded(rec, tmp_path):
    run_cli("run", str(make_video(tmp_path)), "--seg-model", "0.4b", "--limit", "3")
    assert rec.args("segment")[rec.args("segment").index("--seg-model") + 1] == "0.4b"
    for stage in ("depth", "stabilize", "vectorize", "stylize", "export"):
        assert "--seg-model" not in rec.args(stage)
    for stage in ("segment", "depth", "stabilize", "vectorize", "stylize", "export"):
        a = rec.args(stage)
        assert a[a.index("--limit") + 1] == "3"
    assert "--limit" not in rec.args("ingest")


def test_seg_model_never_changes_silently(rec, tmp_path):
    run_cli("run", str(make_video(tmp_path)))
    assert "--seg-model" not in rec.args("segment")
    rec.rc["segment"] = 1                                   # VRAM kurang → TIDAK mengulang dengan 0.4b
    rec.calls.clear()
    assert run_cli("run", str(make_video(tmp_path))) == 1
    assert rec.stages == ["ingest", "segment"]
    assert all("0.4b" not in c[2] for c in rec.calls)


def test_torch_not_imported_by_cli():
    """Prinsip #3: proses induk tidak boleh meng-import torch (CUDA context hanya hidup di proses stage GPU)."""
    code = "import sys, rotoscope.cli; sys.exit(1 if 'torch' in sys.modules else 0)"
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


# ── Exit code + pesan berhenti ─────────────────────
@pytest.mark.parametrize("stage, rc, ran", [
    ("segment", 1, ["ingest", "segment"]), ("segment", 3, ["ingest", "segment"]),
    ("depth", 3, ["ingest", "segment", "depth"]), ("depth", 1, ["ingest", "segment", "depth"]),
    ("stabilize", 1, ["ingest", "segment", "depth", "stabilize"]),
    ("vectorize", 1, ["ingest", "segment", "depth", "stabilize", "vectorize"]),
    ("stylize", 1, ["ingest", "segment", "depth", "stabilize", "vectorize", "stylize"]),
    ("export", 1, ["ingest", "segment", "depth", "stabilize", "vectorize", "stylize", "export"]),
    ("ingest", 1, ["ingest"]),
])
def test_stage_failure_propagates_exit_code(rec, tmp_path, capsys, stage, rc, ran):
    rec.rc[stage] = rc
    assert run_cli("run", str(make_video(tmp_path))) == rc
    assert rec.stages == ran
    err = capsys.readouterr().err
    assert cli.LABELS[stage] in err and f"exit {rc}" in err and "resume" in err


def test_resume_hint_after_restart_from_drops_restart(rec, tmp_path, capsys):
    rec.rc["segment"] = 3
    run_cli("run", str(make_video(tmp_path)), "--restart-from", "segment", "--yes")
    err = capsys.readouterr().err
    assert "TANPA --restart-from segment" in err


def test_cpu_stage_argparse_exit_is_named(rec, tmp_path, capsys):
    rec.exit_stage = "stabilize"
    assert run_cli("run", str(make_video(tmp_path))) == 2
    assert "[3] stabilize" in capsys.readouterr().err
    assert rec.stages[-1] == "stabilize"


def test_interrupt_terminates_child_and_waits(rec, tmp_path, capsys):
    rec.interrupt = "segment"
    assert run_cli("run", str(make_video(tmp_path))) == cli.EXIT_INTERRUPTED
    assert rec.terminated == ["segment"] and rec.waited == ["segment"]
    err = capsys.readouterr().err
    assert "dihentikan" in err and "resume" in err and rec.stages == ["ingest", "segment"]


def test_exception_terminates_child(rec, tmp_path, capsys):
    rec.boom = "depth"
    assert run_cli("run", str(make_video(tmp_path))) == 1
    assert rec.terminated == ["depth"] and rec.waited == ["segment", "depth"]   # segment selesai normal
    assert "depth" in capsys.readouterr().err and "stabilize" not in rec.stages


# ── Pre-flight: gagal → tidak ada stage / subprocess / penghapusan ──
def test_preflight_missing_video(rec, tmp_path, capsys):
    assert run_cli("run", str(tmp_path / "tidak-ada.mp4")) == 1
    assert rec.calls == [] and "tidak ditemukan" in capsys.readouterr().err


@pytest.mark.parametrize("extra", [[], ["--restart-from", "ingest", "--yes"], ["--restart-from", "segment", "--yes"]])
def test_preflight_same_name_other_folder(rec, tmp_path, capsys, extra):
    video = make_video(tmp_path / "b")
    wd = clip_dir(tmp_path)
    write_meta(wd, tmp_path / "a" / "clip.mp4")
    (wd / "frames").mkdir()
    (wd / "frames" / "frame_00000.png").write_bytes(b"A")
    (wd / "seg").mkdir()
    (wd / "seg" / "sentinel").write_text("A")
    assert run_cli("run", str(video), *extra) == 1
    assert rec.calls == []
    assert (wd / "frames" / "frame_00000.png").read_bytes() == b"A" and (wd / "seg" / "sentinel").exists()
    err = capsys.readouterr().err
    assert str((tmp_path / "a" / "clip.mp4")) in err and str(video.resolve()) in err
    assert "--restart" not in err and "Ganti nama" in err


def test_preflight_same_video_other_spelling_is_not_a_clash(rec, tmp_path):
    video = make_video(tmp_path)
    write_meta(clip_dir(tmp_path), video.resolve())
    spelled = tmp_path / "sub" / ".." / "clip.mp4"
    (tmp_path / "sub").mkdir()
    assert run_cli("run", str(spelled)) == 0
    assert rec.stages[0] == "ingest"


def test_preflight_export_target_of_other_video(rec, tmp_path, capsys):
    video = make_video(tmp_path / "b")
    out = tmp_path / "out"
    out.mkdir()
    (out / "clip.mp4").write_bytes(b"mp4 lama")
    (out / "clip.export.json").write_text(json.dumps({"clip": {"source_path": str(tmp_path / "a" / "clip.mp4")}}),
                                          encoding="utf-8")
    assert run_cli("run", str(video), "--restart-from", "ingest", "--yes") == 1
    assert rec.calls == []
    err = capsys.readouterr().err
    assert str(tmp_path / "a" / "clip.mp4") in err and "--restart" not in err and "export.filename" in err
    # --limit menulis <nama>.limitN.mp4 tanpa pengaman → pre-flight (c) dilewati
    assert run_cli("run", str(video), "--limit", "2") == 0


def test_preflight_export_target_same_video_ok(rec, tmp_path):
    video = make_video(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    (out / "clip.mp4").write_bytes(b"mp4 lama")
    (out / "clip.export.json").write_text(json.dumps({"clip": {"source_path": str(video.resolve())}}), encoding="utf-8")
    assert run_cli("run", str(video)) == 0


def write_conf(tmp_path: Path, body: str) -> Path:
    conf = tmp_path / "c.yaml"
    conf.write_text(body, encoding="utf-8")
    return conf


def test_style_forwarded_only_to_stylize(rec, tmp_path):
    style = tmp_path / "s.yaml"
    style.write_text("stroke:\n  width_base: 7.0\n", encoding="utf-8")
    assert run_cli("run", str(make_video(tmp_path)), "--style", str(style)) == 0
    for stage in rec.stages:
        a = rec.args(stage)
        assert ("--style" in a) == (stage == "stylize")
    a = rec.args("stylize")
    assert a[a.index("--style") + 1] == str(style)


@pytest.mark.parametrize("body", ["render:\n  output_width: 1001\n", "stroke:\n  width_base: -3\n", "bukan: [yaml"])
def test_preflight_invalid_style_runs_nothing(rec, tmp_path, capsys, body):
    style = tmp_path / "s.yaml"
    style.write_text(body, encoding="utf-8")
    wd = clip_dir(tmp_path)
    write_meta(wd, make_video(tmp_path))
    (wd / "seg").mkdir()
    (wd / "seg" / "sentinel").write_text("x")
    assert run_cli("run", str(tmp_path / "clip.mp4"), "--style", str(style), "--restart-from", "ingest", "--yes") == 1
    assert rec.calls == [] and (wd / "seg" / "sentinel").exists()
    assert "ERROR" in capsys.readouterr().err


def test_preflight_missing_style_file_runs_nothing(rec, tmp_path):
    assert run_cli("run", str(make_video(tmp_path)), "--style", str(tmp_path / "tidak-ada.yaml")) == 1
    assert rec.calls == []


def test_preflight_invalid_export_source_runs_nothing(rec, tmp_path, capsys):
    conf = write_conf(tmp_path, "export:\n  source: hologram\n")
    assert run_cli("run", str(make_video(tmp_path)), "--config", str(conf)) == 1
    assert rec.calls == [] and "export.source" in capsys.readouterr().err


def test_preflight_ffmpeg_missing_runs_nothing(rec, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli.shutil, "which", lambda name: None)
    assert run_cli("run", str(make_video(tmp_path))) == 1
    assert rec.calls == [] and "ffmpeg" in capsys.readouterr().err


def test_preflight_svg_folder_of_other_video(rec, tmp_path, capsys):
    conf = write_conf(tmp_path, "export:\n  source: strokes\n")
    video = make_video(tmp_path / "b")
    svg = tmp_path / "out" / "svg" / "clip"
    svg.mkdir(parents=True)
    (svg / ex.SVG_MARKER).write_text(json.dumps({"clip": {"source_path": str(tmp_path / "a" / "clip.mp4")}}),
                                     encoding="utf-8")
    assert run_cli("run", str(video), "--config", str(conf), "--restart-from", "ingest", "--yes") == 1
    assert rec.calls == []
    err = capsys.readouterr().err
    assert str(tmp_path / "a" / "clip.mp4") in err and "--restart" not in err
    assert run_cli("run", str(video), "--config", str(conf), "--limit", "2") == 0      # --limit tidak menyalin SVG
    # source silhouette tidak menyalin SVG → folder itu tidak relevan
    sil = write_conf(tmp_path, "export:\n  source: silhouette\n")
    assert run_cli("run", str(video), "--config", str(sil)) == 0


def test_preflight_corrupt_svg_marker(rec, tmp_path, capsys):
    """Penanda rusak ≠ tanpa penanda; --restart-from export/ingest hanya lolos bila semua SVG identik dengan strokes/."""
    conf = write_conf(tmp_path, "export:\n  source: strokes\n")
    video = make_video(tmp_path)
    wd = clip_dir(tmp_path)
    (wd / "strokes").mkdir(parents=True)
    (wd / "strokes" / "frame_00000.svg").write_text("<svg>A</svg>", encoding="utf-8")
    svg = tmp_path / "out" / "svg" / "clip"
    svg.mkdir(parents=True)
    (svg / "frame_00000.svg").write_text("<svg>A</svg>", encoding="utf-8")
    (svg / ex.SVG_MARKER).write_bytes(b"rusak {")
    assert run_cli("run", str(video), "--config", str(conf)) == 1                       # tanpa restart
    assert rec.calls == [] and "ada tetapi bukan JSON valid" in capsys.readouterr().err
    assert run_cli("run", str(video), "--config", str(conf), "--restart-from", "export") == 0   # identik → boleh
    rec.calls.clear()
    (svg / "frame_00000.svg").write_text("<svg>suntingan</svg>", encoding="utf-8")
    assert run_cli("run", str(video), "--config", str(conf), "--restart-from", "ingest", "--yes") == 1
    assert rec.calls == []
    err = capsys.readouterr().err
    assert "frame_00000.svg" in err and "walau --restart" in err
    assert (svg / "frame_00000.svg").read_text(encoding="utf-8") == "<svg>suntingan</svg>"
    # restart yang TIDAK mencakup export tidak melonggarkan pengaman
    assert run_cli("run", str(video), "--config", str(conf), "--restart-from", "stylize") == 1 and rec.calls == []
    # --limit tidak menyalin SVG → pre-flight dilewati
    assert run_cli("run", str(video), "--config", str(conf), "--limit", "2") == 0


def test_limit_forwarded_to_vectorize_stylize_export(rec, tmp_path):
    run_cli("run", str(make_video(tmp_path)), "--limit", "4")
    for stage in ("vectorize", "stylize", "export"):
        a = rec.args(stage)
        assert a[a.index("--limit") + 1] == "4"


# ── Restart: graf dependensi + --yes ───────────────
RESTARTS = {
    "ingest": {"segment", "depth", "stabilize", "vectorize", "stylize", "export"},
    "segment": {"segment"}, "depth": {"depth"}, "stabilize": {"stabilize"}, "vectorize": {"vectorize"},
    "stylize": {"stylize"}, "export": {"export"},
}


@pytest.mark.parametrize("frm", list(RESTARTS))
def test_restart_from_scope(rec, tmp_path, frm):
    assert run_cli("run", str(make_video(tmp_path)), "--restart-from", frm, "--yes") == 0
    got = {c[1] for c in rec.calls if "--restart" in c[2]}
    assert got == RESTARTS[frm]
    assert all("--yes" not in c[2] for c in rec.calls)       # cli membuang --yes


@pytest.mark.parametrize("frm", ["ingest", "segment", "depth"])
def test_restart_from_gpu_requires_yes(rec, tmp_path, capsys, frm):
    wd = clip_dir(tmp_path)
    write_meta(wd, make_video(tmp_path))
    (wd / "seg").mkdir()
    (wd / "seg" / "frame_00000.npz").write_bytes(b"x" * 100)
    assert run_cli("run", str(tmp_path / "clip.mp4"), "--restart-from", frm) == 1
    assert rec.calls == [] and (wd / "seg" / "frame_00000.npz").exists()
    err = capsys.readouterr().err
    assert "--yes" in err and "Tidak ada yang dihapus" in err and "mnt GPU" in err


@pytest.mark.parametrize("frm", ["stabilize", "vectorize", "stylize", "export"])
def test_restart_from_cpu_does_not_need_yes(rec, tmp_path, frm):
    assert run_cli("run", str(make_video(tmp_path)), "--restart-from", frm) == 0


def test_preview_lists_files_and_estimate(rec, tmp_path, capsys):
    wd = clip_dir(tmp_path)
    write_meta(wd, make_video(tmp_path), n=100)
    (wd / "seg").mkdir()
    for i in range(3):
        (wd / "seg" / f"f{i}").write_bytes(b"x" * 1024)
    (wd / "depth").mkdir()
    (wd / "depth" / "f").write_bytes(b"y")
    run_cli("run", str(tmp_path / "clip.mp4"), "--restart-from", "ingest")
    err = capsys.readouterr().err
    assert "[2] segment" in err and "3 file" in err and "[2c] depth" in err and "1 file" in err
    assert f"{100 * 16.5 / 60:.1f} mnt" in err               # 0.8b: 16.5 s/frame


@pytest.mark.parametrize("stage", ["segment", "depth"])
def test_subcommand_restart_requires_yes(rec, tmp_path, capsys, stage):
    video = make_video(tmp_path)
    assert run_cli(stage, str(video), "--restart") == 1
    assert rec.calls == [] and "--yes" in capsys.readouterr().err
    assert run_cli(stage, str(video), "--restart", "--yes") == 0
    a = rec.args(stage)
    assert "--restart" in a and "--yes" not in a


def test_subcommand_passthrough_flags(rec, tmp_path):
    video = make_video(tmp_path)
    run_cli("segment", str(video), "--qc-only")
    run_cli("depth", str(video), "--adopt")
    run_cli("export", str(video), "--limit", "3")
    run_cli("stabilize", str(video), "--limit", "2", "--restart")
    assert "--qc-only" in rec.args("segment") and "--adopt" in rec.args("depth")
    assert rec.args("export")[-2:] == ["--limit", "3"]
    assert "--restart" in rec.args("stabilize")             # CPU: tanpa --yes
    assert all("--work-dir" in c[2] for c in rec.calls)


def test_vectorize_subcommand_and_in_run(rec, tmp_path):
    """T-201a: [4] = subperintah sendiri (CPU, tanpa --yes); sejak T-203b juga bagian urutan `run`."""
    video = make_video(tmp_path)
    assert run_cli("vectorize", str(video), "--limit", "2", "--restart") == 0
    a = rec.args("vectorize")
    assert a[0] == "--work-dir" and Path(a[1]).name == "clip" and "--restart" in a and a[-2:] == ["--limit", "2"]
    assert "--yes" not in a and rec.stages == ["vectorize"]
    assert run_cli("vectorize", str(video), "--work-dir", "x") == 2
    assert "vectorize" in cli.STAGES and "vectorize" in cli.RESTART_SCOPE["ingest"]


def test_work_dir_flag_vectorize(tmp_path):
    import test_vectorize as tvec
    from rotoscope import vectorize as vec_stage
    work, wrong = tvec.make_work(tmp_path), tmp_path / "salah"
    assert vec_stage.main(["--config", str(_conf(tmp_path, wrong)), "--work-dir", str(work), "--limit", "1"]) == 0
    assert (work / "contours" / "manifest.json").is_file() and not wrong.exists()


def test_subcommand_ingest_preflight_and_no_restart(rec, tmp_path, capsys):
    video = make_video(tmp_path / "b")
    write_meta(clip_dir(tmp_path), tmp_path / "a" / "clip.mp4")
    assert run_cli("ingest", str(video)) == 1 and rec.calls == []
    with pytest.raises(SystemExit):
        run_cli("ingest")
    assert cli.main(["ingest", str(make_video(tmp_path / "c", "z.mp4")), "--restart"]) == 2


def test_subcommand_rejects_own_work_dir(rec, tmp_path):
    assert run_cli("segment", str(make_video(tmp_path)), "--work-dir", "x") == 2 and rec.calls == []


def test_run_rejects_adopt_and_qc_only(rec, tmp_path):
    video = make_video(tmp_path)
    for flag in ("--adopt", "--qc-only"):
        assert run_cli("run", str(video), flag) == 1
    assert rec.calls == []


def test_run_validates_limit_and_config(rec, tmp_path):
    video = make_video(tmp_path)
    assert run_cli("run", str(video), "--limit", "0") == 1
    assert run_cli("run", str(video), "--config", str(tmp_path / "tidak-ada.yaml")) == 1
    assert rec.calls == []


# ── QC ─────────────────────────────────────────────
@pytest.mark.parametrize("n_fail", [3, 12])
def test_qc_warning_once_and_run_continues(rec, tmp_path, capsys, n_fail):
    wd = clip_dir(tmp_path)
    wd.mkdir(parents=True)
    (wd / "qc_report.json").write_text(json.dumps({"summary": {"n_frames": 283, "n_fail": n_fail,
                                                               "fail_frames": list(range(100, 100 + n_fail))}}),
                                       encoding="utf-8")
    assert run_cli("run", str(make_video(tmp_path))) == 0
    err = capsys.readouterr().err
    assert err.count("frame gagal QC") == 1 and f"{n_fail}/283 frame gagal QC" in err
    assert "100, 101, 102" in err and ("109" in err) == (n_fail == 12) and "110" not in err
    assert rec.stages[-1] == "export"


def test_no_qc_warning_when_clean(rec, tmp_path, capsys):
    run_cli("run", str(make_video(tmp_path)))
    assert "QC" not in capsys.readouterr().err


# ── download ───────────────────────────────────────
def test_download_default_and_fallback_without_forcing_offline(rec, tmp_path, monkeypatch):
    monkeypatch.delenv(HF_OFFLINE_ENV, raising=False)
    assert run_cli("download") == 0
    assert [(c[1], c[2]) for c in rec.calls] == [("segment", ["--download"]), ("depth", ["--download"])]
    assert all(c[3] is None for c in rec.calls) and HF_OFFLINE_ENV not in os.environ
    rec.calls.clear()
    assert run_cli("download", "--seg-model", "0.4b") == 0
    assert rec.args("segment") == ["--download", "--seg-model", "0.4b"] and rec.args("depth") == ["--download"]


def test_download_stops_on_failure(rec):
    rec.rc["segment"] = 1
    assert run_cli("download") == 1 and rec.stages == ["segment"]


# ── Pesan error stage memakai perintah CLI lengkap ──
@pytest.mark.parametrize("stage", ["segment", "depth"])
def test_stage_errors_use_full_cli_command(tmp_path, stage):
    mod, t, cfg_fn, fac = ((seg, ts, ts.make_cfg, ts.factory) if stage == "segment"
                           else (dep, td, td.make_cfg, td.factory))
    work = t.make_work(tmp_path)
    run = mod.run_segment if stage == "segment" else mod.run_depth
    run(cfg_fn(work), backend_factory=fac(), log=t.quiet)
    meta = json.loads((work / "meta.json").read_text(encoding="utf-8"))
    (work / "meta.json").write_text(json.dumps({**meta, "source_path": "C:/clips/lain.mp4"}), encoding="utf-8")
    with pytest.raises(StageError) as e:
        run(cfg_fn(work), backend_factory=fac(), log=t.quiet)
    msg = str(e.value)
    assert f"python -m rotoscope {stage} C:/clips/lain.mp4 --restart --yes" in msg
    assert "rotoscope.segment" not in msg and "rotoscope.depth" not in msg
    # manifest lama tanpa identitas → saran --adopt + --restart --yes
    m = json.loads((work / ("seg" if stage == "segment" else "depth") / "manifest.json").read_text(encoding="utf-8"))
    m.pop("clip")
    (work / ("seg" if stage == "segment" else "depth") / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(StageError) as e2:
        run(cfg_fn(work), backend_factory=fac(), log=t.quiet)
    assert f"rotoscope {stage} C:/clips/lain.mp4 --adopt" in str(e2.value) and "--restart --yes" in str(e2.value)


def test_stabilize_input_errors_use_full_cli_command(tmp_path):
    work = tst.make_work(tmp_path)
    p = work / "seg" / "manifest.json"
    m = json.loads(p.read_text(encoding="utf-8"))
    m.pop("clip")
    p.write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(StageError) as e:
        stb.run_stabilize(tst.cfg_for(work), log=tst.quiet)
    assert "rotoscope segment C:/clips/a.mp4 --adopt" in str(e.value)
    assert "rotoscope segment C:/clips/a.mp4 --restart --yes" in str(e.value)


# ── --work-dir di tiap stage menang atas config; out_dir tidak berubah ──
def _conf(tmp_path: Path, wrong: Path, out: Path | None = None, source: str | None = None) -> Path:
    conf = tmp_path / "c.yaml"
    out_line = f'  out_dir: "{out.as_posix()}"\n' if out else ""
    src_line = f"export:\n  source: {source}\n" if source else ""
    conf.write_text(f'paths:\n  work_dir: "{wrong.as_posix()}"\n{out_line}{src_line}', encoding="utf-8")
    return conf


def test_work_dir_flag_segment_and_depth(tmp_path):
    work, wrong = ts.make_work(tmp_path), tmp_path / "salah"
    conf = _conf(tmp_path, wrong)
    assert seg.main(["--config", str(conf), "--work-dir", str(work), "--limit", "2"],
                    backend_factory=ts.factory()) == 0
    assert (work / "seg" / "manifest.json").is_file() and not wrong.exists()
    assert dep.main(["--config", str(conf), "--work-dir", str(work), "--limit", "2"],
                    backend_factory=td.factory()) == 0
    assert (work / "depth" / "manifest.json").is_file() and not wrong.exists()


def test_work_dir_flag_stabilize(tmp_path):
    work, wrong = tst.make_work(tmp_path), tmp_path / "salah"
    assert stb.main(["--config", str(_conf(tmp_path, wrong)), "--work-dir", str(work), "--limit", "1"]) == 0
    assert (work / "stable" / "manifest.json").is_file() and not wrong.exists()


@pytest.mark.skipif(te.pytestmark.args[0], reason="ffmpeg/ffprobe tidak ada di PATH")
def test_work_dir_flag_export_keeps_out_dir(tmp_path):
    _, work, _ = te.make_clip(tmp_path)
    out2, wrong = tmp_path / "out2", tmp_path / "salah"
    assert ex.main(["--config", str(_conf(tmp_path, wrong, out2, "silhouette")), "--work-dir", str(work)]) == 0
    assert (out2 / "meme_clip.mp4").is_file() and not wrong.exists()


# ── Layer 2: stage nyata (backend palsu) di folder klip ──
class InProc:
    """Menjalankan main() stage GPU nyata (backend palsu) sinkron, seolah subprocess."""

    def __init__(self, argv, env=None):
        mains = {"rotoscope.segment": lambda a: seg.main(a, backend_factory=ts.factory()),
                 "rotoscope.depth": lambda a: dep.main(a, backend_factory=td.factory())}
        self.rc = mains[argv[2]](list(argv[3:]))

    def wait(self, timeout=None):
        return self.rc

    def poll(self):
        return self.rc

    def terminate(self):
        pass


@pytest.fixture
def real(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    video = make_video(tmp_path)
    work = ts.make_work(tmp_path / "stage")
    meta = json.loads((work / "meta.json").read_text(encoding="utf-8"))
    (work / "meta.json").write_text(json.dumps({**meta, "source_path": str(video.resolve())}), encoding="utf-8")
    dest = clip_dir(tmp_path)
    dest.parent.mkdir(parents=True)
    shutil.move(str(work), str(dest))
    r = Rec()
    monkeypatch.setattr(cli, "_popen", InProc)
    monkeypatch.setattr(cli, "CPU_MAINS", {s: r.cpu(s) for s in CPU_STAGES})
    assert run_cli("run", str(video)) == 0                  # seg + depth terisi (backend palsu)
    for d in ("seg", "depth"):
        (dest / d / "sentinel.keep").write_text("s")
    return video, dest, r


def depth_bytes(wd: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted((wd / "depth").glob("frame_*.npy"))}


def test_real_restart_from_segment_leaves_depth(real):
    video, wd, _ = real
    before = depth_bytes(wd)
    assert before
    assert run_cli("run", str(video), "--restart-from", "segment", "--yes") == 0
    assert not (wd / "seg" / "sentinel.keep").exists() and (wd / "seg" / "manifest.json").is_file()
    assert (wd / "depth" / "sentinel.keep").exists() and depth_bytes(wd) == before


def test_real_restart_from_depth_leaves_segment(real):
    video, wd, _ = real
    assert run_cli("run", str(video), "--restart-from", "depth", "--yes") == 0
    assert not (wd / "depth" / "sentinel.keep").exists() and (wd / "seg" / "sentinel.keep").exists()


def test_real_restart_from_ingest_deletes_everything(real):
    video, wd, r = real
    assert run_cli("run", str(video), "--restart-from", "ingest", "--yes") == 0
    for d in ("seg", "depth"):
        assert not (wd / d / "sentinel.keep").exists() and (wd / d / "manifest.json").is_file()
    assert {c[1] for c in r.calls if "--restart" in c[2]} >= {"stabilize", "export"}


def test_real_restart_without_yes_deletes_nothing(real):
    video, wd, r = real
    before, calls = depth_bytes(wd), len(r.calls)
    for frm in ("ingest", "segment", "depth"):
        assert run_cli("run", str(video), "--restart-from", frm) == 1
    assert run_cli("segment", str(video), "--restart") == 1
    assert (wd / "seg" / "sentinel.keep").exists() and (wd / "depth" / "sentinel.keep").exists()
    assert depth_bytes(wd) == before and len(r.calls) == calls


def test_real_rerun_resumes_without_inference(real, capsys):
    video, wd, _ = real
    capsys.readouterr()
    assert run_cli("run", str(video)) == 0
    out = capsys.readouterr().out
    assert "0 diproses" in out and "dilewati" in out


# ── Ingest = opsi A (selalu ingest ulang): meta.json byte sama → lolos; video berubah di path sama → ditolak ──
@pytest.mark.skipif(te.pytestmark.args[0], reason="ffmpeg/ffprobe tidak ada di PATH")
def test_reingest_same_clip_passes_changed_video_is_rejected(tmp_path):
    import test_ingest as ti
    from rotoscope.ingest import ingest
    work, video = tmp_path / "work", tmp_path / "v.mp4"
    ti.make_video(video, 64, 48, fps=24, duration=0.25)
    ingest(video, work)
    seg_cfg, dep_cfg = ts.make_cfg(work), td.make_cfg(work)
    seg.run_segment(seg_cfg, backend_factory=ts.factory(), log=ts.quiet)
    dep.run_depth(dep_cfg, backend_factory=td.factory(), log=td.quiet)
    before = (work / "meta.json").read_bytes()

    ingest(video, work)                                     # klip yang sama, diingest ulang (opsi A)
    assert (work / "meta.json").read_bytes() == before
    assert seg.run_segment(seg_cfg, backend_factory=ts.factory(), log=ts.quiet)["processed"] == 0
    assert dep.run_depth(dep_cfg, backend_factory=td.factory(), log=td.quiet)["processed"] == 0

    ti.make_video(video, 64, 48, fps=24, duration=0.5)      # video lain menimpa path yang sama
    ingest(video, work)
    assert (work / "meta.json").read_bytes() != before      # meta.json ditulis ulang → identitas berubah
    for run, cfg, fac, q in ((seg.run_segment, seg_cfg, ts.factory, ts.quiet),
                             (dep.run_depth, dep_cfg, td.factory, td.quiet)):
        with pytest.raises(StageError, match="LAIN"):
            run(cfg, backend_factory=fac(), log=q)
