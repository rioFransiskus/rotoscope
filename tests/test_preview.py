"""Test T-204: `run --preview N [--from K]` + `--from K --limit N` di stylize / export. Stage CPU dipalsukan di tingkat cli
(pola test_cli.py); stage nyata di tingkat stage. Tanpa GPU / model / torch."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import test_export as te
from test_cli import clip_dir, real, run_cli  # noqa: F401 — `real` = fixture (seg + depth palsu terisi, 6 frame)
from test_stylize import N_FRAMES, cfg_for, make_stage_work, out_hashes, quiet, style_for

from rotoscope import cli
from rotoscope import export as ex
from rotoscope import stylize as sty
from rotoscope.config import resolve_style
from rotoscope.stage_common import StageError, window_bounds


# ── window_bounds + nama file ──────────────────────
def test_window_bounds():
    assert window_bounds(6, None, None) == (0, 6)
    assert window_bounds(6, None, 2) == (0, 2)
    assert window_bounds(6, 2, 3) == (2, 5)
    assert window_bounds(6, 3, 3) == (3, 6)
    with pytest.raises(StageError, match="wajib bersama"):
        window_bounds(6, 2, None)
    with pytest.raises(StageError, match="melewati klip"):
        window_bounds(6, 4, 3)
    with pytest.raises(StageError, match="--from harus"):
        window_bounds(6, -1, 2)


def test_preview_filename():
    assert ex.resolve_filename("{source}.mp4", "/a/b.mp4", 10, 73) == "b.preview_73-82.mp4"
    assert ex.resolve_filename("{source}.mp4", "/a/b.mp4", 10) == "b.limit10.mp4"
    assert ex.resolve_filename("{source}.mp4", "/a/b.mp4") == "b.mp4"


# ── stylize --from ─────────────────────────────────
def test_stylize_window_bytes_equal_full_run(tmp_path):
    full_work = make_stage_work(tmp_path / "full")
    sty.run_stylize(cfg_for(full_work), style_for(), "t", log=quiet)
    full = out_hashes(full_work)
    work = make_stage_work(tmp_path / "win")
    run = sty.run_stylize(cfg_for(work), style_for(), "t", limit=2, start=1, log=quiet)
    assert run["processed"] == 2 and run["selected"] == 2
    got = out_hashes(work)
    assert set(got) == {f"frame_{i:05d}.{e}" for i in (1, 2) for e in ("svg", "png")}
    assert all(got[k] == full[k] for k in got)                       # byte-identik dengan run penuh
    run = sty.run_stylize(cfg_for(work), style_for(), "t", log=quiet)    # run penuh sesudahnya melanjutkan
    assert run["skipped"] == 2 and run["processed"] == N_FRAMES - 2 and out_hashes(work) == full


def test_stylize_from_requires_limit_and_fits(tmp_path):
    work = make_stage_work(tmp_path)
    with pytest.raises(StageError, match="wajib bersama --limit"):
        sty.run_stylize(cfg_for(work), style_for(), "t", start=1, log=quiet)
    with pytest.raises(StageError, match="melewati klip"):
        sty.run_stylize(cfg_for(work), style_for(), "t", start=N_FRAMES - 1, limit=2, log=quiet)
    assert not (work / "strokes").exists()


def test_stylize_stale_window_warns_once_and_leaves_half_folder(tmp_path):
    work = make_stage_work(tmp_path)
    sty.run_stylize(cfg_for(work), style_for(), "t", log=quiet)
    logs: list[str] = []
    run = sty.run_stylize(cfg_for(work), style_for(**{"stroke.color": "#112233"}), "t", limit=2, start=1,
                          log=logs.append)
    assert run["processed"] == 2
    warns = [m for m in logs if "PERINGATAN" in m]
    assert len(warns) == 1 and f"DIHAPUS seluruhnya ({N_FRAMES} frame)" in warns[0] and "run <video>" in warns[0]
    clip = sty.load_clip(work)
    size = (sty.make_geometry(style_for(**{"stroke.color": "#112233"}), clip.width, clip.height).out_w,) * 2
    assert [sty.frame_valid(clip, n, size) for n in clip.names] == [False, True, True] + [False] * (N_FRAMES - 3)
    # export penuh (source strokes) pada folder separuh: GAGAL KERAS dengan perintah stylize, bukan MP4 separuh
    src = ex.StrokesSource(work, size)
    with pytest.raises(StageError, match="stylize"):
        src.check(list(clip.names))
    src.check(list(clip.names[1:3]))                                  # jendela itu sendiri valid


# ── export --from ──────────────────────────────────
def test_export_window_is_separate_preview(tmp_path):
    cfg, _, out = te.make_clip(tmp_path)
    full = ex.run_export(cfg, log=quiet)["output"]
    before = full.read_bytes(), (out / "meme_clip.export.json").read_bytes()
    run = ex.run_export(cfg, limit=2, start=3, log=quiet)
    assert run["output"].name == "meme_clip.preview_3-4.mp4" and run["svg"] is None
    assert ex.probe_output(run["output"])["frames"] == 2
    assert (full.read_bytes(), (out / "meme_clip.export.json").read_bytes()) == before
    assert sorted(p.name for p in out.glob("*.export.json")) == ["meme_clip.export.json"]
    assert not (out / "svg").exists()
    with pytest.raises(StageError, match="wajib bersama --limit"):
        ex.run_export(cfg, start=1, log=quiet)
    with pytest.raises(StageError, match="melewati klip"):
        ex.run_export(cfg, start=5, limit=2, log=quiet)


# ── run --preview (cli, stage CPU palsu) ───────────
def stage_calls(r) -> list[tuple[str, list[str]]]:
    return [(c[1], c[2]) for c in r.calls]


def opt(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


def test_preview_runs_cpu_stages_with_window_args(real, monkeypatch):   # noqa: F811
    video, wd, r = real
    r.calls.clear()
    spawned = []
    monkeypatch.setattr(cli, "_popen", lambda *a, **k: spawned.append(a))   # subprocess GPU = gagal
    style = wd.parent / "s.yaml"
    assert run_cli("run", str(video), "--preview", "2", "--from", "3", "--style", str(style)) == 1  # style tidak ada
    assert r.calls == [] and not spawned
    style.write_text("stroke:\n  color: '#112233'\n", encoding="utf-8")
    assert run_cli("run", str(video), "--preview", "2", "--from", "3", "--style", str(style)) == 0
    assert r.stages == ["ingest", "stabilize", "vectorize", "stylize", "export"] and not spawned
    a = dict(stage_calls(r))
    assert "--limit" not in a["stabilize"]
    assert opt(a["vectorize"], "--limit") == "5" and "--from" not in a["vectorize"]
    for s in ("stylize", "export"):
        assert opt(a[s], "--from") == "3" and opt(a[s], "--limit") == "2"
    assert opt(a["stylize"], "--style") == str(style) and opt(a["export"], "--style") == str(style)      # T-404a: kertas dibaca export
    assert all("--style" not in a[s] for s in ("stabilize", "vectorize"))
    assert all("--restart" not in v for v in a.values())


def test_preview_default_from_is_zero(real):                            # noqa: F811
    video, _, r = real
    r.calls.clear()
    assert run_cli("run", str(video), "--preview", "3") == 0
    a = dict(stage_calls(r))
    assert opt(a["vectorize"], "--limit") == "3" and opt(a["stylize"], "--from") == "0"


@pytest.mark.parametrize("extra", [["--limit", "2"], ["--restart-from", "vectorize"], ["--adopt"], ["--qc-only"]])
def test_preview_forbidden_combinations(real, extra):                   # noqa: F811
    video, wd, r = real
    r.calls.clear()
    assert run_cli("run", str(video), "--preview", "2", *extra) == 1
    assert r.calls == []


def test_from_without_preview_and_bad_windows(real, capsys):            # noqa: F811
    video, wd, r = real
    r.calls.clear()
    for argv in (["--from", "1"], ["--preview", "0"], ["--preview", "2", "--from", "-1"],
                 ["--preview", "3", "--from", "4"], ["--preview", "7"]):      # klip 6 frame
        assert run_cli("run", str(video), *argv) == 1, argv
    assert r.calls == []
    assert "melewati klip" in capsys.readouterr().err
    with pytest.raises(SystemExit) as e:
        run_cli("run", str(video), "--preview", "x")
    assert e.value.code == 2


def test_preview_seg_missing_in_window_stops_before_any_stage(real, capsys):   # noqa: F811
    video, wd, r = real
    r.calls.clear()
    (wd / "seg" / "probs" / "frame_00003.npz").unlink()
    assert run_cli("run", str(video), "--preview", "2", "--from", "3") == 1
    err = capsys.readouterr().err
    assert r.calls == [] and "seg/" in err and f"segment {video} --limit 5" in err and f"run {video}" in err
    # frame di LUAR jendela hilang: cek jendela tidak menolak (stabilize yang menilai)
    assert run_cli("run", str(video), "--preview", "2", "--from", "0") == 0


def test_preview_depth_missing_in_window_stops_before_any_stage(real, capsys):  # noqa: F811
    video, wd, r = real
    r.calls.clear()
    (wd / "depth" / "frame_00001.npy").unlink()
    assert run_cli("run", str(video), "--preview", "2", "--from", "1") == 1
    err = capsys.readouterr().err
    assert r.calls == [] and "depth/" in err and f"depth {video} --limit 3" in err
    (wd / "depth" / "manifest.json").unlink()
    assert run_cli("run", str(video), "--preview", "2", "--from", "4") == 1
    assert "manifest.json" in capsys.readouterr().err and r.calls == []


def test_preview_seg_manifest_missing(real, capsys):                    # noqa: F811
    video, wd, r = real
    r.calls.clear()
    (wd / "seg" / "manifest.json").unlink()
    assert run_cli("run", str(video), "--preview", "2") == 1
    assert "segment" in capsys.readouterr().err and r.calls == []


def test_preview_seg_model_mismatch_names_both_models_and_command(real, capsys):   # noqa: F811
    video, wd, r = real
    r.calls.clear()
    assert run_cli("run", str(video), "--preview", "2", "--seg-model", "0.4b") == 1
    err = capsys.readouterr().err
    assert r.calls == [] and "0.8b" in err and "0.4b" in err and "--restart --yes" in err and "segment" in err


def test_preview_other_clip_identity_rejected(real, capsys):            # noqa: F811
    video, wd, r = real
    r.calls.clear()
    meta = json.loads((wd / "meta.json").read_text(encoding="utf-8"))
    (wd / "meta.json").write_text(json.dumps({**meta, "source_fps": 99}), encoding="utf-8")   # identitas klip berubah
    assert run_cli("run", str(video), "--preview", "2") == 1
    assert r.calls == []


def test_preview_other_video_in_workdir_rejected(real, tmp_path, capsys):   # noqa: F811
    video, wd, r = real
    r.calls.clear()
    meta = json.loads((wd / "meta.json").read_text(encoding="utf-8"))
    (wd / "meta.json").write_text(json.dumps({**meta, "source_path": str(tmp_path / "lain.mp4")}), encoding="utf-8")
    assert run_cli("run", str(video), "--preview", "2") == 1
    assert "LAIN" in capsys.readouterr().err and r.calls == []


def test_preview_stage_failure_exit_code_and_stop(real, capsys):        # noqa: F811
    video, _, r = real
    r.calls.clear()
    r.rc["vectorize"] = 1
    assert run_cli("run", str(video), "--preview", "2") == 1
    assert r.stages == ["ingest", "stabilize", "vectorize"] and "resume" in capsys.readouterr().err


def test_preview_ignores_main_target_preflight(real, tmp_path):         # noqa: F811
    """Pre-flight (c)/(e) target export utama dilewati seperti --limit: MP4 utama milik video LAIN tidak menghalangi."""
    video, wd, r = real
    out = tmp_path / "out"
    out.mkdir()
    (out / "clip.mp4").write_bytes(b"x")
    (out / "clip.export.json").write_text(json.dumps({"clip": {"source_path": "C:/lain/v.mp4"}}), encoding="utf-8")
    assert run_cli("run", str(video), "--restart-from", "export") == 1      # run penuh ditolak
    r.calls.clear()
    assert run_cli("run", str(video), "--preview", "2") == 0
    assert (out / "clip.mp4").read_bytes() == b"x"


def test_preview_summary_names_window_and_file(real, capsys):           # noqa: F811
    video, _, r = real
    capsys.readouterr()
    assert run_cli("run", str(video), "--preview", "2", "--from", "3") == 0
    out = capsys.readouterr().out
    assert "preview_3-4.mp4" in out and "Waktu per stage" in out and "[6] export" in out


def test_torch_not_imported_by_preview_check(real, tmp_path):           # noqa: F811
    """Cek seg / depth jendela memakai fungsi validasi stage tanpa meng-import torch (proses bersih)."""
    _, wd, _ = real
    code = ("import sys\nfrom pathlib import Path\nfrom rotoscope import cli\n"
            "cfg = cli.load_cfg(None)\n"
            f"cli.check_gpu_inputs(cfg, Path('v.mp4'), Path({str(wd)!r}), 1, 4)\n"
            "sys.exit(1 if 'torch' in sys.modules else 0)\n")
    assert subprocess.run([sys.executable, "-c", code], cwd=tmp_path).returncode == 0
    (wd / "depth" / "frame_00002.npy").unlink()                          # sanity: cek memang berjalan (jendela 1..3)
    assert subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True).returncode != 0


def test_clip_dir_helper_matches_cli(real):                              # noqa: F811
    video, wd, _ = real
    assert wd == clip_dir(Path(wd).parents[2]) and Path(cli.clip_work_dir(cli.load_cfg(None), video)).name == "clip"


# ── rantai nyata kecil (4 frame): stabilize / vectorize / stylize / export nyata, GPU tidak ada ──
@pytest.fixture
def chain(tmp_path, monkeypatch):
    """Klip 4 frame dengan seg + depth sintetis (tst.make_work); ingest palsu, 4 stage CPU lain NYATA."""
    import test_stabilize as tst
    from rotoscope import stabilize as stb
    from rotoscope import vectorize as vec
    from test_cli import Rec

    monkeypatch.chdir(tmp_path)
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x")
    work = tst.make_work(tmp_path / "stage")
    meta = json.loads((work / "meta.json").read_text(encoding="utf-8"))
    meta.update(source_path=str(video.resolve()), target_fps=24, has_audio=False)
    (work / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    import shutil
    dest = clip_dir(tmp_path)
    dest.parent.mkdir(parents=True)
    shutil.move(str(work), str(dest))
    # temporal (default sejak T-302) butuh frames/ (cut) + qc_report.json (bobot)
    import cv2
    import numpy as np
    (dest / "frames").mkdir()
    for i in range(tst.N_FRAMES):
        (dest / "frames" / f"frame_{i:05d}.png").write_bytes(cv2.imencode(".png", np.full((tst.H, tst.W, 3), 90, np.uint8))[1].tobytes())
    (dest / "qc_report.json").write_text(json.dumps({"stage": "segment_qc", "frames": [
        {"frame": f"frame_{i:05d}.png", "index": i, "fail_reasons": []} for i in range(tst.N_FRAMES)]}), encoding="utf-8")
    # manifest seg / depth dibuat ulang untuk meta.json baru (identitas klip)
    from rotoscope.stage_common import clip_identity
    for d in ("seg", "depth"):
        p = dest / d / "manifest.json"
        m = json.loads(p.read_text(encoding="utf-8"))
        p.write_text(json.dumps({**m, "clip": clip_identity(dest)}), encoding="utf-8")
    r = Rec()
    spawned: list = []
    monkeypatch.setattr(cli, "_popen", lambda *a, **k: spawned.append(a))
    monkeypatch.setattr(cli, "CPU_MAINS", {"ingest": r.cpu("ingest"), "stabilize": stb.main, "vectorize": vec.main,
                                           "stylize": sty.main, "export": ex.main})
    return video, dest, tmp_path, spawned


needs_ffmpeg = pytest.mark.skipif(te.pytestmark.args[0], reason="ffmpeg/ffprobe tidak ada di PATH")


def count(d: Path, pattern: str) -> int:
    return len(list(d.glob(pattern)))


@needs_ffmpeg
def test_chain_preview_then_stale_stable_runs_full_stabilize_and_prefix(chain, capsys):
    video, wd, tmp, spawned = chain
    conf = tmp / "c.yaml"
    assert run_cli("run", str(video), "--preview", "2", "--from", "1") == 0           # stable/ belum ada → stabilize penuh
    assert not spawned and count(wd / "stable" / "groups", "*.png") == 4
    assert count(wd / "contours", "frame_*.json") == 3                                # prefiks K+N = 3
    assert count(wd / "strokes", "frame_*.svg") == 2 and count(wd / "strokes", "frame_*.png") == 2
    assert (tmp / "out" / "clip.preview_1-2.mp4").is_file() and not (tmp / "out" / "clip.mp4").exists()
    assert not (tmp / "out" / "svg").exists() and not list((tmp / "out").glob("*.export.json"))
    out = capsys.readouterr().out
    assert "estimasi" in out                                                          # peringatan waktu stabilize penuh
    assert "rantai vectorize 3 frame belum valid, estimasi ±0.3 s" in out              # kasus dingin: 3 × 0,1 s
    assert run_cli("run", str(video), "--preview", "2", "--from", "1") == 0           # hangat: semua valid
    assert "rantai vectorize" not in capsys.readouterr().out
    # stable/ basi (config berubah): stabilize penuh → vectorize basi (prefiks) → stylize basi (hanya jendela)
    conf.write_text("stabilize:\n  island_min_px: 31\n", encoding="utf-8")
    style = tmp / "s.yaml"                                        # klip sintetis tanpa kontur → style juga diubah agar [5] basi
    style.write_text("stroke:\n  color: '#112233'\n", encoding="utf-8")
    capsys.readouterr()
    assert run_cli("run", str(video), "--preview", "2", "--from", "1", "--config", str(conf),
                   "--style", str(style)) == 0
    out = capsys.readouterr().out
    assert "[3] stabilize" in out and "4 diproses" in out and out.count("output [4] basi") == 1
    assert "output [5] basi" in out and "DIHAPUS seluruhnya" in out, out.encode("ascii", "replace").decode()
    assert count(wd / "contours", "frame_*.json") == 3 and count(wd / "strokes", "frame_*.png") == 2
    assert not spawned
    # export penuh sesudah preview: gagal keras (bukan MP4 dari folder separuh); stylize melanjutkan
    assert run_cli("export", str(video), "--config", str(conf)) == 1
    assert not (tmp / "out" / "clip.mp4").exists()


@needs_ffmpeg
def test_chain_stale_stable_with_seg_missing_outside_window_stops_in_stabilize(chain, capsys):
    video, wd, tmp, spawned = chain
    assert run_cli("run", str(video), "--preview", "2", "--from", "1") == 0
    conf = tmp / "c.yaml"
    conf.write_text("stabilize:\n  island_min_px: 31\n", encoding="utf-8")
    (wd / "seg" / "probs" / "frame_00000.npz").unlink()                               # di luar jendela 1..2
    (wd / "depth" / "frame_00003.npy").unlink()
    before = sorted(p.name for p in (wd / "stable" / "groups").glob("*.png"))
    capsys.readouterr()
    assert run_cli("run", str(video), "--preview", "2", "--from", "1", "--config", str(conf)) == 1
    err = capsys.readouterr().err
    assert "[3] stabilize" in err and "segment" in err and f"rotoscope segment {wd.parent.parent.parent / 'clip.mp4'}" in err
    assert not spawned and sorted(p.name for p in (wd / "stable" / "groups").glob("*.png")) == before


# ── data nyata (di-skip bila klip tidak ada): byte-identik dengan run penuh ──
REAL = Path("work/clips/test")


def _real_strokes_current() -> bool:
    """strokes/ klip nyata ada dan contract + algo_rev-nya = stylize sekarang (belum dihitung ulang oleh `run` → basi → dilewati;
    T-406 menaikkan ALGO_REV, jadi dilewati sampai strokes/ klip asli dihitung ulang di Tahap 4)."""
    import json
    m = REAL / "strokes" / "manifest.json"
    if not m.is_file():
        return False
    d = json.loads(m.read_text(encoding="utf-8"))
    if d.get("contract") != sty.CONTRACT or d.get("algo_rev") != sty.ALGO_REV:
        return False
    # T-305b: contours/ klip asli contract lama sampai dihitung ulang di Tahap 4 → [5] menolaknya → dilewati
    cm = REAL / "contours" / "manifest.json"
    return cm.is_file() and json.loads(cm.read_text(encoding="utf-8")).get("contract") in sty.SUPPORTED_CONTOURS_CONTRACTS


@pytest.mark.skipif(not _real_strokes_current(), reason="klip test tidak ada / strokes basi (contract lama)")
def test_real_window_matches_existing_strokes(tmp_path):
    """Frame jendela yang dihitung ulang (stylize --from 73 --limit 3) = frame strokes/ yang sudah ada di run penuh."""
    import hashlib
    import shutil
    work = tmp_path / "w"
    shutil.copytree(REAL, work, ignore=shutil.ignore_patterns("frames", "seg", "depth", "stable", "strokes"))
    shutil.copytree(REAL / "stable", work / "stable")
    (work / "strokes").mkdir()
    shutil.copy2(REAL / "strokes" / "manifest.json", work / "strokes" / "manifest.json")
    cfg = cli.load_cfg(None)
    from rotoscope.config import load_pipeline
    cfg = load_pipeline(None, overrides={"paths.work_dir": str(work), "paths.out_dir": str(tmp_path / "o")})
    run = sty.run_stylize(cfg, sty.load_style(resolve_style("rough-sketch")), "default", start=73, limit=3, log=quiet)
    assert run["processed"] == 3 and not run["stale"]
    for i in range(73, 76):
        for ext in ("svg", "png"):
            a = (work / "strokes" / f"frame_{i:05d}.{ext}").read_bytes()
            b = (REAL / "strokes" / f"frame_{i:05d}.{ext}").read_bytes()
            assert hashlib.sha256(a).digest() == hashlib.sha256(b).digest()
