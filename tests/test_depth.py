"""Test depth.py (T-105): backend palsu, tanpa GPU/model. Test GPU nyata: ROTOSCOPE_GPU_TESTS=1."""

from __future__ import annotations

import dataclasses
import io
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from rotoscope import depth as dep
from rotoscope import stage_common
from rotoscope.config import load_pipeline
from rotoscope.depth import BackendOOM, DepthError, DepthOOMError, ModelInfo, run_depth

H, W = 24, 16
N_FRAMES = 6
REV = "c" * 40
PINNED = "5426e4f0f36572d16453bbda7a8389317b1bef99"
MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    # run_depth men-set HF_HUB_OFFLINE; monkeypatch mengembalikan nilai asli setelah test.
    monkeypatch.setenv(stage_common.HF_OFFLINE_ENV, "1")
    monkeypatch.setattr(stage_common, "REPLACE_DELAY_S", 0)


def make_work(tmp_path: Path, n: int = N_FRAMES, h: int = H, w: int = W) -> Path:
    """work/ sintetis: frame ke-i = kotak merah (subjek) bergeser 1 px per frame."""
    work = tmp_path / "work"
    frames = work / "frames"
    frames.mkdir(parents=True)
    for i in range(n):
        bgr = np.zeros((h, w, 3), np.uint8)
        bgr[4:h - 4, 3 + i % 3:w - 3, 2] = 255
        cv2.imwrite(str(frames / f"frame_{i:05d}.png"), bgr)
    meta = {"source_path": "C:/clips/a.mp4", "frame_count": n, "working_width": w, "working_height": h,
            "frame_index_start": 0}
    (work / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return work


def make_cfg(work: Path, **overrides):
    ov = {"paths.work_dir": str(work), "depth.revision": REV}
    ov.update(overrides)
    return load_pipeline(overrides=ov)


PROCESSOR = {"class": "DPTImageProcessor", "size": {"height": 518, "width": 518}, "keep_aspect_ratio": True,
             "ensure_multiple_of": 14, "do_pad": False, "resample": 3}


class FakeBackend:
    """Disparity sintetis: subjek (piksel merah) dekat = 2.0 + gradien kecil, background 0.5."""

    def __init__(self, cfg, oom_at=None, bad_at=None, bad_value=np.nan, input_size=(924, 518)):
        d = cfg.depth
        self.info_kw = {"model_id": d.model_id, "revision": dep.resolve_revision(cfg), "license": "apache-2.0",
                        "precision": d.precision, "processor": dict(PROCESSOR),
                        "input_size": {"height": input_size[0], "width": input_size[1]}}
        self.oom_at, self.bad_at, self.bad_value = oom_at, bad_at, bad_value
        self.calls = 0
        self.opened = self.closed = 0

    def describe(self, height, width):
        return ModelInfo(**self.info_kw)

    def open(self):
        self.opened += 1
        return {"vram_free_before_load_mib": 3500.0}

    def infer(self, rgb):
        if self.oom_at is not None and self.calls == self.oom_at:
            raise BackendOOM({"free_mib": 12.0, "total_mib": 4096.0, "reserved_mib": 424.0})
        self.calls += 1
        h, w = rgb.shape[:2]
        d = np.full((h, w), 0.5, np.float32)
        red = rgb[..., 0] > 128
        d[red] = 2.0 + np.linspace(0, 1, h, dtype=np.float32)[:, None].repeat(w, 1)[red]
        if self.bad_at is not None and self.calls - 1 == self.bad_at:
            d[0, 0] = self.bad_value
            d[1, 1] = self.bad_value
        return d, {"peak_reserved_mib": 424.0, "peak_allocated_mib": 291.0}

    def close(self):
        self.closed += 1


def factory(**kw):
    made = []

    def f(cfg):
        b = FakeBackend(cfg, **kw)
        made.append(b)
        return b
    f.made = made
    return f


def quiet(_msg):
    pass


def frame_recs(work: Path) -> list[dict]:
    return [r for r in stage_common.read_jsonl(work / "depth" / "frames.jsonl") if r["event"] == "frame"]


# ── Output sesuai kontrak ──────────────────────────
def test_full_run_outputs_match_contract(tmp_path):
    work = make_work(tmp_path)
    run = run_depth(make_cfg(work), backend_factory=factory(), log=quiet)
    assert run["processed"] == N_FRAMES and run["skipped"] == 0 and run["nonfinite_frames"] == 0
    for i in range(N_FRAMES):
        d = np.load(work / "depth" / f"frame_{i:05d}.npy", allow_pickle=False)
        assert d.dtype == np.float16 and d.shape == (H, W) and np.isfinite(d).all()
        assert d.min() == np.float16(0.5) and d.max() > 2.0
    man = json.loads((work / "depth" / "manifest.json").read_text(encoding="utf-8"))
    assert man["stage"] == "depth" and man["model_id"] == MODEL_ID and man["revision"] == REV
    assert man["license"] == "apache-2.0" and man["precision"] == "fp32"
    assert man["processor"] == PROCESSOR and man["input_size"] == {"height": 924, "width": 518}
    assert man["frame_size"] == {"width": W, "height": H} and "disparity relatif" in man["output"]["kind"]
    assert set(dep.MANIFEST_MATCH_KEYS) <= set(man)
    recs = stage_common.read_jsonl(work / "depth" / "frames.jsonl")
    frames = [r for r in recs if r["event"] == "frame"]
    assert len(frames) == N_FRAMES
    assert {"total_s", "infer_s", "peak_reserved_mib", "peak_allocated_mib", "finite", "n_nonfinite",
            "disparity"} <= set(frames[0])
    assert frames[0]["finite"] is True and frames[0]["n_nonfinite"] == 0
    d0 = np.load(work / "depth" / "frame_00000.npy").astype(np.float32)
    assert frames[0]["disparity"] == {"min": 0.5, "median": round(float(np.median(d0)), 5),
                                      "max": round(float(d0.max()), 5)}
    assert [r["event"] for r in recs if r["event"] != "frame"] == ["run_start", "run_end"]
    assert not list((work / "depth").rglob("*.tmp"))


def test_disparity_stats_from_stored_values():
    y, finite, n = dep.to_output(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 100000.0]], np.float32))
    assert y.dtype == np.float16 and not finite and n == 1 and y[1, 2] == 0
    assert dep.disparity_stats(y) == {"min": 0.0, "median": 2.5, "max": 5.0}
    y, finite, n = dep.to_output(np.array([[0.1, 0.2]], np.float32))
    assert finite and n == 0
    assert dep.disparity_stats(y)["max"] == round(float(np.float16(0.2)), 5)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, 1e6])  # 1e6 meluap di float16 → inf
def test_nonfinite_replaced_logged_and_not_reprocessed(tmp_path, bad):
    work = make_work(tmp_path)
    cfg = make_cfg(work)
    run = run_depth(cfg, backend_factory=factory(bad_at=2, bad_value=bad), log=quiet)
    assert run["nonfinite_frames"] == 1
    d = np.load(work / "depth" / "frame_00002.npy")
    assert np.isfinite(d).all() and d[0, 0] == 0 and d[1, 1] == 0
    rec = {r["frame"]: r for r in frame_recs(work)}["frame_00002.png"]
    assert rec["finite"] is False and rec["n_nonfinite"] == 2
    f = factory()
    run2 = run_depth(cfg, backend_factory=f, log=quiet)
    assert run2["processed"] == 0 and run2["skipped"] == N_FRAMES and f.made[0].opened == 0


# ── Resume ─────────────────────────────────────────
def test_resume_skips_valid_frames(tmp_path):
    work = make_work(tmp_path)
    cfg = make_cfg(work)
    run_depth(cfg, backend_factory=factory(), log=quiet)
    f = factory()
    run = run_depth(cfg, backend_factory=f, log=quiet)
    assert run["processed"] == 0 and run["skipped"] == N_FRAMES
    assert f.made[0].opened == 0  # semua valid → GPU tidak disentuh


def test_limit_then_full_run_continues(tmp_path):
    work = make_work(tmp_path)
    cfg = make_cfg(work)
    run = run_depth(cfg, limit=2, backend_factory=factory(), log=quiet)
    assert run["processed"] == 2 and run["selected"] == 2
    run = run_depth(cfg, backend_factory=factory(), log=quiet)
    assert run["skipped"] == 2 and run["processed"] == N_FRAMES - 2
    assert run["frames"][0]["frame"] == "frame_00002.png"


@pytest.mark.parametrize("damage", ["truncate", "shape", "dtype", "nan", "delete"])
def test_damaged_frame_reprocessed(tmp_path, damage):
    work = make_work(tmp_path)
    cfg = make_cfg(work)
    run_depth(cfg, backend_factory=factory(), log=quiet)
    p = work / "depth" / "frame_00003.npy"
    if damage == "truncate":
        p.write_bytes(p.read_bytes()[:60])
    elif damage == "shape":
        np.save(p, np.zeros((H, W + 1), np.float16))
    elif damage == "dtype":
        np.save(p, np.zeros((H, W), np.float32))
    elif damage == "nan":
        a = np.load(p)
        a[0, 0] = np.nan
        np.save(p, a)
    else:
        p.unlink()
    assert not dep.depth_valid(p, H, W)
    run = run_depth(cfg, backend_factory=factory(), log=quiet)
    assert run["processed"] == 1 and run["frames"][0]["frame"] == "frame_00003.png"
    assert dep.depth_valid(p, H, W)


def test_tmp_file_is_not_valid_output_and_cleaned(tmp_path):
    work = make_work(tmp_path)
    cfg = make_cfg(work)
    run_depth(cfg, limit=1, backend_factory=factory(), log=quiet)
    tmp = work / "depth" / "frame_00001.npy.tmp"
    buf = io.BytesIO()
    np.save(buf, np.zeros((H, W), np.float16))
    tmp.write_bytes(buf.getvalue())
    run = run_depth(cfg, backend_factory=factory(), log=quiet)
    assert run["processed"] == N_FRAMES - 1 and not tmp.exists()


# ── Manifest ───────────────────────────────────────
@pytest.mark.parametrize("field,value", [
    ("revision", "d" * 40), ("license", "mit"), ("precision", "fp16"),
    ("processor", {**PROCESSOR, "ensure_multiple_of": 16}), ("input_size", {"height": 518, "width": 518}),
])
def test_manifest_mismatch_rejected(tmp_path, field, value):
    work = make_work(tmp_path)
    run_depth(make_cfg(work), limit=1, backend_factory=factory(), log=quiet)

    def changed(cfg):
        b = FakeBackend(cfg)
        b.info_kw[field] = value
        return b
    with pytest.raises(DepthError, match=rf"(?s){field}:.*--restart"):
        run_depth(make_cfg(work), backend_factory=changed, log=quiet)


def test_manifest_mismatch_output_and_frame_size(tmp_path):
    work = make_work(tmp_path)
    run_depth(make_cfg(work), limit=1, backend_factory=factory(), log=quiet)
    man_path = work / "depth" / "manifest.json"
    man = json.loads(man_path.read_text(encoding="utf-8"))
    man["output"]["dtype"] = "float32"
    man_path.write_text(json.dumps(man), encoding="utf-8")
    with pytest.raises(DepthError, match="output:"):
        run_depth(make_cfg(work), backend_factory=factory(), log=quiet)
    man["output"] = dict(dep.OUTPUT_INFO)
    man["frame_size"] = {"width": W, "height": H + 2}
    man_path.write_text(json.dumps(man), encoding="utf-8")
    with pytest.raises(DepthError, match="frame_size:"):
        run_depth(make_cfg(work), backend_factory=factory(), log=quiet)


def test_outputs_without_manifest_rejected(tmp_path):
    work = make_work(tmp_path)
    run_depth(make_cfg(work), limit=1, backend_factory=factory(), log=quiet)
    (work / "depth" / "manifest.json").unlink()
    with pytest.raises(DepthError, match="tanpa manifest.json"):
        run_depth(make_cfg(work), backend_factory=factory(), log=quiet)


def test_restart_removes_only_depth_outputs(tmp_path):
    work = make_work(tmp_path)
    (work / "seg").mkdir()
    (work / "seg" / "manifest.json").write_text("{}", encoding="utf-8")
    run_depth(make_cfg(work), backend_factory=factory(), log=quiet)
    run = run_depth(make_cfg(work), restart=True, backend_factory=factory(), log=quiet)
    assert run["processed"] == N_FRAMES and run["skipped"] == 0
    assert (work / "seg" / "manifest.json").is_file()  # output [2] tidak disentuh


# ── Lisensi + varian Small ─────────────────────────
@pytest.mark.parametrize("model_id", ["depth-anything/Depth-Anything-V2-Base-hf",
                                      "depth-anything/Depth-Anything-V2-Large-hf"])
def test_non_small_rejected_in_backend_even_if_config_bypassed(tmp_path, model_id):
    cfg = make_cfg(make_work(tmp_path))
    bad = dataclasses.replace(cfg, depth=dataclasses.replace(cfg.depth, model_id=model_id))
    with pytest.raises(DepthError, match="CC-BY-NC"):
        dep.TorchDepthBackend(bad)
    f = factory()
    with pytest.raises(DepthError, match="CC-BY-NC"):
        run_depth(bad, backend_factory=f, log=quiet)
    assert f.made == []


def test_backbone_hidden_size_must_be_small():
    dep.require_small_backbone(384, MODEL_ID)
    for hs in (768, 1024, None):
        with pytest.raises(DepthError, match=r"(?s)hidden_size.*384.*CC-BY-NC"):
            dep.require_small_backbone(hs, MODEL_ID)


def test_card_license_front_matter():
    assert dep.card_license("---\nlicense: apache-2.0\ntags:\n- depth\n---\n# DA") == "apache-2.0"
    assert dep.card_license("﻿---\nlicense: Apache-2.0\n---\n") == "apache-2.0"
    assert dep.card_license("# tanpa front-matter\nlicense: apache-2.0") is None
    assert dep.card_license("---\ntags: [a]\n---\n") is None
    assert dep.card_license("---\nlicense: [\n") is None  # front-matter tidak ditutup
    assert dep.require_license("---\nlicense: apache-2.0\n---\n", MODEL_ID) == "apache-2.0"
    with pytest.raises(DepthError, match=r"cc-by-nc-4\.0.*CC-BY-NC"):
        dep.require_license("---\nlicense: cc-by-nc-4.0\n---\n", MODEL_ID)


def _fake_snapshot(cache: Path, files: dict[str, str]) -> None:
    snap = cache / "models--depth-anything--Depth-Anything-V2-Small-hf" / "snapshots" / REV
    snap.mkdir(parents=True)
    for name, text in files.items():
        (snap / name).write_text(text, encoding="utf-8")


def test_model_card_required_in_cache(tmp_path):
    cache = tmp_path / "hf_cache"
    _fake_snapshot(cache, {"config.json": "{}", "preprocessor_config.json": "{}", "model.safetensors": "x"})
    b = dep.TorchDepthBackend(make_cfg(make_work(tmp_path)), cache_dir=cache)
    with pytest.raises(DepthError, match=r"(?s)hilang: README\.md.*python -m rotoscope download"):
        b.describe(H, W)


def test_wrong_license_in_cache_stops(tmp_path):
    cache = tmp_path / "hf_cache"
    _fake_snapshot(cache, {"config.json": "{}", "preprocessor_config.json": "{}", "model.safetensors": "x",
                           "README.md": "---\nlicense: cc-by-nc-4.0\n---\n"})
    b = dep.TorchDepthBackend(make_cfg(make_work(tmp_path)), cache_dir=cache)
    with pytest.raises(DepthError, match="CC-BY-NC"):
        b.describe(H, W)


# ── Revision + cache ───────────────────────────────
def test_default_revision_is_pinned_hash():
    assert load_pipeline().depth.revision == PINNED
    assert dep.resolve_revision(load_pipeline()) == PINNED


def test_revision_null_stops(tmp_path):
    f = factory()
    with pytest.raises(DepthError, match=r"(?s)depth\.revision = null.*rotoscope download"):
        run_depth(make_cfg(make_work(tmp_path), **{"depth.revision": None}), backend_factory=f, log=quiet)
    assert f.made == []


@pytest.mark.parametrize("rev", ["main", "5426e4f0", "G" * 40, "A" * 40])
def test_revision_must_be_commit_hash(tmp_path, rev):
    with pytest.raises(DepthError, match="40 karakter hex"):
        dep.resolve_revision(make_cfg(make_work(tmp_path), **{"depth.revision": rev}))


def test_revision_not_in_cache_gives_download_command(tmp_path):
    b = dep.TorchDepthBackend(make_cfg(make_work(tmp_path)), cache_dir=tmp_path / "hf_cache")
    with pytest.raises(DepthError, match=r"(?s)tidak ada di cache HF.*python -m rotoscope download"):
        b.describe(H, W)


def test_download_refuses_when_offline(tmp_path):
    with pytest.raises(DepthError, match="HF_HUB_OFFLINE"):
        dep.download(make_cfg(make_work(tmp_path)), log=quiet)


def test_missing_frames_and_bad_limit(tmp_path):
    work = make_work(tmp_path)
    (work / "frames" / "frame_00004.png").unlink()
    with pytest.raises(DepthError, match=r"1 frame hilang.*frame_00004"):
        run_depth(make_cfg(work), backend_factory=factory(), log=quiet)
    run_depth(make_cfg(work), limit=3, backend_factory=factory(), log=quiet)  # frame hilang di luar limit
    with pytest.raises(DepthError, match="--limit"):
        run_depth(make_cfg(work), limit=0, backend_factory=factory(), log=quiet)


# ── VRAM / CUDA / OOM ──────────────────────────────
def _torch_backend(tmp_path, monkeypatch, free_mib, cuda=True):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a: (int(free_mib * dep.MIB), 4096 * dep.MIB))
    b = dep.TorchDepthBackend(make_cfg(make_work(tmp_path)))

    def no_load():
        raise AssertionError("model tidak boleh dimuat")
    monkeypatch.setattr(b, "_load_model", no_load)
    return b


def test_insufficient_vram_stops_before_load(tmp_path, monkeypatch):
    b = _torch_backend(tmp_path, monkeypatch, free_mib=400)
    with pytest.raises(DepthError, match=r"VRAM bebas 400 MiB < dibutuhkan 500"):
        b.open()


def test_cuda_unavailable_clear_error(tmp_path, monkeypatch):
    b = _torch_backend(tmp_path, monkeypatch, free_mib=4000, cuda=False)
    with pytest.raises(DepthError, match="CUDA tidak tersedia"):
        b.open()


def test_oom_mid_run_logs_and_exits_3(tmp_path, capsys):
    work = make_work(tmp_path)
    f = factory(oom_at=3)
    with pytest.raises(DepthOOMError, match="frame_00003"):
        run_depth(make_cfg(work), backend_factory=f, log=quiet)
    assert f.made[0].closed == 1
    assert dep.depth_valid(work / "depth" / "frame_00002.npy", H, W)
    assert not (work / "depth" / "frame_00003.npy").exists()
    oom = [r for r in stage_common.read_jsonl(work / "depth" / "frames.jsonl") if r["event"] == "oom"]
    assert oom == [{**oom[0], "frame": "frame_00003.png", "index": 3, "free_mib": 12.0, "reserved_mib": 424.0}]
    assert main_with(work, factory(oom_at=0)) == dep.EXIT_OOM and "OOM" in capsys.readouterr().err


# ── Entry point ────────────────────────────────────
def main_with(work: Path, backend_factory, *extra: str) -> int:
    cfg_path = work.parent / "c.yaml"
    cfg_path.write_text(f'paths:\n  work_dir: "{work.as_posix()}"\ndepth:\n  revision: "{REV}"\n', encoding="utf-8")
    return dep.main(["--config", str(cfg_path), *extra], backend_factory=backend_factory)


def test_main_exit_codes(tmp_path, capsys):
    work = make_work(tmp_path)
    assert main_with(work, factory(), "--limit", "2") == dep.EXIT_OK
    assert main_with(work, factory()) == dep.EXIT_OK
    assert "NaN/inf 0 frame" in capsys.readouterr().out

    def fp16(cfg):
        b = FakeBackend(cfg)
        b.info_kw["precision"] = "fp16"
        return b
    assert main_with(work, fp16) == dep.EXIT_PRECONDITION
    assert "--restart" in capsys.readouterr().err
    assert main_with(work, fp16, "--restart") == dep.EXIT_OK


def test_main_survives_cp1252_stdout(tmp_path, monkeypatch):
    work = make_work(tmp_path)
    out = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
    err = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
    monkeypatch.setattr(dep.sys, "stdout", out)
    monkeypatch.setattr(dep.sys, "stderr", err)
    assert main_with(work, factory()) == dep.EXIT_OK  # log berisi "×" dan "–"
    (work / "depth" / "manifest.json").unlink()
    assert main_with(work, factory()) == dep.EXIT_PRECONDITION  # pesan berisi "—"
    out.flush()
    assert b"selesai" in out.buffer.getvalue()


def test_main_rejects_flag_combinations(tmp_path):
    work = make_work(tmp_path)
    with pytest.raises(SystemExit):
        main_with(work, factory(), "--download", "--limit", "3")
    with pytest.raises(SystemExit):
        main_with(work, factory(), "--download", "--restart")


# ── GPU nyata (opsional) ───────────────────────────
@pytest.mark.skipif(os.environ.get("ROTOSCOPE_GPU_TESTS") != "1", reason="set ROTOSCOPE_GPU_TESTS=1 (butuh GPU + cache)")
def test_real_backend_one_frame():
    b = dep.TorchDepthBackend(load_pipeline())
    info = b.describe(854, 480)
    assert info.input_size == {"height": 924, "width": 518} and info.license == "apache-2.0"
    b.open()
    try:
        d, stats = b.infer(np.zeros((854, 480, 3), np.uint8))
    finally:
        b.close()
    assert d.shape == (854, 480) and d.dtype == np.float32 and np.isfinite(d).all()
    assert stats["peak_reserved_mib"] > 0
