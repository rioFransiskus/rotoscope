"""Test segment.py (T-102b): backend palsu, tanpa GPU/model. Test GPU nyata: ROTOSCOPE_GPU_TESTS=1."""

from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from rotoscope import segment as seg
from rotoscope.config import load_class_names, load_pipeline
from rotoscope.segment import (
    BackendOOM, ModelInfo, SegmentError, SegmentOOMError, area_vs_median, argmax_mismatch_pct, compute_qc,
    logits_to_outputs, run_qc, run_segment, shifted_window,
)

C = 29
H, W = 24, 16
N_FRAMES = 6
REV = {"0.8b": "a" * 40, "0.4b": "b" * 40}


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    # run_segment men-set HF_HUB_OFFLINE; monkeypatch mengembalikan nilai asli setelah test.
    monkeypatch.setenv(seg.HF_OFFLINE_ENV, "1")
    monkeypatch.setattr(seg, "REPLACE_DELAY_S", 0)


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
    ov = {"paths.work_dir": str(work), "segment.revision": dict(REV)}
    ov.update(overrides)
    return load_pipeline(overrides=ov)


class FakeBackend:
    """Logits sintetis: kelas 1 (atas) / 2 (bawah) di piksel merah, background di luar."""

    def __init__(self, cfg, num_labels=C, oom_at=None, nan_at=None, logits_fn=None):
        s = cfg.segment
        self.info = ModelInfo(model=s.model, model_id=s.model_ids[s.model], revision=seg.resolve_revision(cfg),
                              precision=s.precision, processor={"size": {"height": 1024, "width": 768},
                                                                "do_pad": False}, num_labels=num_labels)
        self.oom_at, self.nan_at, self.logits_fn = oom_at, nan_at, logits_fn
        self.calls = 0
        self.opened = self.closed = 0

    def describe(self):
        return self.info

    def open(self):
        self.opened += 1
        return {"vram_free_before_load_mib": 3500.0}

    def infer(self, rgb):
        if self.oom_at is not None and self.calls == self.oom_at:
            raise BackendOOM({"free_mib": 12.0, "total_mib": 4096.0, "reserved_mib": 3276.0})
        self.calls += 1
        if self.logits_fn is not None:
            return self.logits_fn(rgb), {"peak_reserved_mib": 1.0, "peak_allocated_mib": 1.0}
        h, w = rgb.shape[:2]
        logits = np.zeros((self.info.num_labels, h, w), np.float32)
        red = rgb[..., 0] > 128
        logits[0] = 5.0
        logits[0][red] = 0.0
        logits[1][red] = 6.0
        logits[2][red] = 3.0
        logits[2][h // 2:][red[h // 2:]] = 7.0
        if self.nan_at is not None and self.calls - 1 == self.nan_at:
            logits[3, 0, 0] = np.nan
        return logits, {"peak_reserved_mib": 1.0, "peak_allocated_mib": 1.0}

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


# ── Output sesuai kontrak ──────────────────────────
def test_full_run_outputs_match_contract(tmp_path):
    work = make_work(tmp_path)
    run = run_segment(make_cfg(work), backend_factory=factory(), log=quiet)
    assert run["processed"] == N_FRAMES and run["skipped"] == 0
    for i in range(N_FRAMES):
        cm = cv2.imread(str(work / "seg" / "classmap" / f"frame_{i:05d}.png"), cv2.IMREAD_UNCHANGED)
        assert cm.dtype == np.uint8 and cm.shape == (H, W) and cm.max() < C and set(np.unique(cm)) == {0, 1, 2}
        with np.load(work / "seg" / "probs" / f"frame_{i:05d}.npz") as z:
            assert list(z.keys()) == ["probs"]
            p = z["probs"]
        assert p.dtype == np.uint8 and p.shape == (C, H, W)
    man = json.loads((work / "seg" / "manifest.json").read_text(encoding="utf-8"))
    assert man["model"] == "0.8b" and man["revision"] == REV["0.8b"] and man["num_labels"] == C
    assert man["classes"] == list(load_class_names()) and man["frame_size"] == {"width": W, "height": H}
    assert man["processor"] == {"size": {"height": 1024, "width": 768}, "do_pad": False}
    recs = seg.read_jsonl(work / "seg" / "frames.jsonl")
    frames = [r for r in recs if r["event"] == "frame"]
    assert len(frames) == N_FRAMES
    assert {"total_s", "peak_reserved_mib", "peak_allocated_mib", "finite", "argmax_mismatch_pct"} <= set(frames[0])
    assert [r["event"] for r in recs if r["event"] != "frame"] == ["run_start", "run_end"]
    report = json.loads((work / "qc_report.json").read_text(encoding="utf-8"))
    assert report["summary"]["n_frames"] == N_FRAMES and len(report["frames"]) == N_FRAMES
    assert run["qc"] == report["summary"]
    assert not list((work / "seg").rglob("*.tmp"))


def test_logits_to_outputs_softmax_and_rounding():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(C, 5, 4)).astype(np.float32) * 4
    probs, cm, finite = logits_to_outputs(logits)
    e = np.exp(logits.astype(np.float64) - logits.max(0))
    p = e / e.sum(0)
    assert finite and probs.dtype == np.uint8 and cm.dtype == np.uint8
    assert np.array_equal(probs, np.rint(p * 255).astype(np.uint8))
    assert np.array_equal(cm, logits.argmax(0))
    assert logits.dtype == np.float32 and np.isfinite(logits).all()  # input tidak diubah


def test_argmax_mismatch_recorded_not_fatal(tmp_path):
    # Kelas 0 p = 0.3100, kelas 5 p = 0.3101 → keduanya round(p × 255) = 79; argmax(probs) = 0, classmap = 5.
    p = np.full(C, (1 - 0.3100 - 0.3101) / (C - 2))
    p[0], p[5] = 0.3100, 0.3101

    def near_tie(rgb):
        return np.broadcast_to(np.log(p)[:, None, None], (C,) + rgb.shape[:2]).astype(np.float32)

    probs, cm, _ = logits_to_outputs(near_tie(np.zeros((2, 2, 3))))
    assert (cm == 5).all() and argmax_mismatch_pct(probs, cm) == 100.0
    work = make_work(tmp_path, n=2)
    run_segment(make_cfg(work), backend_factory=factory(logits_fn=near_tie), log=quiet)
    frames = [r for r in seg.read_jsonl(work / "seg" / "frames.jsonl") if r["event"] == "frame"]
    assert all(r["argmax_mismatch_pct"] == 100.0 for r in frames)
    report = json.loads((work / "qc_report.json").read_text(encoding="utf-8"))
    assert report["summary"]["stats"]["argmax_mismatch_pct"]["max"] == 100.0


def test_nan_logits_written_and_fail_qc(tmp_path):
    work = make_work(tmp_path)
    run_segment(make_cfg(work), backend_factory=factory(nan_at=2), log=quiet)
    report = json.loads((work / "qc_report.json").read_text(encoding="utf-8"))
    rows = {r["index"]: r for r in report["frames"]}
    assert rows[2]["finite"] is False and "finite" in rows[2]["fail_reasons"]
    assert rows[1]["finite"] is True
    assert report["summary"]["finite_false"] == 1


# ── Resume ─────────────────────────────────────────
def test_resume_skips_valid_frames(tmp_path):
    work = make_work(tmp_path)
    cfg = make_cfg(work)
    run_segment(cfg, backend_factory=factory(), log=quiet)
    f = factory()
    run = run_segment(cfg, backend_factory=f, log=quiet)
    assert run["processed"] == 0 and run["skipped"] == N_FRAMES
    assert f.made[0].calls == 0 and f.made[0].opened == 0  # GPU tidak disentuh
    assert run["qc"] is not None


def test_limit_then_full_run_continues(tmp_path):
    work = make_work(tmp_path)
    cfg = make_cfg(work)
    run = run_segment(cfg, limit=2, backend_factory=factory(), log=quiet)
    assert run["processed"] == 2 and run["qc"] is None and not (work / "qc_report.json").exists()
    f = factory()
    run = run_segment(cfg, backend_factory=f, log=quiet)
    assert run["processed"] == N_FRAMES - 2 and run["skipped"] == 2
    idx = [r["index"] for r in run["frames"]]
    assert idx == list(range(2, N_FRAMES))
    assert (work / "qc_report.json").is_file()


@pytest.mark.parametrize("damage", ["truncate_png", "truncate_npz", "wrong_shape", "class_out_of_range",
                                    "missing_npz"])
def test_damaged_frame_reprocessed(tmp_path, damage):
    work = make_work(tmp_path)
    cfg = make_cfg(work)
    run_segment(cfg, backend_factory=factory(), log=quiet)
    png = work / "seg" / "classmap" / "frame_00003.png"
    npz = work / "seg" / "probs" / "frame_00003.npz"
    if damage == "truncate_png":
        png.write_bytes(png.read_bytes()[:40])
    elif damage == "truncate_npz":
        npz.write_bytes(npz.read_bytes()[:100])
    elif damage == "wrong_shape":
        np.savez_compressed(npz, probs=np.zeros((C, H, W + 1), np.uint8))
    elif damage == "class_out_of_range":
        cv2.imwrite(str(png), np.full((H, W), 29, np.uint8))
    else:
        npz.unlink()
    run = run_segment(cfg, backend_factory=factory(), log=quiet)
    assert [r["index"] for r in run["frames"]] == [3]
    assert seg.classmap_valid(png, H, W, C) and seg.probs_valid(npz, H, W, C)


def test_tmp_file_is_not_valid_output_and_cleaned(tmp_path):
    work = make_work(tmp_path)
    cfg = make_cfg(work)
    run_segment(cfg, limit=1, backend_factory=factory(), log=quiet)
    png = work / "seg" / "classmap" / "frame_00001.png"
    npz = work / "seg" / "probs" / "frame_00001.npz"
    # Proses mati di tengah tulis: hanya file .tmp yang ada (lengkap sekalipun).
    ok, buf = cv2.imencode(".png", np.zeros((H, W), np.uint8))
    Path(str(png) + ".tmp").write_bytes(buf.tobytes())
    assert not seg.classmap_valid(png, H, W, C)
    run = run_segment(cfg, limit=2, backend_factory=factory(), log=quiet)
    assert [r["index"] for r in run["frames"]] == [1]
    assert png.is_file() and npz.is_file() and not list((work / "seg").rglob("*.tmp"))


@pytest.mark.parametrize("overrides, field", [
    ({"segment.model": "0.4b"}, "model"),
    ({"segment.revision": {"0.8b": "c" * 40}}, "revision"),
    ({"segment.precision": "fp32"}, "precision"),
])
def test_manifest_mismatch_rejected(tmp_path, overrides, field):
    work = make_work(tmp_path)
    run_segment(make_cfg(work), limit=2, backend_factory=factory(), log=quiet)
    with pytest.raises(SegmentError, match=rf"(?s){field}:.*--restart"):
        run_segment(make_cfg(work, **overrides), backend_factory=factory(), log=quiet)


def test_manifest_mismatch_num_labels_and_frame_size(tmp_path):
    work = make_work(tmp_path)
    cfg = make_cfg(work)
    run_segment(cfg, limit=1, backend_factory=factory(), log=quiet)
    man_path = work / "seg" / "manifest.json"
    man = json.loads(man_path.read_text(encoding="utf-8"))
    man["frame_size"] = {"width": W, "height": H + 2}
    man["processor"]["do_pad"] = True
    man_path.write_text(json.dumps(man), encoding="utf-8")
    with pytest.raises(SegmentError, match=r"(?s)processor:.*frame_size:"):
        run_segment(cfg, backend_factory=factory(), log=quiet)


def test_outputs_without_manifest_rejected(tmp_path):
    work = make_work(tmp_path)
    cfg = make_cfg(work)
    run_segment(cfg, limit=1, backend_factory=factory(), log=quiet)
    (work / "seg" / "manifest.json").unlink()
    with pytest.raises(SegmentError, match="--restart"):
        run_segment(cfg, backend_factory=factory(), log=quiet)


def test_restart_removes_old_outputs(tmp_path):
    work = make_work(tmp_path)
    run_segment(make_cfg(work), backend_factory=factory(), log=quiet)
    (work / "seg" / "classmap" / "frame_99999.png").write_bytes(b"sisa lama")
    f = factory()
    run = run_segment(make_cfg(work, **{"segment.model": "0.4b"}), restart=True, backend_factory=f, log=quiet)
    assert run["processed"] == N_FRAMES and f.made[0].calls == N_FRAMES
    assert not (work / "seg" / "classmap" / "frame_99999.png").exists()
    man = json.loads((work / "seg" / "manifest.json").read_text(encoding="utf-8"))
    assert man["model"] == "0.4b" and man["revision"] == REV["0.4b"]
    recs = seg.read_jsonl(work / "seg" / "frames.jsonl")
    assert sum(r["event"] == "frame" for r in recs) == N_FRAMES  # log lama ikut terhapus


# ── Tulis atomik + retry Windows ───────────────────
def test_replace_retries_on_permission_error(tmp_path, monkeypatch):
    real = os.replace
    calls = []

    def flaky(src, dst):
        calls.append(dst)
        if len(calls) == 1:
            raise PermissionError(32, "file sedang dipakai proses lain")
        return real(src, dst)

    monkeypatch.setattr(seg.os, "replace", flaky)
    target = tmp_path / "x.json"
    seg.write_json_atomic(target, {"a": 1})
    assert len(calls) == 2 and json.loads(target.read_text(encoding="utf-8")) == {"a": 1}
    assert not (tmp_path / "x.json.tmp").exists()


def test_replace_gives_clear_error_after_retries(tmp_path, monkeypatch):
    def locked(src, dst):
        raise PermissionError(5, "Access is denied")

    monkeypatch.setattr(seg.os, "replace", locked)
    with pytest.raises(SegmentError, match=rf"{seg.REPLACE_RETRIES} percobaan.*antivirus"):
        seg.write_json_atomic(tmp_path / "x.json", {"a": 1})
    assert not (tmp_path / "x.json.tmp").exists()


def test_json_log_skips_truncated_line(tmp_path):
    p = tmp_path / "frames.jsonl"
    seg.append_jsonl(p, {"event": "frame", "frame": "a"})
    with open(p, "a", encoding="utf-8") as f:
        f.write('{"event": "fra')
    assert seg.read_jsonl(p) == [{"event": "frame", "frame": "a"}]


# ── VRAM / CUDA / OOM ──────────────────────────────
def _torch_backend(tmp_path, monkeypatch, free_mib, cuda=True):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a: (int(free_mib * seg.MIB), 4096 * seg.MIB))
    b = seg.TorchSegBackend(make_cfg(make_work(tmp_path)))

    def no_load():
        raise AssertionError("model tidak boleh dimuat")
    monkeypatch.setattr(b, "_load_model", no_load)
    return b


def test_insufficient_vram_stops_before_load(tmp_path, monkeypatch):
    b = _torch_backend(tmp_path, monkeypatch, free_mib=2000)
    with pytest.raises(SegmentError, match=r"(?s)VRAM bebas 2000 MiB < dibutuhkan 3300.*--seg-model 0\.4b"):
        b.open()


def test_cuda_unavailable_clear_error(tmp_path, monkeypatch):
    b = _torch_backend(tmp_path, monkeypatch, free_mib=4000, cuda=False)
    with pytest.raises(SegmentError, match="CUDA tidak tersedia"):
        b.open()


def test_check_vram_04b_no_fallback_hint():
    seg.check_vram(2300, 4096, 2300, "0.4b")  # pas = cukup
    with pytest.raises(SegmentError) as e:
        seg.check_vram(2299, 4096, 2300, "0.4b")
    assert "--seg-model" not in str(e.value)


def test_oom_mid_run_logs_and_exits_3(tmp_path, capsys):
    work = make_work(tmp_path)
    cfg = make_cfg(work)
    f = factory(oom_at=3)
    with pytest.raises(SegmentOOMError, match="frame_00003"):
        run_segment(cfg, backend_factory=f, log=quiet)
    assert f.made[0].closed == 1
    assert seg.classmap_valid(work / "seg" / "classmap" / "frame_00002.png", H, W, C)
    assert not (work / "seg" / "classmap" / "frame_00003.png").exists()
    oom = [r for r in seg.read_jsonl(work / "seg" / "frames.jsonl") if r["event"] == "oom"]
    assert oom == [{**oom[0], "frame": "frame_00003.png", "index": 3, "free_mib": 12.0, "reserved_mib": 3276.0}]
    # main(): exit code 3; model tidak diganti (manifest tetap 0.8b)
    rc = main_with(work, factory(oom_at=0))
    assert rc == seg.EXIT_OOM and "OOM" in capsys.readouterr().err
    assert json.loads((work / "seg" / "manifest.json").read_text(encoding="utf-8"))["model"] == "0.8b"


def main_with(work: Path, backend_factory, *extra: str) -> int:
    cfg_path = work.parent / "c.yaml"
    cfg_path.write_text(f'paths:\n  work_dir: "{work.as_posix()}"\nsegment:\n  revision:\n'
                        f'    "0.8b": "{REV["0.8b"]}"\n    "0.4b": "{REV["0.4b"]}"\n', encoding="utf-8")
    return seg.main(["--config", str(cfg_path), *extra], backend_factory=backend_factory)


def test_main_exit_codes(tmp_path, capsys):
    work = make_work(tmp_path)
    assert main_with(work, factory(), "--limit", "2") == seg.EXIT_OK
    assert main_with(work, factory(), "--seg-model", "0.4b") == seg.EXIT_PRECONDITION
    assert "--restart" in capsys.readouterr().err
    assert main_with(work, factory()) == seg.EXIT_OK
    assert main_with(work, None, "--qc-only") == seg.EXIT_OK


def test_main_survives_cp1252_stdout(tmp_path, monkeypatch):
    # stdout diarahkan ke file di Windows = cp1252; pesan berisi "→" tidak boleh membuat crash.
    import io
    work = make_work(tmp_path)
    out = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
    err = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
    monkeypatch.setattr(seg.sys, "stdout", out)
    monkeypatch.setattr(seg.sys, "stderr", err)
    assert main_with(work, factory()) == seg.EXIT_OK
    assert main_with(work, factory(), "--seg-model", "0.4b") == seg.EXIT_PRECONDITION  # pesan berisi "→"
    out.flush()
    assert b"QC:" in out.buffer.getvalue()


def test_main_rejects_flag_combinations(tmp_path):
    work = make_work(tmp_path)
    with pytest.raises(SystemExit):
        main_with(work, factory(), "--qc-only", "--restart")
    with pytest.raises(SystemExit):
        main_with(work, factory(), "--download", "--limit", "3")


# ── Revision + cache ───────────────────────────────
def test_revision_null_stops(tmp_path):
    work = make_work(tmp_path)
    f = factory()
    with pytest.raises(SegmentError, match=r"(?s)segment\.revision\.0\.8b = null.*--download"):
        run_segment(make_cfg(work, **{"segment.revision": {"0.8b": None}}), backend_factory=f, log=quiet)
    assert f.made == []  # berhenti sebelum backend dibuat


@pytest.mark.parametrize("rev", ["main", "196a627b", "G" * 40, "A" * 40])
def test_revision_must_be_commit_hash(tmp_path, rev):
    with pytest.raises(SegmentError, match="40 karakter hex"):
        seg.resolve_revision(make_cfg(make_work(tmp_path), **{"segment.revision": {"0.8b": rev}}))


def test_revision_not_in_cache_gives_download_command(tmp_path):
    cfg = make_cfg(make_work(tmp_path), **{"segment.model": "0.4b"})
    with pytest.raises(SegmentError, match=r"(?s)tidak ada di cache HF.*python -m rotoscope\.segment --download "
                                           r"--seg-model 0\.4b"):
        seg.TorchSegBackend(cfg, cache_dir=tmp_path / "hf_cache").describe()


def test_ensure_cached_accepts_complete_snapshot(tmp_path):
    cache = tmp_path / "hf_cache"
    snap = cache / "models--facebook--sapiens2-seg-0.8b" / "snapshots" / REV["0.8b"]
    snap.mkdir(parents=True)
    for name in seg.REQUIRED_FILES[:-1]:
        (snap / name).write_text("{}", encoding="utf-8")
    with pytest.raises(SegmentError, match="model.safetensors"):
        seg.ensure_cached("facebook/sapiens2-seg-0.8b", REV["0.8b"], "0.8b", cache)
    (snap / "model.safetensors").write_bytes(b"x")
    seg.ensure_cached("facebook/sapiens2-seg-0.8b", REV["0.8b"], "0.8b", cache)


def test_download_refuses_when_offline(tmp_path):
    with pytest.raises(SegmentError, match="HF_HUB_OFFLINE"):
        seg.download(make_cfg(make_work(tmp_path)), log=quiet)


# ── Nama kelas ─────────────────────────────────────
def test_num_labels_mismatch_rejected(tmp_path):
    work = make_work(tmp_path)
    with pytest.raises(SegmentError, match=r"num_labels.*30.*29"):
        run_segment(make_cfg(work), backend_factory=factory(num_labels=30), log=quiet)
    assert not (work / "seg").exists()


# ── QC ─────────────────────────────────────────────
def test_shifted_window_edges():
    assert shifted_window(0, 100, 49) == (0, 49)
    assert shifted_window(10, 100, 49) == (0, 49)
    assert shifted_window(24, 100, 49) == (0, 49)
    assert shifted_window(50, 100, 49) == (26, 75)
    assert shifted_window(99, 100, 49) == (51, 100)
    assert shifted_window(80, 100, 49) == (51, 100)
    assert shifted_window(3, 10, 49) == (0, 10)       # n < W → seluruh klip
    for i in range(100):
        s, e = shifted_window(i, 100, 49)
        assert e - s == 49 and s <= i < e


def test_area_vs_median_start_end_and_zero():
    # 60 frame area 100; 5 frame pertama & 5 terakhir area 40 (kaki hilang) → median tetap 100 di tepi.
    areas = [40] * 5 + [100] * 50 + [40] * 5
    out = area_vs_median(areas, 49)
    assert out[0] == (100.0, 0.4) and out[4] == (100.0, 0.4) and out[59] == (100.0, 0.4)
    assert out[30] == (100.0, 1.0)
    assert area_vs_median([0, 0, 0], 3) == [(0.0, None)] * 3


def cm_rect(y0, y1, x0, x1, cls=1, h=100, w=100):
    m = np.zeros((h, w), np.uint8)
    m[y0:y1, x0:x1] = cls
    return m


def qc_rows(cms, **qc_over):
    q = make_cfg_q(**qc_over)
    names = [f"frame_{i:05d}.png" for i in range(len(cms))]
    return compute_qc(cms, q, names, list(range(len(cms))))


def make_cfg_q(**over):
    return load_pipeline(overrides={f"qc.{k}": v for k, v in over.items()}).qc


def test_qc_area_ratio():
    rows = qc_rows([cm_rect(0, 100, 0, 50), cm_rect(0, 10, 0, 10), cm_rect(0, 100, 0, 80)],
                   area_median_window=3, area_drop_min=0.0, iou_min=0.0)
    assert rows[0]["area_ratio"] == 0.5 and rows[0]["fail_reasons"] == []
    assert rows[1]["area_ratio"] == 0.01 and rows[1]["fail_reasons"] == ["area_ratio"]
    assert rows[2]["area_ratio"] == 0.8 and rows[2]["fail_reasons"] == ["area_ratio"]


def test_qc_iou_prev_and_label_agreement():
    a = cm_rect(0, 50, 0, 50, cls=1)
    b = cm_rect(0, 50, 25, 75, cls=1)
    b[0:50, 25:50] = np.where(np.arange(25) < 5, 2, 1)[None, :]  # 5/25 kolom irisan beda kelas
    rows = qc_rows([a, a, b], area_median_window=3)
    assert rows[0]["iou_prev"] is None and rows[0]["label_agreement_prev"] is None
    assert rows[1]["iou_prev"] == 1.0 and rows[1]["label_agreement_prev"] == 1.0
    assert rows[2]["iou_prev"] == pytest.approx(1 / 3)
    assert rows[2]["label_agreement_prev"] == pytest.approx(0.8)
    assert rows[2]["fail_reasons"] == ["iou_prev"]  # label_agreement tidak pernah gagal
    empty = np.zeros((10, 10), np.uint8)
    assert seg.iou(empty != 0, empty != 0) == 1.0


def test_qc_big_blobs():
    two = cm_rect(0, 40, 0, 20)
    two[0:40, 60:80] = 1                        # dua komponen masing-masing 8% > 5%
    two[90:92, 90:92] = 1                       # komponen kecil tidak dihitung
    rows = qc_rows([cm_rect(0, 40, 0, 20), two], area_median_window=3, iou_min=0.0)
    assert rows[0]["big_blobs"] == 1 and "big_blobs" not in rows[0]["fail_reasons"]
    assert rows[1]["big_blobs"] == 2 and "big_blobs" in rows[1]["fail_reasons"]


def test_qc_area_vs_median_fail_reasons_at_clip_start():
    full = cm_rect(0, 60, 0, 50)                # 30%
    legs_missing = cm_rect(0, 30, 0, 50)        # 15% → rasio 0.5 < 0.6
    cms = [legs_missing] * 3 + [full] * 10
    # W = 7: jendela frame 0 = [0, 7) → 3 buruk dari 7, median tetap area penuh.
    rows = qc_rows(cms, area_median_window=7, iou_min=0.0)
    assert [r["fail_reasons"] for r in rows[:3]] == [["area_vs_median"]] * 3
    assert rows[0]["area_vs_median"] == 0.5 and rows[0]["area_median_px"] == 3000
    assert all(r["fail_reasons"] == [] for r in rows[3:])
    # Batas metrik: zona buruk ≥ separuh jendela ikut menurunkan median → tidak tertangkap.
    rows = qc_rows(cms, area_median_window=5, iou_min=0.0)
    assert rows[0]["area_vs_median"] == 1.0 and rows[0]["fail_reasons"] == []


def test_qc_finite_from_log_and_multiple_reasons():
    cms = [cm_rect(0, 60, 0, 50), cm_rect(0, 2, 0, 2)]
    q = make_cfg_q(area_median_window=3)
    names = ["frame_00000.png", "frame_00001.png"]
    log = {"frame_00001.png": {"finite": False, "argmax_mismatch_pct": 0.1}}
    rows = compute_qc(cms, q, names, [0, 1], log)
    assert rows[0]["finite"] is None and rows[0]["fail_reasons"] == []
    assert rows[1]["fail_reasons"] == ["area_ratio", "iou_prev", "area_vs_median", "finite"]
    s = seg.summarize_qc(rows)
    assert s["n_fail"] == 1 and s["fail_frames"] == [1] and s["finite_unknown"] == 1
    assert s["fail_counts"]["finite"] == 1


def test_qc_only_without_backend_and_requires_all_frames(tmp_path):
    work = make_work(tmp_path)
    cfg = make_cfg(work)
    with pytest.raises(SegmentError, match="manifest.json tidak ada"):
        run_qc(cfg, log=quiet)
    run_segment(cfg, limit=3, backend_factory=factory(), log=quiet)
    with pytest.raises(SegmentError, match=r"3/6 belum/rusak"):
        run_qc(cfg, log=quiet)
    run_segment(cfg, backend_factory=factory(), log=quiet)
    (work / "qc_report.json").unlink()
    report = run_qc(make_cfg(work, **{"qc.area_drop_min": 0.9}), log=quiet)
    assert report["qc"]["area_drop_min"] == 0.9 and (work / "qc_report.json").is_file()
    assert report["seg_manifest"]["model"] == "0.8b"


# ── GPU nyata (opsional) ───────────────────────────
@pytest.mark.skipif(os.environ.get("ROTOSCOPE_GPU_TESTS") != "1", reason="set ROTOSCOPE_GPU_TESTS=1 (butuh GPU + cache)")
def test_real_backend_one_frame():
    b = seg.TorchSegBackend(load_pipeline())
    info = b.describe()
    b.open()
    try:
        logits, stats = b.infer(np.zeros((64, 48, 3), np.uint8))
    finally:
        b.close()
    assert logits.shape == (info.num_labels, 64, 48) and logits.dtype == np.float32
    assert stats["peak_reserved_mib"] > 0
