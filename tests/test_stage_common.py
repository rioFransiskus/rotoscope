"""Test stage_common.py (T-105): helper bersama segment.py / depth.py."""

from __future__ import annotations

import json
import os

import pytest

from rotoscope import stage_common as sc
from rotoscope.stage_common import StageError, StageOOMError


@pytest.fixture(autouse=True)
def _no_delay(monkeypatch):
    monkeypatch.setattr(sc, "REPLACE_DELAY_S", 0)


def test_oom_error_is_stage_error():
    assert issubclass(StageOOMError, StageError)


def test_write_json_atomic_retries_then_succeeds(tmp_path, monkeypatch):
    real = os.replace
    calls = []

    def flaky(src, dst):
        calls.append(dst)
        if len(calls) < 3:
            raise PermissionError(32, "file sedang dipakai proses lain")
        return real(src, dst)

    monkeypatch.setattr(sc.os, "replace", flaky)
    target = tmp_path / "x.json"
    sc.write_json_atomic(target, {"a": "→"})
    assert len(calls) == 3 and json.loads(target.read_text(encoding="utf-8")) == {"a": "→"}
    assert not (tmp_path / "x.json.tmp").exists()


def test_write_bytes_atomic_clear_error_after_retries(tmp_path, monkeypatch):
    calls = []

    def locked(src, dst):
        calls.append(dst)
        raise PermissionError(5, "Access is denied")

    monkeypatch.setattr(sc.os, "replace", locked)
    with pytest.raises(StageError, match=rf"{sc.REPLACE_RETRIES} percobaan.*antivirus"):
        sc.write_bytes_atomic(tmp_path / "x.bin", b"abc")
    assert len(calls) == sc.REPLACE_RETRIES and not (tmp_path / "x.bin.tmp").exists()


def test_write_json_atomic_rejects_nan(tmp_path):
    with pytest.raises(ValueError):
        sc.write_json_atomic(tmp_path / "x.json", {"a": float("nan")})


def test_clean_tmp_recursive_and_missing_dir(tmp_path):
    (tmp_path / "a" / "b").mkdir(parents=True)
    (tmp_path / "a" / "x.npy.tmp").write_bytes(b"1")
    (tmp_path / "a" / "b" / "y.png.tmp").write_bytes(b"1")
    (tmp_path / "a" / "keep.npy").write_bytes(b"1")
    sc.clean_tmp(tmp_path / "a")
    assert sorted(p.name for p in (tmp_path / "a").rglob("*") if p.is_file()) == ["keep.npy"]
    sc.clean_tmp(tmp_path / "tidak_ada")  # tidak error


def test_jsonl_truncated_line_and_last_record(tmp_path):
    p = tmp_path / "frames.jsonl"
    sc.append_jsonl(p, {"event": "frame", "frame": "a", "v": 1})
    sc.append_jsonl(p, {"event": "run_end"})
    sc.append_jsonl(p, {"event": "frame", "frame": "a", "v": 2})
    with open(p, "a", encoding="utf-8") as f:
        f.write('{"event": "fra')
    assert len(sc.read_jsonl(p)) == 3
    assert sc.last_frame_records(p) == {"a": {"event": "frame", "frame": "a", "v": 2}}
    assert sc.read_jsonl(tmp_path / "tidak_ada.jsonl") == []


def test_load_frame_list_and_require_frames(tmp_path):
    with pytest.raises(StageError, match="ingest dulu"):
        sc.load_frame_list(tmp_path)
    (tmp_path / "meta.json").write_text(json.dumps(
        {"frame_count": 3, "working_width": 16, "working_height": 24, "frame_index_start": 5}), encoding="utf-8")
    names, indices, w, h = sc.load_frame_list(tmp_path)
    assert names == ("frame_00005.png", "frame_00006.png", "frame_00007.png") and indices == (5, 6, 7)
    assert (w, h) == (16, 24)
    (tmp_path / "frames").mkdir()
    (tmp_path / "frames" / "frame_00005.png").write_bytes(b"x")
    with pytest.raises(StageError, match=r"2 frame hilang.*frame_00006"):
        sc.require_frames(tmp_path, names)
    sc.require_frames(tmp_path, names[:1])


def test_read_rgb_missing_or_corrupt(tmp_path):
    with pytest.raises(StageError, match="gagal membaca frame"):
        sc.read_rgb(tmp_path / "x.png")
    (tmp_path / "y.png").write_bytes(b"bukan png")
    with pytest.raises(StageError, match="gagal membaca frame"):
        sc.read_rgb(tmp_path / "y.png")


def test_ensure_cached_uses_given_files_and_command(tmp_path):
    cache = tmp_path / "hf_cache"
    rev = "e" * 40
    snap = cache / "models--org--m" / "snapshots" / rev
    snap.mkdir(parents=True)
    (snap / "a.json").write_text("{}", encoding="utf-8")
    with pytest.raises(StageError, match=r"(?s)hilang: b\.bin\).*\n    jalankan --unduh"):
        sc.ensure_cached("org/m", rev, ("a.json", "b.bin"), "jalankan --unduh", cache)
    (snap / "b.bin").write_bytes(b"x")
    sc.ensure_cached("org/m", rev, ("a.json", "b.bin"), "jalankan --unduh", cache)
    assert sc.cached_file("org/m", "a.json", rev, cache) == snap / "a.json"
    with pytest.raises(StageError, match="c.txt"):
        sc.cached_file("org/m", "c.txt", rev, cache)


@pytest.mark.parametrize("value,active", [("1", True), ("true", True), (" ON ", True), ("0", False), ("", False)])
def test_hf_offline_active(monkeypatch, value, active):
    monkeypatch.setenv(sc.HF_OFFLINE_ENV, value)
    assert sc.hf_offline_active() is active


def test_commit_hash_re():
    assert sc.COMMIT_HASH_RE.fullmatch("5426e4f0f36572d16453bbda7a8389317b1bef99")
    for bad in ("main", "5426e4f0", "A" * 40, "g" * 40):
        assert not sc.COMMIT_HASH_RE.fullmatch(bad)
