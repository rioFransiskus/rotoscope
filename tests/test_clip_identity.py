"""Test identitas klip di manifest [2]/[2c]/[3]/[6] (T-108, D-010): backend palsu, tanpa GPU/model."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

import test_depth as td
import test_export as te
import test_ingest as ti
import test_segment as ts
import test_stabilize as tst
from rotoscope import depth as dep
from rotoscope import export as ex
from rotoscope import segment as seg
from rotoscope import stabilize as stb
from rotoscope import stage_common as sc
from rotoscope.ingest import ingest
from rotoscope.stage_common import StageError

# fixture autouse modul-modul itu (env offline, tanpa delay) tidak ikut ter-import → pasang lagi
_env_seg, _env_dep = ts._env, td._env


def other_clip(work: Path) -> dict:
    """Simulasi ingest klip lain ke work_dir yang sama: meta.json berubah (frame, ukuran, hitungan sama)."""
    p = work / "meta.json"
    meta = json.loads(p.read_text(encoding="utf-8"))
    meta["source_path"] = "C:/clips/klip_lain.mp4"
    p.write_text(json.dumps(meta), encoding="utf-8")
    return sc.clip_identity(work)


def strip_clip(manifest_path: Path) -> None:
    m = json.loads(manifest_path.read_text(encoding="utf-8"))
    m.pop("clip")
    m.pop("adopted_utc", None)
    manifest_path.write_text(json.dumps(m), encoding="utf-8")


def read_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def seg_bytes(work: Path) -> list[bytes]:
    return [p.read_bytes() for p in sorted((work / "seg").rglob("frame_*"))]


# ── stage_common ───────────────────────────────────
def test_identity_stable_and_sensitive(tmp_path):
    work = ts.make_work(tmp_path)
    a = sc.clip_identity(work)
    assert a == sc.clip_identity(work)
    assert a["source_path"] == "C:/clips/a.mp4" and len(a["meta_sha256"]) == 64
    meta = read_json(work / "meta.json")
    for key, val in (("target_fps", 12), ("source_path", "C:/clips/b.mp4"), ("working_width", 99)):
        (work / "meta.json").write_text(json.dumps({**meta, key: val}), encoding="utf-8")
        assert sc.clip_identity(work)["meta_sha256"] != a["meta_sha256"]


def test_identity_missing_meta(tmp_path):
    with pytest.raises(StageError, match="ingest"):
        sc.clip_identity(tmp_path)


def test_identity_diff_short_hash():
    old = {"source_path": "a", "meta_sha256": "a" * 64}
    new = {"source_path": "b", "meta_sha256": "b" * 64}
    assert sc.identity_diff(old, new) == ["source_path: a → b", "meta_sha256: " + "a" * 12 + " → " + "b" * 12]
    assert sc.identity_diff(old, old) == []
    assert sc.identity_diff(None, new)[0] == "source_path: None → b"


def test_identity_equals_old_inline_export_code(tmp_path):
    """clip_identity (helper baru) = kode inline lama run_export, diuji pada manifest export lama."""
    work = ts.make_work(tmp_path)
    meta_bytes = (work / "meta.json").read_bytes()
    meta = json.loads(meta_bytes.decode("utf-8"))
    old_inline = {"meta_sha256": hashlib.sha256(meta_bytes).hexdigest(), "source_path": str(meta["source_path"])}
    assert sc.clip_identity(work) == sc.clip_identity_from_bytes(meta_bytes) == old_inline
    assert list(sc.clip_identity(work)) == list(old_inline)       # urutan kunci (byte manifest) sama


# ── [2] segment ────────────────────────────────────
def test_segment_manifest_has_identity(tmp_path):
    work = ts.make_work(tmp_path)
    seg.run_segment(ts.make_cfg(work), backend_factory=ts.factory(), log=ts.quiet)
    assert read_json(work / "seg" / "manifest.json")["clip"] == sc.clip_identity(work)


@pytest.mark.parametrize("limit", [None, 2])
def test_segment_other_clip_rejected_before_backend(tmp_path, limit):
    work = ts.make_work(tmp_path)
    seg.run_segment(ts.make_cfg(work), backend_factory=ts.factory(), log=ts.quiet)
    before = seg_bytes(work)
    new = other_clip(work)
    f = ts.factory()
    with pytest.raises(StageError) as e:
        seg.run_segment(ts.make_cfg(work), limit=limit, backend_factory=f, log=ts.quiet)
    msg = str(e.value)
    assert "LAIN" in msg and "C:/clips/a.mp4" in msg and "klip_lain.mp4" in msg and "--restart" in msg
    assert new["meta_sha256"][:12] in msg
    assert f.made == [] and seg_bytes(work) == before             # backend tidak dibuat, output utuh


def test_segment_identity_checked_before_resolve_revision(tmp_path):
    work = ts.make_work(tmp_path)
    seg.run_segment(ts.make_cfg(work), backend_factory=ts.factory(), log=ts.quiet)
    other_clip(work)
    cfg = ts.make_cfg(work, **{"segment.revision": {"0.8b": None, "0.4b": None}})   # revision null → error lain
    with pytest.raises(StageError, match="LAIN"):
        seg.run_segment(cfg, backend_factory=ts.factory(), log=ts.quiet)


def test_segment_restart_after_other_clip(tmp_path):
    work = ts.make_work(tmp_path)
    seg.run_segment(ts.make_cfg(work), backend_factory=ts.factory(), log=ts.quiet)
    new = other_clip(work)
    run = seg.run_segment(ts.make_cfg(work), restart=True, backend_factory=ts.factory(), log=ts.quiet)
    assert run["processed"] == ts.N_FRAMES
    assert read_json(work / "seg" / "manifest.json")["clip"] == new


def test_segment_qc_only_checks_identity(tmp_path):
    work = ts.make_work(tmp_path)
    seg.run_segment(ts.make_cfg(work), backend_factory=ts.factory(), log=ts.quiet)
    seg.run_qc(ts.make_cfg(work), log=ts.quiet)                   # klip sama: jalan
    other_clip(work)
    with pytest.raises(StageError, match="LAIN"):
        seg.run_qc(ts.make_cfg(work), log=ts.quiet)


def test_segment_manifest_without_identity_rejected_suggests_adopt(tmp_path):
    work = ts.make_work(tmp_path)
    seg.run_segment(ts.make_cfg(work), backend_factory=ts.factory(), log=ts.quiet)
    strip_clip(work / "seg" / "manifest.json")
    f = ts.factory()
    with pytest.raises(StageError, match=r"--adopt[\s\S]*--restart"):
        seg.run_segment(ts.make_cfg(work), backend_factory=f, log=ts.quiet)
    with pytest.raises(StageError, match="--adopt"):
        seg.run_qc(ts.make_cfg(work), log=ts.quiet)
    assert f.made == []


def test_segment_adopt_without_inference(tmp_path):
    work = ts.make_work(tmp_path)
    seg.run_segment(ts.make_cfg(work), backend_factory=ts.factory(), log=ts.quiet)
    man_path = work / "seg" / "manifest.json"
    strip_clip(man_path)
    before_man, before_files = read_json(man_path), seg_bytes(work)
    msgs = []
    out = seg.adopt_segment(ts.make_cfg(work), log=msgs.append)   # tanpa backend_factory: tidak ada model
    assert out["adopted"] is True
    after = read_json(man_path)
    assert after["clip"] == sc.clip_identity(work) and "adopted_utc" in after
    assert {k: v for k, v in after.items() if k not in ("clip", "adopted_utc")} == before_man
    assert seg_bytes(work) == before_files
    assert "C:/clips/a.mp4" in msgs[0] and f"{ts.N_FRAMES} frame valid" in msgs[0] and "PERNYATAAN" in msgs[0]
    assert [r["event"] for r in sc.read_jsonl(work / "seg" / "frames.jsonl")][-1] == "adopt"
    f = ts.factory()
    run = seg.run_segment(ts.make_cfg(work), backend_factory=f, log=ts.quiet)
    assert run["processed"] == 0 and run["skipped"] == ts.N_FRAMES


def test_segment_adopt_second_time_noop_and_other_clip_refused(tmp_path):
    work = ts.make_work(tmp_path)
    seg.run_segment(ts.make_cfg(work), backend_factory=ts.factory(), log=ts.quiet)
    before = (work / "seg" / "manifest.json").read_bytes()
    assert seg.adopt_segment(ts.make_cfg(work), log=ts.quiet)["adopted"] is False
    assert (work / "seg" / "manifest.json").read_bytes() == before
    other_clip(work)
    with pytest.raises(StageError, match=r"LAIN[\s\S]*--restart"):
        seg.adopt_segment(ts.make_cfg(work), log=ts.quiet)
    assert (work / "seg" / "manifest.json").read_bytes() == before


def test_segment_adopt_sanity_checks(tmp_path):
    work = ts.make_work(tmp_path)
    seg.run_segment(ts.make_cfg(work), backend_factory=ts.factory(), log=ts.quiet)
    man_path = work / "seg" / "manifest.json"
    strip_clip(man_path)
    stripped = man_path.read_bytes()
    # frame_size manifest ≠ meta.json
    m = read_json(man_path)
    m["frame_size"] = {"width": 99, "height": 99}
    man_path.write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(StageError, match="frame_size"):
        seg.adopt_segment(ts.make_cfg(work), log=ts.quiet)
    man_path.write_bytes(stripped)
    # frame hilang
    gone = work / "seg" / "probs" / "frame_00002.npz"
    keep = gone.read_bytes()
    gone.unlink()
    with pytest.raises(StageError, match="frame belum/rusak"):
        seg.adopt_segment(ts.make_cfg(work), log=ts.quiet)
    gone.write_bytes(keep)
    # file yatim (frame di luar daftar klip)
    orphan = work / "seg" / "classmap" / "frame_00099.png"
    shutil.copy(work / "seg" / "classmap" / "frame_00000.png", orphan)
    with pytest.raises(StageError, match="di luar daftar frame"):
        seg.adopt_segment(ts.make_cfg(work), log=ts.quiet)
    assert man_path.read_bytes() == stripped                       # tidak ada yang ditulis


def test_segment_adopt_needs_manifest(tmp_path):
    work = ts.make_work(tmp_path)
    with pytest.raises(StageError, match="tidak ada yang di-adopt"):
        seg.adopt_segment(ts.make_cfg(work), log=ts.quiet)


@pytest.mark.parametrize("flag", [["--restart"], ["--limit", "2"], ["--qc-only"], ["--download"]])
def test_segment_main_adopt_combination_exit_1(tmp_path, capsys, flag):
    work = ts.make_work(tmp_path)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(f"paths:\n  work_dir: '{work.as_posix()}'\n", encoding="utf-8")
    assert seg.main(["--config", str(cfg), "--adopt", *flag]) == sc.EXIT_PRECONDITION
    assert "--adopt tidak bisa digabung" in capsys.readouterr().err


# ── [2c] depth ─────────────────────────────────────
def test_depth_manifest_has_identity(tmp_path):
    work = td.make_work(tmp_path)
    dep.run_depth(td.make_cfg(work), backend_factory=td.factory(), log=td.quiet)
    assert read_json(work / "depth" / "manifest.json")["clip"] == sc.clip_identity(work)


@pytest.mark.parametrize("limit", [None, 2])
def test_depth_other_clip_rejected_before_backend(tmp_path, limit):
    work = td.make_work(tmp_path)
    dep.run_depth(td.make_cfg(work), backend_factory=td.factory(), log=td.quiet)
    before = [p.read_bytes() for p in sorted((work / "depth").glob("frame_*.npy"))]
    other_clip(work)
    f = td.factory()
    with pytest.raises(StageError, match=r"LAIN[\s\S]*klip_lain[\s\S]*--restart"):
        dep.run_depth(td.make_cfg(work), limit=limit, backend_factory=f, log=td.quiet)
    assert f.made == []
    assert [p.read_bytes() for p in sorted((work / "depth").glob("frame_*.npy"))] == before


def test_depth_identity_checked_before_resolve_revision(tmp_path):
    work = td.make_work(tmp_path)
    dep.run_depth(td.make_cfg(work), backend_factory=td.factory(), log=td.quiet)
    other_clip(work)
    with pytest.raises(StageError, match="LAIN"):
        dep.run_depth(td.make_cfg(work, **{"depth.revision": None}), backend_factory=td.factory(), log=td.quiet)


def test_depth_restart_after_other_clip(tmp_path):
    work = td.make_work(tmp_path)
    dep.run_depth(td.make_cfg(work), backend_factory=td.factory(), log=td.quiet)
    new = other_clip(work)
    run = dep.run_depth(td.make_cfg(work), restart=True, backend_factory=td.factory(), log=td.quiet)
    assert run["processed"] == td.N_FRAMES
    assert read_json(work / "depth" / "manifest.json")["clip"] == new


def test_depth_manifest_without_identity_rejected_suggests_adopt(tmp_path):
    work = td.make_work(tmp_path)
    dep.run_depth(td.make_cfg(work), backend_factory=td.factory(), log=td.quiet)
    strip_clip(work / "depth" / "manifest.json")
    f = td.factory()
    with pytest.raises(StageError, match=r"--adopt[\s\S]*--restart"):
        dep.run_depth(td.make_cfg(work), backend_factory=f, log=td.quiet)
    assert f.made == []


def test_depth_adopt_without_inference(tmp_path):
    work = td.make_work(tmp_path)
    dep.run_depth(td.make_cfg(work), backend_factory=td.factory(), log=td.quiet)
    man_path = work / "depth" / "manifest.json"
    strip_clip(man_path)
    before_man = read_json(man_path)
    before = [p.read_bytes() for p in sorted((work / "depth").glob("frame_*.npy"))]
    msgs = []
    assert dep.adopt_depth(td.make_cfg(work), log=msgs.append)["adopted"] is True
    after = read_json(man_path)
    assert after["clip"] == sc.clip_identity(work) and "adopted_utc" in after
    assert {k: v for k, v in after.items() if k not in ("clip", "adopted_utc")} == before_man
    assert [p.read_bytes() for p in sorted((work / "depth").glob("frame_*.npy"))] == before
    assert f"{td.N_FRAMES} frame valid" in msgs[0] and "C:/clips/a.mp4" in msgs[0]
    run = dep.run_depth(td.make_cfg(work), backend_factory=td.factory(), log=td.quiet)
    assert run["processed"] == 0 and run["skipped"] == td.N_FRAMES
    assert dep.adopt_depth(td.make_cfg(work), log=td.quiet)["adopted"] is False


def test_depth_adopt_sanity_and_other_clip(tmp_path):
    work = td.make_work(tmp_path)
    dep.run_depth(td.make_cfg(work), backend_factory=td.factory(), log=td.quiet)
    man_path = work / "depth" / "manifest.json"
    strip_clip(man_path)
    stripped = man_path.read_bytes()
    gone = work / "depth" / "frame_00001.npy"
    keep = gone.read_bytes()
    gone.unlink()
    with pytest.raises(StageError, match="frame belum/rusak"):
        dep.adopt_depth(td.make_cfg(work), log=td.quiet)
    gone.write_bytes(keep)
    (work / "depth" / "frame_00077.npy").write_bytes(keep)
    with pytest.raises(StageError, match="di luar daftar frame"):
        dep.adopt_depth(td.make_cfg(work), log=td.quiet)
    (work / "depth" / "frame_00077.npy").unlink()
    m = read_json(man_path)
    m["frame_size"] = {"width": 1, "height": 1}
    man_path.write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(StageError, match="frame_size"):
        dep.adopt_depth(td.make_cfg(work), log=td.quiet)
    man_path.write_bytes(stripped)
    dep.adopt_depth(td.make_cfg(work), log=td.quiet)
    other_clip(work)
    with pytest.raises(StageError, match=r"LAIN[\s\S]*--restart"):
        dep.adopt_depth(td.make_cfg(work), log=td.quiet)


@pytest.mark.parametrize("flag", [["--restart"], ["--limit", "2"], ["--download"]])
def test_depth_main_adopt_combination_exit_1(tmp_path, capsys, flag):
    work = td.make_work(tmp_path)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(f"paths:\n  work_dir: '{work.as_posix()}'\n", encoding="utf-8")
    assert dep.main(["--config", str(cfg), "--adopt", *flag]) == sc.EXIT_PRECONDITION
    assert "--adopt tidak bisa digabung" in capsys.readouterr().err


# ── [3] stabilize ──────────────────────────────────
def retag_inputs(work: Path, ident: dict) -> None:
    """Simulasi [2]/[2c] dijalankan ulang untuk klip baru: identitas di manifest input."""
    for stage in ("seg", "depth"):
        p = work / stage / "manifest.json"
        p.write_text(json.dumps({**read_json(p), "clip": ident}), encoding="utf-8")


def test_stabilize_manifest_has_identity(tmp_path):
    work = tst.make_work(tmp_path)
    stb.run_stabilize(tst.cfg_for(work), log=tst.quiet)
    assert read_json(work / "stable" / "manifest.json")["clip"] == sc.clip_identity(work)


def test_stabilize_identity_change_recomputes_with_warning(tmp_path):
    work = tst.make_work(tmp_path)
    stb.run_stabilize(tst.cfg_for(work), log=tst.quiet)
    new = other_clip(work)
    retag_inputs(work, new)
    msgs = []
    run = stb.run_stabilize(tst.cfg_for(work), log=msgs.append)
    assert run["processed"] == tst.N_FRAMES and "basi" in msgs[0]
    assert any(s.startswith("clip.meta_sha256") for s in run["stale"])
    assert any(s.startswith("clip.source_path") for s in run["stale"])
    assert read_json(work / "stable" / "manifest.json")["clip"] == new


def test_stabilize_old_manifest_without_identity_recomputed_once(tmp_path):
    work = tst.make_work(tmp_path)
    stb.run_stabilize(tst.cfg_for(work), log=tst.quiet)
    strip_clip(work / "stable" / "manifest.json")
    run = stb.run_stabilize(tst.cfg_for(work), log=tst.quiet)
    assert run["processed"] == tst.N_FRAMES and any(s.startswith("clip.meta_sha256: None") for s in run["stale"])
    again = stb.run_stabilize(tst.cfg_for(work), log=tst.quiet)
    assert again["processed"] == 0 and again["stale"] == []


@pytest.mark.parametrize("stage, cmd", [("seg", "rotoscope segment"), ("depth", "rotoscope depth")])
def test_stabilize_input_without_identity_suggests_adopt(tmp_path, stage, cmd):
    work = tst.make_work(tmp_path)
    strip_clip(work / stage / "manifest.json")
    with pytest.raises(StageError, match=rf"{cmd} \S+ --adopt[\s\S]*{cmd} \S+ --restart --yes"):
        stb.run_stabilize(tst.cfg_for(work), log=tst.quiet)


@pytest.mark.parametrize("stage, cmd", [("seg", "rotoscope segment"), ("depth", "rotoscope depth")])
def test_stabilize_input_other_clip_suggests_restart(tmp_path, stage, cmd):
    work = tst.make_work(tmp_path)
    new = other_clip(work)
    if stage == "depth":   # input [2] sudah milik klip baru; hanya [2c] yang tertinggal
        p = work / "seg" / "manifest.json"
        p.write_text(json.dumps({**read_json(p), "clip": new}), encoding="utf-8")
    with pytest.raises(StageError, match=rf"{stage}/manifest.json milik klip LAIN[\s\S]*{cmd} \S+ --restart --yes[\s\S]*"
                                         f"ingest klip yang benar"):
        stb.run_stabilize(tst.cfg_for(work), log=tst.quiet)
    assert not (work / "stable").exists()


# ── ingest NYATA klip kedua ke work_dir berisi output klip A ──
@pytest.mark.skipif(ti.pytestmark.args[0], reason="ffmpeg/ffprobe tidak ada di PATH")
def test_real_second_ingest_into_workdir_with_clip_a_outputs(tmp_path):
    """Klip B (frame_size + frame_count SAMA, path beda) di-ingest sungguhan ke work_dir klip A:
    [2]/[2c] menolak sebelum backend dibuat, [3] menolak input klip A; setelah --restart [2]/[2c]
    → [3] menghitung ulang otomatis dengan peringatan."""
    work = tmp_path / "work"
    vid_a = ti.make_video(tmp_path / "a.mp4", 64, 48, fps=24, duration=0.25)
    vid_b = ti.make_video(tmp_path / "b.mp4", 64, 48, fps=24, duration=0.25)
    meta_a = ingest(vid_a, work)
    seg_cfg, dep_cfg = ts.make_cfg(work), td.make_cfg(work)
    seg.run_segment(seg_cfg, backend_factory=ts.factory(), log=ts.quiet)
    dep.run_depth(dep_cfg, backend_factory=td.factory(), log=td.quiet)
    stb.run_stabilize(tst.cfg_for(work), log=tst.quiet)
    ident_a = sc.clip_identity(work)
    assert read_json(work / "stable" / "manifest.json")["clip"] == ident_a

    meta_b = ingest(vid_b, work)                                  # ingest nyata, bukan sunting tangan
    assert (meta_b["working_width"], meta_b["working_height"], meta_b["frame_count"]) == \
           (meta_a["working_width"], meta_a["working_height"], meta_a["frame_count"])
    ident_b = sc.clip_identity(work)
    assert ident_b != ident_a
    fs, fd = ts.factory(), td.factory()
    with pytest.raises(StageError, match=r"LAIN[\s\S]*a\.mp4[\s\S]*b\.mp4"):
        seg.run_segment(seg_cfg, backend_factory=fs, log=ts.quiet)
    with pytest.raises(StageError, match=r"LAIN[\s\S]*a\.mp4[\s\S]*b\.mp4"):
        dep.run_depth(dep_cfg, backend_factory=fd, log=td.quiet)
    assert fs.made == [] and fd.made == []
    with pytest.raises(StageError, match="milik klip LAIN"):
        stb.run_stabilize(tst.cfg_for(work), log=tst.quiet)

    seg.run_segment(seg_cfg, restart=True, backend_factory=ts.factory(), log=ts.quiet)
    dep.run_depth(dep_cfg, restart=True, backend_factory=td.factory(), log=td.quiet)
    msgs = []
    run = stb.run_stabilize(tst.cfg_for(work), log=msgs.append)
    assert "basi" in msgs[0] and run["processed"] == meta_b["frame_count"]
    assert read_json(work / "stable" / "manifest.json")["clip"] == ident_b


# ── [6] export ─────────────────────────────────────
@pytest.mark.skipif(te.pytestmark.args[0], reason="ffmpeg/ffprobe tidak ada di PATH")
def test_export_manifest_identity_equals_old_inline_and_old_manifest_still_skips(tmp_path):
    cfg, work, out = te.make_clip(tmp_path)
    ex.run_export(cfg, log=lambda m: None)
    mp = ex.manifest_path_for(out / "meme_clip.mp4")
    m = read_json(mp)
    meta_bytes = (work / "meta.json").read_bytes()
    old_inline = {"meta_sha256": hashlib.sha256(meta_bytes).hexdigest(),
                  "source_path": str(json.loads(meta_bytes.decode("utf-8"))["source_path"])}
    assert m["clip"] == old_inline
    mp.write_text(json.dumps({**m, "clip": old_inline}), encoding="utf-8")   # manifest "lama"
    assert ex.run_export(cfg, log=lambda m: None)["skipped"] is True


@pytest.mark.skipif(te.pytestmark.args[0], reason="ffmpeg/ffprobe tidak ada di PATH")
@pytest.mark.parametrize("mode", ["missing", "other"])
def test_export_rejects_stable_without_or_with_other_identity(tmp_path, mode):
    cfg, work, out = te.make_clip(tmp_path)
    p = work / "stable" / "manifest.json"
    m = read_json(p)
    if mode == "missing":
        m.pop("clip")
    else:
        m["clip"] = {"meta_sha256": "f" * 64, "source_path": "C:/clips/klip_lain.mp4"}
    p.write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(StageError, match="jalankan ulang stage \\[3\\]"):
        ex.run_export(cfg, log=lambda m: None)
    assert not out.exists() or not list(out.glob("*.mp4"))
