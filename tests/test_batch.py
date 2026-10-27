"""Test batch (T-603): rencana CPU tanpa GPU / tanpa penghapusan; jalan (proses anak dipalsukan); parser --list; VRAM; ringkasan.

Ingest dipalsukan (meta.json + frames/ kosong) — tidak ada ffmpeg, GPU, atau torch. Stage GPU tidak pernah dipanggil.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pytest

from rotoscope import batch
from rotoscope import cli
from rotoscope import crop as crop_mod
from rotoscope import isolate as isolate_stage
from rotoscope.config import load_pipeline
from rotoscope.stage_common import StageError

SAMPLES = ["samples/klip4.mp4", "samples/test.mp4"]


@pytest.fixture
def env(tmp_path, monkeypatch):
    work, out = tmp_path / "work", tmp_path / "out"
    conf = tmp_path / "c.yaml"
    conf.write_text(f'paths:\n  work_dir: "{work.as_posix()}"\n  out_dir: "{out.as_posix()}"\n', encoding="utf-8")
    cfg = load_pipeline(conf)
    monkeypatch.setattr(cli, "preflight_tools", lambda: None)
    monkeypatch.setattr(batch, "free_bytes", lambda p: 10**12)
    monkeypatch.setattr(batch, "gpu_memory_mib", lambda: (4000.0, 4096.0))
    vids = tmp_path / "vid"
    vids.mkdir()
    for n in ("klip4.mp4", "klip5.mp4", "Klip2.MP4"):
        (vids / n).write_bytes(b"x")
    return {"cfg": cfg, "work": work, "out": out, "vids": vids, "tmp": tmp_path, "conf": conf}


def fake_ingest(frames: int = 3, w: int = 480, h: int = 854):
    def _ingest(row, cfg, args):
        row.work_dir.mkdir(parents=True, exist_ok=True)
        (row.work_dir / "frames").mkdir(exist_ok=True)
        meta = {"source_path": str(row.video.resolve()), "frame_count": frames, "frame_index_start": 0,
                "working_width": w, "working_height": h, "trim": row.spec.trim}
        (row.work_dir / "meta.json").write_text(json.dumps(meta, sort_keys=True), encoding="utf-8")
        for i in range(frames):
            (row.work_dir / "frames" / f"frame_{i:05d}.png").write_bytes(b"")
    return _ingest


def args_of(**kw) -> argparse.Namespace:
    base = dict(inputs=[], list_file=None, yes=False, skip_pending=False, config=None, style=None, seg_model=None)
    base.update(kw)
    return argparse.Namespace(**base)


def make_rows(env, names, trim=None):
    specs = [batch.Spec(path=str(env["vids"] / n), trim=trim, line=None) for n in names]
    return batch.build_rows(specs, env["cfg"])


def snapshot(root: Path) -> dict[str, str]:
    out = {}
    for sub in ("seg", "depth", "isolate", "stable", "contours", "strokes", "out"):
        for p in (root / sub).rglob("*") if (root / sub).exists() else []:
            if p.is_file():
                out[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


class FakeProc:
    def __init__(self, rc=0, interrupt=False, log=None, argv=None):
        self.rc, self.interrupt, self.log, self.argv = rc, interrupt, log, argv
        self.alive = True

    def wait(self):
        if self.interrupt:
            raise KeyboardInterrupt
        return self.rc


def popen_factory(script, log):
    """script: daftar rc per panggilan (atau 'ctrlc')."""
    calls = iter(script)

    def _popen(argv):
        v = next(calls)
        log.append(argv)
        return FakeProc(rc=0 if v == "ctrlc" else v, interrupt=(v == "ctrlc"))
    return _popen


# ── Parser --list ─────────────────────────────────
def test_list_quoted_path_with_spaces_and_trim(tmp_path):
    p = tmp_path / "l.txt"
    p.write_text('# komentar\n\n"D:\\Meme Lucu\\video 1.mp4" --start 0 --end 2\n', encoding="utf-8")
    (spec,) = batch.parse_list(p)
    assert spec.path == "D:\\Meme Lucu\\video 1.mp4"           # kutip luar dilepas, backslash tetap
    assert spec.trim["end_frame"] == 48 and spec.line == 3


def test_list_unquoted_and_backslash_plain(tmp_path):
    p = tmp_path / "l.txt"
    p.write_text("samples\\klip4.mp4\n", encoding="utf-8")
    (spec,) = batch.parse_list(p)
    assert spec.path == "samples\\klip4.mp4" and spec.trim is None


def test_list_unquoted_space_path_is_error_with_line(tmp_path):
    p = tmp_path / "l.txt"
    p.write_text("samples\\klip4.mp4\nD:\\Meme Lucu\\video.mp4\n", encoding="utf-8")
    with pytest.raises(batch.BatchError, match=r"baris 2"):
        batch.parse_list(p)


def test_list_broken_quote_is_error_with_line(tmp_path):
    p = tmp_path / "l.txt"
    p.write_text('samples\\klip4.mp4\n"D:\\Meme Lucu\\video 1.mp4\n', encoding="utf-8")
    with pytest.raises(batch.BatchError, match=r"baris 2"):
        batch.parse_list(p)


def test_list_bad_trim_and_unknown_flag(tmp_path):
    p = tmp_path / "l.txt"
    p.write_text("a.mp4 --start 2 --end 1\n", encoding="utf-8")
    with pytest.raises(batch.BatchError, match="baris 1"):
        batch.parse_list(p)
    p.write_text("a.mp4 --limit 3\n", encoding="utf-8")
    with pytest.raises(batch.BatchError, match="argumen tidak dikenal"):
        batch.parse_list(p)


def test_list_path_with_utf8_bom(tmp_path):
    p = tmp_path / "l.txt"
    p.write_bytes("\ufeffsamples\\klip4.mp4\n".encode("utf-8"))
    (spec,) = batch.parse_list(p)
    assert spec.path == "samples\\klip4.mp4"


# ── Masukan CLI, duplikat, flag terlarang ─────────
def test_folder_non_recursive_sorted_and_ext_filter(env):
    d = env["vids"]
    (d / "sub").mkdir()
    (d / "sub" / "dalam.mp4").write_bytes(b"x")
    (d / "catatan.txt").write_text("x", encoding="utf-8")
    specs = batch.expand_inputs([str(d)], env["cfg"].batch.video_exts)
    names = [Path(s.path).name for s in specs]
    assert names == ["Klip2.MP4", "klip4.mp4", "klip5.mp4"]      # urut nama; huruf besar ekstensi diterima; tanpa sub/ dan .txt


def test_duplicate_stem_from_other_folder_rejected(env):
    other = env["tmp"] / "lain"
    other.mkdir()
    (other / "klip4.mp4").write_bytes(b"x")
    specs = [batch.Spec(str(env["vids"] / "klip4.mp4"), None, 1), batch.Spec(str(other / "klip4.mp4"), None, 2)]
    with pytest.raises(batch.BatchError, match="duplikat"):
        batch.build_rows(specs, env["cfg"])


def test_same_stem_different_trim_is_not_duplicate(env):
    specs = [batch.Spec(str(env["vids"] / "klip4.mp4"), None, None),
             batch.Spec(str(env["vids"] / "klip4.mp4"), {"start_s": 0.0, "end_s": 1.0, "start_frame": 0, "end_frame": 24}, None)]
    rows = batch.build_rows(specs, env["cfg"])
    assert rows[0].work_dir != rows[1].work_dir


@pytest.mark.parametrize("flag", ["--restart-from", "--preview", "--from", "--adopt", "--qc-only", "--limit"])
def test_forbidden_flags_exit_1(env, flag, monkeypatch):
    monkeypatch.setattr(batch, "plan", lambda *a, **k: pytest.fail("rencana tidak boleh jalan"))
    assert batch.main([flag, "3", str(env["vids"] / "klip4.mp4")]) == 1


# ── Rencana (CPU) ─────────────────────────────────
def test_plan_siap_counts_gpu_and_disk_exit_0(env, monkeypatch):
    monkeypatch.setattr(batch, "_ingest_quiet", lambda r, c, a: fake_ingest(4)(r, c, a))
    rows = make_rows(env, ["klip4.mp4"])
    batch.plan(rows, env["cfg"], args_of())
    r = rows[0]
    assert r.status == "siap" and r.frames == 4
    assert r.gpu_s == pytest.approx(4 * (16.5 + 0.19))           # seg 0.8b + depth, semua frame belum valid
    assert r.disk_mib == pytest.approx(8.5 * (480 * 854 / 1e6) * 4)
    assert batch.plan_exit(rows, batch.disk_check(rows, env["cfg"])) == 0


def test_plan_writes_nothing_in_gpu_or_out_folders(env, monkeypatch):
    """K2 unit: rencana tanpa --yes tidak mengubah seg/ depth/ isolate/ stable/ contours/ strokes/ out/."""
    root = env["work"] / "clips" / "klip4"
    for sub in ("seg", "depth", "isolate", "stable", "contours", "strokes"):
        (root / sub).mkdir(parents=True)
        (root / sub / "x.bin").write_bytes(b"keep")
    (root / "isolate" / "labels").mkdir()
    (env["out"] / "svg").mkdir(parents=True)
    (env["out"] / "klip4.mp4").write_bytes(b"keep")
    before = snapshot(root) | snapshot(env["out"])
    monkeypatch.setattr(batch, "_ingest_quiet", lambda r, c, a: fake_ingest(3)(r, c, a))
    monkeypatch.setattr(isolate_stage, "remove_outputs", lambda *a: pytest.fail("rencana tidak boleh menghapus isolate/"))
    rows = make_rows(env, ["klip4.mp4"])
    batch.plan(rows, env["cfg"], args_of())
    after = snapshot(root) | snapshot(env["out"])
    assert before == after and (root / "isolate" / "x.bin").exists() and (root / "isolate" / "labels").is_dir()


def test_isolate_dry_run_not_called_without_points(env, monkeypatch):
    """G: tanpa target/points.json, rencana tidak memanggil run_isolate sama sekali (jalur hapus tidak tersentuh)."""
    monkeypatch.setattr(batch, "_ingest_quiet", lambda r, c, a: fake_ingest(2)(r, c, a))
    called = []
    monkeypatch.setattr(isolate_stage, "run_isolate", lambda *a, **k: called.append(k) or {"status": "dry-run"})
    rows = make_rows(env, ["klip4.mp4"])
    batch.plan(rows, env["cfg"], args_of())
    assert called == [] and rows[0].target == "tidak ada"


def test_isolate_gets_clip_work_dir_not_root(env, monkeypatch):
    """Regresi (Tahap 2): run_isolate membaca paths.work_dir → harus folder klip, bukan folder root `work/`."""
    monkeypatch.setattr(batch, "_ingest_quiet", lambda r, c, a: fake_ingest(2)(r, c, a))
    rows = make_rows(env, ["klip4.mp4"])
    (rows[0].work_dir / "target").mkdir(parents=True)
    (rows[0].work_dir / "target" / "points.json").write_text("{}", encoding="utf-8")
    seen = []
    monkeypatch.setattr(isolate_stage, "run_isolate", lambda cfg, **k: seen.append(cfg.paths.work_dir) or {"status": "up-to-date"})
    batch.plan(rows, env["cfg"], args_of())
    assert seen == [rows[0].work_dir] and env["cfg"].paths.work_dir != rows[0].work_dir


def test_isolate_dry_run_always_dry_when_points_present(env, monkeypatch):
    monkeypatch.setattr(batch, "_ingest_quiet", lambda r, c, a: fake_ingest(2)(r, c, a))
    rows = make_rows(env, ["klip4.mp4"])
    (rows[0].work_dir / "target").mkdir(parents=True)
    (rows[0].work_dir / "target" / "points.json").write_text("{}", encoding="utf-8")
    seen = []
    monkeypatch.setattr(isolate_stage, "run_isolate", lambda cfg, **k: seen.append(k.get("dry_run")) or {"status": "up-to-date"})
    batch.plan(rows, env["cfg"], args_of())
    assert seen == [True] and rows[0].target == "ada (up-to-date)"


def test_target_basi_is_perlu_tindakan(env, monkeypatch):
    monkeypatch.setattr(batch, "_ingest_quiet", lambda r, c, a: fake_ingest(2)(r, c, a))
    rows = make_rows(env, ["klip4.mp4"])
    (rows[0].work_dir / "target").mkdir(parents=True)
    (rows[0].work_dir / "target" / "points.json").write_text("{}", encoding="utf-8")

    def stale(*a, **k):
        raise StageError("titik frame 0 tidak bisa dipakai (f0: basi)")
    monkeypatch.setattr(isolate_stage, "run_isolate", stale)
    batch.plan(rows, env["cfg"], args_of())
    assert rows[0].status == "perlu tindakan" and rows[0].target == "BASI"
    assert any("pick ulang" in a for a in rows[0].actions)


def test_identitas_basi_perlu_tindakan_dengan_perintah_restart(env, monkeypatch):
    monkeypatch.setattr(batch, "_ingest_quiet", lambda r, c, a: fake_ingest(2)(r, c, a))
    rows = make_rows(env, ["klip4.mp4"])
    seg = rows[0].work_dir / "seg"
    seg.mkdir(parents=True)
    (seg / "manifest.json").write_text(json.dumps({"clip": {"meta_sha256": "0" * 64, "source_path": "x"},
                                                   "model": env["cfg"].segment.model, "num_labels": 29}), encoding="utf-8")
    batch.plan(rows, env["cfg"], args_of())
    r = rows[0]
    assert r.status == "perlu tindakan" and r.identity == "BASI" and r.gpu_s == 0
    (cmd,) = [a for a in r.actions if "restart-from ingest" in a]
    assert cmd.endswith("--restart-from ingest --yes") and "rotoscope run" in cmd


def test_crop_belum_diputuskan_perlu_tindakan(env, monkeypatch):
    monkeypatch.setattr(batch, "_ingest_quiet", lambda r, c, a: fake_ingest(2)(r, c, a))
    rows = make_rows(env, ["klip4.mp4"])
    rows[0].crop_dir.mkdir(parents=True)
    (rows[0].crop_dir / crop_mod.SARAN_FILENAME).write_text(
        json.dumps({"format": crop_mod.FORMAT_SARAN}), encoding="utf-8")
    batch.plan(rows, env["cfg"], args_of())
    assert rows[0].status == "perlu tindakan" and rows[0].crop == "BELUM diputuskan"
    assert sum(a.startswith("crop: ") for a in rows[0].actions) == 3


def test_klip_gagal_tidak_menghentikan_rencana(env, monkeypatch):
    def ingest(r, c, a):
        if "klip4" in r.name:
            raise batch.BatchError("[1] ingest gagal (exit 1): ffmpeg")
        fake_ingest(2)(r, c, a)
    monkeypatch.setattr(batch, "_ingest_quiet", ingest)
    rows = make_rows(env, ["klip4.mp4", "klip5.mp4"])
    batch.plan(rows, env["cfg"], args_of())
    assert [r.status for r in rows] == ["gagal", "siap"]
    assert batch.plan_exit(rows, batch.disk_check(rows, env["cfg"])) == 1


def test_disk_kurang_exit_1(env, monkeypatch):
    monkeypatch.setattr(batch, "free_bytes", lambda p: 1 * 2**30)        # 1 GiB bebas, cadangan 5 GiB
    monkeypatch.setattr(batch, "_ingest_quiet", lambda r, c, a: fake_ingest(3)(r, c, a))
    rows = make_rows(env, ["klip4.mp4"])
    batch.plan(rows, env["cfg"], args_of())
    need = batch.disk_check(rows, env["cfg"])
    assert not batch.disk_ok(need) and batch.plan_exit(rows, need) == 1


def test_disk_hanya_berkas_yang_belum_ada(env, monkeypatch):
    """E: klip yang sudah lengkap (tanpa frame GPU, manifest hilir ada) → disk baru 0."""
    monkeypatch.setattr(batch, "_ingest_quiet", lambda r, c, a: fake_ingest(3)(r, c, a))
    rows = make_rows(env, ["klip4.mp4"])
    fake = rows[0].work_dir
    for sub in ("stable", "contours", "strokes"):
        (fake / sub).mkdir(parents=True, exist_ok=True)
        (fake / sub / "manifest.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(batch, "_seg_depth_status", lambda r, c, a: (0.0, False))
    batch.plan(rows, env["cfg"], args_of())
    assert rows[0].disk_mib == 0.0


def test_yaml_batch_defaults_and_validation(env, tmp_path):
    b = env["cfg"].batch
    assert b.disk_mib_per_mpx == 8.5 and b.disk_reserve_gib == 5.0 and ".mp4" in b.video_exts
    bad = tmp_path / "bad.yaml"
    bad.write_text("batch:\n  video_exts: [mp4]\n", encoding="utf-8")
    with pytest.raises(Exception, match="video_exts"):
        load_pipeline(bad)


# ── Jalan (proses anak dipalsukan) ────────────────
def run_exec(env, rows, args, monkeypatch, script):
    monkeypatch.setattr(batch, "_ingest_quiet", lambda r, c, a: fake_ingest(2)(r, c, a))
    launched = []
    monkeypatch.setattr(batch, "_popen", popen_factory(script, launched))
    hint = "python -m rotoscope batch"
    code = batch.execute(rows, env["cfg"], args, hint, "20990101T000000Z")
    return code, launched


def summary(env):
    (p,) = sorted((env["work"] / "batch").glob("ringkasan_*.json"))
    return json.loads(p.read_text(encoding="utf-8"))


def test_jalan_perlu_tindakan_berhenti_sebelum_gpu_exit_4(env, monkeypatch):
    rows = make_rows(env, ["klip4.mp4", "klip5.mp4"])
    (rows[0].crop_dir).mkdir(parents=True)
    (rows[0].crop_dir / crop_mod.SARAN_FILENAME).write_text(json.dumps({"format": crop_mod.FORMAT_SARAN}), encoding="utf-8")
    code, launched = run_exec(env, rows, args_of(yes=True), monkeypatch, [0, 0])
    assert code == 4 and launched == []
    s = summary(env)
    assert s["exit_code"] == 4 and s["counts"] == {"belum dijalankan": 1, "dilewati": 1}


def test_skip_pending_jalan_dan_exit_4(env, monkeypatch):
    rows = make_rows(env, ["klip4.mp4", "klip5.mp4"])
    rows[0].crop_dir.mkdir(parents=True)
    (rows[0].crop_dir / crop_mod.SARAN_FILENAME).write_text(json.dumps({"format": crop_mod.FORMAT_SARAN}), encoding="utf-8")
    code, launched = run_exec(env, rows, args_of(yes=True, skip_pending=True), monkeypatch, [0])
    assert code == 4 and len(launched) == 1 and launched[0][-1] == "--yes"
    assert [r.status for r in rows] == ["dilewati", "selesai"]


def test_child_argv_has_yes_trim_config_style(env, monkeypatch):
    trim = {"start_s": 0.0, "end_s": 1.0, "start_frame": 0, "end_frame": 24}
    rows = make_rows(env, ["klip4.mp4"], trim=trim)
    code, launched = run_exec(env, rows, args_of(yes=True, config=env["conf"], style="clean-line", seg_model="0.4b"),
                              monkeypatch, [0])
    argv = launched[0]
    assert argv[1:4] == ["-m", "rotoscope", "run"] and "--yes" == argv[-1]
    assert argv[argv.index("--end") + 1] == "1" and "--config" in argv and argv[argv.index("--style") + 1] == "clean-line"
    assert argv[argv.index("--seg-model") + 1] == "0.4b"


def test_gagal_lalu_lanjut_exit_1(env, monkeypatch):
    rows = make_rows(env, ["klip4.mp4", "klip5.mp4"])
    code, launched = run_exec(env, rows, args_of(yes=True), monkeypatch, [1, 0])
    assert len(launched) == 2 and [r.status for r in rows] == ["gagal", "selesai"]
    assert code == 1 and summary(env)["counts"] == {"gagal": 1, "selesai": 1}


def test_gagal_dan_dilewati_sekaligus_exit_1(env, monkeypatch):
    rows = make_rows(env, ["klip4.mp4", "klip5.mp4"])
    rows[1].crop_dir.mkdir(parents=True)
    (rows[1].crop_dir / crop_mod.SARAN_FILENAME).write_text(json.dumps({"format": crop_mod.FORMAT_SARAN}), encoding="utf-8")
    code, _ = run_exec(env, rows, args_of(yes=True, skip_pending=True), monkeypatch, [1])
    assert code == 1


def test_ctrl_c_stops_batch_writes_summary_exit_130(env, monkeypatch):
    rows = make_rows(env, ["klip4.mp4", "klip5.mp4"])
    stopped = []
    monkeypatch.setattr(cli, "_stop", lambda proc: stopped.append(proc))
    code, launched = run_exec(env, rows, args_of(yes=True), monkeypatch, ["ctrlc", 0])
    assert code == 130 and len(launched) == 1 and len(stopped) == 1
    assert [r.status for r in rows] == ["dihentikan", "belum dijalankan"]
    assert summary(env)["exit_code"] == 130


def test_summary_only_for_jalan_and_has_format(env, monkeypatch):
    rows = make_rows(env, ["klip4.mp4"])
    batch.plan(rows, env["cfg"], args_of())
    assert not (env["work"] / "batch").exists()           # rencana tidak menulis ringkasan
    code, _ = run_exec(env, make_rows(env, ["klip4.mp4"]), args_of(yes=True), monkeypatch, [0])
    s = summary(env)
    assert code == 0 and s["format"] == batch.SUMMARY_FORMAT and s["klip"][0]["status"] == "selesai"


# ── VRAM (H) ──────────────────────────────────────
def test_vram_kurang_exit_1_dengan_pesan(env, monkeypatch):
    monkeypatch.setattr(batch, "gpu_memory_mib", lambda: (2965.0, 4096.0))
    with pytest.raises(batch.BatchError, match="Tutup browser"):
        batch.check_vram(env["cfg"])


def test_vram_ambang_sama_dengan_segment(env, monkeypatch):
    monkeypatch.setattr(batch, "gpu_memory_mib", lambda: (3300.0, 4096.0))
    batch.check_vram(env["cfg"])                           # tepat ambang = lolos
    monkeypatch.setattr(batch, "gpu_memory_mib", lambda: (3299.0, 4096.0))
    with pytest.raises(batch.BatchError):
        batch.check_vram(env["cfg"])


def test_vram_nvidia_smi_gagal_peringatan_lanjut(env, monkeypatch, capsys):
    monkeypatch.setattr(batch, "gpu_memory_mib", lambda: None)
    batch.check_vram(env["cfg"])
    assert "PERINGATAN" in capsys.readouterr().out


def test_vram_hanya_bila_ada_klip_gpu(env, monkeypatch):
    monkeypatch.setattr(batch, "gpu_memory_mib", lambda: pytest.fail("tidak boleh dicek bila tanpa GPU"))
    monkeypatch.setattr(batch, "_ingest_quiet", lambda r, c, a: fake_ingest(2)(r, c, a))
    rows = make_rows(env, ["klip4.mp4"])
    monkeypatch.setattr(batch, "_seg_depth_status", lambda r, c, a: (0.0, False))
    monkeypatch.setattr(batch, "_popen", lambda argv: FakeProc(0))
    batch.execute(rows, env["cfg"], args_of(yes=True), "h", "20990101T000000Z")
    assert rows[0].status == "selesai"


def test_jalan_vram_kurang_berhenti_sebelum_klip_pertama(env, monkeypatch):
    monkeypatch.setattr(batch, "gpu_memory_mib", lambda: (2965.0, 4096.0))
    rows = make_rows(env, ["klip4.mp4"])
    code, launched = run_exec(env, rows, args_of(yes=True), monkeypatch, [0])
    assert code == 1 and launched == []                  # tidak ada proses anak yang dijalankan
    assert rows[0].status == "belum dijalankan"          # ringkasan tetap dicetak (keputusan Rio no. 1)


# ── Ringkasan + konsistensi keputusan ─────────────
def test_exit_codes_plan(env, monkeypatch):
    monkeypatch.setattr(batch, "_ingest_quiet", lambda r, c, a: fake_ingest(2)(r, c, a))
    rows = make_rows(env, ["klip4.mp4"])
    batch.plan(rows, env["cfg"], args_of())
    assert batch.plan_exit(rows, batch.disk_check(rows, env["cfg"])) == 0
    rows[0].status = "perlu tindakan"
    assert batch.plan_exit(rows, batch.disk_check(rows, env["cfg"])) == 4
    rows[0].status = "gagal"
    assert batch.plan_exit(rows, batch.disk_check(rows, env["cfg"])) == 1


def test_run_child_is_only_gpu_path_and_always_yes(env):
    """Setiap perintah anak memuat --yes (GPU) dan berbentuk `run` (bukan stage GPU langsung)."""
    rows = make_rows(env, ["klip4.mp4"])
    argv = batch.child_argv(rows[0], args_of())
    assert argv[3] == "run" and argv[-1] == "--yes"


def test_no_forbidden_flag_in_child_argv(env):
    rows = make_rows(env, ["klip4.mp4"])
    for bad in batch.FORBIDDEN_FLAGS:
        assert bad not in batch.child_argv(rows[0], args_of())


def test_render_plan_shows_reasons_and_next_step(env, monkeypatch):
    monkeypatch.setattr(batch, "_ingest_quiet", lambda r, c, a: fake_ingest(2)(r, c, a))
    rows = make_rows(env, ["klip4.mp4"])
    rows[0].crop_dir.mkdir(parents=True)
    (rows[0].crop_dir / crop_mod.SARAN_FILENAME).write_text(json.dumps({"format": crop_mod.FORMAT_SARAN}), encoding="utf-8")
    batch.plan(rows, env["cfg"], args_of())
    text = batch.render_plan(rows, "python -m rotoscope batch x", batch.disk_check(rows, env["cfg"]), 4)
    assert "Alasan perlu tindakan:" in text and "saran crop belum diputuskan" in text
    assert "Setelah tindakan di atas, jalankan rencana ulang" in text and "--yes" not in text.split("Keluar")[-1]


def test_cli_dispatch_forbidden_flag_exit_1(env, monkeypatch):
    """Dispatch di cli.main terjadi SEBELUM argparse cli (flag terlarang → exit 1, bukan exit 2 argparse)."""
    from rotoscope import cli as cli_main
    monkeypatch.setattr(batch, "plan", lambda *a, **k: pytest.fail("rencana tidak boleh jalan"))
    assert cli_main.main(["batch", "--limit", "3", str(env["vids"] / "klip4.mp4")]) == 1


def _pending_rows(env):
    rows = make_rows(env, ["klip4.mp4", "klip5.mp4"])
    rows[0].crop_dir.mkdir(parents=True)
    (rows[0].crop_dir / crop_mod.SARAN_FILENAME).write_text(json.dumps({"format": crop_mod.FORMAT_SARAN}), encoding="utf-8")
    return rows


def test_mode_yes_ringkasan_dicetak_tanpa_baris_keluar(env, monkeypatch, capsys):
    """Keputusan Rio no. 1 + 2: tabel rencana (mode --yes) tanpa "Keluar"/"Untuk menjalankan"; ringkasan akhir dicetak."""
    rows = _pending_rows(env)
    code, _ = run_exec(env, rows, args_of(yes=True, skip_pending=True), monkeypatch, [0])
    out = capsys.readouterr().out
    assert code == 4
    assert "Keluar:" not in out and "Untuk menjalankan" not in out
    assert "Mulai: 1 klip dijalankan, 1 dilewati (--skip-pending)" in out
    assert "RINGKASAN batch" in out and "Exit 4 = ada klip dilewati" in out
    assert "tindakan: crop: " in out and "Ringkasan JSON: " in out and "output MP4" in out


def test_mulai_tanpa_klip_dilewati_tanpa_skip_pending_note(env, monkeypatch, capsys):
    """Perbaikan (a) Rio: "(--skip-pending)" hanya bila ada klip yang dilewati."""
    rows = make_rows(env, ["klip4.mp4", "klip5.mp4"])
    code, _ = run_exec(env, rows, args_of(yes=True), monkeypatch, [0, 0])
    out = capsys.readouterr().out
    assert code == 0
    assert "Mulai: 2 klip dijalankan, 0 dilewati" in out and "--skip-pending" not in out.split("RINGKASAN")[0]


def test_pre_gpu_stop_prints_summary_exit_4(env, monkeypatch, capsys):
    rows = _pending_rows(env)
    code, launched = run_exec(env, rows, args_of(yes=True), monkeypatch, [0])
    out = capsys.readouterr().out
    assert code == 4 and launched == []
    assert "Berhenti SEBELUM GPU" in out and "RINGKASAN batch" in out and "Exit 4 = " in out
    assert "Mulai:" not in out and "Keluar:" not in out


def test_gpu_dan_disk_tanda_strip_untuk_klip_dilewati(env, monkeypatch, capsys):
    rows = _pending_rows(env)
    run_exec(env, rows, args_of(yes=True, skip_pending=True), monkeypatch, [0])
    table = [ln for ln in capsys.readouterr().out.splitlines() if ln.lstrip()[:1].isdigit() and "klip4" in ln]
    assert table and table[0].rstrip().endswith("dilewati")
    assert table[0].count("—") == 2                       # kolom GPU DAN kolom disk baru = "—" (keputusan Rio no. 2)


def test_ctrl_c_ringkasan_exit_130(env, monkeypatch, capsys):
    rows = make_rows(env, ["klip4.mp4", "klip5.mp4"])
    monkeypatch.setattr(cli, "_stop", lambda proc: None)
    code, _ = run_exec(env, rows, args_of(yes=True), monkeypatch, ["ctrlc", 0])
    out = capsys.readouterr().out
    assert code == 130 and "Exit 130 = dihentikan (Ctrl+C)" in out and "RINGKASAN batch" in out
    assert "Traceback" not in out


def test_rencana_tanpa_yes_tetap_keluar_dan_untuk_menjalankan(env):
    rows = make_rows(env, ["klip4.mp4"])
    batch.plan(rows, env["cfg"], args_of())
    text = batch.render_plan(rows, "python -m rotoscope batch x", batch.disk_check(rows, env["cfg"]), 0)
    assert "Keluar: exit 0" in text and text.rstrip().endswith("--yes")


def test_fmt_comma_decimal():
    assert batch.fmt(1.25, 2) == "1,25" and batch.fmt(3.0, 0) == "3"
