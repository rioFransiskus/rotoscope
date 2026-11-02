"""Test mask_preview.py (T-609, `pick --mask-preview`): SAM TIRUAN, tanpa GPU / model / torch.

Nomor kelompok (M1)–(M17) = rencana test docs/04 "Hasil + rancangan T-609 Tahap 1" no. 6 (+ penjaga device / skala bar).
Test bobot asli di CPU (skip default): ROTOSCOPE_SAM_CPU_TESTS=1 (butuh SAM 2.1 tiny di cache HF).
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from rotoscope import isolate as iso
from rotoscope import mask_preview as mp
from rotoscope import pick
from rotoscope import stage_common
from rotoscope.config import load_pipeline
from rotoscope.ingest import FRAME_PATTERN
from rotoscope.stage_common import StageError, StageInterrupted, clip_identity, load_frame_list

W, H, N = 96, 64, 6
R = 14.0                                                   # jari-jari masker tiruan


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    monkeypatch.setenv(stage_common.HF_OFFLINE_ENV, "1")


# ── Klip + titik sintetis ──────────────────────────
def frame_rgb(i: int) -> np.ndarray:
    y, x = np.indices((H, W))
    return np.stack([(x * 2 + i * 13) % 200 + 20, (y * 3 + i * 7) % 200 + 20, np.full_like(x, 90 + i)], -1).astype(np.uint8)


def make_clip(root: Path, *, trim_start: int | None = None, fps: int = 24) -> Path:
    work = root / "work" / "clips" / "klip"
    (work / "frames").mkdir(parents=True)
    for i in range(N):
        ok, buf = cv2.imencode(".png", cv2.cvtColor(frame_rgb(i), cv2.COLOR_RGB2BGR))
        assert ok
        (work / "frames" / (FRAME_PATTERN % i)).write_bytes(buf.tobytes())
    meta = {"source_path": str((root / "klip.mp4").resolve()), "frame_count": N, "working_width": W, "working_height": H,
            "frame_index_start": 0, "target_fps": fps}
    if trim_start is not None:
        meta["trim"] = {"start_s": trim_start / fps, "end_s": None, "start_frame": trim_start, "end_frame": None}
    (work / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return work


def write_points(work: Path, spec: dict) -> Path:
    """spec = {frame: [(id, [(x, y, label), …], box | None), …]}."""
    _, _, w, h = load_frame_list(work)
    frames = [{"frame_index": f, "frame_file": FRAME_PATTERN % f,
               "frame_pixel_sha256": pick.frame_pixel_sha256(work / "frames" / (FRAME_PATTERN % f)),
               "updated_utc": "2026-10-29T00:00:00+00:00", "display": {},
               "targets": [{"id": tid, "points": [{"x": float(x), "y": float(y), "label": lab} for x, y, lab in pts],
                            "box": [float(v) for v in box] if box is not None else None} for tid, pts, box in targets]}
              for f, targets in sorted(spec.items())]
    doc = pick.new_doc(clip_identity(work), w, h, frames)
    pick.validate_doc(doc)
    path = iso.points_path(work)
    path.parent.mkdir(parents=True, exist_ok=True)
    stage_common.write_json_atomic(path, doc)
    return path


def cfg_for(work: Path, **over):
    return load_pipeline(overrides={"paths.work_dir": str(work), **over})


def tree(d: Path) -> dict[str, bytes]:
    return {p.relative_to(d).as_posix(): p.read_bytes() for p in sorted(d.rglob("*")) if p.is_file()} if d.exists() else {}


def prev_dir(work: Path) -> Path:
    return work / "target" / "pratinjau"


def summary(work: Path) -> dict:
    return json.loads((prev_dir(work) / mp.SUMMARY_FILENAME).read_text(encoding="utf-8"))


# ── SAM tiruan: lingkaran di sekitar klik kiri pertama (atau pusat kotak) ──
class FakeEngine:
    def __init__(self, b, frames, size, obj_id):
        self.b, self.frames, (self.h, self.w), self.obj = b, list(frames), size, obj_id
        self.prompt = None
        b.sessions.append((Path(frames[0]).name, obj_id, len(frames)))

    def add_prompt(self, k, prompt):
        assert k == 0
        self.prompt = prompt
        self.b.prompts.append((self.obj, prompt))

    def run_frame(self, k):
        f = self.prompt.index
        if self.b.interrupt == (f, self.obj):
            raise KeyboardInterrupt
        if (f, self.obj) in self.b.empty:
            return np.full((self.h, self.w), -5.0, np.float32), -3.5
        pos = [xy for xy, lab in zip(self.prompt.points, self.prompt.labels) if lab == 1]
        cx, cy = pos[0] if pos else ((self.prompt.box[0] + self.prompt.box[2]) / 2, (self.prompt.box[1] + self.prompt.box[3]) / 2)
        yy, xx = np.indices((self.h, self.w), dtype=np.float32)
        return (R - np.hypot(xx - cx, yy - cy)).astype(np.float32), 7.25

    def release(self):
        self.b.released += 1


class FakeBackend:
    def __init__(self, cfg, *, empty=(), interrupt=None, device="cpu"):
        self.cfg, self.empty, self.interrupt, self.device = cfg, set(empty), interrupt, device
        self.sessions: list = []
        self.prompts: list = []
        self.released = self.closed = 0

    def describe(self):
        i = self.cfg.isolate
        return iso.ModelInfo(i.model_id, i.revision, "apache-2.0", i.precision, {"class": "Palsu"})

    def open(self):
        return {"device": self.device, "load_s": 0.0}

    def session(self, frames, size, obj_id):
        return FakeEngine(self, frames, size, obj_id)

    def stats(self):
        return {}

    def close(self):
        self.closed += 1


def factory(**kw):
    made: list[FakeBackend] = []

    def f(cfg):
        made.append(FakeBackend(cfg, **kw))
        return made[-1]
    f.made = made
    return f


def preview(work: Path, fac=None, logs: list | None = None, **kw) -> dict:
    over = kw.pop("over", {})
    kw.setdefault("ram_probe", lambda: None)
    kw.setdefault("cuda_probe", lambda: False)
    return mp.run_preview(cfg_for(work, **over), backend_factory=fac or factory(),
                          log=(logs.append if logs is not None else (lambda m: None)), **kw)


A = [(20, 20, 1), (40, 20, 1), (70, 40, 0)]                # target 1 (titik kiri pertama = pusat masker)
B = [(70, 30, 1), (20, 20, 0)]                             # target 2
BOX = (5, 5, 60, 50)


# ── (M1) PNG per (frame, target) + ringkasan ───────
def test_m1_png_per_frame_target_and_summary(tmp_path):
    work = make_clip(tmp_path)
    ppath = write_points(work, {1: [(1, A, BOX)], 3: [(1, A, None), (2, B, None)]})
    logs: list[str] = []
    res = preview(work, logs=logs)
    names = sorted(p.name for p in prev_dir(work).iterdir())
    assert names == ["klik_f00001_t1.png", "klik_f00003_t1.png", "klik_f00003_t2.png", "pratinjau.json"]
    s = summary(work)
    assert s == res and s["format"] == "rotoscope-mask-preview/1" and s["device"] == "cpu"
    assert s["points_sha256"] == hashlib.sha256(ppath.read_bytes()).hexdigest() and s["clip"] == clip_identity(work)
    assert s["neg_points_warn_max"] == 10 and s["mask_threshold"] == 0.0 and "BUKAN hasil pelacakan" in s["note"]
    assert s["run"]["cuda_initialized"] is False and s["model"]["revision"] == load_pipeline().isolate.revision
    f1 = s["frames"][0]
    assert f1["frame"] == 1 and f1["frame_pixel_sha256"] == pick.frame_pixel_sha256(work / "frames" / "frame_00001.png")
    t = f1["targets"][0]
    area = int(((R - np.hypot(*np.meshgrid(np.arange(W) - 20.0, np.arange(H) - 20.0))) > 0).sum())
    assert (t["id"], t["kiri"], t["kanan"], t["kotak"], t["skor_objek"], t["luas_px"]) == (1, 2, 1, True, 7.25, area)
    assert t["porsi_frame"] == round(area / (W * H), 4) and t["komponen_8arah"] == 1 and t["porsi_komponen_terbesar"] == 1.0
    assert [x["id"] for x in s["frames"][1]["targets"]] == [1, 2]
    img = cv2.imdecode(np.fromfile(prev_dir(work) / "klik_f00001_t1.png", np.uint8), cv2.IMREAD_COLOR)
    assert img.shape == (H + mp.BAR_H, W, 3)
    photo = cv2.cvtColor(frame_rgb(1), cv2.COLOR_RGB2BGR).astype(float)
    inside = img[mp.BAR_H + 20, 20 + 10].astype(float)                   # di dalam masker, jauh dari titik / tepi / kotak
    blend = photo[20, 30] * (1 - mp.MASK_ALPHA) + np.array(mp.MASK_BGR) * mp.MASK_ALPHA
    assert np.abs(inside - blend).max() <= 1.0
    assert (img[mp.BAR_H + 60, 90] == photo[60, 90].astype(np.uint8)).all()        # di luar masker / kotak = FOTO
    assert (img[mp.BAR_H + 5, 30] == np.array(mp.BOX_BGR, np.uint8)).all()          # tepi atas kotak kuning
    assert (img[2, :] == mp.BAR_BG).all()                                            # bar teks di atas
    assert any("f1 target 1: skor objek +7,25" in m for m in logs) and any("Buka setiap PNG" in m for m in logs)


# ── (M2) komponen 8-arah ───────────────────────────
def test_m2_components_8_connected_largest_share_of_mask():
    m = np.zeros((10, 10), bool)
    m[1, 1] = m[2, 2] = True                                              # sentuh diagonal = 1 komponen (8-arah)
    m[6:9, 6:9] = True                                                    # 9 piksel
    assert mp.mask_components(m) == (2, 9 / 11)
    assert mp.mask_components(np.zeros((4, 4), bool)) == (0, 0.0)


# ── (M3) CPU dipaksa ───────────────────────────────
@pytest.mark.parametrize("value", ["0", None, ""])
def test_m3_refuses_without_cpu_env_before_backend(tmp_path, monkeypatch, value):
    work = make_clip(tmp_path)
    write_points(work, {1: [(1, A, None)]})
    if value is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", value)
    fac = factory()
    with pytest.raises(StageError, match="hanya berjalan di CPU"):
        preview(work, fac)
    assert fac.made == [] and not (work / "target" / "pratinjau.partial").exists() and not prev_dir(work).exists()


def test_m3b_subprocess_without_env_exits_1_and_never_imports_torch(tmp_path):
    work = make_clip(tmp_path)
    write_points(work, {1: [(1, A, None)]})
    code = ("import sys, rotoscope.mask_preview as m; rc = m.main(['--work-dir', sys.argv[1]]); "
            "print('TORCH' if 'torch' in sys.modules else 'TANPA_TORCH'); sys.exit(rc)")
    env = {k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"}
    p = subprocess.run([sys.executable, "-c", code, str(work)], capture_output=True, text=True, env=env)
    assert p.returncode == 1 and "TANPA_TORCH" in p.stdout and "CUDA_VISIBLE_DEVICES" in p.stderr
    assert not prev_dir(work).exists()


def test_m3c_cpu_backend_and_device_guard(tmp_path):
    work = make_clip(tmp_path)
    write_points(work, {1: [(1, A, None)]})
    assert mp.cpu_backend(cfg_for(work)).device == "cpu"
    with pytest.raises(StageError, match="bukan 'cpu'"):
        preview(work, factory(device="cuda"))
    assert not prev_dir(work).exists()


# ── (M4) torch.cuda terinisialisasi → hasil tidak dipakai ──
def test_m4_cuda_initialized_keeps_old_folder(tmp_path):
    work = make_clip(tmp_path)
    write_points(work, {1: [(1, A, None)]})
    prev_dir(work).mkdir(parents=True)
    (prev_dir(work) / "lama.png").write_bytes(b"lama")
    with pytest.raises(StageError, match="torch.cuda terinisialisasi"):
        preview(work, cuda_probe=lambda: True)
    assert tree(prev_dir(work)) == {"lama.png": b"lama"}
    assert mp.torch_cuda_initialized() is ("torch" in sys.modules and sys.modules["torch"].cuda.is_initialized())


# ── (M5) tukar atomik + sisa run terputus ──────────
def test_m5_folder_replaced_whole_and_interrupt_keeps_old(tmp_path):
    work = make_clip(tmp_path)
    write_points(work, {1: [(1, A, None)], 3: [(1, A, None)]})
    preview(work)
    assert (prev_dir(work) / "klik_f00001_t1.png").is_file()
    write_points(work, {3: [(1, A, None)]})
    preview(work)
    assert sorted(p.name for p in prev_dir(work).iterdir()) == ["klik_f00003_t1.png", "pratinjau.json"]   # PNG basi hilang
    old = tree(prev_dir(work))
    write_points(work, {2: [(1, A, None)], 3: [(1, A, None)]})
    with pytest.raises(StageInterrupted):
        preview(work, factory(interrupt=(3, 1)))
    assert tree(prev_dir(work)) == old and (work / "target" / mp.PARTIAL_DIRNAME).is_dir()
    assert mp.main(["--work-dir", str(work)], backend_factory=factory(interrupt=(2, 1))) == 130
    assert tree(prev_dir(work)) == old
    preview(work)
    assert not (work / "target" / mp.PARTIAL_DIRNAME).exists() and not (work / "target" / mp.OLD_DIRNAME).exists()
    assert sorted(p.name for p in prev_dir(work).iterdir()) == ["klik_f00002_t1.png", "klik_f00003_t1.png", "pratinjau.json"]


# ── (M6)–(M7) prasyarat ────────────────────────────
def test_m6_stale_frame_refused_nothing_written(tmp_path):
    work = make_clip(tmp_path)
    write_points(work, {1: [(1, A, None)], 4: [(1, A, None)]})
    img = frame_rgb(4)
    img[0, 0] = (1, 2, 3)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    (work / "frames" / "frame_00004.png").write_bytes(buf.tobytes())
    fac = factory()
    with pytest.raises(StageError, match=r"f4: basi.*--frame 4"):
        preview(work, fac)
    assert fac.made == [] and not (work / "target" / mp.PARTIAL_DIRNAME).exists()


def test_m7_without_points_or_without_frames(tmp_path):
    work = make_clip(tmp_path)
    with pytest.raises(StageError, match="belum ada titik"):
        preview(work)
    path = write_points(work, {1: [(1, A, None)]})
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["frames"] = []
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(StageError, match="tanpa frame berklik"):
        preview(work)
    assert not (work / "target" / mp.PARTIAL_DIRNAME).exists()


# ── (M8) isolate/, titik, kunci [1b] tidak tersentuh ──
def test_m8_isolate_points_and_key_untouched(tmp_path):
    import test_isolate as ti

    work, _ = ti.make_clip(tmp_path, n=5)
    ti.write_points(work, {1: [(1, ti.P, None)]})
    ti.run(work)
    (work / "target" / "points.prev.json").write_text("{}", encoding="utf-8")
    before = {d: tree(work / d) for d in ("isolate", "frames")} | {"target": tree(work / "target")}
    meta_before = (work / "meta.json").read_bytes()
    preview(work)
    assert {d: tree(work / d) for d in ("isolate", "frames")} == {d: before[d] for d in ("isolate", "frames")}
    target_after = tree(work / "target")
    assert {k: v for k, v in target_after.items() if not k.startswith("pratinjau/")} == before["target"]
    assert (work / "meta.json").read_bytes() == meta_before
    assert iso.run_isolate(ti.cfg_for(work), dry_run=True, log=lambda m: None)["status"] == "up-to-date"   # kunci [1b] tetap


# ── (M9)–(M11) isi ─────────────────────────────────
def test_m9_seconds_use_trim_start(tmp_path):
    work = make_clip(tmp_path, trim_start=54)
    write_points(work, {2: [(1, A, None)]})
    assert preview(work)["frames"][0]["detik"] == round(56 / 24, 3)
    work2 = make_clip(tmp_path / "b")
    write_points(work2, {2: [(1, A, None)]})
    assert preview(work2)["frames"][0]["detik"] == round(2 / 24, 3)


def test_m10_empty_mask_warned_png_still_written(tmp_path):
    work = make_clip(tmp_path)
    write_points(work, {1: [(1, A, None)], 2: [(1, A, None)]})
    logs: list[str] = []
    s = preview(work, factory(empty={(2, 1)}), logs)
    t2 = s["frames"][1]["targets"][0]
    assert t2["luas_px"] == 0 and t2["skor_objek"] == -3.5 and t2["komponen_8arah"] == 0 and "masker kosong" in t2["peringatan"][0]
    assert s["frames"][0]["targets"][0]["peringatan"] == [] and any(w.startswith("f2 target 1: masker kosong") for w in s["peringatan"])
    assert any(m.startswith("PERINGATAN: f2 target 1: masker kosong") for m in logs)
    assert (prev_dir(work) / "klik_f00002_t1.png").is_file()


def test_m11_neg_points_warning_uses_yaml_key(tmp_path):
    work = make_clip(tmp_path)
    many = [(20, 20, 1)] + [(80, 10 + k, 0) for k in range(4)]
    write_points(work, {1: [(1, many, None)]})
    logs: list[str] = []
    s = preview(work, logs=logs)                                          # default 10: 4 kanan → tanpa peringatan
    assert s["neg_points_warn_max"] == 10 and not any("klik kanan" in m for m in logs) and s["peringatan"] == []
    logs.clear()
    s = preview(work, logs=logs, over={"isolate.neg_points_warn_max": 3})
    assert s["neg_points_warn_max"] == 3 and s["frames"][0]["targets"][0]["peringatan"] == ["klik kanan > 3"]
    warn = [m for m in logs if m.startswith("PERINGATAN: klik kanan lebih dari 3 per target: f1 target 1 (4)")]
    assert warn and "--mask-preview" not in warn[0]                       # di pratinjau sendiri: tanpa saran pratinjau


# ── (M12) multi-target: sesi + prompt per target ───
def test_m12_one_session_per_frame_target_with_own_prompt(tmp_path):
    work = make_clip(tmp_path)
    write_points(work, {3: [(1, A, BOX), (2, B, None)], 5: [(2, B, None)]})
    fac = factory()
    preview(work, fac)
    b = fac.made[0]
    assert b.sessions == [("frame_00003.png", 1, 1), ("frame_00003.png", 2, 1), ("frame_00005.png", 2, 1)]
    assert [(obj, p.index, p.points, p.labels, p.box) for obj, p in b.prompts] == [
        (1, 3, tuple((float(x), float(y)) for x, y, _ in A), tuple(lab for *_, lab in A), tuple(float(v) for v in BOX)),
        (2, 3, tuple((float(x), float(y)) for x, y, _ in B), tuple(lab for *_, lab in B), None),
        (2, 5, tuple((float(x), float(y)) for x, y, _ in B), tuple(lab for *_, lab in B), None)]
    assert b.released == 3 and b.closed == 1


# ── (M13) RAM ──────────────────────────────────────
def test_m13_low_ram_stops_before_model(tmp_path):
    work = make_clip(tmp_path)
    write_points(work, {1: [(1, A, None)]})
    fac = factory()
    with pytest.raises(StageError, match="RAM bebas 100 MiB"):
        preview(work, fac, ram_probe=lambda: 100.0)
    assert fac.made == [] and not prev_dir(work).exists()
    preview(work, fac, ram_probe=lambda: 100.0, over={"isolate.ram_min_free_mib": 64})
    assert prev_dir(work).is_dir()


# ── (M15)–(M17) main + bar ─────────────────────────
def test_m15_main_exit_codes(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(iso, "ram_free_mib", lambda: None)
    work = make_clip(tmp_path)
    assert mp.main(["--work-dir", str(work)], backend_factory=factory()) == 1
    assert "belum ada titik" in capsys.readouterr().err
    write_points(work, {1: [(1, A, None)]})
    assert mp.main(["--work-dir", str(work)], backend_factory=factory()) == 0
    out = capsys.readouterr().out
    assert "pratinjau masker (CPU" in out and "Frame di antara klik tidak diperiksa" in out and prev_dir(work).is_dir()
    monkeypatch.setattr(iso, "ram_free_mib", lambda: 10.0)
    assert mp.main(["--work-dir", str(work)], backend_factory=factory()) == 1
    assert "RAM bebas 10 MiB" in capsys.readouterr().err
    bad = tmp_path / "bad.yaml"
    bad.write_text("isolate:\n  neg_points_warn_max: -1\n", encoding="utf-8")
    assert mp.main(["--work-dir", str(work), "--config", str(bad)], backend_factory=factory()) == 1
    assert "neg_points_warn_max" in capsys.readouterr().err


def test_m17_bar_text_shrinks_to_fit():
    long = "frame 123 (detik 12,34)  target 2  skor objek +18,40  masker 17,6% frame"
    assert mp.bar_scale(long, 2000) == mp.FONT_SCALE
    sc = mp.bar_scale(long, 400)
    assert mp.FONT_SCALE_MIN <= sc < mp.FONT_SCALE
    assert sc == mp.FONT_SCALE_MIN or cv2.getTextSize(long, mp.FONT, sc, 2)[0][0] <= 400 - 2 * mp.BAR_PAD
    assert mp.bar_scale(long, 50) == mp.FONT_SCALE_MIN


# ── (M14) bobot asli di CPU (opt-in) ───────────────
@pytest.mark.skipif(os.environ.get("ROTOSCOPE_SAM_CPU_TESTS") != "1", reason="set ROTOSCOPE_SAM_CPU_TESTS=1 (SAM 2.1 tiny di cache)")
def test_m14_real_sam_cpu(tmp_path):
    import test_isolate as ti

    work = ti.square_clip(tmp_path, n=3)
    meta = json.loads((work / "meta.json").read_text(encoding="utf-8"))
    ti.write_points(work, {1: [(1, [(meta["working_width"] / 2, meta["working_height"] / 2, 1)], None)]})
    s = mp.run_preview(ti.cfg_for(work), log=lambda m: None)
    t = s["frames"][0]["targets"][0]
    assert s["device"] == "cpu" and t["skor_objek"] > 0 and t["luas_px"] > 0 and s["run"]["cuda_initialized"] is False
    assert not sys.modules["torch"].cuda.is_initialized()
