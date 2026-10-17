"""Test isolate.py + integrasi cli (T-605, D-011): SAM TIRUAN, tanpa GPU / model / torch.

Test model nyata (skip default): ROTOSCOPE_GPU_TESTS=1 (GPU) dan ROTOSCOPE_SAM_CPU_TESTS=1 (bobot nyata di CPU, ±1 menit; butuh
SAM 2.1 tiny di cache HF).

Nomor kelompok (1)–(38) = daftar test rancangan Tahap 1 (docs/04 "Rancangan T-605 Tahap 1" R6); (39)–(42) = tambahan keputusan
Rio setelah Tahap 1: basi tanpa --yes berhenti; basi + --yes dihitung ulang; points.json dihapus → isolate/ dihapus; --limit
tidak memengaruhi [1b].

SAM tiruan meniru perilaku `transformers` 5.17.0 yang menentukan desain: penanda "input baru" ditimpa tiap `add_prompt` dan
dikosongkan saat frame kondisi dihitung (input frame lain yang belum dihitung → diabaikan, dicatat `input_diabaikan`); sesi tanpa
frame kondisi → ValueError saat melacak.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from rotoscope import cli
from rotoscope import isolate as iso
from rotoscope import pick
from rotoscope import stabilize as stb
from rotoscope import stage_common
from rotoscope.config import ConfigError, load_pipeline
from rotoscope.ingest import FRAME_PATTERN
from rotoscope.stage_common import BackendOOM, StageError, StageOOMError, clip_identity, load_frame_list

W, H = 40, 30
GRAY = (60, 200)                       # dua "adegan": selisih abu 140 / 255 = 0,55 ≫ cut_diff 0,06


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setattr(stage_common, "REPLACE_DELAY_S", 0)
    monkeypatch.setenv(stage_common.HF_OFFLINE_ENV, "1")


def quiet(msg: str) -> None:
    pass


# ── Klip + titik sintetis ──────────────────────────
def write_png(path: Path, rgb: np.ndarray) -> None:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    assert ok
    path.write_bytes(buf.tobytes())


def frame_rgb(i: int, scene: int) -> np.ndarray:
    base = GRAY[scene % 2]
    img = np.full((H, W, 3), base, np.uint8)
    img[0, i % W] = (base + 9) % 256          # sidik piksel unik per frame; skor cut dalam adegan ≈ 0
    return img


def make_clip(root: Path, n: int = 8, cuts=(), name: str = "klip") -> tuple[Path, Path]:
    """Folder kerja = <root>/work/clips/<name> (= lokasi cli bila cwd = root); cut SEBELUM tiap indeks di `cuts`."""
    video = root / f"{name}.mp4"
    root.mkdir(parents=True, exist_ok=True)
    video.write_bytes(b"bukan video sungguhan")
    work = root / "work" / "clips" / name
    (work / "frames").mkdir(parents=True)
    scene = 0
    for i in range(n):
        scene += i in cuts
        write_png(work / "frames" / (FRAME_PATTERN % i), frame_rgb(i, scene))
    meta = {"source_path": str(video.resolve()), "frame_count": n, "working_width": W, "working_height": H,
            "frame_index_start": 0}
    (work / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return work, video


def write_points(work: Path, spec: dict) -> dict:
    """spec = {frame: [(id, [(x, y, label), …], box | None), …]} → target/points.json (format pick, sidik piksel asli)."""
    _, _, w, h = load_frame_list(work)
    frames = []
    for f, targets in sorted(spec.items()):
        frames.append({"frame_index": f, "frame_file": FRAME_PATTERN % f,
                       "frame_pixel_sha256": pick.frame_pixel_sha256(work / "frames" / (FRAME_PATTERN % f)),
                       "updated_utc": "2026-10-17T00:00:00+00:00", "display": {},
                       "targets": [{"id": tid, "points": [{"x": float(x), "y": float(y), "label": lab} for x, y, lab in pts],
                                    "box": [float(v) for v in box] if box is not None else None}
                                   for tid, pts, box in targets]})
    doc = pick.new_doc(clip_identity(work), w, h, frames)
    pick.validate_doc(doc)
    iso.points_path(work).parent.mkdir(parents=True, exist_ok=True)
    stage_common.write_json_atomic(iso.points_path(work), doc)
    return doc


def cfg_for(work: Path, **over):
    return load_pipeline(overrides={"paths.work_dir": str(work), **over})


def tree(d: Path) -> dict[str, bytes]:
    return {p.relative_to(d).as_posix(): p.read_bytes() for p in sorted(d.rglob("*")) if p.is_file()} if d.exists() else {}


def label(work: Path, i: int) -> np.ndarray | None:
    return iso.read_label(iso.label_path(work / iso.ISOLATE_DIRNAME, i))


def rows(work: Path) -> dict[int, dict]:
    return {r["index"]: r for r in iso.read_frame_rows(work / iso.ISOLATE_DIRNAME / iso.FRAMES_LOG_FILENAME)}


def manifest(work: Path) -> dict:
    return json.loads((work / iso.ISOLATE_DIRNAME / iso.MANIFEST_FILENAME).read_text(encoding="utf-8"))


# ── SAM tiruan ─────────────────────────────────────
def first_positive(p: iso.Prompt) -> tuple[float, float]:
    pos = [xy for xy, lab in zip(p.points, p.labels) if lab == 1]
    if pos:
        return pos[0]
    x0, y0, x1, y1 = p.box
    return (x0 + x1) / 2, (y0 + y1) / 2


class FakeEngine:
    def __init__(self, b: "FakeBackend", frames, size, obj_id):
        self.b, self.frames, (self.h, self.w), self.obj = b, list(frames), size, obj_id
        self.index = [int(Path(p).stem.split("_")[1]) for p in self.frames]
        self.inputs: dict[int, iso.Prompt] = {}
        self.new_input, self.cond, self.center = False, set(), None
        b.events.append(("sesi", obj_id, self.index[0], self.index[-1]))

    def _out(self, k: int) -> iso.FrameOut:
        g = self.index[k]
        if self.b.fail is not None and self.b.fail[:2] == (self.obj, g):
            raise self.b.fail[2]
        if g in self.b.empty or self.center is None:
            return np.full((self.h, self.w), -5.0, np.float32), -2.0
        yy, xx = np.indices((self.h, self.w), dtype=np.float32)
        r = self.b.radius.get(self.obj, 4.0)
        return (r - np.hypot(xx - self.center[0], yy - self.center[1])).astype(np.float32), 5.0

    def add_prompt(self, k, prompt):
        self.inputs[k] = prompt
        self.new_input = True                              # transformers: ditimpa tiap panggilan
        self.b.prompts.append((self.obj, self.index[k], prompt))
        self.b.events.append(("tambah", self.obj, self.index[k]))

    def run_frame(self, k):
        g = self.index[k]
        if self.new_input and k in self.inputs:
            self.new_input = False                         # transformers: dikosongkan saat frame kondisi dihitung
            self.cond.add(k)
            self.center = self.center or first_positive(self.inputs[k])
            self.b.events.append(("kondisi", self.obj, g))
            return self._out(k)
        if not self.cond:
            raise ValueError("maskmem_features in conditioning outputs cannot be empty")
        if k in self.inputs:
            self.b.events.append(("input_diabaikan", self.obj, g))
        return self._out(k)

    def propagate(self, start, reverse):
        if not self.cond:
            raise ValueError("maskmem_features in conditioning outputs cannot be empty")
        self.b.events.append(("propagasi", self.obj, self.index[start], reverse))
        for k in (range(start, -1, -1) if reverse else range(start, len(self.frames))):
            if k in self.inputs and k not in self.cond:
                self.b.events.append(("input_diabaikan", self.obj, self.index[k]))
            yield k, self._out(k)

    def release(self):
        self.b.events.append(("lepas", self.obj))


class FakeBackend:
    def __init__(self, cfg, *, empty=(), fail=None, radius=None):
        self.cfg, self.empty, self.fail, self.radius = cfg, set(empty), fail, dict(radius or {})
        self.events: list[tuple] = []
        self.prompts: list[tuple] = []
        self.opened = self.closed = 0

    def describe(self):
        i = self.cfg.isolate
        return iso.ModelInfo(i.model_id, i.revision, "apache-2.0", i.precision, {"class": "Palsu"})

    def open(self):
        self.opened += 1
        return {"device": "palsu", "load_s": 0.0, "vram_free_before_load_mib": 3000.0, "vram_total_mib": 4096.0}

    def session(self, frames, size, obj_id):
        return FakeEngine(self, frames, size, obj_id)

    def stats(self):
        return {"max_reserved_mib": 1.0, "max_allocated_mib": 0.5}

    def close(self):
        self.closed += 1


def factory(**kw):
    made: list[FakeBackend] = []

    def f(cfg):
        made.append(FakeBackend(cfg, **kw))
        return made[-1]
    f.made = made
    return f


def run(work: Path, fac=None, **kw) -> dict:
    kw.setdefault("yes", True)
    return iso.run_isolate(cfg_for(work, **kw.pop("over", {})), backend_factory=fac or factory(), log=quiet,
                           ram_probe=lambda: None, **kw)


def kinds(fac) -> list[tuple]:
    return [e for e in fac.made[-1].events if e[0] in ("kondisi", "propagasi", "input_diabaikan")]


# ── (1)–(2) Adegan ─────────────────────────────────
def test_01_scenes_equal_stage3_cuts(tmp_path):
    work, _ = make_clip(tmp_path, n=8, cuts=(3, 6))
    cfg = cfg_for(work)
    cuts, shots = iso.detect_scenes(iso.load_clip(work), cfg.stabilize.temporal.cut_diff)
    assert cuts == [3, 6] and shots == [(0, 2), (3, 5), (6, 7)]
    (work / "qc_report.json").write_text(json.dumps({"frames": [{"frame": FRAME_PATTERN % i, "fail_reasons": []}
                                                                for i in range(8)]}), encoding="utf-8")
    plan = stb.build_plan(cfg, stb.Clip(work, *load_frame_list(work)), log=quiet)
    assert list(plan.cut_frames) == cuts                                    # fungsi + ambang yang sama dengan [3]
    assert sorted(set(zip(plan.shot_start, plan.shot_end))) == shots


def test_02_cut_diff_zero_is_one_scene(tmp_path):
    work, _ = make_clip(tmp_path, n=6, cuts=(3,))
    cuts, shots = iso.detect_scenes(iso.load_clip(work), 0.0)
    assert cuts == [] and shots == [(0, 5)]


def test_02b_run_uses_stabilize_cut_diff(tmp_path):
    """Ambang cut [1b] = stabilize.temporal.cut_diff dari config (tanpa parameter terpisah)."""
    work, _ = make_clip(tmp_path, n=8, cuts=(4,))
    write_points(work, {1: [(1, P, None)]})
    run(work, over={"stabilize.temporal.cut_diff": 0.0})
    assert manifest(work)["scenes"] == {"cut_frames": [], "shots": [[0, 7]]}
    assert [r["status"] for r in rows(work).values()] == ["target"] * 8
    run(work, over={"stabilize.temporal.cut_diff": 0.06})                   # basi (cut_diff di kunci) → dihitung ulang
    assert manifest(work)["scenes"]["cut_frames"] == [4] and rows(work)[5]["status"] == "tanpa_prompt"


# ── (3)–(6) Perakitan prompt ───────────────────────
P = [(10, 12, 1), (30, 5, 0)]


def test_03_clicks_in_two_scenes_are_two_jobs(tmp_path):
    work, _ = make_clip(tmp_path, n=8, cuts=(4,))
    doc = write_points(work, {1: [(1, P, None)], 6: [(1, P, None)]})
    jobs = iso.build_jobs(doc, [(0, 3), (4, 7)])
    assert [(j.shot, j.target, [p.index for p in j.prompts]) for j in jobs] == [((0, 3), 1, [1]), ((4, 7), 1, [6])]
    fac = factory()
    run(work, fac)
    assert [e for e in fac.made[0].events if e[0] == "sesi"] == [("sesi", 1, 0, 3), ("sesi", 1, 4, 7)]


def test_04_two_click_frames_one_job_both_conditioning(tmp_path):
    work, _ = make_clip(tmp_path, n=8)
    write_points(work, {5: [(1, P, None)], 2: [(1, P, None)]})
    fac = factory()
    run(work, fac)
    ev = fac.made[0].events
    assert [e for e in ev if e[0] == "sesi"] == [("sesi", 1, 0, 7)]
    assert {e[2] for e in ev if e[0] == "kondisi"} == {2, 5}                # koreksi f5 = frame kondisi
    assert not [e for e in ev if e[0] == "input_diabaikan"]
    assert manifest(work)["jobs"][0]["start"] == 2 and manifest(work)["jobs"][0]["prompt_frames"] == [2, 5]


def test_05_target_clicked_later_starts_there(tmp_path):
    work, _ = make_clip(tmp_path, n=8)
    write_points(work, {1: [(1, [(8, 8, 1)], None)], 5: [(1, [(8, 8, 1)], None), (2, [(30, 20, 1)], None)]})
    fac = factory()
    run(work, fac)
    jobs = {j["target"]: j for j in manifest(work)["jobs"]}
    assert jobs[1]["start"] == 1 and jobs[2]["start"] == 5 and jobs[2]["reverse"] == [0, 4]
    assert not [e for e in fac.made[0].events if e[0] == "input_diabaikan"]
    for i in range(8):
        assert set(np.unique(label(work, i))) == {0, 1, 2}                   # target 2 juga dilacak MUNDUR f4 → f0
    assert rows(work)[0]["targets"]["2"]["kind"] == "mundur" and rows(work)[5]["targets"]["2"]["kind"] == "klik"


def test_06_points_labels_box_passed_as_is(tmp_path):
    work, _ = make_clip(tmp_path, n=4)
    write_points(work, {2: [(1, [(10.5, 12.25, 1), (30, 5, 0)], (2, 3, 20.5, 25)), (2, [(33, 7, 1)], None)]})
    fac = factory()
    run(work, fac)
    got = {obj: p for obj, _, p in fac.made[0].prompts}
    assert got[1].points == ((10.5, 12.25), (30.0, 5.0))
    assert got[1].labels == (1, 0) and got[1].box == (2.0, 3.0, 20.5, 25.0)
    assert got[2].points == ((33.0, 7.0),) and got[2].labels == (1,) and got[2].box is None


# ── (7)–(8) Propagasi ──────────────────────────────
class LogEngine:
    """Engine minimal: mencatat panggilan; tiap frame logit = indeks lokal."""

    def __init__(self, n):
        self.n, self.calls = n, []

    def add_prompt(self, k, prompt):
        self.calls.append(("tambah", k))

    def run_frame(self, k):
        self.calls.append(("hitung", k))
        return np.full((2, 2), float(k), np.float32), 1.0

    def propagate(self, start, reverse):
        self.calls.append(("propagasi", start, reverse))
        for k in (range(start, -1, -1) if reverse else range(start, self.n)):
            yield k, (np.full((2, 2), float(k), np.float32), 1.0)

    def release(self):
        pass


def job(shot, *frames, target=1):
    return iso.Job(shot, target, tuple(iso.Prompt(f, ((1.0, 1.0),), (1,), None) for f in frames))


def test_07_order_conditioning_forward_reverse_each_frame_once():
    eng = LogEngine(8)
    out = list(iso.track_job(eng, job((10, 17), 13, 15)))
    assert eng.calls == [("tambah", 3), ("hitung", 3), ("tambah", 5), ("hitung", 5), ("propagasi", 3, False),
                         ("propagasi", 3, True)]
    assert [(i, k) for i, k, _ in out] == [(13, "klik"), (15, "klik"), (14, "maju"), (16, "maju"), (17, "maju"),
                                           (12, "mundur"), (11, "mundur"), (10, "mundur")]
    assert all(float(o[0][0, 0]) == i - 10 for i, _, o in out)             # indeks lokal ↔ global tidak bergeser


def test_08_click_on_first_or_last_frame_of_scene():
    eng = LogEngine(4)
    out = list(iso.track_job(eng, job((0, 3), 0)))
    assert ("propagasi", 0, True) not in eng.calls and [i for i, _, _ in out] == [0, 1, 2, 3]
    eng = LogEngine(4)
    out = list(iso.track_job(eng, job((4, 7), 7)))
    assert [(i, k) for i, k, _ in out] == [(7, "klik"), (6, "mundur"), (5, "mundur"), (4, "mundur")]


def test_07b_missing_frames_from_backend_is_an_error():
    class Short(LogEngine):
        def propagate(self, start, reverse):
            yield from ()
    with pytest.raises(StageError, match="tidak dihasilkan"):
        list(iso.track_job(Short(4), job((0, 3), 1)))


# ── (9)–(12) Peta label ────────────────────────────
def test_09_threshold_is_strict():
    lg = np.array([[0.0, 1e-6, -1e-6, np.nan]], np.float32)
    assert iso.target_mask(lg, 0.0).tolist() == [[False, True, False, False]]
    assert iso.target_mask(lg, -0.5).tolist() == [[True, True, True, False]]


def test_10_overlap_highest_logit_tie_smallest_id():
    m = np.ones((1, 3), bool)
    a = np.array([[2.0, 1.0, 3.0]], np.float16)
    b = np.array([[1.0, 1.0, 5.0]], np.float16)
    assert iso.compose_labels({2: (m, b), 1: (m, a)}, (1, 3)).tolist() == [[1, 1, 2]]
    assert iso.compose_labels({1: (np.zeros((1, 3), bool), a), 2: (m, b)}, (1, 3)).tolist() == [[2, 2, 2]]


def test_11_label_values_are_ids_and_id_over_255_rejected(tmp_path):
    m = np.ones((2, 2), bool)
    lab = iso.compose_labels({7: (m, np.ones((2, 2), np.float16)), 255: (~m, np.ones((2, 2), np.float16))}, (2, 2))
    assert lab.dtype == np.uint8 and set(np.unique(lab)) == {7}
    with pytest.raises(StageError, match="255"):
        iso.compose_labels({256: (m, np.ones((2, 2), np.float16))}, (2, 2))
    work, _ = make_clip(tmp_path, n=3)
    write_points(work, {1: [(256, [(5, 5, 1)], None)]})
    with pytest.raises(StageError, match=r"id target \[256\]"):
        run(work)


def test_12_png_roundtrip(tmp_path):
    lab = (np.arange(30 * 40).reshape(30, 40) % 256).astype(np.uint8)
    p = tmp_path / "labels" / "frame_00001.png"
    p.parent.mkdir()
    iso.write_label(p, lab)
    assert np.array_equal(iso.read_label(p), lab)


# ── (13)–(15) Status ───────────────────────────────
def test_13_15_status_target_kosong_tanpa_prompt(tmp_path):
    work, _ = make_clip(tmp_path, n=8, cuts=(4,))
    write_points(work, {1: [(1, P, None)]})
    run(work, factory(empty={3}))
    r = rows(work)
    assert [r[i]["status"] for i in range(8)] == ["target"] * 3 + ["kosong"] + ["tanpa_prompt"] * 4
    assert all(label(work, i) is not None for i in range(4)) and not label(work, 3).any()
    assert all(label(work, i) is None for i in range(4, 8))                # tanpa_prompt = tanpa PNG
    assert sorted(p.name for p in (work / "isolate" / "labels").iterdir()) == [FRAME_PATTERN % i for i in range(4)]
    f = manifest(work)["frames"]
    assert (f["target"], f["kosong"], f["tanpa_prompt"]) == (3, 1, 4)
    assert f["kosong_ranges"] == [[3, 3]] and f["tanpa_prompt_ranges"] == [[4, 7]]
    assert r[3]["targets"]["1"]["area_px"] == 0 and r[0]["targets"]["1"]["area_px"] > 0 and r[5]["targets"] == {}


# ── (16)–(20) Basi ─────────────────────────────────
def test_16_key_fields(tmp_path):
    work, _ = make_clip(tmp_path, n=4)
    base = iso.build_key(cfg_for(work), "a" * 64, "b" * 64)
    for over in ({"isolate.model_id": "facebook/sam2.1-hiera-small"}, {"isolate.revision": "c" * 40},
                 {"isolate.precision": "fp16"}, {"isolate.mask_threshold": 0.5}, {"stabilize.temporal.cut_diff": 0.07}):
        assert iso.build_key(cfg_for(work, **over), "a" * 64, "b" * 64) != base, over
    for over in ({"isolate.vram_min_free_mib": 100}, {"isolate.ram_min_free_mib": 1}):
        assert iso.build_key(cfg_for(work, **over), "a" * 64, "b" * 64) == base, over
    assert iso.build_key(cfg_for(work), "x" * 64, "b" * 64) != base and iso.build_key(cfg_for(work), "a" * 64, "y" * 64) != base
    assert base["contract"] == "T-605" and set(base["libs"]) == {"torch", "transformers"}
    sha1, _ = iso.frames_fingerprint(iso.load_clip(work))
    img = frame_rgb(2, 0)
    img[5, 5, 1] ^= 1                                                       # satu nilai piksel di frame mana pun
    write_png(work / "frames" / (FRAME_PATTERN % 2), img)
    assert iso.frames_fingerprint(iso.load_clip(work))[0] != sha1


def test_17_up_to_date_no_backend_no_writes(tmp_path):
    work, _ = make_clip(tmp_path, n=6)
    write_points(work, {2: [(1, P, None)]})
    run(work)
    before = tree(work / "isolate")
    for yes in (False, True):
        fac = factory()
        assert run(work, fac, yes=yes)["status"] == "up-to-date"
        assert fac.made == [] and tree(work / "isolate") == before


def test_18_40_stale_with_yes_recomputed_with_warning(tmp_path):
    work, _ = make_clip(tmp_path, n=6)
    write_points(work, {2: [(1, P, None)]})
    run(work)
    old = manifest(work)
    write_points(work, {2: [(1, [(30, 20, 1)], None)]})                     # titik berubah → basi
    logs: list[str] = []
    fac = factory()
    res = iso.run_isolate(cfg_for(work), yes=True, backend_factory=fac, log=logs.append, ram_probe=lambda: None)
    assert res["status"] == "dihitung" and fac.made and fac.made[0].opened == 1
    assert any("PERINGATAN: isolate/ basi" in m and "points_sha256" in m for m in logs)
    assert manifest(work)["key"]["points_sha256"] != old["key"]["points_sha256"]
    for over in ({"isolate.mask_threshold": 1.0}, {"stabilize.temporal.cut_diff": 0.2}):
        fac = factory()
        assert run(work, fac, over=over)["status"] == "dihitung" and fac.made


def test_19_interrupted_keeps_old_and_cleans_partial(tmp_path):
    work, _ = make_clip(tmp_path, n=6)
    write_points(work, {1: [(1, P, None)]})
    run(work)
    before = tree(work / "isolate")
    write_points(work, {1: [(1, [(20, 20, 1)], None)]})
    for exc, err in ((KeyboardInterrupt(), KeyboardInterrupt), (BackendOOM({"free_mib": 1.0}), StageOOMError)):
        fac = factory(fail=(1, 4, exc))
        with pytest.raises(err):
            run(work, fac)
        assert tree(work / "isolate") == before and (work / iso.PARTIAL_DIRNAME).exists()
        assert fac.made[0].closed == 1 and ("lepas", 1) in fac.made[0].events
    logs: list[str] = []
    iso.run_isolate(cfg_for(work), yes=True, backend_factory=factory(), log=logs.append, ram_probe=lambda: None)
    assert any("isolate.partial/" in m for m in logs) and not (work / iso.PARTIAL_DIRNAME).exists()
    assert tree(work / "isolate") != before and not (work / iso.OLD_DIRNAME).exists()


def test_20_missing_or_corrupt_png_is_stale(tmp_path):
    work, _ = make_clip(tmp_path, n=6)
    write_points(work, {1: [(1, P, None)]})
    run(work)
    iso.label_path(work / "isolate", 3).unlink()
    st, why = iso.existing_state(iso.load_clip(work), manifest(work)["key"], clip_identity(work))
    assert st == "basi" and "PNG hilang" in why[0]
    fac = factory()
    assert run(work, fac)["status"] == "dihitung" and label(work, 3) is not None
    iso.label_path(work / "isolate", 2).write_bytes(b"rusak")
    assert run(work, factory())["status"] == "dihitung"
    (work / "isolate" / "labels" / "frame_00099.png").write_bytes(b"x")
    assert run(work, factory())["status"] == "dihitung"


# ── (21)–(24) Titik ────────────────────────────────
def test_21_no_points_skips_without_files(tmp_path):
    work, _ = make_clip(tmp_path, n=4)
    before = tree(work)
    fac = factory()
    assert run(work, fac)["status"] == "dilewati" and fac.made == [] and tree(work) == before
    assert iso.main(["--work-dir", str(work), "--yes"], backend_factory=fac) == 0 and tree(work) == before
    assert sorted(p.name for p in work.iterdir()) == ["frames", "meta.json"]  # tanpa folder baru (isolate/, target/, …)


def test_22_empty_frames_list_no_gpu_no_yes(tmp_path):
    work, _ = make_clip(tmp_path, n=4)
    write_points(work, {})
    fac = factory()
    assert run(work, fac, yes=False)["status"] == "dihitung" and fac.made == []
    assert [r["status"] for r in rows(work).values()] == ["tanpa_prompt"] * 4 and manifest(work)["jobs"] == []
    assert not list((work / "isolate" / "labels").iterdir())


def test_23_stale_or_missing_click_frame_is_rejected(tmp_path):
    work, _ = make_clip(tmp_path, n=4)
    write_points(work, {2: [(1, P, None)]})
    img = frame_rgb(2, 0)
    img[3, 3] = 0
    write_png(work / "frames" / (FRAME_PATTERN % 2), img)                   # isi frame berubah sejak diklik
    with pytest.raises(StageError, match=r"f2: basi.*pick .* --frame 2"):
        run(work)
    assert not (work / "isolate").exists()
    work2, _ = make_clip(tmp_path / "b", n=4)
    doc = write_points(work2, {2: [(1, P, None)]})
    doc["frames"][0].update(frame_index=9, frame_file=FRAME_PATTERN % 9)
    stage_common.write_json_atomic(iso.points_path(work2), doc)
    with pytest.raises(StageError, match=r"f9: hilang"):
        run(work2)


def test_24_points_of_other_clip_or_corrupt(tmp_path):
    work, _ = make_clip(tmp_path, n=4)
    doc = write_points(work, {1: [(1, P, None)]})
    doc["clip"]["source_path"] = "C:/lain/klip.mp4"
    stage_common.write_json_atomic(iso.points_path(work), doc)
    with pytest.raises(StageError, match="klip lain"):
        run(work)
    iso.points_path(work).write_text("{rusak", encoding="utf-8")
    with pytest.raises(StageError, match="tidak valid"):
        run(work)
    assert iso.main(["--work-dir", str(work), "--yes"], backend_factory=factory()) == 1


# ── (25) Manifest ──────────────────────────────────
def test_25_manifest_fields(tmp_path):
    work, _ = make_clip(tmp_path, n=8, cuts=(5,))
    write_points(work, {2: [(1, P, (1, 1, 20, 20))], 4: [(1, [(9, 9, 1)], None)]})
    run(work)
    m = manifest(work)
    for k in ("stage", "format", "clip", "frame_size", "frame_count", "key", "key_sha256", "model", "processor", "mode",
              "scenes", "jobs", "targets", "frames", "output", "run", "created_utc"):
        assert k in m, k
    assert m["format"] == "rotoscope-isolate/1" and m["clip"] == clip_identity(work) and m["frame_count"] == 8
    assert m["key_sha256"] == iso.canonical_sha256(m["key"])
    assert m["key"]["points_sha256"] == hashlib.sha256(iso.points_path(work).read_bytes()).hexdigest()
    assert m["model"] == {"model_id": "facebook/sam2.1-hiera-tiny", "revision": "de431c4043854a71d8101e17995dfe596bf101a5",
                          "license": "apache-2.0", "precision": "fp32"}
    assert m["scenes"] == {"cut_frames": [5], "shots": [[0, 4], [5, 7]]}
    j = m["jobs"][0]
    assert (j["shot"], j["prompt_frames"], j["forward"], j["reverse"], j["n_infer"]) == ([0, 4], [2, 4], [2, 4], [0, 1], 5)
    assert j["prompts"][0] == {"frame": 2, "n_pos": 1, "n_neg": 1, "box": [1.0, 1.0, 20.0, 20.0]}
    for k in ("started_utc", "finished_utc", "wall_s", "key_s", "scenes_s", "infer_s", "vram", "ram", "nonfinite_frames"):
        assert k in m["run"], k
    assert m["run"]["vram"]["vram_free_before_load_mib"] == 3000.0 and m["run"]["vram"]["max_reserved_mib"] == 1.0
    assert all(r["pixel_sha256"] == pick.frame_pixel_sha256(work / "frames" / r["frame"]) for r in rows(work).values())


# ── (26) VRAM / RAM ────────────────────────────────
def test_26_vram_ram_thresholds():
    iso.check_vram(2000.0, 4096.0, 2000)                                    # tepat di ambang = lolos
    with pytest.raises(StageError, match="VRAM bebas 1999 MiB < dibutuhkan 2000"):
        iso.check_vram(1999.4, 4096.0, 2000)
    iso.check_ram(4096.0, 4096)
    iso.check_ram(None, 4096)                                               # tidak terbaca → tidak memblokir
    with pytest.raises(StageError, match="RAM bebas 4000 MiB"):
        iso.check_ram(4000.0, 4096)


def test_26b_ram_checked_before_model_load(tmp_path):
    work, _ = make_clip(tmp_path, n=4)
    write_points(work, {1: [(1, P, None)]})
    fac = factory()
    with pytest.raises(StageError, match="RAM bebas"):
        iso.run_isolate(cfg_for(work), yes=True, backend_factory=fac, log=quiet, ram_probe=lambda: 100.0)
    assert fac.made == [] and not (work / "isolate").exists()


# ── (27)–(33) CLI ──────────────────────────────────
from test_cli import CPU_STAGES, Rec  # noqa: E402


@pytest.fixture
def rec(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    r = Rec()
    monkeypatch.setattr(cli, "_popen", r.popen)
    monkeypatch.setattr(cli, "CPU_MAINS", {s: r.cpu(s) for s in CPU_STAGES})
    return r


def test_27_stage_order_and_gpu_stages():
    assert cli.STAGES == ("ingest", "isolate", "segment", "depth", "stabilize", "vectorize", "stylize", "export")
    assert cli.GPU_STAGES == ("isolate", "segment", "depth") and cli.MODULES["isolate"] == "rotoscope.isolate"
    assert cli.RESTART_SCOPE["isolate"] == ("isolate",) and "isolate" in cli.RESTART_SCOPE["ingest"]


def test_28_run_without_points_no_subprocess_same_argv_no_files(rec, tmp_path):
    work, video = make_clip(tmp_path, n=4)
    before = tree(tmp_path)
    assert cli.main(["run", str(video)]) == 0
    wd = str(work)
    assert [(c[0], c[1], c[2]) for c in rec.calls] == [
        ("cpu", "ingest", [str(video), "--work-dir", wd]), ("gpu", "segment", ["--work-dir", wd]),
        ("gpu", "depth", ["--work-dir", wd]), ("cpu", "stabilize", ["--work-dir", wd]),
        ("cpu", "vectorize", ["--work-dir", wd]), ("cpu", "stylize", ["--work-dir", wd]), ("cpu", "export", ["--work-dir", wd])]
    assert tree(tmp_path) == before                                         # tidak ada berkas baru / berubah
    assert sorted(p.name for p in work.iterdir()) == ["frames", "meta.json"]  # tanpa folder baru


def test_29_run_with_points_spawns_isolate(rec, tmp_path):
    work, video = make_clip(tmp_path, n=4)
    write_points(work, {1: [(1, P, None)]})
    conf = tmp_path / "c.yaml"
    conf.write_text("isolate:\n  mask_threshold: 0.5\n", encoding="utf-8")
    assert cli.main(["run", str(video), "--config", str(conf)]) == 0
    assert rec.stages[:3] == ["ingest", "isolate", "segment"] and rec.calls[1][0] == "gpu"
    assert rec.args("isolate") == ["--work-dir", str(work), "--config", str(conf)]
    rec.calls.clear()
    assert cli.main(["run", str(video), "--yes"]) == 0
    assert rec.args("isolate") == ["--work-dir", str(work), "--yes"]
    assert all("--yes" not in c[2] for c in rec.calls if c[1] != "isolate")


def test_30_subcommand_restart_requires_yes(rec, tmp_path, capsys):
    work, video = make_clip(tmp_path, n=4)
    (work / "isolate").mkdir()
    (work / "isolate" / "x").write_text("x")
    assert cli.main(["isolate", str(video), "--restart"]) == 1
    assert rec.calls == [] and (work / "isolate" / "x").exists() and "--yes" in capsys.readouterr().err
    assert cli.main(["isolate", str(video), "--restart", "--yes"]) == 0
    assert rec.args("isolate") == ["--work-dir", str(work), "--restart", "--yes"]
    rec.calls.clear()
    assert cli.main(["isolate", str(video), "--dry-run"]) == 0 and rec.args("isolate") == ["--work-dir", str(work), "--dry-run"]


def test_31_restart_from_isolate_and_ingest(rec, tmp_path, capsys):
    work, video = make_clip(tmp_path, n=4)
    write_points(work, {1: [(1, P, None)]})
    assert cli.main(["run", str(video), "--restart-from", "isolate"]) == 1 and rec.calls == []
    err = capsys.readouterr().err
    assert "[1b] isolate" in err and "hanya adegan berklik" in err and "--yes" in err
    assert cli.main(["run", str(video), "--restart-from", "isolate", "--yes"]) == 0
    assert {c[1] for c in rec.calls if "--restart" in c[2]} == {"isolate"} and rec.args("isolate")[-2:] == ["--restart", "--yes"]
    rec.calls.clear()
    assert cli.main(["run", str(video), "--restart-from", "ingest", "--yes"]) == 0
    assert "--restart" in rec.args("isolate")


def test_32_33_download_and_no_torch_import():
    code = ("import sys, rotoscope.cli, rotoscope.isolate; "
            "sys.exit(1 if ('torch' in sys.modules or 'transformers' in sys.modules) else 0)")
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0
    p = cli.build_parser()
    assert "SAM 2.1" in p._subparsers._group_actions[0].choices["download"].description   # noqa: SLF001


# ── (34) Rantai CPU nyata: byte-identik dengan / tanpa titik ──
class InProcIso:
    """popen palsu: [1b] = isolate.main nyata (SAM tiruan) sinkron; [2] / [2c] = tidak melakukan apa pun (hasil sudah ada)."""
    fac = None
    seen: list[list[str]] = []

    def __init__(self, argv, env=None):
        InProcIso.seen.append(list(argv[2:]))
        self.rc = iso.main(list(argv[3:]), backend_factory=InProcIso.fac) if argv[2] == "rotoscope.isolate" else 0

    def wait(self, timeout=None):
        return self.rc

    def poll(self):
        return self.rc

    def terminate(self):
        pass


@pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="ffmpeg/ffprobe tidak ada")
def test_34_real_cpu_chain_byte_identical_with_points_and_isolate(tmp_path, monkeypatch):
    import test_pick as tp
    import test_stabilize as tst

    from rotoscope import export as ex
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
        tp.write_png(wd / "frames" / f"frame_{i:05d}.png", tp.frame_rgb(i, tst.W, tst.H))
    (wd / "qc_report.json").write_text(json.dumps({"stage": "segment_qc", "frames": [
        {"frame": f"frame_{i:05d}.png", "index": i, "fail_reasons": []} for i in range(tst.N_FRAMES)]}), encoding="utf-8")
    for d in ("seg", "depth"):
        p = wd / d / "manifest.json"
        p.write_text(json.dumps({**json.loads(p.read_text(encoding="utf-8")), "clip": clip_identity(wd)}), encoding="utf-8")
    r = Rec()
    InProcIso.fac, InProcIso.seen = factory(), []
    monkeypatch.setattr(cli, "_popen", InProcIso)
    monkeypatch.setattr(cli, "CPU_MAINS", {"ingest": r.cpu("ingest"), "stabilize": stb.main, "vectorize": vec.main,
                                           "stylize": sty.main, "export": ex.main})
    out = tmp_path / "out"
    assert cli.main(["run", str(video)]) == 0
    first = tp._content_hashes(wd, out)
    assert any("strokes/" in k for k in first) and not (wd / "isolate").exists()
    assert [a[0] for a in InProcIso.seen] == ["rotoscope.segment", "rotoscope.depth"]
    write_points(wd, {1: [(1, [(3, 4, 1), (8, 9, 0)], None)]})
    for d in ("stable", "contours", "strokes"):
        shutil.rmtree(wd / d)
    shutil.rmtree(out)
    assert cli.main(["run", str(video)]) == 1 and not (wd / "isolate").exists()      # GPU tanpa --yes → berhenti di [1b]
    assert cli.main(["run", str(video), "--yes"]) == 0
    assert (wd / "isolate" / "manifest.json").is_file() and InProcIso.fac.made
    assert tp._content_hashes(wd, out) == first                            # [2]–[6] tidak berubah (T-605 hanya masker)


# ── (35) --dry-run ─────────────────────────────────
def test_35_dry_run_no_writes_no_backend(tmp_path):
    work, _ = make_clip(tmp_path, n=8, cuts=(5,))
    write_points(work, {6: [(1, P, None)]})
    (work / iso.PARTIAL_DIRNAME).mkdir()                                     # sisa run terputus: dry-run tidak menghapusnya
    before = tree(work) | {"_dirs": str(sorted(p.name for p in work.iterdir())).encode()}
    logs: list[str] = []
    fac = factory()
    res = iso.run_isolate(cfg_for(work), dry_run=True, backend_factory=fac, log=logs.append, ram_probe=lambda: None)
    assert res["status"] == "dry-run" and res["state"] == "tidak_ada" and fac.made == []
    assert res["shots"] == [(0, 4), (5, 7)] and res["gpu_frames"] == 3
    assert res["jobs"][0]["reverse"] == [5, 5] and any("dibutuhkan (--yes wajib)" in m for m in logs)
    assert tree(work) | {"_dirs": str(sorted(p.name for p in work.iterdir())).encode()} == before
    code = (f"import sys; from rotoscope import isolate as i; rc = i.main(['--work-dir', r'{work}', '--dry-run']); "
            f"sys.exit(rc or (2 if 'torch' in sys.modules else 0))")
    assert subprocess.run([sys.executable, "-c", code], capture_output=True).returncode == 0


# ── (36) Config ────────────────────────────────────
@pytest.mark.parametrize("over, match", [
    ({"isolate.model_id": "facebook/sam2-hiera-large"}, r"isolate\.model_id.*sam2\.1-hiera-"),
    ({"isolate.precision": "bf16"}, r"isolate\.precision.*P-005"),
    ({"isolate.vram_min_free_mib": 5000}, r"isolate\.vram_min_free_mib"),
    ({"isolate.ram_min_free_mib": -1}, r"isolate\.ram_min_free_mib"),
    ({"isolate.mask_threshold": 11}, r"isolate\.mask_threshold"),
    ({"isolate.revision": " "}, r"isolate\.revision"),
])
def test_36_config_validation(over, match):
    with pytest.raises(ConfigError, match=match):
        load_pipeline(overrides=over)


def test_36b_config_defaults_and_runtime_revision(tmp_path):
    i = load_pipeline().isolate
    assert (i.model_id, i.revision, i.precision, i.vram_min_free_mib, i.ram_min_free_mib, i.mask_threshold) == (
        "facebook/sam2.1-hiera-tiny", "de431c4043854a71d8101e17995dfe596bf101a5", "fp32", 2000, 4096, 0.0)
    assert load_pipeline(overrides={"isolate.revision": None}).isolate.revision is None
    with pytest.raises(StageError, match="null"):
        iso.resolve_revision(load_pipeline(overrides={"isolate.revision": None}))
    with pytest.raises(StageError, match="40 karakter"):
        iso.resolve_revision(load_pipeline(overrides={"isolate.revision": "main"}))


# ── (39)–(42) Tambahan keputusan Rio setelah Tahap 1 ──
def test_39_stale_without_yes_stops_and_changes_nothing(tmp_path):
    work, _ = make_clip(tmp_path, n=6)
    write_points(work, {1: [(1, P, None)]})
    with pytest.raises(StageError, match=r"butuh GPU \(belum ada hasil\).*--yes"):
        run(work, yes=False)                                                # hitung baru juga butuh --yes
    assert not (work / "isolate").exists()
    run(work)
    before = tree(work / "isolate")
    write_points(work, {1: [(1, [(25, 25, 1)], None)]})
    fac = factory()
    with pytest.raises(StageError, match=r"butuh GPU \(hasil lama basi: key\.points_sha256.*Tidak ada yang dihitung"):
        run(work, fac, yes=False)
    assert fac.made == [] and tree(work / "isolate") == before
    assert iso.main(["--work-dir", str(work)], backend_factory=fac) == 1 and tree(work / "isolate") == before


def test_41_points_deleted_removes_isolate(tmp_path, rec):
    work, video = make_clip(tmp_path, n=4)
    write_points(work, {1: [(1, P, None)]})
    run(work)
    assert (work / "isolate").exists()
    iso.points_path(work).unlink()
    logs: list[str] = []
    assert iso.run_isolate(cfg_for(work), log=logs.append)["removed"] == ["isolate/"]
    assert not (work / "isolate").exists() and any("DIHAPUS" in m for m in logs)
    write_points(work, {1: [(1, P, None)]})                                 # lewat `run` (in-process, tanpa subprocess)
    run(work)
    iso.points_path(work).unlink()
    (work / iso.PARTIAL_DIRNAME).mkdir()
    rec.calls.clear()
    assert cli.main(["run", str(video)]) == 0
    assert not (work / "isolate").exists() and not (work / iso.PARTIAL_DIRNAME).exists() and "isolate" not in rec.stages


def test_42_limit_does_not_affect_isolate(rec, tmp_path, capsys):
    work, video = make_clip(tmp_path, n=6)
    write_points(work, {2: [(1, P, None)]})
    assert cli.main(["run", str(video), "--limit", "2", "--yes"]) == 0
    assert "--limit" not in rec.args("isolate") and rec.args("vectorize")[-2:] == ["--limit", "2"]
    assert "tidak berlaku untuk [1b]" in capsys.readouterr().out
    p = iso.main.__code__.co_varnames                                       # modul [1b] tidak punya --limit
    assert "limit" not in p
    with pytest.raises(SystemExit):
        iso.main(["--work-dir", str(work), "--limit", "2"])


# ── (37)–(38) Model nyata (skip default) ───────────
def square_clip(root: Path, n: int = 6) -> Path:
    """Kotak putih 16 × 16 bergerak 4 px / frame di latar hitam (64 × 48); skor cut ±0,04 < 0,06 → satu adegan."""
    video = root / "kotak.mp4"
    video.write_bytes(b"x")
    work = root / "work" / "clips" / "kotak"
    (work / "frames").mkdir(parents=True)
    for i in range(n):
        img = np.zeros((48, 64, 3), np.uint8)
        img[16:32, 8 + 4 * i:24 + 4 * i] = 255
        write_png(work / "frames" / (FRAME_PATTERN % i), img)
    (work / "meta.json").write_text(json.dumps({"source_path": str(video.resolve()), "frame_count": n, "working_width": 64,
                                                "working_height": 48, "frame_index_start": 0}), encoding="utf-8")
    write_points(work, {2: [(1, [(23.5, 23.5, 1), (55, 5, 0)], (14.0, 14.0, 34.0, 33.0))]})
    return work


def _real_check(tmp_path, device):
    work = square_clip(tmp_path)
    fac = lambda cfg: iso.TorchSamBackend(cfg, device=device)  # noqa: E731
    log = lambda m: print(m.encode("ascii", "replace").decode())  # noqa: E731 — stdout pytest cp1252
    res = iso.run_isolate(cfg_for(work), yes=True, backend_factory=fac, log=log, ram_probe=lambda: None)
    assert res["status"] == "dihitung"
    first = tree(work / "isolate" / "labels")
    for i in range(6):
        sq = np.zeros((48, 64), bool)
        sq[16:32, 8 + 4 * i:24 + 4 * i] = True
        lab = label(work, i) == 1
        iou = (lab & sq).sum() / max((lab | sq).sum(), 1)
        assert iou > 0.5, (i, iou)                                          # maju (f3–f5) DAN mundur (f1–f0)
    assert [r["targets"]["1"]["kind"] for r in rows(work).values()] == ["mundur", "mundur", "klik", "maju", "maju", "maju"]
    shutil.rmtree(work / "isolate")
    iso.run_isolate(cfg_for(work), yes=True, backend_factory=fac, log=log, ram_probe=lambda: None)
    return first, tree(work / "isolate" / "labels"), manifest(work)


@pytest.mark.skipif(os.environ.get("ROTOSCOPE_SAM_CPU_TESTS") != "1", reason="set ROTOSCOPE_SAM_CPU_TESTS=1 (bobot SAM di cache)")
def test_38_real_sam_cpu_contract(tmp_path):
    """Kelas sesi `transformers` asli: frame malas, kotak + titik negatif, frame kondisi, maju + mundur; CPU dua kali = sama."""
    first, second, m = _real_check(tmp_path, "cpu")
    assert first == second and m["processor"]["size"] == {"height": 1024, "width": 1024}


@pytest.mark.skipif(os.environ.get("ROTOSCOPE_GPU_TESTS") != "1", reason="set ROTOSCOPE_GPU_TESTS=1 (butuh GPU + cache)")
def test_37_real_sam_gpu(tmp_path):
    first, second, m = _real_check(tmp_path, "cuda")
    assert m["run"]["vram"]["max_reserved_mib"] > 0 and m["run"]["open"]["device"] == "cuda"
    print("GPU dua kali byte-identik:", first == second)
