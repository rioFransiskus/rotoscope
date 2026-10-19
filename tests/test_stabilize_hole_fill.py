"""Test VF (T-305f): isi lubang grup face di stage [3] (docs/04 "Keputusan T-305f setelah Tahap 1").

Aturan (= prototipe `scripts/t305_1b.py::vf_fill`, salinan di `tests/hole_fill_oracle.py`): komponen background 4-arah yang tidak
menyentuh tepi frame dan SELURUH cincin 1 px-nya (8-arah) = face → diisi face; tanpa batas ukuran. Tempat (PB): `run_stabilize`
sesudah `final_depth`, sebelum `write_groups` — hanya peta grup yang ditulis; kedalaman memakai peta sebelum VF.
Sintetis, tanpa model / GPU; integrasi masker T-606 memakai [1b] ASLI dengan SAM tiruan (pola `tests/test_target_mask.py`).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

import hole_fill_oracle as oracle
import test_stabilize as tst
import test_target_mask as ttm
from rotoscope import stabilize as stb
from rotoscope import stage_common
from rotoscope.config import HOLE_FILL_GROUP, ConfigError, load_pipeline, section_hash, to_dict
from rotoscope.depth import write_depth
from rotoscope.segment import write_classmap, write_probs

BG, HAIR, FACE, TORSO = tst.BG, tst.HAIR, tst.FACE, tst.TORSO
CID = tst.CID
H, W = ttm.H, ttm.W                     # 40 × 60 (klip ttm: target T di x 10–26, orang lain N di x 36–52)
N_CLIP, CUT = ttm.N, ttm.CUT            # 10 frame, cut sebelum f6
HOLE = (slice(11, 17), slice(14, 21))   # lubang 6 × 7 = 42 px di muka T (≥ island_min_px 30 → bertahan sampai VF)
HOLE_N = (slice(11, 17), slice(40, 47))
REPO = Path(__file__).resolve().parents[1]
CLIPS_DIR = REPO / "work" / "clips"
REAL_CLIPS = ("test", "test_short", "Klip2", "klip3", "klip4", "klip5")


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setattr(stage_common, "REPLACE_DELAY_S", 0)
    monkeypatch.setenv(stage_common.HF_OFFLINE_ENV, "1")


# ── fungsi murni ───────────────────────────────────
def face_map(h: int = 12, w: int = 12) -> np.ndarray:
    return np.full((h, w), FACE, np.uint8)


def test_constant_group_is_face():
    assert HOLE_FILL_GROUP == "face"


def test_enclosed_hole_filled_input_untouched():
    g = face_map()
    g[4:7, 4:7] = BG
    before = g.copy()
    out, reg, px = stb.fill_group_holes(g, FACE)
    assert (out == FACE).all() and (reg, px) == (1, 9)
    assert out.dtype == np.uint8 and out.shape == g.shape and (g == before).all()


@pytest.mark.parametrize("side", ["atas", "bawah", "kiri", "kanan"])
def test_component_touching_frame_edge_not_filled(side):
    g = face_map()
    notch = {"atas": (slice(0, 3), slice(4, 7)), "bawah": (slice(9, 12), slice(4, 7)),
             "kiri": (slice(4, 7), slice(0, 3)), "kanan": (slice(4, 7), slice(9, 12))}[side]
    g[notch] = BG
    out, reg, px = stb.fill_group_holes(g, FACE)
    assert (out == g).all() and (reg, px) == (0, 0)


@pytest.mark.parametrize("where", ["sisi", "pojok_diagonal"])
def test_one_ring_pixel_of_other_group_blocks(where):
    g = face_map()
    g[4:7, 4:7] = BG
    g[(3, 5) if where == "sisi" else (3, 3)] = TORSO      # pojok cincin = tetangga 8-arah saja
    out, reg, px = stb.fill_group_holes(g, FACE)
    assert (out == g).all() and (reg, px) == (0, 0)


def test_diagonal_background_neighbours_are_separate_and_block_each_other():
    g = face_map()
    g[4, 4] = BG
    g[5, 5] = BG                                           # bersentuhan diagonal: dua komponen 4-arah, saling di cincin
    out, reg, _ = stb.fill_group_holes(g, FACE)
    assert (out == g).all() and reg == 0
    g[8, 8] = BG                                           # lubang lain yang berdiri sendiri tetap diisi
    out, reg, px = stb.fill_group_holes(g, FACE)
    assert out[8, 8] == FACE and out[4, 4] == BG and out[5, 5] == BG and (reg, px) == (1, 1)


@pytest.mark.parametrize("size", [1, 150])
def test_no_size_limit(size):
    g = face_map(size + 10, size + 10)
    g[5:5 + size, 5:5 + size] = BG
    out, reg, px = stb.fill_group_holes(g, FACE)
    assert (out == FACE).all() and (reg, px) == (1, size * size)


def test_hole_holding_island_of_other_group_not_filled():
    g = face_map(20, 20)
    g[5:15, 5:15] = BG
    g[8:12, 8:12] = TORSO                                  # cincin dalam = torso
    out, reg, _ = stb.fill_group_holes(g, FACE)
    assert (out == g).all() and reg == 0


def test_nested_inner_hole_filled_outer_background_not():
    g = np.zeros((20, 20), np.uint8)
    g[3:17, 3:17] = FACE
    g[7:13, 7:13] = BG
    out, reg, px = stb.fill_group_holes(g, FACE)
    assert (out[3:17, 3:17] == FACE).all() and (out[:3] == BG).all() and (reg, px) == (1, 36)


def test_other_group_only_when_asked():
    g = np.full((12, 12), TORSO, np.uint8)
    g[4:7, 4:7] = BG
    assert stb.fill_group_holes(g, FACE)[1] == 0
    out, reg, _ = stb.fill_group_holes(g, TORSO)
    assert (out == TORSO).all() and reg == 1


@pytest.mark.parametrize("g", [np.zeros((10, 10), np.uint8), np.full((10, 10), FACE, np.uint8),
                               np.full((10, 10), TORSO, np.uint8)], ids=["kosong", "semua_face", "tanpa_face"])
def test_maps_without_holes(g):
    out, reg, px = stb.fill_group_holes(g, FACE)
    assert (out == g).all() and (reg, px) == (0, 0)


def random_map(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h, w = rng.integers(20, 50, size=2)
    g = np.full((h, w), FACE, np.uint8)
    for _ in range(int(rng.integers(0, 6))):               # blok grup lain / background
        y, x = rng.integers(0, h), rng.integers(0, w)
        g[y:y + rng.integers(1, 12), x:x + rng.integers(1, 12)] = rng.choice([BG, TORSO, HAIR])
    g[rng.random((h, w)) < rng.uniform(0.02, 0.3)] = BG    # bintik / lubang
    g[rng.random((h, w)) < rng.uniform(0.0, 0.05)] = TORSO
    return g


@pytest.mark.parametrize("seed", range(40))
def test_equals_prototype_oracle_on_random_maps(seed):
    g = random_map(305600 + seed)
    for gid in (FACE, TORSO):
        a, b = stb.fill_group_holes(g, gid), oracle.vf_fill(g, gid)
        assert (a[0] == b[0]).all() and a[1:] == b[1:]


def test_oracle_cases_actually_fill_something():
    """Peta acak memuat lubang interior yang terisi DAN yang tidak terisi (kalau tidak, kesetaraan di atas tidak menguji apa-apa)."""
    filled = interior = 0
    for s in range(40):
        g = random_map(305600 + s)
        h, w = g.shape
        n, _, st, _ = cv2.connectedComponentsWithStats((g == BG).astype(np.uint8), connectivity=4)
        interior += sum(1 for x, y, bw, bh, _ in st[1:] if x > 0 and y > 0 and x + bw < w and y + bh < h)
        filled += oracle.vf_fill(g, FACE)[1]
    assert filled > 40 and interior - filled > 40


def test_group_id_from_config():
    assert stb.hole_fill_group_id(load_pipeline()) == FACE
    groups = {n: list(c) for n, c in load_pipeline().groups}
    moved = {k: groups[k] for k in groups if k != "face"} | {"face": groups["face"]}     # face jadi grup terakhir
    assert stb.hole_fill_group_id(load_pipeline(overrides={"groups": moved})) == len(groups)
    assert stb.hole_fill_group_id(load_pipeline(overrides={"stabilize.hole_fill.enabled": False})) is None


# ── klip sintetis ──────────────────────────────────
def classmap(i: int, hole: bool = True, speck: bool = False) -> np.ndarray:
    """T (x 10–26, muka y 8–20 dengan lubang) + N (x 36–52, muka dengan lubang); bergeser 1 px tiap frame. `speck` = SATU piksel
    Torso di cincin bawah lubang T (pulau < 30 px → dihapus filter pulau → cincin akhir 100% face)."""
    s = i % 2
    cm = np.zeros((H, W), np.uint8)
    for x0 in (10 + s, 36):
        cm[8:32, x0:x0 + 16] = CID["Torso"]
        cm[8:20, x0:x0 + 16] = CID["Face_Neck"]
    if hole:
        cm[HOLE[0], slice(HOLE[1].start + s, HOLE[1].stop + s)] = 0
        cm[HOLE_N] = 0
    if speck:
        cm[HOLE[0].stop, HOLE[1].start + 3 + s] = CID["Torso"]
    return cm


def make_clip(tmp_path: Path, name: str, hole: bool = True, speck: bool = False) -> Path:
    """Klip 10 frame (pola `ttm.make_clip`): frames/ (cut sebelum f6), qc_report.json, kedalaman acak berseed (median foreground
    peka terhadap tiap piksel tambahan → VF yang ikut statistik kedalaman terdeteksi)."""
    work = tst.make_work(tmp_path / name, N_CLIP, H, W)
    (work / "frames").mkdir()
    for i in range(N_CLIP):
        stem = f"frame_{i:05d}"
        cm = classmap(i, hole, speck)
        write_classmap(work / "seg" / "classmap" / f"{stem}.png", cm)
        write_probs(work / "seg" / "probs" / f"{stem}.npz", tst.probs_from_classmap(cm))
        rng = np.random.default_rng(3056 + i)
        d = np.where(cm != 0, 2.0, 1.0) + 0.5 * rng.random((H, W))
        write_depth(work / "depth" / f"{stem}.npy", d.astype(np.float16))
        base = 60 if i < CUT else 200
        img = np.full((H, W, 3), base, np.uint8)
        img[0, i] = base + 9
        cv2.imwrite(str(work / "frames" / f"{stem}.png"), img)
    rows = [{"frame": f"frame_{i:05d}.png", "index": i, "fail_reasons": []} for i in range(N_CLIP)]
    (work / "qc_report.json").write_text(json.dumps({"stage": "segment_qc", "frames": rows}), encoding="utf-8")
    return work


def cfg_for(work: Path, **over):
    return load_pipeline(overrides={"paths.work_dir": str(work), **over})


def run(work: Path, logs: list | None = None, limit: int | None = None, **over) -> dict:
    return stb.run_stabilize(cfg_for(work, **over), limit=limit, log=(logs.append if logs is not None else tst.quiet))


OFF = {"stabilize.hole_fill.enabled": False}


def groups(work: Path) -> np.ndarray:
    return ttm.groups(work)


def depth_bytes(work: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted((work / "stable" / "depth_smooth").iterdir())}


def frame_rows(work: Path) -> list[dict]:
    lines = (work / "stable" / "frames.jsonl").read_text(encoding="utf-8").splitlines()
    return [r for r in map(json.loads, lines) if r.get("event") == "frame"]


def hole_px(g: np.ndarray, i: int) -> np.ndarray:
    """Bagian dalam lubang T (4 × 5 px; sudut ikut mode filter) pada peta SATU frame `g` (frame i). Dicek 2-D + tidak kosong:
    irisan tumpukan (N, H, W) akan kosong dan `.all()` lolos tanpa menguji apa pun."""
    assert g.ndim == 2
    s = i % 2
    out = g[HOLE[0].start + 1:HOLE[0].stop - 1, HOLE[1].start + 1 + s:HOLE[1].stop - 1 + s]
    assert out.size == 20
    return out


# ── integrasi [3] ──────────────────────────────────
@pytest.fixture
def pair(tmp_path):
    """(klip lubang, VF aktif — default), (klip sama, VF mati); keduanya sudah di-stabilize (temporal default)."""
    on, off = make_clip(tmp_path, "on"), make_clip(tmp_path, "off")
    run(on)
    run(off, **OFF)
    return on, off


def test_face_holes_filled_and_depth_identical(pair):
    on, off = pair
    g_on, g_off = groups(on), groups(off)
    for i in range(N_CLIP):
        assert (hole_px(g_off[i], i) == BG).all() and (hole_px(g_on[i], i) == FACE).all()
        assert (g_off[i][HOLE_N[0].start + 1:HOLE_N[0].stop - 1, HOLE_N[1].start + 1:HOLE_N[1].stop - 1] == BG).all()
        assert (g_on[i][HOLE_N[0].start + 1:HOLE_N[0].stop - 1, HOLE_N[1].start + 1:HOLE_N[1].stop - 1] == FACE).all()
    changed = g_on != g_off
    assert (g_off[changed] == BG).all() and (g_on[changed] == FACE).all()          # hanya background → face
    assert depth_bytes(on) == depth_bytes(off)                                     # kedalaman memakai peta sebelum VF


def test_depth_would_differ_with_filled_foreground(pair):
    """Penjaga test di atas: statistik kedalaman dari peta TERISI memang mengubah depth_smooth di klip ini."""
    on, _ = pair
    cfg = cfg_for(on)
    clip = stb.load_clip(on)
    name = clip.names[0]
    d = stb._read_inputs(clip, name, len(tst.CLASSES))[2]
    g = groups(on)[0]
    dc = cfg.stabilize.depth
    a = stb.normalize_depth(d, g != BG, dc.log_eps, dc.iqr_min, dc.normalize)[0]
    assert np.load(clip.depth_smooth_path(name)).tobytes() != a.tobytes()


def test_disabled_is_old_behaviour(tmp_path):
    """enabled false = langkah 1–6 tanpa VF: peta = clean_groups apa adanya (spasial, dihitung langsung), tanpa key VF di log."""
    work = make_clip(tmp_path, "w")
    logs = []
    run(work, logs, **{**OFF, "stabilize.temporal.enabled": False})
    cfg = cfg_for(work)
    lut = stb.class_group_lut(cfg.groups, tst.CLASSES)
    clip = stb.load_clip(work)
    for i, name in enumerate(clip.names):
        pr, cm, _ = stb._read_inputs(clip, name, len(tst.CLASSES))
        want = stb.clean_groups(stb.group_sums(pr, lut, len(cfg.groups)), lut[cm], cfg.stabilize.island_min_px,
                                cfg.stabilize.mode_k)[0]
        assert (groups(work)[i] == want).all() and (hole_px(want, i) == BG).all()
    assert not any("hole_fill" in r for r in frame_rows(work))
    assert not any(", VF " in x for x in logs)
    assert json.loads((work / "stable" / "manifest.json").read_text(encoding="utf-8"))["stabilize"]["hole_fill"] == {"enabled": False}


def test_clip_without_holes_identical_on_and_off(tmp_path):
    a, b = make_clip(tmp_path, "a", hole=False), make_clip(tmp_path, "b", hole=False)
    run(a)
    run(b, **OFF)
    assert ttm.stable_bytes(a) == ttm.stable_bytes(b)


def test_speck_removed_by_island_filter_before_vf(tmp_path):
    """Satu piksel Torso di cincin lubang: di argmax mentah cincin tidak 100% face; filter pulau menghapusnya → VF sesudah
    pulau + mode mengisi lubang. VF di jumlah mentah (P-pre) / sebelum filter pulau tidak mengisinya."""
    work, off = make_clip(tmp_path, "w", speck=True), make_clip(tmp_path, "off", speck=True)
    run(work)
    run(off, **OFF)
    g, g_off = groups(work), groups(off)
    for i in range(N_CLIP):
        assert (hole_px(g_off[i], i) == BG).all() and (hole_px(g[i], i) == FACE).all()


def test_frames_log_and_messages(pair, tmp_path):
    on, _ = pair
    rows = frame_rows(on)
    g = groups(on)
    assert len(rows) == N_CLIP
    for i, r in enumerate(rows):
        assert r["hole_fill_regions"] == 2 and r["hole_fill_px"] > 0
        assert r["fg_px"] == int((g[i] != BG).sum())
    work = make_clip(tmp_path, "log")
    logs = []
    run(work, logs)
    assert any(", VF face)" in x for x in logs) and any(", VF 2 wil / " in x for x in logs)


def test_limit_prefix_identical(pair, tmp_path):
    on, _ = pair
    work = make_clip(tmp_path, "lim")
    r = run(work, limit=4)
    assert r["processed"] >= 1
    full = ttm.stable_bytes(on)
    part = ttm.stable_bytes(work)
    assert part and all(full[k] == v for k, v in part.items())


def masked_clip(tmp_path: Path, empty=()) -> Path:
    work = make_clip(tmp_path, "mask")
    ttm.isolate(work, empty=empty)                          # klik T di f0 (adegan A f0–f5); SAM tiruan = lingkaran r 16
    return work


def test_with_target_masks_vf_after_mask(tmp_path):
    work = masked_clip(tmp_path)
    logs = []
    run(work, logs)
    g = groups(work)
    assert any("masker target [1b] dipakai" in x for x in logs)
    for i in range(N_CLIP):
        assert (hole_px(g[i], i) == FACE).all()                               # lubang muka target terisi
    assert not g[:CUT, :, 34:].any()                                           # adegan berklik: orang lain N dibuang
    n_in = g[CUT:, HOLE_N[0].start + 1:HOLE_N[0].stop - 1, HOLE_N[1].start + 1:HOLE_N[1].stop - 1]
    assert (n_in == FACE).all()                                               # adegan tanpa klik (keep): N digambar, lubangnya terisi
    assert "isolate" in json.loads((work / "stable" / "manifest.json").read_text(encoding="utf-8"))


def test_kosong_frame_stays_empty_with_vf(tmp_path):
    work = masked_clip(tmp_path, empty=(3,))
    run(work)
    g = groups(work)
    rows = frame_rows(work)
    assert not g[3].any() and rows[3]["hole_fill_regions"] == 0 and g[2].any() and g[4].any()


# ── basi ───────────────────────────────────────────
def test_stale_when_enabled_changes(pair):
    on, _ = pair
    logs = []
    r = run(on, logs, **OFF)
    assert r["processed"] == N_CLIP and any(s.startswith("stabilize_hash:") for s in r["stale"])
    assert (hole_px(groups(on)[0], 0) == BG).all()
    r = run(on)
    assert r["processed"] == N_CLIP and r["stale"] and (hole_px(groups(on)[0], 0) == FACE).all()
    assert run(on)["processed"] == 0


def test_old_manifest_without_hole_fill_is_stale_once(pair):
    """Manifest sebelum T-305f (section stabilize tanpa hole_fill, hash dari section itu) → basi sekali, lalu tidak lagi."""
    on, _ = pair
    p = on / "stable" / "manifest.json"
    m = json.loads(p.read_text(encoding="utf-8"))
    old = {k: v for k, v in m["stabilize"].items() if k != "hole_fill"}
    m["stabilize"] = old
    m["stabilize_hash"] = hashlib.sha256(json.dumps(old, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                                                    allow_nan=False).encode("utf-8")).hexdigest()
    p.write_text(json.dumps(m), encoding="utf-8")
    r = run(on)
    assert r["processed"] == N_CLIP and any(s.startswith("stabilize_hash:") for s in r["stale"])
    assert run(on)["processed"] == 0


def test_hash_covers_hole_fill_only_in_stabilize():
    a, b = load_pipeline(), load_pipeline(overrides=OFF)
    assert section_hash(a, "stabilize") != section_hash(b, "stabilize")
    for s in ("groups", "target_mask", "vectorize", "qc"):
        assert section_hash(a, s) == section_hash(b, s)
    assert to_dict(a.stabilize)["hole_fill"] == {"enabled": True}


# ── config ─────────────────────────────────────────
def test_default_is_rio_decision():
    assert load_pipeline().stabilize.hole_fill.enabled is True


@pytest.mark.parametrize("v", ["ya", 1, 0, None, 1.0])
def test_enabled_must_be_bool(v):
    with pytest.raises(ConfigError, match=r"stabilize\.hole_fill\.enabled"):
        load_pipeline(overrides={"stabilize.hole_fill.enabled": v})


def test_group_is_not_a_parameter():
    with pytest.raises(ConfigError, match=r"hole_fill\.group"):
        load_pipeline(overrides={"stabilize.hole_fill.group": "face"})


def test_groups_without_face_need_vf_off():
    groups_ = {n: list(c) for n, c in load_pipeline().groups}
    groups_["head"] = groups_.pop("face")
    with pytest.raises(ConfigError, match=r"stabilize\.hole_fill\.enabled.*'face'"):
        load_pipeline(overrides={"groups": groups_})
    assert load_pipeline(overrides={"groups": groups_, **OFF}).stabilize.hole_fill.enabled is False


def test_yaml_section_parses(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("stabilize:\n  hole_fill:\n    enabled: false\n", encoding="utf-8")
    assert load_pipeline(p).stabilize.hole_fill.enabled is False


# ── klip nyata (dilewati bila stable/ belum dihitung dengan VF) ──
def real_face_id() -> int:
    names = [n for n, _ in load_pipeline().groups]
    return names.index(HOLE_FILL_GROUP) + 1


@pytest.mark.parametrize("clip", REAL_CLIPS)
def test_real_clip_has_no_fillable_face_hole(clip):
    """stable/ klip nyata yang dihitung dengan VF aktif: VF lagi tidak mengubah apa pun (idempoten)."""
    m_path = CLIPS_DIR / clip / "stable" / "manifest.json"
    if not m_path.is_file():
        pytest.skip(f"{clip}: stable/ tidak ada")
    m = json.loads(m_path.read_text(encoding="utf-8"))
    if (m.get("stabilize") or {}).get("hole_fill") != {"enabled": True}:
        pytest.skip(f"{clip}: stable/ belum dihitung dengan VF (T-305f)")
    c = stb.load_clip(CLIPS_DIR / clip)
    fid = real_face_id()
    bad = [n for n in c.names if stb.fill_group_holes(stb.read_groups(c.groups_path(n)), fid)[1]]
    assert not bad, f"{clip}: {len(bad)} frame masih punya lubang yang memenuhi aturan VF, mis. {bad[:3]}"
