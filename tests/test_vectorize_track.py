"""Test stage [4] dengan pelacakan (T-202): skema final, rantai prev_sha256 (resume / kerusakan / pengubahan tangan),
determinisme, prefiks --limit, regresi kanonik terhadap strok mentah, plus data nyata (ditandai; dilewati bila klip /
contours T-202 tidak ada). Fungsi metrik: tests/track_metrics.py."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import track_metrics as tm
from test_vectorize import (FACE, HAIR, LARM, NAMES, PARAMS, TORSO, cfg_for, frame_hashes, make_work, quiet,
                            subject_depth)

from rotoscope import stabilize as stb
from rotoscope import vectorize as vec

RH, RW = 100, 80
N = 5


def rich_groups(i: int) -> np.ndarray:
    """Subjek sintetis bergeser 0–2 px per frame: torso berlubang (ruang negatif 18×18 ≥ min_hole_area), lengan, wajah, rambut."""
    d = i % 3
    g = np.zeros((RH, RW), np.uint8)
    g[30:95, 20 + d:60 + d] = TORSO
    g[45:63, 30 + d:48 + d] = 0
    g[35:70, 8 + d:20 + d] = LARM
    g[15:30, 28 + d:48 + d] = FACE
    g[10:15, 30 + d:46 + d] = HAIR
    return g


def make_rich(tmp_path: Path, n: int = N) -> Path:
    work = make_work(tmp_path, n=n, h=RH, w=RW)
    for i in range(n):
        stb.write_groups(work / "stable" / "groups" / f"frame_{i:05d}.png", rich_groups(i))
        stb.write_depth_smooth(work / "stable" / "depth_smooth" / f"frame_{i:05d}.npy", subject_depth(i, RH, RW))
    return work


def cpath(work: Path, i: int) -> Path:
    return work / "contours" / f"frame_{i:05d}.json"


def docs(work: Path) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted((work / "contours").glob("frame_*.json"))]


def strokes_of(work: Path) -> list[list[dict]]:
    return [d["strokes"] for d in docs(work)]


# ── Skema + rantai ─────────────────────────────────
def test_final_schema_keys_per_type_and_chain(tmp_path):
    work = make_rich(tmp_path)
    run = vec.run_vectorize(cfg_for(work), log=quiet)
    assert run["processed"] == N
    ds = docs(work)
    kinds = {s["type"] for d in ds for s in d["strokes"]}
    assert kinds >= {"silhouette", "silhouette_hole", "group_boundary"}
    for k, d in enumerate(ds):
        assert list(d) == ["frame_index", "width", "height", "source", "prev_sha256", "strokes"]
        assert d["prev_sha256"] == (None if k == 0 else vec.bytes_sha256(cpath(work, k - 1).read_bytes()))
        for s in d["strokes"]:
            assert tuple(s) == vec.final_keys(s["type"])
            assert type(s["track_id"]) is int and s["track_id"] >= 1
            if s["closed"]:
                assert s["anchor"] == 0
    m = json.loads((work / "contours" / "manifest.json").read_text(encoding="utf-8"))
    assert m["contract"] == "T-202" and m["algo_rev"] == 2 and m["pending"] == []
    assert m["vectorize"]["track.max_match_dist_px"] == 16 and "track.max_match_dist_px" in m["vectorize"]
    recs = list(json.loads(x) for x in (work / "contours" / "frames.jsonl").read_text(encoding="utf-8").splitlines())
    frames = [r for r in recs if r["event"] == "frame"]
    assert len(frames) == N and all({"track_s", "n_new_ids", "n_matched", "n_flipped_vs_static"} <= set(r) for r in frames)
    assert frames[0]["n_matched"] == 0 and frames[0]["n_new_ids"] == len(ds[0]["strokes"])


def test_run_metrics_orientation_ids_reversals(tmp_path):
    work = make_rich(tmp_path)
    vec.run_vectorize(cfg_for(work), log=quiet)
    fr = strokes_of(work)
    o = tm.orientation(fr)
    assert o["silhouette_cw_pct"] == 100.0 and o["hole_ccw_pct"] == 100.0 and o["loop_cw_pct"] == 100.0
    assert len(tm.main_silhouette_ids(fr)) == 1
    assert tm.duplicate_ids(fr) == 0 and tm.reversals(fr)["long_tracks_reversed"] == []
    assert max(tm.main_anchor_jumps(fr)) <= 3.0
    assert tm.churn(fr)["new_total"]["silhouette"] == 0


def test_hair_face_anchor_for_new_silhouette_in_run(tmp_path):
    """Frame 0: anchor silhouette = titik kontur terdekat ke piksel hair tertinggi (y = 10, x = 30)."""
    work = make_rich(tmp_path)
    vec.run_vectorize(cfg_for(work), log=quiet)
    (sil,) = [s for s in strokes_of(work)[0] if s["type"] == "silhouette"]
    pts = np.array(sil["points"])
    d = np.hypot(pts[:, 0] - 30.5, pts[:, 1] - 10.5)
    assert np.isclose(d[0], d.min()) and sil["points"][0] == [30.5, 10.5]


def test_regression_canonical_equals_raw_strokes_all_types(tmp_path):
    """Hanya URUTAN titik dalam strok yang berubah: hash kanonik per tipe = strok mentah vectorize_frame."""
    work = make_rich(tmp_path)
    cfg = cfg_for(work)
    vec.run_vectorize(cfg, log=quiet)
    m = json.loads((work / "contours" / "manifest.json").read_text(encoding="utf-8"))
    t = m["depth_thresholds"]
    params = vec.vectorize_params(cfg)
    raw = []
    for i in range(N):
        g = stb.read_groups(work / "stable" / "groups" / f"frame_{i:05d}.png")
        d = vec.read_depth_smooth(work / "stable" / "depth_smooth" / f"frame_{i:05d}.npy", RH, RW)
        raw.append(vec.vectorize_frame(g, d, NAMES, params, t["t_high"], t["t_low"])[0])
    final = strokes_of(work)
    assert tm.canonical_hashes(final) == tm.canonical_hashes(raw)
    assert [[(s["type"], s["groups"]) for s in f] for f in final] == [[(s["type"], s["groups"]) for s in f] for f in raw]
    assert [[len(s["points"]) for s in f] for f in final] == [[len(s["points"]) for s in f] for f in raw]


def test_two_runs_from_scratch_byte_identical_and_restart(tmp_path):
    work = make_rich(tmp_path)
    vec.run_vectorize(cfg_for(work), log=quiet)
    first = frame_hashes(work)
    vec.run_vectorize(cfg_for(work), restart=True, log=quiet)
    assert frame_hashes(work) == first and len(first) == N


def test_limit_is_prefix_of_full_run_and_resume_completes_identically(tmp_path):
    full = make_rich(tmp_path / "full")
    vec.run_vectorize(cfg_for(full), log=quiet)
    part = make_rich(tmp_path / "part")
    vec.run_vectorize(cfg_for(part), limit=2, log=quiet)
    got = frame_hashes(part)
    assert got == {k: v for k, v in frame_hashes(full).items() if k in got} and len(got) == 2
    run = vec.run_vectorize(cfg_for(part), log=quiet)
    assert (run["skipped"], run["processed"]) == (2, N - 2)
    assert frame_hashes(part) == frame_hashes(full)
    assert vec.run_vectorize(cfg_for(part), log=quiet)["processed"] == 0


# ── Rantai prev_sha256 ─────────────────────────────
def test_chain_a_recomputed_identical_predecessor_does_not_recompute_next(tmp_path):
    work = make_rich(tmp_path)
    vec.run_vectorize(cfg_for(work), log=quiet)
    good = frame_hashes(work)
    mtimes = {i: cpath(work, i).stat().st_mtime_ns for i in range(N)}
    cpath(work, 2).unlink()
    run = vec.run_vectorize(cfg_for(work), log=quiet)
    assert (run["processed"], run["skipped"]) == (1, N - 1)
    assert frame_hashes(work) == good
    assert all(cpath(work, i).stat().st_mtime_ns == mtimes[i] for i in (0, 1, 3, 4))


def test_chain_b_changed_predecessor_recomputes_it_and_all_after_identical_to_full_run(tmp_path):
    work = make_rich(tmp_path)
    vec.run_vectorize(cfg_for(work), log=quiet)
    before = frame_hashes(work)
    g = rich_groups(1)
    g[45:63, 30:48] = TORSO                                     # frame 1 kehilangan lubang → hasil lain
    stb.write_groups(work / "stable" / "groups" / "frame_00001.png", g)
    cpath(work, 1).unlink()
    run = vec.run_vectorize(cfg_for(work), log=quiet)
    assert (run["processed"], run["skipped"]) == (N - 1, 1)
    after = frame_hashes(work)
    assert after["frame_00000.json"] == before["frame_00000.json"]
    assert all(after[f"frame_{i:05d}.json"] != before[f"frame_{i:05d}.json"] for i in range(1, N))
    vec.run_vectorize(cfg_for(work), restart=True, log=quiet)       # run penuh dari nol atas masukan yang sama
    assert frame_hashes(work) == after


def test_chain_c_valid_frame_without_valid_predecessor_is_not_valid(tmp_path):
    work = make_rich(tmp_path)
    vec.run_vectorize(cfg_for(work), log=quiet)
    clip = vec.load_clip(work)
    m = json.loads((work / "contours" / "manifest.json").read_text(encoding="utf-8"))
    src = vec.frame_source(m)
    names = clip.names
    prev = cpath(work, 1).read_bytes()
    assert vec.read_valid_frame(cpath(work, 2), 2, clip, src, prev, False) is not None
    assert vec.read_valid_frame(cpath(work, 2), 2, clip, src, None, False) is None            # k-1 hilang
    assert vec.read_valid_frame(cpath(work, 2), 2, clip, src, prev + b" ", False) is None     # k-1 berubah
    assert vec.read_valid_frame(cpath(work, 0), 0, clip, src, None, True) is not None
    assert vec.read_valid_frame(cpath(work, 0), 0, clip, src, prev, True) is not None       # frame pertama: tanpa pendahulu
    d = json.loads(cpath(work, 0).read_text(encoding="utf-8"))
    d["prev_sha256"] = "0" * 64                                                               # frame pertama tak boleh punya rantai
    cpath(work, 0).write_text(json.dumps(d), encoding="utf-8")
    assert vec.read_valid_frame(cpath(work, 0), 0, clip, src, None, True) is None
    assert names[0] == "frame_00000.png"


def test_chain_deleted_middle_frame_with_missing_predecessor_recomputed_in_order(tmp_path):
    work = make_rich(tmp_path)
    vec.run_vectorize(cfg_for(work), log=quiet)
    good = frame_hashes(work)
    cpath(work, 1).unlink()
    cpath(work, 2).unlink()
    run = vec.run_vectorize(cfg_for(work), log=quiet)
    assert run["processed"] == 2 and frame_hashes(work) == good


@pytest.mark.parametrize("damage", ["truncate", "edit_valid_json"])
def test_chain_d_damaged_or_hand_edited_frame_restored_identical(tmp_path, damage):
    work = make_rich(tmp_path)
    vec.run_vectorize(cfg_for(work), log=quiet)
    good = frame_hashes(work)
    p = cpath(work, 2)
    if damage == "truncate":
        p.write_text("{potong", encoding="utf-8")
    else:                                                     # JSON tetap valid + skema final utuh, isi diubah
        d = json.loads(p.read_text(encoding="utf-8"))
        d["strokes"][0]["points"].reverse()
        p.write_text(json.dumps(d, separators=(",", ":")) + "\n", encoding="utf-8")
    run = vec.run_vectorize(cfg_for(work), log=quiet)
    assert run["processed"] == 1 and frame_hashes(work) == good


def test_chain_first_frame_damaged_recomputes_only_if_bytes_change(tmp_path):
    work = make_rich(tmp_path)
    vec.run_vectorize(cfg_for(work), log=quiet)
    good = frame_hashes(work)
    cpath(work, 0).write_text("x", encoding="utf-8")
    run = vec.run_vectorize(cfg_for(work), log=quiet)
    assert run["processed"] == 1 and frame_hashes(work) == good


def test_old_frames_without_track_fields_are_not_reused(tmp_path):
    """Frame T-201b (tanpa track_id / prev_sha256) di bawah manifest T-202 yang sama tidak valid → dihitung ulang."""
    work = make_rich(tmp_path)
    vec.run_vectorize(cfg_for(work), log=quiet)
    good = frame_hashes(work)
    d = json.loads(cpath(work, 3).read_text(encoding="utf-8"))
    for s in d["strokes"]:
        s.pop("track_id")
    cpath(work, 3).write_text(json.dumps(d, separators=(",", ":")) + "\n", encoding="utf-8")
    run = vec.run_vectorize(cfg_for(work), log=quiet)
    assert run["processed"] == 1 and frame_hashes(work) == good


def test_empty_frame_in_the_middle_keeps_chain_and_counter(tmp_path):
    work = make_rich(tmp_path)
    stb.write_groups(work / "stable" / "groups" / "frame_00002.png", np.zeros((RH, RW), np.uint8))
    vec.run_vectorize(cfg_for(work), log=quiet)
    fr = strokes_of(work)
    assert fr[2] == []
    before_max = max(s["track_id"] for f in fr[:2] for s in f)
    assert all(s["track_id"] > before_max for s in fr[3])                     # tanpa padanan → id baru, penghitung lanjut
    ds = docs(work)
    assert ds[3]["prev_sha256"] == vec.bytes_sha256(cpath(work, 2).read_bytes())
    again = frame_hashes(work)
    cpath(work, 3).unlink()
    vec.run_vectorize(cfg_for(work), log=quiet)
    assert frame_hashes(work) == again


@pytest.mark.parametrize("inherit", [True, False])
def test_stage_main_silhouette_inherits_id_beyond_threshold_only_with_switch(tmp_path, monkeypatch, inherit):
    """Subjek melompat 40 px (Chamfer > 16): dengan saklar Y id silhouette utama tetap, tanpa saklar id baru."""
    monkeypatch.setattr(vec, "INHERIT_MAIN_SILHOUETTE", inherit)
    work = make_rich(tmp_path, n=2)
    g = np.zeros((RH, RW), np.uint8)
    g[30:95, 60:79] = TORSO                                   # blok lain, jauh dari posisi frame 0
    stb.write_groups(work / "stable" / "groups" / "frame_00001.png", g)
    vec.run_vectorize(cfg_for(work), log=quiet)
    fr = strokes_of(work)
    ids = [next(s for s in f if s["type"] == "silhouette")["track_id"] for f in fr]
    assert (ids[0] == ids[1]) is inherit


def test_successor_agrees_helper(tmp_path):
    work = make_rich(tmp_path)
    vec.run_vectorize(cfg_for(work), log=quiet)
    clip = vec.load_clip(work)
    data = cpath(work, 1).read_bytes()
    assert vec.successor_agrees(clip, 1, data) and not vec.successor_agrees(clip, 1, data + b" ")
    assert vec.successor_agrees(clip, N - 1, b"apa saja")                     # frame terakhir: tidak ada bukti
    cpath(work, 2).write_text("{potong", encoding="utf-8")
    assert vec.successor_agrees(clip, 1, data + b" ")                         # pengganti rusak: tidak ada bukti


# ── Data nyata (ditandai; dilewati bila contours T-202 klip tidak ada) ──
# Hash kanonik contours klip nyata: orientasi / anchor / arah tidak boleh mengubah himpunan titik (regresi kode vectorize).
# DIPERBARUI di T-302 Tahap 4: `stable/` klip nyata sekarang temporal + `log_median` (default baru), jadi himpunan titik berubah
# SAH. Hash lama (T-201b, `stable/` spasial + `log_median_iqr`) ada di git history (commit T-202); nilai di bawah = keluaran kode
# vectorize yang sama (tidak diubah di T-302) atas `stable/` baru, jadi tetap mendeteksi perubahan tak sengaja di vectorize.
BASELINE = {
    "test_short": {
        "silhouette": "7354bf507a26f0d526e02e836b30f62f854e0aac221ace6dbba358a530528be5",
        "silhouette_hole": "4450e05c4d64b45fff154add932dc705542c2ba361ed341457192a9368fb2ac8",
        "group_boundary": "4a64c5226ecb869618232c9e06ac291d1880bcc42a2d0a38984e06368060f184",
        "occlusion": "e512c7280e682151fbe5f7368d35d4fc33a2a3fa56a862633db912ccab4e615c"},
    "test": {
        "silhouette": "a41a0411e352c3c090a9d98e6482b5b119da867d33696c49954c2427df99742a",
        "silhouette_hole": "cdb6ed48ff1bd1249c7aaaed6c3d0b5860fab68c952b5c214d2a477ecaa26464",
        "group_boundary": "1a91ddf3fdcf973057c404f4896e9570d2b11388230b6590ff40c9e7c490e105",
        "occlusion": "c32a9ab8e5b4d717e7e7a3e5dabe25e018d0e3f5b8e680deab90aec8c66a6c9c"},
}
ANCHOR_MEDIAN_MAX, ANCHOR_P95_MAX, ANCHOR_MAX_MAX = 3.0, 12.0, 24.0       # kriteria lulus (docs/05 T-202, poin 11c)
CLIPS = ["test_short", "test"]


def real_frames(clip: str) -> list[list[dict]]:
    d = Path("work/clips") / clip / "contours"
    try:
        m = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pytest.skip(f"contours {clip} tidak ada")
    if m.get("contract") != "T-202":
        pytest.skip(f"contours {clip} bukan T-202")
    return tm.load_frames(d)


@pytest.mark.parametrize("clip", CLIPS)
def test_real_orientation_ids_reversals_duplicates(clip):
    fr = real_frames(clip)
    o = tm.orientation(fr)
    assert o["silhouette_cw_pct"] == 100.0 and o["hole_ccw_pct"] == 100.0 and o["loop_cw_pct"] == 100.0
    assert tm.duplicate_ids(fr) == 0
    assert tm.reversals(fr)["long_tracks_reversed"] == []


@pytest.mark.parametrize("clip", CLIPS)
def test_real_main_anchor_jump_within_criteria(clip):
    """Kriteria 11c. Tanpa pendekatan Y, ganti id silhouette utama (lengan terlepas, klip test) mengembalikan anchor ke
    titik tertinggi hair ∪ face: lompatan maks 65 px (default lama 12 px)."""
    p = tm.percentiles(tm.main_anchor_jumps(real_frames(clip)), (50, 95, 100))
    assert p["p50"] <= ANCHOR_MEDIAN_MAX and p["p95"] <= ANCHOR_P95_MAX and p["p100"] <= ANCHOR_MAX_MAX, p


@pytest.mark.parametrize("clip", CLIPS)
def test_real_main_silhouette_single_track_id(clip):
    assert len(tm.main_silhouette_ids(real_frames(clip))) == 1


@pytest.mark.parametrize("clip", ["test_short", "test"])
def test_real_canonical_hashes_identical_to_t201b(clip):
    assert tm.canonical_hashes(real_frames(clip)) == BASELINE[clip]


def test_params_module_constants_sane():
    assert vec.CONTRACT == "T-202" and vec.ALGO_REV == 2 and vec.PENDING == ()
    assert vec.INHERIT_MAIN_SILHOUETTE is True and PARAMS["min_stroke_px"] == 6
