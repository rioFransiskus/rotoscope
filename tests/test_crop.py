"""Test crop.py (T-607, D-013): aturan R2 beku, pembulatan kotak, petunjuk, berkas per sumber (deteksi / saran / kotak), pratinjau,
alat `crop`, dan ingest yang memakainya (saran tanpa memotong, kotak diterima, D1, identitas sumber). Video sintetis via ffmpeg
lavfi (konten = testsrc + noise temporal, pita hitam lewat pad); tanpa GPU."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from rotoscope import crop as cr
from rotoscope import ingest as ing
from rotoscope.config import CropConfig, CropDetectConfig, CropPreviewConfig

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
                                  reason="ffmpeg/ffprobe tidak ada di PATH")
P = dict(cr.FROZEN_PARAMS)


# ── Larik sintetis untuk R2 ────────────────────────
def synth(H=100, W=80, bands=None, overlay=None, static_rows=None):
    """m (H, W, 3) / sd (H, W): konten bergerak (sd 20, warna acak) + pita hitam statis seragam per sisi + baris kotak teks (statis
    tetapi tidak seragam) di sebelah dalam pita."""
    rng = np.random.default_rng(0)
    m = rng.uniform(40, 220, (H, W, 3)).astype(np.float32)
    sd = np.full((H, W), 20.0, np.float32)
    bands, overlay = bands or {}, overlay or {}
    sl = {"top": lambda n: np.s_[:n, :], "bottom": lambda n: np.s_[H - n:, :],
          "left": lambda n: np.s_[:, :n], "right": lambda n: np.s_[:, W - n:]}
    for e, n in overlay.items():                   # kotak teks: statis, tidak seragam, tepat di dalam pita
        b = bands.get(e, 0)
        region = {"top": np.s_[b:b + n, :], "bottom": np.s_[H - b - n:H - b, :],
                  "left": np.s_[:, b:b + n], "right": np.s_[:, W - b - n:W - b]}[e]
        sd[region] = 0.0
    for e, n in bands.items():
        if n:
            m[sl[e](n)] = 0.0
            sd[sl[e](n)] = 0.0
    for y in static_rows or []:
        sd[y, :] = 0.0
    return m, sd


def test_fingerprint_default_is_frozen_and_changes_with_params():
    assert cr.rule_fingerprint(P) == cr.FROZEN_FINGERPRINT
    assert cr.r2_params(CropDetectConfig()) == P
    for k in P:
        q = dict(P)
        q[k] = q[k] + 1 if isinstance(q[k], int) else q[k] * 0.5
        assert cr.rule_fingerprint(q) != cr.FROZEN_FINGERPRINT, k


def test_r2_paired_letterbox_is_cut():
    d = cr.detect_crop(*synth(bands={"top": 10, "bottom": 12}), P)
    assert d["box_work"] == [0, 10, 79, 87] and d["band_px"]["top"] == 10 and d["band_px"]["bottom"] == 12


def test_r2_pillarbox_is_cut():
    d = cr.detect_crop(*synth(bands={"left": 9, "right": 11}), P)
    assert d["box_work"] == [9, 0, 68, 99]


def test_r2_one_sided_band_is_not_cut():
    d = cr.detect_crop(*synth(bands={"top": 15}), P)
    assert d["box_work"] == [0, 0, 79, 99]


@pytest.mark.parametrize("n, cut", [(7, False), (8, True)])
def test_r2_band_min_px_boundary(n, cut):
    d = cr.detect_crop(*synth(bands={"top": n, "bottom": n}), P)
    assert (d["box_work"] != [0, 0, 79, 99]) is cut


def test_r2_text_box_top_peeled_after_band():
    d = cr.detect_crop(*synth(bands={"top": 10, "bottom": 10}, overlay={"top": 6}), P)
    assert d["box_work"][1] == 16 and d["overlay_px"]["top"] == 6 and d["band_px"]["top"] == 10


def test_r2_text_box_not_peeled_on_sides():
    d = cr.detect_crop(*synth(bands={"left": 10, "right": 10}, overlay={"left": 6}), P)
    assert d["box_work"][0] == 10 and d["overlay_px"]["left"] == 0


def test_r2_text_box_without_band_is_not_cut():
    """klip6 / klip7 (keputusan Rio): kotak teks tanpa pita = bagian video."""
    d = cr.detect_crop(*synth(overlay={"top": 6}), P)
    assert d["box_work"] == [0, 0, 79, 99]


def test_r2_static_but_not_uniform_band_is_not_band():
    m, sd = synth()
    sd[:10, :] = 0.0
    sd[-10:, :] = 0.0                              # statis tetapi warna acak = bukan pita
    assert cr.detect_crop(m, sd, P)["box_work"] == [0, 0, 79, 99]


def test_r2_min_active_guard():
    d = cr.detect_crop(*synth(bands={"top": 46, "bottom": 46}), P)     # sisa 8 baris = 8% < 10%
    assert d["box_work"] == [0, 0, 79, 99] and not d["applied"]


def test_stats_population_std():
    st = cr.Stats()
    for v in (0, 10):
        st.add(np.full((2, 2, 3), v, np.uint8))
    m, sd = st.result()
    assert m.dtype == np.float32 and float(m[0, 0, 0]) == 5.0 and float(sd[0, 0]) == 5.0


# ── Pembulatan ─────────────────────────────────────
def test_map_box_to_source_klip2_and_klip5():
    assert cr.map_box_to_source([245, 0, 474, 405], (720, 406), (1280, 720)) == [434, 0, 412, 720]   # docs/04 Tahap 1a
    assert cr.map_box_to_source([0, 344, 719, 1011], (720, 1280), (720, 1280)) == [0, 344, 720, 668]


def test_map_box_to_source_clamped_at_odd_source_edge():
    assert cr.map_box_to_source([0, 0, 479, 269], (480, 270), (481, 271)) == [0, 0, 481, 271]


def test_map_box_to_source_is_outward_and_even():
    rng = np.random.default_rng(1)
    for _ in range(200):
        Ww, Hw, Ws, Hs = 720, 406, 1280, 720
        x0, y0 = int(rng.integers(0, 300)), int(rng.integers(0, 150))
        x1, y1 = int(rng.integers(x0 + 10, Ww)), int(rng.integers(y0 + 10, Hw))
        x, y, w, h = cr.map_box_to_source([x0, y0, x1, y1], (Ww, Hw), (Ws, Hs))
        assert x % 2 == 0 and y % 2 == 0 and (w % 2 == 0 or x + w == Ws) and (h % 2 == 0 or y + h == Hs)
        assert x <= x0 * Ws / Ww and y <= y0 * Hs / Hw                              # konten tidak terpotong
        assert x + w >= (x1 + 1) * Ws / Ww - 1e-9 and y + h >= (y1 + 1) * Hs / Hw - 1e-9
        assert x + w <= Ws and y + h <= Hs


def test_round_manual_box_outward_even_and_validation():
    assert cr.round_manual_box([1, 3, 101, 51], (320, 240)) == [0, 2, 102, 52]
    assert cr.round_manual_box([0, 462, 720, 594], (720, 1280)) == [0, 462, 720, 594]    # perintah petunjuk klip11 tidak berubah
    for bad, msg in (([0, 0, 321, 10], "di luar frame"), ([-1, 0, 10, 10], "di luar frame"), ([0, 0, 10, 10], "minimal"),
                     ([0, 0, 320, 240], "--none")):
        with pytest.raises(cr.CropError, match=msg):
            cr.round_manual_box(bad, (320, 240))


# ── Petunjuk ───────────────────────────────────────
def test_hints_mark_static_rows_inside_cut_side_and_alt_box():
    m, sd = synth(bands={"top": 10, "bottom": 10})
    box = [0, 10, 79, 89]
    sd[10:16, :] = 0.0                             # 6 baris diam tepat di dalam garis potong atas (sisa kotak teks)
    sd[16, :76] = 0.0                              # baris ke-7: 76 / 80 = 95% diam → ikut (batas inklusif)
    sd[17, :75] = 0.0                              # 75 / 80 < 95% → berhenti
    sd[85:90, :] = 0.0                             # bawah juga diam
    h = cr.hints(sd, box, cr.cut_px(box, 80, 100), 0.95, P)
    assert h["lines"] == {"top": [10, 16], "bottom": [85, 89]} and h["alt_box_work"] == [0, 17, 79, 84]


def test_hints_only_on_cut_sides():
    m, sd = synth(bands={"top": 10, "bottom": 10})
    box = [0, 10, 79, 89]
    sd[:, 0:5] = 0.0                               # kolom diam di kiri, tetapi kiri TIDAK dipotong
    assert cr.hints(sd, box, cr.cut_px(box, 80, 100), 0.95, P) is None


def test_hints_alt_box_dropped_when_below_min_active():
    m, sd = synth(bands={"top": 10, "bottom": 10})
    box = [0, 10, 79, 89]
    sd[10:85, :] = 0.0                             # hampir seluruh kotak diam → kotak alternatif < 10% frame
    h = cr.hints(sd, box, cr.cut_px(box, 80, 100), 0.95, P)
    assert h["lines"]["top"][0] == 10 and h["alt_box_work"] is None


# ── Berkas kotak + identitas ───────────────────────
IDENT = {"sha256": "a" * 64, "bytes": 1000, "width": 320, "height": 240, "path": "C:/v/a.mp4"}


def test_kotak_write_read_prev_and_identity_without_path(tmp_path):
    d = tmp_path / "crop" / "a"
    assert cr.read_kotak(d) is None and cr.kotak_for_ingest(d, IDENT, "v") is None
    assert cr.write_kotak(d, cr.kotak_doc("manual", IDENT, [0, 2, 100, 50], requested=[1, 3, 99, 48])) is False
    first = (d / cr.KOTAK_FILENAME).read_bytes()
    assert cr.write_kotak(d, cr.kotak_doc("none", IDENT)) is True
    assert (d / cr.KOTAK_PREV_FILENAME).read_bytes() == first
    moved = {**IDENT, "path": "D:/lain/salinan.mp4"}                     # path = info saja
    assert cr.kotak_for_ingest(d, moved, "v")["mode"] == "none"


@pytest.mark.parametrize("key, value", [("sha256", "b" * 64), ("bytes", 1001), ("width", 322), ("height", 238)])
def test_kotak_for_other_source_is_rejected(tmp_path, key, value):
    d = tmp_path / "c"
    cr.write_kotak(d, cr.kotak_doc("manual", IDENT, [0, 2, 100, 50], requested=[0, 2, 100, 50]))
    with pytest.raises(cr.CropError, match="sumber LAIN.*--clear"):
        cr.kotak_for_ingest(d, {**IDENT, key: value}, "v")


def test_corrupt_or_outside_kotak_is_rejected(tmp_path):
    d = tmp_path / "c"
    d.mkdir()
    (d / cr.KOTAK_FILENAME).write_text("{bukan json", encoding="utf-8")
    with pytest.raises(cr.CropError, match="rusak"):
        cr.read_kotak(d)
    cr.write_kotak(d, cr.kotak_doc("manual", IDENT, [300, 0, 40, 40], requested=[300, 0, 40, 40]))
    with pytest.raises(cr.CropError, match="di luar frame"):
        cr.kotak_for_ingest(d, IDENT, "v")


SARAN = {"format": cr.FORMAT_SARAN, "key": {"source": {k: IDENT[k] for k in cr.IDENTITY_KEYS}}, "detect_frame_size": [320, 240],
         "box_work": [0, 20, 319, 219], "box_source": [0, 20, 320, 200], "band_px": {"top": 15, "bottom": 20, "left": 0, "right": 0},
         "overlay_px": {"top": 5, "bottom": 0, "left": 0, "right": 0}, "cut_px": {"top": 20, "bottom": 20, "left": 0, "right": 0},
         "rule": "R2", "rule_version": 1, "rule_fingerprint": cr.FROZEN_FINGERPRINT, "hint_static_frac": 0.95,
         "hints": {"lines": {"top": [20, 31]}, "count": {"top": 12}, "alt_box_work": [0, 32, 319, 219],
                   "alt_box_source": [0, 32, 320, 188]}}


def test_meta_crop_key_only_when_box_used():
    assert cr.meta_crop_key(None) is None and cr.meta_crop_key(cr.kotak_doc("none", IDENT)) is None
    k = cr.meta_crop_key(cr.kotak_doc("saran", IDENT, SARAN["box_source"], saran=SARAN))
    assert k["mode"] == "saran" and k["box_source"] == [0, 20, 320, 200] and k["source_sha256"] == IDENT["sha256"]
    assert k["rule_fingerprint"] == cr.FROZEN_FINGERPRINT and k["box_work"] == SARAN["box_work"] and "box_requested" not in k
    m = cr.meta_crop_key(cr.kotak_doc("manual", IDENT, [0, 2, 100, 50], requested=[1, 3, 99, 48]))
    assert m["box_requested"] == [1, 3, 99, 48] and "rule" not in m and m["rounding"] == cr.ROUNDING


def test_saran_lines_offer_accept_box_none_and_hint_command(tmp_path):
    text = "\n".join(cr.saran_lines(SARAN, tmp_path / "crop" / "a", "C:/v/a.mp4"))
    assert "BELUM dipotong" in text and "atas 20 (pita 15 + kotak teks 5)" in text and "bawah 20" in text
    assert str(tmp_path / "crop" / "a" / cr.PREVIEW_DIRNAME) in text
    assert "crop C:/v/a.mp4 --accept" in text and "--box X Y W H" in text and "crop C:/v/a.mp4 --none" in text
    assert "atas 12 baris (20–31)" in text and "crop C:/v/a.mp4 --box 0 32 320 188" in text
    assert "--crop auto" not in text


def test_alt_command_not_printed_when_equal_to_saran_after_rounding(tmp_path):
    """Usulan A (Tahap 4): kotak alternatif = kotak saran sesudah pembulatan keluar + genap → perintah --box tidak dicetak."""
    alt_work = [0, 21, 319, 219]                                         # 1 baris petunjuk (20); y 21 → genap ke bawah = 20
    alt_src = cr.map_box_to_source(alt_work, (320, 240), (320, 240))
    assert alt_src == SARAN["box_source"]
    same = {**SARAN, "hints": {"lines": {"top": [20, 20]}, "count": {"top": 1}, "alt_box_work": alt_work, "alt_box_source": alt_src}}
    text = "\n".join(cr.saran_lines(same, tmp_path, "C:/v/a.mp4"))
    assert "atas 1 baris (20–20)" in text and "tidak ada perintah lain" in text      # petunjuk tetap disebut
    assert "Kotak tanpa baris / kolom petunjuk:" not in text and "--box 0 20 320 200" not in text
    assert "tidak disarankan" not in text and "crop C:/v/a.mp4 --accept" in text
    info = cr._preview_info(same)[-1]
    assert "tanpa itu = kotak saran" in info and "--box" not in info
    assert "tanpa itu: --box 0 32 320 188" in cr._preview_info(SARAN)[-1]


def test_pending_saran_only_without_kotak(tmp_path):
    d = tmp_path / "c"
    assert cr.pending_saran(d) is None
    d.mkdir()
    (d / cr.SARAN_FILENAME).write_text(json.dumps(SARAN), encoding="utf-8")
    assert cr.pending_saran(d)["box_source"] == SARAN["box_source"]
    cr.write_kotak(d, cr.kotak_doc("none", IDENT))
    assert cr.pending_saran(d) is None


# ── Pratinjau ──────────────────────────────────────
def test_preview_boards_result_panel_hint_shading_and_cut_line(tmp_path):
    rng = np.random.default_rng(2)
    img = rng.integers(0, 255, (120, 100, 3), dtype=np.uint8)
    box, lines = [0, 30, 99, 99], {"top": [30, 39]}
    frames = [(lab, i, img) for lab, i in zip(cr.PREVIEW_FRAMES, (0, 5, 9))]
    doc = cr.write_preview(tmp_path, frames, box, {"lines": lines}, "saran", IDENT, ["info"])
    names = sorted(p.name for p in (tmp_path / cr.PREVIEW_DIRNAME).glob("*.png"))
    assert names == ["1_awal_f00000.png", "2_tengah_f00005.png", "3_akhir_f00009.png"] and doc["hint_lines"] == lines
    board = cv2.imdecode(np.fromfile(tmp_path / cr.PREVIEW_DIRNAME / names[0], np.uint8), cv2.IMREAD_COLOR)
    head = cr.HEADER_PAD_PX + cr.HEADER_LINE_PX * 3
    body = board[head:]
    assert body.shape[0] == 120
    res = body[:, -100:]                                        # panel hasil potong (baris 30..99, dipusatkan)
    oy = (120 - 70) // 2
    assert np.array_equal(res[oy + 10:oy + 70], img[40:100])     # di luar petunjuk = frame apa adanya
    shaded = res[oy:oy + 10].astype(int)
    assert not np.array_equal(shaded, img[30:40].astype(int))   # baris petunjuk diarsir
    expect = np.clip(img[30:40].astype(np.float32) * (1 - cr.HINT_ALPHA) + np.array(cr.HINT_BGR, np.float32) * cr.HINT_ALPHA + 0.5,
                     0, 255).astype(np.uint8)
    assert np.array_equal(res[oy:oy + 10], expect)
    col = body[:, 100 + cr.GAP_PX * 3 + 2 * cr.ZOOM_PX * cr.ZOOM_X:100 + cr.GAP_PX * 3 + 2 * cr.ZOOM_PX * cr.ZOOM_X + 100]
    s = 30 - cr.ZOOM_PX // 2                                     # zoom atas BERPUSAT di garis potong (baris 6..53)
    assert (col[(30 - s) * cr.ZOOM_X] == cr.BOX_BGR).all(axis=1).sum() >= 90
    zoomed = cv2.resize(cr._shade(img, box, lines)[s:s + cr.ZOOM_PX], None, fx=1, fy=cr.ZOOM_X, interpolation=cv2.INTER_NEAREST)
    assert np.array_equal(col[:20], zoomed[:20])                # isi zoom = baris s.. (bukan 48 baris terluar)
    meta = json.loads((tmp_path / cr.PREVIEW_DIRNAME / cr.PREVIEW_JSON).read_text(encoding="utf-8"))
    assert [f["file"] for f in meta["files"]] == names and meta["jenis"] == "saran"


# ── Ingest + alat crop pada video sintetis ─────────
def _ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y", *args], check=True, capture_output=True)


def make_video(path: Path, band: int = 70, w: int = 320, h: int = 180, duration: float = 1.0, seed: int = 0) -> Path:
    """Konten bergerak (testsrc + noise temporal) + pita hitam atas / bawah `band` px (0 = tanpa pita)."""
    g = f"testsrc=size={w}x{h}:rate=30:duration={duration},noise=alls=30:allf=t+u:all_seed={seed}"
    if band:
        g += f",pad={w}:{h + 2 * band}:0:{band}:black"
    _ffmpeg("-f", "lavfi", "-i", g, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", str(path))
    return path


def frames_of(work: Path) -> list[Path]:
    return sorted((work / "frames").glob("frame_*.png"))


def read(p: Path) -> np.ndarray:
    return cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)


def run_ingest(video: Path, work: Path, crop_dir: Path | None, cfg: CropConfig | None = None, **kw):
    logs: list[str] = []
    meta = ing.ingest(video, work, crop_dir=crop_dir, crop_cfg=cfg, log=logs.append, **kw)
    return meta, "\n".join(logs)


@pytest.fixture(scope="module")
def band_video(tmp_path_factory) -> Path:
    return make_video(tmp_path_factory.mktemp("v") / "berpita.mp4")


@needs_ffmpeg
def test_saran_without_kotak_never_cuts_and_matches_r2_on_frames(band_video, tmp_path):
    meta, log = run_ingest(band_video, tmp_path / "w", tmp_path / "crop" / "berpita")
    ing.ingest(band_video, tmp_path / "w0")                                      # tanpa folder crop = perilaku lama
    assert (tmp_path / "w" / "meta.json").read_bytes() == (tmp_path / "w0" / "meta.json").read_bytes()
    a, b = frames_of(tmp_path / "w"), frames_of(tmp_path / "w0")
    assert len(a) == len(b) == meta["frame_count"] and all(x.read_bytes() == y.read_bytes() for x, y in zip(a, b))
    assert "crop" not in meta and (meta["working_width"], meta["working_height"]) == (320, 320)
    d = tmp_path / "crop" / "berpita"
    saran = json.loads((d / cr.SARAN_FILENAME).read_text(encoding="utf-8"))
    st = cr.Stats()
    for p in a:
        st.add(read(p))
    assert saran["box_work"] == cr.detect_crop(*st.result(), P)["box_work"]      # pipa = PNG frames/
    assert 66 <= saran["cut_px"]["top"] <= 74 and 66 <= saran["cut_px"]["bottom"] <= 74
    assert saran["key"]["rule_fingerprint"] == cr.FROZEN_FINGERPRINT and saran["n_frames"] == meta["frame_count"]
    assert sorted(p.name for p in (d / cr.PREVIEW_DIRNAME).iterdir())[:3] == [
        f"1_awal_f00000.png", f"2_tengah_f{meta['frame_count'] // 2:05d}.png", f"3_akhir_f{meta['frame_count'] - 1:05d}.png"]
    det = json.loads((d / cr.DETEKSI_FILENAME).read_text(encoding="utf-8"))
    assert det["pita_terdeteksi"] is True and det["key"] == saran["key"]
    assert "--accept" in log and "--none" in log and "--box X Y W H" in log and not (d / cr.KOTAK_FILENAME).exists()


@needs_ffmpeg
def test_preview_middle_frame_equals_frames_png(band_video, tmp_path):
    meta, _ = run_ingest(band_video, tmp_path / "w", tmp_path / "c")
    mid = meta["frame_count"] // 2
    board = read(tmp_path / "c" / cr.PREVIEW_DIRNAME / f"2_tengah_f{mid:05d}.png")
    head = board.shape[0] - 320                                 # header 3–4 baris (baris petunjuk hanya bila ada)
    assert head in (cr.HEADER_PAD_PX + cr.HEADER_LINE_PX * 3, cr.HEADER_PAD_PX + cr.HEADER_LINE_PX * 4)
    saran = json.loads((tmp_path / "c" / cr.SARAN_FILENAME).read_text(encoding="utf-8"))
    x0, y0, x1, y1 = saran["box_work"]
    res = board[head:, -(x1 - x0 + 1):]
    oy = (320 - (y1 - y0 + 1)) // 2
    frame = read(frames_of(tmp_path / "w")[mid])
    lines = (saran["hints"] or {}).get("lines", {})            # baris peralihan kompresi di tepi pita bisa ≥ 95% diam → diarsir
    want = cr._shade(frame, saran["box_work"], lines)[y0:y1 + 1, x0:x1 + 1]
    assert np.array_equal(res[oy:oy + y1 - y0 + 1], want)
    assert not np.array_equal(read(frames_of(tmp_path / "w")[mid + 1])[y0:y1 + 1, x0:x1 + 1], frame[y0:y1 + 1, x0:x1 + 1])


@needs_ffmpeg
def test_d1_skips_detection_when_key_matches_and_redetects_on_change(band_video, tmp_path, monkeypatch):
    d = tmp_path / "c"
    run_ingest(band_video, tmp_path / "w", d)
    before = {p.name: p.read_bytes() for p in d.rglob("*") if p.is_file()}

    def no_scan(*a, **k):
        raise AssertionError("scan dipanggil padahal kunci D1 sama")

    monkeypatch.setattr(cr, "scan", no_scan)
    _, log = run_ingest(band_video, tmp_path / "w", d)
    assert "--accept" in log and {p.name: p.read_bytes() for p in d.rglob("*") if p.is_file()} == before
    with pytest.raises(AssertionError, match="scan dipanggil"):                 # ambang petunjuk lain → saran dibuat ulang
        run_ingest(band_video, tmp_path / "w", d, CropConfig(preview=CropPreviewConfig(hint_static_frac=0.9)))
    with pytest.raises(AssertionError, match="scan dipanggil"):                 # parameter R2 lain → kunci lain
        run_ingest(band_video, tmp_path / "w", d, CropConfig(detect=CropDetectConfig(band_min_px=9)))


@needs_ffmpeg
def test_d1_key_holds_source_sha_so_same_size_other_video_is_redetected(band_video, tmp_path):
    d = tmp_path / "c"
    run_ingest(band_video, tmp_path / "w", d)
    other = make_video(tmp_path / "lain.mp4", seed=5)                           # ukuran sama, isi lain
    run_ingest(other, tmp_path / "w2", d)
    det = json.loads((d / cr.DETEKSI_FILENAME).read_text(encoding="utf-8"))
    probe = ing.probe_video(other)
    assert det["key"]["source"]["sha256"] == cr.source_identity(other, probe)["sha256"]


@needs_ffmpeg
def test_no_band_records_deteksi_only_and_skips_next_time(tmp_path, monkeypatch):
    v = make_video(tmp_path / "polos.mp4", band=0)
    d = tmp_path / "c"
    _, log = run_ingest(v, tmp_path / "w", d)
    assert log == "" and sorted(p.name for p in d.iterdir()) == [cr.DETEKSI_FILENAME]
    assert json.loads((d / cr.DETEKSI_FILENAME).read_text(encoding="utf-8"))["pita_terdeteksi"] is False
    monkeypatch.setattr(cr, "scan", lambda *a, **k: (_ for _ in ()).throw(AssertionError("scan dipanggil")))
    run_ingest(v, tmp_path / "w", d)                                           # D1: tanpa deteksi ulang


@needs_ffmpeg
def test_stale_saran_removed_when_new_source_has_no_band(band_video, tmp_path):
    d = tmp_path / "c"
    run_ingest(band_video, tmp_path / "w", d)
    assert (d / cr.SARAN_FILENAME).is_file()
    run_ingest(make_video(tmp_path / "polos.mp4", band=0), tmp_path / "w2", d)
    assert not (d / cr.SARAN_FILENAME).exists() and not (d / cr.PREVIEW_DIRNAME).exists()


@needs_ffmpeg
def test_detection_uses_full_source_even_when_trimmed(band_video, tmp_path):
    run_ingest(band_video, tmp_path / "w1", tmp_path / "c1")
    meta, _ = run_ingest(band_video, tmp_path / "w2", tmp_path / "c2", start_s=0.5, end_s=0.75)
    assert meta["frame_count"] == 6
    for name in (cr.SARAN_FILENAME, cr.DETEKSI_FILENAME):
        assert (tmp_path / "c1" / name).read_bytes() == (tmp_path / "c2" / name).read_bytes()


@needs_ffmpeg
def test_accept_then_ingest_crops_from_source_pixel_identical(band_video, tmp_path):
    d = tmp_path / "c"
    full, _ = run_ingest(band_video, tmp_path / "w0", d)
    logs: list[str] = []
    assert cr.run_crop(crop_dir=d, video=band_video, cfg=CropConfig(), action="accept", box=None, target_fps=24,
                       working_width=720, log=logs.append) is True
    kotak = json.loads((d / cr.KOTAK_FILENAME).read_text(encoding="utf-8"))
    saran = json.loads((d / cr.SARAN_FILENAME).read_text(encoding="utf-8"))
    assert kotak["mode"] == "saran" and kotak["box_source"] == saran["box_source"]
    meta, log = run_ingest(band_video, tmp_path / "w1", d)
    x, y, w, h = saran["box_source"]
    assert (meta["working_width"], meta["working_height"]) == (w, h) and log == ""      # tanpa saran lagi
    assert meta["crop"]["mode"] == "saran" and meta["crop"]["box_source"] == [x, y, w, h]
    assert meta["crop"]["source_sha256"] == kotak["source"]["sha256"] and meta["crop"]["rule_fingerprint"] == cr.FROZEN_FINGERPRINT
    assert list(meta)[:13] == list(full)[:13] and meta["frame_count"] == full["frame_count"]
    for p0, p1 in zip(frames_of(tmp_path / "w0"), frames_of(tmp_path / "w1")):       # sumber 320 px = kerja → crop = potong PNG
        assert np.array_equal(read(p0)[y:y + h, x:x + w], read(p1))


@needs_ffmpeg
def test_none_keeps_meta_byte_identical_and_skips_detection(band_video, tmp_path, monkeypatch):
    d = tmp_path / "c"
    run_ingest(band_video, tmp_path / "w0", d)
    cr.run_crop(crop_dir=d, video=band_video, cfg=CropConfig(), action="none", box=None, target_fps=24, working_width=720,
                log=lambda m: None)
    monkeypatch.setattr(cr, "scan", lambda *a, **k: (_ for _ in ()).throw(AssertionError("scan dipanggil")))
    _, log = run_ingest(band_video, tmp_path / "w1", d)
    assert log == "" and (tmp_path / "w0" / "meta.json").read_bytes() == (tmp_path / "w1" / "meta.json").read_bytes()
    assert cr.pending_saran(d) is None


@needs_ffmpeg
def test_manual_box_rounded_preview_and_ingest(band_video, tmp_path):
    d = tmp_path / "c"
    cr.run_crop(crop_dir=d, video=band_video, cfg=CropConfig(), action="box", box=[11, 71, 299, 179], target_fps=24,
                working_width=720, log=lambda m: None)
    kotak = json.loads((d / cr.KOTAK_FILENAME).read_text(encoding="utf-8"))
    assert kotak["mode"] == "manual" and kotak["box_source"] == [10, 70, 300, 180] and kotak["box_requested"] == [11, 71, 299, 179]
    prev = json.loads((d / cr.PREVIEW_DIRNAME / cr.PREVIEW_JSON).read_text(encoding="utf-8"))
    assert prev["jenis"] == "manual" and len(prev["files"]) == 3
    meta, _ = run_ingest(band_video, tmp_path / "w", d)
    assert (meta["working_width"], meta["working_height"]) == (300, 180) and meta["crop"]["box_requested"] == [11, 71, 299, 179]


@needs_ffmpeg
def test_changed_source_rejects_kotak_but_copy_is_accepted(band_video, tmp_path):
    d = tmp_path / "c"
    cr.run_crop(crop_dir=d, video=band_video, cfg=CropConfig(), action="none", box=None, target_fps=24, working_width=720,
                log=lambda m: None)
    other = make_video(tmp_path / "berpita.mp4", seed=7)                         # nama sama, isi lain
    with pytest.raises(ing.IngestError, match="sumber LAIN.*crop .*--clear"):
        run_ingest(other, tmp_path / "w", d)
    assert not (tmp_path / "w" / "meta.json").exists()
    copy = tmp_path / "salinan" / "berpita.mp4"
    copy.parent.mkdir()
    shutil.copyfile(band_video, copy)                                           # byte sama di path lain → diterima
    run_ingest(copy, tmp_path / "w", d)


@needs_ffmpeg
def test_suggest_false_writes_nothing(band_video, tmp_path):
    d = tmp_path / "c"
    _, log = run_ingest(band_video, tmp_path / "w", d, CropConfig(suggest=False))
    assert log == "" and not d.exists()


@needs_ffmpeg
def test_changed_rule_warns_and_accept_refuses(band_video, tmp_path):
    d = tmp_path / "c"
    _, log = run_ingest(band_video, tmp_path / "w", d, CropConfig(detect=CropDetectConfig(uniform_std_max=5.0)))
    assert "PERINGATAN" in log and "belum divalidasi" in log
    with pytest.raises(cr.CropError, match="selain R2 beku"):
        cr.run_crop(crop_dir=d, video=band_video, cfg=CropConfig(), action="accept", box=None, target_fps=24,
                    working_width=720, log=lambda m: None)


@needs_ffmpeg
def test_accept_without_saran_fails_and_clear_moves_to_prev(band_video, tmp_path):
    d = tmp_path / "c"
    with pytest.raises(cr.CropError, match="tidak ada saran"):
        cr.run_crop(crop_dir=d, video=band_video, cfg=CropConfig(), action="accept", box=None, target_fps=24,
                    working_width=720, log=lambda m: None)
    assert cr.run_crop(crop_dir=d, video=band_video, cfg=CropConfig(), action="clear", box=None, target_fps=24,
                       working_width=720, log=lambda m: None) is False
    cr.run_crop(crop_dir=d, video=band_video, cfg=CropConfig(), action="none", box=None, target_fps=24, working_width=720,
                log=lambda m: None)
    first = (d / cr.KOTAK_FILENAME).read_bytes()
    assert cr.run_crop(crop_dir=d, video=band_video, cfg=CropConfig(), action="clear", box=None, target_fps=24,
                       working_width=720, log=lambda m: None) is True
    assert not (d / cr.KOTAK_FILENAME).exists() and (d / cr.KOTAK_PREV_FILENAME).read_bytes() == first


@needs_ffmpeg
def test_ingest_main_prints_saran_after_meta_and_reads_config(band_video, tmp_path, capsys):
    rc = ing.main([str(band_video), "--work-dir", str(tmp_path / "w"), "--crop-dir", str(tmp_path / "c")])
    out = capsys.readouterr().out
    assert rc == 0 and out.index('"frame_index_start"') < out.index("saran crop (BELUM dipotong)")
    conf = tmp_path / "c.yaml"
    conf.write_text("ingest:\n  crop:\n    suggest: false\n", encoding="utf-8")
    rc = ing.main([str(band_video), "--work-dir", str(tmp_path / "w2"), "--crop-dir", str(tmp_path / "c2"), "--config", str(conf)])
    assert rc == 0 and "saran crop" not in capsys.readouterr().out and not (tmp_path / "c2").exists()
    conf2 = tmp_path / "c2.yaml"                                          # --config non-default → perintah siap tempel membawanya
    conf2.write_text("ingest:\n  crop:\n    suggest: true\n", encoding="utf-8")
    rc = ing.main([str(band_video), "--work-dir", str(tmp_path / "w3"), "--crop-dir", str(tmp_path / "c3"), "--config", str(conf2)])
    out = capsys.readouterr().out
    assert rc == 0 and f"crop {band_video.resolve()} --accept --config {conf2.resolve()}" in out
    assert f"crop {band_video.resolve()} --none --config {conf2.resolve()}" in out
