"""T-305f Tahap 3: verifikasi run produksi 6 klip + bahan penilaian klip5 penuh (docs/04 "Keputusan T-305f setelah Tahap 1").

    python scripts/t305f_tahap3.py cek     # 6 klip vs F1 before_t3 + salinan contours lama + salinan PB Tahap 1 -> work/t305f/cek_t3.json
    python scripts/t305f_tahap3.py bahan   # work/t305f/nilai/klip5_penuh_berdampingan.mp4 + _lambat (FOTO | SEBELUM | SESUDAH, f0–f132)

Pembanding (dibuat SEBELUM run Tahap 3): `work/t305f/before_t3_sha256.txt` (F1), `work/t305f_scratch/before/<klip>/contours/` (salinan
contours lama: isi strok), `work/t305f/klip5_sebelum.mp4` (salinan out/klip5.mp4 lama), `work/t305f_scratch/PB/klip5/` (render PB Tahap 1).
Hanya MEMBACA work/clips/ dan out/; menulis ke work/t305f/ (di-ignore git). CPU saja.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import t305_1b as proto  # noqa: E402
import t606_hash as h  # noqa: E402
from t305_newclips import encoder  # noqa: E402
from t305d_stage3 import read_frames, runs_str  # noqa: E402

from rotoscope import stabilize as stb  # noqa: E402

CLIPS_DIR = ROOT / "work" / "clips"
OUT = ROOT / "work" / "t305f"
SCR = ROOT / "work" / "t305f_scratch"
NILAI = OUT / "nilai"
CLIPS = ("test", "test_short", "Klip2", "klip3", "klip4", "klip5")
VF_CLIP = "klip5"
BEFORE = OUT / "before_t3_sha256.txt"
OLD_CONTOURS = SCR / "before"
PB = SCR / "PB" / VF_CLIP
OLD_MP4 = OUT / "klip5_sebelum.mp4"
METADATA_NAMES = ("manifest.json", "frames.jsonl", "clip_stats.json", ".rotoscope-clip.json")


def jload(p: Path):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def jsave(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"  -> {p.relative_to(ROOT).as_posix()}")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rel(p: Path) -> str:
    return p.relative_to(ROOT).as_posix()


def geom(s: dict) -> tuple:
    return (s["type"], s["closed"], tuple(s.get("groups") or ()), json.dumps(s["points"]))


def last_rows(path: Path) -> dict[str, dict]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r.get("event") == "frame":
            rows[r["frame"]] = r
    return rows


# ── cek ────────────────────────────────────────────
def check_clip(c: str, before: dict[str, str]) -> dict:
    work = CLIPS_DIR / c
    clip = stb.load_clip(work)
    diff = {k: [] for k in ("groups", "depth_smooth", "png", "svg", "out_svg", "contours_strokes", "contours_geom",
                            "contours_json_bytes")}
    json_keys = Counter()
    for pos, name in enumerate(clip.names):
        stem = Path(name).stem
        for key, p in (("groups", clip.groups_path(name)), ("depth_smooth", clip.depth_smooth_path(name)),
                       ("png", work / "strokes" / f"{stem}.png"), ("svg", work / "strokes" / f"{stem}.svg"),
                       ("out_svg", ROOT / "out" / "svg" / c / f"{stem}.svg"),
                       ("contours_json_bytes", work / "contours" / f"{stem}.json")):
            if before.get(rel(p)) != sha(p):
                diff[key].append(pos)
        a, b = jload(OLD_CONTOURS / c / "contours" / f"{stem}.json"), jload(work / "contours" / f"{stem}.json")
        if a["strokes"] != b["strokes"]:
            diff["contours_strokes"].append(pos)
        if Counter(map(geom, a["strokes"])) != Counter(map(geom, b["strokes"])):
            diff["contours_geom"].append(pos)
        json_keys.update(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    mp4 = ROOT / "out" / f"{c}.mp4"
    cs_a, cs_b = jload(OLD_CONTOURS / c / "contours" / "clip_stats.json"), jload(work / "contours" / "clip_stats.json")
    rows = last_rows(work / "stable" / "frames.jsonl")
    vf_frames = [pos for pos, n in enumerate(clip.names) if rows[n].get("hole_fill_regions")]
    m_old, m_new = jload(OLD_CONTOURS / c / "stable_manifest.json"), jload(work / "stable" / "manifest.json")
    res = {"frames": len(clip.names), **{k: runs_str(v) for k, v in diff.items()}, "n": {k: len(v) for k, v in diff.items()},
           "contours_json_keys_differ": dict(json_keys), "mp4_identical": before.get(rel(mp4)) == sha(mp4),
           "clip_stats": {k: [cs_a[k], cs_b[k]] for k in ("t_high", "t_low")},
           "stable_manifest": {"stabilize_hash": [m_old["stabilize_hash"][:12], m_new["stabilize_hash"][:12]],
                               "hole_fill": m_new["stabilize"].get("hole_fill"),
                               "only_hash_section_created_differ": sorted(k for k in set(m_old) | set(m_new) if m_old.get(k) != m_new.get(k))},
           "vf_frames_jsonl": runs_str(vf_frames),
           "vf_regions_px": [sum(rows[n].get("hole_fill_regions", 0) for n in clip.names),
                             sum(rows[n].get("hole_fill_px", 0) for n in clip.names)],
           "frames_log_has_hole_fill_keys": all("hole_fill_regions" in rows[n] for n in clip.names)}
    if c == VF_CLIP:
        res["vs_PB_tahap1"] = vs_pb(clip)
    return res


def vs_pb(clip) -> dict:
    """klip5 produksi vs salinan PB Tahap 1: stable/ byte, strok contours, PNG + SVG strokes byte."""
    bad = {k: [] for k in ("groups", "depth_smooth", "contours_strokes", "png", "svg")}
    work = CLIPS_DIR / VF_CLIP
    for pos, name in enumerate(clip.names):
        stem = Path(name).stem
        for key, sub, suf in (("groups", "stable/groups", ".png"), ("depth_smooth", "stable/depth_smooth", ".npy"),
                              ("png", "strokes", ".png"), ("svg", "strokes", ".svg")):
            if sha(work / sub / f"{stem}{suf}") != sha(PB / sub / f"{stem}{suf}"):
                bad[key].append(pos)
        if jload(work / "contours" / f"{stem}.json")["strokes"] != jload(PB / "contours" / f"{stem}.json")["strokes"]:
            bad["contours_strokes"].append(pos)
    cs_a, cs_b = jload(PB / "contours" / "clip_stats.json"), jload(work / "contours" / "clip_stats.json")
    return {**{k: runs_str(v) for k, v in bad.items()},
            "clip_stats_same": [cs_a["t_high"], cs_a["t_low"]] == [cs_b["t_high"], cs_b["t_low"]]}


def classify_f1(before: dict[str, str], after_path: Path) -> dict:
    """Setiap berkas yang berubah / hilang / bertambah di F1 → kategori. 'lainnya' harus kosong."""
    after = h.read_list(after_path)
    changed = sorted(r for r in before if r in after and after[r] != before[r])
    cats: dict[str, list[str]] = {}
    for r in changed + [f"(hilang) {r}" for r in sorted(set(before) - set(after))] + [f"(tambahan) {r}" for r in sorted(set(after) - set(before))]:
        p = r.split(" ", 1)[-1]
        name = p.rsplit("/", 1)[-1]
        if r.startswith("("):
            cat = "lainnya"
        elif name == h.QC_NAME:
            cat = "qc_report.json (cek created_utc)"
        elif name in METADATA_NAMES or name.endswith(".export.json"):
            cat = "metadata (manifest / frames.jsonl / clip_stats / penanda / export.json)"
        elif p.startswith(f"work/clips/{VF_CLIP}/") or p.startswith(f"out/svg/{VF_CLIP}/") or p == f"out/{VF_CLIP}.mp4":
            cat = f"isi {VF_CLIP}"
        elif "/contours/frame_" in p:
            cat = "contours frame JSON klip lain (metadata; isi strok dicek terpisah)"
        else:
            cat = "lainnya"
        cats.setdefault(cat, []).append(r)
    by_folder = Counter(h.folder_of(r.split(" ", 1)[-1]) for v in cats.values() for r in v)
    return {"n_before": len(before), "n_after": len(after), "kategori": {k: len(v) for k, v in cats.items()},
            "lainnya": cats.get("lainnya", []), "per_folder": dict(sorted(by_folder.items())),
            "qc_report": cats.get("qc_report.json (cek created_utc)", [])}


def cmd_cek() -> None:
    before = h.read_list(BEFORE)
    res = {c: check_clip(c, before) for c in CLIPS}
    for c, r in res.items():
        print(c, json.dumps({k: v for k, v in r.items() if k not in ("contours_json_keys_differ",)}, ensure_ascii=False), flush=True)
    after = OUT / "after_t3_sha256.txt"
    if after.is_file():
        res["F1"] = classify_f1(before, after)
        qc_b = jload(OUT / "f1_before_t3.json")["qc_report_normalized_sha256"]
        qc_a = jload(OUT / "f1_after_t3.json")["qc_report_normalized_sha256"]
        res["F1"]["qc_created_utc_only"] = {r: qc_b.get(r) == qc_a.get(r) for r in qc_b}
        print("F1", json.dumps(res["F1"], ensure_ascii=False, indent=1))
    jsave(OUT / "cek_t3.json", res)


# ── bahan ──────────────────────────────────────────
def write_pair(frames, stem: str) -> dict:
    """Normal 24 fps + lambat 0,5× (tiap frame dua kali), dialirkan ke ffmpeg (pola `t305_1b.write_pair`, folder NILAI sendiri)."""
    enc, n = None, 0
    for img in frames:
        if enc is None:
            size = (img.shape[1], img.shape[0])
            enc = (encoder(NILAI / f"{stem}.mp4", size), encoder(NILAI / f"{stem}_lambat.mp4", size))
        raw = np.ascontiguousarray(img).tobytes()
        enc[0].stdin.write(raw)
        enc[1].stdin.write(raw)
        enc[1].stdin.write(raw)
        n += 1
    for p in enc:
        p.stdin.close()
        if p.wait() != 0:
            raise SystemExit(f"{stem}: ffmpeg gagal")
    return {"frames": n, "size": list(size), "lambat_frames": 2 * n}


def cmd_bahan() -> None:
    NILAI.mkdir(parents=True, exist_ok=True)
    clip = stb.load_clip(CLIPS_DIR / VF_CLIP)
    n = len(clip.names)
    rows = last_rows(CLIPS_DIR / VF_CLIP / "stable" / "frames.jsonl")
    before, after = list(read_frames(OLD_MP4)), list(read_frames(ROOT / "out" / f"{VF_CLIP}.mp4"))
    if len(before) != n or len(after) != n:
        raise SystemExit(f"jumlah frame MP4: sebelum {len(before)}, sesudah {len(after)}, klip {n}")

    def gen():
        for f, name in enumerate(clip.names):
            foto = cv2.imdecode(np.fromfile(clip.work_dir / "frames" / name, np.uint8), cv2.IMREAD_COLOR)
            k = rows[name].get("hole_fill_regions", 0)
            yield proto.compose([foto, before[f], after[f]], ["FOTO", "SEBELUM", "SESUDAH +VF" if k else "SESUDAH"], f, n)

    info = write_pair(gen(), "klip5_penuh_berdampingan")
    info["vf_frames"] = runs_str([f for f, nm in enumerate(clip.names) if rows[nm].get("hole_fill_regions")])
    info["sebelum_sha256"], info["sesudah_sha256"] = sha(OLD_MP4), sha(ROOT / "out" / f"{VF_CLIP}.mp4")
    print(json.dumps(info, ensure_ascii=False))
    jsave(OUT / "bahan.json", info)


def main(argv: list[str]) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if argv == ["cek"]:
        cmd_cek()
    elif argv == ["bahan"]:
        cmd_bahan()
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
