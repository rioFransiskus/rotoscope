"""T-606 Tahap 3 (alat sekali pakai, F3): run produksi klip3 dengan masker [3] — simpan keadaan sebelum, verifikasi sesudah, video
berdampingan seluruh klip. CPU saja.

    python scripts/t606_tahap3.py prasyarat  baca-saja: tiap klip punya seg/ + depth/ + qc_report lengkap (run tidak akan memakai GPU)
    python scripts/t606_tahap3.py simpan     SEBELUM `run` klip3: salin strokes/*.png + contours/*.json klip3 (sebelum T-606) ke
                                             work/t606_scratch/sebelum/ + sha256 stable/ contours/ strokes/ -> work/t606/sebelum_t3.json
    python scripts/t606_tahap3.py cek        SESUDAH `run` klip3: (a) f85–f125 stable/ + contours/ + strokes/ vs prototipe M2-H d 4
                                             (work/t606_scratch/M2H/klip3), (b) seluruh klip stable/ + contours/ vs prototipe, (c) frame
                                             di luar adegan: stable/ vs sebelum, (d) perubahan garis oklusi vs sebelum -> work/t606/cek_t3.json
    python scripts/t606_tahap3.py bahan      video foto | sebelum T-606 | sesudah, seluruh klip, normal + lambat 0,5x -> work/t606/nilai/

Hanya MEMBACA work/clips/*/; menulis ke work/t606_scratch/sebelum/ dan work/t606/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import t606_tahap1 as t1  # noqa: E402

CLIPS = ROOT / "work" / "clips"
SRC = t1.SRC
PROTO = t1.vdir("M2H")
BEFORE = t1.SCR / "sebelum"
OTHER = ("test", "test_short", "Klip2", "klip4", "klip5")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def tree_sha(d: Path, sub: str, pattern: str) -> dict[str, str]:
    return {p.name: sha(p) for p in sorted((d / sub).glob(pattern))}


def cmd_prasyarat() -> None:
    out = {}
    for c in ("klip3", *OTHER):
        d = CLIPS / c
        n = int(json.loads((d / "meta.json").read_text(encoding="utf-8"))["frame_count"])
        cnt = {"probs": len(list((d / "seg" / "probs").glob("frame_*.npz"))),
               "classmap": len(list((d / "seg" / "classmap").glob("frame_*.png"))),
               "depth": len(list((d / "depth").glob("frame_*.npy"))),
               "stable": len(list((d / "stable" / "groups").glob("frame_*.png"))),
               "contours": len(list((d / "contours").glob("frame_*.json"))),
               "strokes": len(list((d / "strokes").glob("frame_*.png")))}
        ok = all(v == n for v in cnt.values()) and (d / "qc_report.json").is_file()
        out[c] = {"frame_count": n, **cnt, "qc_report": (d / "qc_report.json").is_file(),
                  "target": (d / "target").exists(), "isolate": (d / "isolate").exists(), "lengkap": ok}
        print(f"{c}: {out[c]}")
    t1.jsave(t1.OUT / "prasyarat_t3.json", out)


def cmd_simpan() -> None:
    if (SRC / "stable" / "manifest.json").is_file() and "isolate" in json.loads(
            (SRC / "stable" / "manifest.json").read_text(encoding="utf-8")):
        raise SystemExit("klip3 stable/ sudah memakai masker (T-606) — `simpan` harus dijalankan SEBELUM run")
    if BEFORE.exists():
        shutil.rmtree(BEFORE)
    for sub, pat in (("strokes", "frame_*.png"), ("contours", "*.json")):
        (BEFORE / sub).mkdir(parents=True)
        for p in sorted((SRC / sub).glob(pat)):
            shutil.copyfile(p, BEFORE / sub / p.name)
    res = {"stable_groups": tree_sha(SRC, "stable/groups", "*.png"),
           "stable_depth_smooth": tree_sha(SRC, "stable/depth_smooth", "*.npy"),
           "contours": tree_sha(SRC, "contours", "frame_*.json"), "strokes_png": tree_sha(SRC, "strokes", "frame_*.png"),
           "strokes_svg": tree_sha(SRC, "strokes", "frame_*.svg"),
           "clip_stats": json.loads((SRC / "contours" / "clip_stats.json").read_text(encoding="utf-8")),
           "out": {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted((ROOT / "out").glob("klip3*"))}}
    t1.jsave(t1.OUT / "sebelum_t3.json", res)
    print(f"disalin: {len(res['strokes_png'])} strokes PNG + {len(res['contours'])} contours -> {BEFORE}; sha -> work/t606/sebelum_t3.json")


def _len(s: dict) -> float:
    return t1._len(s)


def cmd_cek() -> None:
    before = json.loads((t1.OUT / "sebelum_t3.json").read_text(encoding="utf-8"))
    sc = t1.stb.load_clip(SRC)
    lo, hi = t1.SCENE
    m = json.loads((SRC / "stable" / "manifest.json").read_text(encoding="utf-8"))
    res: dict = {"manifest_isolate": m.get("isolate")}
    # (a) + (b) vs prototipe
    diff = Counter()
    scene_diff = Counter()
    for i, name in zip(sc.indices, sc.names):
        st = Path(name).stem
        for sub, suf in (("stable/groups", ".png"), ("stable/depth_smooth", ".npy"), ("contours", ".json")):
            same = sha(SRC / sub / (st + suf)) == sha(PROTO / sub / (st + suf))
            if not same:
                diff[sub] += 1
                if lo <= i <= hi:
                    scene_diff[sub] += 1
        if lo <= i <= hi:
            for suf in (".png", ".svg"):
                if sha(SRC / "strokes" / (st + suf)) != sha(PROTO / "strokes" / (st + suf)):
                    scene_diff["strokes" + suf] += 1
    cs_new = json.loads((SRC / "contours" / "clip_stats.json").read_text(encoding="utf-8"))
    cs_proto = json.loads((PROTO / "contours" / "clip_stats.json").read_text(encoding="utf-8"))
    res["vs_prototipe_d4"] = {"adegan_f85_f125_beda": dict(scene_diff), "klip_penuh_beda": dict(diff),
                              "ambang_sama": [cs_new["t_high"], cs_new["t_low"]] == [cs_proto["t_high"], cs_proto["t_low"]],
                              "t_high_t_low": [cs_new["t_high"], cs_new["t_low"]]}
    # (c) di luar adegan: stable/ vs sebelum
    out_diff = [i for i, name in zip(sc.indices, sc.names) if not lo <= i <= hi and (
        sha(SRC / "stable" / "groups" / (Path(name).stem + ".png")) != before["stable_groups"][Path(name).stem + ".png"]
        or sha(SRC / "stable" / "depth_smooth" / (Path(name).stem + ".npy"))
        != before["stable_depth_smooth"][Path(name).stem + ".npy"])]
    in_diff = [i for i, name in zip(sc.indices, sc.names) if lo <= i <= hi and
               sha(SRC / "stable" / "groups" / (Path(name).stem + ".png")) != before["stable_groups"][Path(name).stem + ".png"]]
    res["stable_vs_sebelum"] = {"luar_adegan_beda": out_diff, "adegan_groups_beda_n": len(in_diff)}
    # (d) garis oklusi vs sebelum (frame di luar adegan)
    occ = {"frames_beda": [], "n_sebelum": 0, "n_sesudah": 0, "len_sebelum": 0.0, "len_sesudah": 0.0,
           "selain_oklusi_beda": [], "hanya_track_id_beda": []}
    for i, name in zip(sc.indices, sc.names):
        if lo <= i <= hi:
            continue
        a = json.loads((BEFORE / "contours" / (Path(name).stem + ".json")).read_text(encoding="utf-8"))["strokes"]
        b = json.loads((SRC / "contours" / (Path(name).stem + ".json")).read_text(encoding="utf-8"))["strokes"]
        oa, ob = [s for s in a if s["type"] == "occlusion"], [s for s in b if s["type"] == "occlusion"]
        na, nb = [s for s in a if s["type"] != "occlusion"], [s for s in b if s["type"] != "occlusion"]
        if t1.strip_ids(oa) != t1.strip_ids(ob):
            occ["frames_beda"].append(i)
            occ["n_sebelum"] += len(oa)
            occ["n_sesudah"] += len(ob)
            occ["len_sebelum"] += sum(_len(s) for s in oa)
            occ["len_sesudah"] += sum(_len(s) for s in ob)
        if t1.strip_ids(na) != t1.strip_ids(nb):
            occ["selain_oklusi_beda"].append(i)
        elif a != b and t1.strip_ids(a) == t1.strip_ids(b):
            occ["hanya_track_id_beda"].append(i)
    occ["len_sebelum"], occ["len_sesudah"] = round(occ["len_sebelum"], 1), round(occ["len_sesudah"], 1)
    occ["len_pct"] = round(100 * (occ["len_sesudah"] / occ["len_sebelum"] - 1), 2) if occ["len_sebelum"] else None
    res["oklusi_luar_adegan_vs_sebelum"] = {**occ, "n_frames_luar": len(sc.names) - (hi - lo + 1),
                                            "ambang_sebelum": [before["clip_stats"]["t_high"], before["clip_stats"]["t_low"]],
                                            "ambang_sesudah": [cs_new["t_high"], cs_new["t_low"]]}
    t1.jsave(t1.OUT / "cek_t3.json", res)
    print(json.dumps({k: v for k, v in res.items()}, indent=1, ensure_ascii=False)[:4000])


def cmd_bahan() -> None:
    ctx = t1.Ctx()
    sc = ctx.sclip
    fps = float(json.loads((SRC / "meta.json").read_text(encoding="utf-8"))["target_fps"])
    lo, hi = t1.SCENE
    missing = [n for n in sc.names if not (BEFORE / "strokes" / n).is_file() or not (SRC / "strokes" / n).is_file()]
    if missing:
        raise SystemExit(f"strokes sebelum / sesudah belum lengkap (mis. {missing[0]})")
    t1.NILAI.mkdir(parents=True, exist_ok=True)
    made = []
    for suffix, f in (("", fps), ("_lambat", fps / 2)):
        def gen():
            for k, (i, name) in enumerate(zip(sc.indices, sc.names), 1):
                scene = "adegan berklik (masker)" if lo <= i <= hi else "tanpa klik (seperti biasa)"
                sub = f"f{i:03d}  t={i / fps:.2f} s  {scene}"
                photo = t1.photo_panel(ctx, i) if lo <= i <= hi else t1.fit(
                    cv2.imdecode(np.fromfile(SRC / "frames" / name, np.uint8), cv2.IMREAD_COLOR))
                ps = [t1.labeled_big(photo, "foto" + (" + tepi masker" if lo <= i <= hi else ""), sub),
                      t1.labeled_big(t1.fit(cv2.imdecode(np.fromfile(BEFORE / "strokes" / name, np.uint8), cv2.IMREAD_COLOR)),
                                     "sebelum T-606", sub),
                      t1.labeled_big(t1.fit(cv2.imdecode(np.fromfile(SRC / "strokes" / name, np.uint8), cv2.IMREAD_COLOR)),
                                     "sesudah T-606", sub)]
                yield t1.even(np.hstack(ps))
        n = t1.write_video(t1.NILAI / f"klip3_produksi_berdampingan{suffix}.mp4", gen(), f)
        made.append({"video": f"klip3_produksi_berdampingan{suffix}.mp4", "n": n, "fps": f})
        print(f"  work/t606/nilai/klip3_produksi_berdampingan{suffix}.mp4: {n} frame, {f:g} fps")
    idx = t1.OUT / "bahan.json"
    old = json.loads(idx.read_text(encoding="utf-8")) if idx.is_file() else {}
    old["produksi"] = made
    t1.jsave(idx, old)


def main(argv: list[str]) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(prog="t606_tahap3.py")
    ap.add_argument("cmd", choices=("prasyarat", "simpan", "cek", "bahan"))
    a = ap.parse_args(argv)
    {"prasyarat": cmd_prasyarat, "simpan": cmd_simpan, "cek": cmd_cek, "bahan": cmd_bahan}[a.cmd]()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
