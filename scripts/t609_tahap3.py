"""T-609 Tahap 3 (alat, F3; CPU): uji nyata pratinjau masker + peringatan klik kanan di SCRATCH, dengan perintah CLI asli.

    python scripts/t609_tahap3.py siapkan   # scratch tambahan (salinan, BUKAN hard link; config = configs/default.yaml, hanya
                                            # `paths` diganti, dicek sesudah load) -> work/t609/tahap3_siapkan.json
                                            #   salah/  klip5@0-2.25 frames + meta + titik T-608 run 1 (iso1: 67 / 47 / 47 kanan;
                                            #           arsip work/t608/arsip_iso1/) = contoh masker SALAH
                                            #   multi/  klip3 frames + meta + titik Rio f85 + target 2 SINTETIS (BUKAN klik Rio;
                                            #           dari seg [2] + masker [1b] produksi f85: komponen latar-depan terbesar di
                                            #           luar target 1) = ukur multi-target (keputusan Rio T-609 no. 7)
                                            #   pick/   klip3 frames + meta + titik Rio f85 = uji jendela pick Rio (U1)
    python scripts/t609_tahap3.py jalan     # perintah CLI asli (subprocess, CUDA_VISIBLE_DEVICES=-1) -> work/t609/tahap3_jalan.json
    python scripts/t609_tahap3.py bahan     # PNG pratinjau scratch -> work/t609/nilai/ (salinan untuk Rio) + ringkasan

Produksi (work/clips/*, work/crop/*, out/) hanya DIBACA. Tanpa GPU (isolate tanpa --yes berhenti sebelum GPU), tanpa unduhan.
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"                 # pengaman: tanpa GPU

import hashlib  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from rotoscope import pick  # noqa: E402
from rotoscope.config import load_pipeline  # noqa: E402
from rotoscope.stage_common import write_json_atomic  # noqa: E402

PROD = ROOT / "work" / "clips"
SCR = ROOT / "work" / "t609_scratch"
OUT = ROOT / "work" / "t609"
DEFAULT_CFG = ROOT / "configs" / "default.yaml"
ISO1_POINTS = ROOT / "work" / "t608" / "arsip_iso1" / "target" / "points.json"
K5, K3 = "klip5@0-2.25", "klip3"
SETS = {"salah": K5, "multi": K3, "pick": K3}
SYNTH_TOOL = "SINTETIS scripts/t609_tahap3.py (target 2 BUKAN klik Rio; target 1 = klik Rio)"
MULTI_FRAME = 85
OPEN_PX = 5                                                # buka morfologi: buang pita tipis sebelum memilih komponen
BOX_PAD = 8
N_POS, POS_MIN_SEP = 5, 40
N_NEG, NEG_MIN_DT = 2, 6
VIDEO5, VIDEO3 = r"samples\klip5.mp4", r"samples\klip3.mp4"


def cfg_rel(name: str | None) -> str:
    return "work\\t609_scratch\\config.yaml" if name is None else f"work\\t609_scratch\\{name}\\config.yaml"


def file_sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def jsave(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"  -> {p.relative_to(ROOT).as_posix()}")


def copy_item(src: Path, dst: Path) -> list[tuple[Path, Path]]:
    pairs = [(src, dst)] if src.is_file() else [(p, dst / p.relative_to(src)) for p in sorted(src.rglob("*")) if p.is_file()]
    for s, d in pairs:
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(s, d)
    return pairs


def write_config(base: Path) -> dict:
    text = DEFAULT_CFG.read_text(encoding="utf-8")
    rel = base.relative_to(ROOT).as_posix()
    for old, new in {'  work_dir: "work"': f'  work_dir: "{rel}/work"', '  out_dir: "out"': f'  out_dir: "{rel}/out"'}.items():
        if text.count(old) != 1:
            raise SystemExit(f"baris {old!r} tidak tepat satu kali di configs/default.yaml")
        text = text.replace(old, new)
    path = base / "config.yaml"
    path.write_text(text, encoding="utf-8")
    a, b = load_pipeline(DEFAULT_CFG), load_pipeline(path)
    diff = sorted(k for k in ("work_dir", "out_dir") if getattr(a.paths, k) != getattr(b.paths, k))
    same_rest = all(getattr(a, f) == getattr(b, f) for f in a.__dataclass_fields__ if f != "paths")
    wd = b.paths.work_dir.resolve()
    if not same_rest or not wd.is_relative_to(base.resolve()):
        raise SystemExit(f"{path}: beda di luar paths / work_dir bukan scratch — berhenti")
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": file_sha(path), "beda": [f"paths.{k}" for k in diff],
            "work_dir": wd.as_posix()}


def spread_maxima(dt: np.ndarray, n: int, min_sep: float) -> list[tuple[int, int]]:
    """n titik serakah pada nilai jarak-tepi terbesar, saling berjarak ≥ min_sep (x, y)."""
    ys, xs = np.nonzero(dt > 0)
    order = np.argsort(-dt[ys, xs], kind="stable")
    out: list[tuple[int, int]] = []
    for k in order:
        x, y = int(xs[k]), int(ys[k])
        if all((x - a) ** 2 + (y - b) ** 2 >= min_sep ** 2 for a, b in out):
            out.append((x, y))
            if len(out) == n:
                break
    return out


def synth_target2(classmap: np.ndarray, label: np.ndarray) -> dict:
    """Target 2 SINTETIS: komponen terbesar (8-arah) latar-depan seg di luar masker [1b] target 1, sesudah buka morfologi."""
    other = ((classmap != 0) & (label == 0)).astype(np.uint8)
    other = cv2.morphologyEx(other, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (OPEN_PX, OPEN_PX)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(other, connectivity=8)
    if n < 2:
        raise SystemExit("tidak ada latar-depan di luar target 1 — target 2 sintetis tidak bisa dibuat")
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    comp = (lab == k).astype(np.uint8)
    h, w = comp.shape
    x, y, bw, bh = (int(v) for v in st[k, :4])
    box = [float(max(0, x - BOX_PAD)), float(max(0, y - BOX_PAD)), float(min(w - 1, x + bw - 1 + BOX_PAD)),
           float(min(h - 1, y + bh - 1 + BOX_PAD))]
    pos = spread_maxima(cv2.distanceTransform(comp, cv2.DIST_L2, 5), N_POS, POS_MIN_SEP)
    t1 = (label > 0).astype(np.uint8)
    dt1 = cv2.distanceTransform(t1, cv2.DIST_L2, 5)
    ys, xs = np.nonzero(dt1 >= NEG_MIN_DT)                 # klik kanan di target 1, paling dekat ke target 2
    d2 = cv2.distanceTransform(1 - comp, cv2.DIST_L2, 5)
    order = np.argsort(d2[ys, xs], kind="stable")
    neg: list[tuple[int, int]] = []
    for i in order:
        p = (int(xs[i]), int(ys[i]))
        if all((p[0] - a) ** 2 + (p[1] - b) ** 2 >= POS_MIN_SEP ** 2 for a, b in neg):
            neg.append(p)
            if len(neg) == N_NEG:
                break
    return {"id": 2, "points": [{"x": float(a), "y": float(b), "label": 1} for a, b in pos]
            + [{"x": float(a), "y": float(b), "label": 0} for a, b in neg], "box": box,
            "_info": {"komponen_px": int(st[k, cv2.CC_STAT_AREA]), "komponen_lain": int(n - 2),
                      "latar_depan_di_luar_target1_px": int(other.sum()), "target1_px": int(t1.sum()),
                      "porsi_frame": round(float(st[k, cv2.CC_STAT_AREA]) / (h * w), 4)}}


def cmd_siapkan() -> None:
    res: dict = {}
    for name, klip in SETS.items():
        base = SCR / name
        clip = base / "work" / "clips" / klip
        if base.exists():
            raise SystemExit(f"{base} sudah ada — salinan tidak ditimpa")
        pairs = copy_item(PROD / klip / "meta.json", clip / "meta.json") + copy_item(PROD / klip / "frames", clip / "frames")
        if name == "salah":
            pairs += copy_item(ISO1_POINTS, clip / "target" / "points.json")
        elif name == "pick":
            pairs += copy_item(PROD / klip / "target" / "points.json", clip / "target" / "points.json")
        bad = [d.relative_to(ROOT).as_posix() for s, d in pairs if file_sha(s) != file_sha(d)]
        linked = [d.relative_to(ROOT).as_posix() for s, d in pairs if os.stat(d).st_nlink != 1 or os.path.samefile(s, d)]
        if bad or linked:
            raise SystemExit(f"{name}: salinan tidak valid (beda {bad[:2]}, link {linked[:2]})")
        info = {"klip": klip, "berkas": len(pairs), "bytes": sum(d.stat().st_size for _, d in pairs), "config": write_config(base)}
        if name == "multi":
            cm = cv2.imread(str(PROD / klip / "seg" / "classmap" / f"frame_{MULTI_FRAME:05d}.png"), cv2.IMREAD_UNCHANGED)
            lb = cv2.imread(str(PROD / klip / "isolate" / "labels" / f"frame_{MULTI_FRAME:05d}.png"), cv2.IMREAD_UNCHANGED)
            meta = json.loads((clip / "meta.json").read_text(encoding="utf-8"))
            size = (meta["working_width"], meta["working_height"])
            if cm.shape[::-1] != size:
                cm = cv2.resize(cm, size, interpolation=cv2.INTER_NEAREST)
            t2 = synth_target2(cm, lb)
            doc = json.loads((PROD / klip / "target" / "points.json").read_text(encoding="utf-8"))
            entry = next(e for e in doc["frames"] if e["frame_index"] == MULTI_FRAME)
            entry["targets"].append({k: v for k, v in t2.items() if k != "_info"})
            doc["tool"] = SYNTH_TOOL
            pick.validate_doc(doc)
            (clip / "target").mkdir(parents=True, exist_ok=True)
            write_json_atomic(clip / "target" / "points.json", doc)
            info["target2_sintetis"] = {**t2["_info"], "kiri": N_POS, "kanan": len(t2["points"]) - N_POS, "box": t2["box"],
                                        "points": t2["points"], "sumber": ["seg/classmap f85 (produksi, dibaca)",
                                                                          "isolate/labels f85 (produksi, dibaca)"]}
            print(f"multi: target 2 SINTETIS {t2['_info']} kotak {t2['box']}")
        res[name] = info
        print(f"{name}: {len(pairs)} berkas disalin; config {info['config']['path']} (beda {info['config']['beda']})")
    jsave(OUT / "tahap3_siapkan.json", res)


def cli(args: list[str]) -> dict:
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "PYTHONIOENCODING": "utf-8"}
    t0 = time.perf_counter()
    p = subprocess.run([str(ROOT / "venv" / "Scripts" / "python.exe"), "-m", "rotoscope", *args], cwd=ROOT, capture_output=True,
                       text=True, encoding="utf-8", errors="replace", env=env)
    return {"cmd": "python -m rotoscope " + " ".join(args), "rc": p.returncode, "s": round(time.perf_counter() - t0, 1),
            "stdout": p.stdout, "stderr": p.stderr}


RUNS = [   # (nama, argv, rc yang diharapkan)
    ("pratinjau_klip5", ["pick", VIDEO5, "--end", "2.25", "--mask-preview", "--config", cfg_rel(None)], 0),
    ("pratinjau_klip3", ["pick", VIDEO3, "--mask-preview", "--config", cfg_rel(None)], 0),
    ("pratinjau_salah", ["pick", VIDEO5, "--end", "2.25", "--mask-preview", "--config", cfg_rel("salah")], 0),
    ("pratinjau_multi", ["pick", VIDEO3, "--mask-preview", "--config", cfg_rel("multi")], 0),
    ("pratinjau_ulang_klip5", ["pick", VIDEO5, "--end", "2.25", "--mask-preview", "--config", cfg_rel(None)], 0),
    ("pratinjau_dengan_frame", ["pick", VIDEO5, "--end", "2.25", "--mask-preview", "--frame", "0", "--config", cfg_rel(None)], 2),
    ("pick_dryrun_salah_f0", ["pick", VIDEO5, "--end", "2.25", "--frame", "0", "--dry-run", "--config", cfg_rel("salah")], 0),
    ("isolate_dryrun_salah", ["isolate", VIDEO5, "--end", "2.25", "--dry-run", "--config", cfg_rel("salah")], 0),
    ("isolate_tanpa_yes_salah", ["isolate", VIDEO5, "--end", "2.25", "--config", cfg_rel("salah")], 1),
    ("isolate_tanpa_yes_klip3", ["isolate", VIDEO3, "--config", cfg_rel(None)], 1),
    ("isolate_dryrun_klip5", ["isolate", VIDEO5, "--end", "2.25", "--dry-run", "--config", cfg_rel(None)], 0),
]


def preview_json(base: Path, klip: str) -> dict:
    return json.loads((base / "work" / "clips" / klip / "target" / "pratinjau" / "pratinjau.json").read_text(encoding="utf-8"))


def cmd_jalan() -> None:
    res: dict = {"runs": {}}
    sha_before = {k: file_sha(PROD / k / "target" / "points.json") for k in (K5, K3)}
    for name, argv, want in RUNS:
        r = cli(argv)
        r["rc_harapan"] = want
        res["runs"][name] = r
        print(f"{name}: rc {r['rc']} (harapan {want}) {r['s']} s")
        if r["rc"] != want:
            print(r["stdout"][-1500:], r["stderr"][-1500:])
    t1 = {k: json.loads((OUT / f"ukur_{k}.json").read_text(encoding="utf-8")) for k in (K5, K3)}
    cmp = {}
    for klip, base in ((K5, SCR), (K3, SCR)):
        pj = preview_json(base, klip)
        ref = {r["frame"]: r for r in t1[klip]["putaran"][0]["frame"]}
        for fr in pj["frames"]:
            t = fr["targets"][0]
            r0 = ref[fr["frame"]]
            cmp[f"{klip} f{fr['frame']}"] = {"skor": t["skor_objek"], "luas": t["luas_px"], "komponen": t["komponen_8arah"],
                                            "terbesar": t["porsi_komponen_terbesar"], "waktu_s": t["waktu_s"],
                                            "sama_tahap1": (t["skor_objek"], t["luas_px"], t["komponen_8arah"])
                                            == (r0["skor_objek"], r0["luas_px"], r0["komponen_8arah"])}
        res[f"pratinjau_{klip}"] = {"run": pj["run"], "device": pj["device"], "peringatan": pj["peringatan"]}
    res["banding_tahap1"] = cmp
    for name, klip in (("salah", K5), ("multi", K3)):
        pj = preview_json(SCR / name, klip)
        res[f"pratinjau_{name}"] = {"run": pj["run"], "peringatan": pj["peringatan"],
                                    "frames": [{"frame": fr["frame"], "targets": [{k: t[k] for k in (
                                        "id", "kiri", "kanan", "kotak", "skor_objek", "luas_px", "porsi_frame", "komponen_8arah",
                                        "porsi_komponen_terbesar", "waktu_s", "png", "peringatan")} for t in fr["targets"]]}
                                        for fr in pj["frames"]]}
    res["points_produksi_tidak_berubah"] = {k: file_sha(PROD / k / "target" / "points.json") == v for k, v in sha_before.items()}
    res["produksi_tanpa_pratinjau"] = {k: not (PROD / k / "target" / "pratinjau").exists() for k in (K5, K3)}
    for k, v in res["banding_tahap1"].items():
        print(f"{k}: {v}")
    for name in ("salah", "multi"):
        for fr in res[f"pratinjau_{name}"]["frames"]:
            for t in fr["targets"]:
                print(f"{name} f{fr['frame']} t{t['id']}: {t['kiri']}/{t['kanan']}/{int(t['kotak'])} skor {t['skor_objek']} luas "
                      f"{t['luas_px']} ({t['porsi_frame']}) komp {t['komponen_8arah']} ({t['porsi_komponen_terbesar']}) "
                      f"{t['waktu_s']} s {t['peringatan']}")
    jsave(OUT / "tahap3_jalan.json", res)


def cmd_bahan() -> None:
    dst = OUT / "nilai"
    if dst.exists():
        raise SystemExit(f"{dst} sudah ada — tidak ditimpa")
    res = {}
    for label, base, klip in (("klip5", SCR, K5), ("klip3", SCR, K3), ("salah_klip5", SCR / "salah", K5),
                              ("multi_klip3_SINTETIS", SCR / "multi", K3)):
        src = base / "work" / "clips" / klip / "target" / "pratinjau"
        pairs = copy_item(src, dst / label)
        res[label] = {"sumber": src.relative_to(ROOT).as_posix(), "berkas": sorted(d.name for _, d in pairs),
                      "sha_sama": all(file_sha(s) == file_sha(d) for s, d in pairs)}
        print(f"{label}: {len(pairs)} berkas -> {(dst / label).relative_to(ROOT).as_posix()}")
    jsave(OUT / "tahap3_bahan.json", res)


def main(argv: list[str]) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    cmds = {"siapkan": cmd_siapkan, "jalan": cmd_jalan, "bahan": cmd_bahan}
    if len(argv) != 1 or argv[0] not in cmds:
        print(__doc__)
        return 2
    cmds[argv[0]]()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
