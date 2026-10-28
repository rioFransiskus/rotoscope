"""T-608 1b-2 diagnosis (alat, F3; CPU): kenapa isolate klip5 kosong di SEMUA frame, termasuk frame klik.

    python scripts/t608_diag.py prompt    # SAM 2.1 tiny di CPU, sesi 1 frame per frame klik × varian prompt -> work/t608/iso_diag_prompt.json
    python scripts/t608_diag.py pratinjau [--points P] [--out DIR] [--uji-buang-kanan]
        # klik per frame dihitung (kanan > 10 -> PERINGATAN, berhenti tanpa render); SAM CPU sesi 1 frame dengan klik APA ADANYA ->
        # PNG FOTO + masker transparan + kotak + titik per frame klik + pratinjau.json. Default: titik scratch iso ->
        # work/t608/nilai_1b/pratinjau_klik/. `--uji-buang-kanan` = HANYA uji alat (klik kanan dibuang, batas 10 dilewati).

Varian dibentuk dari klik Rio di `work/t608_scratch/iso/.../target/points.json` (hanya DIBACA): `rio` (semua titik + kotak = run GPU),
`pos_kotak`, `pos`, `kotak`, `pos_kotak_neg10`, `pos_kotak_neg30` (subset klik kanan berjarak rata menurut urutan klik), `pos_neg_semua`
(tanpa kotak). Ini DIAGNOSIS (apakah prompt memberi objek di frame klik itu sendiri), BUKAN varian hasil: tidak ada pelacakan, tidak
ada masker yang disimpan (angka saja), scratch dan produksi tidak ditulis. CUDA_VISIBLE_DEVICES=-1; model dari cache HF (offline).
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"                 # pengaman: tanpa GPU

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from rotoscope import isolate as iso  # noqa: E402
from rotoscope.config import load_pipeline  # noqa: E402
from rotoscope.ingest import FRAME_PATTERN  # noqa: E402

SCR_CFG = ROOT / "work" / "t608_scratch" / "iso" / "config.yaml"
CLIP = ROOT / "work" / "t608_scratch" / "iso" / "work" / "clips" / "klip5"
OUT = ROOT / "work" / "t608"
GPU_SCORE = {0: -3.911, 22: -0.908, 40: -0.637}            # run GPU Rio 2026-10-28 (isolate/frames.jsonl)
NEG_SUBSETS = (10, 30)
PREVIEW_DIR = OUT / "nilai_1b" / "pratinjau_klik"
NEG_MAX = 10                                               # keputusan 1b-2 no. 3: klik kanan MAKSIMAL 10 per frame
POS_RANGE = (4, 8)                                         # klik kiri 4–8
EXPECTED_FRAMES = (0, 22, 40)
MASK_BGR = (255, 160, 0)                                   # biru muda transparan
MASK_ALPHA = 0.45
POS_BGR, NEG_BGR, BOX_BGR = (0, 200, 0), (0, 0, 255), (0, 215, 255)   # hijau, merah, kuning (= warna pick)
BAR_H = 44


def spread(items: list, n: int) -> list:
    if n >= len(items):
        return list(items)
    idx = np.linspace(0, len(items) - 1, n).round().astype(int)
    return [items[i] for i in sorted(set(idx.tolist()))]


def variants(t: dict) -> dict[str, tuple[list, tuple | None]]:
    pos = [p for p in t["points"] if p["label"] == 1]
    neg = [p for p in t["points"] if p["label"] == 0]
    box = tuple(t["box"]) if t.get("box") else None
    v = {"rio": (t["points"], box), "pos_kotak": (pos, box), "pos": (pos, None), "kotak": ([], box)}
    for n in NEG_SUBSETS:
        v[f"pos_kotak_neg{n}"] = (pos + spread(neg, n), box)
    v["pos_neg_semua"] = (t["points"], None)
    return {k: val for k, val in v.items() if val[0] or val[1] is not None}


def inside(p: dict, box) -> bool:
    return box is not None and box[0] <= p["x"] <= box[2] and box[1] <= p["y"] <= box[3]


def cmd_prompt() -> None:
    cfg = load_pipeline(SCR_CFG)
    doc = json.loads((CLIP / "target" / "points.json").read_text(encoding="utf-8"))
    meta = json.loads((CLIP / "meta.json").read_text(encoding="utf-8"))
    h, w = meta["working_height"], meta["working_width"]
    be = iso.TorchSamBackend(cfg, device="cpu")
    be.describe()
    opened = be.open()
    thr = cfg.isolate.mask_threshold
    res = {"device": opened.get("device"), "load_s": opened.get("load_s"), "mask_threshold": thr, "frame": {}}
    for e in doc["frames"]:
        f = e["frame_index"]
        t = e["targets"][0]
        box = t.get("box")
        neg = [p for p in t["points"] if p["label"] == 0]
        info = {"klik_kiri": sum(p["label"] == 1 for p in t["points"]), "klik_kanan": len(neg),
                "kanan_di_dalam_kotak": sum(inside(p, box) for p in neg), "kotak": box,
                "luas_kotak_px": (box[2] - box[0]) * (box[3] - box[1]) if box else None,
                "skor_gpu_run_rio": GPU_SCORE.get(f), "varian": {}}
        for name, (pts, bx) in variants(t).items():
            eng = be.session([CLIP / "frames" / (FRAME_PATTERN % f)], (h, w), 1)
            prompt = iso.Prompt(f, tuple((float(p["x"]), float(p["y"])) for p in pts), tuple(int(p["label"]) for p in pts),
                                tuple(float(v) for v in bx) if bx is not None else None)
            t0 = time.perf_counter()
            eng.add_prompt(0, prompt)
            logits, score = eng.run_frame(0)
            dt = time.perf_counter() - t0
            eng.release()
            m = iso.target_mask(logits, thr)
            in_box = None
            if bx is not None and m.any():
                x0, y0, x1, y1 = (int(round(v)) for v in bx)
                in_box = round(float(m[y0:y1 + 1, x0:x1 + 1].sum() / m.sum()), 4)
            info["varian"][name] = {"n_kiri": sum(p["label"] == 1 for p in pts), "n_kanan": sum(p["label"] == 0 for p in pts),
                                    "kotak": bx is not None, "skor_objek": round(score, 3), "luas_px": int(m.sum()),
                                    "porsi_frame": round(float(m.mean()), 4), "porsi_di_dalam_kotak": in_box,
                                    "logit_maks": round(float(np.max(logits)), 3), "waktu_s": round(dt, 2)}
            r = info["varian"][name]
            print(f"f{f} {name:16s} kiri {r['n_kiri']:2d} kanan {r['n_kanan']:2d} kotak {int(r['kotak'])}: skor {r['skor_objek']:7.3f}, "
                  f"luas {r['luas_px']:6d} px ({r['porsi_frame']:.3f}), logit maks {r['logit_maks']:.2f} [{dt:.1f} s]", flush=True)
        res["frame"][str(f)] = info
    be.close()
    p = OUT / "iso_diag_prompt.json"
    p.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"  -> {p.relative_to(ROOT).as_posix()}")
    import torch
    if torch.cuda.is_initialized():
        raise SystemExit("CUDA terinisialisasi — tidak boleh")


def count_clicks(doc: dict) -> dict[int, dict]:
    out = {}
    for e in doc["frames"]:
        tg = e["targets"]
        pts = [p for t in tg for p in t["points"]]
        out[e["frame_index"]] = {"target": len(tg), "kiri": sum(p["label"] == 1 for p in pts),
                                 "kanan": sum(p["label"] == 0 for p in pts), "kotak": sum(t.get("box") is not None for t in tg)}
    return out


def render(bgr, mask, pts: list, box, text: str):
    import cv2
    img = bgr.copy()
    if mask.any():
        col = np.zeros_like(img)
        col[:] = MASK_BGR
        img[mask] = (img[mask] * (1 - MASK_ALPHA) + col[mask] * MASK_ALPHA).astype(np.uint8)
        cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, cnts, -1, MASK_BGR, 2, cv2.LINE_AA)
    if box is not None:
        x0, y0, x1, y1 = (int(round(v)) for v in box)
        cv2.rectangle(img, (x0, y0), (x1, y1), BOX_BGR, 2, cv2.LINE_AA)
    for p in pts:
        c = (int(round(p["x"])), int(round(p["y"])))
        cv2.circle(img, c, 7, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img, c, 5, POS_BGR if p["label"] == 1 else NEG_BGR, -1, cv2.LINE_AA)
    bar = np.full((BAR_H, img.shape[1], 3), 235, np.uint8)
    cv2.putText(bar, text, (10, BAR_H - 14), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 0, 0), 2, cv2.LINE_AA)
    return np.vstack([bar, img])


def cmd_pratinjau(points: Path, out: Path, uji_buang_kanan: bool) -> int:
    import cv2
    doc = json.loads(points.read_text(encoding="utf-8"))
    counts = count_clicks(doc)
    res = {"points": points.relative_to(ROOT).as_posix(), "points_sha256": hashlib.sha256(points.read_bytes()).hexdigest(),
           "uji_buang_kanan": uji_buang_kanan, "klik": counts, "peringatan": []}
    for f, c in sorted(counts.items()):
        print(f"f{f}: target {c['target']}, kiri {c['kiri']}, kanan {c['kanan']}, kotak {c['kotak']}")
        if c["kanan"] > NEG_MAX:
            res["peringatan"].append(f"f{f}: klik kanan {c['kanan']} > {NEG_MAX}")
        if not POS_RANGE[0] <= c["kiri"] <= POS_RANGE[1]:
            res["peringatan"].append(f"f{f}: klik kiri {c['kiri']} di luar {POS_RANGE[0]}–{POS_RANGE[1]}")
        if c["kotak"] != 1 or c["target"] != 1:
            res["peringatan"].append(f"f{f}: target {c['target']} / kotak {c['kotak']} (harus 1 / 1)")
    if sorted(counts) != list(EXPECTED_FRAMES):
        res["peringatan"].append(f"frame klik {sorted(counts)} ≠ {list(EXPECTED_FRAMES)}")
    for w in res["peringatan"]:
        print("PERINGATAN:", w)
    over = [w for w in res["peringatan"] if "klik kanan" in w]
    if over and not uji_buang_kanan:
        print("BERHENTI: klik kanan > 10 — pratinjau tidak dibuat; klik ulang frame itu dulu")
        return 1
    cfg = load_pipeline(SCR_CFG)
    meta = json.loads((CLIP / "meta.json").read_text(encoding="utf-8"))
    h, w = meta["working_height"], meta["working_width"]
    be = iso.TorchSamBackend(cfg, device="cpu")
    be.describe()
    be.open()
    out.mkdir(parents=True, exist_ok=True)
    res["frame"] = {}
    for e in doc["frames"]:
        f = e["frame_index"]
        t = e["targets"][0]
        pts = [p for p in t["points"] if not (uji_buang_kanan and p["label"] == 0)]
        box = t.get("box")
        eng = be.session([CLIP / "frames" / (FRAME_PATTERN % f)], (h, w), 1)
        eng.add_prompt(0, iso.Prompt(f, tuple((float(p["x"]), float(p["y"])) for p in pts),
                                     tuple(int(p["label"]) for p in pts), tuple(float(v) for v in box) if box else None))
        logits, score = eng.run_frame(0)
        eng.release()
        m = iso.target_mask(logits, cfg.isolate.mask_threshold)
        bgr = cv2.imdecode(np.fromfile(CLIP / "frames" / (FRAME_PATTERN % f), np.uint8), cv2.IMREAD_COLOR)
        n_comp, big = 0, 0.0
        if m.any():
            n_lab, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
            n_comp = n_lab - 1
            big = float(st[1:, cv2.CC_STAT_AREA].max() / m.sum())
        sec = f"{f / 24:.2f}".replace(".", ",")
        img = render(bgr, m, pts, box, f"frame {f} (detik {sec})   skor objek {score:+.2f}   masker {m.mean() * 100:.1f}% frame")
        name = f"klik_f{f:05d}.png"
        ok, buf = cv2.imencode(".png", img)
        if not ok:
            raise SystemExit("imencode gagal")
        (out / name).write_bytes(buf.tobytes())
        res["frame"][str(f)] = {"png": name, "skor_objek": round(score, 3), "luas_px": int(m.sum()),
                                "porsi_frame": round(float(m.mean()), 4), "komponen_8arah": int(n_comp),
                                "porsi_komponen_terbesar": round(big, 4), "png_ukuran": [int(img.shape[1]), int(img.shape[0])]}
        r = res["frame"][str(f)]
        print(f"f{f}: skor objek {r['skor_objek']:+.3f}, masker {r['luas_px']} px ({r['porsi_frame'] * 100:.1f}%), "
              f"{r['komponen_8arah']} komponen (terbesar {r['porsi_komponen_terbesar'] * 100:.1f}% masker) -> {name}")
    be.close()
    (out / "pratinjau.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"  -> {(out / 'pratinjau.json').relative_to(ROOT).as_posix()}")
    import torch
    if torch.cuda.is_initialized():
        raise SystemExit("CUDA terinisialisasi — tidak boleh")
    return 0


def main(argv: list[str]) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if argv == ["prompt"]:
        cmd_prompt()
        return 0
    if argv[:1] == ["pratinjau"]:
        import argparse
        ap = argparse.ArgumentParser(prog="t608_diag.py pratinjau")
        ap.add_argument("--points", type=Path, default=CLIP / "target" / "points.json")
        ap.add_argument("--out", type=Path, default=PREVIEW_DIR)
        ap.add_argument("--uji-buang-kanan", action="store_true")
        a = ap.parse_args(argv[1:])
        return cmd_pratinjau(a.points.resolve(), a.out.resolve(), a.uji_buang_kanan)
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
