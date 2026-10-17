"""T-605 Tahap 3 (alat sekali pakai, F3): varian scratch + run GPU berurutan (dijalankan Rio) + bahan penilaian (CPU).

    python scripts/t605_tahap3.py siapkan    CPU: salinan scratch V-a / V-b / V-c / V-d di work/t605_scratch/<varian>/
                                             (frames/ di-hard-link, meta.json disalin, target/points.json ditulis)
    python scripts/t605_tahap3.py gpu [--varian V-a,V-b]
                                             GPU (Rio, PowerShell terpisah): [1b] tiap varian BERURUTAN (satu model di GPU),
                                             cek RAM bebas sebelum tiap varian (VRAM dicek stage [1b]); sampel nvidia-smi tiap 2 s;
                                             bisa diulang bila terputus (varian yang sudah up-to-date dilewati tanpa GPU)
    python scripts/t605_tahap3.py cek        CPU, baca-saja: verifikasi 6 hasil (status, kunci = input sekarang, PNG, NaN, waktu,
                                             VRAM / RAM) -> work/t605/cek_run.json
    python scripts/t605_tahap3.py bahan      CPU: video per adegan berklik (foto + tepi masker, normal + lambat 0,5x), video
                                             berdampingan antar varian, video ringkas per klip, metrik, cek determinisme V-a vs
                                             produksi -> work/t605/nilai/ + work/t605/metrik.json
    python scripts/t605_tahap3.py uji        CPU: uji asap `bahan` dengan SAM TIRUAN (lingkaran di sekitar klik) pada salinan
                                             scratch -> work/t605_scratch/uji/ (bukan bahan penilaian)

Varian (keputusan Rio setelah Tahap 1 no. 6):
  V-a klip3_ulang          titik Rio klip3 apa adanya (determinisme vs produksi)
  V-b klip3_t602           titik T-602 hasil konversi (work/t604/t602_converted/klip3.points.json: 15 kiri + 4 kanan)
  V-c klip5_tanpa_kotak64  titik Rio klip5, kotak f64 dibuang (titik f64 tetap)
  V-d klip5_f64saja        hanya entri f64 titik Rio klip5 (propagasi MUNDUR f64 -> f0 teruji)

Hanya MEMBACA work/clips/{klip3,klip5}/ (frames, meta.json, target/points.json, isolate/); menulis ke work/t605_scratch/ dan
work/t605/. Hard link frames/ tidak mengubah isi berkas sumber; [1b] hanya membaca frames/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rotoscope import isolate as iso  # noqa: E402
from rotoscope import pick  # noqa: E402
from rotoscope.config import load_pipeline  # noqa: E402
from rotoscope.ingest import FRAME_PATTERN  # noqa: E402
from rotoscope.stage_common import clip_identity, load_frame_list, read_rgb  # noqa: E402

CLIPS = ROOT / "work" / "clips"
SCR = ROOT / "work" / "t605_scratch"
OUT = ROOT / "work" / "t605"
NILAI = OUT / "nilai"
LOG = OUT / "run_tahap3.log"
GPU_JSON = OUT / "tahap3_gpu.json"
T602_KLIP3 = ROOT / "work" / "t604" / "t602_converted" / "klip3.points.json"
VARIANTS = {
    "V-a": ("klip3", "klip3_ulang", "titik Rio klip3 apa adanya (determinisme vs produksi)"),
    "V-b": ("klip3", "klip3_t602", "titik T-602 (15 kiri + 4 kanan)"),
    "V-c": ("klip5", "klip5_tanpa_kotak64", "titik Rio klip5, kotak f64 dibuang"),
    "V-d": ("klip5", "klip5_f64saja", "hanya klik f64 (maju + MUNDUR)"),
}
NVSMI_PERIOD_S = 2.0
BAR_H = 48
FONT = cv2.FONT_HERSHEY_SIMPLEX
# Toleransi determinisme (keputusan Rio setelah Tahap 1 no. 5)
DET_IOU_MIN, DET_DIFF_MAX = 0.995, 0.001
JUMP_REL = 0.20                       # lompatan luas |dA| / A > 20% (dilaporkan)
IOU_LOW = 0.80


def log(msg: str, fh=None) -> None:
    print(msg, flush=True)
    if fh is not None:
        fh.write(msg + "\n")
        fh.flush()


def dump_points(doc: dict) -> bytes:
    """Cara tulis pick (`stage_common.write_json_atomic`)."""
    return json.dumps(doc, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8")


# ── siapkan ────────────────────────────────────────
def variant_points(var: str) -> bytes:
    src_clip = VARIANTS[var][0]
    if var == "V-b":
        doc = json.loads(T602_KLIP3.read_text(encoding="utf-8"))
    else:
        doc = json.loads(iso.points_path(CLIPS / src_clip).read_text(encoding="utf-8"))
    if var == "V-c":
        for e in doc["frames"]:
            if e["frame_index"] == 64:
                for t in e["targets"]:
                    t["box"] = None
    elif var == "V-d":
        doc["frames"] = [e for e in doc["frames"] if e["frame_index"] == 64]
    pick.validate_doc(doc)
    return dump_points(doc)


def prepare(var: str, root: Path = SCR) -> Path:
    src_clip, name, _ = VARIANTS[var]
    src, dst = CLIPS / src_clip, root / name
    names, _, _, _ = load_frame_list(src)
    (dst / "frames").mkdir(parents=True, exist_ok=True)
    for n in names:                                       # hard link: isi sama, tanpa salinan 1,5 MB / frame
        d = dst / "frames" / n
        if not d.exists():
            os.link(src / "frames" / n, d)
    if not (dst / "meta.json").is_file() or (dst / "meta.json").read_bytes() != (src / "meta.json").read_bytes():
        shutil.copyfile(src / "meta.json", dst / "meta.json")
    pts = variant_points(var)
    p = iso.points_path(dst)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.is_file() or p.read_bytes() != pts:
        p.write_bytes(pts)
    doc = pick.load_points(p)                             # validasi sama dengan [1b]
    pick.check_doc_matches_clip(doc, clip_identity(dst), *load_frame_list(dst)[2:], p)
    st = pick.frame_statuses(doc, dst, set(load_frame_list(dst)[1]), {})
    if any(s != pick.STATUS_OK for s in st.values()):
        raise SystemExit(f"{var}: status frame klik {st} (harus ok)")
    return dst


def cmd_siapkan() -> None:
    for var, (src_clip, name, what) in VARIANTS.items():
        dst = prepare(var)
        doc = pick.load_points(iso.points_path(dst))
        isi = "; ".join(f"f{e['frame_index']}: " + ", ".join(
            f"target {t['id']} {sum(p['label'] == 1 for p in t['points'])} kiri / {sum(p['label'] == 0 for p in t['points'])} "
            f"kanan{' + kotak' if t['box'] else ''}" for t in e["targets"]) for e in doc["frames"])
        print(f"{var} {name} ({what}): {dst.relative_to(ROOT).as_posix()} — {isi}")


# ── gpu (Rio) ──────────────────────────────────────
class NvSampler:
    """Puncak memory.used seluruh GPU (nvidia-smi) tiap 2 s; opsional (tanpa nvidia-smi → None)."""

    def __init__(self):
        self.peak, self.total, self._stop = None, None, threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
                used, total = (float(v) for v in out.split(","))
                self.peak, self.total = max(self.peak or 0.0, used), total
            except (OSError, ValueError, IndexError, subprocess.SubprocessError):
                pass
            self._stop.wait(NVSMI_PERIOD_S)

    def __enter__(self):
        if shutil.which("nvidia-smi"):
            self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()


def cmd_gpu(only: list[str] | None) -> int:
    cfg = load_pipeline(ROOT / "configs" / "default.yaml")
    need = cfg.isolate.ram_min_free_mib
    OUT.mkdir(parents=True, exist_ok=True)
    res = json.loads(GPU_JSON.read_text(encoding="utf-8")) if GPU_JSON.is_file() else {}
    todo = [v for v in VARIANTS if only is None or v in only]
    with open(LOG, "a", encoding="utf-8") as fh:
        log(f"=== t605_tahap3 gpu {time.strftime('%Y-%m-%d %H:%M:%S')}: {', '.join(todo)} ===", fh)
        for var in todo:
            dst = prepare(var)
            free = iso.ram_free_mib()
            if free is not None and free < need:
                log(f"BERHENTI sebelum {var}: RAM bebas {free:.0f} MiB < {need} MiB. Tutup browser / aplikasi lain lalu jalankan "
                    f"ulang perintah yang sama (varian yang sudah selesai dilewati).", fh)
                return 1
            cmd = [sys.executable, "-m", "rotoscope.isolate", "--work-dir", str(dst), "--yes"]
            log(f"\n--- {var} {VARIANTS[var][1]} ({VARIANTS[var][2]}); RAM bebas {free} MiB ---", fh)
            t0 = time.perf_counter()
            with NvSampler() as nv:
                proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                        encoding="utf-8", errors="replace", env={**os.environ, "PYTHONIOENCODING": "utf-8"})
                try:
                    for line in proc.stdout:
                        log(line.rstrip(), fh)
                    rc = proc.wait()
                except KeyboardInterrupt:
                    proc.terminate()
                    proc.wait()
                    log(f"DIHENTIKAN (Ctrl+C) di {var}. Jalankan ulang perintah yang sama untuk melanjutkan.", fh)
                    return 130
            res[var] = {"rc": rc, "wall_s": round(time.perf_counter() - t0, 1), "nvidia_smi_peak_mib": nv.peak,
                        "nvidia_smi_total_mib": nv.total, "ram_free_before_mib": free, "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
            GPU_JSON.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
            log(f"{var}: rc {rc}, {res[var]['wall_s']} s, puncak nvidia-smi {nv.peak} / {nv.total} MiB", fh)
            if rc != 0:
                log(f"BERHENTI: {var} gagal (rc {rc}; 3 = OOM). Lihat pesan di atas. Tutup aplikasi lain yang memakai GPU, lalu "
                    f"jalankan ulang perintah yang sama (varian yang sudah selesai dilewati).", fh)
                return rc
        log("\nSELESAI semua varian. Tempel ke Claude Code: \"T-605 Tahap 3: run GPU selesai, lanjut bahan\"", fh)
    return 0


# ── bahan (CPU) ────────────────────────────────────
def load_result(work: Path) -> dict | None:
    d = work / iso.ISOLATE_DIRNAME
    m = iso.read_manifest(d / iso.MANIFEST_FILENAME)
    if m is None:
        return None
    rows = {r["index"]: r for r in iso.read_frame_rows(d / iso.FRAMES_LOG_FILENAME)}
    meta = json.loads((work / "meta.json").read_text(encoding="utf-8"))
    return {"work": work, "manifest": m, "rows": rows, "fps": float(meta.get("target_fps", 24)),
            "size": (int(meta["working_height"]), int(meta["working_width"]))}


def label_of(res: dict, i: int) -> np.ndarray | None:
    return iso.read_label(iso.label_path(res["work"] / iso.ISOLATE_DIRNAME, i))


def iou(a: np.ndarray, b: np.ndarray) -> float | None:
    u = int((a | b).sum())
    return None if u == 0 else round(int((a & b).sum()) / u, 5)


def berklik(res: dict) -> list[tuple[int, int]]:
    return sorted({tuple(j["shot"]) for j in res["manifest"]["jobs"]})


def metrics(res: dict) -> dict:
    m, rows = res["manifest"], res["rows"]
    h, w = res["size"]
    out: dict = {"frames": m["frames"], "jobs": [{k: j[k] for k in ("shot", "target", "prompt_frames", "n_infer", "infer_s")}
                                                 for j in m["jobs"]], "run": m.get("run")}
    per_t: dict = {}
    for t in m["targets"]:
        prev, areas, ious, jumps, scores, low = None, {}, [], [], {}, []
        for lo, hi in berklik(res):
            prev = None
            for i in range(lo, hi + 1):
                lab = label_of(res, i)
                cur = lab == t
                a = int(cur.sum())
                areas[i] = a
                tr = rows[i]["targets"].get(str(t))
                scores[i] = tr["score_logit"] if tr else None
                if prev is not None:
                    v = iou(prev, cur)
                    if v is not None:
                        ious.append(v)
                        if v < IOU_LOW:
                            low.append(i)
                    pa = int(prev.sum())
                    if max(pa, a) > 0 and abs(a - pa) / max(pa, 1) > JUMP_REL:
                        jumps.append([i - 1, i, pa, a])
                prev = cur
        vals = [a for a in areas.values()]
        per_t[str(t)] = {
            "luas_px": {"min": min(vals) if vals else None, "median": float(np.median(vals)) if vals else None,
                        "max": max(vals) if vals else None},
            "luas_pct_median": round(100 * float(np.median(vals)) / (h * w), 2) if vals else None,
            "iou_t_t1": {"median": float(np.median(ious)) if ious else None, "min": min(ious) if ious else None,
                         "frame_iou_lt_0_8": low},
            "lompatan_luas_gt_20pct": jumps,
            "skor_le_0": [i for i, s in scores.items() if s is not None and s <= 0],
            "skor_min": min((s for s in scores.values() if s is not None), default=None),
            "luas_per_frame": areas, "skor_per_frame": scores}
    out["per_target"] = per_t
    return out


def compare(a: dict, b: dict, frames: list[int]) -> dict:
    """b vs a per frame: byte PNG sama, piksel beda, IoU per target, status sama."""
    same_bytes, diffs, ious, status_same = 0, {}, {}, True
    for i in frames:
        pa = iso.label_path(a["work"] / iso.ISOLATE_DIRNAME, i)
        pb = iso.label_path(b["work"] / iso.ISOLATE_DIRNAME, i)
        if a["rows"][i]["status"] != b["rows"][i]["status"]:
            status_same = False
        if not (pa.is_file() and pb.is_file()):
            continue
        if pa.read_bytes() == pb.read_bytes():
            same_bytes += 1
        la, lb = iso.read_label(pa), iso.read_label(pb)
        nd = int((la != lb).sum())
        area = max(int((la > 0).sum()), 1)
        diffs[i] = {"piksel_beda": nd, "fraksi_luas": round(nd / area, 6)}
        for t in sorted((set(np.unique(la)) | set(np.unique(lb))) - {0}):
            ious.setdefault(str(int(t)), {})[i] = iou(la == t, lb == t)
    all_iou = [v for d in ious.values() for v in d.values() if v is not None]
    return {"frame": len(frames), "png_byte_identik": same_bytes, "status_sama": status_same,
            "iou_min": min(all_iou) if all_iou else None, "iou_median": float(np.median(all_iou)) if all_iou else None,
            "piksel_beda_maks": max((d["piksel_beda"] for d in diffs.values()), default=0),
            "fraksi_beda_maks": max((d["fraksi_luas"] for d in diffs.values()), default=0.0),
            "iou_per_target": ious, "beda_per_frame": {i: d for i, d in diffs.items() if d["piksel_beda"]}}


def overlay(rgb: np.ndarray, lab: np.ndarray | None, text: str, sub: str = "") -> np.ndarray:
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if lab is not None:
        for t in [int(v) for v in np.unique(lab) if v]:
            m = (lab == t).astype(np.uint8)
            col = np.array(pick.TARGET_COLORS[(t - 1) % len(pick.TARGET_COLORS)], np.float32)
            sel = m.astype(bool)
            img[sel] = (0.8 * img[sel] + 0.2 * col).astype(np.uint8)
            cs, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(img, cs, -1, tuple(int(c) for c in col), 2, cv2.LINE_AA)
    bar = np.full((BAR_H, img.shape[1], 3), 28, np.uint8)
    cv2.putText(bar, text, (8, 20), FONT, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
    if sub:
        cv2.putText(bar, sub, (8, 41), FONT, 0.5, (0, 230, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def frame_text(res: dict, name: str, i: int) -> tuple[str, str]:
    r = res["rows"][i]
    h, w = res["size"]
    st = r["status"]
    sub = "TANPA TARGET (adegan tanpa klik)" if st == iso.STATUS_NO_PROMPT else st
    if r["targets"]:
        sub += "  " + "  ".join(f"t{t}: {v['kind']} {100 * v['area_px'] / (h * w):.1f}% skor {v['score_logit']}"
                                for t, v in sorted(r["targets"].items()))
    return f"{name}  f{i:03d}  t={i / res['fps']:.2f} s", sub


def write_video(path: Path, frames, fps: float) -> int:
    proc, n, size = None, 0, None
    for img in frames:
        if proc is None:
            size = (img.shape[1], img.shape[0])
            proc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s",
                                     f"{size[0]}x{size[1]}", "-r", f"{fps:g}", "-i", "-", "-c:v", "libx264", "-preset", "veryfast",
                                     "-crf", "20", "-pix_fmt", "yuv420p", str(path)], stdin=subprocess.PIPE)
        proc.stdin.write(np.ascontiguousarray(img).tobytes())
        n += 1
    if proc is not None:
        proc.stdin.close()
        if proc.wait() != 0:
            raise SystemExit(f"ffmpeg gagal menulis {path}")
    return n


def panel(res: dict, name: str, i: int, scale: float = 1.0) -> np.ndarray:
    rgb = read_rgb(res["work"] / "frames" / (FRAME_PATTERN % i))
    lab = label_of(res, i)
    img = overlay(rgb, lab, *frame_text(res, name, i))
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale != 1.0 else img


def even(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    return img[: h - h % 2, : w - w % 2]


def make_videos(prod: dict[str, dict], var: dict[str, dict], out_dir: Path) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    made = []
    for clip, res in prod.items():
        for lo, hi in berklik(res):
            base = out_dir / f"{clip}_f{lo:03d}-{hi:03d}"
            for suffix, fps in (("", res["fps"]), ("_lambat", res["fps"] / 2)):
                n = write_video(base.with_name(base.name + suffix + ".mp4"),
                                (even(panel(res, clip, i)) for i in range(lo, hi + 1)), fps)
                made.append({"video": base.name + suffix + ".mp4", "frame": [lo, hi], "detik": [round(lo / res["fps"], 2),
                             round((hi + 1) / res["fps"], 2)], "n": n, "fps": fps})
        n = write_video(out_dir / f"{clip}_ringkas.mp4", (even(panel(res, clip, i, 0.5)) for i in sorted(res["rows"])), res["fps"])
        made.append({"video": f"{clip}_ringkas.mp4", "frame": [min(res["rows"]), max(res["rows"])], "n": n, "fps": res["fps"]})
    groups = {"klip3": ["V-b"], "klip5": ["V-c", "V-d"]}
    for clip, vs in groups.items():
        if clip not in prod or not any(v in var for v in vs):
            continue
        res = prod[clip]
        have = [v for v in vs if v in var]
        for lo, hi in berklik(res):
            name = f"{clip}_banding_{'_'.join(have)}_f{lo:03d}-{hi:03d}"
            for suffix, fps in (("", res["fps"]), ("_lambat", res["fps"] / 2)):
                def frames():
                    for i in range(lo, hi + 1):
                        ps = [panel(res, f"{clip} produksi", i, 0.5)] + [
                            panel(var[v], f"{v} {VARIANTS[v][2][:28]}", i, 0.5) for v in have]
                        yield even(np.hstack(ps))
                n = write_video(out_dir / f"{name}{suffix}.mp4", frames(), fps)
                made.append({"video": f"{name}{suffix}.mp4", "frame": [lo, hi], "n": n, "fps": fps, "panel": ["produksi", *have]})
    return made


def box_area(doc: dict, frame: int) -> float | None:
    for e in doc["frames"]:
        if e["frame_index"] == frame:
            for t in e["targets"]:
                if t["box"]:
                    x0, y0, x1, y1 = t["box"]
                    return (x1 - x0) * (y1 - y0)
    return None


def cmd_bahan(prod_dirs: dict[str, Path], var_dirs: dict[str, Path], out_dir: Path, metrik_path: Path) -> int:
    prod = {c: r for c, d in prod_dirs.items() if (r := load_result(d)) is not None}
    var = {v: r for v, d in var_dirs.items() if (r := load_result(d)) is not None}
    missing = [c for c in prod_dirs if c not in prod] + [v for v in var_dirs if v not in var]
    if missing:
        print(f"PERINGATAN: belum ada hasil isolate/ untuk: {', '.join(missing)} (dilewati)")
    if not prod:
        print("tidak ada hasil produksi — jalankan dulu perintah Rio di PANDUAN_UJI.md")
        return 1
    gpu = json.loads(GPU_JSON.read_text(encoding="utf-8")) if GPU_JSON.is_file() else {}
    met: dict = {"produksi": {c: metrics(r) for c, r in prod.items()}, "varian": {v: metrics(r) for v, r in var.items()},
                 "gpu_varian": gpu, "banding": {}}
    if "klip3" in prod and "V-a" in var:
        fr = sorted(prod["klip3"]["rows"])
        c = compare(prod["klip3"], var["V-a"], fr)
        c["lolos_toleransi"] = bool(c["status_sama"] and (c["iou_min"] is None or c["iou_min"] >= DET_IOU_MIN)
                                    and c["fraksi_beda_maks"] <= DET_DIFF_MAX)
        c["byte_identik_semua"] = c["png_byte_identik"] == sum(1 for i in fr if prod["klip3"]["rows"][i]["status"] != iso.STATUS_NO_PROMPT)
        met["determinisme_V-a_vs_produksi"] = c
    for v, clip in (("V-b", "klip3"), ("V-c", "klip5"), ("V-d", "klip5")):
        if clip in prod and v in var:
            fr = [i for lo, hi in berklik(prod[clip]) for i in range(lo, hi + 1)]
            met["banding"][f"{v}_vs_produksi"] = compare(prod[clip], var[v], fr)
    if "klip5" in prod:
        ba = box_area(pick.load_points(iso.points_path(prod["klip5"]["work"])), 64)
        rec = {"luas_kotak_f64_px": ba}
        for name, r in (("produksi", prod["klip5"]), *(((v, var[v]) for v in ("V-c", "V-d") if v in var))):
            a = int((label_of(r, 64) == 1).sum()) if label_of(r, 64) is not None else None
            rec[name] = {"luas_masker_f64_px": a, "masker_per_kotak": round(a / ba, 3) if a is not None and ba else None}
        met["kotak_f64"] = rec
    met["video"] = make_videos(prod, var, out_dir)
    metrik_path.parent.mkdir(parents=True, exist_ok=True)
    metrik_path.write_text(json.dumps(met, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    for c, m in met["produksi"].items():
        f = m["frames"]
        print(f"{c}: target {f['target']}, kosong {f['kosong']}, tanpa_prompt {f['tanpa_prompt']}; "
              + "; ".join(f"t{t}: luas median {d['luas_pct_median']}%, IoU t/t-1 median {d['iou_t_t1']['median']} min "
                          f"{d['iou_t_t1']['min']}, lompatan {len(d['lompatan_luas_gt_20pct'])}, skor<=0 {len(d['skor_le_0'])}"
                          for t, d in m["per_target"].items()))
    if "determinisme_V-a_vs_produksi" in met:
        d = met["determinisme_V-a_vs_produksi"]
        print(f"determinisme V-a vs produksi: PNG byte-identik {d['png_byte_identik']} frame, status sama {d['status_sama']}, "
              f"IoU min {d['iou_min']}, fraksi beda maks {d['fraksi_beda_maks']} -> lolos toleransi {d['lolos_toleransi']}")
    for k, d in met["banding"].items():
        print(f"{k}: IoU median {d['iou_median']} min {d['iou_min']}, status sama {d['status_sama']}")
    print(f"{len(met['video'])} video -> {out_dir}; metrik -> {metrik_path}")
    return 0


# ── cek (CPU, baca-saja) ───────────────────────────
def cmd_cek() -> int:
    """Verifikasi hasil run GPU: status per frame, kunci = input SEKARANG (up-to-date, fungsi stage), PNG lengkap, NaN 0, waktu,
    VRAM / RAM. Tidak menulis apa pun kecuali work/t605/cek_run.json."""
    cfg = load_pipeline(ROOT / "configs" / "default.yaml")
    dirs = {"klip3": CLIPS / "klip3", "klip5": CLIPS / "klip5", **{v: SCR / VARIANTS[v][1] for v in VARIANTS}}
    gpu = json.loads(GPU_JSON.read_text(encoding="utf-8")) if GPU_JSON.is_file() else {}
    out, ok_all = {}, True
    for name, d in dirs.items():
        clip = iso.load_clip(d)
        m = iso.read_manifest(d / iso.ISOLATE_DIRNAME / iso.MANIFEST_FILENAME)
        if m is None:
            out[name] = {"ada": False}
            ok_all = False
            print(f"{name}: isolate/manifest.json TIDAK ADA")
            continue
        frames_sha, per = iso.frames_fingerprint(clip)
        key = iso.build_key(cfg, hashlib.sha256(iso.points_path(d).read_bytes()).hexdigest(), frames_sha)
        state, reasons = iso.existing_state(clip, key, clip_identity(d))
        rows = iso.read_frame_rows(d / iso.ISOLATE_DIRNAME / iso.FRAMES_LOG_FILENAME)
        r = m["run"]
        rec = {"ada": True, "state": state, "alasan": reasons, "key_sha256_cocok": m["key_sha256"] == iso.canonical_sha256(m["key"]),
               "frames": {k: m["frames"][k] for k in ("target", "kosong", "tanpa_prompt")},
               "kosong_ranges": m["frames"]["kosong_ranges"], "n_rows": len(rows),
               "n_png": len(list((d / iso.ISOLATE_DIRNAME / iso.LABELS_DIRNAME).glob("*.png"))),
               "jobs": [{k: j[k] for k in ("shot", "target", "prompt_frames", "start", "reverse", "n_infer", "infer_s")} for j in m["jobs"]],
               "nonfinite_frames": r.get("nonfinite_frames"), "wall_s": r.get("wall_s"), "key_s": r.get("key_s"),
               "scenes_s": r.get("scenes_s"), "load_s": (r.get("open") or {}).get("load_s"), "infer_s": r.get("infer_s"),
               "vram": r.get("vram"), "ram": r.get("ram"), "device": (r.get("open") or {}).get("device"),
               "gpu": (r.get("open") or {}).get("gpu"), "precision": m["key"]["precision"], "model": m.get("model"),
               "libs": m["key"]["libs"], "started_utc": r.get("started_utc"), "finished_utc": r.get("finished_utc"),
               "wrapper": gpu.get(name)}
        ok = (state == "ok" and rec["key_sha256_cocok"] and not rec["nonfinite_frames"] and rec["n_rows"] == len(clip.indices)
              and rec["device"] == "cuda" and rec["precision"] == "fp32")
        rec["lolos"] = ok
        ok_all &= ok
        out[name] = rec
        inf, vr, ram = rec["infer_s"], rec["vram"] or {}, rec["ram"] or {}
        print(f"{name}: {state}, target {rec['frames']['target']} / kosong {rec['frames']['kosong']} / tanpa_prompt "
              f"{rec['frames']['tanpa_prompt']}, PNG {rec['n_png']}, baris {rec['n_rows']}, NaN {rec['nonfinite_frames']}; SAM "
              f"{inf['n']} frame p50 {inf['p50']} s maks {inf['max']} s total {inf['total']} s, wall {rec['wall_s']} s; VRAM bebas "
              f"{vr.get('vram_free_before_load_mib')} / {vr.get('vram_total_mib')}, reserved maks {vr.get('max_reserved_mib')}, "
              f"allocated maks {vr.get('max_allocated_mib')} MiB; RAM bebas {ram.get('free_before_mib')}, puncak proses "
              f"{ram.get('peak_process_mib')} MiB -> {'LOLOS' if ok else 'GAGAL'}")
    (OUT / "cek_run.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"SEMUA LOLOS: {ok_all} -> work/t605/cek_run.json")
    return 0 if ok_all else 1


# ── uji (SAM tiruan) ───────────────────────────────
def cmd_uji() -> int:
    """Uji asap `bahan` tanpa GPU: [1b] dengan SAM tiruan (lingkaran r 60 px di sekitar klik positif pertama) pada salinan
    scratch klip3 / klip5 + V-a–V-d; keluaran di work/t605_scratch/uji/ (BUKAN bahan penilaian)."""
    sys.path.insert(0, str(ROOT / "tests"))
    import test_isolate as ti

    uji = SCR / "uji"
    prod_dirs = {}
    for clip in ("klip3", "klip5"):
        dst = uji / clip
        names, _, _, _ = load_frame_list(CLIPS / clip)
        (dst / "frames").mkdir(parents=True, exist_ok=True)
        for n in names:
            if not (dst / "frames" / n).exists():
                os.link(CLIPS / clip / "frames" / n, dst / "frames" / n)
        shutil.copyfile(CLIPS / clip / "meta.json", dst / "meta.json")
        iso.points_path(dst).parent.mkdir(parents=True, exist_ok=True)
        iso.points_path(dst).write_bytes(iso.points_path(CLIPS / clip).read_bytes())
        prod_dirs[clip] = dst
    var_dirs = {v: prepare(v, uji) for v in VARIANTS}
    cfg_dirs = {**prod_dirs, **var_dirs}
    for name, d in cfg_dirs.items():
        fac = ti.factory(radius={1: 60.0})
        cfg = load_pipeline(ROOT / "configs" / "default.yaml", overrides={"paths.work_dir": str(d)})
        r = iso.run_isolate(cfg, yes=True, backend_factory=fac, log=lambda m: None, ram_probe=lambda: None)
        print(f"uji {name}: {r['status']}")
    return cmd_bahan(prod_dirs, var_dirs, uji / "nilai", uji / "metrik.json")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(prog="python scripts/t605_tahap3.py")
    p.add_argument("cmd", choices=("siapkan", "gpu", "cek", "bahan", "uji"))
    p.add_argument("--varian", default=None, help="mis. V-a,V-c (default semua; hanya untuk gpu)")
    a = p.parse_args()
    if a.cmd == "siapkan":
        cmd_siapkan()
        return 0
    if a.cmd == "gpu":
        only = [v.strip() for v in a.varian.split(",")] if a.varian else None
        bad = [v for v in only or [] if v not in VARIANTS]
        if bad:
            p.error(f"varian tidak dikenal: {bad}")
        return cmd_gpu(only)
    if a.cmd == "cek":
        return cmd_cek()
    if a.cmd == "bahan":
        return cmd_bahan({c: CLIPS / c for c in ("klip3", "klip5")},
                         {v: SCR / VARIANTS[v][1] for v in VARIANTS}, NILAI, OUT / "metrik.json")
    return cmd_uji()


if __name__ == "__main__":
    sys.exit(main())
