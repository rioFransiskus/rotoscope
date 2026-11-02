"""T-609 Tahap 1 (alat, F3; CPU): ukur bagian (b) — pratinjau masker SAM 2.1 tiny di CPU per frame klik.

    python scripts/t609_ukur.py siapkan        # salinan scratch (BUKAN hard link) klip5@0-2.25 + klip3: meta.json, frames/,
                                               # target/points.json + config scratch (configs/default.yaml, hanya `paths` diganti;
                                               # dibandingkan SESUDAH load) -> work/t609/salinan.json
    python scripts/t609_ukur.py ukur KLIP      # SATU proses per klip: muat model di CPU + sesi 1 frame per (frame klik, target),
                                               # prompt APA ADANYA, 2 putaran (putaran 2 = cek ulang determinisme + waktu hangat)
                                               # -> PNG di work/t609_scratch/ukur/KLIP/ + work/t609/ukur_KLIP.json
    python scripts/t609_ukur.py putusan        # gabung ukur_*.json + aturan keluar no. 3 (dikunci sebelum ukur) -> work/t609/ukur_putusan.json

Aturan keluar no. 3 (docs/04 "Keputusan T-609 + T-305e SKIP"): bagian (b) dibuang bila median waktu per frame klik (putaran 1, tanpa
muat model) > 15 s, ATAU puncak RAM proses > 4 GiB (4096 MiB), ATAU tidak bisa berjalan tanpa CUDA. CUDA_VISIBLE_DEVICES=-1 dipasang
SEBELUM import apa pun; model dari cache HF yang dipin (offline, tanpa unduhan). Pembanding kabel: hasil [1b] produksi di frame klik
(`isolate/frames.jsonl`, kind `klik`, GPU) + angka pratinjau T-608 (docs/04 "(d) klik selesai #3"). Produksi hanya DIBACA.
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"                 # pengaman: tanpa GPU (sebelum torch di-import di mana pun)

import ctypes  # noqa: E402
import dataclasses  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import statistics  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from rotoscope import isolate as iso  # noqa: E402
from rotoscope import pick  # noqa: E402
from rotoscope.config import load_pipeline  # noqa: E402
from rotoscope.ingest import FRAME_PATTERN  # noqa: E402
from rotoscope.stage_common import MIB, clip_identity  # noqa: E402

KLIPS = ("klip5@0-2.25", "klip3")
ITEMS = ("meta.json", "frames", "target/points.json")
PROD = ROOT / "work" / "clips"
DEFAULT_CFG = ROOT / "configs" / "default.yaml"
SCR = ROOT / "work" / "t609_scratch"
SCR_CFG = SCR / "config.yaml"
SCR_WORK = SCR / "work"
UKUR_DIR = SCR / "ukur"
OUT = ROOT / "work" / "t609"
PATH_LINES = {'  work_dir: "work"': '  work_dir: "work/t609_scratch/work"',
              '  out_dir: "out"': '  out_dir: "work/t609_scratch/out"'}
EXIT_RULE = {"median_s_max": 15.0, "peak_ram_mib_max": 4096.0}          # keputusan Rio no. 3 (dikunci sebelum ukur)
T608_PREVIEW = {0: {"skor": 18.397, "luas": 84783, "komponen": 35, "terbesar": 0.979},   # docs/04 "(d) klik selesai #3" (#2 = #3 f0/f22)
                22: {"skor": 18.169, "luas": 79520, "komponen": 13, "terbesar": 0.999},
                40: {"skor": 17.079, "luas": 78587, "komponen": 21, "terbesar": 0.996}}
PASSES = 2
MASK_BGR, MASK_ALPHA = (255, 160, 0), 0.45
POS_BGR, NEG_BGR, BOX_BGR = (0, 200, 0), (0, 0, 255), (0, 215, 255)
BAR_H = 44


def file_sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def jsave(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"  -> {p.relative_to(ROOT).as_posix()}")


def as_plain(obj):
    if dataclasses.is_dataclass(obj):
        return {f.name: as_plain(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, dict):
        return {str(k): as_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [as_plain(v) for v in obj]
    if isinstance(obj, Path):
        return obj.as_posix()
    return obj


def flat(d, prefix=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flat(v, f"{prefix}{k}."))
    else:
        out[prefix[:-1]] = d
    return out


# ── siapkan ────────────────────────────────────────
def copy_item(src: Path, dst: Path) -> list[tuple[Path, Path]]:
    pairs = [(src, dst)] if src.is_file() else [(p, dst / p.relative_to(src)) for p in sorted(src.rglob("*")) if p.is_file()]
    for s, d in pairs:
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(s, d)
    return pairs


def cmd_siapkan() -> None:
    if SCR.exists():
        raise SystemExit(f"{SCR} sudah ada — salinan tidak ditimpa (hapus dulu bila memang harus diulang)")
    res: dict = {"klip": {}}
    for k in KLIPS:
        pairs = []
        for it in ITEMS:
            pairs += copy_item(PROD / k / it, SCR_WORK / "clips" / k / it)
        bad = [d.relative_to(ROOT).as_posix() for s, d in pairs if file_sha(s) != file_sha(d)]
        linked = [d.relative_to(ROOT).as_posix() for s, d in pairs if os.stat(d).st_nlink != 1 or os.path.samefile(s, d)]
        res["klip"][k] = {"berkas": len(pairs), "bytes": sum(d.stat().st_size for _, d in pairs), "sha256_beda": bad,
                          "hard_link_atau_sama": linked, "points_sha256": file_sha(PROD / k / "target" / "points.json"),
                          "meta_sha256": file_sha(PROD / k / "meta.json")}
        print(f"{k}: {len(pairs)} berkas; sha256 beda {len(bad)}; hard link {len(linked)}")
        if bad or linked:
            raise SystemExit(f"{k}: salinan tidak valid — berhenti")
    text = DEFAULT_CFG.read_text(encoding="utf-8")
    for old, new in PATH_LINES.items():
        if text.count(old) != 1:
            raise SystemExit(f"baris {old!r} tidak tepat satu kali di configs/default.yaml — berhenti")
        text = text.replace(old, new)
    SCR_CFG.write_text(text, encoding="utf-8")
    base, cur = flat(as_plain(load_pipeline(DEFAULT_CFG))), flat(as_plain(load_pipeline(SCR_CFG)))
    diff = sorted(k for k in set(base) | set(cur) if base.get(k) != cur.get(k))
    wd = load_pipeline(SCR_CFG).paths.work_dir.resolve()
    res["config"] = {"path": SCR_CFG.relative_to(ROOT).as_posix(), "sha256": file_sha(SCR_CFG), "beda_dari_default": diff,
                     "work_dir": wd.as_posix(), "work_dir_di_scratch": wd.is_relative_to(SCR.resolve())}
    print(f"config scratch: beda dari default = {diff}; work_dir {wd} (di scratch: {res['config']['work_dir_di_scratch']})")
    if any(not k.startswith("paths.") for k in diff) or not res["config"]["work_dir_di_scratch"]:
        raise SystemExit("config scratch berbeda di luar paths / work_dir bukan scratch — berhenti")
    jsave(OUT / "salinan.json", res)


# ── ukur ───────────────────────────────────────────
class _ProcMem(ctypes.Structure):
    _fields_ = iso._ProcMem._fields_


def mem_mib() -> dict:
    """Working set sekarang + puncak, private (pagefile) sekarang + puncak — proses ini (GetProcessMemoryInfo)."""
    k32, psapi = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
    k32.GetCurrentProcess.restype = ctypes.c_void_p
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(_ProcMem), ctypes.c_ulong]
    m = _ProcMem()
    m.cb = ctypes.sizeof(m)
    if not psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(m), m.cb):
        raise SystemExit("GetProcessMemoryInfo gagal")
    return {"ws": round(m.WorkingSetSize / MIB, 1), "ws_peak": round(m.PeakWorkingSetSize / MIB, 1),
            "private": round(m.PagefileUsage / MIB, 1), "private_peak": round(m.PeakPagefileUsage / MIB, 1)}


def render(bgr: np.ndarray, mask: np.ndarray, pts: list, box, text: str) -> np.ndarray:
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


def production_klik(k: str) -> dict:
    """Hasil [1b] produksi (GPU) di frame klik + apakah kuncinya = points.json sekarang. Hanya dibaca."""
    d = PROD / k / "isolate"
    man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    pts_sha = file_sha(PROD / k / "target" / "points.json")
    rows = [json.loads(ln) for ln in (d / "frames.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
    klik = {r["index"]: {tid: v for tid, v in r["targets"].items() if v.get("kind") == "klik"} for r in rows}
    return {"kunci_points_sama": man["key"].get("points_sha256") == pts_sha, "device": (man.get("run") or {}).get("open", {}).get("device"),
            "klik": {f: v for f, v in klik.items() if v}}


def cmd_ukur(k: str) -> None:
    import cv2
    t_proc = time.perf_counter()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise SystemExit("CUDA_VISIBLE_DEVICES bukan -1 — berhenti")
    cfg = load_pipeline(SCR_CFG)
    if not cfg.paths.work_dir.resolve().is_relative_to(SCR.resolve()):
        raise SystemExit(f"paths.work_dir {cfg.paths.work_dir} bukan scratch — berhenti")
    clip_dir = cfg.paths.work_dir / "clips" / k
    meta = json.loads((clip_dir / "meta.json").read_text(encoding="utf-8"))
    h, w, fps = meta["working_height"], meta["working_width"], meta["target_fps"]
    start = (meta.get("trim") or {}).get("start_frame", 0)
    ppath = clip_dir / "target" / "points.json"
    doc = pick.load_points(ppath)
    pick.check_doc_matches_clip(doc, clip_identity(clip_dir), w, h, ppath)
    statuses = pick.frame_statuses(doc, clip_dir, set(range(meta["frame_count"])), {})
    if any(s != pick.STATUS_OK for s in statuses.values()):
        raise SystemExit(f"status frame klik bukan ok: {statuses}")
    mem0 = mem_mib()
    t0 = time.perf_counter()
    import torch                                                       # noqa: F401  (waktu import dicatat terpisah)
    import transformers                                                # noqa: F401
    import_s = time.perf_counter() - t0
    res: dict = {"klip": k, "points_sha256": file_sha(ppath), "frame_size": [w, h], "fps": fps, "trim_start_frame": start,
                 "torch": torch.__version__, "transformers": transformers.__version__, "threads": torch.get_num_threads(),
                 "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "hf_hub_offline_sebelum": os.environ.get("HF_HUB_OFFLINE"),
                 "cuda_available": torch.cuda.is_available(), "status_frame": {str(f): s for f, s in statuses.items()},
                 "mem_awal_mib": mem0, "import_s": round(import_s, 2)}
    be = iso.TorchSamBackend(cfg, device="cpu")
    t0 = time.perf_counter()
    info = be.describe()
    res["describe_s"] = round(time.perf_counter() - t0, 2)
    t0 = time.perf_counter()
    opened = be.open()
    res["open_s"] = round(time.perf_counter() - t0, 2)
    res.update(model={"id": info.model_id, "revision": info.revision, "license": info.license, "precision": info.precision},
               open=opened, hf_hub_offline_sesudah=os.environ.get("HF_HUB_OFFLINE"), mem_sesudah_muat_mib=mem_mib())
    thr = cfg.isolate.mask_threshold
    out_dir = UKUR_DIR / k
    out_dir.mkdir(parents=True, exist_ok=True)
    res["putaran"] = []
    masks0: dict[tuple[int, int], np.ndarray] = {}
    for rnd in range(PASSES):
        rows = []
        for e in doc["frames"]:
            f = e["frame_index"]
            fpath = clip_dir / "frames" / (FRAME_PATTERN % f)
            for t in e["targets"]:
                pts, box = t["points"], t.get("box")
                tt = {}
                t_all = time.perf_counter()
                eng = be.session([fpath], (h, w), int(t["id"]))
                tt["sesi_s"] = time.perf_counter() - t_all
                t1 = time.perf_counter()
                eng.add_prompt(0, iso.Prompt(f, tuple((float(p["x"]), float(p["y"])) for p in pts),
                                             tuple(int(p["label"]) for p in pts), tuple(float(v) for v in box) if box else None))
                tt["prompt_s"] = time.perf_counter() - t1
                t1, c1 = time.perf_counter(), time.process_time()
                logits, score = eng.run_frame(0)
                tt["infer_s"], tt["infer_cpu_s"] = time.perf_counter() - t1, time.process_time() - c1
                eng.release()
                t1 = time.perf_counter()
                m = iso.target_mask(logits, thr)
                n_comp, big = 0, 0.0
                if m.any():
                    n_lab, _lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
                    n_comp, big = n_lab - 1, float(st[1:, cv2.CC_STAT_AREA].max() / m.sum())
                tt["masker_s"] = time.perf_counter() - t1
                t1 = time.perf_counter()
                bgr = cv2.imdecode(np.fromfile(fpath, np.uint8), cv2.IMREAD_COLOR)
                sec = f"{(start + f) / fps:.2f}".replace(".", ",")
                img = render(bgr, m, pts, box, f"frame {f} (detik {sec})   skor objek {score:+.2f}   masker {m.mean() * 100:.1f}% frame")
                ok, buf = cv2.imencode(".png", img)
                if not ok:
                    raise SystemExit("imencode gagal")
                name = f"klik_f{f:05d}_t{t['id']}_p{rnd + 1}.png"
                (out_dir / name).write_bytes(buf.tobytes())
                tt["render_s"] = time.perf_counter() - t1
                tt["total_s"] = time.perf_counter() - t_all
                same = None
                if rnd == 0:
                    masks0[(f, t["id"])] = m
                else:
                    same = bool(np.array_equal(masks0[(f, t["id"])], m))
                rows.append({"frame": f, "target": t["id"], "kiri": sum(p["label"] == 1 for p in pts),
                             "kanan": sum(p["label"] == 0 for p in pts), "kotak": box is not None,
                             "skor_objek": round(score, 3), "luas_px": int(m.sum()), "porsi_frame": round(float(m.mean()), 4),
                             "komponen_8arah": int(n_comp), "porsi_komponen_terbesar": round(big, 4),
                             "logit_maks": round(float(np.max(logits)), 3), "png": name, "png_ukuran": [int(img.shape[1]), int(img.shape[0])],
                             "masker_sama_putaran1": same, **{a: round(b, 3) for a, b in tt.items()}})
                r = rows[-1]
                print(f"[{rnd + 1}] {k} f{f} t{t['id']} ({r['kiri']} kiri / {r['kanan']} kanan / kotak {int(r['kotak'])}): skor "
                      f"{r['skor_objek']:+.3f}, luas {r['luas_px']} px ({r['porsi_frame'] * 100:.1f}%), {r['komponen_8arah']} komp "
                      f"(terbesar {r['porsi_komponen_terbesar'] * 100:.1f}%); infer {r['infer_s']:.2f} s, total {r['total_s']:.2f} s",
                      flush=True)
        res["putaran"].append({"putaran": rnd + 1, "frame": rows, "median_total_s": round(statistics.median(r["total_s"] for r in rows), 3),
                               "median_infer_s": round(statistics.median(r["infer_s"] for r in rows), 3), "mem_mib": mem_mib()})
    be.close()
    res["cuda_initialized"] = bool(torch.cuda.is_initialized())
    res["mem_akhir_mib"] = mem_mib()
    res["proses_s"] = round(time.perf_counter() - t_proc, 2)
    try:
        res["produksi_klik"] = production_klik(k)
    except (OSError, ValueError, KeyError) as e:
        res["produksi_klik"] = {"galat": str(e)}
    if k.startswith("klip5"):
        res["t608_pratinjau"] = {str(f): v for f, v in T608_PREVIEW.items()}
    jsave(OUT / f"ukur_{k}.json", res)
    print(f"{k}: muat model describe {res['describe_s']} s + open {res['open_s']} s; median total p1 "
          f"{res['putaran'][0]['median_total_s']} s; RAM puncak working set {res['mem_akhir_mib']['ws_peak']} MiB; "
          f"cuda_available {res['cuda_available']}; cuda_initialized {res['cuda_initialized']}")
    if res["cuda_initialized"]:
        raise SystemExit("CUDA terinisialisasi — tidak boleh")


# ── putusan ────────────────────────────────────────
def cmd_putusan() -> None:
    data = {k: json.loads((OUT / f"ukur_{k}.json").read_text(encoding="utf-8")) for k in KLIPS}
    p1 = [r for d in data.values() for r in d["putaran"][0]["frame"]]
    med = statistics.median(r["total_s"] for r in p1)
    med_inf = statistics.median(r["infer_s"] for r in p1)
    peak = max(d["mem_akhir_mib"]["ws_peak"] for d in data.values())
    no_cuda = all(not d["cuda_available"] and not d["cuda_initialized"] and d["cuda_visible_devices"] == "-1" for d in data.values())
    rule = {"median_total_s_p1": round(med, 3), "median_infer_s_p1": round(med_inf, 3), "n_frame_klik": len(p1),
            "puncak_ram_ws_mib": peak, "tanpa_cuda": no_cuda,
            "lolos_waktu": med <= EXIT_RULE["median_s_max"], "lolos_ram": peak <= EXIT_RULE["peak_ram_mib_max"],
            "aturan": EXIT_RULE}
    rule["bagian_b"] = "DIPERTAHANKAN" if rule["lolos_waktu"] and rule["lolos_ram"] and no_cuda else "DIBUANG"
    cmp = {}
    for k, d in data.items():
        prod = d.get("produksi_klik", {})
        for r in d["putaran"][0]["frame"]:
            pk = (prod.get("klik") or {}).get(str(r["frame"]), {}).get(str(r["target"]))
            t6 = (d.get("t608_pratinjau") or {}).get(str(r["frame"]))
            cmp[f"{k} f{r['frame']} t{r['target']}"] = {
                "cpu_skor": r["skor_objek"], "cpu_luas": r["luas_px"],
                "produksi_gpu_skor": pk.get("score_logit") if pk else None, "produksi_gpu_luas": pk.get("area_px") if pk else None,
                "produksi_kunci_points_sama": prod.get("kunci_points_sama"),
                "t608_skor": t6["skor"] if t6 else None, "t608_luas": t6["luas"] if t6 else None,
                "t608_komponen": t6["komponen"] if t6 else None, "cpu_komponen": r["komponen_8arah"],
                "masker_p2_sama": next(x["masker_sama_putaran1"] for x in d["putaran"][1]["frame"]
                                       if x["frame"] == r["frame"] and x["target"] == r["target"])}
    res = {"aturan_keluar": rule, "pembanding": cmp}
    for name, v in rule.items():
        print(f"{name}: {v}")
    for name, v in cmp.items():
        print(f"{name}: {v}")
    jsave(OUT / "ukur_putusan.json", res)


def main(argv: list[str]) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if argv == ["siapkan"]:
        cmd_siapkan()
    elif len(argv) == 2 and argv[0] == "ukur" and argv[1] in KLIPS:
        cmd_ukur(argv[1])
    elif argv == ["putusan"]:
        cmd_putusan()
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
