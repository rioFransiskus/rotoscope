"""T-606 Tahap 1 — prototipe M1 (alat sekali pakai, F3): frame adegan klip3 f85–f125 dibersihkan (di luar M_d → abu-abu 128),
seg 0.8B diulang di scratch (GPU), lalu [3]–[6] dengan H di atas seg baru. Dijalankan Rio dari PowerShell TERPISAH.

    python scripts\\t606_m1.py --dry-run     rencana + validasi TANPA GPU; tidak menulis / menghapus apa pun
    python scripts\\t606_m1.py               jalankan / LANJUTKAN semua langkah (GPU hanya di langkah `seg`)
    python scripts\\t606_m1.py uji           uji alur TANPA GPU di folder uji terpisah (work/t606_scratch/uji_M1*): seg ASLI
                                             dipakai sebagai pengganti hasil GPU → hasil [3]–[5] harus byte-identik dengan M2-H

Langkah (penanda selesai + sidik masukan di work/t606_scratch/M1_steps.json; langkah selesai dengan sidik sama dilewati):
  1. siapkan (CPU)  work/t606_scratch/M1_seg/klip3/: meta.json (41 frame, mulai f85) + frames/ f85–f125 dengan piksel di luar M_d
                    (masker [1b] + isi lubang + lebar d, parameter usulan t606_tahap1.PROPOSED) = (128, 128, 128)
  2. seg (GPU)      python -m rotoscope.segment --work-dir <M1_seg> (0.8B fp16 dari configs/default.yaml; stage [2] sendiri
                    mengecek VRAM bebas ≥ 3300 MiB dan melanjutkan per frame bila terputus); nvidia-smi disampel tiap 2 s
  3. rakit (CPU)    work/t606_scratch/M1/klip3/ = klip3 PENUH: seg f85–f125 = seg M1 dibatasi H, frame lain + depth ASLI
                    (hard link), qc_report.json asli
  4. render (CPU)   [3] stabilize + [4] vectorize (klip penuh) → [5] stylize + [6] export (jendela f85–f125, out_dir scratch)
  5. bahan (CPU)    metrik V0 / M2-H / M1 → work/t606/metrik.json; video foto | V0 | M2-H | M1 → work/t606/nilai/
Terputus (Ctrl+C / listrik): jalankan perintah yang sama lagi. Hanya MEMBACA work/clips/klip3/; menulis ke work/t606_scratch/ +
work/t606/.
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
sys.path.insert(0, str(ROOT / "scripts"))

from rotoscope import segment as seg  # noqa: E402
from rotoscope.config import load_class_names  # noqa: E402
from rotoscope.ingest import FRAME_PATTERN  # noqa: E402
from rotoscope.stage_common import clip_identity, set_offline, write_json_atomic  # noqa: E402

import t606_h as hh  # noqa: E402
import t606_tahap1 as t1  # noqa: E402

GRAY = (128, 128, 128)
EST_SEG_S_PER_FRAME = 16.5            # seg 0.8B fp16 terukur T-602 (klip3 M1 p50 16,53 s; frame pertama ±21 s)
EST_LOAD_S = 90.0                     # muat model + frame pertama (perkiraan)
VRAM_NEED_MIB = 3300                  # = segment.vram_min_free_mib["0.8b"] (configs/default.yaml); dicek juga oleh [2]
VRAM_SAMPLE_S = 2.0
MIN_DISK_FREE_GIB = 2.0
STEPS = ("siapkan", "seg", "rakit", "render", "bahan")


class Layout:
    def __init__(self, uji: bool) -> None:
        self.uji = uji
        self.var = "uji_M1" if uji else "M1"
        self.seg_dir = t1.SCR / ("uji_M1_seg" if uji else "M1_seg") / t1.CLIP
        self.steps_json = t1.SCR / ("uji_M1_steps.json" if uji else "M1_steps.json")
        self.log_path = (t1.SCR / "logs" / "uji_M1_run.log") if uji else (t1.OUT / "m1_run.log")


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def short(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def nvsmi() -> tuple[float, float] | None:
    """(MiB bebas, total) dari nvidia-smi; None bila tidak ada / gagal (tanpa torch)."""
    if shutil.which("nvidia-smi") is None:
        return None
    r = subprocess.run(["nvidia-smi", "--query-gpu=memory.free,memory.total", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True)
    try:
        free, total = (float(v) for v in r.stdout.strip().splitlines()[0].split(","))
        return free, total
    except (ValueError, IndexError):
        return None


def nvsmi_used() -> float | None:
    v = nvsmi()
    return None if v is None else v[1] - v[0]


# ── sidik per langkah ──────────────────────────────
def fingerprints(ms: dict, state: dict) -> dict:
    p = t1.PROPOSED.to_json()
    base = {"clip": t1.CLIP, "scene": list(t1.SCENE), "isolate_key": ms["key_sha256"],
            "meta": t1.file_sha(t1.SRC / "meta.json")}
    fp = {"siapkan": short({**base, "gray": GRAY, "hole_max_px": p["hole_max_px"], "d_px": p["d_px"]})}
    fp["seg"] = short({"siapkan": fp["siapkan"], "seg_manifest_prod": t1.file_sha(t1.SRC / "seg" / "manifest.json")})
    fp["rakit"] = short({"seg": fp["seg"], "seg_finished": state.get("seg", {}).get("finished_utc"), "h": p,
                         "qc": t1.file_sha(t1.SRC / "qc_report.json")})
    fp["render"] = short({"rakit": fp["rakit"], "cfg": t1.file_sha(t1.CFG_PATH)})
    fp["bahan"] = short({"render": fp["render"], "render_finished": state.get("render", {}).get("finished_utc")})
    return fp


def load_state(lay: Layout) -> dict:
    return json.loads(lay.steps_json.read_text(encoding="utf-8")) if lay.steps_json.is_file() else {}


def save_state(lay: Layout, state: dict) -> None:
    lay.steps_json.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(lay.steps_json, state)


# ── langkah ────────────────────────────────────────
def m_d_of(ctx: t1.Ctx, i: int) -> np.ndarray:
    m = ctx.mask(i)
    m_f, _, _ = hh.fill_holes(m, t1.PROPOSED.hole_max_px)
    return hh.dilate(m_f, t1.PROPOSED.d_px)


def step_siapkan(ctx: t1.Ctx, frames: list[int], lay: Layout, log) -> dict:
    d = lay.seg_dir
    if d.exists():
        shutil.rmtree(d)                      # sidik berubah / belum selesai → seg lama di sini basi
    (d / "frames").mkdir(parents=True)
    meta = json.loads((t1.SRC / "meta.json").read_text(encoding="utf-8"))
    meta.update(frame_count=len(frames), frame_index_start=frames[0])
    meta["t606"] = {"varian": lay.var, "sumber": t1.CLIP, "adegan": [frames[0], frames[-1]], "latar": list(GRAY),
                    "h": t1.PROPOSED.to_json(), "catatan": "salinan scratch prototipe M1 T-606 (frame di luar M_d = abu-abu)"}
    write_json_atomic(d / "meta.json", meta)
    gray_frac = []
    for i in frames:
        name = FRAME_PATTERN % i
        img = cv2.imdecode(np.fromfile(t1.SRC / "frames" / name, np.uint8), cv2.IMREAD_COLOR)
        md = m_d_of(ctx, i)
        img[~md] = GRAY
        ok, buf = cv2.imencode(".png", img)
        if not ok:
            raise SystemExit(f"gagal meng-encode {name}")
        (d / "frames" / name).write_bytes(buf.tobytes())
        gray_frac.append(float((~md).mean()))
    log(f"  siapkan: {len(frames)} frame bersih → {d.relative_to(t1.ROOT).as_posix()} (latar abu-abu rata-rata "
        f"{100 * np.mean(gray_frac):.1f}% piksel)")
    return {"n_frames": len(frames), "gray_frac_mean": round(float(np.mean(gray_frac)), 4),
            "gray_frac_min_max": [round(min(gray_frac), 4), round(max(gray_frac), 4)]}


def run_sub(cmd: list[str], log, logfh) -> tuple[int, float, dict]:
    samples: list[float] = []
    stop = threading.Event()

    def sampler() -> None:
        while not stop.is_set():
            v = nvsmi_used()
            if v is not None:
                samples.append(v)
            stop.wait(VRAM_SAMPLE_S)

    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    t0 = time.perf_counter()
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    p = subprocess.Popen(cmd, cwd=t1.ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                         errors="replace", env=env)
    try:
        for line in p.stdout:
            print(line, end="", flush=True)
            logfh.write(line)
            logfh.flush()
        rc = p.wait()
    except KeyboardInterrupt:
        p.terminate()
        p.wait()
        raise
    finally:
        stop.set()
        th.join()
    return rc, time.perf_counter() - t0, {"nvidia_smi_used_first_mib": samples[0] if samples else None,
                                          "nvidia_smi_used_peak_mib": max(samples) if samples else None,
                                          "n_samples": len(samples)}


def seg_frames_valid(lay: Layout, frames: list[int]) -> list[int]:
    sc = seg.load_clip(lay.seg_dir)
    n = len(load_class_names())
    return [i for i, name in zip(sc.indices, sc.names) if not sc.frame_valid(name, n)]


def step_seg(frames: list[int], lay: Layout, log, logfh) -> dict:
    v = nvsmi()
    if v is not None and v[0] < VRAM_NEED_MIB:
        raise SystemExit(f"VRAM bebas {v[0]:.0f} MiB < {VRAM_NEED_MIB} MiB (nvidia-smi, total {v[1]:.0f}) — tutup aplikasi lain "
                         f"yang memakai GPU (browser, game, editor video), pastikan stage GPU lain tidak jalan, lalu ulangi.")
    rc, wall, gpu = run_sub([sys.executable, "-m", "rotoscope.segment", "--config", str(t1.CFG_PATH), "--work-dir",
                             str(lay.seg_dir)], log, logfh)
    rec = {"rc": rc, "wall_s": round(wall, 1), "vram_free_before_mib": v[0] if v else None, **gpu}
    if rc == 0:
        bad = seg_frames_valid(lay, frames)
        rec["invalid_frames"] = bad
        if bad:
            rec["rc"] = 1
        m = json.loads((lay.seg_dir / "seg" / "manifest.json").read_text(encoding="utf-8"))
        rec["model"] = {k: m.get(k) for k in ("model", "model_id", "revision", "precision")}
        q = lay.seg_dir / "qc_report.json"
        if q.is_file():
            rec["qc_m1_seg_only"] = json.loads(q.read_text(encoding="utf-8"))["summary"]["fail_frames"]
    return rec


def same_seg_model(lay: Layout) -> None:
    """D-009: model seg tidak pernah dicampur dalam satu klip — seg M1 harus model / revision / precision yang sama dengan
    seg produksi (manifest [2] klip penuh disalin dari produksi)."""
    a = json.loads((t1.SRC / "seg" / "manifest.json").read_text(encoding="utf-8"))
    b = json.loads((lay.seg_dir / "seg" / "manifest.json").read_text(encoding="utf-8"))
    keys = ("model", "model_id", "revision", "precision", "num_labels", "classes", "frame_size")
    diff = [k for k in keys if a.get(k) != b.get(k)]
    if diff:
        raise SystemExit(f"seg M1 ≠ seg produksi pada {diff} — model tidak boleh dicampur dalam satu klip (D-009)")


def step_rakit(ctx: t1.Ctx, frames: list[int], lay: Layout, fp: str, log) -> dict:
    same_seg_model(lay)
    sc = seg.load_clip(lay.seg_dir)
    fs = set(frames)

    def m1_seg(i: int):
        if i not in fs:
            return None
        name = FRAME_PATTERN % i
        pr, cm = seg.read_probs(sc.probs_path(name)), seg.read_classmap(sc.classmap_path(name))
        if pr is None or cm is None:
            raise SystemExit(f"seg M1 {name} tidak terbaca — jalankan langkah seg lagi")
        return pr, cm, {}

    d, prep = t1.prepare_copy(lay.var, {"m1": fp, "var": lay.var}, t1.h_seg_for(ctx, frames, t1.PROPOSED, base=m1_seg), log)
    return {"work_dir": d.relative_to(t1.ROOT).as_posix(), "frames": prep.get("frames")}


def step_bahan(lay: Layout, log) -> dict:
    if lay.uji:                               # uji: metrik.json / nilai/ tidak boleh tercampur varian uji
        log("  uji: metrik + bahan dilewati (dibandingkan langsung dengan M2-H di akhir)")
        return {"skipped": "uji"}
    if not (t1.vdir("M2H") / "strokes").is_dir():
        log("  M2-H belum ada di scratch — dirender dulu (CPU ±3 menit)")
        t1.cmd_m2(["M2H"])
    t1.cmd_metrik(["M2H", lay.var])
    t1.cmd_bahan(with_m1=True)
    return {"metrik": (t1.OUT / "metrik.json").relative_to(t1.ROOT).as_posix(), "nilai": t1.NILAI.relative_to(t1.ROOT).as_posix()}


# ── validasi + rencana ─────────────────────────────
def checks(ctx: t1.Ctx) -> tuple[list[str], list[str], dict | None, list[int]]:
    errs, notes = [], []
    ms, frames = None, []
    try:
        ms = ctx.mask_state()
        frames = t1.scene_frames(ctx, ms)
    except SystemExit as e:
        errs.append(str(e))
    missing = [i for i in range(t1.SCENE[0], t1.SCENE[1] + 1) if not (t1.SRC / "frames" / (FRAME_PATTERN % i)).is_file()]
    if missing:
        errs.append(f"{len(missing)} frame klip3 hilang (mis. f{missing[0]})")
    try:
        set_offline()
        s = ctx.cfg.segment
        seg.ensure_cached(s.model_ids[s.model], seg.resolve_revision(ctx.cfg), s.model)
        notes.append(f"model seg {s.model} ({s.model_ids[s.model]}@{s.revision[s.model][:8]}, {s.precision}) ada di cache HF")
    except Exception as e:  # noqa: BLE001 — pesan stage apa adanya
        errs.append(f"model seg: {e}")
    v = nvsmi()
    if v is None:
        notes.append("nvidia-smi tidak ditemukan / gagal — VRAM tetap dicek oleh stage [2] saat memuat model")
    else:
        notes.append(f"VRAM bebas sekarang {v[0]:.0f} / {v[1]:.0f} MiB (syarat ≥ {VRAM_NEED_MIB} MiB, dicek lagi sebelum seg)")
    free = shutil.disk_usage(t1.ROOT).free / (1 << 30)
    notes.append(f"disk bebas {free:.1f} GiB")
    if free < MIN_DISK_FREE_GIB:
        errs.append(f"disk bebas {free:.1f} GiB < {MIN_DISK_FREE_GIB} GiB")
    return errs, notes, ms, frames


def estimates(n: int) -> dict:
    return {"siapkan": "< 1 menit (CPU)",
            "seg": f"≈ {(n * EST_SEG_S_PER_FRAME + EST_LOAD_S) / 60:.0f} menit GPU ({n} frame × {EST_SEG_S_PER_FRAME} s + muat)",
            "rakit": "≈ 1 menit (CPU)", "render": "≈ 3 menit (CPU: [3] ±77 s, [4] ±70 s, [5] ±20 s, [6] ±7 s)",
            "bahan": "≈ 2 menit (CPU; + ±3 menit bila M2-H belum ada)"}


def plan_lines(lay: Layout, state: dict, fp: dict, est: dict, frames: list[int]) -> list[str]:
    """Langkah sesudah langkah yang akan jalan pasti ikut jalan (sidik berantai lewat waktu selesai)."""
    out, pending = [], False
    for s in STEPS:
        done = not pending and bool(state.get(s, {}).get("done")) and state[s].get("fingerprint") == fp[s]
        pending |= not done
        extra = ""
        if s == "seg" and not done and lay.seg_dir.is_dir() and (lay.seg_dir / "seg").is_dir() and frames:
            try:
                left = len(seg_frames_valid(lay, frames))
                extra = f" — {len(frames) - left} / {len(frames)} frame sudah ada (dilanjutkan)"
            except Exception:  # noqa: BLE001
                pass
        out.append(f"  [{'selesai' if done else 'akan jalan'}] {s}: {est[s]}{extra}")
    return out


def run(dry_run: bool, uji: bool) -> int:
    lay = Layout(uji)
    ctx = t1.Ctx()
    errs, notes, ms, frames = checks(ctx)
    state = load_state(lay)
    fp = fingerprints(ms, state) if ms else {s: "-" for s in STEPS}
    est = estimates(len(frames) or 41)
    print(f"T-606 M1{' (UJI, tanpa GPU)' if uji else ''}: klip3 f{t1.SCENE[0]}–f{t1.SCENE[1]} ({len(frames)} frame), H "
          f"{t1.PROPOSED.to_json()}, latar {GRAY}")
    for line in plan_lines(lay, state, fp, est, frames):
        print(line)
    for n in notes:
        print(f"  catatan: {n}")
    if errs:
        print("GALAT (tidak ada yang dijalankan):\n  " + "\n  ".join(errs))
        return 1
    if dry_run:
        print("--dry-run: validasi lolos; tidak ada yang dihitung / ditulis / dihapus (tanpa GPU).")
        return 0
    lay.log_path.parent.mkdir(parents=True, exist_ok=True)
    with lay.log_path.open("a", encoding="utf-8") as logfh:
        def log(msg: str) -> None:
            print(msg, flush=True)
            logfh.write(msg + "\n")
            logfh.flush()

        log(f"\n##### t606_m1 {'uji ' if uji else ''}{utc_now()} #####")
        for s in STEPS:
            fp = fingerprints(ms, state)
            if state.get(s, {}).get("done") and state[s].get("fingerprint") == fp[s]:
                log(f"[{s}] sudah selesai ({state[s].get('finished_utc')}) — dilewati")
                continue
            rec = {"done": False, "fingerprint": fp[s], "started_utc": utc_now()}
            log(f"[{s}] mulai {rec['started_utc']} — {est[s]}")
            t0 = time.perf_counter()
            if s == "siapkan":
                rec.update(step_siapkan(ctx, frames, lay, log))
            elif s == "seg":
                rec.update(fake_seg(lay, frames, log) if uji else step_seg(frames, lay, log, logfh))
            elif s == "rakit":
                rec.update(step_rakit(ctx, frames, lay, fp["rakit"], log))
            elif s == "render":
                rec["steps"] = t1.render(lay.var, log)
            else:
                rec.update(step_bahan(lay, log))
            rec.setdefault("rc", 0)
            rec["wall_s"] = rec.get("wall_s") or round(time.perf_counter() - t0, 1)
            rec["finished_utc"] = utc_now()
            rec["done"] = rec["rc"] == 0
            state[s] = rec
            save_state(lay, state)
            log(f"[{s}] {'SELESAI' if rec['done'] else 'GAGAL'} rc {rec['rc']}, {rec['wall_s']} s"
                + (f", nvidia-smi puncak {rec.get('nvidia_smi_used_peak_mib')} MiB" if s == "seg" and not uji else ""))
            if not rec["done"]:
                log("Berhenti. Perbaiki penyebabnya lalu jalankan lagi perintah yang sama (langkah selesai dilewati; seg "
                    "melanjutkan per frame).")
                return int(rec["rc"]) or 1
        if uji:
            res = uji_compare(lay)
            log(f"UJI: {res}")
            return 0 if res["ok"] else 1
        log("Semua langkah selesai. Bahan: work/t606/nilai/ (PANDUAN.md). Kabari Claude Code untuk verifikasi.")
    return 0


# ── uji tanpa GPU ──────────────────────────────────
def fake_seg(lay: Layout, frames: list[int], log) -> dict:
    """Pengganti langkah GPU di `uji`: seg ASLI frame adegan di-hard-link ke folder seg uji + manifest produksi dengan identitas
    klip uji, lalu QC [2] asli (`segment --qc-only`, CPU) memvalidasi tata letak (nama frame dari frame_index_start, ukuran)."""
    sc = seg.load_clip(t1.SRC)
    for i in frames:
        name = FRAME_PATTERN % i
        t1.link(sc.probs_path(name), lay.seg_dir / "seg" / "probs" / sc.probs_path(name).name)
        t1.link(sc.classmap_path(name), lay.seg_dir / "seg" / "classmap" / sc.classmap_path(name).name)
    m = json.loads((t1.SRC / "seg" / "manifest.json").read_text(encoding="utf-8"))
    m["clip"] = clip_identity(lay.seg_dir)
    write_json_atomic(lay.seg_dir / "seg" / "manifest.json", m)
    r = subprocess.run([sys.executable, "-m", "rotoscope.segment", "--config", str(t1.CFG_PATH), "--work-dir",
                        str(lay.seg_dir), "--qc-only"], cwd=t1.ROOT, capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    log(f"  uji seg (pengganti GPU): {len(frames)} frame seg asli; segment --qc-only rc {r.returncode}: "
        f"{(r.stdout.strip().splitlines() or ['-'])[-1]}")
    return {"rc": r.returncode, "qc_only_stdout_tail": r.stdout.strip().splitlines()[-1:] if r.stdout.strip() else [],
            "stderr_tail": r.stderr.strip().splitlines()[-3:]}


def uji_compare(lay: Layout) -> dict:
    """Seg asli + H = M2-H → stable/ (182 frame) dan strokes/ PNG + SVG f85–f125 harus byte-identik dengan M2-H."""
    a, b = t1.vdir("M2H"), t1.vdir(lay.var)
    diff = {"stable": 0, "strokes": 0}
    for p in sorted((a / "stable" / "groups").glob("*.png")) + sorted((a / "stable" / "depth_smooth").glob("*.npy")):
        q = b / p.relative_to(a)
        if not q.is_file() or t1.file_sha(p) != t1.file_sha(q):
            diff["stable"] += 1
    for i in range(t1.SCENE[0], t1.SCENE[1] + 1):
        for suf in (".png", ".svg"):
            p = a / "strokes" / (t1.stem(i) + suf)
            q = b / "strokes" / (t1.stem(i) + suf)
            if not q.is_file() or t1.file_sha(p) != t1.file_sha(q):
                diff["strokes"] += 1
    return {"vs": "M2H", "diff": diff, "ok": not any(diff.values())}


def main(argv: list[str]) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(prog="t606_m1.py", description=__doc__.splitlines()[0])
    ap.add_argument("perintah", nargs="?", choices=("uji",), help="uji = uji alur tanpa GPU")
    ap.add_argument("--dry-run", action="store_true", help="rencana + validasi tanpa GPU / tulis")
    a = ap.parse_args(argv)
    try:
        return run(a.dry_run, a.perintah == "uji")
    except KeyboardInterrupt:
        print("\nDihentikan (Ctrl+C). Jalankan perintah yang sama lagi untuk melanjutkan.", flush=True)
        return 130


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
