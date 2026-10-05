"""T-204 Tahap 3: alat ukur sekali pakai. SEMUA kerja di salinan scratchpad (work_dir / out_dir di sana); klip asli hanya
dibaca (hash sebelum / sesudah). Tidak pernah menjalankan stage GPU.

    python scripts/t204_measure.py setup   <scratch>     # salin klip + out ke <scratch>/pristine
    python scripts/t204_measure.py timing  <scratch>     # skenario (a)-(f), 3 run
    python scripts/t204_measure.py equiv   <scratch>     # kesetaraan (11a) dua arah + MP4 (11b) + keluaran utama (11d)
    python scripts/t204_measure.py guards  <scratch>     # (11c) tanpa subprocess / torch, klip lain, kombinasi flag
    python scripts/t204_measure.py originals <label>     # hash klip asli (work/clips/*, out/) -> work/t204/originals_<label>.json
Hasil: work/t204/*.json + *.log.
"""

from __future__ import annotations

import json
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))
import preview_metrics as pm  # noqa: E402

PY = sys.executable
VIDEO = REPO / "samples" / "test.mp4"
EVIDENCE = REPO / "work" / "t204"
STAGE_RE = re.compile(r"\[(?:1|3|4|5|6)\] (\w+) ([\d.]+) s")


def run_cli(cwd: Path, *args: str, timeout: int = 900) -> tuple[int, float, str]:
    t0 = time.perf_counter()
    p = subprocess.run([PY, "-m", "rotoscope", *args], cwd=cwd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=timeout)
    return p.returncode, time.perf_counter() - t0, p.stdout + p.stderr


def parts(out: str) -> dict[str, float]:
    line = next((x for x in out.splitlines() if x.startswith("Waktu per stage")), "")
    return {m.group(1): float(m.group(2)) for m in STAGE_RE.finditer(line)}


def restore(run: Path, pristine: Path, *names: str) -> None:
    for n in names:
        dst = run / n
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(pristine / n, dst)


def write_cfg(run: Path, island: int | None = None) -> Path:
    p = run / ("cfg_island.yaml" if island else "cfg.yaml")
    body = 'paths:\n  work_dir: "work"\n  out_dir: "out"\n'
    if island:
        body += f"stabilize:\n  island_min_px: {island}\n"
    p.write_text(body, encoding="utf-8")
    return p


def cmd_setup(s: Path) -> None:
    pr = s / "pristine"
    if pr.exists():
        raise SystemExit(f"{pr} sudah ada")
    (pr / "work" / "clips").mkdir(parents=True)
    for c in ("test", "test_short"):
        shutil.copytree(REPO / "work" / "clips" / c, pr / "work" / "clips" / c)
    shutil.copytree(REPO / "out", pr / "out")
    run = s / "run"
    shutil.copytree(pr, run)
    write_cfg(run)
    write_cfg(run, island=31)
    print("setup selesai:", s)


def reset_for(run: Path, pristine: Path, sid: str, k: int, n: int) -> None:
    """Kembalikan salinan ke keadaan pristine (run penuh T-203b valid), lalu buat keadaan skenario."""
    base, pbase = run / "work/clips/test", pristine / "work/clips/test"
    restore(base, pbase, "strokes", "contours", *(("stable",) if sid == "f" else ()))
    if sid in ("b10", "b20"):                                   # hanya strokes jendela yang hilang
        for i in range(k, k + n):
            for ext in ("svg", "png"):
                (base / "strokes" / f"frame_{i:05d}.{ext}").unlink(missing_ok=True)
    if sid in ("d", "d2"):                                      # contours + strokes dingin (rantai 0..K+N-1)
        for f in (base / "contours").glob("frame_*.json"):
            f.unlink()
        for f in (base / "strokes").glob("frame_*.*"):
            f.unlink()
    if sid == "d2":                                             # + clip_stats.json dan manifest ikut hilang (pass 1 dihitung)
        for rel in ("contours/clip_stats.json", "contours/manifest.json", "contours/frames.jsonl",
                    "strokes/manifest.json", "strokes/frames.jsonl"):
            (base / rel).unlink(missing_ok=True)


def cmd_timing(s: Path) -> None:
    run, pr = s / "run", s / "pristine"
    style = run / "style_alt.yaml"
    style.write_text("stroke:\n  color: '#112233'\n", encoding="utf-8")
    rows = []
    # (id, K, N, extra args, mutasi sebelum run)
    scenarios = [("a", 73, 10, []), ("b10", 73, 10, []), ("b20", 73, 20, []), ("c", 73, 10, ["--style", str(style)]),
                 ("d", 225, 10, []), ("d2", 225, 10, []), ("e", 0, 10, []), ("f", 73, 10, ["--config", "cfg_island.yaml"])]
    for sid, k, n, extra in scenarios:
        times, walls, detail = [], [], []
        for rep in range(3):
            reset_for(run, pr, sid, k, n)
            cfg = [] if "--config" in extra else ["--config", "cfg.yaml"]
            rc, wall, out = run_cli(run, "run", str(VIDEO), "--preview", str(n), "--from", str(k), *cfg, *extra)
            walls.append(wall)
            detail.append({"rc": rc, "stages": parts(out)})
            if rc != 0:
                (EVIDENCE / f"timing_{sid}_{rep}.log").write_text(out, encoding="utf-8")
        row = {"scenario": sid, "K": k, "N": n, "wall_s": [round(x, 2) for x in walls],
               "mean": round(statistics.mean(walls), 2), "min": round(min(walls), 2), "max": round(max(walls), 2),
               "stages_last": detail[-1]["stages"], "rc": [d["rc"] for d in detail]}
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    (EVIDENCE / "timing.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    restore(run / "work/clips/test", pr / "work/clips/test", "strokes", "contours", "stable")


def cmd_equiv(s: Path) -> None:
    run, pr = s / "run", s / "pristine"
    base, pbase = run / "work/clips/test", pr / "work/clips/test"
    res = {}
    windows = [(73, 10), (225, 10), (0, 10)]
    for k, n in windows:
        last = k + n - 1
        ref = pm.window_hashes(pbase, k, last)
        # arah 1: run penuh dulu (= keadaan pristine) lalu preview
        restore(base, pbase, "strokes", "contours", "stable")
        before_out = pm.out_snapshot(run / "out", "test")
        rc, _, out = run_cli(run, "run", str(VIDEO), "--preview", str(n), "--from", str(k), "--config", "cfg.yaml")
        d1 = pm.window_hashes(base, k, last)
        # arah 2: preview dulu pada keadaan dingin lalu run penuh (stage CPU berurutan; seg / depth sudah valid)
        for f in (base / "contours").glob("*"):
            f.unlink()
        shutil.rmtree(base / "strokes")
        rc2, _, out2 = run_cli(run, "run", str(VIDEO), "--preview", str(n), "--from", str(k), "--config", "cfg.yaml")
        d2a = pm.window_hashes(base, k, last)
        rc3 = [run_cli(run, st, str(VIDEO), "--config", "cfg.yaml")[0] for st in ("vectorize", "stylize")]
        d2b = pm.window_hashes(base, k, last)
        full_tree = {"contours": pm.tree_hashes(base / "contours", "frame_*.json") == pm.tree_hashes(pbase / "contours", "frame_*.json"),
                     "strokes": pm.tree_hashes(base / "strokes", "frame_*.*") == pm.tree_hashes(pbase / "strokes", "frame_*.*")}
        res[f"{k}-{last}"] = {"rc": [rc, rc2, rc3], "dir1_preview_after_full_equals_ref": d1 == ref,
                              "dir2_preview_cold_equals_ref": d2a == ref, "dir2_after_full_equals_ref": d2b == ref,
                              "full_trees_equal_pristine_after_dir2": full_tree,
                              "n_files": len(ref), "out_unchanged_dir1": pm.out_snapshot(run / "out", "test") == before_out}
        mp4 = run / "out" / f"test.preview_{k}-{last}.mp4"
        st = pm.ffprobe_stream(mp4)
        res[f"{k}-{last}"]["mp4_vs_png"] = pm.compare_to_sources(mp4, base / "strokes", k, n)
        res[f"{k}-{last}"]["mp4_vs_main"] = pm.compare_preview_to_main(mp4, run / "out" / "test.mp4", k, n,
                                                                       st["width"], st["height"])
        res[f"{k}-{last}"]["ffprobe"] = st
        print(k, last, json.dumps(res[f"{k}-{last}"], ensure_ascii=False)[:600], flush=True)
    res["out_main_unchanged_all"] = pm.out_snapshot(run / "out", "test") == pm.out_snapshot(pr / "out", "test")
    (EVIDENCE / "equivalence.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str), encoding="utf-8")


def cmd_guards(s: Path) -> None:
    run = s / "run"
    res = {}
    code = (
        "import subprocess, sys\n"
        "def boom(*a, **k): raise AssertionError('subprocess dipanggil')\n"
        "subprocess_Popen = subprocess.Popen\n"
        "from rotoscope import cli\n"
        "cli._popen = boom\n"
        f"rc = cli.main(['run', r'{VIDEO}', '--preview', '10', '--from', '73', '--config', 'cfg.yaml'])\n"
        "print('RC', rc, 'TORCH', 'torch' in sys.modules, 'TRANSFORMERS', 'transformers' in sys.modules)\n")
    p = subprocess.run([PY, "-c", code], cwd=run, capture_output=True, text=True, encoding="utf-8", errors="replace")
    res["no_gpu_subprocess_no_torch"] = [x for x in p.stdout.splitlines() if x.startswith("RC")]
    # klip lain: folder kerja test_short berisi klip test
    other = run / "work/clips/test_short"
    keep = s / "keep_test_short"
    if not keep.exists():
        shutil.move(str(other), str(keep))
    shutil.copytree(run / "work/clips/test", other, ignore=shutil.ignore_patterns("frames", "seg", "depth", "stable"))
    rc, _, out = run_cli(run, "run", str(REPO / "samples" / "test_short.mp4"), "--preview", "5", "--config", "cfg.yaml")
    res["other_clip_in_folder"] = {"rc": rc, "msg": [x for x in out.splitlines() if "LAIN" in x][:1]}
    shutil.rmtree(other)
    shutil.move(str(keep), str(other))
    combos = {"limit": ["--limit", "3"], "restart_from": ["--restart-from", "vectorize"], "from_only": None,
              "over": ["--from", "280"]}
    for name, extra in combos.items():
        args = ["run", str(VIDEO), "--config", "cfg.yaml"] + (["--preview", "10", *extra] if name != "from_only" else ["--from", "5"])
        rc, _, out = run_cli(run, *args)
        res[f"combo_{name}"] = {"rc": rc, "msg": out.strip().splitlines()[-1][:160] if out.strip() else ""}
    rc, _, out = run_cli(run, "run", str(VIDEO), "--preview", "10", "--from", "73", "--seg-model", "0.4b", "--config", "cfg.yaml")
    res["seg_model_mismatch"] = {"rc": rc, "msg": out.strip().splitlines()[-1][:300]}
    (EVIDENCE / "guards.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, indent=1, ensure_ascii=False))


def cmd_originals(label: str) -> None:
    snap = {}
    for c in ("test", "test_short"):
        d = REPO / "work" / "clips" / c
        snap[c] = {sub: pm.tree_hashes(d / sub) for sub in ("frames", "seg", "depth", "stable", "contours", "strokes")}
        snap[c]["meta"] = pm.sha256_file(d / "meta.json")
    out = REPO / "out"
    snap["out"] = {p.name: pm.sha256_file(p) for p in sorted(out.glob("*")) if p.is_file() and ".preview_" not in p.name}
    snap["out_svg"] = pm.tree_hashes(out / "svg")
    snap["_digest"] = {k: __import__("hashlib").sha256(json.dumps(v, sort_keys=True).encode()).hexdigest()
                       for k, v in snap.items()}
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    (EVIDENCE / f"originals_{label}.json").write_text(json.dumps(snap["_digest"], indent=1), encoding="utf-8")
    print(json.dumps(snap["_digest"], indent=1))


if __name__ == "__main__":
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    cmd, arg = sys.argv[1], sys.argv[2]
    {"setup": cmd_setup, "timing": cmd_timing, "equiv": cmd_equiv, "guards": cmd_guards,
     "originals": cmd_originals}[cmd](Path(arg) if cmd != "originals" else arg)
