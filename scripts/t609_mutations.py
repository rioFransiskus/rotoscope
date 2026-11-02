"""T-609 mutation check (alat, F3): kode produksi di disk TIDAK diubah — tiap mutasi = penggantian teks pada salinan sumber
`rotoscope/{config,pick,isolate,mask_preview,cli}.py` DI MEMORI, di-exec ke modul yang sudah ter-import (urutan dependensi), dalam
satu proses anak per mutasi (pola scripts/t604_mutations.py). Tiap mutasi harus membuat minimal satu test di TESTS GAGAL;
kontrol (sumber asli SEMUA modul di atas di-exec dengan cara yang sama) harus LOLOS.

    python scripts/t609_mutations.py --chunk I/N   -> potongan ke-I dari N (kontrol selalu di potongan 1) -> work/t609/mutations_I.json
    python scripts/t609_mutations.py --gabung      -> gabung semua potongan -> work/t609/mutations.json + ringkasan
    python scripts/t609_mutations.py "<nama>"      -> satu mutasi (dipakai proses induk)
    python scripts/t609_mutations.py --pilih "<a>" "<b>" ... --simpan LABEL  -> mutasi tertentu -> work/t609/mutations_LABEL.json
Proses anak: CUDA_VISIBLE_DEVICES=-1.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))
sys.path.insert(0, str(REPO / "src"))

TESTS = ["tests/test_pick.py", "tests/test_isolate.py", "tests/test_mask_preview.py", "tests/test_config.py"]
OUT = REPO / "work" / "t609"
G, P, I, M, C = "config", "pick", "isolate", "mask_preview", "cli"
ORDER = (G, P, I, M, C)                                  # urutan exec = urutan dependensi

CONTROL = "kontrol (sumber asli semua modul di-exec; harus LOLOS)"
MUTATIONS: dict[str, list[tuple[str, str, str]]] = {
    CONTROL: [],
    # ── (a) peringatan klik kanan di pick ──
    "ambang >= (10 ikut diperingatkan)": [
        (P, '        if n > limit:\n            out.append((int(t["id"]), n))', '        if n >= limit:\n            out.append((int(t["id"]), n))')],
    "hitungan per frame (jumlah semua target)": [
        (P, '        n = sum(p["label"] == 0 for p in t["points"])\n        if n > limit:',
            '        n = sum(p["label"] == 0 for tt in targets for p in tt["points"])\n        if n > limit:')],
    "menghitung klik kiri, bukan kanan": [
        (P, '        n = sum(p["label"] == 0 for p in t["points"])\n        if n > limit:',
            '        n = sum(p["label"] == 1 for p in t["points"])\n        if n > limit:')],
    "tanda (>N) di baris panel hilang": [
        (P, 'f"  (>{info.neg_max})" if tid in over else ""', '""')],
    "blok kuning panel hilang": [
        (P, '    if over:\n        warn = (f"(>{info.neg_max})', '    if False:\n        warn = (f"(>{info.neg_max})')],
    "panel memakai default kode, bukan nilai YAML": [
        (P, "over = dict(state.over_negative(info.neg_max))", "over = dict(state.over_negative(DEFAULT_NEG_WARN_MAX))")],
    "peringatan terminal sesudah simpan hilang": [
        (P, "    if over:\n        print(over_negative_warning(", "    if False:\n        print(over_negative_warning(")],
    "simpan diblokir bila klik kanan berlebih": [
        (P, "        bad = self.negative_only()\n",
            "        if self.over_negative(DEFAULT_NEG_WARN_MAX):\n            return None\n        bad = self.negative_only()\n")],
    "frame lain dengan klik kanan berlebih tidak dicantumkan": [
        (P, "    if over_others:\n", "    if False:\n")],
    "teks saran lama (3-5 klik kiri)": [
        (P, '        tips = (f"Saran: kotak di sekitar target + 4-8 klik kiri + klik kanan paling banyak {info.neg_max}, hanya di "\n'
            '                f"bagian orang lain yang menempel target.")',
            '        tips = "Saran: 3-5 klik kiri tersebar per target + klik kanan di orang / benda yang menempel ke target."')],
    # ── (b6-i) pick menghapus target/pratinjau/ ──
    "pick tidak menghapus pratinjau saat simpan": [
        (P, "        removed = remove_mask_preview(session.work_dir)\n", "        removed = False\n")],
    "pick menghapus folder lain (pick_dryrun/) alih-alih pratinjau/": [
        (P, "    d = work_dir / TARGET_DIRNAME / PREVIEW_DIRNAME\n", "    d = work_dir / TARGET_DIRNAME / DRYRUN_DIRNAME\n")],
    "pratinjau dihapus juga saat Esc (titik tidak berubah)": [
        (P, '            if act == ACT_EXIT:\n                print("Dibatalkan',
            '            if act == ACT_EXIT:\n                remove_mask_preview(session.work_dir)\n                print("Dibatalkan')],
    # ── (a) isolate + (c) pesan ──
    "isolate --dry-run tanpa peringatan": [
        (I, '        if over:\n            log(neg_points_warning(clip, over, cfg.isolate.neg_points_warn_max))\n        log("  dry-run:',
            '        if False:\n            log(neg_points_warning(clip, over, cfg.isolate.neg_points_warn_max))\n        log("  dry-run:')],
    "isolate --dry-run up-to-date tanpa peringatan": [
        (I, "        if dry_run and over:", "        if False:")],
    "isolate run biasa up-to-date ikut memperingatkan": [
        (I, "        if dry_run and over:", "        if over:")],
    "isolate sebelum GPU tanpa peringatan": [
        (I, "    if over and jobs:", "    if False:")],
    "pesan butuh GPU menyarankan lagi run --yes": [
        (I, "dengan --yes: {stage_cmd(clip.work_dir)} --yes. Rencana tanpa GPU:",
            "dengan --yes: {stage_cmd(clip.work_dir)} --yes (atau {cli_cmd('run', clip.work_dir)} --yes). Rencana tanpa GPU:")],
    # ── key YAML isolate.neg_points_warn_max ──
    "isolate mengabaikan YAML (ambang tetap default)": [
        (I, "    over = pick.doc_over_negative(doc, cfg.isolate.neg_points_warn_max)",
            "    over = pick.doc_over_negative(doc, pick.DEFAULT_NEG_WARN_MAX)")],
    "key YAML masuk kunci basi [1b]": [
        (I, '            "params": {"mask_threshold": i.mask_threshold, "cut_diff": cfg.stabilize.temporal.cut_diff},',
            '            "params": {"mask_threshold": i.mask_threshold, "cut_diff": cfg.stabilize.temporal.cut_diff, '
            '"neg_points_warn_max": i.neg_points_warn_max},')],
    "cli pick mengabaikan YAML (ambang default)": [
        (C, "neg_max=cfg.isolate.neg_points_warn_max)", "neg_max=pick_mod.DEFAULT_NEG_WARN_MAX)")],
    "pratinjau mengabaikan YAML (ambang default)": [
        (M, "    limit = cfg.isolate.neg_points_warn_max\n", "    limit = pick.DEFAULT_NEG_WARN_MAX\n")],
    "validasi key YAML dibuang": [
        (G, '    _at_least("isolate.neg_points_warn_max", i.neg_points_warn_max, 0)\n', "")],
    "default kode key YAML berubah (11)": [
        (G, "    neg_points_warn_max: int = 10\n", "    neg_points_warn_max: int = 11\n")],
    # ── (b) pratinjau CPU ──
    "cli pratinjau tanpa env CPU": [
        (C, "    env = {**os.environ, MASK_PREVIEW_CPU_ENV[0]: MASK_PREVIEW_CPU_ENV[1]}\n", "    env = None\n")],
    "cli pratinjau menerima --frame": [
        (C, "    if a.dry_run or a.frame is not None:\n", "    if a.dry_run:\n")],
    # ── helper proses anak pratinjau sendiri (tanggapan Rio no. 1: run_gpu_stage tidak disentuh) ──
    "pratinjau Ctrl+C: anak tidak diberi waktu selesai sendiri": [
        (C, "        if isinstance(exc, KeyboardInterrupt):\n            try:\n                proc.wait(timeout=TERMINATE_TIMEOUT_S)\n",
            "        if isinstance(exc, KeyboardInterrupt):\n            try:\n                _stop(proc)\n")],
    "pratinjau Ctrl+C: anak yang tidak selesai tidak dihentikan": [
        (C, "            except subprocess.TimeoutExpired:\n                _stop(proc)\n        else:\n            _stop(proc)\n"
            "        if isinstance(exc, KeyboardInterrupt) and proc.returncode != EXIT_INTERRUPTED:\n"
            "            print(interrupted_line(\"pratinjau",
            "            except subprocess.TimeoutExpired:\n                pass\n        else:\n            _stop(proc)\n"
            "        if isinstance(exc, KeyboardInterrupt) and proc.returncode != EXIT_INTERRUPTED:\n"
            "            print(interrupted_line(\"pratinjau")],
    "pratinjau: galat lain tidak menghentikan anak": [
        (C, "        else:\n            _stop(proc)\n        if isinstance(exc, KeyboardInterrupt) and proc.returncode != EXIT_INTERRUPTED:\n"
            "            print(interrupted_line(\"pratinjau",
            "        else:\n            pass\n        if isinstance(exc, KeyboardInterrupt) and proc.returncode != EXIT_INTERRUPTED:\n"
            "            print(interrupted_line(\"pratinjau")],
    "pratinjau Ctrl+C: baris dihentikan tidak dicetak saat anak dimatikan": [
        (C, '            print(interrupted_line("pratinjau masker"), file=sys.stderr, flush=True)\n', "            pass\n")],
    "pratinjau Ctrl+C: cli tidak menangkap (dua baris + ERROR)": [
        (C, "        return run_mask_preview(args)\n    except KeyboardInterrupt:", "        return run_mask_preview(args)\n    except ZeroDivisionError:")],
    "penjaga env CPU dibuang": [
        (M, "    if cur != CUDA_OFF:\n        raise StageError(", "    if False:\n        raise StageError(")],
    "device cuda": [
        (M, 'DEVICE = "cpu"\n', 'DEVICE = "cuda"\n')],
    "penjaga cuda_initialized dibuang": [
        (M, "    if cuda_probe():\n", "    if False:\n")],
    "pratinjau ditulis ke isolate/": [
        (M, "    target_dir = work_dir / pick.TARGET_DIRNAME\n", "    target_dir = work_dir / iso.ISOLATE_DIRNAME\n")],
    "tanpa tukar folder (PNG basi tertinggal)": [
        (M, "    if final.exists():\n        iso._replace_dir(final, old)\n    iso._replace_dir(partial, final)\n",
            "    shutil.copytree(partial, final, dirs_exist_ok=True)\n    shutil.rmtree(partial)\n")],
    "komponen 4-arah": [
        (M, "connectivity=8", "connectivity=4")],
    "porsi komponen terbesar terhadap luas frame": [
        (M, "float(st[1:, cv2.CC_STAT_AREA].max() / mask.sum())", "float(st[1:, cv2.CC_STAT_AREA].max() / mask.size)")],
    "prompt target pertama dipakai untuk semua target": [
        (M, "eng.add_prompt(0, prompt_of(f, t))", 'eng.add_prompt(0, prompt_of(f, e["targets"][0]))')],
    "obj_id sesi selalu 1": [
        (M, "eng = backend.session([fpath], size, tid)", "eng = backend.session([fpath], size, 1)")],
    "detik tanpa awal trim": [
        (M, "            sec = (start + f) / fps\n", "            sec = f / fps\n")],
    "peringatan masker kosong hilang": [
        (M, "if not np.isfinite(score) or score <= 0 or area == 0:", "if False:")],
    "status frame klik tidak dicek": [
        (M, "    doc = iso.load_valid_points(clip, {})\n", "    doc = pick.load_points(clip.points_path)\n")],
    "cek RAM dilewati": [
        (M, "    iso.check_ram(ram_free, cfg.isolate.ram_min_free_mib)\n", "    pass\n")],
}


def apply(name: str) -> None:
    from rotoscope import cli, config, isolate, mask_preview, pick

    mods = {G: config, P: pick, I: isolate, M: mask_preview, C: cli}
    by_mod: dict[str, list[tuple[str, str]]] = {}
    for m, old, new in MUTATIONS[name]:
        by_mod.setdefault(m, []).append((old, new))
    for m in (ORDER if not by_mod else [x for x in ORDER if x in by_mod]):    # kontrol: semua modul asli di-exec
        mod = mods[m]
        src = Path(mod.__file__).read_text(encoding="utf-8")
        for old, new in by_mod.get(m, []):
            n = src.count(old)
            if n != 1:
                raise SystemExit(f"mutasi {name!r}: teks lama muncul {n}x di {m}.py (harus 1x): {old[:60]!r}")
            src = src.replace(old, new)
        exec(compile(src, mod.__file__, "exec"), mod.__dict__)   # noqa: S102 — mutasi hidup di memori proses ini saja


def run_one(name: str) -> dict:
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1"}
    p = subprocess.run([sys.executable, __file__, name], cwd=REPO, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env=env)
    lines = p.stdout.splitlines()
    tail = [x for x in lines if re.search(r"\d+ (passed|failed)", x)][-1:]
    failed = [x.split("::", 1)[1].split(" ")[0] for x in lines if x.startswith("FAILED ")]
    why = [x for x in lines if re.match(r"^[A-Za-z]:\\.*:\d+: ", x)]          # --tb=line: "<path>:<baris>: <galat>"
    return {"rc": p.returncode, "ringkas": tail, "test_gagal": p.returncode != 0, "n_gagal": len(failed),
            "contoh": failed[:4], "gagal": failed, "alasan": why[:8],
            "stderr": p.stderr.strip().splitlines()[-3:] if p.returncode not in (0, 1) else []}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = sys.argv[1:]
    if args and args[0] == "--gabung":
        parts = sorted(OUT.glob("mutations_*.json"))
        out: dict = {}
        for p in parts:
            out.update(json.loads(p.read_text(encoding="utf-8")))
        missing = [n for n in MUTATIONS if n not in out]
        (OUT / "mutations.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
        muts = {k: v for k, v in out.items() if k != CONTROL}
        killed = sum(v["rc"] == 1 and v["n_gagal"] > 0 for v in muts.values())
        print(f"potongan {len(parts)}; belum dijalankan {missing}; kontrol lolos: {out.get(CONTROL, {}).get('rc') == 0}; "
              f"{killed}/{len(muts)} mutasi membuat test gagal -> {OUT / 'mutations.json'}")
        for k, v in muts.items():
            if not (v["rc"] == 1 and v["n_gagal"] > 0):
                print(f"  LOLOS / galat: {k}: rc {v['rc']} {v['ringkas']} {v['stderr']}")
        return
    if args and args[0] == "--pilih" and "--simpan" in args:
        k = args.index("--simpan")
        names, label = args[1:k], args[k + 1]
        bad = [n for n in names if n not in MUTATIONS]
        if bad or not re.fullmatch(r"[A-Za-z0-9]+", label):
            raise SystemExit(f"nama mutasi tidak dikenal {bad} / label {label!r}")
        res = {}
        for name in names:
            res[name] = run_one(name)
            v = res[name]
            print(f"{name}: rc {v['rc']}, {v['ringkas'][0] if v['ringkas'] else '-'}; gagal {v['gagal'][:3]}", flush=True)
        out = OUT / f"mutations_{label}.json"
        if out.exists():
            raise SystemExit(f"{out} sudah ada — tidak ditimpa")
        out.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
        return
    if args and args[0] == "--chunk":
        i, n = (int(x) for x in args[1].split("/"))
        names = list(MUTATIONS)
        sel = [x for k, x in enumerate(names) if k % n == i - 1]
        res = {}
        for name in sel:
            res[name] = run_one(name)
            v = res[name]
            print(f"{name}: rc {v['rc']}, {v['ringkas'][0] if v['ringkas'] else '-'}; contoh {v['contoh'][:2]}", flush=True)
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"mutations_{i}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
        return
    if len(args) == 1 and args[0] in MUTATIONS:
        import pytest

        apply(args[0])
        sys.exit(int(pytest.main([*TESTS, "-q", "-p", "no:cacheprovider", "--tb=line", "-rf"])))
    print(__doc__)
    sys.exit(2)


if __name__ == "__main__":
    main()
