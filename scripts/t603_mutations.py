"""T-603 mutation check (alat, F3): kode produksi di disk TIDAK diubah. Tiap mutasi = penggantian teks pada SALINAN `src/`
di work/t603/mut/src/ (di-ignore git); `PYTHONPATH` diarahkan ke salinan itu, lalu pytest pada berkas test mutasi harus GAGAL.
Kontrol (salinan tanpa perubahan; union semua berkas test) harus LOLOS. Hasil: work/t603/mutations.json.

    python scripts/t603_mutations.py            -> kontrol + semua mutasi, berurutan (satu proses pytest per mutasi)
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "work" / "t603" / "mutations.json"
MUT_SRC = REPO / "work" / "t603" / "mut" / "src"
BATCH, CLI = "batch.py", "cli.py"
SEG, DEP, ISO, SC = "segment.py", "depth.py", "isolate.py", "stage_common.py"
TB, TI, TC = "tests/test_batch.py", "tests/test_interrupt.py", "tests/test_cli.py"

# (nama, berkas sumber, teks lama (harus tepat 1×), teks baru, butir, berkas test)
MUTANTS: list[tuple] = [
    ("A1 identitas basi tidak diperiksa (seg)", BATCH,
     'elif sm is not None and sm.get("clip") != ident:', "elif False:", "Tahap 1 (identitas)", [TB]),
    ("A2 rencana: perlu tindakan tidak → exit 4", BATCH,
     'if any(r.status == "perlu tindakan" for r in rows):\n        return EXIT_ACTION\n    return EXIT_OK',
     "return EXIT_OK", "Tahap 1 (exit rencana)", [TB]),
    ("A3 --skip-pending diabaikan", BATCH,
     'if pending and not args.skip_pending:', 'if pending:', "Tahap 1 (exit jalan / skip)", [TB]),
    ("A4 perintah anak tanpa --yes", BATCH, 'return a + ["--yes"]', "return a", "Tahap 1 (--yes ke GPU)", [TB]),
    ("A5 klip gagal menghentikan batch", BATCH,
     'row.status = "selesai" if rc == EXIT_OK else "gagal"',
     'row.status = "selesai" if rc == EXIT_OK else "gagal"\n        if rc != EXIT_OK:\n            return EXIT_PRECONDITION',
     "Tahap 1 (K4)", [TB]),
    ("A6 Ctrl+C tidak menghentikan anak", BATCH, "cli_mod._stop(proc)", "pass", "Tahap 1 (Ctrl+C)", [TB]),
    ("A7 duplikat klip tidak ditolak", BATCH, "if key in seen:", "if False:", "Tahap 1 (duplikat)", [TB]),
    ("A8 flag terlarang lolos", BATCH, 'if tok.split("=", 1)[0] in FORBIDDEN_FLAGS:',
     "if False:", "Tahap 1 (flag dilarang)", [TB]),
    ("A9 gagal + dilewati tidak → exit 1", BATCH,
     'if any(r.status == "gagal" for r in rows):\n        return EXIT_PRECONDITION\n    if any(r.status == "dilewati"',
     'if any(r.status == "gagal" for r in rows) and False:\n        return EXIT_PRECONDITION\n    if any(r.status == "dilewati"',
     "Tahap 1 (exit jalan)", [TB]),
    ("A10 dilewati tidak → exit 4", BATCH, 'if any(r.status == "dilewati" for r in rows):\n        return EXIT_ACTION',
     'if False:\n        return EXIT_ACTION', "Tahap 1 (exit jalan)", [TB]),
    ("A11 disk tidak diperiksa", BATCH, "    return need[1] >= need[0] + need[2]", "    return True",
     "Tahap 1 (disk)", [TB]),
    ("E1 disk per MP tidak dipakai (per frame tetap)", BATCH,
     'row.disk_mib = float(cfg.batch.disk_mib_per_mpx) * (row.wh[0] * row.wh[1] / 1e6) * row.frames',
     "row.disk_mib = 8.5 * row.frames", "Tahap 2–3 (E)", [TB]),
    ("F1 kutip ganda luar tidak dilepas", BATCH, "        if len(tok) >= 2 and tok.endswith('\"'):\n            tok = tok[1:-1]",
     "        if len(tok) >= 2 and tok.endswith('\"'):\n            tok = tok", "F (kutip)", [TB]),
    ("F2 backslash diperlakukan sebagai escape", BATCH, "shlex.split(line, posix=False)", "shlex.split(line)", "F (backslash)", [TB]),
    ("G1 rencana memanggil isolate tanpa titik + hapus", BATCH,
     'row.target = "tidak ada"\n        return 0.0, False',
     'row.target = "tidak ada"\n        isolate_stage.skip_without_points(row.work_dir, dry_run=False)\n        return 0.0, False',
     "G (jalur hapus)", [TB]),
    ("G2 dry_run=True diganti False", BATCH,
     "run_isolate(clip_cfg, dry_run=True, log=lambda m: None)", "run_isolate(clip_cfg, dry_run=False, log=lambda m: None)",
     "G (dry-run)", [TB]),
    ("G3 regresi Tahap 2: isolate memakai config root, bukan folder klip", BATCH,
     "run_isolate(clip_cfg, dry_run=True, log=lambda m: None)", "run_isolate(cfg, dry_run=True, log=lambda m: None)",
     "G (work_dir klip)", [TB]),
    ("H1 ambang VRAM di-hardcode (2000)", BATCH,
     "segment_stage.check_vram(free, total, cfg.segment.vram_min_free_mib[cfg.segment.model], cfg.segment.model)",
     "segment_stage.check_vram(free, total, 2000, cfg.segment.model)", "H (ambang sama)", [TB]),
    ("H2 VRAM tidak dicek sebelum GPU", BATCH,
     "            if any(r.status == \"siap\" and r.gpu_s > 0 for r in rows):\n                check_vram(cfg)",
     "            if False:\n                check_vram(cfg)", "H (sekali sebelum klip)", [TB]),
    ("J1 ringkasan tidak dicetak (exit 4 / akhir jalan)", BATCH,
     "    print(render_summary(rows, code, path), flush=True)", "    pass", "J (ringkasan dicetak)", [TB]),
    ("J2 ringkasan JSON tidak ditulis saat exit 4", BATCH,
     "    path = write_summary(cfg, rows, code, started)", "    path = None", "J (JSON)", [TB]),
    ("K1 cli: batch tidak di-dispatch sebelum argparse", CLI,
     'if raw[:1] == ["batch"]:', 'if False:', "dispatch (flag dilarang → exit 1)", [TB]),
    # ── keputusan Rio no. 1 + 2: ringkasan + mode --yes ──
    ("B1 mode --yes mencetak baris 'Keluar'", BATCH,
     '    if code is None:\n        return "\\n".join(out)', '    if False:\n        return "\\n".join(out)', "Rio 2 (mode --yes)", [TB]),
    ("B2 GPU klip dilewati tidak '—'", BATCH,
     '        gpu = gpu_text(r.gpu_s) if counted else "—"', "        gpu = gpu_text(r.gpu_s)", "Rio 2 (kolom —)", [TB]),
    ("B3 baris 'Mulai:' hilang", BATCH,
     '            print(f"Mulai: {n_run} klip dijalankan, {n_skip} dilewati{skip_note}", flush=True)',
     "            pass", "Rio 2 (Mulai)", [TB]),
    ("B4 '(--skip-pending)' selalu dicetak (tanpa klip dilewati)", BATCH,
     'skip_note = " (--skip-pending)" if n_skip else ""', 'skip_note = " (--skip-pending)"',
     "Rio a (Mulai)", [TB]),
    # ── keputusan Rio no. 3: Ctrl+C di stage GPU ──
    ("I1 segment: posisi frame hilang (Ctrl+C tidak dikonversi)", SEG,
     "    except KeyboardInterrupt:\n        raise StageInterrupted(where) from None\n    finally:\n        close_quietly(backend)",
     "    finally:\n        close_quietly(backend)", "Rio 3 (posisi)", [TI]),
    ("I2 segment: cleanup tanpa close_quietly (traceback kedua)", SEG,
     "        close_quietly(backend)\n        append_jsonl(clip.frames_log, {\"event\": \"run_end\"",
     "        backend.close()\n        append_jsonl(clip.frames_log, {\"event\": \"run_end\"", "Rio 3 (cleanup)", [TI]),
    ("I3 segment: Ctrl+C di loop tidak dikonversi ke StageInterrupted", SEG,
     "        raise StageInterrupted(where) from None", "        raise", "Rio 3 (posisi)", [TI]),
    ("I4 depth: Ctrl+C memberi exit 1 bukan 130", DEP,
     "        print(interrupted_line(str(e)), file=sys.stderr, flush=True)\n        return EXIT_INTERRUPTED\n    except KeyboardInterrupt:                      # di luar loop frame (mis. memuat model)",
     "        print(interrupted_line(str(e)), file=sys.stderr, flush=True)\n        return EXIT_PRECONDITION\n    except KeyboardInterrupt:                      # di luar loop frame (mis. memuat model)",
     "Rio 3 (exit 130)", [TI]),
    ("I5 isolate: StageInterrupted tidak dipetakan di main", ISO,
     "    except StageInterrupted as e:                  # T-603: satu baris, tanpa traceback; isolate/ lama utuh",
     "    except ZeroDivisionError as e:                  # T-603: satu baris, tanpa traceback; isolate/ lama utuh", "Rio 3 (isolate)", [TI]),
    ("I6 close_quietly tidak menelan Ctrl+C kedua", SC,
     "    try:\n        backend.close()\n    except KeyboardInterrupt:\n        pass",
     "    try:\n        backend.close()\n    except KeyboardInterrupt:\n        raise", "Rio 3 (cleanup)", [TI]),
    ("I7 run: induk tidak mencetak saat anak dimatikan paksa", CLI,
     "        if isinstance(exc, KeyboardInterrupt) and proc.returncode != EXIT_INTERRUPTED:",
     "        if False:", "Rio 3 (satu baris)", [TI]),
    ("I8 run: baris stage GPU ganda (induk + anak)", CLI,
     '        if ctx.state.get("stage") not in GPU_STAGES:',
     '        if True:', "Rio 3 (satu baris)", [TI]),
]
# Mutan baru → berkas test diberikan per mutan; mutan lama = [TB] (ditambahkan di atas).
ALL_TESTS = [TB, TI, TC]


def run_pytest(src: Path, tests: list[str]) -> tuple[int, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(src) + os.pathsep + env.get("PYTHONPATH", "")
    r = subprocess.run([sys.executable, "-m", "pytest", *tests, "-q", "-p", "no:cacheprovider"],
                       cwd=REPO, env=env, capture_output=True, text=True, timeout=900)
    last = (r.stdout.strip().splitlines() or [""])[-1]
    return r.returncode, last


def module_path(src: Path) -> str:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(src) + os.pathsep + env.get("PYTHONPATH", "")
    r = subprocess.run([sys.executable, "-c", "import rotoscope.batch as b; print(b.__file__)"], cwd=REPO, env=env,
                       capture_output=True, text=True)
    return r.stdout.strip()


def fresh_copy(mutation: tuple | None) -> Path:
    if MUT_SRC.parent.exists():
        shutil.rmtree(MUT_SRC.parent)
    shutil.copytree(REPO / "src", MUT_SRC)
    if mutation is not None:
        _, fname, old, new = mutation[:4]
        target = MUT_SRC / "rotoscope" / fname
        text = target.read_text(encoding="utf-8")
        n = text.count(old)
        if n != 1:
            raise SystemExit(f"teks mutasi harus tepat 1×, ditemukan {n}: {old[:60]!r}")
        target.write_text(text.replace(old, new), encoding="utf-8")
    return MUT_SRC


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    results = []
    src = fresh_copy(None)
    mod = module_path(src)
    rc, last = run_pytest(src, ALL_TESTS)
    results.append({"nama": "kontrol (tanpa perubahan; harus LOLOS)", "rc": rc, "ringkasan": last, "modul": mod,
                    "terbunuh": None})
    print(f"kontrol: rc={rc} | {last} | modul: {mod}", flush=True)
    for m in MUTANTS:
        src = fresh_copy(m)
        rc, last = run_pytest(src, m[5])
        killed = rc != 0
        results.append({"nama": m[0], "butir": m[4], "berkas_test": m[5], "rc": rc, "ringkasan": last, "terbunuh": killed})
        flag = "TERBUNUH" if killed else "LOLOS (tidak terdeteksi)"
        print(f"{m[0]:<62} {flag:<26} | {last}", flush=True)
    shutil.rmtree(MUT_SRC.parent, ignore_errors=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    killers = results[1:]
    ok = results[0]["rc"] == 0 and all(r["terbunuh"] for r in killers)
    print(f"\nmutan terbunuh: {sum(r['terbunuh'] for r in killers)}/{len(killers)}; kontrol {'LOLOS' if results[0]['rc'] == 0 else 'GAGAL'}")
    print(f"-> {OUT.relative_to(REPO).as_posix()}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
