"""T-607 mutation check (alat, F3): kode produksi di disk TIDAK diubah — tiap mutasi = penggantian teks pada salinan sumber modul
DI MEMORI, di-exec ke modul yang sudah ter-import (urutan dependensi), dalam satu proses anak per mutasi (pola
`scripts/t305f_mutations.py`). Tiap mutasi harus membuat minimal satu test di daftar test-nya GAGAL; kontrol (sumber asli semua modul
di-exec dengan cara yang sama) harus LOLOS di semua berkas test.

    python scripts/t607_mutations.py                         -> semua (proses anak per mutasi) + work/t607/mutations.json
    python scripts/t607_mutations.py --dari A --sampai B     -> entri ke-A … ke-(B − 1) (0 = kontrol), digabung ke mutations.json
    python scripts/t607_mutations.py "<nama>"                -> satu mutasi (dipakai proses induk)
"""

from __future__ import annotations

import importlib
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))
sys.path.insert(0, str(REPO / "src"))

OUT = REPO / "work" / "t607" / "mutations.json"
ORDER = ("config", "ingest", "stage_common", "crop", "export", "segment", "depth", "pick", "isolate", "stylize", "cli")   # urutan exec
CR, IN, CL, SC, EX, CF, PK, IS, SG, DP, ST = ("crop", "ingest", "cli", "stage_common", "export", "config", "pick", "isolate", "segment",
                                              "depth", "stylize")
T = {k: f"tests/test_{k}.py" for k in ("crop", "ingest", "cli", "clip_identity", "pick", "isolate", "export", "config", "preview")}
ALL = list(T.values())

CONTROL = "kontrol (sumber asli semua modul di-exec; harus LOLOS)"
KEEP = '        keep = {e for e in allowed if band[e] >= p["BAND_MIN_PX"] and band[OPP[e]] >= p["BAND_MIN_PX"]}\n'
NOBAND = "        _clear_saran(crop_dir)\n        write_json_atomic(crop_dir / DETEKSI_FILENAME, deteksi)\n"
END_B = "        b = math.ceil(e * fps - TRIM_EPS)\n"
MUTATIONS: dict[str, tuple[list[tuple[str, str, str]], list[str]]] = {
    CONTROL: ([], ALL),
    # ── R2 (salinan beku) ──
    "01 syarat pasangan dihapus": ([(CR, KEEP, "        keep = set(allowed)\n")], [T["crop"]]),
    "02 BAND_MIN_PX efektif 8 -> 7": ([(CR, KEEP, KEEP.replace('p["BAND_MIN_PX"]', 'p["BAND_MIN_PX"] - 1'))], [T["crop"]]),
    "03 kotak teks kiri / kanan diizinkan": ([(CR, 'OVERLAY_EDGES = ("top", "bottom")', "OVERLAY_EDGES = EDGES")], [T["crop"]]),
    "04 syarat seragam pita dihapus": ([(CR, "            while _line_ok(m, sd, rect, e, True, p):\n",
                                         "            while _line_ok(m, sd, rect, e, False, p):\n")], [T["crop"]]),
    "05 pengaman area aktif 10% dihapus": ([(CR, '    applied = area >= p["MIN_ACTIVE_FRAC"] * H * W\n', "    applied = area > 0\n")],
                                           [T["crop"]]),
    "06 kotak teks tanpa pita ikut dikupas": ([(CR, '        if band[e] >= p["BAND_MIN_PX"]:\n            while _line_ok(m, sd, rect, e, False',
                                                "        if True:\n            while _line_ok(m, sd, rect, e, False")], [T["crop"]]),
    "07 parameter beku diubah (sidik)": ([(CR, 'FROZEN_PARAMS = {"STATIC_STD_MAX": 3.0,', 'FROZEN_PARAMS = {"STATIC_STD_MAX": 3.5,')],
                                         [T["crop"]]),
    # ── pembulatan kotak ──
    "08 pembulatan ke dalam (ceil di awal)": ([(CR, "    xs0, ys0 = _even_floor(math.floor(x0 * sx)), _even_floor(math.floor(y0 * sy))\n",
                                                "    xs0, ys0 = _even_floor(math.ceil(x0 * sx)), _even_floor(math.ceil(y0 * sy))\n")], [T["crop"]]),
    "09 genap dihapus": ([(CR, "    xs0, ys0 = _even_floor(math.floor(x0 * sx)), _even_floor(math.floor(y0 * sy))\n",
                           "    xs0, ys0 = math.floor(x0 * sx), math.floor(y0 * sy)\n")], [T["crop"]]),
    "10 clamp tepi sumber dihapus": ([(CR, "    return [xs0, ys0, min(w + w % 2, Ws - xs0), min(h + h % 2, Hs - ys0)]\n",
                                       "    return [xs0, ys0, w + w % 2, h + h % 2]\n")], [T["crop"]]),
    "11 kotak manual dibulatkan ke dalam": ([(CR, "    x0, y0 = _even_floor(x), _even_floor(y)\n", "    x0, y0 = x + x % 2, y + y % 2\n")],
                                            [T["crop"]]),
    "12 kotak manual tanpa batas minimum": ([(CR, "    if out[2] < MANUAL_MIN_PX or out[3] < MANUAL_MIN_PX:\n", "    if False:\n")],
                                            [T["crop"]]),
    # ── identitas sumber + D1 ──
    "13 sha256 / ukuran tidak dicek": ([(CR, 'IDENTITY_KEYS = ("sha256", "bytes", "width", "height")', 'IDENTITY_KEYS = ("width", "height")')],
                                       [T["crop"]]),
    "14 path ikut dicek": ([(CR, 'IDENTITY_KEYS = ("sha256", "bytes", "width", "height")',
                             'IDENTITY_KEYS = ("sha256", "bytes", "width", "height", "path")')], [T["crop"]]),
    "15 kunci D1 tanpa sha256 sumber": ([(CR, '    return {"source": {k: identity[k] for k in IDENTITY_KEYS}, "target_fps"',
                                          '    return {"source": {k: identity[k] for k in ("width", "height")}, "target_fps"')],
                                        [T["crop"]]),
    "16 ambang petunjuk tidak ikut validitas saran": ([(CR, "                and saran.get(\"hint_static_frac\") == cfg.preview.hint_static_frac "
                                                            "and _preview_valid(crop_dir, saran)):",
                                                        "                and _preview_valid(crop_dir, saran)):")], [T["crop"]]),
    "17 tanpa pita: deteksi.json tidak ditulis": ([(CR, NOBAND, "        _clear_saran(crop_dir)\n")], [T["crop"]]),
    "18 tanpa pita: saran basi tidak dihapus": ([(CR, NOBAND, "        write_json_atomic(crop_dir / DETEKSI_FILENAME, deteksi)\n")],
                                                [T["crop"]]),
    "19 PERINGATAN aturan lain dihapus": ([(CR, "    if fp != FROZEN_FINGERPRINT:\n        log(f\"PERINGATAN", "    if False:\n        log(f\"PERINGATAN")],
                                          [T["crop"]]),
    # ── petunjuk + pratinjau ──
    "20 petunjuk juga di sisi tanpa potongan": ([(CR, '         "left": run(static[y0:y1 + 1, x] for x in range(x0, x1 + 1)) if cut["left"] else 0,',
                                                  '         "left": run(static[y0:y1 + 1, x] for x in range(x0, x1 + 1)),')], [T["crop"]]),
    "21 ambang petunjuk eksklusif (< jadi <=)": ([(CR, "            if float(line.mean()) < frac_min:\n",
                                                   "            if float(line.mean()) <= frac_min:\n")], [T["crop"]]),
    "22 kotak alternatif tanpa pengaman area": ([(CR, '    ok = alt[0] <= alt[2] and alt[1] <= alt[3] and (alt[2] - alt[0] + 1) * (alt[3] - alt[1] + 1) '
                                                      '>= p["MIN_ACTIVE_FRAC"] * H * W\n',
                                                  "    ok = alt[0] <= alt[2] and alt[1] <= alt[3]\n")], [T["crop"]]),
    "23 zoom tidak berpusat di garis potong": ([(CR, "    s = (0 if side in (\"left\", \"top\") else L - n) if cut is None else "
                                                     "min(max(cut - n // 2, 0), L - n)\n",
                                                 "    s = 0 if side in (\"left\", \"top\") else L - n\n")], [T["crop"]]),
    "24 frame tengah pratinjau salah": ([(CR, "    idx = (0, n // 2, n - 1)\n", "    idx = (0, n // 2 + 1, n - 1)\n")], [T["crop"]]),
    # ── pesan + keputusan ──
    "25 pesan saran menyebut --crop auto": ([(CR, "    out += [f\"  Terima saran: {crop_cmd(video_arg, '--accept')}\",",
                                              '    out += [f"  Terima saran: {CLI_PROG} run {video_arg} --crop auto",')], [T["crop"]]),
    "26 key crop ditulis untuk mode none": ([(CR, '    if kotak is None or kotak["mode"] == "none":\n        return None\n    out = {"mode"',
                                              '    if kotak is None:\n        return None\n    out = {"mode"')], [T["crop"]]),
    "27 key crop tanpa source_sha256": ([(CR, '           "source_sha256": kotak["source"]["sha256"]}\n', '           "source_sha256": None}\n')],
                                        [T["crop"]]),
    "28 --accept tanpa saran tidak ditolak": ([(CR, '        if saran is None:\n            raise CropError(f"crop --accept: tidak ada saran',
                                                '        if False:\n            raise CropError(f"crop --accept: tidak ada saran')], [T["crop"]]),
    "29 --accept menerima aturan selain R2 beku": ([(CR, '        if saran.get("rule_fingerprint") != FROZEN_FINGERPRINT:\n', "        if False:\n")],
                                                   [T["crop"]]),
    "30 --clear tanpa kotak.prev.json": ([(CR, "        write_bytes_atomic(crop_dir / KOTAK_PREV_FILENAME, path.read_bytes())\n        path.unlink()\n",
                                           "        path.unlink()\n")], [T["crop"]]),
    # ── ingest: trim ──
    "31 trim sebelum fps": ([(IN, "    if box:\n        vf.append(f\"crop=", "    vf.append(vf.pop(0)) if trim is not None else None\n    if box:\n"
                                                                            "        vf.append(f\"crop=")], [T["ingest"]]),
    "32 awal trim round (bukan ceil)": ([(IN, "    a = math.ceil(s * fps - TRIM_EPS)\n", "    a = round(s * fps)\n")], [T["ingest"]]),
    "33 toleransi 1e-6 dihapus": ([(IN, END_B, "        b = math.ceil(e * fps)\n")], [T["ingest"]]),
    "34 akhir trim inklusif": ([(IN, END_B, "        b = math.ceil(e * fps - TRIM_EPS) + 1\n")], [T["ingest"]]),
    "35 key trim untuk S = 0 tanpa akhir": ([(IN, "    if a == 0 and b is None:\n        return None\n", "")], [T["ingest"], T["cli"]]),
    "36 cek trim melewati akhir dihapus": ([(IN, "    if trim is not None:\n        _check_trim(trim, len(frames), target_fps, src, frames_dir, meta_path)\n",
                                             "")], [T["ingest"]]),
    # ── ingest: saran / kotak ──
    "37 deteksi pada rentang trim": ([(IN, 'detect_vf=f"fps={target_fps},scale={w0}:-2:flags={SCALE_FLAGS}"',
                                       'detect_vf=f"fps={target_fps}" + (f",trim=start_frame={trim[\'start_frame\']}" if trim else "") '
                                       '+ f",scale={w0}:-2:flags={SCALE_FLAGS}"')], [T["crop"]]),
    "38 saran dipakai tanpa kotak.json": ([(IN, '    box = kotak["box_source"] if kotak is not None and kotak["mode"] != "none" else None\n',
                                            '    box = kotak["box_source"] if kotak is not None and kotak["mode"] != "none" else '
                                            '(saran["box_source"] if saran else None)\n')], [T["crop"]]),
    "39 ingest mengabaikan kotak.json": ([(IN, "            kotak = crop_mod.kotak_for_ingest(crop_dir, identity, video_arg)\n",
                                           "            kotak = None\n")], [T["crop"]]),
    "40 deteksi tetap jalan walau sudah diputuskan": ([(IN, "        if kotak is None and cfg_c.suggest:\n", "        if cfg_c.suggest:\n")],
                                                      [T["crop"]]),
    "41 key crop tidak ditulis ke meta.json": ([(IN, '    if crop_key is not None:\n        meta["crop"] = crop_key\n', "")], [T["crop"]]),
    # ── cli ──
    "42 folder kerja tanpa trim": ([(CL, "    return cfg.paths.work_dir / CLIPS_DIRNAME / clip_name(str(video), trim_label(trim))\n",
                                     "    return cfg.paths.work_dir / CLIPS_DIRNAME / clip_name(str(video))\n")], [T["cli"]]),
    "43 run tidak berhenti pada saran tertunda": ([(CL, "            if stage == \"ingest\" and cfg.ingest.crop.suggest and "
                                                        "crop_mod.pending_saran(ctx.crop_dir) is not None:\n", "            if False:\n")],
                                                  [T["cli"]]),
    "44 crop.suggest false diabaikan cli": ([(CL, "            if stage == \"ingest\" and cfg.ingest.crop.suggest and ",
                                              "            if stage == \"ingest\" and ")], [T["cli"]]),
    "45 pre-flight trim dihapus": ([(CL, '    if old and meta.get("trim") != trim:\n', "    if False:\n")], [T["cli"]]),
    "46 pre-flight export tanpa trim": ([(CL, "style=style_label,\n                                                  trim=trim_label(trim))\n",
                                          "style=style_label)\n")], [T["cli"]]),
    "47 folder crop per folder trim (bukan per sumber)": ([(CL, '        a += ["--crop-dir", str(ctx.crop_dir)]\n',
                                                            '        a += ["--crop-dir", str(ctx.crop_dir) + (TRIM_SEP + trim_label(ctx.trim) '
                                                            'if ctx.trim else "")]\n')], [T["cli"]]),
    "48 crop menerima --start / --end": ([(CL, "    if a.trim_start is not None or a.trim_end is not None:\n        raise CliError(\"crop: kotak",
                                           "    if False:\n        raise CliError(\"crop: kotak")], [T["cli"]]),
    "49 perintah restart tanpa flag trim": ([(CL, "            print(f\"  {cli_cmd('run', wd)} --restart-from ingest --yes\", flush=True)\n",
                                              "            print(f\"  {PROG} run {quote_arg(a.video.resolve())} --restart-from ingest --yes\", "
                                              "flush=True)\n")], [T["cli"]]),
    # ── stage_common / export / pesan ──
    "50 cli_cmd tanpa flag trim": ([(SC, '    return f"{CLI_PROG} {stage} {video}{trim_flags(work_dir)}{config_flag()}"\n',
                                     '    return f"{CLI_PROG} {stage} {video}{config_flag()}"\n')], [T["clip_identity"], T["cli"]]),
    "51 pesan [2]/[2c] tanpa run --restart-from ingest": ([(SC, 'f"(MENJALANKAN GPU)" if run_cmd else ""))', 'f"(MENJALANKAN GPU)" if False else ""))')],
                                                          [T["clip_identity"]]),
    "52 segment tidak meneruskan run_cmd": ([(SG, "cmd=stage_cmd(clip.work_dir), run_cmd=f\"{cli_cmd('run', clip.work_dir)} --restart-from ingest --yes\")",
                                              "cmd=stage_cmd(clip.work_dir))")], [T["clip_identity"]]),
    "53 depth tidak meneruskan run_cmd": ([(DP, "cmd=stage_cmd(clip.work_dir), run_cmd=f\"{cli_cmd('run', clip.work_dir)} --restart-from ingest --yes\")",
                                            "cmd=stage_cmd(clip.work_dir))")], [T["clip_identity"]]),
    "54 nama export tanpa trim": ([(EX, '    trim = trim_label(meta.get("trim"))', "    trim = None")], [T["export"]]),
    "55 clip_name tanpa penanda trim": ([(EX, '    return sanitize_source_name(source_path) + (f"{TRIM_SEP}{trim}" if trim else "")\n',
                                          "    return sanitize_source_name(source_path)\n")], [T["export"], T["cli"]]),
    "56 pick_cmd tanpa flag trim": ([(PK, '        return f"{PROG} pick {_quote(self.video)}{trim_flags(self.work_dir)}{config_flag()} --frame {frame}"\n',
                                      '        return f"{PROG} pick {_quote(self.video)}{config_flag()} --frame {frame}"\n')], [T["pick"]]),
    "57 pesan ukuran beda tanpa saran pindah berkas": ([(PK, "Pindahkan berkas itu (mis. ganti nama menjadi points.old.json), lalu klik ulang ",
                                                         "Pindahkan berkas itu, lalu klik ulang ")], [T["pick"], T["isolate"]]),
    "58 pesan basi [1b] versi lama": ([(IS, "(mis. ingest diulang dengan crop / trim / video lain)", "mis. ingest diulang")], [T["isolate"]]),
    # ── config ──
    "59 validasi hint_static_frac dihapus": ([(CF, '    _between("ingest.crop.preview.hint_static_frac", c.ingest.crop.preview.hint_static_frac, 0, 1, '
                                                   'lo_open=True)\n', "")], [T["config"]]),
    "60 default band_min_px 8 -> 9": ([(CF, "    band_min_px: int = 8 ", "    band_min_px: int = 9 ")], [T["config"], T["crop"]]),
    "61 validasi band_min_px dihapus": ([(CF, '    _at_least("ingest.crop.detect.band_min_px", cd.band_min_px, 1)\n', "")], [T["config"]]),
    # ── perintah siap tempel + --config (persetujuan Rio 2026-10-22) ──
    "62 config_flag selalu kosong": ([(SC, '    return f" --config {quote_arg(_message_config)}" if _message_config is not None else ""\n',
                                       '    return ""\n')], [T["clip_identity"], T["cli"]]),
    "63 config default ikut dicetak": ([(SC, "    _message_config = None if p is None or p == MESSAGE_DEFAULT_CONFIG.resolve() else p\n",
                                         "    _message_config = p\n")], [T["clip_identity"]]),
    "64 cli.main tidak mencatat --config": ([(CL, '    set_message_config(getattr(args, "config", None))   # T-607: --config non-default ikut di '
                                                  'perintah siap tempel\n', "")], [T["cli"]]),
    "65 subperintah stage tidak mencatat --config": ([(CL, "    set_message_config(known.config)                 # T-607: --config non-default "
                                                           "ikut di perintah siap tempel\n", "")], [T["cli"]]),
    "66 main segment tidak mencatat --config": ([(SG, "    set_message_config(args.config)          # T-607: --config non-default ikut di "
                                                      "perintah siap tempel\n", "")], [T["clip_identity"]]),
    "67 main ingest tidak mencatat --config": ([(IN, "    set_message_config(args.config)                             # T-607: --config "
                                                     "non-default ikut di perintah siap tempel\n", "")], [T["crop"]]),
    "68 pesan exit 4 tanpa --config": ([(CL, "                      f\"    {crop_mod.crop_cmd(v, '--accept')}\\n\"",
                                         "                      f\"    python -m rotoscope crop {v} --accept\\n\"")], [T["cli"]]),
    "69 perintah crop tanpa --config": ([(CR, '    return f"{CLI_PROG} crop {video_arg} {flags}{config_flag()}"\n',
                                          '    return f"{CLI_PROG} crop {video_arg} {flags}"\n')], [T["crop"], T["cli"]]),
    "70 pesan preview: perintah mentah": ([(CL, '    full, seg_cmd, dep_cmd = (cli_cmd(s, work_dir) for s in ("run", "segment", "depth"))',
                                            '    full, seg_cmd, dep_cmd = (f"{PROG} {s} {video}" for s in ("run", "segment", "depth"))')],
                                          [T["cli"]]),
    "71 pick: perintah ingest tanpa trim / --config": ([(PK, '    ingest_cmd = f"{PROG} ingest {_quote(video)}{trim_flags(work_dir)}{config_flag()}"',
                                                         '    ingest_cmd = f"{PROG} ingest {_quote(video)}"')], [T["pick"]]),
    "72 pick_cmd tanpa --config": ([(PK, '        return f"{PROG} pick {_quote(self.video)}{trim_flags(self.work_dir)}{config_flag()} --frame {frame}"\n',
                                     '        return f"{PROG} pick {_quote(self.video)}{trim_flags(self.work_dir)} --frame {frame}"\n')],
                                   [T["pick"]]),
    "73 saran --adopt di run memakai <video>": ([(CL, "                       f\"{PROG} segment {stage_cmd} {'--adopt' if a.adopt else '--qc-only'}\"",
                                                  "                       f\"{PROG} segment <video> {'--adopt' if a.adopt else '--qc-only'}\"")],
                                                [T["cli"]]),
    "74 stylize: `run <video>` di peringatan": ([(ST, "`{cli_cmd('run', clip.work_dir)}` penuh berikutnya ", "`run <video>` penuh berikutnya ")],
                                                [T["preview"]]),
    # ── Tahap 4: usulan A (alternatif = saran → tanpa perintah) + B (pemetaan trim) — keputusan Rio 2026-10-23 ──
    "75 pesan saran: alternatif = saran tetap dicetak": ([(CR, '        if alt == saran["box_source"]:\n            out.append("  (kotak tanpa',
                                                           '        if False:\n            out.append("  (kotak tanpa')], [T["crop"]]),
    "76 header pratinjau: alternatif = saran tetap dicetak": ([(CR, '        if alt == saran["box_source"]:\n            ht += "; tanpa itu = '
                                                                    'kotak saran', '        if False:\n            ht += "; tanpa itu = kotak saran')],
                                                              [T["crop"]]),
    "77 pemetaan trim: detik kanonik, bukan yang diketik": ([(CL, '(("--start", start), ("--end", end)) if v is not None)',
                                                             '(("--start", trim["start_s"]), ("--end", trim["end_s"])) if v is not None)')],
                                                           [T["cli"]]),
    "78 pemetaan trim: frame akhir inklusif salah (b)": ([(CL, 'f"frame {a}–{b - 1} ({b - a} frame)"', 'f"frame {a}–{b} ({b - a} frame)"')],
                                                         [T["cli"]]),
    "79 pemetaan trim: subperintah ingest diam": ([(CL, '    if stage == "ingest":\n        print_trim_mapping(known.trim_start, known.trim_end, '
                                                        'trim, ctx.work_dir)\n', "")], [T["cli"]]),
    "80 pemetaan trim: run diam": ([(CL, '    print(f"run {video.name}: folder kerja {ctx.work_dir}, output {cfg.paths.out_dir}", flush=True)\n'
                                         '    print_trim_mapping(a.trim_start, a.trim_end, trim, ctx.work_dir)\n',
                                    '    print(f"run {video.name}: folder kerja {ctx.work_dir}, output {cfg.paths.out_dir}", flush=True)\n')],
                                   [T["cli"]]),
    "81 pemetaan trim: pick diam": ([(CL, '(CPU)", flush=True)\n        print_trim_mapping(a.trim_start, a.trim_end, trim, ctx.work_dir)\n',
                                      '(CPU)", flush=True)\n')], [T["cli"]]),
    "82 pemetaan trim: preview diam": ([(CL, 'keluaran {target}", flush=True)\n    print_trim_mapping(a.trim_start, a.trim_end, trim, ctx.work_dir)\n',
                                        'keluaran {target}", flush=True)\n')], [T["cli"]]),
}


def apply(name: str) -> None:
    muts, _ = MUTATIONS[name]
    by_mod: dict[str, list[tuple[str, str]]] = {}
    for m, old, new in muts:
        by_mod.setdefault(m, []).append((old, new))
    targets = [m for m in ORDER if m in by_mod] if by_mod else list(ORDER)    # kontrol: semua modul di-exec apa adanya
    for m in targets:
        mod = importlib.import_module(f"rotoscope.{m}")
        src = Path(mod.__file__).read_text(encoding="utf-8")
        for old, new in by_mod.get(m, []):
            n = src.count(old)
            if n != 1:
                raise SystemExit(f"mutasi {name!r}: teks lama muncul {n}x di {m}.py (harus 1x): {old[:70]!r}")
            src = src.replace(old, new)
        exec(compile(src, mod.__file__, "exec"), mod.__dict__)   # noqa: S102 — mutasi hidup di memori proses ini saja


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    names = list(MUTATIONS)
    if len(sys.argv) == 5 and sys.argv[1] == "--dari" and sys.argv[3] == "--sampai":
        names = names[int(sys.argv[2]):int(sys.argv[4])]
    elif len(sys.argv) > 1:
        import pytest

        apply(sys.argv[1])
        sys.exit(int(pytest.main([*MUTATIONS[sys.argv[1]][1], "-q", "-p", "no:cacheprovider", "--tb=no", "-rf"])))
    out = json.loads(OUT.read_text(encoding="utf-8")) if len(names) < len(MUTATIONS) and OUT.is_file() else {}
    out = {k: v for k, v in out.items() if k in MUTATIONS}
    for name in names:
        p = subprocess.run([sys.executable, __file__, name], cwd=REPO, capture_output=True, text=True, encoding="utf-8",
                           errors="replace")
        lines = p.stdout.splitlines()
        tail = [x for x in lines if re.search(r"\d+ (passed|failed)", x)][-1:]
        failed = [x.split("::", 1)[1].split(" ")[0] for x in lines if x.startswith("FAILED ")]
        out[name] = {"rc": p.returncode, "tests": MUTATIONS[name][1], "ringkas": tail, "test_gagal": p.returncode != 0,
                     "n_gagal": len(failed), "contoh": failed[:4],
                     "stderr": (p.stderr.strip().splitlines() + p.stdout.strip().splitlines())[-3:] if p.returncode not in (0, 1) else []}
        print(f"{name}: rc {p.returncode}, {tail[0] if tail else '-'}; contoh {failed[:2]}", flush=True)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps({k: out[k] for k in MUTATIONS if k in out}, indent=1, ensure_ascii=False), encoding="utf-8")
    muts = {k: v for k, v in out.items() if k != CONTROL}
    killed = sum(v["rc"] == 1 for v in muts.values())
    print(f"kontrol lolos: {out[CONTROL]['rc'] == 0 if CONTROL in out else '-'}; {killed}/{len(muts)} mutasi (dari "
          f"{len(MUTATIONS) - 1}) membuat test gagal (rc 1) -> {OUT}")


if __name__ == "__main__":
    main()
