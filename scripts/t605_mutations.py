"""T-605 mutation check (alat sekali pakai, F3): kode produksi di disk TIDAK diubah — tiap mutasi = penggantian teks pada salinan
sumber `rotoscope/isolate.py` (atau `cli.py`) DI MEMORI, di-exec ke modul yang sudah ter-import, dalam satu proses anak per mutasi
(pola `scripts/t604_mutations.py`). Tiap mutasi harus membuat minimal satu test di TESTS GAGAL; kontrol (sumber asli di-exec dengan
cara yang sama) harus LOLOS.

    python scripts/t605_mutations.py            -> semua (proses anak per mutasi) + work/t605/mutations.json
    python scripts/t605_mutations.py "<nama>"   -> satu mutasi (dipakai proses induk)

Daftar: 27 mutasi rancangan Tahap 1 (docs/04 "Rancangan T-605 Tahap 1" R6; no. 19 dipecah 19a / 19b) + 2 baru keputusan Rio
setelah Tahap 1 (basi diproses tanpa --yes; isolate/ tidak dihapus saat titik hilang) + 2 tambahan CLI (--yes / --limit ke [1b]).
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))
sys.path.insert(0, str(REPO / "src"))

TESTS = ["tests/test_isolate.py"]
OUT = REPO / "work" / "t605" / "mutations.json"
I, C = "isolate", "cli"

CONTROL = "kontrol (sumber asli di-exec; harus LOLOS)"
MUTATIONS: dict[str, list[tuple[str, str, str]]] = {
    CONTROL: [],
    # ── 27 mutasi rancangan Tahap 1 ──
    "01 cut diabaikan (satu adegan)": [
        (I, "    cut_before = stb.detect_cuts(scores, cut_diff)\n", "    cut_before = [False] * len(scores)\n")],
    "02 ambang cut bukan stabilize.temporal.cut_diff": [
        (I, "detect_scenes(clip, cfg.stabilize.temporal.cut_diff, thumbs)", "detect_scenes(clip, 0.06, thumbs)")],
    "03 tanpa propagasi mundur": [
        (I, "    if start > 0:\n        for k, out in engine.propagate(start, reverse=True):",
            "    if False:\n        for k, out in engine.propagate(start, reverse=True):")],
    "04 mundur: frame klik paling awal dihasilkan dua kali": [
        (I, "            if k not in seen:\n                seen.add(k)\n                yield lo + k, KIND_REV, out\n",
            "            seen.add(k)\n            yield lo + k, KIND_REV, out\n")],
    "05 prompt adegan lain bocor ke job": [
        (I, '            groups.setdefault((shot, int(t["id"])), []).append(p)\n',
            '            for s in shots:\n                groups.setdefault((s, int(t["id"])), []).append(p)\n')],
    "06 hanya frame klik paling awal dipakai (koreksi diabaikan)": [
        (I, "    for p in job.prompts:\n        k = p.index - lo\n", "    for p in job.prompts[:1]:\n        k = p.index - lo\n")],
    "07 frame kondisi tidak langsung dihitung (semua input dulu)": [
        (I, "    for p in job.prompts:\n        k = p.index - lo\n        engine.add_prompt(k, p)\n        out = engine.run_frame(k)\n",
            "    for p in job.prompts:\n        engine.add_prompt(p.index - lo, p)\n    for p in job.prompts:\n        k = p.index - lo\n"
            "        out = engine.run_frame(k)\n")],
    "08 kotak dibuang": [
        (I, "tuple(float(v) for v in box) if box is not None else None", "None")],
    "09 label 0 / 1 tertukar": [
        (I, 'tuple(int(q["label"]) for q in t["points"])', 'tuple(1 - int(q["label"]) for q in t["points"])')],
    "10 tumpang tindih: id terbesar selalu menang": [
        (I, "        take = mask & (lg > best)\n", "        take = mask\n")],
    "11 ambang masker >= (bukan >)": [
        (I, "return np.asarray(logits, np.float32) > np.float32(threshold)",
            "return np.asarray(logits, np.float32) >= np.float32(threshold)")],
    "12 frame tanpa_prompt diberi PNG": [
        (I, "    statuses = {i: frame_status(labels.get(i)) for i in clip.indices}\n",
            "    for i in clip.indices:\n        if i not in labels:\n            write_label(label_path(partial, i), np.zeros(size, np.uint8))\n"
            "    statuses = {i: frame_status(labels.get(i)) for i in clip.indices}\n")],
    "13 masker kosong ditandai target": [
        (I, "    return STATUS_TARGET if bool(label.any()) else STATUS_EMPTY\n", "    return STATUS_TARGET\n")],
    "14 kunci basi tanpa titik": [
        (I, '"points_sha256": points_sha256,', '"points_sha256": None,')],
    "15 kunci basi tanpa sidik frame": [
        (I, '"frames_pixel_sha256": frames_sha256,', '"frames_pixel_sha256": None,')],
    "16 kunci basi tanpa revision": [
        (I, '"model": {"model_id": i.model_id, "revision": i.revision}', '"model": {"model_id": i.model_id}')],
    "17 kunci basi tanpa mask_threshold": [
        (I, '"params": {"mask_threshold": i.mask_threshold, "cut_diff": cfg.stabilize.temporal.cut_diff}',
            '"params": {"cut_diff": cfg.stabilize.temporal.cut_diff}')],
    "18 vram_min_free_mib ikut kunci basi": [
        (I, '"precision": i.precision,', '"precision": i.precision, "vram": i.vram_min_free_mib,')],
    "19a tanpa points.json stage tetap membuat isolate/": [
        (I, "    if not present:\n        log(", "    if not present:\n        ensure_dir(work_dir / ISOLATE_DIRNAME)\n        log(")],
    "19b tanpa points.json run tetap membuat subprocess [1b]": [
        (C, 'if stage == "isolate" and not isolate_stage.points_present(ctx.work_dir):', "if False:")],
    "20 tulis langsung ke isolate/ (tanpa isolate.partial/)": [
        (I, "    partial = clip.partial_dir\n    _rmtree(partial)\n", "    partial = clip.out_dir\n    _rmtree(partial)\n"),
        (I, "    promote(clip)\n", "")],
    "21 hasil basi dianggap valid": [
        (I, '    if reasons:\n        return "basi", reasons\n', '    if False:\n        return "basi", reasons\n')],
    "22 frame klik basi diterima": [
        (I, "    bad = {f: s for f, s in statuses.items() if s != pick.STATUS_OK}\n", "    bad = {}\n")],
    "23 id target > 255 diterima": [
        (I, 'if t["id"] > MAX_TARGET_ID})', 'if t["id"] > 100000})')],
    "24 isolate --restart tanpa --yes lolos": [
        (C, "    gpu = [s for s in stages if s in GPU_STAGES]\n",
            '    gpu = [s for s in stages if s in GPU_STAGES and s != "isolate"]\n')],
    "25 --restart-from ingest tanpa isolate": [
        (C, 'RESTART_SCOPE = {"ingest": ("isolate", "segment",', 'RESTART_SCOPE = {"ingest": ("segment",')],
    "26 cek VRAM > (bukan >=)": [
        (I, "    if free_mib >= need_mib:\n        return\n    raise StageError(f\"VRAM",
            "    if free_mib > need_mib:\n        return\n    raise StageError(f\"VRAM")],
    "27 indeks lokal -> global bergeser satu": [
        (I, "            yield lo + k, KIND_FWD, out\n", "            yield lo + k + 1, KIND_FWD, out\n")],
    # ── 2 mutasi baru (keputusan Rio setelah Tahap 1) ──
    "28 basi diproses tanpa --yes": [
        (I, "    if jobs and not yes:\n", '    if jobs and not yes and state == "tidak_ada":\n')],
    "29 isolate/ tidak dihapus saat titik hilang": [
        (I, "    gone = remove_outputs(work_dir)\n", "    gone = []\n")],
    # ── 2 tambahan CLI ──
    "30 --yes tidak diteruskan ke [1b]": [
        (C, '        return a + (["--yes"] if ctx.yes else [])\n', "        return a\n")],
    "31 --limit diteruskan ke [1b]": [
        (C, '    if stage == "isolate":                  # T-605', '    if stage == "isolate" and ctx.limit is None:  # T-605')],
}


def apply(name: str) -> None:
    from rotoscope import cli, isolate

    mods = {I: isolate, C: cli}
    by_mod: dict[str, list[tuple[str, str]]] = {}
    for m, old, new in MUTATIONS[name]:
        by_mod.setdefault(m, []).append((old, new))
    for m in ([I] if not by_mod else by_mod):            # kontrol: isolate.py asli di-exec dengan cara yang sama
        mod = mods[m]
        src = Path(mod.__file__).read_text(encoding="utf-8")
        for old, new in by_mod.get(m, []):
            n = src.count(old)
            if n != 1:
                raise SystemExit(f"mutasi {name!r}: teks lama muncul {n}x di {m}.py (harus 1x): {old[:60]!r}")
            src = src.replace(old, new)
        exec(compile(src, mod.__file__, "exec"), mod.__dict__)   # noqa: S102 — mutasi hidup di memori proses ini saja


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    which = sys.argv[1] if len(sys.argv) > 1 else None
    if which is not None:
        import pytest

        apply(which)
        sys.exit(int(pytest.main([*TESTS, "-q", "-p", "no:cacheprovider", "--tb=no", "-rf"])))
    out = {}
    for name in MUTATIONS:
        p = subprocess.run([sys.executable, __file__, name], cwd=REPO, capture_output=True, text=True, encoding="utf-8",
                           errors="replace")
        lines = p.stdout.splitlines()
        tail = [x for x in lines if re.search(r"\d+ (passed|failed)", x)][-1:]
        failed = [x.split("::", 1)[1].split(" ")[0] for x in lines if x.startswith("FAILED ")]
        out[name] = {"rc": p.returncode, "ringkas": tail, "test_gagal": p.returncode != 0, "n_gagal": len(failed),
                     "contoh": failed[:4], "stderr": p.stderr.strip().splitlines()[-2:] if p.returncode not in (0, 1) else []}
        print(f"{name}: rc {p.returncode}, {tail[0] if tail else '-'}; contoh {failed[:3]}", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    muts = {k: v for k, v in out.items() if k != CONTROL}
    killed = sum(v["rc"] == 1 for v in muts.values())
    print(f"kontrol lolos: {out[CONTROL]['rc'] == 0}; {killed}/{len(muts)} mutasi membuat test gagal (rc 1) -> {OUT}")


if __name__ == "__main__":
    main()
