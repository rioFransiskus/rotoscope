"""T-604 mutation check (alat sekali pakai, F3): kode produksi di disk TIDAK diubah — tiap mutasi = penggantian teks pada salinan
sumber `rotoscope/pick.py` (atau `cli.py`) DI MEMORI, di-exec ke modul yang sudah ter-import, dalam satu proses anak per mutasi.
Tiap mutasi harus membuat minimal satu test di TESTS GAGAL; kontrol (sumber asli di-exec dengan cara yang sama) harus LOLOS.

    python scripts/t604_mutations.py            -> semua (proses anak per mutasi) + work/t604/mutations.json
    python scripts/t604_mutations.py "<nama>"   -> satu mutasi (dipakai proses induk)
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

TESTS = ["tests/test_pick.py"]
OUT = REPO / "work" / "t604" / "mutations.json"
P, C = "pick", "cli"

CONTROL = "kontrol (sumber asli di-exec; harus LOLOS)"
MUTATIONS: dict[str, list[tuple[str, str, str]]] = {
    CONTROL: [],
    # ── 16 mutasi rancangan Tahap 1 ──
    "pusat piksel dibuang (rumus prototipe dx / s)": [
        (P, "        x = (dx + 0.5) / self.sx - 0.5\n        y = (dy + 0.5) / self.sy - 0.5\n",
            "        x = dx / self.sx\n        y = dy / self.sy\n"),
        (P, "        dx = math.floor((x + 0.5) * self.sx)\n        dy = math.floor((y + 0.5) * self.sy)\n",
            "        dx = round(x * self.sx)\n        dy = round(y * self.sy)\n")],
    "s dipakai alih-alih sx / sy": [
        (P, "x = (dx + 0.5) / self.sx - 0.5", "x = (dx + 0.5) / self.scale - 0.5"),
        (P, "y = (dy + 0.5) / self.sy - 0.5", "y = (dy + 0.5) / self.scale - 0.5"),
        (P, "dx = math.floor((x + 0.5) * self.sx)", "dx = math.floor((x + 0.5) * self.scale)"),
        (P, "dy = math.floor((y + 0.5) * self.sy)", "dy = math.floor((y + 0.5) * self.scale)")],
    "tanpa clamp (tampilan -> frame)": [
        (P, "        x = min(max(x, 0.0), self.width - 1.0)\n        y = min(max(y, 0.0), self.height - 1.0)\n", "")],
    "cek sidik frame selalu ok": [
        (P, '    return STATUS_OK if cache[f] == entry["frame_pixel_sha256"] else STATUS_STALE\n', "    return STATUS_OK\n")],
    "sidik frame dari indeks salah": [
        (P, "frame_pixel_sha256(work_dir / FRAMES_DIRNAME / (FRAME_PATTERN % f))",
            "frame_pixel_sha256(work_dir / FRAMES_DIRNAME / (FRAME_PATTERN % (f + 1)))")],
    "gabung menimpa seluruh frames": [
        (P, '    frames = [e for e in (disk or {}).get("frames", []) if e["frame_index"] != frame]\n', "    frames = []\n")],
    "gabung memakai salinan saat dibuka (bukan baca ulang disk)": [
        (P, "        disk = load_points(path)\n", "        disk = session.doc\n")],
    "kotak kedua digabung ke kotak lama (bukan menggantikan)": [
        (P, '        t["box"] = [x0, y0, x1, y1]\n',
            '        t["box"] = [x0, y0, x1, y1] if old is None else [min(old[0], x0), min(old[1], y0), max(old[2], x1), '
            'max(old[3], y1)]\n')],
    "U setelah kotak pengganti tidak mengembalikan kotak lama": [
        (P, '            self.targets[act[1]]["box"] = act[2]\n', '            self.targets[act[1]]["box"] = None\n')],
    "N tanpa putaran": [
        (P, "        self.active = self.active + 1 if self.active + 1 <= top else 1\n", "        self.active = self.active + 1\n")],
    "label 2 / titik di luar frame diterima": [
        (P, "                if not _is_int(lab) or lab not in (0, 1):\n", "                if not _is_int(lab):\n"),
        (P, "                    if not 0 <= v <= lim - 1:\n", "                    if False:\n")],
    "peringatan tanpa klik kanan terbalik": [
        (P, 'if not any(p["label"] == 0 for p in t["points"])]', 'if any(p["label"] == 0 for p in t["points"])]')],
    "ingest dipanggil walau meta.json ada": [
        (C, '    if _read_meta(ctx.work_dir) is None:\n        print(f"frames/ belum ada',
            '    if True:\n        print(f"frames/ belum ada')],
    "frame basi dimuat": [
        (P, 'loaded = entry["targets"] if entry is not None and session.status.get(session.frame) == STATUS_OK else None',
            'loaded = entry["targets"] if entry is not None else None')],
    "hapus entri tanpa konfirmasi": [
        (P, "            if self.entry_exists:\n                self.mode = MODE_CONFIRM_DELETE\n",
            "            if self.entry_exists:\n                return ACT_DELETE\n")],
    "ambang tarik diabaikan": [
        (P, "abs(self.cur[1] - self.press[1])) >= DRAG_MIN_PX:", "abs(self.cur[1] - self.press[1])) >= 0:")],
    # ── 3 mutasi baru (keputusan Rio 2026-10-16) ──
    "sidik dari byte PNG (bukan piksel)": [
        (P, "    return pixel_sha256(read_rgb(path))\n",
            "    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else pixel_sha256(read_rgb(path))\n"),
        (P, "    pixel_sha = pixel_sha256(rgb)\n", "    pixel_sha = frame_pixel_sha256(frame_path)\n")],
    "Esc / X tanpa konfirmasi": [
        (P, "        if self.mode == MODE_CONFIRM_EXIT or not self.dirty:\n            return ACT_EXIT\n",
            "        return ACT_EXIT\n")],
    "target kosong ikut tersimpan": [
        (P, "            for tid, t in state.content().items()]\n", "            for tid, t in sorted(state.targets.items())]\n")],
}


def apply(name: str) -> None:
    from rotoscope import cli, pick

    mods = {P: pick, C: cli}
    by_mod: dict[str, list[tuple[str, str]]] = {}
    for m, old, new in MUTATIONS[name]:
        by_mod.setdefault(m, []).append((old, new))
    for m in ([P] if not by_mod else by_mod):            # kontrol: pick.py asli di-exec dengan cara yang sama
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
