"""T-305f mutation check (alat sekali pakai, F3): kode produksi di disk TIDAK diubah — tiap mutasi = penggantian teks pada salinan
sumber `rotoscope/stabilize.py` / `config.py` DI MEMORI, di-exec ke modul yang sudah ter-import, dalam satu proses anak per mutasi
(pola `scripts/t606_mutations.py`). Tiap mutasi harus membuat minimal satu test di TESTS GAGAL; kontrol (sumber asli di-exec dengan
cara yang sama) harus LOLOS.

    python scripts/t305f_mutations.py            -> semua (proses anak per mutasi) + work/t305f/mutations.json
    python scripts/t305f_mutations.py --dari A --sampai B
                                                 -> hanya entri ke-A … ke-(B − 1) daftar (0 = kontrol), digabung ke mutations.json
    python scripts/t305f_mutations.py "<nama>"   -> satu mutasi (dipakai proses induk)

Mutan setara yang TIDAK dimasukkan (terbukti tidak bisa dibedakan, bukan celah test):
- cincin dibaca dari peta yang sedang diisi (`out`): komponen yang terisi tidak punya background di cincin 8-arahnya, jadi tidak ada
  komponen lain yang bersentuhan dengannya → hasil sama;
- VF dipanggil sebelum `final_depth` di loop: kedalaman membaca peta dari cache (`cache.groups`), bukan variabel lokal → hasil sama.
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

TESTS = ["tests/test_stabilize_hole_fill.py"]
OUT = REPO / "work" / "t305f" / "mutations.json"
S, C = "stabilize", "config"
FACE = "2"                                           # id grup face pada groups: default (dipakai mutan yang memindah VF)
NO_LOOP_VF = (S, "                gmap, n_reg, n_px = fill_group_holes(gmap, hole_fill_id)\n", "                n_reg, n_px = 0, 0\n")

CONTROL = "kontrol (sumber asli di-exec; harus LOLOS)"
MUTATIONS: dict[str, list[tuple[str, str, str]]] = {
    CONTROL: [],
    # ── aturan VF ──
    "01 komponen background 8-arah": [
        (S, "    n, lab, st, _ = cv2.connectedComponentsWithStats((gmap == BACKGROUND_ID).astype(np.uint8), connectivity=4)\n",
            "    n, lab, st, _ = cv2.connectedComponentsWithStats((gmap == BACKGROUND_ID).astype(np.uint8), connectivity=8)\n")],
    "02 cincin 4-arah (kernel silang)": [
        (S, "        ring = cv2.dilate(comp.astype(np.uint8), RING_KERNEL).astype(bool) & ~comp\n",
            "        ring = cv2.dilate(comp.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))).astype(bool) & ~comp\n")],
    "03 cek tepi frame dihapus": [(S, "        if x == 0 or y == 0 or x + bw >= w or y + bh >= h:\n", "        if False:\n")],
    "04 tepi atas / kiri tidak dicek (== 0 jadi < 0)": [
        (S, "        if x == 0 or y == 0 or x + bw >= w or y + bh >= h:\n", "        if x < 0 or y < 0 or x + bw >= w or y + bh >= h:\n")],
    "05 tepi kanan: >= jadi >": [
        (S, "        if x == 0 or y == 0 or x + bw >= w or y + bh >= h:\n", "        if x == 0 or y == 0 or x + bw > w or y + bh >= h:\n")],
    "06 tepi bawah: >= jadi >": [
        (S, "        if x == 0 or y == 0 or x + bw >= w or y + bh >= h:\n", "        if x == 0 or y == 0 or x + bw >= w or y + bh > h:\n")],
    "07 cincin cukup sebagian face (all jadi any)": [
        (S, "        if (gmap[sl][ring] == group_id).all():\n", "        if (gmap[sl][ring] == group_id).any():\n")],
    "08 diisi id salah": [(S, "            out[sl][comp] = group_id\n", "            out[sl][comp] = group_id + 1\n")],
    "09 batas ukuran disisipkan (> 1000 px dilewati)": [
        (S, "        sl = (slice(y - 1, y + bh + 1), slice(x - 1, x + bw + 1))\n        comp = lab[sl] == i\n        ring = cv2.dilate(comp",
            "        if a > 1000:\n            continue\n        sl = (slice(y - 1, y + bh + 1), slice(x - 1, x + bw + 1))\n"
            "        comp = lab[sl] == i\n        ring = cv2.dilate(comp")],
    # ── tempat pasang ──
    "10 VF sebelum filter pulau (di clean_groups)": [
        (S, "    isl = island_filter(raw, island_min_px)\n", f"    raw = fill_group_holes(raw, {FACE})[0]\n    isl = island_filter(raw, island_min_px)\n"),
        NO_LOOP_VF],
    "11 VF di jumlah grup mentah raw(k) (P-pre)": [
        (S, "            self._raw[k] = (sums, tie, d)\n",
            f"            _g0 = group_argmax(sums, tie)[0]\n            _f = fill_group_holes(_g0, {FACE})[0] != _g0\n"
            f"            sums[:, _f] = 0\n            sums[{FACE}, _f] = 255\n            tie[_f] = {FACE}\n"
            "            self._raw[k] = (sums, tie, d)\n"),
        NO_LOOP_VF],
    "12 VF ikut statistik kedalaman (PA)": [
        (S, "            self._groups[k] = clean_groups(p, tie, st.island_min_px, st.mode_k)\n",
            "            _g, _s = clean_groups(p, tie, st.island_min_px, st.mode_k)\n"
            f"            self._groups[k] = (fill_group_holes(_g, {FACE})[0], _s)\n")],
    # ── sakelar / grup / basi / log ──
    "13 enabled diabaikan (selalu aktif)": [(S, "    if not cfg.stabilize.hole_fill.enabled:\n        return None\n", "")],
    "14 VF tidak pernah jalan (selalu mati)": [
        (S, "    return [name for name, _ in cfg.groups].index(HOLE_FILL_GROUP) + 1\n", "    return None\n")],
    "15 grup pengisi salah (hair)": [
        (S, "    return [name for name, _ in cfg.groups].index(HOLE_FILL_GROUP) + 1\n",
            "    return [name for name, _ in cfg.groups].index(\"hair\") + 1\n")],
    "16 hole_fill tidak ikut stabilize_hash": [
        (S, '"stabilize_hash": section_hash(cfg, "stabilize"),',
            '"stabilize_hash": section_hash(__import__("dataclasses").replace(cfg, stabilize=__import__("dataclasses").replace('
            'cfg.stabilize, hole_fill=type(cfg.stabilize.hole_fill)())), "stabilize"),')],
    "17 frames.jsonl tanpa hole_fill": [
        (S, '                gstats = {**gstats, "fg_px": gstats["fg_px"] + n_px, "hole_fill_regions": n_reg, "hole_fill_px": n_px}\n',
            "                pass\n")],
    "18 fg_px tidak ikut piksel terisi": [(S, '"fg_px": gstats["fg_px"] + n_px, ', "")],
    "19 validasi grup face dihapus (config)": [
        (C, "    if c.stabilize.hole_fill.enabled and HOLE_FILL_GROUP not in [name for name, _ in c.groups]:\n", "    if False:\n")],
}


def apply(name: str) -> None:
    from rotoscope import config, stabilize

    mods = {S: stabilize, C: config}
    by_mod: dict[str, list[tuple[str, str]]] = {}
    for m, old, new in MUTATIONS[name]:
        by_mod.setdefault(m, []).append((old, new))
    for m in ([S] if not by_mod else by_mod):           # kontrol: sumber asli di-exec dengan cara yang sama
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
    names = list(MUTATIONS)
    if len(sys.argv) == 5 and sys.argv[1] == "--dari" and sys.argv[3] == "--sampai":
        names = names[int(sys.argv[2]):int(sys.argv[4])]
    elif len(sys.argv) > 1:
        import pytest

        apply(sys.argv[1])
        sys.exit(int(pytest.main([*TESTS, "-q", "-p", "no:cacheprovider", "--tb=no", "-rf"])))
    out = json.loads(OUT.read_text(encoding="utf-8")) if len(names) < len(MUTATIONS) and OUT.is_file() else {}
    out = {k: v for k, v in out.items() if k in MUTATIONS}
    for name in names:
        p = subprocess.run([sys.executable, __file__, name], cwd=REPO, capture_output=True, text=True, encoding="utf-8",
                           errors="replace")
        lines = p.stdout.splitlines()
        tail = [x for x in lines if re.search(r"\d+ (passed|failed)", x)][-1:]
        failed = [x.split("::", 1)[1].split(" ")[0] for x in lines if x.startswith("FAILED ")]
        out[name] = {"rc": p.returncode, "ringkas": tail, "test_gagal": p.returncode != 0, "n_gagal": len(failed),
                     "contoh": failed[:4], "stderr": p.stderr.strip().splitlines()[-2:] if p.returncode not in (0, 1) else []}
        print(f"{name}: rc {p.returncode}, {tail[0] if tail else '-'}; contoh {failed[:3]}", flush=True)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps({k: out[k] for k in MUTATIONS if k in out}, indent=1, ensure_ascii=False), encoding="utf-8")
    muts = {k: v for k, v in out.items() if k != CONTROL}
    killed = sum(v["rc"] == 1 for v in muts.values())
    print(f"kontrol lolos: {out[CONTROL]['rc'] == 0 if CONTROL in out else '-'}; {killed}/{len(muts)} mutasi (dari "
          f"{len(MUTATIONS) - 1}) membuat test gagal (rc 1) -> {OUT}")


if __name__ == "__main__":
    main()
