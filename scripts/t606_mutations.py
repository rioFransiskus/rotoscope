"""T-606 mutation check (alat sekali pakai, F3): kode produksi di disk TIDAK diubah — tiap mutasi = penggantian teks pada salinan
sumber `rotoscope/target_mask.py` / `stabilize.py` / `config.py` DI MEMORI, di-exec ke modul yang sudah ter-import, dalam satu proses
anak per mutasi (pola `scripts/t605_mutations.py`). Tiap mutasi harus membuat minimal satu test di TESTS GAGAL; kontrol (sumber asli
di-exec dengan cara yang sama) harus LOLOS.

    python scripts/t606_mutations.py            -> semua (proses anak per mutasi) + work/t606/mutations.json
    python scripts/t606_mutations.py --dari A --sampai B
                                                -> hanya entri ke-A … ke-(B − 1) daftar (0 = kontrol), BERURUTAN; hasil digabung ke
                                                   work/t606/mutations.json yang ada (untuk memecah run panjang)
    python scripts/t606_mutations.py "<nama>"   -> satu mutasi (dipakai proses induk)
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

TESTS = ["tests/test_target_mask.py", "tests/test_isolate.py::test_34_real_cpu_chain_byte_identical_with_points_and_isolate"]
OUT = REPO / "work" / "t606" / "mutations.json"
T, S, C = "target_mask", "stabilize", "config"

CONTROL = "kontrol (sumber asli di-exec; harus LOLOS)"
MUTATIONS: dict[str, list[tuple[str, str, str]]] = {
    CONTROL: [],
    # ── algoritma H ──
    "01 r_keep: >= jadi >": [(T, "    if r >= p.r_keep:\n", "    if r > p.r_keep:\n")],
    "02 r_drop: <= jadi <": [(T, "    if r <= p.r_drop:\n", "    if r < p.r_drop:\n")],
    "03 utuh / buang tertukar": [(T, "        return CLASS_KEEP\n    if r <= p.r_drop:\n        return CLASS_DROP\n",
                                  "        return CLASS_DROP\n    if r <= p.r_drop:\n        return CLASS_KEEP\n")],
    "04 lubang masker tidak diisi": [(T, "    m_f = fill_holes(mask, p.hole_fill_max_px)\n", "    m_f = mask.copy()\n")],
    "05 isi lubang tanpa cek tepi gambar": [(T, "    fill[border] = False\n", "")],
    "06 batas ukuran lubang <= jadi <": [(T, "        fill &= st[:, cv2.CC_STAT_AREA] <= max_px\n",
                                          "        fill &= st[:, cv2.CC_STAT_AREA] < max_px\n")],
    "07 pelebaran d - 1": [(T, "    m_d = dilate(m_f, p.dilate_px)\n", "    m_d = dilate(m_f, p.dilate_px - 1)\n")],
    "08 pelebaran persegi (bukan elips)": [(T, "cv2.MORPH_ELLIPSE", "cv2.MORPH_RECT")],
    "09 aturan potongan kecil diabaikan": [(T, "        ok = pin >= p.piece_min_px\n", "        ok = np.ones(pn, bool)\n")],
    "10 potongan kecil: >= jadi >": [(T, "        ok = pin >= p.piece_min_px\n", "        ok = pin > p.piece_min_px\n")],
    "11 r memakai M (bukan M_f)": [(T, "    inter = np.bincount(lab[m_f], minlength=n)\n",
                                    "    inter = np.bincount(lab[mask], minlength=n)\n")],
    "12 komponen 4-arah": [(T, "    n, lab, st, _ = cv2.connectedComponentsWithStats(fg.astype(np.uint8), connectivity=8)\n",
                           "    n, lab, st, _ = cv2.connectedComponentsWithStats(fg.astype(np.uint8), connectivity=4)\n")],
    "13 M_d dari M (bukan M_f)": [(T, "    m_d = dilate(m_f, p.dilate_px)\n", "    m_d = dilate(mask, p.dilate_px)\n")],
    "14 grup seri tidak di-nol": [(T, "    tie[remove] = BACKGROUND_ID\n", "")],
    "15 background one-hot = 0 (bukan 255)": [(T, "    sums[BACKGROUND_ID, remove] = PROBS_ONE\n", "")],
    # ── status ──
    "16 tanpa_prompt selalu dikosongkan": [(T, "self.params.no_prompt == NO_PROMPT_CLEAR)", "True)")],
    "17 no_prompt clear diabaikan": [(T, " or (st == STATUS_NO_PROMPT and self.params.no_prompt == NO_PROMPT_CLEAR)", "")],
    "18 frame kosong tidak dimasker": [(T, "MASKED_STATUSES = (STATUS_TARGET, STATUS_EMPTY)", "MASKED_STATUSES = (STATUS_TARGET,)")],
    # ── berlaku / basi ──
    "19 cek kunci [1b] dilewati": [(T, '        if state != "ok":\n', "        if False:\n")],
    "20 target_mask.enabled diabaikan": [(T, "    if not cfg.target_mask.enabled:\n", "    if False:\n")],
    "21 parameter target_mask tidak masuk blok manifest": [
        (T, '            "target_mask": to_dict(cfg.target_mask), "target_mask_hash": section_hash(cfg, "target_mask")}',
            "}")],
    "22 sidik isi label tidak masuk blok manifest": [
        (T, '"labels_sha256": labels_sha256(labels_dir, status),', "")],
    "23 blok isolate tidak dibandingkan (MANIFEST_MATCH_KEYS)": [(S, ', "temporal", tmk.MANIFEST_KEY)', ', "temporal")')],
    "24 key isolate selalu ditulis": [(S, "    if masks is not None:\n        m[tmk.MANIFEST_KEY] = masks.block\n",
                                       "    m[tmk.MANIFEST_KEY] = masks.block if masks is not None else {}\n")],
    # ── titik pasang [3] ──
    "25 F dari classmap (bukan argmax jumlah grup)": [
        (S, "                fg = group_argmax(sums, tie)[0] != BACKGROUND_ID", "                fg = tie != BACKGROUND_ID")],
    "26 indeks frame masker bergeser 1": [(S, "            index = self.clip.indices[k]\n", "            index = self.clip.indices[k] + 1\n")],
    "27 masker tidak sampai ke kernel (jumlah grup asli disimpan)": [
        (S, "            sums, tie = group_sums(probs, self.lut, self.n_groups), self.lut[cm]\n",
            "            sums, tie = group_sums(probs, self.lut, self.n_groups), self.lut[cm]\n            orig = sums.copy()\n"),
        (S, "            self._raw[k] = (sums, tie, d)\n", "            self._raw[k] = (orig, tie, d)\n")],
    "28 masker hanya di frame tengah jendela": [
        (S, "            sums, tie = group_sums(probs, self.lut, self.n_groups), self.lut[cm]\n",
            "            sums, tie = group_sums(probs, self.lut, self.n_groups), self.lut[cm]\n"
            "            self.__dict__.setdefault('_unm', {})[k] = sums.copy()\n"),
        (S, "                    acc += np.float32(wt) * self.raw(j)[0]\n",
            "                    acc += np.float32(wt) * (self.raw(j)[0] if j == k else (self.raw(j), self._unm[j])[1])\n")],
    # ── config ──
    "29 validasi r_drop < r_keep dihapus": [(C, '    _less("target_mask.r_drop", tm.r_drop, "target_mask.r_keep", tm.r_keep)\n', "")],
}


def apply(name: str) -> None:
    from rotoscope import config, stabilize, target_mask

    mods = {T: target_mask, S: stabilize, C: config}
    by_mod: dict[str, list[tuple[str, str]]] = {}
    for m, old, new in MUTATIONS[name]:
        by_mod.setdefault(m, []).append((old, new))
    for m in ([T, S] if not by_mod else by_mod):         # kontrol: sumber asli di-exec dengan cara yang sama
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
