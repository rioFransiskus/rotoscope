"""T-609 Tahap 4 (alat, F3; CPU, baca-saja terhadap produksi): penilaian Rio + data berharga -> JSON ter-track.

    python scripts/t609_tahap4.py penilaian   # isian Rio VERBATIM dari work/t609/PANDUAN_UJI.md (blok "Template jawaban";
                                              # blok "Contoh isian" BUKAN jawaban) -> work/t609/t609_penilaian.json
    python scripts/t609_tahap4.py data        # ringkasan dari work/t609/*.json (ukur Tahap 1, uji nyata Tahap 3, bukti kunci,
                                              # mutasi, kegagalan sesaat, F1) -> work/t609/t609_data.json
    python scripts/t609_tahap4.py cek         # bangun ulang keduanya DI MEMORI, bandingkan byte dengan scripts/t609_*.json

Keluaran JSON: indent 1, ensure_ascii False, LF, diakhiri satu baris baru. Hanya menulis ke work/t609/.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
W = ROOT / "work" / "t609"
GUIDE = W / "PANDUAN_UJI.md"
TEMPLATE_HEAD = "## Template jawaban (salin, isi, kirim)"
PENILAIAN_NAME, DATA_NAME = "t609_penilaian.json", "t609_data.json"
K5, K3 = "klip5@0-2.25", "klip3"


def jload(name: str):
    return json.loads((W / name).read_text(encoding="utf-8"))


def dump(obj) -> bytes:
    return (json.dumps(obj, indent=1, ensure_ascii=False) + "\n").encode("utf-8")


# ── penilaian ──────────────────────────────────────
def build_penilaian() -> dict:
    raw = GUIDE.read_bytes()
    lines = raw.decode("utf-8").splitlines()
    head = lines.index(TEMPLATE_HEAD)
    start = next(i for i in range(head + 1, len(lines)) if lines[i].startswith("```")) + 1
    end = next(i for i in range(start, len(lines)) if lines[i].startswith("```"))
    block = lines[start:end]
    answers = {}
    for ln in block:
        if ln.strip():
            label, value = ln.split(": ", 1)
            answers[label] = value
    return {
        "format": "t609-penilaian/1",
        "sumber": {"path": "work/t609/PANDUAN_UJI.md", "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
                   "blok": TEMPLATE_HEAD, "baris_isi": [start + 1, end]},
        "tanggal_uji": "2026-11-02",
        "verbatim": block,
        "jawaban": answers,
        "catatan": ["Pesan Rio 2026-11-02: semua lolos.",
                    "Blok 'Contoh isian' di bawah template = contoh asli panduan, BUKAN jawaban Rio (diabaikan).",
                    "Isian U4 'tulis baris yang muncul' dan 'angka' dibiarkan kosong oleh Rio (bagian itu terhapus dari baris); "
                    "jawaban 'ya' berlaku dan dicatat apa adanya."],
        "semua_ya": all(v.split(" — ")[0] == "ya" for v in answers.values()),
        "kriteria": {
            "K1": {"lolos": True, "bukti": [
                "test_pick: test_t609_over_negative_per_target_and_strict_threshold, test_t609_panel_mark_and_yellow_block_only_"
                "above_threshold, test_t609_terminal_warning_after_save_not_blocking, test_t609_loaded_points_over_threshold_"
                "marked_at_open (10 = tanpa tanda / tanpa PERINGATAN; 11 = tanda + blok + PERINGATAN; ambang dari YAML)",
                "uji Rio U1 langkah 3-9 = ya (scratch work/t609_scratch/pick/)"]},
            "K2": {"lolos": True, "bukti": [
                "aturan keluar no. 3 tidak memicu: median 1,937 s <= 15 s, RAM 1 247,2 MiB <= 4 096 MiB, tanpa CUDA",
                "satu perintah CPU per klip (pick --mask-preview) -> PNG f0 / f22 / f40 klip5@0-2.25 dan f85 klip3; "
                "cuda_initialized false; CUDA_VISIBLE_DEVICES=-1 dipaksa cli + dicek modul",
                "uji Rio U2: gambar jelas = ya untuk klip5 f0 / f22 / f40 dan klip3 f85; contoh SALAH terlihat salah = ya"]},
            "K3": {"lolos": True, "bukti": [
                "test_isolate::test_44_neg_warning_before_gpu_and_message_without_run (tanpa cli_cmd('run'), tanpa 'rotoscope run', "
                "tanpa '(atau'); mutan 'pesan butuh GPU menyarankan lagi run --yes' tertangkap",
                "uji Rio U3 langkah 1 = ya"]},
            "K4": {"lolos": True, "bukti": [
                "F1: setiap snapshot sesudah = sebelum (11 332 berkas; daftar di t609_data.json f1)",
                "suite Tahap 4: 38 berkas, 1 751 lolos, 5 skip, 0 gagal",
                "mutasi: 44 / 44 tertangkap + kontrol lolos (t609_data.json mutasi)"]}},
    }


# ── data ───────────────────────────────────────────
FRAME_KEYS = ("frame", "target", "kiri", "kanan", "kotak", "skor_objek", "luas_px", "porsi_frame", "komponen_8arah",
              "porsi_komponen_terbesar", "infer_s", "total_s")
PREV_KEYS = ("id", "kiri", "kanan", "kotak", "skor_objek", "luas_px", "porsi_frame", "komponen_8arah", "porsi_komponen_terbesar",
             "waktu_s", "png", "peringatan")


def tahap1() -> dict:
    out = {"salinan": {k: {x: v[x] for x in ("berkas", "bytes", "sha256_beda", "hard_link_atau_sama")}
                       for k, v in jload("salinan.json")["klip"].items()}}
    for k in (K5, K3):
        d = jload(f"ukur_{k}.json")
        out[k] = {"threads": d["threads"], "import_s": d["import_s"], "describe_s": d["describe_s"], "open_s": d["open_s"],
                  "proses_s": d["proses_s"], "torch": d["torch"], "transformers": d["transformers"],
                  "model": d["model"], "cuda_available": d["cuda_available"], "cuda_initialized": d["cuda_initialized"],
                  "mem_mib": {"awal": d["mem_awal_mib"], "sesudah_muat": d["mem_sesudah_muat_mib"], "akhir": d["mem_akhir_mib"]},
                  "putaran": [{"putaran": p["putaran"], "median_total_s": p["median_total_s"], "median_infer_s": p["median_infer_s"],
                               "frame": [{x: r[x] for x in FRAME_KEYS} | {"masker_sama_putaran1": r["masker_sama_putaran1"]}
                                         for r in p["frame"]]} for p in d["putaran"]],
                  "produksi_klik": d["produksi_klik"]}
    p = jload("ukur_putusan.json")
    out["putusan"] = {"aturan_keluar": p["aturan_keluar"], "pembanding": p["pembanding"]}
    return out


def tahap3() -> dict:
    s = jload("tahap3_siapkan.json")
    j = jload("tahap3_jalan.json")
    siap = {}
    for name, v in s.items():
        siap[name] = {"klip": v["klip"], "berkas": v["berkas"], "bytes": v["bytes"], "config": v["config"]["path"]}
        if "target2_sintetis" in v:
            siap[name]["target2_sintetis"] = v["target2_sintetis"]
    prev = {}
    for name in (f"pratinjau_{K5}", f"pratinjau_{K3}", "pratinjau_salah", "pratinjau_multi"):
        v = j[name]
        prev[name] = {"run": v["run"], "peringatan": v["peringatan"]}
        if "frames" in v:
            prev[name]["frames"] = [{"frame": fr["frame"], "targets": [{x: t[x] for x in PREV_KEYS} for t in fr["targets"]]}
                                    for fr in v["frames"]]
    return {"siapkan": siap,
            "runs": {k: {x: v[x] for x in ("cmd", "rc", "rc_harapan", "s")} for k, v in j["runs"].items()},
            "banding_tahap1": j["banding_tahap1"], "pratinjau": prev,
            "points_produksi_tidak_berubah": j["points_produksi_tidak_berubah"],
            "produksi_tanpa_pratinjau": j["produksi_tanpa_pratinjau"],
            "bahan_nilai": {k: v["berkas"] for k, v in jload("tahap3_bahan.json").items()}}


def kunci() -> dict:
    d = jload("kunci_bukti.json")
    klip = {}
    for name, r in d["klip"].items():
        klip[name] = {k: (v["sama_semua"] if "sama_semua" in v else {n: x["status"] for n, x in v.items()}) for k, v in r.items()}
    return {"nilai": d["nilai"], "klip": klip, "semua_sama_dan_up_to_date": d["semua_sama_dan_up_to_date"]}


def mutasi() -> dict:
    merged: dict = {}
    for i in (1, 2, 3, 4):
        merged.update(jload(f"mutations_{i}.json"))
    final_cli = jload("mutations_cli2.json")                 # sesudah run_gpu_stage dikembalikan (tanggapan Rio no. 1)
    merged.update(final_cli)
    rows = {k: {"rc": v["rc"], "n_gagal": v["n_gagal"], "gagal": v.get("gagal", v.get("contoh"))} for k, v in merged.items()}
    control = [k for k in rows if k.startswith("kontrol")]
    muts = {k: v for k, v in rows.items() if k not in control}
    return {"sumber": ["work/t609/mutations_1..4.json (putaran final kode Tahap 2-3)",
                       "work/t609/mutations_cli2.json (mutan cli, sesudah run_gpu_stage dikembalikan; menimpa entri sama)"],
            "kontrol_lolos": all(rows[k]["rc"] == 0 for k in control),
            "tertangkap": sum(v["rc"] == 1 and v["n_gagal"] > 0 for v in muts.values()), "jumlah": len(muts), "mutan": rows}


def flaky() -> dict:
    out = {}
    for label in ("beku", "kunci00", "kunci02"):
        d = jload(f"flaky_{label}.json")
        out[label] = {"jeda_retry_s": d["jeda_retry_s"], "rc": d["rc"],
                      "test": {k: v["hasil"] for k, v in d["test"].items()},
                      "statistik": d["statistik"] if label != "beku" else {"fungsi_dibekukan": len(d["statistik"]["dibekukan"])}}
    out["kesimpulan"] = {"created_utc": "TIDAK terkait - terbukti (mode beku: 3 / 3 lolos)",
                         "kunci_berkas": "dugaan - mekanisme direproduksi (jeda 0: 3 / 3 gagal; jeda 0,2 s: 3 / 3 lolos)"}
    return out


F1_LABELS = ("sebelum_tahap1", "sesudah_tahap1", "sebelum_tahap2", "sesudah_tahap2", "sesudah_tanggapan")   # sebelum data ditulis


def f1() -> list:
    rows = []
    for label in F1_LABELS:
        d = jload(f"f1_{label}.json")
        row = {"label": d["which"], "n": d["n"], "list_sha256": d["list_sha256"]}
        for k, v in d.items():
            if k.startswith("vs_"):
                row[k] = {"identik": v["identical"], "hilang": len(v["missing"]), "berubah": len(v["changed"]),
                          "tambahan": len(v["extra"])}
        rows.append(row)
    return rows


def build_data() -> dict:
    return {
        "format": "t609-data/1",
        "catatan": "Ringkasan data T-609 (docs/04 entri T-609). Sumber: work/t609/*.json (dipertahankan); scratch work/t609_scratch/ "
                   "dihapus di F4 Tahap 4.",
        "tahap1_ukur": tahap1(), "tahap3_uji_nyata": tahap3(), "kunci_bukti": kunci(), "mutasi": mutasi(),
        "kegagalan_sesaat": flaky(), "f1": f1(),
        "skrip": {
            "t609_hash.py": "mandiri (membaca produksi)",
            "t609_kunci.py": "mandiri (baca-saja produksi; menimpa work/t609/kunci_bukti.json)",
            "t609_mutations.py": "mandiri (tests/)",
            "t609_flaky.py": "mandiri (tests/)",
            "t609_tahap4.py": "mandiri selama work/t609/ ada",
            "t609_ukur.py": "TIDAK langsung: ukur / putusan butuh scratch; jalankan `siapkan` dulu (membuat ulang work/t609_scratch/ "
                            "dari produksi) - menimpa work/t609/ukur_*.json",
            "t609_tahap3.py": "TIDAK langsung: jalan / bahan butuh scratch (t609_ukur.py siapkan + t609_tahap3.py siapkan); "
                              "bahan menolak bila work/t609/nilai/ ada; titik U1 Rio di scratch pick/ hilang"},
    }


BUILDERS = {"penilaian": (build_penilaian, PENILAIAN_NAME), "data": (build_data, DATA_NAME)}


def main(argv: list[str]) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if argv in (["penilaian"], ["data"]):
        fn, name = BUILDERS[argv[0]]
        blob = dump(fn())
        (W / name).write_bytes(blob)
        print(f"work/t609/{name}: {len(blob)} B, sha256 {hashlib.sha256(blob).hexdigest()}")
        return 0
    if argv == ["cek"]:
        ok = True
        for fn, name in BUILDERS.values():
            blob, tracked = dump(fn()), ROOT / "scripts" / name
            same = tracked.is_file() and tracked.read_bytes() == blob
            ok &= same
            print(f"scripts/{name}: {'byte-identik' if same else 'BERBEDA / tidak ada'} dengan bangun ulang "
                  f"({len(blob)} B, sha256 {hashlib.sha256(blob).hexdigest()[:12]})")
        return 0 if ok else 1
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
