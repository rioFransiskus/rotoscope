"""T-305f (alat sekali pakai, F3): susun data T-305f yang berharga dari work/t305f/*.json (penutupan Tahap 4).

    python scripts/t305f_data.py susun    -> work/t305f/t305f_data.json (ISI untuk scripts/t305f_data.json; berkas repo ditulis lewat
                                             Write oleh Claude Code, bukan oleh skrip ini)
    python scripts/t305f_data.py cek      -> susun ulang di memori dan bandingkan BYTE dengan scripts/t305f_data.json

Isi: aturan VF + keputusan Rio, Tahap 1 (P-post / P-pre / PA per klip, garis / gambar, waktu, kesetaraan prototipe), Tahap 2 (mutasi,
suite), Tahap 3 (stabilize_hash lama → baru, run 6 klip, verifikasi, frame klip5 yang berubah, F1), bahan, penilaian Rio. Hanya MEMBACA
work/t305f/ (tidak butuh work/t305f_scratch/).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
W = ROOT / "work" / "t305f"
TRACKED = ROOT / "scripts" / "t305f_data.json"
FORMAT = "t305f-data/1"
CLIPS = ("test", "test_short", "Klip2", "klip3", "klip4", "klip5")

ATURAN = {
    "sumber": "scripts/t305_1b.py::vf_fill (prototipe T-305 1b, dinilai Rio); oracle tests/hole_fill_oracle.py",
    "kandidat": "komponen background 4-arah (konektivitas lubang [4])",
    "tepi_frame": "bbox x = 0, y = 0, x + w >= W, y + h >= H -> dilewati",
    "cincin": "dilasi 3x3 (8-arah) dikurangi komponen, dibaca dari peta masukan",
    "syarat": "SELURUH piksel cincin = grup face -> komponen diisi face",
    "batas_ukuran": None,
    "grup": "face (konstanta config.HOLE_FILL_GROUP)",
    "parameter": {"stabilize.hole_fill.enabled": True},
}
KEPUTUSAN = {
    "tempat": "PB: run_stabilize sesudah final_depth, sebelum write_groups (peta akhir sesudah kernel + pulau + mode); "
              "hanya stable/groups, depth_smooth tidak berubah",
    "algo_rev_3": "tidak ada; basi lewat stabilize_hash (section stabilize)",
    "ditolak": {"P-pre": "klip tanpa lubang ikut berubah (klip3), menutup lebih sedikit (23 -> 10 vs 23 -> 5), +40-47 ms/frame",
                "PA": "lubang sama dengan PB, tambah 2 frame gambar berubah (f10, f22)"},
    "lubang_tidak_terisi_diterima": "Q1 5 (f4-5, f28-30, cincin menyentuh baju) + definisi peta 4 (f2, f17, f32, f101; cincin +-51% face)",
    "video_penilaian": "klip5 penuh f0-f132",
}
# Penilaian Rio atas work/t305f/nilai/klip5_penuh_berdampingan*.mp4 (2026-10-19). Detik kerusakan poin 2 tidak dicatat (Rio) -> None;
# diperiksa ulang saat task "pengolahan khusus klip5".
PENILAIAN = {
    "sumber": "work/t305f/nilai/klip5_penuh_berdampingan.mp4 + _lambat.mp4 (2026-10-19)",
    "lingkaran_lubang_muka_berkurang_nyata": True,
    "kerusakan_visual_klip5": {"ada": True, "penyebab_menurut_rio": "proses lain, bukan VF", "detik": None,
                               "catatan": "tidak dicatat — periksa saat pengolahan khusus klip5",
                               "status": "di luar lingkup T-305f -> backlog 'pengolahan khusus klip5'"},
    "klip_lain_tidak_berubah_visual": True,
    "nilai_klip5": 3,
    "nilai_diterima": True,
    "keputusan": "T-305f DONE; stabilize.hole_fill.enabled tetap default true",
}
# Suite penuh Tahap 2 (2026-10-19), 6 potongan berurutan di foreground, 34 berkas test tepat sekali.
SUITE = {"lolos": 1550, "skip": 10, "gagal": 0, "kolom": ["potongan", "lolos", "skip", "gagal", "detik"],
         "potongan": [["cli, clip_identity, config, cut_diff, depth", 349, 1, 0, 24.3],
                      ["export, export_paper, export_strokes, ingest", 112, 0, 0, 105.5],
                      ["isolate, noise, paper, pick, presets, preview", 229, 2, 0, 29.5],
                      ["segment, segment_qc, stabilize, stabilize_hole_fill, stabilize_temporal, stage_common, target_mask, "
                       "temporal_metrics, track", 423, 7, 0, 48.9],
                      ["stylize (5 berkas)", 252, 0, 0, 170.9],
                      ["vectorize (5 berkas)", 185, 0, 0, 147.2]],
         "skip_rinci": "4 GPU / bobot SAM + 6 test klip nyata VF (sebelum run Tahap 3)",
         "sesudah_run_tahap3": {"berkas": ["test_stabilize_hole_fill.py", "test_preview.py", "test_clip_identity.py"],
                                "lolos": 154, "skip": 0, "gagal": 0}}
STAGE_RE = re.compile(r"\[(\w+)\] (\w+) ([\d.]+) s")


def jload(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def sim_part(s: dict) -> dict:
    keep = ("control_ok", "PB", "PA", "PR", "PB_vs_PR", "face_holes_drawn", "face_holes_small")
    out = {k: s[k] for k in keep}
    out["lubang_tidak_terisi_PB"] = s["left_by_PB_detail"]
    out["ms_median"] = {k: (v["median"] if v else None) for k, v in s["ms"].items()}
    return out


def lines_part(v: dict) -> dict:
    s = v["summary"]
    return {"png_berubah": s["png_differ"], "n_png_berubah": s["n_png_differ"], "geometri_berubah": s["geom_differ"],
            "geometri_berubah_di_luar_stable": s["outside_stable_changed"]["geom_differ"], "q1_lubang_prod_var": s["q1_holes"],
            "q1_sisa_frame": s["q1_frames_var"], "strok_per_tipe_prod_var": s["strokes_per_type"],
            "t_high_t_low_prod_var": [v["clip_stats"]["t_high"], v["clip_stats"]["t_low"]]}


def run_times(clip: str) -> dict:
    text = (W / "run" / f"{clip}.log").read_text(encoding="utf-8", errors="replace")
    line = [x for x in text.splitlines() if x.startswith("Waktu per stage:")][-1]
    return {f"[{a}] {b}": float(c) for a, b, c in STAGE_RE.findall(line)}


def build() -> dict:
    ringkas = jload(W / "ringkas.json")
    cek = jload(W / "cek_t3.json")
    muts = jload(W / "mutations.json")
    names = [k for k in muts if not k.startswith("kontrol")]
    k5 = cek["klip5"]
    tahap1 = {"sim": {c: sim_part(ringkas["sim"][c]) for c in CLIPS},
              "klip5_f0_f91": sim_part(ringkas["sim"]["klip5_f0_f91"]),
              "PB_sama_prototipe_klip5_f0_f91": ringkas["PB_equals_prototype_klip5_f0_f91"],
              "prototipe_vfscan_klip5": ringkas["prototype_vfscan"]["klip5"],
              "garis_gambar": {k: lines_part(v) for k, v in jload(W / "lines.json").items()},
              "PB_vs_render_prototipe_f0_f91": ringkas["lines"]["PB/klip5"]["vs_prototype_f0_f91"]}
    tahap3 = {"stabilize_hash": k5["stable_manifest"]["stabilize_hash"],
              "run": {c: run_times(c) for c in CLIPS},
              "lima_klip_identik": {c: {"groups": cek[c]["groups"], "depth_smooth": cek[c]["depth_smooth"], "png": cek[c]["png"],
                                        "svg": cek[c]["svg"], "out_svg": cek[c]["out_svg"], "contours_strokes": cek[c]["contours_strokes"],
                                        "mp4_identik": cek[c]["mp4_identical"], "contours_json_kunci_beda": cek[c]["contours_json_keys_differ"],
                                        "clip_stats": cek[c]["clip_stats"]} for c in CLIPS if c != "klip5"},
              "klip5": {"groups_berubah": k5["groups"], "depth_smooth_berubah": k5["depth_smooth"], "png_svg_berubah": k5["png"],
                        "geometri_berubah": k5["contours_geom"], "wilayah_px": k5["vf_regions_px"], "clip_stats": k5["clip_stats"],
                        "vs_PB_tahap1": k5["vs_PB_tahap1"]},
              "F1": {k: cek["F1"][k] for k in ("n_before", "n_after", "kategori", "lainnya", "per_folder")}
              | {"qc_hanya_created_utc": all(cek["F1"]["qc_created_utc_only"].values())}}
    f1_t1 = jload(W / "f1_after.json")["vs_before"]
    return {"format": FORMAT, "task": "T-305f", "aturan_vf": ATURAN, "keputusan": KEPUTUSAN, "penilaian_rio": PENILAIAN,
            "tahap1": tahap1 | {"F1": {"hilang": len(f1_t1["missing"]), "berubah": len(f1_t1["changed"]),
                                       "tambahan": len(f1_t1["extra"]), "identik": f1_t1["identical_existing"]}},
            "mutasi": {"n": len(names), "terbunuh": sum(muts[k]["rc"] == 1 for k in names),
                       "kontrol_lolos": muts[next(k for k in muts if k.startswith("kontrol"))]["rc"] == 0, "nama": names},
            "suite": SUITE, "tahap3": tahap3, "bahan": jload(W / "bahan.json")}


def dump(obj) -> str:
    return json.dumps(obj, indent=1, ensure_ascii=False) + "\n"


def main(argv: list[str]) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if argv == ["susun"]:
        text = dump(build())
        (W / "t305f_data.json").write_text(text, encoding="utf-8")
        print(f"work/t305f/t305f_data.json: {len(text.encode('utf-8'))} B")
        return 0
    if argv == ["cek"]:
        same = TRACKED.is_file() and TRACKED.read_bytes() == dump(build()).encode("utf-8")
        print(f"scripts/t305f_data.json byte-identik dengan susunan ulang: {same}")
        return 0 if same else 1
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
