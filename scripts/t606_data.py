"""T-606 (alat sekali pakai, F3): susun data T-606 yang berharga dari work/t606/*.json (keputusan T-606 no. 9).

    python scripts/t606_data.py susun    -> work/t606/t606_data.json (ISI untuk scripts/t606_data.json; berkas repo ditulis lewat
                                            Write oleh Claude Code, bukan oleh skrip ini)
    python scripts/t606_data.py cek      -> susun ulang di memori dan bandingkan BYTE dengan scripts/t606_data.json

Isi: parameter H, ukur klip3 (r, komponen gabungan, lubang, jarak tepi), metrik V0 / M2-H / M1 / d 8 / 12 / 20, tepi per d, run M1
(waktu / VRAM / NaN), analisis M1, penilaian Rio 1b + 1c, verifikasi produksi (bila ada), mutasi. Hanya MEMBACA work/t606/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
W = ROOT / "work" / "t606"
TRACKED = ROOT / "scripts" / "t606_data.json"
FORMAT = "t606-data/1"

PENILAIAN = {
    "1b": {"sumber": "work/t606/nilai/klip3_banding_V0_M2H_M1_f085-125*.mp4 (2026-10-17)",
           "V0": {"hanya_target": False, "garis_palsu": False, "tepi_terpotong": False, "nilai": 2},
           "M2-H": {"hanya_target": True, "garis_palsu": False, "tepi_terpotong": True, "frame_salah": 0, "nilai": 4,
                    "terbaik": True},
           "M1": {"hanya_target": True, "garis_palsu": False, "tepi_terpotong": False, "nilai": 3},
           "pergeseran_ambang_4_diterima": True},
    "1c": {"sumber": "work/t606/nilai/klip3_banding_d4_d8_d12_f085-125*.mp4 (dicatat 2026-10-19)",
           "d4": {"tepi_terpotong": False, "orang_lain_ikut": False, "nilai": 4, "terbaik": True},
           "d8": {"nilai": 3}, "d12": {"nilai": 2},
           "catatan": "penilaian 1c MENGGANTIKAN 'tepi terpotong ya' M2-H di 1b (render sama) -> M2-H d 4 lolos semua kriteria"},
}
TEPI_DEF = {"rim": "piksel F di luar M_f pada sinar normal yang berakhir <= 40 px (tepi seg target melewati masker)",
            "contact": "piksel F di luar M_f pada sinar disensor (F berlanjut >= 40 px; zona sentuh orang lain)",
            "cut_px": "rim di komponen gabungan yang tidak dipertahankan H (target terpotong)",
            "neighbor_kept_px": "contact yang dipertahankan H (orang lain ikut)",
            "kept_not_rim_px": "semua piksel dipertahankan di luar M_f yang bukan rim (batas atas orang lain ikut)"}
# Suite penuh T-606 Tahap 2 (2026-10-19), 6 kelompok berkas test berurutan di foreground (33 berkas, tepat sekali); kelompok 3
# diulang sesudah test_34 distabilkan (hasil sama).
SUITE = {"lolos": 1470, "skip": 4, "gagal": 0,
         "kelompok": {"1 cli/clip_identity/config/cut_diff/depth": [349, 1, 0, 24.0],
                      "2 export/export_paper/export_strokes/ingest": [112, 0, 0, 109.7],
                      "3 isolate/noise/paper/pick/presets/preview": [229, 2, 0, 28.6],
                      "4 segment/segment_qc/stabilize/stabilize_temporal/stage_common/target_mask/temporal_metrics/track":
                          [343, 1, 0, 51.7],
                      "5 stylize*": [252, 0, 0, 188.0], "6 vectorize*": [185, 0, 0, 157.9]},
         "kolom": ["lolos", "skip", "gagal", "detik"]}
AGG_KEYS = ("strokes_per_frame", "strokes_on_target_per_frame", "strokes_outside_md_per_frame", "silhouette_per_frame",
            "by_type_per_frame", "births_per_frame", "births_on_target_per_frame", "births_by_type_per_frame",
            "inside_per_frame", "new_inside_total", "px_per_frame")


def jl(name: str) -> dict | None:
    p = W / name
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def variant(agg: dict) -> dict:
    out = {k: agg[k] for k in AGG_KEYS if k in agg}
    lk = agg.get("leak_outside_scene")
    if lk:
        out["leak_outside_scene"] = {
            "n_frames_outside": lk["n_frames_outside"],
            **{k: lk[k]["n"] for k in ("stable_beda", "strok_geometri_beda", "strok_selain_oklusi_beda", "hanya_track_id_beda")},
            "occlusion_len_in_changed_frames": lk["occlusion_len_in_changed_frames"],
            "t_high": {k: v["t_high"] for k, v in lk["clip_stats"].items()},
            "t_low": {k: v["t_low"] for k, v in lk["clip_stats"].items()}}
    return out


def build() -> dict:
    u, m, m1c, t1c, mc, mut = (jl(n) for n in ("ukur.json", "metrik.json", "metrik_1c.json", "tepi_1c.json", "m1_cek.json",
                                               "mutations.json"))
    d = {"format": FORMAT, "task": "T-606", "klip": "klip3", "adegan": u["scene"],
         "keputusan_h": {"r_keep": 0.9, "r_drop": 0.1, "dilate_px": 4, "hole_fill_max_px": 0, "piece_min_px": 30,
                         "no_prompt": "keep", "mode": "M2-H (seg asli dibatasi masker sebelum kernel [3]); M1 tidak dipakai"},
         "penilaian_rio": PENILAIAN}
    d["ukur"] = {
        "mask_key_sha256": u["mask_state"]["key_sha256"], "island_min_px": u["island_min_px"],
        "komponen": {"n": u["r"]["n_comps"], "n_ge_island": u["r"]["n_comps_ge_island"],
                     "hist_r": [[b["r_lo"], b["r_hi"], b["n"]] for b in u["r"]["hist_count_all"] if b["n"]],
                     "per_pita_r": u["r"]["area_by_r_band"]},
        "gabungan_per_frame": {str(r["frame"]): r["potong_r"] for r in u["proposed_per_frame"]},
        "lubang": {k: u["holes"][k] for k in ("n_total", "per_frame", "area", "fg_cover", "n_fg_cover_ge_50pct")},
        "jarak_tepi": {k: u["rays"][k] for k in ("all", "r_mid_0.05_0.95", "r_ge_0.95")},
        "qc_bermasker": {k: u["qc"][k] for k in ("fail_frames_original", "fail_reasons_masked_scene")}}
    d["metrik"] = {"definisi": {"interior_erode_px": m["interior_erode_px"], "new_dist_px": m["new_dist_px"],
                                "inside_frac": m["inside_frac"], "wilayah_target": "M_d d 4 (tetap untuk semua varian)"},
                   "V0": variant(m["variants"]["V0"]["agg"]), "M2-H_d4": variant(m["variants"]["M2H"]["agg"]),
                   "M1": variant(m["variants"]["M1"]["agg"])}
    for k in ("M2H_d8", "M2H_d12", "M2H_d20"):
        d["metrik"][k.replace("M2H_", "M2-H_")] = variant(m1c["variants"][k]["agg"])
    d["tepi_per_d"] = {"definisi": TEPI_DEF, "sinar_komponen_gabungan": t1c["rays_potong"],
                       **{f"d{k}": {x: v["agg"][x] for x in ("rim_px", "cut_px", "neighbor_kept_px", "kept_not_rim_px",
                                                             "cut_pct_of_rim", "cut_by_group_pct", "cut_left_right_pct")}
                          | {"frames_cut_ge_100": len(v["agg"]["frames_cut_ge_100"])} for k, v in t1c["per_d"].items()}}
    r = mc["run"]
    d["m1"] = {"run": {k: r[k] for k in ("seg_frames", "finite_false", "finite_unknown", "total_s", "infer_s_p50",
                                         "peak_reserved_mib_max", "peak_allocated_mib_max", "nvidia_smi", "qc_m1_seg_41")}
               | {"seg_run_start": {k: r["seg_run_start"][k] for k in ("vram_free_before_load_mib", "vram_total_mib",
                                                                        "vram_free_after_load_mib", "load_s")},
                  "seg_wall_s": r["seg_run_end"]["wall_s"],
                  "langkah_s": {k: v["wall_s"] for k, v in r["steps"].items()}},
               "cakupan_stable": mc["coverage_stable"], "cakupan_seg_mentah": mc["coverage_raw_seg"],
               "label_grup_di_mf": mc["group_agreement_in_mf"]["same_group_pct"],
               "t_high": mc["t_high_analysis"]["clip_stats"], "nilai_scene_di_atas_t_high_v0": mc["t_high_analysis"]["scene_inner_values"]}
    t3 = jl("cek_t3.json")
    if t3:
        d["produksi_klip3"] = {"vs_prototipe_d4": t3["vs_prototipe_d4"], "stable_vs_sebelum": t3["stable_vs_sebelum"],
                               "oklusi_luar_adegan": {k: v for k, v in t3["oklusi_luar_adegan_vs_sebelum"].items()
                                                      if k not in ("frames_beda", "selain_oklusi_beda", "hanya_track_id_beda")}
                               | {"n_frames_beda": len(t3["oklusi_luar_adegan_vs_sebelum"]["frames_beda"]),
                                  "n_selain_oklusi_beda": len(t3["oklusi_luar_adegan_vs_sebelum"]["selain_oklusi_beda"]),
                                  "n_hanya_track_id_beda": len(t3["oklusi_luar_adegan_vs_sebelum"]["hanya_track_id_beda"])}}
    if mut:
        d["mutasi"] = {"n": len(mut) - 1, "terbunuh": sum(v["rc"] == 1 for k, v in mut.items() if not k.startswith("kontrol")),
                       "kontrol_lolos": next(v["rc"] == 0 for k, v in mut.items() if k.startswith("kontrol"))}
    d["suite"] = SUITE
    return d


LINE_MAX = 1800                       # Read menampilkan baris ≤ 2000 karakter apa adanya; penulisan ulang lewat Write aman


def _compact(v) -> str:
    return json.dumps(v, ensure_ascii=False, separators=(", ", ": "))


def _block(v, depth: int, ind: str) -> str:
    """Dict sampai kedalaman `depth` dipecah per key (satu key per baris); di bawahnya JSON ringkas satu baris. Baris yang
    masih > LINE_MAX dipecah satu tingkat lagi."""
    if isinstance(v, dict) and v and (depth > 0 or len(ind) + len(_compact(v)) > LINE_MAX):
        inner = ind + " "
        items = [f"{inner}{json.dumps(k, ensure_ascii=False)}: {_block(x, depth - 1, inner)}" for k, x in v.items()]
        return "{\n" + ",\n".join(items) + "\n" + ind + "}"
    return _compact(v)


def text(d: dict) -> str:
    return _block(d, 2, "") + "\n"


def main(argv: list[str]) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    cmd = argv[0] if argv else ""
    if cmd == "susun":
        s = text(build())
        (W / "t606_data.json").write_text(s, encoding="utf-8", newline="\n")
        print(f"work/t606/t606_data.json: {len(s.encode('utf-8'))} B")
        return 0
    if cmd == "cek":
        a = text(build()).encode("utf-8")
        b = TRACKED.read_bytes() if TRACKED.is_file() else b""
        print(f"{TRACKED.relative_to(ROOT).as_posix()}: {len(b)} B; susun ulang {len(a)} B; byte-identik = {a == b}")
        return 0 if a == b else 1
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
