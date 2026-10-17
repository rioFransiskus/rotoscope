"""T-605 Tahap 4 (alat sekali pakai, F3): susun isi `scripts/t605_data.json` dari berkas hasil di work/t605/ (baca-saja).

    python scripts/t605_data.py tulis <path>   -> tulis JSON ke <path> (DI LUAR repo, mis. folder scratchpad sesi)
    python scripts/t605_data.py cek            -> scripts/t605_data.json byte-identik dengan hasil susun ulang?

Berkas repo `scripts/t605_data.json` ditulis lewat Write (aturan berkas repo); skrip ini hanya menyusun isi + mengecek.
Sumber: work/t605/{metrik,cek_run,tahap3_gpu,scenes,mutations}.json, work/t605/arsip_klip5_target/arsip.json,
work/clips/klip3/target/points.json, work/t604/t602_converted/klip3.points.json. Penilaian Rio + diagnosis = docs/04
"Keputusan akhir T-605".
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
T605 = ROOT / "work" / "t605"
REPO_FILE = ROOT / "scripts" / "t605_data.json"
ACTIVE_PX = 720 * 668                       # area video aktif klip5 (letterbox y 344–1011, T-602 R2)
FRAME_PX = 720 * 1280


def jl(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def dump(obj, level: int = 0) -> str:
    """JSON: objek bertingkat ber-indent 1, daftar skalar dalam satu baris."""
    pad, pad1 = " " * level, " " * (level + 1)
    if isinstance(obj, dict):
        if not obj:
            return "{}"
        items = [f"{pad1}{json.dumps(k, ensure_ascii=False)}: {dump(v, level + 1)}" for k, v in obj.items()]
        return "{\n" + ",\n".join(items) + "\n" + pad + "}"
    if isinstance(obj, list):
        if all(not isinstance(v, (dict, list)) for v in obj):
            return json.dumps(obj, ensure_ascii=False)
        return "[\n" + ",\n".join(pad1 + dump(v, level + 1) for v in obj) + "\n" + pad + "]"
    return json.dumps(obj, ensure_ascii=False)


def per_frame(t: dict, frames: list[int]) -> dict:
    return {"frame_awal": frames[0], "frame_akhir": frames[-1],
            "luas_px": [t["luas_per_frame"][str(i)] for i in frames],
            "skor_logit": [t["skor_per_frame"][str(i)] for i in frames]}


def summary(m: dict, frames: list[int], with_frames: bool) -> dict:
    t = m["per_target"]["1"]
    out = {"status": {k: m["frames"][k] for k in ("target", "kosong", "tanpa_prompt")},
           "kosong_ranges": m["frames"]["kosong_ranges"], "tanpa_prompt_ranges": m["frames"]["tanpa_prompt_ranges"],
           "luas_px": t["luas_px"], "iou_t_t1": t["iou_t_t1"], "lompatan_luas_gt_20pct": t["lompatan_luas_gt_20pct"],
           "skor_min": t["skor_min"], "n_skor_le_0": len(t["skor_le_0"]),
           "jobs": [{k: j[k] for k in ("shot", "target", "prompt_frames", "n_infer", "infer_s")} for j in m["jobs"]]}
    if with_frames:
        out["per_frame"] = per_frame(t, frames)
    return out


def coverage(t: dict, lo: int, hi: int) -> list:
    v = [t["luas_per_frame"][str(i)] / ACTIVE_PX for i in range(lo, hi + 1)]
    return [round(min(v), 3), round(max(v), 3)]


def build() -> dict:
    met, cek, gpu = jl(T605 / "metrik.json"), jl(T605 / "cek_run.json"), jl(T605 / "tahap3_gpu.json")
    scenes, muts, arsip = jl(T605 / "scenes.json"), jl(T605 / "mutations.json"), jl(T605 / "arsip_klip5_target" / "arsip.json")
    p3 = ROOT / "work" / "clips" / "klip3" / "target" / "points.json"
    p602 = ROOT / "work" / "t604" / "t602_converted" / "klip3.points.json"
    f3, f5 = list(range(85, 126)), list(range(0, 92))
    hasil = {"klip3": ("produksi", met["produksi"]["klip3"], f3, True),
             "klip5": ("produksi", met["produksi"]["klip5"], f5, True),
             "V-a": ("klip3, titik Rio apa adanya (determinisme)", met["varian"]["V-a"], f3, False),
             "V-b": ("klip3, titik T-602 (15 kiri + 4 kanan)", met["varian"]["V-b"], f3, True),
             "V-c": ("klip5, kotak f64 dibuang", met["varian"]["V-c"], f5, True),
             "V-d": ("klip5, hanya klik f64", met["varian"]["V-d"], f5, True)}
    metrik = {}
    for name, (what, m, frames, wf) in hasil.items():
        metrik[name] = {"isi": what, **summary(m, frames, wf)}
    metrik["V-a"]["per_frame"] = "= klip3 (41 / 41 PNG byte-identik)"
    run = {}
    for name in hasil:
        c = cek[name]
        run[name] = {"wall_s": c["wall_s"], "key_s": c["key_s"], "scenes_s": c["scenes_s"], "load_s": c["load_s"],
                     "infer_s": c["infer_s"], "vram_mib": c["vram"], "ram_mib": c["ram"],
                     "nvidia_smi_puncak_mib": (gpu.get(name) or {}).get("nvidia_smi_peak_mib"),
                     "wall_pembungkus_gpu_s": (gpu.get(name) or {}).get("wall_s"), "nonfinite_frames": c["nonfinite_frames"],
                     "device": c["device"], "gpu": c["gpu"], "precision": c["precision"]}
    det = {k: v for k, v in met["determinisme_V-a_vs_produksi"].items() if k not in ("iou_per_target", "beda_per_frame")}
    banding = {k: {kk: v[kk] for kk in ("iou_median", "iou_min", "status_sama", "png_byte_identik")}
               for k, v in met["banding"].items()}
    t5, tc, td = (met["produksi"]["klip5"]["per_target"]["1"], met["varian"]["V-c"]["per_target"]["1"],
                  met["varian"]["V-d"]["per_target"]["1"])
    return {
        "format": "t605-data/1",
        "catatan": "Data T-605 ([1b] isolasi target, SAM 2.1 hiera tiny) yang disimpan sebelum work/t605_scratch/ dihapus. Disusun "
                   "scripts/t605_data.py dari work/t605/*.json; isi masker klip5 + varian tidak disimpan (dihapus). Luas = px dari "
                   "frame 720 x 1280 (921 600 px); skor = logit skor objek SAM; IoU t/t-1 hanya di adegan berklik.",
        "model": {"model_id": "facebook/sam2.1-hiera-tiny", "revision": "de431c4043854a71d8101e17995dfe596bf101a5",
                  "license": "apache-2.0", "precision": "fp32", "libs": cek["klip3"]["libs"]},
        "adegan": {k: {"cut": scenes["clips"][k]["detected"], "adegan": scenes["clips"][k]["shots"],
                       "klik": scenes["clips"][k]["clicked_frames"], "dilacak": scenes["clips"][k]["tracked_shots"]}
                   for k in ("klip3", "klip5")},
        "metrik": metrik,
        "run": run,
        "determinisme_V-a_vs_produksi_klip3": {**det, "toleransi": "status sama, IoU per target >= 0,995, piksel beda <= 0,1% luas"},
        "banding_varian": banding,
        "kotak_f64_klip5": met["kotak_f64"],
        "penilaian_rio": {
            "klip3": "masker mengikuti target yang benar di semua frame",
            "klip5": "semua salah",
            "klip5_detik_2_6_3_2_zoom": "ya (orang memenuhi layar; masker seluruh gambar di f63-f77 benar)",
            "klip3_V-b_vs_produksi": "lebih baik",
            "klip5_V-c_vs_produksi": "sama buruknya",
            "klip5_V-d_sebelum_f64": "tidak benar",
            "catatan": "klip5 punya banyak masalah video, jadi harus diolah terpisah sebelum masuk model"},
        "keputusan": "T-605 DONE, lolos berdasarkan klip3; klip5 dikecualikan (kriteria awal mencakup klip5)",
        "diagnosis_klip5": {
            "f0_f2": {"masker": "ada", "cakupan_area_aktif": coverage(t5, 0, 2)},
            "f3_f62": {"masker": "kosong (target hilang, skor objek < 0)", "skor_min": min(t5["skor_per_frame"][str(i)] for i in range(3, 63))},
            "f63_f77": {"masker": "seluruh gambar = zoom (BENAR menurut Rio)", "cakupan_area_aktif": coverage(t5, 63, 77)},
            "f78_f91": {"cakupan_area_aktif": coverage(t5, 78, 91)},
            "kotak_f64": "luas 480 950 px = area video aktif 480 960 px (letterbox 720 x 668); benar saat zoom -> peringatan kotak besar di pick TIDAK dibuat",
            "dugaan": "frame kondisi f64 (zoom) mengacaukan memori frame awal (mode offline memakai semua frame kondisi, termasuk "
                      "sesudahnya) — BELUM terbukti; varian 'hanya f0' belum diuji",
            "V-c_cakupan_f64_f84": coverage(tc, 64, 84), "V-d_cakupan_f15_f80": coverage(td, 15, 80),
            "skor_cut_f64": 0.0584, "frame_kembar_f13_f15": "skor cut 0,0000 / 0,0001 (sumber 60 fps)"},
        "titik": {
            "klip3": {"berkas": "work/clips/klip3/target/points.json (tetap; bahan uji T-606)", "sha256": hashlib.sha256(p3.read_bytes()).hexdigest(),
                      "isi": "f85: target 1, 16 kiri / 0 kanan, tanpa kotak"},
            "klip5": {"berkas": "diarsipkan ke work/t605/arsip_klip5_target/ lalu target/ klip5 dihapus (salinan ter-track: scripts/t605_points_rio.json)",
                      **{k: {"bytes": v["bytes"], "sha256": v["sha256"], "isi": v["isi"]} for k, v in arsip["berkas"].items()}},
            "klip3_t602_V-b": {"berkas": "work/t604/t602_converted/klip3.points.json", "sha256": hashlib.sha256(p602.read_bytes()).hexdigest(),
                               "isi": "f85: target 1, 15 kiri / 4 kanan, tanpa kotak"}},
        "mutasi": {"n": len(muts) - 1, "menggagalkan_test": sum(v["rc"] == 1 for k, v in muts.items() if not k.startswith("kontrol")),
                   "kontrol_lolos": next(v["rc"] == 0 for k, v in muts.items() if k.startswith("kontrol"))},
    }


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    text = dump(build()) + "\n"
    if sys.argv[1:2] == ["tulis"] and len(sys.argv) == 3:
        out = Path(sys.argv[2])
        if ROOT in out.resolve().parents:
            print("tulis ke luar repo (berkas repo ditulis lewat Write)")
            return 2
        out.write_text(text, encoding="utf-8", newline="\n")
        print(f"{out}: {len(text.encode('utf-8'))} B")
        return 0
    if sys.argv[1:] == ["cek"]:
        repo = REPO_FILE.read_bytes() if REPO_FILE.is_file() else b""
        same = repo == text.encode("utf-8")
        print(f"{REPO_FILE.relative_to(ROOT).as_posix()}: {len(repo)} B, byte-identik dengan susun ulang: {same}")
        return 0 if same else 1
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
