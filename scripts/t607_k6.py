"""T-607 Tahap 3 bagian 3 (CPU; klip asli work/clips/* + out/ HANYA dibaca): bahan penilaian K6 (Klip2 + klip5 sebelum | sesudah
crop) + konfirmasi K2' (pratinjau klip11) di work/t607/nilai/.

    python scripts/t607_k6.py angka   # strok per tipe / frame, strok seluruhnya di area pita, QC, ukuran -> work/t607/k6_angka.json
    python scripts/t607_k6.py video   # K6_<klip>_berdampingan.mp4 + _0.5x.mp4 (kiri = sebelum out/, kanan = sesudah scratch t3/out/)
                                      # + salinan pratinjau klip11 -> work/t607/nilai/, dicek ffprobe -> work/t607/k6_video.json

Tata letak (konten tampil pada skala yang SAMA di kedua panel): Klip2 (pita kiri / kanan) = tinggi sama 1080 px; klip5 (pita atas /
bawah) = lebar sama 1080 px (ukuran asli), panel sesudah ditempatkan pada baris yang sama dengan area konten di panel sebelum.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLIPS = ROOT / "work" / "clips"
T3 = ROOT / "work" / "t607_scratch" / "t3"
NILAI = ROOT / "work" / "t607" / "nilai"
OUT = ROOT / "work" / "t607"
K6 = ("Klip2", "klip5")
TYPES = ("silhouette", "silhouette_hole", "group_boundary", "occlusion")
GAP_PX, GAP_COLOR, PAD_COLOR = 16, "0x808080", "white"
KLIP2_H = 1080
CRF = 18


def jload(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def jsave(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"  -> {p.relative_to(ROOT).as_posix()}")


def stroke_stats(work: Path, box: list[int] | None) -> dict:
    """Per frame: jumlah strok per tipe + titik; strok yang SELURUH titiknya di luar `box` (= di area pita, hanya ada sebelum crop)."""
    files = sorted((work / "contours").glob("frame_*.json"))
    per_type, n_pts, in_band, n_frames_band = Counter(), 0, 0, 0
    for f in files:
        d = jload(f)
        band_here = 0
        for s in d["strokes"]:
            per_type[s["type"]] += 1
            n_pts += len(s["points"])
            if box is not None:
                x0, y0, x1, y1 = box
                if all(not (x0 <= p[0] <= x1 + 1 and y0 <= p[1] <= y1 + 1) for p in s["points"]):
                    band_here += 1
        in_band += band_here
        n_frames_band += band_here > 0
    n = len(files)
    return {"frames": n, "strokes_per_frame": round(sum(per_type.values()) / n, 2),
            "per_type_per_frame": {t: round(per_type[t] / n, 2) for t in TYPES}, "points_per_frame": round(n_pts / n, 1),
            "strokes_all_in_band": in_band, "frames_with_band_strokes": n_frames_band}


def cmd_angka() -> None:
    res = {}
    for clip in K6:
        before, after = CLIPS / clip, T3 / "work" / "clips" / clip
        saran = jload(T3 / "work" / "crop" / clip / "saran.json")
        qa, qb = jload(before / "qc_report.json")["summary"], jload(after / "qc_report.json")["summary"]
        ma, mb = jload(before / "meta.json"), jload(after / "meta.json")
        ea, eb = jload(ROOT / "out" / f"{clip}.export.json"), jload(T3 / "out" / f"{clip}.export.json")
        res[clip] = {
            "kerja": {"sebelum": [ma["working_width"], ma["working_height"]], "sesudah": [mb["working_width"], mb["working_height"]]},
            "mp4": {"sebelum": [ea["output"]["width"], ea["output"]["height"], ea["output"]["frames"]],
                    "sesudah": [eb["output"]["width"], eb["output"]["height"], eb["output"]["frames"]]},
            "kotak_kerja_sebelum": saran["box_work"], "kotak_sumber": saran["box_source"],
            "qc_gagal": {"sebelum": [qa["n_fail"], qa["n_frames"]], "sesudah": [qb["n_fail"], qb["n_frames"]]},
            "strok": {"sebelum": stroke_stats(before, saran["box_work"]), "sesudah": stroke_stats(after, None)}}
        r = res[clip]
        print(f"{clip}: kerja {r['kerja']}, mp4 {r['mp4']}, QC gagal {r['qc_gagal']}\n  sebelum {r['strok']['sebelum']}\n"
              f"  sesudah {r['strok']['sesudah']}", flush=True)
    jsave(OUT / "k6_angka.json", res)


def probe(p: Path) -> dict:
    q = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
                        "stream=width,height,nb_read_frames,r_frame_rate,duration", "-of", "json", str(p)],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    s = json.loads(q.stdout)["streams"][0]
    return {"width": s["width"], "height": s["height"], "frames": int(s["nb_read_frames"]), "fps": s["r_frame_rate"],
            "duration_s": float(s.get("duration", 0) or 0)}


def graph(clip: str, slow: bool) -> str:
    tail = ",setpts=2.0*PTS" if slow else ""
    if clip == "Klip2":                                   # pita kiri / kanan → tinggi sama
        return (f"[0:v]scale=-2:{KLIP2_H}:flags=lanczos,pad=iw+{GAP_PX}:ih:0:0:color={GAP_COLOR}[a];"
                f"[1:v]scale=-2:{KLIP2_H}:flags=lanczos[b];[a][b]hstack=inputs=2{tail}[v]")
    s = jload(T3 / "work" / "crop" / clip / "saran.json")  # pita atas / bawah → lebar sama (ukuran asli), baris konten sejajar
    m = jload(CLIPS / clip / "meta.json")
    ea = jload(ROOT / "out" / f"{clip}.export.json")["output"]
    y = round(s["box_work"][1] * ea["height"] / m["working_height"])
    return (f"[0:v]pad=iw+{GAP_PX}:ih:0:0:color={GAP_COLOR}[a];[1:v]pad={ea['width']}:{ea['height']}:0:{y}:color={PAD_COLOR}[b];"
            f"[a][b]hstack=inputs=2{tail}[v]")


def cmd_video() -> None:
    NILAI.mkdir(parents=True, exist_ok=True)
    res = {"videos": {}, "pratinjau_klip11": []}
    for clip in K6:
        before, after = ROOT / "out" / f"{clip}.mp4", T3 / "out" / f"{clip}.mp4"
        for slow in (False, True):
            name = f"K6_{clip}_berdampingan{'_0.5x' if slow else ''}.mp4"
            dst = NILAI / name
            cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-i", str(before), "-i", str(after),
                   "-filter_complex", graph(clip, slow), "-map", "[v]", "-an", "-c:v", "libx264", "-crf", str(CRF),
                   "-preset", "medium", "-pix_fmt", "yuv420p", str(dst)]
            p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
            if p.returncode != 0:
                raise SystemExit(f"ffmpeg gagal untuk {name}: {p.stderr[-600:]}")
            res["videos"][name] = {"graph": graph(clip, slow), "sebelum": probe(before), "sesudah": probe(after), "hasil": probe(dst),
                                   "bytes": dst.stat().st_size}
            print(f"{name}: {res['videos'][name]['hasil']} ({dst.stat().st_size / 2**20:.1f} MiB)", flush=True)
    pdir = T3 / "work" / "crop" / "klip11" / "pratinjau"
    for f in jload(pdir / "pratinjau.json")["files"]:
        dst = NILAI / f"13_klip11_pratinjau_{f['file']}"
        shutil.copyfile(pdir / f["file"], dst)
        ok = hashlib.sha256(dst.read_bytes()).hexdigest() == f["sha256"]
        res["pratinjau_klip11"].append({"file": dst.name, "sha256_sama": ok})
        print(f"{dst.name}: sha256 sama {ok}", flush=True)
    jsave(OUT / "k6_video.json", res)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    {"angka": cmd_angka, "video": cmd_video}[sys.argv[1]]()
