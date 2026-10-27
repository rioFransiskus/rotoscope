"""T-603 Tahap 1 (alat ukur, F3): biaya ingest + deteksi crop per sumber dan disk per frame per folder.

    python scripts/t603_ukur.py disk            # baca-saja: ukuran berkas per folder klip produksi + out/ (stat, tanpa isi PNG/MP4)
    python scripts/t603_ukur.py ingest          # CPU: ingest + deteksi crop di SCRATCH (work/t603_scratch/ukur/) — tidak menyentuh work/clips/ atau work/crop/

Hasil JSON: work/t603/ukur_disk.json, work/t603/ukur_ingest.json (keduanya di-ignore git).
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rotoscope import ingest as ing  # noqa: E402
from rotoscope.config import load_pipeline  # noqa: E402

OUT = ROOT / "work" / "t603"
SCRATCH = ROOT / "work" / "t603_scratch" / "ukur"
CLIPS = ROOT / "work" / "clips"
SOURCES = ["Klip2", "klip3", "klip4", "klip5", "test", "test_short"]      # = 6 klip produksi (work/clips/)
FOLDERS = ["frames", "seg", "depth", "isolate", "stable", "contours", "strokes"]


def size_of(p: Path) -> tuple[int, int]:
    if p.is_file():
        return 1, p.stat().st_size
    if not p.is_dir():
        return 0, 0
    files = [f for f in p.rglob("*") if f.is_file()]
    return len(files), sum(f.stat().st_size for f in files)


def disk() -> dict:
    res = {}
    for clip in sorted(p for p in CLIPS.iterdir() if p.is_dir()):
        meta = json.loads((clip / "meta.json").read_text(encoding="utf-8"))
        n = int(meta["frame_count"])
        row = {"frame_count": n, "working_wh": [meta["working_width"], meta["working_height"]], "folders": {}}
        for f in FOLDERS:
            files, size = size_of(clip / f)
            row["folders"][f] = {"files": files, "bytes": size, "bytes_per_frame": round(size / n, 1) if n else None}
        res[clip.name] = row
    out_mp4 = {}
    for clip in res:
        mp4 = ROOT / "out" / f"{clip}.mp4"
        svg_files, svg_bytes = size_of(ROOT / "out" / "svg" / clip)
        out_mp4[clip] = {"mp4_bytes": mp4.stat().st_size if mp4.is_file() else 0, "svg_files": svg_files,
                         "svg_bytes": svg_bytes}
    return {"clips": res, "out": out_mp4}


def ingest_cost() -> dict:
    cfg = load_pipeline(ROOT / "configs" / "default.yaml")
    res = {}
    if SCRATCH.exists():
        shutil.rmtree(SCRATCH)
    for name in SOURCES:
        video = ROOT / "samples" / f"{name}.mp4"
        wd, cd = SCRATCH / name / "plain", SCRATCH / name / "crop"
        t0 = time.perf_counter()
        plain = ing.ingest(video, work_dir=wd, log=lambda m: None)
        t_plain = time.perf_counter() - t0
        t0 = time.perf_counter()
        cold = ing.ingest(video, work_dir=SCRATCH / name / "crop_wd", crop_dir=cd, crop_cfg=cfg.ingest.crop,
                          log=lambda m: None)
        t_cold = time.perf_counter() - t0
        t0 = time.perf_counter()
        warm = ing.ingest(video, work_dir=SCRATCH / name / "crop_wd", crop_dir=cd, crop_cfg=cfg.ingest.crop,
                          log=lambda m: None)
        t_warm = time.perf_counter() - t0
        saran = (cd / "saran.json").is_file()
        res[name] = {"frames": plain["frame_count"], "wh": [plain["working_width"], plain["working_height"]],
                     "ingest_plain_s": round(t_plain, 2), "ingest_crop_cold_s": round(t_cold, 2),
                     "ingest_crop_warm_s": round(t_warm, 2), "saran": saran,
                     "deteksi": json.loads((cd / "deteksi.json").read_text(encoding="utf-8")).get("pita_terdeteksi"),
                     "frames_cold": cold["frame_count"], "frames_warm": warm["frame_count"]}
        print(name, res[name], flush=True)
    return res


def main(argv: list[str]) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    OUT.mkdir(parents=True, exist_ok=True)
    if argv == ["disk"]:
        data = disk()
        (OUT / "ukur_disk.json").write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
        for clip, row in data["clips"].items():
            print(clip, row["frame_count"], {f: v["bytes_per_frame"] for f, v in row["folders"].items()}, data["out"][clip])
        return 0
    if argv == ["ingest"]:
        data = ingest_cost()
        (OUT / "ukur_ingest.json").write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
