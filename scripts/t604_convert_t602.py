"""T-604 (alat sekali pakai, F3): titik Rio dari T-602 (`scripts/t602_points.json`, arsip `t602-points-arsip/1` berisi satu objek
`t602-points/1` per klip) → format pick `rotoscope-target-points/1` di work/t604/t602_converted/<klip>.points.json.
TIDAK dipasang ke work/clips/<klip>/target/ (keputusan Rio 2026-10-16 no. 6; pemasangan diputuskan di T-605).

    python scripts/t604_convert_t602.py

Per klip: sha256 BYTE frame sekarang harus = `frame_sha256` T-602 (frame yang diklik Rio tidak berubah) → sidik PIKSEL
(`pick.frame_pixel_sha256`) dihitung dari frame itu; titik apa adanya (obj_id → id, label sama, koordinat frames/ penuh,
urutan klik; tanpa pembulatan ulang); kotak null; `display` dari T-602 (+ crop R2 yang tampil saat itu). Hasil divalidasi dengan
pemuat modul (`pick.load_points` + `check_doc_matches_clip` + status frame) dan dibandingkan titik demi titik dengan sumber.
Hanya MEMBACA work/clips/<klip>/ ; menulis hanya work/t604/t602_converted/ (di-ignore git).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rotoscope import pick  # noqa: E402
from rotoscope.ingest import FRAME_PATTERN, FRAMES_DIRNAME  # noqa: E402
from rotoscope.stage_common import clip_identity, load_frame_list, write_json_atomic  # noqa: E402

SRC = ROOT / "scripts" / "t602_points.json"
CLIPS = ROOT / "work" / "clips"
OUT = ROOT / "work" / "t604" / "t602_converted"
TOOL = "scripts/t604_convert_t602.py (dari scripts/t602_points.json, studi T-602)"


def convert(clip: str, d: dict) -> tuple[Path, dict]:
    if d.get("format") != "t602-points/1" or d.get("clip") != clip:
        raise SystemExit(f"{clip}: bukan objek t602-points/1 untuk klip ini")
    work = CLIPS / clip
    identity = clip_identity(work)
    _names, indices, w, h = load_frame_list(work)
    if [w, h] != d["frame_wh"]:
        raise SystemExit(f"{clip}: frame_wh T-602 {d['frame_wh']} != meta.json {[w, h]}")
    by_frame: dict[int, list[dict]] = {}
    for pr in d["prompts"]:
        by_frame.setdefault(int(pr["frame_index"]), []).append(pr)
    entries = []
    for f, prompts in sorted(by_frame.items()):
        fp = work / FRAMES_DIRNAME / (FRAME_PATTERN % f)
        byte_sha = hashlib.sha256(fp.read_bytes()).hexdigest()
        if any(pr["frame_sha256"] != byte_sha for pr in prompts):
            raise SystemExit(f"{clip} frame {f}: sha256 byte berbeda dengan saat Rio mengklik (frames/ berubah) — tidak dikonversi")
        targets = [{"id": int(pr["obj_id"]),
                    "points": [{"x": float(p["x"]), "y": float(p["y"]), "label": int(p["label"])} for p in pr["points"]],
                    "box": None} for pr in sorted(prompts, key=lambda r: r["obj_id"])]
        entries.append({"frame_index": f, "frame_file": FRAME_PATTERN % f, "frame_pixel_sha256": pick.frame_pixel_sha256(fp),
                        "updated_utc": d["created_utc"],
                        "display": {"scale": d["display_scale"], "screen_wh": d["screen_physical_wh"], "dpi_aware": None,
                                    "source": "T-602 scripts/t602_study.py pick",
                                    "crop_rule": d["crop_rule"], "crop_xyxy_incl": d["crop_xyxy_incl"]},
                        "targets": targets})
    doc = pick.new_doc(identity, w, h, entries)
    doc["tool"] = TOOL
    pick.validate_doc(doc)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{clip}.points.json"
    write_json_atomic(path, doc)
    # ── validasi ulang lewat pemuat modul (seperti pick / T-605 membacanya) ──
    back = pick.load_points(path)
    pick.check_doc_matches_clip(back, identity, w, h, path)
    status = pick.frame_statuses(back, work, set(indices), {})
    src_pts = {(int(pr["frame_index"]), int(pr["obj_id"])): [(p["x"], p["y"], p["label"]) for p in pr["points"]]
               for pr in d["prompts"]}
    out_pts = {(e["frame_index"], t["id"]): [(p["x"], p["y"], p["label"]) for p in t["points"]]
               for e in back["frames"] for t in e["targets"]}
    counts = {f"f{e['frame_index']} target {t['id']}": {"kiri": sum(p["label"] == 1 for p in t["points"]),
                                                       "kanan": sum(p["label"] == 0 for p in t["points"]),
                                                       "kotak": t["box"] is not None}
              for e in back["frames"] for t in e["targets"]}
    return path, {"file": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size,
                  "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "status": status,
                  "titik_sama_dengan_sumber": src_pts == out_pts, "isi": counts,
                  "frame_byte_sha256_cocok_T602": True, "meta_sha256": identity["meta_sha256"][:12]}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    arsip = json.loads(SRC.read_text(encoding="utf-8"))
    if arsip.get("format") != "t602-points-arsip/1":
        raise SystemExit(f"{SRC}: format {arsip.get('format')!r} bukan t602-points-arsip/1")
    res = {}
    for clip, item in arsip["points"].items():
        path, r = convert(clip, item["data"])
        res[clip] = r
        print(f"{clip}: {r['file']} ({r['bytes']} B) status {r['status']}, titik sama dengan sumber: "
              f"{r['titik_sama_dengan_sumber']}, {r['isi']}", flush=True)
    summary = OUT / "ringkasan.json"
    summary.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"-> {summary.relative_to(ROOT).as_posix()} (TIDAK dipasang ke work/clips/<klip>/target/)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
