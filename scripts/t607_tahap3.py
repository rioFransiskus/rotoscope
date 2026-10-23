"""T-607 Tahap 3 (CPU, scratch; klip asli work/clips/* + out/ HANYA dibaca): [1] lewat CLI produksi (`python -m rotoscope ingest`
/ `crop`) dengan config scratch → bukti K1', K4, K5, K7 + angka K2' klip11, dan cek kotak diterima Klip2 / klip5 sebelum run GPU.

    python scripts/t607_tahap3.py satu    # 6 klip lama + klip6–klip11 ke work/t607_scratch/t3/ (+ t3b untuk K7) -> work/t607/tahap3.json
    python scripts/t607_tahap3.py cek     # root t3cek: ingest → crop --accept → ingest ber-crop (Klip2, klip5) -> work/t607/tahap3_cek.json
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLIPS_DIR = ROOT / "work" / "clips"
SCR = ROOT / "work" / "t607_scratch"
OUT = ROOT / "work" / "t607"
CFG = {k: f"work/t607_scratch/{k}/config.yaml" for k in ("t3", "t3b", "t3cek")}
OLD = ("test", "test_short", "Klip2", "klip3", "klip4", "klip5")
NEW = ("klip6", "klip7", "klip8", "klip9", "klip10", "klip11")
NO_BAND_NEW = ("klip6", "klip7", "klip8", "klip9")          # label Rio: tanpa pita (K1')
T602 = ROOT / "scripts" / "t602_data.json"
K7_CLIPS = ("Klip2", "klip5", "klip11", "klip8")
K2_KLIP11_LINES = {"top": [388, 461]}                      # persetujuan Tahap 1c h (dicek dengan angka)
K2_KLIP11_ALT = [0, 462, 720, 594]
STUDY_PSRC = {"Klip2": SCR / "crop" / "Klip2_outward", "klip5": SCR / "crop" / "klip5_outward"}   # Tahap 1a (ukur)


def cli(*args: str) -> tuple[int, str, str, float]:
    t0 = time.perf_counter()
    p = subprocess.run([sys.executable, "-m", "rotoscope", *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    return p.returncode, p.stdout, p.stderr, round(time.perf_counter() - t0, 2)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def jload(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def jsave(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"  -> {p.relative_to(ROOT).as_posix()}")


def source_of(clip: str) -> Path:
    if clip in OLD:
        return Path(json.loads((CLIPS_DIR / clip / "meta.json").read_text(encoding="utf-8"))["source_path"])
    return ROOT / "samples" / f"{clip}.mp4"


def frames(d: Path) -> list[Path]:
    return sorted((d / "frames").glob("frame_*.png"))


def same_frames(a: list[Path], b: list[Path]) -> dict:
    eq = sum(x.read_bytes() == y.read_bytes() for x, y in zip(a, b))
    return {"n_a": len(a), "n_b": len(b), "bytes_equal": eq, "all_equal": len(a) == len(b) == eq}


def crop_files(root: str, clip: str) -> dict[str, str]:
    d = SCR / root / "work" / "crop" / clip
    return {p.relative_to(d).as_posix(): sha(p) for p in sorted(d.rglob("*")) if p.is_file()} if d.is_dir() else {}


def summarize(root: str, clip: str, folder: str | None = None) -> dict:
    wd = SCR / root / "work" / "clips" / (folder or clip)
    cd = SCR / root / "work" / "crop" / clip
    meta, det, saran = jload(wd / "meta.json"), jload(cd / "deteksi.json"), jload(cd / "saran.json")
    prev = jload(cd / "pratinjau" / "pratinjau.json")
    return {"frame_count": meta and meta["frame_count"], "working_wh": meta and [meta["working_width"], meta["working_height"]],
            "meta_keys_extra": meta and [k for k in meta if k in ("trim", "crop")],
            "deteksi": det and {"pita_terdeteksi": det["pita_terdeteksi"], "box_work": det["box_work"], "n_frames": det["n_frames"],
                                "detect_frame_size": det["detect_frame_size"]},
            "saran": saran and {k: saran[k] for k in ("box_work", "box_source", "band_px", "overlay_px", "cut_px", "hints")},
            "pratinjau": prev and [f["file"] for f in prev["files"]], "kotak": (cd / "kotak.json").is_file()}


def cmd_satu() -> None:
    t602 = json.loads(T602.read_text(encoding="utf-8"))
    res: dict = {"config": CFG["t3"], "clips": {}}
    for clip in OLD + NEW:
        src = source_of(clip)
        rc, out, err, s = cli("ingest", str(src), "--config", CFG["t3"])
        row = {"source": str(src), "rc": rc, "s": s, **summarize("t3", clip),
               "pesan": [ln for ln in out.splitlines() if ln.startswith(("[1] saran", "  ", "PERINGATAN"))][-9:],
               "stderr": err.strip().splitlines()[-3:]}
        if clip in OLD:
            real, wd = CLIPS_DIR / clip, SCR / "t3" / "work" / "clips" / clip
            row["meta_identik_klip_asli"] = (wd / "meta.json").read_bytes() == (real / "meta.json").read_bytes()
            row["frames_vs_klip_asli"] = same_frames(frames(wd), frames(real))
        if clip in ("Klip2", "klip5"):
            box = t602["crop"]["clips"][clip]["crop_xyxy_incl"]
            row["K4_saran_box_eq_t602"] = row["saran"] is not None and row["saran"]["box_work"] == box
            row["t602_box"] = box
        res["clips"][clip] = row
        sar = row["saran"]
        print(f"{clip}: rc {rc} {s} s, {row['frame_count']} f {row['working_wh']}; pita {row['deteksi'] and row['deteksi']['pita_terdeteksi']}"
              f"; saran {sar and sar['box_work']} sumber {sar and sar['box_source']} petunjuk {sar and sar['hints'] and sar['hints']['lines']}"
              + (f"; meta identik {row['meta_identik_klip_asli']}, frame {row['frames_vs_klip_asli']['bytes_equal']}/"
                 f"{row['frames_vs_klip_asli']['n_b']}" if clip in OLD else ""), flush=True)
    # K5: klip5 0–3,8333 s → 92 frame = frames/ klip asli f0–f91
    src5 = source_of("klip5")
    rc, out, err, s = cli("ingest", str(src5), "--start", "0", "--end", "3.8333", "--config", CFG["t3"])
    wd5 = SCR / "t3" / "work" / "clips" / "klip5@0-3.833"
    meta5 = jload(wd5 / "meta.json")
    res["K5_klip5_trim"] = {"rc": rc, "s": s, "folder": wd5.name, "frame_count": meta5 and meta5["frame_count"],
                            "trim": meta5 and meta5.get("trim"), "crop_key": meta5 and "crop" in meta5,
                            "frames_vs_klip_asli_f0_91": same_frames(frames(wd5), frames(CLIPS_DIR / "klip5")[:92])}
    print(f"K5 klip5 trim: {res['K5_klip5_trim']}", flush=True)
    # K7: saran / deteksi / pratinjau byte-deterministik (root kedua)
    det = {}
    for clip in K7_CLIPS:
        rc, *_ = cli("ingest", str(source_of(clip)), "--config", CFG["t3b"])
        a, b = crop_files("t3", clip), crop_files("t3b", clip)
        det[clip] = {"rc": rc, "files": sorted(a), "identik": a == b and bool(a)}
    res["K7_determinisme"] = det
    print(f"K7: {det}", flush=True)
    # K1' + K2' + ringkasan kriteria
    c = res["clips"]
    k11 = c["klip11"]["saran"] or {}
    h11 = k11.get("hints") or {}
    res["kriteria"] = {
        "K1_no_saran_klip6_9": {k: c[k]["saran"] is None and c[k]["deteksi"]["pita_terdeteksi"] is False for k in NO_BAND_NEW},
        "K2_klip10_box_source": (c["klip10"]["saran"] or {}).get("box_source"),
        "K2_klip11_hint_lines": h11.get("lines"), "K2_klip11_alt_box_source": h11.get("alt_box_source"),
        "K2_klip11_lines_ok": h11.get("lines") == K2_KLIP11_LINES, "K2_klip11_alt_ok": h11.get("alt_box_source") == K2_KLIP11_ALT,
        "K2_klip11_pesan_alt": any("--box 0 462 720 594" in ln for ln in c["klip11"]["pesan"]),
        "K4_meta_frames_identik": {k: c[k]["meta_identik_klip_asli"] and c[k]["frames_vs_klip_asli"]["all_equal"] for k in OLD},
        "K4_saran_t602": {k: c[k]["K4_saran_box_eq_t602"] for k in ("Klip2", "klip5")},
        "K4_tanpa_saran_4_klip": {k: c[k]["saran"] is None for k in ("test", "test_short", "klip3", "klip4")},
        "K5": res["K5_klip5_trim"]["frame_count"] == 92 and res["K5_klip5_trim"]["frames_vs_klip_asli_f0_91"]["all_equal"],
        "K7": all(v["identik"] for v in det.values()),
    }
    res["scratch_bytes"] = {r: sum(p.stat().st_size for p in (SCR / r).rglob("*") if p.is_file()) for r in ("t3", "t3b")}
    print(json.dumps(res["kriteria"], indent=1, ensure_ascii=False))
    jsave(OUT / "tahap3.json", res)


def cmd_cek() -> None:
    res: dict = {"config": CFG["t3cek"], "clips": {}}
    for clip in ("Klip2", "klip5"):
        src = source_of(clip)
        r1 = cli("ingest", str(src), "--config", CFG["t3cek"])
        r2 = cli("crop", str(src), "--accept", "--config", CFG["t3cek"])
        r3 = cli("ingest", str(src), "--config", CFG["t3cek"])
        wd = SCR / "t3cek" / "work" / "clips" / clip
        meta = jload(wd / "meta.json")
        kotak = jload(SCR / "t3cek" / "work" / "crop" / clip / "kotak.json")
        res["clips"][clip] = {"rc": [r1[0], r2[0], r3[0]], "s": [r1[3], r2[3], r3[3]], "kotak_box_source": kotak and kotak["box_source"],
                              "kotak_mode": kotak and kotak["mode"], "working_wh": [meta["working_width"], meta["working_height"]],
                              "frame_count": meta["frame_count"], "meta_crop": meta.get("crop"),
                              "frames_vs_studi_psrc": same_frames(frames(wd), sorted(STUDY_PSRC[clip].glob("frame_*.png"))),
                              "crop_stdout": r2[1].strip().splitlines()[-3:]}
        print(f"{clip}: {res['clips'][clip]}", flush=True)
    jsave(OUT / "tahap3_cek.json", res)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    {"satu": cmd_satu, "cek": cmd_cek}[sys.argv[1]]()
