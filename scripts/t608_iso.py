"""T-608 Tahap 1b-1 (alat, F3; CPU): isolasi ulang klip5 di scratch. docs/04 "Keputusan T-608 sesudah Tahap 1 (2026-10-28)".

    python scripts/t608_iso.py config [iso|iso_uji]   # config scratch = configs/default.yaml kecuali `paths` (dibandingkan SESUDAH load)
    python scripts/t608_iso.py siapkan [iso|iso_uji]  # salin produksi klip5 -> work/t608_scratch/<set>/work/; cek sha256 + bukan hard link
    python scripts/t608_iso.py adegan     # adegan persis [1b] (isolate.detect_scenes, cut_diff config) + frame klik -> work/t608/iso_adegan.json
    python scripts/t608_iso.py adegan_lingkup  # lingkup f0–53 + klik Rio f0, f22, f40 (keputusan lingkup) -> work/t608/iso_adegan_lingkup.json
    python scripts/t608_iso.py uji [lingkup]       # titik SINTETIS di salinan UJI -> `isolate --dry-run` + `isolate` tanpa --yes (CLI asli, CPU)
    python scripts/t608_iso.py cek_pick [lingkup]  # `pick --frame N --dry-run` per frame klik di scratch iso (perintah Rio persis) + `isolate --dry-run`
    python scripts/t608_iso.py arsip iso1          # target/ + isolate/ scratch iso -> work/t608/arsip_<nama>/ (sha256 dicek), lalu sumber DIHAPUS

`lingkup` = frame klik dari `iso_adegan_lingkup.json`, keluaran `iso_*_lingkup.json`. Tanpa `lingkup` = rencana 1b-1 awal (f0–62, 6 klik).

Produksi (work/clips/klip5, work/crop/klip5) hanya DIBACA. Salinan = berkas baru (shutil.copy2), BUKAN hard link: pick / ingest / stage
bisa menulis di folder klip. Semua subprocess CLI dijalankan dengan CUDA_VISIBLE_DEVICES=-1 (tanpa GPU). Hasil cek -> work/t608/iso_*.json.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"                 # pengaman: Tahap 1b-1 tanpa GPU

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rotoscope import isolate as iso  # noqa: E402
from rotoscope import pick  # noqa: E402
from rotoscope.config import load_pipeline  # noqa: E402
from rotoscope.ingest import FRAME_PATTERN  # noqa: E402
from rotoscope.stage_common import clip_identity  # noqa: E402

KLIP = "klip5"
VIDEO = r"samples\klip5.mp4"                               # sama persis dengan PANDUAN_PICK (cwd = root repo)
PROD_CLIP = ROOT / "work" / "clips" / KLIP
PROD_CROP = ROOT / "work" / "crop" / KLIP
DEFAULT_CFG = ROOT / "configs" / "default.yaml"
SCR = ROOT / "work" / "t608_scratch"
SETS = {   # nama: (folder scratch, isi folder klip yang disalin)
    "iso": (SCR / "iso", ("meta.json", "qc_report.json", "frames", "seg", "depth")),
    "iso_uji": (SCR / "iso_uji", ("meta.json", "frames")),
}
CFG_REL = {k: f"work\\t608_scratch\\{k}\\config.yaml" for k in SETS}
OUT = ROOT / "work" / "t608"
CROWD_LAST = 62                                            # kerumunan f0–f62 (keputusan Rio no. 2); tanpa klik di f63+
MID_MIN_LEN = 30                                           # adegan > 30 frame → tambah satu frame tengah
SCOPE_LAST = 53                                            # lingkup T-608 f0–53 (keputusan lingkup 2026-10-28)
KLIK_RIO = (0, 22, 40)                                     # klik yang diputuskan Rio (keputusan lingkup no. 3), BUKAN dihitung ulang
SYNTH_TOOL = "SINTETIS scripts/t608_iso.py uji (BUKAN klik Rio)"
VARIANT = {"": "iso_adegan.json", "lingkup": "iso_adegan_lingkup.json"}


def jsave(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"  -> {p.relative_to(ROOT).as_posix()}")


def file_sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cfg_path(name: str) -> Path:
    return SETS[name][0] / "config.yaml"


def clip_dir(name: str) -> Path:
    return SETS[name][0] / "work" / "clips" / KLIP


def crop_dir(name: str) -> Path:
    return SETS[name][0] / "work" / "crop" / KLIP


def as_plain(obj):
    if dataclasses.is_dataclass(obj):
        return {f.name: as_plain(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, dict):
        return {str(k): as_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [as_plain(v) for v in obj]
    if isinstance(obj, Path):
        return obj.as_posix()
    return obj


def flat(d, prefix=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flat(v, f"{prefix}{k}."))
    else:
        out[prefix[:-1]] = d
    return out


def run_cli(args: list[str]) -> dict:
    cmd = [str(ROOT / "venv" / "Scripts" / "python.exe"), "-m", "rotoscope", *args]
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "PYTHONIOENCODING": "utf-8"})
    return {"cmd": "python -m rotoscope " + " ".join(args), "rc": p.returncode, "stdout": p.stdout, "stderr": p.stderr}


# ── config ─────────────────────────────────────────
def cmd_config(names: tuple[str, ...] = tuple(SETS), suffix: str = "") -> None:
    base = flat(as_plain(load_pipeline(DEFAULT_CFG)))
    res = {}
    for name in names:
        cur = flat(as_plain(load_pipeline(cfg_path(name))))
        diff = sorted(k for k in set(base) | set(cur) if base.get(k) != cur.get(k))
        res[name] = {"config": CFG_REL[name], "sha256": file_sha(cfg_path(name)), "beda_dari_default": diff,
                     "paths": {k: cur[k] for k in cur if k.startswith("paths.")}}
        ok = all(k.startswith("paths.") for k in diff)
        res[name]["hanya_paths"] = ok
        print(f"{name}: beda dari default = {diff} -> hanya paths: {ok}")
        if not ok:
            raise SystemExit(f"{name}: config scratch berbeda di luar paths — berhenti")
    jsave(OUT / f"iso_config{suffix}.json", res)


# ── salin ──────────────────────────────────────────
def copy_tree(src: Path, dst: Path) -> list[tuple[Path, Path]]:
    pairs = []
    if src.is_file():
        pairs.append((src, dst))
    else:
        for p in sorted(src.rglob("*")):
            if p.is_file():
                pairs.append((p, dst / p.relative_to(src)))
    for s, d in pairs:
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(s, d)
    return pairs


def cmd_siapkan(names: tuple[str, ...] = tuple(SETS), suffix: str = "") -> None:
    res = {}
    for name in names:
        base, items = SETS[name]
        cd, rd = clip_dir(name), crop_dir(name)
        if cd.exists() or rd.exists():
            raise SystemExit(f"{name}: {cd} / {rd} sudah ada — salinan tidak ditimpa (hapus dulu bila memang harus diulang)")
        pairs = []
        for it in items:
            pairs += copy_tree(PROD_CLIP / it, cd / it)
        pairs += copy_tree(PROD_CROP, rd)
        bad, linked, n_bytes = [], [], 0
        for s, d in pairs:
            if file_sha(s) != file_sha(d):
                bad.append(d.relative_to(ROOT).as_posix())
            if os.stat(d).st_nlink != 1 or os.path.samefile(s, d):
                linked.append(d.relative_to(ROOT).as_posix())
            n_bytes += d.stat().st_size
        res[name] = {"clip_dir": cd.relative_to(ROOT).as_posix(), "crop_dir": rd.relative_to(ROOT).as_posix(),
                     "isi": list(items) + ["crop/klip5/*"], "berkas": len(pairs), "bytes": n_bytes,
                     "per_item": {it: sum(1 for s, _ in pairs if s.is_relative_to(PROD_CLIP / it) or s == PROD_CLIP / it)
                                  for it in items} | {"crop": sum(1 for s, _ in pairs if s.is_relative_to(PROD_CROP))},
                     "sha256_beda": bad, "hard_link_atau_sama": linked,
                     "meta_sha256": file_sha(cd / "meta.json")}
        print(f"{name}: {len(pairs)} berkas, {n_bytes / 2**20:.1f} MiB; sha256 beda {len(bad)}; hard link {len(linked)}")
        if bad or linked:
            raise SystemExit(f"{name}: salinan tidak valid — berhenti")
    res["meta_sha256_produksi"] = file_sha(PROD_CLIP / "meta.json")
    jsave(OUT / f"iso_salinan{suffix}.json", res)


# ── adegan + frame klik ────────────────────────────
def scenes(name: str) -> dict:
    cfg = load_pipeline(cfg_path(name))
    clip = iso.load_clip(clip_dir(name))
    cuts, shots = iso.detect_scenes(clip, cfg.stabilize.temporal.cut_diff)
    crowd = [s for s in shots if s[0] <= CROWD_LAST]
    plan = []
    for lo, hi in crowd:
        top = min(hi, CROWD_LAST)
        clicks = [lo]
        if hi - lo + 1 > MID_MIN_LEN:
            mid = (lo + top) // 2
            if mid != lo:
                clicks.append(mid)
        plan.append({"adegan": [lo, hi], "panjang": hi - lo + 1, "klik": clicks, "melewati_f62": hi > CROWD_LAST,
                     "frame_kriteria_no4": [lo, top]})
    klik = sorted({f for p in plan for f in p["klik"]})
    gpu_frames = sum(p["panjang"] for p in plan)
    crit = sorted({f for p in plan for f in range(p["frame_kriteria_no4"][0], p["frame_kriteria_no4"][1] + 1)})
    return {"cut_diff": cfg.stabilize.temporal.cut_diff, "n_frames": len(clip.indices), "cuts": cuts,
            "shots": [list(s) for s in shots], "adegan_kerumunan": plan, "frame_klik": klik,
            "klik_di_f63_plus": [f for f in klik if f > CROWD_LAST],
            "frame_kriteria_no4": {"n": len(crit), "rentang": iso.ranges(crit)},
            "sam_frame": gpu_frames,
            "perkiraan_s": round(gpu_frames * iso.SECONDS_PER_FRAME_ESTIMATE + iso.LOAD_SECONDS_ESTIMATE, 1),
            "perkiraan_dasar": f"{iso.SECONDS_PER_FRAME_ESTIMATE} s/frame + muat {iso.LOAD_SECONDS_ESTIMATE} s (isolate.py)"}


def cmd_adegan() -> None:
    d = scenes("iso")
    d_prod = {"catatan": "pembanding: adegan dihitung dari frames/ PRODUKSI (baca saja)"}
    cfg = load_pipeline(DEFAULT_CFG)
    cuts_p, shots_p = iso.detect_scenes(iso.load_clip(PROD_CLIP), cfg.stabilize.temporal.cut_diff)
    d_prod.update({"cuts": cuts_p, "sama_dengan_scratch": cuts_p == d["cuts"] and [list(s) for s in shots_p] == d["shots"]})
    d["pembanding_produksi"] = d_prod
    for p in d["adegan_kerumunan"]:
        print(f"adegan f{p['adegan'][0]}–f{p['adegan'][1]} ({p['panjang']} frame): klik {p['klik']}"
              + ("  (adegan melewati f62)" if p["melewati_f62"] else ""))
    print(f"frame klik: {d['frame_klik']}; klik di f63+: {d['klik_di_f63_plus']}; frame SAM {d['sam_frame']} ≈ {d['perkiraan_s']} s; "
          f"frame kriteria no. 4: {d['frame_kriteria_no4']}; adegan scratch = produksi: {d_prod['sama_dengan_scratch']}")
    if d["klik_di_f63_plus"]:
        raise SystemExit("ada frame klik di f63+ — melanggar keputusan Rio no. 2")
    jsave(OUT / "iso_adegan.json", d)


def cmd_adegan_lingkup() -> None:
    """Lingkup f0–53 + klik Rio f0, f22, f40: setiap adegan yang beririsan dengan f0–53 harus berklik; tanpa klik di f54+."""
    cfg = load_pipeline(cfg_path("iso"))
    clip = iso.load_clip(clip_dir("iso"))
    cuts, shots = iso.detect_scenes(clip, cfg.stabilize.temporal.cut_diff)
    in_scope = [s for s in shots if s[0] <= SCOPE_LAST]
    klik = list(KLIK_RIO)
    plan = [{"adegan": list(s), "panjang": s[1] - s[0] + 1, "klik": [f for f in klik if s[0] <= f <= s[1]],
             "melewati_f53": s[1] > SCOPE_LAST, "frame_dinilai": [s[0], min(s[1], SCOPE_LAST)]} for s in in_scope]
    unclicked = [p["adegan"] for p in plan if not p["klik"]]
    sam = sum(p["panjang"] for p in plan if p["klik"])
    judged = sorted({f for p in plan if p["klik"] for f in range(p["frame_dinilai"][0], p["frame_dinilai"][1] + 1)})
    d = {"lingkup": [0, SCOPE_LAST], "klik_rio": klik, "cut_diff": cfg.stabilize.temporal.cut_diff, "cuts": cuts,
         "adegan_lingkup": plan, "adegan_tanpa_klik": unclicked, "frame_klik": klik,
         "klik_di_luar_lingkup": [f for f in klik if f > SCOPE_LAST],
         "frame_dinilai_no4": {"n": len(judged), "rentang": iso.ranges(judged), "ambang": 49},
         "sam_frame": sam, "sam_frame_di_luar_lingkup": sum(max(0, p["adegan"][1] - SCOPE_LAST) for p in plan if p["klik"]),
         "perkiraan_s": round(sam * iso.SECONDS_PER_FRAME_ESTIMATE + iso.LOAD_SECONDS_ESTIMATE, 1),
         "tanpa_prompt_perkiraan": len(clip.indices) - sam}
    for p in plan:
        print(f"adegan f{p['adegan'][0]}–f{p['adegan'][1]} ({p['panjang']} frame): klik {p['klik']}"
              + ("  (adegan melewati f53: masker f54+ ada, tidak dinilai)" if p["melewati_f53"] else ""))
    print(f"frame klik {klik}; adegan tanpa klik {unclicked}; klik di luar lingkup {d['klik_di_luar_lingkup']}; frame SAM {sam} "
          f"(di luar lingkup {d['sam_frame_di_luar_lingkup']}) ≈ {d['perkiraan_s']} s; frame dinilai no. 4 {d['frame_dinilai_no4']}")
    if unclicked or d["klik_di_luar_lingkup"] or len(judged) != SCOPE_LAST + 1:
        raise SystemExit("klik Rio tidak menutup lingkup f0–53 / ada klik di luar lingkup — berhenti")
    jsave(OUT / "iso_adegan_lingkup.json", d)


# ── uji isolate dengan titik SINTETIS (salinan uji) ─
def synthetic_doc(cd: Path, frames: list[int]) -> dict:
    meta = json.loads((cd / "meta.json").read_text(encoding="utf-8"))
    w, h = meta["working_width"], meta["working_height"]
    entries = []
    for f in frames:
        entries.append({"frame_index": f, "frame_file": FRAME_PATTERN % f,
                        "frame_pixel_sha256": pick.frame_pixel_sha256(cd / "frames" / (FRAME_PATTERN % f)),
                        "updated_utc": "SINTETIS",
                        "display": {"scale": 1.0, "screen_wh": [0, 0], "work_area_wh": [0, 0], "dpi_aware": False},
                        "targets": [{"id": 1, "points": [{"x": round(w / 2, 1), "y": round(h / 2, 1), "label": 1},
                                                         {"x": 10.0, "y": 10.0, "label": 0}], "box": None}]})
    doc = pick.new_doc(clip_identity(cd), w, h, entries)
    doc["tool"] = SYNTH_TOOL
    return doc


def tree_state(cd: Path) -> dict:
    return {n: (cd / n).exists() for n in (iso.ISOLATE_DIRNAME, iso.PARTIAL_DIRNAME, iso.OLD_DIRNAME)}


def cmd_uji(variant: str = "") -> None:
    d = json.loads((OUT / VARIANT[variant]).read_text(encoding="utf-8"))
    suffix = f"_{variant}" if variant else ""
    cd = clip_dir("iso_uji")
    pts = cd / pick.TARGET_DIRNAME / pick.POINTS_FILENAME
    if pts.exists():
        raise SystemExit(f"{pts} sudah ada — uji tidak menimpanya")
    pick.write_points(pts, synthetic_doc(cd, d["frame_klik"]))
    res = {"catatan": "titik SINTETIS (1 klik kiri di tengah + 1 klik kanan di pojok per frame klik) di salinan UJI; bukan klik Rio",
           "points_sha256": file_sha(pts), "frame_klik": d["frame_klik"]}
    res["dry_run"] = run_cli(["isolate", VIDEO, "--dry-run", "--config", CFG_REL["iso_uji"]])
    res["sesudah_dry_run"] = tree_state(cd)
    res["tanpa_yes"] = run_cli(["isolate", VIDEO, "--config", CFG_REL["iso_uji"]])
    res["sesudah_tanpa_yes"] = tree_state(cd)
    for k in ("dry_run", "tanpa_yes"):
        r = res[k]
        print(f"== {r['cmd']} -> rc {r['rc']}\n{r['stdout']}{r['stderr']}")
    print("isolate/ / partial / old sesudah:", res["sesudah_dry_run"], res["sesudah_tanpa_yes"])
    res["lolos"] = (res["dry_run"]["rc"] == 0 and res["tanpa_yes"]["rc"] == 1 and "--yes" in res["tanpa_yes"]["stderr"]
                    and not any(res["sesudah_dry_run"].values()) and not any(res["sesudah_tanpa_yes"].values()))
    print("lolos:", res["lolos"])
    jsave(OUT / f"iso_uji{suffix}.json", res)


# ── cek perintah pick Rio (dry-run) di scratch iso ──
def cmd_cek_pick(variant: str = "") -> None:
    d = json.loads((OUT / VARIANT[variant]).read_text(encoding="utf-8"))
    suffix = f"_{variant}" if variant else ""
    cd = clip_dir("iso")
    tdir = cd / pick.TARGET_DIRNAME
    if tdir.exists():
        raise SystemExit(f"{tdir} sudah ada — cek tidak dijalankan (mungkin klik Rio)")
    res = {"pick": [], "isolate_dry_run": None}
    for f in d["frame_klik"]:
        r = run_cli(["pick", VIDEO, "--frame", str(f), "--dry-run", "--config", CFG_REL["iso"]])
        png = tdir / pick.DRYRUN_DIRNAME / (FRAME_PATTERN % f)
        r.update({"dryrun_png": png.is_file(), "ingest_disebut": "ingest" in (r["stdout"] + r["stderr"]).lower(),
                  "points_json_ada": (tdir / pick.POINTS_FILENAME).exists()})
        res["pick"].append(r)
        print(f"== {r['cmd']} -> rc {r['rc']}; png {r['dryrun_png']}; ingest disebut {r['ingest_disebut']}; points.json "
              f"{r['points_json_ada']}")
    print(res["pick"][0]["stdout"] + res["pick"][0]["stderr"])
    r = run_cli(["isolate", VIDEO, "--dry-run", "--config", CFG_REL["iso"]])
    res["isolate_dry_run"] = r
    print(f"== {r['cmd']} -> rc {r['rc']}\n{r['stdout']}{r['stderr']}")
    left = sorted(p.relative_to(tdir).as_posix() for p in tdir.rglob("*") if p.is_file()) if tdir.exists() else []
    only_dry = all(x.startswith(pick.DRYRUN_DIRNAME + "/") for x in left)
    res["target_isi_sebelum_hapus"] = left
    if only_dry and tdir.exists():
        shutil.rmtree(tdir)                                # hanya kanvas dry-run; scratch bersih untuk klik Rio
    res["target_dihapus"] = only_dry and not tdir.exists()
    res["lolos"] = (all(x["rc"] == 0 and x["dryrun_png"] and not x["ingest_disebut"] and not x["points_json_ada"]
                        for x in res["pick"]) and r["rc"] == 0 and res["target_dihapus"] and not any(tree_state(cd).values()))
    print(f"target/ dihapus: {res['target_dihapus']} ({len(left)} berkas dry-run); lolos: {res['lolos']}")
    jsave(OUT / f"iso_cek_pick{suffix}.json", res)


def cmd_arsip(name: str) -> None:
    """Percobaan isolate selesai dipakai: titik + hasil disalin ke arsip, sha256 dicek, baru sumber dihapus (scratch siap klik baru)."""
    cd = clip_dir("iso")
    dst = OUT / f"arsip_{name}"
    if dst.exists():
        raise SystemExit(f"{dst} sudah ada — arsip tidak ditimpa")
    srcs = [cd / pick.TARGET_DIRNAME, cd / iso.ISOLATE_DIRNAME, cd / iso.PARTIAL_DIRNAME, cd / iso.OLD_DIRNAME]
    srcs = [s for s in srcs if s.exists()]
    files = [(p, dst / p.relative_to(cd)) for s in srcs for p in sorted(s.rglob("*")) if p.is_file()]
    for s, d in files:
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(s, d)
    rows = [{"berkas": s.relative_to(cd).as_posix(), "bytes": s.stat().st_size, "sha256": file_sha(s),
             "sama": file_sha(s) == file_sha(d)} for s, d in files]
    if not files or not all(r["sama"] for r in rows):
        raise SystemExit("arsip kosong / sha256 tidak sama — sumber TIDAK dihapus")
    for s in srcs:
        shutil.rmtree(s)
    left = [s.name for s in srcs if s.exists()]
    res = {"arsip": dst.relative_to(ROOT).as_posix(), "dari": cd.relative_to(ROOT).as_posix(), "berkas": len(rows),
           "bytes": sum(r["bytes"] for r in rows), "semua_sha256_sama": True, "dihapus": [s.name + "/" for s in srcs],
           "tersisa": left, "points_json": next((r for r in rows if r["berkas"] == "target/points.json"), None),
           "per_berkas": rows}
    print(f"arsip {res['arsip']}: {res['berkas']} berkas, {res['bytes']} B, sha256 sama semua; dihapus {res['dihapus']}; tersisa {left}")
    (dst / "arsip.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"  -> {(dst / 'arsip.json').relative_to(ROOT).as_posix()}")


def main(argv: list[str]) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if argv[:1] == ["arsip"] and len(argv) == 2 and argv[1].isalnum():
        cmd_arsip(argv[1])
        return 0
    if not argv:
        print(__doc__)
        return 2
    cmd, rest = argv[0], argv[1:]
    if cmd in ("config", "siapkan") and all(r in SETS for r in rest):
        names = tuple(rest) or tuple(SETS)
        (cmd_config if cmd == "config" else cmd_siapkan)(names, "_" + "_".join(rest) if rest else "")
    elif cmd in ("uji", "cek_pick") and rest in ([], ["lingkup"]):
        (cmd_uji if cmd == "uji" else cmd_cek_pick)(rest[0] if rest else "")
    elif cmd == "adegan" and not rest:
        cmd_adegan()
    elif cmd == "adegan_lingkup" and not rest:
        cmd_adegan_lingkup()
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
