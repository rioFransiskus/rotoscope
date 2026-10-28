"""T-608 Tahap 4a (alat, F3; CPU): bukti trim klip5 `--end 2.25`, arsip produksi, cek titik iso2. docs/04 "Keputusan akhir T-608 (2026-10-28)".

    python scripts/t608_tahap4.py siapkan   # scratch BARU work/t608_scratch/trim4a/: crop klip5 produksi disalin (sha256, bukan hard link)
                                            # + config scratch iso disalin (paths lalu disunting dengan Edit, BUKAN skrip)
    python scripts/t608_tahap4.py ingest    # config = default kecuali paths (di dalam scratch); `ingest samples\\klip5.mp4 --end 2.25`
    python scripts/t608_tahap4.py banding   # (a) 54 frame scratch vs produksi f0–53 (sha256 berkas + sidik piksel pick) + meta;
                                            # (c) sidik titik iso2 vs frame produksi + scratch; titik -> scratch; `isolate --dry-run` dan
                                            # `run` TANPA --yes (berhenti sebelum GPU) di scratch; frame + titik dicek ulang sesudahnya
    python scripts/t608_tahap4.py arsip     # (b) out/klip5.mp4 (+ manifest export), out/svg/klip5/, work/clips/klip5/meta.json
                                            #     -> work/t608/arsip_produksi_sebelum/ (sha256 salinan = sumber)

Sesudah "Keputusan folder produksi klip5 (2026-10-28)" (izin A1; config produksi, TANPA --config scratch):
    python scripts/t608_tahap4.py a1        # `ingest samples\\klip5.mp4 --end 2.25` -> work/clips/klip5@0-2.25/; 54 frame = produksi klip5
                                            # f0–53 = scratch trim4a (sha256 + piksel); crop lama dipakai; meta beda hanya frame_count + trim
    python scripts/t608_tahap4.py a2        # SALIN titik iso2 -> work/clips/klip5@0-2.25/target/; sha256, sidik f0/f22/f40,
                                            # check_doc_matches_clip, `isolate … --end 2.25 --dry-run`
    python scripts/t608_tahap4.py a3        # `run samples\\klip5.mp4 --end 2.25` TANPA --yes (terbukti berhenti di [1b] sebelum GPU, 4a (d));
                                            # sesudahnya isolate/ / seg/ / depth/ / out tidak ada, titik sama, frame 54 / 54 sama

Produksi hanya DIBACA (arsip = salinan), kecuali a1 / a2 / a3 yang menulis HANYA di work/clips/klip5@0-2.25/. Semua subprocess CLI dengan
CUDA_VISIBLE_DEVICES=-1. Hasil -> work/t608/t4a_*.json, work/t608/t4a2_*.json.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"                 # pengaman: Tahap 4a tanpa GPU

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import t608_iso as t  # noqa: E402
from rotoscope import cli, pick  # noqa: E402
from rotoscope.config import load_pipeline  # noqa: E402
from rotoscope.export import manifest_path_for  # noqa: E402
from rotoscope.ingest import FRAME_PATTERN  # noqa: E402
from rotoscope.stage_common import clip_identity  # noqa: E402

END_S = 2.25                                               # keputusan akhir T-608 no. 3 (f0–53)
TRIM = ["--end", "2.25"]
N_EXPECT = 54
SET = t.SCR / "trim4a"
CFG = SET / "config.yaml"
CFG_REL = "work\\t608_scratch\\trim4a\\config.yaml"
POINTS_ISO2 = t.clip_dir("iso") / pick.TARGET_DIRNAME / pick.POINTS_FILENAME
POINTS_ISO2_SHA = "9380ba65"                               # titik "pratinjau OK" (docs/04 "Keputusan T-608 1b-2")
ARSIP = t.OUT / "arsip_produksi_sebelum"


def rel(p: Path) -> str:
    return p.relative_to(ROOT).as_posix()


def trim_dir(cfg) -> Path:
    """Folder klip yang dipakai CLI untuk `--end 2.25` (fungsi CLI yang sama: clip_work_dir + requested_trim)."""
    return cli.clip_work_dir(cfg, Path(t.VIDEO), cli.requested_trim(None, END_S))


def scratch_cfg():
    cfg = load_pipeline(CFG)
    base = t.flat(t.as_plain(load_pipeline(t.DEFAULT_CFG)))
    cur = t.flat(t.as_plain(cfg))
    diff = sorted(k for k in set(base) | set(cur) if base.get(k) != cur.get(k))
    inside = all(Path(p).resolve().is_relative_to(SET.resolve()) for p in (cfg.paths.work_dir, cfg.paths.out_dir))
    info = {"config": CFG_REL, "sha256": t.file_sha(CFG), "beda_dari_default": diff,
            "paths": {k: cur[k] for k in cur if k.startswith("paths.")}, "paths_di_scratch": inside}
    if not all(k.startswith("paths.") for k in diff) or not inside:
        raise SystemExit(f"config scratch tidak aman (beda {diff}; paths di scratch: {inside}) — berhenti")
    return cfg, info


def frame_rows(wd: Path, ref: Path = t.PROD_CLIP) -> list[dict]:
    rows = []
    for i in range(N_EXPECT):
        n = FRAME_PATTERN % i
        p, s = ref / "frames" / n, wd / "frames" / n
        ok = s.is_file()
        rows.append({"frame": i, "ada": ok, "bytes_sama": ok and t.file_sha(p) == t.file_sha(s),
                     "piksel_sama": ok and pick.frame_pixel_sha256(p) == pick.frame_pixel_sha256(s)})
    return rows


def frame_summary(wd: Path, ref: Path = t.PROD_CLIP) -> dict:
    rows = frame_rows(wd, ref)
    names = {FRAME_PATTERN % i for i in range(N_EXPECT)}
    present = sorted(p.name for p in (wd / "frames").glob("*.png"))
    return {"n_scratch": len(present), "tambahan": [x for x in present if x not in names],
            "bytes_sama": sum(r["bytes_sama"] for r in rows), "piksel_sama": sum(r["piksel_sama"] for r in rows),
            "beda": [r["frame"] for r in rows if not (r["bytes_sama"] and r["piksel_sama"])]}


# ── siapkan ────────────────────────────────────────
def cmd_siapkan() -> None:
    if SET.exists():
        raise SystemExit(f"{rel(SET)} sudah ada — scratch BARU tidak ditimpa")
    rd = SET / "work" / "crop" / t.KLIP
    pairs = t.copy_tree(t.PROD_CROP, rd)
    shutil.copy2(t.cfg_path("iso"), CFG)
    bad = [rel(d) for s, d in pairs if t.file_sha(s) != t.file_sha(d)]
    linked = [rel(d) for s, d in pairs if os.stat(d).st_nlink != 1 or os.path.samefile(s, d)]
    res = {"scratch": rel(SET), "crop": rel(rd), "berkas_crop": len(pairs), "sha256_beda": bad, "hard_link_atau_sama": linked,
           "config_dari": rel(t.cfg_path("iso")), "catatan": "paths config disunting dengan Edit sesudah langkah ini"}
    print(f"crop: {len(pairs)} berkas; sha256 beda {len(bad)}; hard link {len(linked)}; config disalin -> {CFG_REL}")
    if bad or linked:
        raise SystemExit("salinan tidak valid — berhenti")
    t.jsave(t.OUT / "t4a_siapkan.json", res)


# ── ingest ─────────────────────────────────────────
def cmd_ingest() -> None:
    cfg, info = scratch_cfg()
    wd = trim_dir(cfg)
    if wd.exists():
        raise SystemExit(f"{rel(wd)} sudah ada — ingest bukti tidak diulang di folder lama")
    r = t.run_cli(["ingest", t.VIDEO, *TRIM, "--config", CFG_REL])
    print(f"== {r['cmd']} -> rc {r['rc']}\n{r['stdout']}{r['stderr']}")
    t.jsave(t.OUT / "t4a_ingest.json", {"config": info, "folder": rel(wd), "folder_nama": wd.name, "ingest": r})


# ── banding ────────────────────────────────────────
def cmd_banding() -> None:
    cfg, info = scratch_cfg()
    wd = trim_dir(cfg)
    res: dict = {"config": info, "folder_scratch": rel(wd), "folder_nama": wd.name}
    res["frame"] = frame_summary(wd)
    meta_p = json.loads((t.PROD_CLIP / "meta.json").read_text(encoding="utf-8"))
    meta_s = json.loads((wd / "meta.json").read_text(encoding="utf-8"))
    res["meta_key_beda"] = {k: {"produksi": meta_p.get(k), "scratch": meta_s.get(k)}
                            for k in sorted(set(meta_p) | set(meta_s)) if meta_p.get(k) != meta_s.get(k)}
    res["meta_sama"] = {k: meta_s.get(k) for k in ("crop", "working_width", "working_height", "fps") if meta_p.get(k) == meta_s.get(k)}
    fr = res["frame"]
    print(f"folder {wd.name}: {fr['n_scratch']} frame; bytes sama {fr['bytes_sama']}/{N_EXPECT}; piksel sama "
          f"{fr['piksel_sama']}/{N_EXPECT}; beda {fr['beda']}; tambahan {fr['tambahan']}")
    print(f"meta key beda: {sorted(res['meta_key_beda'])}; sama: {sorted(res['meta_sama'])}")
    for k, v in res["meta_key_beda"].items():
        print(f"  {k}: produksi {json.dumps(v['produksi'], ensure_ascii=False)[:160]} | scratch "
              f"{json.dumps(v['scratch'], ensure_ascii=False)[:160]}")
    ok_a = fr["n_scratch"] == N_EXPECT and fr["bytes_sama"] == N_EXPECT and fr["piksel_sama"] == N_EXPECT and not fr["tambahan"]
    res["a_lolos"] = ok_a
    if not ok_a:
        t.jsave(t.OUT / "t4a_banding.json", res)
        raise SystemExit("(a) GAGAL: frame trim tidak sama dengan produksi f0–53 — BERHENTI")

    # (c) titik iso2
    psha = t.file_sha(POINTS_ISO2)
    if not psha.startswith(POINTS_ISO2_SHA):
        raise SystemExit(f"{rel(POINTS_ISO2)} sha256 {psha[:12]} ≠ {POINTS_ISO2_SHA} — berhenti")
    doc = pick.load_points(POINTS_ISO2)
    pts = []
    for e in doc["frames"]:
        f = e["frame_file"]
        pts.append({"frame": e["frame_index"], "frame_pixel_sha256": e["frame_pixel_sha256"],
                    "produksi_klip5": pick.frame_pixel_sha256(t.PROD_CLIP / "frames" / f) == e["frame_pixel_sha256"],
                    "scratch_trim": pick.frame_pixel_sha256(wd / "frames" / f) == e["frame_pixel_sha256"],
                    "kiri": sum(p["label"] == 1 for tg in e["targets"] for p in tg["points"]),
                    "kanan": sum(p["label"] == 0 for tg in e["targets"] for p in tg["points"]),
                    "kotak": sum(tg.get("box") is not None for tg in e["targets"])})
    cocok = {}
    for name, d in (("produksi_klip5", t.PROD_CLIP), ("scratch_trim", wd)):
        m = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        try:
            pick.check_doc_matches_clip(doc, clip_identity(d), m["working_width"], m["working_height"], POINTS_ISO2)
            cocok[name] = "ok"
        except pick.PickError as e:
            cocok[name] = str(e)
    res["titik"] = {"berkas": rel(POINTS_ISO2), "sha256": psha, "bytes": POINTS_ISO2.stat().st_size,
                    "clip_di_titik": doc["clip"], "frame_size": doc["frame_size"], "per_frame": pts,
                    "check_doc_matches_clip": cocok}
    for p in pts:
        print(f"titik f{p['frame']}: kiri {p['kiri']} kanan {p['kanan']} kotak {p['kotak']}; sidik = produksi klip5 "
              f"{p['produksi_klip5']}, = scratch trim {p['scratch_trim']}")
    print(f"check_doc_matches_clip: {cocok}")

    # titik -> scratch trim; perintah CLI tanpa GPU
    dst = wd / pick.TARGET_DIRNAME / pick.POINTS_FILENAME
    if dst.exists():
        raise SystemExit(f"{rel(dst)} sudah ada — tidak ditimpa")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(POINTS_ISO2, dst)
    res["titik_scratch"] = {"berkas": rel(dst), "sha256_sama": t.file_sha(dst) == psha}
    cmds = [["isolate", t.VIDEO, *TRIM, "--dry-run", "--config", CFG_REL],
            ["run", t.VIDEO, *TRIM, "--restart-from", "ingest", "--config", CFG_REL],
            ["run", t.VIDEO, *TRIM, "--config", CFG_REL]]
    res["perintah"] = []
    for c in cmds:
        r = t.run_cli(c)
        r["sesudah"] = {"isolate_dirs": t.tree_state(wd), "titik_sha256_sama": dst.is_file() and t.file_sha(dst) == psha,
                        "frame": frame_summary(wd)}
        res["perintah"].append(r)
        print(f"\n== {r['cmd']} -> rc {r['rc']}\n{r['stdout']}{r['stderr']}")
        s = r["sesudah"]
        print(f"   sesudah: isolate/ {s['isolate_dirs']}; titik sama {s['titik_sha256_sama']}; frame piksel sama "
              f"{s['frame']['piksel_sama']}/{N_EXPECT}")
    t.jsave(t.OUT / "t4a_banding.json", res)


# ── arsip produksi ─────────────────────────────────
def cmd_arsip() -> None:
    if ARSIP.exists():
        raise SystemExit(f"{rel(ARSIP)} sudah ada — arsip tidak ditimpa")
    mp4 = ROOT / "out" / "klip5.mp4"
    man = manifest_path_for(mp4)
    svg = ROOT / "out" / "svg" / t.KLIP
    srcs = [mp4] + ([man] if man.is_file() else []) + sorted(p for p in svg.rglob("*") if p.is_file()) + [t.PROD_CLIP / "meta.json"]
    rows = []
    for s in srcs:
        d = ARSIP / rel(s)
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(s, d)
        rows.append({"berkas": rel(s), "bytes": s.stat().st_size, "sha256": t.file_sha(s), "sama": t.file_sha(s) == t.file_sha(d)})
    if not all(r["sama"] for r in rows):
        raise SystemExit("arsip: sha256 salinan ≠ sumber — berhenti")
    res = {"arsip": rel(ARSIP), "berkas": len(rows), "bytes": sum(r["bytes"] for r in rows), "semua_sha256_sama": True,
           "mp4": rows[0], "manifest_export": next((r for r in rows if r["berkas"] == rel(man)), None),
           "svg": sum(r["berkas"].startswith("out/svg/") for r in rows), "meta": rows[-1], "per_berkas": rows}
    print(f"arsip {res['arsip']}: {res['berkas']} berkas ({res['svg']} di out/svg/klip5), {res['bytes']} B; sha256 sama semua; "
          f"mp4 {rows[0]['sha256'][:12]}; meta {rows[-1]['sha256'][:12]}; manifest {bool(res['manifest_export'])}")
    (ARSIP / "arsip.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"  -> {rel(ARSIP / 'arsip.json')}")


# ── A1–A3 di produksi (keputusan folder produksi klip5) ─
def prod_trim_dir() -> Path:
    wd = trim_dir(load_pipeline(t.DEFAULT_CFG))
    if wd.resolve() != (ROOT / "work" / "clips" / "klip5@0-2.25").resolve():
        raise SystemExit(f"folder trim produksi tak terduga: {wd} — berhenti")
    return wd


def both_refs(wd: Path) -> dict:
    """Frame folder trim produksi vs produksi klip5 f0–53 DAN vs scratch trim4a."""
    return {"vs_produksi_klip5": frame_summary(wd), "vs_scratch_trim4a": frame_summary(wd, trim_dir(load_pipeline(CFG)))}


def frames_ok(fs: dict) -> bool:
    return all(v["n_scratch"] == N_EXPECT and v["bytes_sama"] == N_EXPECT and v["piksel_sama"] == N_EXPECT and not v["tambahan"]
               for v in fs.values())


def print_frames(fs: dict) -> None:
    for k, v in fs.items():
        print(f"  {k}: {v['n_scratch']} frame; bytes sama {v['bytes_sama']}/{N_EXPECT}; piksel sama {v['piksel_sama']}/{N_EXPECT}; "
              f"beda {v['beda']}; tambahan {v['tambahan']}")


def gpu_outputs(wd: Path) -> dict:
    """Berkas hasil stage [1b]–[6] yang TIDAK boleh ada sebelum run GPU Rio."""
    out = ROOT / "out"
    paths = {**{n: wd / n for n in ("isolate", "isolate.partial", "isolate.old", "seg", "qc_report.json", "depth", "stable",
                                     "contours", "strokes")},
             "out_mp4": out / f"{wd.name}.mp4", "out_svg": out / "svg" / wd.name}
    return {k: p.exists() for k, p in paths.items()}


def cmd_a1() -> None:
    wd = prod_trim_dir()
    if wd.exists():
        raise SystemExit(f"{rel(wd)} sudah ada — A1 tidak diulang")
    crop_before = {rel(p): t.file_sha(p) for p in sorted(t.PROD_CROP.rglob("*")) if p.is_file()}
    r = t.run_cli(["ingest", t.VIDEO, *TRIM])
    print(f"== {r['cmd']} -> rc {r['rc']}\n{r['stdout']}{r['stderr']}")
    res: dict = {"folder": rel(wd), "ingest": r}
    crop_after = {rel(p): t.file_sha(p) for p in sorted(t.PROD_CROP.rglob("*")) if p.is_file()}
    res["crop_berkas_sama"] = crop_before == crop_after
    if r["rc"] != 0:
        t.jsave(t.OUT / "t4a2_a1.json", res)
        raise SystemExit(f"A1: ingest rc {r['rc']}" + (" (exit 4 = saran crop)" if r["rc"] == 4 else "") + " — BERHENTI")
    res["frame"] = both_refs(wd)
    meta_old = json.loads((t.PROD_CLIP / "meta.json").read_text(encoding="utf-8"))
    meta_new = json.loads((wd / "meta.json").read_text(encoding="utf-8"))
    beda = sorted(k for k in set(meta_old) | set(meta_new) if meta_old.get(k) != meta_new.get(k))
    res["meta"] = {"key_beda": beda, "frame_count": [meta_old.get("frame_count"), meta_new.get("frame_count")],
                   "trim_baru": meta_new.get("trim"), "crop_sama": meta_old.get("crop") == meta_new.get("crop"),
                   "crop_mode": (meta_new.get("crop") or {}).get("mode"), "meta_sha256": t.file_sha(wd / "meta.json")}
    print_frames(res["frame"])
    print(f"meta key beda: {beda}; frame_count {res['meta']['frame_count']}; crop sama {res['meta']['crop_sama']} "
          f"(mode {res['meta']['crop_mode']}); berkas crop tidak berubah: {res['crop_berkas_sama']}")
    res["lolos"] = (frames_ok(res["frame"]) and beda == ["frame_count", "trim"] and res["meta"]["crop_sama"]
                    and res["crop_berkas_sama"])
    print(f"A1 lolos: {res['lolos']}")
    t.jsave(t.OUT / "t4a2_a1.json", res)
    if not res["lolos"]:
        raise SystemExit("A1 GAGAL — BERHENTI")


def cmd_a2() -> None:
    wd = prod_trim_dir()
    a1 = json.loads((t.OUT / "t4a2_a1.json").read_text(encoding="utf-8"))
    if not a1.get("lolos"):
        raise SystemExit("A1 belum lolos — A2 tidak dijalankan")
    psha = t.file_sha(POINTS_ISO2)
    if not psha.startswith(POINTS_ISO2_SHA):
        raise SystemExit(f"{rel(POINTS_ISO2)} sha256 {psha[:12]} ≠ {POINTS_ISO2_SHA} — berhenti")
    dst = wd / pick.TARGET_DIRNAME / pick.POINTS_FILENAME
    if dst.exists():
        raise SystemExit(f"{rel(dst)} sudah ada — tidak ditimpa")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(POINTS_ISO2, dst)
    res: dict = {"sumber": rel(POINTS_ISO2), "salinan": rel(dst), "sha256_sumber": psha, "sha256_salinan": t.file_sha(dst),
                 "bytes": dst.stat().st_size, "sumber_masih_ada": POINTS_ISO2.is_file(),
                 "bukan_hard_link": os.stat(dst).st_nlink == 1 and not os.path.samefile(POINTS_ISO2, dst)}
    doc = pick.load_points(dst)
    res["sidik"] = [{"frame": e["frame_index"],
                     "sama_frame_produksi_baru": pick.frame_pixel_sha256(wd / "frames" / e["frame_file"]) == e["frame_pixel_sha256"]}
                    for e in doc["frames"]]
    m = json.loads((wd / "meta.json").read_text(encoding="utf-8"))
    try:
        pick.check_doc_matches_clip(doc, clip_identity(wd), m["working_width"], m["working_height"], dst)
        res["check_doc_matches_clip"] = "ok"
    except pick.PickError as e:
        res["check_doc_matches_clip"] = str(e)
    r = t.run_cli(["isolate", t.VIDEO, *TRIM, "--dry-run"])
    res["dry_run"] = r
    out = r["stdout"] + r["stderr"]
    res["dry_run_cek"] = {"rc0": r["rc"] == 0, "2_adegan_cut_22": "adegan: 2 (cut di frame 22)" in out,
                          "f0_21_klik_f0": "adegan f0–f21 (22 frame): klik di f0;" in out,
                          "f22_53_klik_f22_f40": "adegan f22–f53 (32 frame): klik di f22, f40;" in out,
                          "tanpa_prompt_0": "tanpa_prompt: 0 frame" in out}
    res["sesudah"] = gpu_outputs(wd)
    print(f"salin {res['salinan']}: sha256 sama {res['sha256_sumber'] == res['sha256_salinan']} ({psha[:12]}); bukan hard link "
          f"{res['bukan_hard_link']}; sumber masih ada {res['sumber_masih_ada']}")
    print(f"sidik: {res['sidik']}; check_doc_matches_clip: {res['check_doc_matches_clip']}")
    print(f"== {r['cmd']} -> rc {r['rc']}\n{out}")
    print(f"cek dry-run: {res['dry_run_cek']}; hasil GPU ada: {[k for k, v in res['sesudah'].items() if v]}")
    res["lolos"] = (res["sha256_sumber"] == res["sha256_salinan"] and res["bukan_hard_link"] and res["sumber_masih_ada"]
                    and all(s["sama_frame_produksi_baru"] for s in res["sidik"]) and res["check_doc_matches_clip"] == "ok"
                    and all(res["dry_run_cek"].values()) and not any(res["sesudah"].values()))
    print(f"A2 lolos: {res['lolos']}")
    t.jsave(t.OUT / "t4a2_a2.json", res)
    if not res["lolos"]:
        raise SystemExit("A2 GAGAL — BERHENTI")


def cmd_a3() -> None:
    wd = prod_trim_dir()
    a2 = json.loads((t.OUT / "t4a2_a2.json").read_text(encoding="utf-8"))
    if not a2.get("lolos") or not a2["dry_run_cek"]["rc0"] or "GPU: dibutuhkan (--yes wajib)" not in a2["dry_run"]["stdout"]:
        raise SystemExit("A2 belum lolos / dry-run tidak menyatakan --yes wajib — A3 tidak dijalankan")
    dst = wd / pick.TARGET_DIRNAME / pick.POINTS_FILENAME
    psha = t.file_sha(dst)
    r = t.run_cli(["run", t.VIDEO, *TRIM])                 # TANPA --yes: [1b] menolak sebelum GPU (4a (d))
    out = r["stdout"] + r["stderr"]
    folder_line = next((ln for ln in out.splitlines() if "folder kerja" in ln), "")
    res: dict = {"run": r, "cek": {"rc1": r["rc"] == 1,
                                   "folder_trim": Path(folder_line.split("folder kerja ", 1)[-1].split(", output", 1)[0]).resolve()
                                   == wd.resolve() if folder_line else False,
                                   "trim_0_53": "frame 0–53 (54 frame)" in out, "berhenti_di_1b": "=== [1b] isolate ===" in out
                                   and "[1b] isolate butuh GPU (belum ada hasil)" in out,
                                   "tidak_sampai_2": "=== [2] segment ===" not in out and "=== [2c] depth ===" not in out}}
    res["sesudah"] = {"hasil_gpu_ada": [k for k, v in gpu_outputs(wd).items() if v], "titik_sama": t.file_sha(dst) == psha,
                      "frame": both_refs(wd)}
    print(f"== {r['cmd']} -> rc {r['rc']}\n{out}")
    print(f"cek: {res['cek']}; hasil GPU ada sesudah: {res['sesudah']['hasil_gpu_ada']}; titik sama {res['sesudah']['titik_sama']}")
    print_frames(res["sesudah"]["frame"])
    res["lolos"] = (all(res["cek"].values()) and not res["sesudah"]["hasil_gpu_ada"] and res["sesudah"]["titik_sama"]
                    and frames_ok(res["sesudah"]["frame"]))
    print(f"A3 lolos: {res['lolos']}")
    t.jsave(t.OUT / "t4a2_a3.json", res)


def main(argv: list[str]) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    cmds = {"siapkan": cmd_siapkan, "ingest": cmd_ingest, "banding": cmd_banding, "arsip": cmd_arsip, "a1": cmd_a1, "a2": cmd_a2,
            "a3": cmd_a3}
    if len(argv) != 1 or argv[0] not in cmds:
        print(__doc__)
        return 2
    cmds[argv[0]]()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
