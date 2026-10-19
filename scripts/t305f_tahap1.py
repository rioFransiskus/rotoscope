"""T-305f Tahap 1: VF (isi lubang di dalam muka) — bandingkan tempat pasang di stage [3] (docs/04 "Keputusan T-305f").

    python scripts/t305f_tahap1.py sim KLIP          # [3] in-process (baca-saja): BASE + P-post (PB / PA) + P-pre (PR) -> work/t305f/sim_KLIP.json
    python scripts/t305f_tahap1.py render VAR KLIP   # [4] + [5] seluruh klip di salinan scratch work/t305f_scratch/VAR/KLIP/
    python scripts/t305f_tahap1.py lines             # garis / gambar yang berubah + lubang muka (Q1) per varian -> work/t305f/lines.json
    python scripts/t305f_tahap1.py ringkas           # tabel ringkas dari sim_*.json + lines.json -> work/t305f/ringkas.json

Varian (semua memakai aturan VF prototipe `scripts/t305_1b.py::vf_fill` APA ADANYA, grup face = `groups:` "face"):
  BASE  [3] produksi dihitung ulang in-process (FrameCache asli + masker T-606 bila berlaku) → kontrol: byte-identik dengan stable/ produksi.
  PB    P-post, keluaran saja: VF pada peta grup akhir (sesudah kernel + pulau + mode), kedalaman TIDAK berubah (= prototipe 1b).
  PA    P-post, VF juga dipakai statistik foreground kedalaman (median log disparity dari peta terisi; depth.temporal mati).
  PR    P-pre: VF pada argmax MENTAH di `FrameCache.raw(k)` (sesudah masker T-606), piksel terisi → jumlah grup face one-hot
        (255) + grup seri face, SEBELUM kernel (ikut dihaluskan ke frame tetangga).
Klip asli dan out/ hanya DIBACA (stage tidak dijalankan di folder klip); salinan scratch: meta.json + qc_report.json + stable/manifest.json
+ frames.jsonl disalin, berkas stable/ yang byte-identik dengan produksi di-hard-link, yang berubah ditulis baru. CPU saja.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import t305_1b as proto  # noqa: E402
import temporal_metrics as tm  # noqa: E402
from t305d_stage3 import runs_str  # noqa: E402

from rotoscope import stabilize as stb  # noqa: E402
from rotoscope import target_mask as tmk  # noqa: E402
from rotoscope.config import load_class_names, load_pipeline  # noqa: E402
from rotoscope.stage_common import work_dir_overrides  # noqa: E402

CLIPS_DIR = ROOT / "work" / "clips"
OUT = ROOT / "work" / "t305f"
SCR = ROOT / "work" / "t305f_scratch"
CFG_PATH = ROOT / "configs" / "default.yaml"
CLIPS = ("test", "test_short", "Klip2", "klip3", "klip4", "klip5")
VARS = ("PB", "PA", "PR")
FACE_GROUP = "face"
DIAG, DIAG_RANGE = proto.DIAG, proto.DIAG_RANGE      # klip5 f0–f91 (lingkup prototipe)
PROTO_VF = ROOT / "work" / "t305_scratch" / "1b" / "VF" / "klip5"   # render prototipe VF klip5 ([4] --limit 92, [5] f0–f91)
PROTO_SCAN = ROOT / "work" / "t305" / "1b" / "vfscan.json"
FACE_HOLE_MIN_FRAC = 0.5                             # "lubang muka" di peta: ≥ 50% cincin 1 px = face (definisi pelaporan, bukan aturan VF)


def jload(p: Path):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def jsave(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"  -> {p.relative_to(ROOT).as_posix()}")


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def file_sha(p: Path) -> str:
    return sha(p.read_bytes())


def enc_groups(g: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(stb.GROUPS_SUFFIX, g)                 # = stb.write_groups
    assert ok
    return buf.tobytes()


def enc_depth(y: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, y, allow_pickle=False)                          # = stb.write_depth_smooth
    return buf.getvalue()


def stats(v) -> dict | None:
    v = [float(x) for x in v]
    if not v:
        return None
    return {"n": len(v), "median": round(float(np.median(v)), 3), "p95": round(float(np.percentile(v, 95)), 3),
            "max": round(max(v), 3), "mean": round(float(np.mean(v)), 3)}


# ── VF + analisis lubang ───────────────────────────
def vf(g: np.ndarray, face: int) -> tuple[np.ndarray, int, int]:
    """Aturan VF prototipe, APA ADANYA (`t305_1b.vf_fill`)."""
    return proto.vf_fill(g, face)


def face_holes(g: np.ndarray, face: int, names: tuple[str, ...], min_area: int) -> list[dict]:
    """Komponen background 4-arah yang tidak menyentuh tepi frame dengan ≥ 50% cincin 1 px (8-arah) = face; definisi cincin / tepi =
    `vf_fill`. Untuk pelaporan: terisi VF bila cincin 100% face. `drawn` = luas ≥ vectorize.min_hole_area (digambar [4])."""
    H, W = g.shape
    n, lab, st, _ = cv2.connectedComponentsWithStats((g == stb.BACKGROUND_ID).astype(np.uint8), connectivity=4)
    out = []
    for i in range(1, n):
        x, y, w, h, a = (int(v) for v in st[i])
        if x == 0 or y == 0 or x + w >= W or y + h >= H:
            continue
        sl = (slice(y - 1, y + h + 1), slice(x - 1, x + w + 1))
        comp = lab[sl] == i
        ring = cv2.dilate(comp.astype(np.uint8), proto.K3).astype(bool) & ~comp
        vals = g[sl][ring]
        ff = float((vals == face).mean())
        if ff < FACE_HOLE_MIN_FRAC:
            continue
        cnt = Counter(names[int(v)] for v in vals)
        out.append({"area": a, "bbox_xywh": [x, y, w, h], "ring_px": int(vals.size), "face_frac": round(ff, 4),
                    "fill": ff == 1.0, "drawn": a >= min_area, "ring": dict(cnt.most_common())})
    return out


class PreCache(stb.FrameCache):
    """P-pre: VF pada argmax mentah frame k (sesudah masker T-606), sebelum kernel; piksel terisi → face one-hot."""

    def __init__(self, *a, face: int, **kw) -> None:
        super().__init__(*a, **kw)
        self.face = face
        self.vf_info: dict[int, dict] = {}

    def raw(self, k: int):
        if k not in self._raw:
            sums, tie, _ = super().raw(k)
            t0 = time.perf_counter()
            g = stb.group_argmax(sums, tie)[0]
            g2, reg, px = vf(g, self.face)
            if reg:
                fill = g2 != g
                sums[:, fill] = 0
                sums[self.face, fill] = tmk.PROBS_ONE
                tie[fill] = self.face
            self.vf_info[k] = {"regions": reg, "px": px, "ms": round(1000 * (time.perf_counter() - t0), 3)}
        return self._raw[k]


# ── salinan scratch ────────────────────────────────
def vdir(var: str, clip: str) -> Path:
    return SCR / var / clip


def link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def init_vdir(var: str, clip: str) -> Path:
    src, d = CLIPS_DIR / clip, vdir(var, clip)
    if d.exists():
        shutil.rmtree(d)
    (d / "stable" / "groups").mkdir(parents=True)
    (d / "stable" / "depth_smooth").mkdir(parents=True)
    for f in ("meta.json", "qc_report.json"):
        shutil.copyfile(src / f, d / f)
    for f in (stb.MANIFEST_FILENAME, stb.FRAMES_LOG_FILENAME):
        shutil.copyfile(src / "stable" / f, d / "stable" / f)
    return d


def put(dst: Path, data: bytes, prod: Path, prod_bytes: bytes) -> bool:
    """Tulis `data` ke salinan; sama dengan produksi → hard link. Return True bila berbeda."""
    if data == prod_bytes:
        link(prod, dst)
        return False
    dst.write_bytes(data)
    return True


# ── sim ────────────────────────────────────────────
def cmd_sim(clip_name: str) -> None:
    work = CLIPS_DIR / clip_name
    cfg = load_pipeline(CFG_PATH, overrides=work_dir_overrides(work))
    clip = stb.load_clip(work)
    stb.load_inputs(clip)
    classes = load_class_names()
    lut = stb.class_group_lut(cfg.groups, classes)
    names = ("background", *(g for g, _ in cfg.groups))
    face = names.index(FACE_GROUP)
    plan = stb.build_plan(cfg, clip, print)
    if plan.depth_radius:
        raise SystemExit("stabilize.depth.temporal aktif — jalan pintas PA (normalisasi per frame) tidak berlaku")
    dc = cfg.stabilize.depth
    t0 = time.perf_counter()
    masks = tmk.load_masks(cfg, work, print)
    t_masks = time.perf_counter() - t0
    base = stb.FrameCache(cfg, clip, plan, lut, len(classes), masks)
    pre = PreCache(cfg, clip, plan, lut, len(classes), masks, face=face)
    dirs = {v: init_vdir(v, clip_name) for v in VARS}
    min_hole = cfg.vectorize.min_hole_area
    rows = []
    t_all = time.perf_counter()
    for pos, name in enumerate(clip.names):
        stem = Path(name).stem
        pg, pd = clip.groups_path(name), clip.depth_smooth_path(name)
        prod_g, prod_d = pg.read_bytes(), pd.read_bytes()
        t1 = time.perf_counter()
        gb, _ = base.groups(pos)
        db, _ = base.final_depth(pos)
        t_base = time.perf_counter() - t1
        gb_b, db_b = enc_groups(gb), enc_depth(db)
        t2 = time.perf_counter()
        gpb, reg, px = vf(gb, face)
        t_post = time.perf_counter() - t2
        gpb_b = enc_groups(gpb) if reg else gb_b
        if reg:                                                       # PA: statistik foreground kedalaman dari peta terisi
            nd, info = stb.normalize_depth_f32(base.raw(pos)[2], gpb != stb.BACKGROUND_ID, dc.log_eps, dc.iqr_min, dc.normalize)
            dpa = stb.finish_depth(nd, info)[0]
            dpa_b = enc_depth(dpa)
            med_base = base.depth_f32(pos)[1]["med"]
            dpa_diff = {"med_shift": info["med"] - med_base,
                        "max_abs": float(np.abs(dpa.astype(np.float32) - db.astype(np.float32)).max()),
                        "px_changed": int((dpa != db).sum())}
        else:
            dpa_b, dpa_diff = db_b, None
        t3 = time.perf_counter()
        gpr, _ = pre.groups(pos)
        dpr, _ = pre.final_depth(pos)
        t_pre_total = time.perf_counter() - t3
        gpr_b, dpr_b = enc_groups(gpr), enc_depth(dpr)
        ch = {"PB": (put(dirs["PB"] / "stable" / "groups" / pg.name, gpb_b, pg, prod_g),
                     put(dirs["PB"] / "stable" / "depth_smooth" / pd.name, db_b, pd, prod_d)),
              "PA": (put(dirs["PA"] / "stable" / "groups" / pg.name, gpb_b, pg, prod_g),
                     put(dirs["PA"] / "stable" / "depth_smooth" / pd.name, dpa_b, pd, prod_d)),
              "PR": (put(dirs["PR"] / "stable" / "groups" / pg.name, gpr_b, pg, prod_g),
                     put(dirs["PR"] / "stable" / "depth_smooth" / pd.name, dpr_b, pd, prod_d))}
        holes_b = face_holes(gb, face, names, min_hole)
        holes_pr = face_holes(gpr, face, names, min_hole) if gpr_b != gb_b else holes_b
        pinfo = pre.vf_info.get(pos, {"regions": 0, "px": 0, "ms": None})
        row = {"frame": pos, "name": stem, "control_groups": gb_b == prod_g, "control_depth": db_b == prod_d,
               "base_ms": round(1000 * t_base, 2),
               "post": {"regions": reg, "px": px, "ms": round(1000 * t_post, 3)},
               "pa_depth": dpa_diff,
               "pre": {"raw_regions": pinfo["regions"], "raw_px": pinfo["px"], "raw_ms": pinfo["ms"],
                       "total_ms": round(1000 * t_pre_total, 2),
                       "groups_px_vs_base": int((gpr != gb).sum()), "groups_px_vs_post": int((gpr != gpb).sum()),
                       "depth_px_vs_base": int((dpr != db).sum())},
               "changed": {v: {"groups": a, "depth": b} for v, (a, b) in ch.items()},
               "holes_base": holes_b, "holes_pre": holes_pr}
        rows.append(row)
        lo = plan.needed(pos)[0]
        base.evict_before(lo)
        pre.evict_before(lo)
        if pos % 20 == 0 or reg or pinfo["regions"]:
            print(f"  {stem}: kontrol {row['control_groups']}/{row['control_depth']}, post {reg} wil {px} px, pre mentah "
                  f"{pinfo['regions']} wil {pinfo['px']} px → beda vs base {row['pre']['groups_px_vs_base']} px", flush=True)
    wall = time.perf_counter() - t_all
    for v in VARS:
        if not any(r["changed"][v]["groups"] or r["changed"][v]["depth"] for r in rows):
            shutil.rmtree(dirs[v])                                    # tidak ada beda → tidak perlu render
            if not any(vdir(v, c).exists() for c in CLIPS) and (SCR / v).is_dir():
                shutil.rmtree(SCR / v)
    res = {"clip": clip_name, "n": len(rows), "size": [clip.width, clip.height], "face_id": face, "group_names": names,
           "masks": masks is not None, "masks_block": (masks.block if masks is not None else None), "load_masks_s": round(t_masks, 2),
           "plan": {"radius": plan.radius, "cut_frames": list(plan.cut_frames), "qc_fail_frames": list(plan.qc_fail_frames)},
           "min_hole_area": min_hole, "wall_s": round(wall, 1), "rows": rows}
    res["summary"] = summarize(res)
    jsave(OUT / f"sim_{clip_name}.json", res)
    print(json.dumps(res["summary"], indent=1, ensure_ascii=False))


def summarize(res: dict, rng: range | None = None) -> dict:
    rows = [r for r in res["rows"] if rng is None or r["frame"] in rng]
    fr = lambda pred: [r["frame"] for r in rows if pred(r)]  # noqa: E731
    post_f = fr(lambda r: r["post"]["regions"])
    pre_raw_f = fr(lambda r: r["pre"]["raw_regions"])
    pre_chg = fr(lambda r: r["pre"]["groups_px_vs_base"])
    hb = [(r["frame"], h) for r in rows for h in r["holes_base"]]
    hp = [(r["frame"], h) for r in rows for h in r["holes_pre"]]
    drawn_b = [(f, h) for f, h in hb if h["drawn"]]
    unfilled = [(f, h) for f, h in drawn_b if not h["fill"]]
    return {
        "frames": len(rows), "control_ok": {"groups": sum(r["control_groups"] for r in rows), "depth": sum(r["control_depth"] for r in rows)},
        "PB": {"frames_changed": len(post_f), "runs": runs_str(post_f), "regions": sum(r["post"]["regions"] for r in rows),
               "px": sum(r["post"]["px"] for r in rows)},
        "PA": {"depth_frames_changed": len(fr(lambda r: r["changed"]["PA"]["depth"])),
               "depth_px_changed": sum((r["pa_depth"] or {}).get("px_changed", 0) for r in rows),
               "med_shift_max_abs": max((abs(r["pa_depth"]["med_shift"]) for r in rows if r["pa_depth"]), default=0.0),
               "depth_max_abs": max((r["pa_depth"]["max_abs"] for r in rows if r["pa_depth"]), default=0.0)},
        "PR": {"raw_frames_filled": len(pre_raw_f), "raw_runs": runs_str(pre_raw_f),
               "raw_regions": sum(r["pre"]["raw_regions"] for r in rows), "raw_px": sum(r["pre"]["raw_px"] for r in rows),
               "frames_changed": len(pre_chg), "runs": runs_str(pre_chg), "px_changed": sum(r["pre"]["groups_px_vs_base"] for r in rows),
               "frames_changed_without_own_fill": runs_str([f for f in pre_chg if f not in pre_raw_f]),
               "px_changed_without_own_fill": sum(r["pre"]["groups_px_vs_base"] for r in rows if not r["pre"]["raw_regions"]),
               "depth_frames_changed": len(fr(lambda r: r["pre"]["depth_px_vs_base"]))},
        "PB_vs_PR": {"frames_differ": len(fr(lambda r: r["pre"]["groups_px_vs_post"])),
                     "runs": runs_str(fr(lambda r: r["pre"]["groups_px_vs_post"])),
                     "px_differ": sum(r["pre"]["groups_px_vs_post"] for r in rows)},
        "face_holes_drawn": {"base": len(drawn_b), "base_frames": runs_str(sorted({f for f, _ in drawn_b})),
                             "filled_by_PB": len(drawn_b) - len(unfilled), "left_by_PB": len(unfilled),
                             "left_by_PB_frames": runs_str(sorted({f for f, _ in unfilled})),
                             "left_by_PR": sum(h["drawn"] for _, h in hp),
                             "left_by_PR_frames": runs_str(sorted({f for f, h in hp if h["drawn"]}))},
        "face_holes_small": {"base": sum(not h["drawn"] for _, h in hb), "filled_by_PB": sum(not h["drawn"] and h["fill"] for _, h in hb)},
        "left_by_PB_detail": [{"frame": f, "area": h["area"], "face_frac": h["face_frac"], "ring": h["ring"]} for f, h in unfilled],
        "ms": {"base_frame": stats(r["base_ms"] for r in rows), "post_vf": stats(r["post"]["ms"] for r in rows),
               "pre_vf_raw": stats(r["pre"]["raw_ms"] for r in rows if r["pre"]["raw_ms"] is not None)},
    }


# ── render ─────────────────────────────────────────
def cfg_path(var: str) -> Path:
    p = SCR / var / "cfg.yaml"
    text = CFG_PATH.read_text(encoding="utf-8")
    line = '  out_dir: "out"'
    if text.count(line) != 1:
        raise SystemExit("configs/default.yaml: baris out_dir tidak ditemukan tepat sekali")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text.replace(line, f'  out_dir: "{(SCR / var / "out").as_posix()}"'), encoding="utf-8")
    return p


def run_logged(cmd: list[str], log: Path) -> tuple[int, float]:
    t0 = time.perf_counter()
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(r.stdout + "\n--- stderr ---\n" + r.stderr, encoding="utf-8")
    return r.returncode, time.perf_counter() - t0


def cmd_render(var: str, clip: str) -> None:
    d = vdir(var, clip)
    if not d.is_dir():
        raise SystemExit(f"{d.relative_to(ROOT).as_posix()} tidak ada (varian tidak mengubah klip ini, atau sim belum jalan)")
    cfg = cfg_path(var)
    res = {}
    for name in ("vectorize", "stylize"):
        rc, dt = run_logged([sys.executable, "-m", f"rotoscope.{name}", "--config", str(cfg), "--work-dir", str(d)],
                            SCR / "logs" / f"{var}_{clip}_{name}.log")
        res[name] = {"rc": rc, "s": round(dt, 1)}
        print(f"  {var} {clip} [{name}] rc {rc}, {dt:.1f} s", flush=True)
        if rc:
            raise SystemExit(f"{name} gagal — lihat work/t305f_scratch/logs/{var}_{clip}_{name}.log")
    p = OUT / "render.json"
    allr = jload(p) if p.is_file() else {}
    allr[f"{var}/{clip}"] = res
    jsave(p, allr)


# ── garis / gambar ─────────────────────────────────
def contour_doc(d: Path, stem: str) -> dict:
    return jload(d / "contours" / f"{stem}.json")


def geom(s: dict) -> tuple:
    return (s["type"], s["closed"], tuple(s.get("groups") or ()), json.dumps(s["points"]))


def q1_face_holes(doc: dict, gmap: np.ndarray, face: int) -> int:
    """Q1 1b terbatas pada silhouette_hole: ≥ 50% cincin luar 1..4 px poligon = face (peta stable varian itu sendiri)."""
    H, W = gmap.shape
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * proto.RING_PX + 1, 2 * proto.RING_PX + 1))
    n = 0
    for s in doc["strokes"]:
        if s["type"] != "silhouette_hole":
            continue
        fill = proto.poly_fill(s["points"], H, W)
        ring = cv2.dilate(fill.astype(np.uint8), kern).astype(bool) & ~fill
        if ring.any() and float((gmap[ring] == face).mean()) >= proto.FACE_RING_MIN:
            n += 1
    return n


def cmd_lines() -> None:
    res = {}
    for var in VARS:
        for clip in CLIPS:
            d = vdir(var, clip)
            if not (d / "contours").is_dir():
                continue
            prod = CLIPS_DIR / clip
            c = stb.load_clip(prod)
            sim = jload(OUT / f"sim_{clip}.json")
            face = sim["face_id"]
            stable_changed = {r["frame"] for r in sim["rows"] if r["changed"][var]["groups"] or r["changed"][var]["depth"]}
            rows = []
            for pos, name in enumerate(c.names):
                stem = Path(name).stem
                a, b = contour_doc(prod, stem), contour_doc(d, stem)
                same = a["strokes"] == b["strokes"]
                same_geom = Counter(map(geom, a["strokes"])) == Counter(map(geom, b["strokes"]))
                png = file_sha(prod / "strokes" / f"{stem}.png") == file_sha(d / "strokes" / f"{stem}.png")
                svg = file_sha(prod / "strokes" / f"{stem}.svg") == file_sha(d / "strokes" / f"{stem}.svg")
                ga = stb.read_groups(c.groups_path(name))
                gv = stb.read_groups(d / "stable" / "groups" / f"{stem}.png")
                rows.append({"frame": pos, "strokes_same": same, "geom_same": same_geom, "png_same": png, "svg_same": svg,
                             "stable_changed": pos in stable_changed,
                             "types_prod": dict(Counter(s["type"] for s in a["strokes"])),
                             "types_var": dict(Counter(s["type"] for s in b["strokes"])),
                             "q1_holes_prod": q1_face_holes(a, ga, face), "q1_holes_var": q1_face_holes(b, gv, face)})
            cs_a, cs_b = jload(prod / "contours" / "clip_stats.json"), jload(d / "contours" / "clip_stats.json")
            key = f"{var}/{clip}"
            res[key] = {"rows": rows, "summary": lines_summary(rows),
                        "clip_stats": {k: [cs_a[k], cs_b[k]] for k in ("t_high", "t_low")}}
            if clip == DIAG:
                rng = range(DIAG_RANGE[0], DIAG_RANGE[1] + 1)
                res[key]["summary_f0_f91"] = lines_summary([r for r in rows if r["frame"] in rng])
                if var == "PB" and (PROTO_VF / "contours").is_dir():
                    res[key]["vs_prototype_f0_f91"] = vs_prototype(d, c, rng)
            print(key, json.dumps({k: v for k, v in res[key].items() if k != "rows"}, ensure_ascii=False), flush=True)
    jsave(OUT / "lines.json", res)


def lines_summary(rows: list[dict]) -> dict:
    f = lambda pred: [r["frame"] for r in rows if pred(r)]  # noqa: E731
    outside = [r for r in rows if not r["stable_changed"]]
    return {"frames": len(rows), "stable_changed": runs_str(f(lambda r: r["stable_changed"])),
            "strokes_differ": runs_str(f(lambda r: not r["strokes_same"])),
            "geom_differ": runs_str(f(lambda r: not r["geom_same"])),
            "only_track_id_differ": runs_str(f(lambda r: r["geom_same"] and not r["strokes_same"])),
            "png_differ": runs_str(f(lambda r: not r["png_same"])), "svg_differ": runs_str(f(lambda r: not r["svg_same"])),
            "n_png_differ": len(f(lambda r: not r["png_same"])),
            "outside_stable_changed": {"frames": len(outside), "geom_differ": runs_str([r["frame"] for r in outside if not r["geom_same"]]),
                                       "png_differ": runs_str([r["frame"] for r in outside if not r["png_same"]])},
            "q1_holes": [sum(r["q1_holes_prod"] for r in rows), sum(r["q1_holes_var"] for r in rows)],
            "q1_frames_prod": runs_str(f(lambda r: r["q1_holes_prod"])), "q1_frames_var": runs_str(f(lambda r: r["q1_holes_var"])),
            "q1_gone_frames": runs_str(f(lambda r: r["q1_holes_var"] < r["q1_holes_prod"])),
            "q1_new_frames": runs_str(f(lambda r: r["q1_holes_var"] > r["q1_holes_prod"])),
            "strokes_per_type": {k: [sum(r["types_prod"].get(k, 0) for r in rows), sum(r["types_var"].get(k, 0) for r in rows)]
                                 for k in ("silhouette", "silhouette_hole", "group_boundary", "occlusion")}}


def vs_prototype(d: Path, c, rng: range) -> dict:
    """PB klip5 f0–f91 vs render prototipe VF 1b (salinan scratch T-305 1b): strok contours + PNG strokes."""
    st, png = [], []
    for pos in rng:
        stem = Path(c.names[pos]).stem
        if contour_doc(PROTO_VF, stem)["strokes"] != contour_doc(d, stem)["strokes"]:
            st.append(pos)
        if file_sha(PROTO_VF / "strokes" / f"{stem}.png") != file_sha(d / "strokes" / f"{stem}.png"):
            png.append(pos)
    gs = [pos for pos in rng if file_sha(PROTO_VF / "stable" / "groups" / f"{Path(c.names[pos]).stem}.png")
          != file_sha(d / "stable" / "groups" / f"{Path(c.names[pos]).stem}.png")]
    return {"frames": len(rng), "groups_png_differ": runs_str(gs), "contours_strokes_differ": runs_str(st), "strokes_png_differ": runs_str(png)}


# ── ringkas ────────────────────────────────────────
def cmd_ringkas() -> None:
    out = {"sim": {}, "lines": {}}
    for clip in CLIPS:
        p = OUT / f"sim_{clip}.json"
        if not p.is_file():
            continue
        s = jload(p)
        out["sim"][clip] = s["summary"]
        if clip == DIAG:
            out["sim"]["klip5_f0_f91"] = summarize(s, range(DIAG_RANGE[0], DIAG_RANGE[1] + 1))
    if PROTO_SCAN.is_file():
        scan = jload(PROTO_SCAN)
        out["prototype_vfscan"] = scan
        k5 = out["sim"].get("klip5_f0_f91", {}).get("PB")
        if k5:
            p5 = scan[DIAG]
            out["PB_equals_prototype_klip5_f0_f91"] = ([k5["frames_changed"], k5["regions"], k5["px"]]
                                                      == [p5["frames_changed"], p5["regions_filled"], p5["px_filled"]])
    if (OUT / "lines.json").is_file():
        out["lines"] = {k: {kk: vv for kk, vv in v.items() if kk != "rows"} for k, v in jload(OUT / "lines.json").items()}
    jsave(OUT / "ringkas.json", out)
    print(json.dumps(out, indent=1, ensure_ascii=False))


def main(argv: list[str]) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if argv[:1] == ["sim"] and len(argv) == 2 and argv[1] in CLIPS:
        cmd_sim(argv[1])
    elif argv[:1] == ["render"] and len(argv) == 3 and argv[1] in VARS and argv[2] in CLIPS:
        cmd_render(argv[1], argv[2])
    elif argv == ["lines"]:
        cmd_lines()
    elif argv == ["ringkas"]:
        cmd_ringkas()
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
