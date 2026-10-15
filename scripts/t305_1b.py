"""T-305 induk Tahap 1b: diagnosis klip5 (docs/04 "Penilaian Rio Tahap 1 + keputusan Tahap 1b (2026-10-15)").

    python scripts/t305_1b.py hash before|after   # F1: 4 klip lama + seluruh work/clips/klip4, klip5 + out/ -> work/t305/1b/
    python scripts/t305_1b.py q1                  # "lingkaran di muka": strok melingkar dikelilingi grup face, 5 klip -> work/t305/1b/q1.json
    python scripts/t305_1b.py q2                  # komponen siluet klip5 f0–f91: luas, bbox, tepi, wajah V1, kelas -> work/t305/1b/q2.json
    python scripts/t305_1b.py vfscan              # VF di 5 klip: frame / piksel yang berubah + cakupan lubang Q1 -> work/t305/1b/vfscan.json
    python scripts/t305_1b.py prepare VAR         # salinan scratch klip5 (V0 / VF / S1 / S2), peta grup f0–f91 diubah -> work/t305_scratch/1b/VAR/
    python scripts/t305_1b.py render VAR [KLIP]   # [4] --limit + [5] / [6] --from 0 --limit di salinan (KLIP selain klip5: VF klip lain, seluruh klip)
    python scripts/t305_1b.py prepare_vf KLIP     # salinan VF seluruh klip lain (hanya bila vfscan menemukan frame berubah)
    python scripts/t305_1b.py metrics             # metrik per varian + V0 vs produksi -> work/t305/1b/metrics.json
    python scripts/t305_1b.py media               # work/t305/nilai_1b/klip5_A / klip5_B (normal + lambat 0,5x) + VF klip lain

Aturan simulasi: docs/04 "Aturan simulasi Tahap 1b (dicatat SEBELUM render, 2026-10-15)".
CPU saja. Klip asli dan out/ hanya DIBACA; keluaran analisis ke work/t305/1b/, simulasi ke work/t305_scratch/1b/ (keduanya di-ignore git).
Memakai ulang: hash (`scripts/t701_study.py`), panjang strok (`tests/temporal_metrics.py`), definisi wajah T-701 (FACE_MIN_FRAC, aturan V1).
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import os
import shutil
import subprocess
import time

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import temporal_metrics as tm  # noqa: E402
import t701_study as t701  # noqa: E402
from t305_newclips import encoder, fit_h, time_bar  # noqa: E402
from t305d_stage3 import births as occ_births, label_bar, read_frames, runs_str  # noqa: E402

from rotoscope import stabilize as stb  # noqa: E402
from rotoscope.config import load_pipeline  # noqa: E402

CLIPS_DIR = ROOT / "work" / "clips"
OUT = ROOT / "work" / "t305" / "1b"
SCR = ROOT / "work" / "t305_scratch" / "1b"
CLIPS = ("test", "Klip2", "klip3", "klip4", "klip5")
NEW_HASH_CLIPS = ("klip4", "klip5")              # F1 1b: seluruh folder
DIAG = "klip5"
DIAG_RANGE = (0, 91)                             # lingkup diagnosis: sebelum cut ke objek lain (label Rio)

# ── Q1: definisi "strok melingkar di muka" ─────────
LOOP_MIN_LEN_PX = 15.0                           # strok terbuka lebih pendek tidak dihitung melingkar
LOOP_GAP_FRAC = 0.2                              # strok terbuka melingkar: jarak ujung ≤ 0,2 × panjang
RING_PX = 4                                      # cincin luar poligon (piksel 1..4 di luar isi poligon)
FACE_RING_MIN = 0.5                              # ≥ 50% cincin luar berlabel grup face (peta stable) = dikelilingi face

# ── Q2: kelas komponen ─────────────────────────────
FACE_MIN_FRAC = t701.FACE_MIN_FRAC               # komponen grup face ≥ 0,1% luas frame = wajah (T-701 1b)
ZONE_STD_MIN = 2.0                               # baris "konten" = rata-rata simpangan baku abu-abu antar frame > 2 (video bergerak)
ZONE_BLACK_MAX = 10.0                            # baris statis dengan rata-rata abu-abu ≤ 10 = "hitam" (letterbox), selain itu "teks"
ZONES = ("konten", "teks", "hitam")

# ── simulasi (docs/04 "Aturan simulasi Tahap 1b") ──
VARIANTS = ("V0", "VF", "S1", "S2")
K_OF = {"S1": 1, "S2": 2}
SIM_N = DIAG_RANGE[1] + 1                        # f0–f91 diubah; [4] --limit 92, [5] / [6] --from 0 --limit 92
ASSIGN_MIN_PX = 30                               # komponen kecil >= 30 px boleh ikut blob (klip3_diag.persons_of)
ATTACH_MAX_PX = 10                               # ... hanya bila jaraknya <= 10 px (penyimpangan dari T-701 1c: tanpa batas)
TRACK_FACE_FRAC = 0.08                           # pasangan wajah antar frame: jarak centroid <= 0,08 x H
TRACK_IOU_MIN = 0.3                              # pasangan tanpa wajah: IoU masker >= 0,3
K3 = np.ones((3, 3), np.uint8)
NILAI = ROOT / "work" / "t305" / "nilai_1b"
PANEL_H = 960


def jload(p: Path):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def jsave(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"  -> {p.relative_to(ROOT).as_posix()}")


# ── F1 ─────────────────────────────────────────────
def hash_files() -> list[Path]:
    files = t701.hash_files()                    # 4 klip lama (6 subfolder + qc_report.json) + out/
    for c in NEW_HASH_CLIPS:
        files += [p for p in sorted((CLIPS_DIR / c).rglob("*")) if p.is_file()]
    return files


def cmd_hash(which: str) -> None:
    cur = {p.relative_to(ROOT).as_posix(): t701.file_sha(p) for p in hash_files()}
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{which}_sha256.txt"
    path.write_text("".join(f"{h}  {r}\n" for r, h in cur.items()), encoding="utf-8")
    n_qc = sum(r.endswith("/qc_report.json") for r in cur)
    print(f"{path.relative_to(ROOT).as_posix()}: {len(cur)} berkas (qc_report.json: {n_qc})")
    if which == "after":
        before = dict(reversed(line.split("  ", 1)) for line in (OUT / "before_sha256.txt").read_text(encoding="utf-8").splitlines())
        res = {"before": len(before), "after": len(cur), "missing": sorted(set(before) - set(cur)),
               "extra": sorted(set(cur) - set(before)), "changed": sorted(r for r in before if r in cur and cur[r] != before[r])}
        res["identical"] = not (res["missing"] or res["extra"] or res["changed"])
        jsave(OUT / "f1_check.json", res)
        print(json.dumps({k: (v if not isinstance(v, list) else v[:10]) for k, v in res.items()}, ensure_ascii=False))


# ── pembantu ───────────────────────────────────────
def group_id(cfg, name: str) -> int:
    return [g for g, _ in cfg.groups].index(name) + 1


def contour_doc(work: Path, name: str) -> dict:
    return jload(work / "contours" / (Path(name).stem + ".json"))


def pix(points) -> np.ndarray:
    """Titik pusat piksel (i + 0,5) → indeks piksel (x, y) int32."""
    return np.floor(np.asarray(points, float)).astype(np.int32)


def poly_fill(points, H: int, W: int) -> np.ndarray:
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [pix(points).reshape(-1, 1, 2)], 1)
    return m.astype(bool)


def is_loop(s: dict) -> bool:
    if s["closed"]:
        return True
    p = np.asarray(s["points"], float)
    L = tm.stroke_length(p, False)
    return L >= LOOP_MIN_LEN_PX and float(np.hypot(*(p[0] - p[-1]))) <= LOOP_GAP_FRAC * L


def top_share(c: Counter, k: int = 3) -> list:
    tot = sum(c.values()) or 1
    return [[n, round(100 * v / tot, 1)] for n, v in c.most_common(k)]


# ── Q1 ─────────────────────────────────────────────
def face_loops(cd, gmap: np.ndarray, cm: np.ndarray, doc: dict, face_id: int) -> list[dict]:
    H, W = gmap.shape
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * RING_PX + 1, 2 * RING_PX + 1))
    out = []
    for s in doc["strokes"]:
        if s["type"] == "silhouette" or not is_loop(s):
            continue
        fill = poly_fill(s["points"], H, W)
        ring = cv2.dilate(fill.astype(np.uint8), kern).astype(bool) & ~fill
        if not ring.any():
            continue
        face_frac = float((gmap[ring] == face_id).mean())
        if face_frac < FACE_RING_MIN:
            continue
        raw = Counter(cd.classes[int(v)] for v in cm[fill])
        grp = Counter(("background", *cd.group_names)[int(v)] for v in gmap[fill])
        ys, xs = np.nonzero(fill)
        out.append({"type": s["type"], "track_id": s["track_id"], "groups": s["groups"], "closed": s["closed"],
                    "area_px": int(fill.sum()), "len_px": round(tm.stroke_length(s["points"], s["closed"]), 1),
                    "bbox_xywh": [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)],
                    "face_ring_frac": round(face_frac, 3), "raw_inside": top_share(raw), "stable_inside": top_share(grp),
                    "raw_counts": dict(raw), "stable_counts": dict(grp)})
    return out


def cmd_q1() -> None:
    cfg = load_pipeline()
    face_id = group_id(cfg, "face")
    res = {"definisi": {"loop": f"closed, atau terbuka dengan panjang ≥ {LOOP_MIN_LEN_PX} px dan jarak ujung ≤ {LOOP_GAP_FRAC} × panjang",
                        "di_muka": f"≥ {FACE_RING_MIN:.0%} piksel cincin luar (1..{RING_PX} px di luar isi poligon) = grup face di stable/",
                        "tipe": "silhouette_hole / group_boundary / occlusion (silhouette luar tidak dihitung)"}, "clips": {}}
    for c in CLIPS:
        cd = t701.ClipData(c, cfg)
        rng = range(DIAG_RANGE[0], DIAG_RANGE[1] + 1) if c == DIAG else range(cd.n)
        per_frame, loops = {}, []
        for f in rng:
            name = cd.clip.names[f]
            gmap = stb.read_groups(cd.clip.groups_path(name))
            got = face_loops(cd, gmap, cd.classmap(f), contour_doc(cd.work, name), face_id)
            per_frame[f] = len(got)
            loops += [{"frame": f, **x} for x in got]
        n = len(per_frame)
        by_type = Counter(x["type"] for x in loops)
        raw_all, grp_all = Counter(), Counter()
        for x in loops:
            raw_all.update(x["raw_counts"])
            grp_all.update(x["stable_counts"])
        dom_raw = Counter(x["raw_inside"][0][0] for x in loops)
        frames_with = sorted(f for f, k in per_frame.items() if k)
        summ = {"frames": n, "range": [min(per_frame), max(per_frame)], "loops_total": len(loops),
                "loops_per_frame": round(len(loops) / n, 3), "frames_with_loop": len(frames_with),
                "frames_with_loop_pct": round(100 * len(frames_with) / n, 1), "frames_with_loop_runs": runs_str(frames_with),
                "max_per_frame": max(per_frame.values()), "by_type": dict(by_type),
                "by_type_per_frame": {t: round(v / n, 3) for t, v in by_type.items()},
                "area_px": ({"median": float(np.median([x["area_px"] for x in loops])), "min": min(x["area_px"] for x in loops),
                             "max": max(x["area_px"] for x in loops)} if loops else None),
                "area_by_type_median": {t: float(np.median([x["area_px"] for x in loops if x["type"] == t])) for t in by_type},
                "raw_inside_pixels": top_share(raw_all, 6), "stable_inside_pixels": top_share(grp_all, 6),
                "dominant_raw_class_per_loop": dict(dom_raw.most_common()),
                "groups_of_group_boundary": dict(Counter("|".join(x["groups"]) for x in loops if x["type"] == "group_boundary"))}
        res["clips"][c] = {"summary": summ, "per_frame": per_frame, "loops": loops}
        print(f"{c}: {json.dumps(summ, ensure_ascii=False)}", flush=True)
    for c in res["clips"]:
        for x in res["clips"][c]["loops"]:
            x.pop("raw_counts"), x.pop("stable_counts")
    jsave(OUT / "q1.json", res)


# ── Q2 ─────────────────────────────────────────────
def components(gmap: np.ndarray, face_id: int, min_region: int) -> tuple[np.ndarray, list[dict]]:
    """Komponen foreground 8-arah peta stable (= siluet [4], ≥ min_region_area) + wajah V1 (komponen face ≥ 0,1% frame,
    centroid di setengah atas bbox komponennya)."""
    H, W = gmap.shape
    n, lab, st, cen = cv2.connectedComponentsWithStats((gmap != stb.BACKGROUND_ID).astype(np.uint8), connectivity=8)
    nf, flab, fst, fcen = cv2.connectedComponentsWithStats((gmap == face_id).astype(np.uint8), connectivity=8)
    comps = {}
    for i in range(1, n):
        a = int(st[i, cv2.CC_STAT_AREA])
        if a < min_region:
            continue
        x, y, w, h = (int(v) for v in st[i, :4])
        comps[i] = {"label": i, "area": a, "area_pct": round(100 * a / (H * W), 2), "bbox_xywh": [x, y, w, h],
                    "centroid_xy": [round(float(cen[i][0]), 1), round(float(cen[i][1]), 1)],
                    "touches_edge": bool(x == 0 or y == 0 or x + w >= W or y + h >= H),
                    "edges": [e for e, t in (("kiri", x == 0), ("atas", y == 0), ("kanan", x + w >= W), ("bawah", y + h >= H)) if t],
                    "faces_v1": []}
    for j in range(1, nf):
        a = int(fst[j, cv2.CC_STAT_AREA])
        if a < FACE_MIN_FRAC * H * W:
            continue
        cx, cy = float(fcen[j][0]), float(fcen[j][1])
        owner = int(np.bincount(lab[flab == j]).argmax())
        if owner in comps:
            b = comps[owner]["bbox_xywh"]
            if cy < b[1] + b[3] / 2:
                comps[owner]["faces_v1"].append({"centroid_xy": [round(cx, 1), round(cy, 1)], "area": a})
    return lab, list(comps.values())


def row_zones(work: Path, names) -> np.ndarray:
    """Zona per baris (indeks ZONES) dari frames/ jendela: konten (bergerak) / teks (statis, terang) / hitam (statis, gelap)."""
    st = np.stack([cv2.imdecode(np.fromfile(work / "frames" / n, np.uint8), cv2.IMREAD_GRAYSCALE).astype(np.float32) for n in names])
    std, mean = st.std(axis=0).mean(axis=1), st.mean(axis=(0, 2))
    return np.where(std > ZONE_STD_MIN, 0, np.where(mean <= ZONE_BLACK_MAX, 2, 1))


def zone_spans(z: np.ndarray) -> dict:
    out = defaultdict(list)
    start = 0
    for y in range(1, len(z) + 1):
        if y == len(z) or z[y] != z[start]:
            out[ZONES[int(z[start])]].append([start, y - 1])
            start = y
    return dict(out)


def centrality(c: dict, H: int, W: int) -> float:
    """1 − jarak centroid ke tengah frame / setengah diagonal (0 di pojok, 1 di tengah)."""
    return 1.0 - float(np.hypot(c["centroid_xy"][0] - W / 2, c["centroid_xy"][1] - H / 2)) / float(np.hypot(W / 2, H / 2))


def classify(comps: list[dict], H: int, W: int, blob_min: float) -> None:
    """Q2 (per frame, tanpa pelacakan): orang = punya wajah V1 ATAU luas ≥ blob_min; derau = sisanya (kecil, tanpa wajah).
    Orang utama = orang dengan skor luas × kedekatan ke tengah tertinggi; orang lain = sisanya."""
    people = [c for c in comps if c["faces_v1"] or c["area"] >= blob_min * H * W]
    for c in comps:
        c["score"] = round(c["area"] * centrality(c, H, W), 1)
        c["class"] = "derau"
    if people:
        main = max(people, key=lambda c: c["score"])
        for c in people:
            c["class"] = "orang_utama" if c is main else "orang_lain"


def stroke_class(s: dict, lab: np.ndarray, cls_of: dict) -> str:
    p = pix(s["points"])
    H, W = lab.shape
    v = lab[np.clip(p[:, 1], 0, H - 1), np.clip(p[:, 0], 0, W - 1)]
    v = v[v > 0]
    if not len(v):
        return "lainnya"
    return cls_of.get(int(np.bincount(v).argmax()), "derau_kecil")      # komponen < min_region_area tidak punya siluet


def cmd_q2() -> None:
    cfg = load_pipeline()
    face_id = group_id(cfg, "face")
    work = CLIPS_DIR / DIAG
    clip = stb.load_clip(work)
    H, W = clip.height, clip.width
    min_region = cfg.vectorize.min_region_area
    blob_min = cfg.qc.blob_min
    rng = range(DIAG_RANGE[0], DIAG_RANGE[1] + 1)
    zrow = row_zones(work, [clip.names[f] for f in rng])
    frames, len_cls, len_zone = [], Counter(), Counter()
    for f in rng:
        name = clip.names[f]
        gmap = stb.read_groups(clip.groups_path(name))
        lab, comps = components(gmap, face_id, min_region)
        classify(comps, H, W, blob_min)
        for c in comps:
            zc = np.bincount(zrow[np.nonzero(lab == c["label"])[0]], minlength=len(ZONES))
            c["zone_pct"] = {ZONES[k]: round(100 * float(zc[k]) / c["area"], 1) for k in range(len(ZONES)) if zc[k]}
            c["zone"] = ZONES[int(zc.argmax())]
        cls_of = {c["label"]: c["class"] for c in comps}
        doc = contour_doc(work, name)
        lc = Counter()
        for s in doc["strokes"]:
            L = tm.stroke_length(s["points"], s["closed"])
            lc[stroke_class(s, lab, cls_of)] += L
            ys = np.clip(pix(s["points"])[:, 1], 0, H - 1)
            len_zone[ZONES[int(np.bincount(zrow[ys], minlength=len(ZONES)).argmax())]] += L
        len_cls.update(lc)
        n_sil = sum(s["type"] == "silhouette" for s in doc["strokes"])
        frames.append({"frame": f, "n_components": len(comps), "n_silhouette_strokes": n_sil,
                       "by_class": dict(Counter(c["class"] for c in comps)),
                       "faces_v1": sum(len(c["faces_v1"]) for c in comps),
                       "len_by_class": {k: round(v, 1) for k, v in lc.items()},
                       "components": sorted(comps, key=lambda c: -c["score"])})
    n = len(frames)
    cls_names = ("orang_utama", "orang_lain", "derau")
    tot_len = sum(len_cls.values()) or 1.0
    per_cls = {k: [fr["by_class"].get(k, 0) for fr in frames] for k in cls_names}
    allc = [c for fr in frames for c in fr["components"]]
    summ = {"frames": n, "range": list(DIAG_RANGE), "min_region_area": min_region, "blob_min": blob_min,
            "silhouette_strokes_per_frame": round(float(np.mean([fr["n_silhouette_strokes"] for fr in frames])), 3),
            "components_per_frame": round(float(np.mean([fr["n_components"] for fr in frames])), 3),
            "components_eq_silhouettes_frames": sum(fr["n_components"] == fr["n_silhouette_strokes"] for fr in frames),
            "per_class_per_frame": {k: round(float(np.mean(v)), 3) for k, v in per_cls.items()},
            "per_class_max": {k: int(max(v)) for k, v in per_cls.items()},
            "frames_without_main": sum(fr["by_class"].get("orang_utama", 0) == 0 for fr in frames),
            "frames_with_other_person": sum(fr["by_class"].get("orang_lain", 0) > 0 for fr in frames),
            "len_share_pct": {k: round(100 * v / tot_len, 1) for k, v in len_cls.most_common()},
            "zones_rows": zone_spans(zrow),
            "len_share_by_zone_pct": {k: round(100 * v / tot_len, 1) for k, v in len_zone.most_common()},
            "zone_of_class": {k: dict(Counter(c["zone"] for c in allc if c["class"] == k)) for k in cls_names},
            "faces_v1_hist": dict(sorted(Counter(fr["faces_v1"] for fr in frames).items())),
            "class_stats": {k: {"n": sum(c["class"] == k for c in allc),
                                "area_pct_median": float(np.median([c["area_pct"] for c in allc if c["class"] == k])) if any(c["class"] == k for c in allc) else None,
                                "area_pct_range": ([min(c["area_pct"] for c in allc if c["class"] == k), max(c["area_pct"] for c in allc if c["class"] == k)]
                                                   if any(c["class"] == k for c in allc) else None),
                                "touches_edge_pct": round(100 * float(np.mean([c["touches_edge"] for c in allc if c["class"] == k])), 1) if any(c["class"] == k for c in allc) else None,
                                "with_face_v1_pct": round(100 * float(np.mean([bool(c["faces_v1"]) for c in allc if c["class"] == k])), 1) if any(c["class"] == k for c in allc) else None,
                                "edges": dict(Counter(e for c in allc if c["class"] == k for e in c["edges"]))}
                            for k in cls_names},
            "main_multi_face_frames": sum(any(c["class"] == "orang_utama" and len(c["faces_v1"]) >= 2 for c in fr["components"]) for fr in frames)}
    jsave(OUT / "q2.json", {"summary": summ, "frames": frames})
    print(json.dumps(summ, indent=1, ensure_ascii=False))
    for fr in frames[::8]:
        print(f"  f{fr['frame']}: " + "; ".join(f"{c['class']} {c['area_pct']}% bbox {c['bbox_xywh']} tepi {c['edges']} wajah {len(c['faces_v1'])} zona {c['zone_pct']}"
                                             for c in fr["components"]))


# ── VF ─────────────────────────────────────────────
def vf_fill(g: np.ndarray, face_id: int) -> tuple[np.ndarray, int, int]:
    """Komponen background 4-arah yang tidak menyentuh tepi frame dan seluruh cincin 1 px-nya (8-arah) = face → face."""
    H, W = g.shape
    n, lab, st, _ = cv2.connectedComponentsWithStats((g == stb.BACKGROUND_ID).astype(np.uint8), connectivity=4)
    out, regions, px = g.copy(), 0, 0
    for i in range(1, n):
        x, y, w, h, a = (int(v) for v in st[i])
        if x == 0 or y == 0 or x + w >= W or y + h >= H:
            continue
        sl = (slice(y - 1, y + h + 1), slice(x - 1, x + w + 1))
        comp = lab[sl] == i
        ring = cv2.dilate(comp.astype(np.uint8), K3).astype(bool) & ~comp
        if (g[sl][ring] == face_id).all():
            out[sl][comp] = face_id
            regions += 1
            px += a
    return out, regions, px


def cmd_vfscan() -> None:
    cfg = load_pipeline()
    face_id = group_id(cfg, "face")
    q1 = jload(OUT / "q1.json")
    res = {}
    for c in CLIPS:
        clip = stb.load_clip(CLIPS_DIR / c)
        rng = range(SIM_N) if c == DIAG else range(len(clip.names))
        changed, regions, px = [], 0, 0
        holes = [x for x in q1["clips"][c]["loops"] if x["type"] == "silhouette_hole"]
        covered = 0
        for f in rng:
            g = stb.read_groups(clip.groups_path(clip.names[f]))
            g2, r, p = vf_fill(g, face_id)
            if r:
                changed.append(f)
                regions += r
                px += p
            for x in (h for h in holes if h["frame"] == f):
                bx, by, bw, bh = x["bbox_xywh"]
                covered += bool(((g2 != g)[by:by + bh, bx:bx + bw]).any())
        res[c] = {"frames": len(rng), "frames_changed": len(changed), "frames_changed_runs": runs_str(changed),
                  "regions_filled": regions, "px_filled": px, "q1_holes": len(holes), "q1_holes_filled": covered}
        print(c, json.dumps(res[c], ensure_ascii=False), flush=True)
    jsave(OUT / "vfscan.json", res)


# ── orang (S1 / S2) ────────────────────────────────
def people(g: np.ndarray, face_id: int, blob_min: float) -> list[dict]:
    """Kandidat orang per frame (docs/04 "Aturan simulasi Tahap 1b" langkah 1–4)."""
    H, W = g.shape
    n, lab, st, _ = cv2.connectedComponentsWithStats((g != stb.BACKGROUND_ID).astype(np.uint8), connectivity=8)
    big = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] > blob_min * H * W]
    masks = {b: lab == b for b in big}
    if big:
        dists = {b: cv2.distanceTransform((lab != b).astype(np.uint8), cv2.DIST_L2, 5) for b in big}
        for i in range(1, n):
            if i in big or st[i, cv2.CC_STAT_AREA] < ASSIGN_MIN_PX:
                continue
            comp = lab == i
            d = {b: float(dists[b][comp].min()) for b in big}
            b = min(d, key=d.get)
            if d[b] <= ATTACH_MAX_PX:
                masks[b] |= comp
    nf, flab, fst, fcen = cv2.connectedComponentsWithStats((g == face_id).astype(np.uint8), connectivity=8)
    faces = []
    for j in range(1, nf):
        a = int(fst[j, cv2.CC_STAT_AREA])
        if a < FACE_MIN_FRAC * H * W:
            continue
        comp = flab == j
        inside = [int((comp & masks[b]).sum()) for b in big]
        bi = big[int(np.argmax(inside))] if inside and max(inside) * 2 > a else None
        faces.append({"centroid_xy": [float(fcen[j][0]), float(fcen[j][1])], "area": a, "blob": bi,
                      "comp": int(np.bincount(lab[comp]).argmax())})
    units = {("blob", b): masks[b] for b in big}
    v1 = []
    for fc in faces:
        if fc["blob"] is None:
            u = ("blob", fc["comp"]) if fc["comp"] in masks else ("comp", fc["comp"])
            v1.append({**fc, "unit": u})
            units.setdefault(u, lab == fc["comp"])
            continue
        ys, xs = np.nonzero(masks[fc["blob"]])
        if fc["centroid_xy"][1] < ys.min() + (ys.max() - ys.min() + 1) / 2:
            v1.append({**fc, "unit": ("blob", fc["blob"])})
    parts = []
    for u, m in units.items():
        fids = [k for k, fc in enumerate(v1) if fc["unit"] == u]
        if len(fids) <= 1:
            parts.append({"unit": u, "face": v1[fids[0]]["centroid_xy"] if fids else None, "split": False, "mask": m})
            continue
        ys, xs = np.nonzero(m)
        cen = np.asarray([v1[k]["centroid_xy"] for k in fids], float)
        own = np.argmin((xs[:, None] - cen[None, :, 0]) ** 2 + (ys[:, None] - cen[None, :, 1]) ** 2, axis=1)
        for q, k in enumerate(fids):
            mm = np.zeros((H, W), bool)
            mm[ys[own == q], xs[own == q]] = True
            parts.append({"unit": u, "face": v1[k]["centroid_xy"], "split": True, "mask": mm})
    half_diag = float(np.hypot(W / 2, H / 2))
    for p in parts:
        ys, xs = np.nonzero(p["mask"])
        p["area"] = int(len(xs))
        p["centroid_xy"] = [float(xs.mean()), float(ys.mean())] if len(xs) else [W / 2, H / 2]
        p["centrality"] = 1.0 - float(np.hypot(p["centroid_xy"][0] - W / 2, p["centroid_xy"][1] - H / 2)) / half_diag
        p["score"] = p["area"] * p["centrality"]
    return [p for p in parts if p["area"]]


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    u = np.count_nonzero(a | b)
    return np.count_nonzero(a & b) / u if u else 0.0


def assign_ids(prev: list[dict], cur: list[dict], next_id: int, H: int) -> int:
    """Langkah 5: Hungarian; pasangan wajah (jarak <= 0,08 H) atau tanpa wajah (IoU >= 0,3). Id baru untuk sisanya."""
    dmax = TRACK_FACE_FRAC * H
    big = 1e9
    if prev and cur:
        cost = np.full((len(prev), len(cur)), big)
        for i, a in enumerate(prev):
            for j, b in enumerate(cur):
                if a["face"] is not None and b["face"] is not None:
                    d = float(np.hypot(a["face"][0] - b["face"][0], a["face"][1] - b["face"][1]))
                    if d <= dmax:
                        cost[i, j] = d
                else:
                    iou = mask_iou(a["mask"], b["mask"])
                    if iou >= TRACK_IOU_MIN:
                        cost[i, j] = (1.0 - iou) * dmax
        for i, j in zip(*linear_sum_assignment(cost)):
            if cost[i, j] < big:
                cur[j]["id"] = prev[i]["id"]
    for p in cur:
        if "id" not in p:
            p["id"] = next_id
            next_id += 1
    return next_id


def select_people(clip, face_id: int, blob_min: float, k: int) -> tuple[list[np.ndarray], list[dict]]:
    """VF → orang → pelacakan → k skor teratas. Return (masker terpilih per frame, ringkasan per frame)."""
    prev, next_id, keep, rows, prev_sel = [], 1, [], [], None
    for f in range(SIM_N):
        g, _, _ = vf_fill(stb.read_groups(clip.groups_path(clip.names[f])), face_id)
        cur = people(g, face_id, blob_min)
        next_id = assign_ids(prev, cur, next_id, g.shape[0])
        ranked = sorted(cur, key=lambda p: (-p["score"], p["id"]))
        sel = ranked[:k]
        m = np.zeros(g.shape, bool)
        for p in sel:
            m |= p["mask"]
        keep.append(m)
        ids = sorted(p["id"] for p in sel)
        row = {"frame": f, "n_people": len(cur), "selected": ids,
               "people": [{"id": p["id"], "area_pct": round(100 * p["area"] / g.size, 2), "score": round(p["score"]),
                           "face": [round(v, 1) for v in p["face"]] if p["face"] else None, "split": p["split"],
                           "centroid_xy": [round(v, 1) for v in p["centroid_xy"]]} for p in ranked]}
        if prev_sel is not None and ids != prev_sel:
            cur_ids = {p["id"] for p in cur}
            pos_prev = {p["id"]: p["face"] or p["centroid_xy"] for p in prev}
            pos_cur = {p["id"]: p["face"] or p["centroid_xy"] for p in cur}
            dropped = [i for i in prev_sel if i not in ids]
            row["switch"] = {"dari": prev_sel, "ke": ids,
                             "id_lama_masih_ada": [i for i in prev_sel if i not in ids and i in cur_ids],
                             "id_lama_hilang": [i for i in prev_sel if i not in cur_ids],
                             # lompatan posisi (wajah, atau centroid bila tanpa wajah) id baru vs id terpilih yang keluar di t − 1
                             "lompatan_px": [[i, o, round(float(np.hypot(pos_cur[i][0] - pos_prev[o][0], pos_cur[i][1] - pos_prev[o][1])), 1)]
                                             for i in ids if i not in prev_sel for o in dropped][:4]}
        rows.append(row)
        prev, prev_sel = cur, ids
    return keep, rows


# ── salinan scratch + render ───────────────────────
def vdir(var: str, clip: str = DIAG) -> Path:
    return SCR / var / clip


def prepare_copy(var: str, clip: str) -> Path:
    """Pola t701_proto.prepare: meta.json + qc_report.json + stable/ (groups disalin, depth_smooth di-hard-link)."""
    src, d = CLIPS_DIR / clip, vdir(var, clip)
    if d.exists():
        shutil.rmtree(d)
    (d / "stable" / "groups").mkdir(parents=True)
    (d / "stable" / "depth_smooth").mkdir(parents=True)
    for f in ("meta.json", "qc_report.json"):
        shutil.copyfile(src / f, d / f)
    for f in ("manifest.json", "frames.jsonl"):
        if (src / "stable" / f).is_file():
            shutil.copyfile(src / "stable" / f, d / "stable" / f)
    for p in sorted((src / "stable" / "groups").glob("*.png")):
        shutil.copyfile(p, d / "stable" / "groups" / p.name)
    for p in sorted((src / "stable" / "depth_smooth").glob("*.npy")):
        try:
            os.link(p, d / "stable" / "depth_smooth" / p.name)
        except OSError:
            shutil.copyfile(p, d / "stable" / "depth_smooth" / p.name)
    return d


def cmd_prepare(var: str) -> None:
    cfg = load_pipeline()
    face_id = group_id(cfg, "face")
    d = prepare_copy(var, DIAG)
    clip = stb.load_clip(d)
    info = {"var": var, "frames_changed": [], "px_changed": 0}
    if var == "V0":
        jsave(SCR / var / "prepare.json", info)
        return
    keep, rows = (select_people(clip, face_id, cfg.qc.blob_min, K_OF[var]) if var in K_OF else (None, None))
    for f in range(SIM_N):
        path = clip.groups_path(clip.names[f])
        g = stb.read_groups(path)
        g2, _, _ = vf_fill(g, face_id)
        if keep is not None:
            g2 = np.where(keep[f], g2, stb.BACKGROUND_ID).astype(g.dtype)
        if (g2 != g).any():
            info["frames_changed"].append(f)
            info["px_changed"] += int((g2 != g).sum())
            stb.write_groups(path, g2)
    info["frames_changed"] = runs_str(info["frames_changed"])
    if rows is not None:
        sw = [r for r in rows if "switch" in r]
        info.update({"k": K_OF[var], "frames_without_selected": [r["frame"] for r in rows if not r["selected"]],
                     "n_switches": len(sw), "switch_frames": runs_str([r["frame"] for r in sw]),
                     "switch_rank_swap_frames": runs_str([r["frame"] for r in sw if r["switch"]["id_lama_masih_ada"]]),
                     "switch_track_lost_frames": runs_str([r["frame"] for r in sw if r["switch"]["id_lama_hilang"]]),
                     "switch_details": {r["frame"]: r["switch"] for r in sw},
                     "ids_used": sorted({i for r in rows for i in r["selected"]}),
                     "frames_per_selected_id": dict(Counter(i for r in rows for i in r["selected"])),
                     "people_per_frame": round(float(np.mean([r["n_people"] for r in rows])), 3),
                     "rows": rows})
    jsave(SCR / var / "prepare.json", info)
    print(json.dumps({k: v for k, v in info.items() if k != "rows"}, ensure_ascii=False))


def cfg_path(var: str) -> Path:
    p = SCR / var / "cfg.yaml"
    text = (ROOT / "configs" / "default.yaml").read_text(encoding="utf-8")
    line = '  out_dir: "out"'
    if text.count(line) != 1:
        raise SystemExit("configs/default.yaml: baris out_dir tidak ditemukan tepat sekali")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text.replace(line, f'  out_dir: "{(SCR / var / "out").as_posix()}"'), encoding="utf-8")
    return p


def run_logged(cmd: list[str], log: Path) -> tuple[int, float]:
    t0 = time.perf_counter()
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(r.stdout + "\n--- stderr ---\n" + r.stderr, encoding="utf-8")
    return r.returncode, time.perf_counter() - t0


def cmd_render(var: str, clip: str = DIAG) -> None:
    d, cfg = vdir(var, clip), cfg_path(var)
    n = SIM_N if clip == DIAG else len(stb.load_clip(d).names)
    win = ["--from", "0", "--limit", str(n)] if clip == DIAG else []
    steps = [("vectorize", [sys.executable, "-m", "rotoscope.vectorize", "--config", str(cfg), "--work-dir", str(d)]
              + (["--limit", str(n)] if clip == DIAG else [])),
             ("stylize", [sys.executable, "-m", "rotoscope.stylize", "--config", str(cfg), "--work-dir", str(d), *win]),
             ("export", [sys.executable, "-m", "rotoscope.export", "--config", str(cfg), "--work-dir", str(d), *win])]
    for name, cmd in steps:
        rc, dt = run_logged(cmd, SCR / "logs" / f"{name}_{var}_{clip}.log")
        print(f"  {clip} {var} {name}: rc {rc}, {dt:.1f} s", flush=True)
        if rc:
            raise SystemExit(f"{name} gagal — lihat work/t305_scratch/1b/logs/{name}_{var}_{clip}.log")


def cmd_prepare_vf(clip: str) -> None:
    cfg = load_pipeline()
    face_id = group_id(cfg, "face")
    d = prepare_copy("VF", clip)
    c = stb.load_clip(d)
    changed = []
    for f, name in enumerate(c.names):
        g = stb.read_groups(c.groups_path(name))
        g2, r, _ = vf_fill(g, face_id)
        if r:
            stb.write_groups(c.groups_path(name), g2)
            changed.append(f)
    jsave(SCR / "VF" / f"prepare_{clip}.json", {"clip": clip, "frames_changed": changed})


# ── metrik varian ──────────────────────────────────
def cmd_metrics() -> None:
    cfg = load_pipeline()
    face_id = group_id(cfg, "face")
    prod = CLIPS_DIR / DIAG
    pclip = stb.load_clip(prod)
    H, W = pclip.height, pclip.width
    labs, cls_maps = [], []
    for f in range(SIM_N):
        lab, comps = components(stb.read_groups(pclip.groups_path(pclip.names[f])), face_id, cfg.vectorize.min_region_area)
        classify(comps, H, W, cfg.qc.blob_min)
        labs.append(lab)
        cls_maps.append({c["label"]: c["class"] for c in comps})
    res = {}
    for var in VARIANTS:
        d = vdir(var) / "contours"
        names = [pclip.names[f] for f in range(SIM_N)]
        docs = [contour_doc(vdir(var), n) for n in names]
        lens = [{(s["type"], s["track_id"]): tm.stroke_length(s["points"], s["closed"]) for s in x["strokes"]} for x in docs]
        b_all = [len(set(lens[t]) - set(lens[t - 1])) for t in range(1, len(lens))]
        lc = Counter()
        for f, x in enumerate(docs):
            for s in x["strokes"]:
                lc[stroke_class(s, labs[f], cls_maps[f])] += tm.stroke_length(s["points"], s["closed"])
        tot = sum(lc.values()) or 1.0
        prep = jload(SCR / var / "prepare.json")
        r = {"strokes_per_frame": round(float(np.mean([len(x["strokes"]) for x in docs])), 2),
             "silhouettes_per_frame": round(float(np.mean([sum(s["type"] == "silhouette" for s in x["strokes"]) for x in docs])), 3),
             "holes_per_frame": round(float(np.mean([sum(s["type"] == "silhouette_hole" for s in x["strokes"]) for x in docs])), 3),
             "occlusion_per_frame": round(float(np.mean([sum(s["type"] == "occlusion" for s in x["strokes"]) for x in docs])), 3),
             "births_all_per_frame": round(float(np.mean(b_all)), 3), "births_occ_per_frame": round(occ_births(docs), 3),
             "len_share_pct": {k: round(100 * v / tot, 1) for k, v in lc.most_common()},
             "len_total_per_frame_px": round(tot / len(docs), 1),
             "frames_without_selected": len(prep.get("frames_without_selected", [])) if var in K_OF else None,
             "frames_empty_contours": sum(not x["strokes"] for x in docs),
             "clip_stats": {k: jload(d / "clip_stats.json")[k] for k in ("t_high", "t_low")},
             "groups_frames_changed": prep["frames_changed"], "groups_px_changed": prep["px_changed"]}
        if var in K_OF:
            r.update({k: prep[k] for k in ("n_switches", "switch_frames", "switch_rank_swap_frames", "switch_track_lost_frames",
                                            "switch_details", "ids_used", "frames_per_selected_id", "people_per_frame")})
        res[var] = r
    same_json = sum((vdir("V0") / "contours" / f"{Path(n).stem}.json").read_bytes() == (prod / "contours" / f"{Path(n).stem}.json").read_bytes()
                    for n in pclip.names[:SIM_N])
    same_png = sum(t701.file_sha(vdir("V0") / "strokes" / f"{Path(n).stem}.png") == t701.file_sha(prod / "strokes" / f"{Path(n).stem}.png")
                   for n in pclip.names[:SIM_N])
    res["V0_vs_produksi"] = {"contours_json_identik": same_json, "strokes_png_identik": same_png, "dari": SIM_N}
    res["VF_klip_lain"] = {}
    for c in CLIPS:
        p = SCR / "VF" / f"prepare_{c}.json"
        if c == DIAG or not p.is_file():
            continue
        oc = stb.load_clip(CLIPS_DIR / c)
        st_diff, png_diff = [], []
        for f, n in enumerate(oc.names):
            stem = Path(n).stem
            if contour_doc(vdir("VF", c), n)["strokes"] != contour_doc(CLIPS_DIR / c, n)["strokes"]:
                st_diff.append(f)
            if t701.file_sha(vdir("VF", c) / "strokes" / f"{stem}.png") != t701.file_sha(CLIPS_DIR / c / "strokes" / f"{stem}.png"):
                png_diff.append(f)
        res["VF_klip_lain"][c] = {"groups_frames_changed": jload(p)["frames_changed"], "contours_strokes_differ": runs_str(st_diff),
                                  "strokes_png_differ": runs_str(png_diff),
                                  "clip_stats_identik": all(jload(vdir("VF", c) / "contours" / "clip_stats.json")[k]
                                                            == jload(CLIPS_DIR / c / "contours" / "clip_stats.json")[k] for k in ("t_high", "t_low"))}
    jsave(OUT / "metrics.json", res)
    print(json.dumps(res, indent=1, ensure_ascii=False))


# ── bahan penilaian ────────────────────────────────
def preview(var: str, clip: str = DIAG) -> Path:
    got = sorted((SCR / var / "out").glob(f"{clip}.preview_0-{SIM_N - 1}.mp4" if clip == DIAG else f"{clip}.mp4"))
    if len(got) != 1:
        raise SystemExit(f"preview {var} {clip}: {len(got)} berkas")
    return got[0]


def compose(panels: list[np.ndarray], labels: list[str], f: int, n: int) -> np.ndarray:
    ps = [fit_h(p, PANEL_H) for p in panels]
    gap = np.full((PANEL_H, 12, 3), 255, np.uint8)
    body = ps[0]
    tops = [label_bar(ps[0].shape[1], labels[0])]
    for p, lbl in zip(ps[1:], labels[1:]):
        body = np.hstack([body, gap, p])
        tops += [np.full((label_bar(10, "").shape[0], 12, 3), 255, np.uint8), label_bar(p.shape[1], lbl)]
    img = np.vstack([np.hstack(tops), body])
    return np.ascontiguousarray(np.vstack([img, time_bar(img.shape[1], f, n)]))


def write_pair(frames: list[np.ndarray], stem: str) -> dict:
    size = (frames[0].shape[1], frames[0].shape[0])
    enc = (encoder(NILAI / f"{stem}.mp4", size), encoder(NILAI / f"{stem}_lambat_0.5x.mp4", size))
    for img in frames:
        raw = img.tobytes()
        enc[0].stdin.write(raw)
        enc[1].stdin.write(raw)
        enc[1].stdin.write(raw)
    for p in enc:
        p.stdin.close()
        if p.wait() != 0:
            raise SystemExit(f"{stem}: ffmpeg gagal")
    return {"frames": len(frames), "size": list(size)}


def cmd_media() -> None:
    NILAI.mkdir(parents=True, exist_ok=True)
    clip = stb.load_clip(CLIPS_DIR / DIAG)
    n_all = len(clip.names)
    vids = {v: list(read_frames(preview(v))) for v in VARIANTS}
    if any(len(x) != SIM_N for x in vids.values()):
        raise SystemExit(f"jumlah frame preview: { {v: len(x) for v, x in vids.items()} }")
    fotos = [cv2.imdecode(np.fromfile(clip.work_dir / "frames" / clip.names[f], np.uint8), cv2.IMREAD_COLOR) for f in range(SIM_N)]
    info = {}
    for stem, vs in (("klip5_A", ("V0", "VF")), ("klip5_B", ("S1", "S2"))):
        frames = [compose([fotos[f]] + [vids[v][f] for v in vs], ["FOTO", *vs], f, n_all) for f in range(SIM_N)]
        info[stem] = write_pair(frames, stem)
        print(f"  {stem}: {info[stem]}", flush=True)
    scan = jload(OUT / "vfscan.json")
    for c in CLIPS:
        if c == DIAG or not scan[c]["frames_changed"]:
            continue
        cl = stb.load_clip(CLIPS_DIR / c)
        changed = jload(SCR / "VF" / f"prepare_{c}.json")["frames_changed"]
        base = list(read_frames(ROOT / "out" / f"{c}.mp4"))
        vf = list(read_frames(preview("VF", c)))
        frames = []
        for i, f in enumerate(changed):
            img = compose([base[f], vf[f]], ["BASE", f"VF ({i + 1}/{len(changed)})"], f, len(cl.names))
            frames += [img] * 48                    # tiap frame berubah ditahan 2 s
        info[f"{c}_VF"] = write_pair(frames, f"{c}_base_vs_VF")
        print(f"  {c}_base_vs_VF: {info[f'{c}_VF']} ({len(changed)} frame berubah)", flush=True)
    jsave(OUT / "media.json", info)


def main(argv: list[str]) -> None:
    if argv[:1] == ["hash"] and argv[1:2] in (["before"], ["after"]):
        cmd_hash(argv[1])
    elif argv == ["q1"]:
        cmd_q1()
    elif argv == ["q2"]:
        cmd_q2()
    elif argv == ["vfscan"]:
        cmd_vfscan()
    elif argv[:1] == ["prepare"] and len(argv) == 2 and argv[1] in VARIANTS:
        cmd_prepare(argv[1])
    elif argv[:1] == ["render"] and len(argv) in (2, 3) and argv[1] in VARIANTS:
        cmd_render(argv[1], *argv[2:])
    elif argv[:1] == ["prepare_vf"] and len(argv) == 2 and argv[1] in CLIPS and argv[1] != DIAG:
        cmd_prepare_vf(argv[1])
    elif argv == ["metrics"]:
        cmd_metrics()
    elif argv == ["media"]:
        cmd_media()
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
