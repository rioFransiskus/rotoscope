"""Alat sekali pakai T-201a (seperti look_test.py; BUKAN bagian src/): overlay kontur [4] di atas frame asli.

    python scripts/contour_overlay.py --work-dir work/clips/test [--out-dir work/t201a]
        [--ranges 73-92,183-202,225-240] [--hole-frames 20] [--scale 2] [--samples 80,190,233] [--suffix _fix1]

Membaca <work-dir>/contours/frame_*.json + frames/ (TIDAK menulis apa pun di folder klip). Menulis ke --out-dir
(default work/t201a, ter-ignore):
  overlay_<klip>.mp4   frame di --ranges + sampel frame ber-lubang, skala --scale (default 2×), H.264 lewat
                       export.encode; label "tanpa garis oklusi / anchor - BELUM representatif"
  sample_<klip>_fNNNNN.png   frame --samples (+ satu frame ber-lubang)
  summary_<klip>.json  angka stage [4] dari contours/frames.jsonl + JSON frame (waktu, ukuran, strok per tipe, ...)

Warna: silhouette merah, silhouette_hole cyan, group_boundary hijau. Run titik yang menempel TEPI FRAME
(koordinat tepat di baris/kolom tepi) digambar magenta putus-putus supaya penilaian tertuju pada kontur sebenarnya.
Koordinat strok = pusat piksel (i + 0.5) → digambar di (x · skala − 0.5).

--dropped: video TAMBAHAN overlay_<klip>_dropped.mp4 (frame yang sama; tanpa PNG / ringkasan, tidak menimpa berkas
lain) dengan kontur yang DIBUANG filter ukuran digambar abu-abu + label luas px: lubang < min_hole_area (abu-abu
terang) dan komponen luar < min_region_area (abu-abu gelap). Ambang dibaca dari contours/manifest.json; kontur
dihitung ulang dari stable/groups dengan fungsi yang sama dengan vectorize.silhouette_strokes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

from rotoscope import export as ex
from rotoscope import stabilize as stb
from rotoscope import vectorize as vec
from rotoscope.stage_common import StageError, read_jsonl

SHIFT = 4                                    # bit subpiksel cv2.polylines
COLORS = {"silhouette": (0, 0, 255), "silhouette_hole": (255, 255, 0), "group_boundary": (0, 200, 0)}   # BGR
EDGE_COLOR = (255, 0, 255)
DASH_ON, DASH_OFF = 3, 3                     # panjang dash (segmen) untuk run tepi
FADE = 0.45                                  # frame asli dipucatkan ke putih supaya garis menonjol
DROP_HOLE_COLOR, DROP_REGION_COLOR = (170, 170, 170), (80, 80, 80)    # abu-abu terang / gelap
LABEL = "tanpa garis oklusi / anchor - BELUM representatif"
DEFAULT_RANGES = "73-92,183-202,225-240"
DEFAULT_SAMPLES = "80,190,233"


def parse_ranges(text: str) -> list[int]:
    out: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def on_edge(p, w: int, h: int) -> int:
    """Sisi bingkai (bitmask) tempat titik berada persis di baris/kolom tepi; 0 = bukan tepi."""
    x, y = p
    return (1 if x == 0.5 else 0) | (2 if x == w - 0.5 else 0) | (4 if y == 0.5 else 0) | (8 if y == h - 0.5 else 0)


def px(p, scale: int) -> tuple[int, int]:
    return round((p[0] * scale - 0.5) * (1 << SHIFT)), round((p[1] * scale - 0.5) * (1 << SHIFT))


def draw_strokes(img: np.ndarray, strokes: list[dict], w: int, h: int, scale: int) -> dict:
    """Gambar semua strok pada img (in-place). Return jumlah segmen biasa vs segmen tepi."""
    n_edge = n_norm = 0
    for s in strokes:
        pts = s["points"]
        color = COLORS[s["type"]]
        n = len(pts)
        closed = bool(s["closed"])
        pairs = [(i, i + 1) for i in range(n - 1)] + ([(n - 1, 0)] if closed and n > 1 else [])
        run = []                                     # titik berurutan non-tepi → satu polyline
        dash = 0
        for i, j in pairs:
            shared = on_edge(pts[i], w, h) & on_edge(pts[j], w, h)
            if shared:                               # segmen sepanjang tepi: putus-putus, magenta
                if run:
                    _poly(img, run, color, scale)
                    run = []
                if (dash // DASH_ON) % 2 == 0:
                    cv2.line(img, px(pts[i], scale), px(pts[j], scale), EDGE_COLOR, 2, cv2.LINE_AA, SHIFT)
                dash += 1
                n_edge += 1
            else:
                dash = 0
                if not run:
                    run = [pts[i]]
                run.append(pts[j])
                n_norm += 1
        if run:
            _poly(img, run, color, scale)
        if n == 1:
            cv2.circle(img, px(pts[0], scale), 2, color, -1, cv2.LINE_AA, SHIFT)
    return {"segments": n_norm, "edge_segments": n_edge}


def _poly(img: np.ndarray, pts: list, color: tuple, scale: int) -> None:
    arr = np.array([px(p, scale) for p in pts], np.int32)
    cv2.polylines(img, [arr.reshape(-1, 1, 2)], False, color, 2, cv2.LINE_AA, SHIFT)


def text_box(img: np.ndarray, lines: list[tuple[str, tuple]], x: int, y: int, scale: float = 0.55) -> None:
    font, th = cv2.FONT_HERSHEY_SIMPLEX, 1
    sizes = [cv2.getTextSize(t, font, scale, th)[0] for t, _ in lines]
    bw, lh = max(s[0] for s in sizes) + 12, max(s[1] for s in sizes) + 8
    cv2.rectangle(img, (x, y), (x + bw, y + lh * len(lines) + 6), (30, 30, 30), -1)
    for k, (t, c) in enumerate(lines):
        cv2.putText(img, t, (x + 6, y + lh * (k + 1)), font, scale, c, th, cv2.LINE_AA)


def dropped_contours(gmap: np.ndarray, min_region: int, min_hole: int) -> list[dict]:
    """Kontur luar < min_region dan lubang < min_hole (induk lolos): [{kind, area, points (pusat piksel)}].
    Luas = jumlah piksel, cara yang sama dengan vectorize.silhouette_strokes."""
    pad = vec.PAD_PX
    padded = cv2.copyMakeBorder((gmap != stb.BACKGROUND_ID).astype(np.uint8), pad, pad, pad, pad,
                                cv2.BORDER_CONSTANT, value=0)
    contours, hier = cv2.findContours(padded, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return []
    parent = [int(h[3]) for h in hier[0]]
    _, lab, st, _ = cv2.connectedComponentsWithStats(padded, connectivity=8)
    _, blab, bst, _ = cv2.connectedComponentsWithStats(1 - padded, connectivity=4)
    out, kept_outer = [], {}
    for i, cnt in enumerate(contours):
        if parent[i] < 0:
            x, y = cnt[0, 0]
            area = int(st[lab[y, x], cv2.CC_STAT_AREA])
            kept_outer[i] = area >= min_region
            if not kept_outer[i]:
                out.append({"kind": "region", "area": area, "points": cnt.reshape(-1, 2)})
    for i, cnt in enumerate(contours):
        if parent[i] >= 0 and kept_outer[parent[i]]:
            label = vec._hole_label(padded, blab, cnt)
            area = int(bst[label, cv2.CC_STAT_AREA]) if label >= 0 else 0
            if area < min_hole:
                out.append({"kind": "hole", "area": area, "points": cnt.reshape(-1, 2)})
    for d in out:
        d["points"] = [[x - pad + vec.PIXEL_CENTER, y - pad + vec.PIXEL_CENTER] for x, y in d["points"].tolist()]
    return out


def draw_dropped(img: np.ndarray, drops: list[dict], scale: int) -> None:
    for d in drops:
        color = DROP_HOLE_COLOR if d["kind"] == "hole" else DROP_REGION_COLOR
        arr = np.array([px(p, scale) for p in d["points"]], np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [arr], True, color, 2, cv2.LINE_AA, SHIFT)
        cx, cy = np.mean([[p[0], p[1]] for p in d["points"]], axis=0) * scale
        cv2.putText(img, f"{d['area']}px", (int(cx) - 14, int(cy) + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (20, 20, 20), 1, cv2.LINE_AA)


def render(clip: Path, index: int, scale: int, tag: str, dropped: dict | None = None) -> np.ndarray:
    """Frame asli (dipucatkan) × skala + strok + legenda → RGB uint8. `dropped` = {min_region_area, min_hole_area}
    → kontur yang dibuang filter ukuran ikut digambar (abu-abu + luas px)."""
    doc = read_json(clip / "contours" / f"frame_{index:05d}.json")
    w, h = doc["width"], doc["height"]
    bgr = cv2.imdecode(np.fromfile(clip / "frames" / f"frame_{index:05d}.png", np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise StageError(f"frame {index} tidak terbaca di {clip / 'frames'}")
    big = cv2.resize(bgr, (w * scale, h * scale), interpolation=cv2.INTER_LINEAR)
    big = cv2.addWeighted(big, 1 - FADE, np.full_like(big, 255), FADE, 0)
    st = draw_strokes(big, doc["strokes"], w, h, scale)
    kinds = {k: sum(s["type"] == k for s in doc["strokes"]) for k in COLORS}
    legend = [(LABEL, (255, 255, 255)),
              (f"frame {index}  [{tag}]", (255, 255, 255)),
              (f"silhouette {kinds['silhouette']} (merah)", COLORS["silhouette"]),
              (f"silhouette_hole {kinds['silhouette_hole']} (cyan)", COLORS["silhouette_hole"]),
              (f"group_boundary {kinds['group_boundary']} (hijau)", COLORS["group_boundary"]),
              (f"tepi frame {st['edge_segments']} seg (magenta putus-putus)", EDGE_COLOR)]
    if dropped is not None:
        gmap = stb.read_groups(clip / "stable" / "groups" / f"frame_{index:05d}.png")
        if gmap is None:
            raise StageError(f"peta grup frame {index} tidak terbaca di {clip / 'stable' / 'groups'}")
        drops = dropped_contours(gmap, dropped["min_region_area"], dropped["min_hole_area"])
        draw_dropped(big, drops, scale)
        n_hole = sum(d["kind"] == "hole" for d in drops)
        legend += [(f"DIBUANG lubang < {dropped['min_hole_area']} px: {n_hole} (abu-abu terang, label luas)",
                    DROP_HOLE_COLOR),
                   (f"DIBUANG komponen luar < {dropped['min_region_area']} px: {len(drops) - n_hole} "
                    f"(abu-abu gelap)", (150, 150, 150))]
    text_box(big, legend, 6, 6)
    return cv2.cvtColor(big, cv2.COLOR_BGR2RGB)


class OverlaySource(ex.FrameSource):
    kind = "overlay"

    def __init__(self, clip: Path, plan: dict[str, tuple[int, str]], scale: int, dropped: dict | None = None):
        self.clip, self.plan, self.scale, self.dropped = clip, plan, scale, dropped

    def check(self, names: list[str]) -> None:
        pass

    def render(self, name: str) -> np.ndarray:
        index, tag = self.plan[name]
        return render(self.clip, index, self.scale, tag, self.dropped)


def hole_frames(records: dict[str, dict], exclude: set[int], n: int) -> list[int]:
    """Frame ber-lubang (n_hole ≥ 1) di luar `exclude`, n frame yang tersebar merata."""
    cand = sorted(r["index"] for r in records.values() if r.get("n_hole", 0) >= 1 and r["index"] not in exclude)
    if len(cand) <= n:
        return cand
    return [cand[round(i * (len(cand) - 1) / (n - 1))] for i in range(n)] if n > 1 else cand[:1]


def pct(vals: list[float], q: float) -> float:
    return float(np.percentile(vals, q)) if vals else 0.0


def summarize(clip: Path) -> dict:
    """Angka [4] dari contours/frames.jsonl (record frame terakhir per frame) + manifest."""
    cdir = clip / "contours"
    recs = {r["frame"]: r for r in read_jsonl(cdir / "frames.jsonl") if r.get("event") == "frame"}
    rs = [recs[k] for k in sorted(recs)]
    if not rs:
        raise StageError(f"{cdir / 'frames.jsonl'} tidak memuat record frame")
    t = [r["total_s"] for r in rs]

    def mmm(key: str) -> dict:
        v = [r[key] for r in rs]
        return {"min": min(v), "median": float(np.median(v)), "max": max(v)}

    dropped = sorted(a for r in rs for a in r["holes_dropped_areas"])
    json_bytes = sum(p.stat().st_size for p in cdir.glob("frame_*.json"))
    return {
        "clip": clip.name, "frames": len(rs), "contract": read_json(cdir / "manifest.json").get("contract"),
        "time_s": {"mean": float(np.mean(t)), "p95": pct(t, 95), "max": max(t), "over_1s": sum(x > 1 for x in t)},
        "json_bytes": {"total": json_bytes, "per_frame_mean": json_bytes / len(rs),
                       "per_frame_max": max(r["bytes"] for r in rs)},
        "strokes_per_frame": {"silhouette": mmm("n_silhouette"), "silhouette_hole": mmm("n_hole"),
                              "group_boundary": mmm("n_boundary")},
        "frames_without_silhouette": sum(r["n_silhouette"] == 0 for r in rs),
        "frames_with_holes": sum(r["n_hole"] >= 1 for r in rs),
        "edge_points": {"frames_touching": sum(r["edge_points"] > 0 for r in rs),
                        "mean": float(np.mean([r["edge_points"] for r in rs])),
                        "max": max(r["edge_points"] for r in rs)},
        "calibration_T305": {
            "holes_raw_total": sum(r["holes_raw"] for r in rs),
            "holes_kept_total": sum(r["n_hole"] for r in rs),
            "holes_dropped_by_area_total": len(dropped),
            "holes_dropped_by_parent_total": sum(r["holes_dropped_parent"] for r in rs),
            "holes_dropped_area_px": {"min": dropped[0] if dropped else None,
                                      "median": float(np.median(dropped)) if dropped else None,
                                      "p90": pct(dropped, 90) if dropped else None,
                                      "max": dropped[-1] if dropped else None},
            "regions_raw_total": sum(r["regions_raw"] for r in rs),
            "regions_dropped_total": sum(r["regions_dropped"] for r in rs),
            "group_boundary_loops_total": sum(r["loops"] for r in rs),
            "spurs_dropped_total": sum(r["spurs_dropped"] for r in rs),
            "spurs_dropped_by_length_px": {str(k): sum(r["spur_lengths"].count(k) for r in rs) for k in range(1, 6)},
            "short_fragments_dropped_total": sum(r["short_dropped"] for r in rs)},
    }


def write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--work-dir", type=Path, required=True, help="folder klip, mis. work/clips/test")
    ap.add_argument("--out-dir", type=Path, default=Path("work/t201a"))
    ap.add_argument("--ranges", default=DEFAULT_RANGES, help="indeks frame, mis. 73-92,183-202")
    ap.add_argument("--hole-frames", type=int, default=20, help="jumlah frame ber-lubang tambahan")
    ap.add_argument("--samples", default=DEFAULT_SAMPLES, help="indeks frame untuk PNG sampel")
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--crf", type=int, default=20)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--suffix", default="", help="akhiran nama keluaran (mis. _fix1) supaya tidak menimpa berkas lama")
    ap.add_argument("--dropped", action="store_true",
                    help="HANYA tulis overlay_<klip>_dropped.mp4 (kontur yang dibuang filter ukuran, abu-abu)")
    a = ap.parse_args(argv)
    clip = a.work_dir
    meta = read_json(clip / "meta.json")
    present = {int(p.stem.split("_")[1]) for p in (clip / "contours").glob("frame_*.json")}
    if not present:
        print(f"ERROR: {clip / 'contours'} kosong — jalankan `python -m rotoscope vectorize` dulu", file=sys.stderr)
        return 1
    a.out_dir.mkdir(parents=True, exist_ok=True)
    if not a.dropped:
        summary = summarize(clip)
        summary_path = a.out_dir / f"summary_{clip.name}{a.suffix}.json"
        write_atomic(summary_path, (json.dumps(summary, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
        print(f"ringkasan → {summary_path}")

    wanted = [i for i in parse_ranges(a.ranges) if i in present]
    skipped = [i for i in parse_ranges(a.ranges) if i not in present]
    if skipped:
        print(f"catatan: {len(skipped)} frame di --ranges tidak ada di klip ini (mis. {skipped[0]}) — dilewati")
    recs = {r["frame"]: r for r in read_jsonl(clip / "contours" / "frames.jsonl") if r.get("event") == "frame"}
    holes = hole_frames(recs, set(wanted), a.hole_frames)
    plan: dict[str, tuple[int, str]] = {}
    for i in wanted:
        plan[f"frame_{i:05d}"] = (i, "rentang")
    for i in holes:
        plan.setdefault(f"frame_{i:05d}", (i, "ber-lubang"))

    dropped = None
    if a.dropped:
        params = read_json(clip / "contours" / "manifest.json")["vectorize"]
        dropped = {k: params[k] for k in ("min_region_area", "min_hole_area")}
    else:
        for i in [x for x in parse_ranges(a.samples) if x in present] + holes[:1]:
            img = render(clip, i, a.scale, "sampel")
            ok, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            if ok:
                write_atomic(a.out_dir / f"sample_{clip.name}_f{i:05d}{a.suffix}.png", buf.tobytes())
    if a.no_video or not plan:
        return 0
    first = render(clip, next(iter(plan.values()))[0], a.scale, "x", dropped)
    size = (first.shape[1], first.shape[0])
    out = a.out_dir / f"overlay_{clip.name}{a.suffix}{'_dropped' if a.dropped else ''}.mp4"
    tmp = out.with_name(out.name + ".tmp")
    names = list(plan)
    ex.encode(OverlaySource(clip, plan, a.scale, dropped), names, size, np.array([255, 255, 255], np.uint8),
              float(meta.get("target_fps", 24)), a.crf, "medium", None, tmp)
    os.replace(tmp, out)
    print(f"video → {out} ({len(names)} frame, {out.stat().st_size / 2**20:.2f} MiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
