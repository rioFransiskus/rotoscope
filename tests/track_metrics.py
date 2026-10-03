"""Metrik objektif pelacakan strok (T-202), dipakai test_track.py / test_vectorize_track.py sebagai syarat lulus permanen.

Masukan semua fungsi: `frames` = daftar frame, tiap frame = daftar strok FINAL (dict dengan `track_id`, `type`, `closed`,
`groups`, `points`, ...), urut waktu. Fungsi murni (tanpa I/O) kecuali `load_frames`.

- orientasi: luas bertanda (y ke bawah; > 0 = searah jarum jam di layar) per strok tertutup;
- silhouette utama = silhouette berluas terbesar per frame; berapa `track_id` berbeda sepanjang klip;
- lompatan anchor: jarak `points[0]` antar frame berurutan (silhouette utama / per tipe);
- pembalikan arah: pasangan frame berurutan dengan id sama pada garis terbuka (bukan loop) yang titik awal / akhirnya
  tertukar; track hidup ≥ N frame;
- churn: id baru per frame per tipe, umur track; id ganda dalam satu frame;
- ambiguitas (rasio jarak kandidat terbaik kedua / terbaik < 1,5) dan ketidakcocokan ukuran (panjang strok > 3×);
- hash kanonik per tipe (regresi: urutan titik DALAM strok boleh berubah, himpunan titik dan urutan strok tidak).
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from rotoscope import track as trk

AMBIGUITY_RATIO = 1.5            # rasio kandidat terbaik kedua / terbaik di bawah ini = ambigu
SIZE_MISMATCH_RATIO = 3.0        # panjang dua strok yang dipadankan berselisih lebih dari ini = tidak cocok
MIN_LIFE_FRAMES = 5              # kriteria pembalikan arah hanya untuk track yang hidup ≥ ini
TYPES = ("silhouette", "silhouette_hole", "group_boundary", "occlusion")


def load_frames(contours_dir: Path) -> list[list[dict]]:
    return [json.loads(p.read_text(encoding="utf-8"))["strokes"] for p in sorted(contours_dir.glob("frame_*.json"))]


def arr(stroke: dict) -> np.ndarray:
    return np.asarray(stroke["points"], dtype=np.float64)


def percentiles(values, ps=(50, 95, 100)) -> dict:
    v = np.asarray(list(values), dtype=np.float64)
    return {f"p{p}": (round(float(np.percentile(v, p)), 3) if v.size else None) for p in ps} | {"n": int(v.size)}


def is_loop(stroke: dict) -> bool:
    return (not stroke["closed"]) and trk.is_loop(stroke["points"])


def length(stroke: dict) -> float:
    p = arr(stroke)
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum() + (np.linalg.norm(p[0] - p[-1]) if stroke["closed"] else 0))


# ── Orientasi ──────────────────────────────────────
def orientation(frames) -> dict:
    """Persentase silhouette searah jarum jam dan lubang berlawanan (loop terbuka dihitung terpisah: harus searah)."""
    cw = {"silhouette": [], "silhouette_hole": [], "loop": []}
    for fr in frames:
        for s in fr:
            if s["type"] in ("silhouette", "silhouette_hole"):
                cw[s["type"]].append(trk.signed_area(arr(s)) > 0)
            elif is_loop(s):
                cw["loop"].append(trk.signed_area(arr(s)[:-1]) > 0)
    pct = lambda v, want: (100.0 * sum(x == want for x in v) / len(v)) if v else 100.0  # noqa: E731
    return {"silhouette_cw_pct": pct(cw["silhouette"], True), "hole_ccw_pct": pct(cw["silhouette_hole"], False),
            "loop_cw_pct": pct(cw["loop"], True), "n": {k: len(v) for k, v in cw.items()}}


# ── Silhouette utama, anchor ───────────────────────
def main_silhouettes(frames) -> list[dict | None]:
    out = []
    for fr in frames:
        sil = [s for s in fr if s["type"] == "silhouette"]
        out.append(max(sil, key=lambda s: abs(trk.signed_area(arr(s)))) if sil else None)
    return out


def main_silhouette_ids(frames) -> set[int]:
    return {s["track_id"] for s in main_silhouettes(frames) if s is not None}


def main_anchor_jumps(frames) -> list[float]:
    """Jarak points[0] silhouette utama antar frame berurutan."""
    mains = main_silhouettes(frames)
    return [float(np.linalg.norm(np.asarray(b["points"][0]) - np.asarray(a["points"][0])))
            for a, b in zip(mains, mains[1:]) if a is not None and b is not None]


def track_anchor_jumps(frames, kinds=("silhouette_hole",)) -> list[float]:
    """Jarak points[0] antar frame berurutan untuk track yang sama (tipe `kinds`)."""
    out = []
    for a, b in zip(frames, frames[1:]):
        prev = {s["track_id"]: s for s in a if s["type"] in kinds}
        out += [float(np.linalg.norm(np.asarray(s["points"][0]) - np.asarray(prev[s["track_id"]]["points"][0])))
                for s in b if s["type"] in kinds and s["track_id"] in prev]
    return out


# ── Arah garis terbuka ─────────────────────────────
def reversed_pair(a: np.ndarray, b: np.ndarray) -> bool:
    """b (frame k) berarah terbalik terhadap a (frame k-1)?"""
    same = np.linalg.norm(b[0] - a[0]) + np.linalg.norm(b[-1] - a[-1])
    flip = np.linalg.norm(b[0] - a[-1]) + np.linalg.norm(b[-1] - a[0])
    return bool(flip < same)


def life_frames(frames) -> dict[int, int]:
    return dict(Counter(s["track_id"] for fr in frames for s in fr))


def reversals(frames, min_life: int = MIN_LIFE_FRAMES) -> dict:
    """Pembalikan arah garis terbuka (bukan loop): jumlah pasangan total, dan track hidup ≥ min_life yang pernah terbalik."""
    life = life_frames(frames)
    pairs, bad = 0, set()
    for a, b in zip(frames, frames[1:]):
        prev = {s["track_id"]: s for s in a if not s["closed"] and not is_loop(s)}
        for s in b:
            if s["closed"] or is_loop(s) or s["track_id"] not in prev:
                continue
            if reversed_pair(arr(prev[s["track_id"]]), arr(s)):
                pairs += 1
                bad.add(s["track_id"])
    long_bad = sorted(t for t in bad if life[t] >= min_life)
    return {"reversed_pairs": pairs, "tracks_reversed": len(bad), "long_tracks_reversed": long_bad}


# ── Churn, id ganda ────────────────────────────────
def churn(frames) -> dict:
    """Id baru per frame (setelah frame 0) per tipe + umur track (frame) per tipe."""
    seen: set[int] = set()
    new = defaultdict(list)
    ptype: dict[int, str] = {}
    for k, fr in enumerate(frames):
        cur_new = Counter(s["track_id"] not in seen and s["type"] for s in fr)
        if k > 0:
            for t in TYPES:
                new[t].append(cur_new.get(t, 0))
        for s in fr:
            seen.add(s["track_id"])
            ptype[s["track_id"]] = s["type"]
    life = life_frames(frames)
    by_type = defaultdict(list)
    for tid, n in life.items():
        by_type[ptype[tid]].append(n)
    return {"new_per_frame": {t: (round(float(np.mean(new[t])), 3) if new[t] else 0.0) for t in TYPES},
            "new_total": {t: int(sum(new[t])) for t in TYPES},
            "life": {t: percentiles(by_type[t], (50, 90, 100)) for t in TYPES}}


def duplicate_ids(frames) -> int:
    """Jumlah frame yang memuat track_id sama lebih dari sekali."""
    return sum(len({s["track_id"] for s in fr}) != len(fr) for fr in frames)


# ── Ambiguitas + ketidakcocokan ukuran ─────────────
def match_quality(frames) -> dict:
    """Untuk tiap strok yang id-nya ada di frame sebelumnya (padanan): rasio jarak (Chamfer) kandidat terbaik kedua /
    terbaik di antara strok sekunci frame sebelumnya; persentase rasio < AMBIGUITY_RATIO (satu kandidat = tidak
    ambigu). Ketidakcocokan ukuran: panjang kedua strok berselisih > SIZE_MISMATCH_RATIO."""
    ambig, total, mismatch = Counter(), Counter(), Counter()
    for a, b in zip(frames, frames[1:]):
        prev_items = [trk.make_item(s, s["track_id"]) for s in a]
        cur_items = [trk.make_item(s, s["track_id"]) for s in b]
        cost = trk.cost_matrix(cur_items, prev_items)
        pid = {it.track_id: j for j, it in enumerate(prev_items)}
        for i, it in enumerate(cur_items):
            if it.track_id not in pid:
                continue
            total[it.type] += 1
            row = np.sort(cost[i][np.isfinite(cost[i])])
            if len(row) >= 2 and (row[0] == 0 and row[1] == 0 or (row[0] > 0 and row[1] / row[0] < AMBIGUITY_RATIO)):
                ambig[it.type] += 1
            la, lb = length(a[pid[it.track_id]]), length(b[i])
            if min(la, lb) == 0 or max(la, lb) / min(la, lb) > SIZE_MISMATCH_RATIO:
                mismatch[it.type] += 1
    return {"matches": {t: total[t] for t in TYPES},
            "ambiguous_pct": {t: (round(100.0 * ambig[t] / total[t], 2) if total[t] else 0.0) for t in TYPES},
            "size_mismatch": {t: mismatch[t] for t in TYPES}}


# ── Kanonik (regresi) ──────────────────────────────
def canonical_points(stroke: dict) -> tuple:
    """Titik dalam bentuk kanonik: tertutup / loop = rotasi minimum dari kedua arah (dari titik leksikografis terkecil);
    garis terbuka = urutan atau kebalikannya, mana yang lebih kecil secara leksikografis."""
    pts = [tuple(p) for p in stroke["points"]]
    if stroke["closed"] or is_loop(stroke):
        ring = pts if stroke["closed"] else pts[:-1]
        i = min(range(len(ring)), key=lambda k: ring[k])
        fwd = ring[i:] + ring[:i]
        rev = ring[::-1]
        j = rev.index(ring[i])
        return min(fwd, rev[j:] + rev[:j])
    return min(pts, pts[::-1])


def canonical_stroke(stroke: dict) -> list:
    return [stroke["type"], stroke["closed"], stroke["groups"], canonical_points(stroke), stroke.get("strength")]


def canonical_hashes(frames) -> dict[str, str]:
    """sha256 per tipe atas bentuk kanonik semua strok semua frame (urutan strok per frame TIDAK dikanonikkan)."""
    hs = {t: hashlib.sha256() for t in TYPES}
    for fr in frames:
        for s in fr:
            hs[s["type"]].update(json.dumps(canonical_stroke(s), separators=(",", ":")).encode())
        for t in TYPES:
            hs[t].update(b"\n#frame\n")
    return {t: h.hexdigest() for t, h in hs.items()}
