"""Stage [4] vectorize (T-201a): stable/groups/*.png → contours/frame_*.json + contours/manifest.json
+ contours/frames.jsonl (D-010). Kontrak lengkap: docs/01 [4].

T-201a = siluet + lubang + batas grup SAJA. Garis oklusi + clip_stats.json = T-201b; anchor, orientasi,
`track_id` = T-202 (sengaja TIDAK ditulis: key tiap strok persis {type, closed, groups, points}, supaya [5]
gagal keras kalau memakai nilai yang belum ada). CPU saja, tanpa torch. Per frame:
  1. Siluet: foreground (grup ≠ 0) di-pad 1 px background (kontur tetap tertutup di tepi frame) →
     cv2.findContours RETR_CCOMP / CHAIN_APPROX_NONE. Kontur luar < min_region_area dibuang, lubang <
     min_hole_area dibuang. Ukuran = JUMLAH PIKSEL (komponen foreground 8-arah / komponen background
     4-arah yang terlingkup), bukan cv2.contourArea (bias berlawanan untuk kontur luar vs lubang).
  2. Batas grup per pasangan (a, b), a < b menurut urutan YAML, keduanya ≠ 0: band = piksel a bertetangga
     3×3 b ∪ piksel b bertetangga a → komponen 8-arah < line_min_px dibuang → thinning (Zhang-Suen) →
     tracing (junction = crossing number ≥ 3 dilepas, jalur disambung ke junction, jalur < min_stroke_px dibuang, dua jalur yang
     tersisa di satu junction digabung). Loop tertutup = closed:false dengan titik akhir = titik awal.

Koordinat = ruang kontinu: piksel (i, j) menempati [i, i+1) × [j, j+1), titik disimpan di PUSAT piksel
(i+0.5, j+0.5). Run titik di baris/kolom tepi frame (mis. y = H−0.5) = kontur yang menempel tepi; [5]
(T-203) memutuskan penanganannya.

Manifest berbeda (parameter T-201a / grup / [3] dihitung ulang / identitas klip) → contours/ BASI: dihapus
dan dihitung ulang otomatis dengan peringatan. Output tanpa manifest → ditolak. --restart = paksa.

CLI: python -m rotoscope vectorize <video> [--config PATH] [--restart] [--limit N]; cli.py memanggil main()
ini in-process dengan --work-dir <folder klip>. Exit code: 0 sukses, 1 prasyarat gagal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from rotoscope import stabilize as stb
from rotoscope.config import ConfigError, PipelineConfig, ensure_dir, load_pipeline, section_hash
from rotoscope.stage_common import (
    CLIP_KEY, EXIT_OK, EXIT_PRECONDITION, StageError, add_work_dir_arg, append_jsonl, clean_tmp, cli_cmd,
    clip_identity, describe_identity, load_frame_list, reconfigure_stdio, utc_now, work_dir_overrides,
    write_bytes_atomic, write_json_atomic,
)

# ── Layout output (docs/01 [4]) ────────────────────
CONTOURS_DIRNAME = "contours"
MANIFEST_FILENAME = "manifest.json"
FRAMES_LOG_FILENAME = "frames.jsonl"
FRAME_SUFFIX = ".json"
CONTRACT = "T-201a"          # T-201b / T-202 menaikkan ini → output lama otomatis basi
# Naik 1 setiap perubahan PERILAKU algoritma tanpa perubahan parameter (output lama otomatis basi, peringatan
# menyebut algo_rev lama → baru). 1 = junction ≥ 3 tetangga + Zhang-Suen; 2 = crossing number + prune sudut
# tangga + klaster junction + Guo-Hall.
ALGO_REV = 2
BACKGROUND_NAME = "background"

TYPE_SILHOUETTE = "silhouette"
TYPE_HOLE = "silhouette_hole"
TYPE_BOUNDARY = "group_boundary"
STROKE_TYPES = (TYPE_SILHOUETTE, TYPE_HOLE, TYPE_BOUNDARY)   # urutan = rank sortir
STROKE_KEYS = ("type", "closed", "groups", "points")
PENDING = ("occlusion", "anchor", "track_id", "orientation")
PARAM_KEYS = ("min_region_area", "min_hole_area", "line_min_px", "min_stroke_px")  # parameter T-201a

# ── Konstanta struktural (bukan parameter style; sama dengan RING_KERNEL di stabilize) ──
PAD_PX = 1                                   # padding background di sekeliling foreground / band
KERNEL_3X3 = np.ones((3, 3), np.uint8)       # tetangga 8-arah
PIXEL_CENTER = 0.5                           # titik = pusat piksel (i + 0.5)
COORD_DECIMALS = 1                           # pembulatan 0.1 px (no-op: semua titik = k + 0.5)
N4 = ((0, -1), (0, 1), (-1, 0), (1, 0))      # (dx, dy)
N8 = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))   # (dy, dx), 4-arah dulu
RING8 = ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))  # (dy, dx), melingkar
# Guo-Hall, bukan Zhang-Suen (look test T-102c): Zhang-Suen mengikis habis garis diagonal 45° yang berpadding
# (band 4 px → 2 piksel tersisa, diukur T-201a); Guo-Hall mempertahankan seluruh panjangnya.
THINNING_TYPE = cv2.ximgproc.THINNING_GUOHALL
JUNCTION_CROSSINGS = 3                      # junction = ≥ 3 transisi 0→1 di cincin 8 tetangga (crossing number)
COORDS_INFO = {"space": "continuous", "origin": "top-left corner of pixel (0, 0)",
               "point": "pixel center (i + 0.5, j + 0.5)", "decimals": COORD_DECIMALS}

# Kunci manifest yang harus sama untuk resume; beda → output basi (dihapus + dihitung ulang).
MANIFEST_MATCH_KEYS = ("contract", "algo_rev", "stroke_types", "vectorize", "vectorize_hash", "groups_hash", "stabilize_hash",
                       "seg_model", "stable_created_utc", "frame_size", CLIP_KEY)

DEFAULT_CONFIG = Path("configs") / "default.yaml"


# ── Parameter + hash ───────────────────────────────
def vectorize_params(cfg: PipelineConfig) -> dict:
    """Hanya parameter T-201a (bukan seluruh section `vectorize`): depth_lines / track milik T-201b / T-202."""
    return {k: getattr(cfg.vectorize, k) for k in PARAM_KEYS}


def params_hash(params: dict) -> str:
    """sha256 canonical JSON (format sama dengan config.section_hash)."""
    blob = json.dumps(params, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ── Siluet + lubang ────────────────────────────────
def _hole_label(padded: np.ndarray, blab: np.ndarray, cnt: np.ndarray) -> int:
    """Label (background 4-arah) lubang yang dilingkupi kontur lubang `cnt`; −1 kalau tidak ketemu.

    Piksel background tetangga-4 sebuah titik kontur yang berada DI DALAM poligon terisi kontur itu.
    Piksel background di luar poligon (sisi luar bagian foreground yang tipis) tidak ikut.
    """
    x, y, w, h = cv2.boundingRect(cnt)
    m = np.zeros((h, w), np.uint8)
    cv2.drawContours(m, [cnt], -1, 1, cv2.FILLED, offset=(-x, -y))
    for px, py in cnt.reshape(-1, 2).tolist():
        for dx, dy in N4:
            qx, qy = px + dx, py + dy
            if x <= qx < x + w and y <= qy < y + h and m[qy - y, qx - x] and padded[qy, qx] == 0:
                return int(blab[qy, qx])
    return -1


def silhouette_strokes(gmap: np.ndarray, min_region_area: int, min_hole_area: int) -> tuple[list[dict], dict]:
    """Peta grup → strok silhouette + silhouette_hole (tertutup; titik pertama TIDAK diulang) + statistik."""
    padded = cv2.copyMakeBorder((gmap != stb.BACKGROUND_ID).astype(np.uint8), PAD_PX, PAD_PX, PAD_PX, PAD_PX,
                                cv2.BORDER_CONSTANT, value=0)
    stats = {"regions_raw": 0, "regions_dropped": 0, "holes_raw": 0, "holes_dropped_areas": [],
             "holes_dropped_parent": 0}
    contours, hier = cv2.findContours(padded, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return [], stats
    parent = [int(h[3]) for h in hier[0]]
    _, lab, st, _ = cv2.connectedComponentsWithStats(padded, connectivity=8)
    _, blab, bst, _ = cv2.connectedComponentsWithStats(1 - padded, connectivity=4)
    keep_outer: dict[int, bool] = {}
    out: list[dict] = []
    for i, cnt in enumerate(contours):
        if parent[i] >= 0:
            continue
        stats["regions_raw"] += 1
        x, y = cnt[0, 0]
        area = int(st[lab[y, x], cv2.CC_STAT_AREA])
        keep_outer[i] = area >= min_region_area
        if keep_outer[i]:
            out.append(_closed_stroke(TYPE_SILHOUETTE, cnt))
        else:
            stats["regions_dropped"] += 1
    for i, cnt in enumerate(contours):
        if parent[i] < 0:
            continue
        stats["holes_raw"] += 1
        if not keep_outer[parent[i]]:
            stats["holes_dropped_parent"] += 1
            continue
        label = _hole_label(padded, blab, cnt)
        if label >= 0:
            area = int(bst[label, cv2.CC_STAT_AREA])
        else:  # tidak ada tetangga-4 di dalam poligon (tidak diharapkan): hitung dari poligon terisi
            x, y, w, h = cv2.boundingRect(cnt)
            m = np.zeros((h, w), np.uint8)
            cv2.drawContours(m, [cnt], -1, 1, cv2.FILLED, offset=(-x, -y))
            area = int((m.astype(bool) & (padded[y:y + h, x:x + w] == 0)).sum())
        if area >= min_hole_area:
            out.append(_closed_stroke(TYPE_HOLE, cnt))
        else:
            stats["holes_dropped_areas"].append(area)
    return out, stats


def _closed_stroke(kind: str, cnt: np.ndarray) -> dict:
    pts = [_point(x - PAD_PX, y - PAD_PX) for x, y in cnt.reshape(-1, 2).tolist()]
    return {"type": kind, "closed": True, "groups": [BACKGROUND_NAME], "points": pts}


def _point(x: int, y: int) -> list[float]:
    return [round(x + PIXEL_CENTER, COORD_DECIMALS), round(y + PIXEL_CENTER, COORD_DECIMALS)]


# ── Tracing skeleton ───────────────────────────────
def _removable_lut() -> np.ndarray:
    """LUT 256 kode tetangga (bit i = RING8[i] terisi) → apakah piksel itu 'sudut redundan' (simple point):
    ≥ 2 tetangga, crossing number ≤ 2 (bukan junction), ≥ 2 tetangga ortogonal (sudut siku; piksel ujung
    tangga bertetangga satu ortogonal + satu diagonal TIDAK dihapus, kalau tidak garis memendek berantai),
    tetangga membentuk SATU komponen 8-arah (menghapusnya tidak memutus garis), dan latar yang
    4-bersebelahan dengannya membentuk tepat SATU komponen 4-arah (menghapusnya tidak membuat lubang —
    pusat '+' tidak boleh dihapus)."""
    def near(a, b, metric):
        return metric(abs(RING8[a][0] - RING8[b][0]), abs(RING8[a][1] - RING8[b][1]))

    def components(items, metric):
        left, count = set(items), 0
        while left:
            stack = [left.pop()]
            count += 1
            while stack:
                a = stack.pop()
                for b in [b for b in left if near(a, b, metric) <= 1]:
                    left.discard(b)
                    stack.append(b)
        return count

    lut = np.zeros(256, bool)
    for code in range(256):
        fg = [i for i in range(8) if code >> i & 1]
        bg = [i for i in range(8) if not code >> i & 1]
        crossings = sum(not code >> i & 1 and code >> (i + 1) % 8 & 1 for i in range(8))
        orthogonal = sum(i % 2 == 0 for i in fg)
        if len(fg) < 2 or crossings >= JUNCTION_CROSSINGS or orthogonal < 2 or components(fg, max) != 1:
            continue    # junction (crossing number ≥ 3) tidak pernah dihapus, mis. pusat T
        # latar 4-bersebelahan dengan piksel = posisi cincin genap (N, E, S, W); komponen via latar cincin 4-arah
        bg4 = {i for i in bg if i % 2 == 0}
        comps_with_bg4 = 0
        left = set(bg)
        while left:
            comp, stack = set(), [left.pop()]
            comp.add(stack[0])
            while stack:
                a = stack.pop()
                for b in [b for b in left if near(a, b, lambda dy, dx: dy + dx) <= 1]:
                    left.discard(b)
                    comp.add(b)
                    stack.append(b)
            comps_with_bg4 += bool(comp & bg4)
        lut[code] = comps_with_bg4 == 1
    return lut


REMOVABLE = _removable_lut()


def _neighbor_code(p: np.ndarray, y: int, x: int) -> int:
    return sum(int(p[y + dy, x + dx]) << i for i, (dy, dx) in enumerate(RING8))


def prune_redundant(skel: np.ndarray) -> np.ndarray:
    """Hapus sudut tangga redundan dari skeleton (piksel dengan > 2 tetangga tetapi crossing number ≤ 2).

    Piksel yang dihapus selalu bertetangga dengan piksel yang tersisa (cakupan ≤ 1 px), topologi tidak berubah
    (simple point, lihat `_removable_lut`), dan sesudahnya tidak ada segitiga piksel: tiap piksel di luar
    percabangan sungguhan punya ≤ 2 tetangga → tracing tanpa titik ganda / cabang palsu. Urutan raster, diulang
    sampai stabil (deterministik).
    """
    p = np.pad(skel > 0, 1).astype(np.uint8)
    h, w = p.shape
    while True:
        codes = np.zeros((h, w), np.uint8)
        for i, (dy, dx) in enumerate(RING8):
            codes[1:-1, 1:-1] |= p[1 + dy:h - 1 + dy, 1 + dx:w - 1 + dx] << i
        changed = False
        for y, x in np.argwhere((p == 1) & REMOVABLE[codes]).tolist():
            if REMOVABLE[_neighbor_code(p, y, x)]:
                p[y, x] = 0
                changed = True
        if not changed:
            return p[1:-1, 1:-1] > 0


def _attach(junc: np.ndarray, jlab: np.ndarray, p: tuple[int, int]) -> int | None:
    """Label klaster junction yang bersebelahan dengan ujung jalur p (pertama menurut urutan N8), atau None."""
    h, w = junc.shape
    for dy, dx in N8:
        y, x = p[0] + dy, p[1] + dx
        if 0 <= y < h and 0 <= x < w and junc[y, x]:
            return int(jlab[y, x])
    return None


def _clusters(junc: np.ndarray, jlab: np.ndarray, jpix: np.ndarray) -> dict[int, tuple[set, tuple[int, int]]]:
    """label → (piksel klaster, titik representatif). Representatif = piksel crossing-number ≥ 3 yang paling
    sentral di klaster (jumlah jarak Chebyshev terkecil; seri → (y, x) terkecil): semua cabang yang bertemu di
    klaster berakhir di titik yang SAMA."""
    pix: dict[int, set] = defaultdict(set)
    for y, x in np.argwhere(junc).tolist():
        pix[int(jlab[y, x])].add((y, x))
    out = {}
    for label, ps in pix.items():
        crit = sorted(p for p in ps if jpix[p])
        rep = min(crit, key=lambda c: (sum(max(abs(c[0] - q[0]), abs(c[1] - q[1])) for q in ps), c))
        out[label] = (ps, rep)
    return out


def _bridge(cluster: tuple[set, tuple[int, int]], e: tuple[int, int]) -> list[tuple[int, int]]:
    """Piksel dari tetangga ujung jalur e sampai titik representatif klaster (termasuk), jalur terpendek di dalam
    klaster (BFS, urutan N8) → tiap titik bertetangga dengan sebelumnya, tanpa loncatan."""
    pix, rep = cluster
    if max(abs(e[0] - rep[0]), abs(e[1] - rep[1])) <= 1:
        return [rep]
    prev: dict = {e: None}
    queue = [e]
    for cur in queue:
        for dy, dx in N8:
            q = (cur[0] + dy, cur[1] + dx)
            if q in pix and q not in prev:
                prev[q] = cur
                if q == rep:
                    path = []
                    while q != e:
                        path.append(q)
                        q = prev[q]
                    return path[::-1]
                queue.append(q)
    return [rep]


def _dedupe(pts: list[tuple[int, int]], cluster_px: set) -> list[tuple[int, int]]:
    """Piksel yang muncul dua kali dalam satu strok (sambungan klaster yang saling tumpang, mis. A, X, rep, X, B).

    Hanya lingkaran yang SELURUHNYA berisi piksel klaster yang dipotong; badan jalur (cincin yang berawal dan
    berakhir di klaster yang sama) tidak boleh terpotong. Loop tertutup yang diawali / diakhiri tonjolan
    (rep, x, …, x, rep) dipangkas jadi loop (x, …, x). Penutup loop (titik akhir = titik awal) dipertahankan.
    """
    closed = len(pts) > 3 and pts[0] == pts[-1]
    body = list(pts[:-1] if closed else pts)
    while closed and len(body) > 3 and body[1] == body[-1]:
        body = body[1:-1]
    out: list[tuple[int, int]] = []
    seen: dict[tuple[int, int], int] = {}
    for p in body:
        if p in seen and all(q in cluster_px for q in out[seen[p] + 1:]):
            for q in out[seen[p] + 1:]:
                del seen[q]
            del out[seen[p] + 1:]
        elif p in seen:
            out.append(p)                    # lasso melewati klaster dua kali: dibiarkan (dilaporkan metrik integritas)
        else:
            seen[p] = len(out)
            out.append(p)
    return out + [out[0]] if closed else out


def _walk(pts: set[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    """Komponen sederhana (jalur / siklus; junction sudah dilepas) → daftar jalur (y, x), deterministik."""
    deg = {p: sum((p[0] + dy, p[1] + dx) in pts for dy, dx in N8) for p in pts}
    visited: set[tuple[int, int]] = set()
    paths = []
    while len(visited) < len(pts):
        left = sorted(p for p in pts if p not in visited)
        ends = [p for p in left if deg[p] <= 1]
        cur = ends[0] if ends else left[0]
        path = [cur]
        visited.add(cur)
        while True:
            nxt = next(((cur[0] + dy, cur[1] + dx) for dy, dx in N8
                        if (cur[0] + dy, cur[1] + dx) in pts and (cur[0] + dy, cur[1] + dx) not in visited), None)
            if nxt is None:
                break
            path.append(nxt)
            visited.add(nxt)
            cur = nxt
        paths.append(path)
    return paths


def _is_loop(pts: list[tuple[int, int]]) -> bool:
    return len(pts) > 3 and pts[0] == pts[-1]


def _normalize(pts: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Garis terbuka: awal = ujung dengan (y, x) terkecil. Loop: mulai dari titik (y, x) terkecil."""
    if _is_loop(pts):
        body = pts[:-1]
        i = body.index(min(body))
        body = body[i:] + body[:i]
        return body + [body[0]]
    return pts[::-1] if pts[0] > pts[-1] else pts


def junction_mask(skel: np.ndarray) -> np.ndarray:
    """Piksel skeleton dengan crossing number ≥ JUNCTION_CROSSINGS (percabangan sungguhan).

    Jumlah tetangga ≥ 3 TIDAK dipakai: pada skeleton Zhang-Suen berbentuk tangga 4-arah hampir tiap piksel punya
    3 tetangga (satu diagonal), sehingga ±73% piksel terhitung junction dan garis hancur (diukur T-201a).
    """
    sk = skel > 0
    h, w = sk.shape
    p = np.pad(sk, 1)
    ring = [p[1 + dy:1 + dy + h, 1 + dx:1 + dx + w] for dy, dx in RING8]
    crossings = sum((~ring[i] & ring[(i + 1) % len(ring)]).astype(np.uint8) for i in range(len(ring)))
    return sk & (crossings >= JUNCTION_CROSSINGS)


def trace_skeleton(sk: np.ndarray, min_stroke: int) -> tuple[list[list[tuple[int, int]]], dict]:
    """Skeleton 1 px → polyline (y, x) + {"loops", "spurs_dropped", "short_dropped"}.

    Langkah: (1) `prune_redundant` menghapus sudut tangga redundan (piksel > 2 tetangga tetapi crossing number
    ≤ 2) sebelum tracing; (2) junction = crossing number ≥ 3 (`junction_mask`), piksel junction + tetangganya
    (satu klaster) dilepas — cabang yang menempel diagonal pada junction (mis. cabang atas dan kiri sebuah T)
    masih 8-terhubung kalau hanya piksel junction-nya yang dibuang; (3) jalur dilacak (metode look test T-102c);
    (4) tiap ujung jalur di klaster disambung lewat piksel klaster terpendek ke titik representatif klaster, jadi
    cabang yang bertemu berbagi satu titik yang sama dan tidak ada loncatan. Jalur inti < min_stroke titik
    dibuang (spur kalau menyentuh junction, selain itu fragmen; panjang spur dicatat di `spur_lengths`). Beda
    dari look test: sesudah spur dibuang, klaster yang tersisa tepat dua ujung jalur → kedua jalur digabung
    (garis tunggal kontinu); jalur yang kedua ujungnya di klaster itu juga → loop (titik akhir = titik awal).
    Piksel ganda dalam satu strok dipotong (`_dedupe`).
    """
    s = prune_redundant(sk)
    jpix = junction_mask(s)
    junc = cv2.dilate(jpix.astype(np.uint8), KERNEL_3X3).astype(bool) & s
    rest = s & ~junc
    _, jlab = cv2.connectedComponents(junc.astype(np.uint8), connectivity=8)
    clusters = _clusters(junc, jlab, jpix)
    k, lab, st, _ = cv2.connectedComponentsWithStats(rest.astype(np.uint8), connectivity=8)
    stats = {"loops": 0, "spurs_dropped": 0, "short_dropped": 0, "spur_lengths": []}
    paths: dict[int, dict] = {}
    for i in range(1, k):
        x0, y0, bw, bh = st[i, :4]
        ys, xs = np.nonzero(lab[y0:y0 + bh, x0:x0 + bw] == i)
        pts = set(zip((ys + y0).tolist(), (xs + x0).tolist()))
        for core in _walk(pts):
            a0 = _attach(junc, jlab, core[0])
            a1 = _attach(junc, jlab, core[-1]) if len(core) > 1 else None
            if len(core) < min_stroke:
                if a0 or a1:
                    stats["spurs_dropped"] += 1
                    stats["spur_lengths"].append(len(core))    # 1–2 px = artefak tangga skeleton, bukan spur
                else:
                    stats["short_dropped"] += 1
                continue
            if a0 is None and a1 is None and len(core) > 3 and max(abs(core[0][0] - core[-1][0]),
                                                                    abs(core[0][1] - core[-1][1])) <= 1:
                paths[len(paths)] = {"pts": core + [core[0]], "cl": [None, None]}   # siklus tanpa junction
                continue
            paths[len(paths)] = {"pts": core, "cl": [a0, a1]}
    _merge_at_junctions(paths, clusters)
    polys = []
    cluster_px = set().union(*(c[0] for c in clusters.values())) if clusters else set()
    for _, p in sorted(paths.items()):
        pts = p["pts"]
        if p["cl"][1] is not None:                      # ujung yang tersisa di klaster → sambung ke titik representatif
            pts = pts + _bridge(clusters[p["cl"][1]], pts[-1])
        if p["cl"][0] is not None:
            pts = _bridge(clusters[p["cl"][0]], pts[0])[::-1] + pts
        polys.append(_normalize(_dedupe(pts, cluster_px)))
    stats["loops"] = sum(_is_loop(p) for p in polys)
    return polys, stats


def _merge_at_junctions(paths: dict[int, dict], clusters: dict) -> None:
    """Klaster junction dengan tepat dua ujung jalur tersisa → gabungkan lewat titik representatifnya (dalam
    tempat). Jalur yang kedua ujungnya di klaster itu → loop."""
    changed = True
    while changed:
        changed = False
        for cl in sorted({c for p in paths.values() for c in p["cl"] if c is not None}):
            refs = [(pid, side) for pid, p in sorted(paths.items()) for side in (0, 1) if p["cl"][side] == cl]
            if len(refs) != 2:
                continue
            (a, sa), (b, sb) = refs
            if a == b:
                p = paths[a]
                pts = p["pts"]
                p["pts"] = (pts + _bridge(clusters[cl], pts[-1]) + _bridge(clusters[cl], pts[0])[::-1][1:] + [pts[0]])
                p["cl"] = [None, None]
            else:
                pa, pb = paths[a], paths[b]
                first = pa["pts"] if sa == 1 else pa["pts"][::-1]
                second = pb["pts"] if sb == 0 else pb["pts"][::-1]
                paths[a] = {"pts": (first + _bridge(clusters[cl], first[-1])
                                    + _bridge(clusters[cl], second[0])[::-1][1:] + second),
                            "cl": [pa["cl"][1 - sa], pb["cl"][1 - sb]]}
                del paths[b]
            changed = True
            break


# ── Batas grup ─────────────────────────────────────
def thin_mask(mask: np.ndarray) -> np.ndarray:
    """Thinning (cv2.ximgproc, THINNING_TYPE) mask 0/1 berpadding → skeleton bool."""
    return cv2.ximgproc.thinning((mask > 0).astype(np.uint8) * 255, thinningType=THINNING_TYPE) > 0


def pair_skeletons(gmap: np.ndarray, line_min_px: int) -> list[tuple[int, int, np.ndarray, int, int]]:
    """Skeleton pasca-thinning per pasangan grup (a, b), a < b, keduanya ≠ 0: [(a, b, skel, ox, oy)].

    `skel` = array bool (crop band + PAD_PX); piksel global = (px + ox, py + oy). Band = piksel a bertetangga 3×3
    b ∪ piksel b bertetangga a, komponen 8-arah < line_min_px dibuang SEBELUM thinning (Zhang-Suen). Fungsi
    publik: dipakai group_boundary_strokes dan metrik cakupan di tests/.
    """
    ids = [int(g) for g in np.unique(gmap) if g != stb.BACKGROUND_ID]
    dil = {g: cv2.dilate((gmap == g).astype(np.uint8), KERNEL_3X3).astype(bool) for g in ids}
    out = []
    for ai, a in enumerate(ids):
        ma = gmap == a
        for b in ids[ai + 1:]:
            band = (ma & dil[b]) | ((gmap == b) & dil[a])
            if not band.any():
                continue
            x, y, w, h = cv2.boundingRect(band.astype(np.uint8))
            crop = cv2.copyMakeBorder(band[y:y + h, x:x + w].astype(np.uint8), PAD_PX, PAD_PX, PAD_PX, PAD_PX,
                                      cv2.BORDER_CONSTANT, value=0)
            n, lab, cst, _ = cv2.connectedComponentsWithStats(crop, connectivity=8)
            keep = np.zeros(n, bool)
            keep[1:] = cst[1:, cv2.CC_STAT_AREA] >= line_min_px
            crop = keep[lab].astype(np.uint8)
            if crop.any():
                out.append((a, b, thin_mask(crop), x - PAD_PX, y - PAD_PX))
    return out


def group_boundary_strokes(gmap: np.ndarray, names: tuple[str, ...], line_min_px: int,
                           min_stroke_px: int) -> tuple[list[dict], dict]:
    """Peta grup → strok group_boundary per pasangan (a, b), a < b (id grup = urutan YAML), + statistik."""
    stats = {"loops": 0, "spurs_dropped": 0, "short_dropped": 0, "spur_lengths": []}
    out: list[dict] = []
    for a, b, skel, ox, oy in pair_skeletons(gmap, line_min_px):
        polys, tstats = trace_skeleton(skel, min_stroke_px)
        for k in stats:
            stats[k] += tstats[k]
        for poly in polys:
            out.append({"type": TYPE_BOUNDARY, "closed": False, "groups": [names[a - 1], names[b - 1]],
                        "points": [_point(px + ox, py + oy) for py, px in poly], "_pair": (a, b)})
    return out, stats


# ── Satu frame ─────────────────────────────────────
def _sort_key(s: dict) -> tuple:
    """Urutan tetap: tipe, pasangan grup, titik (y, x) terkecil, jumlah titik, daftar titik (tie-break)."""
    pts = s["points"]
    ymin = min((p[1], p[0]) for p in pts)
    return (STROKE_TYPES.index(s["type"]), s.get("_pair", (0, 0)), ymin, len(pts), [tuple(p) for p in pts])


def vectorize_gmap(gmap: np.ndarray, names: tuple[str, ...], params: dict) -> tuple[list[dict], dict]:
    """Peta grup → (strok terurut, statistik). Key tiap strok persis STROKE_KEYS."""
    sil, sstats = silhouette_strokes(gmap, params["min_region_area"], params["min_hole_area"])
    bnd, bstats = group_boundary_strokes(gmap, names, params["line_min_px"], params["min_stroke_px"])
    strokes = sorted(sil + bnd, key=_sort_key)
    h, w = gmap.shape
    edge = sum(1 for s in strokes if s["type"] == TYPE_SILHOUETTE for x, y in s["points"]
               if x == PIXEL_CENTER or y == PIXEL_CENTER or x == w - PIXEL_CENTER or y == h - PIXEL_CENTER)
    for s in strokes:
        s.pop("_pair", None)
    stats = {"n_silhouette": sum(s["type"] == TYPE_SILHOUETTE for s in strokes),
             "n_hole": sum(s["type"] == TYPE_HOLE for s in strokes),
             "n_boundary": sum(s["type"] == TYPE_BOUNDARY for s in strokes),
             **sstats, **bstats, "edge_points": edge}
    return strokes, stats


def frame_document(index: int, width: int, height: int, source: dict, strokes: list[dict]) -> bytes:
    """JSON frame: kompak, urutan key tetap, tanpa timestamp → byte-identik antar run."""
    doc = {"frame_index": index, "width": width, "height": height, "source": source, "strokes": strokes}
    return (json.dumps(doc, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


# ── Layout klip ────────────────────────────────────
@dataclass(frozen=True)
class Clip:
    work_dir: Path
    names: tuple[str, ...]
    indices: tuple[int, ...]
    width: int
    height: int

    @property
    def stable_clip(self) -> stb.Clip:
        return stb.Clip(self.work_dir, self.names, self.indices, self.width, self.height)

    @property
    def contours_dir(self) -> Path:
        return self.work_dir / CONTOURS_DIRNAME

    @property
    def manifest_path(self) -> Path:
        return self.contours_dir / MANIFEST_FILENAME

    @property
    def frames_log(self) -> Path:
        return self.contours_dir / FRAMES_LOG_FILENAME

    def frame_path(self, name: str) -> Path:
        return self.contours_dir / (Path(name).stem + FRAME_SUFFIX)


def load_clip(work_dir: Path) -> Clip:
    return Clip(work_dir, *load_frame_list(work_dir))


def frame_valid(path: Path, index: int, clip: Clip, source: dict) -> bool:
    """JSON terbaca, frame_index / ukuran / source cocok, strok berupa daftar."""
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (isinstance(d, dict) and d.get("frame_index") == index and d.get("width") == clip.width
            and d.get("height") == clip.height and d.get("source") == source and isinstance(d.get("strokes"), list))


# ── Input [3] ──────────────────────────────────────
def load_stable(clip: Clip, cfg: PipelineConfig) -> dict:
    """stable/manifest.json, dicek terhadap meta.json (identitas klip, ukuran) dan grup config saat ini."""
    path = clip.stable_clip.manifest_path
    stb_cmd = cli_cmd("stabilize", clip.work_dir)
    if not path.is_file():
        raise StageError(f"{path} tidak ada — jalankan stage [3] dulu: {stb_cmd}")
    try:
        m = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise StageError(f"gagal membaca {path} ({e}) — jalankan ulang stage [3]: {stb_cmd}") from None
    current = clip_identity(clip.work_dir)
    old = m.get(CLIP_KEY)
    if not isinstance(old, dict) or not old.get("meta_sha256"):
        raise StageError(f"stable/manifest.json tidak memuat identitas klip (manifest lama) — jalankan ulang "
                         f"stage [3]: {stb_cmd} (dihitung ulang otomatis)")
    if old["meta_sha256"] != current["meta_sha256"]:
        raise StageError(f"stable/manifest.json milik klip LAIN: output [3] = {describe_identity(old)}, meta.json "
                         f"saat ini = {describe_identity(current)}. Jalankan ulang stage [3]: {stb_cmd}")
    size = {"width": clip.width, "height": clip.height}
    if m.get("frame_size") != size:
        raise StageError(f"stable/manifest.json: frame_size {m.get('frame_size')} ≠ meta.json {size} — jalankan "
                         f"ulang stage [3]: {stb_cmd}")
    for key in ("groups_hash", "stabilize_hash", "created_utc"):
        if not isinstance(m.get(key), str):
            raise StageError(f"stable/manifest.json tidak memuat {key} — jalankan ulang stage [3]: {stb_cmd}")
    if not isinstance(m.get("seg"), dict) or not m["seg"].get("model"):
        raise StageError(f"stable/manifest.json tidak memuat seg.model — jalankan ulang stage [3]: {stb_cmd}")
    if m["groups_hash"] != section_hash(cfg, "groups"):
        raise StageError(f"grup di config ≠ grup yang dipakai [3] (groups_hash stable {m['groups_hash'][:12]}, "
                         f"config {section_hash(cfg, 'groups')[:12]}) — nama grup di strok akan salah. Jalankan "
                         f"ulang stage [3]: {stb_cmd}")
    return m


def require_groups(clip: Clip, names) -> None:
    missing = [n for n in names if not clip.stable_clip.groups_path(n).is_file()]
    if missing:
        raise StageError(f"stable/groups belum lengkap: {len(missing)} dari {len(names)} frame hilang, mis. "
                         f"{Path(missing[0]).stem} — jalankan stage [3]: {cli_cmd('stabilize', clip.work_dir)}")


# ── Manifest ───────────────────────────────────────
def build_manifest(cfg: PipelineConfig, clip: Clip, stable: dict) -> dict:
    params = vectorize_params(cfg)
    return {"stage": "vectorize", "contract": CONTRACT, "algo_rev": ALGO_REV, "stroke_types": list(STROKE_TYPES),
            "pending": list(PENDING), "vectorize": params, "vectorize_hash": params_hash(params),
            "groups_hash": stable["groups_hash"], "stabilize_hash": stable["stabilize_hash"],
            "seg_model": stable["seg"]["model"], "stable_created_utc": stable["created_utc"],
            "frame_size": {"width": clip.width, "height": clip.height}, CLIP_KEY: clip_identity(clip.work_dir),
            "coords": dict(COORDS_INFO), "created_utc": utc_now()}


def frame_source(manifest: dict) -> dict:
    return {"seg_model": manifest["seg_model"], "groups_hash": manifest["groups_hash"],
            "stabilize_hash": manifest["stabilize_hash"], "vectorize_hash": manifest["vectorize_hash"]}


def _short(v) -> str:
    return v[:12] if isinstance(v, str) and len(v) > 12 else repr(v)


def manifest_diff(old: dict, new: dict) -> list[str]:
    """Field yang berubah; hash disingkat (lama → baru), dict per sub-field."""
    out = []
    for k in MANIFEST_MATCH_KEYS:
        a, b = old.get(k), new.get(k)
        if a == b:
            continue
        if isinstance(b, dict) and not isinstance(a, dict):
            a = {}
        if isinstance(a, dict) and isinstance(b, dict):
            out += [f"{k}.{s}: {_short(a.get(s))} → {_short(b.get(s))}"
                    for s in sorted(set(a) | set(b)) if a.get(s) != b.get(s)]
        else:
            out.append(f"{k}: {_short(a)} → {_short(b)}")
    return out


def _has_outputs(clip: Clip) -> bool:
    return any(clip.contours_dir.glob("frame_*" + FRAME_SUFFIX))


def restart_outputs(clip: Clip) -> None:
    """Hapus output [4] (contours/ saja)."""
    if clip.contours_dir.exists():
        shutil.rmtree(clip.contours_dir)


# ── Run ────────────────────────────────────────────
def run_vectorize(cfg: PipelineConfig, *, restart: bool = False, limit: int | None = None,
                  log: Callable[[str], None] = print) -> dict:
    """Jalankan stage [4] (T-201a). Return ringkasan run."""
    if limit is not None and limit < 1:
        raise StageError(f"--limit harus ≥ 1, dapat {limit}")
    clip = load_clip(cfg.paths.work_dir)
    selected = clip.names[:limit] if limit else clip.names
    stable = load_stable(clip, cfg)
    require_groups(clip, selected)
    names = tuple(g for g, _ in cfg.groups)
    params = vectorize_params(cfg)
    manifest = build_manifest(cfg, clip, stable)
    source = frame_source(manifest)

    stale = []
    if restart:
        restart_outputs(clip)
    elif clip.manifest_path.is_file():
        old = json.loads(clip.manifest_path.read_text(encoding="utf-8"))
        stale = manifest_diff(old, manifest)
        if stale:
            log("PERINGATAN: output [4] basi (setelan / input berubah) — contours/ dihapus dan dihitung ulang:\n  "
                + "\n  ".join(stale))
            restart_outputs(clip)
    elif _has_outputs(clip):
        raise StageError(f"{clip.contours_dir} berisi output tanpa {MANIFEST_FILENAME} — asal output tidak "
                         f"diketahui. Jalankan {cli_cmd('vectorize', clip.work_dir)} --restart.")
    clean_tmp(clip.contours_dir)

    todo = [(i, n) for i, n in zip(clip.indices[:len(selected)], selected)
            if not frame_valid(clip.frame_path(n), i, clip, source)]
    n_skip = len(selected) - len(todo)
    log(f"[4] vectorize ({CONTRACT}: silhouette + silhouette_hole + group_boundary): {len(selected)} frame dipilih, "
        f"{n_skip} valid dilewati, {len(todo)} diproses")

    run = {"selected": len(selected), "skipped": n_skip, "processed": 0, "stale": stale, "frames": []}
    if not todo:
        return run
    ensure_dir(clip.contours_dir)
    if not clip.manifest_path.is_file():
        write_json_atomic(clip.manifest_path, manifest)
    t_run = time.perf_counter()
    append_jsonl(clip.frames_log, {"event": "run_start", "time_utc": utc_now(), "n_todo": len(todo),
                                   "vectorize_hash": manifest["vectorize_hash"]})
    try:
        for k, (index, name) in enumerate(todo, 1):
            t0 = time.perf_counter()
            gmap = stb.read_groups(clip.stable_clip.groups_path(name))
            if (gmap is None or gmap.shape != (clip.height, clip.width) or int(gmap.max()) > len(names)):
                raise StageError(f"peta grup {Path(name).stem} rusak / ukuran salah — jalankan ulang stage [3]: "
                                 f"{cli_cmd('stabilize', clip.work_dir)}")
            t1 = time.perf_counter()
            strokes, stats = vectorize_gmap(gmap, names, params)
            t2 = time.perf_counter()
            data = frame_document(index, clip.width, clip.height, source, strokes)
            write_bytes_atomic(clip.frame_path(name), data)
            t3 = time.perf_counter()
            rec = {"event": "frame", "frame": name, "index": index, "time_utc": utc_now(),
                   "read_s": round(t1 - t0, 4), "vectorize_s": round(t2 - t1, 4), "write_s": round(t3 - t2, 4),
                   "total_s": round(t3 - t0, 4), "bytes": len(data), "points": sum(len(s["points"]) for s in strokes),
                   **stats}
            append_jsonl(clip.frames_log, rec)
            run["frames"].append(rec)
            run["processed"] += 1
            log(f"  [{k}/{len(todo)}] {name} {rec['total_s']:.3f} s, silhouette {stats['n_silhouette']}, lubang "
                f"{stats['n_hole']}, batas {stats['n_boundary']}, {len(data) / 1024:.1f} KiB")
    finally:
        append_jsonl(clip.frames_log, {"event": "run_end", "time_utc": utc_now(), "n_done": run["processed"],
                                       "wall_s": round(time.perf_counter() - t_run, 2)})
    return run


# ── Entry point stage (dipanggil cli.py) ───────────
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m rotoscope.vectorize",
                                description="Stage [4]: stable/groups → contours/ (siluet, lubang, batas grup)")
    p.add_argument("--config", type=Path, default=None,
                   help=f"YAML pipeline (default: {DEFAULT_CONFIG.as_posix()} kalau ada, selain itu default kode)")
    add_work_dir_arg(p)
    p.add_argument("--restart", action="store_true", help="hapus output [4] lama (contours/) lalu hitung ulang")
    p.add_argument("--limit", type=int, default=None, help="hanya N frame pertama")
    args = p.parse_args(argv)
    reconfigure_stdio()

    def log(msg: str) -> None:
        print(msg, flush=True)

    try:
        path = args.config if args.config is not None else (DEFAULT_CONFIG if DEFAULT_CONFIG.is_file() else None)
        cfg = load_pipeline(path, overrides=work_dir_overrides(args.work_dir))
        t0 = time.perf_counter()
        run = run_vectorize(cfg, restart=args.restart, limit=args.limit, log=log)
        log(f"selesai: {run['processed']} diproses, {run['skipped']} dilewati ({time.perf_counter() - t0:.1f} s)")
    except (StageError, ConfigError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return EXIT_PRECONDITION
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
