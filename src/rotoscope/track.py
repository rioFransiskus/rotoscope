"""Pelacakan strok antar frame (T-202, stage [4]): orientasi, anchor, arah garis terbuka, `track_id`.

Dipanggil `vectorize.run_vectorize` setelah strok mentah sebuah frame terbentuk dan terurut. Hanya URUTAN titik DALAM
strok yang berubah (pembalikan / rotasi); himpunan titik, jumlah titik, jumlah strok, dan urutan strok tidak. CPU
saja (numpy + scipy), tanpa torch. Kontrak lengkap: docs/01 [4] "Aturan anchor + orientasi + `track_id`".

Aturan (semua deterministik; seri selalu diselesaikan dengan aturan tetap, bukan urutan hash / memori):
  - Orientasi (luas bertanda, y ke bawah: luas > 0 = searah jarum jam di layar): `silhouette` dan loop
    (`group_boundary` / `occlusion` dengan titik akhir = titik awal) searah jarum jam, `silhouette_hole` berlawanan.
  - Anchor strok tertutup / loop = `points[0]` setelah rotasi. Track lama: titik terdekat ke `points[0]` padanan di frame
    sebelumnya (seri jarak → (y, x) terkecil). Track baru: `silhouette` = titik kontur terdekat ke piksel hair ∪ face
    tertinggi DI DALAM kontur itu (tidak ada → titik tertinggi kontur); selain itu titik tertinggi (y, x) terkecil.
  - Garis terbuka: aturan statis = ujung dengan proyeksi terkecil pada sumbu utama PCA (tanda sumbu: komponen dominan
    positif; seri → y terkecil, lalu x). Ada padanan → kesinambungan: ujung dipilih agar jarak titik awal + titik akhir ke
    titik awal + titik akhir padanan minimum (seri → aturan statis). Aturan statis saja membalik arah di sudut pemotongan
    sumbu (terukur: 7/35 dan 17/83 track ≥ 5 frame), jadi hanya dipakai untuk track baru.
  - `track_id`: Chamfer simetris rata-rata `(mean d(a→b) + mean d(b→a)) / 2` ke strok frame sebelumnya dengan `type` +
    `groups` sama, penugasan satu-ke-satu OPTIMAL (`linear_sum_assignment`), pasangan ≥ `max_dist` tidak dipakai.
    Tidak cocok → id baru berurutan (mulai 1, urutan strok tetap), unik per klip. Split / merge: satu padanan mendapat id
    lama, sisanya id baru.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

TYPE_SILHOUETTE = "silhouette"
TYPE_HOLE = "silhouette_hole"
FIRST_ID = 1                      # id mulai dari 1: tidak ada nilai falsy yang lolos di [5]
UNMATCHED_COST = 1e9              # biaya pasangan yang tidak boleh dipilih (di atas ambang / kunci beda)
COST_DECIMALS = 6                 # biaya dibulatkan sebelum penugasan (seri benar-benar seri)
TIE_EPS = 1e-9                    # pemecah seri: indeks (cur, prev) lebih kecil menang; < 10^-COST_DECIMALS
FILL_VALUE = 1                    # isi poligon kontur saat mencari piksel hair ∪ face di dalamnya


# ── Geometri dasar ─────────────────────────────────
def signed_area(pts: np.ndarray) -> float:
    """Luas bertanda (rumus shoelace) dengan y ke bawah: > 0 = searah jarum jam seperti terlihat di layar."""
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def is_loop(points) -> bool:
    """Garis terbuka yang titik akhirnya = titik awalnya (tidak punya 'ujung')."""
    return len(points) > 3 and points[0] == points[-1]


def top_index(pts: np.ndarray) -> int:
    """Titik tertinggi: y terkecil, seri → x terkecil."""
    return min(range(len(pts)), key=lambda i: (pts[i, 1], pts[i, 0]))


def nearest_index(pts: np.ndarray, ref: np.ndarray) -> int:
    """Titik terdekat ke `ref` (jarak L2); seri jarak → (y, x) terkecil (tidak bergantung urutan titik)."""
    d2 = ((pts - ref) ** 2).sum(axis=1)
    cand = np.flatnonzero(d2 == d2.min())
    return int(min(cand, key=lambda i: (pts[i, 1], pts[i, 0])))


def orient_cyclic(pts: np.ndarray, clockwise: bool) -> np.ndarray:
    area = signed_area(pts)
    return pts[::-1].copy() if area != 0 and (area > 0) != clockwise else pts


def principal_axis(pts: np.ndarray) -> np.ndarray:
    """Sumbu utama (PCA) satuan; tanda dikunci: komponen dominan (|x| ≥ |y| → x, selain itu y) positif."""
    c = pts - pts.mean(axis=0)
    ax = np.linalg.eigh(c.T @ c)[1][:, 1]
    k = 0 if abs(ax[0]) >= abs(ax[1]) else 1
    return ax if ax[k] > 0 else -ax


def static_open(pts: np.ndarray) -> np.ndarray:
    """Aturan statis: titik awal = ujung dengan proyeksi terkecil pada sumbu utama (seri → y terkecil, lalu x)."""
    proj = pts @ principal_axis(pts)
    a, b = float(proj[0]), float(proj[-1])
    if a == b:
        return pts if (pts[0, 1], pts[0, 0]) <= (pts[-1, 1], pts[-1, 0]) else pts[::-1].copy()
    return pts if a < b else pts[::-1].copy()


def orient_open(pts: np.ndarray, ref: tuple[np.ndarray, np.ndarray] | None) -> tuple[np.ndarray, bool]:
    """(titik, dibalik_dari_statis). `ref` = (titik awal, titik akhir) padanan frame sebelumnya atau None."""
    s = static_open(pts)
    if ref is None:
        return s, False
    r = s[::-1]
    cost_s = np.linalg.norm(s[0] - ref[0]) + np.linalg.norm(s[-1] - ref[1])
    cost_r = np.linalg.norm(r[0] - ref[0]) + np.linalg.norm(r[-1] - ref[1])
    return (r.copy(), True) if cost_r < cost_s else (s, False)


def silhouette_anchor(pts: np.ndarray, hair_face: np.ndarray | None) -> int:
    """Anchor track baru `silhouette`: titik kontur terdekat ke piksel hair ∪ face tertinggi di dalam kontur ini;
    tidak ada hair / face di dalamnya → titik tertinggi kontur."""
    if hair_face is None or not hair_face.any():
        return top_index(pts)
    mask = np.zeros(hair_face.shape, np.uint8)
    cv2.fillPoly(mask, [np.floor(pts).astype(np.int32).reshape(-1, 1, 2)], FILL_VALUE)
    ys, xs = np.nonzero(mask.astype(bool) & hair_face)         # urutan baris-utama: elemen pertama = (y, x) terkecil
    if ys.size == 0:
        return top_index(pts)
    return nearest_index(pts, np.array([xs[0] + 0.5, ys[0] + 0.5]))


# ── Pencocokan ─────────────────────────────────────
@dataclass
class Item:
    """Satu strok untuk pencocokan. `pts` = titik (urutan terkini), `tree` = KD-tree himpunan titik."""
    type: str
    key: tuple
    pts: np.ndarray
    cyclic: bool
    tree: cKDTree
    track_id: int | None = None
    loop: bool = field(default=False)


def make_item(stroke: dict, track_id: int | None = None) -> Item:
    pts = np.asarray(stroke["points"], dtype=np.float64)
    loop = (not stroke["closed"]) and is_loop(stroke["points"])
    return Item(stroke["type"], (stroke["type"], tuple(stroke["groups"])), pts, bool(stroke["closed"]) or loop,
                cKDTree(pts), track_id, loop)


def cost_matrix(cur: list[Item], prev: list[Item]) -> np.ndarray:
    """cost[i, j] = Chamfer simetris rata-rata; inf kalau kunci (type, groups) beda."""
    cost = np.full((len(cur), len(prev)), np.inf)
    for i, a in enumerate(cur):
        for j, b in enumerate(prev):
            if a.key == b.key:
                cost[i, j] = (b.tree.query(a.pts)[0].mean() + a.tree.query(b.pts)[0].mean()) / 2
    return cost


def assign(cost: np.ndarray, max_dist: float) -> dict[int, int]:
    """Penugasan satu-ke-satu optimal (jumlah biaya minimum) atas pasangan dengan biaya < max_dist. Seri: biaya
    dibulatkan, lalu indeks (cur, prev) lebih kecil menang."""
    n, m = cost.shape
    if n == 0 or m == 0:
        return {}
    ok = cost < max_dist
    c = np.where(ok, np.round(cost, COST_DECIMALS), UNMATCHED_COST)
    c = c + (np.arange(n)[:, None] * m + np.arange(m)[None, :]) * TIE_EPS
    rows, cols = linear_sum_assignment(c)
    return {int(i): int(j) for i, j in zip(rows, cols) if ok[i, j]}


def _largest_silhouette(items: list[Item]) -> int | None:
    idx = [i for i, it in enumerate(items) if it.type == TYPE_SILHOUETTE]
    if not idx:
        return None
    return max(idx, key=lambda i: (abs(signed_area(items[i].pts)), -i))


def inherit_main_silhouette(asg: dict[int, int], cur: list[Item], prev: list[Item]) -> dict[int, int]:
    """Pendekatan Y (belum dipakai produksi): silhouette terbesar (luas) frame ini mewarisi id silhouette terbesar
    frame sebelumnya, apa pun jaraknya; padanan lain yang melibatkan keduanya dilepas."""
    ci, pj = _largest_silhouette(cur), _largest_silhouette(prev)
    if ci is None or pj is None:
        return asg
    out = {i: j for i, j in asg.items() if i != ci and j != pj}
    out[ci] = pj
    return out


# ── Pelacak ────────────────────────────────────────
class Tracker:
    """Keadaan antar frame: strok frame sebelumnya (id, titik; `points[0]` = anchor / titik awal) + penghitung id."""

    def __init__(self, max_dist: float, *, inherit_main: bool = False):
        self.max_dist = float(max_dist)
        self.inherit_main = inherit_main
        self.next_id = FIRST_ID
        self.prev: list[Item] = []

    def load(self, strokes: list[dict]) -> None:
        """Keadaan = strok final sebuah frame (dari JSON yang valid). Penghitung id = max id yang pernah dilihat + 1."""
        self.prev = [make_item(s, int(s["track_id"])) for s in strokes]
        for it in self.prev:
            self.next_id = max(self.next_id, it.track_id + 1)

    def step(self, strokes: list[dict], hair_face: np.ndarray | None = None) -> dict:
        """Normalkan `strokes` (mentah, terurut) di tempat: orientasi, anchor, arah, `track_id`. Return statistik."""
        cur = [make_item(s) for s in strokes]
        asg = assign(cost_matrix(cur, self.prev), self.max_dist)
        if self.inherit_main:
            asg = inherit_main_silhouette(asg, cur, self.prev)
        n_new = n_flipped = 0
        for i, (s, it) in enumerate(zip(strokes, cur)):
            ref = self.prev[asg[i]] if i in asg else None
            if it.cyclic:
                base = orient_cyclic(it.pts[:-1] if it.loop else it.pts, clockwise=it.type != TYPE_HOLE)
                if ref is not None:
                    idx = nearest_index(base, ref.pts[0])
                elif it.type == TYPE_SILHOUETTE:
                    idx = silhouette_anchor(base, hair_face)
                else:
                    idx = top_index(base)
                pts = np.roll(base, -idx, axis=0)
                if it.loop:
                    pts = np.vstack([pts, pts[:1]])
            else:
                pts, flipped = orient_open(it.pts, (ref.pts[0], ref.pts[-1]) if ref is not None else None)
                n_flipped += flipped
            if ref is not None:
                tid = ref.track_id
            else:
                tid, n_new = self.next_id, n_new + 1
                self.next_id += 1
            it.pts, it.track_id = pts, tid
            strokes[i] = final_stroke(s, tid, pts)
        self.prev = cur
        return {"n_matched": len(strokes) - n_new, "n_new_ids": n_new, "n_flipped_vs_static": n_flipped}


def final_stroke(s: dict, track_id: int, pts: np.ndarray) -> dict:
    """Strok final dengan urutan key tetap: track_id, type, closed, groups, points, [anchor], [strength]."""
    out = {"track_id": track_id, "type": s["type"], "closed": s["closed"], "groups": s["groups"],
           "points": pts.tolist()}
    if s["closed"]:
        out["anchor"] = 0
    if "strength" in s:
        out["strength"] = s["strength"]
    return out
