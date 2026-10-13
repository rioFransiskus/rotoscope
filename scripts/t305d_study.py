"""T-305d Tahap 1 (CPU, baca-saja terhadap klip): QC `big_blobs` sadar multi-orang — ukur kandidat D / B / C.

    python scripts/t305d_study.py hash before|after   # sha256 frames/seg/depth/stable/contours/strokes + qc_report.json 4 klip + out/ -> work/t305d/
    python scripts/t305d_study.py hash before_t23|after_t23   # idem untuk Tahap 2–3 (arsip Tahap 1 tidak ditimpa)
    python scripts/t305d_study.py qc                  # 5.2: QC ulang 4 klip x kandidat dari seg/classmap -> work/t305d_scratch/qc.json
    python scripts/t305d_study.py synth               # 5.3: uji sintetis (test, klip3) -> work/t305d_scratch/synth.json
    python scripts/t305d_study.py cost                # 5.4: waktu QC per frame per kandidat
    python scripts/t305d_study.py stale               # 5.1: bukti rantai basi [3] pada salinan test_short di scratch

Hanya membaca work/clips/*; keluaran ke work/t305d/ dan work/t305d_scratch/ (di-ignore git). Deterministik: SEED tetap.

Kandidat (n_t = jumlah komponen foreground 8-arah dengan luas > qc.blob_min x luas frame):
    base = aturan sekarang (n_t > max_big_blobs)   D = big_blobs hanya dilaporkan
    B25 / B49 = n_t bukan modus n dalam jendela digeser W
    C3 / C4 / C6 = n_t naik sesaat (run pendek < N frame yang lebih tinggi dari run panjang terdekat)
Gagal akhir kandidat = (metrik QC lain gagal) ATAU (aturan big_blobs kandidat gagal).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rotoscope import segment as seg  # noqa: E402
from rotoscope.config import load_pipeline  # noqa: E402
from rotoscope.stage_common import last_frame_records  # noqa: E402

CLIPS_DIR = ROOT / "work" / "clips"
OUT = ROOT / "work" / "t305d"
SCRATCH = ROOT / "work" / "t305d_scratch"
HASH_CLIPS = ("test", "test_short", "Klip2", "klip3")
HASH_SUBDIRS = ("frames", "seg", "depth", "stable", "contours", "strokes")
SYNTH_CLIPS = ("test", "klip3")

SEED = 305401                       # seed tetap uji sintetis (sub-seed: klip, jenis, durasi)
DURATIONS = (1, 2, 3)
N_POSITIONS = 5
POS_SPACING = 10                    # jarak minimum antar posisi awal (frame)
CUT_FRAMES = (49, 85, 126, 134, 146, 154, 166, 174)   # klip3
CUT_MARGIN = 3
EDGE_MARGIN = 3
FAKE_MIN_DIST_PX = 10               # blob palsu >= 10 px dari foreground mana pun
FAKE_TRIES = 1500
FAKE_S_SCALES = (0.8, 0.7, 0.6, 0.5)   # TAMBAHAN (kind "fake_s"), di luar spesifikasi 5.3
FAKE_S_TRIES = 600
PRIMARY_KINDS = ("fake", "split")
ALL_KINDS = PRIMARY_KINDS + ("fake_s",)
SPLIT_LINE_PX = 6
CANDIDATES = ("base", "D", "B25", "B49", "C3", "C4", "C6")


# ── hash ───────────────────────────────────────────
def hash_files() -> list[Path]:
    files: list[Path] = []
    for c in HASH_CLIPS:
        for d in HASH_SUBDIRS:
            files += [p for p in sorted((CLIPS_DIR / c / d).rglob("*")) if p.is_file()]
        files.append(CLIPS_DIR / c / "qc_report.json")  # subjek task ini: ikut di-hash
    files += [p for p in sorted((ROOT / "out").rglob("*")) if p.is_file()]
    return files


def file_sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cmd_hash(which: str) -> None:
    lines = [f"{file_sha(p)}  {p.relative_to(ROOT).as_posix()}" for p in hash_files()]
    path = OUT / f"{which}_sha256.txt"
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{path.relative_to(ROOT).as_posix()}: {len(lines)} berkas")
    if which.startswith("after"):
        before = (OUT / f"{which.replace('after', 'before', 1)}_sha256.txt").read_text(encoding="utf-8").splitlines()
        same = before == lines
        diff = len(set(before) ^ set(lines))
        print(f"identik dengan before: {same} (before {len(before)} baris, after {len(lines)}, selisih himpunan {diff})")


# ── aturan kandidat (fungsi murni atas deret n) ────
def rule_base(n: list[int], max_big: int) -> list[bool]:
    return [x > max_big for x in n]


def rule_d(n: list[int]) -> list[bool]:
    return [False] * len(n)


def rule_b(n: list[int], window: int) -> list[bool]:
    """Gagal bila n_t bukan salah satu modus n di jendela digeser W (seri → lolos bila sama dengan salah satu modus)."""
    out = []
    for i in range(len(n)):
        s, e = seg.shifted_window(i, len(n), window)
        cnt = Counter(n[s:e])
        top = max(cnt.values())
        out.append(cnt[n[i]] != top)
    return out


def split_runs(n: list[int]) -> list[tuple[int, int, int]]:
    """[(start, end_exclusive, nilai)] — deretan frame berurutan dengan n sama."""
    runs, s = [], 0
    for i in range(1, len(n) + 1):
        if i == len(n) or n[i] != n[s]:
            runs.append((s, i, n[s]))
            s = i
    return runs


def rule_c(n: list[int], min_len: int, info: dict | None = None) -> list[bool]:
    """Frame di run pendek (< N) gagal bila n_t > max(run panjang terdekat sebelum, sesudah); tanpa run panjang → lolos."""
    runs = split_runs(n)
    long_idx = [k for k, (s, e, _) in enumerate(runs) if e - s >= min_len]
    out = [False] * len(n)
    if not long_idx and info is not None:
        info["no_long_run"] = True
    for k, (s, e, v) in enumerate(runs):
        if e - s >= min_len or not long_idx:
            continue
        before = [runs[j][2] for j in long_idx if j < k]
        after = [runs[j][2] for j in long_idx if j > k]
        ref = []
        if before:
            ref.append(before[-1])
        if after:
            ref.append(after[0])
        if v > max(ref):
            for t in range(s, e):
                out[t] = True
    return out


def big_rule(name: str, n: list[int], max_big: int, info: dict | None = None) -> list[bool]:
    if name == "base":
        return rule_base(n, max_big)
    if name == "D":
        return rule_d(n)
    if name[0] == "B":
        return rule_b(n, int(name[1:]))
    if name[0] == "C":
        return rule_c(n, int(name[1:]), info)
    raise ValueError(name)


# ── data klip + metrik ─────────────────────────────
class ClipQC:
    """Classmap + metrik QC per frame (sama dengan seg.compute_qc) + deret n."""

    def __init__(self, name: str, cfg) -> None:
        self.name = name
        self.q = cfg.qc
        self.dir = CLIPS_DIR / name
        self.report = json.loads((self.dir / "qc_report.json").read_text(encoding="utf-8"))
        rows = self.report["frames"]
        self.names = [r["frame"] for r in rows]
        self.indices = [r["index"] for r in rows]
        self.finite = [r["finite"] for r in rows]
        self.cms = [seg.read_classmap(self.dir / "seg" / "classmap" / (Path(n).stem + ".png")) for n in self.names]
        self.h, self.w = self.cms[0].shape
        self.size = self.h * self.w
        frame_log = last_frame_records(self.dir / "seg" / "frames.jsonl")
        self.rows = seg.compute_qc(self.cms, self.q, self.names, self.indices, frame_log)
        self.n = [r["big_blobs"] for r in self.rows]
        self.area = [r["area_px"] for r in self.rows]
        self.iou_prev = [r["iou_prev"] for r in self.rows]
        # reproduksi baseline: fail_reasons dan metrik harus sama dengan qc_report.json
        self.baseline_ok = all(a["fail_reasons"] == b["fail_reasons"] and a["big_blobs"] == b["big_blobs"]
                               and a["area_px"] == b["area_px"] and a["iou_prev"] == b["iou_prev"]
                               and a["area_vs_median"] == b["area_vs_median"]
                               for a, b in zip(self.rows, rows))

    def other_fail(self, area: list[int], iou_prev: list[float | None]) -> list[bool]:
        """Metrik QC selain big_blobs (area_ratio, iou_prev, area_vs_median, finite), dari deret area / iou."""
        q = self.q
        out = []
        avm = seg.area_vs_median(area, q.area_median_window)
        for i in range(len(area)):
            ratio = area[i] / self.size
            bad = ratio < q.area_min or ratio > q.area_max
            bad |= iou_prev[i] is not None and iou_prev[i] < q.iou_min
            bad |= avm[i][1] is not None and avm[i][1] < q.area_drop_min
            bad |= self.finite[i] is False
            out.append(bool(bad))
        return out

    def fails(self, cand: str, n=None, area=None, iou_prev=None, info=None) -> tuple[list[bool], list[bool], list[bool]]:
        """(gagal akhir, gagal big_blobs-kandidat, gagal metrik lain)."""
        n = self.n if n is None else n
        other = self.other_fail(self.area if area is None else area, self.iou_prev if iou_prev is None else iou_prev)
        big = big_rule(cand, n, self.q.max_big_blobs, info)
        return [a or b for a, b in zip(other, big)], big, other


def runs_of(flags: list[bool], idx: list[int]) -> list[tuple[int, int]]:
    """Rentang frame (indeks klip) berurutan yang True."""
    out, s = [], None
    for i, f in enumerate(flags):
        if f and s is None:
            s = i
        if (not f or i == len(flags) - 1) and s is not None:
            e = i if f else i - 1
            out.append((idx[s], idx[e]))
            s = None
    return out


def fmt_runs(r: list[tuple[int, int]]) -> str:
    return ", ".join(f"{a}" if a == b else f"{a}–{b}" for a, b in r) or "-"


def rle(vals: list[int], idx: list[int]) -> str:
    return ", ".join(f"f{idx[s]}–{idx[e - 1]}: n={v}" for s, e, v in split_runs(vals))


# ── 5.2 QC ulang ───────────────────────────────────
def cmd_qc() -> None:
    cfg = load_pipeline(ROOT / "configs" / "default.yaml")
    SCRATCH.mkdir(parents=True, exist_ok=True)
    result: dict = {}
    for clip in HASH_CLIPS:
        c = ClipQC(clip, cfg)
        print(f"\n=== {clip}: {len(c.names)} frame, baseline direproduksi dari classmap = {c.baseline_ok} "
              f"(qc_report: {c.report['summary']['n_fail']} gagal) ===")
        print(f"distribusi n_t: {dict(sorted(Counter(c.n).items()))}")
        if clip == "klip3":
            print("n_t per run:", rle(c.n, c.indices))
        base_fail, _, _ = c.fails("base")
        res = {"n_frames": len(c.names), "baseline_ok": c.baseline_ok, "n_dist": dict(Counter(c.n)), "cands": {}}
        for cand in CANDIDATES:
            info: dict = {}
            fin, big, oth = c.fails(cand, info=info)
            changed = sum(a != b for a, b in zip(fin, base_fail))
            ent = {"n_fail": sum(fin), "fail_runs": runs_of(fin, c.indices), "n_big_fail": sum(big),
                   "n_other_fail": sum(oth), "q_changed_vs_base": changed, "no_long_run": bool(info.get("no_long_run"))}
            res["cands"][cand] = ent
            print(f"  {cand:5s} gagal {ent['n_fail']:3d}/{len(c.names)} (big {ent['n_big_fail']}, lain {ent['n_other_fail']}), "
                  f"q berubah vs base: {changed:3d}"
                  + ("  [TANPA run panjang]" if ent["no_long_run"] else "")
                  + f"  run gagal: {fmt_runs(ent['fail_runs']) if ent['n_fail'] <= 60 else '(banyak) ' + fmt_runs(ent['fail_runs'])[:200]}")
            if clip == "klip3" and cand not in ("base", "D") and ent["n_fail"]:
                ctx = []
                for a, b in ent["fail_runs"]:
                    lo, hi = max(a - 2, 0), min(b + 2, len(c.n) - 1)
                    ctx.append(f"f{a}–{b}: n[{lo}..{hi}]=" + ",".join(str(x) for x in c.n[lo:hi + 1]))
                print("        konteks:", " | ".join(ctx))
        result[clip] = res
    (SCRATCH / "qc.json").write_text(json.dumps(result, indent=1), encoding="utf-8")


# ── 5.3 sintetis ───────────────────────────────────
def largest_blob(fg: np.ndarray) -> tuple[np.ndarray, int] | None:
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg.astype(np.uint8), connectivity=8)
    if n < 2:
        return None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return lab == k, int(stats[k, cv2.CC_STAT_AREA])


def make_fake(cm: np.ndarray, blob: np.ndarray, dx: int, dy: int) -> np.ndarray | None:
    """Salinan blob (label kelas dipertahankan) digeser (dx, dy); None bila keluar frame / < 10 px dari foreground."""
    h, w = cm.shape
    ys, xs = np.nonzero(blob)
    ny, nx = ys + dy, xs + dx
    if ny.min() < 0 or nx.min() < 0 or ny.max() >= h or nx.max() >= w:
        return None
    dist = cv2.distanceTransform((cm == 0).astype(np.uint8), cv2.DIST_L2, 5)
    if (dist[ny, nx] < FAKE_MIN_DIST_PX).any():
        return None
    out = cm.copy()
    out[ny, nx] = cm[ys, xs]
    return out


def make_fake_scaled(cm: np.ndarray, blob: np.ndarray, dx: int, dy: int, scale: float) -> np.ndarray | None:
    """TAMBAHAN (di luar spesifikasi 5.3): seperti make_fake, tetapi salinan diperkecil `scale` (tetangga terdekat) di sudut
    kiri-atas bbox + (dx, dy). Dipakai hanya karena blob penuh tidak muat di klip3."""
    h, w = cm.shape
    ys, xs = np.nonzero(blob)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    m = cv2.resize(blob[y0:y1, x0:x1].astype(np.uint8), None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    lab = cv2.resize(np.where(blob, cm, 0)[y0:y1, x0:x1], None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    sy, sx = np.nonzero(m)
    ny, nx = sy + y0 + dy, sx + x0 + dx
    if ny.min() < 0 or nx.min() < 0 or ny.max() >= h or nx.max() >= w:
        return None
    dist = cv2.distanceTransform((cm == 0).astype(np.uint8), cv2.DIST_L2, 5)
    if (dist[ny, nx] < FAKE_MIN_DIST_PX).any():
        return None
    out = cm.copy()
    out[ny, nx] = lab[sy, sx]
    return out


def split_one(cm: np.ndarray, blob: np.ndarray, axis: int) -> np.ndarray:
    """Garis background SPLIT_LINE_PX px memotong blob (hanya piksel blob itu) di posisi yang membagi luas paling rata."""
    prof = blob.sum(axis=1 - axis)            # axis 0: horizontal (profil per baris); 1: vertikal (per kolom)
    csum = np.concatenate([[0], np.cumsum(prof)])
    total = csum[-1]
    best, best_r = None, None
    for r in range(0, len(prof) - SPLIT_LINE_PX + 1):
        above = csum[r]
        below = total - csum[r + SPLIT_LINE_PX]
        d = abs(above - below)
        if best is None or d < best:
            best, best_r = d, r
    band = np.zeros_like(blob)
    if axis == 0:
        band[best_r:best_r + SPLIT_LINE_PX, :] = True
    else:
        band[:, best_r:best_r + SPLIT_LINE_PX] = True
    out = cm.copy()
    out[blob & band] = 0
    return out


def pick_starts(rng, n: int, dur: int, clip: str) -> list[int]:
    ok = []
    for s in range(EDGE_MARGIN, n - EDGE_MARGIN - dur + 1):
        run = range(s, s + dur)
        if clip == "klip3" and any(abs(t - c) <= CUT_MARGIN for t in run for c in CUT_FRAMES):
            continue
        ok.append(s)
    rng.shuffle(ok)
    chosen: list[int] = []
    for s in ok:
        if all(abs(s - c) >= POS_SPACING for c in chosen):
            chosen.append(s)
        if len(chosen) == N_POSITIONS:
            break
    return sorted(chosen)


def build_case(c: ClipQC, kind: str, start: int, dur: int, rng) -> tuple[dict | None, str]:
    """{t: classmap sintetis} untuk frame sisipan; atau (None, alasan tidak valid)."""
    frames = list(range(start, start + dur))
    blobs = {}
    for t in frames:
        lb = largest_blob(c.cms[t] != 0)
        if lb is None:
            return None, "tanpa foreground"
        blobs[t] = lb
        if lb[1] <= c.q.blob_min * c.size:
            return None, "blob terbesar <= blob_min"
    new: dict[int, np.ndarray] = {}
    if kind == "fake":
        for _ in range(FAKE_TRIES):
            dx = int(rng.integers(-c.w + 1, c.w))
            dy = int(rng.integers(-c.h + 1, c.h))
            outs = {}
            for t in frames:
                o = make_fake(c.cms[t], blobs[t][0], dx, dy)
                if o is None:
                    break
                outs[t] = o
            else:
                new = outs
                break
        if not new:
            return None, "tidak ada offset bebas"
    elif kind == "fake_s":
        for scale in FAKE_S_SCALES:
            for _ in range(FAKE_S_TRIES):
                dx = int(rng.integers(-c.w + 1, c.w))
                dy = int(rng.integers(-c.h + 1, c.h))
                outs = {}
                for t in frames:
                    o = make_fake_scaled(c.cms[t], blobs[t][0], dx, dy, scale)
                    if o is None:
                        break
                    outs[t] = o
                else:
                    if all(seg.big_blob_count(o != 0, c.q.blob_min) >= c.n[t] + 1 for t, o in outs.items()):
                        new = outs
                    break
            if new:
                break
        if not new:
            return None, "tidak ada skala / offset yang valid"
    else:
        for t in frames:
            o = None
            for axis in (0, 1):
                cand = split_one(c.cms[t], blobs[t][0], axis)
                if seg.big_blob_count(cand != 0, c.q.blob_min) >= c.n[t] + 1:
                    o = cand
                    break
            if o is None:
                return None, "pemotongan tidak menambah blob besar"
            new[t] = o
    for t in frames:
        if seg.big_blob_count(new[t] != 0, c.q.blob_min) < c.n[t] + 1:
            return None, "sisipan tidak menambah blob besar"
    return new, ""


def synth_series(c: ClipQC, new: dict[int, np.ndarray]) -> tuple[list[int], list[int], list[float | None]]:
    n, area, iou = list(c.n), list(c.area), list(c.iou_prev)
    last = max(new)
    for t, cm in new.items():
        fg = cm != 0
        n[t] = seg.big_blob_count(fg, c.q.blob_min)
        area[t] = int(fg.sum())
        iou[t] = seg.iou(fg, c.cms[t - 1] != 0 if t - 1 not in new else new[t - 1] != 0) if t > 0 else None
    if last + 1 < len(c.cms):
        nxt = last + 1
        iou[nxt] = seg.iou(c.cms[nxt] != 0, new[last] != 0)
    return n, area, iou


def cmd_synth() -> None:
    cfg = load_pipeline(ROOT / "configs" / "default.yaml")
    SCRATCH.mkdir(parents=True, exist_ok=True)
    cases: list[dict] = []
    verified = 0
    for ci, clip in enumerate(SYNTH_CLIPS):
        c = ClipQC(clip, cfg)
        assert c.baseline_ok, f"baseline {clip} tidak direproduksi"
        base_cache = {cand: c.fails(cand) for cand in CANDIDATES}
        for ki, kind in enumerate(ALL_KINDS):
            for dur in DURATIONS:
                rng = np.random.default_rng([SEED, ci, ki, dur])
                for start in pick_starts(rng, len(c.cms), dur, clip):
                    case = {"clip": clip, "kind": kind, "dur": dur, "start": start}
                    new, why = build_case(c, kind, start, dur, rng)
                    if new is None:
                        case.update(valid=False, reason=why)
                        cases.append(case)
                        continue
                    n, area, iou = synth_series(c, new)
                    if verified < 4:   # kesetiaan: deret sintetis = seg.compute_qc penuh pada classmap yang dimodifikasi
                        cms2 = [new.get(t, cm) for t, cm in enumerate(c.cms)]
                        rows2 = seg.compute_qc(cms2, c.q, c.names, c.indices, last_frame_records(c.dir / "seg" / "frames.jsonl"))
                        assert [r["big_blobs"] for r in rows2] == n and [r["area_px"] for r in rows2] == area, "deret n / area"
                        assert [r["iou_prev"] for r in rows2] == iou, "deret iou"
                        fin_base, _, _ = c.fails("base", n, area, iou)
                        assert fin_base == [bool(r["fail_reasons"]) for r in rows2], "fail base"
                        verified += 1
                    ins = list(range(start, start + dur))
                    case.update(valid=True, reason="", n_ins=[n[t] for t in ins], n_before=[c.n[t] for t in ins], cands={})
                    for cand in CANDIDATES:
                        fin, big, oth = c.fails(cand, n, area, iou)
                        fin0, big0, oth0 = base_cache[cand]
                        hit = [fin[t] for t in ins]
                        collateral = [c.indices[t] for t in range(len(fin)) if t not in ins and fin[t] and not fin0[t]]
                        lost = [c.indices[t] for t in range(len(fin)) if t not in ins and not fin[t] and fin0[t]]
                        case["cands"][cand] = {
                            "all": all(hit), "any": any(hit), "n_hit": sum(hit),
                            "already_failing": sum(fin0[t] for t in ins),
                            "collateral": collateral, "lost": lost,
                            "other_all": all(oth[t] and not oth0[t] for t in ins),
                            "other_any": any(oth[t] and not oth0[t] for t in ins)}
                    cases.append(case)
    (SCRATCH / "synth.json").write_text(json.dumps(cases, indent=1), encoding="utf-8")
    print(f"seed {SEED}; deret sintetis diverifikasi terhadap seg.compute_qc penuh pada {verified} kasus\n")

    for clip in SYNTH_CLIPS:
        for kind in ALL_KINDS:
            sub = [x for x in cases if x["clip"] == clip and x["kind"] == kind]
            inval = Counter(x["reason"] for x in sub if not x["valid"])
            print(f"{clip}/{kind}: {len(sub)} kasus, valid {sum(x['valid'] for x in sub)}, tidak valid {dict(inval)}")
    print("\nDeteksi utama (SEMUA frame sisipan gagal) / sekunder (>= 1) — per kandidat x klip x jenis x durasi:")
    hdr = "kandidat | klip/jenis | " + " | ".join(f"durasi {d}" for d in DURATIONS) + " | gabungan"
    print(hdr)
    summary: dict = {}
    for cand in CANDIDATES:
        for clip in SYNTH_CLIPS:
            for kind in ALL_KINDS:
                cells = []
                for d in DURATIONS:
                    v = [x for x in cases if x["valid"] and x["clip"] == clip and x["kind"] == kind and x["dur"] == d]
                    cells.append(f"{sum(x['cands'][cand]['all'] for x in v)}/{sum(x['cands'][cand]['any'] for x in v)}/{len(v)}")
                v = [x for x in cases if x["valid"] and x["clip"] == clip and x["kind"] == kind]
                cells.append(f"{sum(x['cands'][cand]['all'] for x in v)}/{sum(x['cands'][cand]['any'] for x in v)}/{len(v)}")
                print(f"{cand} | {clip}/{kind} | " + " | ".join(cells))
        v = [x for x in cases if x["valid"] and x["kind"] in PRIMARY_KINDS]   # kriteria (iii): tanpa kasus TAMBAHAN fake_s
        va = [x for x in v if x["clip"] == "klip3"]
        plus1 = [x for x in v if all(b - a == 1 for a, b in zip(x["n_before"], x["n_ins"]))]
        summary[cand] = {"plus1_all": sum(x["cands"][cand]["all"] for x in plus1), "plus1_n": len(plus1),
                         "plus1_klip3_all": sum(x["cands"][cand]["all"] for x in plus1 if x["clip"] == "klip3"),
                         "plus1_klip3_n": sum(x["clip"] == "klip3" for x in plus1),
                         "all": sum(x["cands"][cand]["all"] for x in v), "any": sum(x["cands"][cand]["any"] for x in v), "n": len(v),
                         "all_klip3": sum(x["cands"][cand]["all"] for x in va), "n_klip3": len(va),
                         "all_dur3": sum(x["cands"][cand]["all"] for x in v if x["dur"] == 3), "n_dur3": sum(x["dur"] == 3 for x in v),
                         "collateral_cases": sum(bool(x["cands"][cand]["collateral"]) for x in v),
                         "collateral_frames": sum(len(x["cands"][cand]["collateral"]) for x in v),
                         "lost_cases": sum(bool(x["cands"][cand]["lost"]) for x in v),
                         "already_failing_frames": sum(x["cands"][cand]["already_failing"] for x in v),
                         "ins_frames": sum(x["dur"] for x in v)}
    print("\nformat sel: utama/sekunder/jumlah kasus valid\n")
    print("Gabungan kriteria (iii) — utama (semua frame sisipan gagal), seluruh kasus valid:")
    for cand, s in summary.items():
        print(f"  {cand:5s} utama {s['all']}/{s['n']} = {100 * s['all'] / s['n']:.1f}%  sekunder {s['any']}/{s['n']} = "
              f"{100 * s['any'] / s['n']:.1f}%  | klip3 saja {s['all_klip3']}/{s['n_klip3']} | durasi 3 saja {s['all_dur3']}/{s['n_dur3']} | "
              f"kasus dg frame non-sisipan ikut gagal: {s['collateral_cases']} ({s['collateral_frames']} frame) | "
              f"non-sisipan yang tadinya gagal jadi lolos: {s['lost_cases']} kasus | frame sisipan yang SUDAH gagal sebelum disisipi: "
              f"{s['already_failing_frames']}/{s['ins_frames']}")
    print("\nSensitivitas hanya untuk kasus yang menambah TEPAT +1 blob besar di tiap frame sisipan (utama) — gabungan / klip3:")
    for cand, s in summary.items():
        print(f"  {cand:5s} {s['plus1_all']}/{s['plus1_n']}  | klip3 {s['plus1_klip3_all']}/{s['plus1_klip3_n']}")
    sup = [x for x in cases if x["valid"] and x["kind"] == "fake_s"]
    print(f"\nTAMBAHAN (di luar 5.3, tidak masuk kriteria iii): blob palsu DIPERKECIL (skala 0.8–0.5), {len(sup)} kasus valid")
    for cand in CANDIDATES:
        cells = []
        for clip in SYNTH_CLIPS:
            vv = [x for x in sup if x["clip"] == clip]
            cells.append(f"{clip} {sum(x['cands'][cand]['all'] for x in vv)}/{sum(x['cands'][cand]['any'] for x in vv)}/{len(vv)}")
        print(f"  {cand:5s} " + " | ".join(cells))
    v = [x for x in cases if x["valid"] and x["kind"] in PRIMARY_KINDS]
    print("\nMetrik QC lain (iou_prev / area_ratio / area_vs_median) sebagai konteks — kasus di mana metrik lain SAJA sudah menangkap:")
    for kind in ("fake", "split"):
        for clip in SYNTH_CLIPS:
            vv = [x for x in v if x["clip"] == clip and x["kind"] == kind]
            if vv:
                print(f"  {clip}/{kind}: utama {sum(x['cands']['D']['other_all'] for x in vv)}/{len(vv)}, "
                      f"sekunder {sum(x['cands']['D']['other_any'] for x in vv)}/{len(vv)}")
    (SCRATCH / "synth_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print("\nDaftar kasus (klip jenis durasi start valid n_sisipan <- n_asli):")
    for x in cases:
        print(f"  {x['clip']} {x['kind']} d{x['dur']} f{x['start']} "
              + (f"valid n {x['n_before']}→{x['n_ins']}" if x["valid"] else f"TIDAK VALID ({x['reason']})"))


# ── 5.4 biaya ──────────────────────────────────────
def cmd_cost() -> None:
    cfg = load_pipeline(ROOT / "configs" / "default.yaml")
    print("Biaya per frame (ms), CPU; klip3 (182 frame) dan test (283 frame)")
    for clip in ("klip3", "test"):
        d = CLIPS_DIR / clip
        names = [r["frame"] for r in json.loads((d / "qc_report.json").read_text(encoding="utf-8"))["frames"]]
        tr, tm = [], []
        prev = None
        for nm in names:
            t0 = time.perf_counter()
            cm = seg.read_classmap(d / "seg" / "classmap" / (Path(nm).stem + ".png"))
            t1 = time.perf_counter()
            fg = cm != 0
            seg.big_blob_count(fg, cfg.qc.blob_min)
            seg.iou(fg, prev) if prev is not None else None
            seg.label_agreement(cm, prev) if prev is not None else None
            t2 = time.perf_counter()
            prev = fg
            tr.append(t1 - t0)
            tm.append(t2 - t1)
        p95 = lambda a: float(np.percentile(a, 95)) * 1000  # noqa: E731
        print(f"  {clip}: baca classmap mean {statistics.fmean(tr) * 1000:.2f} p95 {p95(tr):.2f} | metrik per frame (blob, iou, label) "
              f"mean {statistics.fmean(tm) * 1000:.2f} p95 {p95(tm):.2f}")
        n = [r["big_blobs"] for r in json.loads((d / "qc_report.json").read_text(encoding="utf-8"))["frames"]]
        for cand in CANDIDATES:
            reps = []
            for _ in range(200):
                t0 = time.perf_counter()
                big_rule(cand, n, cfg.qc.max_big_blobs)
                reps.append((time.perf_counter() - t0) / len(n))
            print(f"    aturan {cand:5s}: mean {statistics.fmean(reps) * 1e6:.2f} µs/frame, p95 {float(np.percentile(reps, 95)) * 1e6:.2f} µs/frame "
                  f"(200 pengulangan atas deret n seluruh klip)")


# ── 5.1 rantai basi [3] ────────────────────────────
def run_stab(work: Path) -> tuple[str, float]:
    t0 = time.perf_counter()
    r = subprocess.run([sys.executable, "-m", "rotoscope.stabilize", "--work-dir", str(work)], cwd=ROOT,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    return (r.stdout + r.stderr), time.perf_counter() - t0


def show(label: str, out: str, secs: float) -> None:
    keep = [ln for ln in out.splitlines() if not ln.lstrip().startswith("[") or "stabilize" in ln]
    print(f"--- {label} ({secs:.1f} s)")
    for ln in keep[:14]:
        print("   ", ln[:230])


def cmd_stale() -> None:
    src = CLIPS_DIR / "test_short"
    dst = SCRATCH / "stale_test_short"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    rep_path = dst / "qc_report.json"
    orig_bytes = (src / "qc_report.json").read_bytes()
    orig = orig_bytes.decode("utf-8")
    print(f"salinan scratch: {dst.relative_to(ROOT).as_posix()}")
    show("0. salinan apa adanya", *run_stab(dst))

    def edit(idx: int, reasons: list[str]) -> None:
        rep = json.loads(orig)
        rep["frames"][idx]["fail_reasons"] = reasons
        rep_path.write_text(json.dumps(rep, indent=1), encoding="utf-8")

    edit(60, ["big_blobs"])
    show("1. frame index 60 dibuat gagal (big_blobs) di qc_report salinan", *run_stab(dst))
    show("1b. ulang tanpa perubahan", *run_stab(dst))
    edit(60, ["iou_prev"])
    show("2. alasan gagal frame 60 diganti iou_prev (tetap gagal)", *run_stab(dst))
    edit(61, ["iou_prev"])
    show("3. tambah frame 61 gagal", *run_stab(dst))
    rep_path.write_bytes(orig_bytes)
    show("4. qc_report dikembalikan ke asli", *run_stab(dst))


def main(argv: list[str]) -> int:
    if len(argv) == 3 and argv[1] == "hash" and argv[2] in ("before", "after", "before_t23", "after_t23"):
        cmd_hash(argv[2])
    elif len(argv) == 2 and argv[1] in ("qc", "synth", "cost", "stale"):
        {"qc": cmd_qc, "synth": cmd_synth, "cost": cmd_cost, "stale": cmd_stale}[argv[1]]()
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
