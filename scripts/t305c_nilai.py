"""T-305c Tahap 3: bahan penilaian visual Rio (klip3, lantai hi_floor 0 / 0,02 / 0,025) -> work/t305c/nilai/. Baca-saja terhadap
klip asli; sumber = salinan scratch work/t305c_scratch/t3/ (hasil `t305c_stage3.py run`; style default rough-sketch, kertas default).

    python scripts/t305c_nilai.py videos   # 01-03 MP4 bernomor + 04 berdampingan 0 | 0,02 | 0,025 (label besar)
    python scripts/t305c_nilai.py loss     # papan "O hilang" untuk lantai 0,02 dan 0,025
    python scripts/t305c_nilai.py frames   # 4 frame utuh dengan selisih strok terbesar, tiga varian berdampingan
    python scripts/t305c_nilai.py all
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "tests"))
import t305c_eval as ev  # noqa: E402
import t305c_study as st  # noqa: E402

ROOT = st.ROOT
T3 = st.SCRATCH / "t3"
OUT = ROOT / "work" / "t305c" / "nilai"
FLOORS = (("0", "lantai 0 (sekarang)"), ("0.02", "lantai 0.02"), ("0.025", "lantai 0.025"))
VIDEO_SCALE = 0.5
LABEL_H = 110
N_FRAMES_PICK = 4
MIN_GAP = 10                       # jarak minimum antar frame terpilih
FPS = 24


def mp4(floor: str) -> Path:
    return T3 / f"f{floor}" / "out" / "klip3.mp4"


def label_bar(width: int, text: str) -> np.ndarray:
    bar = np.full((LABEL_H, width, 3), 255, np.uint8)
    cv2.putText(bar, text, (14, LABEL_H - 28), cv2.FONT_HERSHEY_DUPLEX, 1.8, (0, 0, 0), 4, cv2.LINE_AA)
    return bar


def read_frames(path: Path):
    cap = cv2.VideoCapture(str(path))
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        yield fr
    cap.release()


def cmd_videos() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    names = {"0": "01_lantai_0.mp4", "0.02": "02_lantai_0.02.mp4", "0.025": "03_lantai_0.025.mp4"}
    for fl, nm in names.items():
        shutil.copyfile(mp4(fl), OUT / nm)
        print(nm)
    tmp = OUT / "_tmp_berdampingan.mp4"
    writer = None
    for frames in zip(*(read_frames(mp4(fl)) for fl, _ in FLOORS)):
        panels = []
        for (fl, title), fr in zip(FLOORS, frames):
            fr = cv2.resize(fr, None, fx=VIDEO_SCALE, fy=VIDEO_SCALE, interpolation=cv2.INTER_AREA)
            panels.append(np.vstack([label_bar(fr.shape[1], title.upper()), fr]))
        img = np.hstack(panels)
        if writer is None:
            writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (img.shape[1], img.shape[0]))
        writer.write(img)
    writer.release()
    final = OUT / "04_berdampingan_0_0.02_0.025.mp4"
    r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp), "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p",
                        "-r", str(FPS), str(final)], capture_output=True, text=True)
    if r.returncode == 0:
        tmp.unlink()
    else:
        shutil.move(tmp, final)
        print("PERINGATAN: ffmpeg gagal, berkas berdampingan = mp4v:", r.stderr[-200:])
    print(final.name)


def kept(base_pts: np.ndarray, cand: list[dict]) -> bool:
    if not cand:
        return False
    tree = cKDTree(np.concatenate([np.asarray(s["points"], float) for s in cand]))
    return bool(np.mean(tree.query(base_pts)[0] <= ev.MATCH_RADIUS) >= ev.MATCH_FRACTION)


def contours_docs(floor: str) -> list[dict]:
    d = T3 / f"f{floor}" / "klip3" / "contours"
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(d.glob("frame_*.json"))]


def loss_cell(clip: st.ClipData, u: dict, number: int, var_doc: dict, caption: str) -> np.ndarray:
    i = u["frame"]
    img = clip.frame_image(i)
    x0, y0, x1, y1 = st.clip_boxes({u["clip"]: clip}, u)
    crop, s = st.fit(img[y0:y1, x0:x1], st.SLOT)
    plain, base, var = crop.copy(), crop.copy(), crop.copy()
    strokes = clip.docs[i]["strokes"]
    for j, sk in enumerate(strokes):
        if j != u["_stroke_idx"]:
            st.draw_poly(base, np.asarray(sk["points"], float), (x0, y0), s, st.GRAY, 1)
    st.draw_poly(base, np.asarray(strokes[u["_stroke_idx"]]["points"], float), (x0, y0), s, st.MAGENTA, 3)
    for sk in var_doc["strokes"]:
        st.draw_poly(var, np.asarray(sk["points"], float), (x0, y0), s, st.GRAY, 1)
    cell = np.full((st.HEADER + st.SLOT + st.GAP, 3 * st.SLOT + 4 * st.GAP, 3), 255, np.uint8)
    cv2.putText(cell, f"{number:02d}", (st.GAP, st.HEADER - 12), cv2.FONT_HERSHEY_DUPLEX, 2.2, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(cell, caption, (150, st.HEADER - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 60), 1, cv2.LINE_AA)
    for k, panel in enumerate((plain, base, var)):
        sl = st.slot(panel)
        cv2.rectangle(sl, (0, 0), (st.SLOT - 1, st.SLOT - 1), (200, 200, 200), 1)
        cell[st.HEADER:st.HEADER + st.SLOT, st.GAP + k * (st.SLOT + st.GAP):st.GAP + k * (st.SLOT + st.GAP) + st.SLOT] = sl
    cv2.rectangle(cell, (0, 0), (cell.shape[1] - 1, cell.shape[0] - 1), (120, 120, 120), 2)
    return cell


def cmd_loss() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    units, _ = ev.load_units()
    clips = {c: st.ClipData(c) for c in st.STUDY_CLIPS}
    labels = st.jload(st.SCRATCH / "labels.json")["labels"]
    num_of = {(v["clip"], v["track_id"], v["run_start"]): n for n, v in labels.items()}
    report = {}
    for fl in ("0.02", "0.025"):
        docs_by = {"klip3": contours_docs(fl)}
        for c in ("test", "Klip2"):
            docs_by[c] = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((T3 / f"f{fl}" / c / "contours").glob("frame_*.json"))]
        lost_units = []
        for u in units:
            if u["label"] != "O":
                continue
            clip = clips[u["clip"]]
            flags = []
            for r in u["rows"]:
                base = next(s for s in clip.docs[r["frame"]]["strokes"] if s["type"] == "occlusion" and s["track_id"] == r["track_id"])
                cand = [s for s in docs_by[u["clip"]][r["frame"]]["strokes"] if s["type"] == "occlusion"]
                flags.append(kept(np.asarray(base["points"], float), cand))
            frac = float(np.mean(flags))
            if frac < 0.5:
                rows = sorted(u["rows"], key=lambda r: r["frame"])
                flg = [f for r, f in sorted(zip(u["rows"], flags), key=lambda t: t[0]["frame"])]
                rep = len(rows) // 2
                pick = rep if not flg[rep] else next(k for k, f in enumerate(flg) if not f)
                lost_units.append((u, rows[pick], f"{u['clip']} f{rows[pick]['frame']} | {sum(not f for f in flags)}/{len(flags)} strok hilang"))
        report[fl] = {"o_units_total": sum(u["label"] == "O" for u in units), "o_units_lost_majority": len(lost_units),
                      "boards": []}
        print(f"lantai {fl}: {len(lost_units)} track O hilang (>= 50% strok)")
        per_page = st.PAGE_COLS * st.PAGE_ROWS
        cell_h, cell_w = st.HEADER + st.SLOT + st.GAP, 3 * st.SLOT + 4 * st.GAP
        for pg in range(max(1, -(-len(lost_units) // per_page))):
            page = np.full((st.PAGE_ROWS * (cell_h + st.GAP) + st.GAP, st.PAGE_COLS * (cell_w + st.GAP) + st.GAP, 3), 255, np.uint8)
            for pos in range(per_page):
                n = pg * per_page + pos
                if n >= len(lost_units):
                    break
                u, row, cap = lost_units[n]
                cell = loss_cell(clips[u["clip"]], {"clip": u["clip"], "frame": row["frame"], "track_id": row["track_id"],
                                                    "_stroke_idx": st.stroke_index(clips, {"clip": u["clip"], "frame": row["frame"], "track_id": row["track_id"]})},
                                 n + 1, docs_by[u["clip"]][row["frame"]], f"{cap} | papan v2 #{num_of[u['key']]}")
                r_, c_ = divmod(pos, st.PAGE_COLS)
                y, x = st.GAP + r_ * (cell_h + st.GAP), st.GAP + c_ * (cell_w + st.GAP)
                page[y:y + cell_h, x:x + cell_w] = cell
            path = OUT / f"06_O_hilang_lantai_{fl}_p{pg + 1:02d}.png"
            cv2.imwrite(str(path), page)
            report[fl]["boards"].append(path.name)
            print(path.name)
    st.jsave(OUT.parent.parent / "t305c_scratch" / "t3" / "loss_report.json", report)


def cmd_frames() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d0, d2, d25 = contours_docs("0"), contours_docs("0.02"), contours_docs("0.025")
    n_occ = lambda d: sum(s["type"] == "occlusion" for s in d["strokes"])      # noqa: E731
    diff = np.array([(n_occ(a) - n_occ(c), n_occ(a) - n_occ(b)) for a, b, c in zip(d0, d2, d25)])
    order = np.lexsort((-diff[:, 1], -diff[:, 0]))
    picked: list[int] = []
    for i in order:
        if all(abs(int(i) - p) >= MIN_GAP for p in picked):
            picked.append(int(i))
        if len(picked) == N_FRAMES_PICK:
            break
    picked.sort()
    caps = {fl: cv2.VideoCapture(str(mp4(fl))) for fl, _ in FLOORS}
    info = []
    for i in picked:
        panels = []
        for fl, title in FLOORS:
            caps[fl].set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, fr = caps[fl].read()
            assert ok
            fr = cv2.resize(fr, None, fx=VIDEO_SCALE, fy=VIDEO_SCALE, interpolation=cv2.INTER_AREA)
            panels.append(np.vstack([label_bar(fr.shape[1], title.upper()), fr]))
        path = OUT / f"07_frame_{i:03d}_tiga_varian.png"
        cv2.imwrite(str(path), np.hstack(panels))
        info.append({"frame": i, "occlusion_strokes": [n_occ(d0[i]), n_occ(d2[i]), n_occ(d25[i])]})
        print(path.name, info[-1])
    st.jsave(T3 / "frames_picked.json", info)


if __name__ == "__main__":
    cmds = {"videos": [cmd_videos], "loss": [cmd_loss], "frames": [cmd_frames], "all": [cmd_videos, cmd_loss, cmd_frames]}
    if len(sys.argv) < 2 or sys.argv[1] not in cmds:
        raise SystemExit(__doc__)
    for f in cmds[sys.argv[1]]:
        f()
