"""Pengukuran revisi T-403 (alat sekali pakai; membaca contours/ SALINAN klip di folder `--s3`, tidak menulis ke klip asli):

    python scripts/t403_fix.py --s3 <dir> edge    # tinta 3 baris/kolom terluar: piksel berubah per pass, SEMUA frame, offset 2,7 / 5,5 / 8 / 12
    python scripts/t403_fix.py --s3 <dir> seam    # distribusi seam_jump / interior_change_max per offset + batas teoretis |gradD| x jarak titik
    python scripts/t403_fix.py --s3 <dir> hash    # sha256 PNG + SVG (render_frame) passes 2 / 3 untuk pembandingan sebelum / sesudah optimasi

Hasil JSON ke work/t403/."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "src"))

import jitter_metrics as jm  # noqa: E402
import multipass_metrics as mm  # noqa: E402

from rotoscope import stylize as sty  # noqa: E402
from rotoscope.config import load_style  # noqa: E402

OUT = ROOT / "work" / "t403"
CLIPS = ("test_short", "test")
OFFSETS = (2.7, 5.5, 8.0, 12.0)


def load(s3: Path, clip: str):
    meta = json.loads((s3 / "work" / "clips" / clip / "meta.json").read_text(encoding="utf-8"))
    return meta, int(meta["frame_count"])


def doc_of(s3: Path, clip: str, i: int) -> dict:
    return json.loads((s3 / "work" / "clips" / clip / "contours" / f"frame_{i:05d}.json").read_text(encoding="utf-8"))


def write(name: str, obj) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"  work/t403/{name}")


def cmd_edge(s3: Path) -> None:
    res = {}
    for clip in CLIPS:
        meta, n = load(s3, clip)
        for off in OFFSETS:
            style = load_style(None, overrides={"multipass.passes": 3, "multipass.offset": off})
            g = sty.make_geometry(style, int(meta["working_width"]), int(meta["working_height"]))
            total, bad_frames, worst = 0, [], 0
            for i in range(n):
                doc = doc_of(s3, clip, i)
                passes, _ = sty.frame_passes(doc, g)
                for k in (1, 2):
                    c = jm.edge_ink_changed(passes[0], passes[k], sty.pass_geometry(g, k))
                    total += c
                    worst = max(worst, c)
                    if c:
                        bad_frames.append((int(doc["frame_index"]), k, c))
            res[f"{clip}/off{off:g}"] = {"frames": n, "pass_frames": 2 * n, "edge_ink_px_total": total, "worst_frame_px": worst,
                                         "frames_nonzero": bad_frames[:20], "n_nonzero": len(bad_frames)}
            print(clip, off, res[f"{clip}/off{off:g}"], flush=True)
    write("s4_edge.json", res)


def cmd_seam(s3: Path) -> None:
    """Per offset (pass 1 dari 2 pass): distribusi seam_jump / interior_change_max, jumlah gagal kriteria relatif 1,25×, jumlah melewati batas
    Lipschitz (jm.seam_bound), jumlah gagal KEDUANYA (jm.seam_ok), dan seam_jump / batas terbesar."""
    res = {}
    for clip in CLIPS:
        meta, n = load(s3, clip)
        for off in OFFSETS:
            style = load_style(None, overrides={"multipass.passes": 2, "multipass.offset": off})
            g = sty.make_geometry(style, int(meta["working_width"]), int(meta["working_height"]))
            gp = sty.pass_geometry(g, 1)
            ratios, bratio, iratio, over_bound, fail_both, sj_all = [], [], [], 0, 0, []
            for i in range(n):
                passes, _ = sty.frame_passes(doc_of(s3, clip, i), g)
                for b, p in zip(passes[0], passes[1]):
                    if b.closed:
                        sj = jm.seam_jump(b, p)
                        bd = jm.seam_bound(b, gp)
                        ratios.append(jm.seam_ratio(b, p))
                        bratio.append(sj / max(bd, 1e-9))
                        iratio.append(jm.interior_change_max(b, p) / max(jm.interior_bound(b, gp), 1e-9))
                        sj_all.append(sj)
                        over_bound += int(sj > bd + 1e-9)
                        fail_both += int(not jm.seam_ok(b, p, gp))
            r, br = np.array(ratios), np.array(bratio)
            res[f"{clip}/off{off:g}"] = {"closed_strokes": len(r), "ratio_p50": float(np.percentile(r, 50)), "ratio_p95": float(np.percentile(r, 95)),
                                         "ratio_max": float(r.max()), "over_1.25": int((r > jm.SEAM_REL_TOL).sum()),
                                         "seam_jump_max_px": float(max(sj_all)), "over_lipschitz_bound": over_bound,
                                         "seam_over_bound_ratio_p95": float(np.percentile(br, 95)), "seam_over_bound_ratio_max": float(br.max()),
                                         "interior_over_bound_ratio_max": float(max(iratio)), "fail_all_criteria": fail_both}
            print(clip, off, res[f"{clip}/off{off:g}"], flush=True)
    write("s4_seam.json", res)


def cmd_hash(s3: Path, tag: str) -> None:
    res = {}
    for clip in CLIPS:
        meta, n = load(s3, clip)
        w, h = int(meta["working_width"]), int(meta["working_height"])
        for name, ov in (("p2_off8_op0.92", {"multipass.passes": 2, "multipass.offset": 8.0, "stroke.opacity": 0.92}),
                         ("p3_off8_op0.92", {"multipass.passes": 3, "multipass.offset": 8.0, "stroke.opacity": 0.92}),
                         ("p3_off2.7_op1.0", {"multipass.passes": 3, "multipass.offset": 2.7}),
                         ("p2_off12_op0.85", {"multipass.passes": 2, "multipass.offset": 12.0, "stroke.opacity": 0.85})):
            st = load_style(None, overrides=ov)
            g = sty.make_geometry(st, w, h)
            hh = hashlib.sha256()
            for i in range(n):
                svg, png, _ = sty.render_frame(doc_of(s3, clip, i), g, st)
                hh.update(svg)
                hh.update(png)
            res[f"{clip}/{name}"] = hh.hexdigest()
            print(clip, name, res[f"{clip}/{name}"], flush=True)
    write(f"s4_hash_{tag}.json", res)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s3", required=True, type=Path)
    ap.add_argument("cmd", choices=["edge", "seam", "hash"])
    ap.add_argument("--tag", default="x")
    a = ap.parse_args()
    {"edge": cmd_edge, "seam": cmd_seam}.get(a.cmd, lambda s: cmd_hash(s, a.tag))(a.s3)


if __name__ == "__main__":
    main()
