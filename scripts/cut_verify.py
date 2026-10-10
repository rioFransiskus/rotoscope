"""Bukti penerapan cut_diff 0,06 (alat sekali pakai; CPU; hasil JSON ke work/cut_scratch/):

    python scripts/cut_verify.py old <klip>   # [3] + [4] dengan cut_diff 0,08 pada SALINAN scratch (work/cut_scratch/old/<klip>) sebagai pembanding "sebelum"
    python scripts/cut_verify.py hashes       # hulu / stable / strokes / out: hash sekarang vs work/cut_scratch/hash_before.txt
    python scripts/cut_verify.py contours     # contours sekarang vs scratch lama, field per field (test, test_short, Klip2)
    python scripts/cut_verify.py klip3        # klip3: cut, jendela kernel, frame stable berubah, IoU di sisi cut, flip-flop, track

Tidak menulis ke work/clips/*; hanya membaca."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "src"))

import temporal_metrics as tm  # noqa: E402

from rotoscope import stabilize as stb  # noqa: E402
from rotoscope.config import load_pipeline  # noqa: E402

SCR = ROOT / "work" / "cut_scratch"
CLIPS = ("test", "test_short", "Klip2", "klip3")


def jload(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def cmd_old(clip: str) -> None:
    src, work = ROOT / "work" / "clips" / clip, SCR / "old" / clip
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    for d in ("frames", "seg", "depth"):
        shutil.copytree(src / d, work / d)
    for f in ("meta.json", "qc_report.json"):
        shutil.copyfile(src / f, work / f)
    cfgfile = SCR / "old" / f"{clip}_old.yaml"
    cfgfile.write_text("stabilize:\n  temporal:\n    cut_diff: 0.08\n", encoding="utf-8")
    for mod in ("stabilize", "vectorize"):
        r = subprocess.run([sys.executable, "-m", f"rotoscope.{mod}", "--config", str(cfgfile), "--work-dir", str(work)], cwd=ROOT,
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        print(clip, mod, "rc", r.returncode, (r.stdout.strip().splitlines() or [""])[-1][:100], flush=True)
        if r.returncode:
            raise SystemExit(r.stderr[-800:])


def before_hashes() -> dict[str, str]:
    out = {}
    for line in (SCR / "hash_before.txt").read_text(encoding="utf-8").splitlines():
        h, p = line.split(maxsplit=1)
        out[p.lstrip("*").strip()] = h
    return out


def cmd_hashes() -> None:
    before = before_hashes()
    res = {}
    for c in CLIPS:
        groups: dict[str, dict] = {}
        for key, sel in (("frames_seg_depth", lambda p: p.startswith(f"work/clips/{c}/") and p.split("/")[3] in ("frames", "seg", "depth")),
                         ("stable_groups_depth_smooth", lambda p: p.startswith(f"work/clips/{c}/stable/") and p.split("/")[4] in ("groups", "depth_smooth")),
                         ("stable_manifest_frames_jsonl", lambda p: p.startswith(f"work/clips/{c}/stable/") and p.split("/")[-1] in ("manifest.json", "frames.jsonl")),
                         ("strokes", lambda p: p.startswith(f"work/clips/{c}/strokes/") and p.endswith((".png", ".svg"))),
                         ("contours", lambda p: p.startswith(f"work/clips/{c}/contours/") and p.endswith(".json")),
                         ("out_mp4", lambda p: p == f"out/{c}.mp4"),
                         ("out_svg", lambda p: p.startswith(f"out/svg/{c}/"))):
            files = [p for p in before if sel(p)]
            same = diff = missing = 0
            first = []
            for p in files:
                f = ROOT / p
                if not f.is_file():
                    missing += 1
                elif sha(f) == before[p]:
                    same += 1
                else:
                    diff += 1
                    if len(first) < 8:
                        first.append(p.split("/")[-1])
            groups[key] = {"files": len(files), "identical": same, "different": diff, "missing": missing, "first_different": first}
        res[c] = groups
        print(c, {k: (v["files"], v["identical"], v["different"]) for k, v in groups.items()})
    (SCR / "hashes_check.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


def diff_fields(a, b, path="") -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{path}/{k}: hanya di {'lama' if k in a else 'baru'}")
            else:
                out += diff_fields(a[k], b[k], f"{path}/{k}")
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: panjang {len(a)} vs {len(b)}"]
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out += diff_fields(x, y, f"{path}[{i}]")
            if len(out) > 20:
                break
        return out
    return [] if a == b else [f"{path}: {str(a)[:40]} vs {str(b)[:40]}"]


def cmd_contours() -> None:
    res = {}
    for c in ("test", "test_short", "Klip2"):
        old, new = SCR / "old" / c, ROOT / "work" / "clips" / c
        stable_same = all(sha(p) == sha(new / "stable" / p.relative_to(old / "stable")) for sub in ("groups", "depth_smooth") for p in sorted((old / "stable" / sub).glob("*")))
        n = len(list((old / "contours").glob("frame_*.json")))
        keys_diff: dict[str, int] = {}
        strokes_equal = 0
        for i in range(n):
            a, b = jload(old / "contours" / f"frame_{i:05d}.json"), jload(new / "contours" / f"frame_{i:05d}.json")
            for d in diff_fields(a, b):
                k = d.split(":")[0].split("[")[0]
                keys_diff[k] = keys_diff.get(k, 0) + 1
            strokes_equal += a["strokes"] == b["strokes"]
        man_old, man_new = jload(old / "contours" / "manifest.json"), jload(new / "contours" / "manifest.json")
        res[c] = {"frames": n, "stable_groups_depth_identical_old_vs_new": stable_same, "frames_strokes_equal": strokes_equal,
                  "frame_fields_differing(count over frames)": keys_diff, "manifest_field_diff": diff_fields(man_old, man_new)[:12],
                  "clip_stats_diff": diff_fields(jload(old / "contours" / "clip_stats.json"), jload(new / "contours" / "clip_stats.json"))[:8]}
        print(c, res[c])
    (SCR / "contours_check.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")


def fgiou(a, b):
    x, y = a != 0, b != 0
    return float((x & y).sum() / max((x | y).sum(), 1))


def gmean(a, b):
    ids = [g for g in np.unique(a) if g != 0]
    return float(np.mean([((a == g) & (b == g)).sum() / max(((a == g) | (b == g)).sum(), 1) for g in ids])) if ids else 1.0


def cmd_klip3() -> None:
    cfg = load_pipeline()
    names = tuple(g for g, _ in cfg.groups)
    W = ROOT / "work" / "clips" / "klip3"
    clip = stb.load_clip(W)
    n = len(clip.names)
    plan = stb.build_plan(cfg, clip, log=lambda *_: None)
    classes = jload(W / "seg" / "manifest.json")["classes"]
    lut = stb.class_group_lut(cfg.groups, classes)
    raw = np.stack([lut[cv2.imdecode(np.fromfile(W / "seg" / "classmap" / nm, np.uint8), cv2.IMREAD_UNCHANGED)] for nm in clip.names])
    after = np.stack([stb.read_groups(clip.groups_path(nm)) for nm in clip.names])
    before = np.stack([stb.read_groups(SCR / "klip3_before" / "groups" / nm.replace(".png", ".png")) for nm in clip.names])
    cuts = list(plan.cut_frames)
    res: dict = {"cuts_detected": cuts, "cut_diff": cfg.stabilize.temporal.cut_diff, "expected": [49, 85, 126, 134, 146, 154, 166, 174],
                 "cuts_equal_expected": cuts == [49, 85, 126, 134, 146, 154, 166, 174],
                 "manifest_cut_frames": jload(W / "stable" / "manifest.json")["temporal"]["cut_frames"], "R": plan.radius}
    crossing = [t for t in range(n) if any(plan.window(t, plan.radius)[0] < c <= plan.window(t, plan.radius)[1] for c in cuts)]
    res["frames_whose_kernel_window_crosses_a_cut"] = crossing
    res["window_samples"] = {str(c): {"t-1": plan.window(c - 1, plan.radius), "t": plan.window(c, plan.radius)} for c in cuts}
    # frame stable yang berubah vs sebelum
    changed = [t for t in range(n) if not (before[t] == after[t]).all()]
    px = {t: int((before[t] != after[t]).sum()) for t in changed}
    near = lambda t: any(c - plan.radius <= t <= c + plan.radius - 1 for c in [49, 85, 126, 154])  # noqa: E731
    expect_window = sorted({t for c in (49, 85, 126, 154) for t in range(c - plan.radius, c + plan.radius)})
    res["stable_changed_frames"] = changed
    res["stable_changed_px"] = px
    res["expected_affected_frames_around_new_cuts(+-R)"] = expect_window
    res["changed_outside_expected"] = [t for t in changed if t not in expect_window]
    res["expected_but_unchanged"] = [t for t in expect_window if t not in changed]
    # IoU stabil vs mentah, kedua sisi tiap cut vs frame non-cut
    selffg = np.array([fgiou(after[t], raw[t]) for t in range(n)])
    selfg = np.array([gmean(raw[t], after[t]) for t in range(n)])
    near_cut = {x for c in cuts for x in (c - 1, c)}
    nonc = [t for t in range(n) if t not in near_cut]
    res["reference_noncut"] = {"fg_p5": float(np.percentile(selffg[nonc], 5)), "fg_p50": float(np.median(selffg[nonc])),
                               "grp_p5": float(np.percentile(selfg[nonc], 5)), "grp_p50": float(np.median(selfg[nonc]))}
    res["cut_sides"] = {str(c): {"t-1": {"fg": float(selffg[c - 1]), "grp": float(selfg[c - 1]), "fg_vs_other_side_raw": fgiou(after[c - 1], raw[c]),
                                         "grp_before": float(gmean(raw[c - 1], before[c - 1]))},
                                 "t": {"fg": float(selffg[c]), "grp": float(selfg[c]), "fg_vs_other_side_raw": fgiou(after[c], raw[c - 1]),
                                       "grp_before": float(gmean(raw[c], before[c]))}} for c in cuts}
    # flip-flop dan track, sebelum vs sesudah
    static = slice(12, 24)
    sel_static = np.zeros(n, bool)
    sel_static[static] = True
    noncut_mask = np.ones(n, bool)
    for c in cuts:
        noncut_mask[max(c - plan.radius, 0):c + plan.radius] = False
    res["flipflop_per10k"] = {"static_f12_23": {"before": tm.flipflop_per10k(before, sel_static), "after": tm.flipflop_per10k(after, sel_static), "raw": tm.flipflop_per10k(raw, sel_static)},
                              "all": {"before": tm.flipflop_per10k(before), "after": tm.flipflop_per10k(after), "raw": tm.flipflop_per10k(raw)},
                              "frames_far_from_all_cuts": {"before": tm.flipflop_per10k(before, noncut_mask), "after": tm.flipflop_per10k(after, noncut_mask), "raw": tm.flipflop_per10k(raw, noncut_mask)},
                              "frames_near_new_cuts_49_85_126_154": {"before": tm.flipflop_per10k(before, ~noncut_mask & np.isin(np.arange(n), expect_window)), "after": tm.flipflop_per10k(after, ~noncut_mask & np.isin(np.arange(n), expect_window))}}
    fgb = (before != 0).sum(axis=(1, 2))
    res["fg_px_change_in_changed_frames"] = {str(t): [int(fgb[t]), int((after[t] != 0).sum())] for t in changed}
    # track: contours sebelum vs sesudah
    types = ("silhouette", "silhouette_hole", "group_boundary", "occlusion")
    fb = tm.stroke_lengths_from_dir(SCR / "klip3_before" / "contours")
    fa = tm.stroke_lengths_from_dir(W / "contours")
    pb, pa = tm.pop_energy_by_type(fb, types), tm.pop_energy_by_type(fa, types)
    res["track"] = {t: {"pop_before": pb["types"][t]["pop_energy_mean"], "pop_after": pa["types"][t]["pop_energy_mean"],
                        "age_before": [pb["types"][t]["age_mean"], pb["types"][t]["age_median"]], "age_after": [pa["types"][t]["age_mean"], pa["types"][t]["age_median"]],
                        "id_new_before": pb["types"][t]["id_new_per_stroke_frame"], "id_new_after": pa["types"][t]["id_new_per_stroke_frame"]} for t in types}
    res["track"]["total_pop_before_after"] = [pb["total_pop_energy_mean"], pa["total_pop_energy_mean"]]
    # contours per frame: berapa frame berubah isi strok
    nb = na = 0
    changed_c = []
    for i in range(n):
        a = jload(SCR / "klip3_before" / "contours" / f"frame_{i:05d}.json")["strokes"]
        b = jload(W / "contours" / f"frame_{i:05d}.json")["strokes"]
        if a != b:
            changed_c.append(i)
    res["contours_frames_with_changed_strokes"] = changed_c
    (SCR / "klip3_check.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k not in ("fg_px_change_in_changed_frames",)}, indent=1, ensure_ascii=False)[:9000])


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["old"]:
        cmd_old(a[1])
    elif a[:1] == ["hashes"]:
        cmd_hashes()
    elif a[:1] == ["contours"]:
        cmd_contours()
    elif a[:1] == ["klip3"]:
        cmd_klip3()
    else:
        sys.exit(__doc__)
