"""T-305b mutation check: monkeypatch DARI LUAR berkas test (kode produksi tidak diubah di disk). Tiap mutasi harus membuat
tests/test_vectorize_reach.py GAGAL. Pakai: python scripts/t305b_mutations.py  → work/t305b/mutations.json.
Satu proses per mutasi (modul bersih)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402

from rotoscope import vectorize as vec  # noqa: E402

TESTS = ["tests/test_vectorize_reach.py"]
ORIG = {n: getattr(vec, n) for n in ("surviving_seeds", "reach_extension", "depth_params", "boundary_distance",
                                      "vectorize_frame", "occlusion_strokes", "vectorize_params", "clip_stats_inputs")}


def mutate_component_without_L_gate():                 # benih tidak disyaratkan lolos L
    vec.surviving_seeds = lambda gmap, seed, params, skip: seed


def mutate_zone_component_without_seed_kept():          # komponen seluruhnya di zona D (tanpa benih) ikut dibuat
    vec.reach_extension = lambda kept, ridge, dist, d, d_low: ridge & ~kept


def mutate_d_low_ignored():
    def dp(params):
        out = ORIG["depth_params"](params)
        out["min_dist_low_px"] = 0.0
        return out
    vec.depth_params = dp


def mutate_extension_not_connected_to_seed():          # ekstensi = semua piksel dekat dasar zona, tanpa hubungan ke benih
    def ext(kept, ridge, dist, d, d_low):
        return ridge & ~kept & (dist < d_low + 3.0)
    vec.reach_extension = ext


def mutate_exclude_removes_group_boundary():
    def vf(gmap, depth, names, params, t_high, t_low):
        strokes, stats = ORIG["vectorize_frame"](gmap, depth, names, params, t_high, t_low)
        ex = set(params.get("depth_lines.exclude_groups", []))
        return [s for s in strokes if not (ex & set(s["groups"]))], stats
    vec.vectorize_frame = vf


def mutate_exclude_ignored():
    def dp(params):
        out = ORIG["depth_params"](params)
        out["exclude_groups"] = []
        return out
    vec.depth_params = dp


def mutate_frame_edge_not_boundary():
    def bd(gmap):                                     # tanpa padding: tepi frame bukan batas
        import cv2
        lm = gmap.astype(np.uint8)
        bnd = cv2.dilate(lm, vec.KERNEL_3X3, borderType=cv2.BORDER_REPLICATE) != cv2.erode(lm, vec.KERNEL_3X3, borderType=cv2.BORDER_REPLICATE)
        return cv2.distanceTransform((~bnd).astype(np.uint8), cv2.DIST_L2, vec.DIST_MASK)
    vec.boundary_distance = bd


def mutate_neutral_not_identical():
    def dp(params):
        out = ORIG["depth_params"](params)
        if out["min_dist_low_px"] == 0:
            out["min_dist_low_px"] = 2.0
        return out
    vec.depth_params = dp


def mutate_angle_not_checked():                        # batas jalur tak terbatas (sudut / panjang tidak diperiksa)
    vec.OCC_REACH_MIN_ANGLE_DEG = 0.5


def mutate_frame_order_changes_result():               # keadaan lintas pemanggilan
    calls = [0]

    def occ(gmap, depth, names, params, t_high, t_low):
        calls[0] += 1
        out = ORIG["occlusion_strokes"](gmap, depth, names, params, t_high, t_low)
        return ([], out[1]) if calls[0] > 40 else out
    vec.occlusion_strokes = occ


def mutate_new_params_not_in_hash():
    def vp(cfg):
        return {k: v for k, v in ORIG["vectorize_params"](cfg).items() if k not in ("depth_lines.min_dist_low_px",
                                                                                      "depth_lines.exclude_groups")}
    vec.vectorize_params = vp


def mutate_clip_stats_depends_on_d_low():
    def csi(params, clip, stable):
        out = ORIG["clip_stats_inputs"](params, clip, stable)
        out["d_low"] = params["depth_lines.min_dist_low_px"]
        return out
    vec.clip_stats_inputs = csi


MUTATIONS = {"benih tidak disyaratkan lolos L": mutate_component_without_L_gate,
             "komponen di zona D tanpa benih dibuat": mutate_zone_component_without_seed_kept,
             "D_low diabaikan": mutate_d_low_ignored, "ekstensi tidak terhubung ke benih": mutate_extension_not_connected_to_seed,
             "exclude_groups menghapus group_boundary": mutate_exclude_removes_group_boundary,
             "exclude_groups diabaikan": mutate_exclude_ignored, "tepi frame bukan batas": mutate_frame_edge_not_boundary,
             "default netral tidak identik": mutate_neutral_not_identical, "sudut / sejajar tidak diperiksa": mutate_angle_not_checked,
             "urutan frame / pemanggilan mengubah hasil": mutate_frame_order_changes_result,
             "parameter baru tidak di hash": mutate_new_params_not_in_hash,
             "clip_stats bergantung D_low": mutate_clip_stats_depends_on_d_low}


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else None
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if which is None:
        out = {}
        for name in MUTATIONS:
            p = subprocess.run([sys.executable, __file__, name], cwd=REPO, capture_output=True, text=True, encoding="utf-8", errors="replace")
            tail = [x for x in p.stdout.splitlines() if " failed" in x or " passed" in x][-1:]
            failed = [x for x in p.stdout.splitlines() if x.startswith("FAILED") or ".py:" in x][:2]
            out[name] = {"rc": p.returncode, "ringkas": tail, "test_gagal": p.returncode != 0, "contoh": failed}
            print(name, out[name], flush=True)
        (REPO / "work" / "t305b").mkdir(parents=True, exist_ok=True)
        (REPO / "work" / "t305b" / "mutations.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"{sum(v['test_gagal'] for v in out.values())}/{len(out)} mutasi membuat test gagal")
        return
    MUTATIONS[which]()
    sys.exit(int(pytest.main([*TESTS, "-q", "-p", "no:cacheprovider", "--tb=line", "-x"])))


if __name__ == "__main__":
    main()
