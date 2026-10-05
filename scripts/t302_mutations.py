"""T-302 mutation check: monkeypatch DARI LUAR berkas test (kode produksi tidak diubah di disk). Tiap mutasi harus membuat
tests/test_stabilize_temporal.py GAGAL. Pakai: python scripts/t302_mutations.py  → work/t302/mutations.json"""

from __future__ import annotations

import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))
sys.path.insert(0, str(REPO / "src"))

from rotoscope import stabilize as stb  # noqa: E402

ORIG = {"weights": stb.TemporalPlan.weights, "window": stb.TemporalPlan.window, "needed": stb.TemporalPlan.needed,
        "norm": stb.normalize_depth_f32, "plan": stb.build_plan, "groups": stb.FrameCache.groups}


def mutate_asymmetric_kernel():
    """Kernel satu arah (hanya frame t dan sebelumnya) → lag."""
    def weights(self, t, radius):
        return [(k, w) for k, w in ORIG["weights"](self, t, radius) if k <= t] or [(t, 1.0)]
    stb.TemporalPlan.weights = weights


def mutate_qc_ignored():
    stb.qc_fail_flags = lambda clip: {n: False for n in clip.names}


def mutate_no_weight_normalisation():
    """Bobot tidak dinormalisasi: p = Σ w·S (tanpa ÷ Σ w)."""
    def groups(self, k):
        if k not in self._groups:
            w = self.plan.weights(k, self.plan.radius)
            sums, tie, _ = self.raw(k)
            if len(w) == 1:
                p = sums
            else:
                acc = np.zeros(sums.shape, np.float32)
                for j, wt in w:
                    acc += np.float32(wt) * self.raw(j)[0]
                p = acc
                if self.plan.boil > 0:
                    p = np.float32(1.0 - self.plan.boil) * p + np.float32(self.plan.boil) * sums
            st = self.cfg.stabilize
            self._groups[k] = stb.clean_groups(p, tie, st.island_min_px, st.mode_k)
        return self._groups[k]
    stb.FrameCache.groups = groups

    def final_depth(self, t):
        w = self.plan.weights(t, self.plan.depth_radius)
        nd, info = self.depth_f32(t)
        if len(w) > 1:
            nd = sum(np.float32(wt) * self.depth_f32(j)[0] for j, wt in w)
        return stb.finish_depth(nd, info)
    stb.FrameCache.final_depth = final_depth


def mutate_cut_guard_off():
    stb.detect_cuts = lambda scores, cut_diff: [False] * len(scores)


def mutate_window_off_by_one():
    def window(self, t, radius):
        lo, hi = ORIG["window"](self, t, radius)
        return lo, min(hi + 1, self.n - 1)
    stb.TemporalPlan.window = window


def mutate_normalize_swapped():
    def norm(d, fg, eps, iqr_min, method=stb.NORMALIZE_A):
        return ORIG["norm"](d, fg, eps, iqr_min, stb.NORMALIZE_B if method == stb.NORMALIZE_A else stb.NORMALIZE_A)
    stb.normalize_depth_f32 = norm


def mutate_boil_ignored():
    def plan(cfg, clip, log=print, **kw):
        return dataclasses.replace(ORIG["plan"](cfg, clip, log, **kw), boil=0.0)
    stb.build_plan = plan


def mutate_limit_window_truncated():
    """--limit menulis frame walau jendela input belum lengkap."""
    stb.TemporalPlan.needed = lambda self, t: (t, t)


def mutate_cut_ignored_in_window():
    """Potongan cut tidak berlaku di jendela (shot = seluruh klip)."""
    def window(self, t, radius):
        return max(t - radius, 0), min(t + radius, self.n - 1)
    stb.TemporalPlan.window = window


MUTATIONS = {"kernel asimetris (lag)": mutate_asymmetric_kernel, "bobot QC diabaikan": mutate_qc_ignored,
             "normalisasi bobot dimatikan": mutate_no_weight_normalisation, "pengaman cut mati (deteksi)": mutate_cut_guard_off,
             "jendela off-by-one": mutate_window_off_by_one, "varian normalisasi tertukar": mutate_normalize_swapped,
             "boil_preserve diabaikan": mutate_boil_ignored, "--limit menulis jendela terpotong": mutate_limit_window_truncated,
             "cut tidak memotong jendela": mutate_cut_ignored_in_window}


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else None
    if which is None:                       # induk: satu proses per mutasi (modul bersih)
        import subprocess
        out = {}
        for name in MUTATIONS:
            t0 = time.perf_counter()
            p = subprocess.run([sys.executable, __file__, name], cwd=REPO, capture_output=True, text=True,
                               encoding="utf-8", errors="replace")
            tail = [x for x in p.stdout.splitlines() if " failed" in x or " passed" in x][-1:]
            out[name] = {"rc": p.returncode, "ringkas": tail, "test_gagal": p.returncode != 0,
                         "detik": round(time.perf_counter() - t0, 1)}
            print(name, out[name], flush=True)
        (REPO / "work" / "t302").mkdir(parents=True, exist_ok=True)
        (REPO / "work" / "t302" / "mutations.json").write_text(json.dumps(out, indent=1, ensure_ascii=False),
                                                               encoding="utf-8")
        return
    MUTATIONS[which]()
    rc = pytest.main(["tests/test_stabilize_temporal.py", "-q", "-p", "no:cacheprovider", "--tb=line"])
    sys.exit(int(rc))


if __name__ == "__main__":
    main()
