"""T-305c mutation check: monkeypatch DARI LUAR berkas test (kode produksi di disk tidak diubah; mutasi hidup di salinan
dalam memori satu proses). Tiap mutasi harus membuat tests/test_vectorize_floor.py GAGAL.
Pakai: python scripts/t305c_mutations.py  -> work/t305c/mutations.json. Satu proses per mutasi (modul bersih)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))
sys.path.insert(0, str(REPO / "src"))

from rotoscope import config  # noqa: E402
from rotoscope import vectorize as vec  # noqa: E402

TESTS = ["tests/test_vectorize_floor.py"]
ORIG = {"compute_clip_stats": vec.compute_clip_stats, "params_hash": vec.params_hash, "at_least": config._at_least}
KEY = "depth_lines.hi_floor"


def mutate_floor_also_on_t_low():                       # lantai ikut menaikkan T_low
    def cs(clip, params, inputs, names):
        out = ORIG["compute_clip_stats"](clip, params, inputs, names)
        if params[KEY] > 0 and out["t_low"] is not None:
            out["t_low"] = max(out["t_low"], float(params[KEY]))
        return out
    vec.compute_clip_stats = cs


def mutate_min_instead_of_max():
    vec.effective_t_high = lambda pct, floor: min(pct, float(floor)) if floor > 0 else pct


def mutate_floor_ignored():
    vec.effective_t_high = lambda pct, floor: pct


def mutate_floor_not_in_hash():
    vec.params_hash = lambda params: ORIG["params_hash"]({k: v for k, v in params.items() if k != KEY})


def mutate_negative_not_validated():
    def al(key, v, lo, strict=False):
        return None if key.endswith("hi_floor") else ORIG["at_least"](key, v, lo, strict)
    config._at_least = al


def mutate_t_high_pct_not_recorded():                   # hanya nilai yang dipakai dicatat; persentil hilang
    def cs(clip, params, inputs, names):
        out = ORIG["compute_clip_stats"](clip, params, inputs, names)
        out["t_high_pct"] = out["t_high"]
        return out
    vec.compute_clip_stats = cs


MUTATIONS = {"lantai juga dikenakan ke T_low": mutate_floor_also_on_t_low,
             "min alih-alih max": mutate_min_instead_of_max,
             "lantai diabaikan": mutate_floor_ignored,
             "hi_floor tidak masuk hash": mutate_floor_not_in_hash,
             "validasi nilai negatif dihapus": mutate_negative_not_validated,
             "t_high_pct mengikuti nilai yang dipakai": mutate_t_high_pct_not_recorded}


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
        (REPO / "work" / "t305c").mkdir(parents=True, exist_ok=True)
        (REPO / "work" / "t305c" / "mutations.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"{sum(v['test_gagal'] for v in out.values())}/{len(out)} mutasi membuat test gagal")
        return
    MUTATIONS[which]()
    sys.exit(int(pytest.main([*TESTS, "-q", "-p", "no:cacheprovider", "--tb=line", "-x"])))


if __name__ == "__main__":
    main()
