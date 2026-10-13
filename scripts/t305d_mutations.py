"""T-305d mutation check: monkeypatch DARI LUAR berkas test (kode produksi di disk tidak diubah; mutasi hidup di salinan
dalam memori satu proses). Tiap mutasi harus membuat minimal satu test di TESTS GAGAL.
Pakai: python scripts/t305d_mutations.py  -> work/t305d/mutations.json. Satu proses per mutasi (modul bersih)."""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))
sys.path.insert(0, str(REPO / "src"))

from rotoscope import config  # noqa: E402
from rotoscope import segment as seg  # noqa: E402

TESTS = ["tests/test_segment_qc.py", "tests/test_segment.py"]
ORIG = {"big_blobs_fail": seg.big_blobs_fail, "at_least": config._at_least,
        "write_json_atomic": seg.write_json_atomic}


def make_transient(*, long_strict: bool = False, combine=max, use_before: bool = True, use_after: bool = True):
    """Salinan seg.transient_big_blobs dengan satu perubahan (dipilih lewat argumen)."""
    def is_long(s: int, e: int, min_run: int) -> bool:
        return e - s > min_run if long_strict else e - s >= min_run

    def transient(n, min_run):
        runs = seg.count_runs(n)
        out = [False] * len(n)
        prev_long, last = [], None
        for s, e, v in runs:
            prev_long.append(last)
            if is_long(s, e, min_run):
                last = v
        next_long = None
        for k in range(len(runs) - 1, -1, -1):
            s, e, v = runs[k]
            if is_long(s, e, min_run):
                next_long = v
                continue
            refs = [x for x, use in ((prev_long[k], use_before), (next_long, use_after)) if use and x is not None]
            if refs and v > combine(refs):
                out[s:e] = [True] * (e - s)
        return out
    return transient


def mutate_min_instead_of_max():
    seg.transient_big_blobs = make_transient(combine=min)


def mutate_long_run_strict():                            # run panjang = > N, bukan ≥ N
    seg.transient_big_blobs = make_transient(long_strict=True)


def mutate_ignore_after_side():
    seg.transient_big_blobs = make_transient(use_after=False)


def mutate_ignore_before_side():
    seg.transient_big_blobs = make_transient(use_before=False)


def mutate_default_back_to_legacy():                     # default kode QcConfig kembali ke perilaku lama
    names = [f.name for f in dataclasses.fields(config.QcConfig)]
    d = list(config.QcConfig.__init__.__defaults__)
    d[names.index("big_blobs_rule")] = "legacy"
    config.QcConfig.__init__.__defaults__ = tuple(d)


def mutate_report_still_fails():                         # report tetap menulis big_blobs ke fail_reasons
    def f(n, q):
        return ORIG["big_blobs_fail"](n, dataclasses.replace(q, big_blobs_rule="legacy") if q.big_blobs_rule == "report" else q)
    seg.big_blobs_fail = f


def mutate_legacy_ge():                                  # legacy: >= max_big_blobs (bukan >)
    def f(n, q):
        return [x >= q.max_big_blobs for x in n] if q.big_blobs_rule == "legacy" else ORIG["big_blobs_fail"](n, q)
    seg.big_blobs_fail = f


def mutate_min_run_not_validated():
    def al(key, v, lo, strict=False):
        return None if key == "qc.transient_min_run" else ORIG["at_least"](key, v, lo, strict)
    config._at_least = al


def mutate_qc_version_not_written():
    def w(path, obj, *a, **kw):
        if isinstance(obj, dict) and obj.get("stage") == "segment_qc":
            obj = {k: v for k, v in obj.items() if k != "qc_version"}
        return ORIG["write_json_atomic"](path, obj, *a, **kw)
    seg.write_json_atomic = w


CONTROL = "kontrol (tanpa mutasi; harus LOLOS)"
MUTATIONS = {CONTROL: lambda: None,
             "max diganti min": mutate_min_instead_of_max,
             "run panjang memakai > N": mutate_long_run_strict,
             "sisi sesudah diabaikan": mutate_ignore_after_side,
             "sisi sebelum diabaikan": mutate_ignore_before_side,
             "default kembali ke legacy": mutate_default_back_to_legacy,
             "report tetap menulis fail_reasons": mutate_report_still_fails,
             "legacy memakai >= max_big_blobs": mutate_legacy_ge,
             "validasi transient_min_run dihapus": mutate_min_run_not_validated,
             "qc_version tidak ditulis ke disk": mutate_qc_version_not_written}


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else None
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if which is None:
        out = {}
        for name in MUTATIONS:
            p = subprocess.run([sys.executable, __file__, name], cwd=REPO, capture_output=True, text=True,
                               encoding="utf-8", errors="replace")
            tail = [x for x in p.stdout.splitlines() if " failed" in x or " passed" in x][-1:]
            failed = [x for x in p.stdout.splitlines() if x.startswith("FAILED") or ".py:" in x][:2]
            out[name] = {"rc": p.returncode, "ringkas": tail, "test_gagal": p.returncode != 0, "contoh": failed}
            print(name, out[name], flush=True)
        (REPO / "work" / "t305d").mkdir(parents=True, exist_ok=True)
        (REPO / "work" / "t305d" / "mutations.json").write_text(json.dumps(out, indent=1, ensure_ascii=False),
                                                              encoding="utf-8")
        muts = {k: v for k, v in out.items() if k != CONTROL}
        print(f"kontrol lolos: {not out[CONTROL]['test_gagal']}; "
              f"{sum(v['test_gagal'] for v in muts.values())}/{len(muts)} mutasi membuat test gagal")
        return
    MUTATIONS[which]()
    sys.exit(int(pytest.main([*TESTS, "-q", "-p", "no:cacheprovider", "--tb=line"])))


if __name__ == "__main__":
    main()
