"""T-609 tanggapan Rio no. 3 (alat, F3; CPU): kenapa test_34 / test_47 / test_m5 sempat gagal sesaat di putaran mutasi pertama?

    python scripts/t609_flaky.py beku         # SEMUA created_utc dibekukan ke SATU nilai (kasus terburuk "detik yang sama")
    python scripts/t609_flaky.py kunci 0      # kunci berkas Windows SESAAT disimulasikan: os.replace -> PermissionError selama
    python scripts/t609_flaky.py kunci 0.2    #   0,15 s pada tiap tujuan ke-7; jeda retry stage_common = 0 (nilai fixture test
                                              #   test_isolate dkk.) atau 0,2 (nilai asli, 5 percobaan)
    -> ringkasan per mode di work/t609/flaky_<mode>.json

Hipotesis 1 (backlog "Basi hilir berbasis created_utc beresolusi detik"): bila mode `beku` tetap lolos, ketiga test tidak bergantung
pada created_utc yang berbeda. Hipotesis 2: kunci berkas sesaat (mis. antivirus / indexer membuka berkas baru) + jeda retry 0 di
fixture. Kode produksi TIDAK diubah: patch hanya di memori proses ini (fungsi `utc_now` / `os.replace` / `REPLACE_DELAY_S`).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"                 # pengaman: tanpa GPU

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))
sys.path.insert(0, str(REPO / "src"))
os.chdir(REPO)

import pytest  # noqa: E402

import rotoscope.cli  # noqa: E402,F401  (memuat semua stage)
import rotoscope.mask_preview  # noqa: E402,F401
from rotoscope import stage_common  # noqa: E402

TESTS = ["tests/test_isolate.py::test_34_real_cpu_chain_byte_identical_with_points_and_isolate",
         "tests/test_isolate.py::test_47_real_cpu_chain_unchanged_when_neg_threshold_changes",
         "tests/test_mask_preview.py::test_m5_folder_replaced_whole_and_interrupt_keeps_old"]
FIXED = "2026-10-29T00:00:00+00:00"
LOCK_S, EVERY = 0.15, 7
OUT = REPO / "work" / "t609"


class Probe:
    """Plugin pytest: jeda retry dipasang SESUDAH fixture (fixture test_isolate memasang REPLACE_DELAY_S = 0); hasil per test."""

    def __init__(self, delay: float | None):
        self.delay, self.results = delay, {}

    @pytest.hookimpl(tryfirst=True)
    def pytest_runtest_call(self, item):
        if self.delay is not None:
            stage_common.REPLACE_DELAY_S = self.delay

    def pytest_runtest_logreport(self, report):
        if report.when == "call" or report.outcome == "failed":
            self.results[report.nodeid.split("::")[-1]] = {
                "hasil": report.outcome, "galat": str(report.longrepr).strip().splitlines()[-1][:300] if report.failed else None}


def main(argv: list[str]) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if not argv or argv[0] not in ("beku", "kunci") or (argv[0] == "kunci") != (len(argv) == 2):
        print(__doc__)
        return 2
    mode, delay = argv[0], (float(argv[1]) if len(argv) == 2 else None)
    stats = {"replace": 0, "tujuan_dikunci": 0, "ditolak": 0}
    if mode == "beku":
        patched = []
        for name, mod in list(sys.modules.items()):
            if name.startswith("rotoscope"):
                for attr in ("utc_now", "_utc_now"):
                    if hasattr(mod, attr):
                        setattr(mod, attr, lambda: FIXED)
                        patched.append(f"{name}.{attr}")
        stats = {"dibekukan": sorted(patched)}
    else:
        real_replace = os.replace
        first: dict[str, float] = {}
        order: dict[str, int] = {}

        def locked_replace(src, dst, *a, **k):
            stats["replace"] += 1
            key = str(dst)
            if key not in first:
                first[key], order[key] = time.perf_counter(), len(order)
                stats["tujuan_dikunci"] += order[key] % EVERY == EVERY - 1
            if order[key] % EVERY == EVERY - 1 and time.perf_counter() - first[key] < LOCK_S:
                stats["ditolak"] += 1
                raise PermissionError(13, "simulasi: berkas sedang dibuka proses lain (mis. antivirus)", key)
            return real_replace(src, dst, *a, **k)
        os.replace = locked_replace
    probe = Probe(delay)
    rc = int(pytest.main([*TESTS, "-q", "-p", "no:cacheprovider", "--tb=line"], plugins=[probe]))
    res = {"mode": mode, "jeda_retry_s": delay, "kunci_s": LOCK_S if mode == "kunci" else None,
           "setiap_tujuan_ke": EVERY if mode == "kunci" else None, "rc": rc, "test": probe.results, "statistik": stats}
    label = mode if delay is None else f"{mode}{str(delay).replace('.', '')}"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"flaky_{label}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"mode {mode}, jeda {delay}: rc {rc}; {probe.results}; {stats if mode == 'kunci' else len(stats['dibekukan'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
