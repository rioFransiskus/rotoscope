"""T-605 Tahap 2 (alat sekali pakai, F3): bukti `run` pada klip NYATA tanpa titik tidak membuat subprocess [1b] dan tidak
menyentuh berkas apa pun.

    python scripts/t605_identik.py

`cli.main(["run", samples/<klip>.mp4])` in-process untuk 6 klip, dengan SEMUA stage dipalsukan pencatat (`cli._popen` untuk stage
GPU, `cli.CPU_MAINS` untuk stage CPU — tidak ada stage yang benar-benar berjalan, tanpa GPU / torch / ffmpeg ingest). Yang ASLI hanya
logika cli: pre-flight (baca-saja) + keputusan [1b] (`isolate.points_present` → subprocess / `skip_without_points`). Sebelum dan
sesudah: daftar SEMUA berkas + folder di work/clips/<klip>/ dengan ukuran + mtime_ns. Klip tanpa titik (test, test_short, Klip2, klip4)
harus: tanpa panggilan [1b], daftar sama persis. klip3 / klip5 (ada titik): [1b] tercatat sebagai subprocess (TIDAK dijalankan).
Hasil → work/t605/identik.json. Tidak menulis apa pun di work/clips/ atau out/.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rotoscope import cli  # noqa: E402

CLIPS = ("test", "test_short", "Klip2", "klip4", "klip3", "klip5")
OUT = ROOT / "work" / "t605" / "identik.json"


class Proc:
    def __init__(self, rc=0):
        self.rc = rc

    def wait(self, timeout=None):
        return self.rc

    def poll(self):
        return self.rc

    def terminate(self):
        pass


def snapshot(d: Path) -> dict[str, list]:
    out = {}
    for p in sorted(d.rglob("*")):
        st = p.stat()
        out[p.relative_to(d).as_posix()] = ["d" if p.is_dir() else "f", st.st_size if p.is_file() else 0, st.st_mtime_ns]
    return out


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    os.chdir(ROOT)
    calls: list[tuple[str, str, list[str]]] = []

    def popen(argv, env=None):
        calls.append(("subprocess", argv[2], list(argv[3:])))
        return Proc()

    def cpu(stage):
        def m(argv):
            calls.append(("in-process", stage, list(argv)))
            return 0
        return m

    cli._popen = popen
    cli.CPU_MAINS = {s: cpu(s) for s in ("ingest", "stabilize", "vectorize", "stylize", "export")}
    res, ok_all = {}, True
    for clip in CLIPS:
        wd = ROOT / "work" / "clips" / clip
        video = ROOT / "samples" / f"{clip}.mp4"
        has_points = (wd / "target" / "points.json").is_file()
        before = snapshot(wd)
        calls.clear()
        rc = cli.main(["run", str(video)])
        after = snapshot(wd)
        iso_calls = [c for c in calls if "isolate" in c[1]]
        same = before == after
        ok = rc == 0 and same and (bool(iso_calls) == has_points)
        ok_all &= ok
        res[clip] = {"titik": has_points, "rc": rc, "panggilan": [f"{c[0]}:{c[1]}" for c in calls],
                     "panggilan_isolate": iso_calls, "n_berkas_folder": len(before), "daftar_sama_persis": same,
                     "berubah": sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))[:10], "lolos": ok}
        print(f"\n>>> {clip}: titik {has_points}, rc {rc}, [1b] subprocess {len(iso_calls)} ({iso_calls[0][2] if iso_calls else '-'}), "
              f"{len(before)} berkas+folder, daftar sama persis {same} -> {'LOLOS' if ok else 'GAGAL'}", flush=True)
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nSEMUA LOLOS: {ok_all} -> {OUT.relative_to(ROOT).as_posix()}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
