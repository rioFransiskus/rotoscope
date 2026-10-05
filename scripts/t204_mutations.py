"""T-204 mutation check: monkeypatch DARI LUAR berkas test (kode produksi tidak diubah di disk). Tiap mutasi harus membuat
tests/test_preview.py GAGAL. Pakai: python scripts/t204_mutations.py  → work/t204/mutations.json"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))
sys.path.insert(0, str(REPO / "src"))

from rotoscope import cli  # noqa: E402
from rotoscope import export as ex  # noqa: E402

ORIG = {"check": cli.check_gpu_inputs, "args": cli.preview_args, "resolve": ex.resolve_filename,
        "window": cli.preview_window, "stages": cli.PREVIEW_STAGES}


def mutate_check_off():
    cli.check_gpu_inputs = lambda *a, **k: None


def mutate_offset_wrong():
    def args(stage, ctx, k, n):
        return ORIG["args"](stage, ctx, k + 1, n)
    cli.preview_args = args


def mutate_writes_main_mp4():
    ex.resolve_filename = lambda template, source_path, limit=None, start=None: ORIG["resolve"](template, source_path)


def mutate_gpu_subprocess():
    cli.PREVIEW_STAGES = ("ingest", "segment", "depth", "stabilize", "vectorize", "stylize", "export")


def mutate_kn_unchecked():
    cli.preview_window = lambda a, work_dir: (a.start or 0, a.preview)


MUTATIONS = {"pengecekan seg/depth dimatikan": mutate_check_off, "offset K salah": mutate_offset_wrong,
             "preview menulis ke MP4 utama": mutate_writes_main_mp4,
             "subprocess GPU dipanggil": mutate_gpu_subprocess, "K+N tidak dicek": mutate_kn_unchecked}


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else None
    if which is None:                       # induk: satu proses per mutasi (modul bersih)
        import subprocess
        out = {}
        for name in MUTATIONS:
            p = subprocess.run([sys.executable, __file__, name], cwd=REPO, capture_output=True, text=True,
                               encoding="utf-8", errors="replace")
            tail = [x for x in p.stdout.splitlines() if " failed" in x or " passed" in x][-1:]
            out[name] = {"rc": p.returncode, "ringkas": tail, "test_gagal": p.returncode != 0}
            print(name, out[name], flush=True)
        (REPO / "work" / "t204").mkdir(parents=True, exist_ok=True)
        (REPO / "work" / "t204" / "mutations.json").write_text(json.dumps(out, indent=1, ensure_ascii=False),
                                                              encoding="utf-8")
        return
    MUTATIONS[which]()
    rc = pytest.main(["tests/test_preview.py", "-q", "-p", "no:cacheprovider", "--tb=line"])
    sys.exit(int(rc))


if __name__ == "__main__":
    main()
