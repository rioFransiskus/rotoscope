"""T-609 (alat, F3; CPU, BACA-SAJA terhadap produksi): bukti data nyata bahwa `isolate.neg_points_warn_max` tidak masuk kunci basi /
hash stage mana pun (keputusan Rio T-609 no. 1).

    python scripts/t609_kunci.py    -> work/t609/kunci_bukti.json

Untuk SETIAP klip produksi (work/clips/*) dan tiga nilai key (default 10, 0, 1000): nilai yang dipakai stage untuk memutuskan basi
dihitung ulang dari config lalu dibandingkan dengan nilai yang TERSIMPAN di manifest produksi:
  [1b] isolate/manifest.json key_sha256 = canonical_sha256(build_key(cfg, sha256 points.json, sidik piksel semua frame));
       + run_isolate(dry_run=True) in-process → harus "up-to-date" (dry-run tidak menulis / menghapus apa pun);
  [2]  qc_report.json qc_hash = section_hash(cfg, "qc");
  [3]  stable/manifest.json stabilize_hash / groups_hash = section_hash; key `isolate`.target_mask_hash = section_hash(target_mask);
  [4]  contours/manifest.json vectorize_hash = params_hash(vectorize_params(cfg));
  [5]  strokes/manifest.json style_hash = params_hash(style_params(style)) (style dari kunci `style` config);
  [6]  out/<klip>.export.json export_hash = export_hash(cfg).
Tidak ada yang ditulis selain work/t609/kunci_bukti.json. Tanpa torch / GPU.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rotoscope import export as ex  # noqa: E402
from rotoscope import isolate as iso  # noqa: E402
from rotoscope import stylize as sty  # noqa: E402
from rotoscope import vectorize as vec  # noqa: E402
from rotoscope.config import load_pipeline, load_style, resolve_style, section_hash  # noqa: E402

CLIPS = ROOT / "work" / "clips"
OUT_DIR = ROOT / "out"
DEFAULT_CFG = ROOT / "configs" / "default.yaml"
VALUES = (10, 0, 1000)
KEY = "isolate.neg_points_warn_max"


def jread(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def stored_vs(stored, computed: dict[int, str]) -> dict:
    return {"tersimpan": stored, "sama_semua": stored is not None and all(v == stored for v in computed.values()),
            "dihitung_unik": len(set(computed.values()))}


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    cfgs = {v: load_pipeline(DEFAULT_CFG, overrides={KEY: v}) for v in VALUES}
    assert all(c.isolate.neg_points_warn_max == v for v, c in cfgs.items())
    res: dict = {"nilai": list(VALUES), "klip": {}}
    ok_all = True
    for clip in sorted(p for p in CLIPS.iterdir() if p.is_dir()):
        r: dict = {}
        qc = jread(clip / "qc_report.json")
        if qc and "qc_hash" in qc:
            r["[2] qc_hash"] = stored_vs(qc["qc_hash"], {v: section_hash(c, "qc") for v, c in cfgs.items()})
        st = jread(clip / "stable" / "manifest.json")
        if st:
            for s in ("stabilize", "groups"):
                r[f"[3] {s}_hash"] = stored_vs(st.get(f"{s}_hash"), {v: section_hash(c, s) for v, c in cfgs.items()})
            if isinstance(st.get("isolate"), dict):
                r["[3] isolate.target_mask_hash"] = stored_vs(st["isolate"].get("target_mask_hash"),
                                                              {v: section_hash(c, "target_mask") for v, c in cfgs.items()})
        ct = jread(clip / "contours" / "manifest.json")
        if ct:
            r["[4] vectorize_hash"] = stored_vs(ct.get("vectorize_hash"),
                                                {v: vec.params_hash(vec.vectorize_params(c)) for v, c in cfgs.items()})
        sk = jread(clip / "strokes" / "manifest.json")
        if sk:
            r["[5] style_hash"] = stored_vs(sk.get("style_hash"), {
                v: sty.params_hash(sty.style_params(load_style(resolve_style(c.style)))[0]) for v, c in cfgs.items()})
        em = jread(OUT_DIR / f"{clip.name}.export.json")
        if em:
            r["[6] export_hash"] = stored_vs(em.get("export_hash"), {v: ex.export_hash(c) for v, c in cfgs.items()})
        im = jread(clip / "isolate" / "manifest.json")
        if im:
            c0 = iso.load_clip(clip)
            frames_sha, _ = iso.frames_fingerprint(c0)
            psha = hashlib.sha256(c0.points_path.read_bytes()).hexdigest()
            r["[1b] key_sha256"] = stored_vs(im.get("key_sha256"), {
                v: iso.canonical_sha256(iso.build_key(c, psha, frames_sha)) for v, c in cfgs.items()})
            dry = {}
            for v in VALUES:
                logs: list[str] = []
                cfg = load_pipeline(DEFAULT_CFG, overrides={KEY: v, "paths.work_dir": str(clip)})
                out = iso.run_isolate(cfg, dry_run=True, log=logs.append)
                dry[str(v)] = {"status": out["status"], "peringatan": [m[:120] for m in logs if "klik kanan" in m]}
            r["[1b] run_isolate --dry-run"] = dry
            ok_all &= all(d["status"] == "up-to-date" for d in dry.values())
        ok_all &= all(x["sama_semua"] for k, x in r.items() if isinstance(x, dict) and "sama_semua" in x)
        res["klip"][clip.name] = r
        flags = {k: (x["sama_semua"] if "sama_semua" in x else {vv: d["status"] for vv, d in x.items()}) for k, x in r.items()}
        print(f"{clip.name}: {flags}")
    res["semua_sama_dan_up_to_date"] = ok_all
    p = ROOT / "work" / "t609" / "kunci_bukti.json"
    p.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"semua nilai tersimpan = dihitung untuk {list(VALUES)} dan [1b] up-to-date: {ok_all} -> {p.relative_to(ROOT).as_posix()}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
