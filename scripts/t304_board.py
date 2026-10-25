"""T-304 papan buta (alat, F3): kunci acak, tata letak, komposisi ffmpeg, dan PANDUAN.md.

    python scripts/t304_board.py kunci      -> scripts/t304_kunci.json (ter-track; JANGAN ditampilkan di laporan / papan / PANDUAN)
    python scripts/t304_board.py layout     -> tata letak tiap papan (cetak saja)
    python scripts/t304_board.py compose    -> work/t304/nilai/<papan>_normal.mp4 + <papan>_0.5x.mp4 (+ nilai/meta.json)
    python scripts/t304_board.py pandu      -> work/t304/nilai/PANDUAN.md (tanpa kunci: detik fokus, daftar frame B-sehat, template)

Papan (lihat docs/04 "Keputusan T-304 lanjutan"):
  test, klip4, klip3      : 4 varian (b0, b0.3, b0.6, b1.0), huruf A–D acak per klip, seed 304 + indeks klip (0, 1, 2)
  klip5_cek               : 4 varian, huruf acak sendiri, seed 314
  klip5_bsehat            : b0.3 vs bs (B-sehat), huruf A/B acak, seed 324
Huruf → varian hanya di scripts/t304_kunci.json. Output tidak memuat nama varian.
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import temporal_metrics as tm  # noqa: E402

SCR_OUT = ROOT / "work" / "t304_scratch" / "out"
SCR = ROOT / "work" / "t304_scratch"
NILAI = ROOT / "work" / "t304" / "nilai"
KUNCI = ROOT / "scripts" / "t304_kunci.json"
RAW = ROOT / "work" / "t304" / "raw"
FONT_DIR = Path("C:/Windows/Fonts")          # ffmpeg (Windows) tidak menerima path drive di fontfile=; font dicari dari cwd ini
FONT = "arialbd.ttf"
FPS = 24
W_SCREEN, H_SCREEN = 1920, 1080
VARS4 = ["b0", "b0.3", "b0.6", "b1.0"]
BOARDS = [
    # (nama papan, klip, varian, seed)
    ("test", "test", VARS4, 304),
    ("klip4", "klip4", VARS4, 305),
    ("klip3", "klip3", VARS4, 306),
    ("klip5_cek", "klip5", VARS4, 314),
    ("klip5_bsehat", "klip5", ["b0.3", "bs"], 324),
]
LETTERS = "ABCD"


def jload(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def cmd_kunci() -> None:
    out = {"versi": 1, "dibuat": date.today().isoformat(),
           "aturan": "huruf = urutan acak (random.Random(seed).shuffle) atas daftar varian papan; JANGAN ditampilkan di laporan, "
                     "papan, atau PANDUAN",
           "papan": {}}
    for name, clip, vs, seed in BOARDS:
        order = list(vs)
        random.Random(seed).shuffle(order)
        out["papan"][name] = {"klip": clip, "seed": seed, "huruf_ke_varian": {LETTERS[i]: v for i, v in enumerate(order)}}
    KUNCI.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"  -> {KUNCI.relative_to(ROOT).as_posix()}")


def frame_size(clip: str) -> tuple[int, int]:
    m = jload(SCR / "b0" / clip / "stable" / "manifest.json")
    fs = m["frame_size"]
    if isinstance(fs, dict):
        return int(fs["width"]), int(fs["height"])
    return int(fs[0]), int(fs[1])


def layout(n: int, w: int, h: int) -> dict:
    """Portrait 4 panel = 1 × 4; landscape / persegi 4 panel = 2 × 2; 2 panel = 1 × 2 (berskala sama, muat 1920 × 1080)."""
    if n == 4:
        cols, rows = (4, 1) if h > w else (2, 2)
    else:
        cols, rows = 2, 1
    cell_w, cell_h = W_SCREEN // cols, H_SCREEN // rows
    s = min(cell_w / w, cell_h / h)
    pw, ph = int(w * s) // 2 * 2, int(h * s) // 2 * 2
    return {"cols": cols, "rows": rows, "pw": pw, "ph": ph, "canvas": [cols * pw, rows * ph], "orientasi": "portrait" if h > w else "landscape"}


def cmd_layout() -> None:
    for name, clip, vs, _ in BOARDS:
        w, h = frame_size(clip)
        print(name, (w, h), layout(len(vs), w, h))


def filt_panel(i: int, letter: str, pw: int, ph: int) -> str:
    lab = (f"drawtext=fontfile={FONT}:text='{letter}':fontsize=72:fontcolor=white:borderw=4:bordercolor=black:x=16:y=8")
    info = (f"drawtext=fontfile={FONT}:text='frame %{{frame_num}}  %{{pts\\:flt\\:2}} s':fontsize=30:fontcolor=white:"
            f"borderw=3:bordercolor=black:x=16:y=92")
    return f"[{i}:v]scale={pw}:{ph},setsar=1,{lab},{info}[p{i}]"


def compose_cmd(name: str, clip: str, vs: list[str], letters: dict[str, str], speed: float, out: Path) -> list[str]:
    w, h = frame_size(clip)
    lay = layout(len(vs), w, h)
    inputs, parts, pos = [], [], []
    keys = sorted(letters)
    for i, letter in enumerate(keys):
        var = letters[letter]
        inputs += ["-i", str(SCR_OUT / var / clip / f"{clip}.mp4")]
        parts.append(filt_panel(i, letter, lay["pw"], lay["ph"]))
        pos.append(f"{(i % lay['cols']) * lay['pw']}_{(i // lay['cols']) * lay['ph']}")
    chain = ";".join(parts) + ";" + "".join(f"[p{i}]" for i in range(len(keys))) + \
        f"xstack=inputs={len(keys)}:layout={'|'.join(pos)}[v]"
    if speed == 1.0:
        chain += ";[v]null[o]"
    else:
        chain += f";[v]setpts={1 / speed:.1f}*PTS[o]"
    return ["ffmpeg", "-y", "-loglevel", "error", *inputs, "-filter_complex", chain, "-map", "[o]", "-an",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-r", str(FPS),
            "-movflags", "+faststart", str(out)]


def cmd_compose() -> None:
    kunci = jload(KUNCI)["papan"]
    NILAI.mkdir(parents=True, exist_ok=True)
    meta = []
    for name, clip, vs, _ in BOARDS:
        letters = kunci[name]["huruf_ke_varian"]
        for speed, tag in ((1.0, "normal"), (0.5, "0.5x")):
            out = NILAI / f"{name}_{tag}.mp4"
            cmd = compose_cmd(name, clip, vs, letters, speed, out)
            print(f"  {out.name} ...", flush=True)
            r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=FONT_DIR)
            if r.returncode != 0:
                raise SystemExit(f"ffmpeg gagal ({out.name}): {r.stderr[-800:]}")
            meta.append({"file": out.name, "papan": name, "speed": speed, "bytes": out.stat().st_size,
                         "layout": layout(len(vs), *frame_size(clip))})
    (NILAI / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    print("  -> work/t304/nilai/meta.json")


def windows_for_board(clip: str, raw: np.ndarray, qc_fails: set[int], wins: dict) -> list[dict]:
    """Detik fokus: statis / cepat / QC dari Tahap 1; klip tanpa statis → pengganti = jendela 21 frame gerak terendah
    (rata-rata kecepatan centroid minimum, tanpa frame gagal QC)."""
    n = len(raw)
    out = []
    for a, b in wins["static"]:
        out.append({"jenis": "statis", "awal": a, "akhir": b})
    if not wins["static"]:
        speed = tm.centroid_speed(raw)
        best = None
        for s in range(0, n - 21 + 1):
            if any(f in qc_fails for f in range(s, s + 21)):
                continue
            v = float(speed[s:s + 21].mean())
            if best is None or v < best[0]:
                best = (v, s)
        if best is not None:
            s = best[1]
            out.append({"jenis": "pengganti statis (gerak terendah)", "awal": s, "akhir": s + 20, "kecepatan_mean_px": round(best[0], 3)})
    for a, b in wins["fast"]:
        out.append({"jenis": "cepat", "awal": a, "akhir": b})
    for a, b in wins["qc"]:
        out.append({"jenis": "QC (gagal)", "awal": a, "akhir": b})
    for o in out:
        o["detik_awal"] = round(o["awal"] / FPS, 2)
        o["detik_akhir"] = round((o["akhir"] + 1) / FPS, 2)
        o["frame"] = f"f{o['awal']}–f{o['akhir']}"
    return out


def cmd_pandu() -> None:
    metrics = jload(ROOT / "work" / "t304" / "tahap1_metrics.json")
    lines = ["# PANDUAN penilaian T-304 — `boil_preserve`", "",
             "Tanggal: " + date.today().isoformat() + ". Papan buta: huruf (A–D) tidak punya arti di luar papan. "
             "Jangan mencari arti huruf. Jawab hanya dengan huruf dan detik.", "",
             "## Cara menonton", "",
             "1. Buka tiap papan di `work/t304/nilai/`, tonton versi **normal** dulu, lalu **0,5×** (lambat: frame bergerak separuh kecepatan).",
             "2. Setiap panel: huruf besar kiri atas; baris kecil = nomor frame dan detik video. Semua panel berskala sama.",
             "3. Tandai detik yang menentukan pilihanmu (bila ada).",
             "4. Papan: `test_normal.mp4` / `test_0.5x.mp4` (1 × 4 atau 2 × 2 sesuai bentuk video), sama untuk `klip4_*` dan `klip3_*`.",
             "5. Papan cek klip5: `klip5_cek_*` (hanya jawab: ada varian yang rusak jelas? huruf + detik / tidak ada).",
             "6. Papan B-sehat klip5: `klip5_bsehat_*` (A / B / tidak terlihat beda). `klip3` frame 121 TIDAK dinilai.", "",
             "## Detik fokus (dari analisis Tahap 1)", "",
             "Jenis: statis = derau diam terendah; cepat = gerak cepat; QC = frame gagal QC (bobot kecil); "
             "pengganti statis = jendela gerak terendah karena klip ini tidak punya jendela statis (aturan T-302).", ""]
    raw_cache = {}
    for name, clip in (("test", "test"), ("klip4", "klip4"), ("klip3", "klip3")):
        m = metrics[clip]
        raw = np.load(RAW / f"{clip}.npy")
        raw_cache[clip] = raw
        fails = set(m["windows"]["qc_fail_frames"])
        lines += [f"### {name}", "", "| jenis | frame | detik |", "|---|---|---|"]
        for w in windows_for_board(clip, raw, fails, m["windows"] | {"static": m["windows"]["static"]}):
            lines.append(f"| {w['jenis']} | {w['frame']} | {w['detik_awal']}–{w['detik_akhir']} |")
        lines.append("")
    lines += ["### klip5 (papan cek dan B-sehat)", "", "Frame gagal QC klip5: " +
              ", ".join(str(f) for f in metrics["klip5"]["windows"]["qc_fail_frames"]) + ".", ""]
    # Daftar B-sehat: SEMUA frame yang peta grupnya berubah (b 0,3 vs B-sehat), urut px terbesar
    from rotoscope import stabilize as stb
    a = stb.load_clip(SCR / "b0.3" / "klip5")
    b = stb.load_clip(SCR / "bs" / "klip5")
    rows = []
    for t, nm in enumerate(a.names):
        ga, gb = stb.read_groups(a.groups_path(nm)), stb.read_groups(b.groups_path(nm))
        px = int((ga != gb).sum())
        if px:
            rows.append((px, t))
    rows.sort(reverse=True)
    lines += ["## Papan B-sehat klip5 — semua frame yang peta grupnya berubah", "",
              "Urut px terbesar. Detik = frame / 24. Frame gagal QC ditandai.", "",
              "| frame | detik | px berubah | gagal QC |", "|---|---|---|---|"]
    fails5 = set(metrics["klip5"]["windows"]["qc_fail_frames"])
    for px, t in rows:
        lines.append(f"| f{t} | {round(t / FPS, 2)} | {px} | {'ya' if t in fails5 else 'tidak'} |")
    lines += ["", "## Template jawaban (salin dan isi)", "",
              "```",
              "test: urutan (1 = paling disukai) _ _ _ _ ; pasangan tidak terlihat beda: ___ ; komentar: ___",
              "klip4: urutan (1 = paling disukai) _ _ _ _ ; pasangan tidak terlihat beda: ___ ; komentar: ___",
              "klip3: urutan (1 = paling disukai) _ _ _ _ ; pasangan tidak terlihat beda: ___ ; komentar: ___",
              "klip5 cek: varian rusak jelas? huruf + detik / tidak ada",
              "B-sehat klip5: A / B / tidak terlihat beda ; detik yang menentukan: ___",
              "```", ""]
    path = NILAI / "PANDUAN.md"
    NILAI.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  -> work/t304/nilai/PANDUAN.md ({len(rows)} frame B-sehat)")


def main(argv: list[str]) -> int:
    if not argv or argv[0] not in ("kunci", "layout", "compose", "pandu"):
        print(__doc__)
        return 2
    {"kunci": cmd_kunci, "layout": cmd_layout, "compose": cmd_compose, "pandu": cmd_pandu}[argv[0]]()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
