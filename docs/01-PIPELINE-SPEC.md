# 01 — PIPELINE SPEC

Arsitektur: **Segmentation → Contour → Stylized Stroke** (Arsitektur A),
dengan **MediaPipe Pose sebagai fallback** (Arsitektur C) untuk frame yang gagal QC.

## Diagram alur

```
video.mp4
   │
   ├─[1] ingest ──────────► frames/*.png  (24 fps, resized)
   │
   ├─[2] segment ─────────► masks/*.png   (binary mask, per frame)
   │         │
   │         └─[2b] QC gagal? ──► fallback: MediaPipe Pose → synthetic mask
   │
   ├─[3] stabilize ───────► masks_smooth/*.png
   │
   ├─[4] vectorize ───────► contours/*.json  (list of point arrays)
   │
   ├─[5] stylize ─────────► strokes/*.svg + strokes/*.png
   │
   └─[6] export ──────────► out/animation.mp4 + out/svg/*.svg
```

Setiap stage menulis ke disk. **Wajib resumable** — kalau stage 4 gagal, stage 1–3
tidak perlu diulang. Ini krusial karena iterasi style akan sering (stage 5 saja).

---

## Kontrak per modul

### [1] `ingest.py`
- **In:** path video, `target_fps` (default 24), `working_width` (default 720)
- **Out:** `frames/frame_%05d.png`, plus `meta.json` (fps asli, durasi, jumlah frame)
- **Lib:** OpenCV (`cv2.VideoCapture`) atau ffmpeg via subprocess
- Resize proporsional. Jangan upscale kalau sumber lebih kecil.

### [2] `segment.py`
- **In:** `frames/*.png`
- **Out:** `masks/frame_%05d.png` (grayscale 0–255), `qc_report.json`
- ⚠️ Backend diganti ke Sapiens2 (D-009) — kontrak lengkap diperbarui setelah T-102c.
- **Model:** ⚠️ backend final ditentukan di **T-102a (D-008)**:
  - MediaPipe SelfieMulticlass 256×256 (Apache 2.0) — default kalau kualitas setara
  - rembg `u2net_human_seg` — lisensi abu-abu, eksperimen saja. Kalau dipakai: **eksplisit**
    `-m u2net_human_seg`, JANGAN pakai model default (D-003)
- Post: threshold → morphological open+close → ambil connected component terbesar

**QC check per frame** (tulis hasilnya ke `qc_report.json`):
| Metrik | Kondisi gagal |
|---|---|
| `area_ratio` | luas mask < 3% atau > 70% dari frame |
| `iou_prev` | IoU dengan mask frame sebelumnya < 0.55 |
| `component_count` | jumlah blob besar (>5% area) berubah drastis |

### [2b] `fallback_pose.py`
- Dipanggil hanya untuk frame yang gagal QC
- **Lib:** MediaPipe Pose (33 landmark)
- Bangun mask sintetik: capsule/polygon antar landmark (torso, lengan, kaki, kepala)
- Blend dengan mask asli — jangan ganti total, agar transisi tidak melompat
- Catat frame mana yang pakai fallback ke `qc_report.json`

### [3] `stabilize.py` — **stage tersulit, alokasikan waktu paling banyak**
- **In:** `masks/*.png`, `qc_report.json`
- **Out:** `masks_smooth/*.png`
- Teknik, terapkan berurutan:
  1. **Temporal EMA** pada mask (alpha ~0.6–0.8) — redam noise frame-to-frame
  2. **Optical flow warp** (Farnebäck, `cv2.calcOpticalFlowFarneback`) — warp mask frame
     sebelumnya ke frame sekarang, lalu blend. Ini yang paling efektif melawan boiling
  3. Re-threshold + morphological cleanup
- ⚠️ Jangan over-smooth. Sedikit boil = hand-drawn feel. Target: hilangkan flicker acak,
  pertahankan getaran halus. Ini kontrol `jitter`, bukan nol.

### [4] `vectorize.py`
- **In:** `masks_smooth/*.png`
- **Out:** `contours/frame_%05d.json` → `{"outer": [[x,y],...], "holes": [[...]]}`
- Langkah:
  1. `cv2.findContours` mode `RETR_EXTERNAL` (siluet luar saja)
  2. `cv2.approxPolyDP` dengan epsilon dari config → simplifikasi
  3. **Resample ke jumlah titik tetap N** (default 200) via interpolasi arc-length
  4. **Point correspondence:** rotasi urutan titik agar titik ke-0 selalu di posisi anatomis
     sama (mis. titik tertinggi). Tanpa ini, garis akan "berputar" antar frame.

### [5] `stylize.py`
- **In:** `contours/*.json`, `style.yaml`
- **Out:** `strokes/frame_%05d.svg` + raster PNG
- Render stroke dari titik kontur (lihat `02-STYLE-PARAMS.md` untuk daftar parameter)
- **Lib:** `svgwrite` untuk SVG, `Pillow` atau `pycairo` untuk raster
- Komponen render:
  - Spline smoothing (Catmull-Rom → cubic Bezier) agar garis tidak patah-patah
  - Width modulation sepanjang path (tebal di bawah/tumpuan, tipis di ujung)
  - Per-point jitter dengan **seed berbasis frame index** — reproducible
  - Tekstur: brush stamping (raster) atau multi-stroke offset (SVG)
  - Paper background layer

### [6] `export.py`
- SVG: copy `strokes/*.svg` ke `out/svg/`
- MP4: raster PNG → video via ffmpeg, `-r 24`, `-pix_fmt yuv420p`

---

## Stack & environment

```
Python 3.11.9
opencv-contrib-python  # ingest, contour, optical flow — dibawa mediapipe.
                       # JANGAN tambah opencv-python (dua paket OpenCV bentrok di modul cv2)
rembg[cpu]             # kandidat segmentasi (onnxruntime CPU) — D-008
mediapipe              # kandidat segmentasi (SelfieMulticlass) + fallback pose
numpy, scipy           # resampling, interpolasi
svgwrite               # export SVG
Pillow                 # raster render
pyyaml                 # config
ffmpeg                 # muxing (binary eksternal, gyan.dev essentials build, via subprocess)
```

Versi terkunci: `requirements.txt` (dependency langsung) dan `requirements-lock.txt` (full freeze).
⚠️ OpenCV 5.x dan mediapipe 1.x adalah versi mayor baru — jangan asumsikan API OpenCV 4.x /
mediapipe 0.10.x dari tutorial lama.

**Catatan GPU:** mulai dengan `onnxruntime` (CPU) dulu agar pipeline jalan. Upgrade ke
`onnxruntime-gpu` belakangan — butuh CUDA 12.x + cuDNN 9 yang versinya harus cocok,
dan ini sumber error setup yang sering di Windows. Jangan blokir progress karenanya.
Tidak relevan kalau T-102a memilih MediaPipe.

## Struktur repo

```
rotoscope/
├── src/rotoscope/
│   ├── ingest.py  segment.py  fallback_pose.py
│   ├── stabilize.py  vectorize.py  stylize.py  export.py
│   ├── config.py  cli.py
├── configs/
│   ├── default.yaml
│   └── styles/rough-sketch.yaml
├── models/        # model .tflite, gitignored
├── scripts/       # smoke_test.py, script A/B sekali pakai
├── work/          # intermediate, gitignored
├── out/
└── tests/
```

## Urutan build (jangan lompat)

| Phase | Isi | Selesai kalau |
|---|---|---|
| 1 | ingest + segment + export naif (mask hitam-putih jadi MP4) | Pipeline end-to-end jalan |
| 2 | vectorize + stylize basic (garis polos seragam) | Sudah keluar outline |
| 3 | stabilize (EMA + optical flow) | Flicker terkendali |
| 4 | style params lengkap + SVG export | Bisa ganti style dari config |
| 5 | fallback pose | Frame blur tidak lagi rusak |
| 6 | (opsional) eksperimen SAM 2 tiny | — |
