# 05 — TASK BOARD

> **File ini bersifat hidup.** File 00–04 adalah spec yang relatif stabil; file ini
> berubah tiap sesi. Update statusnya sendiri, jangan tunggu diminta.

## Legenda status

`TODO` · `WIP` · `DONE` · `BLOCKED` · `SKIP` (dibatalkan, catat alasannya)

## Protokol update

1. Ubah status di baris task saat mulai/selesai
2. Kalau rencana berubah → tulis di **Update log** task tersebut, format:
   `[YYYY-MM-DD] perubahan — alasan`
3. Perubahan yang mempengaruhi arsitektur → **juga** catat di `04-DECISION-LOG.md`
4. Task boleh dipecah atau digabung. Tambahkan sub-ID (`T-201a`, `T-201b`)
5. Jangan hapus task yang dibatalkan — set `SKIP` dan tulis alasannya

## 🔄 Protokol sync (wajib, tiap selesai satu task)

Claude Code **tidak punya akses** ke Project knowledge. Repo dan Project adalah dua
salinan terpisah yang akan bercabang kalau tidak disinkronkan manual.

**Sumber kebenaran = repo Git.** Project knowledge hanya menariknya.

Tiga langkah, jadikan refleks:

```
1. git add -A && git commit -m "T-XXX: <ringkas>"
2. git push
3. Buka Project di claude.ai → ikon Sync → "Sync now"
```

**Aturan:**
- Jangan minta Claude di chat menulis ulang file 00–05 ke Project selama GitHub sync aktif —
  itu membuat salinan kedua yang bertabrakan
- Refresh sync **sebelum** memulai sesi chat yang membahas status project
- Kalau ragu Project sudah update atau belum: cek `git log` — ada commit setelah sync
  terakhir berarti Project ketinggalan

**Yang ikut ter-sync:** nama file + isi file, pada satu branch.
**Yang tidak:** commit history, PR, issues, metadata repo.

---

# PHASE 0 — Setup

### T-001 · Install Claude Code · `DONE`
- **Kerjakan:** cek plan akun (Pro/Max/Team/Enterprise/Console) → install via PowerShell →
  install Git for Windows → login
- **Done when:** `claude --version` print versi, `claude doctor` tanpa error merah
- **Blocker risk:** plan free = tidak bisa lanjut sama sekali
- **Ref:** `03-CLAUDE-CODE-ONBOARDING.md`
- **Update log:**
  - [2026-09-25] DONE — Claude Code 2.1.263, install native (`C:\Users\LEGION\.local\bin\claude.exe`),
    auto-update aktif channel `latest`, 1 instalasi saja. Login via `claude auth login`, akun
    terdeteksi Pro/Max. `claude doctor`: "No installation issues found"
  - [2026-09-25] Git for Windows di-update 2.38.1 → 2.55.0(5) — versi lama ketinggalan security fix
    (CVE-2026-62960, heap overflow wincred). Git Bash terdeteksi di `C:\Program Files\Git\bin\bash.exe`

### T-002 · Inisialisasi repo · `DONE`
- **Kerjakan:** buat folder + struktur dari `01-PIPELINE-SPEC.md` → `git init` → venv Python 3.11 →
  `.gitignore` (`work/`, `out/`, `*.png`, `*.mp4`, `venv/`) → `CLAUDE.md`
- **Done when:** `claude` jalan di folder project dan bisa membaca `CLAUDE.md`
- **Catatan:** `.gitignore` penting — dua alasan: Claude Code tidak mencoba baca ribuan
  frame PNG, dan file berat tidak ikut masuk Project knowledge saat sync
- **Update log:**
  - [2026-09-25] WIP — mulai. Temuan: Python 3.11 sudah security-only, python.org tidak lagi
    menyediakan binary installer untuk rilis 3.11 terbaru — cara install Python diputuskan di task ini
  - [2026-09-25] Python 3.11.9 dari installer resmi python.org (rilis 3.11 terakhir dengan binary
    installer Windows; tidak dapat security patch 3.11.10+ — diterima karena tool berjalan lokal).
    venv di `./venv`. Execution policy `RemoteSigned` scope `CurrentUser` (agar `Activate.ps1` jalan).
    Git global: `user.name`, `user.email`, `init.defaultBranch main`
  - [2026-09-25] DONE — root commit `fcc74d5` di `main` (10 file: `.gitignore`, `CLAUDE.md` 65 baris,
    struktur folder + `.gitkeep`, `src/rotoscope/__init__.py`). `venv/` tidak ter-commit.
    Uji: sesi baru Claude Code menjawab aturan `u2net_human_seg` + alasan lisensi dari `CLAUDE.md`
  - [2026-09-25] Follow-up: Claude Code menyimpulkan `u2net_human_seg` "aman komersial", padahal
    lisensi weights belum diverifikasi (action item D-003) → aturan #1 `CLAUDE.md` diperbarui
    mengikuti D-008 (commit `d49ced7`, `a65dcc2`)
  - [2026-09-25] Tips: prompt panjang (±100 baris) macet saat di-paste langsung ke Claude Code
    ("Pasting…") → simpan prompt ke file (mis. `notepad Txxx-prompt.md`), suruh Claude Code membacanya,
    lalu hapus file itu sebelum commit. Copy prompt pakai tombol copy blok kode (bukan seleksi teks)
    agar tidak muncul karakter escape (`\-`, `\*`, `&#x20;`)

### T-003 · Install dependency + smoke test · `DONE`
- **Kerjakan:** install stack (opencv, rembg, mediapipe, numpy, scipy, svgwrite, Pillow, pyyaml) →
  cek ffmpeg di PATH → smoke test tiap lib import & jalan → unduh 2 model kandidat segmentasi
  (rembg `u2net_human_seg` + MediaPipe `selfie_multiclass_256x256.tflite`) → ukur waktu inferensi
  CPU per frame keduanya
- **Done when:** script `scripts/smoke_test.py` lolos semua, termasuk 1 inferensi per model kandidat
- **Catatan:** pakai `onnxruntime` CPU dulu (`rembg[cpu]`). GPU ditunda ke T-601.
  File model `.tflite` simpan di `models/` dan tambahkan `models/` ke `.gitignore`
- **Update log:**
  - [2026-09-25] Scope ditambah: unduh + benchmark CPU 2 model kandidat — D-008
  - [2026-09-25] Perubahan stack: `opencv-python` → `opencv-contrib-python` (dependency bawaan
    mediapipe; dua paket OpenCV sekaligus bentrok di modul `cv2`). `CLAUDE.md` + `01` stack diperbarui
  - [2026-09-25] ffmpeg 9.0.1 essentials (gyan.dev, via `winget install "FFmpeg (Essentials Build)"`).
    Build GPLv3 + libx264 — dipanggil via subprocess, tidak dibundel
  - [2026-09-25] DONE — commit `26c1b07`. `smoke_test.py` 17 PASS / 0 FAIL. Versi: Python 3.11.9,
    opencv-contrib-python 5.0.0.93, rembg[cpu] 2.0.85, onnxruntime 1.30.0, mediapipe 1.0.1,
    numpy 2.4.6. `pip check` bersih, hanya 1 paket OpenCV. `requirements.txt` (direct + pin
    onnxruntime) + `requirements-lock.txt` (full freeze, 47 paket)
  - [2026-09-25] Benchmark CPU (foto 427×718, 10 run): `u2net_human_seg` 426 ms/frame (≈154 s per
    360 frame), MediaPipe SelfieMulticlass 104 ms/frame (≈38 s per 360 frame). Area mask 28,7% vs
    28,2%. Kualitas belum dinilai → T-102a. Model u2net 176 MB di `~/.rembg/models/` (di luar repo)
  - [2026-09-25] Scan lisensi 49 paket (`--all-licenses`): hanya certifi & tqdm (MPL-2.0, diterima —
    dipakai tanpa modifikasi). llvmlite metadata kosong → diverifikasi manual BSD-2-Clause.
    Tidak ada GPL/AGPL/non-komersial
  - [2026-09-25] ⚠️ OpenCV 5.x & mediapipe 1.x = versi mayor baru; tutorial umumnya 4.x / 0.10.x.
    API lama (mis. `mp.solutions` untuk Pose) perlu dicek di T-501. Peringatan ditambahkan ke `CLAUDE.md`

### T-004 · Siapkan aset · `DONE`
- **Kerjakan:** 1 video test 5–10 detik (gerakan sedang, bukan tercepat) + 2 brush PNG transparan +
  1 tekstur kertas
- **Done when:** aset ada di `assets/` dan `samples/`
- **Update log:**
  - [2026-09-27] DONE — brush `pencil_01.png` (padat, 36,6% terisi) + `pencil_02.png` (tipis,
    26,4%), 128×128 RGBA, dibuat sendiri di Krita, tidak menyentuh tepi kanvas. Kertas
    `rough_01.jpg` = ambientCG Paper001 (CC0), 2048×1201. Asal-usul aset di `assets/LICENSES.md`.
    Video test `samples/test.mp4`: 480×854 portrait, 30 fps, 354 frame, 11,8 s (tidak di-commit)
  - [2026-09-27] Tips Krita: background transparan = Content → Background Opacity 0%
  - [2026-09-27] Temuan: sumber 480 px < `working_width` 720 → pipeline bekerja di 480 px (tanpa
    upscale). Target TikTok/Reels 1080×1920 → perlu `output_width` terpisah, dicatat di T-203

### T-005 · Setup GitHub sync · `DONE`
- **Kerjakan:** buat repo GitHub (private) → pindahkan `docs/00`–`docs/06` ke repo → push →
  connect ke Project → pilih `docs/` + `CLAUDE.md` + `requirements.txt` → hapus salinan lama di `claude/`
- **Done when:** edit file di repo → push → Sync now → perubahan terlihat di chat Project
- **⚠️ Kritis:** hapus 7 dokumen lama di namespace `claude/` (00–06) setelah sync jalan, kalau
  tidak akan ada dua versi di knowledge dan Claude membaca yang mana saja
- **Ref:** lihat `docs/06-GITHUB-SYNC-SETUP.md`
- **Update log:**
  - [2026-09-27] WIP — langkah connect Project diverifikasi ulang dari Help Center (cocok dengan 06).
    Mulai task ini, status diupdate di repo (`docs/05`), bukan lewat chat
  - [2026-09-27] Dokumen 00–06 disalin dari Project knowledge ke `docs/`. `06` diperbarui: repo sudah
    punya commit + branch `main`, `.gitignore` merujuk file di repo, 7 dokumen lama (bukan 6),
    uji sync = tandai T-005 DONE. Selain `docs/`, `CLAUDE.md` + `requirements.txt` ikut di-sync agar
    Claude di chat melihat aturan & versi yang dipakai Claude Code
  - [2026-09-27] DONE — repo private rioFransiskus/rotoscope, sync Project: docs/ + CLAUDE.md + requirements.txt. 7 salinan lama claude/ dihapus dari Project knowledge. Mulai sekarang status diupdate di repo, bukan lewat chat.

---

# PHASE 1 — Skeleton pipeline (end-to-end jalan)

### T-101 · `ingest.py` · `DONE`
- **Kerjakan:** video → frame sequence, normalisasi ke 24 fps, resize proporsional ke
  `working_width`, tulis `meta.json`
- **Done when:** folder `work/frames/` terisi, jumlah frame sesuai durasi × 24
- **Jangan:** upscale kalau sumber lebih kecil
- **Update log:**
  - [2026-09-28] DONE — ffmpeg `fps` + `scale=W:-2` via subprocess (aman VFR, autorotate), metadata
    via ffprobe JSON. `samples/test.mp4` (30 fps, 11,8 s, 480×854, ada audio) → 283 frame 480×854
    (tanpa upscale) dalam 0,71 s. Dimensi frame divalidasi vs hitungan Python (deteksi rotasi/SAR);
    `meta.json` ditulis terakhir (atomik) sebagai penanda sukses. `source_frame_count` = null
    (ffprobe tidak melaporkan `nb_frames` untuk sampel). `tests/test_ingest.py` 19 PASS.
    File setup baru: `pyproject.toml` (setuptools>=64, `pip install -e .`), `requirements-dev.txt`
    (pytest==9.1.1, MIT); `requirements-lock.txt` tetap snapshot runtime

### T-102a · A/B test backend segmentasi · `DONE`
- **Kerjakan:** jalankan MediaPipe SelfieMulticlass 256×256 dan rembg `u2net_human_seg` pada frame
  video test yang sama (output T-101) → render side-by-side → hitung metrik sederhana di script
  A/B sendiri: waktu CPU per frame, IoU rata-rata frame berurutan, jumlah frame gagal `area_ratio`
- **Done when:** keputusan backend tercatat di D-008 (`04-DECISION-LOG.md`) + `01-PIPELINE-SPEC.md`
  stage [2] dan aturan #1 `CLAUDE.md` diperbarui
- **Aturan keputusan:** MediaPipe jadi default kalau kualitas setara. Penilaian visual oleh Rio,
  bukan Claude Code
- **Catatan:** script A/B = alat sekali pakai di `scripts/`, bukan bagian pipeline
- **Update log:**
  - [2026-09-25] Task baru, dipecah dari T-102 — D-008
  - [2026-09-28] WIP — mulai. Rencana script `scripts/ab_segment.py` menunggu approval
  - [2026-09-28] `scripts/ab_segment.py` dijalankan pada `samples/test.mp4` (283 frame 480×854, 24 fps),
    CPU 12 core, threshold 0.5, tanpa morfologi. ms tanpa 3 frame warm-up. Output di `work/ab_t102a/`.
    Varian `mediapipe_pad` = frame di-pad persegi sebelum inferensi (diagnostik aspect ratio).
    u2net memakai jalur raw (tanpa min-max `predict()` rembg), provider hanya `CPUExecutionProvider`

    | backend | ms mean | ms med | ms p95 | iou_prev mean | med | min | iou<0.55 | area<3% | area>70% | >1 blob |
    |---|---|---|---|---|---|---|---|---|---|---|
    | mediapipe | 105.3 | 104.8 | 109.1 | 0.856 | 0.870 | 0.581 | 0 | 0 | 0 | 0 |
    | mediapipe_pad | 106.8 | 106.6 | 109.3 | 0.840 | 0.857 | 0.489 | 3 | 0 | 0 | 1 |
    | u2net | 417.1 | 416.2 | 427.3 | 0.864 | 0.897 | 0.424 | 7 | 0 | 0 | 0 |

    cross_iou (mean / median / min): mediapipe vs u2net 0.805 / 0.832 / 0.451 (frame_00167);
    mediapipe_pad vs u2net 0.750 / 0.783 / 0.432 (frame_00122); mediapipe vs mediapipe_pad
    0.825 / 0.868 / 0.496 (frame_00044). u2net raw: max = 1.000 dan min = 0.000 di semua frame
    (min-max `predict()` praktis tidak mengubah hasil pada klip ini). Menunggu penilaian visual Rio
  - [2026-09-28] DONE — keputusan Rio (D-008): u2net dan mediapipe_pad ditolak; MediaPipe unggul
    atas u2net tapi tidak dipakai sebagai backend final. Style target ditambah garis oklusi →
    backend Sapiens2 (D-009), kelayakan diuji di T-102c
  - **Follow-up (JANGAN dikerjakan tanpa tanya Rio):** keluarkan `rembg` + `onnxruntime` dari
    requirements

### T-102c · Uji kelayakan Sapiens2 lokal (seg + pointmap) di GTX 1650 Ti · `TODO`
- **Kerjakan:** jalankan Sapiens2-seg 0.4B (dan Sapiens2-pointmap 0.4B) lokal pada 283 frame
  output T-101; tentukan strategi environment (Python ≥3.12 + PyTorch ≥2.7, project 3.11.9);
  fp16/fp32, tanpa bf16 (P-005) — D-009
- **Done when:** VRAM dan s/frame terukur, garis oklusi 283 frame dinilai Rio, status D-009 diperbarui
- **Catatan:** pointmap dibuang kalau seg saja sudah cukup. Cadangan kalau gagal: Depth Anything V2
  Small (Apache 2.0)
- **Update log:**
  - [2026-09-28] Task baru — D-009

### T-102b · `segment.py` · `TODO`
- **Kerjakan:** backend Sapiens2 (hasil T-102c) → mask foreground → threshold → morphological
  open+close → ambil connected component terbesar. Simpan class map bagian tubuh (+ pointmap kalau
  dipakai) ke disk
- **Done when:** `work/masks/` terisi mask biner + class map (+ pointmap kalau dipakai), subjek
  terpisah bersih dari background
- **⚠️ Kritis:** rembg/u2net ditolak (D-008), jangan dipakai
- **Update log:**
  - [2026-09-25] Dulu T-102. Backend ditentukan T-102a — D-008
  - [2026-09-28] Backend → Sapiens2 (D-009), menunggu T-102c

### T-103 · `export.py` naif · `TODO`
- **Kerjakan:** PNG sequence → MP4 via ffmpeg, `-r 24 -pix_fmt yuv420p`
- **Done when:** ada `out/animation.mp4` berisi siluet hitam-putih yang bergerak
- **Update log:**

### T-104 · `cli.py` + config loader · `TODO`
- **Kerjakan:** CLI satu perintah jalankan pipeline, loader YAML + validasi range, defaults
  masuk akal (jalan tanpa YAML)
- **Done when:** `python -m rotoscope run samples/test.mp4` menghasilkan MP4
- **🎯 Milestone Phase 1:** pipeline end-to-end jalan
- **Update log:**

---

# PHASE 2 — Vectorize + stylize basic

### T-201 · Contour extraction · `TODO`
- **Kerjakan:** `cv2.findContours` mode `RETR_EXTERNAL` → `approxPolyDP` dengan
  `simplify_epsilon` dari config → buang blob < `min_contour_area`
- **Done when:** `work/contours/*.json` berisi array titik per frame
- **Update log:**

### T-202 · Resample + point correspondence · `TODO`
- **Kerjakan:** interpolasi arc-length ke N titik tetap → rotasi urutan titik ke anchor
  anatomis (mis. titik tertinggi)
- **Done when:** titik ke-0 konsisten posisinya antar frame
- **⚠️ Pitfall P-004:** tanpa ini garis akan tampak "berputar"
- **Update log:**

### T-203 · `stylize.py` garis polos · `TODO`
- **Kerjakan:** Catmull-Rom → cubic Bezier, render stroke tebal seragam, warna solid
- **Done when:** output sudah berupa outline, bukan siluet blok
- **Update log:**
  - [2026-09-27] Keputusan tertunda: tambah parameter `output_width` (mis. 1080) terpisah dari
    `working_width`. Output digambar ulang dari vector (D-001) → resolusi output tidak terikat
    sumber (video test 480 px). Tebal garis & jitter harus relatif terhadap ukuran output, bukan px
    absolut. Ini mengubah kontrak stage [5] → bahas & catat di decision log sebelum implementasi

### T-204 · Flag `--preview N` · `TODO`
- **Kerjakan:** render hanya N frame untuk iterasi cepat
- **Done when:** preview 10 frame selesai < 30 detik
- **Kenapa sekarang:** tanpa ini Phase 3–4 akan menyiksa
- **🎯 Milestone Phase 2:** outline keluar
- **Update log:**

---

# PHASE 3 — Stabilize (stage tersulit)

### T-301 · QC metrics · `TODO`
- **Kerjakan:** hitung `area_ratio`, `iou_prev`, `component_count` per frame →
  `work/qc_report.json` + flag frame gagal
- **Done when:** report bisa menunjuk frame mana yang bermasalah
- **Update log:**

### T-302 · Temporal EMA pada mask · `TODO`
- **Kerjakan:** EMA alpha dari config, re-threshold + morphological cleanup
- **Done when:** flicker acak berkurang terukur (bandingkan `iou_prev` rata-rata)
- **Update log:**

### T-303 · Optical flow warp · `TODO`
- **Kerjakan:** `cv2.calcOpticalFlowFarneback` → warp mask frame sebelumnya ke frame
  sekarang → blend dengan bobot `optical_flow_blend`
- **Done when:** boiling turun signifikan tanpa lag terlihat
- **Catatan:** ini mitigasi paling efektif untuk P-001
- **Update log:**

### T-304 · Kalibrasi `boil_preserve` · `TODO`
- **Kerjakan:** render 3–4 varian nilai berbeda, bandingkan side-by-side, pilih
- **Done when:** ada nilai default yang kamu setujui secara artistik
- **Catatan:** ini penilaian mata, bukan metrik. Tidak bisa didelegasikan ke Claude Code
- **🎯 Milestone Phase 3:** flicker terkendali
- **Update log:**

---

# PHASE 4 — Style lengkap

### T-401 · Width modulation + taper · `TODO`
- **Kerjakan:** variasi tebal sepanjang path (`width_variation`, `width_noise_scale`),
  ujung menipis
- **Update log:**

### T-402 · Jitter deterministic · `TODO`
- **Kerjakan:** Perlin noise per titik, seed = `hash(frame_index, param_seed)`,
  `temporal_drift` untuk perubahan antar frame
- **⚠️ Pitfall P-007:** random murni = tidak reproducible, tidak bisa di-debug
- **Update log:**

### T-403 · Multipass stroke · `TODO`
- **Kerjakan:** garis tumpang tindih dengan offset + opacity falloff
- **Update log:**

### T-404 · Texture + paper layer · `TODO`
- **Kerjakan:** brush stamping (spacing, pressure noise) + background kertas + vignette
- **Done when:** hasil mendekati referensi B
- **Update log:**

### T-405 · SVG export · `TODO`
- **Kerjakan:** `svgwrite`, path per frame, ke `out/svg/`
- **Done when:** SVG bisa dibuka & diedit di Krita/Illustrator
- **Update log:**

### T-406 · Preset library · `TODO`
- **Kerjakan:** 4 preset — `rough-sketch`, `clean-line`, `heavy-marker`, `pencil-light`
- **🎯 Milestone Phase 4:** style bisa diganti dari config
- **Update log:**

---

# PHASE 5 — Fallback pose

### T-501 · `fallback_pose.py` · `TODO`
- **Kerjakan:** MediaPipe Pose 33 landmark → mask sintetik dari capsule/polygon
  (torso, lengan, kaki, kepala)
- **⚠️ Cek dulu:** mediapipe 1.x — pastikan API Pose yang dipakai masih ada (legacy `mp.solutions`
  vs Tasks API `PoseLandmarker`)
- **Update log:**

### T-502 · Integrasi blend + QC · `TODO`
- **Kerjakan:** panggil fallback hanya untuk frame gagal QC, blend dengan mask asli
  (jangan ganti total), catat frame mana yang pakai fallback
- **Done when:** frame motion-blur tidak lagi rusak, transisi tidak melompat
- **🎯 Milestone Phase 5:** pipeline tahan input buruk
- **Update log:**

---

# PHASE 6 — Opsional

### T-601 · `onnxruntime-gpu` · `SKIP`
- **Kerjakan:** CUDA 12.x + cuDNN 9 versi cocok → benchmark sebelum/sesudah
- **Risiko:** sumber error setup yang sering di Windows (P-006). Timebox, jangan dikejar
- **Catatan:** jadi `SKIP` kalau T-102a memilih MediaPipe (rembg + onnxruntime keluar dari stack)
- **Update log:**
  - [2026-09-28] SKIP — `onnxruntime-gpu` tidak relevan lagi: rembg/u2net ditolak (D-008), GPU
    lewat PyTorch untuk Sapiens2 (D-009, T-102c)

### T-602 · Eksperimen SAM 2 tiny · `TODO`
- **Kerjakan:** `sam2.1_hiera_tiny`, frame di-downscale, propagasi per-chunk
- **⚠️ Ganti `bfloat16` → `float16`** (Turing tidak dukung bf16 — P-005)
- **Ekspektasi realistis:** kemungkinan besar OOM di 4 GB. Ini eksperimen, bukan target
- **Update log:**

### T-603 · Batch processing · `TODO`
- **Kerjakan:** proses banyak video sekaligus, progress bar, resume dari crash
- **Update log:**

---

## Papan status

| Phase | Task | Selesai |
|---|---|---|
| 0 Setup | T-001 … T-005 | 5/5 |
| 1 Skeleton | T-101 … T-104 (T-102 → a/b/c) | 2/6 |
| 2 Vectorize | T-201 … T-204 | 0/4 |
| 3 Stabilize | T-301 … T-304 | 0/4 |
| 4 Style | T-401 … T-406 | 0/6 |
| 5 Fallback | T-501 … T-502 | 0/2 |
| 6 Opsional | T-601 … T-603 | 0/3 |

**Total: 30 task** · Selesai: 7/30 (SKIP tidak dihitung selesai)
