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
    requirements → task **T-107** (2026-09-29)

### T-102c · Uji kelayakan Sapiens2 lokal (seg + pointmap) di GTX 1650 Ti · `DONE`
- **Kerjakan:** jalankan Sapiens2-seg 0.4B (dan Sapiens2-pointmap 0.4B) lokal pada 283 frame
  output T-101; tentukan strategi environment (Python ≥3.12 + PyTorch ≥2.7, project 3.11.9);
  fp16/fp32, tanpa bf16 (P-005) — D-009
- **Done when:** VRAM dan s/frame terukur, garis oklusi 283 frame dinilai Rio, status D-009 diperbarui
- **Catatan:** pointmap dibuang kalau seg saja sudah cukup. Cadangan kalau gagal: Depth Anything V2
  Small (Apache 2.0)
- **Update log:**
  - [2026-09-28] Task baru — D-009
  - [2026-09-28] WIP — mulai. Cek environment + usulan opsi setup menunggu pilihan Rio
  - [2026-09-28] Setup Opsi 1a (dipilih Rio): venv utama + `torch==2.7.1+cu118`,
    `torchvision==0.22.1+cu118`, `transformers==5.17.0` (Sapiens2 masuk sejak 5.10.1, Python ≥3.10).
    Driver 517.00 (CUDA maks 11.7) → cu126/cu128 tidak jalan tanpa update driver. `pip check` bersih,
    0 paket existing berubah, 25 paket baru. Checkpoint di cache HF (di luar repo)
  - [2026-09-28] Run `scripts/sapiens2_probe.py` (+ `scripts/sapiens2_groups.json`) → `work/t102c/`.
    Status tetap WIP, menunggu keputusan Rio.

    **(a) Setup:** GTX 1650 Ti (sm_75), driver 517.00, torch 2.7.1+cu118 (CUDA 11.8, arch list memuat
    sm_75), transformers 5.17.0, attention `sdpa`. VRAM terpakai di luar proses sebelum run: 561 / 4096 MiB.
    Input model 1024×768: seg = **stretch** (`do_pad=false`), pointmap = resize jaga rasio + **pad**.

    **(b) Benchmark** (20 frame, s/frame = forward + post-process, tanpa 3 frame warm-up):

    | run | status | s mean | s med | s p95 | peak alloc MiB | peak reserved MiB | bebas NaN/inf |
    |---|---|---|---|---|---|---|---|
    | seg fp16 | ok | 7.992 | 7.990 | 8.008 | 2185 | 2258 | ya |
    | seg fp32 | ok | 2.538 | 2.534 | 2.553 | 2972 | 3188 | ya |
    | pointmap fp16 | ok | 14.690 | 14.692 | 14.697 | 2420 | 3084 | **tidak (20/20 frame)** |
    | pointmap fp32 | **OOM** | – | – | – | 3437 saat gagal | 3516 | – |
    | seg fp32 CPU (3 frame) | ok | 17.401 | 17.047 | 18.077 | – | – | ya |

    Seg fp16 vs fp32: kesamaan kelas 99.998% (min 99.996%). Precision terpilih: **fp32** (fp16
    pointmap menghasilkan NaN/inf). Pointmap: tidak ada precision yang lolos → full run pointmap dilewati.

    **(c) Foreground (kelas ≠ 0) full run seg fp32, 283 frame** vs MediaPipe T-102a:

    | backend | iou_prev mean | med | min | iou<0.55 | area<3% | area>70% | >1 blob |
    |---|---|---|---|---|---|---|---|
    | sapiens2-seg | 0.903 | 0.907 | 0.724 | 0 | 0 | 0 | 0 |
    | mediapipe (T-102a) | 0.856 | 0.870 | 0.581 | 0 | 0 | 0 | 0 |

    label_agreement_prev (irisan foreground): mean 0.951, median 0.963, min 0.790.
    cross_iou vs mediapipe: mean 0.861, median 0.906, min 0.466.

    **(d) cross_iou zona gagal T-102a:** frame 1–7: 0.466 / 0.484 / 0.493 / 0.483 / 0.471 / 0.507 /
    0.483; frame 236–241: 0.883 / 0.701 / 0.851 / 0.909 / 0.933 / 0.949.

    **(e) Pangsa piksel foreground:** Lower_Clothing 50.88%, Upper_Clothing 21.26%, Hair 16.19%,
    Left_Hand 4.69%, Right_Hand 4.01%, Face_Neck 2.22%, Apparel 0.42%, Torso 0.12%, Lower_Lip 0.07%,
    Upper_Lip 0.05%, Upper_Teeth 0.03%, Left_Shoe / Left_Lower_Arm / Lower_Teeth / Tongue 0.01% masing-masing;
    13 kelas lain < 0.005% (Left/Right Foot, Lower_Leg, Upper_Leg, Sock, Upper_Arm; Right_Lower_Arm,
    Right_Shoe, Eyeglass).

    **(f) Waktu:** benchmark 9.8 mnt + full run seg 12.2 mnt (2.569 s/frame total termasuk I/O) +
    metrik/visual 14.5 s ≈ 22 mnt.

    **Status kriteria:** VRAM ≤ 3 GB (3072 MiB) per model — seg fp16 lolos; seg fp32 lolos untuk
    allocated (2972) tapi reserved 3188 > 3072; pointmap fp16 allocated 2420, reserved 3084 > 3072;
    pointmap fp32 OOM. NaN/inf — seg bebas di fp16 dan fp32; pointmap fp16 NaN/inf di semua frame.
    Estimasi klip 15 s (360 frame): seg saja fp32 ≈ 925 s (15.4 mnt); seg + pointmap tidak bisa
    diestimasi karena tidak ada precision pointmap yang lolos.

    **Catatan:** `config.json` checkpoint seg hanya berisi `LABEL_0..28` (dicek lewat
    `AutoConfig.id2label`). Nama kelas untuk grup preview diambil dari tabel resmi
    `facebookresearch/sapiens2` `docs/SEG.md` (fallback di script, hanya kalau id2label generik).
  - [2026-09-29] Review visual Rio atas full run 0.4B: frame 1–7 kaki sudah masuk mask; garis lengan
    bawah/tangan di depan badan muncul dan tepat; garis rambut di bahu diinginkan (grup head tetap);
    getaran garis interior cukup mengganggu; waktu proses bukan batasan, prioritas kualitas.
    Eksperimen lanjutan E1 + E2 → `work/t102c/exp/` (full run 0.4B tidak ditimpa). Status tetap WIP.

    **Fallback nama kelas (A, diterima Rio dengan syarat):** list disimpan sekali di
    `scripts/sapiens2_classes.json` + URL sumber + commit `744905ba762c0ccb369c3206e3a49d81a634e023`
    (`docs/SEG.md`, 2026-05-15). Dibaca lokal tanpa jaringan; self-check jumlah kelas = `num_labels`
    (29 = 29, termasuk seg 0.8B). Default `--grid-frames` + 73, 78, 82, 87, 92 (kaki kanan di depan kaki kiri).

    **Sebelum mulai:** RAM total 15.91 GiB, bebas 6.16 GiB. Unduhan (hanya `model.safetensors` + json):
    seg-0.8b 3.08 GiB, seg-1b 5.48 GiB, Depth-Anything-V2-Small 0.09 GiB (model card `apache-2.0`);
    pointmap-0.4b sudah di cache → total ≈ 8.65 GiB < 15 GB. Script: `scripts/sapiens2_exp.py`
    (tiap langkah = proses terpisah; peak RAM = PeakWorkingSetSize proses via ctypes, termasuk load model
    dan library CUDA; checkpoint dari cache HF dengan `HF_HUB_OFFLINE=1`).

    **E1 — sumber kedalaman, benchmark 20 frame** (tersebar merata, s/frame = forward + post-process,
    tanpa 3 warm-up):

    | sumber | status | NaN/inf | s mean | s med | s p95 | peak alloc MiB | peak reserved MiB | peak RAM MiB | input model |
    |---|---|---|---|---|---|---|---|---|---|
    | (a) pointmap 0.4B, bobot fp32 + autocast fp16, GPU | **OOM** | – | – | – | – | 3469 | 3546 | 4154 | – |
    | (b) pointmap 0.4B fp32, CPU | ok | 0/20 | 30.664 | 31.489 | 32.158 | – | – | 5142 | 1024×768 |
    | (c) Depth-Anything-V2-Small fp32, GPU | ok | 0/20 | 0.159 | 0.159 | 0.161 | 291 | 424 | 3182 | 924×518 |

    (a) percobaan 1: OOM di frame pertama (617 MiB reserved-tak-terpakai → fragmentasi). Satu percobaan
    perbaikan `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128`: OOM di frame kedua. Timebox habis → gagal.
    (c) keluaran = kedalaman relatif terbalik (disparity); panel gradien memakai |grad| sehingga arah skala
    tidak berpengaruh.

    **E1 — full run:**

    | sumber | frame | NaN/inf | infer mean s | total mean s/frame | wall |
    |---|---|---|---|---|---|
    | (c) DA-V2-Small GPU | 283/283 | 0 | 0.161 | 0.185 | 1.0 mnt |
    | (b) pointmap fp32 CPU | **95/283** (frame 0–94, semua valid) | 0 | – | – | – |

    (b) dihentikan Claude Code karena memori sistem kritis saat sesi idle (bukan crash script). Keputusan
    Rio: JANGAN jalankan ulang dulu. Skip per-frame sudah diimplementasikan (`full` melewati frame yang
    output-nya ada dan valid: terbaca, ukuran = frame, semua finite) — belum dijalankan.
    Estimasi sisa: 188 frame × ±31 s ≈ 1.6 jam.

    Visual: `side_by_side_da2s_gpu.mp4`, `frames_grid_da2s_gpu.png`, `side_by_side_da2s_gpu_f073-092.mp4`;
    tanpa inferensi baru: `compare_f073-092.mp4` + `frames_grid_compare.png` (frame 73, 78, 82, 87, 92)
    dengan panel [asli | garis grup seg 0.4B | |grad| DA-V2-Small | |grad| pointmap CPU] — 20/20 frame
    73–92 tersedia untuk kedua sumber.

    **E2 — seg lebih besar, 20 frame berurutan 183–202** (sekitar frame 193, `label_agreement_prev`
    terendah full run 0.4B = 0.790). Pembanding = full run 0.4B fp32 pada frame yang sama:

    | model | device | status | s mean | s med | peak alloc MiB | peak reserved MiB | peak RAM MiB | sama kelas vs 0.4B | fg IoU vs 0.4B | label_agreement_prev | label_agreement_prev 0.4B |
    |---|---|---|---|---|---|---|---|---|---|---|---|
    | seg 0.8B | GPU fp16 | ok, 0 NaN/inf | 15.991 | 15.951 | 3014 | 3276 | 5409 | 0.9909 (min 0.9877) | 0.9830 (min 0.9600) | 0.8899 (min 0.7782) | 0.8889 (min 0.7895) |
    | seg 1B | – | dilewati | – | – | – | – | – | – | – | – | – |

    seg 1B dilewati (keputusan Rio): tidak layak di hardware ini (VRAM tidak muat, RAM CPU tidak memenuhi
    syarat). label_agreement_prev = rata-rata 19 pasangan frame (184–202). Per frame (0.8B / 0.4B):
    193: 0.778 / 0.790; 185: 0.818 / 0.814; 201: 0.817 / 0.834; 189: 0.842 / 0.838.
    Load 0.8B 6.5 s; peak reserved 3276 MiB > 3072 MiB (batas CLAUDE.md). Klip:
    `side_by_side_seg_0.8b_f183-202.mp4` [asli | kelas 0.8B | garis grup 0.8B | garis grup 0.4B].
    Catatan: 0.8B jalan fp16, pembanding 0.4B fp32 (fp16 vs fp32 0.4B: kesamaan kelas 99.998%).
  - [2026-09-29] Penilaian visual Rio atas `compare_f073-092`: batas kaki kanan–kiri terlihat di kedua
    sumber (tipis abu-abu di DA-V2-Small, bayangan gelap tegas di pointmap); stabil antar frame di
    keduanya; posisi tepi tepat di keduanya, pointmap lebih jelas; tidak ada garis palsu di tali cargo,
    saku, atau lipatan. Status tetap WIP.

    **Uji ekstraksi garis dari kedalaman (tanpa inferensi baru)** — `scripts/sapiens2_exp.py lines`,
    frame 73–92, output tersimpan DA-V2-Small (disparity) dan pointmap CPU (Z). Metode: lompatan relatif
    ke tetangga 4-arah `J = max(a,b)/min(a,b) − 1` (sama untuk Z dan 1/Z), hanya pasangan yang keduanya
    di foreground seg 0.4B ter-erode (kernel 5 px); piksel = garis kalau J > threshold; digabung dengan
    garis grup seg. Threshold = persentil J di foreground, dihitung PER KLIP (20 frame digabung,
    1 354 956 nilai per sumber):

    | sumber | p95 | p98 | p99.5 | frame |
    |---|---|---|---|---|
    | DA-V2-Small (disparity) | 0.02063 | 0.06447 | 0.40018 | 20/20 |
    | pointmap 0.4B CPU (Z) | 0.00392 | 0.00552 | 0.00994 | 20/20 |

    Nilai J DA dan pointmap tidak sebanding langsung (disparity DA hanya benar sampai skala + offset).
    Kuantisasi penyimpanan float16: langkah relatif di median ≈ 0.00053 (pointmap, Z median 1.86) dan
    ≈ 0.00059 (DA, median 3.29) → threshold p95 pointmap ≈ 7.5 langkah float16.
    Output `work/t102c/exp/`: `lines_da2_f073-092.mp4`, `lines_pointmap_f073-092.mp4`
    [asli | p95 | p98 | p99.5, threshold di label], `frames_grid_lines_da2.png`,
    `frames_grid_lines_pointmap.png` (frame 73, 82, 92), `lines_thresholds.json`.

    **Poin D:** `Lower_Clothing` dipindah ke grup `torso` di `scripts/sapiens2_groups.json` (garis pinggang
    kaos–celana tidak digambar; grup `lower_clothing` dihapus). Preview seg dirender ulang tanpa inferensi:
    `work/t102c/side_by_side.mp4` + `frames_grid.png` (grid kini memuat 73, 78, 82, 87, 92); versi lama
    disimpan sebagai `*_grup_v1.*`. `metrics_per_frame.csv` identik dengan versi lama. Video `lines_*`
    memakai grup baru; `compare_f073-092.mp4`, `side_by_side_da2s_gpu*` dan `side_by_side_seg_0.8b_*`
    masih memakai grup lama.
  - [2026-09-29] Penilaian Rio: threshold terbaik p95 untuk kedua sumber; DA-V2-Small di p95 SETARA dengan
    pointmap; masih ada garis glitch di area tangan pada DA-V2 p95; E2: di preview garis rambut menyatu
    dengan wajah. Status tetap WIP.

    **Keputusan Rio — rambut dipisahkan dari wajah:** grup `head` di `scripts/sapiens2_groups.json` diganti
    `hair = [Hair]` dan `face = [Face_Neck, Eyeglass, Lower_Lip, Upper_Lip, Lower_Teeth, Upper_Teeth,
    Tongue]`; grup lain tidak diubah. Render tanpa inferensi (`sapiens2_exp.py seglines`):
    `work/t102c/exp/seglines_da2_f183-202.mp4`, 2 panel [seg 0.4B + DA-V2 p95 | seg 0.8B + DA-V2 p95].
    Threshold DA p95 per klip 183–202 = 0.01866 (dari J di foreground seg 0.4B ter-erode), dipakai sama
    untuk kedua panel; garis DA di tiap panel dibatasi foreground seg panel itu.

    **Tabel E2 lengkap (frame 183–202):**

    | model | precision / device | s/frame mean | peak VRAM alloc MiB | peak VRAM reserved MiB | label_agreement_prev mean | min | % kelas sama vs 0.4B mean | min |
    |---|---|---|---|---|---|---|---|---|
    | seg 0.8B | fp16 GPU | 15.991 | 3014 | 3276 | 0.8899 | 0.7782 | 99.09% | 98.77% |
    | seg 0.4B | fp32 GPU (full run) | 2.544 | 2972 | 3188 | 0.8889 | 0.7895 | – | – |

    s/frame 0.4B = rata-rata frame 183–202 di full run; VRAM 0.4B = benchmark fp32 (20 frame tersebar).
    label_agreement_prev = 19 pasangan frame (184–202).

    **Diagnosis glitch tangan (belum diperbaiki)** — `sapiens2_exp.py diag-hands` → `diag_hands_da2.json`.
    Garis gabungan seg 0.4B (grup terbaru, tebal 2 px) + DA-V2 p95 (thr 0.02063, per klip 73–92), frame
    73–92. Zona tangan = Left_Hand ∪ Right_Hand di-dilate 15 px; komponen = komponen terhubung 8-arah yang
    menyentuh zona; "kecil" < 100 px. Frame dengan komponen kecil terbanyak: **frame 90** (8 kecil / 9 total).
    Piksel garis di zona tangan frame 90: seg saja 4293, DA saja 584, seg+DA 196.

    | # | ukuran px | seg saja | DA saja | kelas seg di sekitar komponen |
    |---|---|---|---|---|
    | 1 | 2 | 0 | 2 | Left_Hand |
    | 2 | 2 | 0 | 2 | Left_Hand |
    | 3 | 4 | 0 | 4 | Right_Hand |
    | 4 | 26 | 26 | 0 | Right_Hand + pulau Left_Hand / Upper_Clothing |
    | 5 | 42 | 42 | 0 | Right_Hand + pulau Upper_Clothing |
    | 6 | 45 | 45 | 0 | Right_Hand + pulau Upper_Clothing |
    | 7 | 94 | 0 | 94 | Upper_Clothing dekat Left_Hand (bbox 6×29) |
    | 8 | 99 | 99 | 0 | Right_Hand + pulau Upper_Clothing |
    | 9 | 11039 | 8647 (+207 seg+DA) | 2185 | garis utama (siluet + batas grup) |

    Jumlah komponen kecil per frame 73–92: 0, 1, 3, 1, 2, 1, 2, 4, 1, 4, 4, 0, 1, 3, 5, 3, 4, 8, 3, 2.
  - [2026-09-29] Penilaian Rio atas `seglines_da2_f183-202`: 0.8B terlihat lebih baik dari 0.4B di seluruh
    video (tepi lebih halus, bercak lebih sedikit); garis rambut–wajah sesuai dan stabil.
    **Keputusan Rio: pointmap DIBUANG, full run pointmap CPU tidak dilanjutkan** (tetap 95/283 frame).
    Status tetap WIP.

    **Seg 0.8B fp16 GPU, frame 73–92** (inferensi baru, `seg seg_0.8b --seg-frames 73 92` →
    `e2_seg_0.8b_f073-092.json`, `seg_seg_0.8b/`): VRAM terpakai di luar proses sebelum run (nvidia-smi)
    364 / 4096 MiB; s/frame mean 15.941 (median 15.936, p95 15.965); peak VRAM alloc 3014 MiB, reserved
    3276 MiB; peak RAM 5433 MiB; 0 NaN/inf. label_agreement_prev 0.8B mean 0.9756 (min 0.9518) vs 0.4B
    0.9694 (min 0.9414), 19 pasangan frame; kelas sama vs 0.4B mean 99.33% (min 99.00%).

    **Post-processing** (`sapiens2_exp.py filtered`, tanpa mengubah model, nilai sama untuk semua panel).
    Urutan a → b → garis grup + DA-V2 p95 → c:
    - (a) filter pulau kelas: komponen 8-arah sebuah kelas (termasuk background) < N px diganti kelas
      mayoritas di cincin 1 px sekelilingnya (satu lintasan, cincin dibaca dari peta asli).
      **N = 30.** Pulau penyebab komponen garis seg-only di frame 90 (`diag_hands_da2.json`) berukuran
      1, 3, 7, 28 px → N > 28. Struktur terkecil yang stabil antar frame: Left_Shoe 44–48 px (0.4B frame
      187–190), Left_Hand 36 px (0.8B frame 196) → N ≤ 36. N = 50 (dicoba dulu) menghapus keduanya.
    - (b) penghalusan tepi: mode filter pada peta GRUP — tiap piksel diberi grup yang paling sering di
      jendela K×K (hitungan per grup = box filter atas mask grup; seri dimenangkan grup asli). Background =
      grup 0, jadi siluet luar ikut dihaluskan. **K = 3.** Aturan: K terbesar yang tidak membuat komponen
      grup ≥ N px kehilangan > 50% area, diuji pada 80 peta (0.4B + 0.8B, kedua jendela): K=3 → 0 komponen
      (0.099% piksel foreground berubah grup); K=5 → 12 (mis. potongan left_arm 37–52 px); K=7 → 43.
    - (c) filter komponen garis: komponen 8-arah < M px dibuang. **M = 5.** Komponen DA-only di zona
      tangan frame 90 berukuran 2, 2, 4 px (komponen 94 px tidak bisa dipisahkan dari garis DA sah
      101–488 px dengan ukuran saja); setelah a+b, 231 dari 779 komponen garis (30%) ≤ 4 px, histogram
      ukuran 1–4 px: 129 / 38 / 64 / 17 (0 komponen 1 px karena garis grup tebal 2 px).

    Video 3 panel [0.4B mentah | 0.4B + post | 0.8B + post], garis grup (hair/face terpisah) + DA-V2 p95:
    `work/t102c/exp/filtered_f073-092.mp4` (thr 0.02063), `filtered_f183-202.mp4` (thr 0.01866);
    metrik per frame: `filtered_metrics.json`.

    Komponen kecil (< 100 px, 8-arah) — total 20 frame; "tangan" = menyentuh zona tangan (dilate 15 px,
    definisi = diag-hands), "semua" = seluruh frame; "sebelum c" = setelah a+b:

    | jendela | model | tangan mentah | tangan sebelum c | tangan sesudah | semua mentah | semua sesudah | garis utama mean px mentah → sesudah |
    |---|---|---|---|---|---|---|---|
    | 73–92 | 0.4B | 52 | 33 | 23 | 233 | 116 | 11714 → 11487 |
    | 73–92 | 0.8B | 30 | 24 | 17 | 191 | 102 | 11342 → 11335 |
    | 183–202 | 0.4B | 25 | 25 | 16 | 96 | 50 | 7457 → 7119 |
    | 183–202 | 0.8B | 17 | 13 | 6 | 71 | 31 | 6840 → 6755 |

    Per frame, komponen kecil di zona tangan mentah → sesudah:
    - 73–92 0.4B: 0→0, 1→0, 3→3, 1→1, 2→1, 1→1, 2→1, 4→1, 1→1, 4→2, 4→1, 0→0, 1→0, 3→3, 5→1, 3→2,
      4→3, 8→1, 3→0, 2→1
    - 73–92 0.8B: 1→1, 1→1, 1→1, 0→0, 1→1, 0→0, 2→0, 3→2, 0→0, 2→1, 0→0, 0→0, 3→1, 2→2, 2→1, 2→1,
      2→2, 4→1, 1→1, 3→1
    - 183–202 0.4B: 1→1, 3→2, 4→1, 1→0, 4→1, 2→1, 2→1, 3→1, 1→2, 2→2, 1→2, 0→0 (194–199), 0→1, 0→0,
      1→1
    - 183–202 0.8B: 0→0, 3→2, 0→0, 0→0, 1→0, 1→0, 2→0, 1→1, 1→1, 0→0, 1→1, 1→1, 0→0, 3→0, 1→0, 1→0,
      0→0, 1→0, 0→0, 0→0

    Garis utama (komponen terbesar) per frame ada di `filtered_metrics.json`. Sesudah post-processing
    turun di semua frame 0.4B (kedua jendela) dan 0.8B 183–202; 0.8B 73–92 naik di 4 frame: 78
    (11379→11450), 85 (12260→12273), 88 (11379→11442), 91 (10607→11177).
  - [2026-09-29] Keputusan Rio: 0.8B + post-processing jauh lebih bersih di kedua jendela → **kandidat
    final**. Uji stabilitas sepanjang klip sebelum Tahap 4. Status tetap WIP.

    **Full run seg 0.8B fp16 GPU, 283 frame** (`sapiens2_exp.py seg-full seg_0.8b`; resume per frame:
    frame dengan class map valid — terbaca, uint8, ukuran = frame, id < 29 — dilewati; log per frame di
    `full_seg_0.8b_frames.jsonl`, ringkasan `full_seg_0.8b.json`). Aplikasi berat ditutup Rio.
    - Sebelum run: VRAM terpakai di luar proses (nvidia-smi, sebelum CUDA init) 258 / 4096 MiB;
      `torch.cuda.mem_get_info` sebelum load 3318 / 4096 MiB bebas; setelah load 1568 MiB bebas
      (reserved 1750, allocated 1579). RAM bebas 7.11 / 15.91 GiB.
    - 40 frame (73–92, 183–202) sudah ada & valid → dilewati; 243 frame diinferensi. Status **ok, tanpa
      OOM**; output valid 283/283; **NaN/inf 0/243**.
    - s/frame (243 frame, tanpa 3 warm-up): infer mean 15.847 (median 15.844, p95 15.869, max 15.979);
      total termasuk I/O mean 15.873. Wall 64.5 mnt, load 5.6 s. Estimasi klip 15 s (360 frame) ≈ 95 mnt.
    - Peak VRAM per frame: reserved 3276 MiB di SEMUA 243 frame (min = max; > 3072 batas CLAUDE.md),
      allocated maks 3014 MiB. Peak RAM proses 2485 MiB.

    **Metrik full klip, 0.8B vs 0.4B** (fungsi `compute_metrics` / `summarize_metrics` dari
    `sapiens2_probe.py`; 0.4B dihitung ulang dan identik dengan angka full run sebelumnya) →
    `metrics_full_seg_0.8b.json` + `.csv`:

    | model | iou_prev mean | med | min | iou<0.55 | area<3% | area>70% | >1 blob | label_agreement_prev mean | med | min | cross_iou mp mean |
    |---|---|---|---|---|---|---|---|---|---|---|---|
    | seg 0.4B fp32 | 0.9034 | 0.9069 | 0.7240 | 0 | 0 | 0 | 0 | 0.9509 | 0.9632 | 0.7895 | 0.8611 |
    | seg 0.8B fp16 | 0.9008 | 0.9042 | 0.7126 | 0 | 0 | 0 | 0 | 0.9547 | 0.9688 | 0.7782 | 0.8627 |

    Kelas sama 0.8B vs 0.4B per frame: mean 99.33%, median 99.40%, min 97.50%.
    label_agreement_prev terendah — 0.4B: 193 (0.790), 213 (0.798), 185 (0.814), 205 (0.823), 181 (0.829);
    0.8B: 193 (0.778), 213 (0.800), 201 (0.817), 185 (0.818), 181 (0.828).
    cross_iou vs mediapipe zona T-102a, 0.8B: frame 1–7 0.466 / 0.484 / 0.493 / 0.483 / 0.471 / 0.506 /
    0.481; frame 236–241 0.868 / 0.711 / 0.848 / 0.899 / 0.925 / 0.954.

    **Render full klip** (`filtered-full seg_0.8b`): `work/t102c/exp/filtered_full_0.8b.mp4`, 2 panel
    [asli | garis 0.8B + post N=30, K=3, M=5 + DA-V2 p95], grup terbaru (hair/face terpisah, Lower_Clothing
    di torso). Threshold DA p95 per klip (283 frame, foreground peta grup 0.8B setelah a+b, erode 5 px)
    = 0.01474. Piksel garis per frame: `filtered_full_0.8b.json`.
  - [2026-09-29] Penilaian Rio atas `filtered_full_0.8b.mp4`: KURANG PUAS — terlihat seperti render 3D
    diberi filter, masih ada "bayangan" tertangkap, tepi tidak bersih, tidak terasa digambar tangan.
    Preview masih piksel batas + piksel lompatan kedalaman mentah (belum vectorize + stylize) → **look
    test** sebelum menutup T-102c. Status tetap WIP.

    Alat sekali pakai `scripts/look_test.py` (tanpa inferensi; seg 0.8B full run + DA-V2 tersimpan;
    `src/rotoscope/` tidak disentuh). Output `work/t102c/look/`. Semua parameter = argumen CLI.

    **1. Diagnosis sumber "bayangan"** (`look_test.py diag` → `diag_sources.png` + `.json`), preview
    filtered_full sekarang (DA p95 thr 0.01474), 3 panel [garis seg saja | garis DA saja | gabungan]:

    | frame | garis seg px | garis DA saja px | DA saja / gabungan |
    |---|---|---|---|
    | 1 | 9310 | 3497 | 27.3% |
    | 87 | 9964 | 4888 | 32.9% |
    | 120 | 11808 | 4234 | 26.4% |
    | 183 | 9296 | 2742 | 22.8% |
    | 236 | 12428 | 1845 | 12.9% |

    **2. Garis kedalaman baru** (tipis + selektif, disparity DA-V2): besaran = |grad log d| (Gaussian
    σ = 1 px lalu Sobel/8; log → gradien relatif, sama untuk Z dan 1/Z) → non-maximum suppression searah
    gradien (4 bin arah) → hysteresis (8-arah) → hanya di dalam satu grup → skeleton + filter panjang.
    Nilai dipilih dari data (`look_test.py stats`, persentil per klip dari 18 627 541 nilai di foreground
    ter-erode 5 px; distribusi dari 41 frame sampel):
    - **T_high = p95 = 0.02540** (per klip, seperti sekarang).
    - **T_low = p90 = 0.01114** = 0.44 × T_high (rasio 1 : 2.3). Konvensi Canny T_high : T_low = 2:1–3:1;
      p92 = 0.56× (< 2:1), p88 = 0.37×, p85 = 0.31× (> 3:1). p90 = yang paling selektif di dalam rentang.
    - **D = 7 px** (jarak minimum ke batas grup, termasuk siluet). Histogram jarak piksel tepi
      NMS+hysteresis ke batas grup, 0..8 px: 7294 / 3179 / 2490 / 1531 / 1194 / 1156 / 1118 / 643 / 487 —
      tumpukan 0–6 px (tepi yang sama dengan batas seg, bergeser), turun 42% di 7 px.
    - **L = 30 px** (panjang skeleton minimum). Setelah D = 7: 236 komponen, 201 (85%) ≤ 30 px.
      Komponen terpanjang di frame uji kaki menyilang ada di Lower_Clothing (cy 591–730 dari 854): 73 → 64,
      78 → 73, 82 → 89, 87 → 154, 92 → 38 px; L = 30 mempertahankan semuanya (min 38).

    **3. Stroke sederhana** (bukan stage 5 lengkap): batas grup (morph. gradient 3×3 → thinning) + garis
    kedalaman (thinning) → tracing skeleton jadi polyline (junction dilepas, disambung lagi ke ujung
    jalur) → approxPolyDP → Catmull-Rom → resample → tebal bervariasi + taper ujung → jitter searah
    normal → multipass → supersampling 3× (anti-alias), tinta #1a1a1a di atas kertas #f4f1ea. Default =
    preset rough-sketch docs/02: simplify_epsilon 2.5, resample_points 200 (maks 1 titik/px untuk stroke
    pendek), smooth_tension 0.5, min_contour_area 800 (komponen peta grup < 800 px² digabung ke grup
    mayoritas sekeliling), width_base 3.2, width_variation 0.45, width_noise_scale 0.08, opacity 0.92,
    taper_ends, jitter amplitude 1.8 / frequency 0.12, multipass 2 pass / offset 1.2 / falloff 0.55.
    Noise = value noise 1D (smoothstep), bukan Perlin. Parameter baru di luar preset: taper_px 20,
    taper_min 0.15, spline_steps 8, min_stroke_px 6 (cabang thinning), ss 3, param_seed 0.
    Jitter deterministik: seed = crc32(frame_index, param_seed) + indeks stroke + indeks pass (P-007);
    dicek: render frame 87 dua kali → identik; param_seed lain → beda. Tidak diterapkan:
    temporal_drift / temporal_seed_mode (tanpa stage 3), tekstur kertas/brush.

    **4. Output:** `look_frames.png` (frame 1, 87, 120, 183, 236: [asli | preview filtered_full sekarang |
    look test]), `look_f073-092.mp4` + `look_f183-202.mp4` ([asli | look test]; label: tanpa stabilisasi
    temporal, getaran antar frame BELUM representatif), `look_params.json`. Per frame (43 frame dirender):
    stroke min / median / max 13 / 43 / 68; piksel skeleton garis kedalaman 0 / 45 / 369; 17 frame tanpa
    garis kedalaman (mis. 183, 236). Frame 1 / 87 / 120: 91 / 191 / 107 px. Total `render` 21 s
    (termasuk threshold per klip atas 283 frame).
  - [2026-09-29] **DONE** — keputusan akhir Rio, detail + data lengkap di **D-009** (`docs/04`):
    Sapiens2-seg **0.8B fp16 GPU** (input 1024×768 stretch; full run 283 frame tanpa OOM/NaN, 15.85
    s/frame, 3276 MiB reserved — pengecualian VRAM ≤ 3 GB disetujui dengan syarat cek VRAM + resume per
    frame + fallback 0.4B fp16 untuk seluruh klip); post-processing N = 30 / K = 3 / M = 5; kedalaman
    **Depth Anything V2 Small** fp32 GPU (0.16 s/frame, 424 MiB); pointmap dan seg 1B dibuang; grup
    hair/face terpisah, Lower_Clothing di torso; target konten garis = `filtered_full_0.8b.mp4` tanpa
    bayangan/duplikat siluet. Environment Opsi 1a (cu118) → `CLAUDE.md`, `requirements.txt`,
    `requirements-lock.txt`; anggaran VRAM → D-005. Known issues → D-009, T-102b, T-201.
    Alat sekali pakai: `scripts/sapiens2_probe.py`, `sapiens2_exp.py`, `look_test.py`,
    `sapiens2_groups.json`, `sapiens2_classes.json`

### T-102b · `segment.py` · `DONE`
- **Kerjakan:** kontrak stage [2] `docs/01` (D-010): Sapiens2-seg `segment.model` eksplisit (default
  0.8B fp16) → `seg/classmap/*.png` (argmax, uint8) + `seg/probs/*.npz` (29 kelas, uint8, deflate,
  resolusi kerja; softmax di CPU) + `seg/manifest.json` + `seg/frames.jsonl`. Cek VRAM bebas sebelum load
  (3300 / 2300 MiB) → berhenti dengan pesan jelas; OOM → berhenti tanpa ganti model; resume per frame;
  manifest model beda → tolak kecuali `--restart`. Cek `argmax(probs)` vs `classmap`: laporkan % beda,
  bukan gagal keras. QC per klip (`qc_report.json`, termasuk `area_vs_median` median bergulir, `--qc-only`)
  — **T-301 digabung ke sini**. Nama kelas → data paket `src/rotoscope/data/sapiens2_classes.json`
- **Butuh dulu:** T-104a (config loader)
- **Done when:** `seg/` + `qc_report.json` terisi untuk klip uji; ukuran `probs` diukur di 20 frame
  pertama (tetap format ini kecuali > 3 GB per klip); default `qc.area_drop_min` dikalibrasi pada zona kaki
  hilang MediaPipe T-102a frame 1–7; `tests/test_segment.py` lolos
- **⚠️ Kritis:** rembg/u2net ditolak (D-008), jangan dipakai
- **Update log:**
  - [2026-09-25] Dulu T-102. Backend ditentukan T-102a — D-008
  - [2026-09-28] Backend → Sapiens2 (D-009), menunggu T-102c
  - [2026-09-29] T-102c DONE → backend seg 0.8B fp16 + fallback 0.4B fp16, pointmap dibuang (D-009).
    **Di awal T-102b: kontrak `docs/01` stage [2]/[4]/[5] ditulis ulang** (tanya Rio dulu), termasuk:
    - posisi stage kedalaman (Depth Anything V2 Small, fp32 GPU) di pipeline + format output di disk
    - post-processing seg sebagai parameter YAML: filter pulau kelas N = 30, mode filter peta grup
      K = 3 (background ikut), filter komponen garis M = 5; grup dari file (hair/face terpisah,
      Lower_Clothing di torso)
    - cek VRAM bebas sebelum stage [2] + berhenti dengan pesan jelas; resume per frame (lewati output
      valid); fallback 0.4B fp16 untuk SELURUH klip, tidak dicampur
    - data yang disimpan untuk stabilize (T-302 harus menstabilkan peta grup, bukan hanya mask biner):
      mis. probabilitas per grup, selain class map
  - [2026-09-29] sesi 1: kontrak pipeline ditulis ulang — `docs/01` (diagram, kontrak [2]/[2c]/[3]/[4]/[5],
    stack, struktur repo, urutan build), `docs/02` (parameter pipeline `default.yaml` + validasi), tabel
    Arsitektur `CLAUDE.md`, D-010. Keputusan Rio: probabilitas 29 kelas uint8 npz (Q1-A); grup di
    `default.yaml` (Q2-A); fallback pose ditunda, QC memberi bobot ke temporal (Q3-B); JSON bertipe +
    anchor + `track_id`, siluet termasuk lubang (Q4-B); model seg eksplisit, tanpa ganti otomatis. Task
    baru T-104a/b, T-105, T-106, T-107, T-201a/b, T-305; T-301 digabung ke sini. Status → WIP (sesi
    berikutnya: implementasi, setelah T-104a)
  - [2026-09-30] sesi 2: WIP — implementasi `segment.py` dimulai (T-104a DONE). Snapshot di cache HF
    (`scan_cache_dir`, offline): `facebook/sapiens2-seg-0.8b` `196a627b928676c4429b738ed76f78a21d96c4eb`
    (3157 MiB), `facebook/sapiens2-seg-0.4b` `449b3c5335e6722bb94990abdd1aa6e612432f22` (1551 MiB).
    Rencana (struktur modul, urutan per frame, offline/revision, VRAM/OOM, QC, entry point, test) menunggu
    approval Rio
  - [2026-09-30] sesi 2: implementasi + run klip uji. Keputusan Rio: `segment.revision` = mapping per model
    (dua repo = dua hash; `depth.revision` tetap string), jendela `area_vs_median` DIGESER (selalu W sampel),
    resume menolak SEMUA beda manifest, revision wajib commit hash 40-hex, `os.replace` + retry (Windows),
    exit code 0/1/3 (`docs/01` [2]). Cache HF: `sapiens2-seg-1b` + `sapiens2-pointmap-0.4b` dihapus
    (8.0 GB dibebaskan).
    - Kode: `src/rotoscope/segment.py` (backend Torch terpisah dari logika stage; entry point sementara
      `python -m rotoscope.segment`), `tests/test_segment.py` (backend palsu); revision di-pin di
      `config.py` / `default.yaml` / `docs/02`; `test_config.py` disesuaikan. Total 216 test lolos, 1 skip (GPU).
    - `--limit 20`: 16.51 s/frame (tanpa 3 warm-up; infer 16.2 s, softmax + kuantisasi CPU 0.16 s, tulis
      0.08 s), peak VRAM reserved 3276 / allocated 3014 MiB di semua frame, VRAM bebas sebelum load 3314 MiB
      (batas 3300 — margin 14 MiB). probs npz 0.10 MB/frame → ±41 MB per klip 360 frame (jauh < 3 GB → format
      Q1 tetap); classmap 7 KB/frame. Beda argmax(probs) vs classmap maks 0.0039% (mean 0.0013%).
    - Full run: resume melanjutkan dari `frame_00020` (frame ke-21; 20 dilewati), 263 frame 72.1 mnt (283
      frame total ±78 mnt; est. 360 frame ±99 mnt). NaN/inf 0, OOM 0. QC: 0/283 gagal; iou_prev min 0.713,
      area_ratio 0.080–0.208, big_blobs selalu 1, area_vs_median min 0.652, label_agreement_prev min 0.778.
    - Bug ditemukan saat run: print ringkasan QC crash (`UnicodeEncodeError`, stdout cp1252 saat diarahkan ke
      file) setelah `qc_report.json` tertulis → diperbaiki (`errors="replace"`) + test. Run ulang: 283
      dilewati tanpa GPU, QC OK, exit 0.
    - Kalibrasi `qc.area_drop_min` (definisi `area_vs_median` yang sama, W = 49): MediaPipe T-102a frame 1–7
      = 0.564–0.608 (maks frame 6 = 0.608; frame 0 0.634, frame 8 0.623); Sapiens2 min 0.652 (frame 197–202,
      pose asli: area turun 0.12 → 0.08). Default 0.6 TIDAK menangkap frame 6 MediaPipe. Celah aman
      (0.608, 0.652). Usulan 0.63 (tengah celah, tetap SEMENTARA — satu klip). Sensitif ke W: W = 25 celah
      (0.708, 0.757); W = 73 tidak ada celah (MediaPipe maks 0.597, Sapiens2 min 0.491). Batas metrik: zona
      buruk ≥ separuh jendela ikut menurunkan median → tidak tertangkap. Menunggu keputusan Rio
  - [2026-09-30] DONE — keputusan Rio: `qc.area_drop_min` = 0.63, tanda SEMENTARA tetap (satu klip; celah
    0.608–0.652) di `config.py` / `default.yaml` / `docs/02`, plus catatan `area_median_window` jangan diubah
    tanpa kalibrasi ulang. `docs/01` QC: batas metrik (zona buruk ≥ 25 frame tidak tertangkap) + celah per W.
    `docs/04` D-010: catatan "Hasil T-102b". `--qc-only` ulang: 0/283 gagal (area_vs_median min 0.652 ≥
    0.63); 216 test lolos, 1 skip (GPU)

### T-103 · `export.py` naif · `TODO`
- **Kerjakan:** PNG sequence → MP4 via ffmpeg, `-r 24 -pix_fmt yuv420p`. Input Phase 1 = foreground
  `stable/groups/*.png` (grup ≠ 0) sebagai siluet blok hitam-putih
- **Done when:** ada `out/animation.mp4` berisi siluet hitam-putih yang bergerak
- **Update log:**
  - [2026-09-29] Input diganti dari `masks/*.png` ke `stable/groups/*.png` — D-010

### T-104a · `config.py` — loader YAML + validasi · `DONE`
- **Kerjakan:** loader `configs/default.yaml` + `configs/styles/*.yaml`, default lengkap (jalan tanpa
  YAML), validasi range + aturan grup (`docs/02`), `paths.work_dir` bisa di drive lain
- **Done when:** config tanpa YAML = default `docs/02`; nilai di luar range / grup tidak valid → error
  jelas; `tests/test_config.py` lolos
- **Kenapa duluan:** T-102b dan stage lain membaca parameternya dari sini
- **Update log:**
  - [2026-09-29] Dipecah dari T-104 (loader dibutuhkan sebelum implementasi T-102b) — D-010
  - [2026-09-30] WIP — mulai. Rencana (sumber default, API, aturan, test) menunggu approval Rio
  - [2026-09-30] DONE — `src/rotoscope/config.py` (pyyaml + stdlib, `yaml.safe_load`),
    `configs/default.yaml`, `configs/styles/rough-sketch.yaml`, `tests/test_config.py` (150 test; total
    dengan test_ingest 169 lolos).
    - **Sumber default (opsi 2a = A):** frozen dataclass di `config.py`; `configs/*.yaml` dan dua blok YAML
      `docs/02` wajib identik — dicek test (termasuk urutan grup). Ubah default = ubah ketiganya.
    - **API:** `load_pipeline(path=None, overrides=None)`, `load_style(path=None, overrides=None)` →
      dataclass frozen (mapping read-only, `groups` = tuple berurutan); overrides = key bertitik
      (`{"segment.model": "0.4b"}`) untuk flag CLI T-104b; `section_hash(cfg, "groups")` = sha256 canonical
      JSON (urutan key YAML + int/float tidak berpengaruh; urutan grup berpengaruh, urutan kelas di grup
      tidak); `to_dict()`, `ensure_dir()`, `load_class_names()`, `project_root()`, `ConfigError`.
    - **Aturan:** key tidak dikenal / duplikat = error + saran; YAML parsial di-merge (`groups` diganti
      utuh); `null` hanya di `revision` (stage [2]/[2c] menolak `null` saat runtime — T-102b/T-105);
      `paths.*` relatif ke cwd, aset style relatif ke root project; path dengan karakter kontrol = error;
      load tanpa side effect.
    - Nama kelas dipindah → `src/rotoscope/data/sapiens2_classes.json` (package data di `pyproject.toml`);
      `scripts/sapiens2_probe.py` hanya baris `CLASSES_FILE` (`sapiens2_exp.py` memakai
      `probe.CLASSES_FILE`, tidak diubah).
    - `docs/02` diperbarui atas keputusan Rio: `paths.*` tanpa pembuatan folder saat load, contoh path Windows
      di blok YAML, `normalize` = `log_median_iqr` saja (T-302), baris baru bool / `model_ids` (prefiks
      `facebook/sapiens2-seg-`) / key `0.8b`/`0.4b` / `revision`, aturan path aset relatif root project

### T-104b · `cli.py` · `TODO`
- **Kerjakan:** CLI satu perintah jalankan pipeline (stage GPU [2] dan [2c] sebagai proses sendiri),
  tiap stage juga bisa dijalankan sendiri; flag `--seg-model`, `--restart`, `--qc-only`, `--config`
- **Done when:** `python -m rotoscope run samples/test.mp4` menghasilkan MP4
- **🎯 Milestone Phase 1:** pipeline end-to-end jalan
- **Update log:**
  - [2026-09-29] Dipecah dari T-104 (config loader → T-104a) — D-010

### T-105 · `depth.py` — Depth Anything V2 Small · `TODO`
- **Kerjakan:** kontrak stage [2c] `docs/01`: DA-V2 Small fp32 GPU (proses sendiri, revision di-pin,
  hanya varian Small) → `depth/*.npy` disparity mentah float16 resolusi kerja + `depth/manifest.json` +
  `depth/frames.jsonl`; cek VRAM ≥ 500 MiB bebas; resume per frame
- **Done when:** `depth/` terisi untuk klip uji, 0 NaN/inf; `tests/test_depth.py` lolos
- **Update log:**
  - [2026-09-29] Task baru — D-010

### T-106 · `stabilize.py` spasial saja · `TODO`
- **Kerjakan:** kontrak stage [3] `docs/01` dengan `stabilize.temporal.enabled: false`: probabilitas
  kelas → grup → argmax → filter pulau N = 30 (peta grup) → mode filter K = 3 → `stable/groups/*.png`;
  kedalaman dinormalisasi per frame → `stable/depth_smooth/*.npy`; `stable/manifest.json` (hash grup)
- **Done when:** `stable/` terisi; hash grup berubah → output basi terdeteksi; `tests/test_stabilize.py` lolos
- **Kenapa sekarang:** Phase 2 butuh input [3]; temporal ditambahkan di T-302/T-303
- **Update log:**
  - [2026-09-29] Task baru — D-010

### T-107 · Keluarkan rembg + onnxruntime dari requirements dan venv · `TODO`
- **Kerjakan:** hapus `rembg[cpu]` + `onnxruntime` dari `requirements.txt`, uninstall dari venv,
  perbarui `requirements-lock.txt`, `pip check` bersih, catatan stack `CLAUDE.md` / `docs/01`
- **Done when:** tidak ada rembg / onnxruntime di venv dan requirements; `scripts/smoke_test.py`
  disesuaikan + lolos, test lain lolos
- **Update log:**
  - [2026-09-29] Task baru — follow-up D-008 (T-102a)

---

# PHASE 2 — Vectorize + stylize basic

### T-201a · Siluet + lubang + batas grup · `TODO`
- **Kerjakan:** kontrak stage [4] `docs/01`: `cv2.findContours` mode `RETR_CCOMP` pada foreground
  `stable/groups` → `silhouette` (kontur luar ≥ `min_region_area`) + `silhouette_hole` (lubang ≥
  `min_hole_area`; ruang negatif tertutup wajib digambar); batas antar pasangan grup → `group_boundary`
  (polyline terbuka); filter komponen garis M = 5 sebelum thinning, tracing, `min_stroke_px`
- **Done when:** `work/contours/*.json` berisi polyline bertipe sesuai skema `docs/01`
- **Update log:**
  - [2026-09-29] Dipecah dari T-201; siluet termasuk lubang (bukan `RETR_EXTERNAL` saja);
    `approxPolyDP` pindah ke stage [5] — D-010

### T-201b · Garis oklusi kedalaman · `TODO`
- **Kerjakan:** kontrak stage [4] `docs/01`: dari `stable/depth_smooth` apa adanya (tanpa log kedua) →
  |grad| (Gaussian σ) → NMS → hysteresis T_high / T_low persentil per klip → hanya di dalam grup, jarak ≥
  D → skeleton ≥ L → `occlusion`; threshold per klip disimpan di `contours/clip_stats.json` (dipakai
  `--preview`)
- **Done when:** garis oklusi kaki menyilang (frame 73–92) muncul tanpa "bayangan" duplikat siluet
- **Update log:**
  - [2026-09-29] Dipecah dari T-201 (log T-201 lama di bawah) — D-010
  - [2026-09-29] Syarat dari T-102c (D-009, known issue): garis kedalaman DA mentah menghasilkan
    "bayangan" (duplikat sejajar siluet/batas seg + bercak tebal; `work/t102c/look/diag_sources.png`,
    frame 120: 4234 px garis DA vs 11808 px garis seg). Pembersihan wajib: tepi tipis (NMS searah
    gradien + hysteresis), hanya di DALAM satu grup dengan jarak minimum ke batas seg, panjang minimum.
    Titik awal dari look test (`scripts/look_test.py`): |grad log d| (Gaussian σ 1 px), T_high p95 /
    T_low p90 per klip, D = 7 px, L = 30 px (skeleton). Komponen DA ±94 px di area tangan (frame 90) tidak
    bisa dibedakan dari garis sah hanya dengan ukuran → evaluasi lagi di sini / T-302
  - [2026-09-29] Catatan T-102b sesi 1: garis dihitung dari `depth_smooth` (ternormalisasi di [3]), bukan
    disparity mentah; nilai T_high/T_low look test (atas |grad log d|) tidak langsung berlaku, persentil
    p95/p90 menyesuaikan diri; D / L / persentil dikalibrasi ulang di T-305 — D-010

### T-202 · Anchor + orientasi + `track_id` · `TODO`
- **Kerjakan:** aturan anchor `docs/01` stage [4]: `silhouette` searah jarum jam, `silhouette_hole`
  berlawanan, anchor = titik terdekat ke anchor track yang sama di frame sebelumnya (track baru: titik
  tertinggi hair ∪ face / titik tertinggi lubang); garis terbuka dinormalkan arahnya; `track_id` dari
  pencocokan dengan frame sebelumnya (`vectorize.track.max_match_dist_px`, kalibrasi di sini). Resample
  ke N titik dari anchor dikerjakan di [5]
- **Done when:** titik ke-0 dan `track_id` konsisten antar frame
- **⚠️ Pitfall P-004:** tanpa ini garis akan tampak "berputar"
- **Update log:**
  - [2026-09-29] Scope: anchor untuk semua tipe + `track_id` (seed jitter), resample pindah ke [5] — D-010

### T-203 · `stylize.py` garis polos · `TODO`
- **Kerjakan:** `approxPolyDP` → Catmull-Rom → resample dari anchor, render stroke tebal seragam,
  warna solid. Satu renderer untuk semua `type`, override `stroke.by_type`
- **Done when:** output sudah berupa outline, bukan siluet blok
- **Update log:**
  - [2026-09-27] Keputusan tertunda: tambah parameter `output_width` (mis. 1080) terpisah dari
    `working_width`. Output digambar ulang dari vector (D-001) → resolusi output tidak terikat
    sumber (video test 480 px). Tebal garis & jitter harus relatif terhadap ukuran output, bukan px
    absolut. Ini mengubah kontrak stage [5] → bahas & catat di decision log sebelum implementasi
  - [2026-09-29] Satu renderer + parameter per tipe; simplify/resample pindah dari [4] ke [5] — D-010.
    `output_width` tetap harus diputuskan sebelum implementasi

### T-204 · Flag `--preview N` · `TODO`
- **Kerjakan:** render hanya N frame untuk iterasi cepat. Threshold per klip dibaca dari
  `contours/clip_stats.json`, bukan dihitung dari N frame preview
- **Done when:** preview 10 frame selesai < 30 detik
- **Kenapa sekarang:** tanpa ini Phase 3–4 akan menyiksa
- **🎯 Milestone Phase 2:** outline keluar
- **Update log:**
  - [2026-09-29] Threshold dari `clip_stats.json` — D-010

---

# PHASE 3 — Stabilize (stage tersulit)

### T-301 · QC metrics · `SKIP`
- **Kerjakan:** hitung `area_ratio`, `iou_prev`, `component_count` per frame →
  `work/qc_report.json` + flag frame gagal
- **Done when:** report bisa menunjuk frame mana yang bermasalah
- **Update log:**
  - [2026-09-29] SKIP — digabung ke T-102b: QC dihitung di akhir stage [2] (termasuk `area_vs_median`)
    — D-010

### T-302 · Temporal EMA pada probabilitas grup + kedalaman · `TODO`
- **Kerjakan:** nyalakan `stabilize.temporal`: EMA pada probabilitas grup (bukan mask biner) dengan bobot
  `qc_fail_weight` untuk frame gagal QC; pilih + uji metode normalisasi kedalaman per frame
  (`stabilize.depth.normalize`) lalu EMA kedalaman. Saran: EMA dua arah (maju + mundur) karena offline →
  tanpa lag. Kalibrasi `qc_fail_weight`
- **Done when:** flicker acak berkurang terukur (`iou_prev` + `label_agreement_prev` peta grup rata-rata)
- **Update log:**
  - [2026-09-29] Objek diganti: probabilitas grup + kedalaman ternormalisasi (bukan mask biner) — D-010

### T-303 · Optical flow warp · `TODO`
- **Kerjakan:** `cv2.calcOpticalFlowFarneback` pada `frames/` → warp probabilitas grup + kedalaman
  frame sebelumnya ke frame sekarang → blend dengan bobot `optical_flow_blend`
- **Done when:** boiling turun signifikan tanpa lag terlihat
- **Catatan:** ini mitigasi paling efektif untuk P-001
- **Update log:**
  - [2026-09-29] Objek diganti: probabilitas grup + kedalaman ternormalisasi — D-010

### T-304 · Kalibrasi `boil_preserve` · `TODO`
- **Kerjakan:** render 3–4 varian nilai berbeda, bandingkan side-by-side, pilih
- **Done when:** ada nilai default yang kamu setujui secara artistik
- **Catatan:** ini penilaian mata, bukan metrik. Tidak bisa didelegasikan ke Claude Code
- **Update log:**

### T-305 · Kalibrasi ulang post-processing pada data stabil · `TODO`
- **Kerjakan:** cek ulang N (filter pulau kini di peta GRUP setelah temporal, di T-102c di peta KELAS
  sebelum grup), K, M, D, L, persentil `hi_pct`/`lo_pct` dan `vectorize.min_hole_area` pada output
  [3] dengan temporal aktif; jendela uji frame 73–92 + 183–202 (seperti T-102c)
- **Done when:** default di `configs/default.yaml` + `docs/02` diperbarui dengan data, disetujui Rio
- **🎯 Milestone Phase 3:** flicker terkendali
- **Update log:**
  - [2026-09-29] Task baru — D-010

---

# PHASE 4 — Style lengkap

### T-401 · Width modulation + taper · `TODO`
- **Kerjakan:** variasi tebal sepanjang path (`width_variation`, `width_noise_scale`),
  ujung menipis
- **Update log:**

### T-402 · Jitter deterministic · `TODO`
- **Kerjakan:** Perlin noise per titik, seed = `hash(frame_index, param_seed, track_id)`,
  `temporal_drift` untuk perubahan antar frame
- **⚠️ Pitfall P-007:** random murni = tidak reproducible, tidak bisa di-debug
- **Update log:**
  - [2026-09-29] Seed ditambah `track_id` (bukan indeks stroke) supaya pola getar tidak melompat saat
    urutan stroke berubah — D-010

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

# PHASE 5 — Fallback pose (DITUNDA, D-010)

### T-501 · `fallback_pose.py` · `BLOCKED`
- **Kerjakan:** MediaPipe Pose 33 landmark → mask sintetik dari capsule/polygon
  (torso, lengan, kaki, kepala)
- **⚠️ Cek dulu:** mediapipe 1.x — pastikan API Pose yang dipakai masih ada (legacy `mp.solutions`
  vs Tasks API `PoseLandmarker`)
- **Dibuka lagi kalau:** klip nyata gagal QC dan temporal berbobot QC (T-302) tidak cukup
- **Update log:**
  - [2026-09-29] BLOCKED — foreground Sapiens2 0 frame gagal QC di klip uji; frame gagal QC diisi dari
    frame tetangga lewat temporal berbobot di [3] (Q3-B) — D-010

### T-502 · Integrasi blend + QC · `BLOCKED`
- **Kerjakan:** panggil fallback hanya untuk frame gagal QC, blend dengan mask asli
  (jangan ganti total), catat frame mana yang pakai fallback
- **Done when:** frame motion-blur tidak lagi rusak, transisi tidak melompat
- **🎯 Milestone Phase 5:** pipeline tahan input buruk
- **Update log:**
  - [2026-09-29] BLOCKED — ikut T-501 — D-010

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
| 1 Skeleton | T-101 … T-107 (T-102 → a/b/c, T-104 → a/b) | 5/10 |
| 2 Vectorize | T-201 … T-204 (T-201 → a/b) | 0/5 |
| 3 Stabilize | T-301 … T-305 (T-301 SKIP) | 0/5 |
| 4 Style | T-401 … T-406 | 0/6 |
| 5 Fallback | T-501 … T-502 (BLOCKED) | 0/2 |
| 6 Opsional | T-601 … T-603 | 0/3 |

**Total: 36 task** · Selesai: 10/36 (SKIP tidak dihitung selesai)
