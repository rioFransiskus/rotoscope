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

### T-103 · `export.py` naif · `DONE`
- **Kerjakan:** PNG sequence → MP4 via ffmpeg, `-r 24 -pix_fmt yuv420p`. Input Phase 1 = foreground
  `stable/groups/*.png` (grup ≠ 0) sebagai siluet blok hitam-putih
- **Done when:** ada `out/<nama video sumber>.mp4` (`export.filename` = `{source}.mp4`) berisi siluet hitam-putih
  yang bergerak
- **Update log:**
  - [2026-09-29] Input diganti dari `masks/*.png` ke `stable/groups/*.png` — D-010
  - [2026-10-01] WIP — mulai. Rencana (sumber frame silhouette/strokes lewat satu jalur encode, pipe rawvideo
    ke ffmpeg, parameter `export.*` baru, audio default mati, nama file per klip, manifest + basi, verifikasi
    ffprobe, entry point, daftar test) menunggu approval Rio
  - [2026-10-01] Rencana disetujui + keputusan Rio: `export.filename` default `"{source}.mp4"`; file tujuan
    milik video sumber LAIN → ditolak; `audio: true` tanpa audio di sumber → error; manifest wajib memuat
    identitas klip (sha256 `meta.json` + `source_path`)
  - [2026-10-01] Implementasi + run klip uji. `src/rotoscope/export.py` (satu jalur encode lewat
    `FrameSource`; Phase 1 = `SilhouetteSource`), blok `export:` di `config.py` / `default.yaml` / `docs/02`
    (+ validasi), `tests/test_export.py` (31 test, ffmpeg nyata) + 14 kasus validasi di `test_config.py`.
    - **Klip uji (283 frame, 480×854):** `out/test.mp4` 497.6 KiB (509 571 B), encode + ffprobe 2.4 s.
      ffprobe: 283 frame, 24 fps, h264, yuv420p, 480×854, 11.792 s, 0 stream audio. Run ulang → dilewati
      (up-to-date). `export.audio: true` (file uji dihapus lagi) → 1 stream aac, 283 frame, 688.2 KiB.
    - **Semua test:** 369 lolos, 2 skip (GPU).
    - **Penyimpangan kecil dari rencana (keamanan):** file tujuan ada TANPA manifest → ditolak (`--restart`
      menimpa), bukan ditimpa dengan peringatan — asalnya tak diketahui, sejalan dengan pengaman klip lain.
      `--restart` tidak melewati pengaman klip lain. `filename` tidak ikut hash parameter (hanya nama).
    - **Status tetap WIP** — menunggu Rio menonton `out/test.mp4`
  - [2026-10-01] DONE — Rio menonton `out/test.mp4`: sesuai. Dua penyimpangan (file tujuan tanpa manifest
    ditolak; pengaman sumber lain tidak bisa dilewati `--restart`) disetujui. `docs/01` [6]: rincian yang
    ditetapkan; `docs/04` D-010: catatan "Hasil T-103" (keputusan audio + nama file + known issue identitas
    klip → T-108)

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

### T-108 · Identitas klip di manifest [2]/[2c]/[3] · `DONE`
- **Kerjakan:** identitas klip (sha256 isi `meta.json` + `source_path`, seperti manifest export T-103) masuk
  manifest [2]/[2c]/[3]. Klip berbeda di `work_dir` yang sama → stage GPU ([2], [2c]) **tolak** dengan pesan
  jelas; stage CPU ([3]) hitung ulang otomatis dengan peringatan (aturan manifest beda, prinsip #4 `docs/01`)
- **Done when:** ingest klip B ke `work_dir` berisi output klip A → [2]/[2c] berhenti dengan pesan (exit 1);
  [3] berhenti kalau input [2]/[2c] milik klip lain (atau tanpa identitas), dan menghitung ulang otomatis
  hanya kalau manifest [3] sendiri yang beda; `--adopt` ([2]/[2c]) memigrasi manifest lama tanpa inferensi;
  test per stage (termasuk ingest nyata klip kedua); `docs/01` kontrak [2]/[2c]/[3] diperbarui
- **Kenapa:** manifest [2]/[2c]/[3] tidak memuat identitas klip. Kalau klip lain di-ingest ke `work_dir` yang
  sama, resume [2] menganggap output klip lama valid → hasil salah tanpa error (known issue, `docs/04` D-010)
- **Sebelum:** T-104b (cli.py; selesai — layout akhir `work/clips/<stem>/`). Mitigasi sementara: `work_dir` per klip
- **Update log:**
  - [2026-10-01] Task baru — temuan Rio saat T-103 (manifest export sudah memuat identitas klip)
  - [2026-10-01] WIP — Tahap 1 (rencana) disetujui Rio dengan revisi (identitas beda → hanya `--restart`;
    `--adopt` = pernyataan pengguna; cek identitas sebelum `resolve_revision`/load backend)
  - [2026-10-01] Tahap 2 selesai: `meta.json` terbukti byte-deterministik (`test_ingest`; 13 field, tanpa ukuran/hash
    file sumber); helper identitas di `stage_common.py` (`clip_identity`, `require_same_clip`, `adopt_identity`);
    `clip` di manifest [2]/[2c]/[3]; `--adopt` di segment/depth; export memakai helper + cek `stable` manifest;
    `tests/test_clip_identity.py` baru. 410 test lolos
  - [2026-10-01] Tahap 3 pada klip uji (283 frame): `segment --adopt` 8.45 s, `depth --adopt` 0.70 s (tanpa
    inferensi); [3] hitung ulang otomatis + peringatan, 566 file `groups`/`depth_smooth` sha256 IDENTIK dengan
    sebelumnya; export encode ulang sekali (`stable.created_utc` berubah; `clip` export lama kompatibel), run
    berikutnya dilewati. Simulasi klip lain di salinan work_dir: [2]/[2c] exit 1 (juga `--limit`, `--qc-only`,
    `--adopt`), [3] menolak input klip lain lalu hitung ulang setelah input searah, export menolak menimpa hasil
    sumber lain (exit 1, MP4 utuh). Cadangan manifest lama: `work/_backup_T108/`. Status tetap WIP
  - [2026-10-01] Pengecekan tambahan (Rio): jalur positif pada data asli — `segment --qc-only` exit 0 (0/283
    gagal) dan `depth` exit 0 (283 dilewati, 0 diproses, model tidak dimuat); diff manifest vs cadangan: hanya
    `clip` + `adopted_utc` bertambah. Simulasi Tahap 3 membuat "klip lain" dengan menyunting `meta.json`
    (skrip) → ditambah test ingest nyata klip kedua (`test_real_second_ingest_into_workdir_with_clip_a_outputs`).
    411 test lolos
  - [2026-10-01] DONE — `docs/01` (Identitas klip + Batas yang diketahui, kontrak [2]/[2c]/[3]/[6], tabel manifest),
    `docs/04` D-010 "Hasil T-108"

### T-104b · `cli.py` · `DONE`
- **Kerjakan:** CLI satu perintah jalankan pipeline (stage GPU [2] dan [2c] sebagai proses sendiri),
  tiap stage juga bisa dijalankan sendiri; flag `--seg-model`, `--restart`, `--qc-only`, `--config`
- **Done when:** `python -m rotoscope run samples/test.mp4` menghasilkan MP4
- **🎯 Milestone Phase 1:** pipeline end-to-end jalan
- **Update log:**
  - [2026-09-29] Dipecah dari T-104 (config loader → T-104a) — D-010
  - [2026-10-01] Catatan dari T-103: `work_dir` per klip (mis. `work/<stem>/`; layout akhir `work/clips/<stem>/`) — `work_dir` tunggal membuat klip
    saling menimpa dan manifest [2]/[2c]/[3] belum membedakan klip (T-108). `export.filename` default
    `{source}.mp4` sudah aman untuk beberapa klip di `out/`
  - [2026-10-01] Catatan dari T-108: manifest [2]/[2c]/[3] kini memuat identitas klip; `cli.py` wajib meneruskan
    `--adopt` ke stage [2] dan [2c] (`segment.adopt_segment` / `depth.adopt_depth`; tolak kombinasi dengan
    `--restart` / `--limit` / `--qc-only` / `--download`, exit 1)
  - [2026-10-01] WIP — mulai. Tahap 1 (rencana + daftar test) menunggu approval Rio
  - [2026-10-01] Tahap 1 disetujui dengan revisi Rio: `--work-dir` di 4 `main()` stage (menang atas config; `out_dir`
    tetap); restart mengikuti graf dependensi + `--yes` wajib untuk penghapusan GPU; pre-flight CPU-only
    (video, bentrok folder kerja, bentrok target export); ingest di `run` = opsi A (selalu ulang); layout
    `work/clips/<stem>/` (tanpa daftar nama dicadangkan); `cli.qc_fail_warn_ratio` ditolak (tanpa parameter YAML baru)
  - [2026-10-01] Tahap 2: `src/rotoscope/cli.py` + `__main__.py`; `--work-dir` + `cli_cmd` di `stage_common.py`
    (+ 4 `main()` stage); pesan error stage memakai perintah CLI lengkap (`python -m rotoscope segment "<video>"
    --restart --yes`; `download` menggantikan `--download` di pesan); `tests/test_cli.py` (59 test) + penyesuaian regex
    `test_clip_identity` / `test_segment` / `test_depth`. F1: `work/_backup_T108/` dihapus (5 file, 109 963 B);
    F2: `T108-prompt.md` tidak ada di root (tidak ada yang dihapus)
  - [2026-10-01] Tahap 3 — **F3** (klip uji → `work/clips/test/`): sebelum = sesudah = **1 706 file, 551 533 281 B**
    (`frames` 283 / 51 208 350; `meta.json` 1 / 369; `qc_report.json` 1 / 115 512; `seg` 568 / 33 995 634; `depth` 285 /
    232 152 582; `stable` 568 / 234 060 834). Rencana Tahap 1 tertulis 1 703 file — salah jumlah (283+1+1+568+285+568
    = 1 706); byte cocok. `work/` tidak berisi path absolut `work_dir` (hanya `source_path` video → tetap cocok setelah
    dipindah). Di lokasi baru, tanpa GPU: `segment --qc-only` exit 0 (10.1 s, 0/283 gagal), `depth` exit 0 (8.6 s, 283
    dilewati), `stabilize` exit 0 (1.7 s, 283 dilewati), `export` exit 0 (0.4 s, up-to-date). Tidak disentuh: `t102c`,
    `t106`, `ab_t102a`, `samples/test.mp4`, `out/test.mp4`, `out/test.export.json`
  - **(a)** `run samples/test.mp4` setelah F3: exit 0, 33.2 s, semua dilewati tanpa GPU (ingest 0.8 s, segment 23.6 s
    [283 valid dilewati, QC 0/283], depth 7.2 s, stabilize 1.0 s, export 0.1 s up-to-date); `meta.json` hasil ingest
    ulang byte-identik (identitas klip lolos). **F4:** `samples/test_short.mp4` = 5 s pertama `test.mp4`, h264 + aac,
    493 068 B; 30 fps → 119 frame pada 24 fps (rencana 120)
  - **(b0)** smoke `run samples/test_short.mp4 --limit 3` (VRAM bebas 3324 MiB, batas 3300): exit 0, 90.5 s. ingest 0.4 s;
    segment 0.8b 75.6 s (3 frame: 27.89 s [load model] / 16.31 / 16.31 s, peak reserved 3276 MiB, beda argmax
    0.0012 / 0.0015 / 0.0020 %, QC dilewati karena `--limit`); depth 13.4 s (3 frame: 3.095 / 0.189 / 0.188 s, peak
    424 MiB, NaN/inf 0); stabilize 0.4 s; export `test_short.limit3.mp4` 0.2 s (ffprobe: 3 frame, 24 fps, h264
    yuv420p 480×854, 0.125 s, 7.2 KiB / 7 359 B, 0 stream audio). File itu dihapus setelah bersih (F6)
  - **(b1)** `run samples/test_short.mp4` penuh (VRAM bebas 3324 MiB): exit 0, wall **1 943.8 s** (±32.4 mnt).
    ingest 119 frame 0.5 s; segment 0.8b **116 diproses, 3 dilewati (dari smoke)** 1 895.6 s (stage 1 897.1 s;
    16.20 s/frame), peak reserved 3276 MiB, QC **0/119 gagal** (semua metrik 0 gagal → tanpa peringatan); depth 116
    diproses + 3 dilewati 31.6 s (stage 32.8 s), peak 424 MiB, NaN/inf 0; stabilize 116 + 3 dilewati 12.0 s; export 119
    frame 1.0 s. ffprobe `out/test_short.mp4`: h264, yuv420p, 480×854, 24 fps, 119 frame (`-count_frames`), 4.958 s,
    183 671 B (179.4 KiB), 0 stream audio. Run kedua `test_short.mp4`: exit 0, 21.1 s, semua dilewati (segment 13.0 s,
    depth 6.8 s, stabilize 0.4 s, export 0.1 s); `run samples/test.mp4`: exit 0, 32.2 s, semua dilewati
  - **Deviasi F4:** `test_short.mp4` punya audio aac (re-encode `-c:a aac`) dan 119 frame; TIDAK dibuat ulang tanpa audio
    karena `meta.json` akan berubah → identitas berubah → GPU ulang ±33 mnt. Audio tidak berpengaruh (`export.audio`
    default false → MP4 tanpa audio)
  - **Cek tambahan Rio** (data asli, tanpa GPU): (1) `run samples/test_short.mp4 --restart-from segment` dan `segment
    samples/test_short.mp4 --restart` tanpa `--yes` → keduanya exit 1; pesan memuat "241 file, 11.9 MiB, ulang ≈ 32.7
    mnt GPU (119 frame × 16.5 s)"; `work/clips/test_short` sebelum = sesudah = 722 file / 230 914 088 B. (2) salinan
    `test_short.mp4` di folder temp (stem sama, path beda) → `run` exit 1 di pre-flight sebelum ingest (pesan menyebut
    kedua path, tanpa `--restart`); `work/clips/test_short` 722 / 230 914 088 B dan `out/test_short.mp4` sha256
    `a91fd701…5f0e` tidak berubah. (3) salinan yang sama dengan `paths.work_dir` → folder temp KOSONG (config temp) →
    hanya pengaman export terpicu, exit 1 sebelum ingest; folder temp tetap 0 file. File temp dihapus (`tmpcheck/`
    berisi `cfg.yaml` 152 B + `copy/test_short.mp4` 493 068 B + folder kosong `emptywork/`; skrip pengukur 410 B)
  - **Integritas test + mutation check:** `py_compile tests/test_cli.py` OK; tanpa BOM, 0 CRLF (584 LF, sama dengan file
    test lain), tanpa karakter kontrol, satu newline di akhir. Blok `test_reingest_same_clip_passes_changed_video_is_
    rejected` sempat ditambahkan lewat heredoc shell (melanggar aturan CLAUDE.md "Edit/Write, bukan shell"; isi sudah
    diperiksa, tidak diulang). Mutation check (skrip luar, `mock.patch` pada `require_same_clip` segment / depth,
    kode produksi tidak diubah): test GAGAL (`DID NOT RAISE StageError`) saat segment+depth, segment saja, atau depth
    saja dimatikan; lolos saat normal → test tidak kosong
  - **Checklist → test** (`tests/test_cli.py`): WAJIB 1 restart + `--yes` + pesan CLI penuh = `test_restart_from_scope`,
    `test_restart_from_gpu_requires_yes`, `test_restart_from_cpu_does_not_need_yes`, `test_subcommand_restart_requires_yes`,
    `test_preview_lists_files_and_estimate`, `test_real_restart_from_segment_leaves_depth`,
    `test_real_restart_from_depth_leaves_segment`, `test_real_restart_from_ingest_deletes_everything`,
    `test_real_restart_without_yes_deletes_nothing`, `test_stage_errors_use_full_cli_command` (segment, depth),
    `test_stabilize_input_errors_use_full_cli_command`. WAJIB 2 pre-flight = `test_preflight_missing_video`,
    `test_preflight_same_name_other_folder` (tanpa / dengan `--restart-from`; tidak ada stage, subprocess, penghapusan),
    `test_preflight_same_video_other_spelling_is_not_a_clash`, `test_preflight_export_target_of_other_video`,
    `test_preflight_export_target_same_video_ok`. WAJIB 3 ingest opsi A =
    `test_reingest_same_clip_passes_changed_video_is_rejected` (+ `test_run_order_and_clip_workdir`). WAJIB 4 layout =
    `test_run_order_and_clip_workdir`, `test_config_workdir_and_unsafe_stem`. TAMBAHAN 5 (F3) = verifikasi manual di atas.
    6 proses anak = `test_interrupt_terminates_child_and_waits`, `test_exception_terminates_child`. 7 SystemExit
    stage = `test_cpu_stage_argparse_exit_is_named`. 8 `--limit` = `test_flags_forwarded` + smoke (b0). 9 grep rujukan
    = Tahap 4 (di bawah). 10 download = `test_download_default_and_fallback_without_forcing_offline`,
    `test_download_stops_on_failure`. Lainnya: urutan + subprocess = `test_run_order_and_clip_workdir`,
    `test_gpu_stages_are_subprocess_with_inherited_streams`; exit 1/3 = `test_stage_failure_propagates_exit_code`;
    `--seg-model` = `test_flags_forwarded`, `test_seg_model_never_changes_silently`; tanpa torch =
    `test_torch_not_imported_by_cli`; `--adopt`/`--qc-only` = `test_run_rejects_adopt_and_qc_only`,
    `test_subcommand_passthrough_flags`; QC = `test_qc_warning_once_and_run_continues`, `test_no_qc_warning_when_clean`;
    `--work-dir` per stage (#20) = `test_work_dir_flag_segment_and_depth`, `test_work_dir_flag_stabilize`,
    `test_work_dir_flag_export_keeps_out_dir`; hint resume = `test_resume_hint_after_restart_from_drops_restart`
  - **Tahap 4:** grep seluruh repo untuk `python -m rotoscope.` dan "entry point sementara": `docs/01` (semua kontrak
    stage + bagian baru "CLI"), docstring `segment/depth/stabilize/export/ingest.py`, komentar `config.py`, `CLAUDE.md`
    (cara pakai), `docs/04` D-010 "Hasil T-104b". Entri log lama T-102b / T-105 / T-106 / T-103 di sini tetap
    (catatan sejarah). Suite penuh: **470 lolos, 2 skip (GPU)**
  - [2026-10-01] DONE — 🎯 **milestone Phase 1 tercapai:** `python -m rotoscope run samples/test_short.mp4` dari nol
    menghasilkan `out/test_short.mp4` (119 frame, ffprobe sesuai); fixture cepat Phase 2: `work/clips/test_short/` +
    `out/test_short.*` (0.8b, JANGAN dihapus)

### T-105 · `depth.py` — Depth Anything V2 Small · `DONE`
- **Kerjakan:** kontrak stage [2c] `docs/01`: DA-V2 Small fp32 GPU (proses sendiri, revision di-pin,
  hanya varian Small) → `depth/*.npy` disparity mentah float16 resolusi kerja + `depth/manifest.json` +
  `depth/frames.jsonl`; cek VRAM ≥ 500 MiB bebas; resume per frame
- **Done when:** `depth/` terisi untuk klip uji, 0 NaN/inf; `tests/test_depth.py` lolos
- **Update log:**
  - [2026-09-29] Task baru — D-010
  - [2026-09-30] WIP — mulai. Snapshot di cache HF (`scan_cache_dir`, offline):
    `depth-anything/Depth-Anything-V2-Small-hf` `5426e4f0f36572d16453bbda7a8389317b1bef99` (94.6 MiB; hanya
    `config.json`, `preprocessor_config.json`, `model.safetensors` — model card `README.md` TIDAK ada di cache,
    lisensi belum bisa dicek offline). Arsitektur di config: `DepthAnythingForDepthEstimation`, backbone
    hidden_size 384 (ViT-S = Small). Processor `DPTImageProcessor` 518, `keep_aspect_ratio`, kelipatan 14 →
    input 924×518 untuk frame 854×480 (T-102c). Rencana (kode bersama dengan `segment.py`, urutan per frame,
    NaN/inf, frames.jsonl, manifest, entry point, uji regresi vs T-102c, test) menunggu approval Rio
  - [2026-09-30] implementasi + run klip uji. Keputusan Rio: helper bersama diekstrak ke
    `src/rotoscope/stage_common.py` (nama umum — dipakai juga stage CPU [3]/[4]/[5]) dan `segment.py`
    di-refactor memakainya (`SegmentError`/`SegmentOOMError` = alias error bersama); `README.md` DA-V2 Small
    diunduh (online, `hf_hub_download`) → front-matter `license: apache-2.0`, ikut `--download` + cek cache
    runtime, dicatat di manifest; cek backbone hidden_size 384 (lapis kedua di samping nama "Small");
    NaN/inf → 0 + `finite: false`, tidak diproses ulang (kalimat resume baru untuk `docs/01` [2c]).
    - Kode: `src/rotoscope/depth.py` (entry point sementara `python -m rotoscope.depth [--restart] [--limit N]
      [--download]`, exit code 0/1/3), `stage_common.py`, `tests/test_depth.py`, `tests/test_stage_common.py`;
      `depth.revision` di-pin di `config.py` / `default.yaml` / `docs/02` (`test_config.py`: satu assert);
      `docs/01` struktur repo + `stage_common.py`. Total 274 test lolos, 2 skip (GPU); test GPU depth nyata lolos.
    - Bukti `segment.py` tidak berubah: `test_segment.py` tidak diubah, lolos; `--qc-only` → `qc_report.json`
      identik dengan sebelum refactor (kecuali `created_utc`); `segment --limit 2` ke work_dir sementara →
      classmap + probs frame 0–1 identik bit per bit dengan `work/seg/` (beda maks 0 langkah uint8), manifest sama.
    - `--limit 20`: 0.186 s/frame (infer 0.169 s, tanpa 3 warm-up), peak VRAM reserved 424 / allocated 291 MiB,
      VRAM bebas sebelum load 3314 MiB (batas 500). Input processor terukur 924×518 (H×W) untuk frame 854×480
      (`DPTImageProcessor`). npy 820 KB/frame → ±295 MB per klip 360 frame.
    - Full run: resume melanjutkan dari `frame_00020` (frame ke-21), 263 frame 60.2 s (0.190 s/frame, p95 0.204);
      NaN/inf 0, OOM 0, peak reserved 424 MiB di semua frame. Disparity per frame: min 0.64–1.40, median
      1.44–1.87, maks 3.84–6.93.
    - Regresi vs T-102c (`work/t102c/exp/depth_da2s_gpu/`): **283/283 frame identik bit per bit** (selisih
      maks 0) — revision, processor, fp32 dan post-process GPU sama. Status tetap WIP, menunggu konfirmasi Rio
  - [2026-09-30] DONE — konfirmasi Rio. `docs/01` [2c]: validasi lisensi tiga lapis, urutan per frame + ukuran
    input terukur 518×924, kalimat resume/NaN baru, isi manifest/frames.jsonl, entry point + exit code.
    `docs/04` D-010: catatan "Hasil T-105"

### T-106 · `stabilize.py` spasial saja · `DONE`
- **Kerjakan:** kontrak stage [3] `docs/01` dengan `stabilize.temporal.enabled: false`: probabilitas
  kelas → grup → argmax → filter pulau N = 30 (peta grup) → mode filter K = 3 → `stable/groups/*.png`;
  kedalaman dinormalisasi per frame → `stable/depth_smooth/*.npy`; `stable/manifest.json` (hash grup)
- **Done when:** `stable/` terisi; hash grup berubah → output basi terdeteksi; `tests/test_stabilize.py` lolos
- **Kenapa sekarang:** Phase 2 butuh input [3]; temporal ditambahkan di T-302/T-303
- **Update log:**
  - [2026-09-29] Task baru — D-010
  - [2026-09-30] WIP — mulai. Rencana (probabilitas grup + aturan seri, filter pulau + mode filter pada peta
    grup, normalisasi kedalaman `log_median_iqr` + 2 parameter baru, `stable/manifest.json` + perilaku basi,
    resume, entry point, uji regresi vs T-102c, daftar test) menunggu approval Rio
  - [2026-09-30] implementasi + run klip uji. Keputusan Rio: (1) seri eksak argmax grup → grup dari
    `seg/classmap` kalau ikut seri, selain itu id terkecil (`seg/classmap` jadi input [3]; piksel seri dicatat per
    frame); (2) parameter baru `stabilize.depth.log_eps` 1e-6 (1e-12–0.01) + `iqr_min` 0.01 SEMENTARA (1e-3–1,
    T-302) di `config.py` / `default.yaml` / `docs/02` + validasi; (3) manifest beda → aturan umum: stage GPU
    ([2]/[2c]) tolak + `--restart`; stage CPU murah deterministik ([3]/[4]) output basi dihapus + dihitung ulang
    otomatis dengan peringatan (field lama → baru); output tanpa manifest tetap ditolak; (4) `stable/frames.jsonl`
    masuk kontrak [3].
    - Kode: `src/rotoscope/stabilize.py` (entry point sementara `python -m rotoscope.stabilize [--config]
      [--restart] [--limit N]`, exit 0/1), `tests/test_stabilize.py` (46 test), `tests/test_config.py` (+4 kasus
      validasi). Filter pulau + mode filter = algoritma T-102c, dijalankan di peta GRUP. Total 324 test lolos, 2 skip (GPU).
    - `--limit 20`: 0.10 s/frame (baca 0.03, grup 0.07, depth 0.01, tulis 0.01). Output: groups PNG ±6.6 KB/frame
      (±2.4 MB per 360 frame), depth_smooth npy 820 KB/frame (±295 MB per 360 frame).
    - Full run: resume melanjutkan dari `frame_00020` (263 frame, 32.1 s, 0.119 s/frame, p95 0.138) → ±43 s per
      klip 360 frame. Seri 5–43 px/frame, pulau berubah 0–128 px, mode berubah 9–88 px. depth_smooth semua finite,
      283/283; per frame min −20.2…−2.8, median −13.2…−1.6 (median klip −3.7, termasuk background jauh), maks
      0.50…2.56; IQR log foreground 0.077–0.384 (tidak ada clamp, tidak ada frame fg kosong, depth_finite 283/283).
    - Regresi vs T-102c (pulau N=30 di peta KELAS `work/seg/classmap` → grup → mode K=3), script sekali pakai di
      scratchpad: piksel sama mean **99.989%**, min **99.932%** (frame 232); di foreground (A ∪ B) mean 99.935%, min
      99.636%; IoU foreground min 0.9966. **100% piksel beda berada ≤ 2 px dari batas grup** (0 beda di dalam area);
      rata-rata 45 px/frame; komponen beda terbesar 67 px (frame 233, bg → torso di tepi bawah), median komponen
      terbesar per frame 6 px. Video: `work/t106/compare_f073-092.mp4`, `work/t106/compare_f183-202.mp4`,
      `work/t106/compare_f225-240.mp4` (zona beda terbesar, frame 232–233; permintaan Rio)
      (+ `work/t106/regress_t102c.json`). Status tetap WIP, menunggu penilaian visual Rio
  - [2026-09-30] DONE — penilaian visual Rio: garis grup T-106 setara T-102c di ketiga video, tanpa bentuk aneh di
    frame 232–233; cek |grad| `stable/depth_smooth` di foreground (p99, tanpa threshold, seperti panel 4 T-102c)
    frame 73–92 (`work/t106/depth_grad_f073-092.mp4`): batas kaki kanan–kiri masih terlihat. `docs/01`: prinsip #4
    (aturan manifest beda GPU vs CPU) + rincian [3] (input `seg/classmap`, aturan seri, pulau/mode, normalisasi +
    background, `stable/frames.jsonl`, resume, manifest basi, entry point, waktu + disk terukur). `docs/04` D-010:
    catatan "Hasil T-106"

### T-107 · Keluarkan rembg + onnxruntime dari requirements dan venv · `DONE`
- **Kerjakan:** hapus `rembg[cpu]` + `onnxruntime` dari `requirements.txt`, uninstall dari venv,
  perbarui `requirements-lock.txt`, `pip check` bersih, catatan stack `CLAUDE.md` / `docs/01`
- **Done when:** tidak ada rembg / onnxruntime di venv dan requirements; `scripts/smoke_test.py`
  disesuaikan + lolos, test lain lolos
- **Update log:**
  - [2026-09-29] Task baru — follow-up D-008 (T-102a)
  - [2026-10-02] DONE — opsi B′ (keputusan Rio). Dicopot 14 paket: rembg, onnxruntime, imageio, jsonschema,
    jsonschema-specifications, lazy-loader, llvmlite, numba, pooch, pymatting, referencing, rpds-py, scikit-image,
    tifffile. 6 paket generik (attrs, charset-normalizer, platformdirs, protobuf, requests, urllib3) sengaja
    dibiarkan, bisa dicopot setelah Phase 2 jalan. Paket 79 → 65; lock 76 → 62 baris (14 dihapus, 0 ditambah /
    diubah); venv 6207,9 → 5987,6 MiB (−220,3 MiB); cache `~/.rembg` (167,84 MiB) dihapus. Tidak ada dependensi
    yang perlu ditambah (`opencv-contrib-python` tetap eksplisit). Verifikasi: `pip check` bersih (= baseline);
    import rembg / onnxruntime → `ModuleNotFoundError`; torch 2.7.1+cu118 (CUDA 11.8, `is_available()` True),
    transformers 5.17.0, numpy 2.4.6, opencv 5.0.0, scipy 1.17.1 tidak berubah; `smoke_test.py` 13 PASS (tanpa
    Pose; MediaPipe `ImageSegmenter` 103 ms/frame); impor kelas transformers, huggingface_hub,
    `cv2.ximgproc.thinning`, scipy, PIL, yaml OK; `download` (cache hit) exit 0; `ab_segment.py` / `look_test.py`
    lolos `py_compile` + impor; suite penuh 470 lolos / 2 skip; `run test_short.mp4` + `run test.mp4` exit 0,
    semua dilewati, sha256 `out/*.mp4` identik. `docs/01`, `CLAUDE.md`, `docs/04` ("Hasil T-107") diperbarui;
    anotasi arsip di `scripts/ab_segment.py`

---

# PHASE 2 — Vectorize + stylize basic

### T-201a · Siluet + lubang + batas grup · `DONE`
- **Kerjakan:** kontrak stage [4] `docs/01`: `cv2.findContours` mode `RETR_CCOMP` pada foreground
  `stable/groups` → `silhouette` (kontur luar ≥ `min_region_area`) + `silhouette_hole` (lubang ≥
  `min_hole_area`; ruang negatif tertutup wajib digambar); batas antar pasangan grup → `group_boundary`
  (polyline terbuka); filter komponen garis M = 5 sebelum thinning, tracing, `min_stroke_px`
- **Done when:** `work/clips/<stem>/contours/*.json` berisi polyline bertipe sesuai skema `docs/01`
- **Update log:**
  - [2026-09-29] Dipecah dari T-201; siluet termasuk lubang (bukan `RETR_EXTERNAL` saja);
    `approxPolyDP` pindah ke stage [5] — D-010
  - [2026-10-02] DONE — `src/rotoscope/vectorize.py` (CPU, tanpa torch) + subperintah `python -m rotoscope vectorize <video>
    [--restart] [--limit N]` (belum di `run` sampai T-203). Keputusan Rio: `anchor` / `track_id` tidak ditulis (key strok
    `{type, closed, groups, points}`); padding 1 px (kontur tertutup, garis tepi frame → T-203); [4] belum masuk `run`.
    Filter ukuran = jumlah piksel (bukan `contourArea`); koordinat = pusat piksel (i + 0.5); `contours/manifest.json` baru
    (`contract` `T-201a`, `algo_rev` 2, hash hanya 4 parameter T-201a, identitas klip, basi otomatis). Pelacak batas grup:
    junction = **crossing number ≥ 3** (bukan ≥ 3 tetangga: versi pertama hanya mencakup 21,0% / 22,7% skeleton), `prune_redundant`,
    klaster junction + titik wakil, **thinning Guo-Hall** (Zhang-Suen mengikis diagonal 45°; penyimpangan dari Opsi 1, `docs/04`).
    Terukur (`test_short` 119 / `test` 283 frame): 29–30 ms/frame (p95 34–35 ms, 0 frame > 1 s), JSON 4.4 / 11.0 MB; cakupan
    skeleton 99,83% / 99,74%, cakupan band 99,79% / 99,71%; titik berulang 0, loncatan 0; hash `contours/frame_*.json` identik
    antar run dari nol. Lubang mentah → lolos: 292 → 203 / 703 → 470; komponen luar dibuang 3 / 45; loop `group_boundary`
    25 / 57; foreground menyentuh tepi bawah 119/119 dan 283/283. Test: `tests/test_vectorize.py` (69 kasus) +
    `tests/skeleton_metrics.py` (metrik cakupan permanen) + 2 di `test_cli.py`; suite penuh 541 lolos / 2 skip (470 → 541);
    15 test pelacak gagal bila detektor lama dipasang kembali (plugin pytest di luar repo). Alat: `scripts/contour_overlay.py`.
    Penilaian visual Rio (overlay `_fix1` + PNG kasus terburuk): garis hijau kontinu, tidak memendek / terlepas, tanpa garis
    ganda / cabang liar / lingkaran kecil; frame 156 (celah 9 px) tidak mengganggu; siluet + lubang OK; lubang dibuang
    (`min_hole_area` 200) tidak ada yang seharusnya digambar; garis magenta di tepi frame wajar. Batas yang diketahui (`docs/04`
    "Hasil T-201a"): cakupan skeleton buta terhadap pengikisan thinning (pelengkap: cakupan band); 2 / 6 frame < 98% (komponen
    terisolasi 15–17 px); `docs/01` [4] + [5] + CLI, `docs/04` D-010 "Hasil T-201a" diperbarui

### T-201b · Garis oklusi kedalaman · `DONE`
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
  - [2026-10-03] DONE **dengan batas kaki** (keputusan Rio) — `src/rotoscope/vectorize.py` (CPU, tanpa torch): strok `occlusion`
    (`closed: false`, satu grup, `strength`, loop = titik akhir = titik awal) + `contours/clip_stats.json` (pass 1 atas SEMUA
    frame, juga dengan `--limit`); manifest `contract` `T-201b`, `algo_rev` tetap 2, `vectorize` datar 10 kunci
    (4 T-201a + 6 `depth_lines.*`), `depth_thresholds`; syarat D memakai jarak L2 presisi dan menganggap **tepi frame sebagai
    batas** (disengaja, `docs/01`). Pelacak T-201a dipakai ulang (hanya `thin_band` diekstrak): hash strok tipe lama identik.
    Terukur (`test_short` 119 / `test` 283 frame): T_high / T_low 0,1200 / 0,0532 dan 0,1389 / 0,0641; 71 / 67 ms/frame (p95 78 / 75 ms,
    0 frame > 1 s); JSON 4.5 / 11.2 MB; cakupan skeleton 99,70% / 99,51%, band 99,83% / 99,70%; titik berulang 0, loncatan 0;
    bayangan minimum 7,000 px = D; strok oklusi/frame min / median / maks 0 / 1 / 5 dan 0 / 0 / 5. **Done-when 4/5 di kedua klip**
    (frame 78 gagal): `left_leg` / `right_leg` berisi 0 piksel (Lower_Clothing di `torso`, D-009), jadi kaki menyilang hanya muncul
    sebagai garis oklusi di dalam `torso`. Tepi tumpang tindih kaki ada di `depth_smooth` tetapi di bawah T_low default; normalisasi
    [3] (log + IQR per frame) ikut melemahkannya di frame 73–83; ambang p80 / p70 memberi 5/5 tetapi tambahannya sebagian besar
    lipatan celana. Penilaian visual Rio: garis oranye lengan melipat sah; tidak ada bayangan. Test: `tests/test_vectorize_occlusion.py`
    (56 kasus, sintetis non-sumbu-sejajar + data nyata) + `tests/test_vectorize.py` diperbarui + `tests/skeleton_metrics.py`;
    suite penuh 598 lolos / 2 skip (541 → 598); mutation check (D, tepi frame, NMS, hysteresis kuat, L, pemisahan per grup, junction lama)
    semuanya membuat test gagal. Alat: `scripts/contour_overlay.py` diperluas (oklusi oranye, panel |grad|, `--worst`, ringkasan
    JSON). Detail, sensitivitas (`work/t201b/sensitivity_*.json`), diagnostik kaki, dan batas yang diketahui: `docs/04` D-010
    "Hasil T-201b"
  - ⚠️ **Risiko untuk fase berikut:** (1) frame tanpa oklusi di default **34/119** (`test_short`) dan **160/283** (`test`; run kosong
    terpanjang 28 frame; pengukuran Tahap 1 29/119 dan 149/283 — selisih karena Guo-Hall vs Zhang-Suen, `docs/04`) dan antar frame
    berkedip karena temporal [3] belum aktif → **evaluasi ulang setelah T-302 / T-303, sebelum kalibrasi T-305**; (2) L = 30 dalam
    piksel Guo-Hall (≈ 38 piksel Zhang-Suen); (3) komponen DA ±94 px di area tangan (frame 90) belum bisa dibedakan dari garis sah

### T-202 · Anchor + orientasi + `track_id` · `DONE`
- **Kerjakan:** aturan anchor `docs/01` stage [4]: `silhouette` searah jarum jam, `silhouette_hole`
  berlawanan, anchor = titik terdekat ke anchor track yang sama di frame sebelumnya (track baru: titik
  tertinggi hair ∪ face / titik tertinggi lubang); garis terbuka dinormalkan arahnya; `track_id` dari
  pencocokan dengan frame sebelumnya (`vectorize.track.max_match_dist_px`, kalibrasi di sini). Resample
  ke N titik dari anchor dikerjakan di [5]
- **Done when:** titik ke-0 dan `track_id` konsisten antar frame
- **⚠️ Pitfall P-004:** tanpa ini garis akan tampak "berputar"
- **Update log:**
  - [2026-09-29] Scope: anchor untuk semua tipe + `track_id` (seed jitter), resample pindah ke [5] — D-010
  - [2026-10-03] DONE — `src/rotoscope/track.py` (baru, CPU, tanpa torch) + `vectorize.py` (frame diproses BERURUTAN):
    orientasi (luas bertanda; silhouette + loop searah jarum jam, lubang berlawanan), anchor diputar (`points[0]` = anchor, key
    `anchor` konstan 0; [5] membaca `points[0]`), arah garis terbuka = kesinambungan (B; track baru = aturan statis), `track_id`
    = Chamfer simetris + penugasan optimal (mulai 1), pendekatan Y (silhouette terbesar mewarisi id). Manifest `contract`
    `T-202`, `pending` `[]`, hash + `track.max_match_dist_px`, **`ALGO_REV` tetap 2** (fitur baru = `contract` naik; keputusan
    Rio: `algo_rev` hanya untuk perbaikan perilaku pada kode yang sudah dikontrak). Rantai kesinambungan: key level-frame
    `prev_sha256` (sha256 byte frame sebelumnya) + pemeriksaan frame-pengganti; `--limit N` = prefiks run penuh; stage [4] masih
    di luar `run` (T-203). Perubahan default `max_match_dist_px` **12 → 16** (disetujui Rio; `config.py`, `configs/default.yaml`,
    `docs/02` dalam satu langkah). Skema + aturan: `docs/01` [4]; keputusan, angka, kalibrasi, batas: `docs/04` D-010 "Hasil T-202"
  - **Kriteria lulus (disetujui Rio sebelum implementasi) dan hasil akhir (Y@16), `test_short` / `test`:** **(a)** 100% silhouette
    searah jarum jam dan 100% lubang berlawanan (juga loop searah) — 100% / 100%; **(b)** silhouette utama = satu `track_id`
    sepanjang klip — 1 / 1 (tanpa Y klip `test`: 11); **(c)** lompatan anchor silhouette utama antar frame: median ≤ 3 px, p95 ≤ 12 px,
    maks ≤ 24 px — median 1,0 / 2,24, p95 **6,0 / 9,5**, maks 10,05 / 19,0 (tanpa Y klip `test`: maks 65 px); **(d)** pembalikan arah garis
    terbuka nol untuk track hidup ≥ 5 frame (kecuali loop) — 0 / 0; **(e)** churn dilaporkan (id baru per frame per tipe, umur track
    median / p90 / maks; hanya silhouette utama yang berkriteria keras); **(f)** tidak ada dua strok dalam satu frame yang memakai
    `track_id` sama — 0 / 0 frame; **(g)** hash kanonik semua field identik dengan T-201b untuk keempat tipe — identik / identik. Metrik
    tambahan (Rio): ambiguitas = persentase padanan dengan rasio jarak kandidat terbaik kedua / terbaik < 1,5; ketidakcocokan ukuran =
    panjang kedua strok berselisih > 3×; bobot strok-frame di track berumur ≥ 5 / ≥ 10 frame
  - **Kriteria pemilihan ambang `max_match_dist_px`** (dibandingkan X = ambang global, Y = X + silhouette terbesar mewarisi id,
    Z = ambang per tipe — hanya diajukan; sensitivitas {6, 8, 12, 16, 20, 24}; tabel di `docs/04`): (1) silhouette utama satu id di
    kedua klip dan anchor lulus (c) tanpa bergantung satu kejadian — **kriteria 11(b) sendiri tidak boleh menentukan ambang** (ambang
    global, kunci `hair+torso` 82% / 56% frame dan `torso+arm` 58–80% sudah ambigu; solusi yang hanya bertumpu pada lengan terlepas
    frame 213–266 rapuh untuk gerak lebih cepat); (2) id baru turun / fraksi cocok naik; (3) ambiguitas dan selisih ukuran tidak naik
    berlebihan; (4) tanpa parameter baru. Hasil: Y menyelesaikan silhouette utama di semua ambang; ambang dipilih dari
    `group_boundary` (12 → 16 id baru −22%, 16 → 20 −11%, 20 → 24 −2%; ambiguitas +0,7 poin ke 16) → **Y + 16**. Penilaian visual Rio:
    anchor tetap di puncak kepala, warna `track_id` bertahan, panah arah konsisten, fragmen lengan terpisah wajar sementara
  - **Mutation check** (plugin pytest di luar repo; kode produksi tidak dimodifikasi) — tujuh mutan, semuanya membuat test
    terkait GAGAL: (1) normalisasi orientasi mati; (2) kesinambungan arah mati; (3) pencocokan rakus pengganti optimal;
    (4) **pemeriksaan rantai mati** (`prev_sha256` + frame-pengganti: gagal di `test_chain_b`, `test_chain_c`, `test_chain_d[edit_valid_json]`,
    `test_successor_agrees_helper`); (5) hanya pemeriksaan frame-pengganti mati; (6) tanda sumbu tidak dikunci; (7) anchor
    kesinambungan mati (selalu titik tertinggi; gagal di test dua puncak dan test anchor hair ∪ face). Test rantai: (a) frame k-1
    dihitung ulang byte-identik → k TIDAK dihitung ulang; (b) k-1 berubah → k..N dihitung ulang, hasil akhir identik dengan run
    penuh; (c) k valid tetapi k-1 hilang / berubah → k tidak valid; (d) frame dipotong / diubah isinya (JSON valid) → dipulihkan
  - **Estimasi waktu:** median 74–93 ms/frame antar run (target ≤ 150 ms; tidak ada frame > 1 s; maks 135 ms), pelacakan sendiri
    median 8–11 ms (maks 26 ms), pass 1 ambang 1,3 s (119 frame) / 2,7 s (283 frame); ukuran JSON 4,57 / 11,26 MB (+0,7% dari
    T-201b); biaya terburuk hitung ulang berantai (frame 0 berubah) ≈ jumlah frame × ±80 ms (±23 s untuk 283 frame). Suite penuh
    **698 lolos / 2 skip** (598 → 698)
  - **Batas yang diketahui** (rincian `docs/04`): titik awal garis terbuka melompat sampai 60–127 px saat strok memanjang /
    memendek; padanan di zona 12–16 px (bukan bukti tertukar; 4 dari ±3500 padanan bergeser oleh penugasan optimal); id oklusi
    berkedip lahir-mati (T-302 / T-303 / T-305); fragmen lengan terpisah mendapat id baru; frame terakhir klip yang diubah tangan
    tidak terdeteksi rantai; Y tak bersyarat. Alat: `scripts/contour_overlay.py --track` (warna per `track_id`, anchor, panah arah,
    `--compare`, `--label`); keluaran `work/t202/` (ter-ignore)

### T-203a · `stylize.py` garis polos (stage [5], subperintah `stylize`) · `DONE`
- **Kerjakan (aktual, T-203a):** penghalusan Gaussian (`shape.smooth_px`) → `approxPolyDP` → Catmull-Rom (adaptif) → ekstensi ujung di tepi →
  render garis tebal seragam solid (SVG string manual + PNG supersampling). **Tanpa resample** (penyimpangan dari "resample dari
  anchor": docs/04 "Keputusan T-203" butir 8; resample arc-length dari `points[0]` = T-401 / T-402). Satu renderer untuk semua
  `type`, override `stroke.by_type.*.width_scale`. Kontrak aktual: docs/01 [5]; parameter: docs/02
- **Status ⚠️ dari T-201a / T-202:**
  - ✅ **Run titik di tepi frame** (terjawab): **hide** (keputusan Rio, dinilai visual di frame 233 dan 195 `test`: tubuh tampak
    terpotong frame dengan rapi); `shape.edge_mode: "draw"` tersedia
  - ➡️ **Urutan `run`:** dipindah ke **T-203b** (di bawah)
  - ⏳ **`group_boundary` loop** = `closed: false` dengan titik akhir = titik awal (25 di `test_short`, 57 di `test`): digambar sebagai
    tertutup tanpa takik; **taper loop (taper ujung tidak boleh menipiskan sambungan) ditunda ke Phase 4 (T-401)**
  - Konvensi koordinat pusat piksel: skala `output_width` langsung, x' = s · x (`docs/01` [4])
  - ⚠️ **Titik awal garis terbuka melompat 60–127 px saat strok memanjang / memendek** (T-202, terukur Y@16: 7–13% strok-frame
    memilih ujung berbeda dari aturan statis; lompatan titik awal p50 3–4,5 px, p95 18–23 px, maks 97 / 127 px; sebabnya bentuk
    strok berubah, bukan arah terbalik). Jitter 1D berbasis panjang busur dari `points[0]` akan "pop". Pertimbangkan jitter yang
    dikunci posisi (mis. noise 2D / fase dari titik referensi tetap) atau yang tidak bergantung pada titik awal. Untuk strok
    tertutup `points[0]` = anchor (stabil: median 1–2 px, maks 19 px silhouette utama). [5] membaca `points[0]` (bukan `anchor`) dan
    mengabaikan `prev_sha256`
- **Done when:** output sudah berupa outline, bukan siluet blok — **tercapai** (penilaian visual Rio 2026-10-03: kemulusan sesuai, tepi
  rapi, takik cekung dan sambungan OK; video: getar antar frame dan garis oklusi berkedip wajar sementara — temporal [3] belum aktif)
- **Hasil akhir (default produksi):** ε 2,8 ref, `smooth_px` 5,0, `width_base` 9,0, `edge_mode` hide, `output_width` 1080 (1080×1922).
  Terukur: median 125–128 ms/frame, 14–30 MiB per klip, memori puncak ±127 MiB. Suite **774 lolos / 2 skip** (698 → 774: 53 + 16
  test `test_stylize*.py`, + test config). **Batas yang diketahui:** (a) overshoot spline di takik cekung / sudut tajam (deviasi maks
  per strok p50 5,8 / p95 8,7 / maks 15,0 px output pada silhouette; backlog di bawah); (b) approxPolyDP tidak menjamin ≤ epsilon;
  (c) tebal raster dikuantisasi grid supersampling (galat posisi ≤ 0,3 px, tebal ≤ 0,4 px per kasus); (d) garis oklusi berkedip dan
  getar antar frame (temporal [3]); (e) titik awal jalur tertutup tidak dipertahankan; (f) `stroke.cap` hanya `round`; (g) validasi
  contours hanya untuk frame terpilih; (h) frame contours yang diubah tangan tanpa mengubah manifest tidak terdeteksi (resume
  stage ini memeriksa SVG / PNG, bukan isi contours)
- **Update log:**
  - [2026-10-03] **T-203a DONE.** Rincian di entri-entri di bawah (Tahap 1–3 + putaran visual) dan docs/04 "Hasil T-203a"
  - [2026-09-27] Keputusan tertunda: tambah parameter `output_width` (mis. 1080) terpisah dari
    `working_width`. Output digambar ulang dari vector (D-001) → resolusi output tidak terikat
    sumber (video test 480 px). Tebal garis & jitter harus relatif terhadap ukuran output, bukan px
    absolut. Ini mengubah kontrak stage [5] → bahas & catat di decision log sebelum implementasi
  - [2026-09-29] Satu renderer + parameter per tipe; simplify/resample pindah dari [4] ke [5] — D-010.
    `output_width` tetap harus diputuskan sebelum implementasi
  - [2026-10-03] **T-203 dipecah** T-203a (stage [5] garis polos + subperintah `stylize`) dan T-203b (integrasi `run` / DAG /
    export). Keputusan Rio 1–7: "Keputusan T-203, 2026-10-03" di docs/04. **Log Tahap 1 T-203a (rencana disetujui Rio, lengkap):**
    - **Poin 4 — parameter baru** (tiga tempat: YAML, `config.py`, docs/02; `default.yaml` tidak berubah): `render.output_width` (int
      genap 256–2160, default 1080; batas atas supaya mask supersampling ≤ 75 MiB) dan `shape.edge_mode` (`hide` | `draw`, default
      `hide`). Konstanta struktural bernama di modul (`REF_WIDTH = 1080`, bit sub-piksel, margin ekstensi tepi, batas kontak dangkal,
      kompresi PNG), tanpa magic number. Default look test dikonversi × 2,25 (look test menggambar di px KERJA).
    - **Poin 6 — pipeline geometri:** skala `x' = s·x` (titik = pusat piksel) → penanganan tepi → approxPolyDP (epsilon px output =
      nilai ref × unit) → Catmull-Rom (`smooth_tension`, `spline_steps`; rapat otomatis untuk segmen panjang) → polyline rapat ke
      renderer. **Tanpa resample** di render. Strok tertutup: `points[0]` = anchor, spline periodik, jalur ditutup tanpa takik. Loop
      terbuka (titik akhir = titik awal): titik penutup dibuang, diperlakukan tertutup. Garis oklusi digambar apa adanya (tanpa
      interpolasi antar frame); frame tanpa strok = hanya kertas (valid).
    - **Poin 9 — SVG:** string manual dari titik yang SAMA dengan raster (1 desimal); `width` / `height` / `viewBox` = ukuran output;
      latar `<rect fill=paper.color>`; satu `<g>` per tipe (id tetap) urut silhouette, silhouette_hole, group_boundary, occlusion
      (urutan z tidak berpengaruh: tinta solid satu warna); satu `<path>` per strok (`M … L … [Z]`), `stroke-width` konstan,
      `stroke-linecap` / `stroke-linejoin` `round`, `fill="none"`; tanpa id acak / timestamp (byte-deterministik). Opasitas diabaikan.
    - **Poin 14 — alat visual:** `scripts/strokes_preview.py` (pakai ulang `export.encode`) → `work/t203a/`: (a) video pratinjau
      rentang 73–92, 183–202, 225–240 (+ frame strok terbanyak / jalur terpanjang) kedua klip; (b) hide vs draw (frame 233 + tepi
      kanan); (c) tebal garis `width_base` {3,6; 5,4; 7,2; 9,0} px ref pada frame 80 dan 233; (d) PNG kasus terburuk (deviasi
      terbesar, strok terpanjang, run tepi terpanjang, strok terbanyak, kontak tepi paling dangkal, sudut kanvas); (e) crop zoom 3×
      di sambungan, ujung terbuka, tepi bawah; (f) epsilon {2,8; 5,6} ref pada frame yang sama. Label "garis polos T-203a - belum ada
      jitter / taper / tekstur".
    - **Poin 15 — berkas dan papan:** hasil visual di `work/t203a/` (ter-ignore, dipertahankan); sementara di scratchpad; skrip
      `scripts/strokes_preview.py` masuk git; `T203a-prompt.md` dihapus terakhir (Tahap 4). Phase 2 = 6 task (T-201a, T-201b, T-202,
      T-203a, T-203b, T-204), total 38; setelah T-203a DONE: Phase 2 = 4/6, total 20/38.
    - **Tambahan Rio (Tahap 1):** konversi default disetujui dengan ukuran epsilon 5,6 / 8,0 ref; `resample_points` tidak aktif
      (T-401 / T-402 butuh resample arc-length dari `points[0]`); SVG manual; hash style = parameter aktif; tebal garis Tahap 3
      {3,6; 5,4; 7,2; 9,0} px ref; spline uniform vs centripetal diukur (usulan saja); kontak tepi dangkal < 45° → tegak lurus tepi;
      ujung terbuka dekat tepi tanpa celah ke tepi kanvas; ukur waktu + ukuran PNG dan deviasi akor SVG (≤ 0,1 px output, kalau
      lebih naikkan kerapatan sampling dari sumber yang sama); `SUPPORTED_CONTOURS_CONTRACTS = {"T-202"}`.
  - [2026-10-03] **Log Tahap 2–3 T-203a (angka terukur; status tetap WIP sampai Rio menilai visual):**
    - **Waktu per frame (median, kedua klip serupa; 1080×1922, ss 3):** geometri 24–25 ms, mask 22 ms, compose 27 ms (tabel
      LUT; float32 awal 115–124 ms), encode PNG 41–42 ms (`cv2.imencode`, level 3), SVG 4 ms, tulis berkas 5 ms; total p50 125 /
      128 ms, p95 134 / 137 ms, maks 140 / 148 ms (`test_short` / `test`), 0 frame > 0,3 s. PNG level 1 / 3 / 6 / 9 = 40 / 40 /
      54 / 108 ms dan 68,6 / 65 / 37 / 34,5 KiB per frame (level 3 dipakai). Memori puncak 126,6 MiB. Ukuran: PNG median 67,0 /
      70,8 KiB (total 7,79 / 19,03 MiB), SVG median 37,3 / 39,8 KiB (total 4,18 / 10,43 MiB).
    - **Metrik (a)–(c), angka:** (a) kesetiaan geometri, deviasi maks per strok (px output, ε 5,6 ref; p50 / p95 / maks; `test_short`
      | `test`): silhouette 8,9 / 16,1 / 22,5 | 9,2 / 16,3 / 21,5; silhouette_hole 5,6 / 9,6 / 15,9 | 6,0 / 11,5 / 18,6;
      group_boundary 3,8 / 7,0 / 10,8 | 3,9 / 7,9 / 18,7; occlusion 3,4 / 5,4 / 7,6 | 3,3 / 5,4 / 8,6; ε 2,8 ref: silhouette
      5,4 / 8,0 / 15,5 | 5,3 / 7,9 / 15,3; ε 8,0 ref: 12,0 / 20,3 / 28,9 | 12,9 / 19,7 / 29,6; deviasi akor polyline vs spline
      sebenarnya maks 0,041 px (p95 0,036; target ≤ 0,1). (b) penyelarasan: garis 0° / 30° / 45° / 60°, ss 3 dan 4: galat posisi per
      kasus ≤ 0,3 px output (kuantisasi grid ss; keputusan Rio ±0,25), galat tebal ≤ 0,4 px; rata-rata atas 16 posisi sub-piksel
      acak: posisi ≤ 0,05 px, tebal ≤ 0,1 px. (c) tepi (hide), seluruh frame: jarak titik tengah run tepi ke garis tengah jalur
      terdekat minimum 10,8 px (`test_short`) / 3,2 px (`test`) (draw ≤ 0,01 px); 336 / 1152 ujung diperpanjang, 0 yang tidak
      mencapai tepi kanvas (cakupan ≥ 0,99); kontak dangkal (< 45°) 31 / 336 dan 183 / 1152, sudut terkecil 1,8° / 0,0°; 0 ujung
      tepat di sudut kanvas; 0 strok seluruhnya di tepi; jalur ≥ strok di semua frame.
    - **Mutation check, 8 dari 8 mutan membuat test terkait GAGAL (plugin pytest di luar repo, kode produksi tidak diubah):**
      (1) **pembuangan run tepi mati** → 9 test gagal (hide ×3 di tepi bawah / kanan / dangkal, draw-vs-hide, strok seluruhnya di
      tepi, stale, data nyata ×2); (2) perpanjangan keluar kanvas mati → 5; (3) offset koordinat −0,5 piksel ss hilang → 2
      (rata-rata galat posisi sub-piksel, ss 3 dan 4); (4) komposisi union diganti per strok → 1; (5) SVG memakai titik berbeda dari
      raster → 1; (6) koreksi isian tepi poligon (`FILL_BIAS_SS`) dihapus → 3; (7) sampling spline adaptif dimatikan → 1; (8) aturan
      kontak dangkal dirusak (selalu tegak lurus) → 1 (test unit aturan ekstensi).
    - **Metrik kesetiaan + toleransi (keputusan Rio: epsilon BUKAN batas):** metrik = jarak tiap titik kontur asli terskala
      (titik di tepi dikecualikan pada mode hide) ke polyline akhir terdekat (jarak segmen eksak, `stylize_metrics.stroke_deviation`),
      diringkas per strok (maks) dan pooled (p95). approxPolyDP tidak menjamin deviasi ≤ epsilon: terukur pada ε 5,6 ref, jarak
      tegak lurus ke GARIS akord (cara OpenCV sendiri mengukur; bukan jarak ke segmen) melebihi epsilon sampai 7,33 px (+31%) pada
      22 / 496 strok tertutup dan 3 / 3163 strok terbuka; strok tertutup yang dibuka (+ titik awal) dan diputar titik awalnya tetap
      melebihi (11 / 209 dan 13 / 209), jadi bukan efek segmen-vs-garis, bukan efek penutupan, bukan titik awal. Contoh: `test_short`
      frame 24 strok 0 (tertutup, 1366 titik): OpenCV menyisakan 35 titik; rantai titik 728 → 905 (178 titik) berjarak maks 6,27
      px > 5,6 dari garis akordnya, sedangkan Douglas-Peucker acuan pada rantai yang sama memecahnya di 1 titik tambahan.
      Mekanisme di dalam OpenCV tidak diidentifikasi (kode sumbernya tidak dibaca). Toleransi test: sintetis = 0,35 × tebal garis
      (lingkaran terukur 1,56 px vs epsilon 1,33 px); data nyata = ambang regresi dari angka terukur di atas (silhouette 30,
      hole 22, group_boundary 22, occlusion 12 px output; bukan fungsi epsilon).
    - **Peringatan mode draw bertebal tipis:** `draw_width_warning` — satu baris stderr per run (menyebut tebal per tipe dan syarat
      minimum `draw_mode_min_width` = s), bukan error; test `test_draw_mode_thin_width_warns_once_on_stderr_not_error`.
    - **Backlog (TIDAK dikerjakan di T-203a): spline di sudut tajam.** Overshoot Catmull-Rom seragam di takik cekung tajam
      (maks ±22 px output pada ε 5,6, di tengah silhouette). Opsi: (i) centripetal (alpha 0,5): kurang menang jelas — silhouette p95
      16,1 → 12,2 tetapi p50 8,9 → 9,7; group_boundary maks 10,8 → 13,0 (`test_short`); (ii) tangen nol di titik bersudut tajam
      (fitur baru, butuh ambang sudut); (iii) epsilon lebih kecil (ε 2,8: silhouette maks 15,5). Diputuskan Rio sesudah penilaian
      visual.
  - [2026-10-03] **Putaran visual 2 Rio → default `smooth_px` 5,0 dan `stroke.width_base` 9,0** (docs/04 butir 12; tepi bawah hide,
    takik / sambungan OK). Terukur (kedua klip; silhouette deviasi maks per strok p50 / p95 / maks px output): **5,8 / 8,7 / 15,0**
    (smooth 3,0: 5,6 / 8,2 / 11,4); luas silhouette rata-rata −0,008% (terburuk −0,09%); kelok 86,8 °/100 px dan balik kelengkungan
    1,58/100 px (smooth 3,0: 94,0 / 1,67); stabilitas excess silhouette 0,74 px; waktu geometri 38 ms. Determinisme (hash
    `test_short` 7636be00…, `test` 0a328fe9…), resume, stale, input tidak berubah: lolos; suite 774 lolos / 2 skip.
  - [2026-10-03] **Backlog (TIDAK dikerjakan), tuas berikutnya bila Rio masih menilai garis bergelombang:** (1) naikkan
    `shape.smooth_px` (prototipe 5,0: kelok 86,8 °/100 px dan balik kelengkungan 1,57 vs 94,0 / 1,67 pada 3,0; deviasi p50
    silhouette 5,8 vs 5,6 px; luas terburuk −0,09%; tanpa kode baru); (2) spline di sudut tajam (centripetal atau tangen nol di
    titik bersudut, lihat backlog di atas); (3) penghalusan sesudah approx / `smooth_tension` lebih kecil. Pemetaan tahap waktu:
    "gambar" = mask 22 ms, "downsample" = INTER_AREA + tabel warna (compose) 27 ms, "encode PNG" 41–42 ms, "tulis berkas" 5 ms.
  - [2026-10-03] **Keputusan visual Rio → produksi:** `shape.simplify_epsilon` 2,8 ref + **`shape.smooth_px` 3,0** (Gaussian
    arc-length sebelum approxPolyDP; 0 = mati; aktif, masuk hash style; docs/04 butir 11). Implementasi `stylize.smooth_polyline`
    (re-sample 1 px ref, sudut > 60° pada ±6 sampel dikunci dan memecah jalur, ujung terbuka / titik silang tepi dikunci lewat
    pantulan ganjil, tertutup + loop periodik). Tes: `tests/test_stylize_smooth.py` (16; garis miring 30 / 45 / 60°, busur, sudut 90 /
    60 / 40°, sambungan tertutup, derau berkurang, determinisme); mutan penghalusan mati (7 test gagal), ujung bergeser (7),
    sudut dibulatkan (3). Terukur ulang (produksi, kedua klip; silhouette p50 / p95 / maks deviasi per strok px output): lama
    (ε 5,6, tanpa) 9,1 / 16,2 / 22,5; ε 2,8 tanpa 5,4 / 8,1 / 15,5; **final 5,6 / 8,2 / 11,4**; luas silhouette rata-rata −0,003%
    (terburuk −0,04%); kegelisahan (kelok °/100 px, balik kelengkungan/100 px) 114,5 / 1,89 → **94,0 / 1,67**; stabilitas antar frame
    (excess chamfer silhouette) 0,79 → 0,77 px (tidak peka); waktu geometri 34,6 → 35,7 ms per frame. Determinisme, resume, stale,
    input tidak berubah: lolos (hash final `test_short` fba93116…, `test` 8a8ea41d…). Suite 774 lolos / 2 skip.

### T-203b · Integrasi stage [4] + [5] ke `run`, export `strokes` · `DONE`
- **Kerjakan:** (1) [4] vectorize dan [5] stylize masuk urutan `run` (`ingest → segment → depth → stabilize → vectorize → stylize →
  export`); `cli.STAGES`, `RESTART_SCOPE`, `_targets`, graf dependensi + tabel restart di docs/01 "CLI" (`--restart-from
  vectorize|stylize`, tanpa `--yes`); flag `--style` di `run`; (2) `export.source: "strokes"` diterima (sekarang ditolak oleh
  `export.make_source`): MP4 dari `strokes/*.png` apa adanya (ukuran output 1080×1922 genap, bukan resolusi kerja; verifikasi ffprobe
  memakai ukuran output), manifest export mencatat `strokes/manifest.json`; (3) salin `strokes/*.svg` ke `out/svg/` (docs/01 [6]);
  (4) validasi `export.source` dan konfigurasi style-vs-export di `config.py` / docs/02 bila perlu
- **⚠️ Tersisa dari T-203a:** (a) `export.filename` / `--limit` untuk jalur strokes (`<nama>.limitN.mp4`); (b) pre-flight (c) `run`
  untuk target export tidak berubah; (c) taper loop dan jitter terkunci posisi tetap Phase 4 (bukan T-203b); (d) estimasi waktu
  `run` ([4] ±80 ms + [5] ±130 ms per frame); (e) `--preview N` = T-204
- **Done when:** `python -m rotoscope run <video>` menghasilkan `out/<nama>.mp4` berisi garis polos dari [5] dan `out/svg/*.svg`
  ✅ (2026-10-05; `out/svg/<nama>/`)
- **Sisa / diteruskan:** `--preview N` + milestone Phase 2 = T-204; jitter / taper / tekstur = Phase 4; temporal garis oklusi = T-302/T-303;
  backlog: `segment` / `depth` ±21–28 s per `run` walau dilewati (impor torch + cek di subprocess)
- **Update log:**
  - [2026-10-03] Dipecah dari T-203 (keputusan Rio butir 6–7, docs/04 "Keputusan T-203")
  - [2026-10-05] **Tahap 2–3 selesai (menunggu penilaian visual Rio; Tahap 4 belum).** `run` = ingest → segment → depth →
    stabilize → vectorize → stylize → export (`--style` hanya ke stylize; `--restart-from vectorize|stylize|export` tanpa `--yes`);
    `export.source: strokes` (MP4 ukuran output 1080×1922, tag warna lengkap lewat `COLOR_TAGS` + `setparams`, manifest merujuk
    strokes/manifest.json, versi ffmpeg dicatat di luar hash); salinan `strokes/*.svg` → `out/svg/<nama>/`. Temuan: flag keluaran
    `-color_primaries/-color_trc` saja TIDAK menulis primaries/trc ke stream (ffmpeg 9.0.1) → `-vf setparams=...`; merah jenuh tanpa
    tag bergeser (+12, +15, −2) saat didekode bt709, palet kertas/tinta netral tidak membedakannya (≤ 1 level). Terukur (crf 18, 85
    frame sampel): PSNR min 41,40; MAE tinta maks 3,50; selisih maks 75; kertas 2,00; tinta datar 3,71. Ambang: PSNR ≥ 40,4, MAE
    tinta ≤ 4,2, selisih maks ≤ 90, kertas ≤ 3, tinta datar ≤ 4,5. **Akurat:** crf 23 terukur PSNR 39,60 / MAE 5,09 / selisih maks
    105 (gagal), kertas 2,00 (lolos), tinta datar 4,54 vs ambang 4,5 → hanya LOLOS TIPIS (selisih 0,04); metrik tinta datar BUKAN
    penjaga regresi crf — PSNR + MAE tinta + selisih maks yang menangkap crf 23. Waktu CPU (test_short / test): [4] 9,2 / 21,4 s,
    [5] 16,1 / 39,2 s, [6] 2,4 / 5,4 s; MP4/SVG/strokes/contours byte-identik antar run dari nol. Suite 828 lolos / 2 skip (837 / 2 setelah penanda SVG diperkuat, lihat di bawah).
  - [2026-10-05] **Insiden skenario (g) — dicatat sebagai pelajaran.** Skenario "folder SVG milik video lain" dijalankan lewat
    `run … --restart-from ingest --yes` pada salinan scratch (`paths.work_dir` = `…/scratchpad/t3work/work`, `paths.out_dir` =
    `…/scratchpad/t3work/out`). Penanda yang saya tulis ulang memakai BOM → tidak terbaca → dianggap "tanpa penanda" → pre-flight
    tidak memblokir → ingest + `segment` GPU 0.8b berjalan di salinan scratch; keempat proses dihentikan (VRAM kembali 197 MiB).
    Terverifikasi tidak ada efek ke repo: `out/test.mp4` sha 5fcb5e40…, `out/test_short.mp4` sha 8e7c1c39… dan hash pohon
    `out/svg/test` 4706de19… / `out/svg/test_short` 18ae8a26… identik sebelum dan sesudah; `work/clips/*` asli tidak tersentuh
    (hanya `qc_report.json` `created_utc`, perilaku stage [2] yang sudah ada). **Pelajaran:** (1) skenario destruktif / pre-flight
    HANYA lewat stage palsu (`cli.run_stage` diganti penjaga yang gagal bila terpanggil) atau `--restart-from vectorize`, tidak pernah
    `run --restart-from ingest|segment|depth --yes` — itu MENJALANKAN GPU; (2) jangan menulis ulang berkas penanda dari PowerShell
    (`Set-Content -Encoding utf8` menambah BOM); (3) kegagalan membaca penanda tidak boleh jatuh ke cabang "tanpa penanda".
  - [2026-10-05] **Penanda SVG diperkuat:** `read_svg_marker` toleran BOM (utf-8-sig; penanda ditulis tanpa BOM); penanda ADA tetapi
    tidak terbaca / bukan JSON / bukan objek / skema salah → berhenti dengan penyebab (bukan "tanpa penanda"); `--restart` hanya
    boleh bila SETIAP `*.svg` di folder identik byte dengan `strokes/` (tidak ada suntingan hilang), selain itu ditolak walau
    `--restart` dengan menyebut berkasnya; pre-flight `run` memeriksa kondisi yang sama (restart = export ikut ber-restart). Perilaku MP4
    tanpa manifest tidak diubah. Test baru + mutation check (BOM mati, rusak = tanpa penanda, cek identik dilewati) → test gagal.
  - [2026-10-05] **Penilaian visual Rio + penutup (Tahap 4):** kertas hangat benar, garis tajam setelah kompresi, tepi frame rapi,
    getar / garis oklusi berkedip wajar (temporal belum aktif), SVG di peramban sama dengan video → **default `export.source` = `strokes`**
    (`config.py`, `configs/default.yaml`, docs/02 satu langkah; test yang memerlukan silhouette memakai `export.source: silhouette`
    eksplisit). `run samples/test_short.mp4` dan `run samples/test.mp4` dengan config default: exit 0, export DILEWATI (up-to-date;
    sha256 MP4 / hash pohon SVG / strokes / contours identik). Dokumen: docs/01 (CLI, [4], [5], [6]), docs/02, docs/04 "Hasil T-203b",
    CLAUDE.md; papan Phase 2 = 5/6, total 21/38. Suite 837 lolos / 2 skip. `T203b-prompt.md` dihapus terakhir.
  - [2026-10-05] **Backlog (belum dikerjakan):** `segment` / `depth` memakai ±21–28 s per `run` walau semua frame dilewati (impor
    torch + pengecekan di subprocess) — pertimbangkan pengecekan resume tanpa impor torch di induk subprocess.

### T-204 · Flag `--preview N` · `DONE`
- **Kerjakan:** render hanya N frame untuk iterasi cepat. Threshold per klip dibaca dari
  `contours/clip_stats.json`, bukan dihitung dari N frame preview
- **Done when:** preview 10 frame selesai < 30 detik — terpenuhi untuk skenario HANGAT (`stable/` valid dan contours valid);
  kasus dingin dan `stable/` basi dicatat apa adanya di log di bawah (bukan "lulus semua")
- **Kenapa sekarang:** tanpa ini Phase 3–4 akan menyiksa
- **🎯 Milestone Phase 2 tercapai (2026-10-05):** outline keluar — `run <video>` menghasilkan MP4 + SVG garis polos dan `run --preview N`
  mengiterasinya dalam detik
- **Update log:**
  - [2026-09-29] Threshold dari `clip_stats.json` — D-010
  - [2026-10-05] **DONE.** Konfirmasi Rio: `out/test.preview_73-82.mp4` sama dengan bagian yang sama di `out/test.mp4` (0:03,04-0:03,42),
    warna kertas dan garis sama; (d) 28,6 s dan (d2) 31,3 s diterima dengan peringatan estimasi. Dokumen: docs/01 (CLI, kontrak
    stage, [6]), docs/04 "Hasil T-204", CLAUDE.md. Papan: Phase 2 = 6/6, total 22/38. Suite penuh 864 lolos / 2 skip. `T204-prompt.md`
    dihapus terakhir.
  - [2026-10-05] **Tahap 1–3 selesai (laporan sebelum konfirmasi Rio).** Keputusan di docs/04
    "Keputusan T-204". Kode: `cli.py` (`--preview N [--from K]`, `check_gpu_inputs`, `cmd_preview`, `vectorize_chain_todo`),
    `stylize.py` / `export.py` (`--from K --limit N`, `<nama>.preview_K-<K+N-1>.mp4`), `stage_common.window_bounds`, `stabilize.py`
    (pesan + estimasi; penyimpangan dicatat di docs/04). Test: `tests/test_preview.py` (27), `tests/preview_metrics.py`; suite penuh 864 lolos,
    2 skip.
  - **Waktu end-to-end** (`run <video> --preview N --from K`, klip test 283 frame, salinan scratchpad, 3 run; rata-rata, min–maks):

    | Skenario | Rata-rata | Min–maks | Bagian (s) |
    |---|---|---|---|
    | (a) K=73, N=10, semua valid | 4,17 | 4,08–4,25 | ingest 0,9 · stabilize 0,8 · vectorize 0,4 · stylize 0,0 · export 0,4 |
    | (b) K=73, N=10, strokes jendela hilang | 5,82 | 5,70–6,01 | stylize 1,8 |
    | (b) K=73, N=20 | 8,69 | 8,50–8,89 | stylize 3,5 |
    | (c) K=73, N=10, style basi | 6,35 | 5,98–6,66 | stylize 2,2 |
    | (d) K=225, N=10, contours + strokes dingin | 28,62 | 28,52–28,75 | vectorize 22,9 (18,4 pada Tahap 1; variasi mesin besar) |
    | (d2) seperti (d) + `clip_stats.json` + manifest dihapus (skenario tambahan) | 31,25 | 30,94–31,57 | vectorize 25,4 |
    | (e) K=0, N=10 | 3,72 | 3,71–3,74 | |
    | (f) `stable/` basi (PENGECUALIAN, stabilize penuh) | 51,77 | 51,69–51,90 | stabilize 35,7 · vectorize 11,1 · stylize 2,0 |

    Hangat < 30 s terpenuhi: (a), (b), (c), (e). Dingin K=225: 28,6 s (batas 30 s lolos tipis); 31,3 s bila `clip_stats.json` + manifest
    hilang; (f) 51,8 s di luar anggaran. Peringatan "rantai vectorize M frame belum valid, estimasi ±S s" dicetak sebelum [4] bila M > 0.
  - **Kesetaraan** (jendela 73-82, 225-234, 0-9; 30 berkas/jendela = contours JSON + strokes SVG + PNG): sha256 identik dengan run penuh
    di kedua arah; pohon contours/ dan strokes/ penuh sama dengan keadaan awal; MP4 preview: PSNR vs PNG sumber min 41,34 dB, tinta MAE
    maks 3,76 (ambang 40,4 / 4,2), PSNR vs MP4 utama min 44,11 dB; ffprobe h264 yuv420p 1080x1922 bt709/tv; `out/test.mp4`,
    `.export.json`, `out/svg/` (404) dan klip asli tidak berubah (hash). Tanpa subprocess GPU, `torch` tidak di `sys.modules`.
    Mutation check 5/5 membuat test gagal (seg/depth off 5, offset K 3, MP4 utama 3, subprocess GPU 3, K+N 1). Bukti: `work/t204/`.
  - **Tabel perilaku preview per stage** (poin 5):

    | Stage | Perilaku di `run --preview N --from K` |
    |---|---|
    | [1] ingest | selalu ulang (±0,8 s); `meta.json` byte-identik |
    | [2]/[2c] | TIDAK dijalankan; dicek CPU-only: manifest ada, identitas klip cocok, kunci model seg = config / `--seg-model`, frame jendela valid → bila tidak, exit 1 + perintah (`run <video>` penuh, atau `segment|depth <video> --limit K+N`) |
    | [3] stabilize | TANPA `--limit` (resume penuh; vectorize memvalidasi `stable/` semua frame). Valid ≈ 0,2–0,8 s; belum/basi → dihitung penuh (±0,11 s/frame, ±31 s untuk test) dengan baris estimasi. Tanpa seg/depth lengkap di luar jendela → stabilize berhenti (exit 1, stage + perintah) |
    | [4] vectorize | `--limit K+N`: rantai 0..K+N-1 (hasil = prefiks run penuh). `clip_stats.json` valid → pass 1 dilewati; basi/hilang → dihitung. Peringatan estimasi bila ada frame belum valid |
    | [5] stylize | `--from K --limit N` (+ `--style`). Frame valid dilewati. Basi → `strokes/` DIHAPUS seluruhnya, hanya jendela dihitung, satu peringatan (MP4 utama / out/svg tetap versi lama sampai `run` penuh) |
    | [6] export | `--from K --limit N` → `out/<nama>.preview_K-K+N-1.mp4`; tanpa manifest, tanpa salinan SVG, tanpa pengaman milik-sumber-lain, tanpa audio; tidak menyentuh MP4 utama / out/svg. Export penuh pada `strokes/` separuh → gagal keras (exit 1, perintah `stylize`) |

  - **Poin 6–10 (lengkap):** (6) Export preview: nama dari `export.filename` + `.preview_K-<K+N-1>`; verifikasi ffprobe (jumlah frame = N, ukuran
    output, fps, codec, pix_fmt, tag warna jalur strokes); tanpa audio. (7) Pre-flight preview: video ada, bentrok folder kerja (klip lain →
    exit 1), style valid, ffmpeg/ffprobe, K+N ≤ `frame_count`, seg/depth (poin 4) — semuanya sebelum stage mana pun; pre-flight (c) target
    export utama dan (e) SVG dilewati seperti `--limit`. (8) Urutan: ingest → cek seg/depth → stabilize → vectorize → stylize → export(window);
    cek seg/depth dijalankan sebelum ingest bila `meta.json` ada dan diulang sesudahnya; proses induk tidak meng-import torch. (9) Exit code
    0 / 1 / 2 / 130 (tanpa 3). `--preview` bersama `--limit` / `--restart-from` / `--adopt` / `--qc-only`, `--from` tanpa `--preview`,
    K+N > `frame_count` → exit 1; salah tipe argumen → exit 2. (10) Pesan akhir: jendela, jalur MP4, waktu per stage.
  - **Catatan T-302 (untuk Tahap 4):** preview mempercepat iterasi hilir (vectorize / stylize / export), BUKAN iterasi stabilize; dengan
    temporal dua arah (T-302 / T-303) stabilize penuh jauh lebih lama, jadi evaluasi stabilize memakai subperintah `stabilize` dan
    metriknya sendiri. Backlog (bukan T-204): pengecekan resume segment / depth tanpa torch untuk `run` biasa.

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
  - [2026-10-05] Dari T-204: `run --preview` mempercepat iterasi hilir (vectorize / stylize / export), BUKAN iterasi stabilize itu
    sendiri — preview menjalankan `stabilize` penuh karena vectorize memvalidasi `stable/` semua frame. Dengan temporal dua arah
    (T-302 / T-303) stabilize penuh jauh lebih lama (kini ±110 ms/frame, ±31 s untuk test) dan preview butuh `stable/` penuh, jadi
    evaluasi stabilize memakai subperintah `stabilize` dan metriknya sendiri (`iou_prev`, `label_agreement_prev`), bukan preview.
  - [2026-10-03] Dari T-201b: `normalize: log_median_iqr` melemahkan tepi kaki menyilang di frame 73–83 (pembagi IQR per frame
    `log_iqr` ±0,33 vs median klip 0,19; gradien ±0,5–0,6 × frame lain terhadap ambang per klip). Pipeline oklusi pada depth mentah
    linear memberi Done-when 5/5 dan frame tanpa oklusi 144 (vs 160) di `test`; efek log dan efek IQR belum dipisah. Uji metode
    normalisasi (mis. IQR tetap per klip, domain linear) dengan metrik Done-when T-201b (`docs/04` D-010 "Hasil T-201b")

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
  - [2026-10-03] Catatan dari T-201b: (1) `left_leg` / `right_leg` berisi **0 piksel** di kedua klip (Lower_Clothing ada di
    grup `torso`, D-009): kaki menyilang hanya muncul sebagai garis oklusi di dalam `torso`; kriteria uji = ≥ 80% titik strok di
    Lower_Clothing pada frame 73 / 78 / 82 / 87 / 92 (default 4/5); (2) kalibrasi L atas thinning **Guo-Hall** (L = 30 ≈ 38 piksel
    Zhang-Suen); (3) p80 / p70 memberi 5/5 tetapi tambahannya sebagian besar lipatan celana, jadi menurunkan persentil saja tidak cukup —
    normalisasi di T-302 dulu; (4) bahan: `work/t201b/sensitivity_*.json`, `p80p70_extra.json`, `diag_summary.json`; (5) syarat D menganggap tepi
    frame batas (garis oklusi tidak pernah dalam D px dari tepi)

---

# PHASE 4 — Style lengkap

### T-401 · Width modulation + taper · `TODO`
- **Kerjakan:** variasi tebal sepanjang path (`width_variation`, `width_noise_scale`),
  ujung menipis
- **Update log:**
  - [2026-10-03] T-203a merender garis polos TANPA resample. T-401 butuh **resample arc-length dari `points[0]`** (titik rapat, N =
    max(`shape.resample_points`, ceil(panjang / jarak maks))) untuk tebal per titik; loop dikenali dari titik akhir = titik awal
    (taper tidak menipiskan sambungan); satuan = px ref × `unit` (docs/02)

### T-402 · Jitter deterministic · `TODO`
- **Kerjakan:** Perlin noise per titik, seed = `hash(frame_index, param_seed, track_id)`,
  `temporal_drift` untuk perubahan antar frame
- **⚠️ Pitfall P-007:** random murni = tidak reproducible, tidak bisa di-debug
- **Update log:**
  - [2026-10-03] Jitter terkunci posisi ditunda ke sini (keputusan Rio, T-203a): `points[0]` garis terbuka melompat 60–127 px,
    jadi noise 1D arc-length dari `points[0]` akan "pop". Butuh **resample arc-length dari `points[0]`** (T-203a tidak me-resample)
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
  vs Tasks API `PoseLandmarker`) — **terjawab (T-107, 2026-10-02):** `mediapipe.solutions` tidak ada di
  mediapipe 1.0.1; pakai Tasks API
- **Dibuka lagi kalau:** klip nyata gagal QC dan temporal berbobot QC (T-302) tidak cukup
- **Update log:**
  - [2026-09-29] BLOCKED — foreground Sapiens2 0 frame gagal QC di klip uji; frame gagal QC diisi dari
    frame tetangga lewat temporal berbobot di [3] (Q3-B) — D-010
  - [2026-10-02] Temuan di T-107: mediapipe 1.0.1: `mediapipe.solutions` tidak ada; `vision.PoseLandmarker`
    ada (bisa diimpor) tetapi butuh file model `.task` yang belum ada di `models/`; belum diuji karena T-501
    BLOCKED (keputusan Rio: model `.task` tidak diunduh sekarang)

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
| 1 Skeleton | T-101 … T-108 (T-102 → a/b/c, T-104 → a/b) — ✅ **Phase 1 selesai** (🎯 milestone T-104b, T-107 DONE) | 11/11 |
| 2 Vectorize | T-201 … T-204 (T-201 → a/b, T-203 → a/b) — T-201a ✅, T-201b ✅ DONE (dengan batas kaki), T-202 ✅ DONE, T-203a ✅ DONE, T-203b ✅ DONE, T-204 ✅ DONE — 🎯 **Milestone Phase 2 tercapai** | 6/6 |
| 3 Stabilize | T-301 … T-305 (T-301 SKIP) | 0/5 |
| 4 Style | T-401 … T-406 | 0/6 |
| 5 Fallback | T-501 … T-502 (BLOCKED) | 0/2 |
| 6 Opsional | T-601 … T-603 | 0/3 |

**Total: 38 task** (5 + 11 + 6 + 5 + 6 + 2 + 3) · Selesai: 22/38 (5 + 11 + 6; SKIP — T-301, T-601 — tidak dihitung selesai).
Rekonsiliasi 2026-10-05: T-204 DONE → Phase 2 = 6/6, total 22/38.
Rekonsiliasi 2026-10-05: T-203b DONE → Phase 2 = 5/6, total 21/38.
Rekonsiliasi 2026-10-03: T-203 dipecah jadi T-203a + T-203b (Phase 2 = 6 task); jumlah per Phase = total.
Rekonsiliasi 2026-10-01: sebelum T-104b selesai papan menulis 9/11 + 13/37, padahal 5 + 9 = 14 — total salah hitung 1.
