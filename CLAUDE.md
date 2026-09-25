# CLAUDE.md — Rotoscope Animation Builder

## Apa project ini
Tool Python CLI yang mengubah video meme low-quality (5–15 detik, satu subjek dominan)
menjadi animasi rotoscope bergaya sketsa outline hand-drawn. Output: MP4 + SVG per frame.
Hasil akan dimonetisasi (TikTok/Reels/YouTube) → lisensi semua dependency wajib aman komersial.

## Style target
- Hanya siluet luar. Tanpa detail interior (wajah, lipatan baju, otot)
- Garis tunggal kontinu, tebal-tipis bervariasi, sedikit bergetar
- Background kertas polos, monokrom
- Kasar seperti gesture drawing. BUKAN hasil edge detection, BUKAN vector yang terlalu rapi

## Arsitektur (TERKUNCI — lihat docs/04-DECISION-LOG.md)
Segmentation → Contour → Stylized stroke. MediaPipe Pose hanya fallback untuk frame gagal QC.

| # | Modul | Input → Output |
|---|---|---|
| 1 | ingest.py | video → work/frames/frame_%05d.png + meta.json |
| 2 | segment.py | frames → work/masks/*.png + work/qc_report.json |
| 2b | fallback_pose.py | frame gagal QC → mask sintetik (di-blend, bukan mengganti) |
| 3 | stabilize.py | masks → work/masks_smooth/*.png (temporal EMA + optical flow) |
| 4 | vectorize.py | masks_smooth → work/contours/*.json |
| 5 | stylize.py | contours + style YAML → work/strokes/*.svg + *.png |
| 6 | export.py | strokes → out/animation.mp4 + out/svg/*.svg |

Kontrak lengkap per modul: docs/01-PIPELINE-SPEC.md. Parameter style: docs/02-STYLE-PARAMS.md.

## Aturan keras (jangan dilanggar tanpa izin eksplisit)
1. rembg SELALU pakai model eksplisit `u2net_human_seg`. JANGAN pakai model default
   (bria-rmbg: lisensi komersial berbayar) — D-003
   ⚠️ Lisensi weights U-2-Net BELUM diverifikasi (action item D-003). Jangan klaim aman komersial sebelum dicek.
2. Tolak library/model berlisensi AGPL/copyleft atau lisensi komersial berbayar
   (contoh: YOLOv8 — D-004). Cek lisensi sebelum menambah dependency apa pun
3. Semua parameter style dan threshold dibaca dari YAML (configs/). Tidak ada magic number.
   Setiap parameter punya default yang masuk akal (pipeline jalan tanpa YAML)
4. Randomness wajib deterministic: seed = hash(frame_index, param_seed) — P-007
5. Setiap stage baca dari disk dan tulis ke disk → pipeline resumable, tiap stage bisa
   dijalankan ulang sendiri
6. onnxruntime versi CPU dulu. Jangan setup GPU/CUDA kecuali mengerjakan T-601
7. Jangan tambah dependency baru tanpa bertanya dulu dan membandingkannya dengan stack
   yang sudah disetujui

## Environment
- Windows, PowerShell (Git Bash juga tersedia)
- Python 3.11.9, venv di ./venv → aktifkan: .\venv\Scripts\Activate.ps1
- GPU: NVIDIA GTX 1650 Ti, 4 GB VRAM. Jangan sarankan model/teknik yang butuh >3 GB VRAM
- ffmpeg: binary eksternal
- Stack disetujui: opencv-python, rembg, mediapipe, numpy, scipy, svgwrite, Pillow, pyyaml

## Cara kerja
- Satu task/modul per sesi, sesuai ID task di docs/05-TASK-BOARD.md. Jangan sentuh modul lain
- Ikuti kontrak modul di docs/01. Kalau kontrak perlu berubah: jelaskan alasannya, tanya dulu
- Setiap modul punya test di tests/test_<modul>.py
- Developer level beginner-intermediate: jelaskan singkat istilah CV/ML saat pertama muncul
- JANGAN membaca isi work/, out/, samples/, atau file *.png / *.mp4 (ribuan frame, boros token)

## Struktur repo
- docs/            spesifikasi 00–06 (dipindahkan ke sini di T-005; kalau belum ada, minta ke saya)
- src/rotoscope/   modul pipeline + config.py + cli.py
- configs/         default.yaml + styles/*.yaml
- assets/          brushes/ dan paper/ (tekstur)
- samples/         video test (tidak di-commit)
- scripts/         utilitas, mis. smoke_test.py
- tests/           unit test
- work/, out/      intermediate & hasil (gitignored)
