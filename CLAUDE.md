# CLAUDE.md — Rotoscope Animation Builder

## Apa project ini
Tool Python CLI yang mengubah video meme low-quality (5–15 detik, satu subjek dominan)
menjadi animasi rotoscope bergaya sketsa outline hand-drawn. Output: MP4 + SVG per frame.
Hasil akan dimonetisasi (TikTok/Reels/YouTube) → lisensi semua dependency wajib aman komersial.

## Style target
- Siluet luar + garis oklusi: tepi anggota tubuh yang berada di depan bagian tubuh lain (mis. lengan di depan dada, kaki menyilang) — D-009. Tanpa garis batas baju–kulit, tanpa detail wajah, tanpa lipatan baju/otot
- Garis tunggal kontinu, tebal-tipis bervariasi, sedikit bergetar
- Background kertas polos, monokrom
- Kasar seperti gesture drawing. BUKAN hasil edge detection, BUKAN vector yang terlalu rapi

## Arsitektur (TERKUNCI — lihat docs/04-DECISION-LOG.md)
Segmentation → Contour → Stylized stroke. MediaPipe Pose hanya fallback untuk frame gagal QC.

| # | Modul | Input → Output |
|---|---|---|
| 1 | ingest.py | video → work/frames/frame_%05d.png + meta.json |
| 2 | segment.py | frames → work/seg/classmap/*.png + work/seg/probs/*.npz (Sapiens2-seg, probabilitas mentah 29 kelas) + work/seg/manifest.json + work/qc_report.json |
| 2b | fallback_pose.py | DITUNDA (BLOCKED, D-010) — frame gagal QC diberi bobot temporal kecil di stage 3 |
| 2c | depth.py | frames → work/depth/*.npy (disparity mentah DA-V2 Small) + work/depth/manifest.json |
| 3 | stabilize.py | seg/probs + depth + qc_report → work/stable/groups/*.png (peta grup) + work/stable/depth_smooth/*.npy (temporal EMA + optical flow, filter pulau + mode filter) |
| 4 | vectorize.py | stable/ → work/contours/*.json (polyline bertipe: silhouette, silhouette_hole, group_boundary, occlusion) + work/contours/clip_stats.json |
| 5 | stylize.py | contours + style YAML → work/strokes/*.svg + *.png |
| 6 | export.py | strokes → out/animation.mp4 + out/svg/*.svg |

Path `work/…` di tabel = relatif terhadap folder kerja klip `work/clips/<nama video>/` (T-104b).
Cara pakai: `python -m rotoscope run <video>`; per stage: `python -m rotoscope <stage> <video>`; unduh model:
`python -m rotoscope download` (bagian "CLI" di docs/01). Hasil GPU hanya dihapus dengan `--yes`.
Kontrak lengkap per modul: docs/01-PIPELINE-SPEC.md. Parameter style: docs/02-STYLE-PARAMS.md.

## Aturan keras (jangan dilanggar tanpa izin eksplisit)
1. Segmentasi (stage 2) = Sapiens2-seg 0.8B fp16 GPU; fallback Sapiens2-seg 0.4B fp16 untuk SELURUH
   klip (tidak pernah dicampur dalam satu klip). Kedalaman (garis oklusi) = Depth Anything V2 Small
   (Apache-2.0; Base/Large CC-BY-NC DILARANG). Sapiens2-pointmap dan seg 1B DIBUANG — D-009.
   rembg/u2net_human_seg DITOLAK (D-008): jangan dipakai lagi
2. Tolak library/model berlisensi AGPL/copyleft atau lisensi komersial berbayar
   (contoh: YOLOv8 — D-004). Cek lisensi sebelum menambah dependency apa pun
3. Semua parameter style dan threshold dibaca dari YAML (configs/). Tidak ada magic number.
   Setiap parameter punya default yang masuk akal (pipeline jalan tanpa YAML)
4. Randomness wajib deterministic: seed = hash(frame_index, param_seed, track_id) — P-007, D-010
5. Setiap stage baca dari disk dan tulis ke disk → pipeline resumable, tiap stage bisa
   dijalankan ulang sendiri
6. GPU/CUDA hanya lewat PyTorch (wheel cu118) untuk Sapiens2-seg dan Depth Anything V2 Small, satu model
   di GPU pada satu waktu. Tanpa bf16 (Turing, P-005): fp16/fp32 saja (D-009). onnxruntime-gpu tidak dipakai (T-601 SKIP).
7. Jangan tambah dependency baru tanpa bertanya dulu dan membandingkannya dengan stack
   yang sudah disetujui

## Environment
- Windows, PowerShell (Git Bash juga tersedia)
- Python 3.11.9, venv di ./venv → aktifkan: .\venv\Scripts\Activate.ps1
- GPU: NVIDIA GTX 1650 Ti, 4 GB VRAM, driver 517.00 (CUDA maks 11.7). Jangan sarankan model/teknik yang
  butuh >3 GB VRAM. PENGECUALIAN (disetujui Rio, D-005/D-009): Sapiens2-seg 0.8B fp16 (3276 MiB reserved),
  dengan syarat cek VRAM bebas sebelum stage [2] + berhenti dengan pesan jelas, resume per frame, dan
  fallback 0.4B fp16 (2258 MiB) untuk seluruh klip
- ffmpeg: binary eksternal
- Stack disetujui: opencv-contrib-python (via mediapipe; JANGAN install opencv-python juga), mediapipe, numpy, scipy, svgwrite, Pillow, pyyaml, torch==2.7.1+cu118, torchvision==0.22.1+cu118, transformers==5.17.0 (index PyTorch cu118 di requirements.txt). rembg DITOLAK (D-008) — masih terinstall sampai follow-up pembersihan, jangan dipakai
- Dev-only: pytest (requirements-dev.txt). Package di-install editable: pip install -e .
- Versi mayor baru: OpenCV 5.x dan mediapipe 1.x. Jangan asumsikan API versi lama (OpenCV 4.x / mediapipe 0.10.x); cek dokumentasi versi terinstall dulu.

## Cara kerja
- Satu task/modul per sesi, sesuai ID task di docs/05-TASK-BOARD.md. Jangan sentuh modul lain
- Ikuti kontrak modul di docs/01. Kalau kontrak perlu berubah: jelaskan alasannya, tanya dulu
- Edit file teks di repo pakai tool Edit/Write, BUKAN script Python/sed/heredoc di shell — supaya setiap perubahan tampil sebagai diff yang bisa saya review
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
