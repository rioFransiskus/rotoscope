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
| 3 | stabilize.py | seg/probs + depth + qc_report (+ frames/ untuk cut) → work/stable/groups/*.png (peta grup) + work/stable/depth_smooth/*.npy (+ manifest.json, frames.jsonl). T-302 DONE: temporal DEFAULT AKTIF = kernel eksponensial simetris terpotong ρ^\|k\|·q (α 0,7 → R 2; `TAIL_MASS` 0,01, `R_MAX` 8; q = `qc_fail_weight` 0,1 untuk frame gagal QC; potong di tepi klip dan cut `cut_diff` 0,08; `boil_preserve` 0,3 = campur argmax mentah), TANPA optical flow (DITOLAK di T-303 berdasarkan data, docs/04; parameter `optical_flow_blend` dihapus), lalu filter pulau + mode filter; kedalaman `log_median` (log − median fg; `log_median_iqr` = enum kompatibilitas), EMA kedalaman `depth.temporal` MATI (id oklusi ×3 tanpa flow). `--limit N` hanya menulis frame berjendela LENGKAP (sisa ditunda + pesan). `temporal.enabled: false` = jalur spasial T-106, byte-identik. Metrik permanen: `tests/temporal_metrics.py` |
| 4 | vectorize.py | stable/ → work/contours/*.json (polyline bertipe: silhouette, silhouette_hole, group_boundary, occlusion) + work/contours/manifest.json (+ clip_stats.json, T-201b). T-201a DONE: siluet + lubang + batas grup; T-201b DONE: garis oklusi dari depth_smooth + clip_stats.json (ambang persentil per klip, dihitung dari seluruh klip; batas kaki, kalibrasi di T-302/T-305); T-202 DONE: orientasi (silhouette searah jarum jam, lubang berlawanan), anchor = `points[0]` (diputar; key `anchor` konstan 0), arah garis terbuka berkesinambungan, `track_id` (mulai 1; `src/rotoscope/track.py`, Chamfer simetris + penugasan optimal, `max_match_dist_px` 16, silhouette terbesar mewarisi id) + key level-frame `prev_sha256` (rantai kesinambungan: stage [4] diproses BERURUTAN, resume memeriksa rantai). Subperintah `vectorize`; bagian urutan `run` sejak T-203b |
| 5 | stylize.py | contours + style YAML → work/strokes/*.svg + *.png + manifest.json (+ frames.jsonl). T-203a DONE: garis polos (penghalusan Gaussian `shape.smooth_px` → approxPolyDP → Catmull-Rom; mode tepi `hide`/`draw`; SVG string manual + PNG supersampling; satuan px ref 1080 × `render.output_width`/1080; hash style = parameter aktif). T-401 DONE (contract "T-401"): resample arc-length (N = max(`shape.resample_points` 4, ceil(L / 2 px ref))), tebal per titik = lantai 1 px × `width_base` × (1 + `width_variation` 0,5 × noise 2D terkunci posisi, `noise.py`, seed = hash(`jitter.param_seed`), statis terhadap waktu) × taper smoothstep (`taper_px` 70, `taper_min` 0,5) hanya di ujung BEBAS (bukan di tepi frame / ujung bertemu strok lain ≤ 8 px ref / loop / tertutup; override `stroke.by_type.<tipe>.taper_ends`); raster trapesium + cakram; SVG = satu poligon kontur terisi per jalur (`nonzero`, tebal tidak bisa diubah di editor). T-402 DONE (contract "T-402"): jitter = medan perpindahan KOHEREN vektor 2D (noise 3D posisi + waktu, `jitter.hold_frames` 2, `temporal_drift` 0,35, `stroke_independence` 0, pelunakan tepi, penjaga lipatan r > 0,19) — **DEFAULT MATI** (`jitter.amplitude` 0 = byte-identik T-401; keputusan Rio: terasa acak-acakan; aktif lewat style). Multipass, tekstur, opasitas = Phase 4 berikutnya (T-403…); jitter ditinjau ulang setelah itu. Subperintah `stylize` (`--style --restart --limit`); bagian urutan `run` sejak T-203b |
| 6 | export.py | strokes → out/<nama>.mp4 + out/svg/<nama>/*.svg. T-203b DONE (`SUPPORTED_STROKES_CONTRACTS` = {"T-402"} sejak T-402; strokes lama basi, salinan SVG lama disalin ulang sebagai basi): default `export.source: strokes` (MP4 dari strokes/*.png ukuran output, tag warna bt709 lengkap, crf 18; `silhouette` = Phase 1 tetap ada); SVG disalin dengan penanda `.rotoscope-clip.json` + sha256 per berkas — SVG yang disunting pengguna / penanda rusak / folder milik video lain tidak ditimpa tanpa `--restart` (milik video lain tidak pernah); versi ffmpeg di manifest (tidak ikut hash) |

Path `work/…` di tabel = relatif terhadap folder kerja klip `work/clips/<nama video>/` (T-104b).
Preview cepat (T-204, tanpa GPU): `python -m rotoscope run <video> --preview N [--from K]` → `out/<nama>.preview_K-<K+N-1>.mp4`
(jendela frame K..K+N-1; seg/ dan depth/ harus sudah valid untuk jendela, kalau tidak exit 1 + perintah; tidak bersama `--limit` /
`--restart-from`; MP4 utama dan out/svg tidak disentuh; hangat ±4–6 s, rantai vectorize dingin ±30 s, `stable/` basi ±50 s).
Cara pakai: `python -m rotoscope run <video> [--config P] [--style P] [--limit N] [--restart-from STAGE [--yes]]` (urutan: ingest →
segment → depth → stabilize → vectorize → stylize → export); per stage: `python -m rotoscope <stage> <video>` (stage = ingest, segment,
depth, stabilize, vectorize, stylize, export); unduh model: `python -m rotoscope download` (bagian "CLI" di docs/01). Hasil GPU
(`--restart-from ingest|segment|depth`) hanya dihapus dengan `--yes` — itu MENJALANKAN GPU; vectorize / stylize / export ber-restart
tanpa `--yes`. Skenario uji destruktif / pre-flight hanya lewat stage palsu atau `--restart-from vectorize`.
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
4. Randomness wajib deterministic — P-007, D-010. Jitter T-402: noise 3D dari `jitter.param_seed` (salt terpisah dari tebal), indeks gambar = floor(frame_index / hold_frames), seed per `track_id` hanya untuk s > 0 (rencana lama hash per strok DIGANTI, docs/04 "Keputusan T-402")
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
- Stack disetujui: opencv-contrib-python (via mediapipe; JANGAN install opencv-python juga), mediapipe, numpy, scipy, svgwrite, Pillow, pyyaml, torch==2.7.1+cu118, torchvision==0.22.1+cu118, transformers==5.17.0 (index PyTorch cu118 di requirements.txt). rembg + onnxruntime DITOLAK (D-008) dan sudah dicopot dari venv/requirements (T-107): jangan diinstal lagi. scikit-image juga tidak ada; thinning (Phase 2) = `cv2.ximgproc.thinning` dari opencv-contrib-python (jangan ganti dengan opencv-python / headless)
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
