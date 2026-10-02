# 04 — DECISION LOG & PITFALL

## Keputusan

### D-001 — Arsitektur: Segmentation → Contour (bukan edge detection)
**Tanggal:** 2026-09-15
**Alasan:** Hanya pendekatan ini yang memungkinkan kontrol penuh atas tekstur dan style
garis, karena garis digambar ulang dari vector path — bukan hasil filter pixel.
Edge detection (Canny/XDoG) ikut menangkap detail interior dan noise background,
tidak sesuai style target.
**Alternatif ditolak:** Edge detection (B), Pose-only (C).

### D-002 — MediaPipe Pose sebagai fallback, bukan primary
**Alasan:** Pose skeleton sangat stabil tapi menghasilkan bentuk tubuh generik —
karakter subjek hilang. Dipakai hanya untuk frame yang gagal QC segmentasi.

### D-003 — ⚠️ JANGAN pakai model default rembg
**Alasan:** Default rembg sekarang `bria-rmbg` (BRIA AI, ~1,02 GB, 1024×1024).
RMBG-2.0 dirilis dengan lisensi BRIA yang **mensyaratkan perjanjian berbayar untuk
penggunaan komersial**. Lisensi bobot model berdiri sendiri, terpisah dari lisensi MIT rembg.
**Keputusan:** selalu pass flag eksplisit `-m u2net_human_seg` — model pre-trained khusus
segmentasi manusia, lebih kecil, lebih cepat, tepat sasaran.
**Action item:** verifikasi lisensi weights U-2-Net sebelum monetisasi.
**Status action item (2026-09-25): DIVERIFIKASI — 🟡 abu-abu.**
- Kode U-2-Net: Apache-2.0 ✅. rembg: MIT ✅ (README rembg sendiri menegaskan lisensi weights terpisah)
- Weights `u2net_human_seg`: tidak ada lisensi terpisah; dilatih pada Supervisely Person Dataset
- Supervisely Person Dataset: **non-komersial saja** (riset, pengajaran, publikasi, eksperimen pribadi)
- Lisensi dataset tidak menyebut model hasil training → apakah batasannya menurun ke weights
  belum ada kepastian hukum. Mirror Roboflow berlabel CC BY 4.0 = re-upload pihak ketiga, bukan acuan
- Tindak lanjut → **D-008**. Larangan model default (`bria-rmbg`) tetap berlaku

### D-004 — Hindari YOLOv8-seg
**Alasan:** Lisensi AGPL-3.0 (copyleft). Berisiko untuk penggunaan komersial.

### D-005 — SAM 2 diturunkan ke eksperimen opsional (Phase 6)
**Alasan:** Checkpoint terkecil `sam2.1_hiera_tiny` (38M params) butuh ~4 GB VRAM.
GTX 1650 Ti punya tepat 4 GB, dikurangi alokasi display Windows. Memory bank SAM 2 juga
tumbuh seiring panjang video → risiko OOM tinggi.
**Catatan:** CC 7.5 memenuhi syarat SAM 2 (butuh 7.0+), jadi secara hardware kompatibel —
hanya VRAM yang jadi kendala.
**Anggaran VRAM (T-102c, 2026-09-29, D-009)** — peak PyTorch reserved; GTX 1650 Ti 4096 MiB, di luar
proses ±260–690 MiB (display Windows):

| model | precision | reserved MiB | allocated MiB | peran |
|---|---|---|---|---|
| Sapiens2-seg 0.8B | fp16 | 3276 | 3014 | stage [2] utama — **pengecualian aturan ≤ 3 GB (disetujui Rio)** |
| Sapiens2-seg 0.4B | fp16 | 2258 | 2185 | fallback stage [2] untuk seluruh klip |
| Depth Anything V2 Small | fp32 | 424 | 291 | kedalaman (garis oklusi) |

Model dijalankan **bergantian** (satu model di GPU pada satu waktu), tidak pernah bersamaan. Syarat
pengecualian: cek VRAM bebas sebelum stage [2] + berhenti dengan pesan jelas, resume per frame,
fallback 0.4B fp16 untuk seluruh klip (tidak dicampur) — diimplementasikan di T-102b.

### D-006 — Output: MP4 + SVG per frame
**Alasan:** SVG memungkinkan koreksi manual di Krita/Illustrator untuk frame yang
hasilnya jelek, tanpa render ulang seluruh pipeline.

### D-007 — Pipeline harus resumable, tiap stage tulis ke disk
**Alasan:** Iterasi style (stage 5) akan sangat sering. Tidak masuk akal mengulang
segmentasi tiap kali ganti tebal garis.

### D-008 — Backend segmentasi (stage 2) diputuskan lewat A/B test di T-102a
**Tanggal:** 2026-09-25
**Konteks:** Hasil verifikasi action item D-003: lisensi `u2net_human_seg` abu-abu untuk
monetisasi. Alternatif dengan lisensi jelas: MediaPipe SelfieMulticlass 256×256 — model card
resmi Google mencantumkan Apache License 2.0, mendukung satu/banyak orang, selfie dan full body,
intended use termasuk entertainment.
**Keputusan:** Uji dua kandidat pada video test yang sama di T-102a:
- MediaPipe SelfieMulticlass 256×256 — **default kalau kualitas setara**
- rembg `u2net_human_seg` — **hanya untuk eksperimen/development** sampai keputusan final;
  tidak dipakai untuk konten yang dimonetisasi kecuali keputusan ini direvisi
**Alasan:** Resolusi input mirip (256 vs 320), style target kasar (jitter ±1,8 px) → ada
peluang MediaPipe cukup, tapi harus dibuktikan dengan data. MediaPipe sudah ada di stack (D-002),
model ±15,6 MB vs U2-Net penuh ±170 MB. Menguji u2net saat development masuk kategori
"eksperimen pribadi" yang diizinkan lisensi dataset.
**Kriteria A/B:** perbandingan visual side-by-side (penilaian Rio), waktu inferensi CPU per frame,
stabilitas antar-frame (IoU rata-rata frame berurutan), jumlah frame gagal `area_ratio`.
**Setelah T-102a:** update `01-PIPELINE-SPEC.md` stage [2], aturan #1 `CLAUDE.md`, dan status
D-008. Kalau MediaPipe dipilih: rembg + onnxruntime keluar dari stack, T-601 → `SKIP`.
**Alternatif ditolak:** Tetap u2net tanpa uji (risiko terbawa ke monetisasi); ganti ke MediaPipe
tanpa data (kalau kualitas jelek harus mundur).
**Status (2026-09-28): SELESAI.** Hasil T-102a (`scripts/ab_segment.py`, `samples/test.mp4`,
283 frame 480×854, CPU 12 core, threshold 0.5, tanpa morfologi, ms tanpa 3 frame warm-up):

| backend | ms mean | ms med | ms p95 | iou_prev mean | med | min | iou<0.55 | area<3% | area>70% | >1 blob |
|---|---|---|---|---|---|---|---|---|---|---|
| mediapipe | 105.3 | 104.8 | 109.1 | 0.856 | 0.870 | 0.581 | 0 | 0 | 0 | 0 |
| mediapipe_pad | 106.8 | 106.6 | 109.3 | 0.840 | 0.857 | 0.489 | 3 | 0 | 0 | 1 |
| u2net | 417.1 | 416.2 | 427.3 | 0.864 | 0.897 | 0.424 | 7 | 0 | 0 | 0 |

cross_iou (mean / median / min): mediapipe vs u2net 0.805 / 0.832 / 0.451; mediapipe_pad vs u2net
0.750 / 0.783 / 0.432; mediapipe vs mediapipe_pad 0.825 / 0.868 / 0.496.
- **u2net_human_seg DITOLAK:** lisensi abu-abu (D-003), kaki hilang di frame 236–241, flicker
  (7 frame iou_prev < 0.55 di frame 156–173 dan 236–241)
- **mediapipe_pad DITOLAK:** stabilitas lebih buruk (min iou_prev 0.489, 3 frame < 0.55, 1 multi-blob)
- **MediaPipe unggul atas u2net:** kualitas setara, zona gagal berbeda. MediaPipe kehilangan kaki
  di frame 1–7 — salah tapi stabil, tidak tertangkap QC. **NAMUN tidak dipakai sebagai backend
  final → D-009**
- **Batasan:** hanya 1 klip dengan gerakan sedang

### D-009 — Style target garis oklusi + backend Sapiens2
**Tanggal:** 2026-09-28
**Status (2026-09-29): LOLOS** — uji kelayakan **T-102c** selesai; keputusan final di bagian
**Hasil T-102c** di bawah (seg 0.8B fp16 + Depth Anything V2 Small; pointmap dibuang)
**Konteks:** Siluet saja kehilangan keterbacaan pose saat lengan/kaki menempel ke badan (review
visual Rio, T-102a). Mask MediaPipe tidak menyimpan batas antar anggota tubuh.
**Keputusan (keputusan awal 2026-09-28 — digantikan Keputusan final di bawah):**
- **Sapiens2-seg 0.4B** (29 kelas bagian tubuh) untuk batas anggota tubuh + mask foreground.
  Lolos uji demo visual Rio (2026-09-28)
- **Sapiens2-pointmap 0.4B** untuk batas oklusi di area berpakaian — **BELUM diuji**. Dievaluasi
  di T-102c, dibuang kalau seg saja sudah cukup
**Lisensi:** Sapiens2 License — bebas royalti, TIDAK ada klausul non-komersial.
- Larangan penggunaan: surveillance, biometric processing, deepfake/konten menyesatkan,
  pelanggaran hak pihak ketiga
- Meta dapat mengubah lisensi sepihak. Bukan OSI open source
- **Risiko tercatat:** video input milik pihak ketiga (klausul hak pihak ketiga)
**Syarat teknis:**
- Python ≥3.12 + PyTorch ≥2.7 (project sekarang 3.11.9) → strategi environment diputuskan di T-102c.
  **Koreksi (T-102c):** syarat Python ≥3.12 hanya berlaku untuk repo resmi `facebookresearch/sapiens2`;
  lewat Hugging Face Transformers (≥5.10.1, Python ≥3.10) Sapiens2 jalan di Python 3.11.9
- GPU Turing: tanpa bf16 (P-005) → fp16/fp32
- Pin versi PyTorch. **Koreksi (T-102c):** driver 517.00 hanya sampai CUDA 11.7 → cu126/cu128 tidak
  jalan tanpa update driver; dipakai wheel **cu118** (memuat sm_75)
**Alternatif ditolak:**
- Batas kelas MediaPipe multiclass — hanya kulit vs baju
- Pose landmark — bentuk generik (D-002)
- Edge detection di dalam mask (D-001)
- DensePose dan Sapiens v1 — lisensi non-komersial
- SAM 3D Body — kemungkinan butuh > 4 GB VRAM
- **Cadangan kalau T-102c gagal:** Depth Anything V2 Small (Apache 2.0)
**Ditunda:** benda yang dipegang (gelas dll.) — tidak ada kelasnya di Sapiens2.
**Dampak (dikerjakan setelah T-102c lolos):** kontrak stage [2]/[4]/[5] di `01-PIPELINE-SPEC.md`,
stack + aturan #6 `CLAUDE.md`, D-005 (VRAM).

**Hasil T-102c (2026-09-29)** — `scripts/sapiens2_probe.py`, `scripts/sapiens2_exp.py`,
`scripts/look_test.py`; `samples/test.mp4`, 283 frame 480×854; detail di update log T-102c `docs/05`.

*Environment — Opsi 1a (dipilih Rio):* venv utama Python 3.11.9 + `torch==2.7.1+cu118`,
`torchvision==0.22.1+cu118`, `transformers==5.17.0`; GTX 1650 Ti (sm_75), driver 517.00, attention
`sdpa`. 0 paket existing berubah versi, `pip check` bersih. Checkpoint di cache Hugging Face (di luar repo).

*Benchmark* (s/frame = forward + post-process, tanpa 3 frame warm-up; VRAM = peak PyTorch):

| model | precision / device | status | s/frame | VRAM alloc / reserved MiB | catatan |
|---|---|---|---|---|---|
| seg 0.4B | fp32 GPU | ok | 2.54 | 2972 / 3188 | full run 283 frame, 0 NaN |
| seg 0.4B | fp16 GPU | ok | 7.99 | 2185 / 2258 | kelas sama vs fp32 99.998% |
| **seg 0.8B** | **fp16 GPU** | **ok** | **15.85** | **3014 / 3276** | **full run 283 frame, 0 NaN** |
| seg 1B | – | tidak dijalankan | – | – | tidak layak di hardware ini (VRAM tidak muat, RAM CPU tidak memenuhi syarat) |
| pointmap 0.4B | fp16 GPU | NaN/inf 20/20 | 14.69 | 2420 / 3084 | |
| pointmap 0.4B | fp32 GPU | OOM | – | 3437 saat gagal | |
| pointmap 0.4B | fp32 + autocast fp16 GPU | OOM (2×) | – | 3469 / 3546 | percobaan perbaikan `max_split_size_mb:128` juga OOM |
| pointmap 0.4B | fp32 CPU | ok | 30.66 | RAM 5142 MiB | full run terhenti di 95/283 (RAM sistem habis) |
| **Depth Anything V2 Small** | **fp32 GPU** | **ok** | **0.16** | **291 / 424** | **full run 283 frame, 0 NaN, 0.185 s/frame total** |

*Full run seg 0.8B fp16 (283 frame):* VRAM di luar proses sebelum run 258 MiB (nvidia-smi); bebas
3318 MiB sebelum load, 1568 MiB setelah load (`torch.cuda.mem_get_info`). Tanpa OOM, NaN/inf 0.
s/frame infer 15.85 (p95 15.87), total termasuk I/O 15.87. Peak VRAM reserved **3276 MiB di semua
frame**, allocated maks 3014 MiB. Estimasi klip 15 s (360 frame): seg 0.8B ≈ 95 mnt + DA ≈ 1 mnt
→ **±1.6 jam per klip**; waktu bukan batasan (keputusan Rio).

*Metrik foreground full klip* (definisi `sapiens2_probe.py` = `ab_segment.py`):

| backend | iou_prev mean / min | frame gagal QC | label_agreement_prev mean / min | cross_iou vs MediaPipe mean |
|---|---|---|---|---|
| seg 0.4B fp32 | 0.903 / 0.724 | 0 | 0.951 / 0.790 | 0.861 |
| seg 0.8B fp16 | 0.901 / 0.713 | 0 | 0.955 / 0.778 | 0.863 |
| MediaPipe (T-102a) | 0.856 / 0.581 | 0 | – | – |

Kelas sama 0.8B vs 0.4B per frame: mean 99.33%, min 97.50%.

**Keputusan final (Rio, 2026-09-29):**
- **Seg = Sapiens2-seg 0.8B, fp16 GPU**, input default image processor 1024×768 (**stretch**,
  `do_pad=false`). Alasan: penilaian visual Rio — setelah post-processing jauh lebih bersih dari 0.4B
  (komponen garis kecil < 100 px setelah post-processing, total 20 frame: 102 vs 116 di frame 73–92,
  31 vs 50 di frame 183–202; label_agreement_prev 0.976 vs 0.969 di frame 73–92) — dan lolos full run
  283 frame.
- **PENGECUALIAN aturan VRAM ≤ 3 GB (disetujui Rio):** seg 0.8B reserved 3276 MiB. Syarat, diimplementasikan
  di T-102b: (1) cek VRAM bebas sebelum stage [2], berhenti dengan pesan jelas kalau tidak cukup;
  (2) resume per frame; (3) fallback **seg 0.4B fp16** (2258 MiB reserved) untuk **SELURUH klip** —
  tidak pernah dicampur dengan 0.8B dalam satu klip.
- **Post-processing seg** (default kandidat untuk T-102b/T-201, jadi parameter YAML):
  - filter pulau kelas **N = 30 px** — komponen 8-arah sebuah kelas < N diganti kelas mayoritas di
    cincin 1 px sekelilingnya. Pulau penyebab garis glitch di tangan (frame 90) berukuran 1–28 px → N > 28;
    struktur terkecil yang stabil antar frame: Left_Hand 36 px (0.8B frame 196), Left_Shoe 44–48 px
    (frame 187–190) → N ≤ 36. N = 50 menghapus keduanya
  - mode filter peta grup **K = 3** (background ikut dihitung sebagai grup) — K terbesar yang tidak
    membuat komponen grup ≥ N px kehilangan > 50% area (80 peta uji: K=3 → 0, K=5 → 12, K=7 → 43);
    0.099% piksel foreground berubah grup
  - filter komponen garis **M = 5 px** (8-arah) — bintik garis DA di zona tangan 2–4 px; setelah
    pulau + mode filter, 30% komponen garis ≤ 4 px
- **Seg 1B:** tidak layak di hardware ini.
- **Kedalaman = Depth Anything V2 Small** (`depth-anything/Depth-Anything-V2-Small-hf`, model card
  **Apache-2.0**; Base/Large CC-BY-NC — DILARANG), fp32 GPU, 0.16 s/frame, VRAM 424 MiB reserved. Garis
  oklusi dari lompatan kedalaman relatif (threshold p95 per klip). Penilaian Rio: setara pointmap di
  frame 73–92 (kaki menyilang), tanpa garis palsu di tali cargo/saku/lipatan.
- **Sapiens2-pointmap DIBUANG:** GPU OOM (fp32, fp32 + autocast) dan NaN (fp16); CPU 30.7 s/frame
  (±3.1 jam per klip) + risiko RAM sistem habis.
- **Grup garis** (`scripts/sapiens2_groups.json`): hair dan face terpisah; Lower_Clothing masuk torso
  (tanpa garis pinggang kaos–celana). Batas di dalam satu grup tidak digambar.
- **Target konten garis (disetujui Rio):** tingkat detail `work/t102c/exp/filtered_full_0.8b.mp4` —
  siluet + batas grup + garis oklusi kedalaman — TANPA bayangan dan duplikat siluet. Tampilan
  hand-drawn = tugas stage [4]–[5].
- Nama 29 kelas: `scripts/sapiens2_classes.json` (config HF hanya `LABEL_0..28`; sumber
  `facebookresearch/sapiens2` `docs/SEG.md` commit `744905ba`).

**Known issues (dicatat, belum diperbaiki):**
- Garis kedalaman DA mentah menghasilkan "bayangan": duplikat sejajar siluet/batas seg dan bercak
  tebal (`work/t102c/look/diag_sources.png`; frame 120: 4234 px garis DA vs 11808 px garis seg).
  Syarat T-201: tepi tipis (NMS + hysteresis), hanya di dalam grup dengan jarak minimum ke batas seg,
  panjang minimum. Titik awal dari look test (`scripts/look_test.py`): T_high p95 / T_low p90 per klip
  atas |grad log d|, jarak minimum D = 7 px, panjang skeleton minimum L = 30 px
- Getaran garis interior → T-302 harus menstabilkan **peta grup**, bukan hanya mask biner; kontrak
  T-102b perlu menyimpan data yang dibutuhkan (mis. probabilitas per grup)
- Satu komponen garis DA ±94 px di area tangan (frame 90) tidak bisa dibedakan dari garis sah hanya
  dengan ukuran → dievaluasi lagi di T-201/T-302
- Benda yang dipegang: ditunda (tidak ada kelasnya di Sapiens2)

### D-010 — Kontrak pipeline baru (Sapiens2-seg + DA-V2 Small)
**Tanggal:** 2026-09-29
**Konteks:** D-009 mengganti backend ke Sapiens2-seg 0.8B + Depth Anything V2 Small dan menambah garis
oklusi ke style target. Kontrak lama (`masks/*.png` biner → `masks_smooth` → kontur `RETR_EXTERNAL`) tidak
menyimpan batas antar grup, kedalaman, atau data untuk menstabilkan peta grup (known issue D-009).
Ditulis ulang di T-102b sesi 1, berdasarkan metode yang terbukti di T-102c (`scripts/sapiens2_exp.py`,
`scripts/look_test.py`). Kontrak lengkap: `01-PIPELINE-SPEC.md`; parameter: `02-STYLE-PARAMS.md`.
**Keputusan (Rio):**
- **Alur:** [1] ingest → [2] segment → [2c] depth → [3] stabilize → [4] vectorize → [5] stylize →
  [6] export. [2b] fallback pose ditunda
- **[2] hanya output mentah:** `seg/classmap` (argmax uint8) + `seg/probs` + `seg/manifest.json` + QC.
  Semua pembersihan spasial dan temporal di [3], supaya stabilisasi bekerja dari probabilitas. Softmax
  di CPU (sisa VRAM run 0.8B hanya ±42 MiB). `argmax(probs)` vs `classmap` dilaporkan sebagai % beda
  (kuantisasi uint8), bukan gagal keras
- **GPU:** satu model di GPU pada satu waktu; [2] dan [2c] masing-masing proses sendiri. Cek VRAM bebas
  sebelum load: seg 0.8B 3300 MiB, 0.4B 2300 MiB, DA 500 MiB → kurang = berhenti dengan pesan jelas.
  Revision checkpoint HF di-pin di YAML
- **Model seg eksplisit** (`segment.model: 0.8b`). VRAM kurang → berhenti (saran: tutup aplikasi lain,
  atau `--seg-model 0.4b`). OOM di tengah run → berhenti, tanpa ganti model otomatis. Manifest mencatat
  model; resume menolak model berbeda kecuali `--restart`. Tidak pernah dicampur dalam satu klip (D-009)
- **[3]:** probabilitas kelas → grup → temporal (EMA + optical flow, bobot QC) → argmax → filter pulau
  N = 30 pada peta **grup** → mode filter K = 3. Kedalaman dinormalisasi **per frame** (disparity DA
  hanya benar sampai skala + offset per frame) lalu temporal → `depth_smooth`; metode dipilih di T-302.
  EMA dua arah = saran untuk T-302. Parameter temporal pindah dari style YAML ke `stabilize:` di
  `default.yaml`
- **[4]:** siluet = batas foreground **termasuk lubang** (`RETR_CCOMP`; ruang negatif tertutup wajib
  digambar, filter `min_hole_area`) + batas grup + garis oklusi dari `depth_smooth` apa adanya (tanpa log
  kedua; NMS + hysteresis persentil per klip, jarak ≥ D, skeleton ≥ L). Filter komponen M sebelum
  thinning + `min_stroke_px` setelah tracing. Threshold per klip di `contours/clip_stats.json` (dipakai
  `--preview`)
- **[5]:** satu renderer untuk semua jenis garis, override per tipe (`stroke.by_type`); simplify +
  resample pindah dari [4] ke [5]. Seed jitter = `hash(frame_index, param_seed, track_id)` (P-007)
- **Q1 — probabilitas [2]:** 29 kelas, uint8 = round(p × 255), npz deflate, resolusi kerja. Definisi grup
  bisa berubah tanpa menjalankan ulang [2] (±1.6 jam per klip). Ukuran diukur di 20 frame pertama
  T-102b; tetap format ini kecuali > 3 GB per klip
- **Q2 — grup:** bagian `groups:` di `configs/default.yaml`, divalidasi saat load (nama kelas valid, tiap
  kelas tepat satu grup, kelas tak tercantum = error, `background` dicadangkan, ≤ 255 grup). Hash grup di
  `stable/manifest.json`. `scripts/sapiens2_classes.json` → data paket `src/rotoscope/data/`
- **Q3 — fallback pose:** ditunda (T-501/T-502 `BLOCKED`). Frame gagal QC diberi bobot temporal kecil di
  [3]. QC baru `area_vs_median` (median bergulir, default dikalibrasi di T-102b)
- **Q4 — skema JSON [4]:** stroke bertipe `silhouette` / `silhouette_hole` (tertutup) + `group_boundary` /
  `occlusion` (terbuka), dengan `anchor` (tertutup) dan `track_id` (semua; pencocokan dengan frame
  sebelumnya). Orientasi: siluet searah jarum jam, lubang berlawanan; garis terbuka dinormalkan arahnya
- **Q5 — urutan build:** Phase 1 = T-104a (config) → T-102b → T-105 (depth) → T-106 (stabilize spasial,
  temporal off) → T-103 → T-104b (cli); Phase 2 = T-201a/T-201b/T-202/T-203/T-204; Phase 3 = T-302/T-303/
  T-304/T-305 (kalibrasi ulang N/K/M/D/L/persentil/`min_hole_area`). T-301 digabung ke T-102b. T-107 =
  keluarkan rembg + onnxruntime
- **Disk:** ±1.0–1.9 GB per klip *est.* (probabilitas tanpa kompresi 4.3 GB) → `paths.work_dir` bisa di
  drive lain (C: sisa ±54 GB)
**Alternatif ditolak:**
- Q1: probabilitas per grup (grup berubah → [2] jalan ulang ±1.6 jam; grup sudah berubah 2× di T-102c);
  top-3 kelas (massa hilang di titik temu ≥ 3 kelas, tepat di zona glitch tangan); float16 (2× ukuran,
  kompresi jelek)
- Q2: file grup terpisah (satu file lagi untuk dilacak)
- Q3: fallback pose tetap Phase 5 (mask pose tanpa grup, tidak bisa diuji — 0 frame gagal); hapus total
  (kehilangan cadangan D-002)
- Q4: semua polyline terbuka seperti look test (siluet terpotong, urutan stroke berubah → jitter
  melompat, P-004); korespondensi titik lewat optical flow (berat, ditunda)
- P6: model `auto` (klip bisa diam-diam turun ke 0.4B karena VRAM terpakai aplikasi lain)
- Pembersihan spasial di [2] (stabilisasi dari label yang sudah dibulatkan)

**Hasil T-102b (2026-09-30)** — `src/rotoscope/segment.py`, klip uji 283 frame 480×854:
- `segment.revision` = mapping per model (dua repo HF = dua hash): 0.8b
  `196a627b928676c4429b738ed76f78a21d96c4eb`, 0.4b `449b3c5335e6722bb94990abdd1aa6e612432f22`; runtime wajib
  commit hash 40-hex. `depth.revision` tetap satu string
- Exit code stage [2]: 0 sukses, 1 prasyarat gagal, 3 OOM di tengah run
- probs npz ±0.11 MB/frame → **±41 MB per klip 360 frame** (Q1 terkonfirmasi, jauh < 3 GB)
- Waktu seg 0.8B fp16: 16.5 s/frame → **±99 mnt per 360 frame**; peak VRAM reserved 3276 MiB
- VRAM bebas sebelum load **3314 MiB vs batas 3300 (margin 14 MiB)** — aplikasi GPU lain sedikit saja →
  stage [2] berhenti di cek VRAM
- `qc.area_drop_min` = **0.63 sementara** (satu klip; celah aman 0.608–0.652, W = 49); QC klip uji 0/283 gagal

**Hasil T-105 (2026-09-30)** — `src/rotoscope/depth.py`, klip uji 283 frame 480×854:
- `depth.revision` = `5426e4f0f36572d16453bbda7a8389317b1bef99` (94.6 MiB). Lisensi dicek tiga lapis saat load:
  nama `Small`, backbone hidden_size 384 (ViT-S), front-matter model card `license: apache-2.0` (README.md
  wajib di cache, dicatat di `depth/manifest.json`)
- Input processor terukur 518×924 untuk frame 480×854; NaN/inf → 0 + `finite: false`, tidak diproses ulang
- **0.19 s/frame** (±1.1 mnt per 360 frame), peak VRAM reserved **424 MiB**; npy 820 KB/frame → ±295 MB per
  klip 360 frame; full run NaN/inf 0
- Uji regresi vs T-102c (`work/t102c/exp/depth_da2s_gpu/`): **283/283 frame identik bit per bit**
- **`src/rotoscope/stage_common.py`** = helper bersama untuk semua stage (tulis atomik + retry Windows,
  `frames.jsonl`, daftar frame, error + exit code, stdout cp1252; khusus GPU: offline/revision/cache HF, VRAM,
  OOM). `segment.py` di-refactor memakainya — bukti perilaku tidak berubah: `test_segment.py` tidak diubah dan
  lolos, `qc_report.json` dari `--qc-only` identik, classmap + probs frame 0–1 (`--limit 2` ke work_dir
  sementara) identik bit per bit dengan `work/seg/`

**Hasil T-106 (2026-09-30)** — `src/rotoscope/stabilize.py` (spasial, temporal mati), klip uji 283 frame 480×854:
- **Aturan manifest beda (Rio, berlaku umum):** stage GPU ([2], [2c]) **tolak** + `--restart` (hasil ±1.6 jam
  terlalu mahal untuk terhapus tanpa sengaja); stage CPU murah, deterministik, tanpa data manual ([3], [4])
  → output basi **dihapus + dihitung ulang otomatis** dengan peringatan (field lama → baru), `--restart` tetap
  ada; output **tanpa manifest** tetap ditolak. Dicatat di prinsip #4 `docs/01`
- Seri eksak argmax grup (kuantisasi uint8) → grup dari `seg/classmap` kalau ikut seri, selain itu id terkecil;
  `seg/classmap` jadi input [3]. Klip uji 5–43 px seri per frame
- Parameter baru `stabilize.depth.log_eps` = 1e-6 dan `stabilize.depth.iqr_min` = 0.01 (**sementara**, T-302);
  range menjamin `depth_smooth` muat float16. Background memakai transformasi yang sama (kontinu, finite).
  `stable/frames.jsonl` masuk kontrak [3]
- **0.12 s/frame** (±43 s per 360 frame); groups ±6.6 KB/frame, depth_smooth 820 KB/frame. depth_smooth finite
  283/283, IQR log foreground 0.077–0.384 (tanpa clamp)
- **Regresi vs T-102c** (pulau N = 30 di peta KELAS → grup → mode K = 3): piksel sama mean 99.989%, min 99.932%;
  100% beda ≤ 2 px dari batas grup (0 di dalam area), komponen beda terbesar 67 px. Penilaian visual Rio:
  garis grup setara T-102c (frame 73–92, 183–202, 225–240); |grad| `depth_smooth` frame 73–92 masih
  menunjukkan batas kaki kanan–kiri (input T-201b). Filter pulau di peta grup = default; kalibrasi N/K tetap
  di T-305 (dengan temporal)

**Hasil T-103 (2026-10-01)** — `src/rotoscope/export.py` (naif, siluet), klip uji 283 frame 480×854:
- Satu jalur encode (`FrameSource`): Phase 1 `SilhouetteSource` (`stable/groups`, grup ≠ 0 = foreground);
  Phase 2 cukup menambah sumber `strokes`. Frame di-pipe `rawvideo` ke ffmpeg (tanpa PNG sementara), libx264
  yuv420p, tulis `.tmp` → verifikasi ffprobe → `os.replace`. Klip uji: 497.6 KiB, 2.4 s, ffprobe sesuai
- **Nama file (Rio):** `export.filename` default `"{source}.mp4"` (nama video sumber, disanitasi), bukan
  `animation.mp4`; file tujuan milik video sumber LAIN **ditolak** (hasil klip lain tidak pernah ditimpa).
  Manifest `<nama>.export.json` memuat identitas klip (sha256 `meta.json` + `source_path`); input berbeda → basi
- **Audio (Rio):** `export.audio` default **false** — audio meme hampir selalu milik pihak ketiga (musik);
  tambahkan audio dari library berlisensi di editor platform (TikTok/CapCut). `true` → aac, `-shortest`;
  sumber tanpa audio = **error**
- **Penyimpangan disetujui:** file tujuan tanpa manifest ditolak (`--restart` menimpa); pengaman sumber lain
  tidak bisa dilewati `--restart`
- **Known issue (→ T-108) — DISELESAIKAN di T-108 (2026-10-01):** manifest [2]/[2c]/[3] tidak memuat identitas
  klip. Kalau klip lain di-ingest ke `work_dir` yang sama, resume [2] menganggap output klip lama valid → hasil
  salah **tanpa error**. Mitigasi sementara: `work_dir` per klip (dicatat di T-104b; layout akhir
  `work/clips/<stem>/`, lihat "Hasil T-104b"); perbaikan: lihat "Hasil T-108"

**Hasil T-108 (2026-10-01)** — identitas klip di manifest [2]/[2c]/[3] (`stage_common.py`, `segment.py`,
`depth.py`, `stabilize.py`, `export.py`), klip uji 283 frame:
- **Identitas** = `{meta_sha256 (byte meta.json), source_path}` di field `clip`; `meta.json` terbukti
  byte-deterministik (ingest ulang sama → identik; `target_fps` / `working_width` / `source_path` beda → beda).
  Bentuk sama dengan manifest export T-103, jadi manifest export lama tetap valid
- **Perilaku (Rio, revisi Tahap 1):** identitas beda → [2]/[2c] tolak sebelum `resolve_revision` / load backend
  (juga `--limit`, `--qc-only`), satu-satunya jalan `--restart` (`--adopt` ikut ditolak); [3] berhenti kalau
  input [2]/[2c] milik klip lain / tanpa identitas, hitung ulang otomatis hanya kalau manifest [3] sendiri
  yang beda; export berhenti kalau `stable/manifest.json` tanpa / beda identitas
- **`--adopt`** ([2], [2c]): migrasi manifest lama tanpa inferensi (pernyataan pengguna; cek kewajaran:
  `frame_size`, semua frame valid, tanpa file yatim). Klip uji: 8.45 s ([2]) + 0.70 s ([2c]); field lain manifest
  tidak berubah (hanya `clip` + `adopted_utc`)
- **Verifikasi:** [3] hitung ulang setelah adopt → 566 file `groups` + `depth_smooth` sha256 IDENTIK; export
  di-encode ulang sekali (hanya `stable.created_utc`), lalu dilewati; `segment --qc-only` dan `depth` pada data
  asli lolos (exit 0, model tidak dimuat); simulasi klip lain di salinan work_dir + test ingest nyata klip
  kedua: [2]/[2c] exit 1, [3] berhenti lalu hitung ulang setelah input searah, export tidak menimpa MP4 sumber lain
- **Batas yang diketahui** (rinci di `docs/01` "Identitas klip"): path video berubah = `--restart`; `--adopt`
  = pernyataan pengguna; `meta.json` tanpa ukuran/hash file video; determinisme hanya terbukti pada satu build ffmpeg

**Hasil T-104b (2026-10-01)** — `src/rotoscope/cli.py` + `__main__.py` (`python -m rotoscope run|ingest|segment|depth|
stabilize|export|download`), 🎯 **milestone Phase 1 tercapai**. Rincian: `docs/01` bagian "CLI".
- **Folder kerja per klip** `<paths.work_dir>/clips/<stem>/` (bukan `work/<stem>/`; keputusan Rio): `stem` = fungsi
  sanitasi `{source}` export; `clips/` memisahkan data klip dari folder eksperimen di `work/` sehingga tanpa daftar
  nama dicadangkan. `--work-dir` ditambahkan ke `main()` segment / depth / stabilize / export (menang atas config;
  `paths.out_dir` tetap) — alternatif ditolak: YAML sementara, env var, `python -c`
- **Pre-flight CPU-only sebelum apa pun:** file video, bentrok folder kerja, bentrok target export (pengaman export
  tidak bisa dilewati `--restart`, jadi klip bernama sama tidak boleh gagal setelah ±78 mnt GPU); pesan tidak
  menyarankan `--restart`
- **Ingest selalu ulang** (opsi A, Rio) — opsi B (lewati kalau `meta.json` cocok) ditolak karena melebarkan
  batas (c) identitas klip (video lain di path yang sama tidak terdeteksi); ingest hanya detik dan byte-deterministik
- **Restart = graf dependensi** (`run --restart-from`; depth tidak bergantung pada segment) dan **`--yes` wajib untuk
  setiap penghapusan hasil GPU** lewat cli (±78 mnt per klip 283 frame); tanpa `--yes`: tampilkan jumlah file + estimasi
  waktu GPU, exit 1. Opsi ditolak: `run --restart` polos (satu salah ketik menghapus semuanya), restart hanya di
  subperintah stage (stage hilir yang basi mudah terlupa)
- **Ditolak (Rio):** parameter YAML `cli.qc_fail_warn_ratio` — frame gagal QC cukup satu peringatan, run lanjut
- **Tanpa fallback model otomatis** (D-009): `--seg-model` hanya diteruskan; VRAM kurang → exit 1 dan berhenti
- **Proses:** stage GPU = subprocess (stdout/stderr diwariskan), induk tanpa torch/CUDA (diuji), anak di-terminate
  saat induk di-Ctrl+C / exception; pesan error stage memakai perintah CLI lengkap (`… --restart --yes`)
- **Terukur** (klip 5 s `samples/test_short.mp4`, 119 frame, 0.8b): `run` dari nol 1943.8 s (segment 1897.1 s,
  16.2 s/frame, peak 3276 MiB reserved; depth 32.8 s, 424 MiB; stabilize 12.0 s; export 1.0 s); `run` ulang dan
  `run samples/test.mp4` → semua dilewati, exit 0 (±21 s / ±32 s, tanpa GPU kerja)
- **Batas yang diketahui:** memindahkan file video = `source_path` berubah → bentrok folder kerja (hapus folder klip
  lama sendiri atau `--restart-from ingest --yes` tidak membantu karena pre-flight menolak lebih dulu); nama video
  sama di folder berbeda = ganti nama salah satunya

---

## Pitfall yang sudah diketahui

### P-001 — Temporal flicker / boiling 🔴 RISIKO TERTINGGI
Segmentasi per-frame pada video low-quality menghasilkan kontur yang "mendidih".
Sedikit boil = hand-drawn feel (diinginkan). Boil tak terkendali = terlihat rusak.
**Mitigasi:** stage `stabilize` (EMA + optical flow warp). Alokasikan waktu terbanyak di sini.
Parameter `stabilize.temporal.boil_preserve` (`default.yaml`, D-010) mengontrol seberapa banyak getaran dipertahankan.

### P-002 — Motion blur pada tangan/kaki
Saat subjek menari cepat, limb blur → segmentasi gagal atau limb menyatu ke badan.
**Mitigasi (D-010):** frame gagal QC diberi bobot temporal kecil di [3]
(`stabilize.temporal.qc_fail_weight`), sehingga diisi dari frame tetangga. Fallback pose ditunda
(T-501/T-502 `BLOCKED`). Terima bahwa frame tercepat tidak akan sebersih frame statis. Referensi B
kebetulan pose-nya tenang.
**Catatan (2026-09-25):** model card MediaPipe SelfieMulticlass juga menyebut kualitas mask turun
pada gerakan cepat, noise, backlit, dan occluder besar — fallback tetap dibutuhkan apa pun
backend yang dipilih di D-008.

### P-003 — Perubahan topologi kontur
Saat lengan menyilang badan, jumlah kontur berubah → garis "meletus" antar frame.
**Mitigasi (D-010):** siluet `RETR_CCOMP` (kontur luar + lubang) dengan `track_id` (pencocokan
dengan frame sebelumnya, [4]) + temporal pada probabilitas grup di [3]. Deteksi di QC via `big_blobs`.
Mitigasi lama (`RETR_EXTERNAL` + komponen terbesar saja) tidak berlaku lagi.

### P-004 — Point correspondence drift
Kalau titik ke-0 kontur tidak konsisten posisinya antar frame, garis akan tampak
"berputar". **Mitigasi:** rotasi urutan titik ke anchor anatomis tetap setelah resampling.

### P-005 — SAM 2 + Turing: bfloat16 tidak didukung
Contoh kode resmi SAM 2 memakai `torch.autocast(dtype=torch.bfloat16)`. Arsitektur Turing
(GTX 1650 Ti, SM 7.5) tidak punya dukungan bf16 native — bf16 baru masuk di Ampere.
**Fix:** ganti ke `torch.float16`. Ini akan jadi error pertama kalau copy-paste dari docs.

### P-006 — onnxruntime-gpu setup di Windows
Butuh CUDA 12.x + cuDNN 9 dengan versi yang cocok persis. Sumber error setup yang sering.
**Mitigasi:** mulai dengan `onnxruntime` CPU. Jangan blokir progress pipeline karena ini.

### P-007 — Jitter non-deterministic
Kalau jitter pakai random murni, render ulang menghasilkan animasi berbeda → tidak bisa
di-debug, tidak bisa direproduksi. **Fix:** seed dari `hash(frame_index, param_seed, track_id)`
(hash stabil antar proses, mis. crc32; `track_id` ditambahkan di D-010 supaya pola getar tidak melompat
saat urutan stroke berubah).

### P-008 — Claude Code: PATH tidak propagasi setelah install
`claude` tidak dikenali setelah install di terminal yang sama.
**Fix:** tutup dan buka ulang PowerShell.

---

## Template entry baru

```
### D-0XX / P-0XX — [judul]
**Tanggal:**
**Konteks:**
**Keputusan / Gejala:**
**Alasan / Fix:**
**Alternatif ditolak:**
```
