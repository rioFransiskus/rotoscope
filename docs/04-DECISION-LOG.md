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
**Status:** DIPILIH — menunggu uji kelayakan hardware **T-102c**
**Konteks:** Siluet saja kehilangan keterbacaan pose saat lengan/kaki menempel ke badan (review
visual Rio, T-102a). Mask MediaPipe tidak menyimpan batas antar anggota tubuh.
**Keputusan:**
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
- Python ≥3.12 + PyTorch ≥2.7 (project sekarang 3.11.9) → strategi environment diputuskan di T-102c
- GPU Turing: tanpa bf16 (P-005) → fp16/fp32
- Pin versi PyTorch (wheel cu128 masih mendukung sm_75)
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

---

## Pitfall yang sudah diketahui

### P-001 — Temporal flicker / boiling 🔴 RISIKO TERTINGGI
Segmentasi per-frame pada video low-quality menghasilkan kontur yang "mendidih".
Sedikit boil = hand-drawn feel (diinginkan). Boil tak terkendali = terlihat rusak.
**Mitigasi:** stage `stabilize` (EMA + optical flow warp). Alokasikan waktu terbanyak di sini.
Parameter `temporal.boil_preserve` mengontrol seberapa banyak getaran dipertahankan.

### P-002 — Motion blur pada tangan/kaki
Saat subjek menari cepat, limb blur → segmentasi gagal atau limb menyatu ke badan.
**Mitigasi:** fallback pose (D-002). Terima bahwa frame tercepat tidak akan sebersih
frame statis. Referensi B kebetulan pose-nya tenang.
**Catatan (2026-09-25):** model card MediaPipe SelfieMulticlass juga menyebut kualitas mask turun
pada gerakan cepat, noise, backlit, dan occluder besar — fallback tetap dibutuhkan apa pun
backend yang dipilih di D-008.

### P-003 — Perubahan topologi kontur
Saat lengan menyilang badan, jumlah kontur berubah → garis "meletus" antar frame.
**Mitigasi:** `RETR_EXTERNAL` + ambil komponen terbesar saja. Deteksi di QC via `component_count`.

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
di-debug, tidak bisa direproduksi. **Fix:** seed dari `hash(frame_index, param_seed)`.

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
