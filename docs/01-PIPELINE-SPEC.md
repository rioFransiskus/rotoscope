# 01 — PIPELINE SPEC

Arsitektur: **Segmentation → Contour → Stylized Stroke** (Arsitektur A, D-001).
Segmentasi bagian tubuh = **Sapiens2-seg** (29 kelas), garis oklusi = **Depth Anything V2 Small**
(D-009). Kontrak di bawah ditulis ulang di T-102b sesi 1 — **D-010**.

## Diagram alur

```
video.mp4
   │
   ├─[1]  ingest ─────► frames/*.png + meta.json                  (24 fps, resolusi kerja)
   │
   ├─[2]  segment ────► seg/classmap/*.png + seg/probs/*.npz       (GPU, proses sendiri)
   │         │           + seg/manifest.json + qc_report.json
   │         └─[2b] fallback pose — DITUNDA (BLOCKED, D-010); frame gagal QC ditangani di [3]
   │
   ├─[2c] depth ──────► depth/*.npy + depth/manifest.json          (GPU, proses sendiri)
   │
   ├─[3]  stabilize ──► stable/groups/*.png + stable/depth_smooth/*.npy
   │
   ├─[4]  vectorize ──► contours/*.json + contours/clip_stats.json (polyline bertipe)
   │
   ├─[5]  stylize ────► strokes/*.svg + strokes/*.png
   │
   └─[6]  export ─────► out/animation.mp4 + out/svg/*.svg
```

Semua path relatif terhadap `paths.work_dir` (default `work/`) dan `paths.out_dir` (default `out/`).
`paths.work_dir` boleh diarahkan ke drive lain — data antara ±1–5 GB per klip (lihat anggaran disk).

## Prinsip (D-010)

1. **Setiap stage baca dari disk dan tulis ke disk** → resumable; tiap stage bisa dijalankan ulang
   sendiri (D-007). Iterasi style = jalankan ulang [5] saja.
2. **[2] hanya menyimpan output MENTAH model** (probabilitas + class map) + QC. Semua pembersihan
   spasial dan temporal ada di [3], supaya stabilisasi bekerja dari probabilitas, bukan label yang
   sudah dibulatkan.
3. **Satu model di GPU pada satu waktu; tiap stage GPU ([2], [2c]) = proses sendiri.** Proses yang
   selesai pasti melepas CUDA context (`torch.cuda.empty_cache()` tidak).
4. **Resume per frame:** frame dengan output valid dilewati. Manifest tiap stage mencatat model /
   parameter; output yang dibuat dengan model / parameter berbeda dianggap basi. Perilaku saat manifest
   beda (T-106):
   - **Stage GPU ([2], [2c]):** **tolak** + `--restart` — menghapus hasil ±1.6 jam tanpa sengaja terlalu mahal.
   - **Stage CPU murah, deterministik, tanpa data manual ([3], [4]):** output basi **dihapus lalu dihitung
     ulang otomatis**, dengan peringatan yang menyebut field yang berubah (hash lama → baru). `--restart`
     tetap ada untuk memaksa hitung ulang.
   - Output **tanpa manifest** tetap **ditolak** (asal-usulnya tidak diketahui), bukan dihapus.
   - **Identitas klip** (T-108, di bawah): output milik klip lain / tanpa identitas diperlakukan lebih keras
     daripada parameter beda — lihat "Identitas klip".
5. **Semua parameter di YAML** (`configs/default.yaml` untuk pipeline, `configs/styles/*.yaml`
   untuk style), default = hasil T-102c. Daftar + range: `02-STYLE-PARAMS.md`.
6. **Deterministic:** seed jitter = `hash(frame_index, param_seed, track_id)` (P-007).

### Identitas klip (T-108, D-010)

Satu `work_dir` = satu klip. Supaya klip lain yang di-ingest ke `work_dir` yang sama tidak membuat resume
memakai output klip lama tanpa error, manifest [2], [2c], [3] (dan export sejak T-103) memuat field `clip`:

```json
"clip": {"meta_sha256": "<sha256 byte meta.json>", "source_path": "<meta.json source_path>"}
```

- **Definisi:** helper `stage_common.clip_identity` (`clip_identity_from_bytes` untuk byte yang sudah dibaca).
  Perbandingan memakai `meta_sha256`; `source_path` untuk pesan. `meta.json` ditulis ingest tanpa timestamp
  (13 field), jadi ingest ulang video + parameter yang sama menghasilkan byte identik (test
  `test_meta_json_is_byte_deterministic`), sedangkan `target_fps` / `working_width` / `source_path` berbeda →
  identitas berbeda. Bentuk `clip` sama dengan manifest export lama, jadi manifest export tidak jadi basi.
- **Perilaku saat identitas berbeda:**
  - **[2], [2c] (GPU):** **tolak** (exit 1) SEBELUM `resolve_revision` / load backend; pesan menyebut klip lama
    (`source_path` + hash pendek) vs klip baru. Berlaku juga untuk `--limit` dan `--qc-only` ([2]). Satu-satunya
    jalan: `--restart`. `--restart` melewati cek (output lama memang dihapus).
  - **Manifest [2]/[2c] lama tanpa `clip`:** ditolak dengan saran `--adopt` (output memang milik klip ini) atau
    `--restart`.
  - **[3] (CPU):** input [2]/[2c] milik klip lain → **berhenti** (exit 1; saran `--restart` stage itu atau ingest
    klip yang benar); input tanpa `clip` → berhenti (saran `--adopt` / `--restart`). Hitung ulang otomatis
    dengan peringatan hanya kalau manifest [3] SENDIRI yang beda (termasuk `clip` — klip berubah setelah
    [2]/[2c] dijalankan ulang — atau manifest [3] lama tanpa `clip`, dihitung ulang sekali).
  - **[6]:** `stable/manifest.json` tanpa `clip` / milik klip lain → berhenti, jalankan ulang [3]. Pengaman
    "MP4 milik sumber lain" tidak berubah.
- **`--adopt` ([2], [2c]; migrasi manifest lama tanpa `clip`):** mencatat identitas `meta.json` saat ini ke
  manifest yang ada TANPA inferensi dan tanpa GPU/model (field lain, termasuk `created_utc`, tidak berubah;
  ditambah `clip` dan `adopted_utc`; satu record `adopt` di `frames.jsonl`). Ringkasan (`source_path`,
  `frame_count`, hash pendek, jumlah frame valid) dicetak SEBELUM menulis. Cek kewajaran: `frame_size`
  manifest = `meta.json`, semua frame klip valid, tanpa file output di luar daftar frame. Manifest yang sudah
  memuat identitas sama → no-op; identitas beda → ditolak. Tidak bisa digabung dengan `--restart` / `--limit` /
  `--qc-only` / `--download` (exit 1). [3] tidak butuh `--adopt`.

**Batas yang diketahui:**
- (a) File video dipindah / di-ingest dari path lain → `source_path` berubah → identitas berbeda → [2]/[2c]
  hanya bisa `--restart`; `--adopt` ditolak.
- (b) `--adopt` adalah **pernyataan pengguna** bahwa output milik klip ini. Cek kewajaran tidak bisa
  membedakan klip lain yang ber-`frame_size` dan `frame_count` sama.
- (c) `meta.json` tidak memuat ukuran / hash file video: video lain dengan 13 field yang sama (fps, durasi,
  ukuran, jumlah frame, rotasi, audio) di path yang sama tidak terdeteksi.
- (d) Determinisme `meta.json` hanya terbukti pada mesin + build ffmpeg yang sama; update ffmpeg lalu ingest
  ulang bisa mengubah identitas (belum terbukti) → konsekuensinya `--restart` / hitung ulang, bukan hasil salah.

---

## Kontrak per modul

Resolusi kerja = resolusi `frames/` (`meta.json` `working_width` × `working_height`; klip uji
480×854). Estimasi disk = klip 15 s, 360 frame, 480×854 (409 920 px/frame); tanda *est.* = harus
diukur saat implementasi.

### [1] `ingest.py` — `DONE` (T-101)
- **In:** path video, `target_fps` (default 24), `working_width` (default 720)
- **Out:** `frames/frame_%05d.png`, plus `meta.json` (fps asli, durasi, jumlah frame, resolusi kerja)
- **Lib:** ffmpeg via subprocess (`fps` + `scale`), ffprobe untuk metadata
- Resize proporsional. Jangan upscale kalau sumber lebih kecil.

### [2] `segment.py` — Sapiens2-seg (GPU, proses sendiri)
- **In:** `frames/*.png`, `meta.json`; config `segment`, `qc`
- **Model:** `segment.model` **eksplisit**, default `0.8b` = Sapiens2-seg 0.8B fp16 GPU; `0.4b` =
  Sapiens2-seg 0.4B fp16 GPU (hanya kalau diminta, untuk **seluruh** klip — tidak pernah dicampur).
  Lewat Hugging Face Transformers, revision checkpoint di-pin per model (`segment.revision.<model>`),
  `HF_HUB_OFFLINE=1` + `local_files_only=True`. Input = image processor default 1024×768 **stretch**
  (`do_pad=false`). Tanpa bf16 (P-005).
- **Checkpoint + revision** (run berjalan offline, jadi revision yang di-pin HARUS sudah ada di cache HF):
  - `segment.revision` = mapping per model (dua repo HF = dua commit hash). Default = snapshot di cache
    dari T-102c: `0.8b` `196a627b928676c4429b738ed76f78a21d96c4eb`, `0.4b`
    `449b3c5335e6722bb94990abdd1aa6e612432f22`.
  - Saat runtime revision model yang dipakai wajib **commit hash 40-hex**; `null` atau nama branch
    (mis. `main`) → **berhenti**.
  - Unduhan checkpoint = **langkah terpisah sekali jalan** (online), **bukan** bagian dari run. Sementara:
    `python -m rotoscope.segment --download [--seg-model 0.4b]` (dibungkus subperintah `download` di
    T-104b). Revision `null` → unduh `main` lalu cetak hash-nya untuk di-pin.
  - Revision tidak ada di cache → **berhenti** sebelum cek VRAM, dengan pesan jelas + perintah unduh
    yang harus dijalankan.
- **Inferensi per frame:** logits → interpolasi ke resolusi kerja (di GPU, seperti
  `post_process_semantic_segmentation`) → pindah ke CPU → **softmax di CPU** (sisa VRAM di run 0.8B
  hanya ±42 MiB) → `probs` uint8 = round(p × 255) + `classmap` = argmax logits. Logits NaN/inf →
  diganti 0 sebelum softmax, output tetap ditulis, `finite: false` di `frames.jsonl` → gagal QC.
- **Tulis atomik:** tiap file output (classmap, probs, manifest, qc_report) ditulis ke `<nama>.tmp`
  lalu `os.replace`. Di Windows `os.replace` bisa gagal sesaat (`PermissionError`, file dikunci
  antivirus/indexer) → dicoba ulang beberapa kali dengan jeda singkat, lalu error jelas. File `*.tmp`
  tidak pernah dianggap output dan dihapus di awal run. `frames.jsonl` = append per baris (baris
  terakhir yang terpotong diabaikan saat dibaca).

**Out:**

| path | isi | dtype | resolusi | disk / klip |
|---|---|---|---|---|
| `seg/classmap/frame_%05d.png` | id kelas 0–28 (argmax) | uint8 | kerja | ±5–10 MB *est.* |
| `seg/probs/frame_%05d.npz` | array `probs` shape (29, H, W), round(p × 255), `np.savez_compressed` (deflate) | uint8 | kerja | mentah 4.28 GB; terkompresi 0.1–0.5 GB *est.* |
| `seg/manifest.json` | model (`0.8b`/`0.4b`), model id, revision, precision, processor (size, `do_pad`), `num_labels`, nama 29 kelas, `clip` (identitas klip, T-108), `adopted_utc` (hanya kalau lewat `--adopt`) | – | – | < 10 KB |
| `seg/frames.jsonl` | log per frame: waktu, peak VRAM reserved/allocated, finite, % beda argmax | – | – | < 200 KB |
| `qc_report.json` | QC per frame + ringkasan (tabel di bawah) | – | – | < 200 KB |

- **Ukuran probs:** diukur di 20 frame pertama T-102b. Tetap format ini kecuali > 3 GB per klip.
- **Cek konsistensi:** `argmax(probs)` vs `classmap` — kuantisasi uint8 bisa menggeser piksel yang
  nyaris seri. **Laporkan % piksel beda** per frame (`frames.jsonl`) + ringkasan; **bukan** gagal keras.
  `classmap` = acuan untuk QC; `probs` = input [3].
- **Nama kelas:** data paket `src/rotoscope/data/sapiens2_classes.json` (dipindah dari `scripts/`;
  config HF hanya `LABEL_0..28`). Self-check saat load: jumlah kelas = `num_labels` (29).

**Cek VRAM + fallback (D-009, D-010):**
1. Setelah CUDA init, sebelum load model: `torch.cuda.mem_get_info()` bebas <
   `segment.vram_min_free_mib[model]` (0.8b: 3300, 0.4b: 2300) → **berhenti** dengan pesan jelas:
   VRAM bebas vs dibutuhkan, saran tutup aplikasi lain, atau jalankan ulang dengan `--seg-model 0.4b`.
2. OOM di tengah run → catat frame + kondisi VRAM, **berhenti**. Tidak ada ganti model otomatis.
3. Resume: frame dilewati kalau `classmap` valid (terbaca, uint8, ukuran = frame, id < 29) **dan**
   `probs` valid (terbaca, uint8, shape (29, H, W)). `seg/manifest.json` ditulis sebelum frame pertama.
   **Perbedaan apa pun** antara manifest dan run yang diminta — model, model id, revision, precision,
   processor (size, `do_pad`), `num_labels`, ukuran frame — → **tolak** dengan pesan (daftar field yang
   beda); output di `seg/` tanpa manifest juga ditolak. `--restart` menghapus output [2] lama (`seg/` +
   `qc_report.json`) lalu mulai dari awal. Semua frame sudah valid → GPU tidak disentuh, langsung QC.
4. **Identitas klip (T-108):** manifest memuat `clip`; klip lain / manifest tanpa `clip` → tolak SEBELUM
   `resolve_revision` / load backend (juga `--limit`, `--qc-only`) — lihat "Identitas klip" di Prinsip.

**Entry point sementara** (sampai `cli.py`, T-104b): `python -m rotoscope.segment [--config PATH]
[--seg-model 0.8b|0.4b] [--restart] [--limit N] [--qc-only] [--download] [--adopt]`. `--limit N` =
pastikan N frame pertama valid (QC dilewati kecuali semua frame klip valid); `--qc-only` tanpa GPU/torch,
butuh semua frame valid. `--adopt` = catat identitas klip ke manifest lama tanpa inferensi (terukur klip
uji 283 frame: 8.45 s); tidak bisa digabung dengan `--restart` / `--limit` / `--qc-only` / `--download`.

**Exit code** (dipakai `cli.py`, T-104b):

| kode | arti |
|---|---|
| 0 | sukses |
| 1 | prasyarat gagal: config, revision null / bukan hash / tidak ada di cache, CUDA tidak ada, VRAM bebas kurang, manifest beda, frame hilang, tulis file gagal |
| 3 | OOM di tengah run (frame + kondisi VRAM dicatat di `seg/frames.jsonl`) |

**QC** — langkah per klip setelah semua frame selesai (butuh median bergulir); bisa dijalankan
sendiri (`--qc-only`). Foreground = `classmap ≠ 0`.

| Metrik | Kondisi gagal | Default |
|---|---|---|
| `area_ratio` | luas foreground < `qc.area_min` atau > `qc.area_max` dari frame | 0.03 / 0.70 |
| `iou_prev` | IoU foreground dengan frame sebelumnya < `qc.iou_min` (frame pertama: `null`; dua-duanya kosong: 1.0) | 0.55 |
| `big_blobs` | jumlah komponen foreground 8-arah > `qc.blob_min` × area frame melebihi `qc.max_big_blobs` | 0.05 / 1 |
| `area_vs_median` | area / median area di jendela **digeser** `qc.area_median_window` (W) frame < `qc.area_drop_min` — lihat definisi di bawah | 49 / 0.63 (**sementara** — T-102b, satu klip: MediaPipe T-102a kaki hilang frame 1–7 maks 0.608, Sapiens2 min 0.652) |
| `label_agreement_prev` | dilaporkan saja (kelas sama di irisan foreground dengan frame sebelumnya; irisan kosong: `null`) | – |
| `finite` | logits mengandung NaN/inf (dari `frames.jsonl`; tidak tercatat → `null`, tidak gagal) | – |

**Definisi `area_vs_median`** (n = jumlah frame klip, frame ke-i berindeks 0, h = (W − 1) / 2): jendela
selalu W sampel dan memuat frame i — `start = min(max(i − h, 0), n − W)`, jendela = frame
`[start, start + W)`; di tengah klip jendela berpusat di i, di awal/akhir klip jendela **digeser** (bukan
dipotong), jadi jumlah sampel tetap W. n < W → seluruh klip. Median area jendela = 0 → `null` (tidak
gagal; `area_ratio` sudah gagal).
- **Batas metrik:** zona buruk ≥ separuh jendela (≥ 25 frame, ±1 s pada W = 49) ikut menurunkan median →
  **tidak tertangkap**.
- `qc.area_median_window` jangan diubah tanpa kalibrasi ulang `area_drop_min`: celah aman MediaPipe
  kaki hilang vs Sapiens2 di klip uji = W 25 (0.708, 0.757), W 49 (0.608, 0.652), W 73 tanpa celah.

`qc_report.json` per frame: semua metrik di atas + `fail_reasons` (list nama metrik yang gagal).
Frame gagal QC **tidak** diganti di [2]; [3] memberinya bobot temporal lebih kecil (D-010, Q3).

### [2b] `fallback_pose.py` — DITUNDA (`BLOCKED`, D-010)
Foreground Sapiens2 = 0 frame gagal QC di klip uji. Peran fallback diganti: frame gagal QC diisi dari
frame tetangga lewat temporal berbobot di [3]. Dibuka lagi (T-501/T-502) kalau klip nyata gagal QC
dan temporal fill tidak cukup. D-002 (pose hanya fallback, bukan primary) tetap berlaku.

### [2c] `depth.py` — Depth Anything V2 Small (GPU, proses sendiri)
- **In:** `frames/*.png`; config `depth`. Tidak bergantung pada [2] (urutan [2]/[2c] bebas).
- **Model:** `depth-anything/Depth-Anything-V2-Small-hf` (**Apache-2.0**), fp32 GPU, revision di-pin.
  Validasi saat load: model id wajib varian **Small** — Base/Large CC-BY-NC DILARANG.
- **Checkpoint + revision:** aturan sama dengan [2] — `HF_HUB_OFFLINE=1`; default `depth.revision` =
  commit hash snapshot di cache dari T-102c (dicatat di `default.yaml` saat implementasi T-105); unduhan =
  langkah terpisah sekali jalan (subperintah `download`); revision tidak ada di cache → berhenti dengan
  pesan jelas + perintah unduh.
- **Cek VRAM:** bebas < `depth.vram_min_free_mib` (500) → berhenti dengan pesan jelas.
- **Validasi lisensi saat load (tiga lapis):** model id memuat `Small`; backbone `hidden_size` = 384
  (ViT-S); front-matter `license` model card (`README.md`, wajib ada di cache, ikut `--download`) =
  `apache-2.0`, dicatat di manifest.
- **Per frame:** image processor default (`DPTImageProcessor`: sisi pendek 518, rasio dipertahankan,
  kelipatan 14 → frame 480×854 masuk model sebagai **518×924**, terukur T-105) → forward fp32 →
  `post_process_depth_estimation` ke resolusi kerja (GPU) → CPU → float16 → tulis atomik (aturan sama
  dengan [2]).
- Output = **disparity relatif mentah** (hanya benar sampai skala + offset per frame), di-resize ke
  resolusi kerja (`post_process_depth_estimation`).

| path | isi | dtype | resolusi | disk / klip |
|---|---|---|---|---|
| `depth/frame_%05d.npy` | disparity relatif mentah | float16 | kerja | 295 MB (820 KB/frame, terukur T-105) |
| `depth/manifest.json` | model id, revision, lisensi, precision, processor, ukuran input processor, jenis output, ukuran frame, `clip` (identitas klip, T-108), `adopted_utc` (hanya kalau lewat `--adopt`) | – | – | kecil |
| `depth/frames.jsonl` | log per frame: waktu, peak VRAM, finite + jumlah piksel NaN/inf, disparity min / median / max | – | – | kecil |

- float16 aman: langkah kuantisasi ±0.0006 relatif (T-102c), jauh di bawah threshold garis.
- **Resume:** frame dilewati kalau file terbaca, float16, ukuran = frame, dan semua finite. NaN/inf dari
  model diganti 0 sebelum ditulis (`finite: false` + jumlah piksel di `depth/frames.jsonl`), jadi file
  selalu finite dan frame itu **tidak** diproses ulang otomatis; [3] membaca status `finite` dari
  `frames.jsonl`. File tidak finite di disk = rusak → diproses ulang.
- **Manifest:** perbedaan apa pun (model id, revision, lisensi, precision, processor, ukuran input, jenis
  output, ukuran frame) → **tolak**; output tanpa manifest juga ditolak. `--restart` hanya menghapus `depth/`.
- **Identitas klip (T-108):** sama dengan [2] — klip lain / manifest tanpa `clip` → tolak SEBELUM
  `resolve_revision` / load backend (juga `--limit`); jalan: `--restart`, atau `--adopt` untuk manifest lama
  tanpa `clip`. Semua frame valid + identitas cocok → model tidak dimuat (terukur klip uji: 0 diproses, 8.5 s).
- **Entry point sementara** (sampai `cli.py`, T-104b): `python -m rotoscope.depth [--config PATH]
  [--restart] [--limit N] [--download] [--adopt]`. Exit code sama dengan [2] (0 / 1 / 3). `--adopt`: lihat
  [2] (0.70 s pada klip uji).

### [3] `stabilize.py` — stage tersulit, alokasikan waktu paling banyak (CPU)
- **In:** `seg/probs/`, `seg/classmap/` (tie-break seri, T-106), `seg/manifest.json`, `depth/`,
  `depth/manifest.json`, `depth/frames.jsonl` (status `finite`), `frames/` (untuk optical flow, T-303),
  `qc_report.json` (bobot temporal, T-302 — tidak dibaca selama temporal mati); config `groups`, `stabilize`
- **Langkah:**
  1. Probabilitas kelas → **probabilitas grup** (jumlah per grup / 255, float32). Definisi grup:
     `groups:` di `configs/default.yaml` (lihat 02). Tanpa temporal, argmax dihitung dari jumlah uint8
     per grup (integer, eksak — sama dengan argmax float32). `classes` di `seg/manifest.json` wajib sama
     persis dengan `sapiens2_classes.json` (urutan kelas menentukan pemetaan ke grup).
  2. **Temporal** (kalau `stabilize.temporal.enabled`): EMA (rata-rata bergerak berbobot) + warp
     optical flow Farnebäck (`cv2.calcOpticalFlowFarneback`, gerakan per piksel antar frame) pada
     probabilitas grup. Frame gagal QC diberi bobot `stabilize.temporal.qc_fail_weight`. Formula +
     arah (satu arah vs dua arah maju–mundur, disarankan dua arah karena offline) → T-302/T-303.
  3. argmax → peta grup. **Seri eksak** → grup dari `seg/classmap` (argmax logits tanpa pembulatan)
     kalau grup itu ikut seri; selain itu id terkecil. Jumlah piksel seri dicatat per frame (klip uji
     5–43 px/frame). Setelah temporal aktif (T-302) seri eksak jarang, tapi aturan tetap berlaku.
  4. **Filter pulau** pada peta GRUP: komponen 8-arah sebuah grup (termasuk background, jadi lubang
     kecil di badan ikut terisi) < `stabilize.island_min_px` (N = 30) → grup mayoritas di cincin 1 px
     sekelilingnya. Satu lintasan: cincin dibaca dari peta sebelum filter (urutan tidak berpengaruh).
     Seri mayoritas → id terkecil. Komponen yang menyentuh tepi gambar diperlakukan sama; cincinnya
     dipotong di tepi (tanpa piksel virtual). Cincin kosong (komponen = seluruh gambar) → tidak diubah.
     N = 0 → mati. ⚠️ Di T-102c filter ini dijalankan pada peta KELAS sebelum dijadikan grup → dicek ulang
     di T-305 (T-106: hasil setara, lihat D-010 "Hasil T-106").
  5. **Mode filter** peta grup K×K (`stabilize.mode_k`, K = 3): hitungan per grup = box filter tanpa
     normalisasi (`BORDER_REPLICATE`) + 0.5 untuk grup asli piksel; background ikut dihitung. Seri →
     grup asli; seri antara grup lain → id terkecil. K = 1 → mati.
  6. **Kedalaman:** normalisasi **per frame** (disparity DA hanya benar sampai skala + offset per
     frame; kedalaman mentah tidak boleh langsung di-EMA), lalu temporal (kalau
     `stabilize.depth.temporal`). Metode normalisasi dipilih + diuji di T-302. Kandidat:
     - **affine:** (disparity − median foreground) / IQR foreground — menghilangkan skala **dan**
       offset per frame;
     - **log** (`log_median_iqr`, **diimplementasi T-106**): `(log max(d, log_eps) − median) /
       max(IQR, iqr_min)` — menghilangkan skala saja, **tidak** offset. Median + IQR (p75 − p25) dari
       foreground = peta grup bersih ≠ 0 frame itu; foreground kosong → seluruh frame
       (`region: "frame"` di log). IQR < `iqr_min` → pembagi di-clamp (`iqr_clamped` di log).
     - **Background** memakai transformasi yang sama (nilai kontinu + finite): NaN merusak blur/Sobel di
       [4], nilai konstan membuat lompatan palsu di siluet. Range `log_eps` / `iqr_min` (02) menjamin hasil
       muat float16; tidak finite → berhenti dengan error.
     - Frame `finite: false` di `depth/frames.jsonl` disalin ke `stable/frames.jsonl` (`depth_finite`),
       tanpa perlakuan khusus selama temporal mati.
- Optical flow tidak di-cache (±1.2 GB/klip) — dihitung ulang (±20–50 ms/frame *est.*).
- Phase 1–2: `stabilize.temporal.enabled: false` (hanya langkah 1, 3–6 tanpa temporal).
- ⚠️ Jangan over-smooth. Sedikit boil = hand-drawn feel (`boil_preserve`), bukan nol (P-001).

| path | isi | dtype | resolusi | disk / klip |
|---|---|---|---|---|
| `stable/groups/frame_%05d.png` | id grup: 0 = background, 1..G = urutan `groups:` di YAML | uint8 | kerja | ±2.4 MB (±6.6 KB/frame, terukur T-106) |
| `stable/depth_smooth/frame_%05d.npy` | kedalaman ternormalisasi per frame (+ temporal) | float16 | kerja | 295 MB (820 KB/frame) |
| `stable/manifest.json` | section `stabilize` + `stabilize_hash`, `groups` + `groups_hash`, referensi `seg/manifest.json` (model, model id, revision, precision, processor, `num_labels`, frame_size, classes, `created_utc`) + `depth/manifest.json` (model id, revision, lisensi, precision, processor, ukuran input, output, frame_size, `created_utc`), frame_size, `clip` (identitas klip, T-108) | – | – | kecil |
| `stable/frames.jsonl` | log per frame: waktu per langkah, piksel seri, piksel berubah di filter pulau / mode, luas foreground, `depth_finite`, statistik normalisasi (region, median + IQR log, clamp) + min / median / maks `depth_smooth` | – | – | kecil |

- **Tulis atomik + `*.tmp`:** aturan sama dengan [2] (helper `stage_common.py`).
- **Resume:** frame dilewati kalau `groups` valid (PNG uint8, ukuran = frame, id ≤ G) **dan**
  `depth_smooth` valid (float16, ukuran = frame, semua finite); salah satu tidak valid → keduanya ditulis
  ulang. Input frame terpilih (probs, classmap, depth) wajib ada sebelum mulai, dicek isinya saat dibaca;
  kurang / rusak → berhenti dengan pesan + stage yang harus dijalankan. Manifest input: `frame_size` =
  `meta.json`, kelas = `sapiens2_classes.json`. **Identitas klip (T-108):** `clip` di `seg/manifest.json` dan
  `depth/manifest.json` wajib sama dengan `meta.json` saat ini — milik klip lain → berhenti (saran
  `--restart` stage itu / ingest klip yang benar); tanpa `clip` → berhenti (saran `--adopt` / `--restart`).
- **Manifest beda** (hash `stabilize` — seluruh section, termasuk parameter temporal yang sedang mati —,
  hash grup, `clip`, atau referensi input berubah, termasuk `created_utc` setelah `--restart` di [2]/[2c]) →
  output [3] **basi: `stable/` dihapus + dihitung ulang otomatis** dengan peringatan (prinsip #4). Manifest
  [3] lama tanpa `clip` → basi sekali (dihitung ulang; output byte-identik pada klip uji, 566 file). [4] ikut
  basi; [2] tidak. Output tanpa manifest → ditolak. `--restart` menghapus `stable/` saja. [3] tidak punya
  `--adopt`.
- `stabilize.temporal.enabled: true` → berhenti (belum diimplementasi, T-302/T-303).
- **Entry point sementara** (sampai `cli.py`, T-104b): `python -m rotoscope.stabilize [--config PATH]
  [--restart] [--limit N]`. Exit code sama dengan [2]: 0 sukses, 1 prasyarat gagal (3 tidak dipakai —
  tanpa GPU).

### [4] `vectorize.py` (CPU)
- **In:** `stable/groups/`, `stable/depth_smooth/`, `stable/manifest.json`; config `groups`, `vectorize`
- **Langkah:**
  1. **Siluet** = batas foreground (`groups ≠ 0`) **TERMASUK lubang**: `cv2.findContours` mode
     `RETR_CCOMP` (2 level: kontur luar + lubang), bukan `RETR_EXTERNAL` saja. Ruang negatif tertutup
     (mis. lengan bertolak pinggang) wajib tetap digambar. Kontur luar dengan area <
     `vectorize.min_region_area` dibuang; lubang dengan area < `vectorize.min_hole_area` dibuang.
  2. **Batas grup** = batas antar pasangan grup (a, b), keduanya ≠ 0 → polyline terbuka. Tidak
     termasuk batas dengan background (itu sudah menjadi `silhouette` / `silhouette_hole`). Batas di dalam satu
     grup tidak ada (sudah digabung di [3]).
  3. **Garis oklusi** dari `depth_smooth` **apa adanya — tanpa log kedua** (normalisasi sudah di [3]):
     |grad| (Gaussian σ `depth_lines.blur_sigma` → Sobel) → NMS searah gradien (4 bin arah; non-maximum
     suppression = hanya piksel puncak tepi) → hysteresis (8-arah) dengan T_high / T_low = persentil
     **per klip** (`hi_pct` / `lo_pct`) di foreground ter-erode (`erode_px`) → hanya di dalam satu grup,
     jarak ke batas grup terdekat (termasuk siluet) ≥ `min_dist_px` (D) → skeleton, komponen < `min_len_px`
     (L) dibuang.
  4. **Filter komponen garis** 8-arah < `vectorize.line_min_px` (M = 5) pada mask garis (batas grup +
     oklusi) **sebelum thinning**; lalu thinning → tracing jadi polyline (junction dilepas, disambung
     ke ujung jalur) → jalur < `vectorize.min_stroke_px` (6) dibuang.
  5. **Anchor + orientasi + `track_id`** (P-004) — aturan di bawah. Frame diproses **berurutan**
     (pencocokan dengan frame sebelumnya).
- **Threshold per klip:** T_high / T_low dihitung sekali dari **seluruh** klip dan disimpan di
  `contours/clip_stats.json` — `--preview N` memakai nilai ini, bukan persentil dari N frame preview.
  Persentil menyesuaikan diri dengan distribusi `depth_smooth`, tapi D / L / persentil dikalibrasi
  ulang di T-305.

**Out:**

| path | isi | disk / klip |
|---|---|---|
| `contours/frame_%05d.json` | polyline bertipe, titik rapat (±1 px, dibulatkan 0.1 px) | 20–60 MB *est.* |
| `contours/clip_stats.json` | T_high / T_low, persentil, jumlah nilai, referensi `stable/manifest.json` | kecil |

**Skema JSON (D-010, Q4):**

```json
{"frame_index": 87, "width": 480, "height": 854,
 "source": {"seg_model": "0.8b", "groups_hash": "…", "stabilize_hash": "…", "vectorize_hash": "…"},
 "strokes": [
  {"track_id": 1, "type": "silhouette", "closed": true, "groups": ["background"],
   "points": [[x, y], …], "anchor": 0},
  {"track_id": 3, "type": "silhouette_hole", "closed": true, "groups": ["background"],
   "points": [[x, y], …], "anchor": 0},
  {"track_id": 7, "type": "group_boundary", "closed": false, "groups": ["left_arm", "torso"],
   "points": [[x, y], …]},
  {"track_id": 12, "type": "occlusion", "closed": false, "groups": ["left_leg"],
   "points": [[x, y], …], "strength": 0.034}]}
```

| `type` | `closed` | `groups` | keterangan |
|---|---|---|---|
| `silhouette` | true | `["background"]` | kontur luar foreground |
| `silhouette_hole` | true | `["background"]` | lubang foreground (ruang negatif tertutup) |
| `group_boundary` | false | 2 grup (urut sesuai YAML) | batas antar grup |
| `occlusion` | false | 1 grup | lompatan kedalaman di dalam grup; `strength` = rata-rata |grad| |

**Aturan anchor + orientasi + `track_id`:**
- **`silhouette`:** orientasi searah jarum jam (seperti terlihat di layar). Titik ke-0 (`anchor`) =
  titik terdekat ke anchor track yang sama di frame sebelumnya; frame pertama / track baru = titik
  tertinggi grup hair ∪ face (tidak ada → titik tertinggi kontur).
- **`silhouette_hole`:** orientasi berlawanan jarum jam. Anchor = titik terdekat ke anchor track yang
  sama di frame sebelumnya; track baru = titik tertinggi lubang (seri → x terkecil).
- **Garis terbuka** (`group_boundary`, `occlusion`): arah dinormalkan — titik awal = ujung dengan
  proyeksi terkecil pada sumbu utama polyline (seri → y terkecil).
- **`track_id`:** cocokkan dengan stroke frame sebelumnya yang `type` dan `groups`-nya sama dan jarak
  rata-ratanya < `vectorize.track.max_match_dist_px`; tidak ada yang cocok → id baru (bilangan bulat,
  unik per klip). `track_id` dipakai [5] untuk seed jitter.
- Resample ke N titik tetap dilakukan di [5], mulai dari `anchor`.

### [5] `stylize.py` (CPU)
- **In:** `contours/*.json`, style YAML (`configs/styles/*.yaml`)
- **Out:** `strokes/frame_%05d.svg` (stroke tebal-variabel sebagai polygon) + `strokes/frame_%05d.png`
  (raster). Disk: SVG 100–250 MB, PNG 150–500 MB per klip *est.* (tergantung `output_width`).
- **Satu renderer untuk semua `type`**; parameter dasar + override per tipe (`stroke.by_type`).
- Komponen render (metode look test T-102c, `scripts/look_test.py`):
  - `approxPolyDP` (`shape.simplify_epsilon`) → spline Catmull-Rom (`shape.smooth_tension`,
    `shape.spline_steps`) → resample arc-length (`shape.resample_points`, maks 1 titik/px)
  - Width modulation sepanjang path + taper ujung (`taper_px`, `taper_min`)
  - Jitter searah normal, noise 1D, seed = `hash(frame_index, param_seed, track_id)` + indeks pass —
    reproducible (P-007), dan tidak melompat saat urutan stroke berubah
  - Multipass (offset + opacity falloff), supersampling `render.ss` untuk anti-alias
  - Tekstur: brush stamping (raster) atau multi-stroke offset (SVG); paper background layer
- **Lib:** `svgwrite` untuk SVG, OpenCV / `Pillow` untuk raster
- ⚠️ `output_width` (resolusi output terpisah dari resolusi kerja, satuan tebal/jitter relatif) →
  diputuskan sebelum T-203.

### [6] `export.py` (CPU)
- **SVG** (Phase 2): copy `strokes/*.svg` ke `out/svg/` — belum diimplementasi
- **MP4** (T-103, naif): `out/<nama>.mp4`, `<nama>` = `export.filename` (default `"{source}.mp4"`; `{source}` =
  nama video sumber dari `meta.json`, disanitasi). Satu jalur encode untuk semua sumber gambar
  (`export.source`): `"silhouette"` (Phase 1) = `stable/groups/*.png`, grup ≠ 0 → `foreground_color` di atas
  `background_color`; `"strokes"` (Phase 2) = `strokes/*.png` apa adanya — ditolak sampai ada.
- **Encode:** frame RGB mentah di-pipe ke stdin ffmpeg (tanpa PNG sementara) → libx264, `-crf` / `-preset`
  dari YAML, `-pix_fmt yuv420p`, `-r` = `target_fps` `meta.json`, `+faststart`. Dimensi ganjil dipad 1 px warna
  latar (yuv420p butuh genap). Tulis `<nama>.mp4.tmp` → verifikasi → `os.replace` (retry Windows) — MP4 lama
  tidak rusak oleh run gagal.
- **Verifikasi ffprobe** (sebelum `os.replace`): jumlah frame = `frame_count`, fps, codec h264, pix_fmt
  yuv420p, ukuran = resolusi kerja (setelah pad genap), durasi ±1 frame, jumlah stream audio = `export.audio`.
- **Audio** (`export.audio`, default `false`): audio meme hampir selalu milik pihak ketiga (musik) → default
  tanpa audio; tambahkan audio dari library berlisensi di editor platform (TikTok/CapCut). `true` = audio video
  sumber (aac, `-shortest`); sumber tanpa audio (`has_audio` false) atau file sumber hilang = **error**.
- **Manifest** `out/<nama>.export.json`: hash parameter `export` (kecuali `filename`), identitas klip
  (sha256 `meta.json` + `source_path`), referensi `stable/manifest.json` (`stabilize_hash`, `groups_hash`,
  `created_utc`), jumlah frame, ukuran, fps, audio (ukuran + mtime file sumber), hasil ffprobe.
- **Basi / pengaman** (stage CPU murah + deterministik, prinsip #4):
  - manifest cocok + MP4 lolos ffprobe → **dilewati**; parameter / input berubah (termasuk `meta.json` klip
    yang sama) → **di-encode ulang otomatis** dengan peringatan (field lama → baru);
  - file tujuan ada dan manifest menunjuk video sumber **LAIN** → **ditolak** (ubah `export.filename`, atau
    pindah / hapus file itu); `--restart` **tidak** melewati pengaman ini;
  - file tujuan ada **tanpa manifest** → **ditolak** (asal tidak diketahui); `--restart` menimpa.
- **Entry point sementara** (sampai `cli.py`, T-104b): `python -m rotoscope.export [--config PATH] [--restart]
  [--limit N]`; `--limit N` → `<nama>.limitN.mp4` (preview; tanpa manifest, tanpa pengaman, tidak menyentuh
  hasil utama). Exit code: 0 sukses, 1 prasyarat gagal (3 tidak dipakai — tanpa GPU).
- **Identitas klip (T-108):** `stable/manifest.json` tanpa `clip` / milik klip lain → berhenti (jalankan ulang
  [3]; output basi dihitung ulang otomatis). Field `clip` manifest export memakai helper bersama
  `stage_common.clip_identity_from_bytes` (bentuk sama dengan sebelumnya).
- Terukur klip uji (283 frame, 480×854, crf 18): 2.4 s, 497.6 KiB (509 571 B).

### Anggaran disk per klip (360 frame, *est.*)

| stage | disk |
|---|---|
| [2] classmap + probs + QC | 0.1–0.5 GB (tanpa kompresi 4.3 GB) |
| [2c] depth | 0.3 GB |
| [3] groups + depth_smooth | 0.3 GB |
| [4] contours | < 0.1 GB |
| [5] strokes | 0.25–0.75 GB |
| **total [2]–[5]** | **±1.0–1.9 GB** |

C: sisa ±54 GB → arahkan `paths.work_dir` ke drive lain kalau banyak klip disimpan bersamaan.

### Anggaran waktu + VRAM (T-102c, D-005)

| stage | model | VRAM reserved | waktu / klip 360 frame |
|---|---|---|---|
| [2] | seg 0.8B fp16 | 3276 MiB (cek ≥ 3300 bebas) | ±95 mnt (15.85 s/frame) |
| [2] fallback | seg 0.4B fp16 | 2258 MiB (cek ≥ 2300 bebas) | ±48 mnt (7.99 s/frame) |
| [2c] | DA-V2 Small fp32 | 424 MiB (cek ≥ 500 bebas) | ±1 mnt (0.185 s/frame) |
| [3] spasial | CPU | – | ±43 s (0.12 s/frame, T-106) |
| [4]–[5] | CPU | – | belum diukur |

---

## Stack & environment

```
Python 3.11.9
torch==2.7.1+cu118          # GPU untuk Sapiens2-seg + DA-V2 Small. Wheel cu118: driver 517.00
torchvision==0.22.1+cu118   # (CUDA maks 11.7) tidak bisa cu126/cu128; cu118 memuat sm_75 (Turing)
transformers==5.17.0        # Sapiens2 (sejak 5.10.1, Python >= 3.10) + Depth Anything V2
opencv-contrib-python       # contour, optical flow, thinning (ximgproc) — dibawa mediapipe.
                            # JANGAN tambah opencv-python (dua paket OpenCV bentrok di modul cv2)
mediapipe                   # fallback pose (DITUNDA, D-010)
numpy, scipy                # resampling, interpolasi
svgwrite                    # export SVG
Pillow                      # raster render
pyyaml                      # config
ffmpeg                      # ingest + muxing (binary eksternal, gyan.dev essentials build, via subprocess)
```

- Index PyTorch cu118 = `--extra-index-url` di `requirements.txt`; PyPI tetap index utama.
- Checkpoint model di cache Hugging Face (di luar repo), dimuat dengan `HF_HUB_OFFLINE=1`,
  revision di-pin di YAML.
- GPU: GTX 1650 Ti 4 GB, Turing sm_75 → **tanpa bf16** (P-005), fp16/fp32 saja. Attention `sdpa`.
- Versi terkunci: `requirements.txt` (dependency langsung) dan `requirements-lock.txt` (full freeze).
- `rembg` + `onnxruntime` DITOLAK (D-008), masih ada di `requirements.txt` / venv sampai **T-107**.
  `onnxruntime-gpu` tidak dipakai (T-601 SKIP).
- ⚠️ OpenCV 5.x dan mediapipe 1.x adalah versi mayor baru — jangan asumsikan API OpenCV 4.x /
  mediapipe 0.10.x dari tutorial lama.

## Struktur repo

```
rotoscope/
├── src/rotoscope/
│   ├── ingest.py  segment.py  depth.py  fallback_pose.py (ditunda)
│   ├── stabilize.py  vectorize.py  stylize.py  export.py
│   ├── config.py  cli.py
│   ├── stage_common.py   # helper bersama stage: tulis atomik + retry, frames.jsonl, daftar frame,
│   │                     # error/exit code; khusus GPU: offline/revision/cache HF, VRAM, OOM (T-105)
│   └── data/sapiens2_classes.json   # nama 29 kelas (data paket)
├── configs/
│   ├── default.yaml       # paths, segment, depth, qc, groups, stabilize, vectorize
│   └── styles/rough-sketch.yaml
├── assets/        # brushes/, paper/
├── scripts/       # smoke_test.py, alat sekali pakai (A/B, T-102c)
├── samples/       # video test (tidak di-commit)
├── work/          # intermediate (paths.work_dir), gitignored
├── out/           # hasil (paths.out_dir), gitignored
└── tests/
```

## Urutan build (jangan lompat)

| Phase | Isi | Selesai kalau |
|---|---|---|
| 1 | config loader (T-104a) → segment + QC (T-102b) → depth (T-105) → stabilize spasial saja, temporal off (T-106) → export naif (T-103) → cli (T-104b) | Pipeline end-to-end jalan (siluet blok peta grup → MP4) |
| 2 | vectorize: siluet + lubang + batas grup (T-201a), garis oklusi (T-201b), anchor + `track_id` (T-202) → stylize basic, satu renderer + parameter per tipe (T-203) → `--preview` (T-204) | Sudah keluar outline |
| 3 | temporal pada probabilitas grup + kedalaman ternormalisasi (T-302, T-303) → `boil_preserve` (T-304) → kalibrasi ulang N/K/M/D/L/persentil/`min_hole_area` (T-305) | Flicker terkendali |
| 4 | style params lengkap + SVG export | Bisa ganti style dari config |
| 5 | fallback pose — DITUNDA (`BLOCKED`, D-010) | — |
| 6 | (opsional) eksperimen SAM 2 tiny, batch processing | — |
