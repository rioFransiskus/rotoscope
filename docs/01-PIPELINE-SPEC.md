# 01 — PIPELINE SPEC

Arsitektur: **Segmentation → Contour → Stylized Stroke** (Arsitektur A, D-001).
Segmentasi bagian tubuh = **Sapiens2-seg** (29 kelas), garis oklusi = **Depth Anything V2 Small**
(D-009). Kontrak di bawah ditulis ulang di T-102b sesi 1 — **D-010**.

## Diagram alur

```
video.mp4
   │
   ├─[1]  ingest ─────► frames/*.png + meta.json                  (24 fps, resolusi kerja; T-607: trim --start / --end,
   │                     (+ saran crop di <work_dir>/crop/<nama>/)   crop HANYA bila kotak diterima lewat `crop`, D-013)
   │
   ├─[1b] isolate ────► isolate/labels/*.png + isolate/frames.jsonl (GPU, proses sendiri; HANYA bila
   │                     + isolate/manifest.json                     target/points.json ada — T-605, D-011)
   │
   ├─[2]  segment ────► seg/classmap/*.png + seg/probs/*.npz       (GPU, proses sendiri)
   │         │           + seg/manifest.json + qc_report.json
   │         └─[2b] fallback pose — SKIP (2026-10-27, revisi D-010 Q3); frame gagal QC ditangani di [3]
   │
   ├─[2c] depth ──────► depth/*.npy + depth/manifest.json          (GPU, proses sendiri)
   │
   ├─[3]  stabilize ──► stable/groups/*.png + stable/depth_smooth/*.npy
   │                     (+ masker target isolate/ bila berlaku — T-606, D-012)
   │
   ├─[4]  vectorize ──► contours/*.json + contours/manifest.json (+ clip_stats.json, T-201b) (polyline bertipe)
   │
   ├─[5]  stylize ────► strokes/*.svg + strokes/*.png + strokes/manifest.json (T-203a)
   │
   └─[6]  export ─────► out/<nama>.mp4 + out/svg/<nama>/*.svg (T-203b; source "strokes")
```

Path stage di bawah (`frames/`, `seg/`, `depth/`, `stable/`, …) relatif terhadap **folder kerja klip**
`<paths.work_dir>/clips/<nama video>/` (default `work/clips/<nama>/`; satu video = satu folder, lihat "CLI"),
dan `out/` = `paths.out_dir` (default `out/`, tidak per klip). `paths.work_dir` boleh diarahkan ke drive lain —
data antara ±1–5 GB per klip (lihat anggaran disk).
Di luar pipeline: `python -m rotoscope pick` (T-604, BUKAN stage) menulis `target/points.json` (+ `target/points.prev.json`,
`target/pick_dryrun/`) di folder kerja klip; dibaca stage [1b] isolate (T-605, lihat "`pick.py`" dan "[1b] `isolate.py`").
`isolate/` (T-605) dibaca stage [3] (T-606, D-012) — hanya bila masker berlaku (kunci [1b] = titik + frame sekarang); [2] / [2c]
tidak membacanya.
`python -m rotoscope crop` (T-607, D-013; BUKAN stage) menulis keputusan kotak crop `<paths.work_dir>/crop/<nama>/kotak.json` per
SUMBER video (di luar folder klip); dibaca [1] ingest (lihat "[1] `ingest.py`" dan "`crop.py`").

## Prinsip (D-010)

1. **Setiap stage baca dari disk dan tulis ke disk** → resumable; tiap stage bisa dijalankan ulang
   sendiri (D-007). Iterasi style = jalankan ulang [5] saja.
2. **[2] hanya menyimpan output MENTAH model** (probabilitas + class map) + QC. Semua pembersihan
   spasial dan temporal ada di [3], supaya stabilisasi bekerja dari probabilitas, bukan label yang
   sudah dibulatkan.
3. **Satu model di GPU pada satu waktu; tiap stage GPU ([1b], [2], [2c]) = proses sendiri.** Proses yang
   selesai pasti melepas CUDA context (`torch.cuda.empty_cache()` tidak).
4. **Resume per frame:** frame dengan output valid dilewati. Manifest tiap stage mencatat model /
   parameter; output yang dibuat dengan model / parameter berbeda dianggap basi. Perilaku saat manifest
   beda (T-106):
   - **Stage GPU ([2], [2c]):** **tolak** + `--restart` — menghapus hasil ±1.6 jam tanpa sengaja terlalu mahal.
   - **[1b] isolate (GPU, per KLIP — T-605):** tanpa resume per frame (memori SAM berantai antar frame). Basi → seluruh klip
     dihitung ulang otomatis, TETAPI hanya bila `--yes` ada (keputusan Rio: `--yes` wajib setiap kali [1b] memakai GPU, hitung
     baru maupun basi); tanpa `--yes` → berhenti dengan pesan, tidak ada yang dihapus. `target/points.json` dihapus →
     `isolate/` lama dihapus otomatis dengan pesan.
   - **Stage CPU murah, deterministik, tanpa data manual ([3], [4]):** output basi **dihapus lalu dihitung
     ulang otomatis**, dengan peringatan yang menyebut field yang berubah (hash lama → baru). `--restart`
     tetap ada untuk memaksa hitung ulang.
   - Output **tanpa manifest** tetap **ditolak** (asal-usulnya tidak diketahui), bukan dihapus.
   - **Identitas klip** (T-108, di bawah): output milik klip lain / tanpa identitas diperlakukan lebih keras
     daripada parameter beda — lihat "Identitas klip".
5. **Semua parameter di YAML** (`configs/default.yaml` untuk pipeline, `configs/styles/*.yaml`
   untuk style), default = hasil T-102c. Daftar + range: `02-STYLE-PARAMS.md`.
6. **Deterministic:** jitter = fungsi dari (`frame_index` mutlak lewat indeks gambar, parameter, posisi); seed = hash(`param_seed`, salt, kanal [, `track_id` hanya untuk komponen independen]) — T-402 mengganti `hash(frame_index, param_seed, track_id)` per strok (P-007; docs/04 "Keputusan T-402").

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
- **T-607 (D-013):** key `trim` dan `crop` di akhir `meta.json` HANYA bila dipakai → klip tanpa trim / kotak tetap 13 field, byte-identik
  dengan sebelum T-607 (tanpa hitung ulang; dibuktikan di 4 klip lama + `out/`). Klip yang ingest-nya berubah (kotak diterima / trim) →
  `meta.json` lain → identitas lain → [2] / [2c] menolak; pesan penolakannya menyebut `python -m rotoscope run <video> [--start S --end
  E] --restart-from ingest --yes` (MENJALANKAN GPU).
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
  ukuran, jumlah frame, rotasi, audio) di path yang sama tidak terdeteksi. **T-607:** tertutup untuk klip ber-crop (key
  `crop.source_sha256` = sha256 berkas video; `kotak.json` milik sumber lain juga ditolak ingest).
- (d) Determinisme `meta.json` hanya terbukti pada mesin + build ffmpeg yang sama; update ffmpeg lalu ingest
  ulang bisa mengubah identitas (belum terbukti) → konsekuensinya `--restart` / hitung ulang, bukan hasil salah.

---

## CLI (T-104b, D-010)

`python -m rotoscope <subperintah>` (`src/rotoscope/cli.py` + `__main__.py`; tanpa entry point di `pyproject.toml`,
tanpa dependency baru). Modul stage tetap bisa dijalankan sendiri (`python -m rotoscope.segment …`) — itu entry point
internal yang dipanggil cli, bukan cara pakai utama.

| Subperintah | Fungsi |
|---|---|
| `run <video>` | ingest → isolate (T-605; HANYA bila `target/points.json` ada) → segment → depth → stabilize → vectorize → stylize → export (T-203b); flag `--config --style --seg-model --limit --restart-from --yes` (`--yes` juga = izin GPU untuk [1b]); **T-204:** `--preview N [--from K]` (jendela K..K+N-1, tanpa GPU, tanpa [1b]; lihat "`run --preview`" di bawah); **T-607:** `--start S --end E` (trim → folder `<stem>@S-E`); `run` penuh berhenti sesudah [1] dengan **exit 4** bila saran crop belum diputuskan |
| `isolate <video> [--config P] [--yes] [--restart] [--dry-run]` | **T-605:** stage [1b] sendiri (subprocess); `--yes` wajib bila GPU dipakai (hitung baru / basi); `--restart` butuh `--yes`; `--dry-run` = status + adegan + job + perkiraan tanpa GPU / tulis. Lihat "[1b] `isolate.py`" |
| `ingest\|segment\|depth\|stabilize\|vectorize\|stylize\|export <video> [flag stage]` | satu stage; sisa argumen diteruskan apa adanya ke `main(argv)` stage (flag stage tidak diparse ulang di cli). `vectorize` (T-201a, T-201b): flag `--restart --limit` (`--limit N` tetap membaca SEMUA frame untuk ambang per klip, `clip_stats.json`), CPU in-process, `--restart` tanpa `--yes`. `stylize` (T-203a; `--from` T-204): flag `--style PATH --restart --limit [--from K]`, CPU in-process, `--restart` tanpa `--yes` (hanya menghapus `strokes/`). Keduanya (dan `export`) juga bagian urutan `run` sejak T-203b. **T-607:** `--start S --end E` di semua subperintah ini (dan `isolate`) memilih folder `<stem>@S-E` (ingest: juga memotong); `--work-dir` / `--crop-dir` ditentukan cli (exit 2) |
| `download [--config P] [--seg-model 0.8b\|0.4b]` | unduh checkpoint (online, sekali jalan): Sapiens2-seg (default **0.8b**; `--seg-model 0.4b` = fallback), Depth Anything V2 Small, **dan** SAM 2.1 hiera tiny ([1b], T-605; 6 berkas jalur `transformers`) + model card. Subprocess mewarisi environment; **tidak** memaksa `HF_HUB_OFFLINE=1` (run biasa offline) |
| `pick <video> [--frame N] [--config P] [--start S --end E] [--dry-run]` | **T-604:** jendela klik target (BUKAN stage, tidak masuk urutan `run`) → `target/points.json`. `meta.json` belum ada → [1] ingest dulu (CPU, argumen sama dengan `run`); sudah ada → `frames/` tidak disentuh. Lihat "`pick.py`" di bawah |
| `crop <video> [--accept \| --box X Y W H \| --none \| --clear] [--config P]` | **T-607 (D-013):** keputusan kotak crop per SUMBER video (BUKAN stage, CPU) → `<paths.work_dir>/crop/<nama>/kotak.json`; tanpa aksi = status. `--start` / `--end` ditolak (exit 2: kotak berlaku untuk semua folder `<nama>@…`). Lihat "`crop.py`" di bawah |
| `batch <video\|folder> ... [--list BERKAS] [--yes] [--skip-pending] [--config P] [--style P] [--seg-model 0.8b\|0.4b]` | **T-603 (BUKAN stage):** banyak klip → proses `run <video> … --yes` berurutan (satu proses per klip). Tanpa `--yes` = RENCANA: CPU saja, ingest + status crop / target / identitas / stage GPU + estimasi GPU dan disk; tidak memanggil stage GPU dan tidak menghapus apa pun. Dengan `--yes` = JALAN. Flag `--restart-from --preview --from --adopt --qc-only --limit` DILARANG (exit 1). Lihat "`batch.py`" di bawah |

- **Folder kerja per klip:** `<paths.work_dir>/clips/<stem>/`; `stem` = nama video disanitasi dengan fungsi yang sama
  dengan `{source}` di `export.filename` (`export.sanitize_source_name`). Layout `clips/` memisahkan data klip dari
  folder eksperimen di `work/` (`t102c`, `t106`, …) — tanpa daftar nama dicadangkan. `paths.out_dir` tidak berubah.
  **T-607:** klip ber-trim → `<stem>@S-E` (S, E = detik kanonik, lihat [1]; tanpa akhir = `<stem>@S-end`); `export.clip_name` +
  `stage_common.trim_label` = fungsi yang sama untuk folder kerja, nama export, dan folder SVG. Saran / kotak crop per SUMBER di
  `<paths.work_dir>/crop/<stem>/` (dipakai bersama semua folder `<stem>@…`; tidak tersentuh `--restart-from ingest`).
- **`--work-dir DIR` di setiap `main()` stage** (ingest sudah punya; segment, depth, stabilize, export sejak
  T-104b): menang atas `paths.work_dir` config (lewat `overrides`, T-104a — tanpa menulis YAML); `paths.out_dir` tidak
  berubah. cli selalu menambahkannya sendiri; `--work-dir` di baris perintah cli ditolak (exit 2).
- **Proses:** stage GPU ([1b], [2], [2c]) = **subprocess** `[sys.executable, -m, rotoscope.isolate|segment|depth, …]`, stdout/stderr
  diwariskan (pesan VRAM tetap terbaca), baik lewat `run` maupun subperintah sendiri. Stage CPU (ingest, stabilize,
  export) = in-process lewat `main(argv)`; `SystemExit` dari argparse stage ditangkap dan menyebut nama stage. Proses
  induk **tidak meng-import torch** / menyentuh CUDA (diuji di subprocess bersih); cek VRAM tetap tugas stage.
  Induk menerima `KeyboardInterrupt` / exception saat stage GPU jalan → anak di-`terminate` (lalu `kill` bila macet)
  dan ditunggu selesai; pesan "dihentikan; jalankan ulang untuk resume".
- **Pre-flight CPU-only** (`run`; ingest sendiri hanya (a)+(b)) — SEBELUM penghapusan, ingest, atau subprocess apa pun:
  (a) file video ada; (b) folder kerja tidak berisi klip lain (`meta.json` `source_path` ≠ video, dibandingkan dengan
  `Path.resolve` yang sama dengan ingest, bukan string mentah; T-607: juga key `trim` ≠ trim yang diminta); (c) target export `out/<nama>.mp4` tidak milik video
  lain (`<nama>.export.json` `clip.source_path`; fungsi nama export yang sama; dilewati bila `--limit`). Alasan (c):
  pengaman export tidak bisa dilewati `--restart`, jadi tanpa ini klip bernama sama baru gagal di stage terakhir
  setelah ±78 mnt GPU. Pesan bentrok TIDAK menyarankan `--restart`: ganti nama salah satu video (atau ubah
  `export.filename`). Sejak T-203b: (d) `--style PATH` (atau style default) ada dan lolos `config.load_style` (kegagalan di [5]
  baru terlihat setelah ±78 mnt GPU; `render.output_width` divalidasi genap sehingga ukuran output yuv420p aman);
  (e) `export.source` valid (divalidasi saat config dimuat); (f) `ffmpeg` + `ffprobe` ada di PATH; (g) bila
  `export.source = "strokes"` (dan tanpa `--limit`): `out/svg/<nama>/` bukan milik video lain, dan penanda SVG tidak rusak —
  penanda rusak + `--restart-from` yang mencakup export hanya lolos bila setiap `*.svg` di folder identik byte dengan `strokes/`
  (lihat [6]). Semua CPU-only dan gagal = exit 1 tanpa stage / subprocess / penghapusan apa pun.
- **Ingest di `run` = selalu ulang** (opsi A). `meta.json` byte-deterministik → klip yang sama lolos identitas T-108;
  video lain yang menimpa path yang sama menulis ulang `meta.json` → identitas berubah → [2]/[2c] menolak (exit 1).
  (Melewati ingest kalau `meta.json` cocok akan melebarkan batas (c) "Identitas klip": video apa pun di path yang sama
  tidak terdeteksi.) Ingest hanya detik.
- **Trim (T-607, D-013):** hanya lewat `--start S` / `--end E` (detik dari frame pertama hasil ingest, frame n = n / 24 s; TIDAK ada di
  YAML: nilai YAML berlaku untuk semua klip). Indeks: a = ceil(S · 24 − 1e-6), b = ceil(E · 24 − 1e-6), frame a … b − 1 (akhir
  eksklusif), dinomori ulang dari 0. cli meneruskan detik KANONIK (floor(n · 1000 / 24) / 1000) ke ingest, jadi masukan setara (mis.
  `--end 3.8333` dan `--end 3.83333`) memberi folder yang sama; dua indeks berbeda tidak pernah bernama sama. Sebelum [1] ingest
  ber-trim (`run`, `run --preview`, `ingest`, `pick`) cli mencetak pemetaannya, mis. `[1] trim: --end 3.8334 → frame 0–92 (93 frame)
  pada 24 fps → folder klip5@0-3.875`. `--start 0` tanpa `--end` = tanpa trim. Masukan salah (E ≤ S, tanpa frame, S < 0) → exit 2;
  melewati akhir klip → [1] exit 1 dengan panjang klip.
- **Saran crop di `run` (T-607):** bila sesudah [1] ada saran yang BELUM diputuskan (`saran.json` ada, `kotak.json` tidak), `run`
  penuh **berhenti dengan exit 4** (`EXIT_CROP_DECISION`) SEBELUM stage GPU, dengan tiga perintah siap tempel (`crop … --accept` /
  `--none` / `--box X Y W H`) + "lalu jalankan ulang: `run …`". `run --preview`, `ingest`, dan `pick` tidak berhenti (hanya mencetak
  saran). Klip tanpa saran dan `ingest.crop.suggest: false` tidak terdampak.
- **Perintah siap tempel (T-607, keputusan Rio 2026-10-22):** setiap perintah yang dicetak untuk ditempel memakai `stage_common.cli_cmd`
  (video dari `meta.json` + flag trim + `--config <path absolut>` bila config non-default dipakai; `configs/default.yaml` tidak dicetak)
  atau `crop.crop_cmd`. `--config` dicatat oleh `cli.main`, subperintah stage, dan `main()` setiap stage (juga subprocess GPU) lewat
  `stage_common.set_message_config`.
- **Flag `run`:** `--config`, `--style NAMA|PATH` (T-406: nama preset di `configs/styles/` atau path YAML; diteruskan ke stylize DAN export; flag mengalahkan kunci `style` di
  `default.yaml`; satu fungsi `config.resolve_style` untuk stylize / export / run / preview / pre-flight, nama tak dikenal = exit 1 + daftar preset sebelum stage mana pun;
  pre-flight target export dan folder SVG memakai nama style untuk `{style}`; docs/02 "Memilih style"), `--seg-model 0.8b|0.4b` (hanya segment;
  **tidak pernah fallback otomatis**, D-009), `--limit N` (segment, depth, stabilize, vectorize, stylize, export; export →
  `<nama>.limitN.mp4`, tanpa salinan SVG; **tidak** diteruskan ke [1b]: masker selalu seluruh klip, satu baris pesan), `--restart-from`,
  `--yes` (juga diteruskan ke [1b] sebagai izin GPU; stage lain tidak menerimanya).
  **[1b] di `run` (T-605):** setelah ingest; `target/points.json` tidak ada → TANPA subprocess dan tanpa operasi berkas (bila `isolate/`
  lama ada, dihapus in-process dengan pesan); ada → subprocess `rotoscope.isolate --work-dir … [--config] [--restart] [--yes]`. Hasil
  up-to-date → dilewati tanpa GPU (±3–6 s: sidik piksel semua frame); perlu GPU tanpa `--yes` → [1b] exit 1, `run` berhenti dengan pesan. [4] dan [5] **selalu** dijalankan di `run` (up-to-date →
  dilewati dalam ≈ 0,2–1 s; hanya basi yang dihitung ulang), apa pun `export.source`.
  `--adopt` / `--qc-only` hanya di subperintah `segment` / `depth` (`--adopt` hanya keduanya); di `run` → exit 1.
- **`run --preview N [--from K]` (T-204):** jendela frame K..K+N-1 (K default 0) → `out/<nama>.preview_K-<K+N-1>.mp4` (pola nama
  `export.filename` seperti `.limitN`). Hanya di `run`; hasil terpisah dari MP4 utama.
  - **Flag dan kombinasi:** `--preview N` (int ≥ 1), `--from K` (int ≥ 0, hanya bersama `--preview`), K + N ≤ `frame_count`. Terlarang
    (exit 1): `--preview` bersama `--limit`, `--restart-from`, `--adopt`, `--qc-only`; `--from` tanpa `--preview`; K + N melewati klip.
    Salah tipe argumen → exit 2 (argparse). `--config`, `--style` (nama atau path; target preview memakai nama style untuk `{style}`), `--seg-model` (hanya untuk cek kunci model) berlaku.
  - **Tanpa GPU:** segment / depth TIDAK PERNAH dijalankan (tanpa subprocess; proses induk tidak meng-import torch). Dicek CPU-only
    dengan fungsi validasi stage (`Clip.frame_valid`, `_require_same_clip`): manifest seg/ dan depth/ ada, identitas klip cocok dengan
    `meta.json`, kunci model di `seg/manifest.json` = `segment.model` / `--seg-model` (perbandingan string "0.8b" / "0.4b", tanpa
    `resolve_revision`), frame jendela valid. Gagal → exit 1, pesan menyebut stage + perintah (`run <video>` penuh, atau
    `segment|depth <video> --limit K+N`; ganti model: `segment … --restart --yes`, MENJALANKAN GPU).
  - **Pre-flight** (sebelum stage mana pun): video ada, folder kerja tidak berisi klip LAIN, style valid, ffmpeg + ffprobe, K + N ≤
    `frame_count` (dari `meta.json` bila ada; diulang sesudah ingest), cek seg / depth (sebelum ingest bila `meta.json` ada, diulang
    sesudahnya). Pre-flight (c) target export utama dan (e) folder SVG dilewati seperti pada `--limit`.
  - **Urutan dan perilaku per stage:**

    | Stage | Perilaku |
    |---|---|
    | [1] ingest | selalu ulang (±0,8 s); `meta.json` byte-identik |
    | [2]/[2c] | hanya dicek (di atas) |
    | [3] stabilize | TANPA `--limit` (resume penuh): vectorize memvalidasi `stable/` semua frame. Valid ≈ 0,2–0,8 s; belum / basi → dihitung penuh dengan baris estimasi (±0,11 s/frame) |
    | [4] vectorize | `--limit K+N` (rantai 0..K+N-1; hasil = prefiks run penuh); `clip_stats.json` valid → pass 1 dilewati. Sebelum [4], bila ada frame prefiks belum valid: "rantai vectorize M frame belum valid, estimasi ±S s" (0,1 s/frame) |
    | [5] stylize | `--from K --limit N` (+ `--style`); frame valid dilewati, hasil tetap di `strokes/` (resume) |
    | [6] export | `--from K --limit N` → MP4 preview |

  - **Style basi:** `strokes/` dihapus seluruhnya, hanya jendela dihitung, SATU peringatan: `strokes/` DIHAPUS seluruhnya (X frame), hanya
    jendela dihitung; `out/<nama>.mp4` dan `out/svg/<nama>/` tetap versi lama sampai `run <video>` penuh berikutnya. Akibatnya `strokes/`
    separuh terisi: `export` penuh GAGAL KERAS (exit 1, perintah `stylize`), `stylize` / `run` penuh melanjutkan frame yang hilang.
  - **Export preview:** tanpa manifest, tanpa salinan SVG, tanpa pengaman milik-sumber-lain, tanpa audio; tidak menyentuh MP4 utama,
    `.export.json`, `out/svg/`. Verifikasi ffprobe: N frame, ukuran output, fps, h264 / yuv420p, tag warna jalur strokes.
  - **Exit code:** 0 / 1 / 2 / 130 (tanpa 3). Pesan akhir: jendela, jalur MP4, waktu per stage.
  - **Kontrak stage `--from K --limit N`** (stylize dan export): jendela posisi K..K+N-1; `--from` WAJIB bersama `--limit` (exit 1 bila
    tidak), K ≥ 0, K + N ≤ jumlah frame. Frame jendela byte-identik dengan frame yang sama di run penuh. vectorize / stabilize tidak
    punya `--from` (berantai / validasi penuh).
  - **Batas yang diketahui:** estimasi rantai tidak memuat biaya tetap ±1 s (terlalu rendah untuk M kecil); `stable/` basi = stabilize
    penuh (di luar anggaran); pengecekan resume segment / depth tanpa torch untuk `run` biasa = backlog (bukan T-204).
  - **Anggaran "< 30 s" jujur:** terpenuhi untuk skenario hangat — klip test
  (a) K=73 N=10 4,17 s, (b) 5,82 s (N=20: 8,69 s), (c) style basi 6,35 s, (e) K=0 3,72 s; kasus dingin K=225 (rantai vectorize) 28,6 s
  (vectorize 18,4–22,9 s, variasi mesin besar) dan 31,3 s bila `clip_stats.json` + manifest hilang (skenario tambahan); `stable/` basi
  (stabilize penuh) 51,8 s = PENGECUALIAN. Sebelum [4], bila ada frame prefiks belum valid, dicetak "rantai vectorize M frame belum
  valid, estimasi ±S s" (0,1 s/frame).
  Kombinasi terlarang stage ([2]/[2c] `--adopt` + `--restart`/`--limit`/`--qc-only`/`--download`) tetap exit 1.
- **Restart mengikuti graf dependensi** ingest → {segment, depth} → stabilize → vectorize → stylize → export (depth **tidak**
  bergantung pada segment). `run --restart-from X` menjalankan stage di bawah ini dengan `--restart`; stage sebelum X berjalan normal
  (resume), hilir CPU yang basi dihitung ulang otomatis oleh stage-nya (cli tidak menghapusnya):

  | `--restart-from` | stage ber-`--restart` | butuh `--yes` |
  |---|---|---|
  | `ingest` | isolate, segment, depth, stabilize, vectorize, stylize, export (semua) | ya |
  | `isolate` | hanya [1b] (`isolate/`); tanpa `target/points.json` → `isolate/` dihapus, tidak dihitung | ya |
  | `segment` | hanya [2] (`seg/` + `qc_report.json`); `depth/` utuh | ya |
  | `depth` | hanya [2c] (`depth/`) | ya |
  | `stabilize` / `vectorize` / `stylize` / `export` | hanya stage itu | tidak |

  Graf lengkap sejak T-203b: ingest → {segment, depth} → stabilize → vectorize → stylize → export. `vectorize --restart` menghapus
  `contours/` saja, `stylize --restart` menghapus `strokes/` saja; hilir CPU yang basi (mis. [5] setelah [4] dihitung ulang, [6]
  setelah [5]) dihitung ulang otomatis oleh stage-nya. Terukur `--restart-from vectorize` (test_short / test): [4] 9,2 / 21,4 s,
  [5] 16,1 / 39,2 s, [6] 2,4 / 5,4 s; contours, strokes, MP4 dan SVG byte-identik dengan sebelumnya.

- **`--yes`:** wajib untuk SETIAP penghapusan hasil GPU lewat cli — `run --restart-from ingest|isolate|segment|depth` dan
  subperintah `isolate|segment|depth --restart`. cli membuang `--yes` sebelum meneruskan ke stage, KECUALI ke [1b] (T-605): di sana
  `--yes` juga izin memakai GPU (hitung baru / basi). Tanpa `--yes`: cetak apa
  yang akan dihapus (jumlah file, MiB, estimasi waktu GPU: 0.8b 16.5 s/frame, 0.4b 8 s/frame, depth 0.19 s/frame —
  konstanta pesan di `cli.py`, bukan parameter), exit 1, tidak ada yang terhapus. Pesan error stage yang menyuruh
  `--restart` memakai perintah CLI lengkap (`python -m rotoscope segment "<video>" --restart --yes`; `<video>` dari
  `meta.json`). Setelah `run` yang memakai `--restart-from` gagal, pesan menyebut resume = jalankan ulang TANPA
  `--restart-from`/`--yes` (kalau tidak, hasil dihapus lagi).
- **Frame gagal QC:** setelah [2] satu peringatan `N/283 frame gagal QC (indeks: …, maks 10 ditampilkan)`; run lanjut
  (stage [3] memberi bobot, D-010), exit 0. Tanpa parameter YAML baru.
- **Exit code `run`:** 0 sukses (atau semua dilewati) | 1 prasyarat gagal (config, pre-flight, VRAM, identitas, flag
  terlarang, `--limit` < 1) | 2 salah pakai argumen (argparse; juga argparse stage in-process) | 3 OOM di segment/depth |
  4 saran crop belum diputuskan (T-607; hanya `run` penuh, sesudah [1]) | 130 dihentikan (Ctrl+C). Kode stage yang gagal dikembalikan apa adanya; pesan menyebut stage + cara resume.
- **Batas yang diketahui (backlog):** `segment` / `depth` memakai ±21–28 s per `run` walau semua frame dilewati (impor torch +
  pengecekan resume di subprocess).

---

## Kontrak per modul

Resolusi kerja = resolusi `frames/` (`meta.json` `working_width` × `working_height`; klip uji
480×854). Estimasi disk = klip 15 s, 360 frame, 480×854 (409 920 px/frame); tanda *est.* = harus
diukur saat implementasi.

### [1] `ingest.py` — `DONE` (T-101; crop + trim T-607, D-013)
- **In:** path video, `target_fps` (default 24), `working_width` (default 720); sejak T-607: config section `ingest`
  (`crop.suggest`, `crop.detect.*` = 5 parameter R2 beku, `crop.preview.hint_static_frac`; docs/02), `--start` / `--end` (trim, lihat
  "CLI"), dan `<paths.work_dir>/crop/<nama>/kotak.json` bila ada (keputusan dari alat `crop`).
- **Out:** `frames/frame_%05d.png`, plus `meta.json` (fps asli, durasi, jumlah frame, resolusi kerja; + key `trim` / `crop` HANYA bila
  dipakai) + per SUMBER `<paths.work_dir>/crop/<nama>/deteksi.json` (selalu, juga "tanpa pita") + `saran.json` + `pratinjau/` (hanya bila
  pita terdeteksi dan belum ada `kotak.json`). Format berkas crop: "`crop.py`".
- **Lib:** ffmpeg via subprocess (`fps` [+ `trim`] [+ `crop`] + `scale`), ffprobe untuk metadata
- Resize proporsional. Jangan upscale kalau sumber lebih kecil.
- **Ingest TIDAK pernah memotong tanpa `kotak.json`** (saran + konfirmasi; mode `auto` dihapus sesudah K2 gagal, docs/04). Tanpa
  `kotak.json` dan `crop.suggest` true → deteksi R2 beku pada RENTANG PENUH SUMBER (pipa rawvideo `fps=24,scale=W:-2`, tanpa PNG; trim
  tidak memengaruhi saran) → bila pita ada: saran + pratinjau + pesan (kotak kerja → sumber, path pratinjau, petunjuk + perintah
  alternatif, perintah `crop --accept` / `--box X Y W H` / `--none`), dicetak SESUDAH dump `meta.json`. **D1:** deteksi dilewati bila
  `deteksi.json` memuat kunci yang sama {identitas sumber, `target_fps`, `working_width`, aturan + parameter + sidik R2, versi ffmpeg};
  dengan `kotak.json` valid tidak ada deteksi. Parameter R2 ≠ beku → PERINGATAN "aturan belum divalidasi" (saran itu ditolak `crop
  --accept`). Deteksi gagal (ffmpeg pipa error) → PERINGATAN, ingest tetap jalan tanpa saran.
- **Dengan `kotak.json`:** identitas sumber dicek (sha256 byte + ukuran + dimensi; path hanya info) → beda → **exit 1** + perintah `crop …
  --clear`. `mode: none` = tanpa potong, tanpa key `crop` (byte-identik dengan klip tanpa kotak). `saran` / `manual` → vf
  `fps=24[,trim=start_frame=a:end_frame=b],crop=w:h:x:y:exact=1,scale=…` (crop di koordinat SUMBER, SEBELUM scale = P-src; kotak
  dibulatkan KE LUAR + genap: konten tidak pernah terpotong) → key `meta.json` `crop` = `{mode, box_source, rounding: "outward-even",
  source_sha256}` + mode saran `{rule, rule_version, rule_fingerprint, detect_frame_size, box_work, band_px, overlay_px}` / mode manual
  `box_requested`. Lebar kerja = min(lebar kotak, `working_width`), tanpa upscale.
- **Key `trim`:** `{start_s, end_s, start_frame, end_frame}` (detik kanonik = nama folder; `end_s` / `end_frame` null = sampai akhir).
  Tanpa trim / kotak → perintah ffmpeg dan `meta.json` sama persis dengan sebelum T-607.
- **CLI:** `python -m rotoscope ingest <video> [--start S --end E] [--config P] [--target-fps N] [--working-width N]` (folder kerja =
  `<paths.work_dir>/clips/<nama>[@S-E]/`; pre-flight (a)+(b) di bagian "CLI"). Di `run` ingest **selalu** dijalankan ulang
  (`meta.json` byte-deterministik, jadi klip yang sama lolos identitas T-108). Entry point modul: `python -m rotoscope.ingest <video>
  [--work-dir DIR] [--config P] [--crop-dir DIR] [--start S] [--end E]` (tanpa `--crop-dir` = tanpa saran / kotak; cli selalu
  memberinya). Exit code 0 / 1 (saran tidak membuat ingest gagal; yang berhenti dengan exit 4 hanya `run` penuh).
- **Terukur (Tahap 3a, proses CLI penuh dengan deteksi pertama):** 2,2–21,0 s (klip7 664 frame); D1 sesudahnya ingest saja (klip11
  0,5 s); ingest ber-crop Klip2 / klip5 0,6–0,9 s.
- **Batas yang diketahui:** satu kotak per SUMBER (tata letak yang berubah di tengah klip tidak ditangani); letterbox satu sisi dan crop
  per adegan di luar lingkup; R2 gagal pada kotak teks sebagian / noise kompresi di konten kerumunan (klip11: 74 baris tersisa →
  petunjuk menandainya) dan bisa memotong berlebih pada rentang pendek berobjek gelap di kedua tepi (klip6, K3) — karena itu konfirmasi
  wajib; petunjuk hanya di sisi yang dipotong saran (strip kanan satu sisi 4 px klip11 tidak ditandai).

### [1b] `isolate.py` — isolasi target, SAM 2.1 hiera tiny (GPU, proses sendiri, OPT-IN; T-605, D-011)
- **Peran:** masker target per frame dari klik `pick` (T-604). Hanya MENGHASILKAN masker; pemakainya = **[3] langkah 1b** (T-606,
  D-012, mode M2-H: seg asli dibatasi masker sebelum kernel temporal). [2] / [2c] tidak membacanya (M1 — frame dibersihkan + seg
  ulang — ditolak, D-012).
- **In:** `meta.json`, `frames/*.png`, `target/points.json` (`rotoscope-target-points/1`, divalidasi `pick.load_points` +
  `check_doc_matches_clip`; setiap frame klik harus `ok` menurut sidik piksel, id target ≤ 255); config `isolate`,
  `stabilize.temporal.cut_diff`.
- **Opt-in:** tanpa `target/points.json` → dilewati. `run` tidak membuat subprocess dan tidak menulis apa pun; bila `isolate/`
  (atau sisa `isolate.partial/` / `isolate.old/`) masih ada, folder itu DIHAPUS dengan pesan (keputusan Rio: T-606 tidak boleh
  memakai masker usang).
- **Model:** `isolate.model_id` `facebook/sam2.1-hiera-tiny` (kode + bobot Apache-2.0; wajib diawali `facebook/sam2.1-hiera-`),
  `transformers` 5.17.0 `Sam2VideoModel` + `Sam2VideoProcessor` (tanpa dependency baru). Revision di-pin (`isolate.revision`,
  commit 40-hex; `HF_HUB_OFFLINE=1`; tidak di cache → berhenti + perintah `download`); `license` model card (`README.md`) =
  `apache-2.0` dicek saat load. `isolate.precision` fp32 (fp16 boleh; bf16 ditolak, P-005); satu precision per klip.
- **Cek sebelum load:** RAM fisik bebas ≥ `isolate.ram_min_free_mib` (4096; `GlobalMemoryStatusEx`), VRAM bebas ≥
  `isolate.vram_min_free_mib` (2000). Kurang → berhenti dengan pesan (exit 1), model tidak dimuat.
- **Adegan:** `stabilize.frame_thumb` + `cut_scores` + `detect_cuts(stabilize.temporal.cut_diff)` + `shot_bounds` — fungsi + ambang
  SAMA dengan [3] (dipakai walau `stabilize.temporal.enabled` false; 0 = satu adegan). Adegan tanpa klik → `tanpa_prompt`.
  Cut positif palsu di adegan berklik = pelacakan berhenti (frame sesudahnya tanpa target; tambah klik di awal adegan itu);
  cut terlewat = SAM melacak menembus cut. Rencana mencetak saran klik bila adegan tanpa klik tepat sesudah adegan berklik.
- **Propagasi** (satu sesi offline per (adegan berklik, target); frame adegan diindeks lokal 0..L−1 dan dibaca dari disk saat
  dibutuhkan (`processed_frames` malas; praproses = `processor(images=…)`, resize bilinear 1024 × 1024 tanpa menjaga rasio +
  normalisasi ImageNet); frame + state di CPU, inferensi di GPU):
  1. tiap frame klik target itu di adegan itu, urut naik: `add_inputs_to_inference_session` (titik + label + kotak, koordinat frame
     kerja apa adanya) lalu LANGSUNG `model(frame_idx=k)` → frame kondisi;
  2. `propagate_in_video_iterator` maju dari klik paling awal sampai akhir adegan (frame kondisi lain dipakai dari cache = koreksi);
  3. mundur dari klik paling awal − 1 sampai awal adegan.

  Tiap frame dihitung tepat sekali. Logit (`post_process_masks`, ukuran frame kerja) > `isolate.mask_threshold` (float32) = target.
- **Out:**

  | path | isi |
  |---|---|
  | `isolate/labels/frame_%05d.png` | peta label uint8 1 kanal, ukuran frame kerja: 0 = bukan target, k = id target. HANYA frame di adegan berklik. Tumpang tindih → logit terbesar, seri → id terkecil |
  | `isolate/frames.jsonl` | satu baris `event: frame` per frame klip (urut indeks): `shot`, `status` (`target` ≥ 1 piksel target / `kosong` adegan berklik tanpa piksel target / `tanpa_prompt` adegan tanpa klik, tanpa PNG), `pixel_sha256`, `targets` {id: `kind` (`klik` / `maju` / `mundur`), `area_px`, `score_logit`} |
  | `isolate/manifest.json` | `format` `rotoscope-isolate/1`, `clip`, `frame_size`, `frame_count`, `key` (kontrak `T-605`, sha256 byte `points.json`, sidik piksel gabungan SEMUA frame, model id + revision, precision, `mask_threshold` + `cut_diff`, versi torch / transformers) + `key_sha256`, `model` (+ lisensi), `processor`, `mode`, `scenes` (cut + adegan), `jobs` (adegan, target, frame klik, maju / mundur, jumlah + waktu inferensi), `targets`, `frames` (jumlah per status + rentang), `run` (waktu, VRAM bebas / total / maks reserved + allocated, RAM bebas + puncak proses, frame NaN), `created_utc`. Ditulis TERAKHIR |

- **Basi + resume (per KLIP, bukan per frame):** up-to-date = manifest ada + `key` + `clip` + `frame_count` sama + semua PNG wajib
  terbaca (uint8, ukuran frame) + tanpa PNG lebih + `frames.jsonl` lengkap → dilewati tanpa torch / GPU. Selain itu basi (alasan
  per field dicetak). `vram_min_free_mib` / `ram_min_free_mib` bukan kunci. **`--yes` wajib setiap kali GPU dipakai** (hitung baru
  maupun basi): tanpa `--yes` → exit 1 + pesan (perintah `--yes` + `--dry-run`), tidak ada yang dihitung / dihapus; dengan `--yes` →
  seluruh klip dihitung ulang tanpa `--restart`. `frames: []` → hasil tanpa GPU (semua `tanpa_prompt`, tanpa `--yes`).
- **Tulis atomik per klip:** semua berkas ke `isolate.partial/`; selesai → `isolate/` → `isolate.old/`, `isolate.partial/` →
  `isolate/`, `isolate.old/` dihapus. Terputus (Ctrl+C / OOM / galat) → `isolate/` lama utuh; sisa dihapus di awal run berikutnya.
- **CLI:** `python -m rotoscope isolate <video> [--config P] [--yes] [--restart] [--dry-run]` (`--restart` tanpa `--yes` → exit 1);
  modul `python -m rotoscope.isolate [--config] [--work-dir] [--yes] [--restart] [--dry-run] [--download]`; tanpa `--limit`. Exit 0
  (dihitung / up-to-date / dilewati / dry-run) | 1 (prasyarat, termasuk butuh `--yes`, titik basi, RAM / VRAM kurang) | 3 (OOM) | 130.
- **Terukur (T-605, GTX 1650 Ti, fp32):** reserved maks 866 MiB (41 frame) / 892 MiB (92 frame), p50 0,68–0,74 s per frame, RAM
  proses 3,5–3,8 GiB, puncak `nvidia-smi` ≤ 2450 MiB; up-to-date ±5 s tanpa GPU; dua run berturut = PNG byte-identik.
- **Batas yang diketahui:**
  - **Kerumunan / video kualitas rendah / zoom (klip5, T-605): masker salah di SEMUA frame** (penilaian Rio): target hilang di
    f3–f62, lalu zoom di f63–f77 (orang memenuhi layar; masker seluruh gambar di sana benar, kotak seluruh layar juga benar). Klik
    di frame zoom diduga mengacaukan memori frame awal (mode offline memakai semua frame kondisi, termasuk yang sesudahnya; belum
    terbukti). Klip seperti ini harus diolah terpisah sebelum [1b] (backlog "pengolahan khusus klip5").
  - **Klik kanan (bukan target) membantu:** klip3 dengan 4 klik kanan (titik T-602) dinilai Rio lebih baik daripada 16 klik kiri saja.
  - **Masker memotong per piksel:** lubang kecil (dekat tangan, celah lengan–badan) bisa menjadi garis palsu bila masker dipakai
    langsung per piksel — karena itu [3] memakai algoritma H (pilih komponen + isi lubang + lebar 4 px; T-606, D-012).
  - Frame = frame kerja [1] (sejak T-607 sudah terpotong bila kotak crop diterima; pick ulang bila ukuran berubah); frame dibaca malas lewat atribut dalam `transformers` (`processed_frames`; versi di-pin
    5.17.0, test kontrak bobot asli di CPU `ROTOSCOPE_SAM_CPU_TESTS=1`); sesi per target menghitung ulang fitur gambar per target;
    VRAM ±2 MiB / frame / target dan RAM ±4,3 MiB / frame / target tumbuh selama satu sesi.

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
  - Unduhan checkpoint = **langkah terpisah sekali jalan** (online), **bukan** bagian dari run:
    `python -m rotoscope download [--seg-model 0.4b]` (subperintah `download` T-104b: seg + depth; modul:
    `python -m rotoscope.segment --download`). Revision `null` → unduh `main` lalu cetak hash-nya untuk di-pin.
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

**CLI** (T-104b): `python -m rotoscope segment <video> [--config PATH] [--seg-model 0.8b|0.4b] [--restart --yes]
[--limit N] [--qc-only] [--adopt]` (unduh: `python -m rotoscope download`); di dalam `run` stage ini = subprocess
`python -m rotoscope.segment --work-dir <folder klip> …`. Modul menerima `--work-dir DIR` (menang atas
`paths.work_dir`; `paths.out_dir` tidak berubah) dan `--download`; `--yes` hanya ada di cli (membuang sebelum
meneruskan; `--restart` tanpa `--yes` → exit 1, tidak ada yang dihapus). `--limit N` =
pastikan N frame pertama valid (QC dilewati kecuali semua frame klip valid); `--qc-only` tanpa GPU/torch,
butuh semua frame valid. `--adopt` = catat identitas klip ke manifest lama tanpa inferensi (terukur klip
uji 283 frame: 8.45 s); tidak bisa digabung dengan `--restart` / `--limit` / `--qc-only` / `--download`.

**Exit code** (diteruskan apa adanya oleh `cli.py`, T-104b; tabel `run`: bagian "CLI"):

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
| `big_blobs` | n_t = jumlah komponen foreground 8-arah dengan luas > `qc.blob_min` × area frame (selalu dilaporkan). Kondisi gagal menurut `qc.big_blobs_rule` (T-305d): **`transient`** (default) — n_t **naik sesaat**, definisi di bawah; `legacy` — n_t > `qc.max_big_blobs` (perilaku sebelum T-305d, asumsi satu subjek); `report` — tidak pernah gagal | 0.05 / `transient`, `qc.transient_min_run` 4 (`legacy`: `max_big_blobs` 1) |
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

**Definisi `big_blobs` `transient`** (T-305d; N = `qc.transient_min_run`, int ≥ 2): deret n seluruh klip dipecah
menjadi **run** (frame berurutan dengan n sama); run **panjang** = panjang ≥ N. Frame di run pendek gagal bila
n_t > max(n run panjang terdekat **sebelum**, n run panjang terdekat **sesudah**); hanya satu sisi ada → sisi itu;
tidak ada run panjang sama sekali → lolos. **Turun sesaat tidak menggagalkan** (dua orang bersentuhan = sah).
QC dihitung per klip setelah semua frame selesai, jadi melihat frame sesudah t sah.
- Hasil (docs/04 "Keputusan akhir T-305d"): klip3 (tiga orang) 92 → 1 frame gagal; `test` / `test_short` / Klip2
  tetap 0.
- **Batas:** sensitivitas multi-orang terukur 80% (12 / 15 kasus sintetis "tubuh terbelah" di klip3; gabungan
  32 / 35 = 91,4%); blob palsu multi-orang tidak teruji; lonjakan ≥ N frame tidak tertangkap (dianggap perubahan
  jumlah orang yang sah); subjek kecil rapuh — subjek < `blob_min` membuat n = 0, sehingga lonjakan dari n = 0
  sepanjang ≥ N frame lolos dan < N frame gagal (pengaman "abaikan lonjakan dari n = 0" ditunda ke T-305e).

`qc_report.json` per frame: semua metrik di atas + `fail_reasons` (list nama metrik yang gagal). Tingkat
laporan: **`qc_version`** (2 sejak T-305d; laporan tanpa field = sebelum T-305d, `big_blobs` = aturan `legacy`),
`qc` (nilai parameter), `qc_hash`, `definitions` (termasuk `big_blobs` = definisi aturan yang dipakai), `summary`.
Mengubah `qc.*` tidak membuat `seg/` / `depth/` basi (manifest [2] / [2c] tidak memuat `qc`): run [2] dengan semua
frame valid hanya menghitung ulang `qc_report.json`. [3] basi hanya bila **himpunan** frame gagal berubah
(`temporal.qc_fail_frames` di manifest [3]); alasan gagal / `qc_version` tidak dibaca [3].
Frame gagal QC **tidak** diganti di [2]; [3] memberinya bobot temporal lebih kecil (D-010, Q3).

### [2b] `fallback_pose.py` — SKIP (2026-10-27, docs/04 "Revisi D-010 Q3 + D-002")
Foreground Sapiens2 = 0 frame gagal QC di klip uji. Peran fallback diganti: frame gagal QC diisi dari
frame tetangga lewat temporal berbobot di [3]. Dibuka lagi (T-501/T-502) kalau klip nyata gagal QC
dan temporal fill tidak cukup. D-002 (pose hanya fallback, bukan primary) tetap berlaku.
**SKIP (2026-10-27):** T-501 / T-502 SKIP; fallback pose (juga cadangan D-002) tidak dibangun. Syarat buka ulang sekarang: klip baru
(bukan klip5) dengan frame gerak cepat yang rusak terlihat, dan bobot `qc_fail_weight` tidak cukup (docs/04 "Revisi D-010 Q3 + D-002").

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
- **CLI** (T-104b): `python -m rotoscope depth <video> [--config PATH] [--restart --yes] [--limit N] [--adopt]`
  (unduh: `python -m rotoscope download`); di dalam `run` = subprocess `python -m rotoscope.depth --work-dir
  <folder klip> …`. Modul menerima `--work-dir DIR` (menang atas `paths.work_dir`) dan `--download`; `--yes` hanya
  ada di cli (`--restart` tanpa `--yes` → exit 1, tidak ada yang dihapus). Exit code sama dengan [2] (0 / 1 / 3).
  `--adopt`: lihat [2] (0.70 s pada klip uji).

### [3] `stabilize.py` — stage tersulit, alokasikan waktu paling banyak (CPU)
- **In:** `seg/probs/`, `seg/classmap/` (tie-break seri, T-106), `seg/manifest.json`, `depth/`,
  `depth/manifest.json`, `depth/frames.jsonl` (status `finite`), `frames/` (deteksi cut T-302 — dibaca hanya bila temporal aktif,
  R > 0, `cut_diff` > 0), `qc_report.json` (bobot temporal T-302 — dibaca hanya bila temporal aktif dan R > 0;
  tidak ada / frame tanpa baris → berhenti + perintah `segment`); config `groups`, `stabilize`, `target_mask` (T-606);
  **T-606 (D-012), hanya bila `target/points.json` ada:** `target/points.json` (byte, untuk kunci), `isolate/manifest.json`,
  `isolate/frames.jsonl` (status per frame), `isolate/labels/*.png`, dan sidik piksel SEMUA `frames/` (kunci [1b])
- **Langkah:**
  1. Probabilitas kelas → **probabilitas grup** (jumlah per grup / 255, float32). Definisi grup:
     `groups:` di `configs/default.yaml` (lihat 02). Tanpa temporal, argmax dihitung dari jumlah uint8
     per grup (integer, eksak — sama dengan argmax float32). `classes` di `seg/manifest.json` wajib sama
     persis dengan `sapiens2_classes.json` (urutan kelas menentukan pemetaan ke grup).
  1b. **Masker target (T-606 `DONE` 2026-10-19, D-012; `target_mask.py`, dipanggil di `FrameCache.raw(k)` tepat sesudah `group_sums`, SEBELUM
     kernel temporal — setiap jendela yang memuat frame k memakai versi bermasker, jadi masker ikut dihaluskan).**
     - **Berlaku** hanya bila `target/points.json` + `isolate/` ada, `target_mask.enabled`, dan `isolate.existing_state` = ok untuk
       kunci SEKARANG (`isolate.build_key`: byte titik, sidik piksel semua frame, model, precision, `mask_threshold`, `cut_diff`,
       versi lib; PNG + `frames.jsonl` lengkap) — fungsi [1b] yang sama. Biaya: sidik piksel semua frame (±4,7 s di klip3 182
       frame 720 × 1280) di setiap run [3] klip bertitik. Tanpa titik → tidak ada yang dibaca / dicek (byte-identik). Ada titik
       tetapi tidak berlaku (titik / frame berubah, `isolate/` hilang / basi / PNG hilang) → **[3] TANPA masker + satu baris
       `PERINGATAN`** (alasan + `python -m rotoscope isolate <video> --yes`); lewat `run`, [1b] basi tanpa `--yes` sudah berhenti
       sebelum [3]. `enabled: false` → tanpa masker + satu baris log.
     - **Algoritma H** per frame berstatus `target` / `kosong`: F = foreground seg MENTAH (argmax jumlah grup ≠ 0, seri → grup
       classmap); M = label [1b] ≠ 0 (semua target); M_f = M + lubang (komponen 4-arah ~M yang tidak menyentuh tepi gambar;
       `hole_fill_max_px` 0 = semua); M_d = M_f dilebarkan `dilate_px` (elips (2d + 1)²); komponen 8-arah C dari F dengan
       r = |C ∩ M_f| / |C|: r ≥ `r_keep` → utuh, r ≤ `r_drop` → dibuang, di antaranya → C ∩ M_d dipecah jadi potongan 8-arah,
       potongan dengan |P ∩ M_f| < `piece_min_px` dibuang. Piksel F yang tidak dipertahankan → jumlah grup (255, 0, …, 0) + grup
       seri 0 (= probabilitas kelas one-hot Background). Piksel di luar F tidak diubah.
     - **Status:** `kosong` (adegan berklik, target tidak ditemukan) → M kosong → semua dibuang (tidak menggambar apa pun; dengan
       α 0,7 frame tengah tetap dominan — batas: frame `kosong` yang juga gagal QC berbobot 0,1 bisa terisi tetangga);
       `tanpa_prompt` → tidak disentuh (`no_prompt: keep`, default) atau F dibuang (`clear`).
     - **QC tidak dihitung ulang:** bobot q tetap dari `qc_report.json` [2] (classmap tanpa masker). Cut tetap dari `frames/` asli;
       adegan [1b] = cut [3] (fungsi + ambang sama), jadi jendela kernel tidak mencampur frame bermasker dan tanpa masker.
     - **Efek hilir:** normalisasi depth memakai foreground bermasker (median target saja → geser konstan per frame); ambang [4]
       `clip_stats` (persentil foreground SEMUA frame) ikut bergeser — klip3 T_high +1,0%, garis oklusi adegan tanpa klik −1,3%
       (diterima Rio).
     - **Default** (keputusan Rio T-606, klip3): d 4, r_keep 0,9, r_drop 0,1, piece_min 30, semua lubang, `no_prompt` keep. Terukur
       klip3 f85–f125: siluet 3,54 → 1,00 per frame, strok di luar target 35,3 → 0, garis baru di dalam badan 0, penilaian Rio 4
       (lolos semua kriteria; d 8 / 12 nilai 3 / 2, M1 nilai 3). Produksi klip3 penuh (`run`, CPU): f85–f125 byte-identik dengan
       prototipe d 4; dinilai Rio sama dengan d 4, tanpa masalah baru, nilai 4 (docs/04 "Keputusan akhir T-606").
  2. **Temporal** (T-302, kalau `stabilize.temporal.enabled`; **tanpa optical flow: DITOLAK di T-303 berdasarkan data**, docs/04 "Hasil T-303 (ditolak)"):
     **kernel eksponensial simetris TERPOTONG** (bukan EMA IIR dua arah) pada jumlah grup:
     - frame t: `p_t = Σ_k w_k · S_(t+k) / Σ_k w_k`, `w_k = ρ^|k| · q_(t+k)`, k = −R..+R, ρ = (1 − α) / (1 + α),
       α = `mask_ema_alpha` (≈ bobot frame tengah; α = 1 → R = 0 → persis jalur spasial T-106, byte-identik);
       `q` = 1, atau `qc_fail_weight` bila frame itu gagal QC (`fail_reasons` tidak kosong di `qc_report.json`);
     - **R** = bilangan bulat terkecil dengan `2ρ^(R+1) / (1 + ρ) < TAIL_MASS` (0,01), maks `R_MAX` (8): α 0,85 / 0,7 → 2,
       0,55 → 4, 0,4 → 5, 0,3 → 7. Konstanta modul `stabilize.py`, bukan parameter YAML;
     - jendela dipotong di tepi klip (bobot dinormalisasi ulang atas frame yang ada, tanpa frame virtual) dan di **cut**;
       jumlah bobot < `WEIGHT_SUM_MIN` → hanya frame tengah. Akumulasi float32, urutan naik k = −R..+R (deterministik);
       frame t hanya bergantung pada input mentah t−R..t+R (tanpa rantai) → resume per frame dan `--limit` sah;
     - **cut**: selisih absolut rata-rata abu-abu (thumbnail `CUT_THUMB_WIDTH` = 48 px lebar, INTER_AREA) antara frame t−1 dan t,
       0–1; `> stabilize.temporal.cut_diff` (0,06; 0 = mati; sebelum 2026-10-10 0,08) → jendela tidak melintasi t−1 | t. Daftar cut tercatat di manifest
       (`temporal.cut_frames`) dan di log. Maks terukur `test` 0,038 / `test_short` 0,030 / Klip2 0,0035 (0 cut); klip3: 7 cut berlabel (skor 0,072–0,090) + 1 FP diketahui (f49, 0,065);
     - **`boil_preserve` b**: `p = (1 − b) · p_halus + b · S_t` sebelum argmax (0 = stabilisasi penuh, 1 = tanpa stabilisasi);
     - **`qc_fail_weight`** q SEMENTARA 0,1: frame gagal ikut jendela tetangga dengan bobot kecil, dan dirinya sendiri diisi tetangga.
       **Hubungan dengan ρ:** frame gagal hanya diisi tetangga bila q < ρ (bobot tetangga langsung; ρ = 0,176 pada α 0,7); q 0,25 > ρ
       hanya memulihkan frame gagal TUNGGAL. Terukur sintetis (α 0,7, R = 2, IoU foreground vs frame asli): q 0,1 memulihkan 1 frame
       (0,97 / 0,90), 2 berurutan (0,963) dan tepi klip (0,957 / 0,942); 3 berurutan 0,80, 5 berurutan 0,67 — **batas yang diketahui:
       tidak lebih dari 2 berurutan pada q berapa pun dengan R = 2** (3+ berurutan tidak pulih penuh). Biaya positif palsu (frame sehat
       yang ditandai gagal) kecil: frame itu tetap berbobot ρ-tetangga. Kalibrasi nyata menunggu klip kedua;
     - **Optical flow: DITOLAK (T-303, 2026-10-06).** Parameter `optical_flow_blend`, peringatan, dan field manifest
       `temporal.optical_flow` dihapus (key lama di YAML → "tidak dikenal"). Alasan: DIS / Farnebäck tidak mengalahkan α adaptif
       murah pada derau statis dan menurunkan pop garis akhir hanya −8% / −1%, dengan biaya 2–3× waktu [3]; data di docs/04.
       `mask_ema_alpha` 0,7 tetap (kalibrasi ulang bukan lagi bagian T-303).
  3. argmax → peta grup. **Seri eksak** → grup dari `seg/classmap` (argmax logits tanpa pembulatan)
     kalau grup itu ikut seri; selain itu id terkecil. Jumlah piksel seri dicatat per frame (klip uji
     5–43 px/frame). Setelah temporal aktif (T-302) seri eksak jarang (argmax float32), tapi aturan tetap berlaku.
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
  5b. **Isi lubang grup face (VF, T-305f; `stabilize.hole_fill.enabled`, default `true`):** komponen background **4-arah**
     (konektivitas lubang [4]) yang tidak menyentuh tepi frame (bbox x = 0, y = 0, x + w ≥ W, y + h ≥ H dilewati) dan yang
     SELURUH cincin 1 px-nya (**8-arah**: dilasi 3 × 3 dikurangi komponen, dibaca dari peta sebelum diisi) = grup `face` → diisi
     `face`. Tanpa batas ukuran. Grup pengisi = konstanta kode `HOLE_FILL_GROUP` "face" (bukan parameter); `enabled: true` tanpa
     grup `face` di `groups:` → ConfigError. **Tempat (PB, keputusan Rio):** di `run_stabilize` sesudah langkah 6 dan sebelum
     `write_groups` — peta akhir (sesudah kernel + pulau + mode), HANYA `stable/groups` yang berubah; normalisasi kedalaman memakai
     peta SEBELUM VF (`depth_smooth` tidak berubah). Lubang yang cincinnya menyentuh grup lain (mis. baju) TIDAK diisi (diterima,
     docs/04 "Keputusan T-305f setelah Tahap 1"). Urutan dengan T-606: masker (`raw(k)`, sebelum kernel) → kernel → pulau → mode
     → VF; frame `kosong` tetap kosong. Per frame tanpa jendela → `--limit` / `--preview` prefiks = run penuh. Ditolak: sebelum
     kernel (P-pre: klip tanpa lubang ikut berubah, menutup lebih sedikit, ±13× lebih mahal) dan VF ikut statistik kedalaman (PA).
     Biaya ±3,4 ms / frame (720 × 1280; ±1% [3]). Aturan = prototipe `scripts/t305_1b.py::vf_fill` (oracle `tests/hole_fill_oracle.py`).
  6. **Kedalaman:** normalisasi **per frame** (disparity DA hanya benar sampai skala + offset per
     frame; kedalaman mentah tidak boleh langsung di-EMA), lalu temporal (kalau `temporal.enabled` **dan**
     `stabilize.depth.temporal`; default dan tetap MATI: flow ditolak di T-303). `stabilize.depth.normalize` (T-302, diukur pada kedua klip uji):
     - **`log_median_iqr` (A, T-106, kompatibilitas):** `(log max(d, log_eps) − median) / max(IQR, iqr_min)`. Median + IQR
       (p75 − p25) dari foreground = peta grup bersih ≠ 0 frame itu; foreground kosong → seluruh frame (`region: "frame"` di
       log). IQR < `iqr_min` → pembagi di-clamp (`iqr_clamped` di log). Pembagi IQR per frame membuat gradien antar frame tidak
       konsisten (CV p95 |grad| 0,32 / 0,33; Done-when kaki 4/5) dan melemahkan tepi kaki di frame 73–83;
     - **`log_median` (B, T-302; DEFAULT):** `log max(d, log_eps) − median` (tanpa pembagi; `iqr_min` tidak
       dipakai). CV p95 |grad| 0,157 / 0,270, Done-when kaki 5/5. Ambang T_high / T_low di [4] = persentil per klip, jadi skala
       konstan tidak berpengaruh;
     - **ditolak:** affine linear `(d − median) / IQR` (CV tidak membaik; 12,3% titik strok di luar Lower_Clothing vs 5,5%);
       IQR tetap per klip (C = B ÷ konstanta, metrik identik; menambah pass 1 dan state per klip).
     - **EMA kedalaman:** kernel yang sama (α, R, bobot QC, potongan cut) pada kedalaman ternormalisasi float32; statistik foreground
       tiap frame di jendela memakai peta grup akhir frame itu → jangkauan input = R (grup) + R (kedalaman). Ukuran (prototipe):
       strok oklusi 248 → 691 (α 0,7), Done-when kaki tidak monoton → default mati.
     - **Background** memakai transformasi yang sama (nilai kontinu + finite): NaN merusak blur/Sobel di
       [4], nilai konstan membuat lompatan palsu di siluet. Range `log_eps` / `iqr_min` (02) menjamin hasil
       muat float16; tidak finite → berhenti dengan error.
     - Frame `finite: false` di `depth/frames.jsonl` disalin ke `stable/frames.jsonl` (`depth_finite`),
       tanpa perlakuan khusus selama temporal mati.
- Default sejak T-302 DONE: `stabilize.temporal.enabled: true` (α 0,7, R = 2, b 0,3, `cut_diff` 0,06 (0,08 sampai 2026-10-09), `qc_fail_weight` 0,1, normalisasi
  `log_median`, `depth.temporal: false`); langkah 2 tanpa optical flow (ditolak, T-303). `enabled: false` = hanya langkah 1, 3–6 (spasial, jalur T-106).
  Temporal butuh `frames/` + `qc_report.json` (klip yang di-ingest + di-segment lewat `run` memilikinya).
- ⚠️ Jangan over-smooth. Sedikit boil = hand-drawn feel (`boil_preserve`), bukan nol (P-001).

| path | isi | dtype | resolusi | disk / klip |
|---|---|---|---|---|
| `stable/groups/frame_%05d.png` | id grup: 0 = background, 1..G = urutan `groups:` di YAML | uint8 | kerja | ±2.4 MB (±6.6 KB/frame, terukur T-106) |
| `stable/depth_smooth/frame_%05d.npy` | kedalaman ternormalisasi per frame (+ temporal) | float16 | kerja | 295 MB (820 KB/frame) |
| `stable/manifest.json` | **`temporal`** (T-302: `{enabled: false}`, atau `enabled`, `radius`, `depth_radius`, `cut_frames`, `qc_fail_frames`), section `stabilize` + `stabilize_hash`, `groups` + `groups_hash`, referensi `seg/manifest.json` (model, model id, revision, precision, processor, `num_labels`, frame_size, classes, `created_utc`) + `depth/manifest.json` (model id, revision, lisensi, precision, processor, ukuran input, output, frame_size, `created_utc`), frame_size, `clip` (identitas klip, T-108); T-606: `isolate` (kunci + sidik label [1b], adegan, status, `target_mask`) HANYA bila masker dipakai | – | – | kecil |
| `stable/frames.jsonl` | log per frame: waktu per langkah, piksel seri, piksel berubah di filter pulau / mode, luas foreground, `depth_finite`, statistik normalisasi (region, median + IQR log, clamp) + min / median / maks `depth_smooth`; T-606: `isolate` (status, jumlah komponen utuh / potong / buang, potongan dibuang, piksel F / dibuang) HANYA di frame yang dimasker; T-305f: `hole_fill_regions` + `hole_fill_px` (HANYA bila `hole_fill.enabled`; `fg_px` = peta yang ditulis, termasuk piksel terisi) | – | – | kecil |

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
- **`--limit N` dengan temporal (T-302):** frame t ditulis hanya bila SELURUH jendela inputnya ada — rentang `needed(t)` = jendela
  kedalaman × jendela grup, dipotong hanya di tepi KLIP (dan cut). Input seg / depth / qc ada sampai min(N + R_total, T) − 1 → semua N
  frame ditulis, byte-identik dengan run penuh (R_total = R grup + R kedalaman bila `depth.temporal`). Input di luar N tidak ada → hanya
  frame berjendela lengkap yang ditulis (< N), sisanya DITUNDA dengan peringatan + perintah (`segment` / `depth --limit X+1`); tidak
  pernah ada frame berjendela terpotong oleh `--limit` yang ditulis (resume akan menganggapnya valid padahal beda dari run penuh).
  Temporal mati: R = 0 → perilaku lama (hanya input N frame pertama).
- **Basi (T-302):** `temporal` di manifest (radius, cut, frame gagal QC, status optical flow) ikut dibandingkan; setelan temporal apa pun
  (α, b, `cut_diff`, `qc_fail_weight`, normalisasi, `depth.temporal`, `enabled`) sudah ada di `stabilize_hash` → `stable/` basi.
- **Basi (T-606):** key manifest `isolate` (ikut `MANIFEST_MATCH_KEYS`) **HANYA ditulis bila masker dipakai**:
  `{key_sha256 [1b], labels_sha256 (sha256 gabungan nama + sha256 byte PNG label frame target / kosong), shots, jumlah status,
  target_mask + target_mask_hash}`. Masker muncul / hilang / label berubah / parameter `target_mask.*` berubah → `stable/` basi →
  dihitung ulang (CPU); [1b] diulang dengan label byte-identik → TIDAK basi. Klip tanpa titik: key tidak ada di manifest lama
  maupun baru → manifest dan output tidak berubah, tanpa hitung ulang; `target_mask.*` tidak berpengaruh. **Kontrak / format [3]
  TIDAK naik** (tanpa field versi baru): menaikkannya akan membuat `stable/` SEMUA klip basi dan [4]–[6] dihitung + di-encode ulang.
  `--preview` / `--limit`: H per frame sebelum kernel → frame prefiks = frame run penuh; `--preview` menjalankan [3] apa adanya
  (masker dipakai bila berlaku; [1b] tidak dijalankan preview). `--restart-from isolate --yes` → label baru → [3] basi lewat
  `labels_sha256`; `--restart-from stabilize` → dihitung ulang dengan masker bila berlaku.
- **Basi (T-305f):** `stabilize.hole_fill` ada di section `stabilize` → ikut `stabilize_hash`; [3] TIDAK punya `ALGO_REV` (keputusan Rio).
  Pemasangan T-305f membuat `stable/` SEMUA klip basi sekali (peringatan `stabilize_hash: lama → baru`) → [4] basi (`stabilize_hash` +
  `stable_created_utc`) → [5] → [6]. Isi klip tanpa lubang yang memenuhi aturan tidak berubah (byte-identik; terbukti di 5 klip, docs/04
  "Hasil T-305f Tahap 2–3"). Mengubah `enabled` → basi lagi; `false` = peta grup sebelum T-305f.
- **CLI** (T-104b): `python -m rotoscope stabilize <video> [--config PATH] [--restart] [--limit N]`; di dalam `run`
  dipanggil in-process `stabilize.main([--work-dir <folder klip>, …])` (`--work-dir DIR` menang atas
  `paths.work_dir`). `--restart` stage CPU tidak butuh `--yes`. Exit code sama dengan [2]: 0 sukses, 1 prasyarat
  gagal (3 tidak dipakai — tanpa GPU).

### [4] `vectorize.py` (CPU)

**Status:** **T-305c `DONE`** (2026-10-12: parameter `depth_lines.hi_floor`, default 0 = mati, contract `"T-305c"`, `ALGO_REV` tetap 2; langkah 3 butir 4 dan docs/04 "Keputusan akhir T-305c"). **T-305b `DONE`** (2026-10-09: histeresis jarak terjaga D_low 2 + `exclude_groups [hair]`, contract `"T-305b"`, `ALGO_REV` 2; lihat langkah 3 butir 4 dan docs/04 "Hasil T-305b"). **T-201a `DONE`** (siluet + lubang + batas grup, 2026-10-02), **T-201b `DONE`** (garis oklusi +
`clip_stats.json`, 2026-10-03, **dengan batas kaki** — lihat langkah 3) dan **T-202 `DONE`** (orientasi, anchor, arah garis
terbuka, `track_id`, rantai kesinambungan, 2026-10-03) — strok `silhouette`, `silhouette_hole`, `group_boundary`,
`occlusion`. Subperintah `python -m rotoscope vectorize <video>`; bagian urutan `run` sejak T-203b (lihat "CLI"). Bagian bertanda *(T-201a)* / *(T-201b)* / *(T-202)* sudah diimplementasi dan diukur. Kode pelacakan:
`src/rotoscope/track.py` (stage ini memanggilnya per frame, **berurutan**).

- **In (T-201a + T-201b):** `stable/groups/`, `stable/depth_smooth/`, `stable/manifest.json`, `meta.json`; config `groups`,
  `vectorize.min_region_area`, `min_hole_area`, `line_min_px`, `min_stroke_px`, `vectorize.depth_lines.*` (`blur_sigma`,
  `hi_pct`, `lo_pct`, `erode_px`, `min_dist_px`, `min_len_px`). `stable/depth_smooth/` dibaca di T-201b (float16, harus
  ada untuk SEMUA frame klip, juga dengan `--limit`; hilang / rusak → berhenti, exit 1).
- **Langkah:**
  1. **Siluet** *(T-201a)* = batas foreground (`groups ≠ 0`) **TERMASUK lubang**: `cv2.findContours` mode
     `RETR_CCOMP` (2 level: kontur luar + lubang) dengan `CHAIN_APPROX_NONE`, bukan `RETR_EXTERNAL` saja. Ruang negatif
     tertutup (mis. lengan bertolak pinggang) wajib tetap digambar. Foreground di-pad 1 px background sebelum
     `findContours`, jadi kontur yang menempel tepi frame **tetap tertutup** (lihat "Kasus tepi").
     **Ukuran filter = jumlah piksel, bukan `cv2.contourArea`:** kontur luar dengan piksel komponen foreground 8-arah <
     `vectorize.min_region_area` dibuang; lubang dengan piksel background 4-arah yang terlingkup <
     `vectorize.min_hole_area` dibuang; lubang dari komponen yang dibuang ikut dibuang. Alasan: `contourArea` lewat pusat
     piksel, bias berlawanan untuk kontur luar (persegi 20×40 = 800 px → 741, jatuh di bawah ambang 800) dan lubang
     (lubang 10×10 → 121); jumlah piksel sama dengan satuan `island_min_px` di [3] dan "px²" di YAML. Kontur lubang =
     piksel foreground tetangga-4 lubang (4 sudut tidak ikut).
  2. **Batas grup** *(T-201a)* = batas antar pasangan grup (a, b), a < b menurut urutan `groups:` di YAML, keduanya ≠ 0 →
     polyline terbuka. Tidak termasuk batas dengan background (itu sudah menjadi `silhouette` / `silhouette_hole`).
     Batas di dalam satu grup tidak ada (sudah digabung di [3]). Per pasangan: band = piksel a bertetangga 3×3 b ∪ piksel
     b bertetangga a → komponen 8-arah < `line_min_px` dibuang → thinning → tracing (langkah 4).
  3. **Garis oklusi** *(T-201b)* dari `depth_smooth` **apa adanya — tanpa log kedua** (normalisasi sudah di [3]).
     Urutan operasi akhir (`occlusion_masks` + `occlusion_strokes`):
     1. |grad| = Gaussian σ `depth_lines.blur_sigma` → Sobel 3×3 **÷ 8** (`SOBEL_NORM`: Sobel pada ramp satuan = 8, jadi
        |grad| = selisih `depth_smooth` per piksel) → besar + (gx, gy).
     2. NMS searah gradien, 4 bin arah (0° / 45° / 90° / 135°, y ke bawah; non-maximum suppression = hanya piksel puncak
        tepi, `mag ≥` kedua tetangga searah gradien), hanya di **foreground ter-erode**: `erode_px` × `erode_px` persegi,
        `BORDER_CONSTANT 0` (di luar frame = background).
     3. Hysteresis 8-arah: komponen piksel > T_low dipertahankan bila memuat ≥ 1 piksel > T_high. T_high / T_low =
        persentil `hi_pct` / `lo_pct` **per klip** (default 95 / 90) dari |grad| di foreground ter-erode, SEMUA frame
        (lihat "Threshold per klip" dan `clip_stats.json`).
     4. Syarat jarak: hanya piksel foreground yang berjarak **L2 presisi** (`distanceTransform`, `DIST_MASK_PRECISE`) ≥
        `min_dist_px` (D, default 7) dari piksel batas grup terdekat. Batas = piksel dengan tetangga 3×3 berlabel beda,
        **termasuk batas dengan background dan TEPI FRAME** (peta di-pad 1 px background, sama dengan siluet). Memperlakukan
        tepi frame sebagai batas itu **disengaja** (keputusan Rio): foreground menyentuh tepi bawah di semua frame, jadi tanpa
        itu garis oklusi menduplikasi `silhouette` di dasar frame. Konsekuensinya garis oklusi tidak pernah dalam D px dari
        tepi frame.
        **Histeresis jarak terjaga** *(T-305b, `min_dist_low_px` = D_low, default 2; 0 = mati; valid 0 atau 1 ≤ D_low < D)*: piksel
        berjarak ≥ D = BENIH. Hanya komponen benih yang skeleton-nya lolos L (aturan langkah 5, per grup, sama dengan jalur tanpa
        ekstensi) boleh diperpanjang. Ekstensi = piksel hysteresis (T_high / T_low per klip, langkah 3) di foreground berjarak
        ≥ D_low, bukan benih, yang berada pada jalur 8-arah dari benih lolos-L ke DASAR zona D (jarak < D_low +
        `OCC_REACH_TARGET_BAND_PX` = 1,0) dengan panjang jalur ≤ ceil((D − D_low) / sin θ) langkah, θ =
        `OCC_REACH_MIN_ANGLE_DEG` = 45° (konstanta struktural, BUKAN parameter YAML: mengubahnya = perubahan perilaku →
        `ALGO_REV` naik). Jalur yang tidak sampai dasar (menempel sejajar batas) tidak diperpanjang; komponen seluruhnya di zona D
        tanpa benih tetap dibuang; tepi frame tetap batas. Penelusuran eksplisit TIDAK dibangun; histeresis polos ditolak
        (docs/04 "Keputusan T-305b"). `exclude_groups` *(T-305b, default `[hair]`)*: grup terdaftar tidak mendapat garis oklusi (benih
        dan ekstensi); `silhouette`, `silhouette_hole`, `group_boundary` tidak terpengaruh; nama tak dikenal = error; urutan dan duplikat
        dinormalisasi. D, D_low, `exclude_groups` tidak memengaruhi ambang `clip_stats` (persentil). Batas yang diketahui (docs/04
        "Hasil T-305b"): kedipan sambungan di torso, ujung > 20 px ref tidak terjangkau, `exclude_groups [hair]` bukan jaminan untuk klip lain.
        **Lantai absolut T_high** *(T-305c, `hi_floor`, default 0 = mati; ≥ 0, negatif ditolak)*: T_high yang dipakai = max(T_high
        persentil, `hi_floor`); T_low TIDAK disentuh (tetap persentil). `clip_stats.json` mencatat `hi_floor` (masukan), `t_high_pct`
        (persentil) dan `t_high` (dipakai); bila lantai menaikkan T_high, [4] mencetak satu baris info per klip. Mengubah `hi_floor`
        membuat `contours/` basi (ikut `vectorize_hash`). Default mati (keputusan Rio: 0,02 dievaluasi di klip3 dan ditolak secara
        visual); batas: docs/04 "Keputusan akhir T-305c".
     5. **Per grup** (mask ∩ `groups == g`, jadi satu strok selalu satu grup, juga saat D = 0): filter komponen 8-arah
        < `line_min_px` (M) → thinning Guo-Hall → komponen skeleton 8-arah < `min_len_px` (L, **jumlah piksel skeleton**) dibuang
        → tracing (langkah 4). **L dihitung dalam piksel Guo-Hall**: Guo-Hall membuang sudut tangga yang dipertahankan
        Zhang-Suen, jadi skeleton ±20–25% lebih pendek — L = 30 setara ±38 piksel Zhang-Suen (terukur di klip `test`:
        10 frame jadi kosong, 2 frame sebaliknya, dibanding pengukuran Tahap 1 yang memakai Zhang-Suen). T-305 mengkalibrasi
        L atas Guo-Hall.
     - **Loop:** konvensi sama dengan `group_boundary` (`closed: false`, titik akhir = titik awal). **Tidak ada komponen
       lintas grup:** mask dipisah per grup sebelum thinning (terukur: 0 komponen mask lintas grup).
     - **`strength`** = rata-rata |grad| (nilai sebelum NMS, float64) di titik strok (titik penutup loop tidak dihitung dua
       kali), dibulatkan 3 desimal (`STRENGTH_DECIMALS`).
     - **Persentil** eksak dan deterministik: nilai |grad| dikumpulkan (float32, ±71 MB untuk 283 frame), diurutkan, lalu
       interpolasi linear (`PERCENTILE_METHOD = "linear"`, = `np.percentile` default).
     - ⚠️ **Batas kaki (diketahui, keputusan Rio 2026-10-03):** `left_leg` / `right_leg` berisi **0 piksel** di kedua klip
       (Lower_Clothing ada di grup `torso`, D-009), jadi kaki menyilang hanya bisa muncul sebagai garis oklusi **di dalam
       `torso`**. Kriteria Done-when dinilai per wilayah: strok yang ≥ 80% titiknya di piksel Lower_Clothing
       (`seg/classmap`) pada frame 73 / 78 / 82 / 87 / 92, lolos bila ≥ 4 dari 5 frame. Default (D 7, L 30, p95 / p90):
       **4/5** di kedua klip (frame 78 gagal). Tepi tumpang tindih kaki **ada** di `depth_smooth` (jelas di panel |grad|
       level rendah) tetapi di bawah T_low default; ambang p80 / p70 menaikkan jadi 5/5 tetapi tambahannya sebagian besar
       lipatan celana. Normalisasi [3] (log + pembagi IQR per frame) melemahkan gradien di frame 73–83 (`log_iqr` ±0,33 vs
       median klip 0,19). Kalibrasi: normalisasi di **T-302**, ambang di **T-305**; default **tidak diubah**.
     - ⚠️ **Frame tanpa garis oklusi:** 34/119 (`test_short`) dan 160/283 (`test`) di default (run kosong terpanjang 9 dan 28
       frame). Frame tanpa strok `occlusion` adalah output valid (`"strokes"` tanpa tipe itu); [5] jangan menganggapnya error.
  4. **Filter komponen garis** 8-arah < `vectorize.line_min_px` (M = 5) pada mask garis (batas grup +
     oklusi) **sebelum thinning**; lalu thinning → tracing jadi polyline → jalur < `vectorize.min_stroke_px` (6) dibuang.
     **Tracing batas grup** *(T-201a, `ALGO_REV` 2)*:
     - **Thinning = `cv2.ximgproc.thinning` Guo-Hall**, bukan Zhang-Suen (penyimpangan dari look test; D-010 "Hasil
       T-201a"): Zhang-Suen mengikis habis band diagonal 45° berpadding.
     - **`prune_redundant`:** sudut tangga redundan (piksel > 2 tetangga tetapi crossing number ≤ 2) dihapus sebelum
       tracing — syarat: ≥ 2 tetangga ortogonal, tetangga satu komponen 8-arah, latar 4-bersebelahan satu komponen;
       piksel ujung dan pusat "+" tidak dihapus.
     - **Junction = crossing number ≥ 3** (jumlah transisi 0→1 di cincin 8 tetangga), **bukan** "≥ 3 tetangga": pada
       skeleton bertangga ±73% piksel terhitung junction dan garis hancur (cakupan 21–23%, diukur T-201a).
     - Piksel junction + tetangganya (satu klaster) dilepas; jalur dilacak; tiap ujung jalur di klaster disambung lewat
       jalur BFS terpendek di dalam klaster ke **satu titik wakil** (piksel crossing ≥ 3 paling sentral), jadi cabang yang
       bertemu berbagi satu titik dan tidak ada loncatan. Piksel ganda dalam satu strok dipotong hanya bila lingkaran di
       antaranya seluruhnya piksel klaster.
     - Jalur inti < `min_stroke_px` titik dibuang (spur kalau menyentuh klaster, selain itu fragmen); klaster dengan tepat
       dua ujung tersisa → kedua jalur digabung (garis tunggal kontinu).
     - **Loop tertutup** (grup dikelilingi grup lain): `closed: false` dengan **titik akhir = titik awal**, mulai dari titik
       (y, x) terkecil (skema tidak dilanggar: `group_boundary` selalu terbuka). [5] harus menangani sambungan ini
       (lihat [5], ⚠️ taper).
     - **Pertemuan tiga grup:** diproses per pasangan, jadi tiap garis berhenti dalam ±1–2 px dari titik temu. Batas yang
       menempel siluet berhenti di baris piksel terluar (koordinat sama dengan kontur siluet, selisih ≤ 1 px), tanpa snapping.
  5. **Anchor + orientasi + `track_id`** *(T-202)* (P-004) — aturan di "Aturan anchor + orientasi + `track_id`" di bawah.
     Dijalankan setelah langkah 1–4 membentuk dan mengurutkan strok mentah sebuah frame; **hanya urutan titik DALAM strok
     yang berubah** (pembalikan / rotasi), bukan himpunan titik, jumlah titik, jumlah strok, atau urutan strok. Frame
     diproses **berurutan** (pencocokan dengan frame sebelumnya, rantai `prev_sha256`).
- **Threshold per klip** *(T-201b)*: T_high / T_low dihitung sekali dari **seluruh** klip (pass 1 atas SEMUA frame, sebelum
  frame pertama diproses — juga dengan `--limit N`, jadi hasil N frame = prefiks run penuh) dan disimpan di
  `contours/clip_stats.json` — `--preview N` memakai nilai ini, bukan persentil dari N frame preview.
  Persentil menyesuaikan diri dengan distribusi `depth_smooth`, tapi D / L / persentil dikalibrasi
  ulang di T-305. Terukur (default): `test_short` T_high 0,1200 / T_low 0,0532 (7.913.162 nilai); `test` 0,1389 / 0,0641
  (18.545.710 nilai).
  - **Isi `clip_stats.json`:** masukan (`contract`, `algo_rev`, `blur_sigma`, `erode_px`, `hi_pct`, `lo_pct`, `sobel_norm`,
    `percentile_method`, `n_frames`, `frame_size`, `stable` {manifest, created_utc, groups_hash, stabilize_hash, seg_model},
    `clip`) + hasil (`n_values`, `t_high`, `t_low`; `null` bila klip tanpa foreground). Ditulis atomik.
  - **Basi:** file dipakai ulang HANYA bila blok masukannya sama persis (field hasil tidak dibandingkan); beda → pass 1
    dihitung ulang dan file ditulis ulang. `--restart` menghapus `contours/` (termasuk file ini).

**Out:**

| path | isi | disk / klip |
|---|---|---|
| `contours/frame_%05d.json` | polyline bertipe, titik rapat (±1 px, 1 desimal) | T-202 terukur: 4.57 MB (119 frame, 38.4 KB/frame), 11.26 MB (283 frame, 39.8 KB/frame) desimal (T-201b 4.54 / 11.18 MB; +0.7% untuk `track_id`, `anchor`, `prev_sha256`); ±14 MB *est.* untuk 360 frame |
| `contours/manifest.json` *(T-201a)* | lihat "Manifest [4]" di bawah | < 2 KB |
| `contours/frames.jsonl` *(T-201a, T-201b)* | log per frame: waktu, ukuran, jumlah strok per tipe, statistik filter (lubang / komponen dibuang, loop, spur), titik menempel tepi; T-201b menambah `n_occlusion`, `occ_px_hyst` (piksel setelah NMS + hysteresis), `occ_px_dist` (sesudah syarat D), `occ_px_len` (skeleton sesudah L), `occ_loops`, `occ_spurs_dropped`, `occ_short_dropped`; T-202 menambah `track_s` (waktu pelacakan), `n_matched`, `n_new_ids`, `n_flipped_vs_static` (garis terbuka yang arahnya berbeda dari aturan statis). Hanya frame yang DIHITUNG (bukan yang dipakai ulang) tercatat | ±100 KB |
| `contours/clip_stats.json` *(T-201b)* | T_high / T_low, persentil, jumlah nilai, referensi `stable/manifest.json` (lihat "Threshold per klip") | < 2 KB |

**Skema JSON (D-010, Q4) — skema akhir, ditulis lengkap sejak T-202 (lihat "Status T-202" di bawah):**

```json
{"frame_index": 87, "width": 480, "height": 854,
 "source": {"seg_model": "0.8b", "groups_hash": "…", "stabilize_hash": "…", "vectorize_hash": "…"},
 "prev_sha256": "…",
 "strokes": [
  {"track_id": 1, "type": "silhouette", "closed": true, "groups": ["background"],
   "points": [[x, y], …], "anchor": 0},
  {"track_id": 3, "type": "silhouette_hole", "closed": true, "groups": ["background"],
   "points": [[x, y], …], "anchor": 0},
  {"track_id": 7, "type": "group_boundary", "closed": false, "groups": ["left_arm", "torso"],
   "points": [[x, y], …]},
  {"track_id": 12, "type": "occlusion", "closed": false, "groups": ["torso"],
   "points": [[x, y], …], "strength": 0.034}]}
```

| `type` | `closed` | `groups` | keterangan |
|---|---|---|---|
| `silhouette` | true | `["background"]` | kontur luar foreground |
| `silhouette_hole` | true | `["background"]` | lubang foreground (ruang negatif tertutup) |
| `group_boundary` | false | 2 grup (urut sesuai YAML) | batas antar grup |
| `occlusion` | false | 1 grup | lompatan kedalaman di dalam grup; `strength` = rata-rata |grad| (3 desimal); loop = titik akhir = titik awal |

**Status T-202 (skema yang benar-benar ditulis).** Key tiap strok **persis** (urutan key tetap):
- `silhouette`, `silhouette_hole`: `{track_id, type, closed, groups, points, anchor}`; `anchor` **selalu 0** (konstan).
  `points` DIPUTAR sehingga `points[0]` = anchor; **[5] membaca `points[0]`**, bukan `anchor` (key dipertahankan hanya
  agar skema Q4 / D-010 utuh). Titik tertutup **tidak** mengulang titik pertama.
- `group_boundary`: `{track_id, type, closed, groups, points}` (`closed` selalu `false`; loop = titik akhir = titik awal).
- `occlusion`: `{track_id, type, closed, groups, points, strength}` (`strength` selalu ada, `groups` satu grup).
- `track_id`: bilangan bulat ≥ 1 (id pertama = 1, tidak ada nilai falsy), unik per klip.
- Key **level-frame** `prev_sha256`: sha256 (hex) byte berkas frame SEBELUMNYA di klip; `null` hanya untuk frame pertama.
  Rantai kesinambungan pelacakan (lihat "Aturan anchor + orientasi + `track_id`"). **[5] mengabaikannya.**
Urutan strok TIDAK berubah dari T-201b: tipe (silhouette, silhouette_hole, group_boundary, occlusion), lalu pasangan grup,
titik (y, x) terkecil, ...; urutan ditetapkan dari strok mentah SEBELUM normalisasi (tie-break terakhir memakai daftar
titik mentah, yang berubah kalau diputar). Titik tertutup tidak mengulang titik pertama; `group_boundary` dan `occlusion`
loop mengulangnya (lihat langkah 4).
**Regresi (terukur, kedua klip):** hash kanonik semua field strok (type, closed, groups, strength, titik; jumlah + urutan
strok per frame) identik dengan keluaran T-201b untuk keempat tipe. Kanonik: tertutup / loop = rotasi minimum dari kedua
arah, garis terbuka = urutan atau kebalikannya yang terkecil secara leksikografis. Beda byte hanya: key baru, dan pada loop
titik penutup mengikuti anchor baru (24 dari 30 loop `test_short`, 39 dari 61 `test` berbeda sebagai himpunan titik, semuanya
karena titik penutup); format serialisasi (separator kompak, 1 desimal) tidak berubah.
Blok `source` frame: `seg_model`, `groups_hash`, `stabilize_hash` disalin dari `stable/manifest.json` yang dibaca (bukan
dihitung ulang dari config), `vectorize_hash` = hash 11 parameter (4 T-201a + `depth_lines.*` + `track.max_match_dist_px`).

**Konvensi koordinat (T-201a):** ruang kontinu — piksel (i, j) menempati [i, i+1) × [j, j+1), titik disimpan di **pusat
piksel** (i + 0.5, j + 0.5). [5] menskalakan ke `output_width` langsung: x' = s · x, s = `output_width` / `width` (dengan
indeks mentah hasil bergeser 0.5 · s px, mis. ±1.1 px untuk 1080/480). Silhouette lewat pusat piksel terluar, jadi inset 0.5
px dari tepi sebenarnya (dapat diabaikan). Pembulatan 0.1 px (`COORD_DECIMALS`) di [4] **praktis no-op permanen**: semua
titik = k + 0.5 (1 desimal), kecuali [4] kelak menambah koordinat sub-piksel. Overlay (`scripts/contour_overlay.py`)
menggambar di (x · skala − 0.5).

**Kasus tepi (T-201a, terukur):** foreground menyentuh tepi bawah frame di **283/283** frame `test` dan **119/119**
`test_short` (tepi kanan 59/283; rata-rata 168 titik kontur menempel tepi di `test`, maks 634; 93 / maks 137 di
`test_short`). Padding 1 px menjaga kontur tertutup, tetapi kontur memuat **run titik di baris/kolom tepi** (y = H − 0.5,
x = 0.5, x = W − 0.5) yang tergambar sebagai garis lurus di dasar frame di tiap frame bila tidak ditangani. Run itu
terdeteksi persis dari koordinatnya (tanpa field skema baru; `edge_points` per frame di `frames.jsonl`). Keputusan
penanganan (sembunyikan / pudarkan / gambar) = **diputuskan T-203a: sembunyikan (`hide`)** (lihat [5]). `group_boundary` tidak
butuh padding: garis yang sampai tepi berakhir di tepi (terbuka, sah).

**Manifest [4]** (`contours/manifest.json`, prinsip #4): `stage`, `contract` (`"T-305c"` sejak T-305c; manifest `"T-305b"` / `"T-202"` / `"T-201a"` / `"T-201b"`
lama basi otomatis), `algo_rev`, `stroke_types` (4 tipe, termasuk `occlusion`), `pending` (`[]` sejak T-202),
`vectorize` (dict **datar** 11 kunci: 4 parameter T-201a
`min_region_area`, `min_hole_area`, `line_min_px`, `min_stroke_px` + 6 `depth_lines.blur_sigma`, `depth_lines.hi_pct`,
`depth_lines.lo_pct`, `depth_lines.erode_px`, `depth_lines.min_dist_px`, `depth_lines.min_len_px` + `track.max_match_dist_px`;
sejak T-305b juga `depth_lines.min_dist_low_px` dan `depth_lines.exclude_groups` (daftar terurut), sejak T-305c juga
`depth_lines.hi_floor`, jadi 14 kunci)
+ `vectorize_hash`,
`depth_thresholds` (`{t_high, t_low}`, salinan dari `clip_stats.json`) + `clip_stats` (nama file), `groups_hash`,
`stabilize_hash`, `seg_model` (dari `stable/manifest.json`), `stable_created_utc`, `frame_size`, `clip` (identitas klip,
T-108), `coords`, `created_utc`. Hash = 4 parameter T-201a + `depth_lines.*` + `track.max_match_dist_px`; mengubah salah
satunya membuat `contours/` basi (`clip_stats.json` hanya bergantung pada `depth_lines.*`, jadi mengubah `track.*` tidak
menghitung ulang ambang).
- **`algo_rev`** = konstanta bilangan bulat di `vectorize.py` (`ALGO_REV`), naik 1 **hanya untuk perbaikan PERILAKU pada kode
  yang sudah dikontrak, tanpa perubahan parameter** (sekarang 2: 1 = junction ≥ 3 tetangga + Zhang-Suen; 2 = crossing number +
  prune + klaster + Guo-Hall). **Fitur baru menaikkan `contract`, bukan `algo_rev`** (preseden T-201b dan T-202: tetap 2;
  `contract` yang naik sudah membuat output lama basi). Saklar `INHERIT_MAIN_SILHOUETTE` (pendekatan Y) dan konstanta
  pelacakan lain bukan parameter YAML: mengubahnya = perubahan perilaku → `algo_rev` naik. Ikut perbandingan basi:
  peringatan menyebut `algo_rev` lama → baru (manifest lama tanpa `algo_rev` → `None → 2`), `contours/` dihitung ulang otomatis.
- **Basi** (CPU murah, deterministik): field manifest berubah (parameter T-201a + `depth_lines.*` + `track.*`, `depth_thresholds`, `contract`, `algo_rev`, grup, `stabilize_hash`,
  `seg_model`, `stable_created_utc` — jadi [4] basi bila [3] dihitung ulang —, `frame_size`, `clip`) → `contours/` dihapus +
  dihitung ulang dengan peringatan. `contours/` tanpa manifest → ditolak. `--restart` menghapus `contours/` saja; `--limit
  N` = N frame pertama (prefiks run penuh); frame tanpa foreground = `"strokes": []`, valid.
- **Resume + rantai kesinambungan (T-202):** frame diproses berurutan; frame k valid (dipakai ulang, keadaan pelacakannya
  dimuat dari JSON-nya) hanya bila: `frame_index` / ukuran / `source` cocok, semua strok memuat key final tipenya dengan
  `track_id` bulat ≥ 1, `prev_sha256` = sha256 byte berkas k-1 yang BERLAKU saat ini (frame pertama: `null`), dan — bila
  frame k+1 ada di disk dan terbaca — `prev_sha256` k+1 = sha256 byte frame k (pemeriksaan frame-pengganti: frame yang
  diubah tangan tetapi JSON-nya masih valid tidak lolos). Selain itu dihitung ulang dari `stable/`. Karena byte frame k-1
  memuat `prev_sha256`-nya sendiri, rantai bersifat kumulatif: frame yang berubah mengubah byte-nya, sehingga k..N
  otomatis tidak valid; frame yang dihitung ulang dengan hasil byte-identik **tidak** membuat frame sesudahnya dihitung
  ulang. Counter id = max `track_id` semua frame sebelumnya + 1 (sama dengan run penuh karena id baru hanya diberikan ke
  strok yang ditulis); tidak memakai `frames.jsonl`. Terukur (kedua klip): hapus `contours/` lalu run dari nol → byte-identik;
  `--limit 20` lalu penuh → identik dengan run penuh; frame tengah dihapus / dipotong / diubah isinya (JSON valid) →
  1 frame dihitung ulang, hasil akhir byte-identik; frame 0 dihapus → idem. **Batas yang diketahui:** frame TERAKHIR
  klip yang diubah tangan (JSON valid) tidak terdeteksi (tidak ada frame pengganti sebagai bukti). Biaya terburuk hitung ulang
  berantai (frame 0 berubah) ≈ jumlah frame × ±80 ms.
- **Identitas klip:** `stable/manifest.json` hilang, tanpa `clip`, milik klip lain, `frame_size` ≠ `meta.json`, atau grup
  config ≠ `groups_hash` stable → berhenti (exit 1) dengan perintah `stabilize`; frame `stable/groups` hilang / rusak →
  berhenti. Exit code 0 / 1 (3 tidak dipakai).
- **Determinisme:** urutan strok tetap (tipe, pasangan id grup, titik (y, x) terkecil, jumlah titik, daftar titik); JSON
  kompak tanpa timestamp, tulis atomik; hash `contours/frame_*.json` identik antar run dari nol. `manifest.json`
  (`created_utc`) dan `frames.jsonl` (waktu) **tidak deterministik** → jangan ikut di-hash.

**Aturan anchor + orientasi + `track_id`** *(T-202, `src/rotoscope/track.py`; semua deterministik, seri diselesaikan
dengan aturan tetap)*:
- **Orientasi:** luas bertanda (rumus shoelace, y ke bawah): luas > 0 = searah jarum jam seperti terlihat di layar.
  `silhouette` dan loop (di bawah) searah jarum jam; `silhouette_hole` berlawanan. Terukur: 100% di kedua klip (mentah
  `findContours`: silhouette 100% berlawanan, lubang 100% searah, jadi semuanya dibalik). Kontur yang menempel tepi frame
  tetap benar (padding 1 px menjaga kontur tertutup; run tepi tetap pada koordinatnya karena hanya urutan yang berubah).
- **Anchor tertutup:** `points[0]` setelah rotasi (`anchor` konstan 0). Track lama = titik kontur terdekat (L2) ke
  `points[0]` padanan di frame sebelumnya (seri jarak → (y, x) terkecil, tidak bergantung urutan titik mentah). Track baru:
  `silhouette` = titik kontur terdekat ke piksel hair ∪ face tertinggi DI DALAM kontur itu (poligon terisi; seri → x
  terkecil; tidak ada hair / face di dalamnya, atau grup tidak ada di config → titik tertinggi kontur); `silhouette_hole`
  dan loop = titik tertinggi ((y, x) terkecil).
- **Garis terbuka** (`group_boundary`, `occlusion`) — **penyimpangan dari spesifikasi awal:** aturan awal ("titik awal =
  ujung dengan proyeksi terkecil pada sumbu utama") tidak menentukan tanda sumbu, dan aturan statis apa pun membalik arah di
  sudut pemotongannya (garis dengan sudut berganti-ganti di sekitar pemotongan). Terukur dengan aturan statis (tanda sumbu
  dikunci: komponen dominan positif): **7 dari 35** (`test_short`) dan **17 dari 83** (`test`) track hidup ≥ 5 frame pernah
  terbalik. Karena itu: track **baru** memakai aturan statis (sumbu utama PCA, tanda dikunci komponen dominan positif,
  titik awal = ujung berproyeksi terkecil, seri → y terkecil lalu x); track yang punya padanan memakai **kesinambungan**:
  dari dua arah, pilih yang jumlah jarak (titik awal, titik akhir) ke (titik awal, titik akhir) padanan paling kecil (seri →
  aturan statis). Hasil: 0 pembalikan track ≥ 5 frame di kedua klip (hampir tautologis, lihat ukuran independen di bawah).
- **Loop** (`group_boundary` / `occlusion` dengan titik akhir = titik awal): tidak punya "ujung" → diperlakukan sebagai
  tertutup (orientasi searah jarum jam, anchor dengan aturan tertutup), lalu titik awal diulang di akhir, jadi titik akhir =
  titik awal tetap terjaga (dites setelah normalisasi dan rotasi). Loop ber-padanan strok terbuka (atau sebaliknya) tidak
  gagal: acuan = `points[0]` padanan.
- **`track_id`:** per frame, cocokkan strok dengan strok frame sebelumnya yang `type` dan `groups`-nya sama. Jarak = **Chamfer
  simetris** `(rata-rata d(a→b) + rata-rata d(b→a)) / 2` (simetris supaya fragmen kecil tidak "mencuri" id strok besar yang
  terbelah). Penugasan satu-ke-satu **optimal** (`scipy.optimize.linear_sum_assignment`; jarak dibulatkan 6 desimal lalu indeks
  strok lebih kecil menang pada seri) atas pasangan dengan jarak < `vectorize.track.max_match_dist_px` (**default 16**,
  disetujui Rio; tidak ada yang cocok → id baru, bilangan bulat berurutan mulai 1, urutan strok tetap, unik per klip).
  Split / merge: satu padanan memperoleh id lama, sisanya id baru. Tidak ada toleransi celah (strok hilang 1 frame lalu
  kembali = id baru; terukur: 10 dari 52 oklusi, 3 dari 29 lubang, 3 dari 63 batas `test_short` yang lahir cocok dengan frame
  k-2). Optimal memaksimalkan **jumlah padanan lebih dulu**: pada ambang 16 segelintir padanan dekat bisa digeser demi satu
  padanan tambahan (kasus (b) di bawah).
- **Pendekatan Y** (`vectorize.INHERIT_MAIN_SILHOUETTE = True`, penyimpangan dari spesifikasi, disetujui Rio): setelah
  penugasan, silhouette berluas (poligon) terbesar frame ini mewarisi id silhouette berluas terbesar frame sebelumnya,
  **tanpa memandang jarak** (padanan lain yang melibatkan keduanya dilepas). Alasan: lengan yang sebentar terlepas (klip `test`,
  frame 213–266) membuat Chamfer silhouette utama 13–16 px; tanpa Y id utama berganti 10 kali (ambang 12) dan anchor kembali
  ke puncak rambut (lompatan 65 px). Bukan parameter YAML; mengubahnya = perubahan perilaku (`algo_rev` naik).
- `track_id` dipakai [5] hanya untuk komponen independen jitter (`stroke_independence` > 0; T-402). Resample ke N titik tetap dilakukan di [5], mulai dari `points[0]`.
- ⚠️ **Titik awal garis terbuka tidak stabil secara posisi:** terukur (Y@16, track ≥ 5 frame) titik awal melompat sampai
  60–127 px antar frame saat strok memanjang / memendek (aturan statis juga: 60,9 / 127 px), median 3–4,5 px, p95 18–23 px;
  aturan statis memilih ujung berbeda dari kesinambungan di 7–13% strok-frame. Jitter 1D berbasis panjang busur dari
  `points[0]` akan "pop" (lihat [5]).
- **Batas yang diketahui:** (1) padanan di zona 12–16 px (±1–3% padanan; kebanyakan track `group_boundary` yang bentuknya
  berubah) bukan bukti identitas tertukar; 4 dari ±3500 padanan (kasus (b)) bergeser oleh penugasan optimal padahal ada
  pendahulu lebih dekat; (2) id oklusi kaki berkedip lahir-mati (temporal [3] belum aktif, T-302/T-303/T-305); (3) fragmen
  lengan terpisah (klip `test`) mendapat id baru (2 dari 12 strok-frame mewarisi id fragmen sebelumnya), wajar sementara.

### [5] `stylize.py` (CPU)

**Status:** **T-203a `DONE`** (garis polos, 2026-10-03) + **T-401 `DONE`** (tebal variabel + taper + resample, 2026-10-06) + **T-402 `DONE`**
(jitter koheren, DEFAULT MATI) + **T-403 `DONE`** (multipass + opasitas, 2026-10-07; default 2 pass, offset 5,5, falloff 0,35, opacity 0,92, pass tambahan statis; `contract` `"T-403"`): subperintah
`python -m rotoscope stylize <video>`; bagian urutan `run`, tabel restart DAG, dan sumber
`export.source: "strokes"` sejak **T-203b**. Tekstur **belum aktif** (Phase 4; `ignored_params` di manifest). Preview `--preview N` = T-204. Tanpa GPU: `torch` tidak pernah di-import (diuji di subprocess).

- **In:** `contours/frame_*.json`, `contours/manifest.json`, `meta.json`, style YAML (`--style`, default
  `configs/styles/rough-sketch.yaml`, selain itu default kode); config pipeline (`--config`) hanya untuk `paths.work_dir`.
- **Out:** `strokes/frame_%05d.svg` + `strokes/frame_%05d.png` (RGB, latar `paper.color`) + `strokes/manifest.json` +
  `strokes/frames.jsonl` (log per frame: waktu per tahap, ukuran, jumlah strok / jalur / titik, statistik tepi).
  Terukur T-203a (1080×1922, garis polos): PNG median 67–71 KiB, SVG median 37–40 KiB per frame → 14–30 MiB per klip.
  **Terukur T-401 (default sekarang, `run` nyata):** PNG rata-rata 70,0 / 71,6 KiB, SVG rata-rata 89,9 / 91,9 KiB per frame
  (poligon kontur, `resample_points` 4) → 8,1 + 10,4 MiB (`test_short`, 119 frame) dan 19,8 + 25,4 MiB (`test`, 283 frame).
- **Satuan:** semua parameter panjang style = **px REFERENSI lebar 1080**; `unit = render.output_width / 1080`; geometri dihitung di
  px OUTPUT: titik kontur (pusat piksel kerja) × `s`, `s = output_width / width` (x' = s · x, sama untuk x dan y). Tinggi output
  = `round(height × output_width / width)` dinaikkan ke genap (integer: `(2·H·ow + W) // (2·W)`, +1 bila ganjil; 854 → 1922).
  Default look test dikonversi × 2,25 (look test menggambar di px KERJA); lihat docs/02 "Satuan dan parameter aktif".
- **Validasi masukan (exit 1, pesan menyebut perintah `vectorize`):** `contours/manifest.json` hilang / rusak; `contract` ∉
  `SUPPORTED_CONTOURS_CONTRACTS` (= `{"T-305c"}` sejak T-305c; `export` hanya memeriksa contract strokes; pesan menyebut yang ditemukan vs yang didukung); `pending` tidak kosong;
  `clip` hilang / milik klip lain (`meta_sha256`); `frame_size` ≠ `meta.json`; frame contours terpilih hilang / rusak / `frame_index`,
  ukuran, `source.vectorize_hash`, atau key strok tidak cocok. Hanya frame yang DIPILIH (`--limit`) yang divalidasi. Exit code 0 / 1.
- **Pipeline geometri per strok (satu geometri untuk SVG dan raster):** skala → penanganan tepi (mode hide) → **penghalusan
  Gaussian arc-length** (`shape.smooth_px`) → `approxPolyDP` (`shape.simplify_epsilon`) → spline Catmull-Rom seragam
  (`shape.smooth_tension`; `shape.spline_steps` = minimum titik per segmen, digandakan sampai galat akor ≤ 0,03 px ref) →
  ekstensi ujung di tepi → pembulatan 2 desimal (galat ≤ 0,007 px) → **resample arc-length (T-401)** → tebal per titik →
  **jitter (T-402; lihat "Jitter" di bawah)** → render.
  - **Resample (T-401):** seragam menurut panjang busur dari `points[0]`; `N = max(shape.resample_points, ceil(L / (RESAMPLE_MAX_GAP_REF ×
    unit)) [+1 bila terbuka])`, `RESAMPLE_MAX_GAP_REF` = 2 px ref (jarak titik ≤ 2 px ref, kesetiaan bentuk tidak memburuk: deviasi
    centerline p50 / p95 / maks per tipe sama dengan jalur spline T-203a ±0,01 px). `shape.resample_points` = batas bawah N (default 4).
    (Resample N = 200 TETAP di T-203a merusak bentuk; itu sebabnya N ditentukan jarak maks.)
  - **Tebal per titik (T-401):** `w = max(WIDTH_FLOOR_PX, width_base × unit × by_type.width_scale × (1 + width_variation × n) × taper)`.
    `WIDTH_FLOOR_PX` = 1,0 px OUTPUT (lantai absolut: di bawahnya raster ss = 3 terkuantisasi per 1/3 px, tidak monoton).
    `n` ∈ [−1, 1]: value noise 2D, terkunci POSISI (koordinat titik / `unit / width_noise_scale`), hash bilangan bulat 64-bit
    (splitmix64, numpy saja, `src/rotoscope/noise.py`), fade quintic + interpolasi bilinear → rentang dijamin konstruksi; seed =
    hash(`jitter.param_seed`) SAJA (tanpa `frame_index` / `track_id` / waktu: tebal STATIS terhadap waktu; boil = T-402). Dipilih
    atas varian busur dari `points[0]` (A) dan hibrida (C) dengan data (docs/04 "Keputusan T-401"): A pop p95 0,36–0,52 × `width_base`
    vs B 0,055–0,066; tertutup tanpa takik karena medan global (tanpa titik awal).
  - **Taper (T-401):** hanya ujung BEBAS; profil smoothstep dari `taper_min` (di ujung) ke 1; zona = `min(taper_px × unit, L / 2)`.
    TIDAK ditaper: ujung di tepi frame (ekstensi mode hide), ujung yang bertemu strok lain (≤ `JOIN_DIST_PX` = 8 px ref dari strok
    mana pun; oklusi ↔ oklusi ≤ `JOIN_DIST_OCC_PX` = 2 px ref), strok tertutup dan loop (titik akhir = titik awal). Override per
    tipe: `stroke.by_type.<tipe>.taper_ends` (`null` = warisi `stroke.taper_ends`).
  - **Penghalusan** (keputusan Rio; konstanta struktural bernama): kontur dire-sample 1 px ref; sudut tajam (belok > 60° pada
    ±6 sampel) dikunci dan memecah jalur; ujung strok terbuka (termasuk titik silang tepi) tidak bergeser (pantulan ganjil);
    strok tertutup dan loop (titik akhir = titik awal, dibuang titik penutupnya) periodik tanpa takik. Penyusutan luas silhouette
    rata-rata −0,008% (terburuk −0,09%). approxPolyDP tidak menjamin deviasi ≤ epsilon (terukur sampai +31%): epsilon BUKAN batas.
  - **Strok tertutup** digambar sebagai jalur tertutup (`Z`); `points[0]` tidak dipertahankan sebagai titik awal jalur (approxPolyDP
    memilih titik awalnya sendiri; tanpa efek visual di garis polos).
- **Tepi frame (mode `shape.edge_mode`, keputusan Rio: `hide`):** run titik di y = H − 0,5, x = 0,5, x = W − 0,5 (deteksi koordinat
  persis; cocok dengan `edge_points` di `contours/frames.jsonl` untuk silhouette + lubang: 119/119 dan 282/283 frame) **disembunyikan**: strok dipecah jadi
  k jalur terbuka (k run → k jalur), tiap ujung = titik silang tepi (titik tepi pertama), diperpanjang keluar **KANVAS** (bukan
  bingkai konten 1921,5) sejauh tebal/2 + 1 px: searah tangen akor terakhir bila ≥ 45° terhadap tepi, selain itu (kontak dangkal)
  tegak lurus tepi; sudut kanvas = jumlah normal. Strok yang seluruhnya di tepi tidak digambar. `draw`: run tepi digambar apa adanya
  (inset 0,5·s); peringatan stderr satu kali bila tebal < `draw_mode_min_width` (= s). Terukur: kontak dangkal 31/336 (`test_short`)
  dan 183/1152 (`test`) ujung; 0 ujung tepat di sudut kanvas; 0 ujung tidak mencapai tepi kanvas.
- **Raster:** satu mask supersampling (`render.ss`, union semua strok; tumpang tindih tidak lebih gelap) → `INTER_AREA` → tabel warna
  (kertas/tinta) → PNG (kompresi level 3). Garis = gabungan poligon terisi: trapesium per segmen (tebal berbeda di tiap ujung) +
  cakram berjari-jari per titik (sub-piksel 1/16) karena `cv2.polylines` hanya menghasilkan lebar ganjil. Isian even-odd poligon
  kontur DITOLAK (gagal di tikungan rapat; jari-jari < setengah tebal). `FILL_BIAS_SS` = 0,55 (kalibrasi terhadap luas poligon SVG).
  Koordinat cv2 = x' · ss − 0,5. Tinta solid pada jalur legacy (1 pass, `stroke.opacity` 1,0, `opacity_scale` 1,0); selain itu fraksi tinta f (lihat "Multipass (T-403)").
- **SVG (T-401):** string manual (bukan svgwrite: byte-determinisme penuh, tanpa atribut otomatis, tanpa dependency):
  `<svg width height viewBox>` = ukuran output, `<rect>` latar `paper.color`, satu `<g id=tipe>` per tipe (urutan silhouette,
  silhouette_hole, group_boundary, occlusion), SATU `<path>` terisi per jalur (poligon kontur: sisi kiri + tutup bulat
  `OUTLINE_CAP_STEPS` 8 + sisi kanan; strok tertutup = 2 sub-path orientasi berlawanan), `fill-rule="nonzero"` eksplisit, 2 desimal,
  `M x y x y … Z` (lineto implisit), TANPA `stroke-width` (tebal tidak bisa diubah ulang di editor); geometri SAMA dengan raster.
  Tanpa id acak / timestamp. Kesetaraan SVG–PNG (diukur, data nyata): IoU 0,992–0,996 (ambang 0,990), L1/tinta ≤ 0,0080
  (≤ 0,010), piksel |selisih| > 0,5 maks 0,15% (≤ 0,2%), rasio massa 0,997–1,000 (±0,5%).
- **Satu renderer untuk semua `type`**; parameter dasar + override per tipe (`stroke.by_type.*.width_scale`). Loop (`closed: false`
  dengan titik akhir = titik awal) diperlakukan sebagai tertutup. Garis oklusi berkedip digambar apa adanya (tanpa interpolasi antar
  frame; frame tanpa strok = hanya kertas, valid).
- **Manifest** `strokes/manifest.json` (prinsip #4): `stage`, `contract` `"T-403"` (T-203a / T-401 / T-402 → T-403: strokes lama basi otomatis; T-404a Opsi B TIDAK menaikkan contract), `multipass` (T-403: `passes`, `amplitude_px`, `cell_px`, `fold_r`, `alphas`, `temporal_mode`; tidak ikut pencocokan basi), `jitter` (T-402: `on`, `fold_r`, `fold_warn`, `edge_dead_px`, `edge_fade_px`; tidak ikut pencocokan basi), `algo_rev` (`stylize.ALGO_REV`, mulai 1; naik hanya
  untuk perbaikan perilaku; fitur baru menaikkan `contract`), `style` (nama), `style_hash` + `style_params` (HANYA parameter aktif),
  `ignored_params`, `contours` (`contract`, `vectorize_hash`, `algo_rev`, `created_utc`), `clip`, `frame_size` (kerja), `output_width`,
  `output_size`, `scale`, `unit`, `edge_mode`, `coords`, `created_utc`. **Basi** (CPU murah): field berubah → `strokes/` dihapus + dihitung
  ulang dengan peringatan menyebut field (parameter style aktif, ukuran output, contours, klip, `contract`, `algo_rev`); mengubah
  parameter yang belum aktif tidak membuat basi (dicatat satu baris). `strokes/` tanpa manifest → ditolak. `--restart` menghapus
  `strokes/` saja; `--limit N` = N frame pertama; `--from K --limit N` (T-204) = jendela K..K+N-1 (`--from` wajib bersama `--limit`; basi
  → `strokes/` dihapus seluruhnya, hanya jendela dihitung, dengan satu peringatan). **Resume per frame:** valid = SVG well-formed dengan `viewBox` / `width` / `height`
  benar dan PNG utuh (signature, IHDR = ukuran output, trailer IEND); tulis PNG terakhir.
- **Determinisme:** hash `strokes/frame_*.svg|png` identik antar run dari nol, `--limit 20` lalu penuh, dan setelah basi lalu kembali ke
  setelan awal. `manifest.json` / `frames.jsonl` memuat waktu → jangan di-hash.
- **Parameter style aktif / belum aktif:** docs/02 "Satuan dan parameter aktif (T-203a + T-401)". Aktif tambahan di T-401:
  `shape.resample_points`, `stroke.width_variation`, `stroke.width_noise_scale`, `stroke.taper_ends|px|min`,
  `stroke.by_type.*.taper_ends`, `jitter.param_seed`. Aktif tambahan di T-402: `jitter.amplitude`, `jitter.frequency`,
  `jitter.temporal_seed_mode`, `jitter.temporal_drift`, `jitter.hold_frames`, `jitter.stroke_independence`. Aktif tambahan di T-403: `stroke.opacity`,
  `stroke.by_type.*.opacity_scale`, `multipass.enabled`, `multipass.passes`, `multipass.offset`, `multipass.opacity_falloff`, `multipass.temporal_mode`.
  **T-404a (Opsi B): TIDAK ada parameter aktif baru di [5]** — `paper.enabled|texture_image|texture_opacity|texture_gain|vignette` ada di `ignored_params` (hanya `paper.color` aktif: LUT PNG); mengubahnya TIDAK membuat strokes basi. Kertas bertekstur dipakai export [6] (bagian "Kertas bertekstur" di [6]).
- **Kertas di [5] (T-404a, Opsi B; keputusan Rio, docs/04 "Hasil T-404a"):** [5] TIDAK merender kertas bertekstur / vignette. `strokes/*.png` (kertas DATAR `paper.color`, LUT 256 level `paper.color` → `stroke.color`) dan `strokes/*.svg` byte-identik T-403, `contract` tetap `"T-403"`. Alasan: opsi A (kertas di [5]) membuat PNG 1,6–3,5 MiB per frame (112 KiB datar) dan [5] 175 → 350 ms per frame. Modul `src/rotoscope/paper.py` (CPU-only) menyusun kertas saat export. Rumus (rinci di [6]):
  Kertas = `paper.color` × ((1 − op) + op × t') × vignette; t = luminansi `texture_image` (BT.601 pada sRGB, float32) → putar 90° berlawanan jarum jam bila kanvas portrait / landscape ketat berlawanan dengan gambar → potong "cover" di tengah TANPA resampling bila gambar ≥ kanvas di kedua sumbu, selain itu skala seragam `INTER_CUBIC` (selalu memperbesar) → normalisasi rata-rata 1,0 (akumulasi float64) → t' = max(1 + gain × (t − 1), 0,05); vignette v = 1 − `vignette` × d², d = jarak elips (sudut = 1). Dibulatkan (`rint`) ke uint8 untuk PNG; float32 dipakai metrik.
  Susun (`compose_textured`, saat export): PNG datar → level tinta per piksel (`LevelMap`, invers LUT) → di dalam jendela tinta piksel = `rint(kertas_f32 × (1 − a) + tinta × a)`, a = level / 255 (float32); di luar jendela = kertas uint8; vignette TIDAK mengenai tinta. **Tabrakan LUT** (level bertetangga berwarna sama): level = RATA-RATA level yang berbagi warna (galat ≤ 0,5 lebar kelompok), warna kertas → level 0; palet produksi (#f4f1ea → #1a1a1a) TANPA tabrakan (tiga kanal berkontras 218 / 215 / 208 langkah → warna unik per level) sehingga pemulihan eksak dan susunan export = oracle jalur lama PERSIS; warna di luar LUT = StageError. **Kertas datar** (`paper.enabled` false, atau tekstur tidak aktif [`texture_opacity` 0 atau `texture_gain` 0] dan `vignette` 0) = PNG dipakai apa adanya (MP4 byte-identik T-403; berkas tekstur tidak dimuat dan tidak wajib ada).
- **Metrik + toleransi test (docs/05 T-203):** kesetiaan = jarak titik kontur asli terskala (titik tepi dikecualikan pada hide) ke
  polyline akhir; toleransi sintetis = 0,35 × tebal garis (BUKAN epsilon), data nyata = ambang regresi dari angka terukur. Fungsi:
  `tests/stylize_metrics.py`.
- **Jitter (T-402, Tahap 2–3; keputusan Rio, docs/04 "Keputusan T-402")** — langkah `jitter_pieces` SETELAH resample + tebal per titik, SEBELUM render
  (arc-length taper, ujung bebas, "bertemu" memakai geometri TANPA jitter; SVG dan raster memakai titik yang SAMA, dibulatkan 2 desimal lagi).
  - **Model:** medan perpindahan KOHEREN vektor 2D `D(x, y, k) = amplitude × unit × [√(1 − s) F + √s G_track] × φ(d)`, s = `jitter.stroke_independence`.
    F = dua kanal value noise 3D independen (x dan y; fungsi POSISI + waktu saja: titik di lokasi sama → D sama, jadi sambungan siluet / batas grup /
    oklusi tetap menyambung, seam strok tertutup tidak retak, garis berdekatan tidak bersilang); G_track = medan sama dengan seed per `track_id`
    (hanya s > 0; menjaga varians, bukan puncak; batas |D| ≤ amplitude × unit × √2 × (√(1 − s) + √s)). BUKAN sepanjang normal strok (normal
    siluet dan batas grup berbeda arah di sambungan) dan BUKAN independen per strok (rencana lama D-010: sambungan terbuka sampai 2 × amplitudo, seam retak,
    garis bersilang). Terukur (amplitudo 4, 488 / 564 sambungan nyata): perubahan |jarak ujung ke strok lain| p95 koheren vektor 0,74–0,86 px vs normal
    3,3–4,0, independen 2D 3,6–4,0, independen 1D arc-length 4,7–4,8; seam 1D retak p50 1,6–2,0 px.
  - **Noise 3D** (`noise.value_noise_3d`): hash splitmix64 pada lattice (ix, iy, iz), bilinear + fade kuintik per irisan waktu, irisan bertetangga dicampur
    EQUAL-POWER (cos θ, sin θ; θ = f · π/2 linear; smoothstep ditolak: berhenti-jalan, kecepatan nol di simpul) lalu clamp [-1, 1] (trilinear: RMS "bernapas"
    −30%). Value noise (docs/05 menulis "Perlin"). Seed: `seed_of(param_seed, JITTER_SALT_FIELD, kanal)` dan `seed_of(param_seed, JITTER_SALT_TRACK, track_id, kanal)`;
    stream tebal = `seed_of(param_seed)` tanpa salt (tidak berubah).
  - **Waktu:** t = k × `temporal_drift`; k = floor(`frame_index` / `hold_frames`) dengan `frame_index` ABSOLUT (`doc["frame_index"]`; bukan relatif jendela
    `--from` / `--preview`), 0 pada `temporal_seed_mode` "fixed". Tanpa rantai: `--limit`, `--from`, `--preview`, resume = identik dengan run penuh.
  - **Tepi:** φ(d) = 0 untuk d ≤ `edge_dead` (jarak ke tepi bawah / kiri / kanan; di luar kanvas d < 0), smoothstep sampai 1 pada `edge_dead` +
    `EDGE_FADE_PX` (40 px ref); `edge_dead` = tebal maks / 2 + `EDGE_MARGIN_PX` dihitung dari style (7,75 px pada default). Tinta di 3 baris / kolom
    terluar tidak berubah; ekstensi mode hide tidak tertarik masuk. Tepi atas tidak dilunakkan (tidak dipotong [5]).
  - **Amplitudo 0 (atau frequency 0)** → jitter mati: SVG + PNG byte-identik dengan T-401 untuk nilai parameter jitter lain apa pun.
  - **Penjaga lipatan:** r = amplitude × frequency × (√(1 − s) + √s); r > `JITTER_FOLD_R_WARN` (0,19) → peringatan satu kali per run (stderr) + `strokes/manifest.json`
    → `jitter` (`fold_r`, `fold_warn`, `on`, `edge_dead_px`, `edge_fade_px`); tanpa clamp, tanpa error.
  - **Keselamatan (tests/jitter_metrics.py):** |D| ≤ batas; min det(I + ∇D) > `JACOBIAN_MIN_DET` (0,05) pada titik strok; sambungan (s = 0) |Δ celah| ≤
    `JITTER_JOINT_TOL_GRAD` (3,2) × r × `join_dist`; persilangan baru (pasangan strok yang tidak bersilang sebelum jitter) = 0 (di luar 8 px dari ujung strok);
    tinta tepi tidak berubah; ujung ekstensi tertarik 0.
    Besar getar (amplitudo, frequency, hold, drift, independence) = keputusan MATA Rio, bukan angka.
  - **Batas yang diketahui (docs/04):** garis yang bergerak melewati medan terkunci posisi berubah perpindahannya (efek GERAK, terpisah dari efek WAKTU);
    s > 0: pola G loncat saat `track_id` berganti (oklusi ±0,35 per strok-frame); taper / pop oklusi tidak dikerjakan; r > 0,19 diperingatkan (r 0,21 gagal Jacobian 0,05 di 2–17 frame).
  - **DEFAULT MATI (keputusan Rio, 2026-10-07):** `jitter.amplitude` 0 → stage [5] = T-401 byte-identik; sisa default: frequency 0,053, mode "frame", drift 0,35,
    `hold_frames` 2, `stroke_independence` 0. Fitur diaktifkan lewat style (docs/02, docs/04 "Hasil T-402").
  - **Multipass (T-403 DONE; keputusan Rio, docs/04 "Keputusan T-403" + "Hasil T-403")** — langkah `frame_passes` / `multipass_pieces` SETELAH `jitter_pieces`, SEBELUM raster.
  - **Pass:** pass 0 = garis asli (setelah jitter T-402). Pass k ≥ 1 = titik pass 0 + D_k, D_k = A × [n_x, n_y](x/λ, y/λ, t) × φ(d): mesin jitter T-402 (`pass_geometry` =
    `jitter_pieces` dengan amplitudo / sel / seed pass, s = 0), BUKAN sepanjang normal; seed = `seed_of(param_seed, MULTIPASS_SALT_FIELD 0x5D2B, k, kanal)`; φ(d) = penjaga tepi T-402.
    A = `offset` / `MULTIPASS_SEP_MEDIAN` (0,59) × unit (offset = median |D|), λ = A / `MULTIPASS_FOLD_R` (0,15) → r konstan, skala medan mengikuti offset. Tebal / taper / flag tepi disalin dari pass 0.
    Zona mati tepi pass k ≥ 1 = `edge_dead` pass 0 + `MULTIPASS_EDGE_EXTRA_PX` (1,0 px output): tinta 3 baris / kolom terluar tidak berubah pada semua frame kedua klip.
  - **Waktu:** `multipass.temporal_mode` "fixed" (default; t = 0, statis) | "frame" (t = floor(frame_index / `jitter.hold_frames`) × `jitter.temporal_drift`, frame_index ABSOLUT). Tanpa rantai antar frame.
  - **Alpha:** a_k = `stroke.opacity` × `opacity_falloff`^k (× `opacity_scale` tipe). Dalam pass union; antar pass "over": f = 1 − Π(1 − a_k cov_k) (komutatif). Tipe ber-scale 1,0 = satu lapisan;
    tipe ber-scale ≠ 1 = lapisan sendiri ("over", seperti grup SVG bersarang). Raster: mask per lapisan → INTER_AREA → f → 256 level → LUT warna.
  - **SVG:** jalur legacy (1 pass, opacity 1,0, scale 1,0) = byte-identik T-402. Selain itu `<g id="pass_K" opacity="a_k">` (4 desimal) berisi `<g id="pass_K_<tipe>">` (atribut `opacity` = scale bila ≠ 1);
    poligon tetap per strok; id unik; tanpa id acak / timestamp.
  - **Pengaman:** `multipass.enabled: false` atau `passes: 1` DAN `stroke.opacity: 1.0` = SVG + PNG byte-identik T-402. **Default (penilaian visual Rio, docs/04 "Hasil T-403"):** `passes` 2, `offset` 5,5, `opacity_falloff` 0,35, `temporal_mode` "fixed", `stroke.opacity` 0,92 (semua garis); jitter mati.
  - **Zona mati tepi pass k ≥ 1 (T-406, `ALGO_REV` 2):** zona mati φ(d) per TITIK = max(`edge_dead`, setengah tebal lokal + `EDGE_INK_GUARD_PX` 3,5 px ref), smoothstep kontinu (bukan potong keras);
    hanya pass k ≥ 1 (pass 0 dan jitter tidak berubah). Alasan: margin tetap `edge_dead` + 1 px menyisakan ±2 px dari ujung tinta ke tepi, padahal metrik tinta tepi memeriksa 3 baris;
    pembulatan titik 0,01 px dapat membalik satu subsampel (kebocoran 28/255) → bug laten (terlihat di pencil-light frame 194). Dampak pada rough-sketch hanya di titik zona tepi (docs/04 "Hasil T-406").
    `strokes/` lama (`algo_rev` 1) basi otomatis.
  - **Penjaga lipatan:** r jitter (bila aktif) + 0,15 > 0,19 → peringatan sekali per run + manifest; tanpa clamp. Batas: `opacity_scale` ≠ 1 membuat sambungan antar tipe bisa lebih gelap.
  - **Keselamatan (tests/multipass_metrics.py, per pass):** sambungan |Δ celah| ≤ toleransi, persilangan baru 0, seam (relatif 1,25× interior ATAU batas Lipschitz; interior dalam batas Lipschitz — `jitter_metrics.seam_ok`, docs/04 "Revisi T-403"), tinta tepi 0 piksel, ujung ekstensi, min det Jacobian > 0,05; angka: docs/04 "Hasil T-403".
  - **Raster non-legacy (biaya):** `ink_box` = jendela piksel yang memuat semua tinta semua pass (titik ± tebal/2 ± `INK_BOX_MARGIN_PX` 2); `render_mask(box)` + INTER_AREA + tabel float32 256 entri per lapisan hanya di jendela; hasil identik BIT dengan komposisi penuh-frame (`tests/multipass_metrics.ink_fraction_reference`); PNG ditulis langsung BGR. Jalur legacy tidak berubah.
- ✅ **"Jitter terkunci posisi"** (terjawab, T-402): medan koheren fungsi posisi + waktu; `points[0]` / `anchor` tidak dipakai. [5] membaca `points[0]`
    (bukan `anchor`) dan mengabaikan `prev_sha256`.
- ✅ **`output_width`** (terjawab, T-203a): 1080, satuan px ref × `unit`.
- ✅ **Run titik di tepi frame** (terjawab, T-203a): **hide** (lihat "Tepi frame").
- ✅ **`group_boundary` loop** (terjawab, T-401) = `closed: false` dengan titik akhir = titik awal (57 loop di `test`, 25 di `test_short`;
  terutama `hair|face` dan `torso|left_arm`; juga loop `occlusion`: 4 / 5): diperlakukan sebagai tertutup — tanpa ujung, tanpa taper,
  tanpa takik (noise terkunci posisi → tebal sambungan kontinu).
- ⚠️ **Batas yang diketahui T-401:** (1) 2,8–5,9% pasangan strok-frame bertrack sama berganti kelas ujung bebas ↔ bertemu antar
  frame, sehingga taper muncul / hilang (pop `group_boundary` maks 1,05–1,14 × `width_base`); taper dengan `taper_min` 0,5 +
  oklusi hidup (default): pop oklusi p95 0,39 (`test`) / 0,41 (`test_short`) × `width_base` (taper oklusi MATI 0,059 / 0,061; tipe lain
  0,063–0,071) — taper oklusi hidup dipilih Rio secara visual dengan angka ini diketahui; kestabilan taper lewat
  panjang strok yang dihaluskan antar frame (`track_id` ±2 frame) = bahan T-402. (2) Tebal tidak bisa diubah ulang di editor SVG.
  (3) Tebal STATIS terhadap waktu (tanpa boil).
- ⚠️ **Strok `occlusion` (T-201b):** (1) `strength` (rata-rata |grad|, 3 desimal) tersedia untuk memodulasi tebal / opacity
  tetapi skalanya per klip (bergantung normalisasi [3], akan berubah di T-302): jangan dipakai sebagai ambang absolut;
  (2) **temporal [3] belum aktif** (T-302/T-303), jadi garis oklusi antar frame masih berkedip (muncul / hilang, bergeser):
  frame tanpa `occlusion` valid (34/119 dan 160/283 di default, run kosong terpanjang 28 frame) — [5] jangan mengira itu
  error dan jangan menginterpolasi antar frame; (3) garis oklusi tidak pernah dalam ≥ D px dari siluet / batas grup /
  tepi frame (ujungnya terpotong di D, jadi tidak menyambung ke garis siluet atau batas grup); (4) `track_id` (T-202) tersedia
  untuk seed jitter, tetapi id oklusi berkedip lahir-mati (umur median 1–2 frame; 37–51% strok-frame di track ≥ 5 frame).
- **Urutan `run` (T-203b, DONE):** [4] dan [5] ada di urutan `run` (`ingest → segment → depth → stabilize → vectorize → stylize →
  export`) dan tabel restart DAG (bagian "CLI"); [6] memakai `strokes/*.png` dan menyalin `strokes/*.svg` (lihat [6]).

### [6] `export.py` (CPU)
- **Status:** Phase 1 (T-103, silhouette) + **T-203b `DONE`** (source `strokes`, default sejak T-203b; tag warna; salinan SVG).
- **MP4** (T-103): `out/<nama>.mp4`, `<nama>` = `export.filename` (default `"{source}.mp4"`; `{source}` =
  nama video sumber dari `meta.json`, disanitasi; **T-406:** `{style}` = nama preset / stem berkas style, disanitasi — `{source}_{style}.mp4` memberi MP4 + manifest + folder SVG per style). Satu jalur encode untuk semua sumber gambar
  (`export.source`): `"strokes"` (default) = `strokes/*.png` apa adanya (ukuran output [5], mis. 1080×1922 — bukan resolusi kerja;
  `foreground_color` / `background_color` tidak dipakai dan tidak ikut hash); `"silhouette"` (Phase 1) = `stable/groups/*.png`,
  grup ≠ 0 → `foreground_color` di atas `background_color`, resolusi kerja.
- **Masukan source `strokes`** (divalidasi sebelum encode; semua kegagalan = exit 1 dengan perintah yang benar):
  `strokes/manifest.json` ada, `contract` ∈ `SUPPORTED_STROKES_CONTRACTS` (`{"T-403"}`; T-404a Opsi B tidak mengubahnya; strokes `T-203a` / `T-401` / `T-402` → "jalankan stylize"; salinan SVG lama di `out/svg/<nama>/` disalin ulang sebagai basi, bukan suntingan), `clip` = `meta.json`, `frame_size`,
  `output_size` bilangan genap; referensi `contours` di dalamnya (contract, vectorize_hash, created_utc) = `contours/manifest.json`
  sekarang (rantai diperiksa satu tingkat; [5] sendiri memeriksa contours → stable) → jalankan `stylize`; setiap `frame_*.png`
  ada, utuh (IHDR + IEND) dan berukuran `output_size`.
- **SVG** (source `strokes`, tanpa `--limit`): `strokes/*.svg` disalin ke `out/svg/<nama>/frame_%05d.svg` (`<nama>` =
  `sanitize_source_name`, sama dengan nama MP4), di dalam stage [6] (satu pengaman, satu manifest), juga ketika MP4 dilewati.
  SVG telah diverifikasi terbuka dan bisa diedit di Krita (Rio, 2026-10-08; docs/05 T-405).
  **T-406:** folder SVG = `out/svg/<stem nama hasil>` (stem `export.filename` setelah `{source}` / `{style}` diganti), jadi mengikuti `{style}`.
  Salin atomik (`.tmp` + `os.replace`), diverifikasi jumlah + sha256 = `strokes/*.svg`. Folder memuat penanda
  `.rotoscope-clip.json` (identitas klip + sha256 per berkas SAAT DISALIN; ditulis tanpa BOM, dibaca toleran BOM). Klasifikasi
  per berkas (semua dievaluasi SEBELUM encode dan sebelum ada yang diubah): sama dengan sumber = ok; beda dari sumber tetapi sama
  dengan yang dicatat = basi (strokes berubah) → disalin ulang otomatis; hilang → disalin; **beda dari yang dicatat = disunting
  pengguna → exit 1** (menyebut berkas; hanya `--restart` yang menimpa). Pembersihan hanya untuk berkas yang dicatat penanda dan
  sudah tidak ada di sumber (klip lebih pendek) dan belum disunting; berkas asing / `.tmp` milik pengguna tidak disentuh.
  Penanda milik video LAIN → ditolak walau `--restart`. Penanda ADA tetapi rusak (tidak terbaca / bukan JSON / skema salah) ≠
  "tanpa penanda": ditolak dengan penyebab; `--restart` hanya bila SETIAP `*.svg` di folder identik byte dengan `strokes/`.
  Folder berisi berkas tanpa penanda → ditolak; `--restart` menimpa.
- **Encode:** frame RGB mentah di-pipe ke stdin ffmpeg (tanpa PNG sementara) → libx264, `-crf` / `-preset`
  dari YAML, `-pix_fmt yuv420p`, `-r` = `target_fps` `meta.json`, `+faststart`. Dimensi ganjil dipad 1 px warna
  latar (yuv420p butuh genap). Tulis `<nama>.mp4.tmp` → verifikasi → `os.replace` (retry Windows) — MP4 lama
  tidak rusak oleh run gagal.
- **Tag warna** (hanya source `strokes`; konstanta `COLOR_TAGS` di `export.py`, bukan YAML): colorspace / color_primaries /
  color_trc `bt709` + color_range `tv`. Tanpa tag ffmpeg menulis matriks bt601 tanpa penanda sedangkan pemutar HD mengasumsikan
  bt709: merah jenuh bergeser (+12, +15, −2) (terukur); kertas / tinta netral ≤ 1 level (tidak diskriminatif). Flag keluaran
  `-colorspace/-color_primaries/-color_trc/-color_range` saja hanya menulis colorspace + range ke stream (ffmpeg 9.0.1), jadi
  ditambah `-vf setparams=…` (tanpa filter `scale`; matriks mengikuti tag frame). crf default tetap 18.
- **Verifikasi ffprobe** (sebelum `os.replace`): jumlah frame = `frame_count`, fps, codec h264, pix_fmt
  yuv420p, ukuran = `output_size` (strokes) atau resolusi kerja setelah pad genap (silhouette), durasi ±1 frame, jumlah
  stream audio = `export.audio`; strokes: keempat tag warna (`color_space`, `color_primaries`, `color_transfer`, `color_range`)
  sesuai `COLOR_TAGS`.
- **Audio** (`export.audio`, default `false`): audio meme hampir selalu milik pihak ketiga (musik) → default
  tanpa audio; tambahkan audio dari library berlisensi di editor platform (TikTok/CapCut). `true` = audio video
  sumber (aac, `-shortest`); sumber tanpa audio (`has_audio` false) atau file sumber hilang = **error**.
- **Manifest** `out/<nama>.export.json`: hash parameter `export` (kecuali `filename`; untuk strokes juga tanpa
  `foreground_color` / `background_color`), identitas klip (sha256 `meta.json` + `source_path`), referensi masukan — source
  silhouette: `stable` (`stabilize_hash`, `groups_hash`, `created_utc`); source strokes: `strokes` (`contract`, `style_hash`,
  `created_utc`, `output_size`) + `color_tags` — jumlah frame, ukuran, fps, audio (ukuran + mtime file sumber), hasil ffprobe
  (termasuk tag warna), dan `ffmpeg` (baris pertama `ffmpeg -version`; **tidak** ikut kecocokan manifest — hanya menjelaskan
  perubahan hash MP4 setelah pembaruan ffmpeg). **T-406:** `style` (nama style strokes, source strokes saja) dicatat sebagai INFORMASI, di luar `MANIFEST_MATCH_KEYS`;
  bila target ada, source strokes, manifest lama mencatat style BERBEDA, dan `export.filename` tanpa `{style}` → satu baris `PERINGATAN` (MP4 + folder SVG ditimpa; perilaku tidak berubah;
  manifest lama tanpa `style` = tanpa peringatan). Mengganti `export.source` = basi → di-encode ulang; manifest silhouette lama
  tidak basi selama source tetap silhouette.
- **Basi / pengaman** (stage CPU murah + deterministik, prinsip #4):
  - manifest cocok + MP4 lolos ffprobe → **dilewati**; parameter / input berubah (termasuk `meta.json` klip
    yang sama) → **di-encode ulang otomatis** dengan peringatan (field lama → baru);
  - file tujuan ada dan manifest menunjuk video sumber **LAIN** → **ditolak** (ubah `export.filename`, atau
    pindah / hapus file itu); `--restart` **tidak** melewati pengaman ini;
  - file tujuan ada **tanpa manifest** → **ditolak** (asal tidak diketahui); `--restart` menimpa.
- **Preview jendela (T-204):** `--from K --limit N` → `<nama>.preview_K-<K+N-1>.mp4` (N frame posisi K..K+N-1; `--from` wajib bersama
  `--limit`); seperti `--limit`: tanpa manifest, tanpa pengaman, tanpa salinan SVG, tanpa audio; hanya PNG jendela yang divalidasi.
- **CLI** (T-104b): `python -m rotoscope export <video> [--config PATH] [--restart] [--limit N [--from K]]`; di dalam `run`
  dipanggil in-process `export.main([--work-dir <folder klip>, …])` (`--work-dir DIR` menang atas `paths.work_dir`;
  `paths.out_dir` tidak berubah). `--limit N` → `<nama>.limitN.mp4` (preview; tanpa manifest, tanpa pengaman, tanpa salinan
  SVG, tidak menyentuh hasil utama). Exit code: 0 sukses, 1 prasyarat gagal (3 tidak dipakai — tanpa GPU). Bentrok target export
  dengan video lain dicek lebih awal oleh pre-flight (c) `run` (bagian "CLI").
- **Identitas klip (T-108):** `stable/manifest.json` tanpa `clip` / milik klip lain → berhenti (jalankan ulang
  [3]; output basi dihitung ulang otomatis). Field `clip` manifest export memakai helper bersama
  `stage_common.clip_identity_from_bytes` (bentuk sama dengan sebelumnya).
- **Trim (T-607):** `{source}` = `<stem>@S-E` untuk klip ber-trim (`clip_name(source_path, trim_label(meta["trim"]))`, fungsi yang sama
  dengan folder kerja) → MP4, manifest, dan folder SVG per rentang (`klip5@0-3.833.mp4`, `out/svg/klip5@0-3.833/`). Klip tanpa trim:
  nama tidak berubah — `out/` test, test_short, klip3, klip4 dibuktikan byte-identik dengan kode export baru (Tahap 4, export ke folder
  scratch: MP4 + SVG identik, `.rotoscope-clip.json` hanya `created_utc`). Klip ber-crop: ukuran output mengikuti frame kerja baru
  (rumus [5] tidak berubah; Klip2 412 × 720 → 1080 × 1888, klip5 720 × 668 → 1080 × 1002).
- **Kertas bertekstur (T-404a, Opsi B; keputusan Rio):** `run_export(cfg, style=...)`, CLI `python -m rotoscope export <video> [--style PATH]` dan `run --style` (diteruskan ke stylize DAN export; `run` / `--preview` pre-flight memvalidasi `--style` dan, bila tekstur dipakai, gambar kertas ada + terdekode, sebelum stage mana pun / GPU). Source strokes: per frame baca PNG datar → `LevelMap` (invers LUT dari `paper.color` / `stroke.color` di `strokes/manifest.json` → `style_params`) → `compose_textured` di atas `render_paper` → encode (crf tetap 18). `paper.color` style harus = `paper.color` strokes (beda → StageError). Tanpa `style` (API) atau kertas datar → PNG apa adanya (MP4 byte-identik T-403). Manifest export: `paper` = `paper_ref(style)` (parameter kertas + `texture_sha256`; None bila datar; IKUT pencocokan basi → mengubah parameter kertas / isi berkas tekstur = MP4 di-encode ulang, strokes tetap), `paper_info` (ukuran, orientasi, fit, statistik; informasi). Salinan SVG tidak berubah. `--limit` / `--from` memakai jalur yang sama. Metrik kertas bertekstur (`tests/export_metrics.py`): mask tinta dari f pulih (`stylize_metrics.recover_f` dengan `render_paper(...).f32`), PSNR ≥ `PSNR_TEXTURED_MIN_DB` 39,8 (di antara crf 23 tertinggi 39,44 dan crf 18 terendah 40,19 pada kandidat op ≤ 0,5; aturan harfiah "min crf 18 − 1 dB" = 37,46 tidak membedakan crf 23), MAE tinta ≤ 4,2, selisih maks ≤ 90, bias kertas jauh dari tinta ≤ 3. Terukur (test_short frame 73–112, crf 18): ukuran vs datar ×1,10 (vignette) … ×1,72 (op 0,35 gain 1 + vig) … ×2,26 (gain 3) … ×3,09 (op 0,5 gain 6 / op 1,0 gain 3) … ×4,04 (op 1,0 gain 10) (opsi A, kertas di [5]; Opsi B menghasilkan MP4 setara karena frame sama ≤ 1 level — ukuran default: lihat docs/04 "Hasil T-404a").
- **Tekstur garis DITOLAK (T-404b, keputusan Rio, docs/04 "Hasil T-404b"):** export [6] tidak punya tekstur garis; satu-satunya tekstur adalah kertas (T-404a). Jalur komposisi tetap `paper.compose_textured(img, levels, layer, ink)`; manifest export tidak punya blok `texture`; style yang memuat blok `texture:` ditolak saat load.
- Terukur klip uji, silhouette (T-103; 283 frame, 480×854, crf 18): 2.4 s, 497.6 KiB (509 571 B).
- **Terukur source strokes (T-203b; 1080×1922, crf 18, medium):** test_short (119 frame) 2,4 s, 977,7 KiB, SVG 4,07 MiB (salin
  ≈ 0,12 s); test (283 frame) 5,4 s, 2620,9 KiB, SVG 10,16 MiB (≈ 0,25 s); encode ≈ 50 fps, decode PNG 14–16 ms/frame. Kualitas MP4
  vs PNG sumber (85 frame sampel): PSNR min 41,40 dB, MAE tinta (dilatasi 5 px) maks 3,50, selisih maks 75, kertas ±2,00, tinta datar
  ≤ 3,71. Ambang uji (`tests/export_metrics.py`): PSNR ≥ 40,4, MAE tinta ≤ 4,2, selisih maks ≤ 90, kertas ≤ 3, tinta datar ≤ 4,5.
  Determinisme: MP4 + SVG byte-identik antar run dari nol (mesin + build ffmpeg yang sama; versi ffmpeg ada di manifest).
- **Batas yang diketahui:** (1) determinisme MP4 hanya untuk build ffmpeg yang sama; (2) garis oklusi berkedip dan getar antar
  frame (temporal [3] belum aktif, T-302/T-303; jitter / taper / tekstur Phase 4); (3) tinta datar 4,5 vs crf 23 terukur 4,54 =
  lolos tipis — metrik itu BUKAN penjaga regresi crf (PSNR + MAE tinta + selisih maks yang menangkap crf 23); (4) identitas klip
  tidak melihat isi file video (batas T-108); (5) `--preview N` = T-204.

### `pick.py` — klik target (T-604; alat CLI, BUKAN stage, CPU)
- **Peran:** pengguna mengklik target di satu frame → titik + kotak per target, bahan isolasi target ([1b] `isolate.py`, T-605,
  SAM 2.1 tiny). Bukan stage: tidak masuk urutan `run`. Sejak T-605 `points.json` membuat `run` menjalankan [1b] (opt-in; `--yes`
  untuk GPU); keluaran [2]–[6] tetap byte-identik dengan atau tanpa `target/` (test).
- **In:** `meta.json` + `frames/frame_%05d.png`. `meta.json` belum ada → cli menjalankan [1] ingest dulu (argumen sama dengan
  `run`); sudah ada → `frames/` / `meta.json` tidak disentuh. **Out:** `target/points.json` (+ `target/points.prev.json` = isi
  sebelum simpan terakhir, cadangan satu tingkat); `--dry-run` → `target/pick_dryrun/frame_%05d.png` (kanvas yang akan tampil).
- **CLI:** `python -m rotoscope pick <video> [--frame N] [--config P] [--start S --end E] [--dry-run]` (T-607: trim memilih folder
  `<stem>@S-E`); `--frame` = angka di `frame_%05d.png`
  (default 0), harus di rentang klip. Satu sesi = satu frame; beberapa frame = beberapa sesi ke berkas yang sama. Exit 0
  (tersimpan / tanpa perubahan / dibatalkan / dry-run) | 1 (prasyarat: video, klip lain, ingest gagal, frame di luar klip / hilang,
  `points.json` rusak / milik klip lain / ukuran frame lain, jendela diskalakan sistem, OpenCV tanpa jendela, frame berubah selama
  sesi, gagal tulis) | 2 (argumen, mis. `--frame` < 0) | 130 (Ctrl+C; tidak disimpan).
- **Kontrol** (panel petunjuk Indonesia di kanan gambar; huruf besar / kecil sama):
  - klik kiri = titik target (label 1) pada target aktif; klik kanan = titik bukan target (label 0); tarik kiri ≥ 6 px tampilan =
    kotak target (satu per target per frame, kotak baru menggantikan; sisi < 6 px → diabaikan). Klik di panel diabaikan; tombol
    dilepas di luar jendela → gerak pertama tanpa tombol menyelesaikan kotak.
  - **N** = target berikutnya, berputar 1 … (id terbesar yang dikenal + 1) → 1 (dikenal = id di frame lain berkas + id berisi di
    frame ini). Warna tetap per id (1 hijau, 2 biru, 3 kuning, 4 ungu, 5 oranye, 6 toska).
  - **U** = batalkan aksi terakhir (titik, kotak — kotak lama kembali — atau R); titik lama yang dimuat ikut riwayat (id naik, titik
    urut klik, kotak terakhir). **R** = kosongkan frame ini (bisa di-U).
  - **Enter / S** = simpan + tutup. Target hanya klik kanan tanpa kotak → ditolak (jendela tetap terbuka); frame kosong + entri lama
    → konfirmasi (Enter / S lagi = hapus entri frame itu; tombol lain = batal); tanpa perubahan → tutup tanpa menulis. Target tanpa
    titik dan tanpa kotak tidak disimpan. Target tanpa klik kanan → `(!)` di panel + PERINGATAN di terminal setelah simpan (tidak
    memblokir).
  - **Esc / tombol X jendela**: tanpa perubahan → keluar; dengan perubahan belum disimpan → konfirmasi (Esc / X kedua = keluar tanpa
    simpan; Enter / S = simpan; tombol lain = lanjut; setelah X jendela dibuka lagi untuk konfirmasi).
- **Format `points.json` = `rotoscope-target-points/1`** (tulis atomik, `indent` 2):

  ```json
  {"format": "rotoscope-target-points/1",
   "clip": {"meta_sha256": "<sha256 byte meta.json saat simpan>", "source_path": "C:\\…\\klip5.mp4"},
   "frame_size": {"width": 720, "height": 1280},
   "coords": "piksel frame kerja frames/ (…; pusat piksel kiri-atas = 0, 0); label 1 = target, 0 = bukan target (konvensi SAM); box = [x0, y0, x1, y1] …",
   "frame_hash": "frame_pixel_sha256 = sha256(b'rgb8 <W>x<H>\\n' + piksel RGB uint8 …) - sidik piksel, bukan byte PNG",
   "frames": [
    {"frame_index": 0, "frame_file": "frame_00000.png", "frame_pixel_sha256": "<64 hex>", "updated_utc": "2026-10-16T08:00:00+00:00",
     "display": {"scale": 0.7531, "screen_wh": [1920, 1080], "work_area_wh": [1920, 1020], "dpi_aware": true},
     "targets": [{"id": 1, "points": [{"x": 343.0, "y": 407.0, "label": 1}, {"x": 612.4, "y": 500.9, "label": 0}],
                  "box": [140.0, 352.5, 498.7, 1003.2]},
                 {"id": 2, "points": [{"x": 80.0, "y": 700.0, "label": 1}], "box": null}]}],
   "tool": "python -m rotoscope pick"}
  ```

  Koordinat = piksel frame kerja (x kanan, y bawah, pusat piksel kiri-atas = 0, 0), dibulatkan 0,1, 0 ≤ x ≤ W − 1; urutan titik =
  urutan klik. `frames` urut `frame_index`, satu entri per frame; `targets` urut `id` (bulat ≥ 1, SATU identitas untuk semua frame:
  target 1 di f0 = target 1 di f64). Tanpa daftar target tingkat atas. `clip` = identitas T-108 saat simpan. `display` hanya
  diagnosis (tidak divalidasi).
- **Sidik frame `frame_pixel_sha256`** = sha256(`b"rgb8 <W>x<H>\n"` + byte piksel RGB uint8 hasil decode `stage_common.read_rgb`,
  baris demi baris). Sidik PIKSEL, bukan byte PNG (keputusan Rio 2026-10-16): kompresi PNG lain / ingest ulang dengan piksel sama →
  titik tetap berlaku (diuji: PNG dikodekan ulang, ingest ulang video sintetis).
- **Validasi** (`load_points`, dipakai pick dan kelak T-605). Fatal → pick menolak membuka (exit 1) dan TIDAK menimpa berkas: JSON
  rusak; `format` asing (`t602-points/1` → saran `scripts/t604_convert_t602.py`; versi > 1 → "versi lebih baru"); skema (label ∉
  {0, 1}, koordinat bukan bilangan hingga / di luar frame, kotak bukan 4 angka / terbalik / di luar frame, `frame_index` / `id`
  ganda, id < 1, `frame_file` ≠ indeks, sidik bukan 64 hex, frame tanpa target, target tanpa titik dan tanpa kotak, target hanya
  klik kanan tanpa kotak); `clip.source_path` ≠ `meta.json` (klip lain); `frame_size` ≠ `meta.json` (semua titik basi → pindahkan
  / hapus berkas; T-607: pesan menyebut "frame berubah sejak diklik (ingest memakai crop / trim / ukuran kerja lain)" + saran ganti
  nama ke `points.old.json` + perintah `pick` siap tempel; dipakai juga oleh [1b]). **Status per frame:** `ok` (sidik sama) | `basi` (sidik berbeda: isi frame berubah) | `hilang` (indeks di luar
  klip / berkas tidak terbaca). Frame ini basi → titik lama TIDAK dimuat (pesan di panel + terminal), simpan menggantinya; frame
  lain basi / hilang → peringatan + perintah pick ulang saat buka dan simpan (tidak dihapus diam-diam). `meta_sha256` berbeda saja =
  peringatan.
- **Simpan:** `points.json` DIBACA ULANG dari disk → divalidasi → ganti / hapus SATU entri frame ini (frame lain apa adanya;
  perubahan sesi lain di antara buka dan simpan tetap ada) → sidik frame dihitung ulang (berbeda dengan saat dibuka → tidak
  disimpan, exit 1) → isi lama disalin ke `points.prev.json` → tulis atomik. Entri terakhir dihapus → `frames: []`.
- **Tampilan / DPI Windows:** proses dibuat DPI-aware per monitor (`SetProcessDpiAwareness(2)`; hasil DICEK lewat
  `GetProcessDpiAwareness` karena panggilan kedua gagal tanpa exception; gagal → koordinat logis, klik tetap tepat, gambar lebih
  buram, `display.dpi_aware` false). Batas kanvas = area kerja (`SPI_GETWORKAREA`, tanpa taskbar) − (16, 56) px bingkai jendela;
  s = min(1, H_maks / H, (W_maks − 400) / W); gambar dw = round(W · s), dh = round(H · s) (half-up, `INTER_AREA`); pemetaan
  memakai sx = dw / W, sy = dh / H: tampilan → frame x = (dx + 0,5) / sx − 0,5 (clamp [0, W − 1], bulat 0,1), frame → tampilan
  dx = floor((x + 0,5) · sx); kanvas = (dw + 400) × max(dh, 600). `WINDOW_AUTOSIZE` (tanpa ubah ukuran), jendela di pojok kiri-atas
  area kerja; **penjaga:** ukuran `getWindowImageRect` ≠ kanvas → berhenti sebelum klik apa pun (exit 1); keluar area kerja →
  peringatan. Layar 1920 × 1080 skala 125% (aware, area kerja 1920 × 1020): frame 720 × 1280 → s 0,7531, gambar 542 × 964, kanvas
  942 × 964; 1 px tampilan ≈ 1,33 px frame. Konstanta UI (panel 400, tinggi min 600, cadangan 16 × 56, tarik 6 px, s maks 1, warna)
  = konstanta modul, BUKAN YAML (tidak memengaruhi keluaran pipeline; docs/02 tidak berubah).
- **Batas yang diketahui:** tanpa zoom (tidak diperlukan; diputuskan Rio setelah uji 2026-10-17); letterbox dibuang lewat [1] hanya bila
  kotak crop diterima (T-607; klip5 asli sejak 2026-10-23 720 × 668);
  satu frame per sesi; panel di kanvas pendek (600 px) dengan sangat banyak target + pesan bisa terpotong di bawah (blok saran dibuang
  dulu); belum diuji di multi-monitor / skala Windows selain 125%; teks panel ASCII (font Hershey).

### `crop.py` — saran + keputusan kotak crop (T-607, D-013; alat CLI `crop`, BUKAN stage, CPU)
- **Peran:** deteksi pita letterbox / pillarbox + kotak teks meme (aturan **R2 beku**) → SARAN; kotak dipakai [1] ingest HANYA bila
  diterima di sini (konfirmasi; tidak ada pemotongan otomatis). Modul juga dipakai ingest (deteksi, saran, pratinjau, `kotak.json`).
- **Berkas per SUMBER video** di `<paths.work_dir>/crop/<nama>/` (di luar folder klip; dipakai bersama semua folder `<nama>@S-E`):

  | berkas | format | ditulis oleh | isi |
  |---|---|---|---|
  | `deteksi.json` | `rotoscope-crop-deteksi/1` | ingest | kunci D1 + hasil (`pita_terdeteksi`, `box_work`, `n_frames`), juga "tanpa pita"; ditulis TERAKHIR (valid ⇒ saran + pratinjau lengkap) |
  | `saran.json` | `rotoscope-crop-saran/1` | ingest | kotak kerja (inklusif) + kotak sumber `[x, y, w, h]`, `band_px` / `overlay_px` / `cut_px` per sisi, sidik R2, `hints`; HANYA bila pita terdeteksi |
  | `pratinjau/` | `rotoscope-crop-pratinjau/1` | ingest / `crop --box` | `1_awal_fNNNNN.png`, `2_tengah_…`, `3_akhir_…` + `pratinjau.json` (kotak, frame, sha256) |
  | `kotak.json` | `rotoscope-crop-kotak/1` | HANYA alat `crop` | keputusan `saran` \| `manual` \| `none` + identitas sumber; isi lama → `kotak.prev.json` |

  **Identitas sumber** = `{sha256 byte, bytes, width, height}` (sesudah rotasi) + `path` sebagai info (TIDAK dicek: video yang dipindah
  tetap cocok). `kotak.json` / saran milik sumber lain → tidak dipakai (ingest exit 1 untuk `kotak.json`).
- **Aturan R2 (beku, docs/04 "R2 DIBEKUKAN"):** statistik temporal per piksel pada rentang penuh sumber (rata-rata + simpangan baku,
  BGR) → kupas pita statis (≥ `static_line_frac` 0,99 piksel dengan sd ≤ `static_std_max` 3,0) + seragam (`uniform_std_max` 6,0) dari
  tepi; pita sah hanya BERPASANGAN (atas + bawah, atau kiri + kanan) dan masing-masing ≥ `band_min_px` 8; kotak teks (baris statis tak
  seragam) hanya di atas / bawah dan hanya bila menempel pada pita; area aktif < `min_active_frac` 0,10 → tanpa potong. Sidik
  `d40c3dfd…` (sha256 JSON kanonik aturan + parameter); parameter lain → PERINGATAN dan `--accept` ditolak.
- **Pembulatan ke sumber:** kotak kerja inklusif → sumber `[x, y, w, h]`: awal floor, akhir ceil, lalu genap (yuv420) dan di-clamp ke
  frame = KE LUAR (`outward-even`): konten tidak pernah terpotong, sisa tepi tipis dibiarkan (Klip2 3 kolom gelap + 1 peralihan per sisi,
  klip5 4 baris hampir putih + 2 gelap; dinilai Rio di K6).
- **Petunjuk (informasi; kotak tidak diubah):** baris / kolom di DALAM kotak, dari garis potong ke dalam, yang ≥
  `ingest.crop.preview.hint_static_frac` (0,95) diam — hanya di sisi yang DIPOTONG saran. Diarsir oranye di pratinjau + disebut di
  pesan, dengan perintah `crop <video> --box X Y W H` siap tempel untuk kotak tanpa baris itu. Tidak dicetak bila kotak alternatif <
  `min_active_frac` frame, atau (keputusan Rio 2026-10-23) bila sesudah pembulatan keluar + genap kotak alternatif = kotak saran
  ("tidak ada perintah lain"; petunjuk tetap ditandai). Contoh klip11: atas 74 baris (388–461) → `--box 0 462 720 594`; Klip2: kanan 1
  kolom (474) → `--box 434 0 410 720` (≠ saran 412: 2 kolom sumber lagi).
- **Pratinjau:** 3 frame kerja rentang penuh (awal / tengah / akhir), resolusi kerja, satu PNG per frame: asli + kotak magenta | zoom kiri
  | zoom kanan | zoom atas + bawah (bila dipotong atas / bawah) | hasil potong; zoom 48 px × 4 berpusat di garis potong (48 px terluar bila
  sisi itu tidak dipotong); header ASCII (frame, kotak kerja + sumber, pita / kotak teks per sisi, petunjuk). Konstanta tata letak di
  modul (preseden `pick`), bukan YAML. `crop --box` menulis pratinjau kotak manual (jenis `manual`).
- **CLI:** `python -m rotoscope crop <video> [--accept | --box X Y W H | --none | --clear] [--config P]` (satu aksi; `--config` untuk
  `paths.work_dir` + parameter R2):
  - tanpa aksi = **status**: identitas sumber, saran (pesan yang sama dengan ingest), `kotak.json` (atau "belum diputuskan" / "MILIK
    SUMBER LAIN"), dan per folder klip sumber ini ukuran frame + apakah `meta.json`-nya sudah ber-crop;
  - `--accept` → `kotak.json` `mode: saran`. Ditolak (exit 1) bila tidak ada saran untuk sumber ini (perintah `ingest` siap tempel) atau
    sidik saran ≠ R2 beku;
  - `--box X Y W H` (px SUMBER sesudah rotasi) → dibulatkan keluar + genap, divalidasi (di dalam frame, ≥ 16 × 16, ≠ frame penuh → pakai
    `--none`) → `mode: manual` (`box_requested` + `box_source`) + pratinjau kotak itu;
  - `--none` → `mode: none` (saran ditolak; `meta.json` tanpa key `crop`, byte-identik dengan klip tanpa kotak; `run` tidak berhenti);
  - `--clear` → `kotak.json` dipindah ke `kotak.prev.json` (ingest berikutnya tanpa potong; saran masih ada → `run` berhenti lagi).
  - Sesudah `--accept` / `--box` / `--none` / `--clear`: satu perintah per folder klip sumber ini yang sudah di-ingest:
    `python -m rotoscope run <video> [--start S --end E] --restart-from ingest --yes` (MENJALANKAN GPU) + catatan titik `pick` basi bila
    ukuran frame berubah; belum ada folder → satu perintah `run <video>` biasa. `--start` / `--end` ditolak (exit 2). Exit 0 / 1 / 2.
- **Terukur (Tahap 3a):** deteksi (pipa rentang penuh) 9,5–12,8 ms / frame di 720 × 406, 26–31 ms / frame di 720 × 1280; pratinjau
  0,55–2,2 MB per PNG; berkas JSON < 4 KB. Kotak diterima Klip2 (412 × 720) dan klip5 (720 × 668) = frame P-src studi Tahap 1a
  byte-identik.
- **Batas yang diketahui:** lihat [1] (satu kotak per sumber; R2 bisa salah → konfirmasi wajib; validasi buta K2' klip11 tidak lagi buta);
  UI tarik kotak (seperti `pick`) = backlog docs/05.

### `batch.py` — banyak klip (T-603; subperintah `batch`, BUKAN stage)
- **Peran:** menjalankan `run` untuk tiap klip secara berurutan (satu proses per klip; RAM bersih; tanpa paralel, tanpa model tetap).
  Modul `src/rotoscope/batch.py`; dispatch di `cli.main` SEBELUM argparse cli (flag terlarang → exit 1, bukan exit 2).
- **Masukan:** argumen video atau folder (folder: tidak rekursif, urut nama, hanya ekstensi `batch.video_exts`), dan/atau
  `--list BERKAS`. Format daftar: satu klip per baris `<path> [--start S --end E]`; baris kosong dan `#` = komentar; path
  berspasi WAJIB kutip ganda: `"D:\Meme Lucu\video 1.mp4" --start 0 --end 2`. Parsing: `shlex.split(posix=False)` (backslash
  Windows tetap); kutip ganda luar yang berpasangan dilepas; kutip tidak berpasangan / path tanpa kutip yang berisi spasi → exit 1
  dengan nomor baris. Trim hanya di daftar (tidak di YAML).
- **Dua klip dengan folder kerja sama** (stem sama dari folder berbeda, atau baris dobel) → exit 1 sebelum apa pun.
- **Rencana (tanpa `--yes`):** ingest (sama dengan `run`; opsi A T-108) + status: crop (`tanpa pita` / `diputuskan` / `BELUM
  diputuskan`), target (`tidak ada` / `ada` / `ada (up-to-date)` / `BASI`), identitas [2]/[2c] (`ok` / `BASI`), frame GPU per
  stage, perkiraan GPU (`GPU_SECONDS_PER_FRAME`), disk baru. Tindakan siap tempel (`crop …`, `pick …`, `run … --restart-from
  ingest --yes`) dicetak; batch TIDAK menjalankannya. Dalam rencana `isolate.run_isolate(dry_run=True)` hanya dipanggil bila
  `target/points.json` ada; tanpa titik tidak ada pemanggilan (jalur hapus `skip_without_points(dry_run=False)` tidak tersentuh).
  Rencana tidak menulis di `seg/ depth/ isolate/ stable/ contours/ strokes/ out/` (diuji: snapshot sama).
- **Jalan (`--yes`):** rencana dicek ulang → klip `perlu tindakan` berhenti SEBELUM GPU (exit 4) kecuali `--skip-pending` (klip itu
  `dilewati`) → cek disk (kurang → exit 1 sebelum GPU) → VRAM sekali (`nvidia-smi` memory.free ≥ `segment.vram_min_free_mib[model]`,
  memakai `segment.check_vram`; `nvidia-smi` tidak ada → peringatan, lanjut) → tiap klip `python -m rotoscope run <video> [--start
  --end] [--config] [--style] [--seg-model] --yes`. Perintah anak selalu memuat `--yes`.
- **Exit rencana:** 0 siap · 4 ada klip perlu tindakan · 1 gagal (pre-flight, duplikat, ingest, disk kurang, flag dilarang).
- **Exit jalan:** 0 semua selesai · 1 ada klip gagal (klip lain tetap jalan) · 4 ada klip dilewati tanpa gagal, atau berhenti sebelum
  GPU karena perlu tindakan · 130 Ctrl+C. Gagal + dilewati sekaligus → 1. Ctrl+C: proses anak dihentikan (`_stop`), klip sisa
  berstatus `belum`, batch berhenti.
- **Ringkasan JSON** (mode jalan; juga saat Ctrl+C dan exit 4): `<paths.work_dir>/batch/ringkasan_<UTC>.json`, format
  `rotoscope-batch/1`: `started_utc`, `exit_code`, `counts`, `klip[]` (nama, video, trim, status, exit_code, detik, work_dir, out_mp4).
  Catatan saja: waktu tidak dipakai untuk keputusan.
- **Ringkasan akhir di layar** (T-603 keputusan Rio no. 1, mode jalan; juga saat Ctrl+C dan exit 4): tabel per klip (#, klip, status
  `selesai` / `gagal` / `dilewati` / `dihentikan` / `belum dijalankan`, exit, waktu mnt, output MP4), baris `Exit N = <arti>`
  (0 semua selesai · 1 ada gagal · 4 ada dilewati tanpa gagal · 130 Ctrl+C), klip dilewati dengan alasan + perintah tindakan, path JSON.
  Mode `--yes` tidak mencetak `Keluar` / `Untuk menjalankan`; sebagai gantinya `Mulai: N klip dijalankan, M dilewati` (tambahan
  ` (--skip-pending)` hanya bila M > 0).
  Kolom GPU dan disk = `—` untuk klip `dilewati` / `perlu tindakan` / `gagal`.
- **Ctrl+C di stage GPU** (keputusan Rio no. 3): satu baris `dihentikan (Ctrl+C) di frame k/N (frame_xxxxx.png); jalankan ulang untuk
  melanjutkan`, exit 130, tanpa traceback (segment, depth; isolate: posisi `[1b] frame k/N`). Cleanup backend yang kena Ctrl+C kedua
  dibuang (`stage_common.close_quietly`). Stage CPU via `run`: satu baris `dihentikan (Ctrl+C) di stage [x]`, exit 130.
- **Estimasi disk** (T-603 Tahap 2–3, keputusan E): `batch.disk_mib_per_mpx` × (W × H dari `meta.json`, juta piksel) × frame yang
  belum ada (klip dengan GPU baru atau hilir belum lengkap = semua frame; klip lengkap = 0). Ruang bebas minimum
  `batch.disk_reserve_gib`. Maks terukur 8,08 MiB/MP (klip5, T-603 Tahap 1).
- **Koreksi (T-603 Tahap 1):** angka "±21–33 s" di prompt hanya segment + depth. Overhead `run` klip selesai terukur 30–58 s per klip
  (dominan [2] describe + QC seluruh klip yang dihitung ulang setiap run); di luar lingkup T-603.
- **VRAM:** `nvidia-smi` berjalan di mesin Rio dengan VRAM bebas ±2965 MiB saat Tahap 1 (< ambang segment 0,8b 3300 MiB) → `--yes`
  gagal sampai aplikasi GPU ditutup (pesan exit 1 menyebut browser / aplikasi berat).

### Anggaran disk per klip (360 frame, *est.*)

| stage | disk |
|---|---|
| [2] classmap + probs + QC | 0.1–0.5 GB (tanpa kompresi 4.3 GB) |
| [2c] depth | 0.3 GB |
| [3] groups + depth_smooth | 0.3 GB |
| [4] contours | < 0.1 GB |
| [5] strokes | 0.014–0.030 GB (terukur T-203a); T-401: 0.018 / 0.045 GB (`test_short` 8,1 + 10,4 MiB; `test` 19,8 + 25,4 MiB) |
| **total [2]–[5]** | **±0.75–1.15 GB** (est. lama 1.0–1.9 GB; sisanya tidak berubah) |

C: sisa ±54 GB → arahkan `paths.work_dir` ke drive lain kalau banyak klip disimpan bersamaan.

### Anggaran waktu + VRAM (T-102c, D-005)

| stage | model | VRAM reserved | waktu / klip 360 frame |
|---|---|---|---|
| [2] | seg 0.8B fp16 | 3276 MiB (cek ≥ 3300 bebas) | ±95 mnt (15.85 s/frame) |
| [2] fallback | seg 0.4B fp16 | 2258 MiB (cek ≥ 2300 bebas) | ±48 mnt (7.99 s/frame) |
| [2c] | DA-V2 Small fp32 | 424 MiB (cek ≥ 500 bebas) | ±1 mnt (0.185 s/frame) |
| [1b] (opt-in) | SAM 2.1 tiny fp32 | 892 MiB (T-602, 92 frame; cek ≥ 2000 bebas) | hanya frame di adegan berklik: ±0,67 s/frame (T-602 p50) + muat; ≤ ±4 mnt untuk 360 frame berklik |
| [3] spasial | CPU | – | ±43 s (0.12 s/frame, T-106) |
| [4] T-201a | CPU | – | ±11 s (29–30 ms/frame rata-rata, p95 34–35 ms, maks 50 ms; 283 frame 8.5 s) |
| [4] T-202 (+ oklusi + pelacakan) | CPU | – | median 74–93 ms/frame antar run (p95 84–112 ms, maks 88–135 ms, 0 frame > 1 s, target ≤ 150 ms; pelacakan sendiri median 8–11 ms, maks 19–26 ms), pass 2 saja; pass 1 ambang (baca ulang + gradien semua frame) 1,3 s (119 frame) / 2,7 s (283 frame) |
| [5] T-401 | CPU | – | **terukur `run` nyata (default T-401):** rata-rata 198 ms (`test_short`; p95 238, maks 252, 1 frame > 250) dan 235 ms (`test`; p95 267, maks 407, 60 frame > 250) per frame; 119 frame 23,9 s, 283 frame 67,4 s. Ukur ulang di scratchpad: rata-rata 199 / p95 239 / maks 283 (6 frame > 250); ukuran awal Tahap 3 (resample 4, default lama) 153–157 ms — beda tidak dipecahkan (dugaan beban mesin, belum diverifikasi). Target 250 ms/frame dilampaui pada sebagian frame |
| [5] T-203a | CPU | – | median 125–128 ms/frame pada 1080×1922 (p95 134–137, maks 148, 0 frame > 0,3 s; geometri + penghalusan ±36–38, mask 22, downsample + warna 27, encode PNG 41, SVG 4, tulis 5); 119 frame 16 s, 283 frame 39 s; memori puncak ±127 MiB |

---

## Stack & environment

```
Python 3.11.9
torch==2.7.1+cu118          # GPU untuk Sapiens2-seg + DA-V2 Small. Wheel cu118: driver 517.00
torchvision==0.22.1+cu118   # (CUDA maks 11.7) tidak bisa cu126/cu128; cu118 memuat sm_75 (Turing)
transformers==5.17.0        # Sapiens2 (sejak 5.10.1, Python >= 3.10) + Depth Anything V2
opencv-contrib-python       # contour, optical flow, thinning (ximgproc) — dibawa mediapipe.
                            # JANGAN tambah opencv-python (dua paket OpenCV bentrok di modul cv2)
mediapipe                   # fallback pose — SKIP (2026-10-27, revisi D-010 Q3); paket membawa opencv-contrib-python
numpy, scipy                # resampling, interpolasi
svgwrite                    # terpasang (MIT) tetapi TIDAK dipakai [5]: SVG ditulis sebagai string manual (T-203a)
Pillow                      # raster render
pyyaml                      # config
ffmpeg                      # ingest + muxing (binary eksternal, gyan.dev essentials build, via subprocess)
```

- Index PyTorch cu118 = `--extra-index-url` di `requirements.txt`; PyPI tetap index utama.
- Checkpoint model di cache Hugging Face (di luar repo), dimuat dengan `HF_HUB_OFFLINE=1`,
  revision di-pin di YAML.
- GPU: GTX 1650 Ti 4 GB, Turing sm_75 → **tanpa bf16** (P-005), fp16/fp32 saja. Attention `sdpa`.
- Versi terkunci: `requirements.txt` (dependency langsung) dan `requirements-lock.txt` (full freeze).
- `rembg` + `onnxruntime` DITOLAK (D-008) dan sudah dikeluarkan dari `requirements.txt` / venv (T-107, 2026-10-02);
  `onnxruntime-gpu` tidak dipakai (T-601 SKIP). scikit-image ikut dicopot: thinning di stage [4] =
  `cv2.ximgproc.thinning` dari `opencv-contrib-python` (terverifikasi di T-107). Jangan ganti dengan
  `opencv-python` / headless (modul `ximgproc` hanya ada di contrib).
- ⚠️ OpenCV 5.x dan mediapipe 1.x adalah versi mayor baru — jangan asumsikan API OpenCV 4.x /
  mediapipe 0.10.x dari tutorial lama.

## Struktur repo

```
rotoscope/
├── src/rotoscope/
│   ├── ingest.py  segment.py  depth.py  fallback_pose.py (ditunda)
│   ├── isolate.py        # [1b] (T-605): masker target SAM 2.1 tiny dari target/points.json (opt-in, GPU)
│   ├── stabilize.py  vectorize.py  stylize.py  export.py
│   ├── config.py  cli.py  __main__.py   # python -m rotoscope (T-104b)
│   ├── pick.py           # `pick` (T-604): jendela klik target → <klip>/target/points.json (bukan stage)
│   ├── crop.py           # [1] saran crop R2 beku + alat `crop` (T-607): <work_dir>/crop/<nama>/kotak.json (bukan stage)
│   ├── stage_common.py   # helper bersama stage: tulis atomik + retry, frames.jsonl, daftar frame,
│   │                     # error/exit code; khusus GPU: offline/revision/cache HF, VRAM, OOM (T-105)
│   └── data/sapiens2_classes.json   # nama 29 kelas (data paket)
├── configs/
│   ├── default.yaml       # paths, style, ingest, isolate, segment, depth, qc, groups, stabilize, vectorize, export
│   └── styles/rough-sketch.yaml
├── assets/        # brushes/, paper/
├── scripts/       # smoke_test.py, alat sekali pakai (A/B, T-102c)
├── samples/       # video test (tidak di-commit)
├── work/          # intermediate (paths.work_dir), gitignored; klip di work/clips/<nama video>[@S-E]/, crop di work/crop/<nama>/
├── out/           # hasil (paths.out_dir), gitignored
└── tests/
```

## Urutan build (jangan lompat)

| Phase | Isi | Selesai kalau |
|---|---|---|
| 1 | config loader (T-104a) → segment + QC (T-102b) → depth (T-105) → stabilize spasial saja, temporal off (T-106) → export naif (T-103) → cli (T-104b) | Pipeline end-to-end jalan (siluet blok peta grup → MP4) — **tercapai** (T-104b, 2026-10-01; `python -m rotoscope run samples/test_short.mp4`) |
| 2 | vectorize: siluet + lubang + batas grup (T-201a), garis oklusi (T-201b), anchor + `track_id` (T-202) → stylize garis polos, satu renderer + parameter per tipe (T-203a) → integrasi `run` / export (T-203b) → `--preview` (T-204) | Sudah keluar outline |
| 3 | temporal pada probabilitas grup + kedalaman ternormalisasi (T-302, T-303) → `boil_preserve` (T-304) → kalibrasi ulang N/K/M/D/L/persentil/`min_hole_area` (T-305) | Flicker terkendali |
| 4 | style params lengkap + SVG export | Bisa ganti style dari config |
| 5 | fallback pose — SKIP (2026-10-27, T-501 / T-502 SKIP; docs/04 "Revisi D-010 Q3 + D-002") | — (milestone gugur) |
| 6 | (opsional) eksperimen SAM 2 tiny, batch processing | — |
