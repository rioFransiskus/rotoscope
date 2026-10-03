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
   ├─[4]  vectorize ──► contours/*.json + contours/manifest.json (+ clip_stats.json, T-201b) (polyline bertipe; di luar `run` sampai T-203)
   │
   ├─[5]  stylize ────► strokes/*.svg + strokes/*.png
   │
   └─[6]  export ─────► out/animation.mp4 + out/svg/*.svg
```

Path stage di bawah (`frames/`, `seg/`, `depth/`, `stable/`, …) relatif terhadap **folder kerja klip**
`<paths.work_dir>/clips/<nama video>/` (default `work/clips/<nama>/`; satu video = satu folder, lihat "CLI"),
dan `out/` = `paths.out_dir` (default `out/`, tidak per klip). `paths.work_dir` boleh diarahkan ke drive lain —
data antara ±1–5 GB per klip (lihat anggaran disk).

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

## CLI (T-104b, D-010)

`python -m rotoscope <subperintah>` (`src/rotoscope/cli.py` + `__main__.py`; tanpa entry point di `pyproject.toml`,
tanpa dependency baru). Modul stage tetap bisa dijalankan sendiri (`python -m rotoscope.segment …`) — itu entry point
internal yang dipanggil cli, bukan cara pakai utama.

| Subperintah | Fungsi |
|---|---|
| `run <video>` | ingest → segment → depth → stabilize → export |
| `ingest\|segment\|depth\|stabilize\|vectorize\|export <video> [flag stage]` | satu stage; sisa argumen diteruskan apa adanya ke `main(argv)` stage (flag stage tidak diparse ulang di cli). `vectorize` (T-201a, T-201b): flag `--restart --limit` (`--limit N` tetap membaca SEMUA frame untuk ambang per klip, `clip_stats.json`), CPU in-process, `--restart` tanpa `--yes`; **belum ada di `run`** (masuk bersama [5] di T-203) |
| `download [--config P] [--seg-model 0.8b\|0.4b]` | unduh checkpoint (online, sekali jalan): Sapiens2-seg (default **0.8b**; `--seg-model 0.4b` = fallback) **dan** Depth Anything V2 Small + model card. Subprocess mewarisi environment; **tidak** memaksa `HF_HUB_OFFLINE=1` (run biasa offline) |

- **Folder kerja per klip:** `<paths.work_dir>/clips/<stem>/`; `stem` = nama video disanitasi dengan fungsi yang sama
  dengan `{source}` di `export.filename` (`export.sanitize_source_name`). Layout `clips/` memisahkan data klip dari
  folder eksperimen di `work/` (`t102c`, `t106`, …) — tanpa daftar nama dicadangkan. `paths.out_dir` tidak berubah.
- **`--work-dir DIR` di setiap `main()` stage** (ingest sudah punya; segment, depth, stabilize, export sejak
  T-104b): menang atas `paths.work_dir` config (lewat `overrides`, T-104a — tanpa menulis YAML); `paths.out_dir` tidak
  berubah. cli selalu menambahkannya sendiri; `--work-dir` di baris perintah cli ditolak (exit 2).
- **Proses:** stage GPU ([2], [2c]) = **subprocess** `[sys.executable, -m, rotoscope.segment|depth, …]`, stdout/stderr
  diwariskan (pesan VRAM tetap terbaca), baik lewat `run` maupun subperintah sendiri. Stage CPU (ingest, stabilize,
  export) = in-process lewat `main(argv)`; `SystemExit` dari argparse stage ditangkap dan menyebut nama stage. Proses
  induk **tidak meng-import torch** / menyentuh CUDA (diuji di subprocess bersih); cek VRAM tetap tugas stage.
  Induk menerima `KeyboardInterrupt` / exception saat stage GPU jalan → anak di-`terminate` (lalu `kill` bila macet)
  dan ditunggu selesai; pesan "dihentikan; jalankan ulang untuk resume".
- **Pre-flight CPU-only** (`run`; ingest sendiri hanya (a)+(b)) — SEBELUM penghapusan, ingest, atau subprocess apa pun:
  (a) file video ada; (b) folder kerja tidak berisi klip lain (`meta.json` `source_path` ≠ video, dibandingkan dengan
  `Path.resolve` yang sama dengan ingest, bukan string mentah); (c) target export `out/<nama>.mp4` tidak milik video
  lain (`<nama>.export.json` `clip.source_path`; fungsi nama export yang sama; dilewati bila `--limit`). Alasan (c):
  pengaman export tidak bisa dilewati `--restart`, jadi tanpa ini klip bernama sama baru gagal di stage terakhir
  setelah ±78 mnt GPU. Pesan bentrok TIDAK menyarankan `--restart`: ganti nama salah satu video (atau ubah
  `export.filename`).
- **Ingest di `run` = selalu ulang** (opsi A). `meta.json` byte-deterministik → klip yang sama lolos identitas T-108;
  video lain yang menimpa path yang sama menulis ulang `meta.json` → identitas berubah → [2]/[2c] menolak (exit 1).
  (Melewati ingest kalau `meta.json` cocok akan melebarkan batas (c) "Identitas klip": video apa pun di path yang sama
  tidak terdeteksi.) Ingest hanya detik.
- **Flag `run`:** `--config`, `--seg-model 0.8b|0.4b` (hanya segment; **tidak pernah fallback otomatis**, D-009),
  `--limit N` (segment, depth, stabilize, export; export → `<nama>.limitN.mp4`), `--restart-from`, `--yes`.
  `--adopt` / `--qc-only` hanya di subperintah `segment` / `depth` (`--adopt` hanya keduanya); di `run` → exit 1.
  Kombinasi terlarang stage ([2]/[2c] `--adopt` + `--restart`/`--limit`/`--qc-only`/`--download`) tetap exit 1.
- **Restart mengikuti graf dependensi** ingest → {segment, depth} → stabilize → export (depth **tidak** bergantung pada
  segment). `run --restart-from X` menjalankan stage di bawah ini dengan `--restart`; stage sebelum X berjalan normal
  (resume), hilir CPU yang basi dihitung ulang otomatis oleh stage-nya (cli tidak menghapusnya):

  | `--restart-from` | stage ber-`--restart` | butuh `--yes` |
  |---|---|---|
  | `ingest` | segment, depth, stabilize, export (semua) | ya |
  | `segment` | hanya [2] (`seg/` + `qc_report.json`); `depth/` utuh | ya |
  | `depth` | hanya [2c] (`depth/`) | ya |
  | `stabilize` / `export` | hanya stage itu | tidak |

  Graf lengkap sejak T-201a: ingest → {segment, depth} → stabilize → {vectorize, export}. `--restart-from` **tidak**
  mencakup `vectorize` (stage itu belum ada di `run`); `python -m rotoscope vectorize <video> --restart` menghapus
  `contours/` saja, dan `stabilize --restart` (atau [3] dihitung ulang) membuat [4] basi → dihitung ulang otomatis oleh
  stage-nya. T-203 memasukkan [4] + [5] ke urutan `run` dan baris ini ke tabel di atas.

- **`--yes`:** wajib untuk SETIAP penghapusan hasil GPU lewat cli — `run --restart-from ingest|segment|depth` dan
  subperintah `segment|depth --restart`. cli membuang `--yes` sebelum meneruskan ke stage. Tanpa `--yes`: cetak apa
  yang akan dihapus (jumlah file, MiB, estimasi waktu GPU: 0.8b 16.5 s/frame, 0.4b 8 s/frame, depth 0.19 s/frame —
  konstanta pesan di `cli.py`, bukan parameter), exit 1, tidak ada yang terhapus. Pesan error stage yang menyuruh
  `--restart` memakai perintah CLI lengkap (`python -m rotoscope segment "<video>" --restart --yes`; `<video>` dari
  `meta.json`). Setelah `run` yang memakai `--restart-from` gagal, pesan menyebut resume = jalankan ulang TANPA
  `--restart-from`/`--yes` (kalau tidak, hasil dihapus lagi).
- **Frame gagal QC:** setelah [2] satu peringatan `N/283 frame gagal QC (indeks: …, maks 10 ditampilkan)`; run lanjut
  (stage [3] memberi bobot, D-010), exit 0. Tanpa parameter YAML baru.
- **Exit code `run`:** 0 sukses (atau semua dilewati) | 1 prasyarat gagal (config, pre-flight, VRAM, identitas, flag
  terlarang, `--limit` < 1) | 2 salah pakai argumen (argparse; juga argparse stage in-process) | 3 OOM di segment/depth |
  130 dihentikan (Ctrl+C). Kode stage yang gagal dikembalikan apa adanya; pesan menyebut stage + cara resume.

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
- **CLI:** `python -m rotoscope ingest <video> [--target-fps N] [--working-width N]` (folder kerja =
  `<paths.work_dir>/clips/<nama>/`; pre-flight (a)+(b) di bagian "CLI"). Di `run` ingest **selalu** dijalankan ulang
  (`meta.json` byte-deterministik, jadi klip yang sama lolos identitas T-108). Ingest tidak membaca config (tanpa
  `--config`); entry point modul: `python -m rotoscope.ingest <video> [--work-dir DIR]`. Exit code 0 / 1.

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
- **CLI** (T-104b): `python -m rotoscope depth <video> [--config PATH] [--restart --yes] [--limit N] [--adopt]`
  (unduh: `python -m rotoscope download`); di dalam `run` = subprocess `python -m rotoscope.depth --work-dir
  <folder klip> …`. Modul menerima `--work-dir DIR` (menang atas `paths.work_dir`) dan `--download`; `--yes` hanya
  ada di cli (`--restart` tanpa `--yes` → exit 1, tidak ada yang dihapus). Exit code sama dengan [2] (0 / 1 / 3).
  `--adopt`: lihat [2] (0.70 s pada klip uji).

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
- **CLI** (T-104b): `python -m rotoscope stabilize <video> [--config PATH] [--restart] [--limit N]`; di dalam `run`
  dipanggil in-process `stabilize.main([--work-dir <folder klip>, …])` (`--work-dir DIR` menang atas
  `paths.work_dir`). `--restart` stage CPU tidak butuh `--yes`. Exit code sama dengan [2]: 0 sukses, 1 prasyarat
  gagal (3 tidak dipakai — tanpa GPU).

### [4] `vectorize.py` (CPU)

**Status:** **T-201a `DONE`** (siluet + lubang + batas grup, 2026-10-02), **T-201b `DONE`** (garis oklusi +
`clip_stats.json`, 2026-10-03, **dengan batas kaki** — lihat langkah 3) dan **T-202 `DONE`** (orientasi, anchor, arah garis
terbuka, `track_id`, rantai kesinambungan, 2026-10-03) — strok `silhouette`, `silhouette_hole`, `group_boundary`,
`occlusion`. Subperintah sendiri `python -m rotoscope vectorize <video>`; **belum masuk urutan `run`** sampai T-203 (lihat
"CLI" dan [5]). Bagian bertanda *(T-201a)* / *(T-201b)* / *(T-202)* sudah diimplementasi dan diukur. Kode pelacakan:
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
penanganan (sembunyikan / pudarkan / gambar) = **T-203, wajib sebelum implementasi** (lihat [5]). `group_boundary` tidak
butuh padding: garis yang sampai tepi berakhir di tepi (terbuka, sah).

**Manifest [4]** (`contours/manifest.json`, prinsip #4): `stage`, `contract` (`"T-202"`; manifest `"T-201a"` / `"T-201b"`
lama basi otomatis), `algo_rev`, `stroke_types` (4 tipe, termasuk `occlusion`), `pending` (`[]` sejak T-202),
`vectorize` (dict **datar** 11 kunci: 4 parameter T-201a
`min_region_area`, `min_hole_area`, `line_min_px`, `min_stroke_px` + 6 `depth_lines.blur_sigma`, `depth_lines.hi_pct`,
`depth_lines.lo_pct`, `depth_lines.erode_px`, `depth_lines.min_dist_px`, `depth_lines.min_len_px` + `track.max_match_dist_px`)
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
- `track_id` dipakai [5] untuk seed jitter. Resample ke N titik tetap dilakukan di [5], mulai dari `points[0]`.
- ⚠️ **Titik awal garis terbuka tidak stabil secara posisi:** terukur (Y@16, track ≥ 5 frame) titik awal melompat sampai
  60–127 px antar frame saat strok memanjang / memendek (aturan statis juga: 60,9 / 127 px), median 3–4,5 px, p95 18–23 px;
  aturan statis memilih ujung berbeda dari kesinambungan di 7–13% strok-frame. Jitter 1D berbasis panjang busur dari
  `points[0]` akan "pop" (lihat [5]).
- **Batas yang diketahui:** (1) padanan di zona 12–16 px (±1–3% padanan; kebanyakan track `group_boundary` yang bentuknya
  berubah) bukan bukti identitas tertukar; 4 dari ±3500 padanan (kasus (b)) bergeser oleh penugasan optimal padahal ada
  pendahulu lebih dekat; (2) id oklusi kaki berkedip lahir-mati (temporal [3] belum aktif, T-302/T-303/T-305); (3) fragmen
  lengan terpisah (klip `test`) mendapat id baru (2 dari 12 strok-frame mewarisi id fragmen sebelumnya), wajar sementara.

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
    reproducible (P-007), dan tidak melompat saat urutan stroke berubah. ⚠️ `points[0]` garis terbuka melompat sampai 60–127 px
    saat strok memanjang / memendek ([4] "Aturan anchor"): noise 1D berbasis panjang busur dari titik awal akan "pop";
    pertimbangkan jitter terkunci posisi atau tidak bergantung titik awal. [5] membaca `points[0]` (bukan `anchor`) dan
    mengabaikan `prev_sha256`
  - Multipass (offset + opacity falloff), supersampling `render.ss` untuk anti-alias
  - Tekstur: brush stamping (raster) atau multi-stroke offset (SVG); paper background layer
- **Lib:** `svgwrite` untuk SVG, OpenCV / `Pillow` untuk raster
- ⚠️ `output_width` (resolusi output terpisah dari resolusi kerja, satuan tebal/jitter relatif) →
  diputuskan sebelum T-203.
- ⚠️ **Run titik di tepi frame** (keputusan tertunda dari T-201a): foreground menyentuh tepi bawah di **283/283** frame
  `test` dan **119/119** `test_short` (tepi kanan 59/283). Kontur `silhouette` memuat run titik tepat di baris/kolom tepi
  (y = H − 0.5, x = 0.5, x = W − 0.5; lihat [4] "Kasus tepi"). T-203 **WAJIB memutuskan** penanganannya
  (sembunyikan / pudarkan / gambar) **SEBELUM implementasi**; konvensi koordinat pusat piksel + skala x' = s · x ([4])
  berlaku.
- ⚠️ **`group_boundary` loop** = `closed: false` dengan titik akhir = titik awal (57 loop di `test`, 25 di `test_short`;
  terutama `hair|face` dan `torso|left_arm`). Taper ujung di [5] **jangan menipiskan sambungan** loop: kenali loop dari
  titik akhir = titik awal dan perlakukan sebagai tertutup untuk taper. Berlaku juga untuk loop `occlusion` (4 di
  `test`, 5 di `test_short`).
- ⚠️ **Strok `occlusion` (T-201b):** (1) `strength` (rata-rata |grad|, 3 desimal) tersedia untuk memodulasi tebal / opacity
  tetapi skalanya per klip (bergantung normalisasi [3], akan berubah di T-302): jangan dipakai sebagai ambang absolut;
  (2) **temporal [3] belum aktif** (T-302/T-303), jadi garis oklusi antar frame masih berkedip (muncul / hilang, bergeser):
  frame tanpa `occlusion` valid (34/119 dan 160/283 di default, run kosong terpanjang 28 frame) — [5] jangan mengira itu
  error dan jangan menginterpolasi antar frame; (3) garis oklusi tidak pernah dalam ≥ D px dari siluet / batas grup /
  tepi frame (ujungnya terpotong di D, jadi tidak menyambung ke garis siluet atau batas grup); (4) `track_id` (T-202) tersedia
  untuk seed jitter, tetapi id oklusi berkedip lahir-mati (umur median 1–2 frame; 37–51% strok-frame di track ≥ 5 frame).
- ⚠️ **Urutan `run`:** T-203 memasukkan [4] dan [5] ke urutan `run` (`ingest → segment → depth → stabilize → vectorize →
  stylize → export`), ke tabel restart DAG (bagian "CLI"), dan menyalakan `export.source: "strokes"` (sekarang ditolak).

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
- **CLI** (T-104b): `python -m rotoscope export <video> [--config PATH] [--restart] [--limit N]`; di dalam `run`
  dipanggil in-process `export.main([--work-dir <folder klip>, …])` (`--work-dir DIR` menang atas `paths.work_dir`;
  `paths.out_dir` tidak berubah). `--limit N` → `<nama>.limitN.mp4` (preview; tanpa manifest, tanpa pengaman, tidak
  menyentuh hasil utama). Exit code: 0 sukses, 1 prasyarat gagal (3 tidak dipakai — tanpa GPU). Bentrok target export
  dengan video lain dicek lebih awal oleh pre-flight (c) `run` (bagian "CLI").
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
| [4] T-201a | CPU | – | ±11 s (29–30 ms/frame rata-rata, p95 34–35 ms, maks 50 ms; 283 frame 8.5 s) |
| [4] T-202 (+ oklusi + pelacakan) | CPU | – | median 74–93 ms/frame antar run (p95 84–112 ms, maks 88–135 ms, 0 frame > 1 s, target ≤ 150 ms; pelacakan sendiri median 8–11 ms, maks 19–26 ms), pass 2 saja; pass 1 ambang (baca ulang + gradien semua frame) 1,3 s (119 frame) / 2,7 s (283 frame) |
| [5] | CPU | – | belum diukur |

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
│   ├── stabilize.py  vectorize.py  stylize.py  export.py
│   ├── config.py  cli.py  __main__.py   # python -m rotoscope (T-104b)
│   ├── stage_common.py   # helper bersama stage: tulis atomik + retry, frames.jsonl, daftar frame,
│   │                     # error/exit code; khusus GPU: offline/revision/cache HF, VRAM, OOM (T-105)
│   └── data/sapiens2_classes.json   # nama 29 kelas (data paket)
├── configs/
│   ├── default.yaml       # paths, segment, depth, qc, groups, stabilize, vectorize
│   └── styles/rough-sketch.yaml
├── assets/        # brushes/, paper/
├── scripts/       # smoke_test.py, alat sekali pakai (A/B, T-102c)
├── samples/       # video test (tidak di-commit)
├── work/          # intermediate (paths.work_dir), gitignored; klip di work/clips/<nama video>/
├── out/           # hasil (paths.out_dir), gitignored
└── tests/
```

## Urutan build (jangan lompat)

| Phase | Isi | Selesai kalau |
|---|---|---|
| 1 | config loader (T-104a) → segment + QC (T-102b) → depth (T-105) → stabilize spasial saja, temporal off (T-106) → export naif (T-103) → cli (T-104b) | Pipeline end-to-end jalan (siluet blok peta grup → MP4) — **tercapai** (T-104b, 2026-10-01; `python -m rotoscope run samples/test_short.mp4`) |
| 2 | vectorize: siluet + lubang + batas grup (T-201a), garis oklusi (T-201b), anchor + `track_id` (T-202) → stylize basic, satu renderer + parameter per tipe (T-203) → `--preview` (T-204) | Sudah keluar outline |
| 3 | temporal pada probabilitas grup + kedalaman ternormalisasi (T-302, T-303) → `boil_preserve` (T-304) → kalibrasi ulang N/K/M/D/L/persentil/`min_hole_area` (T-305) | Flicker terkendali |
| 4 | style params lengkap + SVG export | Bisa ganti style dari config |
| 5 | fallback pose — DITUNDA (`BLOCKED`, D-010) | — |
| 6 | (opsional) eksperimen SAM 2 tiny, batch processing | — |
