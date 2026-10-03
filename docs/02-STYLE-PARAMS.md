# 02 — STYLE & PIPELINE PARAMETERS

Semua parameter di bawah **wajib exposed lewat YAML**, tidak boleh hardcoded.
Ini requirement inti project: kontrol tekstur dan style garis.

Dua jenis file (D-010):
- `configs/default.yaml` — parameter **pipeline** (stage [2]–[4] + export [6]): paths, segment, depth, qc,
  groups, stabilize, vectorize, export. Default = hasil T-102c (D-009).
- `configs/styles/*.yaml` — parameter **style** (stage [5]). Ganti style = jalankan ulang [5] saja.

Nilai bertanda **sementara** belum dikalibrasi; task kalibrasinya disebut di kolom catatan.

---

## `configs/default.yaml` — pipeline

```yaml
# ── PATHS ──────────────────────────────────────────
paths:
  # Relatif → terhadap direktori kerja (cwd) saat load. Path Windows: pakai "/" atau kutip TUNGGAL.
  #   work_dir: "D:/rotoscope/work"   → OK
  #   work_dir: 'D:\rotoscope\work'   → OK
  #   work_dir: "D:\rotoscope\new"    → ERROR: di kutip GANDA backslash = escape ("\r", "\n" = kontrol)
  work_dir: "work"           # boleh di drive lain (±1–5 GB per klip)
  out_dir: "out"

# ── SEGMENT [2] — Sapiens2-seg ─────────────────────
segment:
  model: "0.8b"              # "0.8b" | "0.4b" — eksplisit, untuk SELURUH klip (D-009)
  model_ids:
    "0.8b": "facebook/sapiens2-seg-0.8b"   # id HF yang dipakai di T-102c
    "0.4b": "facebook/sapiens2-seg-0.4b"
  revision:                  # commit hash snapshot HF per model (pin T-102b); wajib ada di cache HF
    "0.8b": "196a627b928676c4429b738ed76f78a21d96c4eb"
    "0.4b": "449b3c5335e6722bb94990abdd1aa6e612432f22"
  precision: "fp16"          # "fp16" | "fp32" — TANPA bf16 (P-005)
  vram_min_free_mib:
    "0.8b": 3300             # peak reserved 3276 MiB (T-102c)
    "0.4b": 2300             # peak reserved 2258 MiB
  probs_dtype: "uint8"       # round(p × 255), npz deflate

# ── DEPTH [2c] — Depth Anything V2 Small ───────────
depth:
  model_id: "depth-anything/Depth-Anything-V2-Small-hf"   # HANYA Small (Apache-2.0)
  revision: "5426e4f0f36572d16453bbda7a8389317b1bef99"   # commit hash snapshot HF (pin T-105); wajib ada di cache HF
  precision: "fp32"
  vram_min_free_mib: 500     # peak reserved 424 MiB

# ── QC (akhir stage [2]) ───────────────────────────
qc:
  area_min: 0.03             # fraksi frame
  area_max: 0.70
  iou_min: 0.55              # IoU foreground vs frame sebelumnya
  blob_min: 0.05             # komponen "besar" = > 5% area frame
  max_big_blobs: 1
  area_median_window: 49     # frame (±2 s); jendela DIGESER, bukan median klip. Jangan diubah tanpa kalibrasi
                             # ulang area_drop_min (W=25/49/73 memberi celah berbeda; W=73 tanpa celah)
  area_drop_min: 0.63        # SEMENTARA — T-102b, satu klip: celah aman 0.608 (MediaPipe kaki hilang) – 0.652 (Sapiens2)

# ── GROUPS (dipakai [3] dan [4]) ───────────────────
# Urutan = id grup (1..G); 0 = background (dicadangkan). Batas di dalam satu grup tidak digambar.
groups:
  hair: [Hair]
  face: [Face_Neck, Eyeglass, Lower_Lip, Upper_Lip, Lower_Teeth, Upper_Teeth, Tongue]
  torso: [Torso, Upper_Clothing, Apparel, Lower_Clothing]
  left_arm: [Left_Upper_Arm, Left_Lower_Arm, Left_Hand]
  right_arm: [Right_Upper_Arm, Right_Lower_Arm, Right_Hand]
  left_leg: [Left_Upper_Leg, Left_Lower_Leg, Left_Foot, Left_Shoe, Left_Sock]
  right_leg: [Right_Upper_Leg, Right_Lower_Leg, Right_Foot, Right_Shoe, Right_Sock]

# ── STABILIZE [3] ──────────────────────────────────
stabilize:
  temporal:
    enabled: false           # Phase 1–2: false; dinyalakan di T-302
    mask_ema_alpha: 0.7      # 1.0 = tanpa smoothing. Turun = lebih stabil, lebih lag
    optical_flow_blend: 0.4  # bobot probabilitas hasil warp
    boil_preserve: 0.3       # 0 = mati total, 1 = boiling penuh
    qc_fail_weight: 0.25     # SEMENTARA — bobot frame gagal QC di EMA, dikalibrasi di T-302
  island_min_px: 30          # N — filter pulau pada peta GRUP (T-102c: pada peta kelas → T-305)
  mode_k: 3                  # K — mode filter peta grup, background ikut
  depth:
    normalize: "log_median_iqr"  # SEMENTARA — metode dipilih + diuji di T-302
    log_eps: 1.0e-6          # log(max(d, eps)) — hanya menahan disparity ≤ 0 (min klip uji 0.64)
    iqr_min: 0.01            # SEMENTARA — batas bawah pembagi IQR (log) foreground, dikalibrasi di T-302
    temporal: true           # hanya berlaku kalau temporal.enabled

# ── VECTORIZE [4] ──────────────────────────────────
vectorize:
  min_region_area: 800       # px² — kontur luar foreground lebih kecil dibuang
  min_hole_area: 200         # px² — SEMENTARA, lubang siluet lebih kecil dibuang; dikalibrasi di T-305
  line_min_px: 5             # M — komponen garis 8-arah < M px dibuang, SEBELUM thinning
  min_stroke_px: 6           # jalur hasil tracing < 6 px dibuang
  depth_lines:
    blur_sigma: 1.0          # Gaussian σ (px) sebelum Sobel
    hi_pct: 95               # T_high = persentil per klip |grad depth_smooth|
    lo_pct: 90               # T_low  = persentil per klip
    erode_px: 5              # kernel erosi foreground untuk wilayah persentil
    min_dist_px: 7           # D — jarak minimum ke batas grup (termasuk siluet)
    min_len_px: 30           # L — panjang skeleton minimum
  track:
    max_match_dist_px: 16    # jarak Chamfer simetris maks pencocokan track_id (T-202; 12 → 16 disetujui Rio)

# ── EXPORT [6] ─────────────────────────────────────
export:
  source: "silhouette"       # "silhouette" (stable/groups, Phase 1) | "strokes" (stage [5], Phase 2 — belum ada)
  crf: 18                    # libx264, 0 (lossless) – 51; makin kecil makin tajam + besar
  preset: "medium"           # preset x264: ultrafast … veryslow
  foreground_color: "#000000"   # siluet (hanya source "silhouette")
  background_color: "#ffffff"   # latar kertas; juga warna pad 1 px kalau dimensi ganjil
  audio: false               # true = audio video sumber (aac, -shortest); error kalau sumber tanpa audio.
                             # Audio meme biasanya milik pihak ketiga → default tanpa audio; tambahkan audio
                             # berlisensi di editor platform (TikTok/CapCut)
  filename: "{source}.mp4"   # di paths.out_dir; {source} = nama video sumber (disanitasi). Hasil klip lain tidak ditimpa
```

### Validasi `default.yaml` (saat load, error jelas kalau gagal)

| parameter | range / aturan |
|---|---|
| `paths.work_dir`, `paths.out_dir` | string tidak kosong, tanpa karakter kontrol; relatif → di-resolve terhadap direktori kerja (cwd) saat load; absolut (mis. `D:/…`) boleh. Loader **tidak** membuat folder (tanpa side effect) — stage memanggil `ensure_dir()` |
| `segment.model` | `"0.8b"` atau `"0.4b"` |
| `segment.model_ids`, `segment.revision`, `segment.vram_min_free_mib` | key persis {`0.8b`, `0.4b`} |
| `segment.model_ids.*` | wajib diawali `facebook/sapiens2-seg-` — Sapiens v1 (`facebook/sapiens-seg-…`, CC-BY-NC) ditolak |
| `segment.revision.*`, `depth.revision` | `null` atau string tidak kosong. Stage [2]/[2c] saat runtime menolak `null` dan nilai selain commit hash 40-hex (T-102b/T-105) |
| `segment.precision` | `"fp16"` atau `"fp32"` (`bf16` ditolak, P-005) |
| `segment.vram_min_free_mib.*`, `depth.vram_min_free_mib` | int 0–4096 |
| `segment.probs_dtype` | `"uint8"` |
| `depth.model_id` | wajib mengandung `Small` — Base/Large (CC-BY-NC) ditolak |
| `depth.precision` | `"fp16"` atau `"fp32"` |
| `qc.area_min`, `qc.area_max` | 0 ≤ min < max ≤ 1 |
| `qc.iou_min`, `qc.blob_min`, `qc.area_drop_min` | 0–1 |
| `qc.max_big_blobs` | int ≥ 1 |
| `qc.area_median_window` | int ganjil ≥ 3 |
| `groups` | lihat aturan grup di bawah |
| `stabilize.temporal.mask_ema_alpha` | (0, 1] |
| `stabilize.temporal.optical_flow_blend`, `boil_preserve`, `qc_fail_weight` | 0–1 |
| `stabilize.island_min_px` | int ≥ 0 (0 = mati) |
| `stabilize.mode_k` | int ganjil ≥ 1 (1 = mati) |
| `stabilize.temporal.enabled`, `stabilize.depth.temporal` | bool |
| `stabilize.depth.normalize` | nilai dari daftar metode yang diimplementasi — sekarang hanya `"log_median_iqr"`; diperluas di T-302 (kandidat: affine) |
| `stabilize.depth.log_eps` | 1e-12 ≤ x ≤ 0.01 |
| `stabilize.depth.iqr_min` | 1e-3 ≤ x ≤ 1 (dengan batas `log_eps`, `depth_smooth` dijamin muat float16) |
| `vectorize.min_region_area`, `min_hole_area` | int ≥ 0 |
| `vectorize.line_min_px`, `min_stroke_px` | int ≥ 1 |
| `vectorize.depth_lines.blur_sigma` | ≥ 0 (0 = tanpa blur) |
| `vectorize.depth_lines.lo_pct`, `hi_pct` | 0 < lo_pct < hi_pct < 100 |
| `vectorize.depth_lines.erode_px` | int ganjil ≥ 1 |
| `vectorize.depth_lines.min_dist_px`, `min_len_px` | ≥ 0 |
| `vectorize.track.max_match_dist_px` | > 0 (default 16 px pada frame 480 px; mengubahnya membuat `contours/` basi) |
| `export.source` | `"silhouette"` atau `"strokes"` (`strokes` diterima loader, tetapi stage [6] menolak sampai Phase 2) |
| `export.crf` | int 0–51 |
| `export.preset` | preset x264: `ultrafast`, `superfast`, `veryfast`, `faster`, `fast`, `medium`, `slow`, `slower`, `veryslow` |
| `export.foreground_color`, `export.background_color` | `#rrggbb`; keduanya harus berbeda (tanpa membedakan huruf besar/kecil) |
| `export.audio` | bool |
| `export.filename` | berakhiran `.mp4`; hanya nama file (tanpa `<>:"/\|?*`, tanpa awalan `.`); satu-satunya placeholder `{source}` |

**Aturan grup:**
1. Setiap nama kelas ada di daftar 29 kelas Sapiens2 (`src/rotoscope/data/sapiens2_classes.json`)
2. Setiap kelas ≠ `Background` masuk **tepat satu** grup. Kelas yang tidak tercantum = **error**
   (bukan diam-diam jadi background)
3. `Background` tidak boleh dicantumkan; nama grup `background` dicadangkan (id 0)
4. Nama grup unik, identifier (huruf kecil, angka, `_`); jumlah grup ≤ 255 (id uint8)
5. Hash definisi grup dicatat di `stable/manifest.json`; berubah → output [3]/[4] basi, [2] tidak

---

## `configs/styles/rough-sketch.yaml` — style

```yaml
# SATUAN PANJANG (tebal, epsilon, offset, jarak, frekuensi per px) = px REFERENSI lebar 1080; stage [5] mengalikannya
# dengan unit = render.output_width / 1080 (T-203a). Nilai = nilai look test (px kerja 480 px) × 2.25.

# ── SHAPE ──────────────────────────────────────────
shape:
  simplify_epsilon: 2.8      # cv2.approxPolyDP (px ref). Naik = lebih sedikit titik, lebih kasar (keputusan Rio: 5.6 membuang bentuk)
  smooth_px: 5.0             # penghalusan Gaussian arc-length SEBELUM approxPolyDP (px ref); 0 = mati. Sudut tajam + ujung dijaga
  resample_points: 200       # jumlah titik tetap per stroke (Phase 4; belum dipakai render T-203a)
  smooth_tension: 0.5        # Catmull-Rom tension. 0 = tajam, 0.5 = Catmull-Rom standar
  spline_steps: 8            # titik spline minimum per segmen (otomatis lebih rapat bila galat akor > 0.03 px ref)
  edge_mode: "hide"          # garis di tepi frame: "hide" = disembunyikan (tubuh terpotong frame) | "draw" = digambar
# (min_contour_area pindah ke vectorize.min_region_area di default.yaml — D-010)

# ── STROKE ─────────────────────────────────────────
stroke:
  width_base: 9.0            # tebal dasar garis (px ref); keputusan Rio (visual), setara look test = 7.2
  width_variation: 0.45      # 0 = seragam, 1 = variasi ekstrem
  width_noise_scale: 0.036   # frekuensi perubahan tebal sepanjang path (per px ref)
  color: "#1a1a1a"
  opacity: 0.92
  cap: "round"
  taper_ends: true           # ujung garis menipis
  taper_px: 45               # panjang zona taper di tiap ujung (px ref)
  taper_min: 0.15            # tebal di ujung sebagai fraksi tebal normal
  by_type:                   # override per jenis garis (skema [4]); 1.0 = sama dengan dasar
    silhouette:      {width_scale: 1.0, opacity_scale: 1.0}
    silhouette_hole: {width_scale: 1.0, opacity_scale: 1.0}
    group_boundary:  {width_scale: 1.0, opacity_scale: 1.0}
    occlusion:       {width_scale: 1.0, opacity_scale: 1.0}

# ── JITTER (hand-drawn feel) ───────────────────────
jitter:
  amplitude: 4.0             # pergeseran titik (px ref). 0 = garis mekanis
  frequency: 0.053           # skala noise (per px ref). Kecil = gelombang panjang
  temporal_seed_mode: "frame"  # "frame" = getar tiap frame | "fixed" = diam
  temporal_drift: 0.35       # seberapa cepat pola jitter berubah antar frame
  param_seed: 0              # seed = hash(frame_index, param_seed, track_id) — P-007

# ── MULTI-PASS (kesan sketsa ditimpa) ──────────────
multipass:
  enabled: true
  passes: 2                  # jumlah garis tumpang tindih
  offset: 2.7                # jarak antar pass (px ref)
  opacity_falloff: 0.55      # pass ke-2 lebih pudar

# ── TEXTURE ────────────────────────────────────────
texture:
  mode: "brush_stamp"        # "none" | "brush_stamp" | "grain_overlay"
  brush_image: "assets/brushes/pencil_01.png"
  stamp_spacing: 0.35        # rasio terhadap lebar brush
  pressure_noise: 0.25
  grain_strength: 0.18

# ── PAPER ──────────────────────────────────────────
paper:
  enabled: true
  color: "#f4f1ea"
  texture_image: "assets/paper/rough_01.jpg"
  texture_opacity: 0.35
  vignette: 0.12

# ── RENDER ─────────────────────────────────────────
render:
  ss: 3                      # faktor supersampling raster (anti-alias)
  output_width: 1080         # lebar output (px, genap, 256–2160); tinggi mengikuti rasio klip, dinaikkan ke genap
```

Bagian `temporal:` lama (mask_ema_alpha, optical_flow_blend, boil_preserve) **pindah** ke
`stabilize.temporal` di `default.yaml` — dibaca stage [3], bukan [5] (D-010).

### Validasi style (saat load)

| parameter | range / aturan |
|---|---|
| `shape.simplify_epsilon` | ≥ 0 |
| `shape.resample_points` | int ≥ 4 |
| `shape.smooth_tension` | 0–1 |
| `shape.spline_steps` | int ≥ 1 |
| `shape.smooth_px` | ≥ 0 (0 = penghalusan mati) |
| `shape.edge_mode` | `"hide"` \| `"draw"` |
| `stroke.width_base` | > 0 |
| `stroke.width_variation`, `stroke.opacity`, `stroke.taper_min` | 0–1 |
| `stroke.width_noise_scale` | ≥ 0 |
| `stroke.color`, `paper.color` | hex `#rrggbb` |
| `stroke.cap` | `"round"` \| `"butt"` \| `"square"` |
| `stroke.taper_px` | ≥ 0 |
| `stroke.by_type` | kunci ⊆ {`silhouette`, `silhouette_hole`, `group_boundary`, `occlusion`}; `width_scale` > 0, `opacity_scale` 0–1 |
| `jitter.amplitude`, `jitter.frequency` | ≥ 0 |
| `jitter.temporal_seed_mode` | `"frame"` \| `"fixed"` |
| `jitter.temporal_drift` | 0–1 |
| `jitter.param_seed` | int |
| `multipass.passes` | int ≥ 1 |
| `multipass.offset` | ≥ 0 |
| `multipass.opacity_falloff` | 0–1 |
| `texture.mode` | `"none"` \| `"brush_stamp"` \| `"grain_overlay"` |
| `stroke.taper_ends`, `multipass.enabled`, `paper.enabled` | bool |
| `texture.brush_image`, `paper.texture_image` | path relatif → di-resolve terhadap **root project** (folder berisi `pyproject.toml`, dicari dari lokasi paket — editable install), **bukan** cwd; absolut boleh; root tidak ditemukan → error. File wajib ada hanya kalau dipakai (`texture.mode == "brush_stamp"` / `paper.enabled`). Aturan ini khusus aset input — `paths.*` tetap relatif terhadap cwd |
| `texture.stamp_spacing` | > 0 |
| `texture.pressure_noise`, `texture.grain_strength`, `paper.texture_opacity`, `paper.vignette` | 0–1 |
| `render.ss` | int 1–8 |
| `render.output_width` | int genap 256–2160 |

### Satuan dan parameter aktif (T-203a)

- **Satuan:** semua parameter panjang di tabel di atas adalah **px referensi lebar 1080**. Stage [5] memakai `unit =
  render.output_width / 1080`; nilai efektif di px output = nilai × `unit`. Mengganti `output_width` tidak mengubah tampilan.
  Tinggi output = `round(height × output_width / width)` dinaikkan ke genap (integer: `(2·H·ow + W) // (2·W)`, +1 bila
  ganjil; 854 → 1922 untuk lebar 1080). Skala titik `s = output_width / width` (sama untuk x dan y).
- **Konversi dari look test** (`scripts/look_test.py` menggambar di resolusi KERJA 480 px): × 2.25 (= 1080 / 480) — `width_base`
  3.2 → 7.2, `jitter.amplitude` 1.8 → 4.0, `multipass.offset` 1.2 → 2.7, `taper_px` 20 → 45, `width_noise_scale` 0.08 → 0.036 dan
  `jitter.frequency` 0.12 → 0.053 (per px, dibagi 2.25). **Pengecualian:** `simplify_epsilon` setara look test = 5.6, tetapi
  penilaian visual Rio memutuskan **2.8** (5.6 membuang bentuk) dipadukan dengan `smooth_px` (3.0 masih bergelombang → **5.0**).
  `width_base` setara look test = 7.2, tetapi penilaian visual Rio memilih **9.0**.
- **`shape.smooth_px` (keputusan Rio, perbandingan visual 2026-10-03):** Gaussian arc-length sebelum approxPolyDP, sigma = nilai
  ini (px ref × `unit`), 0 = mati. Kontur dire-sample tiap 1 px ref; sudut tajam (belok > 60° pada jendela ±6 sampel) dikunci
  dan memecah jalur; ujung strok terbuka (termasuk titik silang tepi) tidak bergeser; strok tertutup dan loop dihaluskan periodik
  tanpa takik. Tanpa penyusutan bentuk yang berarti (luas silhouette < 0,01% rata-rata).
- **Aktif di T-203a:** `shape.simplify_epsilon`, `shape.smooth_px`, `shape.smooth_tension`, `shape.spline_steps`, `shape.edge_mode`, `stroke.width_base`,
  `stroke.color`, `stroke.cap` (hanya `"round"`; nilai lain → error stage), `stroke.by_type.*.width_scale`, `paper.color`,
  `render.ss`, `render.output_width`. Hanya parameter ini yang masuk hash style stage [5].
- **Divalidasi tetapi BELUM aktif (dicatat di `strokes/manifest.json` → `ignored_params`; aktif di Phase 4):** `shape.resample_points`
  (render tidak me-resample; resample arc-length dari `points[0]` dibutuhkan Phase 4, T-401 / T-402), `stroke.width_variation`,
  `stroke.width_noise_scale`, `stroke.opacity`, `stroke.taper_*`, `stroke.by_type.*.opacity_scale`, `jitter.*`, `multipass.*`,
  `texture.*`, `paper.enabled`, `paper.texture_*`, `paper.vignette`.
- **`shape.smooth_tension`:** Catmull-Rom seragam (uniform); tangen di titik i = `smooth_tension × (p[i+1] − p[i−1])`
  (0.5 = Catmull-Rom standar; 0 = segmen lurus berkecepatan tidak seragam). Ujung strok terbuka: titik ujung digandakan.

## Preset yang perlu disediakan

| Preset | Karakter |
|---|---|
| `rough-sketch` | Default — sesuai referensi B. Kasar, jitter sedang |
| `clean-line` | Jitter rendah, tebal seragam, cocok untuk motion graphic |
| `heavy-marker` | Garis tebal, multipass tinggi, kesan spidol |
| `pencil-light` | Tipis, opacity rendah, grain kuat |

## Aturan implementasi

1. Setiap parameter harus punya **default yang masuk akal** — pipeline jalan tanpa YAML
2. Validasi range saat load config; error jelas kalau di luar batas
3. Jitter **wajib deterministic**: seed = `hash(frame_index, param_seed, track_id)` (P-007). Hash
   stabil antar proses (mis. crc32), bukan `hash()` Python. Kalau random murni, render ulang
   menghasilkan animasi berbeda dan tidak bisa di-debug. `track_id` (bukan indeks stroke) supaya pola
   getar tidak melompat saat urutan stroke berubah antar frame
4. Sediakan flag `--preview N` untuk render hanya N frame — iterasi style harus cepat,
   bukan render 300 frame tiap ganti angka. Threshold per klip dibaca dari `contours/clip_stats.json`,
   bukan dihitung dari N frame preview
