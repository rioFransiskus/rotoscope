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
    enabled: true            # T-302: kernel simetris terpotong (tanpa optical flow: ditolak di T-303, docs/04); false = hanya spasial
    mask_ema_alpha: 0.7      # 1.0 = tanpa smoothing. Turun = lebih stabil, lebih lag
    boil_preserve: 0.3       # 0 = mati total, 1 = boiling penuh
    qc_fail_weight: 0.1      # SEMENTARA — bobot frame gagal QC di kernel; harus < ρ (0,176 pada α 0,7); klip kedua → T-305
    cut_diff: 0.08           # selisih abu-abu rata-rata antar frame (0–1) yang dianggap cut → jendela temporal dipotong; 0 = mati
  island_min_px: 30          # N — filter pulau pada peta GRUP (T-102c: pada peta kelas → T-305)
  mode_k: 3                  # K — mode filter peta grup, background ikut
  depth:
    normalize: "log_median"  # T-302: log − median (tanpa IQR); "log_median_iqr" = kompatibilitas
    log_eps: 1.0e-6          # log(max(d, eps)) — hanya menahan disparity ≤ 0 (min klip uji 0.64)
    iqr_min: 0.01            # SEMENTARA — batas bawah pembagi IQR (log) foreground, dikalibrasi di T-302
    temporal: false          # EMA kedalaman; hanya berlaku kalau temporal.enabled. Tetap MATI (tanpa flow: ghost edge, id oklusi ×3; flow ditolak di T-303)

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
  source: "strokes"          # "strokes" (stage [5], default sejak T-203b; + out/svg/<nama>/) | "silhouette" (stable/groups, Phase 1)
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
| `stabilize.temporal.boil_preserve`, `qc_fail_weight` | 0–1 |
| `stabilize.temporal.cut_diff` | 0–1 (0 = pengaman cut mati) |
| `stabilize.island_min_px` | int ≥ 0 (0 = mati) |
| `stabilize.mode_k` | int ganjil ≥ 1 (1 = mati) |
| `stabilize.temporal.enabled`, `stabilize.depth.temporal` | bool |
| `stabilize.depth.normalize` | `"log_median_iqr"` (A, kompatibilitas) atau `"log_median"` (B, T-302; default diganti di Tahap 4). Affine linear dan IQR tetap per klip ditolak (D-010, T-302) |
| `stabilize.depth.log_eps` | 1e-12 ≤ x ≤ 0.01 |
| `stabilize.depth.iqr_min` | 1e-3 ≤ x ≤ 1 (dengan batas `log_eps`, `depth_smooth` dijamin muat float16) |
| `vectorize.min_region_area`, `min_hole_area` | int ≥ 0 |
| `vectorize.line_min_px`, `min_stroke_px` | int ≥ 1 |
| `vectorize.depth_lines.blur_sigma` | ≥ 0 (0 = tanpa blur) |
| `vectorize.depth_lines.lo_pct`, `hi_pct` | 0 < lo_pct < hi_pct < 100 |
| `vectorize.depth_lines.erode_px` | int ganjil ≥ 1 |
| `vectorize.depth_lines.min_dist_px`, `min_len_px` | ≥ 0 |
| `vectorize.track.max_match_dist_px` | > 0 (default 16 px pada frame 480 px; mengubahnya membuat `contours/` basi) |
| `export.source` | `"silhouette"` atau `"strokes"` (default `"strokes"` sejak T-203b: MP4 dari `strokes/*.png` apa adanya + salinan SVG ke `out/svg/<nama>/`, tag warna bt709; `"silhouette"` = Phase 1, tanpa SVG) |
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
  resample_points: 4         # batas bawah N titik per strok; N = max(ini, ceil(panjang / 2 px ref)) (T-401)
  smooth_tension: 0.5        # Catmull-Rom tension. 0 = tajam, 0.5 = Catmull-Rom standar
  spline_steps: 8            # titik spline minimum per segmen (otomatis lebih rapat bila galat akor > 0.03 px ref)
  edge_mode: "hide"          # garis di tepi frame: "hide" = disembunyikan (tubuh terpotong frame) | "draw" = digambar
# (min_contour_area pindah ke vectorize.min_region_area di default.yaml — D-010)

# ── STROKE ─────────────────────────────────────────
stroke:
  width_base: 9.0            # tebal dasar garis (px ref); keputusan Rio (visual), setara look test = 7.2
  width_variation: 0.5       # 0 = seragam, 1 = variasi ekstrem (keputusan Rio, visual, T-401)
  width_noise_scale: 0.036   # frekuensi noise tebal (per px ref), terkunci posisi; x2 tampak "merayap" (T-401)
  color: "#1a1a1a"
  opacity: 0.92              # T-403 (keputusan Rio): tinta sedikit transparan, SEMUA garis; < 1 = komposisi "over" antar pass
  cap: "round"
  taper_ends: true           # ujung BEBAS menipis (smoothstep); termasuk oklusi (keputusan Rio, T-401)
  taper_px: 70               # panjang zona taper di tiap ujung (px ref), maks panjang strok / 2
  taper_min: 0.5             # tebal di ujung sebagai fraksi tebal normal
  by_type:                   # override per jenis garis (skema [4]); 1.0 = sama dengan dasar
    silhouette:      {width_scale: 1.0, opacity_scale: 1.0, taper_ends: null}   # taper_ends null = warisi stroke.taper_ends
    silhouette_hole: {width_scale: 1.0, opacity_scale: 1.0, taper_ends: null}
    group_boundary:  {width_scale: 1.0, opacity_scale: 1.0, taper_ends: null}
    occlusion:       {width_scale: 1.0, opacity_scale: 1.0, taper_ends: null}

# ── JITTER (hand-drawn feel) ───────────────────────
jitter:
  amplitude: 0.0             # PUNCAK pergeseran per kanal (px ref); |D| ≤ amplitude × √2. 0 = jitter MATI (default; byte-identik T-401). Penilaian visual T-402: semua varian terasa acak-acakan
  frequency: 0.053           # skala noise (per px ref; sel = 1 / frequency). Kecil = gelombang panjang. 0 = jitter mati. r = amplitude × frequency > 0,19 → peringatan lipatan
  temporal_seed_mode: "frame"  # "frame" = getar tiap gambar | "fixed" = diam (indeks gambar 0)
  temporal_drift: 0.35       # sel waktu per gambar: 0 = beku, 1 = tiap gambar independen
  param_seed: 0              # seed noise tebal (T-401) dan dasar seed jitter (stream terpisah, salt berbeda) — P-007
  hold_frames: 2             # "gambar" jitter diperbarui tiap N frame (indeks gambar = floor(frame_index / N)); 1 = tiap frame, 2 = "on twos"
  stroke_independence: 0.0   # 0 = medan koheren saja (sambungan tetap menyambung), 1 = getar independen per track_id; menjaga RMS

# ── MULTI-PASS (kesan sketsa ditimpa) ──────────────
multipass:
  enabled: true
  passes: 2                  # jumlah garis tumpang tindih; 1 = tanpa pass tambahan (keputusan Rio, T-403)
  offset: 5.5                # MEDIAN |D| pass tambahan vs pass 0 (px ref); medan koheren, skala medan mengikuti offset
  opacity_falloff: 0.35      # alpha pass k = stroke.opacity × opacity_scale × falloff^k
  temporal_mode: "fixed"     # "fixed" = pass tambahan statis | "frame" = evolusi per gambar (jitter.hold_frames, temporal_drift)

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

Bagian `temporal:` lama (mask_ema_alpha, boil_preserve) **pindah** ke
`stabilize.temporal` di `default.yaml` — dibaca stage [3], bukan [5] (D-010). `optical_flow_blend` (juga di bagian lama
itu) DIHAPUS di T-303: optical flow ditolak berdasarkan data (docs/04 "Hasil T-303 (ditolak)").

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
| `stroke.by_type` | kunci ⊆ {`silhouette`, `silhouette_hole`, `group_boundary`, `occlusion`}; `width_scale` > 0, `opacity_scale` 0–1, `taper_ends` bool opsional (tidak ada / `null` = warisi `stroke.taper_ends`; T-401) |
| `jitter.amplitude`, `jitter.frequency` | ≥ 0 |
| `jitter.temporal_seed_mode` | `"frame"` \| `"fixed"` |
| `jitter.temporal_drift` | 0–1 |
| `jitter.param_seed` | int |
| `jitter.hold_frames` | int ≥ 1 (T-402) |
| `jitter.stroke_independence` | 0–1 (T-402) |
| `multipass.passes` | int ≥ 1 |
| `multipass.offset` | ≥ 0 |
| `multipass.opacity_falloff` | 0–1 |
| `multipass.temporal_mode` | `"fixed"` \| `"frame"` (T-403) |
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
  3.2 → 7.2, `jitter.amplitude` 1.8 → 4.0, `multipass.offset` 1.2 → 2.7, `taper_px` 20 → 45 (T-401: dipilih Rio 70), `width_noise_scale` 0.08 → 0.036 dan
  `jitter.frequency` 0.12 → 0.053 (per px, dibagi 2.25). **Pengecualian:** `simplify_epsilon` setara look test = 5.6, tetapi
  penilaian visual Rio memutuskan **2.8** (5.6 membuang bentuk) dipadukan dengan `smooth_px` (3.0 masih bergelombang → **5.0**).
  `width_base` setara look test = 7.2, tetapi penilaian visual Rio memilih **9.0**.
- **`shape.smooth_px` (keputusan Rio, perbandingan visual 2026-10-03):** Gaussian arc-length sebelum approxPolyDP, sigma = nilai
  ini (px ref × `unit`), 0 = mati. Kontur dire-sample tiap 1 px ref; sudut tajam (belok > 60° pada jendela ±6 sampel) dikunci
  dan memecah jalur; ujung strok terbuka (termasuk titik silang tepi) tidak bergeser; strok tertutup dan loop dihaluskan periodik
  tanpa takik. Tanpa penyusutan bentuk yang berarti (luas silhouette < 0,01% rata-rata).
- **Aktif (T-203a + T-401):** `shape.simplify_epsilon`, `shape.smooth_px`, `shape.smooth_tension`, `shape.spline_steps`, `shape.edge_mode`,
  `shape.resample_points`, `stroke.width_base`, `stroke.width_variation`, `stroke.width_noise_scale`, `stroke.taper_ends`,
  `stroke.taper_px`, `stroke.taper_min`, `stroke.color`, `stroke.cap` (hanya `"round"`; nilai lain → error stage),
  `stroke.by_type.*.width_scale`, `stroke.by_type.*.taper_ends`, `jitter.param_seed` (seed noise tebal + dasar seed jitter), `paper.color`,
  `render.ss`, `render.output_width`; **T-402:** `jitter.amplitude`, `jitter.frequency`, `jitter.temporal_seed_mode`,
  `jitter.temporal_drift`, `jitter.hold_frames`, `jitter.stroke_independence`; **T-403:** `stroke.opacity`, `stroke.by_type.*.opacity_scale`,
  `multipass.enabled`, `multipass.passes`, `multipass.offset`, `multipass.opacity_falloff`, `multipass.temporal_mode`. Hanya parameter ini yang masuk hash style stage [5].
- **Divalidasi tetapi BELUM aktif (dicatat di `strokes/manifest.json` → `ignored_params`; aktif di task Phase 4 berikutnya):**
  `texture.*`, `paper.enabled`, `paper.texture_*`, `paper.vignette`.
- **Multipass (T-403; keputusan Rio, docs/04 "Keputusan T-403"):** pass 0 = garis asli (setelah jitter); pass k ≥ 1 = titik pass 0 + medan koheren 2D (mesin jitter T-402,
  salt tersendiri per k, penjaga tepi φ), statis terhadap waktu (`temporal_mode` "fixed"; "frame" = evolusi per gambar lewat `jitter.hold_frames` / `jitter.temporal_drift`).
  `offset` = MEDIAN |D| (px ref): amplitudo A = offset / 0,59, panjang gelombang = A / 0,15 (r konstan 0,15, skala medan mengikuti offset); offset 2,7 ≈ RMS 3,0, p95 4,7.
  Alpha pass k = `stroke.opacity` × `opacity_scale` × `opacity_falloff`^k; union dalam pass, "over" antar pass (f = 1 − Π(1 − a_k cov_k)). `enabled: false` atau `passes: 1`
  DAN `stroke.opacity: 1.0` = SVG + PNG byte-identik T-402 (DEFAULT sejak Tahap 4: `passes` 2, `offset` 5,5, `opacity_falloff` 0,35, `opacity` 0,92, `temporal_mode` "fixed"; jitter tetap mati). Batas: `opacity_scale` ≠ 1 membuat sambungan antar tipe bisa lebih gelap. Peringatan lipatan: r jitter + 0,15 > 0,19.
- **Jitter (T-402; keputusan Rio, docs/04 "Keputusan T-402"):** medan perpindahan KOHEREN vektor 2D, `D = amplitude × unit × [√(1 − s) F + √s G_track] × φ`
  (s = `stroke_independence`). `amplitude` = PUNCAK per kanal (px ref; |D| ≤ amplitude × √2 pada s = 0; batas umum amplitude × √2 × (√(1 − s) + √s)),
  RMS per kanal ≈ 0,40–0,47 × amplitude (terukur). `frequency` (per px ref; sel noise = 1 / frequency; 0 = jitter mati); panjang korelasi
  (ρ = 0,5) = 0,64 sel. `temporal_drift` = sel waktu lattice per gambar: 0 = beku, 1 = tiap gambar independen (korelasi gambar berurutan terukur
  0,96 / 0,81 / 0,46 / 0,00 untuk drift 0,15 / 0,35 / 0,7 / 1,0). `hold_frames`: indeks gambar k = floor(frame_index / hold_frames) dengan
  `frame_index` ABSOLUT (1 = tiap frame, 2 = "on twos"); `temporal_seed_mode` `"fixed"` = k selalu 0. `stroke_independence` s: 0 = hanya medan koheren
  (sambungan menyambung, seam tidak retak); s > 0 mencampur getar per `track_id` dengan varians (RMS) terjaga — sambungan terbuka, persilangan
  baru mungkin (invarian hanya untuk s = 0); id oklusi berganti ±0,35 per strok-frame, jadi pola komponen independen loncat saat id berganti.
  **Amplitudo 0 = SVG + PNG byte-identik dengan T-401** (untuk nilai `hold_frames` / `stroke_independence` / `frequency` / `temporal_*` apa pun).
  **Pelunakan tepi:** φ = 0 untuk jarak ke tepi bawah / kiri / kanan ≤ tebal maks / 2 + 1 px (7,75 pada default; dihitung dari style), smoothstep
  sampai 1 pada +40 px ref; tinta di 3 baris / kolom terluar tidak berubah dan ujung ekstensi tidak tertarik masuk.
  **Penjaga lipatan:** r = amplitude × frequency × (√(1 − s) + √s); r > `JITTER_FOLD_R_WARN` = 0,19 → peringatan SATU kali per run (stderr + `strokes/manifest.json` →
  `jitter.fold_warn`): garis dapat melipat / bersilang (terukur, medan penuh: r ≤ 0,18 lulus Jacobian 0,05; r 0,21 gagal di 2–17 frame; r 0,32 → −0,24;
  r 0,42 → −0,76; amplitudo 4 dengan frequency ×2 (r 0,42) melipat). Tanpa clamp, tanpa error.
  **Default final (keputusan Rio, 2026-10-07): `amplitude` 0 = jitter MATI** (penilaian visual: semua varian terasa acak-acakan, bentuk tubuh lebih jelas
  tanpa jitter); fitur tetap tersedia lewat style. Nilai lain hanya berlaku bila amplitudo > 0: frequency 0,053 / `temporal_seed_mode` "frame" /
  drift 0,35 / `hold_frames` 2 (≥ 1; 0 ditolak) / `stroke_independence` 0 / `param_seed` 0. Nilai papan evaluasi (amplitude 4,0 / frequency 0,053, r 0,21)
  BUKAN default; papan memasangkan amplitudo {2; 4; 6; 8} dengan frequency {0,053; 0,053; 0,035; 0,0265}.
- **Tebal variabel (T-401):** `w = max(1 px output, width_base × unit × by_type.width_scale × (1 + width_variation × n) × taper)`;
  `n` ∈ [−1, 1] = value noise 2D terkunci posisi (sel = `unit / width_noise_scale` px output; seed = hash(`jitter.param_seed`),
  tanpa `frame_index`: tebal statis terhadap waktu); `taper` hanya di ujung bebas (smoothstep dari `taper_min` ke 1 sepanjang
  `min(taper_px, panjang / 2)`). `by_type.<tipe>.taper_ends` (tidak ada / `null` = warisi `stroke.taper_ends`) mengesampingkan per
  tipe. Default (keputusan Rio, penilaian visual 2026-10-06): variasi 0,5; skala noise 0,036 (×1; ×2 tampak "merayap" di garis
  bergerak); `taper_px` 70, `taper_min` 0,5; taper aktif juga untuk oklusi; hierarki tipe rata (semua `width_scale` 1,0);
  `resample_points` 4 (N ditentukan jarak maks 2 px ref).
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
3. Jitter **wajib deterministic** (P-007): hasil sebuah frame = fungsi dari (`frame_index` ABSOLUT lewat indeks gambar, parameter,
   posisi titik) — hash bilangan bulat 64-bit stabil antar proses (splitmix64, bukan `hash()` Python), tanpa rantai antar frame. **T-402
   mengganti** rencana lama `hash(frame_index, param_seed, track_id)` per strok (D-010) dengan medan koheren: seed medan = `hash(param_seed,
   salt jitter, kanal)`; `track_id` hanya untuk komponen independen (`stroke_independence` > 0); stream jitter dan stream tebal terpisah (salt
   berbeda). Kalau random murni, render ulang menghasilkan animasi berbeda dan tidak bisa di-debug
4. Sediakan flag `--preview N` untuk render hanya N frame — iterasi style harus cepat,
   bukan render 300 frame tiap ganti angka. Threshold per klip dibaca dari `contours/clip_stats.json`,
   bukan dihitung dari N frame preview
