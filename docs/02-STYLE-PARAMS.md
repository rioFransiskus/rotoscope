# 02 — STYLE & PIPELINE PARAMETERS

Semua parameter di bawah **wajib exposed lewat YAML**, tidak boleh hardcoded.
Ini requirement inti project: kontrol tekstur dan style garis.

Dua jenis file (D-010):
- `configs/default.yaml` — parameter **pipeline** (stage [2]–[4]): paths, segment, depth, qc, groups,
  stabilize, vectorize. Default = hasil T-102c (D-009).
- `configs/styles/*.yaml` — parameter **style** (stage [5]). Ganti style = jalankan ulang [5] saja.

Nilai bertanda **sementara** belum dikalibrasi; task kalibrasinya disebut di kolom catatan.

---

## `configs/default.yaml` — pipeline

```yaml
# ── PATHS ──────────────────────────────────────────
paths:
  work_dir: "work"           # boleh di drive lain (±1–5 GB per klip)
  out_dir: "out"

# ── SEGMENT [2] — Sapiens2-seg ─────────────────────
segment:
  model: "0.8b"              # "0.8b" | "0.4b" — eksplisit, untuk SELURUH klip (D-009)
  model_ids:
    "0.8b": "facebook/sapiens2-seg-0.8b"   # id HF yang dipakai di T-102c
    "0.4b": "facebook/sapiens2-seg-0.4b"
  revision: null             # commit hash checkpoint HF; diisi di T-102b (pin)
  precision: "fp16"          # "fp16" | "fp32" — TANPA bf16 (P-005)
  vram_min_free_mib:
    "0.8b": 3300             # peak reserved 3276 MiB (T-102c)
    "0.4b": 2300             # peak reserved 2258 MiB
  probs_dtype: "uint8"       # round(p × 255), npz deflate

# ── DEPTH [2c] — Depth Anything V2 Small ───────────
depth:
  model_id: "depth-anything/Depth-Anything-V2-Small-hf"   # HANYA Small (Apache-2.0)
  revision: null             # diisi di T-105 (pin)
  precision: "fp32"
  vram_min_free_mib: 500     # peak reserved 424 MiB

# ── QC (akhir stage [2]) ───────────────────────────
qc:
  area_min: 0.03             # fraksi frame
  area_max: 0.70
  iou_min: 0.55              # IoU foreground vs frame sebelumnya
  blob_min: 0.05             # komponen "besar" = > 5% area frame
  max_big_blobs: 1
  area_median_window: 49     # frame (±2 s); median BERGULIR, bukan median klip
  area_drop_min: 0.6         # SEMENTARA — dikalibrasi di T-102b

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
    max_match_dist_px: 12    # SEMENTARA — jarak rata-rata maks pencocokan track_id, dikalibrasi di T-202
```

### Validasi `default.yaml` (saat load, error jelas kalau gagal)

| parameter | range / aturan |
|---|---|
| `paths.work_dir`, `paths.out_dir` | string tidak kosong; folder dibuat kalau belum ada |
| `segment.model` | `"0.8b"` atau `"0.4b"` |
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
| `stabilize.depth.normalize` | nilai dari daftar metode yang diimplementasi (T-302) |
| `vectorize.min_region_area`, `min_hole_area` | int ≥ 0 |
| `vectorize.line_min_px`, `min_stroke_px` | int ≥ 1 |
| `vectorize.depth_lines.blur_sigma` | ≥ 0 (0 = tanpa blur) |
| `vectorize.depth_lines.lo_pct`, `hi_pct` | 0 < lo_pct < hi_pct < 100 |
| `vectorize.depth_lines.erode_px` | int ganjil ≥ 1 |
| `vectorize.depth_lines.min_dist_px`, `min_len_px` | ≥ 0 |
| `vectorize.track.max_match_dist_px` | > 0 |

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
# ── SHAPE ──────────────────────────────────────────
shape:
  simplify_epsilon: 2.5      # cv2.approxPolyDP. Naik = lebih sedikit titik, lebih kasar
  resample_points: 200       # jumlah titik tetap per stroke (maks 1 titik / px untuk stroke pendek)
  smooth_tension: 0.5        # Catmull-Rom tension. 0 = tajam, 0.5 = Catmull-Rom standar
  spline_steps: 8            # titik spline per segmen sebelum resample
# (min_contour_area pindah ke vectorize.min_region_area di default.yaml — D-010)

# ── STROKE ─────────────────────────────────────────
stroke:
  width_base: 3.2            # tebal dasar garis (px)
  width_variation: 0.45      # 0 = seragam, 1 = variasi ekstrem
  width_noise_scale: 0.08    # frekuensi perubahan tebal sepanjang path
  color: "#1a1a1a"
  opacity: 0.92
  cap: "round"
  taper_ends: true           # ujung garis menipis
  taper_px: 20               # panjang zona taper di tiap ujung (px)
  taper_min: 0.15            # tebal di ujung sebagai fraksi tebal normal
  by_type:                   # override per jenis garis (skema [4]); 1.0 = sama dengan dasar
    silhouette:      {width_scale: 1.0, opacity_scale: 1.0}
    silhouette_hole: {width_scale: 1.0, opacity_scale: 1.0}
    group_boundary:  {width_scale: 1.0, opacity_scale: 1.0}
    occlusion:       {width_scale: 1.0, opacity_scale: 1.0}

# ── JITTER (hand-drawn feel) ───────────────────────
jitter:
  amplitude: 1.8             # pergeseran titik (px). 0 = garis mekanis
  frequency: 0.12            # skala noise. Kecil = gelombang panjang
  temporal_seed_mode: "frame"  # "frame" = getar tiap frame | "fixed" = diam
  temporal_drift: 0.35       # seberapa cepat pola jitter berubah antar frame
  param_seed: 0              # seed = hash(frame_index, param_seed, track_id) — P-007

# ── MULTI-PASS (kesan sketsa ditimpa) ──────────────
multipass:
  enabled: true
  passes: 2                  # jumlah garis tumpang tindih
  offset: 1.2                # jarak antar pass (px)
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
  # output_width: diputuskan sebelum T-203 (resolusi output terpisah dari resolusi kerja)
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
| `texture.brush_image`, `paper.texture_image` | file ada (kalau dipakai) |
| `texture.stamp_spacing` | > 0 |
| `texture.pressure_noise`, `texture.grain_strength`, `paper.texture_opacity`, `paper.vignette` | 0–1 |
| `render.ss` | int 1–8 |

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
