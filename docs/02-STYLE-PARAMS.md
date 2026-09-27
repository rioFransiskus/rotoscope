# 02 — STYLE PARAMETERS

Semua parameter di bawah **wajib exposed lewat YAML**, tidak boleh hardcoded.
Ini requirement inti project: kontrol tekstur dan style garis.

## Contoh `configs/styles/rough-sketch.yaml`

```yaml
# ── SHAPE ──────────────────────────────────────────
shape:
  simplify_epsilon: 2.5      # cv2.approxPolyDP. Naik = lebih sedikit titik, lebih kasar
  resample_points: 200       # jumlah titik tetap per kontur
  smooth_tension: 0.5        # Catmull-Rom tension. 0 = tajam, 1 = sangat bulat
  min_contour_area: 800      # buang blob kecil (px²)

# ── STROKE ─────────────────────────────────────────
stroke:
  width_base: 3.2            # tebal dasar garis (px)
  width_variation: 0.45      # 0 = seragam, 1 = variasi ekstrem
  width_noise_scale: 0.08    # frekuensi perubahan tebal sepanjang path
  color: "#1a1a1a"
  opacity: 0.92
  cap: "round"
  taper_ends: true           # ujung garis menipis

# ── JITTER (hand-drawn feel) ───────────────────────
jitter:
  amplitude: 1.8             # pergeseran titik (px). 0 = garis mekanis
  frequency: 0.12            # Perlin noise scale. Kecil = gelombang panjang
  temporal_seed_mode: "frame"  # "frame" = getar tiap frame | "fixed" = diam
  temporal_drift: 0.35       # seberapa cepat pola jitter berubah antar frame

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

# ── TEMPORAL (dari stage stabilize) ────────────────
temporal:
  mask_ema_alpha: 0.7        # 1.0 = tanpa smoothing. Turun = lebih stabil, lebih lag
  optical_flow_blend: 0.4    # bobot mask hasil warp
  boil_preserve: 0.3         # 0 = mati total, 1 = boiling penuh
```

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
3. Jitter **wajib deterministic**: seed = `hash(frame_index, param_seed)`. Kalau random murni,
   render ulang menghasilkan animasi berbeda dan tidak bisa di-debug
4. Sediakan flag `--preview N` untuk render hanya N frame — iterasi style harus cepat,
   bukan render 300 frame tiap ganti angka
