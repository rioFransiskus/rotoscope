# 00 — PROJECT BRIEF

## Tujuan

Membangun tool Python yang mengubah video meme low-quality dari internet menjadi
animasi rotoscope bergaya sketsa outline hand-drawn, secara otomatis.

## Definisi style target

Referensi: foto A (video TikTok orang menari) → gambar B (sketsa outline di kertas).

Karakteristik style B yang harus dicapai:

| Aspek | Target |
|---|---|
| Detail | **Hanya siluet luar.** Tidak ada detail interior (wajah, lipatan baju, otot) |
| Garis | Tunggal, kontinu, tebal-tipis bervariasi, sedikit bergetar (hand-drawn) |
| Background | Putih/kertas polos. Background asli video dibuang total |
| Warna | Monokrom. Garis gelap di atas kertas terang |
| Feel | Kasar, cepat, seperti sketsa gesture drawing — bukan garis vector yang terlalu rapi |

**Yang BUKAN target:** hasil filter edge-detection (garis putus-putus, noise background ikut
terbawa), atau garis vector yang terlalu bersih dan mekanis.

## Kriteria sukses

1. Input video 5–15 detik → output MP4 + sequence SVG, tanpa intervensi manual
2. Siluet mengikuti bentuk tubuh asli (bukan bentuk generik)
3. Getaran garis antar-frame terkontrol — terasa hand-drawn, bukan rusak/flicker
4. Parameter style (tebal garis, jitter, tekstur) bisa diubah lewat config tanpa ubah kode
5. Seluruh dependency gratis dan bebas dipakai komersial

## Non-goals (jangan dikerjakan)

- Coloring / shading
- Detail wajah atau ekspresi
- Real-time processing
- Multi-person tracking (v1 fokus satu subjek dominan)
- GUI (v1 CLI saja)

## Konteks pemakai

- Konten akhir untuk TikTok / Reels / YouTube → **komersial**, jadi lisensi dependency wajib aman
- Developer: beginner-intermediate Python, baru pertama kali pakai Claude Code
- Hardware: Windows, NVIDIA GTX 1650 Ti (4 GB VRAM)
