# T-305 induk · Tahap 1 — Uji klip baru (pipeline default [1]–[6], tanpa perubahan kode)

## 0. Konteks
Semua kalibrasi sejauh ini berasal dari 3 klip (`test`, Klip2, klip3). Tahap ini menjalankan pipeline default pada 2–3 klip baru dari konten nyata Rio. Rio lalu memilih masalah yang paling mengganggu, sehingga task kalibrasi berikutnya dipilih berdasarkan data.

## 1. Keputusan Rio (final, 2026-10-15) — catat ke docs/04 "Keputusan T-305 induk Tahap 1 (uji klip baru)" SEBELUM run
1. 2–3 klip baru, masing-masing ≤ 10 detik (≤ 240 frame), isi bervariasi, mewakili konten Rio, hak pakai jelas.
2. Config default apa adanya; tanpa perubahan `src/`, `configs/`, `tests/`.
3. Klip baru tidak boleh ter-commit (dipastikan ter-ignore).
4. Metrik otomatis per klip (lihat 4.3), dibandingkan dengan `test`, Klip2, klip3.
5. Bahan penilaian per klip: MP4 hasil + video berdampingan (foto | hasil), normal dan lambat 0,5×.
6. Template penilaian Rio: 3 masalah paling mengganggu dari daftar + detik kejadiannya, dan nilai 1–5 "siap dipakai untuk konten?".
7. Hasil = daftar prioritas; Rio memilih task kalibrasi berikutnya.

Perbarui juga docs/05: T-305 induk, Tahap 1 berjalan (hitungan papan tidak berubah).

## 2. Pelajaran wajib
- SEMUA berkas di repo hanya lewat Edit/Write.
  - Dilarang: `python -` / `python -c` yang menulis berkas, cat >>, sed, heredoc, printf, echo >, Set-Content.
  - Keluaran skrip dan pipeline yang dijalankan boleh.
- Tidak commit. Tidak membuka PNG/MP4.
- Penolakan classifier: laporkan, jangan disiasati.
- Semua urusan berkas dikerjakan Claude Code.
- GPU: satu klip pada satu waktu; ikuti pengecekan VRAM pipeline (seg 0.8B butuh ≥ 3300 MiB bebas). Bila kurang, berhenti dan minta Rio menutup aplikasi lain.
- F1: hash `work/clips/{test,test_short,Klip2,klip3}/...` + 4 `qc_report.json` + berkas `out/` milik 4 klip lama, SEBELUM dan SESUDAH; harus identik. Berkas baru di `out/` hanya boleh milik klip baru.
- Laporkan penyimpangan dan keraguan apa adanya.

## 3. Aturan file
- F1: `work/t305/` — hash dan bahan penilaian (`work/t305/nilai/`).
- F2: `work/t305_scratch/` bila perlu (pastikan ter-ignore).
- F3: `scripts/t305_newclips.py` untuk metrik dan bahan. Pakai ulang alat metrik yang sudah ada (mis. `tests/temporal_metrics.py`); jangan menulis ulang definisi yang sudah ada.

## 4. Langkah

### 4.1 Inventaris (CPU) — BERHENTI di sini bila ada masalah
- Cari video di `samples/` selain test, test_short, Klip2, klip3. Laporkan nama, durasi, fps, resolusi, dan jumlah frame setelah 24 fps (ffprobe).
- Cek `git check-ignore` untuk tiap klip baru. Bila belum ter-ignore, tambahkan pola minimal di `.gitignore` lewat Edit (jangan mengubah status klip lama), lalu verifikasi.
- Perkiraan waktu GPU (±16,5 s/frame seg + depth) dan disk per klip; sisa disk C:.
- BERHENTI dan laporkan bila:
  - klip baru < 2;
  - ada klip > 10 detik (sebutkan durasinya; Rio yang memutuskan);
  - disk C: < 10 GiB setelah perkiraan.

### 4.2 Run (GPU), per klip, berurutan
`python -m rotoscope run samples/<klip>.<ext> --yes` dengan config default.
- Catat waktu per stage dan VRAM puncak.
- Bila terputus, lanjutkan (resume); jangan mulai dari nol.

### 4.3 Metrik per klip
Untuk klip baru dan 3 klip lama (dari data yang ada, tanpa run ulang):
- frame gagal QC + alasannya;
- cut terdeteksi;
- ukuran subjek (median % foreground dari stable);
- jumlah orang (median blob besar dan median wajah V1 dari T-701);
- T_high / T_low;
- strok oklusi per frame;
- kelahiran strok per frame;
- frame tanpa oklusi %;
- waktu per stage;
- VRAM puncak.

### 4.4 Bahan penilaian (`work/t305/nilai/`)
- Per klip: `<klip>_berdampingan.mp4` (kiri foto, kanan hasil, label besar) dan `<klip>_lambat_0.5x.mp4`, ditambah MP4 hasil asli dari `out/`.
- `PANDUAN.md` (bahasa Indonesia, singkat), berisi:
  - Daftar masalah:
    - garis berkedip;
    - lipatan baju;
    - lengan menempel tak bergaris;
    - tangan/kaki hilang;
    - garis putus-putus;
    - siluet salah;
    - garis terlalu banyak;
    - garis terlalu sedikit;
    - gaya terlalu rapi;
    - gaya terlalu kasar;
    - lainnya.
  - Template balasan:<klip>:
1. <masalah> (detik …)
2. <masalah> (detik …)
3. <masalah> (detik …)
nilai siap konten: 1–5
### 4.5 Laporan (lalu BERHENTI)
- inventaris + status ignore;
- tabel metrik (klip baru vs lama);
- waktu dan VRAM;
- jalur bahan;
- hash F1 identik;
- ringkasan docs/04 + docs/05 (LENGKAP, jangan terpotong);
- penyimpangan / keraguan;
- pengamatan angka yang menonjol (label sebagai pengamatan, bukan penilaian visual).