# 06 — GITHUB SYNC SETUP

Tujuan: menjadikan repo Git sebagai **satu-satunya sumber kebenaran**, lalu Project
knowledge menariknya. Ini menghilangkan masalah dua salinan yang bercabang.

Verifikasi dari dokumentasi resmi (dicek ulang 2026-09-27):
- https://claude.com/docs/connectors/github
- https://support.claude.com/en/articles/10167454-use-the-github-integration

---

## Yang perlu kamu tahu sebelum mulai

| Fakta | Implikasi |
|---|---|
| Sync **manual**, bukan otomatis | Harus klik "Sync now" tiap ada perubahan |
| Yang ditarik: **nama file + isi file**, pada satu branch | Commit history, PR, issues **tidak** ikut |
| Bisa pilih file/folder spesifik | Tidak perlu seluruh repo masuk knowledge |
| Private repo butuh grant lewat GitHub App | Kalau repo di bawah organisasi, admin harus approve |

---

## Langkah 1 — Dokumen context di folder `docs/`

```
rotoscope/
├── docs/                    ← di-sync ke Project
│   ├── 00-PROJECT-BRIEF.md
│   ├── 01-PIPELINE-SPEC.md
│   ├── 02-STYLE-PARAMS.md
│   ├── 03-CLAUDE-CODE-ONBOARDING.md
│   ├── 04-DECISION-LOG.md
│   ├── 05-TASK-BOARD.md
│   └── 06-GITHUB-SYNC-SETUP.md
├── src/rotoscope/
├── configs/
├── assets/
├── scripts/
├── samples/                 ← gitignored (kecuali .gitkeep)
├── models/                  ← gitignored
├── work/  out/              ← gitignored
├── CLAUDE.md                ← di-sync ke Project
├── requirements.txt         ← di-sync ke Project
├── requirements-lock.txt
└── .gitignore
```

**Kenapa `docs/` terpisah:** saat connect kamu bisa memilih hanya folder ini (plus 2 file root).
Source code tidak perlu masuk knowledge — Claude Code sudah membacanya langsung dari disk.

## Langkah 2 — Pastikan `.gitignore` benar

Minimal harus mengabaikan: `work/`, `out/`, `venv/`, `models/`, `samples/*`, `*.mp4`, `*.png`
(dengan pengecualian `!assets/**/*.png` agar brush tetap ter-commit). Isi lengkap: `.gitignore`
di root repo.

⚠️ Tanpa `.gitignore` yang benar, ribuan frame PNG akan masuk repo — lambat saat push,
dan bisa ikut tertarik ke knowledge.

## Langkah 3 — Buat repo GitHub & push

Repo lokal sudah punya commit dan branch `main` (`init.defaultBranch main`), jadi tidak perlu
commit awal atau `git branch -M main`.

1. github.com → **New repository** → nama `rotoscope` → pilih **Private**
2. **Jangan** centang README, .gitignore, atau license — repo GitHub harus kosong
3. Di PowerShell:

```powershell
cd C:\projects\rotoscope
git remote add origin https://github.com/<username>/rotoscope.git
git push -u origin main
```

Push pertama membuka browser untuk login GitHub (Git Credential Manager bawaan Git for Windows).

## Langkah 4 — Connect ke Project

1. Buka Project **Rotoscope Animation Builder** di claude.ai
2. Klik tombol **+** di kanan atas bagian project knowledge
3. Pilih **GitHub** dari dropdown
4. Cari repo, atau paste URL repo
5. Pakai file browser → pilih **`docs/`**, **`CLAUDE.md`**, dan **`requirements.txt`**
6. Konten terpilih masuk ke project knowledge

**Kenapa `CLAUDE.md` + `requirements.txt` ikut:** Claude di chat bisa melihat aturan yang dipakai
Claude Code dan versi library yang terpasang, jadi prompt yang ia susun berdasarkan isi aktual,
bukan asumsi. Keduanya kecil (<5 KB).

**Kalau repo tidak bisa diakses setelah paste URL:** itu tanda repo private. Ikuti link ke GitHub App,
lalu grant akses — bisa untuk semua repo atau repo tertentu saja.

## Langkah 5 — 🔴 Hapus salinan lama

**Jangan lewatkan langkah ini.**

Sebelum GitHub sync, dokumen disimpan di Project knowledge dengan prefix `claude/`:

```
claude/00-PROJECT-BRIEF.md
claude/01-PIPELINE-SPEC.md
claude/02-STYLE-PARAMS.md
claude/03-CLAUDE-CODE-ONBOARDING.md
claude/04-DECISION-LOG.md
claude/05-TASK-BOARD.md
claude/06-GITHUB-SYNC-SETUP.md
```

Setelah sync GitHub jalan, dokumen yang sama akan ada **dua kali** di knowledge.
Claude tidak tahu mana yang terbaru dan bisa membaca yang salah — ini lebih berbahaya
daripada tidak sync sama sekali.

Hapus ketujuhnya setelah sync terkonfirmasi berhasil — lewat UI Project, atau minta Claude
di chat menghapusnya.

## Langkah 6 — Uji

1. Tandai T-005 `DONE` di `docs/05-TASK-BOARD.md`
2. `git add -A` → `git commit -m "T-005: GitHub sync aktif"` → `git push`
3. Di Project → ikon **Sync** → **Sync now**
4. Buka chat baru di Project, tanya: "T-005 statusnya apa?"
5. Kalau jawabannya `DONE`, sync bekerja

---

## Workflow harian setelah setup

```
kerja di Claude Code
  ↓
git add -A && git commit -m "T-XXX: <ringkas>"
  ↓
git push
  ↓
Project → Sync now
```

Klik **Configure files** kalau nanti perlu mengubah file mana saja yang di-sync.

Refresh sync **sebelum** memulai sesi chat yang membahas status project — bukan sesudah.

---

## Troubleshooting

**Push ditolak dengan `GH007: Your push would publish a private email address`**
Setting privasi email GitHub memblokir commit berisi email pribadi. Minta panduan di chat —
perbaikannya perlu mengganti email di commit yang sudah ada.

**Repo "Connected" tapi isinya tidak terbaca Claude**
Coba disconnect lalu connect ulang. Kalau tetap gagal, fallback ke cara manual: minta Claude
menulis file ke Project lewat chat.

**Kehilangan akses repo**
Isinya tidak lagi bisa dilihat di Project. Preview repo hilang, tapi history percakapan tetap ada.

**Knowledge membengkak**
Kalau nanti file lain ikut ter-sync dan membengkak, pakai "Configure files" untuk mempersempit
kembali ke `docs/` + `CLAUDE.md` + `requirements.txt`.

---

## Batas yang perlu diterima

Sync tetap **manual**. Tidak ada webhook atau auto-refresh — ada
[feature request terbuka](https://github.com/anthropics/claude-ai-mcp/issues/180) untuk itu,
tapi belum tersedia. Tiga langkah di atas harus jadi kebiasaan; tidak ada cara mengotomatiskannya
dari sisi Claude saat ini.
