# 03 — CLAUDE CODE ONBOARDING (Windows)

Semua langkah di sini diverifikasi dari dokumentasi resmi:
https://code.claude.com/docs/en/install

---

## Langkah 0 — Cek prasyarat

**Akun:** Claude Code butuh Pro, Max, Team, Enterprise, atau Console.
Plan Claude.ai gratis **tidak** termasuk akses Claude Code. Cek ini dulu.

**Sistem:** Windows 10 versi 1809+ atau Windows Server 2019+, RAM 4 GB+, x64/ARM64.

Node.js **tidak diperlukan** untuk native installer.

---

## Langkah 1 — Install

Buka **PowerShell** (bukan CMD — cek prompt: ada `PS` di depan berarti PowerShell):

```powershell
irm https://claude.ai/install.ps1 | iex
```

Tidak perlu Run as Administrator.

**Alternatif WinGet** (kalau kamu sudah terbiasa):
```powershell
winget install Anthropic.ClaudeCode
```
Catatan: WinGet tidak auto-update. Perlu `winget upgrade Anthropic.ClaudeCode` berkala.
Native installer auto-update di background.

### ⚠️ Gotcha #1 — "command not found" setelah install

Installer menambahkan `~/.local/bin` ke PATH, tapi Windows tidak mempropagasi perubahan
environment variable ke terminal yang sedang terbuka.

**Fix: tutup PowerShell, buka baru.** Jangan install ulang.
(Issue terlapor: github.com/anthropics/claude-code/issues/18064)

---

## Langkah 2 — Verifikasi

```powershell
claude --version
```
Harus print versi, contoh: `2.1.211 (Claude Code)`

```powershell
claude doctor
```
Diagnostik read-only: kesehatan instalasi, validasi settings, warning + saran perbaikan.
Tidak memulai sesi.

---

## Langkah 3 — Git for Windows (opsional)

Git for Windows **opsional** di native Windows. Fungsinya mengaktifkan Bash tool lewat
Git Bash. Tanpa itu, Claude Code memakai PowerShell sebagai shell tool.

**Rekomendasi: install saja** (https://git-scm.com/downloads/win) — kamu tetap butuh Git
untuk version control project ini, dan perintah shell di tutorial Python umumnya bash-style.

Kalau Claude Code tidak menemukan Git Bash, set path di `settings.json`:
```json
{
  "env": {
    "CLAUDE_CODE_GIT_BASH_PATH": "C:\\Program Files\\Git\\bin\\bash.exe"
  }
}
```

---

## Langkah 4 — Login

```powershell
claude
```
Browser terbuka untuk autentikasi. Sekali saja.

---

## Langkah 5 — Mulai project

```powershell
mkdir C:\projects\rotoscope
cd C:\projects\rotoscope
git init
claude
```

Claude Code membaca konteks dari folder tempat kamu menjalankannya.
**Selalu `cd` ke folder project dulu, baru `claude`.**

---

## Langkah 6 — Setup `CLAUDE.md`

Buat file `CLAUDE.md` di root project. Claude Code membacanya otomatis tiap sesi.
Isi dengan ringkasan dari `00-PROJECT-BRIEF.md` + `01-PIPELINE-SPEC.md`.

Ini pengganti "Project Instructions" versi Claude Code. Tanpa ini, tiap sesi baru kamu
harus menjelaskan ulang project dari nol.

---

## Workflow harian

| Perintah | Fungsi |
|---|---|
| `claude` | Mulai sesi interaktif |
| `/clear` | Reset context — pakai saat ganti topik, hemat token |
| `/init` | Generate draft `CLAUDE.md` dari struktur repo |
| `claude doctor` | Diagnostik |
| `claude update` | Update manual tanpa tunggu background check |

### Cara memberi instruksi yang efektif

❌ "Buatkan program rotoscoping"
✅ "Implementasikan `src/rotoscope/ingest.py` sesuai kontrak di `01-PIPELINE-SPEC.md` stage 1.
   Tulis test di `tests/test_ingest.py`. Jangan sentuh modul lain."

**Prinsip:**
1. **Satu stage per sesi.** Ikuti urutan Phase 1–6 di pipeline spec
2. **Selalu commit sebelum minta perubahan besar** — supaya bisa `git revert` kalau hasilnya jelek
3. **Minta plan dulu** untuk task kompleks: "Jelaskan rencanamu dulu, jangan tulis kode."
4. **Review diff-nya.** Claude Code minta approval sebelum edit — baca, jangan asal approve
5. Kalau Claude Code ngawur, `/clear` lalu mulai ulang dengan instruksi lebih spesifik

---

## Gotcha #2 — Token

Claude Code membaca file otomatis. Di repo besar ini boros. Mitigasi:
- Isi `.gitignore` dengan `work/`, `out/`, `*.png`, `*.mp4` — jangan biarkan Claude Code
  mencoba membaca ribuan frame PNG
- Pakai `/clear` antar task
