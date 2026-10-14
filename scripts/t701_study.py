"""T-701 Tahap 1a (CPU, baca-saja terhadap klip, TANPA unduhan bobot / GPU): studi garis lengan menempel badan (pose / normal).

    python scripts/t701_study.py hash before|after   # sha256 frames/seg/depth/stable/contours/strokes + qc_report.json 4 klip + out/ -> work/t701/
    python scripts/t701_study.py meta                # metadata HF (API + HEAD, tanpa isi bobot) + HEAD/listing .task + disk + token -> work/t701_scratch/meta.json
    python scripts/t701_study.py samples             # daftar sampel klip3 / test / unit T-305c + jendela statis + bbox P2 -> work/t701_scratch/samples.json

Tahap 1b (docs/04 "Keputusan T-701 Tahap 1b"; izin unduh + GPU dari Rio):
    python scripts/t701_study.py hash before_1b|after_1b
    python scripts/t701_study.py download            # .task float16/1 -> models/ (md5/crc32c/sha256) + Sapiens2 pose/normal (commit pin) -> cache HF
    python scripts/t701_study.py samples1b           # klip3 20 frame + test 8 + wajah / blob per frame -> work/t701_scratch/samples_1b.json
    python scripts/t701_study.py p1                  # MediaPipe Pose Landmarker Heavy (CPU): IMAGE sampel, VIDEO jendela statis -> p1.json
    python scripts/t701_study.py p2 fp16|fp32        # Sapiens2-pose 0.4B (GPU; fp32 hanya 5 frame) -> p2_<prec>.json (titik wajah dibuang sebelum post-process)
    python scripts/t701_study.py normal fp16|mixed|cpu   # Sapiens2-normal 0.4B -> normal_<mode>/*.npz (wajah dimasking) + normal_<mode>.json
    python scripts/t701_study.py metrics [MODE_N]    # tabel metrik + kriteria (a) / (c) + AUC N -> metrics.json
    python scripts/t701_study.py board [MODE_N]      # papan 28 frame (seed 701002) -> work/t701/papan/NN.png + board_map.json (scratch)

Tahap 1c (docs/04 "Keputusan T-701 setelah Tahap 1b"; P2 crop per orang, acuan orang = wajah V1 terdaftar):
    python scripts/t701_study.py hash before_1c|after_1c
    python scripts/t701_study.py p2c [fp32|fp16]     # 20 frame klip3, crop Voronoi per wajah V1 (fp32 bila VRAM cukup) -> p2c_<prec>.json
    python scripts/t701_study.py metrics1c           # terdeteksi per orang V1, (c), pose ganda, waktu / VRAM -> metrics_1c.json
    python scripts/t701_study.py board1c             # frame yang crop-nya berubah (seed 701003) -> work/t701/papan_1c/NN.png + board_map_1c.json

Tahap 3 (prototipe I1): hash F1 `hash before_3|after_3` di sini; prototipe di scripts/t701_proto.py.

Hanya membaca work/clips/* dan scripts/t305c_labels.json; keluaran ke work/t701/ dan work/t701_scratch/ (di-ignore git).
`meta` hanya meminta metadata (model_info, HEAD, listing objek GCS): tidak ada badan berkas bobot yang diunduh.
Kandidat: P1 MediaPipe Pose Landmarker Heavy (CPU), P2 Sapiens2-pose 0.4B, N Sapiens2-normal 0.4B (docs/04 "Keputusan T-701 (studi)").
"""

from __future__ import annotations

import base64
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import temporal_metrics as tm  # noqa: E402
from klip3_diag import persons_of  # noqa: E402

from rotoscope import stabilize as stb  # noqa: E402
from rotoscope.config import load_pipeline  # noqa: E402

CLIPS_DIR = ROOT / "work" / "clips"
OUT = ROOT / "work" / "t701"
SCRATCH = ROOT / "work" / "t701_scratch"
LABELS = ROOT / "scripts" / "t305c_labels.json"
HASH_CLIPS = ("test", "test_short", "Klip2", "klip3")
HASH_SUBDIRS = ("frames", "seg", "depth", "stable", "contours", "strokes")

HF_REPOS = ("facebook/sapiens2-pose-0.4b", "facebook/sapiens2-normal-0.4b")
HF_REF_REPO = ("facebook/sapiens2-seg-0.4b", "449b3c5335e6722bb94990abdd1aa6e612432f22")   # acuan: sudah di cache (T-102c)
TASK_BUCKET = "mediapipe-models"
TASK_PREFIX = "pose_landmarker/pose_landmarker_heavy/"
TASK_URLS = {v: f"https://storage.googleapis.com/{TASK_BUCKET}/{TASK_PREFIX}float16/{v}/pose_landmarker_heavy.task"
             for v in ("1", "latest")}

CUT_MARGIN = 2                     # sampel tidak boleh berjarak <= 2 frame dari cut (5.3)
DIAG2_FRAMES = (19, 24, 30, 36, 42, 60)   # klip3 "Diagnosis 2" (f19 = pengamatan Rio + 5 frame 3-blob non-cut)
N_KLIP3 = 12
N_TEST = 8
HAND_TOUCH_PX = 5                  # komponen tangan didilasi 5 px; menyentuh Upper_Clothing orang yang sama = lengan baju bersambung ke tangan
HAND_MIN_PX = 30                   # komponen tangan lebih kecil diabaikan (= filter pulau N)
ARM_MIN_PX = tm.GROUP_MIN_PX       # komponen lengan `test` >= 200 px
ARM_RING_PX = 3                    # cincin di sekeliling komponen lengan
ARM_FRONT_MIN = 0.5                # fraksi cincin lengan yang berlabel torso: >= ini = lengan di depan / di dalam siluet badan
TEST_MIN_GAP = 12                  # jarak minimum antar sampel `test` (frame)
STATIC_LEN = 12                    # jendela uji statis (preseden klip2_review.py STATIC_FALLBACK)
MATCH_IOU = 0.5                    # unit T-305c dianggap masih ada bila bbox strok oklusi frame wakil IoU >= ini

# ── Tahap 1b (nilai dari docs/04 "Keputusan T-701 Tahap 1b", ditetapkan sebelum mengukur) ──
MIB = 1024 * 1024
N_KLIP3_1B = 20
FACE_MIN_FRAC = 0.001              # komponen grup face (argmax mentah, 8-arah) >= 0,1% luas frame = satu orang
BLOB_DILATE_PX = 10                # titik tengah bahu harus di blob tubuh yang didilasi 10 px
SEED_BOOT = 701001
N_BOOT = 2000
SEED_BOARD = 701002
TASK_PATH = ROOT / "models" / "pose_landmarker_heavy.task"
TASK_URL_PIN = TASK_URLS["1"]
TASK_PIN = {"generation": "1682642785425305", "md5_b64": "RT3sTQLMxNPOgStt6E+lFg==", "crc32c_b64": "0kDHiw==", "size": 30664242}
HF_PIN = {
    "pose": {"repo": "facebook/sapiens2-pose-0.4b", "commit": "73572e121602e8a6eb40eb841732e88f386d250b",
             "sha256": "f1b7b0e5423d667634a355dd86a14c5b03d243f2ac77943301c7db9f93329106", "size": 1702754120},
    "normal": {"repo": "facebook/sapiens2-normal-0.4b", "commit": "52886591372f049acd4d59ae6d0a792ae1a61ac5",
               "sha256": "d2f20e8df7186772b7710546416db0752be224c68003c7f18ccbbf8b980ef1fd", "size": 1813476476},
}
HF_FILES = ["model.safetensors", "config.json", "preprocessor_config.json", "README.md"]
LICENSE_TAG = "sapiens2-license"
ARM_NAMES = ("bahu_L", "bahu_R", "siku_L", "siku_R", "pergelangan_L", "pergelangan_R")
# P1 (BlazePose 33, dokumentasi resmi Pose Landmarker): 11/12 bahu, 13/14 siku, 15/16 pergelangan, 23/24 pinggul. Disimpan 11–32 saja.
P1_ARM = (11, 12, 13, 14, 15, 16)
P1_HIP = (23, 24)
P1_KEEP = tuple(range(11, 33))
P1_SEGMENTS = ((11, 13), (13, 15), (12, 14), (14, 16), (11, 12), (11, 23), (12, 24), (23, 24))
# P2 (Sapiens2 keypoints308, facebookresearch/sapiens2@7e5bae88 sapiens/pose/configs/_base_/keypoints308.py: Goliath 344,
# gigi 220–255 dibuang -> 308; indeks < 220 tidak bergeser): 5/6 bahu, 7/8 siku, 62 left_wrist, 41 right_wrist, 9/10 pinggul.
# Disimpan 5–69 (tubuh, kaki, tangan, olecranon / cubital fossa / acromion, leher); DIBUANG 0–4 (hidung / mata / telinga) + 70–307 (wajah).
P2_ARM = (5, 6, 7, 8, 62, 41)
P2_HIP = (9, 10)
P2_KEEP = tuple(range(5, 70))
P2_DRAW = tuple(range(5, 21)) + (41, 62) + tuple(range(63, 70))
P1_VIS_MIN = 0.5
P1_PRES_MIN = 0.5
P2_SCORE_MIN = 0.3
ARM_MIN_OK = 4
HEAT_SAT = 60000.0
P2_FP32_FRAMES = (("klip3", 19), ("klip3", 60), ("klip3", 120), ("klip3", 159), ("test", 129))
VRAM_NEED_MIB = {"p2_fp16": 2500, "p2_fp32": 3300, "n_fp16": 3100, "n_mixed": 3100}
EDGE_ERODE_PX = 2
FACE_MASK_DILATE_PX = 5
EDGE_MAXF = 5
JITTER_MATCH_FRAC = 0.15           # pencocokan orang antar frame: titik tengah bahu <= 15% tinggi orang
PANEL_H = 960
PERSON_BGR = ((40, 40, 230), (30, 170, 30), (230, 120, 20), (200, 40, 200), (20, 200, 230))


def jload(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def save(name: str, obj) -> None:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    (SCRATCH / name).write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"  -> work/t701_scratch/{name}")


# ── hash (F1) ──────────────────────────────────────
def hash_files() -> list[Path]:
    files: list[Path] = []
    for c in HASH_CLIPS:
        for d in HASH_SUBDIRS:
            files += [p for p in sorted((CLIPS_DIR / c / d).rglob("*")) if p.is_file()]
        files.append(CLIPS_DIR / c / "qc_report.json")
    files += [p for p in sorted((ROOT / "out").rglob("*")) if p.is_file()]
    return files


def file_sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cmd_hash(which: str) -> None:
    lines = [f"{file_sha(p)}  {p.relative_to(ROOT).as_posix()}" for p in hash_files()]
    path = OUT / f"{which}_sha256.txt"
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{path.relative_to(ROOT).as_posix()}: {len(lines)} berkas")
    if which.startswith("after"):
        before = (OUT / f"{which.replace('after', 'before', 1)}_sha256.txt").read_text(encoding="utf-8").splitlines()
        print(f"identik dengan before: {before == lines} (before {len(before)} baris, after {len(lines)}, "
              f"selisih himpunan {len(set(before) ^ set(lines))})")


# ── meta (5.1 / 5.2): tanpa mengunduh bobot ────────
def hf_meta(api, repo: str, revision: str | None = None) -> dict:
    from huggingface_hub import get_hf_file_metadata, hf_hub_url
    from huggingface_hub.utils import GatedRepoError, HfHubHTTPError

    info = api.model_info(repo, revision=revision, files_metadata=True)
    files = []
    for s in info.siblings or []:
        lfs = s.lfs
        files.append({"name": s.rfilename, "size": s.size,
                      "lfs_sha256": (lfs.sha256 if lfs is not None else None)})
    # HEAD (bukan GET) per berkas pada commit yang di-pin: membuktikan token cukup untuk berkas itu tanpa mengunduh isinya
    access = {}
    for f in files:
        try:
            m = get_hf_file_metadata(hf_hub_url(repo, f["name"], revision=info.sha))
            access[f["name"]] = {"ok": True, "size": m.size, "etag": m.etag, "commit": m.commit_hash}
        except GatedRepoError as e:
            access[f["name"]] = {"ok": False, "error": f"GatedRepoError: {str(e).splitlines()[0]}"}
        except HfHubHTTPError as e:
            access[f["name"]] = {"ok": False, "error": f"{type(e).__name__}: {str(e).splitlines()[0]}"}
    card = info.card_data.to_dict() if info.card_data is not None else {}
    return {"repo": repo, "sha": info.sha, "last_modified": str(info.last_modified), "gated": info.gated,
            "private": info.private, "disabled": getattr(info, "disabled", None), "license": card.get("license"),
            "license_name": card.get("license_name"), "license_link": card.get("license_link"),
            "pipeline_tag": info.pipeline_tag, "library_name": getattr(info, "library_name", None),
            "files": files, "total_bytes": sum(f["size"] or 0 for f in files), "head": access}


def task_meta() -> dict:
    import requests

    out: dict = {"head": {}, "listing": None}
    for v, url in TASK_URLS.items():
        r = requests.head(url, timeout=30, allow_redirects=False)
        keep = ("Content-Length", "Content-Type", "ETag", "Last-Modified", "x-goog-generation", "x-goog-hash",
                "x-goog-stored-content-length", "x-goog-metageneration")
        out["head"][v] = {"url": url, "status": r.status_code,
                          "headers": {k: r.headers.get(k) for k in keep}, "x-goog-hash_all": r.headers.get_all("x-goog-hash")
                          if hasattr(r.headers, "get_all") else r.headers.get("x-goog-hash")}
    # listing metadata objek (JSON API GCS; hanya metadata, bukan isi)
    r = requests.get(f"https://storage.googleapis.com/storage/v1/b/{TASK_BUCKET}/o",
                     params={"prefix": TASK_PREFIX, "fields": "items(name,size,md5Hash,crc32c,generation,updated,timeCreated)"},
                     timeout=30)
    out["listing_status"] = r.status_code
    if r.ok:
        out["listing"] = r.json().get("items", [])
    return out


def dir_bytes(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.exists() else 0


def cmd_meta() -> None:
    import huggingface_hub as hh
    from huggingface_hub import HfApi, constants

    token = hh.get_token()
    api = HfApi()
    res: dict = {"huggingface_hub": hh.__version__, "token_present": token is not None}
    try:
        who = api.whoami()
        res["whoami_ok"] = True
        res["token_role"] = (who.get("auth", {}) or {}).get("accessToken", {}).get("role")
    except Exception as e:  # noqa: BLE001 — laporan apa adanya
        res["whoami_ok"] = False
        res["whoami_error"] = f"{type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}"
    res["hf"] = {repo: hf_meta(api, repo) for repo in HF_REPOS}
    res["hf_ref_seg_0.4b"] = hf_meta(api, HF_REF_REPO[0], HF_REF_REPO[1])
    res["task"] = task_meta()
    cache = Path(constants.HF_HUB_CACHE)
    du = shutil.disk_usage(cache if cache.exists() else cache.anchor)
    res["disk"] = {"hf_hub_cache": str(cache), "free_bytes": du.free, "total_bytes": du.total,
                   "cache_bytes": dir_bytes(cache),
                   "cache_by_repo": {p.name: dir_bytes(p) for p in sorted(cache.glob("models--*"))} if cache.exists() else {}}
    rdu = shutil.disk_usage(ROOT)
    res["disk"]["repo_drive_free_bytes"] = rdu.free
    save("meta.json", res)
    # ringkas ke layar (tanpa token)
    print(f"token ada: {res['token_present']}, whoami ok: {res.get('whoami_ok')} role: {res.get('token_role')}")
    for repo, m in list(res["hf"].items()) + [(HF_REF_REPO[0] + " (acuan)", res["hf_ref_seg_0.4b"])]:
        print(f"{repo}: sha {m['sha']} gated {m['gated']} private {m['private']} lisensi {m['license']} "
              f"{m['license_name']} total {m['total_bytes']} B, modified {m['last_modified']}")
        for f in m["files"]:
            h = m["head"].get(f["name"], {})
            print(f"    {f['name']}: {f['size']} B sha256 {f['lfs_sha256']} HEAD ok {h.get('ok')} {h.get('error', '')}")
    for v, h in res["task"]["head"].items():
        print(f".task {v}: {h['status']} {h['headers']}")
    print(f"listing GCS: status {res['task']['listing_status']}")
    for it in res["task"]["listing"] or []:
        print(f"    {it}")
    d = res["disk"]
    print(f"disk cache {d['hf_hub_cache']}: bebas {d['free_bytes'] / 2**30:.2f} GiB dari {d['total_bytes'] / 2**30:.2f}; "
          f"cache {d['cache_bytes'] / 2**30:.2f} GiB; drive repo bebas {d['repo_drive_free_bytes'] / 2**30:.2f} GiB")
    for k, v in d["cache_by_repo"].items():
        print(f"    {k}: {v / 2**30:.3f} GiB")


# ── samples (5.3) ──────────────────────────────────
class ClipData:
    def __init__(self, name: str, cfg) -> None:
        self.name = name
        self.work = CLIPS_DIR / name
        self.clip = stb.load_clip(self.work)
        self.classes = jload(self.work / "seg" / "manifest.json")["classes"]
        self.lut = stb.class_group_lut(cfg.groups, self.classes)
        self.group_names = tuple(g for g, _ in cfg.groups)
        cut_names = jload(self.work / "stable" / "manifest.json")["temporal"].get("cut_frames", [])
        self.cuts = sorted(int(Path(c).stem.split("_")[1]) for c in cut_names)
        self.n = len(self.clip.names)
        self.H, self.W = self.clip.height, self.clip.width

    def classmap(self, f: int) -> np.ndarray:
        return cv2.imdecode(np.fromfile(self.work / "seg" / "classmap" / self.clip.names[f], np.uint8), cv2.IMREAD_UNCHANGED)

    def cls(self, name: str) -> int:
        return self.classes.index(name)

    def near_cut(self, f: int) -> bool:
        return any(abs(f - c) <= CUT_MARGIN for c in self.cuts)

    def segments(self) -> list[tuple[int, int]]:
        """Adegan di antara cut (cut k = frame pertama adegan baru)."""
        b = [0] + self.cuts + [self.n]
        return [(b[i], b[i + 1] - 1) for i in range(len(b) - 1)]


def person_boxes(cd: ClipData, cm: np.ndarray) -> list[dict]:
    """Bbox P2 per orang: komponen foreground > qc.blob_min x luas frame (+ komponen >= 30 px ke orang terdekat; klip3_diag.persons_of)."""
    out = []
    for p in persons_of(cm, cd.H, cd.W):
        ys, xs = np.nonzero(p["mask"])
        x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        out.append({"label": p["label"], "bbox_xywh": [x0, y0, x1 - x0 + 1, y1 - y0 + 1], "area": int(p["mask"].sum()),
                    "big_area": p["big_area"], "touches_edge": bool(x0 == 0 or y0 == 0 or x1 >= cd.W - 1 or y1 >= cd.H - 1),
                    "mask": p["mask"]})
    return out


def sleeve_proxy(cd: ClipData, cm: np.ndarray, persons: list[dict]) -> list[dict]:
    """Proksi "lengan baju menempel" per orang (dari seg mentah): Upper_Arm 0 px DAN >= 1 komponen tangan (>= 30 px)
    yang dilasi 5 px menyentuh Upper_Clothing orang itu (lengan baju bersambung langsung ke tangan, tanpa kelas lengan)."""
    up = (cm == cd.cls("Left_Upper_Arm")) | (cm == cd.cls("Right_Upper_Arm"))
    low = (cm == cd.cls("Left_Lower_Arm")) | (cm == cd.cls("Right_Lower_Arm"))
    cloth = cm == cd.cls("Upper_Clothing")
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * HAND_TOUCH_PX + 1, 2 * HAND_TOUCH_PX + 1))
    res = []
    for p in persons:
        pm = p["mask"]
        touch, hands = 0, 0
        for hc in ("Left_Hand", "Right_Hand"):
            n, lab, st, _ = cv2.connectedComponentsWithStats(((cm == cd.cls(hc)) & pm).astype(np.uint8), connectivity=8)
            for i in range(1, n):
                if st[i, cv2.CC_STAT_AREA] < HAND_MIN_PX:
                    continue
                hands += 1
                ring = cv2.dilate((lab == i).astype(np.uint8), k).astype(bool)
                touch += bool((ring & cloth & pm).any())
        r = {"label": p["label"], "upper_arm_px": int((up & pm).sum()), "lower_arm_px": int((low & pm).sum()),
             "upper_cloth_px": int((cloth & pm).sum()), "hands": hands, "hands_touch_cloth": touch}
        r["sleeve_stuck_proxy"] = bool(r["upper_arm_px"] == 0 and touch >= 1)
        res.append(r)
    return res


def arm_front_score(cd: ClipData, R: np.ndarray) -> dict:
    """`test`: fraksi cincin 3 px di sekeliling komponen lengan (>= 200 px) yang berlabel torso; maks atas komponen."""
    torso = cd.group_names.index("torso") + 1
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ARM_RING_PX + 1, 2 * ARM_RING_PX + 1))
    best = {"score": 0.0, "group": None, "area": 0}
    for g in ("left_arm", "right_arm"):
        gi = cd.group_names.index(g) + 1
        n, lab, st, _ = cv2.connectedComponentsWithStats((R == gi).astype(np.uint8), connectivity=8)
        for i in range(1, n):
            if st[i, cv2.CC_STAT_AREA] < ARM_MIN_PX:
                continue
            comp = lab == i
            ring = cv2.dilate(comp.astype(np.uint8), k).astype(bool) & ~comp
            s = float((R[ring] == torso).mean()) if ring.any() else 0.0
            if s > best["score"]:
                best = {"score": s, "group": g, "area": int(st[i, cv2.CC_STAT_AREA])}
    return best


def static_window(cd: ClipData, raw: np.ndarray) -> dict:
    """Jendela statis definisi T-302 (tm.static_windows dari argmax mentah, tanpa melintasi cut); kosong -> fallback
    STATIC_LEN frame dengan rata-rata kecepatan centroid terendah (preseden scripts/klip2_review.py::pick_windows)."""
    sp = tm.centroid_speed(raw)
    clean = lambda w: not any(w[0] < k <= w[1] for k in cd.cuts)  # noqa: E731
    auto = [w for w in tm.static_windows(raw) if clean(w)]
    info: dict = {"auto": [list(w) for w in auto], "rule": {"speed_max_px": tm.STATIC_SPEED_MAX_PX, "xor_max": tm.STATIC_XOR_MAX,
                                                          "min_len": tm.RUN_MIN_FRAMES}}
    if auto and max(b - a + 1 for a, b in auto) >= STATIC_LEN:
        a, b = max(auto, key=lambda w: w[1] - w[0])
        s = min(range(a, b - STATIC_LEN + 2), key=lambda i: sp[max(i, 1):i + STATIC_LEN].mean())
        info["window"], info["source"] = [s, s + STATIC_LEN - 1], f"otomatis T-302 ({a}-{b}), sub-jendela {STATIC_LEN} frame kecepatan terendah"
    elif auto:
        info["window"], info["source"] = list(max(auto, key=lambda w: w[1] - w[0])), "otomatis T-302 (< 12 frame)"
    else:
        starts = [i for i in range(cd.n - STATIC_LEN + 1) if clean((i, i + STATIC_LEN - 1))]
        s = min(starts, key=lambda i: sp[max(i, 1):i + STATIC_LEN].mean())
        info["window"], info["source"] = [s, s + STATIC_LEN - 1], f"fallback: {STATIC_LEN} frame rata-rata kecepatan terendah, tanpa cut"
    w0, w1 = info["window"]
    info["speed_mean_px"] = float(sp[max(w0, 1):w1 + 1].mean())
    info["xor_mean"] = float(tm.xor_fraction(raw)[max(w0, 1):w1 + 1].mean())
    return info


def strip(persons: list[dict]) -> list[dict]:
    return [{k: v for k, v in p.items() if k != "mask"} for p in persons]


def bbox_iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def unit_match(cd: ClipData, u: dict) -> dict:
    """Strok oklusi di contours/ SEKARANG pada frame wakil dengan bbox paling mirip (label dibuat di contours T-305b)."""
    d = jload(cd.work / "contours" / f"frame_{u['rep_frame']:05d}.json")
    best = {"iou": 0.0, "track_id": None, "centroid_dist_px": None}
    for s in d["strokes"]:
        if s["type"] != "occlusion":
            continue
        p = np.asarray(s["points"], float)
        bb = [p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()]
        iou = bbox_iou(bb, u["rep_stroke_bbox_xyxy"])
        if iou > best["iou"]:
            c = p.mean(axis=0)
            best = {"iou": round(iou, 4), "track_id": s["track_id"],
                    "centroid_dist_px": round(float(np.hypot(*(c - np.asarray(u["rep_stroke_centroid_xy"])))), 2),
                    "points": p}
    best["matched"] = best["iou"] >= MATCH_IOU
    return best


def cmd_samples() -> None:
    cfg = load_pipeline()
    res: dict = {"params": {"cut_margin": CUT_MARGIN, "blob_min": cfg.qc.blob_min, "assign_min_px": 30,
                            "hand_touch_px": HAND_TOUCH_PX, "hand_min_px": HAND_MIN_PX, "arm_min_px": ARM_MIN_PX,
                            "arm_ring_px": ARM_RING_PX, "arm_front_min": ARM_FRONT_MIN, "test_min_gap": TEST_MIN_GAP,
                            "static_len": STATIC_LEN, "match_iou": MATCH_IOU, "p2_box": "bbox ketat orang; processor Sapiens2 "
                            "menambah padding 1,25 dan koreksi rasio 3:4 (768x1024)"}}

    # klip3
    k3 = ClipData("klip3", cfg)
    raw3, rows3 = [], {}
    for f in range(k3.n):
        cm = k3.classmap(f)
        raw3.append(k3.lut[cm])
        ps = person_boxes(k3, cm)
        sp = sleeve_proxy(k3, cm, ps)
        rows3[f] = {"n_persons": len(ps), "n_sleeve": sum(r["sleeve_stuck_proxy"] for r in sp), "persons": strip(ps), "sleeve": sp,
                    "near_cut": k3.near_cut(f)}
    raw3 = np.stack(raw3)
    segs = k3.segments()
    seg_of = {f: i for i, (a, b) in enumerate(segs) for f in range(a, b + 1)}
    picked = [f for f in DIAG2_FRAMES if not rows3[f]["near_cut"]]
    used_segs = {seg_of[f] for f in picked}
    seg_cands = []
    for i, (a, b) in enumerate(segs):
        c = [f for f in range(a, b + 1) if not rows3[f]["near_cut"] and rows3[f]["n_sleeve"] >= 1]
        seg_cands.append({"segment": [a, b], "n_frames": b - a + 1, "n_candidates": len(c),
                          "n_ok_not_near_cut": sum(not rows3[f]["near_cut"] for f in range(a, b + 1))})
        if i in used_segs or not c:
            continue
        mid = (a + b) / 2
        best = max(c, key=lambda f: (rows3[f]["n_sleeve"], rows3[f]["n_sleeve"] == rows3[f]["n_persons"], -abs(f - mid)))
        seg_cands[-1]["pick"] = best
    extra = sorted((s for s in seg_cands if "pick" in s), key=lambda s: -s["n_frames"])
    for s in extra:
        if len(picked) >= N_KLIP3:
            break
        picked.append(s["pick"])
    # adegan panjang dapat sampel kedua bila kuota belum penuh (jarak >= 8 frame dari sampel lain)
    for s in extra:
        if len(picked) >= N_KLIP3:
            break
        a, b = s["segment"]
        c = [f for f in range(a, b + 1) if not rows3[f]["near_cut"] and rows3[f]["n_sleeve"] >= 1
             and all(abs(f - q) >= 8 for q in picked)]
        if c:
            picked.append(max(c, key=lambda f: (rows3[f]["n_sleeve"], -abs(f - (a + b) / 2))))
    picked = sorted(picked)
    res["klip3"] = {"cuts": k3.cuts, "segments": seg_cands, "frames": picked,
                    "rows": {f: rows3[f] for f in picked},
                    "proxy_clip": {"frames_with_sleeve_proxy": sum(r["n_sleeve"] >= 1 for r in rows3.values()),
                                   "n_persons_hist": np.bincount([r["n_persons"] for r in rows3.values()]).tolist(),
                                   "n_sleeve_hist": np.bincount([r["n_sleeve"] for r in rows3.values()]).tolist()},
                    "static": static_window(k3, raw3)}
    w0, w1 = res["klip3"]["static"]["window"]
    res["klip3"]["static"]["n_persons"] = [rows3[f]["n_persons"] for f in range(w0, w1 + 1)]

    # test
    kt = ClipData("test", cfg)
    rawt, scores = [], {}
    for f in range(kt.n):
        R = kt.lut[kt.classmap(f)]
        rawt.append(R)
        scores[f] = arm_front_score(kt, R)
    rawt = np.stack(rawt)
    order = sorted((f for f in scores if not kt.near_cut(f) and scores[f]["score"] >= ARM_FRONT_MIN),
                   key=lambda f: -scores[f]["score"])
    tpick: list[int] = []
    for f in order:
        if all(abs(f - q) >= TEST_MIN_GAP for q in tpick):
            tpick.append(f)
        if len(tpick) >= N_TEST:
            break
    tpick.sort()
    trows = {}
    for f in tpick:
        cm = kt.classmap(f)
        trows[f] = {"arm_front": scores[f], "persons": strip(person_boxes(kt, cm))}
    sc = np.array([scores[f]["score"] for f in range(kt.n)])
    res["test"] = {"cuts": kt.cuts, "frames": tpick, "rows": trows,
                   "score_clip": {"n_ge_min": int((sc >= ARM_FRONT_MIN).sum()), "p50": float(np.median(sc)),
                                  "p95": float(np.percentile(sc, 95)), "max": float(sc.max())},
                   "static": static_window(kt, rawt)}

    # unit T-305c (O / L) -> frame wakil untuk N
    lab = jload(LABELS)
    units = {}
    for num, u in sorted(lab["units"].items()):
        if u["label"] not in ("O", "L") or u["clip"] not in ("test", "klip3"):
            continue
        cd = k3 if u["clip"] == "klip3" else kt
        units[num] = {"clip": u["clip"], "label": u["label"], "rep_frame": u["rep_frame"], "group": u["group"],
                      "near_cut": cd.near_cut(u["rep_frame"]),
                      "match_now": {k: v for k, v in unit_match(cd, u).items() if k != "points"}}
    nf = {c: sorted({u["rep_frame"] for u in units.values() if u["clip"] == c}) for c in ("klip3", "test")}
    res["n_units"] = {"units": units, "frames": nf,
                      "counts": {c: {lb: sum(u["clip"] == c and u["label"] == lb for u in units.values()) for lb in "OL"}
                                 for c in ("klip3", "test")},
                      "matched": {c: sum(u["clip"] == c and u["match_now"]["matched"] for u in units.values()) for c in ("klip3", "test")}}
    allf = {c: sorted(set(res[c]["frames"]) | set(nf[c])) for c in ("klip3", "test")}
    res["union"] = {c: {"frames": allf[c], "n": len(allf[c])} for c in allf}
    save("samples.json", res)

    print(f"klip3 cut {k3.cuts}; segmen (frame, kandidat): " + ", ".join(f"{s['segment']}:{s['n_candidates']}" for s in seg_cands))
    print(f"klip3 proksi lengan baju: frame dgn >=1 orang {res['klip3']['proxy_clip']['frames_with_sleeve_proxy']}/{k3.n}; "
          f"hist orang {res['klip3']['proxy_clip']['n_persons_hist']}, hist orang-proksi {res['klip3']['proxy_clip']['n_sleeve_hist']}")
    print(f"klip3 sampel ({len(picked)}): {picked}")
    for f in picked:
        r = rows3[f]
        print(f"  f{f}: orang {r['n_persons']}, proksi {r['n_sleeve']} | " + "; ".join(
            f"{p['label']} bbox {q['bbox_xywh']} upA {s['upper_arm_px']} lowA {s['lower_arm_px']} tangan {s['hands']}/{s['hands_touch_cloth']}"
            for p, q, s in zip(r["persons"], r["persons"], r["sleeve"])))
    print(f"klip3 statis: {res['klip3']['static']}")
    print(f"test skor lengan-depan: {res['test']['score_clip']}")
    print(f"test sampel ({len(tpick)}): {tpick}")
    for f in tpick:
        print(f"  f{f}: {trows[f]['arm_front']} orang {[p['bbox_xywh'] for p in trows[f]['persons']]}")
    print(f"test statis: {res['test']['static']}")
    print(f"unit N: {res['n_units']['counts']}, cocok sekarang {res['n_units']['matched']}, frame klip3 {len(nf['klip3'])} test {len(nf['test'])}")
    for num, u in units.items():
        if not u["match_now"]["matched"] or u["near_cut"]:
            print(f"  unit {num}: {u}")
    print(f"gabungan frame unik: klip3 {res['union']['klip3']['n']}, test {res['union']['test']['n']}")


# ══ Tahap 1b ═══════════════════════════════════════
def _crc32c_table() -> list[int]:
    table = []
    for i in range(256):
        c = i
        for _ in range(8):
            c = (c >> 1) ^ 0x82F63B78 if c & 1 else c >> 1
        table.append(c)
    return table


def crc32c_update(crc: int, data: bytes, table: list[int]) -> int:
    for b in data:
        crc = table[(crc ^ b) & 0xFF] ^ (crc >> 8)
    return crc


def cmd_download() -> None:
    import requests
    from huggingface_hub import snapshot_download

    res: dict = {"task": {}, "hf": {}}
    # P1: .task berversi (float16/1); diverifikasi md5 + crc32c terhadap nilai GCS Tahap 1a sebelum dipindah ke models/
    tmp = TASK_PATH.with_name(TASK_PATH.name + ".part")
    table = _crc32c_table()
    md5, sha, crc = hashlib.md5(), hashlib.sha256(), 0xFFFFFFFF
    t0 = time.perf_counter()
    if TASK_PATH.exists():
        src, gen = TASK_PATH, "(sudah ada; diverifikasi ulang)"
        with TASK_PATH.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                md5.update(chunk), sha.update(chunk)
                crc = crc32c_update(crc, chunk, table)
    else:
        src = tmp
        with requests.get(TASK_URL_PIN, stream=True, timeout=60) as r:
            r.raise_for_status()
            gen = r.headers.get("x-goog-generation")
            with tmp.open("wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
                    md5.update(chunk), sha.update(chunk)
                    crc = crc32c_update(crc, chunk, table)
    md5_b64 = base64.b64encode(md5.digest()).decode()
    crc_b64 = base64.b64encode((crc ^ 0xFFFFFFFF).to_bytes(4, "big")).decode()
    size = src.stat().st_size
    checks = {"generation": gen == TASK_PIN["generation"] or src == TASK_PATH, "md5": md5_b64 == TASK_PIN["md5_b64"],
              "crc32c": crc_b64 == TASK_PIN["crc32c_b64"], "size": size == TASK_PIN["size"]}
    res["task"] = {"url": TASK_URL_PIN, "path": str(TASK_PATH.relative_to(ROOT)), "x-goog-generation": gen, "size": size,
                   "md5_b64": md5_b64, "crc32c_b64": crc_b64, "sha256": sha.hexdigest(), "checks": checks,
                   "time_s": round(time.perf_counter() - t0, 1)}
    if not all(checks.values()):
        if src == tmp:
            tmp.unlink()
        save("download.json", res)
        raise SystemExit(f".task GAGAL verifikasi {checks} — berkas tidak dipakai")
    if src == tmp:
        tmp.replace(TASK_PATH)
    print(f".task: {size} B md5 {md5_b64} crc32c {crc_b64} sha256 {sha.hexdigest()} cek {checks}")
    # P2 / N: commit yang di-pin, hanya model.safetensors + config + processor + README (lisensi)
    for key, pin in HF_PIN.items():
        t0 = time.perf_counter()
        p = Path(snapshot_download(pin["repo"], revision=pin["commit"], allow_patterns=HF_FILES))
        sizes = {f: (p / f).stat().st_size for f in HF_FILES}
        digest = file_sha(p / "model.safetensors")
        readme = (p / "README.md").read_text(encoding="utf-8", errors="replace")
        head = readme.split("---")[1] if readme.startswith("---") else ""
        checks = {"sha256": digest == pin["sha256"], "size": sizes["model.safetensors"] == pin["size"],
                  "license_tag": LICENSE_TAG in head, "snapshot_commit": p.name == pin["commit"]}
        res["hf"][key] = {"repo": pin["repo"], "commit": pin["commit"], "snapshot": str(p), "sizes": sizes,
                          "total_bytes": sum(sizes.values()), "sha256": digest, "checks": checks,
                          "time_s": round(time.perf_counter() - t0, 1)}
        print(f"{pin['repo']}@{pin['commit'][:8]}: {sizes} sha256 {digest} cek {checks}")
        if not all(checks.values()):
            save("download.json", res)
            raise SystemExit(f"{pin['repo']} GAGAL verifikasi {checks}")
    save("download.json", res)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "download.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")


def read_rgb(cd: ClipData, f: int) -> np.ndarray:
    bgr = cv2.imdecode(np.fromfile(cd.work / "frames" / cd.clip.names[f], np.uint8), cv2.IMREAD_COLOR)
    return np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def frame_people(cd: ClipData, f: int) -> tuple[list[dict], list[dict]]:
    """(blob tubuh dengan masker, komponen wajah) — definisi docs/04 "Keputusan T-701 Tahap 1b"."""
    cm = cd.classmap(f)
    blobs = person_boxes(cd, cm)
    face_id = cd.group_names.index("face") + 1
    n, lab, st, cen = cv2.connectedComponentsWithStats((cd.lut[cm] == face_id).astype(np.uint8), connectivity=8)
    amin = FACE_MIN_FRAC * cd.H * cd.W
    faces = []
    for i in range(1, n):
        a = int(st[i, cv2.CC_STAT_AREA])
        if a < amin:
            continue
        comp = lab == i
        inside = [int((comp & b["mask"]).sum()) for b in blobs]
        bi = int(np.argmax(inside)) if inside and max(inside) * 2 > a else None
        faces.append({"area": a, "centroid_xy": [float(cen[i][0]), float(cen[i][1])],
                      "bbox_xywh": [int(v) for v in st[i, :4]], "blob": bi})
    for k, b in enumerate(blobs):
        b["n_faces"] = sum(fc["blob"] == k for fc in faces)
        b["merged"] = b["n_faces"] >= 2
    return blobs, faces


def people_summary(blobs: list[dict], faces: list[dict]) -> dict:
    return {"n_faces": len(faces), "n_blobs": len(blobs), "merged_blob": any(b["merged"] for b in blobs),
            "faces": faces, "blobs": strip(blobs)}


def cmd_samples1b() -> None:
    cfg = load_pipeline()
    s1 = jload(SCRATCH / "samples.json")
    k3, kt = ClipData("klip3", cfg), ClipData("test", cfg)
    base = list(s1["klip3"]["frames"])
    nsl = {}
    for f in range(k3.n):
        if k3.near_cut(f):
            continue
        cm = k3.classmap(f)
        nsl[f] = sum(s["sleeve_stuck_proxy"] for s in sleeve_proxy(k3, cm, person_boxes(k3, cm)))
    segs = k3.segments()
    seg_valid = [[f for f in range(a, b + 1) if nsl.get(f, 0) >= 1] for a, b in segs]
    picks = [sorted(f for f in base if a <= f <= b) for a, b in segs]
    added: list[int] = []
    # (1) adegan tanpa sampel dulu; (2) prioritas = frame valid / (sampel + 1), frame terjauh dari sampel adegan itu
    for i, (a, b) in enumerate(segs):
        if not picks[i] and seg_valid[i] and len(base) + len(added) < N_KLIP3_1B:
            mid = (a + b) / 2
            f = max(seg_valid[i], key=lambda f: (nsl[f], -abs(f - mid), -f))
            picks[i].append(f)
            added.append(f)
    while len(base) + len(added) < N_KLIP3_1B:
        cand = [i for i in range(len(segs)) if any(f not in picks[i] for f in seg_valid[i])]
        i = max(cand, key=lambda i: (len(seg_valid[i]) / (len(picks[i]) + 1), -i))
        free = [f for f in seg_valid[i] if f not in picks[i]]
        mid = (segs[i][0] + segs[i][1]) / 2
        f = max(free, key=lambda f: (min(abs(f - q) for q in picks[i]) if picks[i] else -abs(f - mid), -f))
        picks[i].append(f)
        added.append(f)
    k3_frames = sorted(base + added)
    samples = {"klip3": k3_frames, "test": list(s1["test"]["frames"])}
    static = {"klip3": s1["klip3"]["static"]["window"], "test": s1["test"]["static"]["window"]}
    unit_frames = s1["n_units"]["frames"]
    n_frames = {c: sorted(set(samples[c]) | set(unit_frames[c])) for c in samples}
    per_frame: dict = {}
    for c, cd in (("klip3", k3), ("test", kt)):
        need = sorted(set(samples[c]) | set(range(static[c][0], static[c][1] + 1)))
        per_frame[c] = {f: people_summary(*frame_people(cd, f)) for f in need}
    res = {"samples": samples, "klip3_added": sorted(added), "klip3_base_1a": base, "static": static, "n_frames": n_frames,
           "segments": [{"segment": [a, b], "valid": len(v), "picks": sorted(p)} for (a, b), v, p in zip(segs, seg_valid, picks)],
           "per_frame": per_frame, "params": {"face_min_frac": FACE_MIN_FRAC, "face_min_px": {c: FACE_MIN_FRAC * cd.H * cd.W
                                                                                               for c, cd in (("klip3", k3), ("test", kt))}}}
    save("samples_1b.json", res)
    print(f"klip3 20: {k3_frames} (tambahan {sorted(added)})")
    print(f"test 8: {samples['test']}; frame N: klip3 {len(n_frames['klip3'])}, test {len(n_frames['test'])}")
    for c in ("klip3", "test"):
        for f in samples[c]:
            r = per_frame[c][f]
            print(f"  {c} f{f}: wajah {r['n_faces']} blob {r['n_blobs']} gabungan {r['merged_blob']} "
                  f"luas wajah {[x['area'] for x in r['faces']]} blob-wajah {[x['blob'] for x in r['faces']]}")


def cds_load(cfg) -> dict:
    return {c: ClipData(c, cfg) for c in ("klip3", "test")}


def p1_rows(result, W: int, H: int) -> list[list[list[float]]]:
    def val(v):
        return float(v) if v is not None else float("nan")
    return [[[i, lm[i].x * W, lm[i].y * H, val(lm[i].visibility), val(lm[i].presence)] for i in P1_KEEP]
            for lm in result.pose_landmarks]


def cmd_p1() -> None:
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    s = jload(SCRATCH / "samples_1b.json")
    cds = cds_load(load_pipeline())
    max_faces = max(s["per_frame"][c][str(f)]["n_faces"] for c in cds for f in s["samples"][c])
    num_poses = max_faces + 1

    def make(mode):
        return vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(TASK_PATH)), running_mode=mode, num_poses=num_poses))

    res: dict = {"num_poses": num_poses, "mediapipe": mp.__version__, "task_sha256": file_sha(TASK_PATH),
                 "thresholds": {"min_pose_detection_confidence": 0.5, "min_pose_presence_confidence": 0.5,
                                "min_tracking_confidence": 0.5}, "image": {}, "video": {}}
    lm = make(vision.RunningMode.IMAGE)
    try:
        c0 = next(iter(cds))
        lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=read_rgb(cds[c0], s["samples"][c0][0])))  # pemanasan, tidak dicatat
        for c, cd in cds.items():
            for f in s["samples"][c]:
                img = mp.Image(image_format=mp.ImageFormat.SRGB, data=read_rgb(cd, f))
                t0 = time.perf_counter()
                r = lm.detect(img)
                res["image"][f"{c}:{f}"] = {"time_s": time.perf_counter() - t0, "poses": p1_rows(r, cd.W, cd.H)}
    finally:
        lm.close()
    for c, cd in cds.items():
        fps = jload(cd.work / "meta.json")["target_fps"]
        w0, w1 = s["static"][c]
        lm = make(vision.RunningMode.VIDEO)
        try:
            for f in range(w0, w1 + 1):
                img = mp.Image(image_format=mp.ImageFormat.SRGB, data=read_rgb(cd, f))
                t0 = time.perf_counter()
                r = lm.detect_for_video(img, int(round(f * 1000 / fps)))
                res["video"][f"{c}:{f}"] = {"time_s": time.perf_counter() - t0, "poses": p1_rows(r, cd.W, cd.H)}
        finally:
            lm.close()
    save("p1.json", res)
    print(f"P1 num_poses {num_poses}; pose per frame sampel: " +
          ", ".join(f"{k}={len(v['poses'])}" for k, v in res["image"].items()))


def gpu_check(torch, key: str) -> tuple[float, float]:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA tidak tersedia — butuh GPU NVIDIA lewat PyTorch cu118")
    free, total = torch.cuda.mem_get_info()
    need = VRAM_NEED_MIB[key]
    if free / MIB < need:
        raise SystemExit(f"VRAM bebas {free / MIB:.0f} MiB < dibutuhkan {need} MiB untuk {key} — tutup aplikasi lain "
                         f"yang memakai GPU lalu jalankan ulang (model belum dimuat)")
    return free / MIB, total / MIB


def hf_load(key: str, cls_name: str, dtype, device: str):
    import transformers
    from transformers import AutoImageProcessor

    pin = HF_PIN[key]
    kw = {"revision": pin["commit"], "local_files_only": True}
    proc = AutoImageProcessor.from_pretrained(pin["repo"], **kw)
    model = getattr(transformers, cls_name).from_pretrained(pin["repo"], dtype=dtype, **kw)
    return proc, model.to(device).eval()


def cmd_p2(prec: str) -> None:
    import torch

    s = jload(SCRATCH / "samples_1b.json")
    cds = cds_load(load_pipeline())
    free, total = gpu_check(torch, f"p2_{prec}")
    dtype = torch.float16 if prec == "fp16" else torch.float32
    t0 = time.perf_counter()
    proc, model = hf_load("pose", "Sapiens2ForPoseEstimation", dtype, "cuda")
    res: dict = {"precision": prec, "vram_free_before_load_mib": round(free, 1), "vram_total_mib": round(total, 1),
                 "load_s": round(time.perf_counter() - t0, 2),
                 "vram_free_after_load_mib": round(torch.cuda.mem_get_info()[0] / MIB, 1),
                 "kept_indices": list(P2_KEEP), "oom": None, "frames": {}}
    if prec == "fp16":
        frames = [(c, f) for c in cds for f in s["samples"][c]]
        frames += [(c, f) for c in cds for f in range(s["static"][c][0], s["static"][c][1] + 1)]
    else:
        frames = list(P2_FP32_FRAMES)
    frames = list(dict.fromkeys(frames))
    keep = torch.tensor(P2_KEEP, device="cuda")
    for c, f in frames:
        cd = cds[c]
        rgb = read_rgb(cd, f)
        blobs = person_boxes(cd, cd.classmap(f))
        torch.cuda.reset_peak_memory_stats()
        persons, tf = [], 0.0
        try:
            for b in blobs:
                box = [float(v) for v in b["bbox_xywh"]]
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                inp = proc(images=rgb, boxes=[[box]], return_tensors="pt")
                pix = inp["pixel_values"].to("cuda", dtype)
                with torch.inference_mode():
                    hm = model(pixel_values=pix).heatmaps
                    st = {"nan": int(torch.isnan(hm).sum()), "inf": int(torch.isinf(hm).sum()),
                          "sat": int((hm.abs() >= HEAT_SAT).sum()), "absmax": float(hm.float().nan_to_num(0.0).abs().amax())}
                    body = hm.index_select(1, keep)        # titik wajah / kepala dibuang DI SINI, sebelum post-process
                    del hm
                    pp = proc.post_process_pose_estimation(SimpleNamespace(heatmaps=body), boxes=[[box]])
                    kp = pp[0][0]["keypoints"].float().cpu().numpy()
                    sc = pp[0][0]["scores"].float().cpu().numpy()
                    del body, pp
                del inp, pix
                torch.cuda.synchronize()
                dt = time.perf_counter() - t1
                tf += dt
                persons.append({"blob_bbox_xywh": b["bbox_xywh"], "time_s": dt, **st,
                                "kp": [[P2_KEEP[j], float(kp[j, 0]), float(kp[j, 1]), float(sc[j])] for j in range(len(P2_KEEP))]})
        except torch.cuda.OutOfMemoryError:
            res["oom"] = {"frame": f"{c}:{f}", "reserved_mib": round(torch.cuda.memory_reserved() / MIB, 1)}
            print(f"OOM di {c}:{f}")
            break
        res["frames"][f"{c}:{f}"] = {"time_s": tf, "n_boxes": len(blobs), "persons": persons,
                                     "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / MIB, 1),
                                     "peak_alloc_mib": round(torch.cuda.max_memory_allocated() / MIB, 1)}
        print(f"  {c}:{f} kotak {len(blobs)} {tf:.2f} s, puncak reserved {res['frames'][f'{c}:{f}']['peak_reserved_mib']} MiB, "
              f"NaN {sum(p['nan'] for p in persons)} inf {sum(p['inf'] for p in persons)}", flush=True)
    save(f"p2_{prec}.json", res)


def normal_forward(model, pix, mode: str):
    if mode != "mixed":
        return model(pixel_values=pix).normals
    cfgm = model.config
    out = model.model(pix)                                     # backbone fp16
    b, _, h, w = pix.shape
    ps = cfgm.patch_size if isinstance(cfgm.patch_size, int) else cfgm.patch_size[0]
    tok = out.last_hidden_state[:, 1 + cfgm.num_register_tokens:]
    fmap = tok.transpose(1, 2).reshape(b, -1, h // ps, w // ps).float()
    return model.decode_head(fmap)                             # head fp32


def edge_map(n: np.ndarray, fg: np.ndarray) -> np.ndarray:
    """Sudut (derajat) antar normal tetangga kanan / bawah, maks keduanya; hanya bila kedua piksel di fg. Luar fg = NaN."""
    e = np.zeros(fg.shape, np.float32)
    ax = np.degrees(np.arccos(np.clip((n[:, :-1] * n[:, 1:]).sum(-1), -1.0, 1.0)))
    e[:, :-1] = np.where(fg[:, :-1] & fg[:, 1:], ax, 0.0)
    ay = np.degrees(np.arccos(np.clip((n[:-1] * n[1:]).sum(-1), -1.0, 1.0)))
    e[:-1] = np.maximum(e[:-1], np.where(fg[:-1] & fg[1:], ay, 0.0))
    return np.where(fg, e, np.nan).astype(np.float32)


def cmd_normal(mode: str) -> None:
    import torch
    import torch.nn.functional as F
    from transformers.image_utils import get_image_size_for_max_height_width

    s = jload(SCRATCH / "samples_1b.json")
    cds = cds_load(load_pipeline())
    outdir = SCRATCH / f"normal_{mode}"
    outdir.mkdir(parents=True, exist_ok=True)
    res: dict = {"mode": mode, "oom": None, "frames": {}}
    t0 = time.perf_counter()
    if mode == "cpu":
        device, in_dtype = "cpu", torch.float32
        proc, model = hf_load("normal", "Sapiens2ForNormalEstimation", torch.float32, "cpu")
        res["threads"] = torch.get_num_threads()
    else:
        free, total = gpu_check(torch, f"n_{mode}")
        res.update({"vram_free_before_load_mib": round(free, 1), "vram_total_mib": round(total, 1)})
        device, in_dtype = "cuda", torch.float16
        if mode == "fp16":
            proc, model = hf_load("normal", "Sapiens2ForNormalEstimation", torch.float16, "cuda")
        else:
            proc, model = hf_load("normal", "Sapiens2ForNormalEstimation", torch.float32, "cpu")
            model.model.half()
            model = model.to("cuda").eval()
        res["vram_free_after_load_mib"] = round(torch.cuda.mem_get_info()[0] / MIB, 1)
    res["load_s"] = round(time.perf_counter() - t0, 2)
    size_h, size_w = int(proc.size["height"]), int(proc.size["width"])
    mean = torch.tensor(proc.image_mean).view(1, 3, 1, 1)
    std = torch.tensor(proc.image_std).view(1, 3, 1, 1)
    k_er = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * EDGE_ERODE_PX + 1, 2 * EDGE_ERODE_PX + 1))
    k_face = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * FACE_MASK_DILATE_PX + 1, 2 * FACE_MASK_DILATE_PX + 1))
    frames = [(c, f) for c in cds for f in s["n_frames"][c]]
    for c, f in frames:
        cd = cds[c]
        rgb = read_rgb(cd, f)
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        try:
            pix_cpu = proc(images=rgb, do_pad=True, return_tensors="pt")["pixel_values"]
            nh, nw = get_image_size_for_max_height_width((cd.H, cd.W), size_h, size_w)
            top, left = (size_h - nh) // 2, (size_w - nw) // 2
            zero = (-mean / std).expand(1, 3, size_h, size_w)
            pad = torch.ones(size_h, size_w, dtype=torch.bool)
            pad[top:top + nh, left:left + nw] = False
            pad_ok = bool(torch.allclose(pix_cpu[..., pad], zero[..., pad], atol=1e-4)) if pad.any() else True
            pix = pix_cpu.to(device, in_dtype)
            with torch.inference_mode():
                raw = normal_forward(model, pix, mode).float()
                n_nan, n_inf = int(torch.isnan(raw).sum()), int(torch.isinf(raw).sum())
                crop = raw[:, :, top:top + nh, left:left + nw]
                up = F.interpolate(crop, size=(cd.H, cd.W), mode="bilinear", align_corners=False)[0]
                nrm = up / up.norm(dim=0, keepdim=True).clamp_min(1e-6)
                n = nrm.permute(1, 2, 0).cpu().numpy()
                del raw, crop, up, nrm, pix
            if device == "cuda":
                torch.cuda.synchronize()
            dt = time.perf_counter() - t1
        except torch.cuda.OutOfMemoryError:
            res["oom"] = {"frame": f"{c}:{f}", "reserved_mib": round(torch.cuda.memory_reserved() / MIB, 1)}
            print(f"OOM di {c}:{f}")
            break
        groups = stb.read_groups(cd.clip.groups_path(cd.clip.names[f]))
        fg = groups != 0
        fg_er = cv2.erode(fg.astype(np.uint8), k_er).astype(bool)
        face = cv2.dilate((groups == cd.group_names.index("face") + 1).astype(np.uint8), k_face).astype(bool)
        e = edge_map(n, fg_er)
        n[face] = np.nan                                       # masking wajah SEBELUM disimpan
        e[face] = np.nan
        code = np.zeros(fg.shape, np.uint8)
        code[fg] = 3
        code[fg_er] = 1
        code[face] = 2
        np.savez_compressed(outdir / f"{c}_{f:05d}.npz", normals=n.astype(np.float16), E=e.astype(np.float16), code=code)
        row = {"time_s": dt, "nan": n_nan, "inf": n_inf, "pad_ok": pad_ok, "valid_hw": [nh, nw], "pad_tl": [top, left],
               "E_p50": float(np.nanmedian(e)) if np.isfinite(e).any() else None,
               "E_p95": float(np.nanpercentile(e, 95)) if np.isfinite(e).any() else None}
        if device == "cuda":
            row.update({"peak_reserved_mib": round(torch.cuda.max_memory_reserved() / MIB, 1),
                        "peak_alloc_mib": round(torch.cuda.max_memory_allocated() / MIB, 1)})
        res["frames"][f"{c}:{f}"] = row
        print(f"  {c}:{f} {dt:.2f} s NaN {n_nan} inf {n_inf} pad_ok {pad_ok} puncak {row.get('peak_reserved_mib')} MiB "
              f"E p50/p95 {row['E_p50']} / {row['E_p95']}", flush=True)
    res["n_frames_done"] = len(res["frames"])
    res["n_frames_total"] = len(frames)
    res["nan_frames"] = sum(r["nan"] > 0 or r["inf"] > 0 for r in res["frames"].values())
    save(f"normal_{mode}.json", res)
    print(f"N {mode}: {res['n_frames_done']}/{len(frames)} frame, frame NaN/inf {res['nan_frames']}, OOM {res['oom']}")


# ── metrik ─────────────────────────────────────────
def pct(a, q):
    a = np.asarray([v for v in a if v is not None and np.isfinite(v)], float)
    return float(np.percentile(a, q)) if len(a) else None


def p1_det(rows: list) -> tuple[bool, list[float], dict]:
    d = {r[0]: r for r in rows}
    vis = [d[i][3] for i in P1_ARM]
    pres = [d[i][4] for i in P1_ARM]
    ok = (np.nanmean(pres) >= P1_PRES_MIN) and sum(v >= P1_VIS_MIN for v in vis) >= ARM_MIN_OK
    mid = [(d[11][1] + d[12][1]) / 2, (d[11][2] + d[12][2]) / 2]
    return bool(ok), mid, d


def p2_det(person: dict) -> tuple[bool, list[float], dict]:
    d = {r[0]: r for r in person["kp"]}
    sc = [d[i][3] for i in P2_ARM]
    ok = sum(v >= P2_SCORE_MIN for v in sc) >= ARM_MIN_OK
    mid = [(d[5][1] + d[6][1]) / 2, (d[5][2] + d[6][2]) / 2]
    return bool(ok), mid, d


def match_faces(mids: list, faces: list[dict], blobs: list[dict], H: int, W: int) -> int:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * BLOB_DILATE_PX + 1, 2 * BLOB_DILATE_PX + 1))
    dil = [cv2.dilate(b["mask"].astype(np.uint8), k).astype(bool) for b in blobs]
    pairs = []
    for i, (x, y) in enumerate(mids):
        xi, yi = int(round(x)), int(round(y))
        if not (0 <= xi < W and 0 <= yi < H):
            continue
        for j, fc in enumerate(faces):
            if fc["blob"] is not None and dil[fc["blob"]][yi, xi]:
                pairs.append((float(np.hypot(x - fc["centroid_xy"][0], y - fc["centroid_xy"][1])), i, j))
    used_i, used_j = set(), set()
    for _, i, j in sorted(pairs):
        if i not in used_i and j not in used_j:
            used_i.add(i)
            used_j.add(j)
    return len(used_j)


def jitter(seq: list[list[tuple[list[float], dict, float]]], arm: tuple, thr_idx: int, thr: float) -> list[float]:
    """seq[t] = daftar (titik tengah bahu, peta titik, tinggi orang). Lacak orang antar frame, lalu |p_t − (p_{t−1}+p_{t+1})/2|."""
    tracks: list[dict] = []
    for t, people in enumerate(seq):
        used = set()
        for mid, d, hgt in people:
            best, bd = None, None
            for k, tr in enumerate(tracks):
                if k in used or tr["last_t"] != t - 1:
                    continue
                dist = float(np.hypot(mid[0] - tr["mid"][0], mid[1] - tr["mid"][1]))
                if dist <= JITTER_MATCH_FRAC * hgt and (bd is None or dist < bd):
                    best, bd = k, dist
            if best is None:
                tracks.append({"pts": {t: d}, "mid": mid, "last_t": t})
                used.add(len(tracks) - 1)
            else:
                tr = tracks[best]
                tr["pts"][t], tr["mid"], tr["last_t"] = d, mid, t
                used.add(best)
    out = []
    for tr in tracks:
        p = tr["pts"]
        for t in p:
            if t - 1 in p and t + 1 in p:
                for i in arm:
                    a, b, c = p[t - 1][i], p[t][i], p[t + 1][i]
                    if min(a[thr_idx], b[thr_idx], c[thr_idx]) >= thr:
                        out.append(float(np.hypot(b[1] - (a[1] + c[1]) / 2, b[2] - (a[2] + c[2]) / 2)))
    return out


def p1_height(d: dict) -> float:
    ys = [d[i][2] for i in P1_KEEP]
    return float(max(ys) - min(ys))


def auc_w(s: np.ndarray, y: np.ndarray, w: np.ndarray) -> float | None:
    pos, neg = y == 1, y == 0
    if not pos.any() or not neg.any():
        return None
    sp, sn, wp, wn = s[pos], s[neg], w[pos], w[neg]
    gt = (sp[:, None] > sn[None, :]).astype(float) + 0.5 * (sp[:, None] == sn[None, :])
    ww = wp[:, None] * wn[None, :]
    return float((gt * ww).sum() / ww.sum())


def choose_n_mode(mode_n: str | None) -> str:
    if mode_n:
        return mode_n
    for m in ("fp16", "mixed", "cpu"):
        p = SCRATCH / f"normal_{m}.json"
        if p.exists():
            r = jload(p)
            if r["nan_frames"] == 0 and not r["oom"] and r["n_frames_done"] == r["n_frames_total"]:
                return m
    raise SystemExit("tidak ada run N yang lengkap tanpa NaN / OOM")


def unit_scores(cds: dict, mode: str) -> list[dict]:
    lab = jload(LABELS)["units"]
    out = []
    for num, u in sorted(lab.items()):
        if u["label"] not in ("O", "L") or u["clip"] not in cds:
            continue
        cd = cds[u["clip"]]
        z = np.load(SCRATCH / f"normal_{mode}" / f"{u['clip']}_{u['rep_frame']:05d}.npz")
        e = z["E"].astype(np.float32)
        ed = cv2.dilate(np.where(np.isnan(e), -1.0, e).astype(np.float32), np.ones((EDGE_MAXF, EDGE_MAXF), np.uint8))
        ed[ed < 0] = np.nan
        m = unit_match(cd, u)
        p = m["points"]
        xs = np.clip(np.floor(p[:, 0]).astype(int), 0, cd.W - 1)
        ys = np.clip(np.floor(p[:, 1]).astype(int), 0, cd.H - 1)
        v = ed[ys, xs]
        score = float(np.nanmedian(v)) if np.isfinite(v).any() else None
        out.append({"unit": num, "clip": u["clip"], "label": u["label"], "cell": u["cell"], "weight": float(u["weight"]),
                    "score": score, "valid_frac": float(np.isfinite(v).mean()), "match_iou": m["iou"]})
    return out


def auc_table(units: list[dict]) -> dict:
    us = [u for u in units if u["score"] is not None]
    s = np.array([u["score"] for u in us])
    y = np.array([1 if u["label"] == "O" else 0 for u in us])
    w = np.array([u["weight"] for u in us])
    clip = np.array([u["clip"] for u in us])
    cells = sorted({u["cell"] for u in us})
    cell_idx = [np.array([i for i, u in enumerate(us) if u["cell"] == c]) for c in cells]
    groups = {"gabungan": np.ones(len(us), bool), "klip3": clip == "klip3", "test": clip == "test"}
    point = {g: {"tb": auc_w(s[m], y[m], np.ones(m.sum())), "b": auc_w(s[m], y[m], w[m])} for g, m in groups.items()}
    rng = np.random.default_rng(SEED_BOOT)
    boot = {g: {"tb": [], "b": []} for g in groups}
    for _ in range(N_BOOT):
        idx = np.concatenate([rng.choice(ci, size=len(ci), replace=True) for ci in cell_idx])
        for g, m in groups.items():
            sel = idx[m[idx]]
            for k, ww in (("tb", np.ones(len(sel))), ("b", w[sel])):
                a = auc_w(s[sel], y[sel], ww)
                if a is not None:
                    boot[g][k].append(a)
    tab = {}
    for g in groups:
        tab[g] = {}
        for k in ("tb", "b"):
            bs = np.asarray(boot[g][k])
            tab[g][k] = {"auc": point[g][k], "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                         "n_boot_valid": int(len(bs))}
        m = groups[g]
        tab[g]["n_O"], tab[g]["n_L"] = int((y[m] == 1).sum()), int((y[m] == 0).sum())
        tab[g]["median_O"] = float(np.median(s[m & (y == 1)])) if (m & (y == 1)).any() else None
        tab[g]["median_L"] = float(np.median(s[m & (y == 0)])) if (m & (y == 0)).any() else None
    tab["memisahkan_gabungan"] = bool(tab["gabungan"]["tb"]["ci95"][0] > 0.6 and tab["gabungan"]["b"]["ci95"][0] > 0.6)
    tab["units_without_score"] = [u["unit"] for u in units if u["score"] is None]
    return tab


def cmd_metrics(mode_n: str | None) -> None:
    s = jload(SCRATCH / "samples_1b.json")
    cds = cds_load(load_pipeline())
    p1 = jload(SCRATCH / "p1.json")
    p2 = jload(SCRATCH / "p2_fp16.json")
    p2_32 = jload(SCRATCH / "p2_fp32.json") if (SCRATCH / "p2_fp32.json").exists() else None
    mode = choose_n_mode(mode_n)
    nres = {m: jload(SCRATCH / f"normal_{m}.json") for m in ("fp16", "mixed", "cpu") if (SCRATCH / f"normal_{m}.json").exists()}
    res: dict = {"n_mode": mode, "per_frame": {}, "conf": {}, "time": {}, "vram": {}, "nan": {}, "jitter": {}}
    # per frame: wajah / blob / terdeteksi
    p1_vis, p1_pres, p2_sc = {i: [] for i in P1_ARM}, {i: [] for i in P1_ARM}, {i: [] for i in P2_ARM}
    for c, cd in cds.items():
        for f in s["samples"][c]:
            blobs, faces = frame_people(cd, f)
            key = f"{c}:{f}"
            d1 = [p1_det(r) for r in p1["image"][key]["poses"]]
            d2 = [p2_det(p) for p in p2["frames"][key]["persons"]]
            for r in p1["image"][key]["poses"]:
                dd = {x[0]: x for x in r}
                for i in P1_ARM:
                    p1_vis[i].append(dd[i][3])
                    p1_pres[i].append(dd[i][4])
            for p in p2["frames"][key]["persons"]:
                dd = {x[0]: x for x in p["kp"]}
                for i in P2_ARM:
                    p2_sc[i].append(dd[i][3])
            cov1 = match_faces([m for ok, m, _ in d1 if ok], faces, blobs, cd.H, cd.W)
            cov2 = match_faces([m for ok, m, _ in d2 if ok], faces, blobs, cd.H, cd.W)
            # POST-HOC V1 (ditambahkan setelah melihat data: kulit kaki di atas sepatu berlabel Face_Neck di klip3):
            # wajah hanya dihitung bila centroid-nya di setengah atas bbox blob tubuhnya. BUKAN definisi terdaftar.
            faces_v1 = [fc for fc in faces if fc["blob"] is None or fc["centroid_xy"][1]
                        < blobs[fc["blob"]]["bbox_xywh"][1] + blobs[fc["blob"]]["bbox_xywh"][3] / 2]
            cov1v = match_faces([m for ok, m, _ in d1 if ok], faces_v1, blobs, cd.H, cd.W)
            cov2v = match_faces([m for ok, m, _ in d2 if ok], faces_v1, blobs, cd.H, cd.W)
            merged_v1 = any(sum(fc["blob"] == k for fc in faces_v1) >= 2 for k in range(len(blobs)))
            res["per_frame"][key] = {"n_faces": len(faces), "n_blobs": len(blobs), "merged_blob": any(b["merged"] for b in blobs),
                                     "p1_returned": len(d1), "p1_detected": sum(ok for ok, _, _ in d1), "p1_faces_covered": cov1,
                                     "p2_boxes": len(d2), "p2_detected": sum(ok for ok, _, _ in d2), "p2_faces_covered": cov2,
                                     "p1_all": cov1 == len(faces) and len(faces) > 0, "p2_all": cov2 == len(faces) and len(faces) > 0,
                                     "v1_n_faces": len(faces_v1), "v1_merged_blob": merged_v1,
                                     "v1_p1_covered": cov1v, "v1_p2_covered": cov2v,
                                     "v1_p1_all": cov1v == len(faces_v1) and len(faces_v1) > 0,
                                     "v1_p2_all": cov2v == len(faces_v1) and len(faces_v1) > 0}
    for i, nm in zip(P1_ARM, ARM_NAMES):
        res["conf"][f"P1_vis_{nm}"] = {"p5": pct(p1_vis[i], 5), "p50": pct(p1_vis[i], 50), "n": len(p1_vis[i])}
        res["conf"][f"P1_pres_{nm}"] = {"p5": pct(p1_pres[i], 5), "p50": pct(p1_pres[i], 50), "n": len(p1_pres[i])}
    for i, nm in zip(P2_ARM, ARM_NAMES):
        res["conf"][f"P2_{nm}"] = {"p5": pct(p2_sc[i], 5), "p50": pct(p2_sc[i], 50), "n": len(p2_sc[i])}
    # getaran di jendela statis
    for c in cds:
        w0, w1 = s["static"][c]
        seq1, seq2 = [], []
        for f in range(w0, w1 + 1):
            ppl = []
            for r in p1["video"][f"{c}:{f}"]["poses"]:
                ok, mid, d = p1_det(r)
                ppl.append((mid, d, p1_height(d)))
            seq1.append(ppl)
            ppl = []
            for p in p2["frames"][f"{c}:{f}"]["persons"]:
                ok, mid, d = p2_det(p)
                ppl.append((mid, d, float(p["blob_bbox_xywh"][3])))
            seq2.append(ppl)
        j1 = jitter(seq1, P1_ARM, 3, P1_VIS_MIN)
        j2 = jitter(seq2, P2_ARM, 3, P2_SCORE_MIN)
        res["jitter"][c] = {"P1": {"p50": pct(j1, 50), "p95": pct(j1, 95), "n": len(j1)},
                            "P2": {"p50": pct(j2, 50), "p95": pct(j2, 95), "n": len(j2)}}
    # waktu / VRAM / NaN
    t1i = [v["time_s"] for v in p1["image"].values()]
    t1v = [v["time_s"] for v in p1["video"].values()]
    t2 = [v["time_s"] for v in p2["frames"].values()]
    t2c = [p["time_s"] for v in p2["frames"].values() for p in v["persons"]]
    res["time"] = {"P1_image": {"mean": float(np.mean(t1i)), "p95": pct(t1i, 95)}, "P1_video": {"mean": float(np.mean(t1v)), "p95": pct(t1v, 95)},
                   "P2_fp16_frame": {"mean": float(np.mean(t2)), "p95": pct(t2, 95)}, "P2_fp16_box": {"mean": float(np.mean(t2c)), "p95": pct(t2c, 95)}}
    res["vram"]["P2_fp16_peak_reserved_mib"] = max(v["peak_reserved_mib"] for v in p2["frames"].values())
    res["vram"]["P2_fp16_peak_alloc_mib"] = max(v["peak_alloc_mib"] for v in p2["frames"].values())
    p1_nan = sum(int(np.isnan([x[1] for r in v["poses"] for x in r] + [x[2] for r in v["poses"] for x in r]).sum())
                 for v in list(p1["image"].values()) + list(p1["video"].values()))
    res["nan"]["P1_coord_nan"] = p1_nan
    res["nan"]["P2_fp16"] = {k: sum(p[k] for v in p2["frames"].values() for p in v["persons"]) for k in ("nan", "inf", "sat")}
    res["nan"]["P2_fp16_absmax"] = max(p["absmax"] for v in p2["frames"].values() for p in v["persons"])
    res["P2_oom"] = p2["oom"]
    if p2_32:
        tt = [v["time_s"] for v in p2_32["frames"].values()]
        res["time"]["P2_fp32_frame"] = {"mean": float(np.mean(tt)) if tt else None, "p95": pct(tt, 95)}
        res["vram"]["P2_fp32_peak_reserved_mib"] = max((v["peak_reserved_mib"] for v in p2_32["frames"].values()), default=None)
        res["nan"]["P2_fp32"] = {k: sum(p[k] for v in p2_32["frames"].values() for p in v["persons"]) for k in ("nan", "inf", "sat")}
        res["nan"]["P2_fp32_absmax"] = max((p["absmax"] for v in p2_32["frames"].values() for p in v["persons"]), default=None)
        diffs = []
        for key, v in p2_32["frames"].items():
            for pa, pb in zip(p2["frames"][key]["persons"], v["persons"]):
                da, db = {x[0]: x for x in pa["kp"]}, {x[0]: x for x in pb["kp"]}
                for i in P2_ARM:
                    if da[i][3] >= P2_SCORE_MIN and db[i][3] >= P2_SCORE_MIN:
                        diffs.append(float(np.hypot(da[i][1] - db[i][1], da[i][2] - db[i][2])))
        res["P2_fp16_vs_fp32_arm_px"] = {"p50": pct(diffs, 50), "p95": pct(diffs, 95), "max": max(diffs) if diffs else None, "n": len(diffs)}
        res["P2_fp32_oom"] = p2_32["oom"]
    for m, r in nres.items():
        tt = [v["time_s"] for v in r["frames"].values()]
        res["time"][f"N_{m}"] = {"mean": float(np.mean(tt)) if tt else None, "p95": pct(tt, 95)}
        res["vram"][f"N_{m}_peak_reserved_mib"] = max((v.get("peak_reserved_mib") or 0 for v in r["frames"].values()), default=None)
        res["nan"][f"N_{m}"] = {"nan_frames": r["nan_frames"], "frames": f"{r['n_frames_done']}/{r['n_frames_total']}", "oom": r["oom"],
                                "pad_ok": all(v["pad_ok"] for v in r["frames"].values())}
    # kriteria
    k3 = [res["per_frame"][f"klip3:{f}"] for f in s["samples"]["klip3"]]
    res["criteria"] = {
        "P1": {"a": p1_nan == 0, "c_frames": sum(r["p1_all"] for r in k3), "c_of": len(k3)},
        "P2": {"a": res["nan"]["P2_fp16"]["nan"] == 0 and res["nan"]["P2_fp16"]["inf"] == 0 and not p2["oom"],
               "c_frames": sum(r["p2_all"] for r in k3), "c_of": len(k3)},
        "N": {"a_mode": mode, "a_fp16": ("fp16" in nres and nres["fp16"]["nan_frames"] == 0 and not nres["fp16"]["oom"])},
    }
    for k in ("P1", "P2"):
        res["criteria"][k]["c"] = res["criteria"][k]["c_frames"] >= int(np.ceil(0.9 * len(k3)))
        res["criteria"][k]["c_v1_posthoc_frames"] = sum(r[f"v1_{k.lower()}_all"] for r in k3)
        res["criteria"][k]["c_v1_posthoc"] = res["criteria"][k]["c_v1_posthoc_frames"] >= int(np.ceil(0.9 * len(k3)))
    units = unit_scores(cds, mode)
    res["n_units"] = units
    res["auc"] = auc_table(units)
    save("metrics.json", res)
    print(json.dumps({k: v for k, v in res.items() if k not in ("per_frame", "n_units")}, indent=1, ensure_ascii=False, default=str))
    for key, r in res["per_frame"].items():
        print(f"  {key}: wajah {r['n_faces']} (V1 {r['v1_n_faces']}) blob {r['n_blobs']}{' (gabungan)' if r['merged_blob'] else ''}"
              f"{' (gabungan V1)' if r['v1_merged_blob'] else ''} | "
              f"P1 kembali {r['p1_returned']} terdeteksi {r['p1_detected']} cocok-wajah {r['p1_faces_covered']} (V1 {r['v1_p1_covered']}) | "
              f"P2 kotak {r['p2_boxes']} terdeteksi {r['p2_detected']} cocok-wajah {r['p2_faces_covered']} (V1 {r['v1_p2_covered']})")


# ── papan ──────────────────────────────────────────
def dim(img: np.ndarray) -> np.ndarray:
    return (img.astype(np.float32) * 0.45 + 255 * 0.55).astype(np.uint8)


def compose_row(num: str, c: str, f: int, cd: ClipData, p1: dict, p2: dict, mode: str) -> np.ndarray:
    sc = PANEL_H / cd.H
    pw = int(round(cd.W * sc))
    bgr = cv2.resize(cv2.cvtColor(read_rgb(cd, f), cv2.COLOR_RGB2BGR), (pw, PANEL_H), interpolation=cv2.INTER_AREA)
    key = f"{c}:{f}"

    def P(x, y):
        return int(round(x * sc)), int(round(y * sc))

    a1 = dim(bgr)
    for k, rows in enumerate(p1["image"][key]["poses"]):
        col = PERSON_BGR[k % len(PERSON_BGR)]
        d = {r[0]: r for r in rows}
        for i, j in P1_SEGMENTS:
            cv2.line(a1, P(d[i][1], d[i][2]), P(d[j][1], d[j][2]), col, 4, cv2.LINE_AA)
        for i in P1_ARM + P1_HIP:
            cv2.circle(a1, P(d[i][1], d[i][2]), 7, col, -1, cv2.LINE_AA)
    a2 = dim(bgr)
    for k, p in enumerate(p2["frames"][key]["persons"]):
        col = PERSON_BGR[k % len(PERSON_BGR)]
        d = {r[0]: r for r in p["kp"]}
        for i in P2_DRAW:
            if d[i][3] >= P2_SCORE_MIN:
                cv2.circle(a2, P(d[i][1], d[i][2]), 8 if i in P2_ARM else 4, col, -1, cv2.LINE_AA)
    z = np.load(SCRATCH / f"normal_{mode}" / f"{c}_{f:05d}.npz")
    e, code = z["E"].astype(np.float32), z["code"]
    g = np.full(e.shape, 238, np.float32)
    v = np.clip(np.nan_to_num(e, nan=0.0) / 30.0, 0, 1)
    g = np.where(code == 1, 255 * (1 - v), g)
    g = np.where(code == 3, 215, g)
    g = np.where(code == 2, 150, g)
    a3 = cv2.cvtColor(cv2.resize(g.astype(np.uint8), (pw, PANEL_H), interpolation=cv2.INTER_AREA), cv2.COLOR_GRAY2BGR)
    gap, head = 12, 90
    W = 4 * pw + 5 * gap
    img = np.full((PANEL_H + head + gap, W, 3), 255, np.uint8)
    for k, (pan, lbl) in enumerate(((bgr, "foto"), (a1, "P1"), (a2, "P2"), (a3, "N"))):
        x = gap + k * (pw + gap)
        img[head:head + PANEL_H, x:x + pw] = pan
        cv2.putText(img, lbl, (x + 8, head - 18), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, num, (W - 150, head - 14), cv2.FONT_HERSHEY_SIMPLEX, 2.4, (0, 0, 200), 6, cv2.LINE_AA)
    return img


def cmd_board(mode_n: str | None) -> None:
    s = jload(SCRATCH / "samples_1b.json")
    cds = cds_load(load_pipeline())
    p1 = jload(SCRATCH / "p1.json")
    p2 = jload(SCRATCH / "p2_fp16.json")
    mode = choose_n_mode(mode_n)
    frames = [(c, f) for c in ("klip3", "test") for f in s["samples"][c]]
    order = np.random.default_rng(SEED_BOARD).permutation(len(frames))
    pdir = OUT / "papan"
    pdir.mkdir(parents=True, exist_ok=True)
    mapping = {}
    for k, idx in enumerate(order):
        num = f"{k + 1:02d}"
        c, f = frames[int(idx)]
        mapping[num] = {"clip": c, "frame": f}
        img = compose_row(num, c, f, cds[c], p1, p2, mode)
        ok, buf = cv2.imencode(".png", img)
        buf.tofile(pdir / f"{num}.png")
        print(f"  {num}.png {img.shape[1]}x{img.shape[0]}")
    save("board_map.json", {"seed": SEED_BOARD, "n_mode": mode, "map": mapping})


# ══ Tahap 1c ═══════════════════════════════════════
# docs/04 "Keputusan T-701 setelah Tahap 1b": P2 dengan crop per orang; acuan orang = wajah V1 (kini terdaftar).
SEED_BOARD_1C = 701003


def faces_v1_of(faces: list[dict], blobs: list[dict]) -> list[dict]:
    """Wajah V1: centroid di setengah atas bbox blob tubuhnya; wajah tanpa blob tubuh tetap dihitung (= V1 di `metrics` 1b)."""
    return [fc for fc in faces if fc["blob"] is None
            or fc["centroid_xy"][1] < blobs[fc["blob"]]["bbox_xywh"][1] + blobs[fc["blob"]]["bbox_xywh"][3] / 2]


def person_parts(blobs: list[dict], faces1: list[dict], H: int, W: int) -> list[dict]:
    """Crop 1c: blob dengan <= 1 wajah V1 -> satu bagian = seluruh blob; blob dengan >= 2 wajah V1 -> piksel blob dibagi ke
    centroid wajah V1 terdekat (Voronoi Euclidean di dalam blob; seri -> wajah dengan urutan lebih kecil). Kotak = bbox ketat."""
    parts = []
    for k, b in enumerate(blobs):
        fids = [j for j, fc in enumerate(faces1) if fc["blob"] == k]
        if len(fids) <= 1:
            parts.append({"blob": k, "face": fids[0] if fids else None, "split": False, "mask": b["mask"]})
            continue
        ys, xs = np.nonzero(b["mask"])
        cen = np.asarray([faces1[j]["centroid_xy"] for j in fids], float)
        own = np.argmin((xs[:, None] - cen[None, :, 0]) ** 2 + (ys[:, None] - cen[None, :, 1]) ** 2, axis=1)
        for q, j in enumerate(fids):
            m = np.zeros((H, W), bool)
            m[ys[own == q], xs[own == q]] = True
            parts.append({"blob": k, "face": j, "split": True, "mask": m})
    for p in parts:
        ys, xs = np.nonzero(p["mask"])
        p["area"] = int(len(xs))
        p["bbox_xywh"] = ([int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]
                          if len(xs) else None)
    return parts


def frame_parts(cd: ClipData, f: int) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    blobs, faces = frame_people(cd, f)
    faces1 = faces_v1_of(faces, blobs)
    return blobs, faces, faces1, person_parts(blobs, faces1, cd.H, cd.W)


def p2_infer(proc, model, rgb: np.ndarray, box: list[float], dtype, keep) -> dict:
    """Satu kotak -> titik tubuh 5–69; titik wajah / kepala dibuang di heatmap sebelum post-process (sama dengan `p2`)."""
    import torch

    torch.cuda.synchronize()
    t1 = time.perf_counter()
    inp = proc(images=rgb, boxes=[[box]], return_tensors="pt")
    pix = inp["pixel_values"].to("cuda", dtype)
    with torch.inference_mode():
        hm = model(pixel_values=pix).heatmaps
        st = {"nan": int(torch.isnan(hm).sum()), "inf": int(torch.isinf(hm).sum()),
              "sat": int((hm.abs() >= HEAT_SAT).sum()), "absmax": float(hm.float().nan_to_num(0.0).abs().amax())}
        body = hm.index_select(1, keep)                # titik wajah / kepala dibuang DI SINI
        del hm
        pp = proc.post_process_pose_estimation(SimpleNamespace(heatmaps=body), boxes=[[box]])
        kp = pp[0][0]["keypoints"].float().cpu().numpy()
        sc = pp[0][0]["scores"].float().cpu().numpy()
        del body, pp
    del inp, pix
    torch.cuda.synchronize()
    return {"time_s": time.perf_counter() - t1, **st,
            "kp": [[P2_KEEP[j], float(kp[j, 0]), float(kp[j, 1]), float(sc[j])] for j in range(len(P2_KEEP))]}


def cmd_p2c(prec: str | None) -> None:
    import torch

    s = jload(SCRATCH / "samples_1b.json")
    cd = ClipData("klip3", load_pipeline())
    if not torch.cuda.is_available():
        raise SystemExit("CUDA tidak tersedia — butuh GPU NVIDIA lewat PyTorch cu118")
    free, total = (v / MIB for v in torch.cuda.mem_get_info())
    if prec is None:
        prec = "fp32" if free >= VRAM_NEED_MIB["p2_fp32"] else "fp16"
    if free < VRAM_NEED_MIB[f"p2_{prec}"]:
        raise SystemExit(f"VRAM bebas {free:.0f} MiB < {VRAM_NEED_MIB[f'p2_{prec}']} MiB untuk P2 {prec} — tutup aplikasi lain "
                         f"yang memakai GPU lalu jalankan ulang (model belum dimuat)")
    dtype = torch.float32 if prec == "fp32" else torch.float16
    t0 = time.perf_counter()
    proc, model = hf_load("pose", "Sapiens2ForPoseEstimation", dtype, "cuda")
    res: dict = {"precision": prec, "vram_free_before_load_mib": round(free, 1), "vram_total_mib": round(total, 1),
                 "load_s": round(time.perf_counter() - t0, 2),
                 "vram_free_after_load_mib": round(torch.cuda.mem_get_info()[0] / MIB, 1),
                 "kept_indices": list(P2_KEEP), "oom": None, "frames": {}}
    keep = torch.tensor(P2_KEEP, device="cuda")
    for f in s["samples"]["klip3"]:
        rgb = read_rgb(cd, f)
        _, _, _, parts = frame_parts(cd, f)
        torch.cuda.reset_peak_memory_stats()
        rows, tf = [], 0.0
        try:
            for p in parts:
                row = {k: v for k, v in p.items() if k != "mask"}
                if p["bbox_xywh"] is not None:
                    row.update(p2_infer(proc, model, rgb, [float(v) for v in p["bbox_xywh"]], dtype, keep))
                    tf += row["time_s"]
                rows.append(row)
        except torch.cuda.OutOfMemoryError:
            res["oom"] = {"frame": f, "reserved_mib": round(torch.cuda.memory_reserved() / MIB, 1)}
            print(f"OOM di klip3:{f}")
            break
        res["frames"][str(f)] = {"time_s": tf, "n_parts": len(parts), "parts": rows,
                                 "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / MIB, 1),
                                 "peak_alloc_mib": round(torch.cuda.max_memory_allocated() / MIB, 1)}
        print(f"  klip3:{f} kotak {len(parts)} (dibagi {sum(p['split'] for p in parts)}) {tf:.2f} s, puncak reserved "
              f"{res['frames'][str(f)]['peak_reserved_mib']} MiB, NaN {sum(r.get('nan', 0) for r in rows)} "
              f"inf {sum(r.get('inf', 0) for r in rows)}", flush=True)
    save(f"p2c_{prec}.json", res)


def part_at(parts: list[dict], x: float, y: float, H: int, W: int) -> int | None:
    xi, yi = int(round(x)), int(round(y))
    if not (0 <= xi < W and 0 <= yi < H):
        return None
    return next((i for i, p in enumerate(parts) if p["mask"][yi, xi]), None)


def cmd_metrics1c() -> None:
    s = jload(SCRATCH / "samples_1b.json")
    cd = ClipData("klip3", load_pipeline())
    m1b = jload(SCRATCH / "metrics.json")
    p2b = jload(SCRATCH / "p2_fp16.json")
    runs = [p for p in ("fp32", "fp16") if (SCRATCH / f"p2c_{p}.json").exists()]
    runs = [p for p in runs if not jload(SCRATCH / f"p2c_{p}.json")["oom"]]
    if not runs:
        raise SystemExit("tidak ada run p2c yang lengkap tanpa OOM")
    prec = runs[0]
    run = jload(SCRATCH / f"p2c_{prec}.json")
    res: dict = {"precision": prec, "per_frame": {}}
    t_part, t_frame, same_box_diff = [], [], []
    for f in s["samples"]["klip3"]:
        blobs, faces, faces1, parts = frame_parts(cd, f)
        fr = run["frames"][str(f)]
        if [r["bbox_xywh"] for r in fr["parts"]] != [p["bbox_xywh"] for p in parts]:
            raise SystemExit(f"f{f}: kotak di p2c_{prec}.json tidak sama dengan kotak yang dihitung ulang")
        poses = []
        for i, r in enumerate(fr["parts"]):
            if "kp" not in r:
                poses.append({"part": i, "ran": False, "ok": False, "mid_part": None, "arm_other": 0})
                continue
            t_part.append(r["time_s"])
            ok, mid, d = p2_det(r)
            arm_parts = {j: part_at(parts, d[j][1], d[j][2], cd.H, cd.W) for j in P2_ARM if d[j][3] >= P2_SCORE_MIN}
            poses.append({"part": i, "ran": True, "ok": ok, "n_arm": sum(d[j][3] >= P2_SCORE_MIN for j in P2_ARM),
                          "mid_xy": [round(mid[0], 1), round(mid[1], 1)], "mid_part": part_at(parts, mid[0], mid[1], cd.H, cd.W),
                          "arm_other": sum(q is not None and q != i for q in arm_parts.values()),
                          "arm_other_pts": [[j, q, d[j][1], d[j][2]] for j, q in arm_parts.items() if q is not None and q != i]})
        # diagnostik (bukan kriteria): titik lengan di bagian orang lain -> jarak ke titik lengan terdekat pose pemilik bagian itu
        for p in poses:
            p["arm_other_detail"] = []
            for j, q, x, y in p.pop("arm_other_pts", []):
                other = fr["parts"][q]
                pts = [(k[1], k[2]) for k in other.get("kp", []) if k[0] in P2_ARM and k[3] >= P2_SCORE_MIN]
                dist = min((float(np.hypot(x - a, y - b)) for a, b in pts), default=None)
                p["arm_other_detail"].append({"kp": j, "in_part": q, "dist_to_owner_arm_px": None if dist is None else round(dist, 1)})
        t_frame.append(fr["time_s"])
        persons = []
        for j, fc in enumerate(faces1):
            i = next((k for k, p in enumerate(parts) if p["face"] == j), None)
            det = i is not None and poses[i]["ok"] and poses[i]["mid_part"] == i
            persons.append({"face": j, "blob": fc["blob"], "part": i, "detected": bool(det),
                            "fail": None if det else ("tanpa blob" if i is None else "lengan < 4/6" if not poses[i]["ok"]
                                                      else "bahu di luar bagian")})
        hosts = [p["mid_part"] for p in poses if p["ok"] and p["mid_part"] is not None]
        double = sorted({h for h in hosts if hosts.count(h) >= 2})
        b1 = m1b["per_frame"][f"klip3:{f}"]
        changed = any(p["split"] for p in parts)
        if not changed:                                # kotak sama dengan 1b -> selisih titik lengan 1b (fp16) vs 1c
            for pa, pb in zip(p2b["frames"][f"klip3:{f}"]["persons"], fr["parts"]):
                da, db = {x[0]: x for x in pa["kp"]}, {x[0]: x for x in pb["kp"]}
                same_box_diff += [float(np.hypot(da[j][1] - db[j][1], da[j][2] - db[j][2])) for j in P2_ARM
                                  if da[j][3] >= P2_SCORE_MIN and db[j][3] >= P2_SCORE_MIN]
        res["per_frame"][str(f)] = {
            "n_faces": len(faces), "n_v1": len(faces1), "n_blobs": len(blobs), "v1_ne_blobs": len(faces1) != len(blobs),
            "v1_no_blob": sum(fc["blob"] is None for fc in faces1),
            "v1_per_blob": [sum(fc["blob"] == k for fc in faces1) for k in range(len(blobs))],
            "crop_changed": changed, "n_parts": len(parts), "parts_area": [p["area"] for p in parts],
            "parts_bbox": [p["bbox_xywh"] for p in parts], "poses": poses, "persons": persons,
            "detected_1c": sum(p["detected"] for p in persons), "pass_1c": len(faces1) > 0 and all(p["detected"] for p in persons),
            "double_parts": double, "arm_in_other_part": sum(p["arm_other"] for p in poses),
            "n_v1_1b": b1["v1_n_faces"], "detected_1b": b1["v1_p2_covered"], "pass_1b": b1["v1_p2_all"],
            "time_s": fr["time_s"], "peak_reserved_mib": fr["peak_reserved_mib"]}
        if b1["v1_n_faces"] != len(faces1):
            raise SystemExit(f"f{f}: jumlah wajah V1 {len(faces1)} != metrics 1b {b1['v1_n_faces']}")
    pf = res["per_frame"].values()
    res["c"] = {"frames": sum(r["pass_1c"] for r in pf), "of": len(res["per_frame"]),
                "pass": sum(r["pass_1c"] for r in pf) >= int(np.ceil(0.9 * len(res["per_frame"]))),
                "failed": [int(f) for f, r in res["per_frame"].items() if not r["pass_1c"]]}
    res["c_1b_v1"] = sum(r["pass_1b"] for r in pf)
    res["v1_ne_blobs"] = [int(f) for f, r in res["per_frame"].items() if r["v1_ne_blobs"]]
    res["crop_changed"] = [int(f) for f, r in res["per_frame"].items() if r["crop_changed"]]
    res["double"] = {f: r["double_parts"] for f, r in res["per_frame"].items() if r["double_parts"]}
    res["arm_in_other_part"] = {f: r["arm_in_other_part"] for f, r in res["per_frame"].items() if r["arm_in_other_part"]}
    res["arm_in_other_part_detail"] = {f: [[p["part"], x["kp"], x["in_part"], x["dist_to_owner_arm_px"]] for p in r["poses"]
                                           for x in p["arm_other_detail"]] for f, r in res["per_frame"].items() if r["arm_in_other_part"]}
    res["time"] = {"part": {"mean": float(np.mean(t_part)), "p50": pct(t_part, 50), "p95": pct(t_part, 95), "n": len(t_part)},
                   "frame": {"mean": float(np.mean(t_frame)), "p50": pct(t_frame, 50), "p95": pct(t_frame, 95), "n": len(t_frame)},
                   "first_part_s": run["frames"][str(s["samples"]["klip3"][0])]["parts"][0]["time_s"]}
    res["vram"] = {"free_before_load_mib": run["vram_free_before_load_mib"], "free_after_load_mib": run["vram_free_after_load_mib"],
                   "peak_reserved_mib": max(r["peak_reserved_mib"] for r in run["frames"].values()),
                   "peak_alloc_mib": max(r["peak_alloc_mib"] for r in run["frames"].values()), "load_s": run["load_s"]}
    res["nan"] = {k: sum(p.get(k, 0) for r in run["frames"].values() for p in r["parts"]) for k in ("nan", "inf", "sat")}
    res["nan"]["absmax"] = max(p["absmax"] for r in run["frames"].values() for p in r["parts"] if "absmax" in p)
    res["same_box_1b_vs_1c_arm_px"] = {"p50": pct(same_box_diff, 50), "p95": pct(same_box_diff, 95),
                                       "max": max(same_box_diff) if same_box_diff else None, "n": len(same_box_diff)}
    save("metrics_1c.json", res)
    print(json.dumps({k: v for k, v in res.items() if k != "per_frame"}, indent=1, ensure_ascii=False))
    for f, r in res["per_frame"].items():
        print(f"  f{f}: wajah {r['n_faces']} V1 {r['n_v1']} (per blob {r['v1_per_blob']}, tanpa blob {r['v1_no_blob']}) blob {r['n_blobs']}"
              f"{' BEDA' if r['v1_ne_blobs'] else ''} | 1b {r['detected_1b']}/{r['n_v1_1b']} | 1c {r['detected_1c']}/{r['n_v1']}"
              f"{' LOLOS' if r['pass_1c'] else ' gagal'} | kotak {r['n_parts']}{' (berubah)' if r['crop_changed'] else ''} "
              f"luas {r['parts_area']} | pose " + "; ".join(
                  f"#{p['part']} ok {p['ok']} lengan {p.get('n_arm')} bahu-di {p['mid_part']} lengan-di-lain {p['arm_other']}"
                  for p in r["poses"]) + " | " + ", ".join(f"wajah{p['face']}:{p['fail']}" for p in r["persons"] if p["fail"])
              + f" | {r['time_s']:.2f} s")


def draw_p2(img: np.ndarray, persons: list[dict], sc: float) -> np.ndarray:
    for k, p in enumerate(persons):
        if "kp" not in p:
            continue
        col = PERSON_BGR[k % len(PERSON_BGR)]
        d = {r[0]: r for r in p["kp"]}
        for i in P2_DRAW:
            if d[i][3] >= P2_SCORE_MIN:
                cv2.circle(img, (int(round(d[i][1] * sc)), int(round(d[i][2] * sc))), 8 if i in P2_ARM else 4, col, -1, cv2.LINE_AA)
    return img


def cmd_board1c() -> None:
    cd = ClipData("klip3", load_pipeline())
    m = jload(SCRATCH / "metrics_1c.json")
    p2b = jload(SCRATCH / "p2_fp16.json")
    run = jload(SCRATCH / f"p2c_{m['precision']}.json")
    frames = list(m["crop_changed"])
    order = np.random.default_rng(SEED_BOARD_1C).permutation(len(frames))
    pdir = OUT / "papan_1c"
    pdir.mkdir(parents=True, exist_ok=True)
    sc = PANEL_H / cd.H
    pw = int(round(cd.W * sc))
    gap, head = 12, 90
    W = 3 * pw + 4 * gap
    mapping = {}
    for k, idx in enumerate(order):
        num = f"{k + 1:02d}"
        f = frames[int(idx)]
        mapping[num] = {"clip": "klip3", "frame": f}
        bgr = cv2.resize(cv2.cvtColor(read_rgb(cd, f), cv2.COLOR_RGB2BGR), (pw, PANEL_H), interpolation=cv2.INTER_AREA)
        a1 = draw_p2(dim(bgr), p2b["frames"][f"klip3:{f}"]["persons"], sc)
        a2 = draw_p2(dim(bgr), run["frames"][str(f)]["parts"], sc)
        img = np.full((PANEL_H + head + gap, W, 3), 255, np.uint8)
        for j, (pan, lbl) in enumerate(((bgr, "foto"), (a1, "P2 1b"), (a2, "P2 1c"))):
            x = gap + j * (pw + gap)
            img[head:head + PANEL_H, x:x + pw] = pan
            cv2.putText(img, lbl, (x + 8, head - 18), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, num, (W - 150, head - 14), cv2.FONT_HERSHEY_SIMPLEX, 2.4, (0, 0, 200), 6, cv2.LINE_AA)
        ok, buf = cv2.imencode(".png", img)
        buf.tofile(pdir / f"{num}.png")
        colored = [int(((pan.max(axis=2).astype(int) - pan.min(axis=2)) > 120).sum()) for pan in (a1, a2)]
        print(f"  {num}.png {img.shape[1]}x{img.shape[0]} piksel berwarna P2 1b / 1c {colored}")
    save("board_map_1c.json", {"seed": SEED_BOARD_1C, "precision_1c": m["precision"], "map": mapping})


def main(argv: list[str]) -> None:
    if len(argv) >= 2 and argv[0] == "hash" and argv[1] in ("before", "after", "before_1b", "after_1b", "before_1c", "after_1c",
                                                             "before_3", "after_3"):
        cmd_hash(argv[1])
    elif argv[:1] == ["meta"]:
        cmd_meta()
    elif argv[:1] == ["samples"]:
        cmd_samples()
    elif argv[:1] == ["download"]:
        cmd_download()
    elif argv[:1] == ["samples1b"]:
        cmd_samples1b()
    elif argv[:1] == ["p1"]:
        cmd_p1()
    elif argv[:1] == ["p2"] and argv[1:2] in (["fp16"], ["fp32"]):
        cmd_p2(argv[1])
    elif argv[:1] == ["normal"] and argv[1:2] in (["fp16"], ["mixed"], ["cpu"]):
        cmd_normal(argv[1])
    elif argv[:1] == ["metrics"]:
        cmd_metrics(argv[1] if len(argv) > 1 else None)
    elif argv[:1] == ["board"]:
        cmd_board(argv[1] if len(argv) > 1 else None)
    elif argv[:1] == ["p2c"] and argv[1:2] in ([], ["fp16"], ["fp32"]):
        cmd_p2c(argv[1] if len(argv) > 1 else None)
    elif argv[:1] == ["metrics1c"]:
        cmd_metrics1c()
    elif argv[:1] == ["board1c"]:
        cmd_board1c()
    else:
        print(__doc__)
        sys.exit(2)


if __name__ == "__main__":
    main(sys.argv[1:])
