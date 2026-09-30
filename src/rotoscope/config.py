"""Config loader: configs/default.yaml (pipeline, stage [2]–[4]) + configs/styles/*.yaml (style, [5]).

Sumber kebenaran default = dataclass di file ini (T-104a, opsi A). configs/default.yaml,
configs/styles/rough-sketch.yaml, dan dua blok YAML di docs/02 wajib identik dengan default ini —
dijaga tests/test_config.py. Ubah default = ubah ketiganya.

Urutan load: default kode → YAML (merge rekursif) → overrides (flag CLI T-104b) → konversi tipe →
validasi (dua tabel + aturan grup di docs/02). Hasil = dataclass frozen (immutable); mapping =
MappingProxyType (dict read-only); `groups` = tuple berurutan ((nama, (kelas, …)), …), urutan = id
grup 1..G (0 = background).

Aturan:
- Key tidak dikenal = error + saran key yang mirip. Key duplikat di file YAML = error (safe_load
  diam-diam memakai yang terakhir).
- YAML parsial = merge dengan default. `groups` diganti UTUH (bukan digabung per grup).
  Section kosong (mis. `stroke:` tanpa isi) = default section itu.
- `null` hanya boleh di `segment.revision` / `depth.revision`.
- `paths.work_dir` / `paths.out_dir`: relatif → terhadap direktori kerja (cwd) saat load.
  Aset style (`texture.brush_image`, `paper.texture_image`): relatif → terhadap root project
  (folder berisi pyproject.toml, dicari dari lokasi paket; editable install). Path absolut boleh.
  Path dengan karakter kontrol = error (di YAML, backslash dalam kutip GANDA adalah escape).
- Load TANPA side effect: folder tidak dibuat. Stage memanggil ensure_dir().

`segment.revision` / `depth.revision` boleh null di loader. Stage [2] (T-102b) dan [2c] (T-105)
WAJIB menolak null saat runtime (run offline butuh revision checkpoint yang di-pin) — bukan loader.

Hash per bagian untuk manifest [2]/[3]/[4]: section_hash(cfg, "groups") = sha256 dari canonical
JSON (key terurut, float dinormalisasi). Untuk `groups`, urutan grup ikut dihitung (menentukan id),
urutan kelas di dalam satu grup tidak. Hash `paths` bergantung mesin (path absolut) — jangan dipakai
sebagai penanda basi.

Overrides = dict key bertitik, mis. {"segment.model": "0.4b"}. Key yang mengandung titik
("0.8b") ditulis sebagai dict bersarang: {"segment.vram_min_free_mib": {"0.8b": 3000}}.
"""

from __future__ import annotations

import dataclasses
import difflib
import hashlib
import json
import math
import os
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources
from pathlib import Path
from types import MappingProxyType
from typing import Any, get_args, get_origin, get_type_hints

import yaml


class ConfigError(ValueError):
    """Config tidak valid: YAML rusak, key tidak dikenal, tipe salah, atau nilai di luar range."""


# ── Konstanta skema (docs/02) ──────────────────────
NUM_CLASSES = 29
BACKGROUND_CLASS = "Background"
BACKGROUND_GROUP = "background"      # id 0, dicadangkan
MAX_GROUPS = 255                     # id grup uint8
GROUP_NAME_RE = re.compile(r"[a-z_][a-z0-9_]*")
SEG_MODELS = ("0.8b", "0.4b")
SEG_MODEL_ID_PREFIX = "facebook/sapiens2-seg-"  # Sapiens v1 (facebook/sapiens-seg-…) = CC-BY-NC
DEPTH_MODEL_ID_REQUIRED = "Small"               # DA-V2 Base/Large = CC-BY-NC
PRECISIONS = ("fp16", "fp32")                   # tanpa bf16 (Turing, P-005)
PROBS_DTYPES = ("uint8",)
VRAM_MAX_MIB = 4096
NORMALIZE_METHODS = ("log_median_iqr",)         # diperluas di T-302 (kandidat: affine)
STROKE_TYPES = ("silhouette", "silhouette_hole", "group_boundary", "occlusion")
STROKE_CAPS = ("round", "butt", "square")
TEMPORAL_SEED_MODES = ("frame", "fixed")
TEXTURE_MODES = ("none", "brush_stamp", "grain_overlay")
TEXTURE_MODE_BRUSH = "brush_stamp"
RESAMPLE_POINTS_MIN = 4
RENDER_SS_MAX = 8
HEX_COLOR_RE = re.compile(r"#[0-9a-fA-F]{6}")

CLASSES_DIR = "data"
CLASSES_FILENAME = "sapiens2_classes.json"
PROJECT_MARKER = "pyproject.toml"
_PACKAGE_DIR = Path(__file__).resolve().parent
_SUGGEST_MAX = 3
_SUGGEST_CUTOFF = 0.6

# Key yang pindah file / section (D-010) → pesan khusus, bukan sekadar "tidak dikenal".
_STYLE_SECTIONS = ("shape", "stroke", "jitter", "multipass", "texture", "paper", "render")
_PIPELINE_SECTIONS = ("paths", "segment", "depth", "qc", "groups", "stabilize", "vectorize")
_MOVED_IN_PIPELINE = {
    s: "ini parameter style — taruh di configs/styles/*.yaml" for s in _STYLE_SECTIONS
}
_MOVED_IN_STYLE = {
    "temporal": "pindah ke stabilize.temporal di configs/default.yaml (D-010)",
    "shape.min_contour_area": "pindah ke vectorize.min_region_area di configs/default.yaml (D-010)",
    **{s: "ini parameter pipeline — taruh di configs/default.yaml" for s in _PIPELINE_SECTIONS},
}


def _frozen(d: dict) -> Callable[[], Mapping]:
    return lambda: MappingProxyType(dict(d))


# ── Pipeline: configs/default.yaml ─────────────────
@dataclass(frozen=True)
class PathsConfig:
    work_dir: Path = Path("work")
    out_dir: Path = Path("out")


@dataclass(frozen=True)
class SegmentConfig:
    model: str = "0.8b"
    model_ids: Mapping[str, str] = field(default_factory=_frozen({
        "0.8b": "facebook/sapiens2-seg-0.8b",
        "0.4b": "facebook/sapiens2-seg-0.4b",
    }))
    revision: str | None = None
    precision: str = "fp16"
    vram_min_free_mib: Mapping[str, int] = field(default_factory=_frozen({"0.8b": 3300, "0.4b": 2300}))
    probs_dtype: str = "uint8"


@dataclass(frozen=True)
class DepthConfig:
    model_id: str = "depth-anything/Depth-Anything-V2-Small-hf"
    revision: str | None = None
    precision: str = "fp32"
    vram_min_free_mib: int = 500


@dataclass(frozen=True)
class QcConfig:
    area_min: float = 0.03
    area_max: float = 0.70
    iou_min: float = 0.55
    blob_min: float = 0.05
    max_big_blobs: int = 1
    area_median_window: int = 49
    area_drop_min: float = 0.6


DEFAULT_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("hair", ("Hair",)),
    ("face", ("Face_Neck", "Eyeglass", "Lower_Lip", "Upper_Lip", "Lower_Teeth", "Upper_Teeth", "Tongue")),
    ("torso", ("Torso", "Upper_Clothing", "Apparel", "Lower_Clothing")),
    ("left_arm", ("Left_Upper_Arm", "Left_Lower_Arm", "Left_Hand")),
    ("right_arm", ("Right_Upper_Arm", "Right_Lower_Arm", "Right_Hand")),
    ("left_leg", ("Left_Upper_Leg", "Left_Lower_Leg", "Left_Foot", "Left_Shoe", "Left_Sock")),
    ("right_leg", ("Right_Upper_Leg", "Right_Lower_Leg", "Right_Foot", "Right_Shoe", "Right_Sock")),
)


@dataclass(frozen=True)
class TemporalConfig:
    enabled: bool = False
    mask_ema_alpha: float = 0.7
    optical_flow_blend: float = 0.4
    boil_preserve: float = 0.3
    qc_fail_weight: float = 0.25


@dataclass(frozen=True)
class StabilizeDepthConfig:
    normalize: str = "log_median_iqr"
    temporal: bool = True


@dataclass(frozen=True)
class StabilizeConfig:
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    island_min_px: int = 30
    mode_k: int = 3
    depth: StabilizeDepthConfig = field(default_factory=StabilizeDepthConfig)


@dataclass(frozen=True)
class DepthLinesConfig:
    blur_sigma: float = 1.0
    hi_pct: float = 95.0
    lo_pct: float = 90.0
    erode_px: int = 5
    min_dist_px: float = 7.0
    min_len_px: float = 30.0


@dataclass(frozen=True)
class TrackConfig:
    max_match_dist_px: float = 12.0


@dataclass(frozen=True)
class VectorizeConfig:
    min_region_area: int = 800
    min_hole_area: int = 200
    line_min_px: int = 5
    min_stroke_px: int = 6
    depth_lines: DepthLinesConfig = field(default_factory=DepthLinesConfig)
    track: TrackConfig = field(default_factory=TrackConfig)


@dataclass(frozen=True)
class PipelineConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    segment: SegmentConfig = field(default_factory=SegmentConfig)
    depth: DepthConfig = field(default_factory=DepthConfig)
    qc: QcConfig = field(default_factory=QcConfig)
    groups: tuple[tuple[str, tuple[str, ...]], ...] = DEFAULT_GROUPS
    stabilize: StabilizeConfig = field(default_factory=StabilizeConfig)
    vectorize: VectorizeConfig = field(default_factory=VectorizeConfig)


# ── Style: configs/styles/*.yaml ───────────────────
@dataclass(frozen=True)
class ShapeConfig:
    simplify_epsilon: float = 2.5
    resample_points: int = 200
    smooth_tension: float = 0.5
    spline_steps: int = 8


@dataclass(frozen=True)
class TypeScaleConfig:
    width_scale: float = 1.0
    opacity_scale: float = 1.0


@dataclass(frozen=True)
class StrokeConfig:
    width_base: float = 3.2
    width_variation: float = 0.45
    width_noise_scale: float = 0.08
    color: str = "#1a1a1a"
    opacity: float = 0.92
    cap: str = "round"
    taper_ends: bool = True
    taper_px: float = 20.0
    taper_min: float = 0.15
    by_type: Mapping[str, TypeScaleConfig] = field(
        default_factory=_frozen({t: TypeScaleConfig() for t in STROKE_TYPES}))


@dataclass(frozen=True)
class JitterConfig:
    amplitude: float = 1.8
    frequency: float = 0.12
    temporal_seed_mode: str = "frame"
    temporal_drift: float = 0.35
    param_seed: int = 0


@dataclass(frozen=True)
class MultipassConfig:
    enabled: bool = True
    passes: int = 2
    offset: float = 1.2
    opacity_falloff: float = 0.55


@dataclass(frozen=True)
class TextureConfig:
    mode: str = "brush_stamp"
    brush_image: Path = Path("assets/brushes/pencil_01.png")
    stamp_spacing: float = 0.35
    pressure_noise: float = 0.25
    grain_strength: float = 0.18


@dataclass(frozen=True)
class PaperConfig:
    enabled: bool = True
    color: str = "#f4f1ea"
    texture_image: Path = Path("assets/paper/rough_01.jpg")
    texture_opacity: float = 0.35
    vignette: float = 0.12


@dataclass(frozen=True)
class RenderConfig:
    ss: int = 3


@dataclass(frozen=True)
class StyleConfig:
    shape: ShapeConfig = field(default_factory=ShapeConfig)
    stroke: StrokeConfig = field(default_factory=StrokeConfig)
    jitter: JitterConfig = field(default_factory=JitterConfig)
    multipass: MultipassConfig = field(default_factory=MultipassConfig)
    texture: TextureConfig = field(default_factory=TextureConfig)
    paper: PaperConfig = field(default_factory=PaperConfig)
    render: RenderConfig = field(default_factory=RenderConfig)


# ── API publik ─────────────────────────────────────
def load_pipeline(path: str | Path | None = None, overrides: Mapping[str, Any] | None = None) -> PipelineConfig:
    """Parameter pipeline (stage [2]–[4]). path=None → default kode saja."""
    return _load(PipelineConfig, path, overrides, _from_cwd, _MOVED_IN_PIPELINE, _validate_pipeline)


def load_style(path: str | Path | None = None, overrides: Mapping[str, Any] | None = None) -> StyleConfig:
    """Parameter style (stage [5]). path=None → default kode (= rough-sketch)."""
    return _load(StyleConfig, path, overrides, _from_project_root, _MOVED_IN_STYLE, _validate_style)


def to_dict(obj: Any) -> Any:
    """Config (atau bagiannya) → struktur JSON biasa (dict/list/str/angka); Path → string '/'."""
    if dataclasses.is_dataclass(obj):
        return {f.name: to_dict(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, Mapping):
        return {k: to_dict(v) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return [to_dict(v) for v in obj]
    if isinstance(obj, Path):
        return obj.as_posix()
    return obj


def section_hash(cfg: PipelineConfig | StyleConfig, section: str) -> str:
    """sha256 (hex) canonical JSON satu bagian config, untuk manifest (mis. groups_hash)."""
    names = [f.name for f in dataclasses.fields(cfg)]
    if section not in names:
        raise ConfigError(f"section_hash: bagian {section!r} tidak ada; pilihan: {', '.join(names)}")
    data = to_dict(getattr(cfg, section))
    if section == "groups":
        data = [[name, sorted(classes)] for name, classes in data]
    blob = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def ensure_dir(path: str | Path) -> Path:
    """Buat folder (beserta induknya) kalau belum ada. Dipanggil stage, bukan loader."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


@lru_cache(maxsize=1)
def load_class_names() -> tuple[str, ...]:
    """Nama 29 kelas Sapiens2-seg (indeks = id kelas) dari data paket."""
    res = resources.files("rotoscope") / CLASSES_DIR / CLASSES_FILENAME
    classes = tuple(json.loads(res.read_text(encoding="utf-8"))["classes"])
    if len(classes) != NUM_CLASSES or classes[0] != BACKGROUND_CLASS:
        raise ConfigError(f"{CLASSES_FILENAME}: harus {NUM_CLASSES} kelas dengan kelas 0 = "
                          f"{BACKGROUND_CLASS!r}, dapat {len(classes)} kelas")
    return classes


def project_root() -> Path:
    """Folder berisi pyproject.toml, dicari naik dari lokasi paket (editable install)."""
    for d in (_PACKAGE_DIR, *_PACKAGE_DIR.parents):
        if (d / PROJECT_MARKER).is_file():
            return d
    raise ConfigError(f"root project (folder berisi {PROJECT_MARKER}) tidak ditemukan dari lokasi paket "
                      f"{_PACKAGE_DIR} — install editable (pip install -e .) atau pakai path absolut")


# ── Load: default → YAML → overrides → tipe → validasi ─
def _load(cls, path, overrides, resolve, moved, validate):
    merged = _defaults(cls)
    source = "default"
    if path is not None:
        source = str(path)
        data = _read_yaml(Path(path))
        try:
            merged = _merge(merged, data, "", moved)
        except ConfigError as e:
            raise ConfigError(f"{path}: {e}") from None
    if overrides:
        try:
            merged = _merge(merged, _unflatten(overrides), "", moved)
        except ConfigError as e:
            raise ConfigError(f"overrides: {e}") from None
        source += " + overrides"
    try:
        cfg = _build(cls, merged, "", resolve)
        validate(cfg)
    except ConfigError as e:
        raise ConfigError(f"{source}: {e}") from None
    return cfg


def _defaults(cls) -> dict:
    """Default kode sebagai dict bersarang (bentuk YAML); groups → {nama: [kelas]}."""
    data = to_dict(cls())
    if "groups" in data:
        data["groups"] = {name: classes for name, classes in data["groups"]}
    return data


def _read_yaml(p: Path) -> dict:
    if not p.is_file():
        raise ConfigError(f"file config tidak ditemukan: {p}")
    text = p.read_text(encoding="utf-8")
    try:
        data = yaml.safe_load(text)
        _check_duplicate_keys(yaml.compose(text, Loader=yaml.SafeLoader), "", p)
    except yaml.YAMLError as e:
        raise ConfigError(f"{p}: YAML tidak valid — {e}\nTip: backslash dalam kutip GANDA adalah escape; "
                          f"untuk path Windows pakai '/' (\"D:/rotoscope/work\") atau kutip tunggal.") from None
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"{p}: isi harus mapping (key: nilai), bukan {type(data).__name__}")
    return data


def _check_duplicate_keys(node, prefix: str, p: Path) -> None:
    if isinstance(node, yaml.MappingNode):
        seen = set()
        for k_node, v_node in node.value:
            key = _join(prefix, k_node.value)
            if k_node.value in seen:
                raise ConfigError(f"{p}: key duplikat {key} (baris {k_node.start_mark.line + 1})")
            seen.add(k_node.value)
            _check_duplicate_keys(v_node, key, p)
    elif isinstance(node, yaml.SequenceNode):
        for item in node.value:
            _check_duplicate_keys(item, prefix, p)


def _join(prefix: str, k: Any) -> str:
    return f"{prefix}.{k}" if prefix else str(k)


def _unflatten(overrides: Mapping[str, Any]) -> dict:
    out: dict = {}
    for dotted, v in overrides.items():
        *parents, leaf = dotted.split(".")
        d = out
        for part in parents:
            d = d.setdefault(part, {})
        d[leaf] = v
    return out


def _merge(base: dict, over: Any, prefix: str, moved: Mapping[str, str]) -> dict:
    if over is None:
        return base  # section kosong = default
    if not isinstance(over, dict):
        raise ConfigError(f"{prefix or '<root>'} ({_fmt(over)}) harus mapping (key: nilai)")
    out = dict(base)
    for k, v in over.items():
        key = _join(prefix, k)
        if k not in base:
            raise ConfigError(_unknown_key_msg(key, k, base, prefix, moved))
        if key == "groups" or not isinstance(base[k], dict):
            out[k] = v  # groups diganti utuh
        else:
            out[k] = _merge(base[k], v, key, moved)
    return out


def _unknown_key_msg(key: str, k: Any, base: dict, prefix: str, moved: Mapping[str, str]) -> str:
    if key in moved:
        return f"{key}: key tidak dikenal — {moved[key]}"
    valid = [str(x) for x in base]
    close = difflib.get_close_matches(str(k), valid, n=_SUGGEST_MAX, cutoff=_SUGGEST_CUTOFF)
    if close:
        return f"{key}: key tidak dikenal — maksudnya {' / '.join(map(repr, close))}?"
    return f"{key}: key tidak dikenal. Key yang valid di {prefix or 'level atas'}: {', '.join(valid)}"


@lru_cache(maxsize=None)
def _hints(cls) -> dict[str, Any]:
    return get_type_hints(cls)


def _build(cls, data: dict, prefix: str, resolve):
    kwargs = {}
    for f in dataclasses.fields(cls):
        key = _join(prefix, f.name)
        kwargs[f.name] = _convert(_hints(cls)[f.name], data[f.name], key, resolve)
    return cls(**kwargs)


def _convert(hint, v, key: str, resolve):
    if key == "groups":
        return _build_groups(v)
    if dataclasses.is_dataclass(hint):
        if not isinstance(v, dict):
            _type_error(key, v, "mapping (key: nilai)")
        return _build(hint, v, key, resolve)
    if get_origin(hint) is Mapping:
        if not isinstance(v, dict):
            _type_error(key, v, "mapping (key: nilai)")
        value_hint = get_args(hint)[1]
        return MappingProxyType({k: _convert(value_hint, vv, _join(key, k), resolve) for k, vv in v.items()})
    if v is None:
        if hint == (str | None):
            return None
        tip = (" — di YAML, '#' tanpa kutip dibaca sebagai komentar; kutip nilainya, mis. \"#1a1a1a\""
               if hint in (str, Path) else "")
        raise ConfigError(f"{key} (null) tidak boleh null{tip}")
    if hint is bool:
        if not isinstance(v, bool):
            _type_error(key, v, "bool (true/false)")
        return v
    if hint is int:
        if isinstance(v, bool) or not isinstance(v, int):
            _type_error(key, v, "int")
        return v
    if hint is float:
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            _type_error(key, v, "angka hingga")
        return float(v)
    if hint in (str, str | None):
        if not isinstance(v, str) or not v.strip():
            _type_error(key, v, "string tidak kosong")
        return v
    if hint is Path:
        return _convert_path(key, v, resolve)
    raise TypeError(f"tipe skema tidak didukung: {key}: {hint!r}")


def _convert_path(key: str, v: Any, resolve) -> Path:
    if not isinstance(v, str) or not v.strip():
        _type_error(key, v, "path (string tidak kosong)")
    bad = [c for c in v if ord(c) < 0x20 or ord(c) == 0x7F]
    if bad:
        raise ConfigError(
            f"{key} ({v!r}) mengandung karakter kontrol {bad[0]!r}. Di YAML, backslash dalam kutip GANDA "
            f"adalah escape (\"\\r\" = carriage return). Pakai '/' (\"D:/rotoscope/work\") atau kutip "
            f"tunggal ('D:\\rotoscope\\work')")
    p = Path(v)
    if not p.is_absolute():
        p = resolve(key, v) / p
    return Path(os.path.normpath(p))  # tanpa akses filesystem


def _from_cwd(key: str, v: str) -> Path:
    return Path.cwd()


def _from_project_root(key: str, v: str) -> Path:
    try:
        return project_root()
    except ConfigError as e:
        raise ConfigError(f"{key} ({v!r}) relatif: {e}") from None


def _build_groups(v: Any) -> tuple[tuple[str, tuple[str, ...]], ...]:
    if not isinstance(v, dict) or not v:
        raise ConfigError(f"groups ({_fmt(v)}) harus mapping nama_grup: [kelas, …] yang tidak kosong")
    if len(v) > MAX_GROUPS:
        raise ConfigError(f"groups ({len(v)} grup) maksimal {MAX_GROUPS} grup (id grup uint8, 0 = background)")
    classes = load_class_names()
    owner: dict[str, str] = {}
    out = []
    for name, members in v.items():
        key = f"groups.{name}"
        if not isinstance(name, str) or not GROUP_NAME_RE.fullmatch(name):
            raise ConfigError(f"{key}: nama grup {name!r} harus identifier huruf kecil "
                              f"(a-z, 0-9, _; tidak diawali angka)")
        if name == BACKGROUND_GROUP:
            raise ConfigError(f"{key}: nama grup {BACKGROUND_GROUP!r} dicadangkan (id 0)")
        if not isinstance(members, list) or not members:
            raise ConfigError(f"{key} ({_fmt(members)}) harus list kelas yang tidak kosong")
        for c in members:
            if c == BACKGROUND_CLASS:
                raise ConfigError(f"{key}: {BACKGROUND_CLASS!r} tidak boleh dicantumkan (otomatis grup 0)")
            if not isinstance(c, str) or c not in classes:
                close = difflib.get_close_matches(str(c), classes, n=_SUGGEST_MAX, cutoff=_SUGGEST_CUTOFF)
                hint = f" — maksudnya {' / '.join(map(repr, close))}?" if close else ""
                raise ConfigError(f"{key}: kelas {c!r} tidak ada di daftar {NUM_CLASSES} kelas Sapiens2{hint}")
            if c in owner:
                raise ConfigError(f"{key}: kelas {c!r} sudah ada di groups.{owner[c]} — "
                                  f"tiap kelas masuk tepat satu grup")
            owner[c] = name
        out.append((name, tuple(members)))
    missing = [c for c in classes if c != BACKGROUND_CLASS and c not in owner]
    if missing:
        raise ConfigError(f"groups: kelas tidak tercantum di grup mana pun: {', '.join(missing)} — "
                          f"kelas yang tidak tercantum TIDAK otomatis jadi background")
    return tuple(out)


# ── Validasi (docs/02) ─────────────────────────────
def _fmt(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    if isinstance(v, Path):
        return repr(v.as_posix())
    return repr(v)


def _fail(key: str, v: Any, rule: str) -> None:
    raise ConfigError(f"{key} ({_fmt(v)}) {rule}")


def _type_error(key: str, v: Any, expected: str) -> None:
    _fail(key, v, f"harus {expected}, bukan {type(v).__name__}")


def _choice(key: str, v: Any, options: tuple) -> None:
    if v not in options:
        _fail(key, v, f"harus salah satu dari {', '.join(map(repr, options))}")


def _between(key: str, v: float, lo: float, hi: float, lo_open: bool = False, hi_open: bool = False) -> None:
    ok = (v > lo if lo_open else v >= lo) and (v < hi if hi_open else v <= hi)
    if not ok:
        _fail(key, v, f"harus di rentang {'(' if lo_open else '['}{_fmt(lo)}, {_fmt(hi)}{')' if hi_open else ']'}")


def _unit(key: str, v: float) -> None:
    _between(key, v, 0, 1)


def _at_least(key: str, v: float, lo: float, strict: bool = False) -> None:
    if (v <= lo) if strict else (v < lo):
        _fail(key, v, f"harus {'>' if strict else '≥'} {_fmt(lo)}")


def _odd(key: str, v: int, lo: int) -> None:
    if v < lo or v % 2 == 0:
        _fail(key, v, f"harus int ganjil ≥ {lo}")


def _less(key_a: str, a: float, key_b: str, b: float) -> None:
    if not a < b:
        raise ConfigError(f"{key_a} ({_fmt(a)}) harus < {key_b} ({_fmt(b)})")


def _precision(key: str, v: str) -> None:
    if v == "bf16":
        _fail(key, v, "ditolak — GPU Turing tanpa bf16 (P-005); pakai 'fp16' atau 'fp32'")
    _choice(key, v, PRECISIONS)


def _hex(key: str, v: str) -> None:
    if not HEX_COLOR_RE.fullmatch(v):
        _fail(key, v, "harus warna hex '#rrggbb'")


def _validate_pipeline(c: PipelineConfig) -> None:
    # model_ids / vram_min_free_mib: key persis SEG_MODELS dijamin merge (key lain = tidak dikenal).
    s = c.segment
    _choice("segment.model", s.model, SEG_MODELS)
    for m, model_id in s.model_ids.items():
        if not model_id.startswith(SEG_MODEL_ID_PREFIX):
            _fail(f"segment.model_ids.{m}", model_id,
                  f"wajib diawali {SEG_MODEL_ID_PREFIX!r} — Sapiens v1 (facebook/sapiens-seg-…) "
                  f"berlisensi CC-BY-NC, dilarang")
    _precision("segment.precision", s.precision)
    for m, mib in s.vram_min_free_mib.items():
        _between(f"segment.vram_min_free_mib.{m}", mib, 0, VRAM_MAX_MIB)
    _choice("segment.probs_dtype", s.probs_dtype, PROBS_DTYPES)

    d = c.depth
    if DEPTH_MODEL_ID_REQUIRED not in d.model_id:
        _fail("depth.model_id", d.model_id,
              f"wajib varian {DEPTH_MODEL_ID_REQUIRED} (Apache-2.0) — Base/Large berlisensi CC-BY-NC, dilarang")
    _precision("depth.precision", d.precision)
    _between("depth.vram_min_free_mib", d.vram_min_free_mib, 0, VRAM_MAX_MIB)

    q = c.qc
    _unit("qc.area_min", q.area_min)
    _unit("qc.area_max", q.area_max)
    _less("qc.area_min", q.area_min, "qc.area_max", q.area_max)
    _unit("qc.iou_min", q.iou_min)
    _unit("qc.blob_min", q.blob_min)
    _unit("qc.area_drop_min", q.area_drop_min)
    _at_least("qc.max_big_blobs", q.max_big_blobs, 1)
    _odd("qc.area_median_window", q.area_median_window, 3)

    t = c.stabilize.temporal
    _between("stabilize.temporal.mask_ema_alpha", t.mask_ema_alpha, 0, 1, lo_open=True)
    _unit("stabilize.temporal.optical_flow_blend", t.optical_flow_blend)
    _unit("stabilize.temporal.boil_preserve", t.boil_preserve)
    _unit("stabilize.temporal.qc_fail_weight", t.qc_fail_weight)
    _at_least("stabilize.island_min_px", c.stabilize.island_min_px, 0)
    _odd("stabilize.mode_k", c.stabilize.mode_k, 1)
    _choice("stabilize.depth.normalize", c.stabilize.depth.normalize, NORMALIZE_METHODS)

    v = c.vectorize
    _at_least("vectorize.min_region_area", v.min_region_area, 0)
    _at_least("vectorize.min_hole_area", v.min_hole_area, 0)
    _at_least("vectorize.line_min_px", v.line_min_px, 1)
    _at_least("vectorize.min_stroke_px", v.min_stroke_px, 1)
    dl = v.depth_lines
    _at_least("vectorize.depth_lines.blur_sigma", dl.blur_sigma, 0)
    _between("vectorize.depth_lines.lo_pct", dl.lo_pct, 0, 100, lo_open=True, hi_open=True)
    _between("vectorize.depth_lines.hi_pct", dl.hi_pct, 0, 100, lo_open=True, hi_open=True)
    _less("vectorize.depth_lines.lo_pct", dl.lo_pct, "hi_pct", dl.hi_pct)
    _odd("vectorize.depth_lines.erode_px", dl.erode_px, 1)
    _at_least("vectorize.depth_lines.min_dist_px", dl.min_dist_px, 0)
    _at_least("vectorize.depth_lines.min_len_px", dl.min_len_px, 0)
    _at_least("vectorize.track.max_match_dist_px", v.track.max_match_dist_px, 0, strict=True)


def _validate_style(c: StyleConfig) -> None:
    sh = c.shape
    _at_least("shape.simplify_epsilon", sh.simplify_epsilon, 0)
    _at_least("shape.resample_points", sh.resample_points, RESAMPLE_POINTS_MIN)
    _unit("shape.smooth_tension", sh.smooth_tension)
    _at_least("shape.spline_steps", sh.spline_steps, 1)

    st = c.stroke
    _at_least("stroke.width_base", st.width_base, 0, strict=True)
    _unit("stroke.width_variation", st.width_variation)
    _unit("stroke.opacity", st.opacity)
    _unit("stroke.taper_min", st.taper_min)
    _at_least("stroke.width_noise_scale", st.width_noise_scale, 0)
    _hex("stroke.color", st.color)
    _choice("stroke.cap", st.cap, STROKE_CAPS)
    _at_least("stroke.taper_px", st.taper_px, 0)
    for t, scale in st.by_type.items():
        _at_least(f"stroke.by_type.{t}.width_scale", scale.width_scale, 0, strict=True)
        _unit(f"stroke.by_type.{t}.opacity_scale", scale.opacity_scale)

    j = c.jitter
    _at_least("jitter.amplitude", j.amplitude, 0)
    _at_least("jitter.frequency", j.frequency, 0)
    _choice("jitter.temporal_seed_mode", j.temporal_seed_mode, TEMPORAL_SEED_MODES)
    _unit("jitter.temporal_drift", j.temporal_drift)

    mp = c.multipass
    _at_least("multipass.passes", mp.passes, 1)
    _at_least("multipass.offset", mp.offset, 0)
    _unit("multipass.opacity_falloff", mp.opacity_falloff)

    tx = c.texture
    _choice("texture.mode", tx.mode, TEXTURE_MODES)
    _at_least("texture.stamp_spacing", tx.stamp_spacing, 0, strict=True)
    _unit("texture.pressure_noise", tx.pressure_noise)
    _unit("texture.grain_strength", tx.grain_strength)
    if tx.mode == TEXTURE_MODE_BRUSH and not tx.brush_image.is_file():
        _fail("texture.brush_image", tx.brush_image, f"tidak ada (dipakai karena texture.mode = {tx.mode!r})")

    p = c.paper
    _hex("paper.color", p.color)
    _unit("paper.texture_opacity", p.texture_opacity)
    _unit("paper.vignette", p.vignette)
    if p.enabled and not p.texture_image.is_file():
        _fail("paper.texture_image", p.texture_image, "tidak ada (dipakai karena paper.enabled = true)")

    _between("render.ss", c.render.ss, 1, RENDER_SS_MAX)
