"""Версійовані обмежені за ресурсами ознаки блока.

Кожна ознака відома ДО рішення про стиснення цього блока. Витрати фіксовані незалежно від
розміру файла: читаються лише три вікна по 4 КіБ і пробний zstd-1 на двох вікнах по 16 КіБ —
це «виміряна стисливість» вибірки, а не результат кодування самого блока (його в ознаках нема,
він приходить окремо як навчальна мітка). Назви файлів, шляхи й байти вмісту не зберігаються:
розширення зводиться до класу типу, сигнатура — до класу формату, файл — до знеособленого
ключа лінії версій (hash від file_id).

Порядок і визначення ознак — частина схеми. Будь-яка зміна набору, порядку чи обчислення
ознаки потребує нового FEATURE_VERSION: артефакт моделі з іншою схемою не завантажується
(контрольований fallback на базову політику).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import time
from dataclasses import dataclass

import numpy as np
import zstandard

FEATURE_SCHEMA = "log3-chunk-features"
FEATURE_VERSION = 1
WINDOW = 4096
PROBE_WINDOW = 8192
LAGS = (1, 2, 4, 8)
_local = threading.local()

TYPE_CLASSES = {
    "text": "txt md log rst ini cfg conf rtf",
    "structured": "csv tsv json xml yaml yml html htm",
    "code": "py js ts c h cpp hpp java cs go rs sh ps1 sql php rb",
    "office_packed": "docx xlsx pptx odt ods odp",
    "archive": "zip 7z rar gz tgz bz2 xz zst jar apk",
    "image_packed": "jpg jpeg png gif webp heic",
    "media": "mp3 mp4 m4a mkv mov avi webm flac ogg",
    "pdf": "pdf",
    "executable": "exe dll so dylib msi",
    "database": "db sqlite sqlite3 mdb accdb",
    "disk_image": "img iso vhd vhdx vmdk qcow2",
}
SUFFIX_CLASS = {s: k for k, v in TYPE_CLASSES.items() for s in v.split()}
TYPE_NAMES = (*TYPE_CLASSES, "other")
MAGIC = (
    ("packed", (b"PK\x03\x04", b"\x1f\x8b", b"\x28\xb5\x2f\xfd", b"\xfd7zXZ\x00", b"BZh", b"7z\xbc\xaf\x27\x1c",
                b"\xff\xd8\xff", b"\x89PNG", b"GIF8", b"RIFF", b"\x1aE\xdf\xa3", b"fLaC", b"OggS")),
    ("pdf", (b"%PDF",)),
    ("executable", (b"MZ", b"\x7fELF", b"\xcf\xfa\xed\xfe")),
    ("sqlite", (b"SQLite format 3\x00",)),
)
MAGIC_NAMES = (*(m for m, _ in MAGIC), "text", "unknown")

NUMERIC = (
    ("log2_chunk_len", "log2(байти блока + 1) / 26"),
    ("log2_file_size", "log2(розмір файла + 1) / 40"),
    ("chunk_position", "частка файла до початку блока, 0..1"),
    ("entropy_sample", "ентропія Шеннона трьох вікон по 4 КіБ, біт/байт / 8"),
    ("entropy_head", "ентропія перших 4 КіБ блока (як у базовій політиці) / 8"),
    ("distinct_bytes", "частка різних значень байта у вибірці"),
    ("repeat_lag", "повторюваність: найбільша частка байтів, що дорівнюють байту на лагах 1/2/4/8, у вибірці"),
    ("zero_bytes", "частка нульових байтів у вибірці"),
    ("printable_bytes", "частка друкованих ASCII і пробільних байтів у вибірці"),
    ("probe_ratio", "виміряна стисливість: zstd-1 на 2 вікнах по 8 КіБ, вихід/вхід"),
    ("probe_log_ns_per_byte", "log10 часу пробного стиснення, нс/байт (швидкість машини зараз)"),
    ("lineage_known", "1, якщо файл уже бачили в попередніх версіях"),
    ("lineage_versions", "log1p(скільки версій бачили) / 5"),
    ("lineage_change_rate", "частка попередніх версій, у яких вміст файла змінився"),
    ("queue_fill", "заповнення черги реплікації 0..1 на момент рішення"),
    ("spool_fill", "заповнення spool 0..1"),
    ("load_per_cpu", "середнє навантаження ОС за 1 хв на один CPU, обрізано до 4"),
)
NAMES = tuple(n for n, _ in NUMERIC) + tuple(f"type_{t}" for t in TYPE_NAMES) + tuple(f"magic_{m}" for m in MAGIC_NAMES)
N_FEATURES = len(NAMES)
_IDX = {n: i for i, n in enumerate(NAMES)}
#: ознаки, яких може не бути на момент рішення (маска доступності)
OPTIONAL = ("lineage_versions", "lineage_change_rate", "queue_fill", "spool_fill", "load_per_cpu")


def schema() -> dict:
    """Опис схеми для артефакту моделі: порядок і визначення ознак."""
    return {"schema": FEATURE_SCHEMA, "version": FEATURE_VERSION, "names": list(NAMES),
            "definitions": dict(NUMERIC) | {f"type_{t}": f"клас розширення {t}" for t in TYPE_NAMES}
            | {f"magic_{m}": f"клас сигнатури {m}" for m in MAGIC_NAMES},
            "type_classes": TYPE_CLASSES, "window_bytes": WINDOW, "probe_window_bytes": PROBE_WINDOW,
            "repeat_lags": list(LAGS)}


def schema_id() -> str:
    raw = json.dumps(schema(), ensure_ascii=False, sort_keys=True).encode("utf-8")
    return f"{FEATURE_SCHEMA}/{FEATURE_VERSION}/{hashlib.sha256(raw).hexdigest()[:16]}"


@dataclass(frozen=True)
class Telemetry:
    """Операційний стан на момент рішення; None — недоступно."""
    queue_fill: float | None = None
    spool_fill: float | None = None
    load_per_cpu: float | None = None


@dataclass(frozen=True)
class Lineage:
    versions_seen: int = 0
    content_changes: int = 0


@dataclass(frozen=True)
class FeatureVector:
    values: np.ndarray        # float32[N_FEATURES]
    mask: np.ndarray          # bool[N_FEATURES]: ознака доступна на момент рішення
    cost_ns: int              # скільки коштувало обчислення

    def get(self, name: str) -> float:
        return float(self.values[_IDX[name]])


def type_class(suffix: str) -> str:
    return SUFFIX_CLASS.get(suffix.lower(), "other")


def magic_class(head: bytes) -> str:
    for name, sigs in MAGIC:
        if any(head.startswith(s) for s in sigs):
            return name
    sample = head[:16]
    if sample and all(32 <= b < 127 or b in (9, 10, 13) for b in sample):
        return "text"
    try:
        sample.decode("utf-8")
        return "text" if sample and not sample.count(0) else "unknown"
    except UnicodeDecodeError:
        return "unknown"


def _entropy(counts: np.ndarray, n: int) -> float:
    if n == 0:
        return 0.0
    p = counts[counts > 0] / n
    return float(-(p * np.log2(p)).sum())


def _windows(data: bytes, size: int, count: int) -> list[bytes]:
    n = len(data)
    if n <= size * count:
        return [data]
    starts = [0, n // 2 - size // 2, n - size][:count] if count == 3 else [0, n // 2 - size // 2][:count]
    return [data[s:s + size] for s in starts]


def probe_compressibility(data: bytes) -> tuple[float, float]:
    """(вихід/вхід zstd-1 на вибірці, log10 нс/байт)."""
    sample = b"".join(_windows(data, PROBE_WINDOW, 2))
    if not sample:
        return 1.0, 0.0
    comp = getattr(_local, "zstd1", None)
    if comp is None:                                   # компресор не потокобезпечний: свій на потік
        comp = _local.zstd1 = zstandard.ZstdCompressor(level=1)
    t0 = time.perf_counter_ns()
    out = comp.compress(sample)
    dt = max(1, time.perf_counter_ns() - t0)
    return min(1.5, len(out) / len(sample)), math.log10(dt / len(sample))


def load_per_cpu() -> float | None:
    try:
        return min(4.0, os.getloadavg()[0] / (os.cpu_count() or 1))
    except (AttributeError, OSError):          # Windows: getloadavg нема
        return None


def extract(data: bytes, *, suffix: str, file_size: int, chunk_index: int, chunk_size: int,
            head_entropy_bits: float, lineage: Lineage | None, telemetry: Telemetry) -> FeatureVector:
    t0 = time.perf_counter_ns()
    v = np.zeros(N_FEATURES, dtype=np.float32)
    m = np.ones(N_FEATURES, dtype=bool)
    sample = np.frombuffer(b"".join(_windows(data, WINDOW, 3)), dtype=np.uint8)
    n = len(data)
    v[_IDX["log2_chunk_len"]] = math.log2(n + 1) / 26
    v[_IDX["log2_file_size"]] = math.log2(max(file_size, n) + 1) / 40
    v[_IDX["chunk_position"]] = min(1.0, chunk_index * chunk_size / max(1, file_size))
    counts = np.bincount(sample, minlength=256)
    v[_IDX["entropy_sample"]] = _entropy(counts, sample.size) / 8
    v[_IDX["entropy_head"]] = head_entropy_bits / 8
    if sample.size:
        v[_IDX["distinct_bytes"]] = np.count_nonzero(counts) / 256
        v[_IDX["zero_bytes"]] = counts[0] / sample.size
        v[_IDX["printable_bytes"]] = (counts[32:127].sum() + counts[9] + counts[10] + counts[13]) / sample.size
    if sample.size > LAGS[-1]:
        v[_IDX["repeat_lag"]] = max(np.count_nonzero(sample[k:] == sample[:-k]) / (sample.size - k) for k in LAGS)
    v[_IDX["probe_ratio"]], v[_IDX["probe_log_ns_per_byte"]] = probe_compressibility(data)
    if lineage is not None and lineage.versions_seen > 0:
        v[_IDX["lineage_known"]] = 1.0
        v[_IDX["lineage_versions"]] = math.log1p(lineage.versions_seen) / 5
        v[_IDX["lineage_change_rate"]] = lineage.content_changes / max(1, lineage.versions_seen - 1) \
            if lineage.versions_seen > 1 else 0.0
    else:
        m[_IDX["lineage_versions"]] = m[_IDX["lineage_change_rate"]] = False
    for name in ("queue_fill", "spool_fill", "load_per_cpu"):
        val = getattr(telemetry, name)
        if val is None:
            m[_IDX[name]] = False
        else:
            v[_IDX[name]] = float(val)
    v[_IDX[f"type_{type_class(suffix)}"]] = 1.0
    v[_IDX[f"magic_{magic_class(data[:16])}"]] = 1.0
    return FeatureVector(v, m, time.perf_counter_ns() - t0)


def index(name: str) -> int:
    return _IDX[name]
