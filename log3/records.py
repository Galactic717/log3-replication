"""Запис журналу: рівно 16 байтів, big-endian (FORMAT_SPEC_P3 §1).

Зсув  Розмір  Поле           Тип
0     4       timestamp_ms   uint32
4     2       source_id      uint16, 1..4095
6     2       event_code     uint16
8     4       metric_value   int32
12    1       category       uint8, 0..15
13    1       severity       uint8, 0..7
14    1       critical       uint8, 0 або 1
15    1       record_flags   uint8, 0..3

Ідентичність запису — його точне 16-байтове подання (§1), тому дедуплікація
у відновлювачі йде саме за pack_record(r).
"""
from __future__ import annotations

import csv
import struct
from dataclasses import dataclass, fields
from pathlib import Path

RECORD_STRUCT = struct.Struct(">IHHiBBBB")
RECORD_SIZE = RECORD_STRUCT.size  # 16

# Допустимі діапазони полів (включно). Типи uint/int перевіряє сам struct,
# тут — додаткові обмеження специфікації.
_RANGES = {
    "timestamp_ms": (0, 2**32 - 1),
    "source_id": (1, 4095),
    "event_code": (0, 2**16 - 1),
    "metric_value": (-(2**31), 2**31 - 1),
    "category": (0, 15),
    "severity": (0, 7),
    "critical": (0, 1),
    "record_flags": (0, 3),
}


@dataclass(frozen=True)
class Record:
    timestamp_ms: int
    source_id: int
    event_code: int
    metric_value: int
    category: int
    severity: int
    critical: int
    record_flags: int


FIELD_NAMES = tuple(f.name for f in fields(Record))


def validate_record(r: Record) -> None:
    """ValueError, якщо хоч одне поле поза діапазоном специфікації."""
    for name in FIELD_NAMES:
        value = getattr(r, name)
        lo, hi = _RANGES[name]
        if not isinstance(value, int) or isinstance(value, bool) or not lo <= value <= hi:
            raise ValueError(f"{name}={value!r} поза діапазоном {lo}..{hi}")


def pack_record(r: Record) -> bytes:
    validate_record(r)
    return RECORD_STRUCT.pack(*(getattr(r, n) for n in FIELD_NAMES))


def unpack_record(b: bytes) -> Record:
    if len(b) != RECORD_SIZE:
        raise ValueError(f"запис має бути {RECORD_SIZE} байтів, отримано {len(b)}")
    r = Record(*RECORD_STRUCT.unpack(b))
    validate_record(r)
    return r


def pack_records(rs: list[Record]) -> bytes:
    return b"".join(pack_record(r) for r in rs)


def unpack_records(b: bytes) -> list[Record]:
    if len(b) % RECORD_SIZE:
        raise ValueError(f"довжина {len(b)} не кратна {RECORD_SIZE}")
    return [unpack_record(b[i:i + RECORD_SIZE]) for i in range(0, len(b), RECORD_SIZE)]


# Назви колонки з міткою класу, які приймаємо в CSV. Точну назву звірити з train.csv
# (див. docs/DECISIONS.md) — поки materials/ без CSV.
LABEL_COLUMNS = ("label", "class", "cls", "y")


def load_records_csv(path: str | Path) -> tuple[list[Record], list[int] | None]:
    """(записи, мітки класів або None, якщо колонки мітки нема). Зайві колонки ігноруються."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        missing = [n for n in FIELD_NAMES if n not in header]
        if missing:
            raise ValueError(f"{path}: бракує колонок {missing}")
        label_col = next((c for c in LABEL_COLUMNS if c in header), None)
        records: list[Record] = []
        labels: list[int] = []
        for row in reader:
            r = Record(*(int(row[n]) for n in FIELD_NAMES))
            validate_record(r)
            records.append(r)
            if label_col:
                labels.append(int(row[label_col]))
    return records, (labels if label_col else None)
