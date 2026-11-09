"""Відновлення журналу з реплік, що пережили сценарій (методичка 11.56–11.57).

Кожна репліка проходить повне приймання (HMAC до розшифрування); зіпсовані
відкидаються й лічаться. Дублікати (кілька копій однієї події) зливаються за
ключем ідентичності FORMAT_SPEC_P3 §1 — точним 16-байтовим поданням запису:
у сеансах timestamp_ms унікальний, тож різні події не склеюються, а копії
однієї події не рахуються двічі (інакше R було б завищене або занижене).
"""
from __future__ import annotations

from .records import Record, pack_record
from .replica import Log3Error, parse_replica


def recover(blobs: list[bytes], password: str | bytes) -> tuple[list[Record], int]:
    """(унікальні записи, відсортовані за timestamp_ms; кількість відхилених реплік)."""
    unique: dict[bytes, Record] = {}
    rejected = 0
    for blob in blobs:
        try:
            replica = parse_replica(blob, password)
        except Log3Error:
            rejected += 1
            continue
        for r in replica.records:
            unique.setdefault(pack_record(r), r)
    ordered = sorted(unique.items(), key=lambda kv: (kv[1].timestamp_ms, kv[0]))
    return [r for _, r in ordered], rejected
