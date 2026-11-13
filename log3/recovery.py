"""Відновлення журналу з реплік, що пережили сценарій (методичка 11.56–11.57).

Кожна репліка проходить повне приймання (HMAC до розшифрування); зіпсовані
відкидаються й лічаться. Дублікати (кілька копій однієї події) зливаються за
ключем ідентичності FORMAT_SPEC_P3 §1 — точним 16-байтовим поданням запису:
у сеансах timestamp_ms унікальний, тож різні події не склеюються, а копії
однієї події не рахуються двічі (інакше R було б завищене або занижене).

PBKDF2 (100000 ітерацій) рахується на кожну нову сіль ДО перевірки HMAC — так велить
формат. Тому сміттєві репліки з різними солями могли б обвалити T: обробляємо солі за
спаданням частоти й не більше MAX_SALTS різних (наш сеанс має одну сіль на всі репліки).
"""
from __future__ import annotations

from collections import Counter

from .records import Record, pack_record
from .replica import Log3Error, parse_header, parse_replica

MAX_SALTS = 16


def _salt(blob: bytes) -> bytes | None:
    try:
        return parse_header(blob).salt
    except Log3Error:
        return None


def recover(blobs: list[bytes], password: str | bytes) -> tuple[list[Record], int]:
    """(унікальні записи, відсортовані за timestamp_ms; кількість відхилених реплік)."""
    unique: dict[bytes, Record] = {}
    salts = Counter(s for s in map(_salt, blobs) if s is not None)
    allowed = {s for s, _ in salts.most_common(MAX_SALTS)}
    rejected = 0
    for blob in blobs:
        s = _salt(blob)
        if s is not None and s not in allowed:          # забагато різних солей — без PBKDF2
            rejected += 1
            continue
        try:
            replica = parse_replica(blob, password)
        except Log3Error:
            rejected += 1
            continue
        for r in replica.records:
            unique.setdefault(pack_record(r), r)
    ordered = sorted(unique.items(), key=lambda kv: (kv[1].timestamp_ms, kv[0]))
    return [r for _, r in ordered], rejected
