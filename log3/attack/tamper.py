"""Tamper-тест: змінюємо репліку і перевіряємо, що приймач її відхиляє.

Модуль приймач-агностичний: працює з будь-яким receiver(blob) -> код помилки | None
(None — репліку прийнято) і НЕ імпортує наш parse_replica. На тижні змагань той самий
тест спрямовують на приймач іншої команди; адаптер до нашого — replica.as_receiver(password).

Види змін (кожна гарантовано дає інші байти):
  bitflip — інвертувати один біт; byte — замінити байт іншим значенням;
  truncate — відрізати хвіст; extend — дописати байти; swap — переставити два різні байти.
Для LOG3 очікуємо: зміни шифротексту чи HMAC -> E_HMAC (HMAC перевіряється до розшифрування),
magic -> E_MAGIC, довжина -> E_SHORT/E_LENGTH. Жодна змінена репліка не має бути прийнята.
"""
from __future__ import annotations

import random
from collections import Counter
from typing import Callable

KINDS = ("bitflip", "byte", "truncate", "extend", "swap")


def mutate(blob: bytes, kind: str, rng: random.Random) -> bytes:
    b = bytearray(blob)
    n = len(b)
    if kind == "bitflip":
        i = rng.randrange(n)
        b[i] ^= 1 << rng.randrange(8)
    elif kind == "byte":
        i = rng.randrange(n)
        b[i] = (b[i] + rng.randrange(1, 256)) % 256
    elif kind == "truncate":
        del b[rng.randrange(n):]
    elif kind == "extend":
        b += bytes(rng.randrange(256) for _ in range(rng.randint(1, 16)))
    elif kind == "swap":
        i, j = rng.randrange(n), rng.randrange(n)
        if b[i] == b[j]:                                   # перестановка однакових — не зміна
            b[i] ^= 0xFF
        else:
            b[i], b[j] = b[j], b[i]
    else:
        raise ValueError(kind)
    assert bytes(b) != blob
    return bytes(b)


def tamper_test(blob: bytes, receiver: Callable[[bytes], str | None], *, trials: int,
                seed: int) -> dict[str, int]:
    """{'trials', 'accepted', 'rejected', <код помилки>: кількість, 'kind:<вид>': кількість,
    'accepted_kind:<вид>': кількість}. accepted > 0 — приймач пропустив зміну (вразливість)."""
    rng = random.Random(seed)
    c: Counter[str] = Counter()
    for _ in range(trials):
        kind = rng.choice(KINDS)
        try:
            code = receiver(mutate(blob, kind, rng))
        except Exception as e:                             # приймач упав — теж відхилення, але погане
            code = f"CRASH:{type(e).__name__}"
        c[f"kind:{kind}"] += 1
        if code is None:
            c["accepted"] += 1
            c[f"accepted_kind:{kind}"] += 1
        else:
            c["rejected"] += 1
            c[code] += 1
    return {"trials": trials, "accepted": c.pop("accepted", 0), "rejected": c.pop("rejected", 0), **dict(c)}
