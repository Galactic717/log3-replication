"""Канонічний Гаффман (FORMAT_SPEC_P3 §3.2, контракт docs/CONTRACTS.md §2).

Власник зони — Yaroslav (feat/huffman). Цю реалізацію написав голова як
тимчасову, бо без неї не читаються стиснені еталонні репліки (C1 №1, №12).
Сигнатури — з контракту; заміна на версію власника нічого не ламає, якщо
tests/test_huffman.py лишається зеленим.

У потік кладуть лише довжини кодів: обидві сторони однаково відновлюють коди
алгоритмом bl_count/next_code (RFC 1951 §3.2.2), дерево не передається.
"""
from __future__ import annotations

import heapq
from collections import Counter

MAX_LEN = 15


class HuffmanError(ValueError):
    pass


def code_lengths(data: bytes, max_len: int = MAX_LEN) -> dict[int, int]:
    """symbol -> довжина коду. Порожні дані -> {}, один символ -> {sym: 1}."""
    freq = Counter(data)
    if not freq:
        return {}
    if len(freq) == 1:
        return {next(iter(freq)): 1}
    # (вага, унікальний tie-break, вузол): tie-break робить дерево детермінованим
    # і не дає heapq порівнювати самі вузли. Листи — int, внутрішні вузли — пари.
    heap = [(w, s, s) for s, w in sorted(freq.items())]
    heapq.heapify(heap)
    next_id = 256
    while len(heap) > 1:
        w1, _, a = heapq.heappop(heap)
        w2, _, b = heapq.heappop(heap)
        heapq.heappush(heap, (w1 + w2, next_id, (a, b)))
        next_id += 1
    lengths: dict[int, int] = {}
    stack = [(heap[0][2], 0)]
    while stack:
        node, depth = stack.pop()
        if isinstance(node, int):
            lengths[node] = depth
        else:
            stack.append((node[0], depth + 1))
            stack.append((node[1], depth + 1))
    if max(lengths.values()) > max_len:
        lengths = _limit_lengths(lengths, max_len)
    return lengths


def _limit_lengths(lengths: dict[int, int], max_len: int) -> dict[int, int]:
    """Обмеження довжини до max_len (процедура JPEG, ITU T.81 Annex K.3).

    Пара листів із найглибшого рівня піднімається, а найглибший коротший лист
    опускається на рівень і стає їхнім сусідом: сума Крафта лишається рівно 1,
    символи не зливаються. Потім довжини роздаються заново: частіші символи
    (коротші в оптимальному дереві) отримують коротші коди.
    """
    # ponytail: евристика, не оптимальна за довжиною; package-merge, якщо втрата помітна
    deepest = max(lengths.values())
    bl = [0] * (deepest + 1)
    for n in lengths.values():
        bl[n] += 1
    for i in range(deepest, max_len, -1):
        while bl[i] > 0:
            j = i - 2
            while bl[j] == 0:
                j -= 1
            bl[i] -= 2
            bl[i - 1] += 1
            bl[j + 1] += 2
            bl[j] -= 1
    order = iter(sorted(lengths, key=lambda s: (lengths[s], s)))
    return {next(order): n for n in range(1, max_len + 1) for _ in range(bl[n])}


def check_lengths(lengths: dict[int, int]) -> None:
    """HuffmanError, якщо таблиця не задає коректний канонічний код."""
    for s, n in lengths.items():
        if not 0 <= s <= 255:
            raise HuffmanError(f"символ {s} поза 0..255")
        if not 1 <= n <= MAX_LEN:
            raise HuffmanError(f"довжина {n} поза 1..{MAX_LEN}")
    if len(lengths) == 1:
        if next(iter(lengths.values())) != 1:
            raise HuffmanError("для одного символу єдино допустима довжина 1")
    elif len(lengths) >= 2:
        kraft = sum(1 << (MAX_LEN - n) for n in lengths.values())
        if kraft != 1 << MAX_LEN:
            raise HuffmanError(f"сума Крафта {kraft / (1 << MAX_LEN)} != 1")


def canonical_codes(lengths: dict[int, int]) -> dict[int, tuple[int, int]]:
    """symbol -> (code, length) стандартним алгоритмом bl_count/next_code."""
    check_lengths(lengths)
    bl_count = [0] * (MAX_LEN + 1)
    for n in lengths.values():
        bl_count[n] += 1
    next_code = [0] * (MAX_LEN + 1)
    code = 0
    for bits in range(1, MAX_LEN + 1):
        code = (code + bl_count[bits - 1]) << 1
        next_code[bits] = code
    codes = {}
    for s in sorted(lengths):
        n = lengths[s]
        codes[s] = (next_code[n], n)
        next_code[n] += 1
    return codes


def encode(data: bytes, lengths: dict[int, int]) -> bytes:
    """Біти MSB-first, останній байт доповнено нулями."""
    codes = canonical_codes(lengths)
    table = {s: format(c, f"0{n}b") for s, (c, n) in codes.items()}
    try:
        bits = "".join(table[b] for b in data)
    except KeyError as e:
        raise HuffmanError(f"символу {e} нема в таблиці") from None
    if not bits:
        return b""
    bits += "0" * (-len(bits) % 8)
    return int(bits, 2).to_bytes(len(bits) // 8, "big")


def decode(bitstream: bytes, lengths: dict[int, int], symbol_count: int) -> bytes:
    """Рівно symbol_count символів; потік має бути спожитий до останнього байта,
    padding — лише нулі, зайвий повний байт — помилка (bad_07_extra_bitstream)."""
    codes = canonical_codes(lengths)
    if symbol_count == 0:
        if bitstream:
            raise HuffmanError("зайві байти після порожнього потоку")
        return b""
    if not codes:
        raise HuffmanError("порожня таблиця для непорожнього потоку")
    lookup = {(n, c): s for s, (c, n) in codes.items()}
    bits = bin(int.from_bytes(bitstream, "big"))[2:].zfill(8 * len(bitstream)) if bitstream else ""
    out = bytearray()
    pos = code = n = 0
    total = len(bits)
    while len(out) < symbol_count:
        if pos >= total:
            raise HuffmanError("обрізаний потік")
        code = (code << 1) | (bits[pos] == "1")
        pos += 1
        n += 1
        s = lookup.get((n, code))
        if s is not None:
            out.append(s)
            code = n = 0
        elif n >= MAX_LEN:
            raise HuffmanError("неіснуючий код")
    if (pos + 7) // 8 != len(bitstream):
        raise HuffmanError("зайві байти після закодованих даних")
    if "1" in bits[pos:]:
        raise HuffmanError("ненульовий padding")
    return bytes(out)


def encoded_bits(data: bytes, lengths: dict[int, int]) -> int:
    """Скільки бітів займе data з цією таблицею (вага w_i для ДП)."""
    return sum(lengths[b] for b in data)
