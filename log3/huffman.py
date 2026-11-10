"""Канонічний Гаффман: довжини кодів через heapq і канонічні коди (етап 1)."""
from __future__ import annotations

import heapq
from collections import Counter

MAX_LEN = 15


class HuffmanError(ValueError):
    """Некоректна таблиця довжин або пошкоджений потік."""


def code_lengths(data: bytes, max_len: int = MAX_LEN) -> dict[int, int]:
    """symbol -> довжина коду. Порожні дані -> {}, один символ -> {sym: 1}."""
    freq = Counter(data)
    if not freq:
        return {}
    if len(freq) == 1:
        return {next(iter(freq)): 1}
    # елемент купи: (вага, tie-break, вузол). Лист — число-символ, внутрішній вузол — пара.
    # tie-break (символ або лічильник >= 256) робить дерево детермінованим.
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
    if max(lengths.values()) > max_len:             # дерево надто глибоке — будуємо обмежене
        lengths = _package_merge(freq, max_len)
    return lengths

def _package_merge(freq: dict[int, int], max_len: int) -> dict[int, int]:
    """Оптимальні довжини кодів не довші за max_len (алгоритм package-merge).

    Кожен символ — «монета» вагою freq. На кожному з max_len рівнів сусідні
    пари найлегших елементів склеюються в «пакети», які додаються до монет.
    З підсумкового списку беремо 2n-2 найлегших елементів; довжина коду
    символу — скільки разів він у них зустрівся.
    """
    syms = sorted(freq, key=lambda s: (freq[s], s))      # детермінований порядок
    n = len(syms)
    if n > 1 << max_len:
        raise HuffmanError(f"{n} символів не вміщаються в коди довжиною до {max_len}")
    leaves = [(freq[s], (s,)) for s in syms]
    cur = leaves
    for _ in range(max_len - 1):
        packages = [(cur[i][0] + cur[i + 1][0], cur[i][1] + cur[i + 1][1])
                    for i in range(0, len(cur) - 1, 2)]
        cur = sorted(leaves + packages, key=lambda item: item[0])
    lengths: dict[int, int] = {}
    for _, members in cur[:2 * n - 2]:
        for s in members:
            lengths[s] = lengths.get(s, 0) + 1
    return lengths

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
            raise HuffmanError("сума Крафта не дорівнює 1")


def canonical_codes(lengths: dict[int, int]) -> dict[int, tuple[int, int]]:
    """symbol -> (code, length) алгоритмом bl_count / next_code (RFC 1951 §3.2.2)."""
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
    """Стискає data. Біти йдуть від старшого до молодшого, останній байт добиваємо нулями."""
    codes = canonical_codes(lengths)
    bits = []
    for b in data:
        if b not in codes:
            raise HuffmanError(f"символу {b} немає в таблиці")
        code, n = codes[b]
        bits.append(format(code, f"0{n}b"))
    stream = "".join(bits)
    if not stream:
        return b""
    stream += "0" * (-len(stream) % 8)               # padding нулями до цілого байта
    return int(stream, 2).to_bytes(len(stream) // 8, "big")


def decode(bitstream: bytes, lengths: dict[int, int], symbol_count: int) -> bytes:
    """Розпаковує рівно symbol_count символів. Будь-яка вада потоку — HuffmanError."""
    codes = canonical_codes(lengths)
    if symbol_count == 0:
        if bitstream:
            raise HuffmanError("зайві байти після порожніх даних")
        return b""
    if not codes:
        raise HuffmanError("порожня таблиця, а символи очікуються")
    by_code = {(n, c): s for s, (c, n) in codes.items()}   # (довжина, код) -> символ
    bits = "".join(format(byte, "08b") for byte in bitstream)
    out = bytearray()
    pos = 0                                           # скільки бітів вже прочитано
    code = n = 0                                      # код, що зараз складаємо, і його довжина
    while len(out) < symbol_count:
        if pos >= len(bits):
            raise HuffmanError("потік обрізаний")
        code = (code << 1) | int(bits[pos])
        pos += 1
        n += 1
        if (n, code) in by_code:
            out.append(by_code[(n, code)])
            code = n = 0
        elif n >= MAX_LEN:
            raise HuffmanError("такого коду немає в таблиці")
    if (pos + 7) // 8 != len(bitstream):
        raise HuffmanError("зайвий байт після закодованих даних")
    if "1" in bits[pos:]:
        raise HuffmanError("padding має складатися з нулів")
    return bytes(out)


def encoded_bits(data: bytes, lengths: dict[int, int]) -> int:
    """Скільки бітів займуть дані (без padding). Це вага запису для ДП."""
    return sum(lengths[b] for b in data)