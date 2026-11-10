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