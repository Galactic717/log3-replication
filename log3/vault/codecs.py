"""Версійований реєстр кодеків стиснення без втрат.

Ідентифікатор кодека записується в заголовок кожного об'єкта, тому декодування залежить лише
від записаного формату. Ідентифікатори ніколи не перевикористовуються; зміна бітового формату
кодека — це новий ідентифікатор. Службовий заголовок об'єкта однаковий для всіх кодеків, тому
вигідність стиснення порівнюється за довжиною payload: якщо він не коротший за вихідні дані,
блок зберігається як Store.
"""
from __future__ import annotations

import struct
from collections.abc import Callable
from dataclasses import dataclass

import zstandard

from .. import huffman
from .errors import Exit, VaultError

STORE, ZSTD, HUFFMAN = 0, 1, 2


@dataclass(frozen=True)
class Codec:
    id: int
    name: str
    version: int
    encode: Callable[[bytes, int], bytes]      # (data, level) -> payload
    decode: Callable[[bytes, int], bytes]      # (payload, plain_len) -> data


def _zstd_dec(payload: bytes, plain_len: int) -> bytes:
    if plain_len == 0:
        return b""
    try:
        return zstandard.ZstdDecompressor().decompress(payload, max_output_size=plain_len)
    except zstandard.ZstdError as e:
        raise VaultError("E_CODEC", f"zstd: {e}", Exit.INTEGRITY) from None


def _huff_enc(data: bytes, _level: int) -> bytes:
    """Таблиця довжин (nsym u16, пари символ/довжина) + бітовий потік канонічного Гаффмана LOG3."""
    lengths = huffman.code_lengths(data)
    table = b"".join(bytes((s, n)) for s, n in sorted(lengths.items()))
    return struct.pack(">H", len(lengths)) + table + huffman.encode(data, lengths)


def _huff_dec(payload: bytes, plain_len: int) -> bytes:
    try:
        (nsym,) = struct.unpack_from(">H", payload)
        table = payload[2:2 + 2 * nsym]
        if len(table) != 2 * nsym:
            raise huffman.HuffmanError("обрізана таблиця довжин")
        lengths = {table[i]: table[i + 1] for i in range(0, len(table), 2)}
        return huffman.decode(payload[2 + 2 * nsym:], lengths, plain_len)
    except (huffman.HuffmanError, struct.error) as e:
        raise VaultError("E_CODEC", f"huffman: {e}", Exit.INTEGRITY) from None


REGISTRY: dict[int, Codec] = {c.id: c for c in (
    Codec(STORE, "store", 1, lambda d, _l: d, lambda p, _n: p),
    Codec(ZSTD, "zstd", 1, lambda d, level: zstandard.ZstdCompressor(level=level).compress(d), _zstd_dec),
    Codec(HUFFMAN, "huffman-log3", 1, _huff_enc, _huff_dec),   # експериментальний: повільний, чистий Python
)}
BY_NAME = {c.name: c for c in REGISTRY.values()}


def get(codec_id: int) -> Codec:
    if codec_id not in REGISTRY:
        raise VaultError("E_CODEC", f"невідомий кодек {codec_id}", Exit.INTEGRITY)
    return REGISTRY[codec_id]


def encode_best(data: bytes, codec_id: int, level: int = 3) -> tuple[int, bytes]:
    """(фактичний кодек, payload). Невигідне або збійне стиснення замінюється Store."""
    if codec_id != STORE and data:
        try:
            payload = get(codec_id).encode(data, level)
        except Exception:                                  # збій кодека не блокує backup
            payload = data
        if len(payload) < len(data):
            return codec_id, payload
    return STORE, data


def decode(codec_id: int, payload: bytes, plain_len: int) -> bytes:
    data = get(codec_id).decode(payload, plain_len)
    if len(data) != plain_len:
        raise VaultError("E_LENGTH", f"після декодування {len(data)} B, очікувалось {plain_len}", Exit.INTEGRITY)
    return data
