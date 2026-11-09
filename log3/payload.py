"""Навантаження репліки LOG3 до шифрування (FORMAT_SPEC_P3 §3).

    record_count  u16
    flags         u8    bit0 PREDICTION, bit1 MULTI_COPY_META (командні розширення),
                        bit2 CLASS_TABLE, bit3 NO_COMPRESSION, bits4-7 = 0
    reserved      u8    = 0
    table_nsym    u16
    table         2*table_nsym   пари (symbol u8, length u8) за зростанням symbol
    symbol_count  u32   = 16 * record_count
    bitstream_len u32
    bitstream     bitstream_len
    class_table   record_count байтів, лише якщо bit2

Службових байтів 14; таблиця до 512 B. Для малих batch сирі записи можуть бути
коротшими за таблицю + потік — тоді NO_COMPRESSION (див. build_payload).
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

from . import huffman
from .records import RECORD_SIZE, Record, pack_records, unpack_records

FLAG_PREDICTION = 0x01
FLAG_MULTI_COPY_META = 0x02
FLAG_CLASS_TABLE = 0x04
FLAG_NO_COMPRESSION = 0x08
KNOWN_FLAGS = FLAG_CLASS_TABLE | FLAG_NO_COMPRESSION   # розширень bit0/bit1 ми не реалізуємо

_HEAD = struct.Struct(">HBBH")   # record_count, flags, reserved, table_nsym
_MID = struct.Struct(">II")      # symbol_count, bitstream_len
OVERHEAD = _HEAD.size + _MID.size  # 14
MAX_CLASS = 2                    # 0 штатна, 1 аномальна, 2 перехід до інциденту; 3..255 зарезервовано


class PayloadError(ValueError):
    """Некоректне навантаження. code: "E_PAYLOAD" або "E_CODE" (некоректна таблиця кодів)."""

    def __init__(self, msg: str, code: str = "E_PAYLOAD"):
        super().__init__(msg)
        self.code = code


@dataclass(frozen=True)
class ParsedPayload:
    records: list[Record]
    class_table: list[int] | None
    flags: int


def payload_size(n_records: int, lengths: dict[int, int] | None, bits: int | None,
                 *, class_table: bool = False) -> int:
    """Розмір payload у байтах: стиснений (lengths, bits) або сирий (lengths=None)."""
    tail = n_records if class_table else 0
    if lengths is None:
        return OVERHEAD + RECORD_SIZE * n_records + tail
    return OVERHEAD + 2 * len(lengths) + (bits + 7) // 8 + tail


def build_payload(records: list[Record], *, compress: bool = True,
                  no_compression_threshold: float | None = None,
                  class_table: list[int] | None = None,
                  lengths: dict[int, int] | None = None) -> bytes:
    """records -> payload.

    lengths — готова таблиця вікна (FORMAT_SPEC §3.5: одна таблиця на все вікно,
    її ж записують у кожну репліку вікна); None — побудувати з самих records.
    Поріг: якщо стиснений payload > поріг × сирий, пишемо сирі байти з NO_COMPRESSION.
    None = 1.0, тобто обираємо коротший варіант.
    """
    raw = pack_records(records)
    n = len(records)
    if n > 0xFFFF:
        raise ValueError("record_count не вміщається в u16")
    flags = 0
    tail = b""
    if class_table is not None:
        if len(class_table) != n or any(not 0 <= c <= MAX_CLASS for c in class_table):
            raise ValueError("class_table: по одному класу 0..2 на запис")
        flags |= FLAG_CLASS_TABLE
        tail = bytes(class_table)

    if compress and n:
        table = lengths if lengths is not None else huffman.code_lengths(raw)
        stream = huffman.encode(raw, table)
        packed = OVERHEAD + 2 * len(table) + len(stream)
        threshold = 1.0 if no_compression_threshold is None else no_compression_threshold
        if packed <= threshold * (OVERHEAD + len(raw)):
            pairs = b"".join(bytes((s, table[s])) for s in sorted(table))
            return (_HEAD.pack(n, flags, 0, len(table)) + pairs
                    + _MID.pack(len(raw), len(stream)) + stream + tail)

    # Сирі записи. Порожній журнал теж іде сюди: 14 байтів, record_count = 0.
    flags |= FLAG_NO_COMPRESSION
    return _HEAD.pack(n, flags, 0, 0) + _MID.pack(len(raw), len(raw)) + raw + tail


def parse_payload(payload: bytes) -> ParsedPayload:
    """Строгий розбір; будь-яка невідповідність -> PayloadError, інших винятків нема."""
    if len(payload) < OVERHEAD:
        raise PayloadError("payload коротший за службові поля")
    n, flags, reserved, nsym = _HEAD.unpack_from(payload, 0)
    if reserved != 0:
        raise PayloadError("reserved != 0")
    if flags & ~(KNOWN_FLAGS | FLAG_PREDICTION | FLAG_MULTI_COPY_META):
        raise PayloadError(f"зарезервовані біти flags: {flags:#04x}")
    if flags & (FLAG_PREDICTION | FLAG_MULTI_COPY_META):
        raise PayloadError(f"невідоме розширення у flags: {flags:#04x}")

    pos = _HEAD.size
    table_end = pos + 2 * nsym
    if table_end + _MID.size > len(payload):
        raise PayloadError("обрізана таблиця кодів")
    pairs = payload[pos:table_end]
    symbols = pairs[0::2]
    lengths = dict(zip(symbols, pairs[1::2]))
    pos = table_end
    symbol_count, bitstream_len = _MID.unpack_from(payload, pos)
    pos += _MID.size
    if symbol_count != RECORD_SIZE * n:
        raise PayloadError("symbol_count != 16 * record_count")

    tail = n if flags & FLAG_CLASS_TABLE else 0
    if pos + bitstream_len + tail != len(payload):
        raise PayloadError("довжина payload не збігається з bitstream_len і class_table")
    stream = payload[pos:pos + bitstream_len]
    pos += bitstream_len

    if flags & FLAG_NO_COMPRESSION:
        if nsym != 0:
            raise PayloadError("NO_COMPRESSION з непорожньою таблицею кодів")
        if bitstream_len != symbol_count:
            raise PayloadError("NO_COMPRESSION: bitstream_len != symbol_count")
        raw = stream
    else:
        if any(a >= b for a, b in zip(symbols, symbols[1:])):
            raise PayloadError("символи таблиці не за строгим зростанням", "E_CODE")
        try:
            huffman.check_lengths(lengths)
        except huffman.HuffmanError as e:
            raise PayloadError(f"таблиця кодів: {e}", "E_CODE") from None
        try:
            raw = huffman.decode(stream, lengths, symbol_count)
        except huffman.HuffmanError as e:
            raise PayloadError(f"бітовий потік: {e}") from None

    try:
        records = unpack_records(raw)
    except ValueError as e:
        raise PayloadError(f"запис: {e}") from None

    class_table = None
    if tail:
        class_table = list(payload[pos:pos + n])
        if any(c > MAX_CLASS for c in class_table):
            raise PayloadError("class_table: зарезервоване значення класу")
    return ParsedPayload(records, class_table, flags)
