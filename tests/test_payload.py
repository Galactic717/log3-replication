import random
import struct

import pytest
from conftest import make_records

from log3 import huffman
from log3.payload import FLAG_CLASS_TABLE, FLAG_NO_COMPRESSION, OVERHEAD, PayloadError, build_payload, parse_payload
from log3.records import Record, pack_records

DEGENERATE = Record(0x01010101, 0x0101, 0x0101, 0x01010101, 1, 1, 1, 1)  # усі 16 байтів = 0x01


def raw_payload(records, *, flags=0, reserved=0, table=None, symbol_count=None,
                stream=None, tail=b"", extra=b""):
    """Payload «руками» для negative-тестів (обходить перевірки build_payload)."""
    raw = pack_records(records)
    if table is None:
        table = {} if flags & FLAG_NO_COMPRESSION else dict(sorted(huffman.code_lengths(raw).items()))
    if stream is None:
        stream = raw if flags & FLAG_NO_COMPRESSION else huffman.encode(raw, table)
    pairs = b"".join(bytes((s, table[s])) for s in table)
    sc = len(raw) if symbol_count is None else symbol_count
    return (struct.pack(">HBBH", len(records), flags, reserved, len(table)) + pairs
            + struct.pack(">II", sc, len(stream)) + stream + extra + tail)


def test_compressed_round_trip():
    rs = make_records(48)
    p = build_payload(rs)
    parsed = parse_payload(p)
    assert parsed.records == rs and parsed.class_table is None
    assert not parsed.flags & FLAG_NO_COMPRESSION
    assert len(p) < OVERHEAD + 16 * 48


def test_small_batch_falls_back_to_raw():
    # таблиця одного запису (до 32 B) + потік довші за 16 сирих байтів
    p = build_payload(make_records(1))
    assert parse_payload(p).flags & FLAG_NO_COMPRESSION
    assert len(p) == OVERHEAD + 16


def test_threshold():
    rs = make_records(48)
    assert parse_payload(build_payload(rs, no_compression_threshold=0.0)).flags & FLAG_NO_COMPRESSION
    one = make_records(1)
    assert not parse_payload(build_payload(one, no_compression_threshold=10.0)).flags & FLAG_NO_COMPRESSION


def test_window_table_is_written():
    window = make_records(48)
    lengths = huffman.code_lengths(pack_records(window))
    batch = window[::3]
    p = build_payload(batch, lengths=lengths, no_compression_threshold=10.0)
    assert struct.unpack_from(">H", p, 4)[0] == len(lengths)
    assert parse_payload(p).records == batch


def test_degenerate_record():
    p = build_payload([DEGENERATE], no_compression_threshold=10.0)
    assert p[6:8] == b"\x01\x01"                 # таблиця: символ 0x01, довжина 1
    assert parse_payload(p).records == [DEGENERATE]


def test_class_table():
    rs = make_records(5)
    parsed = parse_payload(build_payload(rs, class_table=[0, 1, 2, 1, 0]))
    assert parsed.class_table == [0, 1, 2, 1, 0] and parsed.flags & FLAG_CLASS_TABLE


def test_empty():
    for compress in (True, False):
        parsed = parse_payload(build_payload([], compress=compress))
        assert parsed.records == []
    assert parse_payload(raw_payload([])).records == []    # стиснений варіант із nsym = 0


RS = make_records(4, seed=7)


@pytest.mark.parametrize("payload,code", [
    (raw_payload(RS, reserved=1), "E_PAYLOAD"),
    (raw_payload(RS, flags=0x01), "E_PAYLOAD"),                       # PREDICTION
    (raw_payload(RS, flags=0x02), "E_PAYLOAD"),                       # MULTI_COPY_META
    (raw_payload(RS, flags=0x10), "E_PAYLOAD"),                       # зарезервований біт
    (raw_payload(RS, symbol_count=63), "E_PAYLOAD"),
    (raw_payload(RS, extra=b"\x00"), "E_PAYLOAD"),                    # зайвий повний байт (bad_07)
    (raw_payload(RS)[:-1], "E_PAYLOAD"),                              # обрізаний
    (raw_payload(RS)[:10], "E_PAYLOAD"),
    (raw_payload(RS, flags=FLAG_NO_COMPRESSION, table={1: 1}), "E_PAYLOAD"),
    (raw_payload(RS, flags=FLAG_CLASS_TABLE, tail=b"\x00\x01\x02\x03"), "E_PAYLOAD"),
    (raw_payload([DEGENERATE], table={1: 2}, stream=b"\x00" * 4), "E_CODE"),
    (raw_payload([DEGENERATE], table={1: 1, 2: 2}, stream=b"\x00\x00"), "E_CODE"),  # Kraft < 1 (bad_06)
    (raw_payload([DEGENERATE], table={2: 1, 1: 1}, stream=b"\xff\xff"), "E_CODE"),  # не за зростанням
    (raw_payload([DEGENERATE], table={1: 1}, stream=b"\x00\x01"), "E_PAYLOAD"),     # ненульовий padding
])
def test_negative(payload, code):
    with pytest.raises(PayloadError) as e:
        parse_payload(payload)
    assert e.value.code == code


def test_bad_record_in_raw_stream():
    bad = bytearray(raw_payload(RS, flags=FLAG_NO_COMPRESSION))
    bad[OVERHEAD + 4:OVERHEAD + 6] = b"\x00\x00"   # source_id = 0 у першому записі
    with pytest.raises(PayloadError):
        parse_payload(bytes(bad))


def test_fuzz_only_payload_error():
    rng = random.Random(3)
    seeds = [build_payload(make_records(n, seed=n)) for n in (0, 1, 3, 48)]
    seeds.append(build_payload(make_records(3), class_table=[0, 1, 2], no_compression_threshold=9))
    for _ in range(3000):
        b = bytearray(rng.choice(seeds))
        for _ in range(rng.randint(1, 4)):
            op = rng.random()
            if op < 0.6 and b:
                b[rng.randrange(len(b))] = rng.randrange(256)
            elif op < 0.8:
                del b[rng.randrange(len(b) + 1):]
            else:
                b += bytes(rng.randrange(256) for _ in range(rng.randint(1, 3)))
        try:
            parse_payload(bytes(b))
        except PayloadError:
            pass
