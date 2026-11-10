import random

import pytest

from log3.huffman import HuffmanError, canonical_codes, code_lengths, decode, encode, encoded_bits

def test_empty_and_single_symbol():
    assert code_lengths(b"") == {}
    assert code_lengths(b"\x01" * 16) == {1: 1}
    assert canonical_codes({1: 1}) == {1: (0, 1)}


def test_canonical_rfc1951_example():
    # RFC 1951 §3.2.2: ABCDEFGH з довжинами (3,3,3,3,3,2,4,4)
    lengths = dict(zip(b"ABCDEFGH", (3, 3, 3, 3, 3, 2, 4, 4)))
    codes = canonical_codes(lengths)
    assert [format(c, f"0{n}b") for c, n in (codes[s] for s in b"ABCDEFGH")] == \
        ["010", "011", "100", "101", "110", "00", "1110", "1111"]


def test_more_frequent_symbol_not_longer():
    lengths = code_lengths(b"a" * 50 + b"b" * 10 + b"c" * 3 + b"d")
    assert lengths[ord("a")] <= lengths[ord("b")] <= lengths[ord("c")]


def test_kraft_sum_is_one():
    rng = random.Random(7)
    for _ in range(50):
        data = bytes(rng.randrange(rng.randint(2, 40)) for _ in range(rng.randint(2, 300)))
        lengths = code_lengths(data)
        if len(lengths) >= 2:
            assert sum(2.0 ** -n for n in lengths.values()) == 1.0


def test_code_lengths_is_deterministic():
    data = bytes(range(10)) * 3
    assert code_lengths(data) == code_lengths(data)


@pytest.mark.parametrize("lengths", [{1: 2}, {1: 1, 2: 2}, {1: 1, 2: 1, 3: 1}, {1: 16, 2: 1}, {300: 1}])
def test_bad_tables(lengths):
    with pytest.raises(HuffmanError):
        canonical_codes(lengths)

def test_c1_04_canonical_from_lengths():
    data = b"abracadabra, canonical huffman" * 3
    lengths = code_lengths(data)
    stream = encode(data, lengths)
    # приймач має лише довжини — і відновлює ті самі коди
    assert decode(stream, dict(lengths), len(data)) == data
    assert canonical_codes(lengths) == canonical_codes(dict(sorted(lengths.items())))


def test_c1_05_degenerate_alphabet():
    data = b"\x01" * 16
    assert code_lengths(data) == {1: 1}
    assert canonical_codes({1: 1}) == {1: (0, 1)}
    assert encode(data, {1: 1}) == b"\x00\x00"
    assert decode(b"\x00\x00", {1: 1}, 16) == data


def test_empty_round_trip():
    assert encode(b"", {}) == b""
    assert decode(b"", {}, 0) == b""


def test_random_round_trip():
    rng = random.Random(1)
    for _ in range(50):
        data = bytes(rng.choice(range(rng.randint(1, 256))) for _ in range(rng.randint(1, 400)))
        lengths = code_lengths(data)
        stream = encode(data, lengths)
        assert len(stream) == (encoded_bits(data, lengths) + 7) // 8
        assert decode(stream, lengths, len(data)) == data


def test_decode_errors():
    lengths = {0: 1, 1: 1}
    with pytest.raises(HuffmanError):
        decode(b"\x00", lengths, 9)          # потік обрізаний
    with pytest.raises(HuffmanError):
        decode(b"\x00\x00", lengths, 8)      # зайвий повний байт
    with pytest.raises(HuffmanError):
        decode(b"\x01", lengths, 7)          # ненульовий padding
    with pytest.raises(HuffmanError):
        encode(b"\x05", lengths)             # символу немає в таблиці