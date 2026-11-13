import json
import random
import time

import pytest
from conftest import PASSWORD, SALT, make_records

from log3 import crypto
from log3.payload import FLAG_NO_COMPRESSION, build_payload
from log3.records import pack_records
from log3.replica import HEADER_SIZE, Log3Error, as_receiver, build_replica, parse_replica, seal_payload


def blob_of(records, **kw):
    return build_replica(records, PASSWORD, replica_group_id=7, batch_seq=42, salt=SALT, **kw)


def code_of(blob, password=PASSWORD):
    with pytest.raises(Log3Error) as e:
        parse_replica(blob, password)
    return e.value.code


def test_c1_01_single_record():
    r = make_records(1)
    for threshold in (None, 100.0):          # автоматичний вибір і примусове стиснення Гаффманом
        rep = parse_replica(blob_of(r, no_compression_threshold=threshold), PASSWORD)
        assert pack_records(rep.records) == pack_records(r)
        assert (rep.header.replica_group_id, rep.header.batch_seq) == (7, 42)


def test_c1_02_empty():
    for compress in (True, False):
        assert parse_replica(blob_of([], compress=compress), PASSWORD).records == []


def test_c1_03_no_compression():
    rs = make_records(48)
    blob = blob_of(rs, compress=False)
    k_enc, _ = crypto.derive_keys(PASSWORD, SALT)
    payload = crypto.aes_ctr(k_enc, blob[27:39], blob[HEADER_SIZE:-32])
    assert payload[2] & FLAG_NO_COMPRESSION
    assert pack_records(parse_replica(blob, PASSWORD).records) == pack_records(rs)


def test_c1_10_tampered():
    blob = blob_of(make_records(10))
    # replica_group_id, batch_seq, salt, nonce, шифротекст, HMAC — усе під захистом HMAC
    for pos in (5, 8, 12, 30, HEADER_SIZE, HEADER_SIZE + 5, len(blob) - 33, len(blob) - 1):
        b = bytearray(blob)
        b[pos] ^= 0x01
        assert code_of(bytes(b)) == "E_HMAC", pos


def test_c1_11_bad_inputs():
    blob = blob_of(make_records(3))
    assert code_of(blob, "wrong password") == "E_HMAC"
    for cut in (0, 1, 42, 74):
        assert code_of(blob[:cut]) == "E_SHORT"
    assert code_of(blob[:-1]) == "E_LENGTH"
    assert code_of(blob + b"\x00") == "E_LENGTH"
    assert code_of(b"LOG4" + blob[4:]) == "E_MAGIC"
    assert code_of(blob[:4] + b"\x02" + blob[5:]) == "E_MAGIC"
    # reserved != 0 у payload із коректним HMAC: відмова лише після розшифрування
    p = bytearray(build_payload(make_records(3)))
    p[3] = 1
    sealed = seal_payload(bytes(p), PASSWORD, replica_group_id=1, batch_seq=1, salt=SALT)
    assert code_of(sealed) == "E_PAYLOAD"


def test_aes_ctr_nist_vector():
    # NIST SP 800-38A F.5.5 CTR-AES256: лічильник f0..ff = nonce(f0..fb) || uint32_BE(0xfcfdfeff)
    key = bytes.fromhex("603deb1015ca71be2b73aef0857d77811f352c073b6108d72d9810a30914dff4")
    pt = bytes.fromhex("6bc1bee22e409f96e93d7e117393172aae2d8a571e03ac9c9eb76fac45af8e51"
                       "30c81c46a35ce411e5fbc1191a0a52eff69f2445df4f9b17ad2b417be66c3710")
    ct = bytes.fromhex("601ec313775789a5b7a7f504bbf3d228f443e3ca4d62b59aca84e990cacaf5c5"
                       "2b0930daa23de94ce87017ba2d84988ddfc9c58db67aada613c2dd08457941a6")
    nonce = bytes.fromhex("f0f1f2f3f4f5f6f7f8f9fafb")
    assert crypto.aes_ctr(key, nonce, pt, first_counter=0xFCFDFEFF) == ct


def test_counter_starts_at_one():
    key, nonce = bytes(32), bytes(12)
    stream = crypto.aes_ctr(key, nonce, bytes(32))
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    ecb = Cipher(algorithms.AES(key), modes.ECB()).encryptor()
    assert stream == ecb.update(nonce + b"\x00\x00\x00\x01" + nonce + b"\x00\x00\x00\x02")


def test_keys_split_and_cached():
    k_enc, k_mac = crypto.derive_keys(PASSWORD, SALT)
    assert len(k_enc) == len(k_mac) == 32 and k_enc != k_mac
    hits = crypto._derive.cache_info().hits
    crypto.derive_keys(PASSWORD, SALT)
    assert crypto._derive.cache_info().hits == hits + 1


def test_nonce_unique():
    rs = make_records(2)
    assert blob_of(rs)[27:39] != blob_of(rs)[27:39]


def test_receiver():
    recv = as_receiver(PASSWORD)
    blob = blob_of(make_records(2))
    assert recv(blob) is None
    assert recv(blob[:-1] + bytes([blob[-1] ^ 1])) == "E_HMAC"


def test_fuzz_only_log3_error():
    rng = random.Random(5)
    good = blob_of(make_records(20))
    for i in range(600):
        if i % 3 == 0:
            b = bytes(rng.randrange(256) for _ in range(rng.randint(0, 200)))
        else:
            b = bytearray(good)
            for _ in range(rng.randint(1, 3)):
                if rng.random() < 0.7 and b:
                    lo = HEADER_SIZE if rng.random() < 0.8 and len(b) > HEADER_SIZE else 0
                    b[rng.randrange(lo, len(b))] ^= 1 << rng.randrange(8)
                elif b:
                    del b[rng.randrange(len(b)):]
            b = bytes(b)
        try:
            parse_replica(b, PASSWORD)
        except Log3Error:
            pass


# --- еталонні пакети викладача (materials/reference/, локально) -----------------------------

def test_c1_12_reference_packets(reference_dir):
    """Кожен ref_* читається, записи й class_table збігаються з описом, а наш передавач із тими
    самими сіллю, nonce і таблицею дає побайтово ту саму репліку (крос-сумісність формату).
    Кожен bad_* відхиляється рівно з кодом з expected_failures.json."""
    from log3.records import Record
    from log3.replica import parse_header, seal_payload

    start = time.perf_counter()
    refs = sorted(reference_dir.glob("ref_*.bin"))
    assert len(refs) == 6
    for path in refs:
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        blob = path.read_bytes()
        assert len(blob) == meta["довжина_репліки"], path.name
        rep = parse_replica(blob, meta["пароль"])
        records = [Record(**r) for r in meta["записи"]]
        assert rep.records == records and len(records) == meta["record_count"], path.name
        assert rep.class_table == meta["class_table"], path.name
        table = {t["символ"]: t["довжина"] for t in meta["таблиця_кодів"] or []}
        raw = bool(meta["flags"] & 0x08)
        payload = build_payload(records, compress=not raw, lengths=table if records else None,
                                class_table=meta["class_table"], no_compression_threshold=None if raw else 1e9)
        h = parse_header(blob)
        again = seal_payload(payload, meta["пароль"], replica_group_id=h.replica_group_id, batch_seq=h.batch_seq,
                             salt=bytes.fromhex(meta["salt_hex"]), nonce=bytes.fromhex(meta["nonce_hex"]))
        assert again == blob, f"{path.name}: наш передавач дає інші байти"
    expected = json.loads((reference_dir / "expected_failures.json").read_text(encoding="utf-8"))
    password = json.loads((reference_dir / "ref_01_single_record.json").read_text(encoding="utf-8"))["пароль"]
    bad = sorted(reference_dir.glob("bad_*.bin"))
    assert len(bad) == len(expected) == 7
    for path in bad:
        assert code_of(path.read_bytes(), password) == expected[path.name]["очікуваний_код"], path.name
    assert time.perf_counter() - start < 60
