import dataclasses

import pytest

from log3.records import (RECORD_SIZE, Record, load_records_csv, pack_record, pack_records,
                          unpack_record, unpack_records)

R = Record(timestamp_ms=0x01020304, source_id=0x0506, event_code=0x0708, metric_value=-2,
           category=15, severity=7, critical=1, record_flags=3)


def test_layout_big_endian():
    b = pack_record(R)
    assert len(b) == RECORD_SIZE == 16
    assert b == bytes.fromhex("01020304 0506 0708 fffffffe 0f 07 01 03".replace(" ", ""))
    assert unpack_record(b) == R


@pytest.mark.parametrize("field,value", [
    ("source_id", 0), ("source_id", 4096), ("category", 16), ("severity", 8),
    ("critical", 2), ("record_flags", 4), ("timestamp_ms", -1), ("metric_value", 2**31),
])
def test_out_of_range_rejected(field, value):
    with pytest.raises(ValueError):
        pack_record(dataclasses.replace(R, **{field: value}))


def test_unpack_rejects_bad_source_and_length():
    bad = bytearray(pack_record(R))
    bad[4:6] = b"\x00\x00"           # source_id = 0
    with pytest.raises(ValueError):
        unpack_record(bytes(bad))
    with pytest.raises(ValueError):
        unpack_record(b"\x00" * 15)
    with pytest.raises(ValueError):
        unpack_records(b"\x00" * 17)


def test_many_round_trip():
    rs = [dataclasses.replace(R, timestamp_ms=i) for i in range(5)]
    assert unpack_records(pack_records(rs)) == rs
    assert unpack_records(b"") == []


def test_load_csv(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("timestamp_ms,source_id,event_code,metric_value,category,severity,critical,"
                 "record_flags,label,extra\n5,1,2,-3,4,5,1,0,2,ignored\n", encoding="utf-8")
    rs, labels = load_records_csv(p)
    assert rs == [Record(5, 1, 2, -3, 4, 5, 1, 0)] and labels == [2]
