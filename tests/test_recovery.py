from conftest import PASSWORD, SALT, make_records
from log3.records import pack_record
from log3.recovery import recover
from log3.replica import build_replica


def rep(records, seq):
    return build_replica(records, PASSWORD, replica_group_id=0, batch_seq=seq, salt=SALT)


def test_dedup_sort_and_reject():
    rs = make_records(30)
    a, b, c = rep(rs[:20], 1), rep(rs[10:], 2), rep(rs[5:15], 3)       # перекриваються
    broken = bytearray(rep(rs[:3], 4))
    broken[-1] ^= 1
    out, rejected = recover([c, b, bytes(broken), a, b"junk"], PASSWORD)
    assert [pack_record(r) for r in out] == [pack_record(r) for r in rs]  # кожна подія рівно раз
    assert rejected == 2


def test_empty_and_wrong_password():
    assert recover([], PASSWORD) == ([], 0)
    assert recover([rep(make_records(2), 1)], "other") == ([], 1)
