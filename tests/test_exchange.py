import os
import subprocess
import sys

import pytest
from conftest import make_records

from log3.payload import OVERHEAD
from log3.records import pack_records
from log3.replica import HEADER_SIZE, parse_replica

PASSWORD = "public-exchange-test"


def exchange(*args):
    return subprocess.run(
        [sys.executable, "-m", "log3.exchange", *map(str, args)],
        env={**os.environ, "LOG3_REPLICA_PASSWORD": PASSWORD},
        capture_output=True, text=True, check=False, timeout=20,
    )


@pytest.mark.parametrize("data,options", [
    (b"", ()),
    (pack_records(make_records(1, seed=1729)), ("--no-compression",)),
    (pack_records(make_records(73, seed=1729)), ()),
    (b"\x01" * (16 * 65535), ()),
], ids=["empty", "one-raw-record", "new-records", "max-count-single-symbol"])
def test_exchange_round_trip(tmp_path, data, options):
    original = tmp_path / "original.bin"
    replica = tmp_path / "replica.bin"
    restored = tmp_path / "restored.bin"
    original.write_bytes(data)

    result = exchange("pack", original, replica, *options)
    assert result.returncode == 0, result.stderr
    if options:
        body = parse_replica(replica.read_bytes(), PASSWORD)
        assert pack_records(body.records) == data
        assert body.header.ciphertext_len == OVERHEAD + len(data)
    result = exchange("unpack", replica, restored)
    assert result.returncode == 0, result.stderr
    assert restored.read_bytes() == data


def test_exchange_corrupt_replicas_leave_no_output(tmp_path):
    original = tmp_path / "original.bin"
    replica = tmp_path / "replica.bin"
    corrupt = tmp_path / "corrupt.bin"
    restored = tmp_path / "restored.bin"
    original.write_bytes(pack_records(make_records(2, seed=1729)))
    assert exchange("pack", original, replica).returncode == 0
    data = replica.read_bytes()
    changed = data[:HEADER_SIZE] + bytes([data[HEADER_SIZE] ^ 1]) + data[HEADER_SIZE + 1:]
    wrong_length = data[:HEADER_SIZE - 4] + b"\xff" * 4 + data[HEADER_SIZE:]
    for broken, code in ((changed, "E_HMAC"), (data[:-1], "E_LENGTH"), (wrong_length, "E_LENGTH")):
        corrupt.write_bytes(broken)
        result = exchange("unpack", corrupt, restored)
        assert result.returncode == 1 and code in result.stderr, result.stderr
        assert not restored.exists()


def test_exchange_keeps_existing_output_and_rejects_partial_record(tmp_path):
    original = tmp_path / "original.bin"
    replica = tmp_path / "replica.bin"
    original.write_bytes(b"\x01" * 16)
    replica.write_bytes(b"keep")
    result = exchange("pack", original, replica)
    assert result.returncode == 1 and replica.read_bytes() == b"keep"

    original.write_bytes(b"\x01" * 15)
    output = tmp_path / "new.bin"
    result = exchange("pack", original, output)
    assert result.returncode == 1 and "E_INPUT" in result.stderr
    assert not output.exists()
