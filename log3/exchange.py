"""Побайтовий обмін базовими репліками LOG3 v1 між командами."""
from __future__ import annotations

import argparse
import getpass
import os
import sys
from pathlib import Path

from .records import pack_records, unpack_records
from .replica import Log3Error, build_replica, parse_replica


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    pack = commands.add_parser("pack", help="16-байтові записи → репліка LOG3")
    unpack = commands.add_parser("unpack", help="репліка LOG3 → 16-байтові записи")
    for command in (pack, unpack):
        command.add_argument("source", type=Path, help="вхідний файл")
        command.add_argument("output", type=Path, help="новий вихідний файл")
    pack.add_argument("--no-compression", action="store_true", help="записати сирі записи")
    args = parser.parse_args(argv)

    try:
        password = os.environ.get("LOG3_REPLICA_PASSWORD")
        if password is None and sys.stdin.isatty():
            password = getpass.getpass("Пароль репліки: ")
        if not password:
            raise ValueError("введіть пароль або задайте LOG3_REPLICA_PASSWORD")

        source = args.source.read_bytes()
        if args.command == "pack":
            result = build_replica(unpack_records(source), password, replica_group_id=1,
                                   batch_seq=0, compress=not args.no_compression)
        else:
            result = pack_records(parse_replica(source, password).records)
        with args.output.open("xb") as stream:
            stream.write(result)
    except Log3Error as exc:
        print(exc, file=sys.stderr)
        return 1
    except (EOFError, OSError, ValueError) as exc:
        print(f"E_INPUT: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
