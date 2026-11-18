"""Локальний backend: каталог на окремому диску або змонтованому томі.

    <root>/log3-backend.json   маркер; без нього backend вважається недоступним (том не змонтовано)
    <root>/objects/ab/<oid>    зашифровані об'єкти
    <root>/meta/<name>         відкриті службові файли (покажчики версій, конфігурація без секретів)

ObjectDir — спільне сховище об'єктів; ним користується і мережевий вузол (node_server).
"""
from __future__ import annotations

import json
import shutil
from collections.abc import Iterable
from pathlib import Path

from ..errors import BackendError
from ..fsutil import DURABILITY, read_blocks, sha256_file, write_durable
from .base import Ack, check_meta_name, check_oid

MARKER = "log3-backend.json"


class ObjectDir:
    def __init__(self, root: Path):
        self.root = Path(root)

    def create(self) -> None:
        for sub in ("objects", "meta", "tmp"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)
        if not (self.root / MARKER).exists():
            write_durable(self.root / MARKER, [json.dumps({"format": "log3-backend", "format_version": 1}).encode()])

    def ensure(self) -> None:
        if not (self.root / MARKER).is_file():
            raise BackendError(f"каталог {self.root.name!r} недоступний або не ініціалізований як backend")

    def path(self, oid: str) -> Path:
        return self.root / "objects" / check_oid(oid)[:2] / oid

    def store(self, oid: str, chunks: Iterable[bytes], length: int, sha256: str) -> bool:
        """True, якщо такий самий об'єкт уже лежав (нічого не записано). Пошкоджений замінюється."""
        self.ensure()
        dest = self.path(oid)
        if dest.is_file() and sha256_file(dest) == (length, sha256):
            for _ in chunks:                                   # дочитати тіло запиту
                pass
            return True
        try:
            write_durable(dest, chunks, tmp_dir=self.root / "tmp", expect_len=length, expect_sha256=sha256)
        except OSError as e:
            raise BackendError(f"запис не вдався: {e.strerror or e}") from None
        return False

    def stat(self, oid: str, *, rehash: bool = True) -> tuple[int, str] | None:
        self.ensure()
        p = self.path(oid)
        if not p.is_file():
            return None
        return sha256_file(p) if rehash else (p.stat().st_size, "")

    def delete(self, oid: str) -> None:
        self.ensure()
        self.path(oid).unlink(missing_ok=True)

    def put_meta(self, name: str, data: bytes) -> None:
        self.ensure()
        write_durable(self.root / "meta" / check_meta_name(name), [data], tmp_dir=self.root / "tmp")

    def get_meta(self, name: str) -> bytes | None:
        self.ensure()
        p = self.root / "meta" / check_meta_name(name)
        return p.read_bytes() if p.is_file() else None

    def delete_meta(self, name: str) -> None:
        self.ensure()
        (self.root / "meta" / check_meta_name(name)).unlink(missing_ok=True)

    def list_meta(self) -> list[str]:
        self.ensure()
        return sorted(p.name for p in (self.root / "meta").iterdir() if p.is_file())


class LocalDirBackend:
    durability = DURABILITY

    def __init__(self, name: str, path: str | Path):
        self.name, self.dir = name, ObjectDir(Path(path))

    def put(self, oid: str, src: Path, length: int, sha256: str, op_id: str) -> Ack:
        duplicate = self.dir.store(oid, read_blocks(src), length, sha256)
        stored = self.dir.stat(oid)                            # читаємо те, що реально лягло на диск
        if stored is None:
            raise BackendError("об'єкт зник одразу після запису")
        return Ack(oid, stored[0], stored[1], self.durability, duplicate)

    def stat(self, oid: str, *, rehash: bool = True) -> Ack | None:
        s = self.dir.stat(oid, rehash=rehash)
        return None if s is None else Ack(oid, s[0], s[1], self.durability)

    def get(self, oid: str, dest: Path) -> None:
        self.dir.ensure()
        try:
            shutil.copyfile(self.dir.path(oid), dest)
        except FileNotFoundError:
            raise BackendError(f"об'єкта {oid[:12]} нема на backend {self.name}") from None

    def delete(self, oid: str) -> None:
        self.dir.delete(oid)

    def put_meta(self, name: str, data: bytes) -> None:
        self.dir.put_meta(name, data)

    def get_meta(self, name: str) -> bytes | None:
        return self.dir.get_meta(name)

    def delete_meta(self, name: str) -> None:
        self.dir.delete_meta(name)

    def list_meta(self) -> list[str]:
        return self.dir.list_meta()
