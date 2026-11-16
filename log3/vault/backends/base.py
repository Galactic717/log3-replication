"""Контракт backend і значення його підтвердження.

Ack повертається лише після того, як backend виконав запис на своєму документованому рівні
стійкості (поле durability). Ack називає сам об'єкт: ідентифікатор, довжину і SHA-256 байтів,
які реально лежать у сховищі. Тракт приймає Ack тільки якщо всі три збігаються з поточним
об'єктом (check_ack); інакше це помилка, а не «майже успіх».

put ідемпотентний: повтор із тим самим oid і тим самим вмістом нічого не дописує і не
списує квоту вдруге (duplicate=True). op_id — стабільний ідентифікатор завдання, однаковий
у всіх повторах; backend використовує його для обліку й журналу.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ..errors import BackendError

_OID = re.compile(r"[0-9a-f]{64}")
_META = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,120}")


@dataclass(frozen=True)
class Ack:
    oid: str
    length: int
    sha256: str
    durability: str
    duplicate: bool = False


def check_oid(oid: str) -> str:
    if not _OID.fullmatch(oid):
        raise BackendError("некоректний ідентифікатор об'єкта")
    return oid


def check_meta_name(name: str) -> str:
    if not _META.fullmatch(name):
        raise BackendError("некоректна назва службового файла")
    return name


def check_ack(ack: Ack, oid: str, length: int, sha256: str) -> Ack:
    if (ack.oid, ack.length, ack.sha256) != (oid, length, sha256):
        raise BackendError(f"підтвердження не відповідає об'єкту {oid[:12]}: "
                           f"отримано {ack.oid[:12]}/{ack.length} B, очікувалось {length} B")
    return ack


class Backend(Protocol):
    name: str
    durability: str

    def put(self, oid: str, src: Path, length: int, sha256: str, op_id: str) -> Ack: ...

    def stat(self, oid: str, *, rehash: bool = True) -> Ack | None: ...   # None: об'єкта нема

    def get(self, oid: str, dest: Path) -> None: ...                       # потоково у файл dest

    def delete(self, oid: str) -> None: ...

    def put_meta(self, name: str, data: bytes) -> None: ...                # малі відкриті службові файли

    def get_meta(self, name: str) -> bytes | None: ...

    def delete_meta(self, name: str) -> None: ...

    def list_meta(self) -> list[str]: ...
