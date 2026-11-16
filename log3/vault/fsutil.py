"""Стійкий запис файлів: тимчасовий файл -> fsync -> атомарне перейменування -> fsync каталогу.

Гарантія залежить від ОС і носія: fsync обіцяє стійкість лише тоді, коли диск і контролер
чесно виконують скидання кешу. У Windows каталог окремо не синхронізується (os.fsync для
каталогу недоступний): стійкість перейменування там забезпечує журнал NTFS.
"""
from __future__ import annotations

import hashlib
import os
import shutil
from collections.abc import Iterable
from pathlib import Path

from .errors import BackendError

IO_BLOCK = 1 << 20
DURABILITY = ("fsync(file)+rename+fsync(dir)" if os.name == "posix"
              else "FlushFileBuffers(file)+atomic rename; каталог синхронізує журнал NTFS")


def fsync_dir(path: Path) -> None:
    if os.name != "posix":
        return
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_durable(dest: Path, chunks: Iterable[bytes], *, tmp_dir: Path | None = None,
                  expect_len: int | None = None, expect_sha256: str | None = None,
                  sync_dir: bool = True) -> tuple[int, str]:
    """Записує потік у dest і повертає (довжина, sha256).

    Невідповідність очікуваній довжині чи SHA-256 -> BackendError, dest не чіпається."""
    tmp_dir = tmp_dir or dest.parent
    tmp_dir.mkdir(parents=True, exist_ok=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = tmp_dir / f".tmp-{os.getpid()}-{os.urandom(6).hex()}"
    h, n = hashlib.sha256(), 0
    try:
        with open(tmp, "wb") as f:
            for chunk in chunks:
                f.write(chunk)
                h.update(chunk)
                n += len(chunk)
            f.flush()
            os.fsync(f.fileno())
        if (expect_len is not None and n != expect_len) or (expect_sha256 and h.hexdigest() != expect_sha256):
            raise BackendError(f"отримано {n} B з іншою довжиною або SHA-256, ніж заявлено")
        os.replace(tmp, dest)
    finally:
        if tmp.exists():
            tmp.unlink()
    if sync_dir:
        fsync_dir(dest.parent)
    return n, h.hexdigest()


def read_blocks(path: Path) -> Iterable[bytes]:
    with open(path, "rb") as f:
        while block := f.read(IO_BLOCK):
            yield block


def sha256_file(path: Path) -> tuple[int, str]:
    h, n = hashlib.sha256(), 0
    for block in read_blocks(path):
        h.update(block)
        n += len(block)
    return n, h.hexdigest()


def free_bytes(path: Path) -> int:
    return shutil.disk_usage(path).free
