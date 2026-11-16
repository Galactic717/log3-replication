"""Відновлення версії через staging.

Кожен об'єкт перевіряється чотири рази, перш ніж його байти потраплять у файл:
  1. SHA-256 і довжина зашифрованого об'єкта (як у маніфесті);
  2. автентифікація AES-GCM з AAD, що прив'язує об'єкт до його ідентифікатора, типу й кодека;
  3. довжина після декодування;
  4. ключований ідентифікатор відкритих байтів дорівнює object_id.
Порядок блоків задає маніфест; готовий файл звіряється за довжиною і SHA-256 вмісту.

Файли спершу пишуться в <dest>/.log3-restore-*/ і лише після повної перевірки переносяться
на місце (os.replace, атомарно для кожного файла; набір файлів загалом не атомарний).
Захист: шляхи з маніфесту не можуть вийти за dest; жоден наявний компонент шляху не може бути
symlink або junction; колізії імен відхиляються до запису; наявні файли не перезаписуються без
overwrite. Пам'ять: один блок за раз, незалежно від розміру файла.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import stat
import time
from pathlib import Path

import zstandard

from . import codecs, manifest
from .backends.base import Backend
from .cryptobox import KIND_CHUNK, KIND_MANIFEST
from .errors import BackendError, Exit, VaultError
from .fsutil import fsync_dir
from .repo import Repo
from .states import RESTORABLE


def _fetch(repo: Repo, oid: str, length: int, sha256: str, backends: list[Backend] | None) -> bytes:
    """Зашифрований об'єкт із першого джерела, де він цілий: spool, потім backend-и."""
    if backends is None:
        p = repo.spool_path(oid)
        if p.is_file():
            blob = p.read_bytes()
            if len(blob) == length and hashlib.sha256(blob).hexdigest() == sha256:
                return blob
        good = [b for (b,) in repo.cat.q("SELECT backend FROM replicas WHERE oid=? ORDER BY state='CONFIRMED' DESC",
                                         (oid,))]
        names = [n for n in good if n in repo.backend_names()] + [n for n in repo.backend_names() if n not in good]
        backends = [repo.backend(n) for n in names]
    tmp = repo.spool / "tmp" / f"restore-{os.getpid()}-{oid[:16]}"
    for backend in backends:
        try:
            backend.get(oid, tmp)
            blob = tmp.read_bytes()
        except (BackendError, OSError):
            continue
        finally:
            tmp.unlink(missing_ok=True)
        if len(blob) == length and hashlib.sha256(blob).hexdigest() == sha256:
            return blob
    raise VaultError("E_UNAVAILABLE", f"нема жодної цілої копії об'єкта {oid[:12]}", Exit.INTEGRITY)


def open_object(repo: Repo, oid: str, kind: int, length: int, sha256: str,
                backends: list[Backend] | None = None, plain_len: int | None = None) -> bytes:
    codec_id, got_len, payload = repo.box.open(_fetch(repo, oid, length, sha256, backends), kind, oid)
    if kind == KIND_MANIFEST:
        try:
            data = zstandard.ZstdDecompressor().decompress(payload, max_output_size=got_len)
        except zstandard.ZstdError as e:
            raise VaultError("E_CODEC", f"маніфест: {e}", Exit.INTEGRITY) from None
    else:
        data = codecs.decode(codec_id, payload, got_len)
    if len(data) != got_len or (plain_len is not None and plain_len != got_len):
        raise VaultError("E_LENGTH", f"довжина об'єкта {oid[:12]} не відповідає маніфесту", Exit.INTEGRITY)
    if repo.box.object_id(kind, data) != oid:
        raise VaultError("E_ID", f"вміст об'єкта {oid[:12]} не відповідає його ідентифікатору", Exit.INTEGRITY)
    return data


def load_manifest(repo: Repo, version) -> dict:
    o = repo.cat.one("SELECT stored_len, sha256 FROM objects WHERE oid=?", (version["manifest_oid"],))
    if o is None:
        raise VaultError("E_UNAVAILABLE", "маніфесту версії нема в каталозі", Exit.INTEGRITY)
    m = manifest.load(open_object(repo, version["manifest_oid"], KIND_MANIFEST, o["stored_len"], o["sha256"]))
    if m["version_id"] != version["version_id"]:
        raise VaultError("E_MANIFEST", "маніфест належить іншій версії", Exit.INTEGRITY)
    return m


def _is_link(p: Path) -> bool:
    try:
        st = os.lstat(p)
    except FileNotFoundError:
        return False
    return stat.S_ISLNK(st.st_mode) or bool(getattr(st, "st_file_attributes", 0) & 0x400)   # REPARSE_POINT


def _check_target(dest: Path, parts: list[str], is_dir: bool, overwrite: bool) -> Path:
    cur = dest
    for i, part in enumerate(parts):
        cur = cur / part
        if _is_link(cur):
            raise VaultError("E_SYMLINK", f"{cur} — символічне посилання або junction; "
                             "відновлення крізь нього заборонене")
        last = i == len(parts) - 1
        if not os.path.lexists(cur):
            continue
        if not last and not cur.is_dir():
            raise VaultError("E_COLLISION", f"{cur} існує і не є каталогом")
        if last and is_dir != cur.is_dir():
            raise VaultError("E_COLLISION", f"{cur} існує з іншим типом (файл/каталог)")
        if last and not is_dir and not overwrite:
            raise VaultError("E_EXISTS", f"{cur} вже існує; для перезапису вкажіть --overwrite")
    return cur


def restore(repo: Repo, ref: str, dest: Path, *, overwrite: bool = False, only: str | None = None) -> dict:
    t0 = time.monotonic()
    v = repo.cat.find_version(ref)
    if v["state"] not in RESTORABLE:
        raise VaultError("E_STATE", f"версія {v['version_id'][:12]} у стані {v['state']} і не відновлюється")
    m = load_manifest(repo, v)
    entries = [e for e in m["entries"] if only is None or e["path"] == only or e["path"].startswith(only + "/")]
    if only is not None and not entries:
        raise VaultError("E_NOT_FOUND", f"у версії нема шляху {only!r}", Exit.NOT_FOUND)
    dest.mkdir(parents=True, exist_ok=True)
    dest = dest.resolve()
    plan = [(e, manifest.safe_parts(e["path"])) for e in entries]
    for e, parts in plan:                                   # усі перевірки — до першого запису
        _check_target(dest, parts, e["type"] == "dir", overwrite)
    staging = dest / f".log3-restore-{v['version_id'][:8]}-{os.urandom(4).hex()}"
    staging.mkdir()
    out = {"version_id": v["version_id"], "files": 0, "dirs": 0, "bytes": 0}
    try:
        staged = []
        for i, (e, parts) in enumerate(plan):
            if e["type"] != "file":
                continue
            content, part = m["contents"][e["content_id"]], staging / f"{i}.part"
            h, size = hashlib.sha256(), 0
            with open(part, "wb") as f:
                for oid in content["chunks"]:               # порядок блоків — із маніфесту
                    o = m["objects"][oid]
                    data = open_object(repo, oid, KIND_CHUNK, o["stored_len"], o["sha256"], plain_len=o["plain_len"])
                    f.write(data)
                    h.update(data)
                    size += len(data)
                f.flush()
                os.fsync(f.fileno())
            if size != e["size"] or h.hexdigest() != content["sha256"]:
                raise VaultError("E_CONTENT", f"{e['path']}: довжина або SHA-256 вмісту не збігаються", Exit.INTEGRITY)
            staged.append((e, parts, part))
            out["bytes"] += size
        dirs = []
        for e, parts in plan:
            if e["type"] == "dir":
                target = _check_target(dest, parts, True, overwrite)
                target.mkdir(parents=True, exist_ok=True)
                dirs.append((target, e))
                out["dirs"] += 1
        for e, parts, part in staged:
            target = _check_target(dest, parts, False, overwrite)   # повторно: стан міг змінитись
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(part, target)
            if os.name == "posix":
                os.chmod(target, e["mode"] & 0o777)
            os.utime(target, ns=(e["mtime_ns"], e["mtime_ns"]))
            fsync_dir(target.parent)
            out["files"] += 1
        for target, e in reversed(dirs):
            os.utime(target, ns=(e["mtime_ns"], e["mtime_ns"]))
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    out["duration_seconds"] = round(time.monotonic() - t0, 3)
    repo.cat.event("restore_seconds", out["duration_seconds"], v["version_id"])
    return out
