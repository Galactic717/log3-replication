"""Стійкий запис файлів: тимчасовий файл -> fsync -> атомарне перейменування -> fsync каталогу.

Гарантія залежить від ОС і носія: fsync обіцяє стійкість лише тоді, коли диск і контролер
чесно виконують скидання кешу. У Windows каталог окремо не синхронізується (os.fsync для
каталогу недоступний): стійкість перейменування там забезпечує журнал NTFS.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
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


_RESTRICT_PS = (
    "$s=[Security.Principal.WindowsIdentity]::GetCurrent().User.Value;"
    "$acl=New-Object Security.AccessControl.DirectorySecurity; $acl.SetAccessRuleProtection($true,$false);"
    "foreach($x in @($s,'S-1-5-18','S-1-5-32-544')){$acl.AddAccessRule((New-Object "
    "Security.AccessControl.FileSystemAccessRule((New-Object Security.Principal.SecurityIdentifier $x),"
    "'FullControl','ContainerInherit,ObjectInherit','None','Allow')))};"
    "(Get-Item -LiteralPath $env:L3P).SetAccessControl($acl)")


def restrict_dir(path: Path) -> str:
    """Доступ до каталогу лише власнику: POSIX 0700; Windows — новий DACL без успадкування й без
    сторонніх явних записів: лише власник, SYSTEM і Адміністратори (SID, тож працює в локалізованих
    Windows). Повертає опис результату."""
    if os.name == "posix":
        os.chmod(path, 0o700)
        return "posix-0700"
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", _RESTRICT_PS],
                           capture_output=True, text=True, env={**os.environ, "L3P": str(path)}, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return "windows-acl-unavailable"
    return "windows-acl" if r.returncode == 0 and not r.stderr.strip() else "windows-acl-unsupported"


def access_report(path: Path) -> dict:
    """Хто ще, крім власника, має доступ до каталогу (для doctor)."""
    if os.name == "posix":
        mode = os.stat(path).st_mode & 0o777
        return {"mode": oct(mode), "others_have_access": bool(mode & 0o077)}
    # правила одразу як SID: перекладати імена не треба (у CI-раннерах і доменах переклад падає)
    ps = ("try{$a=[IO.Directory]::GetAccessControl($env:L3P);"
          "$u=[Security.Principal.WindowsIdentity]::GetCurrent().User.Value;"
          "$r=@($a.GetAccessRules($true,$true,[Security.Principal.SecurityIdentifier]));"
          "@{user=$u; inherited=[bool]($r|?{$_.IsInherited}); sids=@($r|%{[string]$_.IdentityReference.Value})}"
          "|ConvertTo-Json -Compress}catch{@{error=[string]$_.Exception.Message}|ConvertTo-Json -Compress}")
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps], capture_output=True,
                           text=True, env={**os.environ, "L3P": str(path)}, timeout=60)
        d = json.loads(r.stdout)
    except (OSError, ValueError, subprocess.TimeoutExpired):
        d = {"error": "powershell недоступний або повернув не JSON"}
    if "error" in d or not d.get("sids") or None in d["sids"]:     # не змогли прочитати — «невідомо», не «так/ні»
        return {"acl": "unknown", "error": str(d.get("error", "порожній ACL"))[:200], "others_have_access": None}
    others = sorted(set(d["sids"]) - {d["user"], "S-1-5-18", "S-1-5-32-544"})
    return {"acl_inherited": d["inherited"], "other_sids": others, "others_have_access": bool(others) or d["inherited"]}
