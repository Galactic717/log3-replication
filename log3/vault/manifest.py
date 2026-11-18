"""Маніфест версії: версійований опис того, що і в якому порядку відновлювати.

Маніфест — канонічний JSON (UTF-8), стиснений zstd і зашифрований як об'єкт типу MANIFEST.
Чотири незалежні ідентичності:
  content_id  вміст (ключований HMAC від SHA-256 байтів файла): однакові байти -> один запис у contents
  file_id     файл як об'єкт обліку: зберігається між версіями і, де це видно за inode, між перейменуваннями
  path        шлях у межах джерела ("/" як роздільник, Unicode як у файловій системі)
  version_id  знімок цілого джерела в один момент приймання
Час — цілі наносекунди від 1970-01-01 UTC (поля *_ns) плюс RFC 3339 для людей.

    {"format": "log3-file-manifest", "format_version": 1, "version_id", "operation_id", "repo_id",
     "name", "created_at_ns", "created_at", "source": {"root", "kind"}, "chunk_size_bytes",
     "crypto": {"suite", "key_id", "object_format_version"}, "codecs": {"<id>": {"name", "version"}},
     "entries":  [{"path", "type": "file", "file_id", "content_id", "size", "mtime_ns", "mode"},
                  {"path", "type": "dir", "mtime_ns", "mode"}],
     "contents": {"<content_id>": {"size", "sha256", "chunks": ["<object_id>", ...]}},   # порядок блоків
     "objects":  {"<object_id>": {"plain_len", "stored_len", "sha256", "codec"}},        # sha256 зашифрованого об'єкта
     "skipped":  [{"path", "reason"}]}
"""
from __future__ import annotations

import json
import os
import sys
import unicodedata
from datetime import UTC, datetime

from .errors import Exit, VaultError

FORMAT = "log3-file-manifest"
FORMAT_VERSION = 1
_WIN_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}


def rfc3339(ns: int | None) -> str | None:
    if ns is None:
        return None
    return datetime.fromtimestamp(ns // 10 ** 9, UTC).strftime("%Y-%m-%dT%H:%M:%S") + f".{ns % 10 ** 9:09d}Z"


def dump(manifest: dict) -> bytes:
    return json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _bad(msg: str) -> VaultError:
    return VaultError("E_MANIFEST", msg, Exit.INTEGRITY)


def safe_parts(path: str, *, windows: bool | None = None) -> list[str]:
    """Компоненти відносного шляху або VaultError. Не пропускає вихід за межі каталогу відновлення."""
    windows = os.name == "nt" if windows is None else windows
    if not isinstance(path, str) or not path or "\x00" in path or path.startswith("/"):
        raise _bad(f"небезпечний шлях {path!r}")
    parts = path.split("/")
    for p in parts:
        if p in ("", ".", "..") or "\\" in p:
            raise _bad(f"небезпечний шлях {path!r}")
        if windows and (":" in p or p[-1] in " ." or p.split(".")[0].upper() in _WIN_RESERVED
                        or any(c in p for c in '<>"|?*') or any(ord(c) < 32 for c in p)):
            raise _bad(f"шлях {path!r} недопустимий у Windows")
    return parts


def collision_key(path: str, *, insensitive: bool) -> str:
    return unicodedata.normalize("NFC", path).casefold() if insensitive else path


def load(data: bytes, *, insensitive: bool | None = None) -> dict:
    """Розбирає й повністю перевіряє маніфест до того, як restore щось запише."""
    insensitive = (os.name == "nt" or sys.platform == "darwin") if insensitive is None else insensitive
    try:
        m = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as e:
        raise _bad(f"не JSON: {e}") from None
    if not isinstance(m, dict) or m.get("format") != FORMAT:
        raise _bad("невідомий формат маніфесту")
    if m.get("format_version") != FORMAT_VERSION:
        raise _bad(f"версія маніфесту {m.get('format_version')} не підтримується (підтримується {FORMAT_VERSION})")
    try:
        seen: dict[str, str] = {}
        files: set[str] = set()
        for e in m["entries"]:
            safe_parts(e["path"])
            key = collision_key(e["path"], insensitive=insensitive)
            if key in seen:
                raise _bad(f"колізія імен: {e['path']!r} і {seen[key]!r}")
            seen[key] = e["path"]
            if e["type"] == "file":
                files.add(key)
                c = m["contents"][e["content_id"]]
                if c["size"] != e["size"] or sum(m["objects"][o]["plain_len"] for o in c["chunks"]) != c["size"]:
                    raise _bad(f"довжини блоків не сходяться для {e['path']!r}")
            elif e["type"] != "dir":
                raise _bad(f"невідомий тип запису {e['type']!r}")
        for key, path in seen.items():                      # файл не може бути предком іншого запису
            parent = key.rpartition("/")[0]
            while parent:
                if parent in files:
                    raise _bad(f"{path!r} лежить усередині файла")
                parent = parent.rpartition("/")[0]
    except (KeyError, TypeError) as e:
        raise _bad(f"бракує поля або невірний тип: {e}") from None
    return m
