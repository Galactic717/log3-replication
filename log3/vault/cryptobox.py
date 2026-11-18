"""Криптографічний шар файлового режиму: інтерфейс ключів і автентифіковане шифрування об'єктів.

Файловий тракт знає лише KeyProvider і CryptoBox. Щоб посилити керування ключами (KMS, ротація,
обгортання ключів), досить дати іншу реалізацію KeyProvider: формат об'єкта вже містить key_id.

Об'єкт у сховищі (блок або маніфест), усі числа big-endian:
    magic "L3FB" | format_version u8 | kind u8 | codec u8 | key_id 8 B | nonce 12 B | plain_len u64
    | AES-256-GCM(payload) || tag 16 B
AAD = заголовок (35 B) || object_id (32 B): об'єкт не можна підмінити іншим, змінити його тип,
кодек чи довжину. object_id = HMAC-SHA256(K_id, kind || відкриті байти): ключований, тому
сторона з доступом до вузла не може перевірити здогадку про вміст без ключа.

Примітиви — з бібліотеки cryptography (AES-GCM, scrypt, HKDF). Обмеження: nonce випадковий,
тому один ключ даних розрахований на << 2^32 об'єктів.
"""
from __future__ import annotations

import getpass
import hashlib
import hmac
import os
import stat
import struct
import sys
from pathlib import Path
from typing import Protocol

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .errors import Exit, VaultError
from .keys import RepoKeys, new_kdf_params  # noqa: F401  (new_kdf_params — сумісний імпорт)

MAGIC = b"L3FB"
OBJECT_FORMAT_VERSION = 1
KIND_CHUNK, KIND_MANIFEST = 1, 2
HEADER = struct.Struct(">4sBBB8s12sQ")
TAG_SIZE = 16
OVERHEAD = HEADER.size + TAG_SIZE
SUITE = "AES-256-GCM"
PASSPHRASE_ENV = "LOG3_VAULT_PASSPHRASE"


class KeyProvider(Protocol):
    """Джерело ключів. Реалізація сама вирішує, звідки їх брати; у журнали ключі не потрапляють."""

    key_id: str                                   # 16 hex: ключ, яким шифруються нові об'єкти

    def data_key(self, key_id: str) -> bytes: ...  # 32 B для AES-256-GCM; невідомий key_id -> VaultError

    def id_key(self) -> bytes: ...                 # 32 B для ключованих ідентифікаторів


def read_secret(env: str, *, prompt: str | None = None) -> bytes:
    """Секрет зі змінної середовища ENV, з файла ENV_FILE або з інтерактивного запиту. Не з аргументів CLI.

    Файл секрету на POSIX не може бути доступним групі чи іншим (як ключі ssh)."""
    if os.environ.get(env):
        return os.environ[env].encode("utf-8")
    if os.environ.get(env + "_FILE"):
        path = Path(os.environ[env + "_FILE"])
        if os.name == "posix" and path.stat().st_mode & (stat.S_IRWXG | stat.S_IRWXO):
            raise VaultError("E_KEY_PERMS", f"файл секрету {env}_FILE доступний іншим користувачам; chmod 600",
                             Exit.KEY)
        return path.read_bytes().strip(b"\r\n")
    if prompt and _interactive():
        return getpass.getpass(prompt).encode("utf-8")
    raise VaultError("E_KEY", f"секрет не задано: встановіть змінну {env} або {env}_FILE", Exit.KEY)


def _interactive() -> bool:
    """Чи є справжня консоль для запиту фрази. У Windows NUL (stdin служби чи перенаправлення) теж
    «tty», а getpass читає консоль напряму й зависав би — тому перевіряється GetConsoleMode."""
    if sys.stdin is None or not sys.stdin.isatty():
        return False
    if os.name != "nt":
        return True
    import ctypes
    import msvcrt
    try:
        handle = msvcrt.get_osfhandle(sys.stdin.fileno())
        return bool(ctypes.windll.kernel32.GetConsoleMode(handle, ctypes.byref(ctypes.c_ulong())))
    except OSError:
        return False


class PassphraseKeys(RepoKeys):
    """Ключі покоління 0 (сховище без keyring) — сумісний інтерфейс етапу 1."""

    def __init__(self, passphrase: bytes, kdf: dict):
        super().__init__(passphrase, {"kdf": kdf})


class CryptoBox:
    def __init__(self, keys: KeyProvider):
        self.keys = keys

    def object_id(self, kind: int, plain: bytes) -> str:
        return hmac.new(self.keys.id_key(), bytes((kind,)) + plain, hashlib.sha256).hexdigest()

    def content_id(self, sha256_digest: bytes) -> str:
        return hmac.new(self.keys.id_key(), b"content" + sha256_digest, hashlib.sha256).hexdigest()

    def seal(self, kind: int, oid: str, codec_id: int, plain_len: int, payload: bytes) -> bytes:
        key_id = self.keys.key_id
        header = HEADER.pack(MAGIC, OBJECT_FORMAT_VERSION, kind, codec_id, bytes.fromhex(key_id),
                             os.urandom(12), plain_len)
        aead = AESGCM(self.keys.data_key(key_id))
        return header + aead.encrypt(header[15:27], payload, header + bytes.fromhex(oid))

    def open(self, blob: bytes, kind: int, oid: str) -> tuple[int, int, bytes]:
        """(codec_id, plain_len, payload). Будь-яка невідповідність -> VaultError до видачі даних."""
        if len(blob) < OVERHEAD:
            raise VaultError("E_SHORT", "об'єкт коротший за заголовок і тег", Exit.INTEGRITY)
        magic, version, got_kind, codec_id, key_id, nonce, plain_len = HEADER.unpack_from(blob)
        if magic != MAGIC or version != OBJECT_FORMAT_VERSION:
            raise VaultError("E_MAGIC", "це не об'єкт LOG3 Vault формату 1", Exit.INTEGRITY)
        if got_kind != kind:
            raise VaultError("E_KIND", f"тип об'єкта {got_kind}, очікувався {kind}", Exit.INTEGRITY)
        key = self.keys.data_key(key_id.hex())
        try:
            payload = AESGCM(key).decrypt(nonce, blob[HEADER.size:], blob[:HEADER.size] + bytes.fromhex(oid))
        except InvalidTag:
            raise VaultError("E_AUTH", f"об'єкт {oid[:12]} не пройшов автентифікацію", Exit.INTEGRITY) from None
        return codec_id, plain_len, payload
