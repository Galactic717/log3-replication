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

import base64
import getpass
import hashlib
import hmac
import os
import struct
import sys
from pathlib import Path
from typing import Protocol

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDFExpand
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

from .errors import Exit, VaultError

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
    """Секрет зі змінної середовища ENV, з файла ENV_FILE або з інтерактивного запиту. Не з аргументів CLI."""
    if os.environ.get(env):
        return os.environ[env].encode("utf-8")
    if os.environ.get(env + "_FILE"):
        return Path(os.environ[env + "_FILE"]).read_bytes().strip(b"\r\n")
    if prompt and sys.stdin.isatty():
        return getpass.getpass(prompt).encode("utf-8")
    raise VaultError("E_KEY", f"секрет не задано: встановіть змінну {env} або {env}_FILE", Exit.KEY)


def new_kdf_params(n: int = 2 ** 15) -> dict:
    return {"name": "scrypt", "salt_b64": base64.b64encode(os.urandom(16)).decode(), "n": n, "r": 8, "p": 1}


def _expand(master: bytes, label: bytes) -> bytes:
    return HKDFExpand(hashes.SHA256(), 32, b"log3-vault/v1/" + label).derive(master)


class PassphraseKeys:
    """Мінімальний KeyProvider: один майстер-ключ із парольної фрази (scrypt), підключі через HKDF."""

    def __init__(self, passphrase: bytes, kdf: dict):
        if kdf.get("name") != "scrypt" or not passphrase:
            raise VaultError("E_KEY", "потрібні непорожня парольна фраза і KDF scrypt", Exit.KEY)
        salt = base64.b64decode(kdf["salt_b64"])
        master = Scrypt(salt=salt, length=32, n=int(kdf["n"]), r=int(kdf["r"]), p=int(kdf["p"])).derive(passphrase)
        self._data, self._id = _expand(master, b"data"), _expand(master, b"id")
        self.key_id = hmac.new(_expand(master, b"check"), b"key-id", hashlib.sha256).hexdigest()[:16]

    def data_key(self, key_id: str) -> bytes:
        if key_id != self.key_id:
            raise VaultError("E_KEY", f"об'єкт зашифровано ключем {key_id}, якого нема в цього провайдера", Exit.KEY)
        return self._data

    def id_key(self) -> bytes:
        return self._id


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
