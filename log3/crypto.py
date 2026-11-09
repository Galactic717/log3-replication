"""Криптографія базового формату (FORMAT_SPEC_P3 §4).

    master = PBKDF2-HMAC-SHA256(password, salt, 100000, dkLen=64)
    K_enc = master[0:32]   — лише для AES-256-CTR
    K_mac = master[32:64]  — лише для HMAC-SHA256

Два ключі з одного master: компрометація чи помилка в одному режимі не дає
ключа для іншого. PBKDF2 навмисно повільний (100000 ітерацій), тому ключі
кешуються за (password, salt): одна сіль на сеанс -> один виклик PBKDF2,
інакше компонент T метрики Q обвалиться.
"""
from __future__ import annotations

import hashlib
import hmac
from functools import lru_cache

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

PBKDF2_ITERATIONS = 100_000
SALT_SIZE = 16
NONCE_SIZE = 12
KEY_SIZE = 32
MAC_SIZE = 32


def _to_bytes(password: str | bytes) -> bytes:
    return password.encode("utf-8") if isinstance(password, str) else bytes(password)


@lru_cache(maxsize=64)
def _derive(password: bytes, salt: bytes) -> tuple[bytes, bytes]:
    master = hashlib.pbkdf2_hmac("sha256", password, salt, PBKDF2_ITERATIONS, dklen=2 * KEY_SIZE)
    return master[:KEY_SIZE], master[KEY_SIZE:]


def derive_keys(password: str | bytes, salt: bytes) -> tuple[bytes, bytes]:
    """(K_enc, K_mac); кеш за (password, salt)."""
    if len(salt) != SALT_SIZE:
        raise ValueError(f"сіль має бути {SALT_SIZE} байтів")
    return _derive(_to_bytes(password), bytes(salt))


def aes_ctr(key: bytes, nonce: bytes, data: bytes, first_counter: int = 1) -> bytes:
    """AES-256-CTR, блок лічильника nonce(12) || uint32_BE(i), i від first_counter (за форматом 1).

    Шифрування й розшифрування — одна операція (XOR з keystream).
    cryptography збільшує весь 128-бітовий блок як big-endian число, що збігається
    з інкрементом uint32 у молодших 4 байтах, поки i не переповнюється
    (2^32 блоків = 64 ГіБ — недосяжно для репліки).
    """
    if len(nonce) != NONCE_SIZE:
        raise ValueError(f"nonce має бути {NONCE_SIZE} байтів")
    counter_block = nonce + first_counter.to_bytes(4, "big")
    enc = Cipher(algorithms.AES(key), modes.CTR(counter_block)).encryptor()
    return enc.update(data) + enc.finalize()


def mac(key: bytes, data: bytes) -> bytes:
    return hmac.new(key, data, hashlib.sha256).digest()


def mac_ok(key: bytes, data: bytes, tag: bytes) -> bool:
    """Порівняння за сталий час: час відповіді не підказує, скільки байтів тегу збіглося."""
    return hmac.compare_digest(mac(key, data), tag)
