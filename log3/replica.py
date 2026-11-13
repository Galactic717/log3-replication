"""Репліка LOG3: HEADER(43) | CIPHERTEXT | HMAC-SHA256(32) (FORMAT_SPEC_P3 §2, §5).

Заголовок (big-endian):
    0  4  ASCII "LOG3"
    4  1  version = 1
    5  2  replica_group_id
    7  4  batch_seq
    11 16 salt
    27 12 nonce
    39 4  ciphertext_len

Побудова: payload -> AES-256-CTR(K_enc) -> HMAC(K_mac) над заголовком і шифротекстом
(Encrypt-then-MAC). Приймання — суворо в порядку §5:
    E_SHORT -> E_MAGIC (magic/version) -> E_LENGTH -> E_HMAC -> розшифрування -> E_PAYLOAD/E_CODE.
HMAC перевіряється ДО розшифрування: змінений чи чужий шифротекст навіть не
потрапляє в парсер payload. Неправильний пароль дає інший K_mac, тому виглядає як E_HMAC.
"""
from __future__ import annotations

import os
import struct
from collections.abc import Callable
from dataclasses import dataclass

from . import crypto
from .payload import PayloadError, build_payload, parse_payload
from .records import Record

MAGIC = b"LOG3"
VERSION = 1
HEADER = struct.Struct(">4sBHI16s12sI")
HEADER_SIZE = HEADER.size                    # 43
MIN_SIZE = HEADER_SIZE + crypto.MAC_SIZE     # 75


class Log3Error(Exception):
    """Контрольована відмова приймання. code: E_SHORT | E_MAGIC | E_LENGTH | E_HMAC | E_PAYLOAD | E_CODE."""

    def __init__(self, code: str, msg: str = ""):
        super().__init__(f"{code}: {msg}" if msg else code)
        self.code = code


@dataclass(frozen=True)
class Header:
    replica_group_id: int
    batch_seq: int
    salt: bytes
    nonce: bytes
    ciphertext_len: int


@dataclass(frozen=True)
class Replica:
    header: Header
    records: list[Record]
    class_table: list[int] | None


def seal_payload(payload: bytes, password: str | bytes, *, replica_group_id: int, batch_seq: int,
                 salt: bytes | None = None, nonce: bytes | None = None) -> bytes:
    """Готовий payload -> репліка. salt — одна на сеанс; nonce не можна повторювати
    з тим самим K_enc (однаковий keystream => XOR двох шифротекстів = XOR відкритих
    текстів), тому за замовчуванням він випадковий."""
    salt = os.urandom(crypto.SALT_SIZE) if salt is None else salt
    nonce = os.urandom(crypto.NONCE_SIZE) if nonce is None else nonce
    k_enc, k_mac = crypto.derive_keys(password, salt)
    ct = crypto.aes_ctr(k_enc, nonce, payload)
    header = HEADER.pack(MAGIC, VERSION, replica_group_id, batch_seq, salt, nonce, len(ct))
    return header + ct + crypto.mac(k_mac, header + ct)


def build_replica(records: list[Record], password: str | bytes, *, replica_group_id: int,
                  batch_seq: int, salt: bytes | None = None, nonce: bytes | None = None,
                  compress: bool = True, no_compression_threshold: float | None = None,
                  class_table: list[int] | None = None,
                  lengths: dict[int, int] | None = None) -> bytes:
    payload = build_payload(records, compress=compress,
                            no_compression_threshold=no_compression_threshold,
                            class_table=class_table, lengths=lengths)
    return seal_payload(payload, password, replica_group_id=replica_group_id,
                        batch_seq=batch_seq, salt=salt, nonce=nonce)


def parse_header(blob: bytes) -> Header:
    """Кроки 1–3 приймання: мінімальна довжина, magic/version, точна довжина."""
    if len(blob) < MIN_SIZE:
        raise Log3Error("E_SHORT", f"{len(blob)} < {MIN_SIZE} байтів")
    magic, version, group, seq, salt, nonce, ct_len = HEADER.unpack_from(blob, 0)
    if magic != MAGIC:
        raise Log3Error("E_MAGIC", f"magic {magic!r}")
    if version != VERSION:
        raise Log3Error("E_MAGIC", f"версія {version}")
    if HEADER_SIZE + ct_len + crypto.MAC_SIZE != len(blob):
        raise Log3Error("E_LENGTH", f"ciphertext_len={ct_len}, довжина репліки {len(blob)}")
    return Header(group, seq, salt, nonce, ct_len)


def parse_replica(blob: bytes, password: str | bytes) -> Replica:
    """Приймання в порядку §5. Назовні — лише Log3Error."""
    blob = bytes(blob)
    header = parse_header(blob)
    k_enc, k_mac = crypto.derive_keys(password, header.salt)
    body_end = HEADER_SIZE + header.ciphertext_len
    if not crypto.mac_ok(k_mac, blob[:body_end], blob[body_end:]):
        raise Log3Error("E_HMAC", "HMAC не збігся (змінені дані або невірний пароль)")
    payload = crypto.aes_ctr(k_enc, header.nonce, blob[HEADER_SIZE:body_end])
    try:
        parsed = parse_payload(payload)
    except PayloadError as e:
        raise Log3Error(e.code, str(e)) from None
    return Replica(header, parsed.records, parsed.class_table)


def as_receiver(password: str | bytes) -> Callable[[bytes], str | None]:
    """Адаптер для attack/: receiver(blob) -> None, якщо прийнято, інакше код помилки."""
    def receiver(blob: bytes) -> str | None:
        try:
            parse_replica(blob, password)
        except Log3Error as e:
            return e.code
        return None
    return receiver
