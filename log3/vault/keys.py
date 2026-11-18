"""Ключі сховища: keyring, ротація, зміна парольної фрази, аварійний експорт, підпис службових файлів.

Ієрархія:
  парольна фраза --scrypt(сіль, n, r, p)--> майстер-ключ --HKDF "kek"--> KEK
  KEK загортає (AES-256-GCM, AAD = призначення ключа) ключі даних і ключ ідентифікаторів
  ключ даних (32 B, key_id 16 hex)  шифрує об'єкти; нові об'єкти — лише активним ключем
  ключ ідентифікаторів              object_id, content_id (HMAC); не змінюється ротацією,
                                    інакше зламалась би дедуплікація
  ключ службових файлів = HKDF(ключ ідентифікаторів, "meta")
                                    HMAC repo-config.json і покажчиків версій на backend-ах

Сховища етапів 1–2 не мають keyring: ключ даних і ключ ідентифікаторів виводяться з майстер-ключа
напряму (покоління 0). Перша ротація або зміна фрази загортає саме їх, тож старі об'єкти лишаються
читабельними, а їхній key_id не змінюється.

Nonce AES-GCM — 96 випадкових біт на кожен об'єкт. Щоб імовірність повтору лишалась мізерною,
кількість об'єктів на один ключ обмежено (crypto.max_objects_per_key, типово 2^30 ≪ 2^32):
лічильник у каталозі монотонний, досягнення межі вмикає автоматичну ротацію перед backup.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
from datetime import UTC, datetime

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDFExpand
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

from .errors import Exit, VaultError

KDF_LIMITS = {"n": 2 ** 20, "r": 16, "p": 4}       # межі параметрів із недовіреного repo-config.json
MIN_PASSPHRASE = 12
MAX_OBJECTS_PER_KEY = 2 ** 30
EXPORT_FORMAT = "log3-key-export"
META_DOMAIN = b"log3-vault/v1/meta\x00"


def _b64(b: bytes) -> str:
    return base64.b64encode(b).decode()


def _expand(key: bytes, label: bytes) -> bytes:
    return HKDFExpand(hashes.SHA256(), 32, b"log3-vault/v1/" + label).derive(key)


def new_kdf_params(n: int = 2 ** 15) -> dict:
    return {"name": "scrypt", "salt_b64": _b64(os.urandom(16)), "n": n, "r": 8, "p": 1}


def check_passphrase(passphrase: bytes) -> bytes:
    if len(passphrase.decode("utf-8", "replace")) < MIN_PASSPHRASE:
        raise VaultError("E_WEAK_KEY", f"парольна фраза коротша за {MIN_PASSPHRASE} символів", Exit.KEY)
    return passphrase


def master_key(passphrase: bytes, kdf: dict) -> bytes:
    """scrypt із перевіркою меж: параметри можуть прийти з недовіреного backend-а."""
    try:
        n, r, p, salt = int(kdf["n"]), int(kdf["r"]), int(kdf["p"]), base64.b64decode(kdf["salt_b64"])
    except (KeyError, TypeError, ValueError):
        raise VaultError("E_CONFIG", "некоректні параметри KDF", Exit.KEY) from None
    if kdf.get("name") != "scrypt" or not passphrase or n < 2 or n & (n - 1) or len(salt) < 16 \
            or any(v > KDF_LIMITS[k] or v < 1 for k, v in (("n", n), ("r", r), ("p", p))):
        raise VaultError("E_CONFIG", "KDF має бути scrypt у безпечних межах і з непорожньою фразою", Exit.KEY)
    return Scrypt(salt=salt, length=32, n=n, r=r, p=p).derive(passphrase)


def _wrap(kek: bytes, key: bytes, label: str) -> str:
    nonce = os.urandom(12)
    return _b64(nonce + AESGCM(kek).encrypt(nonce, key, b"log3-vault/v1/wrap/" + label.encode()))


def _unwrap(kek: bytes, blob: str, label: str) -> bytes:
    raw = base64.b64decode(blob)
    try:
        return AESGCM(kek).decrypt(raw[:12], raw[12:], b"log3-vault/v1/wrap/" + label.encode())
    except InvalidTag:
        raise VaultError("E_KEY", "парольна фраза не відповідає цьому сховищу", Exit.KEY) from None


def legacy_key_id(master: bytes) -> str:
    return hmac.new(_expand(master, b"check"), b"key-id", hashlib.sha256).hexdigest()[:16]


class RepoKeys:
    """KeyProvider сховища: активний ключ для нових об'єктів і всі попередні для читання."""

    def __init__(self, passphrase: bytes, crypto: dict):
        master = master_key(passphrase, crypto["kdf"])
        ring = crypto.get("keyring")
        if ring is None:                                    # покоління 0: сховище етапів 1–2
            kid = legacy_key_id(master)
            if crypto.get("key_id") not in (None, kid):
                raise VaultError("E_KEY", "парольна фраза не відповідає цьому сховищу", Exit.KEY)
            self.keys, self._id = {kid: _expand(master, b"data")}, _expand(master, b"id")
            self.key_id = kid
        else:
            kek = _expand(master, b"kek")
            self._id = _unwrap(kek, ring["id_key"], "id")
            self.keys = {k["key_id"]: _unwrap(kek, k["wrapped"], k["key_id"]) for k in ring["keys"]}
            self.key_id = crypto["key_id"]
            if self.key_id not in self.keys:
                raise VaultError("E_KEY", "активного ключа нема в keyring", Exit.KEY)

    def data_key(self, key_id: str) -> bytes:
        if key_id not in self.keys:
            raise VaultError("E_KEY", f"об'єкт зашифровано ключем {key_id}, якого нема в keyring", Exit.KEY)
        return self.keys[key_id]

    def id_key(self) -> bytes:
        return self._id

    def meta_key(self) -> bytes:
        return _expand(self._id, b"meta")


def _ring(keys: RepoKeys, kek: bytes, created: dict) -> dict:
    return {"format_version": 1, "id_key": _wrap(kek, keys.id_key(), "id"),
            "keys": [{"key_id": kid, "created_at": created.get(kid), "wrapped": _wrap(kek, k, kid)}
                     for kid, k in keys.keys.items()]}


def _created(crypto: dict) -> dict:
    return {k["key_id"]: k.get("created_at") for k in (crypto.get("keyring") or {}).get("keys", [])}


def rotate(crypto: dict, passphrase: bytes) -> str:
    """Новий ключ даних стає активним; попередні лишаються для читання старих версій."""
    keys = RepoKeys(passphrase, crypto)
    kid = os.urandom(8).hex()
    keys.keys[kid] = os.urandom(32)
    created = {**_created(crypto), kid: datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")}
    crypto["keyring"] = _ring(keys, _expand(master_key(passphrase, crypto["kdf"]), b"kek"), created)
    crypto["key_id"] = kid
    return kid


def change_passphrase(crypto: dict, old: bytes, new: bytes) -> None:
    """Перезагортає всі ключі під нову фразу; самі ключі й старі об'єкти не змінюються."""
    keys = RepoKeys(old, crypto)
    kdf = {**new_kdf_params(int(crypto["kdf"]["n"])), "r": crypto["kdf"]["r"], "p": crypto["kdf"]["p"]}
    crypto["keyring"] = _ring(keys, _expand(master_key(check_passphrase(new), kdf), b"kek"), _created(crypto))
    crypto["kdf"], crypto["key_id"] = kdf, keys.key_id


def export_bundle(crypto: dict, passphrase: bytes, repo_id: str, export_passphrase: bytes) -> bytes:
    """Аварійний експорт усіх ключів під окремою фразою: restore без config.json і без інсталяції."""
    keys = RepoKeys(passphrase, crypto)
    kdf = new_kdf_params()
    kek = _expand(master_key(check_passphrase(export_passphrase), kdf), b"kek")
    return json.dumps({"format": EXPORT_FORMAT, "format_version": 1, "repo_id": repo_id,
                       "created_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"), "kdf": kdf,
                       "key_id": keys.key_id, "keyring": _ring(keys, kek, _created(crypto))},
                      indent=1).encode()


def load_export(data: bytes, export_passphrase: bytes) -> tuple[str, RepoKeys]:
    """(repo_id, ключі) з файла експорту."""
    try:
        doc = json.loads(data)
        if doc.get("format") != EXPORT_FORMAT:
            raise ValueError
    except ValueError:
        raise VaultError("E_KEY", "це не файл експорту ключів LOG3 Vault", Exit.KEY) from None
    return doc["repo_id"], RepoKeys(export_passphrase, {"kdf": doc["kdf"], "key_id": doc["key_id"],
                                                        "keyring": doc["keyring"]})


def _canonical(obj: dict) -> bytes:
    return json.dumps({k: v for k, v in obj.items() if k != "meta_mac"}, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def sign_meta(meta_key: bytes, name: str, obj: dict) -> bytes:
    """Службовий файл backend-а з HMAC, прив'язаним до його назви (покажчики не переставити)."""
    mac = hmac.new(meta_key, META_DOMAIN + name.encode() + b"\x00" + _canonical(obj), hashlib.sha256).hexdigest()
    return json.dumps({**obj, "meta_mac": mac}, ensure_ascii=False, sort_keys=True).encode("utf-8")


def parse_meta(raw: bytes) -> dict:
    try:
        obj = json.loads(raw)
    except ValueError:
        raise VaultError("E_META", "службовий файл не є JSON", Exit.INTEGRITY) from None
    if not isinstance(obj, dict):
        raise VaultError("E_META", "службовий файл не є JSON-об'єктом", Exit.INTEGRITY)
    return obj


def verify_meta(meta_key: bytes, name: str, obj: dict) -> dict:
    mac = obj.get("meta_mac")
    want = hmac.new(meta_key, META_DOMAIN + name.encode() + b"\x00" + _canonical(obj), hashlib.sha256).hexdigest()
    if not isinstance(mac, str) or not hmac.compare_digest(mac, want):
        raise VaultError("E_META", f"{name}: підпис не збігається (підміна або інше сховище)", Exit.INTEGRITY)
    return {k: v for k, v in obj.items() if k != "meta_mac"}
