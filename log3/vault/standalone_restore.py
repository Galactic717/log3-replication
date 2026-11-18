#!/usr/bin/env python3
"""Аварійне відновлення LOG3 Vault без інсталяції, без каталогу й без ML.

Один файл: потрібні лише Python >= 3.11 і бібліотеки cryptography та zstandard. Пакет log3,
config.json і catalog.sqlite не потрібні. Джерело даних — каталог backend-а (localdir) або вузол.

    python log3_restore.py --backend /mnt/usb/log3 list
    python log3_restore.py --backend /mnt/usb/log3 restore latest ./restored
    python log3_restore.py --node-url https://host:8443 --ca-file node.crt list      # токен: LOG3_NODE_TOKEN

Ключі: парольна фраза сховища (LOG3_VAULT_PASSPHRASE) разом із repo-config.json із backend-а, або
файл експорту ключів (--key-export, фраза LOG3_VAULT_EXPORT_PASSPHRASE) — тоді repo-config.json не
потрібен. Кожен об'єкт перевіряється так само, як у log3.vault.restore: SHA-256 і довжина
зашифрованого об'єкта, AES-256-GCM з AAD, довжина після декодування, ключований object_id;
файл — за SHA-256 вмісту. Підписи службових файлів (HMAC) перевіряються; непідписані — лише з
--accept-legacy-meta. Наявні файли в DEST не перезаписуються.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import http.client
import json
import os
import ssl
import struct
import sys
from pathlib import Path
from urllib.parse import urlsplit

import zstandard
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDFExpand
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

HEADER = struct.Struct(">4sBBB8s12sQ")
KIND_CHUNK, KIND_MANIFEST = 1, 2
KDF_LIMITS = {"n": 2 ** 20, "r": 16, "p": 4}


class Fail(Exception):
    pass


def _expand(key: bytes, label: bytes) -> bytes:
    return HKDFExpand(hashes.SHA256(), 32, b"log3-vault/v1/" + label).derive(key)


def _master(passphrase: bytes, kdf: dict) -> bytes:
    n, r, p = int(kdf["n"]), int(kdf["r"]), int(kdf["p"])
    if kdf.get("name") != "scrypt" or n & (n - 1) or n > KDF_LIMITS["n"] or r > KDF_LIMITS["r"] or p > KDF_LIMITS["p"]:
        raise Fail("KDF поза безпечними межами")
    return Scrypt(salt=base64.b64decode(kdf["salt_b64"]), length=32, n=n, r=r, p=p).derive(passphrase)


def _unwrap(kek: bytes, blob: str, label: str) -> bytes:
    raw = base64.b64decode(blob)
    try:
        return AESGCM(kek).decrypt(raw[:12], raw[12:], b"log3-vault/v1/wrap/" + label.encode())
    except InvalidTag:
        raise Fail("фраза не підходить до цього сховища") from None


def load_keys(passphrase: bytes, crypto: dict) -> tuple[dict, bytes]:
    """({key_id: ключ даних}, ключ ідентифікаторів) — та сама ієрархія, що й log3.vault.keys."""
    master = _master(passphrase, crypto["kdf"])
    ring = crypto.get("keyring")
    if ring is None:
        kid = hmac.new(_expand(master, b"check"), b"key-id", hashlib.sha256).hexdigest()[:16]
        if crypto.get("key_id") not in (None, kid):
            raise Fail("фраза не підходить до цього сховища")
        return {kid: _expand(master, b"data")}, _expand(master, b"id")
    kek = _expand(master, b"kek")
    return ({k["key_id"]: _unwrap(kek, k["wrapped"], k["key_id"]) for k in ring["keys"]},
            _unwrap(kek, ring["id_key"], "id"))


def check_meta(meta_key: bytes, name: str, obj: dict, legacy_ok: bool) -> dict:
    mac = obj.get("meta_mac")
    if mac is None:
        if not legacy_ok:
            raise Fail(f"{name} без підпису; якщо backend довірений, додайте --accept-legacy-meta")
        return obj
    body = json.dumps({k: v for k, v in obj.items() if k != "meta_mac"}, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode()
    want = hmac.new(meta_key, b"log3-vault/v1/meta\x00" + name.encode() + b"\x00" + body, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(str(mac), want):
        raise Fail(f"{name}: підпис не збігається")
    return obj


class DirSource:
    def __init__(self, root: Path):
        self.root = root
        if not (root / "log3-backend.json").is_file():
            raise Fail(f"{root} не є каталогом backend-а LOG3 Vault")

    def meta(self, name: str) -> bytes | None:
        p = self.root / "meta" / name
        return p.read_bytes() if p.is_file() else None

    def names(self) -> list[str]:
        return sorted(p.name for p in (self.root / "meta").iterdir())

    def obj(self, oid: str) -> bytes:
        p = self.root / "objects" / oid[:2] / oid
        if not p.is_file():
            raise Fail(f"об'єкта {oid[:12]} нема")
        return p.read_bytes()


class NodeSource:
    def __init__(self, url: str, ca_file: str):
        u = urlsplit(url)
        if u.scheme != "https":
            raise Fail("потрібна адреса https://")
        self.host, self.port = u.hostname, u.port or 443
        self.ctx = ssl.create_default_context(cafile=ca_file)
        self.token = os.environ.get("LOG3_NODE_TOKEN", "")

    def _get(self, path: str) -> tuple[int, bytes]:
        conn = http.client.HTTPSConnection(self.host, self.port, context=self.ctx, timeout=60)
        try:
            conn.request("GET", path, headers={"Authorization": f"Bearer {self.token}"})
            r = conn.getresponse()
            return r.status, r.read()
        finally:
            conn.close()

    def meta(self, name: str) -> bytes | None:
        status, data = self._get(f"/v1/meta/{name}")
        return data if status == 200 else None

    def names(self) -> list[str]:
        status, data = self._get("/v1/meta")
        if status != 200:
            raise Fail(f"вузол відповів {status}")
        return json.loads(data)["names"]

    def obj(self, oid: str) -> bytes:
        status, data = self._get(f"/v1/objects/{oid}")
        if status != 200:
            raise Fail(f"об'єкт {oid[:12]}: HTTP {status}")
        return data


def _huffman(payload: bytes, plain_len: int) -> bytes:
    """Канонічний Гаффман LOG3 (кодек 2): u16 кількість символів, пари символ/довжина, бітовий потік."""
    (nsym,) = struct.unpack_from(">H", payload)
    table = payload[2:2 + 2 * nsym]
    lengths = {table[i]: table[i + 1] for i in range(0, len(table), 2)}
    count, code, by_code = [0] * 16, 0, {}
    for n in lengths.values():
        count[n] += 1
    nxt = [0] * 16
    for bits in range(1, 16):
        code = (code + count[bits - 1]) << 1
        nxt[bits] = code
    for s in sorted(lengths):
        by_code[(lengths[s], nxt[lengths[s]])] = s
        nxt[lengths[s]] += 1
    out, code, n = bytearray(), 0, 0
    for byte in payload[2 + 2 * nsym:]:
        for i in range(7, -1, -1):
            if len(out) == plain_len:
                break
            code, n = (code << 1) | (byte >> i & 1), n + 1
            if (n, code) in by_code:
                out.append(by_code[(n, code)])
                code = n = 0
            elif n >= 15:
                raise Fail("пошкоджений потік Гаффмана")
    return bytes(out)


class Vault:
    def __init__(self, src, keys: dict, id_key: bytes, legacy_ok: bool):
        self.src, self.keys, self.id_key, self.legacy_ok = src, keys, id_key, legacy_ok
        self.meta_key = _expand(id_key, b"meta")

    def open(self, oid: str, kind: int, length: int, sha256: str) -> bytes:
        blob = self.src.obj(oid)
        if len(blob) != length or hashlib.sha256(blob).hexdigest() != sha256:
            raise Fail(f"об'єкт {oid[:12]}: довжина або SHA-256 не збігаються")
        magic, version, got_kind, codec, key_id, nonce, plain_len = HEADER.unpack_from(blob)
        if magic != b"L3FB" or version != 1 or got_kind != kind or key_id.hex() not in self.keys:
            raise Fail(f"об'єкт {oid[:12]}: невідомий формат, тип або ключ")
        try:
            payload = AESGCM(self.keys[key_id.hex()]).decrypt(nonce, blob[HEADER.size:],
                                                              blob[:HEADER.size] + bytes.fromhex(oid))
        except InvalidTag:
            raise Fail(f"об'єкт {oid[:12]} не пройшов автентифікацію") from None
        if codec == 0:
            data = payload
        elif codec == 1:
            data = zstandard.ZstdDecompressor().decompress(payload, max_output_size=plain_len) if plain_len else b""
        elif codec == 2:
            data = _huffman(payload, plain_len)
        else:
            raise Fail(f"невідомий кодек {codec}")
        if len(data) != plain_len or hmac.new(self.id_key, bytes((kind,)) + data, hashlib.sha256).hexdigest() != oid:
            raise Fail(f"об'єкт {oid[:12]}: вміст не відповідає ідентифікатору")
        return data

    def versions(self) -> list[dict]:
        out = []
        for name in self.src.names():
            if name.startswith("version-"):
                try:
                    out.append(check_meta(self.meta_key, name, json.loads(self.src.meta(name)), self.legacy_ok))
                except (Fail, ValueError):
                    continue
        return sorted(out, key=lambda p: p["created_at_ns"])

    def manifest(self, ptr: dict) -> dict:
        m = json.loads(self.open(ptr["manifest_oid"], KIND_MANIFEST, ptr["manifest_len"], ptr["manifest_sha256"]))
        if m.get("format") != "log3-file-manifest" or m.get("version_id") != ptr["version_id"]:
            raise Fail("маніфест не відповідає покажчику")
        return m

    def restore(self, ptr: dict, dest: Path) -> dict:
        m = self.manifest(ptr)
        dest.mkdir(parents=True, exist_ok=True)
        files = size = 0
        for e in m["entries"]:
            parts = e["path"].split("/")
            if any(p in ("", ".", "..") or "\\" in p or ":" in p for p in parts):
                raise Fail(f"небезпечний шлях {e['path']!r}")
            target = dest.joinpath(*parts)
            if e["type"] == "dir":
                target.mkdir(parents=True, exist_ok=True)
                continue
            if os.path.lexists(target):
                raise Fail(f"{target} вже існує")
            target.parent.mkdir(parents=True, exist_ok=True)
            c, h, part = m["contents"][e["content_id"]], hashlib.sha256(), target.with_name(target.name + ".part")
            with open(part, "wb") as f:
                for oid in c["chunks"]:
                    o = m["objects"][oid]
                    data = self.open(oid, KIND_CHUNK, o["stored_len"], o["sha256"])
                    f.write(data)
                    h.update(data)
                f.flush()
                os.fsync(f.fileno())
            if h.hexdigest() != c["sha256"]:
                part.unlink()
                raise Fail(f"{e['path']}: SHA-256 вмісту не збігається")
            os.replace(part, target)
            os.utime(target, ns=(e["mtime_ns"], e["mtime_ns"]))
            files, size = files + 1, size + e["size"]
        return {"version_id": m["version_id"], "files": files, "bytes": size, "dest": str(dest)}


def _secret(env: str) -> bytes:
    if os.environ.get(env):
        return os.environ[env].encode()
    if os.environ.get(env + "_FILE"):
        return Path(os.environ[env + "_FILE"]).read_bytes().strip(b"\r\n")
    raise Fail(f"задайте {env} або {env}_FILE")


def open_vault(a) -> Vault:
    src = DirSource(Path(a.backend)) if a.backend else NodeSource(a.node_url, a.ca_file)
    if a.key_export:
        doc = json.loads(Path(a.key_export).read_bytes())
        keys, id_key = load_keys(_secret("LOG3_VAULT_EXPORT_PASSPHRASE"), doc)
        return Vault(src, keys, id_key, a.accept_legacy_meta)
    raw = src.meta("repo-config.json")
    if raw is None:
        raise Fail("на backend-і нема repo-config.json; використайте --key-export")
    cfg = json.loads(raw)
    keys, id_key = load_keys(_secret(cfg["crypto"].get("passphrase_env", "LOG3_VAULT_PASSPHRASE")), cfg["crypto"])
    vault = Vault(src, keys, id_key, a.accept_legacy_meta)
    check_meta(vault.meta_key, "repo-config.json", cfg, a.accept_legacy_meta)
    return vault


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):              # Windows: канал stdout не завжди UTF-8
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(prog="log3_restore.py", description=__doc__.split("\n\n")[0])
    p.add_argument("--backend", help="каталог backend-а localdir")
    p.add_argument("--node-url")
    p.add_argument("--ca-file")
    p.add_argument("--key-export", help="файл експорту ключів (log3-vault key export)")
    p.add_argument("--accept-legacy-meta", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    r = sub.add_parser("restore")
    r.add_argument("version", help="ідентифікатор, префікс або latest")
    r.add_argument("dest")
    a = p.parse_args(argv)
    if not a.backend and not (a.node_url and a.ca_file):
        p.error("потрібен --backend або --node-url з --ca-file")
    try:
        vault = open_vault(a)
        versions = vault.versions()
        if a.cmd == "list":
            out = {"versions": [{"version_id": v["version_id"], "created_at_ns": v["created_at_ns"]} for v in versions]}
        else:
            hit = versions[-1:] if a.version == "latest" else [v for v in versions
                                                              if v["version_id"].startswith(a.version)]
            if len(hit) != 1:
                raise Fail(f"версію {a.version!r} не знайдено або префікс неоднозначний")
            out = {"restore": vault.restore(hit[0], Path(a.dest))}
        print(json.dumps({"ok": True, **out}, ensure_ascii=False))
        return 0
    except (Fail, OSError, ValueError, KeyError, zstandard.ZstdError) as e:
        print(json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"}, ensure_ascii=False))
        return 7 if isinstance(e, Fail) else 1


if __name__ == "__main__":
    sys.exit(main())
