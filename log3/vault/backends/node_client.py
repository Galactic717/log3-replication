"""Клієнт віддаленого вузла (мережевий backend).

Транспорт — лише HTTPS. Сертифікат вузла перевіряється за закріпленим файлом ca_file
(разом з іменем хоста), вузол перевіряє клієнта за токеном. Токен береться зі змінної
середовища, назву якої зберігає конфігурація; самого токена в конфігурації нема.
"""
from __future__ import annotations

import http.client
import json
import ssl
from pathlib import Path
from urllib.parse import urlsplit

from ..cryptobox import read_secret
from ..errors import BackendError, VaultError
from ..fsutil import IO_BLOCK
from .base import Ack, check_meta_name, check_oid


class NodeBackend:
    def __init__(self, name: str, url: str, ca_file: str, token_env: str, timeout: float = 60.0):
        u = urlsplit(url)
        if u.scheme != "https" or not u.hostname:
            raise BackendError(f"backend {name}: потрібна адреса https://host:port (незахищений транспорт заборонено)")
        self.name, self.host, self.port = name, u.hostname, u.port or 443
        self.ca_file, self.token_env, self.timeout = ca_file, token_env, timeout
        self.durability = "підтверджує вузол (див. /v1/info)"

    def _request(self, method: str, path: str, *, body=None, headers: dict | None = None,
                 sink: Path | None = None) -> tuple[int, bytes]:
        try:
            token = read_secret(self.token_env).decode()
        except VaultError as e:
            raise BackendError(e.message) from None
        conn = None
        try:
            ctx = ssl.create_default_context(cafile=self.ca_file)   # перевірка сертифіката й імені вузла
            ctx.minimum_version = ssl.TLSVersion.TLSv1_2
            conn = http.client.HTTPSConnection(self.host, self.port, context=ctx, timeout=self.timeout)
            conn.request(method, path, body=body, headers={"Authorization": f"Bearer {token}", **(headers or {})})
            resp = conn.getresponse()
            if sink is not None and resp.status == 200:
                with open(sink, "wb") as f:
                    while block := resp.read(IO_BLOCK):
                        f.write(block)
                return 200, b""
            return resp.status, resp.read()
        except (OSError, http.client.HTTPException) as e:           # ssl.SSLError теж OSError
            raise BackendError(f"вузол {self.name} недоступний: {type(e).__name__}") from None
        finally:
            if conn is not None:
                conn.close()

    def _fail(self, status: int, data: bytes) -> BackendError:
        if status in (401, 403):
            return BackendError(f"вузол {self.name} відхилив автентифікацію")
        if status == 507:
            return BackendError(f"вузол {self.name}: квоту вичерпано")
        return BackendError(f"вузол {self.name}: HTTP {status} {data[:120].decode('utf-8', 'replace')}")

    def _ack(self, data: bytes, duplicate: bool = False) -> Ack:
        try:
            d = json.loads(data)
            return Ack(str(d["oid"]), int(d["length"]), str(d["sha256"]), str(d["durability"]),
                       bool(d.get("duplicate", duplicate)))
        except (ValueError, KeyError, TypeError):
            raise BackendError(f"вузол {self.name}: нерозбірливе підтвердження") from None

    def put(self, oid: str, src: Path, length: int, sha256: str, op_id: str) -> Ack:
        with open(src, "rb") as f:
            status, data = self._request("PUT", f"/v1/objects/{check_oid(oid)}", body=f, headers={
                "Content-Length": str(length), "X-Log3-Sha256": sha256, "X-Log3-Op": op_id})
        if status != 200:
            raise self._fail(status, data)
        return self._ack(data)

    def stat(self, oid: str, *, rehash: bool = True) -> Ack | None:
        status, data = self._request("GET", f"/v1/stat/{check_oid(oid)}?rehash={int(rehash)}")
        if status == 404:
            return None
        if status != 200:
            raise self._fail(status, data)
        return self._ack(data)

    def get(self, oid: str, dest: Path) -> None:
        status, data = self._request("GET", f"/v1/objects/{check_oid(oid)}", sink=dest)
        if status != 200:
            raise self._fail(status, data)

    def delete(self, oid: str) -> None:
        status, data = self._request("DELETE", f"/v1/objects/{check_oid(oid)}")
        if status != 200:
            raise self._fail(status, data)

    def put_meta(self, name: str, data: bytes) -> None:
        status, out = self._request("PUT", f"/v1/meta/{check_meta_name(name)}", body=data)
        if status != 200:
            raise self._fail(status, out)

    def get_meta(self, name: str) -> bytes | None:
        status, data = self._request("GET", f"/v1/meta/{check_meta_name(name)}")
        if status == 404:
            return None
        if status != 200:
            raise self._fail(status, data)
        return data

    def delete_meta(self, name: str) -> None:
        status, data = self._request("DELETE", f"/v1/meta/{check_meta_name(name)}")
        if status != 200:
            raise self._fail(status, data)

    def list_meta(self) -> list[str]:
        status, data = self._request("GET", "/v1/meta")
        if status != 200:
            raise self._fail(status, data)
        return list(json.loads(data)["names"])

    def info(self) -> dict:
        status, data = self._request("GET", "/v1/info")
        if status != 200:
            raise self._fail(status, data)
        return json.loads(data)
