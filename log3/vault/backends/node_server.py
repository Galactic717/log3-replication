"""Віддалений вузол зберігання: HTTPS-сервер над ObjectDir.

    python -m log3.vault node-cert --dir certs --host backup.example.org
    LOG3_NODE_TOKEN=... python -m log3.vault serve-node --data /srv/log3 --cert certs/node.crt --key certs/node.key

Захист: TLS >= 1.2 (клієнт перевіряє сертифікат вузла за закріпленим файлом), кожен запит
несе токен (порівняння за сталий час). Токен читається зі змінної середовища або файла.
Вузол бачить лише зашифровані об'єкти та відкриті службові файли без секретів.

Підтвердження PUT надсилається після fsync файла, перейменування, fsync каталогу й фіксації
обліку квоти. Якщо відповідь загубилась, повтор того самого PUT бачить готовий об'єкт:
нічого не дописує і квоту вдруге не списує (облік — INSERT OR IGNORE за oid).
"""
from __future__ import annotations

import hmac
import ipaddress
import json
import os
import sqlite3
import ssl
import threading
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from ..errors import FAULT_ENV, BackendError
from ..fsutil import DURABILITY, IO_BLOCK, read_blocks
from .localdir import ObjectDir

TOKEN_ENV = "LOG3_NODE_TOKEN"


class Ledger:
    """Облік квоти вузла: один рядок на об'єкт, тому повтор не списує місце вдруге."""

    def __init__(self, path: Path):
        self.lock = threading.Lock()
        self.db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("CREATE TABLE IF NOT EXISTS charges(oid TEXT PRIMARY KEY, length INTEGER NOT NULL,"
                        " op_id TEXT, puts INTEGER NOT NULL DEFAULT 1)")

    def used(self) -> int:
        with self.lock:
            return self.db.execute("SELECT COALESCE(SUM(length),0) FROM charges").fetchone()[0]

    def has(self, oid: str) -> bool:
        with self.lock:
            return self.db.execute("SELECT 1 FROM charges WHERE oid=?", (oid,)).fetchone() is not None

    def charge(self, oid: str, length: int, op_id: str) -> None:
        with self.lock:
            self.db.execute("INSERT INTO charges(oid,length,op_id) VALUES(?,?,?) "
                            "ON CONFLICT(oid) DO UPDATE SET puts=puts+1", (oid, length, op_id))

    def release(self, oid: str) -> None:
        with self.lock:
            self.db.execute("DELETE FROM charges WHERE oid=?", (oid,))


class NodeServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, addr, data: Path, token: bytes, tls: ssl.SSLContext, quota_bytes: int = 0):
        super().__init__(addr, _Handler)
        self.store, self.token, self.tls, self.quota = ObjectDir(data), token, tls, quota_bytes
        self.store.create()
        self.ledger = Ledger(data / "node-ledger.sqlite")

    def get_request(self):
        sock, addr = self.socket.accept()
        return self.tls.wrap_socket(sock, server_side=True, do_handshake_on_connect=False), addr


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: NodeServer

    def setup(self):
        self.request.settimeout(120)
        self.request.do_handshake()                            # рукостискання в потоці запиту, а не в accept
        super().setup()

    def log_message(self, fmt, *args):                         # заголовки (і токен) у журнал не потрапляють
        if os.environ.get("LOG3_NODE_QUIET") != "1":
            super().log_message(fmt, *args)

    def _json(self, status: int, obj) -> None:
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self, length: int):
        left = length
        while left > 0:
            block = self.rfile.read(min(IO_BLOCK, left))
            if not block:
                raise BackendError("з'єднання обірвалось під час передавання")
            left -= len(block)
            yield block

    def _route(self, method: str) -> None:
        srv = self.server
        got = self.headers.get("Authorization", "").removeprefix("Bearer ").encode()
        length = int(self.headers.get("Content-Length") or 0)
        if not hmac.compare_digest(got, srv.token):
            self.close_connection = True
            return self._json(401, {"error": "unauthorized"})
        url = urlsplit(self.path)
        parts = url.path.strip("/").split("/")
        try:
            if parts[:1] != ["v1"] or len(parts) > 3:
                return self._json(404, {"error": "not found"})
            kind, arg = parts[1], parts[2] if len(parts) == 3 else ""
            if kind == "info" and method == "GET":
                return self._json(200, {"durability": DURABILITY, "quota_bytes": srv.quota,
                                        "used_bytes": srv.ledger.used()})
            if kind == "objects" and method == "PUT":
                return self._put(arg, length)
            if kind == "objects" and method == "GET":
                path = srv.store.path(arg)
                if not path.is_file():
                    return self._json(404, {"error": "no such object"})
                self.send_response(200)
                self.send_header("Content-Length", str(path.stat().st_size))
                self.end_headers()
                for block in read_blocks(path):
                    self.wfile.write(block)
                return None
            if kind == "objects" and method == "DELETE":
                srv.store.delete(arg)
                srv.ledger.release(arg)
                return self._json(200, {"deleted": arg})
            if kind == "stat" and method == "GET":
                s = srv.store.stat(arg, rehash="rehash=0" not in url.query)
                if s is None:
                    return self._json(404, {"error": "no such object"})
                return self._json(200, {"oid": arg, "length": s[0], "sha256": s[1], "durability": DURABILITY})
            if kind == "meta" and method == "PUT":
                srv.store.put_meta(arg, b"".join(self._body(length)))
                return self._json(200, {"stored": arg})
            if kind == "meta" and method == "DELETE":
                srv.store.delete_meta(arg)
                return self._json(200, {"deleted": arg})
            if kind == "meta" and method == "GET":
                if not arg:
                    return self._json(200, {"names": srv.store.list_meta()})
                data = srv.store.get_meta(arg)
                if data is None:
                    return self._json(404, {"error": "no such meta"})
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return None
            return self._json(405, {"error": "method not allowed"})
        except BackendError as e:
            self.close_connection = True
            return self._json(400, {"error": str(e)})

    def _put(self, oid: str, length: int) -> None:
        srv = self.server
        if srv.quota and not srv.ledger.has(oid) and srv.ledger.used() + length > srv.quota:
            self.close_connection = True
            return self._json(507, {"error": "quota exceeded"})
        sha = self.headers.get("X-Log3-Sha256", "")
        duplicate = srv.store.store(oid, self._body(length), length, sha)
        srv.ledger.charge(oid, length, self.headers.get("X-Log3-Op", ""))
        flag = srv.store.root / "fault-fired"
        if os.environ.get(FAULT_ENV) == "node_drop_response_once" and not flag.exists():
            flag.write_text("1")                               # тест: запис стійкий, відповідь загублено
            self.close_connection = True
            self.connection.close()
            return None
        return self._json(200, {"oid": oid, "length": length, "sha256": sha, "durability": DURABILITY,
                                "duplicate": duplicate})

    def do_GET(self):
        self._route("GET")

    def do_PUT(self):
        self._route("PUT")

    def do_DELETE(self):
        self._route("DELETE")


def server_tls(cert: Path, key: Path) -> ssl.SSLContext:
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    ctx.load_cert_chain(cert, key)
    return ctx


def make_node_cert(out_dir: Path, hosts: list[str], days: int = 825) -> tuple[Path, Path]:
    """Самопідписаний сертифікат вузла (ECDSA P-256).

    node.crt копіюють клієнтам як ca_file; node.key лишається на вузлі."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, hosts[0])])
    san = []
    for h in hosts:
        try:
            san.append(x509.IPAddress(ipaddress.ip_address(h)))
        except ValueError:
            san.append(x509.DNSName(h))
    now = datetime.now(UTC)
    ski = x509.SubjectKeyIdentifier.from_public_key(key.public_key())
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - timedelta(minutes=5)).not_valid_after(now + timedelta(days=days))
            .add_extension(x509.SubjectAlternativeName(san), critical=False)
            .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
            .add_extension(x509.KeyUsage(digital_signature=True, key_cert_sign=True, content_commitment=False,
                                         key_encipherment=False, data_encipherment=False, key_agreement=False,
                                         crl_sign=False, encipher_only=False, decipher_only=False), critical=True)
            .add_extension(ski, critical=False)
            .add_extension(x509.AuthorityKeyIdentifier.from_issuer_subject_key_identifier(ski), critical=False)
            .sign(key, hashes.SHA256()))
    out_dir.mkdir(parents=True, exist_ok=True)
    crt, pem = out_dir / "node.crt", out_dir / "node.key"
    crt.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    fd = os.open(pem, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                  serialization.NoEncryption()))
    return crt, pem
