"""Віддалений вузол зберігання: HTTPS-сервер над ObjectDir.

    python -m log3.vault node-cert --dir certs --host backup.example.org
    LOG3_NODE_TOKEN=... python -m log3.vault serve-node --data /srv/log3 --cert certs/node.crt --key certs/node.key

Захист: TLS >= 1.2 (клієнт перевіряє сертифікат вузла за закріпленим файлом), кожен запит
несе токен (порівняння за сталий час). Токен читається зі змінної середовища або файла.
Вузол бачить лише зашифровані об'єкти та відкриті службові файли без секретів.

Підтвердження PUT надсилається після стійкого запису файла й фіксації обліку квоти.
Паралельні записи резервують місце до передавання, а повтор того самого PUT не списує
квоту вдруге. Після перезапуску облік звіряється з наявними файлами об'єктів.

Доступ (--tokens-file, команда node-token): у файлі лише SHA-256 токенів, назви й ролі.
  reader  GET/stat/info/meta-list           — перевірка й restore
  writer  + PUT об'єктів і службових файлів  — реплікація й repair
  admin   + DELETE                          — gc
Writer не може видаляти, але може записати під тим самим oid інші байти (так repair замінює
пошкоджену копію); клієнт це виявляє verify і відновлює з іншої копії. Файл перечитується за
зміною mtime, тому відкликання (revoked) діє з наступного запиту без перезапуску вузла.
Без --tokens-file діє один токен зі змінної LOG3_NODE_TOKEN з роллю admin (сумісність етапу 1).
Ліміти: розмір об'єкта й службового файла, одночасні з'єднання, квота, блокування IP після
серії невдалих автентифікацій (429).
"""
from __future__ import annotations

import hashlib
import hmac
import ipaddress
import json
import os
import sqlite3
import ssl
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from ..errors import FAULT_ENV, BackendError
from ..fsutil import DURABILITY, IO_BLOCK, read_blocks
from .base import check_oid
from .localdir import ObjectDir

TOKEN_ENV = "LOG3_NODE_TOKEN"


class QuotaExceeded(Exception):
    """Для нового або збільшеного об'єкта бракує квоти."""


class Ledger:
    """Облік квоти вузла: один рядок на об'єкт, тому повтор не списує місце вдруге."""

    def __init__(self, path: Path):
        self.lock = threading.Condition()
        self.reserved: dict[str, int] = {}
        self.db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("CREATE TABLE IF NOT EXISTS charges(oid TEXT PRIMARY KEY, length INTEGER NOT NULL,"
                        " op_id TEXT, puts INTEGER NOT NULL DEFAULT 1)")

    def used(self) -> int:
        with self.lock:
            return self.db.execute("SELECT COALESCE(SUM(length),0) FROM charges").fetchone()[0]

    @contextmanager
    def reserve(self, oid: str, length: int, quota: int) -> Iterator[None]:
        """Серіалізувати зміни одного oid і врахувати приріст від усіх активних записів."""
        with self.lock:
            self.lock.wait_for(lambda: oid not in self.reserved)
            row = self.db.execute("SELECT length FROM charges WHERE oid=?", (oid,)).fetchone()
            extra = max(0, length - (row[0] if row else 0))
            if quota and extra and self.used() + sum(self.reserved.values()) + extra > quota:
                raise QuotaExceeded
            self.reserved[oid] = extra
        try:
            yield
        finally:
            with self.lock:
                del self.reserved[oid]
                self.lock.notify_all()

    def charge(self, oid: str, length: int, op_id: str) -> None:
        with self.lock:
            self.db.execute("INSERT INTO charges(oid,length,op_id) VALUES(?,?,?) "
                            "ON CONFLICT(oid) DO UPDATE SET length=excluded.length, puts=puts+1", (oid, length, op_id))
            if oid in self.reserved:
                self.reserved[oid] = 0

    def release(self, oid: str) -> None:
        with self.lock:
            self.db.execute("DELETE FROM charges WHERE oid=?", (oid,))

    def reconcile_objects(self, objects: Path) -> None:
        """Відновити облік, якщо процес зупинився між зміною файла й записом у SQLite."""
        sizes = {}
        for path in objects.glob("*/*"):
            try:
                oid = check_oid(path.name)
            except BackendError:
                continue
            if path.is_file() and path.parent.name == oid[:2]:
                sizes[oid] = path.stat().st_size
        with self.lock, self.db:
            self.db.execute("BEGIN IMMEDIATE")
            charged = dict(self.db.execute("SELECT oid, length FROM charges"))
            self.db.executemany("DELETE FROM charges WHERE oid=?", ((oid,) for oid in charged.keys() - sizes.keys()))
            self.db.executemany("INSERT INTO charges(oid,length) VALUES(?,?) "
                                "ON CONFLICT(oid) DO UPDATE SET length=excluded.length",
                                ((oid, size) for oid, size in sizes.items() if size != charged.get(oid)))


ROLES = {"reader": 0, "writer": 1, "admin": 2}
NEED = {("GET", "info"): "reader", ("GET", "objects"): "reader", ("GET", "stat"): "reader",
        ("GET", "meta"): "reader", ("PUT", "objects"): "writer", ("PUT", "meta"): "writer",
        ("DELETE", "objects"): "admin", ("DELETE", "meta"): "admin"}
LIMITS = {"max_object_bytes": 256 << 20, "max_meta_bytes": 1 << 20, "max_connections": 32,
          "auth_fail_limit": 10, "auth_fail_window_seconds": 60.0, "auth_block_seconds": 60.0}
TOKENS_FORMAT = "log3-node-tokens"


def token_hash(token: bytes) -> str:
    return hashlib.sha256(token).hexdigest()


class Tokens:
    """Токени вузла: файл (ролі, відкликання) або один токен зі змінної середовища (admin)."""

    def __init__(self, path: Path | None, env_token: bytes | None):
        self.path, self.mtime, self.lock = path, None, threading.Lock()
        self.entries = [] if path else [{"name": "env", "sha256": token_hash(env_token or b""), "role": "admin"}]
        if path:
            self._reload()

    def _reload(self) -> None:
        mtime = self.path.stat().st_mtime_ns
        if mtime != self.mtime:
            doc = json.loads(self.path.read_text(encoding="utf-8"))
            if doc.get("format") != TOKENS_FORMAT:
                raise BackendError("файл токенів має невідомий формат")
            self.entries = [t for t in doc["tokens"] if not t.get("revoked") and t.get("role") in ROLES]
            self.mtime = mtime

    def check(self, token: bytes) -> dict | None:
        """Запис токена або None. Дайджести порівнюються за сталий час з усіма записами."""
        with self.lock:
            if self.path:
                self._reload()
            got, hit = token_hash(token), None
            for t in self.entries:
                if hmac.compare_digest(got, t["sha256"]) and token:
                    hit = t
            return hit


class NodeServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, addr, data: Path, token: bytes | None, tls: ssl.SSLContext, quota_bytes: int = 0, *,
                 tokens_file: Path | None = None, limits: dict | None = None):
        super().__init__(addr, _Handler)
        self.store, self.tls, self.quota = ObjectDir(data), tls, quota_bytes
        self.tokens = Tokens(tokens_file, token)
        self.limits = {**LIMITS, **(limits or {})}
        self.slots = threading.BoundedSemaphore(self.limits["max_connections"])
        self.failures: dict[str, list[float]] = {}
        self.fail_lock = threading.Lock()
        self.store.create()
        self.ledger = Ledger(data / "node-ledger.sqlite")
        self.ledger.reconcile_objects(data / "objects")

    def get_request(self):
        sock, addr = self.socket.accept()
        return self.tls.wrap_socket(sock, server_side=True, do_handshake_on_connect=False), addr

    def process_request(self, request, client_address):
        if not self.slots.acquire(blocking=False):          # межа одночасних з'єднань
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self.slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.slots.release()

    def blocked(self, ip: str) -> bool:
        lim, now = self.limits, time.monotonic()
        with self.fail_lock:
            recent = [t for t in self.failures.get(ip, []) if now - t < lim["auth_block_seconds"]
                      + lim["auth_fail_window_seconds"]]
            self.failures[ip] = recent
            return sum(now - t < lim["auth_fail_window_seconds"] for t in recent) >= lim["auth_fail_limit"]

    def auth_failed(self, ip: str) -> None:
        with self.fail_lock:
            self.failures.setdefault(ip, []).append(time.monotonic())


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: NodeServer

    def setup(self):
        self.request.settimeout(120)
        self.request.do_handshake()                            # рукостискання в потоці запиту, а не в accept
        super().setup()

    def log_message(self, fmt, *args):                         # заголовки (і токен) у журнал не потрапляють
        if os.environ.get("LOG3_NODE_QUIET") != "1":
            super().log_message("%s " + fmt, getattr(self, "token_name", "-"), *args)

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
        srv, ip = self.server, self.client_address[0]
        if srv.blocked(ip):
            self.close_connection = True
            return self._json(429, {"error": "too many failed authentications"})
        entry = srv.tokens.check(self.headers.get("Authorization", "").removeprefix("Bearer ").encode())
        if entry is None:
            srv.auth_failed(ip)
            self.close_connection = True
            return self._json(401, {"error": "unauthorized"})
        self.token_name = entry["name"]
        url = urlsplit(self.path)
        parts = url.path.strip("/").split("/")
        try:
            if parts[:1] != ["v1"] or not 2 <= len(parts) <= 3:
                return self._json(404, {"error": "not found"})
            kind, arg = parts[1], parts[2] if len(parts) == 3 else ""
            need = NEED.get((method, kind))
            if need is None:
                return self._json(405, {"error": "method not allowed"})
            if ROLES[entry["role"]] < ROLES[need]:
                return self._json(403, {"error": f"role {entry['role']} may not {method} {kind}"})
            length = int(self.headers.get("Content-Length") or -1) if method == "PUT" else 0
            cap = srv.limits["max_object_bytes" if kind == "objects" else "max_meta_bytes"]
            if method == "PUT" and not 0 <= length <= cap:
                self.close_connection = True
                return self._json(413 if length > cap else 411, {"error": f"Content-Length must be 0..{cap}"})
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
                with srv.ledger.reserve(check_oid(arg), 0, quota=0):
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
        except (BackendError, ValueError) as e:
            self.close_connection = True
            return self._json(400, {"error": str(e)})

    def _put(self, oid: str, length: int) -> None:
        srv = self.server
        sha = self.headers.get("X-Log3-Sha256", "")
        try:
            with srv.ledger.reserve(check_oid(oid), length, srv.quota):
                duplicate = srv.store.store(oid, self._body(length), length, sha)
                srv.ledger.charge(oid, length, self.headers.get("X-Log3-Op", ""))
        except QuotaExceeded:
            self.close_connection = True
            return self._json(507, {"error": "quota exceeded"})
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
