"""Локальний каталог і постійна черга в SQLite.

Налаштування з'єднання задано явно, а не «увімкнено WAL і забули»:
  journal_mode=WAL + synchronous=FULL  кожен COMMIT викликає fsync журналу WAL, тобто зафіксована
                                       транзакція переживає аварію процесу і (якщо ОС та диск чесно
                                       виконують fsync) втрату живлення. WAL сам по собі такої
                                       гарантії не дає: із synchronous=NORMAL останні транзакції
                                       після втрати живлення можуть зникнути.
  BEGIN IMMEDIATE                      кожна зміна — явна транзакція; блокування запису береться одразу.
  wal_checkpoint(TRUNCATE)             після публікації версії та наприкінці роботи черги.
  відновлення                          незавершена транзакція відкочується SQLite при наступному
                                       відкритті; integrity_check запускає команда repair.
Від фізичної втрати диска каталог не захищає: для цього є копії на backend-ах і команда recover.
"""
from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

from .errors import Exit, VaultError
from .states import check_transition

SCHEMA_VERSION = 1
SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS files(
  file_id TEXT PRIMARY KEY, root TEXT NOT NULL, path TEXT NOT NULL, dev INTEGER, ino INTEGER,
  UNIQUE(root, path));
CREATE INDEX IF NOT EXISTS files_inode ON files(root, dev, ino);
CREATE TABLE IF NOT EXISTS objects(
  oid TEXT PRIMARY KEY, kind INTEGER NOT NULL, stored_len INTEGER NOT NULL, sha256 TEXT NOT NULL,
  plain_len INTEGER NOT NULL, codec INTEGER NOT NULL, in_spool INTEGER NOT NULL, created_ns INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS versions(
  version_id TEXT PRIMARY KEY, op_id TEXT NOT NULL, name TEXT NOT NULL, root TEXT NOT NULL, state TEXT NOT NULL,
  priority INTEGER NOT NULL, min_copies INTEGER NOT NULL, target_copies INTEGER NOT NULL, manifest_oid TEXT,
  created_ns INTEGER NOT NULL, heartbeat_ns INTEGER NOT NULL, accepted_ns INTEGER, protected_ns INTEGER,
  retention_until_ns INTEGER, files INTEGER DEFAULT 0, bytes INTEGER DEFAULT 0, skipped INTEGER DEFAULT 0,
  error TEXT);
CREATE INDEX IF NOT EXISTS versions_op ON versions(op_id);
CREATE TABLE IF NOT EXISTS version_objects(
  version_id TEXT NOT NULL, oid TEXT NOT NULL, PRIMARY KEY(version_id, oid)) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS version_objects_oid ON version_objects(oid);
CREATE TABLE IF NOT EXISTS replicas(
  oid TEXT NOT NULL, backend TEXT NOT NULL, state TEXT NOT NULL, ack_len INTEGER, ack_sha TEXT,
  acked_ns INTEGER, verified_ns INTEGER, PRIMARY KEY(oid, backend)) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS jobs(
  job_id TEXT PRIMARY KEY, oid TEXT NOT NULL, backend TEXT NOT NULL, state TEXT NOT NULL,
  priority INTEGER NOT NULL, attempts INTEGER NOT NULL DEFAULT 0, next_at_ns INTEGER NOT NULL,
  lease_until_ns INTEGER, enq_ns INTEGER NOT NULL, done_ns INTEGER, last_error TEXT,
  UNIQUE(oid, backend));
CREATE INDEX IF NOT EXISTS jobs_pick ON jobs(state, priority, next_at_ns);
CREATE TABLE IF NOT EXISTS events(
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts_ns INTEGER NOT NULL, kind TEXT NOT NULL, value REAL, detail TEXT);
CREATE INDEX IF NOT EXISTS events_kind ON events(kind, id);
"""


def now_ns() -> int:
    return time.time_ns()


class Catalog:
    def __init__(self, path: Path):
        self.db = sqlite3.connect(path, isolation_level=None, timeout=60)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("PRAGMA busy_timeout=60000")
        self.db.execute("PRAGMA wal_autocheckpoint=1000")
        self._depth = 0
        if not self.one("SELECT name FROM sqlite_master WHERE name='events_kind'"):
            self.db.executescript(SCHEMA)                       # усі оператори IF NOT EXISTS: повтор безпечний
        with self.tx():
            got = self.one("SELECT value FROM meta WHERE key='schema_version'")
            if got is None:
                self.db.execute("INSERT INTO meta VALUES('schema_version', ?)", (str(SCHEMA_VERSION),))
            elif int(got[0]) != SCHEMA_VERSION:
                raise VaultError("E_SCHEMA", f"схема каталогу {got[0]}, підтримується {SCHEMA_VERSION}")

    @contextmanager
    def tx(self):
        """Явна транзакція запису. Вкладені виклики приєднуються до зовнішньої."""
        if self._depth:
            self._depth += 1
            try:
                yield self
            finally:
                self._depth -= 1
            return
        self.db.execute("BEGIN IMMEDIATE")
        self._depth = 1
        try:
            yield self
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        else:
            self.db.execute("COMMIT")
        finally:
            self._depth = 0

    def q(self, sql: str, args: tuple = ()) -> list[sqlite3.Row]:
        return self.db.execute(sql, args).fetchall()

    def one(self, sql: str, args: tuple = ()) -> sqlite3.Row | None:
        return self.db.execute(sql, args).fetchone()

    def scalar(self, sql: str, args: tuple = ()):
        row = self.db.execute(sql, args).fetchone()
        return row[0] if row else None

    def x(self, sql: str, args: tuple = ()) -> int:
        return self.db.execute(sql, args).rowcount

    def checkpoint(self) -> None:
        self.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    def integrity_ok(self) -> bool:
        return self.scalar("PRAGMA integrity_check") == "ok"

    def event(self, kind: str, value: float | None = None, detail: str = "") -> None:
        self.x("INSERT INTO events(ts_ns, kind, value, detail) VALUES(?,?,?,?)", (now_ns(), kind, value, detail))

    def set_meta(self, key: str, value: str) -> None:
        self.x("INSERT INTO meta VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))

    def set_state(self, version_id: str, new: str, **fields) -> None:
        """Єдине місце зміни стану версії; заборонений перехід -> VaultError."""
        with self.tx():
            old = self.scalar("SELECT state FROM versions WHERE version_id=?", (version_id,))
            check_transition(old, new)
            cols = "".join(f", {k}=?" for k in fields)
            self.x(f"UPDATE versions SET state=?{cols} WHERE version_id=?", (str(new), *fields.values(), version_id))
            self.event("state", None, f"{version_id} {old}->{new}")

    def find_version(self, ref: str) -> sqlite3.Row:
        """Версія за повним ідентифікатором, однозначним префіксом або словом latest."""
        if ref == "latest":
            rows = self.q("SELECT * FROM versions WHERE state NOT IN ('FAILED','DELETED','CAPTURING') "
                          "ORDER BY created_ns DESC LIMIT 1")
        else:
            rows = self.q("SELECT * FROM versions WHERE version_id LIKE ? ESCAPE '\\' LIMIT 2",
                          (ref.replace("\\", "").replace("%", "").replace("_", "") + "%",))
        if len(rows) != 1:
            raise VaultError("E_NOT_FOUND", f"версію {ref!r} не знайдено або префікс неоднозначний", Exit.NOT_FOUND)
        return rows[0]
