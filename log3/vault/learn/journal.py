"""Постійний журнал рішень і спостережень навчання: <repo>/learn/journal.sqlite.

Окрема база, а не таблиці каталогу: схема каталогу v1 етапу 1 не змінюється, а збій чи
видалення журналу не зачіпає жодної гарантії захисту даних. Журнал не секретний, але й не
містить назв, шляхів чи байтів файлів: лише знеособлені ключі (object_id — HMAC, group_key —
hash від file_id), числові ознаки й виміряні витрати.

Таблиці:
  decisions   одне рішення про блок: operation/decision ID, час рішення, версія схеми ознак,
              модель і режим, доступні ознаки (вектор + маска), дія політики, дія моделі,
              обрана дія, shadow-рекомендація кандидата (на рішення не впливає), імовірність проби
  outcomes    фактичний результат кодування обраною дією: байти, час, CPU; час результату
  probes      пробні вимірювання інших допустимих дій на тому самому блоці (повна інформація)
  stages      стадії захисту, зібрані з каталогу пізніше; кожна — окремий факт із власним часом:
                encoded            блок закодовано (є outcome)
                queued             версію прийнято локально, завдання реплікації поставлено
                write_confirmed    перша перевірена квитанція backend-а для об'єкта
                policy_satisfied   версія досягла PROTECTED (політику копій виконано)
                read_verified      verify перечитав копію після запису і вона ціла
                verify_failed      verify знайшов копію відсутньою або пошкодженою
              «Відправлено», мережевий timeout чи помилка завдання стадією НЕ є: вони не
              доводять ні запису, ні втрати. Відсутність restore нічого не каже про важливість.
  lineage     знеособлена історія лінії файла: скільки версій бачили і скільки разів змінювався вміст
  operations  підсумок операції backup для навчання: витрати ознак, проб і висновку, drift
  lifecycle   події навчання: train, evaluate, shadow, promote, rollback, freeze, reset, помилки
  counters    дублікати, відсутні й затримані результати, помилки журналу

Журнал працює з synchronous=NORMAL: на відміну від каталогу, втрата останніх записів журналу
після втрати живлення допустима й не впливає на копії.
"""
from __future__ import annotations

import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from .features import N_FEATURES

JOURNAL_VERSION = 1
NAMESPACE = uuid.UUID("6f1c3a52-6a4e-4b8e-9d0e-2b6b3f5c7a11")
STAGES = ("encoded", "queued", "write_confirmed", "policy_satisfied", "read_verified", "verify_failed")
SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS decisions(
  decision_id TEXT PRIMARY KEY, operation_id TEXT NOT NULL, version_id TEXT NOT NULL, object_id TEXT NOT NULL,
  chunk_index INTEGER NOT NULL, group_key TEXT NOT NULL, decided_ns INTEGER NOT NULL, feature_schema TEXT NOT NULL,
  features BLOB NOT NULL, mask BLOB NOT NULL, feature_ns INTEGER NOT NULL, inference_ns INTEGER NOT NULL,
  mode TEXT NOT NULL, model_id TEXT, policy_action TEXT NOT NULL, model_action TEXT, chosen_action TEXT NOT NULL,
  shadow_model TEXT, shadow_action TEXT, fallback TEXT, probe_p REAL NOT NULL,
  quarantine INTEGER NOT NULL DEFAULT 0, published INTEGER NOT NULL DEFAULT 1);
CREATE INDEX IF NOT EXISTS decisions_time ON decisions(decided_ns);
CREATE INDEX IF NOT EXISTS decisions_object ON decisions(object_id);
CREATE INDEX IF NOT EXISTS decisions_version ON decisions(version_id);
CREATE TABLE IF NOT EXISTS outcomes(
  decision_id TEXT PRIMARY KEY, result_ns INTEGER NOT NULL, action TEXT NOT NULL, codec_used TEXT NOT NULL,
  raw_len INTEGER NOT NULL, payload_len INTEGER NOT NULL, stored_len INTEGER NOT NULL, wall_ns INTEGER NOT NULL,
  cpu_ns INTEGER NOT NULL, forced_store INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS probes(
  decision_id TEXT NOT NULL, action TEXT NOT NULL, measured_ns INTEGER NOT NULL, codec_used TEXT NOT NULL,
  payload_len INTEGER NOT NULL, wall_ns INTEGER NOT NULL, cpu_ns INTEGER NOT NULL,
  PRIMARY KEY(decision_id, action)) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS stages(
  subject TEXT NOT NULL, stage TEXT NOT NULL, ts_ns INTEGER NOT NULL, value REAL,
  PRIMARY KEY(subject, stage)) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS lineage(
  group_key TEXT PRIMARY KEY, versions_seen INTEGER NOT NULL, content_changes INTEGER NOT NULL,
  last_content TEXT NOT NULL, last_seen_ns INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS operations(
  version_id TEXT PRIMARY KEY, operation_id TEXT NOT NULL, started_ns INTEGER NOT NULL, finished_ns INTEGER NOT NULL,
  mode TEXT NOT NULL, model_id TEXT, published INTEGER NOT NULL, chunks INTEGER NOT NULL, probes INTEGER NOT NULL,
  probe_cpu_ns INTEGER NOT NULL, feature_ns INTEGER NOT NULL, inference_ns INTEGER NOT NULL,
  encode_cpu_ns INTEGER NOT NULL,
  raw_bytes INTEGER NOT NULL, stored_bytes INTEGER NOT NULL, fallbacks INTEGER NOT NULL, quarantine INTEGER NOT NULL,
  detail TEXT);
CREATE TABLE IF NOT EXISTS lifecycle(
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts_ns INTEGER NOT NULL, kind TEXT NOT NULL, model_id TEXT, detail TEXT);
CREATE TABLE IF NOT EXISTS counters(key TEXT PRIMARY KEY, value INTEGER NOT NULL);
"""


def now_ns() -> int:
    return time.time_ns()


def decision_id(version_id: str, object_id: str) -> str:
    """Детермінований: повторна доставка того самого рішення не створює другого рядка."""
    return str(uuid.uuid5(NAMESPACE, f"{version_id}:{object_id}"))


def pack(values: np.ndarray) -> bytes:
    return np.asarray(values, dtype=np.float32).tobytes()


def unpack(blob: bytes, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(blob, dtype=dtype)


class Journal:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.db = sqlite3.connect(path, isolation_level=None, timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.execute("PRAGMA busy_timeout=30000")
        self.db.executescript(SCHEMA)
        got = self.db.execute("SELECT value FROM meta WHERE key='journal_version'").fetchone()
        if got is None:
            self.db.execute("INSERT OR IGNORE INTO meta VALUES('journal_version', ?)", (str(JOURNAL_VERSION),))
        elif int(got[0]) != JOURNAL_VERSION:
            raise sqlite3.DatabaseError(f"журнал версії {got[0]}, підтримується {JOURNAL_VERSION}")

    def close(self) -> None:
        self.db.close()

    @contextmanager
    def tx(self):
        self.db.execute("BEGIN IMMEDIATE")
        try:
            yield self
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        self.db.execute("COMMIT")

    def q(self, sql: str, args: tuple = ()) -> list[sqlite3.Row]:
        return self.db.execute(sql, args).fetchall()

    def scalar(self, sql: str, args: tuple = ()):
        row = self.db.execute(sql, args).fetchone()
        return row[0] if row else None

    def bump(self, key: str, n: int = 1) -> None:
        if n:
            self.db.execute("INSERT INTO counters VALUES(?,?) ON CONFLICT(key) DO UPDATE SET "
                            "value=value+excluded.value", (key, n))

    def counters(self) -> dict[str, int]:
        return {r["key"]: r["value"] for r in self.q("SELECT * FROM counters ORDER BY key")}

    def lifecycle(self, kind: str, model_id: str | None = None, detail: str = "") -> None:
        self.db.execute("INSERT INTO lifecycle(ts_ns, kind, model_id, detail) VALUES(?,?,?,?)",
                        (now_ns(), kind, model_id, detail))

    # ---- запис рішень і результатів -------------------------------------------------------
    def write_decisions(self, decisions: list[dict], outcomes: list[dict], probes: list[dict]) -> dict:
        """Пакетний запис після рішення. Дублікати (той самий decision_id) не перезаписуються й
        рахуються; результат без рішення приймається (рішення може прийти пізніше)."""
        dup = 0
        with self.tx():
            for d in decisions:
                cur = self.db.execute(
                    "INSERT OR IGNORE INTO decisions(decision_id, operation_id, version_id, object_id, chunk_index, "
                    "group_key, decided_ns, feature_schema, features, mask, feature_ns, inference_ns, mode, model_id, "
                    "policy_action, model_action, chosen_action, shadow_model, shadow_action, fallback, probe_p, "
                    "quarantine, published) "
                    "VALUES(:decision_id,:operation_id,:version_id,:object_id,:chunk_index,:group_key,:decided_ns,"
                    ":feature_schema,:features,:mask,:feature_ns,:inference_ns,:mode,:model_id,:policy_action,"
                    ":model_action,:chosen_action,:shadow_model,:shadow_action,:fallback,:probe_p,:quarantine,"
                    ":published)", d)
                dup += cur.rowcount == 0
            for o in outcomes:
                cur = self.db.execute(
                    "INSERT OR IGNORE INTO outcomes VALUES(:decision_id,:result_ns,:action,:codec_used,:raw_len,"
                    ":payload_len,:stored_len,:wall_ns,:cpu_ns,:forced_store)", o)
                dup += cur.rowcount == 0
                if cur.rowcount:
                    self.db.execute("INSERT OR IGNORE INTO stages VALUES(?, 'encoded', ?, ?)",
                                    (self._object_of(o["decision_id"]) or o["decision_id"], o["result_ns"],
                                     o["stored_len"]))
            for p in probes:
                self.db.execute("INSERT OR IGNORE INTO probes VALUES(:decision_id,:action,:measured_ns,:codec_used,"
                                ":payload_len,:wall_ns,:cpu_ns)", p)
            self.bump("duplicate_records", dup)
        return {"duplicates": dup}

    def _object_of(self, did: str) -> str | None:
        return self.scalar("SELECT object_id FROM decisions WHERE decision_id=?", (did,))

    def write_operation(self, op: dict) -> None:
        with self.tx():
            self.db.execute(
                "INSERT OR REPLACE INTO operations VALUES(:version_id,:operation_id,:started_ns,:finished_ns,:mode,"
                ":model_id,:published,:chunks,:probes,:probe_cpu_ns,:feature_ns,:inference_ns,:encode_cpu_ns,"
                ":raw_bytes,:stored_bytes,:fallbacks,:quarantine,:detail)", op)

    def lineage(self, group_key: str) -> tuple[int, int] | None:
        row = self.db.execute("SELECT versions_seen, content_changes FROM lineage WHERE group_key=?",
                              (group_key,)).fetchone()
        return (row[0], row[1]) if row else None

    def update_lineage(self, items: list[tuple[str, str]], ts_ns: int) -> None:
        """items: (group_key, коротка мітка вмісту). Викликається ПІСЛЯ рішень версії, тож ознаки
        лінії в рішеннях завжди відображають лише попередні версії."""
        with self.tx():
            for key, content in items:
                self.db.execute(
                    "INSERT INTO lineage VALUES(?,1,0,?,?) ON CONFLICT(group_key) DO UPDATE SET "
                    "versions_seen=versions_seen+1, "
                    "content_changes=content_changes+(last_content!=excluded.last_content), "
                    "last_content=excluded.last_content, last_seen_ns=excluded.last_seen_ns", (key, content, ts_ns))

    def prune(self, max_decisions: int) -> int:
        """Обмежує розмір журналу: найстаріші рішення з їхніми результатами й пробами видаляються."""
        n = self.scalar("SELECT COUNT(*) FROM decisions") or 0
        if n <= max_decisions:
            return 0
        cut = self.scalar("SELECT decided_ns FROM decisions ORDER BY decided_ns LIMIT 1 OFFSET ?",
                          (n - max_decisions,))
        with self.tx():
            for table in ("outcomes", "probes"):
                self.db.execute(f"DELETE FROM {table} WHERE decision_id IN "
                                "(SELECT decision_id FROM decisions WHERE decided_ns<?)", (cut,))
            removed = self.db.execute("DELETE FROM decisions WHERE decided_ns<?", (cut,)).rowcount
            self.bump("pruned_decisions", removed)
        return removed

    # ---- стадії захисту з каталогу (можуть приходити із затримкою) --------------------------
    def harvest(self, cat, *, limit: int = 20000) -> dict:
        """Ідемпотентно переносить факти каталогу в stages. Читає каталог лише для об'єктів і
        версій, про які є рішення; повторний виклик нічого не дублює."""
        out = dict.fromkeys(STAGES[1:], 0)
        versions = [r[0] for r in self.q(
            "SELECT DISTINCT d.version_id FROM decisions d WHERE NOT EXISTS(SELECT 1 FROM stages s "
            "WHERE s.subject=d.version_id AND s.stage='policy_satisfied') ORDER BY d.decided_ns DESC LIMIT ?",
            (limit,))]
        rows = []
        for vid in versions:
            v = cat.one("SELECT accepted_ns, protected_ns FROM versions WHERE version_id=?", (vid,))
            if v is None:
                continue
            if v["accepted_ns"]:
                rows.append((vid, "queued", v["accepted_ns"], None))
            if v["protected_ns"]:
                rows.append((vid, "policy_satisfied", v["protected_ns"], (v["protected_ns"] - v["accepted_ns"]) / 1e9))
        objects = [r[0] for r in self.q(
            "SELECT d.object_id FROM decisions d WHERE NOT EXISTS(SELECT 1 FROM stages s WHERE s.subject=d.object_id "
            "AND s.stage IN ('read_verified','verify_failed')) ORDER BY d.decided_ns DESC LIMIT ?", (limit,))]
        for oid in objects:
            acks = cat.q("SELECT state, acked_ns, verified_ns FROM replicas WHERE oid=?", (oid,))
            confirmed = [a for a in acks if a["acked_ns"] is not None]
            if confirmed:
                rows.append((oid, "write_confirmed", min(a["acked_ns"] for a in confirmed), len(confirmed)))
            verified = [a for a in acks if a["state"] == "CONFIRMED" and a["verified_ns"] and a["acked_ns"]
                        and a["verified_ns"] > a["acked_ns"]]
            if verified:
                rows.append((oid, "read_verified", min(a["verified_ns"] for a in verified), len(verified)))
            bad = [a for a in acks if a["state"] in ("MISSING", "CORRUPT") and a["verified_ns"]]
            if bad:
                rows.append((oid, "verify_failed", min(a["verified_ns"] for a in bad), len(bad)))
        with self.tx():
            for r in rows:
                if self.db.execute("INSERT OR IGNORE INTO stages VALUES(?,?,?,?)", r).rowcount:
                    out[r[1]] += 1
        return out

    def stage_counts(self) -> dict[str, int]:
        got = dict(self.q("SELECT stage, COUNT(*) FROM stages GROUP BY stage"))
        return {s: got.get(s, 0) for s in STAGES}

    def completeness(self) -> dict:
        """Рішення без результату (відсутні/затримані) і результати без рішення."""
        return {
            "decisions": self.scalar("SELECT COUNT(*) FROM decisions"),
            "outcomes": self.scalar("SELECT COUNT(*) FROM outcomes"),
            "decisions_without_outcome": self.scalar(
                "SELECT COUNT(*) FROM decisions d WHERE NOT EXISTS(SELECT 1 FROM outcomes o "
                "WHERE o.decision_id=d.decision_id)"),
            "outcomes_without_decision": self.scalar(
                "SELECT COUNT(*) FROM outcomes o WHERE NOT EXISTS(SELECT 1 FROM decisions d "
                "WHERE d.decision_id=o.decision_id)"),
            "probed_decisions": self.scalar("SELECT COUNT(DISTINCT decision_id) FROM probes"),
            "quarantined_decisions": self.scalar("SELECT COUNT(*) FROM decisions WHERE quarantine=1"),
        }

    # ---- навчальна вибірка ------------------------------------------------------------------
    def labeled(self, *, feature_schema: str, actions: list[str], until_ns: int | None = None,
                since_ns: int | None = None, include_quarantine: bool = False, max_rows: int = 50000) -> dict:
        """Блоки з повною інформацією: для кожної допустимої дії є виміряний результат (проба або
        фактичний outcome). Лише такі приклади не залежать від того, яку дію обрала попередня
        політика (проби вибираються за hash(object_id), незалежно від ознак і рішення)."""
        q = ("SELECT d.decision_id, d.group_key, d.decided_ns, d.features, d.mask, d.chosen_action, d.probe_p, "
             "d.policy_action, d.shadow_action, d.shadow_model, d.feature_ns, "
             "o.raw_len, o.action, o.payload_len, o.cpu_ns FROM decisions d JOIN outcomes o ON "
             "o.decision_id=d.decision_id WHERE d.feature_schema=? AND d.probe_p>0 "
             + ("" if include_quarantine else "AND d.quarantine=0 ")
             + ("AND d.decided_ns<=? " if until_ns else "")
             + ("AND d.decided_ns>? " if since_ns else "")
             + "AND EXISTS(SELECT 1 FROM probes p WHERE p.decision_id=d.decision_id) "
             "ORDER BY d.decided_ns DESC LIMIT ?")
        args = (feature_schema, *((until_ns,) if until_ns else ()), *((since_ns,) if since_ns else ()), max_rows)
        rows = self.q(q, args)[::-1]
        k = len(actions)
        X = np.zeros((len(rows), N_FEATURES), dtype=np.float32)
        M = np.zeros((len(rows), N_FEATURES), dtype=bool)
        pay = np.full((len(rows), k), np.nan)
        cpu = np.full((len(rows), k), np.nan)
        raw = np.zeros(len(rows))
        keep, groups, times, chosen, extra = [], [], [], [], []
        col = {a: i for i, a in enumerate(actions)}
        for i, r in enumerate(rows):
            X[i] = unpack(r["features"])
            M[i] = unpack(r["mask"], np.bool_)
            raw[i] = r["raw_len"]
            if r["action"] in col:
                pay[i, col[r["action"]]], cpu[i, col[r["action"]]] = r["payload_len"], r["cpu_ns"]
            for p in self.q("SELECT action, payload_len, cpu_ns FROM probes WHERE decision_id=?", (r["decision_id"],)):
                if p["action"] in col:
                    pay[i, col[p["action"]]], cpu[i, col[p["action"]]] = p["payload_len"], p["cpu_ns"]
            if not np.isnan(pay[i]).any():
                keep.append(i)
            groups.append(r["group_key"])
            times.append(r["decided_ns"])
            chosen.append(r["chosen_action"])
            extra.append((r["policy_action"], r["shadow_action"], r["shadow_model"], r["feature_ns"]))
        idx = np.array(keep, dtype=int)
        return {"X": X[idx], "mask": M[idx], "payload": pay[idx], "cpu_ns": cpu[idx], "raw": raw[idx],
                "groups": [groups[i] for i in keep], "decided_ns": np.array([times[i] for i in keep], dtype=np.int64),
                "chosen": [chosen[i] for i in keep], "policy": [extra[i][0] for i in keep],
                "shadow": [extra[i][1] for i in keep], "shadow_model": [extra[i][2] for i in keep],
                "feature_ns": np.array([extra[i][3] for i in keep], dtype=float), "actions": list(actions),
                "incomplete": len(rows) - len(keep)}
