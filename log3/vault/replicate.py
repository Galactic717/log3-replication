"""Постійна черга реплікації: щонайменше одна доставка + ідемпотентне приймання.

Завдання = (об'єкт, backend). Його job_id — uuid5 від сховища, об'єкта і backend: він однаковий
у всіх повторах і після перезапуску, тому загублена відповідь не породжує ні другої копії,
ні другого посилання, ні другого списання квоти. Дедуплікація спільна для версій: однаковий
блок у двох версіях — одне завдання й одна копія на backend.

Порядок вибору — зважений round-robin за пріоритетом (4 : 2 : 1 на пакет), тому нескінченний
потік високого пріоритету не може назавжди заблокувати решту. Пакет обмежений, отже час і
пам'ять планування не залежать від довжини черги (це не ДП навчального тракту).

Невдала спроба: attempts += 1, повтор через min(cap, base * 2^(attempts-1)) * U(0.5, 1.5).
Коли attempts досягає max_attempts, завдання переходить в ATTENTION, а версія — в DEGRADED;
прийняті дані лишаються в spool і доступні для restore.
"""
from __future__ import annotations

import json
import random
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from .backends.base import Backend, check_ack
from .catalog import now_ns
from .cryptobox import KIND_MANIFEST
from .errors import BackendError, crashpoint
from .repo import Repo
from .states import JobState, ReplicaState, VersionState

WEIGHTS = ((2, 4), (1, 2), (0, 1))      # (пріоритет, частка пакета)
BATCH_UNIT = 8
LIVE = ("ACCEPTED_LOCAL", "REPLICATING", "PROTECTED", "DEGRADED")
CONFIG_META = "repo-config.json"


def pointer_name(version_id: str) -> str:
    return f"version-{version_id}.json"


def pointer_doc(version_id: str, manifest_oid: str, length: int, sha256: str, created_ns: int) -> dict:
    """Покажчик версії на backend-і. Підписаний (keys.sign_meta) під час публікації версії;
    версії, прийняті до етапу 3, мають непідписаний, доки key publish його не підпише."""
    return {"format": "log3-version-pointer", "format_version": 1, "version_id": version_id,
            "manifest_oid": manifest_oid, "manifest_len": length, "manifest_sha256": sha256,
            "created_at_ns": created_ns}


def job_id(repo: Repo, oid: str, backend: str) -> str:
    return str(uuid.uuid5(uuid.UUID(repo.cfg["repo_id"]), f"{oid}:{backend}"))


def ensure_jobs(repo: Repo, version_id: str, backends: list[str], priority: int) -> int:
    """Ставить завдання для об'єктів версії без підтвердженої копії на вказаних backend-ах.
    Наявне завдання не дублюється; ATTENTION і DONE (копію втрачено) повертаються в PENDING."""
    cat, now, n = repo.cat, now_ns(), 0
    with cat.tx():
        for backend in backends:
            rows = cat.q("SELECT vo.oid FROM version_objects vo LEFT JOIN replicas r ON r.oid=vo.oid AND r.backend=? "
                         "AND r.state='CONFIRMED' WHERE vo.version_id=? AND r.oid IS NULL", (backend, version_id))
            for (oid,) in rows:
                n += cat.x(
                    "INSERT INTO jobs(job_id, oid, backend, state, priority, next_at_ns, enq_ns) VALUES(?,?,?,?,?,?,?) "
                    "ON CONFLICT(oid, backend) DO UPDATE SET priority=max(priority, excluded.priority), "
                    "attempts=CASE WHEN state IN ('ATTENTION','DONE') THEN 0 ELSE attempts END, "
                    "next_at_ns=CASE WHEN state IN ('ATTENTION','DONE') THEN excluded.next_at_ns ELSE next_at_ns END, "
                    "state=CASE WHEN state IN ('ATTENTION','DONE') THEN 'PENDING' ELSE state END",
                    (job_id(repo, oid, backend), oid, backend, JobState.PENDING, priority, now, now))
    return n


def protection(repo: Repo, version_id: str, min_copies: int) -> dict:
    """Скільки об'єктів версії мають менше ніж min_copies підтверджених копій у різних доменах відмов."""
    domains = repo.domains()
    per: dict[str, set] = {}
    for oid, backend in repo.cat.q(
            "SELECT vo.oid, r.backend FROM version_objects vo LEFT JOIN replicas r ON r.oid=vo.oid "
            "AND r.state='CONFIRMED' WHERE vo.version_id=?", (version_id,)):
        doms = per.setdefault(oid, set())
        if backend in domains:
            doms.add(domains[backend])
    counts = [len(d) for d in per.values()]
    return {"objects_total": len(counts), "objects_under_replicated": sum(c < min_copies for c in counts),
            "confirmed_copies_min": min(counts, default=0), "min_copies": min_copies}


def refresh_version(repo: Repo, version_id: str) -> str:
    """Перераховує стан версії з фактів каталогу. PROTECTED — лише коли політику виконано для всіх
    блоків і маніфесту."""
    cat = repo.cat
    with cat.tx():
        v = cat.one("SELECT * FROM versions WHERE version_id=?", (version_id,))
        if v is None or v["state"] not in LIVE:
            return v["state"] if v else ""
        under = protection(repo, version_id, v["min_copies"])["objects_under_replicated"]
        active, touched = cat.one(
            "SELECT COALESCE(SUM(j.state IN ('PENDING','RUNNING')),0), "
            "COALESCE(SUM(j.attempts>0 OR j.state IN ('RUNNING','DONE')),0) "
            "FROM jobs j JOIN version_objects vo ON vo.oid=j.oid WHERE vo.version_id=?", (version_id,))
        if under == 0:
            new = VersionState.PROTECTED
        elif active:
            new = VersionState.ACCEPTED_LOCAL if v["state"] == "ACCEPTED_LOCAL" and not touched \
                else VersionState.REPLICATING
        else:
            new = VersionState.DEGRADED
        if new != v["state"]:
            fields = {}
            if new == VersionState.PROTECTED and v["protected_ns"] is None:
                fields["protected_ns"] = now_ns()
                cat.event("time_to_protected_seconds", (fields["protected_ns"] - v["accepted_ns"]) / 1e9, version_id)
            cat.set_state(version_id, new, **fields)
        return str(new)


@dataclass
class _Work:
    job: str
    oid: str
    backend: Backend
    length: int
    sha256: str
    spool: Path | None
    sources: list[Backend]
    metas: list[tuple[str, bytes]]
    tmp: Path


def _execute(w: _Work) -> str | None:
    """Передає один об'єкт. None — успіх із перевіреним підтвердженням, інакше текст помилки."""
    try:
        src = w.spool
        if src is None:                                    # у spool уже нема: беремо з підтвердженої копії
            for other in w.sources:
                try:
                    other.get(w.oid, w.tmp)
                    src = w.tmp
                    break
                except (BackendError, OSError):
                    continue
            if src is None:
                return "нема джерела: об'єкта немає ні в spool, ні на доступних backend-ах"
        check_ack(w.backend.put(w.oid, src, w.length, w.sha256, w.job), w.oid, w.length, w.sha256)
        for name, data in w.metas:
            w.backend.put_meta(name, data)
        return None
    except (BackendError, OSError) as e:
        return str(e) if isinstance(e, BackendError) else f"помилка вводу-виводу: {e.strerror or type(e).__name__}"
    finally:
        w.tmp.unlink(missing_ok=True)


def _lease(repo: Repo) -> list:
    cat, now, rows = repo.cat, now_ns(), []
    with cat.tx():
        recovered = cat.x("UPDATE jobs SET state='PENDING', lease_until_ns=NULL WHERE state='RUNNING' "
                          "AND lease_until_ns<?", (now,))
        if recovered:
            cat.event("stuck_jobs_recovered", recovered)
        for prio, share in WEIGHTS:
            rows += cat.q("SELECT j.*, o.kind, o.stored_len, o.sha256, o.in_spool FROM jobs j JOIN objects o "
                          "ON o.oid=j.oid WHERE j.state='PENDING' AND j.priority=? AND j.next_at_ns<=? "
                          "ORDER BY j.enq_ns, j.job_id LIMIT ?", (prio, now, share * BATCH_UNIT))
        lease = now + int(repo.limits["lease_seconds"] * 1e9)
        for r in rows:
            cat.x("UPDATE jobs SET state='RUNNING', lease_until_ns=? WHERE job_id=?", (lease, r["job_id"]))
    return rows


def _prepare(repo: Repo, r) -> _Work:
    cat = repo.cat
    spool = repo.spool_path(r["oid"])
    sources = [repo.backend(b) for (b,) in cat.q(
        "SELECT backend FROM replicas WHERE oid=? AND state='CONFIRMED' AND backend!=?", (r["oid"], r["backend"]))
        if b in repo.backend_names()]
    metas = []
    if r["kind"] == KIND_MANIFEST:                         # покажчик версії та конфігурація без секретів:
        metas.append((CONFIG_META, repo.config_for_backends()))   # без них recover неможливий
        for v in cat.q("SELECT version_id, created_ns, pointer FROM versions WHERE manifest_oid=?", (r["oid"],)):
            metas.append((pointer_name(v["version_id"]), v["pointer"].encode() if v["pointer"] else json.dumps(
                pointer_doc(v["version_id"], r["oid"], r["stored_len"], r["sha256"], v["created_ns"])).encode()))
    return _Work(r["job_id"], r["oid"], repo.backend(r["backend"]), r["stored_len"], r["sha256"],
                 spool if r["in_spool"] and spool.is_file() else None, sources, metas,
                 repo.spool / "tmp" / f"fetch-{r['job_id']}")


def _apply(repo: Repo, rows: list, results: list[str | None]) -> dict:
    cat, lim, now = repo.cat, repo.limits, now_ns()
    out = {"done": 0, "retry": 0, "attention": 0}
    with cat.tx():
        for r, err in zip(rows, results):
            if err is None:
                cat.x("INSERT INTO replicas(oid, backend, state, ack_len, ack_sha, acked_ns, verified_ns) "
                      "VALUES(?,?,?,?,?,?,?) ON CONFLICT(oid, backend) DO UPDATE SET state=excluded.state, "
                      "ack_len=excluded.ack_len, ack_sha=excluded.ack_sha, acked_ns=excluded.acked_ns, "
                      "verified_ns=excluded.verified_ns",
                      (r["oid"], r["backend"], ReplicaState.CONFIRMED, r["stored_len"], r["sha256"], now, now))
                cat.x("UPDATE jobs SET state='DONE', done_ns=?, lease_until_ns=NULL, last_error=NULL WHERE job_id=?",
                      (now, r["job_id"]))
                out["done"] += 1
                continue
            attempts = r["attempts"] + 1
            exhausted = attempts >= lim["max_attempts"]
            delay = min(lim["backoff_cap_seconds"], lim["backoff_base_seconds"] * 2 ** (attempts - 1))
            cat.x("UPDATE jobs SET state=?, attempts=?, next_at_ns=?, lease_until_ns=NULL, last_error=? WHERE job_id=?",
                  (JobState.ATTENTION if exhausted else JobState.PENDING, attempts,
                   now + int(delay * random.uniform(0.5, 1.5) * 1e9), err[:300], r["job_id"]))
            cat.event("job_error", attempts, f"{r['backend']}: {err[:200]}")
            out["attention" if exhausted else "retry"] += 1
        oids = {r["oid"] for r in rows}
        marks = ",".join("?" * len(oids))
        for (vid,) in cat.q(f"SELECT DISTINCT version_id FROM version_objects WHERE oid IN ({marks})", tuple(oids)):
            refresh_version(repo, vid)
    return out


def evict_spool(repo: Repo) -> int:
    """Прибирає зі spool об'єкти, усі версії яких уже PROTECTED. Це не видалення прийнятих даних:
    політику копій для них виконано, і restore читає їх із backend-ів."""
    cat, n = repo.cat, 0
    while True:
        rows = cat.q("SELECT o.oid FROM objects o WHERE o.in_spool=1 "
                     "AND EXISTS(SELECT 1 FROM version_objects vo WHERE vo.oid=o.oid) "
                     "AND NOT EXISTS(SELECT 1 FROM version_objects vo JOIN versions v ON v.version_id=vo.version_id "
                     "WHERE vo.oid=o.oid AND v.state!='PROTECTED') LIMIT 500")
        if not rows:
            return n
        with cat.tx():
            for (oid,) in rows:
                cat.x("UPDATE objects SET in_spool=0 WHERE oid=?", (oid,))
        for (oid,) in rows:                                # спершу каталог, потім файл: аварія лишає сироту, а не діру
            repo.spool_path(oid).unlink(missing_ok=True)
        n += len(rows)


def run_queue(repo: Repo, *, wait: bool = True, deadline_ns: int | None = None,
              until_version: str | None = None) -> dict:
    """Обробляє чергу. wait=False — лише завдання, час яких настав; wait=True — чекає повторів,
    доки черга не спорожніє, не вийде deadline або until_version не дійде до PROTECTED/DEGRADED."""
    cat, total = repo.cat, {"done": 0, "retry": 0, "attention": 0}
    with ThreadPoolExecutor(max_workers=repo.limits["max_parallel"]) as pool:
        while True:
            if until_version and cat.scalar("SELECT state FROM versions WHERE version_id=?",
                                            (until_version,)) in ("PROTECTED", "DEGRADED"):
                break
            rows = _lease(repo)
            if not rows:
                nxt = cat.scalar("SELECT MIN(COALESCE(lease_until_ns, next_at_ns)) FROM jobs "
                                 "WHERE state IN ('PENDING','RUNNING')")
                now = now_ns()
                if not wait or nxt is None or (deadline_ns is not None and now >= deadline_ns):
                    break
                time.sleep(max(0.01, min(1.0, (nxt - now) / 1e9)))
                continue
            results = list(pool.map(_execute, [_prepare(repo, r) for r in rows]))
            crashpoint("replicate.after_put_before_commit")
            for k, v in _apply(repo, rows, results).items():
                total[k] += v
            if deadline_ns is not None and now_ns() >= deadline_ns:
                break
    total["evicted"] = evict_spool(repo)
    cat.checkpoint()
    return total
