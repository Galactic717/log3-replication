"""Операторські перевірки: doctor, health (пороги), queue, find, restore drill, Prometheus.

Ніщо тут не друкує секретів. Мітки метрик мають обмежену множину значень (стан версії, стан
завдання, назва backend-а з конфігурації); шляхи, ідентифікатори файлів і версій міткою не бувають.
"""
from __future__ import annotations

import fnmatch
import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path

from . import manifest
from .catalog import SCHEMA_VERSION, now_ns
from .errors import BackendError, Exit, VaultError
from .fsutil import access_report, free_bytes, sha256_file
from .keyops import status as key_status
from .maintain import metrics
from .repo import Repo
from .states import LEVEL, VersionState

#: пороги health; перевизначаються секцією config.json "monitoring" або прапорцями CLI
THRESHOLDS = {"max_queue_age_seconds": 3600.0, "max_queue_fill": 0.8, "max_protection_lag_seconds": 3600.0,
              "max_attention_jobs": 0, "min_free_factor": 2.0, "max_verify_age_seconds": 7 * 86400.0,
              "max_drill_age_seconds": 30 * 86400.0, "max_nonce_budget": 0.5}


def _age(ns: int | None) -> float | None:
    return None if ns is None else round((now_ns() - ns) / 1e9, 3)


def extended_metrics(repo: Repo) -> dict:
    cat, m = repo.cat, metrics(repo)
    lag = cat.scalar("SELECT MIN(accepted_ns) FROM versions WHERE state IN ('ACCEPTED_LOCAL','REPLICATING','DEGRADED')")
    drill = cat.one("SELECT value, ts_ns, detail FROM events WHERE kind='restore_drill' ORDER BY id DESC LIMIT 1")
    verify_ns = cat.scalar("SELECT value FROM meta WHERE key='last_verify_ns'")
    ks = key_status(repo)
    active = next(k for k in ks["keys"] if k["active"])
    m.update({
        "protection_lag_seconds": _age(lag),
        "queue_fill_ratio": round(m["queue_jobs"] / max(1, m["queue_capacity_jobs"]), 6),
        "job_errors_last_hour": cat.scalar("SELECT COUNT(*) FROM events WHERE kind='job_error' AND ts_ns>?",
                                           (now_ns() - 3600 * 10 ** 9,)),
        "last_verify_age_seconds": _age(int(verify_ns)) if verify_ns else None,
        "last_verify_problems": cat.scalar("SELECT value FROM events WHERE kind='verify' ORDER BY id DESC LIMIT 1"),
        "last_drill_ok": None if drill is None else bool(drill["value"]),
        "last_drill_age_seconds": None if drill is None else _age(drill["ts_ns"]),
        "last_drill": None if drill is None else json.loads(drill["detail"]),
        "nonce_budget_used": active["nonce_budget_used"],
        "catalog_schema_version": int(cat.scalar("SELECT value FROM meta WHERE key='schema_version'")),
    })
    return m


def health(repo: Repo, overrides: dict | None = None) -> dict:
    """Машинозчитувані перевірки з порогами. ok=False хоча б для однієї — код ATTENTION."""
    th = {**THRESHOLDS, **repo.cfg.get("monitoring", {}), **{k: v for k, v in (overrides or {}).items()
                                                              if v is not None}}
    m = extended_metrics(repo)
    free_need = repo.limits["min_free_bytes"] * th["min_free_factor"]

    def le(value, limit):
        return value is not None and value <= limit

    checks = [
        ("queue_age", m["queue_oldest_job_age_seconds"] is None or m["queue_oldest_job_age_seconds"]
         <= th["max_queue_age_seconds"], m["queue_oldest_job_age_seconds"], th["max_queue_age_seconds"]),
        ("queue_fill", m["queue_fill_ratio"] <= th["max_queue_fill"], m["queue_fill_ratio"], th["max_queue_fill"]),
        ("protection_lag", m["protection_lag_seconds"] is None or m["protection_lag_seconds"]
         <= th["max_protection_lag_seconds"], m["protection_lag_seconds"], th["max_protection_lag_seconds"]),
        ("attention_jobs", m["queue_attention_jobs"] <= th["max_attention_jobs"], m["queue_attention_jobs"],
         th["max_attention_jobs"]),
        ("free_space", m["spool_disk_free_bytes"] >= free_need, m["spool_disk_free_bytes"], free_need),
        ("verify_age", le(m["last_verify_age_seconds"], th["max_verify_age_seconds"]),
         m["last_verify_age_seconds"], th["max_verify_age_seconds"]),
        ("verify_result", not m["last_verify_problems"], m["last_verify_problems"], 0),
        ("drill", bool(m["last_drill_ok"]) and le(m["last_drill_age_seconds"], th["max_drill_age_seconds"]),
         m["last_drill_age_seconds"], th["max_drill_age_seconds"]),
        ("nonce_budget", m["nonce_budget_used"] <= th["max_nonce_budget"], m["nonce_budget_used"],
         th["max_nonce_budget"]),
    ]
    out = [{"check": n, "ok": bool(ok), "value": v, "threshold": t} for n, ok, v, t in checks]
    return {"healthy": all(c["ok"] for c in out), "checks": out, "metrics": m}


def _cert_days_left(path: str) -> float | None:
    try:
        from cryptography import x509
        cert = x509.load_pem_x509_certificate(Path(path).read_bytes())
        return round((cert.not_valid_after_utc - datetime.now(UTC)).total_seconds() / 86400, 1)
    except (OSError, ValueError):
        return None


def doctor(repo: Repo) -> dict:
    """Перевірка інсталяції й середовища без секретів і без змін даних."""
    checks = []

    def add(name: str, ok: bool | None, detail) -> None:
        checks.append({"check": name, "ok": ok, "detail": detail})

    add("config", True, {"format_version": repo.cfg["format_version"], "repo_id": repo.cfg["repo_id"]})
    add("catalog_integrity", repo.cat.integrity_ok(), {"schema_version": int(repo.cat.scalar(
        "SELECT value FROM meta WHERE key='schema_version'")), "supported": SCHEMA_VERSION})
    acc = access_report(repo.path)
    add("repo_access_owner_only", None if acc.get("others_have_access") is None else not acc["others_have_access"],
        acc)
    doms = repo.domains()
    add("policy_domains", len(set(doms.values())) >= repo.policy["min_copies"],
        {"independent_failure_domains": len(set(doms.values())), "min_copies": repo.policy["min_copies"]})
    for spec in repo.cfg["backends"]:
        try:
            repo.backend(spec["name"]).list_meta()
            add(f"backend:{spec['name']}", True, {"type": spec["type"], "domain": doms[spec["name"]]})
        except (BackendError, OSError) as e:
            add(f"backend:{spec['name']}", False, {"type": spec["type"], "error": str(e)[:200]})
        if spec["type"] == "node":
            days = _cert_days_left(spec["ca_file"])
            add(f"node_cert:{spec['name']}", days is not None and days > 30, {"days_left": days})
    free = free_bytes(repo.spool)
    add("free_space", free >= 2 * repo.limits["min_free_bytes"], {"free_bytes": free})
    ks = key_status(repo)
    active = next(k for k in ks["keys"] if k["active"])
    add("nonce_budget", active["nonce_budget_used"] < 0.5, {"active_key_id": active["key_id"],
                                                            "used": active["nonce_budget_used"]})
    env = repo.cfg["crypto"]["passphrase_env"]
    add("passphrase_available", bool(os.environ.get(env) or os.environ.get(env + "_FILE")) or None,
        {"env": env, "note": "потрібна лише для backup/restore/recover/key; черга працює без неї"})
    attention = repo.cat.scalar("SELECT COUNT(*) FROM jobs WHERE state='ATTENTION'")
    add("queue_attention", attention == 0, {"jobs": attention})
    healthy = all(c["ok"] is not False for c in checks)
    return {"ok": healthy, "checks": checks}


def queue(repo: Repo, limit: int = 20) -> dict:
    cat = repo.cat
    by_state = dict(cat.q("SELECT state, COUNT(*) FROM jobs GROUP BY state"))
    by_backend = {b: dict(cat.q("SELECT state, COUNT(*) FROM jobs WHERE backend=? GROUP BY state", (b,)))
                  for b in repo.backend_names()}
    nxt = cat.scalar("SELECT MIN(next_at_ns) FROM jobs WHERE state='PENDING'")
    oldest = cat.scalar("SELECT MIN(enq_ns) FROM jobs WHERE state IN ('PENDING','RUNNING')")
    attention = [{"object": r["oid"][:12], "backend": r["backend"], "attempts": r["attempts"],
                  "last_error": r["last_error"]} for r in cat.q(
        "SELECT oid, backend, attempts, last_error FROM jobs WHERE state='ATTENTION' ORDER BY enq_ns LIMIT ?",
        (limit,))]
    return {"by_state": by_state, "by_backend": by_backend, "capacity_jobs": repo.limits["queue_max_jobs"],
            "oldest_job_age_seconds": _age(oldest),
            "next_retry_in_seconds": None if nxt is None else max(0.0, round((nxt - now_ns()) / 1e9, 3)),
            "max_attempts": repo.limits["max_attempts"], "attention": attention}


def _ts(text: str | None) -> int | None:
    if not text:
        return None
    try:
        d = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        raise VaultError("E_USAGE", f"час {text!r} не у форматі ISO 8601", Exit.USAGE) from None
    return min(int((d if d.tzinfo else d.replace(tzinfo=UTC)).timestamp() * 1e9), (1 << 63) - 1)


def find(repo: Repo, *, version: str | None = None, path: str | None = None, source: str | None = None,
         state: str | None = None, since: str | None = None, until: str | None = None, name: str | None = None,
         limit: int = 200) -> dict:
    """Пошук за метаданими. Версії фільтруються за каталогом без секрету; пошук за шляхом розшифровує
    маніфести відібраних версій локально в пам'яті (потрібна фраза) — backend-и пошуку не бачать."""
    sql, args = "SELECT * FROM versions WHERE state!='DELETED'", []
    if version:
        sql += " AND version_id=?"
        args.append(repo.cat.find_version(version)["version_id"])
    if state:
        if state.upper() not in VersionState.__members__:
            raise VaultError("E_USAGE", f"невідомий стан {state!r}", Exit.USAGE)
        sql += " AND state=?"
        args.append(state.upper())
    for col, val in (("root", source), ("name", name)):
        if val:
            sql += f" AND instr({col}, ?) > 0"
            args.append(val)
    for op, val in ((">=", _ts(since)), ("<=", _ts(until))):
        if val is not None:
            sql += f" AND created_ns {op} ?"
            args.append(val)
    rows = repo.cat.q(sql + " ORDER BY created_ns DESC", tuple(args))
    versions = [{"version_id": v["version_id"], "name": v["name"], "source_root": v["root"], "state": v["state"],
                 "protection_level": LEVEL[VersionState(v["state"])], "created_at": manifest.rfc3339(v["created_ns"]),
                 "files": v["files"], "bytes": v["bytes"]} for v in rows]
    if not path:
        return {"versions": versions[:limit], "matched_versions": len(versions), "manifests_decrypted": 0}
    from .restore import load_manifest
    hits, decrypted = [], 0
    for v in rows:
        if v["state"] not in ("ACCEPTED_LOCAL", "REPLICATING", "PROTECTED", "DEGRADED"):
            continue
        m = load_manifest(repo, v)
        decrypted += 1
        for e in m["entries"]:
            if fnmatch.fnmatchcase(e["path"], path) or fnmatch.fnmatchcase(e["path"].rsplit("/", 1)[-1], path):
                hits.append({"version_id": v["version_id"], "path": e["path"], "type": e["type"],
                             "size": e.get("size"), "mtime": manifest.rfc3339(e["mtime_ns"]),
                             "state": v["state"]})
                if len(hits) >= limit:
                    return {"entries": hits, "truncated": True, "manifests_decrypted": decrypted}
    return {"entries": hits, "truncated": False, "manifests_decrypted": decrypted}


def drill(repo: Repo, ref: str | None = None) -> dict:
    """Пробне відновлення: версія (типово найновіша PROTECTED) відновлюється в тимчасовий каталог
    усередині сховища, кожен файл повторно хешується й звіряється з маніфестом, каталог видаляється.
    Результат (успіх чи помилка) записується в каталог і видно в status/health/metrics."""
    from .restore import load_manifest, restore
    cat = repo.cat
    if ref is None:
        row = cat.one("SELECT version_id FROM versions WHERE state='PROTECTED' ORDER BY created_ns DESC LIMIT 1")
        if row is None:
            raise VaultError("E_NOT_FOUND", "нема жодної версії PROTECTED для пробного відновлення", Exit.NOT_FOUND)
        ref = row[0]
    v = cat.find_version(ref)
    target = repo.spool / "tmp" / f"drill-{os.urandom(4).hex()}"
    result = {"version_id": v["version_id"], "ok": False, "files": 0, "bytes": 0, "mismatched": 0}
    try:
        r = restore(repo, v["version_id"], target)
        m = load_manifest(repo, v)
        for e in m["entries"]:
            if e["type"] != "file":
                continue
            n, sha = sha256_file(target.joinpath(*e["path"].split("/")))
            ok = (n, sha) == (e["size"], m["contents"][e["content_id"]]["sha256"])
            result["mismatched"] += not ok
        result.update(files=r["files"], bytes=r["bytes"], duration_seconds=r["duration_seconds"],
                      ok=result["mismatched"] == 0)
    except VaultError as e:
        result["error"] = e.code
    finally:
        shutil.rmtree(target, ignore_errors=True)
    with cat.tx():
        cat.event("restore_drill", 1.0 if result["ok"] else 0.0, json.dumps(result))
    return result


def prometheus(repo: Repo) -> str:
    """Текстовий формат експозиції Prometheus (без обов'язкового web-сервера: файл або stdout)."""
    m = extended_metrics(repo)
    lines: list[str] = []

    def gauge(name: str, value, help_: str, labels: dict | None = None) -> None:
        full = f"log3_vault_{name}"
        if not any(line.startswith(f"# TYPE {full} ") for line in lines):
            lines.extend([f"# HELP {full} {help_}", f"# TYPE {full} gauge"])
        lab = "" if not labels else "{" + ",".join(
            f'{k}="{str(v).replace(chr(92), chr(92) * 2).replace(chr(34), chr(92) + chr(34))}"'
            for k, v in labels.items()) + "}"
        lines.append(f"{full}{lab} {float('nan') if value is None else float(value)}")

    for key, help_ in (("queue_jobs", "pending or running replication jobs"),
                       ("queue_capacity_jobs", "queue capacity"), ("queue_attention_jobs", "jobs out of retries"),
                       ("queue_oldest_job_age_seconds", "age of the oldest queued job"),
                       ("spool_bytes", "encrypted bytes in local spool"), ("spool_capacity_bytes", "spool limit"),
                       ("spool_disk_free_bytes", "free bytes on the spool disk"),
                       ("under_replicated_versions", "accepted versions without policy copies"),
                       ("protection_lag_seconds", "age of the oldest unprotected accepted version"),
                       ("avg_time_to_protected_seconds", "mean seconds from local accept to protected"),
                       ("job_errors_total", "replication attempt errors"),
                       ("last_verify_age_seconds", "seconds since last verify"),
                       ("last_drill_age_seconds", "seconds since last restore drill"),
                       ("nonce_budget_used", "share of the per-key object limit used by the active key")):
        gauge(key, m[key], help_)
    gauge("last_drill_ok", None if m["last_drill_ok"] is None else int(m["last_drill_ok"]), "last restore drill result")
    for s in VersionState:
        gauge("versions", m["versions_by_state"].get(s.value, 0), "versions by state", {"state": s.value})
    for b, states in queue(repo)["by_backend"].items():
        for s in ("PENDING", "RUNNING", "DONE", "ATTENTION"):
            gauge("backend_jobs", states.get(s, 0), "jobs per configured backend", {"backend": b, "state": s})
    return "\n".join(lines) + "\n"
