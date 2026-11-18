"""Обслуговування: verify (перевірка копій), repair, gc, recover і операційні показники."""
from __future__ import annotations

import json
from pathlib import Path

from . import manifest
from .backends.base import Backend
from .catalog import now_ns
from .cryptobox import KIND_CHUNK, KIND_MANIFEST, read_secret
from .errors import BackendError, Exit, VaultError, crashpoint
from .fsutil import free_bytes, sha256_file
from .keys import sign_meta
from .replicate import CONFIG_META, LIVE, ensure_jobs, pointer_name, protection, refresh_version, run_queue
from .repo import Repo, add_backend, validate_config
from .states import LEVEL, ReplicaState, VersionState

PAGE = 500


def _refresh_all(repo: Repo) -> None:
    for (vid,) in repo.cat.q(f"SELECT version_id FROM versions WHERE state IN ({','.join('?' * len(LIVE))})", LIVE):
        refresh_version(repo, vid)


def verify(repo: Repo, version_id: str | None = None) -> dict:
    """Перечитує кожну копію на backend-і (довжина + SHA-256) і локальний spool.
    Відсутня або пошкоджена копія перестає рахуватись, версія може стати DEGRADED."""
    cat = repo.cat
    out = {"checked": 0, "ok": 0, "missing": 0, "corrupt": 0, "unreachable": 0, "spool_corrupt": 0}
    scope = "AND r.oid IN (SELECT oid FROM version_objects WHERE version_id=?)" if version_id else ""
    last = ("", "")
    while True:
        rows = cat.q(f"SELECT r.oid, r.backend, o.stored_len, o.sha256 FROM replicas r JOIN objects o ON o.oid=r.oid "
                     f"WHERE (r.oid, r.backend) > (?, ?) {scope} ORDER BY r.oid, r.backend LIMIT {PAGE}",
                     (*last, *((version_id,) if version_id else ())))
        if not rows:
            break
        last, updates = (rows[-1]["oid"], rows[-1]["backend"]), []
        for r in rows:
            out["checked"] += 1
            if r["backend"] not in repo.backend_names():
                out["unreachable"] += 1
                continue
            try:
                ack = repo.backend(r["backend"]).stat(r["oid"])
            except (BackendError, OSError):
                out["unreachable"] += 1                     # недоступність — не доказ втрати: стан копії не міняємо
                continue
            state = ReplicaState.MISSING if ack is None else (
                ReplicaState.CONFIRMED if (ack.length, ack.sha256) == (r["stored_len"], r["sha256"])
                else ReplicaState.CORRUPT)
            out[{"CONFIRMED": "ok", "MISSING": "missing", "CORRUPT": "corrupt"}[state]] += 1
            updates.append((state, now_ns(), r["oid"], r["backend"]))
        with cat.tx():
            for u in updates:
                cat.x("UPDATE replicas SET state=?, verified_ns=? WHERE oid=? AND backend=?", u)
    last_oid = ""
    while rows := cat.q("SELECT oid, stored_len, sha256 FROM objects WHERE in_spool=1 AND oid>? ORDER BY oid LIMIT ?",
                        (last_oid, PAGE)):
        last_oid = rows[-1]["oid"]
        for r in rows:
            p = repo.spool_path(r["oid"])
            if not p.is_file() or sha256_file(p) != (r["stored_len"], r["sha256"]):
                out["spool_corrupt"] += 1
                with cat.tx():
                    cat.x("UPDATE objects SET in_spool=0 WHERE oid=?", (r["oid"],))
                p.unlink(missing_ok=True)
    with cat.tx():
        _refresh_all(repo)
        cat.set_meta("last_verify_ns", str(now_ns()))
        cat.event("verify", out["missing"] + out["corrupt"] + out["spool_corrupt"], json.dumps(out))
    return out


def housekeeping(repo: Repo) -> dict:
    """Після аварії: завислі захоплення -> FAILED, файли-сироти й tmp у spool прибираються."""
    cat, out = repo.cat, {"failed_captures": 0, "orphans_removed": 0}
    stale = now_ns() - int(repo.limits["capture_stale_seconds"] * 1e9)
    for (vid,) in cat.q("SELECT version_id FROM versions WHERE state='CAPTURING' AND heartbeat_ns<?", (stale,)):
        cat.set_state(vid, VersionState.FAILED, error="захоплення перервано аварією процесу")
        out["failed_captures"] += 1
    if cat.scalar("SELECT COUNT(*) FROM versions WHERE state='CAPTURING'"):
        return out                                          # іде живе захоплення: його нові файли не чіпаємо
    for p in (repo.spool / "objects").glob("*/*"):
        if cat.one("SELECT 1 FROM objects WHERE oid=? AND in_spool=1", (p.name,)) is None:
            p.unlink(missing_ok=True)
            out["orphans_removed"] += 1
    for p in (repo.spool / "tmp").iterdir():
        p.unlink(missing_ok=True)
    return out


def repair(repo: Repo, *, all_backends: bool = False, run: bool = True, timeout: float | None = None) -> dict:
    """Повертає чергу й копії до політики: прибирає наслідки аварій, заново ставить завдання для
    об'єктів без достатньої кількості копій (зокрема після ATTENTION, MISSING, CORRUPT) і обробляє чергу."""
    cat = repo.cat
    if not cat.integrity_ok():
        raise VaultError("E_CATALOG", "каталог SQLite не пройшов integrity_check; відновіть його командою recover",
                         Exit.INTEGRITY)
    out = housekeeping(repo)
    out["requeued"] = 0
    for v in cat.q(f"SELECT * FROM versions WHERE state IN ({','.join('?' * len(LIVE))})", LIVE):
        if protection(repo, v["version_id"], v["min_copies"])["objects_under_replicated"] == 0:
            continue
        backends = repo.backend_names() if all_backends else repo.targets(v["target_copies"])
        with cat.tx():
            out["requeued"] += ensure_jobs(repo, v["version_id"], backends, v["priority"])
            if v["state"] in ("DEGRADED", "PROTECTED"):
                cat.set_state(v["version_id"], VersionState.REPLICATING)
    crashpoint("repair.after_requeue")
    if run:
        out["queue"] = run_queue(repo, deadline_ns=None if timeout is None else now_ns() + int(timeout * 1e9))
    _refresh_all(repo)
    out["versions_not_protected"] = cat.scalar(
        "SELECT COUNT(*) FROM versions WHERE state IN ('ACCEPTED_LOCAL','REPLICATING','DEGRADED')")
    return out


def gc(repo: Repo, *, apply: bool = False) -> dict:
    """Видаляє версії, у яких минув retention, і об'єкти, на які більше ніхто не посилається.
    Не чіпає: keep_last найновіших версій кожного джерела, версії з незавершеною реплікацією
    (ACCEPTED_LOCAL, REPLICATING), об'єкти з активними завданнями, маніфести живих версій
    (вони і є залежностями декодування). Ключі gc не зачіпає взагалі. Без apply — лише звіт."""
    cat, now = repo.cat, now_ns()
    if cat.scalar("SELECT COUNT(*) FROM versions WHERE state='CAPTURING'"):
        raise VaultError("E_BUSY", "іде захоплення; gc відкладено (після аварії спершу виконайте repair)")
    keep = repo.policy["keep_last"]
    doomed = [r["version_id"] for r in cat.q(
        "SELECT version_id, state, retention_until_ns, ROW_NUMBER() OVER (PARTITION BY root, name, "
        "state IN ('PROTECTED','DEGRADED') ORDER BY created_ns DESC) AS rank FROM versions "
        "WHERE state IN ('PROTECTED','DEGRADED','FAILED')")
        if r["state"] == "FAILED" or (r["rank"] > keep and r["retention_until_ns"] < now)]
    out = {"applied": apply, "versions": len(doomed), "objects": 0, "bytes": 0, "delete_errors": 0}
    if apply:
        with cat.tx():
            for vid in doomed:
                cat.set_state(vid, VersionState.DELETED)
                cat.x("DELETE FROM version_objects WHERE version_id=?", (vid,))
        crashpoint("gc.after_mark_deleted")
    refs = "" if apply else f"AND version_id NOT IN ({','.join('?' * len(doomed))})"
    orphans = cat.q("SELECT oid, stored_len FROM objects o WHERE NOT EXISTS(SELECT 1 FROM version_objects vo "
                    f"WHERE vo.oid=o.oid {refs}) AND NOT EXISTS(SELECT 1 FROM jobs j WHERE j.oid=o.oid "
                    "AND j.state IN ('PENDING','RUNNING'))", () if apply else tuple(doomed))
    for o in orphans:
        out["objects"] += 1
        out["bytes"] += o["stored_len"]
        if not apply:
            continue
        try:
            for (b,) in cat.q("SELECT backend FROM replicas WHERE oid=?", (o["oid"],)):
                if b in repo.backend_names():
                    repo.backend(b).delete(o["oid"])
        except (BackendError, OSError):
            out["delete_errors"] += 1                       # рядок лишається: наступний gc повторить
            continue
        repo.spool_path(o["oid"]).unlink(missing_ok=True)
        with cat.tx():
            for table in ("replicas", "jobs", "objects"):
                cat.x(f"DELETE FROM {table} WHERE oid=?", (o["oid"],))
    if apply:
        for vid in doomed:
            for b in repo.backend_names():
                try:
                    repo.backend(b).delete_meta(pointer_name(vid))
                except (BackendError, OSError):
                    out["delete_errors"] += 1
        cat.event("gc", out["objects"], json.dumps(out))
    return out


def recover(path: Path, spec: dict, backend: Backend, *, accept_legacy_meta: bool = False) -> dict:
    """Відтворює сховище після втрати локального диска з одного backend-а.

    repo-config.json і покажчики версій на backend-і підписані HMAC ключем службових файлів
    (keys.py): підроблений чи переставлений файл відхиляється, а не імпортується. Параметри KDF
    із недовіреного файла обмежено ще до виведення ключа. Файли без підпису (сховища до етапу 3)
    приймаються лише з accept_legacy_meta=True; тоді конфігурацію слід переглянути й виконати
    key publish."""
    from .keys import RepoKeys, parse_meta, verify_meta
    from .restore import open_object

    raw = backend.get_meta(CONFIG_META)
    if raw is None:
        raise VaultError("E_NOT_FOUND", "на backend-і нема repo-config.json: тут не було захищених версій",
                         Exit.NOT_FOUND)
    cfg = parse_meta(raw)
    legacy = "meta_mac" not in cfg
    if legacy and not accept_legacy_meta:
        raise VaultError("E_META", "repo-config.json без підпису (створено до етапу 3); якщо backend довірений, "
                         "повторіть із --accept-legacy-meta", Exit.INTEGRITY)
    try:
        crypto = cfg["crypto"]
        passphrase = read_secret(crypto["passphrase_env"], prompt="Парольна фраза сховища: ")
        keys = RepoKeys(passphrase, crypto)
    except (KeyError, TypeError):
        raise VaultError("E_META", "repo-config.json пошкоджено", Exit.INTEGRITY) from None
    meta_key = keys.meta_key()
    if not legacy:
        cfg = verify_meta(meta_key, CONFIG_META, cfg)
    validate_config(cfg)
    repo = Repo.create(path, cfg, passphrase, new_secret=False)
    add_backend(repo, spec)
    cat, out = repo.cat, {"versions": 0, "skipped": 0, "objects": 0, "rejected_meta": 0,
                          "legacy_unsigned_meta": int(legacy)}
    for name in backend.list_meta():
        if not name.startswith("version-"):
            continue
        try:
            ptr = parse_meta(backend.get_meta(name) or b"{}")
            signed = "meta_mac" in ptr
            if signed:
                ptr = verify_meta(meta_key, name, ptr)
            elif not accept_legacy_meta:
                raise VaultError("E_META", f"{name} без підпису", Exit.INTEGRITY)
            else:
                out["legacy_unsigned_meta"] += 1
        except VaultError:
            out["rejected_meta"] += 1                        # підміна покажчика: версію не імпортуємо
            continue
        try:
            moid = ptr["manifest_oid"]
            doc = open_object(repo, moid, KIND_MANIFEST, ptr["manifest_len"], ptr["manifest_sha256"], [backend])
            m = manifest.load(doc, insensitive=False)
            if m["version_id"] != ptr["version_id"] or cat.one("SELECT 1 FROM versions WHERE version_id=?",
                                                               (m["version_id"],)):
                raise VaultError("E_MANIFEST", "маніфест не відповідає покажчику", Exit.INTEGRITY)
        except (VaultError, BackendError, KeyError, ValueError, TypeError):
            out["skipped"] += 1                             # покажчик без цілого маніфесту: версію не відновити
            continue
        objs = {moid: {"plain_len": len(doc), "stored_len": ptr["manifest_len"], "sha256": ptr["manifest_sha256"],
                       "codec": 1, "kind": KIND_MANIFEST},
                **{oid: {**o, "kind": KIND_CHUNK} for oid, o in m["objects"].items()}}
        files = [e for e in m["entries"] if e["type"] == "file"]
        with cat.tx():
            cat.x("INSERT INTO versions(version_id, op_id, name, root, state, priority, min_copies, target_copies, "
                  "manifest_oid, created_ns, heartbeat_ns, accepted_ns, retention_until_ns, files, bytes, skipped, "
                  "pointer) VALUES(?,?,?,?,?,1,?,?,?,?,?,?,?,?,?,?,?)",
                  (m["version_id"], m["operation_id"], m["name"], m["source"]["root"], VersionState.ACCEPTED_LOCAL,
                   repo.policy["min_copies"], repo.policy["min_copies"], moid, m["created_at_ns"], now_ns(),
                   m["created_at_ns"], m["created_at_ns"] + int(repo.policy["retention_days"] * 86400e9),
                   len(files), sum(e["size"] for e in files), len(m["skipped"]),
                   sign_meta(meta_key, name, ptr).decode("utf-8") if signed else None))
            for oid, o in objs.items():
                cat.x("INSERT OR IGNORE INTO objects(oid, kind, stored_len, sha256, plain_len, codec, in_spool, "
                      "created_ns) VALUES(?,?,?,?,?,?,0,?)", (oid, o["kind"], o["stored_len"], o["sha256"],
                                                              o["plain_len"], o["codec"], m["created_at_ns"]))
                cat.x("INSERT OR IGNORE INTO version_objects VALUES(?,?)", (m["version_id"], oid))
                ack = backend.stat(oid, rehash=False)       # наявність і довжина; повну перевірку робить verify
                if ack is not None and ack.length == o["stored_len"]:
                    cat.x("INSERT OR IGNORE INTO replicas(oid, backend, state, ack_len, ack_sha, acked_ns) "
                          "VALUES(?,?,?,?,?,?)", (oid, spec["name"], ReplicaState.CONFIRMED, ack.length,
                                                  o["sha256"], now_ns()))
            for e in files:
                cat.x("INSERT OR IGNORE INTO files(file_id, root, path) VALUES(?,?,?)",
                      (e["file_id"], m["source"]["root"], e["path"]))
            refresh_version(repo, m["version_id"])
        out["versions"] += 1
        out["objects"] += len(objs)
    return out


def version_view(repo: Repo, v) -> dict:
    """Однозначний опис версії для CLI/JSON: стан, рівень захисту і чи виконано політику копій."""
    prot = protection(repo, v["version_id"], v["min_copies"]) if v["state"] in LIVE else None
    attention = repo.cat.scalar("SELECT COUNT(*) FROM jobs j JOIN version_objects vo ON vo.oid=j.oid "
                                "WHERE vo.version_id=? AND j.state='ATTENTION'", (v["version_id"],))
    return {"version_id": v["version_id"], "operation_id": v["op_id"], "name": v["name"], "source_root": v["root"],
            "state": v["state"], "protection_level": LEVEL[VersionState(v["state"])],
            "policy_satisfied": v["state"] == "PROTECTED", "locally_accepted": v["state"] in LIVE,
            "priority": v["priority"], "copies": prot, "jobs_needing_attention": attention,
            "files": v["files"], "bytes": v["bytes"], "skipped_files": v["skipped"], "error": v["error"],
            "created_at": manifest.rfc3339(v["created_ns"]), "accepted_at": manifest.rfc3339(v["accepted_ns"]),
            "protected_at": manifest.rfc3339(v["protected_ns"]),
            "retention_until": manifest.rfc3339(v["retention_until_ns"])}


def metrics(repo: Repo) -> dict:
    """Операційні показники; одиниця виміру — у назві кожного поля."""
    cat, lim, now = repo.cat, repo.limits, now_ns()
    oldest = cat.scalar("SELECT MIN(enq_ns) FROM jobs WHERE state IN ('PENDING','RUNNING')")
    last = {k: cat.one("SELECT value, ts_ns FROM events WHERE kind=? ORDER BY id DESC LIMIT 1", (k,))
            for k in ("time_to_protected_seconds", "restore_seconds")}
    verify_ns = cat.scalar("SELECT value FROM meta WHERE key='last_verify_ns'")
    states = dict(cat.q("SELECT state, COUNT(*) FROM versions GROUP BY state"))
    return {
        "queue_jobs": cat.scalar("SELECT COUNT(*) FROM jobs WHERE state IN ('PENDING','RUNNING')"),
        "queue_capacity_jobs": lim["queue_max_jobs"],
        "queue_attention_jobs": cat.scalar("SELECT COUNT(*) FROM jobs WHERE state='ATTENTION'"),
        "queue_oldest_job_age_seconds": None if oldest is None else round((now - oldest) / 1e9, 3),
        "spool_bytes": cat.scalar("SELECT COALESCE(SUM(stored_len),0) FROM objects WHERE in_spool=1"),
        "spool_capacity_bytes": lim["spool_max_bytes"],
        "spool_disk_free_bytes": free_bytes(repo.spool),
        "spool_disk_min_free_bytes": lim["min_free_bytes"],
        "max_parallel_transfers": lim["max_parallel"],
        "versions_by_state": states,
        "under_replicated_versions": sum(states.get(s, 0) for s in ("ACCEPTED_LOCAL", "REPLICATING", "DEGRADED")),
        "last_time_to_protected_seconds": last["time_to_protected_seconds"] and last["time_to_protected_seconds"][0],
        "avg_time_to_protected_seconds": cat.scalar(
            "SELECT AVG(value) FROM events WHERE kind='time_to_protected_seconds'"),
        "job_errors_total": cat.scalar("SELECT COUNT(*) FROM events WHERE kind='job_error'"),
        "advisor_errors_total": cat.scalar("SELECT COALESCE(SUM(value),0) FROM events WHERE kind='advisor_errors'"),
        "last_verify_at": manifest.rfc3339(int(verify_ns)) if verify_ns else None,
        "last_restore_duration_seconds": last["restore_seconds"] and last["restore_seconds"][0],
        "measured_at": manifest.rfc3339(now),
    }
