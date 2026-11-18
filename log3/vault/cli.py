"""CLI LOG3 Vault: python -m log3.vault <команда> [--repo DIR] [--json] (після встановлення — log3-vault)

  сховище      init, config check|show, backend add|list, doctor, migrate status|restore
  захист       backup, run [--daemon], status, list, find, queue, verify, repair, gc
  відновлення  restore, drill, recover
  ключі        key status|rotate|passphrase|export|publish
  моніторинг   health, metrics [--format prometheus]
  вузол        serve-node, node-cert, node-token add|revoke|list
  навчання     learn status|train|evaluate|promote|rollback|...

З --json кожна команда друкує в stdout один JSON-об'єкт: ok, command, exit_code, exit_name і дані.
Коди завершення стабільні (errors.Exit). Секрети — лише змінні середовища або файли *_FILE.
backup без --wait лише приймає версію локально й ставить реплікацію в чергу (рівень
accepted_local, код 0); backup --wait повертає 0 тільки для рівня protected.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import secrets
import signal
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from . import __version__, keyops, maintain, ops
from .backends.node_server import TOKEN_ENV, TOKENS_FORMAT, NodeServer, make_node_cert, server_tls, token_hash
from .catalog import SCHEMA_VERSION, now_ns
from .cryptobox import PASSPHRASE_ENV, read_secret
from .errors import BackendError, Exit, VaultError
from .fsutil import write_durable
from .replicate import run_queue
from .repo import MIB, PRIORITIES, Repo, add_backend, make_backend, new_config
from .restore import restore

try:                                         # навчання — необов'язкова частина: без нього решта CLI працює
    from .learn import cli as learn_cli
except Exception:                            # pragma: no cover — лише якщо пакет навчання зламано
    learn_cli = None

REPO_ENV = "LOG3_VAULT_REPO"
EXIT_HELP = "коди завершення: " + ", ".join(f"{int(e)} {e.name}" for e in Exit)


def _parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--repo", default=os.environ.get(REPO_ENV), help=f"каталог сховища (або змінна {REPO_ENV})")
    common.add_argument("--json", action="store_true", help="машинозчитуваний вивід: один JSON-об'єкт у stdout")
    p = argparse.ArgumentParser(prog="log3-vault", description=__doc__.split("\n\n")[0], epilog=EXIT_HELP,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="command", required=True, metavar="команда")

    def cmd(name: str, text: str) -> argparse.ArgumentParser:
        return sub.add_parser(name, parents=[common], help=text, description=text, epilog=EXIT_HELP)

    def backend_args(c: argparse.ArgumentParser, required: bool) -> None:
        c.add_argument("--type", choices=("localdir", "node"), required=required)
        c.add_argument("--domain", required=required, help="домен відмов: диск, майданчик, провайдер")
        c.add_argument("--path", help="localdir: каталог на окремому диску")
        c.add_argument("--url", help="node: https://host:port")
        c.add_argument("--ca-file", help="node: закріплений сертифікат вузла (node.crt)")
        c.add_argument("--token-env", default=TOKEN_ENV, help="node: назва змінної середовища з токеном")

    c = cmd("init", "створити сховище (парольна фраза — LOG3_VAULT_PASSPHRASE, не менше 12 символів)")
    c.add_argument("--min-copies", type=int, default=2, help="жорсткий мінімум незалежних копій (типово 2)")
    c.add_argument("--chunk-size-mib", type=float, default=1.0)
    c.add_argument("--retention-days", type=float, default=30.0)
    c.add_argument("--keep-last", type=int, default=3)
    c = cmd("config", "перевірити або показати конфігурацію (секретів у ній нема)")
    c.add_argument("action", choices=("check", "show"))
    c = cmd("backend", "додати backend або показати налаштовані")
    c.add_argument("action", choices=("add", "list"))
    c.add_argument("name", nargs="?")
    backend_args(c, False)
    cmd("doctor", "перевірити інсталяцію: каталог, права доступу, backend-и, місце, ключі, сертифікати")
    c = cmd("migrate", "стан схеми каталогу або повернення копії, зробленої перед міграцією")
    c.add_argument("action", choices=("status", "restore"))
    c.add_argument("--file", help="restore: файл із <repo>/backups")
    c.add_argument("--yes", action="store_true", help="restore: підтвердження")
    c = cmd("backup", "прийняти версію файла або каталогу")
    c.add_argument("source")
    c.add_argument("--name", help="назва набору (типово ім'я джерела)")
    c.add_argument("--priority", choices=tuple(PRIORITIES), default="normal")
    c.add_argument("--op-id", help="ідентифікатор операції: повтор із ним не створює другої версії")
    c.add_argument("--wait", action="store_true", help="обробляти чергу й чекати рівня protected")
    c.add_argument("--timeout", type=float, default=3600.0, help="секунд для --wait")
    c = cmd("run", "обробити чергу реплікації (секрет не потрібен)")
    c.add_argument("--once", action="store_true", help="лише завдання, час яких настав; не чекати повторів")
    c.add_argument("--timeout", type=float, help="секунд; типово до спорожнення черги")
    c.add_argument("--no-learn", action="store_true", help="не виконувати крок циклу навчання після черги")
    c.add_argument("--daemon", action="store_true", help="служба: прохід черги кожні --interval секунд до SIGTERM")
    c.add_argument("--interval", type=float, default=30.0)
    cmd("list", "перелік версій")
    c = cmd("status", "стан сховища з показниками або стан однієї версії")
    c.add_argument("version", nargs="?")
    c = cmd("find", "пошук версій і файлів за метаданими (шлях потребує фрази: маніфести розшифровуються локально)")
    c.add_argument("--version", dest="ref", help="ідентифікатор, префікс або latest")
    c.add_argument("--path", help="glob шляху або імені файла, напр. '*.pdf' чи 'docs/*'")
    c.add_argument("--source", help="фрагмент шляху джерела")
    c.add_argument("--name", help="фрагмент назви набору")
    c.add_argument("--state", help="стан версії, напр. PROTECTED")
    c.add_argument("--since", help="ISO 8601, напр. 2026-11-01T00:00:00Z")
    c.add_argument("--until")
    c.add_argument("--limit", type=int, default=200)
    c = cmd("queue", "черга реплікації: стани, backend-и, вік, найближчий повтор, завдання без повторів")
    c.add_argument("--limit", type=int, default=20)
    c = cmd("verify", "перечитати й перевірити копії на backend-ах")
    c.add_argument("version", nargs="?")
    c = cmd("restore", "відновити версію в каталог")
    c.add_argument("version", help="ідентифікатор, його префікс або latest")
    c.add_argument("dest")
    c.add_argument("--path", help="лише цей файл або підкаталог версії")
    c.add_argument("--overwrite", action="store_true")
    c = cmd("drill", "пробне відновлення з побайтовою звіркою; результат видно в health і metrics")
    c.add_argument("version", nargs="?", help="типово — найновіша PROTECTED")
    c = cmd("repair", "прибрати наслідки аварій і довести копії до політики")
    c.add_argument("--all-backends", action="store_true", help="використати всі backend-и, а не лише цільові")
    c.add_argument("--no-run", action="store_true", help="лише поставити завдання, чергу не обробляти")
    c.add_argument("--timeout", type=float)
    c = cmd("gc", "прибрати версії після retention і непотрібні об'єкти (без --apply лише звіт)")
    c.add_argument("--apply", action="store_true")
    c = cmd("recover", "відтворити сховище з backend-а після втрати локального диска")
    c.add_argument("name", help="назва backend-а")
    backend_args(c, True)
    c.add_argument("--accept-legacy-meta", action="store_true",
                   help="прийняти непідписані службові файли сховища, створеного до етапу 3")
    c = cmd("key", "ключі: status, rotate, passphrase (LOG3_VAULT_NEW_PASSPHRASE), export, publish")
    c.add_argument("action", choices=("status", "rotate", "passphrase", "export", "publish"))
    c.add_argument("--out", help="export: файл (фраза — LOG3_VAULT_EXPORT_PASSPHRASE)")
    c = cmd("health", "перевірки стану з порогами (код 0 або 4)")
    for flag in ("max-queue-age-seconds", "max-protection-lag-seconds", "max-verify-age-seconds",
                 "max-drill-age-seconds"):
        c.add_argument(f"--{flag}", type=float)
    c = cmd("metrics", "показники сховища: JSON або текстовий формат Prometheus")
    c.add_argument("--format", choices=("json", "prometheus"), default="json")
    c.add_argument("--out", help="prometheus: атомарно записати у файл (textfile collector)")
    c = cmd("serve-node", "запустити віддалений вузол зберігання (HTTPS)")
    c.add_argument("--data", required=True)
    c.add_argument("--cert", required=True)
    c.add_argument("--key", required=True)
    c.add_argument("--host", default="0.0.0.0")
    c.add_argument("--port", type=int, default=8443)
    c.add_argument("--quota-bytes", type=int, default=0, help="0 = без квоти")
    c.add_argument("--tokens-file", help="файл токенів із ролями (node-token); без нього — LOG3_NODE_TOKEN як admin")
    c.add_argument("--max-object-mib", type=float, default=256.0)
    c.add_argument("--max-connections", type=int, default=32)
    c = cmd("node-cert", "створити самопідписаний сертифікат вузла")
    c.add_argument("--dir", required=True)
    c.add_argument("--host", action="append", required=True, help="ім'я або IP вузла; можна кілька")
    c = cmd("node-token", "токени вузла: add (у файл --out), revoke, list; у файлі токенів лише SHA-256")
    c.add_argument("action", choices=("add", "revoke", "list"))
    c.add_argument("name", nargs="?")
    c.add_argument("--tokens-file", required=True)
    c.add_argument("--role", choices=("reader", "writer", "admin"), default="writer")
    c.add_argument("--out", help="add: файл, у який записати новий токен (0600)")
    if learn_cli is not None:
        learn_cli.add_parser(sub, common)
    return p


def _priority(repo: Repo, kind: str):
    """Позначка для навчання: іде restore/repair, тож проби й навчання поступаються. Будь-який збій
    позначки ігнорується — restore і repair від навчання не залежать."""
    try:
        from .learn.settings import priority
        return priority(repo.path, kind)
    except Exception:
        return contextlib.nullcontext()


def _learning_summary(repo: Repo, version_id: str) -> dict | None:
    row = repo.cat.one("SELECT detail FROM events WHERE kind='learning' AND detail LIKE ? ORDER BY id DESC LIMIT 1",
                       (f'%"{version_id}"%',))
    try:
        return json.loads(row[0]) if row else None
    except ValueError:
        return None


def _spec(a) -> dict:
    if not a.name or not a.type or not a.domain:
        raise VaultError("E_USAGE", "потрібні назва, --type і --domain", Exit.USAGE)
    spec = {"name": a.name, "type": a.type, "failure_domain": a.domain}
    need = ("path",) if a.type == "localdir" else ("url", "ca_file", "token_env")
    for k in need:
        if not getattr(a, k):
            raise VaultError("E_USAGE", f"для типу {a.type} потрібен --{k.replace('_', '-')}", Exit.USAGE)
        spec[k] = getattr(a, k)
    return spec


def _need_repo(a) -> Path:
    if not a.repo:
        raise VaultError("E_USAGE", f"вкажіть --repo або змінну {REPO_ENV}", Exit.USAGE)
    return Path(a.repo)


def _state_exit(state: str) -> Exit:
    return {"PROTECTED": Exit.OK, "DEGRADED": Exit.ATTENTION, "FAILED": Exit.ERROR,
            "DELETED": Exit.NOT_FOUND}.get(state, Exit.NOT_PROTECTED)


def _node_token(a) -> tuple[Exit, dict]:
    path = Path(a.tokens_file)
    doc = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"format": TOKENS_FORMAT,
                                                                                "format_version": 1, "tokens": []}
    if doc.get("format") != TOKENS_FORMAT:
        raise VaultError("E_CONFIG", "файл токенів має невідомий формат", Exit.POLICY)
    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    if a.action in ("add", "revoke") and not a.name:
        raise VaultError("E_USAGE", "потрібна назва токена", Exit.USAGE)
    if a.action == "add":
        if not a.out or Path(a.out).exists():
            raise VaultError("E_USAGE", "потрібен --out з новим файлом для токена (токен не друкується)", Exit.USAGE)
        if any(t["name"] == a.name and not t.get("revoked") for t in doc["tokens"]):
            raise VaultError("E_EXISTS", f"активний токен {a.name!r} вже є")
        token = secrets.token_urlsafe(32).encode()
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(token)
        doc["tokens"].append({"name": a.name, "role": a.role, "sha256": token_hash(token), "created_at": stamp})
    elif a.action == "revoke":
        hit = [t for t in doc["tokens"] if t["name"] == a.name and not t.get("revoked")]
        if not hit:
            raise VaultError("E_NOT_FOUND", f"активного токена {a.name!r} нема", Exit.NOT_FOUND)
        for t in hit:
            t.update(revoked=True, revoked_at=stamp)
    if a.action != "list":
        write_durable(path, [json.dumps(doc, ensure_ascii=False, indent=1).encode()])
    return Exit.OK, {"tokens": [{k: t.get(k) for k in ("name", "role", "created_at", "revoked", "revoked_at")}
                                for t in doc["tokens"]], "tokens_file": str(path)}


def _daemon(a, repo: Repo) -> tuple[Exit, dict]:
    stop = {"flag": False}

    def _stop(*_):
        stop["flag"] = True

    signal.signal(signal.SIGINT, _stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _stop)
    passes = 0
    while not stop["flag"]:
        summary = run_queue(repo, wait=False)
        if learn_cli is not None and not a.no_learn:
            with contextlib.suppress(Exception):
                learn_cli.after_run(repo)
        passes += 1
        m = maintain.metrics(repo)
        line = {"pass": passes, "queue": summary, "queue_jobs": m["queue_jobs"],
                "attention": m["queue_attention_jobs"], "at": m["measured_at"]}
        print(json.dumps(line, ensure_ascii=False), flush=True)
        end = time.monotonic() + a.interval
        while not stop["flag"] and time.monotonic() < end:
            time.sleep(min(0.5, a.interval))
    return Exit.OK, {"passes": passes, "stopped": "signal"}


def _run(a) -> tuple[Exit, dict]:
    cmd = a.command
    if cmd == "node-cert":
        crt, key = make_node_cert(Path(a.dir), a.host)
        return Exit.OK, {"cert": str(crt), "key": str(key)}
    if cmd == "node-token":
        return _node_token(a)
    if cmd == "serve-node":
        tokens = Path(a.tokens_file) if a.tokens_file else None
        server = NodeServer((a.host, a.port), Path(a.data), None if tokens else read_secret(TOKEN_ENV),
                            server_tls(Path(a.cert), Path(a.key)), a.quota_bytes, tokens_file=tokens,
                            limits={"max_object_bytes": int(a.max_object_mib * MIB),
                                    "max_connections": a.max_connections})
        print(f"log3 node: https://{a.host}:{server.server_address[1]} дані {a.data}", file=sys.stderr, flush=True)
        with contextlib.suppress(KeyboardInterrupt):
            server.serve_forever()
        return Exit.OK, {}
    if cmd == "init":
        cfg = new_config(min_copies=a.min_copies, chunk_size=int(a.chunk_size_mib * MIB))
        cfg["policy"].update(retention_days=a.retention_days, keep_last=a.keep_last)
        repo = Repo.create(_need_repo(a), cfg, read_secret(PASSPHRASE_ENV, prompt="Нова парольна фраза сховища: "))
        return Exit.OK, {"repo": str(repo.path), "repo_id": cfg["repo_id"], "key_id": repo.cfg["crypto"]["key_id"],
                         "min_copies": a.min_copies,
                         "access": repo.cat.scalar("SELECT value FROM meta WHERE key='access_restriction'")}
    if cmd == "recover":
        spec = _spec(a)
        if spec["type"] == "localdir":
            spec["path"] = str(Path(spec["path"]).absolute())
        return Exit.OK, maintain.recover(_need_repo(a), spec, make_backend(spec),
                                         accept_legacy_meta=a.accept_legacy_meta)
    repo = Repo(_need_repo(a))
    if cmd == "config":
        if a.action == "show":
            return Exit.OK, {"config": repo.cfg}
        return Exit.OK, {"config_valid": True, "effective_policy": repo.policy, "effective_limits": repo.limits,
                         "failure_domains": repo.domains()}
    if cmd == "doctor":
        out = ops.doctor(repo)
        return (Exit.OK if out["ok"] else Exit.ATTENTION), {"doctor": out}
    if cmd == "migrate":
        backups = sorted(p.name for p in (repo.path / "backups").glob("catalog-*.sqlite"))
        if a.action == "status":
            version = int(repo.cat.scalar("SELECT value FROM meta WHERE key='schema_version'"))
            return Exit.OK, {"schema_version": version, "supported": SCHEMA_VERSION, "backups": backups}
        if not a.yes or a.file not in backups:
            raise VaultError("E_USAGE", "вкажіть --file з переліку migrate status і --yes", Exit.USAGE)
        repo.cat.db.close()
        db = repo.path / "catalog.sqlite"
        for suffix in ("-wal", "-shm"):
            Path(str(db) + suffix).unlink(missing_ok=True)
        write_durable(db, [(repo.path / "backups" / a.file).read_bytes()])
        return Exit.OK, {"restored": a.file, "note": "версії, прийняті після міграції, поверне repair або recover"}
    if cmd == "backend":
        if a.action == "add":
            add_backend(repo, _spec(a))
        doms = repo.domains()
        return Exit.OK, {"backends": [{**b, "effective_failure_domain": doms[b["name"]]}
                                      for b in repo.cfg["backends"]],
                         "independent_failure_domains": len(set(doms.values())),
                         "min_copies": repo.policy["min_copies"]}
    if cmd == "learn":
        if learn_cli is None:
            raise VaultError("E_LEARN", "пакет навчання недоступний; копіювання працює на базовій політиці")
        return learn_cli.run(a, repo)
    if cmd == "backup":
        from .capture import backup
        vid, repeated = backup(repo, Path(a.source), name=a.name, priority=PRIORITIES[a.priority], op_id=a.op_id)
        if a.wait:
            run_queue(repo, deadline_ns=now_ns() + int(a.timeout * 1e9), until_version=vid)
            if learn_cli is not None:
                with contextlib.suppress(Exception):          # стадії захисту для журналу навчання
                    learn_cli.L.collect(repo)
        view = maintain.version_view(repo, repo.cat.find_version(vid))
        code = _state_exit(view["state"]) if a.wait else Exit.OK
        if code == Exit.OK and view["skipped_files"]:
            code = Exit.PARTIAL
        return code, {"version": view, "waited": a.wait, "repeated_operation": repeated,
                      "learning": _learning_summary(repo, vid)}
    if cmd == "run":
        if a.daemon:
            return _daemon(a, repo)
        summary = run_queue(repo, wait=not a.once,
                            deadline_ns=None if a.timeout is None else now_ns() + int(a.timeout * 1e9))
        m = maintain.metrics(repo)
        code = Exit.ATTENTION if m["queue_attention_jobs"] else Exit.NOT_PROTECTED if m["queue_jobs"] else Exit.OK
        out = {"queue": summary, "metrics": m}
        if learn_cli is not None and not a.no_learn:
            out["learning"] = learn_cli.after_run(repo)       # навчання — після основного захисту
        return code, out
    if cmd == "list":
        rows = repo.cat.q("SELECT * FROM versions WHERE state!='DELETED' ORDER BY created_ns")
        return Exit.OK, {"versions": [maintain.version_view(repo, v) for v in rows]}
    if cmd == "find":
        return Exit.OK, {"find": ops.find(repo, version=a.ref, path=a.path, source=a.source, state=a.state,
                                          since=a.since, until=a.until, name=a.name, limit=a.limit)}
    if cmd == "queue":
        return Exit.OK, {"queue": ops.queue(repo, a.limit)}
    if cmd == "status":
        if a.version:
            view = maintain.version_view(repo, repo.cat.find_version(a.version))
            return _state_exit(view["state"]), {"version": view}
        m = ops.extended_metrics(repo)
        s = m["versions_by_state"]
        code = Exit.ATTENTION if s.get("DEGRADED") or m["queue_attention_jobs"] else \
            Exit.NOT_PROTECTED if m["under_replicated_versions"] else Exit.OK
        out = {"repo": str(repo.path), "metrics": m, "backends": [b["name"] for b in repo.cfg["backends"]]}
        if learn_cli is not None:
            try:
                ls = learn_cli.status(repo)
                out["learning"] = {"mode": ls["state"]["mode"], "active_model": ls["state"]["active_model"],
                                   "paused": ls["state"]["paused"], "disabled": ls["state"]["disabled"],
                                   "frozen": ls["state"]["frozen"], "errors_total": ls["learning_errors_total"]}
            except Exception as e:
                out["learning"] = {"error": type(e).__name__}
        return code, out
    if cmd == "verify":
        vid = repo.cat.find_version(a.version)["version_id"] if a.version else None
        out = maintain.verify(repo, vid)
        if learn_cli is not None:
            with contextlib.suppress(Exception):              # стадія «перевірено читанням» для журналу
                learn_cli.L.collect(repo)
        code = Exit.INTEGRITY if out["missing"] or out["corrupt"] or out["spool_corrupt"] else \
            Exit.ATTENTION if out["unreachable"] else Exit.OK
        return code, {"verify": out}
    if cmd == "restore":
        with _priority(repo, "restore"):
            return Exit.OK, {"restore": restore(repo, a.version, Path(a.dest), overwrite=a.overwrite, only=a.path)}
    if cmd == "drill":
        with _priority(repo, "restore"):
            out = ops.drill(repo, a.version)
        return (Exit.OK if out["ok"] else Exit.INTEGRITY), {"drill": out}
    if cmd == "repair":
        with _priority(repo, "repair"):
            out = maintain.repair(repo, all_backends=a.all_backends, run=not a.no_run, timeout=a.timeout)
        return (Exit.ATTENTION if out["versions_not_protected"] else Exit.OK), {"repair": out}
    if cmd == "key":
        if a.action == "status":
            return Exit.OK, {"keys": keyops.status(repo)}
        if a.action == "rotate":
            out = keyops.rotate_key(repo)
        elif a.action == "passphrase":
            out = keyops.change_key_passphrase(repo)
        elif a.action == "export":
            if not a.out:
                raise VaultError("E_USAGE", "потрібен --out", Exit.USAGE)
            return Exit.OK, {"keys": keyops.export_keys(repo, Path(a.out))}
        else:
            out = {"publish": keyops.publish(repo)}
        return keyops.publish_exit(out["publish"]), {"keys": out}
    if cmd == "health":
        out = ops.health(repo, {"max_queue_age_seconds": a.max_queue_age_seconds,
                                "max_protection_lag_seconds": a.max_protection_lag_seconds,
                                "max_verify_age_seconds": a.max_verify_age_seconds,
                                "max_drill_age_seconds": a.max_drill_age_seconds})
        return (Exit.OK if out["healthy"] else Exit.ATTENTION), {"health": out}
    if cmd == "metrics":
        if a.format == "json":
            return Exit.OK, {"metrics": ops.extended_metrics(repo)}
        text = ops.prometheus(repo)
        if a.out:
            write_durable(Path(a.out), [text.encode()])
            return Exit.OK, {"written": a.out}
        return Exit.OK, {"prometheus": text}
    out = maintain.gc(repo, apply=a.apply)
    return (Exit.ERROR if out["delete_errors"] else Exit.OK), {"gc": out}


def _human(data: dict, code: Exit) -> str:
    if "prometheus" in data:
        return data["prometheus"].rstrip("\n")
    lines = []
    for v in ([data["version"]] if "version" in data else data.get("versions", [])):
        copies = v["copies"] or {}
        lines.append(f"версія {v['version_id']}  {v['name']}  {v['created_at']}")
        lines.append(f"  стан: {v['state']}  рівень захисту: {v['protection_level']}")
        lines.append("  політику копій виконано: " + ("ТАК" if v["policy_satisfied"] else "НІ")
                     + (f" (об'єктів без {copies['min_copies']} копій: {copies['objects_under_replicated']} з "
                        f"{copies['objects_total']})" if copies else ""))
        lines.append(f"  файлів: {v['files']}  байтів: {v['bytes']}  пропущено: {v['skipped_files']}"
                     + (f"  помилка: {v['error']}" if v["error"] else ""))
    for key, value in data.items():
        if key in ("version", "versions", "waited", "repeated_operation", "rusage"):
            continue
        lines.append(f"{key}: " + (json.dumps(value, ensure_ascii=False, indent=2)
                                   if isinstance(value, dict | list) else str(value)))
    if data.get("waited") is False:
        lines.append("версію прийнято ЛОКАЛЬНО і поставлено в чергу; копії ще не підтверджені. "
                     "Запустіть: log3-vault run")
    lines.append(f"код завершення: {int(code)} ({code.name})")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):          # Windows: консоль може бути не UTF-8
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    a = _parser().parse_args(argv)
    try:
        code, data = _run(a)
        ok = True
    except VaultError as e:
        code, data, ok = e.exit_code, {"error": {"code": e.code, "message": e.message}}, False
    except BackendError as e:
        code, data, ok = Exit.ERROR, {"error": {"code": "E_BACKEND", "message": str(e)}}, False
    if os.environ.get("LOG3_VAULT_RUSAGE") == "1":
        from .procstat import self_usage
        data["rusage"] = self_usage()                         # CPU і пікова RSS цього процесу
    if a.json:
        print(json.dumps({"ok": ok, "command": a.command, "exit_code": int(code), "exit_name": code.name, **data},
                         ensure_ascii=False))
    elif ok:
        print(_human(data, code))
    else:
        print(f"помилка {data['error']['code']}: {data['error']['message']}\nкод завершення: {int(code)} "
              f"({code.name})", file=sys.stderr)
    return int(code)
