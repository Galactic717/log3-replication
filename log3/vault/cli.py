"""CLI файлового режиму: python -m log3.vault <команда> --repo <каталог> [--json]

Команди: init, backend, backup, run, list, status, verify, restore, repair, gc, recover,
serve-node, node-cert, learn (локальне навчання вибору стиснення). З --json кожна команда друкує
в stdout один JSON-об'єкт із полями ok, command, exit_code, exit_name; коди завершення стабільні
(errors.Exit).

backup без --wait лише приймає версію локально й ставить реплікацію в чергу (рівень
accepted_local, код 0). backup --wait сам обробляє чергу і повертає 0 тільки для рівня
protected. Секрети передаються змінними середовища або файлами, не аргументами.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from pathlib import Path

from . import __version__, maintain
from .backends.node_server import TOKEN_ENV, NodeServer, make_node_cert, server_tls
from .catalog import now_ns
from .cryptobox import PASSPHRASE_ENV, read_secret
from .errors import BackendError, Exit, VaultError
from .replicate import run_queue
from .repo import MIB, PRIORITIES, Repo, add_backend, make_backend, new_config
from .restore import restore

try:                                         # навчання — необов'язкова частина: без нього решта CLI працює
    from .learn import cli as learn_cli
except Exception:                            # pragma: no cover — лише якщо пакет навчання зламано
    learn_cli = None

REPO_ENV = "LOG3_VAULT_REPO"


def _parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--repo", default=os.environ.get(REPO_ENV), help=f"каталог сховища (або змінна {REPO_ENV})")
    common.add_argument("--json", action="store_true", help="машинозчитуваний вивід: один JSON-об'єкт у stdout")
    p = argparse.ArgumentParser(prog="python -m log3.vault", description=__doc__.split("\n\n")[0])
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="command", required=True)

    def cmd(name: str, text: str) -> argparse.ArgumentParser:
        return sub.add_parser(name, parents=[common], help=text, description=text)

    def backend_args(c: argparse.ArgumentParser) -> None:
        c.add_argument("--type", choices=("localdir", "node"), required=True)
        c.add_argument("--domain", required=True, help="домен відмов: диск, майданчик, провайдер")
        c.add_argument("--path", help="localdir: каталог на окремому диску")
        c.add_argument("--url", help="node: https://host:port")
        c.add_argument("--ca-file", help="node: закріплений сертифікат вузла (node.crt)")
        c.add_argument("--token-env", default=TOKEN_ENV, help="node: назва змінної середовища з токеном")

    c = cmd("init", "створити сховище")
    c.add_argument("--min-copies", type=int, default=2, help="жорсткий мінімум незалежних копій (типово 2)")
    c.add_argument("--chunk-size-mib", type=float, default=1.0)
    c.add_argument("--retention-days", type=float, default=30.0)
    c.add_argument("--keep-last", type=int, default=3)
    c = cmd("backend", "додати backend або показати налаштовані")
    c.add_argument("action", choices=("add", "list"))
    c.add_argument("name", nargs="?")
    c.add_argument("--type", choices=("localdir", "node"))
    c.add_argument("--domain")
    c.add_argument("--path")
    c.add_argument("--url")
    c.add_argument("--ca-file")
    c.add_argument("--token-env", default=TOKEN_ENV)
    c = cmd("backup", "прийняти версію файла або каталогу")
    c.add_argument("source")
    c.add_argument("--name", help="назва набору (типово ім'я джерела)")
    c.add_argument("--priority", choices=tuple(PRIORITIES), default="normal")
    c.add_argument("--op-id", help="ідентифікатор операції: повтор із ним не створює другої версії")
    c.add_argument("--wait", action="store_true", help="обробляти чергу й чекати рівня protected")
    c.add_argument("--timeout", type=float, default=3600.0, help="секунд для --wait")
    c = cmd("run", "обробити чергу реплікації")
    c.add_argument("--once", action="store_true", help="лише завдання, час яких настав; не чекати повторів")
    c.add_argument("--timeout", type=float, help="секунд; типово до спорожнення черги")
    c.add_argument("--no-learn", action="store_true", help="не виконувати крок циклу навчання після черги")
    cmd("list", "перелік версій")
    c = cmd("status", "стан сховища з показниками або стан однієї версії")
    c.add_argument("version", nargs="?")
    c = cmd("verify", "перечитати й перевірити копії на backend-ах")
    c.add_argument("version", nargs="?")
    c = cmd("restore", "відновити версію в каталог")
    c.add_argument("version", help="ідентифікатор, його префікс або latest")
    c.add_argument("dest")
    c.add_argument("--path", help="лише цей файл або підкаталог версії")
    c.add_argument("--overwrite", action="store_true")
    c = cmd("repair", "прибрати наслідки аварій і довести копії до політики")
    c.add_argument("--all-backends", action="store_true", help="використати всі backend-и, а не лише цільові")
    c.add_argument("--no-run", action="store_true", help="лише поставити завдання, чергу не обробляти")
    c.add_argument("--timeout", type=float)
    c = cmd("gc", "прибрати версії після retention і непотрібні об'єкти (без --apply лише звіт)")
    c.add_argument("--apply", action="store_true")
    c = cmd("recover", "відтворити сховище з backend-а після втрати локального диска")
    c.add_argument("name", help="назва backend-а")
    backend_args(c)
    c = cmd("serve-node", "запустити віддалений вузол зберігання (HTTPS)")
    c.add_argument("--data", required=True)
    c.add_argument("--cert", required=True)
    c.add_argument("--key", required=True)
    c.add_argument("--host", default="0.0.0.0")
    c.add_argument("--port", type=int, default=8443)
    c.add_argument("--quota-bytes", type=int, default=0, help="0 = без квоти")
    c = cmd("node-cert", "створити самопідписаний сертифікат вузла")
    c.add_argument("--dir", required=True)
    c.add_argument("--host", action="append", required=True, help="ім'я або IP вузла; можна кілька")
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


def _repo(a) -> Repo:
    if not a.repo:
        raise VaultError("E_USAGE", f"вкажіть --repo або змінну {REPO_ENV}", Exit.USAGE)
    return Repo(Path(a.repo))


def _state_exit(state: str) -> Exit:
    return {"PROTECTED": Exit.OK, "DEGRADED": Exit.ATTENTION, "FAILED": Exit.ERROR,
            "DELETED": Exit.NOT_FOUND}.get(state, Exit.NOT_PROTECTED)


def _run(a) -> tuple[Exit, dict]:
    cmd = a.command
    if cmd == "node-cert":
        crt, key = make_node_cert(Path(a.dir), a.host)
        return Exit.OK, {"cert": str(crt), "key": str(key)}
    if cmd == "serve-node":
        server = NodeServer((a.host, a.port), Path(a.data), read_secret(TOKEN_ENV),
                            server_tls(Path(a.cert), Path(a.key)), a.quota_bytes)
        print(f"log3 node: https://{a.host}:{server.server_address[1]} дані {a.data}", file=sys.stderr, flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        return Exit.OK, {}
    if cmd == "init":
        if not a.repo:
            raise VaultError("E_USAGE", f"вкажіть --repo або змінну {REPO_ENV}", Exit.USAGE)
        cfg = new_config(min_copies=a.min_copies, chunk_size=int(a.chunk_size_mib * MIB))
        cfg["policy"].update(retention_days=a.retention_days, keep_last=a.keep_last)
        repo = Repo.create(Path(a.repo), cfg, read_secret(PASSPHRASE_ENV, prompt="Нова парольна фраза сховища: "))
        return Exit.OK, {"repo": str(repo.path), "repo_id": cfg["repo_id"], "key_id": repo.cfg["crypto"]["key_id"],
                         "min_copies": a.min_copies}
    if cmd == "recover":
        if not a.repo:
            raise VaultError("E_USAGE", f"вкажіть --repo або змінну {REPO_ENV}", Exit.USAGE)
        spec = _spec(a)
        if spec["type"] == "localdir":
            spec["path"] = str(Path(spec["path"]).absolute())
        return Exit.OK, maintain.recover(Path(a.repo), spec, make_backend(spec))
    repo = _repo(a)
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
    if cmd == "status":
        if a.version:
            view = maintain.version_view(repo, repo.cat.find_version(a.version))
            return _state_exit(view["state"]), {"version": view}
        m = maintain.metrics(repo)
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
    if cmd == "repair":
        with _priority(repo, "repair"):
            out = maintain.repair(repo, all_backends=a.all_backends, run=not a.no_run, timeout=a.timeout)
        return (Exit.ATTENTION if out["versions_not_protected"] else Exit.OK), {"repair": out}
    out = maintain.gc(repo, apply=a.apply)
    return (Exit.ERROR if out["delete_errors"] else Exit.OK), {"gc": out}


def _human(data: dict, code: Exit) -> str:
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
    for key in ("metrics", "verify", "restore", "repair", "gc", "queue", "backends", "learning", "learning_config"):
        if key in data:
            lines.append(f"{key}: " + json.dumps(data[key], ensure_ascii=False, indent=2))
    for key in ("repo", "repo_id", "key_id", "cert", "key", "objects"):
        if key in data and not isinstance(data[key], dict | list):
            lines.append(f"{key}: {data[key]}")
    if data.get("waited") is False:
        lines.append("версію прийнято ЛОКАЛЬНО і поставлено в чергу; копії ще не підтверджені. "
                     "Запустіть: python -m log3.vault run --repo ...")
    lines.append(f"код завершення: {int(code)} ({code.name})")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
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
