"""Команда `python -m log3.vault learn <дія>`: стан, навчання, оцінювання, режими, пауза, rollback.

    status                      режим, моделі, повнота журналу, стадії, overhead, події
    config [--set ключ=JSON]    ефективна конфігурація learning (з перевіркою перед записом)
    collect                     перенести затримані стадії захисту з каталогу в журнал
    train [--force]             навчити й відібрати кандидата (gate); --force ігнорує інтервал
    evaluate                    перевірити кандидата в shadow на свіжих пробах
    promote                     увімкнути перевіреного кандидата (active), попередня -> last_good
    rollback [--reason R]       до останньої справної моделі або бази
    mode baseline|shadow        ручне перемикання (active — лише через promote)
    pause | resume              зупинити/поновити проби, навчання й shadow
    disable | enable            повністю вимкнути/увімкнути навчання
    freeze --reason R | unfreeze [--accept-window]
    reset --yes                 холодний старт: журнал, моделі й стан навчання видаляються
    auto                        один крок циклу (так само викликається після run)
"""
from __future__ import annotations

import json

import numpy as np

from ..errors import Exit, VaultError
from . import lifecycle as L
from .journal import Journal
from .settings import LearnDir, load_state, priority_active, settings

ACTIONS = ("status", "config", "collect", "train", "evaluate", "promote", "rollback", "mode", "pause", "resume",
           "disable", "enable", "freeze", "unfreeze", "reset", "auto")


def add_parser(sub, common) -> None:
    c = sub.add_parser("learn", parents=[common], help="локальне навчання вибору стиснення",
                       description=__doc__.split("\n\n")[0])
    c.add_argument("action", choices=ACTIONS)
    c.add_argument("value", nargs="?", help="для mode: baseline | shadow")
    c.add_argument("--force", action="store_true", help="train: не чекати min_interval_seconds")
    c.add_argument("--reason", default="вручну")
    c.add_argument("--accept-window", action="store_true",
                   help="unfreeze: визнати карантинні операції нормою (вони потраплять у навчання)")
    c.add_argument("--set", action="append", default=[], metavar="КЛЮЧ=JSON", help="config: напр. probe.rate=0.05")
    c.add_argument("--yes", action="store_true", help="reset: підтвердження")


def status(repo) -> dict:
    d = LearnDir(repo.path)
    st, state_error = load_state(d)
    out: dict = {"state": {k: st[k] for k in ("mode", "paused", "disabled", "frozen", "freeze_reason",
                                               "active_model", "candidate_model", "last_good_model",
                                               "candidate_eval")},
                 "state_error": state_error, "priority_active": priority_active(repo.path)}
    try:
        s = settings(repo.cfg)
        out["config"] = {"enabled": s["enabled"], "actions": s["actions"], "cost": s["cost"],
                         "probe_rate": s["probe"]["rate"], "auto": s["auto"]}
    except VaultError as e:
        out["config_error"] = e.message
    out["learning_errors_total"] = repo.cat.scalar(
        "SELECT COALESCE(SUM(value),0) FROM events WHERE kind='learning_errors'")
    if d.models.is_dir():
        out["models"] = sorted(p.name for p in d.models.iterdir() if p.is_dir() and not p.name.startswith("."))
    if not d.journal.exists():
        return out | {"journal": None}
    try:
        j = Journal(d.journal)
    except Exception as e:                                   # пошкоджений журнал — видно тут, backup не зачіпає
        return out | {"journal_error": f"{type(e).__name__}: {e}"[:300]}
    try:
        ops = j.q("SELECT * FROM operations ORDER BY finished_ns DESC LIMIT 20")
        chunks = sum(o["chunks"] for o in ops) or 1
        raw = sum(o["raw_bytes"] for o in ops) or 1
        out["journal"] = {**j.completeness(), "counters": j.counters(), "stages": j.stage_counts(),
                          "operations": j.scalar("SELECT COUNT(*) FROM operations"),
                          "quarantined_operations": j.scalar("SELECT COUNT(*) FROM operations WHERE quarantine=1")}
        out["overhead_last_20_operations"] = {
            "chunks": chunks, "feature_ns_per_chunk": sum(o["feature_ns"] for o in ops) / chunks,
            "inference_ns_per_chunk": sum(o["inference_ns"] for o in ops) / chunks,
            "probe_cpu_seconds_per_gib": sum(o["probe_cpu_ns"] for o in ops) / 1e9 / (raw / 2**30),
            "encode_cpu_seconds_per_gib": sum(o["encode_cpu_ns"] for o in ops) / 1e9 / (raw / 2**30),
            "stored_to_raw": sum(o["stored_bytes"] for o in ops) / raw,
            "fallbacks": sum(o["fallbacks"] for o in ops)}
        out["recent_events"] = [dict(r) for r in j.q("SELECT ts_ns, kind, model_id, detail FROM lifecycle "
                                                      "ORDER BY id DESC LIMIT 10")]
    finally:
        j.close()
    return out


def _set(repo, items: list[str]) -> dict:
    learning = json.loads(json.dumps(repo.cfg.get("learning") or {}))
    for item in items:
        key, sep, raw = item.partition("=")
        if not sep or not key:
            raise VaultError("E_USAGE", f"очікується ключ=JSON, отримано {item!r}", Exit.USAGE)
        try:
            value = json.loads(raw)
        except ValueError:
            value = raw
        node = learning
        *path, last = key.split(".")
        for p in path:
            node = node.setdefault(p, {})
        node[last] = value
    settings({**repo.cfg, "learning": learning})              # перевірка ДО запису
    repo.cfg["learning"] = learning
    repo.save_config()
    return learning


def run(a, repo) -> tuple[Exit, dict]:
    act = a.action
    if act == "status":
        return Exit.OK, {"learning": status(repo)}
    if act == "config":
        if a.set:
            _set(repo, a.set)
        return Exit.OK, {"learning_config": settings(repo.cfg)}
    if act == "collect":
        return Exit.OK, {"learning": L.collect(repo)}
    if act == "train":
        rep = L.train(repo, force=a.force)
        return Exit.OK, {"learning": _jsonable(rep)}
    if act == "evaluate":
        rep = L.evaluate(repo)
        code = Exit.OK if rep.get("passed") or rep["status"] in ("collecting", "no_candidate") else Exit.ATTENTION
        return code, {"learning": _jsonable(rep)}
    if act == "promote":
        return Exit.OK, {"learning": L.promote(repo)}
    if act == "rollback":
        return Exit.OK, {"learning": L.rollback(repo, a.reason)}
    if act == "mode":
        if a.value not in ("baseline", "shadow", "active"):
            raise VaultError("E_USAGE", "learn mode baseline|shadow", Exit.USAGE)
        return Exit.OK, {"learning": L.set_mode(repo, a.value)}
    if act in ("pause", "resume"):
        return Exit.OK, {"learning": L.set_flags(repo, act, paused=act == "pause")}
    if act in ("disable", "enable"):
        return Exit.OK, {"learning": L.set_flags(repo, act, disabled=act == "disable")}
    if act == "freeze":
        return Exit.OK, {"learning": L.set_flags(repo, "freeze", frozen=True, freeze_reason=a.reason)}
    if act == "unfreeze":
        return Exit.OK, {"learning": L.unfreeze(repo, accept_window=a.accept_window)}
    if act == "reset":
        if not a.yes:
            raise VaultError("E_USAGE", "reset видаляє журнал і моделі навчання; підтвердіть --yes", Exit.USAGE)
        return Exit.OK, {"learning": L.reset(repo)}
    return Exit.OK, {"learning": _jsonable(L.auto(repo))}


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def after_run(repo) -> dict:
    """Крок циклу після обробки черги; збій навчання не змінює результату run."""
    try:
        return _jsonable(L.auto(repo))
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"[:300]}
