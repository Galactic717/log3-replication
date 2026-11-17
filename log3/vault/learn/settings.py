"""Налаштування навчання (config.json -> "learning") і стан життєвого циклу (<repo>/learn/state.json).

Секретів тут нема. Відсутня секція означає типові значення: нова й наявна інсталяція працюють
на базовій політиці й локально збирають вимірювання в межах малих лімітів. Некоректна секція
не зупиняє backup: навчання вимикається, а причина видна в `learn status`.

Стан:
  mode            baseline | shadow | active — хто приймає робочі рішення
  paused          проби, навчання й shadow-висновок зупинено; журнал рішень і активна модель працюють
  disabled        навчання вимкнено повністю: лише базова політика, без ознак, журналу й проб
  frozen          оновлення моделей заморожено (drift, сплеск змін, підозріла телеметрія)
  active_model    модель робочих рішень у режимі active
  candidate_model модель у shadow / на оцінюванні
  last_good_model остання модель, що справно працювала в active (ціль rollback)
Перемикання — атомарний запис цілого файла (tmp -> fsync -> rename). Пошкоджений файл читається
як типовий стан baseline; помилка фіксується, копіювання не зупиняється.
"""
from __future__ import annotations

import copy
import json
import os
import time
from pathlib import Path

from ..errors import Exit, VaultError
from ..fsutil import write_durable
from .actions import DEFAULT_ACTIONS, DEFAULT_CPU_SECOND_EQUIV_BYTES, parse_actions

DEFAULTS = {
    "enabled": True,
    "actions": list(DEFAULT_ACTIONS),
    "cost": {"cpu_second_equiv_bytes": DEFAULT_CPU_SECOND_EQUIV_BYTES, "copies": None},
    "probe": {"rate": 0.1, "max_chunks_per_operation": 64, "max_cpu_seconds_per_operation": 5.0,
              "max_cpu_fraction": 0.5, "max_queue_fill": 0.5, "min_free_factor": 2.0, "max_load_per_cpu": 1.5},
    "train": {"max_rows": 20000, "max_seconds": 300.0, "max_memory_mib": 256, "min_interval_seconds": 3600.0,
              "min_rows": 200, "min_groups": 10, "nice": 10, "kinds": ["costtable", "ridge", "mlp"],
              "search_budget": 16, "seed": 0},
    "gate": {"min_relative_gain": 0.02, "min_shadow_decisions": 100, "max_feature_ns_per_chunk": 3_000_000,
             "max_inference_ns_per_chunk": 1_000_000, "max_protect_latency_increase": 0.25, "bootstrap": 300},
    "drift": {"psi_freeze": 0.3, "check_every_chunks": 64, "min_chunks": 64, "high_entropy_jump": 0.4,
              "rewrite_fraction": 0.6},
    "journal": {"max_decisions": 200000},
}
MODES = ("baseline", "shadow", "active")
STATE_FORMAT = "log3-learn-state"
STATE_VERSION = 1
PRIORITY_MAX_AGE = 6 * 3600


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = _merge(out[k], v) if isinstance(out.get(k), dict) and isinstance(v, dict) else v
    return out


def settings(cfg: dict) -> dict:
    """Типові значення + секція learning з config.json, із перевіркою меж."""
    s = _merge(DEFAULTS, cfg.get("learning") or {})

    def need(ok: bool, msg: str) -> None:
        if not ok:
            raise VaultError("E_CONFIG", f"learning: {msg}", Exit.POLICY)
    try:
        acts = parse_actions(s["actions"])
        comp = cfg["compression"]
        default = f"{comp['codec']}:{comp['level'] if comp['codec'] != 'store' else 0}"
        s["actions"] = [a.key for a in parse_actions([*s["actions"], default])]   # дія базової політики теж міряється
        p, t, g = s["probe"], s["train"], s["gate"]
        need(isinstance(s["enabled"], bool), "enabled — true/false")
        need(0.0 <= p["rate"] <= 1.0, "probe.rate у межах 0..1")
        need(p["max_chunks_per_operation"] >= 0 and p["max_cpu_seconds_per_operation"] >= 0, "ліміти проб >= 0")
        need(0 < p["max_cpu_fraction"] <= 4 and 0 < p["max_queue_fill"] <= 1, "probe: частки в межах")
        need(t["max_rows"] >= 50 and t["max_seconds"] > 0 and t["max_memory_mib"] >= 16, "train: ліміти")
        need(set(t["kinds"]) <= {"costtable", "ridge", "mlp"} and t["kinds"], "train.kinds: costtable|ridge|mlp")
        need(g["min_relative_gain"] >= 0 and g["bootstrap"] >= 50, "gate: межі")
        need(s["cost"]["cpu_second_equiv_bytes"] >= 0, "cost.cpu_second_equiv_bytes >= 0")
        need(s["journal"]["max_decisions"] >= 1000, "journal.max_decisions >= 1000")
        need(len(acts) >= 2, "потрібна хоча б одна дія, крім store")
    except (KeyError, TypeError, ValueError) as e:
        raise VaultError("E_CONFIG", f"learning: некоректна секція ({type(e).__name__}: {e})", Exit.POLICY) from None
    return s


class LearnDir:
    """Розкладка <repo>/learn/."""

    def __init__(self, repo_path: Path):
        self.root = Path(repo_path) / "learn"
        self.models = self.root / "models"
        self.journal = self.root / "journal.sqlite"
        self.state = self.root / "state.json"

    def priority_marker(self, kind: str) -> Path:
        return self.root / f"priority-{kind}-{os.getpid()}.marker"


def default_state() -> dict:
    return {"format": STATE_FORMAT, "format_version": STATE_VERSION, "mode": "baseline", "paused": False,
            "disabled": False, "frozen": False, "freeze_reason": None, "active_model": None, "candidate_model": None,
            "last_good_model": None, "last_train_ns": 0, "updated_ns": 0, "history": []}


def load_state(d: LearnDir) -> tuple[dict, str | None]:
    """(стан, помилка). Відсутній файл — холодний старт без помилки; пошкоджений — baseline з помилкою."""
    if not d.state.exists():
        return default_state(), None
    try:
        st = json.loads(d.state.read_bytes())
        if st.get("format") != STATE_FORMAT or st.get("format_version") != STATE_VERSION or st.get("mode") not in MODES:
            raise ValueError("невідомий формат або режим")
        return _merge(default_state(), st), None
    except (OSError, ValueError, TypeError, AttributeError) as e:
        return default_state(), f"state.json не читається ({type(e).__name__}); працює базова політика"


def save_state(d: LearnDir, st: dict, *, event: str, detail: str = "") -> dict:
    st = dict(st)
    st["updated_ns"] = time.time_ns()
    st["history"] = (st.get("history") or [])[-49:] + [{"ts_ns": st["updated_ns"], "event": event, "mode": st["mode"],
                                                         "active_model": st["active_model"], "detail": detail}]
    d.root.mkdir(parents=True, exist_ok=True)
    write_durable(d.state, [json.dumps(st, ensure_ascii=False, indent=1).encode("utf-8")])
    return st


# ---- пріоритет restore / repair над навчанням ----------------------------------------------------
class priority:
    """Позначка «іде restore/repair»: навчання й проби поступаються. Позначку прибирає finally;
    після аварії процесу її ігнорують за віком або мертвим pid."""

    def __init__(self, repo_path: Path, kind: str):
        self.d, self.kind, self.path = LearnDir(repo_path), kind, None

    def __enter__(self):
        try:
            self.d.root.mkdir(parents=True, exist_ok=True)
            self.path = self.d.priority_marker(self.kind)
            self.path.write_text(str(time.time()), encoding="ascii")
        except OSError:
            self.path = None                               # позначка — підказка; її збій не блокує restore
        return self

    def __exit__(self, *exc):
        if self.path is not None:
            self.path.unlink(missing_ok=True)
        return False


def _alive(pid: int) -> bool:
    if os.name != "posix":
        return True                                        # на Windows покладаємось лише на вік позначки
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def priority_active(repo_path: Path) -> list[str]:
    """Які restore/repair зараз ідуть (за живими свіжими позначками)."""
    d, out = LearnDir(repo_path), []
    if not d.root.is_dir():
        return out
    for p in d.root.glob("priority-*.marker"):
        try:
            _, kind, pid = p.stem.split("-", 2)
            age = time.time() - float(p.read_text(encoding="ascii") or 0)
        except (ValueError, OSError):
            continue
        if age < PRIORITY_MAX_AGE and _alive(int(pid)):
            out.append(kind)
    return sorted(out)
