"""Параметри задачі з team_config.json — жодних зашитих констант у коді.

На захисті змінюють умову (вузол вдвічі менш надійний, новий клас із вищою
цінністю, бюджет −30 %). Усе це зводиться до правки JSON або
dataclasses.replace(cfg, ...), без правки коду.

Конфіг викладача лежить у materials/team_config.json (локально, не в git: у ньому
пароль сеансу). Без нього береться config/team_config.dev.json — та сама схема з
dev-значеннями, щоб CI і тести працювали без пакета викладача. LOG3_CONFIG
перевизначає шлях.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MATERIALS = ROOT / "materials"
MATERIALS_CONFIG = MATERIALS / "team_config.json"
DEV_CONFIG = ROOT / "config" / "team_config.dev.json"

# Внутрішні ідентифікатори сценаріїв -> ключі в team_config.json
SCENARIO_KEYS = {"normal": "штатний", "degraded": "деградований", "emergency": "аварійний"}


@dataclass(frozen=True)
class Node:
    capacity_bytes: int
    failure_probability: float
    mean_failure_burst: float
    reliability_weight: float


@dataclass(frozen=True)
class Scenario:
    name: str                          # normal | degraded | emergency
    global_budget_bytes: int           # global_replication_budget_bytes
    nodes: dict[str, Node]
    max_unrecovered_critical: float    # поріг частки втрачених критичних подій

    @property
    def reliability(self) -> dict[str, float]:
        return {n: v.reliability_weight for n, v in self.nodes.items()}

    def as_model_dict(self) -> dict:
        """Секція сценарію у формі, яку приймає replication.Replication."""
        return {"global_replication_budget_bytes": self.global_budget_bytes,
                "вузли": {n: {"capacity_bytes": v.capacity_bytes,
                              "failure_probability": v.failure_probability,
                              "mean_failure_burst": v.mean_failure_burst,
                              "reliability_weight": v.reliability_weight}
                          for n, v in self.nodes.items()}}

    def with_node(self, node: str, **changes) -> Scenario:
        """Копія сценарію зі зміненими полями одного вузла (мікрозміни захисту)."""
        return replace(self, nodes={**self.nodes, node: replace(self.nodes[node], **changes)})


@dataclass(frozen=True)
class Config:
    team: str
    password: str                       # пароль сеансу для реплік
    replication_window_records: int     # розмір вікна реплікації
    max_batch_bytes: int                # межа однієї репліки на вузол (уся репліка)
    replication_budget_bits: int        # бюджет ДП в одному вікні W (біти)
    second_copy_factor: float           # кумулятивні: u(2) = v * f2
    third_copy_factor: float            # u(3) = v * f3
    priority: tuple[float, ...]         # цінність за номером класу; кількість класів = len(priority)
    class_names: tuple[str, ...]
    critical_bonus: float
    scenarios: dict[str, Scenario]
    model_seed: int                     # відкритий seed моделі реплікації
    alpha: float
    beta: float
    gamma: float
    t_ref: float                        # T = t_ref / (t_ref + t), с
    time_limit_s: float
    F1_min: float
    temporal_features: tuple[str, ...]  # мінімальний набір часових ознак (назви з конфігу)
    raw: dict = field(default_factory=dict, compare=False, repr=False)

    @property
    def n_classes(self) -> int:
        return len(self.priority)

    def scenario(self, name: str) -> Scenario:
        key = {v: k for k, v in SCENARIO_KEYS.items()}.get(name, name)
        if key not in self.scenarios:
            raise ValueError(f"сценарій {name!r}, очікується один із {list(self.scenarios)}")
        return self.scenarios[key]


def _scenario(name: str, d: dict, threshold: float) -> Scenario:
    nodes = {n: Node(int(v["capacity_bytes"]), float(v["failure_probability"]),
                     float(v["mean_failure_burst"]), float(v["reliability_weight"]))
             for n, v in d["вузли"].items()}
    return Scenario(name, int(d["global_replication_budget_bytes"]), nodes, float(threshold))


def config_from_dict(d: dict) -> Config:
    try:
        rep, q, clf = d["реплікація"], d["метрика_Q"], d["класифікатор"]
        names = tuple(clf["класи"][k] for k in sorted(clf["класи"], key=int))
        if sorted(int(k) for k in clf["класи"]) != list(range(len(names))):
            raise ValueError("team_config: номери класів мають іти 0..n-1 без пропусків")
        cfg = Config(
            team=str(d.get("команда", "")),
            password=str(d["пароль_сеансу"]),
            replication_window_records=int(rep["replication_window_records"]),
            max_batch_bytes=int(rep["max_batch_bytes"]),
            replication_budget_bits=int(rep["replication_budget_bits"]),
            second_copy_factor=float(rep["second_copy_factor"]),
            third_copy_factor=float(rep["third_copy_factor"]),
            priority=tuple(float(d["пріоритети_класів"][n]) for n in names),
            class_names=names,
            critical_bonus=float(d["critical_bonus"]),
            scenarios={k: _scenario(k, rep["сценарії"][ua], d["пороги_втрати_критичних_подій"][ua])
                       for k, ua in SCENARIO_KEYS.items()},
            model_seed=int(rep["seed_моделі_реплікації"]),
            alpha=float(q["alpha"]), beta=float(q["beta"]), gamma=float(q["gamma"]),
            t_ref=float(q["t_ref_секунд"]),
            time_limit_s=float(d["ліміт_часу_повного_прогону_секунд"]),
            F1_min=float(clf["F1_min"]),
            temporal_features=tuple(clf["мінімальні_часові_ознаки"]),
            raw=d,
        )
    except KeyError as e:
        raise ValueError(f"team_config: бракує ключа {e}") from None
    validate(cfg)
    return cfg


def validate(cfg: Config) -> None:
    """ValueError із зрозумілим повідомленням для некоректних параметрів задачі."""
    if abs(cfg.alpha + cfg.beta + cfg.gamma - 1.0) > 1e-9:
        raise ValueError("team_config: alpha + beta + gamma має дорівнювати 1")
    if min(cfg.alpha, cfg.beta, cfg.gamma) < 0 or cfg.t_ref <= 0:
        raise ValueError("team_config: ваги Q невід'ємні, t_ref > 0")
    if cfg.replication_window_records < 1 or cfg.max_batch_bytes < 1 or cfg.replication_budget_bits < 0:
        raise ValueError("team_config: вікно ≥ 1, max_batch_bytes ≥ 1, бюджет ДП ≥ 0")
    if not 1.0 <= cfg.second_copy_factor <= cfg.third_copy_factor:
        raise ValueError("team_config: потрібно 1 ≤ second_copy_factor ≤ third_copy_factor")
    if not cfg.priority or min(cfg.priority) < 0 or cfg.critical_bonus < 0:
        raise ValueError("team_config: пріоритети класів і critical_bonus невід'ємні, класів ≥ 1")
    if not cfg.password:
        raise ValueError("team_config: порожній пароль сеансу")
    for s in cfg.scenarios.values():
        if not s.nodes or s.global_budget_bytes < 0:
            raise ValueError(f"сценарій {s.name}: потрібні вузли й бюджет ≥ 0")
        for n, v in s.nodes.items():
            if v.capacity_bytes < 0 or not 0 <= v.failure_probability < 1 or v.mean_failure_burst < 1 \
                    or not 0 <= v.reliability_weight <= 1:
                raise ValueError(f"сценарій {s.name}, вузол {n}: некоректні параметри {v}")


def default_config_path() -> Path:
    if os.environ.get("LOG3_CONFIG"):
        return Path(os.environ["LOG3_CONFIG"])
    if MATERIALS_CONFIG.exists():
        return MATERIALS_CONFIG
    print(f"[log3] materials/team_config.json нема, використовую dev-конфіг {DEV_CONFIG.name}",
          file=sys.stderr)
    return DEV_CONFIG


def load_config(path: str | Path | None = None) -> Config:
    p = Path(path) if path else default_config_path()
    with open(p, encoding="utf-8") as f:
        return config_from_dict(json.load(f))
