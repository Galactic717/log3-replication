"""Параметри задачі з team_config.json — жодних зашитих констант у коді.

На захисті змінюють умову (вузол вдвічі менш надійний, новий клас із вищою
цінністю, бюджет −30 %). Усе це має зводитись до правки JSON або
dataclasses.replace(cfg, ...), без правки коду.

Справжній team_config.json лежить у materials/ (локально, не в git). Поки його
нема, LOG3_CONFIG або config/team_config.dev.json — dev-значення, див. DECISIONS.
Назви ключів звірити з реальним файлом, коли він з'явиться.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MATERIALS_CONFIG = ROOT / "materials" / "team_config.json"
DEV_CONFIG = ROOT / "config" / "team_config.dev.json"


@dataclass(frozen=True)
class Config:
    replication_window_records: int     # розмір вікна реплікації (у базовому комплекті 48)
    max_batch_bytes: int                # межа одного batch на вузол (уся репліка, див. DECISIONS)
    replication_budget_bits: int        # бюджет ДП в одному вікні W (біти)
    priority: tuple[float, ...]         # базова цінність за класом; кількість класів = len(priority)
    critical_bonus: float
    second_copy_factor: float           # кумулятивні: u(2) = v * f2
    third_copy_factor: float            # u(3) = v * f3
    reliability_weight: dict[str, float]  # вузол -> оцінка надійності; кількість вузлів = len(...)
    alpha: float
    beta: float
    gamma: float
    t_ref: float = 1.0                  # опорний час для T = t_ref / (t_ref + t), с
    F1_min: float | None = None
    temporal_features: tuple[str, ...] = ()
    password: str = "log3-dev"          # пароль сеансу для власних реплік
    raw: dict = field(default_factory=dict, compare=False, hash=False, repr=False)

    @property
    def n_classes(self) -> int:
        return len(self.priority)

    @property
    def nodes(self) -> list[str]:
        return list(self.reliability_weight)


def _priority(v) -> tuple[float, ...]:
    # список [p0, p1, p2] або словник {"0": p0, ...} — порядок за номером класу
    if isinstance(v, dict):
        return tuple(float(v[k]) for k in sorted(v, key=int))
    return tuple(float(x) for x in v)


def _reliability(d: dict) -> dict[str, float]:
    # {"A": 0.9, ...} на верхньому рівні або "nodes": {"A": {"reliability_weight": 0.9, ...}}
    if "reliability_weight" in d:
        return {k: float(v) for k, v in d["reliability_weight"].items()}
    nodes = d["nodes"]
    return {k: float(v["reliability_weight"] if isinstance(v, dict) else v) for k, v in nodes.items()}


def config_from_dict(d: dict) -> Config:
    try:
        cfg = Config(
            replication_window_records=int(d["replication_window_records"]),
            max_batch_bytes=int(d["max_batch_bytes"]),
            replication_budget_bits=int(d["replication_budget_bits"]),
            priority=_priority(d["priority"]),
            critical_bonus=float(d["critical_bonus"]),
            second_copy_factor=float(d["second_copy_factor"]),
            third_copy_factor=float(d["third_copy_factor"]),
            reliability_weight=_reliability(d),
            alpha=float(d["alpha"]),
            beta=float(d["beta"]),
            gamma=float(d["gamma"]),
            t_ref=float(d.get("t_ref", 1.0)),
            F1_min=None if d.get("F1_min") is None else float(d["F1_min"]),
            temporal_features=tuple(d.get("temporal_features", ())),
            password=str(d.get("password", "log3-dev")),
            raw=d,
        )
    except KeyError as e:
        raise ValueError(f"team_config: бракує ключа {e}") from None
    if abs(cfg.alpha + cfg.beta + cfg.gamma - 1.0) > 1e-9:
        raise ValueError("team_config: alpha + beta + gamma має дорівнювати 1")
    return cfg


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
