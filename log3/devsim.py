"""DEV-ЗАМІННИК моделі реплікації — НЕ replication.py викладача.

Справжня модель (materials/replication.py, REPLICATION_SPEC.md) — чорна скринька,
однакова для всіх команд; її змінювати не можна, і її в нас поки нема. Цей модуль
лише дає пайплайну середовище з тими властивостями, які описує методичка (11.6,
11.44, 11.90), щоб інтеграцію можна було налагоджувати вже зараз:

  * вузли A, B, C: місткість, ланцюг Гілберта–Елліотта GOOD/BAD
    (p_good_to_bad, середня довжина пачки BAD = mean_burst), у BAD запис втрачається
    з імовірністю p_fail_bad;
  * глобальний бюджет байтів усіх спроб запису: спроба, що вмістилась у бюджет,
    списується навіть якщо вузол у BAD і копія втрачена; відхилена до списання
    (перевищення бюджету/місткості) у global_used_bytes не входить.

Підключення справжньої моделі — заміна pipeline.make_model, більше нічого.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from .records import Record


@dataclass
class _Node:
    capacity_bytes: int
    p_good_to_bad: float
    mean_burst: float
    p_fail_bad: float
    used: int = 0
    bad: bool = False
    stored: list[bytes] = field(default_factory=list)


class DevReplicationModel:
    def __init__(self, scenario: dict, seed: int):
        self.rng = random.Random(seed)
        self.budget = int(scenario["global_budget_bytes"])
        self.global_used_bytes = 0
        self.nodes = {name: _Node(**p) for name, p in scenario["nodes"].items()}

    def store(self, node: str, blob: bytes) -> bool:
        n = self.nodes[node]
        size = len(blob)
        if self.global_used_bytes + size > self.budget or n.used + size > n.capacity_bytes:
            return False                              # відхилено до списання
        self.global_used_bytes += size
        n.used += size
        if n.bad:
            n.bad = self.rng.random() >= 1.0 / max(n.mean_burst, 1.0)
        else:
            n.bad = self.rng.random() < n.p_good_to_bad
        if n.bad and self.rng.random() < n.p_fail_bad:
            return False                              # ресурс витрачено, копію втрачено
        n.stored.append(blob)
        return True

    def surviving(self) -> list[bytes]:
        return [b for n in self.nodes.values() for b in n.stored]


def synthetic_session(n: int, seed: int) -> list[Record]:
    """Dev-журнал, поки нема materials/session.csv: штатний фон і кілька інцидентів,
    де частішають події, ростуть severity і metric, частіше critical. timestamp_ms унікальний."""
    rng = random.Random(seed)
    out: list[Record] = []
    t = 1_700_000
    incident_left = 0
    for _ in range(n):
        if incident_left == 0 and rng.random() < 0.008:
            incident_left = rng.randint(15, 40)
        hot = incident_left > 0
        incident_left = max(incident_left - 1, 0)
        t += rng.randint(5, 60) if hot else rng.randint(80, 400)
        sev = min(7, rng.randint(3, 7) if hot else rng.choice((0, 0, 1, 1, 2, 3)))
        out.append(Record(
            timestamp_ms=t,
            source_id=rng.choice((1, 2, 3, 4, 7, 12) if hot else (1, 2, 3)),
            event_code=rng.choice((500, 501, 503, 900) if hot else (100, 101, 102, 200)),
            metric_value=rng.randint(800, 5000) if hot else rng.randint(40, 120),
            category=rng.randint(0, 15) if hot else rng.choice((0, 1, 2)),
            severity=sev,
            critical=int(rng.random() < (0.45 if hot else 0.06)),
            record_flags=rng.choice((0, 0, 0, 1)),
        ))
    return out
