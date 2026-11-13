"""DEV-ЗАМІННИК моделі реплікації для CI — НЕ replication.py викладача.

Справжня модель (materials/replication.py) — чорна скринька, однакова для всіх
команд; у публічний репозиторій її не кладемо. Пайплайн бере її, якщо файл є
(pipeline.replication_class). Цей модуль має той самий інтерфейс —
Replication(scenario, seed), store(node, blob) -> bool, surviving_blobs(), stats() —
і ту саму поведінку бюджету: спроба, що вмістилась у глобальний бюджет і місткість
вузла, списується навіть тоді, коли вузол у стані BAD і копію втрачено; відхилена
до списання — ні. Відмови — ланцюг GOOD/BAD із середньою довжиною пачки.

synthetic_session — dev-журнал для CI, коли нема materials/session.csv.
"""
from __future__ import annotations

import random

from .records import Record


class Replication:
    def __init__(self, scenario: dict, seed: int):
        self.global_budget = int(scenario["global_replication_budget_bytes"])
        self.global_used = 0
        self._rng = random.Random(seed)
        self._nodes = {}
        for name, v in scenario["вузли"].items():
            loss, burst = float(v["failure_probability"]), float(v["mean_failure_burst"])
            p_bad = loss / (loss + burst * (1 - loss)) if loss > 0 else 0.0
            self._nodes[name] = {"cap": int(v["capacity_bytes"]), "p_bad": p_bad, "burst": burst,
                                 "used": 0, "left": 0, "blobs": []}

    def store(self, node: str, blob: bytes) -> bool:
        n = self._nodes[node]
        size = len(blob)
        if self.global_used + size > self.global_budget or n["used"] + size > n["cap"]:
            return False                              # відхилено до списання
        self.global_used += size
        n["used"] += size
        if n["left"] == 0 and self._rng.random() < n["p_bad"]:
            n["left"] = 1
            while n["burst"] > 1 and self._rng.random() > 1 / n["burst"]:
                n["left"] += 1
        if n["left"]:
            n["left"] -= 1
            return False                              # ресурс витрачено, копію втрачено
        n["blobs"].append(bytes(blob))
        return True

    def surviving_blobs(self) -> list[bytes]:
        return [b for n in self._nodes.values() for b in n["blobs"]]

    def stats(self) -> dict:
        return {"global_used_bytes": self.global_used, "global_budget_bytes": self.global_budget}


def synthetic_session(n: int, seed: int) -> list[Record]:
    """Dev-журнал: штатний фон і кілька інцидентів (частішають події, ростуть severity
    і metric, частіше critical). timestamp_ms унікальний і зростає."""
    rng = random.Random(seed)
    out: list[Record] = []
    t = 1_000
    hot = 0
    for _ in range(n):
        if hot == 0 and rng.random() < 0.01:
            hot = rng.randint(15, 40)
        t += rng.randint(5, 60) if hot else rng.randint(80, 400)
        out.append(Record(
            timestamp_ms=t,
            source_id=rng.choice((101, 102, 103, 104, 107, 112) if hot else (101, 102, 103)),
            event_code=rng.choice((500, 501, 503, 900) if hot else (100, 101, 102, 200)),
            metric_value=rng.randint(800, 5000) if hot else rng.randint(40, 120),
            category=rng.randint(0, 15) if hot else rng.choice((0, 1, 2)),
            severity=min(7, rng.randint(3, 7)) if hot else rng.choice((0, 0, 1, 1, 2, 3)),
            critical=int(rng.random() < (0.45 if hot else 0.08)),
            record_flags=rng.choice((0, 0, 0, 1)),
        ))
        hot = max(hot - 1, 0)
    return out
