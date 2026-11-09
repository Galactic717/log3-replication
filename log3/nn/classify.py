"""ЗАГЛУШКА classify_records за docs/CONTRACTS.md («Заглушки»): усе — клас 0.

Перцептрон (власний MLP з вікном історії, macro-F1 ≥ F1_min) ще не реалізовано:
train.csv/valid.csv у materials/ нема, вчити нема на чому.
Поки стоїть заглушка, абляція «без перцептрона» збігається з повною системою.
"""
from __future__ import annotations

from ..records import Record


def classify_records(records: list[Record]) -> list[int]:
    return [0] * len(records)
