"""ЗАГЛУШКА classify_records за docs/CONTRACTS.md («Заглушки»): усе — клас 0.

Власник — Artem (feat/perceptron): власний MLP з вікном історії, macro-F1 ≥ F1_min.
Навчити модель тут нема на чому — train.csv/valid.csv у materials/ ще нема.
Поки стоїть заглушка, абляція «без перцептрона» збігається з повною системою.
"""
from __future__ import annotations

from ..records import Record


def classify_records(records: list[Record]) -> list[int]:
    return [0] * len(records)
