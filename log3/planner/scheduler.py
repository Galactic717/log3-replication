"""Розміщення копій по вузлах (FORMAT_SPEC_P3 §3.5, методичка 11.40–11.44).

Модуль реалізував голова (Galactic717); сигнатури — з контракту §6.

ДП знає лише бюджет вікна W, а не місткість вузлів, тож його r_i — верхня оцінка.
Жадібне розміщення:
  * записи — за спаданням цінності v_i (недоотримують місця найменш важливі);
  * копії одного запису — на різних вузлах; вузол обирається за
    reliability_weight × частка вільного місця в batch;
  * на вузол один batch на вікно, batch ≤ max_batch_bytes (уся репліка: 43 + payload + 32);
  * розмір batch рахується тим самим payload_size, що й у build_payload, тож оцінка = факт.
Це евристика, не доведено оптимальний алгоритм.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..config import Config
from ..huffman import code_lengths
from ..payload import payload_size
from ..records import RECORD_SIZE, Record, pack_records
from ..replica import MIN_SIZE
from .knapsack import base_value, record_weights, solve_dp, solve_greedy, utility_table


@dataclass(frozen=True)
class WindowPlan:
    batches: dict[str, list[int]]   # вузол -> індекси записів; порядок ключів = порядок store()
    copies: list[int]               # r_i від ДП/жадібного (призначено)
    placed: list[int]               # скільки копій реально розміщено
    lengths: dict[int, int] | None  # таблиця Гаффмана вікна (None без стиснення)


def choose_copies(records: list[Record], classes: list[int], cfg: Config, *, use_dp: bool = True,
                  copies_cap: int = 3, second_copy_factor: float | None = None,
                  compress: bool = True):
    """(r_i, таблиця вікна, ваги w_i, цінності v_i)."""
    raw = pack_records(records)
    lengths = code_lengths(raw) if compress and records else None
    weights = record_weights(records, lengths) if lengths else [8 * RECORD_SIZE] * len(records)
    values = [base_value(c, r.critical, cfg) for c, r in zip(classes, records)]
    rmax = min(3, copies_cap, len(cfg.nodes))
    f2 = cfg.second_copy_factor if second_copy_factor is None else second_copy_factor
    util = [row[:rmax + 1] for row in utility_table(values, f2, cfg.third_copy_factor)]
    cost = [[r * w for r in range(rmax + 1)] for w in weights]
    solve = solve_dp if use_dp else solve_greedy
    return solve(util, cost, cfg.replication_budget_bits), lengths, weights, values


def place_copies(copies: list[int], weights: list[int], values: list[float], nsym: int, cfg: Config,
                 *, compress: bool = True, no_compression_threshold: float | None = None
                 ) -> tuple[dict[str, list[int]], list[int]]:
    rel = cfg.reliability_weight
    batch: dict[str, list[int]] = {n: [] for n in cfg.nodes}
    bits = {n: 0 for n in cfg.nodes}
    value = {n: 0.0 for n in cfg.nodes}

    def size(node: str, extra: int | None = None) -> int:
        n = len(batch[node]) + (extra is not None)
        b = bits[node] + (weights[extra] if extra is not None else 0)
        return MIN_SIZE + payload_size(n, nsym, b, compress=compress,
                                       no_compression_threshold=no_compression_threshold)

    placed = [0] * len(copies)
    for i in sorted(range(len(copies)), key=lambda i: (-values[i], i)):
        if not copies[i]:
            continue
        free = sorted(cfg.nodes, key=lambda n: (-rel[n] * (1 - size(n) / cfg.max_batch_bytes), n))
        for node in free:
            if placed[i] == copies[i]:
                break
            if size(node, i) <= cfg.max_batch_bytes:
                batch[node].append(i)
                bits[node] += weights[i]
                value[node] += values[i]
                placed[i] += 1
    # Найцінніші batch пишемо першими: невдала спроба теж з'їдає глобальний бюджет.
    order = sorted((n for n in cfg.nodes if batch[n]), key=lambda n: (-value[n], n))
    return {n: sorted(batch[n]) for n in order}, placed


def plan(records: list[Record], classes: list[int], cfg: Config, *, use_dp: bool = True,
         copies_cap: int = 3, second_copy_factor: float | None = None, compress: bool = True,
         no_compression_threshold: float | None = None) -> WindowPlan:
    copies, lengths, weights, values = choose_copies(
        records, classes, cfg, use_dp=use_dp, copies_cap=copies_cap,
        second_copy_factor=second_copy_factor, compress=compress)
    batches, placed = place_copies(copies, weights, values, len(lengths or ()), cfg,
                                   compress=compress, no_compression_threshold=no_compression_threshold)
    return WindowPlan(batches, copies, placed, lengths)


def plan_window(records: list[Record], classes: list[int], cfg: Config, **kw) -> dict[str, list[int]]:
    """Контракт §6: вузол -> індекси записів."""
    return plan(records, classes, cfg, **kw).batches
