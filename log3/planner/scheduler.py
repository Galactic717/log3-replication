"""Планувальник вікна: скільки копій дати кожному запису і на які вузли їх покласти.

Два кроки:
1. choose_copies — ДП (або жадібний) вирішує r_i ∈ {0..3} у межах бюджету бітів вікна.
2. place_copies  — розкладає копії по вузлах A, B, C з урахуванням правил:
       * копії одного запису — на РІЗНИХ вузлах (r=2 — два вузли, r=3 — усі три);
       * на кожен вузол один batch на вікно, кожен запис у batch не більше одного разу;
       * розмір batch (уся репліка) не більше max_batch_bytes.
Якщо вузли не вміщають усі копії, запис отримує менше копій (placed_i < r_i),
а не дублюється на тому самому вузлі.

ДП не бачить місткості вузлів і накладних витрат, тож r_i — верхня оцінка; розміщення —
евристика (не доведено оптимальна), її якість перевіряємо експериментально.
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
    batches: dict[str, list[int]]   # вузол -> індекси записів; у такому порядку batch пишуть на вузли
    copies: list[int]               # r_i, які призначив ДП (або жадібний)
    placed: list[int]               # скільки копій реально вмістилося на вузли
    lengths: dict[int, int] | None  # таблиця Гаффмана вікна (None без стиснення)


def choose_copies(records: list[Record], classes: list[int], cfg: Config, *, use_dp: bool = True,copies_cap: int = 3, second_copy_factor: float | None = None,compress: bool = True):
    """Крок 1. Повертає (r_i, таблиця Гаффмана вікна, ваги w_i, цінності v_i)."""
    lengths = code_lengths(pack_records(records)) if compress and records else None
    if lengths:
        weights = record_weights(records, lengths)
    else:
        weights = [8 * RECORD_SIZE] * len(records)          # без стиснення: 16 байтів = 128 бітів
    values = [base_value(c, r.critical, cfg) for c, r in zip(classes, records)]
    rmax = min(3, copies_cap, len(cfg.nodes))              # копій не більше, ніж вузлів
    f2 = cfg.second_copy_factor if second_copy_factor is None else second_copy_factor
    util = [row[:rmax + 1] for row in utility_table(values, f2, cfg.third_copy_factor)]
    cost = [[r * w for r in range(rmax + 1)] for w in weights]
    solve = solve_dp if use_dp else solve_greedy
    return solve(util, cost, cfg.replication_budget_bits), lengths, weights, values


def place_copies(copies: list[int], weights: list[int], values: list[float], nsym: int, cfg: Config,
                 *, compress: bool = True, no_compression_threshold: float | None = None) -> tuple[dict[str, list[int]], list[int]]:
    """Крок 2. Повертає (вузол -> індекси записів, скільки копій кожного запису розміщено)."""
    nodes = cfg.nodes
    members: dict[str, list[int]] = {n: [] for n in nodes}   # які записи вже в batch вузла
    bits = {n: 0 for n in nodes}                             # сума ваг записів у batch
    worth = {n: 0.0 for n in nodes}                          # сума цінностей записів у batch

    def replica_size(node: str, extra: int | None = None) -> int:
        """Точний розмір репліки вузла (за бажанням — разом із записом extra)."""
        count, total_bits = len(members[node]), bits[node]
        if extra is not None:
            count, total_bits = count + 1, total_bits + weights[extra]
        return MIN_SIZE + payload_size(count, nsym, total_bits, compress=compress,no_compression_threshold=no_compression_threshold)

    def node_score(node: str) -> float:
        """Надійність вузла × частка вільного місця в його batch: більше — краще."""
        free_share = 1 - replica_size(node) / cfg.max_batch_bytes
        return cfg.reliability_weight[node] * free_share

    placed = [0] * len(copies)
    # найцінніші записи розміщуємо першими: якщо місця забракне, недоотримають найменш важливі
    for i in sorted(range(len(copies)), key=lambda i: (-values[i], i)):
        for node in sorted(nodes, key=lambda n: (-node_score(n), n)):
            if placed[i] == copies[i]:
                break
            if replica_size(node, i) <= cfg.max_batch_bytes:   # кожен вузол — щонайбільше одна копія
                members[node].append(i)
                bits[node] += weights[i]
                worth[node] += values[i]
                placed[i] += 1
    # найцінніші batch пишемо першими: невдала спроба запису теж витрачає глобальний бюджет
    order = sorted((n for n in nodes if members[n]), key=lambda n: (-worth[n], n))
    return {n: sorted(members[n]) for n in order}, placed


def plan(records: list[Record], classes: list[int], cfg: Config, *, use_dp: bool = True,copies_cap: int = 3, second_copy_factor: float | None = None, compress: bool = True,no_compression_threshold: float | None = None) -> WindowPlan:
    """Повний план вікна: копії від ДП + їх розміщення по вузлах."""
    copies, lengths, weights, values = choose_copies(
        records, classes, cfg, use_dp=use_dp, copies_cap=copies_cap,
        second_copy_factor=second_copy_factor, compress=compress)
    batches, placed = place_copies(copies, weights, values, len(lengths or ()), cfg,compress=compress, no_compression_threshold=no_compression_threshold)
    return WindowPlan(batches, copies, placed, lengths)


def plan_window(records: list[Record], classes: list[int], cfg: Config, **kw) -> dict[str, list[int]]:
    """Контракт §6: вузол -> індекси записів вікна."""
    return plan(records, classes, cfg, **kw).batches