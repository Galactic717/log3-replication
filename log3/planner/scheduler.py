"""Планувальник вікна: скільки копій дати кожному запису і на які вузли їх покласти.

Дві окремі задачі (методичка 11.88 — не змішувати):
1. choose_copies — ДП (або жадібний baseline) обирає r_i ∈ {0..3} під бюджет бітів вікна
   (FORMAT_SPEC §3.5: таблиця Гаффмана вікна → w_i, v_i, u_i(r), cost_i(r) = r·w_i).
2. place_copies  — жадібно розкладає призначені копії по вузлах (§3.4):
       * копії одного запису — на РІЗНИХ вузлах; на вузол один batch на вікно;
       * розмір batch (уся репліка) ≤ max_batch_bytes і ≤ залишкової місткості вузла;
       * сумарні байти вікна ≤ allowance — частки глобального бюджету на це вікно;
       * наступною береться копія з найбільшим приростом очікуваної відновлюваності
         на байт: v_i · rel_n · Π(1 − rel_m по вузлах, де копія вже є) / Δбайтів.
   ДП не бачить місткості вузлів і накладних витрат, тож r_i — верхня оцінка (11.40):
   що не вмістилось, не дублюється на тому самому вузлі, а просто не пишеться, і
   недоотримують ресурс найменш цінні події.
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
    sizes: dict[str, int]           # точний розмір репліки кожного вузла, байтів


def choose_copies(records: list[Record], classes: list[int], cfg: Config, *, budget_bits: int | None = None,
                  use_dp: bool = True, copies_cap: int = 3, n_nodes: int = 3,
                  second_copy_factor: float | None = None, critical_scale: float = 1.0, compress: bool = True):
    """Крок 1. Повертає (r_i, таблиця Гаффмана вікна, ваги w_i, цінності v_i)."""
    lengths = code_lengths(pack_records(records)) if compress and records else None
    weights = record_weights(records, lengths) if lengths else [8 * RECORD_SIZE] * len(records)
    values = [base_value(c, r.critical, cfg, critical_scale) for c, r in zip(classes, records)]
    rmax = min(3, copies_cap, n_nodes)                      # копій не більше, ніж вузлів
    f2 = cfg.second_copy_factor if second_copy_factor is None else min(second_copy_factor,
                                                                       cfg.third_copy_factor)
    util = [row[:rmax + 1] for row in utility_table(values, f2, cfg.third_copy_factor)]
    cost = [[r * w for r in range(rmax + 1)] for w in weights]
    budget = cfg.replication_budget_bits if budget_bits is None else budget_bits
    solve = solve_dp if use_dp else solve_greedy
    return solve(util, cost, budget), lengths, weights, values


def place_copies(copies: list[int], weights: list[int], values: list[float], nsym: int,
                 reliability: dict[str, float], max_batch_bytes: int, *,
                 room: dict[str, int] | None = None, allowance: int | None = None,
                 compress: bool = True) -> tuple[dict[str, list[int]], list[int], dict[str, int]]:
    """Крок 2. Повертає (вузол -> індекси записів у порядку запису, розміщено копій, розміри реплік)."""
    nodes = list(reliability)
    members: dict[str, list[int]] = {n: [] for n in nodes}
    bits = dict.fromkeys(nodes, 0)
    size = dict.fromkeys(nodes, 0)                     # 0 — batch на вузлі ще нема
    limit = {n: min(max_batch_bytes, (room or {}).get(n, max_batch_bytes)) for n in nodes}
    spent, cap = 0, allowance if allowance is not None else float("inf")
    placed = [0] * len(copies)
    holders: list[set[str]] = [set() for _ in copies]
    miss = [1.0] * len(copies)                          # P(жодна копія запису не вижила)

    def size_of(count: int, total_bits: int) -> int:
        return MIN_SIZE + payload_size(count, nsym, total_bits, compress=compress)

    while True:
        best, best_key = None, None
        for n in nodes:
            pending = [(values[i] * reliability[n] * miss[i], i) for i, r in enumerate(copies)
                       if placed[i] < r and n not in holders[i]]
            pending = sorted(((g, i) for g, i in pending if g > 0), key=lambda gi: (-gi[0], gi[1]))
            if members[n]:
                # batch уже є: кандидат — одна копія, ціна — приріст розміру репліки
                for gain, i in pending:
                    s = size_of(len(members[n]) + 1, bits[n] + weights[i])
                    delta = s - size[n]
                    if s <= limit[n] and spent + delta <= cap:
                        key = (gain / delta, gain, -i, reliability[n])
                        if best_key is None or key > best_key:
                            best, best_key = (n, [i], s), key
            else:
                # новий batch: заголовок, HMAC і таблиця — фіксована ціна, тож порівнюємо
                # найкращий префікс копій за спаданням приросту (інакше вузол не відкриється ніколи)
                count = total_bits = 0
                gain_sum = 0.0
                for gain, i in pending:
                    s = size_of(count + 1, total_bits + weights[i])
                    if s > limit[n] or spent + s > cap:
                        break
                    count, total_bits, gain_sum = count + 1, total_bits + weights[i], gain_sum + gain
                    key = (gain_sum / s, gain_sum, -pending[0][1], reliability[n])
                    if best_key is None or key > best_key:
                        best, best_key = (n, [j for _, j in pending[:count]], s), key
        if best is None:
            break
        n, chosen, s = best
        spent += s - size[n]
        size[n] = s
        for i in chosen:
            members[n].append(i)
            bits[n] += weights[i]
            holders[i].add(n)
            miss[i] *= 1.0 - reliability[n]
            placed[i] += 1

    # найцінніші batch пишемо першими: невдала спроба запису теж витрачає бюджет (11.44)
    worth = {n: sum(values[i] for i in members[n]) * reliability[n] for n in nodes}
    order = sorted((n for n in nodes if members[n]), key=lambda n: (-worth[n], n))
    return {n: sorted(members[n]) for n in order}, placed, {n: size[n] for n in order}


def plan(records: list[Record], classes: list[int], cfg: Config, reliability: dict[str, float], *,
         budget_bits: int | None = None, room: dict[str, int] | None = None, allowance: int | None = None,
         use_dp: bool = True, copies_cap: int = 3, second_copy_factor: float | None = None,
         critical_scale: float = 1.0, compress: bool = True) -> WindowPlan:
    """Повний план вікна: копії від ДП + їх розміщення по вузлах."""
    if len(classes) != len(records):
        raise ValueError("класів має бути стільки ж, скільки записів")
    copies, lengths, weights, values = choose_copies(
        records, classes, cfg, budget_bits=budget_bits, use_dp=use_dp, copies_cap=copies_cap,
        n_nodes=len(reliability), second_copy_factor=second_copy_factor,
        critical_scale=critical_scale, compress=compress)
    batches, placed, sizes = place_copies(copies, weights, values, len(lengths or ()), reliability,
                                          cfg.max_batch_bytes, room=room, allowance=allowance,
                                          compress=compress)
    return WindowPlan(batches, copies, placed, lengths, sizes)
