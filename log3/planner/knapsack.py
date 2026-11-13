"""Багатоваріантний рюкзак: скільки копій r_i ∈ {0..3} дати кожному запису вікна.

Задача:  максимізувати Σ util[i][r_i]  за умови  Σ cost[i][r_i] <= budget.
util[i][r] — корисність r копій запису i, cost[i][r] — скільки бітів вони займуть.
"""
from __future__ import annotations

import numpy as np

from ..config import Config
from ..records import Record, pack_record


def base_value(cls: int, critical: int, cfg: Config) -> float:
    """Цінність запису: пріоритет його класу плюс бонус, якщо запис критичний."""
    return cfg.priority[cls] + cfg.critical_bonus * critical


def utility_table(values: list[float], second_factor: float, third_factor: float) -> list[list[float]]:
    """Для кожного запису: [u(0), u(1), u(2), u(3)] = [0, v, v*f2, v*f3]."""
    return [[0.0, v, v * second_factor, v * third_factor] for v in values]


def record_weights(records: list[Record], lengths: dict[int, int]) -> list[int]:
    """Вага запису w_i: скільки бітів займуть його 16 байтів після стиснення таблицею вікна."""
    return [sum(lengths[b] for b in pack_record(r)) for r in records]


def solve_dp(util: list[list[float]], cost: list[list[int]], budget: int) -> list[int]:
    """Оптимальна кількість копій кожного запису (динамічне програмування).

    best[c] — найбільша корисність уже розглянутих записів, якщо витрачено не більше c бітів.
    Для кожного запису i перебираємо r = 0..Rmax:  new[c] = max_r (best[c - cost[i][r]] + util[i][r]).
    choice[i][c] запам'ятовує найкраще r, щоб потім пройти назад і відновити відповідь.
    Цикл по c зроблено зсувом масиву numpy: це ті самі обчислення, лише в ~15 разів швидше.
    Час O(n · budget · Rmax), пам'ять O(n · budget).
    """
    W = max(int(budget), 0)
    best = np.zeros(W + 1)                   # до першого запису: корисність 0 за будь-якого бюджету
    choice = np.zeros((len(util), W + 1), dtype=np.int8)
    for i, (u, c) in enumerate(zip(util, cost)):
        new = best.copy()                    # варіант r = 0: запис не копіюємо
        for r in range(1, len(u)):
            cr = c[r]
            if cr > W:                       # r копій не влазять навіть у весь бюджет
                continue
            # cand[c] = best[c - cr] + u[r] для всіх c >= cr одразу; менші c — неможливо
            cand = np.full(W + 1, -np.inf)
            cand[cr:] = best[:W + 1 - cr] + u[r]
            better = cand > new              # строго більше: за рівності лишаємо менше копій
            new[better] = cand[better]
            choice[i][better] = r
        best = new
    copies = [0] * len(util)
    left = W                                 # ідемо від останнього запису до першого
    for i in range(len(util) - 1, -1, -1):
        r = int(choice[i][left])
        copies[i] = r
        left -= cost[i][r]
    return copies


def solve_greedy(util: list[list[float]], cost: list[list[int]], budget: int) -> list[int]:
    """Жадібний baseline: щоразу додає ту наступну копію, що дає найбільше корисності на біт.

    Оптимуму не гарантує: вигідна «на біт» копія може зайняти місце,
    де вмістилася б краща комбінація кількох інших (див. test_greedy_counterexample).
    """
    copies = [0] * len(util)
    spent = 0
    while True:
        best_i, best_ratio = -1, -1.0
        for i, (u, c) in enumerate(zip(util, cost)):
            r = copies[i]
            if r + 1 >= len(u):                       # у запису вже максимум копій
                continue
            extra_cost = c[r + 1] - c[r]
            if spent + extra_cost > budget:           # наступна копія не влазить
                continue
            gain = u[r + 1] - u[r]
            ratio = gain / extra_cost if extra_cost > 0 else float("inf")
            if ratio > best_ratio:                    # строго: за рівності лишається менший індекс
                best_i, best_ratio = i, ratio
        if best_i < 0:                                # жодна копія більше не влазить
            return copies
        r = copies[best_i]
        spent += cost[best_i][r + 1] - cost[best_i][r]
        copies[best_i] = r + 1
        

def total(util: list[list[float]], cost: list[list[int]], copies: list[int]) -> tuple[float, int]:
    """(сумарна корисність, сумарні біти) для обраних копій — для перевірок і таблиць."""
    return (sum(u[r] for u, r in zip(util, copies)), sum(c[r] for c, r in zip(cost, copies)))