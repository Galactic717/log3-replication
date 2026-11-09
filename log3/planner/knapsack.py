"""Багатоваріантний рюкзак r_i ∈ {0..Rmax} (FORMAT_SPEC_P3 §3.5, контракт docs/CONTRACTS.md §6).

Модуль реалізував голова (Galactic717) для наскрізного пайплайна (C1 №6, №7, №8);
сигнатури — з контракту.

    maximize Σ u_i(r_i)  subject to  Σ r_i·w_i ≤ W,
    u_i(0)=0, u_i(1)=v_i, u_i(2)=v_i·f2, u_i(3)=v_i·f3   (factors кумулятивні)

ДП: dp[c] — найбільша корисність перших i подій, якщо витрачено не більше c бітів.
Перехід dp_i[c] = max_r (dp_{i-1}[c - cost_i(r)] + u_i(r)); час O(n·W·Rmax), пам'ять
O(n·W) байтів на відновлення вибору. Рядок по c рахується векторно numpy.
"""
from __future__ import annotations

import numpy as np

from ..config import Config
from ..records import Record, pack_record


def base_value(cls: int, critical: int, cfg: Config) -> float:
    return cfg.priority[cls] + cfg.critical_bonus * critical


def utility_table(values: list[float], second_factor: float, third_factor: float) -> list[list[float]]:
    return [[0.0, v, v * second_factor, v * third_factor] for v in values]


def record_weights(records: list[Record], lengths: dict[int, int]) -> list[int]:
    """w_i = Σ length(byte) по 16 байтах запису — біти однієї стисненої копії
    з таблицею вікна (накладні витрати batch сюди не входять, це оцінка)."""
    return [sum(lengths[b] for b in pack_record(r)) for r in records]


def solve_dp(util: list[list[float]], cost: list[list[int]], budget: int) -> list[int]:
    W = max(int(budget), 0)
    dp = np.zeros(W + 1)
    choice = np.zeros((len(util), W + 1), dtype=np.int8)
    for i, (u, c) in enumerate(zip(util, cost)):
        best = dp.copy()                       # r = 0
        arg = choice[i]
        for r in range(1, len(u)):
            if c[r] > W:
                continue
            cand = np.full(W + 1, -np.inf)
            cand[c[r]:] = dp[:W + 1 - c[r]] + u[r]
            better = cand > best               # строго: за рівності — менше копій
            best[better] = cand[better]
            arg[better] = r
        dp = best
    copies = [0] * len(util)
    w = W
    for i in range(len(util) - 1, -1, -1):
        r = int(choice[i, w])
        copies[i] = r
        w -= cost[i][r]
    return copies


def solve_greedy(util: list[list[float]], cost: list[list[int]], budget: int) -> list[int]:
    """Щоразу додає наступну копію з найбільшим Δu/Δcost, що ще влазить у бюджет.
    Без гарантії оптимуму: локально вигідні копії можуть зайняти місце вигіднішої комбінації."""
    copies = [0] * len(util)
    spent = 0
    while True:
        best, best_key = None, None
        for i, (u, c) in enumerate(zip(util, cost)):
            r = copies[i]
            if r + 1 >= len(u):
                continue
            dc = c[r + 1] - c[r]
            if spent + dc > budget:
                continue
            ratio = (u[r + 1] - u[r]) / dc if dc else float("inf")
            key = (ratio, -i)                  # детермінований tie-break: менший індекс
            if best_key is None or key > best_key:
                best, best_key = i, key
        if best is None:
            return copies
        r = copies[best]
        spent += cost[best][r + 1] - cost[best][r]
        copies[best] = r + 1


def total(util: list[list[float]], cost: list[list[int]], copies: list[int]) -> tuple[float, int]:
    """(Σ u_i(r_i), Σ cost_i(r_i)) — для перевірок і таблиць."""
    return (sum(u[r] for u, r in zip(util, copies)), sum(c[r] for c, r in zip(cost, copies)))
