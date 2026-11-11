"""Генетичний алгоритм для параметрів пайплайна (контракт CONTRACTS.md §7).

Хромосома — три дійсні гени (Params): розмір вікна реплікації, поріг NO_COMPRESSION,
коефіцієнт другої копії f2. Вікно — ціле, тому при декодуванні округлюється.

Один крок покоління:
  1. турнірна селекція: з k випадкових особин батьком стає найкраща (k = tournament_k).
     Більше k — сильніший тиск відбору, менше різноманіття;
  2. кросовер BLX-α (α = 0.5): кожен ген дитини — випадкова точка з відрізка між генами
     батьків, розширеного на α·|різниця| в обидва боки; так нащадок може вийти і за межі
     батьків, і ГА не «стискається» передчасно. Імовірність кросовера p_c;
  3. мутація: до гена з імовірністю p_m додається гаусів шум σ = sigma·(ширина меж);
     результат обрізається до меж;
  4. елітизм: найкраща особина попереднього покоління замінює найгіршу дитину.
     Тому найкращий фітнес у популяції не спадає між поколіннями (history не спадає).

Бюджет обчислень: покоління 0 — pop_size оцінок, кожне наступне — pop_size оцінок дітей
(еліта не переоцінюється: її фітнес уже відомий). Разом рівно pop_size·generations —
стільки ж отримує random_search для чесного порівняння.

Фітнес — pipeline.evaluate (середній Q за кількома відкритими seeds), щоб ГА не
перенавчився під один сценарій відмов: офіційний Q рахують на прихованих seeds.
Межі генів: DEFAULT_BOUNDS або "ga_bounds" у team_config.json.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..pipeline import Params

# Межі генів (включно). Це межі ПОШУКУ, а не умова задачі; конфіг може їх перевизначити.
DEFAULT_BOUNDS: dict[str, tuple[float, float]] = {
    "window_records": (8.0, 192.0),
    "no_compression_threshold": (0.5, 1.5),
    "second_copy_factor": (1.0, 2.5),
}


@dataclass(frozen=True)
class GAResult:
    best_params: Params
    best_fitness: float
    history: list[float]                      # найкращий фітнес за поколінням (для RS — після кожних pop_size оцінок)
    evaluations: int = 0                      # скільки разів викликано фітнес
    mean_history: list[float] = field(default_factory=list)   # середній фітнес покоління (лише ГА)


def bounds_from_config(cfg) -> dict[str, tuple[float, float]]:
    b = dict(DEFAULT_BOUNDS)
    for k, v in (cfg.raw.get("ga_bounds") or {}).items():
        if k in b:
            b[k] = (float(v[0]), float(v[1]))
    return b


def encode(p: Params) -> np.ndarray:
    return np.array([p.window_records, p.no_compression_threshold, p.second_copy_factor], dtype=float)


def decode(x: np.ndarray, bounds: dict[str, tuple[float, float]] = DEFAULT_BOUNDS) -> Params:
    lo, hi = _lo_hi(bounds)
    x = np.clip(x, lo, hi)
    return Params(int(round(x[0])), float(x[1]), float(x[2]))


def _lo_hi(bounds):
    lo = np.array([v[0] for v in bounds.values()])
    hi = np.array([v[1] for v in bounds.values()])
    return lo, hi


def run_ga(fitness: Callable[[Params], float], *, pop_size: int, generations: int, seed: int,
           elitism: bool = True, tournament_k: int = 3, p_crossover: float = 0.9,
           p_mutation: float | None = None, sigma: float = 0.1, blx_alpha: float = 0.5,
           bounds: dict[str, tuple[float, float]] | None = None) -> GAResult:
    if pop_size < 2 or generations < 1:
        raise ValueError("pop_size ≥ 2, generations ≥ 1")
    bounds = bounds or DEFAULT_BOUNDS
    lo, hi = _lo_hi(bounds)
    n_genes = len(lo)
    p_mut = 1.0 / n_genes if p_mutation is None else p_mutation
    rng = np.random.default_rng(seed)
    evals = 0

    def evaluate(pop):
        nonlocal evals
        evals += len(pop)
        return np.array([fitness(decode(ind, bounds)) for ind in pop])

    pop = lo + rng.random((pop_size, n_genes)) * (hi - lo)       # покоління 0: рівномірно в межах
    fit = evaluate(pop)
    best_i = int(fit.argmax())
    best_x, best_f = pop[best_i].copy(), float(fit[best_i])
    history, mean_hist = [float(fit.max())], [float(fit.mean())]

    def tournament():
        idx = rng.choice(pop_size, size=min(tournament_k, pop_size), replace=False)
        return pop[idx[fit[idx].argmax()]]

    for _ in range(generations - 1):
        children = np.empty_like(pop)
        for c in range(pop_size):
            a, b = tournament(), tournament()
            if rng.random() < p_crossover:                         # BLX-α
                d = np.abs(a - b)
                child = rng.uniform(np.minimum(a, b) - blx_alpha * d, np.maximum(a, b) + blx_alpha * d)
            else:
                child = a.copy()
            mask = rng.random(n_genes) < p_mut                     # гаусова мутація
            child = child + mask * rng.normal(0.0, sigma * (hi - lo))
            children[c] = np.clip(child, lo, hi)
        child_fit = evaluate(children)
        if elitism:
            elite_i = int(fit.argmax())
            worst = int(child_fit.argmin())
            children[worst], child_fit[worst] = pop[elite_i], fit[elite_i]
        pop, fit = children, child_fit
        history.append(float(fit.max()))
        mean_hist.append(float(fit.mean()))
        if fit.max() > best_f:
            best_i = int(fit.argmax())
            best_x, best_f = pop[best_i].copy(), float(fit[best_i])

    return GAResult(decode(best_x, bounds), best_f, history, evals, mean_hist)
