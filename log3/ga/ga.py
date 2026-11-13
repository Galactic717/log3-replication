"""Генетичний алгоритм для параметрів пайплайна (умова §3.6).

Хромосома — дійсні гени, по одному на поле pipeline.Params:
  window_records      розмір вікна реплікації (ціле, при декодуванні округлюється);
  budget_scale        частка бюджету ДП вікна replication_budget_bits;
  second_copy_factor  f2 у корисності другої копії (обмежено f2 ≤ f3 з конфігу);
  critical_scale      вага critical_bonus у цінності v_i (вага планувальника);
  pacing_slack        скільки рівних часток глобального бюджету може взяти одне вікно.
Множники відносні, тож мікрозміни умови (бюджет −30 %, новий клас) лишаються в силі.

Один крок покоління:
  1. турнірна селекція: з k випадкових особин батьком стає найкраща (k = tournament_k);
  2. кросовер BLX-α (α = 0.5): ген дитини — випадкова точка з відрізка між генами
     батьків, розширеного на α·|різниця| в обидва боки. Імовірність кросовера p_c;
  3. мутація: до гена з імовірністю p_m додається гаусів шум σ = sigma·(ширина меж),
     результат обрізається до меж;
  4. елітизм: найкраща особина попереднього покоління замінює найгіршу дитину,
     тому найкращий фітнес у популяції не спадає між поколіннями.

Бюджет обчислень: покоління 0 — pop_size оцінок, кожне наступне — pop_size оцінок дітей
(еліта не переоцінюється). Разом рівно pop_size·generations — стільки ж отримує
random_search. Фітнес детермінований (pipeline.evaluate без γ·T), тож збережений фітнес
еліти — не «щасливий замір».
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from ..pipeline import Params

# Межі генів (включно). Це межі ПОШУКУ, а не умова задачі; конфіг може їх перевизначити.
DEFAULT_BOUNDS: dict[str, tuple[float, float]] = {
    "window_records": (16.0, 256.0),
    "budget_scale": (0.2, 1.5),
    "second_copy_factor": (1.0, 1.85),
    "critical_scale": (0.5, 30.0),
    "pacing_slack": (1.0, 3.0),
}


@dataclass(frozen=True)
class GAResult:
    best_params: Params
    best_fitness: float
    history: list[float]                      # найкращий фітнес за поколінням (для RS — після кожних pop_size оцінок)
    evaluations: int = 0                      # скільки разів викликано фітнес
    mean_history: list[float] = field(default_factory=list)   # середній фітнес покоління (лише ГА)


def bounds_from_config(cfg) -> dict[str, tuple[float, float]]:
    """f2 не може перевищувати f3 конфігу: інакше третя копія мала б від'ємний приріст."""
    return {**DEFAULT_BOUNDS, "second_copy_factor": (1.0, cfg.third_copy_factor)}


def encode(p: Params) -> np.ndarray:
    return np.array([getattr(p, k) for k in DEFAULT_BOUNDS], dtype=float)


def decode(x: np.ndarray, bounds: dict[str, tuple[float, float]] = DEFAULT_BOUNDS) -> Params:
    lo, hi = _lo_hi(bounds)
    x = np.clip(x, lo, hi)
    genes = dict(zip(bounds, (float(v) for v in x)))
    genes["window_records"] = int(round(genes["window_records"]))
    return Params(**genes)


def _lo_hi(bounds):
    lo = np.array([v[0] for v in bounds.values()])
    hi = np.array([v[1] for v in bounds.values()])
    return lo, hi


def run_ga(fitness: Callable[[Params], float], *, pop_size: int, generations: int, seed: int,
           elitism: bool = True, tournament_k: int = 3, p_crossover: float = 0.9,
           p_mutation: float | None = None, sigma: float = 0.1, blx_alpha: float = 0.5,
           bounds: dict[str, tuple[float, float]] | None = None, map_fn=map) -> GAResult:
    """map_fn — як оцінювати популяцію (map або pool.map для паралельних процесів)."""
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
        return np.array(list(map_fn(fitness, [decode(ind, bounds) for ind in pop])))

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
