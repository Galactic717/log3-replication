"""ГА і random search: елітизм, чесне порівняння, детермінізм, зв'язка з pipeline.evaluate."""
import numpy as np
import pytest

from log3.ga.ga import DEFAULT_BOUNDS, decode, encode, run_ga
from log3.ga.search import random_search
from log3.pipeline import Params


def _unit(p: Params) -> np.ndarray:
    """Params -> точка в [0, 1]^3 (через межі генів)."""
    return np.array([(v - lo) / (hi - lo) for v, (lo, hi) in zip(encode(p), DEFAULT_BOUNDS.values())])


def rastrigin_fitness(p: Params) -> float:
    """Тестова функція з CONTRACTS («заглушки»): мінус Растригін, максимум 0 у центрі куба."""
    z = (_unit(p) - 0.5) * 10.24
    return -float(10 * len(z) + np.sum(z * z - 10 * np.cos(2 * np.pi * z)))


def sphere_fitness(p: Params) -> float:
    return -float(np.sum((_unit(p) - 0.3) ** 2))


def test_encode_decode_roundtrip():
    p = Params(48, 1.0, 1.55)
    assert decode(encode(p)) == p
    q = decode(np.array([1e9, -5.0, 99.0]))            # за межами -> обрізано до меж
    lo_hi = list(DEFAULT_BOUNDS.values())
    assert q.window_records == lo_hi[0][1]
    assert q.no_compression_threshold == lo_hi[1][0]
    assert q.second_copy_factor == lo_hi[2][1]


def test_d1_ga_elitism_not_worse_than_gen0():
    for seed in range(5):
        r = run_ga(rastrigin_fitness, pop_size=12, generations=15, seed=seed, elitism=True)
        assert len(r.history) == 15
        assert all(b >= a for a, b in zip(r.history, r.history[1:]))   # не спадає
        assert r.history[-1] >= r.history[0]
        assert r.best_fitness == r.history[-1]
        assert rastrigin_fitness(r.best_params) == pytest.approx(r.best_fitness)


def test_equal_evaluation_budget():
    calls = []

    def f(p):
        calls.append(p)
        return sphere_fitness(p)

    r = run_ga(f, pop_size=8, generations=5, seed=0)
    assert len(calls) == 8 * 5 == r.evaluations
    calls.clear()
    s = random_search(f, evaluations=40, seed=0)
    assert len(calls) == 40 == s.evaluations


def test_d1_ga_vs_random_search():
    """За однакової кількості обчислень фітнесу ГА в середньому не гірший за random search."""
    ga, rs = [], []
    for seed in range(6):
        ga.append(run_ga(rastrigin_fitness, pop_size=20, generations=15, seed=seed).best_fitness)
        rs.append(random_search(rastrigin_fitness, evaluations=300, seed=seed).best_fitness)
    assert np.mean(ga) >= np.mean(rs)


def test_deterministic_by_seed():
    a = run_ga(sphere_fitness, pop_size=10, generations=6, seed=3)
    b = run_ga(sphere_fitness, pop_size=10, generations=6, seed=3)
    assert a == b
    assert random_search(sphere_fitness, evaluations=30, seed=1) == random_search(sphere_fitness, evaluations=30, seed=1)


def test_without_elitism_runs():
    r = run_ga(sphere_fitness, pop_size=10, generations=6, seed=0, elitism=False)
    assert len(r.history) == 6 and r.best_fitness >= max(r.history) - 1e-12


def test_ga_on_pipeline_evaluate_smoke():
    """Зв'язка з фітнесом Гліба: pipeline.evaluate приймає Params від ГА."""
    from log3.pipeline import evaluate
    r = run_ga(lambda p: evaluate(p, 0, n_seeds=1), pop_size=3, generations=2, seed=0)
    assert 0.0 <= r.best_fitness <= 1.0
    assert isinstance(r.best_params, Params)
