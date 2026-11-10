import itertools
import random

from log3.config import load_config
from log3.planner.knapsack import base_value, solve_dp, total, utility_table


def brute(util, cost, budget):
    """Перебір усіх комбінацій копій — еталон для малих задач."""
    best = 0.0
    for rs in itertools.product(*(range(len(u)) for u in util)):
        u, c = total(util, cost, list(rs))
        if c <= budget:
            best = max(best, u)
    return best


def random_case(rng, n):
    values = [rng.choice((1, 3, 6)) + 4 * (rng.random() < 0.3) for _ in range(n)]
    util = utility_table(values, 1.55, 1.85)
    weights = [rng.randint(40, 130) for _ in range(n)]
    cost = [[r * w for r in range(4)] for w in weights]
    return util, cost


def test_utility_table_is_cumulative():
    assert utility_table([2.0], 1.55, 1.85) == [[0.0, 2.0, 2.0 * 1.55, 2.0 * 1.85]]


def test_base_value_uses_config():
    cfg = load_config()
    assert base_value(0, 0, cfg) == cfg.priority[0]
    assert base_value(2, 1, cfg) == cfg.priority[2] + cfg.critical_bonus


def test_dp_equals_brute_force_and_respects_budget():
    rng = random.Random(11)
    for _ in range(40):
        n = rng.randint(1, 6)
        util, cost = random_case(rng, n)
        budget = rng.randint(0, 900)
        copies = solve_dp(util, cost, budget)
        u, c = total(util, cost, copies)
        assert c <= budget and all(0 <= r <= 3 for r in copies)
        assert abs(u - brute(util, cost, budget)) < 1e-9


def test_dp_boundaries():
    util, cost = [[0, 5]], [[0, 10]]
    assert solve_dp(util, cost, 0) == [0]           # бюджет 0 — жодної копії
    assert solve_dp(util, cost, 9) == [0]           # на 1 біт не вистачає
    assert solve_dp(util, cost, 10) == [1]          # рівно влазить
    assert solve_dp(utility_table([5], 1.55, 1.85), [[0, 10, 20, 30]], 10**6) == [3]
    assert solve_dp([], [], 100) == []              # порожнє вікно
    assert solve_dp(util, cost, -5) == [0]          # від'ємний бюджет як нуль


def test_dp_prefers_fewer_copies_on_tie():
    # друга копія нічого не додає — ДП не має її брати
    assert solve_dp([[0, 4, 4]], [[0, 10, 20]], 100) == [1]