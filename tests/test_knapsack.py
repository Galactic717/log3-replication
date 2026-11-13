import itertools
import random

from log3.config import load_config
from log3.nn.classify import classify_records
from log3.pipeline import load_session
from log3.planner.knapsack import base_value, solve_dp, solve_greedy, total, utility_table
from log3.planner.scheduler import choose_copies

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


def test_greedy_respects_budget_and_limits():
    rng = random.Random(5)
    for _ in range(40):
        util, cost = random_case(rng, rng.randint(1, 20))
        budget = rng.randint(0, 3000)
        copies = solve_greedy(util, cost, budget)
        assert total(util, cost, copies)[1] <= budget and all(0 <= r <= 3 for r in copies)


def test_greedy_counterexample():
    # Корисність на біт: A = 10/6 ≈ 1.67 > B = C = 6/5 = 1.2.
    # Жадібний бере A (6 бітів), і на B+C (10 бітів) місця вже нема: корисність 10.
    # ДП бере B і C: 10 бітів, корисність 12. Жадібний не оптимальний.
    util = [[0, 10], [0, 6], [0, 6]]
    cost = [[0, 6], [0, 5], [0, 5]]
    assert solve_greedy(util, cost, 10) == [1, 0, 0]
    assert solve_dp(util, cost, 10) == [0, 1, 1]
    assert total(util, cost, [0, 1, 1])[0] > total(util, cost, [1, 0, 0])[0]


def test_c1_06_dp_not_worse_than_greedy_on_session():
    cfg = load_config()
    records = load_session(cfg)
    classes = classify_records(records)
    W = cfg.replication_window_records
    windows = [(records[i:i + W], classes[i:i + W]) for i in range(0, len(records), W)]
    for win, cls in windows:
        dp, *_ = choose_copies(win, cls, cfg, use_dp=True)
        gr, lengths, weights, values = choose_copies(win, cls, cfg, use_dp=False)
        util = utility_table(values, cfg.second_copy_factor, cfg.third_copy_factor)
        cost = [[r * w for r in range(4)] for w in weights]
        assert total(util, cost, dp)[0] >= total(util, cost, gr)[0] - 1e-9
        assert total(util, cost, dp)[1] <= cfg.replication_budget_bits
    rng = random.Random(2)
    for _ in range(30):                     # плюс випадкові набори
        util, cost = random_case(rng, rng.randint(5, 48))
        budget = rng.randint(0, 5000)
        assert total(util, cost, solve_dp(util, cost, budget))[0] >= \
            total(util, cost, solve_greedy(util, cost, budget))[0] - 1e-9