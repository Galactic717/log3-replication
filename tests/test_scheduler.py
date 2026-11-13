import dataclasses

from conftest import make_records
from log3.config import load_config
from log3.planner.knapsack import total, utility_table
from log3.planner.scheduler import choose_copies, place_copies, plan, plan_window
from log3.replica import build_replica


def test_plan_constraints_and_exact_batch_size():
    cfg = load_config()
    for seed, max_batch in ((1, cfg.max_batch_bytes), (2, 400)):
        c = dataclasses.replace(cfg, max_batch_bytes=max_batch)
        win = make_records(48, seed=seed)
        p = plan(win, [0] * 48, c)
        for node, idx in p.batches.items():
            assert node in c.nodes and len(idx) == len(set(idx))   # запис у batch не більше разу
            blob = build_replica([win[i] for i in idx], "pw", replica_group_id=0, batch_seq=0,lengths=p.lengths)
            assert len(blob) <= max_batch                         # оцінка розміру = факт
        for i in range(48):
            on = sum(i in idx for idx in p.batches.values())
            assert on == p.placed[i] <= p.copies[i]                # копії — на різних вузлах
        if max_batch == 400:
            assert sum(p.placed) < sum(p.copies)                   # вузли не вмістили все, що дав ДП


def test_critical_get_more_copies():
    cfg = load_config()
    win = make_records(48, seed=3)
    p = plan(win, [0] * 48, cfg)
    crit = [r for r, rec in zip(p.copies, win) if rec.critical]
    rest = [r for r, rec in zip(p.copies, win) if not rec.critical]
    assert min(crit) >= max(rest) - 1 and sum(crit) / len(crit) > sum(rest) / len(rest)


def test_plan_window_returns_batches_of_plan():
    cfg = load_config()
    win = make_records(48, seed=4)
    assert plan_window(win, [0] * 48, cfg) == plan(win, [0] * 48, cfg).batches


def test_empty_window():
    cfg = load_config()
    p = plan([], [], cfg)
    assert p.batches == {} and p.copies == [] and p.placed == []


def test_two_nodes_mean_at_most_two_copies():
    # вузлів два — третя копія неможлива: ДП не повинен її навіть призначати
    cfg = load_config()
    c = dataclasses.replace(cfg, reliability_weight={"A": 0.9, "B": 0.8})
    p = plan(make_records(48, seed=5), [0] * 48, c)
    assert max(p.copies) <= 2 and set(p.batches) <= {"A", "B"}


def test_tiny_batch_limit_places_nothing():
    # навіть порожня репліка більша за ліміт — нічого не пишемо, але й не падаємо
    cfg = load_config()
    c = dataclasses.replace(cfg, max_batch_bytes=50)
    p = plan(make_records(48, seed=6), [0] * 48, c)
    assert p.batches == {} and sum(p.placed) == 0


def test_valuable_records_are_placed_first():
    # місця мало: критичні записи (цінніші) мають отримати не менше копій, ніж звичайні
    cfg = load_config()
    c = dataclasses.replace(cfg, max_batch_bytes=300)
    win = make_records(48, seed=7)
    p = plan(win, [0] * 48, c)
    crit = [k for k, rec in zip(p.placed, win) if rec.critical]
    rest = [k for k, rec in zip(p.placed, win) if not rec.critical]
    assert min(crit) >= max(rest)


def test_budget_cut_30_percent():
    # мікрозміна: бюджет −30 % — план вкладається в новий бюджет і копій не більше
    cfg = load_config()
    win = make_records(48, seed=8)
    small = dataclasses.replace(cfg, replication_budget_bits=int(cfg.replication_budget_bits * 0.7))
    full_copies, *_ = choose_copies(win, [0] * 48, cfg)
    cut_copies, _, weights, values = choose_copies(win, [0] * 48, small)
    util = utility_table(values, cfg.second_copy_factor, cfg.third_copy_factor)
    cost = [[r * w for r in range(4)] for w in weights]
    assert total(util, cost, cut_copies)[1] <= small.replication_budget_bits
    assert sum(cut_copies) <= sum(full_copies)


def test_less_reliable_node_is_not_preferred():
    # мікрозміна: вузол A стає вдвічі менш надійним — єдина копія йде вже не на A
    cfg = load_config()
    args = ([1], [60], [5.0], 10)
    batches, _ = place_copies(*args, cfg)
    assert list(batches) == ["A"]
    rel = dict(cfg.reliability_weight, A=cfg.reliability_weight["A"] / 2)
    batches, _ = place_copies(*args, dataclasses.replace(cfg, reliability_weight=rel))
    assert list(batches) == ["B"]


def test_new_class_with_higher_value_gets_more_copies():
    # мікрозміна: новий клас 3 з найвищою цінністю — його записи отримують не менше копій
    cfg = load_config()
    c = dataclasses.replace(cfg, priority=tuple(cfg.priority) + (10.0,))
    win = make_records(48, seed=9)
    classes = [3 if i % 4 == 0 else 0 for i in range(48)]
    copies, *_ = choose_copies(win, classes, c)
    top = [r for r, k in zip(copies, classes) if k == 3]
    rest = [r for r, k in zip(copies, classes) if k == 0]
    assert sum(top) / len(top) >= sum(rest) / len(rest)