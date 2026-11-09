import dataclasses

from conftest import make_records
from log3.config import load_config
from log3.planner.scheduler import plan
from log3.replica import build_replica


def test_plan_constraints_and_exact_batch_size():
    cfg = load_config()
    for seed, max_batch in ((1, cfg.max_batch_bytes), (2, 400)):
        c = dataclasses.replace(cfg, max_batch_bytes=max_batch)
        win = make_records(48, seed=seed)
        p = plan(win, [0] * 48, c)
        for node, idx in p.batches.items():
            assert node in c.nodes and len(idx) == len(set(idx))   # запис у batch не більше разу
            blob = build_replica([win[i] for i in idx], "pw", replica_group_id=0, batch_seq=0,
                                 lengths=p.lengths)
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
