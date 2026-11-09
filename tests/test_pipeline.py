import dataclasses

import pytest

from log3 import pipeline
from log3.config import load_config
from log3.pipeline import Params, evaluate, run

SEEDS = range(3)


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def check_metrics(r, cfg):
    assert 0 <= r.S <= 1 and 0 <= r.R <= 1 and 0 < r.T < 1
    assert r.Q == pytest.approx(cfg.alpha * r.S + cfg.beta * r.R + cfg.gamma * r.T)
    assert r.b_ref == 16 * pipeline.load_session(cfg).__len__()


def test_c1_07_normal(cfg):
    for seed in SEEDS:
        r = run("normal", seed, cfg=cfg)
        assert r.recovered_critical == r.total_critical > 0
        check_metrics(r, cfg)


def test_c1_08_degraded(cfg):
    limit = cfg.raw.get("c1_degraded_max_unrecovered_critical", 0.10)
    for seed in SEEDS:
        r = run("degraded", seed, cfg=cfg)
        assert 1 - r.R <= limit
        check_metrics(r, cfg)


def test_emergency_and_switches(cfg):
    base = run("emergency", 0, cfg=cfg)
    check_metrics(base, cfg)
    for kw in ({"use_nn": False}, {"use_dp": False}, {"copies_cap": 1}, {"compress": False}):
        check_metrics(run("emergency", 0, cfg=cfg, **kw), cfg)
    one = run("normal", 0, cfg=cfg, copies_cap=1)
    assert one.copies_assigned <= one.b_ref // 16            # не більше однієї копії на подію
    assert one.b_stored < run("normal", 0, cfg=cfg).b_stored  # друга/третя копії коштують біти


def test_q_zero_when_nothing_critical_recovered(cfg, monkeypatch):
    monkeypatch.setattr(pipeline, "recover", lambda blobs, pw: ([], len(blobs)))
    assert run("normal", 0, cfg=cfg).Q == 0.0

    def boom(blobs, pw):
        raise RuntimeError("recovery crashed")
    monkeypatch.setattr(pipeline, "recover", boom)
    assert run("normal", 0, cfg=cfg).Q == 0.0


def test_evaluate_is_fitness(cfg):
    p = pipeline.default_params(cfg)
    q = evaluate(p, 0, n_seeds=2, cfg=cfg)
    assert 0 < q < 1
    assert evaluate(dataclasses.replace(p, window_records=0), 0, cfg=cfg) == 0.0
    assert evaluate(Params(96, 1.0, 1.3), 0, n_seeds=1, cfg=cfg) > 0
