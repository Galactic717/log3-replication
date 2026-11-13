"""Наскрізний тракт: C1 №7, №8, частина №12, абляції, мікрозміни захисту, правило Q = 0."""
import dataclasses

import pytest

from log3 import pipeline
from log3.config import MATERIALS_CONFIG, load_config
from log3.pipeline import Params, evaluate, run, tuned_params

SEEDS = (301, 302, 303, 304, 305)          # не перетинаються з seeds фітнесу й holdout ГА


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def cache(cfg):
    return pipeline.FitnessCache(cfg)


def params_for(cfg, which):
    if which == "config":
        return pipeline.default_params(cfg)
    if not MATERIALS_CONFIG.exists():
        pytest.skip("параметри ГА налаштовано на пакеті викладача, а тут dev-середовище")
    return tuned_params(cfg)


def kw(cache):
    return {"cfg": cache.cfg, "records": cache.records, "classes": cache.classes, "salt": cache.salt}


def check_metrics(r, cfg, n):
    assert 0 <= r.S <= 1 and 0 <= r.R <= 1 and 0 < r.T < 1
    assert r.Q == pytest.approx(cfg.alpha * r.S + cfg.beta * r.R + cfg.gamma * r.T)
    assert r.b_ref == 16 * n and r.b_stored % 8 == 0


@pytest.mark.parametrize("params", ["config", "ga"])
def test_c1_07_normal_all_critical_recovered(cfg, cache, params):
    p = params_for(cfg, params)
    for seed in SEEDS:
        r = run("normal", seed, p, **kw(cache))
        assert r.recovered_critical == r.total_critical > 0
        check_metrics(r, cfg, len(cache.records))


@pytest.mark.parametrize("params", ["config", "ga"])
def test_c1_08_degraded_threshold(cfg, cache, params):
    p = params_for(cfg, params)
    limit = cfg.scenario("degraded").max_unrecovered_critical
    for seed in SEEDS:
        assert 1 - run("degraded", seed, p, **kw(cache)).R <= limit


def test_emergency_threshold_and_ablations(cfg, cache):
    limit = cfg.scenario("emergency").max_unrecovered_critical
    for seed in SEEDS:
        assert 1 - run("emergency", seed, pipeline.default_params(cfg), **kw(cache)).R <= limit
        if MATERIALS_CONFIG.exists():
            assert 1 - run("emergency", seed, tuned_params(cfg), **kw(cache)).R <= limit
    n = len(cache.records)
    for switches in ({"use_dp": False}, {"copies_cap": 1}, {"compress": False}, {"pacing": False}):
        check_metrics(run("emergency", 1, cfg=cfg, records=cache.records, **switches), cfg, n)
    no_nn = run("emergency", 1, cfg=cfg, records=cache.records, use_nn=False)
    check_metrics(no_nn, cfg, n)


def test_full_run_with_classifier_within_time_limit(cfg):
    r = run("degraded", cfg=cfg)                 # повний прогін: читання сеансу, перцептрон, усе
    assert r.seconds < cfg.time_limit_s and r.Q > 0


def test_copies_cost_bits(cfg, cache):
    # D1: друга/третя копія витрачає біти — в аварійному сценарії max 1 копія пише менше
    one = run("degraded", 1, copies_cap=1, **kw(cache))
    three = run("degraded", 1, copies_cap=3, **kw(cache))
    assert one.copies_placed <= len(cache.records) and one.b_stored <= three.b_stored


def test_micro_change_less_reliable_node_end_to_end(cfg, cache):
    sc = cfg.scenario("degraded")
    worse = sc.with_node("B", reliability_weight=sc.nodes["B"].reliability_weight / 2,
                         failure_probability=2 * sc.nodes["B"].failure_probability)
    r = run("degraded", 1, scenario_cfg=worse, **kw(cache))
    assert r.Q > 0 and 1 - r.R <= sc.max_unrecovered_critical


def test_micro_change_budget_cut_end_to_end(cfg, cache):
    small = dataclasses.replace(cfg, replication_budget_bits=int(cfg.replication_budget_bits * 0.7))
    base = run("normal", 1, **kw(cache))
    cut = run("normal", 1, **{**kw(cache), "cfg": small})
    assert cut.copies_assigned <= base.copies_assigned and cut.R == 1.0


def test_micro_change_new_class_end_to_end(cfg, cache):
    c = dataclasses.replace(cfg, priority=tuple(cfg.priority) + (10.0,),
                            class_names=tuple(cfg.class_names) + ("новий",))
    classes = [3 if k == 2 else k for k in cache.classes]
    r = run("degraded", 1, cfg=c, records=cache.records, classes=classes, salt=cache.salt)
    assert r.Q > 0


def test_q_zero_when_recovery_fails(cfg, cache, monkeypatch):
    monkeypatch.setattr(pipeline, "recover", lambda blobs, pw: ([], len(blobs)))
    assert run("normal", 1, **kw(cache)).Q == 0.0

    def boom(blobs, pw):
        raise RuntimeError("recovery crashed")
    monkeypatch.setattr(pipeline, "recover", boom)
    assert run("normal", 1, **kw(cache)).Q == 0.0


def test_bad_classes_are_rejected(cfg, cache):
    with pytest.raises(ValueError):
        run("normal", 1, cfg=cfg, records=cache.records, classes=[9] * len(cache.records))


def test_evaluate_is_deterministic_fitness(cfg, cache):
    p = pipeline.default_params(cfg)
    q = evaluate(p, (1,), cache=cache)
    assert 0 < q < 1 and q == evaluate(p, (1,), cache=cache)
    assert evaluate(dataclasses.replace(p, window_records=0), (1,), cache=cache) == 0.0
    assert evaluate(Params(96, 0.5, 1.3, 2.0, 1.5), (1,), scenarios=("normal",), cache=cache) > 0
