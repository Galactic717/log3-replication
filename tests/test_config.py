import copy
import dataclasses
import json

import pytest

from log3.config import DEV_CONFIG, MATERIALS_CONFIG, config_from_dict, load_config

DEV = json.loads(DEV_CONFIG.read_text(encoding="utf-8"))


def test_dev_config_has_teacher_schema():
    cfg = load_config(DEV_CONFIG)
    assert cfg.replication_window_records == 48 and cfg.max_batch_bytes == 1024
    assert cfg.n_classes == 3 and cfg.priority == (1.0, 5.0, 4.0)
    assert set(cfg.scenarios) == {"normal", "degraded", "emergency"}
    assert list(cfg.scenario("аварійний").nodes) == ["A", "B", "C"]
    assert cfg.scenario("degraded").max_unrecovered_critical == pytest.approx(0.1)


def test_teacher_config_loads():
    if not MATERIALS_CONFIG.exists():
        pytest.skip("materials/team_config.json нема")
    cfg = load_config(MATERIALS_CONFIG)
    assert cfg.alpha + cfg.beta + cfg.gamma == pytest.approx(1.0)
    assert len(cfg.temporal_features) >= 5 and cfg.F1_min > 0


def test_defence_micro_changes_are_config_only():
    d = copy.deepcopy(DEV)
    d["реплікація"]["replication_budget_bits"] = int(d["реплікація"]["replication_budget_bits"] * 0.7)
    d["класифікатор"]["класи"]["3"] = "новий клас"                      # новий клас із вищою цінністю
    d["пріоритети_класів"]["новий клас"] = 10
    cfg = config_from_dict(d)
    assert cfg.n_classes == 4 and cfg.priority[3] == 10 and cfg.replication_budget_bits == 3500
    sc = cfg.scenario("degraded")
    worse = sc.with_node("B", reliability_weight=sc.nodes["B"].reliability_weight / 2,
                         failure_probability=min(0.99, 2 * sc.nodes["B"].failure_probability))
    assert worse.reliability["B"] == pytest.approx(0.4725) and sc.reliability["B"] == pytest.approx(0.945)
    assert worse.as_model_dict()["вузли"]["B"]["failure_probability"] == pytest.approx(0.11)
    assert dataclasses.replace(cfg, second_copy_factor=1.3).second_copy_factor == 1.3


@pytest.mark.parametrize("path, value", [
    (("метрика_Q", "alpha"), 0.9),                         # α + β + γ ≠ 1
    (("реплікація", "second_copy_factor"), 2.5),           # f2 > f3
    (("реплікація", "replication_budget_bits"), -1),
    (("пароль_сеансу",), ""),
])
def test_validation(path, value):
    d = copy.deepcopy(DEV)
    node = d
    for k in path[:-1]:
        node = node[k]
    node[path[-1]] = value
    with pytest.raises(ValueError):
        config_from_dict(d)


def test_missing_key_and_class_gap():
    d = copy.deepcopy(DEV)
    d["реплікація"].pop("max_batch_bytes")
    with pytest.raises(ValueError, match="max_batch_bytes"):
        config_from_dict(d)
    d = copy.deepcopy(DEV)
    d["класифікатор"]["класи"]["5"] = d["класифікатор"]["класи"].pop("2")   # пропуск номера класу
    d["пріоритети_класів"]["перехід до інциденту"] = 4
    with pytest.raises(ValueError):
        config_from_dict(d)
