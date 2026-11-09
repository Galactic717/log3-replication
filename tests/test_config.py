import dataclasses
import json

import pytest

from log3.config import DEV_CONFIG, config_from_dict, load_config


def test_dev_config_loads():
    cfg = load_config(DEV_CONFIG)
    assert cfg.replication_window_records == 48 and cfg.max_batch_bytes == 1024
    assert cfg.n_classes == len(cfg.priority) and cfg.nodes == ["A", "B", "C"]


def test_defence_micro_changes_are_config_only():
    d = json.loads(DEV_CONFIG.read_text(encoding="utf-8"))
    d["replication_budget_bits"] = int(d["replication_budget_bits"] * 0.7)   # бюджет −30 %
    d["priority"] = {"0": 1, "1": 3, "2": 6, "3": 10}                        # новий клас
    d["reliability_weight"]["B"] /= 2                                         # вузол удвічі гірший
    cfg = config_from_dict(d)
    assert cfg.n_classes == 4 and cfg.priority[3] == 10
    assert cfg.reliability_weight["B"] == pytest.approx(0.425)
    assert dataclasses.replace(cfg, second_copy_factor=1.3).second_copy_factor == 1.3


def test_nodes_shape_and_validation():
    d = json.loads(DEV_CONFIG.read_text(encoding="utf-8"))
    d.pop("reliability_weight")
    d["nodes"] = {"A": {"reliability_weight": 0.5}, "B": 0.25}
    assert config_from_dict(d).reliability_weight == {"A": 0.5, "B": 0.25}
    d["alpha"] = 0.9
    with pytest.raises(ValueError):
        config_from_dict(d)
    d.pop("max_batch_bytes")
    with pytest.raises(ValueError):
        config_from_dict(d)
