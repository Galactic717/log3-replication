"""Достовірність навчальних експериментів v1: абляція, джерела частин, validation, рівноправний відбір."""
from types import SimpleNamespace

import pytest

from log3.ga.search import select_params
from log3.nn.data import load_split
from log3.nn.features import FeatureSpec, feature_names
from log3.nn.train import check_sources, no_history_spec
from log3.pipeline import Params


def test_no_history_ablation_keeps_current_severity_only():
    spec = FeatureSpec(16, 1000.0, ("інтервал до попередньої події", "severity", "середня severity у вікні"))
    flat = no_history_spec(spec)
    assert flat.temporal == ("severity",)
    names = feature_names(flat)
    assert "severity" in names and "середня severity у вікні" not in names


def test_split_sources_are_tracked_and_hidden_mixing_is_refused():
    real, dev = ([], [], "materials"), ([], [], "dev")
    assert check_sources({"train": real, "validation": real, "test": real}) == dict.fromkeys(
        ("train", "validation", "test"), "materials")
    with pytest.raises(ValueError, match="джерел"):
        check_sources({"train": real, "validation": dev, "test": real})
    assert check_sources({"train": real, "validation": dev, "test": real}, allow_mixed=True)["validation"] == "dev"


def test_selection_split_is_called_validation():
    a, b = load_split("validation"), load_split("valid")              # стара назва — лише псевдонім
    assert a[2] == b[2] and len(a[0]) == len(b[0])


def _p(w):
    return Params(window_records=w, budget_scale=1.0, second_copy_factor=1.5, critical_scale=1.0, pacing_slack=1.0)


def _row(rep, ga_val, rs_val):
    return {"repeat": rep, "ga": SimpleNamespace(best_params=_p(10 + rep)), "ga_val": ga_val,
            "rs": SimpleNamespace(best_params=_p(100 + rep)), "rs_val": rs_val}


def test_baseline_and_random_search_are_real_candidates():
    base = _p(64)
    sel = select_params(base, (0.50, 0), [_row(0, (0.55, 0), (0.60, 0))])
    assert sel["source"] == "random" and sel["accepted"] and sel["params"].window_records == 100
    sel = select_params(base, (0.50, 0), [_row(0, (0.70, 1), (0.40, 0))])  # ГА порушив поріг — відкинуто
    assert sel["source"] == "config" and not sel["accepted"]
    sel = select_params(base, (0.50, 0), [_row(0, (0.50, 0), (0.50, 0))])  # нічия — простіший (конфіг)
    assert sel["source"] == "config"
    sel = select_params(base, (0.50, 2), [_row(0, (0.30, 0), (0.20, 3))])  # конфіг сам порушує — ГА
    assert sel["source"] == "ga"
