"""Власний перцептрон: градієнт, навчання, детермінізм, метрики, C1 №9."""
from pathlib import Path

import numpy as np
import pytest

from log3.config import load_config
from log3.nn import classify as clf
from log3.nn.data import dev_labeled_session
from log3.nn.mlp import MLP, Adam, SGD, confusion_matrix, gradient_check, macro_f1
from log3.nn.train import TrainSettings, train_one

ROOT = Path(__file__).resolve().parent.parent
VALID_CSV = ROOT / "materials" / "valid.csv"


def _toy(n=300, d=5, k=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    W = rng.normal(size=(d, k))
    y = (X @ W).argmax(1)
    return X, y


# ---------- backprop ----------

@pytest.mark.parametrize("hidden", [(), (6,), (7, 5)])
def test_gradient_check(hidden):
    """Аналітичний градієнт (backprop) збігається з числовим (центральна різниця)."""
    X, y = _toy(20, 4, 3)
    net = MLP(4, hidden, 3, seed=1, l2=1e-3)
    cw = np.array([1.0, 2.0, 0.5])
    assert gradient_check(net, X, y, class_weights=cw) < 1e-6


def test_loss_decreases_sgd_and_adam():
    X, y = _toy()
    for opt in (SGD(lr=0.1, momentum=0.9), Adam(lr=0.01)):
        net = MLP(5, (16,), 3, seed=0)
        first = net.loss(X, y)
        for _ in range(200):
            _, grads = net.loss_and_grads(X, y)
            opt.step(net.params, grads)
        assert net.loss(X, y) < 0.5 * first


def test_deterministic_by_seed():
    X, y = _toy()
    s = TrainSettings(hidden=(8,), epochs=5, seed=0)
    a, _ = train_one(X, y, X, y, s, n_classes=3)
    b, _ = train_one(X, y, X, y, s, n_classes=3)
    for p, q in zip(a.params, b.params):
        np.testing.assert_array_equal(p, q)
    c, _ = train_one(X, y, X, y, TrainSettings(hidden=(8,), epochs=5, seed=1), n_classes=3)
    assert not np.array_equal(a.params[0], c.params[0])


def test_any_number_of_classes():
    """Новий клас подій на захисті: нічого не припускає рівно 3 класи."""
    X, y = _toy(400, 6, 4, seed=2)
    net, f1 = train_one(X, y, X, y, TrainSettings(hidden=(16,), epochs=40, seed=0), n_classes=4)
    assert net.n_out == 4
    assert f1 > 0.8
    assert confusion_matrix(y, net.predict(X), 4).shape == (4, 4)


# ---------- метрики ----------

def test_macro_f1_and_confusion():
    y = [0, 0, 1, 1, 2, 2]
    p = [0, 1, 1, 1, 2, 0]
    cm = confusion_matrix(y, p, 3)
    assert cm.tolist() == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]
    # F1: клас0 = 0.5, клас1 = 0.8, клас2 = 2/3
    assert macro_f1(y, p) == pytest.approx((0.5 + 0.8 + 2 / 3) / 3)
    assert macro_f1([0, 0, 0], [0, 0, 0]) == 1.0
    # macro-F1 карає за ігнорування рідкісного класу, accuracy — ні
    y2 = [0] * 9 + [2]
    assert macro_f1(y2, [0] * 10) < 0.5


# ---------- classify_records ----------

def test_classify_records_contract():
    recs = dev_labeled_session(300, seed=11)[0]
    a = clf.classify_records(recs)
    assert a == clf.classify_records(recs)           # детермінований
    assert len(a) == len(recs)
    assert all(isinstance(c, int) and 0 <= c < clf.model_n_classes() for c in a)
    assert clf.classify_records([]) == []


def test_dev_model_macro_f1_on_dev_valid():
    """Поки нема materials/: збережена dev-модель має macro-F1 ≥ F1_min на dev-valid."""
    meta = clf.model_meta()
    if meta.get("data_source") != "dev":
        pytest.skip("модель навчено не на dev-даних — діє test_c1_09")
    recs, y = dev_labeled_session(meta["dev_valid_records"], seed=meta["dev_valid_seed"])
    f1 = macro_f1(y, clf.classify_records(recs))
    assert f1 >= (load_config().F1_min or 0.0)


def test_c1_09_perceptron_macro_f1():
    if not VALID_CSV.exists():
        pytest.skip("materials/valid.csv нема — C1 №9 перевіряється на справжніх даних")
    from log3.records import load_records_csv
    recs, y = load_records_csv(VALID_CSV)
    assert y is not None, "у valid.csv нема колонки мітки"
    f1 = macro_f1(y, clf.classify_records(recs))
    F1_min = load_config().F1_min
    assert F1_min is not None, "F1_min нема в team_config.json"
    assert f1 >= F1_min, f"macro-F1 {f1:.3f} < F1_min {F1_min}"
