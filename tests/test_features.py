"""Ознаки перцептрона: вікно історії лише назад, нормалізація за train."""
import numpy as np
import pytest

from log3.nn.data import dev_labeled_session
from log3.nn.features import FeatureSpec, Normalizer, build_features, feature_names
from log3.records import Record


@pytest.fixture(scope="module")
def recs():
    return dev_labeled_session(400, seed=3)[0]


def test_shape_and_names(recs):
    spec = FeatureSpec(history=8)
    X = build_features(recs, spec)
    assert X.shape == (len(recs), len(feature_names(spec)))
    assert np.isfinite(X).all()


def test_empty_log():
    X = build_features([], FeatureSpec(history=8))
    assert X.shape == (0, len(feature_names(FeatureSpec(history=8))))


def test_no_look_ahead(recs):
    """Зміна майбутніх записів не змінює ознак минулих: вікно лише з попередніх записів."""
    spec = FeatureSpec(history=8)
    k = 200
    base = build_features(recs, spec)
    future = [Record(r.timestamp_ms + 7, 4095, 999, -5000, 15, 7, 1, 3) for r in recs[k:]]
    changed = build_features(recs[:k] + future, spec)
    np.testing.assert_array_equal(base[:k], changed[:k])
    # і сама підпослідовність дає ті самі ознаки (нема залежності від довжини потоку)
    np.testing.assert_array_equal(base[:k], build_features(recs[:k], spec))


def test_history_window_matters(recs):
    a = build_features(recs, FeatureSpec(history=1))
    b = build_features(recs, FeatureSpec(history=16))
    assert not np.allclose(a, b)


def test_temporal_features_from_config_names():
    spec = FeatureSpec(history=4, temporal=("events_per_window", "severity_mean"))
    names = feature_names(spec)
    assert "events_per_window" in names and "severity_mean" in names
    assert "category_changes" not in names
    with pytest.raises(ValueError):
        feature_names(FeatureSpec(history=4, temporal=("no_such_feature",)))


def test_normalizer_uses_train_statistics():
    rng = np.random.default_rng(0)
    train = rng.normal(5.0, 2.0, size=(500, 3))
    norm = Normalizer.fit(train)
    other = np.full((4, 3), 5.0)
    out = norm.transform(other)
    np.testing.assert_allclose(out, (other - train.mean(0)) / train.std(0))
    # стовпець-константа не дає ділення на нуль
    const = Normalizer.fit(np.ones((10, 2)))
    assert np.isfinite(const.transform(np.ones((3, 2)))).all()
