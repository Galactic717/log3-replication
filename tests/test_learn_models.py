"""Моделі вартості, власний MLP і безпечне завантаження артефактів."""
import io
import json

import numpy as np
import pytest

from log3.vault.learn import features as F
from log3.vault.learn.actions import CostModel
from log3.vault.learn.models import (
    BaselinePolicy,
    CostTablePolicy,
    FixedPolicy,
    MLPPolicy,
    ModelError,
    RidgePolicy,
    cost_matrix,
    load_artifact,
    save_artifact,
)

ACTIONS = ["store:0", "zstd:1", "zstd:3"]
COST = CostModel(2, 64 * 2**20)


def _data(n=600, seed=0):
    """Половина блоків — текст (zstd:1 дешевший за zstd:3), половина — випадкові байти (лише Store)."""
    rng = np.random.default_rng(seed)
    X = np.zeros((n, F.N_FEATURES), dtype=np.float32)
    text = rng.random(n) < 0.5
    X[:, F.index("log2_chunk_len")] = np.log2(2**20 + 1) / 26
    X[:, F.index("probe_ratio")] = np.where(text, rng.uniform(0.15, 0.3, n), rng.uniform(0.95, 1.0, n))
    X[:, F.index("entropy_sample")] = np.where(text, 0.6, 0.999)
    X[:, F.index("entropy_head")] = X[:, F.index("entropy_sample")]
    X[np.arange(n), F.index("type_text")] = 1
    X[np.arange(n), F.index("magic_text")] = text
    X[np.arange(n), F.index("magic_unknown")] = ~text
    raw = np.full(n, 2.0**20)
    r1 = X[:, F.index("probe_ratio")] * raw
    payload = np.stack([raw, np.where(text, r1, raw), np.where(text, r1 * 1.03, raw)], 1)
    cpu = np.stack([np.full(n, 5e3), np.where(text, 2.2e6, 3.5e5), np.where(text, 3.0e6, 4.3e5)], 1)
    return X, payload, cpu, raw, text


def _regret(policy, X, payload, cpu):
    c = cost_matrix(payload, cpu, COST)
    return float((c[np.arange(len(X)), policy.choose(X)] - c.min(1)).mean() / c.min(1).mean())


@pytest.mark.parametrize("kind", ["costtable", "ridge"])
def test_cost_models_predict_space_time_and_pick_cheapest(kind):
    X, payload, cpu, raw, text = _data()
    cls = {"costtable": CostTablePolicy, "ridge": RidgePolicy}[kind]
    p = cls(ACTIONS, COST).fit(X[:400], payload[:400], cpu[:400], raw[:400])
    ratio, cpu_pb = p.predict(X[400:])
    assert ratio.shape == (200, 3) and (ratio <= 1).all() and (cpu_pb >= 0).all()
    assert abs(ratio[text[400:], 1].mean() - (payload[400:][text[400:], 1] / raw[400:][text[400:]]).mean()) < 0.05
    choice = p.choose(X[400:])
    assert (choice[text[400:]] == 1).mean() > 0.9
    if kind == "costtable":                                # ridge помиляється на частки відсотка байтів
        assert (choice[~text[400:]] == 0).all()
    assert _regret(p, X[400:], payload[400:], cpu[400:]) < 0.01


def test_learned_beats_fixed_and_baseline_on_this_data():
    X, payload, cpu, raw, _ = _data()
    learned = CostTablePolicy(ACTIONS, COST).fit(X[:400], payload[:400], cpu[:400], raw[:400])
    rest = (X[400:], payload[400:], cpu[400:])
    base = BaselinePolicy(ACTIONS, COST, {"default_action": "zstd:3"})
    fixed = FixedPolicy(ACTIONS, COST, {"action": "zstd:3"})
    assert _regret(learned, *rest) < _regret(base, *rest) < _regret(fixed, *rest)
    assert _regret(FixedPolicy(ACTIONS, COST, {"action": "store:0"}), *rest) > _regret(learned, *rest)


def test_mlp_variant_trains_on_cheapest_action():
    X, payload, cpu, raw, _ = _data()
    c = cost_matrix(payload, cpu, COST)
    p = MLPPolicy(ACTIONS, COST, {"hidden": [16], "epochs": 15}).fit(X[:300], c[:300].argmin(1), X[300:400],
                                                                     c[300:400], seed=1)
    assert _regret(p, X[400:], payload[400:], cpu[400:]) < 0.05


@pytest.mark.parametrize("kind", ["costtable", "ridge", "mlp", "fixed", "baseline"])
def test_artifact_roundtrip_gives_same_decisions(tmp_path, kind):
    X, payload, cpu, raw, _ = _data()
    c = cost_matrix(payload, cpu, COST)
    p = {"costtable": lambda: CostTablePolicy(ACTIONS, COST).fit(X, payload, cpu, raw),
         "ridge": lambda: RidgePolicy(ACTIONS, COST).fit(X, payload, cpu, raw),
         "mlp": lambda: MLPPolicy(ACTIONS, COST, {"hidden": [8], "epochs": 3}).fit(X, c.argmin(1), X[:50], c[:50]),
         "fixed": lambda: FixedPolicy(ACTIONS, COST, {"action": "zstd:1"}),
         "baseline": lambda: BaselinePolicy(ACTIONS, COST, {"default_action": "zstd:3"})}[kind]()
    mid = save_artifact(tmp_path, p, {"provenance": {"rows": len(X)}, "metrics": {"x": 1.0}})
    q, doc = load_artifact(tmp_path, mid, admissible=ACTIONS + ["zstd:7"])
    assert np.array_equal(p.choose(X), q.choose(X))
    assert doc["features"]["names"] == list(F.NAMES) and doc["provenance"]["rows"] == len(X)
    assert not list(tmp_path.glob(".tmp-*"))


def _saved(tmp_path):
    X, payload, cpu, raw, _ = _data(100)
    return save_artifact(tmp_path, CostTablePolicy(ACTIONS, COST).fit(X, payload, cpu, raw), {})


def test_missing_corrupt_and_incompatible_models_are_rejected(tmp_path):
    with pytest.raises(ModelError):
        load_artifact(tmp_path, "nope", admissible=ACTIONS)
    with pytest.raises(ModelError):
        load_artifact(tmp_path, "../etc", admissible=ACTIONS)
    mid = _saved(tmp_path)
    with pytest.raises(ModelError, match="допустимі"):
        load_artifact(tmp_path, mid, admissible=["store:0", "zstd:1"])
    doc_path = tmp_path / mid / "model.json"
    doc = json.loads(doc_path.read_text())
    doc_path.write_text(json.dumps({**doc, "feature_schema": "log3-chunk-features/0/old"}))
    with pytest.raises(ModelError, match="схема"):
        load_artifact(tmp_path, mid, admissible=ACTIONS)
    doc_path.write_text(json.dumps(doc))
    arr = tmp_path / mid / "arrays.npz"
    arr.write_bytes(arr.read_bytes()[:-10] + b"0123456789")
    with pytest.raises(ModelError, match="SHA-256"):
        load_artifact(tmp_path, mid, admissible=ACTIONS)
    doc_path.write_text("{not json")
    with pytest.raises(ModelError):
        load_artifact(tmp_path, mid, admissible=ACTIONS)


def test_loading_never_unpickles(tmp_path):
    mid = _saved(tmp_path)
    buf = io.BytesIO()
    np.savez(buf, fine_keys=np.array([{"evil": 1}], dtype=object))
    blob = buf.getvalue()
    (tmp_path / mid / "arrays.npz").write_bytes(blob)
    doc_path = tmp_path / mid / "model.json"
    doc = json.loads(doc_path.read_text())
    import hashlib
    doc["arrays"]["sha256"] = hashlib.sha256(blob).hexdigest()
    doc_path.write_text(json.dumps(doc))
    with pytest.raises(ModelError):
        load_artifact(tmp_path, mid, admissible=ACTIONS)
