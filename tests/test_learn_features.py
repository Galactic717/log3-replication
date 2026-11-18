"""Етап 2: допустимі дії, модель вартості, версійовані обмежені ознаки."""
import os
import time

import numpy as np
import pytest

from log3.vault.errors import VaultError
from log3.vault.learn import features as F
from log3.vault.learn.actions import DEFAULT_ACTIONS, STORE, CostModel, measure, parse_action, parse_actions


def test_actions_are_validated_and_store_is_always_allowed():
    acts = parse_actions(["zstd:3", "zstd:3", "zstd:1"])
    assert acts[0] == STORE and [a.key for a in acts] == ["store:0", "zstd:1", "zstd:3"]
    assert [a.key for a in parse_actions(DEFAULT_ACTIONS)][0] == "store:0"
    for bad in ("zstd:0", "zstd:20", "lzma:1", "store:3", "zstd:x", ""):
        with pytest.raises(VaultError):
            parse_action(bad)


def test_measure_matches_tract_and_unprofitable_becomes_store():
    data = os.urandom(200_000)
    m, payload = measure(data, parse_action("zstd:3"))
    assert m.codec_used == "store" and payload == data and m.cpu_ns >= 0
    text = b"abc def ghi " * 20_000
    m, payload = measure(text, parse_action("zstd:1"))
    assert m.codec_used == "zstd" and m.payload_len == len(payload) < len(text)


def test_cost_model_trades_bytes_for_cpu():
    c = CostModel(copies=2, cpu_second_equiv_bytes=1e6)
    assert c.cost(1000, 0) < c.cost(1000, 1e6)
    assert c.cost(500, 0) < c.cost(1000, 0)
    assert np.allclose(c.cost(np.array([1, 2]), np.array([0, 0])), [c.cost(1, 0), c.cost(2, 0)])


def _vec(data, **kw):
    args = {"suffix": "log", "file_size": len(data), "chunk_index": 0, "chunk_size": 1 << 20,
            "head_entropy_bits": 4.0, "lineage": None, "telemetry": F.Telemetry(0.1, 0.2, None)} | kw
    return F.extract(data, **args)


def test_feature_schema_is_versioned_and_ordered():
    s = F.schema()
    assert s["names"] == list(F.NAMES) and len(set(F.NAMES)) == F.N_FEATURES
    assert F.schema_id().startswith(f"{F.FEATURE_SCHEMA}/{F.FEATURE_VERSION}/")
    assert F.schema_id() == F.schema_id()


def test_features_separate_text_from_random_and_use_classes_not_names():
    text = (b"2026-11-17 INFO svc=api latency_ms=12\n" * 30_000)[:1 << 20]
    rnd = os.urandom(1 << 20)
    ft, fr = _vec(text), _vec(rnd, suffix="txt")
    assert ft.get("probe_ratio") < 0.3 < 0.9 < fr.get("probe_ratio")
    assert ft.get("entropy_sample") < fr.get("entropy_sample")
    assert ft.get("repeat_lag") > 10 * fr.get("repeat_lag")
    assert fr.get("type_text") == 1.0 and fr.get("magic_unknown") == 1.0      # оманливе розширення видно
    assert ft.values.dtype == np.float32 and ft.values.shape == (F.N_FEATURES,)
    assert F.magic_class(b"PK\x03\x04....") == "packed" and F.type_class("DOCX") == "office_packed"


def test_features_read_bounded_windows_only():
    """Байти поза вікнами вибірки не впливають на ознаки: великий блок не читається цілком."""
    a = bytearray(os.urandom(1 << 20))
    b = bytearray(a)
    b[100_000:200_000] = bytes(100_000)                    # між вікнами
    fa, fb = _vec(bytes(a)), _vec(bytes(b))
    keep = [i for i, n in enumerate(F.NAMES) if n != "probe_log_ns_per_byte"]
    assert np.array_equal(fa.values[keep], fb.values[keep])


def test_feature_cost_does_not_grow_with_chunk_size():
    small, big = os.urandom(256 << 10), os.urandom(16 << 20)

    def best(data):
        times = []
        for _ in range(5):
            t0 = time.perf_counter_ns()
            _vec(data)
            times.append(time.perf_counter_ns() - t0)
        return min(times)
    assert best(big) < 4 * best(small) + 2_000_000


def test_unavailable_features_are_masked_not_invented():
    f = _vec(b"x" * 5000, telemetry=F.Telemetry())
    for name in F.OPTIONAL:
        assert not f.mask[F.index(name)] and f.get(name) == 0.0
    g = _vec(b"x" * 5000, lineage=F.Lineage(versions_seen=5, content_changes=2))
    assert g.get("lineage_known") == 1.0 and g.get("lineage_change_rate") == pytest.approx(0.5)
    assert g.mask[F.index("lineage_change_rate")]
