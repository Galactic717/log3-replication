"""Dev-модель має передавати планувальнику витрачену місткість, зокрема після BAD."""
from dataclasses import replace

import pytest

from log3 import devsim, pipeline
from log3.config import DEV_CONFIG, Node, Scenario, load_config
from log3.records import Record


@pytest.mark.parametrize("loss, stored", [(0.0, True), (0.999, False)])
def test_room_accounts_for_successful_and_lost_writes(loss, stored):
    sc = Scenario("normal", 24, {"A": Node(10, loss, 1, 1), "B": Node(9, 0, 1, 1)}, 0)
    model = devsim.Replication(sc.as_model_dict(), seed=0)
    assert pipeline._room(model.stats(), sc) == (24, {"A": 10, "B": 9})

    assert model.store("A", b"1234") is stored
    assert pipeline._room(model.stats(), sc) == (20, {"A": 6, "B": 9})
    assert model.surviving_blobs() == ([b"1234"] if stored else [])

    # Відхилення місткістю до списання не витрачає ресурс.
    assert model.store("A", b"1234567") is False
    assert pipeline._room(model.stats(), sc) == (20, {"A": 6, "B": 9})
    assert model.store("A", b"123456") is stored
    assert model.store("B", b"123") is True
    assert pipeline._room(model.stats(), sc) == (11, {"A": 0, "B": 6})


def test_pipeline_uses_other_nodes_when_dev_node_is_full(monkeypatch):
    cfg = replace(load_config(DEV_CONFIG), replication_window_records=1)
    sc = Scenario("normal", 384, {n: Node(128, 0, 1, 1) for n in "ABC"}, 0)
    records = [Record(i, 1, 100, 0, 0, 0, 1, 0) for i in range(4)]
    monkeypatch.setattr(pipeline, "replication_class", lambda: devsim.Replication)
    result = pipeline.run("normal", 0, cfg=cfg, scenario_cfg=sc, records=records,
                          use_nn=False, compress=False, pacing=False)
    # Один запис у репліці — 105 B: у кожен вузол вміщається лише одна.
    assert result.recovered_critical == 3
    assert result.replicas_written == result.stored_ok == 3
