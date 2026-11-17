"""Журнал рішень і спостережень: дублікати, затримані й відсутні результати, стадії захисту."""
import numpy as np

from log3.vault.catalog import Catalog
from log3.vault.learn import features as F
from log3.vault.learn.journal import STAGES, Journal, decision_id, pack

ACTIONS = ["store:0", "zstd:1", "zstd:3"]


def _decision(vid, oid, t, *, probe_p=1.0, quarantine=0, group="g1"):
    v = np.zeros(F.N_FEATURES, dtype=np.float32)
    v[0] = t % 7
    return {"decision_id": decision_id(vid, oid), "operation_id": "op-" + vid, "version_id": vid, "object_id": oid,
            "chunk_index": 0, "group_key": group, "decided_ns": t, "feature_schema": F.schema_id(),
            "features": pack(v), "mask": np.ones(F.N_FEATURES, dtype=bool).tobytes(), "feature_ns": 10,
            "inference_ns": 0, "mode": "baseline", "model_id": None, "policy_action": "zstd:3", "model_action": None,
            "chosen_action": "zstd:3", "fallback": None, "probe_p": probe_p, "quarantine": quarantine, "published": 1}


def _outcome(vid, oid, t, action="zstd:3", payload=500):
    return {"decision_id": decision_id(vid, oid), "result_ns": t + 5, "action": action, "codec_used": "zstd",
            "raw_len": 1000, "payload_len": payload, "stored_len": payload + 51, "wall_ns": 7, "cpu_ns": 7,
            "forced_store": 0}


def _probe(vid, oid, action, payload):
    return {"decision_id": decision_id(vid, oid), "action": action, "measured_ns": 1, "codec_used": "zstd",
            "payload_len": payload, "wall_ns": 3, "cpu_ns": 3}


def test_duplicates_are_ignored_and_counted(tmp_path):
    j = Journal(tmp_path / "j.sqlite")
    d, o = _decision("v1", "o1", 100), _outcome("v1", "o1", 100)
    assert j.write_decisions([d], [o], [])["duplicates"] == 0
    assert j.write_decisions([d], [o], [])["duplicates"] == 2
    assert j.scalar("SELECT COUNT(*) FROM decisions") == 1 and j.counters()["duplicate_records"] == 2
    assert decision_id("v1", "o1") == decision_id("v1", "o1") != decision_id("v2", "o1")


def test_delayed_and_missing_outcomes_are_visible_not_imputed(tmp_path):
    j = Journal(tmp_path / "j.sqlite")
    j.write_decisions([], [_outcome("v1", "late", 50)], [])            # результат раніше за рішення
    j.write_decisions([_decision("v1", "late", 50), _decision("v1", "lost", 60)], [], [])
    c = j.completeness()
    assert c["decisions_without_outcome"] == 1 and c["outcomes_without_decision"] == 0
    data = j.labeled(feature_schema=F.schema_id(), actions=ACTIONS)
    assert len(data["X"]) == 0                                          # без проб і результату — не мітка


def test_labeled_requires_full_information_and_skips_quarantine(tmp_path):
    j = Journal(tmp_path / "j.sqlite")
    for t, (vid, q) in enumerate((("a", 0), ("b", 0), ("c", 1)), start=1):
        j.write_decisions([_decision(vid, "o", t * 10, quarantine=q)], [_outcome(vid, "o", t * 10)],
                          [_probe(vid, "o", "store:0", 1000), _probe(vid, "o", "zstd:1", 520)])
    j.write_decisions([_decision("d", "o", 99)], [_outcome("d", "o", 99)], [_probe("d", "o", "store:0", 1000)])
    data = j.labeled(feature_schema=F.schema_id(), actions=ACTIONS)
    assert len(data["X"]) == 2 and data["incomplete"] == 1
    assert data["payload"].shape == (2, 3) and data["payload"][0, 2] == 500
    assert list(data["decided_ns"]) == [10, 20]
    assert len(j.labeled(feature_schema=F.schema_id(), actions=ACTIONS, include_quarantine=True)["X"]) == 3
    assert len(j.labeled(feature_schema="other/1/x", actions=ACTIONS)["X"]) == 0


def test_lineage_counts_changes_only_after_decisions(tmp_path):
    j = Journal(tmp_path / "j.sqlite")
    assert j.lineage("g") is None
    j.update_lineage([("g", "c1")], 1)
    j.update_lineage([("g", "c1")], 2)
    j.update_lineage([("g", "c2")], 3)
    assert j.lineage("g") == (3, 1)


def test_prune_bounds_journal(tmp_path):
    j = Journal(tmp_path / "j.sqlite")
    for t in range(20):
        j.write_decisions([_decision("v", f"o{t}", t)], [_outcome("v", f"o{t}", t)],
                          [_probe("v", f"o{t}", "zstd:1", 1)])
    assert j.prune(5) == 15
    assert j.scalar("SELECT COUNT(*) FROM decisions") == 5 and j.scalar("SELECT COUNT(*) FROM probes") == 5
    assert j.scalar("SELECT MIN(decided_ns) FROM decisions") == 15


def test_harvest_distinguishes_protection_stages(tmp_path):
    cat = Catalog(tmp_path / "catalog.sqlite")
    cat.x("INSERT INTO versions(version_id, op_id, name, root, state, priority, min_copies, target_copies, created_ns, "
          "heartbeat_ns, accepted_ns, protected_ns) VALUES('v1','op','n','/r','PROTECTED',1,2,2,1,1,100,400)")
    for oid in ("o1", "o2"):
        cat.x("INSERT INTO version_objects VALUES('v1', ?)", (oid,))
    cat.x("INSERT INTO replicas VALUES('o1','disk','CONFIRMED',10,'x',200,900)")     # перечитано verify пізніше
    cat.x("INSERT INTO replicas VALUES('o1','node','CONFIRMED',10,'x',250,250)")     # лише квитанція
    cat.x("INSERT INTO replicas VALUES('o2','disk','CORRUPT',10,'x',210,950)")
    j = Journal(tmp_path / "j.sqlite")
    j.write_decisions([_decision("v1", "o1", 50), _decision("v1", "o2", 60)],
                      [_outcome("v1", "o1", 50), _outcome("v1", "o2", 60)], [])
    got = j.harvest(cat)
    assert got == {"queued": 1, "write_confirmed": 2, "policy_satisfied": 1, "read_verified": 1, "verify_failed": 1}
    assert j.harvest(cat) == dict.fromkeys(STAGES[1:], 0)              # повтор нічого не дублює
    counts = j.stage_counts()
    assert counts["encoded"] == 2 and counts["read_verified"] == 1
    assert j.scalar("SELECT value FROM stages WHERE subject='v1' AND stage='policy_satisfied'") == 300e-9
