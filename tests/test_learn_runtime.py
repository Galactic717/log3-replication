"""Навчання в тракті capture: холодний старт і контрольований fallback замість зупинки копіювання."""
import json
import os

import pytest

from log3.vault.capture import backup
from log3.vault.learn.actions import CostModel
from log3.vault.learn.journal import Journal
from log3.vault.learn.models import FixedPolicy, save_artifact
from log3.vault.learn.settings import LearnDir, default_state, load_state, save_state
from log3.vault.replicate import run_queue
from log3.vault.repo import Repo, add_backend, new_config
from log3.vault.restore import restore

PASS = "runtime test passphrase"


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.setenv("LOG3_VAULT_PASSPHRASE", PASS)
    cfg = new_config(min_copies=1, chunk_size=64 * 1024, kdf_n=2 ** 10)
    r = Repo.create(tmp_path / "repo", cfg, PASS.encode())
    add_backend(r, {"name": "d", "type": "localdir", "path": str(tmp_path / "disk"), "failure_domain": "a"})
    return r


def _src(tmp_path, name="src"):
    src = tmp_path / name
    src.mkdir()
    (src / "a.log").write_bytes(b"2026-11-17 INFO svc=api latency_ms=12 status=ok\n" * 8000)
    (src / "b.bin").write_bytes(os.urandom(300_000))
    (src / "c.txt").write_bytes(os.urandom(100_000))                    # оманливе розширення
    return src


def _roundtrip(repo, src, tmp_path, out="out"):
    vid, _ = backup(repo, src)
    run_queue(repo, wait=True)
    assert repo.cat.scalar("SELECT state FROM versions WHERE version_id=?", (vid,)) == "PROTECTED"
    restore(repo, vid, tmp_path / out)
    for p in src.iterdir():
        assert (tmp_path / out / p.name).read_bytes() == p.read_bytes()
    return vid


def _events(repo, kind):
    return repo.cat.q("SELECT value, detail FROM events WHERE kind=?", (kind,))


def _set_state(repo, **kw):
    d = LearnDir(repo.path)
    save_state(d, {**default_state(), **kw}, event="test")


def test_cold_start_collects_measurements_without_model(repo, tmp_path):
    vid = _roundtrip(repo, _src(tmp_path), tmp_path)
    j = Journal(LearnDir(repo.path).journal)
    assert j.scalar("SELECT COUNT(*) FROM decisions WHERE version_id=?", (vid,)) > 0
    assert j.scalar("SELECT COUNT(DISTINCT mode) FROM decisions") == 1
    assert j.scalar("SELECT mode FROM decisions LIMIT 1") == "baseline"
    assert j.completeness()["decisions_without_outcome"] == 0
    assert j.scalar("SELECT COUNT(*) FROM probes") > 0                  # проби збирають повну інформацію
    assert j.scalar("SELECT COUNT(*) FROM lineage") == 3
    assert not _events(repo, "learning_errors")
    assert load_state(LearnDir(repo.path)) == (default_state(), None)   # state.json ще не створено


@pytest.mark.parametrize("damage", ["state", "model_missing", "model_corrupt", "journal", "config"])
def test_damaged_learning_falls_back_and_backup_continues(repo, tmp_path, damage):
    d = LearnDir(repo.path)
    if damage == "state":
        d.root.mkdir(parents=True)
        d.state.write_text("{broken")
    elif damage == "model_missing":
        _set_state(repo, mode="active", active_model="costtable-none")
    elif damage == "model_corrupt":
        mid = save_artifact(d.models, FixedPolicy(["store:0", "zstd:1"], CostModel(), {"action": "zstd:1"}), {})
        (d.models / mid / "arrays.npz").write_bytes(b"garbage")
        _set_state(repo, mode="active", active_model=mid)
    elif damage == "journal":
        d.root.mkdir(parents=True)
        d.journal.write_bytes(b"not a sqlite database" * 100)
    else:
        repo.cfg["learning"] = {"probe": {"rate": 7}}
        repo.save_config()
    _roundtrip(repo, _src(tmp_path), tmp_path)
    assert _events(repo, "learning_errors"), damage


def test_active_model_is_used_but_unprofitable_compression_becomes_store(repo, tmp_path):
    d = LearnDir(repo.path)
    acts = repo.cfg.get("learning", {}).get("actions") or ["store:0", "zstd:1", "zstd:3", "zstd:7", "zstd:12"]
    mid = save_artifact(d.models, FixedPolicy(acts, CostModel(), {"action": "zstd:12"}), {})
    _set_state(repo, mode="active", active_model=mid)
    vid = _roundtrip(repo, _src(tmp_path), tmp_path)
    j = Journal(d.journal)
    rows = j.q("SELECT d.chosen_action, o.codec_used, o.forced_store FROM decisions d JOIN outcomes o "
               "USING(decision_id) WHERE d.version_id=?", (vid,))
    assert {r["chosen_action"] for r in rows} == {"zstd:12"}
    assert any(r["forced_store"] and r["codec_used"] == "store" for r in rows)        # випадкові байти
    assert any(r["codec_used"] == "zstd" for r in rows)
    assert j.scalar("SELECT model_id FROM decisions LIMIT 1") == mid


def test_shadow_recommendation_does_not_change_working_decision(repo, tmp_path):
    d = LearnDir(repo.path)
    acts = ["store:0", "zstd:1", "zstd:3", "zstd:7", "zstd:12"]
    mid = save_artifact(d.models, FixedPolicy(acts, CostModel(), {"action": "zstd:12"}), {})
    _set_state(repo, mode="shadow", candidate_model=mid)
    vid = _roundtrip(repo, _src(tmp_path), tmp_path)
    rows = Journal(d.journal).q("SELECT policy_action, model_action, chosen_action FROM decisions WHERE version_id=?",
                                (vid,))
    assert rows and all(r["model_action"] == "zstd:12" and r["chosen_action"] == r["policy_action"] for r in rows)


def test_disabled_learning_leaves_no_trace(repo, tmp_path):
    repo.cfg["learning"] = {"enabled": False}
    repo.save_config()
    _roundtrip(repo, _src(tmp_path), tmp_path)
    assert not LearnDir(repo.path).journal.exists()


def test_restore_needs_no_learning_state(repo, tmp_path):
    vid = _roundtrip(repo, _src(tmp_path), tmp_path)
    import shutil
    shutil.rmtree(LearnDir(repo.path).root)
    restore(repo, vid, tmp_path / "again")
    assert (tmp_path / "again" / "a.log").exists()


def test_repeated_operation_is_not_journaled_twice(repo, tmp_path):
    src = _src(tmp_path)
    vid, _ = backup(repo, src, op_id="same-op")
    again, repeated = backup(repo, src, op_id="same-op")
    assert repeated and again == vid
    j = Journal(LearnDir(repo.path).journal)
    assert j.scalar("SELECT COUNT(*) FROM operations") == 1
    detail = json.loads(_events(repo, "learning")[-1]["detail"])
    assert detail["enabled"] and detail["mode"] == "baseline"
