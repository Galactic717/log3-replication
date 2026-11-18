"""Жорсткі правила мають перевагу над навчанням; drift і підозріла телеметрія; мітки без витоку."""
import json
import os
import stat

import numpy as np
import pytest

from log3.vault.capture import backup
from log3.vault.learn import features as F
from log3.vault.learn import lifecycle as L
from log3.vault.learn.actions import CostModel
from log3.vault.learn.journal import Journal, now_ns, unpack
from log3.vault.learn.models import FixedPolicy, save_artifact
from log3.vault.learn.settings import LearnDir, default_state, load_state, save_state
from log3.vault.replicate import run_queue
from log3.vault.repo import Repo, add_backend, new_config
from log3.vault.restore import restore
from scripts.vault_corpus import make_corpus

PASS = "rules passphrase"
ACTS = ["store:0", "zstd:1", "zstd:3", "zstd:7", "zstd:12"]


def _repo(tmp_path, monkeypatch, name="repo"):
    monkeypatch.setenv("LOG3_VAULT_PASSPHRASE", PASS)
    cfg = new_config(min_copies=1, chunk_size=64 * 1024, kdf_n=2 ** 10)
    cfg["learning"] = {"actions": ACTS, "probe": {"rate": 1.0, "max_load_per_cpu": 100.0, "max_cpu_fraction": 4,
                                 "max_chunks_per_operation": 100000}, "drift": {"min_chunks": 16}}
    r = Repo.create(tmp_path / name, cfg, PASS.encode())
    add_backend(r, {"name": "d", "type": "localdir", "path": str(tmp_path / f"{name}-disk"), "failure_domain": "a"})
    return r


def _activate(repo, action, extra_meta=None):
    d = LearnDir(repo.path)
    mid = save_artifact(d.models, FixedPolicy(ACTS, CostModel(), {"action": action}), extra_meta or {})
    save_state(d, {**default_state(), "mode": "active", "active_model": mid, "active_since_ns": now_ns()},
               event="test")
    return mid


def _snapshot(root):
    return {p.relative_to(root).as_posix(): (p.read_bytes(), stat.S_IMODE(p.stat().st_mode), p.stat().st_mtime_ns)
            for p in sorted(root.rglob("*")) if p.is_file()}


def test_learning_cannot_touch_deletion_retention_copies_permissions_or_encryption(tmp_path, monkeypatch):
    repo = _repo(tmp_path, monkeypatch)
    src = tmp_path / "src"
    make_corpus(src, 1, 1.5, max_file_kib=64)
    (src / "secret.txt").write_bytes(b"confidential plaintext marker " * 4000)
    os.chmod(src / "secret.txt", 0o600)
    before_src, before_cfg = _snapshot(src), json.loads((repo.path / "config.json").read_text())
    # «шкідливий» артефакт: зайві поля про політику мають бути проігноровані
    _activate(repo, "zstd:12", {"policy": {"min_copies": 0, "retention_days": 0}, "delete": ["*"],
                                "crypto": {"suite": "none"}, "permissions": "0777"})
    vid, _ = backup(repo, src)
    run_queue(repo)
    after_cfg = json.loads((repo.path / "config.json").read_text())
    assert {k: after_cfg[k] for k in ("policy", "crypto", "limits", "backends")} == \
        {k: before_cfg[k] for k in ("policy", "crypto", "limits", "backends")}
    assert _snapshot(src) == before_src                                      # джерело не змінено й не видалено
    v = repo.cat.one("SELECT * FROM versions WHERE version_id=?", (vid,))
    assert v["state"] == "PROTECTED" and v["min_copies"] == 1
    assert v["retention_until_ns"] - v["accepted_ns"] == pytest.approx(30 * 86400e9, rel=1e-6)
    stored = b"".join(p.read_bytes() for p in (tmp_path / "repo-disk").rglob("*") if p.is_file())
    assert b"confidential plaintext marker" not in stored                     # шифрування не вимикається
    restore(repo, vid, tmp_path / "out")
    if os.name == "posix":                                   # права POSIX; у Windows режим 0600 не відображається
        assert stat.S_IMODE((tmp_path / "out" / "secret.txt").stat().st_mode) == 0o600


def test_features_do_not_depend_on_chosen_action_and_labels_are_measured(tmp_path, monkeypatch):
    src = tmp_path / "src"
    make_corpus(src, 2, 1, max_file_kib=64)
    vectors = []
    for name, action in (("a", "zstd:12"), ("b", "store:0")):
        repo = _repo(tmp_path, monkeypatch, name)
        _activate(repo, action)
        backup(repo, src)
        j = Journal(LearnDir(repo.path).journal)
        rows = j.q("SELECT d.features, d.chosen_action, d.model_action, o.payload_len, o.action "
                   "FROM decisions d JOIN outcomes o USING(decision_id)")
        assert {r["chosen_action"] for r in rows} == {action} == {r["model_action"] for r in rows}
        data = j.labeled(feature_schema=F.schema_id(), actions=ACTS)
        col = ACTS.index(action)
        measured = {r["payload_len"] for r in rows if r["action"] == action}
        assert set(data["payload"][:, col].astype(int)) <= measured            # мітка — вимір, не прогноз
        assert not np.isnan(data["payload"]).any()                               # інші дії — з проб
        j.close()
        drop = [F.index(n) for n in ("probe_log_ns_per_byte", "queue_fill", "spool_fill", "load_per_cpu")]
        vectors.append(sorted(tuple(np.delete(unpack(r["features"]), drop).round(6)) for r in rows))
    assert vectors[0] == vectors[1]                                              # вибір дії не протікає в ознаки


def test_psi_detects_shift():
    same = [0.1] * 10
    assert L.psi(same, same) == pytest.approx(0.0, abs=1e-9)
    assert L.psi([1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]) > 1.0


def test_feature_drift_freezes_updates_and_falls_back(tmp_path, monkeypatch):
    repo = _repo(tmp_path, monkeypatch)
    ref = {k: [1.0] + [0.0] * 9 for k in L.DRIFT_FEATURES}                   # еталон: «усе дуже стисливе»
    mid = _activate(repo, "zstd:1", {"reference": ref})
    src = tmp_path / "src"
    src.mkdir()
    for i in range(30):
        (src / f"r{i}.bin").write_bytes(os.urandom(64 * 1024))
    backup(repo, src)
    h = L.health(repo)
    assert h["status"] == "frozen" and "PSI" in h["reason"]
    st = load_state(LearnDir(repo.path))[0]
    assert st["frozen"] and st["active_model"] == mid
    (src / "new.bin").write_bytes(os.urandom(64 * 1024))
    vid, _ = backup(repo, src)
    j = Journal(LearnDir(repo.path).journal)
    rows = j.q("SELECT mode, fallback FROM decisions WHERE version_id=?", (vid,))
    assert rows and all(r["mode"] == "baseline" and "заморожено" in r["fallback"] for r in rows)


def test_clock_going_backwards_quarantines_operation(tmp_path, monkeypatch):
    repo = _repo(tmp_path, monkeypatch)
    src = tmp_path / "src"
    make_corpus(src, 3, 0.5, max_file_kib=64)
    backup(repo, src)
    j = Journal(LearnDir(repo.path).journal)
    j.db.execute("UPDATE operations SET finished_ns=finished_ns+3600*1000000000")   # «майбутня» операція
    j.close()
    (src / "more.log").write_bytes(b"line\n" * 20000)
    vid, _ = backup(repo, src)
    j = Journal(LearnDir(repo.path).journal)
    assert j.scalar("SELECT quarantine FROM operations WHERE version_id=?", (vid,)) == 1
    assert "час ОС" in load_state(LearnDir(repo.path))[0]["freeze_reason"]
