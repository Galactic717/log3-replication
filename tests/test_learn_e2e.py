"""Навчання в реальних командах CLI, кожна команда — новий процес (стенд tests/vault_env.py)."""
import json
import subprocess
import sys

import pytest
from vault_env import ROOT, Env

from scripts.vault_corpus import make_corpus, mass_rewrite, mutate

FAST_LEARNING = ["probe.rate=1.0", "probe.max_cpu_fraction=4", "probe.max_load_per_cpu=100",
                 "probe.max_chunks_per_operation=100000", "train.min_rows=50", "train.min_groups=5",
                 "train.search_budget=8", "gate.min_shadow_decisions=20", "gate.min_relative_gain=0.005",
                 'actions=["store:0", "zstd:1", "zstd:3", "zstd:6", "zstd:12"]',
                 "gate.bootstrap=100", "cost.cpu_second_equiv_bytes=0"]


@pytest.fixture
def env(tmp_path):
    e = Env(tmp_path).setup(node=False, min_copies=1)
    sets = [x for item in FAST_LEARNING for x in ("--set", item)]
    e.ok("learn", "config", *sets)
    return e


def _status(env) -> dict:
    return env.ok("learn", "status")["learning"]


def _rounds(env, src, n, seed=0, wait=True):
    out = []
    for i in range(n):
        out.append(env.ok("backup", src, *(["--wait"] if wait else []))["version"]["version_id"])
        mutate(src, seed + i, 0.3, 0.2)
    return out


def test_cold_start_restart_and_restore_without_learning(env, tmp_path):
    src = tmp_path / "src"
    make_corpus(src, 2, 2, max_file_kib=64)
    out = env.ok("backup", src, "--wait")
    assert out["learning"]["enabled"] and out["learning"]["mode"] == "baseline" and out["learning"]["errors"] == 0
    st = _status(env)                                           # новий процес: стан і журнал збереглись
    assert st["state"]["mode"] == "baseline" and st["journal"]["decisions"] > 0 and "models" not in st
    assert st["journal"]["stages"]["policy_satisfied"] == 1     # зібрано після run усередині backup --wait
    assert env.ok("status")["learning"]["mode"] == "baseline"
    vid = out["version"]["version_id"]
    code = subprocess.run(                                       # restore, коли пакет навчання недоступний
        [sys.executable, "-c", "import sys; sys.modules['log3.vault.learn'] = None; "
         "from log3.vault.cli import main; sys.exit(main(sys.argv[1:]))", "restore", vid, str(tmp_path / "out"),
         "--json"], env=env.env, cwd=ROOT, capture_output=True, text=True).returncode
    assert code == 0 and (tmp_path / "out").exists()


def test_delayed_stages_arrive_later_and_are_not_duplicated(env, tmp_path):
    src = tmp_path / "src"
    make_corpus(src, 3, 1, max_file_kib=64)
    env.ok("backup", src)                                         # лише локальне приймання
    st = _status(env)["journal"]["stages"]
    assert st["queued"] == 0 and st["write_confirmed"] == 0       # стадій ще не зібрано
    env.ok("run")                                                 # реплікація + крок навчання (collect)
    st = _status(env)["journal"]["stages"]
    assert st["queued"] == 1 and st["policy_satisfied"] == 1 and st["write_confirmed"] > 0
    env.ok("verify")
    st2 = _status(env)["journal"]["stages"]
    assert st2["read_verified"] > 0 and st2["write_confirmed"] == st["write_confirmed"]
    env.ok("backup", src, "--op-id", "dup")
    env.ok("backup", src, "--op-id", "dup")                       # повтор операції
    assert _status(env)["journal"]["operations"] == 2


def test_damaged_model_falls_back_and_rollback_via_cli(env, tmp_path):
    learn = env.repo / "learn"
    (learn / "models" / "broken").mkdir(parents=True)
    (learn / "models" / "broken" / "model.json").write_text("{}")
    (learn / "state.json").write_text(json.dumps({"format": "log3-learn-state", "format_version": 1,
                                                  "mode": "active", "active_model": "broken"}))
    src = tmp_path / "src"
    make_corpus(src, 4, 1, max_file_kib=64)
    out = env.ok("backup", src, "--wait")                          # копіювання не зупиняється
    assert out["learning"]["fallback"] and out["learning"]["errors"] >= 1
    assert env.ok("status")["learning"]["errors_total"] >= 1
    assert env.ok("learn", "rollback")["learning"]["status"] == "baseline"
    (learn / "state.json").write_text("{garbage")
    assert env.ok("backup", src, "--wait")["version"]["state"] == "PROTECTED"
    assert _status(env)["state_error"]


def test_full_cycle_through_cli(env, tmp_path):
    src = tmp_path / "src"
    make_corpus(src, 5, 6, max_file_kib=96)
    _rounds(env, src, 3)
    rep = env.ok("learn", "train", "--force")["learning"]
    assert rep["status"] == "candidate", rep.get("gate")
    assert _status(env)["state"]["mode"] == "shadow"
    _rounds(env, src, 2, seed=40)
    assert env.ok("learn", "evaluate")["learning"]["passed"]
    assert env.ok("learn", "promote")["learning"]["status"] == "active"
    vid = _rounds(env, src, 1, seed=60)[0]
    env.ok("restore", vid, tmp_path / "r")
    st = _status(env)
    assert st["state"]["mode"] == "active" and st["state"]["active_model"] == rep["model_id"]
    assert env.ok("learn", "rollback", "--reason", "тест")["learning"]["status"] == "baseline"


def test_pause_limits_and_disable(env, tmp_path):
    env.ok("learn", "config", "--set", "probe.max_chunks_per_operation=1")
    src = tmp_path / "src"
    make_corpus(src, 6, 2, max_file_kib=64)
    _rounds(env, src, 1)
    assert env.sql("SELECT 1")                                     # каталог цілий
    j = env.repo / "learn" / "journal.sqlite"
    assert env.sql("SELECT MAX(probes) FROM operations", j)[0][0] <= 1
    env.ok("learn", "pause")
    _rounds(env, src, 1, seed=9)
    assert env.sql("SELECT probes FROM operations ORDER BY finished_ns DESC LIMIT 1", j)[0][0] == 0
    code, out = env.cli("learn", "train", "--force")
    assert code == 4 and out["error"]["code"] == "E_LEARN_PAUSED"
    env.ok("learn", "resume")
    env.ok("learn", "disable")
    n = env.sql("SELECT COUNT(*) FROM decisions", j)[0][0]
    _rounds(env, src, 1, seed=11)
    assert env.sql("SELECT COUNT(*) FROM decisions", j)[0][0] == n
    code, out = env.cli("learn", "config", "--set", "probe.rate=5")
    assert code == 6 and out["error"]["code"] == "E_CONFIG"         # некоректне значення не записується
    code, _ = env.cli("learn", "reset")
    assert code == 2
    assert env.ok("learn", "reset", "--yes")["learning"]["reset"]


def test_mass_rewrite_is_quarantined_and_freezes_updates(env, tmp_path):
    src = tmp_path / "src"
    make_corpus(src, 7, 3, max_file_kib=64)
    _rounds(env, src, 2)
    mass_rewrite(src, 99)
    out = env.ok("backup", src, "--wait")
    assert out["learning"]["quarantine"]
    st = _status(env)
    assert st["state"]["frozen"] and "переписування" in st["state"]["freeze_reason"] + "високоентроп"
    assert st["journal"]["quarantined_operations"] == 1
    code, out = env.cli("learn", "train", "--force")
    assert code == 4 and out["error"]["code"] == "E_LEARN_FROZEN"
    assert env.ok("learn", "unfreeze")["learning"]["frozen"] is False
    j = env.repo / "learn" / "journal.sqlite"
    assert env.sql("SELECT COUNT(*) FROM decisions WHERE quarantine=1", j)[0][0] > 0   # у навчання не йде


def _tree(root):
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def test_real_backup_restore_identical_with_adaptation_off_and_active(env, tmp_path):
    from log3.vault.learn.actions import CostModel
    from log3.vault.learn.models import FixedPolicy, save_artifact
    from log3.vault.learn.settings import LearnDir, default_state, save_state

    src = tmp_path / "src"
    make_corpus(src, 8, 2, max_file_kib=96)
    env.ok("learn", "disable")
    off = env.ok("backup", src, "--wait")["version"]["version_id"]
    env.ok("restore", off, tmp_path / "off")
    assert _tree(tmp_path / "off") == _tree(src)
    env.ok("learn", "enable")
    d = LearnDir(env.repo)
    acts = env.ok("learn", "config")["learning_config"]["actions"]
    mid = save_artifact(d.models, FixedPolicy(acts, CostModel(), {"action": "zstd:6"}), {})
    save_state(d, {**default_state(), "mode": "active", "active_model": mid, "active_since_ns": 1}, event="test")
    mutate(src, 81, 0.5, 0.2)
    on = env.ok("backup", src, "--wait")
    assert on["learning"]["mode"] == "active" and on["learning"]["model_id"] == mid
    env.ok("restore", on["version"]["version_id"], tmp_path / "on")
    assert _tree(tmp_path / "on") == _tree(src)
    env.ok("restore", off, tmp_path / "off-again")                   # стара версія — як і раніше
    assert _tree(tmp_path / "off-again") == _tree(tmp_path / "off")
