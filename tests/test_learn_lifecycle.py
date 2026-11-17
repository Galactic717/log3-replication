"""Цикл baseline -> shadow -> оцінювання -> active -> rollback і чесність оцінювання."""
import numpy as np
import pytest

from log3.vault.capture import backup
from log3.vault.errors import VaultError
from log3.vault.learn import lifecycle as L
from log3.vault.learn.actions import CostModel
from log3.vault.learn.journal import Journal
from log3.vault.learn.models import FixedPolicy, save_artifact
from log3.vault.learn.settings import LearnDir, default_state, load_state, priority, save_state
from log3.vault.replicate import run_queue
from log3.vault.repo import Repo, add_backend, new_config
from log3.vault.restore import restore
from scripts.vault_corpus import make_corpus, mutate

PASS = "lifecycle passphrase"
LEARNING = {"probe": {"rate": 1.0, "max_chunks_per_operation": 100000, "max_cpu_seconds_per_operation": 100,
                      "max_cpu_fraction": 4, "max_load_per_cpu": 100.0},
            "cost": {"cpu_second_equiv_bytes": 0},            # ціль «лише простір»: детермінована механіка
            "train": {"min_rows": 50, "min_groups": 5, "search_budget": 8},
            "gate": {"min_shadow_decisions": 20, "min_relative_gain": 0.01, "bootstrap": 100}}


def _repo(tmp_path, monkeypatch, learning=None):
    monkeypatch.setenv("LOG3_VAULT_PASSPHRASE", PASS)
    cfg = new_config(min_copies=1, chunk_size=64 * 1024, kdf_n=2 ** 10)
    cfg["learning"] = learning if learning is not None else LEARNING
    r = Repo.create(tmp_path / "repo", cfg, PASS.encode())
    add_backend(r, {"name": "d", "type": "localdir", "path": str(tmp_path / "disk"), "failure_domain": "a"})
    return r


def _rounds(repo, src, n, seed=0):
    vids = []
    for i in range(n):
        vids.append(backup(repo, src)[0])
        run_queue(repo)
        mutate(src, seed + i, 0.3, 0.2)
    return vids


@pytest.fixture
def collected(tmp_path, monkeypatch):
    repo = _repo(tmp_path, monkeypatch)
    src = tmp_path / "src"
    make_corpus(src, 5, 6, max_file_kib=96)
    _rounds(repo, src, 3)
    return repo, src


def test_split_is_time_ordered_and_group_exclusive():
    rng = np.random.default_rng(0)
    t = np.sort(rng.integers(0, 10**9, 400))
    groups, n = [], 0
    for _ in t:                                            # нові файли з'являються з часом, старі змінюються
        if n == 0 or rng.random() < 0.5:
            n += 1
            groups.append(f"g{n}")
        else:
            groups.append(f"g{rng.integers(1, n + 1)}")
    parts = L.split({"decided_ns": t, "groups": groups})
    tr, va, te = parts["train"], parts["validation"], parts["test"]
    assert t[tr].max() < t[va].min() and t[va].max() < t[te].min()
    sets = [{groups[i] for i in p} for p in (tr, va, te)]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert len(tr) + len(va) + len(te) + parts["dropped"] == len(t)


def test_bootstrap_gain_sign_and_interval():
    base = np.full(100, 10.0)
    better = np.full(100, 9.0)
    g = L.bootstrap_gain(base, better, [i % 10 for i in range(100)], 200)
    assert g["gain"] == pytest.approx(0.1) and g["low"] > 0
    assert L.bootstrap_gain(base, base + 1, list(range(100)), 200)["high"] < 0


def test_train_without_enough_data_keeps_baseline(tmp_path, monkeypatch):
    repo = _repo(tmp_path, monkeypatch)
    src = tmp_path / "src"
    make_corpus(src, 1, 0.3, max_file_kib=64)
    _rounds(repo, src, 1)
    rep = L.train(repo, force=True)
    assert rep["status"] == "insufficient_data"
    assert load_state(LearnDir(repo.path))[0]["mode"] == "baseline"


def test_full_cycle_shadow_evaluate_promote_rollback(collected, tmp_path):
    repo, src = collected
    rep = L.train(repo, force=True)
    assert rep["status"] == "candidate", rep["gate"]
    names = set(rep["validation"])
    assert {"store", "fixed zstd:3", "baseline (фактичні рішення політики)"} <= names
    assert any(n.startswith("ga:") for n in names) and any(n.startswith("random:") for n in names)
    assert rep["search"]["ga"]["evaluations"] == rep["search"]["random"]["evaluations"]
    prov = rep["provenance"]
    tr, va, te = prov["train"], prov["validation"], prov["test"]
    assert tr["to_ns"] < va["from_ns"] <= va["to_ns"] < te["from_ns"]
    d = LearnDir(repo.path)
    st = load_state(d)[0]
    assert st["mode"] == "shadow" and st["candidate_model"] == rep["model_id"]
    with pytest.raises(VaultError):
        L.promote(repo)                                    # без перевірки в shadow — ні
    _rounds(repo, src, 2, seed=50)
    ev = L.evaluate(repo)
    assert ev["status"] == "evaluated" and ev["passed"], ev
    out = L.promote(repo)
    assert out["active_model"] == rep["model_id"]
    vid = _rounds(repo, src, 1, seed=80)[0]
    j = Journal(d.journal)
    used = j.q("SELECT DISTINCT mode, model_id FROM decisions WHERE version_id=?", (vid,))
    j.close()
    assert [(r["mode"], r["model_id"]) for r in used] == [("active", rep["model_id"])]
    assert L.health(repo)["status"] == "healthy"
    back = L.rollback(repo, "перевірка")
    assert back["status"] == "baseline" and load_state(d)[0]["active_model"] is None
    restore(repo, vid, tmp_path / "restored")                # копії, створені моделлю, відновлюються без неї


def test_health_rolls_back_harmful_active_model(collected):
    repo, src = collected
    d = LearnDir(repo.path)
    acts = L.settings(repo.cfg)["actions"]
    mid = save_artifact(d.models, FixedPolicy(acts, CostModel(), {"action": "store:0"}), {})
    save_state(d, {**default_state(), "mode": "active", "active_model": mid, "active_since_ns": 1}, event="test")
    _rounds(repo, src, 1, seed=90)
    h = L.health(repo)
    assert h["status"] == "rolled_back" and h["realized_gain_vs_baseline"] < 0
    assert load_state(d)[0]["mode"] == "baseline"


def test_training_yields_to_restore_repair_and_full_queue(collected):
    repo, _ = collected
    with priority(repo.path, "restore"):
        with pytest.raises(VaultError) as e:
            L.train(repo, force=True)
    assert e.value.code == "E_LEARN_YIELD"
    repo.cfg["limits"]["queue_max_jobs"] = 1
    repo.cat.x("INSERT INTO jobs(job_id, oid, backend, state, priority, next_at_ns, enq_ns) "
               "VALUES('j1','o1','d','PENDING',1,0,0)")
    with pytest.raises(VaultError) as e:
        L.train(repo, force=True)
    assert e.value.code == "E_LEARN_YIELD"


def test_pause_freeze_disable_and_reset(collected, tmp_path):
    repo, src = collected
    L.set_flags(repo, "pause", paused=True)
    with pytest.raises(VaultError, match="паузі"):
        L.train(repo, force=True)
    L.set_flags(repo, "resume", paused=False)
    L.set_flags(repo, "freeze", frozen=True, freeze_reason="тест")
    with pytest.raises(VaultError, match="заморожено"):
        L.train(repo, force=True)
    assert L.unfreeze(repo)["frozen"] is False
    L.set_flags(repo, "disable", disabled=True)
    vid = _rounds(repo, src, 1, seed=70)[0]
    j = Journal(LearnDir(repo.path).journal)
    assert j.scalar("SELECT COUNT(*) FROM decisions WHERE version_id=?", (vid,)) == 0
    j.close()
    assert L.reset(repo)["reset"] and not LearnDir(repo.path).root.exists()
    restore(repo, vid, tmp_path / "after-reset")
    assert L.auto(repo)["health"]["status"] == "not_active"   # холодний старт знову


def test_set_mode_rules(collected):
    repo, _ = collected
    with pytest.raises(VaultError):
        L.set_mode(repo, "active")
    with pytest.raises(VaultError):
        L.set_mode(repo, "shadow")
    assert L.set_mode(repo, "baseline")["mode"] == "baseline"


def test_candidate_failing_shadow_is_rejected_and_baseline_stays(collected):
    repo, src = collected
    d = LearnDir(repo.path)
    acts = L.settings(repo.cfg)["actions"]
    mid = save_artifact(d.models, FixedPolicy(acts, CostModel(), {"action": "store:0"}), {})
    save_state(d, {**default_state(), "mode": "shadow", "candidate_model": mid, "candidate_set_ns": 1}, event="test")
    _rounds(repo, src, 1, seed=95)
    ev = L.evaluate(repo)
    assert ev["status"] == "rejected" and not ev["passed"] and ev["gain"] < 0
    st = load_state(d)[0]
    assert st["mode"] == "baseline" and st["candidate_model"] is None and st["candidate_eval"]["model_id"] == mid
    with pytest.raises(VaultError):
        L.promote(repo)
