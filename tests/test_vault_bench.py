"""Відправний сценарій етапу 2 і детермінованість синтетичного корпусу вимірювань."""
from scripts import vault_bench
from scripts.vault_corpus import make_corpus, mass_rewrite, mutate


def _snapshot(root):
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def test_corpus_is_deterministic_and_mutations_change_files(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    info = make_corpus(a, seed=7, scale_mib=2, max_file_kib=128)
    make_corpus(b, seed=7, scale_mib=2, max_file_kib=128)
    assert _snapshot(a) == _snapshot(b) and info["files"] > 10
    before = _snapshot(a)
    assert mutate(a, seed=8, fraction=0.2) > 0 and _snapshot(a) != before
    assert mass_rewrite(b, seed=9) >= 0.8 * info["files"]


def test_baseline_backup_restart_verify_restore(tmp_path):
    r = vault_bench.baseline(2.0, 3, tmp_path)
    assert r["state_after_accept"] == "ACCEPTED_LOCAL"
    assert r["state_after_run"] == "PROTECTED" and r["policy_satisfied"]
    assert r["verify"]["ok"] == r["verify"]["checked"] > 0
    assert r["restore_identical"]
    assert r["incremental"]["state"] == "PROTECTED"
    assert 0 < r["storage"]["ratio_stored_to_plain"] < 1


def test_adaptive_scenario_compares_arms_and_reports(tmp_path):
    res = vault_bench.adaptive(1.0, 4, tmp_path, rounds=2, chunk_mib=0.25, max_file_kib=64, arms=["off", "defaults"])
    off, on = res["arms"]["off"]["totals"], res["arms"]["defaults"]["totals"]
    assert off["stored_bytes"] > 0 and on["plain_bytes"] == off["plain_bytes"]
    assert res["arms"]["defaults"]["learning_status"]["journal"]["decisions"] > 0
    assert "defaults" in res["vs_off"]
    res["environment"] = vault_bench.environment()
    md = vault_bench.summary_md(res, "тест")
    assert "| off |" in md and "defaults проти off" in md
    import json
    assert json.loads(vault_bench.dump_json(res)) == json.loads(json.dumps(res))
