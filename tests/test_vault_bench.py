"""Відправний сценарій етапу 2 і детермінованість синтетичного корпусу вимірювань."""
from scripts import vault_bench
from scripts.vault_corpus import make_corpus, mass_rewrite, mutate


def _snapshot(root):
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def test_corpus_is_deterministic_and_mutations_change_files(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    info = make_corpus(a, seed=7, scale_mib=2)
    make_corpus(b, seed=7, scale_mib=2)
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
