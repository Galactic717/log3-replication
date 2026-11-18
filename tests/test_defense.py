"""Модуль захисту: приховування секретів і розбір виводу."""
import sys
from pathlib import Path

from log3.defense.runner import Ctx, Runner, Step, V, last_json

ROOT = Path(__file__).resolve().parent.parent


def test_secrets_are_redacted_in_commands_output_and_journal(tmp_path):
    ctx = Ctx(root=ROOT, work=tmp_path, profile="quick", env={}, secrets=["top-secret-phrase"])
    runner = Runner(ctx, out_dir=tmp_path / "out", quiet=True)
    code = "print('top-secret-phrase'); print('{\"a\": 1}')"
    rec = runner.run_step(Step("друк секрету", [sys.executable, "-c", code],
                               env={"LOG3_VAULT_PASSPHRASE": "top-secret-phrase", "MODE": "demo"},
                               check=lambda c, r: f"a={r.data['a']}"), 1, 1)
    runner.stages.append({"stage": "t", "steps": [rec], "status": "passed"})
    summary = runner.finish({})
    assert rec["status"] == "passed" and rec["verified"] == "a=1" and rec["env"] == {"MODE": "demo"}
    for path in (runner.log_path, runner.summary_path):
        assert "top-secret-phrase" not in path.read_text(encoding="utf-8")
    assert summary["totals"]["steps_failed"] == 0


def test_failed_check_and_unexpected_exit_code_are_reported(tmp_path):
    ctx = Ctx(root=ROOT, work=tmp_path, profile="quick", env={})
    runner = Runner(ctx, out_dir=tmp_path / "out", quiet=True)
    bad_code = runner.run_step(Step("код 3", [sys.executable, "-c", "raise SystemExit(3)"]), 1, 2)
    bad_check = runner.run_step(Step("перевірка", [sys.executable, "-c", "pass"],
                                     check=lambda c, r: (_ for _ in ()).throw(AssertionError("ні"))), 2, 2)
    assert bad_code["status"] == "failed" and bad_code["exit_code"] == 3
    assert bad_check["status"] == "failed" and "ні" in bad_check["verified"]
    ctx.vars["v"] = "abc"
    assert ctx.show_argv(["x", V("v"), V("later")]) == "x abc <later>"
    assert last_json('text\n{"ok": true}\n') == {"ok": True}
    runner.finish({})
