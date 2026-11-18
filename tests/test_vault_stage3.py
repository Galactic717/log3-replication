"""Етап 3: ключі (ротація, зміна фрази, експорт, межа nonce), підпис службових файлів, міграція
каталогу, ролі й ліміти вузла, операторські команди. Кожна команда — новий процес CLI."""
from __future__ import annotations

import http.client
import json
import os
import sqlite3
import ssl

import pytest

from log3.vault.keys import load_export
from tests.vault_env import PASSPHRASE, Env, make_tree, tree


@pytest.fixture
def env(tmp_path):
    e = Env(tmp_path)
    yield e
    e.stop_node()


def _restore_equal(env, ref, src, dest, **kw):
    env.ok("restore", ref, dest, **kw)
    return tree(dest) == tree(src)


def test_rotation_passphrase_change_and_old_versions(env, tmp_path):
    env.setup(node=False, min_copies=1)
    src = make_tree(tmp_path)
    v1 = env.ok("backup", src, "--wait")["version"]["version_id"]
    old_key = env.ok("key", "status")["keys"]["active_key_id"]
    rot = env.ok("key", "rotate")["keys"]
    assert rot["previous_key_id"] == old_key and rot["active_key_id"] != old_key
    (src / "новий.txt").write_text("після ротації", encoding="utf-8")
    v2 = env.ok("backup", src, "--wait")["version"]["version_id"]
    st = env.ok("key", "status")["keys"]
    assert st["keyring"] and {k["key_id"] for k in st["keys"]} == {old_key, rot["active_key_id"]}
    new_pass = "зовсім інша довга фраза 2026"
    env.ok("key", "passphrase", env={"LOG3_VAULT_NEW_PASSPHRASE": new_pass})
    assert env.cli("restore", v1, tmp_path / "old-pass")[0] == 8          # стара фраза більше не відкриває
    new = {"LOG3_VAULT_PASSPHRASE": new_pass}
    env.ok("restore", v1, tmp_path / "r1", env=new)
    env.ok("restore", v2, tmp_path / "r2", env=new)
    assert tree(tmp_path / "r2") == tree(src)
    assert "новий.txt" not in tree(tmp_path / "r1")
    # аварійний експорт: окрема фраза, усі ключі
    out = tmp_path / "keys.export"
    env.ok("key", "export", "--out", out, env={**new, "LOG3_VAULT_EXPORT_PASSPHRASE": "фраза експорту 123456"})
    repo_id, keys = load_export(out.read_bytes(), "фраза експорту 123456".encode())
    assert {old_key, rot["active_key_id"]} <= set(keys.keys)
    assert env.cli("key", "export", "--out", out, env=new)[0] in (1, 8)     # не перезаписує й без фрази не працює


def test_nonce_limit_rotates_key_automatically(env, tmp_path):
    env.setup(node=False, min_copies=1)
    cfg_path = env.repo / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg["crypto"]["max_objects_per_key"] = 3
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    src = make_tree(tmp_path)
    first = env.ok("key", "status")["keys"]["active_key_id"]
    v1 = env.ok("backup", src, "--wait")["version"]["version_id"]          # > 3 об'єктів
    before = tree(src)
    (src / "ще.bin").write_bytes(os.urandom(1000))
    env.ok("backup", src, "--wait")
    st = env.ok("key", "status")["keys"]
    assert st["active_key_id"] != first and st["keys"][0]["sealed_total"] >= 3
    assert any(e[0] == "key_rotated" for e in env.sql("SELECT kind FROM events"))
    env.ok("restore", v1, tmp_path / "r")                                    # стара версія читається
    assert tree(tmp_path / "r") == before


def test_signed_meta_rejects_tampering_and_legacy_needs_flag(env, tmp_path):
    env.setup(node=False, min_copies=1)
    src = make_tree(tmp_path)
    vid = env.ok("backup", src, "--wait")["version"]["version_id"]
    meta = env.disk / "meta"
    cfg_raw = (meta / "repo-config.json").read_bytes()
    assert b"meta_mac" in cfg_raw and (meta / f"version-{vid}.json").read_bytes().find(b"meta_mac") > 0
    # підміна політики на backend-і: recover відмовляє
    doc = json.loads(cfg_raw)
    doc["policy"]["retention_days"] = 0
    (meta / "repo-config.json").write_text(json.dumps(doc), encoding="utf-8")
    code, out = env.cli("recover", "disk", "--type", "localdir", "--path", env.disk, "--domain", "disk-a",
                        "--repo", tmp_path / "rec1")
    assert code == 7 and out["error"]["code"] == "E_META"
    (meta / "repo-config.json").write_bytes(cfg_raw)
    # переставлений покажчик іншої версії відхиляється, справжній — імпортується
    (src / "x.txt").write_text("x", encoding="utf-8")
    vid2 = env.ok("backup", src, "--wait")["version"]["version_id"]
    (meta / f"version-{vid}.json").write_bytes((meta / f"version-{vid2}.json").read_bytes())
    out = env.ok("recover", "disk", "--type", "localdir", "--path", env.disk, "--domain", "disk-a",
                 "--repo", tmp_path / "rec2")
    assert out["rejected_meta"] == 1 and out["versions"] == 1
    # непідписані файли (сховище до етапу 3) — лише з явним прапорцем
    unsigned = {k: v for k, v in json.loads(cfg_raw).items() if k != "meta_mac"}
    (meta / "repo-config.json").write_text(json.dumps(unsigned), encoding="utf-8")
    code, out = env.cli("recover", "disk", "--type", "localdir", "--path", env.disk, "--domain", "disk-a",
                        "--repo", tmp_path / "rec3")
    assert code == 7
    out = env.ok("recover", "disk", "--type", "localdir", "--path", env.disk, "--domain", "disk-a",
                 "--repo", tmp_path / "rec4", "--accept-legacy-meta")
    assert out["legacy_unsigned_meta"] >= 1 and out["versions"] >= 1


def _downgrade_to_schema1(db):
    con = sqlite3.connect(db)
    con.execute("ALTER TABLE objects DROP COLUMN key_id")
    con.execute("ALTER TABLE versions DROP COLUMN pointer")
    con.execute("UPDATE meta SET value='1' WHERE key='schema_version'")
    con.commit()
    con.close()


def test_migration_backup_crash_and_restore(env, tmp_path):
    env.setup(node=False, min_copies=1)
    src = make_tree(tmp_path)
    vid = env.ok("backup", src, "--wait")["version"]["version_id"]
    db = env.repo / "catalog.sqlite"
    _downgrade_to_schema1(db)
    code, _ = env.cli("status", env={"LOG3_VAULT_CRASHPOINT": "catalog.migrate_before_commit"})
    assert code == 77                                                     # аварія посеред міграції
    assert env.sql("SELECT value FROM meta WHERE key='schema_version'") == [("1",)]   # транзакцію відкочено
    st = env.ok("migrate", "status")
    assert st["schema_version"] == 2 and st["backups"]
    assert env.sql("SELECT COUNT(*) FROM objects WHERE key_id IS NULL") == [(0,)]
    assert _restore_equal(env, vid, src, tmp_path / "r")
    assert env.ok("migrate", "restore", "--file", st["backups"][0], "--yes")["restored"]
    assert env.sql("SELECT value FROM meta WHERE key='schema_version'") == [("1",)]


def _node_request(env, method, path, token, body=b"", declared=None):
    """Код відповіді вузла; declared — заявлена довжина без надсилання тіла (перевірка ліміту)."""
    ctx = ssl.create_default_context(cafile=str(env.tmp / "certs" / "node.crt"))
    conn = http.client.HTTPSConnection("127.0.0.1", env.port, context=ctx, timeout=10)
    try:
        conn.putrequest(method, path)
        conn.putheader("Authorization", f"Bearer {token}")
        if method == "PUT":
            conn.putheader("Content-Length", str(len(body) if declared is None else declared))
        conn.endheaders()
        if declared is None and body:
            conn.send(body)
        return conn.getresponse().status
    finally:
        conn.close()


def test_node_roles_revocation_and_limits(env, tmp_path):
    tokens = tmp_path / "node-tokens.json"
    for name, role in (("client", "writer"), ("audit", "reader"), ("ops", "admin")):
        env.ok("node-token", "add", name, "--role", role, "--tokens-file", tokens, "--out", tmp_path / f"{name}.tok")
    tok = {n: (tmp_path / f"{n}.tok").read_text() for n in ("client", "audit", "ops")}
    for v in tok.values():
        assert v not in tokens.read_text(encoding="utf-8")                 # у файлі лише дайджести
    env.start_node(tokens_file=tokens)
    oid = "ab" * 32
    assert _node_request(env, "PUT", "/v1/meta/test", tok["audit"], b"x") == 403
    assert _node_request(env, "PUT", "/v1/meta/test", tok["client"], b"x") == 200
    assert _node_request(env, "DELETE", "/v1/meta/test", tok["client"]) == 403
    assert _node_request(env, "GET", f"/v1/stat/{oid}", tok["audit"]) == 404
    assert _node_request(env, "PUT", "/v1/meta/big", tok["client"], declared=2 << 20) == 413
    assert _node_request(env, "DELETE", "/v1/meta/test", tok["ops"]) == 200
    env.ok("node-token", "revoke", "client", "--tokens-file", tokens)
    assert _node_request(env, "GET", "/v1/info", tok["client"]) == 401     # без перезапуску вузла
    for _ in range(12):
        _node_request(env, "GET", "/v1/info", "wrong")
    assert _node_request(env, "GET", "/v1/info", tok["ops"]) == 429         # IP заблоковано після серії невдач


def test_operator_commands_find_drill_health_prometheus(env, tmp_path):
    env.setup(node=False, min_copies=1)
    src = make_tree(tmp_path)
    env.ok("backup", src, "--wait", "--name", "документи")
    code, out = env.cli("health")
    assert code == 4 and {c["check"] for c in out["health"]["checks"] if not c["ok"]} >= {"verify_age", "drill"}
    env.ok("verify")
    drill = env.ok("drill")["drill"]
    assert drill["ok"] and drill["files"] >= 5 and drill["mismatched"] == 0
    assert env.ok("health")["health"]["healthy"]
    assert env.ok("config", "check")["config_valid"]
    assert env.ok("doctor")["doctor"]["ok"]
    found = env.ok("find", "--path", "*.pdf")["find"]
    assert [e["path"] for e in found["entries"]] == ["звіт ☃ 2026.pdf"] and found["manifests_decrypted"] == 1
    assert env.ok("find", "--state", "protected", "--name", "докум")["find"]["matched_versions"] == 1
    assert env.ok("find", "--since", "2100-01-01T00:00:00Z")["find"]["matched_versions"] == 0
    prom = env.ok("metrics", "--format", "prometheus")["prometheus"]
    assert 'log3_vault_versions{state="PROTECTED"} 1.0' in prom and "log3_vault_last_drill_ok 1.0" in prom
    assert str(src) not in prom and PASSPHRASE not in prom
    assert env.ok("queue")["queue"]["capacity_jobs"] > 0
    acc = env.ok("doctor")["doctor"]["checks"]
    assert next(c for c in acc if c["check"] == "repo_access_owner_only")["ok"] in (True, None)
