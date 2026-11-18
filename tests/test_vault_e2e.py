"""Файловий режим наскрізно. Кожна команда CLI і вузол — окремі процеси, тож стан переживає
перезапуск лише через диск (каталог SQLite, spool, backend-и), а не через пам'ять."""
import json
import os
import shutil
import sys

import pytest
from vault_env import PASSPHRASE, TOKEN, Env, all_bytes, make_tree, tree

SECRET_TEXT = "конфіденційний рядок".encode()


@pytest.fixture
def env(tmp_path):
    e = Env(tmp_path)
    yield e
    e.stop_node()


def test_backup_replicate_verify_restore(env, tmp_path):
    src = make_tree(tmp_path)
    env.setup()
    code, out = env.cli("backup", src, "--wait")
    v = out["version"]
    assert code == 0 and v["state"] == "PROTECTED" and v["protection_level"] == "protected"
    assert v["policy_satisfied"] is True and v["copies"]["objects_under_replicated"] == 0
    assert v["copies"]["confirmed_copies_min"] == 2 and v["files"] == 6 and v["skipped_files"] == 0
    # однаковий вміст під двома іменами зберігся один раз
    unique = {data for data in tree(src).values() if data}
    chunks = env.sql("SELECT COUNT(*) FROM objects WHERE kind=1")[0][0]
    assert chunks == sum(-(-len(d) // 262144) for d in unique) == v["copies"]["objects_total"] - 1
    assert env.ok("verify")["verify"]["ok"] == 2 * v["copies"]["objects_total"]
    out = env.ok("restore", v["version_id"], tmp_path / "out")["restore"]
    assert tree(tmp_path / "out") == tree(src)             # побайтово, з Unicode, порожніми файлом і каталогом
    assert out["files"] == 6 and out["dirs"] == 3
    # сховища не містять відкритого тексту й секретів; spool після PROTECTED порожній
    stored = all_bytes(env.disk, env.node_dir, env.repo)
    assert SECRET_TEXT not in stored and PASSPHRASE.encode() not in stored and TOKEN.encode() not in stored
    assert not list((env.repo / "spool" / "objects").glob("*/*"))
    env.ok("restore", "latest", tmp_path / "one", "--path", "вкладений/глибше")
    assert list(tree(tmp_path / "one")) == ["вкладений", "вкладений/глибше", "вкладений/глибше/таблиця.xlsx"]
    m = env.ok("status")["metrics"]
    assert m["queue_jobs"] == 0 and m["under_replicated_versions"] == 0 and m["last_verify_at"]
    assert m["last_time_to_protected_seconds"] > 0 and m["last_restore_duration_seconds"] >= 0


def test_restart_after_local_accept_and_source_disappears(env, tmp_path):
    src = make_tree(tmp_path)
    env.setup()
    expected = tree(src)
    code, out = env.cli("backup", src)                     # лише приймання й черга
    vid = out["version"]["version_id"]
    assert code == 0 and out["waited"] is False
    assert out["version"]["state"] == "ACCEPTED_LOCAL" and out["version"]["policy_satisfied"] is False
    assert out["version"]["copies"]["confirmed_copies_min"] == 0
    assert env.cli("status", vid)[0] == 3                  # NOT_PROTECTED: прийнято локально — ще не захищено
    shutil.rmtree(src)                                     # джерело зникло після стійкого захоплення
    assert env.ok("restore", vid, tmp_path / "early")      # прийняті дані вже доступні
    assert tree(tmp_path / "early") == expected
    assert env.cli("run")[0] == 0                          # новий процес продовжує з диска
    assert env.ok("status", vid)["version"]["state"] == "PROTECTED"
    env.ok("restore", vid, tmp_path / "out")
    assert tree(tmp_path / "out") == expected


def test_lost_response_after_durable_write_is_idempotent(env, tmp_path):
    src = make_tree(tmp_path)
    env.setup(start=False)
    env.start_node(fault="node_drop_response_once")        # перший PUT записано стійко, відповідь загублено
    code, out = env.cli("backup", src, "--wait")
    total = out["version"]["copies"]["objects_total"]
    assert code == 0 and out["version"]["state"] == "PROTECTED"
    assert env.ok("status")["metrics"]["job_errors_total"] == 1
    # повтор не створив ні зайвих копій, ні зайвих посилань, ні другого списання квоти
    assert env.ledger("SELECT COUNT(*), SUM(puts) FROM charges") == [(total, total + 1)]
    on_disk = sum(p.stat().st_size for p in (env.node_dir / "objects").glob("*/*"))
    assert env.ledger("SELECT SUM(length) FROM charges")[0][0] == on_disk
    assert env.sql("SELECT COUNT(*) FROM replicas WHERE backend='node'")[0][0] == total
    assert env.sql("SELECT COUNT(*), COUNT(DISTINCT job_id) FROM jobs") == [(2 * total, 2 * total)]
    assert env.sql("SELECT COUNT(*) FROM versions")[0][0] == 1


def test_crash_between_block_write_and_catalog_update(env, tmp_path):
    src = make_tree(tmp_path)
    env.setup()
    code, _ = env.cli("backup", src, env={"LOG3_VAULT_CRASHPOINT": "capture.before_publish"})
    assert code == 77                                      # аварія процесу: блоки в spool, каталог про них не знає
    assert list((env.repo / "spool" / "objects").glob("*/*"))
    assert env.sql("SELECT state FROM versions") == [("CAPTURING",)]
    assert env.sql("SELECT COUNT(*) FROM objects")[0][0] == 0
    code, out = env.cli("repair")
    assert code == 0 and out["repair"]["failed_captures"] == 1 and out["repair"]["orphans_removed"] > 0
    assert not list((env.repo / "spool" / "objects").glob("*/*"))
    assert env.sql("SELECT state FROM versions") == [("FAILED",)]
    assert env.cli("backup", src, "--wait")[0] == 0
    env.ok("restore", "latest", tmp_path / "out")
    assert tree(tmp_path / "out") == tree(src)


def test_crash_after_remote_write_before_ack_is_recorded(env, tmp_path):
    src = make_tree(tmp_path)
    env.setup()
    env.set_limits(lease_seconds=0.0)                      # завислі RUNNING одразу повертаються в чергу
    vid = env.ok("backup", src)["version"]["version_id"]
    assert env.cli("run", env={"LOG3_VAULT_CRASHPOINT": "replicate.after_put_before_commit"})[0] == 77
    assert env.sql("SELECT COUNT(*) FROM replicas")[0][0] == 0      # записано на backend-и, але не підтверджено
    assert list((env.node_dir / "objects").glob("*/*")) and env.cli("status", vid)[0] == 3
    assert env.cli("run")[0] == 0
    total = env.ok("status", vid)["version"]["copies"]["objects_total"]
    assert env.ledger("SELECT COUNT(*) FROM charges")[0][0] == total
    assert env.ok("status")["metrics"]["queue_jobs"] == 0


def test_queue_and_disk_overflow_refuse_before_acceptance(env, tmp_path):
    src = make_tree(tmp_path)
    env.setup()
    first = env.ok("backup", src)["version"]["version_id"]         # прийнято й лежить у черзі
    (src / "новий.bin").write_bytes(os.urandom(300_000))
    for limits, code_name in (({"queue_max_jobs": 1}, "E_QUEUE_FULL"), ({"spool_max_bytes": 1000}, "E_SPOOL_FULL"),
                              ({"min_free_bytes": 10 ** 18}, "E_DISK_FULL")):
        spooled = sorted((env.repo / "spool" / "objects").glob("*/*"))
        env.set_limits(**limits)
        code, out = env.cli("backup", src)
        assert code == 5 and out["error"]["code"] == code_name, out         # явна відмова CAPACITY
        assert sorted((env.repo / "spool" / "objects").glob("*/*")) == spooled   # нічого не лишилось і не зникло
        env.set_limits(queue_max_jobs=200_000, spool_max_bytes=10 ** 12, min_free_bytes=0)
    assert env.sql("SELECT COUNT(*) FROM versions WHERE state='ACCEPTED_LOCAL'")[0][0] == 1
    (src / "новий.bin").unlink()
    env.ok("restore", first, tmp_path / "out")             # раніше прийнята версія ціла: тихого видалення нема
    assert tree(tmp_path / "out") == tree(src)
    assert env.cli("backup", src, "--wait")[0] == 0


def test_node_down_then_back_then_corrupted(env, tmp_path):
    src = make_tree(tmp_path)
    env.setup(start=False)                                 # вузол не відповідає
    code, out = env.cli("backup", src, "--wait")
    vid, v = out["version"]["version_id"], out["version"]
    assert code == 4 and v["state"] == "DEGRADED" and v["policy_satisfied"] is False    # ATTENTION, не «захищено»
    assert v["locally_accepted"] is True and v["jobs_needing_attention"] > 0
    assert env.cli("status")[0] == 4
    env.ok("restore", vid, tmp_path / "degraded")          # уже прийняті дані доступні
    assert tree(tmp_path / "degraded") == tree(src)
    env.start_node()                                       # вузол повернувся
    code, out = env.cli("repair")
    assert code == 0 and out["repair"]["versions_not_protected"] == 0
    assert env.ok("status", vid)["version"]["state"] == "PROTECTED"
    # пошкодження копії на локальному диску і втрата об'єкта на вузлі
    victims = sorted((env.disk / "objects").glob("*/*"))
    data = victims[0].read_bytes()
    victims[0].write_bytes(data[:40] + bytes([data[40] ^ 0xFF]) + data[41:])
    other = next(p for p in sorted((env.node_dir / "objects").glob("*/*")) if p.name != victims[0].name)
    other.unlink()
    code, out = env.cli("verify")
    assert code == 7 and out["verify"]["corrupt"] == 1 and out["verify"]["missing"] == 1     # INTEGRITY
    assert env.cli("status", vid)[0] == 4
    env.ok("restore", vid, tmp_path / "from-good-copies")  # restore обходить погану копію
    assert tree(tmp_path / "from-good-copies") == tree(src)
    assert env.cli("repair")[0] == 0
    assert env.ok("verify")["verify"]["corrupt"] == 0
    assert env.ok("status", vid)["version"]["state"] == "PROTECTED"


def test_repeated_operation_and_incremental_backup(env, tmp_path):
    src = make_tree(tmp_path)
    env.setup()
    a = env.ok("backup", src, "--wait", "--op-id", "nightly-2026-11-16")
    b = env.ok("backup", src, "--wait", "--op-id", "nightly-2026-11-16")
    assert b["repeated_operation"] is True and b["version"]["version_id"] == a["version"]["version_id"]
    assert env.sql("SELECT COUNT(*) FROM versions")[0][0] == 1
    before = env.ledger("SELECT COUNT(*) FROM charges")[0][0]
    (src / "нотатки.txt").write_text("змінено\n", encoding="utf-8")
    c = env.ok("backup", src, "--wait")                    # нова версія: передається лише змінене
    assert c["version"]["version_id"] != a["version"]["version_id"]
    assert env.ledger("SELECT COUNT(*) FROM charges")[0][0] == before + 2      # один новий блок + маніфест
    files = {tuple(r) for r in env.sql("SELECT path, COUNT(DISTINCT file_id) FROM files GROUP BY path")}
    assert len(files) == 6 and all(n == 1 for _, n in files)                   # file_id стабільний між версіями
    env.ok("restore", a["version"]["version_id"], tmp_path / "old")
    env.ok("restore", c["version"]["version_id"], tmp_path / "new")
    assert (tmp_path / "old" / "нотатки.txt").stat().st_size > 1000
    assert (tmp_path / "new" / "нотатки.txt").read_text(encoding="utf-8") == "змінено\n"


def test_same_disk_is_not_two_copies_and_min_copies_is_hard(env, tmp_path):
    src = make_tree(tmp_path)
    env.setup(node=False)
    out = env.ok("backend", "add", "disk2", "--type", "localdir", "--path", tmp_path / "disk2", "--domain", "disk-b")
    assert out["independent_failure_domains"] == 1         # дві назви доменів, але один фізичний пристрій
    code, out = env.cli("backup", src, "--wait")
    assert code == 6 and out["error"]["code"] == "E_POLICY"
    assert env.sql("SELECT COUNT(*) FROM versions")[0][0] == 0


def test_recover_after_local_disk_loss(env, tmp_path):
    src = make_tree(tmp_path)
    env.setup()
    vid = env.ok("backup", src, "--wait")["version"]["version_id"]
    shutil.rmtree(env.repo)                                # локальний диск із каталогом і spool втрачено
    for name, args in (("disk", ("--type", "localdir", "--path", env.disk, "--domain", "disk-a")),
                       ("node", ("--type", "node", "--url", f"https://127.0.0.1:{env.port}", "--ca-file",
                                 tmp_path / "certs" / "node.crt", "--domain", "site-b"))):
        repo2 = tmp_path / f"recovered-{name}"
        out = env.ok("recover", "--repo", repo2, name, *args)
        assert out["versions"] == 1 and out["skipped"] == 0
        env.ok("restore", "--repo", repo2, vid, tmp_path / f"out-{name}")
        assert tree(tmp_path / f"out-{name}") == tree(src)
    code, out = env.cli("status", "--repo", tmp_path / "recovered-disk", vid)
    assert code == 4 and out["version"]["state"] == "DEGRADED"     # одна відома копія — це не «захищено»
    assert env.cli("repair", "--repo", tmp_path / "recovered-disk", "--all-backends")[0] == 0
    assert env.ok("status", "--repo", tmp_path / "recovered-disk", vid)["version"]["state"] == "PROTECTED"
    code, out = env.cli("recover", "--repo", tmp_path / "bad", "disk", "--type", "localdir", "--path", env.disk,
                        "--domain", "disk-a", env={"LOG3_VAULT_PASSPHRASE": "wrong"})
    assert code == 8 and out["error"]["code"] == "E_KEY"


def test_restore_refuses_overwrite_symlinks_and_wrong_key(env, tmp_path):
    src = make_tree(tmp_path)
    env.setup(node=False, min_copies=1)
    vid = env.ok("backup", src, "--wait")["version"]["version_id"]
    out_dir = tmp_path / "out"
    env.ok("restore", vid, out_dir)
    code, out = env.cli("restore", vid, out_dir)
    assert code == 1 and out["error"]["code"] == "E_EXISTS"        # неочікуваний перезапис заборонено
    (out_dir / "порожній.txt").write_text("локальна правка", encoding="utf-8")
    env.ok("restore", vid, out_dir, "--overwrite")
    assert tree(out_dir) == tree(src)
    assert not list(out_dir.glob(".log3-restore-*"))               # staging прибрано
    code, out = env.cli("restore", vid, tmp_path / "x", env={"LOG3_VAULT_PASSPHRASE": "wrong"})
    assert code == 8 and not (tmp_path / "x").exists()
    if os.name == "posix":
        trap, outside = tmp_path / "trap", tmp_path / "outside"
        trap.mkdir()
        outside.mkdir()
        (trap / "вкладений").symlink_to(outside, target_is_directory=True)
        code, out = env.cli("restore", vid, trap)
        assert code == 1 and out["error"]["code"] == "E_SYMLINK"
        assert not list(outside.iterdir()) and list(trap.iterdir()) == [trap / "вкладений"]


def test_source_symlink_is_skipped_and_reported(env, tmp_path):
    if os.name != "posix":
        pytest.skip("символічні посилання без прав адміністратора — лише POSIX")
    src = make_tree(tmp_path)
    (src / "посилання").symlink_to(tmp_path)
    env.setup(node=False, min_copies=1)
    code, out = env.cli("backup", src, "--wait")
    assert code == 9 and out["version"]["state"] == "PROTECTED" and out["version"]["skipped_files"] == 1   # PARTIAL
    env.ok("restore", "latest", tmp_path / "out")
    assert "посилання" not in tree(tmp_path / "out")


def test_gc_respects_retention_keep_last_and_live_objects(env, tmp_path):
    src = make_tree(tmp_path)
    env.setup()
    env.set_policy(retention_days=0, keep_last=1)
    old = env.ok("backup", src, "--wait")["version"]["version_id"]
    (src / "program.exe").write_bytes(os.urandom(100_000))
    new = env.ok("backup", src, "--wait")["version"]["version_id"]
    pending = env.ok("backup", src, "--name", "інший набір")["version"]["version_id"]   # незавершена реплікація
    report = env.ok("gc")["gc"]
    assert report["applied"] is False and report["versions"] == 1 and report["objects"] == 2
    count = len(list((env.disk / "objects").glob("*/*")))
    assert env.ok("gc", "--apply")["gc"]["objects"] == 2           # старий блок exe і старий маніфест
    assert len(list((env.disk / "objects").glob("*/*"))) == count - 2
    states = dict(env.sql("SELECT version_id, state FROM versions"))
    assert states == {old: "DELETED", new: "PROTECTED", pending: "ACCEPTED_LOCAL"}
    assert env.cli("run")[0] == 0
    for vid in (new, pending):
        env.ok("restore", vid, tmp_path / f"out-{vid[:6]}")
        assert tree(tmp_path / f"out-{vid[:6]}") == tree(src)
    assert env.ok("verify")["verify"]["missing"] == 0


def test_huffman_codec_and_custom_advisor_hook(env, tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.txt").write_bytes(b"abracadabra " * 3000)
    (tmp_path / "my_advisor.py").write_text(
        "import json, os\n"
        "from log3.vault.advisor import Recommendation\n"
        "class Advisor:\n"
        "    def recommend(self, context):\n"
        "        return Recommendation('huffman-log3', 0)\n"
        "    def observe(self, outcome):\n"
        "        with open(os.environ['ADVISOR_LOG'], 'a') as f:\n"
        "            row = [outcome.codec_used, outcome.raw_len_bytes, outcome.stored_len_bytes]\n"
        "            f.write(json.dumps(row) + '\\n')\n")
    env.setup(node=False, min_copies=1)
    cfg = json.loads((env.repo / "config.json").read_text(encoding="utf-8"))
    cfg["compression"]["advisor"] = "my_advisor:Advisor"
    (env.repo / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    extra = {"PYTHONPATH": os.pathsep.join([env.env["PYTHONPATH"], str(tmp_path)]),
             "ADVISOR_LOG": str(tmp_path / "advisor.log")}
    assert env.cli("backup", src, "--wait", env=extra)[0] == 0
    seen = [json.loads(line) for line in (tmp_path / "advisor.log").read_text().splitlines()]
    assert seen and seen[0][0] == "huffman-log3" and seen[0][2] < seen[0][1]
    env.ok("restore", "latest", tmp_path / "out")          # restore не потребує advisor: модуля в шляху вже нема
    assert tree(tmp_path / "out") == tree(src)


@pytest.mark.skipif(sys.platform != "linux", reason="пікову пам'ять процесу читаємо через wait4 (Linux)")
def test_large_file_restore_with_bounded_memory(env, tmp_path):
    import subprocess

    size = int(os.environ.get("LOG3_VAULT_BIG_MIB", "192")) << 20
    src = tmp_path / "big"
    src.mkdir()
    with open(src / "big.bin", "wb") as f:
        block = os.urandom(1 << 20)
        for i in range(size >> 20):
            f.write(i.to_bytes(8, "big") + block[8:])      # блоки різні: дедуплікація не допомагає
    env.setup(node=False, min_copies=1)
    assert env.cli("backup", src, "--wait")[0] == 0

    def peak_mib(*args) -> float:
        p = subprocess.Popen(env.argv(*args), env=env.env, stdout=subprocess.DEVNULL)
        _, status, usage = os.wait4(p.pid, 0)
        p.returncode = os.waitstatus_to_exitcode(status)
        assert p.returncode == 0
        return usage.ru_maxrss / 1024

    baseline = peak_mib("status")
    peak = peak_mib("restore", "latest", tmp_path / "out")
    assert (tmp_path / "out" / "big.bin").stat().st_size == size
    with open(src / "big.bin", "rb") as a, open(tmp_path / "out" / "big.bin", "rb") as b:
        while block := a.read(1 << 24):
            assert block == b.read(1 << 24)
    assert peak - baseline < 48, (baseline, peak)          # файл 192 МіБ, приріст пам'яті — кілька блоків
