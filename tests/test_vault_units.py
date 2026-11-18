"""Файловий режим: кодеки, крипто, маніфест, стани, advisor, справедливість черги."""
import json
import os

import pytest

from log3.vault import codecs, manifest
from log3.vault.advisor import CompressionContext, Recommendation, SafeAdvisor, load_advisor, sample_entropy
from log3.vault.backends import node_server
from log3.vault.backends.localdir import ObjectDir
from log3.vault.cryptobox import KIND_CHUNK, KIND_MANIFEST, OVERHEAD, CryptoBox, PassphraseKeys, new_kdf_params
from log3.vault.errors import VaultError
from log3.vault.replicate import BATCH_UNIT, _lease
from log3.vault.repo import Repo, new_config, validate_config
from log3.vault.states import TRANSITIONS, VersionState, check_transition

KDF = new_kdf_params(n=2 ** 10)        # швидкий KDF лише для тестів


@pytest.fixture(scope="module")
def box():
    return CryptoBox(PassphraseKeys(b"unit-test", KDF))


def test_node_quota_counts_pending_writes_and_only_replacement_growth(tmp_path):
    ledger = node_server.Ledger(tmp_path / "node-ledger.sqlite")
    a, b = "a" * 64, "b" * 64
    with ledger.reserve(a, 6, 10):
        with pytest.raises(node_server.QuotaExceeded), ledger.reserve(b, 6, 10):
            pytest.fail("pending bytes must count toward the quota")
        ledger.charge(a, 6, "first")
    with ledger.reserve(a, 8, 10):
        ledger.charge(a, 8, "repair")
    assert ledger.used() == 8

    with pytest.raises(RuntimeError, match="write failed"), ledger.reserve(b, 2, 10):
        raise RuntimeError("write failed")
    with ledger.reserve(b, 2, 10):
        ledger.charge(b, 2, "next")
    assert ledger.used() == 10


def test_node_ledger_reconciles_existing_objects_on_restart(tmp_path):
    store = ObjectDir(tmp_path)
    store.create()
    ledger = node_server.Ledger(tmp_path / "node-ledger.sqlite")
    old, added, removed = "a" * 64, "b" * 64, "c" * 64
    ledger.charge(old, 1, "first")
    ledger.charge(old, 1, "duplicate")
    ledger.charge(removed, 100, "removed")
    for oid, payload in ((old, b"changed"), (added, b"new")):
        path = store.path(oid)
        path.parent.mkdir()
        path.write_bytes(payload)

    ledger.reconcile_objects(store.root / "objects")
    assert ledger.used() == 10
    assert ledger.db.execute("SELECT puts FROM charges WHERE oid=?", (old,)).fetchone() == (2,)


@pytest.mark.parametrize("codec_id", sorted(codecs.REGISTRY))
@pytest.mark.parametrize("data", [b"", b"a", b"abc" * 5000, bytes(range(256)) * 40, os.urandom(3000)])
def test_codec_roundtrip_and_store_fallback(codec_id, data):
    used, payload = codecs.encode_best(data, codec_id)
    assert used in (codec_id, codecs.STORE)
    assert len(payload) <= len(data)                       # невигідне стиснення замінено Store
    assert codecs.decode(used, payload, len(data)) == data


def test_random_data_is_stored_and_bad_payload_rejected():
    data = os.urandom(4096)
    assert codecs.encode_best(data, codecs.ZSTD)[0] == codecs.STORE
    used, payload = codecs.encode_best(b"x" * 4096, codecs.ZSTD)
    assert used == codecs.ZSTD
    with pytest.raises(VaultError):
        codecs.decode(used, payload, 4095)                 # записана довжина не збігається
    with pytest.raises(VaultError):
        codecs.decode(99, payload, 4096)                   # невідомий кодек


def test_seal_open_and_every_tampering_is_rejected(box):
    data = b"secret payload" * 10
    oid = box.object_id(KIND_CHUNK, data)
    blob = box.seal(KIND_CHUNK, oid, codecs.STORE, len(data), data)
    assert len(blob) == len(data) + OVERHEAD and data not in blob
    assert box.open(blob, KIND_CHUNK, oid) == (codecs.STORE, len(data), data)
    for i in range(len(blob)):                             # будь-який змінений байт заголовка, тіла чи тегу
        bad = blob[:i] + bytes([blob[i] ^ 1]) + blob[i + 1:]
        with pytest.raises(VaultError):
            box.open(bad, KIND_CHUNK, oid)
    with pytest.raises(VaultError):
        box.open(blob, KIND_MANIFEST, oid)                 # інший тип об'єкта
    with pytest.raises(VaultError):
        box.open(blob, KIND_CHUNK, box.object_id(KIND_CHUNK, b"other"))   # підміна блока іншим
    with pytest.raises(VaultError):
        box.open(blob[:20], KIND_CHUNK, oid)


def test_wrong_passphrase_and_keyed_ids(box):
    other = CryptoBox(PassphraseKeys(b"another", KDF))
    data = b"same bytes"
    oid = box.object_id(KIND_CHUNK, data)
    assert other.object_id(KIND_CHUNK, data) != oid        # ідентифікатор не обчислити без ключа
    with pytest.raises(VaultError) as e:
        other.open(box.seal(KIND_CHUNK, oid, 0, len(data), data), KIND_CHUNK, oid)
    assert e.value.code == "E_KEY"
    assert box.seal(KIND_CHUNK, oid, 0, len(data), data) != box.seal(KIND_CHUNK, oid, 0, len(data), data)


@pytest.mark.parametrize("path", ["", "/etc/passwd", "../x", "a/../b", "a//b", "a/./b", "a\\b", "a\x00b"])
def test_unsafe_paths_rejected(path):
    with pytest.raises(VaultError):
        manifest.safe_parts(path, windows=False)


@pytest.mark.parametrize("path", ["C:/x", "a/CON", "a/nul.txt", "a/b.", "a/b ", "a/x?y"])
def test_windows_paths_rejected(path):
    with pytest.raises(VaultError):
        manifest.safe_parts(path, windows=True)
    assert manifest.safe_parts("тека/файл ☃.txt", windows=True) == ["тека", "файл ☃.txt"]


def _manifest(entries):
    files = [e for e in entries if e["type"] == "file"]
    return manifest.dump({"format": manifest.FORMAT, "format_version": 1, "version_id": "v", "entries": entries,
                          "contents": {e["content_id"]: {"size": 0, "sha256": "", "chunks": []} for e in files},
                          "objects": {}})


def _file(path):
    return {"path": path, "type": "file", "content_id": "c", "size": 0}


def test_manifest_validation():
    assert manifest.load(_manifest([_file("a/b"), {"path": "a", "type": "dir"}]))["version_id"] == "v"
    with pytest.raises(VaultError):
        manifest.load(_manifest([_file("Readme"), _file("README")]), insensitive=True)    # колізія імен
    manifest.load(_manifest([_file("Readme"), _file("README")]), insensitive=False)
    with pytest.raises(VaultError):
        manifest.load(_manifest([_file("a"), _file("a/b")]))                              # файл як каталог
    with pytest.raises(VaultError):
        manifest.load(_manifest([_file("../escape")]))
    with pytest.raises(VaultError):
        manifest.load(json.dumps({"format": manifest.FORMAT, "format_version": 2}).encode())
    with pytest.raises(VaultError):
        manifest.load(b"\xff not json")


def test_rfc3339_handles_real_dates():
    assert manifest.rfc3339(0) == "1970-01-01T00:00:00.000000000Z"
    assert manifest.rfc3339(4_102_444_800 * 10 ** 9 + 5) == "2100-01-01T00:00:00.000000005Z"


def test_state_model():
    V = VersionState
    check_transition(V.CAPTURING, V.ACCEPTED_LOCAL)
    check_transition(V.REPLICATING, V.PROTECTED)
    for bad in ((V.CAPTURING, V.PROTECTED), (V.FAILED, V.PROTECTED), (V.DELETED, V.PROTECTED),
                (V.ACCEPTED_LOCAL, V.CAPTURING), (V.CAPTURING, V.REPLICATING)):
        with pytest.raises(VaultError):
            check_transition(*bad)
    assert set(TRANSITIONS) == set(V) and not TRANSITIONS[V.DELETED]


def _ctx(suffix="txt", entropy=3.0):
    return CompressionContext("op", "ver", suffix, 100, 0, 100, b"head", entropy, "zstd", 3)


def test_baseline_advisor_and_failure_isolation():
    safe = SafeAdvisor()
    assert safe.recommend(_ctx())[0] == Recommendation("zstd", 3)
    assert safe.recommend(_ctx("zip"))[0].codec == "store"
    assert safe.recommend(_ctx(entropy=7.9))[0].codec == "store"
    assert sample_entropy(b"") == 0 and sample_entropy(bytes(range(256))) == 8.0

    class Broken:
        def recommend(self, context):
            raise RuntimeError("model not trained")

        def observe(self, outcome):
            raise RuntimeError("boom")

    class Unknown(Broken):
        def recommend(self, context):
            return Recommendation("lzma-9000", 3)

    for bad in (Broken(), Unknown()):
        safe = SafeAdvisor(bad)
        assert safe.recommend(_ctx()) == (Recommendation("zstd", 3), True)   # базова політика замість збою
        safe.observe(None)
        assert safe.errors == 2
    assert load_advisor("no.such.module:Thing").recommend(_ctx())[0].codec == "zstd"


def test_config_rejects_secrets_and_bad_policy():
    cfg = new_config()
    cfg["backends"] = [{"name": "n", "type": "node", "url": "https://h", "ca_file": "c", "token_env": "T",
                        "failure_domain": "d", "token": "leaked"}]
    with pytest.raises(VaultError):
        validate_config(cfg)
    cfg = new_config()
    cfg["policy"]["min_copies"] = 0
    with pytest.raises(VaultError):
        validate_config(cfg)
    assert "passphrase" not in json.dumps(new_config()).replace("passphrase_env", "")


def test_fair_queue_low_priority_is_not_starved(tmp_path):
    cfg = new_config()
    cfg["crypto"]["kdf"] = KDF
    repo = Repo.create(tmp_path / "repo", cfg, b"unit-test passphrase")
    with repo.cat.tx():
        for i in range(300):                               # 295 високого пріоритету і 5 низького, поставлених останніми
            prio = 0 if i >= 295 else 2
            repo.cat.x("INSERT INTO objects(oid, kind, stored_len, sha256, plain_len, codec, in_spool, created_ns) "
                       "VALUES(?,1,10,'s',10,0,1,0)", (f"{i:064x}",))
            repo.cat.x("INSERT INTO jobs(job_id, oid, backend, state, priority, next_at_ns, enq_ns) "
                       "VALUES(?,?,'b','PENDING',?,0,?)", (f"j{i}", f"{i:064x}", prio, i))
    batch = _lease(repo)
    assert len(batch) == 4 * BATCH_UNIT + 5                # пакет обмежений незалежно від довжини черги
    assert sum(r["priority"] == 0 for r in batch) == 5     # низький пріоритет потрапив у перший же пакет
    assert sum(r["priority"] == 2 for r in batch) == 4 * BATCH_UNIT


def test_file_changing_during_read_is_skipped_not_stored_torn(tmp_path, monkeypatch):
    import types

    from log3.vault import capture
    from log3.vault.repo import add_backend

    monkeypatch.setenv("LOG3_VAULT_PASSPHRASE", "unit-test passphrase")
    cfg = new_config(min_copies=1)
    cfg["crypto"]["kdf"] = KDF
    repo = Repo.create(tmp_path / "repo", cfg, b"unit-test passphrase")
    add_backend(repo, {"name": "d", "type": "localdir", "path": str(tmp_path / "d"), "failure_domain": "x"})
    src = tmp_path / "src"
    src.mkdir()
    (src / "stable.txt").write_bytes(b"stable")
    (src / "busy.db").write_bytes(b"page" * 1000)
    real, calls = os.fstat, [0]

    def moving_fstat(fd):                                  # busy.db «змінюється» між початком і кінцем читання
        st = real(fd)
        if st.st_size == 4000:
            calls[0] += 1
            return types.SimpleNamespace(st_size=st.st_size, st_mtime_ns=st.st_mtime_ns + calls[0],
                                         st_ino=st.st_ino, st_dev=st.st_dev, st_mode=st.st_mode)
        return st

    monkeypatch.setattr(capture.os, "fstat", moving_fstat)
    vid, _ = capture.backup(repo, src)
    v = repo.cat.find_version(vid)
    assert calls[0] == 2 * capture.READ_RETRIES            # три спроби прочитати узгоджено
    assert (v["state"], v["files"], v["skipped"]) == ("ACCEPTED_LOCAL", 1, 1)
