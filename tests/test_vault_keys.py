"""Ключі сховища на рівні бібліотеки: покоління 0, keyring, ротація, зміна фрази, експорт, підпис."""
import pytest

from log3.vault.cryptobox import KIND_CHUNK, CryptoBox
from log3.vault.errors import VaultError
from log3.vault.keys import (
    RepoKeys,
    change_passphrase,
    export_bundle,
    load_export,
    master_key,
    new_kdf_params,
    rotate,
    sign_meta,
    verify_meta,
)

OLD, NEW = b"old passphrase 1234", b"new passphrase 5678"


def _crypto():
    c = {"kdf": new_kdf_params(n=2 ** 10), "key_id": None}
    c["key_id"] = RepoKeys(OLD, c).key_id                     # покоління 0, як у сховищ етапів 1–2
    return c


def _seal(keys, data=b"payload"):
    box = CryptoBox(keys)
    oid = box.object_id(KIND_CHUNK, data)
    return box, oid, box.seal(KIND_CHUNK, oid, 0, len(data), data)


def test_rotation_keeps_old_objects_and_id_key():
    crypto = _crypto()
    k0 = RepoKeys(OLD, crypto)
    _, oid, blob = _seal(k0)
    new_id = rotate(crypto, OLD)
    k1 = RepoKeys(OLD, crypto)
    assert k1.key_id == new_id != k0.key_id and k1.id_key() == k0.id_key()   # дедуплікація не ламається
    assert CryptoBox(k1).open(blob, KIND_CHUNK, oid)[2] == b"payload"        # старий об'єкт читається
    _, oid2, blob2 = _seal(k1, b"new")
    assert blob2[7:15].hex() == new_id                                       # новий — активним ключем
    change_passphrase(crypto, OLD, NEW)
    with pytest.raises(VaultError):
        RepoKeys(OLD, crypto)
    k2 = RepoKeys(NEW, crypto)
    assert CryptoBox(k2).open(blob, KIND_CHUNK, oid)[2] == b"payload"
    assert CryptoBox(k2).open(blob2, KIND_CHUNK, oid2)[2] == b"new"


def test_export_and_meta_signature():
    crypto = _crypto()
    rotate(crypto, OLD)
    data = export_bundle(crypto, OLD, "repo-1", b"export phrase 99")
    repo_id, keys = load_export(data, b"export phrase 99")
    assert repo_id == "repo-1" and set(keys.keys) == set(RepoKeys(OLD, crypto).keys)
    with pytest.raises(VaultError):
        load_export(data, b"wrong export phrase")
    mk = keys.meta_key()
    import json
    signed = json.loads(sign_meta(mk, "version-a.json", {"version_id": "a", "n": 1}))
    assert verify_meta(mk, "version-a.json", signed) == {"version_id": "a", "n": 1}
    for name, doc in (("version-b.json", signed), ("version-a.json", {**signed, "n": 2})):
        with pytest.raises(VaultError):                                      # переставлений або змінений
            verify_meta(mk, name, doc)


def test_kdf_limits_from_untrusted_config():
    for bad in ({"n": 2 ** 24}, {"n": 1000}, {"r": 64}, {"salt_b64": "AAAA"}):
        with pytest.raises(VaultError):
            master_key(OLD, {**new_kdf_params(n=2 ** 10), **bad})
