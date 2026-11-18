"""Операції з ключами сховища (команда key): стан, ротація, зміна фрази, експорт, публікація підписів.

Усі зміни keyring — атомарний запис config.json (tmp -> fsync -> rename) і подія в каталозі без
секретів. Нова конфігурація й підписи службових файлів одразу розсилаються на доступні backend-и:
доки цього не сталося, recover із backend-а бачить попередній keyring (старі фрази там ще діють).
"""
from __future__ import annotations

import os
from pathlib import Path

from .backends.base import Backend
from .cryptobox import read_secret
from .errors import BackendError, Exit, VaultError
from .fsutil import write_durable
from .keys import MAX_OBJECTS_PER_KEY, change_passphrase, export_bundle, rotate, sign_meta
from .replicate import CONFIG_META, pointer_doc, pointer_name
from .repo import Repo

NEW_PASSPHRASE_ENV = "LOG3_VAULT_NEW_PASSPHRASE"
EXPORT_PASSPHRASE_ENV = "LOG3_VAULT_EXPORT_PASSPHRASE"


def _passphrase(repo: Repo) -> bytes:
    return read_secret(repo.cfg["crypto"]["passphrase_env"], prompt="Парольна фраза сховища: ")


def _reload(repo: Repo) -> None:
    for attr in ("keys", "box"):
        repo.__dict__.pop(attr, None)


def status(repo: Repo) -> dict:
    """Без секрету: ключі, кількість об'єктів на кожному і запас до межі nonce."""
    crypto, cat = repo.cfg["crypto"], repo.cat
    ring = crypto.get("keyring")
    ids = [k["key_id"] for k in ring["keys"]] if ring else [crypto["key_id"]]
    limit = crypto.get("max_objects_per_key", MAX_OBJECTS_PER_KEY)
    per_key = dict(cat.q("SELECT COALESCE(key_id, 'unknown'), COUNT(*) FROM objects GROUP BY key_id"))
    return {"keyring": bool(ring), "active_key_id": crypto["key_id"], "max_objects_per_key": limit,
            "keys": [{"key_id": k, "active": k == crypto["key_id"], "objects_now": per_key.get(k, 0),
                      "sealed_total": cat.sealed_count(k), "nonce_budget_used": round(cat.sealed_count(k) / limit, 6),
                      "created_at": next((x.get("created_at") for x in (ring or {}).get("keys", [])
                                          if x["key_id"] == k), None)} for k in ids],
            "objects_unknown_key": per_key.get("unknown", 0),
            "kdf": {k: crypto["kdf"][k] for k in ("name", "n", "r", "p")}}


def rotate_key(repo: Repo, *, reason: str = "вручну") -> dict:
    old = repo.cfg["crypto"]["key_id"]
    new = rotate(repo.cfg["crypto"], _passphrase(repo))
    repo.save_config()
    _reload(repo)
    repo.cat.event("key_rotated", None, f"{old}->{new}: {reason}")
    return {"previous_key_id": old, "active_key_id": new, "publish": publish(repo)}


def change_key_passphrase(repo: Repo) -> dict:
    new = read_secret(NEW_PASSPHRASE_ENV, prompt="Нова парольна фраза: ")
    change_passphrase(repo.cfg["crypto"], _passphrase(repo), new)
    repo.save_config()
    os.environ[repo.cfg["crypto"]["passphrase_env"]] = new.decode("utf-8")   # лише цей процес
    _reload(repo)
    repo.cat.event("passphrase_changed", None, repo.cfg["crypto"]["key_id"])
    return {"active_key_id": repo.cfg["crypto"]["key_id"], "publish": publish(repo)}


def export_keys(repo: Repo, out: Path) -> dict:
    if out.exists():
        raise VaultError("E_EXISTS", f"{out} вже існує")
    data = export_bundle(repo.cfg["crypto"], _passphrase(repo), repo.cfg["repo_id"],
                         read_secret(EXPORT_PASSPHRASE_ENV, prompt="Фраза для файла експорту: "))
    out.parent.mkdir(parents=True, exist_ok=True)
    write_durable(out, [data])
    os.chmod(out, 0o600)
    repo.cat.event("keys_exported", None, out.name)
    return {"export": str(out), "keys": len(repo.keys.keys), "repo_id": repo.cfg["repo_id"]}


def publish(repo: Repo) -> dict:
    """Підписує конфігурацію й покажчики живих версій і розсилає їх на всі доступні backend-и."""
    meta_key, cat = repo.keys.meta_key(), repo.cat
    config = repo.sign_config()
    signed = 0
    rows = cat.q("SELECT v.version_id, v.manifest_oid, v.created_ns, v.pointer, o.stored_len, o.sha256 "
                 "FROM versions v JOIN objects o ON o.oid=v.manifest_oid "
                 "WHERE v.state IN ('ACCEPTED_LOCAL','REPLICATING','PROTECTED','DEGRADED')")
    pointers = []
    with cat.tx():
        for v in rows:
            doc = sign_meta(meta_key, pointer_name(v["version_id"]), pointer_doc(
                v["version_id"], v["manifest_oid"], v["stored_len"], v["sha256"], v["created_ns"]))
            if v["pointer"] != doc.decode("utf-8"):
                cat.x("UPDATE versions SET pointer=? WHERE version_id=?", (doc.decode("utf-8"), v["version_id"]))
                signed += 1
            pointers.append((v["version_id"], v["manifest_oid"], doc))
    out = {"pointers_signed": signed, "backends": {}}
    for name in repo.backend_names():
        b: Backend = repo.backend(name)
        try:
            b.put_meta(CONFIG_META, config)
            n = 0
            for vid, moid, doc in pointers:                    # покажчик — лише туди, де вже лежить маніфест
                if cat.one("SELECT 1 FROM replicas WHERE oid=? AND backend=? AND state='CONFIRMED'", (moid, name)):
                    b.put_meta(pointer_name(vid), doc)
                    n += 1
            out["backends"][name] = {"ok": True, "pointers": n}
        except (BackendError, OSError) as e:
            out["backends"][name] = {"ok": False, "error": str(e)[:200]}
    return out


def publish_exit(out: dict) -> Exit:
    return Exit.OK if all(b["ok"] for b in out["backends"].values()) else Exit.ATTENTION
