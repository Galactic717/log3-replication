"""Сховище (repo): конфігурація без секретів, каталог, spool і backend-и.

    <repo>/config.json       формат, параметри KDF, політика, межі, список backend-ів; секретів нема
    <repo>/catalog.sqlite    каталог версій і постійна черга
    <repo>/spool/objects/    зашифровані об'єкти, які ще не виконали політику копій
    <repo>/spool/tmp/        незавершені записи (прибирає repair)

Парольна фраза потрібна лише для backup, restore і recover. Черга (run), verify, repair, gc
працюють із уже зашифрованими об'єктами і секрету не потребують.
"""
from __future__ import annotations

import json
import os
import uuid
from functools import cached_property
from pathlib import Path

from .backends.base import Backend
from .backends.localdir import LocalDirBackend, ObjectDir
from .backends.node_client import NodeBackend
from .catalog import Catalog, now_ns
from .cryptobox import PASSPHRASE_ENV, SUITE, CryptoBox, PassphraseKeys, new_kdf_params, read_secret
from .errors import Exit, VaultError
from .fsutil import write_durable

CONFIG_FORMAT = "log3-vault-config"
CONFIG_VERSION = 1
PRIORITIES = {"low": 0, "normal": 1, "high": 2}
MIB = 1 << 20
DEFAULT_LIMITS = {
    "queue_max_jobs": 200_000,          # завдань у черзі (PENDING + RUNNING)
    "spool_max_bytes": 20 * 1024 * MIB,  # байтів зашифрованих об'єктів у spool
    "min_free_bytes": 256 * MIB,        # вільне місце на диску spool, нижче якого приймання відмовляє
    "max_parallel": 4,                  # одночасних передавань
    "max_attempts": 8,                  # бюджет автоматичних повторів одного завдання
    "backoff_base_seconds": 2.0,
    "backoff_cap_seconds": 600.0,
    "lease_seconds": 900.0,             # після цього завдання RUNNING вважається завислим
    "capture_stale_seconds": 300.0,     # після цього захоплення без heartbeat вважається аварійним
}
DEFAULT_POLICY = {"min_copies": 2, "extra_copies_high_priority": 1, "retention_days": 30, "keep_last": 3}


def new_config(*, min_copies: int = 2, chunk_size: int = MIB, kdf_n: int = 2 ** 15) -> dict:
    return {"format": CONFIG_FORMAT, "format_version": CONFIG_VERSION, "repo_id": str(uuid.uuid4()),
            "created_at_ns": now_ns(), "chunk_size_bytes": chunk_size,
            "crypto": {"suite": SUITE, "kdf": new_kdf_params(kdf_n), "key_id": None,
                       "passphrase_env": PASSPHRASE_ENV},
            "compression": {"codec": "zstd", "level": 3, "advisor": None},
            "learning": {"enabled": True},          # решта — типові значення log3.vault.learn.settings
            "policy": {**DEFAULT_POLICY, "min_copies": min_copies},
            "limits": dict(DEFAULT_LIMITS), "backends": []}


def validate_config(cfg: dict) -> None:
    def bad(msg: str) -> VaultError:
        return VaultError("E_CONFIG", msg, Exit.POLICY)

    if cfg.get("format") != CONFIG_FORMAT or cfg.get("format_version") != CONFIG_VERSION:
        raise bad("config.json: невідомий формат або версія")
    if not 4096 <= cfg["chunk_size_bytes"] <= 64 * MIB:
        raise bad("chunk_size_bytes має бути в межах 4 КіБ .. 64 МіБ")
    if cfg["policy"]["min_copies"] < 1 or cfg["policy"]["keep_last"] < 0 or cfg["policy"]["retention_days"] < 0:
        raise bad("policy: min_copies >= 1, keep_last >= 0, retention_days >= 0")
    if cfg["limits"]["max_parallel"] < 1 or cfg["limits"]["max_attempts"] < 1:
        raise bad("limits: max_parallel >= 1, max_attempts >= 1")
    names = [b["name"] for b in cfg["backends"]]
    if len(names) != len(set(names)):
        raise bad("назви backend-ів мають бути унікальними")
    for b in cfg["backends"]:
        need = {"localdir": ("path",), "node": ("url", "ca_file", "token_env")}.get(b.get("type"))
        if need is None or any(not b.get(k) for k in (*need, "failure_domain")):
            raise bad(f"backend {b.get('name')!r}: тип localdir (path) або node (url, ca_file, token_env) "
                      "і failure_domain обов'язкові")
        if any(k in b for k in ("token", "password", "secret", "passphrase")):
            raise bad(f"backend {b['name']!r}: секрети в конфігурації заборонені, вкажіть token_env")


class Repo:
    def __init__(self, path: Path):
        self.path = Path(path).absolute()
        cfg_path = self.path / "config.json"
        if not cfg_path.is_file():
            raise VaultError("E_NOT_FOUND", f"{self.path} не є сховищем LOG3 Vault (виконайте init)", Exit.NOT_FOUND)
        self.cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        validate_config(self.cfg)
        self.spool = self.path / "spool"
        self.cat = Catalog(self.path / "catalog.sqlite")
        self._backends: dict[str, Backend] = {}

    @classmethod
    def create(cls, path: Path, cfg: dict, passphrase: bytes) -> Repo:
        path = Path(path).absolute()
        if (path / "config.json").exists():
            raise VaultError("E_EXISTS", f"у {path} вже є сховище")
        cfg["crypto"]["key_id"] = PassphraseKeys(passphrase, cfg["crypto"]["kdf"]).key_id
        validate_config(cfg)
        for sub in ("spool/objects", "spool/tmp"):
            (path / sub).mkdir(parents=True, exist_ok=True)
        write_durable(path / "config.json", [json.dumps(cfg, ensure_ascii=False, indent=2).encode("utf-8")])
        return cls(path)

    def save_config(self) -> None:
        validate_config(self.cfg)
        write_durable(self.path / "config.json", [self.config_bytes()])

    def config_bytes(self) -> bytes:
        return json.dumps(self.cfg, ensure_ascii=False, indent=2).encode("utf-8")

    @cached_property
    def box(self) -> CryptoBox:
        crypto = self.cfg["crypto"]
        keys = PassphraseKeys(read_secret(crypto["passphrase_env"], prompt="Парольна фраза сховища: "), crypto["kdf"])
        if keys.key_id != crypto["key_id"]:
            raise VaultError("E_KEY", "парольна фраза не відповідає цьому сховищу", Exit.KEY)
        return CryptoBox(keys)

    @property
    def limits(self) -> dict:
        return {**DEFAULT_LIMITS, **self.cfg["limits"]}

    @property
    def policy(self) -> dict:
        return {**DEFAULT_POLICY, **self.cfg["policy"]}

    def spool_path(self, oid: str) -> Path:
        return self.spool / "objects" / oid[:2] / oid

    def backend_names(self) -> list[str]:
        return [b["name"] for b in self.cfg["backends"]]

    def backend(self, name: str) -> Backend:
        if name not in self._backends:
            spec = next((b for b in self.cfg["backends"] if b["name"] == name), None)
            if spec is None:
                raise VaultError("E_NOT_FOUND", f"backend {name!r} не налаштовано", Exit.NOT_FOUND)
            self._backends[name] = make_backend(spec)
        return self._backends[name]

    def domains(self) -> dict[str, str]:
        """backend -> фактичний домен відмов. Локальні каталоги на одному пристрої зливаються в один
        домен, хоч би як їх назвали: два каталоги на одному диску не є незалежними копіями."""
        out, by_dev = {}, {}
        for b in self.cfg["backends"]:
            dom = b["failure_domain"]
            if b["type"] == "localdir":
                try:
                    dom = by_dev.setdefault(os.stat(b["path"]).st_dev, dom)
                except OSError:
                    pass
            out[b["name"]] = dom
        return out

    def targets(self, n_domains: int) -> list[str]:
        """Перші backend-и, що покривають n_domains різних доменів відмов (у порядку конфігурації)."""
        chosen, seen = [], set()
        for name, dom in self.domains().items():
            if dom not in seen and len(seen) < n_domains:
                seen.add(dom)
                chosen.append(name)
        return chosen


def make_backend(spec: dict) -> Backend:
    if spec["type"] == "localdir":
        return LocalDirBackend(spec["name"], spec["path"])
    return NodeBackend(spec["name"], spec["url"], spec["ca_file"], spec["token_env"])


def add_backend(repo: Repo, spec: dict) -> None:
    if spec["type"] == "localdir":
        spec["path"] = str(Path(spec["path"]).absolute())
        ObjectDir(Path(spec["path"])).create()
    else:
        spec["ca_file"] = str(Path(spec["ca_file"]).absolute())
    repo.cfg["backends"] = [b for b in repo.cfg["backends"] if b["name"] != spec["name"]] + [spec]
    repo.save_config()
