"""Стенд для тестів файлового режиму: кожна команда CLI — новий процес, вузол — окремий процес."""
from __future__ import annotations

import json
import os
import socket
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PASSPHRASE = "correct horse battery staple"
TOKEN = "node-token-for-tests"
FAST = {"max_attempts": 3, "backoff_base_seconds": 0.05, "backoff_cap_seconds": 0.2, "capture_stale_seconds": 0.0}


class Env:
    def __init__(self, tmp: Path):
        self.tmp, self.repo, self.node_dir, self.disk = tmp, tmp / "repo", tmp / "node", tmp / "disk1"
        self.env = {**os.environ, "PYTHONPATH": str(ROOT), "LOG3_VAULT_PASSPHRASE": PASSPHRASE,
                    "LOG3_NODE_TOKEN": TOKEN, "LOG3_VAULT_REPO": str(self.repo), "LOG3_NODE_QUIET": "1",
                    "PYTHONIOENCODING": "utf-8"}
        self.node: subprocess.Popen | None = None
        self.port = 0

    def argv(self, *args: str) -> list[str]:
        return [sys.executable, "-m", "log3.vault", args[0], "--json", *map(str, args[1:])]

    def cli(self, *args, env: dict | None = None) -> tuple[int, dict]:
        """(код завершення, JSON-вивід) команди в новому процесі."""
        p = subprocess.run(self.argv(*args), env={**self.env, **(env or {})}, capture_output=True, text=True,
                           encoding="utf-8", timeout=300, stdin=subprocess.DEVNULL)
        try:
            return p.returncode, json.loads(p.stdout)
        except ValueError:
            return p.returncode, {"stdout": p.stdout, "stderr": p.stderr}

    def ok(self, *args, env: dict | None = None) -> dict:
        code, out = self.cli(*args, env=env)
        assert code == 0, (args, code, out)
        return out

    def start_node(self, *, fault: str | None = None, quota: int = 0, tokens_file: Path | None = None) -> None:
        certs = self.tmp / "certs"
        if not certs.exists():
            self.ok("node-cert", "--dir", certs, "--host", "127.0.0.1")
        if not self.port:
            with socket.socket() as s:
                s.bind(("127.0.0.1", 0))
                self.port = s.getsockname()[1]
        env = {**self.env, **({"LOG3_VAULT_FAULT": fault} if fault else {})}
        self.node = subprocess.Popen(
            [sys.executable, "-m", "log3.vault", "serve-node", "--data", str(self.node_dir), "--cert",
             str(certs / "node.crt"), "--key", str(certs / "node.key"), "--host", "127.0.0.1", "--port",
             str(self.port), "--quota-bytes", str(quota),
             *(["--tokens-file", str(tokens_file)] if tokens_file else [])], env=env, stderr=subprocess.DEVNULL)
        for _ in range(200):
            try:
                socket.create_connection(("127.0.0.1", self.port), timeout=0.2).close()
                return
            except OSError:
                time.sleep(0.05)
        raise RuntimeError("вузол не запустився")

    def stop_node(self) -> None:
        if self.node is not None:
            self.node.kill()
            self.node.wait()
            self.node = None

    def setup(self, *, node: bool = True, start: bool = True, min_copies: int = 2, **limits) -> Env:
        """init + локальний backend (+ вузол), швидкі повтори для тестів."""
        self.ok("init", "--min-copies", min_copies, "--chunk-size-mib", "0.25")
        self.ok("backend", "add", "disk", "--type", "localdir", "--path", self.disk, "--domain", "disk-a")
        if node:
            if start:
                self.start_node()
            else:                                           # лише сертифікат і порт; вузол «лежить»
                self.start_node()
                self.stop_node()
            self.ok("backend", "add", "node", "--type", "node", "--url", f"https://127.0.0.1:{self.port}",
                    "--ca-file", self.tmp / "certs" / "node.crt", "--domain", "site-b")
        self.set_limits(**{**FAST, **limits})
        return self

    def set_limits(self, **limits) -> None:
        path = self.repo / "config.json"
        cfg = json.loads(path.read_text(encoding="utf-8"))
        cfg["limits"].update(limits)
        path.write_text(json.dumps(cfg), encoding="utf-8")

    def set_policy(self, **policy) -> None:
        path = self.repo / "config.json"
        cfg = json.loads(path.read_text(encoding="utf-8"))
        cfg["policy"].update(policy)
        path.write_text(json.dumps(cfg), encoding="utf-8")

    def sql(self, query: str, db: Path | None = None) -> list[tuple]:
        con = sqlite3.connect(db or self.repo / "catalog.sqlite")
        try:
            return con.execute(query).fetchall()
        finally:
            con.close()

    def ledger(self, query: str) -> list[tuple]:
        return self.sql(query, self.node_dir / "node-ledger.sqlite")


def make_tree(root: Path) -> Path:
    """Unicode-імена, порожній файл і каталог, однаковий вміст під різними іменами, двійкові дані."""
    src = root / "джерело"
    (src / "порожній каталог").mkdir(parents=True)
    (src / "вкладений" / "глибше").mkdir(parents=True)
    blob = os.urandom(700_000)
    (src / "звіт ☃ 2026.pdf").write_bytes(b"%PDF-1.7\n" + blob)
    (src / "вкладений" / "копія звіту.bin").write_bytes(b"%PDF-1.7\n" + blob)       # ті самі байти, інше ім'я
    (src / "вкладений" / "глибше" / "таблиця.xlsx").write_bytes(b"PK\x03\x04" + os.urandom(5000))
    (src / "порожній.txt").write_bytes(b"")
    (src / "нотатки.txt").write_text("конфіденційний рядок для пошуку відкритого тексту\n" * 20000, encoding="utf-8")
    (src / "program.exe").write_bytes(b"MZ" + bytes(range(256)) * 300)
    return src


def tree(root: Path) -> dict[str, bytes | None]:
    """Відносний шлях -> байти файла (None для каталогу)."""
    return {p.relative_to(root).as_posix(): (p.read_bytes() if p.is_file() else None)
            for p in sorted(root.rglob("*"))}


def all_bytes(*roots: Path) -> bytes:
    return b"".join(p.read_bytes() for r in roots if r.exists() for p in r.rglob("*") if p.is_file())
