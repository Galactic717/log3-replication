"""Лабораторія для демонстрацій і матриці відмов: сховище, backend-каталог і вузол в ізольованому каталозі.

Усі процеси (CLI, вузол) працюють на одному комп'ютері й одному диску. Відмови тут — це аварії
процесів і недоступність мережевого вузла; незалежних фізичних дисків і втрати живлення лабораторія
не моделює.
"""
from __future__ import annotations

import hashlib
import json
import random
import socket
import sqlite3
import time
from pathlib import Path

from .runner import Ctx, Result, Step, kill_tree

PASSPHRASE = "demo passphrase for defense only"
TOKEN = "demo-node-token-for-defense"
FAST = {"max_attempts": 3, "backoff_base_seconds": 0.05, "backoff_cap_seconds": 0.2, "lease_seconds": 1.0,
        "capture_stale_seconds": 0.0}
SAME_DISK = ("Процеси CLI, backend-каталог і вузол працюють на одному диску цього комп'ютера: це демонстрація "
             "процесних і мережевих відмов, а не незалежних фізичних дисків.")


def base_env(ctx_root: Path) -> dict:
    import os
    return {**os.environ, "PYTHONPATH": str(ctx_root), "LOG3_VAULT_PASSPHRASE": PASSPHRASE,
            "LOG3_NODE_TOKEN": TOKEN, "LOG3_NODE_QUIET": "1", "PYTHONIOENCODING": "utf-8"}


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def tree_digest(root: Path) -> dict:
    """Відносний шлях -> SHA-256 (None для каталогу)."""
    return {p.relative_to(root).as_posix(): (hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None)
            for p in sorted(root.rglob("*"))}


def make_mixed(root: Path, seed: int = 7, big_mib: int = 6) -> dict:
    """Змішаний каталог довільних файлів: документи, архіви, виконувані, Unicode, порожні, великий файл."""
    rnd = random.Random(seed)
    files = {
        "документи/звіт ☃ 2026.pdf": b"%PDF-1.7\n" + rnd.randbytes(300_000),
        "документи/договір.docx": b"PK\x03\x04" + rnd.randbytes(90_000),
        "документи/копія договору.docx": None,                             # той самий вміст під іншим іменем
        "таблиці/продажі.csv": "\n".join(f"{i},товар {i % 37},{rnd.randint(1, 999)}" for i in range(40_000)).encode(),
        "архіви/backup.zip": b"PK\x05\x06" + rnd.randbytes(150_000),
        "bin/tool.exe": b"MZ" + bytes(range(256)) * 400,
        "фото/IMG_0001.jpg": b"\xff\xd8\xff\xe0" + rnd.randbytes(250_000),
        "порожній.txt": b"",
        "великий/відео.bin": rnd.randbytes(big_mib << 19) + bytes(big_mib << 19),   # половина стисненна
        "日本語/メモ.txt": "Unicode-ім'я та вміст ✓\n".encode() * 500,
    }
    files["документи/копія договору.docx"] = files["документи/договір.docx"]
    for rel, data in files.items():
        p = root.joinpath(*rel.split("/"))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    (root / "порожній каталог").mkdir(parents=True, exist_ok=True)
    return {"files": len(files), "bytes": sum(len(d) for d in files.values())}


class Lab:
    def __init__(self, ctx: Ctx, name: str):
        self.ctx, self.dir, i = ctx, ctx.work / name, 2
        while self.dir.exists() or any(self.dir == other for other in ctx.vars.get("_labs", [])):
            self.dir, i = ctx.work / f"{name}-{i}", i + 1
        ctx.vars.setdefault("_labs", []).append(self.dir)
        self.repo, self.disk, self.node_dir = self.dir / "repo", self.dir / "disk-a", self.dir / "node-b"
        self.certs, self.src = self.dir / "certs", self.dir / "src"
        self.port = free_port()
        self.env = {"LOG3_VAULT_REPO": str(self.repo)}
        self.proc_name = f"node:{name}"
        for s in (PASSPHRASE, TOKEN):
            if s not in ctx.secrets:
                ctx.secrets.append(s)

    def v(self, *args) -> list:
        """argv команди log3.vault з --json."""
        return [self.ctx.py, "-m", "log3.vault", str(args[0]), "--json", *args[1:]]

    def step(self, goal: str, *args, expect=(0,), check=None, env: dict | None = None) -> Step:
        return Step(goal, self.v(*args), check=check, expect=expect, env={**self.env, **(env or {})})

    # --- вузол -----------------------------------------------------------------
    def start_node(self, *, fault: str | None = None, quota: int = 0, tokens: Path | None = None) -> Step:
        argv = [self.ctx.py, "-m", "log3.vault", "serve-node", "--data", str(self.node_dir), "--cert",
                str(self.certs / "node.crt"), "--key", str(self.certs / "node.key"), "--host", "127.0.0.1",
                "--port", str(self.port), "--quota-bytes", str(quota),
                *(["--tokens-file", str(tokens)] if tokens else [])]

        def ready(ctx: Ctx, res: Result) -> str:
            for _ in range(200):
                try:
                    socket.create_connection(("127.0.0.1", self.port), timeout=0.2).close()
                    return f"вузол слухає 127.0.0.1:{self.port}"
                except OSError:
                    time.sleep(0.05)
            raise AssertionError("вузол не запустився")
        return Step("запустити вузол зберігання (окремий процес HTTPS)" + (f", збій: {fault}" if fault else ""),
                    argv, background=self.proc_name, check=ready,
                    env={"LOG3_VAULT_FAULT": fault} if fault else None)

    def stop_node(self, goal: str = "зупинити процес вузла (вузол недоступний)") -> Step:
        def kill(ctx: Ctx) -> str:
            proc = ctx.procs.pop(self.proc_name, None)
            if proc is None:
                return "вузол уже зупинено"
            kill_tree(proc)
            return f"процес вузла PID {proc.pid} завершено"
        return Step(goal, action=kill)

    # --- підготовка ------------------------------------------------------------
    def setup(self, *, node: bool = True, node_up: bool = True, min_copies: int = 2, limits: dict | None = None,
              source: str = "mixed") -> list[Step]:
        def prepare(ctx: Ctx) -> str:
            info = make_mixed(self.src)
            return f"джерело {self.src.name}: {info['files']} файлів, {info['bytes']} B"
        steps = [Step("підготувати джерело зі змішаними файлами", action=prepare)] if source == "mixed" else []
        steps += [self.step("створити сховище (ключ із парольної фрази в змінній середовища)", "init",
                           "--min-copies", min_copies, "--chunk-size-mib", "0.25",
                           check=lambda c, r: f"repo_id {r.data['repo_id'][:8]}…, доступ: {r.data['access']}"),
                 self.step("додати backend-каталог (домен відмов disk-a)", "backend", "add", "disk", "--type",
                           "localdir", "--path", self.disk, "--domain", "disk-a")]
        if node:
            steps += [Step("створити сертифікат вузла", self.v("node-cert", "--dir", self.certs, "--host",
                                                                 "127.0.0.1")),
                      self.start_node(),
                      self.step("додати вузол (домен відмов site-b, закріплений сертифікат)", "backend", "add",
                                "node", "--type", "node", "--url", f"https://127.0.0.1:{self.port}", "--ca-file",
                                self.certs / "node.crt", "--domain", "site-b",
                                check=lambda c, r: f"незалежних доменів: {r.data['independent_failure_domains']}")]
            if not node_up:
                steps.append(self.stop_node())
        steps.append(Step("швидкі повтори для демонстрації (limits у config.json)",
                          action=lambda ctx: self.set_limits(**{**FAST, **(limits or {})})))
        return steps

    def set_limits(self, **limits) -> str:
        path = self.repo / "config.json"
        cfg = json.loads(path.read_text(encoding="utf-8"))
        cfg["limits"].update(limits)
        path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        return "limits: " + ", ".join(f"{k}={v}" for k, v in limits.items())

    def set_policy(self, **policy) -> str:
        path = self.repo / "config.json"
        cfg = json.loads(path.read_text(encoding="utf-8"))
        cfg["policy"].update(policy)
        path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        return "policy: " + ", ".join(f"{k}={v}" for k, v in policy.items())

    # --- перевірки з диска -------------------------------------------------------
    def sql(self, query: str, args: tuple = (), db: Path | None = None) -> list:
        con = sqlite3.connect(f"file:{db or self.repo / 'catalog.sqlite'}?mode=ro", uri=True)
        try:
            return con.execute(query, args).fetchall()
        finally:
            con.close()

    def ledger(self, query: str) -> list:
        return self.sql(query, db=self.node_dir / "node-ledger.sqlite")

    def same_tree(self, dest: Path, src: Path | None = None) -> str:
        a, b = tree_digest(src or self.src), tree_digest(dest)
        if a != b:
            diff = sorted(set(a.items()) ^ set(b.items()))[:5]
            raise AssertionError(f"відновлене відрізняється від джерела: {diff}")
        files = sum(v is not None for v in a.values())
        size = sum(p.stat().st_size for p in dest.rglob("*") if p.is_file())
        return f"побайтово збігається: {files} файлів, {size} B, SHA-256 кожного файла однаковий"

    def objects_on(self, root: Path) -> list[Path]:
        return sorted(p for p in (root / "objects").glob("*/*") if p.is_file())


def version_state(res: Result) -> dict:
    v = (res.data or {}).get("version")
    if not v:
        raise AssertionError(f"нема даних версії у виводі: {res.data}")
    return v


def expect_state(lab: Lab, state: str, var: str | None = None, protected: bool | None = None):
    """Перевірка: стан версії з JSON і чи виконано політику; опційно зберегти version_id у ctx.vars."""
    def check(ctx: Ctx, res: Result) -> str:
        v = version_state(res)
        if var:
            ctx.vars[var] = v["version_id"]
        assert v["state"] == state, f"стан {v['state']}, очікувався {state}"
        want = state == "PROTECTED" if protected is None else protected
        assert v["policy_satisfied"] is want, f"policy_satisfied={v['policy_satisfied']}"
        c = v.get("copies") or {}
        return (f"версія {v['version_id'][:8]}… стан {v['state']}, політику копій виконано: "
                f"{'так' if v['policy_satisfied'] else 'ні'}; об'єктів без копій: "
                f"{c.get('objects_under_replicated', '—')} з {c.get('objects_total', '—')}")
    return check
