"""Вимірювання LOG3 Vault у нових процесах CLI:  python scripts/vault_bench.py baseline

Сценарій baseline (відправна точка етапу 2): init -> backend-и (локальний каталог + HTTPS-вузол
окремим процесом) -> backup без --wait (лише локальне приймання; процес завершується, як після
перезапуску) -> перезапуск вузла -> run -> status -> verify -> restore -> побайтове порівняння.

Кожна команда — новий процес, тож час включає старт інтерпретатора. CPU і пікова RSS — дочірніх
процесів (resource.RUSAGE_CHILDREN). Дані — синтетичний корпус scripts/vault_corpus.py.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.vault_corpus import make_corpus, mutate  # noqa: E402

PASSPHRASE = "bench passphrase, not a secret"
TOKEN = "bench-node-token"


def _cpu() -> tuple[float, int]:
    r = resource.getrusage(resource.RUSAGE_CHILDREN)
    return r.ru_utime + r.ru_stime, r.ru_maxrss


class Bench:
    def __init__(self, tmp: Path, extra_env: dict | None = None):
        self.tmp, self.repo = tmp, tmp / "repo"
        self.env = {**os.environ, "PYTHONPATH": str(ROOT), "LOG3_VAULT_PASSPHRASE": PASSPHRASE,
                    "LOG3_NODE_TOKEN": TOKEN, "LOG3_VAULT_REPO": str(self.repo), "LOG3_NODE_QUIET": "1",
                    "PYTHONIOENCODING": "utf-8", **(extra_env or {})}
        self.node: subprocess.Popen | None = None
        self.port = 0
        self.steps: list[dict] = []

    def cli(self, step: str, *args, check: bool = True) -> dict:
        argv = [sys.executable, "-m", "log3.vault", args[0], "--json", *map(str, args[1:])]
        cpu0, _ = _cpu()
        t0 = time.perf_counter()
        p = subprocess.run(argv, env=self.env, capture_output=True, text=True, encoding="utf-8", timeout=3600)
        wall = time.perf_counter() - t0
        cpu1, rss = _cpu()
        try:
            out = json.loads(p.stdout)
        except ValueError:
            out = {"stdout": p.stdout[-2000:], "stderr": p.stderr[-2000:]}
        self.steps.append({"step": step, "exit_code": p.returncode, "wall_seconds": round(wall, 3),
                           "cpu_seconds": round(cpu1 - cpu0, 3), "children_max_rss_kib": rss})
        if check and p.returncode != 0:
            raise RuntimeError(f"{step}: код {p.returncode}: {out}")
        return out

    def start_node(self) -> None:
        certs = self.tmp / "certs"
        if not certs.exists():
            self.cli("node-cert", "node-cert", "--dir", certs, "--host", "127.0.0.1")
        if not self.port:
            with socket.socket() as s:
                s.bind(("127.0.0.1", 0))
                self.port = s.getsockname()[1]
        self.node = subprocess.Popen(
            [sys.executable, "-m", "log3.vault", "serve-node", "--data", str(self.tmp / "node"), "--cert",
             str(certs / "node.crt"), "--key", str(certs / "node.key"), "--host", "127.0.0.1", "--port",
             str(self.port)], env=self.env, stderr=subprocess.DEVNULL)
        for _ in range(400):
            try:
                socket.create_connection(("127.0.0.1", self.port), timeout=0.2).close()
                return
            except OSError:
                time.sleep(0.05)
        raise RuntimeError("вузол не запустився")

    def stop_node(self) -> None:
        if self.node is not None:
            self.node.terminate()
            self.node.wait()
            self.node = None

    def setup(self, *, chunk_mib: float = 1.0) -> None:
        self.cli("init", "init", "--min-copies", 2, "--chunk-size-mib", chunk_mib)
        self.cli("backend-disk", "backend", "add", "disk", "--type", "localdir", "--path", self.tmp / "disk",
                 "--domain", "disk-a")
        self.start_node()
        self.cli("backend-node", "backend", "add", "node", "--type", "node", "--url",
                 f"https://127.0.0.1:{self.port}", "--ca-file", self.tmp / "certs" / "node.crt", "--domain", "site-b")

    def configure(self, **compression) -> None:
        path = self.repo / "config.json"
        cfg = json.loads(path.read_text(encoding="utf-8"))
        cfg["compression"].update(compression)
        path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    def stored(self, version_id: str) -> dict:
        con = sqlite3.connect(self.repo / "catalog.sqlite")
        try:
            row = con.execute("SELECT COUNT(*), COALESCE(SUM(o.stored_len),0), COALESCE(SUM(o.plain_len),0), "
                              "COALESCE(SUM(o.codec=0),0) FROM version_objects vo JOIN objects o ON o.oid=vo.oid "
                              "WHERE vo.version_id=? AND o.kind=1", (version_id,)).fetchone()
        finally:
            con.close()
        return {"chunks": row[0], "stored_bytes": row[1], "plain_bytes": row[2], "store_chunks": row[3]}


def _same_tree(a: Path, b: Path) -> bool:
    fa = {p.relative_to(a).as_posix(): p for p in a.rglob("*") if p.is_file()}
    fb = {p.relative_to(b).as_posix(): p for p in b.rglob("*") if p.is_file()}
    return fa.keys() == fb.keys() and all(fa[k].read_bytes() == fb[k].read_bytes() for k in fa)


def baseline(scale_mib: float, seed: int, tmp: Path) -> dict:
    """backup -> restart -> verify -> restore на коді етапу 1 (advisor не налаштовано)."""
    b = Bench(tmp)
    src = tmp / "src"
    corpus = make_corpus(src, seed, scale_mib)
    b.setup()
    try:
        acc = b.cli("backup (accept local)", "backup", src)["version"]
        vid = acc["version_id"]
        b.stop_node()                                       # перезапуск вузла між прийманням і реплікацією
        b.start_node()
        b.cli("run (after restart)", "run")
        st = b.cli("status", "status", vid)["version"]
        ver = b.cli("verify", "verify", vid)["verify"]
        b.cli("restore", "restore", vid, tmp / "out")
        metrics = b.cli("status repo", "status")["metrics"]
        same = _same_tree(src, tmp / "out" / src.name) if (tmp / "out" / src.name).exists() \
            else _same_tree(src, tmp / "out")
        changed = mutate(src, seed + 1, 0.15)
        inc = b.cli("backup incremental --wait", "backup", src, "--wait")["version"]
    finally:
        b.stop_node()
    s = b.stored(vid)
    return {
        "scenario": "baseline: backup -> restart -> verify -> restore (етап 1, без навчання)",
        "corpus": corpus, "state_after_accept": acc["state"], "state_after_run": st["state"],
        "policy_satisfied": st["policy_satisfied"], "verify": ver, "restore_identical": same,
        "storage": {**s, "ratio_stored_to_plain": round(s["stored_bytes"] / max(1, s["plain_bytes"]), 4)},
        "time_to_protected_seconds": metrics["last_time_to_protected_seconds"],
        "restore_seconds": metrics["last_restore_duration_seconds"],
        "incremental": {"changed_files": changed, "state": inc["state"], **b.stored(inc["version_id"])},
        "steps": b.steps,
    }


def environment() -> dict:
    return {"python": platform.python_version(), "platform": platform.platform(), "cpus": os.cpu_count()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("scenario", choices=("baseline",))
    ap.add_argument("--scale-mib", type=float, default=32.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=Path, help="куди записати JSON (типово stdout)")
    a = ap.parse_args(argv)
    with tempfile.TemporaryDirectory(prefix="log3-bench-") as d:
        result = {"environment": environment(), **baseline(a.scale_mib, a.seed, Path(d))}
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
