"""Продуктові вимірювання LOG3 Vault і перевірка числових цілей для конкретного комп'ютера.

    python scripts/product_bench.py [--scale-mib 96] [--out results/vault] [--targets results/vault/targets.json]

Кожна команда — новий процес CLI (LOG3_VAULT_RUSAGE=1: процес сам звітує CPU і піковий RSS).
Backend-каталог і вузол лежать на тому самому диску, що й сховище: throughput — це ціна тракту
(читання, стиснення, шифрування, fsync, HTTPS) на одному диску, а не мережа чи незалежні диски.
Цілі задано для обладнання з targets.json (ключ — відбиток машини); на іншому комп'ютері цілі не
перевіряються, а лише записуються виміри. Навчальні S/R/T/Q тут не використовуються.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from log3.defense.lab import PASSPHRASE, TOKEN, free_port, make_mixed, tree_digest  # noqa: E402
from scripts.vault_corpus import make_corpus  # noqa: E402

MIB = 1 << 20


def machine() -> dict:
    cpu = platform.processor()
    if os.name == "nt":
        try:
            cpu = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                                 capture_output=True, text=True, timeout=60).stdout.strip() or cpu
        except OSError:
            pass
    elif Path("/proc/cpuinfo").exists():
        cpu = next((x.split(":", 1)[1].strip() for x in Path("/proc/cpuinfo").read_text().splitlines()
                    if x.startswith("model name")), cpu)
    return {"cpu": cpu, "logical_cpus": os.cpu_count(), "os": platform.platform(), "python": platform.python_version()}


def fingerprint(m: dict) -> str:
    return f"{m['cpu']} | {m['logical_cpus']} CPU | {platform.system()}"


class Cli:
    def __init__(self, tmp: Path, repo: Path):
        self.env = {**os.environ, "PYTHONPATH": str(ROOT), "LOG3_VAULT_PASSPHRASE": PASSPHRASE,
                    "LOG3_NODE_TOKEN": TOKEN, "LOG3_VAULT_REPO": str(repo), "LOG3_NODE_QUIET": "1",
                    "PYTHONIOENCODING": "utf-8", "LOG3_VAULT_RUSAGE": "1"}
        self.tmp = tmp

    def __call__(self, *args, env: dict | None = None, ok=(0,)) -> dict:
        t0 = time.perf_counter()
        p = subprocess.run([sys.executable, "-m", "log3.vault", str(args[0]), "--json", *map(str, args[1:])],
                           env={**self.env, **(env or {})}, capture_output=True, text=True, encoding="utf-8",
                           stdin=subprocess.DEVNULL, timeout=7200)
        out = json.loads(p.stdout)
        assert p.returncode in ok, (args, p.returncode, out)
        out["_wall"] = time.perf_counter() - t0
        return out


def _bytes(root: Path) -> int:
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())


def measure(work: Path, scale_mib: int) -> dict:
    src = work / "src"
    make_corpus(src / "projects", seed=11, scale_mib=scale_mib, max_file_kib=4096)
    make_mixed(src / "mixed", big_mib=16)
    plain = _bytes(src)
    certs, node_dir, disk = work / "certs", work / "node", work / "disk"
    repo = work / "repo"
    cli = Cli(work, repo)
    cli("init", "--min-copies", "2")
    cli("backend", "add", "disk", "--type", "localdir", "--path", disk, "--domain", "disk-a")
    cli("node-cert", "--dir", certs, "--host", "127.0.0.1")
    port = free_port()
    node = subprocess.Popen([sys.executable, "-m", "log3.vault", "serve-node", "--data", str(node_dir), "--cert",
                             str(certs / "node.crt"), "--key", str(certs / "node.key"), "--host", "127.0.0.1",
                             "--port", str(port)], env=cli.env, stdin=subprocess.DEVNULL,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        time.sleep(1.5)
        cli("backend", "add", "node", "--type", "node", "--url", f"https://127.0.0.1:{port}", "--ca-file",
            certs / "node.crt", "--domain", "site-b")
        acc = cli("backup", src)
        vid = acc["version"]["version_id"]
        run = cli("run")
        status = cli("status", vid)
        ver = cli("verify")
        dest = work / "restored"
        rst = cli("restore", vid, dest)
        drill = cli("drill")
        identical = tree_digest(src) == tree_digest(dest)
        stored = _bytes(disk / "objects")
        metrics = cli("metrics")["metrics"]
        rec = Cli(work, work / "repo-recovered")("recover", "disk", "--type", "localdir", "--path", disk,
                                                 "--domain", "disk-a")
        # overhead адаптації: таке саме захоплення з вимкненим навчанням
        off = Cli(work, work / "repo-off")
        off("init", "--min-copies", "1")
        off("backend", "add", "disk", "--type", "localdir", "--path", work / "disk-off", "--domain", "disk-a")
        off("learn", "disable")
        acc_off = off("backup", src)
    finally:
        if os.name == "nt":
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(node.pid)], capture_output=True)
        node.kill()
        node.wait()
    mib = plain / MIB
    return {
        "plain_bytes": plain, "files": acc["version"]["files"],
        "restore_identical_fraction": 1.0 if identical else 0.0,
        "policy_satisfied_fraction": 1.0 if status["version"]["policy_satisfied"] else 0.0,
        "objects_under_replicated": status["version"]["copies"]["objects_under_replicated"],
        "verify_ok_fraction": ver["verify"]["ok"] / max(1, ver["verify"]["checked"]),
        "stored_to_plain_per_copy": round(stored / plain, 4),
        "backup_accept_seconds": round(acc["_wall"], 3),
        "backup_accept_mib_s": round(mib / acc["_wall"], 2),
        "replicate_two_copies_seconds": round(run["_wall"], 3),
        "time_to_protected_seconds": metrics["last_time_to_protected_seconds"],
        "verify_seconds": round(ver["_wall"], 3),
        "restore_seconds": round(rst["_wall"], 3),
        "restore_mib_s": round(mib / rst["_wall"], 2),
        "drill_ok": drill["drill"]["ok"],
        "recover_catalog_seconds": round(rec["_wall"], 3),
        "cpu_seconds": {"backup": acc["rusage"]["cpu_seconds"], "run": run["rusage"]["cpu_seconds"],
                        "restore": rst["rusage"]["cpu_seconds"]},
        "peak_rss_mib": round(max(x["rusage"]["peak_rss_kib"] or 0 for x in (acc, run, rst, ver)) / 1024, 1),
        "adaptation_cpu_overhead": round(acc["rusage"]["cpu_seconds"] / acc_off["rusage"]["cpu_seconds"] - 1, 4),
        "adaptation_wall_overhead": round(acc["_wall"] / acc_off["_wall"] - 1, 4),
    }


def check_targets(res: dict, spec: dict | None) -> list[dict]:
    if not spec:
        return []
    out = []
    for key, rule in spec["targets"].items():
        value = res[key]
        ok = value >= rule["min"] if "min" in rule else value <= rule["max"]
        out.append({"metric": key, "value": value, **rule, "ok": ok})
    return out


def to_md(r: dict) -> str:
    m = r["measurements"]
    lines = [f"# LOG3 Vault — продуктові вимірювання ({r['date']})", "",
             f"Машина: {r['machine']['cpu']}, {r['machine']['logical_cpus']} логічних CPU, {r['machine']['os']}, "
             f"Python {r['machine']['python']}.", "",
             f"Дані: {m['files']} файлів, {m['plain_bytes'] / MIB:.1f} МіБ (синтетичні проєкти + змішаний каталог). "
             "Сховище, backend-каталог і вузол — на одному диску; копій 2 (каталог + вузол HTTPS).", "",
             "| Показник | Значення | Ціль | Виконано |", "| :-- | --: | :-- | :-- |"]
    targets = {t["metric"]: t for t in r["targets"]}
    for k, v in m.items():
        if isinstance(v, dict):
            v = ", ".join(f"{a} {b}" for a, b in v.items())
        t = targets.get(k)
        goal = "" if not t else (f"≥ {t['min']}" if "min" in t else f"≤ {t['max']}")
        lines.append(f"| {k} | {v} | {goal} | {'' if not t else ('так' if t['ok'] else 'НІ')} |")
    lines += ["", r["note"]]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--scale-mib", type=int, default=96)
    p.add_argument("--out", default=str(ROOT / "results" / "vault"))
    p.add_argument("--targets", default=str(ROOT / "results" / "vault" / "targets.json"))
    a = p.parse_args(argv)
    m = machine()
    work = Path(tempfile.mkdtemp(prefix="log3-bench-"))
    try:
        res = measure(work, a.scale_mib)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    specs = json.loads(Path(a.targets).read_text(encoding="utf-8")) if Path(a.targets).exists() else {}
    spec = specs.get(fingerprint(m))
    report = {"date": time.strftime("%Y-%m-%d"), "machine": m, "fingerprint": fingerprint(m),
              "scale_mib": a.scale_mib, "measurements": res, "targets": check_targets(res, spec),
              "note": ("Цілі перевірено для цього комп'ютера." if spec else
                       "Для цього комп'ютера цілей нема: записано лише виміри, нічого не заявлено.")
              + " Числа описують цей прогін на одному диску й не є універсальною продуктивністю."}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "product_bench.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "product_bench.md").write_text(to_md(report), encoding="utf-8")
    print(json.dumps({"ok": all(t["ok"] for t in report["targets"]), "targets_checked": len(report["targets"]),
                      **res}, ensure_ascii=False))
    return 0 if all(t["ok"] for t in report["targets"]) else 4


if __name__ == "__main__":
    sys.exit(main())
