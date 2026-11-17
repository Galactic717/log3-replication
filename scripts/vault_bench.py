"""Вимірювання LOG3 Vault у нових процесах CLI:  python scripts/vault_bench.py baseline|adaptive

Сценарій baseline (відправна точка етапу 2): init -> backend-и (локальний каталог + HTTPS-вузол
окремим процесом) -> backup без --wait (лише локальне приймання; процес завершується, як після
перезапуску) -> перезапуск вузла -> run -> status -> verify -> restore -> побайтове порівняння.

Кожна команда — новий процес, тож час включає старт інтерпретатора. CPU і пікова RSS — дочірніх
процесів (resource.RUSAGE_CHILDREN). Дані — синтетичний корпус scripts/vault_corpus.py.

Сценарій adaptive: однакове навантаження в окремих сховищах для гілок навчання off / defaults /
accelerated; щораунду backup -> run (run виконує крок циклу навчання), між раундами змінюються файли.
Наприкінці — verify, restore, втрата 10 % копій на диску бенчмарку і repair. Порівнюються байти,
CPU, затримка підтвердження захисту, черга, RSS, час restore і repair; для accelerated додається
фінальний звіт відбору (Store, фіксований кодек, база, навчені, ГА проти random search).
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

    def configure(self, section: str, **values) -> None:
        path = self.repo / "config.json"
        cfg = json.loads(path.read_text(encoding="utf-8"))
        cfg.setdefault(section, {}).update(values)
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


def baseline(scale_mib: float, seed: int, tmp: Path, max_file_kib: int = 512) -> dict:
    """backup -> restart -> verify -> restore на коді етапу 1 (advisor не налаштовано)."""
    b = Bench(tmp)
    src = tmp / "src"
    corpus = make_corpus(src, seed, scale_mib, max_file_kib)
    b.setup()
    b.configure("learning", enabled=False)                  # саме тракт етапу 1, без ознак і журналу
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


#: гілки порівняння: однакове навантаження, різні налаштування навчання
ARMS = {
    "off": {"enabled": False},
    "defaults": {"enabled": True, "train": {"min_interval_seconds": 0}},
    "accelerated": {"enabled": True,
                    "probe": {"rate": 1.0, "max_chunks_per_operation": 100000, "max_cpu_seconds_per_operation": 60,
                              "max_cpu_fraction": 4, "max_load_per_cpu": 100},
                    "train": {"min_interval_seconds": 0, "min_rows": 100, "min_groups": 10},
                    "gate": {"min_shadow_decisions": 50}},
}


def _sql(db: Path, q: str, args: tuple = ()):
    con = sqlite3.connect(db)
    try:
        return con.execute(q, args).fetchall()
    finally:
        con.close()


def _arm(name: str, learning: dict, tmp: Path, scale_mib: float, seed: int, rounds: int, chunk_mib: float,
         max_file_kib: int) -> dict:
    b = Bench(tmp)
    src = tmp / "src"
    make_corpus(src, seed, scale_mib, max_file_kib)
    b.setup(chunk_mib=chunk_mib)
    b.configure("learning", **learning)
    cat = b.repo / "catalog.sqlite"
    per_round, events = [], []
    try:
        for r in range(rounds):
            n0 = len(b.steps)
            vid = b.cli(f"backup r{r}", "backup", src)["version"]["version_id"]
            queued = _sql(cat, "SELECT COUNT(*) FROM jobs WHERE state IN ('PENDING','RUNNING')")[0][0]
            run = b.cli(f"run r{r}", "run")
            lat = _sql(cat, "SELECT (protected_ns - accepted_ns) / 1e9 FROM versions WHERE version_id=?", (vid,))[0][0]
            new = _sql(cat, "SELECT COUNT(*), COALESCE(SUM(o.stored_len),0), COALESCE(SUM(o.plain_len),0) "
                       "FROM objects o WHERE o.kind=1 AND o.created_ns=(SELECT created_ns FROM versions "
                       "WHERE version_id=?)", (vid,))[0]
            lr = run.get("learning") or {}
            tr = lr.get("train") or {}
            ev = lr.get("evaluate") or {}
            events.append({"round": r, "train": tr.get("status"), "selected": tr.get("selected"),
                           "gate_gain": (tr.get("gate") or {}).get("gain"),
                           "evaluate": ev.get("status"), "evaluate_passed": ev.get("passed"),
                           "shadow_gain": ev.get("gain"), "shadow_rows": ev.get("probed_shadow_rows"),
                           "promote": (lr.get("promote") or {}).get("status"),
                           "health": (lr.get("health") or {}).get("status"),
                           "train_cpu_seconds": (tr.get("resources") or {}).get("cpu_seconds")})
            per_round.append({"round": r, "backup_wall_seconds": b.steps[n0]["wall_seconds"],
                              "backup_cpu_seconds": b.steps[n0]["cpu_seconds"],
                              "run_wall_seconds": b.steps[n0 + 1]["wall_seconds"],
                              "run_cpu_seconds": b.steps[n0 + 1]["cpu_seconds"], "queue_after_backup": queued,
                              "time_to_protected_seconds": lat, "new_chunks": new[0], "new_stored_bytes": new[1],
                              "new_plain_bytes": new[2]})
            mutate(src, seed * 100 + r, 0.3, 0.1)
        b.cli("verify", "verify")
        t0 = time.perf_counter()
        b.cli("restore", "restore", "latest", tmp / "out")
        restore_s = time.perf_counter() - t0
        objs = sorted(p for p in (tmp / "disk" / "objects").rglob("*") if p.is_file())
        for p in objs[::10]:                                # втрата 10 % копій на диску бенчмарку (не замовника)
            p.unlink()
        b.cli("verify after loss", "verify", check=False)
        t0 = time.perf_counter()
        b.cli("repair", "repair")
        repair_s = time.perf_counter() - t0
        status = b.cli("learn status", "learn", "status")["learning"] if learning.get("enabled") else None
        final_train = None
        if name == "accelerated":
            final_train = b.cli("learn train (final report)", "learn", "train", "--force", check=False)
    finally:
        b.stop_node()
    totals = _sql(cat, "SELECT COALESCE(SUM(stored_len),0), COALESCE(SUM(plain_len),0) FROM objects WHERE kind=1")[0]
    lat = [x["time_to_protected_seconds"] for x in per_round]
    out = {"learning": learning, "rounds": per_round, "lifecycle": events,
           "totals": {"stored_bytes": totals[0], "plain_bytes": totals[1],
                      "stored_to_plain": round(totals[0] / max(1, totals[1]), 5),
                      "backup_cpu_seconds": round(sum(x["backup_cpu_seconds"] for x in per_round), 3),
                      "run_cpu_seconds": round(sum(x["run_cpu_seconds"] for x in per_round), 3),
                      "backup_wall_seconds": round(sum(x["backup_wall_seconds"] for x in per_round), 3),
                      "time_to_protected_p50_seconds": round(float(sorted(lat)[len(lat) // 2]), 3),
                      "time_to_protected_max_seconds": round(max(lat), 3),
                      "queue_after_backup_max": max(x["queue_after_backup"] for x in per_round),
                      "restore_latest_seconds": round(restore_s, 3),
                      "repair_after_10pct_loss_seconds": round(repair_s, 3),
                      "children_max_rss_kib": max(st["children_max_rss_kib"] for st in b.steps)},
           "steps": b.steps}
    if status is not None:
        out["learning_status"] = {k: status.get(k) for k in ("state", "journal", "overhead_last_20_operations",
                                                             "models", "learning_errors_total")}
    if final_train is not None:
        out["final_train_report"] = final_train.get("learning")
    return out


def adaptive(scale_mib: float, seed: int, tmp: Path, *, rounds: int, chunk_mib: float, max_file_kib: int,
             arms: list[str]) -> dict:
    out = {"scenario": "однакове навантаження (backup -> run щораунду, зміни файлів між раундами), гілки "
                       "навчання: off — етап 1, defaults — типові налаштування, accelerated — прискорений збір",
           "scale_mib": scale_mib, "rounds": rounds, "chunk_mib": chunk_mib, "seed": seed, "arms": {}}
    for name in arms:
        out["arms"][name] = _arm(name, ARMS[name], tmp / name, scale_mib, seed, rounds, chunk_mib, max_file_kib)
    off = out["arms"].get("off")
    if off:
        for name, arm in out["arms"].items():
            if name != "off":
                a, o = arm["totals"], off["totals"]
                out.setdefault("vs_off", {})[name] = {
                    "stored_bytes_change": round(a["stored_bytes"] / o["stored_bytes"] - 1, 5),
                    "backup_cpu_change": round(a["backup_cpu_seconds"] / o["backup_cpu_seconds"] - 1, 4),
                    "run_cpu_change": round(a["run_cpu_seconds"] / o["run_cpu_seconds"] - 1, 4),
                    "time_to_protected_p50_change_seconds": round(a["time_to_protected_p50_seconds"]
                                                                  - o["time_to_protected_p50_seconds"], 3),
                    "restore_seconds_change": round(a["restore_latest_seconds"] - o["restore_latest_seconds"], 3),
                    "repair_seconds_change": round(a["repair_after_10pct_loss_seconds"]
                                                   - o["repair_after_10pct_loss_seconds"], 3)}
    return out



def _f(x, nd=3):
    return "—" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def summary_md(res: dict, title: str) -> str:
    """Markdown-звіт сценарію adaptive: гілки, шкода тракту, цикл, відбір кандидатів."""
    arms, env = res["arms"], res["environment"]
    lines = [f"# {title}", "",
             f"Корпус {res['scale_mib']} МіБ (синтетичний, `scripts/vault_corpus.py`, seed {res['seed']}), "
             f"{res['rounds']} раундів backup -> run, блок {res['chunk_mib']} МіБ; середовище: {env['platform']}, "
             f"CPU {env['cpus']}, Python {env['python']}.", "",
             "| гілка | байтів / вхід | CPU backup, с | CPU run, с | затримка захисту p50 / max, с | "
             "черга після backup, max | restore, с | repair після втрати 10 %, с | пікова RSS, КіБ |",
             "| :-- | --: | --: | --: | --: | --: | --: | --: | --: |"]
    for n, a in arms.items():
        t = a["totals"]
        lines.append(f"| {n} | {t['stored_to_plain']:.4f} | {t['backup_cpu_seconds']:.2f} | "
                     f"{t['run_cpu_seconds']:.2f} | {t['time_to_protected_p50_seconds']:.2f} / "
                     f"{t['time_to_protected_max_seconds']:.2f} | {t['queue_after_backup_max']} | "
                     f"{t['restore_latest_seconds']:.2f} | {t['repair_after_10pct_loss_seconds']:.2f} | "
                     f"{t['children_max_rss_kib']} |")
    for n, v in (res.get("vs_off") or {}).items():
        lines += ["", f"**{n} проти off:** байти {v['stored_bytes_change']:+.2%}, CPU backup "
                  f"{v['backup_cpu_change']:+.1%}, CPU run {v['run_cpu_change']:+.1%}, затримка захисту p50 "
                  f"{v['time_to_protected_p50_change_seconds']:+.3f} с, restore {v['restore_seconds_change']:+.3f} с, "
                  f"repair {v['repair_seconds_change']:+.3f} с."]
    for n, a in arms.items():
        ls = a.get("learning_status")
        if not ls:
            continue
        o, st, jr = ls["overhead_last_20_operations"], ls["state"], ls["journal"]
        lines += ["", f"## {n}: overhead і цикл", "",
                  f"Ознаки {o['feature_ns_per_chunk'] / 1e3:.0f} мкс на блок; проби "
                  f"{o['probe_cpu_seconds_per_gib']:.2f} CPU-с/ГіБ проти кодування "
                  f"{o['encode_cpu_seconds_per_gib']:.2f} CPU-с/ГіБ; висновок {o['inference_ns_per_chunk'] / 1e3:.1f} "
                  f"мкс на блок; fallback {o['fallbacks']}. Рішень у журналі {jr['decisions']}, з пробами "
                  f"{jr['probed_decisions']}, помилок навчання {ls['learning_errors_total']}. Підсумковий режим: "
                  f"**{st['mode']}**, активна модель {st['active_model'] or '—'}.", "",
                  "| раунд | train | обрано | виграш на test | evaluate (виграш у shadow, рядків) | promote | health | "
                  "CPU навчання, с |",
                  "| --: | :-- | :-- | --: | :-- | :-- | :-- | --: |"]
        for e in a["lifecycle"]:
            ev = _f(e["evaluate"]) + ("" if e.get("shadow_gain") is None else
                                      f" ({e['shadow_gain']:+.4f}, {e.get('shadow_rows')})")
            lines.append(f"| {e['round']} | {_f(e['train'])} | {_f(e['selected'])} | {_f(e['gate_gain'], 4)} | "
                         f"{ev} | {_f(e['promote'])} | {_f(e['health'])} | {_f(e['train_cpu_seconds'])} |")
        ft = a.get("final_train_report")
        if ft and ft.get("test"):
            g, pv = ft["gate"], ft["provenance"]
            lines += ["", f"### {n}: фінальний відбір кандидатів", "",
                      f"Частини часово відокремлені, лінії файлів не перетинаються: train {pv['train']['rows']} "
                      f"рядків / {pv['train']['groups']} груп, validation {pv['validation']['rows']} / "
                      f"{pv['validation']['groups']}, test {pv['test']['rows']} / {pv['test']['groups']}; відкинуто "
                      f"через перетин груп {pv['rows_dropped_group_overlap']}. Вартість — відносно оракула (1.0 — "
                      "найдешевша дія для кожного блока); навчені кандидати платять overhead ознак, висновку й "
                      "амортизованого навчання.", "",
                      "| кандидат | validation | test | test без overhead | байтів / вхід (test) | "
                      "CPU кодування, с/ГіБ (test) |",
                      "| :-- | --: | --: | --: | --: | --: |"]
            for name, t in ft["test"].items():
                v = ft["validation"][name]
                mark = " **(обрано)**" if name == ft["selected"] else ""
                gross = t.get("cost_vs_oracle_without_overhead")
                lines.append(f"| {name}{mark} | {v['cost_vs_oracle']:.4f} | {t['cost_vs_oracle']:.4f} | "
                             f"{_f(gross, 4)} | {t['stored_to_raw']:.4f} | {t['encode_cpu_seconds_per_gib']:.3f} |")
            ga, rs = ft["search"]["ga"], ft["search"]["random"]
            checks = ", ".join(f"{k} {'✓' if v else '✗'}" for k, v in g["checks"].items())
            lines += ["", f"ГА проти random search за однакового бюджету ({ga['evaluations']} оцінок кожен): "
                      f"найкраща validation-вартість ГА {ga['validation_cost_vs_oracle']:.4f} ({ga['best_kind']}), "
                      f"random search {rs['validation_cost_vs_oracle']:.4f} ({rs['best_kind']}).", "",
                      f"Gate: {'пройдено' if g['passed'] else 'не пройдено'}; виграш над базою на test "
                      f"{g['gain']:+.4f} (90 % bootstrap за групами {g['low']:+.4f} .. {g['high']:+.4f}, груп "
                      f"{g['groups']}), поріг {g['min_relative_gain']}. Перевірки: {checks}."]
    return "\n".join(lines) + "\n"


def dump_json(obj, level: int = 0, max_level: int = 4) -> str:
    """JSON із відступами лише на верхніх рівнях: рядок на раунд, крок чи кандидата, а не на кожне поле."""
    pad = "  " * level
    if level >= max_level or not isinstance(obj, dict | list) or not obj:
        return json.dumps(obj, ensure_ascii=False)
    if isinstance(obj, dict):
        items = [f'{pad}  {json.dumps(str(k), ensure_ascii=False)}: {dump_json(v, level + 1, max_level)}'
                 for k, v in obj.items()]
        return "{\n" + ",\n".join(items) + f"\n{pad}}}"
    return "[\n" + ",\n".join(f"{pad}  {dump_json(v, level + 1, max_level)}" for v in obj) + f"\n{pad}]"


def environment() -> dict:
    return {"python": platform.python_version(), "platform": platform.platform(), "cpus": os.cpu_count()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("scenario", choices=("baseline", "adaptive", "summary"))
    ap.add_argument("--input", type=Path, help="summary: JSON сценарію adaptive")
    ap.add_argument("--md", type=Path, help="adaptive/summary: куди записати markdown-звіт")
    ap.add_argument("--title", default="Навчання вибору стиснення: вимірювання на тракті")
    ap.add_argument("--scale-mib", type=float, default=32.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-file-kib", type=int, default=512)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--chunk-mib", type=float, default=0.25)
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--out", type=Path, help="куди записати JSON (типово stdout)")
    a = ap.parse_args(argv)
    if a.scenario == "summary":
        text = summary_md(json.loads(a.input.read_text(encoding="utf-8")), a.title)
        if a.md:
            a.md.write_text(text, encoding="utf-8")
        print(text)
        return 0
    with tempfile.TemporaryDirectory(prefix="log3-bench-") as d:
        if a.scenario == "baseline":
            result = {"environment": environment(), **baseline(a.scale_mib, a.seed, Path(d), a.max_file_kib)}
        else:
            result = {"environment": environment(), **adaptive(a.scale_mib, a.seed, Path(d), rounds=a.rounds,
                                                               chunk_mib=a.chunk_mib, max_file_kib=a.max_file_kib,
                                                               arms=a.arms)}
    text = dump_json(result)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(text + "\n", encoding="utf-8")
    if a.md and a.scenario == "adaptive":
        a.md.write_text(summary_md(result, a.title), encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
