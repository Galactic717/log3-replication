"""Виконавець демонстрації: справжні команди через subprocess (sys.executable + масив аргументів).

Для кожного кроку друкується мета, фактична команда без секретів, живий вивід, тривалість, код
завершення й перевірений результат. Перевірка — окрема функція, що читає вивід команди або стан
на диску; невдала перевірка позначає крок як провалений, але демонстрація йде далі. Журнал (повний
вивід без секретів) і машинозчитуваний підсумок пишуться в каталог --out.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

LINE_LIMIT = 400


class V(str):
    """Значення, відоме лише під час виконання (ідентифікатор версії тощо): ctx.vars[ім'я]."""


def kill_tree(proc: subprocess.Popen) -> None:
    """Зупинити процес із дочірніми: у Windows python із venv — це лаунчер, справжній інтерпретатор
    є його дочірнім процесом і після kill() лаунчера лишився б працювати."""
    if proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True)
    else:
        proc.kill()
    proc.wait()


@dataclass
class Result:
    code: int | None
    out: str
    seconds: float
    data: dict | None = None          # останній JSON-об'єкт у виводі


@dataclass
class Step:
    goal: str
    argv: list | None = None                          # справжня команда (елементи str | Path | V)
    action: Callable | None = None                    # дія над станом, напр. зупинка вузла: ctx -> опис
    check: Callable | None = None                     # (ctx, Result) -> опис перевіреного; AssertionError — провал
    expect: tuple = (0,)
    env: dict | None = None                           # додаткові змінні (несекретні показуються)
    cwd: Path | None = None
    background: str | None = None                     # назва фонового процесу (вузол): не чекати завершення
    timeout: float = 1800.0


@dataclass
class Stage:
    name: str
    title: str
    kind: str                                         # новий експеримент / перегляд збережених / продукт / ...
    build: Callable                                   # ctx -> list[Step]
    profiles: tuple = ("quick", "full")
    note: str = ""


@dataclass
class Ctx:
    root: Path                                        # корінь проєкту (код, що демонструється)
    work: Path                                        # ізольований робочий каталог
    profile: str
    env: dict
    secrets: list = field(default_factory=list)
    vars: dict = field(default_factory=dict)
    procs: dict = field(default_factory=dict)
    py: str = sys.executable

    def resolve(self, item) -> str:
        return str(self.vars[item]) if isinstance(item, V) else str(item)

    def redact(self, text: str) -> str:
        for s in self.secrets:
            if s:
                text = text.replace(s, "***")
        return text

    def show_argv(self, argv: list) -> str:
        parts = []
        for a in argv:
            s = f"<{a}>" if isinstance(a, V) and a not in self.vars else self.resolve(a)
            if s == self.py:
                s = "python"
            parts.append(s if s and " " not in s else f'"{s}"')
        return self.redact(" ".join(parts))


def last_json(text: str) -> dict | None:
    """Увесь вивід як JSON-об'єкт (з відступами) або останній однорядковий JSON-об'єкт."""
    try:
        whole = json.loads(text)
        if isinstance(whole, dict):
            return whole
    except ValueError:
        pass
    for line in reversed(text.strip().splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except ValueError:
                continue
    return None


class Runner:
    def __init__(self, ctx: Ctx, *, out_dir: Path, dry_run: bool = False, step_mode: bool = False,
                 quiet: bool = False):
        self.ctx, self.dry, self.step_mode, self.quiet = ctx, dry_run, step_mode, quiet
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        self.log_path = out_dir / f"defense-{stamp}.log"
        self.summary_path = out_dir / f"defense-{stamp}.json"
        self.log = open(self.log_path, "w", encoding="utf-8")
        self.stages: list[dict] = []

    def say(self, text: str = "", *, log_only: bool = False) -> None:
        text = self.ctx.redact(text)
        self.log.write(text + "\n")
        self.log.flush()
        if not self.quiet and not log_only:
            print(text, flush=True)

    def _execute(self, step: Step) -> Result:
        argv = [self.ctx.resolve(a) for a in step.argv]
        extra = {k: self.ctx.resolve(v) for k, v in (step.env or {}).items()}
        env = {**self.ctx.env, **extra}
        t0 = time.perf_counter()
        if step.background:
            proc = subprocess.Popen(argv, env=env, cwd=step.cwd or self.ctx.work, stdin=subprocess.DEVNULL,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.ctx.procs[step.background] = proc
            return Result(None, f"фоновий процес PID {proc.pid}", time.perf_counter() - t0)
        proc = subprocess.Popen(argv, env=env, cwd=step.cwd or self.ctx.work, stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                errors="replace")
        lines = []
        watchdog = threading.Timer(step.timeout, kill_tree, (proc,))    # зависла команда не зупиняє демонстрацію
        watchdog.start()
        try:
            for line in proc.stdout:                           # живий вивід, рядок за рядком
                lines.append(line)
                shown = self.ctx.redact(line.rstrip("\n"))
                self.log.write("  │ " + shown + "\n")
                if not self.quiet:
                    extra = len(shown) - LINE_LIMIT
                    cut = f" … (+{extra} симв., повністю — у журналі)" if extra > 0 else ""
                    print("  │ " + shown[:LINE_LIMIT] + cut, flush=True)
            code = proc.wait()
        finally:
            watchdog.cancel()
        out = "".join(lines)
        return Result(code, out, time.perf_counter() - t0, last_json(out))

    def run_stage(self, stage: Stage, index: int, total: int) -> dict:
        self.say("")
        self.say("═" * 78)
        self.say(f"ЕТАП {index}/{total} · {stage.title}   [{stage.kind}]")
        if stage.note:
            self.say(f"  примітка: {stage.note}")
        self.say("═" * 78)
        record = {"stage": stage.name, "title": stage.title, "kind": stage.kind, "note": stage.note, "steps": []}
        try:
            steps = stage.build(self.ctx)
        except Exception as e:                         # етап не зміг підготуватись (напр. нема materials/)
            self.say(f"  етап пропущено: {e}")
            record.update(status="skipped", reason=str(e))
            return record
        failed = 0
        for i, step in enumerate(steps, 1):
            rec = self.run_step(step, i, len(steps))
            record["steps"].append(rec)
            failed += rec["status"] == "failed"
            if rec["status"] == "quit":
                record["status"] = "aborted"
                return record
        record["status"] = "dry-run" if self.dry else ("failed" if failed else "passed")
        return record

    def run_step(self, step: Step, i: int, n: int) -> dict:
        ctx = self.ctx
        self.say(f"\n── крок {i}/{n}: {step.goal}")
        shown_env = {k: v for k, v in (step.env or {}).items() if not any(s in k for s in ("PASSPHRASE", "TOKEN"))}
        if step.argv:
            env_text = " ".join(f"{k}={ctx.resolve(v)}" for k, v in shown_env.items())
            self.say(f"   команда: {(env_text + ' ') if env_text else ''}{ctx.show_argv(step.argv)}")
        else:
            self.say("   дія над станом (без зовнішньої команди)")
        rec = {"goal": step.goal, "command": ctx.show_argv(step.argv) if step.argv else None,
               "env": {k: ctx.redact(str(ctx.resolve(v))) for k, v in shown_env.items()}}
        if self.dry:
            return {**rec, "status": "dry-run"}
        if self.step_mode and sys.stdin.isatty():
            answer = input("   Enter — виконати, s — пропустити, q — завершити: ").strip().lower()
            if answer == "q":
                return {**rec, "status": "quit"}
            if answer == "s":
                return {**rec, "status": "skipped"}
        try:
            if step.argv:
                res = self._execute(step)
            else:
                t0 = time.perf_counter()
                text = step.action(ctx) if step.action else ""
                res = Result(0, text or "", time.perf_counter() - t0)
                if text:
                    self.say(f"  │ {text}")
            ok_code = res.code is None or res.code in step.expect
            verified = step.check(ctx, res) if step.check and ok_code else ""
            status = "passed" if ok_code else "failed"
            if not ok_code:
                verified = f"код {res.code}, очікувався {'/'.join(map(str, step.expect))}"
        except AssertionError as e:
            status, verified = "failed", f"перевірку НЕ пройдено: {e}"
            res = locals().get("res") or Result(None, "", 0.0)
        except Exception as e:                         # збій самої демонстрації — видно, а не приховано
            status, verified, res = "failed", f"помилка демонстрації: {type(e).__name__}: {e}", Result(None, "", 0.0)
        mark = "✔" if status == "passed" else "✘"
        code = "—" if res.code is None else res.code
        self.say(f"   {mark} код {code} · {res.seconds:.2f} с · {verified or 'перевірено: код завершення'}")
        return {**rec, "status": status, "exit_code": res.code, "expected": list(step.expect),
                "seconds": round(res.seconds, 3), "verified": ctx.redact(str(verified))}

    def finish(self, extra: dict) -> dict:
        for proc in self.ctx.procs.values():
            kill_tree(proc)
        steps = [s for st in self.stages for s in st["steps"]]
        summary = {
            "format": "log3-defense-summary", "format_version": 1,
            "finished_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "environment": {"python": platform.python_version(), "os": platform.platform(),
                            "machine": platform.machine(), "processor": platform.processor(),
                            "cpu_count": os.cpu_count()},
            **extra,
            "totals": {"stages": len(self.stages),
                       "stages_passed": sum(s.get("status") == "passed" for s in self.stages),
                       "stages_failed": sum(s.get("status") == "failed" for s in self.stages),
                       "stages_skipped": sum(s.get("status") == "skipped" for s in self.stages),
                       "steps": len(steps), "steps_failed": sum(s["status"] == "failed" for s in steps)},
            "stages": self.stages, "log": str(self.log_path)}
        self.summary_path.write_text(self.ctx.redact(json.dumps(summary, ensure_ascii=False, indent=1)),
                                     encoding="utf-8")
        self.say("")
        self.say("═" * 78)
        t = summary["totals"]
        self.say(f"ПІДСУМОК: етапів {t['stages']}, пройдено {t['stages_passed']}, провалено {t['stages_failed']}, "
                 f"пропущено {t['stages_skipped']}; кроків {t['steps']}, провалено {t['steps_failed']}")
        self.say(f"журнал: {self.log_path}")
        self.say(f"підсумок JSON: {self.summary_path}")
        self.log.close()
        return summary
