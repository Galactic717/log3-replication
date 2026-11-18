"""Автоматичний захист проєкту: python -m log3.defense [--profile quick|full] [--stage ...] [--step] [--dry-run]

Запускає справжні команди (sys.executable + масив аргументів) і для кожного кроку показує мету,
команду без секретів, живий вивід, тривалість, код завершення й перевірений результат.

  --list              перелік етапів і профілів
  --profile quick     швидкий показ (типово): вибірка тестів, перегляд збережених результатів ГА й перцептрона,
                      частина матриці відмов
  --profile full      повний: усі тести, нові експерименти в ізольованій копії, уся матриця відмов
  --stage NAME        лише цей етап (можна кілька; 'matrix' — усі сценарії матриці відмов)
  --step              покроковий режим: Enter перед кожним кроком (s — пропустити, q — завершити)
  --dry-run           показати етапи й команди, нічого не виконуючи
  --out DIR           журнал і машинозчитуваний підсумок (типово ./defense-out)
  --keep              не видаляти робочий каталог із тимчасовими сховищами

Демонстрації працюють у тимчасовому каталозі; навчальні експерименти — в ізольованій копії проєкту,
тому закомічені results/, модель, materials/ і копії користувача не перезаписуються.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from . import matrix
from .lab import base_env
from .runner import Ctx, Runner
from .stages import STAGES, has_materials

ROOT = Path(__file__).resolve().parents[2]


def all_stages() -> list:
    return [*STAGES, *matrix.stages()]


def select(names: list[str] | None, profile: str) -> list:
    stages = all_stages()
    if not names:
        return [s for s in stages if profile in s.profiles]
    out = []
    for n in names:
        hit = [s for s in stages if s.name == n or s.name.startswith(n + ":")]
        if not hit:
            raise SystemExit(f"невідомий етап {n!r}; див. --list")
        out += [s for s in hit if s not in out and (s.name == n or profile in s.profiles)]
    return out


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(prog="python -m log3.defense", description=__doc__.split("\n\n")[0])
    p.add_argument("--profile", choices=("quick", "full"), default="quick")
    p.add_argument("--stage", action="append")
    p.add_argument("--list", action="store_true")
    p.add_argument("--step", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--out", default="defense-out")
    p.add_argument("--keep", action="store_true")
    p.add_argument("--quiet", action="store_true", help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    if a.list:
        for s in all_stages():
            print(f"{s.name:38} {'/'.join(s.profiles):11} [{s.kind}] {s.title}")
        return 0
    stages = select(a.stage, a.profile)
    out = Path(a.out).absolute()
    out.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="log3-defense-", dir=out))
    ctx = Ctx(root=ROOT, work=work, profile=a.profile, env=base_env(ROOT))
    runner = Runner(ctx, out_dir=out, dry_run=a.dry_run, step_mode=a.step, quiet=a.quiet)
    materials = has_materials(ctx)
    runner.say(f"LOG3 · автоматичний захист · профіль {a.profile}{' · DRY-RUN' if a.dry_run else ''}")
    runner.say(f"проєкт: {ROOT}")
    runner.say(f"робочий каталог: {work}")
    runner.say("materials/: " + ("є (пакет викладача T01)" if materials else
                                 "нема — навчальні етапи в dev-режимі, кейси з даними викладача пропускаються"))
    for i, stage in enumerate(stages, 1):
        runner.stages.append(runner.run_stage(stage, i, len(stages)))
        if runner.stages[-1].get("status") == "aborted":
            break
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                                text=True, timeout=30).stdout.strip() or None
    except OSError:
        commit = None
    summary = runner.finish({"profile": a.profile, "dry_run": a.dry_run, "materials_present": materials,
                             "project_commit": commit, "selected": [s.name for s in stages]})
    if not a.keep:
        shutil.rmtree(work, ignore_errors=True)
    t = summary["totals"]
    return 1 if t["stages_failed"] or t["steps_failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
