"""Ліміт розміру кожного нового коміту: додані + видалені текстові рядки ≤ 1000 (не на весь PR).

    python scripts/check_commit_size.py origin/dev..HEAD [--limit 1000]

Рахується `git show --numstat` кожного коміту без злиттів (merge-коміт нових рядків не пише, лише
поєднує вже перевірені коміти). Двійкові файли (numstat «-») не є текстовими рядками й не рахуються.
Код 0 — усі коміти в межах, 1 — є перевищення, 2 — помилка git.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys


def git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, encoding="utf-8", check=True).stdout


def changed_lines(sha: str) -> tuple[int, int]:
    """(текстові рядки, двійкові файли) коміту."""
    text = binary = 0
    for line in git("show", "--numstat", "--format=", sha).splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        if parts[0] == "-":
            binary += 1
        else:
            text += int(parts[0]) + int(parts[1])
    return text, binary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("range", help="діапазон комітів, напр. origin/dev..HEAD")
    p.add_argument("--limit", type=int, default=1000)
    a = p.parse_args(argv)
    try:
        shas = git("rev-list", "--no-merges", "--reverse", a.range).split()
    except subprocess.CalledProcessError as e:
        print(e.stderr, file=sys.stderr)
        return 2
    rows, bad = [], 0
    for sha in shas:
        lines, binary = changed_lines(sha)
        over = lines > a.limit
        bad += over
        subject = git("log", "-1", "--format=%s", sha).strip()
        rows.append({"commit": sha[:10], "lines": lines, "binary_files": binary, "ok": not over, "subject": subject})
        print(f"{'ПЕРЕВИЩЕНО' if over else 'ok':10} {sha[:10]} {lines:5} рядків  {subject[:70]}")
    print(json.dumps({"ok": bad == 0, "limit": a.limit, "commits": len(rows), "over_limit": bad},
                     ensure_ascii=False))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
