"""Перегляд збережених (закомічених) результатів: файл, його SHA-256 і коміт, у якому він востаннє змінювався.

    python -m log3.defense.show results/tables/06_ga_vs_random.md [ще файли ...]

Це не новий експеримент: числа отримано раніше (див. README, «Як розрізняти й відтворювати»).
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    out = []
    for name in argv:
        p = Path(name)
        if not p.is_file():
            print(f"нема файла {name}")
            return 10
        data = p.read_bytes()
        try:
            commit = subprocess.run(["git", "log", "-1", "--format=%h %ad", "--date=short", "--", name],
                                    capture_output=True, text=True, timeout=30).stdout.strip() or None
        except OSError:
            commit = None
        print(f"── {name}  (перегляд збереженого; sha256 {hashlib.sha256(data).hexdigest()[:16]}…, коміт {commit})")
        text = data.decode("utf-8", "replace")
        if p.suffix == ".json":
            text = json.dumps(json.loads(text), ensure_ascii=False, indent=1)
        lines = text.splitlines()
        print("\n".join(lines[:40]) + (f"\n… ще {len(lines) - 40} рядків" if len(lines) > 40 else ""))
        out.append({"file": name, "sha256": hashlib.sha256(data).hexdigest(), "commit": commit})
    print(json.dumps({"ok": True, "viewed": out}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
