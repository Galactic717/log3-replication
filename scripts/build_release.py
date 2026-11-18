"""Релізні артефакти з перевіркою цілісності (у git не потрапляють: dist/ у .gitignore).

    python scripts/build_release.py [--out dist] [--platform manylinux2014_x86_64 ...]
    python scripts/build_release.py --verify dist

Склад: колесо log3 (без залежностей), архів вихідного коду з git, wheelhouse/ із колесами
залежностей для офлайн-встановлення (поточна платформа + вказані --platform), log3_restore.py
(автономний аварійний restore), RELEASE.json (версії, коміт, платформи) і SHA256SUMS.
Збирання потребує мережі для pip download; встановлення з wheelhouse — ні.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNTIME = ["numpy", "cryptography", "zstandard"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def run(*argv: str) -> None:
    print("$", " ".join(argv), flush=True)
    subprocess.run(argv, check=True, cwd=ROOT)


def pins() -> list[str]:
    """Точні версії залежностей із requirements.txt (ті самі, що перевіряє CI)."""
    lines = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    return [x.strip() for x in lines if x.split("==")[0].strip() in RUNTIME]


def build(out: Path, platforms: list[str]) -> dict:
    version = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    if out.exists():
        shutil.rmtree(out)
    (out / "wheelhouse").mkdir(parents=True)
    run(sys.executable, "-m", "pip", "wheel", ".", "--no-deps", "-w", str(out))   # ізольоване збирання (мережа)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    run("git", "archive", "--format=tar.gz", f"--prefix=log3-{version}/", "-o", str(out / f"log3-{version}-src.tar.gz"),
        "HEAD")
    run(sys.executable, "-m", "pip", "download", "-d", str(out / "wheelhouse"), "--only-binary=:all:", *pins())
    for plat in platforms:
        run(sys.executable, "-m", "pip", "download", "-d", str(out / "wheelhouse"), "--only-binary=:all:",
            "--platform", plat, "--python-version", "3.11", "--implementation", "cp", *pins())
    shutil.copyfile(ROOT / "log3" / "vault" / "standalone_restore.py", out / "log3_restore.py")
    info = {"name": "log3", "version": version, "commit": commit, "requirements": pins(),
            "built_with": {"python": platform.python_version(), "os": platform.platform()},
            "platforms": [f"{sys.platform}-{platform.machine()} (поточна)", *platforms],
            "install_offline": "python -m pip install --no-index --find-links wheelhouse log3-*.whl"}
    (out / "RELEASE.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
    (out / "SHA256SUMS").write_text("".join(f"{sha256(p)}  {p.relative_to(out).as_posix()}\n" for p in files),
                                    encoding="utf-8")
    return {**info, "files": len(files)}


def verify(out: Path) -> dict:
    """Перевірка цілісності: кожен файл зі SHA256SUMS існує й має той самий хеш; зайвих файлів нема."""
    listed = {}
    for line in (out / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        listed[name] = digest
    bad = [n for n, d in listed.items() if not (out / n).is_file() or sha256(out / n) != d]
    extra = [p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file() and p.name != "SHA256SUMS"
             and p.relative_to(out).as_posix() not in listed]
    return {"ok": not bad and not extra, "checked": len(listed), "mismatched": bad, "unlisted": extra}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", default=str(ROOT / "dist"))
    p.add_argument("--platform", action="append", default=[], help="додаткова платформа коліс, напр. "
                                                                    "manylinux2014_x86_64 або win_amd64")
    p.add_argument("--verify", metavar="DIR")
    a = p.parse_args(argv)
    out = verify(Path(a.verify)) if a.verify else build(Path(a.out), a.platform)
    print(json.dumps({"ok": out.get("ok", True), **out}, ensure_ascii=False))
    return 0 if out.get("ok", True) else 7


if __name__ == "__main__":
    sys.exit(main())
