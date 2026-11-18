"""Синтетичний корпус файлів для вимірювань LOG3 Vault (не клієнтські дані).

    make_corpus(root, seed, scale_mib)  набір файлів різних типів і стисливості
    mutate(root, seed, fraction)         частина файлів змінюється, як між робочими днями
    mass_rewrite(root, seed)             масове переписування майже всіх файлів випадковими байтами

Файли різних типів лежать упереміш у «проєктах», як у робочих каталогах. Типи підібрано так,
щоб жодна проста ознака не вирішувала задачу сама: є стиснені контейнери
(docx/xlsx/zip — це zip), «PDF» із текстовими й стисненими потоками, логи, CSV, JSON, код,
розріджені образи, двійкові файли, а також файли з оманливим розширенням (.txt із випадковими
байтами, .dat із текстом). Генерація детермінована за seed.
"""
from __future__ import annotations

import io
import json
import os
import random
import zipfile
from pathlib import Path

MIB = 1 << 20
WORDS = ("звіт договір рахунок поставка оплата склад клієнт менеджер проєкт бюджет квартал акт "
         "invoice order total amount customer status shipped pending approved warehouse").split()


def _text(rng: random.Random, n: int) -> bytes:
    out, size = [], 0
    while size < n:
        line = " ".join(rng.choice(WORDS) for _ in range(rng.randint(4, 14))) + f" {rng.randint(0, 99999)}\n"
        out.append(line)
        size += len(line.encode())
    return "".join(out).encode()[:n]


def _log(rng: random.Random, n: int) -> bytes:
    out, size, t = [], 0, 1_763_000_000
    while size < n:
        t += rng.randint(0, 3)
        line = (f"2026-11-{1 + t % 28:02d}T{t % 24:02d}:{t % 60:02d}:{t * 7 % 60:02d}Z "
                f"{rng.choice(('INFO', 'INFO', 'INFO', 'WARN', 'ERROR'))} svc={rng.choice(('api', 'db', 'auth'))} "
                f"req={rng.getrandbits(32):08x} latency_ms={rng.randint(1, 900)}\n")
        out.append(line)
        size += len(line)
    return "".join(out).encode()[:n]


def _csv(rng: random.Random, n: int) -> bytes:
    rows, size = ["id,date,amount,currency,status\n"], 0
    while size < n:
        r = f"{rng.randint(1, 10**7)},2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}," \
            f"{rng.uniform(1, 10000):.2f},{rng.choice(('UAH', 'EUR', 'USD'))},{rng.choice(('ok', 'void'))}\n"
        rows.append(r)
        size += len(r)
    return "".join(rows).encode()[:n]


def _json(rng: random.Random, n: int) -> bytes:
    items, size = [], 0
    while size < n:
        it = {"id": rng.randint(1, 10**9), "name": rng.choice(WORDS), "tags": rng.sample(WORDS, 3),
              "price": round(rng.uniform(0, 500), 2), "active": rng.random() < 0.5}
        items.append(it)
        size += 90
    return json.dumps(items, ensure_ascii=False, indent=1).encode()[:n]


def _code(rng: random.Random, n: int) -> bytes:
    out, size = [], 0
    while size < n:
        name = rng.choice(WORDS[12:])
        block = (f"def {name}_{rng.randint(0, 999)}(x, y):\n    if x > {rng.randint(0, 99)}:\n"
                 f"        return x * y + {rng.randint(0, 9)}\n    return {name!r}\n\n")
        out.append(block)
        size += len(block)
    return "".join(out).encode()[:n]


def _random(rng: random.Random, n: int) -> bytes:
    return rng.randbytes(n)


def _zip(rng: random.Random, n: int, inner: str) -> bytes:
    """Справжній zip-контейнер (як docx/xlsx): deflate-потоки майже не стискаються вдруге."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        def put(name: str, data: bytes) -> None:          # фіксований час: детермінований архів
            z.writestr(zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0)), data, zipfile.ZIP_DEFLATED)
        put("[Content_Types].xml", b'<?xml version="1.0"?><Types/>')
        k = 0
        while buf.tell() < n:
            put(f"{inner}/part{k}.xml", _text(rng, 64 * 1024) + rng.randbytes(16 * 1024))
            k += 1
    return buf.getvalue()


def _pdf(rng: random.Random, n: int) -> bytes:
    """Чергування відкритих текстових потоків і «стиснених» (випадкових) потоків."""
    parts, size = [b"%PDF-1.7\n"], 9
    while size < n:
        body = _text(rng, 24 * 1024) if rng.random() < 0.4 else rng.randbytes(48 * 1024)
        obj = b"obj\n<< /Length %d >>\nstream\n" % len(body) + body + b"\nendstream\nendobj\n"
        parts.append(obj)
        size += len(obj)
    return b"".join(parts)[:n]


def _sparse(rng: random.Random, n: int) -> bytes:
    data = bytearray(n)
    for _ in range(max(1, n // (256 * 1024))):
        pos = rng.randrange(0, max(1, n - 4096))
        data[pos:pos + 4096] = rng.randbytes(4096)
    return bytes(data)


def _dbpages(rng: random.Random, n: int) -> bytes:
    page, out = 4096, bytearray()
    while len(out) < n:
        rec = _csv(rng, 1500) + bytes(page - 1500 - 96) + rng.randbytes(96)
        out += rec
    return bytes(out[:n])


#: (розширення, генератор, частка обсягу, діапазон розміру файла в КіБ)
KINDS = (
    ("log", _log, 0.14, (64, 2048)), ("txt", _text, 0.08, (8, 512)), ("csv", _csv, 0.10, (32, 3072)),
    ("json", _json, 0.06, (16, 1024)), ("py", _code, 0.04, (4, 256)), ("bin", _random, 0.08, (64, 2048)),
    ("jpg", lambda r, n: b"\xff\xd8\xff\xe0" + r.randbytes(n), 0.08, (64, 2048)),
    ("docx", lambda r, n: _zip(r, n, "word"), 0.08, (64, 1536)),
    ("xlsx", lambda r, n: _zip(r, n, "xl"), 0.05, (64, 1024)), ("pdf", _pdf, 0.12, (128, 3072)),
    ("img", _sparse, 0.04, (512, 4096)), ("db", _dbpages, 0.05, (256, 3072)),
    ("txt", _random, 0.02, (32, 512)),        # оманливе розширення: випадкові байти в .txt
    ("dat", _log, 0.02, (32, 512)),           # оманливе розширення: текст у .dat
)


def _pick(rng: random.Random):
    r, acc = rng.random() * sum(k[2] for k in KINDS), 0.0
    for kind in KINDS:
        acc += kind[2]
        if r <= acc:
            return kind
    return KINDS[-1]


def _new_file(rng: random.Random, root: Path, projects: int, max_file_kib: int, tag: str) -> int:
    ext, gen, _, (lo, hi) = _pick(rng)
    n = rng.randint(min(lo, max_file_kib), min(hi, max_file_kib)) * 1024
    sub = root / f"project-{rng.randrange(projects):03d}"
    sub.mkdir(parents=True, exist_ok=True)
    (sub / f"{tag}.{ext}").write_bytes(gen(rng, n))
    return n


def make_corpus(root: Path, seed: int = 1, scale_mib: float = 32.0, max_file_kib: int = 4096) -> dict:
    """Файли різних типів упереміш по «проєктах» (як робочі каталоги замовника)."""
    rng = random.Random(seed)
    root.mkdir(parents=True, exist_ok=True)
    budget, total, files = int(scale_mib * MIB), 0, 0
    projects = max(4, int(scale_mib * MIB / (max(64, max_file_kib // 2) * 1024) / 12))
    while total < budget:
        total += _new_file(rng, root, projects, max_file_kib, f"f{files:05d}")
        files += 1
    return {"files": files, "bytes": total, "seed": seed, "projects": projects}


def mutate(root: Path, seed: int, fraction: float = 0.15, new_files: float = 0.05) -> int:
    """Змінює частку файлів (дописування в текстові, нові байти в двійкові) і додає нові файли."""
    rng = random.Random(seed)
    paths = sorted(p for p in root.rglob("*") if p.is_file())
    changed = 0
    for p in rng.sample(paths, max(1, int(len(paths) * fraction))):
        ext = p.suffix[1:]
        with open(p, "ab") as f:
            f.write(_log(rng, rng.randint(1, 64) * 1024) if ext in ("log", "csv", "txt", "json", "py", "dat")
                    else rng.randbytes(rng.randint(4, 64) * 1024))
        changed += 1
    projects = len([d for d in root.iterdir() if d.is_dir()]) or 1
    for k in range(max(1, int(len(paths) * new_files))):
        _new_file(rng, root, projects, 1024, f"new-{seed}-{k}")
        changed += 1
    return changed


def mass_rewrite(root: Path, seed: int) -> int:
    """Масове переписування: майже кожен файл замінюється випадковими байтами того самого розміру."""
    rng = random.Random(seed)
    n = 0
    for p in sorted(root.rglob("*")):
        if p.is_file() and rng.random() < 0.95:
            p.write_bytes(rng.randbytes(os.path.getsize(p)))
            n += 1
    return n
