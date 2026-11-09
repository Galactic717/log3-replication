import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
MATERIALS = ROOT / "materials"
# replication.py викладача імпортується як модуль із materials/
for p in (ROOT, MATERIALS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from log3.records import Record  # noqa: E402

PASSWORD = "test-password"
SALT = bytes(range(16))


@pytest.fixture
def reference_dir() -> Path:
    d = MATERIALS / "reference"
    if not d.is_dir():
        pytest.skip("materials/reference/ нема — еталонні пакети викладача не покладено локально")
    return d


def make_records(n: int, seed: int = 0) -> list[Record]:
    """Правдоподібні записи: зростаючий час, кілька джерел, повторювані коди."""
    rng = random.Random(seed)
    t = 1_000_000
    out = []
    for _ in range(n):
        t += rng.randint(1, 500)
        out.append(Record(
            timestamp_ms=t,
            source_id=rng.choice((1, 2, 3, 17, 4095)),
            event_code=rng.choice((100, 101, 200, 404, 500)),
            metric_value=rng.randint(-50, 5000),
            category=rng.randint(0, 15),
            severity=rng.randint(0, 7),
            critical=int(rng.random() < 0.2),
            record_flags=rng.randint(0, 3),
        ))
    return out
