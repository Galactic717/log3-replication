"""Дані для перцептрона: train/valid з materials/ або dev-набір із мітками.

Справжні дані — materials/train.csv і materials/valid.csv (колонка мітки: label/class/…,
див. records.LABEL_COLUMNS). Поки пакета викладача нема, load_split() повертає
dev-набір: синтетичний журнал із мітками 0/1/2, побудований за описом класів у
завданні. Він ПОТРІБЕН лише для налагодження; справжні CSV мають пріоритет автоматично.

Як влаштовано dev-набір (щоб було видно, чому без вікна історії клас 2 не розпізнати):
  * клас 0 — штатний фон: джерела 1..3, коди 100..102/200, категорії 0..2, рідкі події;
  * клас 1 — аномалія: поодинока подія з високою severity і різким стрибком metric_value;
  * клас 2 — перехід до інциденту: відрізок, де частота подій росте, category
    скаче, з'являються нові джерела, йде серія споріднених кодів (5xx). Severity і
    metric окремого запису тут майже як у фону, тому поодинокий запис нічого не каже;
  * за переходом іде короткий інцидент (висока severity) — теж клас 1.
  * critical — окреме правило (важливість для зберігання), а не клас.
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from ..config import ROOT
from ..records import FIELD_NAMES, Record, load_records_csv

TRAIN_CSV = ROOT / "materials" / "train.csv"
VALID_CSV = ROOT / "materials" / "valid.csv"
DEV_DATA_DIR = ROOT / "dev_data"          # експорт dev-набору — НЕ materials/

# Розміри й seeds dev-набору (лише для налагодження, не параметри задачі).
DEV_TRAIN = (6000, 101)
DEV_VALID = (3000, 202)


def dev_labeled_session(n: int, seed: int) -> tuple[list[Record], list[int]]:
    rng = random.Random(seed)
    recs: list[Record] = []
    labels: list[int] = []
    t = 1_000_000
    phase, left, step = "normal", 0, 0
    base_code = rng.choice((500, 510, 520))
    while len(recs) < n:
        if phase == "normal" and rng.random() < 0.012:
            phase, left, step = "transition", rng.randint(12, 30), 0
            base_code = rng.choice((500, 510, 520))
        if phase == "transition" and left == 0:
            phase, left = "incident", rng.randint(3, 8)
        if phase == "incident" and left == 0:
            phase = "normal"

        if phase == "normal":
            t += rng.randint(150, 500)
            if rng.random() < 0.04:                               # аномалія, клас 1
                sev = rng.randint(4, 7)
                metric = rng.choice((-1, 1)) * rng.randint(1500, 50000)
                rec = (rng.choice((1, 2, 3, 5)), rng.choice((101, 300, 404, 501)), metric,
                       rng.randint(0, 15), sev)
                label = 1
            else:                                                 # фон, клас 0
                # фон іноді має «чужі» коди, джерела й категорії — щоб клас 2 не впізнавався
                # за одним полем запису, лише за їх згущенням у часі
                code = rng.choice((100, 101, 102, 200)) if rng.random() < 0.75 else base_code + rng.randint(0, 4)
                src = rng.choice((1, 2, 3)) if rng.random() < 0.75 else rng.choice((5, 8, 13, 21))
                cat = rng.choice((0, 1, 2)) if rng.random() < 0.75 else rng.randint(0, 15)
                rec = (src, code, rng.randint(40, 180), cat, rng.choice((0, 0, 1, 1, 2, 3, 4)))
                label = 0
        elif phase == "transition":                               # клас 2
            step += 1
            left -= 1
            t += max(20, int(rng.randint(150, 500) / (1 + 0.25 * step)))   # частота росте
            code = base_code + rng.randint(0, 4) if rng.random() < 0.5 else rng.choice((100, 101, 102, 200))
            src = rng.choice((5, 8, 13, 21)) if rng.random() < 0.5 else rng.choice((1, 2, 3))
            cat = rng.randint(0, 15) if rng.random() < 0.5 else rng.choice((0, 1, 2))
            rec = (src, code, rng.randint(40, 180), cat, rng.choice((0, 1, 1, 2, 3, 4)))
            label = 2
        else:                                                     # інцидент, клас 1
            left -= 1
            t += rng.randint(5, 60)
            rec = (rng.choice((1, 5, 8, 13)), base_code + rng.randint(0, 4),
                   rng.randint(1500, 20000), rng.randint(0, 15), rng.randint(4, 7))
            label = 1
        src, code, metric, cat, sev = rec
        critical = int(sev >= 6 or (code >= 500 and rng.random() < 0.15))
        recs.append(Record(t % 2**32, src, code, metric, cat, sev, critical, rng.choice((0, 0, 0, 1))))
        if rng.random() < 0.02:                                   # шум розмітки, як у реальних даних
            label = rng.randrange(3)
        labels.append(label)
    return recs, labels


def load_split(name: str) -> tuple[list[Record], list[int], str]:
    """('train'|'valid') -> (записи, мітки, джерело 'materials' | 'dev')."""
    path: Path = {"train": TRAIN_CSV, "valid": VALID_CSV}[name]
    if path.exists():
        recs, labels = load_records_csv(path)
        if labels is None:
            raise ValueError(f"{path}: нема колонки мітки класу")
        return recs, labels, "materials"
    n, seed = DEV_TRAIN if name == "train" else DEV_VALID
    recs, labels = dev_labeled_session(n, seed)
    return recs, labels, "dev"


def export_dev_csv(out_dir: Path = DEV_DATA_DIR, *, train=DEV_TRAIN, valid=DEV_VALID) -> dict[str, Path]:
    """Записати dev-набір у CSV того ж формату, що й materials/train.csv (поля Record + label).

    Навмисно в окрему теку dev_data/, а не в materials/: файли в materials/ код вважає
    даними викладача (load_split -> 'materials', test_c1_09). Ці CSV — лише зразок формату
    і dev-дані для налагодження; коли прийде пакет викладача, його кладуть у materials/.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, (n, seed) in (("train", train), ("valid", valid)):
        recs, labels = dev_labeled_session(n, seed)
        p = out_dir / f"dev_{name}.csv"
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([*FIELD_NAMES, "label"])
            for r, y in zip(recs, labels):
                w.writerow([*(getattr(r, k) for k in FIELD_NAMES), y])
        paths[name] = p
    return paths


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Експорт dev-набору з мітками в CSV (не дані викладача).")
    ap.add_argument("--out", type=Path, default=DEV_DATA_DIR)
    for name, path in export_dev_csv(ap.parse_args().out).items():
        print(f"{name}: {path}")
