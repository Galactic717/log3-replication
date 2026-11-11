"""classify_records — класи подій перцептроном (контракт CONTRACTS.md §7).

0 — штатна, 1 — аномальна, 2 — перехід до інциденту (кількість класів — з моделі).
Модель — log3/nn/model.npz (ваги, нормалізація train, вікно історії, список ознак);
шлях можна перевизначити змінною LOG3_MODEL. Перенавчити: python -m log3.nn.train.

Записи йдуть як потік: ознаки запису i рахуються з нього самого і попередніх записів
(без заглядання вперед), тому результат детермінований і не залежить від майбутнього.
Якщо файлу моделі нема — заглушка з CONTRACTS («усі класу 0») і попередження.
"""
from __future__ import annotations

import json
import os
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..records import Record
from .features import FeatureSpec, Normalizer, build_features
from .mlp import MLP

DEFAULT_MODEL = Path(__file__).resolve().parent / "model.npz"


def _model_path() -> Path:
    return Path(os.environ.get("LOG3_MODEL", DEFAULT_MODEL))


@lru_cache(maxsize=4)
def _load(path: str, mtime: float):
    with np.load(path) as d:
        net = MLP.from_arrays(d)
        norm = Normalizer(d["norm_mean"], d["norm_std"])
        meta = json.loads(str(d["meta"]))
    spec = FeatureSpec(history=int(meta["history"]), temporal=tuple(meta["temporal"]))
    return net, norm, spec, meta


def _model():
    p = _model_path()
    if not p.exists():
        return None
    return _load(str(p), p.stat().st_mtime)


def model_meta() -> dict:
    m = _model()
    return {} if m is None else dict(m[3])


def model_n_classes() -> int:
    m = _model()
    return 1 if m is None else m[0].n_out


def classify_records(records: list[Record]) -> list[int]:
    if not records:
        return []
    m = _model()
    if m is None:
        print(f"[log3] моделі {_model_path()} нема — заглушка: усі події класу 0", file=sys.stderr)
        return [0] * len(records)
    net, norm, spec, _ = m
    X = norm.transform(build_features(records, spec))
    return [int(c) for c in net.predict(X)]
