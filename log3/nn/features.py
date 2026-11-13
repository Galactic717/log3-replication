"""Ознаки для перцептрона: ознаки самого запису + часові ознаки за вікном історії.

Вікно історії запису i — це записи i-H .. i-1 (лише ПОПЕРЕДНІ, без заглядання вперед)
разом із самим записом i. Початок потоку: беремо стільки попередніх, скільки є.
Тому ознаки запису i не залежать від того, що буде після нього (тест test_no_look_ahead).

Навіщо часові ознаки: клас 2 («перехід до інциденту») визначається структурою в часі —
ростом частоти, нестабільністю category, зміною джерел, серіями споріднених кодів.
Окремий запис класу 2 схожий на штатний, тож без вікна історії його не відрізнити.

Мінімальний набір часових ознак задає team_config.json (temporal_features); до нього
додаємо кілька власних (EXTRA_TEMPORAL). Нормалізація — за статистикою train (Normalizer).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..records import Record


def _slog(x):
    """Знаковий логарифм: стискає metric_value від -2^31 до 2^31 у ±21, зберігаючи знак."""
    return np.sign(x) * np.log1p(np.abs(x))


# ---- часові ознаки: f(arrays, i, lo) -> float, де lo..i — вікно (lo = max(0, i-H)) ----

def _log_dt(a, i, lo):
    """Інтервал до попередньої події (мс, лог-шкала) — найкоротша «історія»."""
    return float(np.log1p(max(a["t"][i] - a["t"][i - 1], 0.0))) if i > 0 else 0.0


def _delta_metric(a, i, lo):
    return float(_slog(a["metric"][i] - a["metric"][i - 1])) if i > 0 else 0.0


def _events_per_window(a, i, lo):
    k = i - lo                      # скільки попередніх подій у вікні
    if k == 0:
        return 0.0
    span_ms = a["t"][i] - a["t"][lo]
    return float(np.log1p(1000.0 * k / (span_ms + 1.0)))   # подій за секунду, у лог-шкалі


def _changes(col):
    def f(a, i, lo):
        w = a[col][lo:i + 1]
        return float(np.count_nonzero(w[1:] != w[:-1])) / max(len(w) - 1, 1)
    return f


def _severity_mean(a, i, lo):
    return float(a["sev"][lo:i + 1].mean()) / 7.0


def _distinct_sources(a, i, lo):
    w = a["src"][lo:i + 1]
    return len(np.unique(w)) / len(w)


def _code_family_run(a, i, lo):
    """Частка вікна з тим самим «сімейством» коду (event_code // 10) — серії споріднених кодів."""
    fam = a["code"][lo:i + 1] // 10
    return float(np.mean(fam == fam[-1]))


def _dt_trend(a, i, lo):
    """log(середній інтервал старшої половини / молодшої): > 0, коли події частішають."""
    dt = np.diff(a["t"][lo:i + 1])
    if len(dt) < 2:
        return 0.0
    h = len(dt) // 2
    return float(np.log((dt[:h].mean() + 1.0) / (dt[h:].mean() + 1.0)))


def _metric_std(a, i, lo):
    return float(_slog(a["metric"][lo:i + 1]).std())


TEMPORAL = {
    "log_dt": _log_dt,
    "delta_metric": _delta_metric,
    "events_per_window": _events_per_window,
    "category_changes": _changes("cat"),
    "source_changes": _changes("src"),
    "severity_mean": _severity_mean,
    "distinct_sources": _distinct_sources,
    "code_family_run": _code_family_run,
    "dt_trend": _dt_trend,
    "metric_std": _metric_std,
}
EXTRA_TEMPORAL = ("log_dt", "distinct_sources", "code_family_run", "dt_trend", "metric_std")

N_CATEGORIES = 16   # category — uint8 0..15 за форматом запису (FORMAT_SPEC §1), не параметр задачі

BASE = (["severity", "metric_slog", "critical", "record_flags", "code_slog"]
        + [f"cat_{c}" for c in range(N_CATEGORIES)])


@dataclass(frozen=True)
class FeatureSpec:
    history: int = 16                          # H — розмір вікна історії (записів)
    temporal: tuple[str, ...] | None = None    # None — усі з TEMPORAL


def spec_for_config(cfg, history: int = 16) -> FeatureSpec:
    """Мінімальний набір із team_config.json (лише відомі імена) + власні додаткові."""
    names = [n for n in cfg.temporal_features if n in TEMPORAL]
    unknown = [n for n in cfg.temporal_features if n not in TEMPORAL]
    if unknown:
        raise ValueError(f"temporal_features з конфігу не реалізовано: {unknown}")
    names += [n for n in EXTRA_TEMPORAL if n not in names]
    return FeatureSpec(history=int(cfg.raw.get("history_window", history)), temporal=tuple(names))


def feature_names(spec: FeatureSpec) -> list[str]:
    temporal = tuple(TEMPORAL) if spec.temporal is None else spec.temporal
    bad = [n for n in temporal if n not in TEMPORAL]
    if bad:
        raise ValueError(f"невідомі часові ознаки: {bad}")
    return BASE + list(temporal)


def build_features(records: list[Record], spec: FeatureSpec) -> np.ndarray:
    names = feature_names(spec)
    n = len(records)
    X = np.zeros((n, len(names)))
    if n == 0:
        return X
    a = {
        "t": np.array([r.timestamp_ms for r in records], dtype=np.float64),
        "src": np.array([r.source_id for r in records]),
        "code": np.array([r.event_code for r in records]),
        "metric": np.array([r.metric_value for r in records], dtype=np.float64),
        "cat": np.array([r.category for r in records]),
        "sev": np.array([r.severity for r in records], dtype=np.float64),
    }
    X[:, 0] = a["sev"] / 7.0
    X[:, 1] = _slog(a["metric"])
    X[:, 2] = [r.critical for r in records]
    X[:, 3] = [r.record_flags for r in records]
    X[:, 4] = np.log1p(a["code"])
    X[np.arange(n), 5 + a["cat"]] = 1.0
    temporal = names[len(BASE):]
    H = spec.history
    for i in range(n):
        lo = max(0, i - H)
        for j, name in enumerate(temporal):
            X[i, len(BASE) + j] = TEMPORAL[name](a, i, lo)
    return X


@dataclass(frozen=True)
class Normalizer:
    """z-нормалізація за статистикою train; ті самі mean/std — для valid і для classify."""
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray) -> "Normalizer":
        std = X.std(axis=0)
        return cls(X.mean(axis=0), np.where(std > 1e-12, std, 1.0))

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std
