"""Ознаки для перцептрона: ознаки самого запису + часові ознаки за вікном історії.

Вікно історії запису i — лише ПОПЕРЕДНІ записи (i-H .. i) і, для часового вікна,
події за останні time_window_ms. Ознаки запису i не залежать від майбутніх записів
(test_no_look_ahead), тож класифікацію можна робити в потоці.

Клас 2 («перехід до інциденту») визначається часовою структурою: частішають події,
скаче category, змінюються джерела, ідуть серії сусідніх кодів. Окремий запис класу 2
схожий на штатний, тому без вікна історії його не відрізнити (абляція в train.py).

Мінімальний набір часових ознак задає team_config.json («мінімальні_часові_ознаки») —
назви там описові, тому TEMPORAL використовує саме їх як ключі. До нього додаємо кілька
власних (EXTRA_TEMPORAL). Усе рахується векторно (кумулятивні суми), без циклу по записах.
Немонотонний timestamp_ms (несортовані записи, переповнення uint32) не дає NaN:
інтервали обрізаються знизу нулем, а для часового вікна береться біжучий максимум часу.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..records import Record

N_CATEGORIES = 16   # category — uint8 0..15 за форматом запису (FORMAT_SPEC §1)


def _slog(x):
    """Знаковий логарифм: стискає metric_value від -2^31 до 2^31 у ±21, зберігаючи знак."""
    return np.sign(x) * np.log1p(np.abs(x))


def _rolling_sum(x: np.ndarray, h: int) -> np.ndarray:
    """s[i] = x[max(0, i-h+1)] + ... + x[i] — сума за останні h елементів (лише назад)."""
    c = np.cumsum(x, axis=0)
    out = c.copy()
    out[h:] -= c[:-h]
    return out


def _changed(a: np.ndarray) -> np.ndarray:
    """Чи змінилось значення в парі (i-1, i); для i = 0 пари нема."""
    return np.concatenate([[False], a[1:] != a[:-1]])


class _Arrays:
    def __init__(self, records: list[Record], spec: FeatureSpec):
        self.n = len(records)
        self.spec = spec
        self.t = np.maximum.accumulate(np.array([r.timestamp_ms for r in records], dtype=np.float64))
        self.src = np.array([r.source_id for r in records])
        self.code = np.array([r.event_code for r in records], dtype=np.int64)
        self.metric = np.array([r.metric_value for r in records], dtype=np.float64)
        self.cat = np.array([r.category for r in records])
        self.sev = np.array([r.severity for r in records], dtype=np.float64)

    def in_window(self) -> np.ndarray:
        """Кількість подій у вікні історії (включно з поточною): min(i+1, H+1)."""
        return np.minimum(np.arange(self.n) + 1, self.spec.history + 1).astype(np.float64)

    def pair_share(self, flags: np.ndarray) -> np.ndarray:
        """Частка пар (j-1, j) у вікні з ознакою flags: H пар назад."""
        pairs = np.maximum(self.in_window() - 1, 1)
        return _rolling_sum(flags.astype(np.float64), self.spec.history) / pairs


def _log_dt(a: _Arrays):
    dt = np.diff(a.t, prepend=a.t[:1])                  # біжучий максимум -> dt ≥ 0
    return np.log1p(dt)


def _events_in_time_window(a: _Arrays):
    lo = np.searchsorted(a.t, a.t - a.spec.time_window_ms, side="left")
    return np.log1p(np.arange(a.n) - lo + 1)


def _category_changes(a: _Arrays):
    return a.pair_share(_changed(a.cat))


def _distinct_sources(a: _Arrays):
    _, inv = np.unique(a.src, return_inverse=True)
    onehot = np.zeros((a.n, inv.max() + 1))
    onehot[np.arange(a.n), inv] = 1.0
    present = _rolling_sum(onehot, a.spec.history + 1) > 0
    return present.sum(axis=1) / a.in_window()


def _adjacent_codes(a: _Arrays):
    adj = np.concatenate([[False], np.abs(np.diff(a.code)) == 1])
    return a.pair_share(adj)


def _metric_change_same_source(a: _Arrays):
    out = np.zeros(a.n)
    for s in np.unique(a.src):                          # ~12 джерел — цикл по джерелах, не по записах
        idx = np.flatnonzero(a.src == s)
        out[idx[1:]] = _slog(np.diff(a.metric[idx]))
    return out


def _severity(a: _Arrays):
    return a.sev / 7.0


def _severity_mean(a: _Arrays):
    return _rolling_sum(a.sev, a.spec.history + 1) / a.in_window() / 7.0


def _dt_mean_short(a: _Arrays):
    """Середній інтервал за 4 останні пари (лог): коротка шкала частішання подій."""
    dt = np.diff(a.t, prepend=a.t[:1])
    k = np.minimum(np.arange(a.n), 4)
    return np.log1p(_rolling_sum(dt, 4) / np.maximum(k, 1))


def _adjacent_codes_short(a: _Arrays):
    adj = np.concatenate([[False], np.abs(np.diff(a.code)) == 1]).astype(np.float64)
    return _rolling_sum(adj, 4) / np.maximum(np.minimum(np.arange(a.n), 4), 1)


TEMPORAL = {
    "інтервал до попередньої події": _log_dt,
    "кількість подій у ковзному вікні часу": _events_in_time_window,
    "кількість змін category у вікні": _category_changes,
    "кількість різних source_id у вікні": _distinct_sources,
    "частка пар подій із сусідніми event_code у вікні": _adjacent_codes,
    "зміна metric_value відносно попередньої події того самого джерела": _metric_change_same_source,
    "severity": _severity,
    # власні додаткові
    "середня severity у вікні": _severity_mean,
    "середній інтервал за 4 пари": _dt_mean_short,
    "частка сусідніх event_code за 4 пари": _adjacent_codes_short,
}
EXTRA_TEMPORAL = ("середня severity у вікні", "середній інтервал за 4 пари", "частка сусідніх event_code за 4 пари")

BASE = (["metric_slog", "critical", "record_flags", "code_log"]
        + [f"cat_{c}" for c in range(N_CATEGORIES)])


@dataclass(frozen=True)
class FeatureSpec:
    history: int = 16                          # H — вікно історії (записів)
    time_window_ms: float = 1000.0             # ковзне вікно часу для частоти подій
    temporal: tuple[str, ...] = tuple(TEMPORAL)


def spec_for_config(cfg, history: int = 16, time_window_ms: float = 1000.0) -> FeatureSpec:
    """Мінімальний набір із team_config.json + власні додаткові. Невідома назва — помилка:
    краще зупинити навчання, ніж мовчки пропустити обов'язкову ознаку."""
    unknown = [n for n in cfg.temporal_features if n not in TEMPORAL]
    if unknown:
        raise ValueError(f"часові ознаки з team_config.json не реалізовано: {unknown}")
    names = list(cfg.temporal_features) + [n for n in EXTRA_TEMPORAL if n not in cfg.temporal_features]
    return FeatureSpec(history, time_window_ms, tuple(names))


def feature_names(spec: FeatureSpec) -> list[str]:
    bad = [n for n in spec.temporal if n not in TEMPORAL]
    if bad:
        raise ValueError(f"невідомі часові ознаки: {bad}")
    return BASE + list(spec.temporal)


def build_features(records: list[Record], spec: FeatureSpec) -> np.ndarray:
    names = feature_names(spec)
    n = len(records)
    X = np.zeros((n, len(names)))
    if n == 0:
        return X
    a = _Arrays(records, spec)
    X[:, 0] = _slog(a.metric)
    X[:, 1] = [r.critical for r in records]
    X[:, 2] = [r.record_flags for r in records]
    X[:, 3] = np.log1p(a.code)
    X[np.arange(n), 4 + a.cat] = 1.0
    for j, name in enumerate(spec.temporal):
        X[:, len(BASE) + j] = TEMPORAL[name](a)
    return X


@dataclass(frozen=True)
class Normalizer:
    """z-нормалізація за статистикою train; ті самі mean/std — для valid і для classify."""
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray) -> Normalizer:
        std = X.std(axis=0)
        return cls(X.mean(axis=0), np.where(std > 1e-12, std, 1.0))

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std
