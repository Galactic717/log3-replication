"""Політики вибору дії, прогноз витрат і безпечні артефакти моделей.

Від простого до складного (складніше приймається лише за виміряної користі, див. evaluate):
  fixed      одна дія для всіх блоків (Store або фіксований загальний кодек)
  baseline   правило етапу 1 на ознаках: стиснене/високоентропійне -> Store, решта -> кодек конфігурації
  costtable  проста статистика: середні байти й CPU кожної дії в комірках (клас типу, сигнатура,
             виміряна стисливість, ентропія) з відступом до грубіших комірок за малої підтримки
  ridge      модель вартості: лінійна регресія частки байтів і log CPU/байт для кожної дії
  mlp        власний перцептрон log3.nn.mlp, класифікує найдешевшу дію (перевірюваний варіант)
costtable і ridge прогнозують витрати простору й CPU кожної дії, вибір — мінімум вартості.

Артефакт — каталог <repo>/learn/models/<model_id>/ з model.json і arrays.npz. Завантаження не
виконує коду: JSON із жорсткою перевіркою полів і numpy.load(allow_pickle=False) із перевіркою
SHA-256, назв, форм і скінченності масивів. Будь-яка невідповідність (схема ознак, набір дій,
формат, пошкодження) -> ModelError, і тракт працює на базовій політиці.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import time
import uuid
from pathlib import Path

import numpy as np

from ...nn.mlp import MLP, Adam
from ..fsutil import fsync_dir, write_durable
from . import features as F
from .actions import CostModel

MODEL_FORMAT = "log3-advisor-model"
MODEL_VERSION = 1
KINDS = ("fixed", "baseline", "costtable", "ridge", "mlp")
MAX_JSON = 4 << 20
MAX_ARRAYS = 64 << 20
PACKED_TYPES = ("office_packed", "archive", "image_packed", "media")
_TYPE_IDX = [F.index(f"type_{t}") for t in F.TYPE_NAMES]
_MAGIC_IDX = [F.index(f"magic_{m}") for m in F.MAGIC_NAMES]


class ModelError(Exception):
    """Модель відсутня, пошкоджена або несумісна: працює базова політика."""


def raw_len(X: np.ndarray) -> np.ndarray:
    return np.maximum(1.0, np.exp2(X[:, F.index("log2_chunk_len")].astype(np.float64) * 26) - 1)


def cost_matrix(payload: np.ndarray, cpu_ns: np.ndarray, cost: CostModel) -> np.ndarray:
    return cost.cost(payload, cpu_ns)


class Policy:
    kind = ""

    def __init__(self, actions: list[str], cost: CostModel, params: dict | None = None):
        self.actions, self.cost, self.params = list(actions), cost, dict(params or {})

    def choose(self, X: np.ndarray) -> np.ndarray:
        ratio, cpu_pb = self.predict(X)
        raw = raw_len(X)[:, None]
        return np.argmin(self.cost.cost(ratio * raw, cpu_pb * raw), axis=1)

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(частка байтів payload/вхід, CPU нс на байт входу) для кожної дії: форма (n, k)."""
        raise NotImplementedError(f"{self.kind}: прогнозу витрат нема")

    def arrays(self) -> dict[str, np.ndarray]:
        return {}

    def load_arrays(self, arrays: dict[str, np.ndarray]) -> None:
        return None


class FixedPolicy(Policy):
    kind = "fixed"

    def choose(self, X):
        return np.full(len(X), self.actions.index(self.params["action"]), dtype=int)


class BaselinePolicy(Policy):
    """Правило етапу 1 (advisor.BaselineAdvisor), виражене через ознаки."""
    kind = "baseline"

    def choose(self, X):
        packed = X[:, [F.index(f"type_{t}") for t in PACKED_TYPES]].max(axis=1) > 0
        store = packed | (X[:, F.index("entropy_head")] * 8 > 7.5)
        return np.where(store, self.actions.index("store:0"), self.actions.index(self.params["default_action"]))


def _cells(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t = X[:, _TYPE_IDX].argmax(axis=1)
    m = X[:, _MAGIC_IDX].argmax(axis=1)
    p = np.digitize(X[:, F.index("probe_ratio")], (0.3, 0.5, 0.7, 0.9))
    e = np.digitize(X[:, F.index("entropy_sample")] * 8, (5.0, 7.0))
    coarse = p * 3 + e
    return ((t * len(_MAGIC_IDX) + m) * 15 + coarse).astype(np.int64), coarse.astype(np.int64)


class CostTablePolicy(Policy):
    kind = "costtable"

    def fit(self, X, payload, cpu_ns, raw):
        fine, coarse = _cells(X)
        ratio, cpu_pb = payload / raw[:, None], cpu_ns / raw[:, None]
        self.t = {}
        for name, keys in (("fine", fine), ("coarse", coarse)):
            uniq, inv = np.unique(keys, return_inverse=True)
            cnt = np.bincount(inv, minlength=len(uniq)).astype(float)
            r = np.stack([np.bincount(inv, ratio[:, a], len(uniq)) for a in range(ratio.shape[1])], 1) / cnt[:, None]
            c = np.stack([np.bincount(inv, cpu_pb[:, a], len(uniq)) for a in range(ratio.shape[1])], 1) / cnt[:, None]
            self.t[name] = (uniq, cnt, r, c)
        self.t["global"] = (np.zeros(1, np.int64), np.array([len(X)], float), ratio.mean(0)[None], cpu_pb.mean(0)[None])
        return self

    def predict(self, X):
        fine, coarse = _cells(X)
        min_count = self.params.get("min_count", 8)
        _, _, gr, gc = self.t["global"]
        ratio, cpu = np.repeat(gr, len(X), 0), np.repeat(gc, len(X), 0)
        for name, keys in (("coarse", coarse), ("fine", fine)):        # тонша комірка перекриває грубішу
            uniq, cnt, r, c = self.t[name]
            pos = np.clip(np.searchsorted(uniq, keys), 0, len(uniq) - 1)
            ok = (uniq[pos] == keys) & (cnt[pos] >= min_count)
            ratio[ok], cpu[ok] = r[pos[ok]], c[pos[ok]]
        return np.clip(ratio, 0, 1.0), np.clip(cpu, 0, 1e4)

    def arrays(self):
        out = {}
        for name in ("fine", "coarse", "global"):
            for part, arr in zip(("keys", "count", "ratio", "cpu"), self.t[name]):
                out[f"{name}_{part}"] = arr
        return out

    def load_arrays(self, a):
        self.t = {n: (a[f"{n}_keys"].astype(np.int64), a[f"{n}_count"], a[f"{n}_ratio"], a[f"{n}_cpu"])
                  for n in ("fine", "coarse", "global")}


class _Scaler:
    @staticmethod
    def fit(X):
        mu, sd = X.mean(0), X.std(0)
        return mu, np.where(sd > 1e-9, sd, 1.0)


class RidgePolicy(Policy):
    kind = "ridge"

    def fit(self, X, payload, cpu_ns, raw):
        l2 = self.params.get("l2", 1.0)
        self.mu, self.sd = _Scaler.fit(X.astype(np.float64))
        A = np.hstack([np.ones((len(X), 1)), (X - self.mu) / self.sd])
        reg = l2 * np.eye(A.shape[1])
        reg[0, 0] = 0.0
        Y = np.hstack([payload / raw[:, None], np.log(cpu_ns / raw[:, None] + 1e-3)])
        self.W = np.linalg.solve(A.T @ A + reg, A.T @ Y)
        return self

    def predict(self, X):
        A = np.hstack([np.ones((len(X), 1)), (X - self.mu) / self.sd])
        Y = A @ self.W
        k = len(self.actions)
        return np.clip(Y[:, :k], 0, 1.0), np.clip(np.exp(np.clip(Y[:, k:], -20, 10)) - 1e-3, 0, 1e4)

    def arrays(self):
        return {"mu": self.mu, "sd": self.sd, "W": self.W}

    def load_arrays(self, a):
        self.mu, self.sd, self.W = a["mu"], a["sd"], a["W"]


class MLPPolicy(Policy):
    kind = "mlp"

    def fit(self, X, best, Xval, val_costs, *, seed: int = 0):
        p = self.params
        self.mu, self.sd = _Scaler.fit(X.astype(np.float64))
        Xs, Xv = (X - self.mu) / self.sd, (Xval - self.mu) / self.sd
        k = len(self.actions)
        self.net = MLP(X.shape[1], tuple(p.get("hidden", (32,))), k, seed=seed, l2=p.get("l2", 1e-4))
        opt, rng = Adam(lr=p.get("lr", 3e-3)), np.random.default_rng(seed)
        counts = np.bincount(best, minlength=k).astype(float)
        cw = np.where(counts > 0, len(best) / (k * np.maximum(counts, 1)), 0.0)
        best_cost, best_params = np.inf, [w.copy() for w in self.net.params]
        for _ in range(p.get("epochs", 40)):
            order = rng.permutation(len(best))
            for lo in range(0, len(order), 128):
                idx = order[lo:lo + 128]
                _, grads = self.net.loss_and_grads(Xs[idx], best[idx], cw)
                opt.step(self.net.params, grads)
            if len(Xv):
                c = val_costs[np.arange(len(Xv)), self.net.predict(Xv)].mean()
                if c < best_cost:                          # ранню зупинку обирає вартість на validation
                    best_cost, best_params = c, [w.copy() for w in self.net.params]
        if len(Xv):
            self.net.params = best_params
        return self

    def choose(self, X):
        return self.net.predict((X - self.mu) / self.sd)

    def arrays(self):
        return {"mu": self.mu, "sd": self.sd, **{f"net_{k}": v for k, v in self.net.to_arrays().items()}}

    def load_arrays(self, a):
        self.mu, self.sd = a["mu"], a["sd"]
        self.net = MLP.from_arrays({k[4:]: v for k, v in a.items() if k.startswith("net_")})


CLASSES = {c.kind: c for c in (FixedPolicy, BaselinePolicy, CostTablePolicy, RidgePolicy, MLPPolicy)}


# ---- артефакти ------------------------------------------------------------------------------------
def new_model_id(kind: str) -> str:
    return f"{kind}-{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}-{uuid.uuid4().hex[:6]}"


def save_artifact(models_dir: Path, policy: Policy, meta: dict) -> str:
    """Атомарно: тимчасовий каталог -> fsync файлів -> rename -> fsync батьківського каталогу."""
    model_id = meta.get("model_id") or new_model_id(policy.kind)
    buf = io.BytesIO()
    np.savez(buf, **{k: np.asarray(v) for k, v in policy.arrays().items()})
    blob = buf.getvalue()
    doc = {"format": MODEL_FORMAT, "format_version": MODEL_VERSION, "model_id": model_id, "kind": policy.kind,
           "feature_schema": F.schema_id(), "features": F.schema(), "actions": policy.actions,
           "cost": policy.cost.as_dict(), "params": policy.params,
           "arrays": {"file": "arrays.npz", "sha256": hashlib.sha256(blob).hexdigest(), "bytes": len(blob)},
           **{k: v for k, v in meta.items() if k != "model_id"}}
    models_dir.mkdir(parents=True, exist_ok=True)
    tmp = models_dir / f".tmp-{model_id}"
    shutil.rmtree(tmp, ignore_errors=True)
    write_durable(tmp / "arrays.npz", [blob], sync_dir=False)
    write_durable(tmp / "model.json", [json.dumps(doc, ensure_ascii=False, indent=1, default=float).encode()])
    os.replace(tmp, models_dir / model_id)
    fsync_dir(models_dir)
    return model_id


def load_artifact(models_dir: Path, model_id: str, *, admissible: list[str]) -> tuple[Policy, dict]:
    """Перевіряє все до побудови політики; нічого не імпортує й не виконує з артефакту."""
    if not model_id or "/" in model_id or "\\" in model_id or model_id.startswith("."):
        raise ModelError(f"некоректний ідентифікатор моделі {model_id!r}")
    base = models_dir / model_id
    try:
        raw = (base / "model.json").read_bytes()
        if len(raw) > MAX_JSON:
            raise ModelError("model.json завеликий")
        doc = json.loads(raw)
        blob = (base / doc["arrays"]["file"]).read_bytes() if doc["arrays"]["file"] == "arrays.npz" else b""
    except ModelError:
        raise
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise ModelError(f"артефакт {model_id} не читається: {type(e).__name__}") from None
    if doc.get("format") != MODEL_FORMAT or doc.get("format_version") != MODEL_VERSION:
        raise ModelError("невідомий формат артефакту")
    if doc.get("kind") not in CLASSES:
        raise ModelError(f"невідомий тип моделі {doc.get('kind')!r}")
    if doc.get("feature_schema") != F.schema_id() or doc.get("features", {}).get("names") != list(F.NAMES):
        raise ModelError("несумісна схема ознак")
    actions = doc.get("actions")
    if not isinstance(actions, list) or not actions or any(a not in admissible for a in actions):
        raise ModelError("набір дій моделі виходить за допустимі дії конфігурації")
    if len(blob) > MAX_ARRAYS or hashlib.sha256(blob).hexdigest() != doc["arrays"].get("sha256"):
        raise ModelError("масиви моделі пошкоджено (SHA-256 не збігається)")
    try:
        with np.load(io.BytesIO(blob), allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
    except (ValueError, OSError) as e:
        raise ModelError(f"масиви не читаються: {type(e).__name__}") from None
    if any(a.dtype == object or (a.dtype.kind == "f" and not np.isfinite(a).all()) for a in arrays.values()):
        raise ModelError("масиви містять нечислові або нескінченні значення")
    try:
        cost = CostModel(int(doc["cost"]["copies"]), float(doc["cost"]["cpu_second_equiv_bytes"]))
        policy = CLASSES[doc["kind"]](actions, cost, doc.get("params") or {})
        policy.load_arrays(arrays)
        probe = policy.choose(np.zeros((2, F.N_FEATURES), dtype=np.float32))
    except Exception as e:                                # невідповідні форми, ключі, параметри
        raise ModelError(f"модель не будується: {type(e).__name__}") from None
    if probe.shape != (2,) or probe.min() < 0 or probe.max() >= len(actions):
        raise ModelError("модель повертає дії поза набором")
    return policy, doc
