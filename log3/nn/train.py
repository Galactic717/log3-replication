"""Навчання перцептрона і вибір моделі за macro-F1 на valid (умова §3.1, методичка 11.23).

    python -m log3.nn.train            # навчити, зберегти log3/nn/model.npz і results/confusion_matrix.md

Протокол:
  * мала фіксована сітка (вікно історії H × прихований шар), у кожній точці ≥ 3 ініціалізації;
  * у межах запуску — ваги епохи з найкращим macro-F1 на valid (early stopping);
  * обирається найкращий за macro-F1 на valid запуск. Holdout викладач рахує сам;
  * valid уже використано для вибору, тож його F1 оптимістичний. Чесна оцінка — розмічений
    session.csv (test), який у виборі не бере участі.
Дисбаланс класів: ваги в loss c_k = n / (K · n_k) (рідкісний клас «дорожчий»).
Нормалізація — за статистикою train, зберігається разом із вагами.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..config import ROOT, Config, load_config
from .data import load_split
from .features import FeatureSpec, Normalizer, build_features, feature_names, spec_for_config
from .mlp import MLP, SGD, Adam, confusion_matrix, macro_f1

MODEL_PATH = Path(__file__).resolve().parent / "model.npz"
RESULTS = ROOT / "results"
GRID = {"history": (8, 16), "hidden": ((32, 16), (64, 32))}


@dataclass(frozen=True)
class TrainSettings:
    hidden: tuple[int, ...] = (32, 16)
    optimizer: str = "adam"          # "adam" | "sgd"
    lr: float = 2e-3
    epochs: int = 60
    batch: int = 64
    l2: float = 1e-4
    class_weighting: bool = True
    seed: int = 0


def class_weights(y: np.ndarray, n_classes: int) -> np.ndarray:
    counts = np.bincount(y, minlength=n_classes).astype(float)
    w = len(y) / (n_classes * np.maximum(counts, 1.0))
    w[counts == 0] = 0.0
    return w


def train_one(Xtr, ytr, Xva, yva, s: TrainSettings, *, n_classes: int) -> tuple[MLP, float]:
    """Один запуск із seed s.seed; повертає ваги найкращої за valid-F1 епохи і цей F1."""
    ytr, yva = np.asarray(ytr), np.asarray(yva)
    rng = np.random.default_rng(s.seed)
    net = MLP(Xtr.shape[1], s.hidden, n_classes, seed=s.seed, l2=s.l2)
    opt = Adam(lr=s.lr) if s.optimizer == "adam" else SGD(lr=s.lr, momentum=0.9)
    cw = class_weights(ytr, n_classes) if s.class_weighting else None
    best_f1, best = -1.0, [p.copy() for p in net.params]
    for _ in range(s.epochs):
        order = rng.permutation(len(ytr))                 # перемішування — теж від seed
        for lo in range(0, len(order), s.batch):
            idx = order[lo:lo + s.batch]
            _, grads = net.loss_and_grads(Xtr[idx], ytr[idx], cw)
            opt.step(net.params, grads)
        f1 = macro_f1(yva, net.predict(Xva))
        if f1 > best_f1:
            best_f1, best = f1, [p.copy() for p in net.params]
    net.params = best
    return net, best_f1


def save_model(path: Path, net: MLP, norm: Normalizer, spec: FeatureSpec, meta: dict) -> None:
    np.savez(path, **net.to_arrays(), norm_mean=norm.mean, norm_std=norm.std,
             meta=np.array(json.dumps({**meta, "history": spec.history, "time_window_ms": spec.time_window_ms,
                                       "temporal": list(spec.temporal)}, ensure_ascii=False)))


def _cm_markdown(M: np.ndarray) -> list[str]:
    k = len(M)
    rows = ["| справжній \\ передбачений | " + " | ".join(str(j) for j in range(k)) + " |",
            "| :-- |" + " --: |" * k]
    return rows + [f"| {i} | " + " | ".join(str(v) for v in M[i]) + " |" for i in range(k)]


def _per_class(M: np.ndarray) -> list[str]:
    rows = ["| клас | precision | recall | F1 | підтримка |", "| --: | --: | --: | --: | --: |"]
    for k in range(len(M)):
        tp = M[k, k]
        p_ = tp / M[:, k].sum() if M[:, k].sum() else 0.0
        r_ = tp / M[k, :].sum() if M[k, :].sum() else 0.0
        f_ = 2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0.0
        rows.append(f"| {k} | {p_:.3f} | {r_:.3f} | {f_:.3f} | {M[k, :].sum()} |")
    return rows


def _prepare(spec: FeatureSpec, splits: dict):
    norm = Normalizer.fit(build_features(splits["train"][0], spec))      # статистика лише train
    X = {k: norm.transform(build_features(v[0], spec)) for k, v in splits.items()}
    return norm, X


def train_model(cfg: Config | None = None, *, seeds=(0, 1, 2), settings: TrainSettings = TrainSettings(),
                grid: dict = GRID, model_path: Path = MODEL_PATH, report: bool = True) -> dict:
    cfg = cfg or load_config()
    splits = {k: load_split(k) for k in ("train", "valid", "test")}
    y = {k: np.asarray(v[1]) for k, v in splits.items()}
    n_classes = cfg.n_classes
    if max(int(v.max()) for v in y.values()) >= n_classes:
        raise ValueError(f"мітки виходять за {n_classes} класів із team_config.json")

    runs = []
    for history in grid["history"]:
        spec = spec_for_config(cfg, history)
        norm, X = _prepare(spec, splits)
        for hidden in grid["hidden"]:
            for sd in seeds:
                s = TrainSettings(**{**asdict(settings), "hidden": hidden, "seed": sd})
                net, f1 = train_one(X["train"], y["train"], X["valid"], y["valid"], s, n_classes=n_classes)
                f1_test = macro_f1(y["test"], net.predict(X["test"]))
                runs.append({"history": history, "hidden": list(hidden), "seed": sd, "f1_valid": f1,
                             "f1_test": f1_test, "_net": net, "_norm": norm, "_spec": spec, "_X": X})
                print(f"[nn] H={history} hidden={hidden} seed={sd}: macro-F1 valid {f1:.4f}, test {f1_test:.4f}")
    best = max(runs, key=lambda r: (r["f1_valid"], -r["seed"]))
    net, norm, spec, X = best["_net"], best["_norm"], best["_spec"], best["_X"]
    pred = {k: net.predict(X[k]) for k in X}
    meta = {"data_source": splits["train"][2], "history": spec.history, "hidden": best["hidden"],
            "seed": best["seed"], "macro_f1_valid": best["f1_valid"], "macro_f1_test": best["f1_test"],
            "accuracy_valid": float((pred["valid"] == y["valid"]).mean()), "n_classes": n_classes,
            "settings": asdict(settings) | {"hidden": best["hidden"], "seed": best["seed"]}}
    save_model(model_path, net, norm, spec, meta)
    if not report:
        return meta

    # абляція «без вікна історії»: той самий запуск, лише ознаки самого запису
    flat = FeatureSpec(spec.history, spec.time_window_ms, ())
    _, Xf = _prepare(flat, splits)
    s = TrainSettings(**{**asdict(settings), "hidden": tuple(best["hidden"]), "seed": best["seed"]})
    net_f, f1_flat = train_one(Xf["train"], y["train"], Xf["valid"], y["valid"], s, n_classes=n_classes)
    meta["macro_f1_valid_no_history"] = f1_flat
    meta["macro_f1_test_no_history"] = macro_f1(y["test"], net_f.predict(Xf["test"]))
    meta["runs"] = [{k: v for k, v in r.items() if not k.startswith("_")} for r in runs]

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "classifier_metrics.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
                                                     encoding="utf-8")
    lines = [
        "# Матриця помилок перцептрона",
        "",
        f"Згенеровано `python -m log3.nn.train`. Дані: **{meta['data_source']}** "
        "(train.csv — навчання, valid.csv — вибір моделі, session.csv — незалежна перевірка).",
        "",
        f"Класи: {', '.join(f'{i} — {n}' for i, n in enumerate(cfg.class_names))}. F1_min = {cfg.F1_min}.",
        f"Архітектура: {X['train'].shape[1]} ознак → {' → '.join(map(str, best['hidden']))} → {n_classes}, "
        f"ReLU, softmax, {settings.optimizer}, lr={settings.lr}, L2={settings.l2}, {settings.epochs} епох, "
        f"вікно історії H={spec.history}, часове вікно {spec.time_window_ms:.0f} мс.",
        "",
        "Сітка вибору: H ∈ " + str(list(grid["history"])) + ", прихований шар ∈ "
        + str([list(h) for h in grid["hidden"]]) + f", seeds {list(seeds)} — разом {len(runs)} запусків.",
        "",
        "| H | прихований шар | seed | macro-F1 valid | macro-F1 session (test) |",
        "| --: | :-- | --: | --: | --: |",
        *[f"| {r['history']} | {r['hidden']} | {r['seed']} | {r['f1_valid']:.4f} | {r['f1_test']:.4f} |"
          for r in runs],
        "",
        f"Обрано H={spec.history}, {best['hidden']}, seed={best['seed']}: macro-F1 valid = **{best['f1_valid']:.4f}** "
        f"(оптимістична: за valid обирали), session = **{best['f1_test']:.4f}** (чесна оцінка).",
        "",
        f"Без вікна історії (лише ознаки самого запису): macro-F1 valid = {f1_flat:.4f}, "
        f"session = {meta['macro_f1_test_no_history']:.4f}. Різниця — внесок часових ознак, "
        "передусім для класу 2.",
        "",
        "## valid.csv",
        "",
        *_cm_markdown(confusion_matrix(y["valid"], pred["valid"], n_classes)),
        "",
        *_per_class(confusion_matrix(y["valid"], pred["valid"], n_classes)),
        "",
        "## session.csv (не брав участі у виборі)",
        "",
        *_cm_markdown(confusion_matrix(y["test"], pred["test"], n_classes)),
        "",
        *_per_class(confusion_matrix(y["test"], pred["test"], n_classes)),
        "",
        "Ознаки: " + ", ".join(f"`{n}`" for n in feature_names(spec)) + ".",
    ]
    (RESULTS / "confusion_matrix.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=TrainSettings.epochs)
    a = ap.parse_args()
    m = train_model(seeds=a.seeds, settings=TrainSettings(epochs=a.epochs))
    print(json.dumps({k: v for k, v in m.items() if k != "runs"}, ensure_ascii=False, indent=2))
