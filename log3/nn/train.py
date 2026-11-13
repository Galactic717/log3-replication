"""Навчання перцептрона і вибір моделі за macro-F1 на valid.

    python -m log3.nn.train            # навчити, зберегти log3/nn/model.npz і results/confusion_matrix.md
    python -m log3.nn.train --seeds 0 1 2 3 4 --epochs 80

Як обирається модель:
  * ≥ 3 ініціалізації (seeds) — результат MLP залежить від початкових ваг;
  * у межах одного запуску зберігаємо ваги епохи з найкращим macro-F1 на valid (early stopping);
  * серед запусків беремо найкращий за macro-F1 на valid. Holdout викладача ми не бачимо,
    тому не підганяємо гіперпараметри під valid нескінченно: сітка мала й зафіксована тут.
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
from .data import DEV_VALID, load_split
from .features import FeatureSpec, Normalizer, build_features, feature_names, spec_for_config
from .mlp import MLP, SGD, Adam, confusion_matrix, macro_f1

MODEL_PATH = Path(__file__).resolve().parent / "model.npz"
RESULTS = ROOT / "results"


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
             meta=np.array(json.dumps({**meta, "history": spec.history,
                                       "temporal": list(spec.temporal or ())}, ensure_ascii=False)))


def _cm_markdown(M: np.ndarray) -> str:
    k = len(M)
    head = "| справжній \\ передбачений | " + " | ".join(str(j) for j in range(k)) + " |"
    rows = [head, "| :-- |" + " --: |" * k]
    rows += [f"| {i} | " + " | ".join(str(v) for v in M[i]) + " |" for i in range(k)]
    return "\n".join(rows)


def train_model(cfg: Config | None = None, *, seeds=(0, 1, 2), settings: TrainSettings = TrainSettings(),
                history: int = 16, model_path: Path = MODEL_PATH, report: bool = True) -> dict:
    cfg = cfg or load_config()
    spec = spec_for_config(cfg, history)
    tr_recs, ytr, src = load_split("train")
    va_recs, yva, _ = load_split("valid")
    n_classes = max(cfg.n_classes, max(ytr) + 1, max(yva) + 1)
    Xtr_raw = build_features(tr_recs, spec)
    norm = Normalizer.fit(Xtr_raw)                       # статистика лише train
    Xtr, Xva = norm.transform(Xtr_raw), norm.transform(build_features(va_recs, spec))

    runs = []
    for sd in seeds:
        s = TrainSettings(**{**asdict(settings), "seed": sd})
        net, f1 = train_one(Xtr, ytr, Xva, yva, s, n_classes=n_classes)
        runs.append((f1, sd, net))
        print(f"[nn] seed={sd}: macro-F1(valid) = {f1:.4f}")
    f1, sd, net = max(runs, key=lambda r: (r[0], -r[1]))
    pred = net.predict(Xva)
    M = confusion_matrix(yva, pred, n_classes)
    acc = float((pred == np.asarray(yva)).mean())
    meta = {"data_source": src, "seed": sd, "macro_f1_valid": f1, "accuracy_valid": acc,
            "n_classes": n_classes, "settings": asdict(settings) | {"seed": sd},
            "dev_valid_records": DEV_VALID[0], "dev_valid_seed": DEV_VALID[1]}
    save_model(model_path, net, norm, spec, meta)

    if report:
        # абляція «без вікна історії»: ті самі налаштування, лише ознаки самого запису
        flat = FeatureSpec(history=spec.history, temporal=())
        fr = build_features(tr_recs, flat)
        fn = Normalizer.fit(fr)
        _, f1_flat = train_one(fn.transform(fr), ytr, fn.transform(build_features(va_recs, flat)), yva,
                               TrainSettings(**{**asdict(settings), "seed": sd}), n_classes=n_classes)
        meta["macro_f1_valid_no_history"] = f1_flat
        RESULTS.mkdir(exist_ok=True)
        lines = [
            "# Матриця помилок перцептрона (valid)",
            "",
            f"Згенеровано `python -m log3.nn.train`. Дані: **{src}**"
            + (" (dev-набір `log3/nn/data.py`, не `materials/valid.csv` — перенавчити, коли буде пакет викладача)"
               if src == "dev" else "") + ".",
            "",
            f"Класи: 0 — штатна, 1 — аномальна, 2 — перехід до інциденту. F1_min = {cfg.F1_min}.",
            f"Архітектура: {Xtr.shape[1]} → {' → '.join(map(str, settings.hidden))} → {n_classes}, "
            f"ReLU, softmax, {settings.optimizer}, lr={settings.lr}, L2={settings.l2}, "
            f"вікно історії H={spec.history}.",
            "",
            "| seed | macro-F1 (valid) |",
            "| --: | --: |",
            *[f"| {s_} | {f:.4f} |" for f, s_, _ in sorted(runs, key=lambda r: r[1])],
            "",
            f"Обрано seed={sd}: macro-F1 = **{f1:.4f}**, accuracy = {acc:.4f}.",
            "",
            f"Без вікна історії (лише ознаки самого запису, той самий seed): macro-F1 = {f1_flat:.4f}.",
            "Різниця — внесок часових ознак; найбільше вона видна на класі 2.",
            "",
            _cm_markdown(M),
            "",
            "| клас | precision | recall | F1 | підтримка |",
            "| --: | --: | --: | --: | --: |",
        ]
        for k in range(n_classes):
            tp = M[k, k]
            p_ = tp / M[:, k].sum() if M[:, k].sum() else 0.0
            r_ = tp / M[k, :].sum() if M[k, :].sum() else 0.0
            f_ = 2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0.0
            lines.append(f"| {k} | {p_:.3f} | {r_:.3f} | {f_:.3f} | {M[k, :].sum()} |")
        lines.append("")
        lines.append("Ознаки: " + ", ".join(f"`{n}`" for n in feature_names(spec)) + ".")
        (RESULTS / "confusion_matrix.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=TrainSettings.epochs)
    ap.add_argument("--hidden", type=int, nargs="+", default=list(TrainSettings.hidden))
    ap.add_argument("--lr", type=float, default=TrainSettings.lr)
    ap.add_argument("--optimizer", choices=("adam", "sgd"), default="adam")
    ap.add_argument("--history", type=int, default=16)
    a = ap.parse_args()
    m = train_model(seeds=a.seeds, history=a.history,
                    settings=TrainSettings(hidden=tuple(a.hidden), epochs=a.epochs, lr=a.lr,
                                           optimizer=a.optimizer))
    print(json.dumps(m, ensure_ascii=False, indent=2))
