"""Багатошаровий перцептрон на numpy: прямий прохід, backprop, SGD/Adam, метрики.

Архітектура: x -> [Linear -> ReLU] × len(hidden) -> Linear -> softmax.
Втрати: зважена крос-ентропія + L2 на ваги (без зсувів):

    L = Σ_i c_{y_i} · (−log p_{i, y_i}) / Σ_i c_{y_i}  +  (λ/2) · Σ_l ||W_l||²

c_k — вага класу k (боротьба з дисбалансом: рідкісний клас важить більше).

Backprop (виведення для захисту). Позначимо Z_l = A_{l-1} W_l + b_l, A_l = ReLU(Z_l), A_0 = X,
P = softmax(Z_L), Y — one-hot міток, s = Σ_i c_{y_i}.
  1. Для softmax + крос-ентропії: ∂L/∂Z_L = diag(c_y) (P − Y) / s   (рядок i помножено на c_{y_i}).
  2. Для лінійного шару: ∂L/∂W_l = A_{l-1}ᵀ · δ_l + λ W_l,  ∂L/∂b_l = Σ_i δ_l[i],
     ∂L/∂A_{l-1} = δ_l · W_lᵀ.
  3. Для ReLU: δ_{l-1} = ∂L/∂A_{l-1} ⊙ [Z_{l-1} > 0].
Правильність перевіряє gradient_check (центральна різниця) у тестах.

Кількість класів — n_out, нічого не припускає рівно 3 класи.
"""
from __future__ import annotations

import numpy as np


class MLP:
    def __init__(self, n_in: int, hidden: tuple[int, ...], n_out: int, *, seed: int, l2: float = 0.0):
        self.sizes = [n_in, *hidden, n_out]
        self.l2 = l2
        rng = np.random.default_rng(seed)
        self.params: list[np.ndarray] = []
        for fan_in, fan_out in zip(self.sizes[:-1], self.sizes[1:]):
            # ініціалізація He: дисперсія 2/fan_in тримає масштаб активацій ReLU між шарами
            self.params.append(rng.normal(0.0, np.sqrt(2.0 / fan_in), size=(fan_in, fan_out)))
            self.params.append(np.zeros(fan_out))

    @property
    def n_out(self) -> int:
        return self.sizes[-1]

    @property
    def n_layers(self) -> int:
        return len(self.params) // 2

    # ---- прямий прохід ----
    def _forward(self, X: np.ndarray):
        A, cache = X, []
        for k in range(self.n_layers):
            W, b = self.params[2 * k], self.params[2 * k + 1]
            Z = A @ W + b
            cache.append((A, Z))
            A = np.maximum(Z, 0.0) if k < self.n_layers - 1 else Z
        return A, cache                                   # A — логіти

    @staticmethod
    def softmax(Z: np.ndarray) -> np.ndarray:
        Z = Z - Z.max(axis=1, keepdims=True)              # числова стійкість: exp не переповнюється
        E = np.exp(Z)
        return E / E.sum(axis=1, keepdims=True)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.softmax(self._forward(X)[0])

    def predict(self, X: np.ndarray) -> np.ndarray:
        if len(X) == 0:
            return np.zeros(0, dtype=int)
        return self._forward(X)[0].argmax(axis=1)

    # ---- втрати і backprop ----
    def loss_and_grads(self, X: np.ndarray, y: np.ndarray, class_weights: np.ndarray | None = None):
        y = np.asarray(y)
        n = len(y)
        logits, cache = self._forward(X)
        P = self.softmax(logits)
        c = np.ones(n) if class_weights is None else np.asarray(class_weights, dtype=float)[y]
        s = c.sum()
        loss = float(-(c * np.log(P[np.arange(n), y] + 1e-12)).sum() / s)
        loss += 0.5 * self.l2 * sum(float((W ** 2).sum()) for W in self.params[0::2])

        delta = P.copy()
        delta[np.arange(n), y] -= 1.0                    # P − Y
        delta *= (c / s)[:, None]
        grads: list[np.ndarray] = [None] * len(self.params)  # type: ignore[list-item]
        for k in reversed(range(self.n_layers)):
            A_prev, _ = cache[k]
            W = self.params[2 * k]
            grads[2 * k] = A_prev.T @ delta + self.l2 * W
            grads[2 * k + 1] = delta.sum(axis=0)
            if k > 0:
                _, Z_prev = cache[k - 1]
                delta = (delta @ W.T) * (Z_prev > 0)
        return loss, grads

    def loss(self, X, y, class_weights=None) -> float:
        return self.loss_and_grads(X, y, class_weights)[0]

    # ---- збереження ----
    def to_arrays(self) -> dict[str, np.ndarray]:
        d = {f"p{i}": p for i, p in enumerate(self.params)}
        d["sizes"] = np.array(self.sizes)
        d["l2"] = np.array(self.l2)
        return d

    @classmethod
    def from_arrays(cls, d) -> MLP:
        sizes = [int(s) for s in d["sizes"]]
        net = cls(sizes[0], tuple(sizes[1:-1]), sizes[-1], seed=0, l2=float(d["l2"]))
        net.params = [np.array(d[f"p{i}"], dtype=float) for i in range(len(net.params))]
        return net


def gradient_check(net: MLP, X, y, *, class_weights=None, eps: float = 1e-5) -> float:
    """Максимальна відносна похибка між backprop і центральною різницею (L(θ+ε)−L(θ−ε))/2ε."""
    _, grads = net.loss_and_grads(X, y, class_weights)
    worst = 0.0
    for p, g in zip(net.params, grads):
        it = np.nditer(p, flags=["multi_index"])
        for _ in it:
            ix = it.multi_index
            old = p[ix]
            p[ix] = old + eps
            lp = net.loss(X, y, class_weights)
            p[ix] = old - eps
            lm = net.loss(X, y, class_weights)
            p[ix] = old
            num = (lp - lm) / (2 * eps)
            worst = max(worst, abs(num - g[ix]) / max(1e-8, abs(num) + abs(g[ix])))
    return worst


# ---- оптимізатори: змінюють params на місці ----

class SGD:
    def __init__(self, lr: float, momentum: float = 0.0):
        self.lr, self.momentum, self.v = lr, momentum, None

    def step(self, params, grads):
        if self.v is None:
            self.v = [np.zeros_like(p) for p in params]
        for p, g, v in zip(params, grads, self.v):
            v *= self.momentum
            v -= self.lr * g
            p += v


class Adam:
    """Adam (Kingma & Ba): ковзні середні градієнта m і його квадрата v з поправкою зсуву."""

    def __init__(self, lr: float = 1e-3, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self.m = self.v = None
        self.t = 0

    def step(self, params, grads):
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]
        self.t += 1
        for p, g, m, v in zip(params, grads, self.m, self.v):
            m *= self.b1
            m += (1 - self.b1) * g
            v *= self.b2
            v += (1 - self.b2) * g * g
            m_hat = m / (1 - self.b1 ** self.t)
            v_hat = v / (1 - self.b2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ---- метрики ----

def confusion_matrix(y_true, y_pred, n_classes: int | None = None) -> np.ndarray:
    """M[i, j] — скільки подій справжнього класу i передбачено як j."""
    y_true, y_pred = np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)
    k = n_classes or int(max(y_true.max(initial=-1), y_pred.max(initial=-1)) + 1)
    M = np.zeros((k, k), dtype=int)
    np.add.at(M, (y_true, y_pred), 1)
    return M


def macro_f1(y_true, y_pred) -> float:
    """Середнє F1 за класами, що трапляються в y_true або y_pred (кожен клас важить однаково)."""
    y_true, y_pred = np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)
    if len(y_true) == 0:
        return 0.0
    labels = np.union1d(y_true, y_pred)
    M = confusion_matrix(y_true, y_pred, int(labels.max()) + 1)
    f1s = []
    for k in labels:
        tp = M[k, k]
        fp = M[:, k].sum() - tp
        fn = M[k, :].sum() - tp
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s))
