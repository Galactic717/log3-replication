"""ГА відновлює ключ слабкого шифру з відомих пар (відкритий текст, шифротекст).

Хромосома — 8·L бітів ключа. Фітнес — частка бітів, у яких encrypt(кандидат, p) збігається
з відомим c, за всіма парами (1.0 — ключ знайдено). Оператори — для бітових рядків:
турнір, рівномірний кросовер, побітова мутація з імовірністю 1/(8L), елітизм.
Перезапуск при застої: зсув у байті i залежить від сусіднього байта ключа, тож буває
локальний оптимум (≈0.94), з якого не вийти зміною одного біта — треба одночасно змінити
два байти. Якщо найкращий фітнес не росте `stagnation` поколінь, популяцію створюємо
заново (найкращий ключ за весь час пам'ятаємо окремо). Зупинка — фітнес 1.0 або вичерпано
покоління. Для порівняння — random_key_search
з тим самим бюджетом оцінок: на 2^64 ключах він не знаходить нічого кращого за ~0.5.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .weak_cipher import encrypt


@dataclass(frozen=True)
class KeyRecovery:
    key: bytes
    fitness: float
    history: list[float]          # найкращий-досі фітнес за поколінням
    evaluations: int
    restarts: int = 0


def _bits_to_key(bits: np.ndarray) -> bytes:
    return bytes(np.packbits(bits.astype(np.uint8)))


def key_fitness(key: bytes, pairs: list[tuple[bytes, bytes]]) -> float:
    same = total = 0
    for p, c in pairs:
        x = np.frombuffer(encrypt(key, p), dtype=np.uint8) ^ np.frombuffer(c, dtype=np.uint8)
        same += 8 * len(x) - int(np.unpackbits(x).sum())
        total += 8 * len(x)
    return same / total if total else 0.0


def ga_recover_key(known_pairs, *, seed: int, key_len: int = 8, pop_size: int = 80,
                   generations: int = 1000, tournament_k: int = 3, stagnation: int = 25) -> KeyRecovery:
    rng = np.random.default_rng(seed)
    n = 8 * key_len

    def fresh():
        p = rng.integers(0, 2, size=(pop_size, n), dtype=np.uint8)
        return p, np.array([key_fitness(_bits_to_key(ind), known_pairs) for ind in p])

    pop, fit = fresh()
    evals, history = pop_size, [float(fit.max())]
    best_bits, best_f = pop[int(fit.argmax())].copy(), float(fit.max())
    since_improve, restarts = 0, 0

    def tournament():
        idx = rng.choice(pop_size, size=min(tournament_k, pop_size), replace=False)
        return pop[idx[fit[idx].argmax()]]

    for _ in range(generations - 1):
        if best_f == 1.0:
            break
        if since_improve >= stagnation:                               # перезапуск при застої
            pop, fit = fresh()
            evals += pop_size
            since_improve, restarts = 0, restarts + 1
        children = np.empty_like(pop)
        for c in range(pop_size):
            a, b = tournament(), tournament()
            child = np.where(rng.random(n) < 0.5, a, b)              # рівномірний кросовер
            flip = rng.random(n) < 1.0 / n                            # побітова мутація
            children[c] = child ^ flip
        child_fit = np.array([key_fitness(_bits_to_key(ind), known_pairs) for ind in children])
        evals += pop_size
        elite = int(fit.argmax())                                     # елітизм
        worst = int(child_fit.argmin())
        children[worst], child_fit[worst] = pop[elite], fit[elite]
        pop, fit = children, child_fit
        if fit.max() > best_f:
            best_bits, best_f, since_improve = pop[int(fit.argmax())].copy(), float(fit.max()), 0
        else:
            since_improve += 1
        history.append(best_f)
    return KeyRecovery(_bits_to_key(best_bits), best_f, history, evals, restarts)


def random_key_search(known_pairs, *, evaluations: int, seed: int, key_len: int = 8) -> KeyRecovery:
    rng = np.random.default_rng(seed)
    best_k, best_f = b"", -1.0
    for _ in range(evaluations):
        k = bytes(rng.integers(0, 256, size=key_len, dtype=np.uint8))
        f = key_fitness(k, known_pairs)
        if f > best_f:
            best_k, best_f = k, f
    return KeyRecovery(best_k, best_f, [best_f], evaluations)
