"""НАВМИСНО СЛАБКИЙ навчальний шифр — мішень для ГА-атаки. Ніде в системі не використовується.

Справжні репліки шифрує AES-256-CTR, і його не атакуємо (умова). Тут потрібен шифр,
на якому видно, як ГА шукає ключ, коли перебір неможливий (ключ 8 байтів = 2^64 варіантів).

    c_i = rotl8(p_i XOR k[i mod L], k[(i+1) mod L] mod 8)

Чому слабкий: немає дифузії — байт ключа k_j впливає лише на позиції i ≡ j (mod L)
(через XOR) і i ≡ j−1 (через зсув). Тому частково правильний ключ дає частково правильний
шифротекст, і функція «частка збіжних бітів» має плавний ландшафт, яким ГА піднімається
до відповіді. У AES один хибний біт ключа змінює ≈ половину бітів кожного блоку — ландшафт
пласкіший за голку в копиці, і ГА там не кращий за випадкове вгадування.

Зчеплення сусідніх байтів ключа через зсув дає епістаз: оптимум одного гена залежить від
іншого, тож це не L незалежних задач, а саме задача для популяційного пошуку.
**[звірити]** Якщо в пакеті викладача є свій слабкий шифр і навчальний набір — замінити тут.
"""
from __future__ import annotations

import random

from ..records import pack_records


def _rotl(x: int, r: int) -> int:
    return ((x << r) | (x >> (8 - r))) & 0xFF if r else x


def _rotr(x: int, r: int) -> int:
    return ((x >> r) | (x << (8 - r))) & 0xFF if r else x


def encrypt(key: bytes, data: bytes) -> bytes:
    L = len(key)
    return bytes(_rotl(b ^ key[i % L], key[(i + 1) % L] & 7) for i, b in enumerate(data))


def decrypt(key: bytes, data: bytes) -> bytes:
    L = len(key)
    return bytes(_rotr(b, key[(i + 1) % L] & 7) ^ key[i % L] for i, b in enumerate(data))


def random_key(length: int, *, seed: int) -> bytes:
    rng = random.Random(seed)
    return bytes(rng.randrange(256) for _ in range(length))


def known_pairs(key: bytes, *, n_pairs: int, seed: int, records_per_pair: int = 4) -> list[tuple[bytes, bytes]]:
    """Пари (відкритий текст, шифротекст): відкритий текст — упаковані записи журналу
    (атака з відомим відкритим текстом: формат запису публічний, заголовки передбачувані)."""
    from ..nn.data import dev_labeled_session
    recs = dev_labeled_session(n_pairs * records_per_pair, seed)[0]
    out = []
    for k in range(n_pairs):
        p = pack_records(recs[k * records_per_pair:(k + 1) * records_per_pair])
        out.append((p, encrypt(key, p)))
    return out
