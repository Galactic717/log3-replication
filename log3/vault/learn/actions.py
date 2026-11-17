"""Допустимі дії стиснення і модель вартості.

Дія — пара (кодек, рівень) з реєстру log3.vault.codecs. Набір дій задає конфігурація
(learning.actions); модель може обрати лише з нього. Store завжди допустимий і завжди
залишається запасним варіантом: фактично невигідне стиснення тракт замінює Store
незалежно від рекомендації (codecs.encode_best).

Вартість блока в еквівалентних байтах (менше — краще):

    cost = stored_len * copies + encode_cpu_seconds * cpu_second_equiv_bytes

stored_len — довжина зашифрованого об'єкта (payload + службовий заголовок), copies — скільки
разів він буде збережений (min_copies політики), cpu_second_equiv_bytes — скільки байтів
сховища коштує одна секунда CPU на цій машині (налаштування замовника). Час декодування
під час restore не враховано: для zstd він малий, а store/zstd не змінюють формат об'єкта.
"""
from __future__ import annotations

from dataclasses import dataclass

from .. import codecs
from ..cryptobox import OVERHEAD
from ..errors import Exit, VaultError

DEFAULT_ACTIONS = ("store:0", "zstd:1", "zstd:3", "zstd:7", "zstd:12")
LEVELS = {"store": (0, 0), "zstd": (1, 19), "huffman-log3": (0, 0)}
DEFAULT_CPU_SECOND_EQUIV_BYTES = 64 * (1 << 20)


@dataclass(frozen=True, order=True)
class Action:
    codec: str
    level: int

    @property
    def key(self) -> str:
        return f"{self.codec}:{self.level}"

    @property
    def codec_id(self) -> int:
        return codecs.BY_NAME[self.codec].id


STORE = Action("store", 0)


def parse_action(text: str) -> Action:
    codec, _, level = str(text).partition(":")
    try:
        lvl = int(level or 0)
    except ValueError:
        lvl = -1
    lo, hi = LEVELS.get(codec, (1, 0))
    if codec not in codecs.BY_NAME or not lo <= lvl <= hi:
        raise VaultError("E_CONFIG", f"недопустима дія стиснення {text!r}: очікується кодек:рівень "
                         f"із {sorted(LEVELS)}", Exit.POLICY)
    return Action(codec, lvl)


def parse_actions(items) -> tuple[Action, ...]:
    """Впорядкований набір без повторів; Store додається завжди."""
    out = {STORE}
    for item in items:
        out.add(parse_action(item))
    return tuple(sorted(out, key=lambda a: (a.codec != "store", a.codec, a.level)))


@dataclass(frozen=True)
class CostModel:
    copies: int = 2
    cpu_second_equiv_bytes: float = DEFAULT_CPU_SECOND_EQUIV_BYTES

    def cost(self, payload_len, cpu_ns):
        """Працює і зі скалярами, і з numpy-масивами; payload_len без службового заголовка."""
        return (payload_len + OVERHEAD) * self.copies + cpu_ns / 1e9 * self.cpu_second_equiv_bytes

    def as_dict(self) -> dict:
        return {"copies": self.copies, "cpu_second_equiv_bytes": self.cpu_second_equiv_bytes}


@dataclass(frozen=True)
class Measurement:
    action: str            # запитана дія
    codec_used: str        # фактичний кодек ("store", якщо стиснення невигідне)
    payload_len: int       # без службового заголовка об'єкта
    wall_ns: int
    cpu_ns: int            # CPU потоку, що кодував


def measure(data: bytes, action: Action) -> tuple[Measurement, bytes]:
    """Кодує блок так само, як тракт (codecs.encode_best), і вимірює витрати."""
    import time
    t0, c0 = time.perf_counter_ns(), time.thread_time_ns()
    codec_id, payload = codecs.encode_best(data, action.codec_id, action.level)
    wall, cpu = time.perf_counter_ns() - t0, time.thread_time_ns() - c0
    return Measurement(action.key, codecs.get(codec_id).name, len(payload), wall, cpu), payload
