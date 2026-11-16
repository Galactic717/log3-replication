"""Точка розширення для навчання (етап Артема): CompressionAdvisor.

    recommend(context) -> Recommendation    перед стисненням блока
    observe(outcome)   -> None              після того, як блок записано (фактичний результат)

Контракт:
  * у context лише те, що відомо ДО рішення; результат приходить окремо в observe;
  * advisor радить тільки кодек і рівень. Він не впливає на видалення, ключі, retention
    і кількість копій: тракт просто не передає йому таких важелів;
  * будь-яка помилка, невідомий кодек або відсутність advisor -> працює BaselineAdvisor,
    backup і restore не блокуються; restore advisor не викликає взагалі.
Підключення: у config.json  "compression": {"advisor": "пакет.модуль:Клас"}; клас без аргументів.
"""
from __future__ import annotations

import importlib
import math
from collections import Counter
from dataclasses import dataclass
from typing import Protocol

from . import codecs

SAMPLE_BYTES = 4096
_PACKED = frozenset("7z apk avi bz2 docx flac gif gz heic jar jpeg jpg m4a mkv mov mp3 mp4 odt ogg png pptx rar "
                    "webm webp xlsx xz zip zst".split())


@dataclass(frozen=True)
class CompressionContext:
    operation_id: str          # стабільний між повторами тієї самої операції backup
    version_id: str
    path_suffix: str           # розширення без крапки, нижній регістр; "" якщо нема
    file_size_bytes: int       # розмір файла на момент відкриття
    chunk_index: int           # номер блока у файлі від 0
    chunk_len_bytes: int
    head: bytes                # перші 16 байтів блока (сигнатури форматів)
    sample_entropy_bits: float  # ентропія Шеннона перших SAMPLE_BYTES блока, біт/байт, 0..8
    default_codec: str         # кодек і рівень із конфігурації сховища
    default_level: int


@dataclass(frozen=True)
class Recommendation:
    codec: str                 # назва з реєстру codecs.BY_NAME
    level: int = 3


@dataclass(frozen=True)
class CompressionOutcome:
    operation_id: str
    version_id: str
    object_id: str
    path_suffix: str
    chunk_index: int
    codec_requested: str
    codec_used: str            # "store", якщо стиснення виявилось невигідним
    raw_len_bytes: int
    stored_len_bytes: int      # разом зі службовим заголовком і тегом
    encode_ns: int             # час стиснення, наносекунди
    advisor_failed: bool       # рекомендацію відхилено, застосовано базову політику


class CompressionAdvisor(Protocol):
    def recommend(self, context: CompressionContext) -> Recommendation: ...

    def observe(self, outcome: CompressionOutcome) -> None: ...


def sample_entropy(data: bytes) -> float:
    sample = data[:SAMPLE_BYTES]
    if not sample:
        return 0.0
    n = len(sample)
    return -sum(c / n * math.log2(c / n) for c in Counter(sample).values())


class BaselineAdvisor:
    """Робоча політика без навчання: уже стиснене й високоентропійне — Store, решта — кодек із конфігурації."""

    def recommend(self, context: CompressionContext) -> Recommendation:
        if context.path_suffix in _PACKED or context.sample_entropy_bits > 7.5:
            return Recommendation("store", 0)
        return Recommendation(context.default_codec, context.default_level)

    def observe(self, outcome: CompressionOutcome) -> None:
        return None


class SafeAdvisor:
    """Обгортка тракту: ізолює збої зовнішнього advisor і рахує їх."""

    def __init__(self, inner: CompressionAdvisor | None = None):
        self.inner, self.baseline, self.errors = inner, BaselineAdvisor(), 0

    def recommend(self, context: CompressionContext) -> tuple[Recommendation, bool]:
        """(рекомендація, чи довелось відкотитись на базову політику)."""
        if self.inner is not None:
            try:
                rec = self.inner.recommend(context)
                if rec.codec in codecs.BY_NAME and isinstance(rec.level, int) and 0 <= rec.level <= 19:
                    return rec, False
            except Exception:
                pass
            self.errors += 1
        return self.baseline.recommend(context), self.inner is not None

    def observe(self, outcome: CompressionOutcome) -> None:
        if self.inner is not None:
            try:
                self.inner.observe(outcome)
            except Exception:
                self.errors += 1


def load_advisor(spec: str | None) -> SafeAdvisor:
    """"module:Class" -> SafeAdvisor. Якщо модуль не імпортується, працює базова політика."""
    if not spec:
        return SafeAdvisor()
    try:
        module, _, name = spec.partition(":")
        return SafeAdvisor(getattr(importlib.import_module(module), name)())
    except Exception:
        safe = SafeAdvisor()
        safe.errors = 1
        return safe
