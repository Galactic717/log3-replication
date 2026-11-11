"""Модуль атаки: ГА проти навмисно слабкого шифру і tamper-тест приймача.

    python -m log3.attack          # сценарій атаки -> results/attack.md
"""
from .key_search import KeyRecovery, ga_recover_key, random_key_search
from .tamper import tamper_test

__all__ = ["KeyRecovery", "ga_recover_key", "random_key_search", "tamper_test"]
