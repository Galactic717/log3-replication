"""Помилки файлового режиму та стабільні коди завершення CLI."""
from __future__ import annotations

import os
from enum import IntEnum


class Exit(IntEnum):
    """Стабільні exit codes. Нові значення лише додаються, наявні не змінюються."""

    OK = 0             # запитаний рівень досягнуто
    ERROR = 1          # інша помилка
    USAGE = 2          # некоректні аргументи (argparse)
    NOT_PROTECTED = 3  # --wait: час вийшов, версія прийнята локально, але політику копій ще не виконано
    ATTENTION = 4      # версія деградована / бюджет повторів вичерпано — потрібна увага
    CAPACITY = 5       # відмова до стійкого приймання: черга, spool або вільне місце
    POLICY = 6         # конфігурація не може виконати політику копій
    INTEGRITY = 7      # не пройдено перевірку автентичності, довжини або контрольної суми
    KEY = 8            # секрет відсутній або не той
    PARTIAL = 9        # версію прийнято, але частину файлів джерела пропущено
    NOT_FOUND = 10     # нема такої версії, сховища чи backend


class VaultError(Exception):
    def __init__(self, code: str, message: str, exit_code: Exit = Exit.ERROR):
        super().__init__(f"{code}: {message}")
        self.code, self.message, self.exit_code = code, message, exit_code


class BackendError(Exception):
    """Збій backend. Завдання повторюється в межах бюджету; текст не містить секретів."""


CRASH_ENV = "LOG3_VAULT_CRASHPOINT"
FAULT_ENV = "LOG3_VAULT_FAULT"


def crashpoint(name: str) -> None:
    """Точка аварії процесу для тестів: os._exit без finally й atexit. Це не тест втрати живлення."""
    if os.environ.get(CRASH_ENV) == name:
        os._exit(77)
