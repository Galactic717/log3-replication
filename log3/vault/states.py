"""Явна модель станів версії, завдання реплікації та копії.

Версія:
  CAPTURING      дані читаються з джерела; нічого не обіцяно
  ACCEPTED_LOCAL усі блоки й маніфест стійко записані в локальний spool і каталог;
                 джерело більше не потрібне; політику копій НЕ виконано
  REPLICATING    черга передає об'єкти на backend-и; політику копій ще не виконано
  PROTECTED      кожен блок і маніфест мають підтверджені копії в >= min_copies доменах відмов
  DEGRADED       політику не виконано й автоматичних спроб не лишилось (потрібні repair / увага)
  FAILED         захоплення не завершилось; версію не прийнято
  DELETED        версію прибрано GC після завершення retention

Завдання (один об'єкт -> один backend): PENDING -> RUNNING -> DONE | PENDING (повтор) | ATTENTION.
«Відправлено» станом не є: DONE ставиться лише після перевіреного підтвердження backend.
"""
from __future__ import annotations

from enum import StrEnum

from .errors import VaultError


class VersionState(StrEnum):
    CAPTURING = "CAPTURING"
    ACCEPTED_LOCAL = "ACCEPTED_LOCAL"
    REPLICATING = "REPLICATING"
    PROTECTED = "PROTECTED"
    DEGRADED = "DEGRADED"
    FAILED = "FAILED"
    DELETED = "DELETED"


class JobState(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    DONE = "DONE"
    ATTENTION = "ATTENTION"


class ReplicaState(StrEnum):
    CONFIRMED = "CONFIRMED"   # backend підтвердив стійкий запис саме цього об'єкта
    MISSING = "MISSING"       # verify: об'єкта нема
    CORRUPT = "CORRUPT"       # verify: довжина або SHA-256 не збігаються


V = VersionState
TRANSITIONS: dict[VersionState, frozenset[VersionState]] = {
    V.CAPTURING: frozenset({V.ACCEPTED_LOCAL, V.FAILED}),
    V.ACCEPTED_LOCAL: frozenset({V.REPLICATING, V.PROTECTED, V.DEGRADED}),
    V.REPLICATING: frozenset({V.PROTECTED, V.DEGRADED}),
    V.PROTECTED: frozenset({V.DEGRADED, V.REPLICATING, V.DELETED}),
    V.DEGRADED: frozenset({V.REPLICATING, V.PROTECTED, V.DELETED}),
    V.FAILED: frozenset({V.DELETED}),
    V.DELETED: frozenset(),
}
#: стани, у яких дані версії прийнято і їх можна відновлювати
RESTORABLE = frozenset({V.ACCEPTED_LOCAL, V.REPLICATING, V.PROTECTED, V.DEGRADED})
#: рівень захисту для CLI/JSON; "protected" буває лише в стані PROTECTED
LEVEL = {V.CAPTURING: "capturing", V.ACCEPTED_LOCAL: "accepted_local", V.REPLICATING: "replicating",
         V.PROTECTED: "protected", V.DEGRADED: "degraded", V.FAILED: "failed", V.DELETED: "deleted"}


def check_transition(old: str, new: str) -> None:
    if VersionState(new) not in TRANSITIONS[VersionState(old)]:
        raise VaultError("E_STATE", f"заборонений перехід версії {old} -> {new}")
