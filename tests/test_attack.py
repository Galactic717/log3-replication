"""Модуль атаки: ГА відновлює ключ слабкого шифру; tamper-тест через будь-який receiver."""
import pytest
from conftest import PASSWORD, SALT, make_records

from log3.attack import ga_recover_key, tamper_test
from log3.attack.weak_cipher import decrypt, encrypt, known_pairs, random_key
from log3.replica import as_receiver, build_replica


def test_weak_cipher_roundtrip():
    key = random_key(8, seed=1)
    data = bytes(range(256)) * 2
    assert decrypt(key, encrypt(key, data)) == data
    assert encrypt(key, data) != data


def test_attack_recovers_weak_key():
    key = random_key(8, seed=7)
    pairs = known_pairs(key, make_records(64, seed=7), n_pairs=4, seed=7)
    res = ga_recover_key(pairs, seed=0)
    assert res.key == key
    assert res.fitness == 1.0
    # ключ, знайдений ГА, розшифровує нові шифротексти того самого ключа
    secret = bytes(range(64))
    assert decrypt(res.key, encrypt(key, secret)) == secret


@pytest.fixture(scope="module")
def blob():
    return build_replica(make_records(10, seed=4), PASSWORD, replica_group_id=1, batch_seq=0, salt=SALT)


def test_tamper_all_rejected(blob):
    receiver = as_receiver(PASSWORD)
    assert receiver(blob) is None                      # оригінал приймається
    stats = tamper_test(blob, receiver, trials=300, seed=0)
    assert stats["trials"] == 300
    assert stats["accepted"] == 0
    assert stats["rejected"] == 300
    assert stats.get("E_HMAC", 0) > 0                  # змінений шифротекст/HMAC -> E_HMAC
    assert sum(v for k, v in stats.items() if k.startswith("E_")) == 300


def test_tamper_is_receiver_agnostic(blob):
    """Модуль не імпортує наш parse_replica: працює з будь-яким receiver(blob) -> код | None."""
    def naive(b):                                                   # поганий приймач
        return None if b[:4] == b"LOG3" else "E_MAGIC"

    stats = tamper_test(blob, naive, trials=200, seed=1)
    assert stats["accepted"] > 0                       # тест виявляє, що приймач пропускає зміни
    assert stats["trials"] == stats["accepted"] + stats["rejected"]


def test_tamper_deterministic(blob):
    r = as_receiver(PASSWORD)
    assert tamper_test(blob, r, trials=50, seed=5) == tamper_test(blob, r, trials=50, seed=5)
