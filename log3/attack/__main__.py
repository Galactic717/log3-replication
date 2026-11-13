"""Сценарій атаки -> results/attack.md.

    python -m log3.attack [--keys 10] [--trials 2000]

1. ГА проти слабкого шифру (weak_cipher): для кількох випадкових ключів — чи знайдено ключ,
   скільки поколінь/оцінок, і що дає random search з тим самим бюджетом.
2. Tamper-тест нашого приймача: справжня репліка LOG3 (AES-256-CTR + HMAC) і `trials`
   випадкових змін; рахуємо, скільки відхилено і з якими кодами.
"""
from __future__ import annotations

import argparse
import statistics

from ..config import ROOT, load_config
from ..pipeline import load_session
from ..recovery import recover
from ..replica import HEADER_SIZE, as_receiver, build_replica
from .key_search import ga_recover_key, random_key_search
from .tamper import KINDS, tamper_test
from .weak_cipher import known_pairs, random_key


def main(n_keys: int = 10, trials: int = 2000, key_len: int = 8, n_pairs: int = 4) -> str:
    lines = [
        "# Сценарій атаки",
        "",
        f"Згенеровано `python -m log3.attack --keys {n_keys} --trials {trials}`.",
        "",
        "## 1. ГА проти навмисно слабкого шифру",
        "",
        "Шифр (`log3/attack/weak_cipher.py`): `c_i = rotl8(p_i XOR k[i mod L], k[(i+1) mod L] mod 8)`, "
        f"ключ L = {key_len} байтів (2^{8 * key_len} варіантів — перебір неможливий). "
        f"Модель атаки — відомий відкритий текст: {n_pairs} пари по 64 байти (упаковані записи журналу; "
        "формат запису публічний). Фітнес — частка бітів, у яких encrypt(кандидат, p) збігається з c.",
        "AES-256-CTR, яким шифруються справжні репліки, не атакуємо: у нього немає часткової "
        "правильності — хибний біт ключа змінює ≈ половину бітів кожного блоку, і ландшафт фітнесу пласкіший.",
        "",
        "| ключ (seed) | ГА: знайдено | ГА: покоління | ГА: оцінок | перезапусків "
        "| random search: найкращий фітнес (той самий бюджет) |",
        "| --: | :-- | --: | --: | --: | --: |",
    ]
    cfg = load_config()
    session = load_session(cfg)
    found, evals = 0, []
    for s in range(n_keys):
        key = random_key(key_len, seed=s)
        pairs = known_pairs(key, session, n_pairs=n_pairs, seed=s)
        r = ga_recover_key(pairs, seed=s, key_len=key_len)
        rs = random_key_search(pairs, evaluations=r.evaluations, seed=s, key_len=key_len)
        ok = r.key == key
        found += ok
        evals.append(r.evaluations)
        lines.append(f"| {s} | {'так' if ok else f'ні ({r.fitness:.3f})'} | {len(r.history)} | "
                     f"{r.evaluations} | {r.restarts} | {rs.fitness:.3f} |")
    lines += [
        "",
        f"ГА знайшов ключ у {found} з {n_keys} випадків; медіана — {int(statistics.median(evals))} "
        f"оцінок фітнесу (проти 2^{8 * key_len} ≈ {2 ** (8 * key_len):.1e} для перебору). "
        "Random search із тим самим бюджетом не підходить близько до 1.0: без відбору він не накопичує "
        "часткову правильність ключа.",
        "",
        "Чому працює: шифр без дифузії — байт ключа впливає лише на «свої» позиції, тож частково "
        "правильний ключ дає частково правильний шифротекст. Пастка — зсув залежить від сусіднього байта: "
        "буває локальний оптимум ≈ 0.94, з якого одним бітом не вийти, тому ГА перезапускається при застої.",
        "",
    ]

    recs = session[:32]
    blob = build_replica(recs, cfg.password, replica_group_id=1, batch_seq=0)
    receiver = as_receiver(cfg.password)
    assert receiver(blob) is None, "незмінена репліка має прийматися"
    st = tamper_test(blob, receiver, trials=trials, seed=0)
    codes = sorted(k for k in st if k.startswith("E_") or k.startswith("CRASH"))
    lines += [
        "## 2. Tamper-тест приймача LOG3",
        "",
        f"Репліка з {len(recs)} записів, {len(blob)} байтів (заголовок 43 + шифротекст + HMAC 32). "
        f"{trials} випадкових змін: {', '.join(KINDS)}. Приймач — `replica.as_receiver(password)`; "
        "сам `tamper_test` наш `parse_replica` не імпортує, тож на тижні змагань його можна спрямувати "
        "на приймач іншої команди.",
        "",
        "| результат | кількість |",
        "| :-- | --: |",
        f"| змін усього | {st['trials']} |",
        f"| прийнято (вразливість) | {st['accepted']} |",
        f"| відхилено | {st['rejected']} |",
        *[f"| відхилено з `{c}` | {st[c]} |" for c in codes],
        "",
        "Зміна будь-якого байта шифротексту чи HMAC дає `E_HMAC`: HMAC-SHA256 рахується над заголовком і "
        "шифротекстом і перевіряється до розшифрування, тож парсер не бачить підроблених даних. Зміна magic — "
        "`E_MAGIC`, обрізання й дописування — `E_SHORT`/`E_LENGTH` (перевірки йдуть до HMAC).",
        "",
        "## 3. Змістовні сценарії підміни",
        "",
        "| сценарій | результат приймача |",
        "| :-- | :-- |",
        *[f"| {name} | {res} |" for name, res in _targeted(cfg, recs, blob, receiver)],
        "",
        "Повтор справжньої репліки HMAC не ловить (вона справжня), але шкоди немає: відновлювач "
        "дедуплікує записи за 16-байтовим поданням, тож повтор не подвоює подій і не змінює R. "
        "Захист від повтору (лічильник batch_seq на приймачі) у базовий формат не входить.",
    ]
    text = "\n".join(lines) + "\n"
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "attack.md").write_text(text, encoding="utf-8")
    return text


def _targeted(cfg, recs, blob, receiver) -> list[tuple[str, str]]:
    """Атаки з наміром, а не випадкові біти: чужий ключ, підміна полів заголовка, повтор."""
    def verdict(b):
        code = receiver(b)
        return "прийнято" if code is None else f"відхилено, `{code}`"
    foreign = build_replica(recs, cfg.password + "-other-team", replica_group_id=1, batch_seq=0)
    seq = bytearray(blob)
    seq[7:11] = (int.from_bytes(seq[7:11], "big") + 1).to_bytes(4, "big")        # batch_seq + 1
    nonce = bytearray(blob)
    nonce[27] ^= 0x01                                                             # інший nonce
    replayed, _ = recover([blob, blob], cfg.password)
    return [
        ("репліка іншої команди (інший пароль)", verdict(foreign)),
        ("змінено batch_seq у відкритому заголовку", verdict(bytes(seq))),
        ("змінено nonce (спроба іншого keystream)", verdict(bytes(nonce))),
        ("шифротекст іншої репліки під цим заголовком",
         verdict(blob[:HEADER_SIZE] + foreign[HEADER_SIZE:])),
        ("повтор справжньої репліки двічі",
         f"прийнято обидві; після дедуплікації {len(replayed)} записів із {len(recs)}"),
    ]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys", type=int, default=10)
    ap.add_argument("--trials", type=int, default=2000)
    a = ap.parse_args()
    print(main(a.keys, a.trials))
