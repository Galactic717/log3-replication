# LOG3 — пріоритетна реплікація журналів подій

Розумна захищена система пріоритетної реплікації журналів подій.
Командний семестровий проєкт «Проєкт 3», дисципліна «Алгоритми і оптимізація програмних рішень» (КНУ). Формат репліки — `LOG3`.

## Конвеєр

```
події → класифікація (перцептрон) → стиснення (канонічний Гаффман) → вибір копій (ДП)
      → порядок реплікації (жадібні) → AES-256-CTR + HMAC → вузли → відновлення
                ГА налаштовує параметри · модуль атаки перевіряє захист
```

## Команда

| GitHub | Роль | Зони | Завдання |
| :-- | :-- | :-- | :-- |
| [@Galactic717](https://github.com/Galactic717) | голова | формат LOG3, захист, відновлення, інтеграція, звіт | [docs/tasks/Galactic717.md](docs/tasks/Galactic717.md) |
| [@yaroslavronenko-ux](https://github.com/yaroslavronenko-ux) | учасник | Гаффман, ДП, вузли й порядок | [docs/tasks/yaroslavronenko-ux.md](docs/tasks/yaroslavronenko-ux.md) |
| [@artemkovbasuk780-hub](https://github.com/artemkovbasuk780-hub) | учасник | перцептрон, ГА, модуль атаки | [docs/tasks/artemkovbasuk780-hub.md](docs/tasks/artemkovbasuk780-hub.md) |

Починати з [docs/OVERVIEW.md](docs/OVERVIEW.md), потім [docs/CONTRACTS.md](docs/CONTRACTS.md) і своє завдання. Агентам — [AGENTS.md](AGENTS.md).

## Гілки

| Гілка | Призначення |
| :-- | :-- |
| `main` | стабільна версія; лише через PR із `dev`, перед Чекпоінтами 1–3 |
| `dev` | інтеграція; сюди ідуть усі PR із гілок зон |
| `feat/format-crypto` | Гліб: формат LOG3, приймання репліки, PBKDF2/AES/HMAC |
| `feat/integration` | Гліб: відновлювач, пайплайн, метрика Q, звіт |
| `feat/huffman` | Yaroslav: канонічний Гаффман |
| `feat/planner` | Yaroslav: ДП для кількості копій, жадібний baseline, вузли й batch |
| `feat/perceptron` | Artem: перцептрон, ознаки з вікном історії |
| `feat/ga-attack` | Artem: генетичний алгоритм, модуль атаки |

## Правила роботи

- Кожен комітить зі **свого** акаунта: індивідуальна оцінка спирається на внесок у GitHub.
- Зміни — гілка зони (або `feat/<зона>-<задача>` від неї) → PR у `dev` → перегляд іншим учасником.
- `main` завжди запускається; прямих пушів немає.
- Пакет викладача (специфікації, дані, `replication.py`, `reference/`) лежить локально в `materials/` і в git не потрапляє. Не комітити також `gen_project3.py`, `teacher/`, `holdout.csv`, `official_eval.json`, `.env` і ключі.
- Використання ШІ фіксується в `META-USAGE` (обов'язковий артефакт проєкту).

## Запуск

Потрібен Python ≥ 3.11.

```bash
python -m venv .venv
.venv/Scripts/activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
pytest -q                       # усі тести; приймальні кейси — pytest -k test_c1_
python -m log3.pipeline degraded 0          # один прогін: сценарій, seed -> S, R, T, Q
python scripts/reproduce.py                 # відтворює results/ (C1, таблиці 2–5, final_metrics.json)
```

Пакет викладача покладіть у `materials/` (він у `.gitignore`): `FORMAT_SPEC_P3.md`, `REPLICATION_SPEC.md`, `team_config.json`, `replication.py`, `train.csv`, `valid.csv`, `session.csv`, `reference/`. Код підхоплює їх сам. Поки їх нема, працюють dev-заміни: `config/team_config.dev.json`, `log3/devsim.py` (модель реплікації) і синтетичний сеанс. Тести еталонних пакетів (`test_c1_12_*`) у такому разі пропускаються. Таблиці в `results/` у шапці вказують, на якому середовищі їх пораховано. Інший конфіг можна задати так: `LOG3_CONFIG=path/to/team_config.json`.

## Структура коду

| Модуль | Що робить | Зона |
| :-- | :-- | :-- |
| `log3/records.py` | 16-байтовий запис, CSV | Гліб |
| `log3/config.py` | параметри з `team_config.json` | Гліб |
| `log3/payload.py` | payload LOG3, NO_COMPRESSION, строгий розбір | Гліб |
| `log3/crypto.py`, `log3/replica.py` | PBKDF2, AES-256-CTR, HMAC; репліка й приймання за §5 | Гліб |
| `log3/recovery.py`, `log3/pipeline.py` | відновлення з дедуплікацією; наскрізний прогін і Q | Гліб |
| `log3/huffman.py` | канонічний Гаффман | Гліб |
| `log3/planner/` | ДП, жадібний baseline, вузли й batch | Гліб |
| `log3/nn/` | перцептрон *(зараз заглушка: усе класу 0)* | — |
| `log3/devsim.py` | dev-заміна `replication.py`, поки нема пакета | Гліб |
| `scripts/reproduce.py` | усе в `results/` однією командою | Гліб |
