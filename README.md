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
