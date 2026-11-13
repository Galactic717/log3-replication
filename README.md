# LOG3 — розумна захищена система пріоритетної реплікації журналів подій

Командний проєкт 3 з дисципліни «Алгоритми і оптимізація програмних рішень» (КНУ). Система вирішує, які
події журналу зберегти, у скількох копіях і на яких вузлах, щоб за обмеженого бюджету реплікації після
відмов вузлів відновити максимум критичних подій, а копії лишались нечитабельними для сторони з доступом до
вузлів.

```
session.csv → [1] перцептрон (клас 0/1/2) → v_i = priority(клас) + critical_bonus·critical
            → [2] таблиця Гаффмана вікна → w_i = Σ довжин кодів 16 байтів запису
            → [3] ДП: r_i ∈ {0..3} копій під бюджет вікна W
            → [4] жадібне розміщення по вузлах A/B/C (приріст відновлюваності на байт, частка глобального бюджету)
            → [5] payload LOG3 → AES-256-CTR → HMAC-SHA256 → replication.py (відмови вузлів)
            → репліки, що вижили → HMAC до розшифрування → декодування → дедуплікація → S, R, T, Q
[6] генетичний алгоритм налаштовує параметри тракту · [7] модуль атаки: ГА проти слабкого шифру, tamper-тест
```

## Результати

Середовище викладача (`team_config.json` T01, `replication.py`, сеанс 1200 подій / 182 критичні), середнє за
10 seeds моделі відмов, яких генетичний алгоритм не бачив:

| Сценарій | S | R | R min | Q (повна система) | Q (параметри з конфігу) |
| :-- | --: | --: | --: | --: | --: |
| штатний | 0.417 | 1.000 | 1.000 | 0.796 | 0.693 |
| деградований | 0.226 | 1.000 | 1.000 | 0.729 | 0.676 |
| аварійний | 0.226 | 1.000 | 1.000 | 0.729 | 0.655 |

- Приймальні кейси C1: **12 з 12** ([results/c1.md](results/c1.md)); усі 6 еталонних реплік викладача наш
  передавач відтворює побайтово, усі 7 пошкоджених відхиляються з очікуваним кодом.
- Перцептрон: macro-F1 0.947 на valid (вибір моделі), **0.930** на розміченому `session.csv`, який у виборі
  не брав участі; F1_min = 0.56. Без вікна історії — 0.35 ([матриця помилок](results/confusion_matrix.md)).
- ГА проти random search за однакового бюджету — [таблиця 6](results/tables/06_ga_vs_random.md), графік
  [ga_convergence.png](results/ga_convergence.png). Атака: ключ слабкого шифру знайдено в 10 з 10 випадків,
  жодна з 2000 змінених реплік не прийнята ([results/attack.md](results/attack.md)).
- Усі таблиці звіту — `results/tables/`, аналіз — [docs/REPORT.md](docs/REPORT.md).

## Запуск

Потрібен Python ≥ 3.11.

```bash
python -m venv .venv
.venv/Scripts/activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Пакет викладача кладуть у `materials/` (у git його нема: у `team_config.json` пароль сеансу, а репозиторій
публічний):

```
materials/team_config.json  replication.py  session.csv  session.bin  train.csv  valid.csv
materials/reference/ref_*.bin|json  bad_*.bin  expected_failures.json
materials/FORMAT_SPEC_P3.md
```

Без `materials/` усе працює на `config/team_config.dev.json` і dev-моделі `log3/devsim.py` (так іде CI);
кейси, яким потрібні дані викладача, пропускаються.

| Що | Команда |
| :-- | :-- |
| Повний прогін у трьох сценаріях (одна команда, методичка 11.68) | `python -m log3.pipeline` |
| Те саме на кількох seeds моделі відмов | `python -m log3.pipeline --seeds 1 2 3 4 5` |
| Параметри з конфігу замість параметрів ГА | `python -m log3.pipeline --no-ga` |
| Навчити перцептрон | `python -m log3.nn.train` |
| ГА проти random search | `python -m log3.ga.search` |
| Сценарій атаки | `python -m log3.attack` |
| Повне відтворення всіх результатів | `python scripts/reproduce.py` |
| Тести і лінтер | `python -m pytest -q` · `ruff check .` |

`scripts/reproduce.py` по черзі навчає перцептрон, запускає ГА, атаку, приймальні кейси C1 і генерує всі
таблиці звіту в `results/` (≈ 15 хв на 20 ядрах; `--skip-train --skip-ga` бере збережені модель і параметри).

## Структура

```
log3/
  records.py     запис 16 B (FORMAT_SPEC §1), CSV
  huffman.py     канонічний Гаффман, package-merge для довжин ≤ 15
  payload.py     payload LOG3, NO_COMPRESSION, class_table, строгий розбір
  crypto.py      PBKDF2-HMAC-SHA256, AES-256-CTR, HMAC-SHA256
  replica.py     репліка: заголовок 43 B | шифротекст | HMAC 32 B, приймання в порядку §5
  recovery.py    приймання всіх реплік, дедуплікація за 16-байтовим поданням
  config.py      team_config.json → Config, сценарії, перевірки
  pipeline.py    наскрізний тракт, метрика Q, фітнес для ГА, CLI
  devsim.py      dev-замінник replication.py для CI
  planner/       knapsack.py (ДП, жадібний), scheduler.py (розміщення по вузлах)
  nn/            features.py, mlp.py (власний MLP, backprop, Adam/SGD), train.py, classify.py, data.py
  ga/            ga.py (турнір, BLX-α, мутація, елітизм), search.py (random search, експеримент)
  attack/        weak_cipher.py, key_search.py (ГА відновлює ключ), tamper.py, __main__.py (звіт)
tests/           pytest: формат, крипто, еталонні пакети, ДП проти перебору, планувальник, тракт, NN, ГА, атака
scripts/         reproduce.py
results/         C1, таблиці звіту 1–6, матриця помилок, журнал і графік ГА, атака, final_metrics.json
docs/            REPORT.md (звіт), DECISIONS.md (лог рішень)
META-USAGE.md    облік використання ШІ (обов'язковий артефакт курсу)
```

## Команда

| Учасник | Зона |
| :-- | :-- |
| Гліб Степанов ([@Galactic717](https://github.com/Galactic717)), голова | формат LOG3, захист, відновлення, інтеграція, метрика, звіт |
| [@yaroslavronenko-ux](https://github.com/yaroslavronenko-ux) | Гаффман, ДП, розміщення по вузлах |
| [@artemkovbasuk780-hub](https://github.com/artemkovbasuk780-hub) | перцептрон, генетичний алгоритм, модуль атаки |

Рішення кожної зони з обґрунтуванням — у [docs/DECISIONS.md](docs/DECISIONS.md), результати — у
[docs/REPORT.md](docs/REPORT.md). Облік використання ШІ-інструментів, якого вимагає методичка, —
у [META-USAGE.md](META-USAGE.md).
