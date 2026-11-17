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
- Перцептрон: macro-F1 0.947 на validation (`valid.csv`, вибір моделі, оцінка оптимістична), **0.930** на
  розміченому `session.csv`, який у виборі не брав участі; F1_min = 0.56. Без вікна історії — 0.35
  ([матриця помилок](results/confusion_matrix.md)); ця абляція тоді прибирала й поточний severity запису,
  тож завищувала внесок історії (див. «Достовірність експериментів» нижче).
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

### Як розрізняти й відтворювати експерименти

Числа вище та закомічені `results/` отримано на пакеті викладача T01. Це власна оцінка на відкритих seeds,
а не офіційна перевірка на прихованих даних. Для повторення потрібні справжні `team_config.json`,
`replication.py`, `session.csv`, `train.csv`, `valid.csv` і `reference/` у `materials/`.

Нові таблиці 1–5 і C1 вказують джерела окремо: `config`, `session`, `replication` мають значення
`materials`, `dev` або, для власного конфігу, `custom`. `LOG3_CONFIG` перевизначає лише конфіг;
він не перемикає журнал чи модель реплікації. Неповний пакет дає змішане середовище, яке не можна
видавати за перевірку на повному пакеті викладача. `final_metrics.json` зберігає `environment_sources`,
`classifier_training_source`, `dev_environment` і seeds оцінки.

Для швидкого повторення таблиць зі збереженими вагами й параметрами:

```bash
python scripts/reproduce.py --skip-train --skip-ga
```

Dev-перевірки виконуйте в окремому клоні без `materials/`: `python -m pytest -q` і
`python -m log3.pipeline --no-ga`. Збережені `model.npz` і `ga_params.json` походять з експериментів
на матеріалах викладача; використання їх на синтетиці не є новим dev-навчанням чи dev-пошуком ГА.
Повний `reproduce.py` перезаписує модель, параметри та результати, тому dev-прогін слід робити в окремому
клоні й не підміняти ним закомічені результати T01. Час, T і Q залежать від навантаження комп'ютера;
сіль і nonce випадкові, тому шифротексти не відтворюються побайтово між запусками. Порівнюйте також
S, R і `static_q`, що не містить шуму таймера.

### Достовірність експериментів (виправлено на етапі 2)

- Абляція «без вікна історії» лишає поточний severity запису: прибирається лише історія.
- Джерело (`materials` чи синтетичний `dev`) обліковується для кожної частини окремо
  (`data_sources` у `classifier_metrics.json`); приховане змішування справжніх і синтетичних частин
  заборонене, `--allow-mixed-sources` дозволяє його лише явно й видно у звіті.
- Набір відбору називається validation: `valid.csv` для перцептрона, seeds 101..110 для ГА. Незалежна
  перевірка — `session.csv` і таблиці звіту на третьому наборі seeds.
- Підсумкові параметри обираються рівноправно серед конфігу (база), ГА і random search за validation без
  порушень порогів C1. Попередня процедура порівнювала лише ГА з конфігом, хоча на validation random search
  був трохи кращим (0.543 проти 0.520, `docs/REPORT.md`).
- Закомічені `results/` (F1, Q, `ga_params.json`) отримано старим кодом на пакеті викладача T01; нові
  процедури оновлять їх під час `python scripts/reproduce.py` з `materials/`. Без пакета викладача їх не
  перераховано, щоб не підміняти результати T01 синтетикою.
- F1 і Q описують навчальний журнал подій LOG3. Вони нічого не кажуть про якість резервного копіювання
  документів чи їхню бізнесову важливість у файловому режимі.

Чинна S використовує `b_ref = 16·N` у байтах і `b_stored = 8·global_used_bytes` у бітах.
Формулу збережено до уточнення визначення `B_ref` у методичці; її не слід трактувати як звичайну
частку економії байтів. Деталі й відкрите питання — у `docs/DECISIONS.md` за 2026-11-14.

## Файловий режим (LOG3 Vault)

Окремий від навчального тракту продукт у `log3/vault/`: захищене резервне копіювання й реплікація
довільних файлів (PDF, DOCX, архіви, виконувані — будь-які байти) без навчальної вибірки. Навчальний
LOG3 v1 вище не змінено.

```bash
export LOG3_VAULT_PASSPHRASE='довга парольна фраза' LOG3_VAULT_REPO=./vault-repo
python -m log3.vault init --min-copies 1
python -m log3.vault backend add disk --type localdir --path /mnt/other-disk/log3 --domain other-disk
python -m log3.vault backup ~/Documents --wait     # код 0 лише коли рівень protected
python -m log3.vault restore latest ./restored
```

Тракт: читання → блоки → стиснення (Store / Zstandard / Гаффман LOG3) → AES-256-GCM → стійкий spool і
черга в SQLite → реплікація на локальний каталог і віддалений вузол (HTTPS, токен, закріплений
сертифікат) → перевірка → відновлення через staging. Стан версії явний: `accepted_local` не означає
`protected`. Команди, модель станів, гарантії, формати, межі й обмеження —
[docs/FILE_MODE.md](docs/FILE_MODE.md); машинозчитуваний опис етапу — [stage1.json](stage1.json).
Точка розширення для навчання — `CompressionAdvisor` у `log3/vault/advisor.py`.

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
  vault/         файловий режим: формат об'єктів і маніфест, крипто, кодеки, каталог і черга SQLite,
                 backend-и (каталог, HTTPS-вузол), захоплення, реплікація, verify/repair/gc, restore, CLI
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
