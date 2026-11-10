# META-USAGE — облік використання ШІ

Обов'язковий артефакт проєкту. Один рядок на сесію роботи з агентом. Не видаляти й не переписувати чужі рядки.

| Дата | Учасник (GitHub) | Агент / модель | Що просили | Що прийнято | Що відхилено / переписано | Людина перевірила й розуміє (так/ні) |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| 2026-11-09 | Galactic717 | Claude Code (Claude Opus 5.5) | Виконати зону FMT/SEC/INT за `docs/tasks/Galactic717.md`; тимчасово закрити зони, без яких пайплайн не працює | `records`, `config`, `payload`, `crypto`, `replica`, `recovery`, `pipeline`, тести C1 №1–3, 7, 8, 10–12, fuzz, CI, `reproduce.py`; тимчасові `huffman`, `planner`, заглушка `classify`; dev-конфіг і `devsim` | Власний AES не писали (бібліотека `cryptography`, див. DECISIONS); `classify_records` лишено заглушкою — нема `train.csv` | на перевірці — Гліб має прочитати код і пройти питання захисту з `docs/tasks/Galactic717.md` |

| 2026-11-10 | yaroslavronenko-ux | Claude Code (Claude Sonnet 5.5) | Проаналізувати репо, скласти план; написати власний `huffman.py` (heapq, канонічні коди, encode/decode, package-merge) з тестами | Увесь `huffman.py` і `tests/test_huffman.py`, усі тести проходять | Код Гліба замінено власним; евристику обмеження довжин замінено на package-merge |