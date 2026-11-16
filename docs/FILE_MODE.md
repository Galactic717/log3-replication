# LOG3 Vault — файловий режим (етап 1)

Захищене резервне копіювання й реплікація довільних файлів. Працює одразу після встановлення, без
навчальної вибірки й без web-інтерфейсу. Живе в `log3/vault/` і не змінює навчальний LOG3 v1
(`log3.pipeline`, формат реплік v1, ДП, перцептрон, ГА, атака): з v1 імпортується лише `log3.huffman`
як експериментальний кодек.

Машинозчитуваний опис цього етапу — [`stage1.json`](../stage1.json).

## Швидкий старт

Один комп'ютер, дві незалежні копії: каталог на іншому диску і віддалений вузол.

```bash
pip install -r requirements.txt

# секрети — лише змінні середовища або файли (ЗМІННА_FILE=шлях), не аргументи
export LOG3_VAULT_PASSPHRASE='довга парольна фраза'      # PowerShell: $env:LOG3_VAULT_PASSPHRASE = '...'
export LOG3_NODE_TOKEN='довгий випадковий токен'
export LOG3_VAULT_REPO=/var/lib/log3-vault                # замість --repo в кожній команді

# на вузлі зберігання (інша машина)
python -m log3.vault node-cert --dir certs --host backup.example.org
python -m log3.vault serve-node --data /srv/log3-node --cert certs/node.crt --key certs/node.key --port 8443

# на машині замовника (node.crt скопіювати з вузла; node.key лишається на вузлі)
python -m log3.vault init
python -m log3.vault backend add disk --type localdir --path /mnt/usb/log3 --domain usb-disk
python -m log3.vault backend add node --type node --url https://backup.example.org:8443 \
       --ca-file certs/node.crt --domain office-2
python -m log3.vault backup ~/Documents --wait            # 0 лише коли рівень protected
python -m log3.vault status
python -m log3.vault restore latest /tmp/restored
```

## Команди

| Команда | Що робить |
| :-- | :-- |
| `init [--min-copies N]` | створює сховище; парольна фраза з `LOG3_VAULT_PASSPHRASE` |
| `backend add NAME --type localdir\|node --domain D ...` · `backend list` | backend-и та їхні фактичні домени відмов |
| `backup SRC [--wait] [--priority low\|normal\|high] [--op-id ID] [--name N]` | приймає версію; без `--wait` лише ставить у чергу |
| `run [--once] [--timeout S]` | обробляє чергу реплікації (секрет не потрібен) |
| `list` · `status [VERSION]` | версії, рівень захисту, показники |
| `verify [VERSION]` | перечитує копії на backend-ах: довжина + SHA-256 |
| `restore VERSION DEST [--path P] [--overwrite]` | відновлення через staging; `VERSION` — id, префікс або `latest` |
| `repair [--all-backends] [--no-run]` | після аварій і відмов повертає копії до політики |
| `gc [--apply]` | прибирає версії після retention; без `--apply` лише звіт |
| `recover NAME --type ... --domain D` | відтворює сховище з backend-а після втрати локального диска |
| `serve-node` · `node-cert` | віддалений вузол і його сертифікат |

Кожна команда приймає `--repo DIR` і `--json` (один JSON-об'єкт у stdout: `ok`, `command`, `exit_code`,
`exit_name`, далі дані).

### Коди завершення (стабільні)

| Код | Назва | Значення |
| --: | :-- | :-- |
| 0 | OK | запитаний рівень досягнуто (`backup` без `--wait`: прийнято локально й поставлено в чергу) |
| 1 | ERROR | інша помилка (зокрема `E_EXISTS`, `E_SYMLINK`, `E_COLLISION` у restore) |
| 2 | USAGE | некоректні аргументи |
| 3 | NOT_PROTECTED | прийнято локально, але політику копій ще не виконано |
| 4 | ATTENTION | версія DEGRADED або бюджет повторів вичерпано |
| 5 | CAPACITY | відмова до стійкого приймання: черга, spool або вільне місце |
| 6 | POLICY | backend-и не можуть виконати політику копій |
| 7 | INTEGRITY | не пройдено перевірку автентичності, довжини чи контрольної суми |
| 8 | KEY | секрет відсутній або не той |
| 9 | PARTIAL | версію прийнято, але частину файлів джерела пропущено |
| 10 | NOT_FOUND | нема такої версії, сховища чи backend |

## Модель станів

```
CAPTURING ──► ACCEPTED_LOCAL ──► REPLICATING ──► PROTECTED
    │               │                 │            │  ▲
    ▼               └────────────────►▼            ▼  │ repair
  FAILED                           DEGRADED ◄──────┘──┘        PROTECTED/DEGRADED/FAILED ──► DELETED (gc)
```

| Стан | Рівень у JSON | Що гарантовано |
| :-- | :-- | :-- |
| `CAPTURING` | `capturing` | нічого |
| `ACCEPTED_LOCAL` | `accepted_local` | усі блоки й маніфест стійко лежать у локальному spool і каталозі; джерело більше не потрібне. **Політику копій не виконано.** |
| `REPLICATING` | `replicating` | те саме; черга вже передає об'єкти |
| `PROTECTED` | `protected` | кожен блок **і маніфест** мають підтверджені копії щонайменше в `min_copies` різних доменах відмов |
| `DEGRADED` | `degraded` | політику не виконано, автоматичних спроб не лишилось; прийняті дані доступні для restore |
| `FAILED` | `failed` | версію не прийнято |

Переходи задані таблицею `states.TRANSITIONS`, змінює стан лише `Catalog.set_state`; заборонений перехід —
помилка. Поле `policy_satisfied` у JSON дорівнює `true` тільки в стані `PROTECTED`.

### Що означає підтвердження

- **«Прийнято локально»** — зафіксовано транзакцію публікації (нижче). Це не копія: spool лежить на
  тому самому диску, що й каталог.
- **«Відправлено»** станом не є. Завдання стає `DONE`, а копія `CONFIRMED` лише після того, як backend
  повернув підтвердження з ідентифікатором, довжиною і SHA-256 об'єкта, і всі три збіглися з поточним
  об'єктом (`backends.base.check_ack`).
- **Підтвердження backend** надсилається після запису на рівні, який backend документує в полі
  `durability`: для каталогу й вузла це `fsync(file) + rename + fsync(dir)` на POSIX; у Windows —
  `FlushFileBuffers` + атомарне перейменування (каталог синхронізує журнал NTFS).

## Порядок запису й гарантії стійкості

Захоплення версії:

1. рядок версії `CAPTURING`;
2. кожен новий блок: tmp-файл у spool → `fsync` → `rename`;
3. маніфест так само;
4. `fsync` каталогів spool;
5. **одна транзакція SQLite**: об'єкти, посилання версії, завдання черги, стан `ACCEPTED_LOCAL`.
   Її `COMMIT` і є публікацією.

SQLite налаштовано явно: `journal_mode=WAL`, `synchronous=FULL` (кожен `COMMIT` робить `fsync` журналу),
`BEGIN IMMEDIATE` для кожної зміни, `wal_checkpoint(TRUNCATE)` після публікації і наприкінці роботи
черги, `integrity_check` у `repair`. Сам WAL гарантії при втраті живлення не дає — її дає `synchronous=FULL`.

| Подія | Що відбувається | Чим перевірено |
| :-- | :-- | :-- |
| Аварія процесу | зафіксовані транзакції цілі; незавершене захоплення стає `FAILED`, файли-сироти прибирає `repair`; завислі `RUNNING` повертаються в чергу після `lease_seconds` | тести з `os._exit` у нових процесах |
| Втрата живлення | те саме **за умови**, що ОС і носій чесно виконують `fsync` (диски з увімкненим кешем запису без захисту, деякі USB-накопичувачі й мережеві ФС цього не гарантують) | **не перевірялось фізично**: тести аварії процесу не є тестом втрати живлення |
| Фізична втрата локального диска | каталог і spool втрачено; версії в стані `PROTECTED` відновлюються з будь-якого backend-а командою `recover` (потрібна парольна фраза). Версії, що не дійшли до `PROTECTED`, можуть бути втрачені | тест `test_recover_after_local_disk_loss` |
| Втрата backend-а | лишається щонайменше `min_copies − 1` копій; `verify` показує нестачу, `repair` відновлює з цілої копії | тест відмови, повернення й пошкодження вузла |

## Формати

**Об'єкт** (блок або маніфест), big-endian:
`"L3FB" | format_version u8 | kind u8 | codec u8 | key_id 8B | nonce 12B | plain_len u64 | AES-256-GCM(payload) | tag 16B`.
AAD — заголовок і `object_id`. `object_id = HMAC-SHA256(K_id, kind ‖ відкриті байти)`.

**Кодеки** (`codecs.REGISTRY`, id у заголовку об'єкта): `0 store`, `1 zstd`, `2 huffman-log3`
(канонічний Гаффман LOG3, експериментальний і повільний). Якщо payload не коротший за дані — блок
зберігається як Store. Декодування залежить лише від заголовка об'єкта і ключа; ML-модель для restore
не потрібна й не викликається.

**Маніфест** — JSON `log3-file-manifest` версії 1 (схема в докстрингу `log3/vault/manifest.py`):
`entries` (шлях, тип, `file_id`, `content_id`, розмір, `mtime_ns`, права), `contents` (порядок блоків,
довжина, SHA-256 вмісту), `objects` (довжини, SHA-256 зашифрованого об'єкта, кодек), `crypto`, `codecs`,
`skipped`. Ідентичності розділено: `content_id` (вміст), `file_id` (файл між версіями і перейменуваннями),
шлях, `version_id`. Час — наносекунди від 1970-01-01 UTC плюс RFC 3339.

**Ключі.** `KeyProvider` (`cryptobox.py`) — інтерфейс; реалізація етапу 1 — `PassphraseKeys`: scrypt
(n=2^15, r=8, p=1) → майстер-ключ → HKDF → ключ даних, ключ ідентифікаторів, контрольне значення `key_id`.
У `config.json` лежать лише параметри KDF, сіль і `key_id`.

## Конфігурація (`<repo>/config.json`, без секретів)

```json
{
  "format": "log3-vault-config", "format_version": 1, "repo_id": "…", "chunk_size_bytes": 1048576,
  "crypto": {"suite": "AES-256-GCM", "kdf": {"name": "scrypt", "salt_b64": "…", "n": 32768, "r": 8, "p": 1},
             "key_id": "…", "passphrase_env": "LOG3_VAULT_PASSPHRASE"},
  "compression": {"codec": "zstd", "level": 3, "advisor": null},
  "policy": {"min_copies": 2, "extra_copies_high_priority": 1, "retention_days": 30, "keep_last": 3},
  "limits": {"queue_max_jobs": 200000, "spool_max_bytes": 21474836480, "min_free_bytes": 268435456,
             "max_parallel": 4, "max_attempts": 8, "backoff_base_seconds": 2.0, "backoff_cap_seconds": 600.0,
             "lease_seconds": 900.0, "capture_stale_seconds": 300.0},
  "backends": [
    {"name": "disk", "type": "localdir", "path": "/mnt/usb/log3", "failure_domain": "usb-disk"},
    {"name": "node", "type": "node", "url": "https://backup.example.org:8443", "ca_file": "/etc/log3/node.crt",
     "token_env": "LOG3_NODE_TOKEN", "failure_domain": "office-2"}
  ]
}
```

Поля `token`, `password`, `secret`, `passphrase` у backend-і відхиляються при завантаженні.

## Політики

- **Копії.** `min_copies` — жорстка вимога: якщо backend-и дають менше доменів відмов, `backup`
  відмовляє з кодом 6 ще до читання джерела. Локальні каталоги на одному пристрої (`st_dev`) зливаються
  в один домен, хоч би як їх назвали.
- **Пріоритет** впливає на порядок (зважений round-robin 4 : 2 : 1 на пакет, тож низький пріоритет не
  голодує) і на додаткові копії (`extra_copies_high_priority` для `high`, якщо є вільні домени).
  Планування — обмежений пакет на ітерацію; ДП навчального тракту тут не використовується.
- **Переповнення.** Досягнуто `queue_max_jobs`, `spool_max_bytes` або `min_free_bytes` → явна відмова
  з кодом 5 до стійкого приймання; нові файли невдалої спроби прибираються. Прийняті версії та
  оригінали ніколи не видаляються, щоб звільнити місце.
- **Повтори.** `job_id = uuid5(repo_id, oid:backend)` — стабільний між перезапусками. Затримка
  `min(cap, base·2^(n−1)) · U(0.5, 1.5)`; після `max_attempts` завдання стає `ATTENTION`, версія `DEGRADED`.
- **Retention і GC.** Версія видаляється, коли минув `retention_days` і вона не входить у `keep_last`
  найновіших свого набору. GC не чіпає незавершені версії, об'єкти з активними завданнями і маніфести
  живих версій.

## Точка розширення для навчання (етап 2)

`log3/vault/advisor.py`:

```python
class CompressionAdvisor(Protocol):
    def recommend(self, context: CompressionContext) -> Recommendation: ...
    def observe(self, outcome: CompressionOutcome) -> None: ...
```

Підключення: `"compression": {"advisor": "пакет.модуль:Клас"}` у `config.json`. `CompressionContext`
містить лише відоме до рішення (розширення, розмір, номер і довжина блока, перші 16 байтів, ентропія
вибірки, типовий кодек, `operation_id`, `version_id`); `CompressionOutcome` — фактичний результат
(запитаний і фактичний кодек, довжини, час стиснення). Виняток, невідомий кодек чи відсутній модуль →
працює `BaselineAdvisor`; backup не зупиняється, restore advisor не викликає. Advisor не має доступу до
видалення, ключів, retention і кількості копій. Приклад робочого advisor — у тесті
`test_huffman_codec_and_custom_advisor_hook`.

## Відомі обмеження

- Втрату живлення фізично не перевірено; гарантія спирається на `fsync` ОС і носія.
- Узгоджене резервування активних баз даних **не заявляється**: файл, що змінюється під час читання,
  перечитується до 3 разів і потім пропускається (код 9). Потрібен snapshot або інтеграція із застосунком.
- Блоки фіксованого розміру: вставка байтів на початку великого файла зсуває всі наступні блоки.
- Маніфест версії цілком тримається в пам'яті (≈ 150–200 Б на блок і на файл).
- Символічні посилання, junction і спеціальні файли джерела пропускаються із записом у `skipped`;
  власники, ACL і розширені атрибути не зберігаються (лише права POSIX і `mtime`).
- Один ключ даних на сховище; ротація й обгортання ключів — через нову реалізацію `KeyProvider`.
  Випадковий nonce AES-GCM обмежує один ключ ≪ 2³² об'єктами.
- Покажчики версій і `repo-config.json` лежать на backend-ах відкрито (ідентифікатори, час, шляхи
  backend-ів; без секретів і без імен файлів). Конфігурація туди потрапляє разом із маніфестом версії.
- Домен відмов мережевого вузла задає оператор; два розділи одного фізичного диска мають різні
  `st_dev` і автоматично не розпізнаються.
- Публікація restore атомарна для кожного файла, але не для всього набору. Перевірка symlink
  виконується до запису і ще раз перед перенесенням; повністю гонку з локальним зловмисником це не усуває.
- Вузол — однопроцесний HTTPS-сервер зі спільним токеном; S3-сумісного backend-а нема.
- CI і всі перевірки виконано на Linux; код для Windows написано (шляхи, `fsync`, reparse points),
  але в CI не перевіряється.
