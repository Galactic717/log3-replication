# LOG3 Vault — встановлення, служба, оновлення, моніторинг

Підтримувані ОС: Windows 10/11 і Linux (x86-64), Python ≥ 3.11 (CI: Linux 3.11–3.13, Windows 3.11).
macOS не перевірялась. Залежності й ліцензії — [DEPENDENCIES.md](DEPENDENCIES.md).

## 1. Встановлення

**З релізних артефактів (офлайн).** Реліз збирається один раз із мережею (`python scripts/build_release.py
--platform manylinux2014_x86_64`), далі встановлення мережі не потребує:

```bash
python scripts/build_release.py --verify dist          # SHA256SUMS: кожен файл, зайвих нема (код 7 при підміні)
python -m venv /opt/log3                               # Windows: py -3.11 -m venv C:\log3\venv
/opt/log3/bin/python -m pip install --no-index --find-links dist/wheelhouse dist/log3-3.0.0-py3-none-any.whl
/opt/log3/bin/log3-vault --version
```

Без скрипта: Linux `cd dist && sha256sum -c SHA256SUMS`; PowerShell — `Get-FileHash -Algorithm SHA256`.

**З вихідного коду (розробка):** `pip install -e .[dev]` або `pip install -r requirements.txt` і
`python -m log3.vault ...`.

## 2. Налаштування

```bash
export LOG3_VAULT_REPO=/var/lib/log3-vault
export LOG3_VAULT_PASSPHRASE_FILE=/etc/log3/passphrase      # chmod 600; або LOG3_VAULT_PASSPHRASE
log3-vault init --min-copies 2
log3-vault backend add disk --type localdir --path /mnt/usb/log3 --domain usb-disk
log3-vault backend add node --type node --url https://backup.example.org:8443 --ca-file node.crt --domain office-2
log3-vault config check && log3-vault doctor
LOG3_VAULT_EXPORT_PASSPHRASE=... log3-vault key export --out /media/offline/keys.export
log3-vault emergency-kit --out /media/offline/kit       # автономний restore поруч із копіями
```

Вузол (інша машина):

```bash
log3-vault node-cert --dir /etc/log3 --host backup.example.org
log3-vault node-token add laptop --role writer --tokens-file /etc/log3/node-tokens.json --out laptop.token
log3-vault node-token add ops --role admin --tokens-file /etc/log3/node-tokens.json --out ops.token
log3-vault serve-node --data /srv/log3-node --cert /etc/log3/node.crt --key /etc/log3/node.key \
    --tokens-file /etc/log3/node-tokens.json --port 8443
```

Клієнт реплікації отримує `writer`-токен (видаляти копії він не може); `gc` запускається з `admin`-токеном.

## 3. Служба

Черга працює без парольної фрази (`run --daemon` передає вже зашифровані об'єкти):

- Linux: `deploy/systemd/log3-vault.service` (клієнт) і `log3-node.service` (вузол) →
  `/etc/systemd/system/`, `systemctl daemon-reload && systemctl enable --now log3-vault`.
- Windows: `powershell -ExecutionPolicy Bypass -File deploy\windows\install-service.ps1 -Exe
  C:\log3\venv\Scripts\log3-vault.exe -Repo C:\ProgramData\log3-vault -TokenFile C:\ProgramData\log3\node.token`
  (завдання планувальника від SYSTEM під час старту, перезапуск після збою; журнал `service.log`).

Аварійне завершення служби безпечне: зафіксовані транзакції цілі, завислі завдання повертаються після
`lease_seconds`. Перевірки за розкладом: `log3-vault verify` і `log3-vault drill` (cron / планувальник).

## 4. Оновлення й міграції

1. `log3-vault migrate status` — поточна схема каталогу; зупинити службу.
2. Перевірити `SHA256SUMS` нового релізу, встановити колесо поверх (`pip install --no-index ...`).
3. Перший запуск нової версії сам мігрує каталог: спершу повна копія в `<repo>/backups/catalog-schemaN-*.sqlite`,
   потім одна транзакція. `log3-vault doctor`, запустити службу.

**Невдале оновлення.** Аварія посеред міграції відкочує транзакцію: схема лишається попередньою, стара
версія працює далі (етап `update` модуля захисту перевіряє це справжньою версією етапу 2 з git). Якщо нова
версія непридатна вже після міграції: `log3-vault migrate restore --file <копія> --yes` (новою версією),
потім встановити попереднє колесо. Версії, прийняті після міграції, повертає `repair`/`recover`; стара
версія новішу схему не відкриває (`E_SCHEMA`), щоб не пошкодити її мовчки.

## 5. Аварійне відновлення

| Що втрачено | Дія |
| :-- | :-- |
| копія на одному backend-і | `log3-vault verify` → `log3-vault repair` |
| локальний диск (каталог, spool) | `log3-vault recover disk --type localdir --path /mnt/usb/log3 --domain usb-disk --repo NEW` → `repair` |
| інсталяція або вся машина | `python log3_restore.py --backend /mnt/usb/log3 list` / `restore latest DEST` (фраза або `--key-export`) |

## 6. Моніторинг

| Що бачить оператор | Команда / поле |
| :-- | :-- |
| місткість і вік черги | `queue`, `metrics`: `queue_jobs`, `queue_capacity_jobs`, `queue_oldest_job_age_seconds`, `queue_fill_ratio` |
| недостатні репліки | `status`: `under_replicated_versions`; `status VERSION`: `objects_under_replicated` |
| вільне місце | `spool_disk_free_bytes`, `spool_bytes` / `spool_capacity_bytes` |
| затримка захисту | `protection_lag_seconds`, `avg_time_to_protected_seconds` |
| помилки | `job_errors_total`, `job_errors_last_hour`, `queue_attention_jobs`, `queue --limit N` (останні помилки) |
| остання перевірка | `last_verify_age_seconds`, `last_verify_problems` |
| restore drill | `last_drill_ok`, `last_drill_age_seconds`, `last_drill` |

`log3-vault health` — машинозчитувані перевірки з порогами (код 0 або 4); пороги — секція `monitoring` у
`config.json` або прапорці `--max-queue-age-seconds` тощо. `log3-vault metrics --format prometheus --out
/var/lib/node_exporter/textfile/log3.prom` — текстовий формат Prometheus без web-панелі (textfile collector);
мітки лише зі станів і назв backend-ів, без шляхів, id файлів і секретів.
