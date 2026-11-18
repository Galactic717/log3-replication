# LOG3 Vault — продуктові вимірювання (2026-11-18)

Машина: 12th Gen Intel(R) Core(TM) i7-12700H, 20 логічних CPU, Windows-10-10.0.26300-SP0, Python 3.11.9.

Дані: 104 файлів, 114.4 МіБ (синтетичні проєкти + змішаний каталог). Сховище, backend-каталог і вузол — на одному диску; копій 2 (каталог + вузол HTTPS).

| Показник | Значення | Ціль | Виконано |
| :-- | --: | :-- | :-- |
| plain_bytes | 119960677 |  |  |
| files | 104 |  |  |
| restore_identical_fraction | 1.0 | ≥ 1.0 | так |
| policy_satisfied_fraction | 1.0 | ≥ 1.0 | так |
| objects_under_replicated | 0 | ≤ 0 | так |
| verify_ok_fraction | 1.0 | ≥ 1.0 | так |
| stored_to_plain_per_copy | 0.5367 | ≤ 0.75 | так |
| backup_accept_seconds | 6.669 |  |  |
| backup_accept_mib_s | 17.16 | ≥ 6.0 | так |
| replicate_two_copies_seconds | 7.012 |  |  |
| time_to_protected_seconds | 6.9293623 | ≤ 20.0 | так |
| verify_seconds | 5.553 |  |  |
| restore_seconds | 11.156 |  |  |
| restore_mib_s | 10.25 | ≥ 6.0 | так |
| drill_ok | True |  |  |
| recover_catalog_seconds | 0.861 | ≤ 10.0 | так |
| cpu_seconds | backup 2.2344, run 2.75, restore 1.875 |  |  |
| peak_rss_mib | 73.9 | ≤ 200.0 | так |
| adaptation_cpu_overhead | 0.1722 | ≤ 0.6 | так |
| adaptation_wall_overhead | 0.4794 |  |  |

Цілі перевірено для цього комп'ютера. Числа описують цей прогін на одному диску й не є універсальною продуктивністю.
