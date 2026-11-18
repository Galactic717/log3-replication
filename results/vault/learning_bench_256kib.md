# Навчання вибору стиснення: блок 256 КіБ

Корпус 48.0 МіБ (синтетичний, `scripts/vault_corpus.py`, seed 1), 12 раундів backup -> run, блок 0.25 МіБ; середовище: Linux-6.18.44-fc-v553-x86_64-with-glibc2.39, CPU 2, Python 3.13.18.

| гілка | байтів / вхід | CPU backup, с | CPU run, с | затримка захисту p50 / max, с | черга після backup, max | restore, с | repair після втрати 10 %, с | пікова RSS, КіБ |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: |
| off | 0.6333 | 5.64 | 5.87 | 1.68 / 3.99 | 478 | 1.03 | 0.97 | 84032 |
| defaults | 0.6333 | 6.18 | 5.99 | 1.72 / 4.02 | 478 | 1.11 | 0.97 | 85816 |
| accelerated | 0.6333 | 7.06 | 6.10 | 1.64 / 3.89 | 478 | 1.06 | 1.01 | 85828 |

**defaults проти off:** байти +0.00%, CPU backup +9.6%, CPU run +2.0%, затримка захисту p50 +0.038 с, restore +0.080 с, repair +0.000 с.

**accelerated проти off:** байти +0.00%, CPU backup +25.1%, CPU run +3.9%, затримка захисту p50 -0.044 с, restore +0.030 с, repair +0.037 с.

## defaults: overhead і цикл

Ознаки 162 мкс на блок; проби 0.44 CPU-с/ГіБ проти кодування 1.37 CPU-с/ГіБ; висновок 0.0 мкс на блок; fallback 0. Рішень у журналі 1214, з пробами 66, помилок навчання 0. Підсумковий режим: **baseline**, активна модель —.

| раунд | train | обрано | виграш на test | evaluate (виграш у shadow, рядків) | promote | health | CPU навчання, с |
| --: | :-- | :-- | --: | :-- | :-- | :-- | --: |
| 0 | insufficient_data | — | — | — | — | not_active | — |
| 1 | insufficient_data | — | — | — | — | not_active | — |
| 2 | insufficient_data | — | — | — | — | not_active | — |
| 3 | insufficient_data | — | — | — | — | not_active | — |
| 4 | insufficient_data | — | — | — | — | not_active | — |
| 5 | insufficient_data | — | — | — | — | not_active | — |
| 6 | insufficient_data | — | — | — | — | not_active | — |
| 7 | insufficient_data | — | — | — | — | not_active | — |
| 8 | insufficient_data | — | — | — | — | not_active | — |
| 9 | insufficient_data | — | — | — | — | not_active | — |
| 10 | insufficient_data | — | — | — | — | not_active | — |
| 11 | insufficient_data | — | — | — | — | not_active | — |

## accelerated: overhead і цикл

Ознаки 157 мкс на блок; проби 6.51 CPU-с/ГіБ проти кодування 1.34 CPU-с/ГіБ; висновок 0.0 мкс на блок; fallback 0. Рішень у журналі 1214, з пробами 1214, помилок навчання 0. Підсумковий режим: **baseline**, активна модель —.

| раунд | train | обрано | виграш на test | evaluate (виграш у shadow, рядків) | promote | health | CPU навчання, с |
| --: | :-- | :-- | --: | :-- | :-- | :-- | --: |
| 0 | baseline_kept | fixed zstd:3 | 0.0007 | — | — | not_active | 0.041 |
| 1 | baseline_kept | fixed zstd:3 | -0.0081 | — | — | not_active | 0.034 |
| 2 | baseline_kept | fixed zstd:3 | -0.0138 | — | — | not_active | 0.039 |
| 3 | baseline_kept | baseline (фактичні рішення політики) | 0.0000 | — | — | not_active | 0.043 |
| 4 | baseline_kept | fixed zstd:3 | 0.0070 | — | — | not_active | 0.048 |
| 5 | baseline_kept | fixed zstd:3 | -0.0064 | — | — | not_active | 0.048 |
| 6 | baseline_kept | fixed zstd:3 | -0.0103 | — | — | not_active | 0.052 |
| 7 | baseline_kept | baseline (фактичні рішення політики) | 0.0000 | — | — | not_active | 0.060 |
| 8 | baseline_kept | baseline (фактичні рішення політики) | 0.0000 | — | — | not_active | 0.065 |
| 9 | baseline_kept | baseline (фактичні рішення політики) | 0.0000 | — | — | not_active | 0.066 |
| 10 | baseline_kept | fixed zstd:3 | 0.0016 | — | — | not_active | 0.075 |
| 11 | baseline_kept | fixed zstd:3 | 0.0047 | — | — | not_active | 0.080 |

### accelerated: фінальний відбір кандидатів

Частини часово відокремлені, лінії файлів не перетинаються: train 729 рядків / 180 груп, validation 116 / 42, test 99 / 45; відкинуто через перетин груп 270. Вартість — відносно оракула (1.0 — найдешевша дія для кожного блока); навчені кандидати платять overhead ознак, висновку й амортизованого навчання.

| кандидат | validation | test | test без overhead | байтів / вхід (test) | CPU кодування, с/ГіБ (test) |
| :-- | --: | --: | --: | --: | --: |
| store | 2.5589 | 1.5113 | 1.5113 | 1.0000 | 0.009 |
| fixed zstd:3 **(обрано)** | 1.0525 | 1.0314 | 1.0314 | 0.6277 | 1.755 |
| baseline (фактичні рішення політики) | 1.0655 | 1.0363 | 1.0363 | 0.6402 | 1.460 |
| costtable (типові параметри) | 1.1008 | 1.0619 | 1.0006 | 0.6258 | 1.166 |
| ridge (типові параметри) | 1.0911 | 1.0572 | 1.0025 | 0.6266 | 1.178 |
| mlp (типові параметри) | 1.0957 | 1.0607 | 1.0030 | 0.6267 | 1.188 |
| ga: costtable | 1.1279 | 1.0871 | 1.0093 | 0.6330 | 1.117 |
| random: costtable | 1.1228 | 1.0840 | 1.0093 | 0.6330 | 1.117 |

ГА проти random search за однакового бюджету (16 оцінок кожен): найкраща validation-вартість ГА 1.0006 (costtable), random search 1.0006 (costtable).

Gate: не пройдено; виграш над базою на test +0.0047 (90 % bootstrap за групами -0.0072 .. +0.0157, груп 45), поріг 0.02. Перевірки: selected_is_learned ✗, net_gain_vs_baseline_on_test ✗, bootstrap_low_above_zero ✗, feature_overhead_within_limit ✓, inference_overhead_within_limit ✓, finished_within_time_limit ✓.
