# Навчання вибору стиснення: блок 1 МіБ (типовий)

Корпус 96.0 МіБ (синтетичний, `scripts/vault_corpus.py`, seed 1), 12 раундів backup -> run, блок 1.0 МіБ; середовище: Linux-6.18.44-fc-v553-x86_64-with-glibc2.39, CPU 2, Python 3.13.18.

| гілка | байтів / вхід | CPU backup, с | CPU run, с | затримка захисту p50 / max, с | черга після backup, max | restore, с | repair після втрати 10 %, с | пікова RSS, КіБ |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: |
| off | 0.6279 | 7.09 | 5.68 | 1.54 / 2.83 | 328 | 1.31 | 0.79 | 83736 |
| defaults | 0.6279 | 7.96 | 5.77 | 1.58 / 2.83 | 328 | 1.36 | 0.79 | 85572 |
| accelerated | 0.6279 | 11.34 | 6.38 | 1.63 / 2.97 | 328 | 1.37 | 0.80 | 86280 |

**defaults проти off:** байти +0.00%, CPU backup +12.3%, CPU run +1.7%, затримка захисту p50 +0.033 с, restore +0.052 с, repair +0.009 с.

**accelerated проти off:** байти +0.00%, CPU backup +59.9%, CPU run +12.4%, затримка захисту p50 +0.086 с, restore +0.066 с, repair +0.017 с.

## defaults: overhead і цикл

Ознаки 179 мкс на блок; проби 0.46 CPU-с/ГіБ проти кодування 1.42 CPU-с/ГіБ; висновок 0.0 мкс на блок; fallback 0. Рішень у журналі 1043, з пробами 68, помилок навчання 0. Підсумковий режим: **baseline**, активна модель —.

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

Ознаки 186 мкс на блок; проби 6.73 CPU-с/ГіБ проти кодування 1.41 CPU-с/ГіБ; висновок 10.7 мкс на блок; fallback 0. Рішень у журналі 1043, з пробами 1043, помилок навчання 0. Підсумковий режим: **shadow**, активна модель —.

| раунд | train | обрано | виграш на test | evaluate (виграш у shadow, рядків) | promote | health | CPU навчання, с |
| --: | :-- | :-- | --: | :-- | :-- | :-- | --: |
| 0 | insufficient_data | — | — | — | — | not_active | — |
| 1 | insufficient_data | — | — | — | — | not_active | — |
| 2 | insufficient_data | — | — | — | — | not_active | — |
| 3 | insufficient_data | — | — | — | — | not_active | — |
| 4 | insufficient_data | — | — | — | — | not_active | — |
| 5 | baseline_kept | ridge (типові параметри) | -0.0020 | — | — | not_active | 0.041 |
| 6 | baseline_kept | baseline (фактичні рішення політики) | 0.0000 | — | — | not_active | 0.042 |
| 7 | baseline_kept | baseline (фактичні рішення політики) | 0.0000 | — | — | not_active | 0.071 |
| 8 | baseline_kept | mlp (типові параметри) | 0.0326 | — | — | not_active | 0.079 |
| 9 | baseline_kept | mlp (типові параметри) | 0.0191 | — | — | not_active | 0.076 |
| 10 | candidate | ridge (типові параметри) | 0.0266 | — | — | not_active | 0.066 |
| 11 | candidate | ridge (типові параметри) | 0.0202 | rejected (+0.0013, 126) | — | not_active | 0.061 |

### accelerated: фінальний відбір кандидатів

Частини часово відокремлені, лінії файлів не перетинаються: train 626 рядків / 230 груп, validation 53 / 46, test 63 / 54; відкинуто через перетин груп 301. Вартість — відносно оракула (1.0 — найдешевша дія для кожного блока); навчені кандидати платять overhead ознак, висновку й амортизованого навчання.

| кандидат | validation | test | test без overhead | байтів / вхід (test) | CPU кодування, с/ГіБ (test) |
| :-- | --: | --: | --: | --: | --: |
| store | 1.8310 | 1.5253 | 1.5253 | 1.0000 | 0.004 |
| fixed zstd:3 | 1.0417 | 1.0311 | 1.0311 | 0.6173 | 1.879 |
| baseline (фактичні рішення політики) | 1.0493 | 1.0446 | 1.0446 | 0.6349 | 1.597 |
| costtable (типові параметри) | 1.0245 | 1.0245 | 1.0002 | 0.6175 | 1.225 |
| ridge (типові параметри) **(обрано)** | 1.0224 | 1.0231 | 1.0008 | 0.6175 | 1.238 |
| mlp (типові параметри) | 1.0318 | 1.0262 | 1.0037 | 0.6175 | 1.299 |
| ga: ridge | 1.0250 | 1.0261 | 1.0012 | 0.6175 | 1.246 |
| random: costtable | 1.0286 | 1.0287 | 1.0002 | 0.6175 | 1.225 |

ГА проти random search за однакового бюджету (16 оцінок кожен): найкраща validation-вартість ГА 1.0003 (ridge), random search 1.0002 (costtable).

Gate: пройдено; виграш над базою на test +0.0205 (90 % bootstrap за групами +0.0031 .. +0.0395, груп 54), поріг 0.02. Перевірки: selected_is_learned ✓, net_gain_vs_baseline_on_test ✓, bootstrap_low_above_zero ✓, feature_overhead_within_limit ✓, inference_overhead_within_limit ✓, finished_within_time_limit ✓.
