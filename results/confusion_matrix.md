# Матриця помилок перцептрона (valid)

Згенеровано `python -m log3.nn.train`. Дані: **dev** (dev-набір `log3/nn/data.py`, не `materials/valid.csv` — перенавчити, коли буде пакет викладача).

Класи: 0 — штатна, 1 — аномальна, 2 — перехід до інциденту. F1_min = 0.6.
Архітектура: 31 → 32 → 16 → 3, ReLU, softmax, adam, lr=0.002, L2=0.0001, вікно історії H=16.

| seed | macro-F1 (valid) |
| --: | --: |
| 0 | 0.9129 |
| 1 | 0.9083 |
| 2 | 0.9167 |

Обрано seed=2: macro-F1 = **0.9167**, accuracy = 0.9307.

Без вікна історії (лише ознаки самого запису, той самий seed): macro-F1 = 0.7227.
Різниця — внесок часових ознак; найбільше вона видна на класі 2.

| справжній \ передбачений | 0 | 1 | 2 |
| :-- | --: | --: | --: |
| 0 | 2018 | 1 | 100 |
| 1 | 20 | 230 | 0 |
| 2 | 81 | 6 | 544 |

| клас | precision | recall | F1 | підтримка |
| --: | --: | --: | --: | --: |
| 0 | 0.952 | 0.952 | 0.952 | 2119 |
| 1 | 0.970 | 0.920 | 0.945 | 250 |
| 2 | 0.845 | 0.862 | 0.853 | 631 |

Ознаки: `severity`, `metric_slog`, `critical`, `record_flags`, `code_slog`, `cat_0`, `cat_1`, `cat_2`, `cat_3`, `cat_4`, `cat_5`, `cat_6`, `cat_7`, `cat_8`, `cat_9`, `cat_10`, `cat_11`, `cat_12`, `cat_13`, `cat_14`, `cat_15`, `delta_metric`, `events_per_window`, `category_changes`, `source_changes`, `severity_mean`, `log_dt`, `distinct_sources`, `code_family_run`, `dt_trend`, `metric_std`.
