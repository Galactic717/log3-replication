# dev_data — синтетичні дані для налагодження (НЕ пакет викладача)

`dev_train.csv` і `dev_valid.csv` згенеровано командою `python -m log3.nn.data`
(генератор `dev_labeled_session` у `log3/nn/data.py`, seeds фіксовані — файли відтворюються).

- Формат той самий, що очікує `records.load_records_csv`: поля `Record` + колонка `label`
  (0 — штатна, 1 — аномальна, 2 — перехід до інциденту).
- Це **не** `materials/train.csv` / `valid.csv`. Не копіюйте їх у `materials/`: код вважає файли
  в `materials/` даними викладача, і `test_c1_09` тоді «пройде» на вигаданих даних.
- Коли прийде пакет викладача — кладіть його в `materials/` і перенавчайте: `python -m log3.nn.train`.
  Якщо назви колонок у справжньому `train.csv` інші — правка в `records.LABEL_COLUMNS` (Гліб).
