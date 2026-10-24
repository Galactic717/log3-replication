# Контракти між модулями

Джерело правди — `materials/FORMAT_SPEC_P3.md` і `materials/REPLICATION_SPEC.md`. Конфлікт із цим файлом → права специфікація, розбіжність у `DECISIONS.md`.
Зміна сигнатури — PR у `dev` із апрувом голови та власника зачепленого модуля.
Загальне: Python ≥ 3.11, type hints, усе випадкове приймає `seed`, цілі big-endian. Назви полів конфігу бери з `materials/team_config.json`.

## 1. `log3/records.py` — Гліб

```python
@dataclass(frozen=True)
class Record:
    timestamp_ms: int   # uint32
    source_id: int      # 1..4095
    event_code: int     # uint16
    metric_value: int   # int32
    category: int       # 0..15
    severity: int       # 0..7
    critical: int       # 0 або 1
    record_flags: int   # 0..3

RECORD_SIZE = 16
def pack_record(r: Record) -> bytes: ...                       # struct ">IHHiBBBB", рівно 16 байтів
def unpack_record(b: bytes) -> Record: ...                     # len(b) != 16 → ValueError
def pack_records(rs: list[Record]) -> bytes: ...
def unpack_records(b: bytes) -> list[Record]: ...
def load_records_csv(path) -> tuple[list[Record], list[int] | None]: ...   # (записи, мітки класів, якщо колонка є)
```

## 2. `log3/huffman.py` — Yaroslav

```python
class HuffmanError(ValueError): ...

def code_lengths(data: bytes, max_len: int = 15) -> dict[int, int]: ...
    # symbol -> довжина. Порожні дані -> {}. Один різний символ -> {sym: 1}.
    # Дерево будується через heapq; якщо довжина > 15 — length-limited (повне дерево, символи не зливати).
def canonical_codes(lengths: dict[int, int]) -> dict[int, tuple[int, int]]: ...
    # symbol -> (code, length); стандартний алгоритм bl_count/next_code; один символ -> (0, 1).
    # Для >= 2 символів Kraft-сума рівно 1, довжини 1..15; інакше HuffmanError.
def encode(data: bytes, lengths: dict[int, int]) -> bytes: ...             # біти MSB-first, нульовий padding
def decode(bitstream: bytes, lengths: dict[int, int], symbol_count: int) -> bytes: ...
    # HuffmanError: обрізаний потік, зайвий повний байт після symbol_count символів, ненульовий padding.
def encoded_bits(data: bytes, lengths: dict[int, int]) -> int: ...         # для ваги w_i
```

## 3. `log3/payload.py` — Гліб

```python
class PayloadError(ValueError): ...                            # → E_PAYLOAD

@dataclass(frozen=True)
class ParsedPayload:
    records: list[Record]
    class_table: list[int] | None
    flags: int

def build_payload(records: list[Record], *, compress: bool = True,
                  class_table: list[int] | None = None) -> bytes: ...
def parse_payload(payload: bytes) -> ParsedPayload: ...
    # інваріант symbol_count == 16 * record_count; reserved == 0; flags bits 4-7 == 0;
    # невідоме розширення (bit0/bit1) -> PayloadError; class_table лише за flag bit2, значення 0..2.
```

## 4. `log3/crypto.py`, `log3/replica.py` — Гліб

```python
class Log3Error(Exception):
    code: str   # "E_SHORT" | "E_MAGIC" | "E_LENGTH" | "E_HMAC" | "E_PAYLOAD" | "E_CODE"

@dataclass(frozen=True)
class Header:
    replica_group_id: int; batch_seq: int; salt: bytes; nonce: bytes; ciphertext_len: int

@dataclass(frozen=True)
class Replica:
    header: Header; records: list[Record]; class_table: list[int] | None

def derive_keys(password: str | bytes, salt: bytes) -> tuple[bytes, bytes]: ...
    # PBKDF2-HMAC-SHA256, 100000 ітерацій, dkLen=64 -> (K_enc, K_mac); кеш за (password, salt)
def build_replica(records: list[Record], password: str | bytes, *, replica_group_id: int,
                  batch_seq: int, salt: bytes | None = None, nonce: bytes | None = None,
                  compress: bool = True, class_table: list[int] | None = None) -> bytes: ...
def parse_replica(blob: bytes, password: str | bytes) -> Replica: ...
    # порядок: E_SHORT -> E_MAGIC/версія -> E_LENGTH -> HMAC (hmac.compare_digest) -> розшифрування -> payload.
    # Лише Log3Error; жодних інших винятків назовні.
```

## 5. `log3/recovery.py`, `log3/pipeline.py` — Гліб

```python
def recover(blobs: list[bytes], password: str | bytes) -> tuple[list[Record], int]: ...
    # (унікальні записи за 16-байтовим поданням, відсортовані за timestamp_ms; кількість відхилених реплік)

@dataclass(frozen=True)
class Params:
    window_records: int; no_compression_threshold: float; second_copy_factor: float

@dataclass(frozen=True)
class RunResult:
    S: float; R: float; T: float; Q: float
    b_ref: int; b_stored: int; recovered_critical: int; total_critical: int; seconds: float

def run(scenario: str, seed: int, params: Params | None = None, *, use_nn: bool = True,
        use_dp: bool = True, copies_cap: int = 3, compress: bool = True) -> RunResult: ...
    # scenario: "normal" | "degraded" | "emergency"; перемикачі потрібні для абляцій у звіті
def evaluate(params: Params, seed: int) -> float: ...      # Q — фітнес для ГА
```

## 6. `log3/planner/` — Yaroslav

```python
# knapsack.py — багатоваріантний рюкзак, r_i ∈ {0,1,2,3}
def solve_dp(util: list[list[float]], cost: list[list[int]], budget: int) -> list[int]: ...
    # util[i][r], cost[i][r] = r * w_i; максимізує Σ util, Σ cost <= budget; O(n·W·Rmax)
def solve_greedy(util, cost, budget) -> list[int]: ...     # наступна копія за найбільшим Δutil/Δcost
def record_weights(records: list[Record], lengths: dict[int, int]) -> list[int]: ...   # w_i = Σ довжин 16 байтів
def utility_table(values: list[float], second_factor: float, third_factor: float) -> list[list[float]]: ...
    # u(0)=0, u(1)=v, u(2)=v*f2, u(3)=v*f3 (коефіцієнти кумулятивні)
def base_value(cls: int, critical: int, cfg) -> float: ...  # priority(class) + critical_bonus * critical

# scheduler.py — вузли й batch
def plan_window(records: list[Record], classes: list[int], cfg) -> dict[str, list[int]]: ...
    # вузол -> індекси записів. Копії одного запису — на РІЗНИХ вузлах (r=2 два вузли, r=3 — A, B, C);
    # на вузол не більше одного batch на вікно; запис у batch не більше одного разу; batch <= max_batch_bytes.
```

## 7. `log3/nn/`, `log3/ga/`, `log3/attack/` — Artem

```python
# nn/classify.py
def classify_records(records: list[Record]) -> list[int]: ...   # 0 штатна, 1 аномальна, 2 перехід до інциденту
# nn/mlp.py
def macro_f1(y_true, y_pred) -> float: ...
def confusion_matrix(y_true, y_pred, n_classes: int = 3): ...

# ga/ga.py, ga/search.py
@dataclass(frozen=True)
class GAResult:
    best_params: Params; best_fitness: float; history: list[float]   # найкраще за покоління; з елітизмом не спадає
def run_ga(fitness: Callable[[Params], float], *, pop_size: int, generations: int, seed: int,
           elitism: bool = True, tournament_k: int = 3) -> GAResult: ...
def random_search(fitness: Callable[[Params], float], *, evaluations: int, seed: int) -> GAResult: ...
    # evaluations == pop_size * generations для чесного порівняння

# attack/
def ga_recover_key(known_pairs, *, seed: int): ...                      # ключ навмисно слабкого шифру; AES не атакуємо
def tamper_test(blob: bytes, password, *, trials: int, seed: int) -> dict[str, int]: ...   # скільки змін відхилено з E_HMAC
```

## Заглушки до готовності чужих модулів

| Чого немає | Заглушка |
| :-- | :-- |
| `huffman.py` | `compress=False` → прапорець `NO_COMPRESSION` |
| `planner.plan_window` | усі записи по одній копії на вузол A |
| `classify_records` | мітки з `train/valid.csv` або `[0] * len(records)` |
| `pipeline.evaluate` | тестова функція (сфера, Растригін) |
