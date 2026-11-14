"""Наскрізний тракт і метрика Q (умова §4, методичка 11.68).

session -> класи (перцептрон) -> v_i -> таблиця Гаффмана вікна -> ДП r_i -> вузли й batch
        -> AES-256-CTR + HMAC -> replication.py (store) -> відмови
        -> репліки, що вижили -> приймання -> дедуплікація -> S, R, T, Q

    S = B_ref / (B_ref + B_stored),  B_ref = 16·N байтів,  B_stored = 8·global_used_bytes бітів
    Це чинне обчислення зі змішаними одиницями; питання до умови — docs/DECISIONS.md (2026-11-14).
    R = частка критичних подій, які відновлено
    T = t_ref / (t_ref + t),  t — час повного прогону
    Q = α·S + β·R + γ·T;  Q = 0, якщо відновлювач упав або не відновив жодної критичної.

Запуск однією командою (усі три сценарії, відкритий seed моделі з конфігу):
    python -m log3.pipeline
    python -m log3.pipeline --scenario degraded --seeds 1 2 3 4 5
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, fields

from . import devsim
from .config import MATERIALS, ROOT, Config, Scenario, load_config
from .nn.classify import classify_records
from .planner.scheduler import plan
from .records import Record, load_records_csv, pack_record
from .recovery import recover
from .replica import build_replica

SCENARIOS = ("normal", "degraded", "emergency")
SESSION_CSV = MATERIALS / "session.csv"
REPLICATION_PY = MATERIALS / "replication.py"
TUNED_PARAMS = ROOT / "results" / "ga_params.json"


@dataclass(frozen=True)
class Params:
    """Параметри, які налаштовує ГА. Значення за замовчуванням — рівно умова з конфігу."""
    window_records: int              # розмір вікна реплікації
    budget_scale: float = 1.0        # W_вікна = budget_scale × replication_budget_bits
    second_copy_factor: float = 1.55  # f2 у корисності другої копії (≤ f3)
    critical_scale: float = 1.0      # множник critical_bonus у v_i (вага планувальника)
    pacing_slack: float = 1.0        # вікно може взяти до slack × рівної частки залишку глобального бюджету


@dataclass(frozen=True)
class RunResult:
    S: float
    R: float
    T: float
    Q: float
    b_ref: int                       # байти сирих записів; чинну формулу S збережено
    b_stored: int                    # біти списаних спроб запису, включно з BAD
    recovered_critical: int
    total_critical: int
    seconds: float
    copies_assigned: int = 0         # Σ r_i від ДП
    copies_placed: int = 0           # скільки з них потрапило в batch вузлів
    replicas_written: int = 0
    replicas_rejected: int = 0       # відхилені відновлювачем (пошкоджені)
    stored_ok: int = 0               # store() повернув True
    static_q: float = 0.0            # α·S + β·R (0, якщо Q = 0): Q без шуму таймера


def default_params(cfg: Config) -> Params:
    return Params(cfg.replication_window_records, 1.0, cfg.second_copy_factor, 1.0, 1.0)


def tuned_params(cfg: Config) -> Params:
    """Параметри, обрані ГА (results/ga_params.json); без файлу — параметри з конфігу."""
    if not TUNED_PARAMS.exists():
        return default_params(cfg)
    d = json.loads(TUNED_PARAMS.read_text(encoding="utf-8"))["params"]
    return Params(**{f.name: d[f.name] for f in fields(Params)})


def load_session(cfg: Config) -> list[Record]:
    if SESSION_CSV.exists():
        return load_records_csv(SESSION_CSV)[0]
    print("[log3] materials/session.csv нема — синтетичний dev-сеанс (лише для CI)", file=sys.stderr)
    return devsim.synthetic_session(1200, 2026)


def replication_class():
    """Replication з materials/replication.py викладача; без нього — dev-замінник для CI."""
    if not REPLICATION_PY.exists():
        return devsim.Replication
    spec = importlib.util.spec_from_file_location("replication", REPLICATION_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Replication


def _room(stats: dict, scenario: Scenario) -> tuple[int, dict[str, int]]:
    """(залишок глобального бюджету, залишкова місткість кожного вузла) за stats() моделі."""
    used = stats.get("вузли", {})
    room = {n: v.capacity_bytes - used.get(n, {}).get("used_bytes", 0) for n, v in scenario.nodes.items()}
    return scenario.global_budget_bytes - stats["global_used_bytes"], room


def check_classes(classes: list[int], n_records: int, n_classes: int) -> list[int]:
    if len(classes) != n_records or any(not 0 <= c < n_classes for c in classes):
        raise ValueError(f"класифікатор: потрібно {n_records} класів у 0..{n_classes - 1}")
    return classes


def run(scenario: str, seed: int | None = None, params: Params | None = None, *, use_nn: bool = True,
        use_dp: bool = True, copies_cap: int = 3, compress: bool = True, pacing: bool = True,
        cfg: Config | None = None, records: list[Record] | None = None,
        classes: list[int] | None = None, scenario_cfg: Scenario | None = None,
        salt: bytes | None = None) -> RunResult:
    """Один повний прогін. classes — готові класи (кеш для ГА); інакше їх рахує перцептрон,
    і цей час входить у t. scenario_cfg — змінений сценарій (мікрозміни захисту).
    salt — сіль сеансу; за замовчуванням нова випадкова (один PBKDF2 на прогін)."""
    cfg = cfg or load_config()
    sc = scenario_cfg or cfg.scenario(scenario)
    seed = cfg.model_seed if seed is None else seed
    records = load_session(cfg) if records is None else records
    p = params or default_params(cfg)
    if p.window_records < 1 or p.budget_scale < 0 or p.pacing_slack < 1:
        raise ValueError("window_records ≥ 1, budget_scale ≥ 0, pacing_slack ≥ 1")

    start = time.perf_counter()
    model = replication_class()(sc.as_model_dict(), seed)
    if classes is None:
        # без перцептрона — однакова (фіксована) цінність класу 0 для всіх подій (11.80)
        classes = classify_records(records) if use_nn else [0] * len(records)
    classes = check_classes(list(classes), len(records), cfg.n_classes)
    salt = salt or os.urandom(16)             # одна сіль -> один PBKDF2 на сеанс (11.51)
    budget_bits = int(cfg.replication_budget_bits * p.budget_scale)
    n_windows = -(-len(records) // p.window_records)
    assigned = placed = written = ok = 0
    for w in range(n_windows):
        lo = w * p.window_records
        window = records[lo:lo + p.window_records]
        left, room = _room(model.stats(), sc)
        # частка глобального бюджету на вікно (11.44): рівна частка залишку × slack —
        # невитрачене переходить на наступні вікна, «важке» вікно може позичити в майбутніх
        allowance = min(left, int(p.pacing_slack * left / (n_windows - w))) if pacing else left
        wp = plan(window, classes[lo:lo + p.window_records], cfg, sc.reliability,
                  budget_bits=budget_bits, room=room, allowance=allowance, use_dp=use_dp,
                  copies_cap=copies_cap, second_copy_factor=p.second_copy_factor,
                  critical_scale=p.critical_scale, compress=compress)
        assigned += sum(wp.copies)
        placed += sum(wp.placed)
        for node, idx in wp.batches.items():
            blob = build_replica([window[i] for i in idx], cfg.password, replica_group_id=w % 65536,
                                 batch_seq=written, salt=salt, compress=compress, lengths=wp.lengths)
            ok += bool(model.store(node, blob))
            written += 1

    try:
        recovered, rejected = recover(model.surviving_blobs(), cfg.password)
    except Exception as e:                    # відновлювач упав -> Q = 0 за умовою
        print(f"[log3] відновлювач упав: {e!r}", file=sys.stderr)
        recovered, rejected = None, written
    seconds = time.perf_counter() - start

    critical = {pack_record(r) for r in records if r.critical}
    got = {pack_record(r) for r in recovered} if recovered is not None else set()
    rec_crit = len(critical & got)
    b_ref = 16 * len(records)
    b_stored = 8 * model.stats()["global_used_bytes"]
    S = b_ref / (b_ref + b_stored) if b_ref + b_stored else 0.0
    R = rec_crit / len(critical) if critical else 1.0
    T = cfg.t_ref / (cfg.t_ref + seconds)
    failed = recovered is None or rec_crit == 0
    Q = 0.0 if failed else cfg.alpha * S + cfg.beta * R + cfg.gamma * T
    return RunResult(S, R, T, Q, b_ref, b_stored, rec_crit, len(critical), seconds,
                     assigned, placed, written, rejected, ok, 0.0 if failed else cfg.alpha * S + cfg.beta * R)


class FitnessCache:
    """Сеанс, класи перцептрона і сіль (отже, ключі PBKDF2) рахуються один раз —
    фітнес ГА викликає run сотні разів."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.records = load_session(cfg)
        self.classes = classify_records(self.records)
        self.salt = os.urandom(16)


SAFETY = 0.5   # штраф ГА — уже за половини дозволеної втрати критичних подій


def evaluate_detail(params: Params, seeds, *, scenarios=SCENARIOS, cfg: Config | None = None,
                    cache: FitnessCache | None = None) -> tuple[float, int]:
    """(фітнес, кількість прогонів зі справді порушеним порогом втрати критичних подій).

    Штраф −1 накладається, коли втрата перевищує SAFETY × поріг: на кількох seeds рідкісні
    катастрофічні пачки відмов можуть не трапитись, і без запасу ГА обирав «щасливі»
    параметри (вікно 256: R min 0.23 в аварійному на нових seeds)."""
    cache = cache or FitnessCache(cfg or load_config())
    if params.window_records < 1 or params.budget_scale < 0 or params.pacing_slack < 1:
        return 0.0, 0
    vals, bad = [], 0
    for sc in scenarios:
        limit = cache.cfg.scenario(sc).max_unrecovered_critical
        for sd in seeds:
            r = run(sc, sd, params, cfg=cache.cfg, records=cache.records, classes=cache.classes,
                    salt=cache.salt)
            bad += 1.0 - r.R > limit + 1e-12
            vals.append(r.static_q - (1.0 if 1.0 - r.R > SAFETY * limit + 1e-12 else 0.0))
    return sum(vals) / len(vals), bad


def evaluate(params: Params, seeds, *, scenarios=SCENARIOS, cfg: Config | None = None,
             cache: FitnessCache | None = None) -> float:
    """Фітнес для ГА: середнє α·S + β·R за сценаріями й seeds моделі відмов.

    γ·T не входить: t залежить від навантаження машини, а не від параметрів, і цей шум
    зробив би «найкращу» особину просто щасливим заміром. Поріг втрати критичних подій
    сценарію (C1 №7, №8) — жорстке обмеження із запасом SAFETY: порушення знімає 1 з прогону.
    Некоректні параметри дають 0 — найгірший фітнес; будь-яка інша помилка тракту не ховається.
    """
    return evaluate_detail(params, seeds, scenarios=scenarios, cfg=cfg, cache=cache)[0]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", choices=(*SCENARIOS, "all"), default="all")
    ap.add_argument("--seeds", type=int, nargs="*", help="seeds моделі відмов (за замовчуванням — з конфігу)")
    ap.add_argument("--no-ga", action="store_true", help="параметри з конфігу замість обраних ГА")
    ap.add_argument("--json", action="store_true", help="вивести результат у JSON")
    a = ap.parse_args(argv)
    cfg = load_config()
    params = default_params(cfg) if a.no_ga else tuned_params(cfg)
    seeds = a.seeds or [cfg.model_seed]
    out = []
    for sc in SCENARIOS if a.scenario == "all" else (a.scenario,):
        for sd in seeds:
            r = run(sc, sd, params, cfg=cfg)
            out.append({"scenario": sc, "seed": sd, **asdict(r)})
            if not a.json:
                print(f"{sc:9s} seed={sd:<7d} S={r.S:.4f} R={r.R:.4f} T={r.T:.4f} Q={r.Q:.4f} "
                      f"critical {r.recovered_critical}/{r.total_critical}  t={r.seconds:.2f}s")
    if a.json:
        print(json.dumps({"params": asdict(params), "runs": out}, ensure_ascii=False, indent=2))
    elif len(out) > 1:
        print(f"mean Q = {sum(o['Q'] for o in out) / len(out):.4f}  params = {asdict(params)}")


if __name__ == "__main__":
    main()
