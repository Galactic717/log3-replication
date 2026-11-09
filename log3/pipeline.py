"""Наскрізний тракт і метрика Q (умова §4, методичка 11.68).

session -> класи (перцептрон) -> v_i -> таблиця Гаффмана вікна -> ДП r_i -> вузли й batch
        -> AES-256-CTR + HMAC -> модель реплікації (store) -> відмови
        -> репліки, що вижили -> приймання -> дедуплікація -> S, R, T, Q

    S = B_ref / (B_ref + B_stored),  B_ref = 16·N,  B_stored = 8·global_used_bytes
    R = частка критичних подій, які відновлено
    T = t_ref / (t_ref + t),  t — час повного прогону
    Q = α·S + β·R + γ·T;  Q = 0, якщо відновлювач упав або не відновив жодної критичної.

Запуск: python -m log3.pipeline [normal|degraded|emergency] [seed]
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass

from . import devsim
from .config import ROOT, Config, load_config
from .nn.classify import classify_records
from .planner.scheduler import plan
from .records import Record, load_records_csv, pack_record
from .recovery import recover
from .replica import build_replica

SCENARIOS = ("normal", "degraded", "emergency")
SESSION_CSV = ROOT / "materials" / "session.csv"


@dataclass(frozen=True)
class Params:
    window_records: int              # розмір вікна (замість replication_window_records)
    no_compression_threshold: float  # поріг переходу на NO_COMPRESSION, див. build_payload
    second_copy_factor: float        # f2 у корисності другої копії


@dataclass(frozen=True)
class RunResult:
    S: float
    R: float
    T: float
    Q: float
    b_ref: int
    b_stored: int
    recovered_critical: int
    total_critical: int
    seconds: float
    copies_assigned: int = 0         # Σ r_i від ДП
    copies_placed: int = 0           # скільки з них влізло у batch вузлів
    replicas_written: int = 0
    replicas_rejected: int = 0


def default_params(cfg: Config) -> Params:
    return Params(cfg.replication_window_records, 1.0, cfg.second_copy_factor)


def load_session(cfg: Config) -> list[Record]:
    if SESSION_CSV.exists():
        return load_records_csv(SESSION_CSV)[0]
    s = cfg.raw.get("dev_session", {"records": 960, "seed": 2026})
    return devsim.synthetic_session(int(s["records"]), int(s["seed"]))


def make_model(cfg: Config, scenario: str, seed: int):
    # ponytail: dev-модель; коли з'явиться materials/replication.py — адаптер до неї тут
    return devsim.DevReplicationModel(cfg.raw["dev_scenarios"][scenario], seed)


def run(scenario: str, seed: int, params: Params | None = None, *, use_nn: bool = True,
        use_dp: bool = True, copies_cap: int = 3, compress: bool = True,
        cfg: Config | None = None, records: list[Record] | None = None) -> RunResult:
    if scenario not in SCENARIOS:
        raise ValueError(f"сценарій {scenario!r}, очікується один із {SCENARIOS}")
    cfg = cfg or load_config()
    records = load_session(cfg) if records is None else records
    p = params or default_params(cfg)
    if p.window_records < 1:
        raise ValueError("window_records має бути ≥ 1")

    start = time.perf_counter()
    model = make_model(cfg, scenario, seed)
    # без перцептрона — однакова (фіксована) цінність класу для всіх подій
    classes = classify_records(records) if use_nn else [0] * len(records)
    salt = os.urandom(16)                    # одна сіль -> один PBKDF2 на сеанс
    assigned = placed = written = 0
    for w, lo in enumerate(range(0, len(records), p.window_records)):
        window = records[lo:lo + p.window_records]
        wp = plan(window, classes[lo:lo + p.window_records], cfg, use_dp=use_dp,
                  copies_cap=copies_cap, second_copy_factor=p.second_copy_factor,
                  compress=compress, no_compression_threshold=p.no_compression_threshold)
        assigned += sum(wp.copies)
        placed += sum(wp.placed)
        for node, idx in wp.batches.items():
            blob = build_replica([window[i] for i in idx], cfg.password,
                                 replica_group_id=w % 65536, batch_seq=written, salt=salt,
                                 compress=compress, lengths=wp.lengths,
                                 no_compression_threshold=p.no_compression_threshold)
            model.store(node, blob)
            written += 1

    try:
        recovered, rejected = recover(model.surviving(), cfg.password)
    except Exception as e:                    # відновлювач упав -> Q = 0 за умовою
        print(f"[log3] відновлювач упав: {e!r}", file=sys.stderr)
        recovered, rejected = None, written
    seconds = time.perf_counter() - start

    critical = {pack_record(r) for r in records if r.critical}
    got = {pack_record(r) for r in recovered} if recovered is not None else set()
    rec_crit = len(critical & got)
    b_ref = 16 * len(records)
    b_stored = 8 * model.global_used_bytes
    S = b_ref / (b_ref + b_stored) if b_ref + b_stored else 0.0
    R = rec_crit / len(critical) if critical else 1.0
    T = cfg.t_ref / (cfg.t_ref + seconds)
    failed = recovered is None or (critical and rec_crit == 0)
    Q = 0.0 if failed else cfg.alpha * S + cfg.beta * R + cfg.gamma * T
    return RunResult(S, R, T, Q, b_ref, b_stored, rec_crit, len(critical), seconds,
                     assigned, placed, written, rejected)


def evaluate(params: Params, seed: int, *, scenario: str = "degraded", n_seeds: int = 3,
             cfg: Config | None = None) -> float:
    """Фітнес для ГА: середній Q за n_seeds відкритими seeds (seed, seed+1, ...), щоб
    не перенавчитись під один сценарій відмов. Некоректна конфігурація -> 0 (найгірше)."""
    cfg = cfg or load_config()
    records = load_session(cfg)
    try:
        return sum(run(scenario, seed + k, params, cfg=cfg, records=records).Q
                   for k in range(n_seeds)) / n_seeds
    except (ValueError, KeyError, IndexError):
        return 0.0


if __name__ == "__main__":
    sc = sys.argv[1] if len(sys.argv) > 1 else "normal"
    sd = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    print(run(sc, sd))
