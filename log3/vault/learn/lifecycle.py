"""Цикл навчання: baseline -> shadow -> оцінювання -> active -> rollback.

train      часово відокремлені train / validation / test; групи (лінії файлів) не перетинаються між
           частинами, тож версії того самого файла не завищують оцінку. Кандидати: Store, фіксований
           загальний кодек, базова евристика (фактичні рішення політики з журналу), costtable, ridge,
           власний MLP, найкращі з ГА і з random search за однакового бюджету. Відбір — лише за
           validation; test використовується один раз для фінальної перевірки обраного.
gate       обраний кандидат має бути навченим, кращим за базу на test щонайменше на min_relative_gain
           ПІСЛЯ вирахування overhead ознак, висновку й амортизованого навчання, з нижньою межею
           групового bootstrap > 0 і в межах ресурсних лімітів. Інакше база лишається активною.
shadow     кандидат рахує рекомендації в тракті, але робочі рішення не змінює; evaluate порівнює їх
           із базою на свіжих пробах і перевіряє інваріанти (помилки, fallback, карантин, overhead).
active     promote атомарно перемикає state.json; попередня справна модель стає last_good.
rollback   вручну або автоматично (health): погіршення вартості, затримки захисту чи помилки.
Навчання поступається захисту даних: не стартує під час restore/repair, за переповненої черги чи
нестачі диска; обмежене за рядками, пам'яттю, часом і пріоритетом процесу.
"""
from __future__ import annotations

import os
import shutil
import time

import numpy as np

from ...ga.ga import run_ga
from ..errors import Exit, VaultError
from ..fsutil import free_bytes
from . import features as F
from .actions import CostModel
from .journal import Journal, now_ns
from .models import (
    BaselinePolicy,
    CostTablePolicy,
    FixedPolicy,
    MLPPolicy,
    ModelError,
    Policy,
    RidgePolicy,
    load_artifact,
    save_artifact,
)
from .settings import LearnDir, load_state, priority_active, save_state, settings

FRACTIONS = (0.6, 0.8)                       # межі train | validation | test за часом
SIMPLICITY = ("store", "fixed", "baseline", "costtable", "ridge", "mlp")
LEARNED = ("costtable", "ridge", "mlp")
SEARCH_BOUNDS = {"family": (0.0, 1.999), "min_count_log2": (0.0, 7.0), "l2_log10": (-3.0, 3.0)}


def _ctx(repo) -> tuple[dict, LearnDir, dict, CostModel]:
    s = settings(repo.cfg)
    d = LearnDir(repo.path)
    st, _ = load_state(d)
    cost = CostModel(s["cost"]["copies"] or repo.policy["min_copies"], float(s["cost"]["cpu_second_equiv_bytes"]))
    return s, d, st, cost


def yield_reason(repo, s: dict) -> str | None:
    """Чому навчання зараз має поступитися основному захисту даних (None — можна)."""
    busy = priority_active(repo.path)
    if busy:
        return f"іде {', '.join(busy)}: навчання поступається"
    lim = repo.limits
    queued = repo.cat.scalar("SELECT COUNT(*) FROM jobs WHERE state IN ('PENDING','RUNNING')")
    if queued / max(1, lim["queue_max_jobs"]) > s["probe"]["max_queue_fill"]:
        return f"черга реплікації заповнена ({queued} завдань): навчання відкладено"
    if free_bytes(repo.spool) < lim["min_free_bytes"] * s["probe"]["min_free_factor"]:
        return "мало вільного місця на диску spool: навчання відкладено"
    return None


# ---- дані й розбиття ---------------------------------------------------------------------------------
def split(data: dict) -> dict:
    """Часові межі за квантилями часу рішень; група належить частині свого першого появлення.
    Рядки відомих раніше груп у пізніших частинах відкидаються (інакше версії файла «підказують» test)."""
    t, groups = data["decided_ns"], np.array(data["groups"], dtype=object)
    n = len(t)
    if n == 0:
        return {"train": np.array([], int), "validation": np.array([], int), "test": np.array([], int),
                "dropped": 0, "bounds_ns": [None, None]}
    ts = np.sort(t)
    t1, t2 = ts[min(n - 1, int(n * FRACTIONS[0]))], ts[min(n - 1, int(n * FRACTIONS[1]))]
    first: dict[str, int] = {}
    for g, ti in zip(groups, t):
        first[g] = min(first.get(g, ti), ti)
    f = np.array([first[g] for g in groups])
    tr = np.flatnonzero((t <= t1) & (f <= t1))
    va = np.flatnonzero((t > t1) & (t <= t2) & (f > t1))
    te = np.flatnonzero((t > t2) & (f > t2))
    return {"train": tr, "validation": va, "test": te, "dropped": int(n - len(tr) - len(va) - len(te)),
            "bounds_ns": [int(t1), int(t2)]}


def _provenance(data: dict, parts: dict, s: dict) -> dict:
    out = {"source": "локальні вимірювання цієї інсталяції (журнал рішень і проб)",
           "feature_schema": F.schema_id(), "rows_total": int(len(data["X"])), "rows_incomplete": data["incomplete"],
           "rows_dropped_group_overlap": parts["dropped"], "split_bounds_ns": parts["bounds_ns"],
           "quarantine_included": False, "probe_rate": s["probe"]["rate"]}
    for name in ("train", "validation", "test"):
        idx = parts[name]
        out[name] = {"rows": int(len(idx)), "groups": len({data["groups"][i] for i in idx}),
                     "from_ns": int(data["decided_ns"][idx].min()) if len(idx) else None,
                     "to_ns": int(data["decided_ns"][idx].max()) if len(idx) else None}
    return out


def _policy_idx(data: dict, base: BaselinePolicy) -> np.ndarray:
    """Фактичні рішення базової політики з журналу; якщо дії нема в наборі — правило на ознаках."""
    acts = data["actions"]
    rule = base.choose(data["X"])
    return np.array([acts.index(p) if p in acts else rule[i] for i, p in enumerate(data["policy"])], dtype=int)


def score(choice: np.ndarray, C: np.ndarray, payload: np.ndarray, cpu: np.ndarray, raw: np.ndarray,
          extra_cost: float = 0.0) -> dict:
    """Метрики політики на рядках: вартість відносно оракула, байти, CPU, суміш дій."""
    rows = np.arange(len(choice))
    chosen = C[rows, choice].sum() + extra_cost * len(choice)
    oracle = C.min(1).sum()
    return {"rows": int(len(choice)), "cost_vs_oracle": float(chosen / oracle) if oracle else None,
            "cost_per_raw_byte": float(chosen / raw.sum()) if raw.sum() else None,
            "stored_to_raw": float(payload[rows, choice].sum() / raw.sum()) if raw.sum() else None,
            "encode_cpu_seconds_per_gib": float(cpu[rows, choice].sum() / 1e9 / (raw.sum() / 2**30))
            if raw.sum() else None}


def bootstrap_gain(c_base: np.ndarray, c_cand: np.ndarray, groups: list, n: int, seed: int = 0) -> dict:
    """Груповий bootstrap частки виграшу 1 - Σcand/Σbase (групи — лінії файлів)."""
    point = 1.0 - c_cand.sum() / c_base.sum() if c_base.sum() else 0.0
    uniq, inv = np.unique(np.array(groups, dtype=object).astype(str), return_inverse=True)
    if len(uniq) < 2:
        return {"gain": float(point), "low": float(point), "high": float(point), "groups": int(len(uniq))}
    gb, gc = np.bincount(inv, c_base, len(uniq)), np.bincount(inv, c_cand, len(uniq))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(uniq), size=(n, len(uniq)))
    gains = 1.0 - gc[draws].sum(1) / gb[draws].sum(1)
    return {"gain": float(point), "low": float(np.quantile(gains, 0.05)), "high": float(np.quantile(gains, 0.95)),
            "groups": int(len(uniq))}


# ---- кандидати ----------------------------------------------------------------------------------------
def _fit(kind: str, params: dict, acts: list[str], cost: CostModel, D: dict, tr, va, C) -> Policy:
    X, pay, cpu, raw = D["X"], D["payload"], D["cpu_ns"], D["raw"]
    if kind == "costtable":
        return CostTablePolicy(acts, cost, params).fit(X[tr], pay[tr], cpu[tr], raw[tr])
    if kind == "ridge":
        return RidgePolicy(acts, cost, params).fit(X[tr], pay[tr], cpu[tr], raw[tr])
    return MLPPolicy(acts, cost, params).fit(X[tr], C[tr].argmin(1), X[va], C[va], seed=params.get("seed", 0))


def _decode(x) -> tuple[str, dict]:
    if int(x[0]) == 0:
        return "costtable", {"min_count": int(round(2 ** x[1]))}
    return "ridge", {"l2": float(10 ** x[2])}


def _search(method: str, budget: int, seed: int, fitness) -> dict:
    """ГА (власний log3.ga) або random search з однаковою кількістю обчислень фітнесу."""
    if method == "ga":
        pop = 4
        res = run_ga(fitness, pop_size=pop, generations=max(1, budget // pop), seed=seed, bounds=SEARCH_BOUNDS,
                     decode_fn=_decode)
        return {"best": res.best_params, "fitness": res.best_fitness, "evaluations": res.evaluations,
                "history": res.history}
    rng = np.random.default_rng(seed + 1000)
    lo = np.array([b[0] for b in SEARCH_BOUNDS.values()])
    hi = np.array([b[1] for b in SEARCH_BOUNDS.values()])
    best, best_f, hist = None, -np.inf, []
    for _ in range(max(1, budget // 4) * 4):
        cand = _decode(lo + rng.random(len(lo)) * (hi - lo))
        f = fitness(cand)
        if f > best_f:
            best, best_f = cand, f
        hist.append(best_f)
    return {"best": best, "fitness": best_f, "evaluations": len(hist), "history": hist[3::4]}


def _check_ready(st: dict, s: dict) -> None:
    if not s["enabled"] or st["disabled"]:
        raise VaultError("E_LEARN_DISABLED", "навчання вимкнено (learn enable)", Exit.ATTENTION)
    if st["paused"]:
        raise VaultError("E_LEARN_PAUSED", "навчання на паузі (learn resume)", Exit.ATTENTION)
    if st["frozen"]:
        raise VaultError("E_LEARN_FROZEN", f"оновлення заморожено: {st['freeze_reason']} (learn unfreeze)",
                         Exit.ATTENTION)


def train(repo, *, force: bool = False) -> dict:
    s, d, st, cost = _ctx(repo)
    _check_ready(st, s)
    why = yield_reason(repo, s)
    if why:
        raise VaultError("E_LEARN_YIELD", why, Exit.CAPACITY)
    t_cfg, gate = s["train"], s["gate"]
    if not force and now_ns() - st["last_train_ns"] < t_cfg["min_interval_seconds"] * 1e9:
        return {"status": "too_soon", "next_in_seconds": round((st["last_train_ns"] + t_cfg["min_interval_seconds"]
                                                                * 1e9 - now_ns()) / 1e9, 1)}
    if hasattr(os, "nice") and t_cfg["nice"]:
        try:
            os.nice(int(t_cfg["nice"]))                       # фоновий пріоритет процесу навчання
        except OSError:
            pass
    t0, c0 = time.perf_counter(), time.process_time()
    deadline = t0 + t_cfg["max_seconds"]
    acts = s["actions"]
    row_bytes = F.N_FEATURES * 5 + len(acts) * 16 + 64
    max_rows = int(min(t_cfg["max_rows"], t_cfg["max_memory_mib"] * 2**20 / row_bytes / 4))
    j = Journal(d.journal)
    try:
        D = j.labeled(feature_schema=F.schema_id(), actions=acts, max_rows=max_rows)
        parts = split(D)
        prov = _provenance(D, parts, s)
        tr, va, te = parts["train"], parts["validation"], parts["test"]
        if len(tr) < t_cfg["min_rows"] or prov["train"]["groups"] < t_cfg["min_groups"] or len(va) < 20 \
                or len(te) < 20:
            report = {"status": "insufficient_data", "provenance": prov,
                      "need": {"train_rows": t_cfg["min_rows"], "train_groups": t_cfg["min_groups"],
                               "validation_rows": 20, "test_rows": 20}}
            j.lifecycle("train_skipped", None, "недостатньо даних")
            return report
        C = cost.cost(D["payload"], D["cpu_ns"])
        base = BaselinePolicy(acts, cost, {"default_action": _default(repo)})
        recorded = _policy_idx(D, base)
        feat_ns = float(np.mean(D["feature_ns"])) if len(D["feature_ns"]) else 0.0
        window = j.scalar("SELECT COUNT(*) FROM decisions WHERE decided_ns BETWEEN ? AND ?",
                          (int(D["decided_ns"].min()), int(D["decided_ns"].max()))) or 1
        cands: dict[str, dict] = {}

        def overhead_of(kind: str, policy) -> dict:
            """Повний overhead навченої політики на блок: ознаки + висновок + амортизоване навчання."""
            if kind not in LEARNED:
                return {"feature_ns": 0.0, "inference_ns": 0.0, "training_cpu_seconds_per_chunk": 0.0, "bytes": 0.0}
            sample = te[:100] if len(te) else va[:100]
            t_inf = time.perf_counter_ns()
            for i in sample:
                policy.choose(D["X"][i:i + 1])
            inf = (time.perf_counter_ns() - t_inf) / max(1, len(sample))
            train_pc = (time.process_time() - c0) / window
            return {"feature_ns": feat_ns, "inference_ns": inf, "training_cpu_seconds_per_chunk": train_pc,
                    "bytes": ((feat_ns + inf) / 1e9 + train_pc) * cost.cpu_second_equiv_bytes}

        def add(name: str, kind: str, policy, params: dict | None = None) -> None:
            choice = recorded if policy is None else policy.choose(D["X"])
            ov = overhead_of(kind, policy)
            cands[name] = {"kind": kind, "policy": policy, "params": params or {}, "choice": choice, "overhead": ov,
                           "validation": score(choice[va], C[va], D["payload"][va], D["cpu_ns"][va], D["raw"][va],
                                               ov["bytes"])}

        add("store", "store", FixedPolicy(acts, cost, {"action": "store:0"}), {"action": "store:0"})
        add(f"fixed {_default(repo)}", "fixed", FixedPolicy(acts, cost, {"action": _default(repo)}),
            {"action": _default(repo)})
        add("baseline (фактичні рішення політики)", "baseline", None)
        for kind in t_cfg["kinds"]:
            if time.perf_counter() > deadline:
                break
            params = {"mlp": {"hidden": [32], "epochs": 30, "seed": t_cfg["seed"]}}.get(kind, {})
            add(f"{kind} (типові параметри)", kind, _fit(kind, params, acts, cost, D, tr, va, C), params)

        def fitness(cand) -> float:
            kind, params = cand
            p = _fit(kind, params, acts, cost, D, tr, va, C)
            ch = p.choose(D["X"][va])
            return -float(C[va][np.arange(len(va)), ch].sum() / C[va].min(1).sum())

        search = {}
        for method in ("ga", "random"):
            if time.perf_counter() > deadline:
                break
            res = _search(method, t_cfg["search_budget"], t_cfg["seed"], fitness)
            kind, params = res["best"]
            search[method] = {"best_kind": kind, "best_params": params, "validation_cost_vs_oracle": -res["fitness"],
                              "evaluations": res["evaluations"], "history": [-h for h in res["history"]]}
            add(f"{method}: {kind}", kind, _fit(kind, params, acts, cost, D, tr, va, C), params)

        order = {k: i for i, k in enumerate(SIMPLICITY)}
        name = min(cands, key=lambda n: (round(cands[n]["validation"]["cost_vs_oracle"], 6),
                                         order[cands[n]["kind"]]))
        sel = cands[name]
        overhead = {**sel["overhead"], "training_cpu_seconds": time.process_time() - c0,
                    "decisions_in_data_window": int(window)}
        test = {n: score(c["choice"][te], C[te], D["payload"][te], D["cpu_ns"][te], D["raw"][te],
                         c["overhead"]["bytes"]) | {"kind": c["kind"]} for n, c in cands.items()}
        cb = C[te, recorded[te]]
        cc = C[te, sel["choice"][te]] + sel["overhead"]["bytes"]
        boot = bootstrap_gain(cb, cc, [D["groups"][i] for i in te], gate["bootstrap"], t_cfg["seed"])
        checks = {
            "selected_is_learned": sel["kind"] in LEARNED,
            "net_gain_vs_baseline_on_test": boot["gain"] >= gate["min_relative_gain"],
            "bootstrap_low_above_zero": boot["low"] > 0,
            "feature_overhead_within_limit": feat_ns <= gate["max_feature_ns_per_chunk"],
            "inference_overhead_within_limit": sel["overhead"]["inference_ns"] <= gate["max_inference_ns_per_chunk"],
            "finished_within_time_limit": time.perf_counter() <= deadline,
        }
        passed = all(checks.values())
        report = {"status": "candidate" if passed else "baseline_kept", "selected": name,
                  "selected_kind": sel["kind"], "gate": {"passed": passed, "checks": checks, **boot,
                                                         "min_relative_gain": gate["min_relative_gain"]},
                  "validation": {n: c["validation"] for n, c in cands.items()}, "test": test, "search": search,
                  "overhead": overhead, "provenance": prov, "cost_model": cost.as_dict(),
                  "resources": {"wall_seconds": round(time.perf_counter() - t0, 3),
                                "cpu_seconds": round(time.process_time() - c0, 3),
                                "peak_rss_kib": _peak_rss_kib(),
                                "rows_cap": max_rows}}
        model_id = None
        if sel["kind"] in LEARNED:
            model_id = save_artifact(d.models, sel["policy"], {
                "status": "candidate" if passed else "rejected", "created_ns": now_ns(),
                "config": {k: s[k] for k in ("actions", "cost", "train", "gate")},
                "provenance": prov, "metrics": {"validation": sel["validation"], "test": test[name],
                                                "gate": report["gate"], "overhead": overhead}})
            report["model_id"] = model_id
        st, _ = load_state(d)
        st["last_train_ns"] = now_ns()
        if passed:
            st.update(candidate_model=model_id, candidate_set_ns=now_ns(), candidate_eval=None)
            if st["mode"] == "baseline":
                st["mode"] = "shadow"
        save_state(d, st, event="train", detail=f"{report['status']}: {name}")
        j.lifecycle("train", model_id, f"{report['status']}; gain={boot['gain']:.4f} low={boot['low']:.4f}")
        _prune_models(d, st, s["auto"]["max_models"])
        return report
    finally:
        j.close()


def _peak_rss_kib() -> int | None:
    try:
        import resource  # лише POSIX
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except (ImportError, OSError):
        return None


def _default(repo) -> str:
    c = repo.cfg["compression"]
    return f"{c['codec']}:{c['level'] if c['codec'] != 'store' else 0}"


def _prune_models(d: LearnDir, st: dict, keep: int) -> None:
    if not d.models.is_dir():
        return
    pinned = {st.get("active_model"), st.get("candidate_model"), st.get("last_good_model")}
    items = sorted((p for p in d.models.iterdir() if p.is_dir() and not p.name.startswith(".")),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    for p in items[keep:]:
        if p.name not in pinned:
            shutil.rmtree(p, ignore_errors=True)


# ---- shadow, promote, rollback, health -------------------------------------------------------------
def evaluate(repo) -> dict:
    """Перевірка кандидата в shadow на свіжих пробах: виграш над базою та інваріанти."""
    s, d, st, cost = _ctx(repo)
    cand = st["candidate_model"]
    if not cand:
        return {"status": "no_candidate"}
    try:
        load_artifact(d.models, cand, admissible=s["actions"])
    except ModelError as e:
        st.update(candidate_model=None, candidate_eval={"passed": False, "reason": str(e)})
        save_state(d, st, event="candidate_invalid", detail=str(e))
        return {"status": "candidate_invalid", "reason": str(e)}
    j = Journal(d.journal)
    try:
        D = j.labeled(feature_schema=F.schema_id(), actions=s["actions"], since_ns=st["candidate_set_ns"],
                      max_rows=s["train"]["max_rows"])
        keep = np.array([m == cand and a in s["actions"] for m, a in zip(D["shadow_model"], D["shadow"])], bool)
        n_shadow = j.scalar("SELECT COUNT(*) FROM decisions WHERE shadow_model=?", (cand,)) or 0
        errors = repo.cat.scalar("SELECT COALESCE(SUM(value),0) FROM events WHERE kind='learning_errors' AND ts_ns>?",
                                 (st["candidate_set_ns"],))
        quarantined = j.scalar("SELECT COUNT(*) FROM decisions WHERE shadow_model=? AND quarantine=1", (cand,)) or 0
        inf = j.scalar("SELECT AVG(inference_ns) FROM decisions WHERE shadow_model=?", (cand,)) or 0.0
        feat = j.scalar("SELECT AVG(feature_ns) FROM decisions WHERE decided_ns>?", (st["candidate_set_ns"],)) or 0.0
    finally:
        j.close()
    gate = s["gate"]
    if keep.sum() < min(gate["min_shadow_decisions"], 20) or n_shadow < gate["min_shadow_decisions"]:
        return {"status": "collecting", "shadow_decisions": int(n_shadow), "probed_shadow_rows": int(keep.sum()),
                "need": gate["min_shadow_decisions"]}
    acts = s["actions"]
    C = cost.cost(D["payload"], D["cpu_ns"])[keep]
    rows = np.arange(len(C))
    base = _policy_idx({**D, "X": D["X"][keep], "policy": [p for p, k in zip(D["policy"], keep) if k]},
                       BaselinePolicy(acts, cost, {"default_action": _default(repo)}))
    shadow = np.array([acts.index(a) for a, k in zip(D["shadow"], keep) if k])
    extra = (feat + inf) / 1e9 * cost.cpu_second_equiv_bytes
    boot = bootstrap_gain(C[rows, base], C[rows, shadow] + extra, [g for g, k in zip(D["groups"], keep) if k],
                          gate["bootstrap"], s["train"]["seed"])
    checks = {"net_gain_vs_baseline": boot["gain"] >= gate["min_relative_gain"], "bootstrap_low_above_zero":
              boot["low"] > 0, "no_learning_errors": errors == 0, "no_quarantine": quarantined == 0,
              "inference_overhead_within_limit": inf <= gate["max_inference_ns_per_chunk"],
              "feature_overhead_within_limit": feat <= gate["max_feature_ns_per_chunk"]}
    result = {"passed": all(checks.values()), "checks": checks, **boot, "probed_shadow_rows": int(len(C)),
              "shadow_decisions": int(n_shadow), "inference_ns_per_chunk": inf, "feature_ns_per_chunk": feat,
              "evaluated_ns": now_ns()}
    st, _ = load_state(d)
    st["candidate_eval"] = {"model_id": cand, **result}
    save_state(d, st, event="evaluate", detail=f"{cand}: {'passed' if result['passed'] else 'failed'}")
    _log(d, "evaluate", cand, f"passed={result['passed']} gain={boot['gain']:.4f}")
    return {"status": "evaluated", "model_id": cand, **result}


def promote(repo) -> dict:
    s, d, st, _ = _ctx(repo)
    cand, ev = st["candidate_model"], st.get("candidate_eval") or {}
    if not cand or ev.get("model_id") != cand or not ev.get("passed"):
        raise VaultError("E_LEARN_GATE", "кандидат не пройшов перевірку в shadow (learn evaluate)", Exit.ATTENTION)
    if st["frozen"]:
        raise VaultError("E_LEARN_FROZEN", f"оновлення заморожено: {st['freeze_reason']}", Exit.ATTENTION)
    try:
        load_artifact(d.models, cand, admissible=s["actions"])
    except ModelError as e:
        raise VaultError("E_LEARN_MODEL", f"кандидат не завантажується: {e}", Exit.ATTENTION) from None
    prev = st["active_model"] if st["mode"] == "active" else None
    st.update(mode="active", active_model=cand, candidate_model=None, candidate_eval=None,
              last_good_model=prev or st.get("last_good_model"), active_since_ns=now_ns())
    save_state(d, st, event="promote", detail=cand)
    _log(d, "promote", cand, f"попередня: {prev}")
    return {"status": "active", "active_model": cand, "last_good_model": st["last_good_model"]}


def rollback(repo, reason: str = "вручну") -> dict:
    """До останньої справної моделі, якщо вона завантажується, інакше — до базової політики."""
    s, d, st, _ = _ctx(repo)
    bad, target = st["active_model"], st.get("last_good_model")
    if target and target != bad:
        try:
            load_artifact(d.models, target, admissible=s["actions"])
        except ModelError:
            target = None
    else:
        target = None
    st.update(mode="active" if target else "baseline", active_model=target, last_good_model=None,
              active_since_ns=now_ns() if target else 0)
    save_state(d, st, event="rollback", detail=f"{bad} -> {target or 'baseline'}: {reason}")
    _log(d, "rollback", bad, f"-> {target or 'baseline'}: {reason}")
    return {"status": st["mode"], "rolled_back": bad, "active_model": target, "reason": reason}


def health(repo) -> dict:
    """Нагляд за активною моделлю: фактична вартість проти бази на свіжих пробах, помилки, затримка
    підтвердження захисту. Погіршення -> автоматичний rollback."""
    s, d, st, cost = _ctx(repo)
    if st["mode"] != "active" or not st["active_model"]:
        return {"status": "not_active"}
    since, mid = st["active_since_ns"], st["active_model"]
    j = Journal(d.journal)
    try:
        D = j.labeled(feature_schema=F.schema_id(), actions=s["actions"], since_ns=since, max_rows=20000)
        lat_active = [r[0] for r in j.q("SELECT s.value FROM stages s JOIN operations o ON o.version_id=s.subject "
                                        "WHERE s.stage='policy_satisfied' AND o.mode='active' AND o.model_id=?",
                                        (mid,))]
        lat_base = [r[0] for r in j.q("SELECT s.value FROM stages s JOIN operations o ON o.version_id=s.subject "
                                      "WHERE s.stage='policy_satisfied' AND o.mode='baseline' "
                                      "ORDER BY o.finished_ns DESC LIMIT 20")]
    finally:
        j.close()
    errors = repo.cat.scalar("SELECT COALESCE(SUM(value),0) FROM events WHERE kind='learning_errors' AND ts_ns>?",
                             (since,))
    out = {"status": "healthy", "active_model": mid, "learning_errors": errors, "probed_rows": int(len(D["X"]))}
    reasons = []
    if errors:
        reasons.append(f"помилки навчання в тракті: {errors}")
    if len(D["X"]) >= 50:
        acts = s["actions"]
        C = cost.cost(D["payload"], D["cpu_ns"])
        rows = np.arange(len(C))
        base = _policy_idx(D, BaselinePolicy(acts, cost, {"default_action": _default(repo)}))
        chosen = np.array([acts.index(a) if a in acts else base[i] for i, a in enumerate(D["chosen"])])
        realized = 1.0 - C[rows, chosen].sum() / C[rows, base].sum()
        out["realized_gain_vs_baseline"] = float(realized)
        if realized < -s["gate"]["min_relative_gain"]:
            reasons.append(f"фактична вартість гірша за базу на {-realized:.1%}")
    if len(lat_active) >= 3 and len(lat_base) >= 3:
        a, b = float(np.median(lat_active)), float(np.median(lat_base))
        out["protect_latency_seconds"] = {"active_p50": a, "baseline_p50": b}
        if a > b * (1 + s["gate"]["max_protect_latency_increase"]) + 1.0:
            reasons.append(f"затримка підтвердження захисту зросла: {a:.1f} с проти {b:.1f} с")
    if reasons:
        out["status"] = "rolled_back"
        out["rollback"] = rollback(repo, "; ".join(reasons))
    return out


def _log(d: LearnDir, kind: str, model_id: str | None, detail: str) -> None:
    j = Journal(d.journal)
    try:
        j.lifecycle(kind, model_id, detail)
    finally:
        j.close()


# ---- керування ----------------------------------------------------------------------------------------
def set_flags(repo, event: str, **flags) -> dict:
    _, d, st, _ = _ctx(repo)
    st.update(flags)
    st = save_state(d, st, event=event)
    _log(d, event, st.get("active_model"), str(flags))
    return {k: st[k] for k in ("mode", "paused", "disabled", "frozen", "freeze_reason", "active_model",
                               "candidate_model", "last_good_model")}


def set_mode(repo, mode: str) -> dict:
    """baseline — завжди можна; shadow — потрібен кандидат; active — лише через promote."""
    _, d, st, _ = _ctx(repo)
    if mode == "active":
        raise VaultError("E_LEARN_GATE", "active вмикається лише через learn promote після перевірки", Exit.USAGE)
    if mode == "shadow" and not st["candidate_model"]:
        raise VaultError("E_LEARN_GATE", "нема кандидата для shadow (learn train)", Exit.ATTENTION)
    keep = st["active_model"] if mode == "baseline" else None
    return set_flags(repo, f"mode_{mode}", mode=mode, active_model=None,
                     last_good_model=keep or st.get("last_good_model"))


def unfreeze(repo, *, accept_window: bool = False) -> dict:
    """Зняти заморожування. accept_window — оператор явно визнає карантинні операції нормою."""
    _, d, _, _ = _ctx(repo)
    accepted = 0
    if accept_window and d.journal.exists():
        j = Journal(d.journal)
        try:
            accepted = j.db.execute("UPDATE decisions SET quarantine=0 WHERE quarantine=1").rowcount
            j.db.execute("UPDATE operations SET quarantine=0 WHERE quarantine=1")
        finally:
            j.close()
    return {**set_flags(repo, "unfreeze", frozen=False, freeze_reason=None), "accepted_decisions": accepted}


def reset(repo) -> dict:
    """Холодний старт: журнал, моделі й стан навчання видаляються. Каталог, spool і копії не чіпаються."""
    d = LearnDir(repo.path)
    existed = d.root.exists()
    if existed:
        shutil.rmtree(d.root)
    return {"reset": existed, "mode": "baseline"}


def collect(repo) -> dict:
    """Переносить у журнал затримані стадії захисту з каталогу (ідемпотентно)."""
    d = LearnDir(repo.path)
    if not d.journal.exists():
        return {"harvested": {}}
    j = Journal(d.journal)
    try:
        return {"harvested": j.harvest(repo.cat)}
    finally:
        j.close()


def auto(repo) -> dict:
    """Один крок циклу (викликається після run): стадії -> health -> оцінювання/promote -> навчання."""
    s, d, st, _ = _ctx(repo)
    out: dict = {}
    if not s["enabled"] or st["disabled"]:
        return {"status": "disabled"}
    out["collect"] = collect(repo)
    out["health"] = health(repo)
    if not s["auto"]["enabled"] or st["paused"]:
        return out | {"status": "paused" if st["paused"] else "auto_off"}
    st, _ = load_state(d)
    if st["candidate_model"] and not st["frozen"]:
        out["evaluate"] = evaluate(repo)
        if out["evaluate"].get("passed") and s["auto"]["promote"]:
            out["promote"] = promote(repo)
    st, _ = load_state(d)
    if not st["frozen"] and not st["candidate_model"]:
        try:
            rep = train(repo)
            out["train"] = {k: rep.get(k) for k in ("status", "selected", "model_id", "gate", "next_in_seconds")}
        except VaultError as e:
            out["train"] = {"status": e.code, "message": e.message}
    return out
