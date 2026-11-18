"""Random search — базова лінія для ГА — і експеримент «ГА проти випадкового пошуку».

    python -m log3.ga.search        # results/ga_log.csv, ga_convergence.png, ga_params.json, tables/06_ga_vs_random.md

Чесне порівняння: обидва методи отримують ОДНАКОВУ кількість обчислень фітнесу
(pop_size · generations). Фітнес — pipeline.evaluate: середнє α·S + β·R за трьома
сценаріями і seeds FIT_SEEDS моделі відмов, зі штрафом за порушення порогів C1.
Обрані параметри перевіряються на ІНШИХ seeds (VALIDATION_SEEDS): офіційний Q рахують на
прихованих seeds, тож виграш лише на seeds фітнесу нічого не означає (методичка 11.93).
Ці seeds — набір ВІДБОРУ (validation), а не незалежна перевірка: за ними обирають підсумкові
параметри. Кандидати рівноправні: параметри з конфігу (база), найкращі особини ГА і найкращі
точки random search з усіх повторів. Обирається кандидат без порушень порогів C1 на validation із
найкращим validation-фітнесом; за рівності — конфіг. Незалежна перевірка — таблиці звіту на
третьому наборі seeds. Q і фітнес тут — про навчальний журнал LOG3; на роботу з документами
файлового режиму їх не переносимо.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict

import numpy as np

from ..config import ROOT, load_config
from ..pipeline import TUNED_PARAMS, FitnessCache, Params, default_params, evaluate, evaluate_detail
from .ga import DEFAULT_BOUNDS, GAResult, _lo_hi, bounds_from_config, decode, run_ga

RESULTS = ROOT / "results"
FIT_SEEDS = tuple(range(1, 21))
VALIDATION_SEEDS = tuple(range(101, 111))
_cache: FitnessCache | None = None


def _init() -> None:
    global _cache
    _cache = FitnessCache(load_config())


def fitness(p: Params) -> float:
    return evaluate(p, FIT_SEEDS, cache=_cache)


def validation(p: Params) -> tuple[float, int]:
    """(фітнес, порушень порогів C1) на seeds відбору, яких пошук не бачив."""
    return evaluate_detail(p, VALIDATION_SEEDS, cache=_cache)


def select_params(base: Params, base_val: tuple[float, int], rows: list[dict]) -> dict:
    """Рівноправний відбір серед конфігу, ГА і random search за validation (без порушень C1)."""
    cands = [{"source": "config", "repeat": None, "params": base, "validation": base_val}]
    for r in rows:
        cands.append({"source": "ga", "repeat": r["repeat"], "params": r["ga"].best_params, "validation": r["ga_val"]})
        cands.append({"source": "random", "repeat": r["repeat"], "params": r["rs"].best_params,
                      "validation": r["rs_val"]})
    ok = [c for c in cands if c["validation"][1] == 0]
    if not ok:
        return {**cands[0], "accepted": False}
    best = max(ok, key=lambda c: (c["validation"][0], c["source"] == "config"))
    return {**best, "accepted": best["source"] != "config"}


def random_search(fitness: Callable[[Params], float], *, evaluations: int, seed: int,
                  bounds: dict[str, tuple[float, float]] | None = None,
                  block: int | None = None, map_fn=map) -> GAResult:
    """evaluations рівномірних точок у межах генів; history — найкраще-досі після кожних
    `block` оцінок (block = pop_size ГА дає криву, порівнянну з поколіннями ГА)."""
    bounds = bounds or DEFAULT_BOUNDS
    lo, hi = _lo_hi(bounds)
    rng = np.random.default_rng(seed)
    xs = lo + rng.random((evaluations, len(lo))) * (hi - lo)
    fs = np.array(list(map_fn(fitness, [decode(x, bounds) for x in xs])))
    block = block or evaluations
    best_so_far = np.maximum.accumulate(fs)
    history = [float(best_so_far[min(i, evaluations) - 1]) for i in range(block, evaluations + block, block)]
    i = int(fs.argmax())
    return GAResult(decode(xs[i], bounds), float(fs[i]), history, evaluations)


def _plot(path, runs, pop: int, base_fit: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    for k, (ga, rs) in enumerate(runs):
        x = [pop * (g + 1) for g in range(len(ga.history))]
        ax.plot(x, np.maximum.accumulate(ga.history), color="tab:blue", alpha=0.9 if k == 0 else 0.35,
                marker="o" if k == 0 else None, label="ГА: найкращий" if k == 0 else None)
        ax.plot(x[:len(rs.history)], rs.history, color="tab:orange", alpha=0.9 if k == 0 else 0.35,
                marker="s" if k == 0 else None, label="random search: найкращий" if k == 0 else None)
    ga0 = runs[0][0]
    ax.plot([pop * (g + 1) for g in range(len(ga0.mean_history))], ga0.mean_history, color="tab:blue",
            linestyle="--", label="ГА: середній у поколінні (повтор 0)")
    ax.axhline(base_fit, color="grey", linewidth=1, label="параметри з конфігу")
    ax.set_xlabel("обчислень фітнесу")
    ax.set_ylabel("фітнес: середнє α·S + β·R")
    ax.set_title("Збіжність ГА проти випадкового пошуку (тонкі лінії — інші повтори)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def experiment(*, pop: int = 16, gens: int = 10, repeats: int = 3, workers: int | None = None) -> dict:
    cfg = load_config()
    bounds = bounds_from_config(cfg)
    base = default_params(cfg)
    rows = []
    with ProcessPoolExecutor(workers, initializer=_init) as ex:
        base_fit, base_val = ex.map(fitness, [base]), ex.map(validation, [base])
        base_fit, base_val = next(base_fit), next(base_val)
        for r in range(repeats):
            t0 = time.perf_counter()
            ga = run_ga(fitness, pop_size=pop, generations=gens, seed=r, bounds=bounds, map_fn=ex.map)
            t_ga = time.perf_counter() - t0
            t0 = time.perf_counter()
            rs = random_search(fitness, evaluations=pop * gens, seed=1000 + r, bounds=bounds, block=pop,
                               map_fn=ex.map)
            t_rs = time.perf_counter() - t0
            gv, sv = ex.map(validation, [ga.best_params, rs.best_params])
            rows.append({"repeat": r, "ga": ga, "ga_val": gv, "t_ga": t_ga, "rs": rs, "rs_val": sv, "t_rs": t_rs})
            print(f"[ga] repeat {r}: GA {ga.best_fitness:.4f} (validation {gv[0]:.4f}, порушень {gv[1]})  "
                  f"RS {rs.best_fitness:.4f} (validation {sv[0]:.4f}, порушень {sv[1]})  "
                  f"base {base_fit:.4f} ({base_val[0]:.4f}, {base_val[1]})")

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "tables").mkdir(exist_ok=True)
    with open(RESULTS / "ga_log.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["repeat", "generation", "evaluations", "ga_best", "ga_mean", "random_best_so_far"])
        for row in rows:
            ga, rs = row["ga"], row["rs"]
            for g in range(gens):
                w.writerow([row["repeat"], g, pop * (g + 1), f"{ga.history[g]:.6f}", f"{ga.mean_history[g]:.6f}",
                            f"{rs.history[g]:.6f}"])
    _plot(RESULTS / "ga_convergence.png", [(r_["ga"], r_["rs"]) for r_ in rows], pop, base_fit)

    sel = select_params(base, base_val, rows)
    accepted, chosen = sel["accepted"], sel["params"]
    winner = next((r_ for r_ in rows if r_["repeat"] == sel["repeat"]), None)
    TUNED_PARAMS.write_text(json.dumps({
        "source": sel["source"], "repeat": sel["repeat"],
        "params": asdict(chosen),
        "fitness_seeds": list(FIT_SEEDS), "validation_seeds": list(VALIDATION_SEEDS),
        "fitness": (winner["ga" if sel["source"] == "ga" else "rs"].best_fitness if winner else base_fit),
        "validation": sel["validation"][0], "validation_violations": sel["validation"][1],
        "config_fitness": base_fit, "config_validation": base_val[0],
        "note": "validation seeds — набір відбору; незалежна перевірка — таблиці звіту на третьому наборі seeds",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def fmt(p: Params) -> str:
        return (f"({p.window_records}, {p.budget_scale:.2f}, {p.second_copy_factor:.2f}, "
                f"{p.critical_scale:.1f}, {p.pacing_slack:.2f})")

    ga_f = [r_["ga"].best_fitness for r_ in rows]
    rs_f = [r_["rs"].best_fitness for r_ in rows]
    ga_h = [r_["ga_val"][0] for r_ in rows]
    rs_h = [r_["rs_val"][0] for r_ in rows]
    lines = [
        "# Таблиця 6. Збіжність ГА проти випадкового пошуку",
        "",
        f"Згенеровано `python -m log3.ga.search --pop {pop} --gens {gens} --repeats {repeats}`.",
        "Фітнес — середнє α·S + β·R за трьома сценаріями й seeds моделі відмов "
        f"{list(FIT_SEEDS)}; порушення порогу втрати критичних подій сценарію знімає 1. "
        f"Validation — ті самі параметри на seeds {list(VALIDATION_SEEDS)}, яких пошук не бачив; за ними "
        "відбираємо підсумкові параметри серед конфігу, ГА і random search рівноправно.",
        f"Бюджет кожного методу: {pop} × {gens} = {pop * gens} обчислень фітнесу.",
        "Параметри: (window_records, budget_scale, second_copy_factor, critical_scale, pacing_slack).",
        "",
        "| повтор | метод | фітнес | validation | порушень порогів на validation | параметри | час, с |",
        "| --: | :-- | --: | --: | --: | :-- | --: |",
        f"| — | параметри з конфігу | {base_fit:.4f} | {base_val[0]:.4f} | {base_val[1]} | {fmt(base)} | — |",
    ]
    for r_ in rows:
        lines.append(f"| {r_['repeat']} | ГА | {r_['ga'].best_fitness:.4f} | {r_['ga_val'][0]:.4f} | "
                     f"{r_['ga_val'][1]} | {fmt(r_['ga'].best_params)} | {r_['t_ga']:.0f} |")
        lines.append(f"| {r_['repeat']} | random search | {r_['rs'].best_fitness:.4f} | {r_['rs_val'][0]:.4f} | "
                     f"{r_['rs_val'][1]} | {fmt(r_['rs'].best_params)} | {r_['t_rs']:.0f} |")
    lines += [
        "",
        f"Середнє за {repeats} повторами: ГА {np.mean(ga_f):.4f} ± {np.std(ga_f):.4f} "
        f"(validation {np.mean(ga_h):.4f}), random search {np.mean(rs_f):.4f} ± {np.std(rs_f):.4f} "
        f"(validation {np.mean(rs_h):.4f}), конфіг {base_fit:.4f} (validation {base_val[0]:.4f}).",
        "",
        f"Елітизм: найкращий фітнес ГА не спадає між поколіннями в усіх повторах — "
        f"{'так' if all(np.all(np.diff(r_['ga'].history) >= -1e-12) for r_ in rows) else 'НІ'}.",
        "",
        (f"**Обрано параметри: {'ГА' if sel['source'] == 'ga' else 'random search'}** (повтор {sel['repeat']}): на "
         f"validation {sel['validation'][0]:.4f} без порушень порогів проти {base_val[0]:.4f} у конфігу. Вони записані "
         "в `results/ga_params.json`, їх бере `python -m log3.pipeline` (повна система; `--no-ga` — параметри з "
         "конфігу). Кандидата, що порушив поріг C1 хоч на одному validation-прогоні, відкидаємо навіть із вищим "
         "фітнесом: це підгонка під конкретні послідовності відмов seeds фітнесу."
         if accepted else
         "**Залишено параметри з конфігу:** жоден кандидат ГА чи random search не кращий за конфіг на validation без "
         "порушень порогів C1 — виграш на seeds фітнесу виявився підгонкою під конкретні відмови."),
        "",
        "Криві збіжності — `results/ga_log.csv`, графік — `results/ga_convergence.png`.",
    ]
    (RESULTS / "tables" / "06_ga_vs_random.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"ga": ga_f, "random": rs_f, "base": base_fit, "accepted": accepted, "params": asdict(chosen)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pop", type=int, default=16)
    ap.add_argument("--gens", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    print(experiment(pop=a.pop, gens=a.gens, repeats=a.repeats, workers=a.workers))
