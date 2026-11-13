"""Random search — базова лінія для ГА — і експеримент «ГА проти випадкового пошуку».

    python -m log3.ga.search                     # results/ga_log.csv, ga_convergence.png, tables/06_ga_vs_random.md
    python -m log3.ga.search --pop 10 --gens 8 --repeats 3

Чесне порівняння: обидва методи отримують ОДНАКОВУ кількість обчислень фітнесу
(pop_size · generations). Фітнес — pipeline.evaluate: середній Q за n_seeds відкритими seeds
у деградованому сценарії. Обраний ГА варіант потім перевіряється на ІНШИХ seeds (holdout),
щоб побачити, чи не перенавчився ГА під seeds фітнесу.
"""
from __future__ import annotations

import argparse
import csv
import time
from typing import Callable

import numpy as np

from ..config import ROOT, load_config
from ..pipeline import Params, default_params, evaluate
from .ga import DEFAULT_BOUNDS, GAResult, _lo_hi, bounds_from_config, decode, run_ga

RESULTS = ROOT / "results"


def random_search(fitness: Callable[[Params], float], *, evaluations: int, seed: int,
                  bounds: dict[str, tuple[float, float]] | None = None,
                  block: int | None = None) -> GAResult:
    """evaluations рівномірних точок у межах генів; history — найкраще-досі після кожних
    `block` оцінок (block = pop_size ГА дає криву, порівнянну з поколіннями ГА)."""
    bounds = bounds or DEFAULT_BOUNDS
    lo, hi = _lo_hi(bounds)
    rng = np.random.default_rng(seed)
    block = block or evaluations
    best_x, best_f, history = None, -np.inf, []
    for i in range(evaluations):
        x = lo + rng.random(len(lo)) * (hi - lo)
        f = fitness(decode(x, bounds))
        if f > best_f:
            best_x, best_f = x, float(f)
        if (i + 1) % block == 0 or i + 1 == evaluations:
            history.append(best_f)
    return GAResult(decode(best_x, bounds), best_f, history, evaluations)


def _plot(path, ga: GAResult, rs: GAResult, pop: int, base_q: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = [pop * (g + 1) for g in range(len(ga.history))]
    ga_best_so_far = np.maximum.accumulate(ga.history)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, ga_best_so_far, marker="o", label="ГА: найкращий Q")
    ax.plot(x, ga.mean_history, linestyle="--", label="ГА: середній Q покоління")
    ax.plot(x[:len(rs.history)], rs.history, marker="s", label="random search: найкращий Q")
    ax.axhline(base_q, color="grey", linewidth=1, label="параметри з конфігу (без ГА)")
    ax.set_xlabel("обчислень фітнесу")
    ax.set_ylabel("фітнес (середній Q)")
    ax.set_title("Збіжність ГА проти випадкового пошуку")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def experiment(*, pop: int = 10, gens: int = 8, repeats: int = 3, scenario: str = "degraded",
               n_seeds: int = 3, fit_seed: int = 0, holdout_seed: int = 100) -> dict:
    cfg = load_config()
    bounds = bounds_from_config(cfg)

    def fitness(p: Params) -> float:
        return evaluate(p, fit_seed, scenario=scenario, n_seeds=n_seeds, cfg=cfg)

    def holdout(p: Params) -> float:
        return evaluate(p, holdout_seed, scenario=scenario, n_seeds=n_seeds, cfg=cfg)

    base = default_params(cfg)
    base_fit, base_hold = fitness(base), holdout(base)
    rows, main = [], None
    for r in range(repeats):
        t0 = time.perf_counter()
        ga = run_ga(fitness, pop_size=pop, generations=gens, seed=r, bounds=bounds)
        t_ga = time.perf_counter() - t0
        t0 = time.perf_counter()
        rs = random_search(fitness, evaluations=pop * gens, seed=r, bounds=bounds, block=pop)
        t_rs = time.perf_counter() - t0
        rows.append((r, ga, holdout(ga.best_params), t_ga, rs, holdout(rs.best_params), t_rs))
        main = main or (ga, rs)
        print(f"[ga] repeat {r}: GA {ga.best_fitness:.4f}  RS {rs.best_fitness:.4f}")

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "tables").mkdir(exist_ok=True)
    ga, rs = main
    with open(RESULTS / "ga_log.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["generation", "evaluations", "ga_best", "ga_mean", "random_best_so_far"])
        for g in range(gens):
            w.writerow([g, pop * (g + 1), f"{ga.history[g]:.6f}", f"{ga.mean_history[g]:.6f}",
                        f"{rs.history[g]:.6f}" if g < len(rs.history) else ""])
    _plot(RESULTS / "ga_convergence.png", ga, rs, pop, base_fit)

    def fmt(p: Params) -> str:
        return f"({p.window_records}, {p.no_compression_threshold:.3f}, {p.second_copy_factor:.3f})"

    ga_f = [r_[1].best_fitness for r_ in rows]
    rs_f = [r_[4].best_fitness for r_ in rows]
    lines = [
        "# Таблиця 6. Збіжність ГА проти випадкового пошуку",
        "",
        f"Згенеровано `python -m log3.ga.search --pop {pop} --gens {gens} --repeats {repeats}`.",
        f"Фітнес — `pipeline.evaluate`: середній Q, сценарій `{scenario}`, seeds {fit_seed}..{fit_seed + n_seeds - 1}. "
        f"Holdout — ті самі параметри на seeds {holdout_seed}..{holdout_seed + n_seeds - 1}, яких ГА не бачив.",
        f"Бюджет кожного методу: {pop} × {gens} = {pop * gens} обчислень фітнесу. "
        "Параметри: (window_records, no_compression_threshold, second_copy_factor).",
        "",
        "| повтор (seed) | метод | найкращий фітнес | Q holdout | параметри | час, с |",
        "| --: | :-- | --: | --: | :-- | --: |",
        f"| — | без ГА (конфіг) | {base_fit:.4f} | {base_hold:.4f} | {fmt(base)} | — |",
    ]
    for r, g_, gh, tg, s_, sh, ts in rows:
        lines.append(f"| {r} | ГА | {g_.best_fitness:.4f} | {gh:.4f} | {fmt(g_.best_params)} | {tg:.1f} |")
        lines.append(f"| {r} | random search | {s_.best_fitness:.4f} | {sh:.4f} | {fmt(s_.best_params)} | {ts:.1f} |")
    lines += [
        "",
        f"Середнє за {repeats} повторами: ГА {np.mean(ga_f):.4f} ± {np.std(ga_f):.4f}, "
        f"random search {np.mean(rs_f):.4f} ± {np.std(rs_f):.4f}.",
        "",
        f"Holdout: ГА {np.mean([r_[2] for r_ in rows]):.4f}, random search {np.mean([r_[5] for r_ in rows]):.4f}, "
        f"без ГА {base_hold:.4f}.",
        ""
        + ("**Висновок:** на seeds фітнесу ГА кращий за random search, але на holdout його параметри не кращі "
           "за конфіг — ГА перенавчився під кілька seeds фітнесу (модель відмов випадкова, а 3 seeds — малий "
           "вибір). Ліки: більше seeds у фітнесі (`--n-seeds`), відбір фіналіста за holdout, перевірка на "
           "справжньому `replication.py`."
           if np.mean([r_[2] for r_ in rows]) < base_hold else
           "**Висновок:** параметри ГА кращі за конфіг і на holdout — виграш не є перенавчанням під seeds фітнесу."),
        "",
        "Криві збіжності (повтор 0) — `results/ga_log.csv`, графік — `results/ga_convergence.png`.",
        "Фітнес шумний: T залежить від часу прогону, тому порівнювати варто середнє за повторами і holdout, "
        "а не одну цифру.",
    ]
    (RESULTS / "tables" / "06_ga_vs_random.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"ga": ga_f, "random": rs_f, "base": base_fit}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pop", type=int, default=10)
    ap.add_argument("--gens", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--scenario", default="degraded")
    ap.add_argument("--n-seeds", type=int, default=3)
    a = ap.parse_args()
    print(experiment(pop=a.pop, gens=a.gens, repeats=a.repeats, scenario=a.scenario, n_seeds=a.n_seeds))
