"""Повне відтворення results/ однією командою:  python scripts/reproduce.py

Кроки (кожен детермінований за seeds; --skip-train / --skip-ga беруть уже збережені
log3/nn/model.npz і results/ga_params.json):
  1. python -m log3.nn.train   -> log3/nn/model.npz, results/confusion_matrix.md, classifier_metrics.json
  2. python -m log3.ga.search  -> results/ga_params.json, ga_log.csv, ga_convergence.png, tables/06
  3. python -m log3.attack     -> results/attack.md
  4. pytest -k test_c1_        -> results/c1.md
  5. таблиці звіту 1–5         -> results/tables/0[1-5]_*.md, results/final_metrics.json

Числа таблиць 2, 3, 5 — середнє за EVAL_SEEDS моделі відмов: відкритий seed із конфігу
і 9 інших, яких ГА не бачив ні у фітнесі, ні в holdout.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
import zlib
from dataclasses import replace
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from log3 import huffman  # noqa: E402
from log3.config import MATERIALS_CONFIG, load_config  # noqa: E402
from log3.payload import build_payload  # noqa: E402
from log3.pipeline import SCENARIOS, FitnessCache, default_params, run, tuned_params  # noqa: E402
from log3.planner.knapsack import solve_dp, solve_greedy, total, utility_table  # noqa: E402
from log3.planner.scheduler import choose_copies  # noqa: E402
from log3.records import pack_records  # noqa: E402
from log3.replica import HEADER_SIZE, MIN_SIZE, build_replica  # noqa: E402

RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
NAMES = {"normal": "штатний", "degraded": "деградований", "emergency": "аварійний"}

C1 = [
    (1, "Один запис кодується й відновлюється побайтово"),
    (2, "Порожній журнал не спричиняє аварійного завершення"),
    (3, "NO_COMPRESSION відновлює сирі записи точно"),
    (4, "Канонічні коди відновлюються тільки з довжин"),
    (5, "Вироджений алфавіт має код довжини 1"),
    (6, "ДП дає корисність не меншу за жадібний baseline на всіх виданих наборах"),
    (7, "У штатному сценарії відновлено всі критичні події"),
    (8, "У деградованому сценарії частка невідновлених критичних ≤ порогу"),
    (9, "Перцептрон: macro-F1 ≥ F1_min (valid і session; holdout рахує викладач)"),
    (10, "Змінений шифротекст або HMAC відхиляється"),
    (11, "Невірний пароль, обрізана репліка, поганий magic і reserved — без аварії"),
    (12, "Усі еталонні пакети читаються; повний прогін у ліміті часу"),
]


def env_note() -> str:
    if MATERIALS_CONFIG.exists():
        return "Середовище: пакет викладача (`materials/`: team_config.json, replication.py, session.csv)."
    return ("**Середовище: dev-конфіг і dev-модель реплікації (`log3/devsim.py`) — пакета викладача в "
            "`materials/` нема. Числа ілюструють методику, не офіційний Q.**")


def fmt(v) -> str:
    return f"{v:.4f}" if isinstance(v, float) else str(v)


def table(title: str, head: list[str], rows: list[list], note: str = "") -> str:
    lines = [f"# {title}", "", env_note(), ""] + ([note, ""] if note else [])
    lines += ["| " + " | ".join(head) + " |", "|" + " :-- |" * len(head)]
    lines += ["| " + " | ".join(fmt(v) for v in r) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def step(cmd: list[str]) -> None:
    print(f"[reproduce] {' '.join(cmd)}", flush=True)
    subprocess.run([sys.executable, *cmd], cwd=ROOT, check=True)


def avg_runs(scenario: str, cache: FitnessCache, seeds, *, cfg=None, classes=None, **kw) -> dict:
    rs = [run(scenario, s, cfg=cfg or cache.cfg, records=cache.records, salt=cache.salt,
              classes=cache.classes if classes is None else classes, **kw) for s in seeds]
    keys = ("S", "R", "T", "Q", "static_q", "b_stored", "copies_assigned", "copies_placed",
            "replicas_written", "stored_ok", "seconds")
    return {k: mean(getattr(r, k) for r in rs) for k in keys} | {
        "R_min": min(r.R for r in rs), "total_critical": rs[0].total_critical}


def c1_status() -> tuple[str, dict]:
    with tempfile.TemporaryDirectory() as tmp:
        xml_path = Path(tmp) / "junit.xml"
        proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-k", "test_c1_", "-p", "no:cacheprovider",
                               f"--basetemp={Path(tmp) / 'pt'}", f"--junitxml={xml_path}"],
                              cwd=ROOT, capture_output=True, text=True)
        if not xml_path.exists():
            raise RuntimeError(f"pytest не створив звіт (код {proc.returncode}):\n{proc.stdout}\n{proc.stderr}")
        status: dict[int, list[str]] = {}
        for case in ET.parse(xml_path).getroot().iter("testcase"):
            name = case.get("name", "")
            if not name.startswith("test_c1_"):
                continue
            n = int(name.split("_")[2])
            outcome = ("пропущено (нема materials/)" if case.find("skipped") is not None
                       else "НЕ ПРОЙДЕНО" if case.find("failure") is not None
                       or case.find("error") is not None else "пройдено")
            status.setdefault(n, []).append(outcome)
    rows, data = [], {}
    for n, title in C1:
        got = status.get(n, ["нема тесту"])
        verdict = "НЕ ПРОЙДЕНО" if "НЕ ПРОЙДЕНО" in got else got[0]
        rows.append([n, title, verdict, len(got)])
        data[n] = verdict
    passed = sum(v == "пройдено" for v in data.values())
    return table("Приймальні кейси C1", ["№", "Кейс", "Статус", "тестів"], rows,
                 f"Генерується `scripts/reproduce.py` з прогону `pytest -k test_c1_`. Пройдено {passed} з 12. "
                 "Holdout перцептрона і приховані seeds моделі відмов бачить лише викладач."), data


def dp_vs_greedy(cfg, cache) -> tuple[str, dict]:
    """Ті самі вікна, ваги й цінності (умова задачі: вікно і W з конфігу); різниться лише розв'язувач."""
    W = cfg.replication_window_records
    agg = {"ДП": [0.0, 0, 0.0], "жадібний": [0.0, 0, 0.0]}   # корисність, біти, секунди
    better = windows = 0
    for lo in range(0, len(cache.records), W):
        win, cls = cache.records[lo:lo + W], cache.classes[lo:lo + W]
        _, _, weights, values = choose_copies(win, cls, cfg)
        util = utility_table(values, cfg.second_copy_factor, cfg.third_copy_factor)
        cost = [[r * w for r in range(4)] for w in weights]
        got = {}
        for name, solve in (("ДП", solve_dp), ("жадібний", solve_greedy)):
            t0 = time.perf_counter()
            copies = solve(util, cost, cfg.replication_budget_bits)
            dt = time.perf_counter() - t0
            u, c = total(util, cost, copies)
            agg[name][0] += u
            agg[name][1] += c
            agg[name][2] += dt
            got[name] = u
        windows += 1
        better += got["ДП"] > got["жадібний"] + 1e-9
    rows = [[n, a[0], a[1], a[2]] for n, a in agg.items()]
    gap = agg["ДП"][0] - agg["жадібний"][0]
    note = (f"Сума за {windows} вікнами сеансу по {W} записів, бюджет вікна W = {cfg.replication_budget_bits} біт, "
            f"класи — перцептрон. ДП строго кращий у {better} з {windows} вікон (Σ +{gap:.2f} корисності); "
            "у решті рівний. ДП O(n·W·Rmax) повільніший, але на вікні 48 × 5001 це мілісекунди. "
            "Вплив на Q — рядок «без ДП» таблиці 3. Контрприклад для жадібного — "
            "`tests/test_knapsack.py::test_greedy_counterexample`.")
    data = {k: {"utility": a[0], "bits": a[1], "seconds": a[2]} for k, a in agg.items()}
    data["windows"], data["dp_strictly_better"] = windows, better
    return table("Таблиця 1. ДП проти жадібного", ["Алгоритм", "Σ корисність", "Σ біти", "Σ час, с"], rows, note), data


def compress_encrypt(cfg, records) -> tuple[str, dict]:
    raw = pack_records(records)
    salt = os.urandom(16)
    payload_c = build_payload(records, no_compression_threshold=10.0)
    blob_ce = build_replica(records, cfg.password, replica_group_id=0, batch_seq=0, salt=salt,
                            no_compression_threshold=10.0)
    blob_e = build_replica(records, cfg.password, replica_group_id=0, batch_seq=1, salt=salt, compress=False)
    ct = blob_e[HEADER_SIZE:-32]
    lengths = huffman.code_lengths(ct)
    enc_then_huff = MIN_SIZE + 14 + 2 * len(lengths) + len(huffman.encode(ct, lengths))
    data = {
        "raw_bytes": len(raw), "huffman_payload": len(payload_c), "compress_then_encrypt": len(blob_ce),
        "encrypt_only": len(blob_e), "encrypt_then_huffman": enc_then_huff,
        "zlib_raw": len(zlib.compress(raw, 9)), "zlib_ciphertext": len(zlib.compress(ct, 9)),
        "ciphertext_distinct_bytes": len(set(ct)),
    }
    n = len(raw)
    rows = [
        ["сирі записи", data["raw_bytes"], 1.0],
        ["Гаффман (payload, без шифрування)", data["huffman_payload"], data["huffman_payload"] / n],
        ["стиснути → зашифрувати (репліка LOG3)", data["compress_then_encrypt"], data["compress_then_encrypt"] / n],
        ["лише зашифрувати (NO_COMPRESSION)", data["encrypt_only"], data["encrypt_only"] / n],
        ["зашифрувати → Гаффман", data["encrypt_then_huffman"], data["encrypt_then_huffman"] / n],
        ["довідково: zlib сирих записів", data["zlib_raw"], data["zlib_raw"] / n],
        ["довідково: zlib шифротексту", data["zlib_ciphertext"], data["zlib_ciphertext"] / n],
    ]
    note = (f"Увесь сеанс ({len(records)} записів) однією реплікою. Шифротекст AES-CTR статистично схожий на "
            f"випадкові байти ({data['ciphertext_distinct_bytes']} різних значень із 256, частоти рівні), тож "
            "Гаффман дає ≈ 8 біт на байт плюс таблицю: стискати треба до шифрування.")
    return table("Таблиця 4. Стиснення до і після шифрування", ["Варіант", "Байти", "Від сирого"], rows, note), data


def main(skip_train: bool, skip_ga: bool) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    if not skip_train:
        step(["-m", "log3.nn.train"])
    if not skip_ga:
        step(["-m", "log3.ga.search"])
    step(["-m", "log3.attack"])

    cfg = load_config()
    cache = FitnessCache(cfg)
    seeds = (cfg.model_seed, *range(201, 210))
    full, base = tuned_params(cfg), default_params(cfg)
    out: dict = {"eval_seeds": list(seeds), "records": len(cache.records), "params_full": full.__dict__,
                 "params_config": base.__dict__, "dev_environment": not MATERIALS_CONFIG.exists()}

    text, out["c1"] = c1_status()
    (RESULTS / "c1.md").write_text(text, encoding="utf-8")

    # Таблиця 5: повна система у трьох сценаріях
    scen = {s: avg_runs(s, cache, seeds, params=full) for s in SCENARIOS}
    out["scenarios"] = scen
    rows = [[NAMES[s], v["S"], v["R"], v["R_min"], 1 - cfg.scenario(s).max_unrecovered_critical, v["T"], v["Q"],
             int(v["b_stored"] / 8), f'{v["copies_placed"]:.0f}/{v["copies_assigned"]:.0f}',
             f'{v["stored_ok"]:.1f}/{v["replicas_written"]:.1f}']
            for s, v in scen.items()]
    (TABLES / "05_scenarios.md").write_text(table(
        "Таблиця 5. Поведінка у трьох сценаріях реплікації",
        ["Сценарій", "S", "R", "R min", "R поріг", "T", "Q", "global_used, B", "копій розміщено/призначено",
         "реплік записано/спроб"], rows,
        f"Повна система (параметри ГА), середнє за {len(seeds)} seeds моделі відмов. Критичних подій: "
        f"{scen['normal']['total_critical']}. «R поріг» — 1 − поріг втрати з конфігу. ДП не бачить місткості "
        "вузлів і накладних витрат репліки, тож частина призначених копій не розміщується (методичка 11.40)."),
        encoding="utf-8")

    text, out["dp_vs_greedy"] = dp_vs_greedy(cfg, cache)
    (TABLES / "01_dp_vs_greedy.md").write_text(text, encoding="utf-8")

    # Таблиця 2: max 1/2/3 копій
    copies = {(s, cap): avg_runs(s, cache, seeds, params=full, copies_cap=cap) for s in SCENARIOS for cap in (1, 2, 3)}
    out["copies"] = {f"{s}/{cap}": v for (s, cap), v in copies.items()}
    rows = [[NAMES[s], cap, v["S"], v["R"], v["R_min"], v["Q"], int(v["b_stored"])] for (s, cap), v in copies.items()]
    (TABLES / "02_copies.md").write_text(table(
        "Таблиця 2. Вплив максимальної кількості копій", ["Сценарій", "max копій", "S", "R", "R min", "Q",
                                                          "B_stored, біт"], rows,
        f"Повна система, середнє за {len(seeds)} seeds. Друга/третя копія коштує байтів (S) і окупається лише "
        "там, де вузли справді відмовляють."), encoding="utf-8")

    # Таблиця 3: вплив блоків
    only_crit = replace(cfg, priority=tuple(0.0 for _ in cfg.priority))
    variants = {
        "повна система (параметри ГА)": {"params": full},
        "без ГА (параметри з конфігу)": {"params": base},
        "без перцептрона (усі класу 0)": {"params": full, "classes": [0] * len(cache.records)},
        "без ДП (жадібний рюкзак)": {"params": full, "use_dp": False},
        "без стиснення (NO_COMPRESSION)": {"params": full, "compress": False},
        "без розподілу глобального бюджету": {"params": full, "pacing": False},
        "без ГА і без перцептрона": {"params": base, "classes": [0] * len(cache.records)},
        "без ГА і без розподілу бюджету": {"params": base, "pacing": False},
        "лише критичні (цінність класів = 0)": {"params": full, "cfg": only_crit},
    }
    abl = {name: {s: avg_runs(s, cache, seeds, **kw) for s in SCENARIOS} for name, kw in variants.items()}
    out["ablation"] = abl
    rows = [[name, *(v[s]["Q"] for s in SCENARIOS), v["emergency"]["R"], mean(v[s]["Q"] for s in SCENARIOS)]
            for name, v in abl.items()]
    (TABLES / "03_ablation.md").write_text(table(
        "Таблиця 3. Вплив блоків", ["Варіант", "Q штатний", "Q деградований", "Q аварійний", "R аварійний",
                                    "Q середнє"], rows,
        f"Середнє за {len(seeds)} seeds. Q включає γ·T, тому різниці в четвертому знаку — шум таймера. "
        "«Лише критичні» — не варіант системи, а межа: R рахує тільки критичні події, а цінність v_i з умови "
        "винагороджує й некритичні, тож Q і ціль ДП тягнуть у різні боки."), encoding="utf-8")

    text, out["compress_encrypt"] = compress_encrypt(cfg, cache.records)
    (TABLES / "04_compress_encrypt.md").write_text(text, encoding="utf-8")

    (RESULTS / "final_metrics.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str) + "\n",
                                                encoding="utf-8")
    for p in ("c1.md", "tables/05_scenarios.md", "tables/03_ablation.md"):
        print((RESULTS / p).read_text(encoding="utf-8"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-train", action="store_true", help="не перенавчати перцептрон")
    ap.add_argument("--skip-ga", action="store_true", help="не запускати ГА (взяти results/ga_params.json)")
    a = ap.parse_args()
    main(a.skip_train, a.skip_ga)
