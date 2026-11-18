"""Етапи демонстрації захисту. Кожен етап — список справжніх команд і перевірок.

Види етапів (поле kind у підсумку):
  новий експеримент          команда рахує заново в ізольованій копії проєкту (results/ і моделі не чіпаються)
  перегляд збережених        показ закомічених результатів (без перерахунку)
  dev-режим                  без materials/: синтетичний конфіг і модель відмов, не дані викладача
  продукт                    LOG3 Vault у тимчасовому каталозі
  середовище                 версії, лінтер, тести
"""
from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path

from . import matrix
from .lab import SAME_DISK, Lab, expect_state, make_mixed
from .runner import Ctx, Result, Stage, Step, V

PROTECTED_FILES = ("results", "log3/nn/model.npz")


def has_materials(ctx: Ctx) -> bool:
    return (ctx.root / "materials" / "team_config.json").is_file()


def _digest(root: Path) -> str:
    h = hashlib.sha256()
    for rel in PROTECTED_FILES:
        p = root / rel
        for f in sorted(p.rglob("*") if p.is_dir() else [p]):
            if f.is_file():
                h.update(f.relative_to(root).as_posix().encode() + f.read_bytes())
    return h.hexdigest()


def isolated_copy(ctx: Ctx) -> Path:
    """Копія проєкту для нових експериментів: закомічені results/, модель і materials/ не перезаписуються."""
    dest = ctx.work / "project-copy"
    if not dest.exists():
        shutil.copytree(ctx.root, dest, ignore=shutil.ignore_patterns(
            ".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".ruff_cache", "materials", "defense-out",
            "dist", "build", "*.egg-info"))
        if has_materials(ctx):
            shutil.copytree(ctx.root / "materials", dest / "materials")
    ctx.vars.setdefault("committed_digest", _digest(ctx.root))
    return dest


def _untouched(ctx: Ctx, res: Result) -> str:
    assert _digest(ctx.root) == ctx.vars["committed_digest"], "закомічені results/ або модель змінено!"
    return "закомічені results/ і log3/nn/model.npz не змінено (SHA-256 до й після збігаються)"


def _iso(ctx: Ctx, goal: str, module: str, *args, check=None, timeout: float = 1800) -> Step:
    copy = isolated_copy(ctx)
    return Step(goal, [ctx.py, "-m", module, *args], cwd=copy, env={"PYTHONPATH": str(copy)}, check=check,
                timeout=timeout)


def _source_label(ctx: Ctx) -> str:
    return "materials/ T01 (пакет викладача)" if has_materials(ctx) else "dev-режим: синтетичний конфіг і модель"


# --- середовище й навчальний тракт ------------------------------------------------------------

def env_stage(ctx: Ctx) -> list[Step]:
    quick = ["tests/test_records.py", "tests/test_payload.py", "tests/test_replica.py", "tests/test_huffman.py",
             "tests/test_knapsack.py", "tests/test_vault_keys.py", "tests/test_vault_units.py"]
    tests = quick if ctx.profile == "quick" else ["tests"]
    steps = [Step("версії Python, ОС і залежностей", [ctx.py, "-c", (
        "import platform,sys,numpy,cryptography,zstandard;print(platform.platform());"
        "print('python',sys.version.split()[0],'numpy',numpy.__version__,'cryptography',cryptography.__version__,"
        "'zstandard',zstandard.__version__)")], cwd=ctx.root)]
    if shutil.which("ruff") or _module_exists("ruff"):
        steps.append(Step("лінтер ruff", [ctx.py, "-m", "ruff", "check", "."], cwd=ctx.root))
    steps.append(Step(f"тести pytest ({'вибірка' if ctx.profile == 'quick' else 'увесь набір'}; кейси без "
                      f"materials/ пропускаються)", [ctx.py, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                                                    "--basetemp", str(ctx.work / "pytest"), *tests],
                      cwd=ctx.root, timeout=3600, check=_pytest_summary))
    return steps


def _module_exists(name: str) -> bool:
    import importlib.util
    return importlib.util.find_spec(name) is not None


def _pytest_summary(ctx: Ctx, res: Result) -> str:
    tail = [line for line in res.out.strip().splitlines() if " passed" in line or " failed" in line]
    return tail[-1].strip() if tail else "pytest завершився"


def pipeline_stage(ctx: Ctx) -> list[Step]:
    seeds = ["1"] if ctx.profile == "quick" else ["1", "2", "3", "4", "5"]

    def q(c: Ctx, r: Result) -> str:
        rows = (r.data or {}).get("runs")
        if not rows:
            return "pipeline завершився"
        avg = {}
        for row in rows:
            avg.setdefault(row["scenario"], []).append(row)
        parts = [f"{s}: S={sum(x['S'] for x in v) / len(v):.3f} R={sum(x['R'] for x in v) / len(v):.3f} "
                 f"Q={sum(x['Q'] for x in v) / len(v):.3f}" for s, v in avg.items()]
        return f"джерело: {_source_label(c)}; " + "; ".join(parts)
    return [_iso(ctx, "навчальний тракт у трьох сценаріях (S, R, T, Q — навчальні одиниці, див. README)",
                 "log3.pipeline", "--scenario", "all", "--seeds", *seeds, "--json", check=q),
            Step("закомічені результати не змінено", action=lambda c: _untouched(c, None))]


def huffman_stage(ctx: Ctx) -> list[Step]:
    return [Step("канонічний Гаффман: довжини, коди, decode == вхід", [ctx.py, "-m", "log3.defense.demo", "huffman"],
                 cwd=ctx.root, check=lambda c, r: f"{r.data['input_bytes']} B -> {r.data['output_bytes']} B, "
                                                  f"Крафт {r.data['kraft_sum']:.3f}"),
            Step("ДП-рюкзак проти жадібного і повного перебору", [ctx.py, "-m", "log3.defense.demo", "planner"],
                 cwd=ctx.root, check=lambda c, r: f"ДП {r.data['dp'][0]:.2f} = перебір "
                                                  f"{r.data['brute_force_best'][0]:.2f} ≥ жадібний "
                                                  f"{r.data['greedy'][0]:.2f}"),
            Step("тести Гаффмана, ДП і розміщення", [ctx.py, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                                                     "--basetemp", str(ctx.work / "pytest-h"), "tests/test_huffman.py",
                                                     "tests/test_knapsack.py", "tests/test_scheduler.py"],
                 cwd=ctx.root, check=_pytest_summary),
            Step("перегляд збереженої таблиці ДП проти жадібного", [ctx.py, "-m", "log3.defense.show",
                                                                     "results/tables/01_dp_vs_greedy.md"],
                 cwd=ctx.root)]


def classifier_stage(ctx: Ctx) -> list[Step]:
    view = Step("перегляд збережених метрик перцептрона (закомічені, T01)", [
        ctx.py, "-m", "log3.defense.show", "results/classifier_metrics.json", "results/confusion_matrix.md"],
        cwd=ctx.root)
    if ctx.profile == "quick":
        return [view]
    return [view, _iso(ctx, f"новий експеримент: навчити перцептрон заново ({_source_label(ctx)})",
                       "log3.nn.train", timeout=3600),
            Step("нові метрики (в ізольованій копії)", [ctx.py, "-m", "log3.defense.show",
                                                        "results/classifier_metrics.json"],
                 cwd=isolated_copy(ctx)),
            Step("закомічені результати не змінено", action=lambda c: _untouched(c, None))]


def ga_stage(ctx: Ctx) -> list[Step]:
    view = Step("перегляд збереженого порівняння ГА з random search і базою", [
        ctx.py, "-m", "log3.defense.show", "results/tables/06_ga_vs_random.md", "results/ga_params.json"],
        cwd=ctx.root)
    if ctx.profile == "quick":
        return [view]
    return [view, _iso(ctx, f"новий експеримент: ГА проти random search за однакового бюджету ({_source_label(ctx)})",
                       "log3.ga.search", "--pop", "16", "--gens", "10", "--repeats", "3", timeout=3600),
            Step("нова таблиця (в ізольованій копії)", [ctx.py, "-m", "log3.defense.show",
                                                        "results/tables/06_ga_vs_random.md"],
                 cwd=isolated_copy(ctx)),
            Step("закомічені результати не змінено", action=lambda c: _untouched(c, None))]


def attack_stage(ctx: Ctx) -> list[Step]:
    keys, trials = ("2", "300") if ctx.profile == "quick" else ("10", "2000")
    return [_iso(ctx, "новий експеримент: ГА відновлює ключ слабкого шифру; tamper змінених реплік",
                 "log3.attack", "--keys", keys, "--trials", trials),
            Step("звіт атаки (в ізольованій копії)", [ctx.py, "-m", "log3.defense.show", "results/attack.md"],
                 cwd=isolated_copy(ctx)),
            Step("закомічені результати не змінено", action=lambda c: _untouched(c, None))]


# --- продукт ------------------------------------------------------------------------------------

def mixed_stage(ctx: Ctx) -> list[Step]:
    lab = Lab(ctx, "vault-mixed")
    dest = lab.dir / "restored"
    return lab.setup() + [
        lab.step("backup --wait змішаного каталогу (PDF, DOCX, ZIP, EXE, JPG, CSV, Unicode, порожні, 6 МіБ)",
                 "backup", lab.src, "--wait", "--name", "змішаний", check=expect_state(lab, "PROTECTED", "v")),
        lab.step("пошук за метаданими: усі *.docx (маніфест розшифровується локально)", "find", "--path", "*.docx",
                 check=lambda c, r: f"знайдено {len(r.data['find']['entries'])} файл(и), "
                                    f"розшифровано маніфестів: {r.data['find']['manifests_decrypted']}"),
        lab.step("restore у новий каталог", "restore", V("v"), dest, check=lambda c, r: lab.same_tree(dest)),
        lab.step("пробне відновлення (drill) і health", "drill",
                 check=lambda c, r: f"drill ok={r.data['drill']['ok']}, файлів {r.data['drill']['files']}"),
        lab.step("метрики у форматі Prometheus (без шляхів і ідентифікаторів файлів)", "metrics", "--format",
                 "prometheus", check=lambda c, r: _prom(lab, r)),
    ]


def _prom(lab: Lab, r: Result) -> str:
    text = r.data["prometheus"]
    assert str(lab.src) not in text and "звіт" not in text
    return f"{sum(1 for line in text.splitlines() if not line.startswith('#'))} рядків метрик, шляхів файлів нема"


def showcase_stage(ctx: Ctx) -> list[Step]:
    """Вузол недоступний -> стійке локальне приймання -> перезапуск -> черга відновлена -> вузол
    повернувся -> політику копій виконано -> дані відновлено й перевірено."""
    lab = Lab(ctx, "showcase")

    def queue_state(c: Ctx, r: Result) -> str:
        q = r.data["queue"]
        pending = q["by_state"].get("PENDING", 0) + q["by_state"].get("RUNNING", 0)
        assert pending > 0, q
        return (f"черга з диска: {q['by_state']}, найстаріше завдання {q['oldest_job_age_seconds']} с, "
                f"наступний повтор через {q['next_retry_in_seconds']} с")
    return lab.setup(node_up=False, limits={"max_attempts": 50}) + [
        lab.step("1. вузол недоступний; backup приймає дані стійко локально (код 0 = прийнято, НЕ захищено)",
                 "backup", lab.src, check=expect_state(lab, "ACCEPTED_LOCAL", "v", False)),
        lab.step("2. перша спроба реплікації: disk-a приймає, вузол — ні; процес аварійно зупиняється "
                 "до фіксації результату", "run", expect=(77,),
                 env={"LOG3_VAULT_CRASHPOINT": "replicate.after_put_before_commit"},
                 check=lambda c, r: "процес аварійно завершився (os._exit), стан у пам'яті втрачено"),
        lab.step("3. новий процес: черга відновлена з диска", "queue", check=queue_state),
        lab.step("4. новий процес: один прохід черги — disk-a підтверджено, вузол ще недоступний", "run", "--once",
                 expect=(3,), check=lambda c, r: (f"підтверджено {r.data['queue']['done']}, на повтор "
                                                  f"{r.data['queue']['retry']}; у черзі "
                                                  f"{r.data['metrics']['queue_jobs']}")),
        lab.step("5. стан: прийнято, політику копій ще не виконано", "status", V("v"), expect=(3,),
                 check=expect_state(lab, "REPLICATING", protected=False)),
        lab.start_node(),
        lab.step("6. вузол повернувся: черга дообробляється", "run",
                 check=lambda c, r: f"підтверджено {r.data['queue']['done']}, повторів {r.data['queue']['retry']}"),
        lab.step("7. політику копій виконано", "status", V("v"), check=expect_state(lab, "PROTECTED")),
        lab.step("8. verify перечитує кожну копію (довжина + SHA-256)", "verify",
                 check=lambda c, r: f"цілих копій {r.data['verify']['ok']} з {r.data['verify']['checked']}"),
        lab.step("9. restore у новий каталог і побайтова звірка", "restore", V("v"), lab.dir / "restored",
                 check=lambda c, r: lab.same_tree(lab.dir / "restored")),
    ]


def node_repair_stage(ctx: Ctx) -> list[Step]:
    return matrix.m6(ctx)


def learning_stage(ctx: Ctx) -> list[Step]:
    lab = Lab(ctx, "learning")
    fast = ["probe.rate=1.0", "probe.max_cpu_fraction=4", "probe.max_load_per_cpu=100",
            "probe.max_chunks_per_operation=100000", "train.min_rows=50", "train.min_groups=5",
            "train.search_budget=8", "gate.min_shadow_decisions=20", "gate.min_relative_gain=0.005",
            'actions=["store:0", "zstd:1", "zstd:3", "zstd:6", "zstd:12"]', "gate.bootstrap=100",
            "cost.cpu_second_equiv_bytes=0"]
    sets = [x for item in fast for x in ("--set", item)]

    def corpus(seed: int):
        def act(c: Ctx) -> str:
            import sys
            sys.path.insert(0, str(c.root))
            from scripts.vault_corpus import make_corpus, mutate
            if seed == 0:
                make_corpus(lab.dir / "corpus", 5, 6, max_file_kib=96)
                return "синтетичний корпус проєктів (типи впереміш)"
            return f"змінено файлів: {mutate(lab.dir / 'corpus', seed, 0.3, 0.2)}"
        return act

    def rounds(n: int, seed: int) -> list[Step]:
        out = []
        for i in range(n):
            out += [Step("змінити частину файлів", action=corpus(seed + i)),
                    lab.step(f"backup раунд {seed + i}", "backup", lab.dir / "corpus", "--wait",
                             check=expect_state(lab, "PROTECTED", "last"))]
        return out
    return lab.setup(node=False, min_copies=1, source="none") + [
        lab.step("холодний старт: моделі нема, працює базова політика", "learn", "status",
                 check=lambda c, r: f"режим {r.data['learning']['state']['mode']}, активна модель "
                                    f"{r.data['learning']['state']['active_model']}"),
        lab.step("пришвидшені межі навчання для демонстрації (learn config)", "learn", "config", *sets),
        Step("підготувати корпус", action=corpus(0)),
        *rounds(3, 1),
        lab.step("навчання й відбір кандидата (gate)", "learn", "train", "--force",
                 check=lambda c, r: f"статус {r.data['learning']['status']}"),
        *rounds(2, 40),
        lab.step("оцінювання кандидата в shadow", "learn", "evaluate",
                 check=lambda c, r: f"оцінено: passed={r.data['learning'].get('passed')}"),
        lab.step("promote (лише якщо оцінювання пройдено)", "learn", "promote", expect=(0, 1, 6),
                 check=lambda c, r: f"режим: {(r.data.get('learning') or {}).get('status') or r.data.get('error')}"),
        *rounds(1, 60),
        lab.step("restore версії, створеної з адаптацією", "restore", V("last"), lab.dir / "restored",
                 check=lambda c, r: lab.same_tree(lab.dir / "restored", lab.dir / "corpus")),
        lab.step("rollback до бази", "learn", "rollback", "--reason", "демонстрація",
                 check=lambda c, r: f"статус {r.data['learning']['status']}"),
        lab.step("стан навчання після rollback", "learn", "status",
                 check=lambda c, r: f"режим {r.data['learning']['state']['mode']}"),
    ]


def keys_stage(ctx: Ctx) -> list[Step]:
    lab = Lab(ctx, "keys")
    kit = lab.dir / "kit"
    export_pass = "export phrase for the demo"
    ctx.secrets.append(export_pass)
    new_pass = "rotated passphrase for the demo"
    ctx.secrets.append(new_pass)
    clean = {"PYTHONPATH": "", "LOG3_VAULT_REPO": "", "LOG3_VAULT_PASSPHRASE": "",
             "LOG3_VAULT_EXPORT_PASSPHRASE": export_pass}
    return lab.setup(node=False, min_copies=1) + [
        lab.step("backup v1 (ключ покоління 0)", "backup", lab.src, "--wait",
                 check=expect_state(lab, "PROTECTED", "v1")),
        lab.step("ротація: новий активний ключ даних", "key", "rotate",
                 check=lambda c, r: f"{r.data['keys']['previous_key_id']} -> {r.data['keys']['active_key_id']}"),
        lab.step("зміна парольної фрази (ключі перезагорнуто, об'єкти не перешифровуються)", "key", "passphrase",
                 env={"LOG3_VAULT_NEW_PASSPHRASE": new_pass}),
        lab.step("стара фраза більше не відкриває сховище", "restore", V("v1"), lab.dir / "old", expect=(8,)),
        lab.step("стара версія відкривається новою фразою", "restore", V("v1"), lab.dir / "r1",
                 env={"LOG3_VAULT_PASSPHRASE": new_pass}, check=lambda c, r: lab.same_tree(lab.dir / "r1")),
        lab.step("стан ключів (без секретів)", "key", "status",
                 check=lambda c, r: f"ключів у keyring: {len(r.data['keys']['keys'])}"),
        lab.step("аварійний експорт ключів під окремою фразою", "key", "export", "--out", kit / "keys.export",
                 env={"LOG3_VAULT_PASSPHRASE": new_pass, "LOG3_VAULT_EXPORT_PASSPHRASE": export_pass}),
        lab.step("набір аварійного restore", "emergency-kit", "--out", kit),
        Step("автономний restore: python -I, без пакета log3, без config.json і каталогу",
             [ctx.py, "-I", kit / "log3_restore.py", "--backend", lab.disk, "--key-export", kit / "keys.export",
              "restore", "latest", lab.dir / "standalone"], cwd=kit, env=clean,
             check=lambda c, r: lab.same_tree(lab.dir / "standalone")),
    ]


PREVIOUS_RELEASE = "100873e"          # dev після етапу 2 (merge PR #18): схема каталогу 1, без keyring


def update_stage(ctx: Ctx) -> list[Step]:
    """Справжня попередня версія з git створює сховище; нова версія оновлює його з аварією й без."""
    if shutil.which("git") is None or not (ctx.root / ".git").exists():
        raise RuntimeError("потрібні git і повна історія репозиторію (стара версія береться з коміту "
                           f"{PREVIOUS_RELEASE})")
    lab = Lab(ctx, "update")
    old = lab.dir / "old-release"
    old_env = {**lab.env, "PYTHONPATH": str(old)}

    def checkout(c: Ctx) -> str:
        import subprocess
        lab.dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "worktree", "add", "--detach", str(old), PREVIOUS_RELEASE], cwd=c.root, check=True,
                       capture_output=True)
        info = make_mixed(lab.src)
        return f"стара версія {PREVIOUS_RELEASE} у {old.name}; джерело {info['files']} файлів"

    def cleanup(c: Ctx) -> str:
        import subprocess
        subprocess.run(["git", "worktree", "remove", "--force", str(old)], cwd=c.root, capture_output=True)
        return "тимчасову копію старої версії прибрано"

    def old_step(goal: str, *args, expect=(0,), check=None) -> Step:
        return Step(goal + " [стара версія]", lab.v(*args), env=old_env, cwd=old, expect=expect, check=check)

    def schema(c: Ctx, r: Result) -> str:
        assert r.data["schema_version"] == 2 and r.data["backups"], r.data
        return f"схема {r.data['schema_version']}, копія перед міграцією: {r.data['backups'][0]}"
    return [
        Step("взяти попередню версію з git (етап 2)", action=checkout),
        old_step("створити сховище", "init", "--min-copies", "1", "--chunk-size-mib", "0.25"),
        old_step("додати backend-каталог", "backend", "add", "disk", "--type", "localdir", "--path", lab.disk,
                 "--domain", "disk-a"),
        old_step("backup --wait", "backup", lab.src, "--wait", check=expect_state(lab, "PROTECTED", "v")),
        lab.step("нова версія: аварія посеред міграції каталогу (невдале оновлення)", "status", expect=(77,),
                 env={"LOG3_VAULT_CRASHPOINT": "catalog.migrate_before_commit"},
                 check=lambda c, r: "процес оновлення аварійно завершився під час міграції"),
        old_step("стара версія далі працює зі сховищем (транзакцію міграції відкочено)", "status",
                 expect=(0, 3)),
        old_step("restore старою версією", "restore", V("v"), lab.dir / "restored-old",
                 check=lambda c, r: lab.same_tree(lab.dir / "restored-old")),
        lab.step("нова версія: успішна міграція з копією каталогу", "migrate", "status", check=schema),
        lab.step("нова версія відновлює версію, створену старою", "restore", V("v"), lab.dir / "restored-new",
                 check=lambda c, r: lab.same_tree(lab.dir / "restored-new")),
        old_step("стара версія не відкриває новішу схему (захист від тихого пошкодження)", "status", expect=(1,),
                 check=lambda c, r: r.data["error"]["code"]),
        Step("прибрати копію старої версії", action=cleanup),
    ]


def install_stage(ctx: Ctx) -> list[Step]:
    dist, venv = ctx.work / "dist", ctx.work / "venv"
    bindir = venv / ("Scripts" if os.name == "nt" else "bin")
    vpy = bindir / "python"
    lab = Lab(ctx, "installed")
    clean = {"PYTHONPATH": "", **lab.env}

    def tamper(c: Ctx) -> str:
        bad = ctx.work / "dist-tampered"
        shutil.copytree(dist, bad)
        p = bad / "log3_restore.py"
        p.write_bytes(p.read_bytes().replace(b"Fail", b"Fa1l", 1))
        return "копія релізу з одним зміненим байтом у log3_restore.py"

    def where(c: Ctx, r: Result) -> str:
        path = r.out.strip().splitlines()[-1]
        assert str(venv) in path, path
        return f"пакет імпортовано зі встановлення: {path}"
    return [
        Step("зібрати релізні артефакти (потрібна мережа для коліс залежностей)",
             [ctx.py, "scripts/build_release.py", "--out", dist], cwd=ctx.root, timeout=1800,
             check=lambda c, r: f"версія {r.data['version']}, файлів {r.data['files']}"),
        Step("перевірити SHA256SUMS", [ctx.py, "scripts/build_release.py", "--verify", dist], cwd=ctx.root,
             check=lambda c, r: f"перевірено файлів: {r.data['checked']}"),
        Step("підмінити байт у копії релізу", action=tamper),
        Step("перевірка цілісності виявляє підміну", [ctx.py, "scripts/build_release.py", "--verify",
                                                      ctx.work / "dist-tampered"], cwd=ctx.root, expect=(7,),
             check=lambda c, r: f"невідповідні: {r.data['mismatched']}"),
        Step("чисте віртуальне середовище", [ctx.py, "-m", "venv", venv]),
        Step("офлайн-встановлення: --no-index, лише wheelhouse релізу",
             [vpy, "-m", "pip", "install", "--no-index", "--find-links", dist / "wheelhouse",
              *sorted(str(p) for p in dist.glob("log3-*.whl"))], timeout=1800),
        Step("пакет береться зі встановлення, а не з каталогу проєкту",
             [vpy, "-c", "import log3.vault, os; print(os.path.dirname(log3.vault.__file__))"], env=clean,
             check=where),
        Step("встановлена команда log3-vault", [bindir / "log3-vault", "--version"], env=clean),
        Step("init установленою командою", [bindir / "log3-vault", "init", "--json", "--min-copies", "1"],
             env=clean),
        Step("backend", [bindir / "log3-vault", "backend", "add", "disk", "--json", "--type", "localdir", "--path",
                         lab.disk, "--domain", "disk-a"], env=clean),
        Step("підготувати джерело", action=lambda c: str(make_mixed(lab.src))),
        Step("backup --wait", [bindir / "log3-vault", "backup", "--json", lab.src, "--wait"], env=clean,
             check=expect_state(lab, "PROTECTED", "v")),
        Step("restore і побайтова звірка", [bindir / "log3-vault", "restore", "--json", V("v"), lab.dir / "r"],
             env=clean, check=lambda c, r: lab.same_tree(lab.dir / "r")),
        Step("модуль захисту встановлено разом із пакетом", [bindir / "log3-defense", "--list"], env=clean),
    ]


STAGES = [
    Stage("env", "Середовище й тести", "середовище", env_stage),
    Stage("pipeline", "Навчальний pipeline LOG3 (S, R, T, Q)", "новий експеримент", pipeline_stage),
    Stage("huffman", "Гаффман і планування копій", "новий експеримент + перегляд збережених", huffman_stage),
    Stage("classifier", "Класифікатор (перцептрон)", "перегляд збережених / новий експеримент", classifier_stage),
    Stage("ga", "ГА проти бази й random search", "перегляд збережених / новий експеримент", ga_stage),
    Stage("attack", "Навчальна атака й tamper", "новий експеримент", attack_stage),
    Stage("vault-mixed", "Змішаний каталог довільних файлів", "продукт", mixed_stage, note=SAME_DISK),
    Stage("showcase", "Показовий сценарій: вузол недоступний -> приймання -> перезапуск -> черга -> вузол "
          "повернувся -> політика -> restore", "продукт", showcase_stage, note=SAME_DISK),
    Stage("node-repair", "Відмова й пошкодження копій, repair і побайтовий restore", "продукт", node_repair_stage,
          note=SAME_DISK),
    Stage("learning", "Холодний старт, оцінювання й rollback адаптації", "продукт", learning_stage),
    Stage("keys", "Ключі: ротація, зміна фрази, експорт, аварійний restore без інсталяції", "продукт", keys_stage),
    Stage("update", "Оновлення з етапу 2: аварія міграції, робота старої версії, успішна міграція",
          "продукт (стара версія з git)", update_stage),
    Stage("install", "Чиста інсталяція з релізних артефактів офлайн", "продукт (реліз)", install_stage,
          profiles=("full",)),
]
