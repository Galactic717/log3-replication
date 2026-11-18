"""Матриця відмов LOG3 Vault: кожен сценарій — справжні команди CLI в нових процесах і перевірки з диска.

Для кожного сценарію перевіряється три речі: повнота restore для станів, які система заявила
(побайтове порівняння), коректність повторів (ідемпотентність, без подвійного обліку) і відсутність
помилкового protected. Аварії — це os._exit у названих точках коду (LOG3_VAULT_CRASHPOINT), а не
вимкнення живлення; вузол — окремий процес на тому самому диску.

    python -m log3.defense.matrix [--only ID ...] [--out DIR]
"""
from __future__ import annotations

from pathlib import Path

from .lab import SAME_DISK, Lab, expect_state, make_mixed
from .runner import Ctx, Result, Stage, Step, V

KIND = "матриця відмов (продукт)"


def _crashed(ctx: Ctx, res: Result) -> str:
    return "процес аварійно завершився в точці аварії (os._exit 77, без finally/atexit)"


def _no_protected(lab: Lab, what: str):
    def check(ctx: Ctx, res: Result) -> str:
        n = lab.sql("SELECT COUNT(*) FROM versions WHERE state='PROTECTED'")[0][0]
        assert n == 0, f"{n} версій помилково PROTECTED"
        states = dict(lab.sql("SELECT state, COUNT(*) FROM versions GROUP BY state"))
        return f"{what}; з диска: стани версій {states}, жодна не PROTECTED"
    return check


def _restore(lab: Lab, var: str, name: str, src: Path | None = None) -> list[Step]:
    dest = lab.dir / f"restored-{name}"
    return [lab.step("відновити версію в новий каталог (окремий процес)", "restore", V(var), dest,
                     check=lambda c, r: lab.same_tree(dest, src))]


def _single_charge(lab: Lab, ctx: Ctx, res: Result) -> str:
    rows = lab.ledger("SELECT COUNT(*), COALESCE(SUM(length),0), COALESCE(MAX(puts),0) FROM charges")[0]
    files = lab.objects_on(lab.node_dir)
    on_disk = sum(p.stat().st_size for p in files)
    assert rows[0] == len(files) and rows[1] == on_disk, f"облік вузла {rows} не збігається з {len(files)} файлами"
    dup = lab.sql("SELECT COUNT(*) FROM (SELECT oid, backend FROM replicas GROUP BY oid, backend HAVING COUNT(*)>1)")
    assert dup[0][0] == 0
    return (f"вузол: {rows[0]} об'єктів, квоту списано один раз ({rows[1]} B = байти на диску вузла), "
            f"найбільше повторних PUT одного об'єкта: {rows[2]}")


def m1(ctx: Ctx) -> list[Step]:
    lab = Lab(ctx, "m1-crash-after-accept")
    return lab.setup() + [
        lab.step("backup з аварією процесу одразу після стійкого локального приймання", "backup", lab.src,
                 "--op-id", "m1-op", expect=(77,), env={"LOG3_VAULT_CRASHPOINT": "capture.after_publish"},
                 check=_crashed),
        lab.step("новий процес: повтор тієї самої операції не створює другої версії", "backup", lab.src,
                 "--op-id", "m1-op", check=lambda c, r: (
                     expect_state(lab, "ACCEPTED_LOCAL", "v", False)(c, r)
                     + ("; повтор операції: так" if r.data["repeated_operation"] else _fail("друга версія")))),
        lab.step("новий процес: черга з диска передає копії", "run", check=lambda c, r: (
            f"підтверджено {r.data['queue']['done']} завдань, повторів {r.data['queue']['retry']}")),
        lab.step("стан після реплікації", "status", V("v"), check=expect_state(lab, "PROTECTED")),
        Step("облік повторів на вузлі", action=lambda c: _single_charge(lab, c, None)),
        *_restore(lab, "v", "m1"),
    ]


def _fail(msg: str):
    raise AssertionError(msg)


def m2(ctx: Ctx) -> list[Step]:
    lab = Lab(ctx, "m2-lost-response")
    return lab.setup() + [
        lab.stop_node("перезапустити вузол із збоєм: запис стійкий, відповідь губиться"),
        lab.start_node(fault="node_drop_response_once"),
        lab.step("backup --wait: перша відповідь вузла загублена, клієнт повторює", "backup", lab.src, "--wait",
                 check=expect_state(lab, "PROTECTED", "v")),
        Step("перевірити облік: повтор не дописав і не списав квоту вдруге", action=lambda c: (
            _single_charge(lab, c, None) + "; " + _dropped(lab))),
        *_restore(lab, "v", "m2"),
    ]


def _dropped(lab: Lab) -> str:
    puts = lab.ledger("SELECT MAX(puts) FROM charges")[0][0]
    errors = lab.sql("SELECT COUNT(*) FROM events WHERE kind='job_error'")[0][0]
    assert puts >= 2 and errors >= 1, f"очікувався повторний PUT (puts={puts}, помилок={errors})"
    return f"загублена відповідь: {errors} помилк(а) спроби, той самий об'єкт прийнято {puts} рази як дублікат"


def m3(ctx: Ctx) -> list[Step]:
    lab = Lab(ctx, "m3-crash-block-manifest-catalog")
    spool = lab.repo / "spool" / "objects"

    def orphans(c: Ctx) -> str:
        n = len(list(spool.glob("*/*")))
        assert n >= 1, "очікувались файли-сироти в spool"
        return f"на диску {n} файл(ів)-сиріт у spool, у каталозі версія лише CAPTURING"

    def repaired(c: Ctx, r: Result) -> str:
        rep = r.data["repair"]
        assert rep["failed_captures"] == 1 and rep["orphans_removed"] >= 1, rep
        assert not list(spool.glob("*/*")), "сироти лишились"
        return f"захоплення позначено FAILED, прибрано сиріт: {rep['orphans_removed']}"
    crash_steps = []
    for point, what in (("capture.after_first_chunk", "після запису першого блока (до маніфесту)"),
                        ("capture.before_publish", "після блоків і маніфесту (до транзакції каталогу)")):
        crash_steps += [
            lab.step(f"backup з аварією {what}", "backup", lab.src, expect=(77,),
                     env={"LOG3_VAULT_CRASHPOINT": point}, check=_crashed),
            Step("стан на диску після аварії", action=orphans),
            lab.step("repair у новому процесі прибирає наслідки", "repair", "--no-run", expect=(0, 4),
                     check=repaired),
        ]
    return lab.setup() + crash_steps + [
        Step("жодної версії не прийнято й не захищено", action=lambda c: _no_protected(lab, "після двох аварій")(
            c, None)),
        lab.step("backup без --wait (локальне приймання)", "backup", lab.src,
                 check=expect_state(lab, "ACCEPTED_LOCAL", "v", False)),
        lab.step("run з аварією після запису на backend, до фіксації в каталозі", "run", expect=(77,),
                 env={"LOG3_VAULT_CRASHPOINT": "replicate.after_put_before_commit"}, check=_crashed),
        lab.step("після аварії версія не заявлена protected", "status", V("v"), expect=(3,),
                 check=expect_state(lab, "ACCEPTED_LOCAL", protected=False)),
        lab.step("run у новому процесі: lease прострочено, завдання повторюються", "run",
                 check=lambda c, r: f"підтверджено {r.data['queue']['done']}, повторів {r.data['queue']['retry']}"),
        lab.step("стан", "status", V("v"), check=expect_state(lab, "PROTECTED")),
        Step("повтор після аварії не подвоїв облік", action=lambda c: _single_charge(lab, c, None)),
        *_restore(lab, "v", "m3"),
    ]


def m4(ctx: Ctx) -> list[Step]:
    lab = Lab(ctx, "m4-node-outage")

    def degraded(c: Ctx, r: Result) -> str:
        m = r.data["metrics"]
        assert m["queue_attention_jobs"] > 0 and m["versions_by_state"].get("DEGRADED") == 1, m
        return (f"бюджет повторів вичерпано: {m['queue_attention_jobs']} завдань ATTENTION, версія DEGRADED "
                f"(не protected)")
    return lab.setup(node_up=False) + [
        lab.step("backup, поки вузол лежить: дані стійко прийнято локально", "backup", lab.src,
                 check=expect_state(lab, "ACCEPTED_LOCAL", "v", False)),
        lab.step("тривала недоступність: run вичерпує max_attempts", "run", expect=(4,), check=degraded),
        lab.step("версія DEGRADED (код 4), політику не виконано", "status", V("v"), expect=(4,),
                 check=expect_state(lab, "DEGRADED", protected=False)),
        *_restore(lab, "v", "m4-degraded"),
        lab.start_node(),
        lab.step("repair після повернення вузла", "repair", check=lambda c, r: (
            f"повернуто в чергу {r.data['repair']['requeued']}, незахищених версій: "
            f"{r.data['repair']['versions_not_protected']}")),
        lab.step("стан", "status", V("v"), check=expect_state(lab, "PROTECTED")),
        lab.step("verify перечитує всі копії", "verify", check=lambda c, r: (
            f"перевірено {r.data['verify']['checked']}, цілих {r.data['verify']['ok']}")),
    ]


def m5(ctx: Ctx) -> list[Step]:
    lab = Lab(ctx, "m5-disk-full")
    spool = lab.repo / "spool" / "objects"

    def nothing(c: Ctx, r: Result) -> str:
        err = r.data["error"]["code"]
        n = lab.sql("SELECT COUNT(*) FROM versions WHERE state NOT IN ('FAILED')")[0][0]
        files = len(list(spool.glob("*/*")))
        assert n == 0 and files == 0, (n, files)
        return f"{err}: відмова до стійкого приймання, прийнятих версій 0, файлів у spool {files}"

    def quota(c: Ctx, r: Result) -> str:
        used = lab.ledger("SELECT COALESCE(SUM(length),0) FROM charges")[0][0]
        assert used <= 200_000, used
        return expect_state(lab, "DEGRADED", "v", False)(c, r) + f"; на вузлі зайнято {used} B ≤ квоти 200000 B"
    return lab.setup() + [
        Step("локальний диск «заповнено»: min_free_bytes більше за вільне місце",
             action=lambda c: lab.set_limits(min_free_bytes=10 ** 18)),
        lab.step("backup відмовляє до приймання", "backup", lab.src, expect=(5,), check=nothing),
        Step("spool «заповнено»: spool_max_bytes = 200 КБ", action=lambda c: lab.set_limits(
            min_free_bytes=0, spool_max_bytes=200_000)),
        lab.step("backup відмовляє посеред захоплення й прибирає свої файли", "backup", lab.src, expect=(5,),
                 check=nothing),
        Step("повернути межі; віддалений вузол із квотою 200 КБ", action=lambda c: lab.set_limits(
            spool_max_bytes=20 << 30)),
        lab.stop_node(),
        lab.start_node(quota=200_000),
        lab.step("backup --wait: вузол відповідає 507, повтори вичерпано", "backup", lab.src, "--wait",
                 expect=(4,), check=quota),
        lab.stop_node("зупинити вузол, щоб збільшити квоту"),
        lab.start_node(),
        lab.step("repair після звільнення місця на вузлі", "repair"),
        lab.step("стан", "status", V("v"), check=expect_state(lab, "PROTECTED")),
        *_restore(lab, "v", "m5"),
    ]


def m6(ctx: Ctx) -> list[Step]:
    lab = Lab(ctx, "m6-corruption")
    dest = lab.dir / "restored-bad"

    def manifest_path(root: Path) -> Path:
        moid = lab.sql("SELECT manifest_oid FROM versions LIMIT 1")[0][0]
        return root / "objects" / moid[:2] / moid

    def damage(c: Ctx) -> str:
        chunks = [p for p in lab.objects_on(lab.disk) if p != manifest_path(lab.disk)]
        chunk = chunks[0]                                  # різні блоки: у кожного лишається ціла копія
        data = bytearray(chunk.read_bytes())
        data[len(data) // 2] ^= 0xFF
        chunk.write_bytes(bytes(data))
        gone = lab.node_dir / "objects" / chunks[1].parent.name / chunks[1].name
        gone.unlink()
        return f"змінено 1 байт у блоці {chunk.name[:12]}… на disk-a, видалено блок {gone.name[:12]}… на node-b"

    def corrupt_manifest(root: Path):
        def act(c: Ctx) -> str:
            p = manifest_path(root)
            data = bytearray(p.read_bytes())
            data[-5] ^= 0x01
            p.write_bytes(bytes(data))
            return f"пошкоджено маніфест {p.name[:12]}… на {root.name}"
        return act

    def bad_verify(c: Ctx, r: Result) -> str:
        v = r.data["verify"]
        assert v["corrupt"] == 1 and v["missing"] == 1, v
        return f"verify виявив: пошкоджено {v['corrupt']}, відсутні {v['missing']} з {v['checked']}"

    def refused(c: Ctx, r: Result) -> str:
        assert r.data["error"]["code"] in ("E_UNAVAILABLE", "E_AUTH"), r.data
        assert not dest.exists() or not any(dest.rglob("*")), "з'явилися часткові файли"
        return f"{r.data['error']['code']}: restore відмовив, жодного часткового файла не записано"
    return lab.setup() + [
        lab.step("backup --wait", "backup", lab.src, "--wait", check=expect_state(lab, "PROTECTED", "v")),
        Step("пошкодити копію на одному backend-і й видалити на іншому", action=damage),
        lab.step("verify перечитує копії (код 7)", "verify", expect=(7,), check=bad_verify),
        lab.step("версія більше не заявлена protected (DEGRADED, код 4)", "status", V("v"), expect=(4,),
                 check=expect_state(lab, "DEGRADED", protected=False)),
        lab.step("repair відновлює копії з цілих джерел", "repair"),
        lab.step("verify після repair", "verify", check=lambda c, r: (
            f"усі {r.data['verify']['checked']} копій цілі" if r.data["verify"]["ok"] == r.data["verify"]["checked"]
            else _fail(str(r.data["verify"])))),
        Step("пошкодити маніфест на disk-a", action=corrupt_manifest(lab.disk)),
        *_restore(lab, "v", "m6-fallback"),
        Step("пошкодити маніфест і на node-b (цілих копій маніфесту не лишилось)",
             action=corrupt_manifest(lab.node_dir)),
        lab.step("restore без цілого маніфесту", "restore", V("v"), dest, expect=(7,), check=refused),
    ]


def m7(ctx: Ctx) -> list[Step]:
    lab = Lab(ctx, "m7-catalog-loss")
    new_repo = lab.dir / "repo-recovered"
    env = {"LOG3_VAULT_REPO": str(new_repo)}

    def lose(c: Ctx) -> str:
        lab.repo.rename(lab.dir / "repo-lost")
        return "каталог сховища (config.json, catalog.sqlite, spool) недоступний: перейменовано в repo-lost"
    return lab.setup() + [
        lab.step("backup --wait", "backup", lab.src, "--wait", check=expect_state(lab, "PROTECTED", "v")),
        Step("втрата основного каталогу", action=lose),
        Step("recover з backend-каталогу (підписані службові файли, фраза з середовища)",
             lab.v("recover", "disk", "--type", "localdir", "--path", lab.disk, "--domain", "disk-a"), env=env,
             check=lambda c, r: f"імпортовано версій {r.data['versions']}, відхилено підробок "
                                f"{r.data['rejected_meta']}"),
        Step("стан у відтвореному сховищі", lab.v("status", V("v")), env=env, expect=(0, 3, 4),
             check=lambda c, r: f"стан {r.data['version']['state']} (копії вузла ще не звірено)"),
        Step("repair: копії на вузлі визнаються ідемпотентно", lab.v("repair"), env=env),
        Step("стан", lab.v("status", V("v")), env=env, check=expect_state(lab, "PROTECTED")),
        Step("відновити з відтвореного сховища", lab.v("restore", V("v"), lab.dir / "restored-m7"), env=env,
             check=lambda c, r: lab.same_tree(lab.dir / "restored-m7")),
    ]


def m8(ctx: Ctx) -> list[Step]:
    lab = Lab(ctx, "m8-restart-repair-gc")

    def change(c: Ctx) -> str:
        (lab.src / "новий файл.txt").write_text("друга версія\n" * 1000, encoding="utf-8")
        return "у джерело додано файл"

    def drop_node_object(c: Ctx) -> str:
        p = lab.objects_on(lab.node_dir)[0]
        p.unlink()
        return f"на вузлі видалено об'єкт {p.name[:12]}…"

    def gc_crashed(c: Ctx, r: Result) -> str:
        deleted = lab.sql("SELECT COUNT(*) FROM versions WHERE state='DELETED'")[0][0]
        refs = lab.sql("SELECT COUNT(*) FROM objects o WHERE NOT EXISTS(SELECT 1 FROM version_objects vo "
                       "WHERE vo.oid=o.oid)")[0][0]
        assert deleted == 1 and refs > 0, (deleted, refs)
        return f"аварія після позначення: версій DELETED {deleted}, об'єктів без посилань ще {refs}"

    def gc_done(c: Ctx, r: Result) -> str:
        refs = lab.sql("SELECT COUNT(*) FROM objects o WHERE NOT EXISTS(SELECT 1 FROM version_objects vo "
                       "WHERE vo.oid=o.oid)")[0][0]
        assert refs == 0 and r.data["gc"]["delete_errors"] == 0
        return f"gc завершено: прибрано об'єктів {r.data['gc']['objects']}"
    return lab.setup() + [
        Step("retention 0 днів, keep_last 1 (стара версія підлягає gc)",
             action=lambda c: lab.set_policy(retention_days=0, keep_last=1)),
        lab.step("backup v1", "backup", lab.src, "--wait", check=expect_state(lab, "PROTECTED", "v1")),
        Step("змінити джерело", action=change),
        lab.step("backup v2", "backup", lab.src, "--wait", check=expect_state(lab, "PROTECTED", "v2")),
        Step("втрата копії на вузлі", action=drop_node_object),
        lab.step("verify виявляє нестачу", "verify", expect=(7,)),
        lab.step("repair з аварією після постановки завдань", "repair", expect=(77,),
                 env={"LOG3_VAULT_CRASHPOINT": "repair.after_requeue"}, check=_crashed),
        lab.step("новий процес: завдання repair збереглись у черзі", "queue", check=lambda c, r: (
            f"у черзі PENDING: {r.data['queue']['by_state'].get('PENDING', 0)}"
            if r.data["queue"]["by_state"].get("PENDING") else _fail("черга порожня"))),
        lab.step("repair у новому процесі", "repair"),
        lab.step("gc --apply з аварією після позначення версії", "gc", "--apply", expect=(77,),
                 env={"LOG3_VAULT_CRASHPOINT": "gc.after_mark_deleted"}, check=gc_crashed),
        lab.step("gc --apply у новому процесі", "gc", "--apply", check=gc_done),
        lab.step("verify: живі версії цілі", "verify"),
        lab.step("стан v2", "status", V("v2"), check=expect_state(lab, "PROTECTED")),
        *_restore(lab, "v2", "m8"),
    ]


def m9(ctx: Ctx) -> list[Step]:
    lab = Lab(ctx, "m9-keys")

    def key_err(c: Ctx, r: Result) -> str:
        assert r.data["error"]["code"] in ("E_KEY", "E_WEAK_KEY"), r.data
        return f"{r.data['error']['code']}: {r.data['error']['message']}"

    def change(c: Ctx) -> str:
        (lab.src / "ще.txt").write_text("нова версія", encoding="utf-8")
        return "джерело змінено"
    no_pass = {"LOG3_VAULT_PASSPHRASE": ""}
    wrong = {"LOG3_VAULT_PASSPHRASE": "wrong passphrase 0000"}
    return lab.setup() + [
        lab.step("backup без ключа", "backup", lab.src, expect=(8,), env=no_pass, check=key_err),
        lab.step("backup із неправильною фразою", "backup", lab.src, expect=(8,), env=wrong, check=key_err),
        lab.step("backup із правильною фразою", "backup", lab.src, check=expect_state(lab, "ACCEPTED_LOCAL", "v",
                                                                                   False)),
        lab.step("run без фрази: черга працює із зашифрованими об'єктами", "run", env=no_pass),
        lab.step("status без фрази", "status", V("v"), env=no_pass, check=expect_state(lab, "PROTECTED")),
        lab.step("restore без ключа", "restore", V("v"), lab.dir / "nokey", expect=(8,), env=no_pass,
                 check=key_err),
        Step("змінити джерело", action=change),
        lab.step("backup нової версії", "backup", lab.src, check=expect_state(lab, "ACCEPTED_LOCAL", "v2", False)),
        lab.step("run із неправильним токеном вузла", "run", expect=(4,), env={"LOG3_NODE_TOKEN": "wrong-token"},
                 check=lambda c, r: f"вузол відхилив автентифікацію; ATTENTION: "
                                    f"{r.data['metrics']['queue_attention_jobs']}"),
        lab.step("версія не заявлена protected", "status", V("v2"), expect=(4,),
                 check=expect_state(lab, "DEGRADED", protected=False)),
        lab.step("repair із правильним токеном", "repair"),
        lab.step("стан", "status", V("v2"), check=expect_state(lab, "PROTECTED")),
        *_restore(lab, "v2", "m9"),
    ]


SCENARIOS = [
    ("crash-after-accept", "зупинка після приймання, але до реплікації", m1),
    ("lost-response", "запис завершився, відповідь загубилась", m2),
    ("crash-block-manifest-catalog", "аварія між блоком, маніфестом і каталогом", m3),
    ("node-outage", "тривала недоступність вузла", m4),
    ("disk-full", "заповнення локального й віддаленого диска", m5),
    ("corruption", "пошкодження копії та маніфесту", m6),
    ("catalog-loss", "втрата основного каталогу", m7),
    ("restart-repair-gc", "перезапуск при незавершеному repair/GC", m8),
    ("wrong-key", "неправильний або недоступний ключ і токен", m9),
]
QUICK = ("crash-after-accept", "crash-block-manifest-catalog", "corruption", "wrong-key")


def stages() -> list[Stage]:
    return [Stage(f"matrix:{sid}", f"Матриця відмов: {title}", KIND, fn,
                  profiles=("quick", "full") if sid in QUICK else ("full",), note=SAME_DISK)
            for sid, title, fn in SCENARIOS]


def main(argv: list[str] | None = None) -> int:
    from .__main__ import main as defense_main
    return defense_main(["--stage", "matrix", "--profile", "full", *(argv or [])])


if __name__ == "__main__":
    import sys
    sys.exit(main(sys.argv[1:]))


__all__ = ["stages", "SCENARIOS", "make_mixed"]
