"""Повне відтворення results/ однією командою:  python scripts/reproduce.py

  results/c1.md                       статус 12 приймальних кейсів C1 (з прогону pytest)
  results/tables/02_copies.md         вплив max 1/2/3 копій: S, R, Q, біти
  results/tables/03_ablation.md       без перцептрона / без ДП / без ГА / повна система
  results/tables/04_compress_encrypt.md  стиснення до і після шифрування
  results/tables/05_scenarios.md      поведінка у трьох сценаріях
  results/final_metrics.json          усі числа вище в машинному вигляді

Кожне число — середнє за відкритими seeds 0..4. Якщо materials/ нема, прогін іде на
dev-конфігу й dev-моделі реплікації — це видно в шапці кожної таблиці.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
import zlib
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from log3 import huffman  # noqa: E402
from log3.config import MATERIALS_CONFIG, load_config  # noqa: E402
from log3.payload import build_payload  # noqa: E402
from log3.pipeline import SCENARIOS, load_session, run  # noqa: E402
from log3.records import pack_records  # noqa: E402
from log3.replica import build_replica  # noqa: E402

RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
SEEDS = range(5)

C1 = [  # (№, кейс, власник)
    (1, "Один запис кодується й відновлюється побайтово", "Гліб"),
    (2, "Порожній журнал не валить систему", "Гліб"),
    (3, "NO_COMPRESSION відновлює сирі записи точно", "Гліб"),
    (4, "Канонічні коди відновлюються лише з довжин", "Yaroslav"),
    (5, "Вироджений алфавіт має код довжини 1", "Yaroslav"),
    (6, "Корисність ДП ≥ жадібного на всіх виданих наборах", "Yaroslav"),
    (7, "Штатний сценарій: відновлено всі критичні", "Гліб"),
    (8, "Деградований: частка невідновлених критичних ≤ порогу", "Гліб"),
    (9, "Перцептрон: macro-F1 ≥ F1_min на holdout", "Artem"),
    (10, "Змінений шифротекст або HMAC відхиляється", "Гліб"),
    (11, "Невірний пароль, обрізана репліка, magic, reserved — без аварії", "Гліб"),
    (12, "Усі еталонні пакети читаються; прогін у ліміті часу", "Гліб"),
]


def env_note() -> str:
    if MATERIALS_CONFIG.exists():
        return "Середовище: materials/team_config.json."
    return ("**Середовище: dev-конфіг і dev-модель реплікації (log3/devsim.py) — пакет викладача "
            "ще не покладено в materials/. Числа ілюструють методику, не офіційний Q.**")


def table(title: str, head: list[str], rows: list[list], note: str = "") -> str:
    fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)  # noqa: E731
    lines = [f"# {title}", "", env_note(), ""]
    if note:
        lines += [note, ""]
    lines += ["| " + " | ".join(head) + " |", "|" + " :-- |" * len(head)]
    lines += ["| " + " | ".join(fmt(v) for v in r) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def avg_runs(scenario: str, cfg, records, **kw) -> dict:
    rs = [run(scenario, s, cfg=cfg, records=records, **kw) for s in SEEDS]
    keys = ("S", "R", "T", "Q", "b_stored", "copies_assigned", "copies_placed",
            "replicas_written", "seconds")
    return {k: mean(getattr(r, k) for r in rs) for k in keys} | {
        "R_min": min(r.R for r in rs), "total_critical": rs[0].total_critical}


def c1_status() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        xml_path = Path(tmp) / "junit.xml"
        subprocess.run([sys.executable, "-m", "pytest", "-q", "-k", "test_c1_",
                        f"--junitxml={xml_path}"], cwd=ROOT, capture_output=True)
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
    rows = []
    for n, title, owner in C1:
        got = status.get(n)
        verdict = ("нема тесту (зона не здана)" if not got else
                   "НЕ ПРОЙДЕНО" if "НЕ ПРОЙДЕНО" in got else got[0])
        rows.append([n, title, owner, verdict])
    return table("Приймальні кейси C1", ["№", "Кейс", "Власник", "Статус"], rows,
                 "Генерується `scripts/reproduce.py` з прогону `pytest -k test_c1_`.")


def compress_encrypt(cfg, records) -> tuple[str, dict]:
    raw = pack_records(records)
    key_pw = cfg.password
    salt = os.urandom(16)
    payload_c = build_payload(records, no_compression_threshold=10.0)
    blob_ce = build_replica(records, key_pw, replica_group_id=0, batch_seq=0, salt=salt,
                            no_compression_threshold=10.0)
    blob_e = build_replica(records, key_pw, replica_group_id=0, batch_seq=1, salt=salt, compress=False)
    ct = blob_e[43:-32]
    huff_ct = len(build_payload_like(ct))
    data = {
        "raw_bytes": len(raw),
        "huffman_payload": len(payload_c),
        "compress_then_encrypt": len(blob_ce),
        "encrypt_only": len(blob_e),
        "encrypt_then_huffman": huff_ct + 75,
        "zlib_raw": len(zlib.compress(raw, 9)),
        "zlib_ciphertext": len(zlib.compress(ct, 9)),
    }
    rows = [
        ["сирі записи", data["raw_bytes"], 1.0],
        ["Гаффман (payload, без шифрування)", data["huffman_payload"], data["huffman_payload"] / len(raw)],
        ["стиснути → зашифрувати (репліка LOG3)", data["compress_then_encrypt"], data["compress_then_encrypt"] / len(raw)],
        ["лише зашифрувати (NO_COMPRESSION)", data["encrypt_only"], data["encrypt_only"] / len(raw)],
        ["зашифрувати → Гаффман", data["encrypt_then_huffman"], data["encrypt_then_huffman"] / len(raw)],
        ["довідково: zlib сирих записів", data["zlib_raw"], data["zlib_raw"] / len(raw)],
        ["довідково: zlib шифротексту", data["zlib_ciphertext"], data["zlib_ciphertext"] / len(raw)],
    ]
    note = (f"Увесь сеанс ({len(records)} записів) однією реплікою. Шифротекст AES-CTR статистично "
            "схожий на випадкові байти: частоти байтів рівні, Гаффман дає ≈8 біт на байт плюс таблицю, "
            "тому стискати треба до шифрування.")
    return table("Стиснення до і після шифрування", ["Варіант", "Байти", "Від сирого"], rows, note), data


def build_payload_like(data: bytes) -> bytes:
    """Гаффман над довільними байтами з тими самими накладними витратами таблиці, що й у LOG3."""
    lengths = huffman.code_lengths(data)
    return bytes(14 + 2 * len(lengths)) + huffman.encode(data, lengths)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    records = load_session(cfg)
    out: dict = {"seeds": list(SEEDS), "records": len(records),
                 "dev_environment": not MATERIALS_CONFIG.exists()}

    (RESULTS / "c1.md").write_text(c1_status(), encoding="utf-8")

    scen = {s: avg_runs(s, cfg, records) for s in SCENARIOS}
    out["scenarios"] = scen
    rows = [[s, v["S"], v["R"], v["R_min"], v["T"], v["Q"], int(v["b_stored"]),
             f'{v["copies_placed"]:.0f}/{v["copies_assigned"]:.0f}', v["seconds"]]
            for s, v in scen.items()]
    (TABLES / "05_scenarios.md").write_text(table(
        "Поведінка у трьох сценаріях реплікації",
        ["Сценарій", "S", "R", "R min", "T", "Q", "B_stored, біт", "копій розміщено/призначено", "t, с"],
        rows, f"Критичних подій у сеансі: {scen['normal']['total_critical']}."), encoding="utf-8")

    copies = {f"{s}/{cap}": avg_runs(s, cfg, records, copies_cap=cap)
              for s in ("degraded", "emergency") for cap in (1, 2, 3)}
    out["copies"] = copies
    rows = [[k.split("/")[0], k.split("/")[1], v["S"], v["R"], v["Q"], int(v["b_stored"])]
            for k, v in copies.items()]
    (TABLES / "02_copies.md").write_text(table(
        "Вплив максимальної кількості копій", ["Сценарій", "max копій", "S", "R", "Q", "B_stored, біт"],
        rows), encoding="utf-8")

    abl = {}
    for s in ("degraded", "emergency"):
        abl[f"{s}/повна система"] = avg_runs(s, cfg, records)
        abl[f"{s}/без перцептрона"] = avg_runs(s, cfg, records, use_nn=False)
        abl[f"{s}/без ДП (жадібний)"] = avg_runs(s, cfg, records, use_dp=False)
        abl[f"{s}/без стиснення"] = avg_runs(s, cfg, records, compress=False)
    out["ablation"] = abl
    rows = [[*k.split("/"), v["S"], v["R"], v["Q"]] for k, v in abl.items()]
    (TABLES / "03_ablation.md").write_text(table(
        "Вплив блоків", ["Сценарій", "Варіант", "S", "R", "Q"], rows,
        "Перцептрон поки заглушка (усі події класу 0), тож «без перцептрона» = повна система. "
        "Рядок «без ГА» з'явиться, коли ГА (Artem) видасть параметри: зараз повна система "
        "працює на параметрах з конфігу, тобто це і є «без ГА»."), encoding="utf-8")

    text, out["compress_encrypt"] = compress_encrypt(cfg, records)
    (TABLES / "04_compress_encrypt.md").write_text(text, encoding="utf-8")

    (RESULTS / "final_metrics.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                                encoding="utf-8")
    print((RESULTS / "c1.md").read_text(encoding="utf-8"))
    print((TABLES / "05_scenarios.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
