"""Підключення навчання до тракту capture: рішення, проби, журнал, drift, fallback.

Один LearnRuntime на операцію backup. Порядок для кожного нового (не дедуплікованого) блока:
  1. базова політика (BaselineAdvisor або advisor із compression.advisor) дає policy_action;
  2. ознаки блока (обмежені вікна); у shadow — висновок кандидата, який НЕ змінює рішення;
     в active — рішення моделі; тракт усе одно замінить невигідне стиснення на Store;
  3. після кодування — фактичний результат (байти, час, CPU потоку);
  4. проба: з імовірністю probe.rate за hash(object_id) — незалежно від ознак і від рішення —
     той самий блок кодується всіма іншими допустимими діями (без запису) у межах бюджету CPU.
Записи пишуться в журнал пакетами; лінія файлів оновлюється лише після публікації версії.

Що б не сталося в навчанні (некоректна конфігурація, пошкоджений state.json чи модель, помилка
журналу, ознак чи висновку), backup продовжується на базовій політиці; кількість помилок
потрапляє в події каталогу (learning_errors). Restore навчання не використовує взагалі.
"""
from __future__ import annotations

import hashlib
import json
import time

import numpy as np

from ..advisor import CompressionContext, Recommendation, SafeAdvisor
from ..fsutil import free_bytes
from . import features as F
from .actions import CostModel, measure, parse_action
from .journal import Journal, decision_id, now_ns, pack
from .models import ModelError, load_artifact
from .settings import LearnDir, load_state, priority_active, save_state, settings

FLUSH_EVERY = 512
HIGH_ENTROPY = 7.5 / 8


def group_key(repo_id: str, file_id: str | None) -> str:
    """Знеособлений ключ лінії файла: hash, а не сам file_id чи шлях."""
    return hashlib.sha256(f"{repo_id}:{file_id or '-'}".encode()).hexdigest()[:20]


class NullRuntime:
    """Навчання вимкнено: лише базова політика, жодних ознак, журналу чи проб."""
    active = False
    errors = 0

    def __init__(self, advisor: SafeAdvisor):
        self.advisor = advisor

    def decide(self, ctx: CompressionContext, data: bytes, object_id: str, file_key: str | None):
        rec, fell_back = self.advisor.recommend(ctx)
        return rec, fell_back, None

    def outcome(self, *a, **k) -> None:
        return None

    def probe(self, *a, **k) -> None:
        return None

    def file_done(self, *a, **k) -> None:
        return None

    def finish(self, published: bool) -> dict:
        return {"enabled": False}


class LearnRuntime(NullRuntime):
    active = True

    def __init__(self, repo, version_id: str, op_id: str, advisor: SafeAdvisor):
        super().__init__(advisor)
        self.repo, self.version_id, self.op_id = repo, version_id, op_id
        self.s = settings(repo.cfg)
        self.dir = LearnDir(repo.path)
        self.state, err = load_state(self.dir)
        self.notes: list[str] = [err] if err else []
        self.errors = int(bool(err))
        self.actions = [parse_action(a) for a in self.s["actions"]]
        self.cost = CostModel(self.s["cost"]["copies"] or repo.policy["min_copies"],
                              float(self.s["cost"]["cpu_second_equiv_bytes"]))
        self.journal = Journal(self.dir.journal)
        self.mode, self.model, self.model_id, self.shadow, self.shadow_id = self.state["mode"], None, None, None, None
        self.fallback_reason = None
        paused, frozen = self.state["paused"], self.state["frozen"]
        if self.mode == "active":
            if frozen:
                self.fallback_reason = f"оновлення заморожено: {self.state.get('freeze_reason')}"
            else:
                self.model, self.model_id = self._load(self.state["active_model"])
        if self.state["candidate_model"] and not paused and self.mode in ("shadow", "active"):
            self.shadow, self.shadow_id = self._load(self.state["candidate_model"], quiet=True)
        self.telemetry = self._telemetry()
        self.probe_on, why = self._probe_allowed(paused)
        if why:
            self.notes.append(why)
        self.started = now_ns()
        self.cpu0 = time.thread_time_ns()
        self.buf = {"decisions": [], "outcomes": [], "probes": []}
        self.pending: dict[str, list[dict]] = {}        # file_key -> рішення цього файла (уточнення group_key)
        self.lineage_updates: list[tuple[str, str]] = []
        self.handles: dict[str, dict] = {}
        self.t = dict.fromkeys(("chunks", "probes", "probe_cpu_ns", "feature_ns", "inference_ns", "encode_cpu_ns",
                                "raw_bytes", "stored_bytes", "fallbacks", "known_chunks", "known_high_entropy",
                                "known_files", "known_changed"), 0)
        self.quarantine = 0
        self.reference = self._reference()

    # ---- ініціалізація -------------------------------------------------------------------------
    def _load(self, model_id: str | None, *, quiet: bool = False):
        if not model_id:
            if not quiet:
                self.fallback_reason = "активну модель не задано"
            return None, None
        try:
            policy, _ = load_artifact(self.dir.models, model_id, admissible=[a.key for a in self.actions])
            return policy, model_id
        except ModelError as e:
            self.errors += 1
            self.notes.append(f"модель {model_id}: {e}")
            if not quiet:
                self.fallback_reason = f"модель недоступна: {e}"
            return None, None

    def _telemetry(self) -> F.Telemetry:
        try:
            cat, lim = self.repo.cat, self.repo.limits
            queued = cat.scalar("SELECT COUNT(*) FROM jobs WHERE state IN ('PENDING','RUNNING')")
            spool = cat.scalar("SELECT COALESCE(SUM(stored_len),0) FROM objects WHERE in_spool=1")
            return F.Telemetry(min(1.0, queued / max(1, lim["queue_max_jobs"])),
                               min(1.0, spool / max(1, lim["spool_max_bytes"])), F.load_per_cpu())
        except Exception:
            return F.Telemetry()

    def _probe_allowed(self, paused: bool) -> tuple[bool, str | None]:
        """Проби поступаються захисту даних: черга, диск, restore/repair, навантаження, пауза."""
        p, lim = self.s["probe"], self.repo.limits
        if paused:
            return False, "навчання на паузі: проби вимкнено"
        if p["rate"] <= 0 or p["max_chunks_per_operation"] <= 0:
            return False, None
        if (self.telemetry.queue_fill or 0) > p["max_queue_fill"]:
            return False, "черга реплікації заповнена: проби відкладено"
        try:
            if free_bytes(self.repo.spool) < lim["min_free_bytes"] * p["min_free_factor"]:
                return False, "мало вільного місця: проби відкладено"
        except OSError:
            return False, "вільне місце невідоме: проби відкладено"
        if priority_active(self.repo.path):
            return False, "іде restore/repair: проби відкладено"
        if (self.telemetry.load_per_cpu or 0) > p["max_load_per_cpu"]:
            return False, "високе навантаження: проби відкладено"
        return True, None

    def _reference(self) -> float | None:
        """Типова частка високоентропійних блоків відомих файлів за останні операції (без карантину)."""
        rows = self.journal.q("SELECT detail FROM operations WHERE quarantine=0 AND published=1 "
                              "ORDER BY finished_ns DESC LIMIT 5")
        vals = []
        for r in rows:
            try:
                d = json.loads(r["detail"] or "{}")
                if d.get("known_chunks", 0) >= self.s["drift"]["min_chunks"]:
                    vals.append(d["known_high_entropy"] / d["known_chunks"])
            except (ValueError, KeyError, TypeError):
                continue
        return float(np.median(vals)) if vals else None

    # ---- рішення ---------------------------------------------------------------------------------
    def lineage(self, file_key: str | None) -> F.Lineage | None:
        if not file_key:
            return None
        got = self.journal.lineage(file_key)
        return F.Lineage(*got) if got else None

    def decide(self, ctx: CompressionContext, data: bytes, object_id: str, file_key: str | None):
        rec, fell_back = self.advisor.recommend(ctx)
        try:
            return self._decide(ctx, data, object_id, file_key, rec, fell_back)
        except Exception as e:                            # навчання ніколи не зупиняє backup
            self.errors += 1
            self.notes.append(f"рішення: {type(e).__name__}")
            return rec, fell_back, None

    def _decide(self, ctx, data, oid: str, file_key, rec: Recommendation, fell_back: bool):
        decided = now_ns()
        policy_key = f"{rec.codec}:{rec.level if rec.codec != 'store' else 0}"
        fv = F.extract(data, suffix=ctx.path_suffix, file_size=ctx.file_size_bytes, chunk_index=ctx.chunk_index,
                       chunk_size=self.repo.cfg["chunk_size_bytes"], head_entropy_bits=ctx.sample_entropy_bits,
                       lineage=self.lineage(file_key), telemetry=self.telemetry)
        chosen, model_key, fallback, inf_ns = rec, None, self.fallback_reason, 0
        x = fv.values[None]
        if not np.isfinite(x).all():
            fallback = "нескінченні ознаки"
        elif self.model is not None and fallback is None:
            t0 = time.perf_counter_ns()
            a = self.actions_of(self.model)[int(self.model.choose(x)[0])]
            inf_ns = time.perf_counter_ns() - t0
            model_key, chosen = a.key, Recommendation(a.codec, a.level)
        elif self.shadow is not None and self.mode == "shadow":
            t0 = time.perf_counter_ns()                    # shadow: рекомендація записується, рішення не змінює
            model_key = self.actions_of(self.shadow)[int(self.shadow.choose(x)[0])].key
            inf_ns = time.perf_counter_ns() - t0
        if fallback:
            self.t["fallbacks"] += 1
        known = fv.get("lineage_known") > 0
        if known:
            self.t["known_chunks"] += 1
            self.t["known_high_entropy"] += fv.get("entropy_sample") > HIGH_ENTROPY
            self._check_drift()
        chosen_key = f"{chosen.codec}:{chosen.level if chosen.codec != 'store' else 0}"
        d = {"decision_id": decision_id(self.version_id, oid), "operation_id": self.op_id,
             "version_id": self.version_id, "object_id": oid, "chunk_index": ctx.chunk_index,
             "group_key": file_key or "", "decided_ns": decided, "feature_schema": F.schema_id(),
             "features": pack(fv.values), "mask": fv.mask.tobytes(), "feature_ns": fv.cost_ns, "inference_ns": inf_ns,
             "mode": self.mode if not fallback else "baseline", "model_id": self.model_id or self.shadow_id,
             "policy_action": policy_key, "model_action": model_key, "chosen_action": chosen_key,
             "fallback": fallback, "probe_p": self.s["probe"]["rate"] if self.probe_on else 0.0,
             "quarantine": self.quarantine, "published": 1}
        self.t["chunks"] += 1
        self.t["feature_ns"] += fv.cost_ns
        self.t["inference_ns"] += inf_ns
        self.buf["decisions"].append(d)
        self.pending.setdefault(file_key or "", []).append(d)
        self.handles[d["object_id"]] = d
        return chosen, fell_back, d

    def actions_of(self, policy):
        return [parse_action(a) for a in policy.actions]

    def outcome(self, handle: dict | None, *, codec_used: str, payload_len: int, stored_len: int, raw_len: int,
                wall_ns: int, cpu_ns: int) -> None:
        if handle is None:
            return
        try:
            self.buf["outcomes"].append({
                "decision_id": handle["decision_id"], "result_ns": now_ns(), "action": handle["chosen_action"],
                "codec_used": codec_used, "raw_len": raw_len, "payload_len": payload_len, "stored_len": stored_len,
                "wall_ns": wall_ns, "cpu_ns": cpu_ns,
                "forced_store": int(codec_used == "store" and not handle["chosen_action"].startswith("store"))})
            self.t["encode_cpu_ns"] += cpu_ns
            self.t["raw_bytes"] += raw_len
            self.t["stored_bytes"] += stored_len
            if len(self.buf["decisions"]) >= FLUSH_EVERY:
                self._flush()
        except Exception:
            self.errors += 1

    def probe(self, handle: dict | None, data: bytes) -> None:
        """Повна інформація для навчання: усі інші допустимі дії на тому самому блоці, без запису."""
        if handle is None or not self.probe_on or handle["probe_p"] <= 0:
            return
        p = self.s["probe"]
        if int(handle["object_id"][:8], 16) / 2**32 >= p["rate"]:
            return
        spent = (time.thread_time_ns() - self.cpu0) - self.t["probe_cpu_ns"]
        if (self.t["probes"] >= p["max_chunks_per_operation"]
                or self.t["probe_cpu_ns"] >= p["max_cpu_seconds_per_operation"] * 1e9
                or self.t["probe_cpu_ns"] > p["max_cpu_fraction"] * max(spent, 1)):
            return
        try:
            c0 = time.thread_time_ns()
            for a in self.actions:
                if a.key == handle["chosen_action"]:
                    continue
                m, _ = measure(data, a)
                self.buf["probes"].append({"decision_id": handle["decision_id"], "action": a.key,
                                           "measured_ns": now_ns(), "codec_used": m.codec_used,
                                           "payload_len": m.payload_len, "wall_ns": m.wall_ns, "cpu_ns": m.cpu_ns})
            self.t["probe_cpu_ns"] += time.thread_time_ns() - c0
            self.t["probes"] += 1
        except Exception:
            self.errors += 1

    def file_done(self, pre_key: str | None, file_id: str, content_id: str) -> None:
        """Файл прочитано: уточнюємо ключ лінії для групування й запам'ятовуємо мітку вмісту."""
        try:
            key = group_key(self.repo.cfg["repo_id"], file_id)
            for d in self.pending.pop(pre_key or "", []):
                d["group_key"] = key
            known = self.journal.lineage(key)
            mark = hashlib.sha256(content_id.encode()).hexdigest()[:16]
            if known:
                self.t["known_files"] += 1
                last = self.journal.scalar("SELECT last_content FROM lineage WHERE group_key=?", (key,))
                self.t["known_changed"] += last != mark
            self.lineage_updates.append((key, mark))
        except Exception:
            self.errors += 1

    # ---- drift ----------------------------------------------------------------------------------
    def _check_drift(self) -> None:
        dr = self.s["drift"]
        n = self.t["known_chunks"]
        if self.quarantine or self.reference is None or n < dr["min_chunks"] or n % dr["check_every_chunks"]:
            return
        frac = self.t["known_high_entropy"] / n
        if frac - self.reference > dr["high_entropy_jump"]:
            self._quarantine(f"сплеск високоентропійних змін відомих файлів: {frac:.2f} проти типових "
                             f"{self.reference:.2f}")

    def _quarantine(self, reason: str) -> None:
        """Аномалію не вважаємо ні шкідливістю, ні важливістю: рішення цієї операції не стають
        навчальною нормою, оновлення моделей заморожуються, робочі рішення — базова політика."""
        self.quarantine = 1
        for d in self.buf["decisions"]:
            d["quarantine"] = 1
        self.journal.db.execute("UPDATE decisions SET quarantine=1 WHERE version_id=?", (self.version_id,))
        self.model, self.fallback_reason = None, f"drift: {reason}"
        self.notes.append(reason)
        st, _ = load_state(self.dir)
        if not st["frozen"]:
            st.update(frozen=True, freeze_reason=reason)
            save_state(self.dir, st, event="freeze", detail=reason)
            self.journal.lifecycle("freeze", None, reason)

    # ---- завершення ------------------------------------------------------------------------------
    def _flush(self) -> None:
        self.journal.write_decisions(self.buf["decisions"], self.buf["outcomes"], self.buf["probes"])
        self.buf = {"decisions": [], "outcomes": [], "probes": []}

    def finish(self, published: bool) -> dict:
        try:
            if self.quarantine:
                for d in self.buf["decisions"]:
                    d["quarantine"] = 1
            if published and self.t["known_files"] >= 10 and \
                    self.t["known_changed"] / self.t["known_files"] > self.s["drift"]["rewrite_fraction"] and \
                    not self.quarantine:
                self._quarantine(f"масове переписування: змінено {self.t['known_changed']} з "
                                 f"{self.t['known_files']} відомих файлів")
            self._flush()
            if not published:
                self.journal.db.execute("UPDATE decisions SET published=0 WHERE version_id=?", (self.version_id,))
            else:
                self.journal.update_lineage(self.lineage_updates, now_ns())
            self.journal.write_operation({
                "version_id": self.version_id, "operation_id": self.op_id, "started_ns": self.started,
                "finished_ns": now_ns(), "mode": self.mode, "model_id": self.model_id or self.shadow_id,
                "published": int(published), "chunks": self.t["chunks"], "probes": self.t["probes"],
                "probe_cpu_ns": self.t["probe_cpu_ns"], "feature_ns": self.t["feature_ns"],
                "inference_ns": self.t["inference_ns"], "encode_cpu_ns": self.t["encode_cpu_ns"],
                "raw_bytes": self.t["raw_bytes"], "stored_bytes": self.t["stored_bytes"],
                "fallbacks": self.t["fallbacks"], "quarantine": self.quarantine,
                "detail": json.dumps({k: self.t[k] for k in ("known_chunks", "known_high_entropy", "known_files",
                                                             "known_changed")} | {"notes": self.notes[:10]},
                                     ensure_ascii=False)})
            self.journal.prune(self.s["journal"]["max_decisions"])
        except Exception as e:
            self.errors += 1
            self.notes.append(f"журнал: {type(e).__name__}")
        finally:
            try:
                self.journal.close()
            except Exception:
                pass
        return {"enabled": True, "mode": self.mode, "model_id": self.model_id, "fallback": self.fallback_reason,
                "chunks": self.t["chunks"], "probes": self.t["probes"], "errors": self.errors,
                "quarantine": bool(self.quarantine), "notes": self.notes[:10]}


def runtime_for(repo, version_id: str, op_id: str, advisor: SafeAdvisor):
    """LearnRuntime або NullRuntime; збій ініціалізації навчання не блокує backup."""
    try:
        s = settings(repo.cfg)
        if not s["enabled"]:
            return NullRuntime(advisor)
        st, _ = load_state(LearnDir(repo.path))
        if st["disabled"]:
            return NullRuntime(advisor)
        return LearnRuntime(repo, version_id, op_id, advisor)
    except Exception as e:
        null = NullRuntime(advisor)
        null.errors = 1
        null.note = f"навчання не запущено: {type(e).__name__}: {e}"[:300]
        return null
