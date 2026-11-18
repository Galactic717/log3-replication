"""Захоплення джерела: читання -> блоки -> стиснення -> шифрування -> spool -> публікація версії.

Порядок запису, синхронізації та публікації:
  1. рядок версії CAPTURING (нічого не обіцяно);
  2. кожен новий блок: tmp-файл у spool -> fsync -> rename (об'єкт ще ні на що не впливає);
  3. маніфест так само;
  4. fsync каталогів spool, у які щось лягло;
  5. ОДНА транзакція SQLite (synchronous=FULL): об'єкти, посилання версії, завдання черги,
     стан ACCEPTED_LOCAL. Її COMMIT і є публікацією: до нього версії не існує, після нього
     всі її дані й метадані лежать у spool і каталозі, а джерело більше не потрібне.
Аварія до кроку 5 лишає тільки файли-сироти в spool і рядок CAPTURING; repair позначає таку
версію FAILED і прибирає сиріт. Оригінали ніколи не змінюються й не видаляються.

Файл, що змінився під час читання (розмір або mtime до і після не збіглись), перечитується
до 3 разів; якщо не вдалося — пропускається із записом у skipped, а backup завершується кодом
PARTIAL. Це захист від рваного читання одного файла, а не узгоджений знімок: активні бази даних
потребують snapshot або інтеграції із застосунком, і цей режим такого не обіцяє.
"""
from __future__ import annotations

import hashlib
import json
import os
import stat as stat_mod
import time
import uuid
from pathlib import Path

import zstandard

from . import codecs, manifest
from .advisor import CompressionContext, CompressionOutcome, load_advisor, sample_entropy
from .catalog import now_ns
from .cryptobox import KIND_CHUNK, KIND_MANIFEST, OBJECT_FORMAT_VERSION, SUITE
from .errors import Exit, VaultError, crashpoint
from .fsutil import free_bytes, fsync_dir, write_durable
from .keys import MAX_OBJECTS_PER_KEY, sign_meta
from .learn.runtime import group_key, runtime_for
from .replicate import ensure_jobs, pointer_doc, pointer_name
from .repo import Repo
from .states import VersionState

READ_RETRIES = 3
HEARTBEAT_NS = 5 * 10 ** 9
_I63 = (1 << 63) - 1


class _Changed(Exception):
    pass


class _Capture:
    def __init__(self, repo: Repo, version_id: str, op_id: str):
        self.repo, self.cat, self.box = repo, repo.cat, repo.box
        self.version_id, self.op_id = version_id, op_id
        self.chunk_size = repo.cfg["chunk_size_bytes"]
        comp = repo.cfg["compression"]
        self.codec, self.level = comp["codec"], comp["level"]
        self.advisor = load_advisor(comp.get("advisor"))
        self.learn = runtime_for(repo, version_id, op_id, self.advisor)   # збій навчання -> базова політика
        self.objs: dict[str, dict] = {}        # усі об'єкти версії: oid -> поля каталогу
        self.new: list[str] = []               # записані в spool цим захопленням
        self.dirs: set[Path] = set()
        self.spool_bytes = self.cat.scalar("SELECT COALESCE(SUM(stored_len),0) FROM objects WHERE in_spool=1")
        self.beat = now_ns()

    def heartbeat(self) -> None:
        if now_ns() - self.beat > HEARTBEAT_NS:
            self.beat = now_ns()
            self.cat.x("UPDATE versions SET heartbeat_ns=? WHERE version_id=?", (self.beat, self.version_id))

    def put(self, kind: int, oid: str, codec_id: int, plain_len: int, payload: bytes) -> None:
        lim = self.repo.limits
        blob = self.box.seal(kind, oid, codec_id, plain_len, payload)
        if self.spool_bytes + len(blob) > lim["spool_max_bytes"]:
            raise VaultError("E_SPOOL_FULL", f"spool досяг межі {lim['spool_max_bytes']} B", Exit.CAPACITY)
        if free_bytes(self.repo.spool) - len(blob) < lim["min_free_bytes"]:
            raise VaultError("E_DISK_FULL", f"на диску spool лишилось менше {lim['min_free_bytes']} B", Exit.CAPACITY)
        dest = self.repo.spool_path(oid)
        self.new.append(oid)
        _, sha = write_durable(dest, [blob], tmp_dir=self.repo.spool / "tmp", sync_dir=False)
        self.dirs.add(dest.parent)
        self.spool_bytes += len(blob)
        self.objs[oid] = {"kind": kind, "stored_len": len(blob), "sha256": sha, "plain_len": plain_len,
                          "codec": codec_id, "new": True}

    def chunk(self, data: bytes, suffix: str, file_size: int, index: int, file_key: str | None = None) -> str:
        oid = self.box.object_id(KIND_CHUNK, data)
        if oid in self.objs:
            return oid
        row = self.cat.one("SELECT o.*, EXISTS(SELECT 1 FROM replicas r WHERE r.oid=o.oid AND r.state='CONFIRMED') "
                           "AS replicated FROM objects o WHERE o.oid=?", (oid,))
        if row and ((row["in_spool"] and self.repo.spool_path(oid).is_file()) or row["replicated"]):
            self.objs[oid] = {**{k: row[k] for k in ("kind", "stored_len", "sha256", "plain_len", "codec")},
                              "new": False}
            return oid                                      # дедуплікація: блок уже стійко збережено
        ctx = CompressionContext(self.op_id, self.version_id, suffix, file_size, index, len(data), data[:16],
                                 sample_entropy(data), self.codec, self.level)
        rec, fell_back, handle = self.learn.decide(ctx, data, oid, file_key)
        t0, c0 = time.perf_counter_ns(), time.thread_time_ns()
        codec_id, payload = codecs.encode_best(data, codecs.BY_NAME[rec.codec].id, rec.level)  # невигідне -> Store
        encode_ns, cpu_ns = time.perf_counter_ns() - t0, time.thread_time_ns() - c0
        self.put(KIND_CHUNK, oid, codec_id, len(data), payload)
        self.learn.outcome(handle, codec_used=codecs.get(codec_id).name, payload_len=len(payload),
                           stored_len=self.objs[oid]["stored_len"], raw_len=len(data), wall_ns=encode_ns, cpu_ns=cpu_ns)
        self.advisor.observe(CompressionOutcome(
            self.op_id, self.version_id, oid, suffix, index, rec.codec, codecs.get(codec_id).name, len(data),
            self.objs[oid]["stored_len"], encode_ns, fell_back))
        self.learn.probe(handle, data)
        return oid

    def file_key(self, root: str, rel: str) -> str | None:
        """Знеособлений ключ лінії вже відомого файла (до читання); новий файл — None."""
        row = self.cat.one("SELECT file_id FROM files WHERE root=? AND path=?", (root, rel))
        return group_key(self.repo.cfg["repo_id"], row[0]) if row else None

    def read_file(self, path: Path, file_key: str | None = None) -> tuple[bytes, int, list[str], os.stat_result]:
        """(sha256, розмір, блоки по порядку, stat) або _Changed, якщо файл змінився під час читання."""
        suffix = path.suffix[1:].lower()
        h, size, chunks = hashlib.sha256(), 0, []
        with open(path, "rb") as f:
            before = os.fstat(f.fileno())
            while data := f.read(self.chunk_size):
                h.update(data)
                chunks.append(self.chunk(data, suffix, before.st_size, len(chunks), file_key))
                size += len(data)
                self.heartbeat()
            after = os.fstat(f.fileno())
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or size != after.st_size:
            raise _Changed
        return h.digest(), size, chunks, after


def _walk(root: Path, skip: Path):
    """(відносний шлях, абсолютний шлях, тип) у стабільному порядку; посилання не розкриваються."""
    stack = [("", root)]
    while stack:
        prefix, directory = stack.pop()
        try:
            entries = sorted(os.scandir(directory), key=lambda e: e.name)
        except OSError as e:
            yield prefix or ".", directory, f"skip:unreadable:{type(e).__name__}"
            continue
        for e in entries:
            rel, path = f"{prefix}{e.name}", Path(e.path)
            if path == skip:
                continue
            if e.is_symlink() or getattr(e, "is_junction", lambda: False)():
                yield rel, path, "skip:symlink"
            elif e.is_dir(follow_symlinks=False):
                yield rel, path, "dir"
                stack.append((rel + "/", path))
            elif e.is_file(follow_symlinks=False):
                yield rel, path, "file"
            else:
                yield rel, path, "skip:special"


def _file_id(cat, root: str, rel: str, st: os.stat_result, src_root: Path, pending: list) -> str:
    row = cat.one("SELECT file_id FROM files WHERE root=? AND path=?", (root, rel))
    if row:
        return row[0]
    dev, ino = st.st_dev & _I63, st.st_ino & _I63           # SQLite зберігає знакові 64 біти
    if ino:                                           # той самий inode під новим іменем = перейменування
        row = cat.one("SELECT file_id, path FROM files WHERE root=? AND dev=? AND ino=?", (root, dev, ino))
        if row and not os.path.lexists(src_root.joinpath(*row["path"].split("/"))):
            pending.append((row["file_id"], root, rel, dev, ino))
            return row["file_id"]
    fid = str(uuid.uuid4())
    pending.append((fid, root, rel, dev, ino))
    return fid


def _precheck(repo: Repo, priority: int) -> tuple[int, int]:
    """Відмова ДО стійкого приймання: політика копій, черга, spool, вільне місце."""
    lim, pol = repo.limits, repo.policy
    n_domains = len(set(repo.domains().values()))
    if n_domains < pol["min_copies"]:
        raise VaultError("E_POLICY", f"політика вимагає {pol['min_copies']} незалежних копій, а налаштовані backend-и "
                         f"дають {n_domains} домен(ів) відмов; додайте backend в іншому домені", Exit.POLICY)
    queued = repo.cat.scalar("SELECT COUNT(*) FROM jobs WHERE state IN ('PENDING','RUNNING')")
    if queued >= lim["queue_max_jobs"]:
        raise VaultError("E_QUEUE_FULL", f"у черзі {queued} завдань, межа {lim['queue_max_jobs']}", Exit.CAPACITY)
    if free_bytes(repo.spool) < lim["min_free_bytes"]:
        raise VaultError("E_DISK_FULL", f"на диску spool менше {lim['min_free_bytes']} B вільного місця", Exit.CAPACITY)
    target = min(n_domains, pol["min_copies"] + (pol["extra_copies_high_priority"] if priority == 2 else 0))
    return pol["min_copies"], target


def _finish_learning(cat, cap: _Capture, version_id: str, *, published: bool) -> None:
    """Журнал навчання пишеться ПІСЛЯ публікації версії й поза її транзакцією: збій журналу
    не змінює ні стану версії, ні черги."""
    try:
        summary = cap.learn.finish(published)
        if cap.learn.errors:
            cat.event("learning_errors", cap.learn.errors, getattr(cap.learn, "note", version_id))
        cat.event("learning", None, json.dumps({"version_id": version_id, **summary}, ensure_ascii=False)[:2000])
    except Exception:
        pass


def backup(repo: Repo, source: Path, *, name: str | None = None, priority: int = 1,
           op_id: str | None = None) -> tuple[str, bool]:
    """Приймає версію локально. Повертає (version_id, чи це повтор уже прийнятої операції)."""
    cat = repo.cat
    source = Path(os.path.abspath(source))
    if not source.exists():
        raise VaultError("E_NOT_FOUND", f"джерела {source} не існує", Exit.NOT_FOUND)
    op_id = op_id or str(uuid.uuid4())
    for prev in cat.q("SELECT version_id, state, heartbeat_ns FROM versions WHERE op_id=? "
                      "AND state NOT IN ('FAILED','DELETED')", (op_id,)):
        if prev["state"] != "CAPTURING":
            return prev["version_id"], True                 # повтор операції не створює другої версії
        if now_ns() - prev["heartbeat_ns"] < repo.limits["capture_stale_seconds"] * 1e9:
            raise VaultError("E_BUSY", f"операція {op_id} ще виконується іншим процесом")
        cat.set_state(prev["version_id"], VersionState.FAILED, error="захоплення перервано")
    box = repo.box                                          # секрет перевіряється до будь-яких записів
    min_copies, target = _precheck(repo, priority)
    limit = repo.cfg["crypto"].get("max_objects_per_key", MAX_OBJECTS_PER_KEY)
    if cat.sealed_count(box.keys.key_id) >= limit:         # межа випадкових nonce -> новий ключ даних
        from .keyops import rotate_key
        rotate_key(repo, reason="nonce_limit")
        box = repo.box
    repo.sign_config()                                      # backend-и отримають актуальну підписану конфігурацію
    root, kind = (source, "dir") if source.is_dir() else (source.parent, "file")
    version_id, created = uuid.uuid4().hex, now_ns()
    name = name or source.name or "root"
    cat.x("INSERT INTO versions(version_id, op_id, name, root, state, priority, min_copies, target_copies, "
          "created_ns, heartbeat_ns) VALUES(?,?,?,?,?,?,?,?,?,?)",
          (version_id, op_id, name, str(root), VersionState.CAPTURING, priority, min_copies, target, created, created))
    cap = _Capture(repo, version_id, op_id)
    try:
        entries, contents, skipped, pending_files, total = [], {}, [], [], 0
        items = _walk(root, repo.path) if kind == "dir" else [(source.name, source, "file")]
        for rel, path, typ in items:
            try:
                manifest.safe_parts(rel, windows=False)
                rel.encode("utf-8")
            except (VaultError, UnicodeEncodeError):
                skipped.append({"path": rel.encode("utf-8", "replace").decode(), "reason": "unsupported_name"})
                continue
            if typ.startswith("skip:"):
                skipped.append({"path": rel, "reason": typ[5:]})
            elif typ == "dir":
                st = os.lstat(path)
                entries.append({"path": rel, "type": "dir", "mtime_ns": st.st_mtime_ns,
                                "mode": stat_mod.S_IMODE(st.st_mode)})
            else:
                result, reason = None, ""
                pre_key = cap.file_key(str(root), rel)
                for _ in range(READ_RETRIES):
                    try:
                        result = cap.read_file(path, pre_key)
                        break
                    except _Changed:
                        reason = "changed_during_read"
                    except OSError as e:
                        reason = f"unreadable:{type(e).__name__}"
                        break
                if result is None:
                    skipped.append({"path": rel, "reason": reason})
                    continue
                digest, size, chunks, st = result
                cid = box.content_id(digest)
                contents.setdefault(cid, {"size": size, "sha256": digest.hex(), "chunks": chunks})
                fid = _file_id(cat, str(root), rel, st, root, pending_files)
                cap.learn.file_done(pre_key, fid, cid)
                entries.append({"path": rel, "type": "file", "content_id": cid, "size": size, "file_id": fid,
                                "mtime_ns": st.st_mtime_ns, "mode": stat_mod.S_IMODE(st.st_mode)})
                total += size
        used = {oid for c in contents.values() for oid in c["chunks"]}
        doc = manifest.dump({
            "format": manifest.FORMAT, "format_version": manifest.FORMAT_VERSION, "version_id": version_id,
            "operation_id": op_id, "repo_id": repo.cfg["repo_id"], "name": name, "created_at_ns": created,
            "created_at": manifest.rfc3339(created), "source": {"root": str(root), "kind": kind},
            "chunk_size_bytes": cap.chunk_size,
            "crypto": {"suite": SUITE, "key_id": box.keys.key_id, "object_format_version": OBJECT_FORMAT_VERSION},
            "codecs": {str(c.id): {"name": c.name, "version": c.version} for c in codecs.REGISTRY.values()},
            "entries": entries, "contents": contents, "skipped": skipped,
            "objects": {oid: {k: cap.objs[oid][k] for k in ("plain_len", "stored_len", "sha256", "codec")}
                        for oid in sorted(used)}})
        manifest_oid = box.object_id(KIND_MANIFEST, doc)
        cap.put(KIND_MANIFEST, manifest_oid, codecs.ZSTD, len(doc), zstandard.ZstdCompressor(level=3).compress(doc))
        for d in cap.dirs:
            fsync_dir(d)
        fsync_dir(repo.spool / "objects")
        crashpoint("capture.before_publish")
        with cat.tx():                                      # публікація: одна стійка транзакція
            for oid in (*sorted(used), manifest_oid):
                o = cap.objs[oid]
                if o["new"]:
                    cat.x("INSERT INTO objects(oid, kind, stored_len, sha256, plain_len, codec, in_spool, created_ns, "
                          "key_id) VALUES(?,?,?,?,?,?,1,?,?) ON CONFLICT(oid) DO UPDATE SET "
                          "stored_len=excluded.stored_len, sha256=excluded.sha256, codec=excluded.codec, in_spool=1, "
                          "key_id=excluded.key_id",
                          (oid, o["kind"], o["stored_len"], o["sha256"], o["plain_len"], o["codec"], created,
                           box.keys.key_id))
                    cat.x("DELETE FROM replicas WHERE oid=? AND state!='CONFIRMED'", (oid,))
                elif cat.one("SELECT 1 FROM objects WHERE oid=?", (oid,)) is None:
                    raise VaultError("E_RACE", "блок зник із каталогу під час захоплення (паралельний gc); повторіть")
                cat.x("INSERT INTO version_objects VALUES(?,?)", (version_id, oid))
            for f in pending_files:
                cat.x("INSERT OR REPLACE INTO files(file_id, root, path, dev, ino) VALUES(?,?,?,?,?)", f)
            ensure_jobs(repo, version_id, repo.targets(target), priority)
            queued = cat.scalar("SELECT COUNT(*) FROM jobs WHERE state IN ('PENDING','RUNNING')")
            if queued > repo.limits["queue_max_jobs"]:
                raise VaultError("E_QUEUE_FULL", f"версія потребує черги з {queued} завдань, межа "
                                 f"{repo.limits['queue_max_jobs']}", Exit.CAPACITY)
            cat.count_sealed(box.keys.key_id, len(cap.new))
            mo = cap.objs[manifest_oid]
            pointer = sign_meta(box.keys.meta_key(), pointer_name(version_id), pointer_doc(
                version_id, manifest_oid, mo["stored_len"], mo["sha256"], created)).decode("utf-8")
            accepted = now_ns()
            cat.set_state(version_id, VersionState.ACCEPTED_LOCAL, manifest_oid=manifest_oid, accepted_ns=accepted,
                          pointer=pointer,
                          retention_until_ns=accepted + int(repo.policy["retention_days"] * 86400e9),
                          files=sum(e["type"] == "file" for e in entries), bytes=total, skipped=len(skipped))
            if cap.advisor.errors:
                cat.event("advisor_errors", cap.advisor.errors, version_id)
    except BaseException as e:                              # версію не прийнято: прибрати лише її власні нові файли
        cat.count_sealed(box.keys.key_id, len(cap.new))     # nonce використано, хоч об'єкти й не прийнято
        for oid in cap.new:
            if cat.one("SELECT 1 FROM objects WHERE oid=? AND in_spool=1", (oid,)) is None:
                repo.spool_path(oid).unlink(missing_ok=True)
        text = e.message if isinstance(e, VaultError) else type(e).__name__
        cat.set_state(version_id, VersionState.FAILED, error=text[:300])
        _finish_learning(cat, cap, version_id, published=False)
        raise
    cat.checkpoint()
    _finish_learning(cat, cap, version_id, published=True)
    crashpoint("capture.after_publish")
    return version_id, False
