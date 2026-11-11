# wire log: site replies recorded in SQLite, replayed by question scope
from __future__ import annotations

import asyncio
import contextvars
import hashlib
import json
import os
import threading
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fedquest.llm.cache import canonical_json
from fedquest.utils import connect_sqlite, truncate

if TYPE_CHECKING:
    from fedquest.federation.base import Federation, SiteEndpoint

RECORD_ENV = "FEDQUEST_RECORD_WIRE"

_SCHEMA = (
    "CREATE TABLE IF NOT EXISTS replies (sha TEXT PRIMARY KEY, reply TEXT NOT NULL)",
    "CREATE TABLE IF NOT EXISTS wire (scope TEXT NOT NULL, key TEXT NOT NULL, federation TEXT, site TEXT, action TEXT,"
    " kwargs TEXT, sha TEXT NOT NULL, created REAL, PRIMARY KEY (scope, key))",
    "CREATE INDEX IF NOT EXISTS wire_key ON wire (key)",
    "CREATE TABLE IF NOT EXISTS llm (key TEXT NOT NULL, scope TEXT NOT NULL, PRIMARY KEY (key, scope))",
)

_scope: contextvars.ContextVar[str] = contextvars.ContextVar("fedquest_wire_scope", default="")
_in_site: contextvars.ContextVar[bool] = contextvars.ContextVar("fedquest_in_site", default=False)

MISSES: list[str] = []


class ReplayMiss(LookupError):
    pass


def miss(message: str) -> ReplayMiss:
    MISSES.append(message)
    return ReplayMiss(message)


def _plain(obj: Any) -> Any:
    item = getattr(obj, "item", None)
    if callable(item):
        return item()
    raise TypeError(f"site reply holds a {type(obj).__name__}, which is no JSON")


def wire_key(federation: str, site: str, action: str, args: str) -> str:
    return hashlib.sha256("\x1f".join((federation, site, action, args)).encode()).hexdigest()


@contextmanager
def wire_scope(scope: str) -> Iterator[None]:
    token = _scope.set(scope)
    try:
        yield
    finally:
        _scope.reset(token)


class WireLog:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = connect_sqlite(self.path)
        with self._lock, self._db:
            self._db.execute("PRAGMA synchronous=NORMAL")
            for statement in _SCHEMA:
                self._db.execute(statement)

    def put(self, federation: str, site: str, action: str, args: str, reply: Any, scope: str = "") -> Any:
        text = json.dumps(reply, ensure_ascii=False, separators=(",", ":"), default=_plain)
        sha = hashlib.sha256(text.encode()).hexdigest()
        with self._lock, self._db:
            self._db.execute("INSERT OR IGNORE INTO replies (sha, reply) VALUES (?, ?)", (sha, text))
            self._db.execute("INSERT OR REPLACE INTO wire (scope, key, federation, site, action, kwargs, sha, created) "
                             "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                             (scope, wire_key(federation, site, action, args), federation, site, action, args, sha,
                              time.time()))
        return json.loads(text)

    def get(self, federation: str, site: str, action: str, args: str, scope: str = "") -> Any | None:
        with self._lock:
            # the question's own scope first, then an unscoped reply, then the newest
            row = self._db.execute(
                "SELECT r.reply FROM wire w JOIN replies r ON r.sha = w.sha WHERE w.key = ? "
                "ORDER BY w.scope = ? DESC, w.scope = '' DESC, w.created DESC LIMIT 1",
                (wire_key(federation, site, action, args), scope)).fetchone()
        return None if row is None else json.loads(row[0])

    def recorded(self, federation: str, site: str, action: str, scope: str) -> int:
        with self._lock:
            return int(self._db.execute("SELECT COUNT(*) FROM wire WHERE federation = ? AND site = ? AND action = ? "
                                        "AND scope = ?", (federation, site, action, scope)).fetchone()[0])

    def note_llm(self, key: str, scope: str) -> None:
        with self._lock, self._db:
            self._db.execute("INSERT OR IGNORE INTO llm (key, scope) VALUES (?, ?)", (key, scope))

    @staticmethod
    def _under(prefixes: Iterable[str]) -> tuple[str, list[str]]:
        prefixes = list(prefixes)
        if not prefixes:
            return "0", []
        return "(" + " OR ".join("substr(scope, 1, ?) = ?" for _ in prefixes) + ")", \
            [x for p in prefixes for x in (len(p), p)]

    def llm_keys(self, prefixes: Iterable[str]) -> set[str]:
        cond, params = self._under(prefixes)
        with self._lock:
            return {k for (k,) in self._db.execute(f"SELECT DISTINCT key FROM llm WHERE {cond}", params)}

    def scopes(self, prefixes: Iterable[str]) -> set[str]:
        cond, params = self._under(prefixes)
        with self._lock:
            return {s for (s,) in self._db.execute(f"SELECT scope FROM wire WHERE {cond} UNION "
                                                   f"SELECT scope FROM llm WHERE {cond}", params + params)}

    def copy_into(self, dest: WireLog, prefixes: Iterable[str], federations: Iterable[str]) -> None:
        cond, params = self._under(prefixes)
        feds = list(federations)
        info = f"(scope = '' AND action = 'info' AND federation IN ({','.join('?' * len(feds))}))" if feds else "0"
        with dest._lock:
            dest._db.execute("ATTACH DATABASE ? AS src", (str(self.path),))
            try:
                with dest._db:
                    dest._db.execute(
                        f"INSERT INTO wire SELECT * FROM src.wire WHERE {cond} OR {info} ON CONFLICT (scope, key) DO "
                        "UPDATE SET federation = excluded.federation, site = excluded.site, action = excluded.action, "
                        "kwargs = excluded.kwargs, sha = excluded.sha, created = excluded.created "
                        "WHERE excluded.created > wire.created", [*params, *feds])
                    dest._db.execute("INSERT OR IGNORE INTO replies SELECT * FROM src.replies "
                                     "WHERE sha IN (SELECT sha FROM main.wire)")
                    dest._db.execute(f"INSERT OR IGNORE INTO llm SELECT * FROM src.llm WHERE {cond}", params)
            finally:
                dest._db.execute("DETACH DATABASE src")

    def counts(self) -> dict[str, int]:
        with self._lock:
            return {t: int(self._db.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]) for t in ("wire", "replies",
                                                                                                   "llm")}

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def __len__(self) -> int:
        return self.counts()["wire"]


_logs: dict[Path, WireLog] = {}
_logs_lock = threading.Lock()


def open_log(path: str | Path) -> WireLog:
    key = Path(path).expanduser().resolve()
    with _logs_lock:
        if key not in _logs:
            _logs[key] = WireLog(key)
        return _logs[key]


def close_logs() -> None:
    with _logs_lock:
        for log in _logs.values():
            log.close()
        _logs.clear()


def recording() -> bool:
    return bool(os.environ.get(RECORD_ENV))


def recorder() -> WireLog | None:
    path = os.environ.get(RECORD_ENV)
    return open_log(path) if path else None


def note_llm_key(key: str) -> None:
    scope = _scope.get()
    log = recorder() if scope and not _in_site.get() else None
    if log is not None:
        log.note_llm(key, scope)


class RecordingEndpoint:
    def __init__(self, inner: SiteEndpoint, federation: str, log: WireLog) -> None:
        self.inner, self.federation, self.log = inner, federation, log
        self.name = inner.name

    async def call(self, action: str, **kwargs: Any) -> Any:
        return (await self.call_timed(action, **kwargs))[0]

    async def call_timed(self, action: str, **kwargs: Any) -> tuple[Any, float, float]:
        args = canonical_json(kwargs)
        token = _in_site.set(True)
        t0 = time.perf_counter()
        try:
            out = await self.inner.call(action, **kwargs)
        finally:
            _in_site.reset(token)
        dt = time.perf_counter() - t0
        out = await asyncio.to_thread(self.log.put, self.federation, self.name, action, args, out, _scope.get())
        return out, dt, 0.0

    async def close(self) -> None:
        await self.inner.close()


class ReplayEndpoint:
    def __init__(self, federation: str, name: str, log: WireLog, timed: bool | None = None) -> None:
        self.federation, self.name, self.log = federation, name, log
        self.timed = os.environ.get("FEDQUEST_LLM_TIMED", "") == "1" if timed is None else timed

    async def call(self, action: str, **kwargs: Any) -> Any:
        args, scope = canonical_json(kwargs), _scope.get()
        out = await asyncio.to_thread(self.log.get, self.federation, self.name, action, args, scope)
        if self.timed and out is not None and kwargs.get("rid"):
            from fedquest.llm import timing

            await asyncio.sleep(timing.lookup(f"served:{kwargs['rid']}") or 0.0)
        if out is None:
            n = await asyncio.to_thread(self.log.recorded, self.federation, self.name, action, scope)
            raise miss(f"no recorded reply for site {self.name!r} action {action!r} of {self.federation} (question run "
                       f"{scope or '-'}, where {n} other {action} requests of the site are recorded; arguments "
                       f"{truncate(args, 240)}) in {self.log.path}")
        return out

    async def close(self) -> None:
        return None


def recorded(fed: Federation, federation: str) -> Federation:
    fed.name = federation
    log = recorder()
    if log is not None:
        fed.endpoints = {s: RecordingEndpoint(e, federation, log) for s, e in fed.endpoints.items()}
    return fed


def replay_federation(federation: str, log: WireLog | None = None) -> Federation:
    from fedquest.config import load_federation
    from fedquest.federation.base import Federation
    from fedquest.paths import wire_log_path

    cfg = load_federation(federation)
    if log is None:
        path = wire_log_path()
        if not path.exists():
            raise FileNotFoundError(f"no wire log at {path} to replay {federation} from (FEDQUEST_REPLAY_WIRE names it; "
                                    f"runs record one with {RECORD_ENV}=<path>)")
        log = open_log(path)
    return Federation([ReplayEndpoint(cfg.name, s.name, log) for s in cfg.sites],
                      domains={s.name: s.data_domain for s in cfg.sites})
