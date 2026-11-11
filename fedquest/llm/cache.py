# sqlite record/replay cache of model calls, keyed by the whole request
from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from fedquest.utils import connect_sqlite

log = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (
    key TEXT PRIMARY KEY,
    model TEXT,
    reply TEXT,
    usage TEXT,
    created REAL
)
"""


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def cache_key(
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    temperature: float,
    json_mode: bool,
    extra_body: dict[str, Any] | None = None,
) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "json_mode": bool(json_mode),
        "extra_body": extra_body or {},
    }
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()


LOCK_TRIES = 4
RO_ENV = "FEDQUEST_LLM_CACHE_RO"


def read_only_paths() -> list[Path]:
    out: list[Path] = []
    for p in os.environ.get(RO_ENV, "").split(os.pathsep):
        if not p.strip():
            continue
        path = Path(p).expanduser()
        out += sorted(path.glob("*.sqlite")) if path.is_dir() else [path]
    return out


def open_read_only(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path.resolve()}?mode=ro&immutable=1", uri=True, check_same_thread=False)


class LLMCache:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = connect_sqlite(self.path)
        with self._lock, self._db:
            self._db.execute("PRAGMA synchronous=NORMAL")
            self._db.execute(_SCHEMA)
        own = self.path.resolve()
        self._ro = [open_read_only(p) for p in read_only_paths() if p.exists() and p.resolve() != own]

    def get(self, key: str) -> dict[str, Any] | None:
        for attempt in range(LOCK_TRIES):
            try:
                with self._lock:
                    row = self._db.execute("SELECT reply, usage FROM calls WHERE key = ?", (key,)).fetchone()
                    for db in self._ro if row is None else ():
                        row = db.execute("SELECT reply, usage FROM calls WHERE key = ?", (key,)).fetchone()
                        if row is not None:
                            break
                return None if row is None else {"reply": row[0], "usage": json.loads(row[1])}
            except sqlite3.OperationalError as e:
                if attempt == LOCK_TRIES - 1:
                    log.warning("LLM cache %s unreadable (%s): treated as a miss", self.path, e)
                    return None
                time.sleep(0.5 * (attempt + 1))
        return None

    def put(self, key: str, model: str, reply: str, usage: dict[str, Any]) -> None:
        for attempt in range(LOCK_TRIES):
            try:
                with self._lock, self._db:
                    self._db.execute(
                        "INSERT OR IGNORE INTO calls (key, model, reply, usage, created) VALUES (?, ?, ?, ?, ?)",
                        (key, model, reply, json.dumps(usage), time.time()),
                    )
                return
            except sqlite3.OperationalError as e:
                if attempt == LOCK_TRIES - 1:
                    log.warning("LLM cache %s not written (%s): the reply stands, unrecorded", self.path, e)
                    return
                time.sleep(0.5 * (attempt + 1))

    def close(self) -> None:
        with self._lock:
            self._db.close()
            for db in self._ro:
                db.close()

    def __len__(self) -> int:
        with self._lock:
            return int(self._db.execute("SELECT COUNT(*) FROM calls").fetchone()[0])
