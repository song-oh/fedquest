# stable hashes, atomic JSON and JSONL writes, a directory lock, sqlite in WAL mode
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import re
import sqlite3
import threading
import time
import unicodedata
from collections.abc import Awaitable, Callable, Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def stable_hash(*parts: Any, n: int = 16) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(json.dumps(p, sort_keys=True, default=str, ensure_ascii=False).encode())
        h.update(b"\x1f")
    return h.hexdigest()[:n]


def stable_int(*parts: Any) -> int:
    return int(stable_hash(*parts, n=15), 16)


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f"{path.suffix}.{os.getpid()}.{threading.get_ident()}.tmp")
    n = 0
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    tmp.replace(path)
    return n


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f"{path.suffix}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(path)


@contextlib.contextmanager
def dir_lock(path: str | Path, timeout: float = 120.0, stale: float = 600.0) -> Iterator[None]:
    lock = Path(path)
    lock.parent.mkdir(parents=True, exist_ok=True)
    deadline, held = time.monotonic() + timeout, False
    while True:
        try:
            lock.mkdir()
            held = True
            break
        except FileExistsError:
            try:
                if time.time() - lock.stat().st_mtime > stale:
                    lock.rmdir()
                    continue
            except OSError:
                continue
            if time.monotonic() > deadline:
                break
            time.sleep(0.05)
    try:
        yield
    finally:
        if held:
            with contextlib.suppress(OSError):
                lock.rmdir()


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


_WS = re.compile(r"\s+")


def clean_text(s: str) -> str:
    return _WS.sub(" ", unicodedata.normalize("NFKC", s)).strip()


def truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[: max(0, n - 1)].rstrip() + "…"


def approx_tokens(s: str) -> int:
    return max(1, len(s) // 4)


async def gather_limited(
    items: Sequence[T], fn: Callable[[T], Awaitable[R]], limit: int
) -> list[R]:
    sem = asyncio.Semaphore(max(1, limit))

    async def run(x: T) -> R:
        async with sem:
            return await fn(x)

    return list(await asyncio.gather(*(run(x) for x in items)))


def connect_sqlite(path: str | Path) -> sqlite3.Connection:
    db = sqlite3.connect(path, check_same_thread=False, timeout=30)
    for attempt in range(20):
        try:
            db.execute("PRAGMA journal_mode=WAL")
            break
        except sqlite3.OperationalError:
            if attempt == 19:
                raise
            time.sleep(0.05 * (attempt + 1))
    db.execute("PRAGMA busy_timeout=30000")
    return db
