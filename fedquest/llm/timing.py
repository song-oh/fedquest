# timing log of model calls; timed replays look up each call's median wall time
from __future__ import annotations

import json
import os
import statistics
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import IO, Any

_lock = threading.Lock()
_out: IO[str] | None = None
_out_path: Path | None = None
_sources: tuple[Path, ...] | None = None
_table: dict[str, float] | None = None


def configure(log: str | Path | None = None, sources: Sequence[str | Path] | None = None) -> None:
    global _out_path, _sources, _table
    close()
    with _lock:
        env_log = os.environ.get("FEDQUEST_TIMING_LOG") or None
        _out_path = Path(log) if log is not None else Path(env_log) if env_log else None
        if sources is None:
            env = os.environ.get("FEDQUEST_TIMING_FROM") or ""
            sources = [p for p in env.split(os.pathsep) if p] or ([_out_path] if _out_path else [])
        _sources = tuple(Path(p) for p in sources)
        _table = None


def _ensure() -> None:
    if _sources is None:
        configure()


def note(key: str, wall_s: float, service_s: float, **extra: Any) -> None:
    global _out
    _ensure()
    if _out_path is None:
        return
    line = json.dumps({"key": key, "wall_s": round(wall_s, 4), "service_s": round(service_s, 4),
                       "t": round(time.time(), 3), **extra}, sort_keys=True)
    with _lock:
        if _out is None:
            _out_path.parent.mkdir(parents=True, exist_ok=True)
            _out = _out_path.open("a", encoding="utf-8")
        _out.write(line + "\n")
        _out.flush()


def _load() -> dict[str, float]:
    walls: dict[str, list[float]] = {}
    for path in _sources or ():
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                walls.setdefault(d["key"], []).append(float(d["wall_s"]))
    return {k: statistics.median(v) for k, v in walls.items()}


def lookup(key: str) -> float | None:
    global _table
    _ensure()
    with _lock:
        if _table is None:
            if _out is not None:
                _out.flush()
            _table = _load()
        return _table.get(key)


def close() -> None:
    global _out
    with _lock:
        if _out is not None:
            _out.close()
        _out = None
