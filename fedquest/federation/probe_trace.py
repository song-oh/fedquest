# optional JSONL trace of every probe round (FEDQUEST_PROBE_TRACE)
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

_lock = threading.Lock()


def path() -> Path | None:
    p = os.environ.get("FEDQUEST_PROBE_TRACE")
    return Path(p).expanduser() if p else None


_MATCH: dict[str, Any] = {}


def _wanted(scope: str) -> bool:
    pat = os.environ.get("FEDQUEST_PROBE_TRACE_MATCH")
    if not pat:
        return True
    if pat not in _MATCH:
        import re

        _MATCH[pat] = re.compile(pat)
    return bool(_MATCH[pat].search(scope))


def note_round(per_site: dict[str, list[dict[str, Any]]], skipped: int, returned: int) -> None:
    out = path()
    if out is None or not per_site:
        return
    from fedquest.federation.replay import (
        _scope,
    )

    if not _wanted(_scope.get()):
        return
    line = json.dumps({"scope": _scope.get(), "t": round(time.time(), 4), "sites": per_site, "skipped": skipped,
                       "returned": returned}, default=str, ensure_ascii=False)
    with _lock:
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def load(p: str | Path) -> dict[str, list[dict[str, Any]]]:
    rounds: dict[str, list[dict[str, Any]]] = {}
    with Path(p).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                rounds.setdefault(r["scope"], []).append(r)
    for rs in rounds.values():
        rs.sort(key=lambda r: r["t"])
    return rounds
