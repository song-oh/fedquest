from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

from fedquest.types import ExtractionResult
from fedquest.utils import connect_sqlite

_SCHEMA = """
CREATE TABLE IF NOT EXISTS extractions (
    oid TEXT NOT NULL, version TEXT NOT NULL, qhash TEXT NOT NULL DEFAULT '',
    payload TEXT NOT NULL, created REAL,
    PRIMARY KEY (oid, version, qhash)
);
"""


class SiteFactStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = connect_sqlite(self.path)
        with self._lock, self._db:
            self._db.executescript(_SCHEMA)

    def get(self, oid: str, version: str, qhash: str = "") -> ExtractionResult | None:
        with self._lock:
            row = self._db.execute(
                "SELECT payload FROM extractions WHERE oid = ? AND version = ? AND qhash = ?", (oid, version, qhash)
            ).fetchone()
        return None if row is None else ExtractionResult.from_dict(json.loads(row[0]))

    def put(self, result: ExtractionResult, qhash: str = "", key: str | None = None) -> None:
        payload = result.to_dict()
        payload["raw_text"] = None
        with self._lock, self._db:
            self._db.execute(
                "INSERT OR REPLACE INTO extractions (oid, version, qhash, payload, created) VALUES (?, ?, ?, ?, ?)",
                (result.oid, key or result.version, qhash, json.dumps(payload), time.time()),
            )

    def stats(self) -> dict[str, Any]:
        with self._lock:
            rows = self._db.execute("SELECT version, qhash, payload FROM extractions").fetchall()
        by_modality: dict[str, int] = {}
        facts = prompt = completion = 0
        for _, _, payload in rows:
            d = json.loads(payload)
            by_modality[d.get("modality", "?")] = by_modality.get(d.get("modality", "?"), 0) + 1
            facts += len(d.get("facts", []))
            prompt += d.get("usage", {}).get("prompt_tokens", 0)
            completion += d.get("usage", {}).get("completion_tokens", 0)
        return {
            "extractions": len(rows),
            "materializable": sum(1 for v, _, _ in rows if not v.startswith("mgrag-q")),
            "facts": facts,
            "by_modality": by_modality,
            "recorded_prompt_tokens": prompt,
            "recorded_completion_tokens": completion,
        }

    def close(self) -> None:
        with self._lock:
            self._db.close()
