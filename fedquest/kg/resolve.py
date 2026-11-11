# fuzzy lookup of a value among entity keys sharing its first or last token
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from rapidfuzz import fuzz

from fedquest.kg.normalize import norm_entity, same_norm

RESOLVE_LIMIT = 3


class KeyIndex:
    def __init__(self, keys: Iterable[str]) -> None:
        self.keys: set[str] = {k for k in keys if k}
        self.by_end: dict[str, set[str]] = defaultdict(set)
        for k in self.keys:
            tokens = k.split()
            for t in {tokens[0], tokens[-1]}:
                self.by_end[t].add(k)

    def __len__(self) -> int:
        return len(self.keys)

    def resolve(self, value: str, limit: int = RESOLVE_LIMIT) -> list[tuple[str, float]]:
        key = norm_entity(value)
        if not key:
            return []
        if key in self.keys:
            return [(key, 1.0)]
        tokens = key.split()
        near = self.by_end.get(tokens[0], set()) | self.by_end.get(tokens[-1], set())
        hits = sorted(((k, fuzz.ratio(key, k) / 100.0) for k in near if same_norm(key, k)),
                      key=lambda kv: (-kv[1], kv[0]))
        return hits[:limit]
