from __future__ import annotations

import base64
import hashlib
import math
from collections.abc import Iterable
from typing import Any

BITS_PER_KEY = 10
HASHES = 7


class BloomFilter:
    def __init__(self, m: int, k: int = HASHES, bits: bytes | bytearray | None = None) -> None:
        self.m, self.k = max(8, int(m)), int(k)
        self.bits = bytearray(bits) if bits is not None else bytearray((self.m + 7) // 8)
        if len(self.bits) != (self.m + 7) // 8:
            raise ValueError(f"a Bloom filter of {self.m} bits needs {(self.m + 7) // 8} bytes, not {len(self.bits)}")

    @classmethod
    def of(cls, keys: Iterable[str], bits_per_key: int = BITS_PER_KEY, k: int = HASHES) -> BloomFilter:
        keys = list(dict.fromkeys(keys))
        bf = cls(math.ceil(max(1, len(keys)) * bits_per_key), k)
        for key in keys:
            bf.add(key)
        return bf

    def _positions(self, key: str) -> list[int]:
        d = hashlib.sha256(key.encode("utf-8")).digest()
        h1, h2 = int.from_bytes(d[:8], "big"), int.from_bytes(d[8:16], "big") | 1
        return [(h1 + i * h2) % self.m for i in range(self.k)]

    def add(self, key: str) -> None:
        for p in self._positions(key):
            self.bits[p >> 3] |= 1 << (p & 7)

    def __contains__(self, key: str) -> bool:
        return all(self.bits[p >> 3] & (1 << (p & 7)) for p in self._positions(key))

    def to_dict(self) -> dict[str, Any]:
        return {"m": self.m, "k": self.k, "bits": base64.b64encode(bytes(self.bits)).decode("ascii")}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BloomFilter:
        return cls(d["m"], d["k"], base64.b64decode(d["bits"]))


def title_key(title: str) -> str:
    from fedquest.kg.normalize import norm_entity
    from fedquest.llm.prompts import fold

    return fold(norm_entity(title))


def title_entries(keys: Iterable[str]) -> list[str]:
    out: dict[str, None] = {}
    for key in keys:
        if key:
            out[f"t:{key}"] = None
            for w in key.split():
                out[f"w:{w}"] = None
    return list(out)


def may_hold(bf: BloomFilter, key: str) -> bool:
    if not key:
        return False
    if f"t:{key}" in bf:
        return True
    words = key.split()
    return any(len(w) >= 4 for w in words) and all(f"w:{w}" in bf for w in words)
