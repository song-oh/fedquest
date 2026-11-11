from __future__ import annotations

import os
import re
import threading
import time
from collections import OrderedDict
from collections.abc import Sequence
from typing import Any

import numpy as np

from fedquest.llm import prompts

_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
MENTION_BONUS = 0.02
GAP = " … "
MIN_FILL = 40
SCORE_DIGITS = 4
VIEW_VERSION = "edge-view-v1"
CAPACITY = int(os.environ.get("FEDQUEST_EDGE_INDEX", "50000"))


def sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE.split(text or "") if s.strip()]


def pack(sents: Sequence[str], order: Sequence[int], room: int) -> str:
    if not order:
        return ""
    shown: dict[int, str] = {}
    used = 0
    for i in order:
        extra = len(sents[i]) + (len(GAP) if shown else 0)
        if used + extra <= room:
            shown[i] = sents[i]
            used += extra
    if not shown:
        best = sents[order[0]]
        return best if len(best) <= room else best[: max(0, room - 1)].rstrip() + "…"
    rest = room - used - len(GAP)
    left = next((i for i in order if i not in shown), None)
    if left is not None and rest >= MIN_FILL:
        shown[left] = sents[left][: rest - 1].rstrip() + "…"
    keys = sorted(shown)
    out = shown[keys[0]]
    for prev, i in zip(keys, keys[1:], strict=False):
        out += (" " if i == prev + 1 else GAP) + shown[i]
    return out


def order(sents: Sequence[str], vecs: np.ndarray, key: np.ndarray, values: Sequence[str]) -> list[int]:
    if not len(sents):
        return []
    scores = np.asarray(vecs, dtype=np.float32) @ np.asarray(key, dtype=np.float32)
    if values:
        named = np.array([any(prompts.mentions(s, v) for v in values) for s in sents], dtype=bool)
        scores = scores + MENTION_BONUS * named
    rounded = np.round(scores.astype(np.float64), SCORE_DIGITS)
    return sorted(range(len(sents)), key=lambda i: (-float(rounded[i]), i))


def view(sents: Sequence[str], vecs: np.ndarray, key: np.ndarray, values: Sequence[str], room: int) -> str:
    return pack(sents, order(sents, vecs, key, values), room)


def index_name(encoder: str) -> str:
    from fedquest.vdb.encoders import ENCODERS

    spec = ENCODERS.get(encoder)
    return f"{encoder}|{spec.model_id if spec is not None else ''}|{VIEW_VERSION}"


class EdgeIndex:
    def __init__(self, capacity: int = CAPACITY) -> None:
        self.capacity = max(1, int(capacity))
        self._items: OrderedDict[tuple[str, str, str], tuple[list[str], np.ndarray]] = OrderedDict()
        self._lock = threading.Lock()
        self.built = 0
        self.sentences = 0
        self.hits = 0
        self.seconds = 0.0

    def __len__(self) -> int:
        return len(self._items)

    def get(self, encoder_name: str, encoder: Any, objects: Sequence[tuple[str, str, str]],
            ) -> tuple[list[tuple[list[str], np.ndarray]], int, float]:
        keys = [(encoder_name, oid, ver) for oid, ver, _ in objects]
        out: list[tuple[list[str], np.ndarray] | None] = []
        with self._lock:
            for k in keys:
                hit = self._items.get(k)
                if hit is not None:
                    self._items.move_to_end(k)
                out.append(hit)
            self.hits += sum(1 for x in out if x is not None)
        missing = [i for i, x in enumerate(out) if x is None]
        if not missing:
            return [x for x in out if x is not None], 0, 0.0
        split = {i: sentences(objects[i][2]) for i in missing}
        flat = [s for i in missing for s in split[i]]
        t0 = time.perf_counter()
        vecs = encoder.encode_docs(flat) if flat else np.zeros((0, 1), dtype=np.float32)
        took = time.perf_counter() - t0
        start = 0
        with self._lock:
            for i in missing:
                n = len(split[i])
                item = (split[i], np.asarray(vecs[start:start + n], dtype=np.float16))
                start += n
                out[i] = item
                self._items[keys[i]] = item
                self._items.move_to_end(keys[i])
            while len(self._items) > self.capacity:
                self._items.popitem(last=False)
            self.built += len(missing)
            self.sentences += len(flat)
            self.seconds += took
        return [x for x in out if x is not None], len(missing), took

    def stats(self) -> dict[str, float]:
        with self._lock:
            held = sum(len(sents) for sents, _ in self._items.values())
            return {"objects": float(len(self._items)), "held_sentences": float(held), "built": float(self.built),
                    "sentences": float(self.sentences), "hits": float(self.hits), "seconds": round(self.seconds, 3)}


INDEX = EdgeIndex()
