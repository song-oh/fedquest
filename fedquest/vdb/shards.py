# virtual shards: a site as a hash or Dirichlet slice of a parent site's objects
from __future__ import annotations

import functools
import os
import threading
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from fedquest.data.partition import page_key, shard_by_title
from fedquest.types import ObjectRecord

VIRTUAL_ENV = "FEDQUEST_VIRTUAL_SHARDS"


def page_of(o: ObjectRecord) -> str:
    page = o.meta.get("page") if isinstance(o.meta, dict) else None
    if page:
        return str(page)
    if o.modality == "table" and " | " in o.text:
        return o.text.split(" | ", 1)[0]
    return o.title


@dataclass(frozen=True)
class Virtual:
    parent: tuple[str, str, str]
    j: int
    m: int
    seed: int
    scheme: str = "hash"
    beta: float = 0.5
    clusters: int = 8

    @property
    def spec(self) -> tuple[Any, ...]:
        return (self.scheme, self.seed, self.m, self.beta, self.clusters)


def virtual(fed: Any, site: Any, collection: Any) -> Virtual | None:
    if site.data_from or getattr(fed, "data_from", None):
        return None
    shard, parent = collection.shard_of, collection.parent
    if shard is None or parent is None or getattr(collection, "fraction", None) is not None \
            or getattr(collection, "share", None) is not None:
        return None
    if os.environ.get(VIRTUAL_ENV) != "1" and (fed.collection_data_dir(site.name, collection.name) /
                                               "objects.jsonl").exists():
        return None
    j, m = shard
    scheme = "dirichlet" if fed.partition.method == "dirichlet" else "hash"
    return Virtual(parent, j, m, int(fed.partition.shard_hash_seed), scheme, float(fed.partition.beta),
                   int(fed.partition.clusters))


_PAGES: dict[tuple[str, int, int], np.ndarray] = {}
_PAGES_LOCK = threading.Lock()


@functools.lru_cache(maxsize=1 << 20)
def _shard(page: str, m: int, seed: int) -> int:
    return shard_by_title(page, m, seed)


def shards_of(path: Path | str, objects: Sequence[ObjectRecord], m: int, seed: int) -> np.ndarray:
    key = (str(path), m, seed)
    with _PAGES_LOCK:
        hit = _PAGES.get(key)
    if hit is None:
        hit = np.fromiter((_shard(page_key(page_of(o)), m, seed) for o in objects), dtype=np.int16,
                          count=len(objects))
        with _PAGES_LOCK:
            _PAGES[key] = hit
    return hit


_DIRICHLET: dict[tuple[Any, ...], dict[str, int]] = {}
_DIRICHLET_LOCK = threading.Lock()


def _page_text(objs: Iterable[ObjectRecord]) -> str:
    return " ".join(f"{o.title} {o.text[:400]}" for o in objs)


def dirichlet_pages(site_objects: dict[str, Sequence[ObjectRecord]], m: int, seed: int, beta: float,
                    clusters: int) -> dict[str, int]:
    key = (tuple(sorted(site_objects)), m, seed, round(beta, 6), clusters)
    with _DIRICHLET_LOCK:
        hit = _DIRICHLET.get(key)
    if hit is not None:
        return hit
    from fedquest.data.partition import dirichlet_parts

    def rank(path: str) -> tuple[int, str]:
        objs = site_objects[path]
        order = {"text": 0, "table": 1, "image": 2}
        return (order.get(objs[0].modality, 3) if objs else 4, path)

    pages: dict[str, list[ObjectRecord]] = {}
    for path in sorted(site_objects, key=rank):
        for o in site_objects[path]:
            group = pages.setdefault(page_key(page_of(o)), [])
            if len(group) < 3:
                group.append(o)
    names = sorted(pages)
    out = dirichlet_parts(names, [_page_text(pages[n]) for n in names], m, beta, clusters, seed, independent=True)
    with _DIRICHLET_LOCK:
        _DIRICHLET.setdefault(key, out)
        return _DIRICHLET[key]


def shard_column(path: Path | str, objects: Sequence[ObjectRecord], v: Virtual | tuple[Any, ...],
                 site_objects: dict[str, Sequence[ObjectRecord]] | None = None) -> np.ndarray:
    scheme, seed, m, beta, clusters = v.spec if isinstance(v, Virtual) else v
    if scheme == "hash":
        return shards_of(path, objects, m, seed)
    if site_objects is None:
        site_objects = {str(path): objects}
    pages = dirichlet_pages(site_objects, m, seed, beta, clusters)
    return np.fromiter((pages[page_key(page_of(o))] for o in objects), dtype=np.int16, count=len(objects))


def shard_rows(path: Path | str, objects: Sequence[ObjectRecord], v: Virtual,
               site_objects: dict[str, Sequence[ObjectRecord]] | None = None) -> np.ndarray:
    return np.flatnonzero(shard_column(path, objects, v, site_objects) == v.j).astype(np.int64)


class ShardView:
    def __init__(self, parent: Any, rows: dict[str, np.ndarray]) -> None:
        self.parent = parent
        self.rows = rows
        self.namespace = parent.namespace
        self.cfg = parent.cfg

    def _up(self, collection: str, positions: Iterable[int] | None) -> list[int] | None:
        if positions is None:
            return None
        rows = self.rows[collection]
        idx = np.asarray(list(positions), dtype=np.int64)
        idx = idx[(idx >= 0) & (idx < len(rows))]
        return rows[idx].tolist()

    def _down(self, collection: str, parent_rows: Sequence[int]) -> list[int | None]:
        rows = self.rows[collection]
        if not len(parent_rows):
            return []
        p = np.asarray(parent_rows, dtype=np.int64)
        at = np.searchsorted(rows, p)
        ok = (at < len(rows)) & (rows[np.minimum(at, len(rows) - 1)] == p)
        return [int(a) if k else None for a, k in zip(at.tolist(), ok.tolist(), strict=True)]

    @property
    def array(self) -> bool:
        return bool(getattr(self.parent, "array", False))

    def search_scored(self, collection: str, vector: np.ndarray, k: int, offset: int = 0,
                      among: list[int] | None = None, exclude: list[int] | None = None,
                      exact: bool = False) -> list[tuple[int, float]]:
        if collection not in self.rows:
            return self.parent.search_scored(collection, vector, k, offset, among, exclude, exact)
        p_among = self._up(collection, among)
        p_exclude = self._up(collection, exclude) or None
        if p_among is None:
            p_among = self.rows[collection].tolist()
        hits = self.parent.search_scored(collection, vector, k, offset, p_among, p_exclude, exact)
        local = self._down(collection, [h for h, _ in hits])
        return [(pos, score) for pos, (_, score) in zip(local, hits, strict=True) if pos is not None]

    def search(self, collection: str, vector: np.ndarray, k: int, offset: int = 0, among: list[int] | None = None,
               exclude: list[int] | None = None, exact: bool = False) -> list[int]:
        return [pos for pos, _ in self.search_scored(collection, vector, k, offset, among, exclude, exact)]

    def count(self, collection: str) -> int:
        if collection not in self.rows:
            return self.parent.count(collection)
        if not self.parent.count(collection):
            return 0
        return int(len(self.rows[collection]))

    def has(self, collection: str) -> bool:
        return self.parent.has(collection)

    def name(self, collection: str) -> str:
        return self.parent.name(collection)

    def close(self) -> None:
        self.parent.close()
