from __future__ import annotations

import contextlib
import json
import logging
import os
import threading
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from fedquest.config import QdrantCfg

log = logging.getLogger(__name__)

TRANSIENT = frozenset({408, 429, 500, 502, 503, 504})
READY_TIMEOUT_S = 7200.0
TRIES = 4
MAX_WAIT = 10.0


def transient(e: BaseException) -> bool:
    import httpx
    from qdrant_client.common.client_exceptions import ResourceExhaustedResponse
    from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

    if isinstance(e, UnexpectedResponse):
        return e.status_code is None or e.status_code in TRANSIENT
    if isinstance(e, ResourceExhaustedResponse):
        return True
    if isinstance(e, ResponseHandlingException):
        source = getattr(e, "source", None)
        return isinstance(source, httpx.TransportError) and not isinstance(source, httpx.TimeoutException)
    return False


def timed_out(e: BaseException) -> bool:
    import httpx
    from qdrant_client.http.exceptions import ResponseHandlingException

    source = getattr(e, "source", None) if isinstance(e, ResponseHandlingException) else e
    return isinstance(source, httpx.TimeoutException) or "timed out" in str(e).lower()

_EMBEDDED: dict[str, tuple[object, threading.Lock, int]] = {}
_EMBEDDED_LOCK = threading.Lock()
_ARRAYS: dict[tuple[str, int, int], np.ndarray] = {}
_ARRAYS_LOCK = threading.Lock()
SCORE_DECIMALS = 6


def load_rows(path: Path) -> tuple[np.ndarray, np.ndarray]:
    st = path.stat()
    key = (str(path.resolve()), st.st_mtime_ns, st.st_size)
    with _ARRAYS_LOCK:
        hit = _ARRAYS.get(key)
        if hit is not None:
            return hit
        rows = np.load(path, mmap_mode="r")
        if rows.ndim != 2:
            raise ValueError(f"{path}: {rows.ndim}-d vectors, not one row per object")
        if rows.dtype != np.float32:
            rows = np.asarray(rows, dtype=np.float32)
        norms = np.empty(len(rows), dtype=np.float32)
        for start in range(0, len(rows), 65536):
            norms[start:start + 65536] = np.linalg.norm(rows[start:start + 65536], axis=1)
        out = (rows, (1.0 / np.where(norms > 0, norms, 1.0)).astype(np.float32))
        _ARRAYS[key] = out
        return out


def ranked(rows: np.ndarray, vector: np.ndarray, k: int, offset: int = 0, among: list[int] | None = None,
           exclude: Iterable[int] | None = None, inv: np.ndarray | None = None) -> list[int]:
    return [i for i, _ in ranked_scored(rows, vector, k, offset, among, exclude, inv)]


def ranked_scored(rows: np.ndarray, vector: np.ndarray, k: int, offset: int = 0, among: list[int] | None = None,
                  exclude: Iterable[int] | None = None, inv: np.ndarray | None = None) -> list[tuple[int, float]]:
    want = offset + k
    if want <= 0 or k <= 0 or len(rows) == 0:
        return []
    q = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(q))
    q = q / norm if norm > 0 else q
    banned = set(exclude or ())
    if among is not None:
        ids = np.array(sorted({int(i) for i in among if 0 <= int(i) < len(rows)} - banned), dtype=np.int64)
        if ids.size == 0:
            return []
        scores = np.asarray(rows[ids], dtype=np.float32) @ q
        if inv is not None:
            scores = scores * inv[ids]
    else:
        ids = np.arange(len(rows), dtype=np.int64)
        scores = np.asarray(rows @ q, dtype=np.float32)
        if inv is not None:
            scores = scores * inv
        if banned:
            out = np.array(sorted(b for b in banned if 0 <= b < len(rows)), dtype=np.int64)
            if out.size:
                scores = scores.copy()
                scores[out] = -np.inf
    key = np.round(scores.astype(np.float64), SCORE_DECIMALS)
    if want < key.size:
        part = np.argpartition(-key, want - 1)[:want]
        cut = key[part].min()
        part = np.flatnonzero(key >= cut) if np.isfinite(cut) else part
    else:
        part = np.arange(key.size)
    keep = part[np.isfinite(key[part])]
    order = keep[np.lexsort((ids[keep], -key[keep]))][offset:want]
    return [(int(i), float(scores[j])) for i, j in zip(ids[order].tolist(), order.tolist(), strict=True)]


ANN_ENV = "FEDQUEST_ANN"
FULL_SCAN_KB = 10_000
ANN_EF = 128
ACORN_SELECTIVITY = 0.4
SEGMENTS = 2


def ann_live() -> bool:
    return os.environ.get(ANN_ENV, "").lower() == "hnsw"


def live_index(cfg: QdrantCfg) -> QdrantCfg:
    if not ann_live() or cfg.mode != "server" or cfg.approximate:
        return cfg
    from fedquest.config import IndexCfg

    return cfg.model_copy(update={"index": IndexCfg(kind="hnsw", m=16, ef_construct=100, ef=ANN_EF), "exact": False})


class SiteIndex:
    def __init__(self, root: Path, cfg: QdrantCfg, namespace: str, skip: dict[str, set[int]] | None = None,
                 grpc: bool = False) -> None:
        self.cfg = live_index(cfg.resolved())
        self.namespace = namespace
        self._lock = threading.Lock()
        self._path: str | None = None
        self._root = root
        self._skip = {c: set(s) for c, s in (skip or {}).items()}
        if self.cfg.mode == "array":
            if self.cfg.approximate:
                raise ValueError(f"{namespace}: an approximate index ({self.cfg.index.kind if self.cfg.index else '?'})"
                                 " is a Qdrant server's: run it with FEDQUEST_QDRANT_MODE=server (python -m "
                                 "fedquest.remote)")
            self.client = None
            return
        from qdrant_client import QdrantClient

        if self.cfg.mode == "local":
            root.mkdir(parents=True, exist_ok=True)
            self._path = str((root / "qdrant").resolve())
            with _EMBEDDED_LOCK:
                client, lock, users = _EMBEDDED.get(self._path) or (QdrantClient(path=self._path), threading.Lock(), 0)
                _EMBEDDED[self._path] = (client, lock, users + 1)
            self.client, self._lock = client, lock  # type: ignore[assignment]
        elif not self.cfg.url:
            raise RuntimeError("no Qdrant for this site: start it with `fedquest kg up` or "
                               "`bash scripts/qdrant.sh start` (Aurora, Midway3); both record FEDQUEST_QDRANT_URL")
        elif grpc:
            from urllib.parse import urlsplit

            port = urlsplit(self.cfg.url).port or 6333
            self.client = QdrantClient(url=self.cfg.url, grpc_port=port + 1, prefer_grpc=True, timeout=120,
                                       grpc_options={"grpc.enable_http_proxy": 0})
        else:
            self.client = QdrantClient(url=self.cfg.url, timeout=120)

    def name(self, collection: str) -> str:
        return f"{self.namespace}__{collection}"

    def _call(self, what: str, fn):
        for attempt in range(TRIES):
            try:
                with self._lock if self.cfg.mode == "local" else contextlib.nullcontext():
                    return fn()
            except Exception as e:
                if attempt == TRIES - 1 or not transient(e):
                    raise
                wait = max(0.5 * 2 ** attempt, min(float(getattr(e, "retry_after_s", 0) or 0), MAX_WAIT))
                log.warning("Qdrant %s failed (%s: %s); try %d of %d in %.1f s", what, type(e).__name__,
                            " ".join(str(e).split())[:300], attempt + 2, TRIES, wait)
                time.sleep(wait)
        raise AssertionError("unreachable")

    @property
    def array(self) -> bool:
        return self.cfg.mode == "array"

    def _rows_path(self, collection: str) -> Path:
        return self._root / collection / "emb.npy"

    def _left_out(self, collection: str) -> set[int]:
        if collection in self._skip:
            return self._skip[collection]
        meta = self._root / collection / "emb.json"
        missing = (json.loads(meta.read_text()).get("missing") or []) if meta.exists() else []
        self._skip[collection] = {int(i) for i in missing}
        return self._skip[collection]

    def has(self, collection: str) -> bool:
        if self.array:
            return self._rows_path(collection).exists()
        return self._call(f"lookup of {self.name(collection)}",
                          lambda: self.client.collection_exists(self.name(collection)))

    def count(self, collection: str) -> int:
        if self.array:
            path = self._rows_path(collection)
            if not path.exists():
                return 0
            rows = np.load(path, mmap_mode="r").shape[0]
            return rows - len({i for i in self._left_out(collection) if 0 <= i < rows})

        def count() -> int:
            if not self.client.collection_exists(self.name(collection)):
                return 0
            return self.client.count(self.name(collection), exact=True).count

        return self._call(f"count of {self.name(collection)}", count)

    def build(self, collection: str, vectors: np.ndarray, batch: int = 1024, skip: set[int] | None = None,
              wait: bool = True) -> int:
        if self.array:
            self._skip[collection] = set(skip or ())
            return len(vectors) - len({i for i in (skip or ()) if 0 <= i < len(vectors)})
        from qdrant_client import models

        vectors = np.asarray(vectors, dtype=np.float32)
        ids = [i for i in range(len(vectors)) if i not in skip] if skip else list(range(len(vectors)))
        name = self.name(collection)
        approximate = self._approximate()
        quantized = approximate and self.cfg.index is not None and self.cfg.index.kind != "hnsw"
        config = {**self._index_config(), "optimizers_config": models.OptimizersConfigDiff(
            indexing_threshold=0, default_segment_number=SEGMENTS if ann_live() else None)}
        params = models.VectorParams(size=int(vectors.shape[1]), distance=models.Distance.COSINE, on_disk=quantized)
        with self._lock:
            if self.client.collection_exists(name):
                self.client.delete_collection(name)
            self.client.create_collection(name, vectors_config=params, **config)
        for start in range(0, len(ids), batch):
            chunk = ids[start : start + batch]
            self._upsert(name, chunk, vectors[chunk])
        if self.builds_graph:
            self._call(f"indexing of {name}", lambda: self.client.update_collection(
                name, optimizers_config=models.OptimizersConfigDiff(indexing_threshold=1)))
            if wait:
                self.wait_ready(collection, len(ids))
        return len(ids)

    def _upsert(self, name: str, chunk: list[int], vecs: np.ndarray) -> None:
        from qdrant_client import models

        for attempt in range(TRIES):
            try:
                with self._lock:
                    self.client.upsert(name, points=models.Batch(ids=chunk, vectors=vecs.tolist()), wait=True)
                return
            except Exception as e:
                if attempt == TRIES - 1 or not (transient(e) or timed_out(e)):
                    raise
                wait = max(2.0 * 2 ** attempt, min(float(getattr(e, "retry_after_s", 0) or 0), MAX_WAIT))
                log.warning("Qdrant upsert into %s failed (%s: %s); try %d of %d in %.1f s", name, type(e).__name__,
                            " ".join(str(e).split())[:300], attempt + 2, TRIES, wait)
                time.sleep(wait)

    def _approximate(self) -> bool:
        spec = getattr(self.cfg, "index", None)
        return spec is not None and spec.kind != "flat"

    @property
    def builds_graph(self) -> bool:
        return self._approximate() and self.cfg.mode == "server"

    def _index_config(self) -> dict[str, object]:
        from qdrant_client import models

        spec = getattr(self.cfg, "index", None)
        if spec is None or spec.kind == "flat":
            return {}
        out: dict[str, object] = {
            "hnsw_config": models.HnswConfigDiff(m=spec.m, ef_construct=spec.ef_construct,
                                                 full_scan_threshold=FULL_SCAN_KB if ann_live() else 10),
            "optimizers_config": models.OptimizersConfigDiff(indexing_threshold=1),
        }
        if spec.kind == "sq8":
            out["quantization_config"] = models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(type=models.ScalarType.INT8, quantile=0.99, always_ram=True))
        elif spec.kind == "pq":
            out["quantization_config"] = models.ProductQuantization(
                product=models.ProductQuantizationConfig(
                    compression=getattr(models.CompressionRatio, spec.pq_compression.upper()), always_ram=True))
        elif spec.kind == "bq":
            out["quantization_config"] = models.BinaryQuantization(
                binary=models.BinaryQuantizationConfig(always_ram=True))
        return out

    def wait_ready(self, collection: str, points: int, timeout_s: float = READY_TIMEOUT_S) -> None:
        if self.array:
            return
        from qdrant_client import models

        name, deadline = self.name(collection), time.monotonic() + timeout_s
        while True:
            info = self._call(f"status of {name}", lambda: self.client.get_collection(name))
            done = info.status == models.CollectionStatus.GREEN and (info.indexed_vectors_count or 0) >= points
            if done:
                return
            if time.monotonic() > deadline:
                raise TimeoutError(f"{name}: {info.indexed_vectors_count} of {points} points indexed after "
                                   f"{timeout_s:.0f} s (status {info.status})")
            time.sleep(1.0)

    def search(self, collection: str, vector: np.ndarray, k: int, offset: int = 0,
               among: list[int] | None = None, exclude: list[int] | None = None, exact: bool = False) -> list[int]:
        return [pos for pos, _ in self.search_scored(collection, vector, k, offset, among, exclude, exact)]

    def search_scored(self, collection: str, vector: np.ndarray, k: int, offset: int = 0,
                      among: list[int] | None = None, exclude: list[int] | None = None,
                      exact: bool = False) -> list[tuple[int, float]]:
        if k <= 0:
            return []
        if self.array:
            path = self._rows_path(collection)
            if not path.exists():
                raise FileNotFoundError(f"{path}: no vectors for {self.name(collection)} (fedquest data embed or "
                                        "derive)")
            out = set(self._left_out(collection)) | set(exclude or ())
            rows, inv = load_rows(path)
            return ranked_scored(rows, vector, k, offset, among, out, inv)
        from qdrant_client import models

        if self.cfg.mode != "server":
            params = None
        elif exact and not ann_live():
            params = models.SearchParams(exact=True)
        else:
            params = self._search_params(k + offset)
            if among is not None and params is not None and self._approximate():
                params = params.model_copy(update={"acorn": models.AcornSearchParams(
                    enable=True, max_selectivity=ACORN_SELECTIVITY)})
        must: list[Any] = [models.HasIdCondition(has_id=list(among))] if among is not None else []
        must_not = [models.HasIdCondition(has_id=list(exclude))] if exclude else None
        flt = models.Filter(must=must or None, must_not=must_not) if must or must_not else None
        resp = self._call(f"search of {self.name(collection)}", lambda: self.client.query_points(
            self.name(collection),
            query=np.asarray(vector, dtype=np.float32).tolist(),
            limit=k,
            offset=offset,
            query_filter=flt,
            search_params=params,
            with_payload=False,
        ))
        return [(int(p.id), float(getattr(p, "score", 0.0) or 0.0)) for p in resp.points]

    def _search_params(self, depth: int) -> object | None:
        from qdrant_client import models

        spec = getattr(self.cfg, "index", None)
        if spec is None or spec.kind == "flat":
            return models.SearchParams(exact=True) if self.cfg.exact else None
        quant = None if spec.kind == "hnsw" else models.QuantizationSearchParams(
            ignore=False, rescore=spec.rescore, oversampling=spec.oversampling)
        return models.SearchParams(hnsw_ef=max(spec.ef, depth), exact=False, quantization=quant)

    def close(self) -> None:
        if self.array:
            return
        if self._path is not None:
            with _EMBEDDED_LOCK:
                if self._path not in _EMBEDDED:
                    return
                client, lock, users = _EMBEDDED[self._path]
                if users > 1:
                    _EMBEDDED[self._path] = (client, lock, users - 1)
                    return
                del _EMBEDDED[self._path]
        with self._lock:
            self.client.close()
