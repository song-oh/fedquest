from __future__ import annotations

import asyncio
import bisect
import contextlib
import contextvars
import dataclasses
import functools
import itertools
import json
import logging
import os
import subprocess
import threading
import time
from collections import Counter, OrderedDict
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from fedquest.config import SITE_ROLES, CollectionCfg, FederationCfg, SiteCfg, load_federation, load_models
from fedquest.data.embed import index_skip
from fedquest.federation.base import WIRE_VERSION, mention_keys, share_slack, vocab_filters
from fedquest.kg.normalize import norm_entity, release_signal
from fedquest.kg.store import SiteFactStore
from fedquest.llm import prompts
from fedquest.llm.client import LLMClient, LLMReply
from fedquest.llm.parsing import MAX_PART_CHARS
from fedquest.paths import raw_dir
from fedquest.site.extractor import Extractor, content_bytes, load_image_data_uri
from fedquest.site.hippo import HIPPO_DIR, HippoIndex, objects_stamp, open_index
from fedquest.types import (
    Candidate,
    CollectionInfo,
    ExtractionResult,
    HippoRequest,
    ObjectRecord,
    ProbeCounts,
    ProbeRequest,
    RawObject,
    ReadResult,
    SiteInfo,
    Usage,
)
from fedquest.utils import approx_tokens, read_jsonl, stable_hash, truncate
from fedquest.vdb import anchors, shards
from fedquest.vdb import views as edge_views
from fedquest.vdb.encoders import Encoder, load_encoder
from fedquest.vdb.qdrant_index import SiteIndex

log = logging.getLogger(__name__)

RAW_BUDGET_BYTES = 280_000
LEGACY_MODEL = "google/gemma-4-31B-it"
SNIPPET_CHARS = 240
BUSY_FIELDS = ("probe_messages", "probe_requests", "embed_seconds", "search_seconds", "read_calls", "read_seconds",
               "extract_calls", "extract_seconds", "busy_seconds", "view_entries", "view_objects", "view_seconds")
PREFILTER_MAX = 4096
FILTER_TOP = 400
JOINED_CACHE = 2048
POSTINGS_FILE = "postings.npz"
POSTINGS_VERSION = 1
_slack = functools.lru_cache(maxsize=4096)(share_slack)


@functools.lru_cache(maxsize=1)
def code_commit() -> str:
    try:
        out = subprocess.run(["git", "-C", str(Path(__file__).resolve().parents[2]), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


_OBJECTS: dict[tuple[str, int, int], list[ObjectRecord]] = {}
_OBJECTS_LOCK = threading.Lock()
_PARENT_POSTINGS: dict[str, Postings] = {}
_PARENT_POSTINGS_LOCK = threading.Lock()
_OVERLAYS: dict[tuple[str, str], dict[str, dict[str, ObjectRecord]]] = {}
_OVERLAYS_LOCK = threading.Lock()
_EPOCH: contextvars.ContextVar[str | None] = contextvars.ContextVar("fedquest_epoch", default=None)
ENTITY_LINKS = True
HARVEST_VERSION = f"{prompts.READ_VERSION}+harvest"


@contextlib.contextmanager
def epoch_scope(epoch: str | None) -> Iterator[None]:
    token = _EPOCH.set(epoch)
    try:
        yield
    finally:
        _EPOCH.reset(token)


def _objects(path: Path) -> list[ObjectRecord]:
    st = path.stat()
    key = (str(path.resolve()), st.st_mtime_ns, st.st_size)
    with _OBJECTS_LOCK:
        if key not in _OBJECTS:
            _OBJECTS[key] = [ObjectRecord.from_dict(d) for d in read_jsonl(path)]
        return _OBJECTS[key]


@dataclasses.dataclass(frozen=True)
class VirtualSite:
    parent: FederationCfg
    site: str
    objects: dict[str, list[ObjectRecord]]
    rows: dict[str, np.ndarray]
    pskip: dict[str, set[int]]
    skip: dict[str, set[int]]

    @property
    def sources(self) -> dict[str, tuple[Path, np.ndarray]]:
        return {c: (self.parent.collection_data_dir(self.site, c) / "objects.jsonl", r) for c, r in self.rows.items()}


def virtual_site(fed: FederationCfg, scfg: SiteCfg, virt: dict[str, shards.Virtual] | None = None) -> VirtualSite:
    if virt is None:
        virt = {c.name: v for c in scfg.collections if (v := shards.virtual(fed, scfg, c)) is not None}
    if not virt:
        raise ValueError(f"{fed.name}/{scfg.name}: no collection is a virtual shard")
    if len(virt) != len(scfg.collections):
        raise ValueError(f"{fed.name}/{scfg.name}: some collections are virtual shards and some are not")
    parents = {v.parent[:2] for v in virt.values()}
    if len(parents) != 1 or any(v.parent[2] != name for name, v in virt.items()):
        raise ValueError(f"{fed.name}/{scfg.name}: a virtual site reads one parent site's collections of the same "
                         f"names, not {sorted(v.parent for v in virt.values())}")
    pname, psite = next(iter(parents))
    pfed = load_federation(pname)
    objects: dict[str, list[ObjectRecord]] = {}
    skip: dict[str, set[int]] = {}
    pskip: dict[str, set[int]] = {}
    rows: dict[str, np.ndarray] = {}
    parent_objects = {str(pfed.collection_data_dir(psite, c.name) / "objects.jsonl"):
                      _objects(pfed.collection_data_dir(psite, c.name) / "objects.jsonl")
                      for c in pfed.site(psite).collections}
    for c in scfg.collections:
        v = virt[c.name]
        pdir = pfed.collection_data_dir(psite, c.name)
        path = pdir / "objects.jsonl"
        pobjs = _objects(path)
        r = shards.shard_rows(path, pobjs, v, parent_objects)
        objects[c.name] = [pobjs[i] for i in r.tolist()]
        pskip[c.name] = index_skip(pdir, pobjs)
        local = np.searchsorted(r, np.asarray(sorted(pskip[c.name]), dtype=np.int64))
        skip[c.name] = {int(a) for a, prow in zip(local.tolist(), sorted(pskip[c.name]), strict=True)
                        if a < len(r) and int(r[a]) == prow}
        rows[c.name] = r
    return VirtualSite(pfed, psite, objects, rows, pskip, skip)


def titled_text(o: ObjectRecord) -> str:
    if o.modality == "table" or not o.title or o.text.lstrip().lower().startswith(o.title.lower()):
        return o.text
    return f"{o.title}: {o.text}"


class SiteService:
    def __init__(
        self,
        fed: FederationCfg,
        site: SiteCfg,
        *,
        objects: dict[str, list[ObjectRecord]],
        index: SiteIndex,
        store: SiteFactStore,
        extractor: Extractor,
        encoder_device: str = "cpu",
        skip: dict[str, set[int]] | None = None,
    ) -> None:
        self.fed = fed
        self.site = site
        self.objects = objects
        self.where = {o.oid: (c, i) for c, objs in objects.items() for i, o in enumerate(objs)}
        self.index = index
        self.store = store
        self.extractor = extractor
        self.encoder_device = encoder_device
        self.skip = skip or {}
        self._sem = asyncio.Semaphore(site.max_concurrency)
        self._keys: OrderedDict[tuple[str, str], np.ndarray] = OrderedDict()
        self._keys_lock = threading.Lock()
        self._titles: dict[str, TitleIndex] = {}
        self._titles_lock = threading.Lock()
        self._blooms: dict[str, dict[str, Any]] | None = None
        self._postings: dict[str, Postings] = {}
        self._postings_lock = threading.Lock()
        self._joined: OrderedDict[tuple[Any, ...], np.ndarray] = OrderedDict()
        self._joined_lock = threading.Lock()
        self._vocab: dict[str, dict[str, Any]] | None = None
        self._hippo: dict[str, HippoIndex | None] = {}
        self._hippo_lock = threading.Lock()
        self._hippo_encoders: dict[str, Encoder] = {}
        self._started = time.time()
        self._busy = dict.fromkeys(BUSY_FIELDS, 0.0)
        self._busy_lock = threading.Lock()
        self.virtual: dict[str, tuple[Path, np.ndarray]] = {}
        self._zstats: OrderedDict[tuple[str, str], tuple[float, float] | None] = OrderedDict()
        self._zstats_lock = threading.Lock()
        self._links: dict[str, dict[str, np.ndarray]] = {}
        self._links_lock = threading.Lock()

    @classmethod
    def open(cls, federation: str | Path, site: str, *, llm: LLMClient | None = None,
             encoder_device: str = "cpu") -> SiteService:
        fed = load_federation(federation)
        scfg = fed.site(site)
        store = SiteFactStore(fed.store_dir(site) / "facts.sqlite")
        llm = llm or LLMClient.from_env(profile=load_models().site_profile(scfg))
        extractor = Extractor(llm, raw_root=raw_dir(fed.dataset), version=fed.extractor_version)
        virt = {c.name: v for c in scfg.collections if (v := shards.virtual(fed, scfg, c)) is not None}
        if virt:
            return cls._open_virtual(fed, scfg, virt, store, extractor, encoder_device)
        objects = {c.name: _objects(fed.collection_data_dir(site, c.name) / "objects.jsonl") for c in scfg.collections}
        skip = {c.name: index_skip(fed.collection_data_dir(site, c.name), objects[c.name]) for c in scfg.collections}
        index = SiteIndex(fed.site_data_dir(site), fed.qdrant, namespace=fed.index_namespace(site),
                          skip=skip)
        out = cls(fed, scfg, objects=objects, index=index, store=store, extractor=extractor,
                  encoder_device=encoder_device, skip=skip)
        out.vector_source = {c.name: (fed.collection_data_dir(site, c.name) / "emb.npy", None)
                             for c in scfg.collections}
        return out

    @classmethod
    def _open_virtual(cls, fed: FederationCfg, scfg: SiteCfg, virt: dict[str, shards.Virtual], store: SiteFactStore,
                      extractor: Extractor, encoder_device: str) -> SiteService:
        from fedquest.deploy import node_qdrant

        vs = virtual_site(fed, scfg, virt)
        if node_qdrant.enabled():
            index: SiteIndex | shards.ShardView = SiteIndex(fed.site_data_dir(scfg.name), fed.qdrant,
                                                           namespace=fed.index_namespace(scfg.name), skip=vs.skip)
        else:
            parent = SiteIndex(vs.parent.site_data_dir(vs.site), vs.parent.qdrant,
                               namespace=vs.parent.index_namespace(vs.site), skip=vs.pskip)
            index = shards.ShardView(parent, vs.rows)
        out = cls(fed, scfg, objects=vs.objects, index=index, store=store, extractor=extractor,  # type: ignore[arg-type]
                  encoder_device=encoder_device, skip=vs.skip)
        out.virtual = vs.sources
        out.vector_source = {c.name: (vs.parent.collection_data_dir(vs.site, c.name) / "emb.npy", vs.rows[c.name])
                             for c in scfg.collections}
        return out

    def _obj(self, oid: str) -> ObjectRecord | None:
        hit = self.where.get(oid)
        return None if hit is None else self._at(hit[0], hit[1])

    def _at(self, collection: str, pos: int) -> ObjectRecord:
        o = self.objects[collection][pos]
        epoch = _EPOCH.get()
        if epoch is None:
            return o
        return _OVERLAYS.get((self.fed.name, self.site.name), {}).get(epoch, {}).get(o.oid, o)

    def _versions_of(self, cands: list[Candidate], request: ProbeRequest) -> list[Candidate]:
        if request.versions:
            for c in cands:
                o = self._obj(c.oid)
                if o is not None:
                    c.ver = o.version
        return cands

    def _viewed(self, cands: list[Candidate], request: ProbeRequest) -> list[Candidate]:
        if self.site.policy != "snippets" or not (request.views or request.signal):
            return cands
        room = request.room or SNIPPET_CHARS
        todo: dict[str, list[tuple[Candidate, ObjectRecord]]] = {}
        for c in cands:
            if c.modality != "text" or c.view:
                continue
            o = self._obj(c.oid)
            if o is None:
                continue
            if request.signal is not None and len(o.text) > SNIPPET_CHARS and (request.views or room != SNIPPET_CHARS):
                lead = f"{o.title}: {truncate(o.text, SNIPPET_CHARS)}"
                c.lead_signal = any(release_signal(lead, v) for v in request.signal)
            if request.views and request.key and len(o.text) > room:
                todo.setdefault(c.collection, []).append((c, o))
        values = list(dict.fromkeys([*request.mentions, *request.titles]))
        for collection, pairs in todo.items():
            encoder = self.site.collection(collection).encoder
            key = self._embed_key(collection, request.view_key or request.key)
            got, built, took = edge_views.INDEX.get(edge_views.index_name(encoder), self._encoder(collection),
                                                    [(o.oid, o.version, o.text) for _, o in pairs])
            for (c, o), (sents, vecs) in zip(pairs, got, strict=True):
                c.view = True
                c.summary = f"{o.title}: {edge_views.view(sents, vecs, key, values, room)}"
            self._note_busy(view_entries=len(pairs), view_objects=built, view_seconds=took)
        return cands

    def _reference(self, collection: str) -> tuple[np.ndarray | None, bool]:
        c = self.site.collection(collection)
        return anchors.reference(c.encoder, self._encoder(collection), c.modality, self.extractor.raw_root, c.vectors)

    def _zv(self, collection: str, key: str) -> dict[str, Any]:
        mat, images = self._reference(collection)
        if mat is None:
            return {}
        q = self._embed_key(collection, key)
        if images:
            texts = anchors.anchor_matrix(self.site.collection(collection).encoder, self._encoder(collection))
            return anchors.image_variant_backgrounds(mat, texts, q)
        return anchors.variant_backgrounds(mat, q)

    def _scored(self, cand: Candidate, score: float, stats: tuple[float, float] | None,
                variants: dict[str, Any] | None) -> None:
        if stats is not None:
            cand.z = anchors.zscore(score, stats)
        if variants:
            cand.zv = {k: v.score(score) if isinstance(v, anchors.TailBackground) else anchors.zscore(score, v)
                       for k, v in variants.items()}

    def _z(self, collection: str, key: str) -> tuple[float, float] | None:
        c = self.site.collection(collection)
        cell = (f"{c.encoder}|{c.modality}|{c.vectors if c.modality == 'image' else ''}", key)
        with self._zstats_lock:
            if cell in self._zstats:
                return self._zstats[cell]
        mat, _ = self._reference(collection)
        stats = None if mat is None else anchors.background(mat, self._embed_key(collection, key))
        with self._zstats_lock:
            self._zstats[cell] = stats
            if len(self._zstats) > 4096:
                self._zstats.popitem(last=False)
        return stats

    def _note_busy(self, **inc: float) -> None:
        with self._busy_lock:
            for k, v in inc.items():
                self._busy[k] += v

    def _encoder(self, collection: str) -> Encoder:
        return load_encoder(self.site.collection(collection).encoder, self.encoder_device)

    def _embed_key(self, collection: str, key: str) -> np.ndarray:
        name = self.site.collection(collection).encoder
        with self._keys_lock:
            cached = self._keys.get((name, key))
        if cached is None:
            t0 = time.perf_counter()
            cached = self._encoder(collection).encode_queries([key])[0]
            self._note_busy(embed_seconds=time.perf_counter() - t0)
            with self._keys_lock:
                self._keys[(name, key)] = cached
                if len(self._keys) > 4096:
                    self._keys.popitem(last=False)
        return cached

    def _search(self, request: ProbeRequest) -> list[Candidate]:
        out = []
        for c in self.site.collections:
            if request.collections and c.name not in request.collections:
                continue
            vec = self._embed_key(c.name, request.key)
            skip = sorted(self.skip.get(c.name, ()))
            t0 = time.perf_counter()
            found = self.index.search_scored(c.name, vec, self._limit(request, c), request.offset, exclude=skip)
            self._note_busy(search_seconds=time.perf_counter() - t0)
            stats = self._z(c.name, request.key) if request.scores else None
            variants = self._zv(c.name, request.key) if request.zvariants else None
            for i, (pos, score) in enumerate(found):
                cand = candidate(self.site.name, self.site.policy, c.name, self._at(c.name, pos),
                                 request.offset + i + 1, mentions=request.mentions, room=request.room or 0)
                self._scored(cand, score, stats, variants)
                out.append(cand)
        return out

    def _limit(self, r: ProbeRequest, c: CollectionCfg) -> int:
        if r.k_group is None or not r.group_size or (r.encoder is not None and c.encoder != r.encoder):
            return r.k
        m = len(self.objects[c.name])
        return -(-r.k_group * m // r.group_size) + _slack(r.k_group, m / r.group_size)

    def _postings_of(self, collection: str) -> Postings:
        with self._postings_lock:
            if collection in self._postings:
                return self._postings[collection]
            if collection in self.virtual:
                path, rows = self.virtual[collection]
                post = _parent_postings(path).restricted(rows)
                self._postings[collection] = post
                return post
            objs = self.objects[collection]
            path, stamp, post = None, {}, None
            try:
                src = self.fed.collection_data_dir(self.site.name, collection) / "objects.jsonl"
                st = src.stat()
                path, stamp = src.with_name(POSTINGS_FILE), {"size": st.st_size, "mtime_ns": st.st_mtime_ns,
                                                             "objects": len(objs)}
                post = Postings.load(path, stamp)
            except (OSError, KeyError, AttributeError):
                pass
            if post is None:
                t0 = time.perf_counter()
                post = Postings.build([mention_text(o) for o in objs])
                log.info("%s/%s: postings of %d objects (%d words) in %.1f s", self.site.name, collection, post.n,
                         len(post.vocab), time.perf_counter() - t0)
                if path is not None:
                    try:
                        post.save(path, stamp)
                    except OSError as e:
                        log.warning("%s/%s: postings not cached at %s (%s)", self.site.name, collection, path, e)
            self._postings[collection] = post
            return post

    def _hippo_of(self, collection: str) -> HippoIndex | None:
        with self._hippo_lock:
            index = self._hippo.get(collection)
            if index is None:
                index = open_index(self.fed.collection_data_dir(self.site.name, collection), None,
                                   self.objects[collection])
                if index is not None:
                    self._hippo[collection] = index
            return index

    def _passing(self, collection: str, value: str) -> np.ndarray | None:
        key = (collection, value, prompts.mention_test())
        with self._joined_lock:
            hit = self._joined.get(key)
            if hit is not None:
                self._joined.move_to_end(key)
                return hit
        post = self._postings_of(collection)
        cand = post.candidates(value)
        cand = np.arange(post.n, dtype=np.int32) if cand is None else cand
        if cand.size > PREFILTER_MAX:
            return None
        objs = self.objects[collection]
        hit = np.array([i for i in cand.tolist() if prompts.mentions(mention_text(objs[i]), value)], dtype=np.int32)
        linked = self._linked(collection, value)
        if linked.size:
            hit = np.union1d(hit, linked).astype(np.int32)
        with self._joined_lock:
            self._joined[key] = hit
            if len(self._joined) > JOINED_CACHE:
                self._joined.popitem(last=False)
        return hit

    def _linked(self, collection: str, value: str) -> np.ndarray:
        if not ENTITY_LINKS or self.site.collection(collection).modality != "table":
            return _NONE
        with self._links_lock:
            index = self._links.get(collection)
            if index is None:
                where: dict[str, list[int]] = {}
                for i, o in enumerate(self.objects[collection]):
                    for e in (o.meta.get("entities") or []) if isinstance(o.meta, dict) else []:
                        where.setdefault(_title_key(str(e)), []).append(i)
                index = {k: np.asarray(sorted(set(v)), dtype=np.int32) for k, v in where.items()}
                self._links[collection] = index
        return index.get(_title_key(value), _NONE)

    def _semijoin(self, r: ProbeRequest) -> tuple[list[Candidate], list[ProbeCounts]]:
        if self.site.policy == "sealed":
            return self._search(r), []
        out: list[Candidate] = []
        counts: list[ProbeCounts] = []
        values = list(dict.fromkeys(r.mentions))
        for c in self.site.collections:
            if (r.collections and c.name not in r.collections) or c.modality == "image":
                continue
            k, objs, skip = self._limit(r, c), self.objects[c.name], self.skip.get(c.name, set())
            joined: np.ndarray | None = _NONE
            for v in values:
                passing = self._passing(c.name, v)
                if passing is None:
                    joined = None
                    break
                joined = np.union1d(joined, passing)
            if joined is not None and skip:
                joined = joined[~np.isin(joined, sorted(skip))]
            scored: dict[int, float] = {}
            if joined is not None and joined.size <= PREFILTER_MAX:
                found: list[int] = []
                if joined.size and k > 0:
                    vec = self._embed_key(c.name, r.key)
                    t0 = time.perf_counter()
                    hits_s = self.index.search_scored(c.name, vec, k, r.offset, among=joined.tolist(), exact=True)
                    self._note_busy(search_seconds=time.perf_counter() - t0)
                    found, scored = [pos for pos, _ in hits_s], dict(hits_s)
                page, joinable, exact = [(pos, r.offset + j + 1) for j, pos in enumerate(found)], int(joined.size), True
            else:
                vec = self._embed_key(c.name, r.key)
                t0 = time.perf_counter()
                top_s = self.index.search_scored(c.name, vec, FILTER_TOP, 0, exclude=sorted(skip), exact=True)
                self._note_busy(search_seconds=time.perf_counter() - t0)
                top, scored = [pos for pos, _ in top_s], dict(top_s)
                post = self._postings_of(c.name)
                maybe = functools.reduce(np.union1d, [np.arange(post.n) if s is None else s
                                                      for s in map(post.candidates, values)])
                links = set().union(*(self._linked(c.name, v).tolist() for v in values))
                maybe = np.union1d(maybe, np.asarray(sorted(links), dtype=np.int64))
                maybe_top = set(np.intersect1d(np.asarray(top, dtype=np.int64), maybe).tolist())
                hits = [(pos, rank) for rank, pos in enumerate(top, 1)
                        if pos in maybe_top and (pos in links or names_any(objs[pos], values))]
                page, joinable, exact = hits[r.offset : r.offset + max(k, 0)], len(hits), False
            stats = self._z(c.name, r.key) if r.scores else None
            variants = self._zv(c.name, r.key) if r.zvariants else None
            for j, (pos, rank) in enumerate(page):
                cand = candidate(self.site.name, self.site.policy, c.name, self._at(c.name, pos), rank,
                                 mentions=r.mentions, room=r.room or 0)
                cand.jrank = r.offset + j + 1
                if pos in scored:
                    self._scored(cand, scored[pos], stats, variants)
                out.append(cand)
            counts.append(ProbeCounts(c.name, joinable, len(page), exact))
        return out, counts

    def _title_index(self, collection: str) -> TitleIndex:
        with self._titles_lock:
            if collection not in self._titles:
                self._titles[collection] = TitleIndex([o.title for o in self.objects[collection]])
            return self._titles[collection]

    async def blooms(self, vocab: bool = False) -> dict[str, dict[str, Any]]:
        from fedquest.bloom import BloomFilter, title_entries

        if vocab:
            if self._vocab is None:
                self._vocab = {} if self.site.policy == "sealed" else await asyncio.to_thread(lambda: vocab_filters(
                    {c.name: self._postings_of(c.name).vocab for c in self.site.collections if c.modality != "image"}))
            return self._vocab

        def build() -> dict[str, dict[str, Any]]:
            return {c.name: BloomFilter.of(title_entries(self._title_index(c.name).keys)).to_dict()
                    for c in self.site.collections}

        if self._blooms is None:
            self._blooms = await asyncio.to_thread(build)
        return self._blooms

    def _titled(self, request: ProbeRequest) -> list[Candidate]:
        out = []
        for c in self.site.collections:
            if request.collections and c.name not in request.collections:
                continue
            found = titled(self._title_index(c.name), request.titles, contain=c.modality != "text")
            among = sorted({i for hits in found.values() for i in hits} - self.skip.get(c.name, set()))
            if not among:
                continue
            scored: dict[int, float] = {}
            if request.key:
                vec = self._embed_key(c.name, request.key)
                t0 = time.perf_counter()
                hits_s = self.index.search_scored(c.name, vec, len(among), 0, among)
                self._note_busy(search_seconds=time.perf_counter() - t0)
                among, scored = [pos for pos, _ in hits_s], dict(hits_s)
            kept = {i for hits in found.values() for i in [j for j in among if j in hits][: request.k]}
            rank = {i: r for r, i in enumerate((i for i in among if i in kept), 1)}
            stats = self._z(c.name, request.key) if request.scores and request.key else None
            variants = self._zv(c.name, request.key) if request.zvariants and request.key else None
            for title, hits in found.items():
                for i in sorted(hits & kept, key=rank.__getitem__):
                    cand = candidate(self.site.name, self.site.policy, c.name, self._at(c.name, i), rank[i], title,
                                     request.mentions, room=request.room or 0)
                    if i in scored:
                        self._scored(cand, scored[i], stats, variants)
                    out.append(cand)
        return out

    async def _source(self, o: ObjectRecord) -> prompts.ReadObject | None:
        if o.modality != "image":
            return prompts.ReadObject("table row" if o.modality == "table" else "text", _shown(o))
        path = self.extractor.raw_root / o.image_path if o.image_path else None
        uri = await asyncio.to_thread(load_image_data_uri, path, 512, 80) if path else None
        return None if uri is None else prompts.ReadObject("image", o.title, uri)

    async def synopsis(self, k: int = 16, seed: int = 0) -> dict[str, Any]:
        import base64

        cached = getattr(self, "_synopsis", None)
        if cached is not None and cached.get("k") == k and cached.get("seed") == seed:
            return cached
        out: dict[str, Any] = {"k": k, "seed": seed, "collections": {}}
        for c in self.site.collections:
            path, rows = (getattr(self, "vector_source", {}) or {}).get(c.name, (None, None))
            if path is None or not Path(path).exists():
                continue
            mm = np.load(path, mmap_mode="r")
            x = np.asarray(mm[np.asarray(rows)] if rows is not None else mm, dtype=np.float32)
            keep = np.ones(len(x), dtype=bool)
            for i in self.skip.get(c.name, set()):
                if 0 <= i < len(keep):
                    keep[i] = False
            x = x[keep]
            if not len(x):
                continue
            x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)
            if len(x) <= k:
                cents, sizes = x, np.ones(len(x), dtype=np.int64)
            else:
                from sklearn.cluster import MiniBatchKMeans

                km = await asyncio.to_thread(lambda x=x: MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=3,
                                                                         batch_size=4096).fit(x))
                cents = km.cluster_centers_.astype(np.float32)
                cents /= np.maximum(np.linalg.norm(cents, axis=1, keepdims=True), 1e-9)
                sizes = np.bincount(km.labels_, minlength=k)
            out["collections"][c.name] = {
                "encoder": c.encoder, "modality": c.modality, "n": int(len(x)), "dim": int(x.shape[1]),
                "centroids": base64.b64encode(cents.astype(np.float16).tobytes()).decode(),
                "sizes": [int(v) for v in sizes]}
        self._synopsis = out
        return out

    async def catalog(self, collections: list[str] | None = None, offset: int = 0,
                      limit: int = 100_000) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for c in self.site.collections:
            if collections and c.name not in collections:
                continue
            objs = self.objects[c.name]
            out[c.name] = {"n": len(objs), "oids": [[o.oid, o.doc_id, o.modality]
                                                    for o in objs[offset: offset + limit]]}
        return out

    async def info(self) -> dict[str, Any]:
        cols = [CollectionInfo(c.name, c.modality, c.encoder, len(self.objects[c.name])) for c in self.site.collections]
        llm = self.extractor.llm
        return SiteInfo(self.site.name, self.site.machine, self.site.policy, cols, self.fed.extractor_version,
                        prompts.READ_VERSION, models={r: llm.model(r) for r in SITE_ROLES}, trust=llm.trust,
                        wire=WIRE_VERSION, commit=await asyncio.to_thread(code_commit)).to_dict()

    async def probe(self, requests: list[dict[str, Any]]) -> list[list[dict[str, Any]] | dict[str, Any]]:
        reqs = [ProbeRequest.from_dict(r) for r in requests]
        for r in reqs:
            check_probe(r)
            self._known_epoch(r.epoch)
        t0 = time.perf_counter()

        def serve(r: ProbeRequest) -> list[Candidate] | dict[str, Any]:
            with prompts.mention_scope(r.mention), epoch_scope(r.epoch):
                if not r.semijoin:
                    found = self._titled(r) if r.kind == "title" else self._search(r)
                    return self._viewed(self._versions_of(found, r), r)
                t = time.perf_counter()
                cands, counts = self._semijoin(r)
                cands = self._viewed(self._versions_of(cands, r), r)
                return {"c": [c.to_dict() for c in cands], "n": [n.to_dict() for n in counts],
                        "ms": round(1000 * (time.perf_counter() - t), 3)}

        results = await asyncio.to_thread(lambda: [serve(r) for r in reqs])
        self._note_busy(probe_messages=1, probe_requests=len(reqs), busy_seconds=time.perf_counter() - t0)
        return [res if isinstance(res, dict) else [c.to_dict() for c in res] for res in results]

    def _hippo_query(self, collection: str, encoder: str, question: str) -> np.ndarray:
        if encoder == self.site.collection(collection).encoder:
            return self._embed_key(collection, question)
        with self._hippo_lock:
            enc = self._hippo_encoders.get(encoder)
            if enc is None:
                enc = self._hippo_encoders[encoder] = load_encoder(encoder, "cpu")
        return np.asarray(enc.encode_queries([question])[0], dtype=np.float32)

    async def hippo(self, requests: list[dict[str, Any]]) -> list[list[dict[str, Any]] | dict[str, Any]]:
        live = any(bool(r.get("live")) for r in requests)
        reqs = [HippoRequest.from_dict({k: v for k, v in r.items() if k != "live"}) for r in requests]
        llm = self._live_llm() if live else self.extractor.llm
        t0 = time.perf_counter()

        async def serve(r: HippoRequest) -> list[dict[str, Any]] | dict[str, Any]:
            out, used = [], Usage()
            for c in self.site.collections:
                if c.modality != "text" or (r.collections and c.name not in r.collections):
                    continue
                index = await asyncio.to_thread(self._hippo_of, c.name)
                if index is None:
                    continue
                vec = await asyncio.to_thread(self._hippo_query, c.name, index.meta.get("encoder") or c.encoder,
                                              r.question)
                found = await index.retrieve(r.question, r.k, query=vec, llm=llm, slots=self._sem, usage=used)
                out += [candidate(self.site.name, self.site.policy, c.name, self.objects[c.name][pos], rank)
                        for rank, pos in enumerate(found, 1)]
            cands = [c.to_dict() for c in out]
            return {"c": cands, "u": used.to_dict()} if used.calls else cands

        results = await asyncio.gather(*(serve(r) for r in reqs))
        self._note_busy(busy_seconds=time.perf_counter() - t0)
        return list(results)

    async def hippo_index(self, force: bool = False, device: str = "auto",
                          encoder: str | None = None) -> dict[str, dict[str, Any]]:
        if self.site.data_from or self.fed.data_from:
            raise ValueError(f"{self.fed.name}/{self.site.name} reads {self.site.data_from or self.fed.data_from}'s "
                             "data (data_from): its source's HippoRAG 2 index serves it; index the source instead")
        out = {}
        for c in self.site.collections:
            if c.modality != "text":
                continue
            cdir = self.fed.collection_data_dir(self.site.name, c.name)
            index = await HippoIndex.build(cdir / HIPPO_DIR, self.objects[c.name],
                                           encoder=load_encoder(encoder or c.encoder, device), llm=self.extractor.llm,
                                           concurrency=int(os.environ.get("FEDQUEST_HIPPO_CONCURRENCY") or 0)
                                           or self.site.max_concurrency, skip=self.skip.get(c.name, set()),
                                           stamp=objects_stamp(cdir / "objects.jsonl"), force=force,
                                           label=f"{self.fed.name}/{self.site.name}/{c.name}")
            with self._hippo_lock:
                self._hippo[c.name] = index
            out[c.name] = index.stats
        return out

    def _live_llm(self) -> Any:
        with self._hippo_lock:
            if getattr(self, "_hippo_live", None) is None:
                client = self.extractor.llm.clone()
                client.live = True
                self._hippo_live = client
            return self._hippo_live

    def hippo_counts(self) -> dict[str, dict[str, int]]:
        with self._hippo_lock:
            return {c: dict(index.counts) for c, index in self._hippo.items() if index is not None}

    def _store_key(self, version: str, modality: str, block: Sequence[str] | None = None) -> str:
        model = self.extractor.llm.model("extract_image" if modality == "image" else "extract_text")
        if block is not None:
            version = f"{version}#{stable_hash(list(block))}"
        return version if model == LEGACY_MODEL else f"{version}@{model}"

    async def _stand_in(self, stored: ExtractionResult, timing_key: str,
                        call: Callable[[], Any] | None) -> None:
        llm = self.extractor.llm
        if call is not None:
            out = await call()
            made = out if isinstance(out, list) else [out]
            if any(r.error for r in made) or all(getattr(r, "call_wall", None) is None for r in made):
                stored.replay = True
        elif getattr(llm, "timed", False) and hasattr(llm, "hold_and_sleep"):
            from fedquest.llm import timing

            seconds = timing.lookup(timing_key)
            if seconds is None:
                seconds = stored.usage.seconds + float(getattr(llm, "endpoint_s", 0.0) or 0.0)
            async with self._sem:
                await llm.hold_and_sleep(seconds, role="extract_image" if stored.modality == "image" else
                                         "extract_text")

    def _note_call(self, timing_key: str | None, wall: float, service: float) -> None:
        if hasattr(self, "_note_busy"):
            self._note_busy(extract_calls=1)
        if timing_key is not None:
            from fedquest.llm import timing

            timing.note(timing_key, wall, service, live=True, site=self.site.name)

    async def _extract_one(self, oid: str, include_raw: bool, mode: str, query: str | None,
                           live: bool = False) -> ExtractionResult:
        o = self._obj(oid)
        if o is None:
            return ExtractionResult(self.site.name, oid, "text", "", [], Usage(), error="unknown oid")
        conditioned = mode in ("query", "query_t")
        titled = mode == "query_t" and o.modality == "image"
        if conditioned:
            if not query:
                raise ValueError("query-conditioned extraction needs a query")
            version = prompts.QUERY_EXTRACT_T_VERSION if titled else prompts.QUERY_EXTRACT_VERSION
            qhash = stable_hash(query)
        else:
            version, qhash = self.fed.extractor_version, ""
        key = self._store_key(version, o.modality)
        timing_key = f"store:{self.site.name}:{key}:{oid}:{qhash}"

        async def call() -> ExtractionResult:
            async with self._sem:
                t0 = time.perf_counter()
                if conditioned:
                    out = await self.extractor.extract_query(o, self.site.name, query or "", titled, live=live)
                else:
                    out = await self.extractor.extract(o, self.site.name, live=live)
                wall = time.perf_counter() - t0
            self._note_call(timing_key if live else None, getattr(out, "call_wall", None) or wall, out.usage.seconds)
            return out

        stored = await asyncio.to_thread(self.store.get, oid, key, qhash)
        if stored is not None:
            stored.replay = not live
            if not stored.inference_bytes:
                stored.inference_bytes = await asyncio.to_thread(self.extractor.inference_bytes, o, mode)
            stored.trust = self.extractor.llm.trust
            await self._stand_in(stored, timing_key, call if live else None)
            result = stored
        else:
            result = await call()
            if result.error is None:
                await asyncio.to_thread(self.store.put, result, qhash, key)
        if include_raw and self.site.policy == "snippets" and o.modality != "image":
            result.raw_text = titled_text(o)
        return result

    async def extract(self, oids: list[str], include_raw: bool = False, mode: str = "kg",
                      query: str | None = None, blocks: list[list[str]] | None = None,
                      live: bool = False) -> list[dict[str, Any]]:
        t0 = time.perf_counter()
        if mode != "query_block":
            results = await asyncio.gather(*(self._extract_one(oid, include_raw, mode, query, live) for oid in oids))
        else:
            results = await self._extract_blocks(oids, include_raw, query, blocks or [], live)
        dt = time.perf_counter() - t0
        if hasattr(self, "_note_busy"):
            self._note_busy(extract_seconds=dt, busy_seconds=dt)
        return [r.to_dict() for r in results]

    async def _extract_blocks(self, oids: list[str], include_raw: bool, query: str | None, blocks: list[list[str]],
                              live: bool) -> list[ExtractionResult]:
        if not query:
            raise ValueError("query-conditioned extraction needs a query")
        qhash = stable_hash(query)
        out: dict[str, ExtractionResult] = {}
        order = list(dict.fromkeys([*oids, *(oid for block in blocks for oid in block)]))
        placed: set[str] = set()
        calls: list[list[ObjectRecord]] = []
        for block in [*blocks, *([oid] for oid in oids)]:
            by_modality: dict[str, list[ObjectRecord]] = {}
            for oid in dict.fromkeys(block):
                if oid in placed:
                    continue
                placed.add(oid)
                o = self._obj(oid)
                if o is None:
                    out[oid] = ExtractionResult(self.site.name, oid, "text", "", [], Usage(), error="unknown oid")
                else:
                    by_modality.setdefault(o.modality, []).append(o)
            for modality, objs in by_modality.items():
                cap = prompts.PER_CALL.get(modality, 1)
                calls += [objs[i : i + cap] for i in range(0, len(objs), cap)]

        async def run(objs: list[ObjectRecord]) -> list[ExtractionResult]:
            key = self._store_key(prompts.QUERY_EXTRACT_BLOCK_VERSION, objs[0].modality, [o.oid for o in objs])
            timing_key = f"store:{self.site.name}:{key}:{objs[0].oid}:{qhash}"

            async def call() -> list[ExtractionResult]:
                async with self._sem:
                    t0 = time.perf_counter()
                    done = await self.extractor.extract_query_block(objs, self.site.name, query or "", live=live)
                    wall = time.perf_counter() - t0
                self._note_call(timing_key if live else None, getattr(done[0], "call_wall", None) or wall,
                                done[0].usage.seconds)
                return done

            stored = await asyncio.to_thread(lambda: [self.store.get(o.oid, key, qhash) for o in objs])
            hits = [s for s in stored if s is not None]
            if len(hits) == len(objs):
                for s in hits:
                    s.replay, s.trust = not live, self.extractor.llm.trust
                await self._stand_in(hits[0], timing_key, call if live else None)
                return hits
            done = await call()
            ok = [r for r in done if r.error is None]
            if ok:
                await asyncio.to_thread(lambda: [self.store.put(r, qhash, key) for r in ok])
            return done

        for done in await asyncio.gather(*(run(objs) for objs in calls)):
            for r in done:
                out[r.oid] = r
        for r in out.values():
            o = self._obj(r.oid)
            if include_raw and o is not None and self.site.policy == "snippets" and o.modality != "image":
                r.raw_text = o.text
        return [out[oid] for oid in order]

    async def read(self, question: str, unknowns: list[dict[str, Any]], blocks: list[list[str]],
                   prefilter: list[str] | None = None, exempt: list[str] | None = None,
                   labels: dict[str, str] | None = None, include_raw: bool = False,
                   extract: bool = False, live: bool = False, mention: int = 1, versions: bool = False,
                   harvest: int = 0, epoch: str | None = None) -> dict[str, Any]:
        self._known_epoch(epoch)
        with prompts.mention_scope(mention), epoch_scope(epoch):
            return await self._read(question, unknowns, blocks, prefilter, exempt, labels, include_raw, extract, live,
                                    versions, harvest)

    async def _read(self, question: str, unknowns: list[dict[str, Any]], blocks: list[list[str]],
                    prefilter: list[str] | None, exempt: list[str] | None, labels: dict[str, str] | None,
                    include_raw: bool, extract: bool, live: bool, versions: bool = False,
                    harvest: int = 0) -> dict[str, Any]:
        if not unknowns:
            return ReadResult(self.site.name, [], {}, Usage()).to_dict()
        labels = labels or {}
        picked, skipped = _picked(blocks, self._obj, prefilter or [], set(exempt or []))
        if extract:
            objs = [o for block in picked for o in block]
            done = await asyncio.gather(*(self._extract_one(o.oid, False, "query", question, live) for o in objs))
            ok = [r for r in done if r.error is None]
            out = ReadResult(self.site.name, [r.oid for r in ok], {}, Usage(), skipped=skipped, extractions=list(done),
                             modalities={r.oid: r.modality for r in ok}, doc_ids={r.oid: r.doc_id for r in ok},
                             titles={o.oid: labels.get(o.oid, o.title) for o in objs}, trust=self.extractor.llm.trust)
            for r in ok:
                out.usage.add(r.usage)
                out.usage_by_modality.setdefault(r.modality, Usage()).add(r.usage)
            out.replay = bool(ok) and all(r.replay for r in ok)
            return out.to_dict()
        calls: list[tuple[int, list[tuple[ObjectRecord, prompts.ReadObject]]]] = []
        for b, block in enumerate(picked, 1):
            items = [(o, ro) for o in block if (ro := await self._source(o)) is not None]
            calls += [(b, group) for group in _calls(items)]
        objs = [o for _, group in calls for o, _ in group]
        result = ReadResult(self.site.name, [o.oid for o in objs], {}, Usage(), skipped=skipped,
                            modalities={o.oid: o.modality for o in objs}, doc_ids={o.oid: o.doc_id for o in objs},
                            titles={o.oid: labels.get(o.oid, o.title) for o in objs},
                            inference_bytes=sum(content_bytes(ro.text, ro.image) for _, g in calls for _, ro in g),
                            trust=self.extractor.llm.trust)
        if versions:
            result.versions = {o.oid: o.version for o in objs}
        if not calls:
            return result.to_dict()
        unk = [prompts.ReadUnknown.from_dict(u) for u in unknowns]
        t0 = time.perf_counter()
        replies = await asyncio.gather(*(self._read_call(question, group, unk, live, harvest) for _, group in calls),
                                       return_exceptions=True)
        dt = time.perf_counter() - t0
        self._note_busy(read_calls=len(calls), read_seconds=dt, busy_seconds=dt)
        failed = None
        for (b, group), reply in zip(calls, replies, strict=True):
            if isinstance(reply, BaseException):
                log.warning("read at %s failed in block %d (%s): %s", self.site.name, b,
                            ", ".join(o.oid for o, _ in group), reply)
                failed = failed or f"{type(reply).__name__}: {reply}"
        if failed is not None:
            for (_, group), reply in zip(calls, replies, strict=True):
                if isinstance(reply, LLMReply):
                    result.usage.add(reply.usage)
                    result.usage_by_modality.setdefault(group[0][0].modality, Usage()).add(reply.usage)
            result.error = failed
            return result.to_dict()
        done = [r for r in replies if isinstance(r, LLMReply)]
        result.replay = any(r.cached for r in done) if live else all(r.cached for r in done)
        for (_, group), r in zip(calls, done, strict=True):
            result.usage.add(r.usage)
            result.usage_by_modality.setdefault(group[0][0].modality, Usage()).add(r.usage)
            reply = prompts.read_reply(r.text)
            _bind(result, [o for o, _ in group], reply, unk, labels)
            if harvest and group[0][0].modality != "image":
                result.harvest += _harvested([o for o, _ in group], reply, harvest)
        if include_raw and self.site.policy == "snippets":
            cited = {h["oid"] for hits in result.bindings.values() for h in hits}
            result.raw = {o.oid: titled_text(o) for o in objs if o.oid in cited and o.modality != "image"}
        return result.to_dict()

    async def _read_call(self, question: str, group: list[tuple[ObjectRecord, prompts.ReadObject]],
                         unk: list[prompts.ReadUnknown], live: bool = False, harvest: int = 0) -> LLMReply:
        image = group[0][0].modality == "image"
        h = 0 if image else harvest
        messages = prompts.read_messages(question, [ro for _, ro in group], unk, harvest=h)
        role = "extract_image" if image else "extract_text"
        async with self._sem:
            return await self.extractor.llm.chat(messages, role=role, task="read", json_mode=True,
                                                 max_tokens=prompts.read_max_tokens(unk) + prompts.HARVEST_TOKENS * h,
                                                 live=live or None)

    async def versions(self, oids: list[str], epoch: str | None = None) -> dict[str, str]:
        self._known_epoch(epoch)
        with epoch_scope(epoch):
            return {oid: o.version for oid in oids if (o := self._obj(oid)) is not None}

    def _known_epoch(self, epoch: str | None) -> None:
        if epoch is None:
            return
        with _OVERLAYS_LOCK:
            known = epoch in _OVERLAYS.get((self.fed.name, self.site.name), {})
        if not known:
            raise ValueError(f"{self.fed.name}/{self.site.name}: no updates under epoch {epoch!r} here (never applied, "
                             "or this agent restarted since): apply the edits again (fedquest.eval.kg_update apply)")

    async def witnesses(self, items: list[dict[str, Any]], epoch: str | None = None) -> list[bool]:
        self._known_epoch(epoch)
        with epoch_scope(epoch):
            out = []
            for it in items:
                o = self._obj(str(it.get("oid", "")))
                out.append(o is not None and o.modality != "image" and prompts.span_holds(
                    o.text, str(it.get("w") or ""), it.get("known") or None, o.title))
            return out

    async def update(self, epoch: str, edits: list[dict[str, Any]]) -> dict[str, str]:
        out: dict[str, str] = {}
        with _OVERLAYS_LOCK:
            space = _OVERLAYS.setdefault((self.fed.name, self.site.name), {}).setdefault(epoch, {})
            for e in edits:
                hit = self.where.get(str(e["oid"]))
                if hit is None:
                    continue
                o = dataclasses.replace(self.objects[hit[0]][hit[1]], text=str(e["text"]))
                space[o.oid] = o
                out[o.oid] = o.version
        return out

    async def fetch_raw(self, oids: list[str], max_px: int = 384, quality: int = 75,
                        epoch: str | None = None) -> list[dict[str, Any]]:
        self._known_epoch(epoch)

        def load() -> list[RawObject]:
            with epoch_scope(epoch):
                return _load()

        def _load() -> list[RawObject]:
            out: list[RawObject] = []
            used = 0
            for oid in oids:
                o = self._obj(oid)
                if o is None:
                    continue
                if self.site.policy in ("facts", "sealed"):
                    out.append(RawObject(self.site.name, oid, o.modality, o.title, denied=True, doc_id=o.doc_id))
                    continue
                if o.modality == "image":
                    uri = load_image_data_uri(self.extractor.raw_root / (o.image_path or ""), max_side=max_px,
                                              quality=quality)
                    b64 = uri.split(",", 1)[1] if uri else None
                    raw = RawObject(self.site.name, oid, "image", o.title, text=o.text, image_b64=b64,
                                    nbytes=len(b64 or ""), denied=b64 is None, doc_id=o.doc_id)
                else:
                    raw = RawObject(self.site.name, oid, o.modality, o.title, text=o.text,
                                    nbytes=len(o.text.encode()), doc_id=o.doc_id)
                size = len(json.dumps(raw.to_dict()))
                if out and used + size > RAW_BUDGET_BYTES:
                    break
                used += size
                out.append(raw)
            return out

        return [r.to_dict() for r in await asyncio.to_thread(load)]

    async def stats(self, busy_only: bool = False) -> dict[str, Any]:
        if busy_only:
            return {"site": self.site.name, "busy": self.busy()}
        store = await asyncio.to_thread(self.store.stats)
        return {
            "site": self.site.name,
            "federation": self.fed.name,
            "policy": self.site.policy,
            "collections": {c: len(objs) for c, objs in self.objects.items()},
            "store": store,
            "uptime_s": round(time.time() - self._started, 1),
            "busy": self.busy(),
        }

    def busy(self) -> dict[str, float]:
        with self._busy_lock:
            out = {k: round(v, 4) if k.endswith("seconds") else int(v) for k, v in self._busy.items()}
        index = edge_views.INDEX.stats()
        out |= {"view_index_objects": int(index["objects"]), "view_index_sentences": int(index["held_sentences"])}
        return out

    async def warmup(self) -> dict[str, Any]:
        def touch() -> tuple[dict[str, int], dict[str, str]]:
            out, kinds = {}, {}
            for c in self.site.collections:
                mat, images = self._reference(c.name)
                kinds[c.name] = "none" if mat is None else "images" if images else "text"
                out[c.name] = self.index.count(c.name)
            return out, kinds

        indexed, kinds = await asyncio.to_thread(touch)
        return {"site": self.site.name, "indexed": indexed, "anchors": kinds,
                "modalities": {c.name: c.modality for c in self.site.collections}}

    async def close(self) -> None:
        await asyncio.to_thread(self.index.close)
        await asyncio.to_thread(self.store.close)


def _shown(o: ObjectRecord) -> str:
    return o.text if o.modality == "table" else f"{o.title}: {o.text}"


def mention_text(o: Any) -> str:
    return f"{o.title} {o.text}"


def names_any(o: Any, values: Sequence[str]) -> bool:
    return any(prompts.mentions(mention_text(o), v) for v in values)


def check_probe(r: ProbeRequest) -> None:
    if r.kind not in ("question", "bound", "title"):
        raise ValueError(f"unknown probe kind {r.kind!r}")
    if isinstance(r.mention, bool) or r.mention not in prompts.MENTION_VERSIONS:
        raise ValueError(f"unknown mention test {r.mention!r} (prompts.MENTION_VERSIONS)")
    if r.semijoin and (r.kind == "title" or not r.mentions):
        raise ValueError("a semijoin probe ranks the objects that name one of its values: a key request (kind question "
                         "or bound) with its mentions")


def candidate(site: str, policy: str, collection: str, o: ObjectRecord, rank: int, label: str = "",
              mentions: Sequence[str] = (), room: int = 0) -> Candidate:
    if policy in ("facts", "sealed"):
        summary = o.title
    elif o.modality == "image":
        summary = o.text
    else:
        summary = f"{o.title}: {truncate(o.text, room or SNIPPET_CHARS)}"
    complete = policy == "snippets" and o.modality != "image" and len(o.text) <= (room or SNIPPET_CHARS)
    tokens = prompts.IMAGE_TOKENS if o.modality == "image" else approx_tokens(_shown(o))
    named = names_any(o, mentions) if mentions and o.modality != "image" and policy != "sealed" else None
    return Candidate(site, collection, o.oid, o.modality, rank, summary, tokens, o.doc_id, complete, label, named)


def _title_key(title: str) -> str:
    return prompts.fold(norm_entity(title))


class TitleIndex:
    def __init__(self, titles: Sequence[str]) -> None:
        self.keys = [_title_key(t) for t in titles]
        self.same: dict[str, list[int]] = {}
        self.words: dict[str, set[int]] = {}
        for i, key in enumerate(self.keys):
            self.same.setdefault(key, []).append(i)
            for w in set(key.split()):
                self.words.setdefault(w, set()).add(i)

    def find(self, title: str) -> list[int]:
        key = _title_key(title)
        if not key:
            return []
        if key in self.same:
            return list(self.same[key])
        words = key.split()
        if not any(len(w) >= 4 for w in words):
            return []
        pool = set.intersection(*(self.words.get(w, set()) for w in words))
        named = sorted({self.keys[i] for i in pool if f" {key} " in f" {self.keys[i]} "},
                       key=lambda k: (not k.startswith(key), len(k), k))[:2]
        return sorted(i for k in named for i in self.same[k])


def titled(index: TitleIndex, titles: Sequence[str], contain: bool) -> dict[str, set[int]]:
    owner: dict[int, str] = {}
    for t in titles:
        for i in index.same.get(_title_key(t), []):
            owner.setdefault(i, t)
    if contain:
        for t in titles:
            for i in index.find(t):
                owner.setdefault(i, t)
    out: dict[str, set[int]] = {t: set() for t in titles}
    for i, t in owner.items():
        out[t].add(i)
    return out


_NONE = np.zeros(0, dtype=np.int32)


def _parent_postings(path: Path) -> Postings:
    key = str(path.resolve())
    with _PARENT_POSTINGS_LOCK:
        if key in _PARENT_POSTINGS:
            return _PARENT_POSTINGS[key]
        objs = _objects(path)
        st = path.stat()
        stamp = {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "objects": len(objs)}
        cache = path.with_name(POSTINGS_FILE)
        post = Postings.load(cache, stamp)
        if post is None:
            t0 = time.perf_counter()
            post = Postings.build([mention_text(o) for o in objs])
            log.info("%s: postings of %d objects in %.1f s", path.parent, post.n, time.perf_counter() - t0)
            try:
                post.save(cache, stamp)
            except OSError as e:
                log.warning("postings not cached at %s (%s)", cache, e)
        _PARENT_POSTINGS[key] = post
        return post


def _harvested(group: list[ObjectRecord], reply: Any, cap: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for f in prompts.parse_harvest(reply, len(group))[:cap]:
        o = group[f["object"] - 1]
        text = f"{o.title} {o.text}"
        if len(f["value"]) > MAX_PART_CHARS or len(f["subject"]) > MAX_PART_CHARS:
            continue
        if prompts.mentions(text, f["subject"]) and prompts.witnessed(text, f["value"]):
            fact = {"entity": f["subject"], "relation": f["relation"], "value": f["value"], "oid": o.oid}
            span = prompts.witness_span(o.text, f["value"], o.title)
            out.append({**fact, "w": span} if span else fact)
    return out


class Postings:
    def __init__(self, vocab: list[str], starts: np.ndarray, post: np.ndarray, n: int) -> None:
        self.vocab, self.starts, self.post, self.n = vocab, starts, post, n

    @classmethod
    def build(cls, texts: Sequence[str]) -> Postings:
        where: dict[str, list[int]] = {}
        for i, text in enumerate(texts):
            for w in set(prompts._norm_text(text).split()):
                where.setdefault(w, []).append(i)
        vocab = sorted(where)
        starts = np.zeros(len(vocab) + 1, dtype=np.int64)
        np.cumsum([len(where[w]) for w in vocab], out=starts[1:])
        post = np.fromiter(itertools.chain.from_iterable(where[w] for w in vocab), dtype=np.int32,
                           count=int(starts[-1]))
        return cls(vocab, starts, post, len(texts))

    def save(self, path: Path, stamp: dict[str, Any]) -> None:
        meta = json.dumps({**stamp, "version": POSTINGS_VERSION, "words": len(self.vocab)}, sort_keys=True)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        try:
            with open(tmp, "wb") as f:
                np.savez(f, meta=np.frombuffer(meta.encode(), dtype=np.uint8), starts=self.starts, post=self.post,
                         vocab=np.frombuffer("\n".join(self.vocab).encode(), dtype=np.uint8))
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)

    @classmethod
    def load(cls, path: Path, stamp: dict[str, Any]) -> Postings | None:
        try:
            with np.load(path) as z:
                meta = json.loads(bytes(z["meta"]).decode())
                if {k: meta.get(k) for k in stamp} != stamp or meta.get("version") != POSTINGS_VERSION:
                    return None
                vocab = bytes(z["vocab"]).decode().split("\n") if meta["words"] else []
                return cls(vocab, z["starts"], z["post"], int(stamp["objects"]))
        except (OSError, ValueError, KeyError):
            return None

    def restricted(self, rows: np.ndarray) -> Postings:
        rows = np.asarray(rows, dtype=np.int64)
        local = np.full(self.n, -1, dtype=np.int64)
        local[rows] = np.arange(len(rows), dtype=np.int64)
        mapped = local[self.post]
        keep = mapped >= 0
        kept = (np.add.reduceat(keep.astype(np.int64), np.minimum(self.starts[:-1], max(len(keep) - 1, 0)))
                if len(self.vocab) and len(keep) else np.zeros(len(self.vocab), np.int64))
        empty = self.starts[:-1] == self.starts[1:]
        kept = np.where(empty, 0, kept)
        alive = kept > 0
        vocab = [w for w, a in zip(self.vocab, alive.tolist(), strict=True) if a]
        starts = np.zeros(len(vocab) + 1, dtype=np.int64)
        np.cumsum(kept[alive], out=starts[1:])
        return Postings(vocab, starts, mapped[keep].astype(np.int32), int(len(rows)))

    def word(self, w: str) -> np.ndarray:
        i = bisect.bisect_left(self.vocab, w)
        return self.post[self.starts[i] : self.starts[i + 1]] if i < len(self.vocab) and self.vocab[i] == w else _NONE

    def prefixed(self, p: str) -> np.ndarray:
        lo = bisect.bisect_left(self.vocab, p)
        hi = (bisect.bisect_left(self.vocab, p[:-1] + chr(ord(p[-1]) + 1), lo) if ord(p[-1]) < 0x10FFFF
              else len(self.vocab))
        found = self.post[self.starts[lo] : self.starts[hi]]
        return found if hi - lo <= 1 else np.unique(found)

    def candidates(self, value: str) -> np.ndarray | None:
        always, every, half = mention_keys(value)
        if always:
            return None
        found = _NONE
        if every:
            runs = sorted((self.word(w) for w in set(every)), key=len)
            found = runs[0]
            for run in runs[1:]:
                found = np.intersect1d(found, run, assume_unique=True) if found.size else found
        if half:
            tau = prompts.mention_test()[1]
            tally = np.zeros(self.n, dtype=np.int32)
            for w, times in Counter(half).items():
                tally[self.prefixed(w)] += times
            found = np.union1d(found, np.flatnonzero(tau.denominator * tally >= tau.numerator * len(half)))
        return found.astype(np.int32, copy=False)


def _picked(blocks: list[list[str]], find: Callable[[str], Any], prefilter: list[str], exempt: set[str],
            ) -> tuple[list[list[Any]], list[str]]:
    seen: set[str] = set()
    picked: list[list[Any]] = []
    skipped: list[str] = []
    for block in blocks:
        objs = []
        for oid in block:
            o = find(oid)
            if o is None or oid in seen:
                continue
            seen.add(oid)
            if prefilter and o.modality != "image" and oid not in exempt and not names_any(o, prefilter):
                skipped.append(oid)
            else:
                objs.append(o)
        picked.append(objs)
    return picked, skipped


def _calls(items: list[tuple[ObjectRecord, Any]]) -> list[list[tuple[ObjectRecord, Any]]]:
    out = []
    for mod in dict.fromkeys(o.modality for o, _ in items):
        group = [it for it in items if it[0].modality == mod]
        cap = prompts.PER_CALL[mod]
        out += [group[i : i + cap] for i in range(0, len(group), cap)]
    return out


def _bind(result: ReadResult, group: list[ObjectRecord], reply: Any, unknowns: list[prompts.ReadUnknown],
          labels: dict[str, str]) -> None:
    hits = prompts.parse_read(reply, unknowns, len(group))
    for u in unknowns:
        for h in hits.get(u.name, []):
            o = group[h.object - 1]
            hit = _validated(o, u, h, labels.get(o.oid))
            if hit is None:
                result.dropped += 1
            else:
                result.bindings.setdefault(u.name, []).append(hit)


def _validated(o: ObjectRecord, u: prompts.ReadUnknown, h: prompts.ReadHit, label: str | None = None,
               ) -> dict[str, Any] | None:
    if len(h.entity) > MAX_PART_CHARS:
        return None
    if o.modality == "image":
        conds = prompts.claims(None, u, h, f"{o.title} {o.text}")
        entity = h.entity
        if u.mode == "filter":
            own = [label, o.title] if label else [o.title]
            said = prompts.judgment(entity)
            if said is False or (said is None and prompts.canonical(entity, own) is None):
                return None
            entity = own[0]
        return {"entity": entity, "oid": o.oid, "conditions": conds} if conds else None
    text = f"{o.title} {o.text}"
    if not prompts.stated(text, u, h.entity):
        return None
    conds = prompts.claims(text, u, h)
    if not conds:
        return None
    out = {"entity": h.entity, "oid": o.oid, "conditions": conds}
    known = next((u.values[c - 1] for c in conds if 0 < c <= len(u.values) and u.values[c - 1]), None)
    span = prompts.witness_span(o.text, h.entity, o.title, known)
    return {**out, "w": span} if span else out


def decode_image(raw: RawObject) -> str | None:
    return f"data:image/jpeg;base64,{raw.image_b64}" if raw.image_b64 else None
