# Federation: fans each round out to the sites and charges the ledger
from __future__ import annotations

import asyncio
import logging
import math
import time
from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any, Protocol

from fedquest import timeline
from fedquest.bloom import BITS_PER_KEY, HASHES, BloomFilter
from fedquest.config import SITE_ROLES
from fedquest.errors import ReplyTooLarge, SiteRetryable
from fedquest.federation import probe_trace
from fedquest.federation.network import timed_call
from fedquest.kg.normalize import norm_entity
from fedquest.ledger import Ledger
from fedquest.llm import prompts
from fedquest.llm.prompts import READ_VERSION
from fedquest.types import (
    Candidate,
    ExtractionResult,
    ExtractMode,
    HippoRequest,
    ProbeCounts,
    ProbeRequest,
    RawObject,
    ReadResult,
    SiteInfo,
    Usage,
)
from fedquest.utils import truncate

if TYPE_CHECKING:
    from fedquest.config import FederationCfg, ModelsCfg

log = logging.getLogger(__name__)

Ref = tuple[str, str]
SERVED_ACTIONS = ("probe", "read", "extract", "fetch_raw")
IDEMPOTENT = frozenset({"info", "blooms", "probe", "fetch_raw", "versions", "witnesses", "catalog", "synopsis"})
SITE_RETRIES = 2
SPLIT_FIELDS = {"probe": "requests", "hippo": "requests", "fetch_raw": "oids", "versions": "oids", "witnesses": "items"}
PROFILED_ACTIONS = frozenset({"read", "extract", "info", "stats"})


def retryable(action: str, e: Exception) -> bool:
    return action in IDEMPOTENT and isinstance(e, SiteRetryable | ConnectionError)
WIRE_VERSION = "wire-v8"


def _holds(info: SiteInfo | None, collections: list[str] | None) -> bool:
    return not collections or info is None or any(c.name in collections for c in info.collections)


def _texts(info: SiteInfo | None, collections: list[str] | None) -> bool:
    return info is None or any(c.modality == "text" and (not collections or c.name in collections)
                               for c in info.collections)


def _told(r: ProbeRequest) -> tuple[list[str], list[str]]:
    values = [*r.titles, *r.mentions, *([r.value] if r.value else [])]
    if r.kind == "bound" and not r.value and not r.mentions:
        values.append(r.key)
    return [r.key], values


VOCAB_BITS = 1 << 21


def mention_keys(value: str) -> tuple[bool, list[str], list[str]]:
    ne = prompts._norm_text(norm_entity(value)).strip()
    if ne in prompts._JUDGMENTS:
        return True, [], []
    words = ne.split()
    every = [w for w in words if len(w) >= 3 or w.isdigit()] or words
    half = prompts.mention_words(value)
    return False, every, half if len(half) >= 2 else []


def vocab_entries(words: Iterable[str]) -> list[str]:
    out: dict[str, None] = {}
    for w in words:
        out[f"v:{w}"] = None
        if len(w) >= 4:
            out[f"p:{w[:4]}"] = None
        for j in range(1, min(3, len(w)) + 1):
            if not w[:j].isdigit():
                break
            out[f"p:{w[:j]}"] = None
    return list(out)


def vocab_filters(vocabularies: dict[str, Iterable[str]]) -> dict[str, dict[str, Any]]:
    entries = {c: vocab_entries(words) for c, words in vocabularies.items()}
    total = sum(map(len, entries.values())) or 1
    out = {}
    for c, keys in entries.items():
        n = max(1, len(keys))
        m = max(64, min(n * BITS_PER_KEY, VOCAB_BITS * n // total))
        bf = BloomFilter(m, max(1, min(HASHES, round(math.log(2) * m / n))))
        for key in keys:
            bf.add(key)
        out[f"vocab:{c}"] = bf.to_dict()
    return out


SHARE_EPS = 0.05


def share_slack(k: int, p: float, eps: float = SHARE_EPS) -> int:
    if k <= 0 or p <= 0 or p >= 1:
        return 0
    # least s with E[(X - ceil(kp) - s)+] <= eps k p, X ~ Bin(k, p) the group's top k held here
    base = math.ceil(k * p - 1e-12)
    pmf = [math.comb(k, x) * p**x * (1 - p) ** (k - x) for x in range(k + 1)]
    s = 0
    while sum((x - base - s) * q for x, q in enumerate(pmf) if x > base + s) > eps * k * p + 1e-12:
        s += 1
    return s


def may_mention(bf: BloomFilter, value: str) -> bool:
    # yes/no judgments pass; else every whole-word key must be held, or a tau share of 4-char prefixes
    always, every, half = mention_keys(value)
    if always or (every and all(f"v:{w}" in bf for w in every)):
        return True
    tau = prompts.mention_test()[1]
    return bool(half) and tau.denominator * sum(f"p:{w[:4]}" in bf for w in half) >= tau.numerator * len(half)


def _may_join(info: SiteInfo | None, filters: dict[str, BloomFilter] | None, r: ProbeRequest) -> bool:
    if info is None or filters is None:
        return True
    with prompts.mention_scope(r.mention):
        for c in info.collections:
            if r.collections and c.name not in r.collections:
                continue
            if c.modality == "image":
                if r.semijoin:
                    continue
                return True
            bf = filters.get(c.name)
            if bf is None or any(may_mention(bf, v) for v in r.vocab or ()):
                return True
    return False


class SiteEndpoint(Protocol):
    name: str

    async def call(self, action: str, **kwargs: Any) -> Any:
        ...

    async def close(self) -> None: ...


class Federation:
    name: str = ""

    def __init__(self, endpoints: Sequence[SiteEndpoint], *, domains: dict[str, str] | None = None):
        self.endpoints: dict[str, SiteEndpoint] = {e.name: e for e in endpoints}
        self.domains = dict(domains or {})
        self._info: dict[str, SiteInfo] | None = None
        self._blooms: dict[str, dict[str, BloomFilter]] | None = None
        self._vocab: dict[str, dict[str, BloomFilter]] | None = None
        self._reported: set[tuple[str, str]] = set()
        self.live = False
        self.profile: str | None = None

    def _profiled(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        return {**kwargs, "profile": self.profile} if self.profile else kwargs

    @property
    def site_names(self) -> list[str]:
        return list(self.endpoints)

    def domain(self, site: str) -> str:
        if site in self.domains:
            return self.domains[site]
        info = (self._info or {}).get(site)
        return info.machine if info is not None else ""

    async def _know_domains(self, sites: Iterable[str], ledger: Ledger | None) -> None:
        if ledger is not None and self._info is None and any(s not in self.domains for s in sites):
            await self.info()

    def site_models(self) -> dict[str, dict[str, str]]:
        return {s: dict(si.models) for s, si in (self._info or {}).items()}

    async def _call(self, site: str, action: str, kwargs: dict[str, Any]) -> tuple[Any, float, float]:
        try:
            return await timed_call(self.endpoints[site], action, kwargs)
        except ReplyTooLarge:
            field = SPLIT_FIELDS.get(action)
            items = kwargs.get(field) if field else None
            if not items or len(items) < 2:
                raise
            half = len(items) // 2
            parts = [{**kwargs, field: items[:half]}, {**kwargs, field: items[half:]}]
            if "rid" in kwargs:
                parts = [{**p, "rid": request_id(site, action, {k: v for k, v in p.items() if k != "rid"})}
                         for p in parts]
            log.info("%s at %s: a reply over the message cap, sent again as 2 x %d", action, site, half)
            (a, da, ia), (b, db, ib) = await asyncio.gather(*(self._call(site, action, p) for p in parts))
            return ({**a, **b} if isinstance(a, dict) else [*a, *b]), max(da, db), max(ia, ib)

    async def _fanout(
        self, calls: dict[str, tuple[str, dict[str, Any]]], ledger: Ledger | None, *, tolerant: bool = False
    ) -> dict[str, Any]:
        if self.live:
            calls = {s: (a, {**kw, "rid": request_id(s, a, kw)} if a in SERVED_ACTIONS else kw)
                     for s, (a, kw) in calls.items()}

        async def timed(site: str, action: str, kwargs: dict[str, Any]) -> tuple[str, Any, float, float, int]:
            t0 = time.perf_counter()
            for attempt in range(SITE_RETRIES + 1):
                try:
                    out, dt, injected = await self._call(site, action, kwargs)
                    break
                except Exception as e:
                    if attempt == SITE_RETRIES or not retryable(action, e):
                        raise
                    self._note_retry(site, action, e, attempt)
                    await asyncio.sleep(0.5 * 4 ** attempt)
            if attempt:
                dt = time.perf_counter() - t0
            timeline.note_site(site, action, t0, dt, injected, kwargs.get("rid"))
            return site, out, dt, injected, attempt

        results = await asyncio.gather(*(timed(s, a, kw) for s, (a, kw) in calls.items()), return_exceptions=tolerant)
        if tolerant:
            for (s, (a, _)), r in zip(calls.items(), results, strict=True):
                if isinstance(r, BaseException) and not isinstance(r, Exception):
                    raise r
                if isinstance(r, Exception):
                    self._note_dropped(s, a, r)
            results = [r for r in results if not isinstance(r, BaseException)]
            answered = {r[0] for r in results}
            calls = {s: c for s, c in calls.items() if s in answered}
        if ledger is not None:
            for s in self.endpoints:
                ledger.disclose(s)
            if calls:
                ledger.charge_round(
                    {s: dt for s, _, dt, _, _ in results},
                    control={s: kw for s, (_, kw) in calls.items()},
                    reply={s: out for s, out, _, _, _ in results},
                    retries={s: n for s, _, _, _, n in results if n},
                )
                # a round waits on its slowest site, so it adds that site's injected WAN delay
                ledger.wan_seconds += max(results, key=lambda r: r[2])[3]
        return {s: out for s, out, *_ in results}

    @staticmethod
    def _disclose_probes(ledger: Ledger, site: str, requests: Iterable[ProbeRequest]) -> None:
        keys: list[str] = []
        values: list[str] = []
        question = False
        for r in requests:
            k, v = _told(r)
            keys += k
            values += v
            question |= r.kind == "question"
        ledger.disclose(site, keys, values, question)

    def _note_retry(self, site: str, action: str, e: Exception, attempt: int) -> None:
        log.warning("%s at %s failed (%s: %s); try %d of %d", action, site, type(e).__name__,
                    truncate(" ".join(str(e).split()), 300), attempt + 2, SITE_RETRIES + 1)

    def _note_dropped(self, site: str, action: str, e: Exception) -> None:
        what = f"{action}:{type(e).__name__}"
        if (site, what) not in self._reported:
            self._reported.add((site, what))
            log.warning("%s at %s failed (%s: %s): the round goes on without it (repeats are not logged)", action, site,
                        type(e).__name__, truncate(" ".join(str(e).split()), 300))

    def _note_error(self, site: str, oid: str, error: str, ledger: Ledger | None) -> None:
        if ledger is not None:
            ledger.extraction_errors += 1
        error = truncate(error, 120)
        if (site, error) not in self._reported:
            self._reported.add((site, error))
            log.warning("extraction at %s failed: %s (repeats are not logged)", site, error)

    async def info(self) -> dict[str, SiteInfo]:
        if self._info is None:
            raw = await self._fanout({s: ("info", self._profiled({})) for s in self.endpoints}, None)
            self._info = {s: SiteInfo.from_dict(d) for s, d in raw.items()}
        return self._info

    async def blooms(self) -> dict[str, dict[str, BloomFilter]]:
        if self._blooms is None:
            raw = await self._fanout({s: ("blooms", {}) for s in self.endpoints}, None)
            self._blooms = {s: {c: BloomFilter.from_dict(d) for c, d in cols.items()} for s, cols in raw.items()}
        return self._blooms

    async def vocab(self) -> dict[str, dict[str, BloomFilter]]:
        if self._vocab is None:
            raw = await self._fanout({s: ("blooms", {"vocab": True}) for s in self.endpoints}, None)
            self._vocab = {s: {c.removeprefix("vocab:"): BloomFilter.from_dict(d) for c, d in cols.items()
                               if c.startswith("vocab:")} for s, cols in raw.items()}
        return self._vocab

    async def probe(
        self,
        requests: Sequence[ProbeRequest],
        *,
        sites: Sequence[str] | None = None,
        ledger: Ledger | None = None,
        route: bool = True,
    ) -> list[list[Candidate]]:
        if not requests:
            return []
        if any(r.semijoin for r in requests):
            raise ValueError("a semijoin probe request goes through Federation.probe_counted (its reply carries the "
                             "sites' joinable counts)")
        return (await self._probe_round(requests, sites, ledger, route))[0]

    async def probe_counted(
        self,
        requests: Sequence[ProbeRequest],
        *,
        sites: Sequence[str] | None = None,
        ledger: Ledger | None = None,
        route: bool = True,
    ) -> tuple[list[list[Candidate]], list[list[ProbeCounts]]]:
        if not requests:
            return [], []
        return await self._probe_round(requests, sites, ledger, route)

    async def _probe_round(
        self, requests: Sequence[ProbeRequest], sites: Sequence[str] | None, ledger: Ledger | None, route: bool
    ) -> tuple[list[list[Candidate]], list[list[ProbeCounts]]]:
        targets = list(sites) if sites is not None else self.site_names
        routed = any(r.vocab for r in requests)
        info = await self.info() if route or routed else {}
        filters = await self.vocab() if routed else {}
        asked: dict[str, list[int]] = {}
        skipped = 0
        for s in targets:
            idx = [i for i, r in enumerate(requests) if (not route or _holds(info.get(s), r.collections))
                   and (r.sites is None or s in r.sites)]
            if routed:
                kept = [i for i in idx if not requests[i].vocab or _may_join(info.get(s), filters.get(s), requests[i])]
                skipped, idx = skipped + len(idx) - len(kept), kept
            if idx:
                asked[s] = idx
        payload = [r.to_dict() for r in requests]
        raw = await self._fanout({s: ("probe", {"requests": [payload[i] for i in idx]}) for s, idx in asked.items()},
                                 ledger)
        merged: list[list[Candidate]] = [[] for _ in requests]
        counts: list[list[ProbeCounts]] = [[] for _ in requests]
        site_ms = 0.0
        for site in targets:
            for i, reply in zip(asked.get(site, []), raw.get(site, []), strict=True):
                if isinstance(reply, dict):
                    merged[i].extend(map(Candidate.from_dict, reply.get("c", [])))
                    counts[i].extend(ProbeCounts.from_dict({**n, "site": site}) for n in reply.get("n", []))
                    site_ms += float(reply.get("ms", 0.0))
                else:
                    merged[i].extend(map(Candidate.from_dict, reply))
        if ledger is not None:
            ledger.probes += sum(len(idx) for idx in asked.values())
            ledger.reply_entries += sum(map(len, merged))
            ledger.charge_summaries([c.summary for cands in merged for c in cands])
            for cands in merged:
                for c in cands:
                    ledger.charge_to_coord(c.site, len(c.summary.encode()))
            for site, idx in asked.items():
                self._disclose_probes(ledger, site, (requests[i] for i in idx))
            ledger.charge_counted(sum(n.joinable for ns in counts for n in ns), site_ms, skipped)
            ledger.probe_rounds += 1
            for r, cands in zip(requests, merged, strict=True):
                ledger.note_returned(r, cands, ledger.probe_rounds)
        probe_trace.note_round({s: [payload[i] for i in idx] for s, idx in asked.items()}, skipped,
                               sum(map(len, merged)))
        return merged, counts

    async def hippo(self, requests: Sequence[HippoRequest], ledger: Ledger | None = None) -> list[list[Candidate]]:
        if not requests:
            return []
        info = await self.info()
        asked: dict[str, list[int]] = {}
        for s in self.site_names:
            idx = [i for i, r in enumerate(requests) if _texts(info.get(s), r.collections)]
            if idx:
                asked[s] = idx
        live = {"live": True} if getattr(self, "live", False) else {}
        payload = [{**r.to_dict(), **live} for r in requests]
        raw = await self._fanout({s: ("hippo", {"requests": [payload[i] for i in idx]}) for s, idx in asked.items()},
                                 ledger)
        merged: list[list[Candidate]] = [[] for _ in requests]
        used = Usage()
        for site in self.site_names:
            for i, reply in zip(asked.get(site, []), raw.get(site, []), strict=True):
                if isinstance(reply, dict):
                    used.add(Usage.from_dict(reply.get("u") or {}))
                    reply = reply.get("c") or []
                merged[i].extend(map(Candidate.from_dict, reply))
        if ledger is not None:
            ledger.count_fired("hippo_requests", sum(len(idx) for idx in asked.values()))
            if used.calls:
                ledger.extraction.setdefault("hippo_filter", Usage()).add(used)
            ledger.charge_summaries([c.summary for cands in merged for c in cands])
            for cands in merged:
                for c in cands:
                    ledger.charge_to_coord(c.site, len(c.summary.encode()))
            for site, idx in asked.items():
                ledger.disclose(site, [requests[i].question for i in idx], question=True)
            ledger.probe_rounds += 1
            for r, cands in zip(requests, merged, strict=True):
                ledger.note_returned(r, cands, ledger.probe_rounds)
        return merged

    async def probe_each(
        self, requests_by_site: dict[str, Sequence[ProbeRequest]], *, ledger: Ledger | None = None
    ) -> dict[str, list[list[Candidate]]]:
        calls = {s: ("probe", {"requests": [r.to_dict() for r in reqs]}) for s, reqs in requests_by_site.items() if reqs}
        raw = await self._fanout(calls, ledger)
        out = {s: [list(map(Candidate.from_dict, cands)) for cands in raw[s]] for s in calls}
        if ledger is not None:
            ledger.probes += sum(len(r) for r in requests_by_site.values())
            ledger.reply_entries += sum(len(cands) for lists in out.values() for cands in lists)
            ledger.charge_summaries([c.summary for lists in out.values() for cands in lists for c in cands])
            for site in calls:
                self._disclose_probes(ledger, site, requests_by_site[site])
            ledger.probe_rounds += 1
            for site, lists in out.items():
                for r, cands in zip(requests_by_site[site], lists, strict=False):
                    ledger.note_returned(r, cands, ledger.probe_rounds)
        probe_trace.note_round({s: kw["requests"] for s, (_, kw) in calls.items()}, 0,
                               sum(len(cands) for lists in out.values() for cands in lists))
        return out

    async def extract(
        self,
        refs: Sequence[Ref],
        *,
        include_raw: bool = False,
        mode: ExtractMode = "kg",
        query: str | None = None,
        blocks: Sequence[Sequence[Ref]] | None = None,
        ledger: Ledger | None = None,
    ) -> list[ExtractionResult]:
        every = [*refs, *(ref for block in blocks or () for ref in block)]
        by_site: dict[str, list[str]] = defaultdict(list)
        for site, oid in dict.fromkeys(every):
            by_site[site].append(oid)
        kwargs: dict[str, Any] = {"include_raw": include_raw, "mode": mode, "query": query}
        if getattr(self, "live", False):
            kwargs["live"] = True
        kwargs = self._profiled(kwargs)
        site_blocks: dict[str, list[list[str]]] = defaultdict(list)
        for block in blocks or ():
            parts: dict[str, list[str]] = defaultdict(list)
            for site, oid in dict.fromkeys(block):
                parts[site].append(oid)
            for site, oids in parts.items():
                site_blocks[site].append(oids)
        calls = {s: ("extract", {"oids": oids, **kwargs, **({"blocks": site_blocks[s]} if blocks is not None else {})})
                 for s, oids in by_site.items()}
        await self._know_domains(by_site, ledger)
        raw = await self._fanout(calls, ledger)
        out: dict[Ref, ExtractionResult] = {}
        for site, results in raw.items():
            if ledger is not None:
                ledger.disclose(site, question=mode != "kg" and bool(query))
            for d in results:
                r = ExtractionResult.from_dict(d)
                out[(site, r.oid)] = r
                if r.error is not None:
                    self._note_error(site, r.oid, r.error, ledger)
                elif ledger is not None:
                    ledger.charge_extraction(r, self.domain(site))
                    ledger.note_read([r.doc_id] if r.doc_id else [], "site")
        return [out[ref] for ref in dict.fromkeys(every) if ref in out]

    async def read(
        self,
        requests: dict[str, dict[str, Any]],
        question: str,
        unknowns: Sequence[dict[str, Any]],
        *,
        include_raw: bool = False,
        ledger: Ledger | None = None,
    ) -> dict[str, ReadResult]:
        kwargs = {"question": question, "unknowns": list(unknowns), "include_raw": include_raw}
        if self.live:
            kwargs["live"] = True
        kwargs = self._profiled(kwargs)
        calls = {s: ("read", {**kwargs, **req}) for s, req in requests.items() if any(req.get("blocks") or [])}
        await self._know_domains(calls, ledger)
        raw = await self._fanout(calls, ledger)
        if ledger is not None:
            for site, (_, kw) in calls.items():
                ledger.disclose(site, values=_read_values(kw), question=True)
        out: dict[str, ReadResult] = {}
        for site, d in raw.items():
            r = ReadResult.from_dict(d)
            out[site] = r
            if r.error is not None:
                self._note_error(site, ",".join(r.read), r.error, ledger)
                if ledger is not None and r.usage.calls:
                    for mod, u in (r.usage_by_modality or {"text": r.usage}).items():
                        ledger.extraction.setdefault(f"read_{mod}", Usage()).add(u)
                    ledger.reads += r.usage.calls
            elif r.extractions:
                for x in r.extractions:
                    if x.error is not None:
                        self._note_error(site, x.oid, x.error, ledger)
                    elif ledger is not None:
                        ledger.charge_extraction(x, self.domain(site))
            elif ledger is not None:
                ledger.charge_read(r, self.domain(site))
            if ledger is not None and r.error is None:
                ledger.note_read(r.doc_ids.values(), "site")
        return out

    async def fetch_raw(self, refs: Sequence[Ref], *, max_px: int = 384, quality: int = 75,
                        ledger: Ledger | None = None, epoch: str | None = None) -> list[RawObject]:
        by_site: dict[str, list[str]] = defaultdict(list)
        for site, oid in dict.fromkeys(refs):
            by_site[site].append(oid)
        extra = {k: v for k, v, default in (("max_px", max_px, 384), ("quality", quality, 75)) if v != default}
        if epoch is not None:
            extra["epoch"] = epoch
        objs: dict[Ref, RawObject] = {}
        pending = dict(by_site)
        # a site stops at its byte budget; the rest of its oids go in the next round
        while pending:
            raw = await self._fanout({s: ("fetch_raw", {"oids": oids, **extra}) for s, oids in pending.items()},
                                     ledger)
            more: dict[str, list[str]] = {}
            for s, ds in raw.items():
                got = [RawObject.from_dict(d) for d in ds]
                objs.update({(s, o.oid): o for o in got})
                if got and got[-1].oid in pending[s]:
                    rest = pending[s][pending[s].index(got[-1].oid) + 1:]
                    if rest:
                        more[s] = rest
            pending = more
        if ledger is not None:
            ledger.charge_raw(sum(o.nbytes for o in objs.values()))
        return [objs[ref] for ref in dict.fromkeys(refs) if ref in objs]

    async def synopsis(self, k: int = 16) -> dict[str, dict[str, Any]]:
        cache = getattr(self, "_synopses", None)
        if cache is None:
            cache = self._synopses = {}
        if k not in cache:
            raw = await self._fanout({s: ("synopsis", {"k": k}) for s in self.site_names}, None)
            cache[k] = {s: v for s, v in raw.items()}
        return cache[k]

    async def catalog(self, collections: Sequence[str] | None = None, sites: Sequence[str] | None = None,
                      page: int = 20_000) -> dict[str, dict[str, list[list[str]]]]:
        from fedquest.errors import ReplyTooLarge

        names = list(sites) if sites is not None else self.site_names
        out: dict[str, dict[str, list[list[str]]]] = {s: {} for s in names}
        offset = 0
        pending = set(names)
        while pending:
            kw = {"offset": offset, "limit": page, **({"collections": list(collections)} if collections else {})}
            try:
                raw = await self._fanout({s: ("catalog", kw) for s in sorted(pending)}, None)
            except ReplyTooLarge:
                if page <= 1000:
                    raise
                page //= 2
                continue
            done = set()
            for s, cols in raw.items():
                full = False
                for c, d in cols.items():
                    out[s].setdefault(c, []).extend(d["oids"])
                    full = full or offset + page < d["n"]
                if not full:
                    done.add(s)
            pending -= done | (pending - set(raw))
            offset += page
        return out

    async def stats(self, busy_only: bool = False) -> dict[str, Any]:
        return await self._fanout({s: ("stats", self._profiled({"busy_only": True} if busy_only else {}))
                                   for s in self.endpoints}, None)

    async def versions(self, refs: Sequence[Ref], epoch: str | None = None, *,
                       ledger: Ledger | None = None) -> dict[Ref, str]:
        by_site: dict[str, list[str]] = defaultdict(list)
        for site, oid in dict.fromkeys(refs):
            if site in self.endpoints:
                by_site[site].append(oid)
        extra = {"epoch": epoch} if epoch is not None else {}
        raw = await self._fanout({s: ("versions", {"oids": oids, **extra}) for s, oids in by_site.items()}, ledger,
                                 tolerant=True)
        return {(s, str(oid)): str(v) for s, d in raw.items() for oid, v in (d or {}).items() if v is not None}

    async def witnesses(self, items: Sequence[tuple[str, str, str, str]], epoch: str | None = None, *,
                        ledger: Ledger | None = None) -> dict[tuple[str, str, str, str], bool]:
        by_site: dict[str, list[tuple[str, str, str, str]]] = defaultdict(list)
        for it in dict.fromkeys(items):
            if it[0] in self.endpoints and it[2]:
                by_site[it[0]].append(it)
        extra = {"epoch": epoch} if epoch is not None else {}
        calls = {s: ("witnesses", {"items": [{"oid": oid, "w": w, **({"known": k} if k else {})}
                                             for _, oid, w, k in its], **extra}) for s, its in by_site.items()}
        raw = await self._fanout(calls, ledger, tolerant=True)
        return {it: bool(ok) for s, oks in raw.items() for it, ok in zip(by_site[s], oks or [], strict=False)}

    async def update(self, epoch: str, edits_by_site: dict[str, Sequence[dict[str, Any]]]) -> dict[Ref, str]:
        unknown = sorted(set(edits_by_site) - set(self.endpoints))
        if unknown:
            raise KeyError(f"update: no sites {unknown} in this federation (it has {self.site_names})")
        calls = {s: ("update", {"epoch": epoch, "edits": [dict(e) for e in edits]})
                 for s, edits in edits_by_site.items()}
        raw = await self._fanout(calls, None)
        return {(s, str(oid)): str(v) for s, d in raw.items() for oid, v in (d or {}).items() if v is not None}

    async def close(self) -> None:
        await asyncio.gather(*(e.close() for e in self.endpoints.values()))


def request_id(site: str, action: str, kwargs: dict[str, Any]) -> str:
    from fedquest.llm.cache import canonical_json
    from fedquest.utils import stable_hash

    return stable_hash(site, action, canonical_json(kwargs), n=24)


def _read_values(kwargs: dict[str, Any]) -> list[str]:
    unknowns = kwargs.get("unknowns") or []
    return [*(kwargs.get("prefilter") or []), *(kwargs.get("labels") or {}).values(),
            *(v for u in unknowns for v in u.get("values") or [] if v),
            *(c for u in unknowns for c in u.get("candidates") or [])]


def profiled_stats(stats: dict[str, Any], profiled: dict[str, Any] | None, profile: str) -> dict[str, Any]:
    out = {**stats, "profile": profile}
    mine, theirs = stats.get("busy"), (profiled or {}).get("busy")
    if isinstance(mine, dict) and isinstance(theirs, dict):
        busy = {}
        for k in dict.fromkeys([*mine, *theirs]):
            total = mine.get(k, 0) + theirs.get(k, 0)
            busy[k] = round(total, 4) if isinstance(total, float) else total
        out["busy"] = busy
    return out


def expected_site_models(cfg: FederationCfg, models: ModelsCfg,
                         profile: str | None = None) -> dict[str, dict[str, str]]:
    sites = getattr(cfg, "sites", None) or []
    if profile:
        merged = models.profile(profile) if profile in models.profiles else models
        return {s.name: {r: merged.roles[r] for r in SITE_ROLES} for s in sites}
    return {s.name: {r: models.for_site(s).roles[r] for r in SITE_ROLES} for s in sites}


async def require_current(fed: Federation, cfg: FederationCfg, *,
                          expected_models: dict[str, dict[str, str]] | None = None) -> None:
    expected = (cfg.extractor_version, READ_VERSION, WIRE_VERSION)
    info = await fed.info()
    stale = {s: (si.extractor_version, si.reader_version, si.wire) for s, si in info.items()
             if (si.extractor_version, si.reader_version, si.wire) != expected}
    if stale:
        raise RuntimeError(f"{cfg.name}: sites {stale} run other code than (extract, read, wire) = {expected!r}: "
                           "git pull on their machines, restart their endpoints, then bash scripts/remote.sh up")
    wrong = {}
    for s, want in (expected_models or {}).items():
        have = info[s].models if s in info else {}
        diff = {r: (have.get(r), m) for r, m in want.items() if r in SITE_ROLES and have.get(r) != m}
        if diff:
            wrong[s] = diff
    if wrong:
        profile = getattr(fed, "profile", None)
        under = (f" under models profile {profile!r} (the site servers' models config must define it as "
                 "configs/models/final.yaml does)") if profile else ""
        raise RuntimeError(f"{cfg.name}: sites run other models{under} than this coordinator expects, role: (site's, "
                           f"expected) = {wrong}: restart the site servers with configs/remote.yaml's models "
                           "(bash scripts/remote.sh down, then up), or run the coordinator with theirs")
