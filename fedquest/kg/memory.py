from __future__ import annotations

import copy
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from fedquest.kg.normalize import norm_entity
from fedquest.kg.view import AskRecord, Ref, default_writer, harvested, materializable
from fedquest.types import Candidate, ExtractionResult, Fact


class _Namespace:
    def __init__(self) -> None:
        self.coverage: dict[Ref, tuple[str, list[Fact]]] = {}
        self.index: dict[str, list[Fact]] = defaultdict(list)
        self.asks: dict[str, dict[tuple[str, bool], AskRecord]] = defaultdict(dict)
        self.entries: dict[Ref, Candidate] = {}
        self.probes: dict[str, list[Candidate]] = {}
        self.aliases: dict[str, str] = {}
        self.index_stats: dict[str, list[float]] = {}
        self.by_object: dict[Ref, dict[Fact, None]] = defaultdict(dict)
        self.version: dict[Ref, str] = {}
        self.dead: dict[Ref, set[str]] = defaultdict(set)
        self.writer: dict[Fact, str] = {}
        self.qid: dict[Fact, str] = {}
        self.tuples: dict[str, str] = {}

    def is_stale(self, f: Fact) -> bool:
        return bool(f.overs) and f.overs in self.dead.get((f.site, f.oid), ())


class MemoryKG:
    def __init__(self, writer: str | None = None) -> None:
        self._ns: dict[str, _Namespace] = {}
        self._writer = writer

    @property
    def writer(self) -> str:
        return self._writer or default_writer()

    @writer.setter
    def writer(self, value: str | None) -> None:
        self._writer = value

    def _get(self, ns: str) -> _Namespace:
        return self._ns.setdefault(ns, _Namespace())

    async def covered(self, ns: str, refs: Sequence[Ref], version: str) -> dict[Ref, list[Fact]]:
        space = self._ns.get(ns)
        if space is None:
            return {}
        out = {}
        for ref in refs:
            hit = space.coverage.get(ref)
            if hit is not None and hit[0] == version:
                out[ref] = list(hit[1])
        return out

    async def lookup(self, ns: str, entities: Sequence[str], limit: int = 200, *,
                     include_stale: bool = False) -> list[Fact]:
        space = self._ns.get(ns)
        if space is None:
            return []
        out: dict[Fact, None] = {}
        for e in entities:
            for f in space.index.get(norm_entity(e), ()):
                if not include_stale and space.is_stale(f):
                    continue
                out[f] = None
                if len(out) >= limit:
                    return list(out)
        return list(out)

    async def add(self, ns: str, results: Sequence[ExtractionResult]) -> None:
        space = self._get(ns)
        writer = self.writer
        for r in results:
            if not materializable(r):
                continue
            ref = (r.site, r.oid)
            old = space.coverage.get(ref)
            facts = list(dict.fromkeys([*(old[1] if old else []), *r.facts]))
            space.coverage[ref] = (r.version, facts)
            for f in r.facts:
                for name in {norm_entity(f.head), norm_entity(f.tail)}:
                    if f not in space.index[name]:
                        space.index[name].append(f)
                src = (f.site, f.oid)
                space.by_object[src][f] = None
                space.writer.setdefault(f, writer)
                space.qid.setdefault(f, r.qid)
                if f.overs and f.overs not in space.dead.get(src, ()):
                    space.version[src] = f.overs

    async def stale(self, ns: str, refs_versions: Mapping[Ref, str]) -> int:
        todo = {ref: v for ref, v in refs_versions.items() if v}
        if not todo:
            return 0
        space = self._get(ns)
        n = 0
        for ref, v in todo.items():
            facts = space.by_object.get(ref, {})
            newly = ({f.overs for f in facts} | {space.version.get(ref, "")}) - {"", v} - space.dead.get(ref, set())
            n += sum(f.overs in newly for f in facts)
            space.dead[ref] |= newly
            space.version[ref] = v
        return n

    async def versions(self, ns: str, refs: Sequence[Ref]) -> dict[Ref, str]:
        space = self._ns.get(ns)
        if space is None:
            return {}
        return {ref: space.version[ref] for ref in dict.fromkeys(refs) if ref in space.version}

    async def writers(self, ns: str, facts: Sequence[Fact]) -> dict[Fact, str]:
        space = self._ns.get(ns)
        if space is None:
            return {}
        return {f: space.writer[f] for f in dict.fromkeys(facts) if f in space.writer}

    async def qids(self, ns: str, facts: Sequence[Fact]) -> dict[Fact, str]:
        return {f: p["qid"] for f, p in (await self.provenance(ns, facts)).items()}

    async def provenance(self, ns: str, facts: Sequence[Fact]) -> dict[Fact, dict[str, Any]]:
        space = self._ns.get(ns)
        if space is None:
            return {}
        return {f: {"writer": space.writer[f], "qid": space.qid.get(f, ""), "stale": space.is_stale(f)}
                for f in dict.fromkeys(facts) if f in space.writer}

    async def fact_stats(self, ns: str) -> dict[str, Any]:
        space = self._ns.get(ns)
        facts = [] if space is None else await self.facts(ns)
        return {
            "facts": len(facts),
            "stale": sum(space.is_stale(f) for f in facts) if space else 0,
            "entities": len({norm_entity(x) for f in facts for x in (f.head, f.tail)}),
            "harvested": sum(harvested(f) for f in facts),
            "questions": len({space.qid.get(f, "") for f in facts} - {""}) if space else 0,
            "by_writer": dict(Counter(space.writer[f] for f in facts)) if space else {},
            "tuples": len(space.tuples) if space else 0,
        }

    async def aliases(self, ns: str, names: Sequence[str]) -> dict[str, str]:
        space = self._ns.get(ns)
        if space is None:
            return {}
        return {k: space.aliases[k] for k in dict.fromkeys(norm_entity(n) for n in names) if k in space.aliases}

    async def add_aliases(self, ns: str, links: Mapping[str, str]) -> None:
        space = self._get(ns)
        for name, canonical in links.items():
            key = norm_entity(name)
            if key and key != norm_entity(canonical):
                space.aliases.setdefault(key, canonical)

    async def known(self, ns: str, names: Sequence[str]) -> set[str]:
        space = self._ns.get(ns)
        if space is None:
            return set()
        canon = {norm_entity(c) for c in space.aliases.values()}
        return {k for k in dict.fromkeys(norm_entity(n) for n in names)
                if k and (space.index.get(k) or any(r.witnesses for r in space.asks.get(k, {}).values())
                          or k in canon)}

    async def facts(self, ns: str) -> list[Fact]:
        space = self._ns.get(ns)
        if space is None:
            return []
        return list(dict.fromkeys(f for _, facts in space.coverage.values() for f in facts))

    async def stats(self, ns: str | None = None) -> dict[str, Any]:
        spaces = [self._ns[ns]] if ns is not None and ns in self._ns else ([] if ns else list(self._ns.values()))
        facts = {f for s in spaces for _, fs in s.coverage.values() for f in fs}
        sites_by_entity: dict[str, set[str]] = defaultdict(set)
        for f in facts:
            for name in (norm_entity(f.head), norm_entity(f.tail)):
                sites_by_entity[name].add(f.site)
        per_site: dict[str, int] = defaultdict(int)
        for f in facts:
            per_site[f.site] += 1
        tuples = sum(len(s.tuples) for s in spaces)
        return {
            "namespaces": len(spaces),
            "objects": sum(len(s.coverage) for s in spaces),
            "facts": len(facts),
            "entities": len(sites_by_entity),
            "facts_per_site": dict(per_site),
            "cross_site_entities": sum(len(s) >= 2 for s in sites_by_entity.values()),
            "asks": sum(len(a) for s in spaces for a in s.asks.values()),
            "witnesses": sum(len(r.witnesses) for s in spaces for a in s.asks.values() for r in a.values()),
            "negatives": sum(len(r.negatives) for s in spaces for a in s.asks.values() for r in a.values()),
            "aliases": sum(len(s.aliases) for s in spaces),
            "probes": sum(len(s.probes) for s in spaces),
            **({"tuples": tuples} if tuples else {}),
        }

    async def clear(self, ns: str) -> None:
        for key in [k for k in self._ns if k == ns or k.startswith(ns + "/")]:
            del self._ns[key]

    async def close(self) -> None:
        return None

    async def asks(self, ns: str, values: Sequence[str]) -> list[AskRecord]:
        space = self._ns.get(ns)
        if space is None:
            return []
        return [copy.deepcopy(rec) for v in dict.fromkeys(norm_entity(x) for x in values)
                for rec in space.asks.get(v, {}).values()]

    async def nodes(self, ns: str) -> list[str]:
        space = self._ns.get(ns)
        return [] if space is None else [v for v, held in space.asks.items() if held]

    async def index_stats(self, ns: str) -> dict[str, tuple[int, int, float]]:
        space = self._ns.get(ns)
        return {} if space is None else {k: (int(v[0]), int(v[1]), float(v[2])) for k, v in space.index_stats.items()}

    async def add_index_stats(self, ns: str, deltas: Mapping[str, tuple[int, int] | tuple[int, int, float]]) -> None:
        space = self._get(ns)
        for cls, delta in deltas.items():
            n, w, e = (*delta, 0.0)[:3]
            if not (n or w):
                continue
            held = space.index_stats.setdefault(cls, [0, 0, 0.0])
            held[0] += n
            held[1] += w
            held[2] += float(e)

    async def add_asks(self, ns: str, records: Sequence[AskRecord], entries: Sequence[Candidate]) -> None:
        space = self._get(ns)
        for rec in records:
            held = space.asks[rec.value]
            old = held.get((rec.relation, rec.head))
            held[(rec.relation, rec.head)] = copy.deepcopy(rec) if old is None else old.merged(rec)
        for c in entries:
            space.entries.setdefault(c.ref, copy.copy(c))

    async def entries(self, ns: str, refs: Sequence[Ref]) -> dict[Ref, Candidate]:
        space = self._ns.get(ns)
        if space is None:
            return {}
        return {ref: copy.copy(space.entries[ref]) for ref in refs if ref in space.entries}

    async def probes(self, ns: str, keys: Sequence[str]) -> dict[str, list[Candidate]]:
        space = self._ns.get(ns)
        if space is None:
            return {}
        return {k: [copy.copy(c) for c in space.probes[k]] for k in keys if k in space.probes}

    async def add_probes(self, ns: str, results: dict[str, list[Candidate]]) -> None:
        space = self._get(ns)
        for k, cands in results.items():
            space.probes.setdefault(k, [copy.copy(c) for c in cands])

    async def tuples(self, ns: str, keys: Sequence[str]) -> dict[str, dict[str, Any]]:
        space = self._ns.get(ns)
        if space is None:
            return {}
        return {k: json.loads(space.tuples[k]) for k in dict.fromkeys(keys) if k in space.tuples}

    async def add_tuples(self, ns: str, items: Mapping[str, dict[str, Any]]) -> None:
        texts = {k: json.dumps(payload, ensure_ascii=False) for k, payload in items.items()}
        space = self._get(ns)
        for k, text in texts.items():
            space.tuples.setdefault(k, text)
