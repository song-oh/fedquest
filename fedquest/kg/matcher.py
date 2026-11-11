from __future__ import annotations

import itertools
import threading
from collections import Counter, OrderedDict, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from fedquest.kg.normalize import content_stems, norm_entity, same_norm
from fedquest.types import Edge, Fact, Pattern, is_var

MAX_VALUES = 5
MAX_PATHS = 4000
LEXICAL_SCORE = 0.65


class _Encoder(Protocol):
    def encode_docs(self, texts: list[str], batch_size: int = 64) -> np.ndarray: ...


class TextSim:
    def __init__(self, encoder: _Encoder, cache_size: int = 100_000) -> None:
        self.encoder = encoder
        self.cache_size = cache_size
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._queries: OrderedDict[str, np.ndarray] = OrderedDict()
        self._lock = threading.Lock()

    def _lookup(self, cache: OrderedDict[str, np.ndarray], texts: Sequence[str], encode: Any) -> np.ndarray:
        with self._lock:
            known = {t: cache[t] for t in dict.fromkeys(texts) if t in cache}
        missing = [t for t in dict.fromkeys(texts) if t not in known]
        if missing:
            for t, v in zip(missing, np.asarray(encode(missing), dtype=np.float32), strict=True):
                n = float(np.linalg.norm(v))
                known[t] = v / n if n > 0 else v
        with self._lock:
            for t in dict.fromkeys(texts):
                cache[t] = known[t]
                cache.move_to_end(t)
            while len(cache) > self.cache_size:
                cache.popitem(last=False)
        out = [known[t] for t in texts]
        return np.stack(out) if out else np.zeros((0, 1), dtype=np.float32)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        return self._lookup(self._cache, texts, self.encoder.encode_docs)

    def sim(self, a: str, b: str) -> float:
        va, vb = self.embed([a, b])
        return float(va @ vb)

    def sims(self, a: str, bs: Sequence[str]) -> np.ndarray:
        vecs = self.embed([a, *bs])
        return vecs[1:] @ vecs[0]

    def relevance(self, pairs: Sequence[tuple[str, Sequence[str]]]) -> list[np.ndarray]:
        if not pairs:
            return []
        encode = getattr(self.encoder, "encode_queries", self.encoder.encode_docs)
        keys = self._lookup(self._queries, [k for k, _ in pairs], encode)
        docs = self.embed([d for _, ds in pairs for d in ds])
        out, start = [], 0
        for i, (_, ds) in enumerate(pairs):
            out.append(docs[start : start + len(ds)] @ keys[i] if ds else np.zeros(0, dtype=np.float32))
            start += len(ds)
        return out


@dataclass
class EdgeSupport:
    edge_id: int
    head: str
    tail: str
    facts: list[Fact]
    score: float


@dataclass
class MatchResult:
    complete: bool
    bindings: dict[str, str] = field(default_factory=dict)
    support: dict[int, EdgeSupport] = field(default_factory=dict)
    values: dict[str, list[tuple[str, float]]] = field(default_factory=dict)
    supported: set[int] = field(default_factory=set)
    score: float = 0.0

    def support_facts(self) -> list[Fact]:
        return list(dict.fromkeys(f for s in self.support.values() for f in s.facts))


class _Graph:
    def __init__(self, facts: Sequence[Fact]) -> None:
        self.adj: dict[str, list[tuple[Fact, str]]] = defaultdict(list)
        names: dict[str, Counter[str]] = defaultdict(Counter)
        for f in dict.fromkeys(facts):
            h, t = norm_entity(f.head), norm_entity(f.tail)
            if not h or not t or h == t:
                continue
            names[h][f.head] += 1
            names[t][f.tail] += 1
            self.adj[h].append((f, t))
            self.adj[t].append((f, h))
        self.display = {n: c.most_common(1)[0][0] for n, c in names.items()}
        self.nodes = list(self.display)


class Matcher:
    def __init__(self, sim: TextSim, theta: float = 0.6, max_hops: int = 3, node_theta: float = 0.85) -> None:
        self.sim = sim
        self.theta = theta
        self.max_hops = max_hops
        self.node_theta = node_theta

    def _align(self, g: _Graph, value: str, cache: dict[str, list[str]]) -> list[str]:
        if value in cache:
            return cache[value]
        nv = norm_entity(value)
        hits = [n for n in g.nodes if same_norm(nv, n)]
        if not hits and g.nodes and nv:
            sims = self.sim.sims(value, [g.display[n] for n in g.nodes])
            hits = [n for n, s in zip(g.nodes, sims, strict=True) if s > self.node_theta]
        cache[value] = hits
        return hits

    def _path_score(self, relation: str, path: Sequence[Fact]) -> float:
        rels = [f.relation for f in path]
        vecs = self.sim.embed([relation, *rels])
        mean = vecs[1:].mean(axis=0)
        n = float(np.linalg.norm(mean))
        score = float(vecs[0] @ mean / n) if n > 0 else 0.0
        if len(path) == 1 and content_stems(relation) & content_stems(rels[0]):
            score = max(score, LEXICAL_SCORE)
        return score

    def _paths(self, g: _Graph, start: str, max_hops: int) -> list[tuple[str, list[Fact]]]:
        out: list[tuple[str, list[Fact]]] = []
        stack: list[tuple[str, list[Fact], frozenset[str]]] = [(start, [], frozenset([start]))]
        while stack and len(out) < MAX_PATHS:
            node, path, seen = stack.pop()
            for f, nxt in g.adj.get(node, ()):
                if nxt in seen:
                    continue
                p = [*path, f]
                out.append((nxt, p))
                if len(p) < max_hops:
                    stack.append((nxt, p, seen | {nxt}))
        return out

    def _supports(
        self,
        g: _Graph,
        edge: Edge,
        heads: Sequence[str] | None,
        tails: Sequence[str] | None,
        align_cache: dict[str, list[str]],
    ) -> list[EdgeSupport]:
        if heads is None and tails is None:
            return []
        if heads is None:
            starts_from, other, flip = tails, None, True
        else:
            starts_from, other, flip = heads, tails, False
        targets: set[str] | None = None
        if other is not None:
            targets = {n for v in other for n in self._align(g, v, align_cache)}
            if not targets:
                return []
        best: dict[tuple[str, str], EdgeSupport] = {}
        for value in starts_from:
            for start in self._align(g, value, align_cache):
                for end, path in self._paths(g, start, self.max_hops):
                    if targets is not None and end not in targets:
                        continue
                    score = self._path_score(edge.relation, path)
                    if score <= self.theta:
                        continue
                    h, t = (end, start) if flip else (start, end)
                    key = (h, t)
                    if key not in best or score > best[key].score:
                        best[key] = EdgeSupport(edge.id, g.display[h], g.display[t], list(path), score)
        return sorted(best.values(), key=lambda s: -s.score)

    def edge_support(
        self,
        edge: Edge,
        facts: Sequence[Fact],
        head_values: Sequence[str] | None,
        tail_values: Sequence[str] | None,
    ) -> list[EdgeSupport]:
        if not is_var(edge.head):
            head_values = [edge.head]
        if not is_var(edge.tail):
            tail_values = [edge.tail]
        return self._supports(_Graph(facts), edge, head_values, tail_values, {})

    def match(self, pattern: Pattern, facts: Sequence[Fact]) -> MatchResult:
        if not pattern.edges or not facts:
            return MatchResult(complete=False)
        g = _Graph(facts)
        cache: dict[str, list[str]] = {}
        values: dict[str, dict[str, float]] = {v: {} for v in pattern.variables}

        def side(node: str) -> list[str] | None:
            if not is_var(node):
                return [node]
            vals = values.get(node) or {}
            return [v for v, _ in sorted(vals.items(), key=lambda kv: -kv[1])[:MAX_VALUES]] or None

        supported: set[int] = set()
        for _ in range(len(pattern.edges) + 1):
            changed = False
            for e in pattern.edges:
                heads, tails = side(e.head), side(e.tail)
                for s in self._supports(g, e, heads, tails, cache):
                    supported.add(e.id)
                    for node, value in ((e.head, s.head), (e.tail, s.tail)):
                        if is_var(node) and s.score > values[node].get(value, 0.0):
                            values[node][value] = s.score
                            changed = True
            if not changed:
                break

        ranked = {v: [x for x, _ in sorted(d.items(), key=lambda kv: -kv[1])[:MAX_VALUES]] for v, d in values.items()}
        need = {e.id for e in pattern.edges}
        best: tuple[int, float, dict[str, str], dict[int, EdgeSupport]] = (-1, 0.0, {}, {})
        pair_cache: dict[tuple[int, str, str], EdgeSupport | None] = {}
        domains = [ranked[v] or [None] for v in pattern.variables]
        for combo in itertools.islice(itertools.product(*domains), 3125):
            binding = {v: x for v, x in zip(pattern.variables, combo, strict=True) if x is not None}
            sup: dict[int, EdgeSupport] = {}
            for e in pattern.edges:
                h = binding.get(e.head) if is_var(e.head) else e.head
                t = binding.get(e.tail) if is_var(e.tail) else e.tail
                if h is None or t is None:
                    continue
                key = (e.id, h, t)
                if key not in pair_cache:
                    found = self._supports(g, e, [h], [t], cache)
                    pair_cache[key] = found[0] if found else None
                if pair_cache[key] is not None:
                    sup[e.id] = pair_cache[key]  # type: ignore[assignment]
            score = sum(s.score for s in sup.values()) / max(1, len(pattern.edges))
            if (len(need & set(sup)), score) > (best[0], best[1]):
                best = (len(need & set(sup)), score, binding, sup)
        n_sup, score, binding, sup = best
        complete = n_sup == len(need) and all(v in binding for v in pattern.variables)
        return MatchResult(
            complete=complete,
            bindings=binding,
            support=sup,
            values={v: sorted(d.items(), key=lambda kv: -kv[1])[:MAX_VALUES] for v, d in values.items()},
            supported=supported,
            score=score,
        )

    @staticmethod
    def frontier(pattern: Pattern, result: MatchResult, hyps: int = 5,
                 edges: Sequence[Edge] | None = None) -> list[dict[str, Any]]:
        def values(node: str) -> list[tuple[str, float]]:
            if not is_var(node):
                return [(node, 1.0)]
            cands = list(result.values.get(node, []))
            bound = result.bindings.get(node)
            if bound is not None:
                score = next((s for v, s in cands if v == bound), 1.0)
                cands = [(bound, score), *((v, s) for v, s in cands if v != bound)]
            return cands[:hyps]

        def known(node: str) -> bool:
            return not is_var(node) or node in result.bindings

        out: list[dict[str, Any]] = []
        for e in pattern.edges if edges is None else edges:
            if e.id in result.support:
                continue
            if e.head == e.tail or (known(e.head) and known(e.tail)):
                var = e.tail if is_var(e.tail) else e.head if is_var(e.head) else None
                if var is None:
                    continue
                other = values(e.head if var == e.tail else e.tail)
                for v, w in values(var):
                    out.append({"kind": "check", "edge": e.id, "relation": e.relation, "var": var, "value": v,
                                "value_is_head": var == e.head, "weight": float(w),
                                "other": other[0][0] if other else v})
                continue
            for end, var in ((e.head, e.tail), (e.tail, e.head)):
                if is_var(var) and var not in result.bindings:
                    out += [{"kind": "find", "edge": e.id, "relation": e.relation, "var": var, "value": v,
                             "value_is_head": end == e.head, "weight": float(w)} for v, w in values(end)]
        return out


def summarize(result: MatchResult) -> dict[str, Any]:
    return {
        "complete": result.complete,
        "bindings": result.bindings,
        "supported": sorted(result.supported),
        "score": round(result.score, 4),
    }
