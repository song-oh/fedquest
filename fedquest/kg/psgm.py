# PSGM matcher: query nodes onto KG entities, edges onto embedding-guided paths
from __future__ import annotations

import logging
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from fedquest.kg.matcher import MAX_VALUES, EdgeSupport, Matcher, MatchResult, TextSim
from fedquest.kg.normalize import norm_entity
from fedquest.types import Edge, Fact, Pattern, is_var

log = logging.getLogger(__name__)

ANCHOR_PAIRS = 10
MAX_MAPS = 2000
PASSAGE_RELATION = "consists_in"
_PASSAGE = "\x00passage:"


class _KG:
    def __init__(self, facts: Sequence[Fact], entities: Iterable[Sequence[str]] | None,
                 passages: Mapping[str, Sequence[str]] | None) -> None:
        names: dict[str, Counter[str]] = defaultdict(Counter)
        self.out: dict[str, list[tuple[Fact, str]]] = defaultdict(list)
        for f in dict.fromkeys(facts):
            h, t = norm_entity(f.head), norm_entity(f.tail)
            if not h or not t or h == t:
                continue
            names[h][f.head] += 1
            names[t][f.tail] += 1
            self.out[h].append((f, t))
        description: dict[str, str] = {}
        for rec in entities or ():
            name = str(rec[0]).strip() if len(rec) else ""
            key = norm_entity(name)
            if not key:
                continue
            names[key][name] += 1
            if len(rec) > 1 and str(rec[1]).strip():
                description.setdefault(key, str(rec[1]).strip())
        self.display = {k: c.most_common(1)[0][0] for k, c in names.items()}
        self.entities = list(self.display)
        self.text = {k: description.get(k) or self.display[k] for k in self.entities}
        for name, members in (passages or {}).items():
            targets = [k for k in dict.fromkeys(norm_entity(m) for m in members) if k in self.text]
            if targets:
                node = _PASSAGE + name
                self.display[node] = name
                self.out[node] = [(Fact(name, PASSAGE_RELATION, self.display[k], version="psgm"), k) for k in targets]


class PSGMatcher:
    def __init__(self, sim: TextSim, theta: float = 0.6, lp: int = 3, anchor_pairs: int = ANCHOR_PAIRS,
                 unary: str = "loop", max_maps: int = MAX_MAPS) -> None:
        if unary not in ("loop", "ignore"):
            raise ValueError(f"unknown unary handling {unary!r}")
        self.sim = sim
        self.theta = theta
        self.lp = lp
        self.anchor_pairs = anchor_pairs
        self.unary = unary
        self.max_maps = max_maps

    def query_edges(self, pattern: Pattern) -> list[Edge]:
        edges = list(pattern.edges)
        if self.unary == "loop":
            nxt = max((e.id for e in edges), default=-1) + 1
            for var, conds in pattern.conditions.items():
                for c in conds:
                    edges.append(Edge(nxt, var, c, var))
                    nxt += 1
        return edges

    def frontier(self, pattern: Pattern, result: MatchResult, hyps: int = 5) -> list[dict[str, Any]]:
        return Matcher.frontier(pattern, result, hyps, edges=self.query_edges(pattern))

    def match(self, pattern: Pattern, facts: Sequence[Fact], entities: Iterable[Sequence[str]] | None = None,
              passages: Mapping[str, Sequence[str]] | None = None) -> MatchResult:
        edges = self.query_edges(pattern)
        links = [e for e in edges if e.head != e.tail]
        kg = _KG(facts, entities, passages)
        if not edges or not kg.entities:
            return MatchResult(complete=False)
        nodes = list(dict.fromkeys(x for e in edges for x in (e.head, e.tail)))
        named = {u: u[1:] if is_var(u) else u for u in nodes}
        vecs = self.sim.embed([kg.text[k] for k in kg.entities])
        vec = {k: vecs[i] for i, k in enumerate(kg.entities)}
        sims = self.sim.embed([named[u] for u in nodes]) @ vecs.T
        score: dict[str, dict[str, float]] = {}
        for i, u in enumerate(nodes):
            exact = norm_entity(named[u])
            row = {k: 1.0 if k == exact else float(sims[i, j]) for j, k in enumerate(kg.entities)}
            score[u] = {k: s for k, s in sorted(row.items(), key=lambda ks: (-ks[1], ks[0])) if s > self.theta}
        relations = list(dict.fromkeys([*(f.relation for k in kg.entities for f, _ in kg.out.get(k, ())),
                                        *(e.relation for e in links)]))
        rel = dict(zip(relations, self.sim.embed(relations), strict=True)) if relations else {}
        paths: dict[tuple[str, str], list[Fact] | None] = {}

        def path(start: str, target: str) -> list[Fact] | None:
            if (start, target) in paths:
                return paths[(start, target)]
            found: list[Fact] | None = None
            # greedy walk of at most lp facts, preferring the target, then the running sum nearest it
            goal, h_p, node, seen, steps = vec[target], vec[start], start, {start}, []
            for _ in range(self.lp if start != target else 0):
                best: tuple[tuple[Any, ...], Fact, str, np.ndarray] | None = None
                for f, y in kg.out.get(node, ()):
                    if y in seen or y not in vec:
                        continue
                    m = h_p + rel[f.relation] + vec[y]
                    n = float(np.linalg.norm(m))
                    s = float(m @ goal) / n if n > 0 else 0.0
                    rank = (y != target, -s, y, f.relation)
                    if s > self.theta and (best is None or rank < best[0]):
                        best = (rank, f, y, m / n)
                if best is None:
                    break
                _, f, y, h_p = best
                steps.append(f)
                if y == target:
                    found = steps
                    break
                node = y
                seen.add(y)
            paths[(start, target)] = found
            return found

        def edge_score(e: Edge, p: Sequence[Fact]) -> float:
            mean = np.mean([rel[f.relation] for f in p], axis=0)
            n = float(np.linalg.norm(mean))
            return float(mean @ rel[e.relation]) / n if n > 0 else 0.0

        adjacent: dict[str, list[str]] = defaultdict(list)
        for e in links:
            adjacent[e.head].append(e.tail)
            adjacent[e.tail].append(e.head)

        def order_from(start: str) -> tuple[list[str], set[str]]:
            order: list[str] = []
            roots: set[str] = set()
            seen: set[str] = set()
            for root in [start, *(u for u in nodes if not is_var(u)), *nodes]:
                if root in seen:
                    continue
                roots.add(root)
                seen.add(root)
                queue = [root]
                while queue:
                    u = queue.pop(0)
                    order.append(u)
                    for w in adjacent[u]:
                        if w not in seen:
                            seen.add(w)
                            queue.append(w)
            return order, roots

        budget = self.max_maps
        found_values: dict[str, dict[str, float]] = defaultdict(dict)
        supported: set[int] = set()
        best_valid: tuple[float, dict[str, str], dict[int, EdgeSupport]] | None = None
        best_partial: tuple[tuple[int, int, float], dict[str, str], dict[int, EdgeSupport]] | None = None

        def record(fmap: dict[str, str], sup: dict[int, EdgeSupport]) -> None:
            nonlocal best_valid, best_partial
            for u, k in fmap.items():
                if is_var(u) and score[u][k] > found_values[u].get(k, -1.0):
                    found_values[u][k] = score[u][k]
            supported.update(sup)
            mean = sum(s.score for s in sup.values()) / len(edges)
            if len(fmap) == len(nodes) and (best_valid is None or mean > best_valid[0]):
                best_valid = (mean, dict(fmap), dict(sup))
            rank = (len(fmap), len(sup), mean)
            if best_partial is None or rank > best_partial[0]:
                best_partial = (rank, dict(fmap), dict(sup))

        def extend(i: int, order: list[str], roots: set[str], fmap: dict[str, str], used: set[str],
                   sup: dict[int, EdgeSupport]) -> None:
            nonlocal budget
            if i == len(order) or budget <= 0:
                record(fmap, sup)
                return
            u = order[i]
            ties = [e for e in links if (e.head == u and e.tail in fmap) or (e.tail == u and e.head in fmap)]
            placed = False
            if ties or u in roots:
                for k in score[u]:
                    if k in used:
                        continue
                    if budget <= 0:
                        break
                    budget -= 1
                    new: dict[int, EdgeSupport] = {}
                    for e in ties:
                        h = k if e.head == u else fmap[e.head]
                        t = k if e.tail == u else fmap[e.tail]
                        p = path(h, t)
                        if p is None:
                            break
                        new[e.id] = EdgeSupport(e.id, kg.display[h], kg.display[t], list(p), edge_score(e, p))
                    else:
                        placed = True
                        fmap[u] = k
                        used.add(k)
                        extend(i + 1, order, roots, fmap, used, {**sup, **new})
                        del fmap[u]
                        used.discard(k)
            if not placed:
                extend(i + 1, order, roots, fmap, used, sup)

        starts = [u for u in nodes if not is_var(u)] or nodes
        pairs = sorted(((u, k, s) for u in starts for k, s in score[u].items()),
                       key=lambda p: (-p[2], starts.index(p[0]), p[1]))
        for n, (u, k, _) in enumerate(pairs):
            if budget <= 0 or (n >= self.anchor_pairs and best_valid is not None):
                break
            budget -= 1
            order, roots = order_from(u)
            extend(1, order, roots, {u: k}, {k}, {})

        if budget <= 0:
            log.info("PSGM: the search stopped at max_maps=%d trials (%d anchor pairs, %d facts); best so far kept",
                     self.max_maps, len(pairs), len(facts))
        chosen = best_valid or best_partial
        fmap, sup = (chosen[1], chosen[2]) if chosen is not None else ({}, {})
        bindings = {u: kg.display[k] for u, k in fmap.items() if is_var(u)}
        complete = (best_valid is not None and len(links) == len(edges)
                    and all(v in bindings for v in pattern.variables))
        values: dict[str, list[tuple[str, float]]] = {}
        for v in dict.fromkeys([*pattern.variables, *(u for u in nodes if is_var(u))]):
            ranked = sorted(((kg.display[k], s) for k, s in found_values.get(v, {}).items()),
                            key=lambda vs: (vs[0] != bindings.get(v), -vs[1], vs[0]))
            values[v] = ranked[:MAX_VALUES]
        return MatchResult(complete=complete, bindings=bindings, support=sup, values=values, supported=supported,
                           score=sum(s.score for s in sup.values()) / len(edges))
