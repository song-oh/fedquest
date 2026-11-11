# calibration probes: FedQuest's rounds, each candidate labeled against gold
from __future__ import annotations

from collections.abc import Collection, Sequence
from typing import Any, Literal

from fedquest.kg.normalize import norm_entity
from fedquest.llm import prompts
from fedquest.methods import compiler
from fedquest.methods.base import Method, MethodResult, QueryContext, register
from fedquest.methods.common import finish, make_pattern, shaped
from fedquest.methods.fedquest import Asked, FedQuest, Found, Held, _body, _Run, _stated, prune
from fedquest.planner.calibration import VECTOR_KINDS, rank_feature
from fedquest.types import Candidate, Pattern, ProbeCounts, ProbeRequest, Question

K_MAX = 80


def gold_keys(q: Question) -> tuple[dict[str, set[str]], set[str]]:
    if "gold" in q.meta:
        g = q.meta["gold"]
        return ({"first": set(g.get("first", [])), "final": set(g.get("final", []))},
                set(q.meta.get("gold_tables_unknown", [])))
    return {"first": set(), "final": {f"text:{d}" for d in q.gold_doc_ids}}, set(q.gold_doc_ids)


def object_key(c: Candidate) -> str:
    return f"row:{c.oid.rsplit('/', 1)[-1]}" if c.modality == "table" else f"{c.modality}:{c.doc_id}"


def is_evidence(c: Candidate, gold: set[str], unknown_tables: set[str], values: Sequence[str],
                table: str) -> bool | None:
    if object_key(c) in gold:
        return True
    if c.modality == "table" and table and c.doc_id == table and any(
            prompts.witnessed(_stated(c), v) for v in values if witnessable(v)):
        return True
    if c.modality == "table" and c.doc_id in unknown_tables:
        return None
    return False


def hop_type(qtype: str, role: str) -> str:
    if qtype.startswith("Compose(") and qtype.endswith(")"):
        depth, inner = 0, qtype[len("Compose("):-1]
        for i, ch in enumerate(inner):
            depth += ch == "("
            depth -= ch == ")"
            if ch == "," and depth == 0:
                return inner[i + 1:] if role == "first" else inner[:i]
    return qtype


def witnessable(value: str) -> bool:
    return prompts.judgment(value) is None and 0 < len(value) <= 80


class _Probed(_Run):
    def __init__(self, m: FedQuest, q: Question, ctx: QueryContext, pattern: Pattern,
                 pictured: Collection[tuple[str, str]] = (), *, probe: str = "topk", limits: str = "fixed") -> None:
        super().__init__(m, q, ctx, pattern, pictured)
        self.semijoins, self.shares = probe == "semijoin", limits == "share"
        self.lists: dict[int, tuple[ProbeRequest, list[ProbeCounts], bool]] = {}

    def shaped(self, r: ProbeRequest, kind: str) -> list[ProbeRequest]:
        return shaped(r, kind, self.round_kind, self.sizes, self.groups, semijoin=self.semijoins, share=self.shares)

    async def probe(self, reqs: Sequence[ProbeRequest]) -> list[list[Candidate]]:
        kinds = {id(r): meta[0] for r, meta in self.round_lists}
        parts = [self.shaped(r, kinds.get(id(r), "sibling")) for r in reqs]
        if all(len(part) == 1 and part[0] is r for part, r in zip(parts, reqs, strict=True)):
            return await super().probe(reqs)
        sent = self.versioned([x for part in parts for x in part])
        found, counts = await self.ctx.fed.probe_counted(sent, ledger=self.ctx.ledger)
        out, i = [], 0
        for r, part in zip(reqs, parts, strict=True):
            out.append([c for cands in found[i:i + len(part)] for c in cands])
            self.lists[id(r)] = (r, [n for ns in counts[i:i + len(part)] for n in ns], any(x.semijoin for x in part))
            i += len(part)
        return out

    def features(self, f: Found, rnd: str) -> dict[str, Any]:
        c = f.c
        req = self.round_lists[f.probe][0] if f.probe < len(self.round_lists) else None
        _, counts, semijoin = self.lists.get(id(req), (req, [], False))
        encoder, m_site = self.sizes.get((c.site, c.collection), ("", 0))
        vector = rnd != "filter" and f.kind in VECTOR_KINDS
        if vector and semijoin:
            own = [n for n in counts if n.site == c.site and n.collection == c.collection]
            j_site = sum(n.joinable for n in own)
            j_group = sum(n.joinable for n in counts if n.collection == c.collection
                          and self.sizes.get((n.site, n.collection), ("", 0))[0] == encoder)
            jrank = c.jrank if c.jrank is not None else c.rank
            g = rank_feature(f.kind, rnd, c.rank, jrank=jrank, j_site=j_site, j_group=j_group)
            exact = bool(own) and all(n.exact for n in own)
            return {"g": float(g), "mentions": c.mentions, "jrank": jrank, "joinable": j_site,
                    "joinable_group": j_group, "exact": exact, "r": g.r, "n_s": g.n_s, "n": g.n, "encoder": encoder,
                    **({"z": c.z} if c.z is not None else {}), **({"zv": c.zv} if c.zv else {})}
        m_group = self.groups.get((c.collection, encoder), 0)
        g = rank_feature(f.kind, rnd, c.rank, m_site, m_group)
        out: dict[str, Any] = {"g": float(g), "mentions": c.mentions, "encoder": encoder}
        if vector:
            out |= {"objects": m_site, "group_objects": m_group}
        return {**out, "r": g.r, "n_s": g.n_s, "n": g.n, **({"z": c.z} if c.z is not None else {}),
                **({"zv": c.zv} if c.zv else {})}


@register("explore")
class Explore(Method):
    def __init__(self, k: int = 5, titles: Literal["all", "none"] = "all", mentions: bool = True,
                 graph: Literal["mgrag", "typed"] = "mgrag", probe: Literal["topk", "semijoin"] = "topk",
                 limits: Literal["fixed", "share"] = "fixed", layout: str = "", mention: int = 1,
                 scores: bool = False) -> None:
        super().__init__()
        if graph not in ("mgrag", "typed") or probe not in ("topk", "semijoin") or limits not in ("fixed", "share"):
            raise ValueError(f"unknown graph {graph!r}, probe {probe!r} or limits {limits!r}")
        if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= K_MAX:
            raise ValueError(f"k {k!r} is no probe depth from 1 to {K_MAX}")
        if isinstance(mention, bool) or mention not in prompts.MENTION_VERSIONS:
            raise ValueError(f"unknown mention test {mention!r} (prompts.MENTION_VERSIONS)")
        self.k, self.titles, self.mentions = k, titles, mentions
        self.graph, self.probe, self.limits, self.layout, self.mention = graph, probe, limits, layout, int(mention)
        self.scores = bool(scores)

    async def answer(self, q: Question, ctx: QueryContext) -> MethodResult:
        with prompts.mention_scope(self.mention):
            return await self._answer(q, ctx)

    async def _answer(self, q: Question, ctx: QueryContext) -> MethodResult:
        pictured: frozenset[tuple[str, str]] = frozenset()
        if self.graph == "typed":
            try:
                pattern, pictured = await compiler.typed_graph(q, ctx)
            except NotImplementedError as e:
                raise NotImplementedError(f"explore graph='typed' lays its rounds out from FedQuest's typed query "
                                          f"graph, not built yet: {e}") from e
        else:
            pattern = await make_pattern(q, ctx)
        if not pattern.variables:
            pattern = prompts.parse_pattern("", q.question)
        method = FedQuest(k=self.k, titles=self.titles, mentions=self.mentions, planner="hop", visual="regex",
                          graph=self.graph, mention=self.mention)
        run = _Probed(method, q, ctx, prune(pattern, q.question), pictured, probe=self.probe, limits=self.limits)
        run.want_z = self.scores
        await run.setup()
        run.decide()
        gold, unknown = gold_keys(q)
        compose = bool(gold["first"]) and bool(q.bridge)
        answer = run.p.answer if run.p.answer in run.p.variables else None

        def role(xs: Sequence[str], late: bool) -> str:
            return "first" if compose and not late and any(x != answer for x in xs) else "final"

        kind, asks = run.step()
        if not asks:
            return finish(q, self.name, "", pattern=run.p)
        first = list(dict.fromkeys(a.var for a in asks))
        await self.observe(run, q, kind, "filter" if kind == "filter" else "first", asks, gold, unknown,
                           role(first, False))
        self.ask(run, asks)
        for x in first:
            if not any(c.visual and run.owns(c) for c in run.by_var[x]):
                continue
            r = role([x], False)
            values = q.bridge[:1] if r == "first" else q.answers
            if not values:
                continue
            text = [c for c in run.by_var[x] if not c.visual]
            members = list(dict.fromkeys([*values, *q.meta.get("list_titles", [])]))[:run.list_cap]
            run.values[x] = {norm_entity(v): Held(v, {c.id: {""} for c in text}) for v in members}
            self.ask(run, [Asked(x, c, None, "") for c in text])
            kind, filters = run.step()
            if kind == "filter" and filters:
                await self.observe(run, q, kind, "filter", filters, gold, unknown, r)
                self.ask(run, filters)
        if compose:
            bridge = q.bridge[0]
            for x in first:
                if x == answer:
                    continue
                run.bound[x], run.pi[x] = [bridge], {norm_entity(bridge): 1.0}
                self.ask(run, [Asked(x, c, v, "") for c in run.by_var[x]
                               for v in (run.values_of(c.other) if c.other else [None])])
            kind, late = run.step()
            if late:
                await self.observe(run, q, kind, "late", late, gold, unknown, "final")
        return finish(q, self.name, "", pattern=run.p)

    @staticmethod
    def ask(run: _Probed, asks: Sequence[Asked]) -> None:
        for a in asks:
            run.asked.add((a.cond.id, norm_entity(a.value or "")))

    async def observe(self, run: _Probed, q: Question, kind: str, rnd: str, asks: Sequence[Asked],
                      gold: dict[str, set[str]], unknown: set[str], role: str) -> None:
        found: list[Found] = await run.paths(kind, asks)
        objects = gold[role]
        own = (q.meta.get("gold_values") or {}).get(role)
        values = [v for v in (own or (q.bridge[:1] if role == "first" else q.answers)) if v]
        table = q.meta.get("table_id", "")
        listed = values if "ImageListQ" in hop_type(q.qtype, role) else []
        labels = [is_evidence(f.c, objects, unknown, listed, table) for f in found]
        groups = {"image" if k.startswith("image:") else "text" for k in objects} | ({"text"} if listed else set())
        reached = {"image" if f.c.modality == "image" else "text" for f in found}
        v4: dict[str, Any] = {"layout": self.layout, "k": self.k} if self.layout else {}
        run.ctx.log("obs", qid=q.qid, round=rnd, path="gold", groups=sorted(groups & reached), **v4)
        for f, evidence in zip(found, labels, strict=True):
            c = f.c
            entry = ("complete" if c.complete else "title" if c.site in run.titled or c.modality == "image"
                     else "snippet")
            states = None
            if evidence and entry == "snippet" and (checkable := [v for v in values if witnessable(v)]):
                states = (any(prompts.witnessed(_body(c), v) for v in checkable)
                          and (f.value is None or prompts.mentions(c.summary, f.value)))
            run.ctx.log("obs", qid=q.qid, qtype=q.qtype, family=q.meta.get("family", ""), round=rnd, kind=f.kind,
                        site=c.site, collection=c.collection, oid=c.oid, modality=c.modality, rank=c.rank,
                        signal=f.signal, entry=entry, tokens=c.tokens, states=states, probe=f.probe,
                        evidence=evidence, **({"layout": self.layout, **run.features(f, rnd)} if self.layout else {}))
