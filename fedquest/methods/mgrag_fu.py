from __future__ import annotations

import asyncio
import copy
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

from fedquest.kg.matcher import Matcher, MatchResult
from fedquest.kg.normalize import norm_entity
from fedquest.kg.psgm import PSGMatcher
from fedquest.llm import prompts
from fedquest.methods import mgrag
from fedquest.methods.base import Method, MethodResult, QueryContext, register
from fedquest.methods.common import (
    VISUAL,
    Evidence,
    Ref,
    StopTest,
    finish,
    generate_answer,
    key_text,
    make_pattern,
    rrf,
    split_lists,
    tie,
)
from fedquest.methods.fedquest import (
    Asked,
    Cond,
    FedQuest,
    Found,
    Held,
    _grounded,
    _key,
    _Run,
    holds_pictures,
    prune,
    route_pictures,
)
from fedquest.site.service import decode_image
from fedquest.types import Candidate, Edge, ExtractMode, Pattern, ProbeRequest, Question, is_var

SIBLING_PROBE = 10_000
PROMPT_TOKENS = 600
OUT_START = 137.0
ORDERS = ("rrf", "relevance", "calib")
MATCHERS = ("psgm", "shared")
KINDS = ("text", "visual", "filter")
VIA = {"question": "question", "bound": "bound", "bridge": "bound", "title": "title", "sibling": "sibling"}


@dataclass
class Queued:
    c: Candidate
    via: str
    round: int


@dataclass
class _Round:
    asks: dict[str, list[Asked]] = field(default_factory=lambda: defaultdict(list))
    weights: dict[str, float] = field(default_factory=dict)
    checks: dict[str, list[str]] = field(default_factory=dict)
    values: list[str] = field(default_factory=list)

    def add(self, kind: str, ask: Asked) -> None:
        same = (ask.cond.id, ask.var, norm_entity(ask.value or ""))
        if not any((a.cond.id, a.var, norm_entity(a.value or "")) == same for a in self.asks[kind]):
            self.asks[kind].append(ask)

    def note(self, value: str, weight: float) -> None:
        key = norm_entity(value)
        self.weights[key] = max(self.weights.get(key, 0.0), weight)
        if value not in self.values:
            self.values.append(value)

    def check(self, x: str, name: str) -> None:
        names = self.checks.setdefault(x, [])
        if name not in names:
            names.append(name)


@register("mgrag_fu")
class MGRAGFU(Method):
    name = "mgrag_fu"
    added_params: ClassVar[dict[str, Any]] = {"probe": "topk", "limits": "fixed", "mention": 1, "ties": "name",
                                              "calibration": None, "total_budget": None}

    def __init__(self, tokens: int | None = 8000, objects: int | None = 32, k: int = 5, k_question: int = 8,
                 anchors: bool = True, titles: bool = True, siblings: bool = True, mentions: bool = True,
                 batch: int = 1, order: str = "rrf", frontier: str = "shared", matcher: str = "psgm",
                 theta: float | None = None, lp: int | None = None, image_titles: bool = True,
                 answer_images: bool = False, verify: bool | None = False, max_rounds: int = 6, hyps: int = 5,
                 unary: str = "loop", probe: str = "topk", limits: str = "fixed", mention: int = 1,
                 ties: str = "name", calibration: str | None = None, total_budget: int | None = None) -> None:
        super().__init__()
        if total_budget is not None and (isinstance(total_budget, bool) or int(total_budget) <= 0):
            raise ValueError(f"total_budget {total_budget!r} is not a positive number of cost units")
        self.total_budget = int(total_budget) if total_budget is not None else None
        if order not in ORDERS or frontier not in MATCHERS or matcher not in MATCHERS:
            raise ValueError(f"unknown order {order!r}, frontier {frontier!r} or matcher {matcher!r}")
        if batch < 1 or hyps < 1 or max_rounds < 0:
            raise ValueError(f"batch {batch!r} and hyps {hyps!r} must be positive, max_rounds {max_rounds!r} >= 0")
        if unary not in ("loop", "ignore"):
            raise ValueError(f"unknown unary {unary!r}")
        self.tokens, self.objects, self.k, self.k_question = tokens, objects, k, k_question
        self.anchors, self.titles, self.siblings, self.mentions = anchors, titles, siblings, mentions
        self.batch, self.order, self.frontier, self.matcher = batch, order, frontier, matcher
        self.theta, self.lp, self.image_titles, self.answer_images = theta, lp, image_titles, answer_images
        self.verify, self.max_rounds, self.hyps, self.unary = verify, max_rounds, hyps, unary
        self.engine_params: dict[str, Any] = {"probe": probe, "limits": limits, "mention": mention, "ties": ties}
        FedQuest(**self.engine_params)
        self.mention, self.calibration = int(mention), calibration

    async def answer(self, q: Question, ctx: QueryContext) -> MethodResult:
        if self.calibration is not None:
            from fedquest import xfit

            name = xfit.resolve(self.calibration, q.qid) or self.calibration
            ctx = FedQuest.recalibrated(self, ctx, name)  # type: ignore[arg-type]
        pattern = await make_pattern(q, ctx)
        with prompts.mention_scope(self.mention):
            return await _FU(self, q, ctx, pattern).go()


class _FU:
    def __init__(self, m: MGRAGFU, q: Question, ctx: QueryContext, pattern: Pattern) -> None:
        self.m, self.q, self.ctx, self.p = m, q, ctx, pattern
        self.engine = FedQuest(k=m.k, keys="pattern", titles="all" if m.titles else "none", mentions=m.mentions,
                               scan=False, **m.engine_params)
        self.fp: Pattern = pattern
        self.fedges: dict[int, Edge] = {e.id: e for e in pattern.edges}
        self.run = _Run(self.engine, q, ctx, pattern)
        self.bound: dict[str, str] = {}
        self.psgm = PSGMatcher(ctx.matcher.sim, theta=ctx.matcher.theta if m.theta is None else m.theta,
                               lp=ctx.matcher.max_hops if m.lp is None else m.lp, unary=m.unary) \
            if "psgm" in (m.matcher, m.frontier) else None
        query_edges = self.psgm.query_edges(pattern) if self.psgm is not None else pattern.edges
        self.edges: dict[int, Edge] = {e.id: e for e in query_edges}
        self.ev = Evidence()
        self.queue: list[Queued] = []
        self.extracted: set[Ref] = set()
        self.consulted: list[Ref] = []
        self.pictures: list[Ref] = []
        self.asked: set[tuple[Any, ...]] = set()
        self.made: dict[tuple[str, int], Cond] = {}
        self.rounds = 0
        self.keys: dict[str, None] = {}
        self.values: dict[str, None] = {}
        self.calls = 0
        self.out_tokens, self.out_objects = 0, 0
        self.limit = "exhausted"

    async def prepare(self) -> None:
        fp = copy.deepcopy(self.p)
        if not fp.variables:
            fp = prompts.parse_pattern("", self.q.question)
        fp = prune(fp, self.q.question)
        pictured: frozenset[tuple[str, str]] = frozenset()
        if self.engine.visual == "llm" and await holds_pictures(self.ctx):
            fp, pictured = await route_pictures(fp, self.q.question, self.ctx)
        self.fp, self.fedges = fp, {e.id: e for e in fp.edges}
        self.run = _Run(self.engine, self.q, self.ctx, fp, pictured)

    def answer_reserve(self) -> int:
        from fedquest import budget as _budget
        from fedquest.methods.fedquest import ANSWER_FRAME, CONTEXT_TOKENS

        return ANSWER_FRAME + CONTEXT_TOKENS + (4 * _budget.IMAGE_TOKENS if self.m.answer_images else 0)

    def fits_total(self, estimate: float) -> bool:
        if self.m.total_budget is None:
            return True
        led = self.ctx.ledger
        return led.total_tokens + estimate + self.answer_reserve() <= self.m.total_budget

    async def go(self) -> MethodResult:
        from fedquest.budget import BudgetExhausted

        if self.m.total_budget is not None:
            self.ctx.ledger.reserve = self.answer_reserve()
        try:
            return await self._go()
        except BudgetExhausted as exc:
            self.limit = "tokens"
            self.ctx.log("budget_stop", at="mgrag_fu", why=str(exc)[:200])
            return await self.respond(MatchResult(complete=False), False)

    async def _go(self) -> MethodResult:
        await self.prepare()
        await self.run.setup()
        self.run.question_probed = True
        if not self.m.siblings:
            self.run.siblings = _no_siblings  # type: ignore[method-assign]
        await self.first_round()
        match = MatchResult(complete=False)
        stop, done = StopTest(self.q, self.ctx, verify=self.m.verify), False
        while self.queue:
            batch = self.take()
            if not batch:
                break
            await self.extract(batch)
            match = await self.matched(self.m.matcher)
            done = await stop(match, self.ev.facts)
            if done:
                self.limit = "match"
                break
            if self.rounds < self.m.max_rounds and self.affordable():
                await self.follow(match if self.m.frontier == self.m.matcher else await self.matched(self.m.frontier))
        return await self.respond(match, done)

    async def respond(self, match: MatchResult, done: bool) -> MethodResult:
        self.ctx.log("fu_summary", followup_rounds=self.rounds, keys=len(self.keys), values=len(self.values),
                     extractions=self.calls, objects=len(self.consulted), limit=self.limit)
        self.ctx.ledger.reserve = 0
        facts = match.support_facts() if done else _distinct([*match.support_facts(), *self.ev.facts])
        snippets = [self.ev.snippets[r] for r in self.consulted if r in self.ev.snippets]
        if self.m.answer_images:
            images = await consulted_images(self.pictures, self.ctx)
            answer = await generate_answer(self.q, self.ctx, facts[:80], snippets, images or None)
        else:
            answer = await generate_answer(self.q, self.ctx, facts[:80], snippets)
        return finish(self.q, self.m.name, answer, pattern=self.p, match=match, evidence=self.ev, facts=facts[:80],
                      consulted=self.consulted, stopped=done)

    async def first_round(self) -> None:
        cands = (await self.ctx.fed.probe([ProbeRequest(self.q.question, "question", self.m.k_question)],
                                          ledger=self.ctx.ledger))[0]
        visual_q = VISUAL.search(self.q.question) is not None
        found = [(Found(c, "question", self.run.signal(c, "question", None, visual_q), None, (), -1), "first")
                 for c in cands]
        question = [f for f, _ in found]
        if self.m.anchors:
            self.run.decide()
            kind, asks = self.run.step()
            keyed = [a for a in asks if a.cond.edge is not None and a.value and a.value != self.q.question]
            got: list[Found] = []
            if keyed or (kind != "text" and asks):
                siblings = self.run.siblings
                self.run.siblings = _no_siblings  # type: ignore[method-assign]
                got = await self.run.paths(kind, asks)
                self.run.siblings = siblings  # type: ignore[method-assign]
                found += [(f, _round(kind, asks)) for f in got]
                for a in keyed:
                    self.asked.add((a.cond.edge.id, a.var, norm_entity(a.value)))  # type: ignore[union-attr,arg-type]
            if self.m.siblings and keyed:
                rows = await self.run.siblings([*got, *question], asks, SIBLING_PROBE)
                found += [(f, _round(kind, asks)) for f in rows]
        self.push(await self.fuse(found, {}), 0)

    def take(self) -> list[Queued]:
        if self.m.tokens is not None or self.m.total_budget is not None:
            fits = [x for x in self.queue
                    if (self.m.tokens is None or self.ctx.ledger.evidence_tokens + self.estimate([x.c]) <= self.m.tokens)
                    and self.fits_total(self.estimate([x.c]))]
            if not fits:
                self.limit = "tokens"
                return []
        else:
            fits = self.queue
        head = fits[0]
        batch = [head]
        size = min(self.m.batch, prompts.PER_CALL.get(head.c.modality, self.m.batch))
        for item in fits:
            if len(batch) >= size:
                break
            if item is not head and item.c.site == head.c.site and item.c.modality == head.c.modality:
                batch.append(item)
        if self.m.objects is not None:
            batch = batch[:max(0, self.m.objects - len(self.consulted))]
            if not batch:
                self.limit = "objects"
                return []
        while batch and ((self.m.tokens is not None and self.ctx.ledger.evidence_tokens + self.estimate(
                [b.c for b in batch]) > self.m.tokens) or not self.fits_total(self.estimate([b.c for b in batch]))):
            batch = batch[:-1]
        if not batch:
            self.limit = "tokens"
            return []
        taken = {id(b) for b in batch}
        self.queue = [x for x in self.queue if id(x) not in taken]
        return batch

    def affordable(self) -> bool:
        if self.m.objects is not None and len(self.consulted) >= self.m.objects:
            return False
        one = self.estimate([]) + 4.0 * (self.out_tokens / self.out_objects if self.out_objects else OUT_START)
        return (self.m.tokens is None or self.ctx.ledger.evidence_tokens + one <= self.m.tokens) and self.fits_total(one)

    def estimate(self, cands: Sequence[Candidate]) -> float:
        mean = self.out_tokens / self.out_objects if self.out_objects else OUT_START
        return PROMPT_TOKENS + sum(c.tokens for c in cands) + 4.0 * mean * len(cands)

    async def extract(self, batch: Sequence[Queued]) -> None:
        refs = [b.c.ref for b in batch]
        if self.m.batch > 1:
            results = await self.ctx.fed.extract(refs, include_raw=True, mode="query_block", query=self.q.question,
                                                 blocks=[refs], ledger=self.ctx.ledger)
        else:
            mode: ExtractMode = "query_t" if self.m.image_titles else "query"
            results = await self.ctx.fed.extract(refs, include_raw=True, mode=mode, query=self.q.question,
                                                 ledger=self.ctx.ledger)
        self.calls += 1
        for r in results:
            self.ev.add_result(r)
        ok = [r for r in results if r.error is None]
        if ok:
            self.out_tokens += sum(r.usage.completion_tokens for r in results)
            self.out_objects += len(ok)
        for b in batch:
            self.extracted.add(b.c.ref)
            self.consulted.append(b.c.ref)
            if b.c.modality == "image":
                self.pictures.append(b.c.ref)
            self.ctx.log("pick", ref=f"{b.c.site}/{b.c.oid}", rank=b.c.rank, via=b.via, round=b.round)

    async def matched(self, which: str) -> MatchResult:
        if which == "psgm" and self.psgm is not None:
            return await asyncio.to_thread(self.psgm.match, self.p, self.ev.facts, self.ev.entity_records(),
                                           self.ev.passages(self.consulted))
        return await asyncio.to_thread(self.ctx.matcher.match, self.p, self.ev.facts)

    def frontier(self, result: MatchResult) -> list[dict[str, Any]]:
        if self.m.frontier == "psgm" and self.psgm is not None:
            return self.psgm.frontier(self.p, result, hyps=self.m.hyps)
        return Matcher.frontier(self.p, result, hyps=self.m.hyps)

    def edge(self, ref: Any) -> Edge | None:
        if isinstance(ref, Edge):
            ref = ref.id
        if isinstance(ref, int):
            return self.edges.get(ref)
        if isinstance(ref, Sequence) and not isinstance(ref, str) and len(ref) == 3:
            return next((e for e in self.edges.values() if (e.head, e.relation, e.tail) == tuple(ref)), None)
        return None

    def cond(self, x: str, e: Edge) -> Cond:
        own = next((c for c in self.run.conds if c.var == x and c.edge is e and not c.hop), None)
        if own is not None:
            return own
        if (x, e.id) not in self.made:
            self.made[(x, e.id)] = Cond(len(self.run.conds) + len(self.made), x, e,
                                        visual=bool(self.run.collections(image=True)) and e.modality == "image")
        return self.made[(x, e.id)]

    def relation(self, e: Edge, cond: Cond, known: str, value: str) -> str:
        rel = _grounded(e.relation, self.fp, {**{x: v for x, v in self.bound.items() if x in self.fp.variables},
                                              known: value})
        return f"{rel} X that {cond.text}" if cond.text else rel

    def find(self, rnd: _Round, e: Edge, x: str, value: str, weight: float, result: MatchResult | None) -> None:
        u = e.other(x)
        if not self.m.anchors and not is_var(u):
            return
        cond = self.cond(x, e)
        kind = self.run.kind_of(cond)
        if kind == "filter" and result is not None and result.values.get(x):
            for name, w in result.values[x][:self.m.hyps]:
                self.check_picture(rnd, e, x, cond, name, value, float(w))
            return
        key = (e.id, x, norm_entity(value))
        if key not in self.asked:
            self.asked.add(key)
            rnd.add(kind, Asked(x, cond, value, self.relation(e, cond, u, value)))
            rnd.note(value, weight)

    def check(self, rnd: _Round, e: Edge, x: str, name: str, other: str, weight: float) -> None:
        if e.head == e.tail:
            alone = next((c for c in self.run.by_var.get(x, []) if c.edge is None and c.text == e.relation), None)
            if alone is not None and alone.visual:
                self.check_alone(rnd, alone, x, name, weight)
            return
        cond = self.cond(x, e)
        if self.run.kind_of(cond) == "filter":
            self.check_picture(rnd, e, x, cond, name, other, weight)
            return
        o = e.other(x)
        flipped = self.cond(o, e)
        if self.run.kind_of(flipped) == "filter":
            if is_var(o):
                self.check_picture(rnd, e, o, flipped, other, name, weight)
            else:
                self.find(rnd, e, x, o, weight, None)
            return
        key = (e.id, o, norm_entity(name))
        if key not in self.asked:
            self.asked.add(key)
            rnd.add(self.run.kind_of(flipped), Asked(o, flipped, name, self.relation(e, flipped, x, name)))
            rnd.note(name, weight)

    def check_picture(self, rnd: _Round, e: Edge, x: str, cond: Cond, name: str, value: str, weight: float) -> None:
        key = ("picture", cond.id, x, _key(name), norm_entity(value))
        if key in self.asked or _key(name) in self.run.filtered.get(x, ()):
            return
        self.asked.add(key)
        rnd.add("filter", Asked(x, cond, value, self.relation(e, cond, e.other(x), value)))
        rnd.check(x, name)
        rnd.note(name, weight)

    def check_alone(self, rnd: _Round, c: Cond, x: str, name: str, weight: float) -> None:
        key = ("alone", c.id, _key(name))
        if key in self.asked:
            return
        self.asked.add(key)
        rnd.add("filter", Asked(x, c, None, c.text))
        rnd.check(x, name)
        rnd.note(name, weight)

    async def follow(self, result: MatchResult) -> None:
        rnd = _Round()
        self.bound = dict(result.bindings)
        for item in self.frontier(result):
            pe, x, value = self.edge(item.get("edge")), str(item.get("var", "")), str(item.get("value") or "")
            if pe is None or not is_var(x) or x not in (pe.head, pe.tail) or not norm_entity(value):
                continue
            e = pe if pe.head == pe.tail else self.fedges.get(pe.id)
            if e is None:
                continue
            weight, check = float(item.get("weight", 1.0)), item.get("kind") == "check"
            if x not in self.fp.variables:
                const, y = (e.head, e.tail) if x == pe.head else (e.tail, e.head)
                name = str(item.get("other") or "") if check else value
                if not is_var(const) and y in self.fp.variables and norm_entity(name):
                    self.check(rnd, e, y, name, const, weight)
                continue
            if x not in (e.head, e.tail):
                continue
            other = e.other(x)
            if check:
                self.check(rnd, e, x, value, str(item.get("other") or value) if is_var(other) else other, weight)
            else:
                self.find(rnd, e, x, value if is_var(other) else other, weight, result)
        for x in self.fp.variables:
            for c in self.run.by_var.get(x, []):
                if c.edge is None and c.visual:
                    for name, w in result.values.get(x, [])[:self.m.hyps]:
                        self.check_alone(rnd, c, x, name, float(w))
        if not rnd.asks:
            return
        self.rounds += 1
        note = getattr(self.ctx.ledger, "note_origin", None)
        for value in rnd.values:
            self.values.setdefault(value, None)
            site = next((f.site for f in self.ev.facts if norm_entity(value) in (norm_entity(f.head),
                                                                                norm_entity(f.tail))), "")
            if site and note is not None:
                note(value, site)
        found: list[tuple[Found, str]] = []
        keys: list[str] = []
        for kind in KINDS:
            asks = rnd.asks.get(kind, [])
            if not asks:
                continue
            if kind == "filter":
                for x, names in rnd.checks.items():
                    self.run.values[x] = {norm_entity(n): Held(n) for n in names}
            keys += [a.cond.text if a.cond.edge is None else key_text(a.value or "", a.relation, a.cond.edge.head ==
                                                                       a.cond.other) for a in asks]
            found += [(f, _round(kind, asks)) for f in await self.run.paths(kind, asks)]
            if kind == "filter":
                for x, names in rnd.checks.items():
                    self.run.filtered[x] |= {_key(n) for n in names}
                    self.run.values.pop(x, None)
        for k in keys:
            self.keys.setdefault(k, None)
        self.ctx.log("followup", round=self.rounds, keys=list(dict.fromkeys(keys)), values=rnd.values)
        self.push(await self.fuse(found, rnd.weights), self.rounds)

    async def fuse(self, found: Sequence[tuple[Found, str]], weights: dict[str, float]) -> list[tuple[Candidate, str]]:
        kept = [(f, rnd) for f, rnd in found if f.c.ref not in self.extracted and f.c.mentions is not False]
        if not kept:
            return []
        via: dict[Ref, str] = {}
        first: dict[Ref, Candidate] = {}
        for f, rnd in kept:
            via.setdefault(f.c.ref, "check" if rnd == "filter" and f.kind == "title" else VIA.get(f.kind, f.kind))
            first.setdefault(f.c.ref, f.c)
        if self.m.order == "rrf":
            order = rrf(split_lists([f.c for f, _ in kept]), salt=self.q.qid)
        elif self.m.order == "relevance":
            cands = list(first.values())
            (sims,) = await asyncio.to_thread(self.ctx.matcher.sim.relevance,
                                              [(self.q.question, [c.summary for c in cands])])
            score = {c.ref: float(s) for c, s in zip(cands, sims, strict=True)}
            order = sorted(cands, key=lambda c: (-score[c.ref], c.rank, tie(self.q.qid, c), c.oid))
        else:
            alpha = self.calibrated(kept, weights)
            order = sorted(first.values(), key=lambda c: (-alpha[c.ref], c.rank, tie(self.q.qid, c), c.oid))
        return [(c, via[c.ref]) for c in order]

    def calibrated(self, found: Sequence[tuple[Found, str]], weights: dict[str, float]) -> dict[Ref, float]:
        cal = self.ctx.calib
        best: dict[Ref, dict[str, tuple[float, float]]] = defaultdict(dict)
        for f, rnd in found:
            a = cal.alpha(f.c.site, f.c.collection, f.kind, f.c.rank, f.signal, f.c.modality, rnd)
            hyp = norm_entity(f.value) if f.value and f.kind != "question" else ""
            if a > best[f.c.ref].get(hyp, (-1.0, 0.0))[0]:
                best[f.c.ref][hyp] = (a, weights.get(hyp, 1.0) if hyp else 1.0)
        return {ref: max(h.get("", (0.0, 0.0))[0], sum(a * w for k, (a, w) in h.items() if k))
                for ref, h in best.items()}

    def push(self, fused: Sequence[tuple[Candidate, str]], rnd: int) -> None:
        refs = {c.ref for c, _ in fused}
        self.queue = [Queued(c, via, rnd) for c, via in fused] + [x for x in self.queue if x.c.ref not in refs]


def _distinct(facts: Sequence[Any]) -> list[Any]:
    seen: set[tuple[str, str, str]] = set()
    out = []
    for f in facts:
        k = (norm_entity(f.head), " ".join(f.relation.lower().split()), norm_entity(f.tail))
        if k not in seen:
            seen.add(k)
            out.append(f)
    return out


async def _no_siblings(*_: Any) -> list[Found]:
    return []


def _round(kind: str, asks: Sequence[Asked]) -> str:
    return "filter" if kind == "filter" else "late" if any(is_var(a.cond.other or "") for a in asks) else "first"


async def consulted_images(pictures: Sequence[Ref], ctx: QueryContext) -> list[str]:
    if not pictures:
        return []
    chosen = list(pictures[:mgrag.ANSWER_IMAGES])
    raws = {(r.site, r.oid): r for r in await ctx.fed.fetch_raw(chosen, ledger=ctx.ledger)}
    uris = [decode_image(raws[ref]) if ref in raws and not raws[ref].denied else None for ref in chosen]
    return [u for u in uris if u]
