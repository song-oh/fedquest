from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any, ClassVar, Literal

from fedquest.kg.matcher import MatchResult
from fedquest.kg.normalize import norm_entity
from fedquest.kg.psgm import PSGMatcher
from fedquest.kg.view import KEEP_VERSION
from fedquest.methods import compiler
from fedquest.methods.base import Method, MethodResult, QueryContext, register
from fedquest.methods.common import (
    Evidence,
    Ref,
    StopTest,
    finish,
    first_per_ref,
    generate_answer,
    make_pattern,
    rrf,
    split_lists,
    tie,
)
from fedquest.site.service import decode_image
from fedquest.types import (
    Candidate,
    ExtractionResult,
    ExtractMode,
    Fact,
    Pattern,
    ProbeRequest,
    Question,
    Usage,
)

ANSWER_IMAGES = 4
GOLD_CHUNKS = 8
KEEP_LOOKUP = 200


@register("mgrag")
class MGRAG(Method):
    name = "mgrag"
    added_params: ClassVar[dict[str, Any]] = {"graph": "mgrag"}

    def __init__(self, budget: int = 5, k: int | None = None, use_dpr: bool = True, order: str = "rrf",
                 verify: bool | None = None, answer_images: bool = False, matcher: str = "shared",
                 theta: float | None = None, lp: int | None = None, image_titles: bool = False,
                 gold_support: bool = False, unary: str = "loop", keep: bool = False,
                 graph: Literal["mgrag", "typed"] = "mgrag") -> None:
        super().__init__()
        if graph not in ("mgrag", "typed"):
            raise ValueError(f"unknown graph {graph!r}")
        self.graph = graph
        if order not in ("rrf", "relevance"):
            raise ValueError(f"unknown order {order!r}")
        if matcher not in ("shared", "psgm"):
            raise ValueError(f"unknown matcher {matcher!r}")
        if unary not in ("loop", "ignore"):
            raise ValueError(f"unknown unary {unary!r}")
        self.budget, self.k, self.use_dpr, self.order = budget, k, use_dpr, order
        self.verify, self.answer_images = verify, answer_images
        self.matcher, self.theta, self.lp = matcher, theta, lp
        self.image_titles, self.gold_support, self.unary = image_titles, gold_support, unary
        self.keep = keep

    def psgm(self, ctx: QueryContext) -> PSGMatcher | None:
        if self.matcher != "psgm":
            return None
        theta = ctx.matcher.theta if self.theta is None else self.theta
        return PSGMatcher(ctx.matcher.sim, theta=theta, lp=ctx.matcher.max_hops if self.lp is None else self.lp,
                          unary=self.unary)

    async def answer(self, q: Question, ctx: QueryContext) -> MethodResult:
        pattern = (await compiler.typed_graph(q, ctx))[0] if self.graph == "typed" else await make_pattern(q, ctx)
        ev = Evidence()
        held = await self.held(pattern, ctx) if self.keep else []
        match = MatchResult(complete=False)
        stop, done = StopTest(q, ctx, verify=self.verify), False
        psgm = self.psgm(ctx)
        consulted: list[Ref] = []
        kept: list[Candidate] = []
        visited: list[Ref] = []
        pictures: list[Ref] = []
        extracted: list[ExtractionResult] = []

        async def matched() -> bool:
            nonlocal match
            facts = list(dict.fromkeys([*ev.facts, *held])) if held else ev.facts
            if psgm is None:
                match = await asyncio.to_thread(ctx.matcher.match, pattern, facts)
            else:
                match = await asyncio.to_thread(psgm.match, pattern, facts, ev.entity_records(),
                                                ev.passages(visited))
            return await stop(match, facts)

        if held:
            done = await matched()
            ctx.log("reuse", mode="keep", facts=len(held), objects=len({(f.site, f.oid) for f in held}),
                    stopped=done, values=dict(match.bindings) if done else {})
        order = [] if done else await self.candidates(q, ctx)
        covered = await ctx.kg.covered(ctx.ns, [c.ref for c in order[: self.budget]], KEEP_VERSION) \
            if self.keep and order else {}
        mode: ExtractMode = "query_t" if self.image_titles else "query"
        for cand in order[: self.budget]:
            visited.append(cand.ref)
            if cand.modality == "image":
                pictures.append(cand.ref)
            if cand.ref in covered:
                ev.add_facts(covered[cand.ref], cand.ref)
                if cand.doc_id:
                    ev.doc_ids[cand.ref] = cand.doc_id
                ctx.ledger.charge_reuse(covered[cand.ref])
                kept.append(cand)
                ctx.log("pick", ref=f"{cand.site}/{cand.oid}", rank=cand.rank, kept=True)
            else:
                results = await ctx.fed.extract([cand.ref], include_raw=True, mode=mode, query=q.question,
                                                ledger=ctx.ledger)
                for r in results:
                    ev.add_result(r)
                extracted += results
                consulted.append(cand.ref)
                ctx.log("pick", ref=f"{cand.site}/{cand.oid}", rank=cand.rank)
            done = await matched()
            if done:
                break
        if self.keep:
            await self.store(extracted, ctx)
            if kept and self.use_dpr:
                await self.kept_texts(kept, ev, ctx)
            ctx.ledger.count_reuse("keep_facts", len(dict.fromkeys(
                [*held, *(f for c in kept for f in covered[c.ref])])))
            ctx.ledger.count_reuse("keep_skipped", len(kept))
        facts = match.support_facts() if done else list(dict.fromkeys([*ev.facts, *held]))
        snippets = [ev.snippets[r] for r in visited if r in ev.snippets] if self.use_dpr else []
        images = await self.images(pictures, ctx) if self.answer_images else []
        if self.gold_support:
            texts, gold_images = await self.gold(q, ctx)
            snippets = [*snippets, *(t for t in texts if t not in snippets)]
            images = [*gold_images, *(u for u in images if u not in gold_images)][:ANSWER_IMAGES]
        answer = await generate_answer(q, ctx, facts[:80], snippets, images or None)
        return finish(q, self.name, answer, pattern=pattern, match=match, evidence=ev, facts=facts[:80],
                      consulted=consulted, reused=[c.ref for c in kept], stopped=done)

    async def candidates(self, q: Question, ctx: QueryContext) -> list[Candidate]:
        k = self.k or max(self.budget, 5)
        cands = (await ctx.fed.probe([ProbeRequest(q.question, "question", k)], ledger=ctx.ledger))[0]
        if self.order == "relevance":
            (sims,) = await asyncio.to_thread(ctx.matcher.sim.relevance, [(q.question, [c.summary for c in cands])])
            score = {c.ref: float(s) for c, s in zip(cands, sims, strict=True)}
            return first_per_ref(sorted(cands, key=lambda c: (-score[c.ref], c.rank, tie(q.qid, c), c.oid)))
        return rrf(split_lists(cands), salt=q.qid)

    async def held(self, pattern: Pattern, ctx: QueryContext) -> list[Fact]:
        if not pattern.anchors:
            return []
        near = await ctx.kg.lookup(ctx.ns, pattern.anchors, limit=KEEP_LOOKUP)
        seen = {norm_entity(a) for a in pattern.anchors}
        named = []
        for name in (x for f in near for x in (f.head, f.tail)):
            key = norm_entity(name)
            if key and key not in seen:
                seen.add(key)
                named.append(name)
        far = await ctx.kg.lookup(ctx.ns, named, limit=KEEP_LOOKUP) if named else []
        return list(dict.fromkeys([*near, *far]))

    async def kept_texts(self, kept: Sequence[Candidate], ev: Evidence, ctx: QueryContext) -> None:
        policy = {s: si.policy for s, si in (await ctx.fed.info()).items()}
        refs = [c.ref for c in kept if c.modality != "image" and policy.get(c.site) == "snippets"]
        for raw in await ctx.fed.fetch_raw(refs, ledger=ctx.ledger) if refs else []:
            if raw.text and not raw.denied:
                ev.snippets[(raw.site, raw.oid)] = raw.text

    async def store(self, extracted: Sequence[ExtractionResult], ctx: QueryContext) -> None:
        results = [ExtractionResult(r.site, r.oid, r.modality, KEEP_VERSION, list(r.facts), Usage(), doc_id=r.doc_id)
                   for r in extracted if r.error is None]
        if results:
            await ctx.kg.add(ctx.ns, results)

    async def images(self, pictures: list[Ref], ctx: QueryContext) -> list[tuple[str, str]]:
        if not pictures:
            return []
        raws = {(r.site, r.oid): r for r in await ctx.fed.fetch_raw(pictures[:ANSWER_IMAGES], ledger=ctx.ledger)}
        out = []
        for ref in pictures[:ANSWER_IMAGES]:
            raw = raws.get(ref)
            uri = decode_image(raw) if raw is not None and not raw.denied else None
            if uri:
                out.append((raw.title, uri))
        return out

    async def gold(self, q: Question, ctx: QueryContext) -> tuple[list[str], list[str]]:
        oids = [oid for d in q.gold_doc_ids
                for oid in (*(f"{q.dataset}/text/{d}#{i}" for i in range(GOLD_CHUNKS)), f"{q.dataset}/image/{d}")]
        if not oids:
            return [], []
        raws = await ctx.fed.fetch_raw([(s, oid) for s in ctx.fed.site_names for oid in oids], ledger=ctx.ledger)
        texts = list(dict.fromkeys(r.text for r in raws if r.modality != "image" and not r.denied and r.text))
        images = [u for r in raws if r.modality == "image" and not r.denied and (u := decode_image(r))]
        return texts, images[:ANSWER_IMAGES]
