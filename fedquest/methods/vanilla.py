# Vanilla RAG baseline: probe with the question, answer from the top objects
from __future__ import annotations

import asyncio

from fedquest import timeline
from fedquest.methods.base import Method, MethodResult, QueryContext, register
from fedquest.methods.common import Evidence, finish, first_per_ref, generate_answer, rrf, split_lists, tie
from fedquest.site.service import decode_image
from fedquest.types import Candidate, ProbeRequest, Question


@register("vanilla_rag")
class VanillaRAG(Method):
    uses_budget = False

    def __init__(self, k: int = 5, top: int = 5, images: int = 2, budget: int | None = None,
                 order: str = "rrf") -> None:
        super().__init__()
        if order not in ("rrf", "relevance"):
            raise ValueError(f"unknown order {order!r}")
        self.k, self.top, self.images, self.budget, self.order = k, top, images, budget, order
        if budget is not None:
            self.top, self.k = int(budget), max(int(k), int(budget))

    async def ranked(self, q: Question, ctx: QueryContext, cands: list[Candidate]) -> list[Candidate]:
        if self.order == "relevance":
            (sims,) = await asyncio.to_thread(ctx.matcher.sim.relevance, [(q.question, [c.summary for c in cands])])
            score = {c.ref: float(s) for c, s in zip(cands, sims, strict=True)}
            return first_per_ref(sorted(cands, key=lambda c: (-score[c.ref], c.rank, tie(q.qid, c), c.oid)))
        return rrf(split_lists(cands), salt=q.qid)

    async def answer(self, q: Question, ctx: QueryContext) -> MethodResult:
        cands = (await ctx.fed.probe([ProbeRequest(q.question, "question", self.k)], ledger=ctx.ledger))[0]
        top = (await self.ranked(q, ctx, cands))[: self.top]
        with timeline.phase("answer"):
            shipped = await ctx.fed.fetch_raw([c.ref for c in top], ledger=ctx.ledger)
        raws = {(r.site, r.oid): r for r in shipped}
        snippets, images = [], []
        for c in top:
            r = raws.get(c.ref)
            if r is None or r.denied:
                snippets.append(c.summary)
            elif r.modality == "image":
                uri = decode_image(r)
                if uri and len(images) < self.images:
                    images.append((r.title, uri))
                snippets.append(f"[image] {r.title}")
            else:
                snippets.append(f"{r.title}\n{r.text}")
        answer = await generate_answer(q, ctx, [], snippets, images or None)
        ev = Evidence()
        ev.doc_ids.update({c.ref: c.doc_id for c in top if c.doc_id})
        return finish(q, self.name, answer, evidence=ev, consulted=[c.ref for c in top])
