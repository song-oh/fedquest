from __future__ import annotations

from fedquest import timeline
from fedquest.methods.base import Method, MethodResult, QueryContext, register
from fedquest.methods.common import Evidence, finish, generate_answer, rrf, split_lists
from fedquest.types import HippoRequest, Question


@register("hipporag2")
class HippoRAG2(Method):
    def __init__(self, k: int = 5, top: int = 5) -> None:
        super().__init__()
        self.k, self.top = k, top

    async def answer(self, q: Question, ctx: QueryContext) -> MethodResult:
        cands = (await ctx.fed.hippo([HippoRequest(q.question, self.k)], ledger=ctx.ledger))[0]
        if not cands:
            raise RuntimeError("no site returned a HippoRAG 2 ranking: are the text collections indexed? "
                               "(fedquest hippo index <federation>)")
        top = rrf(split_lists(cands), salt=q.qid)[: self.top]
        with timeline.phase("answer"):
            shipped = await ctx.fed.fetch_raw([c.ref for c in top], ledger=ctx.ledger)
        raws = {(r.site, r.oid): r for r in shipped}
        snippets = [c.summary if (r := raws.get(c.ref)) is None or r.denied else f"{r.title}\n{r.text}" for c in top]
        answer = await generate_answer(q, ctx, [], snippets, None)
        ev = Evidence()
        ev.doc_ids.update({c.ref: c.doc_id for c in top if c.doc_id})
        return finish(q, self.name, answer, evidence=ev, consulted=[c.ref for c in top])
