# closed-book baseline: the answer model with no retrieval
from __future__ import annotations

from fedquest.llm import prompts
from fedquest.llm.parsing import parse_answer
from fedquest.methods.base import Method, MethodResult, QueryContext, register
from fedquest.methods.common import finish
from fedquest.types import Question


@register("closed_book")
class ClosedBook(Method):
    uses_budget = False

    async def answer(self, q: Question, ctx: QueryContext) -> MethodResult:
        reply = await ctx.llm.chat(prompts.closed_book_messages(q.question, ctx.answer_style), role="answer",
                                   task="closed_book", max_tokens=128 if ctx.answer_style == "short" else 400,
                                   ledger=ctx.ledger, charge="answer")
        return finish(q, self.name, parse_answer(reply.text))
