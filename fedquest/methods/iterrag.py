# IRCoT baseline: retrieve, reason one sentence, retrieve again with it
from __future__ import annotations

import asyncio
import re
from typing import Any, ClassVar, Literal

from fedquest import timeline
from fedquest.llm import prompts
from fedquest.methods.base import Method, MethodResult, QueryContext, register
from fedquest.methods.common import (
    Evidence,
    Ref,
    finish,
    first_per_ref,
    generate_answer,
    is_abstention,
    ref_str,
    rrf,
    split_lists,
    tie,
)
from fedquest.site.service import decode_image
from fedquest.types import Candidate, ProbeRequest, Question

_ANSWER = re.compile(r"answer is:?\s*(.*)", re.I | re.S)
_DECLINES = re.compile(
    r"\bnot enough (?:information|context|evidence|data|details)\b|\bno (?:information|evidence|details)\b|"
    r"\b(?:i|we) (?:do|did) not (?:have|know)\b|\bi don[’']t (?:have|know)\b|"
    r"\b(?:do|does) not have (?:enough|sufficient|any) (?:information|context|evidence|details)\b|\bunanswerable\b|"
    r"\bunclear\b|\bcannot be determined\b|\bnot (?:clear|known|stated|mentioned|provided|specified|given)\b",
    re.IGNORECASE)
_STOP = re.compile(r"[.!?]+[\"'”’)\]]*")
_OPENERS = "\"'“‘(["
_ABBREV = frozenset("mr mrs ms dr st jr sr prof gen col lt sgt capt rev gov sen rep mt ft vs etc inc ltd co corp bros "
                    "jan feb mar apr jun jul aug sep sept oct nov dec".split())
_NUMBERED = frozenset({"no", "vol", "pp", "ch", "fig"})


def first_sentence(text: str) -> str:
    text = re.sub(r"^\s*A:\s*", "", text or "")
    line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    for m in _STOP.finditer(line):
        rest = line[m.end():]
        if rest and not rest[0].isspace():
            continue
        after = rest.lstrip()[:1]
        if after and not (after.isupper() or after.isdigit() or after in _OPENERS):
            continue
        if m.group().rstrip("\"'”’)]") == ".":
            found = re.search(r"(\w+)$", line[: m.start()])
            word = found.group(1).lower() if found else ""
            if (len(word) == 1 and word.isalpha()) or word in _ABBREV or (word in _NUMBERED and after.isdigit()):
                continue
        return line[: m.end()].strip()
    return line


def declines(answer: str) -> bool:
    return is_abstention(answer) or (len(answer.split()) <= 30 and bool(_DECLINES.search(answer)))


@register("iter_rag")
class IterRAG(Method):
    uses_budget = False
    added_params: ClassVar[dict[str, Any]] = {"read": "raw"}
    images = 0
    image_px = 384
    fresh = False
    image_cap = 0

    def attached(self, pictures: dict[int, str]) -> dict[int, str]:
        return dict(sorted(pictures.items())[-self.images:]) if self.fresh and pictures else pictures

    def __init__(self, hops: int = 4, k: int = 3, top: int = 4, max_paragraphs: int = 15, grounded: bool = True,
                 demos: int = 0, budget: int | None = None, order: str = "rrf",
                 read: Literal["raw", "entries"] = "raw") -> None:
        super().__init__()
        if demos < 0:
            raise ValueError(f"demos must be >= 0, not {demos}")
        if order not in ("rrf", "relevance"):
            raise ValueError(f"unknown order {order!r}")
        if read not in ("raw", "entries"):
            raise ValueError(f"unknown read {read!r}")
        self.read = read
        self.hops, self.k, self.top, self.max_paragraphs, self.grounded = hops, k, top, max_paragraphs, grounded
        self.demos, self.budget, self.order = demos, budget, order
        if budget is not None:
            self.top, self.k = int(budget), max(int(k), int(budget))
            self.max_paragraphs = max(self.max_paragraphs, self.hops * self.top)

    def demonstrations(self, q: Question) -> tuple[str, ...]:
        if not self.demos:
            return ()
        dataset = q.dataset or str(q.meta.get("dataset", ""))
        return prompts.IRCOT_DEMOS.get(dataset, ())[: self.demos]

    async def ranked(self, q: Question, query: str, cands: list[Candidate], ctx: QueryContext) -> list[Candidate]:
        if self.order == "relevance":
            (sims,) = await asyncio.to_thread(ctx.matcher.sim.relevance, [(query, [c.summary for c in cands])])
            score = {c.ref: float(s) for c, s in zip(cands, sims, strict=True)}
            return first_per_ref(sorted(cands, key=lambda c: (-score[c.ref], c.rank, tie(q.qid, c), c.oid)))
        return rrf(split_lists(cands), salt=q.qid)

    async def answer(self, q: Question, ctx: QueryContext) -> MethodResult:
        paragraphs: list[tuple[str, str]] = []
        thoughts: list[str] = []
        steps: list[tuple[str, str]] = []
        seen: set[Ref] = set()
        demos = self.demonstrations(q)
        query, depth, grew = q.question, self.k, True
        answer, wrote = "", False
        ev, read = Evidence(), []
        pictures: dict[int, str] = {}
        shown: list[Ref] = []
        px = {"max_px": self.image_px} if self.images else {}
        cap = self.image_cap if self.fresh else self.images
        sends = 0
        for _ in range(self.hops):
            cands = (await ctx.fed.probe([ProbeRequest(query, "question", depth)], ledger=ctx.ledger))[0]
            room = max(0, self.max_paragraphs - len(paragraphs))
            new = [c for c in await self.ranked(q, query, cands, ctx) if c.ref not in seen][: min(self.top, room)]
            if not new and not grew:
                break
            seen |= {c.ref for c in new}
            read += [c.ref for c in new]
            ev.doc_ids.update({c.ref: c.doc_id for c in new if c.doc_id})
            with timeline.phase("read"):
                raw = bool(new) and self.read == "raw"
                shipped = await ctx.fed.fetch_raw([c.ref for c in new], ledger=ctx.ledger, **px) if raw else []
            raws = {(r.site, r.oid): r for r in shipped}
            for c in new:
                r = raws.get(c.ref)
                text = r.text if r is not None and not r.denied and r.text else c.summary
                if (len(pictures) < cap and r is not None and r.modality == "image" and not r.denied
                        and (uri := decode_image(r))):
                    pictures[len(paragraphs)] = uri
                    shown.append(c.ref)
                title = (r.title if r is not None and r.title else "") or c.summary.split(": ", 1)[0]
                body = text[len(title) + 2:] if title and text.startswith(f"{title}: ") else text
                paragraphs.append((title, body))
            if pictures:
                now = self.attached(pictures)
                sends += len(now)
                messages = prompts.ircot_image_messages(q.question, paragraphs, thoughts, now, demos)
            else:
                messages = (prompts.ircot_demo_messages(q.question, paragraphs, thoughts, demos) if demos
                            else prompts.ircot_messages(q.question, paragraphs, thoughts))
            reply = await ctx.llm.chat(messages, role="answer_vision" if pictures else "reader", task="ircot",
                                       max_tokens=128, ledger=ctx.ledger, charge="reader")
            sentence = first_sentence(reply.text)
            found = _ANSWER.search(sentence)
            said = found.group(1).strip().rstrip(".").strip() if found else ""
            if found and not declines(said):
                steps.append((sentence, "answer"))
                answer, wrote = said, True
                break
            # a new thought is the next query; a repeat, a decline or nothing deepens the last probe by k
            grew = bool(sentence) and not found and sentence not in thoughts
            if grew:
                thoughts.append(sentence)
                query, depth = sentence, self.k
                steps.append((sentence, "chain"))
            else:
                depth += self.k
                steps.append((sentence, "declined" if found else "repeated" if sentence else "empty"))
        for text, kind in steps:
            ctx.log("thought", text=text, kind=kind)
        final = self.attached(pictures)
        grounded = self.grounded or not answer
        logged: dict[str, Any] = {"images": [ref_str(ref) for ref in shown]} if self.images else {}
        if self.fresh:
            logged["image_sends"] = sends + (len(final) if grounded else 0)
        ctx.log("reader_answer", text=answer, steps=len(steps), answer_is=wrote, demos=len(demos),
                declined=sum(kind == "declined" for _, kind in steps), **logged)
        if grounded:
            answer = await generate_answer(q, ctx, [], [f"{t}\n{x}" for t, x in paragraphs],
                                           [(paragraphs[i][0], uri) for i, uri in final.items()] or None)
        return finish(q, self.name, answer, evidence=ev, consulted=read)


@register("iter_rag_mm")
class IterRAGImages(IterRAG):
    name = "iter_rag_mm"
    added_params: ClassVar[dict[str, Any]] = {**IterRAG.added_params, "fresh": False, "image_cap": None}

    def __init__(self, hops: int = 4, k: int = 3, top: int = 4, max_paragraphs: int = 15, grounded: bool = True,
                 demos: int = 0, budget: int | None = None, order: str = "rrf", images: int = 4,
                 image_px: int = 512, fresh: bool = False, image_cap: int | None = None) -> None:
        super().__init__(hops, k, top, max_paragraphs, grounded, demos, budget, order)
        if images < 1:
            raise ValueError(f"images must be >= 1, not {images} (iter_rag reads captions only)")
        if image_px < 1:
            raise ValueError(f"image_px must be >= 1, not {image_px}")
        cap = images if image_cap is None else int(image_cap)
        if cap < images:
            raise ValueError(f"image_cap {image_cap} is below images {images}")
        self.images, self.image_px = int(images), int(image_px)
        self.fresh, self.image_cap = bool(fresh), cap


class _Budgeted:
    total_budget: int | None = None

    def _answer_cost(self, q: Question, ctx: QueryContext, paragraphs: list[tuple[str, str]], images: int) -> int:
        from fedquest import budget

        msgs = prompts.answer_messages(q.question, prompts.format_facts([]), [f"{t}\n{x}" for t, x in paragraphs],
                                       ctx.answer_style)
        return budget.count_messages(msgs) + budget.IMAGE_TOKENS * images + budget.OUT_WEIGHT * 24

    async def answer(self, q: Question, ctx: QueryContext) -> MethodResult:
        if self.total_budget is None:
            return await super().answer(q, ctx)  # type: ignore[misc]
        from fedquest import budget

        led, cap_b = ctx.ledger, int(self.total_budget)
        paragraphs: list[tuple[str, str]] = []
        docs: list[str] = []
        thoughts: list[str] = []
        steps: list[tuple[str, str]] = []
        seen: set[Ref] = set()
        demos = self.demonstrations(q)  # type: ignore[attr-defined]
        query, depth, grew = q.question, self.k, True  # type: ignore[attr-defined]
        answer, wrote, why = "", False, "steps"
        ev, read = Evidence(), []
        pictures: dict[int, str] = {}
        shown: list[Ref] = []
        px = {"max_px": self.image_px} if self.images else {}  # type: ignore[attr-defined]
        cap = self.image_cap if self.fresh else self.images  # type: ignore[attr-defined]
        sends = 0

        def build() -> list[dict[str, Any]]:
            if pictures:
                return prompts.ircot_image_messages(q.question, paragraphs, thoughts, self.attached(pictures), demos)
            return (prompts.ircot_demo_messages(q.question, paragraphs, thoughts, demos) if demos
                    else prompts.ircot_messages(q.question, paragraphs, thoughts))

        for _ in range(self.hops):  # type: ignore[attr-defined]
            try:
                cands = (await ctx.fed.probe([ProbeRequest(query, "question", depth)], ledger=led))[0]
            except budget.BudgetExhausted:
                why = "budget"
                break
            room = max(0, self.max_paragraphs - len(paragraphs))  # type: ignore[attr-defined]
            new = [c for c in await self.ranked(q, query, cands, ctx)  # type: ignore[attr-defined]
                   if c.ref not in seen][: min(self.top, room)]  # type: ignore[attr-defined]
            if not new and not grew:
                why = "nothing new"
                break
            with timeline.phase("read"):
                raw = bool(new) and self.read == "raw"  # type: ignore[attr-defined]
                shipped = await ctx.fed.fetch_raw([c.ref for c in new], ledger=led, **px) if raw else []
            raws = {(r.site, r.oid): r for r in shipped}
            start = len(paragraphs)
            added: list[Any] = []
            for c in new:
                r = raws.get(c.ref)
                text = r.text if r is not None and not r.denied and r.text else c.summary
                if (len(pictures) < cap and r is not None and r.modality == "image" and not r.denied
                        and (uri := decode_image(r))):
                    pictures[len(paragraphs)] = uri
                    shown.append(c.ref)
                title = (r.title if r is not None and r.title else "") or c.summary.split(": ", 1)[0]
                body = text[len(title) + 2:] if title and text.startswith(f"{title}: ") else text
                paragraphs.append((title, body))
                docs.append(c.doc_id or "")
                added.append(c)
            messages = build()
            # drop this hop's newest paragraphs until its step and the final answer both fit the budget
            while True:
                need = budget.call_cost(messages, "ircot", 128) + self._answer_cost(
                    q, ctx, paragraphs, len(self.attached(pictures)))
                if led.total_tokens + need <= cap_b:
                    break
                if len(paragraphs) <= start:
                    break
                i = len(paragraphs) - 1
                paragraphs.pop()
                docs.pop()
                added.pop()
                if i in pictures:
                    del pictures[i]
                    shown.pop()
                messages = build()
            if len(paragraphs) <= start and new:
                why = "budget"
                break
            seen |= {c.ref for c in added}
            read += [c.ref for c in added]
            ev.doc_ids.update({c.ref: c.doc_id for c in added if c.doc_id})
            led.note_read([c.doc_id for c in added if c.doc_id], "coordinator")
            led.reserve = self._answer_cost(q, ctx, paragraphs, len(self.attached(pictures)))
            if pictures:
                sends += len(self.attached(pictures))
            try:
                reply = await ctx.llm.chat(messages, role="answer_vision" if pictures else "reader", task="ircot",
                                           max_tokens=128, ledger=led, charge="reader")
            except budget.BudgetExhausted:
                why = "budget"
                break
            sentence = first_sentence(reply.text)
            found = _ANSWER.search(sentence)
            said = found.group(1).strip().rstrip(".").strip() if found else ""
            if found and not declines(said):
                steps.append((sentence, "answer"))
                answer, wrote, why = said, True, "answered"
                break
            grew = bool(sentence) and not found and sentence not in thoughts
            if grew:
                thoughts.append(sentence)
                query, depth = sentence, self.k  # type: ignore[attr-defined]
                steps.append((sentence, "chain"))
            else:
                depth += self.k  # type: ignore[attr-defined]
                steps.append((sentence, "declined" if found else "repeated" if sentence else "empty"))
        for text, kind in steps:
            ctx.log("thought", text=text, kind=kind)
        final = self.attached(pictures)
        grounded = self.grounded or not answer  # type: ignore[attr-defined]
        logged: dict[str, Any] = {"images": [ref_str(ref) for ref in shown]} if self.images else {}
        if self.fresh:  # type: ignore[attr-defined]
            logged["image_sends"] = sends + (len(final) if grounded else 0)
        ctx.log("reader_answer", text=answer, steps=len(steps), answer_is=wrote, demos=len(demos), stop=why,
                paragraphs=len(paragraphs),
                declined=sum(kind == "declined" for _, kind in steps), **logged)
        led.reserve = 0
        if grounded:
            answer = await generate_answer(q, ctx, [], [f"{t}\n{x}" for t, x in paragraphs],
                                           [(paragraphs[i][0], uri) for i, uri in final.items()] or None,
                                           doc_ids=docs)
        return finish(q, self.name, answer, evidence=ev, consulted=read)  # type: ignore[attr-defined]


@register("ircot_b")
class IterRAGBudget(_Budgeted, IterRAG):
    name = "ircot_b"

    def __init__(self, hops: int = 8, k: int = 8, top: int = 8, max_paragraphs: int = 15, grounded: bool = True,
                 demos: int = 3, order: str = "relevance", read: Literal["raw", "entries"] = "raw",
                 total_budget: int | None = None) -> None:
        super().__init__(hops=hops, k=k, top=top, max_paragraphs=max_paragraphs, grounded=grounded, demos=demos,
                         order=order, read=read)
        if total_budget is not None and int(total_budget) <= 0:
            raise ValueError(f"total_budget {total_budget!r} is not positive")
        self.total_budget = int(total_budget) if total_budget is not None else None


@register("ircot_mm_b")
class IterRAGImagesBudget(_Budgeted, IterRAGImages):
    name = "ircot_mm_b"

    def __init__(self, hops: int = 8, k: int = 8, top: int = 8, max_paragraphs: int = 15, grounded: bool = True,
                 demos: int = 3, order: str = "relevance", images: int = 4, image_px: int = 512, fresh: bool = False,
                 image_cap: int | None = None, total_budget: int | None = None) -> None:
        super().__init__(hops=hops, k=k, top=top, max_paragraphs=max_paragraphs, grounded=grounded, demos=demos,
                         order=order, images=images, image_px=image_px, fresh=fresh, image_cap=image_cap)
        if total_budget is not None and int(total_budget) <= 0:
            raise ValueError(f"total_budget {total_budget!r} is not positive")
        self.total_budget = int(total_budget) if total_budget is not None else None
