# ReWOO baseline: plan every tool call up front, then fill in the evidence
from __future__ import annotations

import asyncio
import re
from typing import Any

from fedquest.llm import prompts
from fedquest.methods.base import Method, MethodResult, QueryContext, register
from fedquest.methods.common import Evidence, Ref, finish, generate_answer
from fedquest.types import Candidate, ProbeRequest, Question

PLAN_MAX_TOKENS = 600
WORKER_MAX_TOKENS = 64
_STEP = re.compile(r"^\s*#E(\d+)\s*=\s*([A-Za-z_]+)\s*\[(.*)\]\s*$")
_PLAN = re.compile(r"^\s*Plan\s*\d*\s*:\s*(.*)$", re.IGNORECASE)
_VAR = re.compile(r"#E(\d+)")
_TOOLS = {"search": "search", "wikipedia": "search", "llm": "llm"}

PLANNER_SYSTEM = """For the following task, make plans that can solve the problem step by step. For each plan, indicate which external tool together with tool input to retrieve evidence. You can store the evidence into a variable #E that can be called by later tools. (Plan, #E1, Plan, #E2, Plan, ...)

Tools can be one of the following:
(1) Search[input]: searches a federation of databases that hold passages, table rows and images (an image appears as its caption) and returns the most relevant objects for the input. Useful when you need facts about an entity or a topic. The input should be a search query, such as an entity's name and the fact you need; it may name an earlier result, like #E2.
(2) LLM[input]: a language model that answers the input from the objects that the Search steps it names returned. Useful to pick out a name, a date or a number from search results before the next search. The input should name the Search step it reads, like #E1.

Describe your plans with rich details. Each Plan should be followed by only one #E. Use at most {steps} steps. Reply with the plan only."""

PLANNER_USER = "For example,\n{demo}\n\nBegin!\nTask: {question}"

DEMOS: dict[str, str] = {
    "mmqa": (
        "Task: In the film where Ayelet Zuer plays Lara Lor-Van, who plays Lois Lane?\n"
        "Plan: Find the film in which Ayelet Zurer plays Lara Lor-Van.\n"
        "#E1 = Search[Ayelet Zurer Lara Lor-Van film]\n"
        "Plan: Name the film from the search results.\n"
        "#E2 = LLM[In which film does Ayelet Zurer play Lara Lor-Van, according to #E1?]\n"
        "Plan: Find who plays Lois Lane in that film.\n"
        "#E3 = Search[Lois Lane in #E2]\n"
        "Plan: Name the actress.\n"
        "#E4 = LLM[Who plays Lois Lane in #E2, according to #E3?]"),
    "musique": (
        "Task: Who is the city where Lesnoye Sanatorium is located named after?\n"
        "Plan: Find the city where Lesnoye Sanatorium is located.\n"
        "#E1 = Search[Lesnoye Sanatorium city]\n"
        "Plan: Name the city from the search results.\n"
        "#E2 = LLM[In which city is Lesnoye Sanatorium located, according to #E1?]\n"
        "Plan: Find whom that city is named after.\n"
        "#E3 = Search[#E2 named after]\n"
        "Plan: Name the person.\n"
        "#E4 = LLM[Whom is #E2 named after, according to #E3?]"),
}
for _name in ("hotpotqa", "twowiki"):
    DEMOS.setdefault(_name, DEMOS["musique"])
DEMOS.setdefault("m3sciqa", DEMOS["mmqa"])

WORKER_SYSTEM = ("You answer a question from the sources given, using only them. Reply with the short answer only (a "
                 "name, a date, a number or a short phrase), or unknown if the sources do not say.")
WORKER_USER = "Sources:\n{sources}\n\nQuestion: {instruction}\nShort answer:"


def parse_plan(text: str) -> list[tuple[int, str, str, str]]:
    steps: list[tuple[int, str, str, str]] = []
    plan, seen = "", set()
    for line in (text or "").splitlines():
        m = _PLAN.match(line)
        if m:
            plan = m.group(1).strip()
            continue
        m = _STEP.match(line)
        if m and int(m.group(1)) not in seen:
            seen.add(int(m.group(1)))
            steps.append((int(m.group(1)), _TOOLS.get(m.group(2).lower(), ""), m.group(3).strip(), plan))
            plan = ""
    return steps


def substitute(text: str, results: dict[int, str]) -> str:
    return _VAR.sub(lambda m: results.get(int(m.group(1)), "") or "unknown", text).strip()


def worker_messages(instruction: str, sources: list[str]) -> list[dict[str, Any]]:
    numbered = "\n".join(f"[{i}] {s.replace(chr(10), ': ', 1)}" for i, s in enumerate(sources, 1)) or "(none)"
    return [{"role": "system", "content": WORKER_SYSTEM},
            {"role": "user", "content": WORKER_USER.format(sources=numbered, instruction=instruction)}]


def short(text: str) -> str:
    line = next((x.strip() for x in (text or "").splitlines() if x.strip()), "")
    line = re.sub(r"^(short answer|answer)\s*:\s*", "", line, flags=re.IGNORECASE).strip().strip('"').strip()
    return line[:200] or "unknown"


@register("rewoo")
class ReWOO(Method):
    name = "rewoo"

    def __init__(self, max_steps: int = 8, k: int = 8, shown: int = 5, demos: int = 1, images: int = 4,
                 image_px: int = 512, total_budget: int | None = None) -> None:
        super().__init__()
        if max_steps < 1 or k < 1 or shown < 1 or demos not in (0, 1) or images < 0:
            raise ValueError(f"max_steps {max_steps}, k {k}, shown {shown}, demos {demos} (0 or 1) or images "
                             f"{images} out of range")
        if total_budget is not None and (isinstance(total_budget, bool) or int(total_budget) <= 0):
            raise ValueError(f"total_budget {total_budget!r} is not a positive number of cost units")
        self.max_steps, self.k, self.shown, self.demos = int(max_steps), int(k), int(shown), int(demos)
        self.images, self.image_px = int(images), int(image_px)
        self.total_budget = int(total_budget) if total_budget is not None else None

    def plan_messages(self, q: Question) -> list[dict[str, Any]]:
        demo = DEMOS.get(q.dataset or str(q.meta.get("dataset", "")), DEMOS["musique"]) if self.demos else ""
        user = PLANNER_USER.format(demo=demo, question=q.question) if demo else f"Task: {q.question}"
        return [{"role": "system", "content": PLANNER_SYSTEM.format(steps=self.max_steps)},
                {"role": "user", "content": user}]

    def _answer_cost(self, q: Question, ctx: QueryContext, texts: list[str], pictures: int) -> int:
        from fedquest import budget

        msgs = prompts.answer_messages(q.question, prompts.format_facts([]), texts, ctx.answer_style)
        return budget.count_messages(msgs) + budget.IMAGE_TOKENS * pictures + budget.OUT_WEIGHT * 24

    def _fits(self, ctx: QueryContext, call: list[dict[str, Any]], task: str, max_tokens: int, reserve: int) -> bool:
        from fedquest import budget

        if self.total_budget is None:
            return True
        return ctx.ledger.total_tokens + budget.call_cost(call, task, max_tokens) + reserve <= self.total_budget

    async def answer(self, q: Question, ctx: QueryContext) -> MethodResult:
        from fedquest import budget

        led = ctx.ledger
        texts: list[str] = []
        docs: list[str] = []
        pictures: list[tuple[str, str]] = []
        seen: set[Ref] = set()
        read: list[Ref] = []
        ev = Evidence()

        def reserve(extra: list[str] | None = None, pics: int | None = None) -> int:
            n = min(len(pictures), self.images) if pics is None else pics
            return self._answer_cost(q, ctx, texts + (extra or []), n)

        plan_text, ended = "", "plan"
        call = self.plan_messages(q)
        if self._fits(ctx, call, "rewoo_plan", PLAN_MAX_TOKENS, reserve()):
            try:
                led.reserve = reserve()
                plan_text = (await ctx.llm.chat(call, role="planner", task="rewoo_plan", max_tokens=PLAN_MAX_TOKENS,
                                                ledger=led, charge="planner")).text
            except budget.BudgetExhausted:
                ended = "budget"
        else:
            ended = "budget"
        steps = [s for s in parse_plan(plan_text) if s[1]][: self.max_steps]
        results: dict[int, str] = {}
        objects: dict[int, list[int]] = {}
        lines: list[str] = []
        last_search: int | None = None
        for n, tool, arg, plan in steps:
            if tool == "search":
                query = substitute(arg, results)
                got = await self.search(ctx, query, seen) if query else []
                added: list[tuple[Candidate, str, tuple[str, str] | None]] = []
                for c, t, pic in got:
                    if pic is not None and len(pictures) + sum(p is not None for *_, p in added) >= self.images:
                        pic = None
                    added.append((c, t, pic))
                while added and self.total_budget is not None:
                    extra = [t for _, t, _ in added]
                    pics = min(len(pictures) + sum(p is not None for *_, p in added), self.images)
                    if led.total_tokens + reserve(extra, pics) <= self.total_budget:
                        break
                    added.pop()
                objects[n] = []
                for c, t, pic in added:
                    seen.add(c.ref)
                    read.append(c.ref)
                    objects[n].append(len(texts))
                    texts.append(t)
                    docs.append(c.doc_id or "")
                    if pic is not None:
                        pictures.append(pic)
                    if c.doc_id:
                        ev.doc_ids[c.ref] = c.doc_id
                led.note_read([c.doc_id for c, _, _ in added if c.doc_id], "coordinator")
                # substituted into later inputs, a Search's #E is its first object's title
                results[n] = texts[objects[n][0]].split("\n", 1)[0] if objects[n] else "unknown"
                last_search = n
                lines.append(f"Plan: {plan}\n#E{n} = Search[{query}]: {len(objects[n])} objects"
                             + (f", the first {results[n]}" if objects[n] else ""))
                continue
            named = [int(k) for k in _VAR.findall(arg) if int(k) in objects]
            src = named or ([last_search] if last_search is not None else [])
            sources = [texts[i] for k in src for i in objects.get(k, [])]
            instruction = substitute(arg, results)
            call = worker_messages(instruction, sources)
            if not self._fits(ctx, call, "rewoo_worker", WORKER_MAX_TOKENS, reserve()):
                results[n], ended = "unknown", "budget"
                lines.append(f"Plan: {plan}\n#E{n} = LLM[{instruction}]: (no budget left)")
                continue
            try:
                led.reserve = reserve()
                reply = await ctx.llm.chat(call, role="reader", task="rewoo_worker", max_tokens=WORKER_MAX_TOKENS,
                                           ledger=led, charge="reader")
                results[n] = short(reply.text)
            except budget.BudgetExhausted:
                results[n], ended = "unknown", "budget"
            lines.append(f"Plan: {plan}\n#E{n} = LLM[{instruction}]: {results[n]}")
        if ended == "plan":
            ended = "steps" if steps else "no plan"
        ctx.log("rewoo", plan=plan_text[:4000], steps=len(steps), ended=ended, read=len(read),
                results={f"#E{k}": v for k, v in results.items()})
        led.reserve = 0
        solver = ["Plan and evidence:\n" + "\n".join(lines)] if lines else []
        answer = await generate_answer(q, ctx, [], solver + texts, pictures[: self.images] or None,
                                       doc_ids=[""] * len(solver) + docs)
        return finish(q, self.name, answer, evidence=ev, consulted=read)

    async def search(self, ctx: QueryContext, query: str, seen: set[Ref]
                     ) -> list[tuple[Candidate, str, tuple[str, str] | None]]:
        from fedquest.site.service import decode_image

        led = ctx.ledger
        cands = (await ctx.fed.probe([ProbeRequest(query, "question", self.k)], ledger=led))[0]
        if not cands:
            return []
        (sims,) = await asyncio.to_thread(ctx.matcher.sim.relevance, [(query, [c.summary for c in cands])])
        score = {c.ref: float(s) for c, s in zip(cands, sims, strict=True)}
        ranked = sorted({c.ref: c for c in cands}.values(), key=lambda c: (-score[c.ref], c.rank, c.oid))
        new = [c for c in ranked if c.ref not in seen][: self.shown]
        shipped = await ctx.fed.fetch_raw([c.ref for c in new], ledger=led, max_px=self.image_px) if new else []
        raws = {(r.site, r.oid): r for r in shipped}
        out = []
        for c in new:
            r = raws.get(c.ref)
            body = r.text if r is not None and not r.denied and r.text else c.summary
            title = (r.title if r is not None and r.title else "") or c.summary.split(": ", 1)[0]
            text = body[len(title) + 2:] if title and body.startswith(f"{title}: ") else body
            pic = None
            if r is not None and r.modality == "image" and not r.denied and (uri := decode_image(r)):
                pic = (title, uri)
            out.append((c, f"{title}\n{text}", pic))
        return out

    def describe(self) -> dict[str, Any]:
        return {"max_steps": self.max_steps, "k": self.k, "shown": self.shown, "demos": self.demos,
                "images": self.images, "total_budget": self.total_budget}
