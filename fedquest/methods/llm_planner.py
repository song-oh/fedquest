from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

from fedquest.kg.normalize import norm_entity, same_entity
from fedquest.llm import prompts
from fedquest.llm.prompts import ReadUnknown
from fedquest.methods.base import Method, MethodResult, QueryContext, register
from fedquest.methods.common import (
    Evidence,
    Ref,
    finish,
    generate_answer,
    key_text,
    ref_str,
    rrf,
    split_lists,
)
from fedquest.types import Candidate, Fact, ProbeRequest, Question, ReadResult
from fedquest.utils import approx_tokens

MAX_SUMMARIES = 15
SHOWN = 8
OPS_CAP = 3
LIST_CAP = 24
SIBLING_K = 24
PICTURES = 2
OPS_FACTS = 60
OPS_SNIPPETS = 24
OPS_CONTEXT = 1500
OPS_OBJECTS = 24
OPS_MAX_TOKENS = 400
FRAME = 8
OUT_EMPTY = 6
OUT_HIT = 26
_FIELD = {"search": "text", "probe": "value", "title": "name", "rows": "table", "read": "ids",
          "check_pictures": "candidates", "finish": "answer"}
_ALIASES = {"check_picture": "check_pictures", "picture_check": "check_pictures", "lookup": "title",
            "table_rows": "rows"}
_X = re.compile(r"\??[xX]")
_LEADING_X = re.compile(r"^\??X(?:\s*[:,]\s*|\s+|$)")


@register("llm_planner_ops", tools="fedquest", budget=None, max_steps=12, k=5)
class LLMPlanner(Method):
    added_params: ClassVar[dict[str, Any]] = {"views": False, "total_budget": None}

    def __init__(self, budget: int | None = 5, max_steps: int = 10, k: int = 3, grounded: bool = True,
                 tools: str = "search", tokens: int | None = 8000, demos: int = 0, prompt: str = "v13",
                 views: bool = False, total_budget: int | None = None) -> None:
        super().__init__()
        if total_budget is not None and (isinstance(total_budget, bool) or int(total_budget) <= 0):
            raise ValueError(f"total_budget {total_budget!r} is not a positive number of cost units")
        if total_budget is not None and tools != "fedquest":
            raise ValueError("total_budget is the operator agent's (tools 'fedquest'); the plain agent is ReAct")
        self.total_budget = int(total_budget) if total_budget is not None else None
        if tools not in ("search", "fedquest") or (tools == "search" and budget is None):
            raise ValueError(f"unknown tools {tools!r}, or tools 'search' without an extraction budget")
        if prompt not in ("v13", "v14") or demos < 0:
            raise ValueError(f"unknown prompt {prompt!r} or demos {demos!r}")
        self.budget, self.max_steps, self.k, self.grounded = budget, max_steps, k, grounded
        self.tools, self.tokens, self.demos, self.prompt = tools, tokens, demos, prompt
        if not isinstance(views, bool):
            raise ValueError(f"views {views!r} is not true or false")
        self.views = views

    def demonstrations(self, q: Question) -> tuple[str, ...]:
        if not self.demos:
            return ()
        sets = prompts.PLANNER_OPS_DEMOS if self.tools == "fedquest" else prompts.PLANNER_DEMOS
        return sets.get(q.dataset or str(q.meta.get("dataset", "")), ())[: self.demos]

    async def answer(self, q: Question, ctx: QueryContext) -> MethodResult:
        if self.tools == "fedquest":
            return await _Ops(self, q, ctx).run()
        assert self.budget is not None
        transcript: list[dict[str, str]] = []
        listed: dict[str, Candidate] = {}
        ev = Evidence()
        consulted = []
        answer, steps, finished = "", 0, False
        demos = self.demonstrations(q)
        for _ in range(self.max_steps):
            left = self.budget - len(consulted)
            messages = (prompts.planner_messages(q.question, transcript, left) if self.prompt == "v13" and not demos
                        else prompts.agent_messages(prompts.PLANNER_SYSTEM, q.question, transcript,
                                                    prompts.planner_budget(left), demos, self.prompt))
            obj, _reply = await ctx.llm.chat_json(
                messages, role="planner", task="planner", max_tokens=300, ledger=ctx.ledger, charge="planner",
                default={},
            )
            steps += 1
            action = str(obj.get("action", "")).lower() if isinstance(obj, dict) else ""
            arg = str(obj.get("input", "")).strip() if isinstance(obj, dict) else ""
            transcript.append({"role": "assistant", "content": f"{action}({arg})"})
            if action == "finish":
                answer, finished = arg, True
                break
            if action == "search" and arg:
                cands = (await ctx.fed.probe([ProbeRequest(arg, "question", self.k)], ledger=ctx.ledger))[0]
                lines = []
                for c in rrf(split_lists(cands), salt=q.qid)[:8]:
                    cid = str(len(listed) + 1)
                    listed[cid] = c
                    lines.append(f"[{cid}] {c.site} | {c.modality} | {c.summary[:200]}")
                observation = "\n".join(lines) or "no results"
            elif action == "extract" and arg.strip("[]#") in listed:
                if len(consulted) >= self.budget:
                    observation = "budget exhausted: finish now"
                else:
                    c = listed[arg.strip("[]#")]
                    results = await ctx.fed.extract([c.ref], mode="kg", ledger=ctx.ledger)
                    for r in results:
                        ev.add_result(r)
                    consulted.append(c.ref)
                    facts = [f for r in results for f in r.facts]
                    observation = "\n".join(f"({f.head} | {f.relation} | {f.tail})" for f in facts[:25]) or "no facts"
            else:
                observation = "invalid action: use search, extract or finish"
            transcript.append({"role": "observation", "content": observation})
        for t in transcript:
            ctx.log("step", **t)
        ended = "finish" if finished else "budget" if len(consulted) >= self.budget else "cap"
        ctx.log("planner_answer", text=answer, steps=steps, ended=ended, demos=len(demos))
        if self.grounded or not answer:
            shown = list(dict.fromkeys(c.summary[:200] for c in listed.values()))[:MAX_SUMMARIES] if self.grounded else []
            answer = await generate_answer(q, ctx, ev.facts[:60], shown)
        return finish(q, self.name, answer, evidence=ev, facts=ev.facts[:60], consulted=consulted)


@dataclass(frozen=True)
class Condition:
    phrase: str
    value: str | None
    head: str | None
    relation: str
    tail: str | None

    def fact(self, entity: str, ref: Ref, modality: str) -> Fact:
        if self.value is None:
            return Fact(entity, self.relation, "yes", ref[0], ref[1], prompts.READ_VERSION, modality)
        return Fact(self.head or entity, self.relation, self.tail or entity, ref[0], ref[1], prompts.READ_VERSION,
                    modality)

    def key(self) -> str:
        return self.relation if self.value is None else key_text(self.value, self.relation, self.head == self.value)


def condition(text: str) -> Condition:
    t = " ".join(str(text).split())
    body = t[1:-1].strip() if t.startswith("(") and t.endswith(")") else t
    parts = [p.strip() for p in body.split("|")]
    if len(parts) == 3 and all(parts):
        head, rel, tail = parts
        hx, tx = _X.fullmatch(head) is not None, _X.fullmatch(tail) is not None
        if tx and not hx:
            return Condition(f"({head} | {rel} | X)", head, head, rel, None)
        if hx and not tx:
            return Condition(f"(X | {rel} | {tail})", tail, None, rel, tail)
    phrase = _LEADING_X.sub("", t, count=1).strip() or t
    return Condition(f"X {phrase}", None, None, phrase, None)


def _action(obj: Any) -> tuple[str, dict[str, Any]]:
    if not isinstance(obj, dict):
        return "", {}
    action = re.sub(r"[\s-]+", "_", str(obj.get("action", "")).strip().lower())
    action = _ALIASES.get(action, action)
    raw = obj.get("input")
    args = {k: v for k, v in obj.items() if k not in ("thought", "action", "input")}
    if isinstance(raw, dict):
        args.update(raw)
    elif raw not in (None, "", [], {}) and action in _FIELD:
        args[_FIELD[action]] = raw
    return action, args


def _text(args: dict[str, Any], *names: str) -> str:
    for n in names:
        v = args.get(n)
        if v is not None and not isinstance(v, dict | list) and str(v).strip():
            return str(v).strip()
    return ""


def _ids(raw: Any) -> list[str]:
    if isinstance(raw, str):
        items: list[Any] = re.findall(r"\d+", raw)
    else:
        items = raw if isinstance(raw, list) else [] if raw is None else [raw]
    ids = [str(x).strip().strip("[]#").strip() for x in items if x is not None]
    return [str(int(i)) if i.isdigit() else i for i in ids]


def _names(raw: Any) -> list[str]:
    items = raw if isinstance(raw, list) else str(raw).split(";") if raw is not None else []
    return list(dict.fromkeys(str(x).strip() for x in items if x is not None and str(x).strip()))


def _flag(raw: Any, default: bool = False) -> bool:
    if raw is None or raw == "":
        return default
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() not in ("false", "no", "0", "none", "null")


def _norm(value: str) -> str:
    return " ".join(str(value).split()).casefold()


def _body(c: Candidate) -> str:
    return c.summary.split(": ", 1)[1] if ": " in c.summary else c.summary


def _table(c: Candidate) -> str | None:
    parts = _body(c).split(" | ")
    return " | ".join(parts[:2]) if c.modality == "table" and len(parts) >= 3 else None


def _asked(u: ReadUnknown) -> tuple[Any, ...]:
    return (tuple(u.conditions), u.mode, u.cap, u.answer)


def _conditions(args: dict[str, Any]) -> list[Condition]:
    raw = args.get("condition", args.get("conditions"))
    items = raw if isinstance(raw, list) else [raw] if raw is not None else []
    return [condition(str(t)) for t in items if t is not None and str(t).strip()]


def _message_text(messages: list[dict[str, Any]]) -> str:
    return "\n".join(p.get("text", "") if isinstance(p, dict) else str(p) for m in messages
                     for p in (m["content"] if isinstance(m["content"], list) else [m["content"]]))


def _cost(cands: Sequence[Candidate], setup: float, cap: int, skipped: Collection[Ref] = frozenset(),
          out_empty: float = OUT_EMPTY, out_hit: float = OUT_HIT) -> float:
    groups: dict[tuple[str, str], list[Candidate]] = defaultdict(list)
    for c in cands:
        if c.ref not in skipped:
            groups[(c.site, c.modality)].append(c)
    reply = 4 * (out_empty + out_hit * min(cap, OPS_CAP))
    per_object = 4 * out_hit if cap > OPS_CAP else 0
    total = 0.0
    for (_, modality), group in groups.items():
        per_call = prompts.PER_CALL.get(modality, 1)
        for i in range(0, len(group), per_call):
            total += setup + reply + sum(c.tokens + FRAME + per_object for c in group[i : i + per_call])
    return total


def _context(texts: Sequence[str], tokens: int = OPS_CONTEXT, items: int = OPS_SNIPPETS) -> list[str]:
    out, used = [], 0
    for t in texts:
        n = approx_tokens(t)
        if out and (used + n > tokens or len(out) >= items):
            break
        out.append(t)
        used += n
    return out


class _Ops:
    def __init__(self, m: LLMPlanner, q: Question, ctx: QueryContext) -> None:
        self.m, self.q, self.ctx = m, q, ctx
        self.listed: dict[str, Candidate] = {}
        self.ids: dict[Ref, str] = {}
        self.labels: dict[Ref, str] = {}
        self.unnamed: dict[Ref, set[str]] = defaultdict(set)
        self.current: list[str] = []
        self.facts: list[Fact] = []
        self.texts: dict[Ref, str] = {}
        self.consulted: list[Ref] = []
        self.doc_ids: dict[Ref, str] = {}
        self.spent = 0
        self.judged: set[tuple[Ref, tuple[Any, ...]]] = set()
        self.bindings: dict[str, str] = {}
        self.transcript: list[dict[str, str]] = []
        self.sovereign: set[str] = set()
        self.pictures: list[str] = []
        self.tables: list[str] = []
        cal = ctx.calib
        self.out_empty = cal.read_cost("out_empty", OUT_EMPTY)
        self.out_hit = cal.read_cost("out_hit", OUT_HIT)

    def read_tokens(self) -> int:
        planner = self.ctx.ledger.coordinator.get("planner")
        return self.ctx.ledger.evidence_tokens - (planner.cost_units if planner is not None else 0)

    def answer_reserve(self) -> int:
        from fedquest.methods.fedquest import ANSWER_FRAME

        return ANSWER_FRAME + OPS_CONTEXT

    def left(self) -> int | None:
        out = None if self.m.tokens is None else max(0, self.m.tokens - self.read_tokens())
        if self.m.total_budget is not None:
            total = max(0, self.m.total_budget - self.ctx.ledger.total_tokens - self.answer_reserve())
            out = total if out is None else min(out, total)
        return out

    def objects_left(self) -> int:
        return max(0, OPS_OBJECTS - self.spent)

    def no_room(self, cut: str | None) -> str:
        if cut == "object cap":
            return f"object cap reached ({OPS_OBJECTS} objects read): finish now"
        left = self.left()
        if left is not None and left <= 0:
            return "read budget exhausted: finish now"
        return f"the read budget left ({left} tokens) cannot pay for this read: read fewer or smaller objects, or finish"

    def skips(self, c: Candidate, prefilter: Sequence[str]) -> bool:
        return bool(prefilter) and c.modality != "image" and all(
            _norm(v) in self.unnamed.get(c.ref, ()) for v in prefilter)

    def fit(self, cands: Sequence[Candidate], unknown: ReadUnknown, prefilter: Sequence[str],
            ) -> tuple[list[Candidate], str | None]:
        left, room = self.left(), self.objects_left()
        setup = approx_tokens(_message_text(prompts.read_messages(self.q.question, [], [unknown])))
        skipped = {c.ref for c in cands if self.skips(c, prefilter)}
        out: list[Candidate] = []
        for c in cands:
            if c.ref not in skipped and sum(x.ref not in skipped for x in out) >= room:
                return out, "object cap"
            if left is not None and _cost([*out, c], setup, unknown.cap, skipped, self.out_empty, self.out_hit) > left:
                return out, "read budget"
            out.append(c)
        return out, None

    async def run(self) -> MethodResult:
        info = await self.ctx.fed.info()
        self.sovereign = {s for s, si in info.items() if si.policy in ("facts", "sealed")}
        self.pictures = sorted({c.name for si in info.values() for c in si.collections if c.modality == "image"})
        self.tables = sorted({c.name for si in info.values() for c in si.collections if c.modality == "table"})
        from fedquest import budget as _budget

        answer, finished, steps, out_of_budget = "", False, 0, False
        demos = self.m.demonstrations(self.q)
        led = self.ctx.ledger
        if self.m.total_budget is not None:
            led.reserve = self.answer_reserve()
        for _ in range(self.m.max_steps):
            if self.m.prompt == "v13" and not demos:
                messages = prompts.planner_ops_messages(self.q.question, self.transcript, self.left(),
                                                        self.objects_left())
            else:
                messages = prompts.agent_messages(prompts.PLANNER_OPS_SYSTEM, self.q.question, self.transcript,
                                                  prompts.ops_budget(self.left(), self.objects_left()), demos,
                                                  self.m.prompt)
            if self.m.total_budget is not None and led.total_tokens + _budget.call_cost(
                    messages, "planner", OPS_MAX_TOKENS) + self.answer_reserve() > self.m.total_budget:
                out_of_budget = True
                break
            try:
                obj, _reply = await self.ctx.llm.chat_json(
                    messages, role="planner", task="planner_ops", max_tokens=OPS_MAX_TOKENS, ledger=led,
                    charge="planner", default={},
                )
            except _budget.BudgetExhausted:
                out_of_budget = True
                break
            steps += 1
            action, args = _action(obj)
            self.transcript.append({"role": "assistant",
                                    "content": f"{action}({json.dumps(args, ensure_ascii=False)})"})
            if action == "finish":
                answer, finished = _text(args, "answer"), True
                break
            try:
                observation = await self.act(action, args)
            except _budget.BudgetExhausted:
                out_of_budget = True
                break
            self.transcript.append({"role": "observation", "content": observation})
        for t in self.transcript:
            self.ctx.log("step", **t)
        planner = self.ctx.ledger.coordinator.get("planner")
        left = self.left()
        spent = out_of_budget or (left is not None and left <= 0) or self.objects_left() <= 0
        led.reserve = 0
        self.ctx.log("planner_answer", text=answer, read_tokens=self.read_tokens(),
                     planner_tokens=planner.cost_units if planner is not None else 0, objects_read=self.spent,
                     steps=steps, ended="finish" if finished else "budget" if spent else "cap", demos=len(demos))
        facts = list(dict.fromkeys(self.facts))[:OPS_FACTS]
        if self.m.grounded or not answer:
            answer = await generate_answer(self.q, self.ctx, facts, self.context(facts) if self.m.grounded else [])
        ev = Evidence()
        ev.doc_ids.update(self.doc_ids)
        return finish(self.q, self.m.name, answer, evidence=ev, facts=facts, consulted=self.consulted,
                      stopped=finished, bindings=self.bindings)

    async def act(self, action: str, args: dict[str, Any]) -> str:
        if action == "search" and (text := _text(args, "text", "query")):
            return await self.listing(ProbeRequest(text, "question", self.m.k, mentions=self.mentions()))
        if action == "probe" and (value := _text(args, "value")) and (relation := _text(args, "relation")):
            key = key_text(value, relation, _flag(args.get("value_first"), True))
            return await self.listing(ProbeRequest(key, "bound", self.m.k, mentions=self.mentions(value), value=value))
        if action == "title" and (name := _text(args, "name", "title")):
            relation = _text(args, "relation")
            key = key_text(name, relation, _flag(args.get("value_first"), True)) if relation else name
            return await self.listing(ProbeRequest(key, "title", self.m.k, titles=[name], mentions=self.mentions(name),
                                                   value=name))
        if action == "rows":
            return await self.rows(args.get("table", args.get("id", args.get("ids"))))
        if action == "read" and (conds := _conditions(args)):
            return await self.read(_ids(args.get("ids", args.get("id"))), conds, many=_flag(args.get("list")),
                                   final=_flag(args.get("final")))
        if action == "check_pictures" and (conds := _conditions(args)):
            return await self.check(_names(args.get("candidates")), conds)
        return "invalid action: use search, probe, title, rows, read, check_pictures or finish with their inputs"

    def mentions(self, value: str | None = None) -> list[str]:
        return list(dict.fromkeys(v for v in (value, *self.current) if v))

    def note(self, cands: Sequence[Candidate], mentions: Sequence[str]) -> None:
        values = {_norm(v) for v in mentions}
        for c in cands:
            if values and c.mentions is False:
                self.unnamed[c.ref] |= values

    async def listing(self, request: ProbeRequest) -> str:
        if self.m.views and request.key:
            request.views = True
        cands = (await self.ctx.fed.probe([request], ledger=self.ctx.ledger))[0]
        self.note(cands, request.mentions)
        return self.show(rrf(split_lists(cands), salt=self.q.qid)[:SHOWN]) or "no results"

    def table_of(self, raw: Any) -> str | None:
        if isinstance(raw, list):
            raw = raw[0] if raw else None
        if raw is None or isinstance(raw, dict):
            return None
        text = " ".join(str(raw).split())
        cid = text.strip("[]#").strip()
        cid = str(int(cid)) if cid.isdigit() else cid
        if cid in self.listed:
            return _table(self.listed[cid])
        parts = [p.strip() for p in (text.split(" | ") if " | " in text else text.split(" — ", 1))]
        return " | ".join(parts[:2]) if len(parts) >= 2 and all(parts[:2]) else None

    async def rows(self, raw: Any) -> str:
        head = self.table_of(raw)
        if head is None:
            return 'no table: give the id of a listed table row, or its table as "page | table"'
        if not self.tables:
            return "no site holds table rows"
        page = head.split(" | ", 1)[0]
        request = ProbeRequest(head, "bound", SIBLING_K, collections=self.tables, mentions=self.mentions(page),
                               value=page)
        cands = (await self.ctx.fed.probe([request], ledger=self.ctx.ledger))[0]
        self.note(cands, request.mentions)
        mine = [c for c in cands if _norm(_body(c)).startswith(f"{_norm(head)} | ")]
        return self.show(rrf(split_lists(mine), salt=self.q.qid)) or f"no rows of {head} found"

    def show(self, cands: Sequence[Candidate]) -> str:
        lines = []
        for c in cands:
            cid = self.ids.get(c.ref)
            if cid is None:
                cid = self.ids[c.ref] = str(len(self.listed) + 1)
                self.listed[cid] = c
            if c.label:
                self.labels[c.ref] = c.label
            found = f" (found as {c.label})" if c.label else ""
            lines.append(f"[{cid}] {c.site} | {c.modality} | {c.summary[:200]}{found}")
        return "\n".join(lines)

    def tag(self, ref: Ref) -> str:
        return f"[{self.ids[ref]}]" if ref in self.ids else ref_str(ref)

    async def push(self, chosen: Sequence[Candidate], unknown: ReadUnknown, prefilter: list[str],
                   ) -> dict[str, ReadResult]:
        asked = _asked(unknown)
        requests: dict[str, dict[str, Any]] = {}
        for c in chosen:
            req = requests.setdefault(c.site, {"blocks": [[]], "prefilter": prefilter, "exempt": [], "labels": {}})
            req["blocks"][0].append(c.oid)
            if c.ref in self.labels:
                req["labels"][c.oid] = self.labels[c.ref]
            self.judged.add((c.ref, asked))
        results = await self.ctx.fed.read(requests, self.q.question, [unknown.to_dict()], include_raw=True,
                                          ledger=self.ctx.ledger)
        for site, r in results.items():
            if r.error is not None:
                continue
            self.spent += len(r.read)
            for oid in r.read:
                ref = (site, oid)
                if ref not in self.consulted:
                    self.consulted.append(ref)
                if r.doc_ids.get(oid):
                    self.doc_ids[ref] = r.doc_ids[oid]
            self.texts.update({(site, oid): text for oid, text in r.raw.items()})
        return results

    def derive(self, results: dict[str, ReadResult], unknown: ReadUnknown, conds: Sequence[Condition],
               ) -> list[tuple[str, Ref, list[int]]]:
        out = []
        known = {norm_entity(c.value) for c in conds if c.value}
        for site, r in results.items():
            for h in r.bindings.get(unknown.name, []):
                ref, entity = (site, str(h.get("oid", ""))), str(h.get("entity", "")).strip()
                nums = sorted({int(n) for n in h.get("conditions") or [] if str(n).isdigit()
                               and 1 <= int(n) <= len(conds)})
                if not entity or not nums or norm_entity(entity) in known:
                    continue
                self.facts += [conds[n - 1].fact(entity, ref, r.modalities.get(ref[1], "text")) for n in nums]
                out.append((entity, ref, nums))
        if out:
            self.bindings.setdefault("; ".join(unknown.conditions), out[0][0])
        return out

    def outcome(self, results: dict[str, ReadResult], chosen: Sequence[Candidate], fresh: Sequence[Candidate],
                prefilter: list[str], cut: str | None) -> list[str]:
        read = [self.tag((s, o)) for s, r in results.items() if r.error is None for o in r.read]
        lines = [f"read {' '.join(read)}" if read else "nothing read"]
        skipped = [self.tag((s, o)) for s, r in results.items() for o in r.skipped]
        if skipped:
            lines.append(f"skipped (they name none of {'; '.join(prefilter)}): {' '.join(skipped)}")
        lines += [f"read failed at {s}" for s, r in results.items() if r.error]
        if len(chosen) < len(fresh):
            lines.append(f"not read ({cut or 'read budget'}): {' '.join(self.tag(c.ref) for c in fresh[len(chosen):])}")
        return lines

    async def read(self, ids: list[str], conds: list[Condition], many: bool = False, final: bool = False) -> str:
        self.current = [c.value for c in conds if c.value]
        unknown = ReadUnknown("X", [c.phrase for c in conds], [c.value for c in conds],
                              LIST_CAP if many else OPS_CAP, answer=final)
        cands = [self.listed[i] for i in dict.fromkeys(ids) if i in self.listed]
        if not cands:
            return "no listed object among these ids: read the ids a search listed"
        fresh = [c for c in cands if (c.ref, _asked(unknown)) not in self.judged]
        if not fresh:
            return "already read for this condition"
        prefilter = [c.value for c in conds if c.value] if all(c.value for c in conds) else []
        chosen, cut = self.fit(fresh, unknown, prefilter)
        if not chosen:
            return self.no_room(cut)
        results = await self.push(chosen, unknown, prefilter)
        hits = self.derive(results, unknown, conds)
        lines = [f"X = {e} {self.tag(ref)}" + (f" (conditions {', '.join(map(str, nums))})" if len(conds) > 1 else "")
                 for e, ref, nums in hits] or ["no value found"]
        return "\n".join([*lines, *self.outcome(results, chosen, fresh, prefilter, cut)])

    async def check(self, names: list[str], conds: list[Condition]) -> str:
        if not names:
            return "no candidates given"
        if not self.pictures:
            return "no site holds pictures"
        self.current = [c.value for c in conds if c.value]
        key = "; ".join(dict.fromkeys(c.key() for c in conds))
        found = (await self.ctx.fed.probe([ProbeRequest(key, "title", PICTURES, collections=self.pictures,
                                                         titles=names)], ledger=self.ctx.ledger))[0]
        pictures = rrf(split_lists(found), salt=self.q.qid)
        self.show(pictures)
        unknown = ReadUnknown("X", [c.phrase for c in conds], [c.value for c in conds], OPS_CAP, "filter", names)
        fresh = [c for c in pictures if (c.ref, _asked(unknown)) not in self.judged]
        missing = [n for n in names if not any(c.label == n for c in pictures)]
        none = f"no picture found for: {'; '.join(missing)}" if missing else ""
        if not fresh:
            return none or "already checked for this condition"
        chosen, cut = self.fit(fresh, unknown, [])
        if not chosen:
            return self.no_room(cut)
        results = await self.push(chosen, unknown, [])
        hits = self.derive(results, unknown, conds)
        passed: dict[str, Ref] = {}
        for entity, ref, _ in hits:
            name = entity if entity in names else next((n for n in names if same_entity(n, entity)), entity)
            passed.setdefault(name, ref)
        lines = [f"pass: {'; '.join(f'{n} {self.tag(ref)}' for n, ref in passed.items())}" if passed
                 else "no candidate's picture shows it"]
        judged = [n for n in names if n not in passed and any(c.label == n for c in chosen)]
        if judged:
            lines.append(f"judged, not shown: {'; '.join(judged)}")
        if none:
            lines.append(none)
        return "\n".join([*lines, *self.outcome(results, chosen, fresh, [], cut)])

    def context(self, facts: Sequence[Fact]) -> list[str]:
        if facts:
            texts = [self.texts[r] for r in dict.fromkeys((f.site, f.oid) for f in facts) if r in self.texts]
        else:
            texts = [c.summary for c in self.listed.values() if c.site not in self.sovereign]
        return _context(texts)


@register("react")
class ReAct(Method):
    name = "react"

    def __init__(self, max_steps: int = 8, k: int = 8, shown: int = 5, demos: int = 2, images: int = 4,
                 image_px: int = 512, total_budget: int | None = None) -> None:
        super().__init__()
        if max_steps < 1 or k < 1 or shown < 1 or demos < 0 or images < 0:
            raise ValueError(f"max_steps {max_steps}, k {k}, shown {shown}, demos {demos} or images {images} out of "
                             "range")
        if total_budget is not None and (isinstance(total_budget, bool) or int(total_budget) <= 0):
            raise ValueError(f"total_budget {total_budget!r} is not a positive number of cost units")
        self.max_steps, self.k, self.shown, self.demos = int(max_steps), int(k), int(shown), int(demos)
        self.images, self.image_px = int(images), int(image_px)
        self.total_budget = int(total_budget) if total_budget is not None else None

    def demonstrations(self, q: Question) -> tuple[str, ...]:
        if not self.demos:
            return ()
        return prompts.REACT_DEMOS.get(q.dataset or str(q.meta.get("dataset", "")), ())[: self.demos]

    def _answer_cost(self, q: Question, ctx: QueryContext, texts: list[str], pictures: int) -> int:
        from fedquest import budget

        msgs = prompts.answer_messages(q.question, prompts.format_facts([]), texts, ctx.answer_style)
        return budget.count_messages(msgs) + budget.IMAGE_TOKENS * pictures + budget.OUT_WEIGHT * 24

    async def answer(self, q: Question, ctx: QueryContext) -> MethodResult:
        import asyncio

        from fedquest import budget
        from fedquest.site.service import decode_image

        led = ctx.ledger
        demos = self.demonstrations(q)
        transcript: list[dict[str, str]] = []
        texts: list[str] = []
        docs: list[str] = []
        pictures: list[tuple[str, str]] = []
        seen: set[Ref] = set()
        read: list[Ref] = []
        ev = Evidence()
        answer, steps, ended = "", 0, "steps"

        def left() -> int | None:
            return None if self.total_budget is None else self.total_budget - led.total_tokens

        def step_messages() -> list[dict[str, Any]]:
            return prompts.agent_messages(prompts.REACT_SYSTEM, q.question, transcript,
                                          prompts.react_budget(left()), demos, "v14")

        def fits() -> bool:
            if self.total_budget is None:
                return True
            need = budget.call_cost(step_messages(), "planner", OPS_MAX_TOKENS) + \
                self._answer_cost(q, ctx, texts, min(len(pictures), self.images))
            return led.total_tokens + need <= self.total_budget

        for _ in range(self.max_steps):
            if self.total_budget is not None:
                led.reserve = self._answer_cost(q, ctx, texts, min(len(pictures), self.images))
                if not fits():
                    ended = "budget"
                    break
            try:
                obj, _reply = await ctx.llm.chat_json(step_messages(), role="planner", task="planner",
                                                      max_tokens=OPS_MAX_TOKENS, ledger=led, charge="planner",
                                                      default={})
            except budget.BudgetExhausted:
                ended = "budget"
                break
            steps += 1
            action, args = _action(obj)
            arg = _text(args, "query", "text", "answer", "input")
            thought = str(obj.get("thought", "")).strip() if isinstance(obj, dict) else ""
            transcript.append({"role": "assistant", "content": json.dumps(
                {"thought": thought, "action": action, "input": arg}, ensure_ascii=False)})
            if action == "finish":
                answer, ended = arg, "finish"
                break
            if action != "search" or not arg:
                transcript.append({"role": "observation", "content": "invalid action: use search or finish"})
                continue
            try:
                cands = (await ctx.fed.probe([ProbeRequest(arg, "question", self.k)], ledger=led))[0]
            except budget.BudgetExhausted:
                ended = "budget"
                break
            (sims,) = await asyncio.to_thread(ctx.matcher.sim.relevance, [(arg, [c.summary for c in cands])])
            score = {c.ref: float(s) for c, s in zip(cands, sims, strict=True)}
            ranked = sorted({c.ref: c for c in cands}.values(), key=lambda c: (-score[c.ref], c.rank, c.oid))
            new = [c for c in ranked if c.ref not in seen][: self.shown]
            shipped = await ctx.fed.fetch_raw([c.ref for c in new], ledger=led, max_px=self.image_px) if new else []
            raws = {(r.site, r.oid): r for r in shipped}
            lines: list[str] = []
            added: list[tuple[Candidate, str, tuple[str, str] | None]] = []
            for c in new:
                r = raws.get(c.ref)
                body = r.text if r is not None and not r.denied and r.text else c.summary
                title = (r.title if r is not None and r.title else "") or c.summary.split(": ", 1)[0]
                text = body[len(title) + 2:] if title and body.startswith(f"{title}: ") else body
                pic = None
                if (r is not None and r.modality == "image" and not r.denied and len(pictures) < self.images
                        and (uri := decode_image(r))):
                    pic = (title, uri)
                added.append((c, f"{title}\n{text}", pic))
            while added:
                lines = [f"[{len(read) + i + 1}] {t.replace(chr(10), ': ', 1)}" for i, (_, t, _) in enumerate(added)]
                transcript.append({"role": "observation", "content": "\n".join(lines)})
                trial_texts, trial_pics = texts + [t for _, t, _ in added], pictures + [p for *_, p in added if p]
                ok = self.total_budget is None
                if not ok:
                    need = budget.call_cost(step_messages(), "planner", OPS_MAX_TOKENS) + \
                        self._answer_cost(q, ctx, trial_texts, min(len(trial_pics), self.images))
                    ok = led.total_tokens + need <= self.total_budget
                transcript.pop()
                if ok:
                    break
                added.pop()
            if not added:
                transcript.append({"role": "observation", "content": "no new results" if not new else
                                   "no budget left for these results: finish now"})
                if new and self.total_budget is not None:
                    ended = "budget"
                    break
                continue
            transcript.append({"role": "observation", "content": "\n".join(lines)})
            for c, t, pic in added:
                seen.add(c.ref)
                read.append(c.ref)
                texts.append(t)
                docs.append(c.doc_id or "")
                if pic is not None:
                    pictures.append(pic)
                if c.doc_id:
                    ev.doc_ids[c.ref] = c.doc_id
            led.note_read([c.doc_id for c, _, _ in added if c.doc_id], "coordinator")
        for t in transcript:
            ctx.log("step", **t)
        ctx.log("planner_answer", text=answer, steps=steps, ended=ended, demos=len(demos), read=len(read))
        led.reserve = 0
        answer = await generate_answer(q, ctx, [], texts, pictures[: self.images] or None, doc_ids=docs)
        return finish(q, self.name, answer, evidence=ev, consulted=read)

    def describe(self) -> dict[str, Any]:
        return {"max_steps": self.max_steps, "k": self.k, "shown": self.shown, "demos": self.demos,
                "images": self.images, "total_budget": self.total_budget}
