from __future__ import annotations

import asyncio
import re
import string
import threading
import time
from collections import Counter
from typing import Any, ClassVar

import numpy as np

from fedquest import budget
from fedquest.dgrag import (
    CANDIDATES,
    ENCODER,
    GATE_THRESHOLD,
    MATCH_THRESHOLD,
    RETRIEVE_THRESHOLD,
    TEMPERATURE,
    TOP_K,
    TOP_M,
)
from fedquest.llm import prompts
from fedquest.llm.parsing import parse_answer
from fedquest.methods.base import Method, MethodResult, QueryContext, register
from fedquest.methods.common import Evidence, finish, generate_answer, is_abstention
from fedquest.types import Fact, Question
from fedquest.utils import stable_int

LOCAL_TOKENS = 4000
CLOUD_TOKENS = 12000
CLOUD_RESERVE = 4000
GATE_RESERVE = 900
FACT_SHARE = 1 / 3
LOCAL_CHUNKS, CLOUD_CHUNKS = 32, 64

KEYWORDS_SYSTEM = ("You are a helpful assistant tasked with identifying both high-level and low-level keywords in the "
                   "user's query.")


def keywords_messages(question: str) -> list[dict[str, Any]]:
    user = ("---Goal---\nGiven the query, list both high-level and low-level keywords. High-level keywords focus on "
            "overarching concepts or themes, while low-level keywords focus on specific entities, details, or concrete "
            "terms.\n\n---Instructions---\n- Output the keywords in JSON format.\n- The JSON should have two keys:\n"
            '  - "high_level_keywords" for overarching concepts or themes.\n'
            '  - "low_level_keywords" for specific entities or details.\n\n'
            "---Example---\nQuery: \"How does international trade influence global economic stability?\"\n"
            'Output: {"high_level_keywords": ["International trade", "Global economic stability", "Economic impact"], '
            '"low_level_keywords": ["Trade agreements", "Tariffs", "Currency exchange", "Imports", "Exports"]}\n\n'
            f"---Real Data---\nQuery: {question}\nOutput:")
    return [{"role": "system", "content": KEYWORDS_SYSTEM}, {"role": "user", "content": user}]


GATE_SYSTEM = "You assess candidate answers that a data owner's model wrote from its local knowledge alone."


def gate_messages(question: str, candidates: list[str]) -> list[dict[str, Any]]:
    listed = "\n".join(f"[{i}] {c}" for i, c in enumerate(candidates, 1))
    user = (f"Question: {question}\nCandidate answers:\n{listed}\n\n"
            'Reply with JSON only: {"confident": true if every answer gives a definite answer and none says the '
            'information is insufficient or that more details are needed, else false, "consistency": a number from 0 '
            'to 1, how consistent the answers\' core claims are with each other, "best": the number of the best '
            'answer, "reason": a few words}')
    return [{"role": "system", "content": GATE_SYSTEM}, {"role": "user", "content": user}]


_ARTICLES = re.compile(r"\b(a|an|the)\b")


def _tokens(text: str) -> set[str]:
    t = "".join(c for c in text.lower() if c not in string.punctuation)
    return set(_ARTICLES.sub(" ", t).split())


def jaccard(texts: list[str]) -> float:
    sets = [_tokens(t) for t in texts]
    pairs = [(a, b) for i, a in enumerate(sets) for b in sets[i + 1:]]
    if not pairs:
        return 1.0
    return float(np.mean([len(a & b) / len(a | b) if a | b else 1.0 for a, b in pairs]))


def cosine(vecs: np.ndarray) -> float:
    n = len(vecs)
    if n < 2:
        return 1.0
    sims = vecs @ vecs.T
    return float(np.mean([sims[i, j] for i in range(n) for j in range(i + 1, n)]))


class _Embedder:
    _lock = threading.Lock()

    @classmethod
    def queries(cls, name: str, texts: list[str]) -> np.ndarray:
        from fedquest.vdb.encoders import load_encoder

        with cls._lock:
            return load_encoder(name).encode_queries(texts)

    @classmethod
    def docs(cls, name: str, texts: list[str]) -> np.ndarray:
        from fedquest.vdb.encoders import load_encoder

        with cls._lock:
            return load_encoder(name).encode_docs(texts)


@register("dgrag")
class DGRAG(Method):
    name = "dgrag"
    batched: ClassVar[bool | None] = None

    def __init__(self, index: str | None = None, total_budget: int | None = None, top_m: int = TOP_M,
                 match_threshold: float | None = MATCH_THRESHOLD, retrieve_threshold: float | None = RETRIEVE_THRESHOLD,
                 candidates: int = CANDIDATES, gate_threshold: float = GATE_THRESHOLD,
                 temperature: float = TEMPERATURE, top_k: int = TOP_K, local_tokens: int = LOCAL_TOKENS,
                 cloud_tokens: int = CLOUD_TOKENS, encoder: str = ENCODER, gate: str = "dgrag") -> None:
        super().__init__()
        if not index:
            raise ValueError("dgrag needs index: the federation its offline build indexed (python -m fedquest.dgrag "
                             "build <federation>)")
        if gate not in ("dgrag", "local", "cloud"):
            raise ValueError(f"gate {gate!r}: dgrag (the paper's), local (always the local answer) or cloud (always "
                             "escalate)")
        if int(candidates) < 1 or int(top_m) < 1 or int(top_k) < 1:
            raise ValueError("candidates, top_m and top_k must be at least 1")
        if total_budget is not None and int(total_budget) <= 0:
            raise ValueError(f"total_budget {total_budget!r} is not positive")
        self.index, self.top_m, self.top_k = index, int(top_m), int(top_k)
        self.match_threshold = None if match_threshold is None else float(match_threshold)
        self.retrieve_threshold = None if retrieve_threshold is None else float(retrieve_threshold)
        self.candidates, self.gate_threshold, self.temperature = int(candidates), float(gate_threshold), \
            float(temperature)
        self.local_tokens, self.cloud_tokens = int(local_tokens), int(cloud_tokens)
        self.encoder, self.gate = encoder, gate
        self.total_budget = int(total_budget) if total_budget is not None else None

    async def _keywords(self, q: Question, ctx: QueryContext) -> tuple[list[str], list[str]]:
        try:
            obj, _ = await ctx.llm.chat_json(keywords_messages(q.question), role="reader", task="dgrag_keywords",
                                             max_tokens=200, ledger=ctx.ledger, charge="dgrag_keywords", default={})
        except budget.BudgetExhausted:
            obj = {}
        obj = obj if isinstance(obj, dict) else {}

        def words(key: str) -> list[str]:
            v = obj.get(key)
            return [str(x).strip() for x in v if str(x).strip()] if isinstance(v, list) else []

        return words("high_level_keywords"), words("low_level_keywords")

    def _local_cap(self, ctx: QueryContext) -> int:
        left = budget.remaining(ctx.ledger)
        if left is None:
            return self.local_tokens
        k = 1 if DGRAG.batched is not False else self.candidates
        frame = 400 + budget.OUT_WEIGHT * 24 * self.candidates
        room = (left - CLOUD_RESERVE - GATE_RESERVE) // k - frame
        return int(max(300, min(self.local_tokens, room)))

    async def _context(self, ctx: QueryContext, idx: Any, knowledge: list[Any], cap: int, nchunks: int,
                       charged: bool) -> tuple[list[Fact], list[str], list[str], list[tuple[str, str]]]:
        t = idx.tables
        facts: list[tuple[float, Fact]] = []
        seen: set[int] = set()
        for k in knowledge:
            for f, s in k.facts:
                if f not in seen:
                    seen.add(f)
                    h, tl = t.facts[f]
                    facts.append((s, Fact(head=t.names[h], relation=t.relations[f], tail=t.names[tl], site=k.site,
                                          oid=t.fact_oids[f])))
        facts.sort(key=lambda x: -x[0])
        kept_facts: list[Fact] = []
        used = 0
        for _, fact in facts:
            n = budget.count_text(f"({fact.head} | {fact.relation} | {fact.tail})") + 1
            if used + n > cap * FACT_SHARE:
                break
            kept_facts.append(fact)
            used += n
        refs = sorted(((s, k.site, o) for k in knowledge for o, s in k.chunks), key=lambda x: -x[0])
        order = list(dict.fromkeys((site, oid) for _, site, oid in refs))[:nchunks]
        raws = await ctx.fed.fetch_raw(order, ledger=ctx.ledger if charged else None) if order else []
        texts, docs, consulted = [], [], []
        for r in raws:
            if r.text and not r.denied:
                texts.append(f"{r.title}\n{r.text}" if r.title and not r.text.startswith(r.title) else r.text)
                docs.append(r.doc_id or "")
                consulted.append((r.site, r.oid))
        kept = budget.fit_indices(texts, max(0, cap - used))
        return kept_facts, [x for _, x in kept], [docs[i] for i, _ in kept], [consulted[i] for i, _ in kept]

    async def _gate(self, q: Question, ctx: QueryContext, cands: list[str]) -> dict[str, Any]:
        if self.gate != "dgrag":
            return {"local": self.gate == "local", "best": 0, "forced": self.gate}
        if any(is_abstention(c) for c in cands):
            return {"local": False, "confident": False, "why": "declined", "best": 0}
        vecs = await asyncio.to_thread(_Embedder.docs, self.encoder, cands)
        cos, jac = cosine(np.asarray(vecs, dtype=np.float32)), jaccard(cands)
        try:
            obj, _ = await ctx.llm.chat_json(gate_messages(q.question, cands), role="reader", task="dgrag_gate",
                                             max_tokens=200, ledger=ctx.ledger, charge="dgrag_gate", default={})
        except budget.BudgetExhausted:
            obj = None
        if not isinstance(obj, dict):
            return {"local": True, "confident": None, "cosine": round(cos, 4), "jaccard": round(jac, 4),
                    "why": "budget", "best": _majority(cands)}
        confident = bool(obj.get("confident"))
        try:
            consistency = min(1.0, max(0.0, float(obj.get("consistency", 0.0))))
        except (TypeError, ValueError):
            consistency = 0.0
        try:
            best = min(len(cands), max(1, int(obj.get("best", 1)))) - 1
        except (TypeError, ValueError):
            best = _majority(cands)
        score = (cos + jac + consistency) / 3
        return {"local": confident and score >= self.gate_threshold, "confident": confident, "cosine": round(cos, 4),
                "jaccard": round(jac, 4), "consistency": round(consistency, 4), "score": round(score, 4),
                "best": best}

    async def answer(self, q: Question, ctx: QueryContext) -> MethodResult:
        from fedquest.dgrag.index import DGRAGIndex

        idx = await asyncio.to_thread(DGRAGIndex.load, self.index)
        led = ctx.ledger
        if not idx.edges:
            raise ValueError(f"DGRAG's index of {self.index} has no edge")
        origin = stable_int("dgrag:origin", q.qid) % len(idx.edges)
        high, low = await self._keywords(q, ctx)
        t0 = time.thread_time()
        qv, lv, hv = np.asarray(await asyncio.to_thread(
            _Embedder.queries, self.encoder, [q.question, ", ".join(low) or q.question, ", ".join(high) or q.question]),
            dtype=np.float32)
        local = idx.retrieve(origin, lv, hv, self.top_k)
        led.plan_seconds += time.thread_time() - t0
        ev = Evidence()
        facts, snippets, docs, consulted = await self._context(ctx, idx, [local], self._local_cap(ctx), LOCAL_CHUNKS,
                                                               charged=False)
        _returned(led, [local])
        led.note_read([d for d in docs if d], "site")
        messages = prompts.answer_messages(q.question, prompts.format_facts(facts), snippets, ctx.answer_style)
        try:
            texts, reply = await ctx.llm.chat_n(messages, self.candidates, role="reader", task="dgrag_local",
                                                max_tokens=128 if ctx.answer_style == "short" else 400,
                                                temperature=self.temperature, ledger=led, charge="dgrag_local")
            if DGRAG.batched is None and self.candidates > 1:
                DGRAG.batched = (reply.samples or 1) >= self.candidates
            cands = [parse_answer(t) for t in texts]
        except budget.BudgetExhausted:
            cands = []
        decision = await self._gate(q, ctx, cands) if cands else {"local": False, "why": "no candidate", "best": 0}
        ctx.log("dgrag_local", edge=idx.sites[origin], high=high, low=low, facts=len(facts), chunks=len(snippets),
                candidates=cands, batched=DGRAG.batched, **{k: v for k, v in decision.items() if k != "local"},
                escalated=not decision["local"])
        if decision["local"]:
            led.note_context([d for d in docs if d])
            ev.doc_ids.update({ref: d for ref, d in zip(consulted, docs, strict=True) if d})
            answer = cands[int(decision.get("best", 0))]
            return finish(q, self.name, answer, evidence=ev, facts=facts, consulted=consulted)
        matched = idx.match(qv, self.top_m, self.match_threshold) or idx.match(qv, 1, None)
        owners = idx.owners(matched)
        knowledge = [idx.retrieve(ei, lv, hv, self.top_k, self.retrieve_threshold) for ei in owners]
        sites = [idx.sites[ei] for ei in owners]
        led.messages += len(sites)
        led.bytes_control += len(sites) * len(q.question.encode())
        for k in knowledge:
            nbytes = sum(len(idx.tables.fact_text(f).encode()) + 8 for f, _ in k.facts)
            led.bytes_facts += nbytes
            led.bytes_reply += nbytes
            led.to_coord[k.site] = led.to_coord.get(k.site, 0) + nbytes
        _returned(led, knowledge)
        left = budget.remaining(led)
        cap = self.cloud_tokens if left is None else max(300, min(self.cloud_tokens, left - 400))
        g_facts, g_snippets, g_docs, g_consulted = await self._context(ctx, idx, knowledge, cap, CLOUD_CHUNKS,
                                                                       charged=True)
        led.note_read([d for d in g_docs if d], "coordinator")
        ev.doc_ids.update({ref: d for ref, d in zip(g_consulted, g_docs, strict=True) if d})
        ctx.log("dgrag_cloud", summaries=[[i, round(s, 4)] for i, s in matched], edges=sites,
                facts=len(g_facts), chunks=len(g_snippets))
        answer = await generate_answer(q, ctx, g_facts, g_snippets, doc_ids=g_docs)
        return finish(q, self.name, answer, evidence=ev, facts=g_facts, consulted=g_consulted)

    def describe(self) -> dict[str, Any]:
        return {"index": self.index, "top_m": self.top_m, "match_threshold": self.match_threshold,
                "retrieve_threshold": self.retrieve_threshold, "candidates": self.candidates,
                "gate_threshold": self.gate_threshold, "temperature": self.temperature, "top_k": self.top_k,
                "local_tokens": self.local_tokens, "cloud_tokens": self.cloud_tokens, "encoder": self.encoder,
                "gate": self.gate, "total_budget": self.total_budget}


def _returned(led: Any, knowledge: list[Any]) -> None:
    for k in knowledge:
        for oid, _ in k.chunks:
            d = oid.split("/", 2)[-1].split("#", 1)[0]
            if d in led.gold:
                e = led.evidence_log.setdefault(d, dict(led.gold[d]))
                e.setdefault("returned", 1)
                e.setdefault("site", k.site)
                e.setdefault("kind", "kg")


def _majority(cands: list[str]) -> int:
    if not cands:
        return 0
    keys = [" ".join(sorted(_tokens(c))) for c in cands]
    top = Counter(keys).most_common(1)[0][0]
    return keys.index(top)
