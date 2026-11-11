from __future__ import annotations

import asyncio
import time
from typing import Any, ClassVar

from fedquest.methods.base import Method, MethodResult, QueryContext, register
from fedquest.methods.common import Evidence, finish, generate_answer
from fedquest.types import Fact, ProbeRequest, Question


@register("fedgrag")
@register("fedgrag_text", passages=8)
@register("fedgrag_tuned", settings="tuned")
@register("fedgrag_trained", encoder="trained", settings="tuned")
class FedGRAG(Method):
    name = "fedgrag"
    added_params: ClassVar[dict[str, Any]] = {"router": None, "encoder": None, "settings": None}

    def __init__(self, index: str | None = None, b: int | None = None, passages: int = 0,
                 total_budget: int | None = None, router: str | None = None, encoder: str | None = None,
                 settings: str | None = None) -> None:
        super().__init__()
        from fedquest.fedgrag.model import ENCODERS

        if not index:
            raise ValueError("fedgrag needs index: the federation its offline build indexed (python -m fedquest.fedgrag "
                             "build <federation>)")
        if b is not None and int(b) < 1:
            raise ValueError(f"b {b!r} must be at least 1")
        if encoder is not None and encoder not in ENCODERS:
            raise ValueError(f"encoder {encoder!r} is not one of {ENCODERS} (None: the training-free one)")
        if settings not in (None, "tuned"):
            raise ValueError(f"settings {settings!r} is not 'tuned' (None: the main runs')")
        self.index, self.b, self.passages = index, (int(b) if b is not None else None), int(passages)
        self.total_budget = int(total_budget) if total_budget is not None else None
        self.encoder, self.settings = encoder, settings
        if router is not None and (not isinstance(router, str) or not router.strip()):
            raise ValueError(f"router {router!r} is not a router name")
        self.router = router

    async def answer(self, q: Question, ctx: QueryContext) -> MethodResult:
        from fedquest import xfit
        from fedquest.fedgrag.model import DEFAULT, Encoder, FedGRAGIndex, Settings, doc_of

        router = xfit.resolve(self.router, q.qid)
        idx = await asyncio.to_thread(FedGRAGIndex.load, self.index, router, self.encoder)
        s, b = DEFAULT, self.b
        if self.settings == "tuned":
            tuned = idx.tuned(router)
            if tuned is None:
                raise RuntimeError(f"FedGRAG layout {self.index} (router {router or 'all'}, encoder "
                                   f"{self.encoder or 'free'}) is not tuned: python -m fedquest.fedgrag tune "
                                   f"{self.index}" + (f" --encoder {self.encoder}" if self.encoder else ""))
            s, b = Settings.of(tuned["settings"]), (self.b or int(tuned["b"]))
        qv = (await asyncio.to_thread(Encoder.encode, [q.question]))[0]
        t0 = time.thread_time()
        chosen, per, composed = await asyncio.to_thread(idx.retrieve, qv, b, s)
        ctx.ledger.plan_seconds += time.thread_time() - t0
        sites = [idx.clients[i].site for i in chosen]
        led = ctx.ledger
        t1 = time.perf_counter()
        try:
            await ctx.fed.probe([ProbeRequest(q.question, "question", 1)], sites=sites, ledger=None)
        except Exception as e:
            ctx.log("fedgrag_round", error=f"{type(e).__name__}: {e}")
        led.critical_seconds += time.perf_counter() - t1
        led.rounds += 1
        led.messages += len(sites)
        led.bytes_control += len(sites) * len(q.question.encode())
        for ci, facts in per.items():
            nbytes = idx.package_bytes(facts)
            led.bytes_facts += nbytes
            led.bytes_reply += nbytes
            led.to_coord[idx.clients[ci].site] = led.to_coord.get(idx.clients[ci].site, 0) + nbytes
        cor = idx.corpus
        returned = {doc_of(cor.fact_oids[f]): idx.clients[ci].site for ci, facts in per.items() for f, _ in facts}
        for d, site in returned.items():
            if d in led.gold:
                e = led.evidence_log.setdefault(d, dict(led.gold[d]))
                e.setdefault("returned", 1)
                e.setdefault("site", site)
                e.setdefault("kind", "kg")
        facts_out = [Fact(head=cor.names[cor.facts[f][0]], relation=cor.relations[f], tail=cor.names[cor.facts[f][1]],
                          site=idx.clients[ci].site, oid=cor.fact_oids[f]) for f, ci, _ in composed]
        led.note_context([doc_of(f.oid) for f in facts_out])
        snippets: list[str] = []
        docs: list[str] = []
        consulted: list[tuple[str, str]] = []
        if self.passages:
            refs = list(dict.fromkeys((f.site, f.oid) for f in facts_out))[: self.passages]
            for raw in await ctx.fed.fetch_raw(refs, ledger=led):
                if raw.text and not raw.denied:
                    snippets.append(f"{raw.title}\n{raw.text}")
                    docs.append(raw.doc_id or doc_of(raw.oid))
                    consulted.append((raw.site, raw.oid))
            led.note_read(docs, "coordinator")
        ev = Evidence()
        ev.doc_ids.update({(f.site, f.oid): doc_of(f.oid) for f in facts_out})
        extra = {"encoder": self.encoder} if self.encoder else {}
        if self.settings:
            extra["settings"] = {k: v for k, v in s.as_dict().items() if v != getattr(DEFAULT, k)}
        ctx.log("fedgrag", clients=sites, b=len(sites), facts=len(facts_out), router=router,
                per_client={idx.clients[ci].site: len(fs) for ci, fs in per.items()}, **extra)
        answer = await generate_answer(q, ctx, facts_out, snippets, doc_ids=docs or None)
        out: MethodResult = finish(q, self.name, answer, evidence=ev, facts=facts_out,
                                   consulted=consulted or list(dict.fromkeys((f.site, f.oid) for f in facts_out)))
        return out

    def describe(self) -> dict[str, Any]:
        out = {"index": self.index, "b": self.b, "passages": self.passages, "router": self.router}
        out.update({k: v for k, v in (("encoder", self.encoder), ("settings", self.settings)) if v is not None})
        return out
