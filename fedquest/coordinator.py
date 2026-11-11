# coordinator: a fresh ledger and query context per question, run in its wire scope
from __future__ import annotations

from dataclasses import dataclass, field

from fedquest.config import MatcherCfg
from fedquest.federation.base import Federation
from fedquest.federation.replay import wire_scope
from fedquest.kg.matcher import Matcher, TextSim
from fedquest.kg.view import KnowledgeView
from fedquest.ledger import Ledger
from fedquest.llm.client import LLMClient
from fedquest.methods.base import Method, MethodResult, QueryContext
from fedquest.planner.calibration import Calibrator
from fedquest.types import Question
from fedquest.vdb.encoders import load_encoder


def build_matcher(cfg: MatcherCfg | None = None) -> Matcher:
    cfg = cfg or MatcherCfg()
    return Matcher(TextSim(load_encoder(cfg.encoder)), cfg.theta, cfg.max_hops, cfg.node_theta)


@dataclass
class Coordinator:
    fed: Federation
    llm: LLMClient
    kg: KnowledgeView
    matcher: Matcher
    answer_style: str = "short"
    verify: bool = True
    calib: Calibrator = field(default_factory=Calibrator)

    def context(self, ns: str) -> QueryContext:
        return QueryContext(self.fed, self.llm, self.kg, self.matcher, self.calib, Ledger(), ns,
                            answer_style=self.answer_style, verify=self.verify)

    async def run(self, method: Method, q: Question, ns: str) -> MethodResult:
        ctx = self.context(ns)
        budget = getattr(method, "total_budget", None)
        if budget is not None:
            ctx.ledger.budget = int(budget)
        # gold positions only feed the evidence log (eval/evidence.py)
        ctx.ledger.gold = gold_positions(q)
        with wire_scope(f"{ns}#{q.qid}"):
            return await method.run(q, ctx)


def gold_positions(q: Question) -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    for i, h in enumerate((q.meta or {}).get("hops") or []):
        d = h.get("doc_id")
        if d and d not in out:
            out[d] = {"hop": i, "later": bool(h.get("after"))}
    for d in q.gold_doc_ids:
        out.setdefault(d, {"hop": 0, "later": False})
    return out
