# Method base class and registry, with the per-question context and result
from __future__ import annotations

import inspect
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

from fedquest import timeline
from fedquest.errors import AuthExpired, auth_watch
from fedquest.ledger import Ledger
from fedquest.llm.prompts import EXTRACT_VERSION
from fedquest.types import Question, Record

if TYPE_CHECKING:
    from fedquest.federation.base import Federation
    from fedquest.kg.matcher import Matcher
    from fedquest.kg.view import KnowledgeView
    from fedquest.llm.client import LLMClient
    from fedquest.planner.calibration import Calibrator


@dataclass
class QueryContext:
    fed: Federation
    llm: LLMClient
    kg: KnowledgeView
    matcher: Matcher
    calib: Calibrator
    ledger: Ledger
    ns: str
    answer_style: str = "short"
    version: str = EXTRACT_VERSION
    verify: bool = True
    trace: list[dict[str, Any]] = field(default_factory=list)

    def log(self, event: str, **data: Any) -> None:
        self.trace.append({"event": event, **data})


@dataclass
class MethodResult(Record):
    qid: str
    method: str
    answer: str
    matched: bool = False
    pattern: dict[str, Any] | None = None
    bindings: dict[str, str] = field(default_factory=dict)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    consulted: list[str] = field(default_factory=list)
    reused: list[str] = field(default_factory=list)
    indexed: list[str] = field(default_factory=list)
    consulted_doc_ids: list[str] = field(default_factory=list)
    support_oids: list[str] = field(default_factory=list)
    ledger: dict[str, Any] = field(default_factory=dict)
    trace: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None


class Method(ABC):
    name: str = ""
    needs_build: ClassVar[bool] = False
    added_params: ClassVar[dict[str, Any]] = {}

    def __init__(self, **params: Any) -> None:
        if params:
            raise TypeError(f"{type(self).__name__} got unknown parameters: {sorted(params)}")

    @abstractmethod
    async def answer(self, q: Question, ctx: QueryContext) -> MethodResult: ...

    async def run(self, q: Question, ctx: QueryContext) -> MethodResult:
        with timeline.recording(timeline.enabled()) as tl, auth_watch() as refused:
            t0 = time.perf_counter()
            try:
                result = await self.answer(q, ctx)
            except AuthExpired:
                raise
            except Exception as exc:
                result = MethodResult(qid=q.qid, method=self.name, answer="", error=f"{type(exc).__name__}: {exc}")
            t1 = time.perf_counter()
        if refused:
            raise refused[0]
        ctx.ledger.wall_seconds = t1 - t0
        if tl is not None:
            att = timeline.attribute(tl, t0, t1)
            ctx.ledger.timeline = att.fields()
            timeline.dump(tl, att, qid=q.qid, method=self.name, ns=ctx.ns)
        result.ledger = ctx.ledger.summary()
        result.trace = ctx.trace
        return result


METHODS: dict[str, tuple[type[Method], dict[str, Any]]] = {}


def register(name: str, **defaults: Any) -> Callable[[type[Method]], type[Method]]:
    def wrap(cls: type[Method]) -> type[Method]:
        if name in METHODS:
            raise ValueError(f"method {name!r} registered twice")
        METHODS[name] = (cls, defaults)
        if not cls.name:
            cls.name = name
        return cls

    return wrap


def build_method(name: str, params: dict[str, Any] | None = None) -> Method:
    import fedquest.methods  # noqa: F401

    if name not in METHODS:
        raise KeyError(f"unknown method {name!r}; known: {sorted(METHODS)}")
    cls, defaults = METHODS[name]
    method = cls(**{**defaults, **(params or {})})
    method.name = name
    return method


def method_accepts(name: str, param: str) -> bool:
    import fedquest.methods  # noqa: F401

    cls, _ = METHODS[name]
    return param in inspect.signature(cls.__init__).parameters
