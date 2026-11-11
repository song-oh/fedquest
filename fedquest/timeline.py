# per-question span timeline, attributed to stages and resources of the wall time
from __future__ import annotations

import heapq
import itertools
import json
import logging
import math
import os
import threading
import time
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

STAGES = ("plan", "search", "read", "answer", "other")
RESOURCES = ("llm_service", "llm_overhead", "site", "cpu")
ENV = "FEDQUEST_TIMELINE"
LOG_ENV = "FEDQUEST_TIMELINE_LOG"
OVERLAP_TOL = 1e-3

TASK_STAGE: dict[str, str] = {
    "pattern": "plan",
    "visual": "plan",
    "repair": "plan",
    "ground": "plan",
    "planner": "plan",
    "planner_ops": "plan",
    "read_index": "read",
    "read_facts": "read",
    "extract_query": "read",
    "verify": "read",
    "ircot": "read",
    "read": "read",
    "extract_text": "read",
    "extract_table": "read",
    "extract_image": "read",
    "answer": "answer",
    "closed_book": "answer",
}
CHARGE_STAGE: dict[str, str] = {"pattern": "plan", "repair": "plan", "ground": "plan", "planner": "plan",
                                "bind": "read", "bind_index": "read", "extract_entry": "read", "verify": "read",
                                "reader": "read", "answer": "answer"}
UNTIMED = frozenset({"judge", "check", "canary"})
SITE_STAGE: dict[str, str] = {"info": "search", "probe": "search", "read": "read", "extract": "read",
                              "fetch_raw": "answer", "hippo": "search"}

STAGE_FIELDS = tuple(f"stage_{s}_s" for s in STAGES)
RESOURCE_FIELDS = tuple(f"res_{r}_s" for r in RESOURCES)
COUNT_FIELDS = ("timeline_spans", "timeline_overlap", "timeline_unmapped")


@dataclass
class Span:
    phase: str
    t0: float
    t1: float
    parts: dict[str, float]
    nested: bool = False
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def seconds(self) -> float:
        return self.t1 - self.t0


@dataclass
class Timeline:
    spans: list[Span] = field(default_factory=list)
    unmapped: int = 0


@dataclass
class Attribution:
    t0: float
    t1: float
    stages: dict[str, float]
    resources: dict[str, float]
    site_llm: float = 0.0
    spans: int = 0
    overlap: int = 0
    unmapped: int = 0

    def fields(self) -> dict[str, float]:
        out = {f"stage_{s}_s": self.stages[s] for s in STAGES}
        out.update({f"res_{r}_s": self.resources[r] for r in RESOURCES})
        out.update(res_site_llm_s=self.site_llm, timeline_spans=self.spans, timeline_overlap=self.overlap,
                   timeline_unmapped=self.unmapped)
        return out


_current: ContextVar[Timeline | None] = ContextVar("fedquest_timeline", default=None)
_phase: ContextVar[str | None] = ContextVar("fedquest_timeline_phase", default=None)
_site: ContextVar[str | None] = ContextVar("fedquest_timeline_site", default=None)
_warned: set[str] = set()
_dump_lock = threading.Lock()


def enabled() -> bool:
    return os.environ.get(ENV, "") == "1" or bool(os.environ.get(LOG_ENV))


def current() -> Timeline | None:
    return _current.get()


@contextmanager
def recording(on: bool = True) -> Iterator[Timeline | None]:
    if not on:
        yield None
        return
    tl = Timeline()
    token = _current.set(tl)
    try:
        yield tl
    finally:
        _current.reset(token)


def _stage(stage: str) -> str:
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; stages: {STAGES}")
    return stage


@contextmanager
def phase(stage: str) -> Iterator[None]:
    _stage(stage)
    if _current.get() is None:
        yield
        return
    token = _phase.set(stage)
    try:
        yield
    finally:
        _phase.reset(token)


@contextmanager
def in_site(site: str) -> Iterator[None]:
    if _current.get() is None:
        yield
        return
    token = _site.set(site)
    try:
        yield
    finally:
        _site.reset(token)


@contextmanager
def span(stage: str, **meta: Any) -> Iterator[None]:
    _stage(stage)
    tl = _current.get()
    if tl is None:
        yield
        return
    t0 = time.perf_counter()
    try:
        yield
    finally:
        t1 = time.perf_counter()
        tl.spans.append(Span(stage, t0, t1, {"cpu": t1 - t0}, _site.get() is not None, {"kind": "cpu", **meta}))


def stage_of(task: str, charge: str | None = None) -> str | None:
    return TASK_STAGE.get(task) or CHARGE_STAGE.get(charge or "")


def _unmapped(tl: Timeline, what: str) -> None:
    tl.unmapped += 1
    if what not in _warned:
        _warned.add(what)
        log.warning("timeline: %s names no stage (counted in timeline_unmapped; see fedquest.timeline)", what)


def note_llm(task: str, charge: str | None, t0: float, service: float | None, tried: float = 0.0,
             replay: bool = False) -> None:
    tl = _current.get()
    if tl is None or task in UNTIMED:
        return
    t1 = time.perf_counter()
    if service is None and not tried:
        t0 = t1
    stage = stage_of(task, charge)
    if stage is None:
        _unmapped(tl, f"model task {task!r} (charge {charge!r})")
        stage = "read"
    served = min(max(service or 0.0, 0.0), t1 - t0)
    site = _site.get()
    meta: dict[str, Any] = {"kind": "llm", "task": task}
    if charge:
        meta["charge"] = charge
    if site is not None:
        meta["site"] = site
    if replay:
        meta["replay"] = True
    tl.spans.append(Span(stage, t0, t1, {"llm_service": served, "llm_overhead": t1 - t0 - served}, site is not None,
                         meta))


def note_site(site: str, action: str, t0: float, seconds: float, injected: float = 0.0,
              rid: str | None = None) -> None:
    tl = _current.get()
    if tl is None:
        return
    stage = _phase.get() or SITE_STAGE.get(action)
    if stage is None:
        _unmapped(tl, f"site action {action!r}")
        stage = "search"
    meta: dict[str, Any] = {"kind": "site", "site": site, "action": action}
    if injected:
        meta["injected"] = injected
    if rid:
        meta["rid"] = rid
    tl.spans.append(Span(stage, t0, t0 + seconds, {"site": seconds}, _site.get() is not None, meta))


def attribute(tl: Timeline, t0: float, t1: float) -> Attribution:
    stages = dict.fromkeys(STAGES, 0.0)
    res = dict.fromkeys(RESOURCES, 0.0)
    top: list[tuple[float, float, Span]] = []
    inner: list[tuple[float, float, str]] = []
    for s in tl.spans:
        a, b = max(s.t0, t0), min(s.t1, t1)
        if b <= a:
            continue
        if not s.nested:
            top.append((a, b, s))
        elif s.meta.get("kind") == "llm":
            inner.append((a, b, s.meta.get("site", "")))
    top.sort(key=lambda x: x[0])
    starts, ends = sorted(inner), sorted(inner, key=lambda x: x[1])
    busy: Counter[str] = Counter()
    active: list[tuple[float, float, int]] = []
    won = [0.0] * len(top)
    site_llm, nxt, i, j = 0.0, 0, 0, 0
    cuts = sorted({t0, t1, *(x for a, b, _ in top for x in (a, b)), *(x for a, b, _ in inner for x in (a, b))})
    # sweep: each slice between cuts goes to the open top-level span that ends last
    for a, b in itertools.pairwise(cuts):
        while nxt < len(top) and top[nxt][0] <= a:
            heapq.heappush(active, (-top[nxt][1], top[nxt][0], nxt))
            nxt += 1
        while active and -active[0][0] <= a:
            heapq.heappop(active)
        while i < len(starts) and starts[i][0] <= a:
            busy[starts[i][2]] += 1
            i += 1
        while j < len(ends) and ends[j][1] <= a:
            busy[ends[j][2]] -= 1
            j += 1
        piece = b - a
        if not active:
            stages["other"] += piece
            res["cpu"] += piece
            continue
        k = active[0][2]
        s = top[k][2]
        won[k] += piece
        stages[s.phase] += piece
        for r, x in s.parts.items():
            res[r] += piece * x / s.seconds
        if s.meta.get("kind") == "site" and busy[s.meta.get("site", "")] > 0:
            site_llm += piece
    overlap = sum(OVERLAP_TOL < w < (b - a) - OVERLAP_TOL for w, (a, b, _) in zip(won, top, strict=True))
    return Attribution(t0, t1, stages, res, site_llm, len(tl.spans), overlap, tl.unmapped)


def _ms(values: dict[str, float], total: float) -> dict[str, float]:
    # largest-remainder rounding to whole ms, so the parts sum to the rounded wall time
    exact = {k: max(v, 0.0) * 1000 for k, v in values.items()}
    ms = {k: math.floor(v) for k, v in exact.items()}
    for k in sorted(exact, key=lambda k: ms[k] - exact[k])[: max(0, round(total * 1000) - sum(ms.values()))]:
        ms[k] += 1
    return {k: n / 1000 for k, n in ms.items()}


def row_fields(fields: dict[str, float], wall: float) -> dict[str, float]:
    out = {**_ms({k: fields[k] for k in STAGE_FIELDS}, wall), **_ms({k: fields[k] for k in RESOURCE_FIELDS}, wall)}
    out["res_site_llm_s"] = min(round(fields["res_site_llm_s"], 3), out["res_site_s"])
    out.update({k: int(fields[k]) for k in COUNT_FIELDS})
    return out


def dump(tl: Timeline, att: Attribution, *, qid: str, method: str, ns: str) -> None:
    path = os.environ.get(LOG_ENV)
    if not path:
        return
    parts = ns.split("/")
    label = parts[-2] if len(parts) >= 3 and parts[-1] in (qid, "eager") else None
    line = {"qid": qid, "label": label, "method": method, "ns": ns, "wall_s": round(att.t1 - att.t0, 6),
            "stages": {k: round(v, 6) for k, v in att.stages.items()},
            "resources": {k: round(v, 6) for k, v in att.resources.items()}, "site_llm_s": round(att.site_llm, 6),
            "overlap": att.overlap, "unmapped": att.unmapped,
            "spans": [{"phase": s.phase, "t0": round(s.t0 - att.t0, 6), "t1": round(s.t1 - att.t0, 6),
                       "parts": {k: round(v, 6) for k, v in s.parts.items()}, "nested": s.nested, **s.meta}
                      for s in tl.spans]}
    text = json.dumps(line, ensure_ascii=False, default=str) + "\n"
    try:
        with _dump_lock:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(text)
    except OSError as e:
        if path not in _warned:
            _warned.add(path)
            log.warning("timeline: cannot append to the span log %s (%s: %s)", path, type(e).__name__, e)
