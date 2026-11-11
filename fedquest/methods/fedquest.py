from __future__ import annotations

import asyncio
import copy
import math
import re
import time
from collections import Counter, defaultdict
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass, field, replace
from itertools import product
from typing import Any, ClassVar, Literal

from rapidfuzz import fuzz

from fedquest import timeline
from fedquest.bloom import may_hold, title_key
from fedquest.budget import BudgetExhausted
from fedquest.federation.base import share_slack
from fedquest.kg.normalize import norm_entity
from fedquest.kg.resolve import KeyIndex
from fedquest.kg.view import AskRecord
from fedquest.llm import prompts
from fedquest.llm.cache import canonical_json
from fedquest.llm.parsing import parse_json
from fedquest.llm.prompts import ReadObject, ReadUnknown
from fedquest.methods import compiler
from fedquest.methods.base import Method, MethodResult, QueryContext, register
from fedquest.methods.common import (
    VISUAL,
    Evidence,
    Ref,
    finish,
    fold_media,
    generate_answer,
    group_rank,
    group_sizes,
    hint_modalities,
    is_abstention,
    key_text,
    make_pattern,
    release_signal,
    rrf,
    shaped,
    split_lists,
    tie,
)
from fedquest.methods.compiler import FALLBACK_VISUAL, prune  # noqa: F401
from fedquest.planner.blocks import PLACE_KINDS, Call, Cand, Place, allocate, calls, curve, plan_call
from fedquest.planner.calibration import load_calibration
from fedquest.types import (
    Candidate,
    Edge,
    ExtractionResult,
    Fact,
    Pattern,
    ProbeCounts,
    ProbeRequest,
    Question,
    RawObject,
    Usage,
    is_var,
)
from fedquest.utils import approx_tokens, stable_hash

MAX_SNIPPETS = 24
CONTEXT_TOKENS = 1500
ANSWER_FRAME = 600
COMPILE_COST = 2750
EVIDENCE_REF = 6000
VERIFY_FRAC = 0.35
READ_CAP = 3
LIST_CAP = 24
SIBLING_K = 24
FILTER_K = 2
SIBLING_TABLES = 2
COORD = "coordinator"
COORD_CAP = 12
SAFETY_CALLS = 64
REPAIR_MIN = 500
BEAM_ROUNDS = 6
OUT_EMPTY = 6.0
OUT_HIT = 26.0
FRAME = 8.0
SIZE = {"text": 105.0, "table": 39.0}
REFERENCE = ReadUnknown("x", ["(x | relation | value)"], ["value"], READ_CAP)
DIRECT_OBJECTS = 6
CHAIN_ROUNDS = 3
CHAIN_FACTS = 6
CHAIN_FLOOR = 2500
CHAIN_OBJECTS = 8
OWNER_CALLS = 1

VISUAL_SYSTEM = (
    "You decide how each condition of a question can be checked: from texts and tables, or only by looking at a "
    "picture. Return JSON only."
)
VISUAL_USER = (
    "Question: {question}\n"
    "Conditions (an unknown is named like film1 or person1):\n"
    "{conditions}\n"
    "Which conditions can only be verified by looking at a picture of an entity: its appearance (colors, shapes, "
    "clothing, hair, what is visible on it), what a poster, logo, cover, flag or photo depicts, or actions or text "
    "shown in an image? A condition that a text or a table can state (names, dates, places, roles, relations, numbers, "
    "awards) does not need a picture.\n"
    'Reply with JSON only, on one line: {{"picture": [<condition numbers>]}}, or {{"picture": []}} if none.'
)


@dataclass(frozen=True)
class Cond:
    id: int
    var: str
    edge: Edge | None = None
    text: str = ""
    visual: bool = False
    fallback: bool = False
    hop: bool = False
    parent: int | None = None
    deps: tuple[str, ...] = ()

    @property
    def other(self) -> str | None:
        return None if self.edge is None else self.edge.other(self.var)

    def phrase(self, value: str | None, relation: str) -> str:
        if self.edge is None:
            return f"X {self.text}"
        head = "X" if self.edge.head == self.var else value
        tail = "X" if self.edge.tail == self.var else value
        return f"({head} | {relation} | {tail})"

    def fact(self, entity: str, value: str | None, relation: str, ref: Ref, version: str, modality: str) -> Fact:
        if self.edge is None:
            return Fact(entity, self.text, "yes", ref[0], ref[1], version, modality)
        head = entity if self.edge.head == self.var else value or ""
        tail = entity if self.edge.tail == self.var else value or ""
        return Fact(head, relation, tail, ref[0], ref[1], version, modality)


@dataclass
class Held:
    name: str
    witnessed: dict[int, set[str]] = field(default_factory=dict)
    refs: list[Ref] = field(default_factory=list)
    facts: list[Fact] = field(default_factory=list)


@dataclass
class Asked:
    var: str
    cond: Cond
    value: str | None
    relation: str


@dataclass
class Found:
    c: Candidate
    kind: str
    signal: bool
    value: str | None
    conds: tuple[int, ...]
    probe: int


@dataclass
class Item:
    c: Candidate
    alpha: float
    r: float = 0.0
    p: float = 0.0
    beta: float = 1.0
    mentions: bool | None = None


def _key(name: str) -> str:
    return prompts.fold(norm_entity(name))


def _grounded(relation: str, pattern: Pattern, known: dict[str, str]) -> str:
    for x in pattern.variables:
        name = x.lstrip("?")
        if re.fullmatch(r"[A-Za-z_]+\d+", name) and re.search(rf"\b{re.escape(name)}\b", relation):
            value = known.get(x, "").replace("\\", "\\\\")
            relation = re.sub(rf"\b{re.escape(name)}\b", value, relation)
    return " ".join(relation.split())


def _heads(c: Candidate, value: str) -> bool:
    title = c.summary.split(": ", 1)[0]
    page = _body(c).split(" | ", 1)[0] if c.modality == "table" else ""
    return prompts.mentions(title, value) or bool(page and prompts.mentions(page, value))


_DISTINCT = re.compile(r"^(?:[ivxlcdm]+|\d+(?:st|nd|rd|th)?|first|second|third|fourth|fifth|sixth|seventh|eighth|"
                       r"ninth|tenth|junior|senior|jr|sr)$")


def _titled(raw: RawObject) -> str:
    if raw.modality == "table" or not raw.title or raw.text.lstrip().lower().startswith(raw.title.lower()):
        return raw.text
    return f"{raw.title}: {raw.text}"


def restates(key: str, known: str) -> bool:
    if key == known:
        return True
    if fuzz.ratio(key, known) < 90:
        return False
    a, b = key.split(), known.split()
    differ = set(a) ^ set(b)
    if not differ and len(a) == len(b):
        differ = {x for x, y in zip(a, b, strict=True) if x != y}
    return not any(_DISTINCT.match(w) for w in differ)


def _body(c: Candidate) -> str:
    return c.summary.split(": ", 1)[1] if ": " in c.summary else c.summary


def _head(c: Candidate) -> str | None:
    return _table(c) if c.complete else None


def _table(c: Candidate) -> str | None:
    parts = _body(c).split(" | ")
    return " | ".join(parts[:2]) if c.modality == "table" and len(parts) >= 3 else None


def lead_signal(c: Candidate, values: Iterable[str]) -> bool:
    return c.lead_signal if c.lead_signal is not None else any(release_signal(c.summary, v) for v in values)


LEAD_ROOM = 240


def _context(texts: Iterable[str], tokens: int = CONTEXT_TOKENS, items: int = MAX_SNIPPETS) -> list[str]:
    out, used = [], 0
    for t in texts:
        if len(out) >= items:
            break
        n = approx_tokens(t)
        if out and used + n > tokens:
            continue
        out.append(t)
        used += n
    return out


def _context_refs(refs: Sequence[Ref], texts: dict[Ref, str], tokens: int, items: int) -> list[Ref]:
    out, used = [], 0
    for r in refs:
        if len(out) >= items:
            break
        n = approx_tokens(texts[r])
        if out and used + n > tokens:
            continue
        out.append(r)
        used += n
    return out


def _cited(hits: dict[str, list[dict[str, Any]]]) -> list[str]:
    return list(dict.fromkeys(str(h.get("oid", "")) for v in hits.values() for h in v if h.get("oid")))


def _stated(c: Candidate) -> str:
    body = _body(c)
    parts = body.split(" | ", 2) if c.modality == "table" else []
    return parts[2] if len(parts) == 3 else body


def named_in(relation: str, pattern: Pattern) -> list[str]:
    return [x for x in pattern.variables if re.fullmatch(r"[A-Za-z_]+\d+", x.lstrip("?"))
            and re.search(rf"\b{re.escape(x.lstrip('?'))}\b", relation)]


def links(pattern: Pattern) -> list[tuple[str, str]]:
    out = []
    for e in pattern.edges:
        ends = [n for n in (e.head, e.tail) if is_var(n)]
        if len(ends) == 2 and ends[0] != ends[1]:
            out.append((ends[0], ends[1]))
        out += [(end, y) for end in ends for y in named_in(e.relation, pattern) if y != end]
    return out


def holders(pattern: Pattern) -> dict[int, str]:
    return {id(pattern.edges[i]): x for i, x in compiler.marks(pattern, "", pictures=False).holders.items()}


def conditions(pattern: Pattern, question: str, pictured: Collection[tuple[str, str]] = (),
               pictures: bool = True, marks: compiler.Marks | None = None) -> list[Cond]:
    lowered = marks if marks is not None else compiler.marks(pattern, question, pictured, pictures)
    out: list[Cond] = []
    for i, e in enumerate(pattern.edges):
        for end in (e.head, e.tail):
            if is_var(end) and e.other(end) != end and lowered.holders.get(i, end) == end:
                out.append(Cond(len(out), end, e, visual=i in lowered.image, deps=lowered.uses.get(i, ())))

    def visual_filter(x: str, text: str, fallback: bool = False) -> None:
        f = Cond(len(out), x, None, text, visual=True, fallback=fallback)
        out.append(f)
        for e in pattern.incident(x):
            if not is_var(e.other(x)):
                out.append(Cond(len(out), x, e, "" if fallback else text, True, fallback, hop=True, parent=f.id))

    for x, texts in pattern.conditions.items():
        for t in texts:
            if (x, t) in lowered.pictured:
                visual_filter(x, t)
            else:
                out.append(Cond(len(out), x, None, t))
    if lowered.question is not None:
        visual_filter(lowered.question, FALLBACK_VISUAL, fallback=True)
    return out


def routed(pattern: Pattern, question: str) -> list[tuple[str, Any, str]]:
    out: list[tuple[str, Any, str]] = []
    for i, e in enumerate(pattern.edges):
        if question not in (e.head, e.tail):
            out.append(("edge", i, f"{e.head.lstrip('?')} | {e.relation} | {e.tail.lstrip('?')}"))
    for x, texts in pattern.conditions.items():
        out += [("unary", (x, t), f"{x.lstrip('?')} {t}") for t in texts]
    return out


def visual_messages(question: str, phrases: Sequence[str]) -> list[dict[str, Any]]:
    lines = "\n".join(f"({i}) {p}" for i, p in enumerate(phrases, 1))
    return [{"role": "system", "content": VISUAL_SYSTEM},
            {"role": "user", "content": VISUAL_USER.format(question=question, conditions=lines)}]


def parse_pictures(text: str, n: int) -> list[int] | None:
    obj = parse_json(text)
    if isinstance(obj, dict):
        obj = obj.get("picture", obj.get("pictures"))
    if isinstance(obj, int | float | str) and not isinstance(obj, bool):
        obj = [obj]
    if not isinstance(obj, list):
        return None
    out: set[int] = set()
    for v in obj:
        s = v.strip().strip("()") if isinstance(v, str) else ""
        k = int(v) if isinstance(v, int | float) and not isinstance(v, bool) and float(v).is_integer() else \
            int(s) if s.isdigit() else 0
        if 1 <= k <= n:
            out.add(k)
    return sorted(out)


async def route_pictures(pattern: Pattern, question: str, ctx: QueryContext,
                         ) -> tuple[Pattern, frozenset[tuple[str, str]]]:
    items = routed(pattern, question)
    if not items:
        return pattern, frozenset()
    regex = [phrase for kind, ref, phrase in items
             if (pattern.edges[ref].modality == "image" if kind == "edge" else VISUAL.search(ref[1]))]
    messages = visual_messages(question, [phrase for _, _, phrase in items])
    try:
        reply = await ctx.llm.chat(messages, role="pattern", task="visual", max_tokens=200, ledger=ctx.ledger,
                                   charge="pattern")
        picked = parse_pictures(reply.text, len(items))
    except Exception:
        picked = None
    if picked is None:
        ctx.log("visual", picture=[], regex=regex, failed=True)
        return pattern, frozenset()
    routed_pattern, pictured = copy.deepcopy(pattern), set()
    for n in picked:
        kind, ref, _ = items[n - 1]
        if kind == "edge":
            routed_pattern.edges[ref].modality = "image"
        else:
            pictured.add(ref)
    ctx.log("visual", picture=[items[n - 1][2] for n in picked], regex=regex)
    return routed_pattern, frozenset(pictured)


async def holds_pictures(ctx: QueryContext) -> bool:
    return any(c.modality == "image" for si in (await ctx.fed.info()).values() for c in si.collections)


GROUND_TITLES = 6
GROUND_WORDS = 14


def ground_targets(pattern: Pattern, question: str) -> list[str]:
    out: dict[str, None] = {}
    for e in pattern.edges:
        if e.modality != "image":
            for n in (e.head, e.tail):
                if not is_var(n) and n.strip() and n != question:
                    out[n] = None
    return list(out)


def joins(cands: Sequence[Candidate], anchor: str) -> bool:
    return any(c.mentions or release_signal(c.summary, anchor) for c in cands)


def entities(cands: Sequence[Candidate], salt: str, n: int = GROUND_TITLES) -> list[tuple[str, str]]:
    out: dict[str, tuple[str, str]] = {}
    for c in rrf(split_lists(cands), salt=salt):
        title = c.summary.split(": ", 1)[0].strip()
        if c.modality == "table":
            title = (_body(c).split(" | ", 1)[0] if ": " in c.summary else title.split(" — ", 1)[0]).strip()
        if title and _key(title) not in out:
            words = " ".join(_body(c).split()[:GROUND_WORDS]) if ": " in c.summary else ""
            out[_key(title)] = (title, f"{title}: {words}" if words else title)
            if len(out) >= n:
                break
    return list(out.values())


def relinked(pattern: Pattern, links: dict[str, str]) -> Pattern:
    out = copy.deepcopy(pattern)
    for e in out.edges:
        e.head, e.tail = links.get(e.head, e.head), links.get(e.tail, e.tail)
    out.anchors = list(dict.fromkeys(links.get(a, a) for a in out.anchors))
    return out


INDEX_CLASSES = ("same", "entity", "silent_same", "silent_entity")
CACHED = ("verify", "index")


@register("fedquest")
class FedQuest(Method):
    name = "fedquest"
    added_params: ClassVar[dict[str, Any]] = {"graph": "mgrag", "probe": "topk", "limits": "fixed", "route": "all",
                                              "ties": "name", "calibration": None, "direct": "off",
                                              "context": "chain", "pages": 2, "router": False, "compile_fix": False,
                                              "mention": 1, "scores": "rank", "zcal": None, "kg": "off",
                                              "kg_check": "version", "harvest_n": 3, "epoch": None, "kg_ns": None,
                                              "kg_write": True, "cache": "off", "harvest_policy": "fixed",
                                              "recover": "direct", "anchor_titles": False, "views": False,
                                              "view_key": "edge", "room": 240,
                                              "total_budget": None, "context_max": None, "scale_limits": True,
                                              "route_policy": "threshold", "route_fit": None,
                                              "owners": "off", "keep_probe": False}

    def __init__(self, tokens: int | None = 8000, objects: int = 24, k: int = 5, width: int = 1, order: str = "calib",
                 size: str | int = "plan", keys: str = "pattern", calibrated: bool = True, place: str = "site",
                 include_raw: bool = True, max_rounds: int = 6, scan: bool = True,
                 planner: Literal["query", "hop"] = "hop", visual: Literal["llm", "regex"] = "llm",
                 costs: Literal["measured", "v12"] = "measured", mentions: bool = True,
                 titles: Literal["all", "none"] = "all", beam: int | None = 1, rollout: bool = True,
                 semijoin: bool = True, charge: Literal["shadow", "realized"] = "shadow",
                 repair: bool = False, repair_rounds: int = 3,
                 repair_on: Literal["fail", "fail+abstain"] = "fail+abstain", repair_floor: int = 1500,
                 reuse: Literal["exact", "verify", "index"] = "exact", negatives: bool = True, probe_cache: bool = True,
                 rel_sim: float = 0.85, bloom: bool = False, ground: bool = False,
                 retrieve: Literal["fixed", "elastic"] = "fixed", k_max: int = 40, page_theta: float = 0.1,
                 graph: Literal["mgrag", "typed"] = "mgrag", probe: Literal["topk", "semijoin"] = "topk",
                 limits: Literal["fixed", "share"] = "fixed",
                 route: Literal["all", "vocab", "proto", "vocab+proto"] = "all",
                 ties: Literal["name", "salted"] = "name", calibration: str | None = None,
                 direct: Literal["off", "fail", "always", "budget"] = "off", context: Literal["chain", "union"] = "chain",
                 pages: int = 2, router: bool = False, compile_fix: bool = False, mention: int = 1,
                 scores: Literal["rank", "anchor", "ranknaive"] = "rank", zcal: str | None = None,
                 kg: Literal["off", "reuse", "harvest"] = "off",
                 kg_check: Literal["version", "trust", "witness"] = "version",
                 harvest_n: int = 3, epoch: str | None = None, kg_ns: str | None = None,
                 kg_write: bool = True, cache: Literal["off", "tuple"] = "off",
                 harvest_policy: Literal["fixed", "admit"] = "fixed",
                 recover: Literal["direct", "chain"] = "direct", anchor_titles: bool = False,
                 views: bool = False, view_key: Literal["edge", "question"] = "edge", room: int = 240,
                 total_budget: int | None = None, context_max: int | None = None, scale_limits: bool = True,
                 route_policy: Literal["threshold", "topb"] = "threshold", route_fit: str | None = None,
                 owners: Literal["off", "titles", "titles+kg"] = "off", keep_probe: bool = False) -> None:
        super().__init__()
        probe = {"filtered": "semijoin", "dense": "topk"}.get(probe, probe)  # type: ignore[assignment]
        if scores not in ("rank", "anchor", "ranknaive") or kg not in ("off", "reuse", "harvest") \
                or kg_check not in ("version", "trust", "witness") or harvest_n < 0:
            raise ValueError(f"unknown scores {scores!r}, kg {kg!r}, kg_check {kg_check!r} or harvest_n {harvest_n!r}")
        if scores == "anchor" and not zcal:
            raise ValueError('scores="anchor" prices candidates by a z calibration: name it (zcal: <family>_z)')
        self.scores, self.zcal, self.kg, self.kg_check, self.harvest_n = scores, zcal, kg, kg_check, int(harvest_n)
        self.epoch = epoch
        self.kg_ns = self.kg_namespace = kg_ns
        self.kg_write = bool(kg_write)
        if harvest_policy not in ("fixed", "admit"):
            raise ValueError(f"unknown harvest_policy {harvest_policy!r}")
        self.harvest_policy = harvest_policy
        if cache not in ("off", "tuple"):
            raise ValueError(f"unknown cache {cache!r}: off or tuple")
        self.cache = cache
        self._zc: dict[str, Any] = {}
        sized = isinstance(size, dict) and bool(size) and all(
            k in PLACE_KINDS and (v == "cap" or (isinstance(v, int) and not isinstance(v, bool) and v > 0))
            for k, v in size.items())
        if place not in ("site", "coordinator", "extract", "sites", "extract_all") or order not in ("calib", "sim") or \
                keys not in ("pattern", "question", "titles") or not (
                sized or size in ("plan", "unit", "cap") or (isinstance(size, int) and size > 0)):
            raise ValueError(f"unknown place {place!r}, order {order!r}, size {size!r} or keys {keys!r}")
        if charge not in ("shadow", "realized"):
            raise ValueError(f"unknown charge {charge!r}")
        if planner not in ("query", "hop") or visual not in ("llm", "regex") or costs not in ("measured", "v12") or \
                titles not in ("all", "none") or not (beam is None or (isinstance(beam, int) and beam >= 1)):
            raise ValueError(f"unknown planner {planner!r}, visual {visual!r}, costs {costs!r}, titles {titles!r} or "
                             f"beam {beam!r}")
        self.tokens, self.objects, self.k, self.width, self.max_rounds = tokens, objects, k, max(1, width), max_rounds
        self.order, self.size, self.keys, self.calibrated, self.place = order, size, keys, calibrated, place
        self.include_raw, self.scan = include_raw, scan
        self.planner, self.visual, self.costs, self.mentions, self.titles, self.beam = (
            planner, visual, costs, bool(mentions), titles, beam)
        self.rollout, self.semijoin, self.charge = bool(rollout), bool(semijoin), charge
        if repair_on not in ("fail", "fail+abstain") or reuse not in ("exact", "verify", "index") or not 0.0 < rel_sim <= 1.0 \
                or repair_rounds < 1 or repair_floor < 0:
            raise ValueError(f"unknown repair_on {repair_on!r}, reuse {reuse!r}, rel_sim {rel_sim!r}, repair_rounds "
                             f"{repair_rounds!r} or repair_floor {repair_floor!r}")
        self.repair, self.repair_rounds, self.repair_on = bool(repair), int(repair_rounds), repair_on
        self.repair_floor = int(repair_floor)
        self.reuse, self.negatives, self.probe_cache, self.rel_sim = reuse, bool(negatives), bool(probe_cache), rel_sim
        self.bloom, self.ground = bool(bloom), bool(ground)
        if retrieve not in ("fixed", "elastic") or (retrieve == "elastic" and int(k_max) < int(k)) \
                or not 0.0 <= float(page_theta) <= 1.0:
            raise ValueError(f"unknown retrieve {retrieve!r}, or k_max {k_max!r} < k {k!r}, or page_theta "
                             f"{page_theta!r} outside [0, 1]")
        self.retrieve, self.k_max, self.page_theta = retrieve, int(k_max), float(page_theta)
        if graph not in ("mgrag", "typed") or probe not in ("topk", "semijoin") or limits not in ("fixed", "share") \
                or route not in ("all", "vocab", "proto", "vocab+proto") or ties not in ("name", "salted") \
                or direct not in ("off", "fail", "always", "budget") or context not in ("chain", "union"):
            raise ValueError(f"unknown graph {graph!r}, probe {probe!r}, limits {limits!r}, route {route!r}, ties "
                             f"{ties!r}, direct {direct!r} or context {context!r} (YAML reads a bare off as false: "
                             'write direct: "off")')
        if isinstance(pages, bool) or not isinstance(pages, int) or pages < 0:
            raise ValueError(f"pages {pages!r} is not a whole number of pages (0 or more)")
        if calibration is not None and (not isinstance(calibration, str) or not calibration.strip()):
            raise ValueError(f"calibration {calibration!r} is not a calibration name or path")
        if not isinstance(router, bool) or not isinstance(compile_fix, bool):
            raise ValueError(f"router {router!r} or compile_fix {compile_fix!r} is not true or false")
        if isinstance(mention, bool) or mention not in prompts.MENTION_VERSIONS:
            raise ValueError(f"unknown mention test {mention!r}: 1 or 2 (qualifier-free)")
        self.mention = int(mention)
        if recover not in ("direct", "chain") or not isinstance(anchor_titles, bool):
            raise ValueError(f"unknown recover {recover!r} (direct | chain) or anchor_titles {anchor_titles!r}")
        self.recover, self.anchor_titles = recover, anchor_titles
        if not isinstance(views, bool) or view_key not in ("edge", "question") or isinstance(room, bool) \
                or not isinstance(room, int) or room < 40:
            raise ValueError(f"views {views!r}, view_key {view_key!r} (edge | question) or room {room!r} (characters, "
                             "40 or more)")
        self.views, self.view_key, self.room = views, view_key, room
        if retrieve == "elastic" and (probe == "semijoin" or limits == "share"):
            raise ValueError('retrieve="elastic" pages top-k lists of fixed limits: a semijoin pages its own joinable '
                             'objects (pages), share-aware limits size a list by its group')
        self.graph, self.probe, self.limits, self.route, self.ties = graph, probe, limits, route, ties
        self.calibration, self.direct, self.context, self.pages = calibration, direct, context, pages
        self.router, self.compile_fix = router, compile_fix
        if total_budget is not None and (isinstance(total_budget, bool) or int(total_budget) <= 0):
            raise ValueError(f"total_budget {total_budget!r} is not a positive number of cost units")
        self.total_budget = int(total_budget) if total_budget is not None else None
        if route_policy not in ("threshold", "topb"):
            raise ValueError(f"unknown route_policy {route_policy!r}: threshold or topb")
        self.route_policy = route_policy
        if route_fit is not None and (not isinstance(route_fit, str) or not route_fit.strip()):
            raise ValueError(f"route_fit {route_fit!r} is not a fit name")
        self.route_fit = route_fit
        if owners not in ("off", "titles", "titles+kg") or not isinstance(keep_probe, bool):
            raise ValueError(f"unknown owners {owners!r} (off | titles | titles+kg) or keep_probe {keep_probe!r}")
        self.owners, self.keep_probe = owners, keep_probe
        self.context_max = int(context_max) if context_max is not None else None
        self.scale_limits = bool(scale_limits)
        self.direct_objects = DIRECT_OBJECTS
        self.snippet_items = MAX_SNIPPETS
        if self.total_budget is not None:
            self.tokens = self.total_budget
            if self.scale_limits:
                f = max(1.0, self.evidence_room() / EVIDENCE_REF)
                self.objects = max(self.objects, int(round(24 * f)))
                self.max_rounds = max(self.max_rounds, 6 + int(round(2 * math.log2(f))))
                self.pages = max(self.pages, 2 + (1 if f >= 2 else 0))
                self.direct_objects = max(DIRECT_OBJECTS, int(round(DIRECT_OBJECTS * f)))
                self.snippet_items = max(MAX_SNIPPETS, int(round(MAX_SNIPPETS * min(f, 2.0))))

    def context_tokens(self) -> int:
        if self.total_budget is None:
            return CONTEXT_TOKENS
        cap = self.context_max if self.context_max is not None else 6000
        return int(min(cap, max(CONTEXT_TOKENS, 0.15 * self.total_budget)))

    def answer_reserve(self) -> int:
        return ANSWER_FRAME + self.context_tokens()

    def evidence_room(self) -> int:
        return max(0, (self.total_budget or 0) - COMPILE_COST - self.answer_reserve())

    def verify_room(self, ctx: QueryContext) -> bool:
        if self.total_budget is None:
            return False
        left = self.total_budget - ctx.ledger.total_tokens - self.answer_reserve()
        return left >= VERIFY_FRAC * self.total_budget

    def evidence_cap(self, ctx: QueryContext) -> float:
        led = ctx.ledger
        led.reserve = self.answer_reserve()
        left = (self.total_budget or 0) - led.total_tokens - led.reserve
        return float(led.evidence_tokens + max(0, left))

    @property
    def persistent_kg(self) -> bool:
        return self.kg != "off" or self.cache != "off"

    @property
    def versioned_reads(self) -> bool:
        return self.kg != "off" or self.cache != "off"

    def zc(self, qid: str | None = None) -> Any:
        if self.scores != "anchor":
            return None
        from fedquest import xfit

        if xfit.crossfitted(self.zcal) and qid is None:
            raise ValueError(f"zcal {self.zcal!r} is cross-fitted: it resolves per question")
        name = (xfit.resolve(self.zcal, qid) if qid is not None else self.zcal) or ""
        if name not in self._zc:
            from fedquest.planner.zcal import load_zcal

            self._zc[name] = load_zcal(name)
        return self._zc[name]

    @property
    def extracts(self) -> bool:
        return self.place in ("extract", "extract_all")

    @property
    def audited(self) -> bool:
        return self.graph == "typed"

    async def answer(self, q: Question, ctx: QueryContext) -> MethodResult:
        with prompts.mention_scope(self.mention):
            return await self._answer(q, ctx)

    async def _answer(self, q: Question, ctx: QueryContext) -> MethodResult:
        if self.calibration is not None:
            from fedquest import xfit

            ctx = self.recalibrated(ctx, xfit.resolve(self.calibration, q.qid) or self.calibration)
        pictured: frozenset[tuple[str, str]] = frozenset()
        compiled: compiler.CompiledQuery | None = None
        if self.graph == "typed":
            if self.total_budget is not None:
                ctx.ledger.reserve = self.answer_reserve()
            try:
                compiled = await compiler.compile_question(q, ctx, fix=self.compile_fix)
            except BudgetExhausted:
                ctx.log("budget_stop", at="compile")
                compiled = await compiler.budget_fallback(q.question, ctx)
            pattern, pictured = compiled.pattern, compiled.pictured
            if self.router and self.visual == "llm" and await holds_pictures(ctx):
                pattern, routed = await route_pictures(pattern, q.question, ctx)
                pictured = pictured | routed
                compiled = compiler.relabeled(compiled, pattern, pictured, q.question)
        else:
            pattern = await make_pattern(q, ctx)
            if not pattern.variables:
                pattern = prompts.parse_pattern("", q.question)
            pattern = prune(pattern, q.question)
            if self.visual == "llm" and await holds_pictures(ctx):
                pattern, pictured = await route_pictures(pattern, q.question, ctx)
        if self.ground:
            pattern = await self.grounded(pattern, q, ctx)
        run = _Run(self, q, ctx, pattern, pictured, compiled=compiled)
        await run.go()
        if self.direct != "off":
            out = await self.directed(q, ctx, run)
        elif not self.repair:
            out = await run.finish()
        else:
            out = await self.repaired(q, ctx, run)
        if self.audited:
            run.invariants()
        return out

    async def directed(self, q: Question, ctx: QueryContext, run: _Run) -> MethodResult:
        if single_edge(run.p, q.question):
            ctx.log("direct", ran=False, why="fallback graph")
            return await (self.repaired(q, ctx, run) if self.repair else run.finish())
        first: MethodResult | None = None
        trigger = "always" if self.direct == "always" else None
        if trigger is None:
            if not run.stop_test():
                trigger = "fail"
            elif self.direct == "budget" and self.verify_room(ctx):
                trigger = "verify"
            elif self.repair_on != "fail+abstain":
                return await run.finish()
            else:
                first = await run.finish(cache=False)
                if not is_abstention(first.answer):
                    await run.remember(facts=False)
                    return first
                trigger = "abstain"
        if first is None:
            run.log_stop(run=1)
            await run.remember(cache=False)
        if self.recover == "chain":
            direct: _Run = run
            for step in range(CHAIN_ROUNDS):
                cur = _Chain(self, q, ctx, direct, step)
                await cur.go()
                ctx.ledger.count_repair("chain_rounds")
                direct = cur
                if cur.stop_test() or not cur.fresh or not cur.can_repair():
                    break
        else:
            direct = _Direct(self, q, ctx, run)
            await direct.go()
        ctx.ledger.count_repair("direct_ran")
        if direct.stop_test():
            ctx.ledger.count_repair("direct_matched")
        ctx.log("direct", trigger=trigger, candidates=len(run.question_top), matched=direct.stop_test(),
                read=direct.spent - run.spent, rounds=direct.rounds)
        if self.repair:
            return await self.repaired(q, ctx, run, last=direct)
        out = await direct.finish()
        await run.remember(facts=False)
        return out

    def recalibrated(self, ctx: QueryContext, name: str) -> QueryContext:
        cal = load_calibration(name)
        ctx.log("calibration", name=name, **cal.source)
        return replace(ctx, calib=cal)

    async def grounded(self, pattern: Pattern, q: Question, ctx: QueryContext) -> Pattern:
        targets = ground_targets(pattern, q.question)
        if not targets:
            return pattern
        led = ctx.ledger
        led.count_ground("ground_anchors", len(targets))
        learned = await ctx.kg.aliases(ctx.ns, targets)
        links = {a: learned[norm_entity(a)] for a in targets if norm_entity(a) in learned}
        rest = [a for a in targets if a not in links]
        known = await ctx.kg.known(ctx.ns, rest) if rest else set()
        cached = [a for a in rest if norm_entity(a) in known]
        rest = [a for a in rest if a not in cached]
        led.count_ground("ground_cached", len(links) + len(cached))
        textual = {(s, c.name) for s, si in (await ctx.fed.info()).items() for c in si.collections
                   if c.modality != "image"}
        titled: list[str] = []
        if rest:
            filters = [bf for s, cols in (await ctx.fed.blooms()).items() for c, bf in cols.items() if (s, c) in textual]
            titled = [a for a in rest if (k := title_key(a)) and any(f"t:{k}" in bf for bf in filters)]
            rest = [a for a in rest if a not in titled]
        led.count_ground("ground_titled", len(titled))
        unjoined: dict[str, list[tuple[str, str]]] = {}
        probed = rest if textual else []
        if probed:
            names = sorted({c for _, c in textual})
            with timeline.phase("plan"):
                found = await ctx.fed.probe([ProbeRequest(a, "bound", self.k, collections=names, mentions=[a],
                                                          value=a, mention=self.mention) for a in probed], ledger=led)
            led.count_ground("ground_probed", len(probed))
            unjoined = {a: entities(cands, q.qid) for a, cands in zip(probed, found, strict=True)
                        if not joins(cands, a)}
            led.count_ground("ground_unjoined", len(unjoined))
        shown = {a: es for a, es in unjoined.items() if es}
        new: dict[str, str] = {}
        failed = False
        if shown:
            messages = prompts.ground_messages(q.question, [(a, [line for _, line in es]) for a, es in shown.items()])
            try:
                reply = await ctx.llm.chat(messages, role="pattern", task="ground", max_tokens=120, ledger=led,
                                           charge="ground")
                picks = prompts.parse_links(reply.text, [len(es) for es in shown.values()])
            except Exception:
                picks = None
            led.count_ground("ground_calls")
            failed = picks is None
            for i, (a, es) in enumerate(shown.items(), 1):
                j = (picks or {}).get(i, 0)
                if j and norm_entity(es[j - 1][0]) != norm_entity(a):
                    new[a] = es[j - 1][0]
            if new:
                await ctx.kg.add_aliases(ctx.ns, new)
        links |= new
        led.count_ground("ground_linked", len(links))
        ctx.log("ground", anchors=targets, cached=[*(a for a in links if a not in new), *cached], titled=titled,
                probed=probed, unjoined=list(unjoined), linked=links, **({"failed": True} if failed else {}))
        return relinked(pattern, links) if links else pattern

    async def repaired(self, q: Question, ctx: QueryContext, run: _Run, last: _Run | None = None) -> MethodResult:
        led = ctx.ledger
        tail = last or run
        trigger, first = (None, None) if run.stop_test() or (last is not None and last.stop_test()) else ("fail", None)

        async def unrepaired() -> MethodResult:
            if first is None:
                out = await tail.finish()
            else:
                await tail.remember(facts=False)
                out = first
            if last is not None:
                await run.remember(facts=False)
            return out

        if trigger is None:
            if self.repair_on != "fail+abstain":
                return await unrepaired()
            first = await tail.finish(cache=False)
            if not is_abstention(first.answer):
                return await unrepaired()
            trigger = "abstain"

        led.count_repair(f"repair_{trigger}")
        if not tail.can_repair():
            ctx.log("repair", trigger=trigger, ran=False, why="budget")
            return await unrepaired()
        graph, facts, problem = run.diagnosis(trigger)
        pictured: frozenset[tuple[str, str]] = frozenset()
        compiled: compiler.CompiledQuery | None = None
        if self.graph == "typed":
            try:
                compiled = await compiler.compile_repair(q.question, graph, facts, problem, ctx)
            except BudgetExhausted:
                ctx.log("repair", trigger=trigger, ran=False, why="budget")
                return await unrepaired()
            repaired, pictured = compiled.pattern, compiled.pictured
            led.count_repair("repair_calls")
            new = run.new_edges(repaired) if repaired.variables else []
            if new and self.router and self.visual == "llm" and await holds_pictures(ctx):
                repaired, routed = await route_pictures(repaired, q.question, ctx)
                pictured = pictured | routed
                compiled = compiler.relabeled(compiled, repaired, pictured, q.question)
        else:
            try:
                reply = await ctx.llm.chat(prompts.repair_messages(q.question, graph, facts, problem),
                                           role="pattern", task="repair", max_tokens=600, ledger=led,
                                           charge="repair")
            except BudgetExhausted:
                ctx.log("repair", trigger=trigger, ran=False, why="budget")
                return await unrepaired()
            led.count_repair("repair_calls")
            repaired = prune(hint_modalities(fold_media(prompts.parse_pattern(reply.text, q.question))), q.question)
            new = run.new_edges(repaired) if repaired.variables else []
            if new and self.visual == "llm" and await holds_pictures(ctx):
                repaired, pictured = await route_pictures(repaired, q.question, ctx)
        edges = [f"{e.head} | {e.relation} | {e.tail}" for e in repaired.edges]
        if not new:
            ctx.log("repair", trigger=trigger, ran=False, why="nothing new", edges=edges)
            return await unrepaired()
        kinds = compiler.edits(run.p, run.bound, repaired) if compiled is not None and self.audited else None
        ctx.log("repair", trigger=trigger, ran=True, edges=edges, new=len(new), answer=repaired.answer,
                diagnosis=graph, **({"edits": kinds} if kinds is not None else {}))
        led.count_repair("repair_ran")
        for kind in kinds or ():
            led.count_repair(f"repair_edit_{kind}")
        if first is None:
            tail.log_stop(run=1 if last is None else "direct")
            await tail.remember(cache=False)
        if self.ground:
            repaired = await self.grounded(repaired, q, ctx)
        again = _Run(self, q, ctx, repaired, pictured, prior=tail, compiled=compiled)
        if last is not None:
            again.tokens = tail.tokens
        await again.go()
        led.count_repair("repair_rounds", again.rounds)
        out = await again.finish()
        await tail.remember(facts=False)
        if last is not None:
            await run.remember(facts=False)
        return out


for _order, _size in product(("calib", "sim"), ("plan", "unit", "cap")):
    register("fedquest_q" + ("_sim" if _order == "sim" else "") + ("" if _size == "plan" else f"_{_size}"),
             keys="question", order=_order, size=_size, scan=False, planner="hop")(FedQuest)


class _Run:
    RUN: ClassVar[int | str] = 2
    want_z: bool = False

    def __init__(self, m: FedQuest, q: Question, ctx: QueryContext, pattern: Pattern,
                 pictured: Collection[tuple[str, str]] = (), prior: _Run | None = None,
                 compiled: compiler.CompiledQuery | None = None) -> None:
        self.m, self.q, self.ctx, self.p = m, q, ctx, pattern
        self.max_rounds, self.tokens = m.max_rounds, m.tokens
        self.prior = prior
        self.compiled = compiled
        self.marks = compiled.marks if compiled is not None else compiler.marks(pattern, q.question, pictured)
        self.edge_index = {id(e): i for i, e in enumerate(pattern.edges)}
        self.conds = conditions(pattern, q.question, marks=self.marks)
        self.root: _Run = prior.root if prior is not None else self
        if prior is None:
            self.family: list[_Run] = []
            self.breaches: dict[str, list[str]] = defaultdict(list)
            self.witnessable: set[Ref] = set()
            self.zero: set[tuple[str, str, str]] = set()
            self.overs: dict[Ref, str] = {}
            self.harvested: list[Fact] = []
            self.homes: dict[str, set[str]] = defaultdict(set)
        self.root.family.append(self)
        self.by_var: dict[str, list[Cond]] = defaultdict(list)
        for c in self.conds:
            self.by_var[c.var].append(c)
        self.values: dict[str, dict[str, Held]] = defaultdict(dict)
        self.bound: dict[str, list[str]] = {}
        self.pi: dict[str, dict[str, float]] = {}
        self.partial: set[str] = set()
        self.dead: set[str] = set()
        self.asked: set[tuple[int, str]] = set()
        self.read: list[Ref] = []
        self.spent = 0
        self.unresolved: set[int] = set()
        self.loose = False
        tied, todo = {pattern.answer} if pattern.answer in pattern.variables else set(), [pattern.answer]
        pairs = links(pattern)
        while todo:
            x = todo.pop()
            for a, b in pairs:
                for y in ((b,) if a == x else (a,) if b == x else ()):
                    if y not in tied:
                        tied.add(y)
                        todo.append(y)
        self.connected = tied >= set(pattern.variables)
        self.seen: set[tuple[int, str, Ref]] = set()
        self.entries: set[tuple[int, str, Ref]] = set()
        self.p_read: dict[Ref, float] = {}
        self.reused: list[Ref] = []
        self.reused_modality: dict[Ref, str] = {}
        self.made: dict[tuple[int, str], Asked] = {}
        self.released: dict[Ref, Candidate] = {}
        self.probed: dict[str, list[Candidate]] = {}
        self.memoed: set[tuple[str, str, bool]] = set()
        self.negated: set[tuple[int, str, Ref]] = set()
        self.skipped: set[Ref] = set()
        self.decisive: set[tuple[str, str]] = set()
        self.filtered: dict[str, set[str]] = defaultdict(set)
        self.reasked: dict[str, int] = {}
        self.shadow = 0.0
        self.indexed: list[Ref] = []
        self.entries_read: dict[Ref, Candidate] = {}
        self.question_top: list[Candidate] = []
        self.list_cap = LIST_CAP if m.scan else 12
        self.snippets: dict[Ref, str] = {}
        self.doc_ids: dict[Ref, str] = {}
        self.question_probed = False
        self.rounds = 0
        self.free = 0
        self.titled: set[str] = set()
        self.cols: dict[str, dict[str, str]] = {}
        self.reserves: dict[tuple[str, bool, str, str], float] = {}
        self.curves: dict[tuple[bool, str, str], list[tuple[float, float]]] = {}
        self._nodes: dict[int, str | None] = {}
        self.kappas: dict[tuple[Any, ...], float] = {}
        self.witnesses: dict[tuple[str, int], dict[Ref, set[tuple[str, str]]]] = defaultdict(dict)
        self.stops: dict[str, str] = {}
        self.depth: set[int] = set()
        self.cut: set[str] = set()
        self.pruned: dict[str, list[str]] = {}
        self.coverage = 1.0
        self.reach = m.k
        self.round_lists: list[tuple[ProbeRequest, tuple[str, str | None, tuple[int, ...], str | None]]] = []
        self.deferred: list[tuple[int, list[str]]] = []
        self.sizes: dict[tuple[str, str], tuple[str, int]] = {}
        self.groups: Counter[tuple[str, str]] = Counter()
        self.round_kind = ""
        self.counted: dict[int, tuple[list[ProbeRequest], list[ProbeCounts]]] = {}
        self.salt = q.qid if m.ties == "salted" else None
        self.hop_start = 0.0
        self.round_open = False
        self.round_hit = False
        self.returning: set[str] = set()
        self.cached_refs: set[Ref] = set()
        self.unmentioned = 0
        self.limit: str | None = None
        self.keys: KeyIndex | None = prior.keys if prior is not None else None
        self.istats: dict[str, tuple[int, int]] | None = prior.istats if prior is not None else None
        self.ientries: dict[int, dict[Ref, str]] = {}
        self.iextra: dict[tuple[int, Ref], tuple[Candidate, str, str, float, float]] = {}
        self.iclass: dict[tuple[int, Ref], str] = {}
        self.icounted: set[tuple[int, Ref]] = set()
        self.plan_cid = -1
        self.ideltas: dict[str, list[float]] = defaultdict(lambda: [0, 0, 0.0])
        self.iprior: dict[tuple[int, Ref], float] = {}
        cal = ctx.calib if ctx is not None else None
        self.out_empty = cal.read_cost("out_empty", OUT_EMPTY) if cal is not None else OUT_EMPTY
        self.out_hit = cal.read_cost("out_hit", OUT_HIT) if cal is not None else OUT_HIT
        if prior is not None:
            self.max_rounds = m.repair_rounds
            if m.tokens is not None:
                self.tokens = max(m.tokens, math.ceil(prior.evidence()) + m.repair_floor)
            self.read, self.spent, self.indexed = list(prior.read), prior.spent, list(prior.indexed)
            self.snippets, self.doc_ids, self.p_read = dict(prior.snippets), dict(prior.doc_ids), dict(prior.p_read)
            self.entries_read, self.question_top = dict(prior.entries_read), list(prior.question_top)
            self.reused, self.reused_modality = list(prior.reused), dict(prior.reused_modality)
            self.released, self.probed = dict(prior.released), dict(prior.probed)
            self.question_probed, self.unmentioned = prior.question_probed, prior.unmentioned
        if m.total_budget is not None and ctx is not None:
            self.tokens = m.evidence_cap(ctx) if prior is None else min(self.tokens, m.evidence_cap(ctx))

    def rep(self, c: Candidate) -> str | None:
        if c.view:
            return f"{'qview' if self.m.view_key == 'question' else 'view'}@{self.m.room}"
        return f"lead@{self.m.room}" if self.m.room != LEAD_ROOM else None

    def values_of(self, node: str) -> list[str]:
        return self.bound.get(node, []) if is_var(node) else [node]

    def weight(self, node: str | None, value: str | None) -> float:
        if node is None or value is None or not is_var(node):
            return 1.0
        return self.pi.get(node, {}).get(norm_entity(value), 1.0)

    def owns(self, c: Cond) -> bool:
        if c.hop:
            return False
        if c.edge is None:
            return True
        i = self.edge_index.get(id(c.edge))
        return c.var == (self.marks.owners[i] if i is not None and i in self.marks.owners else compiler.owner(c.edge))

    def status(self, c: Cond) -> str:
        other = c.other or ""
        if c.edge is not None and is_var(other):
            if other in self.dead:
                return "void"
            if other not in self.bound:
                return "later"
        if c.id in self.unresolved:
            return "void"
        if not self.loose and any(self.alive(y) for y in c.deps):
            return "later"
        if c.parent is not None:
            filt = self.status(self.conds[c.parent])
            if filt in ("pending", "later"):
                return "later"
            if filt == "done":
                return "void"
        if c.edge is None:
            asked = (c.id, "") in self.asked
        else:
            asked = all((c.id, norm_entity(v)) in self.asked for v in self.values_of(other))
        if asked:
            return "done"
        if c.visual and self.owns(c):
            text = [d for d in self.by_var[c.var] if not d.visual]
            if any(self.status(d) in ("pending", "later") for d in text) and not self.early(c.var, text):
                return "later"
            if text and not self.values[c.var]:
                return "void"
        return "pending"

    def early(self, x: str, text: Sequence[Cond]) -> bool:
        if self.m.planner != "query":
            return False
        asked = [d for d in text if self.status(d) == "done" or self.begun(d)]
        return sum(any(self.satisfies(h, d) for d in asked) for h in self.values[x].values()) >= 2

    def begun(self, c: Cond) -> bool:
        if self.m.planner != "query" or c.edge is None or not is_var(c.other or ""):
            return False
        return any((c.id, norm_entity(v)) in self.asked for v in self.bound.get(c.other or "", []))

    def satisfies(self, held: Held, c: Cond) -> bool:
        got = held.witnessed.get(c.id)
        if not got:
            return False
        if c.edge is None or not is_var(c.other or ""):
            return True
        return bool(got & {norm_entity(v) for v in self.bound.get(c.other or "", [])})

    def evaluated(self, x: str, now: Iterable[int] = ()) -> list[Cond]:
        now = set(now)
        return [c for c in self.by_var[x] if c.id not in now and (self.status(c) == "done" or self.begun(c))]

    def full(self, x: str, now: Iterable[int] = ()) -> list[Held]:
        conds = self.evaluated(x, now)
        return [h for h in self.values[x].values() if conds and all(self.satisfies(h, c) for c in conds)]

    def candidates(self, x: str, now: Iterable[int] = (), skip: Iterable[str] = ()) -> list[str]:
        skip = set(skip)
        held = self.full(x, now) or list(self.values[x].values())
        return [h.name for h in held if _key(h.name) not in skip][:self.list_cap]

    def best(self, x: str) -> list[Held]:
        conds = self.evaluated(x)
        score = [((sum(self.satisfies(h, c) for c in conds), any(c.visual and self.satisfies(h, c) for c in conds)), h)
                 for h in self.values[x].values()]
        top = max((s for s, _ in score), default=(0, False))
        return [h for s, h in score if s == top and s[0] > 0]

    def alive(self, x: str) -> bool:
        return x not in self.bound and x not in self.dead

    def decide(self) -> None:
        changed = True
        while changed:
            changed = False
            for x in self.p.variables:
                if not self.alive(x) or any(self.status(c) in ("pending", "later") for c in self.by_var[x]):
                    continue
                full, best = self.full(x), self.best(x)
                if not full and best and self.reask(x):
                    changed = True
                    continue
                # values meeting every evaluated condition, else those meeting the most
                chosen = full or best
                if chosen:
                    pi = self.chances(chosen)
                    chosen = sorted(chosen, key=lambda h: -pi[_key(h.name)])
                    self.bound[x] = [h.name for h in chosen]
                    self.pi[x] = {norm_entity(h.name): pi[_key(h.name)] for h in chosen}
                    if not full:
                        self.partial.add(x)
                        if self.stops.get(x) == "allocation":
                            self.cut.add(x)
                    self.ctx.log("bound", var=x, values=self.bound[x][:self.list_cap],
                                 pi=[round(pi[_key(h.name)], 3) for h in chosen][:self.list_cap], partial=not full)
                else:
                    self.dead.add(x)
                    self.ctx.log("bound", var=x, values=[], pi=[], partial=False)
                changed = True

    def chances(self, helds: Sequence[Held]) -> dict[str, float]:
        mass = {_key(h.name): sum(1.0 if ref in self.reused else self.p_read.get(ref, 0.0)
                                  for ref in dict.fromkeys(h.refs)) for h in helds}
        total = sum(mass.values())
        return {k: (v / total if total > 0 else 1.0 / len(mass)) for k, v in mass.items()}

    def answered(self) -> list[str]:
        if self.compiled is not None:
            return list(self.compiled.awaited)
        if self.p.answer is not None and not self.p.raw.get("answer_inferred") and self.connected:
            return [self.p.answer]
        return list(self.p.variables)

    def targets(self) -> list[str]:
        if self.p.answer is not None and not self.p.raw.get("answer_inferred"):
            return [self.p.answer]
        return list(self.p.variables)

    def upstream(self) -> set[str]:
        pairs, seen = links(self.p), set(self.answered())
        todo = list(seen)
        while todo:
            x = todo.pop()
            for a, b in pairs:
                for y in ((b,) if a == x else (a,) if b == x else ()):
                    if y not in seen:
                        seen.add(y)
                        todo.append(y)
        return seen

    def done(self) -> bool:
        return not any(self.alive(x) for x in self.answered())

    def read_question(self) -> str:
        return self.q.question

    def stop_test(self) -> bool:
        return self.done() and all(x in self.bound and x not in self.partial for x in self.answered()) \
            and not (self.dead & self.upstream())

    def matched(self) -> bool:
        return self.stop_test()

    def can_repair(self) -> bool:
        if self.left() <= 0:
            return False
        return (self.tokens is None or (self.prior is None and self.m.repair_floor >= REPAIR_MIN)
                or self.tokens - self.evidence() >= REPAIR_MIN)

    def _shown(self, node: str) -> str:
        if not is_var(node):
            return node
        vals = self.bound.get(node, [])
        return f"{node.lstrip('?')} = {', '.join(vals[:3])}" if vals else node.lstrip("?")

    def outcome(self, e: Edge) -> str:
        out = []
        for c in (c for c in self.conds if c.edge is e):
            x = c.var.lstrip("?")
            if c.var in self.bound:
                vals = ", ".join(self.bound[c.var][:3])
                out.append(f"found {x} = {vals}" + (" (meets only some of its conditions)" if c.var in self.partial
                                                    else ""))
            elif self.status(c) == "void":
                out.append(f"not searched: {(c.other or '').lstrip('?')} was never found")
            elif self.status(c) == "later":
                out.append(f"not searched yet: waits for {(c.other or '').lstrip('?')}")
            else:
                judged = {ref for cid, _, ref in self.seen if cid == c.id}
                out.append(f"searched {len(judged)} candidate sources for {x}: none states it" if judged else
                           f"searched: no candidate source for {x} was found")
        return "; ".join(dict.fromkeys(out)) or "both ends are given"

    def diagnosis(self, trigger: str) -> tuple[list[str], list[str], str]:
        graph = [f"{i}. ({self._shown(e.head)} | {e.relation} | {self._shown(e.tail)}) -> {self.outcome(e)}"
                 for i, e in enumerate(self.p.edges, 1)]
        chain = {(x, norm_entity(v)) for x, vals in self.bound.items() for v in vals}
        first = [f for x in self.p.variables for held in self.values[x].values()
                 if (x, norm_entity(held.name)) in chain for f in held.facts]
        rest = [f for x in self.p.variables for held in self.values[x].values() for f in held.facts]
        facts = [f"({f.head} | {f.relation} | {f.tail})" for f in dict.fromkeys([*first, *rest])][:24]
        targets = ", ".join(x.lstrip("?") for x in self.answered())
        if trigger == "abstain":
            problem = (f"every unknown the query waits for ({targets}) was found, but the answer could not be given "
                       "from these facts: the graph may end at the wrong entity or miss a relation the query asks for.")
        else:
            dead = [x.lstrip("?") for x in self.p.variables if x in self.dead]
            problem = f"the answer ({targets}) was not found" + (f"; no source was found for {', '.join(dead)}"
                                                                if dead else "") + "."
        return graph, facts, problem

    def _triple(self, e: Edge) -> tuple[str, str, str]:
        def end(node: str) -> str:
            return norm_entity(self.bound[node][0]) if is_var(node) and self.bound.get(node) else \
                ("?" if is_var(node) else norm_entity(node))
        return end(e.head), _relation(e.relation), end(e.tail)

    def new_edges(self, repaired: Pattern) -> list[Edge]:
        done = {self._triple(e) for e in self.p.edges}
        mine = {(norm_entity(e.head) if not is_var(e.head) else "?", _relation(e.relation),
                 norm_entity(e.tail) if not is_var(e.tail) else "?") for e in self.p.edges}
        out = []
        for e in repaired.edges:
            t = (norm_entity(e.head) if not is_var(e.head) else "?", _relation(e.relation),
                 norm_entity(e.tail) if not is_var(e.tail) else "?")
            if t not in done and t not in mine:
                out.append(e)
        return out

    def chain(self) -> list[_Run]:
        runs: list[_Run] = []
        r: _Run | None = self
        while r is not None:
            runs.append(r)
            r = r.prior
        return runs

    def log_stop(self, **extra: Any) -> None:
        self.ctx.log("stop", matched=self.stop_test(), read=self.spent, rounds=self.rounds, bound=self.bound,
                     dead=sorted(self.dead), reused=len(self.reused), limit=None if self.stop_test() else self.limit,
                     **extra)

    def known(self, value: str) -> bool:
        k = _key(value)
        return any(k == _key(a) for r in self.root.family for a in r.p.anchors) or any(
            k in held for r in self.root.family for held in r.values.values())

    def admit(self, reqs: Sequence[ProbeRequest]) -> None:
        if not self.m.audited:
            return
        filters = {t for ts in self.p.conditions.values() for t in ts}
        for r in reqs:
            why = compiler.inadmissible(r, self.q.question, self.known, filters)
            if why is not None:
                self.root.breaches["access"].append(why)

    def capable(self, site: str, cands: Iterable[Candidate]) -> None:
        if not self.m.audited:
            return
        for c in cands:
            if c.site != site:
                self.root.breaches["capability"].append(f"{site} is asked to read {c.site}/{c.oid}")
            elif c.modality == "image" and self.cols.get(site, {}).get(c.collection) != "image":
                self.root.breaches["capability"].append(f"{site} reads the picture {c.oid} outside an image collection")

    def invariants(self) -> dict[str, list[str]]:
        found = compiler.invariants(self.root.family)
        self.ctx.log("invariants", **{name: len(v) for name, v in found.items()},
                     violations=[{"invariant": name, "detail": d} for name, v in found.items() for d in v][:20])
        for name, v in found.items():
            self.ctx.ledger.count_invariant(f"invariant_{name}", len(v))
        return found

    def set_reads(self, unknowns: Sequence[ReadUnknown], calls: int = 1) -> None:
        sets = {str(x).lstrip("?") for x in self.p.raw.get("sets", ())}
        if self.m.audited and sets and any(u.name in sets and u.cap > READ_CAP for u in unknowns):
            self.ctx.ledger.count_fired("set_reads", calls)

    def left(self) -> int:
        return max(0, self.m.objects - self.spent)

    def evidence(self) -> float:
        if self.m.extracts and self.m.charge == "shadow":
            led = self.ctx.ledger
            return (led.coordinator["bind_index"].cost_units if "bind_index" in led.coordinator else 0) + self.shadow
        return self.ctx.ledger.evidence_tokens

    def cap(self, x: str) -> int:
        if self.m.graph == "typed" and x in self.p.raw.get("sets", ()):
            return self.list_cap
        filt = any(c.visual and self.owns(c) and self.status(c) not in ("done", "void") for c in self.by_var[x])
        return self.list_cap if len(self.by_var[x]) >= 2 or filt or self.pictured(x) else READ_CAP

    def pictured(self, x: str) -> list[Cond]:
        if not self.m.scan:
            return []
        return [c for c in self.by_var[x] if c.visual and self.owns(c) and c.parent is None
                and c.id not in self.unresolved and not any(t[0] == c.id for t in self.asked)]

    def judged(self, asks: Iterable[Asked], ref: Ref) -> bool:
        asks = list(asks)
        return bool(asks) and all((a.cond.id, norm_entity(a.value or ""), ref) in self.seen for a in asks)

    def mark(self, asks: Iterable[Asked], refs: Iterable[Ref]) -> None:
        self.seen |= {(a.cond.id, norm_entity(a.value or ""), ref) for ref in refs for a in asks}

    def step(self) -> tuple[str, list[Asked]]:
        known = {a: a for a in self.p.anchors} | {x: v[0] for x, v in self.bound.items() if v}
        query = self.m.planner == "query"
        asks: list[Asked] = []
        beamed: set[int] = set()
        for x in self.p.variables:
            if not self.alive(x):
                continue
            for c in self.by_var[x]:
                if self.status(c) != "pending":
                    continue
                if c.edge is None:
                    asks.append(Asked(x, c, None, c.text))
                    continue
                values = self.values_of(c.other or "")
                fresh = [v for v in values if (c.id, norm_entity(v)) not in self.asked]
                if query and self.m.beam is not None and len(values) >= 2 and self.prunable(x, c):
                    fresh = fresh[:self.width(x, c, len(fresh))]
                    beamed.add(c.id)
                for v in fresh:
                    rel = _grounded(c.edge.relation, self.p, {**known, **({c.other: v} if c.other else {})})
                    asks.append(Asked(x, c, v, f"{rel} X that {c.text}" if c.text else rel))
        text = [a for a in asks if not a.cond.visual]
        if query and (filters := [a for a in asks if a.cond.visual and self.owns(a.cond)]):
            kind, asks = "filter", filters
        elif text:
            kind, asks = "text", text
        else:
            filters = [a for a in asks if self.owns(a.cond)]
            kind, asks = ("filter", filters) if filters else ("visual", asks)
        self.depth = {a.cond.id for a in asks} & beamed
        return kind, asks

    def prunable(self, x: str, c: Cond) -> bool:
        y = c.other or ""
        if self.m.planner != "query" or self.m.beam is None or c.edge is None or not is_var(y):
            return False
        if x in self.targets() and y not in self.partial:
            return False
        rest = [d for d in self.by_var[x] if d.id != c.id]
        if any(self.status(d) in ("pending", "later") for d in rest):
            return False
        done = [d for d in rest if self.status(d) == "done"]
        return not done or any(all(self.satisfies(h, d) for d in done) for h in self.values[x].values())

    def unknowns(self, asks: Sequence[Asked], mode: str = "value", listing: bool = False,
                 ) -> tuple[list[ReadUnknown], dict[str, list[Asked]]]:
        by_var: dict[str, list[Asked]] = defaultdict(list)
        for a in asks:
            by_var[a.var].append(a)
        out = []
        for x, items in by_var.items():
            held = self.candidates(x, [a.cond.id for a in items], self.filtered.get(x, ()) if mode == "filter" else ())
            asked_q = [a.value == self.q.question for a in items]
            later = [] if mode == "filter" or any(a.cond.visual for a in items) else [
                c.phrase(c.other if c.edge is not None and not is_var(c.other or "") else None,
                         c.edge.relation if c.edge is not None else c.text) for c in self.pictured(x)]
            if later or listing:
                held = []
            out.append(ReadUnknown(x.lstrip("?"), ["X answers the question" if q else a.cond.phrase(a.value, a.relation)
                                                   for a, q in zip(items, asked_q, strict=True)],
                                   [a.value if a.cond.edge is not None and not q else None
                                    for a, q in zip(items, asked_q, strict=True)],
                                   READ_CAP if mode == "filter" else self.cap(x), mode, held,
                                   answer=x in self.targets(), later=later))
        return out, by_var

    def merge(self, hits: dict[str, list[dict[str, Any]]], by_var: dict[str, list[Asked]], version: str,
              modality: dict[Ref, str], site: str | None = None) -> list[Ref]:
        accepted: list[Ref] = []
        for name, items in hits.items():
            x = f"?{name}" if f"?{name}" in by_var else name
            asked = by_var.get(x)
            if not asked:
                continue
            known = [_key(a.value) for a in asked if a.value and a.value != self.q.question]
            anchors = [_key(a) for a in self.p.anchors]
            for h in items:
                entity = str(h.get("entity", "")).strip()
                ref = (site or str(h.get("site", "")), str(h.get("oid", "")))
                claimed = [asked[int(n) - 1] for n in h.get("conditions") or [] if 1 <= int(n) <= len(asked)]
                key = _key(entity)
                if (not key or not claimed or not (compiler.admitted(x, entity, self.marks) or x in self.targets())
                        or any(restates(key, k) for k in [*anchors, *known])):
                    continue
                held = self.values[x].setdefault(key, Held(entity))
                if ref not in held.refs:
                    held.refs.append(ref)
                accepted.append(ref)
                if self.ctx is not None:
                    self.ctx.ledger.note_origin(entity, ref[0])
                for a in claimed:
                    other = norm_entity(a.value or "")
                    held.witnessed.setdefault(a.cond.id, set()).add(other)
                    self.witnesses[(x, a.cond.id)].setdefault(ref, set()).add((key, other))
                    fact = a.cond.fact(entity, a.value, a.relation, ref, version, modality.get(ref, ""))
                    if h.get("w"):
                        fact = replace(fact, wspan=str(h["w"]))
                    if fact not in held.facts:
                        held.facts.append(fact)
        if accepted and self.round_open and not self.round_hit and self.ctx is not None:
            self.round_hit = True
            led = self.ctx.ledger
            spent = round(self.evidence() - self.hop_start)
            if spent > 0:
                led.count_reuse("witness_tokens", spent)
                led.count_reuse("witness_rounds")
                led.count_reuse("witness_found")
            else:
                led.count_reuse("witness_free")
        return accepted

    def harvest_size(self, unknowns: Sequence[ReadUnknown]) -> int:
        if self.m.kg != "harvest" or not self.m.harvest_n:
            return 0
        if self.m.harvest_policy == "fixed":
            return self.m.harvest_n
        known = {norm_entity(v) for u in unknowns for v in u.values if v}
        admitted = bool(known & self.root.returning)
        self.ctx.ledger.count_reuse("kg_harvest_admitted" if admitted else "kg_harvest_skipped")
        return self.m.harvest_n if admitted else 0

    def witness_round(self) -> None:
        if self.round_open and not self.round_hit and self.ctx is not None:
            spent = round(self.evidence() - self.hop_start)
            if spent > 0:
                led = self.ctx.ledger
                led.count_reuse("witness_tokens", spent)
                led.count_reuse("witness_rounds")
        self.round_open = False

    def covered(self, x: str) -> bool:
        tried = self.filtered.get(x, set())
        return any((x, k) in self.decisive and k not in tried for k in (_key(h.name) for h in self.full(x)))

    def settle(self, hits: dict[str, list[dict[str, Any]]], by_var: dict[str, list[Asked]],
               decisive: Any = None) -> None:
        for name, items in hits.items():
            x = f"?{name}" if f"?{name}" in by_var else name
            for h in items:
                key = _key(str(h.get("entity", "")))
                if key in self.values.get(x, {}) and (decisive is None or decisive(h)):
                    self.decisive.add((x, key))

    def prefilter(self, asks: Sequence[Asked]) -> list[str]:
        if any(a.cond.edge is None or not a.value or a.value == self.q.question for a in asks):
            return []
        return list(dict.fromkeys(a.value for a in asks if a.value))

    async def setup(self) -> None:
        info = await self.ctx.fed.info()
        self.titled = {s for s, si in info.items() if si.policy in ("facts", "sealed")}
        self.cols = {s: {c.name: c.modality for c in si.collections} for s, si in info.items()}
        self.sizes, self.groups = group_sizes(info)
        if self.m.visual == "llm" and not self.collections(image=True) and any(c.visual for c in self.conds):
            self.marks = compiler.marks(self.p, self.q.question, pictures=False)
            self.conds = conditions(self.p, self.q.question, marks=self.marks)
            self.by_var = defaultdict(list)
            for c in self.conds:
                self.by_var[c.var].append(c)

    async def go(self) -> None:
        try:
            await self._go()
        except BudgetExhausted as exc:
            self.limit = "tokens"
            self.ctx.log("budget_stop", at="run", rounds=self.rounds, why=str(exc)[:200])

    async def _go(self) -> None:
        await self.setup()
        if self.compiled is not None:
            compiler.recorded(self.compiled, self)
        self.decide()
        # a round asks the pending hops with the values bound so far, then rebinds the variables
        while not self.done() and (not self.out_of_rounds() or self.deepening()):
            kind, asks = self.step()
            if not asks and not self.loose:
                self.loose = True
                self.decide()
                kind, asks = self.step()
            if not asks:
                break
            self.rounds += 1
            if self.deeper(asks):
                self.free += 1
            for a in asks:
                self.asked.add((a.cond.id, norm_entity(a.value or "")))
                if a.cond.parent is not None:
                    self.unresolved.add(a.cond.parent)
            self.round_open, self.round_hit = True, False
            led = self.ctx.ledger
            before = self.held_facts()
            ev0, rnd0, msg0 = self.evidence(), led.rounds, led.messages
            try:
                await self.hop(kind, asks)
            finally:
                new = [f for f in self.held_facts() if f not in before]
                self.ctx.log("hop7", round=self.rounds, kind=kind, run=str(self.RUN if self.prior else 1),
                             asks=[{"var": a.var, "value": a.value, "rel": a.relation} for a in asks][:12],
                             tokens=round(self.evidence() - ev0), wan_rounds=led.rounds - rnd0,
                             messages=led.messages - msg0,
                             facts=[{"h": f.head, "r": f.relation, "t": f.tail, "site": f.site, "oid": f.oid,
                                     "ver": f.overs} for f in new][:24])
            self.witness_round()
            self.narrow(asks)
            self.decide()

    def out_of_rounds(self) -> bool:
        return self.rounds - self.free >= self.max_rounds or self.rounds >= self.max_rounds + BEAM_ROUNDS

    def deepening(self) -> bool:
        if self.m.planner != "query" or self.m.beam is None or self.rounds >= self.max_rounds + BEAM_ROUNDS:
            return False
        return self.deeper(self.step()[1])

    def deeper(self, asks: Sequence[Asked]) -> bool:
        if self.m.planner != "query" or self.m.beam is None or not asks:
            return False
        return all(a.cond.edge is not None and is_var(y := a.cond.other or "") and len(self.bound.get(y, [])) >= 2
                   and any((a.cond.id, norm_entity(v)) in self.asked for v in self.bound[y]) for a in asks)

    def narrow(self, asks: Sequence[Asked]) -> None:
        if self.m.planner != "query" or self.m.beam is None:
            return
        for x in dict.fromkeys(a.var for a in asks):
            if not self.covered(x):
                continue
            full = self.full(x)
            for cid in dict.fromkeys(a.cond.id for a in asks if a.var == x and a.cond.id in self.depth):
                y = self.conds[cid].other or ""
                hyps = self.bound.get(y, [])
                if len(hyps) < 2:
                    continue
                produced = {v for h in full for v in h.witnessed.get(cid, ())}
                kept = [v for v in hyps if norm_entity(v) in produced]
                pruned = [v for v in hyps if norm_entity(v) not in produced]
                if not kept or not pruned:
                    continue
                self.asked |= {(cid, norm_entity(v)) for v in pruned}
                pi = self.pi.get(y, {})
                mass = sum(pi.get(norm_entity(v), 0.0) for v in kept)
                self.pi[y] = {norm_entity(v): pi.get(norm_entity(v), 0.0) / mass if mass > 0 else 1.0 / len(kept)
                              for v in kept}
                self.bound[y] = kept
                lost = self.pruned.setdefault(y, [])
                lost += [v for v in pruned if v not in lost]
                self.ctx.log("beam", var=y, kept=kept[:self.list_cap], pruned=pruned[:self.list_cap])

    def collections(self, image: bool) -> list[str]:
        return sorted({c for s in self.cols.values() for c, mod in s.items() if (mod == "image") == image})

    def finished(self, x: str, kind: str) -> bool:
        if kind == "filter":
            return not (self.m.scan and x in self.targets()) and bool(self.full(x))
        return self.covered(x)

    async def hop(self, kind: str, asks: list[Asked]) -> None:
        mode = "filter" if kind == "filter" else "value"
        unknowns, by_var = self.unknowns(asks, mode)
        self.hop_start = self.evidence()
        for x in by_var:
            self.stops.pop(x, None)
        self.ientries, self.iextra = {}, {}
        if kind != "filter":
            if self.m.reuse in CACHED:
                await self.memo(kind, asks)
            else:
                await self.recall(by_var)
            if all(self.covered(a.var) for a in asks):
                if self.m.keep_probe and not self.question_probed and self.m.keys == "pattern":
                    await self.probe_question(kind, asks)
                return
        found = await self.paths(kind, asks)
        pool, outside, coverage = self.pool(kind, asks, found)
        self.coverage = coverage
        if self.negated:
            self.skips(pool, asks)
        if kind == "filter" and not any(it.c.modality == "image" for it in pool.values()):
            self.unresolved |= {a.cond.id for a in asks}
            return
        self.ctx.log("pool", round=kind, var=list(dict.fromkeys(a.var for a in asks)),
                     n=len({it.c.ref for it in pool.values()}), by_kind=dict(Counter(f.kind for f in found)),
                     mass={str(c): round(1.0 - o, 3) for c, o in outside.items()}, coverage=round(coverage, 3),
                     owners=len({it.c.site for it in pool.values()}),
                     owners_able=len({it.c.site for it in pool.values() if it.mentions is not False}))
        if kind == "filter" and self.m.audited:
            self.ctx.ledger.count_fired("image_semijoins")
        if self.deferred:
            self.ctx.ledger.count_fired("owner_hops")
            owned = {self.conds[cid].var for i, _ in self.deferred for cid in self.round_lists[i][1][2]}
            await self.search(pool, outside, asks, kind, calls=OWNER_CALLS * max(1, len(owned)))
            if all(self.finished(a.var, kind) for a in asks):
                self.ctx.ledger.count_fired("owner_bound")
            pool, outside = await self.expand(kind, asks, found, pool, outside)
        await self.search(pool, outside, asks, kind)
        if self.m.retrieve == "elastic" and kind != "filter":
            pool = await self.deepen(kind, asks, found, pool)
        if self.m.probe == "semijoin" and self.m.pages and kind == "text":
            pool = await self.join_pages(kind, asks, found, pool)
        if kind == "text" and self.m.scan:
            await self.complete_units(pool, asks)
        if kind == "filter":
            for x in dict.fromkeys(a.var for a in asks):
                if not self.full(x):
                    ids = [a.cond.id for a in asks if a.var == x]
                    self.filtered[x] |= {_key(n) for n in self.candidates(x, ids, self.filtered[x])}

    async def search(self, pool: dict[tuple[int, Ref], Item], outside: dict[int, float], asks: Sequence[Asked],
                     kind: str, calls: int | None = None) -> None:
        mode = "filter" if kind == "filter" else "value"
        for i in range(SAFETY_CALLS + 1):
            rest = [a for a in asks if not self.finished(a.var, kind)]
            if not rest or (calls is not None and i >= calls):
                break
            if i == SAFETY_CALLS:
                self.limit = "calls"
                break
            unknowns, by_var = self.unknowns(rest, mode)
            chosen = self.plan(pool, outside, rest, unknowns, kind)
            if not chosen or not await self.execute(chosen, pool, rest, unknowns, by_var):
                break

    async def owner_first(self, reqs: list[ProbeRequest],
                          meta: Sequence[tuple[str, str | None, tuple[int, ...], str | None]]) -> None:
        self.deferred = []
        if any(mod == "table" for s in self.cols.values() for mod in s.values()):
            return
        blooms = await self.ctx.fed.blooms()
        for i, (r, (ckind, value, conds, _)) in enumerate(zip(reqs, meta, strict=True)):
            if ckind not in ("bound", "bridge", "title") or not value or not conds:
                continue
            if not all((c := self.conds[cid]).edge is not None and c.edge.head == c.other for cid in conds):
                continue
            names = r.collections
            reach = {s for s in (r.sites if r.sites is not None else self.cols)
                     if any(mod != "image" and (names is None or c in names)
                            for c, mod in self.cols.get(s, {}).items())}
            key = title_key(value)
            titled = {s for s in reach if any(may_hold(bf, key) for c, bf in blooms.get(s, {}).items()
                                              if (names is None or c in names)
                                              and self.cols.get(s, {}).get(c) != "image")}
            if ckind == "title":
                if titled and titled != reach:
                    r.sites = sorted(titled)
                    self.ctx.ledger.count_fired("owner_title_skips", len(reach - titled))
                continue
            own = titled | (self.root.homes.get(norm_entity(value), set()) & reach
                            if self.m.owners == "titles+kg" else set())
            if not own or own == reach:
                continue
            r.sites = sorted(own)
            self.deferred.append((i, sorted(reach - own)))
        if self.deferred:
            self.ctx.log("owners", lists=len(self.deferred),
                         sites={reqs[i].value or reqs[i].key: reqs[i].sites for i, _ in self.deferred})

    async def expand(self, kind: str, asks: Sequence[Asked], found: list[Found], pool: dict[tuple[int, Ref], Item],
                     outside: dict[int, float]) -> tuple[dict[tuple[int, Ref], Item], dict[int, float]]:
        todo, self.deferred = self.deferred, []
        open_ids = {a.cond.id for a in asks if not self.finished(a.var, kind)}
        todo = [(i, sites) for i, sites in todo if open_ids.intersection(self.round_lists[i][1][2])]
        if not todo:
            return pool, outside
        reqs = [replace(self.round_lists[i][0], sites=sites) for i, sites in todo]
        results = await self.probe(reqs, kinds={id(x): self.round_lists[i][1][0]
                                                for (i, _), x in zip(todo, reqs, strict=True)})
        for (i, _), x in zip(todo, reqs, strict=True):
            first = self.round_lists[i][0]
            if id(x) in self.counted:
                parts, counts = self.counted.pop(id(x))
                p0, c0 = self.counted.get(id(first), ([], []))
                self.counted[id(first)] = (p0 + parts, c0 + counts)
        visual_q = self.marks.visual
        new = []
        for (i, _), cands in zip(todo, results, strict=True):
            ckind, value, conds, _ = self.round_lists[i][1]
            new += [Found(c, ckind, self.signal(c, ckind, value, visual_q), value or c.label or None, conds, i)
                    for c in cands]
        found.extend(new)
        if self.m.reuse in CACHED:
            for f in new:
                self.released.setdefault(f.c.ref, f.c)
        self.ctx.ledger.count_fired("owner_expanded", len(reqs))
        self.ctx.log("expand", lists=len(reqs), sites=sorted({s for _, ss in todo for s in ss}), new=len(new))
        if not new:
            return pool, outside
        pool, outside, coverage = self.pool(kind, asks, found)
        self.coverage = coverage
        if self.negated:
            self.skips(pool, asks)
        return pool, outside

    async def probe_question(self, kind: str, asks: Sequence[Asked]) -> None:
        self.question_probed = True
        cols = self.collections(kind != "text")
        if not cols:
            return
        req = ProbeRequest(self.q.question, "question", self.m.k, collections=cols)
        texts = {c for s in self.cols.values() for c, mod in s.items() if mod != "image"}
        if self.m.mentions and texts.intersection(cols):
            req.mentions = list(self.prefilter(asks))
        self.round_kind = kind
        got = (await self.probe([req], kinds={id(req): "question"}))[0]
        if not self.question_top:
            self.question_top = rrf(split_lists(got), salt=self.q.qid)
        if self.m.reuse in CACHED:
            for c in got:
                self.released.setdefault(c.ref, c)
        self.ctx.ledger.count_fired("kept_probes")
        self.ctx.log("keep_probe", n=len(got))

    def full_lists(self, found: Sequence[Found]) -> dict[int, set[str]]:
        n: Counter[tuple[int, str, str]] = Counter()
        for f in found:
            if f.probe < len(self.round_lists):
                n[(f.probe, f.c.site, f.c.collection)] += 1
        out: dict[int, set[str]] = defaultdict(set)
        for (i, site, _), count in n.items():
            if count >= self.reach:
                out[i].add(site)
        return out

    async def deepen(self, kind: str, asks: Sequence[Asked], found: list[Found],
                     pool: dict[tuple[int, Ref], Item]) -> dict[tuple[int, Ref], Item]:
        cal = self.ctx.calib
        image = kind != "text"
        late = any(is_var(a.cond.other or "") for a in asks)
        rnd = "late" if late else "first"
        visual_q = self.marks.visual
        while True:
            open_ids = {a.cond.id for a in asks if not self.finished(a.var, kind)}
            if not open_ids or self.reach >= self.m.k_max or not self.round_lists:
                return pool
            if self.tokens is not None and \
                    self.tokens - self.evidence() < self.prior_call(image, "bridge" if late else "bound", rnd):
                return pool
            full = self.full_lists(found)
            pages = [(i, req, meta, sorted(full[i])) for i, (req, meta) in enumerate(self.round_lists)
                     if req.kind in ("question", "bound") and i in full and set(meta[2]) & open_ids]
            if not pages:
                return pool
            step = min(self.reach, self.m.k_max - self.reach)
            y = cal.page_yield(rnd, image, self.reach, self.reach + step) if self.m.calibrated else 0.0
            if y < self.m.page_theta:
                self.ctx.ledger.count_depth("page_skips")
                self.ctx.log("page_skip", depth=self.reach, y=round(y, 4), lists=len(pages))
                return pool
            reqs = [replace(req, offset=self.reach, k=step, sites=sites) for _, req, _, sites in pages]
            results = await self.probe(reqs)
            new = [Found(c, ckind, self.signal(c, ckind, value, visual_q), value or c.label or None, conds, i)
                   for (i, _, (ckind, value, conds, _), _), cands in zip(pages, results, strict=True) for c in cands]
            found.extend(new)
            if self.m.reuse in CACHED:
                for f in new:
                    self.released.setdefault(f.c.ref, f.c)
            self.reach += step
            led = self.ctx.ledger
            led.count_depth("pages")
            led.count_depth("paged_requests", len(reqs))
            led.count_depth("paged_candidates", len(new))
            led.count_depth("depth_max", self.reach)
            self.ctx.log("page", depth=self.reach, y=round(y, 4), lists=len(reqs), new=len(new))
            if not new:
                return pool
            pool, outside, coverage = self.pool(kind, asks, found)
            self.coverage = coverage
            if self.negated:
                self.skips(pool, asks)
            await self.search(pool, outside, asks, kind)

    def dry(self, pool: dict[tuple[int, Ref], Item], asks: Sequence[Asked], kind: str) -> set[int]:
        rest = [a for a in asks if not self.finished(a.var, kind)]
        if not rest:
            return set()
        unknowns, _ = self.unknowns(rest, "filter" if kind == "filter" else "value")
        values, extra = self.prefilter(rest), self.per_object(unknowns)
        return {cid for cid in dict.fromkeys(a.cond.id for a in rest) if not self.places(pool, rest, cid, values, extra)}

    async def join_pages(self, kind: str, asks: Sequence[Asked], found: list[Found],
                         pool: dict[tuple[int, Ref], Item]) -> dict[tuple[int, Ref], Item]:
        late = any(is_var(a.cond.other or "") for a in asks)
        rnd = "late" if late else "first"
        visual_q = self.marks.visual
        led = self.ctx.ledger
        for _ in range(self.m.pages):
            if self.left() <= 0 or (self.tokens is not None and self.tokens - self.evidence()
                                    < self.prior_call(False, "bridge" if late else "bound", rnd)):
                return pool
            dry = self.dry(pool, asks, kind)
            pages: list[tuple[int, ProbeRequest, ProbeRequest]] = []
            for i, (req, (_, _, conds, _)) in enumerate(self.round_lists):
                parts, counts = self.counted.get(id(req), ([], []))
                if not dry.intersection(conds):
                    continue
                for n in counts:
                    part = next((x for x in parts if x.semijoin and (x.collections is None or n.collection in
                                 x.collections) and (x.sites is None or n.site in x.sites)), None)
                    if part is not None and n.joinable > n.returned:
                        pages.append((i, req, self.page(part, n, counts)))
            if not pages:
                return pool
            self.admit([x for _, _, x in pages])
            got, counted = await self.ctx.fed.probe_counted(self.versioned([x for _, _, x in pages]), ledger=led)
            new = []
            for (i, req, _), cands, ns in zip(pages, got, counted, strict=True):
                ckind, value, conds, _ = self.round_lists[i][1]
                new += [Found(c, ckind, self.signal(c, ckind, value, visual_q), value or c.label or None, conds, i)
                        for c in cands]
                counts = self.counted[id(req)][1]
                for n in ns:
                    at = next((j for j, o in enumerate(counts) if (o.site, o.collection) == (n.site, n.collection)),
                              None)
                    if at is None:
                        counts.append(n)
                        continue
                    o = counts[at]
                    counts[at] = replace(o, joinable=max(o.joinable, n.joinable), returned=o.returned + n.returned,
                                         exact=o.exact and n.exact)
            found.extend(new)
            if self.m.reuse in CACHED:
                for f in new:
                    self.released.setdefault(f.c.ref, f.c)
            led.count_depth("join_pages")
            led.count_depth("join_paged_requests", len(pages))
            led.count_depth("join_paged_candidates", len(new))
            self.ctx.log("join_page", lists=len({i for i, _, _ in pages}), requests=len(pages), new=len(new),
                         conds=sorted(dry))
            if not new:
                return pool
            pool, outside, coverage = self.pool(kind, asks, found)
            self.coverage = coverage
            if self.negated:
                self.skips(pool, asks)
            await self.search(pool, outside, asks, kind)
        return pool

    def page(self, part: ProbeRequest, n: ProbeCounts, counts: Sequence[ProbeCounts]) -> ProbeRequest:
        encoder = self.sizes.get((n.site, n.collection), ("", 0))[0]
        group = sum(o.joinable for o in counts if o.collection == n.collection
                    and self.sizes.get((o.site, o.collection), ("", 0))[0] == encoder)
        share = n.joinable / group if group > 0 else 1.0
        k = part.k_group if part.k_group is not None else part.k
        return replace(part, k=max(1, math.ceil(k * share - 1e-12) + share_slack(k, share)), offset=n.returned,
                       collections=[n.collection], sites=[n.site], k_group=None, group_size=None, encoder=None)

    async def complete_units(self, pool: dict[tuple[int, Ref], Item], asks: Sequence[Asked]) -> None:
        for x in dict.fromkeys(a.var for a in asks):
            if self.cap(x) <= READ_CAP:
                continue
            these = [a for a in asks if a.var == x]
            ids = {a.cond.id for a in these}
            items = {ref: it.c for (cid, ref), it in pool.items() if cid in ids}
            witnesses = {ref for h in self.values[x].values() for ref in h.refs}
            tables = {t for ref in witnesses if ref in items and (t := _table(items[ref]))}
            rows = [c for ref, c in items.items() if c.complete and c.site not in self.titled and _table(c) in tables
                    and not self.judged(these, ref)]
            if not rows:
                continue
            for t in tables:
                seen = [self.p_read[ref] for ref in witnesses
                        if ref in self.p_read and ref in items and _table(items[ref]) == t]
                for c in rows:
                    if _table(c) == t and seen:
                        self.p_read[c.ref] = max(self.p_read.get(c.ref, 0.0), sum(seen) / len(seen))
            unknowns, by_var = self.unknowns(these, listing=True)
            for i in range(0, len(rows), self.list_cap):
                block = rows[i : i + self.list_cap]
                cost = self.kappa(unknowns, "table") + sum(approx_tokens(c.summary) + FRAME + self.per_object(unknowns)
                                                           for c in block)
                if cost + self.named(unknowns) > self.room(asks, x):
                    self.limit = "tokens"
                    break
                self.ctx.log("unit", var=x, n=len(block), cost=round(cost))
                if self.m.audited:
                    self.ctx.ledger.count_fired("table_units")
                await self.read_entries(block, unknowns, by_var)

    def reask(self, x: str) -> bool:
        witnesses = sum(len(v) for h in self.values[x].values() for v in h.witnessed.values())
        if self.reasked.get(x) == witnesses:
            return False
        if self.out_of_rounds():
            self.limit = "rounds"
            return False
        if self.m.planner == "query":
            if self.stops.get(x) == "allocation":
                return False
            if self.tokens is not None and self.tokens - self.evidence() < self.again(x):
                self.limit = "tokens"
                return False
        elif self.tokens is not None and self.tokens - self.evidence() < self.prior_call(False, "bound"):
            self.limit = "tokens"
            return False
        self.reasked[x] = witnesses
        ids = {c.id for c in self.by_var[x] if c.parent is None and self.status(c) == "done"}
        if not ids:
            return False
        self.asked = {t for t in self.asked if t[0] not in ids}
        self.ctx.log("reask", var=x, conditions=sorted(ids), filtered=len(self.filtered.get(x, ())))
        return True

    async def paths(self, kind: str, asks: Sequence[Asked]) -> list[Found]:
        self.round_kind = kind
        image = kind != "text"
        cols = self.collections(image)
        reqs: list[ProbeRequest] = []
        meta: list[tuple[str, str | None, tuple[int, ...], str | None]] = []
        pattern_keys = self.m.keys == "pattern"
        title_keys = self.m.keys == "titles"
        graph_keys = pattern_keys or title_keys
        titles_ok = self.m.titles == "all"
        visual_q = self.marks.visual
        tested = self.prefilter(asks)
        texts = {c for s in self.cols.values() for c, mod in s.items() if mod != "image"}

        def add(req: ProbeRequest, ckind: str, value: str | None, served: Iterable[Asked], node: str | None) -> None:
            if req.collections == []:
                return
            if self.m.mentions and (req.collections is None or texts.intersection(req.collections)):
                req.mentions = list(tested)
            reqs.append(req)
            meta.append((ckind, value, tuple(dict.fromkeys(a.cond.id for a in served)), node))

        if kind == "filter":
            by_x: dict[str, list[Asked]] = defaultdict(list)
            for a in asks:
                by_x[a.var].append(a)
            for x, items in by_x.items():
                fallback = any(a.cond.fallback for a in items)
                key = self.q.question if fallback else "; ".join(dict.fromkeys(
                    a.cond.text if a.cond.edge is None else key_text(a.value or "", a.relation, a.cond.edge.head
                                                                       == a.cond.other) for a in items))
                keyed = None if fallback else next((a.value for a in items if a.cond.edge is not None and a.value
                                                    and a.value != self.q.question), None)
                titles = self.candidates(x, [a.cond.id for a in items], self.filtered.get(x, ())) \
                    if graph_keys and titles_ok else []
                if titles:
                    add(ProbeRequest(key, "title", min(self.m.k, FILTER_K), collections=cols, titles=titles,
                                     value=keyed), "title", None, items, x)
                elif pattern_keys:
                    add(ProbeRequest(key, "bound", self.m.k, collections=cols, value=keyed), "bound", None, items, None)
                else:
                    add(ProbeRequest(self.q.question, "question", self.m.k, collections=cols), "question", None,
                        items, None)
        else:
            if pattern_keys:
                for a in asks:
                    if a.cond.edge is None or not a.value or a.value == self.q.question:
                        continue
                    late = is_var(a.cond.other or "")
                    add(ProbeRequest(key_text(a.value, a.relation, a.cond.edge.head == a.cond.other), "bound",
                                     self.m.k, collections=cols, value=a.value), "bridge" if late else "bound", a.value,
                        [a], a.cond.other)
            elif title_keys:
                pictures = self.collections(image=True)
                for a in asks:
                    if a.cond.edge is None or not a.value or a.value == self.q.question or not titles_ok:
                        continue
                    rel = key_text(a.value, a.relation, a.cond.edge.head == a.cond.other)
                    if kind == "text":
                        add(ProbeRequest(rel, "title", self.m.k, collections=cols, titles=[a.value], value=a.value),
                            "title", a.value, [a], a.cond.other)
                    own = kind == "visual" or not any(c.visual for c in self.by_var[a.var])
                    if own and pictures:
                        add(ProbeRequest(rel, "title", self.m.k, collections=pictures, titles=[a.value],
                                         value=a.value), "title", a.value, [a], a.cond.other)
            first = not self.question_probed
            if first or not graph_keys or not reqs or (image and any(a.cond.fallback for a in asks)):
                self.question_probed = True
                add(ProbeRequest(self.q.question, "question", self.m.k, collections=cols), "question", None, asks, None)
            if pattern_keys:
                pictures = self.collections(image=True)
                for a in asks:
                    if a.cond.edge is None or not a.value or a.value == self.q.question:
                        continue
                    late = is_var(a.cond.other or "")
                    rel = key_text(a.value, a.relation, a.cond.edge.head == a.cond.other)
                    if kind == "text" and (late or self.m.anchor_titles) and titles_ok:
                        add(ProbeRequest(rel, "title", self.m.k, collections=cols, titles=[a.value], value=a.value),
                            "title", a.value, [a], a.cond.other)
                    own = kind == "visual" or not any(c.visual for c in self.by_var[a.var])
                    if own and pictures and titles_ok:
                        add(ProbeRequest(rel, "title", self.m.k, collections=pictures, titles=[a.value],
                                         value=a.value), "title", a.value, [a], a.cond.other)
        if self.m.bloom and reqs:
            reqs, meta = await self.route_titles(reqs, meta)
        self.deferred = []
        if self.m.owners != "off" and kind == "text" and pattern_keys and reqs:
            await self.owner_first(reqs, meta)
        self.reach, self.round_lists = self.m.k, list(zip(reqs, meta, strict=True))
        self.counted = {}
        results = await self.probe(reqs) if reqs else []
        if self.m.bloom:
            self.ctx.ledger.count_reuse("bloom_false", sum(1 for r, got in zip(reqs, results, strict=True)
                                                           if r.kind == "title" and r.sites and not got))
        found = [Found(c, ckind, self.signal(c, ckind, value, visual_q), value or c.label or None, conds, i)
                 for i, ((ckind, value, conds, _), cands) in enumerate(zip(meta, results, strict=True)) for c in cands]
        self._nodes = {i: node for i, (_, _, _, node) in enumerate(meta)}
        if not self.question_top:
            asked = [c for (ckind, _, _, _), cands in zip(meta, results, strict=True) if ckind == "question"
                     for c in cands]
            self.question_top = rrf(split_lists(asked), salt=self.q.qid)
        if kind == "text" and graph_keys:
            found += await self.siblings(found, asks, len(reqs))
        if self.m.reuse in CACHED:
            for f in found:
                self.released.setdefault(f.c.ref, f.c)
        return found

    async def route_titles(self, reqs: list[ProbeRequest], meta: list[Any]) -> tuple[list[ProbeRequest], list[Any]]:
        blooms = await self.ctx.fed.blooms()
        keep_r, keep_m = [], []
        for r, m in zip(reqs, meta, strict=True):
            if r.kind != "title":
                keep_r.append(r)
                keep_m.append(m)
                continue
            keys = [title_key(t) for t in r.titles]
            sites = [s for s, cols in blooms.items()
                     if any(may_hold(bf, k) for c, bf in cols.items() if r.collections is None or c in r.collections
                            for k in keys)]
            if not sites:
                self.ctx.ledger.count_reuse("bloom_skipped")
                continue
            r.sites = sites
            keep_r.append(r)
            keep_m.append(m)
        return keep_r, keep_m

    async def probe(self, reqs: Sequence[ProbeRequest], kinds: dict[int, str] | None = None) -> list[list[Candidate]]:
        self.admit(reqs)
        self.versioned(reqs)
        routed = await self.routed_sites() if "proto" in self.m.route else None
        if routed is not None:
            for r in reqs:
                if r.kind == "question" and r.sites is None:
                    r.sites = list(routed)
        if self.m.probe == "semijoin" or self.m.limits == "share":
            known = {id(r): meta[0] for r, meta in self.round_lists}
            known.update(kinds or {})
            parts = [shaped(r, known.get(id(r), "sibling"), self.round_kind, self.sizes, self.groups,
                            semijoin=self.m.probe == "semijoin", share=self.m.limits == "share",
                            vocab="vocab" in self.m.route) for r in reqs]
            if self.m.probe == "semijoin" and self.root.zero:
                parts = [[y for x in part for y in self.unzeroed(x)] for part in parts]
            if any(len(part) != 1 or part[0] is not r for part, r in zip(parts, reqs, strict=True)):
                out = await self.probe_counted(reqs, parts)
                self.note_versions(out)
                return out
        if self.m.reuse not in CACHED or not self.m.probe_cache:
            out = await self.ctx.fed.probe(reqs, ledger=self.ctx.ledger)
            self.note_versions(out)
            return out
        keys = [canonical_json(r.to_dict()) for r in reqs]
        held = await self.ctx.kg.probes(self.ctx.ns, list(dict.fromkeys(keys)))
        miss = [i for i, k in enumerate(keys) if k not in held]
        fresh = await self.ctx.fed.probe([reqs[i] for i in miss], ledger=self.ctx.ledger) if miss else []
        got = dict(zip(miss, fresh, strict=True))
        for i, cands in got.items():
            self.probed.setdefault(keys[i], cands)
        self.ctx.ledger.count_reuse("probe_cache_hits", len(reqs) - len(miss))
        return [got[i] if i in got else [copy.copy(c) for c in held[k]] for i, k in enumerate(keys)]

    async def routed_sites(self) -> list[str] | None:
        root = self.root
        if getattr(root, "_routed", None) is not None:
            return root._routed or None
        from fedquest import routing

        fed = self.ctx.fed
        fed_name = getattr(fed, "name", "") or ""
        fit_name = routing.fit_name(fed_name, self.m.route_fit, self.q.qid)
        fit = routing.load_fit(fit_name)
        if fit is None and self.m.route_fit is not None:
            raise FileNotFoundError(f"no router fit {routing.routing_path(fit_name)} (route_fit "
                                    f"{self.m.route_fit!r}): python -m fedquest.routing calibrate {fed_name}")
        if fit is None or (fit.get("theta") is None and self.m.route_policy == "threshold"):
            self.ctx.log("route", fitted=False, fit=fit_name)
            root._routed = []
            return None
        router = await routing.Router.for_federation(fed)
        scores = await asyncio.to_thread(router.scores, self.q.question)
        sites = routing.Router.select(scores, self.m.route_policy, fit.get("theta"), fit.get("b"),
                                      int(fit.get("min_sites", routing.MIN_SITES)))
        root._routed = sites
        self.ctx.ledger.count_fired("routed_sites", len(sites))
        self.ctx.ledger.count_fired("routed_out", len(router.sites) - len(sites))
        self.ctx.log("route", policy=self.m.route_policy, sites=sites, of=len(router.sites),
                     theta=fit.get("theta"), b=fit.get("b"), fit=fit_name)
        return sites

    async def probe_counted(self, reqs: Sequence[ProbeRequest], parts: Sequence[Sequence[ProbeRequest]],
                            ) -> list[list[Candidate]]:
        sent = self.versioned([x for part in parts for x in part])
        found, counts = await self.ctx.fed.probe_counted(sent, ledger=self.ctx.ledger)
        out, i = [], 0
        for r, part in zip(reqs, parts, strict=True):
            out.append([c for cands in found[i:i + len(part)] for c in cands])
            self.counted[id(r)] = (list(part), [n for ns in counts[i:i + len(part)] for n in ns])
            for x, ns in zip(part, counts[i:i + len(part)], strict=True):
                if self.m.probe == "semijoin" and x.semijoin and x.mentions:
                    self.root.zero |= {(x.mentions[0], n.collection, n.site) for n in ns if not n.joinable and n.exact}
            i += len(part)
        return out

    def unzeroed(self, x: ProbeRequest) -> list[ProbeRequest]:
        if not x.semijoin or not x.mentions:
            return [x]
        value, zero = x.mentions[0], self.root.zero
        names = list(x.collections) if x.collections is not None else sorted({c for s in self.cols.values() for c in s})
        sites = [s for s in (x.sites if x.sites is not None else sorted(self.cols))
                 if any(c in self.cols.get(s, {}) for c in names)]
        gone = {s: [c for c in names if c in self.cols.get(s, {}) and (value, c, s) in zero] for s in sites}
        if not any(gone.values()):
            return [x]
        self.ctx.ledger.count_fired("zero_skips", sum(map(len, gone.values())))
        groups: dict[tuple[str, ...], list[str]] = {}
        for s in sites:
            kept = tuple(c for c in names if c not in gone[s])
            if any(c in self.cols.get(s, {}) for c in kept):
                groups.setdefault(kept, []).append(s)
        return [replace(x, collections=list(kept), sites=group) for kept, group in groups.items()]

    def versioned(self, reqs: Sequence[ProbeRequest]) -> Sequence[ProbeRequest]:
        if self.m.mention != 1:
            for r in reqs:
                if r.mentions:
                    r.mention = self.m.mention
        if self.m.scores == "anchor" or self.want_z:
            for r in reqs:
                r.scores = True
                r.zvariants = self.want_z
        if self.m.epoch is not None:
            for r in reqs:
                r.epoch = self.m.epoch
        if self.m.versioned_reads:
            for r in reqs:
                r.versions = True
        if self.m.views or self.m.room != LEAD_ROOM:
            for r in reqs:
                if r.key and r.kind in ("question", "bound", "title"):
                    r.views = self.m.views
                    r.room = self.m.room if self.m.room != LEAD_ROOM else None
                    r.view_key = self.q.question if self.m.views and self.m.view_key == "question" else None
                    r.signal = [r.value] if r.value else list(self.p.anchors)
        return reqs

    def note_versions(self, lists: Sequence[Sequence[Candidate]]) -> None:
        for cands in lists:
            for c in cands:
                if c.ver:
                    self.root.overs.setdefault(c.ref, c.ver)
                if c.view:
                    self.ctx.ledger.count_fired("view_entries")

    def signal(self, c: Candidate, kind: str, value: str | None, visual_q: bool) -> bool:
        if kind == "title" and c.modality == "image":
            return visual_q
        if c.lead_signal is not None:
            return c.lead_signal
        if value is None:
            return any(release_signal(c.summary, a) for a in self.p.anchors)
        return release_signal(c.summary, value)

    async def siblings(self, found: Sequence[Found], asks: Sequence[Asked], first_probe: int) -> list[Found]:
        vals = {a.value: a.cond.other for a in asks if a.value and a.value != self.q.question}
        heads: dict[str, tuple[Found, str]] = {}
        for f in found:
            head = _head(f.c)
            named = next((v for v in ([f.value] if f.value else vals) if v and prompts.mentions(f.c.summary, v)), None)
            if head and named and head not in heads:
                heads[head] = (f, named)
        heads = dict(list(heads.items())[:SIBLING_TABLES])
        rows = sorted({c for s in self.cols.values() for c, mod in s.items() if mod == "table"})
        if not heads or not rows:
            return []
        tested = self.prefilter(asks) if self.m.mentions else []
        own = self.m.probe == "semijoin"
        reqs = [ProbeRequest(h, "bound", SIBLING_K, collections=rows, mentions=list(tested), value=named,
                             sites=[f.c.site] if own else None) for h, (f, named) in heads.items()]
        results = await self.probe(reqs)
        out = []
        for i, ((h, (f, named)), cands) in enumerate(zip(heads.items(), results, strict=True)):
            self._nodes[first_probe + i] = vals.get(named)
            out += [Found(c, "sibling", release_signal(c.summary, named), named, f.conds, first_probe + i)
                    for c in cands if _body(c).startswith(f"{h} | ")]
        return out

    def pool(self, kind: str, asks: Sequence[Asked], found: Sequence[Found],
             ) -> tuple[dict[tuple[int, Ref], Item], dict[int, float], float]:
        cal = self.ctx.calib
        rnd = "filter" if kind == "filter" else "late" if any(is_var(a.cond.other or "") for a in asks) else "first"
        v4 = self.m.calibrated and getattr(cal, "version", 0) >= 4
        witnessed = v4 or self.m.probe == "semijoin"
        alpha: dict[tuple[int, Ref], dict[str, float]] = defaultdict(dict)
        weights: dict[tuple[int, Ref], dict[str, float]] = defaultdict(dict)
        first: dict[tuple[int, Ref], Candidate] = {}
        bits: dict[tuple[int, Ref], bool] = {}
        var_of = {a.cond.id: a.var for a in asks}
        provisional = {x: self.chances(list(self.values[x].values())) for x in dict.fromkeys(var_of.values())
                       if kind == "filter" and self.values[x]}
        zc = self.m.zc(self.q.qid) if self.m.calibrated else None
        for f in found:
            za = zc.alpha(f.kind, f.c.z, f.signal, f.c.modality, rnd, f.c.mentions,
                          self.sizes.get((f.c.site, f.c.collection), ("", 0))[0]) \
                if zc is not None and f.c.z is not None else None
            if za is not None:
                a = za
                self.ctx.ledger.count_fired("z_priced")
            elif self.m.scores == "anchor" and v4:
                self.ctx.ledger.count_fired("rank_priced")
                a = cal.alpha(f.c.site, f.c.collection, f.kind, self.g(f, rnd), f.signal, f.c.modality, rnd,
                              mention=f.c.mentions)
            elif v4 and self.m.scores == "ranknaive":
                a = cal.alpha(f.c.site, f.c.collection, f.kind, self.g(f, rnd).r, f.signal, f.c.modality, rnd,
                              mention=f.c.mentions)
            elif v4:
                a = cal.alpha(f.c.site, f.c.collection, f.kind, self.g(f, rnd), f.signal, f.c.modality, rnd,
                              mention=f.c.mentions)
            else:
                a = cal.alpha(f.c.site, f.c.collection, f.kind, f.c.rank, f.signal, f.c.modality, rnd) \
                    if self.m.calibrated else 1.0
            node = self._nodes.get(f.probe)
            hyp = norm_entity(f.value) if f.value else ""
            if kind == "filter" and f.kind == "title":
                w = provisional.get(node or "", {}).get(_key(f.value or ""), 0.0)
            else:
                w = self.weight(node, f.value) if f.kind != "question" else 1.0
            for cid in f.conds:
                key = (cid, f.c.ref)
                first.setdefault(key, f.c)
                if a > alpha[key].get(hyp, -1.0):
                    alpha[key][hyp], weights[key][hyp] = a, w
                if f.c.mentions is not None:
                    bits[key] = bits.get(key, False) or bool(f.c.mentions)
        if self.m.reuse == "index" and kind != "filter":
            for (cid, ref), (c, _, hyp, w, chance) in self.iextra.items():
                key = (cid, ref)
                if cid in var_of and key not in alpha and not self.judged([a for a in asks if a.cond.id == cid], ref):
                    first[key] = c
                    alpha[key][hyp], weights[key][hyp] = chance, w
            for key, by_hyp in alpha.items():
                cls = self.ientries.get(key[0], {}).get(key[1])
                if cls is not None:
                    self.iclass[key] = cls
                    self.iprior.setdefault(key, max(by_hyp.values(), default=0.0))
                    for h in by_hyp:
                        by_hyp[h] = self.yield_of(cls, by_hyp[h])
        score: dict[Ref, float] = defaultdict(float)
        seen_c: dict[Ref, Candidate] = {}
        for f in found:
            score[f.c.ref] += 1.0 / (60 + f.c.rank)
            seen_c.setdefault(f.c.ref, f.c)
        order = sorted(seen_c, key=lambda r: (-score[r], seen_c[r].rank, tie(self.q.qid, seen_c[r]), r[1]))
        position = {r: i for i, r in enumerate(order)}
        for _, ref in first:
            position.setdefault(ref, len(position))
        coverage = cal.coverage(rnd, kind != "text", self.reach) if self.m.calibrated else 1.0
        pool: dict[tuple[int, Ref], Item] = {}
        for key, by_hyp in alpha.items():
            c = first[key]
            question = by_hyp.get("", 0.0)
            mixed = sum(weights[key][h] * v for h, v in by_hyp.items() if h)
            beta = 1.0
            if not c.complete and c.modality != "image" and c.site not in self.titled:
                beta = cal.beta(c.modality, c.tokens, approx_tokens(c.summary), rep=self.rep(c)) \
                    if self.m.calibrated else \
                    min(1.0, approx_tokens(c.summary) / max(1, c.tokens))
            pool[key] = Item(c, max(question, mixed), float(position[c.ref]), beta=beta, mentions=bits.get(key))
        outside: dict[int, float] = {}
        for cid in {k[0] for k in pool}:
            items = [it for k, it in pool.items() if k[0] == cid]
            able = [it for it in items if it.mentions is not False] if witnessed else items
            z = max(1.0, sum(it.alpha for it in able) / coverage) if coverage > 0 else 1.0
            for it in items:
                it.p = it.alpha / z
            outside[cid] = max(0.0, 1.0 - sum(it.p for it in able))
        return pool, outside, coverage

    def g(self, f: Found, rnd: str) -> float:
        req = self.round_lists[f.probe][0] if f.probe < len(self.round_lists) else None
        parts, counts = self.counted.get(id(req), ([], [])) if req is not None else ([], [])
        return group_rank(f.kind, rnd, f.c, self.sizes, self.groups,
                          counts if any(p.semijoin for p in parts) else None)

    def kappa(self, unknowns: Sequence[ReadUnknown], modality: str | None = None) -> float:
        measured = self.m.costs == "measured"
        key = (modality if measured else None, *((u.name, tuple(u.conditions), u.cap, u.mode, tuple(u.candidates),
                                                  *((tuple(u.later),) if measured else ())) for u in unknowns))
        if key not in self.kappas:
            prompt = self.prompt_tokens(unknowns)
            setup = self.ctx.calib.kappa(key[0]) if key[0] is not None and self.ctx is not None else None
            estimate = prompt + 4 * self.out_empty
            self.kappas[key] = estimate if setup is None else \
                max(estimate, setup - 4 * self.out_hit + prompt - self.prompt_tokens([REFERENCE]))
        return self.kappas[key]

    def prompt_tokens(self, unknowns: Sequence[ReadUnknown]) -> int:
        return approx_tokens(" ".join(m["content"] if isinstance(m["content"], str) else " ".join(
            p.get("text", "") for p in m["content"]) for m in prompts.read_messages(self.q.question, [], unknowns)))

    def named(self, unknowns: Sequence[ReadUnknown]) -> float:
        return 4 * self.out_hit * sum(min(u.cap, READ_CAP) for u in unknowns)

    def per_object(self, unknowns: Sequence[ReadUnknown]) -> float:
        return 4 * self.out_hit if any(u.cap > READ_CAP for u in unknowns) else 0.0

    def call_cap(self, place: str, unknowns: Sequence[ReadUnknown]) -> int:
        cal = self.ctx.calib
        lists = any(u.cap > READ_CAP for u in unknowns)
        if place == COORD:
            cap = int(cal.cap("coordinator", COORD_CAP))
            if lists and self.m.scan:
                cap = max(cap, self.list_cap)
        else:
            mod = place.rsplit("|", 1)[1]
            cap = min(int(cal.cap(mod, prompts.PER_CALL.get(mod, READ_CAP))), self.left())
        if lists:
            cap = min(cap, self.list_cap)
        return cap

    def target(self, asks: Sequence[Asked], kind: str) -> list[tuple[str, int]]:
        out = []
        for x in dict.fromkeys(a.var for a in asks):
            if self.finished(x, kind):
                continue
            cids = list(dict.fromkeys(a.cond.id for a in asks if a.var == x))
            open_ = [cid for cid in cids if not any(cid in h.witnessed for h in self.values[x].values())]
            out += [(x, cid) for cid in [*open_, *[c for c in cids if c not in open_]]]
        return out

    def path(self, kind: str, late: bool) -> tuple[bool, str, str]:
        if kind == "filter":
            return True, "title" if self.m.titles == "all" else "bound", "filter"
        return kind != "text", "bridge" if late else "bound", "late" if late else "first"

    def kind_of(self, c: Cond) -> str:
        return "filter" if c.visual and self.owns(c) else "visual" if c.visual else "text"

    def room(self, asks: Sequence[Asked], x: str | None = None, kind: str = "text") -> float:
        if self.tokens is None:
            return math.inf
        order = list(dict.fromkeys(a.var for a in asks))
        later = order[order.index(x) + 1:] if x in order else []
        own = 0.0
        for y in later:
            if self.finished(y, kind):
                continue
            late = any(is_var(a.cond.other or "") for a in asks if a.var == y)
            hop = (y, *self.path(kind, late))
            if hop not in self.reserves:
                self.reserves[hop] = self.prior_call(*hop[1:])
            own += self.reserves[hop]
        return self.tokens - self.evidence() - self.reserve(asks) - own

    def ahead(self, asks: Sequence[Asked]) -> list[tuple[str, bool, str, str]]:
        now = {a.cond.id for a in asks}
        out = []
        for x in self.p.variables:
            if not self.alive(x):
                continue
            for c in self.by_var[x]:
                if c.id in now or c.parent is not None or self.status(c) not in ("pending", "later"):
                    continue
                hop = (x, *self.path(self.kind_of(c), c.edge is not None and is_var(c.other or "")))
                if hop not in out:
                    out.append(hop)
        return out

    def reserve(self, asks: Sequence[Asked]) -> float:
        total = 0.0
        for hop in self.ahead(asks):
            if hop not in self.reserves:
                self.reserves[hop] = self.prior_call(*hop[1:])
            total += self.reserves[hop]
        return total

    def prior_places(self, image: bool, kind: str, rnd: str | None = None) -> tuple[list[Place], float]:
        cal = self.ctx.calib
        rnd = rnd or ("filter" if kind == "title" else "late" if kind == "bridge" else "first")
        places: dict[str, list[Cand]] = defaultdict(list)
        for site, cols in self.cols.items():
            for col, mod in cols.items():
                if (mod == "image") != image:
                    continue
                size = cal.tokens(site, col, prompts.IMAGE_TOKENS if mod == "image" else SIZE.get(mod, SIZE["text"]), mod)
                for rank in range(1, self.m.k + 1):
                    a = cal.alpha(site, col, kind, rank, None, mod, rnd) if self.m.calibrated else 1.0
                    places[f"{site}|{mod}"].append(Cand((site, col, rank), a, size + FRAME))
        mass = sum(x.p for cs in places.values() for x in cs)
        coverage = cal.coverage(rnd, image, self.m.k) if self.m.calibrated else 1.0
        z = max(1.0, mass / coverage) if coverage > 0 else 1.0
        pl = [Place(n, self.kappa([REFERENCE], n.rsplit("|", 1)[1]), prompts.PER_CALL.get(n.rsplit("|", 1)[1], READ_CAP),
                    [Cand(x.key, x.p / z, x.c) for x in cs]) for n, cs in places.items()]
        return pl, max(0.0, 1.0 - mass / z)

    def prior_call(self, image: bool, kind: str, rnd: str | None = None) -> float:
        pl, outside = self.prior_places(image, kind, rnd)
        options = calls(pl, outside, size="unit", salt=self.salt)
        return options[0].cost + 4 * self.out_hit if options else 0.0

    def prior_curve(self, image: bool, kind: str, rnd: str) -> list[tuple[float, float]]:
        key = (image, kind, rnd)
        if key not in self.curves:
            pl, outside = self.prior_places(image, kind, rnd)
            self.curves[key] = curve(pl, outside, 4 * self.out_hit, self.m.size, salt=self.salt)
        return self.curves[key]

    def holding(self, x: str, cid: int) -> float:
        tried = self.filtered.get(x, set())
        viable = {k for k in (_key(h.name) for h in self.full(x)) if k not in tried}
        if not viable:
            return 0.0
        c, other = self.conds[cid], self.conds[cid].other or ""
        live = {norm_entity(v) for v in self.bound.get(other, [])} if c.edge is not None and is_var(other) else None
        refs = [ref for ref, pairs in self.witnesses.get((x, cid), {}).items()
                if any(k in viable and (live is None or hyp in live) for k, hyp in pairs)]
        return min(self.coverage, sum(1.0 if ref in self.reused else self.p_read.get(ref, 0.0) for ref in refs))

    def later_hops(self, asks: Sequence[Asked], x: str, kind: str) -> list[tuple[str, bool, str, str]]:
        order = list(dict.fromkeys(a.var for a in asks))
        out = [(kind, *self.path(kind, any(is_var(a.cond.other or "") for a in asks if a.var == y)))
               for y in (order[order.index(x) + 1:] if x in order else []) if not self.finished(y, kind)]
        out += [("filter" if rnd == "filter" else "visual" if image else "text", image, access, rnd)
                for _, image, access, rnd in self.ahead(asks)]
        return out

    def allot(self, asks: Sequence[Asked], x: str, cid: int, pl: Sequence[Place], outside: float,
              unknowns: Sequence[ReadUnknown], kind: str) -> tuple[float, list[list[Any]]]:
        later = self.later_hops(asks, x, kind)
        if not later:
            return math.inf, []
        curves = [curve(pl, outside, 4 * self.out_hit, self.m.size, salt=self.salt),
                  *(self.prior_curve(image, access, rnd) for _, image, access, rnd in later)]
        left = (self.tokens or 0) - self.evidence() - self.named(unknowns)
        shares = allocate(curves, left, [self.holding(x, cid), *(0.0 for _ in later)])
        return shares[0], [[k, round(s)] for (k, *_), s in zip(later, shares[1:], strict=True)]

    def opening(self, image: bool, access: str, rnd: str) -> float:
        steps = self.prior_curve(image, access, rnd)
        return steps[0][0] if steps else 0.0

    def again(self, x: str) -> float:
        mine = [self.opening(*self.path(self.kind_of(c), c.edge is not None and is_var(c.other or "")))
                for c in self.by_var[x] if c.parent is None and self.status(c) == "done"]
        return min(mine, default=0.0) + sum(self.opening(*hop[1:]) for hop in self.ahead([]) if hop[0] != x)

    def width(self, x: str, c: Cond, n: int) -> int:
        need = max(1, self.m.beam or n)
        later = sum(1 for y in self.p.variables if y != x and self.alive(y))
        need = max(need, math.ceil(n / max(1, self.max_rounds + BEAM_ROUNDS - self.rounds - later)))
        if self.tokens is not None and self.ctx is not None:
            step = self.opening(*self.path(self.kind_of(c), True))
            left = self.tokens - self.evidence() - sum(self.opening(*hop[1:]) for hop in self.ahead([])
                                                         if hop[0] != x)
            if step > 0:
                need = max(need, math.ceil(n / max(1, int(left // step))))
        return need

    def places(self, pool: dict[tuple[int, Ref], Item], asks: Sequence[Asked], cid: int, values: Sequence[str],
               extra: float) -> dict[str, list[Cand]]:
        these = [a for a in asks if a.cond.id == cid]
        places: dict[str, list[Cand]] = defaultdict(list)
        unmentioned: list[Ref] = []
        for (c_id, ref), it in pool.items():
            if c_id != cid or self.judged(these, ref):
                continue
            c = it.c
            glanced = all((a.cond.id, norm_entity(a.value or ""), ref) in self.entries for a in these)
            w = it.p * (1.0 - it.beta) if glanced else it.p
            if w <= 1e-12 and self.m.order == "calib":
                continue
            names = not self.m.semijoin or c.modality == "image" or not values or any(
                prompts.mentions(c.summary, v) for v in values)
            if c.complete and not names:
                continue
            entry = self.m.place != "sites" and c.site not in self.titled and (
                c.modality != "image" or self.m.place == "coordinator")
            size = approx_tokens(c.summary) + FRAME + extra
            if entry and (c.complete or c.modality == "image"):
                places[COORD].append(Cand(("e", ref), w, size, it.r))
            elif entry and names and not glanced:
                places[COORD].append(Cand(("e", ref), w * it.beta, size, it.r))
            elif self.m.place != "coordinator":
                if self.m.mentions and values and c.modality != "image" and it.mentions is False:
                    unmentioned.append(ref)
                    continue
                places[f"{c.site}|{c.modality}"].append(Cand(("o", ref), w, c.tokens + FRAME + extra, it.r))
        if unmentioned:
            self.mark(these, unmentioned)
            self.unmentioned += len(unmentioned)
        return places

    def modality(self, place: str, cands: Sequence[Cand], pool: dict[tuple[int, Ref], Item], cid: int) -> str:
        if place != COORD:
            return place.rsplit("|", 1)[1]
        return "table" if all(pool[(cid, x.key[1])].c.modality == "table" for x in cands) else "text"

    def choose(self, pl: Sequence[Place], outside: float, room: float, bound: float) -> list[Call]:
        kappa = {p.name: p.kappa for p in pl}
        left = self.left()
        chosen: list[Call] = []
        if self.m.order == "calib" and self.m.size == "plan" and self.m.rollout:
            best = plan_call(pl, outside, room, salt=self.salt)
            if best is not None:
                chosen.append(best)
                room -= best.cost + bound
                left -= len(best.block) if best.place != COORD else 0
        for option in calls(pl, outside, self.m.size, self.m.order, salt=self.salt):
            if len(chosen) >= self.m.width:
                break
            if chosen and (option.place in {c.place for c in chosen} or COORD in (option.place, chosen[0].place)):
                continue
            block = list(option.block)[: max(0, left) if option.place != COORD else None]
            while block and kappa[option.place] + sum(x.c for x in block) > room:
                block = block[:-1]
            if block:
                mass = sum(x.p for x in option.block)
                p = option.p * sum(x.p for x in block) / mass if mass > 0 else 0.0
                call = Call(option.place, tuple(block), p, kappa[option.place] + sum(x.c for x in block))
                chosen.append(call)
                room -= call.cost + bound
                left -= len(block) if option.place != COORD else 0
        return chosen

    def plan(self, pool: dict[tuple[int, Ref], Item], outside: dict[int, float], asks: Sequence[Asked],
             unknowns: Sequence[ReadUnknown], kind: str) -> list[Call]:
        t0 = time.thread_time()
        ended: dict[str, str | None] = {}
        try:
            values = self.prefilter(asks)
            extra = self.per_object(unknowns)
            if self.m.costs == "v12":
                self.kappa(unknowns)
            split = self.m.planner == "query" and self.tokens is not None
            skipped, logged = self.unmentioned, False
            for x, cid in self.target(asks, kind):
                ended.setdefault(x, None)
                places = self.places(pool, asks, cid, values, extra)
                pl = [Place(n, self.kappa(unknowns, self.modality(n, cs, pool, cid)), cap, cs)
                      for n, cs in places.items() if (cap := self.call_cap(n, unknowns)) > 0]
                if not pl:
                    if places and self.left() == 0:
                        self.limit = "objects"
                        ended[x] = ended[x] or "objects"
                    continue
                bound = self.named(unknowns)
                room = self.room(asks, x, kind) - bound
                miss = outside.get(cid, 0.0)
                allowance, later = self.allot(asks, x, cid, pl, miss, unknowns, kind) if split else (math.inf, [])
                chosen = self.choose(pl, miss, min(room, allowance), bound)
                if not chosen:
                    if allowance < room and self.choose(pl, miss, room, bound):
                        self.limit = ended[x] = "allocation"
                        if not logged:
                            logged = True
                            self.ctx.log("alloc", var=x, cond=cid, allowance=round(allowance),
                                         spent=round(self.evidence() - self.hop_start), ahead=later)
                    else:
                        self.limit = "tokens"
                        ended[x] = "allocation" if ended[x] == "allocation" else "tokens"
                    continue
                ended[x] = None
                for i, c in enumerate(chosen):
                    dropped = {"unmentioned": self.unmentioned - skipped} if i == 0 and self.unmentioned > skipped \
                        else {}
                    self.ctx.log("plan", place=c.place, n=len(c.block), p=round(c.p, 3), cost=round(c.cost),
                                 rho=float(f"{c.rho:.3g}"), **dropped)
                self.plan_cid = cid
                return chosen
            return []
        finally:
            for y, why in ended.items():
                if why is None:
                    self.stops.pop(y, None)
                else:
                    self.stops[y] = why
            self.ctx.ledger.plan_seconds += time.thread_time() - t0

    async def execute(self, chosen: Sequence[Call], pool: dict[tuple[int, Ref], Item], asks: Sequence[Asked],
                      unknowns: list[ReadUnknown], by_var: dict[str, list[Asked]]) -> bool:
        before = (len(self.seen), self.spent, len(self.entries))
        item = {ref: it for (_, ref), it in pool.items()}
        for call in chosen:
            for x in call.block:
                ref = x.key[1]
                self.p_read[ref] = max(self.p_read.get(ref, 0.0), x.p)
        if chosen[0].place == COORD and self.m.place == "extract_all":
            await self.extract_entries([item[x.key[1]].c for x in chosen[0].block], unknowns, by_var, chosen[0])
        elif chosen[0].place == COORD:
            await self.read_entries([item[x.key[1]].c for x in chosen[0].block], unknowns, by_var, chosen[0])
        else:
            requests: dict[str, dict[str, Any]] = {}
            values = self.prefilter(asks) if self.m.semijoin else []
            for call in chosen:
                site = call.place.rsplit("|", 1)[0]
                req = requests.setdefault(site, {"blocks": [], "prefilter": values, "exempt": [], "labels": {}})
                oids = [x.key[1][1] for x in call.block]
                req["blocks"].append(oids)
                req["labels"].update({x.key[1][1]: item[x.key[1]].c.label for x in call.block
                                      if item[x.key[1]].c.label})
                self.capable(site, (item[x.key[1]].c for x in call.block))
            await self.push(requests, unknowns, by_var, chosen)
        if self.iclass:
            for call in chosen:
                for x in call.block:
                    key = (self.plan_cid, x.key[1])
                    if key in self.iclass and key not in self.icounted:
                        self.icounted.add(key)
                        got = self.ideltas[self.iclass[key]]
                        got[2] += self.iprior.get(key, 0.0)
                        got[0] += 1
                        got[1] += int(key[1] in self.witnesses.get((self.conds[key[0]].var, key[0]), {}))
        return (len(self.seen), self.spent, len(self.entries)) != before

    async def read_entries(self, entries: Sequence[Candidate], unknowns: list[ReadUnknown],
                           by_var: dict[str, list[Asked]], call: Call | None = None) -> None:
        if not entries:
            return
        asks = [a for items in by_var.values() for a in items]
        for c in entries:
            self.entries_read.setdefault(c.ref, c)
        self.ctx.ledger.note_read([c.doc_id for c in entries if c.doc_id], "coordinator")
        self.mark(asks, (c.ref for c in entries if c.complete or c.modality == "image"))
        self.entries |= {(a.cond.id, norm_entity(a.value or ""), c.ref) for c in entries
                         if not c.complete and c.modality != "image" for a in asks}
        kind = {c.ref: "index entry" if not c.complete else "table row" if c.modality == "table" else "text"
                for c in entries}
        self.root.witnessable.update(c.ref for c in entries)
        self.set_reads(unknowns)
        cached_hits: dict[str, list[dict[str, Any]]] = defaultdict(list)
        all_entries = list(entries)
        if self.m.cache == "tuple":
            keys = {self.tuple_key("entry", c.site, c.oid, v, unknowns): c for c in entries
                    if (v := self.root.overs.get(c.ref) or c.ver)}
            held = await self.ctx.kg.tuples(self.ctx.ns, list(keys)) if keys else {}
            self.ctx.ledger.count_reuse("tuple_lookups", len(keys))
            self.ctx.ledger.count_reuse("tuple_hits", len(held))
            for key, payload in held.items():
                c = keys[key]
                for u, hits in zip(unknowns, payload.get("hits") or [], strict=False):
                    cached_hits[u.name] += [{"entity": h["entity"], "site": c.site, "oid": c.oid,
                                             "conditions": list(h["conditions"]), "_name": u.name} for h in hits]
            done_refs = {keys[k].ref for k in held}
            entries = [c for c in entries if c.ref not in done_refs]
        harvest = self.harvest_size(unknowns)
        objs = [ReadObject(kind[c.ref], c.summary) for c in entries]
        obj: Any = {}
        if entries:
            reply = await self.ctx.llm.chat(prompts.read_messages(self.read_question(), objs, unknowns, harvest=harvest),
                                            role="answer", task="read_index",
                                            max_tokens=prompts.read_max_tokens(unknowns)
                                            + prompts.HARVEST_TOKENS * harvest,
                                            json_mode=True, ledger=self.ctx.ledger, charge="bind_index")
            obj = prompts.read_reply(reply.text)
        if harvest:
            from fedquest.site.service import HARVEST_VERSION

            got = []
            for f in prompts.parse_harvest(obj, len(entries))[:harvest]:
                c = entries[f["object"] - 1]
                if c.modality != "image" and prompts.mentions(c.summary, f["subject"]) \
                        and prompts.witnessed(c.summary, f["value"]):
                    got.append(Fact(f["subject"], f["relation"], f["value"], c.site, c.oid, HARVEST_VERSION,
                                    c.modality, self.root.overs.get(c.ref, c.ver or "")))
            self.root.harvested += got
            self.ctx.ledger.count_reuse("kg_harvested", len(got))
        hits: dict[str, list[dict[str, Any]]] = defaultdict(list)
        parsed = prompts.parse_read(obj, unknowns, len(entries))
        for u in unknowns:
            for h in parsed.get(u.name, []):
                c, entity = entries[h.object - 1], h.entity
                if u.mode == "filter":
                    named = [n for n in u.candidates if prompts.witnessed(c.summary, n)]
                    said = prompts.judgment(h.entity)
                    entity = (named[0] if said and named else "") if said is not None else \
                        prompts.canonical(h.entity, named) or ""
                    ok = bool(entity)
                else:
                    ok = prompts.stated(_stated(c), u, entity)
                if ok and (conds := prompts.claims(c.summary, u, h)):
                    hit = {"entity": entity, "site": c.site, "oid": c.oid, "conditions": conds, "_name": u.name}
                    if c.modality != "image":
                        title = c.summary.split(": ", 1)[0] if ": " in c.summary else ""
                        known = next((u.values[i - 1] for i in conds if 0 < i <= len(u.values) and u.values[i - 1]),
                                     None)
                        if span := prompts.witness_span(_body(c), entity, title, known):
                            hit["w"] = span
                    hits[u.name].append(hit)
        if self.m.cache == "tuple":
            items = {}
            for c in entries:
                v = self.root.overs.get(c.ref) or c.ver
                said = [[{"entity": h["entity"], "conditions": h["conditions"]} for h in hits.get(u.name, [])
                         if (h["site"], h["oid"]) == c.ref] for u in unknowns]
                if v and any(said):
                    items[self.tuple_key("entry", c.site, c.oid, v, unknowns)] = {"hits": said}
            if items and self.m.kg_write:
                await self.ctx.kg.add_tuples(self.ctx.ns, items)
            for name, more in cached_hits.items():
                hits[name] += more
            entries = all_entries
        by_ref = {c.ref: c for c in entries}
        accepted = self.merge(hits, by_var, prompts.INDEX_BIND_VERSION, {c.ref: c.modality for c in entries})
        known = {name: [a.value for a in by_var.get(f"?{name}", by_var.get(name, [])) if a.value
                        and a.value != self.q.question] for name in hits}
        single = len(self.p.variables) == 1
        self.settle(hits, by_var, lambda h: not single or not known.get(h.get("_name", ""), []) or any(
            _heads(by_ref[(h["site"], h["oid"])], v) for v in known[h["_name"]]))
        for ref in accepted:
            if ref not in self.indexed:
                self.indexed.append(ref)
                self.snippets.setdefault(ref, by_ref[ref].summary)
                if by_ref[ref].doc_id:
                    self.doc_ids[ref] = by_ref[ref].doc_id
        self.ctx.log("bind_index", entries=len(entries), p=None if call is None else round(call.p, 3),
                     cost=None if call is None else round(call.cost),
                     values={n: [h["entity"] for h in v] for n, v in hits.items()},
                     read=[c.oid for c in entries], cited=_cited(hits))

    async def extract_entries(self, entries: Sequence[Candidate], unknowns: list[ReadUnknown],
                              by_var: dict[str, list[Asked]], call: Call | None = None) -> None:
        if not entries:
            return
        asks = [a for items in by_var.values() for a in items]
        for c in entries:
            self.entries_read.setdefault(c.ref, c)
        self.mark(asks, (c.ref for c in entries if c.complete or c.modality == "image"))
        self.entries |= {(a.cond.id, norm_entity(a.value or ""), c.ref) for c in entries
                         if not c.complete and c.modality != "image" for a in asks}

        async def one(c: Candidate) -> tuple[str, ExtractionResult, str]:
            reply = await self.ctx.llm.chat(prompts.extract_query_messages(self.q.question, c.summary),
                                            role="extract_text", task="extract_query", max_tokens=1024,
                                            ledger=self.ctx.ledger, charge="extract_entry")
            facts = [Fact(h, r, t, c.site, c.oid, prompts.QUERY_EXTRACT_VERSION, c.modality)
                     for h, r, t in prompts.parse_query_extraction(reply.text)]
            return c.site, ExtractionResult(c.site, c.oid, c.modality, prompts.QUERY_EXTRACT_VERSION,  # type: ignore[arg-type]
                                            facts, Usage(), doc_id=c.doc_id), c.summary.partition(": ")[0]

        done = await asyncio.gather(*(one(c) for c in entries))
        self.ctx.log("extract_entries", entries=len(entries), p=None if call is None else round(call.p, 3),
                     cost=None if call is None else round(call.cost), read=[c.oid for c in entries])
        await self.facts_read([d for d in done if d[1].facts], unknowns, by_var)

    async def recall(self, by_var: dict[str, list[Asked]]) -> None:
        if self.m.kg == "off" and self.m.cache != "off":
            return
        asks = [a for items in by_var.values() for a in items]
        values = list(dict.fromkeys(a.value for a in asks
                                    if a.value and a.cond.edge is not None and a.value != self.q.question))
        fkg = self.m.kg != "off"
        trusted = fkg and self.m.kg_check == "trust"
        evidence = fkg and self.m.kg_check == "witness"
        facts = await self.ctx.kg.lookup(self.ctx.ns, values, include_stale=trusted or evidence) if values else []
        if self.m.owners == "titles+kg":
            for f in facts:
                for e in (f.head, f.tail):
                    if f.site:
                        self.root.homes[norm_entity(e)].add(f.site)
        if fkg and facts:
            named = {norm_entity(e) for f in facts for e in (f.head, f.tail)}
            self.root.returning |= {norm_entity(v) for v in values if norm_entity(v) in named}
        if not facts:
            return
        sim = getattr(self.ctx.matcher, "sim", None) if fkg else None
        hits: dict[str, list[dict[str, Any]]] = defaultdict(list)
        used: list[Fact] = []
        known: dict[Fact, str] = {}
        of_fact: list[tuple[str, int, Fact]] = []
        for x, items in by_var.items():
            for i, a in enumerate(items, 1):
                if a.cond.edge is None or not a.value:
                    continue
                known_head = a.cond.edge.head == a.cond.other
                mine = [f for f in facts if norm_entity((f.head, f.tail)[0 if known_head else 1]) ==
                        norm_entity(a.value)]
                same = [f for f in mine if _relation(f.relation) == _relation(a.relation)]
                if fkg and sim is not None and not same and mine:
                    rels = list(dict.fromkeys(_relation(f.relation) for f in mine))
                    close = {r for r, v in zip(rels, sim.sims(_relation(a.relation), rels), strict=True)
                             if float(v) >= self.m.rel_sim}
                    same = [f for f in mine if _relation(f.relation) in close]
                for f in same:
                    other = f.tail if known_head else f.head
                    hits[x.lstrip("?")].append({"entity": other, "site": f.site, "oid": f.oid, "conditions": [i]})
                    used.append(f)
                    known.setdefault(f, a.value)
                    of_fact.append((x.lstrip("?"), len(hits[x.lstrip("?")]) - 1, f))
        if fkg and used and self.m.kg_check in ("version", "witness"):
            refs = list(dict.fromkeys((f.site, f.oid) for f in used))
            now = await self.ctx.fed.versions(refs, epoch=self.m.epoch, ledger=self.ctx.ledger)
            fresh = {(f.site, f.oid, f.overs) for f in used if f.overs and now.get((f.site, f.oid)) == f.overs}
            witness_kept: set[Fact] = set()
            if evidence:
                changed = [f for f in dict.fromkeys(used) if (f.site, f.oid, f.overs) not in fresh and f.wspan
                           and (f.site, f.oid) in now]
                if changed:
                    items = [(f.site, f.oid, f.wspan, known.get(f, "")) for f in changed]
                    held = await self.ctx.fed.witnesses(items, epoch=self.m.epoch, ledger=self.ctx.ledger)
                    witness_kept = {f for f, it in zip(changed, items, strict=True) if held.get(it)}
                    self.ctx.ledger.count_reuse("kg_witness_checked", len(items))
                    self.ctx.ledger.count_reuse("kg_witness_kept", len(witness_kept))
                    for f in witness_kept:
                        self.root.overs[(f.site, f.oid)] = now[(f.site, f.oid)]
            stale = {(f.site, f.oid): now[(f.site, f.oid)] for f in used
                     if (f.site, f.oid) in now and f.overs and now[(f.site, f.oid)] != f.overs}
            if stale and self.m.kg_write:
                await self.ctx.kg.stale(self.ctx.ns, stale)
            self.ctx.ledger.count_reuse("kg_checked", len(refs))
            self.ctx.ledger.count_reuse("kg_stale", len(stale))
            keep = [f for f in used if (f.site, f.oid, f.overs) in fresh or f in witness_kept]
            if evidence:
                kept_facts = set(keep)
                drop = {(n, j) for n, j, f in of_fact if f not in kept_facts}
                hits = {n: [h for j, h in enumerate(v) if (n, j) not in drop] for n, v in hits.items()}
            else:
                ok = {(f.site, f.oid) for f in keep}
                hits = {n: [h for h in v if (h["site"], h["oid"]) in ok] for n, v in hits.items()}
            hits = {n: v for n, v in hits.items() if v}
            used = keep
        if fkg and used:
            self.ctx.ledger.count_reuse("kg_hits", len(used))
            for f in used:
                if f.overs:
                    self.root.overs.setdefault((f.site, f.oid), f.overs)
        if not hits:
            return
        self.root.witnessable.update((f.site, f.oid) for f in used)
        kept = self.merge(hits, by_var, prompts.READ_VERSION, {(f.site, f.oid): f.modality for f in used})
        self.settle(hits, by_var)
        self.reused.extend(r for r in dict.fromkeys(kept) if r not in self.reused)
        self.reused_modality.update({(f.site, f.oid): f.modality for f in used})
        self.ctx.ledger.charge_reuse(used, len(set(kept)))
        self.ctx.log("reuse", values={n: [h["entity"] for h in v] for n, v in hits.items()})

    async def memo(self, kind: str, asks: Sequence[Asked]) -> None:
        made = [a for a in asks if a.cond.edge is not None and a.value and a.value != self.q.question]
        for a in made:
            self.made.setdefault((a.cond.id, norm_entity(a.value or "")), a)
        index = self.m.reuse == "index"
        resolved: dict[str, dict[str, float]] = {}
        if index and made:
            if self.keys is None:
                self.keys = KeyIndex(await self.ctx.kg.nodes(self.ctx.ns))
                self.istats = await self.ctx.kg.index_stats(self.ctx.ns)
            resolved = {a.value or "": dict(self.keys.resolve(a.value or "")) for a in made}
            lookup = list(dict.fromkeys(k for keys in resolved.values() for k in keys))
        else:
            lookup = list(dict.fromkeys(a.value or "" for a in made))
        records = await self.ctx.kg.asks(self.ctx.ns, lookup) if made and lookup else []
        if not records:
            return
        sim = getattr(self.ctx.matcher, "sim", None)
        offered: list[tuple[Asked, dict[Ref, float]]] = []
        tiers: list[str] = []
        negated = 0
        extra: dict[tuple[int, Ref], tuple[str, str, float, float]] = {}
        for a in made:
            v, r = norm_entity(a.value or ""), _relation(a.relation)
            head = a.cond.edge is not None and a.cond.edge.head == a.cond.other
            rho = resolved.get(a.value or "", {}) if index else {v: 1.0}
            mine = [rec for rec in records if rec.value in rho]
            same = [rec for rec in mine if rec.head == head]
            exact = [rec for rec in same if rec.relation == r]
            other = [rec for rec in same if rec.relation != r and rec.witnesses]
            sims = [float(s) for s in sim.sims(r, [rec.relation for rec in other])] if other and sim is not None else []
            para = [other[i] for i in sorted(range(len(sims)), key=lambda i: -sims[i]) if sims[i] >= self.m.rel_sim]
            no: set[Ref] = set()
            if not index:
                no = {ref for rec in exact for ref in rec.negatives} if self.m.negatives else set()
                fresh = [ref for ref in no if (a.cond.id, v, ref) not in self.seen]
                self.mark([a], fresh)
                self.negated |= {(a.cond.id, v, ref) for ref in fresh}
                negated += len(fresh)
            cands: dict[Ref, float] = {}
            for rec in [*exact, *para]:
                for ref in rec.witnesses:
                    if ref not in no and not self.judged([a], ref):
                        base = rec.chance.get(ref, 0.0) * rho.get(rec.value, 1.0)
                        if index:
                            key = (a.cond.id, ref)
                            self.iprior[key] = max(self.iprior.get(key, 0.0), base)
                        cands[ref] = max(cands.get(ref, 0.0), self.yield_of("same", base) if index else base)
            if index:
                entry = self.ientries.setdefault(a.cond.id, {})
                for ref in cands:
                    entry[ref] = self.iclass[(a.cond.id, ref)] = "same"
                hyp, w = norm_entity(a.value or ""), self.weight(a.cond.other, a.value)
                for rec in mine:
                    for ref in rec.witnesses:
                        if ref not in entry and not self.judged([a], ref):
                            entry[ref] = "entity"
                            extra[(a.cond.id, ref)] = ("entity", hyp, w,
                                                      rec.chance.get(ref, 0.0) * rho.get(rec.value, 1.0))
                silent_here = {ref for rec in exact for ref in rec.negatives}
                for rec in mine:
                    for ref in rec.negatives:
                        entry.setdefault(ref, "silent_same" if ref in silent_here else "silent_entity")
            tier = "reuse_exact" if any(rec.witnesses for rec in exact) else "reuse_para" if para else None
            if tier is not None:
                tiers.append(tier)
                if (v, r, head) not in self.memoed:
                    self.memoed.add((v, r, head))
                    self.ctx.ledger.count_reuse(tier)
            if cands:
                offered.append((a, cands))
        refs = list(dict.fromkeys([*(ref for _, cands in offered for ref in cands), *(ref for _, ref in extra)]))
        held = await self.ctx.kg.entries(self.ctx.ns, refs) if refs else {}
        if index:
            self.iextra = {key: (held[key[1]], *rest) for key, rest in extra.items() if key[1] in held}
            n = Counter(c for entry in self.ientries.values() for c in entry.values())
            for c in INDEX_CLASSES:
                self.ctx.ledger.count_reuse(f"index_{c}", n[c])
            self.ctx.ledger.count_reuse("index_resolved", sum(1 for keys in resolved.values() if keys and
                                                               not any(x == 1.0 for x in keys.values())))
            self.ctx.log("index", round=kind, keys={v: [[k, round(x, 3)] for k, x in keys.items()]
                                                    for v, keys in resolved.items() if keys},
                         **{c: n[c] for c in INDEX_CLASSES})
        pool, outside = self.memo_pool(offered, held)
        if not pool and not negated:
            return
        seen, glanced, spent = set(self.seen), set(self.entries), self.ctx.ledger.evidence_tokens
        if pool:
            self.coverage = 1.0
            await self.search(pool, outside, asks, kind)
            if kind == "text" and self.m.scan:
                await self.complete_units(pool, asks)
        mine = {ref for _, ref in pool}
        reads = len({ref for _, _, ref in (self.seen - seen) | (self.entries - glanced) if ref in mine})
        tokens = self.ctx.ledger.evidence_tokens - spent
        self.ctx.ledger.count_reuse("memo_reads", reads)
        self.ctx.ledger.count_reuse("memo_tokens", tokens)
        asked = {(a.cond.id, norm_entity(a.value or "")) for a, _ in offered}
        values: dict[str, list[str]] = defaultdict(list)
        for cid, ref in pool:
            x = self.conds[cid].var
            for k, o in sorted(self.witnesses.get((x, cid), {}).get(ref, ())):
                if (cid, o) in asked and self.values[x][k].name not in values[x.lstrip("?")]:
                    values[x.lstrip("?")].append(self.values[x][k].name)
        self.ctx.log("reuse", mode=self.m.reuse, exact=tiers.count("reuse_exact"), para=tiers.count("reuse_para"),
                     negatives=negated, n=len(mine), read=reads, tokens=tokens, values=dict(values))

    def yield_of(self, cls: str, prior: float) -> float:
        n, w, e = (*(self.istats or {}).get(cls, (0, 0)), 0.0)[:3]
        if not n:
            return prior
        if e <= 0:
            return (w + prior) / (n + 1)
        return min(1.0, prior * (w + 1.0) / (e + 1.0))

    def memo_pool(self, offered: Sequence[tuple[Asked, dict[Ref, float]]], held: dict[Ref, Candidate],
                  ) -> tuple[dict[tuple[int, Ref], Item], dict[int, float]]:
        cal = self.ctx.calib
        alpha: dict[tuple[int, Ref], float] = defaultdict(float)
        position: dict[Ref, int] = {}
        for a, cands in offered:
            w = self.weight(a.cond.other, a.value)
            for ref, chance in cands.items():
                if ref in held:
                    position.setdefault(ref, len(position))
                    alpha[(a.cond.id, ref)] += w * chance
        pool: dict[tuple[int, Ref], Item] = {}
        for (cid, ref), chance in alpha.items():
            c, beta = held[ref], 1.0
            if not c.complete and c.modality != "image" and c.site not in self.titled:
                beta = cal.beta(c.modality, c.tokens, approx_tokens(c.summary), rep=self.rep(c)) \
                    if self.m.calibrated else \
                    min(1.0, approx_tokens(c.summary) / max(1, c.tokens))
            pool[(cid, ref)] = Item(c, chance, float(position[ref]), beta=beta)
        outside: dict[int, float] = {}
        for cid in dict.fromkeys(cid for cid, _ in pool):
            items = [it for k, it in pool.items() if k[0] == cid]
            z = max(1.0, sum(it.alpha for it in items))
            for it in items:
                it.p = it.alpha / z
            outside[cid] = max(0.0, 1.0 - sum(it.p for it in items))
        return pool, outside

    def skips(self, pool: dict[tuple[int, Ref], Item], asks: Sequence[Asked]) -> None:
        new: set[Ref] = set()
        for cid, ref in pool:
            these = [a for a in asks if a.cond.id == cid]
            if ref not in self.skipped and self.judged(these, ref) and any(
                    (cid, norm_entity(a.value or ""), ref) in self.negated for a in these):
                new.add(ref)
        self.skipped |= new
        self.ctx.ledger.count_reuse("negatives_skipped", len(new))

    def records(self) -> list[AskRecord]:
        read = set(self.read) | {ref for ref, c in self.entries_read.items() if c.complete or c.modality == "image"}
        out = []
        for (cid, v), a in self.made.items():
            if a.cond.edge is None:
                continue
            stated: dict[Ref, list[str]] = {}
            for ref, pairs in self.witnesses.get((a.var, cid), {}).items():
                if names := [self.values[a.var][k].name for k, other in sorted(pairs) if other == v]:
                    stated[ref] = names
            negatives = {ref for c, value, ref in self.seen if c == cid and value == v and ref in read} - set(stated)
            if stated or negatives:
                out.append(AskRecord(v, _relation(a.relation), a.cond.edge.head == a.cond.other, stated,
                                     {ref: self.p_read.get(ref, 0.0) for ref in stated}, negatives))
        return out

    async def remember(self, facts: bool = True, cache: bool = True) -> None:
        if facts:
            await self.remember_facts()
        if cache:
            await self.remember_cache()

    async def remember_facts(self) -> None:
        if not self.m.kg_write or (self.m.kg == "off" and self.m.cache != "off"):
            return
        by_src: dict[Ref, list[Fact]] = defaultdict(list)
        overs = self.root.overs
        for x in self.p.variables:
            for held in self.values[x].values():
                for f in held.facts:
                    if not f.overs and (f.site, f.oid) in overs:
                        f = replace(f, overs=overs[(f.site, f.oid)])
                    by_src[(f.site, f.oid)].append(f)
        if self is self.root and self.root.harvested:
            for f in self.root.harvested:
                by_src[(f.site, f.oid)].append(f)
            self.root.harvested = []
        results = [ExtractionResult(site, oid, facts[0].modality or "text", prompts.READ_VERSION,  # type: ignore[arg-type]
                                    list(dict.fromkeys(facts)), Usage(), qid=self.q.qid)
                   for (site, oid), facts in by_src.items() if facts and site]
        if results:
            await self.ctx.kg.add(self.ctx.ns, results)

    async def remember_cache(self) -> None:
        if self.m.reuse not in CACHED:
            return
        if self.m.reuse == "index" and self.ideltas:
            deltas = {c: (n, w, e) for c, (n, w, e) in self.ideltas.items() if n}
            await self.ctx.kg.add_index_stats(self.ctx.ns, deltas)
            for c, (n, w, _) in deltas.items():
                self.ctx.ledger.count_reuse(f"index_read_{c}", n)
                self.ctx.ledger.count_reuse(f"index_hit_{c}", w)
            self.ctx.log("index_use", read={c: n for c, (n, _, _) in deltas.items()},
                         witnessed={c: w for c, (_, w, _) in deltas.items()},
                         expected={c: round(e, 3) for c, (_, _, e) in deltas.items()},
                         stats={c: list(v) for c, v in (self.istats or {}).items()})
            self.ideltas = defaultdict(lambda: [0, 0, 0.0])
        if records := self.records():
            named = dict.fromkeys(ref for rec in records for ref in [*rec.witnesses, *sorted(rec.negatives)])
            await self.ctx.kg.add_asks(self.ctx.ns, records, [self.released[r] for r in named if r in self.released])
        if self.probed:
            await self.ctx.kg.add_probes(self.ctx.ns, self.probed)

    def tuple_key(self, kind: str, site: str, oid: str, version: str, unknowns: Sequence[ReadUnknown],
                  label: str = "") -> str:
        spec = [{k: v for k, v in u.to_dict().items() if k != "name"} for u in unknowns]
        scoped = any("the question" in c for u in unknowns for c in u.conditions)
        return stable_hash("tuple-v2", kind, site, oid, version, canonical_json(spec), label,
                           *([self.q.question] if scoped else []), n=24)

    async def tuple_split(self, requests: dict[str, dict[str, Any]], unknowns: Sequence[ReadUnknown],
                          ) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, list[dict[str, Any]]]]]:
        keys: dict[str, tuple[str, str]] = {}
        for site, req in requests.items():
            labels = req.get("labels") or {}
            for block in req.get("blocks") or []:
                for oid in block:
                    ver = self.root.overs.get((site, oid))
                    if ver:
                        keys[self.tuple_key("site", site, oid, ver, unknowns, labels.get(oid, ""))] = (site, oid)
        self.cached_oids: dict[str, set[str]] = defaultdict(set)
        if not keys:
            return requests, {}
        held = await self.ctx.kg.tuples(self.ctx.ns, list(keys))
        self.ctx.ledger.count_reuse("tuple_lookups", len(keys))
        self.ctx.ledger.count_reuse("tuple_hits", len(held))
        out: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
        for key, payload in held.items():
            site, oid = keys[key]
            self.cached_oids[site].add(oid)
            for u, hits in zip(unknowns, payload.get("hits") or [], strict=False):
                out[site][u.name] += [{"entity": h["entity"], "oid": oid, "conditions": list(h["conditions"])}
                                      for h in hits]
        kept = {}
        for site, req in requests.items():
            blocks = [[oid for oid in block if oid not in self.cached_oids[site]] for block in req.get("blocks") or []]
            kept[site] = {**req, "blocks": [b for b in blocks if b]}
        return kept, {s: dict(v) for s, v in out.items()}

    async def tuple_store(self, requests: dict[str, dict[str, Any]], results: dict[str, Any],
                          unknowns: Sequence[ReadUnknown]) -> None:
        if not self.m.kg_write:
            return
        items: dict[str, dict[str, Any]] = {}
        for site, r in results.items():
            if r.error is not None:
                continue
            labels = (requests.get(site) or {}).get("labels") or {}
            for oid in r.read:
                ver = r.versions.get(oid) or self.root.overs.get((site, oid))
                if not ver:
                    continue
                hits = [[{"entity": h["entity"], "conditions": h.get("conditions") or []}
                         for h in r.bindings.get(u.name, []) if h.get("oid") == oid] for u in unknowns]
                if not any(hits):
                    continue
                items[self.tuple_key("site", site, oid, ver, unknowns, labels.get(oid, ""))] = {"hits": hits}
        if items:
            await self.ctx.kg.add_tuples(self.ctx.ns, items)

    async def push(self, requests: dict[str, dict[str, Any]], unknowns: list[ReadUnknown],
                   by_var: dict[str, list[Asked]], chosen: Sequence[Call] = ()) -> dict[str, Any]:
        extract = self.m.extracts
        if extract:
            requests = {s: {**req, "extract": True} for s, req in requests.items()}
        if self.m.mention != 1:
            requests = {s: {**req, "mention": self.m.mention} for s, req in requests.items()}
        if self.m.epoch is not None:
            requests = {s: {**req, "epoch": self.m.epoch} for s, req in requests.items()}
        if self.m.versioned_reads:
            harvest = self.harvest_size(unknowns)
            more = {"versions": True, **({"harvest": harvest} if harvest else {})}
            requests = {s: {**req, **more} for s, req in requests.items()}
        cached: dict[str, dict[str, list[dict[str, Any]]]] = {}
        if self.m.cache == "tuple" and not extract:
            requests, cached = await self.tuple_split(requests, unknowns)
        sent = [u.to_dict() for u in unknowns]
        results = await self.ctx.fed.read(requests, self.read_question(), sent,
                                          include_raw=self.m.include_raw and not extract, ledger=self.ctx.ledger) \
            if any(any(req.get("blocks") or []) for req in requests.values()) else {}
        extracted: list[tuple[str, ExtractionResult, str]] = []
        asks = [a for items in by_var.values() for a in items]
        if self.m.cache == "tuple" and not extract:
            await self.tuple_store(requests, results, unknowns)
            for site, bindings in cached.items():
                oids = {h["oid"] for hits in bindings.values() for h in hits} | set(self.cached_oids.get(site, ()))
                self.root.cached_refs.update((site, oid) for oid in oids)
                self.root.witnessable.update((site, oid) for oid in oids)
                self.mark(asks, ((site, oid) for oid in oids))
                self.merge(bindings, by_var, prompts.READ_VERSION, {}, site)
                self.settle(bindings, by_var)
        for site, r in results.items():
            req = requests.get(site, {})
            blocks = req.get("blocks", [])
            asked = {oid for block in blocks for oid in block}
            self.root.witnessable.update((site, oid) for oid in r.read if oid in asked)
            if not extract:
                self.set_reads(unknowns, len(blocks))
            self.mark(asks, ((site, oid) for block in blocks for oid in block))
            if r.error is None:
                self.spent += len(r.read)
                self.read.extend(ref for ref in ((site, oid) for oid in r.read) if ref not in self.read)
            self.merge(r.bindings, by_var, prompts.READ_VERSION, {(site, o): mod for o, mod in r.modalities.items()}, site)
            self.settle(r.bindings, by_var)
            for oid, text in r.raw.items():
                self.snippets.setdefault((site, oid), text)
            self.doc_ids.update({(site, oid): d for oid, d in r.doc_ids.items() if d})
            if r.versions:
                self.root.overs.update({(site, oid): v for oid, v in r.versions.items()})
            if r.harvest:
                from fedquest.site.service import HARVEST_VERSION

                self.root.harvested += [Fact(h["entity"], h["relation"], h["value"], site, h["oid"], HARVEST_VERSION,
                                             r.modalities.get(h["oid"], "text"),
                                             r.versions.get(h["oid"], ""), str(h.get("w") or "")) for h in r.harvest]
                self.ctx.ledger.count_reuse("kg_harvested", len(r.harvest))
            extracted += [(site, x, r.titles.get(x.oid, "")) for x in r.extractions if x.error is None and x.facts]
            sizes = {x.key[1][1]: x.c for call in chosen if call.place.rsplit("|", 1)[0] == site for x in call.block
                     if isinstance(x.key[1], tuple)}
            self.ctx.log("read", site=site, read=r.read, skipped=len(r.skipped), dropped=r.dropped, error=r.error,
                         tokens=r.usage.cost_units, seconds=round(r.usage.seconds, 3),
                         obj_tokens=round(sum(sizes.get(oid, 0.0) for oid in r.read)),
                         values={n: [h["entity"] for h in v] for n, v in r.bindings.items()},
                         cited=_cited(r.bindings),
                         **({"blocks": blocks, "unknowns": sent, "labels": req.get("labels") or {},
                             "prefilter": req.get("prefilter") or [], "exempt": req.get("exempt") or []}
                            if self.m.audited else {}))
        if extract:
            for call in chosen:
                site = call.place.rsplit("|", 1)[0]
                skipped = set(results[site].skipped) if site in results else set()
                kept = [x for x in call.block if x.key[1][1] not in skipped]
                if kept:
                    self.shadow += call.cost - sum(x.c for x in call.block) + sum(x.c for x in kept)
            await self.facts_read(extracted, unknowns, by_var)
        return results

    async def facts_read(self, extracted: Sequence[tuple[str, ExtractionResult, str]], unknowns: list[ReadUnknown],
                         by_var: dict[str, list[Asked]]) -> None:
        if not extracted:
            return
        self.root.witnessable.update((s, x.oid) for s, x, _ in extracted)
        self.set_reads(unknowns)
        objs = [ReadObject("facts", f"{title}: {prompts.format_facts(x.facts, 25)}") for _, x, title in extracted]
        reply = await self.ctx.llm.chat(prompts.read_messages(self.q.question, objs, unknowns), role="answer",
                                        task="read_facts", max_tokens=prompts.read_max_tokens(unknowns),
                                        json_mode=True, ledger=self.ctx.ledger, charge="bind")
        obj = prompts.read_reply(reply.text)
        hits: dict[str, list[dict[str, Any]]] = defaultdict(list)
        parsed = prompts.parse_read(obj, unknowns, len(extracted))
        for u in unknowns:
            for h in parsed.get(u.name, []):
                site, x, title = extracted[h.object - 1]
                entity: str | None = h.entity
                if x.modality == "image" and u.mode == "filter":
                    said = prompts.judgment(h.entity)
                    named = said if said is not None else prompts.canonical(h.entity, [title]) is not None
                    entity = title if named else None
                elif not prompts.stated(" ; ".join([title, *(f"{f.head} ; {f.tail}" for f in x.facts)]), u, h.entity):
                    entity = None
                text = f"{title}: " + " ; ".join(f.text() for f in x.facts)
                if entity and (conds := prompts.claims(None if x.modality == "image" else text, u, h, title)):
                    hits[u.name].append({"entity": entity, "site": site, "oid": x.oid, "conditions": conds})
        self.merge(hits, by_var, prompts.QUERY_EXTRACT_VERSION, {(s, x.oid): x.modality for s, x, _ in extracted})
        self.settle(hits, by_var)
        self.ctx.log("read", site="coordinator (triples)", read=[x.oid for _, x, _ in extracted], error=None,
                     values={n: [h["entity"] for h in v] for n, v in hits.items()}, cited=_cited(hits))
        modality = {(s, x.oid): x.modality for s, x, _ in extracted}
        cited = [ref for ref in dict.fromkeys((h["site"], h["oid"]) for v in hits.values() for h in v)
                 if ref[0] not in self.titled and modality.get(ref) != "image"]
        if self.m.include_raw and cited:
            for raw in await self.ctx.fed.fetch_raw(cited, ledger=self.ctx.ledger, epoch=self.m.epoch):
                if raw.text and not raw.denied:
                    self.snippets.setdefault((raw.site, raw.oid), _titled(raw))

    def held_facts(self) -> list[Fact]:
        return list(dict.fromkeys(f for x in self.p.variables for held in self.values[x].values() for f in held.facts))

    def doc_index(self) -> dict[Ref, str]:
        docs: dict[Ref, str] = {}
        for c in [*self.question_top, *self.entries_read.values()]:
            if c.doc_id:
                docs.setdefault(c.ref, c.doc_id)
        for r in self.chain():
            docs.update({ref: d for ref, d in r.doc_ids.items() if d})
        return docs

    async def finish(self, cache: bool = True) -> MethodResult:
        await self.remember(cache=cache)
        if self.m.include_raw:
            wanted = [r for r in dict.fromkeys([*self.reused, *sorted(self.root.cached_refs)])
                      if r not in self.snippets and r[0] not in self.titled and self.reused_modality.get(r) != "image"]
            for raw in await self.ctx.fed.fetch_raw(wanted, ledger=self.ctx.ledger,
                                                    epoch=self.m.epoch) if wanted else []:
                if raw.text and not raw.denied:
                    self.snippets.setdefault((raw.site, raw.oid), _titled(raw))
        chain = {(x, norm_entity(v)) for x, vals in self.bound.items() for v in vals}
        first = [f for x in self.p.variables for held in self.values[x].values()
                 if (x, norm_entity(held.name)) in chain for f in held.facts]
        rest = [f for x in self.p.variables for held in self.values[x].values() for f in held.facts]
        runs = self.chain()
        for r in runs[1:]:
            pc = {(x, norm_entity(v)) for x, vals in r.bound.items() for v in vals}
            first += [f for x in r.p.variables for held in r.values[x].values()
                      if (x, norm_entity(held.name)) in pc for f in held.facts]
            rest += [f for x in r.p.variables for held in r.values[x].values() for f in held.facts]
        facts = list(dict.fromkeys([*first, *rest]))[:60]
        cited = list(dict.fromkeys((f.site, f.oid) for f in facts))
        texts = {ref: c.summary for ref, c in self.entries_read.items() if c.site not in self.titled}
        texts.update(self.snippets)
        union = self.m.context == "union"
        order = [*[r for r in cited if r in texts], *([c.ref for c in self.question_top] if union else []),
                 *self.indexed, *self.read, *texts]
        if not facts or union:
            texts.update({c.ref: c.summary for c in self.question_top
                          if c.site not in self.titled and c.ref not in texts})
            order += [] if union else [c.ref for c in self.question_top]
        refs = [r for r in dict.fromkeys(order) if r in texts]
        kept = _context_refs(refs, texts, self.m.context_tokens(), self.m.snippet_items)
        snippets = [texts[r] for r in kept]
        docs = self.doc_index()
        answer = await generate_answer(self.q, self.ctx, facts, snippets if self.m.include_raw else [],
                                       doc_ids=[docs.get(r, "") for r in kept] if self.m.include_raw else None)
        stopped = self.matched()
        if not self.done() and self.out_of_rounds():
            self.limit = "rounds"
        limit = None if stopped else self.limit
        extra: dict[str, Any] = {"unmentioned": self.unmentioned} if self.m.mentions else {}
        if self.m.planner == "query":
            extra |= {"partial": sorted(self.cut), "pruned": {y: list(v) for y, v in self.pruned.items()}}
            if self.cut & self.upstream():
                limit = "allocation"
        if self.prior is not None:
            extra["run"] = self.RUN
        self.ctx.log("stop", matched=stopped, read=self.spent, rounds=self.rounds, bound=self.bound,
                     dead=sorted(self.dead), reused=len(self.reused), limit=limit, **extra)
        ev = Evidence()
        ev.doc_ids.update(self.doc_ids)
        bindings: dict[str, str] = {}
        for r in reversed(runs):
            bindings.update({x: v[0] for x, v in r.bound.items() if v})
        pattern = runs[-1].p
        return finish(self.q, self.m.name, answer, pattern=pattern, evidence=ev, facts=facts, consulted=self.read,
                      indexed=self.indexed, stopped=stopped, bindings=bindings, reused=self.reused)


class _Direct(_Run):
    RUN = "direct"

    def __init__(self, m: FedQuest, q: Question, ctx: QueryContext, prior: _Run) -> None:
        super().__init__(m, q, ctx, prompts.parse_pattern("", q.question), (), prior=prior)
        self.max_rounds = 1
        self.marks = compiler.marks(self.p, q.question, pictures=False)
        self.conds = conditions(self.p, q.question, marks=self.marks)
        self.by_var = defaultdict(list)
        for c in self.conds:
            self.by_var[c.var].append(c)
        self.called: set[str] = set()
        self.at_site: Counter[str] = Counter()

    async def paths(self, kind: str, asks: Sequence[Asked]) -> list[Found]:
        self.round_kind = kind
        self.reach, self.round_lists, self.counted, self._nodes = self.m.k, [], {}, {0: None}
        conds = tuple(dict.fromkeys(a.cond.id for a in asks))
        anchors = self.chain()[-1].p.anchors
        return [Found(c, "question", lead_signal(c, anchors), None, conds, 0)
                for c in self.question_top]

    def call_cap(self, place: str, unknowns: Sequence[ReadUnknown]) -> int:
        if place in self.called:
            return 0
        cap = super().call_cap(place, unknowns)
        return cap if place == COORD else max(0, min(cap, self.m.direct_objects - self.at_site[place.split("|", 1)[0]]))

    def plan(self, pool: dict[tuple[int, Ref], Item], outside: dict[int, float], asks: Sequence[Asked],
             unknowns: Sequence[ReadUnknown], kind: str) -> list[Call]:
        chosen = super().plan(pool, outside, asks, unknowns, kind)
        for call in chosen:
            self.called.add(call.place)
            if call.place != COORD:
                self.at_site[call.place.split("|", 1)[0]] += len(call.block)
        return chosen

    async def deepen(self, kind: str, asks: Sequence[Asked], found: list[Found],
                     pool: dict[tuple[int, Ref], Item]) -> dict[tuple[int, Ref], Item]:
        return pool

    async def join_pages(self, kind: str, asks: Sequence[Asked], found: list[Found],
                         pool: dict[tuple[int, Ref], Item]) -> dict[tuple[int, Ref], Item]:
        return pool

    async def complete_units(self, pool: dict[tuple[int, Ref], Item], asks: Sequence[Asked]) -> None:
        return None

    def matched(self) -> bool:
        return self.stop_test() or (self.prior is not None and self.prior.matched())


class _Chain(_Direct):
    RUN = "chain"

    def __init__(self, m: FedQuest, q: Question, ctx: QueryContext, prior: _Run, step: int = 0) -> None:
        super().__init__(m, q, ctx, prior)
        self.step_no = step
        self.fresh = 0
        if m.tokens is not None:
            self.tokens = max(m.tokens, math.ceil(prior.evidence()) + CHAIN_FLOOR)
        if m.total_budget is not None:
            self.tokens = min(self.tokens, m.evidence_cap(ctx))
        self.object_room = prior.spent + CHAIN_OBJECTS

    def left(self) -> int:
        return max(0, max(self.m.objects, self.object_room) - self.spent)

    def call_cap(self, place: str, unknowns: Sequence[ReadUnknown]) -> int:
        cap = _Run.call_cap(self, place, unknowns)
        return 0 if place in self.called else cap

    def known_facts(self) -> list[Fact]:
        out: list[Fact] = []
        for r in reversed(self.chain()[1:]):
            bound = {(x, norm_entity(v)) for x, vals in r.bound.items() for v in vals}
            out += [f for x in r.p.variables for held in r.values[x].values()
                    if (x, norm_entity(held.name)) in bound for f in held.facts]
            out += [f for x in r.p.variables for held in r.values[x].values() for f in held.facts]
        return list(dict.fromkeys(out))

    def fact_lines(self) -> list[str]:
        facts = [f for f in self.known_facts() if f.head != self.q.question]
        return [f"{f.head} {f.relation} {f.tail}" for f in facts[-CHAIN_FACTS:]]

    def chain_key(self) -> str:
        lines = self.fact_lines()
        return self.q.question if not lines else f"{self.q.question} {'. '.join(lines)}."

    def read_question(self) -> str:
        lines = self.fact_lines()
        if not lines:
            return self.q.question
        return f"{self.q.question}\nKnown so far: " + "; ".join(lines) + "."

    async def paths(self, kind: str, asks: Sequence[Asked]) -> list[Found]:
        self.round_kind = kind
        self.reach, self.counted, self._nodes = self.m.k, {}, {0: None}
        conds = tuple(dict.fromkeys(a.cond.id for a in asks))
        anchors = self.chain()[-1].p.anchors
        cols = self.collections(image=kind != "text")
        req = ProbeRequest(self.chain_key(), "question", self.m.k, collections=cols)
        self.round_lists = [(req, ("question", None, conds, None))]
        read = set(self.read)
        got = (await self.probe([req]))[0] if cols else []
        found = [Found(c, "question", lead_signal(c, anchors), None, conds, 0)
                 for c in got if c.ref not in read]
        self.fresh = len(found)
        if self.step_no == 0:
            seen = {f.c.ref for f in found}
            found += [Found(c, "question", lead_signal(c, anchors), None, conds, 0)
                      for c in self.question_top if c.ref not in read and c.ref not in seen]
        self.ctx.log("chain", step=self.step_no, key=req.key, fresh=self.fresh, offered=len(found))
        return found


def single_edge(pattern: Pattern, question: str) -> bool:
    if pattern.raw.get("fallback"):
        return True
    if len(pattern.edges) != 1:
        return False
    e = pattern.edges[0]
    return e.head == question and e.relation == "answer" and e.tail == pattern.answer


def _relation(r: str) -> str:
    return " ".join(r.lower().split())
