from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import replace

from fedquest import budget
from fedquest.kg.matcher import MatchResult, TextSim
from fedquest.kg.normalize import (  # noqa: F401
    norm_entity,
    release_signal,
    same_entity,
)
from fedquest.llm import prompts
from fedquest.llm.parsing import parse_answer
from fedquest.methods.base import MethodResult, QueryContext
from fedquest.planner.calibration import VECTOR_KINDS, GroupRank, rank_feature
from fedquest.types import (
    Candidate,
    ExtractionResult,
    Fact,
    Pattern,
    ProbeCounts,
    ProbeRequest,
    Question,
    SiteInfo,
    is_var,
)
from fedquest.utils import stable_int

Ref = tuple[str, str]
Sizes = Mapping[tuple[str, str], tuple[str, int]]
Note = Callable[..., None]


def ref_str(ref: Ref) -> str:
    return f"{ref[0]}/{ref[1]}"


VISUAL = re.compile(
    r"\b(posters?|images?|pictures?|pictured|photos?|photographs?|logos?|flags?|emblems?|crest|"
    r"colou?r(s|ed)?|depict(s|ed|ing)?|wears?|wearing|worn|paintings?|portraits?|drawings?|illustrat\w*|"
    r"visible|hair|beard|mustache|moustache|glasses|smil(e|es|ed|ing)|facial|title screen)\b"
    r"|\b(cover|artwork)s?\b(?:\s+\w+){0,2}\s+(?:show\w*|featur\w*|depict\w*|contain\w*)\b"
    r"|\bon\s+(?:the|its|their|his|her|a)\s+(?:album\s+|front\s+)?cover\b|\b(?:his|her|their|its)\s+face\b", re.IGNORECASE)


def hint_modalities(pattern: Pattern, note: Note | None = None) -> Pattern:
    for e in pattern.edges:
        if e.modality is None and VISUAL.search(e.relation):
            e.modality = "image"
            if note is not None:
                note("visual_lexicon", f"atom ({e.head} | {e.relation} | {e.tail})")
    return pattern


_MEDIUM = r"(poster|logo|cover|image|picture|photo|photograph|flag|emblem|crest|portrait|artwork)"
_HAS_MEDIUM = re.compile(rf"^(?:has|have|with)\s+(?:an?\s+|the\s+|its\s+)?(?:official\s+|title\s+|movie\s+|album\s+)?"
                         rf"{_MEDIUM}s?(?:\s+of)?$", re.IGNORECASE)


def fold_media(pattern: Pattern, note: Note | None = None) -> Pattern:
    for e in list(pattern.edges):
        m, owner = e.tail, e.head
        word = re.search(_MEDIUM, m, re.IGNORECASE) if is_var(m) else None
        if word is None or m == pattern.answer or not _HAS_MEDIUM.match(e.relation.strip()):
            continue
        others = [f for f in pattern.edges if f is not e and m in (f.head, f.tail)]
        if not others:
            continue
        medium = word.group(1).lower()
        if note is not None:
            lost = "" if is_var(owner) or not pattern.conditions.get(m) else \
                f" (its filters left out: {'; '.join(pattern.conditions[m])})"
            note("fold_media", f"{m} into {owner}{lost}", [(f.head, f.relation, f.tail) for f in (e, *others)])
        for f in others:
            if f.head == m:
                f.head, f.relation = owner, f"{medium} {f.relation}"
            if f.tail == m:
                f.tail, f.relation = owner, f"{f.relation} in {medium} of"
            f.modality = "image"
        if m in pattern.conditions and is_var(owner):
            moved = [f"{medium} {c}" for c in pattern.conditions.pop(m)]
            pattern.conditions[owner] = [*pattern.conditions.get(owner, []), *moved]
        pattern.edges = [f for f in pattern.edges if f is not e and f.head != f.tail]
        pattern.variables = [v for v in pattern.variables if any(v in (f.head, f.tail) for f in pattern.edges)]
        pattern.conditions = {v: c for v, c in pattern.conditions.items() if v in pattern.variables}
    return pattern


_DECLINED = frozenset({"", "unknown", "none", "n/a", "na", "no answer", "not mentioned", "not available",
                       "not specified", "not provided", "not given", "not stated", "cannot be determined", "undetermined"})
_REFUSAL = re.compile(
    r"\b(?:not (?:mentioned|provided|specified|stated|available|given|found|included|known)|"
    r"(?:does|do|did) not (?:contain|mention|say|state|specify|provide|include|give|indicate)|"
    r"(?:cannot|can't|could not|couldn't|unable to) (?:be determined|determine|answer|be answered|find|tell|say)|"
    r"no (?:information|mention|details|data) (?:about|on|of|regarding|is|was)|insufficient (?:information|context|"
    r"evidence)|i am sorry|i'm sorry|there is no (?:information|mention))\b", re.IGNORECASE)


def is_abstention(answer: str) -> bool:
    text = " ".join((answer or "").split())
    if re.sub(r"[^\w\s/]", "", text).strip().lower() in _DECLINED:
        return True
    return len(text.split()) <= 30 and bool(_REFUSAL.search(text))


async def make_pattern(q: Question, ctx: QueryContext) -> Pattern:
    reply = await ctx.llm.chat(prompts.pattern_messages(q.question), role="pattern", task="pattern",
                               max_tokens=600, ledger=ctx.ledger, charge="pattern")
    return hint_modalities(fold_media(prompts.parse_pattern(reply.text, q.question)))


class Evidence:
    def __init__(self) -> None:
        self._facts: dict[Fact, None] = {}
        self.by_ref: dict[Ref, list[Fact]] = defaultdict(list)
        self.snippets: dict[Ref, str] = {}
        self.doc_ids: dict[Ref, str] = {}
        self.entities: dict[Ref, list[tuple[str, str]]] = {}

    def entity_records(self) -> list[tuple[str, str]]:
        return list(dict.fromkeys(e for recs in self.entities.values() for e in recs))

    def passages(self, refs: Iterable[Ref] | None = None) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for ref in dict.fromkeys(refs if refs is not None else [*self.by_ref, *self.entities]):
            names = [n for n, _ in self.entities.get(ref, [])] or [x for f in self.by_ref.get(ref, []) for x in
                                                                  (f.head, f.tail)]
            if names:
                out[ref_str(ref)] = list(dict.fromkeys(names))
        return out

    @property
    def facts(self) -> list[Fact]:
        return list(self._facts)

    def add_facts(self, facts: Iterable[Fact], ref: Ref | None = None) -> int:
        added = 0
        for f in facts:
            if f not in self._facts:
                self._facts[f] = None
                added += 1
            if ref is not None and f not in self.by_ref[ref]:
                self.by_ref[ref].append(f)
        return added

    def add_result(self, r: ExtractionResult) -> int:
        ref = (r.site, r.oid)
        if r.raw_text:
            self.snippets[ref] = r.raw_text
        if r.doc_id:
            self.doc_ids[ref] = r.doc_id
        if r.entities:
            kept = self.entities.setdefault(ref, [])
            for rec in r.entities:
                pair = (str(rec[0]), str(rec[1]) if len(rec) > 1 else "") if rec else ("", "")
                if pair[0] and pair not in kept:
                    kept.append(pair)
        return self.add_facts(r.facts, ref)


def split_lists(cands: Sequence[Candidate]) -> list[list[Candidate]]:
    groups: dict[tuple[str, str], list[Candidate]] = defaultdict(list)
    for c in cands:
        groups[(c.site, c.collection)].append(c)
    return [sorted(g, key=lambda c: c.rank) for g in groups.values()]


def tie(salt: str, c: Candidate) -> int:
    return stable_int(salt, c.site, c.collection)


def first_per_ref(cands: Sequence[Candidate]) -> list[Candidate]:
    seen: set[tuple[str, str]] = set()
    out = []
    for c in cands:
        if c.ref not in seen:
            seen.add(c.ref)
            out.append(c)
    return out


def rrf(lists: Sequence[Sequence[Candidate]], k: int = 60, salt: str = "") -> list[Candidate]:
    score: dict[Ref, float] = defaultdict(float)
    first: dict[Ref, Candidate] = {}
    for lst in lists:
        for c in lst:
            score[c.ref] += 1.0 / (k + c.rank)
            first.setdefault(c.ref, c)
    return sorted(first.values(), key=lambda c: (-score[c.ref], c.rank, tie(salt, c), c.oid))


def key_text(value: str, relation: str, value_is_head: bool) -> str:
    return f"{value} {relation}" if value_is_head else f"{relation} {value}"


def group_sizes(info: Mapping[str, SiteInfo],
                ) -> tuple[dict[tuple[str, str], tuple[str, int]], Counter[tuple[str, str]]]:
    sizes: dict[tuple[str, str], tuple[str, int]] = {}
    groups: Counter[tuple[str, str]] = Counter()
    for site, si in info.items():
        for c in si.collections:
            sizes[(site, c.name)] = (c.encoder, c.size)
            groups[(c.name, c.encoder)] += c.size
    return sizes, groups


def shaped(r: ProbeRequest, kind: str, round_kind: str, sizes: Sizes, groups: Mapping[tuple[str, str], int], *,
           semijoin: bool = False, share: bool = False, vocab: bool = False) -> list[ProbeRequest]:
    if round_kind == "filter" or kind not in VECTOR_KINDS:
        return [r]
    if semijoin and round_kind == "text" and kind in ("bound", "bridge") and r.value and not r.semijoin:
        r = replace(r, semijoin=True, mentions=[r.value], **({"vocab": [r.value]} if vocab else {}))
    if not share or r.k_group is not None:
        return [r]
    names = r.collections if r.collections is not None else sorted({c for _, c in sizes})
    parts = []
    for name in names:
        for enc in sorted({e for (_, c), (e, _) in sizes.items() if c == name}):
            sites = sorted(s for (s, c), (e, _) in sizes.items()
                           if c == name and e == enc and (r.sites is None or s in r.sites))
            if sites:
                parts.append((name, enc, sites))
    if all(len(sites) == 1 for _, _, sites in parts):
        return [r]
    def size(name: str, enc: str, sites: list[str]) -> int:
        if kind == "question" and r.sites is not None:
            return sum(sizes[(s, name)][1] for s in sites) or groups[(name, enc)]
        return groups[(name, enc)]

    return [replace(r, collections=[name], sites=sites, k_group=r.k, group_size=size(name, enc, sites), encoder=enc)
            if len(sites) > 1 else replace(r, collections=[name], sites=sites) for name, enc, sites in parts]


def group_rank(kind: str, rnd: str, c: Candidate, sizes: Sizes, groups: Mapping[tuple[str, str], int],
               counts: Sequence[ProbeCounts] | None = None) -> GroupRank:
    encoder, m_site = sizes.get((c.site, c.collection), ("", 0))
    if counts is not None and rnd != "filter" and kind in VECTOR_KINDS:
        j_site = sum(n.joinable for n in counts if n.site == c.site and n.collection == c.collection)
        j_group = sum(n.joinable for n in counts if n.collection == c.collection
                      and sizes.get((n.site, n.collection), ("", 0))[0] == encoder)
        return rank_feature(kind, rnd, c.rank, jrank=c.jrank if c.jrank is not None else c.rank, j_site=j_site,
                            j_group=j_group)
    return rank_feature(kind, rnd, c.rank, m_site, groups.get((c.collection, encoder), 0))


def answer_facts(match: MatchResult | None, facts: Sequence[Fact], sim: TextSim, question: str,
                 limit: int = 60) -> list[Fact]:
    support = match.support_facts() if match is not None else []
    rest = [f for f in dict.fromkeys(facts) if f not in support]
    if rest and len(support) + len(rest) > limit:
        scores = sim.sims(question, [f.text() for f in rest])
        rest = [f for _, f in sorted(zip(scores, rest, strict=True), key=lambda t: -t[0])]
    return (support + rest)[:limit]


class StopTest:
    def __init__(self, q: Question, ctx: QueryContext, limit: int = 40, verify: bool | None = None) -> None:
        self.q, self.ctx, self.limit = q, ctx, limit
        self.verify = ctx.verify if verify is None else verify
        self._seen: dict[frozenset[Fact], bool] = {}
        self.rejected = 0

    async def __call__(self, match: MatchResult, facts: Sequence[Fact]) -> bool:
        if not match.complete:
            return False
        if not self.verify:
            return True
        shown = answer_facts(match, facts, self.ctx.matcher.sim, self.q.question, self.limit)
        key = frozenset(shown)
        if key not in self._seen:
            obj, _ = await self.ctx.llm.chat_json(
                prompts.verify_messages(self.q.question, prompts.format_facts(shown)), role="answer", task="verify",
                max_tokens=200, ledger=self.ctx.ledger, charge="verify", default={},
            )
            ok = isinstance(obj, dict) and obj.get("answerable") in (True, "true", "yes")
            self._seen[key] = ok
            self.rejected += not ok
            self.ctx.log("verify", answerable=ok, facts=len(shown))
        return self._seen[key]


async def generate_answer(q: Question, ctx: QueryContext, facts: Sequence[Fact], snippets: Sequence[str],
                          images: Sequence[str | tuple[str, str]] | None = None,
                          doc_ids: Sequence[str] | None = None) -> str:
    left = budget.remaining(ctx.ledger)
    if left is not None:
        def frame_of(fs: Sequence[Fact]) -> int:
            return budget.count_messages(prompts.answer_messages(q.question, prompts.format_facts(fs), [],
                                                                 ctx.answer_style, images))

        reply_cost = budget.OUT_WEIGHT * 24
        frame = frame_of(facts)
        if len(facts) > 1 and frame + reply_cost > left:
            lo, hi = 1, len(facts)
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if frame_of(facts[:mid]) + reply_cost <= left:
                    lo = mid
                else:
                    hi = mid - 1
            ctx.log("answer_cut_facts", kept=lo, offered=len(facts))
            facts = list(facts)[:lo]
            frame = frame_of(facts)
        room = left - frame - reply_cost
        kept = budget.fit_indices(list(snippets), max(0, room))
        if len(kept) < len(snippets):
            ctx.log("answer_cut", kept=len(kept), offered=len(snippets), room=room)
        if doc_ids is not None:
            ids = list(doc_ids)
            doc_ids = [ids[i] for i, _ in kept if i < len(ids) and ids[i]]
        snippets = [t for _, t in kept]
    if doc_ids:
        ctx.ledger.note_context(doc_ids)
    messages = prompts.answer_messages(q.question, prompts.format_facts(facts), snippets, ctx.answer_style, images)
    reply = await ctx.llm.chat(messages, role="answer_vision" if images else "answer", task="answer",
                               max_tokens=128 if ctx.answer_style == "short" else 400,
                               ledger=ctx.ledger, charge="answer")
    return parse_answer(reply.text)


def finish(
    q: Question,
    name: str,
    answer: str,
    *,
    pattern: Pattern | None = None,
    match: MatchResult | None = None,
    evidence: Evidence | None = None,
    facts: Sequence[Fact] = (),
    consulted: Sequence[Ref] = (),
    reused: Sequence[Ref] = (),
    indexed: Sequence[Ref] = (),
    stopped: bool | None = None,
    bindings: dict[str, str] | None = None,
) -> MethodResult:
    ev = evidence or Evidence()
    support = match.support_facts() if match is not None else []
    return MethodResult(
        qid=q.qid,
        method=name,
        answer=answer,
        matched=bool(match and match.complete) if stopped is None else stopped,
        pattern=pattern.to_dict() if pattern else None,
        bindings=dict(bindings) if bindings is not None else dict(match.bindings) if match else {},
        evidence=[f.to_dict() for f in facts],
        consulted=[ref_str(r) for r in consulted],
        reused=[ref_str(r) for r in reused],
        indexed=[ref_str(r) for r in indexed],
        consulted_doc_ids=[ev.doc_ids[r] for r in dict.fromkeys([*consulted, *reused, *indexed]) if r in ev.doc_ids],
        support_oids=sorted({f"{f.site}/{f.oid}" for f in support}),
    )
