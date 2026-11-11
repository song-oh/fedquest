# typed query graph compiler: prompts, lowering to a Pattern, checks and repair
from __future__ import annotations

import copy
import dataclasses
import json
import math
import re
from collections import Counter, defaultdict, deque
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cache
from typing import TYPE_CHECKING, Any

from rapidfuzz import fuzz

from fedquest.config import configs_dir
from fedquest.kg.normalize import _LETTER, _NUMERIC_NAMES, norm_entity
from fedquest.llm import prompts
from fedquest.llm.parsing import _unquote, parse_json
from fedquest.methods.common import _MEDIUM, VISUAL, Note, fold_media, hint_modalities
from fedquest.paths import CONFIGS
from fedquest.types import Edge, Pattern, ProbeRequest, Question, is_var

if TYPE_CHECKING:
    from fedquest.methods.base import QueryContext
    from fedquest.methods.fedquest import Cond, _Run

TYPED_VERSION = "typed-v1"
FORMS = ("entity", "yes/no", "choice", "count", "date", "number")

FALLBACK_VISUAL = "is the one whose picture shows what the question describes"
_OF_OWNER = re.compile(r"(?:^|\b(?:in|on|from|of)\s+(?:the\s+|a\s+|its\s+)?)(?:album\s+)?(?:posters?|logos?|covers?|"
                       r"images?|pictures?|photos?|photographs?|flags?|emblems?|crests?|portraits?|artworks?)\s+of$",
                       re.I)
_TYPE_EDGE = re.compile(r"^(?:is|was|are|were)(?:\s+an?)?$|^(?:is\s+an?\s+)?(?:instance|type|kind|sort)\s+of$", re.I)
LOWERING_RULES = ("concept", "fold_media", "visual_lexicon", "type_edge", "described", "leaf", "capability",
                  "question_visual", "owner_of_medium", "numeric_name")
LOWERING_KINDS: dict[str, tuple[str, str]] = {
    "concept": ("equivalence", "the unknown is a concept the question names word for word, so its name is its value; "
                               "a middle one's two atoms compose into one relation"),
    "fold_media": ("equivalence", "a picture is its object's: the variable that only names it is the object, read on "
                                  "its picture"),
    "visual_lexicon": ("heuristic", "a word list (VISUAL) decides that an atom or a filter needs image evidence"),
    "type_edge": ("relaxation", "drops an atom that only says what its other end is (\"is a\", \"type of\")"),
    "described": ("heuristic", "the question's description of a leaf (\"?pyramid1\": \"pyramid\") is taken as its "
                               "name"),
    "leaf": ("relaxation", "drops an atom whose non-answer leaf no source can name"),
    "capability": ("relaxation", "without an image collection an image predicate's evidence widens to texts: the "
                                 "atom is kept, read on what exists"),
    "question_visual": ("heuristic", "a picture word in the question (VISUAL) adds FALLBACK_VISUAL on one variable"),
    "owner_of_medium": ("heuristic", "a pattern on the relation (a picture \"of\" its tail) decides whose picture an "
                                     "image atom reads"),
    "numeric_name": ("heuristic", "a variable's name decides whether a value without letters may bind it"),
}
LOWERING_KIND_NAMES = ("equivalence", "relaxation", "heuristic")

TYPED_SYSTEM = ("You plan how a question is answered from many sources: you write its typed query graph, the facts "
                "a search must find one at a time. Reply with JSON only.")

_SCHEMA = ('{"constants": [name], "variables": [{"name": "?x", "set": true|false}], "atoms": [{"subject": term, '
           '"relation": phrase, "object": term, "evidence": "text"|"image"}], "filters": [{"variable": "?x", '
           '"condition": phrase, "evidence": "text"|"image"}], "answer": "?x"|null, "form": "entity"|"yes/no"|'
           '"choice"|"count"|"date"|"number"}')

_TYPED_RULES = """Write the typed query graph of the question at the end: the facts a search must find, one at a time, in texts, tables and images, to answer it. Reply with one JSON object on one line:
{schema}
- constants: the names the question gives (people, works, places, teams, events), written as in the question.
- variables: the unknowns, each named by a type and a number ("?person1", "?film2", "?date1"); "set" is true when the question asks for several members or ranges over several (the clubs of a league, the songs of an album).
- atoms: one fact each, between two terms (a term is a constant or a variable); the relation is a short phrase read from subject to object ("directed by", "born in", "played for in 2011").
- filters: a condition on one variable alone ("has a horse on its logo").
- answer: the variable whose value answers the question; null when the answer is yes/no, a choice between the items the question names, or what both sides of a comparison share.
- form: what the answer is.

Rules:
1. Give a variable to every intermediate entity that must be found before its own fact can be searched, and chain the atoms through it. A relation never hides an unknown: in "born in the country of his father" the father is a variable with an atom of its own, and a mother-in-law is the mother of the spouse (two atoms through ?person1).
2. One atom per fact a single source states. The cells of one table row are one fact: the away team's score in a cup round is (?team1 | score as away team in round 1 of the 2010 Cup | ?score1), with no variable for the match. No hop the question does not need: no "is a", type or category atom that restates what a variable is.
3. A description that names an entity is an atom to that entity as a constant (the winner of the 1998 FA Cup: (?team1 | won | 1998 FA Cup)). A description that names no entity ("the team whose logo has crossed swords", "a man holding a guitar") is a filter on its variable, never a constant.
4. Evidence is "image" only for what an image alone can verify: appearance, what a poster, logo, cover, flag, photo or map shows, text visible in it. Everything else, tables included, is "text". Never a variable for an image itself: what a picture shows goes on the entity it belongs to, as a filter ("its poster shows a red car") or as an atom read from that entity's image.
5. A date or year alone is never a constant: it goes into the relation it qualifies ("played for in 2011"). A named event with its year ("2004 Summer Olympics") is a constant.
6. A comparison gives each side its own variables and atoms. When the question asks which members meet several conditions, every condition goes on the one variable.
7. Every atom has one subject and one object, each a single constant or variable (never two joined by "and"); a fact that also involves a third term names it in the relation ((?person1 | married ?person2 in | ?date1)); a fact about one variable alone is a filter.
8. Never answer the question and never put a guessed value in a constant: every value the question does not give is a variable.

Examples:
{examples}"""


def _var(name: str, many: bool = False) -> dict[str, Any]:
    return {"name": name, "set": many}


def _atom(subject: str, relation: str, obj: str, evidence: str = "text") -> dict[str, str]:
    return {"subject": subject, "relation": relation, "object": obj, "evidence": evidence}


def _graph(constants: list[str], variables: list[dict[str, Any]], atoms: list[dict[str, str]], answer: str | None,
           form: str, filters: Sequence[tuple[str, str, str]] = ()) -> dict[str, Any]:
    return {"constants": constants, "variables": variables, "atoms": atoms,
            "filters": [{"variable": v, "condition": c, "evidence": ev} for v, c, ev in filters], "answer": answer,
            "form": form}


EXAMPLES: tuple[tuple[str, dict[str, Any]], ...] = (
    ('How many red stripes would one find if one were to look at the flag of the nation in which "Where Eagles '
     'Dare" was filmed?',
     _graph(["Where Eagles Dare"], [_var("?country1"), _var("?number1")],
            [_atom("Where Eagles Dare", "filmed in", "?country1"),
             _atom("?country1", "number of red stripes on its flag", "?number1", "image")], "?number1", "count")),
    ("What is the country of citizenship for the father of the River Song performer?",
     _graph(["River Song"], [_var("?person1"), _var("?person2"), _var("?country1")],
            [_atom("River Song", "performed by", "?person1"), _atom("?person1", "has father", "?person2"),
             _atom("?person2", "country of citizenship", "?country1")], "?country1", "entity")),
    ("What player was transferred from the team that has crossed swords on its logo to the Warrington Wolves in the "
     "2018 season?",
     _graph(["Warrington Wolves"], [_var("?team1"), _var("?person1")],
            [_atom("?person1", "transferred from", "?team1"),
             _atom("?person1", "transferred to in the 2018 season", "Warrington Wolves")], "?person1", "entity",
            [("?team1", "has crossed swords on its logo", "image")])),
    ("Which locations featuring mountains did Masahiko Harada win at in the 1997/98 season?",
     _graph(["Masahiko Harada"], [_var("?location1", True)],
            [_atom("Masahiko Harada", "won at in the 1997/98 season", "?location1")], "?location1", "entity",
            [("?location1", "features mountains", "image")])),
    ("Which film has the director who was born later, Complete Unknown or Dr. Goldfoot And The Girl Bombs?",
     _graph(["Complete Unknown", "Dr. Goldfoot And The Girl Bombs"],
            [_var("?person1"), _var("?date1"), _var("?person2"), _var("?date2")],
            [_atom("Complete Unknown", "directed by", "?person1"), _atom("?person1", "date of birth", "?date1"),
             _atom("Dr. Goldfoot And The Girl Bombs", "directed by", "?person2"),
             _atom("?person2", "date of birth", "?date2")], None, "choice")),
    ("Are Hanafuda and Okey both card games?",
     _graph(["Hanafuda", "Okey"], [_var("?type1"), _var("?type2")],
            [_atom("Hanafuda", "type of game", "?type1"), _atom("Okey", "type of game", "?type2")], None, "yes/no")),
)

_REPAIR_RULES = """A first search for the question at the end followed its typed query graph atom by atom; each atom shows what the search found. Write a corrected typed query graph for the question, as one JSON object on one line in the same language:
{schema}
Rules:
1. Values found so far are constants.
2. If an atom found nothing because its relation hides an intermediate entity, split it into two atoms joined by a new variable; if a constant is a description rather than a name, make it a filter on a new variable.
3. If a variable has no source of its own and only its neighbor can be found, merge its two atoms into one.
4. If the question asks for a property of a found entity that no atom covers, add that atom, ending in a new variable.
5. Leave out atoms that were found and are no longer needed. The answer is the variable whose value answers the question (null for a yes/no question or a choice between named items).
6. Evidence is "image" only for what an image alone can verify (appearance, what a poster, logo, cover, flag, photo or map shows, visible text); everything else is "text".

Example: for "What succeeded the company of the first British settlers in the country where Meyer Sound Laboratories is based?", where the first search found (Meyer Sound Laboratories | based in | America) and then nothing for (country1 = America | company of the first British settlers succeeded by | company1):
{example}"""

REPAIR_EXAMPLE = _graph(["America"], [_var("?company1"), _var("?company2")],
                        [_atom("?company1", "first British settlers in", "America"),
                         _atom("?company1", "succeeded by", "?company2")], "?company2", "entity")


def _dump(graph: dict[str, Any]) -> str:
    return json.dumps(graph, ensure_ascii=False)


TYPED_USER = _TYPED_RULES.format(schema=_SCHEMA, examples="\n\n".join(f"Question: {q}\n{_dump(g)}"
                                                                      for q, g in EXAMPLES))
TYPED_REPAIR_USER = _REPAIR_RULES.format(schema=_SCHEMA, example=_dump(REPAIR_EXAMPLE))


def typed_messages(question: str) -> list[dict[str, Any]]:
    return [{"role": "system", "content": TYPED_SYSTEM},
            {"role": "user", "content": f"{TYPED_USER}\n\nQuestion: {question}"}]


def typed_repair_messages(question: str, graph: Sequence[str], facts: Sequence[str], problem: str,
                          ) -> list[dict[str, Any]]:
    tail = (f"Question: {question}\n\nFirst graph:\n{chr(10).join(graph) or '(none)'}\n\nFacts found so far:\n"
            f"{chr(10).join(facts) or '(none)'}\n\nProblem: {problem}")
    return [{"role": "system", "content": TYPED_SYSTEM},
            {"role": "user", "content": f"{TYPED_REPAIR_USER}\n\n{tail}"}]


async def typed_graph(q: Question, ctx: QueryContext) -> tuple[Pattern, frozenset[tuple[str, str]]]:
    reply = await ctx.llm.chat(typed_messages(q.question), role="pattern", task="typed_graph", json_mode=True,
                               max_tokens=700, ledger=ctx.ledger, charge="pattern")
    return lower(reply.text, q.question)


def lower(text: str, question: str, note: Note | None = None) -> tuple[Pattern, frozenset[tuple[str, str]]]:
    pattern, pictured = translate(text, question, note)
    if not pattern.variables:
        return _fallback(question, "variables", pattern.raw), frozenset()
    return pattern, pictured


async def typed_repair(question: str, graph_lines: Sequence[str], facts: Sequence[str], problem: str,
                       ctx: QueryContext) -> Pattern:
    reply = await ctx.llm.chat(typed_repair_messages(question, graph_lines, facts, problem), role="pattern",
                               task="typed_repair", json_mode=True, max_tokens=600, ledger=ctx.ledger, charge="repair")
    return translate(reply.text, question)[0]


_COLON = re.compile(r'"(constants|variables|atoms|filters|answer|form|name|subject|relation|object|evidence|variable|'
                    r'condition) "')


class _Unusable(ValueError):
    pass


def _string(value: Any, where: str = "") -> str:
    if isinstance(value, bool) or not isinstance(value, str | int | float):
        raise _Unusable("json", f"{where or 'a term'}: {json.dumps(value)[:60]} is not a string")
    return _unquote(str(value))


def _items(graph: dict[str, Any], key: str, required: bool = False) -> list[Any]:
    value = graph.get(key)
    if value is None and not required:
        return []
    if not isinstance(value, list):
        raise _Unusable("json", f'"{key}" is {"missing" if value is None else "not a list"}')
    return value


def _parse(text: str) -> tuple[Any, bool]:
    graph = parse_json(text)
    # retry once with keys that lost their closing quote and colon ("subject "...)
    if graph is None and _COLON.search(text or ""):
        graph = parse_json(_COLON.sub(r'"\1": "', text))
        return graph, graph is not None
    return graph, False


def translate(text: str, question: str, note: Note | None = None) -> tuple[Pattern, frozenset[tuple[str, str]]]:
    graph, mended = _parse(text)
    try:
        pattern, pictured, sets = _translate(graph, question, note)
    except _Unusable as bad:
        raw = {"typed": graph if isinstance(graph, dict) else None}
        if bad.args[0] == "json":
            raw["reply"] = (text or "")[:2000]
        pattern, pictured = _fallback(question, bad.args[0], raw), frozenset()
    else:
        form = graph.get("form")
        pattern.raw.update(typed=graph, sets=sets, form=form if form in FORMS else None, pictured=sorted(pictured),
                           unbindable=executable(pattern, question))
    if mended:
        pattern.raw["mended"] = True
    return pattern, pictured


def _fallback(question: str, why: str, raw: dict[str, Any]) -> Pattern:
    pattern = prompts.parse_pattern("", question)
    typed = raw.get("typed")
    form = typed.get("form") if isinstance(typed, dict) else None
    pattern.raw.update({k: raw[k] for k in ("typed", "reply") if k in raw}, fallback=why, sets=[], pictured=[],
                       form=form if form in FORMS else None)
    pattern.raw["unbindable"] = executable(pattern, question)
    return pattern


def _note(code: str, message: str) -> dict[str, str]:
    return {"code": code, "message": message}


@dataclass
class _Reading:
    anchors: list[str]
    declared: list[tuple[str, bool]]
    nodes: dict[str, str]
    edges: list[Edge]
    unary: dict[str, list[str]]
    pictured: set[tuple[str, str]]
    answer: str | None
    edge_of: list[int | None]
    notes: list[dict[str, str]]


def _read(graph: Any, question: str) -> _Reading:
    if not isinstance(graph, dict):
        raise _Unusable("json", "the reply is no JSON object")
    notes: list[dict[str, str]] = []
    anchors = list(dict.fromkeys(c for c in (_string(c, f"constants[{i}]")
                                             for i, c in enumerate(_items(graph, "constants"))) if c))
    declared: list[tuple[str, bool]] = []
    for i, v in enumerate(_items(graph, "variables")):
        name, many = (v.get("name"), v.get("set")) if isinstance(v, dict) else (v, False)
        name = _string(name, f"variables[{i}]").lstrip("?").strip()
        if name:
            declared.append((f"?{name}", many is True or str(many).lower() == "true"))
    nodes = {prompts._key(a): a for a in anchors}
    for name, _ in declared:
        nodes.setdefault(prompts._key(name), name)

    def node(term: Any, where: str) -> str:
        t = _string(term, where)
        if not t:
            raise _Unusable("json", f"{where} is empty")
        k = prompts._key(t)
        if k in nodes:
            return nodes[k]
        if t.startswith("?") and t.lstrip("?").strip():
            nodes[k] = f"?{t.lstrip('?').strip()}"
            notes.append(_note("declared", f"{where}: {nodes[k]} is used undeclared: a variable"))
            return nodes[k]
        if not t.startswith("?") and prompts._words(t) in prompts._words(question):
            anchors.append(t)
            nodes[k] = t
            notes.append(_note("declared", f'{where}: "{t}" is used undeclared: a name the question gives'))
            return t
        raise _Unusable("terms", f'{where}: "{t}" is no constant, no variable and no name the question gives')

    edges: list[Edge] = []
    seen: dict[tuple[str, str, str], Edge] = {}
    unary: dict[str, list[str]] = {}
    pictured: set[tuple[str, str]] = set()
    edge_of: list[int | None] = []
    for i, a in enumerate(_items(graph, "atoms", required=True)):
        where = f"atoms[{i}]"
        if not isinstance(a, dict):
            raise _Unusable("json", f"{where} is not an object")
        sides = [(side, x) for side, x in (("subject", a.get("subject")), ("object", a.get("object")))
                 if not (x is None or isinstance(x, str) and not x.strip())]
        if not sides:
            raise _Unusable("json", f"{where} has no subject and no object")
        h, t = node(sides[0][1], f"{where}.{sides[0][0]}"), node(sides[-1][1], f"{where}.{sides[-1][0]}")
        rel = _string(a.get("relation", ""), f"{where}.relation")
        image = str(a.get("evidence", "")).strip().lower() == "image"
        edge_of.append(None)
        if not rel:
            notes.append(_note("dropped", f"{where} has no relation: left out"))
            continue
        # one end given, or both ends equal: a fact about one term, a filter if it is a variable
        if h == t:
            if is_var(h):
                if rel not in unary.get(h, []):
                    unary.setdefault(h, []).append(rel)
                if image:
                    pictured.add((h, rel))
            notes.append(_note("filter", f"{where} is a fact about {h} alone: " + (
                "a filter" if is_var(h) else "nothing to find, left out")))
            continue
        e = seen.get((h, rel.lower(), t))
        if e is None:
            e = seen[(h, rel.lower(), t)] = Edge(len(edges), h, rel, t)
            edges.append(e)
        else:
            notes.append(_note("duplicate", f"{where} repeats ({h} | {rel} | {t}): merged"))
        edge_of[-1] = e.id
        if image:
            e.modality = "image"
    for i, f in enumerate(_items(graph, "filters")):
        where = f"filters[{i}]"
        if not isinstance(f, dict):
            raise _Unusable("json", f"{where} is not an object")
        x, cond = node(f.get("variable"), f"{where}.variable"), _string(f.get("condition", ""), f"{where}.condition")
        if not is_var(x) or not cond:
            notes.append(_note("dropped", f"{where} " + ("has no condition" if not cond else
                                                         f"is on the constant {x}: nothing to find") + ", left out"))
            continue
        if cond not in unary.get(x, []):
            unary.setdefault(x, []).append(cond)
        else:
            notes.append(_note("duplicate", f"{where} repeats a condition on {x}: merged"))
        if str(f.get("evidence", "")).strip().lower() == "image":
            pictured.add((x, cond))
    answer_term = graph.get("answer")
    named = node(answer_term, "answer") if isinstance(answer_term, str) and answer_term.strip().lower() not in (
        "", "null", "none") else None
    return _Reading(anchors, declared, nodes, edges, unary, pictured, named, edge_of, notes)


def _translate(graph: Any, question: str, note: Note | None = None,
               ) -> tuple[Pattern, set[tuple[str, str]], list[str]]:
    r = _read(graph, question)
    anchors, edges, unary, pictured, named, nodes, declared = (r.anchors, r.edges, r.unary, r.pictured, r.answer,
                                                               r.nodes, r.declared)
    if not edges:
        loose = [c for conds in unary.values() for c in conds]
        pattern = Pattern(anchors=[], variables=["?answer"], edges=[Edge(0, question, "answer", "?answer")],
                          answer="?answer", raw={"fallback": "atoms"}, conditions={"?answer": loose} if loose else {})
        return pattern, {("?answer", c) for _, c in pictured}, []
    variables = list(dict.fromkeys(n for e in edges for n in (e.head, e.tail) if is_var(n)))
    answer = named if named in variables else None
    if answer is None and named is not None and named in unary:
        variables.append(named)
        answer = named
    inferred = answer is None
    if inferred:
        tails_only = [v for v in variables if all(e.head != v for e in edges)]
        answer = tails_only[-1] if tails_only else (variables[-1] if variables else None)
    before, marked = {id(e): e.relation for e in edges}, [e for e in edges if e.modality == "image"]
    was, unknowns = [(e.head, e.relation, e.tail) for e in edges], list(variables)
    constants = prompts._fold_concepts(edges, variables, answer, question)
    if note is not None:
        for v in (v for v in unknowns if v not in variables):
            name = v[1:].strip()
            note("concept", f"{v} is the constant {name}" if name in constants else f"{v} contracted",
                 [t for t in was if v in (t[0], t[2])])
    # an image atom folded into a composed relation passes its image evidence on
    for gone in (e for e in marked if all(e is not f for f in edges)):
        for f in edges:
            if before.get(id(f)) != f.relation and f.tail == gone.tail and f.relation.endswith(f" {gone.relation}"):
                f.modality = "image"
    anchors = [*anchors, *(n for n in constants if n not in anchors)]
    pattern = Pattern(anchors=[a for a in anchors if any(a in (e.head, e.tail) for e in edges)], variables=variables,
                      edges=edges, answer=answer, raw={"answer_inferred": True} if inferred else {},
                      conditions={v: conds for v, conds in unary.items() if v in variables})
    hint_modalities(fold_media(pattern, note), note)
    kept = set()
    for x, c in pictured:
        if c in pattern.conditions.get(x, []):
            kept.add((x, c))
            continue
        kept |= {(y, t) for y, conds in pattern.conditions.items() for t in conds
                 if t.endswith(f" {c}") and re.fullmatch(_MEDIUM, t[: -len(c) - 1], re.IGNORECASE)}
    sets = [nodes[prompts._key(name)] for name, many in declared if many]
    return pattern, kept, [x for x in dict.fromkeys(sets) if x in pattern.variables]


def awaited(pattern: Pattern) -> list[str]:
    from fedquest.methods.fedquest import links

    tied, todo = ({pattern.answer} if pattern.answer in pattern.variables else set()), [pattern.answer]
    pairs = links(pattern)
    while todo:
        x = todo.pop()
        for a, b in pairs:
            for y in ((b,) if a == x else (a,) if b == x else ()):
                if y not in tied:
                    tied.add(y)
                    todo.append(y)
    if pattern.answer is not None and not pattern.raw.get("answer_inferred") and tied >= set(pattern.variables):
        return [pattern.answer]
    return list(pattern.variables)


def executable(pattern: Pattern, question: str, waits: Sequence[str] | None = None) -> list[str]:
    from fedquest.methods.fedquest import conditions

    bound = _reach(conditions(pattern, question, pictures=False), pattern.variables)
    return [x for x in (awaited(pattern) if waits is None else waits) if x not in bound]


def _reach(conds: Iterable[Cond], variables: Sequence[str]) -> set[str]:
    held: dict[str, list[Cond]] = defaultdict(list)
    for c in conds:
        held[c.var].append(c)
    bound: set[str] = set()
    changed = True
    while changed:
        changed = False
        for x in variables:
            if x not in bound and any((c.edge is None or not is_var(c.other or "") or c.other in bound)
                                      and all(d in bound for d in c.deps) for c in held[x]):
                bound.add(x)
                changed = True
    return bound


Triple = tuple[str, str, str]


def triple(head: str, relation: str, tail: str) -> Triple:
    return " ".join(head.split()), " ".join(relation.split()).lower(), " ".join(tail.split())


@dataclass
class Lowering:
    applied: list[tuple[str, str]] = field(default_factory=list)
    atoms: dict[Triple, str] = field(default_factory=dict)

    def __call__(self, rule: str, target: str, atoms: Iterable[tuple[str, str, str]] = ()) -> None:
        self.applied.append((rule, target))
        for a in atoms:
            self.atoms.setdefault(triple(*a), rule)


def prune(pattern: Pattern, question: str = "", typed: bool = False, note: Note | None = None) -> Pattern:
    changed = True
    while changed:
        changed = False
        for x in list(pattern.variables):
            on = pattern.incident(x)
            # only a leaf is pruned: not the answer, no filter, no constant neighbor, one atom at most
            if x == pattern.answer or pattern.conditions.get(x) or any(not is_var(e.other(x)) for e in on):
                continue
            if len(on) > 1:
                continue
            name = prompts.described(x, question) if on and not _TYPE_EDGE.match(on[0].relation.strip()) else None
            if typed and not name and on and (
                    any(x in uses_of(e.relation, pattern.variables) for e in pattern.edges if e is not on[0])
                    or (pattern.raw.get("answer_inferred") and all(e.head != x for e in pattern.edges)
                        and not _TYPE_EDGE.match(on[0].relation.strip()))):
                continue
            atoms = [(e.head, e.relation, e.tail) for e in on]
            if name:
                e = on[0]
                e.head, e.tail = (name if e.head == x else e.head), (name if e.tail == x else e.tail)
                if name not in pattern.anchors:
                    pattern.anchors.append(name)
                if note is not None:
                    note("described", f"{x} is the constant {name}", atoms)
            else:
                pattern.edges = [e for e in pattern.edges if e not in on]
                if note is not None:
                    rule = "type_edge" if on and _TYPE_EDGE.match(on[0].relation.strip()) else "leaf"
                    note(rule, f"{x} with ({' | '.join(atoms[0])})" if atoms else f"{x} (on no atom)", atoms)
            pattern.variables.remove(x)
            changed = True
    return pattern


@dataclass
class Marks:
    pictures: bool
    image: frozenset[int]
    pictured: frozenset[tuple[str, str]]
    question: str | None
    visual: bool
    holders: dict[int, str]
    uses: dict[int, tuple[str, ...]]
    owners: dict[int, str]
    numeric: frozenset[str]
    applied: list[tuple[str, str]] = field(default_factory=list)


def marks(pattern: Pattern, question: str, pictured: Iterable[tuple[str, str]] = (), pictures: bool = True) -> Marks:
    low = Lowering()
    atoms = _adornments(pattern, question)[0]
    holders = {i: a["out"] for i, a in atoms.items() if a["access"] == "join"}
    image = frozenset(i for i, e in enumerate(pattern.edges) if pictures and e.modality == "image")
    label, filters = set(pictured), set()
    for x, texts in (pattern.conditions.items() if pictures else ()):
        for t in texts:
            if (x, t) in label:
                filters.add((x, t))
            elif VISUAL.search(t):
                filters.add((x, t))
                low("visual_lexicon", f"filter {x}: {t}")
    visual = VISUAL.search(question) is not None
    held = [(i, end) for i, e in enumerate(pattern.edges) for end in (e.head, e.tail)
            if is_var(end) and e.other(end) != end and holders.get(i, end) == end]
    target = None
    if pictures and visual and not filters and not any(i in image for i, _ in held) and pattern.variables:
        anchored = [end for i, end in held if not is_var(pattern.edges[i].other(end)) and end != pattern.answer]
        target = anchored[0] if anchored else pattern.answer if len(pattern.variables) == 1 else None
        if target:
            low("question_visual", target)
    owners: dict[int, str] = {}
    for i, e in enumerate(pattern.edges):
        of = _OF_OWNER.search(e.relation.strip()) is not None
        owners[i] = e.tail if of else e.head
        if of and i in image:
            low("owner_of_medium", f"atom ({e.head} | {e.relation} | {e.tail}): {e.tail}'s picture")
    numeric = frozenset(x for x in pattern.variables if any(h in x.lower() for h in _NUMERIC_NAMES))
    answered = [pattern.answer] if pattern.answer is not None and not pattern.raw.get("answer_inferred") else list(
        pattern.variables)
    for x in (x for x in pattern.variables if x in numeric and x not in answered):
        low("numeric_name", x)
    return Marks(pictures, image, frozenset(filters), target or None, visual, holders,
                 {i: tuple(a["uses"]) for i, a in atoms.items()}, owners, numeric, low.applied)


def admitted(variable: str, value: str, lowered: Marks) -> bool:
    return bool(_LETTER.search(value)) or variable in lowered.numeric


def owner(e: Edge) -> str:
    return e.tail if _OF_OWNER.search(e.relation.strip()) else e.head


def relabeled(compiled: CompiledQuery, pattern: Pattern, pictured: Iterable[tuple[str, str]], question: str,
              ) -> CompiledQuery:
    pictured = frozenset(pictured) if compiled.pictures else frozenset()
    return dataclasses.replace(compiled, pattern=pattern, pictured=pictured,
                               marks=marks(pattern, question, pictured, compiled.pictures))


SCHEMA_FILE = "typed_query.schema.json"
_JSON_TYPES: dict[str, type] = {"object": dict, "array": list, "string": str, "boolean": bool, "null": type(None)}
_PLACEHOLDER = re.compile(r"[A-Za-z_]+\d+")
_REFERENCE = re.compile(r"\?([^\W\d]\w*)")


@cache
def schema() -> dict[str, Any]:
    path = configs_dir() / SCHEMA_FILE
    return json.loads((path if path.exists() else CONFIGS / SCHEMA_FILE).read_text(encoding="utf-8"))


def validate(graph: Any) -> list[str]:
    out: list[str] = []
    _check(graph, schema(), "$", out)
    return out


_LABELS = {"name": 'a name (no leading "?")', "variable": 'a variable ("?x")', "phrase": "a phrase", "term": "a term"}


def _label(node: dict[str, Any]) -> str:
    if "$ref" in node:
        return _LABELS.get(node["$ref"].rsplit("/", 1)[-1], "a value")
    return "null" if node.get("type") == "null" else f"a {node.get('type', 'value')}"


def _check(value: Any, node: dict[str, Any], where: str, out: list[str], label: str = "a value") -> None:
    while "$ref" in node:
        label = _label(node)
        node = schema()["$defs"][node["$ref"].rsplit("/", 1)[-1]]
    shown = json.dumps(value, ensure_ascii=False)[:60]
    if "anyOf" in node:
        if all(_failed(value, n) for n in node["anyOf"]):
            out.append(f"{where}: {shown} is not " + " nor ".join(_label(n) for n in node["anyOf"]))
        return
    kind = node.get("type")
    if kind is not None and (not isinstance(value, _JSON_TYPES[kind]) or kind != "boolean" and isinstance(value, bool)):
        out.append(f"{where}: {shown} is not {'an' if kind[0] in 'aeiou' else 'a'} {kind}")
        return
    if "enum" in node and value not in node["enum"]:
        out.append(f"{where}: {shown} is not one of " + ", ".join(json.dumps(v) for v in node["enum"]))
    if "pattern" in node and isinstance(value, str) and not re.search(node["pattern"], value):
        out.append(f"{where}: {shown} is not {label}")
    if isinstance(value, dict):
        props = node.get("properties", {})
        out += [f"{where}: no {k!r}" for k in node.get("required", ()) if k not in value]
        for k, v in value.items():
            if k in props:
                _check(v, props[k], f"{where}.{k}", out)
            elif node.get("additionalProperties") is False:
                out.append(f"{where}: unknown key {k!r}")
    if isinstance(value, list) and "items" in node:
        for i, v in enumerate(value):
            _check(v, node["items"], f"{where}[{i}]", out)


def _failed(value: Any, node: dict[str, Any]) -> bool:
    out: list[str] = []
    _check(value, node, "", out)
    return bool(out)


def uses_of(relation: str, variables: Iterable[str]) -> list[str]:
    return [x for x in variables if _PLACEHOLDER.fullmatch(x.lstrip("?"))
            and re.search(rf"\b{re.escape(x.lstrip('?'))}\b", relation)]


def normalize(graph: Any, question: str) -> tuple[dict[str, Any], list[dict[str, str]]]:
    return _normal(_read(graph, question), graph)[1:]


def _normal(r: _Reading, graph: dict[str, Any]) -> tuple[_Reading, dict[str, Any], list[dict[str, str]]]:
    notes = list(r.notes)
    on = {n for e in r.edges for n in (e.head, e.tail)}
    variables = [x for x in dict.fromkeys([*(r.nodes[prompts._key(n)] for n, _ in r.declared), *r.nodes.values()])
                 if is_var(x) and (x in on or x in r.unary)]
    unused = [n for n, _ in r.declared if r.nodes[prompts._key(n)] not in variables]
    notes += [_note("unused", f"{n} is on no atom and no filter: left out") for n in dict.fromkeys(unused)]
    sets = {r.nodes[prompts._key(n)] for n, many in r.declared if many}
    answer = r.answer if r.answer in variables else None
    if r.answer is not None and answer is None:
        notes.append(_note("answer", f"the answer {r.answer} is no variable of the graph: inferred"))
    form = graph.get("form") if graph.get("form") in FORMS else None
    if graph.get("form") is not None and form is None:
        notes.append(_note("form", f"the form {json.dumps(graph.get('form'))[:40]} is none of FORMS: null"))

    def ws(s: str) -> str:
        return " ".join(s.split())

    atoms: dict[tuple[str, str, str], dict[str, Any]] = {}
    for e in r.edges:
        uses = [y for y in uses_of(e.relation, variables) if y not in (e.head, e.tail)]
        atom = {"subject": ws(e.head), "relation": ws(e.relation), "object": ws(e.tail),
                "evidence": "image" if e.modality == "image" else "text", **({"uses": uses} if uses else {})}
        same = atoms.setdefault((atom["subject"], atom["relation"].lower(), atom["object"]), atom)
        if same is not atom:
            notes.append(_note("duplicate", f'({atom["subject"]} | {atom["relation"]} | {atom["object"]}) is written '
                                            "twice: merged"))
            same["evidence"] = "image" if "image" in (same["evidence"], atom["evidence"]) else "text"
    ir = {"constants": [ws(a) for a in dict.fromkeys(a for a in r.anchors if a in on)],
          "variables": [{"name": ws(x), "set": x in sets} for x in variables], "atoms": list(atoms.values()),
          "filters": [{"variable": ws(x), "condition": ws(c), "evidence": "image" if (x, c) in r.pictured else "text"}
                      for x, conds in r.unary.items() for c in conds],
          "answer": answer and ws(answer), "form": form}
    return r, ir, notes


def _hidden(graph: dict[str, Any], r: _Reading, ir: dict[str, Any]) -> list[dict[str, str]]:
    out = []
    variables = [v["name"] for v in ir["variables"]]
    for a in ir["atoms"]:
        shown = f'({a["subject"]} | {a["relation"]} | {a["object"]})'
        for token in dict.fromkeys(_REFERENCE.findall(a["relation"])):
            x = r.nodes.get(prompts._key(token))
            if x not in variables:
                out.append(_note("hidden", f"the relation of {shown} names ?{token}, which no atom finds: give "
                                           f"?{token} an atom of its own (a relation never hides an unknown)"))
            elif x not in (a["subject"], a["object"], *a.get("uses", ())):
                out.append(_note("hidden", f'the relation of {shown} names {x}, but only a variable named by a type '
                                           'and a number ("?person2") can be named inside a relation'))
    for f in ir["filters"]:
        for token in dict.fromkeys(_REFERENCE.findall(f["condition"])):
            out.append(_note("hidden", f'the filter on {f["variable"]} ("{f["condition"]}") names ?{token}: a fact '
                                       "between two terms is an atom"))
    for i, a in enumerate(graph.get("atoms") or []):
        given = a.get("uses") if isinstance(a, dict) else None
        j = r.edge_of[i] if i < len(r.edge_of) else None
        if not isinstance(given, list) or j is None:
            continue
        e = r.edges[j]
        derived = [y for y in uses_of(e.relation, variables) if y not in (e.head, e.tail)]
        for u in given:
            if r.nodes.get(prompts._key(str(u))) not in derived:
                out.append(_note("hidden", f'atoms[{i}] uses {json.dumps(u)[:40]}, which its relation "{e.relation}" '
                                           "does not name"))
    return out


NEAR = 85


def grounding(constant: str, question: str, known: Sequence[str] = ()) -> str:
    ne = prompts._norm_text(norm_entity(constant)).strip()
    nq = prompts._norm_text(question)
    if not ne:
        return "ungrounded"
    if f" {ne} " in nq:
        return "span"
    if any(f" {ne} " in prompts._norm_text(k) for k in known):
        return "known"
    words = [w for w in ne.split() if len(w) >= 3 or w.isdigit()]
    if words and all(f" {w} " in nq for w in words):
        return "near"
    qw, n = nq.split(), len(ne.split())
    for size in sorted({max(1, n - 1), n, n + 1}):
        if any(fuzz.ratio(" ".join(qw[i:i + size]), ne) >= NEAR for i in range(len(qw) - size + 1)):
            return "near"
    return "ungrounded"


ERRORS = ("json", "schema", "terms", "hidden", "ungrounded", "unbindable", "empty")
VALUE_FORMS = ("entity", "count", "date", "number")
EVERY = ("yes/no", "choice")


@dataclass
class CompiledQuery:
    pattern: Pattern
    pictured: frozenset[tuple[str, str]]
    ir: dict[str, Any] | None
    errors: list[dict[str, str]]
    warnings: list[dict[str, str]]
    grounded: dict[str, str]
    atoms: dict[int, dict[str, Any]]
    order: list[str]
    deps: dict[str, list[str]]
    loose: bool
    bindable: list[str]
    awaited: list[str]
    unbindable: list[str]
    modalities: list[str] | None
    pictures: bool
    lowered: list[str]
    marks: Marks
    lowering: Lowering = field(default_factory=Lowering)
    reply: str | None = None
    source: tuple[Pattern, frozenset[tuple[str, str]]] | None = None
    repair: bool = False
    first: dict[str, Any] | None = None
    fixed: bool = False
    fallback: bool = False

    def ungrounded(self) -> list[str]:
        return [a for a, g in self.grounded.items() if g == "ungrounded"]

    def counts(self) -> dict[str, int]:
        first = self.first or {"errors": self.errors, "ungrounded": self.ungrounded()}
        return {"compile_errors": len(first["errors"]), "compile_fixed": int(self.fixed),
                "compile_fallback": int(self.fallback), "ungrounded": len(first["ungrounded"])}

    def applied(self) -> list[tuple[str, str]]:
        return [*self.lowering.applied, *self.marks.applied]

    def lowered_counts(self) -> dict[str, int]:
        return {f"lowered_{rule}": n for rule, n in Counter(rule for rule, _ in self.applied()).items()}

    def record(self, violations: Sequence[dict[str, str]] = ()) -> dict[str, Any]:
        out: dict[str, Any] = {"version": TYPED_VERSION, **self.counts(), "errors": self.errors,
                               "warnings": sorted({w["code"] for w in self.warnings}),
                               "grounded": {a: g for a, g in self.grounded.items() if g != "span"},
                               "access": [f'{a["adornment"]} {a["access"]}' if (a := self.atoms.get(i)) else None
                                          for i in range(len(self.pattern.edges))],
                               "order": self.order, "awaited": self.awaited, "unbindable": self.unbindable,
                               "pictures": self.pictures, "lowered": self.lowered,
                               "rules": [list(a) for a in self.applied()], "violations": list(violations)}
        if self.repair:
            out["repair"] = True
        if self.first is not None:
            out["first"] = self.first
        return out


def compile_pattern(pattern: Pattern, question: str, modalities: Iterable[str] | None = None,
                    known: Sequence[str] = (), *, pictured: Iterable[tuple[str, str]] = (), form: str | None = None,
                    ir: dict[str, Any] | None = None, problems: Sequence[dict[str, str]] = (),
                    reply: str | None = None, source: tuple[Pattern, frozenset[tuple[str, str]]] | None = None,
                    repair: bool = False, lowering: Lowering | None = None) -> CompiledQuery:
    errors = [p for p in problems if p["code"] in ERRORS]
    warnings = [p for p in problems if p["code"] not in ERRORS]
    pictured = frozenset(pictured)
    why = pattern.raw.get("fallback")
    if why is True:
        if not pattern.raw.get("records"):
            errors.append(_note("json", "the reply holds no record of the graph's language"))
        elif not pattern.conditions:
            errors.append(_note("empty", "the graph has no relation between two terms: nothing to search for"))
    if why == "variables" and form in VALUE_FORMS:
        errors.append(_note("empty", f"no atom has an unknown, but the answer is {'an' if form == 'entity' else 'a'} "
                                     f"{form}: the answer must be a variable on an atom"))
    if why == "atoms" and not pattern.conditions:
        errors.append(_note("empty", "the graph has no atom between two terms and no filter: nothing to search for"))
    grounded = {a: grounding(a, question, known) for a in pattern.anchors if a != question}
    errors += [_note("ungrounded", f'the constant "{a}" is no name the question gives: never put a guessed value in a '
                                   "constant; make it a variable with the atom that finds it")
               for a, g in grounded.items() if g == "ungrounded"]
    warnings += [_note("near", f'the constant "{a}" is spelled otherwise in the question')
                 for a, g in grounded.items() if g == "near"]
    analysis = _analysis(pattern, question, form)
    mods = None if modalities is None else sorted(set(modalities))
    pictures = mods is None or "image" in mods
    lowered = [] if pictures else _lowered(pattern, pictured, question)
    low = lowering if lowering is not None else Lowering()
    for item in lowered:
        low("capability", item)
    if lowered:
        warnings.append(_note("lowered", f"no site holds an image: {len(lowered)} image predicate(s) read on texts"))
    unbindable = executable(pattern, question, analysis["awaited"])
    if unbindable:
        errors.append(_note("unbindable", _unreached(unbindable, pattern, analysis["atoms"])))
    if analysis["loose"]:
        warnings.append(_note("loose", "relations name each other's unknowns: read without waiting for them"))
    if analysis["closed"]:
        warnings.append(_note("closed", f"{len(analysis['closed'])} atom(s) between two constants: no search finds "
                                        "them (the answer's reader checks them)"))
    if not analysis["connected"] and pattern.answer is not None and not pattern.raw.get("answer_inferred"):
        warnings.append(_note("disconnected", "the answer is not linked to every unknown: every unknown awaited"))
    pictured = pictured if pictures else frozenset()
    return CompiledQuery(pattern, pictured, ir, errors, warnings, grounded, analysis["atoms"], analysis["order"],
                         analysis["deps"], analysis["loose"], analysis["bindable"], analysis["awaited"], unbindable,
                         mods, pictures, lowered, marks(pattern, question, pictured, pictures), low, reply=reply,
                         source=source, repair=repair)


def _unreached(xs: list[str], p: Pattern, atoms: dict[int, dict[str, Any]]) -> str:
    away = [f"({p.edges[i].head} | {p.edges[i].relation} | {p.edges[i].tail}) finds {a['out']} from {a['key']}"
            for i, a in atoms.items() if a["access"] == "join" and a["key"] in xs]
    answer = f"{p.answer}{' as the graph names none' if p.raw.get('answer_inferred') else ''}"
    why = f" (an atom between two unknowns is read toward the answer, {answer}: {'; '.join(away)})" if away else ""
    return (f"{', '.join(xs)} cannot be reached: no atom finds {'it' if len(xs) == 1 else 'them'} from a constant or "
            f"from a variable found before{why}; every unknown the answer needs is found from the question's "
            "constants, one atom at a time")


_ON_CONSTANT = re.compile(r"^\$\.(filters\[\d+\])\.variable: (.+) is not a variable")


def _explained(violation: str, dropped: set[str]) -> str:
    if (m := _ON_CONSTANT.match(violation)) and m[1] in dropped:
        return (f"{m[1]} is a condition on {m[2]}, a constant: what the question says about a constant needs no "
                'search (leave the filter out); a filter is a condition on one unknown ("?x")')
    return violation


def _lowered(pattern: Pattern, pictured: frozenset[tuple[str, str]], question: str) -> list[str]:
    out = []
    for i, e in enumerate(pattern.edges):
        if e.modality == "image":
            e.modality = "text"
            out.append(f"atom {i} ({e.head} | {e.relation} | {e.tail})")
    out += [f"filter {x}: {t}" for x, conds in pattern.conditions.items() for t in conds
            if VISUAL.search(t) or (x, t) in pictured]
    if not out and pattern.variables and VISUAL.search(question):
        out.append("the question's picture condition")
    return out


class _Cycle(Exception):
    pass


def _levels(deps: dict[str, list[str]]) -> dict[str, int] | None:
    level: dict[str, int] = {}
    active: set[str] = set()

    def visit(x: str) -> int:
        if x in level:
            return level[x]
        if x in active:
            raise _Cycle
        active.add(x)
        level[x] = 1 + max((visit(y) for y in deps.get(x, ())), default=-1)
        active.discard(x)
        return level[x]

    try:
        for x in deps:
            visit(x)
    except _Cycle:
        return None
    return level


def _analysis(p: Pattern, question: str, form: str | None) -> dict[str, Any]:
    variables = list(p.variables)
    index = {x: i for i, x in enumerate(variables)}
    atoms, closed, links = _adornments(p, question)
    full: dict[str, list[str]] = {x: [] for x in variables}
    keyed: dict[str, list[str]] = {x: [] for x in variables}
    for a in atoms.values():
        inputs = [a["key"]] if a["access"] == "join" else []
        full.setdefault(a["out"], []).extend(y for y in dict.fromkeys([*inputs, *a["uses"]]) if y not in full[a["out"]])
        keyed.setdefault(a["out"], []).extend(y for y in inputs if y not in keyed[a["out"]])
    # binding order from join keys and relation uses; on a cycle, from join keys alone (loose)
    levels = _levels(full)
    loose = levels is None
    if levels is None:
        levels = _levels(keyed) or {}
    order = sorted(levels, key=lambda x: (levels[x], index.get(x, len(index))))
    unary = {x for x, conds in p.conditions.items() if conds}
    bound: set[str] = set()
    changed = True
    while changed:
        changed = False
        for x in variables:
            if x not in bound and (x in unary or any(a["out"] == x and (a["access"] != "join" or a["key"] in bound)
                                                     and all(y in bound for y in a["uses"]) for a in atoms.values())):
                bound.add(x)
                changed = True
    tied, todo = ({p.answer} if p.answer in index else set()), [p.answer]
    while todo:
        x = todo.pop()
        for a, b in links:
            for y in ((b,) if a == x else (a,) if b == x else ()):
                if y not in tied:
                    tied.add(y)
                    todo.append(y)
    connected = tied >= set(variables)
    if form not in EVERY and p.answer is not None and not p.raw.get("answer_inferred") and connected:
        waits = [p.answer]
    else:
        waits = list(variables)
    return {"atoms": atoms, "order": order, "deps": {x: full.get(x, []) for x in order}, "loose": loose,
            "bindable": [x for x in variables if x in bound], "awaited": waits, "connected": connected,
            "closed": closed}


def _adornments(p: Pattern, question: str) -> tuple[dict[int, dict[str, Any]], list[int], list[tuple[str, str]]]:
    variables = list(p.variables)
    index = {x: i for i, x in enumerate(variables)}
    uses = [[y for y in uses_of(e.relation, variables) if y not in (e.head, e.tail)] for e in p.edges]
    links: list[tuple[str, str]] = []
    for e in p.edges:
        ends = [n for n in (e.head, e.tail) if is_var(n)]
        if len(ends) == 2 and ends[0] != ends[1]:
            links.append((ends[0], ends[1]))
        links += [(end, y) for end in ends for y in uses_of(e.relation, variables) if y != end]
    depth: dict[str, int] = {}
    for root in [*([p.answer] if p.answer in index else []), *reversed(variables)]:
        if root in depth:
            continue
        depth[root], queue = 0, deque([root])
        while queue:
            x = queue.popleft()
            for a, b in links:
                y = b if a == x else a if b == x else None
                if y is not None and y not in depth:
                    depth[y] = depth[x] + 1
                    queue.append(y)
    atoms: dict[int, dict[str, Any]] = {}
    closed = []
    for i, e in enumerate(p.edges):
        ends = [n for n in (e.head, e.tail) if is_var(n)]
        if not ends:
            closed.append(i)
            continue
        if e.head == e.tail:
            continue
        # an atom between two unknowns binds the end nearer the answer, keyed by the other
        if len(ends) == 2:
            out = min(ends, key=lambda x: (depth.get(x, 0), -index.get(x, 0)))
            access = "join"
        else:
            out = ends[0]
            access = "question" if (e.tail if out == e.head else e.head) == question else "constant"
        atoms[i] = {"out": out, "key": e.tail if out == e.head else e.head, "access": access,
                    "adornment": "fb" if out == e.head else "bf", "uses": uses[i]}
    return atoms, closed, links


def compile_reply(text: str, question: str, modalities: Iterable[str] | None = None, known: Sequence[str] = (),
                  repair: bool = False) -> CompiledQuery:
    graph, mended = _parse(text)
    low = Lowering()
    source = translate(text, question, low) if repair else lower(text, question, low)
    pattern = prune(copy.deepcopy(source[0]), question, typed=True, note=low)
    problems = [_note("mended", "a key's closing quote and colon were missing: mended")] if mended else []
    ir = None
    violations = validate(graph) if graph is not None else []
    try:
        r, ir, notes = _normal(_read(graph, question), graph)
    except _Unusable as bad:
        detail = bad.args[1] if len(bad.args) > 1 else "no JSON object"
        if bad.args[0] == "json":
            problems.append(_note("json", "the reply is not one JSON object in the schema: " + "; ".join(
                dict.fromkeys([detail, *violations[:3]]))))
        else:
            problems.append(_note("terms", f"{detail}: every value the question does not give is a variable "
                                           '("?x"), and a fact about one variable alone is a filter'))
            problems += [_note("schema", v) for v in violations]
    else:
        dropped = {n["message"].split(" ", 1)[0] for n in notes
                   if n["code"] == "dropped" and " is on the constant " in n["message"]}
        problems += [*notes, *(_note("schema", _explained(v, dropped)) for v in violations), *_hidden(graph, r, ir)]
    form = graph.get("form") if isinstance(graph, dict) and graph.get("form") in FORMS else None
    return compile_pattern(pattern, question, modalities, known, pictured=source[1], form=form, ir=ir,
                           problems=problems, reply=text, source=source, repair=repair, lowering=low)


def typed_of(pattern: Pattern, pictured: Iterable[tuple[str, str]] = (), form: str | None = None,
             ) -> dict[str, Any] | None:
    if pattern.raw.get("fallback"):
        return None
    pictured = set(pictured)
    sets = set(pattern.raw.get("sets", ()))
    return {"constants": list(pattern.anchors),
            "variables": [{"name": x, "set": x in sets} for x in pattern.variables],
            "atoms": [{"subject": e.head, "relation": e.relation, "object": e.tail,
                       "evidence": "image" if e.modality == "image" else "text",
                       **({"uses": u} if (u := [y for y in uses_of(e.relation, pattern.variables)
                                                if y not in (e.head, e.tail)]) else {})} for e in pattern.edges],
            "filters": [{"variable": x, "condition": t, "evidence": "image" if (x, t) in pictured else "text"}
                        for x, conds in pattern.conditions.items() for t in conds],
            "answer": None if pattern.raw.get("answer_inferred") else pattern.answer, "form": form}


async def catalog(ctx: QueryContext) -> list[str] | None:
    if ctx.fed is None:
        return None
    return sorted({c.modality for si in (await ctx.fed.info()).values() for c in si.collections})


async def compile_question(q: Question, ctx: QueryContext, modalities: Iterable[str] | None = None,
                           fix: bool = False) -> CompiledQuery:
    mods = await catalog(ctx) if modalities is None else list(modalities)
    reply = await ctx.llm.chat(typed_messages(q.question), role="pattern", task="typed_graph", json_mode=True,
                               max_tokens=700, ledger=ctx.ledger, charge="pattern")
    compiled = compile_reply(reply.text, q.question, mods)
    return await fixed(q.question, compiled, ctx, mods) if fix and compiled.errors else compiled


async def compile_repair(question: str, graph_lines: Sequence[str], facts: Sequence[str], problem: str,
                         ctx: QueryContext) -> CompiledQuery:
    mods = await catalog(ctx)
    reply = await ctx.llm.chat(typed_repair_messages(question, graph_lines, facts, problem), role="pattern",
                               task="typed_repair", json_mode=True, max_tokens=600, ledger=ctx.ledger, charge="repair")
    return compile_reply(reply.text, question, mods, known=[*graph_lines, *facts], repair=True)


_FIX_RULES = """A compiler checked the typed query graph written for the question at the end and found the errors listed after it. Write the corrected typed query graph, as one JSON object on one line in the same language:
{schema}
Rules:
1. Correct every error listed and keep the rest of the graph as it is.
2. Constants are names the question gives, written as in the question; every value the question does not give is a variable ("?x"), with an atom that finds it.
3. Every atom has one subject and one object, each a single constant or variable; a relation never hides an unknown; a fact about one variable alone is a filter.
4. Every unknown the answer needs is found from the question's constants, one atom at a time."""

TYPED_FIX_USER = _FIX_RULES.format(schema=_SCHEMA)


def typed_fix_messages(question: str, shown: str, errors: Sequence[dict[str, str]]) -> list[dict[str, Any]]:
    listed = "\n".join(f"{i}. {e['message']}" for i, e in enumerate(errors, 1))
    tail = f"Question: {question}\n\nGraph:\n{shown}\n\nErrors:\n{listed}"
    return [{"role": "system", "content": TYPED_SYSTEM}, {"role": "user", "content": f"{TYPED_FIX_USER}\n\n{tail}"}]


async def budget_fallback(question: str, ctx: QueryContext, modalities: Iterable[str] | None = None,
                          ) -> CompiledQuery:
    mods = await catalog(ctx) if modalities is None else list(modalities)
    pattern = _fallback(question, "budget", {})
    out = compile_pattern(prune(copy.deepcopy(pattern), question, typed=True), question, mods, reply="",
                          source=(pattern, frozenset()))
    out.fallback = True
    return out


async def fixed(question: str, first: CompiledQuery, ctx: QueryContext, modalities: Iterable[str] | None,
                ) -> CompiledQuery:
    graph, _ = _parse(first.reply or "")
    shown = _dump(graph) if isinstance(graph, dict) else (first.reply or "").strip()[:2000] or "(no reply)"
    reply = await ctx.llm.chat(typed_fix_messages(question, shown, first.errors), role="pattern", task="typed_fix",
                               json_mode=True, max_tokens=700, ledger=ctx.ledger, charge="pattern")
    again = compile_reply(reply.text, question, modalities)
    summary = {"errors": first.errors, "ungrounded": first.ungrounded()}
    if not again.errors:
        again.pattern.raw["typed_first"] = first.pattern.raw.get("typed")
        again.first, again.fixed = summary, True
        return again
    raw = {k: again.pattern.raw[k] for k in ("typed", "reply") if k in again.pattern.raw}
    pattern = _fallback(question, "compile", raw)
    pattern.raw["typed_first"] = first.pattern.raw.get("typed")
    out = compile_pattern(prune(copy.deepcopy(pattern), question, typed=True), question, modalities,
                          reply=reply.text, source=(pattern, frozenset()))
    out.first, out.fallback = {**summary, "fix": again.errors}, True
    return out


TRANSCRIPTION_CHECKS = ("awaited", "atom", "order", "pictures", "owner", "trace", "sets")
INDEPENDENT_CHECKS = ("own_awaited", "bindable", "dropped", "capability")


def contract(compiled: CompiledQuery, run: _Run) -> list[dict[str, str]]:
    p, out = run.p, []

    def bad(check: str, detail: str) -> None:
        out.append({"check": check, "detail": detail})

    def shown(e: Edge) -> str:
        return f"({e.head} | {e.relation} | {e.tail})"

    own = [p.answer] if p.answer is not None and not p.raw.get("answer_inferred") and run.connected else list(
        p.variables)
    if run.answered() != compiled.awaited:
        bad("awaited", f"the run waits for {run.answered()}, the compiled query for {compiled.awaited}")
    if missing := [x for x in own if x not in compiled.awaited]:
        bad("own_awaited", f"the executor's own rule waits for {missing} too")
    rank = {x: i for i, x in enumerate(compiled.order)}
    seen: set[int] = set()
    index = {id(e): i for i, e in enumerate(p.edges)}
    for c in run.conds:
        if c.edge is None or c.hop:
            continue
        i = index.get(id(c.edge), -1)
        a = compiled.atoms.get(i)
        if a is None:
            bad("atom", f"the run evaluates {shown(c.edge)}, an atom the compiled query does not")
            continue
        seen.add(i)
        inputs = [*([a["key"]] if a["access"] == "join" else []), *a["uses"]]
        if c.var != a["out"] or (a["access"] == "join" and c.other != a["key"]) or sorted(c.deps) != sorted(a["uses"]):
            bad("atom", f"the run reads {shown(c.edge)} for {c.var} after {[c.other, *c.deps]}, the compiled query "
                        f"for {a['out']} after {inputs}")
        elif not compiled.loose and any(rank.get(y, len(rank)) >= rank.get(c.var, -1) for y in inputs):
            bad("order", f"{shown(c.edge)}: {inputs} do not come before {c.var} in the binding order {compiled.order}")
    for i in sorted(set(compiled.atoms) - seen):
        bad("atom", f"atom {i} {shown(p.edges[i])} is never evaluated")
    reach = _reach(run.conds, p.variables)
    ours, theirs = ([x for x in compiled.awaited if x not in held] for held in (set(compiled.bindable), reach))
    if ours != theirs:
        bad("bindable", f"the analysis cannot bind {ours}, the run's conditions {theirs}")
    mk, visual = compiled.marks, [c for c in run.conds if c.visual]
    if not compiled.pictures and visual:
        bad("capability", "no site holds an image, yet the run reads "
                          f"{[c.text or (c.edge.relation if c.edge else '') for c in visual]} on pictures")
    elif visual or mk.pictured or mk.image or mk.question:
        read = {(c.var, c.text) for c in visual if c.edge is None and not c.fallback}
        lowered = set(mk.pictured) | set(compiled.pictured)
        if lost := sorted(lowered - read):
            bad("pictures", f"the picture filters {lost} are read on texts")
        if extra := sorted(read - lowered):
            bad("pictures", f"the run reads {extra} on pictures, which the lowered query reads on texts")
        held = {index.get(id(c.edge), -1) for c in visual if c.edge is not None and not c.hop}
        if lost := sorted(i for i in mk.image if i in compiled.atoms and i not in held):
            bad("pictures", f"the image atoms {[shown(p.edges[i]) for i in lost]} are read on texts")
        if extra := sorted(i for i in held if i not in mk.image):
            bad("pictures", f"the run reads {[shown(p.edges[i]) if i >= 0 else '?' for i in extra]} on pictures, "
                            "which the lowered query reads on texts")
        asked = [c.var for c in visual if c.fallback and c.edge is None]
        if asked != ([mk.question] if mk.question else []):
            bad("pictures", f"the question-level picture condition is on {asked}, the lowered query's on "
                            f"{[mk.question] if mk.question else []}")
    for c in visual:
        if c.edge is not None and not c.hop and (i := index.get(id(c.edge))) is not None and \
                run.owns(c) != (c.var == mk.owners.get(i)):
            bad("owner", f"{shown(c.edge)} on {c.var}: the run reads {'its own' if run.owns(c) else 'the other'} "
                         f"end's picture, the lowered query {mk.owners.get(i)}'s")
    filters = {(x, t) for x, ts in compiled.pattern.conditions.items() for t in ts}
    by_id = {c.id: c for c in run.conds}
    for c in run.conds:
        if c.edge is None and not c.fallback and (c.var, c.text) not in filters:
            bad("trace", f"the run reads {c.var} {c.text!r}, a filter the lowered query does not have")
        if c.hop and not ((f := by_id.get(c.parent if c.parent is not None else -1)) is not None and f.edge is None
                          and f.visual and f.var == c.var and c.edge is not None and id(c.edge) in index
                          and c.var in (c.edge.head, c.edge.tail) and not is_var(c.edge.other(c.var))):
            bad("trace", f"the standby hop on {c.var} {shown(c.edge) if c.edge else ''} stands for no picture filter "
                         "of its variable over one of its anchored atoms")
    if compiled.ir is not None:
        done = {triple(e.head, e.relation, e.tail) for c in run.conds if c.edge is not None and not c.hop
                for e in [_edge_at(compiled.pattern, index.get(id(c.edge)), c.edge)]}
        for a in compiled.ir.get("atoms") or []:
            ends = [str(a.get("subject", "")), str(a.get("object", ""))]
            t = triple(ends[0], str(a.get("relation", "")), ends[1])
            if any(is_var(n) for n in ends) and t not in done and t not in compiled.lowering.atoms:
                bad("dropped", f"the atom ({' | '.join(t)}) is neither evaluated nor rewritten by a lowering rule")
    if run.m.graph == "typed":
        for x in p.raw.get("sets", ()):
            if x in p.variables and run.cap(x) != run.list_cap:
                bad("sets", f"{x} is set-valued, yet a read may name {run.cap(x)} of its values")
    return out


def _edge_at(pattern: Pattern, i: int | None, default: Edge) -> Edge:
    return pattern.edges[i] if i is not None and 0 <= i < len(pattern.edges) else default


def recorded(compiled: CompiledQuery, run: _Run) -> list[dict[str, str]]:
    violations = contract(compiled, run)
    run.p.raw["compile"] = compiled.record(violations)
    first = compiled.first["errors"] if compiled.first is not None else compiled.errors
    run.ctx.log("compile", repair=compiled.repair, **compiled.counts(), errors=[e["code"] for e in first],
                violations=violations, lowered=[list(a) for a in compiled.applied()])
    led = run.ctx.ledger
    for what, n in {**({} if compiled.repair else compiled.counts()), **compiled.lowered_counts(),
                    "contract_violations": len(violations)}.items():
        led.count_compile(what, n)
    return violations


INVARIANTS = ("access", "witnessed", "cost", "capability")


def inadmissible(r: ProbeRequest, question: str, known: Callable[[str], bool], filters: Collection[str],
                 ) -> str | None:
    if r.kind == "bound" and r.key != question:
        if r.value is not None and not known(r.value):
            return f"a value key on {r.value!r}, no constant of the query and no value bound so far"
        if r.value is None and r.key not in filters and not all(part in filters for part in r.key.split("; ")):
            return f"the key {r.key!r} on no constant, no bound value and no filter of the query"
    elif r.kind not in ("question", "title", "bound"):
        return f"a probe of kind {r.kind!r}"
    told = [v for v in (*r.titles, *r.mentions) if not known(v)]
    return f"it tells the site {told}, no constant of the query and no value bound so far" if told else None


def invariants(runs: Sequence[_Run]) -> dict[str, list[str]]:
    first, last = runs[0], runs[-1]
    out = {name: list(first.breaches.get(name, ())) for name in INVARIANTS}
    for r in runs:
        for x, held in r.values.items():
            for h in held.values():
                out["witnessed"] += [f"{x} = {h.name!r} cites {s}/{o}, which no read of the question validated"
                                     for s, o in h.refs if (s, o) not in first.witnessable]
    if last.m.tokens is not None and (spent := last.evidence()) > last.m.tokens + last.m.repair_floor:
        out["cost"].append(f"{math.ceil(spent)} evidence tokens, beyond the budget {last.m.tokens} and the repair "
                           f"floor {last.m.repair_floor}")
    texts = dict.fromkeys(ref for r in runs for ref in r.snippets)
    out["capability"] += [f"the raw text of {s}/{o}, whose policy releases titles only" for s, o in texts
                          if s in first.titled]
    return out


REPAIR_EDITS = ("split", "constant_to_filter", "merge", "add_atom", "drop_atom", "relax", "rename", "other")


def _words(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def edits(first: Pattern, bound: Mapping[str, Sequence[str]], repaired: Pattern) -> list[str]:
    def end(n: str, values: Mapping[str, Sequence[str]]) -> frozenset[str] | None:
        if not is_var(n):
            return frozenset({norm_entity(n)})
        return frozenset(norm_entity(v) for v in values[n]) if values.get(n) else None

    def meet(a: frozenset[str] | None, b: frozenset[str] | None) -> bool:
        return a is None and b is None or a is not None and b is not None and bool(a & b)

    def rel(x: tuple[Edge, Any, Any]) -> str:
        return " ".join(x[0].relation.lower().split())

    def ends_meet(a: tuple[Edge, Any, Any], b: tuple[Edge, Any, Any]) -> bool:
        return meet(a[1], b[1]) and meet(a[2], b[2]) or meet(a[1], b[2]) and meet(a[2], b[1])

    def across(x: tuple[Edge, Any, Any], v: str) -> frozenset[str] | None:
        return x[2] if x[0].head == v else x[1]

    def on(x: tuple[Edge, Any, Any]) -> list[str]:
        return [n for n in dict.fromkeys((x[0].head, x[0].tail)) if is_var(n)]

    # ends compare by value: a first-graph variable is its bound values, any other variable None
    ones = [(e, end(e.head, bound), end(e.tail, bound)) for e in first.edges]
    twos = [(e, end(e.head, {}), end(e.tail, {})) for e in repaired.edges]
    found: list[str] = []

    def pair(xs: list[Any], ys: list[Any], test: Any) -> list[tuple[Any, tuple[Any, Any]]]:
        out = []
        for x in list(xs):
            hit = next(((y1, y2) for y1 in ys for y2 in ys if y1 is not y2 for v in on(y1)
                        if v in on(y2) and test(x, y1, y2, v)), None)
            if hit is not None:
                xs.remove(x)
                ys.remove(hit[0])
                ys.remove(hit[1])
                out.append((x, hit))
        return out

    for a in list(ones):
        b = next((b for b in twos if ends_meet(a, b) and rel(a) == rel(b)), None)
        if b is not None:
            ones.remove(a)
            twos.remove(b)
    found += ["split"] * len(pair(ones, twos, lambda a, b, c, v: meet(across(b, v), a[1]) and meet(across(c, v), a[2])))
    found += ["merge"] * len(pair(twos, ones, lambda b, a, c, v: not bound.get(v) and meet(across(a, v), b[1])
                                  and meet(across(c, v), b[2])))
    for a in list(ones):
        for k, constant in ((2, a[0].head), (1, a[0].tail)):
            b = next((b for b in twos for v in on(b) if repaired.conditions.get(v) and meet(across(b, v), a[k])),
                     None) if not is_var(constant) and a in ones else None
            if b is not None:
                ones.remove(a)
                twos.remove(b)
                found.append("constant_to_filter")
    for a in list(ones):
        b = next((b for b in twos if ends_meet(a, b)), None)
        if b is not None:
            ones.remove(a)
            twos.remove(b)
            found.append("relax" if _words(rel(b)) < _words(rel(a)) else "rename")
    found += ["add_atom"] * len(twos)
    found += ["drop_atom" for a in ones if a[1] is None or a[2] is None]
    after = [_words(t) for ts in repaired.conditions.values() for t in ts]
    for t in (t for x, ts in first.conditions.items() if not bound.get(x) for t in ts):
        if not any(u >= _words(t) for u in after):
            found.append("relax")
    return [kind for kind in REPAIR_EDITS if kind in found] or ["other"]
