# prompts and reply parsers for every model task, with the mention and witness tests
from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

from rapidfuzz import fuzz, process

from fedquest.llm.parsing import (
    COMPLETION_DELIM,
    RECORD_DELIM,
    TUPLE_DELIM,
    Triple,
    json_objects,
    json_values,
    parse_json,
    parse_records,
)
from fedquest.types import Edge, Fact, Pattern

PATTERN_VERSION = "qg-v1"
EXTRACT_VERSION = "kg-v3"
QUERY_EXTRACT_VERSION = "mgrag-q-v2"
QUERY_EXTRACT_T_VERSION = "mgrag-qt-v1"
QUERY_EXTRACT_BLOCK_VERSION = "mgrag-qb-v1"
INDEX_BIND_VERSION = "bind-index-v1"

Message = dict[str, Any]

_DELIMS = {
    "tuple_delimiter": TUPLE_DELIM,
    "record_delimiter": RECORD_DELIM,
    "completion_delimiter": COMPLETION_DELIM,
}


def _image_part(data_uri: str) -> dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": data_uri}}


def _content(text: str, images: Sequence[str] = (), images_first: bool = False) -> str | list[dict[str, Any]]:
    if not images:
        return text
    parts = [_image_part(u) for u in images]
    text_part = {"type": "text", "text": text}
    return [*parts, text_part] if images_first else [text_part, *parts]


MGRAG_SYSTEM = (
    "You are an AI assistant that extracts query related entity nodes and relation edges from "
    "multimodal information sources."
)

_PATTERN_USER = """
<QUERY>
{query_content}
</QUERY>

Given the above query, decompose the query and extract a query graph that contains entity nodes and relation edges for solving the query.
You need to represent the graph in triplet format.
Your extracted triples should consists of entity nodes and relation edges.
For the key entities nodes for solving the query, and not explicitly mentioned in the query, you can represent them with placeholder entities that suit the node's semantic feature, such as "person1", "person2", "location1", "location2", "date1", "date2" etc.
For the key entities nodes that are explicitly mentioned in the query, mark them as "anchor_entity".
Also mark the placeholder entity whose value answers the query as ("answer_entity"{tuple_delimiter}<entity_name>).

Guidelines:
1. Extract other significant entities, concepts, or actors mentioned in the text or image.
2. DO NOT create nodes for relationships or actions, create them as edges.
3. DO NOT create nodes for temporal information like dates, times or years, instead, added them to edges later.
4. Be as explicit as possible in your entity node and relation edge names, using full names.

Output format:
Step 1. For each identified entity node, extract and output the following information:
- entity_name: Name of the entity, use same language as input text.
- entity_description: Short and precise description of the entity's attributes and activities
Format entity as ("entity"{tuple_delimiter}<entity_name>), and the anchor entities should be marked as ("anchor_entity"{tuple_delimiter}<entity_name>).

Step 2. From the entities identified in step 1, identify all relation edges as pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of relation edges, extract the following information:
- source_entity: Name of the source entity, as identified in step 1
- target_entity: Name of the target entity, as identified in step 1
- relation_description: Short and precise explanation as to why you think the source entity and the target entity are related to each other, also include the temporal information here
- relation_names: One or more high-level key words that summarize the overarching nature of the relation, focusing on concepts or themes rather than specific details
Format each relation edge as ("relation"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_name>)

Step 3. Split different record with {record_delimiter}. Remember to firstly output entities, then relations. When finished, output {completion_delimiter}

An example output template for the query "Where was the performer of Count 'Em 88 born" is given as follow, strictly output as this template, do not output any other words:
("anchor_entity"{tuple_delimiter}Count 'Em 88){record_delimiter}
("entity"{tuple_delimiter}person1){record_delimiter}
("entity"{tuple_delimiter}location1){record_delimiter}
("answer_entity"{tuple_delimiter}location1){record_delimiter}
("relation"{tuple_delimiter}Count 'Em 88{tuple_delimiter}person1{tuple_delimiter}performed by){record_delimiter}
("relation"{tuple_delimiter}person1{tuple_delimiter}location1{tuple_delimiter}born in){completion_delimiter}
"""


def pattern_messages(question: str) -> list[Message]:
    return [
        {"role": "system", "content": MGRAG_SYSTEM},
        {"role": "user", "content": _PATTERN_USER.format(query_content=question, **_DELIMS)},
    ]


REPAIR_VERSION = "repair-v1"

_REPAIR_USER = """
<QUERY>
{query_content}
</QUERY>

A first attempt decomposed the query into the query graph below and searched the data for each relation edge. Each edge shows what the search found.

<GRAPH>
{graph}
</GRAPH>

Facts found so far:
{facts}

Problem: {problem}

Write a corrected query graph for the query in the format of the first attempt. Rules:
1. Use the values found so far as anchor entities.
2. If an edge found nothing because its relation hides an intermediate entity, or because an anchor entity is a description rather than a name, split it into two edges joined by a new placeholder entity (such as "person1" or "location1").
3. If a placeholder has no source of its own and only its neighbor can be found, merge its two edges into one.
4. If the query asks for a property of a found entity that no edge covers, add that edge, ending in a new placeholder.
5. Mark the placeholder whose value answers the query as ("answer_entity"{tuple_delimiter}<entity_name>). Leave out edges that were found and are no longer needed.

Format each entity as ("anchor_entity"{tuple_delimiter}<entity_name>) or ("entity"{tuple_delimiter}<placeholder>), and each relation edge as ("relation"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relation name>). Output the entities first, then the relations; split records with {record_delimiter}; when finished, output {completion_delimiter}. Output nothing else.

Example: for "What river flows through the city where the designer of Southeast Library died?", where the first attempt found (Southeast Library | designed by | Ralph Rapson) and (Ralph Rapson | died in | Minneapolis) but its graph ended at the city:
("anchor_entity"{tuple_delimiter}Minneapolis){record_delimiter}
("entity"{tuple_delimiter}river1){record_delimiter}
("answer_entity"{tuple_delimiter}river1){record_delimiter}
("relation"{tuple_delimiter}river1{tuple_delimiter}Minneapolis{tuple_delimiter}flows through){completion_delimiter}
"""


def repair_messages(question: str, graph: Sequence[str], facts: Sequence[str], problem: str) -> list[Message]:
    return [
        {"role": "system", "content": MGRAG_SYSTEM},
        {"role": "user", "content": _REPAIR_USER.format(query_content=question, graph="\n".join(graph) or "(none)",
                                                         facts="\n".join(facts) or "(none)", problem=problem,
                                                         **_DELIMS)},
    ]


GROUND_VERSION = "ground-v1"

GROUND_SYSTEM = "You link the names and descriptions a question uses to the entities of a collection. Return JSON only."

_GROUND_USER = """Question: {question}

Each numbered phrase below comes from the question, but no source in the collection names it as written. Under each phrase are the entities a search for it found (title: the start of its entry).

{phrases}

For each phrase, which listed entity does the question mean by it? The phrase may be another name, a misspelling, a shortening or a description of that entity. Answer 0 when the question means none of them.
Reply with JSON only, on one line, mapping each phrase number to an entity number, for example {{"1": 2, "2": 0}}."""


def ground_messages(question: str, phrases: Sequence[tuple[str, Sequence[str]]]) -> list[Message]:
    blocks = "\n".join(f'({i}) "{phrase}"\n' + "\n".join(f"    {j}. {line}" for j, line in enumerate(lines, 1))
                       for i, (phrase, lines) in enumerate(phrases, 1))
    return [{"role": "system", "content": GROUND_SYSTEM},
            {"role": "user", "content": _GROUND_USER.format(question=question, phrases=blocks)}]


def parse_links(text: str, sizes: Sequence[int]) -> dict[int, int] | None:
    obj = parse_json(text)

    def num(v: Any) -> int | None:
        if isinstance(v, bool):
            return None
        if isinstance(v, int | float) and float(v).is_integer():
            return int(v)
        s = v.strip().strip("()") if isinstance(v, str) else ""
        return int(s) if s.isdigit() else None

    if isinstance(obj, list):
        obj = {i: v for i, v in enumerate(obj, 1)}
    if not isinstance(obj, dict):
        return None
    out: dict[int, int] = {}
    for k, v in obj.items():
        i, j = num(k), num(v)
        if i is not None and 1 <= i <= len(sizes):
            out[i] = j if j is not None and 1 <= j <= sizes[i - 1] else 0
    return out


_NON_ALNUM = re.compile(r"[^0-9a-z]+")
_GENERIC_NODES = frozenset(
    "film movie title show series episode person people actor actress team player album song track book novel game "
    "city country state location place date year time company organization school university building award prize "
    "category role character network channel league club station artist band singer writer author director event "
    "season tournament answer entity".split()
)


def _key(s: str) -> str:
    return _NON_ALNUM.sub("", s.lower())


_NUMBERED = re.compile(r"(.*[a-z])(\d+)")


def _numbered_apart(a: str, b: str) -> bool:
    ma, mb = _NUMBERED.fullmatch(a), _NUMBERED.fullmatch(b)
    return bool(ma and mb and ma.group(1) == mb.group(1) and int(ma.group(2)) != int(mb.group(2)))


def _words(s: str) -> str:
    return " " + " ".join(_NON_ALNUM.sub(" ", s.lower()).split()) + " "


def _named(v: str, answer: str | None, question: str) -> bool:
    name = v[1:].strip()
    return not (v == answer or re.search(r"\d\s*$", name) or len(_key(name)) < 3
                or _words(name).strip() in _GENERIC_NODES or _words(name) not in _words(question))


def described(variable: str, question: str) -> str | None:
    name = re.sub(r"(?<=[^\W\d]|_)\d{1,2}$", "", variable.lstrip("?")).replace("_", " ").strip()
    if not name or _words(name).strip() in _GENERIC_NODES:
        return None
    return name if mentions(question, name) else None


# a placeholder the question names outright: a constant on one edge, folded into the relation between two
def _fold_concepts(edges: list[Edge], variables: list[str], answer: str | None, question: str) -> list[str]:
    constants = []
    for v in list(variables):
        if not _named(v, answer, question):
            continue
        on = [e for e in edges if v in (e.head, e.tail)]
        name = v[1:].strip()
        if len(on) == 1:
            e = on[0]
            e.head, e.tail = (name if e.head == v else e.head), (name if e.tail == v else e.tail)
            constants.append(name)
        elif len(on) == 2:
            into = next((e for e in on if e.tail == v), None)
            out = next((e for e in on if e.head == v), None)
            if into is None or out is None or into.head == out.tail or not (into.head.startswith("?") or
                                                                             out.tail.startswith("?")):
                continue
            into.relation, into.tail = f"{into.relation} {name} {out.relation}", out.tail
            edges.remove(out)
        else:
            continue
        variables.remove(v)
    for i, e in enumerate(edges):
        e.id = i
    return constants


def parse_pattern(text: str, question: str) -> Pattern:
    records = parse_records(text)
    anchors: list[str] = []
    entities: list[str] = []
    answer_names: list[str] = []
    relations: list[tuple[str, str, str]] = []
    for rec in records:
        kind = rec[0].lower().strip()
        if kind == "anchor_entity" and len(rec) >= 2 and rec[1]:
            anchors.append(rec[1])
        elif kind == "entity" and len(rec) >= 2 and rec[1]:
            entities.append(rec[1])
        elif kind == "answer_entity" and len(rec) >= 2 and rec[1]:
            answer_names.append(rec[1])
        elif kind == "relation" and len(rec) >= 4 and rec[1] and rec[2] and rec[3].strip():
            relations.append((rec[1], rec[2], rec[3].strip()))
    answer_name = answer_names[-1] if answer_names else None

    anchors = list(dict.fromkeys(anchors))
    anchor_keys = {_key(a): a for a in anchors}
    placeholders = [e for e in dict.fromkeys(entities) if _key(e) not in anchor_keys]
    nodes: dict[str, str] = dict(anchor_keys)
    for e in placeholders:
        nodes.setdefault(_key(e), f"?{e}")
    undeclared = {_key(a) for a in answer_names if _key(a) and _key(a) not in nodes}
    for a in answer_names:
        if _key(a) in undeclared:
            nodes.setdefault(_key(a), f"?{a}")
    unnamed = set(undeclared)

    def resolve(name: str, skip: str | None = None) -> str:
        k = _key(name)
        if k in nodes and k != skip:
            if k in unnamed:
                unnamed.discard(k)
                nodes[k] = f"?{name}"
            return nodes[k]
        # an unknown name snaps to a node at fuzz ratio 80, but never person1 to person2
        pool = [n for n in nodes if n != skip and not _numbered_apart(k, n)]
        if pool:
            match = process.extractOne(k, pool, scorer=fuzz.ratio)
            if match is not None and match[1] >= 80:
                unnamed.discard(match[0])
                return nodes[match[0]]
        nodes.setdefault(k, f"?{name}")
        return nodes[k]

    edges: list[Edge] = []
    seen: set[tuple[str, str, str]] = set()
    unary: dict[str, list[str]] = {}
    for head, tail, rel in relations:
        h, t = resolve(head), resolve(tail)
        if h == t:
            if h.startswith("?") and rel not in unary.get(h, []):
                unary.setdefault(h, []).append(rel)
            continue
        if (h, rel.lower(), t) in seen:
            continue
        seen.add((h, rel.lower(), t))
        edges.append(Edge(id=len(edges), head=h, relation=rel, tail=t))

    if not edges:
        loose = [c for conds in unary.values() for c in conds]
        return Pattern(
            anchors=[],
            variables=["?answer"],
            edges=[Edge(0, question, "answer", "?answer")],
            answer="?answer",
            raw={"records": records, "fallback": True},
            conditions={"?answer": loose} if loose else {},
        )
    variables = list(dict.fromkeys(n for e in edges for n in (e.head, e.tail) if n.startswith("?")))
    answer = None
    named_answer = resolve(answer_name) if answer_name is not None else None
    if (answer_name is not None and _key(answer_name) in undeclared and named_answer not in variables
            and named_answer not in unary):
        named_answer = resolve(answer_name, skip=_key(answer_name))
    if named_answer is not None and named_answer in variables:
        answer = named_answer
    elif named_answer is not None and named_answer in unary:
        variables.append(named_answer)
        answer = named_answer
    inferred = answer is None
    if answer is None:
        tails_only = [v for v in variables if all(e.head != v for e in edges)]
        answer = tails_only[-1] if tails_only else (variables[-1] if variables else None)
    constants = _fold_concepts(edges, variables, answer, question)
    anchors = [*anchors, *(name for name in constants if name not in anchors)]
    used_anchors = [a for a in anchors if any(a in (e.head, e.tail) for e in edges)]
    raw: dict[str, Any] = {"records": records}
    if inferred:
        raw["answer_inferred"] = True
    return Pattern(
        anchors=used_anchors, variables=variables, edges=edges, answer=answer, raw=raw,
        conditions={v: conds for v, conds in unary.items() if v in variables},
    )


KG_SYSTEM = (
    "You build a knowledge graph from one source. Extract only facts stated in or clearly shown by "
    "the source. Return JSON only."
)

_KG_STEPS = """List the source's facts as subject-predicate-object triples about its key entities: people, organizations, places, creative works (films, books, songs, albums), events, products, dates, numbers and concepts. Use full canonical names and refer to the main subject by the title. Use short lowercase predicates such as "acted in", "directed by", "released in", "located in", "born on", "member of". Replace pronouns and phrases like "the film" with the named entity. Keep dates, numbers and attributes as objects. At most 25 triples.{extra}
Return {{"triples": [["subject", "predicate", "object"], ...]}}"""


def extract_text_messages(title: str, text: str) -> list[Message]:
    user = f'Source title: {title}\nSource text:\n"""\n{text}\n"""\n' + _KG_STEPS.format(extra="")
    return [{"role": "system", "content": KG_SYSTEM}, {"role": "user", "content": user}]


def extract_table_messages(title: str, text: str) -> list[Message]:
    page = text.split(" | ", 1)[0].strip() or title
    extra = (
        "\nRelate the row's main entity (usually the first named cell) to every other cell using the column "
        f"names as predicates, and relate the page entity '{page}' to the row's main entity (for example "
        '["Ben Piazza", "filmography includes", "The Bad News Bears"]).'
    )
    user = f'Source title: {title}\nSource table row:\n"""\n{text}\n"""\n' + _KG_STEPS.format(extra=extra)
    return [{"role": "system", "content": KG_SYSTEM}, {"role": "user", "content": user}]


def extract_image_messages(title: str, caption: str | None, image_data_uri: str) -> list[Message]:
    cap = f" Caption: {caption}" if caption else ""
    text = (
        f'This image belongs to "{title}".{cap}\n'
        f'List what is visible as triples, using "{title}" as the subject where possible: depicted people, '
        "objects, animals, sports or activities, colors, text written in the image, setting and counts "
        f'(for example ["{title}", "depicts", "baseball"], ["{title}", "has color", "green"]). At most 20 triples.\n'
        'Return {"triples": [[s, p, o], ...]}'
    )
    return [{"role": "system", "content": KG_SYSTEM}, {"role": "user", "content": _content(text, [image_data_uri])}]


_QUERY_EXTRACT_USER = """
<QUERY>
{query_content}
</QUERY>

<TEXT>
{text_content}
</TEXT>

Given text or image that is potentially relevant to the above query,
extract fact triples from text or image that are explicitly or implicitly mentioned and relevant to the query provided.
The extracted fact triples should consists of entity nodes and relation edges.

Guidelines:
1. Extract other significant entities, concepts, or actors mentioned in the text or image.
2. DO NOT create nodes for relationships or actions, create them as edges.
3. DO NOT create nodes for temporal information like dates, times or years, instead, added them to edges later.
4. Be as explicit as possible in your entity node and relation edge names, using full names.

Output format:
Step 1. For each identified entity node, extract and output the following information:
- entity_name: Name of the entity, use same language as input text.
- entity_description: Short and precise description of the entity's attributes and activities
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_description>)

Step 2. From the entities identified in step 1, identify all relation edges as pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of relation edges, extract the following information:
- source_entity: Name of the source entity, as identified in step 1
- target_entity: Name of the target entity, as identified in step 1
- relation_description: Short and precise explanation as to why you think the source entity and the target entity are related to each other, also include the temporal information here
- relation_names: One or more high-level key words that summarize the overarching nature of the relation, focusing on concepts or themes rather than specific details
Format each relation edge as ("relation"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_name>{tuple_delimiter}<relationship_description>)

Step 3. Split different record with {record_delimiter}. When finished, output {completion_delimiter}

An example output template for 2 entities and 1 relation is given as follow, strictly output as this template, do not output any other words:
("entity"{tuple_delimiter}entity1{tuple_delimiter}description for entity1){record_delimiter}
("entity"{tuple_delimiter}entity2{tuple_delimiter}description for entity2){record_delimiter}
("relation"{tuple_delimiter}entity1{tuple_delimiter}entity2{tuple_delimiter}relation1{tuple_delimiter}description for relation1){completion_delimiter}
"""


def extract_query_messages(query: str, text: str, image_data_uri: str | None = None) -> list[Message]:
    user = _QUERY_EXTRACT_USER.format(query_content=query, text_content=text, **_DELIMS)
    images = [image_data_uri] if image_data_uri else []
    return [{"role": "system", "content": MGRAG_SYSTEM}, {"role": "user", "content": _content(user, images, True)}]


def image_of(title: str) -> str:
    return f"Image of {title}"


def extract_query_block_messages(query: str, sources: Sequence[str],
                                 image_data_uris: Sequence[str] | None = None) -> list[Message]:
    text = "\n\n".join(f"[{i}] {s}".rstrip() for i, s in enumerate(sources, 1))
    user = _QUERY_EXTRACT_USER.format(query_content=query, text_content=text, **_DELIMS)
    if not image_data_uris:
        return [{"role": "system", "content": MGRAG_SYSTEM}, {"role": "user", "content": user}]
    parts: list[dict[str, Any]] = []
    for i, (shown, uri) in enumerate(zip(sources, image_data_uris, strict=True), 1):
        parts += [{"type": "text", "text": f"[{i}] {shown}".rstrip()}, _image_part(uri)]
    return [{"role": "system", "content": MGRAG_SYSTEM},
            {"role": "user", "content": [*parts, {"type": "text", "text": user}]}]


def parse_query_records(text: str, limit: int = 40) -> tuple[list[tuple[str, str]], list[Triple]]:
    entities: dict[str, tuple[str, str]] = {}
    out: list[Triple] = []
    seen: set[tuple[str, str, str]] = set()
    for rec in parse_records(text):
        kind = rec[0].lower()
        if kind == "entity" and len(rec) >= 2 and rec[1].strip() and len(rec[1].strip()) <= 200:
            name = rec[1].strip()
            description = (rec[-1] if len(rec) > 2 else "").strip()
            entities.setdefault(name.casefold(), (name, description))
        elif kind == "relation" and len(rec) >= 4 and all(x.strip() for x in rec[1:4]):
            triple = (rec[1].strip(), rec[3].strip(), rec[2].strip())
            key = tuple(x.casefold() for x in triple)
            if key not in seen and all(len(x) <= 200 for x in triple):
                seen.add(key)  # type: ignore[arg-type]
                out.append(triple)
    return list(entities.values())[: 2 * limit], out[:limit]


def parse_query_extraction(text: str) -> list[Triple]:
    return parse_query_records(text)[1]


ANSWER_SYSTEM = """
# Task
Imagine you are a text QA expert, skilled in delivering contextually relevant answers. You will receive:
1. Query.
2. Contexts.

Your task is to answer the query based solely on the content of the context.

# Requirements
Ensure that your answer does not include any additional information outside the context. Please note that your answer should be in pure text format.

# Output Format
Provide the answer in pure text format. Answer the question in <ANS> Answer here </ANS>. Do not include any information beyond what is contained in the context.
"""

_STYLE = {
    "short": "\nAnswer with only the short answer (a few words, no explanation) inside <ANS></ANS>.",
    "long": "\nAnswer in one to four sentences inside <ANS></ANS>.",
}


def format_facts(facts: Sequence[Fact], limit: int = 60) -> str:
    return "\n".join(f"({f.head} | {f.relation} | {f.tail})" for f in list(facts)[:limit])


def answer_messages(
    question: str,
    facts_block: str,
    snippets: Sequence[str],
    style: str = "short",
    image_data_uris: Sequence[str | tuple[str, str]] | None = None,
) -> list[Message]:
    text = "Here is the real question:\n" + question
    if snippets:
        text += "\nHere is the supporting material to help you answer the question:\n" + "\n\n".join(snippets)
    if facts_block:
        text += "\nHere is some graph information that might be helpful:\n" + facts_block
    text += _STYLE.get(style, _STYLE["short"])
    images = list(image_data_uris or [])
    if images and all(isinstance(i, tuple) for i in images):
        parts: list[dict[str, Any]] = [{"type": "text", "text": text}]
        for n, (label, uri) in enumerate(images, 1):
            parts += [{"type": "text", "text": f"Image {n}: {label}"}, _image_part(uri)]
        return [{"role": "system", "content": ANSWER_SYSTEM}, {"role": "user", "content": parts}]
    uris = [i[1] if isinstance(i, tuple) else i for i in images]
    return [{"role": "system", "content": ANSWER_SYSTEM}, {"role": "user", "content": _content(text, uris)}]


VERIFY_SYSTEM = "You check whether evidence is sufficient to answer a question. Use no outside knowledge."


def verify_messages(question: str, facts_block: str) -> list[Message]:
    user = (
        f"Question: {question}\nEvidence facts:\n{facts_block}\n"
        "Do these facts alone determine the answer to the question? "
        'Reply with JSON only: {"answerable": true or false, "answer": "<short answer, or empty>"}'
    )
    return [{"role": "system", "content": VERIFY_SYSTEM}, {"role": "user", "content": user}]


READ_VERSION = "read-v5"
PER_CALL = {"text": 6, "table": 12, "image": 4}
IMAGE_TOKENS = 280

READ_SYSTEM = (
    "You read numbered sources that a search returned for one step of a question: text passages, table rows, "
    "images, or index entries. You name the entities that fill the step's unknowns. Use only what the sources state "
    "or show, never your own knowledge. Return JSON only."
)

_JUDGMENTS = frozenset({"yes", "no", "true", "false"})
_MONTHS = {m[:3]: m for m in ("january", "february", "march", "april", "may", "june", "july", "august",
                              "september", "october", "november", "december")}


@dataclass
class ReadObject:
    kind: str
    text: str
    image: str | None = None


@dataclass
class ReadUnknown:
    name: str
    conditions: list[str]
    values: list[str | None]
    cap: int = 3
    mode: str = "value"
    candidates: list[str] = field(default_factory=list)
    answer: bool = False
    later: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ReadUnknown:
        conds = [str(c) for c in d.get("conditions") or []]
        vals = list(d.get("values") or [None] * len(conds))
        return cls(str(d["name"]), conds, (vals + [None] * len(conds))[: len(conds)], int(d.get("max", 3)),
                   str(d.get("mode", "value")), [str(c) for c in d.get("candidates") or []], bool(d.get("answer")),
                   [str(c) for c in d.get("later") or []])

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "conditions": self.conditions, "values": self.values, "max": self.cap,
                "mode": self.mode, "candidates": self.candidates, "answer": self.answer, "later": self.later}


def read_max_tokens(unknowns: Sequence[ReadUnknown]) -> int:
    return min(1200, 100 + 45 * sum(u.cap for u in unknowns))


HARVEST_TOKENS = 30
HARVEST_KEY = "_facts"


def _harvest_ask(unknowns: Sequence[ReadUnknown], harvest: int) -> str:
    known = list(dict.fromkeys(v for u in unknowns for v in u.values if v))
    about = (f"about {'; '.join(known)} or about the entities you name" if known else "about the entities you name")
    return (f" Also list, under \"{HARVEST_KEY}\", up to {harvest} other facts these sources state {about}, each "
            '{"s": "<subject>", "r": "<short relation>", "v": "<value>", "src": <source number>}, names copied '
            "exactly as the source writes them ([] if none).")


def parse_harvest(obj: Any, n_objects: int) -> list[dict[str, Any]]:
    items = obj.get(HARVEST_KEY) if isinstance(obj, dict) else None
    out = []
    for it in items if isinstance(items, list) else []:
        if not isinstance(it, dict):
            continue
        s, r, v = (str(it.get(k, "")).strip() for k in ("s", "r", "v"))
        try:
            k = int(it.get("src"))
        except (TypeError, ValueError):
            continue
        if s and r and v and 1 <= k <= n_objects and v.lower() not in {"null", "none", "unknown", "n/a"}:
            out.append({"subject": s, "relation": r, "value": v, "object": k})
    return out


def read_messages(question: str, objects: Sequence[ReadObject], unknowns: Sequence[ReadUnknown],
                  harvest: int = 0) -> list[Message]:
    lines = []
    for u in unknowns:
        how = (f"at most {u.cap}; X is the title an image is labelled with, if that image shows the conditions"
               if u.mode == "filter" else f"at most {u.cap}")
        lines.append(f"- {u.name} ({how}): " + "; ".join(f"({i}) {c}" for i, c in enumerate(u.conditions, 1)))
        if u.candidates:
            lines.append(f"  Candidates for X (found earlier; X may also be another entity): {'; '.join(u.candidates)}")
        if u.later:
            lines.append(f"  Not in this step (X's own picture is checked later): {'; '.join(u.later)}. List every X "
                         "the sources state for the conditions above, whatever its picture shows.")
    shape = ", ".join(f'"{u.name}": [{{"entity": "<name>", "object": <source number>'
                      + (', "conditions": [<numbers>]}]' if len(u.conditions) > 1 else "}]") for u in unknowns)
    ask = (
        "Unknowns (X is the entity to name):\n" + "\n".join(lines) + "\n"
        "For each unknown, list the entities that satisfy its conditions, best first, each with the source that states "
        "or shows it" + (" and, for an unknown with several conditions, the numbers of the conditions that this source "
                         "states or shows for it (an entity a source shows for only some conditions is listed with "
                         "just those numbers)" if any(len(u.conditions) > 1 for u in unknowns) else "")
        + ". Copy each name exactly as the source writes it; for a yes/no unknown write yes or no; for a count, the "
        "number. Apply the qualifiers the question puts on X (years, roles, places, numbers), except a condition "
        "marked for a later step. Use [] if no source states a fitting entity."
        + (_harvest_ask(unknowns, harvest) if harvest > 0 else "") + "\n"
        + f"Reply with JSON only, on one line: {{{shape}"
        + (f', "{HARVEST_KEY}": [...]' if harvest > 0 else "") + "}"
    )
    head = f"Question (context only): {question}\nSources:"
    if not any(o.image for o in objects):
        body = "\n".join(f"[{i}] ({o.kind}) {o.text}" for i, o in enumerate(objects, 1))
        return [{"role": "system", "content": READ_SYSTEM}, {"role": "user", "content": f"{head}\n{body}\n{ask}"}]
    parts: list[dict[str, Any]] = [{"type": "text", "text": head}]
    for i, o in enumerate(objects, 1):
        parts.append({"type": "text", "text": f"[{i}] ({o.kind}) {o.text}"})
        if o.image:
            parts.append(_image_part(o.image))
    parts.append({"type": "text", "text": ask})
    return [{"role": "system", "content": READ_SYSTEM}, {"role": "user", "content": parts}]


_HIT_KEY = re.compile(r'"(\??[A-Za-z_][\w ]*)"\s*:\s*[\[{]')


def read_reply(text: str) -> Any:
    values: list[Any] = []
    for v in json_values(text):
        if v not in values:
            values.append(v)
    first = next((v for v in values if isinstance(v, dict)), None)
    if first is not None and not ("entity" in first and len(values) == 1):
        return first
    # no keyed object: file each entity object under the nearest unknown key before it
    keys = [(m.start(), m.group(1)) for m in _HIT_KEY.finditer(text or "") if m.group(1) != "conditions"]
    out: dict[str, list[dict[str, Any]]] = {}
    loose: list[dict[str, Any]] = []
    for start, hit in json_objects(text):
        if isinstance(hit, dict) and "entity" in hit:
            name = next((k for pos, k in reversed(keys) if pos < start), None)
            (out.setdefault(name, []) if name else loose).append(hit)
    return out or loose or next((v for v in values if isinstance(v, list)), {})


@dataclass
class ReadHit:
    entity: str
    object: int
    conditions: list[int]


def parse_read(obj: Any, unknowns: Sequence[ReadUnknown], n_objects: int) -> dict[str, list[ReadHit]]:
    out: dict[str, list[ReadHit]] = {}
    if isinstance(obj, list) and len(unknowns) == 1:
        obj = {unknowns[0].name: obj}
    if not isinstance(obj, dict):
        return out
    for u in unknowns:
        raw = obj.get(u.name, obj.get(f"?{u.name}"))
        items = [raw] if isinstance(raw, dict) else raw if isinstance(raw, list) else []
        hits: list[ReadHit] = []
        for it in items:
            if not isinstance(it, dict):
                continue
            value = it.get("entity")
            if isinstance(value, bool):
                entity = "yes" if value else "no"
            elif isinstance(value, int | float):
                entity = str(int(value)) if float(value).is_integer() else str(value)
            elif isinstance(value, str):
                entity = value.strip().strip('"').strip()
            else:
                continue
            try:
                k = int(it.get("object"))
            except (TypeError, ValueError):
                continue
            if not entity or entity.lower() in {"null", "none", "unknown", "n/a"} or not 1 <= k <= n_objects:
                continue
            said = it.get("conditions")
            said = said if isinstance(said, list) else [said] if isinstance(said, int | str) else []
            nums = sorted({int(c) for c in said if (isinstance(c, int | float) or (isinstance(c, str) and c.strip().isdigit()))
                           and 1 <= int(c) <= len(u.conditions)})
            prev = next((h for h in hits if h.entity.lower() == entity.lower() and h.object == k), None)
            if prev is not None:
                prev.conditions = sorted({*prev.conditions, *nums})
                continue
            hits.append(ReadHit(entity, k, nums or ([1] if len(u.conditions) == 1 else [])))
        out[u.name] = hits[: max(1, u.cap)]
    return out


def canonical(entity: str, names: Sequence[str]) -> str | None:
    from fedquest.kg.normalize import same_entity

    if judgment(entity) is not None:
        return None
    return next((n for n in names if n and (same_entity(n, entity) or witnessed(n, entity))), None)


def judgment(entity: str) -> bool | None:
    e = entity.strip().lower()
    return True if e in ("yes", "true") else False if e in ("no", "false") else None


def claims(text: str | None, u: ReadUnknown, hit: ReadHit, label: str = "") -> list[int]:
    if text is None:
        if hit.conditions:
            return hit.conditions
        named = [i for i, v in enumerate(u.values, 1) if v is None or (label and mentions(label, v))]
        return named or list(range(1, len(u.conditions) + 1))
    ok = [i for i in hit.conditions if u.values[i - 1] is None or mentions(text, u.values[i - 1] or "")]
    if not hit.conditions:
        ok = [i for i, v in enumerate(u.values, 1) if v is not None and mentions(text, v)]
    return ok


MENTION_VERSIONS = (1, 2)
MENTION_TAU = Fraction(1, 2)
_MENTION: ContextVar[tuple[int, Fraction]] = ContextVar("fedquest_mention", default=(1, MENTION_TAU))


@contextmanager
def mention_scope(version: int = 1, tau: Fraction | float | str | None = None) -> Iterator[None]:
    if isinstance(version, bool) or version not in MENTION_VERSIONS:
        raise ValueError(f"unknown mention test {version!r}: 1 or 2 (qualifier-free)")
    share = MENTION_TAU if tau is None else Fraction(tau).limit_denominator(1000)
    if not 0 < share <= 1 or (version == 1 and share != MENTION_TAU):
        raise ValueError(f"mention tau {tau!r}: a share in (0, 1], and v1's is {MENTION_TAU}")
    token = _MENTION.set((version, share))
    try:
        yield
    finally:
        _MENTION.reset(token)


def mention_version() -> int:
    return _MENTION.get()[0]


def mention_test() -> tuple[int, Fraction]:
    return _MENTION.get()


def mention_words(value: str) -> list[str]:
    if _MENTION.get()[0] == 1:
        name = _norm_text(value)
    else:
        from fedquest.kg.normalize import norm_entity

        name = _norm_text(norm_entity(value))
    return [w for w in name.split() if len(w) >= 4 or w.isdigit()]


def mentions(text: str, value: str) -> bool:
    if witnessed(text, value):
        return True
    # else a tau share of the value's long words must appear; with fewer than two, only a witness counts
    tau = _MENTION.get()[1]
    words = mention_words(value)
    nt = _norm_text(text)
    return len(words) >= 2 and tau.denominator * sum(f" {w}" in nt for w in words) >= tau.numerator * len(words)


_NUMBER = re.compile(r"[\d.,%$\s]+")
_YEARLIKE = re.compile(r"\s*(1\d{3}|20\d{2})\s*")


_SENTENCE = re.compile(r"(?<=[.!?;])\s+|\n+|\s+\|\s+")


def _spans(text: str) -> list[str]:
    return [n for n in (_norm_text(part).strip() for part in _SENTENCE.split(text)) if n]


def _candidate_spans(text: str, title: str) -> list[str]:
    spans = _spans(text)
    t = _norm_text(title).strip() if title else ""
    return [t, *spans] if t else spans


def witness_span(text: str, entity: str, title: str = "", known: str | None = None) -> str:
    from fedquest.utils import stable_hash

    whole = f"{title} {text}" if title else text
    if judgment(entity) is not None or not witnessed(whole, entity):
        return ""
    spans = _candidate_spans(text, title)
    if known:
        for span in spans:
            if witnessed(span, entity) and mentions(span, known):
                return stable_hash("wspan-v1", span, n=16)
    for span in spans:
        if witnessed(span, entity):
            return stable_hash("wspan-v1", span, n=16)
    return stable_hash("wspan-v1", _norm_text(whole).strip(), n=16)


def span_holds(text: str, wspan: str, known: str | None = None, title: str = "") -> bool:
    from fedquest.utils import stable_hash

    if not wspan:
        return False
    whole = f"{title} {text}" if title else text
    if known and not mentions(whole, known):
        return False
    return any(stable_hash("wspan-v1", span, n=16) == wspan for span in _candidate_spans(text, title)) or \
        stable_hash("wspan-v1", _norm_text(whole).strip(), n=16) == wspan


def stated(text: str, u: ReadUnknown, entity: str) -> bool:
    return witnessed(text, entity) or (u.answer and _NUMBER.fullmatch(entity) is not None
                                       and _YEARLIKE.fullmatch(entity) is None)


def fold(s: str) -> str:
    import unicodedata

    return "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))


def _norm_text(s: str) -> str:
    import unicodedata

    s = re.sub(r"(?<=\d)[,\u202f ](?=\d{3}\b)", "", fold(unicodedata.normalize("NFKC", s)).lower())
    words = re.sub(r"[\W_]+", " ", s).split()
    return " " + " ".join(_MONTHS.get(w, w) for w in words) + " "


def witnessed(text: str, entity: str) -> bool:
    from fedquest.kg.normalize import norm_entity

    ne = _norm_text(norm_entity(entity)).strip()
    nt = _norm_text(text)
    if not ne:
        return False
    if ne in _JUDGMENTS or f" {ne} " in nt:
        return True
    words = [w for w in ne.split() if len(w) >= 3 or w.isdigit()]
    return bool(words) and all(f" {w} " in nt for w in words)


CLOSED_BOOK_SYSTEM = "You are a knowledgeable assistant. Answer the question from your own knowledge."


def closed_book_messages(question: str, style: str = "short") -> list[Message]:
    text = f"Question: {question}" + _STYLE.get(style, _STYLE["short"])
    return [{"role": "system", "content": CLOSED_BOOK_SYSTEM}, {"role": "user", "content": text}]


_JUDGE_1 = (
    "Instruction: You are a world class state of the art assistant for rating "
    "a User Answer given a Question. The Question is completely answered by the Reference Answer.\n"
    "Say 4, if User Answer is full contained and equivalent to Reference Answer"
    "in all terms, topics, numbers, metrics, dates and units.\n"
    "Say 2, if User Answer is partially contained and almost equivalent to Reference Answer"
    "in all terms, topics, numbers, metrics, dates and units.\n"
    "Say 0, if User Answer is not contained in Reference Answer or not accurate in all terms, topics,"
    "numbers, metrics, dates and units or the User Answer do not answer the question.\n"
    "Do not explain or justify your rating. Your rating must be only 4, 2 or 0 according to the instructions above.\n"
    "### Question: {query}\n"
    "### {answer0}: {sentence_inference}\n"
    "### {answer1}: {sentence_true}\n"
    "The rating is:\n"
)
_JUDGE_2 = (
    "I will rate the User Answer in comparison to the Reference Answer for a given Question.\n"
    "A rating of 4 indicates that the User Answer is entirely consistent with the Reference Answer, covering all aspects, topics, numbers, metrics, dates, and units.\n"
    "A rating of 2 signifies that the User Answer is mostly aligned with the Reference Answer, with minor discrepancies in some areas.\n"
    "A rating of 0 means that the User Answer is either inaccurate, incomplete, or unrelated to the Reference Answer, or it fails to address the Question.\n"
    "I will provide the rating without any explanation or justification, adhering to the following scale: 0 (no match), 2 (partial match), 4 (exact match).\n"
    "Do not explain or justify my rating. My rating must be only 4, 2 or 0 only.\n\n"
    "Question: {query}\n\n"
    "{answer0}: {sentence_inference}\n\n"
    "{answer1}: {sentence_true}\n\n"
    "Rating: "
)


def judge_messages(question: str, reference: str, response: str, variant: int) -> list[Message]:
    if variant == 1:
        text = _JUDGE_1.format(
            query=question, answer0="User Answer", sentence_inference=response,
            answer1="Reference Answer", sentence_true=reference,
        )
    else:
        text = _JUDGE_2.format(
            query=question, answer0="Reference Answer", sentence_inference=reference,
            answer1="User Answer", sentence_true=response,
        )
    return [{"role": "user", "content": text}]


def parse_rating(text: str) -> float | None:
    t = text or ""
    m = re.search(r"(?<!\d)[024](?!\d)", t) or re.search(r"[0-4]", t)
    return int(m.group()) / 4 if m else None


IRCOT_SYSTEM = "Answer multi-hop questions by reasoning one sentence at a time over the given paragraphs."


def ircot_messages(question: str, passages: Sequence[tuple[str, str]], thoughts: Sequence[str]) -> list[Message]:
    blocks = "".join(f"Wikipedia Title: {title}\n{text}\n\n" for title, text in passages)
    user = (
        f"{blocks}Q: {question}\nA: {' '.join(thoughts)}\n\n"
        "Write the next reasoning sentence. When you know the answer, write 'So the answer is: <answer>.'"
    )
    return [{"role": "system", "content": IRCOT_SYSTEM}, {"role": "user", "content": user}]


PLANNER_SYSTEM = """You answer a question by calling tools over a federation of databases at several sites.
Tools:
- search(query): returns the top objects (texts, table rows, images) at every site for the query, each with an id and a short summary.
- extract(id): reads one listed object at its site and returns its facts. Each extract costs budget.
- finish(answer): ends with the final short answer.
Reply with JSON only: {"thought": "...", "action": "search" | "extract" | "finish", "input": "..."}"""


def planner_messages(question: str, transcript: Sequence[dict[str, str]], budget_left: int) -> list[Message]:
    history = "\n".join(f"{t['role'].upper()}: {t['content']}" for t in transcript)
    user = f"Question: {question}\nExtraction budget left: {budget_left}\n{history}\nNext action (JSON):"
    return [{"role": "system", "content": PLANNER_SYSTEM}, {"role": "user", "content": user}]


PLANNER_OPS_SYSTEM = """You answer a question by calling tools over a federation of databases at several sites that hold texts, table rows and images. Searches list objects as [id] site | modality | summary. A read runs a model over listed objects where they are stored and returns only the values those objects state, each with the id of its source.
Tools:
- search(text): the top objects at every site for a text such as the question.
- probe(value, relation): the top objects at every site for a known value and a relation, e.g. {"value": "Ben Piazza", "relation": "acted in"}; add "value_first": false when the value ends the phrase, e.g. {"value": "Danny Boyle", "relation": "directed by", "value_first": false}.
- title(name, relation): the named entity's own objects (its article, its table rows, its pictures); the relation is optional and ranks them, e.g. {"name": "Ben Piazza", "relation": "acted in"}.
- rows(table): every row of one table, given as the id of one of its listed rows or as "page | table", e.g. {"table": 3} or {"table": "Ben Piazza | Filmography"}.
- read(ids, condition): reads the listed objects for one unknown X and returns the X values they state, at most 3. Write the condition about X: "(known value | relation | X)", "(X | relation | known value)", or a phrase such as "X was born in 1950". Add "list": true when every X the objects state is wanted (at most 24), and "final": true when X is the question's answer (only then may X be a number the objects let you compute, such as a count). When every condition has a known value, a text or table row that names none of them is skipped at no cost. Each read costs budget.
- check_pictures(candidates, condition): looks at the candidates' own pictures and returns the candidates whose picture shows the condition, e.g. {"candidates": ["Jaws", "Up"], "condition": "poster shows a shark"}. It costs budget.
- finish(answer): ends with the final short answer.
Reads and picture checks give at most 24 objects to a model per question, within the read budget in tokens.
Reply with JSON only: {"thought": "...", "action": "search" | "probe" | "title" | "rows" | "read" | "check_pictures" | "finish", "input": {...}}
Inputs: search {"text": "..."}; probe {"value": "...", "relation": "..."}; title {"name": "...", "relation": "..."}; rows {"table": "..."}; read {"ids": [1, 2], "condition": "...", "list": false, "final": false}; check_pictures {"candidates": ["...", "..."], "condition": "..."}; finish {"answer": "..."}"""


def planner_ops_messages(question: str, transcript: Sequence[dict[str, str]], tokens_left: int | None,
                         objects_left: int | None = None) -> list[Message]:
    history = "\n".join(f"{t['role'].upper()}: {t['content']}" for t in transcript)
    left = "no token limit" if tokens_left is None else f"{tokens_left} tokens"
    if objects_left is not None:
        left += f", {objects_left} objects"
    user = f"Question: {question}\nRead budget left: {left}\n{history}\nNext action (JSON):"
    return [{"role": "system", "content": PLANNER_OPS_SYSTEM}, {"role": "user", "content": user}]


IRCOT_DEMOS: dict[str, tuple[str, ...]] = {
    "mmqa": (
        "Wikipedia Title: Paul Greene (actor) — Television\n"
        "Paul Greene (actor) | Television | Year: 2006; Title: Shark; Role: Scott Ransom; Notes: Episode: \"LAPD Blue\"\n\n"
        "Wikipedia Title: Paul Greene (actor) — Television\n"
        "Paul Greene (actor) | Television | Year: 2017–present; Title: When Calls the Heart; Role: Dr. Carson Shepard; "
        "Notes: 25 episodes\n\n"
        "Wikipedia Title: When Calls the Heart\n"
        "The series originally debuted as a two-hour television movie pilot in October 2013, starring Maggie Grace as "
        "young teacher Elizabeth Thatcher and Stephen Amell as North West Mounted Police officer Wynn Delaney. In the "
        "television series Erin Krakow is cast as her niece, and Daniel Lissing plays a Mountie named Jack Thornton.\n\n"
        "Q: In the show that Paul Greene has the role of Dr. Carson Shepard who plays the mountie\n"
        "A: Paul Greene plays Dr. Carson Shepard in When Calls the Heart. In When Calls the Heart, Daniel Lissing plays "
        "a Mountie named Jack Thornton. So the answer is: Daniel Lissing.",
        "Wikipedia Title: Elisabeth Dermot Walsh — Television\n"
        "Elisabeth Dermot Walsh | Television | Year: 1999; Title: Unfinished Business; Role: Rachel; Notes: Recurring "
        "role; 9 episodes\n\n"
        "Wikipedia Title: Elisabeth Dermot Walsh — Television\n"
        "Elisabeth Dermot Walsh | Television | Year: 2009; Title: Holby City; Role: Elizabeth Farrington; Notes: 1 "
        "episode: \"Proceed with Caution\"\n\n"
        "Wikipedia Title: Holby City\n"
        "Holby City was created by Tony McHale and Mal Young as a spin-off from the BBC medical drama Casualty, which is "
        "set in the emergency department of the fictional Holby City Hospital. Young wanted to explore what happened to "
        "patients treated in Casualty once they were taken away to the hospital's surgical wards.\n\n"
        "Q: Was the TV show in which Elisabeth Dermot Walsh played Elizabeth Farrington set in the same hospital as the "
        "TV show \"Casualty\"?\n"
        "A: Elisabeth Dermot Walsh played Elizabeth Farrington in Holby City. Holby City follows the surgical wards of "
        "Holby City Hospital, the hospital whose emergency department Casualty is set in. So the answer is: yes.",
        "Wikipedia Title: Love in the Time of Cholera\n"
        "Love in the Time of Cholera (Spanish: El amor en los tiempos del cólera) is a novel by Colombian Nobel prize "
        "winning author Gabriel García Márquez. The novel was first published in Spanish in 1985. Alfred A. Knopf "
        "published an English translation in 1988, and an English-language movie adaptation was released in 2007.\n\n"
        "Wikipedia Title: Alicia Borrachero — Filmography | Film\n"
        "Alicia Borrachero | Filmography | Film | Year: 1992; Title: Shooting Elizabeth; Role: Pillow Mint Maid\n\n"
        "Wikipedia Title: Alicia Borrachero — Filmography | Film\n"
        "Alicia Borrachero | Filmography | Film | Year: 2007; Title: Love in the Time of Cholera; Role: Escolástica\n\n"
        "Q: In what year was Alicia Borrachero title which is what el amor en tiempos de colera stands for in english?\n"
        "A: El amor en los tiempos del cólera is Love in the Time of Cholera in English. Alicia Borrachero played "
        "Escolástica in Love in the Time of Cholera in 2007. So the answer is: 2007.",
    ),
    "musique": (
        "Wikipedia Title: Zbyněk Hejda\n"
        "Zbyněk Hejda (2 February 1930, Hradec Králové – 16 November 2013, Prague) was a Czech poet, essayist and "
        "translator (mainly from English - Emily Dickinson; and German - Georg Trakl, Gottfried Benn).\n\n"
        "Wikipedia Title: Prague astronomical clock\n"
        "The oldest part of the Orloj, the mechanical clock and astronomical dial, dates back to 1410 when it was made "
        "by clockmaker Mikuláš of Kadaň and Jan Šindel. The first recorded mention of the clock was on 9 October 1410.\n\n"
        "Q: Zbyněk Hejda died in a city with an astronomical clock that was built when?\n"
        "A: Zbyněk Hejda died in Prague. The astronomical clock in Prague was built in 1410. So the answer is: 1410.",
        "Wikipedia Title: The Florida Review\n"
        "The Florida Review is a national, non-profit literary journal published twice a year by the College of Arts and "
        "Humanities at the University of Central Florida.\n\n"
        "Wikipedia Title: Scott Frost\n"
        "Scott Andrew Frost (born January 4, 1975) is an American football coach and former player. He is currently the "
        "head coach at the University of Nebraska. He was previously the head coach at the University of Central "
        "Florida.\n\n"
        "Q: Where is the coach from the publisher of The Florida Review going?\n"
        "A: The Florida Review is published by the University of Central Florida. Scott Frost, the coach of the "
        "University of Central Florida, is going to the University of Nebraska. So the answer is: University of "
        "Nebraska.",
        "Wikipedia Title: Hafizullah Amin\n"
        "Hafizullah Amin (1 August 1929 – 27 December 1979) was an Afghan communist politician during the Cold War. "
        "After a few years as a teacher, he went to the United States to study. He would visit the United States a "
        "second time before moving permanently to Afghanistan.\n\n"
        "Wikipedia Title: Monetary policy of the United States\n"
        "In the United States, the Federal Reserve is in charge of monetary policy, and implements it primarily by "
        "performing operations that influence short-term interest rates.\n\n"
        "Wikipedia Title: Quantitative easing\n"
        "The US Federal Reserve belatedly implemented policies similar to the current quantitative easing during the "
        "Great Depression of the 1930s.\n\n"
        "Q: When was quantitative easing started by the group that controls the monetary policy in the country that "
        "Amin visited for the second time?\n"
        "A: Amin visited the United States for the second time. The Federal Reserve controls the monetary policy of the "
        "United States. The Federal Reserve started quantitative easing during the Great Depression of the 1930s. So "
        "the answer is: during the Great Depression of the 1930s.",
    ),
}


def ircot_demo_messages(question: str, passages: Sequence[tuple[str, str]], thoughts: Sequence[str],
                        demos: Sequence[str]) -> list[Message]:
    messages = ircot_messages(question, passages, thoughts)
    if not demos:
        return messages
    head = "".join(f"{d}\n\n\n" for d in demos)
    return [messages[0], {"role": "user", "content": head + messages[1]["content"]}]


def ircot_image_messages(question: str, passages: Sequence[tuple[str, str]], thoughts: Sequence[str],
                         images: dict[int, str], demos: Sequence[str] = ()) -> list[Message]:
    messages = ircot_demo_messages(question, passages, thoughts, demos)
    if not images:
        return messages
    content = messages[1]["content"]
    end = len(content) - len(ircot_messages(question, passages, thoughts)[1]["content"])
    parts: list[dict[str, Any]] = []
    cut = 0
    for i, (title, text) in enumerate(passages):
        end += len(f"Wikipedia Title: {title}\n{text}\n\n")
        if i in images:
            parts += [{"type": "text", "text": content[cut:end]}, _image_part(images[i])]
            cut = end
    parts.append({"type": "text", "text": content[cut:]})
    return [messages[0], {"role": "user", "content": parts}]


PLANNER_DEMOS: dict[str, tuple[str, ...]] = {
    "mmqa": (
        "Question: In the film where Ayelet Zuer plays Lara Lor-Van, who plays Lois Lane?\n"
        'ASSISTANT: {"thought": "First find the film in which Ayelet Zurer plays Lara Lor-Van.", "action": "search", '
        '"input": "Ayelet Zurer Lara Lor-Van film"}\n'
        "OBSERVATION: [1] midway3 | table | Ayelet Zurer — Filmography | Film: Ayelet Zurer | Filmography | Film | "
        "Year: 2013; Title: Man of Steel; Role: Lara Lor-Van\n"
        "[2] aurora | image | Ayelet Zurer: a woman in a white dress\n"
        'ASSISTANT: {"thought": "Row 1 states the role and the film.", "action": "extract", "input": "1"}\n'
        "OBSERVATION: (Ayelet Zurer | played | Lara Lor-Van)\n(Lara Lor-Van | role in | Man of Steel)\n"
        "(Man of Steel | released in | 2013)\n"
        'ASSISTANT: {"thought": "The film is Man of Steel. Now find who plays Lois Lane in it.", "action": "search", '
        '"input": "Lois Lane in Man of Steel"}\n'
        "OBSERVATION: [3] aurora | text | Amy Adams: ... In 2013 she began portraying Lois Lane in superhero films set "
        "in the DC Extended Universe.\n"
        "[4] aurora | text | Man of Steel (film): Man of Steel is a 2013 superhero film featuring the DC Comics "
        "character Superman. ... stars Henry Cavill, Amy Adams, Michael Shannon, Kevin Costner\n"
        'ASSISTANT: {"thought": "Amy Adams\'s article names her role.", "action": "extract", "input": "3"}\n'
        "OBSERVATION: (Amy Adams | portrays | Lois Lane)\n(Amy Adams | began portraying Lois Lane in | 2013)\n"
        'ASSISTANT: {"thought": "Amy Adams plays Lois Lane in Man of Steel (2013).", "action": "finish", '
        '"input": "Amy Adams"}',
        "Question: What awards did Kim Sun-a receive in 2011 for the TV series Scent of a Woman?\n"
        'ASSISTANT: {"thought": "The series\' awards table lists each award with its year and recipient.", '
        '"action": "search", "input": "Scent of a Woman TV series awards Kim Sun-a 2011"}\n'
        "OBSERVATION: [1] midway3 | table | Scent of a Woman (TV series) — Awards and nominations: ... | Year: 2011; "
        "Award: CETV Awards; Category: Top Ten Hottest Asia Award; Recipient: Kim Sun-a\n"
        "[2] midway3 | table | Scent of a Woman (TV series) — Awards and nominations: ... | Year: 2011; Award: SBS "
        "Drama Awards; Category: Top Excellence Award, Actress in a Weekend Drama; Recipient: Kim Sun-a\n"
        "[3] midway3 | table | Scent of a Woman (TV series) — Awards and nominations: ... | Year: 2012; Award: 48th "
        "Baeksang Arts Awards; Category: Best Actress (TV); Recipient: Kim Sun-a\n"
        'ASSISTANT: {"thought": "Rows 1 and 2 are awards of 2011; row 3 is of 2012.", "action": "extract", '
        '"input": "1"}\n'
        "OBSERVATION: (Kim Sun-a | received | CETV Awards)\n(CETV Awards | year | 2011)\n"
        'ASSISTANT: {"thought": "Extract the other 2011 row.", "action": "extract", "input": "2"}\n'
        "OBSERVATION: (Kim Sun-a | received | SBS Drama Awards)\n(SBS Drama Awards | year | 2011)\n"
        'ASSISTANT: {"thought": "Both awards of 2011 are found; the Baeksang award is of 2012.", "action": "finish", '
        '"input": "CETV Awards, SBS Drama Awards"}',
    ),
    "musique": (
        "Question: Who is the city where Lesnoye Sanatorium is located named after?\n"
        'ASSISTANT: {"thought": "First find the city of Lesnoye Sanatorium.", "action": "search", '
        '"input": "Lesnoye Sanatorium city"}\n'
        "OBSERVATION: [1] aurora | text | Lesnoye Sanatorium: Lesnoye Sanatorium () is the oldest medical institution in "
        "the city of Tolyatti, Russia. Its main focus is tuberculosis treatment.\n"
        'ASSISTANT: {"thought": "Passage 1 names the city.", "action": "extract", "input": "1"}\n'
        "OBSERVATION: (Lesnoye Sanatorium | located in | Tolyatti)\n(Tolyatti | country | Russia)\n"
        'ASSISTANT: {"thought": "Now find whom Tolyatti is named after.", "action": "search", '
        '"input": "Tolyatti named after"}\n'
        "OBSERVATION: [2] midway3 | text | Tolyatti: ... In 1964, the city was renamed Tolyatti (after Palmiro Togliatti, "
        "the longest-serving secretary of the Italian Communist Party).\n"
        'ASSISTANT: {"thought": "Passage 2 names its namesake.", "action": "extract", "input": "2"}\n'
        "OBSERVATION: (Tolyatti | named after | Palmiro Togliatti)\n(Tolyatti | renamed in | 1964)\n"
        'ASSISTANT: {"thought": "Tolyatti is named after Palmiro Togliatti.", "action": "finish", '
        '"input": "Palmiro Togliatti"}',
        "Question: Who won the All-Star MVP for the league that the team that Jim Wilson plays for is a part of?\n"
        'ASSISTANT: {"thought": "First find Jim Wilson\'s team.", "action": "search", "input": "Jim Wilson baseball '
        'team"}\n'
        "OBSERVATION: [1] aurora | text | Jim Wilson (first baseman): He was released by the Indians following the 1986 "
        "season. After a brief tour in the Minnesota Twins organization, Wilson signed as a free agent with the Seattle "
        "Mariners\n"
        'ASSISTANT: {"thought": "Passage 1 names his team.", "action": "extract", "input": "1"}\n'
        "OBSERVATION: (Jim Wilson | played for | Cleveland Indians)\n(Jim Wilson | released in | 1986)\n"
        'ASSISTANT: {"thought": "His team is the Cleveland Indians. Find its league.", "action": "search", '
        '"input": "Cleveland Indians league"}\n'
        "OBSERVATION: [2] midway3 | text | Cleveland Indians: The Cleveland Indians are an American professional "
        "baseball team based in Cleveland, Ohio. The Indians compete in Major League Baseball (MLB) as a member club of "
        "the American League\n"
        'ASSISTANT: {"thought": "Passage 2 names the league.", "action": "extract", "input": "2"}\n'
        "OBSERVATION: (Cleveland Indians | competes in | Major League Baseball)\n"
        'ASSISTANT: {"thought": "Now find who won the All-Star MVP of Major League Baseball.", "action": "search", '
        '"input": "Major League Baseball All-Star Game MVP"}\n'
        "OBSERVATION: [3] midway3 | text | Major League Baseball All-Star Game Most Valuable Player Award: ... Alex "
        "Bregman of the Houston Astros is the most recent MLB All-Star Game MVP, winning the award in 2018.\n"
        'ASSISTANT: {"thought": "Passage 3 names the winner.", "action": "extract", "input": "3"}\n'
        "OBSERVATION: (Alex Bregman | won | MLB All-Star Game MVP)\n(Alex Bregman | won in | 2018)\n"
        'ASSISTANT: {"thought": "Alex Bregman won the MLB All-Star Game MVP.", "action": "finish", '
        '"input": "Alex Bregman"}',
    ),
}

PLANNER_OPS_DEMOS: dict[str, tuple[str, ...]] = {
    "mmqa": (
        "Question: In the film where Ayelet Zuer plays Lara Lor-Van, who plays Lois Lane?\n"
        'ASSISTANT: {"thought": "Find the film among the actress\'s roles.", "action": "probe", '
        '"input": {"value": "Ayelet Zurer", "relation": "played Lara Lor-Van in"}}\n'
        "OBSERVATION: [1] midway3 | table | Ayelet Zurer — Filmography | Film: Ayelet Zurer | Filmography | Film | "
        "Year: 2013; Title: Man of Steel; Role: Lara Lor-Van\n"
        "[2] midway3 | table | Ayelet Zurer — Filmography | Film: Ayelet Zurer | Filmography | Film | Year: 2009; "
        "Title: Angels & Demons; Role: Vittoria Vetra\n"
        'ASSISTANT: {"thought": "Read the rows for the film.", "action": "read", '
        '"input": {"ids": [1, 2], "condition": "(Ayelet Zurer | played Lara Lor-Van in | X)"}}\n'
        "OBSERVATION: X = Man of Steel [1]\nread [1] [2]\n"
        'ASSISTANT: {"thought": "Now find who plays Lois Lane in Man of Steel.", "action": "probe", '
        '"input": {"value": "Man of Steel", "relation": "actress who plays Lois Lane in", "value_first": false}}\n'
        "OBSERVATION: [3] aurora | text | Amy Adams: ... In 2013 she began portraying Lois Lane in superhero films set "
        "in the DC Extended Universe.\n"
        "[4] aurora | text | Man of Steel (film): Man of Steel is a 2013 superhero film featuring the DC Comics "
        "character Superman. ... stars Henry Cavill, Amy Adams, Michael Shannon, Kevin Costner\n"
        'ASSISTANT: {"thought": "Read both for the actress.", "action": "read", '
        '"input": {"ids": [3, 4], "condition": "X plays Lois Lane in Man of Steel (2013)", "final": true}}\n'
        "OBSERVATION: X = Amy Adams [3]\nread [3] [4]\n"
        'ASSISTANT: {"thought": "Amy Adams plays Lois Lane in Man of Steel.", "action": "finish", '
        '"input": {"answer": "Amy Adams"}}',
        "Question: What awards did Kim Sun-a receive in 2011 for the TV series Scent of a Woman?\n"
        'ASSISTANT: {"thought": "The series\' own objects include its awards table.", "action": "title", '
        '"input": {"name": "Scent of a Woman (TV series)", "relation": "awards received by Kim Sun-a"}}\n'
        "OBSERVATION: [1] midway3 | table | Scent of a Woman (TV series) — Awards and nominations: ... | Year: 2011; "
        "Award: CETV Awards; Category: Top Ten Hottest Asia Award; Recipient: Kim Sun-a (found as Scent of a Woman (TV "
        "series))\n"
        "[2] aurora | text | Scent of a Woman (TV series): Scent of a Woman is a 2011 South Korean television drama "
        "series starring Kim Sun-a, Lee Dong-wook, Um Ki-joon and Seo Hyo-rim. (found as Scent of a Woman (TV series))\n"
        'ASSISTANT: {"thought": "List every row of the awards table.", "action": "rows", "input": {"table": 1}}\n'
        "OBSERVATION: [1] midway3 | table | ... | Year: 2011; Award: CETV Awards; Category: Top Ten Hottest Asia "
        "Award; Recipient: Kim Sun-a\n"
        "[3] midway3 | table | ... | Year: 2011; Award: 4th Korea Drama Awards; Category: Best Supporting Actor; "
        "Recipient: Um Ki-joon\n"
        "[4] midway3 | table | ... | Year: 2011; Award: SBS Drama Awards; Category: Top Excellence Award, Actress in a "
        "Weekend Drama; Recipient: Kim Sun-a\n"
        "[5] midway3 | table | ... | Year: 2012; Award: 48th Baeksang Arts Awards; Category: Best Actress (TV); "
        "Recipient: Kim Sun-a\n"
        'ASSISTANT: {"thought": "Read the rows for every award Kim Sun-a received in 2011.", "action": "read", '
        '"input": {"ids": [1, 3, 4, 5], "condition": "(Kim Sun-a | received award in 2011 | X)", "list": true, '
        '"final": true}}\n'
        "OBSERVATION: X = CETV Awards [1]\nX = SBS Drama Awards [4]\nread [1] [4] [5]\n"
        "skipped (they name none of Kim Sun-a): [3]\n"
        'ASSISTANT: {"thought": "The awards of 2011 are the CETV Awards and the SBS Drama Awards; the Baeksang award is '
        'of 2012.", "action": "finish", "input": {"answer": "CETV Awards, SBS Drama Awards"}}',
    ),
    "musique": (
        "Question: Who is the city where Lesnoye Sanatorium is located named after?\n"
        'ASSISTANT: {"thought": "Find the city through the sanatorium\'s own article.", "action": "title", '
        '"input": {"name": "Lesnoye Sanatorium", "relation": "located in city"}}\n'
        "OBSERVATION: [1] aurora | text | Lesnoye Sanatorium: Lesnoye Sanatorium () is the oldest medical institution in "
        "the city of Tolyatti, Russia. Its main focus is tuberculosis treatment. (found as Lesnoye Sanatorium)\n"
        'ASSISTANT: {"thought": "Read the city.", "action": "read", '
        '"input": {"ids": [1], "condition": "(Lesnoye Sanatorium | located in | X)"}}\n'
        "OBSERVATION: X = Tolyatti [1]\nread [1]\n"
        'ASSISTANT: {"thought": "Now find whom Tolyatti is named after.", "action": "probe", '
        '"input": {"value": "Tolyatti", "relation": "named after"}}\n'
        "OBSERVATION: [2] midway3 | text | Tolyatti: ... In 1964, the city was renamed Tolyatti (after Palmiro Togliatti, "
        "the longest-serving secretary of the Italian Communist Party).\n"
        'ASSISTANT: {"thought": "Read whom it is named after.", "action": "read", '
        '"input": {"ids": [2], "condition": "(Tolyatti | named after | X)", "final": true}}\n'
        "OBSERVATION: X = Palmiro Togliatti [2]\nread [2]\n"
        'ASSISTANT: {"thought": "Tolyatti is named after Palmiro Togliatti.", "action": "finish", '
        '"input": {"answer": "Palmiro Togliatti"}}',
        "Question: Who won the All-Star MVP for the league that the team that Jim Wilson plays for is a part of?\n"
        'ASSISTANT: {"thought": "First find Jim Wilson\'s team.", "action": "probe", '
        '"input": {"value": "Jim Wilson", "relation": "plays for team"}}\n'
        "OBSERVATION: [1] aurora | text | Jim Wilson (first baseman): He was released by the Indians following the 1986 "
        "season. After a brief tour in the Minnesota Twins organization, Wilson signed as a free agent with the Seattle "
        "Mariners\n"
        'ASSISTANT: {"thought": "Read his team.", "action": "read", '
        '"input": {"ids": [1], "condition": "(Jim Wilson | played for | X)"}}\n'
        "OBSERVATION: X = Cleveland Indians [1]\nread [1]\n"
        'ASSISTANT: {"thought": "Find the league of the Cleveland Indians.", "action": "title", '
        '"input": {"name": "Cleveland Indians", "relation": "league"}}\n'
        "OBSERVATION: [2] midway3 | text | Cleveland Indians: The Cleveland Indians are an American professional "
        "baseball team based in Cleveland, Ohio. The Indians compete in Major League Baseball (MLB) as a member club of "
        "the American League (found as Cleveland Indians)\n"
        'ASSISTANT: {"thought": "Read the league.", "action": "read", '
        '"input": {"ids": [2], "condition": "(Cleveland Indians | competes in | X)"}}\n'
        "OBSERVATION: X = Major League Baseball [2]\nread [2]\n"
        'ASSISTANT: {"thought": "Now find who won the All-Star MVP of Major League Baseball.", "action": "probe", '
        '"input": {"value": "Major League Baseball", "relation": "All-Star Game most valuable player"}}\n'
        "OBSERVATION: [3] midway3 | text | Major League Baseball All-Star Game Most Valuable Player Award: ... Alex "
        "Bregman of the Houston Astros is the most recent MLB All-Star Game MVP, winning the award in 2018.\n"
        'ASSISTANT: {"thought": "Read the winner.", "action": "read", '
        '"input": {"ids": [3], "condition": "(X | won the All-Star MVP of | Major League Baseball)", "final": true}}\n'
        "OBSERVATION: X = Alex Bregman [3]\nread [3]\n"
        'ASSISTANT: {"thought": "Alex Bregman won the MLB All-Star Game MVP.", "action": "finish", '
        '"input": {"answer": "Alex Bregman"}}',
    ),
}

REACT_SYSTEM = """You answer a question by reasoning step by step and searching a federation of databases that hold passages, table rows and images (an image appears as its caption).
Tools:
- search(query): returns the most relevant objects for the query across every database, each with an id, its title and its text. Search with any text: the question, an entity you found, or the fact you need next.
- finish(answer): ends with the final short answer.
Reply with JSON only: {"thought": "...", "action": "search" | "finish", "input": "..."}"""

REACT_DEMOS: dict[str, tuple[str, ...]] = {
    "mmqa": (
        "Question: In the film where Ayelet Zuer plays Lara Lor-Van, who plays Lois Lane?\n"
        'ASSISTANT: {"thought": "First find the film in which Ayelet Zurer plays Lara Lor-Van.", "action": "search", '
        '"input": "Ayelet Zurer Lara Lor-Van film"}\n'
        "OBSERVATION: [1] Ayelet Zurer — Filmography: Year: 2013; Title: Man of Steel; Role: Lara Lor-Van\n"
        "[2] Ayelet Zurer: a woman in a white dress\n"
        'ASSISTANT: {"thought": "Row 1 says she plays Lara Lor-Van in Man of Steel. Now find who plays Lois Lane in '
        'it.", "action": "search", "input": "Lois Lane in Man of Steel"}\n'
        "OBSERVATION: [3] Amy Adams: ... In 2013 she began portraying Lois Lane in superhero films set in the DC "
        "Extended Universe.\n"
        "[4] Man of Steel (film): Man of Steel is a 2013 superhero film featuring the DC Comics character Superman. "
        "... stars Henry Cavill, Amy Adams, Michael Shannon, Kevin Costner\n"
        'ASSISTANT: {"thought": "Amy Adams plays Lois Lane in Man of Steel (2013).", "action": "finish", '
        '"input": "Amy Adams"}',
        "Question: What awards did Kim Sun-a receive in 2011 for the TV series Scent of a Woman?\n"
        'ASSISTANT: {"thought": "The series\' awards table lists each award with its year and recipient.", '
        '"action": "search", "input": "Scent of a Woman TV series awards Kim Sun-a 2011"}\n'
        "OBSERVATION: [1] Scent of a Woman (TV series) — Awards and nominations: Year: 2011; Award: CETV Awards; "
        "Category: Top Ten Hottest Asia Award; Recipient: Kim Sun-a\n"
        "[2] Scent of a Woman (TV series) — Awards and nominations: Year: 2011; Award: SBS Drama Awards; Category: Top "
        "Excellence Award, Actress in a Weekend Drama; Recipient: Kim Sun-a\n"
        "[3] Scent of a Woman (TV series) — Awards and nominations: Year: 2012; Award: 48th Baeksang Arts Awards; "
        "Category: Best Actress (TV); Recipient: Kim Sun-a\n"
        'ASSISTANT: {"thought": "Rows 1 and 2 are her awards of 2011; row 3 is of 2012.", "action": "finish", '
        '"input": "CETV Awards, SBS Drama Awards"}',
    ),
    "musique": (
        "Question: Who is the city where Lesnoye Sanatorium is located named after?\n"
        'ASSISTANT: {"thought": "First find the city of Lesnoye Sanatorium.", "action": "search", '
        '"input": "Lesnoye Sanatorium city"}\n'
        "OBSERVATION: [1] Lesnoye Sanatorium: Lesnoye Sanatorium () is the oldest medical institution in the city of "
        "Tolyatti, Russia. Its main focus is tuberculosis treatment.\n"
        'ASSISTANT: {"thought": "The sanatorium is in Tolyatti. Now find whom Tolyatti is named after.", '
        '"action": "search", "input": "Tolyatti named after"}\n'
        "OBSERVATION: [2] Tolyatti: ... In 1964, the city was renamed Tolyatti (after Palmiro Togliatti, the "
        "longest-serving secretary of the Italian Communist Party).\n"
        'ASSISTANT: {"thought": "Tolyatti is named after Palmiro Togliatti.", "action": "finish", '
        '"input": "Palmiro Togliatti"}',
        "Question: Who won the All-Star MVP for the league that the team that Jim Wilson plays for is a part of?\n"
        'ASSISTANT: {"thought": "First find Jim Wilson\'s team.", "action": "search", "input": "Jim Wilson baseball '
        'team"}\n'
        "OBSERVATION: [1] Jim Wilson (first baseman): He was released by the Indians following the 1986 season. After "
        "a brief tour in the Minnesota Twins organization, Wilson signed as a free agent with the Seattle Mariners\n"
        'ASSISTANT: {"thought": "He played for the Cleveland Indians. Find its league.", "action": "search", '
        '"input": "Cleveland Indians league"}\n'
        "OBSERVATION: [2] Cleveland Indians: The Cleveland Indians are an American professional baseball team based in "
        "Cleveland, Ohio. The Indians compete in Major League Baseball (MLB) as a member club of the American League\n"
        'ASSISTANT: {"thought": "The league is Major League Baseball. Find who won its All-Star MVP.", '
        '"action": "search", "input": "Major League Baseball All-Star Game MVP"}\n'
        "OBSERVATION: [3] Major League Baseball All-Star Game Most Valuable Player Award: ... Alex Bregman of the "
        "Houston Astros is the most recent MLB All-Star Game MVP, winning the award in 2018.\n"
        'ASSISTANT: {"thought": "Alex Bregman won the MLB All-Star Game MVP.", "action": "finish", '
        '"input": "Alex Bregman"}',
    ),
}


def react_budget(tokens_left: int | None) -> str:
    return "Budget left: no token limit" if tokens_left is None else f"Budget left: {max(0, tokens_left)} tokens"


# HotpotQA and 2Wiki borrow MuSiQue's demos, M3SciQA those of MMQA
for _demos in (IRCOT_DEMOS, PLANNER_DEMOS, PLANNER_OPS_DEMOS, REACT_DEMOS):
    for _name in ("hotpotqa", "twowiki"):
        _demos.setdefault(_name, _demos["musique"])
    _demos.setdefault("m3sciqa", _demos["mmqa"])

AGENT_DEMO_HEADER = "Example episodes on other questions (each reply is one JSON action; the observations are shortened):"
AGENT_V14_USER = "Question: {question}\n{history}{budget}\nNext action (JSON):"


def planner_budget(budget_left: int) -> str:
    return f"Extraction budget left: {budget_left}"


def ops_budget(tokens_left: int | None, objects_left: int | None = None) -> str:
    left = "no token limit" if tokens_left is None else f"{tokens_left} tokens"
    if objects_left is not None:
        left += f", {objects_left} objects"
    return f"Read budget left: {left}"


def agent_messages(system: str, question: str, transcript: Sequence[dict[str, str]], budget: str,
                   demos: Sequence[str] = (), layout: str = "v14") -> list[Message]:
    if layout not in ("v13", "v14"):
        raise ValueError(f"unknown agent prompt layout {layout!r}")
    history = "\n".join(f"{t['role'].upper()}: {t['content']}" for t in transcript)
    if demos:
        system = f"{system}\n\n{AGENT_DEMO_HEADER}\n\n" + "\n\n".join(demos)
    if layout == "v13":
        user = f"Question: {question}\n{budget}\n{history}\nNext action (JSON):"
    else:
        user = AGENT_V14_USER.format(question=question, history=f"{history}\n" if history else "", budget=budget)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
