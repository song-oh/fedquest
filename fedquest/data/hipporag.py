from __future__ import annotations

import json
import re
from functools import cache
from typing import Any

from fedquest.data.chunking import WORDS_PER_TOKEN, chunk_text
from fedquest.kg.normalize import norm_entity
from fedquest.paths import raw_dir
from fedquest.types import ObjectRecord, Question
from fedquest.utils import stable_hash

DATASETS = {
    "hotpotqa": ("hotpotqa", "hotpotqa.json", "hotpotqa_corpus.json"),
    "twowiki": ("2wikimultihopqa", "2wikimultihopqa.json", "2wikimultihopqa_corpus.json"),
}
TYPES = {
    "hotpotqa": ("bridge", "comparison"),
    "twowiki": ("compositional", "inference", "comparison", "bridge_comparison"),
}
BRIDGED = frozenset({"bridge", "compositional", "inference", "bridge_comparison"})
_QUALIFIER = re.compile(r"\s*\([^()]*\)\s*$")


def is_dataset(name: str) -> bool:
    return name in DATASETS


def is_type(qtype: str) -> bool:
    return any(qtype in types for types in TYPES.values())


def family(qtype: str) -> str:
    return qtype if is_type(qtype) else "other"


def subsets(qtype: str) -> set[str]:
    if not is_type(qtype):
        return set()
    if qtype in BRIDGED:
        return {"bridged", "bridge_textual", "join_all", "multihop"}
    return {"multihop"}


def doc_id(title: str, text: str) -> str:
    return stable_hash(title, text)


def entity_of(title: str) -> str:
    bare = _QUALIFIER.sub("", title).strip()
    return bare or title


def _raw(dataset: str, which: int) -> Any:
    folder, *files = DATASETS[dataset]
    return json.loads((raw_dir(folder) / files[which]).read_text(encoding="utf-8"))


@cache
def load_corpus(dataset: str) -> tuple[tuple[str, str], ...]:
    return tuple((c["title"], c["text"]) for c in _raw(dataset, 1))


@cache
def _by_title(dataset: str) -> dict[str, tuple[str, str]]:
    out: dict[str, tuple[str, str]] = {}
    for title, text in load_corpus(dataset):
        out.setdefault(title, (title, text))
    return out


def _by_norm(dataset: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for title in _by_title(dataset):
        out.setdefault(norm_entity(title), title)
    return out


def _doc(dataset: str, title: str) -> str:
    passage = _by_title(dataset).get(title)
    if passage is None:
        raise KeyError(f"{dataset}: no corpus passage titled {title!r}")
    return doc_id(*passage)


def _mentions(text: str, title: str) -> bool:
    name = norm_entity(entity_of(title))
    return bool(name) and re.search(rf"(?<!\w){re.escape(name)}(?!\w)", norm_entity(text)) is not None


def _hotpot(d: dict[str, Any]) -> Question:
    titles = list(dict.fromkeys(t for t, _ in d["supporting_facts"]))
    texts = {t: _by_title("hotpotqa")[t][1] for t in titles}
    qtype = d["type"]
    rule = "parallel"
    if qtype == "bridge" and len(titles) == 2:
        a, b = titles
        named = [x for x, y in ((a, b), (b, a)) if _mentions(texts[y], x)]
        answer = norm_entity(d["answer"])
        holds = [t for t in titles if answer and answer in norm_entity(texts[t])]
        if len(named) == 1:
            second, rule = named[0], "title"
        elif len(holds) == 1:
            second, rule = holds[0], "answer"
        else:
            second, rule = b, "order"
        first = a if second == b else b
        hops = [{"question": None, "answer": entity_of(second), "doc_id": _doc("hotpotqa", first), "after": []},
                {"question": None, "answer": d["answer"], "doc_id": _doc("hotpotqa", second), "after": [1]}]
    else:
        hops = [{"question": None, "answer": None, "doc_id": _doc("hotpotqa", t), "after": []} for t in titles]
    return Question(
        qid=d["_id"],
        dataset="hotpotqa",
        question=d["question"],
        answers=[d["answer"]],
        qtype=qtype,
        gold_doc_ids=list(dict.fromkeys(h["doc_id"] for h in hops)),
        bridge=[h["answer"] for h in hops[:-1] if h["answer"] and any(1 in g["after"] for g in hops[1:])],
        meta={"family": qtype, "hops": hops, "hop_rule": rule, "level": d.get("level", "")},
    )


def _twowiki(d: dict[str, Any]) -> Question:
    support = list(dict.fromkeys(t for t, _ in d["supporting_facts"]))
    norms = _by_norm("twowiki")
    evidences = d.get("evidences") or []

    def page(subject: str, i: int) -> str:
        key = norm_entity(subject)
        for t in support:
            if norm_entity(t) == key:
                return t
        if key in norms:
            return norms[key]
        return support[min(i, len(support) - 1)]

    hops = []
    for i, (s, r, o) in enumerate(evidences):
        after = [j + 1 for j, (_, _, o2) in enumerate(evidences[:i]) if norm_entity(o2) == norm_entity(s)]
        hops.append({"question": f"{s} | {r}", "answer": o, "doc_id": _doc("twowiki", page(s, i)), "after": after})
    if not hops:
        hops = [{"question": None, "answer": None, "doc_id": _doc("twowiki", t), "after": []} for t in support]
    referenced = {j for h in hops for j in h["after"]}
    gold = [h["doc_id"] for h in hops] + [_doc("twowiki", t) for t in support]
    return Question(
        qid=d["_id"],
        dataset="twowiki",
        question=d["question"],
        answers=[d["answer"]],
        qtype=d["type"],
        gold_doc_ids=list(dict.fromkeys(gold)),
        bridge=list(dict.fromkeys(hops[j - 1]["answer"] for j in sorted(referenced) if hops[j - 1]["answer"])),
        meta={"family": d["type"], "hops": hops},
    )


def load_questions(dataset: str) -> list[Question]:
    convert = {"hotpotqa": _hotpot, "twowiki": _twowiki}[dataset]
    return [convert(d) for d in _raw(dataset, 0)]


def supporting_titles(dataset: str) -> dict[str, frozenset[str]]:
    return {d["_id"]: frozenset(t for t, _ in d["supporting_facts"]) for d in _raw(dataset, 0)}


def passage_objects(dataset: str, title: str, text: str, chunk_tokens: int = 256,
                    overlap: int = 32) -> list[ObjectRecord]:
    did = doc_id(title, text)
    pieces = [text] if len(text.split()) <= max(8, int(chunk_tokens * WORDS_PER_TOKEN)) else \
        chunk_text(text, chunk_tokens, overlap)
    return [ObjectRecord(oid=f"{dataset}/text/{did}#{i}", doc_id=did, modality="text", title=title, text=piece)
            for i, piece in enumerate(pieces)]
