from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator
from typing import Any

from fedquest.data.chunking import WORDS_PER_TOKEN, chunk_text
from fedquest.paths import raw_dir
from fedquest.types import ObjectRecord, Question
from fedquest.utils import stable_hash

FAMILIES = ("2hop", "3hop", "4hop")
QUESTIONS = {"hipporag": "musique.json", "train": "musique_ans_v1.0_train.jsonl"}
CORPUS = "musique_corpus.json"
TUNE_N = 50
SPLITS = ("confirm", "tune", "confirm14", "confirm18")
SUBSETS = frozenset({"bridged", "bridge_textual", "join_all", "multihop"})
_TYPE = re.compile(r"[234]hop\d?")
_REF = re.compile(r"#(\d+)")


def is_type(qtype: str) -> bool:
    return _TYPE.fullmatch(qtype) is not None


def family(qtype: str) -> str:
    return qtype[:4] if is_type(qtype) else "other"


def subsets(qtype: str) -> set[str]:
    return set(SUBSETS) if is_type(qtype) else set()


def doc_id(title: str, text: str) -> str:
    return stable_hash(title, text)


def refs(question: str) -> list[int]:
    return sorted({int(n) for n in _REF.findall(question)})


def iter_raw(split: str) -> Iterator[dict[str, Any]]:
    if split not in QUESTIONS:
        raise ValueError(f"MuSiQue has the question splits {sorted(QUESTIONS)}, not {split!r}")
    path = raw_dir("musique") / QUESTIONS[split]
    if path.suffix == ".jsonl":
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
    else:
        yield from json.loads(path.read_text(encoding="utf-8"))


def to_question(d: dict[str, Any]) -> Question:
    paragraphs = {p["idx"]: p for p in d["paragraphs"]}
    hops = []
    for i, h in enumerate(d["question_decomposition"]):
        p = paragraphs[h["paragraph_support_idx"]]
        after = [j for j in refs(h["question"]) if j <= i]
        hops.append({"question": h["question"], "answer": h["answer"],
                     "doc_id": doc_id(p["title"], p["paragraph_text"]), "after": after})
    supporting = [doc_id(p["title"], p["paragraph_text"]) for p in d["paragraphs"] if p.get("is_supporting")]
    qtype = d["id"].split("__", 1)[0]
    return Question(
        qid=d["id"],
        dataset="musique",
        question=d["question"],
        answers=list(dict.fromkeys([d["answer"], *d.get("answer_aliases", [])])),
        qtype=qtype,
        gold_doc_ids=list(dict.fromkeys([*(h["doc_id"] for h in hops), *supporting])),
        bridge=list(dict.fromkeys(h["answer"] for h in hops[:-1])),
        meta={"family": family(qtype), "hops": hops},
    )


def load_questions(split: str = "hipporag") -> list[Question]:
    return [to_question(d) for d in iter_raw(split)]


def single_hops(d: dict[str, Any]) -> frozenset[int]:
    return frozenset(int(h["id"]) for h in d["question_decomposition"])


def paragraphs(split: str, qids: Iterable[str]) -> list[tuple[str, str]]:
    wanted = set(qids)
    return [(p["title"], p["paragraph_text"]) for d in iter_raw(split) if d["id"] in wanted for p in d["paragraphs"]]


def load_corpus() -> list[tuple[str, str]]:
    return [(c["title"], c["text"]) for c in json.loads((raw_dir("musique") / CORPUS).read_text(encoding="utf-8"))]


def passages(extra: Iterable[tuple[str, str]] = ()) -> tuple[list[tuple[str, str]], int, int]:
    corpus = load_corpus()
    out = list(dict.fromkeys(corpus))
    published = len(out)
    out = list(dict.fromkeys([*out, *extra]))
    return out, published, len(out) - published


def passage_objects(title: str, text: str, chunk_tokens: int = 256, overlap: int = 32) -> list[ObjectRecord]:
    did = doc_id(title, text)
    pieces = [text] if len(text.split()) <= max(8, int(chunk_tokens * WORDS_PER_TOKEN)) else \
        chunk_text(text, chunk_tokens, overlap)
    return [ObjectRecord(oid=f"musique/text/{did}#{i}", doc_id=did, modality="text", title=title, text=piece)
            for i, piece in enumerate(pieces)]


def explore_gold(q: Question) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    hops = q.meta.get("hops") or []
    rounds = {"first": [h for h in hops if not h["after"]], "final": [h for h in hops if 1 in h["after"]]}
    gold = {r: sorted({f"text:{h['doc_id']}" for h in hs}) for r, hs in rounds.items()}
    values = {r: list(dict.fromkeys(h["answer"] for h in hs)) for r, hs in rounds.items()}
    return gold, values


def bridges(q: Question) -> list[tuple[str, str]]:
    hops = q.meta.get("hops") or []
    return [(hops[j - 1]["answer"], h["doc_id"]) for h in hops for j in h["after"] if 0 < j <= len(hops)]
