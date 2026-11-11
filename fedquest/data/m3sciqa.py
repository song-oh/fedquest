# M3SciQA: from a paper's figure or table to passages of a reference paper
from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from collections.abc import Iterator, Sequence
from functools import cache
from pathlib import Path
from typing import Any

from fedquest.data.chunking import chunk_text
from fedquest.paths import raw_dir
from fedquest.types import ObjectRecord, Question
from fedquest.utils import stable_hash

FOLDER = "m3sciqa"
REVISION = "99946c354cad84294766e58558f249f71e8bfffc"
FILES = {
    "validation": ("data/validation-00000-of-00001.parquet",
                   "d669168d6280fccdacad25f4563d69e93f084f373a623fccb8a4b78c490160b0"),
    "test": ("data/test-00000-of-00001.parquet", "0d14a73b56f775ba5b556c8d945237b4b91f089675e38140048a9d9899000dbe"),
}
COMMIT = "7634ec6220adf522788fe0a078fbf23a25bee472"
PAPERS = ("paper_full_content.json", "3720779dc95130edc5589211f9314183e2322f425e1f1f00ebceeec00dd21b56")
PAPERS_URL = f"https://raw.githubusercontent.com/yale-nlp/M3SciQA/{COMMIT}/{PAPERS[0]}"
FIGURES = "figures"
IMAGES = 82
EXCLUDE = frozenset({("2310.07188", "CFQ_table.png")})
TYPES = ("m3sci_table", "m3sci_figure")
SUBSETS = frozenset({"bridged", "join_all", "multihop"})
LEVELS = ("passage", "fuzzy", "numbers", "paper")
LOCATED = frozenset({"passage", "fuzzy", "numbers"})
PREFIX = 100
FUZZY = 0.5
COVER = 0.9
_ANCHORS = Path(__file__).with_name("m3sciqa_anchors.json")
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_WORD = re.compile(r"[a-z0-9]+")
_DECIMAL = re.compile(r"(?<![\d.])\d+\.\d+(?![\d.])")
_SUFFIX = re.compile(r"_\d+$")


def url(split: str) -> str:
    return f"https://huggingface.co/datasets/yale-nlp/M3SciQA/resolve/{REVISION}/{FILES[split][0]}"


def is_type(qtype: str) -> bool:
    return qtype in TYPES


def family(qtype: str) -> str:
    return qtype if is_type(qtype) else "other"


def subsets(qtype: str) -> set[str]:
    return set(SUBSETS) if is_type(qtype) else set()


def _ws(text: str | None) -> str:
    return " ".join((text or "").split())


def _plain(text: str) -> str:
    out = []
    for ch in unicodedata.normalize("NFKD", text):
        name = unicodedata.name(ch, "")
        if unicodedata.combining(ch):
            continue
        out.append(f" {name.replace(' SYMBOL', '').split()[-1].lower()} " if name.startswith("GREEK") else
                   " in " if ch == "∈" else ch)
    return "".join(out)


def _grams(text: str) -> set[tuple[str, ...]]:
    words = _WORD.findall(_plain(text).lower())
    return {tuple(words[i : i + 4]) for i in range(len(words) - 3)}


def qid(anchor: str, figure: str, question: str) -> str:
    return f"{FOLDER}-{stable_hash(anchor, figure, question)[:12]}"


def image_doc_id(anchor: str, figure: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", f"{anchor}_{stable_hash(anchor, figure)[:8]}")


def figure_path(anchor: str, figure: str) -> str:
    if not (_NAME.fullmatch(anchor) and _NAME.fullmatch(figure)):
        raise ValueError(f"M3SciQA: unsafe image name {anchor!r}/{figure!r}")
    return f"{FIGURES}/{anchor}/{figure}"


def _read(split: str) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    path = raw_dir(FOLDER) / f"{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: fedquest data download m3sciqa")
    return pq.read_table(path).to_pylist()


@cache
def _rows(split: str) -> tuple[dict[str, Any], ...]:
    return tuple({**r, "figure": r["figure"]["path"]} for r in _read(split))


def figures() -> Iterator[tuple[str, str, bytes]]:
    seen: dict[tuple[str, str], bytes] = {}
    for split in FILES:
        for r in _read(split):
            key, data = (r["anchor_arxiv_id"], r["figure"]["path"]), r["figure"]["bytes"]
            if key in EXCLUDE:
                continue
            if seen.setdefault(key, data) != data:
                raise ValueError(f"M3SciQA: two different images named {key[0]}/{key[1]}")
    for (anchor, figure), data in sorted(seen.items()):
        yield anchor, figure, data


@cache
def load_papers() -> dict[str, dict[str, Any]]:
    path = raw_dir(FOLDER) / PAPERS[0]
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: fedquest data download m3sciqa")
    return json.loads(path.read_text(encoding="utf-8"))


@cache
def anchor_titles() -> dict[str, str]:
    return {a: _ws(t) for a, t in json.loads(_ANCHORS.read_text(encoding="utf-8"))["titles"].items()}


def _anchor_title(anchor: str) -> str:
    title = anchor_titles().get(anchor)
    if not title:
        raise KeyError(f"M3SciQA: anchor paper {anchor} has no title in {_ANCHORS.name}")
    return title


def _heading(name: str | None) -> str:
    head = _ws(_SUFFIX.sub("", name or ""))
    return head if re.search(r"[A-Za-z]", head) and not re.search(r"[\\{}$]", head) and len(head) <= 120 else ""


def passage_objects(arxiv: str, paper: dict[str, Any], chunk_tokens: int = 256,
                    overlap: int = 32) -> list[ObjectRecord]:
    title = _ws(paper.get("title"))
    pieces: list[tuple[int | str, int, str]] = [
        ("abstract", i, chunk) for i, chunk in enumerate(chunk_text(_ws(paper.get("abstract")), chunk_tokens, overlap))]
    for s, section in enumerate(paper.get("full_text") or []):
        body, head = _ws(section.get("paragraphs")), _heading(section.get("section_name"))
        text = f"{head}: {body}" if head and body else body
        pieces += [(s, i, chunk) for i, chunk in enumerate(chunk_text(text, chunk_tokens, overlap))]
    out = []
    for part, i, text in pieces:
        did = stable_hash(FOLDER, arxiv, part, i)[:16]
        out.append(ObjectRecord(oid=f"{FOLDER}/text/{did}#0", doc_id=did, modality="text", title=title, text=text))
    return out


@cache
def corpus(chunk_tokens: int = 256, overlap: int = 32) -> dict[str, tuple[ObjectRecord, ...]]:
    papers = load_papers()
    return {a: tuple(passage_objects(a, papers[a], chunk_tokens, overlap)) for a in sorted(papers)}


@cache
def image_kinds() -> dict[tuple[str, str], str]:
    modals: dict[tuple[str, str], Counter[str]] = {}
    for split in FILES:
        for r in _rows(split):
            if (r["anchor_arxiv_id"], r["figure"]) not in EXCLUDE:
                modals.setdefault((r["anchor_arxiv_id"], r["figure"]), Counter())[r["modal"]] += 1
    out = {}
    for key, labels in sorted(modals.items()):
        words = set(Path(key[1]).stem.lower().split("_"))
        out[key] = "table" if "table" in words else "figure" if "figure" in words else labels.most_common(1)[0][0]
    return out


@cache
def image_objects() -> tuple[ObjectRecord, ...]:
    out = []
    for (anchor, figure), kind in image_kinds().items():
        title, did = _anchor_title(anchor), image_doc_id(anchor, figure)
        out.append(ObjectRecord(oid=f"{FOLDER}/image/{did}", doc_id=did, modality="image", title=title,
                                text=f'{kind.capitalize()} of the paper "{title}"', image_path=figure_path(anchor, figure)))
    return tuple(out)


def _cover(chosen: list[str], held: dict[str, set[tuple[str, ...]]], grams: set[tuple[str, ...]]) -> list[str]:
    out, covered = list(chosen), set().union(*(held[d] for d in chosen))
    while len(covered) < COVER * len(grams):
        rest = grams - covered
        best = max((d for d in held if d not in out), key=lambda d: len(held[d] & rest), default=None)
        if best is None or len(held[best] & rest) < 0.5 * len(rest):
            break
        out.append(best)
        covered |= held[best]
    return out


def _numbers(evidence: str, passages: Sequence[ObjectRecord]) -> list[str]:
    numbers = set(_DECIMAL.findall(evidence))
    if not numbers:
        return []
    found = [o.doc_id for o in passages
             if all(re.search(rf"(?<![\d.]){re.escape(n)}(?!\d)", o.text) for n in numbers)]
    return found if len(found) <= 2 else []


# evidence found by exact prefix, then 4-gram overlap, then its decimals; else the whole paper
def locate(evidence: str, passages: Sequence[ObjectRecord]) -> tuple[str, list[str]]:
    grams = _grams(evidence)
    held = {o.doc_id: grams & _grams(o.text) for o in passages} if grams else {}
    key = _ws(evidence).lower()[:PREFIX]
    exact = [o.doc_id for o in passages if grams and key in _ws(o.text).lower()]
    if exact:
        return "passage", _cover(sorted(exact, key=lambda d: -len(held[d])), held, grams)
    fuzzy = sorted((d for d, g in held.items() if len(g) >= FUZZY * len(grams)), key=lambda d: -len(held[d]))
    if not fuzzy and held:
        joint = _cover([max(held, key=lambda d: len(held[d]))], held, grams)
        if len(set().union(*(held[d] for d in joint))) >= FUZZY * len(grams):
            fuzzy = joint
    if fuzzy:
        return "fuzzy", _cover(fuzzy, held, grams)
    numbers = _numbers(evidence, passages)
    if numbers:
        return "numbers", numbers
    return "paper", [o.doc_id for o in passages]


def to_question(row: dict[str, Any], passages: Sequence[ObjectRecord]) -> Question:
    anchor, figure, modal, reference = row["anchor_arxiv_id"], row["figure"], row["modal"], row["reference_arxiv_id"]
    if not passages:
        raise ValueError(f"M3SciQA: reference paper {reference} has no passages")
    image, kind = image_doc_id(anchor, figure), image_kinds()[(anchor, figure)]
    level, evidence = locate(row["evidence_reference"], passages)
    qtype = f"m3sci_{kind}"
    hops = [{"question": row["question_anchor"], "answer": row["answer_anchor"], "doc_id": image, "modality": "image",
             "after": []},
            {"question": row["question_reference"], "answer": row["answer"], "doc_id": evidence[0],
             "doc_ids": evidence, "modality": "text", "after": [1]}]
    return Question(
        qid=qid(anchor, figure, row["question"]),
        dataset=FOLDER,
        question=f'This question is about a {kind}, shown as an image, in the paper "{_anchor_title(anchor)}". '
                 f'{row["question"]}',
        answers=[row["answer"]],
        qtype=qtype,
        gold_doc_ids=[image, *evidence],
        bridge=[row["answer_anchor"]],
        meta={"family": family(qtype), "hops": hops, "anchor": anchor, "reference": reference, "figure": figure,
              "modal": modal, "anchor_reasoning_type": row["anchor_reasoning_type"],
              "reference_reasoning_type": row["reference_reasoning_type"], "evidence_level": level,
              "gold_oids": [f"{FOLDER}/image/{image}", *(f"{FOLDER}/text/{d}#0" for d in evidence)]},
    )


def load_questions(split: str = "validation", chunk_tokens: int = 256, overlap: int = 32) -> list[Question]:
    papers = corpus(chunk_tokens, overlap)
    out: dict[str, Question] = {}
    for row in _rows(split):
        if row["answer"].strip() and (row["anchor_arxiv_id"], row["figure"]) not in EXCLUDE:
            q = to_question(row, papers.get(row["reference_arxiv_id"], ()))
            out.setdefault(q.qid, q)
    return list(out.values())


def explore_gold(q: Question) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    image, passage = q.meta["hops"][0], q.meta["hops"][-1]
    return ({"first": [f"image:{image['doc_id']}"], "final": [f"text:{d}" for d in passage["doc_ids"]]},
            {"first": [image["answer"]], "final": [passage["answer"]]})
