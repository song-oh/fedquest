from __future__ import annotations

from functools import cache
from typing import Any

from fedquest.paths import raw_dir
from fedquest.types import ObjectRecord, Question
from fedquest.utils import stable_hash

FOLDER = "msmarco"
REVISION = "a47ee7aae8d7d466ba15f9f0bfac3b3681087b3a"
FILES = {
    "train": ("v1.1/train-00000-of-00001.parquet", "11d5e51d70ce87b47b1c132ef7c55f4cfb93b22c122058aa0591a65c4c0dc1f1"),
    "validation": ("v1.1/validation-00000-of-00001.parquet",
                   "67aca2e1b2fffe418a6bd317142887c90973c9c98dd8d940eaae3f8521daed7c"),
    "test": ("v1.1/test-00000-of-00001.parquet", "28721adaa7767035faa26c647052db537ecc0675ccd85c6d3a559a157bb14f82"),
}
TYPES = ("description", "numeric", "entity", "location", "person")
NO_ANSWER = "No Answer Present."


def url(split: str) -> str:
    return f"https://huggingface.co/datasets/microsoft/ms_marco/resolve/{REVISION}/{FILES[split][0]}"


def is_type(qtype: str) -> bool:
    return qtype in TYPES


def family(qtype: str) -> str:
    return qtype if is_type(qtype) else "other"


def subsets(qtype: str) -> set[str]:
    return set()


def doc_id(title: str, text: str) -> str:
    return stable_hash(title, text)[:16]


def oid(title: str, text: str) -> str:
    return f"msmarco/text/{doc_id(title, text)}#0"


@cache
def _rows(split: str) -> tuple[dict[str, Any], ...]:
    import pyarrow.parquet as pq

    path = raw_dir(FOLDER) / f"{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: fedquest data download msmarco")
    return tuple(pq.read_table(path).to_pylist())


def _passages(row: dict[str, Any]) -> list[tuple[str, str, bool]]:
    p = row["passages"]
    return [(u, t, bool(s)) for u, t, s in zip(p["url"], p["passage_text"], p["is_selected"], strict=True)]


@cache
def load_corpus() -> tuple[tuple[str, str], ...]:
    seen: dict[tuple[str, str], None] = {}
    for split in ("train", "validation", "test"):
        for row in _rows(split):
            for u, t, _ in _passages(row):
                if t.strip():
                    seen.setdefault((u, t), None)
    return tuple(seen)


def _question(row: dict[str, Any]) -> Question | None:
    answers = [a for a in row.get("answers") or [] if a and a != NO_ANSWER]
    gold = [oid(u, t) for u, t, sel in _passages(row) if sel and t.strip()]
    if not answers or not gold:
        return None
    return Question(qid=f"msmarco-{row['query_id']}", dataset="msmarco", question=row["query"], answers=answers,
                    qtype=row["query_type"], gold_doc_ids=[g.split("/")[-1].split("#")[0] for g in gold],
                    meta={"family": family(row["query_type"]), "gold_oids": gold,
                          "hops": [{"question": row["query"], "answer": answers[0],
                                   "doc_id": gold[0].split("/")[-1].split("#")[0]}]})


def load_questions(split: str) -> list[Question]:
    return [q for q in map(_question, _rows(split)) if q is not None]


def passage_objects(title: str, text: str) -> list[ObjectRecord]:
    did = doc_id(title, text)
    return [ObjectRecord(oid=f"msmarco/text/{did}#0", doc_id=did, modality="text", title=title, text=text)]
