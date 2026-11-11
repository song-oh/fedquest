from __future__ import annotations

import gzip
import json
import re
from collections.abc import Iterator
from typing import Any

from fedquest.data import hipporag, m3sciqa, msmarco, musique
from fedquest.data.chunking import chunk_text
from fedquest.paths import raw_dir
from fedquest.types import ObjectRecord, Question

FAMILIES = ("single", "table", "compose", "intersect", "compare")
_MODALITY = {"TextQ": "text", "TableQ": "table", "ImageQ": "image", "ImageListQ": "image"}


def _read_gz(name: str) -> Iterator[dict[str, Any]]:
    with gzip.open(raw_dir("mmqa") / name, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def family(qtype: str) -> str:
    if qtype in ("TextQ", "ImageQ"):
        return "single"
    if qtype in ("TableQ", "ImageListQ"):
        return "table"
    for prefix in ("Compose", "Intersect", "Compare"):
        if qtype.startswith(prefix):
            return prefix.lower()
    return "other"


LEGACY_SUBSETS = {"join": "picture_join", "bridged": "bridge_picture"}


def subsets(qtype: str) -> set[str]:
    if musique.is_type(qtype):
        return musique.subsets(qtype)
    if hipporag.is_type(qtype):
        return hipporag.subsets(qtype)
    if msmarco.is_type(qtype):
        return msmarco.subsets(qtype)
    if m3sciqa.is_type(qtype):
        return m3sciqa.subsets(qtype)
    out = set()
    if "ImageListQ" in qtype:
        out |= {"list", "picture_join"}
    elif "ImageQ" in qtype and qtype != "ImageQ":
        out |= {"bridge_picture", "picture_join"}
    if "Compose(" in qtype:
        out.add("bridged")
        if "Image" not in qtype:
            out.add("bridge_textual")
    if out & {"bridged", "list"}:
        out.add("join_all")
    if family(qtype) in ("compose", "intersect", "compare"):
        out.add("multihop")
    return out


def _answer_str(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def to_question(d: dict[str, Any]) -> Question:
    meta = d.get("metadata", {})
    qtype = meta.get("type", "")
    answers = [_answer_str(a["answer"]) for a in d.get("answers", [])]
    bridge = [_answer_str(a["answer"]) for group in meta.get("intermediate_answers", []) for a in group]
    return Question(
        qid=d["qid"],
        dataset="mmqa",
        question=d["question"],
        answers=answers,
        qtype=qtype,
        gold_doc_ids=[s["doc_id"] for s in d.get("supporting_context", [])],
        bridge=list(dict.fromkeys(bridge)),
        session=meta.get("table_id", ""),
        image_answer=any(a.get("modality") == "image" for a in d.get("answers", [])),
        meta={
            "family": family(qtype),
            "table_id": meta.get("table_id", ""),
            "text_doc_ids": meta.get("text_doc_ids", []),
            "image_doc_ids": meta.get("image_doc_ids", []),
            "modalities": meta.get("modalities", []),
            "answer_modalities": sorted({a.get("modality", "") for a in d.get("answers", [])}),
        },
    )


def load_questions(split: str) -> list[Question]:
    return [to_question(d) for d in _read_gz(f"MMQA_{split}.jsonl.gz")]


def load_raw(split: str) -> dict[str, dict[str, Any]]:
    return {d["qid"]: d for d in _read_gz(f"MMQA_{split}.jsonl.gz")}


def gold_objects(d: dict[str, Any]) -> tuple[dict[str, list[str]], list[str]]:
    meta = d.get("metadata", {})
    tid = meta.get("table_id", "")
    inner = [a for group in meta.get("intermediate_answers", []) for a in group]
    compose = re.fullmatch(r"Compose\((\w+),(\w+)\)", meta.get("type", ""))
    if compose and compose.group(1) in _MODALITY and compose.group(2) in _MODALITY:
        roles = {"first": (inner, _MODALITY[compose.group(2)]),
                 "final": (d.get("answers", []), _MODALITY[compose.group(1)])}
    else:
        roles = {"final": ([*d.get("answers", []), *inner], None)}
    gold: dict[str, list[str]] = {"first": [], "final": []}
    unknown = set()
    for role, (answers, part) in roles.items():
        keys: set[str] = set()
        for a in answers:
            keys |= {f"row:{tid}#{ri}" for ri, _ in a.get("table_indices") or []}
            keys |= {f"text:{t['doc_id']}" for t in a.get("text_instances") or []}
            keys |= {f"image:{im['doc_id']}" for im in a.get("image_instances") or []}
        rows = any(k.startswith("row:") for k in keys)
        for s in d.get("supporting_context", []):
            if part not in (None, s.get("doc_part")):
                continue
            if s.get("doc_part") in ("text", "image"):
                keys.add(f"{s['doc_part']}:{s['doc_id']}")
            elif s.get("doc_part") == "table" and not rows:
                unknown.add(s["doc_id"])
        gold[role] = sorted(keys)
    return gold, sorted(unknown)


def iter_texts() -> Iterator[dict[str, Any]]:
    return _read_gz("MMQA_texts.jsonl.gz")


def iter_tables() -> Iterator[dict[str, Any]]:
    return _read_gz("MMQA_tables.jsonl.gz")


def iter_images() -> Iterator[dict[str, Any]]:
    return _read_gz("MMQA_images.jsonl.gz")


def load_captions() -> dict[str, str]:
    path = raw_dir("mgrag") / "mmqa_caption.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {k: v.get("caption", "").strip() for k, v in data.items() if v.get("caption", "").strip()}


def text_objects(row: dict[str, Any], chunk_tokens: int = 256, overlap: int = 32) -> list[ObjectRecord]:
    return [
        ObjectRecord(oid=f"mmqa/text/{row['id']}#{i}", doc_id=row["id"], modality="text", title=row["title"], text=chunk)
        for i, chunk in enumerate(chunk_text(row["text"], chunk_tokens, overlap))
    ]


def row_objects(table: dict[str, Any]) -> list[ObjectRecord]:
    t = table["table"]
    name = (t.get("table_name") or "").strip()
    header = [h.get("column_name", "") for h in t.get("header", [])]
    page = table["title"]
    title = f"{page} — {name}" if name else page
    out = []
    for r, row in enumerate(t.get("table_rows", [])):
        cells = [(h, c.get("text", "").strip()) for h, c in zip(header, row, strict=False)]
        body = "; ".join(f"{h}: {v}" if h else v for h, v in cells if v)
        entities = sorted({link["wiki_title"] for c in row for link in c.get("links", []) if link.get("wiki_title")})
        out.append(
            ObjectRecord(
                oid=f"mmqa/row/{table['id']}#{r}",
                doc_id=table["id"],
                modality="table",
                title=title,
                text=f"{page} | {name} | {body}" if name else f"{page} | {body}",
                meta={"row": r, "entities": entities},
            )
        )
    return out


def image_object(row: dict[str, Any], captions: dict[str, str] | None = None) -> ObjectRecord:
    caption = (captions or {}).get(row["id"], "")
    return ObjectRecord(
        oid=f"mmqa/image/{row['id']}",
        doc_id=row["id"],
        modality="image",
        title=row["title"],
        text=f"{row['title']}. {caption}" if caption else row["title"],
        image_path=f"final_dataset_images/{row['path']}",
    )
