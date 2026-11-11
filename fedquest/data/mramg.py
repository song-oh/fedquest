# MRAMG-Bench documents and questions, as text and image objects
from __future__ import annotations

import json
import re
from typing import Any

from fedquest.data.chunking import chunk_text
from fedquest.paths import raw_dir
from fedquest.types import ObjectRecord, Question
from fedquest.utils import clean_text

SUBSETS = ("web", "wit", "wiki", "recipe", "manual", "arxiv")
_ARTIFACT = re.compile(r"_x[0-9A-Fa-f]{4}_")
_PIC = "<PIC>"
_HEADING = re.compile(r"^\s*#+\s*(.+)$", re.M)


def load_docs(subset: str) -> dict[str, dict[str, Any]]:
    docs: dict[str, dict[str, Any]] = {}
    with open(raw_dir("mramg") / f"doc_{subset}.jsonl", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            doc_id, content, images = json.loads(line)
            key = str(doc_id)
            if key not in docs:
                docs[key] = {"id": key, "content": _ARTIFACT.sub("", content), "images": [str(i) for i in images]}
    return docs


def load_images_info(subset: str) -> dict[str, dict[str, Any]]:
    path = raw_dir("mramg") / "IMAGE" / "images_info" / f"{subset}_imgs_collection.json"
    if not path.exists():
        return {}
    return {str(k): v for k, v in json.loads(path.read_text(encoding="utf-8")).items()}


def load_questions(subset: str) -> list[Question]:
    out = []
    with open(raw_dir("mramg") / f"{subset}_mqa.jsonl", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            provenance = [str(p) for p in d["provenance"]]
            out.append(
                Question(
                    qid=str(d["id"]),
                    dataset="mramg",
                    question=d["question"],
                    answers=[clean_text(d["ground_truth"].replace(_PIC, " "))],
                    qtype=f"mramg-{subset}",
                    gold_doc_ids=provenance,
                    session=provenance[0] if provenance else "",
                    meta={"family": "mramg", "subset": subset, "images_list": [str(i) for i in d["images_list"]]},
                )
            )
    return out


def doc_title(content: str) -> str:
    heading = _HEADING.search(content)
    first = heading.group(1) if heading else content.replace(_PIC, " ")
    return clean_text(first)[:80] or "document"


def doc_objects(doc: dict[str, Any], subset: str, chunk_tokens: int = 256, overlap: int = 32) -> list[ObjectRecord]:
    title = doc_title(doc["content"])
    text = clean_text(doc["content"].replace(_PIC, " "))
    return [
        ObjectRecord(oid=f"mramg/{subset}/text/{doc['id']}#{i}", doc_id=doc["id"], modality="text", title=title, text=c)
        for i, c in enumerate(chunk_text(text, chunk_tokens, overlap))
    ]


def _context(content: str, index: int, width: int = 300) -> str:
    positions = [m.start() for m in re.finditer(re.escape(_PIC), content)]
    if index >= len(positions):
        return ""
    pos = positions[index]
    around = content[max(0, pos - width) : pos] + " " + content[pos + len(_PIC) : pos + len(_PIC) + width]
    return clean_text(around.replace(_PIC, " "))[: 2 * width]


def image_objects(doc: dict[str, Any], info: dict[str, dict[str, Any]], subset: str) -> list[ObjectRecord]:
    title = doc_title(doc["content"])
    out: dict[str, ObjectRecord] = {}
    for i, key in enumerate(doc["images"]):
        meta = info.get(key)
        if meta is None or key in out:
            continue
        rel = f"IMAGE/images/{subset.upper()}/{meta['image_path']}"
        if not (raw_dir("mramg") / rel).exists():
            continue
        caption = clean_text(meta.get("image_caption") or "") or _context(doc["content"], i)
        out[key] = ObjectRecord(
            oid=f"mramg/{subset}/image/{key}",
            doc_id=doc["id"],
            modality="image",
            title=title,
            text=caption or title,
            image_path=rel,
            meta={"key": key},
        )
    return list(out.values())
