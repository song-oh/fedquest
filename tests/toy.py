from __future__ import annotations

import json
import os
import re
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from fedquest.types import ObjectRecord, Question

QUESTION = "What sport is on the poster of the 1976 film Ben Piazza acted in?"

PATTERN = (
    '("anchor_entity"[SEP]Ben Piazza)[REC]("entity"[SEP]film1)[REC]("entity"[SEP]sport1)[REC]'
    '("answer_entity"[SEP]sport1)[REC]("relation"[SEP]Ben Piazza[SEP]film1[SEP]acted in 1976 film)[REC]'
    '("relation"[SEP]film1[SEP]sport1[SEP]poster depicts sport)[END]'
)

TRIPLES: dict[str, list[list[str]]] = {
    "Ben Piazza — Filmography#1976": [["Ben Piazza", "filmography includes", "The Bad News Bears"],
                                      ["The Bad News Bears", "released in", "1976"]],
    "Ben Piazza — Filmography#1962": [["Ben Piazza", "filmography includes", "No Exit"],
                                      ["No Exit", "released in", "1962"]],
    "The Bad News Bears": [["The Bad News Bears", "poster depicts", "baseball"]],
    "Ben Piazza Theatre": [["Ben Piazza Theatre", "depicts", "a theatre building"]],
    "Ben Piazza": [["Ben Piazza", "was", "American actor"]],
    "Walter Matthau": [["Walter Matthau", "starred in", "The Bad News Bears"]],
}

FEDERATION = {
    "name": "toy_fed",
    "dataset": "toy",
    "answer_style": "short",
    "qdrant": {"mode": "local"},
    "sites": [
        {"name": "aurora", "machine": "aurora", "policy": "snippets", "collections": [
            {"name": "image", "modality": "image", "encoder": "hash", "source": "image:*"},
            {"name": "text", "modality": "text", "encoder": "hash", "source": "text:0"}]},
        {"name": "midway3", "machine": "midway3", "policy": "snippets", "collections": [
            {"name": "row", "modality": "table", "encoder": "hash", "source": "table:*"}]},
        {"name": "edge", "machine": "edge", "policy": "facts", "collections": [
            {"name": "text", "modality": "text", "encoder": "hash", "source": "text:1"}]},
    ],
}


@dataclass
class ToyWorld:
    root: Path
    config: Path
    questions: list[Question]


def _objects() -> dict[tuple[str, str], list[ObjectRecord]]:
    def row(year: str, film: str, role: str) -> ObjectRecord:
        return ObjectRecord(f"toy/row/piazza#{year}", "piazza", "table", "Ben Piazza — Filmography",
                            f"Ben Piazza | Filmography | Year: {year}; Title: {film}; Role: {role}")

    return {
        ("midway3", "row"): [row("1962", "No Exit", "Camarero"), row("1976", "The Bad News Bears", "Bob Whitewood")],
        ("aurora", "image"): [
            ObjectRecord("toy/image/bnb", "bnb-poster", "image", "The Bad News Bears", "The Bad News Bears",
                         image_path="images/bnb.png"),
            ObjectRecord("toy/image/theatre", "theatre", "image", "Ben Piazza Theatre", "Ben Piazza Theatre",
                         image_path="images/theatre.png"),
        ],
        ("aurora", "text"): [ObjectRecord("toy/text/matthau#0", "matthau", "text", "Walter Matthau",
                                          "Walter Matthau starred in The Bad News Bears (1976).")],
        ("edge", "text"): [ObjectRecord("toy/text/piazza#0", "piazza-bio", "text", "Ben Piazza",
                                          "Ben Piazza (1933-1991) was an American actor and writer.")],
    }


IMAGE_TITLES: dict[str, str] = {}


def _seen_image(messages: list[dict[str, Any]]) -> str:
    for m in messages:
        for part in m["content"] if isinstance(m["content"], list) else []:
            if part.get("type") == "image_url":
                return IMAGE_TITLES.get(part["image_url"]["url"], "")
    return ""


def responder(task: str, messages: list[dict[str, Any]]) -> str:
    from fedquest.llm.fake import message_text

    text = message_text(messages)
    if task == "pattern":
        return PATTERN if "Ben Piazza" in text else '("anchor_entity"[SEP]x)[REC]("entity"[SEP]y)[REC]("relation"[SEP]x[SEP]y[SEP]r)[END]'
    if task in ("extract_text", "extract_table", "extract_image", "extract_query"):
        title = re.search(r'(?:Source title: |belongs to "|Look at the source "|<TEXT>\n)([^"\n]+)', text)
        key = title.group(1) if title else _seen_image(messages)
        year = re.search(r"Year: (\d{4})", text)
        triples = TRIPLES.get(f"{key}#{year.group(1)}" if year else key, [])
        if task == "extract_query":
            return "".join(f'("relation"[SEP]{h}[SEP]{t}[SEP]{r}[SEP]-)[REC]' for h, r, t in triples) + "[END]"
        return json.dumps({"triples": triples})
    if task in ("read", "read_index", "read_facts"):
        from fedquest.llm.fake import fake_read

        seen = re.sub(r"^(\[\d+\] \(image\) )(.+)$", lambda m: m.group(1) + m.group(2) + ": " + "; ".join(
            f"({h} | {r} | {t})" for h, r, t in TRIPLES.get(m.group(2).strip(), [])), text, flags=re.M)
        return json.dumps(fake_read(seen))
    if task == "verify":
        return '{"answerable": %s, "answer": ""}' % ("true" if "baseball" in text else "false")
    if task == "answer":
        return "<ANS>baseball</ANS>" if "baseball" in text else "<ANS>unknown</ANS>"
    if task == "judge":
        return "4" if "baseball" in text.lower() else "0"
    if task == "ircot":
        return "So the answer is: baseball." if "baseball" in text else "The 1976 film is The Bad News Bears."
    if task == "planner":
        return '{"thought": "done", "action": "finish", "input": "baseball"}'
    return "{}"


def build_toy(root: Path) -> ToyWorld:
    from PIL import Image, ImageDraw

    from fedquest.config import load_federation
    from fedquest.data.embed import build_index, embed_federation
    from fedquest.paths import collection_dir, questions_path, raw_dir
    from fedquest.utils import write_jsonl

    config = root / "toy_fed.yaml"
    root.mkdir(parents=True, exist_ok=True)
    config.write_text(yaml.safe_dump(FEDERATION, sort_keys=False))
    images = raw_dir("toy") / "images"
    images.mkdir(parents=True, exist_ok=True)
    from fedquest.site.extractor import load_image_data_uri

    for name, title, color in (("bnb", "The Bad News Bears", (30, 120, 60)), ("theatre", "Ben Piazza Theatre", (120, 30, 30))):
        im = Image.new("RGB", (64, 64), color)
        ImageDraw.Draw(im).text((4, 24), name, fill=(255, 255, 255))
        im.save(images / f"{name}.png")
        for side, quality in ((768, 85), (384, 75)):
            IMAGE_TITLES[load_image_data_uri(images / f"{name}.png", side, quality) or ""] = title
    for (site, coll), objs in _objects().items():
        write_jsonl(collection_dir("toy_fed", site, coll) / "objects.jsonl", (o.to_dict() for o in objs))
    questions = [Question("toy-1", "toy", QUESTION, ["baseball"], qtype="Compose(ImageQ,TableQ)",
                          bridge=["The Bad News Bears"], session="piazza", meta={"family": "compose"})]
    write_jsonl(questions_path("toy_fed", "test"), (q.to_dict() for q in questions))
    write_jsonl(questions_path("toy_fed", "calib"), (q.to_dict() for q in questions))
    cfg = load_federation(config)
    embed_federation(cfg, device="cpu")
    for site in cfg.site_names:
        build_index(cfg, site)
    return ToyWorld(root, config, questions)


@contextmanager
def toy_env(root: Path):
    keys = ("FEDQUEST_DATA", "FEDQUEST_RUNS", "FEDQUEST_LLM", "FEDQUEST_QDRANT_MODE")
    old = {k: os.environ.get(k) for k in keys}
    os.environ.update(FEDQUEST_DATA=str(root / "data"), FEDQUEST_RUNS=str(root / "runs"), FEDQUEST_LLM="fake",
                      FEDQUEST_QDRANT_MODE="local")
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
