# fake backend for offline tests: canned replies pulled out of the prompt text
from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

Responder = Callable[[str, list[dict[str, Any]]], str]

_SVO = re.compile(
    r"^(.+?) (is|was|are|were|has|had|acted in|played|directed|released in|located in|born in|won) (.+)$", re.I
)
_CAPS = re.compile(r"\b(?:[A-Z][\w'’-]*)(?:\s+(?:[A-Z][\w'’-]*|of|the|and|de))*")


def message_text(messages: list[dict[str, Any]]) -> str:
    out = []
    for m in messages:
        content = m.get("content")
        parts = content if isinstance(content, list) else [content]
        for p in parts:
            if isinstance(p, dict):
                out.append(p.get("text", ""))
            elif p:
                out.append(str(p))
    return "\n".join(out)


def _source(text: str) -> tuple[str, str]:
    title = re.search(r'(?:Source title: |belongs to ")([^"\n]+)', text)
    body = re.search(r'"""\n(.*?)\n"""', text, re.S)
    return (title.group(1) if title else "source"), (body.group(1) if body else "")


def _triples(title: str, body: str) -> list[list[str]]:
    out = []
    for sentence in re.split(r"(?<=[.!?])\s+", body.strip()):
        sentence = sentence.strip().rstrip(".")
        if not sentence:
            continue
        m = _SVO.match(sentence)
        out.append([m.group(1), m.group(2).lower(), m.group(3)] if m else [title, "mentions", " ".join(sentence.split()[:6])])
    return out or [[title, "is", "a source"]]


_READ_OBJECT = re.compile(r"^\[(\d+)\] \(([^)]*)\) (.*?)(?=^\[\d+\] \(|\Z)", re.M | re.S)
_READ_UNKNOWN = re.compile(r"^- (.+?) \(at most (\d+)(; [^)]*)?\): (.*)$", re.M)
_READ_COND = re.compile(r"\((\d+)\) (?:\((.+?) \| (.+?) \| (.+?)\)|X (.+?))(?=; \(\d+\) |$)")
_CELL = re.compile(r"(?:Title|Name|Film|Work|Album|Nominee|Player|Team): ([^;|]+)")
_CELLS = re.compile(r"(?:^|; |\| )([A-Z][\w ]*?): ([^;|]+)")
_INLINE_FACT = re.compile(r"\(([^|()]+) \| ([^|()]+) \| ([^|()]+)\)")


def fake_read(text: str) -> dict[str, list[dict[str, Any]]]:
    objects = [(int(n), body) for n, _, body in _READ_OBJECT.findall(text.split("\nUnknowns")[0])]
    out: dict[str, list[dict[str, Any]]] = {}
    for name, cap, _mode, conds in _READ_UNKNOWN.findall(text):
        hits: dict[str, dict[str, Any]] = {}
        for num, head, _rel, tail, unary in _READ_COND.findall(conds):
            years = re.findall(r"\b(?:1[0-9]|20)\d{2}\b", f"{head} {_rel} {tail}")
            for n, body in objects:
                low = body.lower()
                if years and not all(y in body for y in years):
                    continue
                if unary:
                    if all(w.lower() in low for w in re.findall(r"[A-Za-z]{4,}", unary)):
                        title = body.split(":", 1)[0].split(" (")[0].strip()
                        hit = hits.setdefault(title.lower(), {"entity": title, "object": n, "conditions": []})
                        hit["conditions"].append(int(num))
                    continue
                # X is the condition's unknown end; inline facts that match its known end supply X
                known, other = (head, 2) if tail == "X" else (tail, 0)
                words = set(re.findall(r"[a-z]{4,}", _rel.lower()))
                facts = [(len(words & set(re.findall(r"[a-z]{4,}", f[1].lower()))), f[other].strip())
                         for f in _INLINE_FACT.findall(body) if f[2 - other].strip().lower() == known.lower()]
                top = max((n for n, _ in facts), default=0)
                found = [e for n, e in facts if n == top]
                if not found and known.lower() in low:
                    fit = [v.strip() for h, v in _CELLS.findall(body) if set(re.findall(r"[a-z]{4,}", h.lower())) & words]
                    found = fit[:1] or ([cell.group(1).strip()] if (cell := _CELL.search(body)) else [])
                for entity in found:
                    if entity.lower() != known.lower():
                        hit = hits.setdefault(entity.lower(), {"entity": entity, "object": n, "conditions": []})
                        if int(num) not in hit["conditions"]:
                            hit["conditions"].append(int(num))
        out[name] = list(hits.values())[: int(cap)]
    if '"_facts"' in text:
        known = [v.strip() for _, _, _, v, _ in _READ_COND.findall(text) if v and v.strip() != "X"] + \
            [h.strip() for _, h, _, _, _ in _READ_COND.findall(text) if h and h.strip() != "X"]
        facts = []
        for n, body in objects:
            for k in known:
                if k and k.lower() in body.lower():
                    facts += [{"s": k, "r": h.strip().lower(), "v": v.strip(), "src": n}
                              for h, v in _CELLS.findall(body) if v.strip() and v.strip().lower() != k.lower()]
        out["_facts"] = facts[:3]  # type: ignore[assignment]
    return out


def _anchor(question: str) -> str:
    caps = sorted(_CAPS.findall(question), key=len, reverse=True)
    return caps[0] if caps else (question.split() or ["question"])[0]


def typed_reply(question: str) -> str:
    anchor = _anchor(question)
    return json.dumps({"constants": [anchor], "variables": [{"name": "?answer1", "set": False}],
                       "atoms": [{"subject": anchor, "relation": "related to", "object": "?answer1",
                                  "evidence": "text"}], "filters": [], "answer": "?answer1", "form": "entity"})


def default_responder(task: str, messages: list[dict[str, Any]]) -> str:
    text = message_text(messages)
    if task == "pattern":
        q = re.search(r"<QUERY>\n(.*?)\n</QUERY>", text, re.S)
        question = q.group(1) if q else text
        anchor = _anchor(question)
        return (
            f'("anchor_entity"[SEP]{anchor})[REC]("entity"[SEP]answer1)[REC]("answer_entity"[SEP]answer1)[REC]'
            f'("relation"[SEP]{anchor}[SEP]answer1[SEP]related to)[END]'
        )
    if task in ("typed_graph", "typed_repair"):
        asked = re.findall(r"^Question: (.*)$", text, re.M)
        return typed_reply(asked[-1].strip() if asked else text)
    if task in {"extract_text", "extract_table", "extract_image"}:
        title, body = _source(text)
        return json.dumps({"triples": _triples(title, body or title)})
    if task == "extract_query":
        body = re.search(r"<TEXT>\n(.*?)\n</TEXT>", text, re.S)
        triples = _triples("source", body.group(1) if body else "")
        return "".join(f'("relation"[SEP]{h}[SEP]{t}[SEP]{r}[SEP]fake)[REC]' for h, r, t in triples) + "[END]"
    if task == "answer":
        fact = re.search(r"\(([^|()\n]+) \| ([^|()\n]+) \| ([^|()\n]+)\)", text)
        return f"<ANS>{fact.group(3).strip() if fact else 'unknown'}</ANS>"
    if task == "verify":
        return '{"answerable": true, "answer": ""}'
    if task in ("read", "read_index", "read_facts"):
        return json.dumps(fake_read(text))
    if task == "closed_book":
        return "<ANS>unknown</ANS>"
    if task == "judge":
        user = re.search(r"User Answer: (.*)", text)
        ref = re.search(r"Reference Answer: (.*)", text)
        a = set((user.group(1) if user else "").lower().split())
        b = set((ref.group(1) if ref else "").lower().split())
        return "4" if a and b and a & b else "0"
    if task == "ircot":
        return "So the answer is: unknown."
    if task in ("planner", "planner_ops"):
        return '{"thought": "done", "action": "finish", "input": "unknown"}'
    if task == "visual":
        return '{"picture": []}'
    return "{}"


class FakeBackend:
    def __init__(self, responder: Responder | None = None) -> None:
        self.responder = responder or default_responder
        self.calls: list[tuple[str, str]] = []

    async def complete(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int,
        temperature: float,
        json_mode: bool,
        task: str,
        n: int = 1,
    ) -> tuple[str, dict[str, int], str | None, float]:
        self.calls.append((model, task))
        if n > 1:
            replies = [self.responder(task, messages) for _ in range(n)]
            usage = {"prompt_tokens": max(1, len(message_text(messages)) // 4),
                     "completion_tokens": sum(max(1, len(r) // 4) for r in replies)}
            return json.dumps(replies), usage, "stop", 0.0
        reply = self.responder(task, messages)
        usage = {"prompt_tokens": max(1, len(message_text(messages)) // 4), "completion_tokens": max(1, len(reply) // 4)}
        return reply, usage, "stop", 0.0

    async def aclose(self) -> None:
        return None
