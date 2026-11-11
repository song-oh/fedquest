# lenient reply parsing: JSON spans, triples, [SEP]/[REC] records and <ANS> answers
from __future__ import annotations

import json
import re
from collections.abc import Iterator, Sequence
from typing import Any

TUPLE_DELIM = "[SEP]"
RECORD_DELIM = "[REC]"
COMPLETION_DELIM = "[END]"

MAX_TRIPLES = 40
MAX_PART_CHARS = 200

_FENCE = re.compile(r"```[ \t]*(?:json)?[ \t]*\r?\n?(.*?)```", re.DOTALL | re.IGNORECASE)
_OPENER = re.compile(r"[{\[]")
_CLOSE_OF = {"{": "}", "[": "]"}
_STRING_OR_TRAILING_COMMA = re.compile(r'("(?:[^"\\]|\\.)*")|,(\s*[}\]])', re.DOTALL)
_ANSWER_TAG = re.compile(r"<ANS>(.*?)(?:</ANS>|\Z)", re.DOTALL | re.IGNORECASE)
_ANSWER_PREFIX = re.compile(r"^answer\s*:\s*", re.IGNORECASE)
_QUOTES = {'"': '"', "'": "'", "`": "`", "“": "”", "‘": "’"}
_TRIPLE_KEYS = (("subject", "head"), ("predicate", "relation"), ("object", "tail"))

Triple = tuple[str, str, str]


def parse_json(text: str) -> Any | None:
    if not text:
        return None
    fenced = [m.group(1) for m in _FENCE.finditer(text)]
    for chunk in (*fenced, text):
        for span in _balanced_spans(chunk):
            for candidate in (span, _STRING_OR_TRAILING_COMMA.sub(_keep_string, span)):
                try:
                    return json.loads(candidate)
                except ValueError:
                    continue
    return None


def json_values(text: str) -> Iterator[Any]:
    if not text:
        return
    for chunk in (*(m.group(1) for m in _FENCE.finditer(text)), text):
        for span in _balanced_spans(chunk):
            for candidate in (span, _STRING_OR_TRAILING_COMMA.sub(_keep_string, span)):
                try:
                    yield json.loads(candidate)
                    break
                except ValueError:
                    continue


def json_objects(text: str) -> Iterator[tuple[int, Any]]:
    for start in (i for i, c in enumerate(text or "") if c == "{"):
        end = _balanced_end(text, start)
        # innermost objects only: a span holding another { is skipped
        if end is None or "{" in text[start + 1 : end]:
            continue
        span = text[start : end + 1]
        for candidate in (span, _STRING_OR_TRAILING_COMMA.sub(_keep_string, span)):
            try:
                yield start, json.loads(candidate)
                break
            except ValueError:
                continue


def _keep_string(m: re.Match[str]) -> str:
    return m.group(1) or m.group(2)


def _balanced_spans(text: str) -> Iterator[str]:
    pos = 0
    while (opener := _OPENER.search(text, pos)) is not None:
        end = _balanced_end(text, opener.start())
        if end is None:
            return
        yield text[opener.start() : end + 1]
        pos = end + 1


def _balanced_end(text: str, start: int) -> int | None:
    stack: list[str] = []
    in_string = escaped = False
    for i in range(start, len(text)):
        c = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif c == "\\":
                escaped = True
            elif c == '"':
                in_string = False
        elif c == '"':
            in_string = True
        elif c in _CLOSE_OF:
            stack.append(_CLOSE_OF[c])
        elif c in "}]":
            if not stack or stack.pop() != c:
                return None
            if not stack:
                return i
    return None


def parse_triples(obj: Any) -> list[Triple]:
    seen: set[Triple] = set()
    out: list[Triple] = []
    for item in _triple_items(obj):
        triple = _as_triple(item)
        if triple is None:
            continue
        key = (triple[0].casefold(), triple[1].casefold(), triple[2].casefold())
        if key in seen:
            continue
        seen.add(key)
        out.append(triple)
        if len(out) == MAX_TRIPLES:
            break
    return out


def _triple_items(obj: Any) -> list[Any]:
    if isinstance(obj, dict):
        obj = next((obj[k] for k in ("triples", "relations") if isinstance(obj.get(k), list)), [])
    return obj if isinstance(obj, list) else []


def _as_triple(item: Any) -> Triple | None:
    if isinstance(item, dict):
        raw = [next((item[k] for k in keys if k in item), None) for keys in _TRIPLE_KEYS]
    elif isinstance(item, (list, tuple)) and len(item) == 3:
        raw = list(item)
    else:
        return None
    parts = [_clean_part(p) for p in raw]
    if any(not p or len(p) > MAX_PART_CHARS for p in parts):
        return None
    return (parts[0], parts[1], parts[2])


def _clean_part(value: Any) -> str:
    if value is None or isinstance(value, (dict, list)):
        return ""
    return " ".join(str(value).split())


def parse_records(
    text: str,
    tuple_delim: str = TUPLE_DELIM,
    record_delims: Sequence[str] = (RECORD_DELIM, COMPLETION_DELIM),
) -> list[list[str]]:
    splitter = "|".join(re.escape(d) for d in (*record_delims, "\n"))
    records: list[list[str]] = []
    for chunk in re.split(splitter, text or ""):
        start, end = chunk.find("("), chunk.rfind(")")
        if start < 0 or end <= start:
            continue
        attrs = [_unquote(a) for a in chunk[start + 1 : end].split(tuple_delim)]
        if any(attrs):
            records.append(attrs)
    return records


def _unquote(s: str) -> str:
    s = s.strip()
    while len(s) >= 2 and _QUOTES.get(s[0]) == s[-1]:
        s = s[1:-1].strip()
    return s


def parse_answer(text: str) -> str:
    text = text or ""
    m = _ANSWER_TAG.search(text)
    body = m.group(1) if m else text
    return _ANSWER_PREFIX.sub("", body.strip(), count=1).strip()
