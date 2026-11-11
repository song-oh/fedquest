from __future__ import annotations

import re
import unicodedata

from rapidfuzz import fuzz

_PAREN = re.compile(r"\([^()]*\)")
_EDGE = re.compile(r"^[\W_]+|[\W_]+$")
_ARTICLE = re.compile(r"^(?:the|an?)\s+(?=\S)")
_SEP = re.compile(r"[\W_]+")
_WORD = re.compile(r"[^\W\d_]+")

STOP_WORDS = frozenset(
    "about above after again against also among been before being below between both cannot could does "
    "doing done down during each from further have having here into itself just like many more most much "
    "must once only other over same shall should some such than that their them then there these they this "
    "those through under until upon very were what when where which while whom whose will with would your".split()
)
_SUFFIXES = ("ing", "ed", "es", "ion", "s")


def _drop_parens(s: str) -> str:
    while (out := _PAREN.sub(" ", s)) != s:
        s = out
    return s


def norm_entity(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).lower()
    bare = _drop_parens(s)
    if _SEP.sub("", bare):
        s = bare
    return _SEP.sub(" ", _ARTICLE.sub("", _EDGE.sub("", s))).strip()


def same_norm(na: str, nb: str, threshold: float = 92) -> bool:
    if not na or not nb:
        return False
    if na == nb:
        return True
    ta, tb = na.split(), nb.split()
    short, long = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    k = len(short)
    if k < 2:
        return False
    if long[:k] == short or long[-k:] == short:
        return True
    return fuzz.token_set_ratio(na, nb) >= threshold


def same_entity(a: str, b: str, threshold: float = 92) -> bool:
    return same_norm(norm_entity(a), norm_entity(b), threshold)


def _stem(token: str) -> str:
    for suffix in _SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            token = token[: -len(suffix)]
            break
    return token[:-1] if token.endswith("e") and len(token) > 3 else token


def content_stems(s: str) -> set[str]:
    return {_stem(w) for w in _WORD.findall(s.lower()) if len(w) >= 4 and w not in STOP_WORDS}


_LETTER = re.compile(r"[^\W\d_]")
_NUMERIC_NAMES = ("date", "year", "time", "day", "month", "decade", "century", "era", "period", "season",
                  "number", "count", "amount", "age", "population", "height", "length", "size", "score",
                  "total", "percent", "price", "rank", "quantity", "duration", "value", "answer", "entity")


def admits(variable: str, value: str) -> bool:
    return bool(_LETTER.search(value)) or any(h in variable.lower() for h in _NUMERIC_NAMES)


def release_signal(summary: str, value: str) -> bool:
    title = summary.split(":", 1)[0]
    nv = norm_entity(value)
    return bool(nv) and (same_entity(title, value) or f" {nv} " in f" {norm_entity(summary)} ")
