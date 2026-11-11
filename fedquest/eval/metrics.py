from __future__ import annotations

import asyncio
import re
import string
from collections import Counter
from collections.abc import Sequence
from functools import cache
from typing import Any

import numpy as np

from fedquest.llm import prompts
from fedquest.llm.client import LLMClient

_ARTICLES = re.compile(r"\b(a|an|the)\b")


def normalize_answer(s: str) -> str:
    s = "".join(ch for ch in s.lower() if ch not in set(string.punctuation))
    return " ".join(_ARTICLES.sub(" ", s).split())


def exact_match(pred: str, golds: Sequence[str]) -> float:
    p = normalize_answer(pred)
    return float(any(p == normalize_answer(g) for g in golds))


def f1(pred: str, golds: Sequence[str]) -> float:
    def one(p: list[str], g: list[str]) -> float:
        common = sum((Counter(p) & Counter(g)).values())
        if not p or not g or common == 0:
            return float(p == g)
        precision, recall = common / len(p), common / len(g)
        return 2 * precision * recall / (precision + recall)

    toks = normalize_answer(pred).split()
    return max((one(toks, normalize_answer(g).split()) for g in golds), default=0.0)


@cache
def _rouge() -> Any:
    from rouge_score import rouge_scorer

    return rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def rouge_l(pred: str, golds: Sequence[str]) -> float:
    return max((_rouge().score(g.lower(), pred.lower())["rougeL"].fmeasure for g in golds), default=0.0)


def bleu1(pred: str, golds: Sequence[str]) -> float:
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

    cand = pred.lower().split()
    if not cand:
        return 0.0
    smooth = SmoothingFunction().method1
    return max((float(sentence_bleu([g.lower().split()], cand, weights=(1,), smoothing_function=smooth))
                for g in golds), default=0.0)


def sim(pred: str, golds: Sequence[str], encoder: Any) -> float:
    if not golds:
        return 0.0
    vecs = encoder.encode_docs([pred or " ", *[g or " " for g in golds]])
    return float(np.max(vecs[1:] @ vecs[0]))


async def judge_accuracy(question: str, pred: str, golds: Sequence[str], llm: LLMClient, role: str = "judge") -> float:
    if not pred.strip() or not golds:
        return 0.0
    if exact_match(pred, golds) >= 1.0:
        return 1.0
    best, parsed = 0.0, False
    for gold in golds:
        if best >= 1.0:
            break
        replies = await asyncio.gather(*(llm.chat(prompts.judge_messages(question, gold, pred, variant), role=role,
                                                  task="judge", max_tokens=1024, temperature=0.0)
                                         for variant in (1, 2)))
        ratings = [r for r in (prompts.parse_rating(x.text) for x in replies) if r is not None]
        if ratings:
            parsed = True
            best = max(best, sum(ratings) / len(ratings))
    return best if parsed else None
