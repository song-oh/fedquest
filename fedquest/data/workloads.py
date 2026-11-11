from __future__ import annotations

import random
from collections import OrderedDict

from fedquest.types import Question


def cold(questions: list[Question]) -> list[Question]:
    return list(questions)


def _groups(questions: list[Question]) -> list[list[Question]]:
    groups: OrderedDict[str, list[Question]] = OrderedDict()
    for q in questions:
        groups.setdefault(q.session or q.qid, []).append(q)
    return list(groups.values())


def sessions(questions: list[Question], order: str = "natural", seed: int = 13) -> list[Question]:
    if order == "random":
        out = list(questions)
        random.Random(seed).shuffle(out)
        return out
    return [q for group in _groups(questions) for q in group]


def streams(questions: list[Question], n_agents: int, seed: int = 13, deal: str = "overlap") -> list[list[Question]]:
    groups = _groups(questions)
    random.Random(seed).shuffle(groups)
    out: list[list[Question]] = [[] for _ in range(max(1, n_agents))]
    if deal == "sessions":
        for i, group in enumerate(groups):
            out[i % len(out)].extend(group)
        return out
    k = 0
    for group in groups:
        for q in group:
            out[k % len(out)].append(q)
            k += 1
    return out
