# locality splits: warm streams covering more of the eval questions' hop docs
from __future__ import annotations

import random
from collections import Counter
from collections.abc import Sequence

from fedquest.types import Question

EVAL_N, WARM_N = 200, 360
LEVELS = {"loc00": 0.0, "loc25": 0.25, "loc50": 0.5, "loc75": 0.75, "loc100": 1.0}
SPLITS = ("loc_eval", *LEVELS, "loc_smoke")
SMOKE_N = 10


def hop_docs(q: Question) -> list[str]:
    hops = q.meta.get("hops") or []
    return list(dict.fromkeys(h["doc_id"] for h in hops if h.get("doc_id"))) or list(q.gold_doc_ids)


def _stratified(pool: Sequence[Question], n: int, rng: random.Random) -> list[Question]:
    by_type: dict[str, list[Question]] = {}
    for q in sorted(pool, key=lambda q: q.qid):
        by_type.setdefault(q.qtype, []).append(q)
    share = {t: n * len(qs) / len(pool) for t, qs in by_type.items()}
    quota = {t: int(s) for t, s in share.items()}
    for t in sorted(share, key=lambda t: (quota[t] - share[t], t))[: n - sum(quota.values())]:
        quota[t] += 1
    return [q for t in sorted(quota) for q in rng.sample(by_type[t], min(quota[t], len(by_type[t])))]


def locality_splits(questions: Sequence[Question], seed: int = 13) -> dict[str, list[Question]]:
    rng = random.Random(f"{seed}:loc-eval")
    evals = _stratified(questions, EVAL_N, rng)
    held = {q.qid for q in evals}
    slots = Counter(d for q in evals for d in hop_docs(q))
    total = sum(slots.values())
    pool = sorted((q for q in questions if q.qid not in held), key=lambda q: q.qid)
    zero = [q for q in pool if not set(hop_docs(q)) & set(slots)]
    random.Random(f"{seed}:loc-zero").shuffle(zero)
    covered: set[str] = set()
    cover: list[tuple[Question, float]] = []
    left = [q for q in pool if set(hop_docs(q)) & set(slots)]
    while left and len(cover) < WARM_N:
        gains = [sum(slots[d] for d in set(hop_docs(q)) if d in slots and d not in covered) for q in left]
        best = max(range(len(left)), key=lambda i: (gains[i], -i))
        if gains[best] == 0:
            break
        q = left.pop(best)
        covered |= set(hop_docs(q))
        cover.append((q, sum(slots[d] for d in covered if d in slots) / total))
    top = cover[-1][1] if cover else 0.0
    out: dict[str, list[Question]] = {"loc_eval": [_marked(q, True, None, None) for q in evals]}
    head = [q for q, _ in cover[:SMOKE_N]]
    seen = {d for q in head for d in hop_docs(q)}
    hit = [q for q in evals if set(hop_docs(q)) & seen][:SMOKE_N]
    out["loc_smoke"] = [*(_marked(q, False, 1.0, None) for q in head), *(_marked(q, True, 1.0, None) for q in hit)]
    for name, level in LEVELS.items():
        target = level * top
        warm = [q for q, c in cover[: next((i + 1 for i, (_, c) in enumerate(cover) if c >= target - 1e-9), 0)]] \
            if level > 0 else []
        reached = cover[len(warm) - 1][1] if warm else 0.0
        fill = [q for q in zero if q not in warm][: WARM_N - len(warm)]
        if len(warm) + len(fill) < WARM_N:
            rest = [q for q, _ in reversed(cover) if q not in warm]
            fill += rest[: WARM_N - len(warm) - len(fill)]
            reached = sum(slots[d] for d in {d for q in [*warm, *fill] for d in hop_docs(q)} if d in slots) / total
        stream = list(warm + fill)
        random.Random(f"{seed}:{name}:warm").shuffle(stream)
        tail = list(evals)
        random.Random(f"{seed}:{name}:eval").shuffle(tail)
        out[name] = [*(_marked(q, False, level, reached) for q in stream),
                     *(_marked(q, True, level, reached) for q in tail)]
    return out


def _marked(q: Question, evaluated: bool, level: float | None, reached: float | None) -> Question:
    d = q.to_dict()
    meta = dict(d.get("meta") or {})
    meta["loc_eval"] = evaluated
    if level is not None:
        meta["loc_level"], meta["loc_coverage"] = level, round(reached or 0.0, 4)
    d["meta"], d["session"] = meta, "loc"
    return Question.from_dict(d)
