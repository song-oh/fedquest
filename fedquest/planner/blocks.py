from __future__ import annotations

import math
from collections.abc import Hashable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from fedquest.utils import stable_int

Size = Literal["plan", "unit", "cap"] | int
Order = Literal["calib", "sim"]
Sizes = Size | Mapping[str, Size]
PLACE_KINDS = ("coordinator", "text", "table", "image")


def place_kind(place: str) -> str:
    return place.rsplit("|", 1)[1] if "|" in place else "coordinator"


def size_at(size: Sizes, place: str) -> Size:
    if isinstance(size, Mapping):
        return size.get(place_kind(place), "cap")
    return size


@dataclass(frozen=True)
class Cand:
    key: Hashable
    p: float
    c: float
    r: float = 0.0

    @property
    def ratio(self) -> float:
        return self.p / self.c if self.c > 0 else math.inf


def by_ratio(cands: Sequence[Cand], salt: str | None = None) -> list[Cand]:
    if salt is None:
        return sorted(cands, key=lambda x: -x.ratio)
    return sorted(cands, key=lambda x: (-x.ratio, x.r, stable_int(salt, x.key)))


def _live(places: Sequence[Place], sim: bool, salt: str | None) -> list[Place]:
    live = [pl for pl in places if pl.cands and (sim or pl.mass > 0)]
    if salt is None:
        return sorted(live, key=lambda pl: pl.name)
    return sorted(live, key=lambda pl: (stable_int(salt, pl.name), pl.name))


def by_rank(cands: Sequence[Cand]) -> list[Cand]:
    return sorted(cands, key=lambda x: x.r)


def block_plan(cands: Sequence[Cand], kappa: float, survival: float = 1.0, cap: int | None = None,
               ordered: bool = False, salt: str | None = None) -> list[list[Cand]]:
    order = list(cands) if ordered else by_ratio(cands, salt)
    n = len(order)
    if n == 0:
        return []
    limit = n if cap is None else max(1, min(cap, n))
    left = [survival]
    for x in order:
        left.append(max(0.0, left[-1] - x.p))
    cost = [0.0]
    for x in order:
        cost.append(cost[-1] + x.c)
    best = [0.0] * (n + 1)
    cut = [n] * (n + 1)
    for i in range(n - 1, -1, -1):
        best[i] = math.inf
        for k in range(i + 1, min(n, i + limit) + 1):
            # a block costs kappa plus its reads, paid only if the earlier blocks missed
            v = (kappa + cost[k] - cost[i]) * left[i] + best[k]
            if v < best[i] - 1e-12:
                best[i], cut[i] = v, k
    blocks, i = [], 0
    while i < n:
        blocks.append(order[i : cut[i]])
        i = cut[i]
    return blocks


def plan_cost(blocks: Sequence[Sequence[Cand]], kappa: float, survival: float = 1.0) -> float:
    total, left = 0.0, survival
    for b in blocks:
        total += (kappa + sum(x.c for x in b)) * left
        left = max(0.0, left - sum(x.p for x in b))
    return total


def first_block(cands: Sequence[Cand], kappa: float, survival: float = 1.0, cap: int | None = None,
                size: Size = "plan", ordered: bool = False, salt: str | None = None) -> list[Cand]:
    order = list(cands) if ordered else by_ratio(cands, salt)
    if not order:
        return []
    limit = len(order) if cap is None else max(1, min(cap, len(order)))
    if size == "plan":
        return block_plan(order, kappa, survival, limit, ordered=True)[0]
    n = 1 if size == "unit" else limit if size == "cap" else int(size)
    return order[: max(1, min(n, limit))]


@dataclass
class Place:
    name: str
    kappa: float
    cap: int
    cands: list[Cand]

    @property
    def mass(self) -> float:
        return sum(x.p for x in self.cands)


@dataclass(frozen=True)
class Call:
    place: str
    block: tuple[Cand, ...]
    p: float
    cost: float

    @property
    def rho(self) -> float:
        return self.p / self.cost if self.cost > 0 else math.inf


def _call(pl: Place, block: Sequence[Cand], missing: float) -> Call:
    pb = sum(x.p for x in block)
    return Call(pl.name, tuple(block), pb / missing if missing > 0 else 0.0, pl.kappa + sum(x.c for x in block))


def calls(places: Sequence[Place], outside: float = 0.0, size: Sizes = "plan", order: Order = "calib",
          salt: str | None = None) -> list[Call]:
    sim = order == "sim"
    live = _live(places, sim, salt)
    missing = outside + sum(pl.mass for pl in live)
    out = []
    for pl in live:
        cands = by_rank(pl.cands) if sim else pl.cands
        out.append(_call(pl, first_block(cands, pl.kappa, outside + pl.mass, pl.cap, size_at(size, pl.name),
                                         ordered=sim, salt=salt), missing))
    if sim:
        first = {pl.name: min(x.r for x in pl.cands) for pl in live}
        return sorted(out, key=lambda c: (first[c.place], c.cost))
    return sorted(out, key=lambda c: -c.rho)


def rule(places: Sequence[Place], outside: float = 0.0, size: Sizes = "plan", order: Order = "calib",
         salt: str | None = None) -> Iterator[tuple[Call, float]]:
    left = {pl.name: list(pl.cands) for pl in places}
    spec = {pl.name: pl for pl in places}
    missing = outside + sum(x.p for pl in places for x in pl.cands)
    while missing > 1e-12:
        options = calls([Place(n, spec[n].kappa, spec[n].cap, cs) for n, cs in left.items() if cs], outside,
                        size, order, salt)
        if not options:
            break
        best = options[0]
        yield best, missing
        read = {x.key for x in best.block}
        left[best.place] = [x for x in left[best.place] if x.key not in read]
        missing -= sum(x.p for x in best.block)


def expected_cost(places: Sequence[Place], outside: float = 0.0, size: Sizes = "plan", order: Order = "calib",
                  salt: str | None = None) -> float:
    return sum((call.cost * missing for call, missing in rule(places, outside, size, order, salt)), 0.0)


def curve(places: Sequence[Place], outside: float = 0.0, reply: float = 0.0, size: Sizes = "plan",
          order: Order = "calib", salt: str | None = None) -> list[tuple[float, float]]:
    return [(call.cost + reply, sum(x.p for x in call.block)) for call, _ in rule(places, outside, size, order, salt)]


def allocate(curves: Sequence[Sequence[tuple[float, float]]], budget: float, found: Sequence[float]) -> list[float]:
    n = len(curves)
    have = [max(0.0, f) for f in found]
    steps, spent, total = [0] * n, [0.0] * n, 0.0

    def fund(h: int) -> None:
        nonlocal total
        cost, mass = curves[h][steps[h]]
        steps[h] += 1
        spent[h] += cost
        have[h] += mass
        total += cost

    fresh = [h for h in range(n) if curves[h] and have[h] <= 0.0 and curves[h][0][1] > 0.0]
    if sum(curves[h][0][0] for h in fresh) <= budget:
        for h in fresh:
            fund(h)
    elif fresh and fresh[0] == 0:
        fund(0)
    while True:
        pick, gain = None, 0.0
        for h in range(n):
            if steps[h] >= len(curves[h]):
                continue
            cost, mass = curves[h][steps[h]]
            if mass <= 0.0 or total + cost > budget:
                continue
            g = math.inf if have[h] <= 0.0 or cost <= 0.0 else (math.log(have[h] + mass) - math.log(have[h])) / cost
            if g > gain:
                pick, gain = h, g
        if pick is None:
            return spent
        fund(pick)


def plan_call(places: Sequence[Place], outside: float = 0.0, room: float = math.inf,
              salt: str | None = None) -> Call | None:
    live = _live(places, False, salt)
    if not live:
        return None
    missing = outside + sum(pl.mass for pl in live)
    best: tuple[float, Call] | None = None
    for pl in live:
        order = by_ratio(pl.cands, salt)
        cap = max(1, min(pl.cap, len(order)))
        fits = 0
        for n in range(1, cap + 1):
            if pl.kappa + sum(x.c for x in order[:n]) > room:
                break
            fits = n
        if not fits:
            continue
        cuts = {len(first_block(order, pl.kappa, outside + pl.mass, cap, "plan", ordered=True)),
                len(first_block(order, pl.kappa, missing, cap, "plan", ordered=True)), 1, cap, fits}
        for n in sorted(k for k in cuts if k <= fits):
            block = order[:n]
            read = {x.key for x in block}
            rest = [Place(q.name, q.kappa, q.cap, [x for x in q.cands if q.name != pl.name or x.key not in read])
                    for q in live]
            call = _call(pl, block, missing)
            value = call.cost * missing + expected_cost(rest, outside, salt=salt)
            if best is None or value < best[0] - 1e-9:
                best = (value, call)
    return None if best is None else best[1]
