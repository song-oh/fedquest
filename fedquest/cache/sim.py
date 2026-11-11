# offline cache study: a stream's hop asks replayed against a fact cache
from __future__ import annotations

import argparse
import json
import math
import random
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from fedquest.kg.normalize import norm_entity


@dataclass
class Hop:
    q: int
    asks: list[tuple[str, str]]
    tokens: float
    rounds: int
    facts: list[dict[str, Any]]
    sites: set[str] = field(default_factory=set)


@dataclass
class Entry:
    key: tuple[str, str]
    facts: list[dict[str, Any]]
    cost: float
    size: int
    born: int
    validated: int
    freq: int = 1
    prio: float = 0.0
    last: int = 0


def load_stream(paths: list[Path], label: str | None = None) -> tuple[list[list[Hop]], float, int]:
    rows = []
    for p in paths:
        for line in p.open(encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                if label and r.get("label") != label:
                    continue
                rows.append(r)
    rows.sort(key=lambda r: (r.get("stream_pos", r.get("position", 0)), r.get("qid", "")))
    out: list[list[Hop]] = []
    total_tokens, total_rounds = 0.0, 0
    for i, r in enumerate(rows):
        led = r.get("ledger") or {}
        total_tokens += float(led.get("total_tokens", 0))
        total_rounds += int(led.get("rounds", 0))
        sites = {k.removeprefix("disclosed_keys_") for k, v in led.items() if k.startswith("disclosed_keys_") and v}
        hops = []
        for e in r.get("trace") or []:
            if e.get("event") != "hop7":
                continue
            asks = [(norm_entity(a.get("value") or ""), a.get("rel") or "") for a in e.get("asks") or []
                    if a.get("value")]
            if not asks:
                continue
            hops.append(Hop(i, asks, float(e.get("tokens", 0)), int(e.get("wan_rounds", 0)), e.get("facts") or [],
                            sites))
        out.append(hops)
    return out, total_tokens, total_rounds


class RelSim:
    def __init__(self, relations: list[str]) -> None:
        from fedquest.vdb.encoders import load_encoder

        uniq = sorted(set(relations))
        enc = load_encoder("bge-base")
        vec = np.asarray(enc.encode_docs(uniq), dtype=np.float32) if uniq else np.zeros((0, 768), np.float32)
        vec /= np.maximum(np.linalg.norm(vec, axis=1, keepdims=True), 1e-9)
        self.ix = {r: i for i, r in enumerate(uniq)}
        self.vec = vec

    def sim(self, a: str, b: str) -> float:
        if a == b:
            return 1.0
        ia, ib = self.ix.get(a), self.ix.get(b)
        if ia is None or ib is None:
            return 0.0
        return float(self.vec[ia] @ self.vec[ib])


def _facts_for(value: str, facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [f for f in facts if norm_entity(f.get("h", "")) == value or norm_entity(f.get("t", "")) == value]


def simulate(stream: list[list[Hop]], total_tokens: float, total_rounds: int, relsim: RelSim, *,
             capacity: float = math.inf, eviction: str = "gdsf", validity: str = "piggyback", ttl: int = 100,
             update_rate: float = 0.0, rel_sim: float = 0.85, seed: int = 0, objects: int = 20000) -> dict[str, Any]:
    rng = random.Random(seed)
    cache: dict[tuple[str, str], Entry] = {}
    by_value: dict[str, list[tuple[str, str]]] = defaultdict(list)
    lru: OrderedDict[tuple[str, str], None] = OrderedDict()
    used = 0.0
    inflation = 0.0
    changed_at: dict[tuple[str, str], int] = {}
    heard: dict[str, int] = {}
    seen_obj: list[tuple[str, str]] = []
    seen_set: set[tuple[str, str]] = set()
    stats = defaultdict(float)
    per_change = update_rate / 1000.0

    def fresh(e: Entry, now: int) -> bool:
        return all(changed_at.get((f.get("site", ""), f.get("oid", "")), -1) <= e.validated for f in e.facts)

    def evict_one() -> None:
        nonlocal used, inflation
        if not cache:
            return
        if eviction == "lru" or eviction == "fifo":
            k = next(iter(lru))
        elif eviction == "lfu":
            k = min(cache, key=lambda k: (cache[k].freq, cache[k].last))
        else:
            k = min(cache, key=lambda k: (cache[k].prio, cache[k].last))
            inflation = cache[k].prio
        e = cache.pop(k)
        lru.pop(k, None)
        by_value[k[0]].remove(k)
        used -= e.size
        stats["evictions"] += 1

    # GDSF priority: inflation plus freq * cost / size; benefit also decays it by age at the update rate
    def priority(e: Entry, now: int) -> float:
        base = e.freq * e.cost / max(1, e.size)
        if eviction == "benefit" and per_change > 0:
            base *= math.exp(-per_change * max(0, now - e.validated))
        return inflation + base

    for pos, hops in enumerate(stream):
        if per_change > 0:
            n_changes = np.random.default_rng(seed * 100003 + pos).poisson(per_change * objects)
            for _ in range(int(n_changes)):
                if seen_obj and rng.random() < len(seen_obj) / objects:
                    changed_at[rng.choice(seen_obj)] = pos
        contacted = set().union(*(h.sites for h in hops)) if hops else set()
        # piggyback: a contacted site reports which of its objects changed since we last heard from it
        if validity == "piggyback":
            for s in contacted:
                last = heard.get(s, -1)
                for k in [k for k, e in cache.items() if any(
                        f.get("site") == s and last < changed_at.get((s, f.get("oid", "")), -2) <= pos
                        for f in e.facts)]:
                    e = cache.pop(k)
                    lru.pop(k, None)
                    by_value[k[0]].remove(k)
                    used -= e.size
                    stats["invalidated"] += 1
                heard[s] = pos
        for h in hops:
            share = h.tokens / max(1, len(h.asks))
            all_hit = True
            for value, rel in h.asks:
                stats["asks"] += 1
                hit_key = None
                for k in by_value.get(value, []):
                    if relsim.sim(k[1], rel) >= rel_sim:
                        hit_key = k
                        break
                if hit_key is not None:
                    e = cache[hit_key]
                    age = pos - e.validated
                    need_check = validity == "validate" or (validity == "ttl" and age > ttl) or \
                        (validity == "piggyback" and any(pos - heard.get(f.get("site", ""), -10**9) > ttl
                                                         for f in e.facts))
                    if need_check:
                        stats["validation_rounds"] += 1
                        if not fresh(e, pos):
                            cache.pop(hit_key)
                            lru.pop(hit_key, None)
                            by_value[value].remove(hit_key)
                            used -= e.size
                            stats["validated_stale"] += 1
                            hit_key = None
                        else:
                            e.validated = pos
                    if hit_key is not None:
                        stats["hits"] += 1
                        stats["tokens_saved"] += share
                        if not fresh(e, pos):
                            stats["stale_hits"] += 1
                        e.freq += 1
                        e.last = pos
                        e.prio = priority(e, pos)
                        lru.move_to_end(hit_key) if eviction == "lru" else None
                        continue
                all_hit = False
                facts = _facts_for(value, h.facts)
                if not facts:
                    continue
                size = sum(len(json.dumps(f)) for f in facts)
                if size > capacity:
                    continue
                key = (value, rel)
                if key in cache:
                    continue
                e = Entry(key, facts, share, size, pos, pos, last=pos)
                e.prio = priority(e, pos)
                while used + size > capacity and cache:
                    evict_one()
                cache[key] = e
                lru[key] = None
                by_value[value].append(key)
                used += size
                for f in facts:
                    o = (f.get("site", ""), f.get("oid", ""))
                    if o not in seen_set:
                        seen_set.add(o)
                        seen_obj.append(o)
            if all_hit and h.asks:
                stats["rounds_saved"] += h.rounds
        stats["peak_bytes"] = max(stats["peak_bytes"], used)
    asks = max(1.0, stats["asks"])
    return {"capacity": None if math.isinf(capacity) else capacity, "eviction": eviction, "validity": validity,
            "ttl": ttl, "update_rate": update_rate, "rel_sim": rel_sim, "questions": len(stream),
            "asks": int(stats["asks"]), "hit_rate": round(stats["hits"] / asks, 4),
            "tokens_saved_frac": round(stats["tokens_saved"] / max(1.0, total_tokens), 4),
            "rounds_saved": int(stats["rounds_saved"]), "validation_rounds": int(stats["validation_rounds"]),
            "net_rounds_saved_frac": round((stats["rounds_saved"] - stats["validation_rounds"]) / max(1, total_rounds),
                                           4),
            "stale_hits": int(stats["stale_hits"]),
            "stale_hit_rate": round(stats["stale_hits"] / max(1.0, stats["hits"]), 4),
            "invalidated": int(stats["invalidated"]), "evictions": int(stats["evictions"]),
            "peak_bytes": int(stats["peak_bytes"])}


def reuse_profile(stream: list[list[Hop]], relsim: RelSim, rel_sim: float = 0.85) -> dict[str, Any]:
    first: dict[tuple[str, str], int] = {}
    dists: list[int] = []
    for pos, hops in enumerate(stream):
        for h in hops:
            for value, rel in h.asks:
                hit = next((k for k in first if k[0] == value and relsim.sim(k[1], rel) >= rel_sim), None)
                if hit is not None:
                    dists.append(pos - first[hit])
                elif _facts_for(value, h.facts):
                    first[(value, rel)] = pos
    arr = np.asarray(dists) if dists else np.zeros(0)
    q = (lambda p: float(np.percentile(arr, p)) if len(arr) else None)  # noqa: E731
    return {"entries": len(first), "reuses": len(dists), "p50": q(50), "p90": q(90), "p99": q(99),
            "max": float(arr.max()) if len(arr) else None}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m fedquest.cache.sim")
    ap.add_argument("rows", nargs="+", type=Path, help="row files of the cold stream run (rows or rows_scored)")
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--objects", type=int, default=20000, help="objects in the corpus (the update process)")
    args = ap.parse_args(argv)
    stream, tokens, rounds = load_stream(args.rows, args.label)
    rels = [rel for hops in stream for h in hops for _, rel in h.asks]
    relsim = RelSim(rels)
    sizes = [sum(len(json.dumps(f)) for f in h.facts) for hops in stream for h in hops]
    full = max(1, sum(sizes))
    out = []
    for cap_frac in (0.01, 0.03, 0.1, 0.3, 1.0, math.inf):
        cap = math.inf if math.isinf(cap_frac) else cap_frac * full
        for ev in ("lru", "lfu", "fifo", "gdsf", "benefit"):
            if math.isinf(cap) and ev != "gdsf":
                continue
            out.append({"study": "capacity", "cap_frac": None if math.isinf(cap_frac) else cap_frac,
                        **simulate(stream, tokens, rounds, relsim, capacity=cap, eviction=ev, validity="trust",
                                   objects=args.objects)})
    for rate in (0.0, 1.0, 5.0, 20.0):
        for val in ("trust", "validate", "ttl", "piggyback"):
            for ttl in ((25, 100, 400) if val in ("ttl", "piggyback") else (0,)):
                out.append({"study": "validity", **simulate(stream, tokens, rounds, relsim, validity=val, ttl=ttl,
                                                            update_rate=rate, objects=args.objects)})
    for th in (0.75, 0.8, 0.85, 0.9, 0.95, 1.0):
        out.append({"study": "rel_sim", **simulate(stream, tokens, rounds, relsim, rel_sim=th, validity="trust",
                                                   objects=args.objects)})
    prof = reuse_profile(stream, relsim)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
        f.write(json.dumps({"study": "reuse_profile", **prof}) + "\n")
    print(f"{len(out)} configurations, reuse profile {prof}: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
