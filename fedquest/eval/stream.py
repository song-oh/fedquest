# stream runs: per-question KG views, flushed in batches behind shard barriers
from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import socket
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from fedquest.kg.memory import MemoryKG
from fedquest.kg.normalize import norm_entity
from fedquest.kg.view import AskRecord, Ref, materializable
from fedquest.types import Candidate, ExtractionResult, Fact

log = logging.getLogger(__name__)

BARRIER_POLL_S = 2.0
BARRIER_TIMEOUT_S = 1800.0
PHASES = ("ran", "done")
WRITES = ("add", "add_asks", "add_probes", "add_aliases", "add_index_stats", "stale", "add_tuples")


def barrier_dir() -> Path:
    from fedquest.paths import runs_root

    return Path(os.environ.get("FEDQUEST_BARRIER_DIR") or runs_root() / "barriers")


def barrier_timeout() -> float:
    text = os.environ.get("FEDQUEST_BARRIER_TIMEOUT_S", "").strip()
    return float(text) if text else BARRIER_TIMEOUT_S


def _first(order: Iterable[Any], held: Mapping[Any, Any], mine: Mapping[Any, Any]) -> dict[Any, Any]:
    out = {k: held[k] if k in held else mine[k] for k in order if k in held or k in mine}
    for k, v in [*held.items(), *mine.items()]:
        out.setdefault(k, v)
    return out


def _summed(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        if k not in out:
            out[k] = v
        elif isinstance(v, dict) and isinstance(out[k], dict):
            out[k] = _summed(out[k], v)
        elif isinstance(v, (int, float)) and not isinstance(v, bool) and isinstance(out[k], (int, float)):
            out[k] = out[k] + v
    return out


# writes go to a private view and queue in pending until flush replays them on the shared KG
class StreamKG:
    def __init__(self, base: Any) -> None:
        self.base = base
        self.own = MemoryKG(writer=base.writer)
        self.pending: list[tuple[str, tuple[Any, ...]]] = []
        self._cover: dict[str, dict[Ref, str]] = defaultdict(dict)
        self._marks: dict[str, dict[Ref, set[str]]] = defaultdict(lambda: defaultdict(set))

    @property
    def writer(self) -> str:
        return self.base.writer

    def _keep(self, name: str, *args: Any) -> None:
        self.pending.append((name, copy.deepcopy(args)))

    def _own_stale(self, ns: str, f: Fact) -> bool:
        marks = self._marks.get(ns, {}).get((f.site, f.oid))
        return bool(f.overs and marks and any(v != f.overs for v in marks))

    async def flush(self) -> int:
        todo, self.pending = self.pending, []
        for name, args in todo:
            await getattr(self.base, name)(*args)
        return len(todo)

    async def add(self, ns: str, results: Sequence[ExtractionResult]) -> None:
        self._keep("add", ns, list(results))
        for r in results:
            if materializable(r):
                self._cover[ns][(r.site, r.oid)] = r.version
        await self.own.add(ns, results)

    async def stale(self, ns: str, refs_versions: Mapping[Ref, str]) -> int:
        self._keep("stale", ns, dict(refs_versions))
        for ref, v in refs_versions.items():
            if v:
                self._marks[ns][ref].add(v)
        return await self.own.stale(ns, refs_versions)

    async def add_asks(self, ns: str, records: Sequence[AskRecord], entries: Sequence[Candidate]) -> None:
        self._keep("add_asks", ns, list(records), list(entries))
        await self.own.add_asks(ns, records, entries)

    async def add_probes(self, ns: str, results: dict[str, list[Candidate]]) -> None:
        self._keep("add_probes", ns, dict(results))
        await self.own.add_probes(ns, results)

    async def add_aliases(self, ns: str, links: Mapping[str, str]) -> None:
        self._keep("add_aliases", ns, dict(links))
        await self.own.add_aliases(ns, links)

    async def add_index_stats(self, ns: str, deltas: Mapping[str, tuple[int, int] | tuple[int, int, float]]) -> None:
        self._keep("add_index_stats", ns, dict(deltas))
        await self.own.add_index_stats(ns, deltas)

    async def add_tuples(self, ns: str, items: Mapping[str, dict[str, Any]]) -> None:
        self._keep("add_tuples", ns, dict(items))
        await self.own.add_tuples(ns, items)

    async def clear(self, ns: str) -> None:
        raise RuntimeError(f"a stream question's knowledge view clears no namespace ({ns}): the runner does")

    async def close(self) -> None:
        return None

    async def lookup(self, ns: str, entities: Sequence[str], limit: int = 200, *,
                     include_stale: bool = False) -> list[Fact]:
        entities = list(entities)
        mine = await self.own.lookup(ns, entities, limit, include_stale=include_stale)
        want = limit
        while True:
            got = await self.base.lookup(ns, entities, want, include_stale=include_stale)
            held = got if include_stale else [f for f in got if not self._own_stale(ns, f)]
            if len(held) >= limit or len(got) < want:
                break
            want *= 2
        return list(dict.fromkeys([*held, *mine]))[:limit]

    async def covered(self, ns: str, refs: Sequence[Ref], version: str) -> dict[Ref, list[Fact]]:
        refs = list(refs)
        held = await self.base.covered(ns, refs, version)
        cover = self._cover.get(ns)
        if not cover:
            return held
        mine = await self.own.covered(ns, refs, version)
        out: dict[Ref, list[Fact]] = {}
        for ref in refs:
            v = cover.get(ref)
            if v is None:
                if ref in held:
                    out[ref] = held[ref]
            elif v == version:
                out[ref] = list(dict.fromkeys([*held.get(ref, []), *mine.get(ref, [])]))
        return out

    async def facts(self, ns: str) -> list[Fact]:
        return list(dict.fromkeys([*await self.base.facts(ns), *await self.own.facts(ns)]))

    async def versions(self, ns: str, refs: Sequence[Ref]) -> dict[Ref, str]:
        refs = list(dict.fromkeys(refs))
        mine = await self.own.versions(ns, refs)
        rest = [r for r in refs if r not in mine]
        held = await self.base.versions(ns, rest) if rest else {}
        return {r: mine[r] if r in mine else held[r] for r in refs if r in mine or r in held}

    async def _per_fact(self, name: str, ns: str, facts: Sequence[Fact]) -> tuple[dict[Fact, Any], dict[Fact, Any]]:
        facts = list(dict.fromkeys(facts))
        held = await getattr(self.base, name)(ns, facts) if facts else {}
        rest = [f for f in facts if f not in held]
        return held, (await getattr(self.own, name)(ns, rest) if rest else {})

    async def writers(self, ns: str, facts: Sequence[Fact]) -> dict[Fact, str]:
        held, mine = await self._per_fact("writers", ns, facts)
        return _first(dict.fromkeys(facts), held, mine)

    async def qids(self, ns: str, facts: Sequence[Fact]) -> dict[Fact, str]:
        held, mine = await self._per_fact("qids", ns, facts)
        return _first(dict.fromkeys(facts), held, mine)

    async def provenance(self, ns: str, facts: Sequence[Fact]) -> dict[Fact, dict[str, Any]]:
        held, mine = await self._per_fact("provenance", ns, facts)
        held = {f: {**p, "stale": True} if self._own_stale(ns, f) else p for f, p in held.items()}
        return _first(dict.fromkeys(facts), held, mine)

    async def fact_stats(self, ns: str) -> dict[str, Any]:
        return _summed(await self.base.fact_stats(ns), await self.own.fact_stats(ns))

    async def stats(self, ns: str | None = None) -> dict[str, Any]:
        return _summed(await self.base.stats(ns), await self.own.stats(ns))

    async def asks(self, ns: str, values: Sequence[str]) -> list[AskRecord]:
        values = list(values)
        held = await self.base.asks(ns, values)
        mine = {(r.value, r.relation, r.head): r for r in await self.own.asks(ns, values)}
        if not mine:
            return held
        out = [r.merged(mine.pop((r.value, r.relation, r.head))) if (r.value, r.relation, r.head) in mine else r
               for r in held]
        return out + list(mine.values())

    async def nodes(self, ns: str) -> list[str]:
        return list(dict.fromkeys([*await self.base.nodes(ns), *await self.own.nodes(ns)]))

    async def index_stats(self, ns: str) -> dict[str, tuple[int, int, float]]:
        out = {c: (int(v[0]), int(v[1]), float((*v, 0.0)[2])) for c, v in (await self.base.index_stats(ns)).items()}
        for c, (n, w, e) in (await self.own.index_stats(ns)).items():
            a = out.get(c, (0, 0, 0.0))
            out[c] = (a[0] + n, a[1] + w, a[2] + e)
        return out

    async def entries(self, ns: str, refs: Sequence[Ref]) -> dict[Ref, Candidate]:
        refs = list(dict.fromkeys(refs))
        held = await self.base.entries(ns, refs)
        rest = [r for r in refs if r not in held]
        return _first(refs, held, await self.own.entries(ns, rest) if rest else {})

    async def probes(self, ns: str, keys: Sequence[str]) -> dict[str, list[Candidate]]:
        keys = list(dict.fromkeys(keys))
        held = await self.base.probes(ns, keys)
        rest = [k for k in keys if k not in held]
        return _first(keys, held, await self.own.probes(ns, rest) if rest else {})

    async def tuples(self, ns: str, keys: Sequence[str]) -> dict[str, dict[str, Any]]:
        keys = list(dict.fromkeys(keys))
        held = await self.base.tuples(ns, keys)
        rest = [k for k in keys if k not in held]
        return _first(keys, held, await self.own.tuples(ns, rest) if rest else {})

    async def aliases(self, ns: str, names: Sequence[str]) -> dict[str, str]:
        names = list(names)
        held = await self.base.aliases(ns, names)
        return _first(dict.fromkeys(norm_entity(n) for n in names), held, await self.own.aliases(ns, names))

    async def known(self, ns: str, names: Sequence[str]) -> set[str]:
        names = list(names)
        return set(await self.base.known(ns, names)) | set(await self.own.known(ns, names))


class Barrier:
    def __init__(self, root: Path, shard: int, shards: int, *, timeout: float | None = None,
                 poll: float | None = None) -> None:
        self.root, self.shard, self.shards = Path(root), shard, shards
        self.timeout = barrier_timeout() if timeout is None else float(timeout)
        self.poll = BARRIER_POLL_S if poll is None else float(poll)
        self.late: set[int] = set()

    def path(self, batch: int, phase: str, shard: int | None = None) -> Path:
        return self.root / f"b{batch:04d}" / f"shard{self.shard if shard is None else shard:02d}.{phase}"

    def mark(self, batch: int, phase: str, **info: Any) -> Path:
        path = self.path(batch, phase)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps({"shard": self.shard, "shards": self.shards, "batch": batch, "phase": phase,
                                   "host": socket.gethostname(), "pid": os.getpid(), "at": time.time(), **info}))
        tmp.replace(path)
        return path

    def _counts(self, path: Path) -> bool:
        try:
            doc = json.loads(path.read_text())
        except FileNotFoundError:
            return False
        except (OSError, ValueError):
            doc = {}
        return int(doc.get("shards", self.shards)) == self.shards

    def present(self, batch: int, phase: str) -> set[int]:
        return {i for i in range(self.shards) if self._counts(self.path(batch, phase, i))}

    def past(self, batch: int, ran: bool = False) -> set[int]:
        out = self.present(batch, "done") | (self.present(batch, "ran") if ran else set())
        for d in self.root.glob("b[0-9][0-9][0-9][0-9]"):
            if int(d.name[1:]) > batch:
                out |= {i for i in range(self.shards) for ph in PHASES if self._counts(self.path(int(d.name[1:]), ph, i))}
        return out

    def unmark(self, batch: int) -> None:
        for phase in PHASES:
            self.path(batch, phase).unlink(missing_ok=True)

    async def wait(self, batch: int, phase: str) -> set[int]:
        t0 = time.monotonic()
        while True:
            have = self.present(batch, phase)
            self.late -= have
            missing = set(range(self.shards)) - have
            # shards already late at an earlier barrier are not waited for again
            if not missing - self.late:
                return missing
            if time.monotonic() - t0 >= self.timeout:
                log.warning("%s: barrier %s of batch %d: shards %s missing after %.0f s: proceeding without them (what "
                            "they write may reach later batches late)", self.root, phase, batch, sorted(missing),
                            self.timeout)
                self.late |= missing
                return missing
            await asyncio.sleep(self.poll)

    def forget(self) -> int:
        gone = 0
        for p in self.root.glob(f"b*/shard{self.shard:02d}.*"):
            p.unlink(missing_ok=True)
            gone += 1
        return gone
