# the KnowledgeView protocol that the memory and Neo4j stores implement
from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from fedquest.types import Candidate, ExtractionResult, Fact

Ref = tuple[str, str]

KEEP_VERSION = "mgrag-keep-v1"


def default_writer() -> str:
    return os.environ.get("FEDQUEST_SHARD") or str(os.getpid())


def harvested(f: Fact) -> bool:
    return f.version.endswith("+harvest")


def materializable(r: ExtractionResult) -> bool:
    # MGRAG's query-conditioned extractions never enter the KG; its keep version does
    return r.error is None and (not r.version.startswith("mgrag") or r.version == KEEP_VERSION)


@dataclass
class AskRecord:
    value: str
    relation: str
    head: bool
    witnesses: dict[Ref, list[str]] = field(default_factory=dict)
    chance: dict[Ref, float] = field(default_factory=dict)
    negatives: set[Ref] = field(default_factory=set)

    def merged(self, other: AskRecord) -> AskRecord:
        witnesses = {ref: list(dict.fromkeys([*self.witnesses.get(ref, []), *other.witnesses.get(ref, [])]))
                     for ref in dict.fromkeys([*self.witnesses, *other.witnesses])}
        chance = {ref: max(self.chance.get(ref, 0.0), other.chance.get(ref, 0.0)) for ref in witnesses}
        return AskRecord(self.value, self.relation, self.head, witnesses, chance,
                         (self.negatives | other.negatives) - set(witnesses))


class KnowledgeView(Protocol):
    writer: str

    async def covered(self, ns: str, refs: Sequence[Ref], version: str) -> dict[Ref, list[Fact]]:
        ...

    async def lookup(self, ns: str, entities: Sequence[str], limit: int = 200, *,
                     include_stale: bool = False) -> list[Fact]:
        ...

    async def add(self, ns: str, results: Sequence[ExtractionResult]) -> None:
        ...

    async def facts(self, ns: str) -> list[Fact]:
        ...

    async def stale(self, ns: str, refs_versions: Mapping[Ref, str]) -> int:
        ...

    async def versions(self, ns: str, refs: Sequence[Ref]) -> dict[Ref, str]:
        ...

    async def writers(self, ns: str, facts: Sequence[Fact]) -> dict[Fact, str]:
        ...

    async def qids(self, ns: str, facts: Sequence[Fact]) -> dict[Fact, str]:
        ...

    async def provenance(self, ns: str, facts: Sequence[Fact]) -> dict[Fact, dict[str, Any]]:
        ...

    async def fact_stats(self, ns: str) -> dict[str, Any]:
        ...

    async def stats(self, ns: str | None = None) -> dict[str, Any]:
        ...

    async def clear(self, ns: str) -> None:
        ...

    async def close(self) -> None: ...

    async def asks(self, ns: str, values: Sequence[str]) -> list[AskRecord]:
        ...

    async def nodes(self, ns: str) -> list[str]:
        ...

    async def index_stats(self, ns: str) -> dict[str, tuple[int, int, float]]:
        ...

    async def add_index_stats(self, ns: str, deltas: Mapping[str, tuple[int, int] | tuple[int, int, float]]) -> None:
        ...

    async def add_asks(self, ns: str, records: Sequence[AskRecord], entries: Sequence[Candidate]) -> None:
        ...

    async def entries(self, ns: str, refs: Sequence[Ref]) -> dict[Ref, Candidate]:
        ...

    async def probes(self, ns: str, keys: Sequence[str]) -> dict[str, list[Candidate]]:
        ...

    async def add_probes(self, ns: str, results: dict[str, list[Candidate]]) -> None:
        ...

    async def tuples(self, ns: str, keys: Sequence[str]) -> dict[str, dict[str, Any]]:
        ...

    async def add_tuples(self, ns: str, items: Mapping[str, dict[str, Any]]) -> None:
        ...

    async def aliases(self, ns: str, names: Sequence[str]) -> dict[str, str]:
        ...

    async def add_aliases(self, ns: str, links: Mapping[str, str]) -> None:
        ...

    async def known(self, ns: str, names: Sequence[str]) -> set[str]:
        ...


def open_view(kind: Literal["memory", "neo4j"] = "memory") -> KnowledgeView:
    if kind == "memory":
        from fedquest.kg.memory import MemoryKG

        return MemoryKG()
    from fedquest.kg.neo4j_store import Neo4jKG

    uri, password = os.environ.get("NEO4J_URI"), os.environ.get("NEO4J_PASSWORD")
    if not uri or not password:
        raise RuntimeError("no FedQuest Neo4j configured: run `fedquest kg up` (it records NEO4J_URI and "
                           "NEO4J_PASSWORD for its dedicated database in .fedquest.env)")
    return Neo4jKG(uri, os.environ.get("NEO4J_USER", "neo4j"), password, os.environ.get("NEO4J_DATABASE") or None)
