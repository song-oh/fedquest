# wire records (probes, candidates, facts, reads) and the query graph's edges and pattern
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Literal, TypeVar

Modality = Literal["text", "table", "image"]
ProbeKind = Literal["question", "bound", "title"]
ReleasePolicy = Literal["snippets", "facts", "sealed"]
ExtractMode = Literal["kg", "query", "query_t", "query_block"]

T = TypeVar("T", bound="Record")


class Record:
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)  # type: ignore[call-overload]

    @classmethod
    def from_dict(cls: type[T], d: dict[str, Any]) -> T:
        names = {f.name for f in fields(cls)}  # type: ignore[arg-type]
        return cls(**{k: v for k, v in d.items() if k in names})


@dataclass
class ObjectRecord(Record):
    oid: str
    doc_id: str
    modality: Modality
    title: str
    text: str
    image_path: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def version(self) -> str:
        from fedquest.utils import stable_hash

        return stable_hash(self.oid, self.title, self.text, self.image_path or "", n=12)


@dataclass
class Question(Record):
    qid: str
    dataset: str
    question: str
    answers: list[str]
    qtype: str = ""
    gold_doc_ids: list[str] = field(default_factory=list)
    bridge: list[str] = field(default_factory=list)
    session: str = ""
    image_answer: bool = False
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class CollectionInfo(Record):
    name: str
    modality: Modality
    encoder: str
    size: int


@dataclass
class SiteInfo(Record):
    name: str
    machine: str
    policy: ReleasePolicy
    collections: list[CollectionInfo]
    extractor_version: str
    reader_version: str = ""
    models: dict[str, str] = field(default_factory=dict)
    trust: str = ""
    wire: str = ""
    commit: str = ""

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SiteInfo:
        d = dict(d)
        d["collections"] = [c if isinstance(c, CollectionInfo) else CollectionInfo.from_dict(c)
                            for c in d.get("collections", [])]
        return super().from_dict(d)


WIRE_V5: dict[str, Any] = {"semijoin": False, "k_group": None, "group_size": None, "encoder": None, "mention": 1,
                           "scores": False, "epoch": None, "versions": False, "zvariants": False, "views": False,
                           "room": None, "view_key": None, "signal": None}


@dataclass
class ProbeRequest(Record):
    key: str
    kind: ProbeKind
    k: int = 5
    offset: int = 0
    collections: list[str] | None = None
    titles: list[str] = field(default_factory=list)
    mentions: list[str] = field(default_factory=list)
    value: str | None = None
    sites: list[str] | None = None
    semijoin: bool = False
    k_group: int | None = None
    group_size: int | None = None
    encoder: str | None = None
    mention: int = 1
    scores: bool = False
    epoch: str | None = None
    versions: bool = False
    zvariants: bool = False
    views: bool = False
    room: int | None = None
    view_key: str | None = None
    signal: list[str] | None = None
    vocab: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        # value, sites and vocab never leave the coordinator; fields at their WIRE_V5 default are left out
        d = asdict(self)
        for name in ("value", "sites", "vocab"):
            d.pop(name, None)
        for name, default in WIRE_V5.items():
            if d[name] == default:
                del d[name]
        return d


@dataclass
class Candidate(Record):
    site: str
    collection: str
    oid: str
    modality: Modality
    rank: int
    summary: str
    tokens: int
    doc_id: str = ""
    complete: bool = False
    label: str = ""
    mentions: bool | None = None
    jrank: int | None = None
    z: float | None = None
    ver: str | None = None
    zv: dict[str, float] | None = None
    view: bool | None = None
    lead_signal: bool | None = None

    @property
    def ref(self) -> tuple[str, str]:
        return (self.site, self.oid)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if d["jrank"] is None:
            del d["jrank"]
        if d["z"] is None:
            del d["z"]
        if d["ver"] is None:
            del d["ver"]
        if d["zv"] is None:
            del d["zv"]
        for name in ("view", "lead_signal"):
            if d[name] is None:
                del d[name]
        return d


@dataclass
class ProbeCounts(Record):
    collection: str
    joinable: int
    returned: int
    exact: bool = True
    site: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if not d["site"]:
            del d["site"]
        return d


@dataclass
class HippoRequest(Record):
    question: str
    k: int = 5
    collections: list[str] | None = None


@dataclass(frozen=True)
class Fact(Record):
    head: str
    relation: str
    tail: str
    site: str = ""
    oid: str = ""
    version: str = ""
    modality: str = ""
    overs: str = ""
    wspan: str = ""

    def text(self) -> str:
        return f"{self.head} | {self.relation} | {self.tail}"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if not d["wspan"]:
            del d["wspan"]
        return d


@dataclass
class Usage(Record):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    seconds: float = 0.0
    calls: int = 0
    bytes_in: int = 0
    bytes_out: int = 0

    @property
    def cost_units(self) -> int:
        # the same weight as budget.OUT_WEIGHT
        return self.prompt_tokens + 4 * self.completion_tokens

    def add(self, other: Usage) -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.reasoning_tokens += other.reasoning_tokens
        self.seconds += other.seconds
        self.calls += other.calls
        self.bytes_in += other.bytes_in
        self.bytes_out += other.bytes_out


@dataclass
class ExtractionResult(Record):
    site: str
    oid: str
    modality: Modality
    version: str
    facts: list[Fact]
    usage: Usage
    replay: bool = False
    raw_text: str | None = None
    doc_id: str = ""
    error: str | None = None
    inference_bytes: int = 0
    trust: str = ""
    entities: list[list[str]] = field(default_factory=list)
    qid: str = ""

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ExtractionResult:
        d = dict(d)
        d["facts"] = [f if isinstance(f, Fact) else Fact.from_dict(f) for f in d.get("facts", [])]
        u = d.get("usage")
        d["usage"] = u if isinstance(u, Usage) else Usage.from_dict(u or {})
        return super().from_dict(d)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["facts"] = [f.to_dict() for f in self.facts]
        return d


@dataclass
class ReadResult(Record):
    site: str
    read: list[str]
    bindings: dict[str, list[dict[str, Any]]]
    usage: Usage
    replay: bool = False
    raw: dict[str, str] = field(default_factory=dict)
    modalities: dict[str, str] = field(default_factory=dict)
    doc_ids: dict[str, str] = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)
    usage_by_modality: dict[str, Usage] = field(default_factory=dict)
    dropped: int = 0
    titles: dict[str, str] = field(default_factory=dict)
    extractions: list[ExtractionResult] = field(default_factory=list)
    error: str | None = None
    inference_bytes: int = 0
    trust: str = ""
    versions: dict[str, str] = field(default_factory=dict)
    harvest: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ReadResult:
        d = dict(d)
        u = d.get("usage")
        d["usage"] = u if isinstance(u, Usage) else Usage.from_dict(u or {})
        d["extractions"] = [x if isinstance(x, ExtractionResult) else ExtractionResult.from_dict(x)
                            for x in d.get("extractions") or []]
        d["usage_by_modality"] = {m: v if isinstance(v, Usage) else Usage.from_dict(v)
                                  for m, v in (d.get("usage_by_modality") or {}).items()}
        return super().from_dict(d)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for name in ("versions", "harvest"):
            if not d[name]:
                del d[name]
        return d


@dataclass
class RawObject(Record):
    site: str
    oid: str
    modality: Modality
    title: str
    text: str | None = None
    image_b64: str | None = None
    nbytes: int = 0
    denied: bool = False
    doc_id: str = ""


def is_var(node: str) -> bool:
    return node.startswith("?")


@dataclass
class Edge(Record):
    id: int
    head: str
    relation: str
    tail: str
    modality: str | None = None

    def endpoints(self) -> tuple[str, str]:
        return (self.head, self.tail)

    def other(self, node: str) -> str:
        return self.tail if node == self.head else self.head


@dataclass
class Pattern(Record):
    anchors: list[str]
    variables: list[str]
    edges: list[Edge]
    answer: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)
    conditions: dict[str, list[str]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Pattern:
        d = dict(d)
        d["edges"] = [e if isinstance(e, Edge) else Edge.from_dict(e) for e in d.get("edges", [])]
        return super().from_dict(d)

    def incident(self, node: str) -> list[Edge]:
        return [e for e in self.edges if node in (e.head, e.tail)]
