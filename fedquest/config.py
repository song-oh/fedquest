from __future__ import annotations

import hashlib
import logging
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal, get_args

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator

from fedquest.paths import CONFIGS, calibration_path, collection_dir, site_dir
from fedquest.types import Modality, ReleasePolicy

log = logging.getLogger(__name__)


class _Cfg(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CollectionCfg(_Cfg):
    name: str
    modality: Modality
    encoder: str
    source: str
    vectors: Literal["fused", "pixels", "titles"] = "fused"
    vectors_from: str | None = None
    shard: str | None = None
    keep: str | None = None
    fraction: float | None = None
    share: str | None = None

    @model_validator(mode="after")
    def _check(self) -> CollectionCfg:
        if self.shard is not None:
            j, _, m = self.shard.partition("/")
            if not (j.isdigit() and m.isdigit() and 0 <= int(j) < int(m)):
                raise ValueError(f"collection {self.name!r}: shard {self.shard!r} is not j/m with 0 <= j < m")
        if self.fraction is not None and not 0.0 < self.fraction <= 1.0:
            raise ValueError(f"collection {self.name!r}: fraction {self.fraction!r} is not in (0, 1]")
        if self.share is not None:
            from fedquest.data.partition import SHARE_SCALE

            lo, _, hi = self.share.partition(":")
            if not (lo.isdigit() and hi.isdigit() and 0 <= int(lo) < int(hi) <= SHARE_SCALE):
                raise ValueError(f"collection {self.name!r}: share {self.share!r} is not lo:hi with "
                                 f"0 <= lo < hi <= {SHARE_SCALE}")
        if self.keep is not None:
            names = [s.strip() for s in self.keep.split(",")]
            known = get_args(ExperimentCfg.model_fields["split"].annotation)
            if self.fraction is None or not all(names) or any(s not in known for s in names):
                raise ValueError(f"collection {self.name!r}: keep {self.keep!r} names the splits (comma-separated: "
                                 f"{', '.join(known)}) whose ground-truth pages a fraction keeps, and needs a fraction")
        if self.vectors != "fused":
            from fedquest.vdb.encoders import ENCODERS

            spec = ENCODERS.get(self.encoder)
            if self.modality != "image" or (spec is not None and spec.kind != "clip"):
                raise ValueError(f"collection {self.name!r}: vectors {self.vectors!r} needs an image collection with "
                                 f"a CLIP encoder (has {self.modality}, {self.encoder})")
        if self.vectors_from is not None and len([p for p in self.vectors_from.split("/") if p]) != 3:
            raise ValueError(f"collection {self.name!r}: vectors_from {self.vectors_from!r} is not "
                             "<federation>/<site>/<collection>")
        return self

    @model_serializer(mode="wrap")
    def _dump(self, handler: Any) -> dict[str, Any]:
        out = handler(self)
        if isinstance(out, dict):
            for name in ("keep", "fraction", "share"):
                if getattr(self, name) is None:
                    out.pop(name, None)
        return out

    @property
    def parent(self) -> tuple[str, str, str] | None:
        if not self.vectors_from:
            return None
        fed, site, coll = self.vectors_from.split("/")
        return fed, site, coll

    @property
    def shard_of(self) -> tuple[int, int] | None:
        if self.shard is None:
            return None
        j, _, m = self.shard.partition("/")
        return int(j), int(m)

    @property
    def share_of(self) -> tuple[int, int] | None:
        if self.share is None:
            return None
        lo, _, hi = self.share.partition(":")
        return int(lo), int(hi)

    @property
    def sources(self) -> list[tuple[str, int | None]]:
        out = []
        for term in self.source.split(","):
            kind, _, part = term.strip().partition(":")
            out.append((kind, None if part in ("", "*") else int(part)))
        return out

    @property
    def source_kind(self) -> str:
        return self.sources[0][0]

    @property
    def source_part(self) -> int | None:
        return self.sources[0][1]


class SiteCfg(_Cfg):
    name: str
    machine: str
    policy: ReleasePolicy = "snippets"
    collections: list[CollectionCfg]
    max_concurrency: int = 16
    models: str | None = None
    domain: str | None = None
    data_from: str | None = None

    @model_validator(mode="after")
    def _check(self) -> SiteCfg:
        if self.data_from is not None:
            parts = self.data_from.split("/")
            if len(parts) != 2 or not all(p.strip() for p in parts):
                raise ValueError(f"site {self.name!r}: data_from {self.data_from!r} is not <federation>/<site>")
        return self

    @model_serializer(mode="wrap")
    def _dump(self, handler: Any) -> dict[str, Any]:
        out = handler(self)
        if self.data_from is None and isinstance(out, dict):
            out.pop("data_from", None)
        return out

    @property
    def data_domain(self) -> str:
        return self.domain or self.machine

    @property
    def data_source(self) -> tuple[str, str] | None:
        if not self.data_from:
            return None
        fed, src = self.data_from.split("/")
        return fed, src

    def collection(self, name: str) -> CollectionCfg:
        for c in self.collections:
            if c.name == name:
                return c
        raise KeyError(f"site {self.name!r} has no collection {name!r}")


SPLITS_WITH_PARAGRAPHS = ("final_test",)


class CorpusCfg(_Cfg):
    scope: Literal["pooled", "full"] = "full"
    extra_texts: int = 0
    extra_tables: int = 0
    extra_images: int = 0
    chunk_tokens: int = 256
    chunk_overlap: int = 32
    distractors: int = 0
    paragraphs_of: str | None = None
    calib_paragraphs: bool = True

    @model_validator(mode="after")
    def _check(self) -> CorpusCfg:
        if self.paragraphs_of is not None:
            names = [s.strip() for s in self.paragraphs_of.split(",")]
            if not all(names) or any(s not in SPLITS_WITH_PARAGRAPHS for s in names):
                raise ValueError(f"corpus: paragraphs_of {self.paragraphs_of!r} names the drawn splits whose "
                                 "questions' paragraphs the corpus adds (comma-separated: "
                                 f"{', '.join(SPLITS_WITH_PARAGRAPHS)})")
        return self

    @model_serializer(mode="wrap")
    def _dump(self, handler: Any) -> dict[str, Any]:
        out = handler(self)
        if self.paragraphs_of is None and isinstance(out, dict):
            out.pop("paragraphs_of", None)
        if self.calib_paragraphs and isinstance(out, dict):
            out.pop("calib_paragraphs", None)
        return out

    @property
    def paragraph_splits(self) -> list[str]:
        return [s.strip() for s in (self.paragraphs_of or "").split(",") if s.strip()]


class QuestionsCfg(_Cfg):
    split: str = "dev"
    n: int | None = 300
    types: list[str] | None = None
    stratify: bool = True
    quotas: dict[str, int] | None = None
    subset_file: str | None = None
    calib_split: str = "train"
    calib_n: int = 100
    calib_doc_fraction: float = 0.2
    seed: int = 13
    order: Literal["file", "seeded"] = "file"

    @model_serializer(mode="wrap")
    def _dump(self, handler: Any) -> dict[str, Any]:
        out = handler(self)
        if self.order == "file" and isinstance(out, dict):
            out.pop("order", None)
        return out


class PartitionCfg(_Cfg):
    method: Literal["by_title", "dirichlet", "single"] = "by_title"
    parts: int = 3
    beta: float = 0.5
    clusters: int = 8
    seed: int = 13
    shard_seed: int | None = None
    share_seed: int | None = None

    @model_serializer(mode="wrap")
    def _dump(self, handler: Any) -> dict[str, Any]:
        out = handler(self)
        if isinstance(out, dict):
            for name in ("shard_seed", "share_seed"):
                if getattr(self, name) is None:
                    out.pop(name, None)
        return out

    @property
    def shard_hash_seed(self) -> int:
        return self.seed if self.shard_seed is None else self.shard_seed

    @property
    def share_hash_seed(self) -> int:
        return self.seed if self.share_seed is None else self.share_seed


class IndexCfg(_Cfg):
    kind: Literal["flat", "hnsw", "sq8", "pq", "bq"] = "flat"
    m: int = 16
    ef_construct: int = 100
    ef: int = 64
    rescore: bool = True
    oversampling: float = 2.0
    pq_compression: Literal["x4", "x8", "x16", "x32", "x64"] = "x16"

    def tag(self) -> str:
        if self.kind == "flat":
            return ""
        base = f"{self.kind}_m{self.m}_ef{self.ef}"
        return base + (f"_{self.pq_compression}" if self.kind == "pq" else "")


class QdrantCfg(_Cfg):
    mode: Literal["local", "server", "array"] = "server"
    url: str | None = None
    exact: bool = True
    index: IndexCfg | None = None

    @property
    def approximate(self) -> bool:
        return self.index is not None and self.index.kind != "flat"

    def resolved(self) -> QdrantCfg:
        return self.model_copy(
            update={
                "mode": os.environ.get("FEDQUEST_QDRANT_MODE", self.mode),
                "url": os.environ.get("FEDQUEST_QDRANT_URL", self.url),
            }
        )


class FederationCfg(_Cfg):
    name: str
    dataset: Literal["mmqa", "mramg", "musique", "hotpotqa", "twowiki", "msmarco", "m3sciqa", "toy"]
    subset: str | None = None
    answer_style: Literal["short", "long"] = "short"
    extractor_version: str = "kg-v3"
    corpus: CorpusCfg = Field(default_factory=CorpusCfg)
    questions: QuestionsCfg = Field(default_factory=QuestionsCfg)
    partition: PartitionCfg = Field(default_factory=PartitionCfg)
    qdrant: QdrantCfg = Field(default_factory=QdrantCfg)
    data_from: str | None = None
    calibration: str | None = None
    sites: list[SiteCfg]

    @model_validator(mode="after")
    def _check(self) -> FederationCfg:
        names = [s.name for s in self.sites]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate site names in {self.name}: {names}")
        if self.dataset == "mramg" and not self.subset:
            raise ValueError("MRAMG federations need a subset")
        if self.data_from == self.name:
            raise ValueError(f"{self.name}: data_from names the federation itself")
        own = [s.name for s in self.sites if s.data_source and s.data_source[0] == self.name]
        if own:
            raise ValueError(f"{self.name}: sites {own} read data_from a site of their own federation")
        if self.calibration is not None and (not self.calibration or "/" in self.calibration
                                             or self.calibration.endswith(".json")):
            raise ValueError(f"{self.name}: calibration {self.calibration!r} is a name under configs/calibration/ "
                             "(e.g. mmqa), not a path")
        if self.corpus.paragraphs_of is not None and self.dataset != "musique":
            raise ValueError(f"{self.name}: corpus.paragraphs_of adds MuSiQue questions' paragraphs; a "
                             f"{self.dataset} corpus already holds its questions' evidence")
        return self

    def site(self, name: str) -> SiteCfg:
        for s in self.sites:
            if s.name == name:
                return s
        raise KeyError(f"federation {self.name!r} has no site {name!r}")

    @property
    def data_name(self) -> str:
        return self.data_from or self.name

    def site_source(self, site: str) -> tuple[str, str]:
        found = next((s for s in self.sites if s.name == site), None)
        source = found.data_source if found is not None else None
        return source or (self.data_name, site)

    def site_data_dir(self, site: str) -> Path:
        return site_dir(*self.site_source(site))

    def collection_data_dir(self, site: str, collection: str) -> Path:
        return collection_dir(*self.site_source(site), collection)

    def store_dir(self, site: str) -> Path:
        own = os.environ.get("FEDQUEST_SITE_STORE_DIR")
        if own:
            return Path(own).expanduser() / self.data_name / "sites" / site
        return site_dir(self.data_name, site)

    def index_namespace(self, site: str) -> str:
        tag = self.qdrant.index.tag() if self.qdrant.approximate and self.qdrant.index is not None else ""
        data, source = self.site_source(site)
        return f"{data}~{tag}__{source}" if tag else f"{data}__{source}"

    def own_site(self, site: str, what: str) -> SiteCfg:
        s = self.site(site)
        if s.data_from:
            raise ValueError(f"{self.name}/{site} reads {s.data_from}'s data (data_from), which `fedquest data {what}` "
                             f"of {self.name} never touches: {what} {s.data_from.split('/')[0]} instead")
        return s

    def own_sites(self, sites: Iterable[str] | None, what: str) -> list[str]:
        wanted = [n for n in sites or () if n]
        for name in wanted:
            if any(s.name == name for s in self.sites):
                self.own_site(name, what)
        if wanted:
            return wanted
        for s in self.sites:
            if s.data_from:
                log.info("%s/%s reads %s's data (data_from): `fedquest data %s` skips it", self.name, s.name,
                         s.data_from, what)
        return [s.name for s in self.sites if not s.data_from]

    @property
    def site_names(self) -> list[str]:
        return [s.name for s in self.sites]

    def match_sites(self, patterns: Iterable[str] | None = None) -> list[str]:
        from fnmatch import fnmatchcase

        wanted = [p for p in patterns or () if p]
        if not wanted:
            return self.site_names
        unmatched = [p for p in wanted if not any(fnmatchcase(n, p) for n in self.site_names)]
        if unmatched:
            raise KeyError(f"{self.name} has no sites {sorted(unmatched)}")
        return [n for n in self.site_names if any(fnmatchcase(n, p) for p in wanted)]


ROLES = ("pattern", "extract_text", "extract_image", "answer", "answer_vision", "planner", "reader", "judge")
SITE_ROLES = ("extract_text", "extract_image")
POOL_PREFIX = "pool:"


class EndpointCfg(_Cfg):
    provider: Literal["alcf", "openai"]
    base_url: str
    api_key_env: str | None = "OPENAI_API_KEY"
    trust: str | None = None


class ModelProfileCfg(_Cfg):
    provider: Literal["alcf", "openai", "fake", "replay"] | None = None
    base_url: str | None = None
    api_key_env: str | None = None
    trust: str | None = None
    roles: dict[str, str] = Field(default_factory=dict)
    json_mode: list[str] | None = None
    model_concurrency: dict[str, int] | None = None
    max_concurrency: int | None = None
    timeout_s: float | None = None
    endpoints: dict[str, EndpointCfg] = Field(default_factory=dict)
    extra_body: dict[str, dict[str, Any]] = Field(default_factory=dict)
    headroom: dict[str, int] = Field(default_factory=dict)


class ModelsCfg(_Cfg):
    provider: Literal["alcf", "openai", "fake", "replay"] = "alcf"
    base_url: str = "https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1"
    roles: dict[str, str]
    json_mode: list[str] = Field(default_factory=list)
    extra_body: dict[str, dict[str, Any]] = Field(default_factory=dict)
    model_concurrency: dict[str, int] = Field(default_factory=dict)
    max_concurrency: int = 24
    timeout_s: float = 300.0
    max_retries: int = 6
    token_refresh_s: int = 900
    api_key_env: str | None = "OPENAI_API_KEY"
    trust: str = "alcf"
    profiles: dict[str, ModelProfileCfg] = Field(default_factory=dict)
    machines: dict[str, str] = Field(default_factory=dict)
    endpoints: dict[str, EndpointCfg] = Field(default_factory=dict)
    headroom: dict[str, int] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> ModelsCfg:
        missing = [r for r in ROLES if r not in self.roles]
        if missing:
            raise ValueError(f"models config misses roles: {missing}")
        for name, p in self.profiles.items():
            unknown = sorted(set(p.roles) - set(ROLES))
            if unknown:
                raise ValueError(f"models profile {name!r} has unknown roles {unknown}")
        undefined = sorted({p for p in self.machines.values() if p not in self.profiles})
        if undefined:
            raise ValueError(f"models config: machines name undefined profiles {undefined}")
        listed = set(self.endpoints).union(*(p.endpoints for p in self.profiles.values()))
        named = set(self.roles.values()).union(*(p.roles.values() for p in self.profiles.values()))
        unused = sorted(listed - named)
        if unused:
            raise ValueError(f"models config: endpoints name models no role uses: {unused}")
        unresolved = [f"{role}={model!r}" + (f" (profile {name})" if name else "")
                      for name, cfg in [("", self), *((n, self.profile(n)) for n in self.profiles)]
                      for role, model in cfg.roles.items() if not cfg._resolves(model)]
        if unresolved:
            raise ValueError(f"models config: roles whose model resolves to no server (a model name, and an "
                             f"http(s) base URL: its endpoint's or the config's): {unresolved}")
        return self

    def _resolves(self, model: str) -> bool:
        own = self.endpoints.get(model)
        if own is None and self.provider in ("fake", "replay"):
            return bool(model)
        url = own.base_url if own is not None else self.base_url
        return bool(model) and url.startswith(("http://", "https://", POOL_PREFIX))

    def profile(self, name: str) -> ModelsCfg:
        if name not in self.profiles:
            raise KeyError(f"no models profile {name!r} (defined: {sorted(self.profiles)})")
        p = self.profiles[name]
        merged = ("roles", "endpoints", "extra_body", "headroom")
        update: dict[str, Any] = {f: getattr(p, f) for f in p.model_fields_set
                                  if f not in merged and (getattr(p, f) is not None or f == "api_key_env")}
        return self.model_copy(update={**update, "roles": {**self.roles, **p.roles},
                                       "endpoints": {**self.endpoints, **p.endpoints},
                                       "extra_body": {**self.extra_body, **p.extra_body},
                                       "headroom": {**self.headroom, **p.headroom}})

    def fingerprint(self) -> dict[str, Any]:
        def view(c: ModelsCfg) -> dict[str, Any]:
            own = {m: e.model_dump(exclude={"api_key_env"}) for m, e in c.endpoints.items()
                   if not e.base_url.startswith(POOL_PREFIX)}
            return {"provider": c.provider, "base_url": c.base_url, "trust": c.trust, "roles": dict(c.roles),
                    "extra_body": c.extra_body, "json_mode": sorted(c.json_mode), "headroom": dict(c.headroom),
                    "endpoints": own}

        return {**view(self), "profiles": {name: view(self.profile(name)) for name in self.profiles},
                "machines": dict(self.machines)}

    def site_profile(self, site: SiteCfg) -> str | None:
        return site.models or self.machines.get(site.machine)

    def for_site(self, site: SiteCfg) -> ModelsCfg:
        name = self.site_profile(site)
        return self.profile(name) if name else self


class MatcherCfg(_Cfg):
    encoder: str = "bge-base"
    theta: float = 0.6
    max_hops: int = 3
    node_theta: float = 0.85
    verify: bool = True


class MethodSpec(_Cfg):
    method: str
    label: str | None = None
    federation: str | None = None
    sweep: bool = False
    reuse: bool = True
    params: dict[str, Any] = Field(default_factory=dict)
    preset: str | None = None
    limit: int | None = None

    def name_for(self, budget: int | None = None) -> str:
        base = self.label or self.method
        return f"{base}@{budget}" if budget is not None else base


class NetworkCfg(_Cfg):
    rtt_ms: float = 0.0
    site_rtt_ms: dict[str, float] = Field(default_factory=dict)
    site_extra_ms: dict[str, float] = Field(default_factory=dict)
    jitter: Literal["none", "lognormal"] = "none"
    sigma: float = 0.29
    mbps: float | None = None
    down: list[str] = Field(default_factory=list)
    deadline_ms: float | None = None
    endpoint_ms: float = 0.0
    seed: int = 13

    @property
    def active(self) -> bool:
        return bool(self.rtt_ms or self.site_rtt_ms or self.site_extra_ms or self.mbps or self.down or self.deadline_ms)

    def rtt_for(self, site: str) -> float:
        from fnmatch import fnmatch

        for pattern, ms in self.site_rtt_ms.items():
            if fnmatch(site, pattern):
                return float(ms)
        return float(self.rtt_ms)

    def extra_for(self, site: str) -> float:
        from fnmatch import fnmatch

        return next((float(ms) for pattern, ms in self.site_extra_ms.items() if fnmatch(site, pattern)), 0.0)


class ExperimentCfg(_Cfg):
    name: str
    federation: str
    env: dict[str, str] = Field(default_factory=dict)
    split: Literal["test", "tune", "holdout", "latency", "calib", "confirm", "latency2", "confirm14",
                   "latency3", "stream", "sessions", "confirm18",
                   "loc_eval", "loc00", "loc25", "loc50", "loc75", "loc100", "loc_smoke",
                   "final_test", "live4", "devA", "devB", "imgbridge",
                   "kg_base", "kg_update",
                   "full", "heldout"] = "test"
    limit: int | None = None
    outside_prefix: int | None = None
    families: list[str] | None = None
    workload: Literal["cold", "session", "concurrent"] = "cold"
    orders: list[Literal["natural", "random"]] = Field(default_factory=lambda: ["natural"])
    agents: list[int] = Field(default_factory=lambda: [1])
    sharing: list[Literal["shared", "private"]] = Field(default_factory=lambda: ["shared"])
    deals: list[Literal["overlap", "sessions"]] = Field(default_factory=lambda: ["overlap"])
    kg: Literal["memory", "neo4j"] = "memory"
    concurrency: int = 16
    calibration: str | None = "auto"
    require_calibration: bool = False
    budgets: list[int] = Field(default_factory=lambda: [8])
    matcher: MatcherCfg = Field(default_factory=MatcherCfg)
    network: NetworkCfg = Field(default_factory=NetworkCfg)
    interleave: bool = False
    interleave_seed: str | None = None
    live_wire: bool = False
    canary_s: float | None = None
    timing_from: list[str] = Field(default_factory=list)
    site_profile: str | None = None
    models: str | None = None
    lane: str = "sophia"
    stage: int = 0
    after: list[str] = Field(default_factory=list)
    before: list[str] = Field(default_factory=list)
    judge: bool = True
    optional: bool = False
    stream_batch: int | None = None
    stream_seed: int | None = None
    live_calls: bool = False
    methods: list[MethodSpec]

    @model_validator(mode="after")
    def _check(self) -> ExperimentCfg:
        if self.name in self.after:
            raise ValueError(f"experiment {self.name!r} waits for itself (after)")
        if self.stream_batch is not None:
            if self.stream_batch < 1:
                raise ValueError(f"experiment {self.name!r}: stream_batch {self.stream_batch} is no batch size (>= 1)")
            if self.workload != "cold" or self.interleave:
                raise ValueError(f"experiment {self.name!r}: a stream (stream_batch) is a cold workload's, not "
                                 f"{'an interleaved one' if self.interleave else self.workload}")
        elif self.stream_seed is not None:
            raise ValueError(f"experiment {self.name!r}: stream_seed orders a stream: it needs stream_batch")
        if self.site_profile is not None and not self.site_profile.strip():
            raise ValueError(f"experiment {self.name!r}: site_profile names no profile")
        if not self.lane.strip():
            raise ValueError(f"experiment {self.name!r}: lane names no lane")
        return self

    @model_serializer(mode="wrap")
    def _dump(self, handler: Any) -> dict[str, Any]:
        out = handler(self)
        if isinstance(out, dict):
            for name, default in LIVE_DEFAULTS.items():
                if getattr(self, name) == default:
                    out.pop(name, None)
        return out


LIVE_DEFAULTS: dict[str, Any] = {"site_profile": None, "models": None, "lane": "sophia", "stage": 0, "after": [],
                                 "before": [], "judge": True, "optional": False, "stream_batch": None,
                                 "stream_seed": None, "live_calls": False, "env": {}}


class FacilityCfg(_Cfg):
    endpoint: str
    config: dict[str, Any] = Field(default_factory=dict)
    placement: dict[str, int] = Field(default_factory=dict)

    @property
    def nodes(self) -> int:
        return int(self.config.get("nodes", 1))

    def slot_of(self, key: str | None) -> int:
        return int(self.placement.get(key, 0)) if key else 0


class RemoteCfg(_Cfg):
    facilities: dict[str, FacilityCfg]
    federations: list[str]
    models: str = "configs/models/final.yaml"
    action_timeout_s: float = 900.0
    env: dict[str, str] = Field(default_factory=dict)

    def facility_of(self, machine: str) -> str | None:
        if machine in self.facilities:
            return machine
        base, _, j = machine.rpartition("_")
        return base if j.isdigit() and base in self.facilities else None

    def keys_of(self, facility: str) -> list[str]:
        out: dict[str, None] = {}
        for name in self.federations:
            fed = load_federation(name)
            for site in fed.sites:
                if self.facility_of(site.machine) == facility:
                    out[f"{fed.name}/{site.name}"] = None
        return list(out)

    def keys_at(self, facility: str, slot: int) -> list[str]:
        fc = self.facilities[facility]
        return [k for k in self.keys_of(facility) if fc.slot_of(k) == slot]

    def check_placement(self) -> list[str]:
        problems = []
        for f, fc in self.facilities.items():
            keys = set(self.keys_of(f)) if fc.placement else set()
            for key, slot in fc.placement.items():
                if not 0 <= int(slot) < fc.nodes:
                    problems.append(f"{f}: {key} at slot {slot}, but the facility has {fc.nodes} nodes")
                if key not in keys:
                    problems.append(f"{f}: the placement names {key}, which no federation of the run places there")
        return problems


def configs_dir() -> Path:
    return Path(os.environ.get("FEDQUEST_CONFIGS") or CONFIGS)


def _resolve(name_or_path: str | Path, subdir: str) -> Path:
    p = Path(name_or_path)
    if p.suffix in {".yaml", ".yml"} and p.exists():
        return p
    candidate = configs_dir() / subdir / f"{name_or_path}.yaml"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"no config {name_or_path!r} (looked in {candidate})")


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_federation(name_or_path: str | Path) -> FederationCfg:
    cfg = FederationCfg(**_load_yaml(_resolve(name_or_path, "federations")))
    check_data_from(cfg)
    return cfg


def _collection_specs(site: SiteCfg) -> list[tuple[Any, ...]]:
    return [(c.name, c.modality, c.encoder, c.source, c.shard, c.vectors, c.keep, c.fraction, c.share)
            for c in site.collections]


def check_data_from(cfg: FederationCfg) -> None:
    for s in cfg.sites:
        if s.data_source is None:
            continue
        fed, src = s.data_source
        where = f"{cfg.name}/{s.name}: data_from {s.data_from!r}"
        try:
            ref = FederationCfg(**_load_yaml(_resolve(fed, "federations")))
        except FileNotFoundError:
            raise ValueError(f"{where} names no federation config ({configs_dir() / 'federations'})") from None
        if ref.data_from:
            raise ValueError(f"{where}: {fed} reads {ref.data_from}'s data itself; name {ref.data_from}/{src}")
        if ref.dataset != cfg.dataset:
            raise ValueError(f"{where}: {fed}'s dataset is {ref.dataset}, {cfg.name}'s {cfg.dataset}")
        try:
            source = ref.site(src)
        except KeyError:
            raise ValueError(f"{where}: {fed} has no site {src!r} (it has {ref.site_names})") from None
        if source.data_from:
            raise ValueError(f"{where}: {fed}/{src} reads {source.data_from}'s data itself; name that site")
        if _collection_specs(s) != _collection_specs(source):
            raise ValueError(f"{where}: its collections {_collection_specs(s)} are not {fed}/{src}'s "
                             f"{_collection_specs(source)} (name, modality, encoder, source, shard, vectors) and pages "
                             "kept (keep, fraction, share)")


def calibration_file(federation: str) -> Path:
    try:
        name = load_federation(federation).calibration
    except FileNotFoundError:
        name = None
    if not name:
        return calibration_path(federation)
    portable = configs_dir() / "calibration" / f"{name}.json"
    return portable if portable.exists() else calibration_path(name)


def load_experiment(name_or_path: str | Path) -> ExperimentCfg:
    return ExperimentCfg(**_load_yaml(_resolve(name_or_path, "experiments")))


def load_models(path: str | Path | None = None, profile: str | None = None) -> ModelsCfg:
    path = Path(path) if path else Path(os.environ.get("FEDQUEST_MODELS", CONFIGS / "models.yaml"))
    if not path.is_absolute() and not path.exists() and (CONFIGS.parent / path).exists():
        path = CONFIGS.parent / path
    cfg = ModelsCfg(**_load_yaml(path))
    provider = os.environ.get("FEDQUEST_LLM")
    if provider:
        cfg = cfg.model_copy(update={"provider": provider})
    base_url = os.environ.get("FEDQUEST_LLM_BASE_URL")
    if base_url:
        cfg = cfg.model_copy(update={"base_url": base_url})
    return with_profile(cfg, profile)


def with_profile(cfg: ModelsCfg, profile: str | None) -> ModelsCfg:
    if not profile:
        return cfg
    merged = cfg.profile(profile)
    offline = "replay" if cfg.provider == "replay" else os.environ.get("FEDQUEST_LLM")
    if offline in ("fake", "replay"):
        merged = merged.model_copy(update={"provider": offline})
    return merged


def models_sha(cfg: ModelsCfg) -> str:
    from fedquest.llm.cache import canonical_json

    return hashlib.sha256(canonical_json(cfg.fingerprint()).encode()).hexdigest()


def _expand_env(obj: Any) -> Any:
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    if isinstance(obj, dict):
        return {k: _expand_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env(v) for v in obj]
    return obj


def load_remote(path: str | Path | None = None) -> RemoteCfg:
    path = Path(path or os.environ.get("FEDQUEST_REMOTE_CONFIG") or CONFIGS / "remote.yaml")
    if not path.is_absolute() and not path.exists() and (CONFIGS.parent / path).exists():
        path = CONFIGS.parent / path
    return RemoteCfg(**_expand_env(_load_yaml(path)))
