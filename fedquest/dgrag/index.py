# DGRAG index: summaries pick owner sites; entity and fact retrieval per site
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from fedquest.dgrag import TOP_K, corpus_dir, layout_dir


@dataclass
class Tables:
    names: list[str]
    facts: np.ndarray
    relations: list[str]
    fact_oids: list[str]
    emb_nodes: np.ndarray
    emb_facts: np.ndarray

    def fact_text(self, f: int) -> str:
        h, t = self.facts[f]
        return f"{self.names[h]} {self.relations[f]} {self.names[t]}"


@dataclass
class Edge:
    site: str
    fact_ids: np.ndarray
    node_ids: np.ndarray
    incident: dict[int, np.ndarray]


@dataclass
class Knowledge:
    site: str
    entities: list[tuple[int, float]]
    facts: list[tuple[int, float]]
    chunks: list[tuple[str, float]]


class DGRAGIndex:
    def __init__(self, tables: Tables, edges: list[Edge], summaries: list[str], summary_edge: np.ndarray,
                 emb_summaries: np.ndarray, config: dict[str, Any]) -> None:
        self.tables, self.edges, self.summaries = tables, edges, summaries
        self.summary_edge, self.emb_summaries, self.config = summary_edge, emb_summaries, config
        self.sites = [e.site for e in edges]

    @classmethod
    def load(cls, federation: str) -> DGRAGIndex:
        return _load(federation)

    def match(self, q: np.ndarray, m: int, threshold: float | None) -> list[tuple[int, float]]:
        if not len(self.summaries):
            return []
        s = self.emb_summaries @ q
        order = np.argsort(-s, kind="stable")[:m]
        return [(int(i), float(s[i])) for i in order if threshold is None or s[i] >= threshold]

    def owners(self, matched: list[tuple[int, float]]) -> list[int]:
        return list(dict.fromkeys(int(self.summary_edge[i]) for i, _ in matched))

    # low-level keywords reach entities and then their facts; high-level ones reach facts directly
    def retrieve(self, ei: int, low: np.ndarray | None, high: np.ndarray | None, top_k: int = TOP_K,
                 threshold: float | None = None) -> Knowledge:
        e, t = self.edges[ei], self.tables
        ent: dict[int, float] = {}
        rel: dict[int, float] = {}
        if low is not None and len(e.node_ids):
            s = t.emb_nodes[e.node_ids] @ low
            for i in np.argsort(-s, kind="stable")[:top_k]:
                ent[int(e.node_ids[i])] = float(s[i])
            near: dict[int, float] = {}
            for n in list(ent):
                fids = e.incident.get(n)
                if fids is not None and len(fids):
                    for f, v in zip(fids.tolist(), (t.emb_facts[fids] @ low).tolist(), strict=True):
                        near[f] = max(near.get(f, -1.0), float(v))
            for f, v in sorted(near.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]:
                rel[f] = max(rel.get(f, -1.0), v)
        if high is not None and len(e.fact_ids):
            s = t.emb_facts[e.fact_ids] @ high
            for i in np.argsort(-s, kind="stable")[:top_k]:
                f, v = int(e.fact_ids[i]), float(s[i])
                rel[f] = max(rel.get(f, -1.0), v)
                for n in t.facts[f]:
                    ent[int(n)] = max(ent.get(int(n), -1.0), v)
        if threshold is not None:
            ent = {k: v for k, v in ent.items() if v >= threshold}
            rel = {k: v for k, v in rel.items() if v >= threshold}
        facts = sorted(rel.items(), key=lambda kv: (-kv[1], kv[0]))
        count: dict[str, int] = {}
        best: dict[str, float] = {}
        for f, v in facts:
            oid = t.fact_oids[f]
            count[oid] = count.get(oid, 0) + 1
            best[oid] = max(best.get(oid, -1.0), v)
        chunks = sorted(best, key=lambda o: (-count[o], -best[o], o))
        return Knowledge(e.site, sorted(ent.items(), key=lambda kv: (-kv[1], kv[0])), facts,
                         [(o, best[o]) for o in chunks])


def tables_of(dataset: str) -> tuple[list[str], np.ndarray, list[str], list[str], str]:
    from fedquest.fedgrag.model import dataset_dir

    d = dataset_dir(dataset)
    meta = json.loads((d / "tables.json").read_text())
    with np.load(d / "tables.npz") as z:
        facts = z["facts"].astype(np.int64).reshape(-1, 2)
    return meta["names"], facts, meta["relations"], meta["fact_oids"], str(meta.get("sig", ""))


def tables_digest(names: list[str], facts: np.ndarray, relations: list[str]) -> str:
    import hashlib

    h = hashlib.sha256(json.dumps([names, relations], ensure_ascii=False).encode())
    h.update(np.ascontiguousarray(facts, dtype=np.int64).tobytes())
    return h.hexdigest()[:16]


@lru_cache(maxsize=4)
def load_tables(dataset: str) -> Tables:
    names, facts, relations, fact_oids, _ = tables_of(dataset)
    d = corpus_dir(dataset)
    meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
    if meta.get("digest") != tables_digest(names, facts, relations):
        raise ValueError(f"DGRAG's vectors of {dataset} ({d}) were made for other tables: python -m fedquest.dgrag build "
                         "<federation> embeds them again")
    return Tables(names, facts, relations, fact_oids, np.load(d / "emb_nodes.npy").astype(np.float32),
                  np.load(d / "emb_facts.npy").astype(np.float32))


def edges_from(tables_facts: np.ndarray, sites: list[str], fact_ids: list[np.ndarray],
               node_ids: list[np.ndarray]) -> list[Edge]:
    edges = []
    for site, fids, nids in zip(sites, fact_ids, node_ids, strict=True):
        inc: dict[int, list[int]] = {}
        for f in fids.tolist():
            for n in tables_facts[f].tolist():
                inc.setdefault(int(n), []).append(int(f))
        edges.append(Edge(site, fids, nids, {n: np.asarray(v, dtype=np.int64) for n, v in inc.items()}))
    return edges


@lru_cache(maxsize=8)
def _load(federation: str) -> DGRAGIndex:
    d = layout_dir(federation)
    if not (d / "config.json").exists():
        raise FileNotFoundError(f"no DGRAG index of {federation} ({d}): python -m fedquest.dgrag build {federation}")
    cfg = json.loads((d / "config.json").read_text())
    tables = load_tables(cfg["dataset"])
    with np.load(d / "edges.npz") as z:
        fids = [z[f"facts_{i}"].astype(np.int64) for i in range(len(cfg["clients"]))]
        nids = [z[f"nodes_{i}"].astype(np.int64) for i in range(len(cfg["clients"]))]
    edges = edges_from(tables.facts, cfg["clients"], fids, nids)
    subs = json.loads((d / "subgraphs.json").read_text())
    emb = np.load(d / "summaries.npy").astype(np.float32)
    return DGRAGIndex(tables, edges, [s["summary"] for s in subs], np.asarray([s["edge"] for s in subs],
                                                                            dtype=np.int64), emb, cfg)


def clear_caches() -> None:
    _load.cache_clear()
    load_tables.cache_clear()
