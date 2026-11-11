# DGRAG offline build: KG embeddings, subgraph partitions and their summaries
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from collections import deque
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from fedquest.dgrag import (
    ENCODER,
    MIN_ENTITIES,
    SUBGRAPH_ENTITIES,
    SUMMARY_FACTS,
    SUMMARY_WORDS,
    corpus_dir,
    layout_dir,
)

log = logging.getLogger("fedquest.dgrag")

DATASETS = ("musique", "twowiki", "hotpotqa")
EMBED_BATCH = 256
SUMMARY_CONCURRENCY = 32
SUMMARY_TOKENS = 256


def _threads() -> None:
    try:
        import torch

        torch.set_num_threads(int(os.environ.get("FEDQUEST_BUILD_THREADS", "16")))
    except Exception:
        pass


def corpus_embeddings(dataset: str, encoder: str = ENCODER, force: bool = False) -> Path:
    from fedquest.dgrag.index import load_tables, tables_digest, tables_of
    from fedquest.fedgrag.build import corpus
    from fedquest.fedgrag.model import dataset_dir
    from fedquest.vdb.encoders import load_encoder

    if not (dataset_dir(dataset) / "tables.json").exists():
        corpus(dataset)
    names, facts, relations, _oids, sig = tables_of(dataset)
    d = corpus_dir(dataset)
    meta_path = d / "meta.json"
    want = {"sig": sig, "digest": tables_digest(names, facts, relations), "encoder": encoder,
            "entities": len(names), "facts": len(facts)}
    if not force and meta_path.exists() and json.loads(meta_path.read_text()) == want and \
            (d / "emb_nodes.npy").exists() and (d / "emb_facts.npy").exists():
        return d
    _threads()
    enc = load_encoder(encoder)
    t0 = time.time()
    nodes = enc.encode_docs(list(names), batch_size=EMBED_BATCH)
    texts = [f"{names[h]} {relations[i]} {names[t]}" for i, (h, t) in enumerate(facts.tolist())]
    fact_vecs = enc.encode_docs(texts, batch_size=EMBED_BATCH)
    log.info("%s: %d entities and %d relationships embedded with %s in %.0f s", dataset, len(names), len(texts),
             encoder, time.time() - t0)
    d.mkdir(parents=True, exist_ok=True)
    for name, arr in (("emb_nodes.npy", nodes), ("emb_facts.npy", fact_vecs)):
        tmp = d / f".{name}.{os.getpid()}.tmp.npy"
        np.save(tmp, np.asarray(arr, dtype=np.float16))
        tmp.replace(d / name)
    meta_path.write_text(json.dumps(want, indent=1))
    load_tables.cache_clear()
    return d


def _bfs_chunks(g: Any, size: int) -> list[list[int]]:
    seen: set[int] = set()
    order: list[int] = []
    for start in sorted(g.nodes):
        if start in seen:
            continue
        seen.add(start)
        queue = deque([start])
        while queue:
            n = queue.popleft()
            order.append(n)
            for m in sorted(g.neighbors(n)):
                if m not in seen:
                    seen.add(m)
                    queue.append(m)
    return [sorted(order[i:i + size]) for i in range(0, len(order), size)]


def partition(nodes: Sequence[int], pairs: Sequence[tuple[int, int]], seed: int = 0,
              size: int = SUBGRAPH_ENTITIES, small: int = MIN_ENTITIES) -> list[list[int]]:
    import networkx as nx

    g = nx.Graph()
    g.add_nodes_from(int(n) for n in nodes)
    for h, t in pairs:
        if h != t:
            w = g.get_edge_data(h, t, default={}).get("weight", 0)
            g.add_edge(int(h), int(t), weight=w + 1)
    if not g.number_of_nodes():
        return []
    todo = [sorted(c) for c in nx.community.leiden_communities(g, weight="weight", metric="modularity", seed=seed)]
    done: list[list[int]] = []
    while todo:
        c = todo.pop()
        if len(c) <= size:
            done.append(c)
            continue
        sub = g.subgraph(c)
        parts = [sorted(p) for p in nx.community.leiden_communities(sub, weight="weight", metric="modularity",
                                                                    seed=seed)]
        todo += parts if len(parts) > 1 else _bfs_chunks(sub, size)
    groups: dict[int, list[int]] = {i: c for i, c in enumerate(sorted(done))}
    where = {n: i for i, c in groups.items() for n in c}
    # merge each small subgraph into its most linked neighbor that still fits in size
    merged = True
    while merged:
        merged = False
        for i in sorted((i for i, c in groups.items() if len(c) < small), key=lambda i: (len(groups[i]), groups[i][0])):
            if i not in groups or len(groups[i]) >= small:
                continue
            links: dict[int, float] = {}
            for n in groups[i]:
                for m, data in g.adj[n].items():
                    j = where[m]
                    if j != i:
                        links[j] = links.get(j, 0.0) + float(data.get("weight", 1))
            fits = [j for j in links if len(groups[j]) + len(groups[i]) <= size]
            if not fits:
                continue
            j = max(fits, key=lambda j: (links[j], -j))
            groups[j] = sorted(groups[j] + groups.pop(i))
            for n in groups[j]:
                where[n] = j
            merged = True
    keep = [c for c in groups.values() if len(c) >= small]
    left = sorted((c for c in groups.values() if len(c) < small), key=lambda c: c[0])
    pack: list[int] = []
    for c in left:
        if len(pack) + len(c) > size:
            keep.append(sorted(pack))
            pack = []
        pack += c
    if pack:
        keep.append(sorted(pack))
    return sorted(keep, key=lambda c: c[0])


SUMMARY_SYSTEM = ("You summarize part of a knowledge graph that one data owner holds, so that a server can tell which "
                  "owners hold knowledge relevant to a question without seeing the data itself.")


def summary_messages(entities: Sequence[str], facts: Sequence[str]) -> list[dict[str, Any]]:
    user = ("Entities:\n" + "\n".join(f"- {e}" for e in entities) + "\n\nRelationships (head | relation | tail):\n"
            + "\n".join(facts) + "\n\n"
            f"Write a concise summary of this subgraph in at most {SUMMARY_WORDS} words: its subject areas, its main "
            "entities and its key relationships. Do not list every fact. Reply with the summary only.")
    return [{"role": "system", "content": SUMMARY_SYSTEM}, {"role": "user", "content": user}]


def subgraph_prompt(tables: Any, members: Sequence[int], incident: dict[int, np.ndarray]) -> list[dict[str, Any]]:
    inside = set(members)
    internal: list[int] = []
    boundary: list[int] = []
    for n in members:
        for f in incident.get(n, np.zeros(0, np.int64)).tolist():
            h, t = tables.facts[f].tolist()
            (internal if h in inside and t in inside else boundary).append(f)
    texts = list(dict.fromkeys(f"({tables.names[tables.facts[f][0]]} | {tables.relations[f]} | "
                               f"{tables.names[tables.facts[f][1]]})" for f in sorted(set(internal)) + sorted(set(boundary))))
    return summary_messages([tables.names[n] for n in members], texts[:SUMMARY_FACTS])


async def summarize(prompts: list[list[dict[str, Any]]], llm: Any = None) -> tuple[list[str], dict[str, Any]]:
    from fedquest.ledger import Ledger
    from fedquest.llm.client import LLMClient

    own = llm is None
    client = LLMClient.from_env() if own else llm
    led = Ledger()
    sem = asyncio.Semaphore(SUMMARY_CONCURRENCY)
    out: list[str] = [""] * len(prompts)
    t0 = time.time()

    async def one(i: int) -> None:
        async with sem:
            reply = await client.chat(prompts[i], role="reader", task="dgrag_summary", max_tokens=SUMMARY_TOKENS,
                                      ledger=led, charge="dgrag_summary")
            out[i] = " ".join(reply.text.split())

    try:
        await asyncio.gather(*(one(i) for i in range(len(prompts))))
    finally:
        if own:
            await client.aclose()
    u = led.coordinator.get("dgrag_summary")
    cost = {"calls": len(prompts), "prompt_tokens": getattr(u, "prompt_tokens", 0),
            "completion_tokens": getattr(u, "completion_tokens", 0),
            "cost_units": getattr(u, "cost_units", 0) if u is not None else 0,
            "model_seconds": round(getattr(u, "seconds", 0.0), 1), "wall_seconds": round(time.time() - t0, 1)}
    return out, cost


def build_layout(federation: str, mode: str = "remote", members: dict[str, list[str]] | None = None,
                 llm: Any = None, encoder: str = ENCODER) -> Path:
    from fedquest.config import load_federation
    from fedquest.dgrag.index import clear_caches, edges_from, load_tables
    from fedquest.fedgrag.build import membership
    from fedquest.vdb.encoders import load_encoder

    dataset = load_federation(federation).dataset
    if dataset not in DATASETS:
        raise ValueError(f"DGRAG runs on text knowledge graphs: {federation} is {dataset}")
    corpus_embeddings(dataset, encoder)
    tables = load_tables(dataset)
    if members is None:
        members = asyncio.run(membership(federation, mode))
    by_oid: dict[str, list[int]] = {}
    for f, oid in enumerate(tables.fact_oids):
        by_oid.setdefault(oid, []).append(f)
    sites = sorted(s for s, oids in members.items() if oids)
    fids = [np.asarray(sorted(f for o in members[s] for f in by_oid.get(o, [])), dtype=np.int64) for s in sites]
    nids = [np.asarray(sorted({int(e) for f in fs.tolist() for e in tables.facts[f].tolist()}), dtype=np.int64)
            for fs in fids]
    # an edge is one site's share of the graph: its facts and the entities they touch
    edges = edges_from(tables.facts, sites, fids, nids)
    subgraphs: list[dict[str, Any]] = []
    prompts: list[list[dict[str, Any]]] = []
    stats: dict[str, Any] = {}
    t0 = time.time()
    for ei, e in enumerate(edges):
        parts = partition(e.node_ids.tolist(), [tuple(tables.facts[f].tolist()) for f in e.fact_ids.tolist()])
        for p in parts:
            subgraphs.append({"edge": ei, "nodes": p})
            prompts.append(subgraph_prompt(tables, p, e.incident))
        stats[e.site] = {"objects": len(members[e.site]), "facts": int(len(e.fact_ids)),
                         "entities": int(len(e.node_ids)), "subgraphs": len(parts),
                         "mean_entities": round(float(np.mean([len(p) for p in parts])), 1) if parts else 0.0}
        log.info("%s/%s: %d entities, %d relationships, %d subgraphs", federation, e.site, len(e.node_ids),
                 len(e.fact_ids), len(parts))
    partition_s = time.time() - t0
    texts, cost = asyncio.run(summarize(prompts, llm))
    for s, text in zip(subgraphs, texts, strict=True):
        s["summary"] = text
    _threads()
    emb = load_encoder(encoder).encode_docs(texts, batch_size=64) if texts else np.zeros((0, 1), np.float32)
    out = layout_dir(federation)
    out.mkdir(parents=True, exist_ok=True)
    arrays = {f"facts_{i}": fs.astype(np.int32) for i, fs in enumerate(fids)}
    arrays |= {f"nodes_{i}": ns.astype(np.int32) for i, ns in enumerate(nids)}
    np.savez_compressed(out / "edges.npz", **arrays)
    np.save(out / "summaries.npy", np.asarray(emb, dtype=np.float16))
    (out / "subgraphs.json").write_text(json.dumps(subgraphs))
    summary_bytes = sum(len(t.encode()) for t in texts)
    cfg = {"federation": federation, "dataset": dataset, "clients": sites, "encoder": encoder, "stats": stats,
           "subgraphs": len(subgraphs), "settings": {"subgraph_entities": SUBGRAPH_ENTITIES,
                                                     "min_entities": MIN_ENTITIES, "summary_facts": SUMMARY_FACTS},
           "offline": {**cost, "partition_seconds": round(partition_s, 1), "summary_upload_bytes": summary_bytes}}
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    clear_caches()
    log.info("%s: %d subgraph summaries (%d calls, %s cost units)", federation, len(subgraphs), cost["calls"],
             cost["cost_units"])
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m fedquest.dgrag")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="the dataset's embeddings (once), then each layout's edges and summaries")
    b.add_argument("federations", nargs="+")
    b.add_argument("--mode", default="remote", choices=("remote", "local"))
    b.add_argument("--missing", action="store_true", help="build only the layouts with no index yet")
    s = sub.add_parser("status", help="what is built")
    s.add_argument("federations", nargs="*")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.cmd == "status":
        for f in args.federations or sorted(p.name for p in (layout_dir("x").parent).glob("*")):
            p = layout_dir(f) / "config.json"
            if p.exists():
                cfg = json.loads(p.read_text())
                print(f"{f}: {len(cfg['clients'])} edges, {cfg['subgraphs']} subgraphs, offline "
                      f"{cfg['offline'].get('calls')} calls, {cfg['offline'].get('cost_units')} cost units")
            else:
                print(f"{f}: NOT BUILT")
        return 0
    failed = 0
    for f in args.federations:
        if args.missing and (layout_dir(f) / "config.json").exists():
            log.info("DGRAG index of %s exists: kept (--missing)", f)
            continue
        try:
            build_layout(f, args.mode)
        except Exception as e:
            log.exception("DGRAG build of %s failed: %s", f, e)
            failed += 1
    return 1 if failed else 0
