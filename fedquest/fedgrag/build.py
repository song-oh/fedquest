# FedGRAG build: OpenIE fact tables, site prototypes, trained routers, tuning
from __future__ import annotations

import argparse
import asyncio
import hashlib
import itertools
import json
import logging
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from fedquest.fedgrag.model import (
    DEFAULT,
    ENCODERS,
    GAMMA,
    K_PROTOS,
    L_NEIGHBORS,
    LATENT_DIM,
    Corpus,
    Encoder,
    FedGRAGIndex,
    Settings,
    dataset_dir,
    doc_of,
    layout_dir,
    norm_name,
    variant_dir,
)
from fedquest.paths import runs_root

log = logging.getLogger("fedquest.fedgrag")

DATASETS = ("musique", "twowiki", "hotpotqa")
EPOCHS = 10
LR = 1e-3
TRAIN_SPLITS_EXCLUDED = ("final_test",)


OPENIE_HOMES = {"musique": "musique_next_home", "twowiki": "twowiki_home", "hotpotqa": "hotpot_home"}


def openie_dir(dataset: str) -> Path:
    from fedquest.config import load_federation

    home = OPENIE_HOMES.get(dataset)
    if home is not None:
        try:
            cfg = load_federation(home)
            site = cfg.sites[0]
            d = cfg.collection_data_dir(site.name, site.collections[0].name) / "hippo"
        except (FileNotFoundError, IndexError):
            d = None
        if d is not None and ((d / "openie.jsonl").exists() or any(d.glob("openie.part*.jsonl"))):
            return d
    return runs_root() / "inputs" / "openie" / dataset


def _openie_files(dataset: str) -> list[Path]:
    d = openie_dir(dataset)
    base = [d / "openie.jsonl"] if (d / "openie.jsonl").exists() else []
    return base + sorted(p for p in d.glob("openie.part*.jsonl"))


def load_openie(dataset: str) -> dict[str, dict[str, Any]]:
    files = _openie_files(dataset)
    if not files:
        raise FileNotFoundError(f"no OpenIE records under {openie_dir(dataset)} (fedquest hippo openie "
                                f"{OPENIE_HOMES.get(dataset, '<home>')}, tool:hippo_<bench>)")
    out: dict[str, dict[str, Any]] = {}
    for p in files:
        with p.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if r.get("oid"):
                        out[r["oid"]] = r
    return out


def corpus(dataset: str, force: bool = False) -> Corpus:
    files = _openie_files(dataset)
    sig = hashlib.sha256(json.dumps([[p.name, p.stat().st_size, int(p.stat().st_mtime)] for p in files])
                         .encode()).hexdigest()[:16]
    d = dataset_dir(dataset)
    if not force and (d / "tables.json").exists() and json.loads((d / "tables.json").read_text()).get("sig") == sig:
        return Corpus.load(dataset)
    records = load_openie(dataset)
    name_id: dict[str, int] = {}
    names: list[str] = []
    facts: list[tuple[int, int]] = []
    relations: list[str] = []
    fact_oids: list[str] = []
    seen: set[tuple[int, str, int, str]] = set()

    def node(s: str) -> int:
        k = norm_name(s)
        if k not in name_id:
            name_id[k] = len(names)
            names.append(" ".join(str(s).split()))
        return name_id[k]

    for oid in sorted(records):
        for tr in records[oid].get("triples") or []:
            if not isinstance(tr, list | tuple) or len(tr) != 3 or not all(str(x).strip() for x in tr):
                continue
            h, r, t = (str(x) for x in tr)
            hi, ti = node(h), node(t)
            key = (hi, norm_name(r), ti, oid)
            if key in seen:
                continue
            seen.add(key)
            facts.append((hi, ti))
            relations.append(" ".join(r.split()))
            fact_oids.append(oid)
    log.info("%s: %d objects with records, %d entities, %d facts", dataset, len(records), len(names), len(facts))
    t0 = time.time()
    emb_nodes = Encoder.encode(names)
    fact_texts = [f"{names[h]} {relations[i]} {names[t]}" for i, (h, t) in enumerate(facts)]
    emb_facts = Encoder.encode(fact_texts)
    log.info("%s: embedded in %.0f s", dataset, time.time() - t0)
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "emb_nodes.npy", emb_nodes.astype(np.float16))
    np.save(d / "emb_facts.npy", emb_facts.astype(np.float16))
    np.savez_compressed(d / "tables.npz", facts=np.asarray(facts, dtype=np.int32).reshape(-1, 2))
    (d / "tables.json").write_text(json.dumps({"sig": sig, "names": names, "relations": relations,
                                               "fact_oids": fact_oids, "records": len(records)}))
    from fedquest.fedgrag.model import _load_corpus

    _load_corpus.cache_clear()
    return Corpus.load(dataset)


async def membership(federation: str, mode: str) -> dict[str, list[str]]:
    if mode == "local":
        from fedquest.federation.local import open_local_federation

        fed = await asyncio.to_thread(open_local_federation, federation)
    else:
        from fedquest.federation.gc import open_gc_federation

        fed = open_gc_federation(federation)
    try:
        info = await fed.info()
        cols = sorted({c.name for si in info.values() for c in si.collections if c.modality == "text"})
        cat = await fed.catalog(collections=cols)
    finally:
        await fed.close()
    return {s: sorted({o for c in by.values() for o, *_ in c}) for s, by in cat.items()}


def _structure(cor: Corpus, fids: np.ndarray, nids: np.ndarray) -> np.ndarray:
    a_hat = _norm_adj(cor, fids, nids)
    x = cor.emb_nodes[nids]
    for _ in range(3):
        x = a_hat @ x
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)


def _norm_adj(cor: Corpus, fids: np.ndarray, nids: np.ndarray) -> Any:
    from scipy import sparse

    pos = {int(n): i for i, n in enumerate(nids)}
    rows, cols = [], []
    for f in fids:
        h, t = cor.facts[int(f)]
        a, b = pos[int(h)], pos[int(t)]
        if a != b:
            rows += [a, b]
            cols += [b, a]
    n = len(nids)
    a = sparse.coo_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n, n)).tocsr()
    a.data[:] = 1.0
    a = a + sparse.identity(n, dtype=np.float32, format="csr")
    deg = np.asarray(a.sum(axis=1)).ravel()
    dinv = sparse.diags(1.0 / np.sqrt(np.maximum(deg, 1e-9)))
    return dinv @ a @ dinv


def _prototypes(fused: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    from sklearn.cluster import MiniBatchKMeans

    if len(fused) <= k:
        p = fused.copy()
    else:
        km = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=3, batch_size=2048).fit(fused)
        p = km.cluster_centers_.astype(np.float32)
    return p / np.maximum(np.linalg.norm(p, axis=1, keepdims=True), 1e-9)


def _proto_graph(protos: np.ndarray, proto_client: np.ndarray) -> np.ndarray:
    sim = protos @ protos.T
    np.fill_diagonal(sim, -np.inf)
    m = len(protos)
    adj = np.zeros((m, m), dtype=np.float32)
    kk = min(L_NEIGHBORS, m - 1)
    if kk > 0:
        nbr = np.argsort(-sim, axis=1)[:, :kk]
        for i in range(m):
            for j in nbr[i]:
                w = max(0.0, float(sim[i, j]))
                adj[i, j] = max(adj[i, j], w)
                adj[j, i] = max(adj[j, i], w)
    deg = adj.sum(axis=1)
    dinv = 1.0 / np.sqrt(np.maximum(deg, 1e-9))
    return (adj * dinv[:, None]) * dinv[None, :]


FED_HOMES = {"musique": "musique_fed", "twowiki": "twowiki_fed", "hotpotqa": "hotpot_fed"}
EVAL_SPLITS = {"musique": ("test",), "twowiki": ("full",), "hotpotqa": ("full",)}


def training_questions(federation: str, dataset: str) -> list[dict[str, Any]]:
    from fedquest.data.splits import HOMES, parent_of, split_path
    from fedquest.paths import fed_dir

    dirs = {fed_dir(FED_HOMES[dataset]), *(fed_dir(h) for h in HOMES.get(dataset, {}).values())}
    parent = parent_of(federation)
    if parent:
        dirs.add(fed_dir(parent))
    excluded: set[str] = set()
    for name in EVAL_SPLITS.get(dataset, ("final_test",)):
        ft = split_path(federation, name)
        if ft.exists():
            excluded |= {json.loads(line)["qid"] for line in ft.open(encoding="utf-8") if line.strip()}
    pool: dict[str, dict[str, Any]] = {}
    for d in sorted(dirs):
        for p in sorted((d / "questions").glob("*.jsonl")) if (d / "questions").exists() else []:
            if p.stem in EVAL_SPLITS.get(dataset, TRAIN_SPLITS_EXCLUDED):
                continue
            for line in p.open(encoding="utf-8"):
                if line.strip():
                    r = json.loads(line)
                    pool.setdefault(r["qid"], r)
    return [r for q, r in sorted(pool.items()) if q not in excluded and r.get("gold_doc_ids")]


def train_router(index: FedGRAGIndex, queries: np.ndarray, positives: list[set[int]], seed: int = 0
                 ) -> tuple[np.ndarray, float, dict[str, Any]]:
    import torch

    torch.manual_seed(seed)
    d = index.protos.shape[1]
    protos = torch.tensor(index.protos)
    adj = torch.tensor(index.adj)
    pc = torch.tensor(index.proto_client)
    m = len(index.clients)
    w = torch.nn.Parameter(torch.eye(d))
    a_logit = torch.nn.Parameter(torch.zeros(()))
    opt = torch.optim.Adam([w, a_logit], lr=LR)
    q = torch.tensor(np.stack([index.fuse_query(v) for v in queries])) if len(queries) else torch.zeros((0, d))
    # only questions whose gold sits on some sites but not all give the router a ranking to learn
    usable = [i for i, p in enumerate(positives) if 0 < len(p) < m]
    history = []
    for _epoch in range(EPOCHS):
        perm = torch.randperm(len(usable)).tolist()
        total, pairs = 0.0, 0
        for start in range(0, len(perm), 32):
            idx = [usable[perm[i]] for i in range(start, min(start + 32, len(perm)))]
            qq = torch.nn.functional.normalize(q[idx] @ w.T, dim=1)
            r0 = qq @ protos.T
            top = torch.topk(r0, k=min(L_NEIGHBORS, r0.shape[1]), dim=1).indices
            mask = torch.zeros_like(r0).scatter_(1, top, 1.0)
            start_r = r0 * mask
            alpha = torch.sigmoid(a_logit)
            r = start_r
            for _ in range(2):
                r = alpha * start_r + (1 - alpha) * (r @ adj.T)
            scores = torch.full((len(idx), m), -1e9).scatter_reduce(1, pc.expand(len(idx), -1), r, reduce="amax")
            loss = torch.zeros(())
            n = 0
            for bi, qi in enumerate(idx):
                pos = sorted(positives[qi])
                neg = [j for j in range(m) if j not in positives[qi]]
                sp, sn = scores[bi, pos], scores[bi, neg]
                loss = loss + torch.relu(GAMMA - sp[:, None] + sn[None, :]).sum()
                n += len(pos) * len(neg)
            if n == 0:
                continue
            loss = loss / n
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss) * n
            pairs += n
        history.append(round(total / max(1, pairs), 5))
    return w.detach().numpy(), float(torch.sigmoid(a_logit).detach()), {"loss": history, "questions": len(usable)}


def gold_recall(index: FedGRAGIndex, queries: np.ndarray, golds: list[set[str]], b: int, s: Settings = DEFAULT
                ) -> float:
    vals = []
    for v, g in zip(queries, golds, strict=True):
        if not g:
            continue
        _, _, facts = index.retrieve(v, b, s)
        got = {doc_of(index.corpus.fact_oids[f]) for f, _, _ in facts}
        vals.append(len(g & got) / len(g))
    return float(np.mean(vals)) if vals else 0.0


def _xfit_questions(federation: str, split: str, half: str) -> list[dict[str, Any]]:
    from fedquest import xfit
    from fedquest.eval.runner import load_questions

    return [q.to_dict() for q in load_questions(federation, split) if xfit.in_half(q.qid, half) and q.gold_doc_ids]


def _train(index: FedGRAGIndex, qs: list[dict[str, Any]], sites: list[str], members: dict[str, list[str]]
           ) -> tuple[np.ndarray, float, dict[str, Any]]:
    site_of = {o: i for i, s in enumerate(sites) for o in members[s]}
    doc_site: dict[str, set[int]] = {}
    for o, i in site_of.items():
        doc_site.setdefault(doc_of(o), set()).add(i)
    qs = [q for q in qs if any(g in doc_site for g in q["gold_doc_ids"])]
    qvec = Encoder.encode([q["question"] for q in qs])
    pos = [set().union(*(doc_site.get(g, set()) for g in q["gold_doc_ids"])) for q in qs]
    golds = [{g for g in q["gold_doc_ids"] if g in doc_site} for q in qs]
    w, alpha, hist = train_router(index, qvec, pos)
    trained = FedGRAGIndex(index.corpus, index.clients, index.protos, index.proto_client, index.adj,
                           w.astype(np.float32), alpha, index.b, index.config, qmap=index.qmap)
    tune = list(range(len(qs)))[:: max(1, len(qs) // 300)]
    recalls = {b: gold_recall(trained, qvec[tune], [golds[i] for i in tune], b) for b in range(1, len(sites) + 1)}
    best = max(recalls.values()) if recalls else 0.0
    # B: the fewest routed sites whose gold recall is within 0.01 of the best
    b = min((b for b, r in recalls.items() if r >= best - 0.01), default=1)
    hits = [float(bool(set(trained.route(qvec[i], b)) & pos[i])) for i in tune]
    rec = {"b": int(b), "recall_by_b": {str(k): round(v, 4) for k, v in recalls.items()},
           "router": {**hist, "alpha": round(alpha, 4)}, "train_questions": len(qs),
           "hit_at_b": round(float(np.mean(hits)), 4) if hits else None}
    return w, alpha, rec


def build_layout(federation: str, mode: str = "remote", members: dict[str, list[str]] | None = None,
                 halves: bool = False, split: str = "test") -> Path:
    from fedquest import xfit
    from fedquest.config import load_federation
    from fedquest.fedgrag.model import _load_clients, clear_caches, router_file

    fcfg = load_federation(federation)
    dataset = fcfg.dataset
    if dataset not in DATASETS:
        raise ValueError(f"FedGRAG runs on text knowledge graphs: {federation} is {dataset}")
    cor = corpus(dataset)
    out = layout_dir(federation)
    out.mkdir(parents=True, exist_ok=True)
    if members is None:
        members = asyncio.run(membership(federation, mode))
    (out / "membership.json").write_text(json.dumps({s: len(v) for s, v in members.items()}, indent=1))
    by_oid: dict[str, list[int]] = {}
    for f, oid in enumerate(cor.fact_oids):
        by_oid.setdefault(oid, []).append(f)
    sites = sorted(s for s, oids in members.items() if oids)
    arrays: dict[str, np.ndarray] = {}
    protos, owner = [], []
    stats = {}
    for i, site in enumerate(sites):
        fids = np.asarray(sorted(f for o in members[site] for f in by_oid.get(o, [])), dtype=np.int64)
        nids = np.asarray(sorted({int(e) for f in fids for e in cor.facts[int(f)]}), dtype=np.int64)
        struct = _structure(cor, fids, nids) if len(nids) else np.zeros((0, cor.emb_nodes.shape[1]), np.float32)
        # a node is its name vector beside its structure vector, scaled back to unit length
        fused = np.concatenate([cor.emb_nodes[nids], struct], axis=1) / np.sqrt(2.0) if len(nids) else \
            np.zeros((0, 2 * cor.emb_nodes.shape[1]), np.float32)
        p = _prototypes(fused, K_PROTOS) if len(fused) else np.zeros((0, fused.shape[1]), np.float32)
        protos.append(p)
        owner += [i] * len(p)
        arrays[f"facts_{i}"], arrays[f"nodes_{i}"] = fids.astype(np.int32), nids.astype(np.int32)
        arrays[f"struct_{i}"] = struct.astype(np.float16)
        stats[site] = {"objects": len(members[site]), "facts": int(len(fids)), "entities": int(len(nids)),
                       "prototypes": int(len(p))}
    np.savez_compressed(out / "clients.npz", **arrays)
    allp = np.concatenate(protos, axis=0).astype(np.float32)
    proto_client = np.asarray(owner, dtype=np.int32)
    adj = _proto_graph(allp, proto_client)
    d = allp.shape[1]
    cfg: dict[str, Any] = {"federation": federation, "dataset": dataset, "clients": sites,
                           "b": max(1, len(sites) // 2), "k_protos": K_PROTOS, "l_neighbors": L_NEIGHBORS,
                           "stats": stats, "prototype_upload_bytes": int(allp.size * 4)}
    for old in [router_file(federation), *(router_file(federation, f"x{h}") for h in xfit.HALVES)]:
        old.unlink(missing_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    clear_caches()
    corpus_, clients = _load_clients(federation)
    untrained = FedGRAGIndex(corpus_, clients, allp, proto_client.astype(np.int64), adj, np.eye(d, dtype=np.float32),
                             0.5, cfg["b"], cfg)
    if halves:
        cfg["routers"] = {}
        for h in xfit.HALVES:
            w, alpha, rec = _train(untrained, _xfit_questions(federation, split, h), sites, members)
            np.savez(router_file(federation, f"x{h}"), protos=allp, proto_client=proto_client, adj=adj,
                     w=w.astype(np.float32), alpha=np.float32(alpha))
            cfg["routers"][f"x{h}"] = {**rec, "trained_on": f"{split}:{xfit.split_of(h)}"}
            log.info("%s: router x%s on %d questions of %s: B=%d, recall %.3f, hit@B %s", federation, h,
                     rec["train_questions"], xfit.split_of(h), rec["b"], rec["recall_by_b"].get(str(rec["b"]), 0),
                     rec["hit_at_b"])
        cfg["b"] = None
    else:
        w, alpha, rec = _train(untrained, training_questions(federation, dataset), sites, members)
        np.savez(router_file(federation), protos=allp, proto_client=proto_client, adj=adj, w=w.astype(np.float32),
                 alpha=np.float32(alpha))
        cfg.update(rec)
        log.info("%s: %d clients, B=%d, recall %.3f, router loss %s", federation, len(sites), rec["b"],
                 rec["recall_by_b"].get(str(rec["b"]), 0), rec["router"]["loss"][-1:] or None)
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    clear_caches()
    return out


def retrain_router(federation: str) -> Path:
    from fedquest.config import load_federation
    from fedquest.fedgrag.model import _load_clients, clear_caches, router_file

    dataset = load_federation(federation).dataset
    out = layout_dir(federation)
    cfg = json.loads((out / "config.json").read_text())
    clear_caches()
    corpus_, clients = _load_clients(federation)
    with np.load(router_file(federation)) as rf:
        protos, proto_client, adj = rf["protos"], rf["proto_client"], rf["adj"]
    sites = [c.site for c in clients]
    members = {c.site: sorted({corpus_.fact_oids[int(f)] for f in c.fact_ids}) for c in clients}
    untrained = FedGRAGIndex(corpus_, clients, protos, proto_client.astype(np.int64), adj,
                             np.eye(protos.shape[1], dtype=np.float32), 0.5, cfg.get("b") or 1, cfg)
    w, alpha, rec = _train(untrained, training_questions(federation, dataset), sites, members)
    np.savez(router_file(federation), protos=protos, proto_client=proto_client, adj=adj, w=w.astype(np.float32),
             alpha=np.float32(alpha))
    cfg.update(rec)
    cfg["trained_outside"] = list(EVAL_SPLITS.get(dataset, ("final_test",)))
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    clear_caches()
    log.info("%s: router retrained on %d questions outside %s: B=%d, recall %.3f, hit@B %s", federation,
             rec["train_questions"], cfg["trained_outside"], rec["b"], rec["recall_by_b"].get(str(rec["b"]), 0),
             rec["hit_at_b"])
    return out


TUNE_GRID: dict[str, tuple[int, ...]] = {"n_seeds": (5, 10, 20), "hops": (1, 2, 3), "n_edges": (30, 60),
                                         "start_facts": (5, 10)}
TUNE_QUESTIONS = 300


def _router_keys(federation: str) -> list[str | None]:
    cfg = json.loads((layout_dir(federation) / "config.json").read_text())
    return sorted(cfg["routers"]) if cfg.get("routers") else [None]


def _router_questions(federation: str, key: str | None) -> list[dict[str, Any]]:
    from fedquest.config import load_federation

    if key is None:
        return training_questions(federation, load_federation(federation).dataset)
    cfg = json.loads((layout_dir(federation) / "config.json").read_text())
    split = str(cfg["routers"][key]["trained_on"]).split(":", 1)[0]
    return _xfit_questions(federation, split, key[1:])


def _tuning_set(index: FedGRAGIndex, qs: list[dict[str, Any]]) -> tuple[np.ndarray, list[set[str]]]:
    held = {doc_of(index.corpus.fact_oids[int(f)]) for c in index.clients for f in c.fact_ids}
    qs = [q for q in qs if any(g in held for g in q["gold_doc_ids"])]
    qs = qs[:: max(1, len(qs) // TUNE_QUESTIONS)]
    golds = [{g for g in q["gold_doc_ids"] if g in held} for q in qs]
    return Encoder.encode([q["question"] for q in qs]), golds


def recall_by_b(index: FedGRAGIndex, queries: np.ndarray, golds: list[set[str]], s: Settings = DEFAULT
                ) -> dict[int, float]:
    m = len(index.clients)
    sums = np.zeros(m + 1)
    n = 0
    for v, g in zip(queries, golds, strict=True):
        if not g:
            continue
        per: dict[int, list[tuple[int, float]]] = {}
        for b, ci in enumerate(index.route(v, m), start=1):
            per[ci] = index.client_facts(ci, v, s)
            got = {doc_of(index.corpus.fact_oids[f]) for f, _, _ in index.compose(per, s)}
            sums[b] += len(g & got) / len(g)
        n += 1
    return {b: float(sums[b] / max(1, n)) for b in range(1, m + 1)}


def pick_b(recalls: dict[int, float]) -> int:
    best = max(recalls.values()) if recalls else 0.0
    return min((b for b, r in recalls.items() if r >= best - 0.01), default=1)


def tune_layout(federation: str, encoder: str | None = None, source: str | None = None,
                questions: dict[str | None, list[dict[str, Any]]] | None = None) -> dict[str, Any]:
    from fedquest.fedgrag.model import clear_caches

    out: dict[str, Any] = {}
    for key in _router_keys(federation):
        index = FedGRAGIndex.load(federation, key, encoder)
        qvec, golds = _tuning_set(index, questions[key] if questions is not None else
                                  _router_questions(federation, key))
        default = recall_by_b(index, qvec, golds, DEFAULT)
        b0 = pick_b(default)
        rec: dict[str, Any] = {"questions": len(golds), "encoder": encoder,
                               "default": {"settings": DEFAULT.as_dict(), "b": b0, "recall": round(default[b0], 4)}}
        if source is not None:
            src = (json.loads((variant_dir(source, key, encoder) / "config.json").read_text()).get("tuned") or {}
                   ).get(key or "all")
            if src is None:
                raise FileNotFoundError(f"{source} has no tuned settings for router {key or 'all'}: python -m "
                                        f"fedquest.fedgrag tune {source}" + (f" --encoder {encoder}" if encoder else ""))
            s = Settings.of(src["settings"])
            recalls = default if s == DEFAULT else recall_by_b(index, qvec, golds, s)
            rec["from"] = source
        else:
            grid = []
            for combo in itertools.product(*TUNE_GRID.values()):
                s_ = replace(DEFAULT, **dict(zip(TUNE_GRID, combo, strict=True)))
                r = default if s_ == DEFAULT else recall_by_b(index, qvec, golds, s_)
                grid.append((r[pick_b(r)], s_, pick_b(r), r))
            grid.sort(key=lambda t: (-round(t[0], 4), t[1].n_edges, t[1].n_seeds, t[1].hops, t[1].start_facts))
            _, s, _, recalls = grid[0]
            rec["grid"] = [{**g[1].as_dict(), "b": g[2], "recall": round(g[0], 4)} for g in grid]
        b = pick_b(recalls)
        rec.update({"settings": s.as_dict(), "b": int(b), "recall": round(recalls[b], 4),
                    "recall_by_b": {str(k): round(v, 4) for k, v in recalls.items()}})
        d = variant_dir(federation, key, encoder)
        cfg = json.loads((d / "config.json").read_text())
        cfg.setdefault("tuned", {})[key or "all"] = rec
        (d / "config.json").write_text(json.dumps(cfg, indent=1))
        clear_caches()
        out[key or "all"] = rec
        log.info("%s%s router %s: tuned %s, B=%d, recall %.3f (the main runs' %.3f at B=%d) on %d questions",
                 federation, f" ({encoder})" if encoder else "", key or "all",
                 {k: v for k, v in s.as_dict().items() if k in TUNE_GRID}, b, recalls[b], default[b0], b0, len(golds))
    return out


GNN_LAYERS = 3
GNN_HIDDEN = 256
TAU = 0.1
WEIGHT_DECAY = 5e-4
BATCH = 32


def _encoder_modules(d_in: int) -> Any:
    import torch

    dims = [d_in] + [GNN_HIDDEN] * GNN_LAYERS
    return torch.nn.ModuleDict({
        "gnn": torch.nn.ModuleList(torch.nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:], strict=True)),
        "fuse": torch.nn.Sequential(torch.nn.Linear(d_in + GNN_HIDDEN, GNN_HIDDEN), torch.nn.ReLU(),
                                    torch.nn.Linear(GNN_HIDDEN, LATENT_DIM)),
        "query": torch.nn.Linear(d_in, LATENT_DIM, bias=False)})


def _encode_nodes(m: Any, a_hat: Any, x: Any) -> Any:
    import torch

    h = x
    for lin in m["gnn"]:
        h = torch.relu(torch.sparse.mm(a_hat, lin(h)))
    return torch.nn.functional.normalize(m["fuse"](torch.cat([x, h], dim=1)), dim=1)


def _union(cor: Corpus, clients: list[Any]) -> tuple[Any, Any, np.ndarray]:
    import torch
    from scipy import sparse

    offsets = np.cumsum([0] + [len(c.node_ids) for c in clients])
    # one block per site: the union graph never links entities across sites
    blocks = [_norm_adj(cor, c.fact_ids, c.node_ids) for c in clients if len(c.node_ids)]
    a = sparse.block_diag(blocks, format="coo") if blocks else sparse.coo_matrix((0, 0), dtype=np.float32)
    a_t = torch.sparse_coo_tensor(torch.tensor(np.vstack([a.row, a.col]), dtype=torch.long),
                                  torch.tensor(a.data, dtype=torch.float32), a.shape,
                                  check_invariants=True).coalesce()
    x = torch.tensor(np.concatenate([cor.emb_nodes[c.node_ids] for c in clients]).astype(np.float32))
    return a_t, x, offsets


def _evidence_nodes(cor: Corpus, clients: list[Any], offsets: np.ndarray, qs: list[dict[str, Any]]
                    ) -> list[np.ndarray]:
    by_doc: dict[str, set[int]] = {}
    for i, c in enumerate(clients):
        pos = {int(n): j for j, n in enumerate(c.node_ids)}
        for f in c.fact_ids:
            nodes = by_doc.setdefault(doc_of(cor.fact_oids[int(f)]), set())
            nodes.update(int(offsets[i]) + pos[int(e)] for e in cor.facts[int(f)])
    return [np.asarray(sorted(set().union(*(by_doc.get(g, set()) for g in q["gold_doc_ids"]))), dtype=np.int64)
            for q in qs]


def train_encoder(cor: Corpus, clients: list[Any], qs: list[dict[str, Any]], seed: int = 0
                  ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    import torch

    torch.manual_seed(seed)
    a_t, x, offsets = _union(cor, clients)
    ev = _evidence_nodes(cor, clients, offsets, qs)
    keep = [i for i, e in enumerate(ev) if len(e)]
    qv = torch.tensor(Encoder.encode([qs[i]["question"] for i in keep]))
    pos = [torch.tensor(ev[i]) for i in keep]
    m = _encoder_modules(int(x.shape[1]))
    opt = torch.optim.Adam(m.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    history = []
    for _epoch in range(EPOCHS):
        perm = torch.randperm(len(keep)).tolist()
        total = 0.0
        for start in range(0, len(perm), BATCH):
            idx = perm[start:start + BATCH]
            z = _encode_nodes(m, a_t, x)
            logits = torch.nn.functional.normalize(m["query"](qv[idx]), dim=1) @ z.T / TAU
            loss = torch.stack([torch.logsumexp(logits[k], 0) - torch.logsumexp(logits[k, pos[j]], 0)
                                for k, j in enumerate(idx)]).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(idx)
        history.append(round(total / max(1, len(keep)), 5))
    return {k: v.detach().numpy() for k, v in m.state_dict().items()}, {"loss": history, "questions": len(keep)}


def apply_encoder(state: dict[str, np.ndarray], cor: Corpus, clients: list[Any]) -> tuple[list[np.ndarray],
                                                                                        np.ndarray]:
    import torch

    a_t, x, offsets = _union(cor, clients)
    m = _encoder_modules(int(x.shape[1]))
    m.load_state_dict({k: torch.tensor(v) for k, v in state.items()})
    with torch.no_grad():
        z = _encode_nodes(m, a_t, x).numpy().astype(np.float32)
    return [z[offsets[i]:offsets[i + 1]] for i in range(len(clients))], state["query.weight"].astype(np.float32)


def build_trained(federation: str, source: str | None = None,
                  questions: dict[str | None, list[dict[str, Any]]] | None = None) -> list[Path]:
    from fedquest.fedgrag.model import _load_clients, _load_variant_clients, clear_caches

    base = json.loads((layout_dir(federation) / "config.json").read_text())
    clear_caches()
    cor, clients = _load_clients(federation)
    sites = [c.site for c in clients]
    members = {c.site: sorted({cor.fact_oids[int(f)] for f in c.fact_ids}) for c in clients}
    outs = []
    for key in _router_keys(federation):
        qs = questions[key] if questions is not None else _router_questions(federation, key)
        if source is None:
            state, record = train_encoder(cor, clients, qs)
        else:
            with np.load(variant_dir(source, key, "trained") / "encoder.npz") as z:
                state = {k: z[k] for k in z.files}
            record = {"from": source}
        fused, qmap = apply_encoder(state, cor, clients)
        protos, owner = [], []
        for i, z in enumerate(fused):
            p = _prototypes(z, K_PROTOS) if len(z) else np.zeros((0, LATENT_DIM), np.float32)
            protos.append(p)
            owner += [i] * len(p)
        allp = np.concatenate(protos, axis=0).astype(np.float32)
        proto_client = np.asarray(owner, dtype=np.int32)
        adj = _proto_graph(allp, proto_client)
        d = variant_dir(federation, key, "trained")
        d.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {}
        for i, c in enumerate(clients):
            arrays[f"facts_{i}"], arrays[f"nodes_{i}"] = c.fact_ids.astype(np.int32), c.node_ids.astype(np.int32)
            arrays[f"struct_{i}"] = fused[i].astype(np.float16)
        np.savez_compressed(d / "clients.npz", **arrays)
        np.savez(d / "encoder.npz", **state)
        (d / "router.npz").unlink(missing_ok=True)
        cfg: dict[str, Any] = {"federation": federation, "dataset": base["dataset"], "clients": sites,
                               "encoder": "trained", "router": key, "b": 1, "k_protos": K_PROTOS,
                               "l_neighbors": L_NEIGHBORS, "encoder_training": record,
                               "prototype_upload_bytes": int(allp.size * 4)}
        (d / "config.json").write_text(json.dumps(cfg, indent=1))
        clear_caches()
        vcor, vclients = _load_variant_clients(str(d))
        untrained = FedGRAGIndex(vcor, vclients, allp, proto_client.astype(np.int64), adj,
                                 np.eye(LATENT_DIM, dtype=np.float32), 0.5, 1, cfg, qmap=qmap)
        w, alpha, rec = _train(untrained, qs, sites, members)
        np.savez(d / "router.npz", protos=allp, proto_client=proto_client, adj=adj, w=w.astype(np.float32),
                 alpha=np.float32(alpha), qmap=qmap)
        cfg.update(rec)
        (d / "config.json").write_text(json.dumps(cfg, indent=1))
        clear_caches()
        log.info("%s router %s: trained encoders (%s), B=%d, recall %.3f, hit@B %s", federation, key or "all",
                 record.get("from") or f"loss {record['loss'][:1]} -> {record['loss'][-1:]}", rec["b"],
                 rec["recall_by_b"].get(str(rec["b"]), 0), rec["hit_at_b"])
        outs.append(d)
    return outs


def summary(federations: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for f in dict.fromkeys(federations):
        if not (layout_dir(f) / "config.json").exists():
            out[f] = None
            continue
        per: dict[str, Any] = {}
        for key in _router_keys(f):
            for enc in (None, *ENCODERS):
                p = variant_dir(f, key, enc) / "config.json"
                if not p.exists():
                    continue
                cfg = json.loads(p.read_text())
                own = cfg["routers"][key] if enc is None and key is not None else cfg
                per.setdefault(key or "all", {})[enc or "free"] = {
                    **{k: own.get(k) for k in ("b", "recall_by_b", "hit_at_b", "train_questions")},
                    "tuned": (cfg.get("tuned") or {}).get(key or "all"),
                    **({"encoder_training": cfg.get("encoder_training")} if enc else {})}
        out[f] = per
    return out


def trained_built(federation: str) -> bool:
    names = ("config.json", "clients.npz", "router.npz", "encoder.npz")
    return (layout_dir(federation) / "config.json").exists() and all(
        (variant_dir(federation, key, "trained") / n).exists() for key in _router_keys(federation) for n in names)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m fedquest.fedgrag")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="corpus tables (once per dataset), then each layout's clients and router")
    b.add_argument("federations", nargs="+")
    b.add_argument("--mode", default="remote", choices=("remote", "local"))
    b.add_argument("--halves", action="store_true",
                   help="Cross-fitting: one router per half of --split's questions (router_xA, router_xB)")
    b.add_argument("--split", default="test", help="the questions the --halves routers train on (default test)")
    b.add_argument("--missing", action="store_true",
                   help="build only the layouts with no index yet (a built layout, router included, is kept)")
    c = sub.add_parser("corpus", help="only the corpus tables of the datasets")
    c.add_argument("datasets", nargs="+", choices=DATASETS)
    c.add_argument("--force", action="store_true")
    s = sub.add_parser("status", help="what is built")
    s.add_argument("federations", nargs="*")
    r = sub.add_parser("retrain", help="only the routers of built layouts, on questions outside the evaluation pool")
    r.add_argument("federations", nargs="+")
    e = sub.add_parser("trained", help="Each built layout's trained-encoder index (FedGRAG's GNN and fusion "
                                       "MLP), per router of its build")
    e.add_argument("federations", nargs="+")
    e.add_argument("--encoder-from", dest="source",
                   help="apply this layout's trained weights (its benchmark's main federation) instead of training, "
                        "on the layouts of its dataset")
    e.add_argument("--missing", action="store_true", help="only the layouts with no trained-encoder index yet")
    t = sub.add_parser("tune", help="The open retrieval settings of built layouts, on each router's own "
                                    "training questions (LLM-free)")
    t.add_argument("federations", nargs="+")
    t.add_argument("--encoder", choices=ENCODERS, help="the trained-encoder index (default: the training-free one)")
    t.add_argument("--from", dest="source",
                   help="this layout's tuned settings (its benchmark's main federation), with only B picked anew, on "
                        "the layouts of its dataset")
    m = sub.add_parser("summary", help="Every layout's indexes as built and tuned, as JSON")
    m.add_argument("federations", nargs="+")
    m.add_argument("--out", help="write it here (default: print it)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.cmd == "summary":
        text = json.dumps(summary(args.federations), indent=1)
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(text + "\n")
        else:
            print(text)
        return 0
    if args.cmd == "retrain":
        for f in args.federations:
            retrain_router(f)
        return 0
    if args.cmd in ("trained", "tune"):
        from fedquest.config import load_federation

        source_dataset = load_federation(args.source).dataset if args.source else None
        failed = 0
        for f in args.federations:
            same = args.source and f != args.source and load_federation(f).dataset == source_dataset
            src = args.source if same else None
            if args.cmd == "trained" and args.missing and trained_built(f):
                log.info("FedGRAG trained encoders of %s exist: kept (--missing)", f)
                continue
            try:
                if args.cmd == "trained":
                    build_trained(f, src)
                else:
                    tune_layout(f, args.encoder, src)
            except Exception as exc:
                log.exception("FedGRAG %s of %s failed: %s", args.cmd, f, exc)
                failed += 1
        return 1 if failed else 0
    if args.cmd == "corpus":
        for d in args.datasets:
            corpus(d, force=args.force)
        return 0
    if args.cmd == "status":
        for f in args.federations or sorted(p.name for p in (runs_root() / "fedgrag" / "layouts").glob("*")):
            p = layout_dir(f) / "config.json"
            if p.exists():
                cfg = json.loads(p.read_text())
                if cfg.get("routers"):
                    for name, r in sorted(cfg["routers"].items()):
                        print(f"{f} {name}: {len(cfg['clients'])} clients, B={r['b']}, hit@B={r.get('hit_at_b')}, "
                              f"recall@B={r.get('recall_by_b', {}).get(str(r['b']))}, trained on "
                              f"{r.get('train_questions')} questions of {r.get('trained_on')}")
                else:
                    print(f"{f}: {len(cfg['clients'])} clients, B={cfg['b']}, hit@B={cfg.get('hit_at_b')}, "
                          f"recall@B={cfg.get('recall_by_b', {}).get(str(cfg['b']))}")
                for key in _router_keys(f):
                    for enc in (None, *ENCODERS):
                        vp = variant_dir(f, key, enc) / "config.json"
                        vc = json.loads(vp.read_text()) if vp.exists() else {}
                        tuned = (vc.get("tuned") or {}).get(key or "all")
                        if enc is not None and vc:
                            print(f"{f} {key or 'all'} {enc}: B={vc.get('b')}, "
                                  f"recall@B={vc.get('recall_by_b', {}).get(str(vc.get('b')))}")
                        if tuned:
                            print(f"{f} {key or 'all'} {enc or 'free'} tuned: B={tuned['b']}, recall {tuned['recall']} "
                                  f"(main runs' {tuned['default']['recall']}), "
                                  f"{ {k: tuned['settings'][k] for k in TUNE_GRID} }")
            else:
                print(f"{f}: NOT BUILT")
        return 0
    failed = 0
    for f in args.federations:
        if args.missing and (layout_dir(f) / "config.json").exists() and (layout_dir(f) / "clients.npz").exists() \
                and (layout_dir(f) / "router.npz").exists():
            log.info("FedGRAG index of %s exists: kept (--missing)", f)
            continue
        try:
            build_layout(f, args.mode, halves=args.halves, split=args.split)
        except Exception as e:
            log.exception("FedGRAG build of %s failed: %s", f, e)
            failed += 1
    return 1 if failed else 0
