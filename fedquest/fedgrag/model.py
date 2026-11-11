from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, fields
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from fedquest.paths import runs_root

ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
K_PROTOS = 16
L_NEIGHBORS = 6
T_STEPS = 2
GAMMA = 0.1
N_SEEDS = 5
HOPS = 2
FRONTIER = 50
HUB_CAP = 200
N_EDGES = 30
START_FACTS = 5
MAX_FACTS = 60
LATENT_DIM = 64
ENCODERS = ("trained",)


@dataclass(frozen=True)
class Settings:
    n_seeds: int = N_SEEDS
    hops: int = HOPS
    frontier: int = FRONTIER
    hub_cap: int = HUB_CAP
    n_edges: int = N_EDGES
    start_facts: int = START_FACTS
    max_facts: int = MAX_FACTS

    @classmethod
    def of(cls, values: dict[str, Any] | None) -> Settings:
        names = {f.name for f in fields(cls)}
        return cls(**{k: int(v) for k, v in (values or {}).items() if k in names})

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


DEFAULT = Settings()


def root() -> Path:
    return runs_root() / "fedgrag"


def dataset_dir(dataset: str) -> Path:
    return root() / "corpus" / dataset


def layout_dir(federation: str) -> Path:
    return root() / "layouts" / federation


def variant_dir(federation: str, router: str | None = None, encoder: str | None = None) -> Path:
    if encoder is None:
        return layout_dir(federation)
    if encoder not in ENCODERS:
        raise ValueError(f"FedGRAG encoder {encoder!r} is not one of {ENCODERS} (None: the training-free one)")
    return layout_dir(federation) / (encoder if router is None else f"{encoder}_{router}")


def norm_name(s: str) -> str:
    return " ".join(str(s).split()).casefold()


def doc_of(oid: str) -> str:
    tail = oid.split("/", 2)[-1]
    return tail.split("#", 1)[0]


class Encoder:
    _lock = threading.Lock()
    _model: Any = None

    @classmethod
    def encode(cls, texts: list[str], batch: int = 256) -> np.ndarray:
        with cls._lock:
            if cls._model is None:
                from sentence_transformers import SentenceTransformer

                cls._model = SentenceTransformer(ENCODER, device="cpu")
            if not texts:
                return np.zeros((0, 384), dtype=np.float32)
            out = cls._model.encode(texts, batch_size=batch, normalize_embeddings=True, show_progress_bar=False,
                                    convert_to_numpy=True)
        return np.asarray(out, dtype=np.float32)


@dataclass
class Corpus:
    names: list[str]
    facts: np.ndarray
    relations: list[str]
    fact_oids: list[str]
    emb_nodes: np.ndarray
    emb_facts: np.ndarray

    def fact_text(self, f: int) -> str:
        h, t = self.facts[f]
        return f"{self.names[h]} {self.relations[f]} {self.names[t]}"

    @classmethod
    def load(cls, dataset: str) -> Corpus:
        return _load_corpus(dataset)


@lru_cache(maxsize=4)
def _load_corpus(dataset: str) -> Corpus:
    d = dataset_dir(dataset)
    meta = json.loads((d / "tables.json").read_text())
    arr = np.load(d / "tables.npz")
    return Corpus(meta["names"], arr["facts"], meta["relations"], meta["fact_oids"],
                  np.load(d / "emb_nodes.npy").astype(np.float32), np.load(d / "emb_facts.npy").astype(np.float32))


@dataclass
class Client:
    site: str
    fact_ids: np.ndarray
    node_ids: np.ndarray
    struct: np.ndarray
    incident: dict[int, np.ndarray]


class FedGRAGIndex:
    def __init__(self, corpus: Corpus, clients: list[Client], protos: np.ndarray, proto_client: np.ndarray,
                 adj: np.ndarray, w: np.ndarray, alpha: float, b: int, config: dict[str, Any],
                 qmap: np.ndarray | None = None) -> None:
        self.corpus, self.clients, self.protos, self.proto_client = corpus, clients, protos, proto_client
        self.adj, self.w, self.alpha, self.b, self.config = adj, w, float(alpha), int(b), config
        self.qmap = None if qmap is None else np.asarray(qmap, dtype=np.float32)
        self.node_pos = [{int(n): i for i, n in enumerate(c.node_ids)} for c in clients]

    @classmethod
    def load(cls, federation: str, router: str | None = None, encoder: str | None = None) -> FedGRAGIndex:
        return _load_index(federation, router, encoder)

    def tuned(self, router: str | None = None) -> dict[str, Any] | None:
        return (self.config.get("tuned") or {}).get(router or "all")

    def fuse_query(self, q: np.ndarray) -> np.ndarray:
        if self.qmap is not None:
            v = self.qmap @ q
            return (v / max(1e-9, float(np.linalg.norm(v)))).astype(np.float32)
        v = np.concatenate([q, q]) / np.sqrt(2.0)
        return v.astype(np.float32)

    def client_scores(self, q: np.ndarray) -> np.ndarray:
        qf = self.fuse_query(q) @ self.w.T
        qf = qf / max(1e-9, float(np.linalg.norm(qf)))
        r0 = self.protos @ qf
        start = np.zeros_like(r0)
        top = np.argsort(-r0)[:L_NEIGHBORS]
        start[top] = r0[top]
        r = start
        for _ in range(T_STEPS):
            r = self.alpha * start + (1.0 - self.alpha) * (self.adj @ r)
        scores = np.full(len(self.clients), -np.inf, dtype=np.float32)
        np.maximum.at(scores, self.proto_client, r.astype(np.float32))
        return scores

    def route(self, q: np.ndarray, b: int | None = None) -> list[int]:
        scores = self.client_scores(q)
        order = sorted(range(len(self.clients)), key=lambda i: (-float(scores[i]), self.clients[i].site))
        return order[: max(1, min(len(order), b or self.b))]

    def client_facts(self, ci: int, q: np.ndarray, s: Settings = DEFAULT) -> list[tuple[int, float]]:
        c, cor = self.clients[ci], self.corpus
        if not len(c.node_ids):
            return []
        if self.qmap is None:
            psi = cor.emb_nodes[c.node_ids]
            sims = 0.5 * (psi @ q + c.struct @ q)
        else:
            sims = c.struct @ self.fuse_query(q)
        seeds = list(np.argsort(-sims)[:s.n_seeds])
        reached = set(int(n) for n in seeds)
        frontier = [int(n) for n in seeds]
        for _ in range(s.hops):
            nxt: dict[int, float] = {}
            for n in frontier:
                fids = c.incident.get(n)
                if fids is None or not len(fids):
                    continue
                rel = cor.emb_facts[fids] @ q
                for j in np.argsort(-rel)[:s.hub_cap]:
                    f = int(fids[j])
                    for e in cor.facts[f]:
                        loc = self.node_pos[ci].get(int(e))
                        if loc is not None and loc not in reached:
                            nxt[loc] = max(nxt.get(loc, -1.0), float(rel[j]))
            frontier = [n for n, _ in sorted(nxt.items(), key=lambda kv: -kv[1])[:s.frontier]]
            reached.update(frontier)
        cand: set[int] = set()
        for n in reached:
            fids = c.incident.get(n)
            if fids is None:
                continue
            for f in fids:
                h, t = cor.facts[int(f)]
                lh, lt = self.node_pos[ci].get(int(h)), self.node_pos[ci].get(int(t))
                if lh in reached and lt in reached:
                    cand.add(int(f))
        if not cand:
            return []
        ids = np.fromiter(cand, dtype=np.int64)
        rel = cor.emb_facts[ids] @ q
        best = np.argsort(-rel)[:s.n_edges]
        return [(int(ids[i]), float(rel[i])) for i in best]

    def compose(self, per_client: dict[int, list[tuple[int, float]]], s: Settings = DEFAULT
                ) -> list[tuple[int, int, float]]:
        best: dict[int, tuple[int, float]] = {}
        for ci, facts in per_client.items():
            for f, rel in facts:
                if f not in best or rel > best[f][1]:
                    best[f] = (ci, rel)
        if not best:
            return []
        by_entity: dict[int, list[int]] = {}
        for f in best:
            for e in self.corpus.facts[f]:
                by_entity.setdefault(int(e), []).append(f)
        ranked = sorted(best, key=lambda f: (-best[f][1], f))
        order: list[int] = []
        seen: set[int] = set()
        queue = ranked[:s.start_facts]
        seen.update(queue)
        while queue and len(order) < s.max_facts:
            f = queue.pop(0)
            order.append(f)
            nbrs = {g for e in self.corpus.facts[f] for g in by_entity.get(int(e), []) if g not in seen}
            for g in sorted(nbrs, key=lambda g: (-best[g][1], g)):
                seen.add(g)
                queue.append(g)
        for f in ranked:
            if len(order) >= s.max_facts:
                break
            if f not in seen:
                order.append(f)
                seen.add(f)
        return [(f, best[f][0], best[f][1]) for f in order[:s.max_facts]]

    def retrieve(self, q: np.ndarray, b: int | None = None, s: Settings = DEFAULT
                 ) -> tuple[list[int], dict[int, list[tuple[int, float]]], list[tuple[int, int, float]]]:
        chosen = self.route(q, b)
        per = {ci: self.client_facts(ci, q, s) for ci in chosen}
        return chosen, per, self.compose(per, s)

    def package_bytes(self, facts: list[tuple[int, float]]) -> int:
        nodes = {int(e) for f, _ in facts for e in self.corpus.facts[f]}
        vocab = {w for f, _ in facts for w in self.corpus.fact_text(f).split()}
        return len(nodes) * LATENT_DIM * 4 + len(facts) * 12 + sum(len(w) + 1 for w in vocab)


def router_file(federation: str, router: str | None = None, encoder: str | None = None) -> Path:
    if encoder is not None:
        return variant_dir(federation, router, encoder) / "router.npz"
    return layout_dir(federation) / ("router.npz" if router is None else f"router_{router}.npz")


@lru_cache(maxsize=32)
def _load_index(federation: str, router: str | None = None, encoder: str | None = None) -> FedGRAGIndex:
    d = variant_dir(federation, router, encoder)
    cfg = json.loads((d / "config.json").read_text())
    corpus, clients = _load_clients(federation) if encoder is None else _load_variant_clients(str(d))
    path = router_file(federation, router, encoder)
    if not path.exists():
        raise FileNotFoundError(f"no FedGRAG router {path}: python -m fedquest.fedgrag "
                                + (f"build {federation}" + (" --halves" if router else "") if encoder is None
                                   else f"trained {federation}"))
    arr = np.load(path)
    b = int(cfg["b"]) if router is None or encoder is not None else int(cfg["routers"][router]["b"])
    qmap = arr["qmap"].astype(np.float32) if "qmap" in arr.files else None
    return FedGRAGIndex(corpus, clients, arr["protos"].astype(np.float32), arr["proto_client"].astype(np.int64),
                        arr["adj"].astype(np.float32), arr["w"].astype(np.float32), float(arr["alpha"]), b, cfg,
                        qmap=qmap)


@lru_cache(maxsize=16)
def _load_clients(federation: str) -> tuple[Corpus, list[Client]]:
    return _read_clients(layout_dir(federation))


@lru_cache(maxsize=16)
def _load_variant_clients(d: str) -> tuple[Corpus, list[Client]]:
    return _read_clients(Path(d))


def _read_clients(d: Path) -> tuple[Corpus, list[Client]]:
    cfg = json.loads((d / "config.json").read_text())
    corpus = Corpus.load(cfg["dataset"])
    clients: list[Client] = []
    with np.load(d / "clients.npz") as z:
        for i, site in enumerate(cfg["clients"]):
            fids = z[f"facts_{i}"].astype(np.int64)
            nids = z[f"nodes_{i}"].astype(np.int64)
            struct = z[f"struct_{i}"].astype(np.float32)
            pos = {int(n): j for j, n in enumerate(nids)}
            inc: dict[int, list[int]] = {}
            for f in fids:
                for e in corpus.facts[int(f)]:
                    j = pos.get(int(e))
                    if j is not None:
                        inc.setdefault(j, []).append(int(f))
            clients.append(Client(site, fids, nids, struct, {j: np.asarray(v, dtype=np.int64)
                                                              for j, v in inc.items()}))
    return corpus, clients


def clear_caches() -> None:
    _load_index.cache_clear()
    _load_clients.cache_clear()
    _load_variant_clients.cache_clear()
