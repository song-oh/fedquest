# HippoRAG 2 at a site: OpenIE, the phrase-passage graph and PPR retrieval
from __future__ import annotations

import ast
import asyncio
import difflib
import hashlib
import itertools
import json
import logging
import os
import re
import shutil
import threading
import time
from collections import Counter
from collections.abc import Collection, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic import BaseModel, Field, TypeAdapter

from fedquest.errors import AuthExpired
from fedquest.site import hippo_prompts as hp
from fedquest.types import ObjectRecord, Usage
from fedquest.utils import stable_hash

if TYPE_CHECKING:
    import scipy.sparse as sp

    from fedquest.llm.client import LLMClient, LLMReply
    from fedquest.vdb.encoders import Encoder

log = logging.getLogger(__name__)

HIPPO_VERSION = "hippo2-v1"
HIPPO_DIR = "hippo"
LINKING_TOP_K = 5
PASSAGE_NODE_WEIGHT = 0.05
DAMPING = 0.5
QA_TOP_K = 5
SYNONYMY_TOP_K = 2047
SYNONYMY_THRESHOLD = 0.8
SYNONYMY_CAP = 100
SYNONYMY_MIN_ALNUM = 2
SYNONYMY_BATCH = 1000
OPENIE_MAX_TOKENS = 2048
FILTER_MAX_TOKENS = 512
PPR_TOL = 1e-12
PPR_MAX_ITER = 1000
ROLE = "extract_text"


class HippoBuildError(RuntimeError):
    pass


def text_processing(text: Any) -> Any:
    if isinstance(text, list):
        return [text_processing(t) for t in text]
    if not isinstance(text, str):
        text = str(text)
    return re.sub("[^A-Za-z0-9 ]", " ", text.lower()).strip()


def min_max_normalize(x: np.ndarray) -> np.ndarray:
    min_val = np.min(x)
    max_val = np.max(x)
    range_val = max_val - min_val
    if range_val == 0:
        return np.ones_like(x)
    return (x - min_val) / range_val


def fix_broken_generated_json(json_str: str) -> str:
    def find_unclosed(json_str: str) -> list[str]:
        unclosed: list[str] = []
        inside_string = False
        escape_next = False
        for char in json_str:
            if inside_string:
                if escape_next:
                    escape_next = False
                elif char == "\\":
                    escape_next = True
                elif char == '"':
                    inside_string = False
            elif char == '"':
                inside_string = True
            elif char in "{[":
                unclosed.append(char)
            elif char in "}]" and unclosed and ((char == "}" and unclosed[-1] == "{")
                                               or (char == "]" and unclosed[-1] == "[")):
                unclosed.pop()
        return unclosed

    try:
        json.loads(json_str)
        return json_str
    except json.JSONDecodeError:
        pass
    # truncated JSON: drop the unfinished item after the last comma, then close what is open
    last_comma_index = json_str.rfind(",")
    if last_comma_index != -1:
        json_str = json_str[:last_comma_index]
    closing_map = {"{": "}", "[": "]"}
    for open_char in reversed(find_unclosed(json_str)):
        json_str += closing_map[open_char]
    return json_str


def filter_invalid_triples(triples: Sequence[Any]) -> list[list[str]]:
    unique_triples = set()
    valid_triples = []
    for triple in triples:
        if len(triple) != 3:
            continue
        valid_triple = [str(item) for item in triple]
        if tuple(valid_triple) not in unique_triples:
            unique_triples.add(tuple(valid_triple))
            valid_triples.append(valid_triple)
    return valid_triples


_NER_JSON = re.compile(r'\{[^{}]*"named_entities"\s*:\s*\[[^\]]*\][^{}]*\}', re.DOTALL)
_TRIPLES_JSON = re.compile(r'\{[^{}]*"triples"\s*:\s*\[[^\]]*\][^{}]*\}', re.DOTALL)


def reply_text(reply: LLMReply) -> str:
    return fix_broken_generated_json(reply.text) if reply.finish_reason == "length" else reply.text


def parse_ner(text: str) -> list[Any]:
    try:
        match = _NER_JSON.search(text)
        if match is None:
            return []
        return list(dict.fromkeys(ast.literal_eval(match.group())["named_entities"]))
    except Exception:
        return []


def parse_triples(text: str) -> list[list[str]]:
    try:
        match = _TRIPLES_JSON.search(text)
        if match is None:
            return []
        return filter_invalid_triples(ast.literal_eval(match.group())["triples"])
    except Exception:
        return []


class _Fact(BaseModel):
    fact: list[list[str]] = Field(description="A list of facts, each fact is a list of 3 strings: "
                                              "[subject, predicate, object]")


_FACT = TypeAdapter(_Fact)
_FIELD_HEADER = re.compile(r"\[\[ ## (\w+) ## \]\]")


def parse_filter(response: str) -> list[list[str]]:
    sections: list[tuple[str | None, list[str]]] = [(None, [])]
    for line in response.splitlines():
        match = _FIELD_HEADER.match(line.strip())
        if match:
            sections.append((match.group(1), []))
        else:
            sections[-1][1].append(line)
    parsed: list[list[str]] = []
    for k, value in ((k, "\n".join(v).strip()) for k, v in sections):
        if k == "fact_after_filter":
            try:
                try:
                    parsed_value: Any = json.loads(value)
                except json.JSONDecodeError:
                    try:
                        parsed_value = ast.literal_eval(value)
                    except (ValueError, SyntaxError):
                        parsed_value = value
                parsed = _FACT.validate_python(parsed_value).fact
            except Exception:
                log.debug("a filter reply's section does not parse: %r", value[:200])
    return parsed


def match_filtered(generated: Sequence[Sequence[str]], candidates: Sequence[tuple[str, ...]]) -> list[int]:
    if not candidates:
        return []
    shown = [str(c) for c in candidates]
    out = []
    for fact in generated:
        closest = difflib.get_close_matches(str(fact), shown, n=1, cutoff=0.0)[0]
        out.append(shown.index(closest))
    return out


async def recognition_filter(question: str, candidates: Sequence[tuple[str, ...]], indices: Sequence[int],
                             llm: LLMClient, keep: int = LINKING_TOP_K, slots: asyncio.Semaphore | None = None,
                             ) -> tuple[list[int], list[tuple[str, ...]], Usage]:
    fact_before_filter = {"fact": [list(c) for c in candidates]}
    messages = hp.filter_messages(question, json.dumps(fact_before_filter))
    if slots is None:
        reply = await llm.chat(messages, role=ROLE, task="hippo_filter", max_tokens=FILTER_MAX_TOKENS)
    else:
        async with slots:
            reply = await llm.chat(messages, role=ROLE, task="hippo_filter", max_tokens=FILTER_MAX_TOKENS)
    kept = match_filtered(parse_filter(reply.text), candidates)
    return [indices[i] for i in kept][:keep], [candidates[i] for i in kept][:keep], reply.usage


async def openie(passage: str, llm: LLMClient) -> dict[str, Any]:
    ner = await llm.chat(hp.ner_messages(passage), role=ROLE, task="hippo_ner", max_tokens=OPENIE_MAX_TOKENS)
    entities = parse_ner(reply_text(ner))
    triple = await llm.chat(hp.triple_messages(passage, entities), role=ROLE, task="hippo_triples",
                            max_tokens=OPENIE_MAX_TOKENS)
    usage = Usage()
    usage.add(ner.usage)
    usage.add(triple.usage)
    return {"entities": entities, "triples": parse_triples(reply_text(triple)), "usage": usage.to_dict(),
            "replayed": int(ner.cached) + int(triple.cached)}


def extract_entity_nodes(chunk_triples: Sequence[Sequence[Sequence[str]]]) -> tuple[list[str], list[list[str]]]:
    chunk_triple_entities = []
    for triples in chunk_triples:
        triple_entities: set[str] = set()
        for t in triples:
            if len(t) == 3:
                triple_entities.update([t[0], t[2]])
        chunk_triple_entities.append(sorted(triple_entities))
    return sorted({e for ents in chunk_triple_entities for e in ents}), chunk_triple_entities


def flatten_facts(chunk_triples: Sequence[Sequence[Sequence[str]]]) -> list[tuple[str, ...]]:
    return sorted({tuple(t) for triples in chunk_triples for t in triples})


def synonyms(vectors: np.ndarray, phrases: Sequence[str], threshold: float = SYNONYMY_THRESHOLD,
             cap: int = SYNONYMY_CAP, batch: int = SYNONYMY_BATCH) -> list[tuple[int, int, float]]:
    n = len(phrases)
    if n == 0:
        return []
    vecs = np.asarray(vectors, dtype=np.float32)
    vecs = vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12)
    eligible = [i for i, p in enumerate(phrases) if len(re.sub("[^A-Za-z0-9]", "", p)) > SYNONYMY_MIN_ALNUM]
    depth = min(n, cap + 4, SYNONYMY_TOP_K)
    rows_at_once = max(1, min(batch, (1 << 25) // n))
    out = []
    for start in range(0, len(eligible), rows_at_once):
        rows = eligible[start : start + rows_at_once]
        sims = vecs[rows] @ vecs.T
        if depth < n:
            top = np.argpartition(sims, n - depth, axis=1)[:, n - depth :]
        else:
            top = np.broadcast_to(np.arange(n), (len(rows), n))
        for r, i in enumerate(rows):
            scores = sims[r, top[r]].tolist()
            num_nns = 0
            for nn, score in sorted(zip(top[r].tolist(), scores, strict=True), key=lambda x: (-x[1], x[0])):
                if score < threshold or num_nns > cap:
                    break
                if nn != i and phrases[nn] != "":
                    out.append((i, nn, score))
                    num_nns += 1
    return out


def graph_edges(chunk_triples: Sequence[Sequence[Sequence[str]]], phrases: Sequence[str],
                synonym_edges: Sequence[tuple[int, int, float]],
                ) -> tuple[dict[tuple[int, int], float], np.ndarray, dict[str, int]]:
    pid = {p: i for i, p in enumerate(phrases)}
    node_to_node_stats: dict[tuple[int, int], float] = {}
    ent_node_to_chunk_ids: list[set[int]] = [set() for _ in phrases]
    fact_pairs = 0
    for chunk_key, triples in enumerate(chunk_triples):
        entities_in_chunk: set[int] = set()
        for triple in triples:
            node_key, node_2_key = pid[triple[0]], pid[triple[2]]
            node_to_node_stats[(node_key, node_2_key)] = node_to_node_stats.get((node_key, node_2_key), 0.0) + 1
            node_to_node_stats[(node_2_key, node_key)] = node_to_node_stats.get((node_2_key, node_key), 0.0) + 1
            entities_in_chunk.update((node_key, node_2_key))
            fact_pairs += 2
        for node in entities_in_chunk:
            ent_node_to_chunk_ids[node].add(chunk_key)
    _, chunk_triple_entities = extract_entity_nodes(chunk_triples)
    passage_pairs = 0
    # passage nodes come after the phrase nodes; each links to its entities with weight 1
    for idx, ents in enumerate(chunk_triple_entities):
        for chunk_ent in ents:
            node_to_node_stats[(len(phrases) + idx, pid[chunk_ent])] = 1.0
            passage_pairs += 1
    for node_key, nn, score in synonym_edges:
        node_to_node_stats[(node_key, nn)] = score
    counts = {"fact_pairs": fact_pairs, "passage_pairs": passage_pairs, "synonymy_pairs": len(synonym_edges)}
    return node_to_node_stats, np.array([len(c) for c in ent_node_to_chunk_ids], dtype=np.int64), counts


def adjacency(node_to_node_stats: dict[tuple[int, int], float], n: int) -> sp.csr_matrix:
    import scipy.sparse as sp

    m = len(node_to_node_stats)
    keys = np.fromiter(itertools.chain.from_iterable(node_to_node_stats), dtype=np.int64, count=2 * m).reshape(m, 2)
    weights = np.fromiter(node_to_node_stats.values(), dtype=np.float64, count=m)
    loops = keys[:, 0] == keys[:, 1]
    a, b, w = keys[~loops, 0], keys[~loops, 1], weights[~loops]
    graph = sp.coo_matrix((np.concatenate([w, w]), (np.concatenate([a, b]), np.concatenate([b, a]))), shape=(n, n))
    return graph.tocsr()


def ppr(graph: sp.spmatrix, reset: np.ndarray, damping: float = DAMPING, tol: float = PPR_TOL,
        max_iter: int = PPR_MAX_ITER, degree: np.ndarray | None = None) -> np.ndarray:
    reset = np.where(np.isnan(reset) | (reset < 0), 0, reset).astype(np.float64)
    r = reset / reset.sum()
    if degree is None:
        degree = np.asarray(graph.sum(axis=1), dtype=np.float64).ravel()
    inv = np.divide(1.0, degree, out=np.zeros_like(degree), where=degree > 0)
    sink = degree == 0
    walk = graph.T
    x = r.copy()
    for _ in range(max_iter):
        # the mass at sink nodes (no edges) restarts from the reset vector
        new = damping * (walk @ (x * inv) + x[sink].sum() * r) + (1 - damping) * r
        change = float(np.abs(new - x).sum())
        x = new
        if change < tol:
            break
    return x / x.sum()


def _rank(scores: np.ndarray) -> np.ndarray:
    return np.argsort(-scores, kind="stable")


def passage_text(o: Any) -> str:
    return f"{o.title}\n{o.text}"


def objects_stamp(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:16]


def _write(path: Path, write: Any, binary: bool = False) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        with open(tmp, "wb" if binary else "w", encoding=None if binary else "utf-8") as f:
            write(f)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _write_json(path: Path, obj: Any) -> None:
    _write(path, lambda f: f.write(json.dumps(obj, indent=1, ensure_ascii=False) + "\n"))


def _write_npy(path: Path, arr: np.ndarray) -> None:
    _write(path, lambda f: np.save(f, np.asarray(arr, dtype=np.float32)), binary=True)


def read_openie(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict) and "oid" in rec:
                out[rec["oid"]] = rec
    return out


PART_GLOB = "openie.part*.jsonl"


def read_all_openie(hdir: Path) -> dict[str, dict[str, Any]]:
    out = read_openie(hdir / "openie.jsonl")
    for part in sorted(hdir.glob(PART_GLOB)):
        out.update(read_openie(part))
    return out


class HippoIndex:
    def __init__(self, *, oids: list[str], positions: list[int], phrases: list[str], phrase_passages: np.ndarray,
                 facts: list[tuple[str, ...]], fact_emb: np.ndarray, passage_emb: np.ndarray, graph: sp.spmatrix,
                 meta: dict[str, Any] | None = None, stats: dict[str, Any] | None = None) -> None:
        self.oids, self.positions, self.phrases, self.facts = oids, positions, phrases, facts
        self.phrase_passages, self.fact_emb, self.passage_emb = phrase_passages, fact_emb, passage_emb
        self.graph = graph.tocsr()
        self.degree = np.asarray(self.graph.sum(axis=1), dtype=np.float64).ravel()
        self.meta, self.stats = meta or {}, stats or {}
        self.phrase_id = {p: i for i, p in enumerate(phrases)}
        self.counts: Counter[str] = Counter()

    @property
    def n_nodes(self) -> int:
        return len(self.phrases) + len(self.oids)

    def fact_scores(self, query: np.ndarray) -> np.ndarray:
        if len(self.facts) == 0:
            return np.array([])
        return min_max_normalize(np.asarray(self.fact_emb @ query))

    def dense_passage_retrieval(self, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        query_doc_scores = min_max_normalize(np.asarray(self.passage_emb @ query))
        sorted_doc_ids = _rank(query_doc_scores)
        return sorted_doc_ids, query_doc_scores[sorted_doc_ids]

    async def rerank_facts(self, question: str, query_fact_scores: np.ndarray, llm: LLMClient,
                           slots: asyncio.Semaphore | None = None,
                           ) -> tuple[list[int], list[tuple[str, ...]], dict[str, Any]]:
        if len(query_fact_scores) == 0 or len(self.facts) == 0:
            return [], [], {}
        candidate_fact_indices = _rank(query_fact_scores)[:LINKING_TOP_K].tolist()
        candidate_facts = [self.facts[i] for i in candidate_fact_indices]
        top_k_fact_indices, top_k_facts, usage = await recognition_filter(
            question, candidate_facts, candidate_fact_indices, llm, LINKING_TOP_K, slots)
        return top_k_fact_indices, top_k_facts, {"facts_before_rerank": candidate_facts,
                                                 "facts_after_rerank": top_k_facts, "usage": usage}

    def get_top_k_weights(self, link_top_k: int, all_phrase_weights: np.ndarray,
                          linking_score_map: dict[str, float]) -> tuple[np.ndarray, dict[str, float]]:
        linking_score_map = dict(sorted(linking_score_map.items(), key=lambda x: x[1], reverse=True)[:link_top_k])
        keep = np.zeros(len(all_phrase_weights), dtype=bool)
        keep[[self.phrase_id[p] for p in linking_score_map]] = True
        all_phrase_weights[~keep] = 0.0
        return all_phrase_weights, linking_score_map

    def graph_search_with_fact_entities(self, query: np.ndarray, link_top_k: int, query_fact_scores: np.ndarray,
                                        top_k_facts: Sequence[tuple[str, ...]], top_k_fact_indices: Sequence[int],
                                        passage_node_weight: float = PASSAGE_NODE_WEIGHT,
                                        ) -> tuple[np.ndarray, np.ndarray]:
        phrase_weights = np.zeros(self.n_nodes)
        passage_weights = np.zeros(self.n_nodes)
        number_of_occurs = np.zeros(self.n_nodes)
        phrases_and_ids: dict[str, int | None] = {}
        for rank, f in enumerate(top_k_facts):
            fact_score = query_fact_scores[top_k_fact_indices[rank]] if query_fact_scores.ndim > 0 else \
                query_fact_scores
            # each entity of a fact gets its score over the entity's passage count, averaged over facts
            for phrase in (f[0].lower(), f[2].lower()):
                phrase_id = self.phrase_id.get(phrase)
                if phrase_id is not None:
                    weighted_fact_score = fact_score
                    if self.phrase_passages[phrase_id] > 0:
                        weighted_fact_score /= int(self.phrase_passages[phrase_id])
                    phrase_weights[phrase_id] += weighted_fact_score
                    number_of_occurs[phrase_id] += 1
                phrases_and_ids.setdefault(phrase, phrase_id)
        with np.errstate(invalid="ignore", divide="ignore"):
            phrase_weights /= number_of_occurs
        linking_score_map = {p: float(phrase_weights[i]) for p, i in phrases_and_ids.items() if i is not None}
        if link_top_k:
            phrase_weights, linking_score_map = self.get_top_k_weights(link_top_k, phrase_weights, linking_score_map)
        dpr_sorted_doc_ids, dpr_sorted_doc_scores = self.dense_passage_retrieval(query)
        normalized_dpr_sorted_scores = min_max_normalize(dpr_sorted_doc_scores)
        passage_weights[len(self.phrases) + dpr_sorted_doc_ids] = normalized_dpr_sorted_scores * passage_node_weight
        node_weights = phrase_weights + passage_weights
        if not np.nansum(node_weights) > 0:
            return dpr_sorted_doc_ids, dpr_sorted_doc_scores
        doc_scores = ppr(self.graph, node_weights, DAMPING, degree=self.degree)[len(self.phrases):]
        sorted_doc_ids = _rank(doc_scores)
        return sorted_doc_ids, doc_scores[sorted_doc_ids]

    async def retrieve(self, question: str, k: int, *, query: np.ndarray, llm: LLMClient,
                       slots: asyncio.Semaphore | None = None, usage: Usage | None = None) -> list[int]:
        if not self.oids or k <= 0:
            return []
        query = np.asarray(query, dtype=np.float32)
        query_fact_scores = await asyncio.to_thread(self.fact_scores, query)
        top_k_fact_indices, top_k_facts, rerank_log = await self.rerank_facts(question, query_fact_scores, llm, slots)
        self.counts["retrievals"] += 1
        if rerank_log:
            used = rerank_log["usage"]
            self.counts.update(filter_calls=1, filter_prompt_tokens=used.prompt_tokens,
                               filter_completion_tokens=used.completion_tokens, kept_facts=len(top_k_facts))
            if usage is not None:
                usage.add(used)
        if len(top_k_facts) == 0:
            self.counts["dense"] += 1
            sorted_doc_ids, _ = await asyncio.to_thread(self.dense_passage_retrieval, query)
        else:
            self.counts["graph"] += 1
            sorted_doc_ids, _ = await asyncio.to_thread(
                self.graph_search_with_fact_entities, query, LINKING_TOP_K, query_fact_scores, top_k_facts,
                top_k_fact_indices, PASSAGE_NODE_WEIGHT)
        return [self.positions[i] for i in sorted_doc_ids[:k].tolist()]

    @classmethod
    def load(cls, path: Path) -> HippoIndex | None:
        import scipy.sparse as sp

        try:
            meta = json.loads((path / "meta.json").read_text())
            if meta.get("version") != HIPPO_VERSION or meta.get("state") != "ready":
                return None
            nodes = json.loads((path / "nodes.json").read_text())
            stats = json.loads((path / "stats.json").read_text()) if (path / "stats.json").exists() else {}
            return cls(oids=nodes["passages"], positions=nodes["positions"], phrases=nodes["phrases"],
                       phrase_passages=np.asarray(nodes["phrase_passages"], dtype=np.int64),
                       facts=[tuple(f) for f in json.loads((path / "facts.json").read_text())],
                       fact_emb=np.load(path / "fact_emb.npy", mmap_mode="r"),
                       passage_emb=np.load(path / "passage_emb.npy", mmap_mode="r"),
                       graph=sp.load_npz(path / "graph.npz"), meta=meta, stats=stats)
        except (OSError, ValueError, KeyError) as e:
            log.warning("%s: the HippoRAG 2 index does not load (%s: %s)", path, type(e).__name__, e)
            return None

    def save(self, path: Path, entity_emb: np.ndarray) -> None:
        import scipy.sparse as sp

        path.mkdir(parents=True, exist_ok=True)
        _write(path / "graph.npz", lambda f: sp.save_npz(f, self.graph), binary=True)
        _write_json(path / "nodes.json", {"phrases": self.phrases, "phrase_passages": self.phrase_passages.tolist(),
                                          "passages": self.oids, "positions": self.positions})
        _write_json(path / "facts.json", [list(f) for f in self.facts])
        _write_npy(path / "entity_emb.npy", entity_emb)
        _write_npy(path / "fact_emb.npy", self.fact_emb)
        _write_npy(path / "passage_emb.npy", self.passage_emb)
        _write_json(path / "stats.json", self.stats)
        _write_json(path / "meta.json", self.meta)

    @classmethod
    async def build(cls, path: Path, objects: Sequence[ObjectRecord], *, encoder: Encoder, llm: LLMClient,
                    concurrency: int = 16, skip: Collection[int] = (), stamp: str = "", force: bool = False,
                    label: str = "") -> HippoIndex:
        model = llm.model(ROLE)
        want = {"version": HIPPO_VERSION, "model": model, "prompts": hp.PROMPT_SHA}
        if force and path.exists():
            shutil.rmtree(path)
        old = cls.load(path) if (path / "meta.json").exists() else None
        current = {**want, "objects_sha256": stamp, "encoder": encoder.spec.name}
        if old is not None and all(old.meta.get(k) == v for k, v in current.items()):
            log.info("%s: HippoRAG 2 index up to date (%d passages)", label, len(old.oids))
            return old
        path.mkdir(parents=True, exist_ok=True)
        # OpenIE records of an earlier build are reused: it must match in version, model and prompts
        if (path / "meta.json").exists():
            meta = json.loads((path / "meta.json").read_text())
            if any(meta.get(k) != v for k, v in want.items()):
                raise HippoBuildError(f"{label}: {path} holds a build of another version, model or prompts "
                                      f"({ {k: meta.get(k) for k in want} }): rebuild it with --force")
        _write_json(path / "meta.json", {**want, "state": "openie", "encoder": encoder.spec.name,
                                         "objects_sha256": stamp})
        t0 = time.perf_counter()
        left_out = set(skip)
        passages = [(i, o) for i, o in enumerate(objects) if o.modality == "text" and i not in left_out]
        records = await run_openie(path / "openie.jsonl", passages, llm, concurrency, label)
        t1 = time.perf_counter()
        index, entity_emb, counts, embed_s = await asyncio.to_thread(cls.assemble, passages, records, encoder, label)
        usage = Usage()
        for _, o in passages:
            usage.add(Usage.from_dict(records[o.oid]["usage"]))
        replayed = sum(int(records[o.oid].get("replayed", 0)) for _, o in passages)
        index.stats = {
            "version": HIPPO_VERSION, "label": label, "encoder": encoder.spec.name, "model": model,
            "upstream": hp.UPSTREAM, **counts,
            "empty_passages": sum(1 for _, o in passages if not records[o.oid]["triples"]),
            "openie": {"calls": usage.calls, "replayed": replayed, "made": usage.calls - replayed,
                       "prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens,
                       "reasoning_tokens": usage.reasoning_tokens, "cost_units": usage.cost_units,
                       "seconds": round(usage.seconds, 3)},
            "wall_seconds": {"openie": round(t1 - t0, 1), "embed": round(embed_s, 1),
                             "graph": round(time.perf_counter() - t1 - embed_s, 1)},
        }
        index.meta = {**want, "state": "ready", "encoder": encoder.spec.name, "objects_sha256": stamp,
                      "counts": counts, "upstream": hp.UPSTREAM,
                      "defaults": {"linking_top_k": LINKING_TOP_K, "passage_node_weight": PASSAGE_NODE_WEIGHT,
                                   "damping": DAMPING, "synonymy_edge_topk": SYNONYMY_TOP_K,
                                   "synonymy_edge_sim_threshold": SYNONYMY_THRESHOLD,
                                   "synonymy_neighbours": SYNONYMY_CAP + 1, "qa_top_k": QA_TOP_K}}
        await asyncio.to_thread(index.save, path, entity_emb)
        log.info("%s: HippoRAG 2 index of %d passages, %d phrases, %d facts (%.0f s)", label, len(index.oids),
                 len(index.phrases), len(index.facts), time.perf_counter() - t0)
        return index

    @classmethod
    def assemble(cls, passages: Sequence[tuple[int, ObjectRecord]], records: dict[str, dict[str, Any]],
                 encoder: Encoder, label: str = "") -> tuple[HippoIndex, np.ndarray, dict[str, int], float]:
        chunk_triples = [[text_processing(t) for t in filter_invalid_triples(records[o.oid]["triples"])]
                         for _, o in passages]
        entity_nodes, _ = extract_entity_nodes(chunk_triples)
        facts = flatten_facts(chunk_triples)

        def embed(texts: list[str]) -> np.ndarray:
            return np.asarray(encoder.encode_docs(texts), dtype=np.float32) if texts else \
                np.zeros((0, encoder.dim), dtype=np.float32)

        t0 = time.perf_counter()
        entity_emb, fact_emb = embed(entity_nodes), embed([str(f) for f in facts])
        passage_emb = embed([passage_text(o) for _, o in passages])
        embed_s = time.perf_counter() - t0
        synonym_edges = synonyms(entity_emb, entity_nodes)
        node_to_node_stats, phrase_passages, pairs = graph_edges(chunk_triples, entity_nodes, synonym_edges)
        graph = adjacency(node_to_node_stats, len(entity_nodes) + len(passages))
        counts = {"passages": len(passages), "phrases": len(entity_nodes), "facts": len(facts), **pairs,
                  "edges": int(graph.nnz // 2)}
        log.info("%s: %s; embedded in %.0f s", label, json.dumps(counts), embed_s)
        index = cls(oids=[o.oid for _, o in passages], positions=[i for i, _ in passages], phrases=entity_nodes,
                    phrase_passages=phrase_passages, facts=facts, fact_emb=fact_emb, passage_emb=passage_emb,
                    graph=graph)
        return index, entity_emb, counts, embed_s


async def run_openie(path: Path, passages: Sequence[tuple[int, ObjectRecord]], llm: LLMClient, concurrency: int,
                     label: str = "") -> dict[str, dict[str, Any]]:
    done = read_all_openie(path.parent)
    sha = {o.oid: stable_hash(passage_text(o)) for _, o in passages}
    records = {oid: rec for oid, rec in done.items() if oid in sha and rec.get("sha") == sha[oid]}
    todo = [o for _, o in passages if o.oid not in records]
    if records:
        log.info("%s: OpenIE of %d passages kept from %s, %d to go", label, len(records), path, len(todo))
    slots = asyncio.Semaphore(max(1, concurrency))
    lock = threading.Lock()
    failed: list[tuple[str, str]] = []
    stopped: list[BaseException] = []
    t0, made = time.perf_counter(), [0]

    async def one(o: ObjectRecord) -> None:
        async with slots:
            if stopped:
                return
            try:
                rec = {"oid": o.oid, "sha": sha[o.oid], **await openie(passage_text(o), llm)}
            except AuthExpired as e:
                stopped.append(e)
                return
            except Exception as e:
                failed.append((o.oid, f"{type(e).__name__}: {e}"))
                return
            line = json.dumps(rec, ensure_ascii=False) + "\n"
            with lock, open(path, "a", encoding="utf-8") as f:
                f.write(line)
            records[o.oid] = rec
            made[0] += 1
            if made[0] % 500 == 0:
                log.info("%s: OpenIE of %d of %d passages done (%.2f passages/s)", label, made[0], len(todo),
                         made[0] / max(1e-9, time.perf_counter() - t0))

    await asyncio.gather(*(one(o) for o in todo))
    if stopped:
        raise stopped[0]
    if failed:
        raise HippoBuildError(f"{label}: the OpenIE of {len(failed)} of {len(passages)} passages failed (the first, "
                              f"{failed[0][0]}: {failed[0][1][:300]}); the {len(records)} done are kept in {path}: "
                              "build the index again to retry the others")
    return records


_LOADED: dict[tuple[str, int, int], HippoIndex | None] = {}
_LOADED_LOCK = threading.Lock()


HIPPO_ENCODER = "bge-base"


def open_index(cdir: Path, encoder: str | None, objects: Sequence[Any]) -> HippoIndex | None:
    meta = cdir / HIPPO_DIR / "meta.json"
    try:
        st = meta.stat()
    except OSError:
        return None
    key = (str(meta.resolve()), st.st_mtime_ns, st.st_size)
    with _LOADED_LOCK:
        if key not in _LOADED:
            _LOADED[key] = HippoIndex.load(cdir / HIPPO_DIR)
        index = _LOADED[key]
    if index is None:
        log.warning("%s: no finished HippoRAG 2 index (%s) there: `fedquest hippo index` builds it", cdir,
                    HIPPO_VERSION)
        return None
    why = ""
    if encoder is not None and index.meta.get("encoder") != encoder:
        why = f"its encoder is {index.meta.get('encoder')}, not {encoder}"
    elif (src := cdir / "objects.jsonl").exists() and index.meta.get("objects_sha256") != objects_stamp(src):
        why = "objects.jsonl changed since it was built"
    elif any(p >= len(objects) or objects[p].oid != oid for p, oid in zip(index.positions, index.oids, strict=True)):
        why = "its passages are not the collection's objects"
    if why:
        log.warning("%s: the HippoRAG 2 index is stale (%s): `fedquest hippo index --force` rebuilds it", cdir, why)
        return None
    return index


def local_service(endpoint: Any) -> Any:
    while not hasattr(endpoint, "service"):
        endpoint = endpoint.inner
    return endpoint.service


def bench_of(dataset: str) -> str:
    return {"hotpotqa": "hotpot"}.get(dataset, dataset)


def _total(stats: dict[str, dict[str, Any]]) -> dict[str, Any]:
    keys = ("calls", "replayed", "made", "prompt_tokens", "completion_tokens", "reasoning_tokens", "cost_units",
            "seconds")
    out: dict[str, Any] = {"passages": sum(s.get("passages", 0) for s in stats.values())}
    out.update({k: round(sum(s.get("openie", {}).get(k, 0) for s in stats.values()), 3) for k in keys})
    return out


async def index_federation(federation: str, *, sites: Sequence[str] | None = None, force: bool = False,
                           device: str = "auto", encoder: str | None = HIPPO_ENCODER) -> tuple[Path, dict[str, Any]]:
    from fedquest.config import load_federation
    from fedquest.federation.local import open_local_federation
    from fedquest.llm.client import LLMClient
    from fedquest.paths import results_dir
    from fedquest.utils import write_json

    cfg = load_federation(federation)
    if cfg.data_from:
        raise ValueError(f"{cfg.name} reads {cfg.data_from}'s data (data_from): index {cfg.data_from} instead")
    names = cfg.match_sites(sites)
    own = [n for n in names if not cfg.site(n).data_from]
    for n in names:
        if n not in own:
            log.info("%s/%s reads %s's data (data_from): that site's index serves it", cfg.name, n,
                     cfg.site(n).data_from)
    llm = LLMClient.from_env()
    built: dict[str, dict[str, Any]] = {}
    try:
        fed = open_local_federation(federation, llm=llm)
        try:
            for name in own:
                service = local_service(fed.endpoints[name])
                for coll, stats in (await service.hippo_index(force=force, device=device, encoder=encoder)).items():
                    built[f"{name}/{coll}"] = stats
        finally:
            await fed.close()
    finally:
        await llm.aclose()
    path = results_dir("hippo") / bench_of(cfg.dataset) / f"index_{cfg.name}.json"
    every = {**(json.loads(path.read_text()).get("sites", {}) if path.exists() else {}), **built}
    out = {"federation": cfg.name, "dataset": cfg.dataset, "version": HIPPO_VERSION, "upstream": hp.UPSTREAM,
           "sites": dict(sorted(every.items())), "total": _total(every)}
    write_json(path, out)
    return path, out


async def openie_part(federation: str, part: int, of: int, concurrency: int | None = None) -> dict[str, Any]:
    from fedquest.config import load_federation
    from fedquest.data.embed import index_skip
    from fedquest.llm.client import LLMClient
    from fedquest.utils import read_jsonl

    if not 0 <= part < of:
        raise ValueError(f"part {part} of {of}: 0 <= part < of")
    cfg = load_federation(federation)
    if cfg.data_from:
        raise ValueError(f"{cfg.name} reads {cfg.data_from}'s data (data_from): index {cfg.data_from} instead")
    conc = concurrency or int(os.environ.get("FEDQUEST_HIPPO_CONCURRENCY") or 0) or 160
    llm = LLMClient.from_env()
    out: dict[str, Any] = {}
    try:
        want = {"version": HIPPO_VERSION, "model": llm.model(ROLE), "prompts": hp.PROMPT_SHA}
        for site in cfg.sites:
            if site.data_from:
                continue
            for c in site.collections:
                if c.modality != "text":
                    continue
                cdir = cfg.collection_data_dir(site.name, c.name)
                objects = [ObjectRecord.from_dict(d) for d in read_jsonl(cdir / "objects.jsonl")]
                skip = index_skip(cdir, objects)
                hdir = cdir / HIPPO_DIR
                hdir.mkdir(parents=True, exist_ok=True)
                if (hdir / "meta.json").exists():
                    meta = json.loads((hdir / "meta.json").read_text())
                    if any(meta.get(k) != v for k, v in want.items()):
                        raise HippoBuildError(f"{hdir} holds a build of another version, model or prompts "
                                              f"({ {k: meta.get(k) for k in want} }): rebuild it with --force")
                passages = [(i, o) for i, o in enumerate(objects)
                            if o.modality == "text" and i not in skip and i % of == part]
                label = f"{cfg.name}/{site.name}/{c.name} part {part}/{of}"
                records = await run_openie(hdir / f"openie.part{part:03d}of{of:03d}.jsonl", passages, llm, conc,
                                           label)
                out[f"{site.name}/{c.name}"] = {"passages": len(passages),
                                                "done": sum(1 for _, o in passages if o.oid in records)}
    finally:
        await llm.aclose()
    log.info("%s: OpenIE part %d of %d: %s", federation, part, of, out)
    return out


RECALL_AT = (1, 2, 5, 10)


def recall_at(ranked: Sequence[str], gold: Sequence[str], ns: Sequence[int] = RECALL_AT) -> dict[str, float]:
    want = set(gold)
    return {f"recall@{n}": len(want & set(ranked[:n])) / len(want) for n in ns}


def _mean(rows: Sequence[dict[str, float]]) -> dict[str, float]:
    return {k: round(sum(r[k] for r in rows) / len(rows), 4) for k in rows[0]} if rows else {}


async def recall_study(federation: str, *, split: str = "final_test", k: int = 5, limit: int | None = None,
                       concurrency: int = 8) -> tuple[Path, dict[str, Any]]:
    from contextlib import AsyncExitStack

    from fedquest.config import load_federation
    from fedquest.eval.runner import load_questions, open_federations
    from fedquest.llm.client import LLMClient
    from fedquest.methods.common import rrf, split_lists
    from fedquest.paths import results_dir
    from fedquest.types import HippoRequest, ProbeRequest
    from fedquest.utils import gather_limited, write_json

    cfg = load_federation(federation)
    questions = [q for q in load_questions(federation, split, limit) if q.gold_doc_ids]
    async with AsyncExitStack() as stack:
        llm = LLMClient.from_env()
        stack.push_async_callback(llm.aclose)
        fed = await open_federations("local", llm, stack).get(federation)

        async def one(q: Any) -> dict[str, Any]:
            (hippo,), (dense,) = await asyncio.gather(fed.hippo([HippoRequest(q.question, k)]),
                                                      fed.probe([ProbeRequest(q.question, "question", k)]))
            top = {name: [c.doc_id for c in rrf(split_lists(cands), salt=q.qid)][: max(RECALL_AT)]
                   for name, cands in (("hipporag2", hippo), ("dense", dense))}
            return {"qid": q.qid, "qtype": q.qtype, "gold": list(dict.fromkeys(q.gold_doc_ids)), **top}

        rows = await gather_limited(questions, one, concurrency)
        did = {f"{name}/{coll}": counts for name, endpoint in fed.endpoints.items()
               for coll, counts in local_service(endpoint).hippo_counts().items()}
    recall = [{m: recall_at(r[m], r["gold"]) for m in ("hipporag2", "dense")} for r in rows]

    def summary(which: Sequence[int]) -> dict[str, Any]:
        return {"questions": len(which), **{m: _mean([recall[i][m] for i in which]) for m in ("hipporag2", "dense")}}

    types = sorted({r["qtype"] for r in rows})
    out = {"federation": cfg.name, "dataset": cfg.dataset, "split": split, "k": k, "limit": limit,
           "version": HIPPO_VERSION, **summary(range(len(rows))),
           "by_type": {t: summary([i for i, r in enumerate(rows) if r["qtype"] == t]) for t in types},
           "retrievals": did, "rows": rows}
    path = results_dir("hippo") / bench_of(cfg.dataset) / f"recall_{cfg.name}.json"
    write_json(path, out)
    return path, out


def index_ready(federation: str) -> dict[str, str]:
    from fedquest.config import load_federation

    cfg = load_federation(federation)
    out: dict[str, str] = {}
    for site in cfg.sites:
        if site.data_from or cfg.data_from:
            continue
        for c in site.collections:
            if c.modality != "text":
                continue
            cdir = cfg.collection_data_dir(site.name, c.name)
            try:
                meta = json.loads((cdir / HIPPO_DIR / "meta.json").read_text())
            except (OSError, ValueError):
                out[f"{site.name}/{c.name}"] = "missing"
                continue
            src = cdir / "objects.jsonl"
            if meta.get("version") != HIPPO_VERSION or meta.get("state") != "ready":
                out[f"{site.name}/{c.name}"] = "building"
            elif src.exists() and meta.get("objects_sha256") != _stamp_of(src):
                out[f"{site.name}/{c.name}"] = "stale"
            else:
                out[f"{site.name}/{c.name}"] = "ready"
    return out


_STAMPS: dict[tuple[str, int, int], str] = {}


def _stamp_of(path: Path) -> str:
    st = path.stat()
    key = (str(path.resolve()), st.st_mtime_ns, st.st_size)
    if key not in _STAMPS:
        _STAMPS[key] = objects_stamp(path)
    return _STAMPS[key]


def wait_ready(federation: str, timeout_s: float = 6 * 3600.0, every_s: float = 120.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while True:
        state = index_ready(federation)
        if state and all(s == "ready" for s in state.values()):
            log.info("%s: every HippoRAG 2 index is ready (%s)", federation, ", ".join(state))
            return True
        if time.monotonic() >= deadline:
            log.error("%s: HippoRAG 2 indexes not ready after %.0f s: %s", federation, timeout_s, state)
            return False
        log.info("%s: waiting for the HippoRAG 2 indexes (%s)", federation, state)
        time.sleep(every_s)


def remote_state(federation: str) -> dict[str, str] | None:
    import asyncio

    from fedquest.config import load_federation, load_remote
    from fedquest.remote.gateway import _ask

    cfg, remote = load_federation(federation), load_remote()

    async def go() -> dict[str, str] | None:
        out: dict[str, str] = {}
        asked: set[str] = set()
        for site in cfg.sites:
            facility = remote.facility_of(site.machine)
            if facility is None or facility in asked:
                continue
            asked.add(facility)
            r = await _ask(facility, "hippo_state", key=f"{cfg.name}/{site.name}", timeout_s=120.0)
            if not r.get("ok"):
                log.warning("%s: no HippoRAG 2 state from %s (%s)", federation, facility, r.get("error"))
                return None
            out.update(r.get("value") or {})
        return out

    return asyncio.run(go())


def wait_ready_remote(federation: str, timeout_s: float = 6 * 3600.0, every_s: float = 120.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while True:
        state = remote_state(federation)
        if state and all(s == "ready" for s in state.values()):
            log.info("%s: every HippoRAG 2 index is ready (%s)", federation, ", ".join(state))
            return True
        if time.monotonic() >= deadline:
            log.error("%s: HippoRAG 2 indexes not ready after %.0f s: %s", federation, timeout_s, state)
            return False
        log.info("%s: waiting for the HippoRAG 2 indexes (%s)", federation, state)
        time.sleep(every_s)


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(prog="python -m fedquest.site.hippo")
    sub = ap.add_subparsers(dest="cmd", required=True)
    st = sub.add_parser("status", help="each text collection's index state; exit 0 when all are ready")
    st.add_argument("federations", nargs="+")
    w = sub.add_parser("wait", help="wait until every text collection's index is ready (exit 1 on timeout)")
    w.add_argument("federations", nargs="+")
    w.add_argument("--timeout", type=float, default=6 * 3600.0)
    w.add_argument("--every", type=float, default=120.0)
    for p in (st, w):
        p.add_argument("--remote", action="store_true",
                       help="ask the facility that holds the index, through the deployment (the gateway), instead "
                            "of reading this machine's files")
    args = ap.parse_args(argv)
    if args.cmd == "status":
        states = {f: (remote_state(f) if args.remote else index_ready(f)) for f in args.federations}
        print(json.dumps(states, indent=1))
        return 0 if all(s and all(v == "ready" for v in s.values()) for s in states.values()) else 1
    wait = wait_ready_remote if args.remote else wait_ready
    return 0 if all(wait(f, args.timeout, args.every) for f in args.federations) else 1


if __name__ == "__main__":
    raise SystemExit(main())
