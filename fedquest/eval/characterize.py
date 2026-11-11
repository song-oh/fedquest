# workload characterization: cost profiles and the late-binding gap
from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Mapping
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from fedquest.config import load_federation
from fedquest.data import hipporag, m3sciqa, musique
from fedquest.data.mmqa import subsets
from fedquest.eval.runner import load_questions, open_federations
from fedquest.llm.client import LLMClient
from fedquest.paths import collection_dir, results_dir
from fedquest.types import ProbeRequest, Question
from fedquest.utils import read_jsonl, write_json

log = logging.getLogger(__name__)

GROUPS = ("bridged", "bridge_picture", "bridge_textual", "list", "join_all", "picture_join")
PRIMARY = "mmqa_fed"


def cost_profile(rows: list[dict[str, Any]]) -> dict[str, Any]:
    led = [r.get("ledger") or {} for r in rows]
    evidence, total = (sum(d.get(k, 0) for d in led) for k in ("evidence_tokens", "total_tokens"))
    seconds = sum(d.get("extraction_seconds", 0) + d.get("coordinator_seconds", 0) for d in led)
    yields = [len(set(r.get("support_oids") or []) & set(r["consulted"])) / len(set(r["consulted"]))
              for r in rows if r.get("consulted")]
    seen: set[str] = set()
    repeats = calls = 0
    for r in rows:
        for ref in r.get("consulted") or []:
            calls += 1
            repeats += ref in seen
            seen.add(ref)
    return {
        "questions": len(rows),
        "evidence_share": evidence / total if total else None,
        "evidence_seconds_share": sum(d.get("evidence_seconds", 0) for d in led) / seconds if seconds else None,
        "yield": sum(yields) / len(yields) if yields else None,
        "redundancy": repeats / calls if calls else None,
        "extractions_per_question": calls / len(rows) if rows else None,
    }


def chained(q: Question) -> bool:
    return musique.is_type(q.qtype) or hipporag.is_type(q.qtype) or m3sciqa.is_type(q.qtype)


def _targets(q: Question, where: Mapping[str, Any]) -> list[tuple[str, str, list[str]]]:
    groups = [g for g in GROUPS if g in subsets(q.qtype)]
    if chained(q):
        if q.meta.get("evidence_level") == "paper":
            return []
        return [(value, doc, [q.meta.get("family", ""), *groups]) for value, doc in musique.bridges(q)
                if value and doc in where]
    value = q.bridge[0] if q.bridge else q.answers[0] if q.answers else None
    if not value:
        return []
    table = q.meta.get("table_id")
    finals = [d for d in q.gold_doc_ids if d != table and d in where] or [d for d in q.gold_doc_ids if d in where]
    return [(value, doc, groups) for doc in finals[:1]]


async def late_binding_gap(federation: str, *, mode: str = "local", limit: int | None = None, k: int = 5,
                           split: str = "test", questions_from: str | None = None,
                           ks: tuple[int, ...] = ()) -> dict[str, Any]:
    depths = sorted({k, *ks})
    k = depths[-1]
    cfg = load_federation(federation)
    qs = [q for q in load_questions(questions_from or federation, split, limit)
          if (q.meta.get("family") == "compose" and q.bridge) or "ImageListQ" in q.qtype or chained(q)]
    wanted = {d for q in qs for d in q.gold_doc_ids}
    where: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for s in cfg.sites:
        for c in s.collections:
            for d in read_jsonl(collection_dir(cfg.data_name, s.name, c.name) / "objects.jsonl"):
                if d["doc_id"] in wanted:
                    where[d["doc_id"]].append((s.name, c.name, c.encoder))
    hits: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    async with AsyncExitStack() as stack:
        llm = LLMClient.from_env()
        stack.push_async_callback(llm.aclose)
        fed = await open_federations(mode, llm, stack).get(federation)
        for q in qs:
            for value, doc, groups in _targets(q, where):
                for site, coll, encoder in dict.fromkeys(where[doc]):
                    reqs = [ProbeRequest(q.question, "question", k, collections=[coll]),
                            ProbeRequest(value, "bound", k, collections=[coll]),
                            ProbeRequest(value, "title", k, collections=[coll], titles=[value])]
                    by_question, by_value, by_title = await fed.probe(reqs, sites=[site])
                    # gold's rank by question, by bound value and by title probe; 0 if not returned
                    found = tuple(next((i + 1 for i, c in enumerate(cands) if c.doc_id == doc), 0)
                                  for cands in (by_question, by_value, by_title))
                    for g in (encoder, "all", *groups):
                        hits[g].append(found)  # type: ignore[arg-type]

    def share(v: list[tuple[int, int, int]], which: int, d: int) -> float:
        return sum(1 for f in v if 0 < f[which] <= d) / len(v)

    return {g: {"n": len(v), **{f"{name}_top{d}": share(v, i, d) for d in depths
                                 for i, name in enumerate(("question", "value", "title"))}}
            for g, v in hits.items() if v}


def output_path(exp_name: str, federation: str) -> Path:
    name = "characterize.json" if federation == PRIMARY else f"characterize_{federation}.json"
    return results_dir(exp_name) / name


async def characterize(exp_name: str, *, federation: str, mode: str = "local", limit: int | None = None,
                       label_prefix: str = "mgrag", split: str = "test",
                       questions_from: str | None = None, ks: tuple[int, ...] = ()) -> dict[str, Any]:
    rows_dir = results_dir(exp_name) / "rows"
    profiles = {p.stem: cost_profile(list(read_jsonl(p))) for p in sorted(rows_dir.glob(f"{label_prefix}*.jsonl"))}
    out = {"federation": federation, "questions": questions_from or federation, "split": split,
           "cost_profiles": profiles,
           "late_binding": await late_binding_gap(federation, mode=mode, limit=limit, split=split,
                                                  questions_from=questions_from, ks=ks)}
    write_json(output_path(exp_name, federation), out)
    return out
