# KG update study: answer and neutral edits of gold objects, applied at an epoch
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import re
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from fedquest.config import load_federation
from fedquest.kg.normalize import norm_entity
from fedquest.paths import questions_path, runs_root
from fedquest.types import ObjectRecord, Question
from fedquest.utils import read_jsonl, write_jsonl

log = logging.getLogger(__name__)

BASE_SPLIT, UPDATE_SPLIT = "kg_base", "kg_update"
NEUTRAL_TEXT = " This entry was revised."
_YEAR = re.compile(r"^(1\d{3}|20\d{2})$")
_NUMBER = re.compile(r"^[\d.,]+$")


def edits_path(federation: str) -> Path:
    return runs_root() / "kg_update" / f"{federation}.edits.jsonl"


_MONTH = re.compile(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\b", re.I)


def _kind(answer: str) -> str:
    a = answer.strip()
    if _YEAR.match(a):
        return "year"
    if _MONTH.search(a) and re.search(r"\d", a):
        return "date"
    if _NUMBER.match(a):
        return "number"
    return "other" if re.search(r"\d", a) else "name"


def _compatible(a: str, b: str) -> bool:
    ka = _kind(a)
    if ka != _kind(b) or ka == "other" or norm_entity(a) == norm_entity(b):
        return False
    if ka != "name":
        return True
    wa, wb = len(a.split()), len(b.split())
    return abs(wa - wb) <= 1 and (a[:1].isupper() == b[:1].isupper())


def _closeness(a: str, b: str) -> tuple[int, int]:
    same_tail = a.split()[-1].lower() == b.split()[-1].lower() if a.split() and b.split() else False
    return (0 if same_tail else 1, abs(len(a.split()) - len(b.split())))


def _objects_by_doc(federation: str) -> dict[str, list[tuple[str, ObjectRecord]]]:
    cfg = load_federation(federation)
    out: dict[str, list[tuple[str, ObjectRecord]]] = defaultdict(list)
    for site in cfg.sites:
        for c in site.collections:
            if c.modality == "image":
                continue
            path = cfg.collection_data_dir(site.name, c.name) / "objects.jsonl"
            if not path.exists():
                continue
            for d in read_jsonl(path):
                o = ObjectRecord.from_dict(d)
                out[o.doc_id].append((site.name, o))
                if o.modality == "table":
                    out[o.oid.rsplit("/", 1)[-1]].append((site.name, o))
    return out


def _final_docs(q: Question) -> list[str]:
    hops = q.meta.get("hops") or []
    if hops:
        return [str(hops[-1]["doc_id"])]
    gold = q.meta.get("gold") or {}
    return [k.split(":", 1)[1] for k in gold.get("final", []) if ":" in k]


def _attach_gold(federation: str, questions: list[Question], split: str) -> None:
    from fedquest.config import load_federation as _load

    cfg = _load(federation)
    if cfg.dataset != "mmqa":
        try:
            from fedquest.eval.calibrate import attach_gold_objects

            attach_gold_objects(federation, questions, split)
        except Exception as e:
            log.warning("no object-level gold for %s/%s (%s): questions without hops are skipped", federation, split, e)
        return
    from fedquest.data import mmqa

    wanted = {q.qid for q in questions if not q.meta.get("gold")}
    for raw_split in dict.fromkeys([cfg.questions.split, "train", "dev"]):
        if not wanted:
            break
        try:
            raw = mmqa.load_raw(raw_split)
        except (OSError, FileNotFoundError) as e:
            log.warning("MMQA raw split %s not readable (%s)", raw_split, e)
            continue
        for q in questions:
            if q.qid in wanted and q.qid in raw:
                gold, unknown = mmqa.gold_objects(raw[q.qid])
                q.meta.update(gold=gold, gold_tables_unknown=unknown)
                wanted.discard(q.qid)
    if wanted:
        log.warning("%d of %d %s/%s questions are in no MMQA raw split: skipped", len(wanted), len(questions),
                    federation, split)


def _replace_all(text: str, old: str, new: str) -> str:
    return re.sub(re.escape(old), lambda _m: new, text, flags=re.IGNORECASE)


def _overlaps(a: str, b: str) -> bool:
    na, nb = f" {norm_entity(a)} ", f" {norm_entity(b)} "
    return na.strip() in nb or nb.strip() in na


def make(federation: str, split: str = "final_test", n: int = 120, seed: int = 13) -> dict[str, Any]:
    from fedquest.eval.runner import load_questions

    questions = load_questions(federation, split)
    if any(not (q.meta.get("hops")) for q in questions):
        _attach_gold(federation, questions, split)
    docs = _objects_by_doc(federation)
    rng = random.Random(f"{seed}:kg_update:{federation}")
    pool = [a for q in questions for a in q.answers[:1] if a.strip()]
    order = list(questions)
    rng.shuffle(order)
    from fedquest.llm import prompts
    from fedquest.site.service import SNIPPET_CHARS

    base, updated, edits = [], [], []
    edited: set[tuple[str, str]] = set()
    for q in order:
        if len(base) >= n or not q.answers:
            continue
        answer = q.answers[0].strip()
        if len(answer) < 2:
            continue
        found = {(s, o.oid): (s, o) for d in _final_docs(q) for s, o in docs.get(d, [])
                 if prompts.witnessed(o.text, answer)}
        held = list(found.values())
        refs = {(s, o.oid) for s, o in held}
        if not held or refs & edited or any(prompts.witnessed(o.title, answer) for _, o in held):
            continue
        # kept questions alternate: answer edits swap the answer, neutral ones only append text
        kind = "answer" if len(base) % 2 == 0 else "neutral"
        if kind == "neutral":
            new_texts = []
            for _, o in held:
                body = o.text.rstrip()
                body += "" if not body or body[-1] in ".!?;" else "."
                new_texts.append(body + NEUTRAL_TEXT)
            # a neutral edit must not change whether the object fits whole in a snippet
            if any((len(o.text) <= SNIPPET_CHARS) != (len(t) <= SNIPPET_CHARS) for (_, o), t in zip(held, new_texts,
                                                                                                strict=True)):
                continue
            for (site, o), text in zip(held, new_texts, strict=True):
                edits.append({"site": site, "oid": o.oid, "text": text, "qid": q.qid, "kind": kind})
            edited |= refs
            base.append(q)
            updated.append(Question.from_dict({**q.to_dict(), "meta": {**q.meta, "edit": {
                "kind": kind, "old": answer, "new": answer, "oids": [o.oid for _, o in held]}}}))
            continue
        options = sorted({b for b in pool if _compatible(answer, b) and not _overlaps(answer, b)
                          and not any(prompts.witnessed(f"{o.title} {o.text}", b) for _, o in held)})
        if not options:
            continue
        best = min(_closeness(answer, b) for b in options)
        new = rng.choice([b for b in options if _closeness(answer, b) == best])
        new_texts = [_replace_all(o.text, answer, new) for _, o in held]
        if any(prompts.witnessed(f"{o.title} {t}", answer) for (_, o), t in zip(held, new_texts, strict=True)):
            continue
        for (site, o), text in zip(held, new_texts, strict=True):
            edits.append({"site": site, "oid": o.oid, "text": text, "qid": q.qid, "kind": kind})
        edited |= refs
        base.append(q)
        updated.append(Question.from_dict({**q.to_dict(), "answers": [new],
                                           "meta": {**q.meta, "edit": {"kind": kind, "old": answer, "new": new,
                                                                       "oids": [o.oid for _, o in held]}}}))
    if len(base) < n:
        log.warning("kg_update %s: %d of the %d questions asked for qualify", federation, len(base), n)
    write_jsonl(questions_path(federation, BASE_SPLIT), (q.to_dict() for q in base))
    write_jsonl(questions_path(federation, UPDATE_SPLIT), (q.to_dict() for q in updated))
    path = edits_path(federation)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(path, edits)
    out = {"federation": federation, "questions": len(base), "edits": len(edits), "edits_file": str(path),
           "edit_kinds": {k: sum(1 for q in updated if q.meta["edit"]["kind"] == k) for k in ("answer", "neutral")},
           "kinds": {k: sum(1 for q in base if _kind(q.answers[0]) == k) for k in ("year", "date", "number", "name")}}
    log.info("kg_update: %s", out)
    return out


def derive(federation: str, source: str) -> dict[str, Any]:
    a, b = load_federation(federation), load_federation(source)
    if a.dataset != b.dataset:
        raise ValueError(f"{federation} ({a.dataset}) and {source} ({b.dataset}) are not layouts of one corpus")
    for split in (BASE_SPLIT, UPDATE_SPLIT):
        src = questions_path(source, split)
        if not src.exists():
            raise FileNotFoundError(f"{src}: make {source}'s update study first (python -m fedquest.eval.kg_update make)")
        write_jsonl(questions_path(federation, split), read_jsonl(src))
    rows = [{**e, "site": "*"} for e in read_jsonl(edits_path(source))]
    path = edits_path(federation)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(path, rows)
    out = {"federation": federation, "source": source, "edits": len(rows), "edits_file": str(path)}
    log.info("kg_update derive: %s", out)
    return out


async def apply(federation: str, epoch: str, mode: str = "remote", path: Path | None = None) -> dict[str, Any]:
    from contextlib import AsyncExitStack

    from fedquest.eval.runner import open_federations
    from fedquest.llm.client import LLMClient

    rows = list(read_jsonl(path or edits_path(federation)))
    by_site: dict[str, list[dict[str, str]]] = defaultdict(list)
    anywhere = [{"oid": e["oid"], "text": e["text"]} for e in rows if e["site"] == "*"]
    for e in rows:
        if e["site"] != "*":
            by_site[e["site"]].append({"oid": e["oid"], "text": e["text"]})
    async with AsyncExitStack() as stack:
        llm = LLMClient.from_env()
        stack.push_async_callback(llm.aclose)
        fed = await open_federations(mode, llm, stack).get(federation)
        for s in fed.site_names:
            by_site.setdefault(s, []).extend(anywhere)
        done = await fed.update(epoch, dict(by_site))
    out = {"federation": federation, "epoch": epoch, "edits": len(rows), "updated": len(done)}
    log.info("kg_update apply: %s", out)
    if len(done) != len(rows):
        raise RuntimeError(f"{len(rows) - len(done)} of {len(rows)} edits reached no site that holds their object")
    return out


async def fact_stats(namespaces: Sequence[str] | None = None) -> dict[str, Any]:
    import yaml

    from fedquest.kg.view import open_view
    from fedquest.paths import CONFIGS

    if not namespaces:
        namespaces = []
        for path in sorted((CONFIGS / "experiments").glob("*.yaml")):
            exp = yaml.safe_load(path.read_text())
            for m in exp.get("methods") or []:
                params = m.get("params") or {}
                if params.get("kg", "off") != "off" or params.get("cache", "off") != "off":
                    namespaces.append(params.get("kg_ns") or f"{exp['name']}/{m.get('label') or m['method']}")
    kg = open_view("neo4j")
    out: dict[str, Any] = {}
    try:
        for ns in dict.fromkeys(namespaces):
            out[ns] = await kg.fact_stats(ns)
    finally:
        await kg.close()
    by_exp: dict[str, dict[str, Any]] = defaultdict(dict)
    for ns, stats in out.items():
        by_exp[ns.split("/", 1)[0]][ns] = stats
    for exp, stats in by_exp.items():
        path = runs_root() / "results" / exp / "fact_stats.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(stats, indent=1, sort_keys=True))
    return out


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(prog="python -m fedquest.eval.kg_update")
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--federation", required=True)
    m.add_argument("--split", default="final_test")
    m.add_argument("--n", type=int, default=120)
    m.add_argument("--seed", type=int, default=13)
    a = sub.add_parser("apply")
    a.add_argument("--federation", required=True)
    a.add_argument("--epoch", required=True)
    a.add_argument("--mode", default="remote")
    a.add_argument("--edits", default=None)
    d = sub.add_parser("derive", help="carry --source's update study to --federation, another layout of its corpus")
    d.add_argument("--federation", required=True)
    d.add_argument("--source", required=True)
    st = sub.add_parser("stats", help="the federated KG's fact statistics per namespace, into results/<exp>/")
    st.add_argument("namespaces", nargs="*")
    args = ap.parse_args(argv)
    if args.cmd == "make":
        print(json.dumps(make(args.federation, args.split, args.n, args.seed)))
    elif args.cmd == "derive":
        print(json.dumps(derive(args.federation, args.source)))
    elif args.cmd == "stats":
        print(json.dumps(asyncio.run(fact_stats(args.namespaces)), indent=1, default=str))
    else:
        print(json.dumps(asyncio.run(apply(args.federation, args.epoch, args.mode,
                                           Path(args.edits) if args.edits else None))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
