# scoring of run rows: answer metrics, judge accuracy, set scores and summaries
from __future__ import annotations

import csv
import logging
import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from fedquest.data import hipporag, m3sciqa, msmarco, musique
from fedquest.data.mmqa import FAMILIES as MMQA_FAMILIES
from fedquest.data.mmqa import family as mmqa_family
from fedquest.data.mmqa import subsets
from fedquest.eval.metrics import bleu1, exact_match, f1, judge_accuracy, normalize_answer, rouge_l, sim
from fedquest.kg.normalize import same_entity
from fedquest.llm.client import LLMClient
from fedquest.paths import results_dir
from fedquest.utils import gather_limited, read_jsonl, write_jsonl
from fedquest.vdb.encoders import load_encoder

log = logging.getLogger(__name__)

FAMILIES = ("single", "table", "compose", "intersect", "compare", "mramg", *musique.FAMILIES,
            *dict.fromkeys(t for ts in hipporag.TYPES.values() for t in ts),
            *msmarco.TYPES, *m3sciqa.TYPES)
LEDGER_COLUMNS = ("evidence_tokens", "evidence_seconds", "evidence_calls", "extraction_tokens", "extraction_seconds",
                  "objects_read", "read_calls", "coordinator_tokens", "coordinator_seconds", "coordinator_tokens_bind",
                  "coordinator_tokens_bind_index", "coordinator_tokens_verify", "coordinator_tokens_planner",
                  "total_tokens", "plan_seconds",
                  "rounds", "reused_objects", "bytes_raw", "bytes_summary", "bytes_facts", "critical_seconds",
                  "reembed_seconds", "evidence_in", "evidence_out", "total_in", "total_out", "bytes_control", "probes",
                  "bytes_inference_cross", "bytes_inference_coordinator", "question_sites",
                  "messages", "bytes_reply", "fanout_max_sum", "fanout_med_sum", "deadline_misses", "wan_seconds",
                  "coord_replays", "live_seconds", "llm_wall_seconds", "llm_retries",
                  "llm_backoff_seconds", "llm_429", "llm_5xx", "prefix_cached_ideal", "values_cross")
SUBSETS = ("bridged", "bridge_picture", "bridge_textual", "list", "join_all", "picture_join", "multihop")
SITE_FLOW = ("bytes_inference_", "disclosed_keys_", "disclosed_values_")
FLOW_TOTALS = ("bytes_inference_cross", "bytes_inference_coordinator")
CLOSED_POOL = "bu2025"
LIST_SEP = re.compile(r"\s*(?:[;\n|•]|,\s+(?:and\s+|&\s+)?|\s+(?:and|&)\s+)\s*", re.IGNORECASE)
LIST_MARK = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
SET_FIELDS = ("set_precision", "set_recall", "set_f1")


def list_golds(row: dict[str, Any]) -> list[str] | None:
    if mmqa_family(row.get("qtype") or "") not in MMQA_FAMILIES:
        return None
    golds = list(dict.fromkeys(g for g in row.get("golds") or [] if g))
    return golds if len({normalize_answer(g) for g in golds} - {""}) > 1 else None


def answer_items(pred: str, golds: Sequence[str] = ()) -> list[str]:
    text, kept = pred or "", {}
    # golds that contain a separator are masked first so the split keeps them whole
    for i, g in enumerate(sorted({g.strip() for g in golds if LIST_SEP.search(g.strip())}, key=len, reverse=True)):
        text, n = re.subn(re.escape(g), f"\x00{i}\x00", text, flags=re.IGNORECASE)
        if n:
            kept[f"\x00{i}\x00"] = g
    out: list[str] = []
    for part in LIST_SEP.split(text):
        for mark, g in kept.items():
            part = part.replace(mark, g)
        item = normalize_answer(LIST_MARK.sub("", part))
        if item and item not in out:
            out.append(item)
    return out


def set_scores(pred: str, golds: Sequence[str]) -> dict[str, float]:
    left = list(dict.fromkeys(x for x in map(normalize_answer, golds) if x))
    n_gold, items, hits = len(left), answer_items(pred, golds), 0
    for item in items:
        match = item if item in left else next((g for g in left if f" {g} " in f" {item} "), None)
        if match is not None:
            left.remove(match)
            hits += 1
    p = hits / len(items) if items else 0.0
    r = hits / n_gold if n_gold else 0.0
    return {"set_precision": p, "set_recall": r, "set_f1": 2 * p * r / (p + r) if p + r else 0.0}


def flow_recorded(ledger: dict[str, Any]) -> bool:
    return "bytes_inference_coordinator" in ledger or "question_sites" in ledger


def site_flow_keys(ledgers: Iterable[dict[str, Any]]) -> list[str]:
    found = {k for led in ledgers for k in led if k.startswith(SITE_FLOW) and k not in FLOW_TOTALS}
    return sorted(found, key=lambda k: (next(i for i, p in enumerate(SITE_FLOW) if k.startswith(p)), k))


RECALL_AT = (1, 2, 5)


def evidence_metrics(row: dict[str, Any]) -> dict[str, Any]:
    gold = set(row.get("gold_doc_ids") or [])
    order = list(dict.fromkeys(row.get("consulted_doc_ids") or []))
    found = set(order)
    bridge = row.get("bridge") or []
    values = list((row.get("bindings") or {}).values())
    out = {
        "gold_recall": len(gold & found) / len(gold) if gold else None,
        "all_gold_found": float(gold <= found) if gold else None,
        "bridge_bound": float(any(same_entity(b, v) or b.lower() == v.lower() for b in bridge for v in values))
        if bridge else None,
    }
    for k in RECALL_AT:
        out[f"recall_at_{k}"] = len(gold & set(order[:k])) / len(gold) if gold else None
    return out


async def score_rows(rows: list[dict[str, Any]], *, llm: LLMClient | None, encoder: Any,
                     concurrency: int = 16, judge2: bool = False) -> list[dict[str, Any]]:
    async def one(row: dict[str, Any]) -> dict[str, Any]:
        pred, golds = row.get("answer") or "", [g for g in row.get("golds") or [] if g]
        question = row.get("question", "")
        scores = {
            "em": exact_match(pred, golds),
            "f1": f1(pred, golds),
            "rouge_l": rouge_l(pred, golds),
            "bleu1": bleu1(pred, golds),
            "sim": sim(pred, golds, encoder),
            "acc": await judge_accuracy(question, pred, golds, llm) if llm else None,
        }
        if judge2 and llm:
            scores["acc_j2"] = await judge_accuracy(question, pred, golds, llm, role="judge2")
        listed = list_golds(row)
        scores["list_answer"] = listed is not None
        if listed is None:
            scores |= {"acc_set": scores["acc"], **dict.fromkeys(SET_FIELDS)}
            if "acc_j2" in scores:
                scores["acc_set_j2"] = scores["acc_j2"]
        else:
            joined = ["; ".join(listed)]
            scores |= set_scores(pred, listed)
            scores["acc_set"] = await judge_accuracy(question, pred, joined, llm) if llm else None
            if judge2 and llm:
                scores["acc_set_j2"] = await judge_accuracy(question, pred, joined, llm, role="judge2")
        return {**row, **scores, **evidence_metrics(row)}

    return await gather_limited(rows, one, concurrency)


def _keep_second_judge(scored: list[dict[str, Any]], before: Path) -> list[dict[str, Any]]:
    if not before.exists():
        return scored
    kept = {(r["qid"], r.get("agent", 0)): r for r in read_jsonl(before) if r.get("acc_j2") is not None}
    same = ("question", "answer", "golds")
    for r in scored:
        old = kept.get((r["qid"], r.get("agent", 0)))
        if old is not None and "acc_j2" not in r and all(old.get(k) == r.get(k) for k in same):
            r["acc_j2"] = old["acc_j2"]
            set_j2 = old.get("acc_set_j2", None if r.get("list_answer") else old["acc_j2"])
            if set_j2 is not None:
                r["acc_set_j2"] = set_j2
    return scored


def latest_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return list({(r["qid"], r.get("agent", 0)): r for r in rows}.values())


def _mean(xs: list[Any]) -> float | None:
    vals = [x for x in xs if x is not None]
    return float(np.mean(vals)) if vals else None


def _led(row: dict[str, Any], key: str) -> Any:
    return (row.get("ledger") or {}).get(key, 0 if key.startswith("coordinator_tokens_") else None)


def summarize(label: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    walls = [_led(r, "wall_seconds") or 0.0 for r in rows]
    out: dict[str, Any] = {"label": label, "n": len(rows), "errors": sum(1 for r in rows if r.get("error")),
                           "budget": next((r["budget"] for r in rows if r.get("budget") is not None), None),
                           "tokens": next((r["tokens"] for r in rows if r.get("tokens") is not None), None)}
    for m in ("acc", "acc_j2", "em", "f1", "rouge_l", "bleu1", "sim", "gold_recall", "all_gold_found", "bridge_bound",
              *(f"recall_at_{k}" for k in RECALL_AT), "acc_set", "acc_set_j2", *SET_FIELDS):
        out[m] = _mean([r.get(m) for r in rows])
    out["n_list"] = sum(1 for r in rows if r.get("list_answer"))
    for fam in FAMILIES:
        out[f"acc_{fam}"] = _mean([r.get("acc") for r in rows if r.get("family") == fam])
        out[f"em_{fam}"] = _mean([r.get("em") for r in rows if r.get("family") == fam])
        out[f"f1_{fam}"] = _mean([r.get("f1") for r in rows if r.get("family") == fam])
    for sub in SUBSETS:
        out[f"acc_{sub}"] = _mean([r.get("acc") for r in rows if sub in subsets(r.get("qtype") or "")])
    out["acc_image_answer"] = _mean([r.get("acc") for r in rows if r.get("image_answer")])
    for key in LEDGER_COLUMNS:
        out[key] = _mean([_led(r, key) for r in rows])
    for key in site_flow_keys(r.get("ledger") or {} for r in rows):
        out[key] = _mean([(r.get("ledger") or {}).get(key, 0) for r in rows if flow_recorded(r.get("ledger") or {})])
    out["wall_p50"] = float(np.percentile(walls, 50)) if walls else None
    out["wall_p95"] = float(np.percentile(walls, 95)) if walls else None
    out["evidence_tokens_total"] = float(sum(_led(r, "evidence_tokens") or 0 for r in rows))
    for what in ("evidence", "total"):
        ins, outs = out.get(f"{what}_in"), out.get(f"{what}_out")
        out[f"{what}_tokens_1x"] = ins + outs if ins is not None and outs is not None else None
    out["model_seconds"] = _mean([model_seconds(r.get("ledger") or {}) for r in rows])
    return out


def model_seconds(ledger: dict[str, Any]) -> float | None:
    ext, coord = ledger.get("extraction_seconds"), ledger.get("coordinator_seconds")
    return None if ext is None and coord is None else float((ext or 0) + (coord or 0))


def by_qtype(label: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[r.get("qtype") or ""].append(r)
    return [{"label": label, "qtype": t, "n": len(rs), "acc": _mean([r.get("acc") for r in rs]),
             "evidence_tokens": _mean([_led(r, "evidence_tokens") for r in rs]),
             "evidence_seconds": _mean([_led(r, "evidence_seconds") for r in rs])}
            for t, rs in sorted(groups.items())]


def curves(label: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not any("index" in r for r in rows):
        return []
    ordered = sorted(rows, key=lambda r: (r.get("finished_at") or 0, r.get("agent", 0), r.get("index", 0)))
    evidence, out = 0.0, []
    for i, r in enumerate(ordered, 1):
        evidence += _led(r, "evidence_tokens") or 0
        out.append({"label": label, "queries": i, "cumulative_evidence_tokens": evidence})
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


async def score_experiment(name: str, *, judge: bool = True, sim_encoder: str = "bge-base",
                           labels: list[str] | None = None, judge2: bool = False) -> Path:
    out = results_dir(name)
    llm = LLMClient.from_env() if judge else None
    if judge2 and llm is not None and "judge2" not in llm.cfg.roles:
        await llm.aclose()
        raise ValueError("the models config defines no judge2 role: add one (e.g. judge2: <model>) to score with a "
                         "second judge")
    encoder = load_encoder(sim_encoder)
    summary, qtypes, all_curves = [], [], []
    try:
        for path in sorted((out / "rows").glob("*.jsonl")):
            label, scored_path = path.stem, out / "rows_scored" / path.name
            if labels and not ({label, label.split("@")[0], label.split("+")[0]} & set(labels)):
                if not scored_path.exists():
                    continue
                scored = list(read_jsonl(scored_path))
            else:
                second = judge2 and llm is not None
                scored = await score_rows(latest_rows(read_jsonl(path)), llm=llm, encoder=encoder, judge2=second)
                if not second:
                    scored = _keep_second_judge(scored, scored_path)
                write_jsonl(scored_path, scored)
                log.info("%s: scored %d rows of %s", name, len(scored), label)
            summary.append(summarize(label, scored))
            qtypes += by_qtype(label, scored)
            all_curves += curves(label, scored)
    finally:
        if llm is not None:
            await llm.aclose()
    _write_csv(out / "summary.csv", summary)
    _write_csv(out / "by_qtype.csv", qtypes)
    _write_csv(out / "curves.csv", all_curves)
    return out / "summary.csv"
