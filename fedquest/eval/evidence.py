# evidence funnel of gold objects: returned, read, in context, by hop and path
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from fedquest.paths import runs_root

STAGES = ("returned", "read", "in_context")


def _read(e: dict[str, Any]) -> bool:
    return bool(e.get("read_site") or e.get("read_coordinator"))


def _facility(site: str) -> str:
    return site.split("_", 1)[0] if site else "?"


def gold_records(row: dict[str, Any]) -> list[dict[str, Any]]:
    led = row.get("ledger") or {}
    log = led.get("evidence_log") or {}
    golds = row.get("gold_doc_ids") or []
    consulted = set(row.get("consulted_doc_ids") or [])
    logged = bool(led.get("context_logged"))
    out = []
    for d in golds:
        e = log.get(d, {})
        read = _read(e) or d in consulted
        # rows without a context log count every consulted gold as in context
        ctx = bool(e.get("in_context")) or (not logged and d in consulted)
        out.append({"doc": d, "hop": int(e.get("hop", 0)), "later": bool(e.get("later", False)),
                    "returned": bool(e.get("returned")) or read or ctx,
                    "read": read, "in_context": ctx,
                    "kind": ("semijoin" if e.get("semijoin") else e.get("kind")) or "none",
                    "site": e.get("site", ""), "facility": _facility(e.get("site", "")),
                    "modality": e.get("modality", ""), "rank": e.get("best_rank"), "z": e.get("z"),
                    "round": e.get("first_round")})
    return out


def _acc(row: dict[str, Any]) -> float | None:
    a = row.get("acc")
    return None if a is None else float(a)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [r for r in rows if not r.get("error")]
    recs = [(r, g) for r in rows for g in gold_records(r)]
    if not rows:
        return {"n": 0}

    def rate(items: Iterable[dict[str, Any]], stage: str) -> float | None:
        items = list(items)
        return round(sum(g[stage] for g in items) / len(items), 4) if items else None

    out: dict[str, Any] = {"n": len(rows), "objects": len(recs)}
    golds = [g for _, g in recs]
    out["funnel_objects"] = {s: rate(golds, s) for s in STAGES}
    q_stage = {s: [] for s in STAGES}
    for r in rows:
        gs = gold_records(r)
        for s in STAGES:
            q_stage[s].append(bool(gs) and all(g[s] for g in gs))
    out["funnel_questions"] = {s: round(float(np.mean(v)), 4) if v else None for s, v in q_stage.items()}
    accs = [a for r in rows if (a := _acc(r)) is not None]
    out["acc"] = round(float(np.mean(accs)), 4) if accs else None
    by_hop: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for g in golds:
        by_hop[f"hop{g['hop'] + 1}"].append(g)
        by_hop["later" if g["later"] else "first"].append(g)
    out["by_hop"] = {k: {"n": len(v), **{s: rate(v, s) for s in STAGES}} for k, v in sorted(by_hop.items())}
    for name, key in (("by_path", "kind"), ("by_facility", "facility"), ("by_modality", "modality")):
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for g in golds:
            groups[g[key] or "?"].append(g)
        out[name] = {k: {"n": len(v), **{s: rate(v, s) for s in STAGES}} for k, v in sorted(groups.items())}
    later_q = [(r, gold_records(r)) for r in rows if any(g["later"] for g in gold_records(r))]
    reached = [(r, gs) for r, gs in later_q if all(g["read"] or g["in_context"] for g in gs if g["later"])]
    unreached = [(r, gs) for r, gs in later_q if (r, gs) not in reached]
    mean = (lambda xs: round(float(np.mean(xs)), 4) if xs else None)  # noqa: E731
    out["later_hop_questions"] = {"n": len(later_q), "reach": mean([1.0] * len(reached) + [0.0] * len(unreached)),
                                  "acc_reached": mean([a for r, _ in reached if (a := _acc(r)) is not None]),
                                  "acc_unreached": mean([a for r, _ in unreached if (a := _acc(r)) is not None])}
    attrib: dict[str, int] = defaultdict(int)
    for r in rows:
        a = _acc(r)
        if a is None or a >= 0.5:
            continue
        gs = gold_records(r)
        if not gs:
            attrib["no ground truth"] += 1
        elif not all(g["returned"] for g in gs):
            attrib["not returned"] += 1
        elif not all(g["read"] or g["in_context"] for g in gs):
            attrib["returned, not read"] += 1
        elif not all(g["in_context"] for g in gs):
            attrib["read, not in context"] += 1
        else:
            attrib["in context, answer wrong"] += 1
    out["errors"] = dict(sorted(attrib.items()))
    tok = [float((r.get("ledger") or {}).get("total_tokens", 0)) for r in rows]
    gold_read = [sum(g["read"] for g in gold_records(r)) for r in rows]
    read_n = [len(set(r.get("consulted_doc_ids") or [])) for r in rows]
    ret = [int((r.get("ledger") or {}).get("returned_total", 0)) for r in rows]
    out["efficiency"] = {
        "tokens_mean": round(float(np.mean(tok)), 1),
        "tokens_per_gold_read": round(sum(tok) / max(1, sum(gold_read)), 1),
        "read_precision": round(sum(gold_read) / max(1, sum(read_n)), 4),
        "returned_mean": round(float(np.mean(ret)), 1),
        "messages_mean": round(float(np.mean([(r.get("ledger") or {}).get("messages", 0) for r in rows])), 2),
        "rounds_mean": round(float(np.mean([(r.get("ledger") or {}).get("rounds", 0) for r in rows])), 2),
    }
    return out


def paired(a: list[dict[str, Any]], b: list[dict[str, Any]], what: str, n: int = 2000, seed: int = 0
           ) -> dict[str, Any] | None:
    def val(r: dict[str, Any]) -> float | None:
        if what == "acc":
            return _acc(r)
        gs = gold_records(r)
        if not gs:
            return None
        if what == "reach":
            lg = [g for g in gs if g["later"]]
            return None if not lg else float(all(g["read"] or g["in_context"] for g in lg))
        if what == "read_all":
            return float(all(g["read"] for g in gs))
        if what == "returned_all":
            return float(all(g["returned"] for g in gs))
        raise ValueError(what)

    va = {r["qid"]: v for r in a if not r.get("error") and (v := val(r)) is not None}
    vb = {r["qid"]: v for r in b if not r.get("error") and (v := val(r)) is not None}
    keys = sorted(set(va) & set(vb))
    if not keys:
        return None
    d = np.asarray([va[k] - vb[k] for k in keys])
    rng = np.random.default_rng(seed)
    boots = [float(d[rng.integers(0, len(d), len(d))].mean()) for _ in range(n)]
    return {"n": len(keys), "diff": round(float(d.mean()), 4), "lo": round(float(np.percentile(boots, 2.5)), 4),
            "hi": round(float(np.percentile(boots, 97.5)), 4)}


def load_rows(exp: str, label: str, runs: Path | None = None) -> list[dict[str, Any]]:
    base = (runs or runs_root()) / "results" / exp
    for sub in ("rows_scored", "rows"):
        p = base / sub / f"{label}.jsonl"
        if p.exists():
            latest: dict[str, dict[str, Any]] = {}
            for line in p.open(encoding="utf-8"):
                if line.strip():
                    r = json.loads(line)
                    latest[r["qid"]] = r
            return list(latest.values())
    return []


def labels_of(exp: str, runs: Path | None = None) -> list[str]:
    base = (runs or runs_root()) / "results" / exp
    sub = base / "rows_scored" if (base / "rows_scored").exists() else base / "rows"
    return sorted(p.stem for p in sub.glob("*.jsonl")) if sub.exists() else []


def report(experiments: list[str], out: Path, runs: Path | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {}
    lines = ["# Evidence analysis (fedquest.eval.evidence)", ""]
    for exp in experiments:
        result[exp] = {}
        lines += [f"## {exp}", "", "| label | n | acc | returned | read | in context | Q: all read | later reach | "
                  "acc reached | acc unreached | tokens | tok/gold read | read precision | returned/q |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for lab in labels_of(exp, runs):
            s = summarize(load_rows(exp, lab, runs))
            result[exp][lab] = s
            if not s.get("n"):
                continue
            f, fq, lh, ef = s["funnel_objects"], s["funnel_questions"], s["later_hop_questions"], s["efficiency"]
            fmt = (lambda v: "" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v)))  # noqa: E731, B023
            lines.append("| " + " | ".join([lab, str(s["n"]), fmt(s["acc"]), fmt(f["returned"]), fmt(f["read"]),
                                             fmt(f["in_context"]), fmt(fq["read"]), fmt(lh["reach"]),
                                             fmt(lh["acc_reached"]), fmt(lh["acc_unreached"]),
                                             fmt(ef["tokens_mean"]), fmt(ef["tokens_per_gold_read"]),
                                             fmt(ef["read_precision"]), fmt(ef["returned_mean"])]) + " |")
        lines.append("")
        for lab, s in result[exp].items():
            if not s.get("n"):
                continue
            lines += [f"### {exp} / {lab}: by hop, access path and facility", "",
                      "| group | n | returned | read | in context |", "|---|---|---|---|---|"]
            for name in ("by_hop", "by_path", "by_facility", "by_modality"):
                for k, v in s[name].items():
                    lines.append(f"| {name[3:]}={k} | {v['n']} | {v['returned']} | {v['read']} | {v['in_context']} |")
            lines += ["", f"errors by first missing stage: {s['errors']}", ""]
    out.mkdir(parents=True, exist_ok=True)
    (out / "evidence.json").write_text(json.dumps(result, indent=1))
    (out / "evidence.md").write_text("\n".join(lines) + "\n")
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m fedquest.eval.evidence")
    ap.add_argument("experiments", nargs="*", help="experiments (default: every main_, frontier_, abl_, scale_)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--compare", nargs=4, metavar=("EXP", "LABEL_A", "LABEL_B", "WHAT"), action="append",
                    help="paired a - b of acc | reach | read_all | returned_all")
    args = ap.parse_args(argv)
    runs = runs_root()
    exps = args.experiments or sorted(p.name for p in (runs / "results").glob("*")
                                      if p.is_dir() and p.name.split("_")[0] in ("main", "frontier", "abl", "scale",
                                                                                 "route", "stream"))
    out = args.out or runs / "results" / "evidence"
    report(exps, out, runs)
    for exp, a, b, what in args.compare or []:
        print(exp, a, "-", b, what, paired(load_rows(exp, a, runs), load_rows(exp, b, runs), what))
    print(f"evidence analysis of {len(exps)} experiments: {out}/evidence.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
