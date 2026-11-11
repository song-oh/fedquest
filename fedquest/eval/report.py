from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from fedquest.eval.evidence import load_rows, summarize
from fedquest.eval.stats import holm, paired_bootstrap, paired_permutation
from fedquest.paths import runs_root

BENCHES = ("mmqa", "musique", "twowiki", "hotpot")
TITLE = {"mmqa": "MMQA", "musique": "MuSiQue", "twowiki": "2Wiki", "hotpot": "HotpotQA"}
BUDGETS = (4000, 8000, 16000, 32000)
MAIN_B = 16000


def _labels(exp: str, runs: Path) -> list[str]:
    d = runs / "results" / exp
    sub = d / "rows_scored" if (d / "rows_scored").exists() else d / "rows"
    return sorted(p.stem for p in sub.glob("*.jsonl")) if sub.exists() else []


def _val(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    out = {}
    for r in rows:
        if r.get("error"):
            continue
        v = r.get(key) if key in r else (r.get("ledger") or {}).get(key)
        if v is not None:
            out[r["qid"]] = float(v)
    return out


def _mean(d: dict[str, float]) -> float | None:
    return float(np.mean(list(d.values()))) if d else None


def _fmt(v: Any, nd: int = 3) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def row_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in rows if not r.get("error")]
    led = [r.get("ledger") or {} for r in ok]
    tok_in = [float(x.get("total_in", 0)) for x in led]
    tok_out = [float(x.get("total_out", 0)) for x in led]
    wall = [float(x.get("wall_seconds", 0)) for x in led]
    budget_left = [x.get("budget_left") for x in led if x.get("budget_left") is not None]
    return {"n": len(ok), "errors": len(rows) - len(ok), "acc": _mean(_val(ok, "acc")),
            "acc_j2": _mean(_val(ok, "acc_j2")), "em": _mean(_val(ok, "em")), "f1": _mean(_val(ok, "f1")),
            "r1": _mean(_val(ok, "recall_at_1")), "r2": _mean(_val(ok, "recall_at_2")),
            "r5": _mean(_val(ok, "recall_at_5")),
            "cost": float(np.mean([float(x.get("total_tokens", 0)) for x in led])) if led else None,
            "tokens": float(np.mean(np.add(tok_in, tok_out))) if led else None,
            "calls": float(np.mean([float(x.get("coordinator_calls", 0)) + float(x.get("read_calls", 0))
                                    for x in led])) if led else None,
            "messages": float(np.mean([float(x.get("messages", 0)) for x in led])) if led else None,
            "returned": float(np.mean([float(x.get("returned_total", 0)) for x in led])) if led else None,
            "rounds": float(np.mean([float(x.get("rounds", 0)) for x in led])) if led else None,
            "kb": float(np.mean([(float(x.get("bytes_control", 0)) + float(x.get("bytes_reply", 0))) / 1024
                                 for x in led])) if led else None,
            "wall_p50": float(np.percentile(wall, 50)) if wall else None,
            "wall_p95": float(np.percentile(wall, 95)) if wall else None,
            "overrun_max": (-min(budget_left) if budget_left and min(budget_left) < 0 else 0) if budget_left else None}


def lead(a: list[dict[str, Any]], b: list[dict[str, Any]], qids: set[str] | None = None,
         key: str = "acc") -> dict[str, Any] | None:
    va, vb = _val(a, key), _val(b, key)
    if qids is not None:
        va = {k: v for k, v in va.items() if k in qids}
        vb = {k: v for k, v in vb.items() if k in qids}
    bs = paired_bootstrap(va, vb)
    if bs is None:
        return None
    return {"n": len(set(va) & set(vb)), "diff": bs[0], "lo": bs[1], "hi": bs[2], "p": paired_permutation(va, vb)}


def held_out(b: str) -> set[str] | None:
    from fedquest.eval.runner import load_questions

    fed = {"mmqa": "mmqa_s4_k05_r0", "musique": "musique_s4_k05_r0", "twowiki": "twowiki_s4_k05_r0",
           "hotpot": "hotpot_s4_k05_r0"}[b]
    try:
        return {q.qid for q in load_questions(fed, "heldout")}
    except (FileNotFoundError, OSError):
        return None


def section_heldout(runs: Path, out: dict[str, Any]) -> list[str]:
    lines = ["## 1h. Main comparison on the held-out questions (no earlier run touched them: configs/exposed.json; "
             "MMQA, 2Wiki and HotpotQA: main's rows of their held-out splits; MuSiQue: heldout_musique, the "
             "MuSiQue-Ans dev questions outside HippoRAG 2's release whose evidence is in its corpus)", ""]
    for b in BENCHES:
        if b == "musique":
            exp, qids = "heldout_musique", None
            extra = {lab: r for lab, r in (("hipporag2_pooled", load_rows("hippo_heldout_musique", "hipporag2_pooled",
                                                                          runs)),) if r}
        else:
            exp, qids, extra = f"main_{b}", held_out(b), None
            if qids is None:
                lines += [f"_{TITLE[b]}: no held-out split on this machine (scripts/make_pools.py)._", ""]
                continue
            extra = {lab: [r for r in rows if r["qid"] in qids] for lab, rows in _beside_main(b, runs).items()}
        lines += _main_table(b, exp, runs, out.setdefault("heldout", {}), extra, qids,
                             f"### {TITLE[b]} (held out)")
    return lines


def section_main(runs: Path, out: dict[str, Any]) -> list[str]:
    lines = ["## 1. Main comparison (16K cost units for every model call of FedQuest, IRCoT, ReAct, ReWOO, MGRAG+FU "
             "and DGRAG (dgrag_<bench>, the same questions); the one-shot baselines and IRCoT 8x8 at their own cost; "
             "HippoRAG 2 over the pooled corpus, no federation, where it ran: hippo_<bench>, the same questions)", ""]
    for b in BENCHES:
        lines += _main_table(b, f"main_{b}", runs, out.setdefault("main", {}), _beside_main(b, runs), None,
                             f"### {TITLE[b]}")
    return lines


def _main_table(b: str, exp: str, runs: Path, out: dict[str, Any], extra: dict[str, list[dict[str, Any]]] | None,
                qids: set[str] | None, title: str) -> list[str]:
    lines: list[str] = []
    labs = _labels(exp, runs)
    if not labs:
        return [f"_{exp}: no rows yet._", ""]
    rows = {lab: [r for r in load_rows(exp, lab, runs) if qids is None or r["qid"] in qids] for lab in labs}
    for lab, more in (extra or {}).items():
        rows[lab] = more
        labs = [*labs, lab]
    stats = {lab: row_stats(r) for lab, r in rows.items()}
    ref = "fedquest"
    leads = {lab: lead(rows[ref], rows[lab]) for lab in labs if lab != ref and ref in rows}
    leads_f1 = {lab: lead(rows[ref], rows[lab], key="f1") for lab in labs if lab != ref and ref in rows}
    adj = holm({k: (v or {}).get("p") for k, v in leads.items()})
    adj_f1 = holm({k: (v or {}).get("p") for k, v in leads_f1.items()})
    out[b] = {"stats": stats, "leads": leads, "holm": adj, "leads_f1": leads_f1, "holm_f1": adj_f1}
    lines += [title, "", "| label | n | EM | F1 | ACC | ACC (judge 2) | R@1 | R@2 | R@5 | cost/q | "
              "tokens/q | calls/q | KB/q | FedQuest F1 lead (95% CI) | p (Holm) | FedQuest ACC lead | p (Holm) | "
              "overrun max |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]

    def ci(ld: dict[str, Any] | None) -> str:
        return "" if ld is None else f"{100 * ld['diff']:+.1f} [{100 * ld['lo']:+.1f}, {100 * ld['hi']:+.1f}]"

    for lab in sorted(labs, key=lambda x: -(stats[x]["f1"] or 0)):
        s = stats[lab]
        lines.append(f"| {lab} | {s['n']} | {_fmt(s['em'])} | {_fmt(s['f1'])} | {_fmt(s['acc'])} | "
                     f"{_fmt(s['acc_j2'])} | {_fmt(s['r1'])} | {_fmt(s['r2'])} | {_fmt(s['r5'])} | "
                     f"{_fmt(s['cost'], 0)} | "
                     f"{_fmt(s['tokens'], 0)} | {_fmt(s['calls'], 1)} | {_fmt(s['kb'], 1)} | {ci(leads_f1.get(lab))} | "
                     f"{_fmt(adj_f1.get(lab), 4)} | {ci(leads.get(lab))} | {_fmt(adj.get(lab), 4)} | "
                     f"{_fmt(s['overrun_max'], 0)} |")
    lines.append("")
    return lines


def qgroup(bench: str, qtype: str) -> str:
    if bench != "mmqa":
        return qtype or "?"
    mods = "+".join(m for m, tag in (("image", "Image"), ("table", "Table"), ("text", "Text")) if tag in qtype)
    hops = "multi-hop" if any(op in qtype for op in ("Compose", "Compare", "Intersect")) else "single-hop"
    return f"{mods or '?'}, {hops}"


def section_types(runs: Path, out: dict[str, Any]) -> list[str]:
    lines = ["## 1b. Main by question type (MMQA: the modalities a question's hops use; the others: the benchmark's "
             "own types): ACC per label, n questions per group, FedQuest's lead over the best other label", ""]
    for b in BENCHES:
        exp = f"main_{b}"
        labs = _labels(exp, runs)
        if "fedquest" not in labs:
            continue
        rows = {lab: {r["qid"]: r for r in load_rows(exp, lab, runs) if not r.get("error")} for lab in labs}
        for lab, extra in _beside_main(b, runs).items():
            rows[lab] = {r["qid"]: r for r in extra if not r.get("error")}
            labs = [*labs, lab]
        groups: dict[str, set[str]] = defaultdict(set)
        for q, r in rows["fedquest"].items():
            groups[qgroup(b, str(r.get("qtype") or ""))].add(q)
        order = ["fedquest", *sorted(lab for lab in labs if lab != "fedquest")]
        lines += [f"### {TITLE[b]}", "", "| group | n | " + " | ".join(order) + " | FedQuest - best other |",
                  "|---|---|" + "---|" * len(order) + "---|"]
        res: dict[str, dict[str, Any]] = {}
        for g in sorted(groups, key=lambda g: (-len(groups[g]), g)):
            qs = groups[g]
            acc = {}
            for lab in order:
                v = [float(rows[lab][q]["acc"]) for q in qs if q in rows[lab] and rows[lab][q].get("acc") is not None]
                acc[lab] = float(np.mean(v)) if v else None
            others = [v for lab, v in acc.items() if lab != "fedquest" and v is not None]
            gap = (acc["fedquest"] - max(others)) if others and acc["fedquest"] is not None else None
            res[g] = {"n": len(qs), "acc": acc, "lead_over_best": gap}
            lines.append(f"| {g} | {len(qs)} | " + " | ".join(_fmt(acc[lab]) for lab in order)
                         + f" | {'' if gap is None else f'{100 * gap:+.1f}'} |")
        out.setdefault("types", {})[b] = res
        lines.append("")
    return lines


STUDIES = ("confirm_", "dist_", "session_", "kgu_", "hippo_", "mx_")


def _beside_main(b: str, runs: Path) -> dict[str, list[dict[str, Any]]]:
    out = {}
    for exp, lab in ((f"dgrag_{b}", "dgrag@16k"), (f"hippo_{b}", "hipporag2_pooled")):
        rows = load_rows(exp, lab, runs)
        if rows:
            out[lab] = rows
    return out


def section_studies(runs: Path, out: dict[str, Any]) -> list[str]:
    lines = ["## 10. The other studies (the confirmation, the distribution control, the federated KG's sessions and "
             "updates, HippoRAG 2, the backbone cells)", ""]
    exps = sorted(p.name for p in (runs / "results").glob("*") if p.name.startswith(STUDIES))
    if not exps:
        return [*lines, "_none has rows yet._", ""]
    for exp in exps:
        labs = _labels(exp, runs)
        if not labs:
            continue
        stats = {lab: row_stats(load_rows(exp, lab, runs)) for lab in labs}
        out.setdefault("studies", {})[exp] = stats
        lines += [f"### {exp}", "", "| label | n | EM | F1 | ACC | R@5 | cost/q | calls/q |", "|---|---|---|---|---|---|---|---|"]
        for lab in labs:
            s = stats[lab]
            lines.append(f"| {lab} | {s['n']} | {_fmt(s['em'])} | {_fmt(s['f1'])} | {_fmt(s['acc'])} | "
                         f"{_fmt(s['r5'])} | {_fmt(s['cost'], 0)} | {_fmt(s['calls'], 1)} |")
        lines.append("")
    return lines


def section_routing(runs: Path) -> list[str]:
    p = runs / "results" / "routing_study" / "routing_study.md"
    if not p.exists():
        return ["## 11. Source selection", "", "_no routing study yet (tool:routing_study)._", ""]
    return ["## 11. Source selection (LLM-free; FedGRAG's Hit@k)", "", p.read_text(), ""]


def section_frontier(runs: Path, out: dict[str, Any]) -> list[str]:
    lines = ["## 2. The budget frontier (study questions): FedQuest vs. IRCoT's better configuration per budget", ""]
    figs = {}
    for b in BENCHES:
        fr, main = f"frontier_{b}", f"main_{b}"
        fr_labs = _labels(fr, runs)
        if not fr_labs:
            lines += [f"_{fr}: no rows yet._", ""]
            continue
        sub = {r["qid"] for r in load_rows(fr, fr_labs[0], runs)}
        pts: dict[str, list[tuple[float, float, int]]] = defaultdict(list)
        table = []
        for x in BUDGETS:
            k = f"{x // 1000}k"
            fq = load_rows(main, "fedquest", runs) if x == 16000 else load_rows(fr, f"fedquest@{k}", runs)
            fq = [r for r in fq if r["qid"] in sub]
            ir = {}
            for t in (4, 8):
                src = main if x == 16000 else fr
                ir[t] = [r for r in load_rows(src, f"ircot_t{t}@{k}", runs) if r["qid"] in sub]
            best_t = max(ir, key=lambda t: _mean(_val(ir[t], "acc")) or -1)
            sf, si = row_stats(fq), row_stats(ir[best_t])
            ld = lead(fq, ir[best_t])
            table.append((k, sf, si, best_t, ld))
            if sf["acc"] is not None:
                pts["FedQuest"].append((sf["cost"], sf["acc"], x))
            if si["acc"] is not None:
                pts["IRCoT (best of 4/8 a step)"].append((si["cost"], si["acc"], x))
            for lab, name_ in (("react", "ReAct"), ("mgrag_fu", "MGRAG+FU")):
                src = main if x == MAIN_B else fr
                s_ = row_stats([r for r in load_rows(src, f"{lab}@{k}", runs) if r["qid"] in sub])
                if s_["acc"] is not None:
                    pts[name_].append((s_["cost"], s_["acc"], x))
        for lab in ("react_ops@16k", "mgrag_fed@8", "vanilla_rag@8", "ircot@8x8", "fedgrag", "fedgrag_text",
                    "closed_book"):
            rows = [r for r in load_rows(main, lab, runs) if r["qid"] in sub]
            s = row_stats(rows)
            if s["acc"] is not None:
                pts[lab].append((s["cost"], s["acc"], 0))
        for lab, extra in _beside_main(b, runs).items():
            s = row_stats([r for r in extra if r["qid"] in sub])
            if s["acc"] is not None:
                pts[lab].append((s["cost"], s["acc"], 0))
        figs[b] = pts
        out.setdefault("frontier", {})[b] = {"table": [(k, sf, si, t, ld) for k, sf, si, t, ld in table],
                                             "points": pts}
        lines += [f"### {TITLE[b]} ({len(sub)} questions)", "", "| budget | FedQuest ACC | cost | IRCoT ACC (top) | cost | "
                  "FedQuest - IRCoT (95% CI) | p |", "|---|---|---|---|---|---|---|"]
        for k, sf, si, t, ld in table:
            lines.append(f"| {k} | {_fmt(sf['acc'])} | {_fmt(sf['cost'], 0)} | {_fmt(si['acc'])} ({t}) | "
                         f"{_fmt(si['cost'], 0)} | "
                         + ("" if ld is None else f"{100 * ld['diff']:+.1f} [{100 * ld['lo']:+.1f}, "
                            f"{100 * ld['hi']:+.1f}] | {_fmt(ld['p'], 4)}") + " |")
        lines.append("")
    _plot_frontier(figs, runs)
    return lines


def _plot_frontier(figs: dict[str, dict[str, list[tuple[float, float, int]]]], runs: Path) -> None:
    if not figs:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, axes = plt.subplots(1, len(figs), figsize=(3.2 * len(figs), 2.8), squeeze=False)
    for ax, (b, pts) in zip(axes[0], figs.items(), strict=False):
        for lab, ps in pts.items():
            ps = sorted(ps)
            xs, ys = [p[0] / 1000 for p in ps], [p[1] for p in ps]
            if len(ps) > 1:
                ax.plot(xs, ys, marker="o", label=lab, lw=1.5)
            else:
                ax.scatter(xs, ys, label=lab, s=18)
        ax.set_title(TITLE[b])
        ax.set_xlabel("cost units per question (K)")
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("accuracy")
    axes[0][-1].legend(fontsize=6, loc="lower right")
    fig.tight_layout()
    d = runs / "results" / "report"
    d.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(d / f"fig_frontier.{ext}", dpi=160)
    plt.close(fig)


def section_evidence(runs: Path, out: dict[str, Any]) -> list[str]:
    lines = ["## 3. What evidence each method retrieves, reads and answers from (main questions)", "",
             "| benchmark | label | ACC | returned | read | in context | later-hop reach | ACC reached | ACC not | "
             "tokens per gold read | read precision | errors: not returned / not read / not in context / answer |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    keep = ("fedquest", "ircot_t8@16k", "ircot_t4@16k", "ircot@8x8", "react@16k", "react_ops@16k", "mgrag_fed@8",
            "mgrag_fu@16k", "fedgrag", "fedgrag_text", "vanilla_rag@8", "closed_book")
    for b in BENCHES:
        for lab in keep:
            rows = load_rows(f"main_{b}", lab, runs)
            if not rows:
                continue
            s = summarize(rows)
            if not s.get("n"):
                continue
            out.setdefault("evidence", {}).setdefault(b, {})[lab] = s
            f, lh, ef, er = s["funnel_objects"], s["later_hop_questions"], s["efficiency"], s["errors"]
            lines.append(f"| {TITLE[b]} | {lab} | {_fmt(s['acc'])} | {_fmt(f['returned'])} | {_fmt(f['read'])} | "
                         f"{_fmt(f['in_context'])} | {_fmt(lh['reach'])} | {_fmt(lh['acc_reached'])} | "
                         f"{_fmt(lh['acc_unreached'])} | {_fmt(ef['tokens_per_gold_read'], 0)} | "
                         f"{_fmt(ef['read_precision'])} | {er.get('not returned', 0)} / {er.get('returned, not read', 0)}"
                         f" / {er.get('read, not in context', 0)} / {er.get('in context, answer wrong', 0)} |")
    lines += ["", "By hop (FedQuest and IRCoT at 16K): share of each hop's ground-truth objects returned / read.", "",
              "| benchmark | label | " + " | ".join(f"hop{i}" for i in range(1, 5)) + " | first hops | later hops |",
              "|---|---|---|---|---|---|---|---|"]
    for b in ("musique", "twowiki", "hotpot"):
        for lab in ("fedquest", "ircot_t8@16k", "mgrag_fed@8", "fedgrag"):
            s = out.get("evidence", {}).get(b, {}).get(lab)
            if not s:
                continue
            h = s["by_hop"]
            cell = (lambda k: f"{_fmt(h[k]['returned'], 2)} / {_fmt(h[k]['read'], 2)}" if k in h else "")  # noqa: E731, B023
            lines.append(f"| {TITLE[b]} | {lab} | " + " | ".join(cell(f"hop{i}") for i in range(1, 5))
                         + f" | {cell('first')} | {cell('later')} |")
    lines.append("")
    return lines


def section_ablations(runs: Path, out: dict[str, Any]) -> list[str]:
    lines = ["## 4. Ablations and the keys x reads factorial (study questions, 16K)", "",
             "| benchmark | variant | ACC | vs FedQuest (95% CI) | later-hop reach | cost/q | messages/q |",
             "|---|---|---|---|---|---|---|"]
    for b in BENCHES:
        exp = f"abl_{b}"
        labs = _labels(exp, runs)
        if not labs:
            continue
        sub = {r["qid"] for r in load_rows(exp, labs[0], runs)}
        ref = [r for r in load_rows(f"main_{b}", "fedquest", runs) if r["qid"] in sub]
        others = {lab: load_rows(exp, lab, runs) for lab in labs}
        others["mgrag_fed@8 (main)"] = [r for r in load_rows(f"main_{b}", "mgrag_fed@8", runs) if r["qid"] in sub]
        others["mgrag_fu@16k (main)"] = [r for r in load_rows(f"main_{b}", "mgrag_fu@16k", runs) if r["qid"] in sub]
        rs = summarize(ref) if ref else {}
        lines.append(f"| {TITLE[b]} | fedquest (main) | {_fmt(rs.get('acc'))} | | "
                     f"{_fmt((rs.get('later_hop_questions') or {}).get('reach'))} | "
                     f"{_fmt((rs.get('efficiency') or {}).get('tokens_mean'), 0)} | "
                     f"{_fmt((rs.get('efficiency') or {}).get('messages_mean'), 1)} |")
        for lab, rows in others.items():
            if not rows:
                continue
            s, ld = summarize(rows), lead(rows, ref)
            if not s.get("n"):
                continue
            out.setdefault("ablations", {}).setdefault(b, {})[lab] = {"summary": s, "vs_fedquest": ld}
            lines.append(f"| {TITLE[b]} | {lab} | {_fmt(s['acc'])} | "
                         + ("" if ld is None else f"{100 * ld['diff']:+.1f} [{100 * ld['lo']:+.1f}, {100 * ld['hi']:+.1f}]")
                         + f" | {_fmt(s['later_hop_questions']['reach'])} | {_fmt(s['efficiency']['tokens_mean'], 0)} | "
                         f"{_fmt(s['efficiency']['messages_mean'], 1)} |")
    lines.append("")
    return lines


SCALE_EXPS = ("scale_musique", "seeds_musique")
SCALE_LABEL = re.compile(r"(.+?)_s(\d+)(?:_k05)?_r(\d+)")


def _sd(v: list[float]) -> float | None:
    return float(np.std(v, ddof=1)) if len(v) > 1 else None


def section_scale(runs: Path, out: dict[str, Any]) -> list[str]:
    lines = ["## 5. Scale: MuSiQue over 2 to 32 databases (one a facility at 2; above, each facility's pages "
             "topic-skewed over its N/2 databases by Dirichlet(0.5)), the scale study's questions; means over the "
             "placement seeds each label ran (seed 0: every label; seeds 1-4: FedQuest, FedGRAG, Vanilla RAG, MGRAG)", ""]
    cells: dict[tuple[str, int], dict[int, list[dict[str, Any]]]] = defaultdict(dict)
    for exp in SCALE_EXPS:
        for lab in _labels(exp, runs):
            m = SCALE_LABEL.fullmatch(lab)
            if m:
                cells[(m.group(1), int(m.group(2)))][int(m.group(3))] = load_rows(exp, lab, runs)
    if not cells:
        return lines + ["_scale_musique: no rows yet._", ""]
    lines += ["| N | label | seeds | ACC (mean, sd over seeds) | F1 | R@5 | cost/q | KB/q | site calls/q | rounds/q | "
              "wall p50 (s) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    agg: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for (base, n), by_seed in sorted(cells.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        st = [row_stats(r) for r in by_seed.values()]

        def mean(k: str, st: list[dict[str, Any]] = st) -> float | None:
            v = [x[k] for x in st if x.get(k) is not None]
            return float(np.mean(v)) if v else None

        a = {k: mean(k) for k in ("acc", "f1", "r5", "cost", "kb", "messages", "rounds", "wall_p50")}
        a["acc_sd"] = _sd([x["acc"] for x in st if x.get("acc") is not None])
        a["seeds"] = sorted(by_seed)
        agg[n][base] = a
        sd = f" (sd {_fmt(a['acc_sd'])})" if a["acc_sd"] is not None else ""
        lines.append(f"| {n} | {base} | {len(by_seed)} | {_fmt(a['acc'])}{sd} | {_fmt(a['f1'])} | {_fmt(a['r5'])} | "
                     f"{_fmt(a['cost'], 0)} | {_fmt(a['kb'], 1)} | {_fmt(a['messages'], 1)} | {_fmt(a['rounds'], 1)} | "
                     f"{_fmt(a['wall_p50'], 1)} |")
    out["scale"] = agg
    lines += ["", "FedQuest's ACC lead over each label per N: its mean over the seeds both ran (sd; the seeds FedQuest "
              "led), and at seed 0 the paired 95% CI over questions.", "",
              "| N | over | seeds | mean lead (sd) | seeds led | seed 0 lead (95% CI) |", "|---|---|---|---|---|---|"]
    for n in sorted(agg):
        fq = cells.get(("fedquest", n), {})
        for base in sorted(agg[n]):
            if base == "fedquest" or not fq:
                continue
            other = cells[(base, n)]
            both = sorted(set(fq) & set(other))
            diffs = []
            for sd_ in both:
                a_, b_ = row_stats(fq[sd_])["acc"], row_stats(other[sd_])["acc"]
                if a_ is not None and b_ is not None:
                    diffs.append(a_ - b_)
            ld = lead(fq[0], other[0]) if 0 in both else None
            sd = _sd(diffs)
            lines.append(f"| {n} | {base} | {len(diffs)} | "
                         + (f"{100 * float(np.mean(diffs)):+.1f}" + (f" ({100 * sd:.1f})" if sd is not None else "")
                            if diffs else "")
                         + f" | {sum(d > 0 for d in diffs)}/{len(diffs)} | "
                         + ("" if ld is None else f"{100 * ld['diff']:+.1f} [{100 * ld['lo']:+.1f}, {100 * ld['hi']:+.1f}]")
                         + " |")
    lines += ["", "Routing (seed 0): routed FedQuest against broadcast: ACC difference and site calls.", "",
              "| N | ACC routed - broadcast (95% CI) | site calls routed / broadcast |", "|---|---|---|"]
    for n in sorted(agg):
        a_, b_ = cells.get(("fedquest", n), {}).get(0), cells.get(("fedquest_all", n), {}).get(0)
        if not a_ or not b_:
            continue
        ld = lead(a_, b_)
        lines.append(f"| {n} | " + ("" if ld is None else f"{100 * ld['diff']:+.1f} [{100 * ld['lo']:+.1f}, "
                                    f"{100 * ld['hi']:+.1f}]")
                     + f" | {_fmt(row_stats(a_)['messages'], 1)} / {_fmt(row_stats(b_)['messages'], 1)} |")
    lines.append("")
    _plot_scale(agg, runs)
    return lines


def _plot_scale(agg: dict[int, dict[str, dict[str, Any]]], runs: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    ns = sorted(agg)
    if not ns:
        return
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.8))
    for base in ("fedquest", "fedquest_all", "fedgrag", "mgrag_fed@8", "vanilla_rag@8", "ircot_t8@16k"):
        xs = [n for n in ns if base in agg[n]]
        if not xs:
            continue
        for ax, key in zip(axes, ("acc", "cost", "messages"), strict=True):
            ys = [agg[n][base][key] for n in xs]
            err = [agg[n][base]["acc_sd"] or 0.0 for n in xs] if key == "acc" else None
            ax.errorbar(xs, ys, yerr=err, marker="o", capsize=2, label=base)
    for ax, t in zip(axes, ("accuracy (sd over seeds)", "cost units / question", "site calls / question"), strict=True):
        ax.set_xscale("log", base=2)
        ax.set_title(t)
        ax.set_xlabel("databases")
        ax.grid(alpha=0.3)
    axes[2].legend(fontsize=6)
    fig.tight_layout()
    d = runs / "results" / "report"
    d.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(d / f"fig_scale.{ext}", dpi=160)
    plt.close(fig)


def section_grid(runs: Path, out: dict[str, Any]) -> list[str]:
    p = runs / "results" / "scale_grid" / "grid.jsonl"
    lines = ["## 6. The federation tier's scaling grid (probe-trace replay, no model call)", ""]
    if not p.exists():
        return lines + ["_no grid results yet (bash scripts/grid.sh run)._", ""]
    rows = [json.loads(line) for line in p.open() if line.strip()]
    out["grid"] = rows
    lines += ["| nodes/facility | DBs/node | DBs | mode | clients | questions/s | question p50 / p95 (s) | "
              "round p50 / p99 (s) | busiest site (s) | all sites busy (s) |", "|---|---|---|---|---|---|---|---|---|---|"]

    def mode(r: dict[str, Any]) -> str:
        return r.get("mode") or ("routed" if r.get("routed") else "broadcast")

    for r in sorted(rows, key=lambda r: (r.get("nodes", 0), r.get("per_node", 0), mode(r), r.get("clients", 0))):
        lines.append(f"| {r.get('nodes')} | {r.get('per_node')} | {r.get('databases')} | {mode(r)} | "
                     f"{r.get('clients')} | {_fmt(r.get('q_per_s'))} | {_fmt(r.get('q_p50'), 2)} / "
                     f"{_fmt(r.get('q_p95'), 2)} | {_fmt(r.get('round_p50'), 2)} / {_fmt(r.get('round_p99'), 2)} | "
                     f"{_fmt(r.get('busy_max_s'), 1)} | {_fmt(r.get('busy_total_s'), 1)} |")
    lines.append("")
    return lines


def section_cache(runs: Path, out: dict[str, Any]) -> list[str]:
    lines = ["## 7. The fact cache: the live stream and the policies' simulation", ""]
    exp = "session_musique"
    labs = _labels(exp, runs)
    if labs:
        lines += ["| arm | n | ACC | cost/q | rounds/q | wall p50 | reused facts/q |", "|---|---|---|---|---|---|---|"]
        for lab in labs:
            rows = load_rows(exp, lab, runs)
            s = row_stats(rows)
            reuse = np.mean([float((r.get("ledger") or {}).get("reused_objects", 0)) for r in rows]) if rows else None
            lines.append(f"| {lab} | {s['n']} | {_fmt(s['acc'])} | {_fmt(s['cost'], 0)} | {_fmt(s['rounds'], 2)} | "
                         f"{_fmt(s['wall_p50'], 1)} | {_fmt(float(reuse) if reuse is not None else None, 2)} |")
        lines.append("")
    sims = sorted((runs / "results" / "cache_sim").glob("*.jsonl"))
    if not sims:
        return [*lines, "_no cache simulation yet (bash scripts/post.sh)._", ""]
    for p in sims:
        lines += [f"### The simulation over {p.stem}'s cold FedQuest (fedquest.cache.sim)", ""]
        rows = [json.loads(line) for line in p.open() if line.strip()]
        out.setdefault("cache_sim", {})[p.stem] = rows
        cap = [r for r in rows if r.get("study") == "capacity"]
        val = [r for r in rows if r.get("study") == "validity"]
        prof = next((r for r in rows if r.get("study") == "reuse_profile"), None)
        lines += ["Capacity x eviction (no updates): hit rate / tokens saved.", "",
                  "| capacity (share of all facts) | " + " | ".join(("lru", "lfu", "fifo", "gdsf", "benefit")) + " |",
                  "|---|---|---|---|---|---|"]
        for frac in sorted({r.get("cap_frac") for r in cap}, key=lambda x: (x is None, x or 0)):
            cells = []
            for ev in ("lru", "lfu", "fifo", "gdsf", "benefit"):
                r = next((r for r in cap if r.get("cap_frac") == frac and r["eviction"] == ev), None)
                cells.append("" if r is None else f"{_fmt(r['hit_rate'])} / {_fmt(r['tokens_saved_frac'])}")
            lines.append(f"| {'unbounded' if frac is None else frac} | " + " | ".join(cells) + " |")
        lines += ["", "Validity under updates (changes per 1,000 objects per question): stale hits / validation rounds /"
                  " net rounds saved.", "", "| update rate | trust | validate | ttl 25 | ttl 100 | ttl 400 | piggyback 25"
                  " | piggyback 100 | piggyback 400 |", "|---|---|---|---|---|---|---|---|---|"]
        for rate in sorted({r["update_rate"] for r in val}):
            cells = []
            for v, ttl in (("trust", 0), ("validate", 0), ("ttl", 25), ("ttl", 100), ("ttl", 400), ("piggyback", 25),
                           ("piggyback", 100), ("piggyback", 400)):
                r = next((r for r in val if r["update_rate"] == rate and r["validity"] == v and r["ttl"] == ttl), None)
                cells.append("" if r is None else f"{_fmt(r['stale_hit_rate'])} / {r['validation_rounds']} / "
                             f"{_fmt(r['net_rounds_saved_frac'])}")
            lines.append(f"| {rate} | " + " | ".join(cells) + " |")
        if prof:
            lines += ["", f"Reuse distances (questions between a fact's first read and its reuses): p50 {prof['p50']},"
                      f" p90 {prof['p90']}, p99 {prof['p99']} over {prof['reuses']} reuses of {prof['entries']} "
                      "entries."]
        lines.append("")
    return lines


def section_merge(runs: Path) -> list[str]:
    p = runs / "results" / "merge_study" / "merge_study.md"
    if not p.exists():
        return ["## 8. Result merging", "", "_no merge study yet (tool:merge_study)._", ""]
    body = p.read_text().split("\n", 2)[-1]
    return ["## 8. Result merging across heterogeneous databases (LLM-free, calibration probes)", "", body, ""]


def section_latency(runs: Path, out: dict[str, Any]) -> list[str]:
    lines = ["## 9. Latency (live model calls, 4 questions in flight)", "",
             "| benchmark | label | n | wall p50 (s) | wall p95 (s) | cost/q |", "|---|---|---|---|---|---|"]
    for b in ("mmqa", "musique"):
        exp = f"latency_{b}"
        for lab in _labels(exp, runs):
            s = row_stats(load_rows(exp, lab, runs))
            lines.append(f"| {TITLE[b]} | {lab} | {s['n']} | {_fmt(s['wall_p50'], 1)} | {_fmt(s['wall_p95'], 1)} | "
                         f"{_fmt(s['cost'], 0)} |")
    lines.append("")
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m fedquest.eval.report")
    ap.add_argument("--runs", type=Path, default=None)
    args = ap.parse_args(argv)
    runs = args.runs or runs_root()
    out: dict[str, Any] = {}
    lines = ["# FedQuest: report", ""]
    for sec in (section_main, section_heldout, section_types, section_frontier, section_evidence, section_ablations,
                section_scale,
                section_grid, section_cache, section_studies):
        try:
            lines += sec(runs, out)
        except Exception as e:
            lines += [f"_{sec.__name__} failed: {type(e).__name__}: {e}_", ""]
    lines += section_merge(runs)
    lines += section_latency(runs, out)
    lines += section_routing(runs)
    d = runs / "results" / "report"
    d.mkdir(parents=True, exist_ok=True)
    (d / "report.md").write_text("\n".join(lines) + "\n")
    (d / "report.json").write_text(json.dumps(out, indent=1, default=str))
    print(f"report: {d / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
