from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import re
import time
from pathlib import Path
from typing import Any

import numpy as np

from fedquest.paths import runs_root

log = logging.getLogger("fedquest.routing_study")
HITS = (1, 2, 5)
LAYOUT_TIMEOUT_S = 900.0
_QVECS: dict[tuple[str, str], tuple[np.ndarray, float, float]] = {}
_FVECS: dict[str, np.ndarray] = {}


def random_hit(n: int, g: int, k: int) -> float:
    if g <= 0:
        return 0.0
    k = min(k, n)
    return 1.0 - math.comb(n - g, k) / math.comb(n, k) if n - g >= k else 1.0


def _ranked(scores: dict[str, float]) -> list[str]:
    return sorted(scores, key=lambda s: (-scores[s], s))


def _seed_size(name: str) -> tuple[int, int, str]:
    m = re.search(r"_s(\d+)(?:_k\d+)?_r(\d+)$", name)
    return (int(m.group(2)), int(m.group(1)), name) if m else (99, 0, name)


def embedded(encoders: list[str], texts: list[str]) -> list[dict[str, tuple[np.ndarray, float, float]]]:
    from fedquest.routing import _encoder
    from fedquest.vdb.anchors import anchor_matrix, background

    for enc in encoders:
        todo = list(dict.fromkeys(t for t in texts if (enc, t) not in _QVECS))
        if not todo:
            continue
        e = _encoder(enc)
        vecs = np.asarray(e.encode_queries(todo), dtype=np.float32)
        vecs /= np.maximum(1e-9, np.linalg.norm(vecs, axis=1, keepdims=True))
        anchors = anchor_matrix(enc, e)
        for t, v in zip(todo, vecs, strict=False):
            mu, sd = background(anchors, v) if anchors is not None else (0.0, 1.0)
            _QVECS[(enc, t)] = (v, mu, max(sd, 1e-6))
    return [{enc: _QVECS[(enc, t)] for enc in encoders} for t in texts]


async def study(layout: str, mode: str = "remote", split: str | None = None) -> dict[str, Any]:
    from fedquest import routing
    from fedquest.config import load_federation
    from fedquest.fedgrag.build import EVAL_SPLITS

    if split is None:
        split = EVAL_SPLITS.get(load_federation(layout).dataset, ("final_test",))[0]
    from fedquest.eval.runner import load_questions
    from fedquest.fedgrag.model import Encoder, FedGRAGIndex, layout_dir

    if mode == "local":
        from fedquest.federation.local import open_local_federation

        fed = await asyncio.to_thread(open_local_federation, layout)
    else:
        from fedquest.federation.gc import open_gc_federation

        fed = open_gc_federation(layout)

    async def deployment() -> tuple[Any, Any]:
        return await fed.synopsis(routing.K), await fed.catalog()

    try:
        syn, cat = await asyncio.wait_for(deployment(), LAYOUT_TIMEOUT_S)
    finally:
        await fed.close()
    router = routing.Router(syn)
    direct = routing.Router(routing.mean_synopses(syn))
    doc_sites: dict[str, set[str]] = {}
    for site, cols in cat.items():
        for rows in cols.values():
            for _oid, doc, _mod in rows:
                doc_sites.setdefault(doc, set()).add(site)
    has_fedgrag = (layout_dir(layout) / "config.json").exists()
    questions = [q for q in load_questions(layout, split) if q.gold_doc_ids]
    texts = [q.question for q in questions]
    embs = embedded(sorted({enc for enc, _ in [*router.cents.values(), *direct.cents.values()]}), texts)
    qvecs = None
    if has_fedgrag:
        todo = [t for t in dict.fromkeys(texts) if t not in _FVECS]
        if todo:
            _FVECS.update(zip(todo, Encoder.encode(todo), strict=False))
        qvecs = [_FVECS[t] for t in texts]
    from fedquest import xfit

    halves = has_fedgrag and bool(json.loads((layout_dir(layout) / "config.json").read_text()).get("routers"))
    if halves:
        idxs = {h: FedGRAGIndex.load(layout, f"x{h}") for h in xfit.HALVES}
        fits = {h: routing.load_fit(f"{layout}_x{h}") for h in xfit.HALVES}
    else:
        one = FedGRAGIndex.load(layout, None) if has_fedgrag else None
        idxs = dict.fromkeys(xfit.HALVES, one)
        fits = dict.fromkeys(xfit.HALVES, routing.load_fit(layout))
    acc: dict[str, dict[str, list[float]]] = {r: {} for r in ("fedquest", "direct", "fedgrag", "random")}

    def note(r: str, key: str, v: float) -> None:
        acc[r].setdefault(key, []).append(float(v))

    n_sites = len(router.sites)
    for i, q in enumerate(questions):
        gold = set().union(*(doc_sites.get(d, set()) for d in q.gold_doc_ids))
        if not gold:
            continue
        sc = router.scores(q.question, embs[i])
        order = _ranked(sc)
        dorder = _ranked(direct.scores(q.question, embs[i]))
        for k in HITS:
            note("fedquest", f"hit@{k}", bool(set(order[:k]) & gold))
            note("direct", f"hit@{k}", bool(set(dorder[:k]) & gold))
            note("random", f"hit@{k}", random_hit(n_sites, len(gold), k))
        fit, idx = fits[xfit.other(q.qid)], idxs[xfit.other(q.qid)]
        if fit is not None and fit.get("theta") is not None:
            sel = set(routing.Router.select(sc, "threshold", fit["theta"], None,
                                            int(fit.get("min_sites", routing.MIN_SITES))))
            note("fedquest", "selected", len(sel))
            note("fedquest", "recall", len(sel & gold) / len(gold))
        if qvecs is not None and idx is not None:
            sc_f = idx.client_scores(qvecs[i])
            forder = [idx.clients[j].site for j in sorted(range(len(idx.clients)), key=lambda j: (-float(sc_f[j]),
                                                                                                 idx.clients[j].site))]
            for k in HITS:
                note("fedgrag", f"hit@{k}", bool(set(forder[:k]) & gold))
            sel = set(forder[: idx.b])
            note("fedgrag", "selected", len(sel))
            note("fedgrag", "recall", len(sel & gold) / len(gold))
    out = {"layout": layout, "split": split, "sites": n_sites, "questions": len(questions), "crossfitted": halves,
           "routers": {r: {k: round(float(np.mean(v)), 4) for k, v in m.items()} for r, m in acc.items() if m}}
    return out


def served_layouts() -> list[str]:
    from fedquest.config import load_remote

    built = sorted(p.name for p in (runs_root() / "fedgrag" / "layouts").glob("*") if (p / "config.json").exists())
    served = set(load_remote().federations)
    left = [n for n in built if n not in served]
    if left:
        log.info("left out, not deployed: %s", ", ".join(left))
    return sorted((n for n in built if n in served), key=_seed_size)


def write(out_dir: Path, results: list[dict[str, Any]], failed: list[str]) -> Path:
    lines = ["# Source selection (fedquest.eval.routing_study): Hit@k on each dataset's evaluation questions, each "
             "router fitted on other questions", "",
             "| layout | sites | questions | router | hit@1 | hit@2 | hit@5 | selected | recall of the evidence's "
             "databases |", "|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        for name, m in r["routers"].items():
            lines.append(f"| {r['layout']} | {r['sites']} | {r['questions']} | {name} | {m.get('hit@1', '')} | "
                         f"{m.get('hit@2', '')} | {m.get('hit@5', '')} | {m.get('selected', '')} | "
                         f"{m.get('recall', '')} |")
    if failed:
        lines += ["", f"Failed (the log says why): {', '.join(failed)}."]
    (out_dir / "routing_study.json").write_text(json.dumps(results, indent=1))
    (out_dir / "routing_study.md").write_text("\n".join(lines) + "\n")
    return out_dir / "routing_study.md"


async def run_all(layouts: list[str], mode: str, split: str | None,
                  out_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    results: list[dict[str, Any]] = []
    failed: list[str] = []
    for j, layout in enumerate(layouts, 1):
        t0 = time.time()
        try:
            r = await study(layout, mode, split)
        except Exception as e:
            log.exception("routing study of %s failed: %s", layout, e)
            failed.append(layout)
        else:
            results.append(r)
            fq, fg = r["routers"].get("fedquest", {}), r["routers"].get("fedgrag", {})
            log.info("%d/%d %s: %d questions, %d sites, %.0f s; hit@1 fedquest %s, fedgrag %s", j, len(layouts),
                     layout, r["questions"], r["sites"], time.time() - t0, fq.get("hit@1"), fg.get("hit@1"))
        write(out_dir, results, failed)
    return results, failed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m fedquest.eval.routing_study")
    ap.add_argument("layouts", nargs="*", help="layouts (default: every layout FedGRAG was built on that the "
                                               "deployment serves, seed 0's first)")
    ap.add_argument("--mode", default="remote", choices=("remote", "local"))
    ap.add_argument("--split", default=None, help="the questions (default: each dataset's evaluation pool)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.layouts:
        layouts = list(args.layouts)
    elif args.mode == "remote":
        layouts = served_layouts()
    else:
        layouts = sorted((p.name for p in (runs_root() / "fedgrag" / "layouts").glob("*")
                          if (p / "config.json").exists()), key=_seed_size)
    out_dir = args.out or runs_root() / "results" / "routing_study"
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("%d layouts: %s", len(layouts), ", ".join(layouts))
    results, failed = asyncio.run(run_all(layouts, args.mode, args.split, out_dir))
    print(f"{len(results)} layouts ({len(failed)} failed): {out_dir}/routing_study.md")
    return 1 if failed and not results else 0


if __name__ == "__main__":
    raise SystemExit(main())
