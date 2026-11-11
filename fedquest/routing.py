# database routing: centroid synopses, anchor-z site scores and threshold fits
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import math
import threading
from pathlib import Path
from typing import Any

import numpy as np

from fedquest.paths import runs_root

log = logging.getLogger("fedquest.routing")

K = 16
TARGET = 0.98
MIN_SITES = 2
_ENC: dict[str, Any] = {}
_ENC_LOCK = threading.Lock()


def routing_path(federation: str) -> Path:
    return runs_root() / "routing" / f"{federation}.json"


def _encoder(name: str) -> Any:
    from fedquest.vdb.encoders import load_encoder

    with _ENC_LOCK:
        if name not in _ENC:
            _ENC[name] = load_encoder(name)
        return _ENC[name]


def _decode(entry: dict[str, Any]) -> np.ndarray:
    raw = base64.b64decode(entry["centroids"])
    return np.frombuffer(raw, dtype=np.float16).reshape(-1, int(entry["dim"])).astype(np.float32)


def mean_synopses(synopses: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for site, syn in synopses.items():
        cols: dict[str, Any] = {}
        for col, entry in (syn.get("collections") or {}).items():
            if entry.get("modality") == "image":
                cols[col] = entry
                continue
            cents = _decode(entry)
            w = np.asarray(entry.get("sizes") or [1] * len(cents), dtype=np.float32)
            m = (w[:, None] * cents).sum(axis=0)
            m /= max(float(np.linalg.norm(m)), 1e-9)
            cols[col] = {**entry, "centroids": base64.b64encode(m.astype(np.float16).tobytes()).decode(),
                         "sizes": [int(w.sum())]}
        out[site] = {**syn, "collections": cols}
    return out


class Router:
    def __init__(self, synopses: dict[str, dict[str, Any]]) -> None:
        self.cents: dict[tuple[str, str], tuple[str, np.ndarray]] = {}
        self.always: set[str] = set()
        for site, syn in synopses.items():
            for col, entry in (syn.get("collections") or {}).items():
                if entry.get("modality") == "image":
                    self.always.add(site)
                    continue
                self.cents[(site, col)] = (entry["encoder"], _decode(entry))
        self.sites = sorted(synopses)

    @classmethod
    async def for_federation(cls, fed: Any) -> Router:
        cached = getattr(fed, "_router", None)
        if cached is None:
            cached = cls(await fed.synopsis(K))
            fed._router = cached
        return cached

    def scores(self, question: str,
               embedded: dict[str, tuple[np.ndarray, float, float]] | None = None) -> dict[str, float]:
        from fedquest.vdb.anchors import anchor_matrix, background

        by_enc: dict[str, tuple[np.ndarray, float, float]] = {} if embedded is None else embedded
        out: dict[str, float] = {s: -math.inf for s in self.sites}
        for (site, _), (enc, cents) in self.cents.items():
            if enc not in by_enc:
                e = _encoder(enc)
                q = np.asarray(e.encode_queries([question])[0], dtype=np.float32)
                q /= max(1e-9, float(np.linalg.norm(q)))
                anchors = anchor_matrix(enc, e)
                mu, sd = background(anchors, q) if anchors is not None else (0.0, 1.0)
                by_enc[enc] = (q, mu, max(sd, 1e-6))
            q, mu, sd = by_enc[enc]
            # a site scores its best centroid, z-scored against the anchor background
            z = float(((cents @ q) - mu).max() / sd) if len(cents) else -math.inf
            out[site] = max(out[site], z)
        for s in self.always:
            out[s] = math.inf
        return out

    @staticmethod
    def select(scores: dict[str, float], policy: str, theta: float | None, b: int | None,
               min_sites: int = MIN_SITES) -> list[str]:
        order = sorted(scores, key=lambda s: (-scores[s], s))
        if policy == "all" or (policy == "threshold" and theta is None) or (policy == "topb" and not b):
            return order
        if policy == "topb":
            keep = order[: max(min_sites, int(b or 0))]
        else:
            keep = [s for s in order if scores[s] >= float(theta)]
            if len(keep) < min_sites:
                keep = order[:min_sites]
        keep += [s for s in order if math.isinf(scores[s]) and scores[s] > 0 and s not in keep]
        return sorted(set(keep), key=order.index)


def load_fit(federation: str) -> dict[str, Any] | None:
    p = routing_path(federation)
    return json.loads(p.read_text()) if p.exists() else None


def fit_name(federation: str, route_fit: str | None, qid: str | None = None) -> str:
    if route_fit is None:
        return federation
    from fedquest import xfit

    name = route_fit.replace("{federation}", federation)
    return (xfit.resolve(name, qid) if qid is not None else name) or federation


async def calibrate(federation: str, mode: str = "remote", split: str = "calib", target: float = TARGET,
                    min_sites: int = MIN_SITES, halves: bool = False) -> list[dict[str, Any]]:
    from fedquest import xfit
    from fedquest.eval.runner import load_questions

    if mode == "local":
        from fedquest.federation.local import open_local_federation

        fed = await asyncio.to_thread(open_local_federation, federation)
    else:
        from fedquest.federation.gc import open_gc_federation

        fed = open_gc_federation(federation)
    try:
        router = await Router.for_federation(fed)
        cat = await fed.catalog()
    finally:
        await fed.close()
    doc_sites: dict[str, set[str]] = {}
    for site, cols in cat.items():
        for rows in cols.values():
            for _oid, doc, _mod in rows:
                doc_sites.setdefault(doc, set()).add(site)
    questions = load_questions(federation, split)
    if not halves:
        return [_fit(federation, federation, split, router, doc_sites, questions, target, min_sites)]
    return [_fit(f"{federation}_x{h}", federation, f"{split}:{xfit.split_of(h)}", router, doc_sites,
                 [q for q in questions if xfit.in_half(q.qid, h)], target, min_sites) for h in xfit.HALVES]


def _fit(name: str, federation: str, split: str, router: Router, doc_sites: dict[str, set[str]],
         questions: list[Any], target: float, min_sites: int) -> dict[str, Any]:
    pairs: list[float] = []
    per_q: list[tuple[dict[str, float], set[str]]] = []
    for q in questions:
        first = [h.get("doc_id") for h in (q.meta or {}).get("hops") or [] if not h.get("after")]
        docs = [d for d in (first or q.gold_doc_ids) if d]
        gold_sites = set().union(*(doc_sites.get(d, set()) for d in docs)) if docs else set()
        if not gold_sites:
            continue
        sc = router.scores(q.question)
        per_q.append((sc, gold_sites))
        pairs += [sc[s] for s in gold_sites if s in sc]
    finite = sorted(v for v in pairs if math.isfinite(v))
    theta = None
    if finite:
        # theta: the (1 - target) quantile of the gold sites' scores
        cut = int(math.floor((1.0 - target) * len(finite)))
        theta = float(finite[max(0, min(len(finite) - 1, cut))]) - 1e-6
    n = len(router.sites)
    sel = [len(Router.select(sc, "threshold", theta, None, min_sites)) for sc, _ in per_q]
    rec = [len(g & set(Router.select(sc, "threshold", theta, None, min_sites))) / len(g) for sc, g in per_q]
    b_needed = None
    for b in range(1, n + 1):
        r = [len(g & set(Router.select(sc, "topb", None, b, min_sites))) / len(g) for sc, g in per_q]
        if r and float(np.mean(r)) >= target:
            b_needed = b
            break
    fit = {"federation": federation, "fit": name, "split": split, "target": target, "min_sites": min_sites, "k": K,
           "theta": theta, "questions": len(per_q), "sites": n, "recall": round(float(np.mean(rec)), 4) if rec else None,
           "selected_mean": round(float(np.mean(sel)), 3) if sel else None, "b": b_needed or n}
    p = routing_path(name)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(fit, indent=1))
    return fit


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m fedquest.routing")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calibrate", help="fit the question-probe router's threshold per layout (LLM-free)")
    c.add_argument("federations", nargs="+")
    c.add_argument("--mode", default="remote", choices=("remote", "local"))
    c.add_argument("--split", default="calib")
    c.add_argument("--target", type=float, default=TARGET)
    c.add_argument("--min-sites", type=int, default=MIN_SITES)
    c.add_argument("--halves", action="store_true",
                   help="Cross-fitting: one fit per half of the split's questions (<federation>_xA, _xB)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    bad = 0
    for f in args.federations:
        try:
            for fit in asyncio.run(calibrate(f, args.mode, args.split, args.target, args.min_sites, args.halves)):
                print(f"{fit['fit']}: theta={fit['theta']} recall={fit['recall']} selected={fit['selected_mean']} of "
                      f"{fit['sites']} on {fit['questions']} questions (FedGRAG-style top-B for the same recall: "
                      f"{fit['b']})")
        except Exception as e:
            log.exception("routing calibration of %s failed: %s", f, e)
            bad += 1
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
