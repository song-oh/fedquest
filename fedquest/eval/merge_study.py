from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from fedquest.paths import data_root, runs_root
from fedquest.utils import stable_int

STRATEGIES = ("raw", "minmax", "zlocal", "rrf", "roundrobin", "central", "crossenc", "anchorz", "anchorz_cal",
              "anchorz_title", "fused", "oracle")
FEATURES = {"anchorz_cal": ("z", "mention"), "anchorz_title": ("z", "mention", "t_in", "t_cov"),
            "fused": ("z", "mention", "t_in", "t_cov", "ce")}
BUDGETS = (250, 500, 1000, 2000)
TEXT_HOMES = {"musique": ["musique_next_home", "musique_next", "musique_fed"], "twowiki": ["twowiki_home", "twowiki_fed"],
              "hotpotqa": ["hotpot_home", "hotpot_fed"], "mmqa": ["mmqa_home", "mmqa_fed"]}


def probes_path(layout: str) -> Path:
    for root in (runs_root() / "calibration", runs_root() / "inputs" / "calibration"):
        p = root / f"{layout}.obs.jsonl"
        if p.exists():
            return p
    raise FileNotFoundError(f"no calibration probes for {layout} (runs/calibration or runs/inputs/calibration)")


def load_lists(layout: str, kinds: tuple[str, ...]) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    out: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for line in probes_path(layout).open(encoding="utf-8"):
        r = json.loads(line)
        if r.get("path") == "gold" or r.get("kind") not in kinds or r.get("evidence") is None:
            continue
        out[(r["qid"], r["round"], r["kind"])].append(r)
    return {k: v for k, v in out.items() if any(c["evidence"] for c in v)}


def _objects(dataset: str, oids: set[str], titles: dict[str, str] | None = None) -> dict[str, str]:
    found: dict[str, str] = {}
    for fed in TEXT_HOMES.get(dataset, []):
        for p in (data_root() / "fed" / fed / "sites").glob("*/*/objects.jsonl"):
            with p.open(encoding="utf-8") as f:
                for line in f:
                    if '"oid"' not in line:
                        continue
                    o = json.loads(line)
                    if o["oid"] in oids and o["oid"] not in found:
                        found[o["oid"]] = f"{o.get('title', '')}: {o.get('text', '')}"[:280]
                        if titles is not None:
                            titles[o["oid"]] = o.get("title", "")
        if len(found) >= len(oids):
            break
    return found


def _questions(dataset: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for fed in TEXT_HOMES.get(dataset, []):
        for p in (data_root() / "fed" / fed / "questions").glob("*.jsonl"):
            for line in p.open(encoding="utf-8"):
                if line.strip():
                    q = json.loads(line)
                    out.setdefault(q["qid"], q["question"])
    return out


def _backgrounds(questions: dict[str, str], encoders: set[str]) -> dict[tuple[str, str], tuple[float, float]]:
    from fedquest.vdb.anchors import anchor_matrix, background
    from fedquest.vdb.encoders import load_encoder

    out = {}
    for name in sorted(e for e in encoders if e in ("bge-base", "e5-base")):
        enc = load_encoder(name)
        anchors = anchor_matrix(name, enc)
        qids = sorted(questions)
        vecs = enc.encode_queries([questions[q] for q in qids])
        for q, v in zip(qids, vecs, strict=True):
            out[(q, name)] = background(anchors, v)
    return out


def _x(c: dict[str, Any], names: tuple[str, ...], extra: dict[str, Any]) -> list[float]:
    t_in, t_cov = extra.get("titles", {}).get(id(c), (0.0, 0.0))
    every = {"z": float(c.get("z") or 0.0), "mention": float(bool(c.get("mentions"))), "t_in": t_in, "t_cov": t_cov,
             "ce": float(extra.get("crossenc", {}).get(id(c), 0.0))}
    return [every[n] for n in names]


def _fit_platt(train: list[dict[str, Any]], names: tuple[str, ...] = FEATURES["anchorz_cal"],
               extra: dict[str, Any] | None = None) -> dict[str, tuple[float, ...]]:
    from sklearn.linear_model import LogisticRegression

    cells: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in train:
        cells[f"{c.get('encoder')}|{c.get('modality')}|{c.get('kind')}"].append(c)
        cells["all"].append(c)
    out = {}
    for cell, cs in cells.items():
        y = np.asarray([bool(c["evidence"]) for c in cs])
        if y.sum() < 3 or (~y).sum() < 3:
            continue
        x = np.asarray([_x(c, names, extra or {}) for c in cs])
        m = LogisticRegression(C=1.0, max_iter=500).fit(x, y)
        out[cell] = (float(m.intercept_[0]), *(float(v) for v in m.coef_[0]))
    return out


def _platt(cell_w: dict[str, tuple[float, ...]], c: dict[str, Any], names: tuple[str, ...] = FEATURES["anchorz_cal"],
           extra: dict[str, Any] | None = None) -> float:
    w = cell_w.get(f"{c.get('encoder')}|{c.get('modality')}|{c.get('kind')}") or cell_w.get("all") or \
        (0.0, 1.0, *([0.0] * (len(names) - 1)))
    t = w[0] + sum(wi * xi for wi, xi in zip(w[1:], _x(c, names, extra or {}), strict=True))
    return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, t))))


_STOP = frozenset("the a an of in on at to for by with from and or is was were are be been who what which when where "
                  "how whose whom that this did does do as its it his her their".split())


def title_feats(question: str, title: str) -> tuple[float, float]:
    from fedquest.data.hipporag import entity_of
    from fedquest.kg.normalize import norm_entity

    def words(text: str) -> list[str]:
        return [w for w in re.findall(r"[a-z0-9]+", norm_entity(text or "")) if w not in _STOP]

    if not title:
        return 0.0, 0.0
    q = " ".join(re.findall(r"[a-z0-9]+", norm_entity(question)))
    ent = " ".join(re.findall(r"[a-z0-9]+", norm_entity(entity_of(title))))
    tw, qw = words(entity_of(title)), set(words(question))
    t_in = float(len(ent) >= 3 and re.search(rf"(^| ){re.escape(ent)}( |$)", q) is not None)
    return t_in, (sum(w in qw for w in tw) / len(tw)) if tw else 0.0


def _order(strategy: str, cands: list[dict[str, Any]], extra: dict[str, Any]) -> list[dict[str, Any]]:
    salt = lambda c: stable_int(f"{c['qid']}|{c['site']}|{c['oid']}")  # noqa: E731
    by_list: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for c in cands:
        by_list[(c["site"], c["collection"])].append(c)
    if strategy == "oracle":
        return sorted(cands, key=lambda c: (not c["evidence"], c["rank"], salt(c)))
    if strategy == "roundrobin":
        lists = [sorted(v, key=lambda c: c["rank"]) for _, v in sorted(by_list.items())]
        out = []
        for i in range(max(len(v) for v in lists)):
            out += [v[i] for v in lists if i < len(v)]
        return out
    if strategy == "rrf":
        key = {id(c): 1.0 / (60 + c["rank"]) for c in cands}
    elif strategy == "anchorz":
        key = {id(c): float(c.get("z") or -99) for c in cands}
    elif strategy in FEATURES:
        if strategy == "fused" and not extra.get("crossenc"):
            return []
        w = extra[f"platt_{strategy}"]
        key = {id(c): _platt(w, c, FEATURES[strategy], extra) for c in cands}
    elif strategy in ("raw", "minmax", "zlocal"):
        if "raw" not in extra:
            return []
        raw = {id(c): extra["raw"].get(id(c)) for c in cands}
        if any(v is None for v in raw.values()):
            return []
        if strategy == "raw":
            key = raw
        else:
            key = {}
            for _, v in by_list.items():
                xs = np.asarray([raw[id(c)] for c in v])
                if strategy == "minmax":
                    lo, hi = float(xs.min()), float(xs.max())
                    for c in v:
                        key[id(c)] = (raw[id(c)] - lo) / (hi - lo) if hi > lo else 1.0
                else:
                    mu, sd = float(xs.mean()), float(xs.std())
                    for c in v:
                        key[id(c)] = (raw[id(c)] - mu) / sd if sd > 0 else 0.0
    elif strategy in ("central", "crossenc"):
        sc = extra.get(strategy) or {}
        if not sc:
            return []
        key = {id(c): sc.get(id(c), -math.inf) for c in cands}
    else:
        raise ValueError(strategy)
    return sorted(cands, key=lambda c: (-key[id(c)], c["rank"], salt(c)))


def _metrics(order: list[dict[str, Any]]) -> dict[str, float]:
    first = next((i for i, c in enumerate(order) if c["evidence"]), None)
    if first is None:
        return {}
    tokens_before = float(sum(int(c.get("tokens") or 0) for c in order[:first]))
    out = {"rr": 1.0 / (first + 1), "tokens_before": tokens_before}
    for k in (1, 3, 5, 10, 20):
        out[f"r@{k}"] = float(first < k)
    for t in BUDGETS:
        out[f"budget@{t}"] = float(tokens_before + int(order[first].get("tokens") or 0) <= t)
    return out


def study(layout: str, dataset: str, kinds: tuple[str, ...] = ("question",), text_rerankers: bool = True
          ) -> dict[str, Any]:
    lists = load_lists(layout, kinds)
    qids = sorted({q for q, _, _ in lists})
    fold = {q: stable_int(f"fold|{q}") % 2 for q in qids}
    every = [c for v in lists.values() for c in v]
    extra_base: dict[str, Any] = {}
    questions = _questions(dataset)
    if "question" in kinds:
        bg = _backgrounds({q: questions[q] for q in qids if q in questions}, {c.get("encoder") for c in every})
        raw = {}
        for c in every:
            st = bg.get((c["qid"], c.get("encoder")))
            if st is not None and c.get("kind") == "question" and c.get("z") is not None:
                raw[id(c)] = float(c["z"]) * st[1] + st[0]
        extra_base["raw"] = raw
    titles: dict[str, str] = {}
    if text_rerankers and "question" in kinds:
        texts = _objects(dataset, {c["oid"] for c in every if c.get("modality") != "image"}, titles)
        pairs = [(c, questions.get(c["qid"]), texts.get(c["oid"])) for c in every]
        ok = [(c, q, t) for c, q, t in pairs if q and t]
        if ok:
            from fedquest.vdb.encoders import load_encoder

            enc = load_encoder("bge-base")
            uq = sorted({q for _, q, _ in ok})
            qv = dict(zip(uq, enc.encode_queries(uq), strict=True))
            ut = sorted({t for _, _, t in ok})
            tv = dict(zip(ut, enc.encode_docs(ut), strict=True))
            extra_base["central"] = {id(c): float(np.dot(qv[q], tv[t]) / (np.linalg.norm(qv[q]) * np.linalg.norm(tv[t])
                                                                           + 1e-9)) for c, q, t in ok}
            try:
                from sentence_transformers import CrossEncoder

                ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
                scores = ce.predict([(q, t) for _, q, t in ok], batch_size=128, show_progress_bar=False)
                extra_base["crossenc"] = {id(c): float(s) for (c, _, _), s in zip(ok, scores, strict=True)}
            except Exception:
                pass
    if not titles and "question" in kinds:
        _objects(dataset, {c["oid"] for c in every if c.get("modality") != "image"}, titles)
    extra_base["titles"] = {id(c): title_feats(questions.get(c["qid"], ""), titles.get(c["oid"], "")) for c in every}
    results: dict[str, dict[str, list[float]]] = {s: defaultdict(list) for s in STRATEGIES}
    for f in (0, 1):
        train = [c for (q, _, _), v in lists.items() if fold[q] != f for c in v]
        extra = {**extra_base, **{f"platt_{s}": _fit_platt(train, names, extra_base) for s, names in FEATURES.items()
                                  if s != "fused" or extra_base.get("crossenc")}}
        for (q, _rnd, _kind), cands in lists.items():
            if fold[q] != f:
                continue
            for s in STRATEGIES:
                order = _order(s, cands, extra)
                if not order:
                    continue
                for k, v in _metrics(order).items():
                    results[s][k].append(v)
    out: dict[str, Any] = {"layout": layout, "dataset": dataset, "kinds": list(kinds), "lists": len(lists),
                           "questions": len(qids), "strategies": {},
                           "text_coverage": round(len(extra_base.get("central", {})) / max(1, len(every)), 4)}
    for s, ms in results.items():
        if not ms.get("rr"):
            continue
        tb = np.asarray(ms["tokens_before"])
        out["strategies"][s] = {"n": len(ms["rr"]), "mrr": round(float(np.mean(ms["rr"])), 4),
                                **{k: round(float(np.mean(v)), 4) for k, v in ms.items() if k.startswith(("r@",
                                                                                                         "budget@"))},
                                "tokens_before_mean": round(float(tb.mean()), 1),
                                "tokens_before_p90": round(float(np.percentile(tb, 90)), 1)}
    return out


DEFAULT = [("musique_s2_r0", "musique"), ("musique_s4_r0", "musique"), ("musique_s4_k05_r0", "musique"),
           ("musique_s8_r0", "musique"), ("musique_s16_r0", "musique"), ("musique_s32_r0", "musique"),
           ("mmqa_s2_r0", "mmqa"), ("mmqa_s4_r0", "mmqa"), ("mmqa_s4_k05_r0", "mmqa"),
           ("twowiki_s2_r0", "twowiki"), ("hotpot_s2_r0", "hotpotqa")]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m fedquest.eval.merge_study")
    ap.add_argument("--layouts", nargs="*", default=None, help="layout:dataset pairs (default: every probed layout)")
    ap.add_argument("--no-text", action="store_true", help="skip the central re-embedding and the cross-encoder")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    todo = [tuple(x.split(":", 1)) for x in args.layouts] if args.layouts else DEFAULT
    out_dir = args.out or runs_root() / "results" / "merge_study"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_res = []
    lines = ["# Result merging across heterogeneous databases (fedquest.eval.merge_study)", ""]
    for layout, dataset in todo:
        for kinds in (("question",), ("bound", "bridge")):
            try:
                res = study(layout, dataset, kinds, text_rerankers=not args.no_text)
            except FileNotFoundError as e:
                print(f"skip {layout}: {e}")
                break
            all_res.append(res)
            lines += [f"## {layout} ({'+'.join(kinds)} keys; {res['lists']} lists, {res['questions']} questions)", "",
                      "| strategy | MRR | R@1 | R@5 | R@10 | tokens before 1st evidence (mean / p90) | found within "
                      "500 / 1000 tokens |", "|---|---|---|---|---|---|---|"]
            for s, m in res["strategies"].items():
                lines.append(f"| {s} | {m['mrr']:.3f} | {m['r@1']:.3f} | {m['r@5']:.3f} | {m['r@10']:.3f} | "
                             f"{m['tokens_before_mean']:.0f} / {m['tokens_before_p90']:.0f} | "
                             f"{m['budget@500']:.3f} / {m['budget@1000']:.3f} |")
            lines.append("")
    (out_dir / "merge_study.json").write_text(json.dumps(all_res, indent=1))
    (out_dir / "merge_study.md").write_text("\n".join(lines) + "\n")
    print(f"{len(all_res)} studies: {out_dir}/merge_study.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
