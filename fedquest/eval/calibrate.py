# score calibration: probe the calib questions, fit or refit the observations
from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Sequence
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from fedquest.config import MatcherCfg, load_federation
from fedquest.coordinator import Coordinator, build_matcher
from fedquest.data import m3sciqa, mmqa, musique
from fedquest.eval.runner import load_questions, open_federations
from fedquest.federation.base import expected_site_models, require_current
from fedquest.kg.view import open_view
from fedquest.llm.client import LLMClient
from fedquest.methods.base import build_method
from fedquest.paths import REPO, calibration_path, collection_dir
from fedquest.planner.calibration import LABEL, LABEL_V3, LABEL_V4, LENGTHS, SCHEMA, Calibrator, length
from fedquest.types import Question
from fedquest.utils import gather_limited, read_jsonl, write_json, write_jsonl

log = logging.getLogger(__name__)

EXPLORE_SHAPE: dict[str, Any] = {"probe": "topk", "limits": "fixed", "graph": "mgrag", "mention": 1}


def attach_gold_objects(federation: str, questions: list[Question], split: str) -> None:
    cfg = load_federation(federation)
    if cfg.dataset in ("musique", "hotpotqa", "twowiki", "m3sciqa"):
        explore_gold = m3sciqa.explore_gold if cfg.dataset == "m3sciqa" else musique.explore_gold
        for q in questions:
            gold, values = explore_gold(q)
            q.meta.update(gold=gold, gold_tables_unknown=[], gold_values=values)
        return
    if cfg.dataset != "mmqa":
        return
    raw = mmqa.load_raw(cfg.questions.calib_split if split == "calib" else cfg.questions.split)
    for q in questions:
        if q.qid in raw:
            gold, unknown = mmqa.gold_objects(raw[q.qid])
            q.meta.update(gold=gold, gold_tables_unknown=unknown)


def attach_lists(federation: str, questions: list[Question]) -> None:
    cfg = load_federation(federation)
    wanted = {d for q in questions for d in q.meta.get("image_doc_ids") or []}
    titles: dict[str, str] = {}
    for s in cfg.sites if wanted else []:
        for c in s.collections:
            path = collection_dir(cfg.data_name, s.name, c.name) / "objects.jsonl"
            if c.modality == "image" and path.exists():
                titles.update({d["doc_id"]: d["title"] for d in read_jsonl(path) if d["doc_id"] in wanted})
    for q in questions:
        q.meta["list_titles"] = list(dict.fromkeys(titles[d] for d in q.meta.get("image_doc_ids") or [] if d in titles))


async def calibrate(federation: str, *, mode: str = "local", limit: int | None = None,
                    concurrency: int = 16, k: int | None = None, probe: str = "topk", limits: str = "fixed",
                    graph: str = "mgrag", mention: int = 1, scores: bool | None = None,
                    split: str = "calib") -> Path:
    if scores is None:
        scores = os.environ.get("FEDQUEST_CALIBRATE_SCORES") == "1"
    fed_cfg = load_federation(federation)
    questions = load_questions(federation, split, limit)
    attach_gold_objects(federation, questions, split)
    attach_lists(federation, questions)
    async with AsyncExitStack() as stack:
        llm = LLMClient.from_env()
        stack.push_async_callback(llm.aclose)
        fed = await open_federations(mode, llm, stack).get(federation)
        if mode != "local":
            await require_current(fed, fed_cfg, expected_models=expected_site_models(fed_cfg, llm.cfg))
        else:
            await require_current(fed, fed_cfg)
        coord = Coordinator(fed, llm, open_view("memory"), build_matcher(MatcherCfg(encoder="hash")),
                            fed_cfg.answer_style)
        shape = {n: v for n, v in (("probe", probe), ("limits", limits), ("graph", graph), ("mention", mention))
                 if v != EXPLORE_SHAPE[n]}
        method = build_method("explore", {**({"k": int(k)} if k else {}), **shape, "layout": federation,
                                          **({"scores": True} if scores else {})})
        results = await gather_limited(questions, lambda q: coord.run(method, q, f"calibrate/{federation}/{q.qid}"),
                                       concurrency)
    obs = [{key: v for key, v in e.items() if key != "event"} for r in results for e in r.trace
           if e.get("event") == "obs"]
    failed = [r.error for r in results if r.error]
    unbuilt = [e for e in failed if e.startswith("NotImplementedError")]
    if unbuilt:
        raise NotImplementedError(f"{len(unbuilt)} of {len(results)} calibration questions of {federation} need code "
                                  f"not built yet: {unbuilt[0]}")
    if failed:
        log.warning("%d of %d calibration questions failed (first: %s)", len(failed), len(results), failed[0])
    return _fit(federation, obs, len(questions), failed, depth=int(k) if k else None, shape=shape)


def refit_calibration(federation: str, questions: int | None = None) -> Path:
    obs_path = calibration_path(federation).with_suffix(".obs.jsonl")
    if not obs_path.exists():
        raise FileNotFoundError(f"{obs_path} is missing: run `fedquest calibrate {federation}` first")
    obs = list(read_jsonl(obs_path))
    summary = obs_path.with_suffix("").with_suffix(".summary.json")
    before = json.loads(summary.read_text()) if summary.exists() else {}
    depth = before.get("depth") or probed_depth(obs)
    if questions is None:
        n = before.get("questions", len({o.get("qid") for o in obs}))
        return _fit(federation, obs, n, ["(as before the refit)"] * int(before.get("errors", 0)), depth=depth)
    keep = list(dict.fromkeys(o.get("qid") for o in obs))[:questions]
    name = f"{federation}@{questions}"
    costs = calibration_path(federation).with_suffix(".costs.json")
    if costs.exists():
        calibration_path(name).with_suffix(".costs.json").write_bytes(costs.read_bytes())
    return _fit(name, [o for o in obs if o.get("qid") in set(keep)], len(keep), [], depth=depth)


def _shown(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO.resolve()))
    except ValueError:
        return str(path)


def refit_portable(federation: str | None, out: str | Path, exclude_sites: Sequence[str] = (), *,
                   label: str | None = None, inputs: Sequence[str | Path] | None = None) -> Path:
    if label not in (None, "v4"):
        raise ValueError(f"label {label!r}: a portable refit writes the final design's file (no label) or v4")
    if label == "v4":
        return refit_v4(federation, out, exclude_sites, inputs)
    if federation is None:
        raise ValueError("a portable refit names the federation whose observations it refits")
    obs_path = calibration_path(federation).with_suffix(".obs.jsonl")
    if not obs_path.exists():
        raise FileNotFoundError(f"{obs_path} is missing: run `fedquest calibrate {federation}` first")
    obs = list(read_jsonl(obs_path))
    cands = [o for o in obs if o.get("path") != "gold"]
    unknown = sorted(set(exclude_sites) - {o["site"] for o in cands})
    if unknown:
        raise ValueError(f"{federation}'s observations have no sites {unknown} to exclude")
    out = Path(out)
    if out.suffix != ".json":
        raise ValueError(f"{out}: a calibration file ends in .json (its costs go next to it as .costs.json)")
    depth = probed_depth(obs)
    calib = Calibrator(portable=True)
    labeled = calib.fit(obs, depth=depth, exclude_sites=exclude_sites)
    if not labeled:
        raise RuntimeError(f"no labeled candidates in {obs_path} outside the excluded sites {sorted(exclude_sites)}")
    kept = [o for o in cands if o["site"] not in calib.excluded]
    costs, costs_out = calibration_path(federation).with_suffix(".costs.json"), out.with_suffix(".costs.json")
    command = (f"fedquest calibrate {federation} --refit --portable"
               + (f" --exclude-sites {','.join(calib.excluded)}" if calib.excluded else "") + f" --out {_shown(out)}")
    calib.meta = {
        "command": command, "federation": federation, "depth": depth,
        "observations": f"runs/calibration/{obs_path.name}",
        "observations_sha256": hashlib.sha256(obs_path.read_bytes()).hexdigest(),
        "costs": f"runs/calibration/{costs.name}" if costs.exists() else None,
        "costs_sha256": hashlib.sha256(costs.read_bytes()).hexdigest() if costs.exists() else None,
        "questions": len({o["qid"] for o in obs}), "sites": sorted({o["site"] for o in cands}),
        "candidates": len(cands), "labeled": sum(1 for o in cands if o.get("evidence") is not None),
        "evidence": sum(1 for o in cands if o.get("evidence")),
        "fitted": {"candidates": len(kept), "labeled": labeled, "evidence": sum(1 for o in kept if o.get("evidence"))},
    }
    calib.save(out)
    if costs.exists():
        costs_out.write_bytes(costs.read_bytes())
    else:
        log.warning("%s is missing: %s has no read costs (the planner keeps its defaults)", costs, out)
    log.info("portable calibration of %s saved to %s (%d labeled candidates, %d evidence, strength %g)", federation,
             out, labeled, calib.meta["fitted"]["evidence"], calib.strength)
    return out


def _runs_path(path: Path) -> str:
    home = calibration_path("x").parent.resolve()
    return f"runs/calibration/{path.name}" if path.resolve().parent == home else _shown(path)


def refit_v4(federation: str | None, out: str | Path, exclude_sites: Sequence[str] = (),
             inputs: Sequence[str | Path] | None = None) -> Path:
    if inputs:
        paths = [Path(p) for p in inputs]
    elif federation:
        paths = [calibration_path(federation).with_suffix(".obs.jsonl")]
    else:
        raise ValueError("calibration v4 refits observation files: give inputs or a federation")
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{', '.join(missing)} missing: run `fedquest calibrate <layout>` first")
    if len({p.resolve() for p in paths}) < len(paths):
        raise ValueError("an observation file is given twice: its units would count twice")
    out = Path(out)
    if out.suffix != ".json":
        raise ValueError(f"{out}: a calibration file ends in .json (its costs go next to it as .costs.json)")
    lines = [list(read_jsonl(p)) for p in paths]
    owner: dict[str, Path] = {}
    for path, obs in zip(paths, lines, strict=True):
        for layout in sorted({str(o.get("layout")) for o in obs}):
            if owner.setdefault(layout, path) != path:
                raise ValueError(f"layout {layout} is in {owner[layout]} and {path}: its units would count twice")
    obs = [o for each in lines for o in each]
    cands = [o for o in obs if o.get("path") != "gold"]
    unknown = sorted(set(exclude_sites) - {o["site"] for o in cands})
    if unknown:
        raise ValueError(f"the observations have no sites {unknown} to exclude")
    calib = Calibrator(version=4, portable=True)
    labeled = calib.fit(obs, exclude_sites=exclude_sites)
    if not labeled:
        raise RuntimeError(f"no labeled candidates in {[str(p) for p in paths]} outside the excluded sites "
                           f"{sorted(exclude_sites)}")
    kept = [o for o in cands if o["site"] not in calib.excluded]
    costs = next((c for c in (p.with_suffix("").with_suffix(".costs.json") for p in paths) if c.exists()), None)
    shown = [_runs_path(p) for p in paths]
    command = ("fedquest calibrate --refit --portable --label v4 --inputs " + " ".join(shown) if inputs
               else f"fedquest calibrate {federation} --refit --portable --label v4")
    command += (f" --exclude-sites {','.join(calib.excluded)}" if calib.excluded else "") + f" --out {_shown(out)}"
    calib.meta = {
        "command": command, "label": LABEL_V4, "depth": calib.depth,
        "inputs": [{"observations": name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                    "layouts": sorted({str(o.get("layout")) for o in each}),
                    "k": sorted({int(o["k"]) for o in each if o.get("path") == "gold" and o.get("k")}),
                    "questions": len({o["qid"] for o in each}),
                    "candidates": sum(1 for o in each if o.get("path") != "gold")}
                   for p, name, each in zip(paths, shown, lines, strict=True)],
        "costs": _runs_path(costs) if costs else None,
        "costs_sha256": hashlib.sha256(costs.read_bytes()).hexdigest() if costs else None,
        "questions": len({o["qid"] for o in obs}), "sites": sorted({o["site"] for o in cands}),
        "candidates": len(cands), "labeled": sum(1 for o in cands if o.get("evidence") is not None),
        "evidence": sum(1 for o in cands if o.get("evidence")),
        "fitted": {"candidates": len(kept), "labeled": labeled, "evidence": sum(1 for o in kept if o.get("evidence"))},
    }
    calib.save(out)
    if costs:
        out.with_suffix(".costs.json").write_bytes(costs.read_bytes())
    else:
        log.warning("no input has read costs (<stem>.costs.json): %s has none (the planner keeps its defaults)", out)
    log.info("calibration v4 saved to %s (%d labeled candidates of %d layouts, %d evidence, strength %g)", out,
             labeled, len(owner), calib.meta["fitted"]["evidence"], calib.strength)
    return out


def probed_depth(obs: list[dict[str, Any]]) -> int | None:
    ranks = [int(o["rank"]) for o in obs if o.get("path") != "gold" and o.get("kind") in ("question", "bound", "bridge")]
    return max(ranks) if ranks else None


COVERAGE_DEPTHS = (1, 3, 5, 10, 20, 40)


def _fit(federation: str, obs: list[dict[str, Any]], questions: int, failed: list[str],
         depth: int | None = None, shape: dict[str, str] | None = None) -> Path:
    path = calibration_path(federation)
    calib = Calibrator()
    labeled = calib.fit(obs, depth=depth)
    if not labeled:
        raise RuntimeError(f"no labeled candidates while calibrating {federation} ({len(failed)} questions failed"
                           f"{': ' + failed[0] if failed else ''}): check 'fedquest deploy status'")
    calib.save(path)
    write_jsonl(path.with_suffix(".obs.jsonl"), obs)
    cands = [o for o in obs if o.get("path") != "gold"]
    evidence = sum(1 for o in cands if o.get("evidence"))
    rounds = sorted(k for k in calib.cover if "|" in k)
    snippets = sorted(k for k in calib.snippets if k != "all" and "|" not in k)
    by_depth = {k: {d: round(calib.coverage(k.split("|")[0], k.endswith("|image"), d), 4)
                    for d in COVERAGE_DEPTHS if d <= (calib.depth or 0)} for k in rounds} if calib.version >= 3 else {}
    write_json(path.with_suffix(".summary.json"), {
        "federation": federation, "label": LABEL_V3 if calib.version >= 3 else LABEL, "schema": SCHEMA,
        "depth": calib.depth, "questions": questions, "errors": len(failed),
        "candidates": len(cands), "labeled": labeled, "evidence": evidence, "strength": calib.strength,
        "coverage": {k: round(calib.coverage(k.split("|")[0], k.endswith("|image")), 4) for k in rounds},
        "coverage_by_depth": by_depth,
        "beta": {f"{m}|{length(t)}": round(calib.beta(m, t, t), 4) for m in snippets for t in (0, *LENGTHS)},
        "cells": calib.snapshot(),
        **({"explore": shape} if shape else {}),
    })
    log.info("calibration for %s saved to %s (%d labeled candidates, %d evidence, strength %g)", federation, path,
             labeled, evidence, calib.strength)
    return path
