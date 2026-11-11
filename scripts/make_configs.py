#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "configs" / "experiments"
FEDERATIONS = ROOT / "configs" / "federations"
MODELS = ROOT / "configs" / "models"
TOOLS_FILE = ROOT / "configs" / "tools.json"
BENCHES = ("mmqa", "musique", "twowiki", "hotpot")
TEXT = ("musique", "twowiki", "hotpot")
TITLE = {"mmqa": "MMQA", "musique": "MuSiQue", "twowiki": "2Wiki", "hotpot": "HotpotQA"}
FAMILY = {"mmqa": "mmqa", "musique": "musique", "twowiki": "musique", "hotpot": "musique"}
FED2 = {b: f"{b}_s2_r0" for b in BENCHES}
MAIN_FED = {b: f"{b}_s4_k05_r0" for b in BENCHES}
HIPPO = {"musique": "musique_next_home", "twowiki": "twowiki_home", "hotpot": "hotpot_home"}
HIPPO_WAIT = ["python", "-m", "fedquest.site.hippo", "wait", "--remote", "--timeout", "21600"]
MAIN_SPLIT = {"mmqa": "full", "musique": "test", "twowiki": "full", "hotpot": "full"}
POOL = {"mmqa": "MMQA's dev split", "musique": "HippoRAG 2's MuSiQue dev questions",
        "twowiki": "HippoRAG 2's 2Wiki dev questions", "hotpot": "HippoRAG 2's HotpotQA dev questions"}
MAIN_N = {"mmqa": 2441, "musique": 1000, "twowiki": 1000, "hotpot": 1000}
ANALYSIS_N = 300
HELD = "heldout"
HELD_MUSIQUE_N = 365
SUB = dict.fromkeys(BENCHES, ANALYSIS_N)
SCALE_N = ANALYSIS_N
MX_N = ANALYSIS_N
LAT_N = ANALYSIS_N
KGU_N = ANALYSIS_N
STREAM_B = 32
CONC = 24
MAIN_CONC = {"mmqa": 64, "musique": 32, "twowiki": 32, "hotpot": 32}
XFIT = ("twowiki", "hotpot")
XFIT_ROUTE = "{federation}_x{x}"
XFIT_ROUTER = "x{x}"
FEDGRAG_CHECKS = ("fedgrag_tuned", "fedgrag_trained")

MAIN_B = 16000
FRONTIER_B = (4000, 8000, 32000)
IRCOT_TOPS = (4, 8)
SCALE_BENCH = "musique"
SCALE_SITES = (2, 4, 8, 16, 32)
SCALE_SEEDS = (0, 1, 2, 3, 4)

NODES = 2
WALLTIME = {"aurora": "71:59:00", "midway3": "24:00:00"}
GRID_NODES = (1, 2, 4)
GRID_PER_NODE = (1, 2, 4, 8)
GRID_MAX = 32
SCALE_NODES = max(GRID_NODES)
SCALE_WALLTIME = "06:00:00"
SCALE_QUEUE_AURORA, SCALE_WALLTIME_AURORA = "capacity", SCALE_WALLTIME

METIS = "https://inference-api.alcf.anl.gov/resource_server/metis/api/v1"
SOPHIA = "https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1"
MINERVA = "https://inference-api.alcf.anl.gov/resource_server/minerva/api/v1"
TARA = "https://inference-api.alcf.anl.gov/resource_server/tara/api/v1"
GPTOSS_POOL = f"pool:{TARA},{SOPHIA}"
NEMOTRON = TARA
GEMMA = "google/gemma-4-31B-it"
BASE = SOPHIA
JUDGE = "openai/gpt-oss-120b"
JUDGE2 = "nemotron-3-ultra"
ROLES_TEXT = ("pattern", "extract_text", "answer", "planner", "reader")
ROLES_IMAGE = ("extract_image", "answer_vision")
CELLS: dict[str, tuple[str, str, dict[str, Any] | None, int | None, bool, int]] = {
    "gemma4_e4b": ("google/gemma-4-E4B-it", SOPHIA, None, None, False, 8),
    "gptoss_20b": ("openai/gpt-oss-20b", GPTOSS_POOL, {"reasoning_effort": "low"}, 512, False, 16),
    "gptoss_120b": ("openai/gpt-oss-120b", GPTOSS_POOL, {"reasoning_effort": "low"}, 512, False, 16),
    "llama31_8b": ("meta-llama/Meta-Llama-3.1-8B-Instruct", SOPHIA, None, None, True, 8),
    "llama31_70b": ("meta-llama/Meta-Llama-3.1-70B-Instruct", SOPHIA, None, None, True, 8),
    "nemotron3_ultra": ("nemotron-3-ultra", NEMOTRON, {"chat_template_kwargs": {"enable_thinking": False}},
                        4096, False, 8),
}
COMMON = {"timeout_s": 330, "max_retries": 8, "token_refresh_s": 900}


def top_models() -> dict[str, Any]:
    roles = {r: GEMMA for r in (*ROLES_TEXT, *ROLES_IMAGE)} | {"judge": JUDGE, "judge2": JUDGE2}
    profiles: dict[str, Any] = {"latency": {"model_concurrency": {GEMMA: 16, JUDGE: 16, JUDGE2: 8}}}
    for cell, (model, url, extra, headroom, json_mode, conc) in CELLS.items():
        p: dict[str, Any] = {"roles": {r: model for r in ROLES_TEXT}}
        if url != BASE:
            p["endpoints"] = {model: {"provider": "alcf", "base_url": url}}
        if extra:
            p["extra_body"] = {model: extra}
        if headroom:
            p["headroom"] = {model: headroom}
        if json_mode:
            p["json_mode"] = [model]
        p["model_concurrency"] = {GEMMA: 16, model: conc, JUDGE: 16}
        p["max_concurrency"] = 32
        profiles[f"mx_{cell}"] = p
    return {"provider": "alcf", "base_url": BASE, "trust": "alcf", "roles": roles,
            "endpoints": {JUDGE: {"provider": "alcf", "base_url": GPTOSS_POOL},
                          JUDGE2: {"provider": "alcf", "base_url": NEMOTRON}},
            "extra_body": {JUDGE: {"reasoning_effort": "low"},
                           JUDGE2: {"chat_template_kwargs": {"enable_thinking": False}}},
            "headroom": {JUDGE2: 1024},
            "model_concurrency": {GEMMA: 32, JUDGE: 32, JUDGE2: 16}, "max_concurrency": 64, **COMMON,
            "profiles": profiles}


def cell_models(cell: str) -> dict[str, Any]:
    model, url, extra, headroom, json_mode, conc = CELLS[cell]
    top = top_models()
    out = {k: v for k, v in top.items() if k != "profiles"}
    out["roles"] = {**top["roles"], **{r: model for r in ROLES_TEXT}}
    if url != BASE:
        out["endpoints"] = {**top["endpoints"], model: {"provider": "alcf", "base_url": url}}
    if extra:
        out["extra_body"] = {**top["extra_body"], model: extra}
    if headroom:
        out["headroom"] = {**top["headroom"], model: headroom}
    if json_mode:
        out["json_mode"] = [model]
    out["model_concurrency"] = {**top["model_concurrency"], model: conc}
    return out


def spec(method: str, label: str | None = None, federation: str | None = None, preset: str | None = None,
         **params: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"method": method}
    for key, value in (("label", label), ("federation", federation), ("preset", preset)):
        if value is not None:
            out[key] = value
    if params:
        out["params"] = params
    return out


def name(s: dict[str, Any]) -> str:
    return s.get("label") or s["method"]


def at(s: dict[str, Any], federation: str, suffix: str) -> dict[str, Any]:
    params = dict(s.get("params", {}))
    if s["method"].startswith("fedgrag"):
        params["index"] = federation
    return spec(s["method"], f"{name(s)}_{suffix}", federation, s.get("preset"), **params)


def kb(b: int) -> str:
    return f"{b // 1000}k"


def lay(n: int, seed: int = 0) -> str:
    return f"{SCALE_BENCH}_s2_r0" if n == 2 else f"{SCALE_BENCH}_s{n}_k05_r{seed}"


def lsfx(n: int, seed: int = 0) -> str:
    return lay(n, seed).removeprefix(f"{SCALE_BENCH}_")


def fq(bench: str, label: str = "fedquest", budget: int = MAIN_B, method: str = "fedquest",
       **params: Any) -> dict[str, Any]:
    fam = FAMILY[bench]
    base: dict[str, Any] = {"calibration": f"{fam}_v4e", "scores": "anchor", "zcal": f"{fam}_z", "views": True,
                            "total_budget": budget, "route": "vocab+proto"}
    if bench in XFIT:
        base["route_fit"] = XFIT_ROUTE
    return spec(method, label, None, "final", **{**base, **params})


def ircot(bench: str, top: int, budget: int) -> dict[str, Any]:
    method = "ircot_mm_b" if bench == "mmqa" else "ircot_b"
    return spec(method, f"ircot_t{top}@{kb(budget)}", None, None, top=top, k=top, total_budget=budget)


def ircot_ref(bench: str) -> dict[str, Any]:
    if bench == "mmqa":
        return spec("iter_rag_mm", "ircot@8x8", None, "ircot", order="relevance", budget=None, hops=8, top=8, k=8,
                    max_paragraphs=48, images=4, image_px=512)
    return spec("iter_rag", "ircot@8x8", None, "ircot", order="relevance", budget=None, hops=8, top=8, k=8,
                max_paragraphs=48)


def react(bench: str, budget: int) -> dict[str, Any]:
    return spec("react", f"react@{kb(budget)}", total_budget=budget)


def rewoo(bench: str, budget: int) -> dict[str, Any]:
    return spec("rewoo", f"rewoo@{kb(budget)}", total_budget=budget)


def react_ops(bench: str, budget: int) -> dict[str, Any]:
    return spec("llm_planner_ops", f"react_ops@{kb(budget)}", None, "ops", max_steps=16, views=True,
                total_budget=budget, tokens=None)


def mgrag_fu(bench: str, budget: int) -> dict[str, Any]:
    fam = FAMILY[bench]
    img = {"answer_images": True} if bench == "mmqa" else {}
    return spec("mgrag_fu", f"mgrag_fu@{kb(budget)}", None, "fu_alt", probe="semijoin", limits="share", mention=2,
                ties="salted", calibration=f"{fam}_v4", total_budget=budget, **img)


def baselines(bench: str, federation: str) -> dict[str, dict[str, Any]]:
    img = {"answer_images": True} if bench == "mmqa" else {}
    out = [spec("mgrag", "mgrag_fed@8", None, "mgrag", budget=8, **img),
           spec("vanilla_rag", "vanilla_rag@8", order="relevance", budget=8),
           spec("closed_book"),
           ircot_ref(bench)]
    if bench in TEXT:
        own = {"router": XFIT_ROUTER} if bench in XFIT else {}
        out += [spec("fedgrag", "fedgrag", index=federation, **own),
                spec("fedgrag_text", "fedgrag_text", index=federation, **own)]
    return {name(s): s for s in out}


def dgrag(bench: str, budget: int = MAIN_B) -> dict[str, Any]:
    return spec("dgrag", f"dgrag@{kb(budget)}", index=MAIN_FED[bench], total_budget=budget)


def unfitted(s: dict[str, Any]) -> dict[str, Any]:
    params = {k: v for k, v in s.get("params", {}).items() if k not in ("route_fit", "router")}
    return {**{k: v for k, v in s.items() if k != "params"}, **({"params": params} if params else {})}


def budgeted(bench: str, budget: int) -> list[dict[str, Any]]:
    lab = "fedquest" if budget == MAIN_B else f"fedquest@{kb(budget)}"
    out = [fq(bench, lab, budget), *[ircot(bench, t, budget) for t in IRCOT_TOPS]]
    if budget >= 8000:
        out += [react(bench, budget), rewoo(bench, budget), mgrag_fu(bench, budget)]
    if budget == MAIN_B:
        out.append(react_ops(bench, budget))
    return out


def ablations(bench: str) -> list[dict[str, Any]]:
    out = [fq(bench, "fedquest_q", method="fedquest_q", scan=True),
           fq(bench, "fedquest_verify", direct="budget"),
           fq(bench, "fedquest_ctx1500", context_max=1500)]
    if bench in ("mmqa", "musique"):
        out += [fq(bench, "fedquest_dense", probe="dense", limits="fixed"),
                fq(bench, "fedquest_norecover", direct="off", repair=False),
                fq(bench, "fedquest_mgraph", graph="mgrag")]
    return out


def kgu_arms(b: str) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    ns = f"kgu_{b}"
    warm = [fq(b, "fedquest_kg_warm", kg="harvest", kg_check="version", kg_ns=ns, keep_probe=True)]
    keep = {"kg_ns": ns, "epoch": "u1", "kg_write": False, "keep_probe": True}
    after = [fq(b, "fedquest_kg_trust", kg="reuse", kg_check="trust", **keep),
             fq(b, "fedquest_kg_version", kg="reuse", kg_check="version", **keep),
             fq(b, "fedquest_kg_witness", kg="reuse", kg_check="witness", **keep),
             fq(b, "fedquest_kg_version_noprobe", kg="reuse", kg_check="version", kg_ns=ns, epoch="u1", kg_write=False),
             fq(b, "fedquest_cold", epoch="u1")]
    return ns, warm, after


def grid_cells() -> list[tuple[str, int, int, int]]:
    out = []
    for n in GRID_NODES:
        for c in GRID_PER_NODE:
            total = 2 * n * c
            if total <= GRID_MAX:
                out.append((f"{lay(total)}_n{n}", total, n, c))
    return out


def cell_layout(layout: str, nodes: int) -> tuple[str, str]:
    src = (FEDERATIONS / f"{layout}.yaml").read_text()
    dst = f"{layout}_n{nodes}"
    body = re.sub(r"^#.*\n", "", src, flags=re.M).replace(f"name: {layout}", f"name: {dst}", 1)
    return dst, body


def cell_files() -> dict[Path, str]:
    files: dict[Path, str] = {}
    for fed, total, nodes, _ in grid_cells():
        dst, text = cell_layout(lay(total), nodes)
        assert dst == fed
        files[FEDERATIONS / f"{dst}.yaml"] = text
    return files


def scale_plan() -> list[dict[str, Any]]:
    trace = "$FEDQUEST_RUNS/traces/scale_musique.jsonl"
    out = []
    for fed, total, nodes, per in grid_cells():
        for label, mode in (("fedquest", "routed"), ("fedquest_all", "broadcast"), ("fedquest_own", "owners")):
            out.append({"tag": f"n{nodes}_c{per}_{mode}_k32", "federation": fed, "trace": trace,
                        "label": f"{label}_{lsfx(total)}", "clients": 32, "limit": SCALE_N, "warmup": SCALE_N,
                        "nodes": nodes, "per_node": per, "databases": total, "routed": mode != "broadcast",
                        "mode": mode})
    return out


def seeded(federation: str) -> bool:
    return re.search(r"_r[1-9]\d*$", federation) is not None


def _labels_on(catalog: list[tuple[str, list[str], dict[str, Any]]]) -> dict[str, list[str]]:
    out: dict[str, set[str]] = {k: set() for k in ("routed", "routed_xfit", "indexed", "indexed_xfit", "dgrag",
                                                   "tuned", "trained")}
    for _, _, cfg in catalog:
        for m in cfg["methods"]:
            fed = m.get("federation") or cfg["federation"]
            params = m.get("params", {})
            if m["method"].startswith("fedquest") and "proto" in str(params.get("route", "")):
                out["routed_xfit" if params.get("route_fit") else "routed"].add(fed)
            if m["method"].startswith("fedgrag"):
                out["indexed_xfit" if params.get("router") else "indexed"].add(params["index"])
            if m["method"] in FEDGRAG_CHECKS:
                out["tuned"].add(params["index"])
            if m["method"] == "fedgrag_trained":
                out["trained"].add(params["index"])
            if m["method"] == "dgrag":
                out["dgrag"].add(params["index"])
    return {k: sorted(v) for k, v in out.items()}


def _pool_split(federation: str) -> str:
    from fedquest.config import load_federation

    dataset = load_federation(federation).dataset
    bench = {"hotpotqa": "hotpot"}.get(dataset, dataset)
    return MAIN_SPLIT[bench]


def tools(catalog: list[tuple[str, list[str], dict[str, Any]]]) -> list[dict[str, Any]]:
    need = _labels_on([c for c in catalog if not c[0].startswith("smoke_")])
    routed, indexed = need["routed"], need["indexed"]
    out: list[dict[str, Any]] = []

    def add(n: str, argv: list[str], lane: str, after: list[str] | None = None, **kw: Any) -> None:
        out.append({"id": f"tool:{n}", "argv": argv, "lane": lane, "after": after or [], **kw})

    for suffix, pick, after in (("", False, []), ("_seeds", True, ["tool:route"])):
        feds = [f for f in routed if seeded(f) == pick]
        if feds:
            add(f"route{suffix}", ["python", "-m", "fedquest.routing", "calibrate", *feds, "--split", "calib", "--mode",
                                   "remote"], "probe", after, optional=True)
    splits = sorted({_pool_split(f) for f in [*need["routed_xfit"], *need["indexed_xfit"]]})
    if len(splits) > 1:
        raise ValueError(f"cross-fitted layouts on several pools ({splits}): one route_xfit and fedgrag_xfit task each")
    for split in splits[:1]:
        feds = [f for f in need["routed_xfit"] if _pool_split(f) == split]
        if feds:
            add("route_xfit", ["python", "-m", "fedquest.routing", "calibrate", *feds, "--split", split, "--halves",
                               "--mode", "remote"], "probe", optional=True)
    for suffix, pick, after in (("", False, []), ("_seeds", True, ["tool:fedgrag_build"])):
        feds = [f for f in indexed if seeded(f) == pick]
        if feds:
            add(f"fedgrag{suffix or '_build'}", ["python", "-m", "fedquest.fedgrag", "build", *feds, "--missing",
                                                 "--mode", "remote"], "cpu", after, optional=True)
    for split in splits[:1]:
        feds = [f for f in need["indexed_xfit"] if _pool_split(f) == split]
        if feds:
            add("fedgrag_xfit", ["python", "-m", "fedquest.fedgrag", "build", *feds, "--halves", "--split", split,
                                 "--mode", "remote"], "cpu", optional=True)
    src = MAIN_FED[SCALE_BENCH]
    built = [t["id"] for t in out if t["id"] in ("tool:fedgrag_build", "tool:fedgrag_seeds", "tool:fedgrag_xfit")]

    def source_first(feds: list[str]) -> list[str]:
        return sorted(feds, key=lambda f: (f != src, f))

    if need["trained"]:
        add("fedgrag_trained", ["python", "-m", "fedquest.fedgrag", "trained", *source_first(need["trained"]),
                                "--encoder-from", src, "--missing"], "cpu", built, optional=True)
    if need["tuned"]:
        add("fedgrag_tune", ["python", "-m", "fedquest.fedgrag", "tune", *source_first(need["tuned"]), "--from", src],
            "cpu", built, optional=True)
    if need["trained"]:
        add("fedgrag_tune_trained", ["python", "-m", "fedquest.fedgrag", "tune", *source_first(need["trained"]),
                                     "--encoder", "trained", "--from", src], "cpu", ["tool:fedgrag_trained"],
            optional=True)
    if need["dgrag"]:
        add("dgrag_build", ["python", "-m", "fedquest.dgrag", "build", *need["dgrag"], "--missing", "--mode", "remote"],
            "cpu", [t["id"] for t in out if t["id"] in ("tool:fedgrag_build", "tool:fedgrag_xfit")], optional=True)
    for b in ("mmqa", "musique"):
        add(f"kgu_make_{b}", ["python", "-m", "fedquest.eval.kg_update", "make", "--federation", FED2[b], "--split",
                              "final_test", "--n", str(KGU_N)], "cpu")
    add("live_free", ["bash", "scripts/calibrate.sh", "free"], "cpu", optional=True)
    add("merge_study", ["python", "-m", "fedquest.eval.merge_study"], "cpu", optional=True)
    add("routing_study", ["python", "-m", "fedquest.eval.routing_study"], "cpu",
        [t["id"] for t in out if t["id"] in ("tool:route", "tool:route_xfit", "tool:fedgrag_build",
                                              "tool:fedgrag_xfit")], optional=True)
    return out


def post_tools(names: list[str]) -> list[dict[str, Any]]:
    return [{"id": "tool:post", "argv": ["bash", "scripts/post.sh"], "lane": "post", "after": list(names),
             "after_any": True}]


def experiments() -> list[tuple[str, list[str], dict[str, Any]]]:
    out: list[tuple[str, list[str], dict[str, Any]]] = []

    def add(name_: str, methods: list[dict[str, Any]], **cfg: Any) -> None:
        names = [name(s) for s in methods]
        if len(names) != len(set(names)):
            raise ValueError(f"{name_}: a label twice ({sorted(n for n in names if names.count(n) > 1)})")
        out.append((name_, [], {"name": name_, "workload": "cold", "lane": "gemma", **cfg,
                                  "require_calibration": True, "methods": methods}))

    def needs(methods: list[dict[str, Any]]) -> list[str]:
        out_ = []
        for m in methods:
            fed = m.get("federation") or ""
            params = m.get("params", {})
            if m["method"].startswith("fedquest") and "proto" in str(params.get("route", "")):
                out_.append("tool:route_xfit" if params.get("route_fit") else
                            "tool:route_seeds" if seeded(fed) else "tool:route")
            if m["method"].startswith("fedgrag"):
                out_.append("tool:fedgrag_xfit" if params.get("router") else
                            "tool:fedgrag_seeds" if seeded(params["index"]) else "tool:fedgrag_build")
            if m["method"] in FEDGRAG_CHECKS:
                out_.append("tool:fedgrag_tune_trained" if m["method"] == "fedgrag_trained" else "tool:fedgrag_tune")
            if m["method"] == "dgrag":
                out_.append("tool:dgrag_build")
        return sorted(set(out_))

    for b in ("mmqa", "musique", "hotpot"):
        base = baselines(b, MAIN_FED[b])
        smoke = [unfitted(s) for s in (*budgeted(b, MAIN_B), *base.values())]
        if b == SCALE_BENCH:
            smoke += [at(fq(b), lay(32), lsfx(32)), at(base["fedgrag"], lay(32), lsfx(32))]
        add(f"smoke_{b}", smoke, federation=MAIN_FED[b], split="calib", limit=2, concurrency=8, stage=0, judge=False)
    for b in BENCHES:
        methods = [*budgeted(b, MAIN_B), *baselines(b, MAIN_FED[b]).values()]
        add(f"main_{b}", methods, federation=MAIN_FED[b], split=MAIN_SPLIT[b], limit=MAIN_N[b], concurrency=MAIN_CONC[b],
            interleave=True, stage=1, after=needs(methods),
            env={"FEDQUEST_RECORD_WIRE": f"$FEDQUEST_RUNS/cache/wire_main_{b}.sqlite"})
    for b in TEXT:
        methods = [dgrag(b)]
        add(f"dgrag_{b}", methods, federation=MAIN_FED[b], split=MAIN_SPLIT[b], limit=MAIN_N[b], concurrency=32, stage=1,
            after=needs(methods))
    b = "musique"
    methods = [*budgeted(b, MAIN_B), *baselines(b, MAIN_FED[b]).values(), dgrag(b)]
    add("heldout_musique", methods, federation=MAIN_FED[b], split=HELD, limit=HELD_MUSIQUE_N, concurrency=MAIN_CONC[b], interleave=True,
        stage=1, after=needs(methods))
    for b in BENCHES:
        methods = [m for x in FRONTIER_B for m in budgeted(b, x)]
        add(f"frontier_{b}", methods, federation=MAIN_FED[b], split=MAIN_SPLIT[b], limit=SUB[b], concurrency=CONC, interleave=True,
            stage=2, after=needs(methods))
    for b in BENCHES:
        methods = ablations(b)
        add(f"abl_{b}", methods, federation=MAIN_FED[b], split=MAIN_SPLIT[b], limit=SUB[b], concurrency=CONC, interleave=True,
            stage=2, after=needs(methods))
    b = SCALE_BENCH
    base4 = baselines(b, MAIN_FED[b])
    sched = [fq(b, "fedquest_rank", scores="rank", zcal=None), fq(b, "fedquest_ranknaive", scores="ranknaive",
                                                                  zcal=None),
             fq(b, "fedquest_e5held", zcal=f"{FAMILY[b]}_z_noe5")]
    scale: list[dict[str, Any]] = []
    for n in SCALE_SITES:
        fed, sfx = lay(n), lsfx(n)
        scale += [at(fq(b), fed, sfx), at(fq(b, "fedquest_all", route="all"), fed, sfx),
                  at(fq(b, "fedquest_own", owners="titles"), fed, sfx),
                  at(base4["fedgrag"], fed, sfx), at(base4["vanilla_rag@8"], fed, sfx),
                  at(base4["mgrag_fed@8"], fed, sfx), at(ircot(b, 8, MAIN_B), fed, sfx)]
        if n == SCALE_SITES[-1]:
            scale += [at(s, fed, sfx) for s in sched]
    add("scale_musique", scale, federation=MAIN_FED[b], split=MAIN_SPLIT[b], limit=SCALE_N, concurrency=24, interleave=True, stage=1,
        after=needs(scale),
        env={"FEDQUEST_PROBE_TRACE": "$FEDQUEST_RUNS/traces/scale_musique.jsonl",
             "FEDQUEST_PROBE_TRACE_MATCH": r"/fedquest(_all|_own)?_s\d+(_k05)?_r0/"})
    seeds: list[dict[str, Any]] = []
    for n in SCALE_SITES[1:]:
        for sd in SCALE_SEEDS[1:]:
            fed, sfx = lay(n, sd), lsfx(n, sd)
            seeds += [at(fq(b), fed, sfx), at(base4["fedgrag"], fed, sfx), at(base4["vanilla_rag@8"], fed, sfx),
                      at(base4["mgrag_fed@8"], fed, sfx)]
    add("seeds_musique", seeds, federation=MAIN_FED[b], split=MAIN_SPLIT[b], limit=SCALE_N, concurrency=CONC, interleave=True, stage=2,
        after=needs(seeds))
    for b in TEXT:
        own = {"router": XFIT_ROUTER} if b in XFIT else {}
        methods = [spec(m, m, index=MAIN_FED[b], **own) for m in FEDGRAG_CHECKS]
        add(f"fedgrag_check_{b}", methods, federation=MAIN_FED[b], split=MAIN_SPLIT[b], limit=MAIN_N[b], concurrency=CONC, interleave=True,
            stage=2, optional=True, after=needs(methods))
    b = SCALE_BENCH
    checks: list[dict[str, Any]] = []
    for n in SCALE_SITES:
        for sd in SCALE_SEEDS[:1] if n == SCALE_SITES[0] else SCALE_SEEDS:
            fed, sfx = lay(n, sd), lsfx(n, sd)
            checks += [at(spec(m, m, index=MAIN_FED[b]), fed, sfx) for m in FEDGRAG_CHECKS]
    add("fedgrag_check_scale", checks, federation=MAIN_FED[b], split=MAIN_SPLIT[b], limit=SCALE_N, concurrency=CONC, interleave=True,
        stage=2, optional=True, after=needs(checks))
    b = "musique"
    methods = [fq(b, "fedquest_cold"), fq(b, "fedquest_kg", kg="reuse", kg_check="version"),
               fq(b, "fedquest_kg_keep", kg="reuse", kg_check="version", keep_probe=True),
               fq(b, "fedquest_kg_own", kg="reuse", kg_check="version", keep_probe=True, owners="titles+kg"),
               fq(b, "fedquest_own", owners="titles")]
    add("session_musique", methods, federation=MAIN_FED[b], split="sessions", concurrency=STREAM_B, stage=2, kg="neo4j",
        stream_batch=STREAM_B, after=needs(methods))
    for b in ("mmqa", "musique"):
        ns, warm, after_ = kgu_arms(b)
        add(f"kgu_warm_{b}", warm, federation=MAIN_FED[b], split="kg_base", concurrency=CONC, stage=2, kg="neo4j",
            after=[*needs(warm), f"tool:kgu_make_{b}"],
            before=["python", "-m", "fedquest.eval.kg_update", "derive", "--federation", MAIN_FED[b], "--source",
                    FED2[b]])
        add(f"kgu_after_{b}", after_, federation=MAIN_FED[b], split="kg_update", concurrency=CONC, stage=3, kg="neo4j",
            after=[f"kgu_warm_{b}"],
            before=["python", "-m", "fedquest.eval.kg_update", "apply", "--federation", MAIN_FED[b], "--epoch", "u1",
                    "--mode", "remote"])
    for b in BENCHES:
        base = baselines(b, MAIN_FED[b])
        methods = [at(fq(b), FED2[b], "s2_r0"), at(base["mgrag_fed@8"], FED2[b], "s2_r0")]
        if b in HIPPO:
            methods += [at(fq(b), HIPPO[b], "pooled"), at(base["mgrag_fed@8"], HIPPO[b], "pooled")]
        add(f"dist_{b}", methods, federation=MAIN_FED[b], split=MAIN_SPLIT[b], limit=SUB[b], concurrency=CONC, interleave=True,
            stage=2, optional=True, after=needs(methods))
    for b in HIPPO:
        add(f"hippo_{b}", [spec("hipporag2", "hipporag2_pooled", HIPPO[b], k=5, top=5)], federation=MAIN_FED[b], split=MAIN_SPLIT[b],
            limit=MAIN_N[b], concurrency=CONC, stage=3, before=[*HIPPO_WAIT, HIPPO[b]], optional=True, lane="hippo")
    b = "musique"
    add("hippo_heldout_musique", [spec("hipporag2", "hipporag2_pooled", HIPPO[b], k=5, top=5)], federation=MAIN_FED[b], split=HELD,
        limit=HELD_MUSIQUE_N, concurrency=CONC, stage=3, before=[*HIPPO_WAIT, HIPPO[b]], optional=True, lane="hippo")
    for cell in CELLS:
        for b in ("mmqa", "musique"):
            base = baselines(b, MAIN_FED[b])
            methods = [fq(b), ircot(b, 8, MAIN_B), mgrag_fu(b, MAIN_B), base["mgrag_fed@8"]]
            add(f"mx_{cell}_{b}", methods, federation=MAIN_FED[b], split=MAIN_SPLIT[b], limit=MX_N, concurrency=16, interleave=True,
                stage=2, models=f"configs/models/cells/{cell}.yaml", site_profile=f"mx_{cell}", lane="mx",
                optional=True, after=needs(methods))
    quiet = [n for n, _, c in out if c.get("lane") == "gemma" and c.get("stage", 1) in (1, 2, 3) and
             not n.startswith("smoke_") and not c.get("optional")]
    for b in ("mmqa", "musique"):
        base = baselines(b, MAIN_FED[b])
        methods = [fq(b), ircot(b, 8, MAIN_B), react(b, MAIN_B), base["mgrag_fed@8"], base["vanilla_rag@8"]]
        if b in TEXT:
            methods += [base["fedgrag"], dgrag(b)]
        if b == SCALE_BENCH:
            methods += [at(fq(b), lay(n), lsfx(n)) for n in SCALE_SITES[1:]]
        add(f"latency_{b}", methods, federation=MAIN_FED[b], split=MAIN_SPLIT[b], limit=LAT_N, concurrency=4, interleave=True, stage=9,
            lane="latency", live_calls=True, after=[*quiet, *needs(methods)], site_profile="latency")
    return out


ORDER = ("name", "federation", "split", "limit", "outside_prefix", "workload", "concurrency", "require_calibration",
         "interleave", "stage", "lane", "after", "before", "models", "site_profile", "kg", "stream_batch",
         "stream_seed", "live_calls", "env", "judge", "optional", "methods")


def flow(obj: Any) -> str:
    if isinstance(obj, (dict, list)):
        return yaml.safe_dump(obj, default_flow_style=True, sort_keys=False, width=10_000).strip()
    return yaml.safe_dump([obj], default_flow_style=True, width=10_000).strip()[1:-1]


def render(name_: str, cfg: dict[str, Any]) -> str:
    unknown = set(cfg) - set(ORDER)
    if unknown:
        raise KeyError(f"{name_}: keys {sorted(unknown)} are not rendered (ORDER)")
    lines = []
    for key in ORDER:
        value = cfg.get(key)
        if value is None or value == [] or value == {} or (value is False and key != "judge"):
            continue
        if key == "methods":
            lines.append("methods:")
            lines += [f"  - {flow(m)}" for m in value]
        else:
            lines.append(f"{key}: {flow(value)}")
    return "\n".join(lines) + "\n"


def model_files() -> dict[Path, str]:
    out = {MODELS / "final.yaml": yaml.safe_dump(top_models(), sort_keys=False, width=120)}
    for cell in CELLS:
        out[MODELS / "cells" / f"{cell}.yaml"] = yaml.safe_dump(cell_models(cell), sort_keys=False, width=120)
    return out


def tool_file(catalog: list[tuple[str, list[str], dict[str, Any]]]) -> str:
    names = [n for n, _, _ in catalog if not n.startswith("smoke_")]
    return json.dumps({"tools": tools(catalog), "post": post_tools(names)}, indent=1) + "\n"


REMOTE = ROOT / "configs" / "remote.yaml"
REMOTE_SCALE = ROOT / "configs" / "remote_scale.yaml"
REMOTE_ENV = {"FEDQUEST_LLM_AIMD": "1", "FEDQUEST_LLM_RATE": "4.5", "FEDQUEST_ANN": "hnsw",
              "FEDQUEST_VIRTUAL_SHARDS": "1"}


def facilities(nodes: int, walltime: str | dict[str, str], workers: int,
               extra: dict[str, Any] | None = None) -> dict[str, Any]:
    extra = extra or {}
    wall = walltime if isinstance(walltime, dict) else dict.fromkeys(("aurora", "midway3"), walltime)
    return {"aurora": {"endpoint": "${FEDQUEST_AURORA_CN_ENDPOINT}",
                       "config": {"nodes": nodes, "workers": workers, "walltime": wall["aurora"], "queue": "capacity",
                                  **extra}},
            "midway3": {"endpoint": "${FEDQUEST_MIDWAY3_CN_ENDPOINT}",
                        "config": {"nodes": nodes, "workers": workers, "walltime": wall["midway3"], **extra}}}


def _where(machine: str, names: set[str]) -> str | None:
    base, _, j = machine.rpartition("_")
    return machine if machine in names else (base if j.isdigit() and base in names else None)


def placement(federations: list[str], facility: str, nodes: int, names: set[str]) -> dict[str, int]:
    from fedquest.config import load_federation

    entries = []
    for fname in federations:
        sites = [s.name for s in load_federation(fname).sites if _where(s.machine, names) == facility]
        if sites:
            entries.append((fname, sites))
    load = [0.0] * nodes
    out: dict[str, int] = {}
    for fname, sites in sorted(entries, key=lambda e: len(e[1])):
        for start in range(0, len(sites), nodes):
            chunk = sites[start:start + nodes]
            slots = sorted(sorted(range(nodes), key=lambda j: (load[j], j))[:len(chunk)])
            for site, j in zip(chunk, slots, strict=True):
                out[f"{fname}/{site}"] = j
                load[j] += 1.0 / len(sites)
    return out


def main_federations(catalog: list[tuple[str, list[str], dict[str, Any]]]) -> list[str]:
    feds: dict[str, None] = dict.fromkeys([*FED2.values(), "mmqa_s4_r0"])
    for _, _, cfg in catalog:
        feds[cfg["federation"]] = None
        for m in cfg["methods"]:
            if m.get("federation"):
                feds[m["federation"]] = None
            if m["method"].startswith("fedgrag"):
                feds[m["params"]["index"]] = None
    return list(feds)


def remote_main(catalog: list[tuple[str, list[str], dict[str, Any]]]) -> str:
    feds = main_federations(catalog)
    fac = facilities(NODES, WALLTIME, 192)
    names = set(fac)
    body = {"facilities": {f: {**fc, "placement": placement(feds, f, NODES, names)} for f, fc in fac.items()},
            "federations": feds, "models": "configs/models/final.yaml", "action_timeout_s": 900, "env": REMOTE_ENV}
    return yaml.safe_dump(body, sort_keys=False, width=120)


def remote_scale() -> str:
    from fedquest.config import load_federation

    fac = facilities(SCALE_NODES, SCALE_WALLTIME, 128, {"min_blocks": 0})
    fac["aurora"]["config"].update(queue=SCALE_QUEUE_AURORA, walltime=SCALE_WALLTIME_AURORA)
    names = set(fac)
    feds = [c[0] for c in grid_cells()]
    placements: dict[str, dict[str, int]] = {f: {} for f in fac}
    for fed, _total, nodes, _per in grid_cells():
        cfg = load_federation(fed)
        for f in fac:
            sites = sorted((s.name for s in cfg.sites if _where(s.machine, names) == f),
                           key=lambda s: int(s.rsplit("_", 1)[-1]))
            for j, s in enumerate(sites):
                placements[f][f"{fed}/{s}"] = j % nodes
    body = {"facilities": {f: {**fc, "placement": placements[f]} for f, fc in fac.items()},
            "federations": feds, "models": "configs/models/final.yaml", "action_timeout_s": 900,
            "env": {**REMOTE_ENV, "FEDQUEST_REMOTE_CONFIG": "configs/remote_scale.yaml"}}
    return yaml.safe_dump(body, sort_keys=False, width=120)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify the configs instead of writing them")
    ap.add_argument("--list", action="store_true", help="print the catalog's experiment names in run order")
    args = ap.parse_args(argv)
    cells = cell_files()
    for p, text in cells.items():
        if not args.check and (not p.exists() or p.read_text() != text):
            p.write_text(text)
    catalog = experiments()
    if args.list:
        print("\n".join(n for n, _, _ in catalog))
        return 0
    files = {EXPERIMENTS / f"{n}.yaml": render(n, cfg) for n, _, cfg in catalog}
    if len(files) != len(catalog):
        raise ValueError("an experiment named twice")
    files |= cells
    files |= model_files()
    files[REMOTE] = remote_main(catalog)
    files[REMOTE_SCALE] = remote_scale()
    files[ROOT / "configs" / "scale_plan.json"] = json.dumps(scale_plan(), indent=1) + "\n"
    files[TOOLS_FILE] = tool_file(catalog)
    stale = [str(p.relative_to(ROOT)) for p, text in files.items() if not p.exists() or p.read_text() != text]
    extra = sorted(p.name for p in EXPERIMENTS.glob("*.yaml") if p not in files)
    if args.check:
        if stale or extra:
            print("configs not current (python scripts/make_configs.py): "
                  + ", ".join(stale + [f"{x} (not in the tables)" for x in extra]))
            return 1
        print(f"{len(catalog)} experiment configs, {len(tools(catalog))} tool tasks, both deployments, the scaling "
              f"plan and {len(cells)} grid layouts are current")
        return 0
    for p, text in files.items():
        if not p.exists() or p.read_text() != text:
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(p.suffix + ".tmp")
            tmp.write_text(text)
            tmp.replace(p)
    for x in extra:
        (EXPERIMENTS / x).unlink()
    print(f"wrote {len(catalog)} experiment configs, {len(tools(catalog))} tool tasks, {len(model_files())} models "
          f"files, both deployments, the scaling plan and {len(cells)} grid layouts"
          + (f"; removed {len(extra)} other experiment configs" if extra else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
