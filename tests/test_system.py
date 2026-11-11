from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from fedquest import budget
from fedquest.methods.base import build_method
from fedquest.types import Question
from tests.test_fedquest_core import answering_world, tasks
from tests.test_methods import RUNNING_QUESTION

QG = Question("q1", "mmqa", RUNNING_QUESTION, ["baseball"], qtype="Compose(ImageQ,TableQ)",
              gold_doc_ids=["t1", "p1"], bridge=["The Bad News Bears"],
              meta={"hops": [{"doc_id": "t1", "after": []}, {"doc_id": "p1", "after": [1]}]})


def final_params(**extra):
    from fedquest.eval import runner

    presets, _, _ = runner.load_presets()
    return {**presets["final"], **extra}


def test_token_counts_add_the_chat_frame_and_images():
    msgs = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hello world"}]
    n = budget.count_messages(msgs)
    assert n == budget.FRAME_TOKENS + 2 * budget.MESSAGE_TOKENS + budget.count_text("be brief") + \
        budget.count_text("hello world")
    img = [{"role": "user", "content": [{"type": "text", "text": "x"}, {"type": "image_url", "image_url": {}}]}]
    assert budget.count_messages(img) >= budget.IMAGE_TOKENS


def test_the_guard_refuses_a_call_over_the_budget_but_never_the_answer():
    from fedquest.ledger import Ledger
    from fedquest.types import Usage

    led = Ledger()
    led.budget = 1000
    led.charge_llm("pattern", Usage(prompt_tokens=900, completion_tokens=10, calls=1))
    msgs = [{"role": "user", "content": "word " * 200}]
    with pytest.raises(budget.BudgetExhausted):
        budget.check(led, msgs, task="read_index", max_tokens=100, charge="bind_index")
    assert led.budget_refusals == 1
    budget.check(led, msgs, task="answer", max_tokens=128, charge="answer")
    led.budget = None
    budget.check(led, msgs, task="read_index", max_tokens=100, charge="bind_index")


def test_fit_indices_skips_an_oversized_text_and_cuts_the_first_when_none_fits():
    texts = ["a " * 10, "b " * 400, "c " * 10]
    kept = budget.fit_indices(texts, room=60)
    assert [i for i, _ in kept] == [0, 2]
    one = budget.fit_indices(["d " * 400], room=50)
    assert len(one) == 1 and len(one[0][1]) < len("d " * 400)


async def test_fedquest_under_a_total_budget_answers_and_stays_inside_it():
    coord, sites, _ = answering_world(answers=False)
    m = build_method("fedquest", final_params(total_budget=16000, route="vocab+proto"))
    r = await coord.run(m, QG, "b16")
    assert r.error is None and r.answer == "baseball"
    led = r.ledger
    assert led["budget"] == 16000 and led["budget_left"] >= 0 and led["total_tokens"] <= 16000
    assert "evidence_log" in led and {"t1", "p1"} <= set(led["evidence_log"])
    assert led["evidence_log"]["t1"]["hop"] == 0 and led["evidence_log"]["p1"]["later"] is True
    assert any(e["event"] == "hop7" for e in r.trace)
    assert any(e["event"] == "route" and e.get("fitted") is False for e in r.trace)


async def test_fedquest_with_a_tiny_budget_degrades_to_the_question_probe_without_an_error():
    coord, _, _ = answering_world(answers=True)
    r = await coord.run(build_method("fedquest", final_params(total_budget=1200)), QG, "tiny")
    assert r.error is None and r.answer
    assert any(e["event"] == "budget_stop" for e in r.trace) or r.ledger["total_tokens"] <= 1200 + 600


async def test_fedquest_without_a_total_budget_keeps_its_calls():
    coord, _, _ = answering_world(answers=False)
    r6 = await coord.run(build_method("fedquest", final_params()), QG, "unbudgeted")
    assert r6.error is None and r6.answer == "baseball" and "budget" not in r6.ledger
    assert "typed_graph" in tasks(coord)


async def test_ircot_under_a_budget_stops_reading_before_it_overruns():
    coord, _, _ = answering_world(answers=False)
    big = await coord.run(build_method("ircot_b", {"total_budget": 32000, "top": 2, "k": 2}), QG, "i32")
    coord2, _, _ = answering_world(answers=False)
    small = await coord2.run(build_method("ircot_b", {"total_budget": 900, "top": 2, "k": 2}), QG, "i1")
    for r, b in ((big, 32000), (small, 900)):
        assert r.error is None and r.ledger["budget"] == b
    assert small.ledger["total_tokens"] <= 900 + 200
    assert small.ledger["coordinator_calls"] <= big.ledger["coordinator_calls"]
    stops = [e for e in small.trace if e["event"] == "reader_answer"]
    assert stops and stops[0]["stop"] in ("budget", "answered", "nothing new", "steps")


def test_router_selection_rules():
    from fedquest.routing import Router

    scores = {"a": 3.0, "b": 1.0, "c": -0.5, "d": math.inf}
    assert Router.select(scores, "threshold", 0.5, None, min_sites=1) == ["d", "a", "b"]
    assert Router.select(scores, "threshold", 5.0, None, min_sites=2) == ["d", "a"]
    assert Router.select(scores, "topb", None, 2, min_sites=1) == ["d", "a"]
    assert Router.select(scores, "all", None, None) == ["d", "a", "b", "c"]
    assert Router.select(scores, "threshold", None, None) == ["d", "a", "b", "c"]


def test_direct_matching_reduces_each_synopsis_to_its_weighted_mean():
    import base64

    from fedquest.routing import _decode, mean_synopses

    cents = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float16)
    entry = {"encoder": "bge", "modality": "text", "n": 4, "dim": 2, "sizes": [3, 1],
             "centroids": base64.b64encode(cents.tobytes()).decode()}
    img = {"encoder": "clip", "modality": "image", "n": 1, "dim": 2, "sizes": [1], "centroids": entry["centroids"]}
    out = mean_synopses({"s1": {"k": 2, "collections": {"text": entry, "images": img}}})
    m = _decode(out["s1"]["collections"]["text"])
    assert m.shape == (1, 2) and out["s1"]["collections"]["text"]["sizes"] == [4]
    assert np.allclose(m[0], np.array([3.0, 1.0]) / np.sqrt(10.0), atol=1e-3)
    assert out["s1"]["collections"]["images"] is img


def test_the_report_counts_every_rounds_requests_and_replies_as_communication():
    from fedquest.eval.report import row_stats

    rows = [{"qid": "a", "ledger": {"bytes_control": 1024, "bytes_reply": 3072}},
            {"qid": "b", "ledger": {"bytes_control": 0, "bytes_reply": 2048}}]
    assert row_stats(rows)["kb"] == pytest.approx(3.0)


def test_a_routed_question_probe_spreads_its_k_over_the_routed_sites_only():
    from fedquest.methods.common import shaped
    from fedquest.types import ProbeRequest

    sizes = {("s1", "text"): ("bge", 100), ("s2", "text"): ("bge", 300), ("s3", "text"): ("bge", 600)}
    groups = {("text", "bge"): 1000}
    plain = shaped(ProbeRequest("q", "question", 5), "question", "text", sizes, groups, share=True)
    assert plain[0].group_size == 1000 and plain[0].sites == ["s1", "s2", "s3"]
    routed = shaped(ProbeRequest("q", "question", 5, sites=["s1", "s2"]), "question", "text", sizes, groups,
                    share=True)
    assert routed[0].group_size == 400 and routed[0].sites == ["s1", "s2"]


def _fake_openie(root: Path) -> dict[str, list[str]]:
    d = root / "inputs" / "openie" / "musique"
    d.mkdir(parents=True)
    recs = [
        ("musique/text/aaaa#0", [["Beitou District", "located in", "Taipei"]]),
        ("musique/text/bbbb#0", [["Fantasy Land Tour 2004", "performer", "S.H.E"], ["S.H.E", "performed in", "Taipei"]]),
        ("musique/text/cccc#0", [["Taipei", "capital of", "Taiwan"]]),
        ("musique/text/dddd#0", [["Kaohsiung", "located in", "Taiwan"]]),
    ]
    with (d / "openie.jsonl").open("w") as f:
        for oid, tr in recs:
            f.write(json.dumps({"oid": oid, "sha": "x", "entities": [], "triples": tr}) + "\n")
    return {"aurora_0": ["musique/text/aaaa#0"], "aurora_1": ["musique/text/bbbb#0"],
            "midway3_0": ["musique/text/cccc#0"], "midway3_1": ["musique/text/dddd#0"]}


def test_fedgrag_builds_clients_routes_and_composes_facts(tmp_path, monkeypatch):
    pytest.importorskip("sentence_transformers")
    monkeypatch.setenv("FEDQUEST_RUNS", str(tmp_path))
    from fedquest.fedgrag import build, model

    members = _fake_openie(tmp_path)
    model._load_corpus.cache_clear()
    model._load_index.cache_clear()
    out = build.build_layout("musique_s4_k05_r0", members=members)
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["clients"] == sorted(members) and 1 <= cfg["b"] <= 4
    idx = model.FedGRAGIndex.load("musique_s4_k05_r0")
    q = model.Encoder.encode(["Who performed in the city where Beitou District is located?"])[0]
    chosen, per, facts = idx.retrieve(q, b=4)
    assert len(chosen) == 4 and facts
    texts = {idx.corpus.fact_text(f) for f, _, _ in facts}
    assert any("Beitou District" in t for t in texts) and any("S.H.E" in t for t in texts)
    assert idx.package_bytes(per[chosen[0]]) > 0


def _stream_rows() -> list[dict]:
    rows = []
    for i in range(6):
        facts = [{"h": "Taipei", "r": "capital of", "t": "Taiwan", "site": "s1", "oid": "o1", "ver": "v"}]
        rows.append({"qid": f"q{i}", "position": i, "label": "fedquest_cold",
                     "ledger": {"total_tokens": 1000, "rounds": 4, "disclosed_keys_s1": 1},
                     "trace": [{"event": "hop7", "asks": [{"value": "Taipei", "rel": "capital of"}], "tokens": 400,
                                "wan_rounds": 2, "facts": facts}]})
    return rows


def test_cache_simulation_hits_repeats_and_counts_staleness(tmp_path):
    pytest.importorskip("sentence_transformers")
    from fedquest.cache import sim

    p = tmp_path / "rows.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in _stream_rows()))
    stream, tokens, rounds = sim.load_stream([p])
    relsim = sim.RelSim([r for hops in stream for h in hops for _, r in h.asks])
    trust = sim.simulate(stream, tokens, rounds, relsim, validity="trust")
    assert trust["asks"] == 6 and trust["hit_rate"] == pytest.approx(5 / 6, abs=1e-3)
    assert trust["tokens_saved_frac"] == pytest.approx(5 * 400 / 6000, abs=1e-3)
    val = sim.simulate(stream, tokens, rounds, relsim, validity="validate")
    assert val["validation_rounds"] == 5 and val["stale_hits"] == 0
    churn = sim.simulate(stream, tokens, rounds, relsim, validity="trust", update_rate=1000.0, objects=1)
    assert churn["stale_hits"] > 0
    pig = sim.simulate(stream, tokens, rounds, relsim, validity="piggyback", update_rate=1000.0, objects=1, ttl=100)
    assert pig["stale_hits"] == 0 and pig["validation_rounds"] == 0
    tiny = sim.simulate(stream, tokens, rounds, relsim, capacity=1.0, validity="trust")
    assert tiny["hit_rate"] == 0.0


def test_evidence_funnel_and_error_attribution():
    from fedquest.eval.evidence import summarize

    def row(qid, acc, log, consulted=("d1", "x"), logged=True):
        return {"qid": qid, "acc": acc, "gold_doc_ids": ["d1", "d2"], "consulted_doc_ids": list(consulted),
                "ledger": {"evidence_log": log, "total_tokens": 1000, "returned_total": 10, "context_logged": logged}}

    rows = [row("a", 1.0, {"d1": {"hop": 0, "later": False, "returned": 1, "read_site": 1, "in_context": True},
                           "d2": {"hop": 1, "later": True, "returned": 1, "read_coordinator": 1, "in_context": True}}),
            row("b", 0.0, {"d1": {"hop": 0, "later": False, "returned": 1, "read_site": 1, "in_context": True},
                           "d2": {"hop": 1, "later": True}}),
            row("c", 0.0, {"d1": {"hop": 0, "later": False}, "d2": {"hop": 1, "later": True}}, consulted=())]
    s = summarize(rows)
    assert s["funnel_objects"]["returned"] == pytest.approx(3 / 6, abs=1e-3)
    assert s["errors"] == {"not returned": 2}
    assert s["later_hop_questions"]["reach"] == pytest.approx(1 / 3, abs=1e-3)
    assert s["by_hop"]["hop2"]["read"] == pytest.approx(1 / 3, abs=1e-3)
    base = summarize([row("v", 1.0, {"d1": {"hop": 0, "later": False, "returned": 1},
                                     "d2": {"hop": 1, "later": True, "returned": 1}}, consulted=("d1",), logged=False)])
    assert base["funnel_objects"] == {"returned": 1.0, "read": 0.5, "in_context": 0.5}


def test_the_catalog_is_current_and_every_label_builds():
    import subprocess
    import sys

    from fedquest.config import load_experiment
    from fedquest.eval.runner import expand

    root = Path(__file__).resolve().parents[1]
    out = subprocess.run([sys.executable, "scripts/make_configs.py", "--check"], cwd=root, capture_output=True,
                         text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    for p in sorted((root / "configs" / "experiments").glob("*.yaml")):
        exp = load_experiment(p.stem)
        for _label, spec, params in expand(exp):
            build_method(spec.method, params)
    plan = json.loads((root / "configs" / "scale_plan.json").read_text())
    cells = {(r["nodes"], r["per_node"]) for r in plan}
    assert cells == {(n, c) for n in (1, 2, 4) for c in (1, 2, 4, 8) if 2 * n * c <= 32}
    assert all(r["databases"] == 2 * r["nodes"] * r["per_node"] for r in plan)
    assert len(plan) == 3 * len(cells) and {(r["clients"], r["warmup"], r["limit"]) for r in plan} == {(32, 300, 300)}
    assert len({r["tag"] for r in plan}) == len(plan)
    assert {r["label"].split("_s", 1)[1] for r in plan} == {"2_r0", "4_k05_r0", "8_k05_r0", "16_k05_r0", "32_k05_r0"}


def test_the_scaling_pool_places_c_databases_on_each_of_n_nodes():
    import yaml

    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / "configs" / "remote_scale.yaml").read_text())
    for fac, fc in cfg["facilities"].items():
        assert fc["config"]["nodes"] == 4 and fc["config"]["min_blocks"] == 0
        assert (fc["config"].get("queue"), fc["config"]["walltime"]) == (
            ("capacity", "06:00:00") if fac == "aurora" else (None, "06:00:00"))
        per: dict[tuple[str, int], int] = {}
        for key, slot in fc["placement"].items():
            fed = key.split("/", 1)[0]
            per[(fed, slot)] = per.get((fed, slot), 0) + 1
        for (fed, slot), count in per.items():
            n = int(fed.rsplit("_n", 1)[1])
            total = int(fed.split("_s", 1)[1].split("_", 1)[0])
            assert slot < n and count == total // (2 * n), (fac, fed, slot, count)
    assert np.isfinite(len(cfg["federations"]))


def _labels(name: str) -> dict:
    from fedquest.config import load_experiment
    from fedquest.eval.runner import expand, resolved_params

    out = {}
    for label, spec, params in expand(load_experiment(name)):
        out[label] = (spec, build_method(spec.method, resolved_params(spec.method, params)))
    return out


def test_main_compares_every_iterative_method_at_one_budget_and_the_rest_at_their_own_cost():
    for b, text in (("mmqa", False), ("musique", True), ("twowiki", True), ("hotpot", True)):
        labs = _labels(f"main_{b}")
        budgeted = {"fedquest", "ircot_t4@16k", "ircot_t8@16k", "react@16k", "react_ops@16k", "mgrag_fu@16k"}
        assert budgeted <= set(labs), b
        for lab in budgeted:
            assert labs[lab][1].total_budget == 16000, (b, lab)
        for lab in ("mgrag_fed@8", "vanilla_rag@8", "closed_book", "ircot@8x8"):
            assert getattr(labs[lab][1], "total_budget", None) is None, (b, lab)
        assert ("fedgrag" in labs) == text and ("fedgrag_text" in labs) == text
        xfit = b in ("twowiki", "hotpot")
        if text:
            assert labs["fedgrag"][1].index == f"{b}_s4_k05_r0"
            assert labs["fedgrag"][1].router == ("x{x}" if xfit else None)
            dg = _labels(f"dgrag_{b}")
            assert set(dg) == {"dgrag@16k"} and dg["dgrag@16k"][1].total_budget == 16000
            assert dg["dgrag@16k"][1].index == f"{b}_s4_k05_r0"
        fq = labs["fedquest"][1]
        assert fq.route_fit == ("{federation}_x{x}" if xfit else None)
        assert fq.objects > 24 and fq.max_rounds > 6 and fq.context_tokens() > 1500 and fq.route == "vocab+proto"
        assert labs["ircot@8x8"][1].max_paragraphs == 48
        assert labs["react_ops@16k"][1].tools == "fedquest" and labs["react_ops@16k"][1].tokens is None


def test_every_study_draws_from_its_benchmarks_dev_questions():
    import json as _json

    from fedquest.config import load_experiment

    pools = {"mmqa": ("full", 2441), "musique": ("test", 1000), "twowiki": ("full", 1000), "hotpot": ("full", 1000)}
    for b, (split, n) in pools.items():
        assert (load_experiment(f"main_{b}").split, load_experiment(f"main_{b}").limit) == (split, n), b
        for name in (f"frontier_{b}", f"abl_{b}", f"dist_{b}"):
            exp = load_experiment(name)
            assert (exp.split, exp.limit) == (split, 300), name
        if b != "mmqa":
            assert (load_experiment(f"dgrag_{b}").split, load_experiment(f"dgrag_{b}").limit) == (split, n)
            assert (load_experiment(f"hippo_{b}").split, load_experiment(f"hippo_{b}").limit) == (split, n)
    for name in ("scale_musique", "seeds_musique", "latency_musique", "mx_llama31_8b_musique"):
        assert (load_experiment(name).split, load_experiment(name).limit) == ("test", 300), name
    assert (load_experiment("latency_mmqa").split, load_experiment("latency_mmqa").limit) == ("full", 300)
    root = Path(__file__).resolve().parents[1] / "data" / "fed" / "musique_next" / "questions"
    if (root / "test.jsonl").exists():
        test = {_json.loads(x)["qid"] for x in (root / "test.jsonl").open()}
        calib = {_json.loads(x)["qid"] for x in (root / "calib.jsonl").open()}
        train_pool = {_json.loads(x)["qid"] for x in (root / "final_test.jsonl").open()}
        assert len(test) == 1000 and not test & calib and not test & train_pool


def test_the_frontier_holds_fedquest_and_ircot_at_4k_8k_32k_and_the_agents_at_8k_32k():
    for b in ("mmqa", "musique", "twowiki", "hotpot"):
        labs = _labels(f"frontier_{b}")
        for k, budget_ in (("4k", 4000), ("8k", 8000), ("32k", 32000)):
            for lab in (f"fedquest@{k}", f"ircot_t4@{k}", f"ircot_t8@{k}",
                        *((f"react@{k}", f"mgrag_fu@{k}") if budget_ >= 8000 else ())):
                assert labs[lab][1].total_budget == budget_, (b, lab)
        assert "react@4k" not in labs and "mgrag_fu@4k" not in labs
        f4, f32 = labs["fedquest@4k"][1], labs["fedquest@32k"][1]
        assert f32.objects > f4.objects and f32.context_tokens() >= f4.context_tokens()


def test_the_scale_study_traces_the_grids_labels_and_runs_every_baseline_at_every_n():
    import re

    from fedquest.config import load_experiment

    exp = load_experiment("scale_musique")
    pat = re.compile(exp.env["FEDQUEST_PROBE_TRACE_MATCH"])
    labs = _labels("scale_musique")
    for n in (2, 4, 8, 16, 32):
        sfx = "s2_r0" if n == 2 else f"s{n}_k05_r0"
        for base in ("fedquest", "fedquest_all", "fedgrag", "vanilla_rag@8", "mgrag_fed@8", "ircot_t8@16k"):
            assert f"{base}_{sfx}" in labs, (base, n)
        assert labs[f"fedgrag_{sfx}"][1].index == f"musique_{sfx}"
        assert pat.search(f"scale_musique/fedquest_{sfx}/q1#q1") and pat.search(f"scale_musique/fedquest_all_{sfx}/q1#q1")
        assert not pat.search(f"scale_musique/fedgrag_{sfx}/q1#q1")
    assert labs["fedquest_all_s32_k05_r0"][1].route == "all" and labs["fedquest_s32_k05_r0"][1].route == "vocab+proto"
    assert not any("topb" in lab or re.search(r"_s\d+_r[1-9]", lab) for lab in labs)
    assert {"fedquest_rank_s32_k05_r0", "fedquest_ranknaive_s32_k05_r0", "fedquest_e5held_s32_k05_r0"} <= set(labs)


def test_the_seeds_study_runs_four_methods_at_four_more_placements_of_every_n():
    from fedquest.config import load_experiment

    exp = load_experiment("seeds_musique")
    labs = _labels("seeds_musique")
    want = {f"{base}_s{n}_k05_r{sd}" for n in (4, 8, 16, 32) for sd in (1, 2, 3, 4)
            for base in ("fedquest", "fedgrag", "vanilla_rag@8", "mgrag_fed@8")}
    assert set(labs) == want and exp.limit == 300 and exp.stage == 2
    assert set(exp.after) == {"tool:route_seeds", "tool:fedgrag_seeds"}
    assert labs["fedgrag_s32_k05_r3"][1].index == "musique_s32_k05_r3"


def test_the_backbone_matrix_has_six_models_and_runs_ircot_beside_fedquest():
    from fedquest.config import load_experiment

    cells = sorted({p.stem.removeprefix("mx_").rsplit("_", 1)[0]
                    for p in (Path(__file__).resolve().parents[1] / "configs" / "experiments").glob("mx_*.yaml")})
    assert cells == ["gemma4_e4b", "gptoss_120b", "gptoss_20b", "llama31_70b", "llama31_8b", "nemotron3_ultra"]
    exp = load_experiment("mx_llama31_8b_mmqa")
    assert [m.label or m.method for m in exp.methods] == ["fedquest", "ircot_t8@16k", "mgrag_fu@16k", "mgrag_fed@8"]


def test_tools_build_only_what_the_catalog_needs():
    root = Path(__file__).resolve().parents[1]
    tools = json.loads((root / "configs" / "tools.json").read_text())
    ids = {t["id"]: t for t in tools["tools"]}
    route = ids["tool:route"]["argv"]
    assert "musique_s32_k05_r0" in route and "mmqa_s4_k05_r0" in route and "--split" in route
    build = ids["tool:fedgrag_build"]["argv"]
    assert "--missing" in build and "musique_s4_k05_r0" in build and not any(a.startswith("mmqa") for a in build)
    xroute, xbuild = ids["tool:route_xfit"]["argv"], ids["tool:fedgrag_xfit"]["argv"]
    assert {"twowiki_s4_k05_r0", "hotpot_s4_k05_r0"} <= set(xroute) and "--halves" in xroute
    assert xroute[xroute.index("--split") + 1] == "full" and xbuild[xbuild.index("--split") + 1] == "full"
    assert set(xbuild) >= {"twowiki_s4_k05_r0", "hotpot_s4_k05_r0", "--halves"} and "--missing" not in xbuild
    assert not any(a.startswith(("twowiki", "hotpot")) for a in route + build)
    dgrag = ids["tool:dgrag_build"]["argv"]
    assert {"musique_s4_k05_r0", "twowiki_s4_k05_r0", "hotpot_s4_k05_r0"} <= set(dgrag) and "--missing" in dgrag
    assert ids["tool:kgu_make_musique"]["argv"][-2:] == ["--n", "300"]
    seeds = ids["tool:route_seeds"]["argv"] + ids["tool:fedgrag_seeds"]["argv"]
    assert "musique_s32_k05_r4" in seeds and not any(a.endswith("_r0") for a in seeds)
    assert not any(a.endswith(("_r1", "_r2", "_r3", "_r4")) for a in route + build)
    assert ids["tool:fedgrag_seeds"]["after"] == ["tool:fedgrag_build"] and ids["tool:route_seeds"]["after"] == [
        "tool:route"]
    assert tools["post"][0]["argv"] == ["bash", "scripts/post.sh"] and tools["post"][0]["after_any"]


def _react_world(actions):
    from tests.fakes import running_responder
    from tests.test_methods import coordinator

    queue = list(actions)

    def responder(task, messages):
        if task == "planner":
            return queue.pop(0) if queue else '{"thought": "done", "action": "finish", "input": "baseball"}'
        return running_responder(task, messages)

    return coordinator(responder=responder)


async def test_react_searches_with_its_own_text_reads_whole_objects_and_answers_grounded():
    coord, sites = _react_world(['{"thought": "find the film", "action": "search", "input": "Ben Piazza 1976 film"}',
                                 '{"thought": "the poster", "action": "search", "input": "The Bad News Bears poster"}'])
    r = await coord.run(build_method("react", {"total_budget": 16000}), QG, "react")
    assert r.error is None and r.answer
    steps = [e for e in r.trace if e["event"] == "step" and e["role"] == "assistant"]
    assert len(steps) == 3 and '"search"' in steps[0]["content"] and '"finish"' in steps[-1]["content"]
    led = r.ledger
    assert led["budget"] == 16000 and led["total_tokens"] <= 16000
    ended = next(e for e in r.trace if e["event"] == "planner_answer")
    assert ended["ended"] == "finish" and ended["read"] >= 1
    assert r.consulted


async def test_react_under_a_tiny_budget_stops_searching_and_still_answers():
    actions = ['{"thought": "s", "action": "search", "input": "Ben Piazza"}'] * 6
    coord, _ = _react_world(actions)
    r = await coord.run(build_method("react", {"total_budget": 1500}), QG, "react_tiny")
    assert r.error is None
    ended = next(e for e in r.trace if e["event"] == "planner_answer")
    assert ended["ended"] in ("budget", "finish") and r.ledger["total_tokens"] <= 1500 + 600


async def test_mgrag_fu_under_a_total_budget_extracts_less_and_answers():
    from tests.test_methods import coordinator

    params = {"probe": "semijoin", "limits": "share", "mention": 2, "ties": "salted", "batch": 4, "tokens": 32000,
              "objects": 32}
    coord, _ = coordinator()
    big = await coord.run(build_method("mgrag_fu", params), QG, "fu")
    coord2, _ = coordinator()
    small = await coord2.run(build_method("mgrag_fu", {**params, "total_budget": 2500}), QG, "fu_small")
    assert big.error is None and small.error is None and small.answer
    assert small.ledger["budget"] == 2500 and "budget" not in big.ledger
    assert small.ledger["total_tokens"] <= big.ledger["total_tokens"]
    assert small.ledger["objects_read"] <= big.ledger["objects_read"]


async def test_react_ops_under_a_total_budget_stops_its_steps_and_reads_and_answers():
    from tests.fakes import running_responder
    from tests.test_methods import coordinator

    steps = ['{"thought": "find the film", "action": "search", "input": {"text": "Ben Piazza 1976 film"}}'] * 12

    def responder(task, messages):
        if task == "planner_ops":
            return steps.pop(0) if steps else '{"thought": "done", "action": "finish", "input": "baseball"}'
        return running_responder(task, messages)

    coord, _ = coordinator(responder=responder)
    m = build_method("llm_planner_ops", {"max_steps": 16, "views": True, "total_budget": 3000, "tokens": None})
    r = await coord.run(m, QG, "react_ops")
    assert r.error is None and r.answer
    ended = next(e for e in r.trace if e["event"] == "planner_answer")
    assert ended["ended"] in ("budget", "finish", "cap") and ended["steps"] < 12
    assert r.ledger["budget"] == 3000 and r.ledger["total_tokens"] <= 3000 + 600
