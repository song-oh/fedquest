# owner-first routing and the ReWOO baseline, plus held-out splits and the merge study
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from fedquest.kg.memory import MemoryKG
from fedquest.methods.base import build_method
from fedquest.methods.rewoo import parse_plan, short, substitute
from fedquest.types import Question
from tests.fakes import FakeObject, FakeSite, fake_federation, rules, running_responder
from tests.test_methods import Q, coordinator

ROOT = Path(__file__).resolve().parents[1]
QB = Question("qb", "musique", "In which city was the director of the film Big born?", ["New York City"],
              qtype="2hop", gold_doc_ids=["big", "pm"])
PATTERN_B = ('("anchor_entity"[SEP]Big)[REC]("entity"[SEP]person1)[REC]("entity"[SEP]city1)[REC]'
             '("answer_entity"[SEP]city1)[REC]("relation"[SEP]Big[SEP]person1[SEP]directed by)[REC]'
             '("relation"[SEP]person1[SEP]city1[SEP]born in)[END]')


def responder(task, messages):
    if task == "pattern":
        return PATTERN_B
    if task == "answer":
        text = " ".join(m["content"] if isinstance(m["content"], str) else "" for m in messages)
        return "<ANS>New York City</ANS>" if "New York City" in text else "<ANS>unknown</ANS>"
    return running_responder(task, messages)


def big_world(owner_states: bool = True):
    film = FakeObject("w/text/big#0", "text", "text", "Big (film)",
                      "Big is a 1988 American fantasy comedy film." +
                      (" (Big | directed by | Penny Marshall)" if owner_states else ""), [], doc_id="big")
    person = FakeObject("w/text/pm#0", "text", "text", "Penny Marshall",
                        "Penny Marshall was an American actress and director. (Penny Marshall | born in | New York City)",
                        [], doc_id="pm")
    other = FakeObject("w/text/y88#0", "text", "text", "1988 in film",
                       "Films of 1988 include Big." + ("" if owner_states else " (Big | directed by | Penny Marshall)"),
                       [], doc_id="y88")
    decoy = FakeObject("w/text/pl#0", "text", "text", "Penny Lane", "A street in Liverpool.", [], doc_id="pl")
    north = FakeSite("north", [film], rules([], {"text": [film.oid]}))
    south = FakeSite("south", [person], rules([], {"text": [person.oid]}))
    east = FakeSite("east", [other], rules([("directed by", "text", [other.oid])], {"text": []}))
    west = FakeSite("west", [decoy], rules([], {"text": [decoy.oid]}))
    sites = {"north": north, "south": south, "east": east, "west": west}
    return fake_federation(north, south, east, west), sites


PARAMS = {"probe": "semijoin", "limits": "share", "route": "vocab"}


def semijoins(site, value):
    return [x for x in site.probe_requests if x.semijoin and value in (x.mentions or [])]


async def run_big(params, owner_states=True):
    coord, sites = coordinator(world=big_world(owner_states), responder=responder, calib="none")
    r = await coord.run(build_method("fedquest", {**PARAMS, **params}), QB, "big")
    assert r.error is None, r.error
    return r, sites


async def test_owner_first_binds_each_forward_hop_at_its_title_owner_and_never_reaches_the_others():
    broadcast, bsites = await run_big({})
    owners, osites = await run_big({"owners": "titles"})
    assert broadcast.answer == owners.answer == "New York City"
    assert semijoins(bsites["east"], "Big")
    assert semijoins(osites["north"], "Big") and not semijoins(osites["east"], "Big")
    assert semijoins(osites["south"], "Penny Marshall")
    assert not any(semijoins(osites[s], "Penny Marshall") for s in ("north", "east", "west"))
    led = owners.ledger
    assert led["owner_hops"] == 2 and led["owner_bound"] == 2 and "owner_expanded" not in led
    assert any(e["event"] == "owners" for e in owners.trace) and not any(e["event"] == "expand" for e in owners.trace)
    assert led["messages"] < broadcast.ledger["messages"]


async def test_a_hop_its_owner_cannot_bind_goes_to_every_other_database_and_binds_as_broadcast_does():
    broadcast, _ = await run_big({}, owner_states=False)
    owners, osites = await run_big({"owners": "titles"}, owner_states=False)
    assert broadcast.answer == owners.answer == "New York City"
    assert semijoins(osites["east"], "Big")
    led = owners.ledger
    assert led["owner_expanded"] >= 1 and led["owner_bound"] == 1
    expand = next(e for e in owners.trace if e["event"] == "expand")
    assert "east" in expand["sites"] and "north" not in expand["sites"] and expand["new"] >= 1


async def test_owners_are_off_by_default_and_a_backward_hop_or_a_table_federation_goes_as_before():
    r, sites = await run_big({})
    assert "owner_hops" not in r.ledger and not any(e["event"] == "owners" for e in r.trace)
    with pytest.raises(ValueError, match="owners"):
        build_method("fedquest", {"owners": "kg"})
    coord, _ = coordinator()
    r = await coord.run(build_method("fedquest", {**PARAMS, "owners": "titles"}), Q, "tables")
    assert r.error is None and r.answer == "baseball" and "owner_hops" not in r.ledger


async def test_with_the_kg_a_value_routes_to_the_databases_its_stored_facts_came_from_too():
    kg = MemoryKG()
    coord, sites = coordinator(world=big_world(owner_states=False), responder=responder, calib="none", kg=kg)
    first = await coord.run(build_method("fedquest", {**PARAMS, "kg": "reuse"}), QB, "own")
    assert first.answer == "New York City"
    again = Question("qb2", "musique", "Where was the director of Big born?", ["New York City"], qtype="2hop")
    r = await coord.run(build_method("fedquest", {**PARAMS, "kg": "reuse", "owners": "titles+kg",
                                                  "keep_probe": True}), again, "own")
    assert r.error is None and r.answer == "New York City" and r.ledger.get("kg_hits", 0) >= 1


SAME = Question("q2", "mmqa", "Which sport is pictured on the poster of the 1976 movie with Ben Piazza?", ["baseball"],
                qtype="Compose(ImageQ,TableQ)", bridge=["The Bad News Bears"])


async def test_a_round_the_kg_answers_still_sends_the_questions_own_probe_under_keep_probe():
    for keep in (False, True):
        kg = MemoryKG()
        coord, sites = coordinator(kg=kg)
        await coord.run(build_method("fedquest", {"kg": "reuse"}), Q, "keep")
        for s in sites.values():
            s.probe_requests.clear()
        r = await coord.run(build_method("fedquest", {"kg": "reuse", "keep_probe": keep}), SAME, "keep")
        assert r.error is None and r.answer == "baseball" and r.ledger.get("kg_hits", 0) >= 1
        asked = [x for s in sites.values() for x in s.probe_requests if x.kind == "question"]
        texts = [x for x in asked if x.collections and any(c != "image" for c in x.collections)]
        if keep:
            assert r.ledger["kept_probes"] == 1 and texts
            assert any(e["event"] == "keep_probe" for e in r.trace)
        else:
            assert "kept_probes" not in r.ledger and not texts


PLAN = ("Plan: Find the 1976 film Ben Piazza acted in.\n#E1 = Search[Ben Piazza 1976 film]\n"
        "Plan: Name the film.\n#E2 = LLM[Which 1976 film did Ben Piazza act in, according to #E1?]\n"
        "Plan: Find its poster.\n#E3 = Search[#E2 poster]\n"
        "Plan: Name the sport.\n#E4 = LLM[What sport is on the poster of #E2, according to #E3?]")


def rewoo_responder(seen):
    def respond(task, messages):
        seen.append((task, messages))
        text = " ".join(m["content"] if isinstance(m["content"], str) else
                        " ".join(p.get("text", "") for p in m["content"] if isinstance(p, dict)) for m in messages)
        if task == "rewoo_plan":
            return PLAN
        if task == "rewoo_worker":
            return "The Bad News Bears" if "film did" in text else "baseball"
        if task == "answer":
            return "<ANS>baseball</ANS>" if "baseball" in text else "<ANS>unknown</ANS>"
        return running_responder(task, messages)
    return respond


def test_a_rewoo_plan_parses_into_steps_and_its_variables_take_their_results():
    steps = parse_plan(PLAN + "\n#E5 = Calculator[1 + 1]\n#E1 = Search[again]")
    assert [(n, t) for n, t, _, _ in steps] == [(1, "search"), (2, "llm"), (3, "search"), (4, "llm"), (5, "")]
    assert steps[1][3] == "Name the film."
    assert substitute("#E2 poster", {2: "The Bad News Bears"}) == "The Bad News Bears poster"
    assert substitute("#E9 poster", {}) == "unknown poster"
    assert short('Short answer: "Amy Adams"\nbecause') == "Amy Adams" and short("") == "unknown"


async def test_rewoo_plans_once_runs_its_steps_with_reacts_search_and_answers_grounded_inside_the_budget():
    seen = []
    coord, sites = coordinator(responder=rewoo_responder(seen))
    r = await coord.run(build_method("rewoo", {"total_budget": 16000}), Q, "rw")
    assert r.error is None and r.answer == "baseball"
    tasks = [t for t, _ in seen]
    assert tasks.count("rewoo_plan") == 1 and tasks.count("rewoo_worker") == 2 and tasks[-1] == "answer"
    searched = [x.key for s in sites.values() for x in s.probe_requests if x.kind == "question"]
    assert "Ben Piazza 1976 film" in searched and "The Bad News Bears poster" in searched
    led = r.ledger
    assert led["budget"] == 16000 and led["total_tokens"] <= 16000 and r.consulted
    event = next(e for e in r.trace if e["event"] == "rewoo")
    assert event["steps"] == 4 and event["results"]["#E2"] == "The Bad News Bears"
    tiny = await coordinator(responder=rewoo_responder([]))[0].run(build_method("rewoo", {"total_budget": 900}), Q,
                                                                    "rw2")
    assert tiny.error is None and tiny.answer


def test_the_held_out_splits_hold_no_question_an_earlier_run_touched():
    exposed = json.loads((ROOT / "configs" / "exposed.json").read_text())
    data = ROOT / "data" / "fed"
    files = {"mmqa": "mmqa_fed", "twowiki": "twowiki_fed", "hotpot": "hotpot_fed", "musique": "musique_next"}
    if not all((data / h / "questions" / "heldout.jsonl").exists() for h in files.values()):
        pytest.skip("the held-out splits are not written under this clone's data/ (scripts/make_pools.py --out data)")
    sizes = {}
    for b, home in files.items():
        held = [json.loads(x)["qid"] for x in (data / home / "questions" / "heldout.jsonl").open() if x.strip()]
        assert held and not set(held) & set(exposed[b]) and len(set(held)) == len(held), b
        sizes[b] = len(held)
    assert sizes == {"mmqa": 507, "twowiki": 300, "hotpot": 200, "musique": 365}


def test_the_catalog_runs_rewoo_beside_react_the_held_out_study_and_owner_first_at_every_scale():
    from fedquest.config import load_experiment

    def labels(name):
        return {m.label or m.method for m in load_experiment(name).methods}

    for b in ("mmqa", "musique", "twowiki", "hotpot"):
        assert {"react@16k", "rewoo@16k"} <= labels(f"main_{b}")
        assert {"rewoo@8k", "rewoo@32k"} <= labels(f"frontier_{b}")
    held = load_experiment("heldout_musique")
    assert held.split == "heldout" and held.limit == 365
    assert labels("main_musique") | {"dgrag@16k"} <= labels("heldout_musique")
    scale = labels("scale_musique")
    pat = re.compile(load_experiment("scale_musique").env["FEDQUEST_PROBE_TRACE_MATCH"])
    for sfx in ("s2_r0", "s4_k05_r0", "s8_k05_r0", "s16_k05_r0", "s32_k05_r0"):
        assert f"fedquest_own_{sfx}" in scale and pat.search(f"scale_musique/fedquest_own_{sfx}/q1#q1")
    assert {"fedquest_kg", "fedquest_kg_keep", "fedquest_kg_own", "fedquest_own"} <= labels("session_musique")
    assert {"fedquest_kg_version", "fedquest_kg_version_noprobe", "fedquest_cold"} <= labels("kgu_after_musique")
    plan = json.loads((ROOT / "configs" / "scale_plan.json").read_text())
    assert {p["mode"] for p in plan} == {"routed", "broadcast", "owners"}
    assert all(p["warmup"] == p["limit"] for p in plan)


def test_the_merge_study_fits_the_titles_features_and_orders_by_them_cross_fitted():
    from fedquest.eval import merge_study as ms

    assert ms.title_feats("Who directed the film Big?", "Big (film)") == (1.0, 1.0)
    assert ms.title_feats("Who directed the film Big?", "Penny Lane") == (0.0, 0.0)
    train, extra = [], {"titles": {}}
    for i in range(40):
        ev = {"qid": f"q{i}", "site": "a", "oid": f"e{i}", "rank": 2, "z": 3.0, "mentions": True, "kind": "question",
              "encoder": "bge-base", "modality": "text", "evidence": True, "collection": "text"}
        de = {**ev, "oid": f"d{i}", "rank": 1, "z": 4.0, "evidence": False}
        extra["titles"][id(ev)], extra["titles"][id(de)] = (1.0, 1.0), (0.0, 0.0)
        train += [ev, de]
    for s in ("anchorz_cal", "anchorz_title"):
        extra[f"platt_{s}"] = ms._fit_platt(train, ms.FEATURES[s], extra)
    pair = train[:2]
    assert ms._order("anchorz", pair, extra)[0]["evidence"] is False
    assert ms._order("anchorz_title", pair, extra)[0]["evidence"] is True
    assert ms._order("fused", pair, extra) == []


def test_a_shutdown_that_waited_in_a_queue_stops_nothing_and_starts_nothing(monkeypatch):
    import time

    from fedquest.remote import task

    started = []
    monkeypatch.setattr(task, "start", lambda *a, **k: started.append(a))
    monkeypatch.setattr(task, "ensure_server", lambda *a, **k: started.append(a))
    old = {"facility": "aurora", "slot": 0, "key": None, "action": "shutdown", "kwargs": {}, "env": {},
           "timeout_s": 60, "issued": time.time() - 2 * task.SHUTDOWN_STALE_S}
    out = task.handle(old)
    assert out["ok"] and out["value"]["stopping"] is None and out["value"]["stale_s"] >= task.SHUTDOWN_STALE_S
    assert not started
