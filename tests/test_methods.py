# coordinator and method behavior on the running example and small scripted worlds
import json

import pytest

from fedquest.config import load_models
from fedquest.coordinator import Coordinator
from fedquest.kg.matcher import Matcher, TextSim
from fedquest.kg.memory import MemoryKG
from fedquest.llm import prompts
from fedquest.llm.auth import TokenProvider
from fedquest.llm.client import LLMClient
from fedquest.llm.fake import FakeBackend
from fedquest.methods.base import Method, build_method
from fedquest.methods.common import hint_modalities
from fedquest.methods.explore import is_evidence
from fedquest.types import Candidate, Question
from fedquest.vdb.encoders import load_encoder
from tests.fakes import (
    RUNNING_PATTERN,
    RUNNING_QUESTION,
    FakeObject,
    FakeSite,
    fake_federation,
    rules,
    running_example,
    running_responder,
)

FEDQUESTS = ("fedquest",)

Q = Question("q1", "mmqa", RUNNING_QUESTION, ["baseball"], qtype="Compose(ImageQ,TableQ)", bridge=["The Bad News Bears"])


# evidence rate by (kind, modality, round) and (rank, signal); typical() fits a calibrator to it
TYPICAL = {
    ("question", "text", "first"): {(1, True): .35, (1, False): .08, (3, True): .15, (3, False): .04, (5, False): .02},
    ("question", "table", "first"): {(1, True): .4, (1, False): .1, (3, True): .2, (3, False): .05},
    ("question", "image", "first"): {(1, True): .3, (1, False): .05, (3, False): .03},
    ("bound", "text", "first"): {(1, True): .35, (1, False): .08, (3, True): .15, (3, False): .04, (5, False): .02},
    ("bound", "table", "first"): {(1, True): .4, (1, False): .1, (3, True): .2, (3, False): .05, (5, False): .03},
    ("bound", "image", "first"): {(1, True): .3, (1, False): .05, (3, False): .03},
    ("bridge", "text", "late"): {(1, True): .5, (1, False): .1, (3, True): .2, (3, False): .05},
    ("bridge", "image", "late"): {(1, True): .5, (1, False): .08, (3, False): .04},
    ("title", "image", "first"): {(1, True): .08, (1, False): .02},
    ("title", "text", "late"): {(1, True): .5, (1, False): .1, (3, True): .2},
    ("title", "image", "late"): {(1, True): .65, (1, False): .1, (3, True): .15},
    ("title", "image", "filter"): {(1, True): .6, (1, False): .1, (3, True): .15},
    ("sibling", "table", "first"): {(1, True): .15, (1, False): .05, (3, False): .05, (8, False): .05},
}


def typical(n=200):
    from fedquest.planner.calibration import Calibrator

    obs = []
    for (kind, modality, rnd), rates in TYPICAL.items():
        for (rank, signal), rate in rates.items():
            hits = round(rate * n)
            obs += [{"qid": f"q{i}", "round": rnd, "kind": kind, "site": "s", "collection": "c", "oid": f"o{i}",
                     "modality": modality, "rank": rank, "signal": signal, "entry": "complete", "tokens": 50,
                     "states": None, "evidence": i < hits} for i in range(n)]
    for rnd in ("first", "filter", "late"):
        obs += [{"qid": f"q{i}", "round": rnd, "path": "gold", "groups": ["text", "image"]} for i in range(4)]
        obs += [{"qid": f"q{i}", "round": rnd, "kind": "bound", "site": "s", "collection": "c", "oid": f"g{i}",
                 "modality": m, "rank": 1, "signal": True, "entry": "complete", "tokens": 50, "states": None,
                 "evidence": True} for i in range(4) for m in ("text", "image")]
    cal = Calibrator()
    cal.fit(obs)
    return cal


def coordinator(image_first=True, kg=None, world=None, responder=running_responder, calib=None):
    from fedquest.planner.calibration import Calibrator

    fed, sites = world or running_example(image_first=image_first)
    cfg = load_models().model_copy(update={"provider": "fake"})
    llm = LLMClient(cfg, cache=None, tokens=TokenProvider("fake"), backend=FakeBackend(responder))
    matcher = Matcher(TextSim(load_encoder("hash")), theta=0.6)
    cal = Calibrator() if calib == "none" else calib or TYPICAL_CAL
    return Coordinator(fed, llm, kg or MemoryKG(), matcher, calib=cal), sites


TYPICAL_CAL = typical()


async def test_fedquest_binds_the_bridge_from_a_complete_entry_and_reads_only_the_poster():
    coord, sites = coordinator()
    r = await coord.run(build_method("fedquest"), Q, "ns1")
    assert r.error is None and r.matched and r.answer == "baseball"
    assert r.indexed == ["midway3/mmqa/row/t1#6"] and r.consulted == ["aurora/mmqa/image/p1"]
    assert r.bindings == {"?film1": "The Bad News Bears", "?sport1": "baseball"}
    assert not sites["midway3"].read_calls and r.ledger["objects_read"] == 1 and r.ledger["read_calls"] == 1
    by_version = {f["version"]: f for f in r.evidence}
    assert by_version[prompts.INDEX_BIND_VERSION]["tail"] == "The Bad News Bears"
    assert by_version[prompts.READ_VERSION] == {**by_version[prompts.READ_VERSION], "tail": "baseball",
                                                  "site": "aurora", "oid": "mmqa/image/p1"}


async def test_the_bound_value_is_the_next_key_at_every_site_and_visual_keys_go_to_images():
    coord, sites = coordinator()
    seen, probe = [], coord.fed.probe

    async def recording(requests, **kwargs):
        seen.extend(requests)
        return await probe(requests, **kwargs)

    coord.fed.probe = recording
    await coord.run(build_method("fedquest"), Q, "late")
    late = [q for q in seen if q.key == "The Bad News Bears poster depicts sport"]
    assert late and late[0].kind == "bound" and late[0].collections == ["image"]
    assert sum(q.kind == "question" for q in seen) == 1
    question, qsites = coordinator()
    n = await question.run(build_method("fedquest_q"), Q, "question-keys")
    assert n.error is None and not any("bad news bears" in k.lower() for s in qsites.values() for k in s.probes)
    assert {k for s in qsites.values() for k in s.probes} == {RUNNING_QUESTION}


def cast_world():
    inside = FakeObject("mmqa/text/io#0", "text", "text", "Inside Out (2015 film)",
                        "Inside Out is a 2015 American animated coming-of-age film produced by Pixar Animation Studios "
                        "for Walt Disney Pictures and directed by Pete Docter, from a screenplay he co-wrote. "
                        "(Amy Poehler | starred in | Inside Out); (Richard Kind | starred in | Inside Out)", [],
                        doc_id="io")
    allwe = FakeObject("mmqa/text/aw#0", "text", "text", "All We Had",
                       "(Judy Greer | starred in | All We Had); (Richard Kind | starred in | All We Had)", [], doc_id="aw")
    aurora = FakeSite("aurora", [inside], rules([("inside out", "text", [inside.oid])], {"text": [inside.oid]}))
    edge = FakeSite("edge", [allwe], rules([("all we had", "text", [allwe.oid])], {"text": [allwe.oid]}),
                      policy="facts")
    return fake_federation(aurora, edge), {"aurora": aurora, "edge": edge}


CAST_PATTERN = ('("anchor_entity"[SEP]Inside Out)[REC]("anchor_entity"[SEP]All We Had)[REC]("answer_entity"[SEP]person1)'
                '[REC]("relation"[SEP]person1[SEP]Inside Out[SEP]starred in)[REC]'
                '("relation"[SEP]person1[SEP]All We Had[SEP]starred in)[END]')


def cast_responder(task, messages):
    from fedquest.llm.fake import message_text

    if task == "pattern":
        return CAST_PATTERN
    if task == "answer":
        facts = message_text(messages)
        both = [n for n in ("Richard Kind", "Judy Greer", "Amy Poehler")
                if f"({n} | starred in | Inside Out)" in facts and f"({n} | starred in | All We Had)" in facts]
        return f"<ANS>{both[0] if both else 'unknown'}</ANS>"
    return running_responder(task, messages)


async def test_two_conditions_witnessed_by_different_objects_meet_as_an_intersection():
    coord, sites = coordinator(world=cast_world(), responder=cast_responder)
    q = Question("q3", "mmqa", "What voice actor starred in Inside Out and All We Had?", ["Richard Kind"])
    r = await coord.run(build_method("fedquest"), q, "cast")
    assert r.answer == "Richard Kind" and r.bindings["?person1"] == "Richard Kind"
    assert {c.split("/")[0] for c in r.consulted} == {"aurora", "edge"}
    assert r.ledger["bytes_raw"] == len(sites["aurora"].objects["mmqa/text/io#0"].text.encode())


async def test_a_name_the_cited_object_does_not_state_is_dropped_at_the_site():
    coord, sites = coordinator()
    sites["aurora"].reader = lambda text: {"sport1": [{"entity": "cricket", "object": 1, "conditions": [1]}]}
    r = await coord.run(build_method("fedquest"), Q, "witness")
    assert r.bindings.get("?sport1") == "cricket"
    coord2, sites2 = coordinator()
    sites2["edge"].objects["mmqa/text/b1#0"].text = "Ben Piazza (1933-1991) was an American actor."
    sites2["midway3"].objects["mmqa/row/t1#6"].text += " …"
    sites2["midway3"].reader = lambda text: {"film1": [{"entity": "Jaws", "object": 1, "conditions": [1]}]}
    r2 = await coord2.run(build_method("fedquest"), Q, "witness2")
    assert "Jaws" not in json.dumps(r2.bindings) and not any(f["tail"] == "Jaws" for f in r2.evidence)


def poster_world():
    rows = [FakeObject(f"mmqa/row/vv#{i}", "row", "table", "Vakkantham Vamsi — Films",
                       f"Vakkantham Vamsi | Films | Year: {y}; Title: {t}; Cast: {c}", [], doc_id="vv")
            for i, (y, t, c) in enumerate([(2014, "Race Gurram", "Allu Arjun"), (2016, "Kick 2", "Ravi Teja")])]
    posters = [FakeObject("mmqa/image/rg", "image", "image", "Race Gurram", "Race Gurram",
                          [("Race Gurram", "poster features", "motorcycle")], doc_id="rg"),
               FakeObject("mmqa/image/k2", "image", "image", "Kick 2", "Kick 2",
                          [("Kick 2", "poster features", "a man in a suit")], doc_id="k2")]
    midway3 = FakeSite("midway3", rows, rules([("vakkantham", "row", [r.oid for r in rows])], {"row": []}))
    aurora = FakeSite("aurora", posters, rules([], {"image": [p.oid for p in posters]}))
    return fake_federation(midway3, aurora), {"midway3": midway3, "aurora": aurora}


POSTER_PATTERN = ('("anchor_entity"[SEP]Vakkantham Vamsi)[REC]("entity"[SEP]film1)[REC]("answer_entity"[SEP]film1)[REC]'
                  '("relation"[SEP]Vakkantham Vamsi[SEP]film1[SEP]wrote)[REC]'
                  '("relation"[SEP]film1[SEP]film1[SEP]poster features a motorcycle)[END]')


async def test_a_visual_condition_reads_the_candidates_own_images_in_one_call():
    def responder(task, messages):
        if task == "pattern":
            return POSTER_PATTERN
        if task == "read_index":
            return json.dumps({"film1": [{"entity": "Race Gurram", "object": 1, "conditions": [1]},
                                         {"entity": "Kick 2", "object": 2, "conditions": [1]}]})
        if task == "answer":
            from fedquest.llm.fake import message_text

            return "<ANS>Race Gurram</ANS>" if "Race Gurram | poster features a motorcycle" in message_text(messages)                 else "<ANS>unknown</ANS>"
        return running_responder(task, messages)

    coord, sites = coordinator(world=poster_world(), responder=responder)
    q = Question("q4", "mmqa", "Which film written by Vakkantham Vamsi has a poster featuring a motorcycle?", ["Race Gurram"])
    r = await coord.run(build_method("fedquest"), q, "list")
    assert r.pattern["conditions"] == {"?film1": ["poster features a motorcycle"]}
    assert sites["aurora"].read_calls == [["mmqa/image/rg", "mmqa/image/k2"]]
    assert r.bindings == {"?film1": "Race Gurram"} and r.answer == "Race Gurram"


@pytest.mark.parametrize("label", FEDQUESTS)
async def test_the_budget_counts_every_object_a_model_reads_at_a_site(label):
    coord, sites = coordinator(world=poster_world(), responder=lambda t, m: POSTER_PATTERN if t == "pattern"
                               else json.dumps({"film1": [{"entity": "Race Gurram", "object": 1, "conditions": [1]},
                                                          {"entity": "Kick 2", "object": 2, "conditions": [1]}]})
                               if t == "read_index" else running_responder(t, m))
    q = Question("q5", "mmqa", "Which film written by Vakkantham Vamsi has a poster featuring a motorcycle?", [])
    r = await coord.run(build_method(label, {"objects": 1}), q, "budget")
    assert r.ledger["objects_read"] == 1 and len(r.consulted) == 1


@pytest.mark.parametrize("label", FEDQUESTS)
async def test_one_object_per_call_costs_more_than_reading_the_group_together(label):
    def responder(task, messages):
        return POSTER_PATTERN if task == "pattern" else running_responder(task, messages)

    q = Question("q6", "mmqa", "Which film written by Vakkantham Vamsi has a poster featuring a motorcycle?", [])
    together, _ = coordinator(world=poster_world(), responder=responder)
    single, _ = coordinator(world=poster_world(), responder=responder)
    a = await together.run(build_method(label), q, "t")
    b = await single.run(build_method(label, {"size": "unit"}), q, "s")
    assert a.ledger["read_calls"] == 1 and a.ledger["objects_read"] == 2
    assert b.ledger["read_calls"] == b.ledger["objects_read"]
    per_object = [x.ledger["extraction_tokens"] / x.ledger["objects_read"] for x in (a, b)]
    assert per_object[1] > per_object[0]


def recording(coord):
    seen, probe = [], coord.fed.probe

    async def record(requests, **kwargs):
        seen.extend(requests)
        return await probe(requests, **kwargs)

    coord.fed.probe = record
    return seen


def calls(coord, task):
    return sum(t == task for _, t in coord.llm.backend.calls)


def inside_out_world(shared=True):
    pad = ("Inside Out is a 2015 American animated coming-of-age film produced by Pixar Animation Studios for Walt "
           "Disney Pictures and directed by Pete Docter, from a screenplay he co-wrote with others. ")
    row = FakeObject("mmqa/row/mc#3", "row", "table", "Mark Consuelos — Filmography",
                     "Mark Consuelos | Filmography | Year: 2016; Title: All We Had; Role: Vic", [], doc_id="mc")
    inside = FakeObject("mmqa/text/io#0", "text", "text", "Inside Out (2015 film)",
                        pad + "(Amy Poehler | starred in | Inside Out); (Richard Kind | starred in | Inside Out); "
                        "(Phyllis Smith | starred in | Inside Out)", [], doc_id="io")
    third = "Richard Kind" if shared else "Eve Lindley"
    allwe = FakeObject("mmqa/text/aw#0", "text", "text", "All We Had",
                       f"(Judy Greer | starred in | All We Had); ({third} | starred in | All We Had); "
                       "(Katie Holmes | starred in | All We Had)", [], doc_id="aw")
    midway3 = FakeSite("midway3", [row], rules([("mark consuelos", "row", [row.oid])], {"row": []}))
    aurora = FakeSite("aurora", [inside], rules([("inside out", "text", [inside.oid])], {"text": []}))
    edge = FakeSite("edge", [allwe], rules([("all we had", "text", [allwe.oid])], {"text": []}), policy="facts")
    return fake_federation(midway3, aurora, edge), {"midway3": midway3, "aurora": aurora, "edge": edge}


INSIDE_OUT_Q = "Which actor starred in Inside Out and in the 2016 movie with Mark Consuelos that has the character Vic?"
INSIDE_OUT_PATTERN = (
    '("anchor_entity"[SEP]Inside Out)[REC]("anchor_entity"[SEP]Mark Consuelos)[REC]("anchor_entity"[SEP]Vic)[REC]'
    '("entity"[SEP]person1)[REC]("entity"[SEP]movie1)[REC]("answer_entity"[SEP]person1)[REC]'
    '("relation"[SEP]person1[SEP]Inside Out[SEP]starred in)[REC]("relation"[SEP]person1[SEP]movie1[SEP]starred in)[REC]'
    '("relation"[SEP]Mark Consuelos[SEP]movie1[SEP]starred in 2016 movie)[REC]'
    '("relation"[SEP]movie1[SEP]Vic[SEP]contains character)[END]')


def inside_out_responder(task, messages):
    from fedquest.llm.fake import message_text

    if task == "pattern":
        return INSIDE_OUT_PATTERN
    if task == "answer":
        facts = message_text(messages)
        both = [n for n in ("Richard Kind", "Judy Greer", "Amy Poehler")
                if f"({n} | starred in | Inside Out)" in facts and f"({n} | starred in | All We Had)" in facts]
        return f"<ANS>{both[0] if both else 'unknown'}</ANS>"
    return running_responder(task, messages)


async def test_the_answer_waits_for_its_join_partner_and_meets_it_by_intersection():
    from fedquest.llm.fake import fake_read

    coord, sites = coordinator(world=inside_out_world(), responder=inside_out_responder)
    shown = []
    sites["edge"].reader = lambda text: (shown.append(text), fake_read(text))[1]
    r = await coord.run(build_method("fedquest"), Question("io", "mmqa", INSIDE_OUT_Q, ["Richard Kind"]), "io")
    bound = [e["var"] for e in r.trace if e["event"] == "bound"]
    assert bound == ["?movie1", "?person1"]
    assert r.bindings == {"?movie1": "All We Had", "?person1": "Richard Kind"} and r.answer == "Richard Kind"
    assert "Candidates for X (found earlier" in shown[0] and "Amy Poehler" in shown[0]
    assert "midway3/mmqa/row/mc#3" in r.indexed and sites["midway3"].read_calls == []
    facts = {(f["head"], f["tail"]) for f in r.evidence}
    assert ("Judy Greer", "All We Had") in facts and ("Judy Greer", "Inside Out") not in facts
    stop = next(e for e in r.trace if e["event"] == "stop")
    assert r.matched and stop["read"] == r.ledger["objects_read"] == 2


async def test_disjoint_casts_reach_the_answerer_unresolved():
    coord, _ = coordinator(world=inside_out_world(shared=False), responder=inside_out_responder)
    r = await coord.run(build_method("fedquest"), Question("io2", "mmqa", INSIDE_OUT_Q, []), "io2")
    facts = {(f["head"], f["tail"]) for f in r.evidence}
    assert {("Amy Poehler", "Inside Out"), ("Judy Greer", "All We Had")} <= facts
    assert not r.matched


def metheny_world():
    albums = [("1992", "Secret Story", "Best Contemporary Jazz Album", "a jumping dog"),
              ("1989", "Letter from Home", "Best Jazz Fusion Performance", "a field at dusk"),
              ("1982", "Offramp", "Best Jazz Fusion Performance", "a hand on a road")]
    rows = [FakeObject(f"mmqa/row/pm#{i}", "row", "table", "Pat Metheny — Discography",
                       f"Pat Metheny | Discography | Year: {y}; Album: {t}; Award: Grammy Award for {c}", [], doc_id="pm")
            for i, (y, t, c, _) in enumerate(albums)]
    covers = [FakeObject(f"mmqa/image/pm{i}", "image", "image", t, t, [(t, "cover featuring", d)], doc_id=f"c{i}")
              for i, (_, t, _, d) in enumerate(albums)]
    midway3 = FakeSite("midway3", rows, rules([("secret story", "row", [rows[0].oid]),
                                               ("pat metheny", "row", [r.oid for r in rows])], {"row": []}))
    aurora = FakeSite("aurora", covers, rules([], {"image": [covers[1].oid, covers[2].oid, covers[0].oid]}))
    return fake_federation(midway3, aurora), {"midway3": midway3, "aurora": aurora}


METHENY_Q = "Pat Metheny won a Grammy Award in what category for the title that has a cover featuring a jumping dog?"
METHENY_PATTERN = (
    '("anchor_entity"[SEP]Pat Metheny)[REC]("entity"[SEP]album1)[REC]("entity"[SEP]Grammy Award)[REC]'
    '("answer_entity"[SEP]category1)[REC]("relation"[SEP]Pat Metheny[SEP]album1[SEP]released album)[REC]'
    '("relation"[SEP]album1[SEP]album1[SEP]has cover featuring a jumping dog)[REC]'
    '("relation"[SEP]album1[SEP]Grammy Award[SEP]won award)[REC]'
    '("relation"[SEP]Grammy Award[SEP]category1[SEP]awarded in category)[END]')


async def test_a_list_is_filtered_by_its_own_images_and_the_row_seen_for_it_answers_the_next_hop():
    coord, sites = coordinator(world=metheny_world(),
                               responder=lambda t, m: METHENY_PATTERN if t == "pattern" else running_responder(t, m))
    seen = recording(coord)
    r = await coord.run(build_method("fedquest"), Question("pm", "mmqa", METHENY_Q, []), "pm")
    sibling = [q for q in seen if q.key == "Pat Metheny | Discography"]
    assert sibling and sibling[0].k == 24 and sibling[0].collections == ["row"]
    covers = [q for q in seen if q.kind == "title" and q.collections == ["image"]]
    assert covers and set(covers[0].titles) == {"Secret Story", "Letter from Home", "Offramp"}
    assert covers[0].key == "has cover featuring a jumping dog"
    (req,) = sites["aurora"].read_requests
    assert req["unknowns"][0]["mode"] == "filter" and len(sites["aurora"].read_calls[0]) == 3
    assert r.bindings == {"?album1": "Secret Story", "?category1": "Grammy Award for Best Contemporary Jazz Album"}
    assert sites["midway3"].read_calls == [] and len(r.consulted) == 3
    assert r.indexed[0] == "midway3/mmqa/row/pm#0"


def test_a_mixed_set_of_conditions_runs_its_text_conditions_first():
    from fedquest.methods.fedquest import FedQuest, Held, _Run

    text = ('("anchor_entity"[SEP]Ben Piazza)[REC]("anchor_entity"[SEP]baseball)[REC]("entity"[SEP]film1)[REC]'
            '("answer_entity"[SEP]film1)[REC]("relation"[SEP]Ben Piazza[SEP]film1[SEP]acted in)[REC]'
            '("relation"[SEP]film1[SEP]baseball[SEP]poster depicts)[END]')
    p = hint_modalities(prompts.parse_pattern(text, "Which film with Ben Piazza has a poster of baseball?"))
    run = _Run(FedQuest(), Question("x", "mmqa", "Which film with Ben Piazza has a poster of baseball?", []), None, p)
    kind, asks = run.step()
    assert kind == "text" and [a.value for a in asks] == ["Ben Piazza"]
    run.asked.add((asks[0].cond.id, "ben piazza"))
    run.values["?film1"]["the bad news bears"] = Held("The Bad News Bears", {asks[0].cond.id: {"ben piazza"}})
    kind, asks = run.step()
    assert kind == "filter" and [a.value for a in asks] == ["baseball"]


async def test_a_visual_condition_to_a_named_constant_filters_the_candidates_own_images():
    pattern = ('("anchor_entity"[SEP]Vakkantham Vamsi)[REC]("entity"[SEP]film1)[REC]("entity"[SEP]motorcycle)[REC]'
               '("answer_entity"[SEP]film1)[REC]("relation"[SEP]Vakkantham Vamsi[SEP]film1[SEP]wrote)[REC]'
               '("relation"[SEP]film1[SEP]motorcycle[SEP]poster features)[END]')
    coord, sites = coordinator(world=poster_world(),
                               responder=lambda t, m: pattern if t == "pattern" else running_responder(t, m))
    q = Question("q8", "mmqa", "Which film written by Vakkantham Vamsi has a poster featuring a motorcycle?",
                 ["Race Gurram"])
    seen = recording(coord)
    r = await coord.run(build_method("fedquest"), q, "constant")
    assert any(e["tail"] == "motorcycle" and e["modality"] == "image" for e in r.pattern["edges"])
    posters = [p for p in seen if p.kind == "title" and p.collections == ["image"]]
    assert posters and set(posters[0].titles) == {"Race Gurram", "Kick 2"}
    assert posters[0].key == "poster features motorcycle"
    (req,) = sites["aurora"].read_requests
    assert req["unknowns"][0]["mode"] == "filter" and r.bindings == {"?film1": "Race Gurram"}

async def test_a_text_hop_reads_no_images_and_a_visual_hop_reads_the_values_own_images():
    coord, sites = coordinator()
    row = sites["midway3"].objects["mmqa/row/t1#6"]
    row.text = "Ben Piazza | Filmography | " + "Notes: stage and screen; " * 12 + row.text.split(" | ", 2)[2]
    seen = recording(coord)
    r = await coord.run(build_method("fedquest"), Q, "route")
    first = [oid for req in sites["midway3"].read_calls for oid in req]
    assert first and all("row" in oid for oid in first)
    (req,) = sites["aurora"].read_requests
    assert any(p.kind == "title" and p.titles == ["The Bad News Bears"] and p.collections == ["image"] for p in seen)
    assert sites["aurora"].read_calls[0][0] == "mmqa/image/p1" and r.answer == "baseball"


async def test_a_truncated_entry_is_read_at_the_coordinator_before_its_object():
    coord, sites = coordinator()
    sites["midway3"].objects["mmqa/row/t1#6"].text += " " + "and more notes " * 20
    r = await coord.run(build_method("fedquest"), Q, "snippet")
    first = next(e for e in r.trace if e["event"] == "bind_index")
    assert first["values"] == {"film1": ["The Bad News Bears"]}
    assert not sites["midway3"].read_calls and r.answer == "baseball"


@pytest.mark.parametrize("label", FEDQUESTS)
async def test_with_no_budget_left_the_coordinator_still_binds_from_entries(label):
    coord, sites = coordinator()
    r = await coord.run(build_method(label, {"objects": 0}), Q, "zero")
    assert r.bindings == {"?film1": "The Bad News Bears"} and r.consulted == []
    assert not any(s.read_calls for s in sites.values())


async def test_a_one_hop_table_question_is_answered_with_zero_site_reads():
    pattern = ('("anchor_entity"[SEP]Ben Piazza)[REC]("entity"[SEP]role1)[REC]("answer_entity"[SEP]role1)[REC]'
               '("relation"[SEP]Ben Piazza[SEP]role1[SEP]played role in 1976 film)[END]')
    coord, sites = coordinator(responder=lambda t, m: pattern if t == "pattern" else running_responder(t, m))
    q = Question("t", "mmqa", "What role did Ben Piazza play in his 1976 film?", ["Bob Whitewood"])
    r = await coord.run(build_method("fedquest"), q, "tableq")
    assert r.bindings == {"?role1": "Bob Whitewood"} and r.consulted == [] and r.indexed == ["midway3/mmqa/row/t1#6"]


async def test_no_coordinator_call_when_no_complete_entry_names_a_value_of_the_step():
    coord, sites = coordinator()
    for oid in ("mmqa/row/t1#6", "mmqa/row/t1#2"):
        o = sites["midway3"].objects[oid]
        o.title, o.text = "Walter Matthau — Filmography", o.text.replace("Ben Piazza |", "Walter Matthau |")
    await coord.run(build_method("fedquest"), Q, "gate")
    assert calls(coord, "read_index") == 0


def test_a_visual_question_whose_pattern_lost_the_picture_gets_a_question_level_condition():
    from fedquest.methods.fedquest import FALLBACK_VISUAL, conditions

    q = "What actors were in the film written by Vakkantham Vamsi whose poster features a motorcycle?"
    text = ('("anchor_entity"[SEP]Vakkantham Vamsi)[REC]("entity"[SEP]film1)[REC]("answer_entity"[SEP]actor1)[REC]'
            '("relation"[SEP]Vakkantham Vamsi[SEP]film1[SEP]wrote)[REC]("relation"[SEP]film1[SEP]actor1[SEP]starred)[END]')
    conds = conditions(prompts.parse_pattern(text, q), q)
    assert [(c.var, c.text, c.visual) for c in conds if c.edge is None] == [("?film1", FALLBACK_VISUAL, True)]
    assert not any(c.edge is None for c in conditions(prompts.parse_pattern(text, "Who starred in it?"), "Who?"))


def test_unconstrained_leaves_are_pruned_only_when_the_answer_is_explicit():
    from fedquest.methods.fedquest import holders, prune

    text = ('("anchor_entity"[SEP]Vakkantham Vamsi)[REC]("entity"[SEP]film1)[REC]("entity"[SEP]film)[REC]'
            '("answer_entity"[SEP]actor1)[REC]("relation"[SEP]Vakkantham Vamsi[SEP]film1[SEP]wrote)[REC]'
            '("relation"[SEP]film1[SEP]actor1[SEP]starred)[REC]("relation"[SEP]film1[SEP]film[SEP]is a)[END]')
    p = prune(prompts.parse_pattern(text, "q"))
    assert p.variables == ["?film1", "?actor1"] and len(p.edges) == 2
    assert list(holders(p).values()) == ["?actor1"]
    kept = prune(prompts.parse_pattern(text.replace('("answer_entity"[SEP]actor1)[REC]', ""), "q"))
    assert "?film" in kept.variables


async def test_a_fallback_pattern_makes_one_question_keyed_round():
    coord, sites = coordinator(responder=lambda t, m: "garbage" if t == "pattern" else running_responder(t, m))
    r = await coord.run(build_method("fedquest"), Q, "fallback")
    assert r.error is None and set(sites["aurora"].probes) == {RUNNING_QUESTION}


async def test_a_yes_no_answer_is_read_off_an_image():
    pattern = ('("anchor_entity"[SEP]Race Gurram)[REC]("entity"[SEP]answer1)[REC]("answer_entity"[SEP]answer1)[REC]'
               '("relation"[SEP]Race Gurram[SEP]answer1[SEP]poster features a motorcycle)[END]')
    coord, sites = coordinator(world=poster_world(),
                               responder=lambda t, m: pattern if t == "pattern" else running_responder(t, m))
    sites["aurora"].reader = lambda text: {"answer1": [{"entity": "yes", "object": 1, "conditions": [1]}]}
    q = Question("yn", "mmqa", "Does the poster of Race Gurram feature a motorcycle?", ["yes"])
    r = await coord.run(build_method("fedquest"), q, "yn")
    assert r.bindings == {"?answer1": "yes"} and sites["aurora"].read_calls[0][0] == "mmqa/image/rg"


async def test_a_comparison_binds_both_branches_before_it_stops():
    import re

    from fedquest.llm.fake import message_text

    pattern = ('("anchor_entity"[SEP]The Bad News Bears)[REC]("anchor_entity"[SEP]No Exit)[REC]("entity"[SEP]year1)'
               '[REC]("entity"[SEP]year2)[REC]("relation"[SEP]The Bad News Bears[SEP]year1[SEP]released in)[REC]'
               '("relation"[SEP]No Exit[SEP]year2[SEP]released in)[END]')

    def responder(task, messages):
        if task == "pattern":
            return pattern
        if task == "read_index":
            text = message_text(messages)
            rows = dict(re.findall(r"^\[(\d+)\] \(table row\) .*?Year: (\d{4}); Title: ([^;]+);", text, re.M) and
                        [(t, (int(n), y)) for n, y, t in re.findall(r"^\[(\d+)\] \(table row\) .*?Year: (\d{4}); "
                                                                    r"Title: ([^;]+);", text, re.M)])
            return json.dumps({v: [{"entity": rows[t][1], "object": rows[t][0], "conditions": [1]}]
                               for v, t in (("year1", "The Bad News Bears"), ("year2", "No Exit"))
                               if f"- {v} " in text and t in rows})
        return running_responder(task, messages)

    coord, _ = coordinator(responder=responder)
    q = Question("cmp", "mmqa", "Which was released first, The Bad News Bears or No Exit?", ["No Exit"])
    r = await coord.run(build_method("fedquest"), q, "cmp")
    assert r.pattern["raw"]["answer_inferred"] and r.bindings == {"?year1": "1976", "?year2": "1962"}
    assert r.matched and r.consulted == []


async def test_a_hopeless_question_costs_at_most_its_bound():
    hopeless = ('("anchor_entity"[SEP]Nobody)[REC]("entity"[SEP]thing1)[REC]("answer_entity"[SEP]thing1)[REC]'
                '("relation"[SEP]Nobody[SEP]thing1[SEP]owns)[END]')
    coord, sites = coordinator(responder=lambda t, m: hopeless if t == "pattern" else running_responder(t, m))
    q = Question("h", "mmqa", "What does Nobody own?", [])
    r = await coord.run(build_method("fedquest", {"objects": 3}), q, "hopeless")
    rounds = sum(e["event"] == "bind_index" for e in r.trace)
    assert calls(coord, "pattern") == 1 and calls(coord, "answer") == 1 and calls(coord, "verify") == 0
    assert calls(coord, "read_index") == rounds and len(r.consulted) <= 3

def durban_world():
    text = FakeObject("mmqa/text/dur#0", "text", "text", "Durban",
                      "Durban is a coastal city. (beach of Durban | contains | sand); (beach of Durban | contains | piers)",
                      [], doc_id="dur")
    beach = FakeObject("mmqa/image/dur", "image", "image", "Durban", "Durban",
                       [("beach of Durban", "contains", "sand"), ("beach of Durban", "contains multi-colored", "umbrella")],
                       doc_id="durimg")
    aurora = FakeSite("aurora", [text, beach], rules([("durban", "text", [text.oid]), ("durban", "image", [beach.oid])]))
    return fake_federation(aurora), {"aurora": aurora}


async def test_a_visual_condition_whose_candidates_have_no_picture_reads_the_anchors_picture():
    pattern = ('("anchor_entity"[SEP]beach of Durban)[REC]("entity"[SEP]item1)[REC]("answer_entity"[SEP]item1)[REC]'
               '("relation"[SEP]beach of Durban[SEP]item1[SEP]contains)[REC]'
               '("relation"[SEP]item1[SEP]item1[SEP]is multi-colored)[END]')
    coord, sites = coordinator(world=durban_world(),
                               responder=lambda t, m: pattern if t == "pattern" else running_responder(t, m))
    q = Question("dur", "mmqa", "On the beach of Durban, what item is multi-colored?", ["umbrella"])
    r = await coord.run(build_method("fedquest"), q, "durban")
    (hop,) = sites["aurora"].read_requests
    assert hop["unknowns"][0]["mode"] == "value"
    assert sites["aurora"].read_calls == [["mmqa/image/dur"]]
    assert r.bindings == {"?item1": "umbrella"}


def flag_world():
    flag = FakeObject("mmqa/image/dk", "image", "image", "Denmark", "Denmark",
                      [("flag for Denmark", "contains", "2"), ("flag for Denmark", "has color", "red")], doc_id="dk")
    aurora = FakeSite("aurora", [flag], rules([("denmark", "image", [flag.oid])]))
    return fake_federation(aurora), {"aurora": aurora}


async def test_a_single_unknown_visual_question_reads_the_named_entitys_picture():
    pattern = ('("anchor_entity"[SEP]flag for Denmark)[REC]("entity"[SEP]red shape1)[REC]'
               '("answer_entity"[SEP]red shape1)[REC]("relation"[SEP]flag for Denmark[SEP]red shape1[SEP]contains)[END]')
    coord, sites = coordinator(world=flag_world(),
                               responder=lambda t, m: pattern if t == "pattern" else running_responder(t, m))
    q = Question("dk", "mmqa", "How many separate red shapes are on the flag for Denmark?", ["2"])
    r = await coord.run(build_method("fedquest"), q, "flag")
    assert sites["aurora"].read_calls == [["mmqa/image/dk"]]
    assert r.bindings == {"?red shape1": "2"}


async def test_the_token_budget_bounds_every_read_and_keeps_the_next_hops_first_call():
    from fedquest.methods.fedquest import FedQuest, _Run

    coord, sites = coordinator()
    poor = await coord.run(build_method("fedquest", {"tokens": 1200}), Q, "tokens")
    run = _Run(FedQuest(), Q, coord.context("x"), prompts.parse_pattern(RUNNING_PATTERN, Q.question))
    await run.setup()
    kept = run.prior_call(image=True, kind="bridge")
    assert 1200 - kept < 301 + 35 + run.named([run.unknowns(run.step()[1])[0][0]])
    assert poor.ledger["evidence_tokens"] == 0 and not any(s.read_calls for s in sites.values())
    enough = await coordinator()[0].run(build_method("fedquest", {"tokens": 2000}), Q, "enough")
    assert enough.answer == "baseball" and enough.ledger["evidence_tokens"] <= 2000


def truncated_rows():
    fed, sites = running_example()
    for oid in ("mmqa/row/t1#6", "mmqa/row/t1#2"):
        o = sites["midway3"].objects[oid]
        o.text = "Ben Piazza | Filmography | " + "Notes: stage and screen; " * 12 + o.text.split(" | ", 2)[2]
    return fed, sites


async def test_the_similarity_order_reads_mgrag_feds_candidates_one_object_per_call_by_fused_rank():
    from fedquest.methods.common import rrf, split_lists

    a, sa = coordinator(world=truncated_rows())
    seen = recording(a)
    r = await a.run(build_method("fedquest_q_sim_unit"), Q, "sim")
    assert {p.key for p in seen} == {RUNNING_QUESTION}
    reads = [e for e in r.trace if e["event"] in ("read", "bind_index")]
    assert all(len(e["read"]) == 1 for e in reads if e["event"] == "read")
    assert all(e["entries"] == 1 for e in reads if e["event"] == "bind_index")
    b, _ = coordinator(world=truncated_rows())
    cands = [c for cs in await b.fed.probe([__import__("fedquest.types", fromlist=["ProbeRequest"]).ProbeRequest(
        RUNNING_QUESTION, "question", 5, collections=["row", "text"])]) for c in cs]
    fused = [c.ref for c in rrf(split_lists(cands), salt=Q.qid)]
    first = next(e for e in r.trace if e["event"] in ("read", "bind_index"))
    assert first["event"] == "bind_index" or (first["site"], first["read"][0]) == fused[0]


async def test_a_filter_drops_what_the_reader_judged_no():
    import re

    coord, sites = coordinator(world=metheny_world(),
                               responder=lambda t, m: METHENY_PATTERN if t == "pattern" else running_responder(t, m))

    def reader(text):
        lines = re.findall(r"^\[(\d+)\] \(image\) ([^:]+):", text, re.M)
        return {"album1": [{"entity": t if t == "Secret Story" else "no", "object": int(n), "conditions": [1]}
                           for n, t in lines]}

    sites["aurora"].reader = reader
    r = await coord.run(build_method("fedquest"), Question("pm2", "mmqa", METHENY_Q, []), "judged")
    assert r.bindings["?album1"] == "Secret Story"
    assert not any(f["head"] in ("Letter from Home", "Offramp") and "cover" in f["relation"] for f in r.evidence)


def club_world():
    rows = [FakeObject(f"mmqa/row/wfl#{i}", "row", "table", "2003-04 Wessex Football League — League table",
                       f"2003-04 Wessex Football League | League table | Pos: {i + 1}; Club: {c}; GD: {gd}", [],
                       doc_id="wfl") for i, (c, gd) in enumerate([("Brockenhurst", "-25"), ("Stockbridge", "+3")])]
    logos = [FakeObject(f"mmqa/image/{c[:5].lower()}", "image", "image", f"{c} F.C.", f"{c} F.C.",
                        [(f"{c} F.C.", "logo shows", animal)], doc_id=c[:5])
             for c, animal in (("Brockenhurst", "a badger"), ("Stockbridge", "a bird"))]
    midway3 = FakeSite("midway3", rows, rules([("wessex", "row", [r.oid for r in rows])], {"row": []}))
    aurora = FakeSite("aurora", logos, rules([], {"image": [lg.oid for lg in logos]}))
    return fake_federation(midway3, aurora), {"midway3": midway3, "aurora": aurora}


async def test_a_candidate_finds_its_picture_under_its_pages_title():
    pattern = ('("anchor_entity"[SEP]2003-04 Wessex Football League)[REC]("entity"[SEP]club1)[REC]'
               '("answer_entity"[SEP]club1)[REC]("relation"[SEP]2003-04 Wessex Football League[SEP]club1[SEP]contains club)'
               '[REC]("relation"[SEP]club1[SEP]club1[SEP]has a badger on its logo)[END]')
    coord, sites = coordinator(world=club_world(),
                               responder=lambda t, m: pattern if t == "pattern" else running_responder(t, m))
    q = Question("club", "mmqa", "Which club in the 2003-04 Wessex Football League table has a badger on its logo?",
                 ["Brockenhurst"])
    r = await coord.run(build_method("fedquest"), q, "club")
    assert sorted(sites["aurora"].read_calls[0]) == ["mmqa/image/brock", "mmqa/image/stock"]
    assert r.bindings == {"?club1": "Brockenhurst"}


@pytest.mark.parametrize("label", FEDQUESTS)
async def test_a_round_never_reads_more_than_its_share_however_many_sites_hold_images(label):
    rows = [FakeObject(f"mmqa/row/vv#{i}", "row", "table", "Vakkantham Vamsi — Films",
                       f"Vakkantham Vamsi | Films | Year: {y}; Title: {t}", [], doc_id="vv")
            for i, (y, t) in enumerate([(2014, "Race Gurram"), (2016, "Kick 2")])]
    posters = [FakeObject(f"mmqa/image/{i}", "image", "image", t, t, [(t, "poster features", d)], doc_id=f"p{i}")
               for i, (t, d) in enumerate([("Race Gurram", "motorcycle"), ("Kick 2", "a man in a suit")])]
    midway3 = FakeSite("midway3", rows, rules([("vakkantham", "row", [r.oid for r in rows])], {"row": []}))
    one = FakeSite("aurora", posters[:1], rules([], {"image": [posters[0].oid]}))
    two = FakeSite("aurora2", posters[1:], rules([], {"image": [posters[1].oid]}))
    coord, sites = coordinator(world=(fake_federation(midway3, one, two), {"midway3": midway3, "aurora": one,
                                                                           "aurora2": two}),
                               responder=lambda t, m: POSTER_PATTERN if t == "pattern" else running_responder(t, m))
    q = Question("q9", "mmqa", "Which film written by Vakkantham Vamsi has a poster featuring a motorcycle?", [])
    r = await coord.run(build_method(label, {"objects": 1}), q, "split")
    assert r.ledger["objects_read"] <= 1


async def test_an_object_judged_for_one_condition_is_still_new_for_the_next():
    fed, sites = inside_out_world()
    aw = "mmqa/text/aw#0"
    sites["edge"].ranker = rules([("mark consuelos", "text", [aw]), ("all we had", "text", [aw])], {"text": []})
    coord, _ = coordinator(world=(fed, sites), responder=inside_out_responder)
    r = await coord.run(build_method("fedquest"), Question("io3", "mmqa", INSIDE_OUT_Q, ["Richard Kind"]), "io3")
    assert r.bindings["?person1"] == "Richard Kind"


def test_the_fallback_patterns_condition_has_no_known_value():
    from fedquest.methods.fedquest import FedQuest, _Run

    p = prompts.parse_pattern("garbage", RUNNING_QUESTION)
    run = _Run(FedQuest(), Q, None, p)
    _, asks = run.step()
    (u,), _ = run.unknowns(asks)
    assert u.conditions[0] == "X answers the question" and u.values[0] is None

def run_of(pattern_text, question, planner="hop"):
    from fedquest.methods.fedquest import FedQuest, _Run, prune

    p = prune(hint_modalities(prompts.parse_pattern(pattern_text, question)), question)
    return _Run(FedQuest(planner=planner), Question("x", "mmqa", question, []), None, p)


def test_a_leaf_the_question_describes_becomes_a_constant_and_a_type_leaf_goes():
    q = "What 2001 New Orleans Saints regular season opponent made it to the Super Bowl in 2017?"
    text = ('("anchor_entity"[SEP]2001 New Orleans Saints season)[REC]("entity"[SEP]team1)[REC]'
            '("entity"[SEP]Super Bowl 2017)[REC]("entity"[SEP]team)[REC]("answer_entity"[SEP]team1)[REC]'
            '("relation"[SEP]2001 New Orleans Saints season[SEP]team1[SEP]has regular season opponent)[REC]'
            '("relation"[SEP]team1[SEP]Super Bowl 2017[SEP]made it to)[REC]("relation"[SEP]team1[SEP]team[SEP]is a)[END]')
    run = run_of(text, q)
    assert run.p.variables == ["?team1"] and "Super Bowl 2017" in run.p.anchors
    assert sorted(c.other for c in run.conds) == ["2001 New Orleans Saints season", "Super Bowl 2017"]


@pytest.mark.parametrize("planner", ["hop", "query"])
def test_a_relation_that_names_another_unknown_waits_for_it(planner):
    q = "What role did Maggie Smith play in the film where Michelle Dockery played Lady Mary Talbot?"
    text = ('("anchor_entity"[SEP]Maggie Smith)[REC]("anchor_entity"[SEP]Michelle Dockery)[REC]("entity"[SEP]role1)'
            '[REC]("entity"[SEP]film1)[REC]("answer_entity"[SEP]role1)[REC]'
            '("relation"[SEP]Maggie Smith[SEP]role1[SEP]played role in film1)[REC]'
            '("relation"[SEP]Michelle Dockery[SEP]film1[SEP]played Lady Mary Talbot in)[END]')
    run = run_of(text, q, planner)
    _, asks = run.step()
    assert [a.var for a in asks] == ["?film1"]
    run.asked.add((asks[0].cond.id, "michelle dockery"))
    run.bound["?film1"] = ["Downton Abbey"]
    _, asks = run.step()
    assert [(a.var, a.relation) for a in asks] == [("?role1", "played role in Downton Abbey")]


def test_a_filter_resolves_the_intersection_of_the_text_conditions_not_their_union():
    from fedquest.methods.fedquest import Held

    q = "Which film starring Tom Hanks and Meg Ryan has a poster featuring a dog?"
    text = ('("anchor_entity"[SEP]Tom Hanks)[REC]("anchor_entity"[SEP]Meg Ryan)[REC]("entity"[SEP]film1)[REC]'
            '("answer_entity"[SEP]film1)[REC]("relation"[SEP]Tom Hanks[SEP]film1[SEP]starred in)[REC]'
            '("relation"[SEP]Meg Ryan[SEP]film1[SEP]starred in)[REC]'
            '("relation"[SEP]film1[SEP]film1[SEP]poster featuring a dog)[END]')
    run = run_of(text, q)
    hanks, ryan = (c for c in run.conds if c.edge is not None and not c.hop)
    for name, conds in (("Big", [hanks]), ("Splash", [hanks]), ("Sleepless in Seattle", [hanks, ryan]),
                        ("When Harry Met Sally", [ryan])):
        run.values["?film1"][name.lower()] = Held(name, {c.id: {c.other.lower()} for c in conds})
    run.asked |= {(hanks.id, "tom hanks"), (ryan.id, "meg ryan")}
    kind, asks = run.step()
    assert kind == "filter"
    run.asked |= {(a.cond.id, "") for a in asks}
    (u,), _ = run.unknowns(asks, mode="filter")
    assert u.candidates == ["Sleepless in Seattle"]


def test_a_depiction_of_keeps_the_picture_with_its_head():
    q = "There is a flaming torch on the logo for which team listed in the scoring leaders of the 2013-14 QMJHL season?"
    text = ('("anchor_entity"[SEP]2013-14 QMJHL season)[REC]("entity"[SEP]team1)[REC]("answer_entity"[SEP]team1)[REC]'
            '("relation"[SEP]2013-14 QMJHL season[SEP]team1[SEP]has scoring leader from)[REC]'
            '("relation"[SEP]team1[SEP]flaming torch[SEP]logo contains image of)[END]')
    run = run_of(text, q)
    visual = [c for c in run.conds if c.visual and c.edge is not None]
    assert visual and run.owns(visual[0])
    assert not run.owns(next(iter([c for c in run_of(
        '("anchor_entity"[SEP]Jaws)[REC]("entity"[SEP]shark1)[REC]("answer_entity"[SEP]shark1)[REC]'
        '("relation"[SEP]shark1[SEP]Jaws[SEP]is shown in poster of)[END]', "What is shown in the poster of Jaws?").conds
        if c.edge is not None])))


def test_values_merge_by_name_and_a_sequel_stays_apart():
    from fedquest.methods.fedquest import Asked

    run = run_of('("anchor_entity"[SEP]Robert Downey Jr.)[REC]("anchor_entity"[SEP]Mickey Rourke)[REC]'
                 '("entity"[SEP]film1)[REC]("answer_entity"[SEP]film1)[REC]'
                 '("relation"[SEP]Robert Downey Jr.[SEP]film1[SEP]starred in)[REC]'
                 '("relation"[SEP]Mickey Rourke[SEP]film1[SEP]starred in)[END]',
                 "Which film starred both Robert Downey Jr. and Mickey Rourke?")
    rdj, mr = (c for c in run.conds)
    by_var = {"?film1": [Asked("?film1", rdj, "Robert Downey Jr.", "starred in"),
                         Asked("?film1", mr, "Mickey Rourke", "starred in")]}
    run.merge({"film1": [{"entity": "Iron Man", "site": "m", "oid": "rdj#0", "conditions": [1]},
                         {"entity": "Iron Man 2", "site": "m", "oid": "mr#0", "conditions": [2]},
                         {"entity": "The Iron Man 2", "site": "m", "oid": "rdj#1", "conditions": [1]}]}, by_var, "v", {})
    assert sorted(run.values["?film1"]) == ["iron man", "iron man 2"]
    facts = {(f.head, f.tail) for h in run.values["?film1"].values() for f in h.facts}
    assert ("Mickey Rourke", "Iron Man") not in facts


async def test_a_list_still_waiting_for_a_condition_reads_its_sites_after_one_entry():
    fed, sites = inside_out_world()
    awards = FakeObject("mmqa/row/ioa#0", "row", "table", "Inside Out — Awards",
                        "Inside Out | Awards | Name: Phyllis Smith; Award: Annie Award for Voice Acting", [], doc_id="ioa")
    sites["midway3"].objects[awards.oid] = awards
    sites["midway3"].ranker = rules([("mark consuelos", "row", ["mmqa/row/mc#3"]), ("inside out", "row", [awards.oid])],
                                    {"row": []})
    coord, _ = coordinator(world=(fed, sites), responder=inside_out_responder)
    r = await coord.run(build_method("fedquest"), Question("io4", "mmqa", INSIDE_OUT_Q, ["Richard Kind"]), "io4")
    assert r.bindings["?person1"] == "Richard Kind"


@pytest.mark.parametrize("label", FEDQUESTS)
async def test_a_pattern_without_an_unknown_asks_the_question_itself(label):
    pattern = ('("anchor_entity"[SEP]Ben Piazza)[REC]("anchor_entity"[SEP]The Bad News Bears)[REC]'
               '("relation"[SEP]Ben Piazza[SEP]The Bad News Bears[SEP]acted in)[END]')
    coord, sites = coordinator(responder=lambda t, m: pattern if t == "pattern" else running_responder(t, m))
    q = Question("yn2", "mmqa", "Did Ben Piazza act in The Bad News Bears?", ["yes"])
    r = await coord.run(build_method(label), q, "novar")
    assert r.pattern["variables"] == ["?answer"] and any(q.question in s.probes for s in sites.values())
    assert r.ledger["rounds"] >= 1

def test_a_picture_variable_folds_into_its_object():
    from fedquest.methods.common import fold_media
    from fedquest.types import Edge, Pattern

    pig = Pattern(["Billy West", "Hamilton J. Pig"], ["?video_game1", "?poster1", "?pig_presence"],
                  [Edge(0, "Billy West", "voiced by", "Hamilton J. Pig"),
                   Edge(1, "Hamilton J. Pig", "character in", "?video_game1"),
                   Edge(2, "?video_game1", "has poster", "?poster1"),
                   Edge(3, "?poster1", "contains image of pig", "?pig_presence")], "?pig_presence")
    out = fold_media(pig)
    assert [(e.head, e.relation, e.tail, e.modality) for e in out.edges][1:] == [
        ("Hamilton J. Pig", "character in", "?video_game1", None),
        ("?video_game1", "poster contains image of pig", "?pig_presence", "image")]
    assert out.variables == ["?video_game1", "?pig_presence"]
    loop = Pattern(["Garth Jennings"], ["?film1", "?poster1"], [Edge(0, "Garth Jennings", "had a role in", "?film1"),
                   Edge(1, "?film1", "has title poster of", "?poster1"),
                   Edge(2, "?poster1", "depicts two police officers", "?film1")], "?film1")
    assert [e.id for e in fold_media(loop).edges] == [0]
    asked = Pattern(["Ready Player One"], ["?poster1"], [Edge(0, "Ready Player One", "has poster", "?poster1")],
                    "?poster1")
    assert len(fold_media(asked).edges) == 1
    other = Pattern(["Dr. Fox"], ["?movie_poster1", "?kid"], [Edge(0, "Dr. Fox", "featured in movie poster of",
                    "?movie_poster1"), Edge(1, "?movie_poster1", "contains silhouette of", "?kid")], "?kid")
    assert len(fold_media(other).edges) == 2


def test_visual_relations_get_the_image_hint():
    p = hint_modalities(prompts.parse_pattern(RUNNING_PATTERN, RUNNING_QUESTION))
    assert [e.modality for e in p.edges] == [None, "image"]


async def test_mgrag_extracts_in_rank_order_and_never_materializes():
    kg = MemoryKG()
    coord, sites = coordinator(kg=kg)
    r = await coord.run(build_method("mgrag"), Q, "m")
    rank1 = {"aurora/mmqa/image/p9", "midway3/mmqa/row/t1#6", "edge/mmqa/text/b1#0"}
    assert set(r.consulted[:3]) <= rank1 and r.consulted[0] in rank1
    again = await coordinator()[0].run(build_method("mgrag"), Q, "m2")
    assert again.consulted == r.consulted
    assert all(mode == "query" for _, mode in sites["aurora"].extract_calls)
    assert await kg.facts("m") == []


def test_fusion_ties_are_site_neutral_and_fixed_per_question():
    from fedquest.methods.common import rrf
    from fedquest.types import Candidate

    lists = [[Candidate(s, "c", f"{s}-o{r}", "text", r, "", 1) for r in (1, 2)] for s in ("aurora", "edge", "midway3")]
    orders = {tuple(c.site for c in rrf(lists, salt=f"q{i}")[:3]) for i in range(40)}
    assert len(orders) > 1
    assert all(c.rank == 1 for c in rrf(lists, salt="q1")[:3])
    assert rrf(lists, salt="q1") == rrf(lists, salt="q1")


@pytest.mark.parametrize("name,params", [
    ("vanilla_rag", {}), ("iter_rag", {}), ("fedquest_q", {}), ("fedquest_q_unit", {}),
    ("fedquest_q_cap", {}), ("fedquest_q_sim", {}), ("fedquest_q_sim_unit", {}), ("fedquest_q_sim_cap", {}),
    ("fedquest", {"size": 2}), ("fedquest", {"width": 2}),
    ("mgrag", {"order": "relevance"}), ("iter_rag", {"grounded": False}),
])
async def test_every_method_runs(name, params):
    coord, _ = coordinator()
    r = await coord.run(build_method(name, params), Q, f"ns-{name}")
    assert r.error is None, r.error
    assert r.ledger["rounds"] >= 1 and r.ledger["bytes_summary"] > 0


async def test_closed_book_answers_without_any_retrieval():
    coord, sites = coordinator()
    r = await coord.run(build_method("closed_book"), Q, "cb")
    assert r.error is None and r.answer == "basketball"
    assert r.ledger["rounds"] == 0 and not any(s.probes for s in sites.values())


async def test_failures_become_error_rows():
    class Broken(Method):
        async def answer(self, q, ctx):
            raise RuntimeError("boom")

    coord, _ = coordinator()
    r = await coord.run(Broken(), Q, "e")
    assert r.error == "RuntimeError: boom" and r.answer == ""


def test_unknown_parameters_are_rejected():
    with pytest.raises(TypeError):
        build_method("fedquest", {"budgte": 3})


async def test_calibration_probes_label_the_methods_own_access_paths_per_round():
    coord, sites = coordinator()
    q = Question("cal", "mmqa", Q.question, ["baseball"], qtype="Compose(ImageQ,TableQ)", bridge=["The Bad News Bears"],
                 meta={"gold": {"first": ["row:t1#6"], "final": ["image:p1"]}, "gold_tables_unknown": [],
                       "table_id": "t1", "family": "compose"})
    r = await coord.run(build_method("explore"), q, "explore")
    obs = [e for e in r.trace if e["event"] == "obs" and e.get("path") != "gold"]
    gold = [e for e in r.trace if e["event"] == "obs" and e.get("path") == "gold"]
    assert {g["round"] for g in gold} == {"first", "late"} and all(g["groups"] for g in gold)
    fields = {"qid", "round", "kind", "site", "collection", "oid", "modality", "rank", "signal", "entry", "tokens",
              "states", "probe", "evidence"}
    assert all(fields <= set(o) for o in obs)
    evidence = {(o["round"], o["kind"], o["oid"]) for o in obs if o["evidence"]}
    assert ("first", "bound", "mmqa/row/t1#6") in evidence
    assert ("late", "title", "mmqa/image/p1") in evidence
    assert {o["kind"] for o in obs} >= {"question", "bound", "bridge", "title", "sibling"}
    assert not any(s.read_calls for s in sites.values())


async def test_a_calibration_round_counts_only_the_gold_its_paths_reach_and_a_yes_no_labels_no_row():
    coord, _ = coordinator()
    q = Question("cal", "mmqa", Q.question, ["baseball"], qtype="Compose(ImageQ,TableQ)", bridge=["The Bad News Bears"],
                 meta={"gold": {"first": ["row:t1#6"], "final": ["image:p1", "text:elsewhere"]},
                       "gold_tables_unknown": [], "table_id": "t1", "family": "compose"})
    r = await coord.run(build_method("explore"), q, "explore")
    late = [e for e in r.trace if e["event"] == "obs" and e.get("path") == "gold" and e["round"] == "late"]
    assert [g["groups"] for g in late] == [["image"]]

    coord, _ = coordinator()
    q = Question("yn", "mmqa", "Did The Bad News Bears win?", ["No"], qtype="TableQ",
                 meta={"gold": {"first": [], "final": []}, "gold_tables_unknown": ["t1"], "table_id": "t1"})
    r = await coord.run(build_method("explore"), q, "explore")
    rows = [e for e in r.trace if e["event"] == "obs" and e.get("modality") == "table"]
    assert rows and all(e["evidence"] is None for e in rows)
    row = Candidate("aurora", "mmqa", "mmqa/row/t1#0", "table", 1, "Bears: Bears | Films | Title: The Bad News Bears",
                    40, doc_id="t1", complete=True)
    assert is_evidence(row, set(), set(), ["No"], "t1") is False
    assert is_evidence(row, set(), set(), ["The Bad News Bears"], "t1") is True


BORN_PATTERN = ('("anchor_entity"[SEP]Ben Piazza)[REC]("entity"[SEP]year1)[REC]("answer_entity"[SEP]year1)[REC]'
                '("relation"[SEP]Ben Piazza[SEP]year1[SEP]born in)[END]')


def born_world():
    row = FakeObject("mmqa/row/bw#0", "row", "table", "Broadway Revival — Cast",
                     "Broadway Revival | Cast | Actor: Ben Piazza; Born: 1950", [], doc_id="bw")
    bio = FakeObject("mmqa/text/bp#0", "text", "text", "Ben Piazza",
                     "Ben Piazza was an American actor. (Ben Piazza | born in | 1933)", [], doc_id="bp")
    midway3 = FakeSite("midway3", [row], rules([("ben piazza", "row", [row.oid])], {"row": [row.oid]}))
    edge = FakeSite("edge", [bio], rules([("ben piazza", "text", [bio.oid])], {"text": [bio.oid]}),
                      policy="facts")
    return fake_federation(midway3, edge), {"midway3": midway3, "edge": edge}


async def test_a_row_of_another_entitys_table_cannot_settle_a_one_hop_question_alone():
    coord, sites = coordinator(world=born_world(),
                               responder=lambda t, m: BORN_PATTERN if t == "pattern" else running_responder(t, m))
    r = await coord.run(build_method("fedquest"), Question("born", "mmqa", "In what year was Ben Piazza born?",
                                                           ["1933"]), "born")
    assert sites["edge"].read_calls == [["mmqa/text/bp#0"]]
    assert "1933" in {f["tail"] for f in r.evidence}


async def test_a_coordinator_block_reads_the_whole_table_of_a_row_it_touches():
    seen = []
    coord, sites = coordinator()
    probe = coord.fed.probe

    async def recording(requests, **kwargs):
        seen.extend(requests)
        return await probe(requests, **kwargs)

    coord.fed.probe = recording
    r = await coord.run(build_method("fedquest"), Q, "tables")
    assert any(q.key == "Ben Piazza | Filmography" and q.k == 24 for q in seen)
    first = next(e for e in r.trace if e["event"] == "bind_index")
    assert first["entries"] == 2 and r.answer == "baseball"


async def test_a_late_bound_hop_looks_up_the_bound_values_own_article_by_title():
    fed, sites = running_example()
    article = FakeObject("mmqa/text/bnb#0", "text", "text", "The Bad News Bears",
                         "The Bad News Bears is a 1976 sports comedy film. (The Bad News Bears | directed by | "
                         "Michael Ritchie)", [], doc_id="bnbt")
    sites["edge"].objects[article.oid] = article
    pattern = ('("anchor_entity"[SEP]Ben Piazza)[REC]("entity"[SEP]film1)[REC]("entity"[SEP]person1)[REC]'
               '("answer_entity"[SEP]person1)[REC]("relation"[SEP]Ben Piazza[SEP]film1[SEP]acted in 1976 film)[REC]'
               '("relation"[SEP]film1[SEP]person1[SEP]directed by)[END]')
    coord, _ = coordinator(world=(fed, sites), responder=lambda t, m: pattern if t == "pattern" else running_responder(t, m))
    q = Question("dir", "mmqa", "Who directed the 1976 film Ben Piazza acted in?", ["Michael Ritchie"])
    seen = recording(coord)
    r = await coord.run(build_method("fedquest"), q, "article")
    assert any(p.kind == "title" and p.titles == ["The Bad News Bears"] and "text" in (p.collections or [])
               for p in seen)
    assert r.bindings.get("?person1") == "Michael Ritchie"


async def test_an_undecided_text_condition_is_read_on_the_known_values_own_picture():
    pattern = ('("anchor_entity"[SEP]Dannebrog)[REC]("entity"[SEP]shape1)[REC]("answer_entity"[SEP]shape1)'
               '[REC]("relation"[SEP]Dannebrog[SEP]shape1[SEP]contains)[END]')
    fed, sites = flag_world()
    flag = sites["aurora"].objects["mmqa/image/dk"]
    flag.title = flag.text = "Dannebrog"
    flag.triples = [("Dannebrog", "contains", "a white cross")]
    coord, _ = coordinator(world=(fed, sites), responder=lambda t, m: pattern if t == "pattern" else running_responder(t, m))
    q = Question("shape", "mmqa", "Which shape does the Dannebrog contain?", ["a white cross"])
    seen = recording(coord)
    await coord.run(build_method("fedquest"), q, "picture")
    assert any(p.kind == "title" and p.titles == ["Dannebrog"] and p.collections == ["image"] for p in seen)
    assert sites["aurora"].read_calls == [["mmqa/image/dk"]]


def test_a_filter_reads_flat_candidates_in_calls_theorem_1_sizes():
    from fedquest.methods.fedquest import FRAME
    from fedquest.planner.blocks import Cand, block_plan

    def first(m):
        cands = [Cand(i, 0.9 / m, prompts.IMAGE_TOKENS + FRAME) for i in range(m)]
        return len(block_plan(cands, 550.0, 1.0, prompts.PER_CALL["image"], ordered=True)[0])

    assert first(1) == 1 and all(1 <= first(m) <= prompts.PER_CALL["image"] for m in (2, 5, 10))
    assert first(10) == prompts.PER_CALL["image"]


async def test_a_bound_values_own_article_is_looked_up_by_its_exact_title_only():
    fed, sites = running_example()
    sequel = FakeObject("mmqa/text/seq#0", "text", "text", "The Bad News Bears in Breaking Training",
                        "The Bad News Bears in Breaking Training is a 1977 sequel. (The Bad News Bears | directed by | "
                        "Michael Pressman)", [], doc_id="seq")
    article = FakeObject("mmqa/text/bnb#0", "text", "text", "The Bad News Bears",
                         "The Bad News Bears is a 1976 sports comedy film. (The Bad News Bears | directed by | "
                         "Michael Ritchie)", [], doc_id="bnbt")
    sites["edge"].objects[article.oid] = article
    sites["midway3"].objects[sequel.oid] = sequel
    pattern = ('("anchor_entity"[SEP]Ben Piazza)[REC]("entity"[SEP]film1)[REC]("entity"[SEP]person1)[REC]'
               '("answer_entity"[SEP]person1)[REC]("relation"[SEP]Ben Piazza[SEP]film1[SEP]acted in 1976 film)[REC]'
               '("relation"[SEP]film1[SEP]person1[SEP]directed by)[END]')
    coord, _ = coordinator(world=(fed, sites), responder=lambda t, m: pattern if t == "pattern" else running_responder(t, m))
    r = await coord.run(build_method("fedquest"), Question("dir2", "mmqa", "Who directed the 1976 film Ben Piazza "
                                                                          "acted in?", ["Michael Ritchie"]), "exact")
    assert not any(sequel.oid in call for site in sites.values() for call in site.read_calls)
    assert r.bindings.get("?person1") == "Michael Ritchie"


@pytest.mark.parametrize("label", FEDQUESTS)
async def test_a_round_never_spends_past_the_token_budget_at_any_width(label):
    def wide_world():
        texts = [FakeObject(f"mmqa/text/{s}{i}", "text", "text", f"Zorg {s}{i}",
                            "Zorg Corp. " + "background " * 120 + f"(Zorg | founded in | {1900 + i})", [], doc_id=f"{s}{i}")
                 for s in ("a", "m") for i in range(3)]
        a = FakeSite("aurora", texts[:3], rules([("zorg", "text", [t.oid for t in texts[:3]])]))
        m = FakeSite("midway3", texts[3:], rules([("zorg", "text", [t.oid for t in texts[3:]])]))
        return fake_federation(a, m), {"aurora": a, "midway3": m}

    pattern = ('("anchor_entity"[SEP]Zorg)[REC]("entity"[SEP]year1)[REC]("answer_entity"[SEP]year1)[REC]'
               '("relation"[SEP]Zorg[SEP]year1[SEP]founded in)[END]')
    for width in (1, 2, 4):
        coord, _ = coordinator(world=wide_world(), responder=lambda t, m: pattern if t == "pattern" else
                               '{"year1": []}' if t == "read" else running_responder(t, m))
        r = await coord.run(build_method(label, {"tokens": 2000, "width": width}),
                            Question("z", "mmqa", "When was Zorg founded?", ["1900"]), f"w{width}")
        evidence = r.ledger["extraction_tokens"] + r.ledger.get("coordinator_tokens_bind_index", 0)
        assert evidence <= 2000, (width, evidence)


async def test_a_filter_reads_the_candidates_pictures_where_they_are_over_several_image_sites():
    rows = [FakeObject(f"mmqa/row/vv#{i}", "row", "table", "Vakkantham Vamsi — Films",
                       f"Vakkantham Vamsi | Films | Year: {y}; Title: {t}", [], doc_id="vv")
            for i, (y, t) in enumerate([(2014, "Race Gurram"), (2016, "Kick 2")])]
    posters = [FakeObject(f"mmqa/image/{i}", "image", "image", t, t, [(t, "poster features", d)], doc_id=f"p{i}")
               for i, (t, d) in enumerate([("Race Gurram", "motorcycle"), ("Kick 2", "a man in a suit")])]
    midway3 = FakeSite("midway3", rows, rules([("vakkantham", "row", [r.oid for r in rows])], {"row": []}))
    empty = FakeSite("aurora", [FakeObject("mmqa/image/x", "image", "image", "Unrelated", "Unrelated", [])],
                     rules([], {"image": []}))
    holder = FakeSite("edge2", posters, rules([], {"image": [p.oid for p in posters]}))
    for qid in ("f1", "f2", "f3", "f4"):
        coord, sites = coordinator(world=(fake_federation(midway3, empty, holder),
                                          {"midway3": midway3, "aurora": empty, "edge2": holder}),
                                   responder=lambda t, m: POSTER_PATTERN if t == "pattern" else running_responder(t, m))
        q = Question(qid, "mmqa", "Which film written by Vakkantham Vamsi has a poster featuring a motorcycle?", [])
        r = await coord.run(build_method("fedquest"), q, f"multi-{qid}")
        assert r.bindings.get("?film1") == "Race Gurram" and not empty.read_calls


@pytest.mark.parametrize("label", FEDQUESTS)
async def test_a_session_reuses_a_visual_edge_with_no_read(label):
    kg = MemoryKG()
    coord, sites = coordinator(kg=kg)
    first = await coord.run(build_method(label), Q, "stream")
    calls = len(sites["aurora"].read_calls)
    again = await coord.run(build_method(label), Q, "stream")
    assert first.answer == again.answer == "baseball"
    assert len(sites["aurora"].read_calls) == calls and again.reused


def discography(n=16, dog=13):
    rows = [FakeObject(f"mmqa/row/pm#{i}", "row", "table", "Pat Metheny — Discography",
                       f"Pat Metheny | Discography | Year: {1970 + i}; Album: Album {i}", [], doc_id="pm")
            for i in range(n)]
    covers = [FakeObject(f"mmqa/image/c{i}", "image", "image", f"Album {i}", f"Album {i}",
                         [(f"Album {i}", "cover featuring", "a jumping dog" if i == dog else "a field at dusk")],
                         doc_id=f"c{i}") for i in range(n)]
    midway3 = FakeSite("midway3", rows, rules([("pat metheny", "row", [r.oid for r in rows])], {"row": []}))
    aurora = FakeSite("aurora", covers, rules([], {"image": [c.oid for c in covers]}))
    return fake_federation(midway3, aurora), {"midway3": midway3, "aurora": aurora}


DOG_PATTERN = ('("anchor_entity"[SEP]Pat Metheny)[REC]("entity"[SEP]album1)[REC]("answer_entity"[SEP]album1)[REC]'
               '("relation"[SEP]Pat Metheny[SEP]album1[SEP]released album)[REC]'
               '("relation"[SEP]album1[SEP]album1[SEP]has cover featuring a jumping dog)[END]')


async def test_a_list_is_read_whole_and_its_pictures_checked_in_one_pass():
    from fedquest.llm.fake import message_text

    prompts_seen: list[str] = []

    def responder(t, m):
        if t == "read_index":
            prompts_seen.append(message_text(m))
        return DOG_PATTERN if t == "pattern" else running_responder(t, m)

    coord, sites = coordinator(world=discography(), responder=responder)
    seen = recording(coord)
    q = Question("pm16", "mmqa", "Which Pat Metheny album has a cover featuring a jumping dog?", ["Album 13"])
    r = await coord.run(build_method("fedquest", {"tokens": 20000, "objects": 64}), q, "pm16")
    (covers,) = [set(p.titles) for p in seen if p.kind == "title"]
    assert covers == {f"Album {i}" for i in range(16)} and r.bindings == {"?album1": "Album 13"}
    assert any(e["event"] == "unit" for e in r.trace) and not any(e["event"] == "reask" for e in r.trace)
    assert all(len(call) <= 4 for call in sites["aurora"].read_calls)
    reads = [e for e in r.trace if e["event"] == "bind_index"]
    assert sum(len(e["values"].get("album1", [])) for e in reads) == 16
    assert len(prompts_seen) == 2 and not any("Candidates for X" in p for p in prompts_seen)


async def test_a_witness_from_outside_the_rounds_pool_does_not_break_the_table_unit():
    from fedquest.methods.fedquest import FedQuest, Held, _Run, prune

    coord, _ = coordinator(world=discography(),
                           responder=lambda t, m: DOG_PATTERN if t == "pattern" else running_responder(t, m))
    q = Question("pm", "mmqa", "Which Pat Metheny album has a cover featuring a jumping dog?", [])
    run = _Run(FedQuest(), q, coord.context("unit"),
               prune(hint_modalities(prompts.parse_pattern(DOG_PATTERN, q.question)), q.question))
    await run.setup()
    kind, asks = run.step()
    run.asked |= {(a.cond.id, "pat metheny") for a in asks}
    pool, _, _ = run.pool(kind, asks, await run.paths(kind, asks))
    x, cid = asks[0].var, asks[0].cond.id
    row = next(it.c for it in pool.values() if it.c.modality == "table")
    elsewhere = ("edge", "mmqa/text/elsewhere#0")
    run.values[x]["album 0"] = Held("Album 0", {cid: {"pat metheny"}}, [row.ref])
    run.values[x]["other"] = Held("Other", {cid: {"pat metheny"}}, [elsewhere])
    run.p_read.update({row.ref: 0.4, elsewhere: 0.5})
    run.mark(asks, [row.ref])
    await run.complete_units(pool, asks)
    assert any(e["event"] == "unit" for e in run.ctx.trace) and run.p_read[elsewhere] == 0.5


def test_a_list_read_leaves_the_picture_condition_to_the_picture_check():
    from fedquest.llm.fake import message_text
    from fedquest.methods.fedquest import FedQuest, _Run, prune

    q = Question("pm", "mmqa", "Which Pat Metheny album has a cover featuring a jumping dog?", [])
    run = _Run(FedQuest(), q, None, prune(hint_modalities(prompts.parse_pattern(DOG_PATTERN, q.question)), q.question))
    kind, asks = run.step()
    run.asked |= {(a.cond.id, "pat metheny") for a in asks}
    (u,), _ = run.unknowns(asks)
    assert kind == "text" and u.cap == 24 and u.later == ["X has cover featuring a jumping dog"]
    text = message_text(prompts.read_messages(q.question, [prompts.ReadObject("table row", "Pat Metheny | x")], [u]))
    assert "Not in this step (X's own picture is checked later): X has cover featuring a jumping dog" in text
    assert '"album1": [{"entity": "<name>", "object": <source number>}]' in text
    two = prompts.ReadUnknown("x", ["(A | r | X)", "(B | s | X)"], ["A", "B"], 12)
    assert '"conditions": [<numbers>]' in message_text(prompts.read_messages("q", [], [two]))
    assert prompts.ReadUnknown.from_dict(u.to_dict()) == u
    (v,), _ = _Run(FedQuest(scan=False), q, None, run.p).unknowns(asks)
    assert v.later == [] and v.cap == 12


COMPARE_PATTERN = ('("anchor_entity"[SEP]The Bad News Bears)[REC]("anchor_entity"[SEP]No Exit)[REC]'
                   '("entity"[SEP]year1)[REC]("entity"[SEP]year2)[REC]'
                   '("relation"[SEP]The Bad News Bears[SEP]year1[SEP]released in)[REC]'
                   '("relation"[SEP]No Exit[SEP]year2[SEP]released in)[END]')


async def test_a_comparisons_first_branch_leaves_its_second_branch_a_read():
    filler = "The Bad News Bears is remembered by many fans and critics as a story of a team of misfits and a coach "
    texts = [FakeObject(f"mmqa/text/bnb{i}", "text", "text", f"The Bad News Bears reception {i}", filler * 18, [],
                        doc_id=f"bnb{i}") for i in range(10)]
    no_exit = FakeObject("mmqa/text/ne#0", "text", "text", "No Exit (1962 film)",
                         "No Exit is a 1962 drama film. (No Exit | released in | 1962)", [], doc_id="ne")
    aurora = FakeSite("aurora", texts, rules([("bad news bears", "text", [t.oid for t in texts])], {"text": []}))
    edge = FakeSite("edge", [no_exit], rules([("no exit released", "text", [no_exit.oid])], {"text": []}),
                      policy="facts")
    coord, _ = coordinator(world=(fake_federation(aurora, edge), {"aurora": aurora, "edge": edge}),
                           responder=lambda t, m: COMPARE_PATTERN if t == "pattern" else running_responder(t, m))
    q = Question("cmp2", "mmqa", "Which was released first, The Bad News Bears or No Exit?", ["No Exit"])
    r = await coord.run(build_method("fedquest", {"tokens": 4000}), q, "cmp2")
    assert r.bindings.get("?year2") == "1962" and r.ledger["evidence_tokens"] <= 4000


async def test_a_truncated_entry_is_read_once_before_its_object_and_each_object_at_most_once():
    coord, sites = coordinator(world=truncated_rows())
    r = await coord.run(build_method("fedquest_q_sim_unit"), Q, "twins")
    site_reads = [oid for s in sites.values() for call in s.read_calls for oid in call]
    assert len(site_reads) == len(set(site_reads))
    entries = [e for e in r.trace if e["event"] == "bind_index"]
    rows = [oid for s in ("midway3",) for call in sites[s].read_calls for oid in call]
    assert not rows or entries


@pytest.mark.parametrize("label", FEDQUESTS)
async def test_a_wide_round_keeps_the_objects_cap(label):
    posters = [FakeObject(f"mmqa/image/{i}", "image", "image", f"Poster {i}", f"Poster {i}", [], doc_id=f"p{i}")
               for i in range(6)]
    one = FakeSite("aurora", posters[:3], rules([("poster", "image", [p.oid for p in posters[:3]])]))
    two = FakeSite("aurora2", posters[3:], rules([("poster", "image", [p.oid for p in posters[3:]])]))
    pattern = ('("anchor_entity"[SEP]Poster)[REC]("entity"[SEP]thing1)[REC]("answer_entity"[SEP]thing1)[REC]'
               '("relation"[SEP]Poster[SEP]thing1[SEP]poster shows)[END]')
    for objects in (1, 2, 4):
        coord, _ = coordinator(world=(fake_federation(one, two), {"aurora": one, "aurora2": two}),
                               responder=lambda t, m: pattern if t == "pattern" else '{"thing1": []}' if t == "read"
                               else running_responder(t, m))
        r = await coord.run(build_method(label, {"width": 2, "objects": objects}),
                            Question("w", "mmqa", "What does the poster show?", []), f"w{objects}")
        assert r.ledger["objects_read"] <= objects
