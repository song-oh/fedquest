# FedQuest's typed graph, semijoin probes, v4 pricing and the plan 2 fallback
import math

from fedquest.llm.fake import fake_read, message_text
from fedquest.methods import compiler
from fedquest.methods.base import build_method
from fedquest.methods.fedquest import LIST_CAP, READ_CAP, FedQuest, _Run, conditions, prune
from fedquest.planner.calibration import expected_rank, joinable_rank
from fedquest.types import Question
from tests.fakes import FakeObject, FakeSite, fake_federation, rules, running_responder
from tests.test_methods import TYPICAL_CAL, Q, coordinator
from tests.test_typed_query import RUNNING_TYPED, typed


def tasks(coord) -> list[str]:
    return [task for _, task in coord.llm.backend.calls]


def typed_world(**replies: str):
    replies = {"typed_graph": RUNNING_TYPED, **replies}

    def responder(task, messages):
        return replies[task] if task in replies else running_responder(task, messages)

    return coordinator(responder=responder)


COMPARISON = typed(["Big", "Splash"], ["?person1", "?date1", "?person2", "?date2"],
                   [("Big", "directed by", "?person1"), ("?person1", "date of birth", "?date1"),
                    ("Splash", "directed by", "?person2"), ("?person2", "date of birth", "?date2")], form="choice")


def test_typed_prune_keeps_the_other_side_of_a_comparison_and_a_placeholder_another_relation_names():
    q = "Which film has the director born later, Big or Splash?"
    assert compiler.translate(COMPARISON, q)[0].raw["answer_inferred"]
    assert prune(compiler.translate(COMPARISON, q)[0], q).variables == ["?person1", "?person2", "?date2"]
    p = prune(compiler.translate(COMPARISON, q)[0], q, typed=True)
    assert p.variables == ["?person1", "?date1", "?person2", "?date2"] and len(p.edges) == 4
    q = "When did the director of Big marry his spouse?"
    reply = typed(["Big"], ["?person1", "?person2", "?date1"],
                  [("Big", "directed by", "?person1"), ("?person1", "spouse", "?person2"),
                   ("?person1", "married ?person2 in", "?date1")], answer="?date1")
    assert "?person2" not in prune(compiler.translate(reply, q)[0], q).variables
    p = prune(compiler.translate(reply, q)[0], q, typed=True)
    assert p.variables == ["?person1", "?person2", "?date1"]
    assert [c.deps for c in conditions(p, q) if c.var == "?date1"] == [("?person2",)]
    reply = typed(["Big"], ["?person1", "?person2"], [("Big", "directed by", "?person1"),
                                                      ("?person1", "is a", "?person2")], answer="?person1")
    assert prune(compiler.translate(reply, q)[0], q, typed=True).variables == ["?person1"]


def test_a_set_variable_of_the_typed_graph_reads_as_a_list():
    q = "Which locations did Masahiko Harada win at in the 1997/98 season?"
    reply = typed(["Masahiko Harada"], [("?location1", True)],
                  [("Masahiko Harada", "won at in the 1997/98 season", "?location1")], answer="?location1")
    p = prune(compiler.translate(reply, q)[0], q, typed=True)
    assert p.raw["sets"] == ["?location1"]
    question = Question("set", "mmqa", q, ["Planica"])
    assert _Run(FedQuest(graph="typed"), question, None, p).cap("?location1") == LIST_CAP
    assert _Run(FedQuest(), question, None, p).cap("?location1") == READ_CAP


async def test_the_router_adds_its_picks_to_the_typed_graphs_only_when_asked():
    coord, _ = typed_world()
    plain = await coord.run(build_method("fedquest", {"graph": "typed"}), Q, "typed")
    coord2, _ = typed_world()
    routed = await coord2.run(build_method("fedquest", {"graph": "typed", "router": True}), Q, "routed")
    assert plain.error is None and routed.error is None and plain.answer == routed.answer == "baseball"
    assert "visual" not in tasks(coord) and tasks(coord2).count("visual") == 1
    assert [e["event"] for e in routed.trace].count("visual") == 1
    assert routed.ledger["coordinator_calls"] == plain.ledger["coordinator_calls"] + 1
    for bad in (1, "yes", None):
        try:
            FedQuest(router=bad)
        except ValueError:
            continue
        raise AssertionError(bad)


async def test_typed_repair_rewrites_in_the_typed_language_and_mgrag_repair_stays_mgrags():
    failing = typed(["Kevin Bacon"], ["?film1", "?sport1"], [("Kevin Bacon", "acted in 1976 film", "?film1"),
                                                             ("?film1", "poster depicts sport", "?sport1", "image")],
                    answer="?sport1")
    coord, _ = typed_world(typed_graph=failing, typed_repair=RUNNING_TYPED)
    r = await coord.run(build_method("fedquest", {"graph": "typed", "repair": True}), Q, "repair")
    assert r.error is None, r.error
    assert "typed_repair" in tasks(coord) and "repair" not in tasks(coord) and "visual" not in tasks(coord)
    event = next(e for e in r.trace if e["event"] == "repair")
    assert event["ran"] and event["edges"] == ["Ben Piazza | acted in 1976 film | ?film1",
                                               "?film1 | poster depicts sport | ?sport1"]
    assert r.answer == "baseball" and r.ledger["repair_calls"] == 1
    assert r.pattern["anchors"] == ["Kevin Bacon"]
    coord, _ = typed_world(typed_graph=failing, typed_repair=RUNNING_TYPED)
    r = await coord.run(build_method("fedquest", {"graph": "typed", "repair": True, "router": True}), Q, "routed")
    assert tasks(coord).count("visual") == 2 and r.answer == "baseball"
    coord, _ = coordinator()
    r = await coord.run(build_method("fedquest", {"repair": True}), Q, "mgrag")
    assert "typed_repair" not in tasks(coord) and "typed_graph" not in tasks(coord)


def rows_world():
    no_exit = FakeObject("mmqa/row/t1#2", "row", "table", "Ben Piazza — Filmography",
                         "Ben Piazza | Filmography | Year: 1962; Title: No Exit; Role: Camarero", [], doc_id="t1")
    bears = FakeObject("mmqa/row/t2#1", "row", "table", "Ben Piazza — Film roles",
                       "Ben Piazza | Film roles | Year: 1976; Title: The Bad News Bears; Role: Bob Whitewood", [],
                       doc_id="t2")
    season = FakeObject("mmqa/row/t9#1", "row", "table", "Bears — Seasons", "Bears | Seasons | Year: 1976; Wins: 7",
                        [], doc_id="t9")
    season2 = FakeObject("mmqa/row/t9#2", "row", "table", "Bears — Seasons", "Bears | Seasons | Year: 1977; Wins: 9",
                         [], doc_id="t9")
    poster = FakeObject("mmqa/image/p1", "image", "image", "The Bad News Bears", "The Bad News Bears",
                        [("The Bad News Bears", "poster depicts", "baseball")], doc_id="p1")
    aurora = FakeSite("aurora", [poster], rules([("bad news bears", "image", [poster.oid])], {"image": [poster.oid]}))
    midway3 = FakeSite("midway3", [no_exit, bears, season],
                       rules([("acted in", "row", [season.oid, no_exit.oid, bears.oid])],
                             {"row": [season.oid, no_exit.oid, bears.oid]}))
    east = FakeSite("east", [season2], rules([], {"row": [season2.oid]}))
    sites = {"aurora": aurora, "midway3": midway3, "east": east}
    return fake_federation(aurora, midway3, east), sites


async def run_rows(params, calib=None):
    coord, sites = coordinator(world=rows_world(), calib=calib)
    r = await coord.run(build_method("fedquest", params), Q, "rows")
    assert r.error is None, r.error
    return r, sites


async def test_a_semijoin_sends_value_keys_on_their_value_and_pages_an_exhausted_search():
    r, sites = await run_rows({"k": 1, "probe": "semijoin"})
    assert r.answer == "baseball" and r.matched
    sent = sites["midway3"].probe_requests
    joins = [x for x in sent if x.semijoin]
    assert joins and all(x.kind == "bound" and x.mentions == ["Ben Piazza"] for x in joins)
    page = [x for x in joins if x.offset]
    assert [(x.offset, x.collections) for x in page] == [(1, ["row"])]
    assert not any(x.offset for x in sites["east"].probe_requests)
    assert r.ledger["join_pages"] == 1 and r.ledger["join_paged_candidates"] == 1 and r.ledger["joinable"] >= 2
    assert [e["requests"] for e in r.trace if e["event"] == "join_page"] == [1]
    for params in ({"k": 1, "probe": "semijoin", "pages": 0}, {"k": 1}):
        r, sites = await run_rows(params)
        assert r.answer != "baseball" and "join_pages" not in r.ledger
    assert not any(x.semijoin for x in sites["midway3"].probe_requests)


async def test_vocab_routing_skips_a_site_whose_words_cannot_name_the_value():
    r, sites = await run_rows({"k": 1, "probe": "semijoin", "route": "vocab"})
    assert r.answer == "baseball"
    assert not any(x.semijoin for x in sites["east"].probe_requests) and r.ledger["skipped_sites"] >= 1
    assert any(x.semijoin for x in sites["midway3"].probe_requests)
    r, sites = await run_rows({"k": 1, "probe": "semijoin"})
    assert any(x.semijoin for x in sites["east"].probe_requests) and not r.ledger.get("skipped_sites")


async def test_a_tables_sibling_probe_goes_to_the_witnessing_rows_site_only_under_semijoin():
    for probe, east in (("topk", True), ("semijoin", False)):
        r, sites = await run_rows({"probe": probe})
        siblings = [x for s in sites.values() for x in s.probe_requests if x.key == "Ben Piazza | Filmography"]
        assert siblings, probe
        assert any(x.key == "Ben Piazza | Filmography" for x in sites["east"].probe_requests) == east, probe


class V4Spy:
    version = 4

    def __init__(self) -> None:
        self.asked: list[tuple[str, float, bool | None]] = []

    def __getattr__(self, name):
        return getattr(TYPICAL_CAL, name)

    def alpha(self, site, collection, kind, rank, signal, modality="", round="first", mention=None):
        self.asked.append((kind, rank, mention))
        return TYPICAL_CAL.alpha(site, collection, kind, math.ceil(rank), signal, modality, round)


async def test_a_v4_calibration_prices_by_the_group_rank_and_the_mention_bit(monkeypatch):
    seen = []
    pool = _Run.pool

    def spy(self, kind, asks, found):
        out = pool(self, kind, asks, found)
        seen.append((self, found, out))
        return out

    monkeypatch.setattr(_Run, "pool", spy)
    cal = V4Spy()
    r, _ = await run_rows({"probe": "semijoin"}, calib=cal)
    assert r.answer == "baseball" and cal.asked
    run, found, (items, outside, coverage) = seen[0]
    sizes = {("midway3", "row"): ("fake", 3), ("east", "row"): ("fake", 1), ("aurora", "image"): ("fake", 1)}
    assert run.sizes == sizes and run.groups[("row", "fake")] == 4
    kinds = {f.kind for f in found}
    assert {"bound", "question"} <= kinds
    priced = iter(cal.asked)
    for f in found:
        kind, g, mention = next(priced)
        assert kind == f.kind and mention == f.c.mentions
        if f.kind == "bound":
            assert f.c.jrank is not None and g == joinable_rank(f.c.jrank, 2, 2)
        elif f.kind == "question":
            encoder, m = sizes[(f.c.site, f.c.collection)]
            assert g == expected_rank(f.c.rank, m, run.groups[(f.c.collection, encoder)])
        else:
            assert g == f.c.rank
    for cid in {k[0] for k in items}:
        its = [it for k, it in items.items() if k[0] == cid]
        able = [it for it in its if it.mentions is not False]
        z = max(1.0, sum(it.alpha for it in able) / coverage)
        assert all(abs(it.p - it.alpha / z) < 1e-12 for it in its)
        assert abs(outside[cid] - max(0.0, 1 - sum(it.p for it in able))) < 1e-12
    assert any(it.mentions is False for it in items.values())


def test_salted_ties_break_equal_places_and_candidates_by_a_question_hash_and_none_keeps_name_order():
    from fedquest.planner.blocks import Cand, Place, by_ratio, calls, plan_call

    def places(n=4):
        return [Place(f"site{i}|image", 10.0, 4, [Cand(("o", (f"site{i}", "p")), 0.2, 290.0, float(i))])
                for i in range(n)]

    assert [c.place for c in calls(places())][0] == "site0|image" == plan_call(places()).place
    firsts = {plan_call(places(), salt=f"q{i}").place for i in range(40)}
    assert len(firsts) == 4
    for i in range(10):
        assert plan_call(places(), salt=f"q{i}").place == calls(places(), salt=f"q{i}")[0].place
        assert plan_call(places(), salt=f"q{i}") == plan_call(places(), salt=f"q{i}")
    same = [Cand(("o", ("b", "x")), 0.1, 10.0, 2.0), Cand(("o", ("a", "y")), 0.1, 10.0, 1.0),
            Cand(("o", ("c", "z")), 0.3, 10.0, 3.0)]
    assert [x.key[1][0] for x in by_ratio(same)] == ["c", "b", "a"]
    assert [x.key[1][0] for x in by_ratio(same, salt="q1")] == ["c", "a", "b"]
    q = Question("qid-7", "mmqa", "?", ["a"])
    p = prune(compiler.translate(typed(["A"], ["?x"], [("A", "r", "?x")]), q.question)[0], q.question)
    assert _Run(FedQuest(ties="salted"), q, None, p).salt == "qid-7" and _Run(FedQuest(), q, None, p).salt is None


async def test_fedquest_runs_with_salted_ties():
    coord, _ = coordinator()
    r = await coord.run(build_method("fedquest", {"ties": "salted"}), Q, "salted")
    assert r.error is None and r.answer == "baseball"


FAILING = typed(["Kevin Bacon"], ["?film1", "?sport1"], [("Kevin Bacon", "acted in 1976 film", "?film1"),
                                                         ("?film1", "poster depicts sport", "?sport1", "image")],
                answer="?sport1")


def answering_world(world=None, answers: bool = True, calib=None, **replies: str):
    kept: list[str] = []
    replies = {"typed_graph": RUNNING_TYPED, **replies}

    def responder(task, messages):
        if task == "answer":
            kept.append(message_text(messages))
        return replies[task] if task in replies else running_responder(task, messages)

    coord, sites = coordinator(responder=responder, world=world, calib=calib)
    if answers:
        sites["edge"].objects["mmqa/text/b1#0"].text += " The sport on his 1976 film's poster is baseball."
        sites["edge"].reader = lambda text: ({"answer": [{"entity": "baseball", "object": 1}]}
                                               if "X answers the question" in text else fake_read(text))
    return coord, sites, kept


async def test_plan_2_reads_the_question_probes_candidates_when_run_1_fails():
    results = {}
    for direct in ("off", "fail"):
        coord, _, _ = answering_world(typed_graph=FAILING)
        results[direct] = await coord.run(build_method("fedquest", {"graph": "typed", "direct": direct}), Q, "d")
    off, r = results["off"], results["fail"]
    assert off.error is None and r.error is None, r.error
    assert off.answer != "baseball" and r.answer == "baseball"
    event = next(e for e in r.trace if e["event"] == "direct")
    assert event["trigger"] == "fail" and event["matched"] and event["candidates"] > 0 and event["rounds"] == 1
    assert r.ledger["direct_ran"] == r.ledger["direct_matched"] == 1
    assert r.ledger["probes"] == off.ledger["probes"]
    assert [e.get("run") for e in r.trace if e["event"] == "stop"] == [1, "direct"]
    assert r.pattern["anchors"] == ["Kevin Bacon"]
    assert any(f["relation"] == "answer" and f["tail"] == "baseball" for f in r.evidence)


async def test_plan_2_reads_one_call_per_place_and_six_objects_per_site(monkeypatch):
    from fedquest.methods.fedquest import COORD, DIRECT_OBJECTS, _Direct

    made = []
    plan = _Direct.plan

    def spy(self, *args):
        chosen = plan(self, *args)
        made.extend(chosen)
        return chosen

    monkeypatch.setattr(_Direct, "plan", spy)
    coord, sites, _ = answering_world(answers=False, typed_graph=FAILING)
    r = await coord.run(build_method("fedquest", {"graph": "typed", "direct": "always"}), Q, "d")
    assert r.error is None and made
    places = [c.place for c in made]
    assert len(places) == len(set(places))
    assert all(len(c.block) <= DIRECT_OBJECTS for c in made if c.place != COORD)
    assert not any(e["event"] == "join_page" for e in r.trace)


async def test_repair_goes_on_from_plan_2_under_the_same_floor_and_the_answer_reads_every_run(monkeypatch):
    runs = []
    go = _Run.go

    async def spy(self):
        runs.append(self)
        await go(self)

    monkeypatch.setattr(_Run, "go", spy)
    coord, sites, _ = answering_world(answers=False, typed_graph=FAILING, typed_repair=RUNNING_TYPED)
    r = await coord.run(build_method("fedquest", {"graph": "typed", "direct": "fail", "repair": True}), Q, "r")
    assert r.error is None and r.answer == "baseball"
    first, direct, again = runs
    assert type(direct).__name__ == "_Direct" and direct.prior is first and again.prior is direct
    assert again.tokens == direct.tokens == max(8000, math.ceil(first.evidence()) + 1500)
    assert again.chain() == [again, direct, first]
    assert [e.get("run") for e in r.trace if e["event"] == "stop"] == [1, "direct", 2]
    assert next(e for e in r.trace if e["event"] == "repair")["ran"]
    runs.clear()
    coord, sites, _ = answering_world()
    r = await coord.run(build_method("fedquest", {"graph": "typed", "direct": "always", "repair": True}), Q, "a")
    assert r.answer == "baseball" and len(runs) == 2 and not any(e["event"] == "repair" for e in r.trace)
    relations = {f["relation"] for f in r.evidence}
    assert "answer" in relations and relations - {"answer"}


async def test_direct_fail_after_a_passing_run_1_and_the_union_context():
    coord, _, _ = answering_world()
    plain = await coord.run(build_method("fedquest", {"graph": "typed", "repair": True}), Q, "p")
    coord, _, _ = answering_world()
    fail = await coord.run(build_method("fedquest", {"graph": "typed", "repair": True, "direct": "fail"}), Q, "p")
    assert plain.answer == fail.answer == "baseball" and fail.trace == plain.trace
    contexts = {}
    for context in ("chain", "union"):
        coord, _, kept = answering_world(rows_world(), answers=False)
        r = await coord.run(build_method("fedquest", {"context": context}), Q, "c")
        assert r.answer == "baseball" and r.evidence
        contexts[context] = kept[-1]
    assert "Wins: 9" not in contexts["chain"] and "Wins: 9" in contexts["union"]
    union = contexts["union"]
    assert union.index("Ben Piazza | Film roles") < union.index("Wins: 9")


async def test_mgrag_on_the_typed_graph_evaluates_fedquests_graph_with_mgrags_matcher():
    from fedquest.methods.mgrag import MGRAG

    coord, _ = typed_world()
    r = await coord.run(build_method("mgrag", {"graph": "typed"}), Q, "typed")
    assert r.error is None and r.answer == "baseball", r.error
    assert "typed_graph" in tasks(coord) and "pattern" not in tasks(coord) and "visual" not in tasks(coord)
    assert r.pattern["anchors"] == ["Ben Piazza"]
    coord, _ = typed_world()
    plain = await coord.run(build_method("mgrag"), Q, "mgrag")
    assert plain.answer == "baseball" and "typed_graph" not in tasks(coord) and "pattern" in tasks(coord)
    assert MGRAG.added_params == {"graph": "mgrag"} and MGRAG().graph == "mgrag"
    try:
        MGRAG(graph="kg")
    except ValueError:
        return
    raise AssertionError("graph='kg' accepted")


async def test_ircot_on_entries_reads_released_summaries_and_never_fetches_an_object():
    from fedquest.methods.iterrag import IterRAG

    runs = {}
    for read in ("raw", "entries"):
        coord, _, kept = answering_world(answers=False)
        r = await coord.run(build_method("iter_rag", {"read": read}), Q, read)
        assert r.error is None, r.error
        runs[read] = (r, kept)
    raw, entries = runs["raw"][0], runs["entries"][0]
    assert raw.ledger["bytes_raw"] > 0 and entries.ledger["bytes_raw"] == 0
    assert entries.consulted == raw.consulted
    answer_prompt = runs["entries"][1][-1]
    assert "Ben Piazza — Filmography\nBen Piazza | Filmography" in answer_prompt
    assert "Ben Piazza — Filmography: Ben Piazza | Filmography" not in answer_prompt
    assert IterRAG.added_params == {"read": "raw"} and IterRAG().read == "raw"
    try:
        IterRAG(read="objects")
    except ValueError:
        return
    raise AssertionError("read='objects' accepted")


async def test_plan_2_always_keeps_run_1s_match_and_elastic_pages_refuse_the_next_lists():
    coord, _, _ = answering_world(answers=False)
    r = await coord.run(build_method("fedquest", {"graph": "typed", "direct": "always"}), Q, "always")
    assert r.answer == "baseball" and r.matched
    assert not next(e for e in r.trace if e["event"] == "direct")["matched"]
    assert next(e for e in r.trace if e["event"] == "stop" and e.get("run") == "direct")["matched"]
    for bad in ({"probe": "semijoin"}, {"limits": "share"}):
        try:
            FedQuest(retrieve="elastic", **bad)
        except ValueError:
            continue
        raise AssertionError(bad)
