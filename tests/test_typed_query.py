from __future__ import annotations

import json

from fedquest.config import load_models
from fedquest.ledger import Ledger
from fedquest.llm import prompts
from fedquest.llm.auth import TokenProvider
from fedquest.llm.client import LLMClient
from fedquest.llm.fake import FakeBackend
from fedquest.llm.parsing import COMPLETION_DELIM, RECORD_DELIM, TUPLE_DELIM
from fedquest.methods import compiler
from fedquest.methods.base import QueryContext, build_method
from fedquest.methods.common import fold_media, hint_modalities
from fedquest.methods.fedquest import conditions, prune
from fedquest.types import Pattern, Question
from tests.fakes import RUNNING_QUESTION, running_responder
from tests.test_methods import Q, coordinator


def records(*recs: tuple[str, ...]) -> str:
    return RECORD_DELIM.join("(" + TUPLE_DELIM.join([f'"{r[0]}"', *r[1:]]) + ")" for r in recs) + COMPLETION_DELIM


def mgrag(text: str, question: str) -> Pattern:
    return hint_modalities(fold_media(prompts.parse_pattern(text, question)))


def typed(constants=(), variables=(), atoms=(), filters=(), answer=None, form="entity") -> str:
    return json.dumps({"constants": list(constants),
                       "variables": [{"name": v, "set": False} if isinstance(v, str) else {"name": v[0], "set": v[1]}
                                     for v in variables],
                       "atoms": [{"subject": s, "relation": r, "object": o, "evidence": ev[0] if ev else "text"}
                                 for s, r, o, *ev in atoms],
                       "filters": [{"variable": v, "condition": c, "evidence": ev[0] if ev else "text"}
                                   for v, c, *ev in filters],
                       "answer": answer, "form": form})


def sig(p: Pattern) -> tuple:
    edges = tuple((e.id, e.head, e.relation, e.tail, e.modality) for e in p.edges)
    return (tuple(p.anchors), tuple(p.variables), edges,
            p.answer, tuple((k, tuple(v)) for k, v in p.conditions.items()), bool(p.raw.get("answer_inferred")))


class Recording(FakeBackend):
    def __init__(self, replies: dict[str, str]) -> None:
        super().__init__(lambda task, messages: replies.get(task, "{}"))
        self.seen: list[dict] = []

    async def complete(self, model, messages, *, max_tokens, temperature, json_mode, task):
        self.seen.append({"model": model, "task": task, "json_mode": json_mode, "max_tokens": max_tokens,
                          "messages": messages})
        return await super().complete(model, messages, max_tokens=max_tokens, temperature=temperature,
                                      json_mode=json_mode, task=task)


def context(replies: dict[str, str]) -> tuple[QueryContext, Recording]:
    backend = Recording(replies)
    cfg = load_models().model_copy(update={"provider": "fake"})
    llm = LLMClient(cfg, cache=None, tokens=TokenProvider("fake"), backend=backend)
    return QueryContext(fed=None, llm=llm, kg=None, matcher=None, calib=None, ledger=Ledger(), ns="b"), backend


RUNNING_TYPED = typed(["Ben Piazza"], ["?film1", "?sport1"],
                      [("Ben Piazza", "acted in 1976 film", "?film1"),
                       ("?film1", "poster depicts sport", "?sport1", "image")],
                      answer="?sport1")


def test_a_typed_graph_translates_to_the_pattern_and_records_its_raw_fields():
    q = "Which locations featuring mountains did Masahiko Harada win at in the 1997/98 season?"
    reply = typed(["Masahiko Harada"], [("?location1", True)],
                  [("Masahiko Harada", "won at in the 1997/98 season", "?location1")],
                  [("?location1", "features mountains", "image")], answer="?location1")
    p, pictured = compiler.translate(reply, q)
    assert (p.anchors, p.variables, p.answer) == (["Masahiko Harada"], ["?location1"], "?location1")
    assert [(e.head, e.relation, e.tail, e.modality) for e in p.edges] == [
        ("Masahiko Harada", "won at in the 1997/98 season", "?location1", None)]
    assert p.conditions == {"?location1": ["features mountains"]}
    assert pictured == {("?location1", "features mountains")}
    assert p.raw["typed"] == json.loads(reply) and p.raw["sets"] == ["?location1"] and p.raw["form"] == "entity"
    assert p.raw["pictured"] == [("?location1", "features mountains")] and p.raw["unbindable"] == []
    assert "fallback" not in p.raw and "answer_inferred" not in p.raw
    p, _ = compiler.translate(typed(["A", "B"], ["?date1", "?date2"], [("A", "born on", "?date1"),
                                                                       ("B", "born on", "?date2")], form="when"),
                              "Who was born first, A or B?")
    assert p.raw["form"] is None and p.answer == "?date2" and p.raw["answer_inferred"] is True


def test_the_translation_is_parse_patterns_on_graphs_both_languages_can_say():
    running = compiler.translate(RUNNING_TYPED.replace('"image"', '"text"'), RUNNING_QUESTION)[0]
    from tests.fakes import RUNNING_PATTERN

    assert sig(running) == sig(mgrag(RUNNING_PATTERN, RUNNING_QUESTION))
    cases = [
        ("Where was the performer of Count 'Em 88 born?",
         records(("anchor_entity", "Count 'Em 88"), ("entity", "person1"), ("entity", "location1"),
                 ("answer_entity", "location1"), ("relation", "Count 'Em 88", "person1", "performed by"),
                 ("relation", "person1", "location1", "born in")),
         typed(["Count 'Em 88"], ["?person1", "?location1"], [("count 'em 88", "performed by", "?Person1"),
                                                               ("?person1", "born in", "location1")],
               answer="?location1")),
        ("Which driver with a beard drove for Jordan-Ford in the 2004 Hungarian Grand Prix?",
         records(("anchor_entity", "Jordan-Ford"), ("anchor_entity", "2004 Hungarian Grand Prix"),
                 ("entity", "person1"),
                 ("answer_entity", "person1"), ("relation", "person1", "Jordan-Ford", "drove for"),
                 ("relation", "person1", "2004 Hungarian Grand Prix", "raced in"),
                 ("relation", "person1", "person1", "has a beard")),
         typed(["Jordan-Ford", "2004 Hungarian Grand Prix"], ["?person1"],
               [("?person1", "drove for", "Jordan-Ford"), ("?person1", "raced in", "2004 Hungarian Grand Prix"),
                ("?person1", "drove for", "Jordan-Ford")], [("?person1", "has a beard")], answer="?person1")),
        ("Who was born first, Ronnie Radke or Dave Brockie?",
         records(("anchor_entity", "Ronnie Radke"), ("anchor_entity", "Dave Brockie"), ("entity", "date1"),
                 ("entity", "date2"), ("relation", "Ronnie Radke", "date2", "born on"),
                 ("relation", "Dave Brockie", "date1", "born on")),
         typed(["Ronnie Radke", "Dave Brockie"], ["?date1", "?date2"], [("Ronnie Radke", "born on", "?date2"),
                                                                       ("Dave Brockie", "born on", "?date1")],
               form="choice")),
        ("Which album with a pyramid on its cover won the Grammy Award in its category?",
         records(("entity", "album1"), ("entity", "Grammy Award"), ("entity", "category1"), ("entity", "pyramid"),
                 ("answer_entity", "album1"), ("relation", "album1", "Grammy Award", "won"),
                 ("relation", "Grammy Award", "category1", "awarded in category"),
                 ("relation", "album1", "pyramid", "has on its cover")),
         typed([], ["?album1", "?Grammy Award", "?category1", "?pyramid"],
               [("?album1", "won", "?Grammy Award"), ("?Grammy Award", "awarded in category", "?category1"),
                ("?album1", "has on its cover", "?pyramid")], answer="?album1")),
        ("What sport is on the poster of the film Ben Piazza acted in?",
         records(("anchor_entity", "Ben Piazza"), ("entity", "film1"), ("entity", "poster1"), ("entity", "sport1"),
                 ("answer_entity", "sport1"), ("relation", "Ben Piazza", "film1", "acted in"),
                 ("relation", "film1", "poster1", "has poster"), ("relation", "poster1", "sport1", "depicts"),
                 ("relation", "poster1", "poster1", "shows a ball")),
         typed(["Ben Piazza"], ["?film1", "?poster1", "?sport1"],
               [("Ben Piazza", "acted in", "?film1"), ("?film1", "has poster", "?poster1"),
                ("?poster1", "depicts", "?sport1")], [("?poster1", "shows a ball")], answer="?sport1")),
        ("Which film whose poster shows a dog did Ben Piazza act in?",
         records(("anchor_entity", "Ben Piazza"), ("entity", "film1"), ("entity", "person1"),
                 ("answer_entity", "film2"), ("relation", "Ben Piazza", "person1", "known as"),
                 ("relation", "film2", "film2", "poster shows a dog")),
         typed(["Ben Piazza"], ["?film1", "?person1", "?film2"], [("Ben Piazza", "known as", "?person1")],
               [("?film2", "poster shows a dog")], answer="?film2")),
    ]
    for q, mgrag_reply, typed_reply in cases:
        p, _ = compiler.translate(typed_reply, q)
        assert sig(p) == sig(mgrag(mgrag_reply, q)), q
        assert "fallback" not in p.raw, q
    folded, _ = compiler.translate(cases[3][2], cases[3][0])
    assert folded.anchors == ["pyramid"] and [e.relation for e in folded.edges] == [
        "won Grammy Award awarded in category", "has on its cover"]
    media, _ = compiler.translate(cases[4][2], cases[4][0])
    assert [(e.head, e.relation, e.tail, e.modality) for e in media.edges] == [
        ("Ben Piazza", "acted in", "?film1", None), ("?film1", "poster depicts", "?sport1", "image")]
    assert media.conditions == {"?film1": ["poster shows a ball"]}


def test_image_evidence_marks_what_the_router_would_and_survives_both_folds():
    q = "What did the player who is left-handed win?"
    graph = typed(["Wimbledon"], ["?person1", "?prize1"],
                  [("?person1", "played at", "Wimbledon"), ("?person1", "holds above his head", "?prize1", "image"),
                   ("?person1", "won", "?prize1")],
                  [("?person1", "has a red cap on", "image"), ("?person1", "is left-handed")], answer="?prize1")
    p, pictured = compiler.translate(graph, q)
    assert [e.modality for e in p.edges] == [None, "image", None]
    assert pictured == {("?person1", "has a red cap on")}
    routed, _ = compiler.translate(graph.replace('"image"', '"text"'), q)
    routed.edges[1].modality = "image"
    assert conditions(p, q, pictured) == conditions(routed, q, frozenset({("?person1", "has a red cap on")}))
    assert [c.visual for c in conditions(p, q, pictured) if c.edge is None] == [True, False]
    assert not any(c.visual for c in conditions(routed, q, frozenset()) if c.edge is None)
    q = "Which album won the Grammy Award shown in its category logo?"
    p, _ = compiler.translate(typed([], ["?album1", "?Grammy Award", "?category1"],
                                    [("?album1", "won", "?Grammy Award"),
                                     ("?Grammy Award", "in category", "?category1", "image")], answer="?category1"), q)
    assert [(e.relation, e.modality) for e in p.edges] == [("won Grammy Award in category", "image")]
    q = "What sport is on the poster of the film Ben Piazza acted in?"
    p, pictured = compiler.translate(typed(["Ben Piazza"], ["?film1", "?poster1", "?sport1"],
                                           [("Ben Piazza", "acted in", "?film1"), ("?film1", "has poster", "?poster1"),
                                            ("?poster1", "depicts", "?sport1")], [("?poster1", "has a ball", "image")],
                                           answer="?sport1"), q)
    assert pictured == {("?film1", "poster has a ball")} and p.raw["pictured"] == [("?film1", "poster has a ball")]


def test_an_unusable_reply_is_exactly_the_fallback_graph():
    q = "Is it red?"
    base = prompts.parse_pattern("", q)
    for reply, why in (("not json at all", "json"), ('{"constants": []}', "json"), ('["a", "b"]', "json"),
                       ('{"atoms": [["a", "b", "c"]]}', "json"), ('{"atoms": [{"relation": "r", "object": " "}]}', "json"),
                       ('{"atoms": [{"subject": "?x", "relation": "r", "object": ["?x"]}]}', "json"),
                       (typed(["Paris"], ["?x"], [("Paris", "capital of", "Lyon")], answer="?x"), "terms"),
                       (typed(["Paris"], ["?x"], [("Paris", "capital of", "?x")], answer="Lyon"), "terms"),
                       (typed(["Paris"], ["?x"], [("Paris", "capital of", "?x")], [("Lyon", "is red")]), "terms")):
        p, pictured = compiler.translate(reply, q)
        assert sig(p) == sig(base) and pictured == frozenset(), reply
        assert p.raw["fallback"] == why and p.raw["unbindable"] == [] and p.raw["sets"] == [], reply
        assert ("reply" in p.raw) == (why == "json"), reply
    q = "What is Paris the capital of, and is it large?"
    p, _ = compiler.translate(typed([], [], [("Paris", "capital of", "?x")], [("?y", "is large")], answer="?x"), q)
    assert (p.anchors, p.variables, p.answer, p.conditions) == (["Paris"], ["?x"], "?x", {})
    assert "fallback" not in p.raw
    one = json.loads(typed(["Paris"], ["?x"], [("Paris", "capital of", "?x")], answer="?x"))
    one["atoms"].append({"subject": "?x", "relation": "is large", "object": None, "evidence": "image"})
    p, pictured = compiler.translate(json.dumps(one), q)
    assert p.conditions == {"?x": ["is large"]} and pictured == {("?x", "is large")} and len(p.edges) == 1
    slip = typed(["Paris"], ["?x"], [("Paris", "capital of", "?x")], answer="?x").replace('"subject": "', '"subject "')
    p, _ = compiler.translate(slip, q)
    assert p.raw["mended"] is True and p.answer == "?x" and "fallback" not in p.raw
    assert "mended" not in compiler.translate(slip.replace('"subject "', '"subject": "'), q)[0].raw
    p, pictured = compiler.translate(typed([], ["?film1"], [], [("?film1", "poster shows a dog", "image"),
                                                                ("?film1", "released in 1990")], answer="?film1"), q)
    assert sig(p) == sig(mgrag(records(("entity", "film1"), ("answer_entity", "film1"),
                                       ("relation", "film1", "film1", "poster shows a dog"),
                                       ("relation", "film1", "film1", "released in 1990")), q))
    assert p.raw["fallback"] == "atoms" and pictured == {("?answer", "poster shows a dog")}


async def test_typed_graph_is_one_json_call_charged_pattern_with_the_question_last():
    q = Question("q", "mmqa", "Where was the performer of Count 'Em 88 born?", ["x"])
    reply = typed(["Count 'Em 88"], ["?person1", "?location1"], [("Count 'Em 88", "performed by", "?person1"),
                                                                 ("?person1", "born in", "?location1")],
                  answer="?location1")
    ctx, backend = context({"typed_graph": reply})
    p, pictured = await compiler.typed_graph(q, ctx)
    assert [(c["task"], c["json_mode"], c["max_tokens"]) for c in backend.seen] == [("typed_graph", True, 700)]
    assert backend.seen[0]["model"] == ctx.llm.model("pattern")
    assert ctx.ledger.coordinator["pattern"].calls == 1 and set(ctx.ledger.coordinator) == {"pattern"}
    sent = json.dumps(backend.seen[0]["messages"])
    assert backend.seen[0]["messages"][-1]["content"].endswith(f"Question: {q.question}")
    assert prompts.MGRAG_SYSTEM not in sent and "anchor_entity" not in sent
    assert p.answer == "?location1" and pictured == frozenset() and p.raw["unbindable"] == []
    ctx, _ = context({"typed_graph": typed(["Hanafuda", "Okey"], [], [("Hanafuda", "same kind of game as", "Okey")],
                                           form="yes/no")})
    p, pictured = await compiler.typed_graph(Question("q", "hotpot", "Are Hanafuda and Okey alike?", ["no"]), ctx)
    assert sig(p) == sig(prompts.parse_pattern("", "Are Hanafuda and Okey alike?")) and pictured == frozenset()
    assert p.raw["fallback"] == "variables" and p.raw["form"] == "yes/no" and p.raw["typed"]["atoms"]


def test_the_prompts_worked_examples_are_typed_graphs_that_translate():
    for q, graph in compiler.EXAMPLES:
        p, _ = compiler.translate(json.dumps(graph), q)
        assert "fallback" not in p.raw and p.raw["unbindable"] == [] and p.raw["form"] in compiler.FORMS, q
        assert json.dumps(graph, ensure_ascii=False) in compiler.TYPED_USER and q in compiler.TYPED_USER
    assert compiler.TYPED_VERSION == "typed-v1"


async def test_typed_repair_rewrites_the_graph_in_the_typed_language():
    graph = ["1. (Meyer Sound Laboratories | based in | country1 = America) -> found country1 = America",
             "2. (country1 = America | company succeeded by | company1) -> searched 3 candidate sources for company1: "
             "none states it"]
    facts, problem = ["(Meyer Sound Laboratories | based in | America)"], "the answer (company1) was not found."
    question = "What succeeded the company of the first British settlers in the country where Meyer Sound is based?"
    reply = typed(["America"], ["?company1", "?company2"], [("?company1", "first British settlers in", "America"),
                                                            ("?company1", "succeeded by", "?company2")],
                  [("?company2", "has a ship on its logo", "image")], answer="?company2")
    ctx, backend = context({"typed_repair": reply})
    p = await compiler.typed_repair(question, graph, facts, problem, ctx)
    assert [(c["task"], c["json_mode"], c["max_tokens"]) for c in backend.seen] == [("typed_repair", True, 600)]
    assert ctx.ledger.coordinator["repair"].calls == 1 and "pattern" not in ctx.ledger.coordinator
    sent = backend.seen[0]["messages"][-1]["content"]
    assert all(line in sent for line in [*graph, *facts, problem, question]) and "anchor_entity" not in sent
    assert p.anchors == ["America"] and p.answer == "?company2"
    assert p.raw["pictured"] == [("?company2", "has a ship on its logo")]
    ctx, _ = context({"typed_repair": typed(["America", "London Company"], [],
                                            [("America", "first settled by", "London Company")])})
    p = await compiler.typed_repair(question, graph, facts, problem, ctx)
    assert p.variables == [] and "fallback" not in p.raw
    ctx, _ = context({"typed_repair": "no graph"})
    p = await compiler.typed_repair(question, graph, facts, problem, ctx)
    assert sig(p) == sig(prompts.parse_pattern("", question)) and p.raw["fallback"] == "json"


async def test_graph_typed_replaces_the_pattern_and_router_calls_and_leaves_the_evaluator_unchanged():
    def responder(task, messages):
        return RUNNING_TYPED if task == "typed_graph" else running_responder(task, messages)

    coord, _ = coordinator(responder=responder)
    old = await coord.run(build_method("fedquest", {}), Q, "mgrag")
    coord, _ = coordinator(responder=responder)
    new = await coord.run(build_method("fedquest", {"graph": "typed"}), Q, "typed")
    assert old.error is None and new.error is None and old.answer == new.answer == "baseball"
    assert [e["event"] for e in old.trace].count("visual") == 1 and "visual" not in [e["event"] for e in new.trace]
    assert new.ledger["coordinator_calls"] == old.ledger["coordinator_calls"] - 1
    keep = ("evidence_tokens", "probes", "rounds", "objects_read", "read_calls")
    assert tuple(new.ledger[k] for k in keep) == tuple(old.ledger[k] for k in keep)
    pattern = Pattern.from_dict(new.pattern)
    assert prune(compiler.translate(RUNNING_TYPED, RUNNING_QUESTION)[0], RUNNING_QUESTION).edges == pattern.edges
    assert pattern.raw["typed"] == json.loads(RUNNING_TYPED) and pattern.raw["unbindable"] == []
