# parsing model replies, building read prompts, and the client's cache and retries
import asyncio

import pytest

from fedquest.config import load_models
from fedquest.ledger import Ledger
from fedquest.llm import prompts
from fedquest.llm.auth import TokenProvider
from fedquest.llm.cache import LLMCache
from fedquest.llm.client import LLMClient
from fedquest.llm.fake import FakeBackend
from fedquest.llm.parsing import parse_answer, parse_json, parse_records, parse_triples


def test_parse_json_tolerates_fences_prose_and_trailing_commas():
    assert parse_json('```json\n{"a": [1, 2,],}\n```') == {"a": [1, 2]}
    assert parse_json('Sure! {"x": "a } b"} trailing') == {"x": "a } b"}
    assert parse_json("no json here") is None


def test_parse_triples_shapes_and_dedupe():
    obj = {"triples": [["A", "acted in", "B"], ["a", "ACTED IN", "b"], ["", "x", "y"], {"subject": "C", "predicate": "p", "object": "D"}]}
    assert parse_triples(obj) == [("A", "acted in", "B"), ("C", "p", "D")]


def test_parse_records_and_answer():
    recs = parse_records('("anchor_entity"[SEP]Ben Piazza)[REC]("relation"[SEP]A[SEP]B[SEP]r)[END]')
    assert recs == [["anchor_entity", "Ben Piazza"], ["relation", "A", "B", "r"]]
    assert parse_answer("blah <ANS> baseball </ANS>") == "baseball"
    assert parse_answer("Answer: 1976") == "1976"


def test_parse_pattern_running_example_and_fallback():
    text = (
        '("anchor_entity"[SEP]Ben Piazza)[REC]("entity"[SEP]film1)[REC]("entity"[SEP]sport1)[REC]'
        '("answer_entity"[SEP]sport1)[REC]("relation"[SEP]Ben Piazza[SEP]film1[SEP]acted in 1976 film)[REC]'
        '("relation"[SEP]film 1[SEP]sport1[SEP]poster depicts sport)[END]'
    )
    p = prompts.parse_pattern(text, "q")
    assert p.anchors == ["Ben Piazza"] and p.answer == "?sport1"
    assert [(e.head, e.tail) for e in p.edges] == [("Ben Piazza", "?film1"), ("?film1", "?sport1")]
    fallback = prompts.parse_pattern("garbage", "Who won?")
    assert fallback.raw["fallback"] and fallback.edges[0].head == "Who won?"


def test_parse_pattern_keeps_conditions_on_one_unknown_and_named_concepts():
    q = "Pat Metheny won a Grammy Award in what category for the title that has a cover featuring a jumping dog?"
    text = (
        '("anchor_entity"[SEP]Pat Metheny)[REC]("entity"[SEP]album1)[REC]("entity"[SEP]Grammy Award)[REC]'
        '("answer_entity"[SEP]category1)[REC]("relation"[SEP]Pat Metheny[SEP]album1[SEP]released album)[REC]'
        '("relation"[SEP]album1[SEP]album1[SEP]has cover featuring a jumping dog)[REC]'
        '("relation"[SEP]album1[SEP]Grammy Award[SEP]won award)[REC]'
        '("relation"[SEP]Grammy Award[SEP]category1[SEP]awarded in category)[END]'
    )
    p = prompts.parse_pattern(text, q)
    assert p.conditions == {"?album1": ["has cover featuring a jumping dog"]}
    assert "?Grammy Award" not in p.variables and p.answer == "?category1"
    assert [(e.head, e.relation, e.tail) for e in p.edges if e.tail == "?category1"] == [
        ("?album1", "won award Grammy Award awarded in category", "?category1")]
    assert not p.raw.get("answer_inferred")
    guessed = prompts.parse_pattern(text.replace('("answer_entity"[SEP]category1)[REC]', ""), q)
    assert guessed.raw["answer_inferred"] and guessed.answer == "?category1"
    q2 = "What actors were in the film title written by Vakkantham Vamsi whose poster features a motorcycle?"
    text2 = (
        '("anchor_entity"[SEP]Vakkantham Vamsi)[REC]("entity"[SEP]film1)[REC]("entity"[SEP]motorcycle)[REC]'
        '("entity"[SEP]film)[REC]("answer_entity"[SEP]actor1)[REC]'
        '("relation"[SEP]Vakkantham Vamsi[SEP]film1[SEP]wrote)[REC]'
        '("relation"[SEP]film1[SEP]motorcycle[SEP]poster features)[REC]'
        '("relation"[SEP]film1[SEP]actor1[SEP]starred)[REC]("relation"[SEP]film1[SEP]film[SEP]is a)[END]'
    )
    p2 = prompts.parse_pattern(text2, q2)
    assert "motorcycle" in p2.anchors and "?motorcycle" not in p2.variables
    assert any(e.tail == "motorcycle" and e.head == "?film1" for e in p2.edges)
    assert "?film" in p2.variables and p2.answer == "?actor1"


def test_folding_a_picture_moves_its_conditions_to_the_object():
    from fedquest.methods.common import fold_media

    q = "On the poster of the 2004 film Ann Jillian starred in, is the woman smiling?"
    text = (
        '("anchor_entity"[SEP]Ann Jillian)[REC]("entity"[SEP]film1)[REC]("entity"[SEP]poster1)[REC]'
        '("answer_entity"[SEP]smiling1)[REC]("relation"[SEP]Ann Jillian[SEP]film1[SEP]starred in 2004 film)[REC]'
        '("relation"[SEP]film1[SEP]poster1[SEP]has poster)[REC]'
        '("relation"[SEP]poster1[SEP]poster1[SEP]shows a woman)[REC]'
        '("relation"[SEP]poster1[SEP]smiling1[SEP]woman is smiling)[END]'
    )
    p = fold_media(prompts.parse_pattern(text, q))
    assert "?poster1" not in p.variables and p.conditions == {"?film1": ["poster shows a woman"]}


def test_one_visual_vocabulary_for_edges_and_conditions():
    from fedquest.methods.common import VISUAL

    for phrase in ("has cover featuring a jumping dog", "short blonde hair", "logo has a dragon", "poster depicts sport",
                   "is wearing glasses", "artwork shows a castle", "has man holding a guitar on the cover",
                   "has what on his face"):
        assert VISUAL.search(phrase), phrase
    for phrase in ("is a cover of", "featuring Rihanna", "starred in TV show", "won award", "released album", "face-off",
                   "faced charges"):
        assert not VISUAL.search(phrase), phrase


def test_witnessed_normalizes_numbers_and_dates_and_rejects_names_the_text_lacks():
    w = prompts.witnessed
    assert w("attendance was 1,200 people", "1200") and w("born Mar 3, 1990 in Ohio", "3 March 1990")
    assert w("Inside Out (voiced by Richard Kind)", "Richard Kind") and w("anything at all", "yes")
    assert w("Secret Story is an album", "Secret Story (album)")
    assert not w("Ben Piazza was an American actor and writer.", "The Bad News Bears")


def test_known_values_are_matched_leniently_but_named_values_strictly():
    assert prompts.witnessed("Björk videography: Human Behaviour (1993)", "Bjork")
    m = prompts.mentions
    assert m("England at the FIFA World Cup: England are the only team", "world cup of soccer")
    assert m("Climate of Valencia: Valencia has a hot-summer Mediterranean climate", "valencia spain")
    assert not m("All We Had: (Judy Greer | starred in | All We Had)", "Inside Out")
    assert not prompts.witnessed("England at the FIFA World Cup", "world cup of soccer")

def test_claims_need_the_source_to_name_the_known_value():
    two = prompts.ReadUnknown("x", ["(Ann | acted in | X)", "(X | released in | 1976)"], ["Ann", "1976"], 3)
    one = prompts.ReadUnknown("y", ["(Ann | acted in | X)"], ["Ann"], 3)
    out = prompts.parse_read({"x": [{"entity": "Jaws", "object": 1}], "y": [{"entity": "Jaws", "object": 1}]},
                             [two, one], 1)
    assert out["x"][0].conditions == [] and out["y"][0].conditions == [1]
    assert prompts.claims("Ann acted in Jaws", two, out["x"][0]) == [1]
    assert prompts.claims("Ann acted in Jaws", two, prompts.ReadHit("Jaws", 1, [1, 2])) == [1]
    assert prompts.claims(None, two, out["x"][0]) == [1, 2]
    assert prompts.parse_read({"x": [{"entity": "n/a", "object": 1}, {"entity": "Jaws", "object": 3}]}, [two], 2) == {
        "x": []}


def test_read_messages_number_sources_and_conditions_and_interleave_images():
    u = prompts.ReadUnknown("album1", ["X has cover featuring a jumping dog"], [None], 3, "filter",
                            ["Secret Story", "Offramp"])
    objs = [prompts.ReadObject("image", "Secret Story", "data:image/png;base64,AAAA"),
            prompts.ReadObject("image", "Offramp", "data:image/png;base64,BBBB")]
    parts = prompts.read_messages("Which cover shows a dog?", objs, [u])[1]["content"]
    assert [p["type"] for p in parts] == ["text", "text", "image_url", "text", "image_url", "text"]
    assert parts[0]["text"].startswith("Question (context only): Which cover") and parts[1]["text"] == "[1] (image) Secret Story"
    assert "- album1 (at most 3; X is the title an image is labelled with, if that image shows the conditions): (1) X " \
           "has cover featuring a jumping dog" in parts[-1]["text"]
    assert "Candidates for X (found earlier; X may also be another entity): Secret Story; Offramp" in parts[-1]["text"]
    assert prompts.read_max_tokens([u]) == 235 and prompts.ReadUnknown.from_dict(u.to_dict()) == u


def test_a_read_reply_cut_off_at_the_limit_keeps_its_complete_hits():
    u = prompts.ReadUnknown("film1", ["(A | r | X)"], ["A"], 12)
    v = prompts.ReadUnknown("person1", ["(X | s | B)", "(X | t | C)"], ["B", "C"], 12)
    cut = ('```json\n{"film1": [{"entity": "Jaws", "object": 1, "conditions": [1]}, {"entity": "Alien", "object": 2, '
           '"conditions": [1]}], "person1": [{"entity": "Kind", "object": 3, "conditions": ["1", 2]}, {"entity": "Gree')
    hits = prompts.parse_read(prompts.read_reply(cut), [u, v], 3)
    assert [h.entity for h in hits["film1"]] == ["Jaws", "Alien"] and hits["person1"][0].conditions == [1, 2]
    loop = '{"film1": [' + ", ".join(['{"entity": "Jaws", "object": 1, "conditions": [1]}'] * 9) + "]}"
    assert len(prompts.parse_read(prompts.read_reply(loop), [u], 3)["film1"]) == 1

def test_a_named_concept_on_three_edges_stays_a_variable():
    q = "Pat Metheny won a Grammy Award in what category for the title that has a cover featuring a jumping dog?"
    text = ('("anchor_entity"[SEP]Pat Metheny)[REC]("entity"[SEP]album1)[REC]("entity"[SEP]Grammy Award)[REC]'
            '("answer_entity"[SEP]category1)[REC]("relation"[SEP]Pat Metheny[SEP]album1[SEP]released album)[REC]'
            '("relation"[SEP]album1[SEP]Grammy Award[SEP]won award)[REC]'
            '("relation"[SEP]Grammy Award[SEP]category1[SEP]awarded in category)[REC]'
            '("relation"[SEP]Pat Metheny[SEP]Grammy Award[SEP]won)[END]')
    p = prompts.parse_pattern(text, q)
    ends = {n for e in p.edges for n in (e.head, e.tail) if n.startswith("?")}
    assert "?Grammy Award" in p.variables and ends <= set(p.variables)


def test_reads_survive_the_shapes_a_real_model_writes():
    u = prompts.ReadUnknown("n1", ["(A | r | X)"], ["A"], 3, answer=True)
    hits = prompts.parse_read({"n1": [{"entity": 12, "object": 1}, {"entity": True, "object": 1}]}, [u], 1)
    assert [h.entity for h in hits["n1"]] == ["12", "yes"]
    v = prompts.ReadUnknown("p", ["(X | s | B)", "(X | t | C)"], ["B", "C"], 12)
    twice = {"p": [{"entity": "Kind", "object": 1, "conditions": [1]}, {"entity": "Kind", "object": 1, "conditions": [2]}]}
    assert prompts.parse_read(twice, [v], 1)["p"][0].conditions == [1, 2]
    prose = 'Source [1] states it:\n{"n1": [{"entity": "Jaws", "object": 1, "conditions": [1], "reason": "row 1"}]}'
    assert prompts.parse_read(prompts.read_reply(prose), [u], 1)["n1"][0].entity == "Jaws"
    bare = '```json\n[{"entity": "Jaws", "object": 1, "conditions": [1]}]\n```'
    assert prompts.parse_read(prompts.read_reply(bare), [u], 1)["n1"][0].entity == "Jaws"
    assert prompts.stated("a film released in the United States", u, "12")
    assert not prompts.stated("a film released in the United States", u, "1979")


def test_an_image_hit_without_conditions_claims_only_what_its_label_names():
    u = prompts.ReadUnknown("s", ["(The Bad News Bears | poster depicts | X)", "(No Exit | poster depicts | X)"],
                            ["The Bad News Bears", "No Exit"], 3)
    hit = prompts.ReadHit("baseball", 1, [])
    assert prompts.claims(None, u, hit, "The Bad News Bears") == [1]
    assert prompts.claims(None, u, hit, "untitled") == [1, 2]

def test_query_extraction_and_judge_prompts():
    triples = prompts.parse_query_extraction('("entity"[SEP]A[SEP]d)[REC]("relation"[SEP]A[SEP]B[SEP]directed by[SEP]why)[END]')
    assert triples == [("A", "directed by", "B")]
    m1 = prompts.judge_messages("q", "1976", "in 1976", 1)[0]["content"]
    assert "User Answer: in 1976" in m1 and "Reference Answer: 1976" in m1
    m2 = prompts.judge_messages("q", "1976", "in 1976", 2)[0]["content"]
    assert m2.index("Reference Answer: 1976") < m2.index("User Answer: in 1976")
    assert prompts.parse_rating("Rating: 4") == 1.0 and prompts.parse_rating("none") is None


def test_image_messages_use_content_parts():
    msgs = prompts.extract_image_messages("Poster", None, "data:image/jpeg;base64,AAAA")
    parts = msgs[1]["content"]
    assert parts[0]["type"] == "text" and parts[1]["image_url"]["url"].startswith("data:image/jpeg")
    q = prompts.extract_query_messages("q", "", "data:image/jpeg;base64,AAAA")[1]["content"]
    assert q[0]["type"] == "image_url"


def _client(tmp_path, backend, provider="fake"):
    cfg = load_models().model_copy(update={"provider": provider, "max_retries": 3})
    return LLMClient(cfg, cache=LLMCache(tmp_path / "c.sqlite"), tokens=TokenProvider("fake"), backend=backend)


async def test_cache_hit_replays_and_charges_recorded_usage(tmp_path):
    backend = FakeBackend(lambda task, msgs: "<ANS>x</ANS>")
    client = _client(tmp_path, backend)
    msgs = [{"role": "user", "content": "hello world " * 10}]
    l1, l2 = Ledger(), Ledger()
    r1 = await client.chat(msgs, role="answer", task="answer", ledger=l1, charge="answer")
    r2 = await client.chat(msgs, role="answer", task="answer", ledger=l2, charge="answer")
    assert not r1.cached and r2.cached and len(backend.calls) == 1
    assert l1.coordinator["answer"].cost_units == l2.coordinator["answer"].cost_units > 0


async def test_retries_transient_errors_but_not_wrapped_4xx(tmp_path, monkeypatch):
    import httpx
    import openai

    async def no_sleep(*_):
        return None

    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    req = httpx.Request("POST", "http://x")

    class Flaky(FakeBackend):
        def __init__(self, error):
            super().__init__(lambda t, m: "ok")
            self.error, self.n = error, 0

        async def complete(self, *a, **k):
            self.n += 1
            if self.n == 1:
                raise self.error
            return await super().complete(*a, **k)

    transient = openai.APIStatusError("busy", response=httpx.Response(503, request=req), body=None)
    flaky = Flaky(transient)
    reply = await _client(tmp_path, flaky).chat([{"role": "user", "content": "a"}], role="answer", task="answer")
    assert reply.text == "ok" and flaky.n == 2

    wrapped = openai.APIStatusError("API request failed with status code: 400", response=httpx.Response(500, request=req), body=None)
    with pytest.raises(openai.APIStatusError):
        await _client(tmp_path, Flaky(wrapped)).chat([{"role": "user", "content": "b"}], role="answer", task="answer")


async def test_chat_json_retries_once(tmp_path):
    replies = iter(["not json", '{"ok": true}'])
    client = _client(tmp_path, FakeBackend(lambda t, m: next(replies)))
    obj, _ = await client.chat_json([{"role": "user", "content": "x"}], role="planner", task="planner")
    assert obj == {"ok": True}


async def test_per_model_concurrency_cap(tmp_path):
    active, peak = 0, 0

    class Slow(FakeBackend):
        async def complete(self, *a, **k):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.01)
            active -= 1
            return "ok", {"prompt_tokens": 1, "completion_tokens": 1}, "stop", 0.0

    cfg = load_models().model_copy(update={"provider": "fake", "model_concurrency": {"google/gemma-4-31B-it": 2}})
    client = LLMClient(cfg, cache=None, tokens=TokenProvider("fake"), backend=Slow())
    await asyncio.gather(*(client.chat([{"role": "user", "content": str(i)}], role="answer", task="answer") for i in range(8)))
    assert peak == 2
