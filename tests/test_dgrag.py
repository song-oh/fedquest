from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fedquest.methods.base import build_method
from fedquest.types import Question

FED = "musique_s4_k05_r0"
SITES = ("aurora_0", "aurora_1", "midway3_0", "midway3_1")
PASSAGES = {
    "aurora_0": ("musique/text/aaaa#0", "aaaa", "Beitou District", "Beitou District is located in Taipei.",
                 [("Beitou District", "located in", "Taipei")]),
    "aurora_1": ("musique/text/bbbb#0", "bbbb", "Fantasy Land Tour 2004",
                 "S.H.E performed the Fantasy Land Tour 2004 in Taipei.",
                 [("Fantasy Land Tour 2004", "performer", "S.H.E"), ("S.H.E", "performed in", "Taipei")]),
    "midway3_0": ("musique/text/cccc#0", "cccc", "Taipei", "Taipei is the capital of Taiwan.",
                  [("Taipei", "capital of", "Taiwan")]),
    "midway3_1": ("musique/text/dddd#0", "dddd", "Kaohsiung", "Kaohsiung is located in Taiwan.",
                  [("Kaohsiung", "located in", "Taiwan")]),
}
Q = Question("q7", "musique", "Where did the performer of the Fantasy Land Tour 2004 perform?", ["Taipei"],
             qtype="2hop", gold_doc_ids=["bbbb", "aaaa"],
             meta={"hops": [{"doc_id": "bbbb", "after": []}, {"doc_id": "aaaa", "after": [1]}]})


def _tables(runs: Path) -> None:
    names: list[str] = []
    ids: dict[str, int] = {}
    facts, relations, oids = [], [], []
    for site in SITES:
        oid, _doc, _title, _text, triples = PASSAGES[site]
        for h, r, t in triples:
            for n in (h, t):
                if n.casefold() not in ids:
                    ids[n.casefold()] = len(names)
                    names.append(n)
            facts.append((ids[h.casefold()], ids[t.casefold()]))
            relations.append(r)
            oids.append(oid)
    d = runs / "fedgrag" / "corpus" / "musique"
    d.mkdir(parents=True)
    (d / "tables.json").write_text(json.dumps({"sig": "test", "names": names, "relations": relations,
                                               "fact_oids": oids, "records": len(SITES)}))
    np.savez_compressed(d / "tables.npz", facts=np.asarray(facts, dtype=np.int32).reshape(-1, 2))


def _responder(local_answers: list[str] | None = None, gate: dict | None = None):
    from tests.fakes import running_responder

    turn = list(local_answers or [])

    def respond(task, messages):
        text = "\n".join(str(m["content"]) for m in messages)
        if task == "dgrag_summary":
            ents = [line[2:] for line in text.splitlines() if line.startswith("- ")]
            return "This subgraph covers " + ", ".join(ents) + "."
        if task == "dgrag_keywords":
            return json.dumps({"high_level_keywords": ["concert tour location"],
                               "low_level_keywords": ["Fantasy Land Tour 2004", "S.H.E"]})
        if task == "dgrag_local":
            return f"<ANS>{turn.pop(0) if turn else 'Taipei'}</ANS>"
        if task == "dgrag_gate":
            return json.dumps(gate or {"confident": True, "consistency": 1.0, "best": 1, "reason": "agree"})
        if task == "answer":
            return "<ANS>Taipei</ANS>"
        return running_responder(task, messages)

    return respond


@pytest.fixture
def world(tmp_path, monkeypatch):
    from fedquest.config import load_models
    from fedquest.dgrag import build, index
    from fedquest.llm.auth import TokenProvider
    from fedquest.llm.client import LLMClient
    from fedquest.llm.fake import FakeBackend
    from fedquest.methods.dgrag import DGRAG
    from tests.fakes import FakeObject, FakeSite, fake_federation, rules

    monkeypatch.setenv("FEDQUEST_RUNS", str(tmp_path))
    monkeypatch.setattr("fedquest.fedgrag.build.corpus", lambda dataset, force=False: None)
    monkeypatch.setattr(DGRAG, "batched", None)
    index.clear_caches()
    _tables(tmp_path)
    cfg = load_models().model_copy(update={"provider": "fake"})
    llm = LLMClient(cfg, cache=None, tokens=TokenProvider("fake"), backend=FakeBackend(_responder()))
    members = {s: [PASSAGES[s][0]] for s in SITES}
    out = build.build_layout(FED, members=members, llm=llm, encoder="hash")
    sites = [FakeSite(s, [FakeObject(PASSAGES[s][0], "text", "text", PASSAGES[s][2], PASSAGES[s][3],
                                     PASSAGES[s][4], doc_id=PASSAGES[s][1])], rules([])) for s in SITES]
    yield out, (fake_federation(*sites), {s.name: s for s in sites})
    index.clear_caches()


def test_partition_keeps_every_entity_once_in_subgraphs_of_at_most_40():
    from fedquest.dgrag import MIN_ENTITIES, SUBGRAPH_ENTITIES
    from fedquest.dgrag.build import partition

    rng = np.random.default_rng(0)
    pairs = []
    for c in range(4):
        base = 60 * c
        pairs += [(base + int(a), base + int(b)) for a, b in rng.integers(0, 60, size=(240, 2))]
    pairs += [(0, 60), (60, 120), (120, 180)]
    pairs += [(1000 + 2 * i, 1001 + 2 * i) for i in range(30)]
    nodes = sorted({n for p in pairs for n in p} | {5000})
    parts = partition(nodes, pairs)
    flat = [n for p in parts for n in p]
    assert sorted(flat) == nodes and len(flat) == len(set(flat))
    assert all(len(p) <= SUBGRAPH_ENTITIES for p in parts)
    assert sum(len(p) < MIN_ENTITIES for p in parts) <= 1
    assert partition(nodes, pairs) == parts


def test_the_build_writes_edges_subgraph_summaries_and_their_cost(world):
    out, _ = world
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["clients"] == list(SITES) and cfg["dataset"] == "musique"
    subs = json.loads((out / "subgraphs.json").read_text())
    assert len(subs) == cfg["subgraphs"] == sum(s["subgraphs"] for s in cfg["stats"].values()) >= len(SITES)
    assert all(s["summary"].startswith("This subgraph covers") for s in subs)
    assert cfg["offline"]["calls"] == len(subs) and cfg["offline"]["cost_units"] > 0
    assert np.load(out / "summaries.npy").shape[0] == len(subs)


async def test_agreeing_local_candidates_answer_at_the_edge(world):
    from tests.test_methods import coordinator

    _, w = world
    coord, _ = coordinator(world=w, responder=_responder())
    r = await coord.run(build_method("dgrag", {"index": FED, "total_budget": 16000, "encoder": "hash"}), Q, "dg")
    assert r.error is None and r.answer == "Taipei"
    local = next(e for e in r.trace if e["event"] == "dgrag_local")
    assert not local["escalated"] and local["candidates"] == ["Taipei"] * 3 and local["batched"] is True
    assert local["score"] == pytest.approx(1.0) and local["edge"] in SITES
    assert not any(e["event"] == "dgrag_cloud" for e in r.trace)
    led = r.ledger
    assert led["budget"] == 16000 and led["total_tokens"] <= 16000
    assert led.get("bytes_raw", 0) == 0


async def test_disagreeing_candidates_escalate_to_the_edges_whose_summaries_match(world):
    from tests.test_methods import coordinator

    _, w = world
    coord, sites = coordinator(world=w, responder=_responder(["Taipei", "Kaohsiung", "Taiwan"]))
    m = build_method("dgrag", {"index": FED, "total_budget": 16000, "encoder": "hash", "match_threshold": None,
                               "retrieve_threshold": None})
    r = await coord.run(m, Q, "dg2")
    assert r.error is None and r.answer == "Taipei"
    local = next(e for e in r.trace if e["event"] == "dgrag_local")
    assert local["escalated"] and local["jaccard"] < 0.5
    cloud = next(e for e in r.trace if e["event"] == "dgrag_cloud")
    assert cloud["edges"] and len(cloud["summaries"]) <= 5 and cloud["chunks"] >= 1
    assert set(cloud["edges"]) <= set(SITES)
    led = r.ledger
    assert led["total_tokens"] <= 16000 and led["bytes_raw"] > 0
    assert r.consulted


async def test_a_declined_candidate_escalates_without_the_gates_call(world):
    from tests.test_methods import coordinator

    _, w = world
    coord, _ = coordinator(world=w, responder=_responder(["Not mentioned", "Taipei", "Taipei"]))
    r = await coord.run(build_method("dgrag", {"index": FED, "encoder": "hash", "match_threshold": None}), Q, "dg3")
    local = next(e for e in r.trace if e["event"] == "dgrag_local")
    assert local["escalated"] and local["why"] == "declined"
    assert "coordinator_calls_dgrag_gate" not in r.ledger and r.ledger["coordinator_calls_dgrag_local"] == 1


async def test_chat_n_samples_alone_when_the_endpoint_ignores_n():
    from fedquest.config import load_models
    from fedquest.llm.auth import TokenProvider
    from fedquest.llm.client import LLMClient

    class OneChoice:
        def __init__(self):
            self.calls = 0

        async def complete(self, model, messages, *, max_tokens, temperature, json_mode, task, n=1):
            self.calls += 1
            return f"s{self.calls}", {"prompt_tokens": 10, "completion_tokens": 2}, "stop", 0.0

        async def aclose(self):
            return None

    cfg = load_models().model_copy(update={"provider": "fake"})
    backend = OneChoice()
    llm = LLMClient(cfg, cache=None, tokens=TokenProvider("fake"), backend=backend)
    texts, reply = await llm.chat_n([{"role": "user", "content": "x"}], 3, role="reader", task="t")
    assert reply.samples == 1 and texts == ["s1", "s2", "s3"] and backend.calls == 3
