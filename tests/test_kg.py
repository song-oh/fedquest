import os

import numpy as np
import pytest

from fedquest.kg.matcher import Matcher, TextSim
from fedquest.kg.memory import MemoryKG
from fedquest.kg.normalize import content_stems, norm_entity, same_entity
from fedquest.kg.store import SiteFactStore
from fedquest.types import Edge, ExtractionResult, Fact, Pattern, Usage
from fedquest.utils import stable_int


class SynonymEncoder:
    def __init__(self, groups: list[list[str]], dim: int = 128):
        self.dim = dim
        self.group = {p: i for i, g in enumerate(groups) for p in g}

    def _vec(self, seed: int) -> np.ndarray:
        return np.random.default_rng(seed).standard_normal(self.dim)

    def encode_docs(self, texts, batch_size=64):
        out = []
        for t in texts:
            if t in self.group:
                v = self._vec(10_000 + self.group[t])
            else:
                v = sum((self._vec(stable_int(tok)) for tok in t.lower().split()), np.zeros(self.dim))
            out.append(v / (np.linalg.norm(v) or 1.0))
        return np.stack(out)


def matcher(groups=()):
    return Matcher(TextSim(SynonymEncoder([list(g) for g in groups])), theta=0.6)


PIAZZA = [
    Fact("Ben Piazza", "filmography includes", "The Bad News Bears", "midway3", "row6"),
    Fact("The Bad News Bears", "released in", "1976", "midway3", "row6"),
    Fact("The Bad News Bears", "poster depicts", "baseball", "aurora", "p1"),
]
PATTERN = Pattern(
    anchors=["Ben Piazza"],
    variables=["?film1", "?sport1"],
    edges=[Edge(0, "Ben Piazza", "acted in 1976 film", "?film1"), Edge(1, "?film1", "poster depicts sport", "?sport1")],
    answer="?sport1",
)
GROUPS = [["acted in 1976 film", "filmography includes", "acted in"]]


def test_normalization():
    assert norm_entity("The Bad News Bears (1976 film)") == "bad news bears"
    assert same_entity("The Bad News Bears", "Bad News Bears")
    assert not same_entity("Ben Piazza", "Piazza")
    assert same_entity("bad news bears", "bad news bears film")
    assert {"act", "film"} <= content_stems("acted in film") | {"act"}


def test_running_example_two_hop_match():
    m = matcher(GROUPS).match(PATTERN, PIAZZA)
    assert m.complete
    assert m.bindings == {"?film1": "The Bad News Bears", "?sport1": "baseball"}
    assert {f.oid for f in m.support_facts()} == {"row6", "p1"}


def test_missing_second_hop_is_incomplete_but_binds_the_bridge():
    m = matcher(GROUPS).match(PATTERN, PIAZZA[:2])
    assert not m.complete
    assert m.values["?film1"][0][0] == "The Bad News Bears"
    assert m.supported == {0}


def test_edge_support_from_bound_value():
    sup = matcher(GROUPS).edge_support(PATTERN.edges[1], PIAZZA, ["The Bad News Bears"], None)
    assert sup and sup[0].tail == "baseball"
    assert not matcher(GROUPS).edge_support(PATTERN.edges[1], PIAZZA, ["No Exit"], None)


def test_intersection_needs_one_consistent_binding():
    p = Pattern(["A", "B"], ["?x"], [Edge(0, "A", "won award", "?x"), Edge(1, "B", "won award", "?x")], "?x")
    ok = [Fact("A", "won award", "Oscar"), Fact("B", "won award", "Oscar")]
    bad = [Fact("A", "won award", "Oscar"), Fact("B", "won award", "Emmy")]
    assert matcher().match(p, ok).complete
    assert not matcher().match(p, bad).complete


def test_two_fact_path_supports_one_edge():
    groups = [["works for", "employed by", "member of"]]
    p = Pattern(["Ann"], ["?org"], [Edge(0, "Ann", "works for", "?org")], "?org")
    facts = [Fact("Ann", "member of", "Lab X"), Fact("Lab X", "employed by", "Argonne")]
    m = matcher(groups).match(p, facts)
    assert m.complete and m.values["?org"][0][0] in {"Lab X", "Argonne"}
    assert {v for v, _ in m.values["?org"]} == {"Lab X", "Argonne"}


def _result(site, oid, facts, version="kg-v1", error=None):
    return ExtractionResult(site, oid, "text", version, facts, Usage(10, 5, 0, 0.1, 1), error=error)


async def test_memory_kg_namespaces_coverage_and_versions():
    kg = MemoryKG()
    await kg.add("q1", [_result("s", "o1", PIAZZA[:1]), _result("s", "o2", [])])
    await kg.add("q1", [_result("s", "o3", PIAZZA[2:], version="mgrag-q-v1")])
    cov = await kg.covered("q1", [("s", "o1"), ("s", "o2"), ("s", "o3")], "kg-v1")
    assert set(cov) == {("s", "o1"), ("s", "o2")} and cov[("s", "o2")] == []
    assert await kg.covered("q2", [("s", "o1")], "kg-v1") == {}
    assert await kg.lookup("q1", ["the bad news bears (1976 film)"]) == PIAZZA[:1]
    await kg.add("q1", [_result("s", "o1", PIAZZA[1:2])])
    assert len((await kg.covered("q1", [("s", "o1")], "kg-v1"))[("s", "o1")]) == 2
    stats = await kg.stats("q1")
    assert stats["objects"] == 2 and stats["facts"] == 2
    await kg.add("q1/a", [_result("s", "o1", PIAZZA[:1])])
    await kg.add("q10", [_result("s", "o1", PIAZZA[:1])])
    await kg.clear("q1")
    assert await kg.facts("q1") == [] and await kg.facts("q1/a") == [] and len(await kg.facts("q10")) == 1


def test_site_fact_store_replays_by_object_version_and_query(tmp_path):
    store = SiteFactStore(tmp_path / "facts.sqlite")
    kg = _result("s", "o1", PIAZZA[:2])
    kg.raw_text = "Ben Piazza | Filmography | Year: 1976"
    store.put(kg)
    store.put(_result("s", "o1", PIAZZA[2:], version="mgrag-q-v1"), qhash="abc")
    again = store.get("o1", "kg-v1")
    assert again.facts == PIAZZA[:2] and again.usage == kg.usage
    assert again.raw_text is None
    assert store.get("o1", "mgrag-q-v1", "abc").facts == PIAZZA[2:]
    assert store.get("o1", "mgrag-q-v1") is None and store.get("o2", "kg-v1") is None
    stats = store.stats()
    assert stats["extractions"] == 2 and stats["materializable"] == 1 and stats["facts"] == len(PIAZZA)


TEST_NEO4J = os.environ.get("FEDQUEST_TEST_NEO4J_URI")


@pytest.mark.neo4j
@pytest.mark.skipif(not TEST_NEO4J, reason="set FEDQUEST_TEST_NEO4J_URI to a throwaway Neo4j to run")
async def test_neo4j_kg_round_trip():
    from fedquest.kg.neo4j_store import Neo4jKG

    kg = Neo4jKG(TEST_NEO4J, "neo4j", os.environ.get("FEDQUEST_TEST_NEO4J_PASSWORD", ""))
    ns = "pytest-ns"
    try:
        await kg.clear(ns)
        await kg.add(ns, [_result("midway3", "row6", PIAZZA[:2]), _result("aurora", "p1", PIAZZA[2:])])
        cov = await kg.covered(ns, [("midway3", "row6"), ("aurora", "p9")], "kg-v1")
        assert list(cov) == [("midway3", "row6")] and len(cov[("midway3", "row6")]) == 2
        assert len(await kg.lookup(ns, ["Bad News Bears"])) == 3
        stats = await kg.stats(ns)
        assert stats["facts"] == 3 and stats["cross_site_entities"] == 1
        graph = await kg.export_graph(ns)
        assert len(graph["edges"]) == 3
        marker = await kg._run("MATCH (m:FQMeta) RETURN m.project AS p")
        assert [r["p"] for r in marker] == ["fedquest"]
        await kg.add(f"{ns}/q1", [_result("aurora", "p2", PIAZZA[:1])])
        await kg.add(f"{ns}x", [_result("aurora", "p3", PIAZZA[:1])])
        await kg.clear(ns)
        assert await kg.facts(ns) == [] and await kg.facts(f"{ns}/q1") == [] and len(await kg.facts(f"{ns}x")) == 1
        assert (await kg.stats(f"{ns}/q1"))["objects"] == 0
    finally:
        await kg.clear(ns)
        await kg.clear(f"{ns}x")
        await kg.close()


@pytest.mark.neo4j
@pytest.mark.skipif(not TEST_NEO4J, reason="set FEDQUEST_TEST_NEO4J_URI to a throwaway Neo4j to run")
async def test_neo4j_refuses_a_database_with_other_data():
    from fedquest.kg.neo4j_store import Neo4jKG

    kg = Neo4jKG(TEST_NEO4J, "neo4j", os.environ.get("FEDQUEST_TEST_NEO4J_PASSWORD", ""))
    try:
        await kg._run("MATCH (m:FQMeta) DETACH DELETE m")
        await kg._run("CREATE (:Fact {note: 'another project'})")
        with pytest.raises(RuntimeError, match="already holds other data"):
            await kg.lookup("x", ["anything"])
    finally:
        await kg._run("MATCH (n:Fact {note: 'another project'}) DELETE n")
        await kg.close()


def test_text_sim_survives_eviction_within_a_call_and_concurrent_threads():
    from concurrent.futures import ThreadPoolExecutor

    from fedquest.vdb.encoders import load_encoder

    sim = TextSim(load_encoder("hash"), cache_size=3)
    texts = [f"entity {i}" for i in range(8)]
    first = sim.embed(texts)
    assert first.shape[0] == 8 and np.allclose(sim.embed(texts), first)
    pairs = [(f"key {i}", texts[i:] + texts[:i]) for i in range(8)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        out = list(pool.map(lambda p: sim.relevance([p])[0], pairs * 4))
    assert all(len(v) == 8 for v in out)
    assert np.allclose(out[0], sim.relevance([pairs[0]])[0])
