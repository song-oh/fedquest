import asyncio
import dataclasses
import multiprocessing
import os
import random
import time
import uuid

import pytest

from fedquest.kg.memory import MemoryKG
from fedquest.types import ExtractionResult, Fact, Usage
from fedquest.utils import stable_hash

TEST_URI = os.environ.get("FEDQUEST_TEST_NEO4J_URI")
TEST_PASSWORD = os.environ.get("FEDQUEST_TEST_NEO4J_PASSWORD")

TXT, PIC, ROW = ("aurora", "mmqa/text/o"), ("aurora", "mmqa/image/p"), ("midway3", "mmqa/row/q")
NO_FACTS = {"facts": 0, "stale": 0, "entities": 0, "harvested": 0, "questions": 0, "by_writer": {}, "tuples": 0}


def result(ref, facts, version="read-v5", qid=""):
    return ExtractionResult(ref[0], ref[1], "text", version, list(facts), Usage(), qid=qid)


def stats_of(**counts):
    return {**NO_FACTS, **counts}


def _neo4j(kg):
    return not isinstance(kg, MemoryKG)


async def _layout(kg, ns):
    rows = await kg._run(
        "RETURN COUNT { MATCH ()-[r:FQ_FACT]->() WHERE r.ns = $ns } AS edges, "
        "size(COLLECT { MATCH ()-[r:FQ_FACT]->() WHERE r.ns = $ns RETURN DISTINCT r.fid }) AS fids, "
        "COUNT { MATCH (e:FQNode) WHERE e.ns = $ns } AS nodes, COUNT { MATCH (o:FQObject) WHERE o.ns = $ns } AS objects, "
        "COUNT { MATCH (v:FQVersion) WHERE v.ns = $ns } AS versions", ns=ns)
    return rows[0].data()


def _new_neo4j(writer=None):
    from fedquest.kg.neo4j_store import Neo4jKG

    return Neo4jKG(TEST_URI, "neo4j", TEST_PASSWORD, writer=writer)


@pytest.fixture(params=["memory", pytest.param("neo4j", marks=pytest.mark.neo4j)])
async def store(request):
    root = f"fkg-{uuid.uuid4().hex[:10]}"
    if request.param == "memory":
        yield MemoryKG(), root
        return
    if not (TEST_URI and TEST_PASSWORD):
        pytest.skip("set FEDQUEST_TEST_NEO4J_URI and FEDQUEST_TEST_NEO4J_PASSWORD to a throwaway Neo4j to run")
    kg = _new_neo4j()
    try:
        yield kg, root
    finally:
        await kg.clear(root)
        await kg.close()


@pytest.fixture
async def neo4j_kg():
    if not (TEST_URI and TEST_PASSWORD):
        pytest.skip("set FEDQUEST_TEST_NEO4J_URI and FEDQUEST_TEST_NEO4J_PASSWORD to a throwaway Neo4j to run")
    root = f"fkg-{uuid.uuid4().hex[:10]}"
    kg = _new_neo4j()
    try:
        yield kg, root
    finally:
        await kg.clear(root)
        await kg.close()


def test_a_facts_version_is_part_of_its_id_and_a_fact_without_one_keeps_its_old_id():
    from fedquest.kg.neo4j_store import _fid

    r, f = result(("s", "o1"), []), Fact("Ben Piazza", "born in", "Little Rock", "s", "o1", "kg-v1", "text")
    assert _fid("ns", r, f) == stable_hash("ns", "s", "o1", "Ben Piazza", "born in", "Little Rock", "s", "o1",
                                           "kg-v1", "text", n=24)
    ids = {_fid("ns", r, dataclasses.replace(f, overs=v)) for v in ("", "aaa111", "bbb222")}
    assert len(ids) == 3


async def test_a_facts_object_version_round_trips_and_two_versions_are_two_facts(store):
    kg, root = store
    ns = f"{root}/overs"
    v1 = Fact("Ben Piazza", "born in", "Little Rock", "s", "o1", "read-v5", "text", "aaa111")
    v2 = dataclasses.replace(v1, overs="bbb222")
    old = Fact("Ben Piazza", "died in", "Sherman Oaks", "s", "o2", "read-v5", "text")
    await kg.add(ns, [result(("s", "o1"), [v1, v2]), result(("s", "o2"), [old])])
    assert await kg.lookup(ns, ["BEN PIAZZA"]) == [v1, v2, old]
    assert await kg.facts(ns) == [v1, v2, old] and [f.overs for f in await kg.facts(ns)] == ["aaa111", "bbb222", ""]
    assert await kg.covered(ns, [("s", "o1"), ("s", "o2")], "read-v5") == {("s", "o1"): [v1, v2], ("s", "o2"): [old]}
    assert (await kg.stats(ns))["facts"] == 3
    assert await kg.versions(ns, [("s", "o2"), ("s", "o1")]) == {("s", "o1"): "bbb222"}
    assert await kg.versions(ns, []) == {} and await kg.versions(f"{root}/other", [("s", "o1")]) == {}


async def test_stale_hides_an_objects_old_version_facts_and_keeps_the_new_ones(store):
    kg, root = store
    ns = f"{root}/stale"
    o1 = [Fact("Bad News Bears", "released in", "1976", *TXT, "read-v5", "text", "v1"),
          Fact("Bad News Bears", "directed by", "Michael Ritchie", *TXT, "read-v5", "text", "v1")]
    p1 = [Fact("Bad News Bears", "poster shows", "baseball", *PIC, "read-v5", "image", "w1")]
    q0 = [Fact("Bad News Bears", "stars", "Walter Matthau", *ROW, "read-v5", "table")]
    await kg.add(ns, [result(TXT, o1), result(PIC, p1), result(ROW, q0)])
    every = [*o1, *p1, *q0]
    assert await kg.lookup(ns, ["The Bad News Bears"]) == every
    assert await kg.stale(ns, {TXT: "v2", PIC: "w1", ROW: "q9"}) == 2
    assert await kg.lookup(ns, ["Bad News Bears"]) == [*p1, *q0]
    assert await kg.lookup(ns, ["Bad News Bears"], include_stale=True) == every
    assert await kg.lookup(ns, ["Michael Ritchie", "Walter Matthau"]) == q0
    assert await kg.lookup(ns, ["Bad News Bears"], limit=1) == p1
    assert await kg.facts(ns) == every
    assert await kg.versions(ns, [ROW, TXT, ("s", "none"), PIC]) == {ROW: "q9", TXT: "v2", PIC: "w1"}
    o2 = [Fact("Bad News Bears", "released in", "1976", *TXT, "read-v5", "text", "v2"),
          Fact("Bad News Bears", "remade in", "2005", *TXT, "read-v5", "text", "v2")]
    await kg.add(ns, [result(TXT, o2)])
    assert await kg.lookup(ns, ["Bad News Bears"]) == [*p1, *q0, *o2]
    assert await kg.stale(ns, {TXT: "v2"}) == 0
    late = Fact("Bad News Bears", "sequel", "Breaking Training", *TXT, "read-v5", "text", "v1")
    await kg.add(ns, [result(TXT, [late, o1[0]])])
    assert await kg.lookup(ns, ["Bad News Bears", "Breaking Training"]) == [*p1, *q0, *o2]
    assert await kg.versions(ns, [TXT]) == {TXT: "v2"} and (await kg.fact_stats(ns))["stale"] == 3
    assert await kg.stale(ns, {TXT: "v1"}) == 2
    assert await kg.lookup(ns, ["Bad News Bears"]) == [*p1, *q0]
    assert await kg.versions(ns, [TXT]) == {TXT: "v1"}
    assert await kg.fact_stats(ns) == stats_of(facts=7, stale=5, entities=7, by_writer={kg.writer: 7})
    assert await kg.stale(ns, {PIC: ""}) == 0 and await kg.versions(ns, [PIC]) == {PIC: "w1"}
    assert await kg.stale(f"{root}/other", {}) == 0 and await kg.lookup(f"{root}/other", ["Bad News Bears"]) == []


async def test_stale_marks_an_object_whose_version_was_recorded_before_any_fact(store):
    kg, root = store
    ns = f"{root}/first"
    assert await kg.stale(ns, {TXT: "v1"}) == 0
    f1 = Fact("Smile", "written by", "Charlie Chaplin", *TXT, "read-v5", "text", "v1")
    await kg.add(ns, [result(TXT, [f1])])
    assert await kg.lookup(ns, ["Smile"]) == [f1]
    assert await kg.stale(ns, {TXT: "v2"}) == 1 and await kg.lookup(ns, ["Smile"]) == []
    assert (await kg.stats(ns))["namespaces"] == 1


async def test_a_repeated_write_changes_nothing(store):
    kg, root = store
    ns = f"{root}/again"
    facts = [Fact("Hub", "names", f"E{i}", "s", f"o{i % 3}", "read-v5", "text", f"v{i % 3}") for i in range(9)]
    batch = [result(("s", f"o{k}"), [f for f in facts if f.oid == f"o{k}"]) for k in range(3)]
    kg.writer = "c1"
    await kg.add(ns, batch)
    held = (await kg.facts(ns), await kg.lookup(ns, ["Hub"]), await kg.stats(ns), await kg.fact_stats(ns),
            await kg.versions(ns, [("s", "o0"), ("s", "o1"), ("s", "o2")]))
    layout = await _layout(kg, ns) if _neo4j(kg) else None
    kg.writer = "c2"
    await kg.add(ns, batch)
    await kg.add(ns, list(reversed(batch)))
    assert (await kg.facts(ns), await kg.lookup(ns, ["Hub"]), await kg.stats(ns), await kg.fact_stats(ns),
            await kg.versions(ns, [("s", "o0"), ("s", "o1"), ("s", "o2")])) == held
    assert held[3] == stats_of(facts=9, entities=10, by_writer={"c1": 9})
    assert await kg.writers(ns, [facts[4], Fact("Hub", "names", "nobody"), facts[0]]) == {facts[4]: "c1",
                                                                                           facts[0]: "c1"}
    if layout is not None:
        assert await _layout(kg, ns) == layout == {"edges": 9, "fids": 9, "nodes": 10, "objects": 3, "versions": 3}


async def test_namespaces_stay_apart_around_a_hub_entity(store):
    kg, root = store
    big, small = f"{root}/big", f"{root}/small"
    hub = [Fact("United States", "has state", f"State {i}", "s", f"o{i // 10}", "read-v5", "text", "v1")
           for i in range(150)]
    await kg.add(big, [result(("s", f"o{k}"), hub[10 * k: 10 * k + 10]) for k in range(15)])
    mine = [Fact("United States", "capital", "Washington", "s", "o0", "read-v5", "text", "v1"),
            Fact("Ohio", "state of", "United States", "s", "o1", "read-v5", "text", "v1")]
    await kg.add(small, [result(("s", "o0"), mine[:1]), result(("s", "o1"), mine[1:])])
    assert await kg.lookup(small, ["United States"]) == mine
    assert await kg.lookup(big, ["United States"], limit=1000) == hub
    assert await kg.stale(big, {("s", "o0"): "v2"}) == 10
    assert await kg.lookup(small, ["United States"]) == mine
    assert len(await kg.lookup(big, ["United States"], limit=1000)) == 140
    assert await kg.versions(small, [("s", "o0")]) == {("s", "o0"): "v1"}
    assert await kg.known(small, ["State 3", "Ohio"]) == {"ohio"}
    assert (await kg.fact_stats(small))["facts"] == 2 and (await kg.stats(small))["entities"] == 3
    assert (await kg.fact_stats(big))["stale"] == 10 and (await kg.fact_stats(small))["stale"] == 0


async def test_fact_stats_count_facts_stale_entities_harvested_and_writers(store):
    kg, root = store
    ns = f"{root}/fstats"
    row, poster = ("midway3", "mmqa/row/t1#6"), ("aurora", "mmqa/image/p1")
    asked = [Fact("Ben Piazza", "acted in", "The Bad News Bears", *row, "read-v5", "table", "r1")]
    more = [Fact("Ben Piazza", "born in", "Little Rock", *row, "read-v5+harvest", "table", "r1"),
            Fact("Ben Piazza", "died in", "Sherman Oaks", *row, "read-v5+harvest", "table", "r1")]
    shows = [Fact("The Bad News Bears", "poster shows", "baseball", *poster, "read-v5", "image", "p1")]
    kg.writer = "c0"
    await kg.add(ns, [result(row, [*asked, *more])])
    kg.writer = "c1"
    await kg.add(ns, [result(poster, shows), result(row, asked)])
    assert await kg.fact_stats(ns) == stats_of(facts=4, entities=5, harvested=2, by_writer={"c0": 3, "c1": 1})
    assert await kg.writers(ns, [shows[0], asked[0], *more]) == {shows[0]: "c1", asked[0]: "c0", more[0]: "c0",
                                                                 more[1]: "c0"}
    assert await kg.stale(ns, {row: "r2"}) == 3
    assert await kg.fact_stats(ns) == stats_of(facts=4, stale=3, entities=5, harvested=2,
                                               by_writer={"c0": 3, "c1": 1})
    assert await kg.fact_stats(f"{root}/never") == NO_FACTS and await kg.writers(f"{root}/never", shows) == {}


async def test_clear_takes_the_versions_and_marks_with_the_namespace(store):
    kg, root = store
    fact = Fact("X", "r", "Y", *TXT, "read-v5", "text", "v1")
    for sub in ("a", "a/b", "ab"):
        await kg.add(f"{root}/{sub}", [result(TXT, [fact])])
        assert await kg.stale(f"{root}/{sub}", {TXT: "v2"}) == 1
    await kg.clear(f"{root}/a")
    for sub in ("a", "a/b"):
        ns = f"{root}/{sub}"
        assert await kg.versions(ns, [TXT]) == {} and await kg.fact_stats(ns) == NO_FACTS
        assert await kg.lookup(ns, ["X"], include_stale=True) == [] and await kg.facts(ns) == []
        if _neo4j(kg):
            assert await _layout(kg, ns) == {"edges": 0, "fids": 0, "nodes": 0, "objects": 0, "versions": 0}
    await kg.add(f"{root}/a", [result(TXT, [fact])])
    assert await kg.lookup(f"{root}/a", ["X"]) == [fact]
    ab = f"{root}/ab"
    assert await kg.versions(ab, [TXT]) == {TXT: "v2"} and await kg.fact_stats(ab) == stats_of(
        facts=1, stale=1, entities=2, by_writer={kg.writer: 1})


async def test_a_fact_keeps_the_question_of_its_first_write(store):
    kg, root = store
    ns = f"{root}/qids"
    row, text = ("midway3", "mmqa/row/t1#6"), ("aurora", "mmqa/text/d1#0")
    acted = Fact("Ben Piazza", "acted in", "The Bad News Bears", *row, "read-v5", "table", "r1")
    born = Fact("Ben Piazza", "born in", "Little Rock", *row, "read-v5+harvest", "table", "r1")
    died = Fact("Ben Piazza", "died in", "Sherman Oaks", *text, "read-v5", "text", "t1")
    kg.writer = "c0"
    await kg.add(ns, [result(row, [acted, born], qid="q1")])
    kg.writer = "c1"
    await kg.add(ns, [result(row, [acted], qid="q2"), result(text, [died], qid="q2")])
    late = Fact("Ben Piazza", "nationality", "American", *text, "read-v5", "text", "t1")
    await kg.add(ns, [result(text, [late])])
    assert await kg.qids(ns, [died, acted, Fact("Ben Piazza", "acted in", "nothing"), late, born]) == {
        died: "q2", acted: "q1", late: "", born: "q1"}
    assert await kg.provenance(ns, [acted, died]) == {acted: {"writer": "c0", "qid": "q1", "stale": False},
                                                      died: {"writer": "c1", "qid": "q2", "stale": False}}
    assert await kg.facts(ns) == [acted, born, died, late] and (await kg.stats(ns))["facts"] == 4
    assert await kg.fact_stats(ns) == stats_of(facts=4, entities=5, harvested=1, questions=2,
                                               by_writer={"c0": 2, "c1": 2})
    assert await kg.stale(ns, {row: "r2"}) == 2
    assert (await kg.provenance(ns, [born]))[born] == {"writer": "c0", "qid": "q1", "stale": True}
    assert (await kg.fact_stats(ns))["questions"] == 2
    assert await kg.qids(f"{root}/never", [acted]) == {} and await kg.provenance(ns, []) == {}


async def test_the_tuple_cache_keeps_a_keys_first_payload(store):
    kg, root = store
    ns, other = f"{root}/tuples", f"{root}/tuples-other"
    hits = {"hits": [{"entity": "Théâtre de la Ville", "object": 1, "conditions": [1, 2]}], "at": ("v1", 3)}
    await kg.add_tuples(ns, {"k1": hits, "k2": {"hits": []}})
    await kg.add_tuples(ns, {"k1": {"hits": "a later read"}, "k3": {"n": 1.5, "none": None}})
    first = {"hits": hits["hits"], "at": ["v1", 3]}
    assert await kg.tuples(ns, ["k3", "missing", "k1", "k3"]) == {"k3": {"n": 1.5, "none": None}, "k1": first}
    assert list(await kg.tuples(ns, ["k2", "k1"])) == ["k2", "k1"] and await kg.tuples(ns, []) == {}
    assert await kg.tuples(other, ["k1"]) == {}
    await kg.add_tuples(other, {"k1": {"hits": "other's"}})
    assert await kg.tuples(other, ["k1"]) == {"k1": {"hits": "other's"}} and (await kg.tuples(ns, ["k1"]))["k1"] == first
    assert (await kg.fact_stats(ns))["tuples"] == 3 and (await kg.stats(ns))["tuples"] == 3
    assert "tuples" not in await kg.stats(f"{root}/never") and (await kg.fact_stats(f"{root}/never"))["tuples"] == 0
    await kg.add_tuples(f"{ns}/sub", {"k": {}})
    await kg.clear(ns)
    assert await kg.tuples(ns, ["k1", "k2", "k3"]) == {} and await kg.tuples(f"{ns}/sub", ["k"]) == {}
    assert (await kg.fact_stats(ns))["tuples"] == 0 and await kg.tuples(other, ["k1"]) == {"k1": {"hits": "other's"}}
    with pytest.raises(TypeError):
        await kg.add_tuples(ns, {"bad": {"x": object()}})
    assert await kg.tuples(ns, ["bad"]) == {}


def _db_hits(plan):
    return plan.get("dbHits", 0) + sum(_db_hits(c) for c in plan.get("children", []))


@pytest.mark.neo4j
async def test_a_lookup_reads_none_of_another_namespaces_edges_of_a_hub(neo4j_kg):
    kg, root = neo4j_kg
    big, small = f"{root}/big", f"{root}/small"
    hub = [Fact("United States", "has state", f"State {i}", "s", f"o{i // 10}", "read-v5", "text", "v1")
           for i in range(300)]
    await kg.add(big, [result(("s", f"o{k}"), hub[10 * k: 10 * k + 10]) for k in range(30)])
    mine = [Fact("United States", "capital", "Washington", "s", "o0", "read-v5", "text", "v1"),
            Fact("Ohio", "state of", "United States", "s", "o1", "read-v5", "text", "v1")]
    await kg.add(small, [result(("s", "o0"), mine[:1]), result(("s", "o1"), mine[1:])])
    for include_stale in (False, True):
        query, params = kg._lookup_args(small, ["United States"], 200, include_stale)
        profiled = await kg.driver.execute_query("PROFILE " + query, params, database_=kg.database)
        assert [r["tail"] for r in profiled.records] == ["Washington", "United States"]
        assert _db_hits(profiled.summary.profile) < 100


def _operators(plan):
    return {plan.get("operatorType", "").split("@")[0], *(op for c in plan.get("children", []) for op in _operators(c))}


@pytest.mark.neo4j
async def test_a_tuple_is_one_node_read_by_an_index_seek(neo4j_kg):
    kg, root = neo4j_kg
    ns = f"{root}/tuples"
    await kg.add_tuples(ns, {f"k{i}": {"i": i} for i in range(500)})
    await kg.add_tuples(ns, {"k7": {"i": -1}})
    [node] = await kg._run("MATCH (x:FQTuple {uid: $uid}) RETURN x.ns AS ns, x.key AS key, x.payload AS payload",
                           uid=f"{ns}|k7")
    assert node.data() == {"ns": ns, "key": "k7", "payload": '{"i": 7}'}
    profiled = await kg.driver.execute_query("PROFILE " + _neo4j_tuples(), {"uids": [f"{ns}|k7", f"{ns}|k9"]},
                                             database_=kg.database)
    assert sorted(r["payload"] for r in profiled.records) == ['{"i": 7}', '{"i": 9}']
    assert "NodeUniqueIndexSeek" in _operators(profiled.summary.profile)
    assert _db_hits(profiled.summary.profile) < 50


def _neo4j_tuples():
    from fedquest.kg.neo4j_store import _TUPLES

    return _TUPLES


SHARDS, SPAN, COMMON = 8, 40, 20


def _ring(j):
    j %= SHARDS * SPAN
    return Fact("Hub City", f"rel {j % 3}", f"Entity {j}", "s", f"o{j // 4}", "read-v5", "text", f"v{j // 4}")


def _common(j):
    return Fact("Hub City", "harvested", f"Shared {j}", "c", f"c{j // 4}", "read-v5+harvest", "text", f"w{j // 4}")


def _shard_batches(i):
    by_ref: dict[tuple[str, str], list[Fact]] = {}
    for f in [*(_ring(j) for j in range(i * SPAN, (i + 2) * SPAN)), *(_common(j) for j in range(COMMON))]:
        by_ref.setdefault((f.site, f.oid), []).append(f)
    results = [result(ref, facts, qid=f"q{i}") for ref, facts in by_ref.items()]
    random.Random(i).shuffle(results)
    if i == 0:
        results.sort(key=lambda r: r.oid != "o0")
    return [results[k: k + 3] for k in range(0, len(results), 3)]


def _write_shard(uri, password, ns, i, barrier):
    os.environ["FEDQUEST_SHARD"] = f"c{i}"

    async def go():
        from fedquest.kg.neo4j_store import Neo4jKG

        kg = Neo4jKG(uri, "neo4j", password)
        try:
            await kg.ensure_schema()
            barrier.wait(timeout=120)
            for k, batch in enumerate(_shard_batches(i)):
                await kg.add(ns, batch)
                if i == 0 and k == 0:
                    await kg.stale(ns, {("s", "o0"): "v-new"})
                await kg.add_tuples(ns, {f"t/{r.oid}": {"shard": i, "oid": r.oid} for r in batch})
        finally:
            await kg.close()

    asyncio.run(go())


@pytest.mark.neo4j
async def test_eight_writer_processes_on_one_namespace_lose_nothing_and_duplicate_nothing(neo4j_kg):
    kg, root = neo4j_kg
    ns = f"{root}/shared"
    await kg.ensure_schema()
    ctx = multiprocessing.get_context("spawn")
    barrier = ctx.Barrier(SHARDS)
    procs = [ctx.Process(target=_write_shard, args=(TEST_URI, TEST_PASSWORD, ns, i, barrier)) for i in range(SHARDS)]
    t0 = time.perf_counter()
    for p in procs:
        p.start()
    for p in procs:
        await asyncio.to_thread(p.join, 300)
    seconds = time.perf_counter() - t0
    assert [p.exitcode for p in procs] == [0] * SHARDS, f"a writer failed ({seconds:.1f} s)"
    every = {*(_ring(j) for j in range(SHARDS * SPAN)), *(_common(j) for j in range(COMMON))}
    held = await kg.facts(ns)
    assert len(held) == len(every) == 340 and set(held) == every
    dead = {f for f in every if f.oid == "o0"}
    assert set(await kg.lookup(ns, ["Hub City"], limit=1000)) == every - dead
    assert set(await kg.lookup(ns, ["Hub City"], limit=1000, include_stale=True)) == every
    stats = await kg.fact_stats(ns)
    assert {k: stats[k] for k in ("facts", "stale", "entities", "harvested")} == {
        "facts": 340, "stale": 4, "entities": 1 + SHARDS * SPAN + COMMON, "harvested": COMMON}
    assert sum(stats["by_writer"].values()) == 340 and set(stats["by_writer"]) <= {f"c{i}" for i in range(SHARDS)}
    prov = await kg.provenance(ns, held)
    assert len(prov) == 340 and all(p["qid"] == "q" + p["writer"][1:] for p in prov.values())
    assert stats["questions"] == len({p["qid"] for p in prov.values()}) and 1 < stats["questions"] <= SHARDS
    oids = {f.oid for f in every}
    assert stats["tuples"] == len(oids) == 85
    cached = await kg.tuples(ns, [f"t/{oid}" for oid in sorted(oids)])
    assert len(cached) == 85 and all(v["oid"] == k[2:] and 0 <= v["shard"] < SHARDS for k, v in cached.items())
    assert await _layout(kg, ns) == {"edges": 340, "fids": 340, "nodes": 341, "objects": 85, "versions": 85}
    assert await kg.versions(ns, [("s", "o0"), ("s", "o1")]) == {("s", "o0"): "v-new", ("s", "o1"): "v1"}
    rows = sum(len(b) + sum(len(r.facts) for r in b) for i in range(SHARDS) for b in _shard_batches(i))
    [space] = await kg._run("MATCH (s:FQSpace {ns: $ns}) RETURN s.n AS n", ns=ns)
    assert space["n"] == rows + 1
