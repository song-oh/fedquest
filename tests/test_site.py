import inspect
import json
import logging
import re
import shutil

import pytest

import fedquest.config as config_module
from fedquest.config import FederationCfg
from fedquest.federation.base import Federation, require_current
from fedquest.federation.local import LocalEndpoint
from fedquest.ledger import Ledger
from fedquest.llm import prompts
from fedquest.llm.client import LLMClient
from fedquest.llm.fake import FakeBackend, message_text
from fedquest.llm.prompts import ReadHit, ReadUnknown
from fedquest.paths import CONFIGS
from fedquest.remote.host import ACTIONS, PROFILED, SiteHost
from fedquest.site.service import SiteService, TitleIndex, _validated, candidate
from fedquest.types import ObjectRecord, ProbeRequest, ReadResult, Usage
from fedquest.utils import approx_tokens
from tests.fakes import RUNNING_QUESTION, FakeObject, FakeSite, rules, running_example
from tests.toy import build_toy, responder, toy_env

ROWS = ["toy/row/piazza#1962", "toy/row/piazza#1976"]
FILM = ReadUnknown("film1", ["(Ben Piazza | acted in | X)", "(X | starred | Walter Matthau)"],
                   ["Ben Piazza", "Walter Matthau"]).to_dict()
SPORT = ReadUnknown("sport1", ["(The Bad News Bears | poster depicts sport | X)"], ["The Bad News Bears"]).to_dict()


@pytest.fixture
def world(tmp_path, monkeypatch):
    with toy_env(tmp_path):
        w = build_toy(tmp_path / "world")
        configs = tmp_path / "configs"
        (configs / "federations").mkdir(parents=True)
        (configs / "experiments").mkdir()
        shutil.copy(w.config, configs / "federations" / "toy_fed.yaml")
        shutil.copy(CONFIGS / "models.yaml", configs / "models.yaml")
        monkeypatch.setattr(config_module, "CONFIGS", configs)
        monkeypatch.setenv("FEDQUEST_MODELS", str(configs / "models.yaml"))
        yield w


def llm():
    return LLMClient.from_env(backend=FakeBackend(responder))


def scripted(read):
    def respond(task, messages):
        return json.dumps(read(message_text(messages))) if task == "read" else responder(task, messages)

    return LLMClient.from_env(backend=FakeBackend(respond))


def source(text, needle):
    return next(int(m.group(1)) for m in re.finditer(r"^\[(\d+)\] (.*)$", text, re.M) if needle in m.group(2))


def title_probe(key, titles, k=5, collections=None):
    return ProbeRequest(key, "title", k, collections=collections, titles=titles).to_dict()


def test_the_read_signature_is_the_same_at_the_site_and_the_fake():
    def params(f, skip):
        return [(p.name, p.kind, p.default) for p in list(inspect.signature(f).parameters.values())[skip:]]

    site = params(SiteService.read, 1)
    assert [name for name, _, _ in site][:3] == ["question", "unknowns", "blocks"]
    assert site == params(FakeSite.read, 1)
    assert "read" in ACTIONS and "read" in PROFILED


async def test_stale_sites_are_refused():
    cfg = FederationCfg.model_construct(name="mmqa_fed", extractor_version=prompts.EXTRACT_VERSION)
    fed, _ = running_example()
    await require_current(fed, cfg)
    stale, sites = running_example()
    current = sites["midway3"].info

    async def info():
        return {**await current(), "reader_version": "read-v3"}

    sites["midway3"].info = info
    with pytest.raises(RuntimeError, match=r"mmqa_fed: sites \{'midway3'.*read-v3.*remote.sh up"):
        await require_current(stale, cfg)


async def test_a_site_gets_a_read_only_when_its_blocks_hold_an_oid():
    fed, sites = running_example()
    ledger = Ledger()
    out = await fed.read({"aurora": {"blocks": [[]]}, "midway3": {"blocks": []}}, RUNNING_QUESTION, [SPORT],
                         ledger=ledger)
    assert out == {} and ledger.rounds == 0 and not any(s.read_requests for s in sites.values())
    none = await fed.read({"aurora": {"blocks": [["mmqa/image/p1"]]}}, RUNNING_QUESTION, [], ledger=ledger)
    assert none["aurora"].read == [] and not sites["aurora"].read_calls


def test_a_candidate_carries_its_read_tokens_and_whether_its_summary_is_whole():
    row = ObjectRecord("r", "d", "table", "Ben Piazza — Filmography", "Ben Piazza | Filmography | Year: 1976")
    text = ObjectRecord("t", "d", "text", "Ben Piazza", "x " * 121)
    pic = ObjectRecord("i", "d", "image", "The Bad News Bears", "The Bad News Bears. a poster of a baseball team")
    c = candidate("midway3", "snippets", "row", row, 1)
    assert c.complete and c.tokens == approx_tokens(row.text) and c.summary == f"{row.title}: {row.text}"
    t = candidate("aurora", "snippets", "text", text, 2, "Ben Piazza")
    assert not t.complete and t.summary.endswith("…") and t.label == "Ben Piazza"
    assert t.tokens == approx_tokens(f"Ben Piazza: {text.text}")
    i = candidate("aurora", "snippets", "image", pic, 1)
    assert i.tokens == prompts.IMAGE_TOKENS and not i.complete and i.summary == pic.text
    f = candidate("edge", "facts", "row", row, 1)
    assert f.summary == row.title and not f.complete and f.tokens == c.tokens


def test_a_title_names_the_same_normalized_title_else_at_most_two_titles_that_contain_it():
    index = TitleIndex(["Secret Story (album)", "Björk", "Brockenhurst F.C.", "AFC Brockenhurst Reserves",
                        "History of Brockenhurst", "Brockenhurst F.C.", "Ben Piazza — Filmography", "Ben"])
    assert index.find("Secret Story") == [0] and index.find("Bjork") == [1]
    assert index.find("Ben") == [7] and index.find("Ben Piazza") == [6]
    assert index.find("Brockenhurst") == [2, 4, 5]
    assert index.find("Story") == [0] and index.find("Bjo") == [] and index.find("of") == []
    assert index.find("Brocken") == [] and index.find("") == []


async def test_title_probes_find_an_entitys_own_objects_ranked_by_the_key(world):
    client = llm()
    midway, aurora = SiteService.open("toy_fed", "midway3", llm=client), SiteService.open("toy_fed", "aurora", llm=client)
    (rows,) = await midway.probe([title_probe("No Exit 1962 Camarero", ["Ben Piazza"])])
    assert [(c["oid"], c["rank"], c["label"]) for c in rows] == [(ROWS[0], 1, "Ben Piazza"), (ROWS[1], 2, "Ben Piazza")]
    assert all(c["complete"] and c["tokens"] > 0 for c in rows)
    (top,) = await midway.probe([title_probe("The Bad News Bears 1976 Bob Whitewood", ["Ben Piazza"], k=1)])
    assert [c["oid"] for c in top] == [ROWS[1]]
    pics, short = await aurora.probe([
        title_probe("poster", ["the bad news bears (1976 film)", "Ben Piazza", "No Exit"], k=1, collections=["image"]),
        title_probe("poster", ["Ben"])])
    assert [(c["oid"], c["rank"], c["label"]) for c in pics] == [
        ("toy/image/bnb", 2, "the bad news bears (1976 film)"), ("toy/image/theatre", 1, "Ben Piazza")]
    assert short == []
    with pytest.raises(ValueError, match="unknown probe kind"):
        await aurora.probe([{"key": "Ben Piazza", "kind": "sibling"}])
    for s in (midway, aurora):
        await s.close()
    await client.aclose()


async def test_the_fake_site_finds_titles_as_the_site_does():
    chunks = [FakeObject(f"m/text/dur#{i}", "text", "text", "Durban", f"Durban, part {i}.", []) for i in range(3)]
    beach = FakeObject("m/image/dur", "image", "image", "Durban (city)", "Durban (city)", [])
    pier = FakeObject("m/image/pier", "image", "image", "Durban Pier", "Durban Pier", [])
    site = FakeSite("aurora", [*chunks, beach, pier],
                    rules([("beach", "text", ["m/text/dur#2", "m/text/dur#0"]), ("beach", "image", [pier.oid])]))
    texts, pics = await site.probe([title_probe("beach", ["Durban"], k=2, collections=["text"]),
                                    title_probe("beach", ["Durban", "Durban (city)", "Pier"], collections=["image"])])
    assert [(c["oid"], c["rank"], c["label"]) for c in texts] == [("m/text/dur#2", 1, "Durban"),
                                                                  ("m/text/dur#0", 2, "Durban")]
    assert all(c["complete"] and c["tokens"] == approx_tokens(f"Durban: {chunks[0].text}") for c in texts)
    assert [(c["oid"], c["label"]) for c in pics] == [("m/image/dur", "Durban"), ("m/image/pier", "Pier")]
    assert pics[0]["tokens"] == prompts.IMAGE_TOKENS
    with pytest.raises(ValueError, match="unknown probe kind"):
        await site.probe([{"key": "Durban", "kind": "sibling"}])


async def test_site_read_releases_only_witnessed_values_and_claims(world):
    client = scripted(lambda text: {"job1": [{"entity": "actor", "object": 1, "conditions": [1]}]} if "job1" in text else {
        "film1": [{"entity": "Jaws", "object": source(text, "Bad News"), "conditions": [1]},
                  {"entity": "The Bad News Bears", "object": source(text, "Bad News"), "conditions": [1, 2]}]})
    midway, edge = SiteService.open("toy_fed", "midway3", llm=client), SiteService.open("toy_fed", "edge", llm=client)
    r = ReadResult.from_dict(await midway.read("q", [FILM], [ROWS], prefilter=["The Bad News Bears"], include_raw=True))
    assert r.read == [ROWS[1]] and r.skipped == [ROWS[0]]
    assert _spanless(r.bindings) == {"film1": [{"entity": "The Bad News Bears", "oid": ROWS[1], "conditions": [1]}]}
    assert all(h.get("w") for h in r.bindings["film1"])
    assert r.dropped == 1 and r.usage.calls == 1
    both = ReadResult.from_dict(await midway.read("q2", [FILM], [ROWS], include_raw=True))
    assert both.read == ROWS and list(both.raw) == [ROWS[1]]
    job = ReadUnknown("job1", ["(Ben Piazza | was | X)"], ["Ben Piazza"]).to_dict()
    bio = ReadResult.from_dict(await edge.read("q", [job], [["toy/text/piazza#0"]], include_raw=True))
    assert bio.bindings["job1"][0]["entity"] == "actor" and bio.raw == {}
    for s in (midway, edge):
        await s.close()
    await client.aclose()


async def test_site_read_judges_images_by_what_they_show_and_filters_to_candidates(world):
    client = scripted(lambda text: {
        "sport1": [{"entity": "cricket", "object": 1, "conditions": [1]}],
        "film1": [{"entity": "a dog", "object": 1, "conditions": [1]}, {"entity": "Bad News Bears", "object": 1}]})
    aurora = SiteService.open("toy_fed", "aurora", llm=client)
    r = ReadResult.from_dict(await aurora.read("q", [SPORT], [["toy/image/bnb"]]))
    assert r.bindings["sport1"][0]["entity"] == "cricket"
    film = ReadUnknown("film1", ["X poster shows baseball"], [None], 3, "filter", ["The Bad News Bears", "No Exit"])
    (found,) = await aurora.probe([title_probe("poster shows baseball", film.candidates, k=6, collections=["image"])])
    assert [c["oid"] for c in found] == ["toy/image/bnb"]
    f = ReadResult.from_dict(await aurora.read("q", [film.to_dict()], [[c["oid"] for c in found]],
                                               labels={c["oid"]: c["label"] for c in found}))
    assert f.bindings == {"film1": [{"entity": "The Bad News Bears", "oid": "toy/image/bnb", "conditions": [1]}]}
    assert f.dropped == 1
    await aurora.close()
    await client.aclose()


async def test_labels_drive_filter_validation(world):
    client = scripted(lambda text: {"film1": [{"entity": "Kick 2", "object": 1, "conditions": [1]}]})
    aurora = SiteService.open("toy_fed", "aurora", llm=client)
    film = ReadUnknown("film1", ["X poster shows baseball"], [None], 3, "filter", ["Kick 2", "No Exit"]).to_dict()
    bare = ReadResult.from_dict(await aurora.read("q", [film], [["toy/image/bnb"]]))
    assert not bare.bindings and bare.dropped == 1
    named = ReadResult.from_dict(await aurora.read("q", [film], [["toy/image/bnb"]], labels={"toy/image/bnb": "Kick 2"}))
    assert named.bindings == {"film1": [{"entity": "Kick 2", "oid": "toy/image/bnb", "conditions": [1]}]}
    assert named.titles == {"toy/image/bnb": "Kick 2"}
    await aurora.close()
    await client.aclose()


async def test_a_block_is_one_call_split_only_where_it_mixes_modalities_or_passes_the_cap(world, monkeypatch):
    client = llm()
    midway, aurora = SiteService.open("toy_fed", "midway3", llm=client), SiteService.open("toy_fed", "aurora", llm=client)
    film = ReadUnknown("film1", ["(Ben Piazza | acted in | X)"], ["Ben Piazza"]).to_dict()
    one = ReadResult.from_dict(await midway.read("q", [film], [ROWS]))
    two = ReadResult.from_dict(await midway.read("q", [film], [ROWS[:1], ROWS[1:]]))
    assert one.read == two.read == ROWS and one.usage.calls == 1 and two.usage.calls == 2
    mixed = ReadResult.from_dict(await aurora.read(
        "q", [SPORT], [["toy/image/bnb", "toy/text/matthau#0", "unknown/oid", "toy/image/theatre", "toy/image/bnb"]]))
    assert mixed.read == ["toy/image/bnb", "toy/image/theatre", "toy/text/matthau#0"]
    assert mixed.usage_by_modality["image"].calls == 1 and mixed.usage_by_modality["text"].calls == 1
    monkeypatch.setitem(prompts.PER_CALL, "table", 1)
    assert ReadResult.from_dict(await midway.read("q3", [film], [ROWS])).usage.calls == 2
    for s in (midway, aurora):
        await s.close()
    await client.aclose()


async def test_the_prefilter_skips_texts_and_rows_never_pictures(world):
    client = llm()
    aurora = SiteService.open("toy_fed", "aurora", llm=client)
    block = [["toy/image/theatre", "toy/text/matthau#0"]]
    r = ReadResult.from_dict(await aurora.read("q", [SPORT], block, prefilter=["No Exit"]))
    assert r.read == ["toy/image/theatre"] and r.skipped == ["toy/text/matthau#0"]
    kept = ReadResult.from_dict(await aurora.read("q", [SPORT], block, prefilter=["No Exit"], exempt=[block[0][1]]))
    assert kept.read == block[0] and kept.skipped == []
    gone = ReadResult.from_dict(await aurora.read("q", [SPORT], [block[0][1:]], prefilter=["No Exit"]))
    assert gone.read == [] and gone.usage.calls == 0
    await aurora.close()
    await client.aclose()


async def test_no_unknowns_means_no_call_and_no_image_decoding(world, monkeypatch):
    import fedquest.site.service as service

    client = llm()
    aurora = SiteService.open("toy_fed", "aurora", llm=client)

    def no_decoding(*_args):
        raise AssertionError("an image was decoded")

    monkeypatch.setattr(service, "load_image_data_uri", no_decoding)
    r = ReadResult.from_dict(await aurora.read("q", [], [["toy/image/bnb", "toy/text/matthau#0"]], prefilter=["x"]))
    assert r.read == [] and r.skipped == [] and r.usage == Usage() and not client.backend.calls
    await aurora.close()
    await client.aclose()


async def test_a_failed_call_fails_the_read_names_its_block_and_charges_the_blocks_that_answered(world, caplog):
    def respond(task, messages):
        if task == "read" and "1962" in message_text(messages):
            raise RuntimeError("ALCF login required on this machine")
        return responder(task, messages)

    client = LLMClient.from_env(backend=FakeBackend(respond))
    fed = Federation([LocalEndpoint(SiteService.open("toy_fed", "midway3", llm=client), "midway3")])
    ledger = Ledger()
    film = ReadUnknown("film1", ["(Ben Piazza | acted in | X)"], ["Ben Piazza"]).to_dict()
    with caplog.at_level(logging.WARNING):
        r = (await fed.read({"midway3": {"blocks": [ROWS[1:], ROWS[:1]]}}, "q", [film], ledger=ledger))["midway3"]
    assert r.error == "RuntimeError: ALCF login required on this machine" and not r.bindings
    assert r.usage.calls == 1 and r.usage.prompt_tokens > 0
    assert f"read at midway3 failed in block 2 ({ROWS[0]})" in caplog.text
    assert ledger.extraction_errors == 1 and ledger.extractions == 0 and ledger.reads == 1
    assert sum(u.calls for u in ledger.extraction.values()) == 1
    await fed.close()
    await client.aclose()


async def test_the_fake_site_reads_as_the_site_does():
    fed, sites = running_example()
    sites["aurora"].reader = lambda text: {"sport1": [{"entity": "baseball", "object": source(text, "Bad News"),
                                                       "conditions": [1]}]}
    req = {"aurora": {"blocks": [["mmqa/image/p9", "mmqa/image/p1"], ["mmqa/image/p1"]], "prefilter": ["No Exit"]},
           "midway3": {"blocks": [["mmqa/row/t1#6", "mmqa/row/t1#2"]], "prefilter": ["The Bad News Bears"]}}
    out = await fed.read(req, RUNNING_QUESTION, [SPORT])
    assert sites["aurora"].read_calls == [["mmqa/image/p9", "mmqa/image/p1"]]
    assert out["aurora"].bindings == {"sport1": [{"entity": "baseball", "oid": "mmqa/image/p1", "conditions": [1]}]}
    assert sites["midway3"].read_calls == [["mmqa/row/t1#6"]] and out["midway3"].skipped == ["mmqa/row/t1#2"]
    assert sites["midway3"].read_requests[0]["blocks"] == req["midway3"]["blocks"]


async def test_a_bare_list_of_hits_stands_for_the_only_unknown_never_for_each_of_several():
    fed, sites = running_example()
    sites["midway3"].reader = lambda text: [{"entity": "The Bad News Bears", "object": 1, "conditions": [1]}]
    film = ReadUnknown("film1", ["(Ben Piazza | acted in | X)"], ["Ben Piazza"]).to_dict()
    role = ReadUnknown("role1", ["(Ben Piazza | played | X)"], ["Ben Piazza"]).to_dict()
    req = {"midway3": {"blocks": [["mmqa/row/t1#6"]]}}
    one = (await fed.read(req, RUNNING_QUESTION, [film]))["midway3"]
    assert _spanless(one.bindings) == {"film1": [{"entity": "The Bad News Bears", "oid": "mmqa/row/t1#6",
                                                  "conditions": [1]}]}
    two = (await fed.read(req, RUNNING_QUESTION, [film, role]))["midway3"]
    assert two.bindings == {} and two.usage.calls == 1


def test_no_value_longer_than_a_released_fact_part_leaves_a_site():
    text = "Ben Piazza was an American actor and writer. " * 6
    o = ObjectRecord("toy/text/long#0", "long", "text", "Ben Piazza", text)
    u = ReadUnknown("job1", ["(Ben Piazza | was | X)"], ["Ben Piazza"])
    assert _validated(o, u, ReadHit(text.strip(), 1, [1])) is None
    assert _validated(o, u, ReadHit("American actor", 1, [1]))["entity"] == "American actor"


async def test_site_service_policies_and_replay(world):
    client = llm()
    midway = SiteService.open("toy_fed", "midway3", llm=client)
    edge = SiteService.open("toy_fed", "edge", llm=client)
    (cands,) = await midway.probe([ProbeRequest("Ben Piazza acted in 1976 film", "bound", 2).to_dict()])
    assert [c["rank"] for c in cands] == [1, 2] and "Filmography" in cands[0]["summary"]
    assert "score" not in cands[0] and "vector" not in cands[0]
    first = (await midway.extract(["toy/row/piazza#1976"], include_raw=True))[0]
    again = (await midway.extract(["toy/row/piazza#1976"]))[0]
    assert not first["replay"] and again["replay"] and again["usage"] == first["usage"]
    assert first["raw_text"] and first["facts"][0]["head"] == "Ben Piazza"
    (bio,) = await edge.probe([ProbeRequest("Ben Piazza", "bound", 1).to_dict()])
    assert bio[0]["summary"] == "Ben Piazza" and not bio[0]["complete"]
    assert (await edge.fetch_raw(["toy/text/piazza#0"]))[0]["denied"]
    assert (await edge.extract(["toy/text/piazza#0"], include_raw=True))[0]["raw_text"] is None
    aurora = SiteService.open("toy_fed", "aurora", llm=client)
    raw = (await aurora.fetch_raw(["toy/image/bnb"]))[0]
    assert raw["image_b64"] and raw["nbytes"] > 0
    for s in (midway, edge, aurora):
        await s.close()
    await client.aclose()


async def test_site_read_never_mixes_pixels_and_text_and_replays_at_recorded_cost(world, monkeypatch):
    monkeypatch.setitem(prompts.PER_CALL, "image", 1)
    client = llm()
    fed = Federation([LocalEndpoint(SiteService.open("toy_fed", "aurora", llm=client), "aurora")])
    ledger = Ledger()
    req = {"aurora": {"blocks": [["toy/image/bnb", "toy/image/theatre", "toy/text/matthau#0"]]}}
    r = (await fed.read(req, "q", [SPORT], ledger=ledger))["aurora"]
    assert r.usage.calls == 3 and r.usage_by_modality["image"].calls == 2 and r.usage_by_modality["text"].calls == 1
    assert r.bindings["sport1"][0] == {"entity": "baseball", "oid": "toy/image/bnb", "conditions": [1]}
    assert ledger.rounds == 1 and ledger.extractions == 3 and ledger.reads == 3
    assert ledger.extraction["read_image"].calls == 2 and ledger.extraction["read_text"].calls == 1
    again = (await fed.read(req, "q", [SPORT], ledger=ledger))["aurora"]
    assert again.replay and again.usage == r.usage and ledger.replays == 1
    await fed.close()
    await client.aclose()


async def test_federation_rounds_and_accounting(world):
    client = llm()
    fed = Federation([LocalEndpoint(SiteService.open("toy_fed", s, llm=client), s) for s in ("aurora", "midway3", "edge")])
    ledger = Ledger()
    results = await fed.probe([ProbeRequest("The Bad News Bears", "bound", 2)], ledger=ledger)
    assert ledger.rounds == 1 and ledger.probes == 3 and len(results[0]) >= 3
    ext = await fed.extract([("midway3", "toy/row/piazza#1976"), ("aurora", "toy/image/bnb")], ledger=ledger)
    assert ledger.rounds == 2 and ledger.extractions == 2 and ledger.extraction_tokens > 0
    assert {r.modality for r in ext} == {"table", "image"}
    assert set(await fed.info()) == {"aurora", "midway3", "edge"}
    await fed.close()
    await client.aclose()


class HostEndpoint:
    def __init__(self, host: SiteHost, federation: str, name: str) -> None:
        self.host, self.key, self.name = host, f"{federation}/{name}", name

    async def call(self, action, **kwargs):
        return await self.host.dispatch(self.key, action, kwargs)

    async def close(self):
        return None


async def site_hosts(federation: str, sites: list[str]) -> dict[str, SiteHost]:
    hosts = {s: SiteHost(s, [f"{federation}/{s}"]) for s in sites}
    for host in hosts.values():
        await host.startup()
    return hosts


async def test_site_hosts_serve_the_federation_round_as_the_sites(world):
    hosts = await site_hosts("toy_fed", ["aurora", "midway3", "edge"])
    try:
        health = await hosts["midway3"].dispatch(None, "health", {})
        assert health["facility"] == "midway3" and health["reader_version"] == prompts.READ_VERSION
        fed = Federation([HostEndpoint(h, "toy_fed", s) for s, h in hosts.items()])
        ledger = Ledger()
        (cands,) = await fed.probe([ProbeRequest("Ben Piazza acted in 1976 film", "bound", 1)], ledger=ledger)
        assert {c.site for c in cands} == {"aurora", "midway3", "edge"}
        (res,) = await fed.extract([("midway3", "toy/row/piazza#1976")], ledger=ledger)
        assert res.error is None and ledger.rounds == 2
        film = ReadUnknown("film1", ["(Ben Piazza | acted in | X)"], ["Ben Piazza"]).to_dict()
        read = (await fed.read({"midway3": {"blocks": [ROWS]}}, "q", [film], ledger=ledger))["midway3"]
        assert read.error is None and read.read == ROWS and read.usage.calls == 1 and ledger.rounds == 3
        with pytest.raises(KeyError, match="does not host"):
            await hosts["edge"].dispatch("toy_fed/aurora", "info", {})
    finally:
        for host in hosts.values():
            await host.shutdown()


async def test_an_exact_title_keeps_its_own_picture_and_an_own_article_is_never_a_sequels():
    batman = FakeObject("m/image/br", "image", "image", "Batman Returns", "Batman Returns", [])
    sequel = FakeObject("m/text/bnb2#0", "text", "text", "The Bad News Bears in Breaking Training", "A 1977 sequel.", [])
    film = FakeObject("m/text/bnb#0", "text", "text", "The Bad News Bears (1976 film)", "A 1976 comedy.", [])
    row = FakeObject("m/row/bp#1", "row", "table", "Ben Piazza — Filmography", "Ben Piazza | Filmography | 1976", [])
    site = FakeSite("aurora", [batman, sequel, film, row], rules([]))
    pics, texts, rows = await site.probe([
        title_probe("poster", ["Batman", "Batman Returns"], collections=["image"]),
        title_probe("director", ["The Bad News Bears"], collections=["text"]),
        title_probe("films", ["Ben Piazza"], collections=["row"])])
    assert [(c["oid"], c["label"]) for c in pics] == [("m/image/br", "Batman Returns")]
    assert [c["oid"] for c in texts] == ["m/text/bnb#0"]
    assert [c["oid"] for c in rows] == ["m/row/bp#1"]


def test_a_fenced_reply_with_one_hit_stands_for_its_only_unknown():
    u = ReadUnknown("film1", ["(Ben Piazza | acted in | X)"], ["Ben Piazza"])
    hit = '{"entity": "The Bad News Bears", "object": 1, "conditions": [1]}'
    for text in (hit, f"```json\n{hit}\n```"):
        (h,) = prompts.parse_read(prompts.read_reply(text), [u], 1)["film1"]
        assert h.entity == "The Bad News Bears" and h.object == 1


def _spanless(bindings):
    return {name: [{k: v for k, v in h.items() if k != "w"} for h in hits] for name, hits in bindings.items()}
