# how the calibrator shrinks cell rates, picks its strength and fits from probes
import json
import logging
import random

import pytest

from fedquest.planner.calibration import LABEL, SCHEMA, STRENGTHS, Calibrator, bucket, length
from fedquest.utils import read_jsonl

FIELDS = {"qid", "qtype", "family", "round", "kind", "site", "collection", "oid", "modality", "rank", "signal", "entry",
          "tokens", "states", "probe", "evidence"}


def cand(qid, evidence, *, rnd="first", kind="bound", site="a", collection="text", modality="text", rank=1,
         signal=True, entry="complete", tokens=100, states=None, probe=0, oid=None):
    return {"qid": qid, "qtype": "TextQ", "family": "single", "round": rnd, "kind": kind, "site": site,
            "collection": collection, "oid": oid or f"{collection}/{qid}/{rank}", "modality": modality, "rank": rank,
            "signal": signal, "entry": entry, "tokens": tokens, "states": states, "probe": probe, "evidence": evidence}


def gold(qid, *groups, rnd="first"):
    return {"qid": qid, "round": rnd, "path": "gold", "groups": list(groups)}


def shrunk(root, cells, s):
    a = root
    for hit, n in cells:
        a = (hit + s * a) / (n + s)
    return a


def test_an_empty_calibrator_makes_every_candidate_equally_likely_and_keeps_the_defaults():
    cal = Calibrator()
    assert not cal.calibrated
    assert cal.alpha("a", "text", "bound", 1, True, "text") == 1.0 and cal.alpha("b", "image", "title", 9, None) == 1.0
    assert cal.coverage("first", False) == 1.0 and cal.coverage("late", True) == 1.0
    assert cal.beta("text", 200, 50) == pytest.approx(0.25)
    assert cal.tokens("a", "text", 105.0) == 105.0
    assert cal.read_cost("out_empty", 6.0) == 6.0 and cal.read_cost("out_hit", 26.0) == 26.0
    assert cal.cap("table", 12) == 12 and cal.cap("coordinator", 12) == 12


def test_rank_and_object_length_buckets():
    assert [bucket(r) for r in (1, 2, 3, 4, 5, 6, 10, 11, 40)] == ["1", "2-3", "2-3", "4-5", "4-5", "6-10", "6-10",
                                                                    "11+", "11+"]
    assert [length(t) for t in (0, 59, 60, 99, 100, 174, 175, 299, 300, 2000)] == [
        "<60", "<60", "60-99", "60-99", "100-174", "100-174", "175-299", "175-299", "300+", "300+"]


def test_alpha_shrinks_each_cell_toward_the_coarser_one_from_the_pooled_rate():
    obs = [cand(f"q{i}", i < 3) for i in range(4)]
    obs += [cand(f"q{i}", False, site="m", collection="row", modality="table", rank=2, signal=False) for i in range(6)]
    obs.append(cand("q9", None, signal=False))
    cal = Calibrator()
    assert cal.fit(obs) == 10 and cal.calibrated and cal.strength in STRENGTHS
    s, root = cal.strength, 3 / 10
    seen = cal.alpha("a", "text", "bound", 1, True, "text")
    assert seen == pytest.approx(shrunk(root, [(3, 4)] * 6, s))
    assert cal.alpha("z", "text", "bound", 1, True, "text") == pytest.approx(shrunk(root, [(3, 4)] * 5, s))
    assert cal.alpha("z", "image", "bound", 11, False, "image") == pytest.approx(shrunk(root, [(0, 6)] * 2, s))
    assert cal.alpha("a", "text", "bound", 1, True, "text", "late") == pytest.approx(shrunk(root, [(3, 4)] * 3, s))
    assert cal.alpha("a", "text", "title", 1, True, "text") == seen
    for rank, w in ((1, 5 / 7), (2, 1 / 8), (7, 1 / 2)):
        named, unnamed = (cal.alpha("m", "row", "bound", rank, sig, "table") for sig in (True, False))
        assert cal.alpha("m", "row", "bound", rank, None, "table") == pytest.approx(w * named + (1 - w) * unnamed)


def test_the_strength_is_the_one_whose_cells_best_predict_held_out_questions():
    def split(rates):
        rng = random.Random(0)
        return [cand(f"q{q}", rng.random() < p, site=f"s{i}", oid=f"o{q}-{i}")
                for q in range(40) for i, p in enumerate(rates)]

    unlike, alike = Calibrator(), Calibrator()
    unlike.fit(split([0.95, 0.05] * 5))
    alike.fit(split([0.3] * 10))
    assert unlike.strength == min(STRENGTHS) and alike.strength == max(STRENGTHS)
    assert Calibrator.from_dict(alike.to_dict()).strength == max(STRENGTHS)


def test_coverage_counts_the_rounds_whose_gold_a_group_holds_and_shrinks_to_the_round_then_the_split():
    obs = [gold("q0", "text", "image"), gold("q1", "text", "image"), gold("q2", "text"), gold("q3", "text"),
           gold("q0", "text", rnd="late")]
    obs += [cand("q0", True), cand("q1", False), cand("q2", False), cand("q2", True, rank=2), cand("q3", True),
            cand("q0", True, site="b", collection="image", modality="image"),
            cand("q1", False, site="b", collection="image", modality="image"),
            cand("q2", False, site="b", collection="image", modality="image"),
            cand("q0", True, rnd="late", kind="bridge")]
    cal = Calibrator()
    cal.fit(obs)
    s, root = cal.strength, 5 / 7
    assert cal.coverage("first", False) == pytest.approx(shrunk(root, [(4, 6), (3, 4)], s))
    assert cal.coverage("first", True) == pytest.approx(shrunk(root, [(4, 6), (1, 2)], s))
    assert cal.coverage("late", True) == pytest.approx(shrunk(root, [(1, 1)], s))
    assert cal.coverage("filter", False) == pytest.approx(root)
    assert cal.snapshot()["coverage"]["first|image"] == {"hit": 1, "n": 2, "rate": 0.5}


def test_beta_is_the_share_of_evidence_snippets_stating_the_value_by_object_length():
    obs = [cand(f"q{i}", True, entry="snippet", tokens=80, states=i < 3, oid=f"short{i}") for i in range(4)]
    obs += [cand(f"q{i}", True, entry="snippet", tokens=400, states=False, oid=f"long{i}") for i in range(2)]
    obs.append(cand("q0", True, kind="question", entry="snippet", tokens=400, states=False, oid="long0", probe=1))
    obs.append(cand("q5", True, site="m", collection="row", modality="table", entry="snippet", tokens=40, states=True))
    obs += [cand("q6", True, entry="complete", tokens=80), cand("q7", False, entry="snippet", tokens=80)]
    cal = Calibrator()
    cal.fit(obs)
    s, root = cal.strength, 4 / 7
    assert cal.beta("text", 80, 30) == pytest.approx(shrunk(root, [(3, 6), (3, 4)], s))
    assert cal.beta("text", 400, 30) == pytest.approx(shrunk(root, [(3, 6), (0, 2)], s))
    assert cal.beta("text", 120, 30) == pytest.approx(shrunk(root, [(3, 6)], s))
    assert cal.beta("table", 40, 20) == pytest.approx(shrunk(root, [(1, 1), (1, 1)], s))
    plain = Calibrator()
    plain.fit([cand("q0", True), cand("q1", False)])
    assert plain.calibrated and plain.beta("text", 300, 60) == pytest.approx(0.2)


def test_tokens_is_the_mean_size_of_the_objects_a_collection_showed():
    obs = [cand("q0", False, oid="x", tokens=100), cand("q1", True, oid="x", tokens=100, kind="question"),
           cand("q1", False, oid="y", tokens=50),
           cand("q1", None, site="m", collection="row", modality="table", oid="r", tokens=30)]
    cal = Calibrator()
    cal.fit(obs)
    assert cal.tokens("a", "text", 999.0) == pytest.approx(75.0)
    assert cal.tokens("m", "row", 999.0) == pytest.approx(30.0)
    assert cal.tokens("a", "image", 280.0) == 280.0


def test_fit_refuses_observations_without_rounds_or_oids():
    v8 = {"qid": "q0", "site": "a", "collection": "text", "modality": "text", "kind": "bound", "rank": 1,
          "signal": True, "relevance": 0.7, "hop": "anchor", "evidence": True}
    with pytest.raises(ValueError, match="schema 1"):
        Calibrator().fit([v8])


def test_a_calibration_round_trips_and_an_old_label_loads_as_no_calibration(tmp_path, caplog):
    obs = [gold("q0", "text"), gold("q1", "text", "image"), cand("q0", True, entry="snippet", tokens=90, states=True),
           cand("q1", False, rank=4, signal=False), cand("q1", True, site="b", collection="image", modality="image"),
           cand("q1", False, kind="question", site="m", collection="row", modality="table", tokens=40)]
    cal = Calibrator()
    cal.fit(obs)
    path = tmp_path / "toy.json"
    cal.save(path)
    again = Calibrator.load(path)
    assert again.to_dict() == cal.to_dict() and again.to_dict()["label"] == LABEL and again.costs == {}
    for args in (("a", "text", "bound", 1, True, "text"), ("m", "row", "question", 4, None, "table"),
                 ("b", "image", "title", 2, False, "image")):
        assert again.alpha(*args) == cal.alpha(*args)
    assert again.coverage("first", True) == cal.coverage("first", True) and again.beta("text", 90, 60) == \
        cal.beta("text", 90, 60) and again.tokens("m", "row", 0.0) == cal.tokens("m", "row", 0.0) == 40.0
    old = tmp_path / "old.json"
    old.write_text(json.dumps({"label": "evidence", "strength": 8.0, "encoder": "bge-base",
                               "tiers": {"question": [0.5, 0.6, 0.7]}, "counts": {"L0|a|text|bound|1|True|-": [9, 1]},
                               "costs": {"a|text": [3.0, 1, False]}}))
    with caplog.at_level(logging.WARNING):
        legacy = Calibrator.load(old)
    assert not legacy.calibrated and legacy.alpha("a", "text", "bound", 1, True, "text") == 1.0
    assert legacy.costs == {} and any("another label" in m for m in caplog.messages)


def test_fit_writes_the_calibration_its_observations_and_a_summary(tmp_path, monkeypatch):
    from fedquest.eval.calibrate import _fit, refit_calibration

    monkeypatch.setenv("FEDQUEST_RUNS", str(tmp_path))
    obs = [gold("q0", "text", "image"), gold("q1", "text"),
           cand("q0", True, entry="snippet", tokens=80, states=True),
           cand("q0", False, site="b", collection="image", modality="image", entry="title", tokens=280),
           cand("q1", False, rank=3, signal=False),
           cand("q1", None, site="m", collection="row", modality="table", tokens=40)]
    path = _fit("toy", obs, 2, [])
    assert path == tmp_path / "calibration" / "toy.json"
    cal = Calibrator.load(path)
    assert cal.calibrated and list(read_jsonl(path.with_suffix(".obs.jsonl"))) == obs
    summary = json.loads(path.with_suffix(".summary.json").read_text())
    assert summary["schema"] == SCHEMA == 2 and summary["label"] == LABEL and summary["federation"] == "toy"
    assert [summary[k] for k in ("questions", "errors", "candidates", "labeled", "evidence")] == [2, 0, 4, 3, 1]
    assert summary["strength"] == cal.strength
    assert summary["coverage"] == {"first|image": round(cal.coverage("first", True), 4),
                                   "first|text": round(cal.coverage("first", False), 4)}
    assert summary["beta"] == {f"text|{b}": round(cal.beta("text", t, t), 4)
                               for b, t in (("<60", 0), ("60-99", 60), ("100-174", 100), ("175-299", 175),
                                            ("300+", 300))}
    assert summary["cells"]["alpha"]["all"] == {"hit": 1, "n": 3, "rate": round(1 / 3, 4)}
    assert summary["cells"]["tokens"] == {"a|text": 90.0, "b|image": 280.0, "m|row": 40.0}
    path.unlink()
    assert refit_calibration("toy") == path and Calibrator.load(path).to_dict() == cal.to_dict()
    with pytest.raises(RuntimeError, match="no labeled candidates while calibrating toy"):
        _fit("toy", [cand("q0", None)], 1, ["TimeoutError: aurora"])


def on_the_running_example(module, monkeypatch, tmp_path):
    from fedquest.config import FederationCfg, load_models
    from fedquest.eval import calibrate
    from fedquest.kg.matcher import Matcher, TextSim
    from fedquest.llm import prompts
    from fedquest.llm.auth import TokenProvider
    from fedquest.llm.client import LLMClient
    from fedquest.llm.fake import FakeBackend
    from fedquest.types import Question
    from fedquest.vdb.encoders import load_encoder
    from tests.fakes import RUNNING_QUESTION, running_example, running_responder

    monkeypatch.setenv("FEDQUEST_RUNS", str(tmp_path))
    q = Question("q1", "mmqa", RUNNING_QUESTION, ["baseball"], qtype="Compose(ImageQ,TableQ)",
                 bridge=["The Bad News Bears"], meta={"family": "compose", "table_id": "t1", "gold_tables_unknown": [],
                                                      "gold": {"first": ["row:t1#6"], "final": ["image:p1"]}})
    cfg = FederationCfg.model_construct(name="toy", dataset="toy", answer_style="short",
                                        extractor_version=prompts.EXTRACT_VERSION)
    llm = LLMClient(load_models().model_copy(update={"provider": "fake"}), cache=None, tokens=TokenProvider("fake"),
                    backend=FakeBackend(running_responder))
    world = {}

    class Federations:
        async def get(self, name):
            return world["fed"]

    class Client:
        @staticmethod
        def from_env():
            return llm

    for m in {module, calibrate}:
        monkeypatch.setattr(m, "load_federation", lambda name: cfg)
    monkeypatch.setattr(module, "load_questions", lambda name, split, limit=None: [q] if split == "calib" else [])
    monkeypatch.setattr(module, "open_federations", lambda mode, llm, stack: Federations())
    monkeypatch.setattr(module, "LLMClient", Client)
    monkeypatch.setattr(module, "build_matcher", lambda cfg=None: Matcher(TextSim(load_encoder("hash")), 0.6))

    def fresh():
        world["fed"], sites = running_example()
        return sites

    return fresh


async def test_calibrate_fits_the_observations_explore_logs_on_the_calibration_questions(tmp_path, monkeypatch):
    from fedquest.eval import calibrate

    fresh = on_the_running_example(calibrate, monkeypatch, tmp_path)
    sites = fresh()
    current = sites["aurora"].info

    async def stale():
        return {**await current(), "reader_version": "read-v3"}

    sites["aurora"].info = stale
    with pytest.raises(RuntimeError, match="aurora"):
        await calibrate.calibrate("toy")
    assert not sites["midway3"].probes

    sites = fresh()
    path = await calibrate.calibrate("toy")
    assert not any(s.read_calls for s in sites.values())
    obs = list(read_jsonl(path.with_suffix(".obs.jsonl")))
    golds = [o for o in obs if o.get("path") == "gold"]
    cands = [o for o in obs if o.get("path") != "gold"]
    assert golds and cands and all({"qid", "round", "groups"} <= set(o) for o in golds)
    assert all(set(o) >= FIELDS for o in cands)
    assert {"first", "late"} <= {o["round"] for o in golds}
    assert any(o["oid"] == "mmqa/row/t1#6" and o["round"] == "first" and o["evidence"] for o in cands)
    cal = Calibrator.load(path)
    assert cal.calibrated
    schema2 = [o if o.get("path") == "gold" else {k: o[k] for k in FIELDS} for o in obs]
    exact = Calibrator()
    exact.fit(schema2)
    assert exact.to_dict() == cal.to_dict()
    summary = json.loads(path.with_suffix(".summary.json").read_text())
    assert summary["questions"] == 1 and summary["errors"] == 0 and summary["candidates"] == len(cands)


def block(lst, size, found, hits=None, *, modality="text", error=None, calls=1, read=None):
    hits = int(found) if hits is None else hits
    out = 6 + 26 * hits
    return {"qid": f"q{lst}", "list": lst, "modality": modality, "size": size, "site": "a",
            "read": size if read is None else read, "found": found, "false_hits": hits - int(found), "hits": hits,
            "prompt_tokens": 500 + 100 * size - 4 * out, "completion_tokens": out, "cost": 500 + 100 * size,
            "seconds": 0.5 + 0.1 * size, "calls": calls, "cached": False, "error": error}
