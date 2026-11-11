# edge views (packed sentences per object) and the release and lead signals
import dataclasses
import importlib.util
import json
import re

import numpy as np
import pytest

from fedquest.kg.normalize import release_signal
from fedquest.methods.fedquest import FedQuest, lead_signal
from fedquest.paths import REPO
from fedquest.planner.calibration import Calibrator
from fedquest.types import Candidate, ObjectRecord, ProbeRequest
from fedquest.vdb import views
from fedquest.vdb.encoders import load_encoder
from tests.test_site import llm, world  # noqa: F401

TEXT = ("Tony Hsieh was an American internet entrepreneur and venture capitalist. He was the chief executive of the "
        "online shoe and clothing company Zappos from 1999 to 2020. Hsieh was born in Urbana, Illinois. He graduated "
        "from Harvard University with a degree in computer science.")


def test_sentences_split_at_sentence_ends_only():
    assert views.sentences(TEXT)[-1] == "He graduated from Harvard University with a degree in computer science."
    assert len(views.sentences(TEXT)) == 4
    assert views.sentences("He served in the U.S. army in 1942.") == ["He served in the U.S. army in 1942."]


def test_pack_takes_the_best_sentences_in_text_order_and_fills_the_room():
    ss = views.sentences(TEXT)
    v = views.pack(ss, [3, 0, 2, 1], 160)
    assert v.startswith("Tony Hsieh was an American") and v.endswith("computer science.")
    assert len(v) <= 160 and views.GAP in v and "Zappos" not in v
    full = views.pack(ss, [3, 0, 2, 1], 240)
    assert views.GAP not in full and full.count("…") == 1 and len(full) <= 240
    best = views.pack(ss, [3], 30)
    assert best.endswith("…") and len(best) <= 30
    assert views.pack(ss, [], 240) == ""
    filled = views.pack(["A" * 150 + ".", "B" * 150 + "."], [0, 1], 240)
    assert filled.startswith("A") and filled.endswith("…") and "B" in filled


def test_order_ranks_by_the_key_and_breaks_near_ties_by_the_join_value():
    vecs = np.eye(3, dtype=np.float32)
    key = np.array([0.5, 0.5, 0.0], dtype=np.float32)
    ss = ["Paris is large.", "Rome is old.", "Oslo is cold."]
    assert views.order(ss, vecs, key, []) == [0, 1, 2]
    assert views.order(ss, vecs, key, ["Rome"]) == [1, 0, 2]
    far = np.array([0.9, 0.1, 0.0], dtype=np.float32)
    assert views.order(ss, vecs, far, ["Rome"]) == [0, 1, 2]
    noisy = np.array([0.5, 0.5 + 1e-7, 0.0], dtype=np.float32)
    assert views.order(ss, vecs, noisy, []) == [0, 1, 2]


def test_the_edge_index_encodes_an_object_once_per_version_and_names_its_model():
    enc = load_encoder("hash")
    index = views.EdgeIndex(capacity=2)
    name = views.index_name("hash")
    assert views.VIEW_VERSION in name and views.index_name("bge-base") != views.index_name("e5-base")
    got, built, _ = index.get(name, enc, [("o1", "v1", TEXT), ("o2", "v1", "One. Two.")])
    assert built == 2 and len(got[0][0]) == 4 and got[0][1].dtype == np.float16
    _, built, _ = index.get(name, enc, [("o1", "v1", TEXT)])
    assert built == 0 and index.hits == 1
    _, built, _ = index.get(name, enc, [("o1", "v2", TEXT + " Later.")])
    assert built == 1 and len(index) == 2
    assert index.stats()["held_sentences"] == len(views.sentences(TEXT)) + len(views.sentences(TEXT + " Later."))


def test_the_new_fields_travel_only_when_set():
    plain = ProbeRequest("t(v, r)", "bound", 5).to_dict()
    assert not {"views", "room", "view_key", "signal"} & set(plain)
    d = ProbeRequest("t(v, r)", "bound", 5, views=True, room=360, view_key="q?", signal=["v"]).to_dict()
    assert d["views"] is True and d["room"] == 360 and d["view_key"] == "q?" and d["signal"] == ["v"]
    c = Candidate("s", "text", "o", "text", 1, "T: lead", 10)
    assert "view" not in c.to_dict() and "lead_signal" not in c.to_dict()
    c.view, c.lead_signal = True, False
    back = Candidate.from_dict(c.to_dict())
    assert back.view is True and back.lead_signal is False
    assert ProbeRequest.from_dict({"key": "k", "kind": "bound", "k": 3}).views is False


@pytest.mark.asyncio
async def test_a_site_releases_the_edge_view_once_and_the_leads_signal_as_a_bit(world):  # noqa: F811
    from fedquest.site.service import SNIPPET_CHARS, SiteService

    site = SiteService.open("toy_fed", "aurora", llm=llm())
    try:
        coll = next(c.name for c in site.site.collections if c.modality == "text")
        long = ObjectRecord("toy/text/hsieh#0", "hsieh", "text", "Tony Hsieh", TEXT)
        site._obj = lambda oid: long if oid == long.oid else None  # type: ignore[method-assign]
        lead = f"Tony Hsieh: {TEXT[:SNIPPET_CHARS - 1]}…"
        cand = Candidate("aurora", coll, long.oid, "text", 1, lead, 60)
        short = Candidate("aurora", coll, "toy/text/other#0", "text", 2, "Other: short", 3, complete=True)
        req = ProbeRequest("Tony Hsieh graduated from", "bound", 3, mentions=["Tony Hsieh"], views=True,
                           signal=["Zappos"])
        out = site._viewed([dataclasses.replace(cand), dataclasses.replace(short)], req)
        assert out[0].view is True and out[0].summary.startswith("Tony Hsieh: ")
        assert len(out[0].summary) <= len("Tony Hsieh: ") + SNIPPET_CHARS
        assert out[0].lead_signal is release_signal(lead, "Zappos") is True
        assert out[1].view is None and out[1].summary == "Other: short"
        same = site._viewed([dataclasses.replace(cand)], dataclasses.replace(req, views=False, signal=None))
        assert same[0].view is None and same[0].summary == lead and same[0].lead_signal is None
        longer = site._viewed([dataclasses.replace(cand)], dataclasses.replace(req, views=False, room=360))
        assert longer[0].view is None and longer[0].lead_signal is True
        q = site._viewed([dataclasses.replace(cand)], dataclasses.replace(req, view_key="Where was Hsieh born?"))
        assert q[0].view is True
        busy = site.busy()
        assert busy["view_entries"] == 2 and busy["view_objects"] == 1 and busy["view_index_objects"] >= 1
    finally:
        await site.close()


def test_fedquest_asks_for_its_representation_on_key_requests_only():
    from fedquest.methods.fedquest import _Run

    class Stub:
        def __init__(self, m):
            self.m, self.want_z = m, False
            self.q = type("Q", (), {"question": "Who?"})()
            self.p = type("P", (), {"anchors": ["Anchor"]})()

    def reqs():
        return [ProbeRequest("q", "question", 5), ProbeRequest("t", "bound", 5, mentions=["v"], value="v"),
                ProbeRequest("r", "title", 5, titles=["v"], value="v"), ProbeRequest("", "title", 5, titles=["v"])]

    for m, want in ((FedQuest(), None), (FedQuest(views=True), (True, None, None)),
                    (FedQuest(views=True, view_key="question", room=360), (True, 360, "Who?")),
                    (FedQuest(room=480), (False, 480, None))):
        rs = reqs()
        _Run.versioned(Stub(m), rs)  # type: ignore[arg-type]
        if want is None:
            assert all(not r.views and r.room is None and r.signal is None for r in rs)
            continue
        for r in rs[:3]:
            assert (r.views, r.room, r.view_key) == want
        assert rs[0].signal == ["Anchor"] and rs[1].signal == ["v"] and rs[3].signal is None
    assert FedQuest.added_params["views"] is False and FedQuest.added_params["room"] == 240


def test_the_release_signal_reads_the_sites_bit_when_the_entry_is_not_the_lead():
    c = Candidate("s", "text", "o", "text", 1, "Tony Hsieh: He graduated from Harvard.", 60)
    assert lead_signal(c, ["Harvard"]) is True
    c.lead_signal = False
    assert lead_signal(c, ["Harvard"]) is False


def test_each_representation_reads_its_own_detection_table():
    cal = Calibrator.from_dict(json.loads((REPO / "configs" / "calibration" / "musique_v4e.json").read_text()))
    base = Calibrator.from_dict(json.loads((REPO / "configs" / "calibration" / "musique_v4.json").read_text()))
    assert {"view@240", "qview@240", "view@360", "lead@360", "lead@480"} <= set(cal.entries) and not base.entries
    lead = cal.beta("text", 200, 60)
    assert lead == base.beta("text", 200, 60) == base.beta("text", 200, 60, rep="view@240")
    assert cal.beta("text", 200, 60, rep="view@240") > lead
    assert cal.beta("text", 200, 60, rep="lead@480") > cal.beta("text", 200, 60, rep="lead@360") > lead
    assert cal.beta("text", 200, 60, rep="nothing@1") == lead
    assert cal.to_dict()["entries"] == cal.entries and "entries" not in base.to_dict()


def test_the_v4e_files_are_the_frozen_v4_fits_plus_the_detection_tables():
    for fam in ("musique", "mmqa"):
        v4 = json.loads((REPO / "configs" / "calibration" / f"{fam}_v4.json").read_text())
        v4e = json.loads((REPO / "configs" / "calibration" / f"{fam}_v4e.json").read_text())
        assert {k: v for k, v in v4e.items() if k not in ("entries", "meta")} == {k: v for k, v in v4.items()
                                                                                 if k != "meta"}
        assert v4e["meta"]["views"]["view_version"] == views.VIEW_VERSION
        assert all(t["all"][1] > 0 for t in v4e["entries"].values())
        assert "does not reproduce" not in json.dumps(v4e["meta"]["views"]["records"])


spec = importlib.util.spec_from_file_location("make_configs_edges", REPO / "scripts" / "make_configs.py")
mc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mc)
CATALOG = {n: cfg for n, _, cfg in mc.experiments()}
LEADS = {"fedquest_lead", "fedquest_lead360", "fedquest_lead480", "fedquest_lead_sites"}
LAYOUT = re.compile(r"(.+?)_(?:s\d+(?:_k\d+)?_r\d+|pooled|s2_r0)")


def base_label(label: str) -> str:
    m = LAYOUT.fullmatch(label)
    return m.group(1) if m else label


def test_outside_prefix_keeps_the_questions_no_prefix_study_ran(monkeypatch):
    from fedquest.config import ExperimentCfg
    from fedquest.eval import runner
    from fedquest.types import Question

    qs = [Question(f"q{i}", "d", "?", ["a"], qtype="bridge" if i % 4 else "comparison") for i in range(40)]
    monkeypatch.setattr(runner, "load_questions", lambda *a, **k: qs)
    got = runner.run_questions(ExperimentCfg(name="x", federation="f", split="final_test", outside_prefix=30,
                                             methods=[{"method": "fedquest"}]))
    inside = {q.qid for q in runner.stratified_prefix(qs, 30)}
    assert len(got) == 10 and not inside & {q.qid for q in got}


def _row(qid, acc, trace, ev=1, tokens=5000, gold=()):
    return {"qid": qid, "acc": acc, "evidence": [{}] * ev, "trace": trace, "gold_doc_ids": list(gold),
            "ledger": {"total_in": tokens, "total_out": 100, "evidence_tokens": tokens, "view_entries": 7,
                       "bytes_summary": 2400, "reply_entries": 10}}


VIEWS_TRACE = [{"event": "pool", "var": ["?a"]},
               {"event": "bind_index", "read": ["m/text/x#0", "m/text/y#0"], "values": {"a": ["A"]},
                "cited": ["m/text/x#0"], "cost": 900},
               {"event": "pool", "var": ["?b"]},
               {"event": "bind_index", "read": ["m/text/z#0"], "values": {"b": ["B"]}, "cited": ["m/text/z#0"],
                "cost": 500},
               {"event": "stop", "matched": True}]
LEADS_TRACE = [{"event": "pool", "var": ["?a"]},
               {"event": "bind_index", "read": ["m/text/x#0", "m/text/y#0"], "values": {}, "cited": [], "cost": 900},
               {"event": "read", "site": "s1", "blocks": [["m/text/x#0"]], "tokens": 400, "values": {"a": ["A"]}},
               {"event": "pool", "var": ["?b"]},
               {"event": "bind_index", "read": ["m/text/z#0"], "values": {}, "cited": [], "cost": 500},
               {"event": "stop", "matched": False, "limit": "tokens"}]


def _runs(tmp_path, layout):
    for (exp, label), (acc_, trace) in layout.items():
        d = tmp_path / "results" / exp / "rows_scored"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{label}.jsonl").write_text("\n".join(json.dumps(_row(f"q{i}", acc_, trace, gold=("x", "z")))
                                                    for i in range(4)))
    from fedquest.eval import review_live as rl
    return rl, rl.Rows(tmp_path / "results")
