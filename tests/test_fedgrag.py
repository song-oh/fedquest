# FedGRAG's tuned and trained-encoder checks on a tiny four-database layout
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pytest

FED = "musique_s4_k05_r0"
RECS = [
    ("musique/text/aaaa#0", [["Beitou District", "located in", "Taipei"], ["Beitou District", "known for", "hot springs"]]),
    ("musique/text/bbbb#0", [["Fantasy Land Tour 2004", "performer", "S.H.E"], ["S.H.E", "performed in", "Taipei"],
                             ["S.H.E", "genre", "Mandopop"]]),
    ("musique/text/cccc#0", [["Taipei", "capital of", "Taiwan"], ["Taiwan", "located in", "East Asia"]]),
    ("musique/text/dddd#0", [["Kaohsiung", "located in", "Taiwan"], ["Kaohsiung", "port on", "Taiwan Strait"]]),
    ("musique/text/eeee#0", [["Jay Chou", "born in", "Taipei"], ["Jay Chou", "genre", "Mandopop"]]),
]
MEMBERS = {"aurora_0": ["musique/text/aaaa#0", "musique/text/eeee#0"], "aurora_1": ["musique/text/bbbb#0"],
           "midway3_0": ["musique/text/cccc#0"], "midway3_1": ["musique/text/dddd#0"]}
TRAIN = [
    {"qid": "t1", "question": "Who performed in the city where Beitou District is?", "gold_doc_ids": ["aaaa", "bbbb"]},
    {"qid": "t2", "question": "What is Taipei the capital of?", "gold_doc_ids": ["cccc"]},
    {"qid": "t3", "question": "Which strait is Kaohsiung a port on?", "gold_doc_ids": ["dddd"]},
    {"qid": "t4", "question": "Where was Jay Chou born?", "gold_doc_ids": ["eeee"]},
    {"qid": "t5", "question": "What genre does S.H.E perform?", "gold_doc_ids": ["bbbb"]},
    {"qid": "t6", "question": "What is Beitou District known for?", "gold_doc_ids": ["aaaa"]},
]


@pytest.fixture
def world(tmp_path, monkeypatch):
    pytest.importorskip("sentence_transformers")
    pytest.importorskip("torch")
    monkeypatch.setenv("FEDQUEST_RUNS", str(tmp_path))
    monkeypatch.setenv("FEDQUEST_DATA", str(tmp_path / "data"))
    d = tmp_path / "inputs" / "openie" / "musique"
    d.mkdir(parents=True)
    with (d / "openie.jsonl").open("w") as f:
        for oid, tr in RECS:
            f.write(json.dumps({"oid": oid, "sha": "x", "entities": [], "triples": tr}) + "\n")
    from fedquest.fedgrag import build, model

    model._load_corpus.cache_clear()
    model.clear_caches()
    monkeypatch.setattr(build, "training_questions", lambda federation, dataset: list(TRAIN))
    build.build_layout(FED, members=MEMBERS)
    return tmp_path


def test_settings_defaults_are_the_main_runs_constants():
    from fedquest.fedgrag import model

    s = model.Settings()
    assert (s.n_seeds, s.hops, s.frontier, s.hub_cap, s.n_edges, s.start_facts, s.max_facts) == (
        model.N_SEEDS, model.HOPS, model.FRONTIER, model.HUB_CAP, model.N_EDGES, model.START_FACTS, model.MAX_FACTS)
    assert model.Settings.of(None) == model.DEFAULT == model.Settings.of({})
    assert model.Settings.of({"n_seeds": "10", "unknown": 3}).n_seeds == 10
    assert model.Settings.of(model.Settings(hops=3).as_dict()) == model.Settings(hops=3)


def test_default_settings_retrieve_as_before_and_recall_by_b_is_gold_recall(world):
    from fedquest.fedgrag import build, model

    idx = model.FedGRAGIndex.load(FED)
    qv = model.Encoder.encode([q["question"] for q in TRAIN])
    golds = [set(q["gold_doc_ids"]) for q in TRAIN]
    for v in qv:
        for b in range(1, 5):
            assert idx.retrieve(v, b) == idx.retrieve(v, b, model.DEFAULT)
    for s in (model.DEFAULT, model.Settings(n_seeds=1, hops=1, n_edges=2, start_facts=1, max_facts=3)):
        by_b = build.recall_by_b(idx, qv, golds, s)
        for b in range(1, 5):
            assert by_b[b] == pytest.approx(build.gold_recall(idx, qv, golds, b, s))
    assert build.pick_b({1: 0.5, 2: 0.795, 3: 0.8}) == 2 and build.pick_b({}) == 1


def test_tune_records_the_best_grid_setting_and_its_b(world):
    from fedquest.fedgrag import build, model

    rec = build.tune_layout(FED)["all"]
    grid_keys = set(build.TUNE_GRID)
    assert len(rec["grid"]) == len(list(itertools.product(*build.TUNE_GRID.values())))
    assert rec["recall"] == max(g["recall"] for g in rec["grid"]) >= rec["default"]["recall"]
    assert {k: rec["settings"][k] for k in grid_keys} in [{k: g[k] for k in grid_keys} for g in rec["grid"]]
    assert rec["settings"]["max_facts"] == model.MAX_FACTS and 1 <= rec["b"] <= 4
    idx = model.FedGRAGIndex.load(FED)
    assert idx.tuned() == rec and idx.tuned("xA") is None
    other = "musique_s8_k05_r0"
    build.build_layout(other, members=MEMBERS)
    got = build.tune_layout(other, source=FED)["all"]
    assert got["settings"] == rec["settings"] and got["from"] == FED and "grid" not in got


def test_trained_encoders_build_an_index_beside_the_main_one(world):
    from fedquest.fedgrag import build, model

    before = {p.name: p.read_bytes() for p in model.layout_dir(FED).glob("*.npz")}
    (d,) = build.build_trained(FED)
    assert d == model.layout_dir(FED) / "trained"
    assert {p.name: p.read_bytes() for p in model.layout_dir(FED).glob("*.npz")} == before
    cfg = json.loads((d / "config.json").read_text())
    assert cfg["encoder"] == "trained" and len(cfg["encoder_training"]["loss"]) == build.EPOCHS and 1 <= cfg["b"] <= 4
    idx = model.FedGRAGIndex.load(FED, None, "trained")
    assert idx.qmap.shape == (model.LATENT_DIM, 384) and idx.protos.shape[1] == model.LATENT_DIM
    assert all(c.struct.shape[1] == model.LATENT_DIM for c in idx.clients if len(c.node_ids))
    v = model.Encoder.encode(["Who performed in the city where Beitou District is?"])[0]
    chosen, per, facts = idx.retrieve(v, 4)
    assert len(chosen) == 4 and facts and abs(float((idx.fuse_query(v) ** 2).sum()) - 1.0) < 1e-5
    rec = build.tune_layout(FED, encoder="trained")["all"]
    assert rec["encoder"] == "trained" and model.FedGRAGIndex.load(FED, None, "trained").tuned() == rec
    assert model.FedGRAGIndex.load(FED).tuned() is None
    other = "musique_s8_k05_r0"
    build.build_layout(other, members=MEMBERS)
    (d2,) = build.build_trained(other, source=FED)
    assert json.loads((d2 / "config.json").read_text())["encoder_training"] == {"from": FED}
    with np.load(d / "encoder.npz") as a, np.load(d2 / "encoder.npz") as b:
        assert all((a[k] == b[k]).all() for k in a.files)


def test_variant_dirs_and_unknown_encoders():
    from fedquest.fedgrag import model

    assert model.variant_dir(FED) == model.layout_dir(FED)
    assert model.variant_dir(FED, "xA", "trained") == model.layout_dir(FED) / "trained_xA"
    assert model.router_file(FED, "xA", "trained") == model.layout_dir(FED) / "trained_xA" / "router.npz"
    assert model.router_file(FED, "xA") == model.layout_dir(FED) / "router_xA.npz"
    with pytest.raises(ValueError):
        model.variant_dir(FED, None, "gnn")


def test_methods_register_the_checks_and_keep_the_main_labels_description():
    from fedquest.methods.base import build_method

    plain = build_method("fedgrag", {"index": FED, "router": "x{x}"})
    assert plain.describe() == {"index": FED, "b": None, "passages": 0, "router": "x{x}"}
    assert build_method("fedgrag_text", {"index": FED}).describe() == {"index": FED, "b": None, "passages": 8,
                                                                       "router": None}
    tuned = build_method("fedgrag_tuned", {"index": FED})
    assert (tuned.encoder, tuned.settings, tuned.passages) == (None, "tuned", 0)
    assert tuned.describe()["settings"] == "tuned" and "encoder" not in tuned.describe()
    trained = build_method("fedgrag_trained", {"index": FED, "router": "x{x}"})
    assert (trained.encoder, trained.settings) == ("trained", "tuned")
    assert type(plain).added_params == {"router": None, "encoder": None, "settings": None}
    with pytest.raises(ValueError):
        build_method("fedgrag", {"index": FED, "encoder": "gnn"})
    with pytest.raises(ValueError):
        build_method("fedgrag", {"index": FED, "settings": "best"})


def test_tuned_method_fails_loudly_on_an_untuned_layout(world):
    import asyncio

    from fedquest.methods.base import build_method

    class Ctx:
        ledger = None

        def log(self, *a, **k):
            pass

    q = type("Q", (), {"qid": "q1", "question": "Where was Jay Chou born?"})()
    with pytest.raises(RuntimeError, match="not tuned"):
        asyncio.run(build_method("fedgrag_tuned", {"index": FED}).answer(q, Ctx()))


def test_the_checks_rerun_fedgrag_on_mains_questions_and_at_every_scale_layout():
    from fedquest.config import load_experiment
    from fedquest.methods.base import build_method

    for b, xfit in (("musique", False), ("twowiki", True), ("hotpot", True)):
        exp, main = load_experiment(f"fedgrag_check_{b}"), load_experiment(f"main_{b}")
        assert (exp.federation, exp.split, exp.limit) == (main.federation, main.split, main.limit) and exp.optional
        labs = {m.label: build_method(m.method, m.params) for m in exp.methods}
        assert set(labs) == {"fedgrag_tuned", "fedgrag_trained"}
        assert all(m.index == f"{b}_s4_k05_r0" and m.router == ("x{x}" if xfit else None) for m in labs.values())
        assert (labs["fedgrag_tuned"].encoder, labs["fedgrag_trained"].encoder) == (None, "trained")
        assert {"tool:fedgrag_tune", "tool:fedgrag_tune_trained"} <= set(exp.after)
    exp, scale = load_experiment("fedgrag_check_scale"), load_experiment("scale_musique")
    assert (exp.split, exp.limit) == (scale.split, scale.limit) and exp.optional
    want = {f"{m}_{sfx}" for m in ("fedgrag_tuned", "fedgrag_trained")
            for sfx in ["s2_r0", *(f"s{n}_k05_r{sd}" for n in (4, 8, 16, 32) for sd in range(5))]}
    assert {m.label for m in exp.methods} == want
    assert all(m.params["index"] == "musique_" + m.label.split("_", 2)[2] for m in exp.methods)
    assert not any(a.startswith("fedgrag_check") for a in load_experiment("latency_musique").after)


def test_tools_train_and_tune_the_checks_indexes_source_first():
    root = Path(__file__).resolve().parents[1]
    ids = {t["id"]: t for t in json.loads((root / "configs" / "tools.json").read_text())["tools"]}
    trained, tune, tune_t = ids["tool:fedgrag_trained"], ids["tool:fedgrag_tune"], ids["tool:fedgrag_tune_trained"]
    for t in (trained, tune, tune_t):
        feds = [a for a in t["argv"][4:] if not a.startswith("--") and a not in ("trained",)]
        assert t["argv"][4] == "musique_s4_k05_r0" and {"twowiki_s4_k05_r0", "hotpot_s4_k05_r0",
                                                         "musique_s32_k05_r4", "musique_s2_r0"} <= set(feds)
        assert t["optional"] and t["lane"] == "cpu"
    assert trained["argv"][-3:] == ["--encoder-from", "musique_s4_k05_r0", "--missing"]
    assert tune["argv"][-2:] == ["--from", "musique_s4_k05_r0"] and "--encoder" not in tune["argv"]
    assert tune_t["argv"][-4:] == ["--encoder", "trained", "--from", "musique_s4_k05_r0"]
    assert set(trained["after"]) == set(tune["after"]) == {"tool:fedgrag_build", "tool:fedgrag_seeds",
                                                           "tool:fedgrag_xfit"}
    assert tune_t["after"] == ["tool:fedgrag_trained"]


def test_from_applies_only_to_the_sources_dataset(world, monkeypatch):
    from fedquest.fedgrag import build

    calls = []
    monkeypatch.setattr(build, "tune_layout", lambda f, encoder, src: calls.append((f, encoder, src)) or {})
    monkeypatch.setattr(build, "build_trained", lambda f, src: calls.append((f, "trained", src)) or [])
    assert build.main(["tune", FED, "musique_s8_k05_r0", "twowiki_s4_k05_r0", "--from", FED]) == 0
    assert calls == [(FED, None, None), ("musique_s8_k05_r0", None, FED), ("twowiki_s4_k05_r0", None, None)]
    calls.clear()
    assert build.main(["trained", FED, "hotpot_s4_k05_r0", "--encoder-from", FED]) == 0
    assert calls == [(FED, "trained", None), ("hotpot_s4_k05_r0", "trained", None)]


def test_summary_lists_every_index_and_its_tuning(world, tmp_path):
    from fedquest.fedgrag import build

    build.build_trained(FED)
    build.tune_layout(FED)
    out = tmp_path / "indexes.json"
    assert build.main(["summary", FED, "hotpot_s4_k05_r0", FED, "--out", str(out)]) == 0
    got = json.loads(out.read_text())
    assert got["hotpot_s4_k05_r0"] is None and set(got[FED]) == {"all"}
    free, trained = got[FED]["all"]["free"], got[FED]["all"]["trained"]
    assert free["tuned"]["settings"]["max_facts"] == 60 and trained["tuned"] is None
    assert 1 <= free["b"] <= 4 and len(trained["encoder_training"]["loss"]) == build.EPOCHS
