# data loaders, gold objects, deterministic partitions and the question workloads
import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from fedquest.config import FederationCfg
from fedquest.data import mmqa, mramg, workloads
from fedquest.data.chunking import chunk_text
from fedquest.data.embed import build_index, embed_federation
from fedquest.data.partition import dirichlet_parts, part_by_title
from fedquest.data.prepare import prepare_federation
from fedquest.paths import collection_dir, data_root, questions_path, site_dir
from fedquest.types import ObjectRecord, Question
from fedquest.utils import read_jsonl
from fedquest.vdb.encoders import load_encoder
from fedquest.vdb.qdrant_index import SiteIndex

SCRATCH = data_root() / "raw"

PIAZZA_TABLE = {
    "title": "Ben Piazza", "id": "t1",
    "table": {
        "table_name": "Filmography",
        "header": [{"column_name": "Year"}, {"column_name": "Title"}, {"column_name": "Role"}],
        "table_rows": [
            [{"text": "1962", "links": []}, {"text": "No Exit", "links": []}, {"text": "Camarero", "links": []}],
            [{"text": "1976", "links": []},
             {"text": "The Bad News Bears", "links": [{"wiki_title": "The Bad News Bears"}]},
             {"text": "Bob Whitewood", "links": []}],
        ],
    },
}


def _question(qid, qtype, gold, bridge=None):
    return {
        "qid": qid, "question": f"question {qid}?", "answers": [{"answer": 1976.0, "type": "number", "modality": "table"}],
        "metadata": {"type": qtype, "modalities": ["table"], "text_doc_ids": ["x1"], "image_doc_ids": ["i1"],
                     "table_id": "t1", "intermediate_answers": [[{"answer": bridge}]] if bridge else []},
        "supporting_context": [{"doc_id": g, "doc_part": "table"} for g in gold],
    }


def _gz(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


@pytest.fixture
def data_env(tmp_path, monkeypatch):
    monkeypatch.setenv("FEDQUEST_DATA", str(tmp_path / "data"))
    monkeypatch.setenv("FEDQUEST_QDRANT_MODE", "local")
    return tmp_path / "data"


def _fake_mmqa(root: Path):
    raw = root / "raw" / "mmqa"
    qs = [_question(f"q{i}", t, ["t1"], bridge="The Bad News Bears" if t.startswith("Compose") else None)
          for i, t in enumerate(["TextQ", "ImageQ", "TableQ", "Compose(ImageQ,TableQ)", "Compose(TextQ,TableQ)", "TextQ"])]
    _gz(raw / "MMQA_dev.jsonl.gz", qs)
    _gz(raw / "MMQA_train.jsonl.gz", [dict(q, qid="tr" + q["qid"]) for q in qs])
    _gz(raw / "MMQA_texts.jsonl.gz", [{"id": f"x{i}", "title": f"Article {i % 4}", "url": "", "text": f"Fact number {i}. " * 5} for i in range(12)])
    _gz(raw / "MMQA_tables.jsonl.gz", [PIAZZA_TABLE])
    _gz(raw / "MMQA_images.jsonl.gz", [{"id": "i1", "title": "The Bad News Bears", "url": "", "path": "i1.jpg"}])


MMQA_CFG = {
    "name": "tiny_mmqa", "dataset": "mmqa",
    "questions": {"split": "dev", "quotas": {"single": 2, "compose": 1}, "calib_split": "train", "calib_n": 2},
    "partition": {"method": "by_title", "parts": 2, "seed": 1},
    "qdrant": {"mode": "local"},
    "sites": [
        {"name": "a", "machine": "edge", "collections": [
            {"name": "text", "modality": "text", "encoder": "hash", "source": "text:0"},
            {"name": "image", "modality": "image", "encoder": "hash", "source": "image:*"}]},
        {"name": "b", "machine": "edge", "policy": "facts", "collections": [
            {"name": "text", "modality": "text", "encoder": "hash", "source": "text:1"},
            {"name": "row", "modality": "table", "encoder": "hash", "source": "table:*"}]},
        {"name": "home", "machine": "edge", "collections": [
            {"name": "all", "modality": "text", "encoder": "hash", "source": "text:*,table:*,image:*"}]},
    ],
}


def test_chunking():
    assert chunk_text("One sentence.") == ["One sentence."]
    text = " ".join(f"Sentence {i} has some words in it." for i in range(80))
    chunks = chunk_text(text, chunk_tokens=64, overlap=16)
    assert len(chunks) > 3 and all(len(c.split()) <= 48 for c in chunks)
    assert chunks[0].split()[-3:] == chunks[1].split()[:3] or chunks[1].startswith(chunks[0].split()[-12])


def test_mmqa_rows_questions_and_families():
    rows = mmqa.row_objects(PIAZZA_TABLE)
    assert rows[1].text == "Ben Piazza | Filmography | Year: 1976; Title: The Bad News Bears; Role: Bob Whitewood"
    assert rows[1].oid == "mmqa/row/t1#1" and rows[1].meta["entities"] == ["The Bad News Bears"]
    q = mmqa.to_question(_question("q", "Compose(ImageQ,TableQ)", ["t1"], "The Bad News Bears"))
    assert q.answers == ["1976"] and q.bridge == ["The Bad News Bears"] and q.meta["family"] == "compose"
    assert [mmqa.family(t) for t in ("TextQ", "ImageListQ", "Intersect(TableQ,TextQ)")] == ["single", "table", "intersect"]


def _raw(qid, qtype, answers, inner, support):
    return {"qid": qid, "question": f"question {qid}?", "answers": answers,
            "metadata": {"type": qtype, "table_id": "t1", "intermediate_answers": [inner] if inner else []},
            "supporting_context": [{"doc_id": d, "doc_part": p} for d, p in support]}


ROW = {"answer": "The Bad News Bears", "type": "string", "modality": "table", "table_indices": [[1, 1]]}
POSTER = {"answer": "baseball", "type": "string", "modality": "image", "image_instances": [{"doc_id": "i1"}]}


def test_mmqa_gold_objects_by_the_hop_they_serve():
    compose = _raw("c", "Compose(ImageQ,TableQ)", [POSTER], [ROW], [("t1", "table"), ("i1", "image")])
    assert mmqa.gold_objects(compose) == ({"first": ["row:t1#1"], "final": ["image:i1"]}, [])
    listed = _raw("l", "Compose(TableQ,ImageListQ)", [{"answer": "2019", "table_indices": [[3, 2]]}],
                  [{"answer": "FC Lorient", "image_instances": [{"doc_id": "i7"}]}],
                  [("i7", "image"), ("i8", "image"), ("t1", "table")])
    assert mmqa.gold_objects(listed) == ({"first": ["image:i7", "image:i8"], "final": ["row:t1#3"]}, [])
    unnamed = _raw("u", "Compose(TextQ,TableQ)", [{"answer": "a", "text_instances": [{"doc_id": "x1"}]}],
                   [{"answer": "b"}], [("t1", "table"), ("x1", "text"), ("x2", "text")])
    assert mmqa.gold_objects(unnamed) == ({"first": [], "final": ["text:x1", "text:x2"]}, ["t1"])
    for qtype in ("TableQ", "Intersect(TableQ,TextQ)", "Compare(Compose(TableQ,ImageQ),TableQ)"):
        other = _raw("o", qtype, [{"answer": "x", "table_indices": [[0, 0]], "text_instances": [{"doc_id": "x1"}]}],
                     [{"answer": "2018", "table_indices": [[2, 0]]}], [("t1", "table"), ("x1", "text")])
        assert mmqa.gold_objects(other) == ({"first": [], "final": ["row:t1#0", "row:t1#2", "text:x1"]}, [])


def test_attach_gold_objects_reads_the_raw_split_each_question_split_was_drawn_from(data_env, monkeypatch):
    from fedquest.eval import calibrate

    raw = data_env / "raw" / "mmqa"
    compose = _raw("c", "Compose(ImageQ,TableQ)", [POSTER], [ROW], [("t1", "table"), ("i1", "image")])
    _gz(raw / "MMQA_train.jsonl.gz", [compose])
    _gz(raw / "MMQA_dev.jsonl.gz", [_raw("c", "ImageQ", [POSTER], [], [("i1", "image")])])
    monkeypatch.setattr(calibrate, "load_federation", lambda name: FederationCfg(**MMQA_CFG))
    calib, test = [mmqa.to_question(compose)], [mmqa.to_question(compose)]
    calibrate.attach_gold_objects("tiny_mmqa", calib, "calib")
    calibrate.attach_gold_objects("tiny_mmqa", test, "test")
    assert calib[0].meta["gold"] == {"first": ["row:t1#1"], "final": ["image:i1"]}
    assert test[0].meta["gold"] == {"first": [], "final": ["image:i1"]} and test[0].meta["gold_tables_unknown"] == []
    mramg = [Question("m", "mramg", "?", ["a"], gold_doc_ids=["d1"])]
    monkeypatch.setattr(calibrate, "load_federation", lambda name: FederationCfg.model_construct(dataset="mramg"))
    calibrate.attach_gold_objects("tiny_mramg", mramg, "calib")
    assert "gold" not in mramg[0].meta


@pytest.mark.skipif(not (SCRATCH / "mmqa" / "MMQA_dev.jsonl.gz").exists(), reason="real MMQA sample not present")
def test_real_mmqa_compose_record():
    with gzip.open(SCRATCH / "mmqa" / "MMQA_dev.jsonl.gz", "rt") as f:
        d = next(json.loads(line) for line in f if json.loads(line)["qid"] == "0d8f2779137fb47db953c4af5247ffe5")
    q = mmqa.to_question(d)
    assert q.answers == ["baseball"] and q.bridge == ["The Bad News Bears"] and q.image_answer


def test_partitions_are_deterministic():
    assert part_by_title("Ben Piazza", 3, 13) == part_by_title(" ben  piazza ", 3, 13)
    ids = [f"d{i}" for i in range(30)]
    texts = [("film actor movie " if i % 2 else "recipe bake flour ") * 5 for i in range(30)]
    a = dirichlet_parts(ids, texts, 3, 0.5, 4, 7)
    assert a == dirichlet_parts(list(reversed(ids)), list(reversed(texts)), 3, 0.5, 4, 7)
    assert set(a.values()) == {0, 1, 2}


def test_prepare_embed_index_mmqa(data_env):
    _fake_mmqa(data_env)
    cfg = FederationCfg(**MMQA_CFG)
    manifest = prepare_federation(cfg)
    test = [Question.from_dict(d) for d in read_jsonl(questions_path("tiny_mmqa", "test"))]
    assert len(test) == 3 and {q.meta["family"] for q in test} == {"single", "compose"}
    assert manifest["n_calib"] == 2
    a_text = list(read_jsonl(collection_dir("tiny_mmqa", "a", "text") / "objects.jsonl"))
    b_text = list(read_jsonl(collection_dir("tiny_mmqa", "b", "text") / "objects.jsonl"))
    assert len(a_text) + len(b_text) == 12 and not {d["oid"] for d in a_text} & {d["oid"] for d in b_text}
    pooled = list(read_jsonl(collection_dir("tiny_mmqa", "home", "all") / "objects.jsonl"))
    assert {d["modality"] for d in pooled} == {"text", "table", "image"}
    again = prepare_federation(cfg)
    assert again["config_sha"] == manifest["config_sha"]
    assert list(read_jsonl(questions_path("tiny_mmqa", "test"))) == [q.to_dict() for q in test]

    embed_federation(cfg, device="cpu")
    counts = build_index(cfg, "b")
    assert counts == {"text": len(b_text), "row": 2}
    index = SiteIndex(site_dir("tiny_mmqa", "b"), cfg.qdrant, "tiny_mmqa__b")
    hits = index.search("row", load_encoder("hash").encode_queries(["Ben Piazza 1976 Bad News Bears"])[0], k=2)
    assert hits[0] == 1
    assert index.search("row", np.ones(256), k=1, offset=1) and len(index.search("row", np.ones(256), k=5)) == 2
    index.close()


def test_mramg_loader_and_prepare(data_env):
    raw = data_env / "raw" / "mramg"
    (raw / "IMAGE" / "images" / "ARXIV").mkdir(parents=True)
    (raw / "IMAGE" / "images_info").mkdir(parents=True)
    docs = [[60000 + i, f"# Paper {i}\nWe study topic {i % 2}. <PIC> The figure shows results_x0002_ok. " * 3,
             [f"img{i}_0", f"img{i}_1", f"img{i}_2"]] for i in range(10)]
    (raw / "doc_arxiv.jsonl").write_text("\n".join(json.dumps(d) for d in docs))
    qa = [{"id": f"ARXIV_{i}", "question": f"What about paper {i}?", "provenance": [60000 + i],
           "ground_truth": f"Answer {i}.<PIC>", "images_list": [f"img{i}_0"]} for i in range(10)]
    (raw / "arxiv_mqa.jsonl").write_text("\n".join(json.dumps(q) for q in qa))
    info = {f"img{i}_{j}": {"image_url": "", "image_path": f"img{i}_{j}.png", "image_caption": "" if j else "A plot"}
            for i in range(10) for j in range(3)}
    (raw / "IMAGE" / "images_info" / "arxiv_imgs_collection.json").write_text(json.dumps(info))
    for _key, v in info.items():
        (raw / "IMAGE" / "images" / "ARXIV" / v["image_path"]).write_bytes(b"png")
    loaded = mramg.load_docs("arxiv")
    assert "_x0002_" not in loaded["60000"]["content"]
    imgs = mramg.image_objects(loaded["60000"], mramg.load_images_info("arxiv"), "arxiv")
    assert imgs[0].text == "A plot" and "figure shows" in imgs[1].text
    cfg = FederationCfg(**{
        "name": "tiny_mramg", "dataset": "mramg", "subset": "arxiv", "answer_style": "long",
        "questions": {"split": "all", "n": None, "calib_doc_fraction": 0.2},
        "partition": {"method": "dirichlet", "parts": 2, "beta": 0.5, "clusters": 2},
        "qdrant": {"mode": "local"},
        "sites": [{"name": f"s{p}", "machine": "edge", "collections": [
            {"name": "text", "modality": "text", "encoder": "hash", "source": f"text:{p}"},
            {"name": "image", "modality": "image", "encoder": "hash", "source": f"image:{p}"}]} for p in range(2)],
    })
    m = prepare_federation(cfg)
    assert m["n_test"] == 8 and m["n_calib"] == 2
    q = Question.from_dict(next(read_jsonl(questions_path("tiny_mramg", "test"))))
    assert q.answers[0].startswith("Answer") and "<PIC>" not in q.answers[0]


def test_workloads_sessions_and_streams():
    qs = [Question(f"q{i}", "x", "?", ["a"], session=f"s{i % 3}") for i in range(9)]
    natural = workloads.sessions(qs)
    assert [q.session for q in natural] == ["s0"] * 3 + ["s1"] * 3 + ["s2"] * 3
    streams = workloads.streams(qs, 2)
    assert sorted(q.qid for s in streams for q in s) == sorted(q.qid for q in qs)
    assert all(len({q.session for q in s}) >= 1 for s in streams)


def test_run_questions_keep_whole_sessions(monkeypatch):
    from fedquest.config import ExperimentCfg
    from fedquest.eval import runner

    qs = [Question(f"q{i}", "x", "?", ["a"], session=f"s{i % 3}") for i in range(9)]
    monkeypatch.setattr(runner, "load_questions", lambda federation, split: qs)
    exp = ExperimentCfg(name="t", federation="toy_fed", workload="session", methods=[{"method": "vanilla_rag"}])
    assert [q.qid for q in runner.run_questions(exp, 4)] == ["q0", "q3", "q6", "q1"]
    assert len(runner.run_questions(exp)) == 9
    cold = exp.model_copy(update={"workload": "cold", "limit": 4})
    assert [q.qid for q in runner.run_questions(cold)] == ["q0", "q1", "q2", "q3"]


def test_a_family_slice_is_a_subset_of_the_stratified_run(monkeypatch):
    from fedquest.config import ExperimentCfg
    from fedquest.eval import runner

    fams = ["single", "compose", "table", "compose", "single", "compose", "single", "compose"]
    qs = [Question(f"q{i}", "x", "?", ["a"], qtype=f"T{f}", meta={"family": f}) for i, f in enumerate(fams)]
    monkeypatch.setattr(runner, "load_questions", lambda federation, split: qs)
    exp = ExperimentCfg(name="t", federation="toy_fed", limit=5, methods=[{"method": "vanilla_rag"}])
    run = [q.qid for q in runner.run_questions(exp)]
    sliced = [q.qid for q in runner.run_questions(exp, families=["compose"])]
    assert sliced and set(sliced) <= set(run) and all(fams[int(x[1:])] == "compose" for x in sliced)


def test_object_record_round_trip():
    o = ObjectRecord("o", "d", "image", "t", "x", image_path="p.jpg", meta={"k": 1})
    assert ObjectRecord.from_dict(o.to_dict()) == o


def test_independent_dirichlet_seeds_draw_apart():
    import numpy as np

    def props(seed, independent):
        return [np.random.default_rng([seed, c] if independent else seed + c).dirichlet(np.full(4, 0.5))
                for c in range(8)]

    shared = sum(np.allclose(a, b) for a in props(0, False) for b in props(1, False))
    assert shared == 7 and sum(np.allclose(a, b) for a in props(0, True) for b in props(1, True)) == 0
    ids = [f"d{i}" for i in range(60)]
    texts = [f"topic {i % 5} words {i % 7} more {i}" for i in range(60)]
    a = dirichlet_parts(ids, texts, 4, 0.5, 4, 1, independent=True)
    assert a == dirichlet_parts(list(reversed(ids)), list(reversed(texts)), 4, 0.5, 4, 1, independent=True)
    assert set(a.values()) == {0, 1, 2, 3}
