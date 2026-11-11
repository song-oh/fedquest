from __future__ import annotations

import functools
import logging
import random
from collections import defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from fedquest.config import CollectionCfg, FederationCfg, QuestionsCfg, check_data_from, load_federation
from fedquest.data import distractors, hipporag, m3sciqa, mmqa, mramg, msmarco, musique, splits
from fedquest.data.partition import (
    FRACTION_SCALE,
    dirichlet_parts,
    facility_rank,
    fraction_rank,
    page_key,
    part_by_title,
    shard_by_title,
)
from fedquest.paths import collection_dir, fed_dir, questions_path
from fedquest.types import ObjectRecord, Question
from fedquest.utils import stable_hash, write_json, write_jsonl

log = logging.getLogger(__name__)


def _quota_sample(pool: list[Question], quotas: dict[str, int], rng: random.Random) -> list[Question]:
    out: list[Question] = []
    for fam, n in quotas.items():
        members = [q for q in pool if q.meta.get("family") == fam]
        if len(members) < n:
            log.warning("family %s has %d questions, fewer than its quota %d", fam, len(members), n)
        out += rng.sample(members, min(n, len(members)))
    return out


def _stratified(pool: list[Question], n: int, rng: random.Random) -> list[Question]:
    by_type: dict[str, list[Question]] = defaultdict(list)
    for q in pool:
        by_type[q.qtype].append(q)
    total = len(pool)
    shares = {t: n * len(qs) / total for t, qs in by_type.items()}
    counts = {t: int(s) for t, s in shares.items()}
    for t in sorted(shares, key=lambda t: counts[t] - shares[t])[: n - sum(counts.values())]:
        counts[t] += 1
    return [q for t in sorted(by_type) for q in rng.sample(by_type[t], min(counts[t], len(by_type[t])))]


def select_questions(pool: list[Question], qcfg: QuestionsCfg, n: int | None, quotas: dict[str, int] | None,
                     salt: str) -> list[Question]:
    rng = random.Random(f"{qcfg.seed}:{salt}")
    pool = sorted(pool, key=lambda q: q.qid)
    if qcfg.subset_file:
        keep = set(Path(qcfg.subset_file).read_text().split())
        pool = [q for q in pool if q.qid in keep]
    if qcfg.types:
        pool = [q for q in pool if any(q.qtype.startswith(t) for t in qcfg.types)]
    if quotas:
        out = _quota_sample(pool, quotas, rng)
    elif n is None or n >= len(pool):
        out = list(pool)
    elif qcfg.stratify:
        out = _stratified(pool, n, rng)
    else:
        out = rng.sample(pool, n)
    out.sort(key=lambda q: q.qid)
    rng.shuffle(out)
    return out


def _scaled(quotas: dict[str, int], n: int) -> dict[str, int]:
    total = sum(quotas.values()) or 1
    return {fam: max(1, round(n * v / total)) for fam, v in quotas.items()}


Shard = tuple[int, int] | None


def _pages(parts: int, seed: int, shard_seed: int | None = None) -> Callable[[str, int | None, Shard], bool]:
    salt = seed if shard_seed is None else shard_seed

    @functools.cache
    def part_of(title: str) -> int:
        return part_by_title(title, parts, seed)

    @functools.cache
    def shard_of(title: str, m: int) -> int:
        return shard_by_title(title, m, salt)

    def keep(title: str, part: int | None, shard: Shard) -> bool:
        return (part is None or part_of(title) == part) and (shard is None or shard_of(title, shard[1]) == shard[0])

    return keep


def _text_pages(cfg: FederationCfg, passages: list[tuple[str, str]]) -> Callable[[str, int | None, Shard], bool]:
    p = cfg.partition
    if p.method != "dirichlet":
        return _pages(p.parts, p.seed, p.shard_hash_seed)
    pages: dict[str, list[str]] = defaultdict(list)
    for title, text in passages:
        pages[title].append(text)
    titles = sorted(pages)
    part_of = dirichlet_parts(titles, [" ".join([t, *pages[t]])[:4000] for t in titles], p.parts, p.beta,
                              p.clusters, p.seed)

    @functools.cache
    def shard_of(title: str, m: int) -> int:
        return shard_by_title(title, m, p.shard_hash_seed)

    def keep(title: str, part: int | None, shard: Shard) -> bool:
        return (part is None or part_of.get(title) == part) and (shard is None or shard_of(title, shard[1]) == shard[0])

    return keep


Sized = Callable[[str], bool]


def _share_test(share: tuple[int, int], seed: int) -> Sized:
    lo, hi = share

    @functools.cache
    def test(title: str) -> bool:
        return lo <= facility_rank(title, seed) < hi

    return test


def _page_test(cfg: FederationCfg, coll: CollectionCfg,
               sized: Callable[[CollectionCfg], Sized | None] | None) -> Sized | None:
    tests = [t for t in (sized(coll) if sized is not None else None,
                         _share_test(coll.share_of, cfg.partition.share_hash_seed) if coll.share_of else None)
             if t is not None]
    if len(tests) < 2:
        return tests[0] if tests else None
    first, second = tests
    return lambda title: first(title) and second(title)


def _write_collections(cfg: FederationCfg, sites: Iterable[str], build: Any,
                       sized: Callable[[CollectionCfg], Sized | None] | None = None
                       ) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for site in cfg.sites:
        if site.name not in sites or site.data_from:
            continue
        counts[site.name] = {}
        for coll in site.collections:
            objs: dict[str, ObjectRecord] = {}
            keep = _page_test(cfg, coll, sized)
            for kind, part in coll.sources:
                args = (kind, part, coll.shard_of) if keep is None else (kind, part, coll.shard_of, keep)
                for obj in build(*args):
                    objs.setdefault(obj.oid, obj)
            path = collection_dir(cfg.name, site.name, coll.name) / "objects.jsonl"
            counts[site.name][coll.name] = write_jsonl(path, (o.to_dict() for o in objs.values()))
            log.info("%s/%s/%s: %d objects", cfg.name, site.name, coll.name, counts[site.name][coll.name])
    return counts


def _prepare_mmqa(cfg: FederationCfg, sites: set[str]) -> dict[str, Any]:
    qc = cfg.questions
    test = select_questions(mmqa.load_questions(qc.split), qc, qc.n, qc.quotas, "test")
    calib_quotas = _scaled(qc.quotas, qc.calib_n) if qc.quotas else None
    calib = select_questions(mmqa.load_questions(qc.calib_split), qc, qc.calib_n, calib_quotas, "calib")
    if not sites:
        return {"test": test, "calib": calib, "counts": {}}

    texts = list(mmqa.iter_texts())
    tables = list(mmqa.iter_tables())
    images = list(mmqa.iter_images())
    if cfg.corpus.scope == "pooled":
        rng = random.Random(f"{qc.seed}:pool")
        wanted_t = {d for q in test + calib for d in q.meta["text_doc_ids"]}
        wanted_i = {d for q in test + calib for d in q.meta["image_doc_ids"]}
        wanted_tab = {q.meta["table_id"] for q in test + calib}
        wanted_t |= {r["id"] for r in rng.sample(texts, min(cfg.corpus.extra_texts, len(texts)))}
        wanted_i |= {r["id"] for r in rng.sample(images, min(cfg.corpus.extra_images, len(images)))}
        wanted_tab |= {r["id"] for r in rng.sample(tables, min(cfg.corpus.extra_tables, len(tables)))}
        texts = [r for r in texts if r["id"] in wanted_t]
        images = [r for r in images if r["id"] in wanted_i]
        tables = [r for r in tables if r["id"] in wanted_tab]
    if cfg.partition.method not in ("by_title", "single"):
        raise ValueError(f"{cfg.name}: MMQA pages are partitioned by title (by_title, or single for one site), not "
                         f"{cfg.partition.method!r}")
    keep = _pages(cfg.partition.parts, cfg.partition.seed, cfg.partition.shard_hash_seed)
    captions = mmqa.load_captions()

    def build(kind: str, part: int | None, shard: Shard = None,
              sized: Sized | None = None) -> Iterable[ObjectRecord]:
        chunks = cfg.corpus.chunk_tokens, cfg.corpus.chunk_overlap

        def ok(title: str) -> bool:
            return keep(title, part, shard) and (sized is None or sized(title))

        if kind == "text":
            return (o for r in texts if ok(r["title"]) for o in mmqa.text_objects(r, *chunks))
        if kind == "table":
            return (o for t in tables if ok(t["title"]) for o in mmqa.row_objects(t))
        if kind == "image":
            return (mmqa.image_object(r, captions) for r in images if ok(r["title"]))
        raise ValueError(f"unknown source kind {kind!r}")

    titles = {r["id"]: r["title"] for rows in (texts, tables, images) for r in rows}
    gold: dict[str, frozenset[str]] = {}

    def sized(coll: CollectionCfg) -> Sized | None:
        if coll.fraction is None:
            return None
        if coll.keep not in gold:
            kept = [q for name in (coll.keep or "").split(",") if name.strip()
                    for q in _kept_questions(cfg, name.strip(), {"test": test, "calib": calib})]
            gold[coll.keep or ""] = frozenset(_mmqa_gold_pages(kept, titles))
        return _fraction_test(coll.fraction, cfg.partition.seed, gold[coll.keep or ""])

    counts = _write_collections(cfg, sites, build, sized)
    kept = {"manifest": {"ground_truth_pages": {k: len(v) for k, v in sorted(gold.items())}}} if gold else {}
    return {"test": test, "calib": calib, "counts": counts, **kept}


def _kept_questions(cfg: FederationCfg, split: str, own: dict[str, list[Question]]) -> list[Question]:
    if split in own:
        return own[split]
    if split not in splits.NEXT_SPLITS:
        raise ValueError(f"{cfg.name}: keep names {split!r}; a data-size layout keeps the ground truth of "
                         f"test, calib or a next split ({', '.join(splits.NEXT_SPLITS)})")
    found = splits.existing(cfg.name, split)
    if found is None:
        log.warning("%s: %s is not drawn yet (`fedquest data splits`): its ground-truth pages are the draw's",
                    cfg.name, split)
        found = splits.draw(splits.home(cfg.name, split), split)
    return found


def _mmqa_gold_pages(questions: Iterable[Question], titles: dict[str, str]) -> set[str]:
    out = set()
    for q in questions:
        table = [q.meta["table_id"]] if "Table" in q.qtype and q.meta.get("table_id") else []
        out |= {page_key(titles[d]) for d in [*q.gold_doc_ids, *table] if d in titles}
    return out


def _fraction_test(fraction: float, seed: int, kept: frozenset[str]) -> Sized:
    cut = round(FRACTION_SCALE * fraction)

    @functools.cache
    def test(title: str) -> bool:
        return fraction_rank(title, seed) < cut or page_key(title) in kept

    return test


def _prepare_mramg(cfg: FederationCfg, sites: set[str]) -> dict[str, Any]:
    subset = cfg.subset or ""
    qc = cfg.questions
    docs = mramg.load_docs(subset)
    info = mramg.load_images_info(subset)
    doc_ids = sorted(docs)
    held = set(random.Random(f"{qc.seed}:calib-docs").sample(doc_ids, round(qc.calib_doc_fraction * len(doc_ids))))
    questions = mramg.load_questions(subset)
    calib = [q for q in questions if q.gold_doc_ids and all(d in held for d in q.gold_doc_ids)]
    calib_ids = {q.qid for q in calib}
    test = select_questions([q for q in questions if q.qid not in calib_ids], qc, qc.n, None, "test")
    if cfg.partition.method == "dirichlet":
        p = cfg.partition
        part_of = dirichlet_parts(doc_ids, [docs[d]["content"] for d in doc_ids], p.parts, p.beta, p.clusters, p.seed)
    else:
        part_of = {d: 0 for d in doc_ids}

    def build(kind: str, part: int | None, shard: Shard = None) -> Iterable[ObjectRecord]:
        chosen = [d for d in doc_ids if (part is None or part_of[d] == part)
                  and (shard is None or shard_by_title(str(d), shard[1], cfg.partition.shard_hash_seed) == shard[0])]
        if kind == "text":
            return (o for d in chosen for o in mramg.doc_objects(docs[d], subset, cfg.corpus.chunk_tokens, cfg.corpus.chunk_overlap))
        if kind == "image":
            return (o for d in chosen for o in mramg.image_objects(docs[d], info, subset))
        raise ValueError(f"MRAMG has no {kind!r} objects")

    counts = _write_collections(cfg, sites, build)
    return {"test": test, "calib": calib, "counts": counts, "partition": {str(k): v for k, v in part_of.items()}}


def _like(pool: list[Question], reference: list[Question], n: int, rng: random.Random) -> list[Question]:
    types = _count(q.qtype for q in reference)
    share = {t: n * c / len(reference) for t, c in types.items()}
    quota = {t: int(s) for t, s in share.items()}
    for t in sorted(share, key=lambda t: (quota[t] - share[t], t))[: n - sum(quota.values())]:
        quota[t] += 1
    by_type: dict[str, list[Question]] = defaultdict(list)
    for q in sorted(pool, key=lambda q: q.qid):
        by_type[q.qtype].append(q)
    out = [q for t in sorted(quota) for q in rng.sample(by_type[t], min(quota[t], len(by_type[t])))]
    out.sort(key=lambda q: q.qid)
    rng.shuffle(out)
    return out


def musique_splits(cfg: FederationCfg, questions: list[Question] | None = None) -> dict[str, list[Question]]:
    qc = cfg.questions
    pool = musique.load_questions(qc.split) if questions is None else questions
    confirm = select_questions(pool, qc, qc.n, None, "confirm")
    held = {q.qid for q in confirm}
    tune = select_questions([q for q in pool if q.qid not in held], qc, musique.TUNE_N, None, "tune")
    held |= {q.qid for q in tune}
    confirm14 = select_questions([q for q in pool if q.qid not in held], qc, qc.n, None, "confirm14")
    held |= {q.qid for q in confirm14}
    confirm18 = select_questions([q for q in pool if q.qid not in held], qc, qc.n, None, "confirm18")
    return {"confirm": confirm, "tune": tune, "confirm14": confirm14, "confirm18": confirm18}


def musique_corpus(cfg: FederationCfg, questions: list[Question] | None = None,
                   calib: list[Question] | None = None) -> tuple[list[tuple[str, str]], dict[str, int]]:
    qc = cfg.questions
    if questions is None:
        questions = musique.load_questions(qc.split)
    if calib is None:
        train = musique.load_questions(qc.calib_split)
        calib = _like(train, questions, qc.calib_n, random.Random(f"{qc.seed}:calib"))
    calib_paragraphs = musique.paragraphs(qc.calib_split, {q.qid for q in calib})
    found, published, added = musique.passages(calib_paragraphs)
    sizes = {"published": published, "calibration_added": added}
    if "final_test" in cfg.corpus.paragraph_splits:
        final, paragraphs = splits.musique_final_test(cfg, questions, calib, calib_paragraphs)
        written = splits.existing(cfg.name, "final_test")
        if written is not None and [q.qid for q in written] != [q.qid for q in final]:
            raise ValueError(f"{cfg.name}: {splits.split_path(cfg.name, 'final_test')} is not the final_test "
                             "draw (fedquest.data.splits), so its questions' paragraphs would not be in this "
                             "corpus")
        before = len(found)
        found = list(dict.fromkeys([*found, *paragraphs]))
        sizes["final_test_added"] = len(found) - before
    return found, sizes


def _prepare_musique(cfg: FederationCfg, sites: set[str]) -> dict[str, Any]:
    qc = cfg.questions
    questions = musique.load_questions(qc.split)
    calib = _like(musique.load_questions(qc.calib_split), questions, qc.calib_n, random.Random(f"{qc.seed}:calib"))
    found, sizes = musique_corpus(cfg, questions, calib)
    extra = distractors.take("musique", cfg.corpus.distractors, {t for t, _ in found})
    found = found + extra
    keep = _text_pages(cfg, found)

    def build(kind: str, part: int | None, shard: Shard = None,
              sized: Sized | None = None) -> Iterable[ObjectRecord]:
        if kind != "text":
            raise ValueError(f"MuSiQue has no {kind!r} objects (text only)")
        return (o for title, text in found if keep(title, part, shard) and (sized is None or sized(title))
                for o in musique.passage_objects(title, text, cfg.corpus.chunk_tokens, cfg.corpus.chunk_overlap))

    counts = _write_collections(cfg, sites, build)
    splits = musique_splits(cfg, questions)
    return {"test": questions, "calib": calib, "splits": splits, "counts": counts,
            "manifest": {"splits": {s: len(qs) for s, qs in splits.items()},
                         "corpus": {"passages": len(found), **sizes,
                                    **({"distractors": len(extra)} if extra else {})}}}


def _prepare_textqa(cfg: FederationCfg, sites: set[str]) -> dict[str, Any]:
    qc = cfg.questions
    if cfg.dataset == "msmarco":
        questions = msmarco.load_questions("validation")
        calib = _like(msmarco.load_questions("train"), questions, qc.calib_n, random.Random(f"{qc.seed}:calib"))
    else:
        questions = hipporag.load_questions(cfg.dataset)
        calib = _like(questions, questions, qc.calib_n, random.Random(f"{qc.seed}:calib"))
    held = {q.qid for q in calib}
    test = select_questions([q for q in questions if q.qid not in held], qc, qc.n, None, "test")
    if not sites:
        return {"test": test, "calib": calib, "counts": {}}
    found = list(dict.fromkeys(msmarco.load_corpus() if cfg.dataset == "msmarco"
                               else hipporag.load_corpus(cfg.dataset)))
    published = len(found)
    extra = distractors.take(cfg.dataset, cfg.corpus.distractors, {t for t, _ in found})
    found += extra
    keep = _text_pages(cfg, found)

    def build(kind: str, part: int | None, shard: Shard = None) -> Iterable[ObjectRecord]:
        if kind != "text":
            raise ValueError(f"{cfg.dataset} has no {kind!r} objects (text only)")
        if cfg.dataset == "msmarco":
            return (o for title, text in found if keep(title, part, shard)
                    for o in msmarco.passage_objects(title, text))
        return (o for title, text in found if keep(title, part, shard)
                for o in hipporag.passage_objects(cfg.dataset, title, text, cfg.corpus.chunk_tokens,
                                                  cfg.corpus.chunk_overlap))

    counts = _write_collections(cfg, sites, build)
    rules = _count(q.meta.get("hop_rule", "") for q in questions if q.meta.get("hop_rule"))
    return {"test": test, "calib": calib, "counts": counts,
            "manifest": {"corpus": {"passages": len(found), "published": published,
                                    **({"distractors": len(extra)} if extra else {})},
                         **({"hop_rules": dict(sorted(rules.items()))} if rules else {})}}


def _prepare_m3sciqa(cfg: FederationCfg, sites: set[str]) -> dict[str, Any]:
    qc = cfg.questions
    chunks = cfg.corpus.chunk_tokens, cfg.corpus.chunk_overlap
    questions = m3sciqa.load_questions("validation", *chunks)
    located = [q for q in questions if q.meta["evidence_level"] in m3sciqa.LOCATED]
    calib = _like(located, questions, qc.calib_n, random.Random(f"{qc.seed}:calib"))
    held = {q.qid for q in calib}
    test = select_questions([q for q in questions if q.qid not in held], qc, qc.n, None, "test")

    def levels(qs: list[Question]) -> dict[str, int]:
        found = _count(q.meta["evidence_level"] for q in qs)
        return {lv: found.get(lv, 0) for lv in m3sciqa.LEVELS}

    evidence = {"all": levels(questions), "test": levels(test), "calib": levels(calib)}
    if not sites:
        return {"test": test, "calib": calib, "counts": {}, "manifest": {"evidence": evidence}}
    if cfg.partition.method not in ("by_title", "single"):
        raise ValueError(f"{cfg.name}: M3SciQA's papers are partitioned by title (by_title, or single for one site), "
                         f"not {cfg.partition.method!r}")
    keep = _pages(cfg.partition.parts, cfg.partition.seed, cfg.partition.shard_hash_seed)
    papers = m3sciqa.corpus(*chunks)
    images = m3sciqa.image_objects()

    def build(kind: str, part: int | None, shard: Shard = None) -> Iterable[ObjectRecord]:
        if kind == "text":
            return (o for ps in papers.values() if ps and keep(ps[0].title, part, shard) for o in ps)
        if kind == "image":
            return (o for o in images if keep(o.title, part, shard))
        raise ValueError(f"M3SciQA has no {kind!r} objects (text and image)")

    counts = _write_collections(cfg, sites, build)
    corpus = {"papers": sum(1 for ps in papers.values() if ps), "passages": sum(len(ps) for ps in papers.values()),
              "images": len(images)}
    return {"test": test, "calib": calib, "counts": counts, "manifest": {"corpus": corpus, "evidence": evidence}}


def _check_variant(cfg: FederationCfg, parent: FederationCfg) -> None:
    def shape(f: FederationCfg) -> list[Any]:
        return [(s.name, s.machine, s.domain, s.data_from, [c.model_dump() for c in s.collections]) for s in f.sites]

    if parent.data_from:
        raise ValueError(f"{cfg.name}: data_from {parent.name!r} is itself a variant (of {parent.data_from!r})")
    if (cfg.dataset, shape(cfg)) != (parent.dataset, shape(parent)):
        raise ValueError(f"{cfg.name}: a data_from variant needs {parent.name}'s sites and collections (it reads their "
                         "data); only the sites' policies may differ")


def prepare_federation(cfg: FederationCfg, sites: list[str] | None = None) -> dict[str, Any]:
    chosen = set(sites or cfg.site_names)
    unknown = chosen - set(cfg.site_names)
    if unknown:
        raise KeyError(f"{cfg.name} has no sites {sorted(unknown)}")
    check_data_from(cfg)
    chosen = set(cfg.own_sites(sites, "prepare"))
    if cfg.data_from:
        _check_variant(cfg, load_federation(cfg.data_from))
        chosen = set()
    sized = [f"{s.name}/{c.name}" for s in cfg.sites if s.name in chosen for c in s.collections
             if c.fraction is not None or c.keep is not None]
    if sized and cfg.dataset != "mmqa":
        raise NotImplementedError(f"{cfg.name}: collections {sized} keep a fraction of the pages (fraction, "
                                  "keep): track D builds MMQA's data-size layouts only")
    shared = [f"{s.name}/{c.name}" for s in cfg.sites if s.name in chosen for c in s.collections
              if c.share is not None]
    if shared and cfg.dataset not in ("mmqa", "musique"):
        raise NotImplementedError(f"{cfg.name}: collections {shared} keep a share of the pages (share): "
                                  "track D3 builds MMQA's and MuSiQue's data-share layouts only")
    prepare = {"mmqa": _prepare_mmqa, "musique": _prepare_musique, "mramg": _prepare_mramg,
               "hotpotqa": _prepare_textqa, "twowiki": _prepare_textqa, "msmarco": _prepare_textqa,
               "m3sciqa": _prepare_m3sciqa}.get(cfg.dataset)
    if prepare is None:
        raise ValueError(f"{cfg.name}: no prepare step for dataset {cfg.dataset!r}")
    out = prepare(cfg, chosen)
    write_jsonl(questions_path(cfg.name, "test"), (q.to_dict() for q in out["test"]))
    write_jsonl(questions_path(cfg.name, "calib"), (q.to_dict() for q in out["calib"]))
    for split, qs in out.get("splits", {}).items():
        write_jsonl(questions_path(cfg.name, split), (q.to_dict() for q in qs))
    manifest = {
        "federation": cfg.name,
        "dataset": cfg.dataset,
        "subset": cfg.subset,
        "config_sha": stable_hash(cfg.model_dump(mode="json")),
        "n_test": len(out["test"]),
        "n_calib": len(out["calib"]),
        "test_families": dict(sorted(_count(q.meta.get("family", "") for q in out["test"]).items())),
        "objects": out["counts"],
        **out.get("manifest", {}),
    }
    path = fed_dir(cfg.name) / "manifest.json"
    previous: dict[str, Any] = {}
    if path.exists():
        import json

        previous = json.loads(path.read_text())
    if previous.get("config_sha") == manifest["config_sha"]:
        manifest["objects"] = {**previous.get("objects", {}), **manifest["objects"]}
    write_json(path, manifest)
    return manifest


def _count(items: Iterable[str]) -> dict[str, int]:
    out: dict[str, int] = defaultdict(int)
    for x in items:
        out[x] += 1
    return dict(out)
