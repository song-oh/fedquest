# next splits (final_test, live4, imgbridge, dev halves): drawn once, hashed
from __future__ import annotations

import hashlib
import logging
import random
import re
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from fedquest.config import FederationCfg, load_federation
from fedquest.data.partition import page_key
from fedquest.paths import REPO, data_root, questions_path
from fedquest.types import Question
from fedquest.utils import read_jsonl, stable_int, write_jsonl

log = logging.getLogger(__name__)

NEXT_SPLITS = ("final_test", "live4", "devA", "devB", "imgbridge")
FINAL_TEST_N = {"mmqa": 400, "musique": 300, "twowiki": 300, "hotpotqa": 400}
MUSIQUE_HOPS = {"2hop": 155, "3hop": 94, "4hop": 51}
LIVE4_N = 150
IMGBRIDGE_N = 200
DEV = {"mmqa_fed": "confirm18", "musique_fed": "confirm18", "twowiki_fed": "test", "hotpot_fed": "test"}
HOMES: dict[str, dict[str, str]] = {
    "mmqa": {"final_test": "mmqa_fed", "live4": "mmqa_fed", "devA": "mmqa_fed", "devB": "mmqa_fed",
             "imgbridge": "mmqa_fed"},
    "musique": {"final_test": "musique_next", "devA": "musique_fed", "devB": "musique_fed"},
    "twowiki": {"final_test": "twowiki_fed", "devA": "twowiki_fed", "devB": "twowiki_fed"},
    "hotpotqa": {"final_test": "hotpot_fed", "devA": "hotpot_fed", "devB": "hotpot_fed"},
}
MMQA_EARLIER = ("test", "calib", "tune", "holdout", "latency", "confirm", "latency2", "confirm14", "latency3",
                "confirm18", "stream", "sessions")
NOTES = REPO / "configs" / "splits"
SHA_FILE = NOTES / "splits.sha256"
PLAN = (("mmqa_fed", "final_test"), ("mmqa_fed", "live4"), ("mmqa_fed", "imgbridge"), ("musique_next", "final_test"),
        ("twowiki_fed", "final_test"), ("hotpot_fed", "final_test"),
        *((fed, half) for fed in DEV for half in ("devA", "devB")))


def home(federation: str, split: str) -> str:
    if split not in NEXT_SPLITS:
        return federation
    try:
        cfg = load_federation(federation)
    except FileNotFoundError:
        return federation
    return HOMES.get(cfg.dataset, {}).get(split, federation)


def check_reads(cfg: FederationCfg, split: str) -> None:
    if split not in NEXT_SPLITS:
        return
    if cfg.dataset == "musique" and split == "final_test" and split not in cfg.corpus.paragraph_splits:
        raise ValueError(f"{cfg.name}'s corpus lacks the final_test questions' paragraphs "
                         "(corpus.paragraphs_of): run final_test on musique_next or one of its layouts")
    for s in cfg.sites:
        for c in s.collections:
            if c.fraction is not None and split not in [k.strip() for k in (c.keep or "").split(",")]:
                raise ValueError(f"{cfg.name}/{s.name}/{c.name} keeps a fraction of the pages and not "
                                 f"{split}'s ground truth (keep: {c.keep}): add {split} to keep and prepare "
                                 "it again")


def parent_of(federation: str) -> str | None:
    try:
        cfg = load_federation(federation)
    except FileNotFoundError:
        return None
    for site in cfg.sites:
        if site.data_source:
            return site.data_source[0]
        for c in site.collections:
            if c.parent and c.shard_of is not None:
                return c.parent[0]
    return None


def split_path(federation: str, split: str) -> Path:
    if split in NEXT_SPLITS:
        try:
            cfg = load_federation(federation)
        except FileNotFoundError:
            return questions_path(federation, split)
        check_reads(cfg, split)
    path = questions_path(home(federation, split), split)
    if not path.exists() and split not in NEXT_SPLITS:
        parent = parent_of(federation)
        if parent is not None and parent != federation and questions_path(parent, split).exists():
            return questions_path(parent, split)
    return path


def _read(federation: str, split: str) -> list[Question]:
    from fedquest.eval.runner import load_questions

    return load_questions(federation, split)


def existing(federation: str, split: str) -> list[Question] | None:
    path = questions_path(home(federation, split), split)
    return [Question.from_dict(d) for d in read_jsonl(path)] if path.exists() else None


def _like(pool: list[Question], reference: list[Question], n: int, salt: str, seed: int) -> list[Question]:
    from fedquest.data.prepare import _like as like

    return like(pool, reference, n, random.Random(f"{seed}:{salt}"))


def _mmqa_held(fed: str) -> tuple[set[str], set[str]]:
    held = {q.qid for s in MMQA_EARLIER for q in _read(fed, s)}
    tables = {q.meta.get("table_id", "") for q in _read(fed, "calib")} - {""}
    return held, tables


def _mmqa_pool(held: set[str], tables: set[str]) -> list[Question]:
    from fedquest.data import mmqa

    return [q for q in mmqa.load_questions("train")
            if q.qid not in held and q.meta.get("table_id", "") not in tables]


def mmqa_final_test(fed: str = "mmqa_fed") -> list[Question]:
    seed, n = load_federation(fed).questions.seed, FINAL_TEST_N["mmqa"]
    return _like(_mmqa_pool(*_mmqa_held(fed)), _read(fed, "confirm18"), n, "final_test", seed)


def mmqa_live4(fed: str = "mmqa_fed", final_test: list[Question] | None = None) -> list[Question]:
    held, tables = _mmqa_held(fed)
    final = final_test if final_test is not None else existing(fed, "final_test") or mmqa_final_test(fed)
    pool = _mmqa_pool(held | {q.qid for q in final}, tables)
    return _like(pool, _read(fed, "confirm18"), LIVE4_N, "live4", load_federation(fed).questions.seed)


def _recorded(dataset: str, skip: tuple[str, str]) -> set[str]:
    out: set[str] = set()
    for sums in sorted(NOTES.glob("*.sha256")):
        for line in sums.read_text(encoding="utf-8").splitlines():
            m = re.fullmatch(r"data/fed/([\w.-]+)/questions/([\w.-]+)\.jsonl", line.strip().partition("  ")[2])
            if not m or (m[1], m[2]) == skip:
                continue
            try:
                if load_federation(m[1]).dataset != dataset:
                    continue
            except FileNotFoundError:
                continue
            if m[2] in NEXT_SPLITS:
                qs = existing(m[1], m[2])
                out |= {q.qid for q in (draw(m[1], m[2]) if qs is None else qs)}
            else:
                out |= {q.qid for q in _read(m[1], m[2])}
    return out


def mmqa_imgbridge(fed: str = "mmqa_fed") -> list[Question]:
    from fedquest.data.mmqa import subsets

    def bridge(q: Question) -> bool:
        return "bridge_picture" in subsets(q.qtype)

    held, tables = _mmqa_held(fed)
    final = existing(fed, "final_test") or mmqa_final_test(fed)
    live = existing(fed, "live4") or mmqa_live4(fed, final)
    held |= {q.qid for q in [*final, *live]} | _recorded("mmqa", (fed, "imgbridge"))
    reference = [q for q in _read(fed, "confirm18") if bridge(q)]
    if not reference:
        raise ValueError(f"{fed}'s confirm18 holds no image bridge: imgbridge has no type proportions to follow")
    pool = [q for q in _mmqa_pool(held, tables) if bridge(q)]
    return _like(pool, reference, IMGBRIDGE_N, "imgbridge", load_federation(fed).questions.seed)


def musique_final_test(cfg: FederationCfg, questions: list[Question] | None = None,
                       calib: list[Question] | None = None,
                       calib_paragraphs: list[tuple[str, str]] | None = None
                       ) -> tuple[list[Question], list[tuple[str, str]]]:
    from fedquest.data import musique
    from fedquest.data.prepare import _like as like

    qc = cfg.questions
    if questions is None:
        questions = musique.load_questions(qc.split)
    if calib is None:
        train = musique.load_questions(qc.calib_split)
        calib = like(train, questions, qc.calib_n, random.Random(f"{qc.seed}:calib"))
    calib_ids = {q.qid for q in calib}
    if calib_paragraphs is None:
        calib_paragraphs = musique.paragraphs(qc.calib_split, calib_ids)
    held = {q.qid for q in questions} | calib_ids
    hops: set[int] = set()
    for d in musique.iter_raw(qc.split):
        hops |= musique.single_hops(d)
    titles = {page_key(t) for t, _ in musique.passages(calib_paragraphs)[0]}
    summary: dict[str, tuple[frozenset[int], bool, bool]] = {}
    for d in musique.iter_raw(qc.calib_split):
        support = {page_key(p["title"]) for p in d["paragraphs"] if p.get("is_supporting")}
        summary[d["id"]] = (musique.single_hops(d), bool(d.get("answerable", True)), not support & titles)
        if d["id"] in calib_ids:
            hops |= musique.single_hops(d)
    # unseen train questions: answerable, no single hop shared with eval or calib, no page in the corpus
    pool = sorted(q for q, (h, ok, own) in summary.items() if q not in held and ok and own and not h & hops)
    rng = random.Random(f"{qc.seed}:final_test")
    chosen: list[str] = []
    for fam, n in MUSIQUE_HOPS.items():
        members = [q for q in pool if musique.family(q.split("__", 1)[0]) == fam]
        if len(members) < n:
            raise ValueError(f"MuSiQue train holds {len(members)} {fam} questions for final_test, not {n}")
        chosen += rng.sample(members, n)
    chosen.sort()
    rng.shuffle(chosen)
    order = {q: i for i, q in enumerate(chosen)}
    raw: dict[str, Any] = {}
    paragraphs: list[tuple[str, str]] = []
    for d in musique.iter_raw(qc.calib_split):
        if d["id"] in order:
            raw[d["id"]] = d
            paragraphs += [(p["title"], p["paragraph_text"]) for p in d["paragraphs"]]
    return [musique.to_question(raw[q]) for q in chosen], paragraphs


def textqa_final_test(fed: str) -> list[Question]:
    from fedquest.data import hipporag

    cfg = load_federation(fed)
    test, calib = _read(fed, "test"), _read(fed, "calib")
    held = {q.qid for q in test} | {q.qid for q in calib}
    support = hipporag.supporting_titles(cfg.dataset)
    dev_titles = {page_key(t) for q in test for t in support[q.qid]}
    pool = [q for q in hipporag.load_questions(cfg.dataset)
            if q.qid not in held and not {page_key(t) for t in support[q.qid]} & dev_titles]
    return _like(pool, test, FINAL_TEST_N[cfg.dataset], "final_test", cfg.questions.seed)


def dev_half(qid: str) -> str:
    return "devA" if stable_int("next:dev", qid) % 2 == 0 else "devB"


def draw(fed: str, split: str) -> list[Question]:
    if split in ("devA", "devB"):
        if fed not in DEV:
            raise ValueError(f"{fed} has no dev split (those of {', '.join(DEV)})")
        return [q for q in _read(fed, DEV[fed]) if dev_half(q.qid) == split]
    if (fed, split) == ("mmqa_fed", "final_test"):
        return mmqa_final_test(fed)
    if (fed, split) == ("mmqa_fed", "live4"):
        return mmqa_live4(fed)
    if (fed, split) == ("mmqa_fed", "imgbridge"):
        return mmqa_imgbridge(fed)
    if (fed, split) == ("musique_next", "final_test"):
        return musique_final_test(load_federation(fed))[0]
    if split == "final_test" and fed in ("twowiki_fed", "hotpot_fed"):
        return textqa_final_test(fed)
    raise ValueError(f"{fed}/{split} is not a next split (PLAN: {', '.join(f'{f}/{s}' for f, s in PLAN)})")


def _gold_found(questions: Iterable[Question], present: Callable[[str], bool]) -> dict[str, int]:
    qs = list(questions)
    missing = [d for q in qs for d in q.gold_doc_ids if not present(d)]
    complete = sum(all(present(d) for d in q.gold_doc_ids) and bool(q.gold_doc_ids) for q in qs)
    return {"questions": len(qs), "complete": complete, "gold_docs": sum(len(q.gold_doc_ids) for q in qs),
            "missing_docs": len(missing)}


def ground_truth(fed: str, questions: list[Question]) -> dict[str, int]:
    cfg = load_federation(fed)
    if cfg.dataset == "mmqa":
        from fedquest.data import mmqa
        from fedquest.data.unreadable import MMQA_UNREADABLE

        ids = {r["id"] for it in (mmqa.iter_texts(), mmqa.iter_images(), mmqa.iter_tables()) for r in it}
        ids -= MMQA_UNREADABLE
        def docs(q: Question) -> list[str]:
            table = [q.meta["table_id"]] if "Table" in q.qtype and q.meta.get("table_id") else []
            return list(dict.fromkeys([*q.gold_doc_ids, *table]))

        with_tables = [Question(q.qid, q.dataset, q.question, q.answers, q.qtype, docs(q)) for q in questions]
        return _gold_found(with_tables, ids.__contains__)
    if cfg.dataset == "musique":
        from fedquest.data import musique
        from fedquest.data.prepare import musique_corpus

        ids = {musique.doc_id(t, x) for t, x in musique_corpus(cfg)[0]}
        return _gold_found(questions, ids.__contains__)
    from fedquest.data import hipporag

    ids = {hipporag.doc_id(t, x) for t, x in hipporag.load_corpus(cfg.dataset)}
    return _gold_found(questions, ids.__contains__)


def sha_line(fed: str, split: str) -> str:
    path = questions_path(fed, split)
    return f"{hashlib.sha256(path.read_bytes()).hexdigest()}  data/fed/{fed}/questions/{split}.jsonl"


def write_splits(plan: Iterable[tuple[str, str]] | None = None,
                 sha_file: Path | None = None) -> list[dict[str, Any]]:
    plan = list(PLAN if plan is None else plan)
    sha_file = SHA_FILE if sha_file is None else sha_file
    out = []
    for fed, split in plan:
        path = questions_path(fed, split)
        written = not path.exists()
        if written:
            qs = draw(fed, split)
            tmp = path.with_name(f"{path.name}.draw")
            write_jsonl(tmp, (q.to_dict() for q in qs))
            tmp.replace(path)
            log.info("%s/%s: %d questions drawn", fed, split, len(qs))
        out.append({"federation": fed, "split": split, "questions": sum(1 for _ in read_jsonl(path)),
                    "written": written})
    sha_file.parent.mkdir(parents=True, exist_ok=True)
    sha_file.write_text("".join(sha_line(f, s) + "\n" for f, s in plan), encoding="utf-8")
    return out


def check_splits(sha_file: Path | None = None) -> list[str]:
    sha_file = SHA_FILE if sha_file is None else sha_file
    if not sha_file.exists():
        return [f"{sha_file} is missing: fedquest data splits writes it"]
    bad = []
    for line in sha_file.read_text(encoding="utf-8").splitlines():
        digest, _, rel = line.strip().partition("  ")
        if not rel:
            continue
        path = data_root() / rel.removeprefix("data/")
        if not path.exists():
            bad.append(f"{rel}: missing (fedquest data splits draws it; or copy the recorded file)")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            bad.append(f"{rel}: not the file {sha_file} records")
    return bad
