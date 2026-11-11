#!/usr/bin/env python

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fedquest.config import load_federation  # noqa: E402
from fedquest.paths import data_root  # noqa: E402
from fedquest.utils import read_jsonl, write_jsonl  # noqa: E402

SPLIT = "full"
HELD = "heldout"
HASHES = ROOT / "configs" / "pools.sha256"
EXPOSED = ROOT / "configs" / "exposed.json"
MUSIQUE_HOME = "musique_next"
POOLS = {"mmqa_fed": ("mmqa", "test", 2441), "twowiki_fed": ("twowiki", "final_test", 1000),
         "hotpot_fed": ("hotpotqa", "final_test", 1000)}
COPIES = {"mmqa_fed": (), "twowiki_fed": ("twowiki_home",), "hotpot_fed": ("hotpot_home",)}


def _questions(dataset: str) -> list[dict]:
    if dataset == "mmqa":
        from fedquest.data import mmqa

        return [q.to_dict() for q in mmqa.load_questions("dev")]
    from fedquest.data import hipporag

    return [q.to_dict() for q in hipporag.load_questions(dataset)]


def pool(home: str, root: Path) -> list[dict]:
    dataset, head_split, n = POOLS[home]
    every = _questions(dataset)
    by_id = {q["qid"]: q for q in every}
    if len(by_id) != len(every) or len(every) != n:
        raise SystemExit(f"{home}: the raw {dataset} dev set has {len(every)} questions ({len(by_id)} ids), not {n}")
    head = list(read_jsonl(root / "fed" / home / "questions" / f"{head_split}.jsonl"))
    for q in head:
        if by_id.get(q["qid"]) != q:
            raise SystemExit(f"{home}: {head_split}'s {q['qid']} is not the converter's record: the raw files or the "
                             "converter changed since the splits were drawn")
    held = {q["qid"] for q in head}
    rest = sorted((q for q in every if q["qid"] not in held), key=lambda q: q["qid"])
    random.Random(f"{load_federation(home).questions.seed}:{SPLIT}").shuffle(rest)
    out = head + rest
    assert len(out) == n and len({q["qid"] for q in out}) == n
    return out


def path_of(home: str, root: Path, split: str = SPLIT) -> Path:
    return root / "fed" / home / "questions" / f"{split}.jsonl"


def exposed(dataset: str) -> set[str]:
    key = {"hotpotqa": "hotpot"}.get(dataset, dataset)
    return set(json.loads(EXPOSED.read_text())[key])


def held_out(home: str, rows: list[dict]) -> list[dict]:
    gone = exposed(POOLS[home][0])
    return [q for q in rows if q["qid"] not in gone]


def musique_held_out() -> list[dict]:
    import zipfile

    from fedquest.data import musique
    from fedquest.paths import raw_dir

    corpus = {musique.doc_id(t, x) for t, x in musique.load_corpus()}
    released = {q.qid for q in musique.load_questions("hipporag")}
    gone = exposed("musique")
    with zipfile.ZipFile(raw_dir("musique") / "musique_data_v1.0.zip") as z:
        dev = [json.loads(line) for line in z.open("data/musique_ans_v1.0_dev.jsonl")]
    out = []
    for d in dev:
        sup = [p for p in d["paragraphs"] if p.get("is_supporting")]
        if d["id"] in released or d["id"] in gone or not d.get("answerable", True) or not sup:
            continue
        if all(musique.doc_id(p["title"], p["paragraph_text"]) in corpus for p in sup):
            out.append(musique.to_question(d).to_dict())
    out.sort(key=lambda q: q["qid"])
    random.Random(f"{load_federation(MUSIQUE_HOME).questions.seed}:{HELD}").shuffle(out)
    return out


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def targets() -> list[tuple[str, str]]:
    return [(h, h) for h in POOLS] + [(c, h) for h in POOLS for c in COPIES[h]]


def files() -> list[tuple[str, str]]:
    return [(f, SPLIT) for f, _ in targets()] + [(f, HELD) for f, _ in targets()] + [(MUSIQUE_HOME, HELD)]


def check(root: Path) -> int:
    want = dict(line.split()[::-1] for line in HASHES.read_text().splitlines() if line.strip())
    bad = 0
    for home, split in files():
        p = path_of(home, root, split)
        rel = f"fed/{home}/questions/{split}.jsonl"
        if not p.exists():
            print(f"missing: {p}")
            bad += 1
        elif digest(p) != want.get(rel):
            print(f"differs from configs/pools.sha256: {p}")
            bad += 1
        else:
            print(f"ok: {p} ({sum(1 for _ in read_jsonl(p))} questions)")
    return 1 if bad else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="only check the pool files against configs/pools.sha256")
    ap.add_argument("--hashes", action="store_true", help="write configs/pools.sha256 from the pool files")
    ap.add_argument("--out", type=Path, default=None, help="the root the pools are written under (default: the data "
                                                           "root)")
    args = ap.parse_args(argv)
    out = (args.out or data_root()).expanduser()
    if args.check:
        return check(out)
    if args.hashes:
        HASHES.write_text("".join(f"{digest(path_of(f, out, sp))}  fed/{f}/questions/{sp}.jsonl\n"
                                  for f, sp in files()))
        print(f"wrote {HASHES}")
        return 0
    for home in POOLS:
        rows = pool(home, data_root())
        held = held_out(home, rows)
        for fed in (home, *COPIES[home]):
            write_jsonl(path_of(fed, out), rows)
            write_jsonl(path_of(fed, out, HELD), held)
            print(f"{fed}: {len(rows)} questions -> {path_of(fed, out)}; held out {len(held)}")
    held = musique_held_out()
    write_jsonl(path_of(MUSIQUE_HOME, out, HELD), held)
    print(f"{MUSIQUE_HOME}: held out {len(held)} -> {path_of(MUSIQUE_HOME, out, HELD)}")
    return check(out) if HASHES.exists() else 0


if __name__ == "__main__":
    sys.exit(main())
