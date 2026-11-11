# distractor pools: passages outside a benchmark's corpus, in seeded hash order
from __future__ import annotations

import bz2
import hashlib
import heapq
import json
import logging
import tarfile
import zipfile
from collections.abc import Iterable, Iterator
from pathlib import Path

from fedquest.paths import raw_dir

log = logging.getLogger(__name__)

POOL = "distractors.jsonl"
HOTPOT_WIKI = "enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2"
HOTPOT_WIKI_URL = f"https://nlp.stanford.edu/projects/hotpotqa/{HOTPOT_WIKI}"
FOLDERS = {"musique": "musique", "hotpotqa": "hotpotqa", "twowiki": "2wikimultihopqa"}


def pool_path(dataset: str) -> Path:
    return raw_dir(FOLDERS[dataset]) / POOL


def order_key(seed: int, title: str, text: str) -> str:
    return hashlib.sha256(json.dumps([f"{seed}:distractor", title, text]).encode()).hexdigest()


MUSIQUE_ZIP = "musique_data_v1.0.zip"


def _musique_paragraphs() -> Iterator[tuple[str, str]]:
    with zipfile.ZipFile(raw_dir("musique") / MUSIQUE_ZIP) as z:
        for name in sorted(n for n in z.namelist() if n.endswith(".jsonl")):
            with z.open(name) as f:
                for line in f:
                    if line.strip():
                        for p in json.loads(line).get("paragraphs", []):
                            yield p["title"], p["paragraph_text"]


def _hotpot_articles(path: Path) -> Iterator[tuple[str, str]]:
    with tarfile.open(path, "r|bz2") as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith(".bz2"):
                continue
            raw = tar.extractfile(member)
            if raw is None:
                continue
            for line in bz2.decompress(raw.read()).splitlines():
                if not line.strip():
                    continue
                d = json.loads(line)
                sents = d.get("text") or []
                if sents and isinstance(sents[0], list):
                    sents = sents[0]
                text = "".join(s for s in sents if isinstance(s, str)).strip()
                title = (d.get("title") or "").strip()
                if title and text:
                    yield title, text


def build_pool(dataset: str, *, seed: int = 13, limit: int | None = None, source: Path | None = None) -> Path:
    if dataset == "musique":
        from fedquest.data import musique

        published = {t for t, _ in musique.load_corpus()}
        passages: Iterable[tuple[str, str]] = _musique_paragraphs()
    elif dataset == "hotpotqa":
        from fedquest.data import hipporag

        published = {t for t, _ in hipporag.load_corpus("hotpotqa")}
        path = source or raw_dir("hotpotqa") / HOTPOT_WIKI
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing: fedquest data download hotpotqa-wiki")
        passages = _hotpot_articles(path)
        limit = 1_200_000 if limit is None else limit
    else:
        raise ValueError(f"no distractor pool for {dataset!r} (musique or hotpotqa)")
    keep: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    count = 0
    for title, text in passages:
        if title in published:
            continue
        key = order_key(seed, title, text)
        if key[:16] in seen:
            continue
        seen.add(key[:16])
        count += 1
        # inverted key: the min-heap evicts the largest order key, so the limit smallest stay
        neg = "".join(chr(0x10FFFF - ord(c)) for c in key)
        if limit is None or len(keep) < limit:
            heapq.heappush(keep, (neg, title, text))
        elif neg > keep[0][0]:
            heapq.heapreplace(keep, (neg, title, text))
        if count % 500_000 == 0:
            log.info("%s pool: %d passages read, %d kept", dataset, count, len(keep))
    ordered = sorted(((order_key(seed, t, x), t, x) for _, t, x in keep))
    out = pool_path(dataset)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".jsonl.part")
    with open(tmp, "w", encoding="utf-8") as f:
        for _, title, text in ordered:
            f.write(json.dumps({"title": title, "text": text}, ensure_ascii=False) + "\n")
    tmp.replace(out)
    log.info("%s pool: %d unique passages outside the corpus, %d written to %s", dataset, count, len(ordered), out)
    return out


CHAIN = {"musique": ("musique", "hotpotqa")}


def take(dataset: str, n: int, exclude: set[str]) -> list[tuple[str, str]]:
    if n <= 0:
        return []
    out: list[tuple[str, str]] = []
    taken: set[str] = set()
    for pool in CHAIN.get(dataset, (dataset,)):
        path = pool_path(pool)
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing: fedquest data pool {pool}")
        first = pool == dataset
        with open(path, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                if d["title"] in exclude or (not first and d["title"] in taken):
                    continue
                out.append((d["title"], d["text"]))
                taken.add(d["title"])
                if len(out) == n:
                    return out
    log.warning("%s pools hold %d distractors outside the corpus, fewer than %d", dataset, len(out), n)
    return out
