# data placement: title-hash parts and ranks; Dirichlet over topic clusters
from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np

from fedquest.utils import stable_int


def part_by_title(title: str, parts: int, seed: int) -> int:
    return stable_int(seed, " ".join(title.lower().split())) % parts


def shard_by_title(title: str, m: int, seed: int = 13) -> int:
    return stable_int(f"{seed}:shard", " ".join(title.lower().split())) % m


FRACTION_SCALE = 1000


def page_key(title: str) -> str:
    return " ".join(title.lower().split())


def fraction_rank(title: str, seed: int = 13) -> int:
    return stable_int(f"{seed}:frac", page_key(title)) % FRACTION_SCALE


SHARE_SCALE = 100


def facility_rank(title: str, seed: int = 13) -> int:
    return stable_int(f"{seed}:fac", page_key(title)) % SHARE_SCALE


def dirichlet_parts(
    doc_ids: Sequence[str], doc_texts: Sequence[str], parts: int, beta: float, clusters: int, seed: int,
    *, independent: bool = False,
) -> dict[str, int]:
    order = sorted(range(len(doc_ids)), key=lambda i: doc_ids[i])
    ids = [doc_ids[i] for i in order]
    texts = [doc_texts[i] for i in order]
    if parts <= 1 or len(ids) <= parts:
        return {d: i % max(1, parts) for i, d in enumerate(ids)}
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer

    x = TfidfVectorizer(max_features=20000, stop_words="english").fit_transform(texts)
    k = max(1, min(clusters, len(ids)))
    labels = KMeans(n_clusters=k, n_init=4, random_state=seed).fit_predict(x)
    members: dict[int, list[str]] = defaultdict(list)
    for d, label in zip(ids, labels, strict=True):
        members[int(label)].append(d)
    assignment: dict[str, int] = {}
    for cluster in sorted(members):
        docs = members[cluster]
        rng = np.random.default_rng([seed, cluster] if independent else seed + cluster)
        props = rng.dirichlet(np.full(parts, beta))
        counts = np.floor(props * len(docs)).astype(int)
        for i in np.argsort(-(props * len(docs) - counts))[: len(docs) - counts.sum()]:
            counts[i] += 1
        start = 0
        for p, n in enumerate(counts):
            for d in docs[start : start + n]:
                assignment[d] = p
            start += n
    sizes = np.bincount(list(assignment.values()), minlength=parts)
    for p in np.flatnonzero(sizes == 0):
        donor = int(np.argmax(sizes))
        moved = next(d for d in ids if assignment[d] == donor)
        assignment[moved] = int(p)
        sizes[donor] -= 1
        sizes[p] += 1
    return assignment
