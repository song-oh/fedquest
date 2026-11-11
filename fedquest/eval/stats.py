# paired tests: bootstrap intervals, sign-flip p values, Holm, weighted kappa
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from itertools import product

import numpy as np

EXACT_UP_TO = 16
FLIP_CELLS = 4_000_000


def paired_bootstrap(a: Mapping[str, float], b: Mapping[str, float], n: int = 2000,
                     seed: int = 0) -> tuple[float, float, float] | None:
    keys = sorted(set(a) & set(b))
    if not keys:
        return None
    diff = np.array([a[k] - b[k] for k in keys], dtype=float)
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, len(diff), size=(n, len(diff)))].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(diff.mean()), float(lo), float(hi)


def paired_permutation(a: Mapping[str, float], b: Mapping[str, float], n: int = 50000,
                       seed: int = 0) -> float | None:
    keys = sorted(set(a) & set(b))
    if not keys:
        return None
    diff = np.array([a[k] - b[k] for k in keys], dtype=float)
    diff = diff[np.abs(diff) > 1e-12]
    if diff.size == 0:
        return 1.0
    observed = diff.sum() - 1e-9
    if diff.size <= EXACT_UP_TO:
        signs = np.array(list(product((1.0, -1.0), repeat=diff.size)))
        return float(np.mean(signs @ np.abs(diff) >= observed))
    rng = np.random.default_rng(seed)
    signs = rng.choice((1.0, -1.0), size=(n, diff.size))
    return float((1 + np.sum(signs @ np.abs(diff) >= observed)) / (n + 1))


def holm(pvalues: Mapping[str, float | None]) -> dict[str, float | None]:
    tested = sorted(((p, k) for k, p in pvalues.items() if p is not None))
    m, out, running = len(tested), {k: None for k in pvalues}, 0.0
    for i, (p, k) in enumerate(tested):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def differences(a: Mapping[str, float], b: Mapping[str, float]) -> dict[str, float]:
    return {k: a[k] - b[k] for k in sorted(set(a) & set(b))}


def weighted_kappa(a: Mapping[str, float], b: Mapping[str, float], levels: int = 9,
                   weights: str = "quadratic") -> float | None:
    keys = sorted(set(a) & set(b))
    if not keys:
        return None
    x, y = (np.clip(np.rint(np.array([s[k] for k in keys], dtype=float) * (levels - 1)), 0, levels - 1).astype(int)
            for s in (a, b))
    observed = np.zeros((levels, levels))
    np.add.at(observed, (x, y), 1.0)
    observed /= observed.sum()
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0))
    i, j = np.indices((levels, levels))
    w = np.abs(i - j) / (levels - 1)
    w = w ** 2 if weights == "quadratic" else w
    chance = float((w * expected).sum())
    return 1.0 if chance <= 1e-12 else float(1.0 - (w * observed).sum() / chance)


def _ordered(values: Mapping[str, float]) -> np.ndarray:
    return np.array([values[k] for k in sorted(values)], dtype=float)


def sign_flip(values: Mapping[str, float], n: int = 50000, seed: int = 0) -> float | None:
    if not values:
        return None
    d = _ordered(values)
    d = d[np.abs(d) > 1e-12]
    if d.size == 0:
        return 1.0
    observed = d.sum() - 1e-9
    if d.size <= EXACT_UP_TO:
        signs = np.array(list(product((1.0, -1.0), repeat=d.size)))
        return float(np.mean(signs @ np.abs(d) >= observed))
    rng = np.random.default_rng(seed)
    chunk, done, hits = max(1, FLIP_CELLS // d.size), 0, 0
    while done < n:
        k = min(chunk, n - done)
        hits += int(np.sum(rng.choice((1.0, -1.0), size=(k, d.size)) @ np.abs(d) >= observed))
        done += k
    return float((1 + hits) / (n + 1))


def bootstrap_bound(values: Mapping[str, float], side: str = "lower", n: int = 2000, seed: int = 0,
                    level: float = 0.95) -> tuple[float, float] | None:
    if not values:
        return None
    d = _ordered(values)
    means = d[np.random.default_rng(seed).integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    return float(d.mean()), float(np.quantile(means, 1.0 - level if side == "lower" else level))


def intersection_union(pvalues: Iterable[float | None]) -> float | None:
    ps = list(pvalues)
    return None if not ps or any(p is None for p in ps) else float(max(ps))


def cluster_ratio(num: Mapping[str, float], den: Mapping[str, float], n: int = 2000,
                  seed: int = 0) -> tuple[float, float, float] | None:
    keys = sorted(k for k in den if den[k] > 0)
    if not keys:
        return None
    x = np.array([num.get(k, 0.0) for k in keys], dtype=float)
    y = np.array([den[k] for k in keys], dtype=float)
    idx = np.random.default_rng(seed).integers(0, len(keys), size=(n, len(keys)))
    lo, hi = np.quantile(x[idx].sum(axis=1) / y[idx].sum(axis=1), [0.025, 0.975])
    return float(x.sum() / y.sum()), float(lo), float(hi)


def paired_quantile(a: Mapping[str, float], b: Mapping[str, float], q: float, n: int = 2000,
                    seed: int = 0) -> tuple[float, float, float] | None:
    keys = sorted(set(a) & set(b))
    if not keys:
        return None
    x, y = (np.array([s[k] for k in keys], dtype=float) for s in (a, b))
    idx = np.random.default_rng(seed).integers(0, len(keys), size=(n, len(keys)))
    lo, hi = np.quantile(np.quantile(x[idx], q, axis=1) - np.quantile(y[idx], q, axis=1), [0.025, 0.975])
    return float(np.quantile(x, q) - np.quantile(y, q)), float(lo), float(hi)


def least_squares(x: Sequence[Sequence[float]], y: Sequence[float]) -> tuple[list[float], float | None] | None:
    if not len(y):
        return None
    xs, ys = np.array(x, dtype=float), np.array(y, dtype=float)
    coef = np.linalg.lstsq(xs, ys, rcond=None)[0]
    total = float(((ys - ys.mean()) ** 2).sum())
    r2 = None if total <= 1e-12 else 1.0 - float(((ys - xs @ coef) ** 2).sum()) / total
    return [float(c) for c in coef], r2
