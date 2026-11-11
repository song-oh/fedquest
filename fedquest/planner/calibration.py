from __future__ import annotations

import bisect
import functools
import hashlib
import itertools
import json
import logging
import math
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

LABEL = "evidence-v2"
LABEL_V3 = "evidence-v3"
LABEL_V4 = "evidence-v4"
LABELS = (LABEL, LABEL_V3, LABEL_V4)
VERSIONS = {LABEL: 2, LABEL_V3: 3, LABEL_V4: 4}
SCHEMA = 2
LENGTHS = (60, 100, 175, 300)
STRENGTHS = (2.0, 4.0, 8.0, 16.0, 32.0)
PORTABLE_STRENGTHS = (*STRENGTHS, 64.0, 128.0, 256.0, 512.0)
V3_BUCKETS = ((1, "1"), (3, "2-3"), (5, "4-5"), (10, "6-10"), (20, "11-20"), (40, "21-40"))
V3_FIRST_RANKS = (1, 2, 4, 6, 11, 21, 41)
V4_BUCKETS = ((1, "1"), (3, "2-3"), (5, "4-5"), (10, "6-10"), (20, "11-20"), (40, "21-40"), (80, "41-80"),
              (160, "81-160"))
V4_FIRST_RANKS = (1, 2, 4, 6, 11, 21, 41, 81, 161)
TAIL = 1e-6


def whole(rank: float) -> int:
    if isinstance(rank, int):
        return rank
    return math.ceil(float(rank) - 1e-9)


def bucket(rank: float, version: int = 2) -> str:
    rank = whole(rank)
    if version >= 3:
        for top, name in V4_BUCKETS if version >= 4 else V3_BUCKETS:
            if rank <= top:
                return name
        return "161+" if version >= 4 else "41+"
    if rank <= 1:
        return "1"
    if rank <= 3:
        return "2-3"
    if rank <= 5:
        return "4-5"
    if rank <= 10:
        return "6-10"
    return "11+"


def length(tokens: float) -> str:
    if tokens < LENGTHS[0]:
        return f"<{LENGTHS[0]}"
    for lo, hi in zip(LENGTHS, LENGTHS[1:], strict=False):
        if tokens < hi:
            return f"{lo}-{hi - 1}"
    return f"{LENGTHS[-1]}+"


VECTOR_KINDS = ("question", "bound", "bridge")


def expected_rank(rank: float, m_site: int, m_group: int) -> float:
    if rank < 1 or m_site < 0 or m_group < 0:
        raise ValueError(f"rank {rank!r} < 1 or a negative size ({m_site!r}, {m_group!r})")
    m_site = max(m_site, whole(rank))
    return float(rank) * (max(m_group, m_site) + 1) / (m_site + 1)


def joinable_rank(jrank: float, j_site: int, j_group: int) -> float:
    return expected_rank(jrank, j_site, j_group)


class GroupRank(float):
    r: int
    n_s: int
    n: int

    def __new__(cls, g: float, r: float, n_s: int = 0, n: int = 0) -> GroupRank:
        self = super().__new__(cls, g)
        self.r = whole(r)
        self.n_s = max(int(n_s), self.r)
        self.n = max(int(n), self.n_s)
        return self

    def __reduce__(self) -> tuple[Any, ...]:
        return GroupRank, (float(self), self.r, self.n_s, self.n)


def rank_feature(kind: str, rnd: str, rank: int, m_site: int = 0, m_group: int = 0, jrank: int | None = None,
                 j_site: int = 0, j_group: int = 0) -> GroupRank:
    if rnd == "filter" or kind not in VECTOR_KINDS:
        return GroupRank(float(rank), rank)
    if jrank is not None:
        return GroupRank(joinable_rank(jrank, j_site, j_group), jrank, j_site, j_group)
    return GroupRank(expected_rank(rank, m_site, m_group), rank, m_site, m_group)


def _v4_index(rank: int) -> int:
    return bisect.bisect_right(V4_FIRST_RANKS, rank) - 1


@functools.lru_cache(maxsize=1 << 16)
def merged_law(r: int, n_s: int, n: int) -> tuple[tuple[int, float], ...]:
    r = max(1, int(r))
    n_s = max(int(n_s), r)
    n = max(int(n), n_s)
    if n_s >= n:
        return ((_v4_index(r), 1.0),)
    masses = [0.0] * len(V4_FIRST_RANKS)
    log_p = sum(math.log((n_s - i) / (n - i)) for i in range(r))
    top, total, g = n - n_s + r, 0.0, r
    while True:
        if g >= V4_FIRST_RANKS[-1]:
            masses[-1] += max(0.0, 1.0 - total)
            break
        p = math.exp(log_p)
        masses[_v4_index(g)] += p
        total += p
        if total >= 1.0 - TAIL or g >= top:
            break
        log_p += math.log(g / (g - r + 1)) + math.log((n - g - n_s + r) / (n - g))
        g += 1
    return tuple((i, m) for i, m in enumerate(masses) if m > 0)


def _k(*parts: Any) -> str:
    return "|".join(str(p) for p in parts)


def _levels(site: str, collection: str, modality: str, kind: str, rank: float, signal: bool,
            rnd: str = "first", version: int = 2, portable: bool = False, mention: bool | None = None) -> list[str]:
    b = bucket(rank, version)
    cells = [_k("signal", signal), _k("kind", kind, signal), _k("bucket", kind, b, signal)]
    if mention is not None:
        cells.append(_k("mention", kind, b, signal, int(mention)))
        kind = f"{kind}:m{int(mention)}"
    cells += [_k("round", kind, b, rnd, signal), _k("modality", modality, kind, b, rnd, signal)]
    if portable:
        return [*cells, _k("coll", modality, collection, kind, b, rnd, signal)] if version >= 4 and collection else cells
    return [*cells, _k("site", site, collection, kind, b, rnd, signal)]


def _feature(o: dict[str, Any]) -> float:
    return float(o["g"]) if o.get("g") is not None else o["rank"]


def _unit(o: dict[str, Any], group: str, layered: bool) -> tuple[str, str, str, str]:
    return (o.get("layout") or "" if layered else "", o["qid"], o["round"], group)


def _depth_key(depth: int, *parts: Any) -> str:
    return _k(f"@{depth}", *parts)


def _add(table: dict[str, list[Any]], key: str, hit: float, n: int = 1) -> None:
    cell = table.setdefault(key, [0, 0])
    cell[0] += hit
    cell[1] += n


def _shrunk(table: dict[str, list[Any]], keys: Sequence[str], strength: float, base: str = "all") -> float:
    hit, n = table[base]
    a = hit / n
    for key in keys:
        hit, n = table.get(key, (0, 0))
        a = (hit + strength * a) / (n + strength)
    return a


def _group(o: dict[str, Any]) -> str:
    return "image" if o.get("modality") == "image" else "text"


class Calibrator:
    def __init__(self, strength: float = 8.0, version: int = 2, depth: int | None = None,
                 portable: bool = False) -> None:
        self.strength = strength
        self.version = version
        self.depth = depth
        self.portable = portable
        self.excluded: list[str] = []
        self.cross_fit: dict[float, float] = {}
        self.meta: dict[str, Any] = {}
        self.counts: dict[str, list[int]] = {}
        self.mentioned = False
        self.signals: dict[str, list[int]] = {}
        self.cover: dict[str, list[int]] = {}
        self.snippets: dict[str, list[int]] = {}
        self.entries: dict[str, dict[str, list[int]]] = {}
        self.sizes: dict[str, list[float]] = {}
        self.costs: dict[str, Any] = {}
        self.source: dict[str, Any] = {}
        self._rates: dict[tuple[Any, ...], tuple[list[tuple[float, float, float]], list[float]]] = {}

    def alpha(self, site: str, collection: str, kind: str, rank: float, signal: bool | None,
              modality: str = "", round: str = "first", mention: bool | None = None) -> float:
        if not self.counts:
            return 1.0
        law = rank if self.version >= 4 and isinstance(rank, GroupRank) else None
        rank = whole(rank)
        if _k("kind", kind, True) not in self.counts and _k("kind", kind, False) not in self.counts:
            kind = "bound"
        if self.version >= 4:
            bit = mention if self.mentioned and kind == "question" else None
            if law is not None:
                return self._transfer(site, collection, kind, law, signal, modality, round, bit)
            return self._alpha4(site, collection, kind, rank, signal, modality, round, bit)
        if signal is None:
            t, n = self.signals.get(_k(kind, bucket(rank, self.version)), (0, 0))
            w = (t + 1) / (n + 2)
            return (w * self.alpha(site, collection, kind, rank, True, modality, round, mention)
                    + (1 - w) * self.alpha(site, collection, kind, rank, False, modality, round, mention))
        if self.version < 3:
            return _shrunk(self.counts, _levels(site, collection, modality, kind, rank, signal, round,
                                                portable=self.portable), self.strength)
        return min(_shrunk(self.counts, _levels(site, collection, modality, kind, r, signal, round, self.version,
                                                self.portable), self.strength) for r in V3_FIRST_RANKS if r <= max(1, rank))

    def _transfer(self, site: str, collection: str, kind: str, g: GroupRank, signal: bool | None, modality: str,
                  rnd: str, bit: bool | None) -> float:
        f, w = self._fitted(site, collection, kind, modality, rnd, bit)
        law = merged_law(g.r, g.n_s, g.n)
        if signal is None:
            return sum(m * f[b][2] for b, m in law) / sum(m for _, m in law)
        weighed = [(m * (w[b] if signal else 1.0 - w[b]), b) for b, m in law]
        return sum(x * f[b][0 if signal else 1] for x, b in weighed) / sum(x for x, _ in weighed)

    def _fitted(self, site: str, collection: str, kind: str, modality: str, rnd: str, bit: bool | None,
                ) -> tuple[list[tuple[float, float, float]], list[float]]:
        key = ("", collection) if self.portable else (site, collection)
        key = (*key, kind, modality, rnd, bit, self.strength)
        if key in self._rates:
            return self._rates[key]
        size = len(V4_FIRST_RANKS)
        seen = [self.signals.get(_k(kind, bucket(r, self.version)), (0, 0)) for r in V4_FIRST_RANKS]
        w = [(t + 1) / (n + 2) for t, n in seen]
        deep = max((i for i, (_, n) in enumerate(seen) if n), default=None)
        if deep is not None:
            w = [w[min(i, deep)] for i in range(size)]

        def at(i: int, s: bool) -> float:
            return _shrunk(self.counts, _levels(site, collection, modality, kind, V4_FIRST_RANKS[i], s, rnd,
                                                self.version, self.portable, bit), self.strength)

        def measured(i: int, s: bool) -> bool:
            b = bucket(V4_FIRST_RANKS[i], self.version)
            cell = _k("bucket", kind, b, s) if bit is None else _k("mention", kind, b, s, int(bit))
            return bool(self.counts.get(cell, (0, 0))[1])

        rates = {True: [at(i, True) for i in range(size)], False: [at(i, False) for i in range(size)]}
        rates[None] = [w[i] * rates[True][i] + (1 - w[i]) * rates[False][i] for i in range(size)]
        f: dict[bool | None, list[float]] = {}
        for s, rate in rates.items():
            low = list(itertools.accumulate(rate, min))
            signals = (True, False) if s is None else (s,)
            last = max((i for i in range(size) if any(measured(i, x) for x in signals)), default=None)
            f[s] = low if last is None else [low[min(i, last)] for i in range(size)]
        out = ([(f[True][i], f[False][i], f[None][i]) for i in range(size)], w)
        self._rates[key] = out
        return out

    def _alpha4(self, site: str, collection: str, kind: str, rank: int, signal: bool | None, modality: str, rnd: str,
                bit: bool | None) -> float:
        def at(r: int, s: bool) -> float:
            return _shrunk(self.counts, _levels(site, collection, modality, kind, r, s, rnd, self.version,
                                                self.portable, bit), self.strength)

        def chance(r: int) -> float:
            if signal is not None:
                return at(r, signal)
            t, n = self.signals.get(_k(kind, bucket(r, self.version)), (0, 0))
            w = (t + 1) / (n + 2)
            return w * at(r, True) + (1 - w) * at(r, False)

        def measured(r: int) -> bool:
            b = bucket(r, self.version)
            return any(self.counts.get(_k("bucket", kind, b, s) if bit is None else _k("mention", kind, b, s, int(bit)),
                                       (0, 0))[1] for s in ((True, False) if signal is None else (signal,)))

        top = next((r for r in V4_FIRST_RANKS if measured(r)), 1)
        return chance(top) if rank < top else min(chance(r) for r in V4_FIRST_RANKS if top <= r <= max(1, rank))

    def coverage(self, round: str, image: bool, depth: int | None = None) -> float:
        if not self.cover:
            return 1.0
        group = "image" if image else "text"
        if self.version >= 3 and depth is not None and self.depth:
            d = max(1, min(whole(depth) if self.version >= 4 else int(depth), int(self.depth)))
            if _depth_key(d) in self.cover:
                return _shrunk(self.cover, [_depth_key(d, round), _depth_key(d, round, group)], self.strength,
                               base=_depth_key(d))
        return _shrunk(self.cover, [round, _k(round, group)], self.strength)

    def page_yield(self, round: str, image: bool, depth: int, deeper: int) -> float:
        if self.version < 3 or not self.depth or depth >= self.depth:
            return 0.0
        now, then = self.coverage(round, image, depth), self.coverage(round, image, deeper)
        return max(0.0, then - now) / max(1e-9, 1.0 - now)

    def beta(self, modality: str, tokens: float, snippet_tokens: float, rep: str | None = None) -> float:
        table = self.entries.get(rep) or self.snippets if rep else self.snippets
        if not table:
            return min(1.0, snippet_tokens / tokens) if tokens > 0 else 1.0
        return _shrunk(table, [modality, _k(modality, length(tokens))], self.strength)

    def tokens(self, site: str, collection: str, default: float, modality: str | None = None) -> float:
        if self.portable:
            total, n = self.sizes.get(modality, (0.0, 0)) if modality else (0.0, 0)
        else:
            total, n = self.sizes.get(_k(site, collection), (0.0, 0))
        return total / n if n else default

    def read_cost(self, name: str, default: float) -> float:
        value = self.costs.get(name)
        return default if value is None else float(value)

    def cap(self, modality: str, default: int) -> int:
        value = (self.costs.get("modalities") or {}).get(modality, {}).get("cap")
        return default if value is None else int(value)

    def kappa(self, modality: str) -> float | None:
        value = (self.costs.get("modalities") or {}).get(modality, {}).get("kappa")
        return None if value is None else float(value)

    def fit(self, observations: Iterable[dict[str, Any]], depth: int | None = None,
            exclude_sites: Iterable[str] = ()) -> int:
        obs = list(observations)
        cands = [o for o in obs if o.get("path") != "gold"]
        if any("round" not in o or "oid" not in o for o in cands):
            raise ValueError("observations without a round or an oid (obs schema 1): run `fedquest calibrate` again")
        if self.version >= 4:
            if any("layout" not in o for o in obs) or any(o.get("g") is None for o in cands):
                raise ValueError("observations without a layout or a rank feature g (recorded before calibration v4): "
                                 "probe again with `fedquest calibrate <federation>`")
            split = [o for o in cands if o.get("n") is not None and int(o.get("n_s") or 0) < int(o["n"])]
            if split:
                raise ValueError(f"{len(split)} observations (layout {split[0]['layout']!r}, ...) were ranked by a site "
                                 "holding part of its group (n_s < n): v4 is fitted where every group is one site's "
                                 "(the 2-site federation); score other layouts with `fedquest calibrate --reliability`")
            ks = [int(o["k"]) for o in obs if o.get("path") == "gold" and o.get("k")]
            self.depth = int(depth) if depth is not None else max(ks) if ks else None
        elif depth is not None and int(depth) > 5:
            self.version, self.depth = 3, int(depth)
        self.excluded = sorted(set(exclude_sites))
        labeled = [o for o in cands if o.get("evidence") is not None and o["site"] not in self.excluded]
        qids = sorted({o["qid"] for o in labeled})
        if len(qids) > 1:
            fold = {q: i % 2 for i, q in enumerate(qids)}
            halves = [[o for o in labeled if fold[o["qid"]] == f] for f in (0, 1)]
            grid = PORTABLE_STRENGTHS if self.portable else STRENGTHS
            self.cross_fit = {s: _cross_fit(halves, s, self.version, self.depth, self.portable) for s in grid}
            self.strength = max(grid, key=lambda s: self.cross_fit[s])
        self._count(obs)
        return len(labeled)

    def _count(self, obs: Sequence[dict[str, Any]]) -> None:
        self.counts, self.signals, self.cover, self.snippets, self.sizes = {}, {}, {}, {}, {}
        self._rates = {}
        v4 = self.version >= 4
        cands = [o for o in obs if o.get("path") != "gold"]
        for o in cands:
            if o["site"] in self.excluded:
                continue
            rank, signal = _feature(o) if v4 else int(o["rank"]), bool(o.get("signal"))
            _add(self.signals, _k(o["kind"], bucket(rank, self.version)), signal)
            if o.get("evidence") is not None:
                args = (o["site"], o["collection"], o.get("modality", ""), o["kind"], rank, signal,
                        o.get("round", "first"), self.version, self.portable)
                cells = _levels(*args)
                if v4 and o["kind"] == "question" and o.get("mentions") is not None:
                    cells = list(dict.fromkeys([*cells, *_levels(*args, mention=bool(o["mentions"]))]))
                for key in ("all", *cells):
                    _add(self.counts, key, int(bool(o["evidence"])))
        self.mentioned = any(key.startswith("mention|") for key in self.counts)
        held = {_unit(o, _group(o), v4) for o in cands if o.get("evidence")}
        units = {_unit(o, g, v4) for o in obs if o.get("path") == "gold" for g in o.get("groups", [])}
        for unit in sorted(units):
            rnd, group = unit[2], unit[3]
            for key in ("all", rnd, _k(rnd, group)):
                _add(self.cover, key, int(unit in held))
        if self.version >= 3 and self.depth:
            first: dict[tuple[str, str, str, str], int] = {}
            for o in cands:
                if o.get("evidence"):
                    unit = _unit(o, _group(o), v4)
                    first[unit] = min(first.get(unit, 1 << 30), whole(_feature(o)) if v4 else int(o["rank"]))
            probed = {_unit(o, g, v4): int(o["k"]) for o in obs if v4 and o.get("path") == "gold" and o.get("k")
                      for g in o.get("groups", [])}
            for d in range(1, int(self.depth) + 1):
                for unit in sorted(units):
                    if probed.get(unit, d) < d:
                        continue
                    rnd, group = unit[2], unit[3]
                    hit = int(first.get(unit, 1 << 30) <= d)
                    for key in (_depth_key(d), _depth_key(d, rnd), _depth_key(d, rnd, group)):
                        _add(self.cover, key, hit)
        stated = {(*_unit(o, o["site"], v4), o["oid"]): o for o in cands
                  if o.get("entry") == "snippet" and o.get("states") is not None}
        for o in stated.values():
            for key in ("all", o["modality"], _k(o["modality"], length(float(o.get("tokens") or 0)))):
                _add(self.snippets, key, int(bool(o["states"])))
        objects = {(o.get("layout") or "" if v4 else "", o["site"], o["oid"]): o for o in cands if o.get("tokens")}
        for o in objects.values():
            _add(self.sizes, o.get("modality", "") if self.portable else _k(o["site"], o["collection"]),
                 float(o["tokens"]))

    def to_dict(self) -> dict[str, Any]:
        label = LABEL_V4 if self.version >= 4 else LABEL_V3 if self.version >= 3 else LABEL
        out: dict[str, Any] = {"label": label, "strength": self.strength}
        if self.version >= 3:
            out["depth"] = self.depth
        if self.portable:
            out.update(portable=True, excluded=self.excluded, meta=self.meta,
                       cross_fit={f"{s:g}": round(ll, 2) for s, ll in self.cross_fit.items()})
        return {**out, "counts": self.counts, "signals": self.signals, "cover": self.cover,
                "snippets": self.snippets, **({"entries": self.entries} if self.entries else {}), "sizes": self.sizes}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Calibrator:
        if d.get("label") not in LABELS:
            log.warning("this calibration counts another label (%r): every candidate is equally likely; run "
                        "`fedquest calibrate <federation>`", d.get("label"))
            return cls()
        version = VERSIONS[d["label"]]
        cal = cls(float(d.get("strength", 8.0)), version,
                  int(d["depth"]) if version >= 3 and d.get("depth") else None, bool(d.get("portable")))
        cal.excluded = list(d.get("excluded") or [])
        cal.meta = dict(d.get("meta") or {})
        cal.cross_fit = {float(s): float(ll) for s, ll in (d.get("cross_fit") or {}).items()}
        cal.counts = {k: [int(v[0]), int(v[1])] for k, v in d.get("counts", {}).items()}
        cal.mentioned = version >= 4 and any(k.startswith("mention|") for k in cal.counts)
        cal.signals = {k: [int(v[0]), int(v[1])] for k, v in d.get("signals", {}).items()}
        cal.cover = {k: [int(v[0]), int(v[1])] for k, v in d.get("cover", {}).items()}
        cal.snippets = {k: [int(v[0]), int(v[1])] for k, v in d.get("snippets", {}).items()}
        cal.entries = {rep: {k: [int(v[0]), int(v[1])] for k, v in table.items()}
                       for rep, table in (d.get("entries") or {}).items()}
        cal.sizes = {k: [float(v[0]), int(v[1])] for k, v in d.get("sizes", {}).items()}
        return cal

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=1))

    @classmethod
    def load(cls, path: str | Path) -> Calibrator:
        path = Path(path)
        cal = cls.from_dict(json.loads(path.read_text()))
        costs = path.with_suffix(".costs.json")
        if costs.exists():
            cal.costs = json.loads(costs.read_text())
        return cal

    @property
    def calibrated(self) -> bool:
        return bool(self.counts)

    def snapshot(self) -> dict[str, Any]:
        def cells(table: dict[str, list[int]]) -> dict[str, dict[str, Any]]:
            return {k: {"hit": s, "n": n, "rate": round(s / n, 4) if n else None}
                    for k, (s, n) in sorted(table.items())}

        return {"strength": self.strength, "alpha": cells(self.counts), "coverage": cells(self.cover),
                "snippets": cells(self.snippets), **{f"entries {r}": cells(t) for r, t in self.entries.items()},
                "tokens": {k: round(t / n, 1) for k, (t, n) in sorted(self.sizes.items()) if n}}


def calibration_source(name_or_path: str | Path) -> Path:
    from fedquest.config import configs_dir
    from fedquest.paths import REPO, calibration_path

    text = str(name_or_path)
    if isinstance(name_or_path, Path) or "/" in text or text.endswith(".json"):
        path = Path(text).expanduser()
        return REPO / path if not path.is_absolute() and not path.exists() and (REPO / path).exists() else path
    portable = configs_dir() / "calibration" / f"{text}.json"
    return portable if portable.exists() else calibration_path(text)


_LOADED: dict[tuple[str, str, str | None], Calibrator] = {}


def load_calibration(name_or_path: str | Path) -> Calibrator:
    path = calibration_source(name_or_path)
    if not path.exists():
        raise FileNotFoundError(f"no calibration {name_or_path!r} (looked for {path}): a portable one is committed "
                                "under configs/calibration/, a layout's own comes from `fedquest calibrate`")
    raw = path.read_bytes()
    costs = path.with_suffix(".costs.json")
    cost_raw = costs.read_bytes() if costs.exists() else None
    key = (str(path.resolve()), hashlib.sha256(raw).hexdigest(),
           hashlib.sha256(cost_raw).hexdigest() if cost_raw is not None else None)
    cal = _LOADED.get(key)
    if cal is None:
        cal = Calibrator.from_dict(json.loads(raw))
        if cost_raw is not None:
            cal.costs = json.loads(cost_raw)
        cal.source = {"file": key[0], "sha256": key[1], "costs": key[2]}
        _LOADED[key] = cal
    return cal


def _rank_of(o: dict[str, Any]) -> float:
    g = _feature(o)
    if o.get("r") is not None and o.get("n_s") is not None and o.get("n") is not None:
        return GroupRank(g, o["r"], int(o["n_s"]), int(o["n"]))
    return g


def _chance(cal: Calibrator, o: dict[str, Any]) -> float:
    if cal.version >= 4:
        return cal.alpha(o["site"], o["collection"], o["kind"], _rank_of(o), bool(o.get("signal")),
                         o.get("modality", ""), o.get("round", "first"), o.get("mentions"))
    return cal.alpha(o["site"], o["collection"], o["kind"], int(o["rank"]), bool(o.get("signal")),
                     o.get("modality", ""), o.get("round", "first"))


def _log_lik(a: float, evidence: Any) -> float:
    a = min(1 - 1e-6, max(1e-6, a))
    return math.log(a if evidence else 1 - a)


def _cross_fit(halves: Sequence[Sequence[dict[str, Any]]], strength: float, version: int = 2,
               depth: int | None = None, portable: bool = False) -> float:
    total = 0.0
    for train, test in (halves, halves[::-1]):
        cal = Calibrator(strength, version, depth, portable)
        cal._count(train)
        for o in test:
            total += _log_lik(_chance(cal, o), o["evidence"])
    return total


def reliability(cal: Calibrator | str | Path, observations: Iterable[str | Path], bins: int = 10) -> dict[str, Any]:
    if not isinstance(cal, Calibrator):
        cal = load_calibration(cal)
    by_layout: dict[str, list[tuple[float, int]]] = {}
    units: dict[str, dict[tuple[str, str], dict[tuple[str, str], list[Any]]]] = {}
    for path in map(Path, observations):
        name = path.name.removesuffix(".jsonl").removesuffix(".obs")
        for line in path.read_text().splitlines():
            o = json.loads(line) if line.strip() else {}
            if o and o.get("path") != "gold" and o.get("evidence") is not None:
                layout, a, e = o.get("layout") or name, _chance(cal, o), bool(o["evidence"])
                by_layout.setdefault(layout, []).append((a, int(e)))
                unit = units.setdefault(layout, {}).setdefault((str(o["qid"]), str(o.get("round", "first"))), {})
                held = unit.setdefault((str(o["site"]), str(o["oid"])), [a, e, float(o.get("tokens") or 0)])
                held[0], held[1] = max(held[0], a), held[1] or e
    label = LABEL_V4 if cal.version >= 4 else LABEL_V3 if cal.version >= 3 else LABEL
    every = [u for by_unit in units.values() for u in by_unit.values()]
    return {"label": label, "strength": cal.strength, "portable": cal.portable, "source": dict(cal.source),
            "layouts": {name: {**_scores(pairs, bins), "first_witness": first_witness(units[name].values())}
                        for name, pairs in sorted(by_layout.items())},
            "all": {**_scores([p for pairs in by_layout.values() for p in pairs], bins),
                    "first_witness": first_witness(every)}}


def _spent(items: Sequence[tuple[float, bool, float]]) -> float:
    spent = 0.0
    for _, tied in itertools.groupby(sorted(items, key=lambda x: -x[0]), key=lambda x: x[0]):
        group = list(tied)
        w = sum(1 for _, e, _ in group if e)
        if not w:
            spent += sum(t for _, _, t in group)
            continue
        return spent + sum(t for _, e, t in group if not e) / (w + 1) + sum(t for _, e, t in group if e) / w
    return spent


def first_witness(units: Iterable[dict[Any, list[Any]]]) -> dict[str, Any]:
    spent, uniform, found, n = 0.0, 0.0, 0, 0
    for unit in units:
        items = [(float(a), bool(e), max(1.0, float(t))) for a, e, t in unit.values()]
        n += 1
        found += any(e for _, e, _ in items)
        spent += _spent([(a / t, e, t) for a, e, t in items])
        uniform += _spent([(1.0 / t, e, t) for _, e, t in items])
    if not n:
        return {"units": 0, "found": 0.0, "tokens": 0.0, "tokens_uniform": 0.0}
    return {"units": n, "found": round(found / n, 6), "tokens": round(spent / n, 3),
            "tokens_uniform": round(uniform / n, 3)}


def _scores(pairs: Sequence[tuple[float, int]], bins: int) -> dict[str, Any]:
    n = len(pairs)
    if not n:
        return {"n": 0}
    pairs = sorted(pairs)
    first: dict[float, int] = {}
    for i, (a, _) in enumerate(pairs):
        first.setdefault(a, i)
    cut: dict[int, list[tuple[float, int]]] = {}
    for a, e in pairs:
        cut.setdefault(bins * first[a] // n, []).append((a, e))
    out = [{"bin": b, "lo": round(xs[0][0], 6), "hi": round(xs[-1][0], 6), "n": len(xs),
            "evidence": sum(e for _, e in xs), "chance": round(sum(a for a, _ in xs) / len(xs), 6),
            "rate": round(sum(e for _, e in xs) / len(xs), 6)} for b, xs in sorted(cut.items())]
    evidence = sum(e for _, e in pairs)
    ll = sum(_log_lik(a, e) for a, e in pairs)
    return {"n": n, "evidence": evidence, "rate": round(evidence / n, 6),
            "chance": round(sum(a for a, _ in pairs) / n, 6), "log_lik": round(ll, 4),
            "mean_log_lik": round(ll / n, 6), "brier": round(sum((a - e) ** 2 for a, e in pairs) / n, 6),
            "ece": round(sum(x["n"] * abs(x["chance"] - x["rate"]) for x in out) / n, 6), "bins": out}
