from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

LABEL = "evidence-z1"
MIN_N = 40
MIN_POS = 4
C_REG = 1.0


def _sig(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, x))))


def _cells(kind: str, modality: str, rnd: str, encoder: str = "") -> list[str]:
    pooled = [f"{kind}|{modality}|{rnd}", f"{kind}|{modality}", kind, "all"]
    return [f"{kind}|{modality}|{encoder}|{rnd}", f"{kind}|{modality}|{encoder}", *pooled] if encoder else pooled


def _x(o: dict[str, Any]) -> list[float]:
    m = o.get("mentions")
    return [float(o["z"]), float(bool(o.get("signal"))), float(bool(m)) if m is not None else 0.0,
            0.0 if m is not None else 1.0]


class ZCalibrator:
    def __init__(self, cells: dict[str, dict[str, Any]] | None = None, meta: dict[str, Any] | None = None) -> None:
        self.cells = cells or {}
        self.meta = meta or {}
        self.source: dict[str, Any] = {}

    def alpha(self, kind: str, z: float, signal: bool | None, modality: str, rnd: str = "first",
              mention: bool | None = None, encoder: str = "") -> float | None:
        cell = next((self.cells[c] for c in _cells(kind, modality, rnd, encoder) if c in self.cells), None)
        if cell is None:
            return None
        a, b, cs, cm, cn = cell["coef"]
        base = a + b * float(z) + (cm * float(bool(mention)) if mention is not None else cn)
        if signal is None:
            return 0.5 * (_sig(base) + _sig(base + cs))
        return _sig(base + cs * float(bool(signal)))

    @property
    def calibrated(self) -> bool:
        return bool(self.cells)

    @classmethod
    def fit(cls, observations: Iterable[dict[str, Any]]) -> ZCalibrator:
        import numpy as np
        from sklearn.linear_model import LogisticRegression

        obs = [o for o in observations if o.get("path") != "gold" and o.get("evidence") is not None
               and o.get("z") is not None]
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for o in obs:
            for c in _cells(o["kind"], o.get("modality", ""), o.get("round", "first"), str(o.get("encoder") or "")):
                groups[c].append(o)
        cells: dict[str, dict[str, Any]] = {}
        for name in sorted(groups, key=lambda c: c.count("|")):
            rows = groups[name]
            y = np.array([int(bool(o["evidence"])) for o in rows])
            if len(rows) < MIN_N or y.sum() < MIN_POS or y.sum() == len(rows):
                continue
            x = np.array([_x(o) for o in rows], dtype=np.float64)
            model = LogisticRegression(C=C_REG, max_iter=1000).fit(x, y)
            coef = [float(model.intercept_[0]), *map(float, model.coef_[0])]
            if coef[1] < 0:
                model = LogisticRegression(C=C_REG, max_iter=1000).fit(x[:, 1:], y)
                coef = [float(model.intercept_[0]), 0.0, *map(float, model.coef_[0])]
            cells[name] = {"coef": coef, "n": int(len(rows)), "evidence": int(y.sum())}
        meta = {"label": LABEL, "candidates": len(obs), "evidence": sum(int(bool(o["evidence"])) for o in obs),
                "min_n": MIN_N, "min_pos": MIN_POS}
        return cls(cells, meta)

    def to_dict(self) -> dict[str, Any]:
        return {"label": LABEL, "cells": self.cells, "meta": self.meta}

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=1, sort_keys=True))
        return path

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ZCalibrator:
        if d.get("label") != LABEL:
            raise ValueError(f"not a z calibration ({d.get('label')!r}, expected {LABEL!r})")
        return cls(d.get("cells") or {}, d.get("meta") or {})


def zcal_source(name_or_path: str | Path) -> Path:
    from fedquest.paths import REPO, runs_root

    p = Path(name_or_path).expanduser()
    if p.suffix == ".json" and p.exists():
        return p
    for root in (REPO / "configs" / "calibration", runs_root() / "calibration"):
        cand = root / f"{name_or_path}.json"
        if cand.exists():
            return cand
    raise FileNotFoundError(f"no z calibration {name_or_path!r} (configs/calibration or runs/calibration): fit it "
                            "with python -m fedquest.planner.zcal fit")


@functools.lru_cache(maxsize=16)
def _load(path: str, sha: str) -> ZCalibrator:
    return ZCalibrator.from_dict(json.loads(Path(path).read_text()))


def load_zcal(name_or_path: str | Path) -> ZCalibrator:
    path = zcal_source(name_or_path)
    data = path.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    cal = _load(str(path), sha)
    cal.source = {"zcal": str(path), "zcal_sha256": sha}
    return cal


def _read(paths: Sequence[str | Path]) -> list[dict[str, Any]]:
    out = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            out += [json.loads(line) for line in f if line.strip()]
    return out


def score(cal: ZCalibrator, observations: Iterable[dict[str, Any]], bins: int = 10) -> dict[str, Any]:
    by: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for o in observations:
        if o.get("path") == "gold" or o.get("evidence") is None or o.get("z") is None:
            continue
        a = cal.alpha(o["kind"], o["z"], o.get("signal"), o.get("modality", ""), o.get("round", "first"),
                      o.get("mentions"), str(o.get("encoder") or ""))
        if a is None:
            continue
        pair = (min(max(a, 1e-6), 1 - 1e-6), int(bool(o["evidence"])))
        by[o.get("layout") or ""].append(pair)
        by["all"].append(pair)

    def stats(pairs: list[tuple[float, int]]) -> dict[str, float]:
        n = len(pairs)
        if not n:
            return {"n": 0}
        ll = sum(math.log(p) if y else math.log(1 - p) for p, y in pairs) / n
        brier = sum((p - y) ** 2 for p, y in pairs) / n
        ece, edges = 0.0, [i / bins for i in range(bins + 1)]
        for lo, hi in zip(edges, edges[1:], strict=False):
            b = [(p, y) for p, y in pairs if lo <= p < hi or (hi == 1.0 and p == 1.0)]
            if b:
                ece += len(b) / n * abs(sum(p for p, _ in b) / len(b) - sum(y for _, y in b) / len(b))
        return {"n": n, "evidence": sum(y for _, y in pairs), "mean_log_lik": ll, "brier": brier, "ece": ece}

    return {k: stats(v) for k, v in by.items()}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m fedquest.planner.zcal")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit", help="fit a z calibration from explore observations")
    f.add_argument("--inputs", nargs="+", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--drop-encoder", action="append", default=[],
                   help="fit without this encoder's candidates (repeatable): the zero-label transfer arm")
    s = sub.add_parser("score", help="score a z calibration on labeled observations")
    s.add_argument("calibration")
    s.add_argument("--inputs", nargs="+", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "fit":
        drop = set(args.drop_encoder)
        cal = ZCalibrator.fit(o for o in _read(args.inputs) if str(o.get("encoder") or "") not in drop)
        if drop:
            cal.meta["dropped_encoders"] = sorted(drop)
        path = cal.save(args.out)
        print(f"{path}: {len(cal.cells)} cells from {cal.meta['candidates']} candidates "
              f"({cal.meta['evidence']} evidence)")
        return 0 if cal.cells else 1
    report = score(load_zcal(args.calibration), _read(args.inputs))
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
