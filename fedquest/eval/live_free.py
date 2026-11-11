from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

from fedquest.planner.calibration import Calibrator, _chance, _scores, first_witness, load_calibration
from fedquest.planner.zcal import ZCalibrator, load_zcal

Z_BINS = (-math.inf, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, math.inf)
VECTOR = ("question", "bound", "bridge")
Chance = Callable[[dict[str, Any]], float]


def _read(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    out = []
    for p in map(Path, paths):
        name = p.name.removesuffix(".jsonl").removesuffix(".obs")
        for line in p.read_text().splitlines():
            if line.strip():
                o = json.loads(line)
                o.setdefault("layout", name)
                out.append(o)
    return out


def _labeled(obs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [o for o in obs if o.get("path") != "gold" and o.get("evidence") is not None]


def _with_z(obs: Sequence[dict[str, Any]], variant: str | None) -> list[dict[str, Any]]:
    if variant is None:
        return list(obs)
    return [o if o.get("path") == "gold" else {**o, "z": (o.get("zv") or {}).get(variant)} for o in obs]


def rank_chance(rcal: Calibrator) -> Chance:
    return lambda o: _chance(rcal, o)


def naive_chance(rcal: Calibrator) -> Chance:
    return lambda o: rcal.alpha(o["site"], o["collection"], o["kind"], int(o.get("r") or o["rank"]),
                                bool(o.get("signal")), o.get("modality", ""), o.get("round", "first"),
                                o.get("mentions"))


def z_chance(zc: ZCalibrator, fallback: Chance) -> Chance:
    def chance(o: dict[str, Any]) -> float:
        if o.get("z") is not None:
            a = zc.alpha(o["kind"], o["z"], bool(o.get("signal")), o.get("modality", ""), o.get("round", "first"),
                         o.get("mentions"), str(o.get("encoder") or ""))
            if a is not None:
                return a
        return fallback(o)

    return chance


def _fit_rank(obs: Sequence[dict[str, Any]]) -> Calibrator:
    cal = Calibrator(version=4, portable=True)
    cal.fit(obs)
    return cal


def _fit_all(train: list[dict[str, Any]]) -> dict[str, Chance]:
    rcal = _fit_rank(train)
    rank = rank_chance(rcal)
    return {"z": z_chance(ZCalibrator.fit(train), rank), "rank_v4": rank, "rank_naive": naive_chance(rcal)}


def _cross_fit(fit_obs: Sequence[dict[str, Any]], test_obs: Sequence[dict[str, Any]],
               fit: Callable[[list[dict[str, Any]]], dict[str, Chance]]) -> dict[str, Chance]:
    qids = sorted({str(o["qid"]) for o in [*fit_obs, *test_obs]})
    fold = {q: i % 2 for i, q in enumerate(qids)}
    tables: dict[str, dict[int, float]] = defaultdict(dict)
    for f in (0, 1):
        chances = fit([o for o in fit_obs if fold[str(o["qid"])] != f])
        for o in test_obs:
            if o.get("path") != "gold" and fold[str(o["qid"])] == f:
                for name, fn in chances.items():
                    tables[name][id(o)] = fn(o)
    return {name: (lambda o, t=t: t[id(o)]) for name, t in tables.items()}


def _units(obs: Sequence[dict[str, Any]], chance: Chance) -> dict[str, list[dict[Any, list[Any]]]]:
    units: dict[str, dict[tuple[str, str], dict[tuple[str, str], list[Any]]]] = defaultdict(dict)
    for o in _labeled(obs):
        a, e = chance(o), bool(o["evidence"])
        unit = units[o["layout"]].setdefault((str(o["qid"]), str(o.get("round", "first"))), {})
        held = unit.setdefault((str(o["site"]), str(o["oid"])), [a, e, float(o.get("tokens") or 0)])
        held[0], held[1] = max(held[0], a), held[1] or e
    return {k: list(v.values()) for k, v in units.items()}


def evaluate(obs: Sequence[dict[str, Any]], chances: dict[str, Chance], pooled: str | None = None) -> dict[str, Any]:
    labeled = _labeled(obs)
    out: dict[str, Any] = {}
    for name, fn in chances.items():
        pairs: dict[str, list[tuple[float, int]]] = defaultdict(list)
        for o in labeled:
            pairs[o["layout"]].append((fn(o), int(bool(o["evidence"]))))
        units = _units(labeled, fn)
        if pooled:
            pairs[pooled] = [p for ps in list(pairs.values()) for p in ps]
            units[pooled] = [u for us in list(units.values()) for u in us]
        for layout, ps in pairs.items():
            row = out.setdefault(layout, {})
            sc = _scores(ps, 10)
            sc.pop("bins", None)
            fw = first_witness(units.get(layout, []))
            row[name] = {**sc, "first_witness": fw}
            row["uniform_tokens"] = fw["tokens_uniform"]
    return out


def site_curves(obs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    cells: dict[str, dict[str, dict[int, list[int]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(
        lambda: [0, 0])))
    for o in _labeled(obs):
        if o.get("z") is None:
            continue
        b = next(i for i in range(len(Z_BINS) - 1) if Z_BINS[i] <= float(o["z"]) < Z_BINS[i + 1])
        enc = f"{o.get('encoder') or o['site']}|{o.get('modality', '')}"
        for key in ([enc] if o["kind"] in VECTOR else []) + [f"{enc}|{o['kind']}"]:
            cell = cells[o["layout"]][key][b]
            cell[0] += 1
            cell[1] += int(bool(o["evidence"]))
    out: dict[str, Any] = {}
    for layout, groups in cells.items():
        out[layout] = {g: [{"lo": Z_BINS[b], "hi": Z_BINS[b + 1], "n": n, "evidence": e,
                            "rate": round(e / n, 6) if n else None} for b, (n, e) in sorted(bins.items())]
                       for g, bins in sorted(groups.items())}
    return out


def schedulers(rcal: Calibrator, zc: ZCalibrator, obs: Sequence[dict[str, Any]],
               fit_obs: Sequence[dict[str, Any]] = ()) -> dict[str, Any]:
    if fit_obs:
        return evaluate(obs, _cross_fit(fit_obs, obs, _fit_all))
    rank = rank_chance(rcal)
    return evaluate(obs, {"z": z_chance(zc, rank), "rank_v4": rank, "rank_naive": naive_chance(rcal)})


def held_out_encoders(fit_obs: Sequence[dict[str, Any]], test_obs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    encoders = sorted({str(o["encoder"]) for o in _labeled(fit_obs) if o.get("encoder")})
    out: dict[str, Any] = {}
    for enc in encoders:
        rest = [o for o in fit_obs if o.get("encoder") != enc]
        if not _labeled(rest):
            continue

        def fit(train: list[dict[str, Any]], enc: str = enc) -> dict[str, Chance]:
            others = [o for o in train if o.get("encoder") != enc]
            seen, held = _fit_all(train), _fit_all(others)
            return {"z": seen["z"], "z_held": held["z"], "rank_v4": seen["rank_v4"], "rank_v4_held": held["rank_v4"]}

        mine = [o for o in test_obs if o.get("path") != "gold" and o.get("encoder") == enc]
        out[enc] = evaluate(mine, _cross_fit(fit_obs, mine, fit), pooled="all")
        out[enc]["fit"] = {"candidates": len(_labeled(rest)), "z_cells": len(ZCalibrator.fit(rest).cells)}
    return out


def domains(rcal: Calibrator, zc: ZCalibrator, obs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for layout in sorted({o["layout"] for o in obs}):
        mine = [o for o in obs if o["layout"] == layout]
        rank = rank_chance(rcal)

        chances = {"z": z_chance(zc, rank), "rank_v4": rank, "rank_naive": naive_chance(rcal)}
        if len({o["qid"] for o in mine}) > 3:
            own = _cross_fit(mine, mine, _fit_all)
            chances |= {"z_indomain": own["z"], "rank_v4_indomain": own["rank_v4"]}
        out |= evaluate(mine, chances)
    return out


def sensitivity(fit_obs: Sequence[dict[str, Any]], test_obs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    variants = sorted({k for o in _labeled(fit_obs) for k in (o.get("zv") or {})})
    ranks: dict[tuple[str, ...], Chance] = {}

    def fit(train: list[dict[str, Any]]) -> dict[str, Chance]:
        key = tuple(sorted({str(o["qid"]) for o in train}))
        if key not in ranks:
            ranks[key] = rank_chance(_fit_rank(train))
        return {"z": z_chance(ZCalibrator.fit(train), ranks[key])}

    out: dict[str, Any] = {}
    for v in [None, *variants]:
        fit_v, test_v = _with_z(fit_obs, v), _with_z(test_obs, v)
        rows = evaluate(test_v, _cross_fit(fit_v, test_v, fit), pooled="all")
        out[v or "all"] = {layout: row["z"] for layout, row in rows.items()} | {
            "z_cells": len(ZCalibrator.fit(fit_v).cells)}
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m fedquest.eval.live_free")
    ap.add_argument("--rank", required=True, help="the rank calibration v4 (name or path), e.g. mmqa_v4")
    ap.add_argument("--zcal", required=True, help="the z calibration (name or path), e.g. mmqa_z")
    ap.add_argument("--fit", default=None, help="the observations both were fitted on (the 2-site federation's)")
    ap.add_argument("--inputs", nargs="+", required=True, help="held-out layouts' observation files (*.obs.jsonl)")
    ap.add_argument("--domain", nargs="*", default=[], help="other benchmarks' observation files (held-out domains)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    rcal, zc = load_calibration(args.rank), load_zcal(args.zcal)
    fit_obs = _read([args.fit]) if args.fit else []
    held = _read(args.inputs)
    every = fit_obs + held
    report: dict[str, Any] = {
        "rank": args.rank, "zcal": args.zcal, "schedulers": schedulers(rcal, zc, every, fit_obs),
        "site_curves": site_curves(every)}
    if fit_obs:
        report["cross_fit"] = {"by": "question", "folds": 2, "questions": len({str(o["qid"]) for o in fit_obs})}
        report["schedulers_frozen"] = schedulers(rcal, zc, every)
        report["held_out_encoders"] = held_out_encoders(fit_obs, every)
        report["sensitivity"] = sensitivity(fit_obs, held)
    if args.domain:
        dom = _read(args.domain)
        report["domains"] = domains(rcal, zc, dom)
        report["site_curves_domains"] = site_curves(dom)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))

    def line(row: dict[str, Any], name: str) -> str:
        r = row.get(name, {})
        return f"{name} {r.get('first_witness', {}).get('tokens')} (LL {r.get('mean_log_lik')})"

    for title, rows in (("schedulers", report["schedulers"]), ("domains", report.get("domains", {}))):
        for layout, row in sorted(rows.items()):
            names = [k for k in row if k != "uniform_tokens"]
            print(f"{title} {layout}: tokens to the first witness: " + ", ".join(line(row, n) for n in names)
                  + f", uniform {row.get('uniform_tokens')}")
    for enc, rows in sorted(report.get("held_out_encoders", {}).items()):
        row = rows.get("all", {})
        print(f"held-out encoder {enc}: " + ", ".join(line(row, n) for n in ("z", "z_held", "rank_v4", "rank_v4_held")))
    for v, rows in sorted(report.get("sensitivity", {}).items()):
        print(f"anchors {v}: tokens {rows.get('all', {}).get('first_witness', {}).get('tokens')} "
              f"LL {rows.get('all', {}).get('mean_log_lik')}")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
