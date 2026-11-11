from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from fedquest.federation import probe_trace

log = logging.getLogger("fedquest.scale")


def questions(trace: Path, label: str | None, limit: int | None) -> list[list[dict[str, Any]]]:
    by = probe_trace.load(trace)
    keys = sorted(k for k in by if label is None or f"/{label}/" in f"/{k}")
    out = [by[k] for k in keys]
    return out[:limit] if limit else out


def _busy(stats: dict[str, Any]) -> dict[str, float]:
    return {s: float((v or {}).get("busy", {}).get("busy_seconds", 0.0)) for s, v in stats.items()}


async def replay(federation: str, rounds_by_q: list[list[dict[str, Any]]], clients: int, warmup: int = 0
                 ) -> dict[str, Any]:
    from fedquest.federation.gc import open_gc_federation

    fed = open_gc_federation(federation)
    names = set(fed.site_names)
    try:
        async def one(q_rounds: list[dict[str, Any]]) -> tuple[float, list[float], int]:
            t_q = time.perf_counter()
            times, probes = [], 0
            for r in q_rounds:
                calls = {s: ("probe", {"requests": reqs}) for s, reqs in r["sites"].items() if s in names and reqs}
                if not calls:
                    continue
                t0 = time.perf_counter()
                await fed._fanout(calls, None)
                times.append(time.perf_counter() - t0)
                probes += sum(len(reqs) for _, (_, kw) in calls.items() for reqs in [kw["requests"]])
            return time.perf_counter() - t_q, times, probes

        if warmup:
            pending = list(rounds_by_q[:warmup])

            async def warm() -> None:
                while pending:
                    await one(pending.pop(0))

            await asyncio.gather(*(warm() for _ in range(max(1, clients))))
        before = _busy(await fed.stats(busy_only=True))
        queue = list(range(len(rounds_by_q)))
        results: list[tuple[float, list[float], int]] = []

        async def worker() -> None:
            while queue:
                i = queue.pop(0)
                results.append(await one(rounds_by_q[i]))

        t0 = time.perf_counter()
        await asyncio.gather(*(worker() for _ in range(max(1, clients))))
        wall = time.perf_counter() - t0
        after = _busy(await fed.stats(busy_only=True))
    finally:
        await fed.close()
    q_lat = np.asarray([r[0] for r in results])
    r_lat = np.asarray([t for r in results for t in r[1]])
    busy = {s: round(after.get(s, 0.0) - before.get(s, 0.0), 3) for s in after}
    pct = (lambda a, p: round(float(np.percentile(a, p)), 3) if len(a) else None)  # noqa: E731
    return {"federation": federation, "clients": clients, "questions": len(results), "wall_s": round(wall, 3),
            "q_per_s": round(len(results) / wall, 4) if wall else None,
            "probes_per_s": round(sum(r[2] for r in results) / wall, 2) if wall else None,
            "q_p50": pct(q_lat, 50), "q_p95": pct(q_lat, 95), "q_mean": round(float(q_lat.mean()), 3) if len(q_lat)
            else None, "round_p50": pct(r_lat, 50), "round_p95": pct(r_lat, 95), "round_p99": pct(r_lat, 99),
            "rounds": int(len(r_lat)), "busy_s": busy, "busy_total_s": round(sum(busy.values()), 3),
            "busy_max_s": round(max(busy.values()), 3) if busy else None}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m fedquest.scale.replay")
    ap.add_argument("--plan", type=Path, required=True,
                    help="JSON list of runs: {federation, trace, label, clients, limit, warmup, tag}")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only", nargs="*", default=None, help="tags to run (default: all not yet in --out)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    plan = json.loads(args.plan.read_text())
    done = set()
    if args.out.exists():
        done = {json.loads(line).get("tag") for line in args.out.open() if line.strip()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    failed = 0
    for run in plan:
        tag = run["tag"]
        if (args.only and tag not in args.only) or tag in done:
            continue
        from fedquest.paths import runs_root

        trace = Path(str(run["trace"]).replace("$FEDQUEST_RUNS", str(runs_root()))).expanduser()
        qs = questions(trace, run.get("label"), run.get("limit"))
        if not qs:
            log.warning("%s: no traced question for label %s in %s", tag, run.get("label"), run["trace"])
            failed += 1
            continue
        log.info("%s: %d questions on %s, %d clients", tag, len(qs), run["federation"], run["clients"])
        try:
            res = asyncio.run(replay(run["federation"], qs, int(run["clients"]), int(run.get("warmup", 0))))
        except Exception as e:
            log.exception("%s failed: %s", tag, e)
            failed += 1
            continue
        res.update({k: v for k, v in run.items() if k not in ("trace",)})
        with args.out.open("a") as f:
            f.write(json.dumps(res) + "\n")
        log.info("%s: %.2f q/s, question p50 %.2f s, round p50 %.2f s", tag, res["q_per_s"] or 0, res["q_p50"] or 0,
                 res["round_p50"] or 0)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
