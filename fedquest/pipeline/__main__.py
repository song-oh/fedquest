from __future__ import annotations

import argparse
import logging
import sys

from fedquest.pipeline import driver


def _lanes(text: str | None) -> dict[str, int]:
    out = {}
    for term in (text or "").split(","):
        if term.strip():
            lane, _, n = term.partition("=")
            out[lane.strip()] = int(n)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m fedquest.pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan", help="write the plan from the catalog")
    p.add_argument("--experiments", nargs="*", help="only these experiments (shell patterns), in catalog order")
    p.add_argument("--tools", choices=("auto", "yes", "no"), default="auto",
                   help="The tool and post tasks of configs/tools.json (auto: with the whole catalog only)")
    r = sub.add_parser("run", help="run the plan until every task ended")
    r.add_argument("--lanes", help="slots per lane, e.g. gemma=4,mx=3,latency=1,judge=3")
    s = sub.add_parser("status", help="every task's state")
    s.add_argument("--open", action="store_true", help="only the tasks not done")
    sub.add_parser("stop", help="stop the driver and its tasks")
    t = sub.add_parser("retry", help="give failed tasks their tries back")
    t.add_argument("--experiments", nargs="*", help="only these experiments (shell patterns)")
    t.add_argument("--errors", action="store_true", help="also re-open finished runs that hold error rows")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
    if args.cmd == "plan":
        names = driver.catalog()
        if args.experiments:
            names = driver.select(args.experiments, names)
        with_tools = {"yes": True, "no": False}.get(args.tools, not args.experiments)
        plan = driver.build(names, with_tools=with_tools)
        path = driver.write_plan(plan)
        lanes: dict[str, int] = {}
        for t in plan["tasks"]:
            lanes[t["lane"]] = lanes.get(t["lane"], 0) + 1
        print(f"{len(plan['tasks'])} tasks of {len(names)} experiments ({lanes}) -> {path}")
        return 0
    if args.cmd == "run":
        return driver.Driver(_lanes(args.lanes)).run()
    if args.cmd == "status":
        return driver.status("open" if args.open else "all")
    if args.cmd == "stop":
        return driver.stop()
    return driver.retry(args.experiments, args.errors)


if __name__ == "__main__":
    sys.exit(main())
