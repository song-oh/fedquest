# question halves A and B, and {x} fit names for cross-fitted calibrations and routers
from __future__ import annotations

from fedquest.data.splits import dev_half

PLACEHOLDER = "{x}"
HALVES = ("A", "B")


def half(qid: str) -> str:
    return "A" if dev_half(qid) == "devA" else "B"


def other(qid: str) -> str:
    return "B" if half(qid) == "A" else "A"


def split_of(h: str) -> str:
    if h not in HALVES:
        raise ValueError(f"a half is one of {HALVES}, not {h!r}")
    return f"dev{h}"


def crossfitted(name: str | None) -> bool:
    return bool(name) and PLACEHOLDER in str(name)


# cross-fitting: a question uses the fit from the half it is not in
def resolve(name: str | None, qid: str) -> str | None:
    if not crossfitted(name):
        return name
    return str(name).replace(PLACEHOLDER, other(qid))


def names(name: str | None) -> list[str]:
    if not name:
        return []
    if not crossfitted(name):
        return [str(name)]
    return [str(name).replace(PLACEHOLDER, h) for h in HALVES]


def in_half(qid: str, h: str | None) -> bool:
    return h is None or half(qid) == h


def split_file(path: str, out_dir: str | None = None) -> dict[str, str]:
    import json
    from pathlib import Path

    src = Path(path)
    stem = src.name.removesuffix(".obs.jsonl").removesuffix(".jsonl")
    dest = Path(out_dir) if out_dir else src.parent / "xfit"
    dest.mkdir(parents=True, exist_ok=True)
    outs = {h: dest / f"{stem}_x{h}.obs.jsonl" for h in HALVES}
    handles = {h: (outs[h].with_name(outs[h].name + ".tmp")).open("w", encoding="utf-8") for h in HALVES}
    counts = dict.fromkeys(HALVES, 0)
    try:
        with src.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                qid = json.loads(line).get("qid")
                if not qid:
                    raise ValueError(f"{src}: a record without a qid cannot be given to a half")
                h = half(str(qid))
                handles[h].write(line if line.endswith("\n") else line + "\n")
                counts[h] += 1
    finally:
        for fh in handles.values():
            fh.close()
    for h in HALVES:
        outs[h].with_name(outs[h].name + ".tmp").replace(outs[h])
    return {h: f"{outs[h]} ({counts[h]} records)" for h in HALVES}


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="python -m fedquest.xfit")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("split", help="split calibration probe files by question half (fits per half)")
    s.add_argument("files", nargs="+")
    s.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)
    for f in args.files:
        for h, out in split_file(f, args.out_dir).items():
            print(f"{f} -> half {h}: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
