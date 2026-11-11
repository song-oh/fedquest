# node slot registry on shared disk: claims and published server addresses
from __future__ import annotations

import contextlib
import json
import os
import re
import socket
import time
from pathlib import Path
from typing import Any


def job_id() -> str:
    for k in ("FEDQUEST_JOB_ID", "PBS_JOBID", "SLURM_JOB_ID"):
        v = os.environ.get(k)
        if v:
            return re.sub(r"[^A-Za-z0-9_.-]", "_", v)
    return "local"


def host_name() -> str:
    return os.environ.get("FEDQUEST_HOSTNAME") or socket.gethostname()


def address() -> str:
    return os.environ.get("FEDQUEST_HOST_ADDR") or socket.gethostname()


def addresses() -> list[str]:
    if os.environ.get("FEDQUEST_HOST_ADDR"):
        return []
    try:
        return [a for a in socket.gethostbyname_ex(socket.gethostname())[2] if not a.startswith("127.")]
    except OSError:
        return []


def base(facility: str) -> Path:
    from fedquest.paths import runs_root

    path = runs_root() / "remote_registry" / facility
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    return path


def job_dir(facility: str) -> Path:
    path = base(facility) / job_id()
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    return path


def _read(path: Path) -> dict[str, Any] | None:
    try:
        out = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return out if isinstance(out, dict) else None


HEARTBEAT_S = 60.0
STALE_S = 600.0


def _create(path: Path, me: str) -> bool:
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w") as f:
        json.dump({"host": me, "pid": os.getpid(), "t": time.time()}, f)
    return True


def claim(facility: str, nodes: int) -> int:
    forced = os.environ.get("FEDQUEST_SLOT")
    if forced:
        return int(forced)
    d, me = job_dir(facility), host_name()
    for j in range(nodes):
        held = _read(d / f"slot{j}.claim")
        if held is not None and held.get("host") == me:
            return j
    for j in range(nodes):
        if _create(d / f"slot{j}.claim", me):
            return j
    for j in range(nodes):
        path = d / f"slot{j}.claim"
        try:
            stale = time.time() - path.stat().st_mtime > STALE_S
        except OSError:
            stale = False
        if not stale:
            continue
        try:
            os.rename(path, d / f".slot{j}.stale.{me}.{os.getpid()}")
        except OSError:
            continue
        unpublish(facility, j)
        if _create(path, me):
            return j
    held = {j: (_read(d / f"slot{j}.claim") or {}).get("host") for j in range(nodes)}
    raise RuntimeError(f"every one of the {nodes} slots of {facility} is claimed in {d}: {held} (this host: {me})")


def keep_claim(facility: str, slot: int) -> None:
    import threading

    path = job_dir(facility) / f"slot{slot}.claim"

    def beat() -> None:
        while True:
            with contextlib.suppress(OSError):
                os.utime(path)
            time.sleep(HEARTBEAT_S)

    threading.Thread(target=beat, name=f"claim-{facility}-{slot}", daemon=True).start()


def publish(facility: str, slot: int, info: dict[str, Any]) -> Path:
    d = job_dir(facility)
    path = d / f"slot{slot}.json"
    tmp = d / f".slot{slot}.{host_name()}.{os.getpid()}.tmp"
    fd = os.open(tmp, os.O_CREAT | os.O_TRUNC | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump({**info, "slot": slot, "published": time.time()}, f)
    os.replace(tmp, path)
    return path


def unpublish(facility: str, slot: int) -> None:
    with contextlib.suppress(OSError):
        (job_dir(facility) / f"slot{slot}.json").unlink()


def lookup(facility: str, slot: int) -> dict[str, Any] | None:
    return _read(job_dir(facility) / f"slot{slot}.json")


def table(facility: str) -> dict[int, dict[str, Any]]:
    out = {}
    for path in sorted(job_dir(facility).glob("slot*.json")):
        entry = _read(path)
        if entry is not None and isinstance(entry.get("slot"), int):
            out[entry["slot"]] = {k: v for k, v in entry.items() if k != "token"}
    return out
