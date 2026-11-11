# the Globus Compute task: start this node's site server and forward the call
from __future__ import annotations

import contextlib
import fcntl
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from fedquest.remote import wire

START_WAIT_S = 2400.0
ENTRY_TTL_S = 60.0
HOLD_MAX_S = 300.0
SHUTDOWN_STALE_S = 900.0
_SPAWNED: list[subprocess.Popen[bytes]] = []
_ENTRIES: dict[tuple[str, int], tuple[dict[str, Any], float]] = {}
FORWARDED = ("FEDQUEST_MODELS", "FEDQUEST_ANN", "FEDQUEST_VIRTUAL_SHARDS", "FEDQUEST_ANCHORS", "FEDQUEST_HOST_THREADS",
             "FEDQUEST_REMOTE_CONFIG", "FEDQUEST_LLM_AIMD", "FEDQUEST_LLM_AIMD_START", "FEDQUEST_LLM_RATE")


def site_call(request: dict) -> dict:
    from fedquest.remote.task import handle

    return handle(request)


def handle(request: dict[str, Any]) -> dict[str, Any]:
    from fedquest.remote import registry

    out = _handle(request)
    out["via"] = registry.host_name()
    return out


def _handle(request: dict[str, Any]) -> dict[str, Any]:
    facility = str(request["facility"])
    env = request.get("env") or {}
    timeout = float(request.get("timeout_s") or 1800.0)
    try:
        if request.get("action") == "hold":
            seconds = float((request.get("kwargs") or {}).get("seconds") or 0.0)
            time.sleep(max(0.0, min(seconds, HOLD_MAX_S)))
            return {"ok": True, "value": {"host": socket.gethostname(), "held_s": seconds}}
        if request.get("action") == "ready":
            return {"ok": True, "value": ready(facility, env)}
        # a shutdown issued over SHUTDOWN_STALE_S ago (a stale queued task) stops nothing
        issued = request.get("issued")
        if request.get("action") == "shutdown" and issued is not None and time.time() - float(issued) > SHUTDOWN_STALE_S:
            return {"ok": True, "value": {"stopping": None, "stale_s": round(time.time() - float(issued)),
                                          "host": socket.gethostname()}}
        slot = request.get("slot")
        if slot is not None and not _is_mine(facility, int(slot)):
            start(facility, env)
            return forward_slot(facility, int(slot), request, timeout)
        if request.get("action") == "shutdown" and not connectable(wire.server_socket(facility)):
            return {"ok": True, "value": {"stopping": None, "host": socket.gethostname()}}
        ensure_server(facility, env, float(request.get("start_wait_s") or START_WAIT_S))
        return forward(facility, request, timeout)
    except Exception as e:
        return wire.error_envelope(e, retryable=True)


def local_state(facility: str) -> dict[str, Any] | None:
    try:
        out = json.loads((wire.sock_dir() / f"site-{facility}.json").read_text())
    except (OSError, ValueError):
        return None
    pid = out.get("pid")
    return out if isinstance(pid, int) and _alive(pid) else None


def _is_mine(facility: str, slot: int) -> bool:
    state = local_state(facility)
    if state is not None:
        return state.get("slot") == slot
    from fedquest.remote import registry

    entry = _entry(facility, slot)
    return entry is not None and entry.get("host") == registry.host_name()


def _entry(facility: str, slot: int, fresh: bool = False) -> dict[str, Any] | None:
    from fedquest.remote import registry

    held = _ENTRIES.get((facility, slot))
    if held is not None and not fresh and time.monotonic() - held[1] < ENTRY_TTL_S:
        return held[0]
    entry = registry.lookup(facility, slot)
    if entry is None:
        _ENTRIES.pop((facility, slot), None)
    else:
        _ENTRIES[(facility, slot)] = (entry, time.monotonic())
    return entry


def forward_slot(facility: str, slot: int, request: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    from fedquest.errors import SiteRetryable

    body = {"key": request.get("key"), "action": request["action"], "kwargs": request.get("kwargs") or {}}
    for attempt in (0, 1):
        entry = _entry(facility, slot, fresh=attempt > 0)
        if entry is None:
            raise SiteRetryable(f"slot {slot} of {facility} has no site server up yet (its node has had no task, or "
                                "its Qdrant is still building)")
        error: Exception | None = None
        for addr in [entry["addr"], *(entry.get("ips") or [])]:
            try:
                return wire.call_tcp(addr, entry["port"], entry["token"], body, timeout_s)
            except socket.gaierror as e:
                error = e
            except (ConnectionError, OSError) as e:
                error = e
                break
        if attempt:
            raise ConnectionError(f"slot {slot} of {facility} at {entry.get('host')}:{entry.get('port')}: "
                                  f"{type(error).__name__}: {error}") from error
    raise AssertionError("unreachable")


def ready(facility: str, env: dict[str, str]) -> dict[str, Any]:
    from fedquest.remote import registry

    host = socket.gethostname()
    out: dict[str, Any] = {"ready": False, "host": host}
    if connectable(wire.server_socket(facility)):
        reply = forward(facility, {"action": "health", "key": None, "kwargs": {}}, 60.0)
        if reply.get("ok"):
            out.update(ready=True, slot=reply["value"].get("slot"), health=reply["value"])
    if not out["ready"]:
        start(facility, env)
        state = local_state(facility)
        pid = _read_pid(facility)
        out.update(slot=state.get("slot") if state else None, starting=pid is not None and _alive(pid),
                   log_tail=_tail(log_path(facility)))
    out["slots"] = {j: e.get("host") for j, e in registry.table(facility).items()}
    return out


def connectable(path: Path) -> bool:
    if not path.exists():
        return False
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(5.0)
        try:
            s.connect(str(path))
        except OSError:
            return False
    return True


def log_path(facility: str) -> Path:
    from fedquest.paths import runs_root

    path = runs_root() / "remote_logs" / f"{facility}_{socket.gethostname().split('.')[0]}.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _pid_file(facility: str) -> Path:
    return wire.sock_dir() / f"site-{facility}.pid"


def _read_pid(facility: str) -> int | None:
    try:
        return int(_pid_file(facility).read_text().strip())
    except (OSError, ValueError):
        return None


def _alive(pid: int) -> bool:
    for p in list(_SPAWNED):
        if p.poll() is not None:
            _SPAWNED.remove(p)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    with contextlib.suppress(OSError):
        if Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[-1].split()[0] == "Z":
            return False
    return True


def start(facility: str, env: dict[str, str]) -> None:
    if connectable(wire.server_socket(facility)):
        return
    lock_path = wire.sock_dir() / f"site-{facility}.lock"
    with open(lock_path, "a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            if connectable(wire.server_socket(facility)):
                return
            pid = _read_pid(facility)
            if pid is not None and _alive(pid):
                return
            from fedquest.paths import REPO

            child_env = dict(os.environ)
            for k, v in env.items():
                if k in FORWARDED and v is not None:
                    child_env[k] = str(v)
            out = open(log_path(facility), "ab")  # noqa: SIM115
            proc = subprocess.Popen([sys.executable, "-m", "fedquest.remote.server", "--facility", facility],
                                    cwd=REPO, env=child_env, stdin=subprocess.DEVNULL, stdout=out,
                                    stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
            out.close()
            _SPAWNED.append(proc)
            _pid_file(facility).write_text(str(proc.pid))
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def ensure_server(facility: str, env: dict[str, str], wait_s: float) -> None:
    path = wire.server_socket(facility)
    if connectable(path):
        return
    start(facility, env)
    deadline = time.monotonic() + wait_s
    while not connectable(path):
        pid = _read_pid(facility)
        if pid is None or not _alive(pid):
            raise RuntimeError(f"the site server of {facility} on {socket.gethostname()} is not running; the end "
                               f"of its log ({log_path(facility)}):\n{_tail(log_path(facility))}")
        if time.monotonic() > deadline:
            raise TimeoutError(f"the site server of {facility} on {socket.gethostname()} is not ready after "
                               f"{wait_s:.0f} s")
        time.sleep(1.0)


def forward(facility: str, request: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    body = {"key": request.get("key"), "action": request["action"], "kwargs": request.get("kwargs") or {}}
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(timeout_s)
        s.connect(str(wire.server_socket(facility)))
        s.sendall(wire.pack(body))
        return wire.recv_frame(s)


def _tail(path: Path, lines: int = 15) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])
    except OSError:
        return "(no log)"
