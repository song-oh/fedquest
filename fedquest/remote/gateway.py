from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import os
import signal
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

from fedquest.remote import wire

log = logging.getLogger(__name__)

SPACING_S = 0.6
MAX_BATCH = 200
RESUBMIT = 8
REBUILD_S = 10.0
CALL_LOG = "gateway_calls.jsonl"
HOLD_S = 90.0
HOLDS = 2


class Gateway:
    def __init__(self) -> None:
        from fedquest.config import load_remote

        self.cfg = load_remote()
        self.env = {"FEDQUEST_MODELS": self.cfg.models, **self.cfg.env}
        for k in ("FEDQUEST_ANN", "FEDQUEST_VIRTUAL_SHARDS", "FEDQUEST_ANCHORS"):
            if os.environ.get(k):
                self.env.setdefault(k, os.environ[k])
        self.executors: dict[str, Any] = {}
        self.pending: dict[str, deque[tuple[dict[str, Any], asyncio.Future[dict[str, Any]], int]]] = {
            f: deque() for f in self.cfg.facilities}
        self.wake = asyncio.Event()
        self.loop: asyncio.AbstractEventLoop | None = None
        self.counts = {f: {"sent": 0, "ok": 0, "failed": 0, "resubmitted": 0} for f in self.cfg.facilities}
        self.last_error: dict[str, str] = {}
        self.built: dict[str, float] = {}
        self._warned: dict[str, list[float]] = {}
        self._log: Any = None
        self._token = b""

    def executor(self, facility: str) -> Any:
        ex = self.executors.get(facility)
        if ex is not None and getattr(ex, "_stopped", False):
            wait = self.built.get(facility, 0.0) + REBUILD_S - time.monotonic()
            if wait > 0:
                raise RuntimeError(f"the Globus Compute executor of {facility} stopped; a new one in {wait:.0f} s")
            log.warning("the Globus Compute executor of %s stopped (after: %s): starting a new one", facility,
                        self.last_error.get(facility, "no error seen"))
            with contextlib.suppress(Exception):
                ex.shutdown(wait=False, cancel_futures=True)
            ex = None
        if ex is None:
            self.executors[facility] = ex = make_executor(self.cfg.facilities[facility], facility)
            self.built[facility] = time.monotonic()
        return ex

    def _submit(self, facility: str, batch: list[tuple[dict[str, Any], asyncio.Future[dict[str, Any]], int]]) -> None:
        from fedquest.remote.task import site_call

        assert self.loop is not None
        for i, (request, fut, tries) in enumerate(batch):
            try:
                ex = self.executor(facility)
            except Exception as e:
                for request, fut, tries in batch[i:]:
                    self.loop.call_soon_threadsafe(self._retry, facility, request, fut, tries, e)
                return
            try:
                cf = ex.submit(site_call, request)
            except Exception as e:
                if "shutdown" in str(e):
                    for request, fut, tries in batch[i:]:
                        self.loop.call_soon_threadsafe(self._retry, facility, request, fut, tries, e)
                    return
                log.error("Globus Compute would not take a call to %s (%s: %s)", facility, type(e).__name__, e)
                self.loop.call_soon_threadsafe(_settle, fut, wire.error_envelope(e, retryable=False))
                continue
            cf.add_done_callback(lambda cf, r=request, f=fut, n=tries: self._done(facility, r, f, n, cf))

    def _done(self, facility: str, request: dict[str, Any], fut: asyncio.Future[dict[str, Any]], tries: int,
              cf: Any) -> None:
        assert self.loop is not None
        try:
            out = cf.result()
        except BaseException as e:
            if getattr(cf, "task_id", None) is None:
                if not cf.cancelled():
                    self.last_error[facility] = f"{type(e).__name__}: {' '.join(str(e).split())[:600]}"
                self.loop.call_soon_threadsafe(self._retry, facility, request, fut, tries, e)
                return
            out = self._failure(facility, e)
        self.loop.call_soon_threadsafe(_settle, fut, out)

    def _failure(self, facility: str, e: BaseException) -> dict[str, Any]:
        out = wire.error_envelope(e if isinstance(e, Exception) else RuntimeError(repr(e)), retryable=True)
        cause = self.last_error.get(facility)
        out["error"]["message"] = f"Globus Compute ({facility}): {out['error']['message']}" + (
            f" (the submission error: {cause})" if cause and cause not in out["error"]["message"] else "")
        return out

    def _retry(self, facility: str, request: dict[str, Any], fut: asyncio.Future[dict[str, Any]], tries: int,
               e: BaseException) -> None:
        if tries + 1 >= RESUBMIT:
            _settle(fut, self._failure(facility, e))
            return
        self.counts[facility]["resubmitted"] += 1
        warned = self._warned.setdefault(facility, [-1e9, 0])
        warned[1] += 1
        if time.monotonic() - warned[0] > 10.0:
            log.warning("Globus Compute refused %d call(s) to %s (%s; the submission error: %s); they go again in "
                        "2-30 s", warned[1], facility, " ".join(str(e).split())[:300],
                        self.last_error.get(facility, "none"))
            warned[:] = [time.monotonic(), 0]
        assert self.loop is not None
        self.loop.call_later(_backoff(tries + 1), self._enqueue, facility, request, fut, tries + 1)

    def _enqueue(self, facility: str, request: dict[str, Any], fut: asyncio.Future[dict[str, Any]],
                 tries: int = 0) -> None:
        if not fut.done():
            self.pending[facility].append((request, fut, tries))
            self.wake.set()

    async def pump(self) -> None:
        order = list(self.cfg.facilities)
        turn = 0
        while True:
            await self.wake.wait()
            for _ in range(len(order)):
                facility = order[turn % len(order)]
                turn += 1
                queue = self.pending[facility]
                if not queue:
                    continue
                batch = [queue.popleft() for _ in range(min(MAX_BATCH, len(queue)))]
                batch = [b for b in batch if not b[1].done()]
                if batch:
                    self.counts[facility]["sent"] += len(batch)
                    try:
                        await asyncio.to_thread(self._submit, facility, batch)
                    except Exception as e:
                        log.exception("submitting to %s failed", facility)
                        for _, fut, _ in batch:
                            _settle(fut, wire.error_envelope(e, retryable=True))
                    await asyncio.sleep(SPACING_S)
                    break
            if not any(self.pending.values()):
                self.wake.clear()

    async def call(self, facility: str, key: str | None, action: str, kwargs: dict[str, Any],
                   timeout_s: float, slot: int | None = None) -> dict[str, Any]:
        if facility not in self.pending:
            return wire.error_envelope(KeyError(f"no facility {facility!r} in configs/remote.yaml"), retryable=False)
        assert self.loop is not None
        fut: asyncio.Future[dict[str, Any]] = self.loop.create_future()
        if slot is None and key:
            slot = self.cfg.facilities[facility].slot_of(key)
        request = {"facility": facility, "slot": slot, "key": key, "action": action, "kwargs": kwargs,
                   "env": self.env, "timeout_s": timeout_s}
        if action == "shutdown":
            request["issued"] = time.time()
        t0 = time.perf_counter()
        self._enqueue(facility, request, fut)
        out = await fut
        dt = time.perf_counter() - t0
        self.counts[facility]["ok" if out.get("ok") else "failed"] += 1
        if self._log is not None and action not in ("ready", "hold"):
            self._log.write(json.dumps({"t": round(time.time(), 3), "facility": facility, "action": action,
                                        "s": round(dt, 4), "served_s": out.get("served_s"),
                                        "ok": bool(out.get("ok"))}) + "\n")
        return out

    async def keepalive(self, facility: str, delay: float) -> None:
        await asyncio.sleep(delay)
        while True:
            out = await self.call(facility, None, "hold", {"seconds": HOLD_S}, HOLD_S + 600.0)
            if not out.get("ok"):
                await asyncio.sleep(5.0)

    async def _conn_tcp(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        import hmac

        try:
            got = await asyncio.wait_for(reader.readexactly(wire.TOKEN_BYTES), 15.0)
        except (asyncio.IncompleteReadError, TimeoutError, ConnectionError):
            writer.close()
            return
        if not hmac.compare_digest(got, self._token):
            log.warning("a TCP connection from %s presented a wrong token: closed", writer.get_extra_info("peername"))
            writer.close()
            return
        await self._conn(reader, writer)

    async def _conn(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        lock = asyncio.Lock()
        tasks: set[asyncio.Task[None]] = set()

        async def one(frame: dict[str, Any]) -> None:
            if frame.get("op") == "stats":
                out: dict[str, Any] = {"ok": True, "value": self.stats()}
            else:
                out = await self.call(frame["facility"], frame.get("key"), frame["action"], frame.get("kwargs") or {},
                                      float(frame.get("timeout_s") or self.cfg.action_timeout_s + 60),
                                      frame.get("slot"))
            async with lock:
                with contextlib.suppress(ConnectionError, RuntimeError):
                    writer.write(wire.pack({"id": frame["id"], **out}))
                    await writer.drain()

        try:
            while (frame := await wire.read_frame(reader)) is not None:
                t = asyncio.create_task(one(frame))
                tasks.add(t)
                t.add_done_callback(tasks.discard)
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        finally:
            writer.close()

    async def _listen_tcp(self) -> tuple[Any, Any]:
        import secrets
        import socket

        listen = os.environ.get("FEDQUEST_GATEWAY_LISTEN")
        if not listen:
            return None, None
        host, _, port = listen.rpartition(":")
        self._token = secrets.token_hex(wire.TOKEN_BYTES // 2).encode()
        assert len(self._token) == wire.TOKEN_BYTES
        server = await asyncio.start_server(self._conn_tcp, host or "0.0.0.0", int(port or 0), limit=1 << 22,
                                            backlog=4096)
        bound = server.sockets[0].getsockname()[1]
        try:
            ips = [a for a in socket.gethostbyname_ex(socket.gethostname())[2] if not a.startswith("127.")]
        except OSError:
            ips = []
        wildcard = host in ("", "0.0.0.0", "::")
        addr = os.environ.get("FEDQUEST_GATEWAY_PUBLISH") or (socket.gethostname() if wildcard else host)
        info = {"host": socket.gethostname(), "addr": addr, "ips": ips if wildcard else [], "port": bound, "token": self._token.decode(), "pid": os.getpid(), "pool": wire.pool_name(),
                "remote_config": os.environ.get("FEDQUEST_REMOTE_CONFIG") or "configs/remote.yaml",
                "started": time.time()}
        path = wire.gateway_info_path()
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        fd = os.open(tmp, os.O_CREAT | os.O_TRUNC | os.O_WRONLY, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump(info, f)
        os.replace(tmp, path)
        log.info("gateway listening on TCP %s:%d too (published in %s)", info["addr"], bound, path)
        return server, path

    def stats(self) -> dict[str, Any]:
        return {"pid": os.getpid(), "pending": {f: len(q) for f, q in self.pending.items()}, "counts": self.counts,
                "last_submission_error": self.last_error}

    async def serve(self) -> None:
        from fedquest.paths import runs_root

        self.loop = asyncio.get_running_loop()
        problems = self.cfg.check_placement()
        if problems:
            raise SystemExit("configs/remote.yaml's placement: " + "; ".join(problems[:5]))
        for f in self.cfg.facilities:
            if not self.cfg.facilities[f].endpoint or self.cfg.facilities[f].endpoint.startswith("$"):
                raise SystemExit(f"no Globus Compute endpoint id for {f}: export the variable configs/remote.yaml "
                                 f"names ({self.cfg.facilities[f].endpoint})")
            await asyncio.to_thread(self.executor, f)
        logs = runs_root() / "remote_logs"
        logs.mkdir(parents=True, exist_ok=True)
        self._log = open(logs / CALL_LOG, "a", buffering=1)  # noqa: SIM115
        path = wire.gateway_socket()
        tmp = path.with_name(path.name + f".{os.getpid()}")
        tmp.unlink(missing_ok=True)
        server = await asyncio.start_unix_server(self._conn, path=str(tmp), limit=1 << 22, backlog=4096)
        os.chmod(tmp, 0o600)
        tmp.replace(path)
        (path.parent / "gateway.pid").write_text(str(os.getpid()))
        tcp_server, info_path = await self._listen_tcp()
        pump = asyncio.create_task(self.pump())
        holds = [asyncio.create_task(self.keepalive(f, i * HOLD_S / HOLDS)) for f in self.cfg.facilities
                 for i in range(HOLDS)]
        stop = asyncio.Event()
        for sig in (signal.SIGINT, signal.SIGTERM):
            self.loop.add_signal_handler(sig, stop.set)
        log.info("gateway listening on %s for %s", path, ", ".join(f"{f} ({c.endpoint})"
                                                                   for f, c in self.cfg.facilities.items()))
        try:
            await stop.wait()
        finally:
            server.close()
            if tcp_server is not None:
                tcp_server.close()
            if info_path is not None:
                info_path.unlink(missing_ok=True)
            pump.cancel()
            for t in holds:
                t.cancel()
            path.unlink(missing_ok=True)
            for ex in self.executors.values():
                with contextlib.suppress(Exception):
                    await asyncio.to_thread(ex.shutdown, False, cancel_futures=True)
            self._log.close()


_CLIENT: Any = None


def make_executor(fc: Any, facility: str) -> Any:
    from globus_compute_sdk import Client, Executor

    global _CLIENT
    if _CLIENT is None:
        _CLIENT = Client()
    port = os.environ.get("FEDQUEST_GC_AMQP_PORT")
    return Executor(endpoint_id=fc.endpoint, client=_CLIENT, user_endpoint_config=fc.config or None,
                    batch_size=MAX_BATCH, api_burst_limit=8, api_burst_window_s=10,
                    amqp_port=int(port) if port else None, label=f"fedquest-gateway-{facility}")


def probe(facility: str, wait_s: float) -> int:
    import traceback

    from fedquest.config import load_remote
    from fedquest.remote.task import site_call

    cfg = load_remote()
    fc = cfg.facilities[facility]
    print(f"{facility}: endpoint {fc.endpoint}, options {fc.config}")
    env = {"FEDQUEST_MODELS": cfg.models, **cfg.env}
    ex = make_executor(fc, facility)
    cf = None
    try:
        cf = ex.submit(site_call, {"facility": facility, "key": None, "action": "ready", "kwargs": {}, "env": env,
                                   "timeout_s": 120})
        out = cf.result(timeout=wait_s)
    except Exception as e:
        if isinstance(e, TimeoutError) and getattr(cf, "task_id", None) is not None:
            print(f"ACCEPTED: Globus Compute took the task (task {cf.task_id}); no result in {wait_s:.0f} s: the "
                  "endpoint's batch job is queued or starting (qstat -u $USER on Aurora, squeue -u $USER on Midway3)")
            return 0
        traceback.print_exc()
        print("PROBE FAILED")
        return 1
    print(json.dumps(out, indent=1, default=str)[:4000])
    print("PROBE OK (the task ran on a compute node)")
    return 0


CHECK_CONFIG = "configs/remote_check.yaml"


def _value(text: str) -> Any:
    return int(text) if text.isdigit() else text


def nodecheck(facility: str, nodes: int, sets: list[str], wait_s: float, config: str = CHECK_CONFIG) -> int:
    from fedquest.config import load_remote
    from fedquest.remote.task import site_call

    cfg = load_remote()
    plan = load_remote(config)
    fc = cfg.facilities[facility]
    options = {**fc.config, **{k: _value(v) for k, v in (x.split("=", 1) for x in sets)}, "nodes": nodes}
    env = {"FEDQUEST_MODELS": cfg.models, **cfg.env, "FEDQUEST_REMOTE_CONFIG": config}
    print(f"{time.strftime('%H:%M:%S')} {facility}: a {nodes}-node test job, options {options}; its servers serve "
          f"{config}", flush=True)
    ex = make_executor(fc.model_copy(update={"config": options}), facility)

    def task(action: str, slot: int | None = None, key: str | None = None) -> Any:
        return ex.submit(site_call, {"facility": facility, "slot": slot, "key": key, "action": action, "kwargs": {},
                                     "env": env, "timeout_s": 120,
                                     **({"issued": time.time()} if action == "shutdown" else {})})

    deadline, last, slots, seen = time.monotonic() + wait_s, -1e9, {}, None
    while len(slots) < nodes:
        if time.monotonic() > deadline:
            print(f"NODES NOT UP after {wait_s:.0f} s: slots {sorted(slots)}; the last reply: {seen}")
            return 1
        for f in [task("ready") for _ in range(2)]:
            try:
                out = f.result(timeout=90)
            except TimeoutError:
                continue
            except Exception as e:
                print(f"{time.strftime('%H:%M:%S')} {facility}: {type(e).__name__}: {e}", flush=True)
                continue
            seen = out
            if out.get("ok"):
                slots.update({int(j): h for j, h in (out["value"].get("slots") or {}).items()})
        if time.monotonic() - last > 60:
            v = (seen or {}).get("value") or {}
            why = ("no reply yet: the test job is queued or starting" if seen is None else
                   f"{v.get('host')}: {'ready' if v.get('ready') else 'starting'} "
                   f"({(v.get('log_tail') or '').splitlines()[-1:] or ''})" if seen.get("ok") else
                   (seen.get("error") or {}).get("message"))
            print(f"{time.strftime('%H:%M:%S')} {facility}: {len(slots)} of {nodes} slots up {sorted(slots)}; {why}",
                  flush=True)
            last = time.monotonic()
        if len(slots) < nodes:
            time.sleep(3.0)
    problems, crossed = [], 0
    for j in range(nodes):
        for _ in range(3):
            out = task("health", slot=j).result(timeout=300)
            if not out.get("ok"):
                problems.append(f"slot {j}: {(out.get('error') or {}).get('message')}")
                continue
            h = out["value"]
            crossed += int(out.get("via") not in (None, h.get("host")))
            build = (h.get("node_qdrant") or {}).get("build_s")
            print(f"slot {j}: served by {h.get('host')} (its slot {h.get('slot')}, {h.get('sites')} sites, Qdrant "
                  f"built in {build} s, commit {h.get('commit')}), forwarded by {out.get('via')}", flush=True)
            if h.get("slot") != j:
                problems.append(f"slot {j} was answered by the server of slot {h.get('slot')}")
        for key in plan.keys_at(facility, j)[:1]:
            out = task("info", slot=j, key=key).result(timeout=600)
            print(f"slot {j}: info of {key}: {'ok' if out.get('ok') else (out.get('error') or {}).get('message')}",
                  flush=True)
            if not out.get("ok"):
                problems.append(f"slot {j}: info of {key} failed")
    for j in range(nodes):
        with contextlib.suppress(Exception):
            task("shutdown", slot=j).result(timeout=120)
    if nodes > 1 and not crossed:
        print("(every health call landed on its slot's own node: the TCP path was not exercised; run it again)")
    for p in problems:
        print(f"PROBLEM: {p}")
    print(f"NODES OK: {nodes} nodes, each its own slot; {crossed} calls crossed to another node over TCP"
          if not problems else "NODES FAILED")
    return 1 if problems else 0


def _backoff(tries: int) -> float:
    return min(30.0, 2.0 ** tries)


def _settle(fut: asyncio.Future[dict[str, Any]], out: dict[str, Any]) -> None:
    if not fut.done():
        fut.set_result(out)


async def _ask(facility: str, action: str, key: str | None = None, kwargs: dict[str, Any] | None = None,
               timeout_s: float = 120.0, slot: int | None = None) -> dict[str, Any]:
    from fedquest.federation.gc import gateway_client

    try:
        return await (await gateway_client()).call(facility, key, action, kwargs or {}, timeout_s, slot)
    except TimeoutError:
        return wire.error_envelope(TimeoutError(f"no reply in {timeout_s + 60:.0f} s: no worker of {facility} has run "
                                                "a task yet (its batch job is queued or starting: qstat -u $USER on "
                                                "Aurora, squeue -u $USER on Midway3)"), retryable=True)
    except (ConnectionError, OSError) as e:
        return wire.error_envelope(e, retryable=True)


async def up(wait_s: float) -> int:
    from fedquest.config import load_remote

    cfg = load_remote()
    deadline = time.monotonic() + wait_s
    last = {f: -1e9 for f in cfg.facilities}
    refusals = asyncio.create_task(_print_refusals())

    async def slots_up(f: str, nodes: int) -> str | None:
        seen: dict[int, str] = {}
        while len(seen) < nodes:
            if time.monotonic() > deadline:
                return f"{f}: {len(seen)} of {nodes} node slots up after {wait_s:.0f} s ({seen})"
            replies = await asyncio.gather(*(_ask(f, "ready", timeout_s=180.0) for _ in range(2)))
            for r in replies:
                v = r.get("value") or {}
                if r.get("ok"):
                    seen.update({int(j): h for j, h in (v.get("slots") or {}).items()})
            if len(seen) < nodes and time.monotonic() - last[f] > 120:
                r = replies[-1]
                v = r.get("value") or {}
                why = r.get("error", {}).get("message") if not r.get("ok") else \
                    f"{v.get('host')}: {'ready' if v.get('ready') else 'starting'} " \
                    f"({(v.get('log_tail') or '').splitlines()[-1:] or ''})"
                print(f"{time.strftime('%H:%M:%S')} {f}: {len(seen)} of {nodes} node slots up "
                      f"{sorted(seen)}; the last reply: {why}", flush=True)
                last[f] = time.monotonic()
            if len(seen) < nodes:
                await asyncio.sleep(3.0)
        return None

    try:
        down = [m for m in await asyncio.gather(*(slots_up(f, fc.nodes) for f, fc in cfg.facilities.items())) if m]
    finally:
        refusals.cancel()
    if down:
        print("\n".join(f"NOT READY: {m}" for m in down))
        return 1
    ready = await _healths(cfg)
    return 1 if _report(cfg, ready) else 0


async def _healths(cfg: Any) -> dict[str, dict[int, Any]]:
    out: dict[str, dict[int, Any]] = {}
    for f, fc in cfg.facilities.items():
        replies = await asyncio.gather(*(_ask(f, "health", timeout_s=120.0, slot=j) for j in range(fc.nodes)))
        out[f] = {j: r.get("value") if r.get("ok") else {"error": r.get("error")} for j, r in enumerate(replies)}
    return out


async def _print_refusals() -> None:
    from fedquest.federation.gc import gateway_client

    seen: dict[str, str] = {}
    while True:
        with contextlib.suppress(Exception):
            errors = (await (await gateway_client()).stats()).get("last_submission_error") or {}
            for f, e in errors.items():
                if seen.get(f) != e:
                    seen[f] = e
                    print(f"{time.strftime('%H:%M:%S')} {f}: Globus Compute refused a submission: {e}", flush=True)
        await asyncio.sleep(10.0)


def _report(cfg: Any, healths: dict[str, dict[int, Any]]) -> list[str]:
    from fedquest.eval.runner import _git_commit
    from fedquest.federation.base import WIRE_VERSION

    here = _git_commit()
    problems = []
    for f, slots in healths.items():
        want = {j: len(cfg.keys_at(f, j)) for j in slots}
        for j, h in sorted(slots.items()):
            if "error" in h:
                problems.append(f"{f} slot {j}: {(h['error'] or {}).get('message')}")
                continue
            q = h.get("node_qdrant") or {}
            print(f"{f} slot {j} on {h.get('host')}: {h.get('sites')} sites, served {h.get('served')} (failed "
                  f"{h.get('failed')}), commit {h.get('commit')}, wire {h.get('wire')}, Qdrant built in "
                  f"{q.get('build_s')} s, up {h.get('uptime_s')} s")
            if h.get("slot") != j:
                problems.append(f"{f} slot {j} was answered by the server of slot {h.get('slot')}")
            if h.get("sites") != want[j]:
                problems.append(f"{f} slot {j} serves {h.get('sites')} sites, the placement gives it {want[j]}")
            if h.get("wire") != WIRE_VERSION:
                problems.append(f"{f} slot {j} speaks {h.get('wire')}, this coordinator {WIRE_VERSION}")
            if h.get("commit") != here:
                problems.append(f"{f} slot {j} runs commit {h.get('commit')}, this clone {here}: pull there, then "
                                "`python -m fedquest.remote.gateway down` and `up` again")
        hosts = [h.get("host") for h in slots.values() if "error" not in h]
        if len(set(hosts)) != len(hosts):
            problems.append(f"{f}: two slots on one host ({hosts})")
    for p in problems:
        print(f"PROBLEM: {p}")
    print("OK" if not problems else "NOT READY")
    return problems


async def status() -> int:
    from fedquest.config import load_remote
    from fedquest.federation.gc import gateway_client

    cfg = load_remote()
    client = await gateway_client()
    print(json.dumps((await client.stats()), indent=1))
    return 1 if _report(cfg, await _healths(cfg)) else 0


async def fresh() -> int:
    from fedquest.config import load_remote

    cfg = load_remote()
    problems = []
    for f, slots in (await _healths(cfg)).items():
        for j, h in sorted(slots.items()):
            at = h.get("at_start") if "error" not in h else None
            where = f"{f} slot {j} on {h.get('host')}"
            if "error" in h:
                problems.append(f"{f} slot {j}: no health ({(h['error'] or {}).get('message')})")
            elif not isinstance(at, dict) or at.get("llm_calls") is None or at.get("site_stores") is None:
                problems.append(f"{where}: no host record file or fact store directory set ({at})")
            elif at["llm_calls"] or at["site_stores"]:
                problems.append(f"{where}: {at['llm_calls']} model calls recorded and {at['site_stores']} fact stores "
                                f"when it started ({at.get('llm_cache')}, {at.get('store_dir')})")
            else:
                print(f"{where}: fresh (no record, no fact store at its start)")
    for p in problems:
        print(f"PROBLEM: {p}")
    print("FRESH" if not problems else "NOT FRESH")
    return 1 if problems else 0


async def down() -> int:
    from fedquest.config import load_remote

    cfg = load_remote()
    for f, fc in cfg.facilities.items():
        replies = await asyncio.gather(*(_ask(f, "shutdown", timeout_s=60.0, slot=j) for j in range(fc.nodes)))
        print(f, [r.get("value") or r.get("error") for r in replies])
    return 0


def check(facility: str) -> int:
    from fedquest.config import load_remote
    from fedquest.data.embed import _sha
    from fedquest.deploy.node_qdrant import index_jobs
    from fedquest.paths import collection_dir
    from fedquest.vdb.shards import virtual

    keys = load_remote().keys_of(facility)
    lines, problems = [], []
    try:
        jobs = index_jobs(keys)
    except ValueError as e:
        jobs, problems = [], [str(e)]
    checked: dict[Path, str | None] = {}
    for job in jobs:
        fed, site = job.cfg, job.site
        for coll in fed.own_site(site, "index").collections:
            name, cdir = f"{fed.name}/{site}/{coll.name}", fed.collection_data_dir(site, coll.name)
            v = virtual(fed, fed.site(site), coll)
            if v is not None:
                name, cdir = "/".join(v.parent), collection_dir(*v.parent)
            if cdir not in checked:
                missing = [n for n in ("objects.jsonl", "emb.npy", "emb.json") if not (cdir / n).is_file()]
                checked[cdir] = f"{name}: no {', '.join(missing)} in {cdir}" if missing else (
                    f"{cdir}/emb.npy was computed for other objects (prepare and embed {name} again)"
                    if json.loads((cdir / "emb.json").read_text()).get("objects_sha256") != _sha(cdir / "objects.jsonl")
                    else None)
                if checked[cdir]:
                    problems.append(checked[cdir])
        lines.append(f"{facility}: {fed.name}/{site} ({job.namespace}{', its own collections' if job.shard else ''})")
    print("\n".join([f"{len(keys)} sites at {facility}", *lines, *(f"PROBLEM: {p}" for p in problems),
                     "OK" if not problems else "NOT READY"]))
    return 1 if problems else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="The coordinator's gateway to the facilities' Globus Compute endpoints.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("serve")
    p = sub.add_parser("up")
    p.add_argument("--wait", type=float, default=5400.0, help="seconds to wait (queue wait plus the Qdrant build)")
    sub.add_parser("status")
    sub.add_parser("down")
    sub.add_parser("fresh", help="Check that no site server found an earlier attempt's records when it started")
    c = sub.add_parser("check", help="the data a facility's site server builds from (no server, no gateway)")
    c.add_argument("--facility", required=True)
    sub.add_parser("login", help="log this machine into Globus Compute (once, before `serve`: it opens a browser)")
    pr = sub.add_parser("probe", help="one task to a facility's endpoint without the gateway: the full error, if any")
    pr.add_argument("--facility", required=True)
    pr.add_argument("--wait", type=float, default=1800.0, help="seconds to wait for its result (a queued batch job)")
    nc = sub.add_parser("nodecheck", help="the multi-node machinery on a small test job of its own (no gateway)")
    nc.add_argument("--facility", required=True)
    nc.add_argument("--nodes", type=int, default=2)
    nc.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="an endpoint option for the test job (queue=debug, walltime=00:30:00 ...)")
    nc.add_argument("--wait", type=float, default=3600.0, help="seconds to wait for the test job and its servers")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    for noisy in ("httpx", "urllib3", "globus_sdk", "globus_compute_sdk", "pika"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    if args.cmd == "serve":
        asyncio.run(Gateway().serve())
        logging.shutdown()
        os._exit(0)
    if args.cmd == "check":
        return check(args.facility)
    if args.cmd == "login":
        from globus_compute_sdk import Client

        client = Client()
        print(f"Globus Compute login OK (web service {client.web_service_address})")
        return 0
    if args.cmd in ("probe", "nodecheck"):
        code = probe(args.facility, args.wait) if args.cmd == "probe" else \
            nodecheck(args.facility, args.nodes, args.set, args.wait)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
    return asyncio.run({"up": lambda: up(args.wait), "status": status, "down": down, "fresh": fresh}[args.cmd]())


if __name__ == "__main__":
    sys.exit(main())
