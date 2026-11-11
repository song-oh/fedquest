from __future__ import annotations

import asyncio
import contextlib
import itertools
import logging
import time
from typing import Any

from fedquest.config import load_federation, load_remote
from fedquest.federation.base import IDEMPOTENT, Federation
from fedquest.federation.replay import recorded
from fedquest.remote import wire

log = logging.getLogger(__name__)

LOST_REPLY_S = 300.0


class GatewayClient:
    def __init__(self) -> None:
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._pending: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self._ids = itertools.count(1)
        self._lock = asyncio.Lock()
        self._listener: asyncio.Task[None] | None = None

    async def _connect(self) -> None:
        if self._writer is not None and not self._writer.is_closing():
            return
        tcp = wire.gateway_tcp()
        if tcp is not None:
            addrs, port, token = tcp
            error: Exception | None = None
            for addr in addrs:
                try:
                    self._reader, self._writer = await asyncio.wait_for(
                        asyncio.open_connection(addr, port, limit=1 << 22), 15.0)
                except (OSError, TimeoutError) as e:
                    error = e
                    continue
                self._writer.write(token)
                await self._writer.drain()
                break
            else:
                raise ConnectionError(f"no gateway at {addrs}:{port} ({type(error).__name__}: {error}): is it "
                                      "running on the login node (bash scripts/remote.sh status)?") from error
        else:
            path = wire.gateway_socket()
            try:
                self._reader, self._writer = await asyncio.open_unix_connection(str(path), limit=1 << 22)
            except OSError as e:
                raise ConnectionError(f"no gateway at {path}: start it on this login node (bash scripts/remote.sh "
                                      "start)") from e
        self._listener = asyncio.create_task(self._listen(self._reader, self._writer))

    async def _listen(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while (frame := await wire.read_frame(reader)) is not None:
                fut = self._pending.pop(frame.pop("id", None), None)
                if fut is not None and not fut.done():
                    fut.set_result(frame)
        except Exception as e:
            log.warning("the gateway connection broke: %s", e)
        finally:
            writer.close()
            if self._writer is writer:
                self._writer = None
            pending, self._pending = self._pending, {}
            for fut in pending.values():
                if not fut.done():
                    fut.set_exception(ConnectionError("the gateway connection closed"))

    async def _send(self, frame: dict[str, Any], timeout_s: float) -> dict[str, Any]:
        async with self._lock:
            await self._connect()
            fid = next(self._ids)
            fut: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
            self._pending[fid] = fut
            assert self._writer is not None
            self._writer.write(wire.pack({"id": fid, **frame}))
            await self._writer.drain()
        try:
            return await asyncio.wait_for(fut, timeout_s)
        finally:
            self._pending.pop(fid, None)

    async def call(self, facility: str, key: str | None, action: str, kwargs: dict[str, Any],
                   timeout_s: float, slot: int | None = None) -> dict[str, Any]:
        frame = {"facility": facility, "key": key, "action": action, "kwargs": kwargs, "timeout_s": timeout_s + 30.0}
        if slot is not None:
            frame["slot"] = slot
        return await self._send(frame, timeout_s + 60.0)

    async def stats(self) -> dict[str, Any]:
        return (await self._send({"op": "stats"}, 30.0)).get("value", {})

    async def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            with contextlib.suppress(Exception):
                await self._writer.wait_closed()
        if self._listener is not None:
            self._listener.cancel()


_CLIENTS: dict[int, GatewayClient] = {}


async def gateway_client() -> GatewayClient:
    loop = id(asyncio.get_running_loop())
    if loop not in _CLIENTS:
        _CLIENTS[loop] = GatewayClient()
    return _CLIENTS[loop]


class GatewayEndpoint:
    def __init__(self, facility: str, key: str, name: str, timeout_s: float) -> None:
        self.facility = facility
        self.key = key
        self.name = name
        self.timeout_s = timeout_s

    async def call_timed(self, action: str, **kwargs: Any) -> tuple[Any, float, float]:
        from fedquest.llm import timing

        timeout = min(self.timeout_s, LOST_REPLY_S) if action in IDEMPOTENT else self.timeout_s
        t0 = time.perf_counter()
        try:
            out = await (await gateway_client()).call(self.facility, self.key, action, kwargs, timeout)
        except TimeoutError as e:
            if action in IDEMPOTENT:
                raise ConnectionError(f"{self.name}: no reply to {action} in {timeout:.0f} s") from e
            raise
        dt = time.perf_counter() - t0
        if not out.get("ok"):
            raise wire.rebuild(out.get("error") or {}, f"{action} at {self.name}")
        rid = kwargs.get("rid")
        if rid is not None and out.get("served_s") is not None:
            timing.note(f"served:{rid}", float(out["served_s"]), 0.0, site=self.key, action=action)
        return out.get("value"), dt, 0.0

    async def call(self, action: str, **kwargs: Any) -> Any:
        return (await self.call_timed(action, **kwargs))[0]

    async def close(self) -> None:
        return None


def open_gc_federation(federation: str) -> Federation:
    cfg = load_federation(federation)
    remote = load_remote()
    if federation not in remote.federations:
        raise KeyError(f"{federation} is not deployed: add it to configs/remote.yaml's federations")
    endpoints = []
    for s in cfg.sites:
        facility = remote.facility_of(s.machine)
        if facility is None:
            raise KeyError(f"{federation}/{s.name} runs on machine {s.machine!r}, at no facility of configs/remote.yaml")
        endpoints.append(GatewayEndpoint(facility, f"{cfg.name}/{s.name}", s.name, remote.action_timeout_s))
    return recorded(Federation(endpoints, domains={s.name: s.data_domain for s in cfg.sites}), cfg.name)
