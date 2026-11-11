from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import pickle
import socket
import time
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from fedquest.errors import ReplyTooLarge, wire_error

log = logging.getLogger(__name__)

TOKEN_REFRESH_S = 900
HOST_THREADS = 128
REPLY_LIMIT_BYTES = 9 * 1024 * 1024
ACTIONS = frozenset({"health", "info", "blooms", "probe", "hippo", "extract", "read", "fetch_raw", "versions",
                     "witnesses", "update", "stats", "warmup", "hippo_state", "catalog",
                     "synopsis"})
PROFILED = frozenset({"read", "extract", "info", "stats"})


def sized(where: str, action: str, out: Any) -> Any:
    n = len(pickle.dumps(out, protocol=pickle.HIGHEST_PROTOCOL))
    if n > REPLY_LIMIT_BYTES:
        raise ReplyTooLarge(f"{action} at {where}: a reply of about {n} bytes is over the transport's result cap "
                            f"({REPLY_LIMIT_BYTES}): ask for fewer items a call")
    return out


def llm_cache_file(facility: str) -> Path | None:
    root = os.environ.get("FEDQUEST_HOST_LLM_CACHE_DIR")
    if not root:
        return None
    return Path(root).expanduser() / f"llm_{facility}_{socket.gethostname().split('.')[0]}.sqlite"


def records_at_start(llm: Any) -> dict[str, Any]:
    own = os.environ.get("FEDQUEST_SITE_STORE_DIR")
    store = Path(own).expanduser() if own else None
    cache = getattr(llm, "cache", None)
    return {"llm_calls": len(cache) if cache is not None else None,
            "llm_cache": str(cache.path) if cache is not None else None,
            "site_stores": sum(1 for _ in store.glob("*/sites/*/facts.sqlite")) if store else None,
            "store_dir": str(store) if store else None}


class SiteHost:
    def __init__(self, facility: str, keys: list[str], slot: int = 0) -> None:
        self.facility = facility
        self.slot = slot
        self.keys = list(keys)
        self._keyset = set(self.keys)
        self._services: dict[str, Any] = {}
        self._profiled: dict[tuple[str, str], Any] = {}
        self._locks: dict[Any, asyncio.Lock] = {}
        self._llm: Any = None
        self._clients: dict[str | None, Any] = {}
        self._qdrant: Any = None
        self._refresh: asyncio.Task[None] | None = None
        self._started = time.time()
        self.served = 0
        self.failed = 0
        self.at_start: dict[str, Any] = {}

    @property
    def where(self) -> str:
        return f"{self.facility}@{socket.gethostname().split('.')[0]}"

    async def startup(self) -> None:
        running = asyncio.get_running_loop()
        threads = max(1, int(os.environ.get("FEDQUEST_HOST_THREADS") or HOST_THREADS))
        running.set_default_executor(ThreadPoolExecutor(threads, thread_name_prefix=f"host-{self.facility}"))
        from fedquest.deploy import node_qdrant
        from fedquest.llm.client import LLMClient

        cache = llm_cache_file(self.facility)
        self._llm = LLMClient.from_env(cache_path=cache) if cache else LLMClient.from_env()
        self._clients = {None: self._llm}
        self.at_start = await asyncio.to_thread(records_at_start, self._llm if cache else None)
        if node_qdrant.enabled() and self.keys:
            self._qdrant = await asyncio.to_thread(node_qdrant.serve_slices, self.facility, self.keys,
                                                   float(os.environ.get("FEDQUEST_QDRANT_READY_S") or 300))
        self._refresh = asyncio.create_task(self._refresh_tokens())
        log.info("site host %s up with %d sites", self.where, len(self.keys))

    async def shutdown(self) -> None:
        if self._refresh is not None:
            self._refresh.cancel()
        try:
            closed: set[int] = set()
            for service in [*self._services.values(), *self._profiled.values()]:
                if id(service) not in closed:
                    closed.add(id(service))
                    await service.close()
            for client in self._clients.values():
                await client.aclose()
        finally:
            if self._qdrant is not None:
                await asyncio.to_thread(self._qdrant.stop)
                self._qdrant = None

    async def _refresh_tokens(self) -> None:
        while True:
            await asyncio.sleep(TOKEN_REFRESH_S)
            for profile, client in list(self._clients.items()):
                try:
                    await asyncio.to_thread(client.tokens.get)
                except Exception as e:
                    log.warning("token refresh failed at %s (profile %s): %s", self.where, profile or "-", e)

    def _client(self, profile: str | None) -> Any:
        if profile not in self._clients:
            self._clients[profile] = self._llm.for_profile(profile)
        return self._clients[profile]

    async def _service(self, key: str, profile: str | None = None) -> Any:
        if key not in self._keyset:
            raise KeyError(f"{self.where} does not host {key!r}")
        held = self._services.get(key) if profile is None else self._profiled.get((key, profile))
        if held is not None:
            return held
        lock = self._locks.setdefault(key if profile is None else (key, profile), asyncio.Lock())
        async with lock:
            from fedquest.config import load_federation
            from fedquest.site.service import SiteService

            federation, site = key.split("/", 1)
            if profile is None:
                if key not in self._services:
                    scfg = (await asyncio.to_thread(load_federation, federation)).site(site)
                    client = self._client(self._llm.cfg.site_profile(scfg))
                    self._services[key] = await asyncio.to_thread(SiteService.open, federation, site, llm=client)
                return self._services[key]
            if (key, profile) not in self._profiled:
                scfg = (await asyncio.to_thread(load_federation, federation)).site(site)
                if profile == self._llm.cfg.site_profile(scfg):
                    self._profiled[(key, profile)] = await self._service(key)
                else:
                    self._profiled[(key, profile)] = await asyncio.to_thread(
                        SiteService.open, federation, site, llm=self._client(profile))
            return self._profiled[(key, profile)]

    @contextlib.asynccontextmanager
    async def _served(self, key: str, action: str) -> AsyncIterator[None]:
        from fedquest import timeline

        try:
            with timeline.in_site(key.split("/", 1)[-1]):
                yield
        except Exception as e:
            self.failed += 1
            log.warning("%s of %s at %s failed", action, key, self.where, exc_info=True)
            sent = wire_error(e, self.where, action)
            if sent is e:
                raise
            raise sent from e
        self.served += 1

    async def dispatch(self, key: str | None, action: str, kwargs: dict[str, Any]) -> Any:
        if action not in ACTIONS:
            raise ValueError(f"unknown action {action!r}")
        if action == "health":
            return self.health()
        kwargs = dict(kwargs)
        profile = kwargs.pop("profile", None)
        kwargs.pop("rid", None)
        if action == "warmup":
            return await self.warmup(kwargs.get("keys"))
        assert key is not None
        if action == "hippo_state":
            if key not in self._keyset:
                raise KeyError(f"{self.where} does not host {key!r}")
            from fedquest.site.hippo import index_ready

            return await asyncio.to_thread(index_ready, key.split("/", 1)[0])
        async with self._served(key, action):
            if action == "stats":
                return await self._stats(key, profile, **kwargs)
            service = await self._service(key, profile if action in PROFILED else None)
            return sized(self.where, action, await getattr(service, action)(**kwargs))

    async def _stats(self, key: str, profile: str | None, busy_only: bool = False) -> dict[str, Any]:
        from fedquest.federation.base import profiled_stats

        own = await self._service(key)
        out = await own.stats(busy_only=busy_only)
        if profile is None:
            return out
        other = self._profiled.get((key, profile))
        theirs = await other.stats(busy_only=True) if other is not None and other is not own else None
        return profiled_stats(out, theirs, profile)

    async def warmup(self, keys: list[str] | None = None) -> dict[str, Any]:
        sem = asyncio.Semaphore(8)

        async def one(key: str) -> tuple[str, Any]:
            async with sem:
                try:
                    return key, await (await self._service(key)).warmup()
                except Exception as e:
                    return key, {"error": f"{type(e).__name__}: {e}"}

        return dict(await asyncio.gather(*(one(k) for k in keys or self.keys)))

    def health(self) -> dict[str, Any]:
        from fedquest.federation.base import WIRE_VERSION
        from fedquest.llm.prompts import READ_VERSION

        def roles(client: Any) -> dict[str, str]:
            return {r: m for r, m in client.cfg.roles.items() if r.startswith("extract")}

        out = {
            "facility": self.facility,
            "slot": self.slot,
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "sites": len(self.keys),
            "opened": len(self._services),
            "uptime_s": round(time.time() - self._started, 1),
            "served": self.served,
            "failed": self.failed,
            "data": os.environ.get("FEDQUEST_DATA", "<repo>/data"),
            "reader_version": READ_VERSION,
            "wire": WIRE_VERSION,
            "commit": _commit(),
            "models": roles(self._llm) if self._llm else None,
            "trust": self._llm.trust if self._llm else None,
            "at_start": self.at_start,
        }
        if self._qdrant is not None:
            out["node_qdrant"] = {k: v for k, v in self._qdrant.describe().items() if k != "slices"}
        if self._llm is not None and llm_cache_file(self.facility):
            out["llm_cache"] = str(self._llm.cache.path)
        return out


def _commit() -> str:
    from fedquest.eval.runner import _git_commit

    try:
        return _git_commit()
    except Exception:
        return "?"
