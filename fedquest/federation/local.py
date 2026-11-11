from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import Any

from fedquest import timeline
from fedquest.config import SiteCfg, load_federation
from fedquest.errors import wire_error
from fedquest.federation.base import PROFILED_ACTIONS, Federation, profiled_stats
from fedquest.federation.replay import recorded
from fedquest.llm import timing
from fedquest.llm.client import LLMClient
from fedquest.site.service import SiteService

ProfileOpener = Callable[[str], tuple[SiteService, LLMClient | None]]


class LocalEndpoint:
    def __init__(self, service: SiteService, name: str, owns: LLMClient | None = None,
                 open_profile: ProfileOpener | None = None) -> None:
        self.service = service
        self.name = name
        self.owns = owns
        self.open_profile = open_profile
        self.profiled: dict[str, SiteService] = {}
        self._owned: list[LLMClient] = []
        self._opening = asyncio.Lock()

    async def _for(self, profile: str) -> SiteService:
        if profile not in self.profiled:
            async with self._opening:
                if profile not in self.profiled:
                    if self.open_profile is None:
                        raise ValueError(f"{self.name}: no service can be opened under models profile {profile!r}")
                    service, client = await asyncio.to_thread(self.open_profile, profile)
                    if client is not None:
                        self._owned.append(client)
                    self.profiled[profile] = service
        return self.profiled[profile]

    async def _serve(self, action: str, profile: str | None, kwargs: dict[str, Any]) -> Any:
        if profile is None or action not in PROFILED_ACTIONS:
            return await getattr(self.service, action)(**kwargs)
        if action == "stats":
            out = await self.service.stats(**kwargs)
            other = self.profiled.get(profile)
            theirs = await other.stats(busy_only=True) if other is not None and other is not self.service else None
            return profiled_stats(out, theirs, profile)
        return await getattr(await self._for(profile), action)(**kwargs)

    async def call(self, action: str, **kwargs: Any) -> Any:
        rid = kwargs.pop("rid", None)
        profile = kwargs.pop("profile", None)
        t0 = time.perf_counter()
        try:
            with timeline.in_site(self.name):
                out = await self._serve(action, profile, kwargs)
        except Exception as e:
            sent = wire_error(e, self.name, action)
            if sent is e:
                raise
            raise sent from e
        if rid is not None:
            timing.note(f"served:{rid}", time.perf_counter() - t0, 0.0, site=self.name, action=action)
        return out

    async def close(self) -> None:
        await self.service.close()
        closed = {id(self.service)}
        for service in self.profiled.values():
            if id(service) not in closed:
                closed.add(id(service))
                await service.close()
        if self.owns is not None:
            await self.owns.aclose()
        for client in self._owned:
            await client.aclose()


def _profile_opener(federation: str, site: SiteCfg, own: str | None, service: SiteService, llm: LLMClient,
                    clients: dict[tuple[str | None, str], LLMClient]) -> ProfileOpener:
    def open_profile(profile: str) -> tuple[SiteService, LLMClient | None]:
        if profile == own:
            return service, None
        key = (profile, site.machine)
        opened = key not in clients
        if opened:
            clients[key] = llm.for_profile(profile)
        return SiteService.open(federation, site.name, llm=clients[key]), clients[key] if opened else None

    return open_profile


def open_local_federation(federation: str, *, llm: LLMClient | None = None) -> Federation:
    cfg = load_federation(federation)
    llm = llm or LLMClient.from_env()
    clients: dict[tuple[str | None, str], LLMClient] = {}
    endpoints = []
    for s in cfg.sites:
        profile = llm.cfg.site_profile(s)
        key = (profile, s.machine)
        opened = key not in clients
        if opened:
            clients[key] = llm.clone() if profile is None else llm.for_profile(profile)
        service = SiteService.open(federation, s.name, llm=clients[key])
        endpoints.append(LocalEndpoint(service, s.name, owns=clients[key] if opened else None,
                                       open_profile=_profile_opener(federation, s, profile, service, llm, clients)))
    return recorded(Federation(endpoints, domains={s.name: s.data_domain for s in cfg.sites}), cfg.name)
