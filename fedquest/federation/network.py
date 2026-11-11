from __future__ import annotations

import asyncio
import math
import time
from statistics import NormalDist
from typing import TYPE_CHECKING, Any

from fedquest.config import NetworkCfg
from fedquest.federation.replay import _scope
from fedquest.llm.cache import canonical_json
from fedquest.utils import stable_int

if TYPE_CHECKING:
    from fedquest.federation.base import Federation, SiteEndpoint

_UNIT = float(16**15)
_NORMAL = NormalDist()


class DelayedEndpoint:
    def __init__(self, inner: SiteEndpoint, net: NetworkCfg) -> None:
        self.inner, self.net = inner, net
        self.name = inner.name
        self._sent: dict[str, int] = {}

    def delay(self, scope: str, n: int) -> float:
        rtt = self.net.rtt_for(self.name) / 1000.0
        extra = self.net.extra_for(self.name) / 1000.0
        if rtt <= 0 or self.net.jitter == "none":
            return max(0.0, rtt) + extra
        u = (stable_int(self.net.seed, "rtt", scope, self.name, n) + 0.5) / _UNIT
        return rtt * math.exp(self.net.sigma * _NORMAL.inv_cdf(u)) + extra

    def transfer(self, nbytes: int) -> float:
        return nbytes * 8 / (self.net.mbps * 1e6) if self.net.mbps else 0.0

    async def call_timed(self, action: str, **kwargs: Any) -> tuple[Any, float, float]:
        scope = _scope.get()
        n = self._sent.get(scope, 0)
        self._sent[scope] = n + 1
        rtt = self.delay(scope, n)
        there = rtt / 2 + (self.transfer(len(canonical_json(kwargs).encode())) if self.net.mbps else 0.0)
        t0 = time.perf_counter()
        await asyncio.sleep(there)
        t1 = time.perf_counter()
        out, inner_s, _ = await timed_call(self.inner, action, kwargs)
        back = rtt / 2 + (self.transfer(len(canonical_json(out).encode())) if self.net.mbps else 0.0)
        t2 = time.perf_counter()
        await asyncio.sleep(back)
        return out, (t1 - t0) + inner_s + (time.perf_counter() - t2), there + back

    async def call(self, action: str, **kwargs: Any) -> Any:
        return (await self.call_timed(action, **kwargs))[0]

    async def close(self) -> None:
        await self.inner.close()


async def timed_call(endpoint: Any, action: str, kwargs: dict[str, Any]) -> tuple[Any, float, float]:
    inner = getattr(endpoint, "call_timed", None)
    if inner is not None:
        return await inner(action, **kwargs)
    t0 = time.perf_counter()
    out = await endpoint.call(action, **kwargs)
    return out, time.perf_counter() - t0, 0.0


def delayed(fed: Federation, net: NetworkCfg | None) -> Federation:
    if net is None or not net.active:
        return fed
    if net.down or net.deadline_ms:
        raise ValueError("network: site failures (down) and per-round deadlines are not emulated")
    fed.endpoints = {s: DelayedEndpoint(e, net) for s, e in fed.endpoints.items()}
    return fed
