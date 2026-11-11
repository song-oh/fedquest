# call pacing, optionally shared across processes by a file, and AIMD concurrency windows
from __future__ import annotations

import asyncio
import fcntl
import os
import struct
import threading
import time
from pathlib import Path

_SLOT = struct.Struct("d")


class RatePacer:
    def __init__(self, rate: float, path: Path | None = None) -> None:
        self.gap = 1.0 / rate if rate > 0 else 0.0
        self.path = path
        self._next = 0.0
        self._lock = threading.Lock()
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch(exist_ok=True)

    def _claim(self) -> float:
        if self.path is None:
            with self._lock:
                now = time.time()
                start = max(now, self._next)
                self._next = start + self.gap
                return start
        try:
            with open(self.path, "r+b") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                try:
                    raw = f.read(_SLOT.size)
                    now = time.time()
                    start = max(now, _SLOT.unpack(raw)[0] if len(raw) == _SLOT.size else 0.0)
                    f.seek(0)
                    f.write(_SLOT.pack(start + self.gap))
                    f.flush()
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)
        except OSError as e:
            import logging

            logging.getLogger(__name__).warning("the shared rate bucket %s fails (%s): pacing within this process",
                                                self.path, e)
            self.path = None
            return self._claim()
        return start

    async def wait(self) -> None:
        if self.gap <= 0:
            return
        start = await asyncio.to_thread(self._claim) if self.path is not None else self._claim()
        delay = start - time.time()
        if delay > 0:
            await asyncio.sleep(delay)


_PACERS: dict[tuple[float, str | None], RatePacer] = {}


def pacer() -> RatePacer | None:
    try:
        rate = float(os.environ.get("FEDQUEST_LLM_RATE") or 0)
    except ValueError:
        rate = 0.0
    if rate <= 0:
        return None
    path = os.environ.get("FEDQUEST_LLM_RATE_FILE") or None
    key = (rate, path)
    if key not in _PACERS:
        _PACERS[key] = RatePacer(rate, Path(path).expanduser() if path else None)
    return _PACERS[key]


class Window:
    HALVE_S = 2.0

    def __init__(self, start: float, hi: int) -> None:
        self.hi = max(1, int(hi))
        self.limit = float(min(max(1.0, start), self.hi))
        self.inflight = 0
        self._cond: asyncio.Condition | None = None
        self._halved = 0.0

    def _c(self) -> asyncio.Condition:
        if self._cond is None:
            self._cond = asyncio.Condition()
        return self._cond

    async def acquire(self) -> None:
        cond = self._c()
        async with cond:
            while self.inflight >= int(self.limit):
                await cond.wait()
            self.inflight += 1

    async def release(self, *, throttled: bool = False, ok: bool = False) -> None:
        cond = self._c()
        async with cond:
            self.inflight -= 1
            now = time.monotonic()
            if throttled and now - self._halved > self.HALVE_S:
                self.limit = max(1.0, self.limit / 2)
                self._halved = now
            elif ok:
                self.limit = min(float(self.hi), self.limit + 1.0 / max(1.0, self.limit))
            cond.notify_all()


_WINDOWS: dict[tuple[int, str, str], Window] = {}


def window(base_url: str, model: str, hi: int) -> Window | None:
    if os.environ.get("FEDQUEST_LLM_AIMD") != "1":
        return None
    try:
        loop = id(asyncio.get_running_loop())
    except RuntimeError:
        return None
    key = (loop, base_url, model)
    if key not in _WINDOWS:
        _WINDOWS[key] = Window(float(os.environ.get("FEDQUEST_LLM_AIMD_START") or 8), hi)
    return _WINDOWS[key]
