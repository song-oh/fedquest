from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from fedquest import budget, timeline
from fedquest.config import POOL_PREFIX, ModelsCfg, load_models, with_profile
from fedquest.errors import AuthExpired
from fedquest.federation.replay import miss, note_llm_key, recording
from fedquest.ledger import Ledger
from fedquest.llm import timing
from fedquest.llm.auth import TokenProvider, relogin
from fedquest.llm.cache import LLMCache, cache_key, canonical_json
from fedquest.llm.fake import FakeBackend
from fedquest.llm.pace import pacer, window
from fedquest.llm.parsing import parse_json
from fedquest.paths import llm_cache_path
from fedquest.types import Usage
from fedquest.utils import truncate

log = logging.getLogger(__name__)

_RETRY_STATUS = {408, 409, 429, 502, 503, 504, 520, 521, 522, 523, 524}
_COMPUTED = ("bytes_in", "bytes_out")
_WRAPPED_4XX = re.compile(r"status code: 4\d\d")
_REFUSED = re.compile(r"high-assurance|permission denied from internal policies", re.I)


class Backend(Protocol):
    async def complete(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int,
        temperature: float,
        json_mode: bool,
        task: str,
    ) -> tuple[str, dict[str, int], str | None, float]:
        ...

    async def aclose(self) -> None: ...


@dataclass
class LLMReply:
    text: str
    usage: Usage
    cached: bool
    model: str
    finish_reason: str | None = None
    live_seconds: float | None = None
    wall: float | None = None
    samples: int | None = None


@dataclass
class CallStats:
    wall: float = 0.0
    retries: int = 0
    backoff: float = 0.0
    http429: int = 0
    http5xx: int = 0
    in_flight: int = 0
    errors: list[str] = field(default_factory=list)
    service: float | None = None


def _retry_after(exc: BaseException) -> float:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) or {}
    try:
        return float(headers.get("retry-after") or 0)
    except (TypeError, ValueError):
        return 0.0


def _flag(name: str) -> bool:
    return os.environ.get(name, "") == "1"


def is_retriable(exc: BaseException) -> bool:
    import openai

    if isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError, asyncio.TimeoutError)):
        return True
    if isinstance(exc, openai.APIStatusError):
        if exc.status_code in _RETRY_STATUS:
            return True
        if exc.status_code == 500:
            return not _WRAPPED_4XX.search(f"{exc} {exc.body}")
    return False


def pool_urls(name: str) -> list[str]:
    if "://" in name:
        return [u.strip() for u in name.split(",") if u.strip()]
    var = "FEDQUEST_POOL_" + re.sub(r"[^A-Za-z0-9]", "_", name).upper()
    listed = os.environ.get(var, "")
    urls = [u.strip() for u in listed.split(",") if u.strip()]
    if not urls and os.environ.get("FEDQUEST_POOL_FILE"):
        import json

        path = Path(os.environ["FEDQUEST_POOL_FILE"])
        if path.exists():
            urls = [str(u) for u in (json.loads(path.read_text()).get(name) or [])]
    if not urls:
        raise RuntimeError(f"model pool {name!r} has no replicas: set {var} (comma-separated base URLs) or list them "
                           "under that name in FEDQUEST_POOL_FILE (scripts/hippo/serve.sh writes it)")
    return urls


class ReplicaPool:
    PENALTY_S = 20.0

    def __init__(self, name: str, urls: list[str] | None, open_client: Any) -> None:
        self.name, self._open = name, open_client
        self.urls: list[str] = []
        self.clients: list[Any] = []
        self.in_flight: list[int] = []
        self.down_until: list[float] = []
        self.calls: list[int] = []
        if urls is not None:
            self._resolve(urls)

    def _resolve(self, urls: list[str]) -> None:
        self.urls = list(urls)
        self.clients = [self._open(u) for u in self.urls]
        self.in_flight = [0] * len(self.urls)
        self.down_until = [0.0] * len(self.urls)
        self.calls = [0] * len(self.urls)

    def pick(self) -> int:
        if not self.urls:
            self._resolve(pool_urls(self.name))
        now = time.monotonic()
        live = [i for i in range(len(self.urls)) if self.down_until[i] <= now] or list(range(len(self.urls)))
        least = min(self.in_flight[i] for i in live)
        return random.choice([i for i in live if self.in_flight[i] == least])

    def failed(self, i: int, exc: BaseException) -> None:
        status = getattr(exc, "status_code", None)
        if status is None or (isinstance(status, int) and status >= 500):
            self.down_until[i] = time.monotonic() + self.PENALTY_S


class OpenAIBackend:
    def __init__(self, cfg: ModelsCfg, tokens: TokenProvider) -> None:
        self.cfg, self.tokens = cfg, tokens
        self.pools: list[ReplicaPool] = []
        self.client = self._server(cfg.base_url, tokens)
        servers: dict[tuple[Any, ...], tuple[Any, TokenProvider]] = {
            (cfg.provider, cfg.base_url, cfg.api_key_env): (self.client, tokens)}
        self.routes: dict[str, tuple[Any, TokenProvider]] = {}
        for model, e in cfg.endpoints.items():
            server = (e.provider, e.base_url, e.api_key_env)
            if server not in servers:
                own = TokenProvider(e.provider, cfg.token_refresh_s, e.api_key_env)
                servers[server] = (self._server(e.base_url, own), own)
            self.routes[model] = servers[server]
        self._clients = [client for client, _ in servers.values() if not isinstance(client, ReplicaPool)]

    def _server(self, base_url: str, tokens: TokenProvider) -> Any:
        if not base_url.startswith(POOL_PREFIX):
            return self._open(base_url, tokens)
        pool = ReplicaPool(base_url.removeprefix(POOL_PREFIX), None, lambda url: self._open(url, tokens))
        self.pools.append(pool)
        return pool

    def _open(self, base_url: str, tokens: TokenProvider) -> Any:
        from openai import AsyncOpenAI

        async def api_key() -> str:
            return await asyncio.to_thread(tokens.get)

        return AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=self.cfg.timeout_s, max_retries=0)

    async def complete(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int,
        temperature: float,
        json_mode: bool,
        task: str,
        n: int = 1,
    ) -> tuple[str, dict[str, int], str | None, float]:
        kwargs: dict[str, Any] = {"model": model, "messages": messages, "max_tokens": max_tokens,
                                  "temperature": temperature}
        if n > 1:
            kwargs["n"] = n
        if json_mode and model in self.cfg.json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        extra = self.cfg.extra_body.get(model)
        if extra:
            kwargs["extra_body"] = dict(extra)
        client, tokens = self.routes.get(model, (self.client, self.tokens))
        pool, replica = (client, client.pick()) if isinstance(client, ReplicaPool) else (None, -1)
        if pool is not None:
            client = pool.clients[replica]
            pool.in_flight[replica] += 1
            pool.calls[replica] += 1
        t0 = time.perf_counter()
        try:
            resp = await client.chat.completions.create(**kwargs)
        except Exception as exc:
            if getattr(exc, "status_code", None) == 401:
                tokens.invalidate()
            if pool is not None:
                pool.failed(replica, exc)
            raise
        finally:
            if pool is not None:
                pool.in_flight[replica] -= 1
        wall = time.perf_counter() - t0
        choice = resp.choices[0]
        text = choice.message.content or ""
        if n > 1:
            text = json.dumps([c.message.content or "" for c in resp.choices], ensure_ascii=False)
        usage: dict[str, int] = {}
        if resp.usage is not None:
            usage["prompt_tokens"] = resp.usage.prompt_tokens or 0
            usage["completion_tokens"] = resp.usage.completion_tokens or 0
            details = getattr(resp.usage, "completion_tokens_details", None)
            usage["reasoning_tokens"] = getattr(details, "reasoning_tokens", 0) or 0
        service = (resp.model_extra or {}).get("response_time")
        return text, usage, choice.finish_reason, float(service) if service else wall

    async def aclose(self) -> None:
        for client in [*self._clients, *(c for pool in self.pools for c in pool.clients)]:
            await client.close()


class ReplayBackend:
    def __init__(self, cfg: ModelsCfg, cache: Path | None = None) -> None:
        self.cfg, self.cache = cfg, cache

    async def complete(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int,
        temperature: float,
        json_mode: bool,
        task: str,
        n: int = 1,
    ) -> tuple[str, dict[str, int], str | None, float]:
        key = cache_key(model, messages, max_tokens, temperature, json_mode, _keyed_extra(self.cfg, model, n))
        raise miss(f"no recorded reply for the {task!r} call to {model} (LLM cache key {key}) in "
                   f"{self.cache or 'no cache'}")

    async def aclose(self) -> None:
        return None


class LLMClient:
    def __init__(
        self,
        cfg: ModelsCfg,
        *,
        cache: LLMCache | None,
        tokens: TokenProvider,
        backend: Backend | None = None,
        timed: bool | None = None,
        live: bool | None = None,
        endpoint_s: float | None = None,
    ) -> None:
        self.cfg = cfg
        self.cache = cache
        self.tokens = tokens
        self.timed = _flag("FEDQUEST_LLM_TIMED") if timed is None else timed
        self.live = _flag("FEDQUEST_LLM_LIVE") if live is None else live
        self.endpoint_s = float(os.environ.get("FEDQUEST_ENDPOINT_MS") or 0) / 1000 if endpoint_s is None \
            else endpoint_s
        self.in_flight = 0
        self._borrowed = False
        if backend is None:
            backend = FakeBackend() if cfg.provider == "fake" else \
                ReplayBackend(cfg, cache.path if cache is not None else None) if cfg.provider == "replay" else \
                OpenAIBackend(cfg, tokens)
        self.backend = backend
        self._global = asyncio.Semaphore(cfg.max_concurrency)
        self._per_model: dict[str, asyncio.Semaphore] = {}

    @classmethod
    def from_env(
        cls,
        *,
        cache_path: Path | None = None,
        models_path: Path | None = None,
        backend: Backend | None = None,
        profile: str | None = None,
        provider: str | None = None,
    ) -> LLMClient:
        cfg = load_models(models_path, profile)
        if provider:
            cfg = cfg.model_copy(update={"provider": provider})
        cache = LLMCache(cache_path or llm_cache_path())
        return cls(cfg, cache=cache, tokens=TokenProvider(cfg.provider, cfg.token_refresh_s, cfg.api_key_env),
                   backend=backend)

    def for_profile(self, profile: str | None) -> LLMClient:
        if not profile:
            return self
        cfg = with_profile(self.cfg, profile)
        backend = self.backend if cfg.provider == "fake" and isinstance(self.backend, FakeBackend) else None
        return LLMClient(cfg, cache=LLMCache(self.cache.path) if self.cache is not None else None,
                         tokens=TokenProvider(cfg.provider, cfg.token_refresh_s, cfg.api_key_env), backend=backend,
                         timed=self.timed, live=self.live, endpoint_s=self.endpoint_s)

    def clone(self) -> LLMClient:
        out = LLMClient(self.cfg, cache=LLMCache(self.cache.path) if self.cache is not None else None,
                        tokens=self.tokens, backend=self.backend, timed=self.timed, live=self.live,
                        endpoint_s=self.endpoint_s)
        out._borrowed = True
        return out

    @property
    def trust(self) -> str:
        return self.cfg.trust

    def model(self, role: str) -> str:
        return self.cfg.roles[role]

    def _base_url(self, model: str) -> str:
        own = self.cfg.endpoints.get(model)
        return own.base_url if own is not None else self.cfg.base_url

    def _model_sem(self, model: str) -> asyncio.Semaphore:
        if model not in self._per_model:
            self._per_model[model] = asyncio.Semaphore(self.cfg.model_concurrency.get(model, self.cfg.max_concurrency))
        return self._per_model[model]

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        role: str,
        task: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        json_mode: bool = False,
        ledger: Ledger | None = None,
        charge: str | None = None,
        live: bool | None = None,
        fallback: bool = True,
        n: int = 1,
    ) -> LLMReply:
        model = self.model(role)
        if ledger is not None and ledger.budget is not None:
            budget.check(ledger, messages, task=task, max_tokens=max_tokens, charge=charge)
        max_tokens += self.cfg.headroom.get(model, 0)
        live = self.live if live is None else live
        t0 = time.perf_counter()
        key = cache_key(model, messages, max_tokens, temperature, json_mode, _keyed_extra(self.cfg, model, n))
        if recording():
            await asyncio.to_thread(note_llm_key, key)
        hit = await asyncio.to_thread(self.cache.get, key) if self.cache is not None else None
        stats = CallStats()
        if hit is not None:
            reply = LLMReply(hit["reply"], Usage.from_dict(hit["usage"]), True, model, hit["usage"].get("finish_reason"))
            if live:
                try:
                    _, _, _, served = await self._call(model, messages, max_tokens, temperature, json_mode, task,
                                                       stats, n)
                except Exception as exc:
                    if not fallback or isinstance(exc, AuthExpired):
                        raise
                    log.warning("live call to %s failed (%s: %s): its recorded reply stands in", model,
                                type(exc).__name__, exc)
                else:
                    reply.live_seconds = reply.wall = stats.wall + stats.backoff
                    reply.cached = False
                    timing.note("llm:" + key, reply.wall, served, live=True, task=task, in_flight=stats.in_flight)
            elif self.timed:
                logged = timing.lookup("llm:" + key)
                await self.hold_and_sleep(logged if logged is not None else reply.usage.seconds + self.endpoint_s,
                                          model=model)
                stats.service = reply.usage.seconds
        else:
            text, usage, finish, seconds = await self._call(model, messages, max_tokens, temperature, json_mode, task,
                                                            stats, n)
            timing.note("llm:" + key, stats.wall + stats.backoff, seconds, task=task, in_flight=stats.in_flight)
            u = Usage(
                prompt_tokens=usage.get("prompt_tokens") or _approx(messages),
                completion_tokens=usage.get("completion_tokens") or max(1, len(text) // 4),
                reasoning_tokens=usage.get("reasoning_tokens", 0),
                seconds=seconds,
                calls=1,
            )
            reply = LLMReply(text, u, False, model, finish)
            reply.wall = stats.wall + stats.backoff
            if live:
                reply.live_seconds = reply.wall
            if self.cache is not None:
                record = {k: v for k, v in u.to_dict().items() if k not in _COMPUTED} | {"finish_reason": finish}
                await asyncio.to_thread(self.cache.put, key, model, text, record)
                first = await asyncio.to_thread(self.cache.get, key)
                if first is not None and (first["reply"], first["usage"]) != (text, record):
                    reply = LLMReply(first["reply"], Usage.from_dict(first["usage"]), False, model,
                                     first["usage"].get("finish_reason"), reply.live_seconds, reply.wall)
        sent = canonical_json(messages)
        reply.usage.bytes_in = len(sent.encode())
        reply.usage.bytes_out = len(reply.text.encode())
        if ledger is not None and charge is not None:
            ledger.charge_llm(charge, reply.usage)
            ledger.charge_model_path(stats.wall, stats.retries, stats.backoff, stats.http429, stats.http5xx)
            if reply.live_seconds is not None:
                ledger.charge_live(charge, reply.live_seconds)
            if reply.cached:
                ledger.coord_replays += 1
            if not _has_image(messages):
                ledger.note_prompt(role, _text(messages), reply.usage.prompt_tokens)
        timeline.note_llm(task, charge, t0, stats.service, stats.wall + stats.backoff, reply.cached)
        return reply

    async def hold_and_sleep(self, seconds: float, *, model: str | None = None, role: str | None = None) -> None:
        model = model or (self.model(role) if role else None)
        if model is None:
            async with self._global:
                await asyncio.sleep(max(0.0, seconds))
            return
        async with self._model_sem(model), self._global:
            await asyncio.sleep(max(0.0, seconds))

    async def chat_n(
        self,
        messages: list[dict[str, Any]],
        n: int,
        *,
        role: str,
        task: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        ledger: Ledger | None = None,
        charge: str | None = None,
    ) -> tuple[list[str], LLMReply]:
        if n < 1:
            raise ValueError(f"n {n!r} must be at least 1")
        reply = await self.chat(messages, role=role, task=task, max_tokens=max_tokens, temperature=temperature,
                                ledger=ledger, charge=charge, n=n)
        texts = _samples(reply.text) if n > 1 else [reply.text]
        reply.samples = len(texts)
        for i in range(len(texts), n):
            try:
                more = await self.chat(messages, role=role, task=task, max_tokens=max_tokens,
                                       temperature=temperature + 1e-3 * i, ledger=ledger, charge=charge)
            except budget.BudgetExhausted:
                break
            texts.append(more.text)
        return texts[:n], reply

    async def chat_json(
        self,
        messages: list[dict[str, Any]],
        *,
        role: str,
        task: str,
        max_tokens: int = 1024,
        ledger: Ledger | None = None,
        charge: str | None = None,
        default: Any = None,
    ) -> tuple[Any, LLMReply]:
        reply = await self.chat(messages, role=role, task=task, max_tokens=max_tokens, json_mode=True,
                                ledger=ledger, charge=charge)
        obj = parse_json(reply.text)
        if obj is None:
            retry = [*messages, {"role": "assistant", "content": reply.text},
                     {"role": "user", "content": "Return valid JSON only."}]
            reply = await self.chat(retry, role=role, task=task, max_tokens=max_tokens, json_mode=True,
                                    ledger=ledger, charge=charge)
            obj = parse_json(reply.text)
        return (default if obj is None else obj), reply

    async def _call(
        self, model: str, messages: list[dict[str, Any]], max_tokens: int, temperature: float, json_mode: bool, task: str,
        stats: CallStats | None = None, n: int = 1,
    ) -> tuple[str, dict[str, int], str | None, float]:
        stats = stats if stats is not None else CallStats()
        attempt, refreshed = 0, False
        while True:
            try:
                async with self._model_sem(model), self._global:
                    remote = self.cfg.provider not in ("fake", "replay")
                    win = window(self._base_url(model), model, self.cfg.model_concurrency.get(
                        model, self.cfg.max_concurrency)) if remote else None
                    if win is not None:
                        await win.acquire()
                    throttled = ok = False
                    try:
                        paced = pacer() if remote else None
                        if paced is not None:
                            await paced.wait()
                        stats.in_flight = self.in_flight
                        self.in_flight += 1
                        t0 = time.perf_counter()
                        try:
                            more = {"n": n} if n > 1 else {}
                            out = await self.backend.complete(
                                model, messages, max_tokens=max_tokens, temperature=temperature, json_mode=json_mode,
                                task=task, **more
                            )
                            ok = True
                        except Exception as exc:
                            throttled = getattr(exc, "status_code", None) == 429
                            raise
                        finally:
                            stats.wall += time.perf_counter() - t0
                            self.in_flight -= 1
                    finally:
                        if win is not None:
                            await win.release(throttled=throttled, ok=ok)
                    stats.service = out[3]
                    return out
            except AuthExpired:
                raise
            except Exception as exc:
                attempt += 1
                status = getattr(exc, "status_code", None)
                if status == 429:
                    stats.http429 += 1
                elif isinstance(status, int) and status >= 500:
                    stats.http5xx += 1
                if status == 401 and (refreshed or attempt > self.cfg.max_retries
                                      or _REFUSED.search(f"{exc} {getattr(exc, 'body', '')}")):
                    raise self._refused(model, exc, refreshed) from None
                expired = status == 401
                if attempt > self.cfg.max_retries or not (expired or is_retriable(exc)):
                    raise
                if expired:
                    self.tokens.invalidate()
                    refreshed = True
                delay = min(60.0, 2.0 * 2 ** (attempt - 1))
                if status == 429:
                    delay = max(_retry_after(exc), 1.0) + random.random() * min(4.0, attempt)
                else:
                    delay *= 0.5 + random.random() / 2
                log.warning("LLM call to %s failed (%s: %s); retry %d in %.1fs", model, type(exc).__name__, exc, attempt, delay)
                stats.retries += 1
                stats.backoff += delay
                await asyncio.sleep(delay)

    def _refused(self, model: str, exc: Exception, refreshed: bool) -> AuthExpired:
        own = self.cfg.endpoints.get(model)
        provider, key_env = (own.provider, own.api_key_env) if own else (self.cfg.provider, self.cfg.api_key_env)
        body = getattr(exc, "body", None)
        said = body.get("message") if isinstance(body, dict) else None
        again = " again, with a fresh token," if refreshed else ""
        return AuthExpired(f"the {provider} endpoint refused this machine's login{again} for {model} (HTTP 401: "
                           f"{truncate(' '.join(str(said or exc).split()), 200)}). {relogin(provider, key_env)}")

    async def aclose(self) -> None:
        if not self._borrowed:
            await self.backend.aclose()
        if self.cache is not None:
            self.cache.close()


def _keyed_extra(cfg: ModelsCfg, model: str, n: int) -> dict[str, Any] | None:
    extra = cfg.extra_body.get(model)
    return extra if n <= 1 else {**(extra or {}), "n": int(n)}


def _samples(text: str) -> list[str]:
    try:
        out = json.loads(text)
    except (TypeError, ValueError):
        return [text]
    return [str(t) for t in out] if isinstance(out, list) else [text]


def _has_image(messages: list[dict[str, Any]]) -> bool:
    return any(isinstance(m.get("content"), list) and any(isinstance(p, dict) and p.get("type") == "image_url"
                                                          for p in m["content"]) for m in messages)


def _text(messages: list[dict[str, Any]]) -> str:
    from fedquest.llm.fake import message_text

    return "\n".join(f"{m.get('role', '')}: {message_text([m])}" for m in messages)


def _approx(messages: list[dict[str, Any]]) -> int:
    from fedquest.llm.fake import message_text

    return max(1, len(message_text(messages)) // 4)
