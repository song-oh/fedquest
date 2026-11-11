from __future__ import annotations

import pickle
import re
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_WATCH: ContextVar[list[AuthExpired] | None] = ContextVar("fedquest_auth_watch", default=None)
_AUTH_ERROR = re.compile(r"\bAuthExpired\b|\bAuthenticationError\b|Error code: 401\b|ALCF login required")


class AuthExpired(RuntimeError):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)
        seen = _WATCH.get()
        if seen is not None:
            seen.append(self)


@contextmanager
def auth_watch() -> Iterator[list[AuthExpired]]:
    seen: list[AuthExpired] = []
    token = _WATCH.set(seen)
    try:
        yield seen
    finally:
        _WATCH.reset(token)


_NETWORK_ERROR = re.compile(r"ClientConnectorDNSError|ClientPayloadError|ConnectionResetError|ServerDisconnectedError|"
                            r"Cannot connect to host|Temporary failure in name resolution|nodename nor servname")


def network_error(error: str | None) -> bool:
    return bool(error) and _NETWORK_ERROR.search(error) is not None


def auth_error(error: str | None) -> bool:
    return bool(error) and _AUTH_ERROR.search(error) is not None


class ReplyTooLarge(RuntimeError):
    pass


class SiteRetryable(RuntimeError):
    pass


def wire_error(e: Exception, machine: str, action: str) -> Exception:
    from fedquest.vdb.qdrant_index import transient

    detail = " ".join(str(e).split())[:600]
    named = f"{action} at {machine}: {type(e).__module__}.{type(e).__name__}: {detail}"
    if transient(e):
        return SiteRetryable(named)
    try:
        if type(pickle.loads(pickle.dumps(e))) is type(e):
            return e
    except Exception:
        pass
    return RuntimeError(named)
