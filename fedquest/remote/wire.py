# length-prefixed pickle frames, socket paths and error envelopes
from __future__ import annotations

import asyncio
import os
import pickle
import socket
import struct
from pathlib import Path
from typing import Any

HEADER = struct.Struct(">I")
MAX_FRAME = 1 << 30


def sock_dir() -> Path:
    path = Path(os.environ.get("FEDQUEST_SOCK_DIR") or f"/tmp/fedquest-{os.getuid()}")
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    return path


def server_socket(facility: str) -> Path:
    return sock_dir() / f"site-{facility}.sock"


def gateway_socket() -> Path:
    return Path(os.environ.get("FEDQUEST_GATEWAY_SOCK") or sock_dir() / "gateway.sock")


TOKEN_BYTES = 32


def pool_name() -> str:
    return os.environ.get("FEDQUEST_POOL") or "main"


def gateway_info_path(pool: str | None = None) -> Path:
    from fedquest.paths import runs_root

    path = runs_root() / "remote_registry" / "gateway"
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    return path / f"{pool or pool_name()}.json"


def gateway_tcp() -> tuple[list[str], int, bytes] | None:
    import json

    addr = os.environ.get("FEDQUEST_GATEWAY_ADDR")
    if addr:
        host, _, port = addr.rpartition(":")
        token_file = os.environ.get("FEDQUEST_GATEWAY_TOKEN_FILE")
        if not token_file:
            raise ConnectionError("FEDQUEST_GATEWAY_ADDR needs FEDQUEST_GATEWAY_TOKEN_FILE (the gateway's token)")
        return [host], int(port), Path(token_file).read_text().strip().encode()
    if os.environ.get("FEDQUEST_GATEWAY_TCP") != "1":
        return None
    path = gateway_info_path()
    try:
        info = json.loads(path.read_text())
    except (OSError, ValueError) as e:
        raise ConnectionError(f"no gateway published at {path}: start it on the login node with "
                              f"FEDQUEST_GATEWAY_LISTEN set (bash scripts/remote.sh start)") from e
    return [info["addr"], *(info.get("ips") or [])], int(info["port"]), str(info["token"]).encode()


def pack(obj: Any) -> bytes:
    body = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    return HEADER.pack(len(body)) + body


async def read_frame(reader: asyncio.StreamReader) -> Any | None:
    try:
        head = await reader.readexactly(HEADER.size)
    except asyncio.IncompleteReadError:
        return None
    (n,) = HEADER.unpack(head)
    if n > MAX_FRAME:
        raise ValueError(f"a frame of {n} bytes")
    return pickle.loads(await reader.readexactly(n))


def recv_frame(sock: socket.socket) -> Any:
    head = _recv_exactly(sock, HEADER.size)
    (n,) = HEADER.unpack(head)
    if n > MAX_FRAME:
        raise ValueError(f"a frame of {n} bytes")
    return pickle.loads(_recv_exactly(sock, n))


def _recv_exactly(sock: socket.socket, n: int) -> bytes:
    chunks, got = [], 0
    while got < n:
        chunk = sock.recv(min(n - got, 1 << 20))
        if not chunk:
            raise ConnectionError("the site server closed the connection")
        chunks.append(chunk)
        got += len(chunk)
    return b"".join(chunks)


def call_tcp(addr: str, port: int, token: str, body: dict[str, Any], timeout_s: float) -> Any:
    with socket.create_connection((addr, int(port)), timeout=min(timeout_s, 15.0)) as s:
        s.settimeout(timeout_s)
        s.sendall(pack({"auth": token}) + pack(body))
        return recv_frame(s)


def error_envelope(e: BaseException, retryable: bool | None = None) -> dict[str, Any]:
    from fedquest.errors import SiteRetryable

    if retryable is None:
        retryable = isinstance(e, SiteRetryable | ConnectionError | TimeoutError)
    return {"ok": False, "error": {"type": type(e).__name__, "module": type(e).__module__,
                                   "message": " ".join(str(e).split())[:4000], "retryable": bool(retryable)}}


def rebuild(error: dict[str, Any], where: str) -> Exception:
    from fedquest.errors import AuthExpired, ReplyTooLarge, SiteRetryable

    kind, msg = error.get("type") or "RuntimeError", error.get("message") or ""
    if kind == "AuthExpired":
        return AuthExpired(msg)
    if kind == "ReplyTooLarge":
        return ReplyTooLarge(msg)
    if kind == "SiteRetryable":
        return SiteRetryable(msg)
    if error.get("retryable"):
        return ConnectionError(f"{where}: {kind}: {msg}")
    if kind in ("KeyError", "ValueError", "TypeError"):
        return {"KeyError": KeyError, "ValueError": ValueError, "TypeError": TypeError}[kind](msg)
    return RuntimeError(f"{where}: {error.get('module', '')}.{kind}: {msg}")
