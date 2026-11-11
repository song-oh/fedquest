#!/usr/bin/env python3
# token-guarded CONNECT proxy from Midway3 compute nodes to a few hosts on port 443

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import logging
import os
import secrets
import signal
import time
from collections.abc import Awaitable, Callable
from pathlib import Path

ALLOW = ("exchange.academy-agents.org", "auth.globus.org", "inference-api.alcf.anl.gov")
PORT = "443"
HEAD_S, CONNECT_S = 30.0, 20.0
URL_FILE = "~/.fedquest/fqproxy.url"
log = logging.getLogger("fqproxy")

Connect = Callable[[str, int], Awaitable[tuple[asyncio.StreamReader, asyncio.StreamWriter]]]


def write_secret(path: Path, text: str) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.fchmod(fd, 0o600)
        os.write(fd, text.encode())
    finally:
        os.close(fd)
    os.replace(tmp, path)


def token_for(url_file: Path) -> str:
    path = url_file.with_name("fqproxy.token")
    with contextlib.suppress(FileNotFoundError):
        if token := path.read_text().strip():
            return token
    token = secrets.token_urlsafe(18)
    write_secret(path, token + "\n")
    return token


def withdraw(url_file: Path, url: str) -> None:
    with contextlib.suppress(FileNotFoundError):
        if url_file.read_text().strip() == url:
            url_file.unlink()


async def _pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, count: dict[str, int], key: str) -> None:
    try:
        while data := await reader.read(1 << 16):
            count[key] += len(data)
            writer.write(data)
            await writer.drain()
    except (ConnectionError, OSError):
        pass
    finally:
        if not writer.is_closing():
            writer.close()


async def _refuse(writer: asyncio.StreamWriter, status: str, extra: str = "") -> None:
    writer.write(f"HTTP/1.1 {status}\r\n{extra}Content-Length: 0\r\nConnection: close\r\n\r\n".encode())
    with contextlib.suppress(ConnectionError, OSError):
        await writer.drain()
    writer.close()


def _authorized(header: str, token: str) -> bool:
    scheme, _, credentials = header.strip().partition(" ")
    want = base64.b64encode(f"fq:{token}".encode()).decode()
    return scheme.lower() == "basic" and secrets.compare_digest(credentials.strip().encode(), want.encode())


def handler(token: str, allow: frozenset[str], connect: Connect = asyncio.open_connection):
    async def handle(creader: asyncio.StreamReader, cwriter: asyncio.StreamWriter) -> None:
        peer = (cwriter.get_extra_info("peername") or ("?",))[0]
        t0 = time.monotonic()
        try:
            head = await asyncio.wait_for(creader.readuntil(b"\r\n\r\n"), HEAD_S)
        except (TimeoutError, asyncio.IncompleteReadError, asyncio.LimitOverrunError, ConnectionError):
            cwriter.close()
            return
        lines = head.decode("latin-1").split("\r\n")
        parts = lines[0].split()
        headers = {k.strip().lower(): v.strip() for k, _, v in (ln.partition(":") for ln in lines[1:] if ln)}
        if len(parts) != 3 or parts[0] != "CONNECT":
            log.warning("%s refused: %s", peer, lines[0][:80])
            return await _refuse(cwriter, "405 Method Not Allowed")
        host, _, port = parts[1].rpartition(":")
        if not _authorized(headers.get("proxy-authorization", ""), token):
            log.warning("%s refused: no or wrong token for %s", peer, parts[1])
            return await _refuse(cwriter, "407 Proxy Authentication Required", 'Proxy-Authenticate: Basic realm="fq"\r\n')
        if port != PORT or host.lower() not in allow:
            log.warning("%s refused: target %s not allowed", peer, parts[1])
            return await _refuse(cwriter, "403 Forbidden")
        try:
            ureader, uwriter = await asyncio.wait_for(connect(host, int(PORT)), CONNECT_S)
        except (OSError, TimeoutError) as e:
            log.warning("%s: cannot reach %s (%s)", peer, parts[1], e)
            return await _refuse(cwriter, "502 Bad Gateway")
        cwriter.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
        count = {"up": 0, "down": 0}
        try:
            await cwriter.drain()
            await asyncio.gather(_pipe(creader, uwriter, count, "up"), _pipe(ureader, cwriter, count, "down"))
        finally:
            for w in (uwriter, cwriter):
                if not w.is_closing():
                    w.close()
        log.info("%s %s up=%d down=%d %.1fs", peer, host, count["up"], count["down"], time.monotonic() - t0)

    return handle


async def serve(host: str, port: int, token: str, allow: frozenset[str],
                connect: Connect = asyncio.open_connection) -> asyncio.Server:
    return await asyncio.start_server(handler(token, allow, connect), host, port, limit=1 << 16)


def _more_files() -> None:
    with contextlib.suppress(ImportError, OSError, ValueError):
        import resource

        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard if hard != resource.RLIM_INFINITY else max(soft, 65536), hard))


async def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", required=True, help="the address compute nodes reach (the login node's name: $(hostname))")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--allow", nargs="*", default=list(ALLOW), help="hosts a tunnel may reach (port 443 only)")
    ap.add_argument("--url-file", default=URL_FILE, help="where the proxy URL (with its token) is announced")
    ap.add_argument("--log", default=None, help="log file (default: stderr; its directory is made)")
    args = ap.parse_args(argv)
    if args.log:
        Path(args.log).expanduser().parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    logging.basicConfig(filename=args.log and Path(args.log).expanduser(), level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    _more_files()
    url_file = Path(args.url_file).expanduser()
    token = token_for(url_file)
    server = await serve(args.host, args.port, token, frozenset(h.lower() for h in args.allow))
    port = server.sockets[0].getsockname()[1]
    url = f"http://fq:{token}@{args.host}:{port}"
    write_secret(url_file, url + "\n")
    log.info("fqproxy on %s:%d (pid %d) for %s; URL in %s", args.host, port, os.getpid(), ", ".join(args.allow), url_file)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    try:
        await stop.wait()
    finally:
        withdraw(url_file, url)
        server.close()
        log.info("fqproxy stopped; %s removed", url_file)


if __name__ == "__main__":
    asyncio.run(main())
