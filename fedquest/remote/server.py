from __future__ import annotations

import argparse
import asyncio
import hmac
import json
import logging
import os
import secrets
import signal
import socket
import sys
import time
from pathlib import Path
from typing import Any

from fedquest.remote import wire
from fedquest.remote.host import SiteHost

log = logging.getLogger(__name__)


def facility_keys(facility: str) -> list[str]:
    from fedquest.config import load_remote

    return load_remote().keys_of(facility)


def state_file(facility: str) -> Path:
    return wire.sock_dir() / f"site-{facility}.json"


def _write_state(facility: str, **state: Any) -> None:
    path = state_file(facility)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps({"pid": os.getpid(), "host": socket.gethostname(), **state}))
    tmp.replace(path)


async def serve(facility: str) -> None:
    from fedquest.config import load_remote
    from fedquest.remote import registry

    cfg = load_remote()
    if facility not in cfg.facilities:
        raise SystemExit(f"configs/remote.yaml has no facility {facility!r}")
    nodes = cfg.facilities[facility].nodes
    slot = registry.claim(facility, nodes)
    registry.keep_claim(facility, slot)
    keys = cfg.keys_at(facility, slot)
    _write_state(facility, slot=slot, ready=False, started=time.time())
    os.environ["FEDQUEST_LLM_RATE_FILE"] = str(registry.base(facility) / "llm.rate")
    log.info("site server of %s on %s: slot %d of %d, %d sites", facility, socket.gethostname(), slot, nodes,
             len(keys))
    host = SiteHost(facility, keys, slot=slot)
    await host.startup()
    path = wire.server_socket(facility)
    stop = asyncio.Event()
    token = secrets.token_hex(16)

    async def one(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, auth: bool = False) -> None:
        try:
            if auth:
                first = await wire.read_frame(reader)
                if not (isinstance(first, dict) and hmac.compare_digest(str(first.get("auth", "")), token)):
                    return
            while (request := await wire.read_frame(reader)) is not None:
                t0 = time.perf_counter()
                action = request.get("action")
                if action == "shutdown":
                    reply: dict[str, Any] = {"ok": True, "value": {"stopping": host.where, "slot": slot}}
                    stop.set()
                else:
                    try:
                        value = await host.dispatch(request.get("key"), action, request.get("kwargs") or {})
                        reply = {"ok": True, "value": value}
                    except Exception as e:
                        reply = wire.error_envelope(e)
                reply["served_s"] = round(time.perf_counter() - t0, 4)
                writer.write(wire.pack(reply))
                await writer.drain()
        except (ConnectionError, asyncio.IncompleteReadError, ValueError):
            pass
        finally:
            writer.close()

    tmp = path.with_name(path.name + f".{os.getpid()}")
    tmp.unlink(missing_ok=True)
    server = await asyncio.start_unix_server(one, path=str(tmp), limit=1 << 22, backlog=1024)
    os.chmod(tmp, 0o600)
    tcp = await asyncio.start_server(lambda r, w: one(r, w, auth=True), host=os.environ.get("FEDQUEST_SITE_BIND",
                                     "0.0.0.0"), port=int(os.environ.get("FEDQUEST_SITE_PORT") or 0),
                                     limit=1 << 22, backlog=1024)
    port = tcp.sockets[0].getsockname()[1]
    tmp.replace(path)
    _write_state(facility, slot=slot, ready=True, started=time.time(), sites=len(keys), port=port)
    registry.publish(facility, slot, {"host": registry.host_name(), "addr": registry.address(),
                                      "ips": registry.addresses(), "port": port,
                                      "token": token, "pid": os.getpid(), "sites": len(keys), "nodes": nodes})
    log.info("site server %s (slot %d) listening on %s and TCP port %d", host.where, slot, path, port)
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    try:
        await stop.wait()
    finally:
        registry.unpublish(facility, slot)
        server.close()
        tcp.close()
        path.unlink(missing_ok=True)
        state_file(facility).unlink(missing_ok=True)
        await host.shutdown()
        log.info("site server %s (slot %d) stopped", host.where, slot)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Serve the sites of this node's slot of a facility (fedquest.remote).")
    ap.add_argument("--facility", required=True)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    for noisy in ("httpx", "urllib3", "neo4j", "sentence_transformers", "globus_sdk"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    asyncio.run(serve(args.facility))
    return 0


if __name__ == "__main__":
    sys.exit(main())
