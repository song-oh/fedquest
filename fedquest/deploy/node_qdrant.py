# a node's own Qdrant server, built at start-up for its slot's sites
from __future__ import annotations

import http.client
import logging
import os
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from fedquest.config import FederationCfg, load_federation

log = logging.getLogger(__name__)

HOST = "127.0.0.1"
VERSION = "v1.19.1"
READY_S = 60.0
POLL_S = 0.25
STOP_S = 30.0
BUILD_THREADS = 4
_RUNNING: list[NodeQdrant] = []


def enabled() -> bool:
    return os.environ.get("FEDQUEST_NODE_QDRANT") == "1"


def binary() -> Path:
    default = Path.home() / ".local" / f"qdrant-{VERSION}" / "qdrant"
    return Path(os.environ.get("FEDQUEST_QDRANT_BIN") or default).expanduser()


def storage_root(machine: str) -> Path:
    tmp = os.environ.get("TMPDIR") or "/tmp"
    return Path(os.environ.get("FEDQUEST_NODE_QDRANT_DIR") or Path(tmp) / f"fedquest-qdrant-{machine}").expanduser()


def _bindable(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((HOST, port))
        except OSError:
            return False
    return True


def free_port() -> int:
    for _ in range(100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((HOST, 0))
            port = s.getsockname()[1]
        # Qdrant's gRPC port is the HTTP port plus one, so both must be free
        if port < 65535 and _bindable(port + 1):
            return port
    raise RuntimeError("no two consecutive free ports on 127.0.0.1")


def answers(url: str, timeout: float = 5.0) -> bool:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(f"{url}/", timeout=timeout) as reply:
            return 200 <= reply.status < 300
    except (OSError, http.client.HTTPException):
        return False


def _proxied() -> str | None:
    proxies = urllib.request.getproxies()
    if proxies.get("http") and not urllib.request.proxy_bypass_environment(HOST, proxies):
        return proxies["http"]
    return None


class NodeQdrant:
    def __init__(self, machine: str, binary_path: Path | None = None, root: Path | None = None) -> None:
        self.machine = machine
        self.binary = Path(binary_path) if binary_path else binary()
        self.root = Path(root) if root else storage_root(machine)
        self.dir: Path | None = None
        self.port: int | None = None
        self.proc: subprocess.Popen[bytes] | None = None
        self.collections: dict[str, dict[str, int]] = {}
        self.build_s: float | None = None
        self.built_s: dict[str, float] = {}
        self.slices: dict[str, str] = {}

    @property
    def url(self) -> str:
        return f"http://{HOST}:{self.port}"

    def start(self, ready_s: float = READY_S) -> str:
        if not (self.binary.is_file() and os.access(self.binary, os.X_OK)):
            raise RuntimeError(f"FEDQUEST_NODE_QDRANT: no Qdrant binary at {self.binary}: set FEDQUEST_QDRANT_BIN, or "
                               f"install Qdrant {VERSION} there (`bash scripts/qdrant.sh start` on a login node does)")
        self.root.mkdir(parents=True, exist_ok=True)
        self.dir = Path(tempfile.mkdtemp(prefix="run-", dir=self.root))
        try:
            self.port = free_port()
            env = {**os.environ, "QDRANT__SERVICE__HOST": HOST, "QDRANT__SERVICE__HTTP_PORT": str(self.port),
                   "QDRANT__SERVICE__GRPC_PORT": str(self.port + 1),
                   "QDRANT__STORAGE__STORAGE_PATH": str(self.dir / "storage"),
                   "QDRANT__STORAGE__SNAPSHOTS_PATH": str(self.dir / "snapshots"),
                   "QDRANT__TELEMETRY_DISABLED": "true", "QDRANT_INIT_FILE_PATH": str(self.dir / ".qdrant-initialized")}
            with open(self.dir / "qdrant.log", "wb") as out:
                self.proc = subprocess.Popen([str(self.binary)], cwd=self.dir, env=env, stdin=subprocess.DEVNULL,
                                             stdout=out, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + ready_s
            while True:
                time.sleep(POLL_S)
                if self.proc.poll() is not None:
                    raise RuntimeError(f"Qdrant ({self.binary}) exited with code {self.proc.returncode} during start-up "
                                       f"on {socket.gethostname()}; the end of its log:\n{self._tail()}")
                if answers(self.url) and self.proc.poll() is None:
                    break
                if time.monotonic() > deadline:
                    raise RuntimeError(f"Qdrant ({self.binary}, pid {self.proc.pid}) did not answer at {self.url} "
                                       f"within {ready_s:.0f} s on {socket.gethostname()}; the end of its log:\n"
                                       f"{self._tail()}")
        except BaseException:
            self.stop()
            raise
        log.info("node Qdrant of %s on %s: %s (pid %d, storage %s)", self.machine, socket.gethostname(), self.url,
                 self.proc.pid, self.dir)
        return self.url

    def stop(self) -> None:
        proc, self.proc = self.proc, None
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(STOP_S)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            log.info("node Qdrant of %s stopped (pid %d)", self.machine, proc.pid)
        if self.dir is not None:
            shutil.rmtree(self.dir, ignore_errors=True)
            self.dir = None
        if self in _RUNNING:
            _RUNNING.remove(self)

    def describe(self) -> dict[str, Any]:
        out = {"url": self.url, "pid": self.proc.pid if self.proc else None, "dir": str(self.dir) if self.dir else None,
               "collections": self.collections, "build_s": self.build_s}
        if self.slices:
            seen: set[str] = set()
            out["slices"] = {}
            for key, ns in self.slices.items():
                out["slices"][key] = {"namespace": ns, "points": sum(self.collections.get(ns, {}).values()),
                                      "build_s": self.built_s.get(ns), "shared": ns in seen}
                seen.add(ns)
        return out

    def _tail(self, lines: int = 20) -> str:
        try:
            text = (self.dir / "qdrant.log").read_text(errors="replace") if self.dir else ""
        except OSError:
            text = ""
        return "\n".join(text.splitlines()[-lines:]) or "(empty)"


def index_source(cfg: FederationCfg, site: str) -> tuple[FederationCfg, str]:
    federation, source = cfg.site_source(site)
    if federation == cfg.name or (cfg.qdrant.approximate and not cfg.site(site).data_from):
        out = (cfg, site)
    else:
        out = (load_federation(federation), source)
    if out[0].index_namespace(out[1]) != cfg.index_namespace(site):
        raise ValueError(f"{cfg.name}/{site} searches {cfg.index_namespace(site)}, which no `fedquest data index` "
                         f"builds (an index variant of a site reading another's data): a node cannot build it")
    return out


def _virtual(cfg: FederationCfg, site: str) -> bool:
    from fedquest.vdb import shards

    scfg = cfg.site(site)
    return any(shards.virtual(cfg, scfg, c) is not None for c in scfg.collections)


@dataclass(frozen=True)
class Job:
    cfg: FederationCfg
    site: str
    shard: bool = False

    @property
    def namespace(self) -> str:
        return self.cfg.index_namespace(self.site)


def index_jobs(keys: list[str]) -> list[Job]:
    jobs: dict[str, Job] = {}
    for key in keys:
        federation, site = key.split("/", 1)
        cfg = load_federation(federation)
        job = Job(cfg, site, shard=True) if _virtual(cfg, site) else Job(*index_source(cfg, site))
        for c in (cfg, job.cfg):
            mode = c.qdrant.resolved().mode
            if mode != "server":
                raise ValueError(f"FEDQUEST_NODE_QDRANT=1 serves {key} from this node's Qdrant server, but {c.name} "
                                 f"runs Qdrant in mode {mode!r} here (FEDQUEST_QDRANT_MODE, else its qdrant.mode)")
        jobs.setdefault(job.namespace, job)
    return list(jobs.values())


def slice_namespaces(keys: list[str]) -> dict[str, str]:
    out = {}
    for key in keys:
        federation, site = key.split("/", 1)
        out[key] = load_federation(federation).index_namespace(site)
    return out


def build_threads() -> int:
    return max(1, int(os.environ.get("FEDQUEST_NODE_BUILD_THREADS") or BUILD_THREADS))


_VECTORS: dict[str, np.ndarray] = {}
_VECTORS_LOCK = threading.Lock()


def _parent_vectors(cdir: Path, vectors: str) -> np.ndarray:
    from fedquest.data.embed import check_vectors

    key = str(cdir.resolve())
    with _VECTORS_LOCK:
        if key not in _VECTORS:
            check_vectors(cdir, vectors)
            _VECTORS[key] = np.load(cdir / "emb.npy", mmap_mode="r")
        return _VECTORS[key]


def build_shard(cfg: FederationCfg, site: str, grpc: bool = True) -> dict[str, int]:
    from fedquest.site.service import virtual_site
    from fedquest.vdb.qdrant_index import SiteIndex

    scfg = cfg.site(site)
    vs = virtual_site(cfg, scfg)
    index = SiteIndex(cfg.site_data_dir(site), cfg.qdrant, namespace=cfg.index_namespace(site), grpc=grpc)
    counts: dict[str, int] = {}
    try:
        for coll in scfg.collections:
            emb = _parent_vectors(vs.parent.collection_data_dir(vs.site, coll.name), coll.vectors)
            rows = vs.rows[coll.name]
            vectors = np.asarray(emb[rows], dtype=np.float32) if len(rows) else np.zeros((0, emb.shape[1]), np.float32)
            counts[coll.name] = index.build(coll.name, vectors, skip=vs.skip[coll.name], wait=False)
        if index.builds_graph:
            for name, points in counts.items():
                index.wait_ready(name, points)
    finally:
        index.close()
    log.info("%s/%s: its own collections %s (from %s/%s's vectors)", cfg.name, site, counts, vs.parent.name,
             vs.site)
    return counts


def _build(job: Job, grpc: bool) -> tuple[dict[str, int], float]:
    from fedquest.data.embed import build_index

    t = time.perf_counter()
    counts = build_shard(job.cfg, job.site, grpc) if job.shard else build_index(job.cfg, job.site, record=False,
                                                                                 grpc=grpc)
    return counts, round(time.perf_counter() - t, 1)


def serve_slices(machine: str, keys: list[str], ready_s: float = READY_S) -> NodeQdrant:
    if _RUNNING:
        raise RuntimeError(f"a node Qdrant already runs in this process (for {_RUNNING[0].machine}): "
                           "FEDQUEST_NODE_QDRANT=1 takes one agent per process")
    jobs = index_jobs(keys)
    slices = slice_namespaces(keys)
    proxy = _proxied()
    if proxy:
        raise RuntimeError(f"FEDQUEST_NODE_QDRANT: requests for http://{HOST} would go to the proxy {proxy} (no_proxy "
                           f"does not list {HOST}), not to this node's Qdrant: add {HOST} to no_proxy and NO_PROXY "
                           "(deploy/globus/aurora_cn/user_config_template.yaml.j2 does)")
    server = NodeQdrant(machine)
    url = server.start(ready_s)
    _RUNNING.append(server)
    before = os.environ.get("FEDQUEST_QDRANT_URL")
    try:
        os.environ["FEDQUEST_QDRANT_URL"] = url
        t0 = time.perf_counter()
        pool = ThreadPoolExecutor(build_threads(), thread_name_prefix=f"qdrant-build-{machine}")
        try:
            futures = {pool.submit(_build, job, True): job for job in sorted(jobs, key=lambda j: j.shard)}
            for fut in as_completed(futures):
                job = futures[fut]
                server.collections[job.namespace], server.built_s[job.namespace] = fut.result()
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
            with _VECTORS_LOCK:
                _VECTORS.clear()
        server.build_s = round(time.perf_counter() - t0, 1)
        server.slices = slices
    except BaseException:
        server.stop()
        if before is None:
            os.environ.pop("FEDQUEST_QDRANT_URL", None)
        else:
            os.environ["FEDQUEST_QDRANT_URL"] = before
        raise
    log.info("node Qdrant of %s holds %s (built in %.1f s)", machine, server.collections, server.build_s)
    for key, row in server.describe().get("slices", {}).items():
        log.info("node Qdrant of %s: %s searches %s, %d points%s", machine, key, row["namespace"], row["points"],
                 " (built for an earlier slice)" if row["shared"] else f", built in {row['build_s']:.1f} s")
    return server
