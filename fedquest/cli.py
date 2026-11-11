from __future__ import annotations

import asyncio
import atexit
import json
import logging
import os
import signal
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from fedquest.paths import REPO, data_root

app = typer.Typer(help="FedQuest: multi-hop question answering over federated vector databases.",
                  no_args_is_help=True, add_completion=False)
data_app = typer.Typer(help="Download, prepare, embed and index benchmark data.", no_args_is_help=True)
llm_app = typer.Typer(help="Inference endpoints: login and health check.", no_args_is_help=True)
kg_app = typer.Typer(help="Local Neo4j and Qdrant services and the knowledge graph.", no_args_is_help=True)
deploy_app = typer.Typer(help="The two-facility deployment: site servers behind Globus Compute.", no_args_is_help=True)
eval_app = typer.Typer(help="Experiments and scoring.", no_args_is_help=True)
for sub, name in ((data_app, "data"), (llm_app, "llm"), (kg_app, "kg"), (deploy_app, "remote"), (eval_app, "eval")):
    app.add_typer(sub, name=name)
console = Console()


@app.callback()
def main(verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Debug logging.")] = False) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format="%(message)s",
                        handlers=[RichHandler(console=console, show_path=False, rich_tracebacks=True)])
    for noisy in ("httpx", "httpx2", "urllib3", "neo4j", "sentence_transformers", "globus_sdk", "globus_compute_sdk"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    signal.signal(signal.SIGINT, _interrupt)


def _die_by_sigint() -> None:
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    os.kill(os.getpid(), signal.SIGINT)


def _interrupt(signum: int, frame: Any) -> None:
    if getattr(_interrupt, "pressed", False):
        _die_by_sigint()
    _interrupt.pressed = True  # type: ignore[attr-defined]
    atexit.register(_die_by_sigint)
    console.print("[yellow]stopping (Ctrl-C again to quit at once)")
    raise KeyboardInterrupt


def _csv(value: str | None) -> list[str] | None:
    return [v for v in (value or "").split(",") if v] or None


@data_app.command("download", help="Download a benchmark's raw files.")
def data_download(
    dataset: Annotated[str, typer.Argument(help="mmqa, mramg, musique, hotpotqa, twowiki, hotpotqa-wiki, msmarco "
                                                  "or m3sciqa")],
    images: Annotated[bool, typer.Option(help="Also fetch images.")] = True,
    image_scope: Annotated[str, typer.Option(help="MMQA: all (2.36 GB zip) or dev (dev question pools only).")] = "all",
    subsets: Annotated[str, typer.Option(help="MRAMG subsets whose images to fetch, comma-separated.")] = "manual,arxiv,wiki",
    captions: Annotated[bool, typer.Option(help="MMQA: also fetch MGRAG's image captions.")] = True,
) -> None:
    from fedquest.data import download

    if dataset == "mmqa":
        root = download.download_mmqa(images=images, image_scope=image_scope)
        if captions:
            download.download_mgrag_captions()
    elif dataset == "mramg":
        root = download.download_mramg(_csv(subsets) or [], images=images)
    elif dataset == "musique":
        root = download.download_musique()
    elif dataset in ("hotpotqa", "twowiki"):
        root = download.download_hipporag(dataset)
    elif dataset == "hotpotqa-wiki":
        root = download.download_hotpot_wiki()
    elif dataset == "msmarco":
        root = download.download_msmarco()
    elif dataset == "m3sciqa":
        root = download.download_m3sciqa(images=images)
    else:
        raise typer.BadParameter("dataset must be mmqa, mramg, musique, hotpotqa, twowiki, hotpotqa-wiki, msmarco or "
                                 "m3sciqa")
    console.print(f"[green]{dataset} ready in {root}")


@data_app.command("prepare", help="Write a federation's questions and each site's objects.")
def data_prepare(federations: list[str], sites: Annotated[str | None, typer.Option(help="Only these sites.")] = None) -> None:
    from fedquest.config import load_federation
    from fedquest.data.prepare import prepare_federation

    for fed in federations:
        m = prepare_federation(load_federation(fed), _csv(sites))
        console.print(f"[green]{fed}[/]: {m['n_test']} test / {m['n_calib']} calibration questions; "
                      f"families {m['test_families']}; objects {json.dumps(m['objects'])}"
                      + "".join(f"; {k} {json.dumps(m[k])}" for k in ("splits", "corpus") if k in m))


@data_app.command("embed", help="Embed each site's objects with the site's own encoder.")
def data_embed(federation: str, sites: str | None = None, device: str = "auto", batch_size: int = 256,
               force: bool = False) -> None:
    from fedquest.config import load_federation
    from fedquest.data.embed import embed_federation

    embed_federation(load_federation(federation), _csv(sites), device, batch_size, force)


@data_app.command("derive", help="Copy a federation's vectors from its parent collections.")
def data_derive(federation: str,
                sites: Annotated[str | None, typer.Option(help="Only these sites, comma-separated.")] = None,
                device: str = "auto", batch_size: int = 256, force: bool = False) -> None:
    from fedquest.data.embed import derive_federation

    out = derive_federation(federation, _csv(sites), device=device, batch_size=batch_size, force=force)
    console.print_json(json.dumps(out, default=str))


@data_app.command("index", help="Load a site's vectors into its Qdrant.")
def data_index(federation: str, sites: str | None = None) -> None:
    from fedquest.config import load_federation
    from fedquest.data.embed import build_index

    cfg = load_federation(federation)
    for site in cfg.own_sites(_csv(sites), "index"):
        console.print(f"{federation}/{site}: {build_index(cfg, site)}")


@data_app.command("splits", help="Write the question splits and check them against configs/splits.")
def data_splits(check: Annotated[bool, typer.Option(
        help="Only check that every recorded split file holds the bytes configs/splits/splits.sha256 "
             "records.")] = False
                ) -> None:
    from fedquest.data import splits

    if check:
        bad = splits.check_splits()
        for line in bad:
            console.print(f"[red]{line}")
        if bad:
            raise typer.Exit(1)
        console.print(f"[green]{splits.SHA_FILE}[/]: every split file matches")
        return
    for row in splits.write_splits():
        how = "drawn now" if row["written"] else "kept (already written)"
        console.print(f"{row['federation']}/{row['split']}: {row['questions']} questions, {how}")
    for fed, split in splits.PLAN:
        if split == "final_test":
            found = splits.ground_truth(fed, splits.existing(fed, split) or [])
            console.print(f"{fed}/final_test ground truth in the corpus: {json.dumps(found)}")
    console.print(f"[green]{splits.SHA_FILE}")


@data_app.command("status", help="Objects, vectors and indexed points of every site.")
def data_status(federation: str) -> None:
    from fedquest.config import load_federation

    cfg = load_federation(federation)
    table = Table("site", "collection", "encoder", "objects", "embedded", "indexed")
    for s in cfg.sites:
        index = None
        try:
            from fedquest.vdb.qdrant_index import SiteIndex

            index = SiteIndex(cfg.site_data_dir(s.name), cfg.qdrant, cfg.index_namespace(s.name))
        except Exception:
            index = None
        for c in s.collections:
            d = cfg.collection_data_dir(s.name, c.name)
            objs = 0
            if (d / "objects.jsonl").exists():
                with open(d / "objects.jsonl") as f:
                    objs = sum(1 for _ in f)
            emb = json.loads((d / "emb.json").read_text())["count"] if (d / "emb.json").exists() else "-"
            try:
                indexed = index.count(c.name) if index else "?"
            except Exception:
                indexed = "?"
            table.add_row(s.name, c.name, c.encoder, str(objs), str(emb), str(indexed))
        if index:
            index.close()
    console.print(table)


@data_app.command("fetch-models", help="Download the encoders' weights.")
def data_fetch_models(encoders: Annotated[list[str] | None, typer.Argument()] = None) -> None:
    from fedquest.vdb.encoders import fetch_models

    fetch_models(encoders or ["bge-base", "e5-base", "minilm", "clip"])


@llm_app.command("auth", help="Log in to the inference service.")
def llm_auth(download_only: bool = typer.Option(False, help="Only fetch the ALCF auth script.")) -> None:
    from fedquest.llm.auth import auth_script_path, download_auth_script

    path = auth_script_path()
    if not path.exists():
        download_auth_script(path)
    if download_only:
        console.print(f"ALCF auth script at {path}")
        return
    subprocess.run(["python", str(path), "authenticate"], check=True)


CHECK_PROBE_S = 120.0


def _no_reply(exc: BaseException) -> str:
    if isinstance(exc, TimeoutError):
        return f"no reply in {CHECK_PROBE_S:.0f}s"
    body = getattr(exc, "body", None)
    status = getattr(exc, "status_code", None)
    message = (body.get("message") if isinstance(body, dict) else None) or exc
    return (f"HTTP {status}: {message}" if status else f"{type(exc).__name__}: {message}")[:100]


@llm_app.command("check", help="Check that every configured model answers.")
def llm_check(
    call: Annotated[bool, typer.Option(help="Also send one tiny live request per configured model.")] = True,
    wait: Annotated[float, typer.Option(help="Minutes to keep retrying, every minute, until every model the runs "
                                             "need answers (0: one try).")] = 0.0,
) -> None:
    import contextlib

    import requests
    from rich.markup import escape

    from fedquest.config import ROLES
    from fedquest.llm.client import LLMClient, is_retriable

    client = LLMClient.from_env()
    cfg = client.cfg
    if cfg.provider != "alcf":
        console.print(f"provider is {cfg.provider}: nothing to check")
        return
    try:
        token = client.tokens.get()
    except RuntimeError as e:
        console.print(f"[red]{escape(str(e))}")
        raise typer.Exit(1) from e
    root, _, served = cfg.base_url.partition("/resource_server/")
    root, cluster = root + "/resource_server", served.split("/")[0]
    client.cfg = cfg.model_copy(update={"max_retries": 1})
    required = {model for role, model in cfg.roles.items() if role in ROLES}
    replies: dict[str, str] = {}
    answered: set[str] = set()
    hopeless: set[str] = set()

    async def probe(role: str, model: str) -> None:
        try:
            r = await asyncio.wait_for(client.chat([{"role": "user", "content": "Reply with the word OK."}], role=role,
                                                   task="check", max_tokens=256, live=True, fallback=False),
                                       CHECK_PROBE_S)
        except Exception as e:
            replies[model] = _no_reply(e)
            if not is_retriable(e):
                hopeless.add(model)
        else:
            answered.add(model)
            replies[model] = f"{r.text.strip()[:20]!r} ({r.live_seconds or r.usage.seconds:.1f}s)"

    async def tiny_calls() -> None:
        deadline, until = time.monotonic() + wait * 60, time.strftime("%H:%M", time.localtime(time.time() + wait * 60))
        try:
            while True:
                pending = {}
                for role, model in cfg.roles.items():
                    if model not in answered:
                        pending.setdefault(model, role)
                await asyncio.gather(*(probe(role, model) for model, role in pending.items()))
                missing = sorted(required - answered)
                if not missing or hopeless & set(missing) or time.monotonic() + 60 > deadline:
                    return
                console.print(f"{time.strftime('%H:%M')} no reply from {', '.join(missing)} "
                              f"({escape(replies[missing[0]])}): trying again every minute until {until}")
                await asyncio.sleep(60)
        finally:
            await client.aclose()

    if call:
        asyncio.run(tiny_calls())
    headers = {"Authorization": f"Bearer {token}"}

    def body(path: str) -> Any:
        try:
            return requests.get(f"{root}/{path}", headers=headers, timeout=60).json()
        except (requests.RequestException, ValueError) as e:
            console.print(f"[yellow]no {path} from {root} ({type(e).__name__})")
            return {}

    def clusters_of(model: str) -> list[str]:
        own = cfg.endpoints.get(model)
        base = own.base_url if own is not None else cfg.base_url
        urls = base.removeprefix("pool:").split(",") if base.startswith("pool:") else [base]
        return [u.partition("/resource_server/")[2].split("/")[0] or cluster for u in urls]

    listed = body("list-endpoints")
    wanted = sorted({c for model in cfg.roles.values() for c in clusters_of(model)})
    models: dict[str, set[str]] = {c: set() for c in wanted}
    state: dict[tuple[str, str], str] = {}
    for c in wanted:
        with contextlib.suppress(KeyError, TypeError, AttributeError):
            models[c] = {m for fw in listed["clusters"][c]["frameworks"].values() for m in fw.get("models") or []}
        jobs = body(f"{c}/jobs")
        with contextlib.suppress(TypeError, AttributeError):
            for kind in ("queued", "running"):
                for j in jobs.get(kind) or []:
                    for m in str(j.get("Models", "")).split(","):
                        if state.get((c, m.strip())) != "running":
                            state[(c, m.strip())] = str(j.get("Model Status") or kind)
    table = Table("role", "model", "cluster", "listed", "job", "reply")
    for role, model in cfg.roles.items():
        where = clusters_of(model)
        on = [c for c in where if model in models.get(c, set())]
        jobs_ = [state.get((c, model), "none") for c in where]
        job = "running" if "running" in jobs_ else next((j for j in jobs_ if j != "none"), "none")
        reply = escape(replies.get(model, ""))
        table.add_row(role if role in ROLES else f"{role} (optional)", model, "+".join(where),
                      "yes" if len(on) == len(where) else (f"[yellow]{'+'.join(on)}" if on else "[red]no"),
                      job if job == "running" else f"[yellow]{escape(job)}",
                      reply if model in answered else f"[red]{reply}")
    console.print(table)
    if not call:
        return
    if optional := sorted(set(cfg.roles.values()) - required - answered):
        console.print(f"[yellow]no reply from {', '.join(optional)}: only an optional role (judge2: `fedquest eval "
                      f"score --judge2`) uses it")
    if missing := sorted(required - answered):
        console.print(f"[red]no reply from {', '.join(missing)} (on "
                      f"{escape(', '.join('+'.join(clusters_of(m)) for m in missing))}): no run can answer without it. "
                      f"`fedquest llm check --wait 120` retries until it answers; ALCF support: support@alcf.anl.gov")
        raise typer.Exit(1)
    console.print("[green]every model the runs need answers")


NEO4J_PORT, NEO4J_HTTP_PORT, QDRANT_PORT = 7688, 7475, 6340
PROJECT_LABEL = "org.fedquest.project=fedquest"
CONTAINERS = ("fedquest-neo4j", "fedquest-qdrant")
ENV_FILE = REPO / ".fedquest.env"


def _docker(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["docker", *args], capture_output=True, text=True)


def _env_file() -> dict[str, str]:
    path = ENV_FILE
    if not path.exists():
        return {}
    pairs = (ln.split("=", 1) for ln in path.read_text().splitlines() if "=" in ln and not ln.startswith("#"))
    return {k.strip(): v.strip() for k, v in pairs}


def _set_env(key: str, value: str) -> None:
    path = ENV_FILE
    lines = [ln for ln in (path.read_text().splitlines() if path.exists() else []) if not ln.startswith(f"{key}=")]
    path.write_text("\n".join([*lines, f"{key}={value}"]) + "\n")
    path.chmod(0o600)


def _neo4j_password(store: Path) -> str:
    import secrets

    password = _env_file().get("NEO4J_PASSWORD")
    if password:
        return password
    found = _docker("inspect", "-f", "{{json .Config.Env}}", "fedquest-neo4j")
    if found.returncode == 0:
        for item in json.loads(found.stdout) or []:
            if item.startswith("NEO4J_AUTH=neo4j/"):
                return item.removeprefix("NEO4J_AUTH=neo4j/")
    if any((store / "neo4j").glob("*")):
        raise typer.BadParameter(f"{store}/neo4j holds a Neo4j database whose password is unknown (not in "
                                 ".fedquest.env, no fedquest-neo4j container to read it from); "
                                 "`fedquest kg down --purge` starts fresh")
    return secrets.token_urlsafe(18)


def _wait_ready(url: str, name: str, timeout_s: float = 90.0) -> None:
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            with urllib.request.urlopen(url, timeout=3):
                return
        except OSError as e:
            if time.monotonic() >= deadline:
                raise typer.BadParameter(f"{name} did not answer {url} within {timeout_s:.0f}s; "
                                         f"see `docker logs {name}`") from e
            time.sleep(1)


@kg_app.command("up", help="Start Neo4j and Qdrant in Docker.")
def kg_up() -> None:
    store = data_root()
    password = _neo4j_password(store)
    _set_env("NEO4J_PASSWORD", password)
    specs = {
        "fedquest-neo4j": ["-p", f"127.0.0.1:{NEO4J_PORT}:7687", "-p", f"127.0.0.1:{NEO4J_HTTP_PORT}:7474",
                              "-e", f"NEO4J_AUTH=neo4j/{password}", "-e", "NEO4J_dbms_usage__report_enabled=false",
                              "-e", "NEO4J_server_memory_heap_max__size=2g", "-v", f"{store}/neo4j:/data",
                              "neo4j:5.26-community"],
        "fedquest-qdrant": ["-p", f"127.0.0.1:{QDRANT_PORT}:6333", "-p", f"127.0.0.1:{QDRANT_PORT + 1}:6334",
                               "-e", "QDRANT__TELEMETRY_DISABLED=true", "-v", f"{store}/qdrant:/qdrant/storage",
                               "qdrant/qdrant:v1.19.1"],
    }
    for name, args in specs.items():
        found = _docker("inspect", "-f", '{{index .Config.Labels "org.fedquest.project"}} {{.State.Running}}', name)
        if found.returncode == 0:
            project, running = (found.stdout.split() + ["", ""])[:2]
            if project != "fedquest":
                raise typer.BadParameter(f"a container named {name} exists but is not FedQuest's; remove or rename it")
            if running != "true":
                started = _docker("start", name)
                if started.returncode != 0:
                    raise typer.BadParameter(f"{name}: {started.stderr.strip()}")
            console.print(f"{name}: running")
            continue
        res = _docker("run", "-d", "--name", name, "--label", PROJECT_LABEL, *args)
        if res.returncode != 0:
            raise typer.BadParameter(f"{name}: {res.stderr.strip()}")
        console.print(f"{name}: created (data in {store})")
    _wait_ready(f"http://127.0.0.1:{QDRANT_PORT}/readyz", "fedquest-qdrant")
    _wait_ready(f"http://localhost:{NEO4J_HTTP_PORT}/", "fedquest-neo4j")
    _set_env("NEO4J_URI", f"bolt://localhost:{NEO4J_PORT}")
    _set_env("FEDQUEST_QDRANT_URL", f"http://127.0.0.1:{QDRANT_PORT}")
    console.print(f"Neo4j bolt://localhost:{NEO4J_PORT} (browser http://localhost:{NEO4J_HTTP_PORT}, user neo4j, "
                  f"password in .fedquest.env); Qdrant http://127.0.0.1:{QDRANT_PORT}")


@kg_app.command("down", help="Stop the Docker services.")
def kg_down(purge: bool = typer.Option(False, help="Also delete the containers and their data (clean slate).")) -> None:
    for name in CONTAINERS:
        console.print(f"{name}: {'stopped' if _docker('stop', name).returncode == 0 else 'not running'}")
    if purge and typer.confirm(f"Delete {', '.join(CONTAINERS)} and {data_root()}/neo4j, {data_root()}/qdrant?"):
        import shutil

        for name in CONTAINERS:
            _docker("rm", "-f", name)
        for sub in ("neo4j", "qdrant"):
            shutil.rmtree(data_root() / sub, ignore_errors=True)
        console.print("purged; `fedquest kg up` starts from empty databases")


@kg_app.command("reset", help="Empty the knowledge graph.")
def kg_reset() -> None:
    import shutil

    store = data_root()
    _docker("rm", "-f", "fedquest-neo4j")
    src = store / "neo4j"
    if src.exists() and any(src.iterdir()):
        dst = store / "archive" / f"neo4j_{time.strftime('%Y%m%d_%H%M%S')}"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        console.print(f"archived {src} -> {dst}")
    kg_up()


@kg_app.command("stats", help="Facts per namespace.")
def kg_stats(ns: Annotated[str | None, typer.Option(help="Namespace (default: all).")] = None) -> None:
    from fedquest.kg.view import open_view

    async def go() -> dict:
        view = open_view("neo4j")
        try:
            return await view.stats(ns)
        finally:
            await view.close()

    console.print_json(json.dumps(asyncio.run(go())))


@kg_app.command("export", help="Write one namespace's facts as JSON.")
def kg_export(ns: str, out: Path) -> None:
    from fedquest.kg.neo4j_store import Neo4jKG
    from fedquest.kg.view import open_view

    async def go() -> dict:
        view = open_view("neo4j")
        assert isinstance(view, Neo4jKG)
        try:
            return await view.export_graph(ns)
        finally:
            await view.close()

    graph = asyncio.run(go())
    out.write_text(json.dumps(graph, indent=1))
    console.print(f"{len(graph['nodes'])} entities, {len(graph['edges'])} facts -> {out}")


@kg_app.command("clear", help="Delete namespaces.")
def kg_clear(namespaces: list[str]) -> None:
    from fedquest.kg.view import open_view

    async def go() -> None:
        view = open_view("neo4j")
        try:
            for ns in namespaces:
                await view.clear(ns)
        finally:
            await view.close()

    asyncio.run(go())
    console.print(f"cleared {', '.join(namespaces)}")


@deploy_app.command("up", help="Wait until every site server of the deployment runs.")
def remote_up(wait: Annotated[float, typer.Option(help="Seconds to wait: the batch jobs' queue waits, then the "
                                                       "servers' Qdrant builds.")] = 5400.0) -> None:
    from fedquest.remote.gateway import up

    raise typer.Exit(asyncio.run(up(wait)))


@deploy_app.command("status", help="Health of the gateway and the site servers.")
def remote_status() -> None:
    from fedquest.remote.gateway import status

    raise typer.Exit(asyncio.run(status()))


@deploy_app.command("down", help="Stop the site servers.")
def remote_down() -> None:
    from fedquest.remote.gateway import down

    raise typer.Exit(asyncio.run(down()))


@app.command("calibrate", help="Fit the evidence probabilities from probes on the calibration split.")
def calibrate(federation: Annotated[list[str] | None, typer.Argument(
                  help="The federation (layout) to probe or refit; with --inputs, more observation files.")] = None,
              mode: str = "local", limit: int | None = None,
              refit: Annotated[bool, typer.Option(help="Refit from the probes the last calibration recorded.")] = False,
              questions: Annotated[int | None, typer.Option(
                  help="With --refit: fit on the first N calibration questions only, as <federation>@N.")] = None,
              k: Annotated[int | None, typer.Option(
                  help="Probe depth, at most 80: deeper than 5 also fits deep rank buckets and the coverage at "
                       "every depth up to k.")] = None,
              portable: Annotated[bool, typer.Option(
                  help="With --refit --out: cells without site or collection levels, read the same on "
                       "any layout; the read costs are copied next to --out.")] = False,
              exclude_sites: Annotated[str | None, typer.Option(
                  help="With --portable: sites whose candidates the chances leave out, comma-separated.")] = None,
              out: Annotated[str | None, typer.Option(
                  help="With --portable: the file to write, e.g. configs/calibration/mmqa.json; with --reliability: "
                       "the JSON to write.")] = None,
              probe: Annotated[str | None, typer.Option(
                  help="The value keys' lists, topk or semijoin (each site ranks and counts the "
                       "objects that name the value).")] = None,
              limits: Annotated[str | None, typer.Option(
                  help="A vector list's size per site, fixed (k) or share (the site's share of its "
                       "(collection, encoder) group's k).")] = None,
              graph: Annotated[str | None, typer.Option(
                  help="The query graph's front end, mgrag or typed.")] = None,
              mention: Annotated[int | None, typer.Option(
                  help="The mention test the probes run, 1 or 2 (qualifier-free: "
                       "FedQuest's mention).")] = None,
              layouts: Annotated[str | None, typer.Option(
                  help="More layouts (federations) to probe the same way, comma-separated, e.g. "
                       "mmqa_s4_r99: each writes its own observations.")] = None,
              label: Annotated[str | None, typer.Option(
                  help="With --refit --portable --out: v4, calibration v4 pooled from --inputs.")] = None,
              inputs: Annotated[list[str] | None, typer.Option(
                  help="With --label v4 or --reliability: observation files (repeat, comma-separate, or list them "
                       "after it).")] = None,
              reliability: Annotated[str | None, typer.Option(
                  help="A calibration (name or path) to score on --inputs' labeled candidates, per layout: bins of "
                       "chance deciles, held-out log-likelihood.")] = None,
              split: Annotated[str, typer.Option(
                  help="The split whose questions the probes run (2Wiki and HotpotQA probe test and fit once per "
                       "half: fedquest.xfit).")] = "calib",
              ) -> None:
    from fedquest.eval.calibrate import calibrate as run
    from fedquest.eval.calibrate import refit_calibration

    names = list(federation or [])
    files = [f for value in inputs or [] for f in _csv(value) or []]
    shape: dict[str, Any] = {n: v for n, v in (("probe", probe), ("limits", limits), ("graph", graph),
                                               ("mention", mention)) if v is not None}
    allowed: dict[str, tuple[Any, ...]] = {"probe": ("topk", "semijoin"), "limits": ("fixed", "share"),
                                           "graph": ("mgrag", "typed"), "mention": (1, 2)}
    for n, v in shape.items():
        if v not in allowed[n]:
            raise typer.BadParameter(f"--{n} {v!r}: one of {', '.join(map(str, allowed[n]))}")
    if k is not None and not 1 <= k <= 80:
        raise typer.BadParameter(f"--k {k}: a probe depth from 1 to 80")
    if reliability is not None:
        from fedquest.planner.calibration import reliability as score

        files += names
        if not files or refit or portable or label or shape or layouts or k or questions or exclude_sites:
            raise typer.BadParameter("--reliability <calibration> scores observation files: --inputs <obs...> "
                                     "[--out <file.json>] and nothing else")
        report = score(reliability, files)
        for name, r in [*report["layouts"].items(), ("all", report["all"])]:
            if r["n"]:
                fw = r["first_witness"]
                console.print(f"{name}: {r['n']} labeled, {r['evidence']} evidence; mean log-likelihood "
                              f"{r['mean_log_lik']:.4f}, ECE {r['ece']:.4f}, Brier {r['brier']:.4f}; tokens to the "
                              f"first witness {fw['tokens']:.1f} ({fw['tokens_uniform']:.1f} with equal chances) over "
                              f"{fw['units']} units, {fw['found']:.3f} with one", markup=False, highlight=False)
        if out:
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            Path(out).write_text(json.dumps(report, indent=1))
            console.print(f"[green]{out}")
        return
    if label is not None or files:
        files += names if files else []
        if label != "v4" or not (refit and portable and out) or questions or k or shape or layouts \
                or any(not f.endswith(".jsonl") for f in files) or (not files and len(names) != 1):
            raise typer.BadParameter("calibration v4 refits observations into a file: --refit --portable --label v4 "
                                     "(--inputs <obs...> | <federation>) --out <file> [--exclude-sites a,b]")
        from fedquest.eval.calibrate import refit_portable
        from fedquest.planner.calibration import Calibrator

        path = refit_portable(None if files else names[0], out, _csv(exclude_sites) or (), label="v4",
                              inputs=files or None)
        cal = Calibrator.load(path)
        console.print(f"{path} (+ {path.with_suffix('.costs.json').name if cal.costs else 'no costs'}): v4, "
                      f"strength {cal.strength:g} of {sorted(cal.cross_fit)}; {len(cal.counts)} alpha cells; "
                      f"depth {cal.depth}; {cal.meta['fitted']['labeled']} labeled candidates, "
                      f"{cal.meta['fitted']['evidence']} evidence, layouts "
                      f"{sorted({x for i in cal.meta['inputs'] for x in i['layouts']})}", markup=False,
                      highlight=False)
        return
    if len(names) != 1:
        raise typer.BadParameter("give one federation (more layouts: --layouts a,b)")
    federation_name = names[0]
    if portable or exclude_sites or out:
        if not (refit and portable and out) or questions or k or shape or layouts:
            raise typer.BadParameter("a portable calibration refits into a file: --refit --portable --out <file> "
                                     "[--exclude-sites a,b] (no --questions, no --k)")
        from fedquest.eval.calibrate import refit_portable
        from fedquest.planner.calibration import Calibrator

        path = refit_portable(federation_name, out, _csv(exclude_sites) or ())
        cal = Calibrator.load(path)
        fitted = cal.meta["fitted"]
        console.print(f"{path} (+ {path.with_suffix('.costs.json').name if cal.costs else 'no costs'}): "
                      f"strength {cal.strength:g} of {sorted(cal.cross_fit)}; {len(cal.counts)} alpha cells; "
                      f"{fitted['labeled']} labeled candidates, {fitted['evidence']} evidence "
                      f"(of {cal.meta['labeled']} / {cal.meta['evidence']}; excluded {cal.excluded or 'none'}); "
                      f"coverage {json.dumps({k: c for k, c in cal.cover.items() if '|' in k})}; "
                      f"object tokens {json.dumps(cal.snapshot()['tokens'])}", markup=False, highlight=False)
        return
    extra: dict[str, Any] = {**({"k": k} if k else {}), **shape}
    if questions and not refit:
        raise typer.BadParameter("--questions refits: give --refit too")
    if refit and (shape or layouts):
        raise typer.BadParameter("--probe, --limits, --graph, --mention and --layouts probe: a refit reads the "
                                 "recorded probes")
    refit_args = (federation_name, questions) if questions else (federation_name,)
    if refit:
        console.print(f"[green]{refit_calibration(*refit_args)}")
        return
    for name in dict.fromkeys([federation_name, *(_csv(layouts) or [])]):
        console.print(f"[green]{asyncio.run(run(name, mode=mode, limit=limit, **extra, **({'split': split} if split != 'calib' else {})))}")


@eval_app.command("run", help="Run an experiment.")
def eval_run(experiment: str, mode: str = "local",
             methods: Annotated[str | None, typer.Option(
                 help="Only these labels or label bases, e.g. fedquest,mgrag_fed (every mgrag_fed budget).")] = None,
             limit: int | None = None,
             split: Annotated[str | None, typer.Option(
                 help="Override the experiment's question split, e.g. calib.")] = None,
             qids: Annotated[str | None, typer.Option(help="Only these questions (comma-separated ids).")] = None,
             family: Annotated[str | None, typer.Option(
                 help="Only these families among the selected questions, e.g. compose or single,table.")] = None,
             shard: Annotated[str | None, typer.Option(
                 help="Only shard i of n of each label's questions (i/n, by a hash of the question id, after "
                      "the experiment's and the label's own selection; FEDQUEST_SHARD).")] = None,
             resume: bool = True) -> None:
    from fedquest.config import load_experiment
    from fedquest.eval.runner import parse_shard, run_experiment

    if shard:
        parse_shard(shard)
        os.environ["FEDQUEST_SHARD"] = shard
    exp = load_experiment(experiment)
    if split:
        exp = exp.model_validate({**exp.model_dump(), "split": split})
    from fedquest.errors import AuthExpired

    try:
        out = asyncio.run(run_experiment(exp, mode=mode, methods=_csv(methods), limit=limit, resume=resume,
                                         qids=_csv(qids), families=_csv(family)))
    except AuthExpired as e:
        from rich.markup import escape

        console.print(f"[red]{experiment} stopped: {escape(str(e))}")
        raise typer.Exit(3) from None
    console.print(f"[green]rows in {out / 'rows'}")


@eval_app.command("score", help="Score an experiment's answers with the LLM judges.")
def eval_score(experiment: str, judge: bool = True, labels: str | None = None,
               judge2: Annotated[bool, typer.Option(
                   "--judge2", help="Also rate the answers with the second judge (role judge2 of the models config): "
                                    "acc_j2; the first judge's calls stay exactly as without it.")] = False) -> None:
    from fedquest.config import load_experiment, load_models
    from fedquest.eval.score import score_experiment

    if judge2 and "judge2" not in load_models().roles:
        console.print("[red]--judge2: the models config defines no judge2 role (add e.g. `judge2: <model>` under "
                      "roles, or point FEDQUEST_MODELS at a config that has one)")
        raise typer.Exit(1)
    name = load_experiment(experiment).name
    path = asyncio.run(score_experiment(name, judge=judge, labels=_csv(labels), judge2=judge2))
    console.print(f"[green]{path}")


@eval_app.command("characterize", help="Search reach and per-stage token profiles of an experiment's methods.")
def eval_characterize(experiment: str, federation: str = "mmqa_fed", mode: str = "local",
                      limit: int | None = None,
                      split: Annotated[str | None, typer.Option(
                          help="Questions of the late-binding gap (default: the experiment's split).")] = None,
                      ks: Annotated[str | None, typer.Option(
                          help="Also the reach at these depths, e.g. 5,10,20,40 (one probe at the deepest).")]
                      = None,
                      ) -> None:
    from fedquest.config import load_experiment
    from fedquest.eval.characterize import characterize

    exp = load_experiment(experiment)
    depths = tuple(int(x) for x in ks.split(",") if x.strip()) if ks else ()
    out = asyncio.run(characterize(exp.name, federation=federation, mode=mode, limit=limit, split=split or exp.split,
                                   questions_from=exp.federation, **({"ks": depths} if depths else {})))
    console.print_json(json.dumps(out))


hippo_app = typer.Typer(help="HippoRAG 2: offline OpenIE indexes and retrieval recall.", no_args_is_help=True)
app.add_typer(hippo_app, name="hippo")


@hippo_app.command("index", help="Build HippoRAG 2's index of each text collection.")
def hippo_index(
    federation: Annotated[str, typer.Argument(help="The federation whose sites index their text collections.")],
    sites: Annotated[str | None, typer.Option(help="Only these sites (names or patterns), comma-separated.")] = None,
    force: Annotated[bool, typer.Option(
        help="Rebuild from scratch (the OpenIE calls replay from the LLM record/replay cache).")] = False,
    device: Annotated[str, typer.Option(help="The encoders' device for the index's embeddings.")] = "auto",
    encoder: Annotated[str, typer.Option(
        help="The index's encoder at every site (fedquest.site.hippo.HIPPO_ENCODER: BGE, whose scale the official "
             "synonymy threshold assumes; 'own' keeps each collection's encoder).")] = "bge-base",
) -> None:
    from fedquest.errors import AuthExpired
    from fedquest.site.hippo import HippoBuildError, index_federation

    try:
        path, out = asyncio.run(index_federation(federation, sites=_csv(sites), force=force, device=device,
                                                 encoder=None if encoder == "own" else encoder))
    except AuthExpired as e:
        from rich.markup import escape

        console.print(f"[red]hippo index {federation} stopped: {escape(str(e))}")
        raise typer.Exit(3) from None
    except (HippoBuildError, ValueError, KeyError) as e:
        console.print(str(e), style="red", markup=False, highlight=False)
        raise typer.Exit(1) from None
    for name, s in out["sites"].items():
        console.print(f"{name}: {s['passages']} passages, {s['phrases']} phrases, {s['facts']} facts, {s['edges']} "
                      f"edges; OpenIE {s['openie']['calls']} calls ({s['openie']['made']} made, "
                      f"{s['openie']['replayed']} replayed)", markup=False, highlight=False)
    console.print(f"[green]{path}")


@hippo_app.command("openie", help="Run HippoRAG 2's OpenIE over each text collection.")
def hippo_openie(
    federation: Annotated[str, typer.Argument(help="The federation whose text collections' OpenIE this share runs.")],
    part: Annotated[str, typer.Option(help="This share, i/n: the passages at positions p with p % n == i.")],
    concurrency: Annotated[int | None, typer.Option(
        help="Passages in flight (default FEDQUEST_HIPPO_CONCURRENCY, else 160).")] = None,
) -> None:
    from fedquest.errors import AuthExpired
    from fedquest.site.hippo import HippoBuildError, openie_part

    i, sep, n = part.partition("/")
    if not sep or not i.isdigit() or not n.isdigit():
        raise typer.BadParameter(f"--part {part!r}: i/n, e.g. 3/8")
    try:
        out = asyncio.run(openie_part(federation, int(i), int(n), concurrency))
    except AuthExpired as e:
        from rich.markup import escape

        console.print(f"[red]hippo openie {federation} stopped: {escape(str(e))}")
        raise typer.Exit(3) from None
    except (HippoBuildError, ValueError, KeyError) as e:
        console.print(str(e), style="red", markup=False, highlight=False)
        raise typer.Exit(1) from None
    console.print(json.dumps(out), markup=False, highlight=False)


@hippo_app.command("recall", help="Recall of HippoRAG 2's retrieval on the questions' supporting passages.")
def hippo_recall(
    federation: Annotated[str, typer.Argument(help="The federation whose sites' HippoRAG 2 indexes rank.")],
    split: Annotated[str, typer.Option(help="The questions' split.")] = "final_test",
    k: Annotated[int, typer.Option(help="Candidates per collection, from HippoRAG 2 and from the question probe.")] = 5,
    limit: Annotated[int | None, typer.Option(help="Only the first N questions of the split.")] = None,
) -> None:
    from fedquest.errors import AuthExpired
    from fedquest.site.hippo import recall_study

    try:
        path, out = asyncio.run(recall_study(federation, split=split, k=k, limit=limit))
    except AuthExpired as e:
        from rich.markup import escape

        console.print(f"[red]hippo recall {federation} stopped: {escape(str(e))}")
        raise typer.Exit(3) from None
    except (FileNotFoundError, ValueError) as e:
        console.print(str(e), style="red", markup=False, highlight=False)
        raise typer.Exit(1) from None
    for name in ("hipporag2", "dense"):
        console.print(f"{name}: {json.dumps(out[name])}", markup=False, highlight=False)
    for name, did in out["retrievals"].items():
        console.print(f"{name}: {did.get('graph', 0)} of {did.get('retrievals', 0)} ranked by the graph, "
                      f"{did.get('filter_calls', 0)} filter calls", markup=False, highlight=False)
    console.print(f"[green]{path}[/] ({out['questions']} questions, k {k})")


if __name__ == "__main__":
    app()
