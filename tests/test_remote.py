# remote mode on the toy federation, with threads standing in for Globus Compute
import asyncio
import shutil
import threading
from concurrent.futures import Future
from pathlib import Path

import pytest

import fedquest.config as config_module
from fedquest.llm.client import LLMClient
from fedquest.llm.fake import FakeBackend
from fedquest.paths import CONFIGS
from tests.toy import build_toy, responder, toy_env


@pytest.fixture
def world(tmp_path, monkeypatch):
    with toy_env(tmp_path):
        w = build_toy(tmp_path / "world")
        configs = tmp_path / "configs"
        (configs / "federations").mkdir(parents=True)
        shutil.copy(w.config, configs / "federations" / "toy_fed.yaml")
        shutil.copy(CONFIGS / "models.yaml", configs / "models.yaml")
        (configs / "remote.yaml").write_text(
            "facilities:\n  aurora: {endpoint: ep-aurora}\n  midway3: {endpoint: ep-midway3}\n"
            "  edge: {endpoint: ep-edge}\nfederations: [toy_fed]\nmodels: " + str(configs / "models.yaml") + "\n")
        monkeypatch.setattr(config_module, "CONFIGS", configs)
        monkeypatch.setenv("FEDQUEST_MODELS", str(configs / "models.yaml"))
        monkeypatch.setenv("FEDQUEST_REMOTE_CONFIG", str(configs / "remote.yaml"))
        monkeypatch.setenv("FEDQUEST_SOCK_DIR", str(Path("/tmp") / f"fq-test-{abs(hash(str(tmp_path))) % 10**8}"))
        original = LLMClient.from_env.__func__
        monkeypatch.setattr(LLMClient, "from_env", classmethod(
            lambda cls, **kw: original(cls, **{**kw, "backend": FakeBackend(responder)})))
        import fedquest.remote.gateway as gateway

        monkeypatch.setattr(gateway, "HOLDS", 0)
        yield w
        shutil.rmtree(Path(__import__("os").environ["FEDQUEST_SOCK_DIR"]), ignore_errors=True)


class ThreadExecutor:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.submitted = 0

    def submit(self, fn, *args):
        self.submitted += 1
        fut: Future = Future()
        fut.task_id = f"t{self.submitted}"

        def run():
            try:
                fut.set_result(fn(*args))
            except Exception as e:
                fut.set_exception(e)

        threading.Thread(target=run, daemon=True).start()
        return fut

    def shutdown(self, *a, **k):
        return None


class RefusingExecutor:
    refusals = 0
    made: list[str] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self._stopped = False
        RefusingExecutor.made.append(kwargs.get("label", ""))

    def submit(self, fn, *args):
        if self._stopped:
            raise RuntimeError(f"{self!r} is shutdown; no new functions may be executed")
        fut: Future = Future()
        if RefusingExecutor.refusals > 0:
            RefusingExecutor.refusals -= 1
            fut.task_id = None
            self._stopped = True
            fut.set_exception(RuntimeError("GlobusAPIError (503, SERVICE_UNAVAILABLE): try again later"))
        else:
            fut.task_id = "t1"
            fut.set_result({"ok": True, "value": {"ready": True, "host": "node1", "health": {"sites": 1}}})
        return fut

    def shutdown(self, *a, **k):
        return None


async def _start_gateway(monkeypatch, refusals):
    import fedquest.remote.gateway as gateway
    from fedquest.remote import wire

    RefusingExecutor.refusals, RefusingExecutor.made = refusals, []
    monkeypatch.setattr("globus_compute_sdk.Executor", RefusingExecutor)
    monkeypatch.setattr(gateway, "REBUILD_S", 0.0)
    monkeypatch.setattr(gateway, "_backoff", lambda tries: 0.01)
    gw = gateway.Gateway()
    serving = asyncio.create_task(gw.serve())
    for _ in range(200):
        if wire.gateway_socket().exists():
            break
        await asyncio.sleep(0.05)
    return gw, serving


async def test_a_refused_batch_stops_one_executor_not_the_gateway(world, monkeypatch):
    import fedquest.remote.gateway as gateway

    gw, serving = await _start_gateway(monkeypatch, refusals=1)
    try:
        r = await gateway._ask("aurora", "ready")
        assert r["ok"] and r["value"]["ready"], r
        assert RefusingExecutor.made.count("fedquest-gateway-aurora") == 2
        stats = gw.stats()
        assert "SERVICE_UNAVAILABLE" in stats["last_submission_error"]["aurora"]
        assert stats["counts"]["aurora"]["resubmitted"] == 1
        r = await gateway._ask("aurora", "ready")
        assert r["ok"] and RefusingExecutor.made.count("fedquest-gateway-aurora") == 2
    finally:
        serving.cancel()


async def test_a_call_refused_every_time_fails_with_the_refusal(world, monkeypatch):
    import fedquest.remote.gateway as gateway

    gw, serving = await _start_gateway(monkeypatch, refusals=10**6)
    try:
        r = await gateway._ask("aurora", "ready")
        assert r["ok"] is False and r["error"]["retryable"] is True
        assert "SERVICE_UNAVAILABLE" in r["error"]["message"], r
        assert gw.stats()["counts"]["aurora"]["resubmitted"] == gateway.RESUBMIT - 1
    finally:
        serving.cancel()


async def _start_servers(facilities):
    from fedquest.remote import server

    tasks = {f: asyncio.create_task(server.serve(f)) for f in facilities}
    from fedquest.remote import wire

    for _ in range(400):
        if all(wire.server_socket(f).exists() for f in facilities):
            break
        await asyncio.sleep(0.05)
    return tasks


async def test_the_remote_chain_serves_what_an_in_process_site_serves(world, monkeypatch):
    import fedquest.remote.gateway as gateway
    from fedquest.config import load_remote
    from fedquest.federation.gc import open_gc_federation
    from fedquest.federation.local import open_local_federation
    from fedquest.types import ProbeRequest

    cfg = load_remote()
    assert cfg.facility_of("aurora") == "aurora" and cfg.facility_of("aurora_3") == "aurora"
    assert cfg.facility_of("nowhere_1") is None
    assert sorted(cfg.keys_of("midway3")) == ["toy_fed/midway3"]

    servers = await _start_servers(["aurora", "midway3", "edge"])
    monkeypatch.setattr("globus_compute_sdk.Executor", ThreadExecutor)
    gw = gateway.Gateway()
    serving = asyncio.create_task(gw.serve())
    from fedquest.remote import wire

    for _ in range(200):
        if wire.gateway_socket().exists():
            break
        await asyncio.sleep(0.05)
    try:
        remote = open_gc_federation("toy_fed")
        local = open_local_federation("toy_fed")
        req = ProbeRequest(key="baseball", kind="question", k=3)
        r_remote = await remote.probe([req])
        r_local = await local.probe([req])
        assert [[(c.site, c.oid, c.rank) for c in cs] for cs in r_remote] == \
            [[(c.site, c.oid, c.rank) for c in cs] for cs in r_local]
        assert r_remote[0], "the probe found nothing: the comparison would be empty"
        info = await remote.info()
        assert set(info) == {"aurora", "midway3", "edge"}
        with pytest.raises(ValueError, match="never-registered"):
            await remote.endpoints["aurora"].call("versions", oids=["x"], epoch="never-registered")
        stats = gw.stats()
        assert stats["counts"]["aurora"]["ok"] >= 2
        health = await gateway._ask("aurora", "ready")
        assert health["ok"] and health["value"]["ready"] and health["value"]["health"]["sites"] == 1
        assert health["value"]["slot"] == 0 and set(health["value"]["slots"]) == {0}
        assert await gateway.up(60) == 0
        await local.close()
    finally:
        for f in ("aurora", "midway3", "edge"):
            await gateway._ask(f, "shutdown")
        await asyncio.wait_for(asyncio.gather(*servers.values(), return_exceptions=True), 30)
        serving.cancel()


def test_a_host_reports_the_records_it_would_replay_at_its_start(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from fedquest.llm.cache import LLMCache
    from fedquest.remote.host import records_at_start

    monkeypatch.delenv("FEDQUEST_SITE_STORE_DIR", raising=False)
    assert records_at_start(None) == {"llm_calls": None, "llm_cache": None, "site_stores": None, "store_dir": None}
    store = tmp_path / "site_stores"
    monkeypatch.setenv("FEDQUEST_SITE_STORE_DIR", str(store))
    llm = SimpleNamespace(cache=LLMCache(tmp_path / "llm_aurora_x1.sqlite"))
    empty = records_at_start(llm)
    assert empty["llm_calls"] == 0 and empty["site_stores"] == 0 and empty["llm_cache"].endswith("llm_aurora_x1.sqlite")
    llm.cache.put("k1", "gemma", "a reply", {"prompt_tokens": 3, "completion_tokens": 2})
    (store / "toy_fed" / "sites" / "aurora").mkdir(parents=True)
    (store / "toy_fed" / "sites" / "aurora" / "facts.sqlite").touch()
    assert {k: v for k, v in records_at_start(llm).items() if k in ("llm_calls", "site_stores")} == \
        {"llm_calls": 1, "site_stores": 1}


async def test_fresh_names_every_slot_that_found_records_and_up_names_every_facility_down(monkeypatch, capsys):
    import fedquest.remote.gateway as gateway

    ok = {"llm_calls": 0, "site_stores": 0, "llm_cache": "c", "store_dir": "s"}
    healths = {"aurora": {0: {"host": "x1", "at_start": ok}, 1: {"host": "x2", "at_start": ok}},
               "midway3": {0: {"host": "m1", "at_start": ok}, 1: {"host": "m2", "at_start": {**ok, "llm_calls": 7}}}}
    cfg = type("Cfg", (), {"facilities": {"aurora": None, "midway3": None}})()
    monkeypatch.setattr("fedquest.config.load_remote", lambda *a, **k: cfg)

    async def fake_healths(_cfg):
        return healths

    monkeypatch.setattr(gateway, "_healths", fake_healths)
    assert await gateway.fresh() == 1
    out = capsys.readouterr().out
    assert "PROBLEM: midway3 slot 1 on m2: 7 model calls recorded" in out and out.rstrip().endswith("NOT FRESH")
    healths["midway3"][1] = {"host": "m2", "at_start": {**ok, "store_dir": None, "site_stores": None}}
    assert await gateway.fresh() == 1 and "no host record file or fact store directory" in capsys.readouterr().out
    healths["midway3"][1] = {"host": "m2", "at_start": ok}
    assert await gateway.fresh() == 0 and capsys.readouterr().out.rstrip().endswith("FRESH")

    class Two:
        facilities = {"aurora": type("F", (), {"nodes": 2})(), "midway3": type("F", (), {"nodes": 2})()}

    async def no_reply(facility, action, key=None, kwargs=None, timeout_s=120.0, slot=None):
        return {"ok": False, "error": {"message": "no reply"}}

    async def no_refusals():
        await asyncio.sleep(3600)

    monkeypatch.setattr("fedquest.config.load_remote", lambda *a, **k: Two())
    monkeypatch.setattr(gateway, "_ask", no_reply)
    monkeypatch.setattr(gateway, "_print_refusals", no_refusals)
    assert await gateway.up(0.01) == 1
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("NOT READY: ")]
    assert [ln.split(":")[1].strip() for ln in lines] == ["aurora", "midway3"]


def test_a_task_reports_a_missing_server_as_retryable(world, monkeypatch):
    from fedquest.remote import task

    monkeypatch.setattr(task, "start", lambda facility, env: None)
    out = task.handle({"facility": "aurora", "key": "toy_fed/aurora", "action": "info", "kwargs": {},
                       "start_wait_s": 0.5})
    assert out["ok"] is False and out["error"]["retryable"] is True


def test_a_hold_task_keeps_its_worker_and_starts_no_server(world, monkeypatch):
    import time

    from fedquest.remote import task

    monkeypatch.setattr(task, "start", lambda facility, env: pytest.fail("a hold task starts no server"))
    t0 = time.monotonic()
    out = task.handle({"facility": "aurora", "action": "hold", "kwargs": {"seconds": 0.2}})
    assert out["ok"] and out["value"]["held_s"] == 0.2 and 0.15 < time.monotonic() - t0 < 5 and out["via"]
    monkeypatch.setattr(task, "HOLD_MAX_S", 0.05)
    t0 = time.monotonic()
    assert task.handle({"facility": "aurora", "action": "hold", "kwargs": {"seconds": 1e9}})["ok"]
    assert time.monotonic() - t0 < 5


async def test_the_gateway_keeps_a_hold_task_in_flight_at_every_facility(world, monkeypatch):
    import fedquest.remote.gateway as gateway

    gw = gateway.Gateway()
    calls: list[tuple[str, str, dict]] = []

    async def fake_call(facility, key, action, kwargs, timeout_s, slot=None):
        calls.append((facility, action, kwargs))
        await asyncio.sleep(0.01)
        return {"ok": True, "value": {}}

    monkeypatch.setattr(gw, "call", fake_call)
    loop = asyncio.create_task(gw.keepalive("midway3", 0.0))
    await asyncio.sleep(0.2)
    loop.cancel()
    assert len(calls) >= 3 and all(c == ("midway3", "hold", {"seconds": gateway.HOLD_S}) for c in calls)
    assert gateway.HOLDS >= 0 and gateway.HOLD_S < 300


def test_rebuilt_errors_keep_the_types_the_coordinator_acts_on():
    from fedquest.errors import AuthExpired, ReplyTooLarge, SiteRetryable
    from fedquest.remote import wire

    for e, kind in ((SiteRetryable("busy"), SiteRetryable), (ReplyTooLarge("big"), ReplyTooLarge),
                    (AuthExpired("login"), AuthExpired), (KeyError("k"), KeyError), (ValueError("v"), ValueError)):
        assert isinstance(wire.rebuild(wire.error_envelope(e)["error"], "x"), kind)
    assert isinstance(wire.rebuild(wire.error_envelope(OSError("cut"), retryable=True)["error"], "x"), ConnectionError)
    assert isinstance(wire.rebuild(wire.error_envelope(ZeroDivisionError("z"))["error"], "x"), RuntimeError)


async def test_an_experiment_runs_in_remote_mode_as_in_local_mode(world, monkeypatch):
    import json

    import fedquest.remote.gateway as gateway
    from fedquest.config import ExperimentCfg
    from fedquest.eval.runner import run_experiment
    from fedquest.paths import calibration_path, results_dir
    from fedquest.planner.calibration import Calibrator
    from fedquest.remote import wire
    from fedquest.utils import read_jsonl

    calib = calibration_path("toy_fed")
    calib.parent.mkdir(parents=True, exist_ok=True)
    Calibrator().save(calib)
    calib.with_suffix(".costs.json").write_text(json.dumps({"federation": "toy_fed", "out_empty": 6, "out_hit": 26}))
    methods = [{"method": "fedquest"}, {"method": "vanilla_rag"}, {"method": "mgrag", "label": "mgrag_fed"},
               {"method": "iter_rag", "label": "ircot", "params": {"hops": 2, "top": 2, "k": 2}}]
    servers = await _start_servers(["aurora", "midway3", "edge"])
    monkeypatch.setattr("globus_compute_sdk.Executor", ThreadExecutor)
    gw = gateway.Gateway()
    serving = asyncio.create_task(gw.serve())
    for _ in range(200):
        if wire.gateway_socket().exists():
            break
        await asyncio.sleep(0.05)
    try:
        local = ExperimentCfg(name="toy_local", federation="toy_fed", workload="cold", concurrency=2,
                              matcher={"encoder": "hash"}, methods=methods)
        remote = local.model_copy(update={"name": "toy_remote"})
        await run_experiment(local, mode="local")
        await run_experiment(remote, mode="remote")
        for label in ("fedquest", "vanilla_rag", "mgrag_fed", "ircot"):
            a = {r["qid"]: r for r in read_jsonl(results_dir("toy_local") / "rows" / f"{label}.jsonl")}
            b = {r["qid"]: r for r in read_jsonl(results_dir("toy_remote") / "rows" / f"{label}.jsonl")}
            assert a and set(a) == set(b), label
            for q in a:
                assert b[q]["error"] is None, (label, b[q]["error"])
                assert (a[q]["answer"], a[q]["consulted"]) == (b[q]["answer"], b[q]["consulted"]), label
        assert gw.stats()["counts"]["aurora"]["ok"] > 0 and gw.stats()["counts"]["aurora"]["failed"] == 0
    finally:
        for f in ("aurora", "midway3", "edge"):
            await gateway._ask(f, "shutdown")
        await asyncio.wait_for(asyncio.gather(*servers.values(), return_exceptions=True), 30)
        serving.cancel()


def test_each_node_claims_one_slot_and_a_dead_servers_slot_is_taken_over(tmp_path, monkeypatch):
    import os
    import time

    from fedquest.remote import registry

    monkeypatch.setenv("FEDQUEST_RUNS", str(tmp_path))
    monkeypatch.setenv("FEDQUEST_JOB_ID", "job-1")
    monkeypatch.delenv("FEDQUEST_SLOT", raising=False)
    monkeypatch.setenv("FEDQUEST_HOSTNAME", "a")
    assert registry.claim("aurora", 2) == 0 and registry.claim("aurora", 2) == 0
    monkeypatch.setenv("FEDQUEST_HOSTNAME", "b")
    assert registry.claim("aurora", 2) == 1
    monkeypatch.setenv("FEDQUEST_HOSTNAME", "c")
    with pytest.raises(RuntimeError, match="claimed"):
        registry.claim("aurora", 2)
    old = time.time() - 2 * registry.STALE_S
    os.utime(registry.job_dir("aurora") / "slot0.claim", (old, old))
    assert registry.claim("aurora", 2) == 0
    monkeypatch.setenv("FEDQUEST_JOB_ID", "job-2")
    assert registry.claim("aurora", 2) == 0
    registry.publish("aurora", 0, {"host": "c", "token": "t", "port": 1})
    assert registry.lookup("aurora", 0)["token"] == "t" and "token" not in registry.table("aurora")[0]


def test_a_call_reaches_the_node_its_site_is_placed_on(world, tmp_path, monkeypatch):
    import os
    import subprocess
    import sys
    import time

    from fedquest.paths import REPO
    from fedquest.remote import registry, task, wire

    configs = config_module.CONFIGS
    (configs / "remote.yaml").write_text(
        "facilities:\n  aurora: {endpoint: ep-aurora, config: {nodes: 2}, placement: {toy_fed/aurora: 1}}\n"
        "  midway3: {endpoint: ep-midway3}\n  edge: {endpoint: ep-edge}\nfederations: [toy_fed]\nmodels: "
        + str(configs / "models.yaml") + "\n")
    monkeypatch.setenv("FEDQUEST_JOB_ID", "job-t")
    env = {**os.environ, "FEDQUEST_CONFIGS": str(configs), "FEDQUEST_HOST_ADDR": "127.0.0.1",
           "FEDQUEST_SITE_BIND": "127.0.0.1"}
    procs, dirs = [], {}
    try:
        for slot, node in enumerate(("node-a", "node-b")):
            d = Path("/tmp") / f"fq-{node}-{abs(hash(str(tmp_path))) % 10**8}"
            d.mkdir(mode=0o700, exist_ok=True)
            dirs[node] = d
            log = open(tmp_path / f"{node}.log", "wb")  # noqa: SIM115
            procs.append(subprocess.Popen([sys.executable, "-m", "fedquest.remote.server", "--facility", "aurora"],
                                          cwd=REPO, env={**env, "FEDQUEST_SOCK_DIR": str(d), "FEDQUEST_HOSTNAME": node},
                                          stdout=log, stderr=subprocess.STDOUT))
            for _ in range(600):
                if registry.lookup("aurora", slot) is not None:
                    break
                assert procs[-1].poll() is None, (tmp_path / f"{node}.log").read_text()
                time.sleep(0.1)
            assert registry.lookup("aurora", slot)["host"] == node
        monkeypatch.setenv("FEDQUEST_SOCK_DIR", str(dirs["node-a"]))
        monkeypatch.setenv("FEDQUEST_HOSTNAME", "node-a")

        def call(slot, action, key=None):
            return task.handle({"facility": "aurora", "slot": slot, "key": key, "action": action, "kwargs": {},
                                "env": {}, "timeout_s": 60})

        far = call(1, "health")
        assert far["ok"] and far["value"]["slot"] == 1 and far["value"]["sites"] == 1
        near = call(0, "health")
        assert near["ok"] and near["value"]["slot"] == 0 and near["value"]["sites"] == 0
        info = call(1, "info", "toy_fed/aurora")
        assert info["ok"], info
        wrong = call(0, "info", "toy_fed/aurora")
        assert not wrong["ok"] and "does not host" in wrong["error"]["message"]
        ready = task.handle({"facility": "aurora", "action": "ready", "env": {}})["value"]
        assert ready["ready"] and ready["slot"] == 0 and ready["slots"] == {0: "node-a", 1: "node-b"}
        entry = registry.lookup("aurora", 1)
        with pytest.raises(ConnectionError):
            wire.call_tcp("127.0.0.1", entry["port"], "not-the-token", {"key": None, "action": "health",
                                                                          "kwargs": {}}, 10)
        nobody = call(5, "health")
        assert not nobody["ok"] and nobody["error"]["retryable"] and "no site server" in nobody["error"]["message"]
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            p.wait(30)
        for d in dirs.values():
            shutil.rmtree(d, ignore_errors=True)
    assert registry.table("aurora") == {}


@pytest.mark.parametrize("facility", ["aurora", "midway3"])
def test_the_endpoint_templates_render_as_the_endpoint_renders_them(facility):
    import json

    import jinja2
    import jinja2.sandbox
    import yaml

    from fedquest.config import load_remote
    from fedquest.paths import REPO

    text = (REPO / "deploy" / "globus" / f"{facility}_cn" / "user_config_template.yaml.j2").read_text()
    schema = json.loads((REPO / "deploy" / "globus" / f"{facility}_cn" / "user_config_schema.json").read_text())
    template = jinja2.sandbox.SandboxedEnvironment(undefined=jinja2.StrictUndefined).from_string(text)
    options = load_remote(REPO / "configs" / "remote.yaml").facilities[facility].config
    assert set(options) <= set(schema["properties"])
    for given in (options, {}):
        quoted = {k: json.dumps(v) if isinstance(v, str) else v for k, v in given.items()}
        cfg = yaml.safe_load(template.render(**quoted))
        provider = cfg["engine"]["provider"]
        assert provider["walltime"] == (given.get("walltime") or "12:00:00")
        assert provider["init_blocks"] == provider["min_blocks"] == provider["max_blocks"] == 1
        assert provider["nodes_per_block"] == given.get("nodes", 1)
        assert cfg["engine"]["max_workers_per_node"] == given.get("workers", 64)


def test_nodecheck_brings_up_a_test_job_and_reads_through_its_slots(world, monkeypatch, capsys):
    import fedquest.remote.gateway as gateway

    configs = config_module.CONFIGS
    check = configs / "remote_check.yaml"
    check.write_text("facilities:\n  aurora: {endpoint: ep-aurora, config: {nodes: 1}, placement: {toy_fed/aurora: 0}}\n"
                     "  midway3: {endpoint: ep-midway3}\n  edge: {endpoint: ep-edge}\nfederations: [toy_fed]\n"
                     "models: " + str(configs / "models.yaml") + "\n")
    monkeypatch.setenv("FEDQUEST_JOB_ID", "job-nodecheck")
    monkeypatch.setenv("FEDQUEST_CONFIGS", str(configs))
    monkeypatch.setattr("globus_compute_sdk.Executor", ThreadExecutor)
    code = gateway.nodecheck("aurora", 1, ["queue=debug", "walltime=00:30:00"], 180.0, config=str(check))
    out = capsys.readouterr().out
    assert code == 0, out
    assert "NODES OK" in out and "info of toy_fed/aurora: ok" in out and "'queue': 'debug'" in out
