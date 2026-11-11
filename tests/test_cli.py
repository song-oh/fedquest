# CLI commands checked against a fake Docker and fake model backends
import asyncio
import http.client
import http.server
import json
import subprocess
import threading
import time
import urllib.error
import urllib.request
from contextlib import nullcontext
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from fedquest import cli
from fedquest.config import load_models
from fedquest.llm.auth import TokenProvider
from fedquest.llm.client import LLMClient
from fedquest.llm.fake import FakeBackend

NEO4J, QDRANT = cli.CONTAINERS


def flag_values(args: tuple[str, ...], flag: str) -> list[str]:
    return [v for f, v in zip(args, args[1:], strict=False) if f == flag]


class FakeDocker:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.containers: dict[str, dict] = {}
        self.failures: dict[tuple[str, str], str] = {}
        self.env_file_at_run: dict[str, str] = {}
        self.waits: list[tuple[str, str]] = []

    def add(self, name: str, *, running: bool = True, password: str | None = None) -> None:
        env = ["PATH=/usr/bin", *([f"NEO4J_AUTH=neo4j/{password}"] if password else [])]
        self.containers[name] = {"project": "fedquest", "running": running, "env": env}

    def password(self, name: str = NEO4J) -> str:
        return next(e for e in self.containers[name]["env"] if e.startswith("NEO4J_AUTH=")).split("/", 1)[1]

    def runs(self) -> list[str]:
        return [flag_values(c, "--name")[0] for c in self.calls if c[0] == "run"]

    def __call__(self, *args: str) -> subprocess.CompletedProcess[str]:
        self.calls.append(args)
        verb = args[0]
        name = flag_values(args, "--name")[0] if verb == "run" else args[-1]
        if (verb, name) in self.failures:
            return subprocess.CompletedProcess(args, 1, "", self.failures[(verb, name)])
        if verb == "inspect":
            box = self.containers.get(name)
            if box is None:
                return subprocess.CompletedProcess(args, 1, "", f"Error: No such object: {name}")
            out = json.dumps(box["env"]) if "Config.Env" in args[2] else f"{box['project']} {box['running']}".lower()
            return subprocess.CompletedProcess(args, 0, out, "")
        if verb == "start":
            self.containers[name]["running"] = True
        else:
            for volume in flag_values(args, "-v"):
                host = Path(volume.split(":")[0])
                host.mkdir(parents=True, exist_ok=True)
                (host / "initialized").touch()
            project = flag_values(args, "--label")[0].split("=", 1)[1]
            self.containers[name] = {"project": project, "running": True, "env": flag_values(args, "-e")}
            self.env_file_at_run[name] = cli.ENV_FILE.read_text() if cli.ENV_FILE.exists() else ""
        return subprocess.CompletedProcess(args, 0, f"{name}\n", "")


@pytest.fixture
def docker(tmp_path, monkeypatch):
    fake = FakeDocker()
    monkeypatch.setattr(cli, "_docker", fake)
    monkeypatch.setattr(cli, "ENV_FILE", tmp_path / ".fedquest.env")
    monkeypatch.setattr(cli, "_wait_ready", lambda url, name: fake.waits.append((url, name)))
    monkeypatch.setenv("FEDQUEST_DATA", str(tmp_path / "data"))
    monkeypatch.setenv("FEDQUEST_RUNS", str(tmp_path / "runs"))
    return fake


def test_kg_up_creates_both_services_and_records_them(docker, tmp_path):
    (tmp_path / "data" / "neo4j").mkdir(parents=True)
    cli.kg_up()
    env = cli._env_file()
    assert docker.runs() == [NEO4J, QDRANT] and docker.password() == env["NEO4J_PASSWORD"]
    assert env["NEO4J_URI"] == "bolt://localhost:7688" and env["FEDQUEST_QDRANT_URL"] == "http://127.0.0.1:6340"
    assert docker.waits == [("http://127.0.0.1:6340/readyz", QDRANT), ("http://localhost:7475/", NEO4J)]
    assert cli.ENV_FILE.stat().st_mode & 0o777 == 0o600


def test_password_is_recorded_before_the_first_container_is_created(docker):
    docker.failures["run", QDRANT] = "Bind for 127.0.0.1:6340 failed: port is already allocated"
    with pytest.raises(typer.BadParameter, match="port is already allocated"):
        cli.kg_up()
    password = docker.password()
    assert f"NEO4J_PASSWORD={password}" in docker.env_file_at_run[NEO4J]
    assert cli._env_file() == {"NEO4J_PASSWORD": password}


def test_rerun_after_a_failed_second_container_keeps_the_neo4j_password(docker):
    docker.failures["run", QDRANT] = "port is already allocated"
    with pytest.raises(typer.BadParameter):
        cli.kg_up()
    created_with = docker.password()
    docker.failures.clear()
    cli.kg_up()
    assert docker.runs() == [NEO4J, QDRANT, QDRANT]
    assert cli._env_file()["NEO4J_PASSWORD"] == created_with == docker.password()
    assert cli._env_file()["NEO4J_URI"] == "bolt://localhost:7688"


def test_password_is_recovered_from_an_existing_container(docker, tmp_path):
    docker.add(NEO4J, password="from-container")
    docker.add(QDRANT)
    (tmp_path / "data" / "neo4j").mkdir(parents=True)
    (tmp_path / "data" / "neo4j" / "dbms").touch()
    cli.kg_up()
    assert cli._env_file()["NEO4J_PASSWORD"] == "from-container"
    assert not docker.runs() and not [c for c in docker.calls if c[0] == "start"]


def test_a_failed_docker_start_is_not_reported_as_running(docker):
    docker.add(NEO4J, running=False, password="pw")
    docker.add(QDRANT)
    docker.failures["start", NEO4J] = "Bind for 127.0.0.1:7688 failed: port is already allocated"
    with pytest.raises(typer.BadParameter, match=f"{NEO4J}: Bind for .* port is already allocated"):
        cli.kg_up()
    assert docker.waits == [] and "NEO4J_URI" not in cli._env_file()


def test_refuses_to_create_neo4j_over_a_database_with_an_unknown_password(docker, tmp_path):
    (tmp_path / "data" / "neo4j").mkdir(parents=True)
    (tmp_path / "data" / "neo4j" / "dbms").touch()
    with pytest.raises(typer.BadParameter, match=r"password is unknown.*kg down --purge"):
        cli.kg_up()
    assert not docker.runs()
    assert not cli.ENV_FILE.exists()


def test_recreating_neo4j_over_existing_data_uses_the_recorded_password(docker, tmp_path):
    cli._set_env("NEO4J_PASSWORD", "recorded")
    (tmp_path / "data" / "neo4j").mkdir(parents=True)
    (tmp_path / "data" / "neo4j" / "dbms").touch()
    cli.kg_up()
    assert docker.runs() == [NEO4J, QDRANT] and docker.password() == "recorded"


def test_urls_are_recorded_only_after_the_services_answer(docker, monkeypatch):
    def never_ready(url, name):
        raise typer.BadParameter(f"{name} did not answer {url}")

    monkeypatch.setattr(cli, "_wait_ready", never_ready)
    with pytest.raises(typer.BadParameter, match="did not answer"):
        cli.kg_up()
    assert set(cli._env_file()) == {"NEO4J_PASSWORD"}


def test_wait_ready_polls_every_second_until_the_service_answers(monkeypatch):
    outcomes = [urllib.error.URLError("refused"), http.client.RemoteDisconnected("reset"), nullcontext()]
    sleeps = []

    def urlopen(url, timeout):
        out = outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return out

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(time, "sleep", sleeps.append)
    cli._wait_ready("http://127.0.0.1:6340/readyz", QDRANT)
    assert sleeps == [1, 1] and not outcomes


class Readyz(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200 if self.path == "/readyz" else 503)
        self.end_headers()

    def log_message(self, *args):
        pass


def test_wait_ready_against_a_real_server_and_the_timeout_hint():
    server = http.server.HTTPServer(("127.0.0.1", 0), Readyz)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        cli._wait_ready(f"{base}/readyz", QDRANT, timeout_s=5)
        with pytest.raises(typer.BadParameter, match=f"{base}/starting within 0s; see `docker logs {QDRANT}`"):
            cli._wait_ready(f"{base}/starting", QDRANT, timeout_s=0)
    finally:
        server.shutdown()
        server.server_close()


def test_remote_status_reports_the_site_servers_health(monkeypatch):
    import fedquest.remote.gateway as gateway

    async def status():
        print("aurora here: 1 sites")
        return 0

    monkeypatch.setattr(gateway, "status", status)
    result = CliRunner().invoke(cli.app, ["remote", "status"])
    assert result.exit_code == 0 and "aurora here" in result.output


def test_llm_check_runs_every_call_on_one_event_loop_and_closes_the_client(monkeypatch):
    class LoopBound(FakeBackend):
        def __init__(self):
            super().__init__()
            self.loop = None
            self.closed = False

        async def complete(self, *args, **kwargs):
            self.loop = self.loop or asyncio.get_running_loop()
            if asyncio.get_running_loop() is not self.loop:
                raise RuntimeError("Event loop is closed")
            return await super().complete(*args, **kwargs)

        async def aclose(self):
            self.closed = True

    class Json:
        def __init__(self, body):
            self.body = body

        def json(self):
            return self.body

    monkeypatch.setenv("FEDQUEST_LLM", "fake")
    cfg = load_models().model_copy(update={"provider": "alcf"})
    backend = LoopBound()
    client = LLMClient(cfg, cache=None, tokens=TokenProvider("fake"), backend=backend)
    monkeypatch.setattr(LLMClient, "from_env", classmethod(lambda cls, **kw: client))
    monkeypatch.setattr("requests.get", lambda url, **kw: Json({}))
    cli.llm_check(call=True)
    assert sorted(model for model, _ in backend.calls) == sorted(set(cfg.roles.values()))
    assert backend.closed


class Down(FakeBackend):
    def __init__(self, down, times=1_000_000, error=asyncio.TimeoutError):
        super().__init__()
        self.down, self.times, self.error = down, times, error

    async def complete(self, model, *args, **kwargs):
        if model == self.down and self.times > 0:
            self.times -= 1
            self.calls.append((model, kwargs["task"]))
            raise self.error()
        return await super().complete(model, *args, **kwargs)


def check_with(monkeypatch, role, bodies=None, **down):
    async def instant(seconds):
        return None

    monkeypatch.setenv("FEDQUEST_LLM", "fake")
    cfg = load_models().model_copy(update={"provider": "alcf"})
    backend = Down(cfg.roles[role], **down)
    client = LLMClient(cfg, cache=None, tokens=TokenProvider("fake"), backend=backend)
    monkeypatch.setattr(LLMClient, "from_env", classmethod(lambda cls, **kw: client))
    answer = (bodies or {}).get
    monkeypatch.setattr("requests.get", lambda url, **kw: type("Json", (), {
        "json": lambda self: answer(url.rsplit("/", 1)[-1], {})})())
    monkeypatch.setattr(asyncio, "sleep", instant)
    return cfg, backend


def test_llm_check_exits_1_when_a_model_the_runs_need_does_not_answer(monkeypatch):
    cfg, backend = check_with(monkeypatch, "answer")
    with pytest.raises(typer.Exit) as e:
        cli.llm_check(call=True)
    assert e.value.exit_code == 1
    assert [m for m, _ in backend.calls].count(cfg.roles["answer"]) == 2


def test_llm_check_wait_retries_until_the_model_answers(monkeypatch):
    cfg, backend = check_with(monkeypatch, "answer", times=3)
    cli.llm_check(call=True, wait=5)
    assert [m for m, _ in backend.calls].count(cfg.roles["answer"]) == 4


def test_llm_check_does_not_wait_on_a_failure_waiting_cannot_mend(monkeypatch):
    cfg, backend = check_with(monkeypatch, "answer", error=ValueError)

    async def no_wait(seconds):
        raise AssertionError("waited on a failure waiting cannot mend")

    monkeypatch.setattr(asyncio, "sleep", no_wait)
    with pytest.raises(typer.Exit):
        cli.llm_check(call=True, wait=60)
    assert [m for m, _ in backend.calls].count(cfg.roles["answer"]) == 1


@pytest.mark.parametrize("bodies", [{"jobs": []}, {"jobs": {"running": None, "queued": [None, "x"]}},
                                    {"list-endpoints": {"clusters": None}}, {"list-endpoints": [1]},
                                    {"jobs": {"running": [{"Models": "google/gemma-4-31B-it", "Model Status": "[/x]"}]}}])
def test_llm_check_decides_on_the_replies_whatever_shape_the_model_states_come_in(monkeypatch, bodies):
    check_with(monkeypatch, "judge2", bodies=bodies)
    cli.llm_check(call=True)


def test_llm_check_passes_when_only_the_optional_judge2_does_not_answer(monkeypatch):
    cfg, _ = check_with(monkeypatch, "judge2")
    assert cfg.roles["judge2"] not in {m for r, m in cfg.roles.items() if r != "judge2"}
    cli.llm_check(call=True)


def test_eval_run_has_no_budget_override():
    result = CliRunner().invoke(cli.app, ["eval", "run", "smoke_mmqa", "--budgets", "5"])
    assert result.exit_code == 2 and "No such option" in result.output


def test_eval_score_judge2_needs_the_role(tmp_path, monkeypatch):
    import yaml

    models = yaml.safe_load((Path(cli.REPO) / "configs" / "models.yaml").read_text())
    models["roles"].pop("judge2", None)
    (tmp_path / "models.yaml").write_text(yaml.safe_dump(models))
    monkeypatch.setenv("FEDQUEST_MODELS", str(tmp_path / "models.yaml"))
    result = CliRunner().invoke(cli.app, ["eval", "score", "smoke_mmqa", "--judge2"])
    assert result.exit_code == 1 and "defines no judge2 role" in result.output
