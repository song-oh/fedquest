# pipeline driver on tiny plans: lanes, dependencies, failures and retries
import json
import sys
import time

import pytest

from fedquest.pipeline import driver as d


@pytest.fixture
def plan(tmp_path, monkeypatch):
    monkeypatch.setenv("FEDQUEST_PIPELINE", str(tmp_path / "pipeline"))
    monkeypatch.setattr(d, "POLL_S", 0.05)
    monkeypatch.setattr(d, "RETRY_WAIT_S", 0.0)

    def write(tasks):
        d.write_plan({"created": time.time(), "experiments": [], "tasks": [t.__dict__ for t in tasks]})

    return write


def py(code: str) -> list[str]:
    return [sys.executable, "-c", code]


def test_lanes_dependencies_and_scores(plan, tmp_path):
    out = tmp_path / "order.txt"
    note = f"import time; open({str(out)!r}, 'a').write('{{}}\\n'.format(__import__('os').environ['FEDQUEST_PIPELINE_TASK']))"
    plan([d.Task("a", "run", "a", "gemma", 1, 0, py(note)),
          d.Task("b", "run", "b", "gemma", 2, 1, py(note), after=["a"]),
          d.Task("score:a", "score", "a", "judge", 1, 0, py(note), after=["a"]),
          d.Task("c", "run", "c", "mx", 2, 2, py("import sys; sys.exit(5)"), optional=True)])
    drv = d.Driver({"gemma": 1, "mx": 1, "judge": 1, "latency": 1})
    assert drv.run() == 0
    order = out.read_text().split()
    assert order.index("a") < order.index("b") and order.index("a") < order.index("score:a")
    st = d.States()
    assert st.get("c")["state"] == "failed" and st.get("c")["tries"] == d.TRIES
    assert all(st.get(t)["state"] == "done" for t in ("a", "b", "score:a"))
    st.put("a", done_at=time.time() + 5)
    drv2 = d.Driver({"gemma": 1, "mx": 1, "judge": 1, "latency": 1})
    assert not drv2.done("score:a")


def test_a_failure_blocks_what_waits_for_it_and_retry_reopens_it(plan, tmp_path):
    flag = tmp_path / "ok"
    plan([d.Task("a", "run", "a", "gemma", 1, 0, py(f"import os, sys; sys.exit(0 if os.path.exists({str(flag)!r}) else 1)")),
          d.Task("b", "run", "b", "gemma", 1, 1, py("pass"), after=["a"])])
    assert d.Driver().run() == 1
    st = d.States()
    assert st.get("a")["tries"] == d.TRIES and st.get("b") == {}
    flag.write_text("")
    d.retry(None, errors=False)
    assert d.Driver().run() == 0
    assert d.States().get("b")["state"] == "done"


def test_a_refused_login_stops_new_starts(plan):
    plan([d.Task("a", "run", "a", "gemma", 1, 0, py("import sys; sys.exit(3)")),
          d.Task("b", "run", "b", "gemma", 2, 1, py("pass"))])
    assert d.Driver({"gemma": 1}).run() == d.AUTH_EXIT
    assert d.States().get("b") == {}


def test_status_runs_on_a_plan(plan, capsys):
    plan([d.Task("a", "run", "a", "gemma", 1, 0, py("pass"))])
    d.status()
    assert "waiting" in capsys.readouterr().out


def test_the_catalog_plans(tmp_path, monkeypatch):
    monkeypatch.setenv("FEDQUEST_PIPELINE", str(tmp_path / "pipeline"))
    plan = d.build()
    ids = [t["id"] for t in plan["tasks"]]
    assert "main_mmqa" in ids and "score:main_mmqa" in ids and "score2:main_mmqa" in ids
    assert not any(i.startswith("smoke_") for i in ids)
    lat = next(t for t in plan["tasks"] if t["id"] == "latency_mmqa")
    assert "main_mmqa" in lat["after"] and "scale_musique" in lat["after"] and "scale_mmqa" not in ids
    assert json.loads(json.dumps(plan))


def test_a_tool_task_without_dependencies_counts_as_done(plan):
    plan([d.Task("tool:build", "tool", "tool:build", "cpu", -1, 0, py("pass")),
          d.Task("main", "run", "main", "gemma", 1, 1, py("pass"), after=["tool:build"])])
    assert d.Driver({"gemma": 1, "cpu": 1}).run() == 0
    assert d.States().get("main")["state"] == "done"
    d.status()


def test_an_optional_task_out_of_tries_releases_what_waits_for_it(plan):
    plan([d.Task("tool:route", "tool", "tool:route", "probe", -1, 0, py("import sys; sys.exit(1)"), optional=True),
          d.Task("main", "run", "main", "gemma", 1, 1, py("pass"), after=["tool:route"])])
    assert d.Driver({"gemma": 1, "probe": 1}).run() == 0
    st = d.States()
    assert st.get("tool:route")["tries"] == d.TRIES and st.get("main")["state"] == "done"
