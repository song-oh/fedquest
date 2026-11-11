# pipeline driver: catalog tasks run in lanes, with dependencies and retries
from __future__ import annotations

import contextlib
import fnmatch
import importlib.util
import json
import logging
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]
LANES = {"gemma": 6, "mx": 2, "hippo": 2, "latency": 1, "judge": 3, "probe": 2, "cpu": 2, "llm": 2, "post": 1}
TOOLS = "configs/tools.json"
STOP_FILE = "STOP"
TRIES = 3
RETRY_WAIT_S = 120.0
POLL_S = 5.0
STOP_WAIT_S = 120.0
MODELS = "configs/models/final.yaml"
JUDGE2 = ("main_", "dgrag_", "mx_")
AUTH_EXIT = 3
INTERRUPTED = -2
_TASK_THREADS = os.environ.get("FEDQUEST_TASK_THREADS", "4")
THREADS = {"OMP_NUM_THREADS": _TASK_THREADS, "MKL_NUM_THREADS": _TASK_THREADS, "OPENBLAS_NUM_THREADS": _TASK_THREADS,
           "NUMEXPR_MAX_THREADS": _TASK_THREADS, "FEDQUEST_TORCH_THREADS": _TASK_THREADS,
           "TOKENIZERS_PARALLELISM": "false"}


def root() -> Path:
    from fedquest.paths import runs_root

    return Path(os.environ.get("FEDQUEST_PIPELINE") or runs_root() / "pipeline")


@dataclass
class Task:
    id: str
    kind: str
    experiment: str
    lane: str
    stage: int
    index: int
    argv: list[str]
    env: dict[str, str] = field(default_factory=dict)
    after: list[str] = field(default_factory=list)
    optional: bool = False
    after_any: bool = False


def _fedquest() -> list[str]:
    exe = REPO / ".venv" / "bin" / "fedquest"
    return [str(exe)] if exe.exists() else [sys.executable, "-m", "fedquest.cli"]


def _program(argv: list[str]) -> list[str]:
    python = REPO / ".venv" / "bin" / "python"
    py = str(python) if python.exists() else sys.executable
    return [py, *argv[1:]] if argv and argv[0] in ("python", "python3") else list(argv)


def catalog() -> list[str]:
    spec = importlib.util.spec_from_file_location("fedquest_make_configs", REPO / "scripts" / "make_configs.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return [n for n, _, _ in mod.experiments() if not n.startswith("smoke_")]


def _expand(env: dict[str, str]) -> dict[str, str]:
    from fedquest.paths import runs_root

    return {k: os.path.expanduser(str(v).replace("$FEDQUEST_RUNS", str(runs_root()))) for k, v in (env or {}).items()}


def select(patterns: list[str], names: list[str]) -> list[str]:
    out: dict[str, None] = {}
    for p in patterns:
        hits = [n for n in names if fnmatch.fnmatchcase(n, p)]
        if not hits:
            raise KeyError(f"{p!r} matches no experiment of the plan")
        out.update(dict.fromkeys(hits))
    return list(out)


def tool_tasks(path: Path | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    path = path or REPO / TOOLS
    if not path.exists():
        return [], []
    data = json.loads(path.read_text())
    return list(data.get("tools") or []), list(data.get("post") or [])


def build(experiments: list[str] | None = None, with_tools: bool | None = None) -> dict[str, Any]:
    from fedquest.config import load_experiment

    names = experiments or catalog()
    if with_tools is None:
        with_tools = experiments is None
    tools, post = tool_tasks() if with_tools else ([], [])
    tasks: list[Task] = []
    fq = _fedquest()
    tool_ids = {t["id"] for t in tools}
    for j, t in enumerate(tools):
        tasks.append(Task(t["id"], "tool", t["id"], t["lane"], -1, j, _program(list(t["argv"])),
                          {"FEDQUEST_MODELS": MODELS, **_expand(t.get("env") or {})},
                          [d for d in t.get("after") or [] if d in tool_ids], bool(t.get("optional")),
                          bool(t.get("after_any"))))
    for i, name in enumerate(names):
        exp = load_experiment(name)
        run = [*fq, "eval", "run", name, "--mode", "remote"]
        argv = ["bash", "-c", f"set -e; {shlex.join(_program(exp.before))}; exec {shlex.join(run)}"] \
            if exp.before else run
        env = {"FEDQUEST_MODELS": exp.models or MODELS, **({"FEDQUEST_LLM_LIVE": "1"} if exp.live_calls else {}),
               **_expand(exp.env)}
        tasks.append(Task(name, "run", name, exp.lane, exp.stage, i, argv, env,
                          [d for d in exp.after if d in names or d in tool_ids], exp.optional))
        if exp.judge:
            tasks.append(Task(f"score:{name}", "score", name, "judge", exp.stage, i, [*fq, "eval", "score", name],
                              {"FEDQUEST_MODELS": MODELS}, [name], exp.optional))
            if name.startswith(JUDGE2):
                tasks.append(Task(f"score2:{name}", "score2", name, "judge", exp.stage, i,
                                  [*fq, "eval", "score", name, "--judge2"], {"FEDQUEST_MODELS": MODELS},
                                  [f"score:{name}"], exp.optional))
    before = [t.id for t in tasks]
    for j, t in enumerate(post):
        tasks.append(Task(t["id"], "tool", t["id"], t["lane"], 99, j, _program(list(t["argv"])),
                          {"FEDQUEST_MODELS": MODELS, **_expand(t.get("env") or {})}, list(before),
                          bool(t.get("optional")), bool(t.get("after_any", True))))
    unknown = sorted({t.lane for t in tasks} - set(LANES))
    if unknown:
        raise ValueError(f"lanes {unknown} have no slots (LANES)")
    return {"created": time.time(), "experiments": names, "tasks": [asdict(t) for t in tasks]}


def write_plan(plan: dict[str, Any]) -> Path:
    path = root() / "plan.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(plan, indent=1))
    tmp.replace(path)
    return path


def load_plan() -> list[Task]:
    path = root() / "plan.json"
    if not path.exists():
        raise FileNotFoundError(f"no plan at {path}: python -m fedquest.pipeline plan")
    return [Task(**t) for t in json.loads(path.read_text())["tasks"]]


def _safe(task_id: str) -> str:
    return task_id.replace(":", "__").replace("/", "_")


class States:
    def __init__(self) -> None:
        self.dir = root() / "tasks"
        self.dir.mkdir(parents=True, exist_ok=True)

    def get(self, task_id: str) -> dict[str, Any]:
        try:
            return json.loads((self.dir / f"{_safe(task_id)}.json").read_text())
        except (OSError, ValueError):
            return {}

    def put(self, task_id: str, **update: Any) -> dict[str, Any]:
        st = {**self.get(task_id), **update}
        path = self.dir / f"{_safe(task_id)}.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(st))
        tmp.replace(path)
        return st


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def driver_info(anywhere: bool = False) -> dict[str, Any] | None:
    try:
        info = json.loads((root() / "driver.json").read_text())
    except (OSError, ValueError):
        return None
    if not info.get("pid"):
        return None
    if info.get("host") == socket.gethostname():
        return {**info, "alive": True} if _alive(int(info["pid"])) else None
    return {**info, "alive": None} if anywhere else None


class Driver:
    def __init__(self, lanes: dict[str, int] | None = None) -> None:
        self.tasks = load_plan()
        self.by_id = {t.id: t for t in self.tasks}
        self.lanes = {**LANES, **(lanes or {})}
        self.states = States()
        self.procs: dict[str, subprocess.Popen[bytes]] = {}
        self.stopping = False
        self.auth = False
        (root() / "logs").mkdir(parents=True, exist_ok=True)

    def done(self, task_id: str) -> bool:
        st = self.states.get(task_id)
        if st.get("state") != "done":
            return False
        t = self.by_id[task_id]
        # a score goes stale once the task it follows finishes again after it
        if t.kind in ("score", "score2"):
            ran = self.states.get(t.after[0]).get("done_at", 0)
            return st.get("done_at", 0) >= ran
        return True

    # spent: out of tries, or waiting on a required dependency that is spent
    def spent(self, task_id: str) -> bool:
        st = self.states.get(task_id)
        if st.get("state") == "failed" and st.get("tries", 0) >= TRIES:
            return True
        t = self.by_id[task_id]
        if t.after_any:
            return False
        return any(self.spent(d) and not self.by_id[d].optional for d in t.after if d in self.by_id)

    def ready(self, t: Task) -> bool:
        if t.id in self.procs or self.done(t.id) or self.spent(t.id):
            return False
        st = self.states.get(t.id)
        if st.get("state") == "failed" and time.time() - st.get("ended", 0) < RETRY_WAIT_S:
            return False
        if t.after_any:
            return all(self.done(d) or self.spent(d) for d in t.after if d in self.by_id)
        return all(self.done(d) or (self.spent(d) and self.by_id[d].optional) for d in t.after if d in self.by_id)

    def launch(self, t: Task) -> None:
        st = self.states.get(t.id)
        env = {**os.environ, **THREADS, **t.env, "FEDQUEST_PIPELINE_TASK": t.id}
        logf = open(root() / "logs" / f"{_safe(t.id)}.log", "ab")  # noqa: SIM115
        logf.write(f"\n==== {time.strftime('%F %T')} {t.id} try {st.get('tries', 0) + 1}: {shlex.join(t.argv)}\n"
                   .encode())
        logf.flush()
        proc = subprocess.Popen(t.argv, cwd=REPO, env=env, stdout=logf, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, start_new_session=True)
        logf.close()
        self.procs[t.id] = proc
        self.states.put(t.id, state="running", pid=proc.pid, host=socket.gethostname(), started=time.time(),
                        tries=st.get("tries", 0) + 1)
        log.info("started %s (lane %s, try %d, pid %d)", t.id, t.lane, st.get("tries", 0) + 1, proc.pid)

    def reap(self) -> None:
        for tid, proc in list(self.procs.items()):
            code = proc.poll()
            if code is None:
                continue
            del self.procs[tid]
            st = self.states.get(tid)
            codes = [*st.get("codes", []), code]
            if code == 0:
                self.states.put(tid, state="done", codes=codes, ended=time.time(), done_at=time.time())
                log.info("%s done (%.0f s)", tid, time.time() - st.get("started", time.time()))
            # killed by our own stop: not a failure, so the try is given back
            elif self.stopping and code in (-signal.SIGINT, -signal.SIGKILL, 130, 137):
                self.states.put(tid, state="failed", codes=[*st.get("codes", []), INTERRUPTED], ended=0,
                                tries=max(0, st.get("tries", 1) - 1))
            else:
                self.states.put(tid, state="failed", codes=codes, ended=time.time())
                if code == AUTH_EXIT:
                    self.auth = True
                    log.error("%s: the ALCF login was refused (exit 3): no task starts until the driver runs again "
                              "after a fresh login (python third_party/inference_auth_token.py authenticate --force)",
                              tid)
                else:
                    left = TRIES - st.get("tries", 1)
                    log.warning("%s failed (exit %s); %s", tid, code,
                                f"{left} more tries" if left > 0 else "no tries left" + (
                                    " (optional)" if self.by_id[tid].optional else ""))

    def adopt(self) -> None:
        for t in self.tasks:
            st = self.states.get(t.id)
            if st.get("state") != "running":
                continue
            pid = int(st.get("pid") or 0)
            if pid and st.get("host") == socket.gethostname() and _alive(pid):
                log.warning("%s still runs from an earlier driver (pid %d): waiting for it", t.id, pid)
                while _alive(pid):
                    time.sleep(POLL_S)
            self.states.put(t.id, state="failed", codes=[*st.get("codes", []), INTERRUPTED], ended=0,
                            tries=max(0, st.get("tries", 1) - 1))

    def finished(self) -> bool:
        return not self.procs and all(self.done(t.id) or self.spent(t.id) for t in self.tasks)

    def step(self) -> None:
        self.reap()
        if not self.stopping and (root() / STOP_FILE).exists():
            log.info("%s appeared: stopping", root() / STOP_FILE)
            (root() / STOP_FILE).unlink(missing_ok=True)
            self.stop()
        if self.stopping or self.auth:
            return
        for lane, slots in self.lanes.items():
            busy = sum(1 for tid in self.procs if self.by_id[tid].lane == lane)
            if busy >= slots:
                continue
            order = sorted((t for t in self.tasks if t.lane == lane and self.ready(t)),
                           key=lambda t: (t.stage, t.index, t.kind != "run"))
            for t in order[: slots - busy]:
                self.launch(t)

    def stop(self, *_: Any) -> None:
        if self.stopping:
            return
        self.stopping = True
        log.info("stopping: SIGINT to %d tasks", len(self.procs))
        for proc in self.procs.values():
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGINT)

    def run(self) -> int:
        info = driver_info()
        if info is not None and info["pid"] != os.getpid():
            raise SystemExit(f"a driver already runs (pid {info['pid']}): python -m fedquest.pipeline status")
        (root() / STOP_FILE).unlink(missing_ok=True)
        (root() / "driver.json").write_text(json.dumps({"pid": os.getpid(), "host": socket.gethostname(),
                                                         "started": time.time(), "lanes": self.lanes}))
        signal.signal(signal.SIGINT, self.stop)
        signal.signal(signal.SIGTERM, self.stop)
        self.adopt()
        try:
            while True:
                self.step()
                if self.stopping:
                    deadline = time.time() + STOP_WAIT_S
                    while self.procs and time.time() < deadline:
                        time.sleep(1.0)
                        self.reap()
                    for proc in self.procs.values():
                        with contextlib.suppress(ProcessLookupError):
                            os.killpg(proc.pid, signal.SIGKILL)
                    time.sleep(1.0)
                    self.reap()
                    log.info("stopped")
                    return 0
                if self.auth and not self.procs:
                    return AUTH_EXIT
                if self.finished():
                    failed = [t.id for t in self.tasks if self.spent(t.id) and not t.optional]
                    log.info("every task ended%s", f"; failed: {failed}" if failed else "")
                    return 1 if failed else 0
                time.sleep(POLL_S)
        finally:
            with contextlib.suppress(OSError):
                (root() / "driver.json").unlink()


def _rows(experiment: str) -> tuple[int, int]:
    from fedquest.paths import results_dir

    n = err = 0
    for p in (results_dir(experiment) / "rows").glob("*.jsonl"):
        with open(p, errors="replace") as f:
            for line in f:
                n += 1
                err += '"error": null' not in line and '"error":' in line
    return n, err


def status(show: str = "all") -> int:
    tasks = load_plan()
    states = States()
    drv = driver_info(anywhere=True)
    where = "" if not drv else (f"running, pid {drv['pid']}" if drv.get("alive") else
                               f"recorded as running on {drv.get('host')}, pid {drv['pid']} (not checkable here)")
    print(f"driver: {where or 'not running'}; plan of {len(tasks)} tasks")
    by: dict[str, int] = {}
    lines = []
    d = Driver.__new__(Driver)
    d.tasks, d.by_id, d.states, d.procs = tasks, {t.id: t for t in tasks}, states, {}
    for t in tasks:
        st = states.get(t.id)
        state = st.get("state") or "waiting"
        if state == "done" and not d.done(t.id):
            state = "stale"
        if state == "failed" and d.spent(t.id):
            state = "failed (optional)" if t.optional else "FAILED"
        by[state] = by.get(state, 0) + 1
        if show == "all" or state not in ("done",):
            rows = _rows(t.experiment) if t.kind == "run" else None
            took = (st.get("ended") or time.time()) - st["started"] if st.get("started") else None
            lines.append(f"{t.id:44s} {t.lane:8s} {state:18s} tries {st.get('tries', 0)}"
                         + (f"  {took / 60:6.1f} min" if took else "")
                         + (f"  rows {rows[0]} (errors {rows[1]})" if rows and rows[0] else "")
                         + (f"  exits {st.get('codes')}" if st.get("codes") and state != "done" else ""))
    print("\n".join(lines))
    print(", ".join(f"{k} {v}" for k, v in sorted(by.items())))
    return 0


def stop() -> int:
    info = driver_info()
    if info is None:
        other = driver_info(anywhere=True)
        if other is not None:
            (root() / STOP_FILE).write_text(f"asked from {socket.gethostname()} at {time.ctime()}\n")
            print(f"asked the driver on {other.get('host')} (pid {other['pid']}) to stop ({root() / STOP_FILE}); "
                  "it stops its tasks within a few seconds")
            return 0
        print("no driver runs on this host")
        return 0
    os.kill(int(info["pid"]), signal.SIGTERM)
    deadline = time.time() + STOP_WAIT_S + 30
    while _alive(int(info["pid"])) and time.time() < deadline:
        time.sleep(1.0)
    if _alive(int(info["pid"])):
        print(f"the driver (pid {info['pid']}) still runs")
        return 1
    print("stopped")
    return 0


def retry(patterns: list[str] | None, errors: bool) -> int:
    if driver_info(anywhere=True) is not None:
        raise SystemExit("the driver runs: python -m fedquest.pipeline stop first")
    tasks = load_plan()
    states = States()
    names = sorted({t.experiment for t in tasks})
    chosen = set(select(patterns, names)) if patterns else set(names)
    n = 0
    for t in tasks:
        if t.experiment not in chosen:
            continue
        st = states.get(t.id)
        if st.get("state") == "failed":
            states.put(t.id, tries=0, ended=0)
            n += 1
        elif errors and t.kind == "run" and st.get("state") == "done" and _rows(t.experiment)[1]:
            states.put(t.id, state="failed", tries=0, ended=0)
            n += 1
    print(f"{n} tasks will run again at the next start")
    return 0
