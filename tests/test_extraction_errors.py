# a failing site's extractions and reads are counted and logged, never charged
import json
import logging
import shutil
from typing import Any

import pytest

import fedquest.config as config_module
import fedquest.eval.runner as runner
from fedquest.config import ExperimentCfg, MatcherCfg, load_models
from fedquest.coordinator import Coordinator, build_matcher
from fedquest.federation.base import Federation
from fedquest.kg.matcher import Matcher, TextSim
from fedquest.kg.memory import MemoryKG
from fedquest.ledger import Ledger
from fedquest.llm.auth import TokenProvider
from fedquest.llm.client import LLMClient
from fedquest.llm.fake import FakeBackend
from fedquest.llm.prompts import ReadUnknown
from fedquest.methods.base import build_method
from fedquest.paths import CONFIGS
from fedquest.types import ProbeRequest, Question, Usage
from fedquest.utils import write_jsonl
from fedquest.vdb.encoders import load_encoder
from tests.fakes import RUNNING_QUESTION, running_example, running_responder
from tests.toy import build_toy, responder, toy_env

LOGIN = "RuntimeError: ALCF login required on this machine"
POSTER, DECOY, ROW = ("aurora", "mmqa/image/p1"), ("aurora", "mmqa/image/p9"), ("midway3", "mmqa/row/t1#6")
Q = Question("q1", "mmqa", RUNNING_QUESTION, ["baseball"], qtype="Compose(ImageQ,TableQ)",
             bridge=["The Bad News Bears"])


class BrokenEndpoint:
    def __init__(self, inner: Any, error: str = LOGIN) -> None:
        self.inner, self.name, self.error = inner, inner.name, error

    def failed(self, d: dict[str, Any]) -> dict[str, Any]:
        return {**d, "facts": [], "usage": Usage().to_dict(), "raw_text": None, "error": self.error}

    async def call(self, action: str, **kwargs: Any) -> Any:
        out = await self.inner.call(action, **kwargs)
        if action == "extract":
            return [self.failed(d) for d in out]
        if action == "read" and kwargs.get("extract"):
            return {**out, "read": [], "usage": Usage().to_dict(), "usage_by_modality": {},
                    "extractions": [self.failed(x) for x in out["extractions"]]}
        if action == "read":
            return {**out, "bindings": {}, "usage": Usage().to_dict(), "usage_by_modality": {}, "raw": {},
                    "error": self.error}
        return out

    async def close(self) -> None:
        await self.inner.close()


def breaking(fed: Federation, *sites: str, error: str = LOGIN) -> Federation:
    return Federation([BrokenEndpoint(e, error) if e.name in (sites or fed.site_names) else e
                       for e in fed.endpoints.values()])


def coordinator(fed: Federation) -> Coordinator:
    cfg = load_models().model_copy(update={"provider": "fake"})
    llm = LLMClient(cfg, cache=None, tokens=TokenProvider("fake"), backend=FakeBackend(running_responder))
    return Coordinator(fed, llm, MemoryKG(), Matcher(TextSim(load_encoder("hash")), theta=0.6))


async def test_extract_counts_failures_and_logs_each_distinct_error_once(caplog):
    fed = breaking(running_example()[0], "aurora")
    ledger = Ledger()
    with caplog.at_level(logging.WARNING):
        results = await fed.extract([POSTER, DECOY, ROW], ledger=ledger)
        await fed.extract([POSTER], ledger=ledger)
    assert [r.error for r in results] == [LOGIN, LOGIN, None]
    assert ledger.extraction_errors == 3 and ledger.extractions == 1
    assert ledger.summary()["extraction_errors"] == 3
    assert caplog.messages == [f"extraction at aurora failed: {LOGIN} (repeats are not logged)"]

    fed.endpoints["aurora"].error = "TimeoutError: no reply"
    with caplog.at_level(logging.WARNING):
        await fed.extract([POSTER], ledger=ledger)
    assert len(caplog.messages) == 2 and "TimeoutError: no reply" in caplog.messages[1]
    assert ledger.extraction_errors == 4


async def test_an_object_that_failed_stays_on_offer_so_no_probe_depends_on_who_met_it_first():
    fed = breaking(running_example()[0], "aurora", error="image missing")
    request = ProbeRequest("The Bad News Bears poster", "bound", 5)
    before = [c.ref for c in (await fed.probe([request]))[0]]
    ledger = Ledger()
    await fed.extract([POSTER], ledger=ledger)
    assert ledger.extraction_errors == 1 and POSTER in before
    assert [c.ref for c in (await fed.probe([request]))[0]] == before


async def test_the_same_error_at_another_site_is_logged_separately(caplog):
    fed = breaking(running_example()[0])
    with caplog.at_level(logging.WARNING):
        await fed.extract([POSTER, ROW, ("edge", "mmqa/text/b1#0")])
        await fed.extract([ROW])
    assert [m.split(" failed")[0] for m in caplog.messages] == [
        "extraction at aurora", "extraction at midway3", "extraction at edge"]


async def test_long_errors_are_truncated_before_they_are_compared(caplog):
    fed = breaking(running_example()[0], "aurora", error="RateLimitError: " + "x" * 300)
    with caplog.at_level(logging.WARNING):
        await fed.extract([POSTER])
        fed.endpoints["aurora"].error = "RateLimitError: " + "x" * 200 + " id=7"
        await fed.extract([DECOY])
    assert len(caplog.messages) == 1 and len(caplog.messages[0]) < 200


async def test_a_failed_read_is_counted_logged_once_and_never_charged(caplog):
    fed = breaking(running_example()[0], "aurora")
    ledger = Ledger()
    unknowns = [ReadUnknown("sport1", ["(The Bad News Bears | poster depicts sport | X)"], ["The Bad News Bears"]).to_dict()]
    with caplog.at_level(logging.WARNING):
        out = await fed.read({"aurora": {"blocks": [[POSTER[1]]]}}, RUNNING_QUESTION, unknowns, ledger=ledger)
        await fed.read({"aurora": {"blocks": [[DECOY[1]]]}}, RUNNING_QUESTION, unknowns, ledger=ledger)
    assert out["aurora"].error == LOGIN and not out["aurora"].bindings
    assert ledger.extraction_errors == 2 and ledger.extractions == 0 and ledger.reads == 0
    assert len(caplog.messages) == 1


async def test_healthy_sites_report_no_errors(caplog):
    fed, _ = running_example()
    ledger = Ledger()
    with caplog.at_level(logging.WARNING):
        await fed.extract([POSTER, ROW], ledger=ledger)
    assert ledger.extraction_errors == 0 and ledger.extractions == 2 and not caplog.messages


async def test_fedquest_spends_no_budget_on_a_site_whose_reads_fail():
    coord = coordinator(breaking(running_example()[0], "aurora"))
    r = await coord.run(build_method("fedquest"), Q, "ns")
    assert r.error is None and r.ledger["extraction_errors"] > 0 and r.consulted == []
    failed = [e for e in r.trace if e["event"] == "read" and e["error"]]
    assert failed and all(e["site"] == "aurora" and e["error"] == LOGIN for e in failed)
    assert r.bindings.get("?film1") == "The Bad News Bears"


@pytest.fixture
def world(tmp_path, monkeypatch):
    with toy_env(tmp_path):
        w = build_toy(tmp_path / "world")
        configs = tmp_path / "configs"
        (configs / "federations").mkdir(parents=True)
        (configs / "experiments").mkdir()
        shutil.copy(w.config, configs / "federations" / "toy_fed.yaml")
        shutil.copy(CONFIGS / "models.yaml", configs / "models.yaml")
        monkeypatch.setattr(config_module, "CONFIGS", configs)
        monkeypatch.setenv("FEDQUEST_MODELS", str(configs / "models.yaml"))
        yield w


def fail_tasks(monkeypatch, *tasks: str) -> None:
    def respond(task: str, messages: list[dict[str, Any]]) -> str:
        if task in tasks:
            raise RuntimeError("ALCF login required on this machine")
        return responder(task, messages)

    original = LLMClient.from_env.__func__

    def from_env(cls, **kw):
        return original(cls, **{**kw, "backend": FakeBackend(respond)})

    monkeypatch.setattr(LLMClient, "from_env", classmethod(from_env))
    monkeypatch.setattr(runner, "build_matcher", lambda cfg=None: build_matcher(MatcherCfg(encoder="hash")))


async def test_run_experiment_warns_when_rows_have_failed_extractions(world, monkeypatch, caplog):
    fail_tasks(monkeypatch, "read")
    exp = ExperimentCfg(name="toy_errors", federation="toy_fed", workload="cold", budgets=[5],
                        matcher={"encoder": "hash"}, methods=[{"method": "fedquest"}])
    with caplog.at_level(logging.WARNING):
        out = await runner.run_experiment(exp, mode="local")
    row = json.loads((out / "rows" / "fedquest.jsonl").read_text().splitlines()[0])
    failed = row["ledger"]["extraction_errors"]
    assert failed > 0
    assert f"toy_errors: fedquest has {failed} failed extractions in its rows" in caplog.messages


def test_resume_retries_failed_questions_and_scoring_keeps_the_latest_row(tmp_path):
    from fedquest.eval.runner import _done
    from fedquest.eval.score import latest_rows

    path = tmp_path / "rows.jsonl"
    write_jsonl(path, [{"qid": "a", "answer": "x"}, {"qid": "b", "answer": "", "error": "TimeoutError: no reply"}])
    assert _done(path) == {("a", 0), ("b", 0)}
    assert _done(path, retry_errors=True) == {("a", 0)}
    rows = latest_rows([{"qid": "a", "answer": "x"}, {"qid": "b", "error": "TimeoutError"}, {"qid": "b", "answer": "y"}])
    assert [(r["qid"], r.get("answer")) for r in rows] == [("a", "x"), ("b", "y")]
