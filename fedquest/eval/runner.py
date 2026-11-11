from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import os
import platform
import random
import time
from collections import Counter
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from contextlib import AsyncExitStack, suppress
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

import fedquest.config as config_module
from fedquest.config import (
    ExperimentCfg,
    MethodSpec,
    ModelsCfg,
    NetworkCfg,
    calibration_file,
    load_federation,
    load_models,
    models_sha,
)
from fedquest.coordinator import Coordinator, build_matcher
from fedquest.data import mmqa, splits, workloads
from fedquest.errors import auth_error, network_error
from fedquest.eval.score import CLOSED_POOL
from fedquest.eval.stream import Barrier, StreamKG, barrier_dir
from fedquest.federation.base import Federation, require_current
from fedquest.federation.network import delayed
from fedquest.kg.view import open_view
from fedquest.ledger import Ledger
from fedquest.llm import timing
from fedquest.llm.client import LLMClient
from fedquest.methods.base import METHODS, Method, MethodResult, build_method, method_accepts
from fedquest.paths import PRESETS_FILE, calibration_path, collection_dir, questions_path, results_dir
from fedquest.planner.calibration import Calibrator
from fedquest.types import Question
from fedquest.utils import append_jsonl, dir_lock, gather_limited, read_jsonl, write_json, write_jsonl

TRACE_EVENTS = ("pick", "pool", "plan", "unit", "bind_index", "read", "bound", "reuse", "verify", "stop",
                "reader_answer", "planner_answer", "reask", "visual", "alloc", "beam", "followup", "fu_summary",
                "repair", "ground", "index", "index_use",
                "calibration",
                "direct", "chain",
                "hop7", "route", "budget_stop", "fedgrag", "answer_cut", "thought")

log = logging.getLogger(__name__)


TUNE_N = 120
HOLDOUT_N = 300
LATENCY_N = 50
CONFIRM_N = 400
LATENCY2_N = 50
CONFIRM14_N = 400
LATENCY3_N = 200
CONFIRM18_N = 400
STREAM_N = 1000
SESSIONS_TABLES = 120
SESSION_MIN, SESSION_MAX = 3, 6
RETRY_ERRORS = 2
EXECUTOR_THREADS = 128
CANARY = [{"role": "user", "content": "Reply with the word OK."}]
NEXT_SPLITS = splits.NEXT_SPLITS


def load_questions(federation: str, split: str, limit: int | None = None) -> list[Question]:
    path = splits.split_path(federation, split)
    if split in NEXT_SPLITS and not path.exists():
        raise FileNotFoundError(f"{federation} has no {split} split ({path}): the splits "
                                f"({', '.join(NEXT_SPLITS)}) are drawn once by `fedquest data` (`fedquest "
                                "data splits`), never on first use")
    if split in DRAWN and not path.exists():
        tmp = path.with_name(f"{path.name}.{os.getpid()}.draw")
        write_jsonl(tmp, (q.to_dict() for q in _draw(federation, split)))
        if path.exists():
            tmp.unlink(missing_ok=True)
        else:
            tmp.replace(path)
    qs = [Question.from_dict(d) for d in read_jsonl(path)]
    return qs[:limit] if limit else qs


def tune_split(federation: str, n: int = TUNE_N) -> list[Question]:
    return _drawn(federation, "tune", n, set())


def holdout_split(federation: str, n: int = HOLDOUT_N) -> list[Question]:
    return _drawn(federation, "holdout", n, {q.qid for q in load_questions(federation, "tune")})


def latency_split(federation: str, n: int = LATENCY_N) -> list[Question]:
    used = {q.qid for split in ("tune", "holdout") for q in load_questions(federation, split)}
    return _drawn(federation, "latency", n, used)


def confirm_split(federation: str, n: int = CONFIRM_N) -> list[Question]:
    used = {q.qid for split in ("tune", "holdout", "latency") for q in load_questions(federation, split)}
    return _drawn(federation, "confirm", n, used)


def latency2_split(federation: str, n: int = LATENCY2_N) -> list[Question]:
    used = {q.qid for split in ("tune", "holdout", "latency", "confirm") for q in load_questions(federation, split)}
    return _drawn(federation, "latency2", n, used)


def confirm14_split(federation: str, n: int = CONFIRM14_N) -> list[Question]:
    return _drawn(federation, "confirm14", n, set(), source="train")


def latency3_split(federation: str, n: int = LATENCY3_N) -> list[Question]:
    return _drawn(federation, "latency3", n, {q.qid for q in load_questions(federation, "confirm14")}, source="train")


def confirm18_split(federation: str, n: int = CONFIRM18_N) -> list[Question]:
    used = {q.qid for split in ("confirm14", "latency3") for q in load_questions(federation, split)}
    return _drawn(federation, "confirm18", n, used, source="train")


def stream_split(federation: str, n: int = STREAM_N) -> list[Question]:
    import random

    cfg = _mmqa(federation, "stream")
    held = {q.qid for q in load_questions(federation, "test")}
    pool = sorted((q for q in mmqa.load_questions(cfg.questions.split) if q.qid not in held), key=lambda q: q.qid)
    return random.Random(f"{cfg.questions.seed}:stream").sample(pool, min(n, len(pool)))


def sessions_split(federation: str, tables: int = SESSIONS_TABLES) -> list[Question]:
    import random

    cfg = _mmqa(federation, "sessions")
    held = {q.qid for q in load_questions(federation, "test")}
    by_table: dict[str, list[Question]] = {}
    for q in sorted(mmqa.load_questions(cfg.questions.split), key=lambda q: q.qid):
        if q.qid not in held and q.session:
            by_table.setdefault(q.session, []).append(q)
    eligible = sorted(t for t, qs in by_table.items() if len(qs) >= SESSION_MIN)
    chosen = random.Random(f"{cfg.questions.seed}:sessions").sample(eligible, min(tables, len(eligible)))
    return [q for t in chosen for q in by_table[t][:SESSION_MAX]]


def _mmqa(federation: str, name: str) -> Any:
    cfg = load_federation(federation)
    if cfg.dataset != "mmqa":
        raise ValueError(f"{federation}: only MMQA federations have a {name} split (MuSiQue's stream is its test split: "
                         "all 1,000 HippoRAG questions)")
    return cfg


def _musique_only(federation: str) -> list[Question]:
    raise ValueError(f"{federation}: only MuSiQue federations have locality streams (fedquest.data.locality)")


DRAWN = {"tune": tune_split, "holdout": holdout_split, "latency": latency_split, "confirm": confirm_split,
         "latency2": latency2_split, "confirm14": confirm14_split, "latency3": latency3_split,
         "confirm18": confirm18_split,
         "stream": stream_split, "sessions": sessions_split,
         **dict.fromkeys(("loc_eval", "loc00", "loc25", "loc50", "loc75", "loc100", "loc_smoke"), _musique_only)}


def _draw(federation: str, split: str) -> list[Question]:
    cfg = load_federation(federation)
    if cfg.dataset != "musique":
        return DRAWN[split](federation)
    from fedquest.data import locality, musique
    from fedquest.data.musique import SPLITS
    from fedquest.data.prepare import musique_splits

    if split in locality.SPLITS:
        return locality.locality_splits(musique.load_questions(cfg.questions.split), cfg.questions.seed)[split]
    if split not in SPLITS:
        raise ValueError(f"{federation}: a MuSiQue federation has the splits test and calib (`fedquest data prepare`) "
                         f"and {' and '.join(SPLITS)}, not {split}")
    return musique_splits(cfg)[split]


def _drawn(federation: str, name: str, n: int, exclude: set[str], source: str | None = None) -> list[Question]:
    import random

    cfg = load_federation(federation)
    if cfg.dataset != "mmqa":
        raise ValueError(f"{federation}: only MMQA federations have a {name} split (MRAMG tests on every question)")
    test = load_questions(federation, "test")
    held = {q.qid for q in test} | exclude
    candidates = mmqa.load_questions(source or cfg.questions.split)
    if source == "train":
        calib = load_questions(federation, "calib")
        held |= {q.qid for q in calib}
        tables = {q.meta.get("table_id", "") for q in calib} - {""}
        candidates = [q for q in candidates if q.meta.get("table_id", "") not in tables]
    pool: dict[str, list[Question]] = {}
    for q in sorted(candidates, key=lambda q: q.qid):
        if q.qid not in held:
            pool.setdefault(q.qtype, []).append(q)
    types = Counter(q.qtype for q in test)
    share = {t: n * c / len(test) for t, c in types.items()}
    quota = {t: int(s) for t, s in share.items()}
    for t in sorted(share, key=lambda t: (quota[t] - share[t], t))[: n - sum(quota.values())]:
        quota[t] += 1
    rng = random.Random(f"{cfg.questions.seed}:{name}")
    out = [q for t in sorted(quota) for q in rng.sample(pool.get(t, []), min(quota[t], len(pool.get(t, []))))]
    rng.shuffle(out)
    return out


def stratified_prefix(questions: list[Question], n: int) -> list[Question]:
    types = Counter(q.qtype for q in questions)
    if len(types) <= 1 or n >= len(questions):
        return questions[:n]
    share = {t: n * c / len(questions) for t, c in types.items()}
    quota = {t: int(s) for t, s in share.items()}
    for t in sorted(share, key=lambda t: (quota[t] - share[t], t))[: n - sum(quota.values())]:
        quota[t] += 1
    taken: Counter[str] = Counter()
    out = []
    for q in questions:
        if taken[q.qtype] < quota[q.qtype]:
            taken[q.qtype] += 1
            out.append(q)
    return out


def run_questions(exp: ExperimentCfg, limit: int | None = None, qids: list[str] | None = None,
                  families: list[str] | None = None) -> list[Question]:
    questions = load_questions(exp.federation, exp.split)
    if exp.outside_prefix:
        inside = {q.qid for q in stratified_prefix(questions, exp.outside_prefix)}
        questions = [q for q in questions if q.qid not in inside]
    families = families or exp.families
    if qids:
        missing = set(qids) - {q.qid for q in questions}
        if missing:
            raise KeyError(f"{exp.split} split of {exp.federation} has no questions {sorted(missing)}")
        questions = [q for q in questions if q.qid in set(qids)]
    n = limit or exp.limit
    if exp.workload == "session":
        return workloads.sessions(questions)[:n]
    chosen = stratified_prefix(questions, n) if n else questions
    return [q for q in chosen if q.meta.get("family") in families] if families else chosen


def _answer_text(value: Any) -> str:
    if isinstance(value, dict):
        value = next((value[k] for k in ("pred", "prediction", "answer") if k in value), "")
    if isinstance(value, list):
        return "; ".join(_answer_text(v) for v in value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return "" if value is None else str(value)


def import_predictions(path: Path, exp: ExperimentCfg, label: str = CLOSED_POOL) -> tuple[Path, int, int]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        preds = {str(k): v for k, v in data.items()}
    elif isinstance(data, list):
        preds = {}
        for d in data:
            qid = d.get("qid", d.get("question_id")) if isinstance(d, dict) else None
            if qid is None:
                raise ValueError(f"{path}: a prediction without qid or question_id: {str(d)[:120]}")
            preds[str(qid)] = d
    else:
        raise ValueError(f"{path}: expected a dict qid -> answer or a list of predictions, not {type(data).__name__}")
    questions = run_questions(exp)
    covered = [q for q in questions if q.qid in preds]
    zero = Ledger().summary()
    extra = {"workload": "cold", "ns": "", "federation": exp.federation, "budget": None, "tokens": None,
             "params": {"closed_pool": True, "source": Path(path).name}, "closed_pool": True}
    target = results_dir(exp.name) / "rows" / f"{label}.jsonl"
    write_jsonl(target, (_row(q, MethodResult(qid=q.qid, method=label, answer=_answer_text(preds[q.qid]),
                                              ledger=dict(zero)), label, extra) for q in covered))
    return target, len(covered), len(questions) - len(covered)


def expand(exp: ExperimentCfg, only: list[str] | None = None) -> list[tuple[str, MethodSpec, dict[str, Any]]]:
    import fedquest.methods  # noqa: F401

    out, seen = [], set()
    presets: dict[str, dict[str, Any]] | None = None
    for spec in exp.methods:
        if spec.method not in METHODS:
            raise KeyError(f"unknown method {spec.method!r} in {exp.name}")
        given = dict(spec.params)
        if spec.preset is not None:
            presets = load_presets()[0] if presets is None else presets
            if spec.preset not in presets:
                raise KeyError(f"{exp.name}: {spec.label or spec.method} names preset {spec.preset!r}, which "
                               f"{presets_path()} does not define (presets: {', '.join(sorted(presets)) or 'none'})")
            given = {**presets[spec.preset], **given}
        budgeted = (method_accepts(spec.method, "budget") and "budget" not in given
                    and "budget" not in METHODS[spec.method][1])
        if spec.sweep and not budgeted:
            raise ValueError(f"{exp.name}: {spec.label or spec.method} sweeps the budgets, but {spec.method} takes no "
                             "budget or sets its own (FedQuest's budget is a param: {tokens: 2000})")
        for b in exp.budgets if spec.sweep else [max(exp.budgets)]:
            label = spec.name_for(b if spec.sweep else None)
            if label in seen:
                raise ValueError(f"{exp.name}: two runs are labeled {label!r}")
            seen.add(label)
            if only and label not in only and (spec.label or spec.method) not in only:
                continue
            out.append((label, spec, {**given, "budget": b} if budgeted else dict(given)))
    return out


def presets_path() -> Path:
    return config_module.CONFIGS / PRESETS_FILE


def load_presets(path: Path | None = None) -> tuple[dict[str, dict[str, Any]], dict[str, Any], str | None]:
    path = path or presets_path()
    if not path.exists():
        return {}, {}, None
    raw = path.read_bytes()
    doc = yaml.safe_load(raw) or {}
    return dict(doc.get("presets") or {}), dict(doc.get("gates") or {}), hashlib.sha256(raw).hexdigest()


def resolved_params(method: str, params: dict[str, Any]) -> dict[str, Any]:
    cls, defaults = METHODS[method]
    signature = inspect.signature(cls.__init__).parameters.values()
    return {**{p.name: p.default for p in signature if p.default is not p.empty}, **defaults, **params}


def _as_stored(params: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(params, sort_keys=True, default=str))


_UNSET = "(unset)"


def _recorded_params(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: dict[str, dict[str, Any]] = {}
    for r in rows:
        if r.get("params") is not None:
            key = json.dumps(r["params"], sort_keys=True, default=str)
            if key not in seen:
                seen[key] = json.loads(key)
    return list(seen.values())


def _differs(ran: dict[str, Any], now: dict[str, Any], key: str, added: dict[str, Any]) -> bool:
    if key in ran and key in now:
        return ran[key] != now[key]
    value = ran[key] if key in ran else now[key]
    return key not in added or value != added[key]


def _redefined(labels: dict[str, dict[str, Any]], recorded: dict[str, Any], rows_dir: Path,
               added: dict[str, dict[str, Any]] | None = None,
               ) -> tuple[dict[str, dict[str, tuple[str, str]]], list[str]]:
    def shown(params: dict[str, Any], key: str) -> str:
        return json.dumps(params[key]) if key in params else _UNSET

    files: dict[str, list[Path]] = {}
    for p in sorted(rows_dir.glob("*.jsonl")):
        files.setdefault(p.stem.split("+")[0], []).append(p)
    changed: dict[str, dict[str, tuple[str, str]]] = {}
    unrecorded = []
    for label, params in labels.items():
        if label not in files:
            continue
        ran_all = (_recorded_params(r for p in files[label] for r in read_jsonl(p))
                   or ([_as_stored(recorded[label])] if label in recorded else []))
        if not ran_all:
            unrecorded.append(label)
            continue
        now = _as_stored(params)
        neutral = _as_stored((added or {}).get(label) or {})
        ran = next((x for x in ran_all if any(_differs(x, now, k, neutral) for k in set(x) | set(now))), None)
        if ran is not None:
            changed[label] = {k: (shown(ran, k), shown(now, k)) for k in sorted(set(ran) | set(now))
                              if _differs(ran, now, k, neutral)}
    return changed, unrecorded


def heal_rows(path: Path) -> None:
    try:
        with open(path, "rb+") as f:
            size = f.seek(0, os.SEEK_END)
            if size == 0:
                return
            f.seek(size - 1)
            if f.read(1) == b"\n":
                return
            pos, cut = size, 0
            while pos > 0:
                start = max(0, pos - 65536)
                f.seek(start)
                i = f.read(pos - start).rfind(b"\n")
                if i >= 0:
                    cut = start + i + 1
                    break
                pos = start
            f.seek(cut)
            tail = f.read()
            try:
                json.loads(tail)
            except ValueError:
                f.truncate(cut)
                log.warning("%s: its torn last line (%d bytes, a run killed while writing it) is dropped", path,
                            len(tail))
                return
            f.seek(0, os.SEEK_END)
            f.write(b"\n")
    except FileNotFoundError:
        return


def _done(path: Path, retry_errors: bool = False, max_tries: int | None = None) -> set[tuple[str, int]]:
    if not path.exists():
        return set()
    heal_rows(path)
    rows = list(read_jsonl(path))
    failed = Counter((d["qid"], d.get("agent", 0)) for d in rows
                     if d.get("error") and not auth_error(d["error"]) and not network_error(d["error"]))
    return {(d["qid"], d.get("agent", 0)) for d in rows
            if not d.get("error") or (retry_errors is False) or (
                not auth_error(d["error"]) and not network_error(d["error"]) and max_tries is not None
                and failed[(d["qid"], d.get("agent", 0))] >= max_tries)}


def _extraction_errors(rows_dir: Path, label: str) -> int:
    files = [p for p in rows_dir.glob("*.jsonl") if p.stem == label or p.stem.startswith(f"{label}+")]
    return sum((d.get("ledger") or {}).get("extraction_errors", 0) for p in files for d in read_jsonl(p))


def _calibration(exp: ExperimentCfg, federation: str) -> Path | None:
    if exp.calibration is None:
        return None
    if exp.calibration == "auto":
        return calibration_file(federation)
    if "/" not in exp.calibration and not exp.calibration.endswith(".json"):
        return calibration_path(exp.calibration)
    return Path(exp.calibration)


def _calibrator(path: Path | None, federation: str) -> Calibrator:
    if path is not None and path.exists():
        return Calibrator.load(path)
    if path is not None:
        log.warning("no calibration at %s: FedQuest's candidates on %s are all equally likely (run `fedquest "
                    "calibrate %s`)", path, federation, federation)
    return Calibrator()


def _digest(path: Path | None) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16] if path is not None and path.exists() else None


def _git_commit() -> str:
    import subprocess

    here = Path(__file__).resolve().parent
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=here, capture_output=True, text=True, timeout=10)
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=here, capture_output=True,
                               text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return ""
    if out.returncode != 0:
        return ""
    return out.stdout.strip() + ("-dirty" if dirty.returncode == 0 and dirty.stdout.strip() else "")


def _expected_models(mode: str, cfg: Any, llm: LLMClient, profile: str | None = None) -> dict[str, Any]:
    import fedquest.federation.base as wire

    expected = getattr(wire, "expected_site_models", None)
    if mode in ("local", "replay") or expected is None \
            or "expected_models" not in inspect.signature(require_current).parameters:
        return {}
    if profile is not None:
        return {"expected_models": expected(cfg, llm.cfg, profile=profile)}
    return {"expected_models": expected(cfg, llm.cfg)}


def kg_backend(exp: ExperimentCfg) -> str:
    kind = os.environ.get("FEDQUEST_KG", "").strip() or exp.kg
    if kind not in ("memory", "neo4j"):
        raise ValueError(f"FEDQUEST_KG={kind!r}: the knowledge graph's backend is memory or neo4j")
    return kind


def label_namespace(exp: ExperimentCfg, label: str, method: Method) -> tuple[str | None, bool]:
    if method.needs_build:
        return None, False
    named = getattr(method, "kg_namespace", None)
    if named:
        return str(named), False
    if getattr(method, "persistent_kg", False):
        return f"{exp.name}/{label}", True
    return None, False


def persistent_labels(exp: ExperimentCfg) -> dict[str, str]:
    out = {}
    for label, spec, params in expand(exp):
        ns, _ = label_namespace(exp, label, build_method(spec.method, params))
        if ns is not None:
            out[label] = ns
    return out


async def _site_models(fed: Federation) -> dict[str, dict[str, str]] | None:
    method = getattr(fed, "site_models", None)
    if method is None:
        return None
    out = method()
    return await out if inspect.isawaitable(out) else out


def _row(q: Question, r: MethodResult, label: str, extra: dict[str, Any]) -> dict[str, Any]:
    row = r.to_dict()
    row["trace"] = [e for e in r.trace if e.get("event") in TRACE_EVENTS]
    row.update(label=label, question=q.question, golds=q.answers, qtype=q.qtype, family=q.meta.get("family", ""),
               gold_doc_ids=q.gold_doc_ids, bridge=q.bridge, image_answer=q.image_answer, **extra)
    return row


def _object_refs(federation: str) -> list[tuple[str, str]]:
    cfg = load_federation(federation)
    return [(s.name, d["oid"]) for s in cfg.sites for c in s.collections
            for d in read_jsonl(collection_dir(cfg.data_name, s.name, c.name) / "objects.jsonl")]


class Federations:
    def __init__(self, mode: str, llm: LLMClient, stack: AsyncExitStack) -> None:
        self.mode, self.llm, self.stack = mode, llm, stack
        self.network: NetworkCfg | None = None
        self.live_wire = False
        self.profile: str | None = None
        self._feds: dict[str, Federation] = {}
        self._client: Any = None
        self._lock = asyncio.Lock()

    async def get(self, name: str) -> Federation:
        async with self._lock:
            if name not in self._feds:
                if self.mode == "local":
                    from fedquest.federation.local import open_local_federation

                    fed = await asyncio.to_thread(open_local_federation, name, llm=self.llm)
                    self.stack.push_async_callback(fed.close)
                elif self.mode == "replay":
                    from fedquest.federation.replay import replay_federation

                    fed = replay_federation(name)
                else:
                    from fedquest.federation.gc import open_gc_federation

                    fed = open_gc_federation(name)
                fed = delayed(fed, self.network)
                fed.live = (self.llm.live and self.mode != "replay") or self.live_wire
                fed.profile = self.profile
                self._feds[name] = fed
            return self._feds[name]


def open_federations(mode: str, llm: LLMClient, stack: AsyncExitStack) -> Federations:
    return Federations(mode, llm, stack)


async def run_experiment(exp: ExperimentCfg, *, mode: str = "local", methods: list[str] | None = None,
                         limit: int | None = None, resume: bool = True, qids: list[str] | None = None,
                         families: list[str] | None = None) -> Path:
    out = results_dir(exp.name)
    rows_dir = out / "rows"
    _default_executor()
    timing_log, timing_from = _timing_files(exp, out)
    timing.configure(timing_log, timing_from)
    if os.environ.get("FEDQUEST_LLM_TIMED", "") == "1":
        timing.lookup("")
    questions = run_questions(exp, limit, qids, families)
    specs = expand(exp, methods)
    federations = dict.fromkeys(spec.federation or exp.federation for _, spec, _ in specs)
    calibrations = {f: _calibration(exp, f) for f in federations}
    missing = [f for f, p in calibrations.items() if p is None or not p.exists()] if exp.require_calibration else []
    if missing:
        raise RuntimeError(f"{exp.name} needs a calibration for " + ", ".join(
            f"{f} ({calibrations[f]} is missing)" if calibrations[f] else f"{f} (the experiment sets calibration: null)"
            for f in missing) + ": a portable one is committed under configs/calibration/ (its README regenerates "
                                "it), a layout's own comes from `fedquest calibrate <federation>`")
    log.info("%s: %d questions x %d runs (%s workload, %s federation)", exp.name, len(questions), len(specs),
             exp.workload, mode)
    split_file = questions_path(exp.federation, exp.split)
    meta = {"experiment": exp.model_dump(mode="json"), "mode": mode,
            "questions": len(run_questions(exp, limit)),
            "questions_sha256": hashlib.sha256(split_file.read_bytes()).hexdigest() if split_file.exists() else None,
            "git_commit": _git_commit(),
            "labels": {label: resolved_params(spec.method, params) for label, spec, params in specs},
            "shard": os.environ.get("FEDQUEST_SHARD") or None,
            "calibration": {f: {"file": str(p) if p else None, "sha256": _digest(p),
                                "costs": _digest(p.with_suffix(".costs.json") if p else None)}
                            for f, p in calibrations.items()},
            "python": platform.python_version(), "started": time.time()}
    from fedquest import budget

    meta["token_counts"] = {"exact": budget.counter().exact, "tokenizer": budget.counter().path}
    if any(spec.preset for _, spec, _ in specs):
        meta["presets"] = {"file": str(presets_path()), "sha256": load_presets()[2]}
    flags = {**_llm_flags(mode), "timing_log": str(timing_log), "timing_from": [str(p) for p in timing_from]}
    meta["llm_flags"] = flags
    models_path = Path(exp.models) if exp.models else None
    meta["models_sha"] = models_sha(_guarded_models(load_models(models_path)))
    kg_kind = kg_backend(exp)
    if kg_kind != exp.kg:
        meta["kg"] = kg_kind
    if exp.stream_batch:
        meta["barrier_dir"] = str(barrier_dir())
    before = json.loads((out / "meta.json").read_text()) if (out / "meta.json").exists() else {}
    if resume:
        mine = {label for label, _, _ in specs}
        for p in rows_dir.glob("*.jsonl"):
            if p.stem.split("+")[0] in mine:
                heal_rows(p)
    if resume and before and any(rows_dir.glob("*.jsonl")):
        was = before.get("experiment", {})
        recorded = before.get("calibration") or {}
        changed = [f for f, c in meta["calibration"].items() if f in recorded
                   and (recorded[f].get("sha256"), recorded[f].get("costs")) != (c["sha256"], c["costs"])]
        redrawn = before.get("questions_sha256") not in (None, meta["questions_sha256"])
        renetted = "network" in was and was["network"] != exp.network.model_dump(mode="json")
        then = before.get("llm_flags") or {}
        reflagged = bool(then) and any(then.get(k) != flags[k] for k in ("timed", "live"))
        then_models = before.get("models_sha")
        remodeled = bool(then_models) and then_models != meta["models_sha"] and \
            os.environ.get("FEDQUEST_ALLOW_MODELS_CHANGE", "") != "1"
        then_stream = (was.get("stream_batch"), was.get("stream_seed"))
        restreamed = then_stream != (exp.stream_batch, exp.stream_seed)
        if was.get("split") != exp.split or changed or redrawn or renetted or reflagged or remodeled or restreamed:
            what = (f"split {was.get('split')} -> {exp.split}" if was.get("split") != exp.split else
                    f"stream layout (stream_batch/stream_seed {then_stream[0]}/{then_stream[1]} -> "
                    f"{exp.stream_batch}/{exp.stream_seed})" if restreamed else
                    f"models configuration (models_sha {then_models[:12]} -> {meta['models_sha'][:12]}: FEDQUEST_MODELS "
                    "or the file it names changed; FEDQUEST_ALLOW_MODELS_CHANGE=1 resumes anyway)" if remodeled else
                    f"version of the {exp.split} split file ({split_file})" if redrawn else
                    f"emulated network ({was.get('network')} -> {exp.network.model_dump(mode='json')})" if renetted else
                    f"model-client mode (timed/live {then.get('timed')}/{then.get('live')} -> {flags['timed']}/"
                    f"{flags['live']})" if reflagged else
                    f"calibration or read costs of {', '.join(changed)}")
            raise RuntimeError(f"{out} holds rows of another {what}: start clean (move the named rows files away, or rerun with --no-resume) "
                               "or rerun the labels with --no-resume")
        redefined, unrecorded = _redefined(meta["labels"], before.get("labels") or {}, rows_dir,
                                           {label: METHODS[spec.method][0].added_params for label, spec, _ in specs})
        if redefined:
            diffs = "; ".join(f"{label}: " + ", ".join(f"{k} {then} -> {now}" for k, (then, now) in keys.items())
                              for label, keys in redefined.items())
            files = ", ".join(f"{d}/{p.name}" for label in redefined for d in ("rows", "rows_scored")
                              for p in sorted((out / d).glob("*.jsonl")) if p.stem.split("+")[0] == label)
            raise RuntimeError(f"{out}: the rows of {', '.join(redefined)} ran another definition of "
                               f"{'that label' if len(redefined) == 1 else 'those labels'} (resolved parameters then "
                               f"-> now: {diffs}); resuming would keep them under the new one. Delete those labels' "
                               f"rows ({files}), start clean (move the named rows files away), or rerun the labels with "
                               "--no-resume")
        for label in unrecorded:
            log.warning("%s: %s has rows that recorded no parameters, nor does meta.json: resuming them unchecked",
                        out, label)
    if before:
        meta["labels"] = {**(before.get("labels") or {}), **meta["labels"]}
    async with AsyncExitStack() as stack:
        llm = LLMClient.from_env(provider="replay" if mode == "replay" else None, models_path=models_path)
        stack.push_async_callback(llm.aclose)
        if exp.network.endpoint_ms:
            llm.endpoint_s = exp.network.endpoint_ms / 1000
        meta["llm_flags"] = flags = {**flags, "timed": llm.timed, "live": llm.live, "endpoint_ms": llm.endpoint_s * 1000}
        meta["models"] = llm.cfg.roles
        _write_meta(out, meta)
        view = open_view(kg_kind)  # type: ignore[arg-type]
        stack.push_async_callback(view.close)
        matcher = build_matcher(exp.matcher)
        feds = open_federations(mode, llm, stack)
        feds.network, feds.live_wire, feds.profile = exp.network, exp.live_wire, exp.site_profile
        coords: dict[str, Coordinator] = {}
        site_models: dict[str, Any] = {}
        site_commits: dict[str, Any] = {}
        for fed_name, path in calibrations.items():
            fed_cfg = load_federation(fed_name)
            fed = await feds.get(fed_name)
            await require_current(fed, fed_cfg, **_expected_models(mode, fed_cfg, llm, exp.site_profile))
            site_models[fed_name] = await _site_models(fed)
            site_commits[fed_name] = {s: si.commit for s, si in (await fed.info()).items()}
            coords[fed_name] = Coordinator(fed, llm, view, matcher, fed_cfg.answer_style, verify=exp.matcher.verify,
                                           calib=_calibrator(path, fed_name))
        meta["site_models"] = site_models
        meta["site_commits"] = site_commits
        _write_meta(out, meta)
        canary = asyncio.create_task(_canary(llm, out / "canary.jsonl", exp.canary_s)) if exp.canary_s else None

        def info_for(label: str, spec: MethodSpec, method: Method) -> dict[str, Any]:
            return {"federation": spec.federation or exp.federation, "budget": getattr(method, "budget", None),
                    "tokens": getattr(method, "tokens", None), "params": meta["labels"][label]}

        async def one(label: str, spec: MethodSpec, params: dict[str, Any]) -> None:
            fed_name = spec.federation or exp.federation
            method = build_method(spec.method, params)
            t0 = time.perf_counter()
            busy = await _busy(coords[fed_name].fed, mode)
            if exp.stream_batch:
                await _run_stream(exp, label, spec, method, coords[fed_name], _stream_for_label(exp, questions, spec),
                                  rows_dir, resume, info_for(label, spec, method))
            else:
                await _run_label(exp, label, spec, method, coords[fed_name], _for_label(questions, spec), rows_dir,
                                 resume, info_for(label, spec, method))
            _site_stats(out, label, busy, await _busy(coords[fed_name].fed, mode))
            log.info("%s: %s done in %.1fs", exp.name, label, time.perf_counter() - t0)
            failed = _extraction_errors(rows_dir, label)
            if failed:
                log.warning("%s: %s has %d failed extractions in its rows", exp.name, label, failed)

        try:
            if exp.workload == "session":
                await asyncio.gather(*(one(*s) for s in specs))
            elif exp.workload == "cold" and exp.interleave:
                await _run_interleaved(exp, specs, coords, questions, rows_dir, resume, mode, info_for)
            else:
                for s in specs:
                    await one(*s)
        finally:
            if canary is not None:
                canary.cancel()
                with suppress(asyncio.CancelledError):
                    await canary
            timing.close()
    meta["finished"] = time.time()
    _write_meta(out, meta)
    return out


def _default_executor() -> None:
    loop = asyncio.get_running_loop()
    if not getattr(loop, "_fedquest_executor", False):
        loop.set_default_executor(ThreadPoolExecutor(EXECUTOR_THREADS, thread_name_prefix="fedquest"))
        loop._fedquest_executor = True  # type: ignore[attr-defined]


def _timing_files(exp: ExperimentCfg, out: Path) -> tuple[Path, list[Path]]:
    own = Path(os.environ.get("FEDQUEST_TIMING_LOG") or out / "timing.jsonl")
    extra = [Path(x) for x in (os.environ.get("FEDQUEST_TIMING_FROM") or "").split(os.pathsep) if x]
    sources = [results_dir(name) / "timing.jsonl" for name in exp.timing_from] + extra
    return own, sources or [own]


def _guarded_models(cfg: ModelsCfg) -> ModelsCfg:
    return cfg.model_copy(update={"provider": "alcf"}) if cfg.provider in ("fake", "replay") else cfg


def _llm_flags(mode: str) -> dict[str, Any]:
    return {"timed": os.environ.get("FEDQUEST_LLM_TIMED", "") == "1",
            "live": os.environ.get("FEDQUEST_LLM_LIVE", "") == "1",
            "endpoint_ms": float(os.environ.get("FEDQUEST_ENDPOINT_MS") or 0),
            "record_wire": os.environ.get("FEDQUEST_RECORD_WIRE") or None, "mode": mode}


def parse_shard(text: str) -> tuple[int, int]:
    i, sep, n = text.strip().partition("/")
    if not sep or not i.isdigit() or not n.isdigit() or not 0 <= int(i) < int(n):
        raise ValueError(f"shard {text!r} is not i/n with 0 <= i < n")
    return int(i), int(n)


def shard_of(qid: str, n: int) -> int:
    return int(hashlib.sha256(qid.encode()).hexdigest()[:12], 16) % n


def active_shard() -> tuple[int, int] | None:
    text = os.environ.get("FEDQUEST_SHARD", "").strip()
    return parse_shard(text) if text else None


def in_shard(questions: list[Question]) -> list[Question]:
    shard = active_shard()
    return questions if shard is None else [q for q in questions if shard_of(q.qid, shard[1]) == shard[0]]


def _for_label(questions: list[Question], spec: MethodSpec) -> list[Question]:
    return in_shard(stratified_prefix(questions, spec.limit) if spec.limit else questions)


def stream_order(exp: ExperimentCfg, questions: list[Question]) -> list[Question]:
    out = list(questions)
    if exp.stream_seed is not None:
        random.Random(f"{exp.stream_seed}:stream").shuffle(out)
    return out


def _stream_for_label(exp: ExperimentCfg, questions: list[Question], spec: MethodSpec) -> list[Question]:
    rank = {q.qid: i for i, q in enumerate(stream_order(exp, questions))}
    return sorted(stratified_prefix(questions, spec.limit) if spec.limit else questions, key=lambda q: rank[q.qid])


async def _busy(fed: Federation, mode: str) -> dict[str, Any] | None:
    if mode == "replay":
        return None
    try:
        stats = await fed.stats(busy_only=True)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
    return {site: (d or {}).get("busy") for site, d in stats.items()}


async def _busy_all(coords: dict[str, Coordinator], feds: list[str], mode: str) -> dict[str, Any] | None:
    if mode == "replay":
        return None
    out: dict[str, Any] = {}
    for f in feds:
        busy = await _busy(coords[f].fed, mode) or {}
        out.update({(site if len(feds) == 1 else f"{f}/{site}"): b for site, b in busy.items()})
    return out


def _write_meta(out: Path, meta: dict[str, Any]) -> None:
    path = out / "meta.json"
    with dir_lock(out / ".meta.lock"):
        try:
            now = json.loads(path.read_text()) if path.exists() else {}
        except (OSError, ValueError):
            now = {}
        write_json(path, {**meta, "labels": {**(now.get("labels") or {}), **(meta.get("labels") or {})}})


def _site_stats(out: Path, key: str, before: Any, after: Any) -> None:
    path = out / "site_stats.json"
    with dir_lock(out / ".site_stats.lock"):
        try:
            doc = json.loads(path.read_text()) if path.exists() else {}
        except (OSError, ValueError):
            doc = {}
        doc[key] = {"before": before, "after": after}
        write_json(path, doc)


async def _kg_stats(coord: Coordinator, out: Path, key: str, ns: str, answered: int, resumed: bool) -> None:
    if not answered:
        return
    try:
        stats = await coord.kg.stats(ns)
    except Exception as e:
        stats = {"error": f"{type(e).__name__}: {e}"}
    path = out / "kg_stats.json"
    doc = json.loads(path.read_text()) if path.exists() else {}
    doc[key] = {**stats, "questions": answered, "resumed": resumed}
    write_json(path, doc)


def _throughput(out: Path, label: str, rows: list[dict[str, Any]], concurrency: int, interleaved: bool,
                resumed: bool) -> None:
    if not rows:
        return
    first, last = min(r["started_at"] for r in rows), max(r["finished_at"] for r in rows)
    write_json(out / f"throughput_{label}.json",
               {"label": label, "questions": len(rows), "seconds": last - first, "concurrency": concurrency,
                "first_start": first, "last_finish": last, "interleaved": interleaved, "resumed": resumed})


async def _canary(llm: LLMClient, path: Path, every: float) -> None:
    while True:
        t, error = time.time(), None
        try:
            await llm.chat(CANARY, role="answer", task="canary", max_tokens=8, live=True, fallback=False)
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
        append_jsonl(path, {"t": t, "seconds": time.time() - t, "error": error})
        await asyncio.sleep(every)


async def _run_interleaved(exp: ExperimentCfg, specs: list[tuple[str, MethodSpec, dict[str, Any]]],
                           coords: dict[str, Coordinator], questions: list[Question], rows_dir: Path, resume: bool,
                           mode: str, info_for: Any) -> None:
    pairs: list[tuple[str, MethodSpec, Method, Question, dict[str, Any], Path]] = []
    spaces: dict[str, str | None] = {}
    for label, spec, params in specs:
        method = build_method(spec.method, params)
        fed_name = spec.federation or exp.federation
        if method.needs_build:
            await _build(exp, label, method, coords[fed_name], fed_name, rows_dir.parent, resume)
        path = rows_dir / f"{label}.jsonl"
        if not resume and path.exists():
            path.unlink()
        spaces[label], clearable = label_namespace(exp, label, method)
        if not resume and clearable:
            await _clear_persistent(coords[fed_name], spaces[label])
        pairs += [(label, spec, method, q, info_for(label, spec, method), path) for q in _for_label(questions, spec)]
    random.Random(f"{exp.interleave_seed or exp.name}:interleave").shuffle(pairs)
    done = {label: _done(path, retry_errors=True, max_tries=RETRY_ERRORS + 1) if resume else set()
            for label, path in {pair[0]: pair[5] for pair in pairs}.items()}
    todo = [(i, pair) for i, pair in enumerate(pairs) if (pair[3].qid, 0) not in done[pair[0]]]
    feds = sorted({spec.federation or exp.federation for _, spec, _ in specs})
    before = await _busy_all(coords, feds, mode)
    written: dict[str, list[dict[str, Any]]] = {label: [] for label, _, _ in specs}

    async def run(item: tuple[int, tuple[str, MethodSpec, Method, Question, dict[str, Any], Path]]) -> None:
        i, (label, spec, method, q, info, path) = item
        coord = coords[spec.federation or exp.federation]
        kg_ns = spaces[label]
        ns = f"{exp.name}/{label}/eager" if method.needs_build else kg_ns or f"{exp.name}/{label}/{q.qid}"
        t0 = time.time()
        r = await coord.run(method, q, ns)
        row = _row(q, r, label, {"workload": "cold", "ns": ns, **info, "started_at": t0, "finished_at": time.time(),
                                 "position": i, "kg_ns": kg_ns})
        append_jsonl(path, row)
        written[label].append(row)

    await gather_limited(todo, run, exp.concurrency)
    after = await _busy_all(coords, feds, mode)
    out = rows_dir.parent
    _site_stats(out, "__all__", before, after)
    resumed = len(todo) < len(pairs)
    for label, rows in written.items():
        _throughput(out, label, rows, exp.concurrency, True, resumed)


async def _run_label(exp: ExperimentCfg, label: str, spec: MethodSpec, method: Method, coord: Coordinator,
                     questions: list[Question], rows_dir: Path, resume: bool, info: dict[str, Any]) -> None:
    base_ns = f"{exp.name}/{label}"
    if method.needs_build:
        await _build(exp, label, method, coord, info["federation"], rows_dir.parent, resume)

    def ns_for(q: Question, stream_ns: str) -> str:
        if method.needs_build:
            return f"{base_ns}/eager"
        return stream_ns if spec.reuse else f"{stream_ns}/{q.qid}"

    if exp.workload == "cold":
        path = rows_dir / f"{label}.jsonl"
        done = _done(path, retry_errors=True, max_tries=RETRY_ERRORS + 1) if resume else set()
        if not resume and path.exists():
            path.unlink()
        kg_ns, clearable = label_namespace(exp, label, method)
        if not resume and clearable:
            await _clear_persistent(coord, kg_ns)
        todo = [q for q in questions if (q.qid, 0) not in done]
        written: list[dict[str, Any]] = []

        async def one(q: Question) -> None:
            ns = f"{base_ns}/eager" if method.needs_build else kg_ns or f"{base_ns}/{q.qid}"
            t0 = time.time()
            r = await coord.run(method, q, ns)
            row = _row(q, r, label, {"workload": "cold", "ns": ns, **info, "started_at": t0,
                                     "finished_at": time.time(), "kg_ns": kg_ns})
            append_jsonl(path, row)
            written.append(row)

        await gather_limited(todo, one, exp.concurrency)
        _throughput(rows_dir.parent, label, written, exp.concurrency, False, len(todo) < len(questions))
        return

    if exp.workload == "session":
        async def stream(order: str) -> None:
            sub = f"{label}+{order}"
            path = rows_dir / f"{sub}.jsonl"
            stream_ns = f"{exp.name}/{sub}"
            done = _done(path) if resume else set()
            if not done:
                if path.exists():
                    path.unlink()
                await coord.kg.clear(stream_ns)
            answered = 0
            for i, q in enumerate(workloads.sessions(questions, order, seed=13)):
                if (q.qid, 0) in done:
                    continue
                ns = ns_for(q, stream_ns)
                r = await coord.run(method, q, ns)
                append_jsonl(path, _row(q, r, sub, {"workload": "session", "order": order, "index": i, "ns": ns,
                                                    **info}))
                answered += 1
            await _kg_stats(coord, rows_dir.parent, sub, stream_ns, answered, bool(done))

        await asyncio.gather(*(stream(o) for o in exp.orders))
        return

    combos = [(n, s, d) for n in exp.agents for s in exp.sharing for d in exp.deals]
    combos = [c for c in combos if c[0] != 1 or c[1:] == (exp.sharing[0], exp.deals[0])]
    for n, sharing, deal in combos:
        sub = f"{label}+{sharing}{n}" + ("" if deal == "overlap" else f"+{deal}")
        path = rows_dir / f"{sub}.jsonl"
        done = _done(path) if resume else set()
        if not done and path.exists():
            path.unlink()
        streams = workloads.streams(questions, n, seed=13, deal=deal)
        answered: Counter[str] = Counter()

        async def agent(i: int, stream: list[Question], sub: str = sub, sharing: str = sharing, n: int = n,
                        deal: str = deal, done: set = done, path: Path = path, answered: Counter = answered) -> None:
            stream_ns = f"{exp.name}/{sub}" if sharing == "shared" else f"{exp.name}/{sub}/agent{i}"
            if not done:
                await coord.kg.clear(stream_ns)
            for j, q in enumerate(stream):
                if (q.qid, i) in done:
                    continue
                ns = ns_for(q, stream_ns)
                r = await coord.run(method, q, ns)
                append_jsonl(path, _row(q, r, sub, {"workload": "concurrent", "agents": n, "sharing": sharing,
                                                    "deal": deal, "agent": i, "index": j, "ns": ns, **info,
                                                    "finished_at": time.time()}))
                answered[stream_ns] += 1

        t0 = time.time()
        await asyncio.gather(*(agent(i, s) for i, s in enumerate(streams)))
        for stream_ns, count in answered.items():
            await _kg_stats(coord, rows_dir.parent, stream_ns.removeprefix(f"{exp.name}/"), stream_ns, count,
                            bool(done))
        write_json(rows_dir.parent / f"throughput_{sub}.json",
                   {"label": sub, "agents": n, "sharing": sharing, "deal": deal, "questions": len(questions),
                    "seconds": time.time() - t0})


async def _run_stream(exp: ExperimentCfg, label: str, spec: MethodSpec, method: Method, coord: Coordinator,
                      questions: list[Question], rows_dir: Path, resume: bool, info: dict[str, Any],
                      shard: tuple[int, int] | None = None) -> None:
    i, k = shard or active_shard() or (0, 1)
    size = int(exp.stream_batch or 0)
    base_ns = f"{exp.name}/{label}"
    if method.needs_build:
        await _build(exp, label, method, coord, info["federation"], rows_dir.parent, resume)
    path = rows_dir / f"{label}.jsonl"
    done = _done(path, retry_errors=True, max_tries=RETRY_ERRORS + 1) if resume else set()
    if not resume and path.exists():
        path.unlink()
    kg_ns, clearable = label_namespace(exp, label, method)
    if not resume and clearable:
        await _clear_persistent(coord, kg_ns)
    barrier = Barrier(barrier_dir() / exp.name / label, i, k)
    if not resume and barrier.forget():
        log.warning("%s: --no-resume forgets this shard's barrier markers; another shard's from an earlier run may "
                    "still let a peer pass", barrier.root)
    written: list[dict[str, Any]] = []
    record_path = rows_dir.parent / f"stream_{label}.json"
    record = _stream_record(record_path, i, k, size) if resume else {}
    skipped = 0
    for b in range(-(-len(questions) // size) if size else 0):
        mine = [(b * size + j, q) for j, q in enumerate(questions[b * size:(b + 1) * size]) if shard_of(q.qid, k) == i]
        todo = [(pos, q) for pos, q in mine if (q.qid, 0) not in done]
        skipped += len(mine) - len(todo)
        ran: dict[str, tuple[int, Question, MethodResult, float, float, StreamKG, str]] = {}
        ahead: list[int] = []
        if todo:
            stale = barrier.path(b, "ran").exists()
            barrier.unmark(b)
            ahead = sorted(barrier.past(b, ran=stale) - {i})
        if ahead:
            log.warning("%s: %s batch %d: shards %s flushed it before it ran here: its questions may see their writes "
                        "of the batch", exp.name, label, b, ahead)

        async def one(item: tuple[int, Question], ran: dict = ran) -> None:
            pos, q = item
            view = StreamKG(coord.kg)
            ns = f"{base_ns}/eager" if method.needs_build else kg_ns or f"{base_ns}/{q.qid}"
            t0 = time.time()
            r = await replace(coord, kg=view).run(method, q, ns)
            ran[q.qid] = (pos, q, r, t0, time.time(), view, ns)

        t0 = time.time()
        await gather_limited(todo, one, exp.concurrency)
        barrier.mark(b, "ran", questions=len(mine), ran=len(todo))
        t1 = time.time()
        late_ran = await barrier.wait(b, "ran")
        t2 = time.time()
        writes = 0
        for qid in sorted(ran):
            writes += await ran[qid][5].flush()
        for pos, q, r, started, finished, _, ns in sorted(ran.values(), key=lambda x: x[0]):
            row = _row(q, r, label, {"workload": "cold", "ns": ns, **info, "started_at": started,
                                     "finished_at": finished, "kg_ns": kg_ns, "stream_pos": pos, "stream_batch": b})
            append_jsonl(path, row)
            written.append(row)
        barrier.mark(b, "done", questions=len(mine), ran=len(todo), writes=writes)
        t3 = time.time()
        late_done = await barrier.wait(b, "done")
        missing = sorted(late_ran | late_done)
        if todo or missing or ahead or b not in record:
            attempts = [*record.get(b, {}).get("attempts", []),
                        {"at": round(t0, 3), "ran": len(todo), "writes": writes, "run_s": round(t1 - t0, 3),
                         "wait_ran_s": round(t2 - t1, 3), "flush_s": round(t3 - t2, 3),
                         "wait_done_s": round(time.time() - t3, 3), "missing": missing, "ahead": ahead}]
            record[b] = {"batch": b, "questions": len(mine), "ran": sum(a.get("ran", 0) for a in attempts),
                         "writes": sum(a.get("writes", 0) for a in attempts),
                         "missing": sorted({s for a in attempts for s in a.get("missing", [])}),
                         "ahead": sorted({s for a in attempts for s in a.get("ahead", [])}), "attempts": attempts}
        if todo or missing:
            log.info("%s: %s batch %d: %d of its questions here (%d run), barrier waits %.1fs + %.1fs%s", exp.name,
                     label, b, len(mine), len(todo), t2 - t1, time.time() - t3,
                     f", without shards {missing}" if missing else "")
        write_json(record_path, {"label": label, "shard": i, "shards": k, "batch_size": size, "seed": exp.stream_seed,
                                 "questions": len(questions), "barriers": str(barrier.root),
                                 "batches": [record[x] for x in sorted(record)]})
    _throughput(rows_dir.parent, label, written, exp.concurrency, False, skipped > 0)


def _stream_record(path: Path, shard: int, shards: int, size: int) -> dict[int, dict[str, Any]]:
    try:
        doc = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    if (doc.get("shard"), doc.get("shards"), doc.get("batch_size")) != (shard, shards, size):
        return {}
    keys = ("ran", "writes", "run_s", "wait_ran_s", "flush_s", "wait_done_s", "missing", "ahead")
    return {int(x["batch"]): {**x, "attempts": x.get("attempts") or [{k: x[k] for k in keys if k in x}]}
            for x in doc.get("batches") or []}


async def _clear_persistent(coord: Coordinator, ns: str | None) -> None:
    if ns is None:
        return
    if active_shard() is not None:
        log.warning("%s: --no-resume clears this persistent namespace, which every shard of its label shares: what "
                    "the other shards wrote there so far goes too", ns)
    await coord.kg.clear(ns)


async def _build(exp: ExperimentCfg, label: str, method: Method, coord: Coordinator, fed_name: str,
                 out: Path, resume: bool) -> None:
    ns = f"{exp.name}/{label}/eager"
    record = out / f"eager_build_{label}.json"
    if resume and record.exists():
        return
    await coord.kg.clear(ns)
    ledger = Ledger()
    refs = _object_refs(fed_name)
    t0 = time.perf_counter()
    await method.build(coord.fed, coord.kg, ns, ledger, refs)  # type: ignore[attr-defined]
    write_json(record, {"label": label, "objects": len(refs), "seconds": time.perf_counter() - t0,
                        "ledger": ledger.summary()})
