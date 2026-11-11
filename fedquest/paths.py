from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CONFIGS = REPO / "configs"
PRESETS_FILE = "presets.yaml"


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip().removeprefix("export ").strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


_load_env_file(REPO / ".fedquest.env")


def data_root() -> Path:
    return Path(os.environ.get("FEDQUEST_DATA", REPO / "data")).expanduser()


def runs_root() -> Path:
    return Path(os.environ.get("FEDQUEST_RUNS", REPO / "runs")).expanduser()


def raw_dir(dataset: str) -> Path:
    return data_root() / "raw" / dataset


def fed_dir(federation: str) -> Path:
    return data_root() / "fed" / federation


def questions_path(federation: str, split: str) -> Path:
    return fed_dir(federation) / "questions" / f"{split}.jsonl"


def site_dir(federation: str, site: str) -> Path:
    return fed_dir(federation) / "sites" / site


def collection_dir(federation: str, site: str, collection: str) -> Path:
    return site_dir(federation, site) / collection


def llm_cache_path() -> Path:
    return Path(os.environ.get("FEDQUEST_LLM_CACHE", runs_root() / "cache" / "llm.sqlite")).expanduser()


def wire_log_path() -> Path:
    return Path(os.environ.get("FEDQUEST_REPLAY_WIRE", runs_root() / "cache" / "wire.sqlite")).expanduser()


def calibration_path(federation: str) -> Path:
    return runs_root() / "calibration" / f"{federation}.json"


def results_dir(experiment: str) -> Path:
    return runs_root() / "results" / experiment


def deploy_state_path() -> Path:
    return runs_root() / "deploy" / "state.json"
