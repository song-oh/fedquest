# DGRAG reimplementation: settings, and where its corpus and layouts live
from __future__ import annotations

from pathlib import Path

from fedquest.paths import runs_root

ENCODER = "qwen3-0.6b"
SUBGRAPH_ENTITIES = 40
MIN_ENTITIES = 10
SUMMARY_FACTS = 80
SUMMARY_WORDS = 120
TOP_M = 5
MATCH_THRESHOLD = 0.4
RETRIEVE_THRESHOLD = 0.4
TOP_K = 60
CANDIDATES = 3
GATE_THRESHOLD = 0.8
TEMPERATURE = 0.7


def root() -> Path:
    return runs_root() / "dgrag"


def corpus_dir(dataset: str) -> Path:
    return root() / "corpus" / dataset


def layout_dir(federation: str) -> Path:
    return root() / "layouts" / federation
