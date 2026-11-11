# FedQuest

Code for the submission *FedQuest: Graph-Guided Joins over Federated Vector Databases for Multi-Hop Queries*.

FedQuest answers multi-hop questions over vector databases that keep their own data and encoders. It compiles a question into a query graph and runs each hop as a filtered vector search keyed by the values that earlier hops extracted. The coordinator merges the databases' results with anchor-normalized scores, reads the most promising candidates under one token budget, and records each extracted fact with its source in a knowledge graph.

## Layout

```
fedquest/
  methods/        FedQuest (fedquest.py, compiler.py) and the baselines
  site/           a database site: filtered search, reads, fact extraction
  federation/     the coordinator's view of the databases (in-process or remote)
  remote/         Globus Compute transport: gateway and site servers
  vdb/            Qdrant indexes and anchor-normalized scores
  planner/        score calibration and read scheduling
  kg/             knowledge graph, provenance and fact reuse (Neo4j)
  fedgrag/        FedGRAG reimplementation
  dgrag/          DGRAG reimplementation
  eval/           experiment runner, scoring, analyses, report
  pipeline/       driver that runs every experiment
  routing.py      database routing
  budget.py       token accounting
configs/
  experiments/    one file per study, written by scripts/make_configs.py
  federations/    databases, encoders and data placement
  models/         model roles and endpoints
  calibration/    fitted score calibrations
scripts/          data preparation, calibration, deployment
deploy/globus/    Globus Compute endpoint templates for Aurora and Midway3
tests/            offline tests
```

Baselines: Vanilla RAG (`vanilla.py`), MGRAG and MGRAG+FU (`mgrag.py`, `mgrag_fu.py`), HippoRAG 2 (`hipporag2.py`,
`site/hippo.py`), FedGRAG, DGRAG, IRCoT (`iterrag.py`), ReAct (`llm_planner.py`) and ReWOO (`rewoo.py`).

In a federation's name, `s<N>` is the number of databases, `k05` a topic-skewed placement (Dirichlet 0.5) and `r<i>`
the placement seed; `*_home` pools a benchmark in one database. `mmqa_fed`, `musique_next`, `twowiki_fed` and
`hotpot_fed` hold the prepared data that the other federations read.

## Setup

Python 3.12:

```bash
pip install -e ".[dev]"
pytest
```

The tests use scripted databases and a fake LLM, so they need no data or model endpoint. `fedquest --help` lists the commands.

Model calls go to OpenAI-compatible endpoints. `configs/models/final.yaml` is the paper's setup: Gemma 4 31B in every role, with gpt-oss-120b and Nemotron 3 Ultra as judges, served by the ALCF inference service (`fedquest llm auth`). `source scripts/env.sh` selects it (`FEDQUEST_MODELS`). For other endpoints, set `provider: openai`, `base_url` and `api_key_env` there. Token counts use Gemma 4's tokenizer (`FEDQUEST_TOKENIZER`, the path to its `tokenizer.json`). Data and results go to `FEDQUEST_DATA` and `FEDQUEST_RUNS` (default `data/` and `runs/`).

## Data

The benchmarks are public: [MultiModalQA](https://github.com/allenai/multimodalqa) (dev split) and
[HippoRAG 2's releases](https://huggingface.co/datasets/osunlp/HippoRAG_2) of
[MuSiQue](https://github.com/StonyBrookNLP/musique), [2WikiMultiHopQA](https://github.com/Alab-NII/2wikimultihop)
and [HotpotQA](https://hotpotqa.github.io). MuSiQue's full release supplies its calibration and held-out questions,
and [MGRAG](https://github.com/Wyb0627/MGRAG) the captions of MultiModalQA's images. On each facility,
`prepare_data.sh` downloads them, checks the question and corpus files against pinned SHA-256 hashes, writes the facility's databases and embeds them with their encoders (BGE and CLIP at Aurora, E5 at Midway3). The coordinator then writes the question splits:

```bash
bash scripts/prepare_data.sh aurora
fedquest data splits
python scripts/make_pools.py
```

`make_pools.py` checks the evaluation pools against `configs/pools.sha256`, and `fedquest data splits` records the splits' hashes in `configs/splits/splits.sha256`, so `git diff configs` shows any question file that differs from the paper's. HippoRAG 2's index, whose OpenIE triples FedGRAG and DGRAG also use, is built per pooled corpus
(`musique_next_home`, `twowiki_home`, `hotpot_home`) with `fedquest hippo index <home>`, or on 16 Aurora nodes with `qsub -l select=16 -l walltime=02:00:00 -q capacity scripts/hippo/hippo_index.pbs`.

The calibrations in `configs/calibration/` are frozen. `bash scripts/calibrate.sh mmqa|musique` reruns their probes and refits the rank (`*_v4.json`) and anchor-z (`*_z.json`) calibrations.

## Running

The paper's federation has four databases, two at ALCF Aurora (PBS) and two at UChicago Midway3 (Slurm), each facility behind one Globus Compute endpoint. The endpoint templates expect the clone at `~/fedquest`. On each facility's login node:

```bash
bash scripts/setup_env.sh aurora
source .venv/bin/activate
fedquest llm auth
bash scripts/qdrant.sh install
bash scripts/facility.sh up
```

(`midway3` on Midway3.) `facility.sh up` prints the endpoint id. Then, on the Aurora login node:

```bash
bash scripts/coord.sh start MIDWAY3_ENDPOINT_ID
bash scripts/coord.sh status
```

`coord.sh` starts the gateway, Neo4j and the pipeline driver in the background. The driver builds the routing fits and the FedGRAG and DGRAG indexes (`configs/tools.json`), runs and scores every experiment, retries failed questions, and ends with `scripts/post.sh`. Each experiment writes its per-question rows and `summary.csv` to `runs/results/<experiment>/`; the report is `runs/results/report/report.md`. The full run makes about one million model calls.

A single study runs on the same deployment once the gateway is up (`bash scripts/remote.sh start`, then `bash scripts/remote.sh up`):

```bash
fedquest eval run main_musique --mode remote
fedquest eval score main_musique --judge2
```

With `--mode local`, every database runs in the coordinator's process instead. This needs a Qdrant server (`bash scripts/qdrant.sh start`), both facilities' data on one machine with their vectors loaded (`fedquest data index` for the `*_home` and parent federations), and the tool tasks that the study lists under `after:`, run with `--mode local`.

## Experiments

| Paper | Experiments |
|---|---|
| Table I; Figs. 9–11 | `main_*`, `dgrag_*`, `hippo_*` |
| Fig. 8 | `mx_*`, `main_mmqa`, `main_musique` |
| Fig. 12 | `main_mmqa`, `main_musique`, `frontier_musique` |
| Fig. 13 | `frontier_*` |
| Fig. 14 | `tool:merge_study`, `scale_musique`, `seeds_musique` |
| Fig. 15 | `main_*`, `frontier_*` |
| Fig. 16 | `scale_musique` |
| Fig. 17 | `latency_mmqa`, `latency_musique` |
| Table II | `kgu_warm_*`, `kgu_after_*`, `session_musique` |
| Table III | `abl_*` |
| Pooled vs. split corpora (Sec. V-F) | `dist_*` |
| Held-out questions | `heldout_musique`, `hippo_heldout_musique` |

`*` is `mmqa`, `musique`, `twowiki` or `hotpot`. The `smoke_*` studies run every method on two questions.
