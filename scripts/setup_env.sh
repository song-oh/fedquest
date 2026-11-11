#!/usr/bin/env bash
# per-machine install: uv venv, the package, and store paths in .fedquest.env
set -euo pipefail
MACHINE=${1:?usage: setup_env.sh local|aurora|midway3}
cd "$(dirname "$0")/.."

env_set() {
  grep -v "^$1=" .fedquest.env 2>/dev/null > .fedquest.env.tmp || true
  echo "$1=$2" >> .fedquest.env.tmp && mv .fedquest.env.tmp .fedquest.env
}
STORE=${FEDQUEST_STORE:-}
if [[ $MACHINE == aurora ]]; then STORE=${STORE:-/flare/${FEDQUEST_PROJECT:-radix-io}/$USER/fedquest}; fi
RUNS_NAME=${FEDQUEST_RUNS_NAME:-runs}
if [[ ! $RUNS_NAME =~ ^[A-Za-z0-9_][A-Za-z0-9_.-]*$ ]]; then
  echo "FEDQUEST_RUNS_NAME is a directory name: $RUNS_NAME"; exit 1
fi
if [[ -n $STORE ]]; then
  mkdir -p "$STORE"
  env_set FEDQUEST_DATA "$STORE/data"
  env_set FEDQUEST_RUNS "$STORE/$RUNS_NAME"
  env_set HF_HOME "$STORE/hf"
  export UV_CACHE_DIR=${UV_CACHE_DIR:-$STORE/cache/uv} UV_LINK_MODE=copy
fi

if ! command -v uv >/dev/null 2>&1; then
  if [[ $MACHINE == midway3 ]] && command -v module >/dev/null 2>&1; then
    set +eu; module load uv; set -eu
  fi
  command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
if [[ $MACHINE == midway3 && -z ${UV_CACHE_DIR:-} ]]; then
  export UV_CACHE_DIR=/scratch/midway3/$USER/uv-cache
fi

VENV=.venv
if [[ $MACHINE == aurora ]]; then
  VENV=$STORE/venv
  export UV_PYTHON_INSTALL_DIR=${UV_PYTHON_INSTALL_DIR:-$STORE/uv/python}
fi
uv venv --clear --python 3.12 "$VENV"
if [[ $VENV != .venv ]]; then
  if [[ -e .venv && ! -L .venv ]]; then rm -rf .venv; fi
  ln -sfn "$VENV" .venv
fi
PY=.venv/bin/python
if [[ $MACHINE == local ]]; then
  uv pip install --python $PY -e ".[dev]"
else
  uv pip install --python $PY torch --index-url https://download.pytorch.org/whl/cpu
  uv pip install --python $PY -e ".[endpoint,dev]"
fi
.venv/bin/fedquest llm auth --download-only
if [[ -n $STORE ]]; then echo "data, runs and model weights go to $STORE (recorded in .fedquest.env)"; fi
echo "FedQuest environment ready: source .venv/bin/activate"
