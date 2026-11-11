if [ -n "${BASH_SOURCE:-}" ]; then _fq_src=${BASH_SOURCE[0]}; else _fq_src=$0; fi
_fq_root=$(cd "$(dirname "$_fq_src")/.." && pwd)
cd "$_fq_root" || return 1
source .venv/bin/activate
eval "$(.venv/bin/python - <<'PY'
import os, shlex
from fedquest.paths import data_root, runs_root
keys = [k for k in sorted(os.environ) if k.startswith(("FEDQUEST_", "NEO4J_", "HF_"))]
for k in keys:
    print(f"export {k}={shlex.quote(os.environ[k])}")
print(f"export FEDQUEST_RUNS={shlex.quote(str(runs_root()))} FEDQUEST_DATA={shlex.quote(str(data_root()))}")
PY
)"
if [ -f "$HOME/.fedquest/endpoints.env" ]; then
  set -a
  . "$HOME/.fedquest/endpoints.env"
  set +a
fi
export R=$FEDQUEST_RUNS
export FEDQUEST_MODELS=configs/models/final.yaml FEDQUEST_ANN=hnsw FEDQUEST_VIRTUAL_SHARDS=1 GIT_OPTIONAL_LOCKS=0
export FEDQUEST_LLM_AIMD=1 FEDQUEST_LLM_RATE=11 FEDQUEST_LLM_RATE_FILE=/tmp/fedquest-$(id -u)/llm.rate
export FEDQUEST_TIMELINE=1
unset FEDQUEST_LLM_LIVE
unset _fq_root _fq_src
