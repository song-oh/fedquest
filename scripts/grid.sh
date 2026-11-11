#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export FEDQUEST_POOL=scale FEDQUEST_SOCK_DIR=/tmp/fedquest-$(id -u)/scale FEDQUEST_REMOTE_CONFIG=configs/remote_scale.yaml
PY=.venv/bin/python
R=${R:-$($PY -c "from fedquest.paths import runs_root; print(runs_root())")}
OUT=$R/results/scale_grid/grid.jsonl
case ${1:-} in
  up)
    bash scripts/remote.sh start && bash scripts/remote.sh up "${2:-10800}"
    ;;
  run)
    if [[ ! -s $R/traces/scale_musique.jsonl ]]; then
      echo "no probe trace yet ($R/traces/scale_musique.jsonl): scale_musique must have run first" >&2
      exit 1
    fi
    mkdir -p "$(dirname "$OUT")"
    exec $PY -m fedquest.scale.replay --plan configs/scale_plan.json --out "$OUT"
    ;;
  status)
    bash scripts/remote.sh status || true
    if [[ -f $OUT ]]; then
      echo "grid: $(wc -l < "$OUT" | tr -d ' ') of $($PY -c 'import json; print(len(json.load(open("configs/scale_plan.json"))))') runs written"
    fi
    ;;
  down)
    bash scripts/remote.sh down
    ;;
  watch)
    total=$($PY -c 'import json; print(len(json.load(open("configs/scale_plan.json"))))')
    written() { if [[ -f $OUT ]]; then wc -l < "$OUT" | tr -d ' '; else echo 0; fi; }
    echo "$(date): waiting for scale_musique (the driver's status)"
    until $PY -m fedquest.pipeline status 2>/dev/null | grep -Eq '^scale_musique +[a-z]+ +done'; do sleep 300; done
    mkdir -p "$(dirname "$OUT")"
    bash scripts/remote.sh start
    for try in $(seq 20); do
      if (( $(written) >= total )); then break; fi
      echo "$(date): try $try, $(written) of $total runs written"
      if ! bash scripts/remote.sh up 7200; then sleep 120; continue; fi
      $PY -m fedquest.scale.replay --plan configs/scale_plan.json --out "$OUT" || true
    done
    bash scripts/remote.sh down || true
    echo "$(date): grid done, $(written) of $total runs written"
    ;;
  *)
    echo "usage: bash scripts/grid.sh up [S] | run | status | down | watch" >&2
    exit 2
    ;;
esac
