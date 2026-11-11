#!/usr/bin/env bash
# after the runs: cache simulations, the evidence analysis and the report
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
R=${R:-$($PY -c "from fedquest.paths import runs_root; print(runs_root())")}
mkdir -p "$R/results/cache_sim" "$R/run_logs"
sim() {
  if [[ -s $1 ]]; then
    $PY -m fedquest.cache.sim "$1" --out "$R/results/cache_sim/$2.jsonl" --objects 18150 || echo "cache sim $2 failed"
  else
    echo "skip cache sim $2: no rows at $1"
  fi
}
sim "$R/results/main_musique/rows/fedquest.jsonl" main_musique
sim "$R/results/session_musique/rows/fedquest_cold.jsonl" session_musique
$PY -m fedquest.eval.evidence || echo "evidence analysis failed"
$PY -m fedquest.eval.report || echo "report failed"
