#!/usr/bin/env bash
# coordinator on the Aurora login node: start | resume | grid | status | stop
set -uo pipefail
cd "$(dirname "$0")/.."
ME=$(id -u)
running() { pgrep -u "$ME" -f "bash scripts/coord.sh run" > /dev/null; }
launch() {
  local s=""
  command -v setsid > /dev/null && s=setsid
  mkdir -p "$R/run_logs"
  $s nohup bash scripts/coord.sh run >> "$R/run_logs/coord.out" 2>&1 < /dev/null &
  $s nohup bash scripts/coord.sh grid >> "$R/run_logs/coord.out" 2>&1 < /dev/null &
  hostname > "$R/run_logs/coord.host"
  sleep 3
  running || fail "the supervisor did not start: $R/run_logs/coord.out"
}
LANES=${FEDQUEST_COORD_LANES:-gemma=8,mx=3,judge=4}
export FEDQUEST_NEO4J_HOME=${FEDQUEST_NEO4J_HOME:-$HOME/fedquest-neo4j}
export FEDQUEST_TASK_THREADS=${FEDQUEST_TASK_THREADS:-1}
fail() { echo "FAILED: $*" >&2; exit 1; }
env_set() {
  (umask 077 && { grep -v "^$1=" .fedquest.env 2>/dev/null || true; echo "$1=$2"; } > .fedquest.env.tmp)
  mv .fedquest.env.tmp .fedquest.env && chmod 600 .fedquest.env
}
load_env() {
  unset NEO4J_URI NEO4J_USER NEO4J_PASSWORD
  unset FEDQUEST_DATA FEDQUEST_RUNS FEDQUEST_SITE_STORE_DIR FEDQUEST_LLM_CACHE FEDQUEST_LLM_CACHE_RO
  source scripts/env.sh > /dev/null || fail "source scripts/env.sh"
  ulimit -n "$(ulimit -Hn)" 2> /dev/null || ulimit -n 4096 2> /dev/null || true
}

case ${1:-} in
  start)
    [[ $(hostname -f 2>/dev/null || hostname) == *aurora* ]] || fail "not an Aurora login node: $(hostname)"
    if running; then
      fail "a supervisor runs already on this node (bash scripts/coord.sh status; to start over: bash scripts/coord.sh stop)"
    fi
    echo "== commit $(git log --oneline -1)"
    [[ -z $(git status --porcelain --untracked-files=no) ]] || fail "tracked files differ from the commit (git status)"
    load_env
    echo "== endpoints"
    A=$(globus-compute-endpoint list 2> /dev/null | grep -- "fedquest-aurora-cn" | grep -oE '[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}' | head -1)
    [[ -n $A ]] || fail "no fedquest-aurora-cn in this node's endpoint list: bash scripts/facility.sh up first"
    M=${2:-${FEDQUEST_MIDWAY3_ENDPOINT:-}}
    [[ $M =~ ^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$ ]] \
      || fail "Midway3's endpoint id is needed the first time: bash scripts/coord.sh start <the id facility.sh up printed on Midway3>"
    env_set FEDQUEST_MIDWAY3_ENDPOINT "$M"
    bash scripts/remote.sh endpoints "$A" "$M" > /dev/null
    echo "Aurora $A, Midway3 $M"
    load_env
    echo "== runs root $R"
    echo "== inputs"
    [[ -s $R/inputs/gemma4_tokenizer.json ]] || fail "$R/inputs/gemma4_tokenizer.json is missing (Gemma 4's tokenizer.json)"
    env_set FEDQUEST_TOKENIZER "$R/inputs/gemma4_tokenizer.json"
    load_env
    [[ -z ${FEDQUEST_LLM_CACHE_RO:-} ]] || fail "FEDQUEST_LLM_CACHE_RO is set ($FEDQUEST_LLM_CACHE_RO): a run reads no earlier run's records"
    left=$(find "$R/cache" "$R/site_stores" "$R/dgrag" -type f \( -name '*.sqlite' -o -name '*.npy' \) 2> /dev/null | head -3)
    [[ -z $left ]] || fail "an earlier attempt's records are in $R ($(echo $left)): bash scripts/coord.sh stop, then move the runs roots of both facilities aside"
    echo "a fresh run: every model call goes live"
    python scripts/make_pools.py --check > /dev/null || fail "the evaluation pools (python scripts/make_pools.py)"
    python - <<'PY' || fail "a study's questions do not load"
import sys
from pathlib import Path

from fedquest import budget
from fedquest.config import load_experiment
from fedquest.eval.runner import load_questions

c = budget.counter()
if not c.exact:
    sys.exit(f"token counts would be estimated (no Gemma 4 tokenizer at {c.path})")
names = sorted(p.stem for p in Path("configs/experiments").glob("*.yaml"))
bad = 0
for n in names:
    exp = load_experiment(n)
    try:
        qs = load_questions(exp.federation, exp.split)
        if exp.limit and len(qs) < exp.limit:
            print(f"{n}: {exp.federation}/{exp.split} has {len(qs)} questions, the study takes {exp.limit}")
            bad += 1
    except Exception as e:
        print(f"{n}: {exp.federation}/{exp.split}: {e}")
        bad += 1
print(f"questions: {len(names) - bad} of {len(names)} studies load; token counts exact ({c.path})")
sys.exit(1 if bad else 0)
PY
    echo "== encoders the coordinator loads (fetched once into HF_HOME if missing)"
    HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python - <<'PY' || fail "an encoder could not be fetched"
from huggingface_hub import snapshot_download
for m in ("BAAI/bge-base-en-v1.5", "BAAI/bge-small-en-v1.5", "intfloat/e5-base-v2",
          "sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/clip-ViT-B-32",
          "cross-encoder/ms-marco-MiniLM-L-6-v2", "Qwen/Qwen3-Embedding-0.6B"):
    snapshot_download(m)
print("encoders ready")
PY
    echo "== logins"
    python -m fedquest.remote.gateway login || fail "Globus Compute login"
    python third_party/inference_auth_token.py get_access_token > /dev/null \
      || fail "ALCF login: python third_party/inference_auth_token.py authenticate --force, then this again"
    fedquest llm check --wait 20 || fail "a model the runs need does not answer (fedquest llm check)"
    echo "== Neo4j (the federated KG), from an empty graph, under $FEDQUEST_NEO4J_HOME"
    bash scripts/neo4j.sh install || fail "neo4j install"
    bash scripts/neo4j.sh reset || fail "neo4j reset"
    bash scripts/neo4j.sh start || fail "neo4j start"
    load_env
    fedquest kg stats > /dev/null || fail "the coordinator cannot reach Neo4j ($NEO4J_URI)"
    echo "== the earlier attempt's outputs archived (results, driver state, logs, traces)"
    A_DIR=$R/archive/attempt_$(date +%m%d_%H%M%S)
    mkdir -p "$A_DIR"
    for d in results pipeline run_logs traces remote_logs; do
      if [[ -n $(ls -A "$R/$d" 2> /dev/null) ]]; then mv "$R/$d" "$A_DIR/"; fi
    done
    find "$R/cache" -maxdepth 1 -name 'wire_main_*' -exec mv {} "$A_DIR/" \; 2> /dev/null
    rmdir "$A_DIR" 2> /dev/null || echo "moved to $A_DIR"
    mkdir -p "$R/results" "$R/run_logs"
    python -m fedquest.pipeline plan | tail -1
    echo "== the supervisor and the grid's watcher, detached"
    launch
    echo "== OK: the run goes on on $(hostname) by itself (you can log out; from another login node: ssh $(hostname))"
    echo "   progress:  bash scripts/coord.sh status      the supervisor's log:  tail -f $R/run_logs/coord.log"
    ;;

  run)
    load_env
    LOG=$R/run_logs/coord.log
    mkdir -p "$R/run_logs"
    log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
    stage() { CUR=$*; echo "$*" > "$R/run_logs/coord.stage"; log "== $*"; }
    waiting() { echo "WAITING: $* (then: $CUR)" > "$R/run_logs/coord.stage"; log "WAITING: $*"; }
    heal() {
      if grep -q '^NOT READY: aurora:' "$R/run_logs/pool.last"; then
        if queued=$(qselect -u "$USER" 2> /dev/null) && [[ -z $queued ]]; then
          log "Aurora's slots are down and no batch job of yours is there: its endpoint again (facility.sh restart)"
          bash scripts/facility.sh restart >> "$LOG" 2>&1 || log "facility.sh restart failed (above)"
        fi
      fi
      if grep -q '^NOT READY: midway3:' "$R/run_logs/pool.last"; then
        waiting "Midway3's slots are down: its queue wait, or its endpoint (if squeue -u \$USER on Midway3 shows no job of yours, run bash scripts/facility.sh up there; the run goes on by itself once its slots are up)"
      fi
    }
    pool() {
      local try tries=${FEDQUEST_POOL_TRIES:-36}
      for try in $(seq "$tries"); do
        { bash scripts/remote.sh start && bash scripts/remote.sh up "${FEDQUEST_POOL_TRY_S:-1800}"; } 2>&1 \
          | tee "$R/run_logs/pool.last" >> "$LOG"
        if [[ ${PIPESTATUS[0]} == 0 ]]; then echo "$CUR" > "$R/run_logs/coord.stage"; return 0; fi
        log "the main pool is not ready (try $try of $tries); again in 2 min"
        heal
        sleep 120
      done
      return 1
    }
    stage "main pool up (queue waits, then each node's Qdrant build)"
    pool || { stage "STOPPED: the main pool did not come up ($LOG)"; exit 1; }
    if [[ -f $R/run_logs/coord.smoked ]]; then
      log "the smokes passed before ($(cat "$R/run_logs/coord.smoked"))"
    else
      if ! python -m fedquest.remote.gateway fresh >> "$LOG" 2>&1; then
        stage "STOPPED: a site server found an earlier attempt's records ($LOG): bash scripts/coord.sh stop, move that facility's runs root aside, then bash scripts/coord.sh start"
        exit 1
      fi
      stage "smokes"
      for b in mmqa musique hotpot; do
        fedquest eval run "smoke_$b" --mode remote --no-resume > "$R/run_logs/smoke_$b.log" 2>&1 &
      done
      wait
      errs=$(cat "$R"/results/smoke_*/rows/*.jsonl 2> /dev/null | grep -c '"error": "')
      over=$(grep -ho '"budget_left": -[0-9]*' "$R"/results/smoke_*/rows/*.jsonl 2> /dev/null | awk -F': ' '$2 < -400' | wc -l | tr -d ' ')
      rows=$(cat "$R"/results/smoke_*/rows/*.jsonl 2> /dev/null | wc -l | tr -d ' ')
      log "smokes: $rows rows, $errs error rows, $over over budget"
      if [[ $rows -lt 50 || $errs -gt 0 || $over -gt 0 ]]; then
        grep -h '"error": "' "$R"/results/smoke_*/rows/*.jsonl 2> /dev/null | head -3 | cut -c1-400 | tee -a "$LOG"
        stage "STOPPED: the smokes failed (rows, errors or budgets: $LOG, $R/run_logs/smoke_*.log)"
        exit 1
      fi
      date > "$R/run_logs/coord.smoked"
    fi
    last=""
    for round in 1 2 3 4 5 6; do
      stage "pipeline round $round"
      pool || log "the main pool is not ready; the driver runs anyway (its tasks retry)"
      python -m fedquest.pipeline run --lanes "$LANES" >> "$R/run_logs/pipeline.log" 2>&1
      code=$?
      if (( code > 128 )); then
        log "the driver ended (exit $code: killed by signal $((code - 128)), $(kill -l $((code - 128)) 2> /dev/null))"
      else
        log "the driver ended (exit $code)"
      fi
      if [[ $code == 3 ]]; then
        stage "WAITING: the ALCF login was refused: in another shell run python third_party/inference_auth_token.py authenticate --force, then touch $R/run_logs/coord.relogin"
        until [[ -f $R/run_logs/coord.relogin ]]; do sleep 60; done
        rm -f "$R/run_logs/coord.relogin"
        continue
      fi
      open=$(python -m fedquest.pipeline status --open 2> /dev/null | grep -cE ' (failed|waiting|running) ')
      errors=$(cat "$R"/results/*/rows/*.jsonl 2> /dev/null | grep -c '"error": "')
      log "open tasks $open, error rows $errors"
      if [[ $code == 0 && $open == 0 && $errors == 0 ]]; then break; fi
      if [[ $round == 6 || "$open $errors" == "$last" ]]; then log "no progress since the last round: done retrying"; break; fi
      last="$open $errors"
      python -m fedquest.pipeline retry --errors >> "$LOG" 2>&1
    done
    stage "waiting for the grid's watcher (4 h at most)"
    for _ in $(seq 48); do
      grep -q 'grid done' "$R/run_logs/grid.log" 2> /dev/null && break
      pgrep -u "$(id -u)" -f "grid.sh watch" > /dev/null || break
      sleep 300
    done
    grep -q 'grid done' "$R/run_logs/grid.log" 2> /dev/null \
      || log "the grid's watcher ended before the grid was done ($R/run_logs/grid.log): the report goes on without it"
    stage "the report (scripts/post.sh) and the graph's statistics"
    bash scripts/post.sh >> "$LOG" 2>&1 || log "post.sh failed (see above)"
    fedquest kg stats > "$R/results/kg_stats.json" 2>> "$LOG" || true
    python -m fedquest.pipeline status > "$R/results/pipeline_status.txt" 2>&1
    date > "$R/run_logs/coord.done"
    stage "DONE: $R/results/report/report.md"
    ;;

  resume)
    [[ $(hostname -f 2>/dev/null || hostname) == *aurora* ]] || fail "not an Aurora login node: $(hostname)"
    load_env
    if running; then fail "a supervisor runs already on this node (bash scripts/coord.sh status)"; fi
    [[ -s $R/pipeline/plan.json ]] || fail "no plan in $R/pipeline: bash scripts/coord.sh start"
    bash scripts/neo4j.sh start || fail "neo4j start"
    launch
    echo "== OK: resumed on $(hostname) (bash scripts/coord.sh status)"
    ;;

  grid)
    load_env
    mkdir -p "$R/run_logs"
    bash scripts/grid.sh watch >> "$R/run_logs/grid.log" 2>&1
    ;;

  status)
    load_env
    echo "== $(hostname): stage: $(cat "$R/run_logs/coord.stage" 2> /dev/null || echo 'not started')"
    if running; then echo "supervisor: running"; else echo "supervisor: not running (ended, or on $(cat "$R/run_logs/coord.host" 2> /dev/null || echo 'another node'))"; fi
    pgrep -u "$ME" -f "scripts/grid.sh watch" > /dev/null && echo "grid's watcher: running" || echo "grid's watcher: not running"
    python -m fedquest.pipeline status --open 2> /dev/null | grep -v ' waiting ' | head -25
    echo "waiting tasks: $(python -m fedquest.pipeline status --open 2> /dev/null | grep -c ' waiting ')"
    echo "error rows: $(cat "$R"/results/*/rows/*.jsonl 2> /dev/null | grep -c '"error": "')"
    tail -2 "$R/run_logs/grid.log" 2> /dev/null
    echo "Aurora's batch jobs of yours: $(qselect -u "$USER" 2> /dev/null | xargs -r qstat 2> /dev/null | tail -n +3 | awk '{print $1, $(NF-1), $NF}' | tr '\n' ';')"
    bash scripts/neo4j.sh status
    tail -3 "$R/run_logs/coord.log" 2> /dev/null
    ;;

  stop)
    have=0
    if [[ -x .venv/bin/python ]] && .venv/bin/python -c 'import fedquest' 2> /dev/null; then
      load_env
      have=1
    else
      echo "no working venv here: stopping by process name only"
    fi
    pkill -u "$ME" -f "bash scripts/coord.sh run" 2> /dev/null || true
    if (( have )); then python -m fedquest.pipeline stop 2> /dev/null || true; fi
    pkill -u "$ME" -f "fedquest.pipeline run" 2> /dev/null || true
    pkill -u "$ME" -f "bash scripts/coord.sh grid" 2> /dev/null || true
    pkill -u "$ME" -f "scripts/grid.sh watch" 2> /dev/null || true
    pkill -u "$ME" -f "fedquest.scale.replay" 2> /dev/null || true
    pkill -u "$ME" -f "fedquest.remote.gateway up" 2> /dev/null || true
    pkill -u "$ME" -f "eval run smoke_" 2> /dev/null || true
    n=0
    for p in $(pgrep -u "$ME" 2> /dev/null); do
      if [[ $p != "$$" ]] && grep -qz '^FEDQUEST_PIPELINE_TASK=' "/proc/$p/environ" 2> /dev/null; then
        kill "$p" 2> /dev/null && n=$((n + 1))
      fi
    done
    (( n == 0 )) || { echo "stopped $n task processes no driver owned"; sleep 5; }
    if (( have )); then
      FEDQUEST_POOL=scale bash scripts/remote.sh down 2> /dev/null || true
      bash scripts/remote.sh down 2> /dev/null || true
      bash scripts/neo4j.sh stop || true
    else
      pkill -u "$ME" -f "org.neo4j" 2> /dev/null || true
    fi
    pkill -u "$ME" -f "fedquest.remote.gateway" 2> /dev/null || true
    echo "stopped (the endpoints run on: bash scripts/facility.sh down ends them)"
    ;;

  *)
    echo "usage: bash scripts/coord.sh start [MIDWAY3_ENDPOINT_ID] | run | resume | grid | status | stop" >&2
    exit 2
    ;;
esac
