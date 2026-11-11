#!/usr/bin/env bash
# this facility's Globus Compute endpoint: up | down | restart
set -uo pipefail
cd "$(dirname "$0")/.."
case "$(hostname -f 2>/dev/null || hostname)" in
  *aurora*) FAC=aurora ;;
  *midway3*|*rcc*) FAC=midway3 ;;
  *) echo "not an Aurora or Midway3 login node: $(hostname)" >&2; exit 2 ;;
esac
EP=fedquest-$FAC-cn
if [[ $FAC == aurora ]]; then source scripts/env.sh; else source .venv/bin/activate; fi
fail() { echo "FAILED: $*" >&2; exit 1; }
ours() {
  local p
  for p in $(pgrep -u "$(id -u)" -f "Globus Compute Endpoint|globus-compute-endpoint|interchange\.py"); do
    grep -qz '^VIRTUAL_ENV=.*fedquest' "/proc/$p/environ" 2> /dev/null && echo "$p"
  done
}
stop_ep() {
  local left
  globus-compute-endpoint stop "$EP" > /dev/null 2>&1
  sleep 3
  left=$(ours)
  if [[ -n $left ]]; then
    kill -9 $left 2> /dev/null
    echo "killed endpoint processes a stop left ($(echo $left)); their PID files go stale in 90 s (a start refuses until then)"
    sleep 100
  fi
  local pid=~/.globus_compute/$EP/daemon.pid
  if [[ -f $pid ]] && (( $(date +%s) - $(stat -c %Y "$pid") < 60 )); then
    rm -f "$pid"
    echo "$EP runs on another login node (its PID file is fresh): its PID file removed, so it stops itself"
    sleep 45
  fi
}
running() {
  local line i
  for i in 1 2 3 4 5 6; do
    line=$(globus-compute-endpoint list 2> /dev/null | grep -- "$EP")
    [[ $line == *Running* ]] && { echo "$line"; return 0; }
    sleep 10
  done
  echo "${line:-$EP is not listed}"
  return 1
}
jobs_end() {
  if [[ $FAC == aurora ]]; then
    qselect -u "$USER" | xargs -r qdel; sleep 10; qstat -u "$USER"
  else
    scancel -u "$USER"; sleep 10; squeue -u "$USER"
  fi
}

case ${1:-} in
  up)
    echo "== $FAC, commit $(git log --oneline -1)"
    stop_ep
    pkill -9 -f "fedquest.remote.gateway serve"; pkill -f fedquest.remote.server
    jobs_end
    echo "== data check"
    bash scripts/prepare_data.sh "$FAC" && python -m fedquest.remote.gateway check --facility "$FAC" || fail "data check"
    if [[ $FAC == aurora ]]; then
      python -m fedquest.site.hippo status musique_next_home twowiki_home hotpot_home || true
    fi
    python third_party/inference_auth_token.py get_access_token > /dev/null \
      || fail "ALCF login: run python third_party/inference_auth_token.py authenticate --force, then this again"
    echo "ALCF login OK"
    if [[ $FAC == midway3 ]] && ! pgrep -f scripts/midway3/fqproxy.py > /dev/null; then
      mkdir -p ~/.fedquest
      nohup .venv/bin/python scripts/midway3/fqproxy.py --host "$(hostname)" --port $((40000 + $(id -u) % 1000)) \
        --log ~/.fedquest/fqproxy.log > /dev/null 2>&1 &
      sleep 2
      pgrep -af scripts/midway3/fqproxy.py || fail "the proxy did not start (~/.fedquest/fqproxy.log)"
    fi
    echo "== endpoint"
    if [[ ! -d ~/.globus_compute/$EP ]]; then
      globus-compute-endpoint configure --manager-config "deploy/globus/${FAC}_cn/config.yaml" \
        --template-config "deploy/globus/${FAC}_cn/user_config_template.yaml.j2" \
        --schema-config "deploy/globus/${FAC}_cn/user_config_schema.json" "$EP" || fail "configure $EP"
    fi
    cp "deploy/globus/${FAC}_cn/user_config_template.yaml.j2" "deploy/globus/${FAC}_cn/user_config_schema.json" \
      ~/.globus_compute/"$EP"/ || fail "copy the endpoint template"
    globus-compute-endpoint start --detach "$EP" || fail "start $EP"
    sleep 5
    running || fail "$EP is not Running (its line above; Initialized: its Globus login did not finish): bash scripts/facility.sh restart"
    echo "== OK: $FAC ready. Login node: $(hostname). Endpoint id: the first column of the line above."
    ;;
  down)
    stop_ep
    if [[ $FAC == midway3 ]]; then pkill -f scripts/midway3/fqproxy.py; squeue -u "$USER"; else qstat -u "$USER"; fi
    ;;
  restart)
    echo "== $FAC: the endpoint again ($(date '+%F %T'))"
    stop_ep
    jobs_end
    [[ -d ~/.globus_compute/$EP ]] || fail "$EP is not configured here: bash scripts/facility.sh up"
    globus-compute-endpoint start --detach "$EP" || fail "start $EP"
    sleep 5
    running || fail "$EP is not Running (its line above; Initialized: its Globus login did not finish)"
    ;;
  *)
    echo "usage: bash scripts/facility.sh up|down|restart" >&2
    exit 2
    ;;
esac
