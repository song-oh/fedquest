#!/usr/bin/env bash
# gateway to the site servers, for the main or scale pool (FEDQUEST_POOL)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
R=${R:-$($PY -c "from fedquest.paths import runs_root; print(runs_root())")}
mkdir -p "$R/run_logs"
POOL=${FEDQUEST_POOL:-main}
if [[ $POOL == scale ]]; then
  export FEDQUEST_SOCK_DIR=${FEDQUEST_SOCK_DIR:-/tmp/fedquest-$(id -u)/scale}
  export FEDQUEST_REMOTE_CONFIG=${FEDQUEST_REMOTE_CONFIG:-configs/remote_scale.yaml}
  LOG=$R/run_logs/gateway_scale.log
else
  LOG=$R/run_logs/gateway.log
fi
SOCK_DIR=${FEDQUEST_SOCK_DIR:-/tmp/fedquest-$(id -u)}
gateway_pid() {
  local f
  f=$SOCK_DIR/gateway.pid
  if [[ -s $f ]] && kill -0 "$(cat "$f")" 2>/dev/null; then cat "$f"; fi
}
case ${1:-} in
  endpoints)
    A=${2:?usage: remote.sh endpoints AURORA_ID MIDWAY3_ID} M=${3:?usage: remote.sh endpoints AURORA_ID MIDWAY3_ID}
    mkdir -p "$HOME/.fedquest"
    printf 'FEDQUEST_AURORA_CN_ENDPOINT=%s\nFEDQUEST_MIDWAY3_CN_ENDPOINT=%s\n' "$A" "$M" > "$HOME/.fedquest/endpoints.env"
    chmod 600 "$HOME/.fedquest/endpoints.env"
    echo "written to ~/.fedquest/endpoints.env: source scripts/env.sh again"
    ;;
  start)
    if [[ -n $(gateway_pid) ]]; then echo "the gateway runs (pid $(gateway_pid))"; exit 0; fi
    : "${FEDQUEST_AURORA_CN_ENDPOINT:?no endpoint ids: bash scripts/remote.sh endpoints AURORA_ID MIDWAY3_ID, then source scripts/env.sh}"
    mkdir -p "$SOCK_DIR" && chmod 700 "$SOCK_DIR"
    nohup $PY -m fedquest.remote.gateway serve >> "$LOG" 2>&1 < /dev/null &
    for _ in $(seq 60); do
      if [[ -S $SOCK_DIR/gateway.sock && -n $(gateway_pid) ]]; then echo "$POOL gateway up (pid $(gateway_pid))"; exit 0; fi
      sleep 1
    done
    echo "the $POOL gateway did not come up; the end of its log:" >&2
    tail -20 "$LOG" >&2
    exit 1
    ;;
  up)
    exec $PY -m fedquest.remote.gateway up --wait "${2:-5400}"
    ;;
  status)
    exec $PY -m fedquest.remote.gateway status
    ;;
  down)
    $PY -m fedquest.remote.gateway down || true
    pid=$(gateway_pid)
    if [[ -n $pid ]]; then kill "$pid"; echo "gateway stopped"; fi
    ;;
  *)
    echo "usage: bash scripts/remote.sh endpoints AURORA_ID MIDWAY3_ID | start | up [S] | status | down" >&2
    exit 2
    ;;
esac
