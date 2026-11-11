#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=$PWD
VERSION=v1.19.1
BIN=$HOME/.local/qdrant-$VERSION/qdrant
if [[ -z ${FEDQUEST_DATA:-} && -f .fedquest.env ]]; then
  FEDQUEST_DATA=$(sed -n 's/^FEDQUEST_DATA=//p' .fedquest.env | tail -n 1)
fi
DATA=${FEDQUEST_DATA:-$REPO/data}/qdrant
PORT=${FEDQUEST_QDRANT_PORT:-$((16333 + $(id -u) % 1000))}
HOST=$(hostname)
mkdir -p "$DATA"
DATA=$(cd "$DATA" && pwd)
PIDFILE=$DATA/qdrant.pid

answers() { curl -fs --noproxy '*' -m 5 "http://127.0.0.1:$1/" > /dev/null; }
is_qdrant() { [[ $(ps -p "$1" -o comm= 2>/dev/null) == *qdrant* ]]; }
record() {
  grep -v '^FEDQUEST_QDRANT_URL=' .fedquest.env 2>/dev/null > .fedquest.env.tmp || true
  echo "FEDQUEST_QDRANT_URL=http://127.0.0.1:$1" >> .fedquest.env.tmp && mv .fedquest.env.tmp .fedquest.env
}

QHOST= QPID= QPORT=
if [[ -f $PIDFILE ]]; then read -r QHOST QPID QPORT < "$PIDFILE" || true; fi
[[ $QPID =~ ^[0-9]+$ && $QPORT =~ ^[0-9]+$ ]] || QHOST= QPID= QPORT=

case ${1:-status} in
  install)
    if [[ ! -x $BIN ]]; then
      case "$(uname -s)-$(uname -m)" in
        Linux-x86_64) ASSET=qdrant-x86_64-unknown-linux-musl.tar.gz ;;
        Darwin-arm64) ASSET=qdrant-aarch64-apple-darwin.tar.gz ;;
        *) echo "no Qdrant binary for $(uname -sm)"; exit 1 ;;
      esac
      mkdir -p "$(dirname "$BIN")"
      curl -fsSL "https://github.com/qdrant/qdrant/releases/download/$VERSION/$ASSET" | tar xz -C "$(dirname "$BIN")"
    fi
    echo "$BIN"
    ;;
  start)
    if [[ -n $QHOST && $QHOST != "$HOST" ]]; then
      echo "Qdrant for this clone runs on $QHOST (pid $QPID, port $QPORT), not on $HOST."
      echo "Run this command there ('ssh $QHOST'), or stop it there first."
      exit 1
    fi
    if [[ -n $QPID ]] && is_qdrant "$QPID"; then
      record "$QPORT"
      echo "Qdrant already running on $HOST (pid $QPID): http://127.0.0.1:$QPORT (recorded in .fedquest.env)"
      exit 0
    fi
    rm -f "$PIDFILE"
    if answers "$PORT"; then
      echo "Port $PORT on $HOST is held by another server (another user's or an earlier attempt's Qdrant)."
      echo "Pick another port: FEDQUEST_QDRANT_PORT=<free port> bash scripts/qdrant.sh start"
      exit 1
    fi
    if [[ ! -x $BIN ]]; then
      case "$(uname -s)-$(uname -m)" in
        Linux-x86_64) ASSET=qdrant-x86_64-unknown-linux-musl.tar.gz ;;
        Darwin-arm64) ASSET=qdrant-aarch64-apple-darwin.tar.gz ;;
        *) echo "no Qdrant binary for $(uname -sm)"; exit 1 ;;
      esac
      mkdir -p "$(dirname "$BIN")"
      curl -fsSL "https://github.com/qdrant/qdrant/releases/download/$VERSION/$ASSET" | tar xz -C "$(dirname "$BIN")"
    fi
    (cd "$DATA" &&
      QDRANT__SERVICE__HOST=127.0.0.1 QDRANT__SERVICE__HTTP_PORT=$PORT QDRANT__SERVICE__GRPC_PORT=$((PORT + 1)) \
      QDRANT__STORAGE__STORAGE_PATH=$DATA/storage QDRANT__STORAGE__SNAPSHOTS_PATH=$DATA/snapshots \
      QDRANT__TELEMETRY_DISABLED=true QDRANT_INIT_FILE_PATH=$DATA/.qdrant-initialized \
      exec nohup "$BIN") > "$DATA/qdrant.log" 2>&1 &
    QPID=$!
    echo "$HOST $QPID $PORT" > "$PIDFILE"
    for _ in $(seq 60); do
      sleep 1
      if ! kill -0 "$QPID" 2>/dev/null; then
        rm -f "$PIDFILE"
        echo "Qdrant exited during start-up (port taken? set FEDQUEST_QDRANT_PORT=<free port>); end of its log:"
        tail -n 5 "$DATA/qdrant.log"
        exit 1
      fi
      answers "$PORT" && break
    done
    answers "$PORT" || { echo "Qdrant (pid $QPID) did not answer within 60 s: see $DATA/qdrant.log"; exit 1; }
    record "$PORT"
    echo "Qdrant on $HOST: http://127.0.0.1:$PORT (recorded in .fedquest.env)"
    ;;
  stop)
    if [[ -z $QPID ]]; then
      echo "not running"
    elif [[ $QHOST != "$HOST" ]]; then
      echo "Qdrant for this clone runs on $QHOST (pid $QPID, port $QPORT), not on $HOST."
      echo "Run this command there ('ssh $QHOST')."
      exit 1
    else
      if is_qdrant "$QPID"; then
        kill "$QPID"
        for _ in $(seq 30); do is_qdrant "$QPID" || break; sleep 1; done
        if is_qdrant "$QPID"; then echo "Qdrant (pid $QPID) is still shutting down: run stop again"; exit 1; fi
      fi
      rm -f "$PIDFILE"
      echo "stopped"
    fi
    ;;
  status)
    if [[ -z $QPID ]]; then
      echo "not running"
    elif [[ $QHOST != "$HOST" ]]; then
      echo "Qdrant on $QHOST (not this node): pid $QPID, port $QPORT"
    elif answers "$QPORT"; then
      echo "Qdrant on $QHOST: pid $QPID, port $QPORT, answering"
    else
      echo "Qdrant on $QHOST: pid $QPID, port $QPORT, not answering"
    fi
    ;;
  *) echo "usage: bash scripts/qdrant.sh install|start|stop|status"; exit 1 ;;
esac
