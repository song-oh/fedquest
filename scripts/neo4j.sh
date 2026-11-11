#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
VERSION=5.26.29
BASE=${FEDQUEST_NEO4J_HOME:-$HOME/fedquest-neo4j}
SERVER=$BASE/neo4j-community-$VERSION
CONF=$SERVER/conf/neo4j.conf
export JAVA_HOME=$BASE/jdk-21 NEO4J_HOME=$SERVER NEO4J_CONF=$SERVER/conf
export PATH=$JAVA_HOME/bin:$PATH
HOST=$(hostname)

env_set() {
  (umask 077 && { grep -v "^$1=" .fedquest.env 2>/dev/null || true; echo "$1=$2"; } > .fedquest.env.tmp)
  mv .fedquest.env.tmp .fedquest.env && chmod 600 .fedquest.env
}
fetch() {
  [[ -d $2 ]] && return
  rm -rf "$2.part" && mkdir -p "$2.part"
  curl -fsSL "$1" | tar xz -C "$2.part" --strip-components=1
  mv "$2.part" "$2"
}
configure() {
  local block keys
  block="server.default_listen_address=127.0.0.1
server.default_advertised_address=127.0.0.1
server.bolt.listen_address=127.0.0.1:${FEDQUEST_NEO4J_PORT:-7688}
server.http.listen_address=127.0.0.1:${FEDQUEST_NEO4J_HTTP_PORT:-7475}
server.https.enabled=false
server.memory.heap.initial_size=${FEDQUEST_NEO4J_HEAP:-4g}
server.memory.heap.max_size=${FEDQUEST_NEO4J_HEAP:-4g}
server.memory.pagecache.size=${FEDQUEST_NEO4J_PAGECACHE:-2g}
server.directories.data=$BASE/data
server.directories.logs=$BASE/logs
server.directories.run=$BASE/run
server.directories.import=$BASE/import
dbms.usage_report.enabled=false"
  keys=$(sed 's/=.*//' <<< "$block" | paste -sd'|' -)
  { grep -vE "^($keys)=" "$CONF" || true; echo "$block"; } > "$CONF.tmp" && mv "$CONF.tmp" "$CONF"
  env_set NEO4J_URI "bolt://127.0.0.1:${FEDQUEST_NEO4J_PORT:-7688}"
  env_set NEO4J_USER neo4j
}
port() { { sed -n 's/^server.bolt.listen_address=127.0.0.1://p' "$CONF" 2> /dev/null || true; } | tail -n 1; }
answers() { timeout 3 bash -c "exec 3<>/dev/tcp/127.0.0.1/$1" 2>/dev/null; }
running() { "$SERVER/bin/neo4j" status > /dev/null 2>&1; }
where() {
  local at; at=$(cat "$BASE/run/host" 2>/dev/null || true)
  if [[ -n $at && $at != "$HOST" ]]; then echo "Neo4j for this clone was started on $at, not $HOST: run this there"; exit 1; fi
}

case ${1:-status} in
  install)
    mkdir -p "$BASE"
    ARCH=$(uname -m) && [[ $ARCH == x86_64 ]] && ARCH=x64
    fetch "https://api.adoptium.net/v3/binary/latest/21/ga/linux/$ARCH/jdk/hotspot/normal/eclipse" "$JAVA_HOME"
    fetch "https://dist.neo4j.org/neo4j-community-$VERSION-unix.tar.gz" "$SERVER"
    configure
    PASSWORD=$({ sed -n 's/^NEO4J_PASSWORD=//p' .fedquest.env 2> /dev/null || true; } | tail -n 1)
    if [[ -d $BASE/data/databases/system ]]; then
      [[ -n $PASSWORD ]] || { echo "$BASE/data holds a database whose password is not in .fedquest.env;" \
                                   "remove $BASE/data to start from an empty one"; exit 1; }
    else
      PASSWORD=$(python3 -c 'import secrets; print(secrets.token_hex(16))')
      "$SERVER/bin/neo4j-admin" dbms set-initial-password "$PASSWORD" > /dev/null
    fi
    env_set NEO4J_PASSWORD "$PASSWORD"
    echo "Neo4j $VERSION in $SERVER (data in $BASE/data); bolt://127.0.0.1:$(port) recorded in .fedquest.env"
    ;;
  start)
    [[ -x $SERVER/bin/neo4j && -x $JAVA_HOME/bin/java ]] || { echo "not installed: bash $0 install"; exit 1; }
    where
    configure
    P=$(port)
    if running && answers "$P"; then echo "Neo4j already running on $HOST: bolt://127.0.0.1:$P"; exit 0; fi
    if answers "$P"; then echo "port $P on $HOST is taken: FEDQUEST_NEO4J_PORT=<free port> bash $0 start"; exit 1; fi
    ulimit -n "$(ulimit -Hn)" 2> /dev/null || true
    mkdir -p "$BASE/run" && echo "$HOST" > "$BASE/run/host"
    running || "$SERVER/bin/neo4j" start > /dev/null
    for _ in $(seq 90); do
      answers "$P" && break
      running || { echo "Neo4j exited during start-up; end of $BASE/logs/neo4j.log:"; tail -n 20 "$BASE/logs/neo4j.log"
                   rm -f "$BASE/run/host"; exit 1; }
      sleep 2
    done
    answers "$P" || { echo "Neo4j did not answer within 180 s: see $BASE/logs/neo4j.log"; exit 1; }
    echo "Neo4j on $HOST: bolt://127.0.0.1:$P (user neo4j, password in .fedquest.env)"
    ;;
  reset)
    [[ -x $SERVER/bin/neo4j && -x $JAVA_HOME/bin/java ]] || { echo "not installed: bash $0 install"; exit 1; }
    where
    if running; then "$SERVER/bin/neo4j" stop > /dev/null; fi
    rm -f "$BASE/run/host"
    if [[ -d $BASE/data ]]; then
      mkdir -p "$BASE/archive" && mv "$BASE/data" "$BASE/archive/data_$(date +%m%d_%H%M%S)"
      echo "the earlier database is in $BASE/archive/ (remove it when no longer needed)"
    fi
    configure
    PASSWORD=$(python3 -c 'import secrets; print(secrets.token_hex(16))')
    "$SERVER/bin/neo4j-admin" dbms set-initial-password "$PASSWORD" > /dev/null
    env_set NEO4J_PASSWORD "$PASSWORD"
    echo "an empty database with a new password (in .fedquest.env): bash $0 start"
    ;;
  stop)
    where
    if running; then "$SERVER/bin/neo4j" stop > /dev/null; fi
    rm -f "$BASE/run/host"
    echo "stopped"
    ;;
  status)
    P=$(port)
    [[ -n $P ]] || { echo "not installed: bash $0 install"; exit 0; }
    at=$(cat "$BASE/run/host" 2>/dev/null || true)
    if [[ -z $at ]]; then echo "not running (bolt://127.0.0.1:$P when started)"
    elif [[ $at != "$HOST" ]]; then echo "running on $at, not this node: bolt://127.0.0.1:$P there"
    elif answers "$P"; then echo "running on $HOST, answering: bolt://127.0.0.1:$P"
    else echo "started on $HOST but not answering: bolt://127.0.0.1:$P (see $BASE/logs/neo4j.log)"
    fi
    ;;
  *) echo "usage: bash $0 install|reset|start|stop|status"; exit 1 ;;
esac
