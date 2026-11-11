#!/usr/bin/env bash
# from a Midway3 compute node, reach Globus auth and the ALCF API (--proxy: via fqproxy)
set -uo pipefail
ACCOUNT=${ACCOUNT:-pi-chard}
PARTITION=${PARTITION:-caslake}
URL_FILE=${FQPROXY_URL_FILE:-$HOME/.fedquest/fqproxy.url}
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
if [[ ${1:-} == --proxy ]]; then
  [[ -s $URL_FILE ]] || { echo "no proxy URL in $URL_FILE: start scripts/midway3/fqproxy.py first" >&2; exit 2; }
  export https_proxy=$(cat "$URL_FILE")
fi
srun --account="$ACCOUNT" --partition="$PARTITION" --nodes=1 --ntasks=1 --time=00:05:00 --job-name=fq-netcheck bash -c 'for h in auth.globus.org inference-api.alcf.anl.gov; do echo "$h $(curl -s -o /dev/null -m 20 -w "%{http_code} %{time_total}s" https://$h/)"; done'
