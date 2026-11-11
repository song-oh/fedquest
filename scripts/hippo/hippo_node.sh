#!/bin/bash
set -u
cd "${PBS_O_WORKDIR:?run from scripts/hippo/hippo_index.pbs}" || exit 1
RANK=${PALS_RANKID:-${PMI_RANK:-${PALS_RANK:-}}}
[[ -n $RANK ]] || { echo "no MPI rank in the environment (PALS_RANKID, PMI_RANK)"; exit 1; }
read -r -a PLAN <<< "${HIPPO_PLAN:?}"
r=$RANK FED= PART= OF=
for entry in "${PLAN[@]}"; do
  home=${entry%%:*} n=${entry##*:}
  if ((r < n)); then FED=$home PART=$r OF=$n; break; fi
  r=$((r - n))
done
[[ -n $FED ]] || { echo "rank $RANK: no share in HIPPO_PLAN ($HIPPO_PLAN)"; exit 0; }
R=$(.venv/bin/python -c "from fedquest.paths import runs_root; print(runs_root())")
JOB=${PBS_JOBID%%.*}
LOGDIR=$R/hippo_build/$JOB/$FED
mkdir -p "$LOGDIR" "$R/cache/hippo"
exec > "$LOGDIR/node$PART.log" 2>&1
echo "== $(date '+%F %T') $(hostname) rank $RANK: $FED share $PART of $OF"
cleanup() { pkill -u "$USER" -f "vllm serve" 2> /dev/null; sleep 5; pkill -9 -u "$USER" -f "vllm serve" 2> /dev/null; }
trap cleanup EXIT
SERVED=1
if ! POD_GEMMA=${POD_GEMMA:-6} POD_JUDGE=0 bash scripts/hippo/serve.sh "$LOGDIR/pool$PART.json" "$LOGDIR/vllm$PART"; then
  echo "== $(date '+%F %T') no Gemma replica answers on $(hostname): share $PART not extracted here (the build does it)"
  SERVED=
  touch "$LOGDIR/part$PART.noserve"
fi
export no_proxy="127.0.0.1,localhost${no_proxy:+,$no_proxy}" NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}"
export FEDQUEST_MODELS=configs/models/hippo_pool.yaml FEDQUEST_POOL_FILE=$LOGDIR/pool$PART.json
export FEDQUEST_LLM_CACHE=$R/cache/hippo/$FED.part$PART.sqlite FEDQUEST_HIPPO_CONCURRENCY=${HIPPO_CONCURRENCY:-160}
export FEDQUEST_QDRANT_MODE=array
unset FEDQUEST_LLM_CACHE_RO FEDQUEST_LLM_LIVE
if [[ -n $SERVED ]]; then
  for attempt in 1 2 3; do
    echo "== $(date '+%F %T') fedquest hippo openie $FED --part $PART/$OF (attempt $attempt)"
    .venv/bin/fedquest hippo openie "$FED" --part "$PART/$OF" && break
    echo "== $(date '+%F %T') attempt $attempt exited $? (1: passages failed, retried next; 3: a refused login)"
  done
fi
touch "$LOGDIR/part$PART.done"
BUILDER=
if [[ -n $SERVED ]] && ((PART == 0)); then BUILDER=1; fi
if [[ -n $SERVED ]] && ((PART == 1)); then
  until [[ -e $LOGDIR/part0.done ]] || ((SECONDS > ${HIPPO_WAIT_S:-5400})); do sleep 30; done
  [[ -e $LOGDIR/part0.noserve ]] && BUILDER=1
fi
if [[ -n $BUILDER ]]; then
  wait_s=${HIPPO_WAIT_S:-5400}
  deadline=$((SECONDS + wait_s))
  while ((SECONDS < deadline)); do
    done_n=$(find "$LOGDIR" -maxdepth 1 -name 'part*.done' | wc -l)
    ((done_n >= OF)) && break
    sleep 30
  done
  echo "== $(date '+%F %T') $(find "$LOGDIR" -maxdepth 1 -name 'part*.done' | wc -l) of $OF shares done: building $FED"
  for attempt in 1 2 3; do
    .venv/bin/fedquest hippo index "$FED" && break
    echo "== $(date '+%F %T') build attempt $attempt exited $?"
  done
  .venv/bin/python -m fedquest.site.hippo status "$FED"
fi
echo "== $(date '+%F %T') done"
exit 0
