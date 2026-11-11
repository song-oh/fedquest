#!/bin/bash
set -u
POOL_FILE=${1:?pool file} LOGDIR=${2:?log dir}
mkdir -p "$LOGDIR" "$(dirname "$POOL_FILE")"
set +eu
type module > /dev/null 2>&1 || source /etc/profile > /dev/null 2>&1
module load frameworks || { echo "module load frameworks failed" >&2; exit 1; }
set -u
command -v vllm > /dev/null || { echo "no vllm after module load frameworks" >&2; exit 1; }
echo "== $(hostname): $(python -c 'import vllm, torch; print("vllm", vllm.__version__, "torch", torch.__version__, "xpu", torch.xpu.device_count())' 2>&1 | tail -n 1)"
STORE_HF=$(sed -n 's/^HF_HOME=//p' .fedquest.env 2> /dev/null | tail -n 1)
export HF_HOME=${POD_HF_HOME:-${STORE_HF:-$HOME/.cache/huggingface}} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
TIK=${POD_TIKTOKEN:-$HF_HOME/tiktoken}
if [[ -s $TIK/o200k_base.tiktoken && -s $TIK/cache/fb374d419588a4632f3f557e76b4b70aebbca790 ]]; then
  export TIKTOKEN_ENCODINGS_BASE=$TIK TIKTOKEN_RS_CACHE_DIR=$TIK/cache
else
  echo "== no tiktoken vocab under $TIK: the judge cannot start offline" >&2
fi
export TMPDIR=/tmp RAY_TMPDIR=/tmp no_proxy="localhost,127.0.0.1" NO_PROXY="localhost,127.0.0.1"
export ZE_FLAT_DEVICE_HIERARCHY=FLAT CCL_PROCESS_LAUNCHER=none VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_HOST_IP=127.0.0.1
VT=${POD_VLLM_THREADS:-8}
export OMP_NUM_THREADS=$VT MKL_NUM_THREADS=$VT OPENBLAS_NUM_THREADS=$VT NUMEXPR_MAX_THREADS=$VT RAYON_NUM_THREADS=$VT
export TOKENIZERS_PARALLELISM=false
export VLLM_CACHE_ROOT=/tmp/vllm-$USER XDG_CACHE_HOME=/tmp/xdg-$USER TRITON_CACHE_DIR=/tmp/triton-$USER
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor-$USER
SETSID=$(command -v setsid || true)
[[ -n $SETSID ]] && SETSID="$SETSID -w"
GEMMA=${POD_GEMMA:-5} JUDGE=${POD_JUDGE:-1} TP=${POD_TP:-2} JTP=${POD_JUDGE_TP:-2}
START_S=${POD_START_S:-1500}
JUDGE_WAIT_S=${POD_JUDGE_WAIT_S:-600}
GEMMA_MODEL=google/gemma-4-31B-it JUDGE_MODEL=openai/gpt-oss-120b
COMMON=(--host 127.0.0.1 --dtype bfloat16 --enforce-eager --gpu-memory-utilization "${POD_GPU_UTIL:-0.90}"
        --max-num-batched-tokens "${POD_BATCHED_TOKENS:-16384}")
read -r -a EXTRA <<< "${POD_VLLM_EXTRA:-}"
tile=0
declare -a NAMES PORTS PIDS
start() {
  local name=$1 port=$2 tiles=$3 mask
  shift 3
  mask=$(seq -s, "$tile" $((tile + tiles - 1)))
  mask=${mask%,}
  tile=$((tile + tiles))
  echo "== $(date '+%F %T') $name on tiles $mask, port $port"
  ZE_AFFINITY_MASK=$mask $SETSID nohup vllm serve "$@" --port "$port" "${COMMON[@]}" ${EXTRA[@]+"${EXTRA[@]}"} \
    > "$LOGDIR/$name.log" 2>&1 &
  NAMES+=("$name") PORTS+=("$port") PIDS+=("$!")
}
for ((r = 0; r < GEMMA; r++)); do
  start "gemma_$r" $((8100 + r)) "$TP" "$GEMMA_MODEL" --served-model-name "$GEMMA_MODEL" --tensor-parallel-size "$TP" \
    --max-model-len "${POD_MAX_LEN:-65536}" --max-num-seqs "${POD_MAX_SEQS:-64}" \
    --limit-mm-per-prompt '{"image": 16}'
done
for ((r = 0; r < JUDGE; r++)); do
  start "gptoss_$r" $((8150 + r)) "$JTP" "$JUDGE_MODEL" --served-model-name "$JUDGE_MODEL" --tensor-parallel-size "$JTP" \
    --max-model-len 16384 --max-num-seqs 128
done
answers() {
  curl -fs --noproxy '*' -m 120 "http://127.0.0.1:$1/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\": \"$2\", \"messages\": [{\"role\": \"user\", \"content\": \"Reply with the word OK.\"}], \"max_tokens\": 8, \"temperature\": 0}" \
    > /dev/null 2>&1
}
deadline=$((SECONDS + START_S))
declare -A UP DEAD
gemma_done=
while ((SECONDS < deadline)); do
  pending=0 gemma_pending=0
  for i in "${!NAMES[@]}"; do
    n=${NAMES[$i]} p=${PORTS[$i]}
    [[ -n ${UP[$n]:-} || -n ${DEAD[$n]:-} ]] && continue
    if ! kill -0 "${PIDS[$i]}" 2> /dev/null; then
      DEAD[$n]=1
      echo "== $(date '+%F %T') $n exited before it answered"
      continue
    fi
    m=$GEMMA_MODEL; [[ $n == gptoss_* ]] && m=$JUDGE_MODEL
    if curl -fs --noproxy '*' -m 5 "http://127.0.0.1:$p/v1/models" > /dev/null 2>&1 && answers "$p" "$m"; then
      UP[$n]=1
      echo "== $(date '+%F %T') $n answers on port $p"
    else
      pending=$((pending + 1))
      [[ $n == gemma_* ]] && gemma_pending=$((gemma_pending + 1))
    fi
  done
  ((pending == 0)) && break
  ((gemma_pending == 0)) && : "${gemma_done:=$SECONDS}"
  if [[ -n $gemma_done ]] && ((SECONDS - gemma_done >= JUDGE_WAIT_S)); then
    echo "== $(date '+%F %T') the judge did not answer within ${JUDGE_WAIT_S} s of the Gemma replicas: left out"
    break
  fi
  sleep 10
done
gemma=() judge=()
for i in "${!NAMES[@]}"; do
  n=${NAMES[$i]}
  if [[ -z ${UP[$n]:-} ]]; then echo "== $n did not come up: tail $LOGDIR/$n.log"; tail -n 5 "$LOGDIR/$n.log"; continue; fi
  url="\"http://127.0.0.1:${PORTS[$i]}/v1\""
  if [[ $n == gemma_* ]]; then gemma+=("$url"); else judge+=("$url"); fi
done
join() { local IFS=,; echo "$*"; }
echo "{\"gemma\": [$(join ${gemma[@]+"${gemma[@]}"})], \"gptoss\": [$(join ${judge[@]+"${judge[@]}"})]}" > "$POOL_FILE.tmp"
mv "$POOL_FILE.tmp" "$POOL_FILE"
echo "== $(date '+%F %T') $(hostname): ${#gemma[@]} Gemma and ${#judge[@]} judge replicas live ($POOL_FILE)"
((${#gemma[@]} > 0))
