#!/usr/bin/env bash
# probe and refit the rank and anchor-z calibrations; free: the live_free study
set -euo pipefail
WHAT=${1:?usage: calibrate.sh mmqa|musique|free}
cd "$(dirname "$0")/.."
R=$(.venv/bin/python -c "from fedquest.paths import runs_root; print(runs_root())")
C=$R/calibration
mkdir -p "$C" "$R/results"
probe() {
  FEDQUEST_CALIBRATE_SCORES=1 .venv/bin/fedquest calibrate "$1" --mode remote --probe semijoin --limits share \
    --graph typed --mention 2 --k 80 ${2:+--layouts "$2"}
}
held_layouts() {
  if [[ $1 == musique ]]; then echo "musique_s4_k05_r0,musique_s4_r0,musique_s8_r0,musique_s16_r0,musique_s32_r0"
  else echo "${1}_s4_k05_r0,${1}_s4_r0"; fi
}
free() {
  local fam=$1 held=() dom=() f
  shift
  for f in $(held_layouts "$fam" | tr , ' '); do held+=("$C/$f.obs.jsonl"); done
  for f in "$@"; do
    if [[ -s $C/$f.obs.jsonl ]]; then dom+=("$C/$f.obs.jsonl"); else echo "== no $C/$f.obs.jsonl: domain $f left out"; fi
  done
  .venv/bin/python -m fedquest.eval.live_free --rank "${fam}_v4" --zcal "${fam}_z" --fit "$C/${fam}_s2_r0.obs.jsonl" \
    --inputs "${held[@]}" ${dom[@]+--domain "${dom[@]}"} --out "$R/results/live_free_$fam.json"
}
case $WHAT in
  mmqa | musique)
    FAM=$WHAT
    probe "${FAM}_s2_r0" "$(held_layouts "$FAM")"
    if [[ $FAM == musique ]]; then
      probe twowiki_s2_r0 "" || echo "== twowiki_s2_r0's probes failed: the free study goes without that domain"
      probe hotpot_s2_r0 "" || echo "== hotpot_s2_r0's probes failed: the free study goes without that domain"
    fi
    cp "configs/calibration/$FAM.costs.json" "$C/${FAM}_s2_r0.costs.json"
    .venv/bin/fedquest calibrate --refit --portable --label v4 --inputs "$C/${FAM}_s2_r0.obs.jsonl" \
      --out "configs/calibration/${FAM}_v4.json"
    .venv/bin/python -m fedquest.planner.zcal fit --inputs "$C/${FAM}_s2_r0.obs.jsonl" \
      --out "configs/calibration/${FAM}_z.json"
    ls -l "configs/calibration/${FAM}_v4.json" "configs/calibration/${FAM}_z.json"
    ;;
  free)
    free musique twowiki_s2_r0 hotpot_s2_r0 mmqa_s2_r0
    free mmqa musique_s2_r0
    ls -l "$R/results/live_free_musique.json" "$R/results/live_free_mmqa.json"
    ;;
  *)
    echo "usage: calibrate.sh mmqa|musique|free" >&2
    exit 2
    ;;
esac
