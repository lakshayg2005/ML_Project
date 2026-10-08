#!/usr/bin/env bash
# Ablation grid for Limitation 3 (CACG).
# Usage: bash run_grid.sh IEMOCAP "2007 1 2" [variants...]   (default: all variants)
set -u
DS=${1:-IEMOCAP}
SEEDS=${2:-"2007 1 2"}
shift 2 || true
VARIANTS=${*:-"baseline softmax_fixed cacg_noloss cacg_entropy_sop cacg_utility cacg_utility_T025 baseline_fixattn"}
OUT=../runs
cd "$(dirname "$0")"
mkdir -p "$OUT/logs"
args_for() {
  case $1 in
    baseline)          echo "--gate baseline" ;;
    baseline_fixattn)  echo "--gate baseline --fix_empty_attention 1" ;;
    softmax_fixed)     echo "--gate softmax --gate_loss none" ;;
    cacg_noloss)       echo "--gate cacg --gate_loss none" ;;
    cacg_entropy_sop)  echo "--gate cacg --gate_loss entropy --lambda_gate 1.0" ;;
    cacg_utility)      echo "--gate cacg --gate_loss utility --aux_heads --lambda_gate 1.0 --lambda_aux 0.5" ;;
    cacg_utility_T025) echo "--gate cacg --gate_loss utility --aux_heads --lambda_gate 1.0 --lambda_aux 0.5 --utility_T 0.25" ;;
  esac
}
for name in $VARIANTS; do
  for s in $SEEDS; do
    if [ -f "$OUT/$DS/${name}_seed${s}.json" ]; then echo "skip $name $s"; continue; fi
    python train.py --dataset "$DS" --seed "$s" --name "$name" $(args_for "$name") > "$OUT/logs/${DS}_${name}_seed${s}.log" 2>&1
    tail -1 "$OUT/logs/${DS}_${name}_seed${s}.log"
  done
done
echo GRID_DONE "$DS"
