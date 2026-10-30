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
    baseline)          echo "--gate baseline --focal_gamma 0" ;;
    baseline_fixattn)  echo "--gate baseline --fix_empty_attention 1 --focal_gamma 0" ;;
    softmax_fixed)     echo "--gate softmax --gate_loss none --focal_gamma 0" ;;
    cacg_noloss)       echo "--gate cacg --gate_loss none --focal_gamma 0" ;;
    cacg_entropy_sop)  echo "--gate cacg --gate_loss entropy --lambda_gate 1.0 --focal_gamma 0" ;;
    cacg_utility)      echo "--gate cacg --gate_loss utility --aux_heads --lambda_gate 1.0 --lambda_aux 0.5 --focal_gamma 0" ;;
    cacg_utility_T025) echo "--gate cacg --gate_loss utility --aux_heads --lambda_gate 1.0 --lambda_aux 0.5 --utility_T 0.25 --focal_gamma 0" ;;
    # The variants above fix gamma=0 (L3 grid, run before the MELD release loss was spotted; IEMOCAP is unaffected).
    # merged model: L1 (IEMOCAP: WavLM audio, VAD aux loss, soft labels), L3; L2 LLM fusion is post hoc.
    # *_g1 = MELD with the release loss
    is10_l1)           echo "--lambda_vad 1.0 --soft_beta 0.5" ;;
    wavlm)             echo "--audio wavlm" ;;
    wavlm_l1)          echo "--audio wavlm --lambda_vad 1.0 --soft_beta 0.5" ;;
    wavlm_cacg)        echo "--audio wavlm --gate cacg --gate_loss none" ;;
    wavlm_l1_cacg)     echo "--audio wavlm --lambda_vad 1.0 --soft_beta 0.5 --gate cacg --gate_loss none" ;;
    baseline_g1)       echo "--gate baseline --focal_gamma 1" ;;
    cacg_utility_g1)   echo "--gate cacg --gate_loss utility --aux_heads --lambda_gate 1.0 --lambda_aux 0.5 --focal_gamma 1" ;;
  esac
}
for name in $VARIANTS; do
  for s in $SEEDS; do
    if [ -f "$OUT/$DS/${name}_seed${s}.json" ]; then echo "skip $name $s"; continue; fi
    python -I train.py --dataset "$DS" --seed "$s" --name "$name" $(args_for "$name") > "$OUT/logs/${DS}_${name}_seed${s}.log" 2>&1
    tail -1 "$OUT/logs/${DS}_${name}_seed${s}.log"
  done
done
echo GRID_DONE "$DS"
