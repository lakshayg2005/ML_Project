#!/usr/bin/env bash
# L2 + L3 ablation. Run from Merge-L2-L3/ext after `python prepare_data.py`.
#   bash run_ablation.sh train     # all training runs (finished runs are skipped)
#   bash run_ablation.sh freeze    # MELD: choose the LLM fusion weight and temperatures on dev only
#   bash run_ablation.sh test      # MELD: score the fused rows on test, once
set -e
cd "$(dirname "$0")"
OUT=../runs
mkdir -p "$OUT/logs"

args_for() {
  case "$1" in
    baseline)  echo "" ;;                                         # release objective, original Feature Gating
    cb0.999)   echo "--task_loss cb --cb_beta 0.999" ;;            # Limitation 2: class-balanced weights
    cacg_none) echo "--gate cacg --gate_loss none" ;;              # Limitation 3: CACG, dynamic tau, no gate loss
  esac
}

run() {  # dataset name seed
  if [ -f "$OUT/$1/${2}_seed${3}.json" ]; then echo "skip $1 $2 $3"; return; fi
  python train.py --dataset "$1" --seed "$3" --name "$2" $(args_for "$2") > "$OUT/logs/${1}_${2}_seed${3}.log" 2>&1
  tail -1 "$OUT/logs/${1}_${2}_seed${3}.log"
}

case "${1:-}" in
  train)
    for s in 0 1 2; do for v in baseline cb0.999 cacg_none; do run MELD "$v" "$s"; done; done
    for s in 2007 1 2; do for v in baseline cacg_none; do run IEMOCAP "$v" "$s"; done; done
    ;;
  freeze) python fuse_llm.py freeze --variants baseline,cb0.999,cacg_none --seeds 0,1,2 ;;
  test)   python fuse_llm.py test ;;
  *) echo "usage: bash run_ablation.sh train|freeze|test"; exit 1 ;;
esac
