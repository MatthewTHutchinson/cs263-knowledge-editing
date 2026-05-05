#!/usr/bin/env bash
set -euo pipefail

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate cs263-project

cd "$HOME/cs263-knowledge-editing"
mkdir -p logs

stamp="$(date +%Y%m%d_%H%M%S)"
log_path="logs/baseline_memit_checkpoint_${stamp}.log"
printf '%s\n' "$log_path" > logs/baseline_memit_latest.path

export EASYEDIT_STATS_CHECKPOINT_INTERVAL="${EASYEDIT_STATS_CHECKPOINT_INTERVAL:-10}"

python scripts/baseline_memit.py \
  --data_path data/counterfact/counterfact-edit.json \
  2>&1 | tee "$log_path"
