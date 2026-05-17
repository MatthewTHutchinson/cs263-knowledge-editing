#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs
stamp="$(date -u +%Y%m%d_%H%M%S)"
log_path="logs/external_n100_${stamp}.log"

exec > >(tee -a "$log_path") 2>&1

echo "External n=100 sweep started at $(date -u --iso-8601=seconds)"
echo "Log path: $log_path"
echo

python scripts/eval_mquake.py --method ROME --n_cases 100 --edit_mode one
python scripts/eval_mquake.py --method MEMIT --n_cases 100 --edit_mode all
python scripts/eval_mquake.py --method IKE --n_cases 100 --edit_mode all
python scripts/eval_ripple_edits.py --method ROME --subset POPULAR --n_cases 100 --require_criteria Relation_Specificity,Logical_Generalization,Subject_Aliasing
python scripts/eval_ripple_edits.py --method MEMIT --subset POPULAR --n_cases 100 --require_criteria Relation_Specificity,Logical_Generalization,Subject_Aliasing
python scripts/eval_ripple_edits.py --method IKE --subset POPULAR --n_cases 100 --require_criteria Relation_Specificity,Logical_Generalization,Subject_Aliasing

echo
echo "External n=100 sweep finished at $(date -u --iso-8601=seconds)"
