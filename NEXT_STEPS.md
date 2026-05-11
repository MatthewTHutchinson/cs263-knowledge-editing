# Next Steps for VM/Codex

Use this file to pick up the project on the VM. Current midterm results are valid as preliminary logged-run results, but the RippleEdits adapter was fixed after the logged sweeps, so relation specificity needs a fresh run before it is reported quantitatively.

## Immediate Checks

1. Pull latest repo changes on the VM:
   ```bash
   git pull
   ```

2. Confirm the environment and data are present:
   ```bash
   python3 scripts/inspect_benchmarks.py --mquake data/mquake/MQuAKE-CF-3k-v2.json --ripple data/ripple_edits/POPULAR.json --sample 0
   python3 -m unittest discover -s tests
   ```

3. Confirm current logged results:
   ```bash
   python3 scripts/show_results.py --csv_dir results/csv
   ```

## Priority Experiments

1. Rerun RippleEdits with relation specificity included.
   - Reason: local data uses `Relation_Specificity`, but earlier logged runs excluded it because the evaluator used the upstream typo.
   - Start with small controlled sweeps before scaling:
     ```bash
     python3 scripts/eval_ripple_edits.py --method ROME --subset POPULAR --n_cases 10 --require_criteria Relation_Specificity,Logical_Generalization,Subject_Aliasing
     python3 scripts/eval_ripple_edits.py --method MEMIT --subset POPULAR --n_cases 10 --require_criteria Relation_Specificity,Logical_Generalization,Subject_Aliasing
     python3 scripts/eval_ripple_edits.py --method IKE --subset POPULAR --n_cases 25 --require_criteria Relation_Specificity,Logical_Generalization,Subject_Aliasing
     ```

2. Run controlled equal-sample external benchmark sweeps.
   - Reason: the midterm report currently compares IKE n=25 against ROME/MEMIT n=10, so the comparison is directional only.
   - Preferred next target: use the same `n_cases` and seed for all three methods on MQuAKE and RippleEdits.

3. Add MEMIT RippleEdits results.
   - Reason: Table 3 currently includes IKE and ROME for RippleEdits but not MEMIT.
   - Suggested first run:
     ```bash
     python3 scripts/eval_ripple_edits.py --method MEMIT --subset POPULAR --n_cases 10 --require_criteria Logical_Generalization,Subject_Aliasing
     ```

## Report Updates After Runs

1. Regenerate result summaries:
   ```bash
   python3 scripts/show_results.py --csv_dir results/csv
   ```

2. Update `overleaf_midterm/main.tex` if new results change any Table 3 values or conclusions.

3. If relation specificity is rerun successfully, replace the current caveat with the new metric values and state whether the criterion changes the RippleEdits conclusion.

4. Recompile on Overleaf after updating `overleaf_midterm/main.tex`.

## Final-Report Planning

1. Keep specificity and consistency separate in the final framing.
2. Do not compare EasyEdit rephrase accuracy directly to paper values; keep it as a within-repo relative metric.
3. Treat IKE as an inference-time in-context baseline, not a persistent stored-weight edit.
4. Prefer equal-sample comparisons before making strong claims about which method is best on external benchmarks.
