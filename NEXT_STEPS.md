# Next Steps for VM/Codex

Use this file to pick up the project on the VM. Current midterm results are valid as preliminary logged-run results, but the RippleEdits adapter was fixed after the logged sweeps, so relation specificity needs a fresh run before it is reported quantitatively.

Current active run: equal-sample external sweeps were launched on 2026-05-16 in tmux session `external_sweeps`. Logs are written to `logs/external_equal_sweeps_20260516.log`. The n=25 MQuAKE block is complete; the queue is continuing through n=25 RippleEdits, then n=100 MQuAKE/RippleEdits across ROME, MEMIT, and IKE.

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

Status on 2026-05-16: these checks passed after pulling `main`; unit tests passed with 12 tests, benchmark inspection found 3,000 MQuAKE records and 885 RippleEdits POPULAR records, and CSV export completed.

## Priority Experiments

1. Monitor equal-sample external benchmark sweeps.
   - Reason: the midterm report currently compares IKE n=25 against ROME/MEMIT n=10, and relation specificity was excluded from prior RippleEdits sweeps.
   - Completed so far:
     - ROME MQuAKE n=25: edited_fact_acc=0.4925, multihop_acc=0.1200, delta_multihop_acc=+0.0133.
     - MEMIT MQuAKE n=25: edited_fact_acc=0.5821, multihop_acc=0.0800, delta_multihop_acc=-0.0267.
     - IKE MQuAKE n=25: edited_fact_acc=0.9104, multihop_acc=0.4533, delta_multihop_acc=+0.3466.
   - Active session:
     ```bash
     tmux capture-pane -pt external_sweeps -S -80
     tail -f logs/external_equal_sweeps_20260516.log
     nvidia-smi
     ```
   - Queued commands:
     ```bash
     python scripts/eval_mquake.py --method ROME --n_cases 25 --edit_mode one
     python scripts/eval_mquake.py --method MEMIT --n_cases 25 --edit_mode all
     python scripts/eval_mquake.py --method IKE --n_cases 25 --edit_mode all
     python scripts/eval_ripple_edits.py --method ROME --subset POPULAR --n_cases 25 --require_criteria Relation_Specificity,Logical_Generalization,Subject_Aliasing
     python scripts/eval_ripple_edits.py --method MEMIT --subset POPULAR --n_cases 25 --require_criteria Relation_Specificity,Logical_Generalization,Subject_Aliasing
     python scripts/eval_ripple_edits.py --method IKE --subset POPULAR --n_cases 25 --require_criteria Relation_Specificity,Logical_Generalization,Subject_Aliasing
     python scripts/eval_mquake.py --method ROME --n_cases 100 --edit_mode one
     python scripts/eval_mquake.py --method MEMIT --n_cases 100 --edit_mode all
     python scripts/eval_mquake.py --method IKE --n_cases 100 --edit_mode all
     python scripts/eval_ripple_edits.py --method ROME --subset POPULAR --n_cases 100 --require_criteria Relation_Specificity,Logical_Generalization,Subject_Aliasing
     python scripts/eval_ripple_edits.py --method MEMIT --subset POPULAR --n_cases 100 --require_criteria Relation_Specificity,Logical_Generalization,Subject_Aliasing
     python scripts/eval_ripple_edits.py --method IKE --subset POPULAR --n_cases 100 --require_criteria Relation_Specificity,Logical_Generalization,Subject_Aliasing
     ```

2. After the tmux sweep finishes, regenerate and inspect summaries.
   ```bash
   python scripts/show_results.py --all
   python scripts/show_results.py --csv_dir results/csv
   git status --short
   ```
   Commit new `results/runs.jsonl` rows and any new `results/benchmark_details/*.json` files that represent completed runs.

3. Rerun or scale if the n=100 results are stable.
   - Preferred scale-up targets: n=250 first, then n=500 if runtime is acceptable.
   - Keep sample sizes equal across ROME, MEMIT, and IKE before making strong method-ranking claims.

4. Rerun RippleEdits with relation specificity included if the current tmux run fails before RippleEdits.
   - Reason: local data uses `Relation_Specificity`, but earlier logged runs excluded it because the evaluator used the upstream typo.
   - Fallback commands:
     ```bash
     python3 scripts/eval_ripple_edits.py --method ROME --subset POPULAR --n_cases 10 --require_criteria Relation_Specificity,Logical_Generalization,Subject_Aliasing
     python3 scripts/eval_ripple_edits.py --method MEMIT --subset POPULAR --n_cases 10 --require_criteria Relation_Specificity,Logical_Generalization,Subject_Aliasing
     python3 scripts/eval_ripple_edits.py --method IKE --subset POPULAR --n_cases 25 --require_criteria Relation_Specificity,Logical_Generalization,Subject_Aliasing
     ```

## Evaluation Improvements

1. Rerun the expanded custom probe set.
   - Current source has 225 probes: 15 edit topics x 5 categories x 3 probes.
   - The existing logged probe results are from the earlier 100-probe set, so do not mix the old and new probe tables.
   - Before GPU runs, validate:
     ```bash
     python3 scripts/audit_probes.py --min_total 200 --strict
     ```
   - Then rerun all methods:
     ```bash
     python3 scripts/run_probes.py --method ROME
     python3 scripts/run_probes.py --method MEMIT
     python3 scripts/run_probes.py --method IKE --data_path data/counterfact/counterfact-edit.json
     ```

2. Fix the CounterFact rephrase/paraphrase issue.
   - Reason: current EasyEdit rephrase prompts are noisy and should not be compared directly to paper values.
   - Investigate whether the original CounterFact `paraphrase_prompts` can be loaded and used instead of the noisy EasyEdit-converted prompts.
   - After fixing, rerun a small ROME sanity check before rerunning all methods.

3. Reconfirm all reported results end-to-end before final report writing.
   - Check that each method uses the intended editing mode: ROME single-edit restored per case, MEMIT single-edit and true batch where labeled, and IKE in-context only.
   - Check that each table value can be traced to `results/runs.jsonl`, `results/probe_results.jsonl`, or regenerated CSVs.
   - Re-run unit tests, benchmark inspectors, and result summaries after any evaluator change.
   - Record any rerun commands and seeds in `STATUS.md` so the final report is reproducible.

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
