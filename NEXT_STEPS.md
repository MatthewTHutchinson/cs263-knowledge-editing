# Next Steps for VM/Codex

Use this file to pick up the project on the VM and move from preliminary midterm results toward final-report results. The midterm report has already been submitted; do not spend time updating `overleaf_midterm/` unless you explicitly need to archive a revised midterm artifact.

Current active run: none. The checkpointed original-paraphrase CounterFact runs completed on 2026-05-17, including the IKE `k=4/8/16` ablation.

Important artifact note: the CounterFact baseline checkpoints under `results/checkpoints/` are intentionally gitignored. They exist in this workspace for ROME, MEMIT, and IKE, but they store EasyEdit metric objects rather than decoded model generations. This affects all CounterFact baseline methods, though it matters most for the planned IKE locality examples. Probe, MQuAKE, and RippleEdits detail files already include decoded generations.

CounterFact locality metric note: EasyEdit locality is a specificity check against the pre-edit model behavior. For the IKE audit table, compare the saved `metric_token_prediction` fields, especially `metric_matches_pre_context`, rather than treating free-generation containment of the dataset ground truth as the metric.

The expanded 225-probe queue completed on 2026-05-17. It produced `results/probe_results_225.jsonl` with 675 rows: 225 each for ROME, MEMIT, and IKE. The same ROME/MEMIT/IKE probe commands remain safe to rerun because `scripts/run_probes.py` skips existing method/probe rows in the output file by default.

The equal-sample external sweep queue completed on 2026-05-17 at `2026-05-17T11:28:01+00:00`. Logs are written to `logs/external_n100_20260517_055056.log`, completed run rows are in `results/runs.jsonl`, and completed per-case checkpoints are in `results/benchmark_partials/`.

Update after the 2026-05-17 spot VM interruption: the tmux server was gone, but all n=25 MQuAKE and RippleEdits runs completed and were written to `results/runs.jsonl`. The external benchmark scripts now write per-case partial JSONL files under `results/benchmark_partials/` and automatically resume matching commands unless `--no_resume` is passed.

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

Status on 2026-05-16: these checks passed after pulling `main`; unit tests passed, benchmark inspection found 3,000 MQuAKE records and 885 RippleEdits POPULAR records, and CSV export completed.

## Priority Experiments

1. Equal-sample external benchmark sweeps are complete.
   - Reason: current final-report claims should use equal sample sizes when comparing methods, and relation specificity was excluded from the earlier RippleEdits report-level rows.
   - Completed results:
     - ROME MQuAKE n=25: edited_fact_acc=0.4925, multihop_acc=0.1200, delta_multihop_acc=+0.0133.
     - MEMIT MQuAKE n=25: edited_fact_acc=0.5821, multihop_acc=0.0800, delta_multihop_acc=-0.0267.
     - IKE MQuAKE n=25: edited_fact_acc=0.9104, multihop_acc=0.4533, delta_multihop_acc=+0.3466.
     - ROME MQuAKE n=100: edited_fact_acc=0.4650, multihop_acc=0.0733, delta_multihop_acc=+0.0333.
     - MEMIT MQuAKE n=100: edited_fact_acc=0.5210, multihop_acc=0.0467, delta_multihop_acc=+0.0067.
     - IKE MQuAKE n=100: edited_fact_acc=0.8601, multihop_acc=0.4800, delta_multihop_acc=+0.4400.
     - ROME RippleEdits POPULAR n=100: overall_acc=0.1232, delta_overall_acc=+0.0514, Relation_Specificity_acc=0.0893, Logical_Generalization_acc=0.0336, Subject_Aliasing_acc=0.2998.
     - MEMIT RippleEdits POPULAR n=100: overall_acc=0.0749, delta_overall_acc=+0.0031, Relation_Specificity_acc=0.1137, Logical_Generalization_acc=0.0436, Subject_Aliasing_acc=0.0336.
     - IKE RippleEdits POPULAR n=100: overall_acc=0.3526, delta_overall_acc=+0.2808, Relation_Specificity_acc=0.2138, Logical_Generalization_acc=0.2315, Subject_Aliasing_acc=0.7962.
   <details>
   <summary>Completed external-sweep commands</summary>

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

   </details>
   - The partial files remain useful for future larger sweeps: matching commands automatically skip completed cases unless `--no_resume` is passed.

2. Regenerate and inspect summaries after any new result run.
   ```bash
   python scripts/show_results.py --all
   python scripts/show_results.py --csv_dir results/csv
   git status --short
   ```
   Commit new `results/runs.jsonl` rows and any new `results/benchmark_details/*.json` files that represent completed runs.

3. Optional scale-up now that n=100 is complete.
   - Preferred scale-up targets: n=250 first, then n=500 if runtime is acceptable.
   - Keep sample sizes equal across ROME, MEMIT, and IKE before making strong method-ranking claims.
   - The current n=100 pattern is stable relative to n=25 and consistent with the MQuAKE/RippleEdits qualitative story: direct edited-fact recall improves, but ROME/MEMIT do not reliably propagate edits through multi-hop/ripple queries. Treat exact numeric comparison to the papers as non-comparable because this repo uses GPT-2 XL and a local evaluator/prompting setup.

4. Relation specificity is now covered by the completed n=25/n=100 RippleEdits rows. Do not use the earlier targeted rows as the primary final-report comparison unless explicitly discussing historical pilot runs.

## Evaluation Improvements

1. Expanded custom probe set is complete.
   - Current source has 225 probes: 15 edit topics x 5 categories x 3 probes.
   - The completed 225-probe results are in `results/probe_results_225.jsonl`; do not mix them with the earlier 100-probe `results/legacy/probe_results_100_legacy.jsonl`.
   - Summary command:
     ```bash
     python3 scripts/show_results.py --probes --probes_path results/probe_results_225.jsonl
     ```

2. Rerun CounterFact with original ROME paraphrase prompts.
   - Status: complete for the current n=300 comparison and IKE `k` ablation.
   - Reason: current EasyEdit rephrase prompts are noisy. Original ROME `paraphrase_prompts` should make paraphrase/generalization numbers more paper-comparable.
   - `baseline_rome.py`, `baseline_memit.py`, and `baseline_ike.py` now checkpoint each completed sampled record under `results/checkpoints/` and resume from matching rows by default. Use `--no_resume` only when intentionally rerunning from scratch.
   - Completed commands/results:
     ```bash
     python3 scripts/prepare_counterfact_original.py --max_records 2500
     python3 scripts/baseline_rome.py --data_path data/counterfact/counterfact-original-easyedit.json --n_edits 100 --seed 42
     python3 scripts/baseline_rome.py --data_path data/counterfact/counterfact-original-easyedit.json --n_edits 300 --seed 42
     python3 scripts/baseline_memit.py --data_path data/counterfact/counterfact-original-easyedit.json --n_edits 300 --seed 42
     python3 scripts/baseline_ike.py --data_path data/counterfact/counterfact-original-easyedit.json --n_edits 300 --seed 42
     python3 scripts/baseline_ike.py --data_path data/counterfact/counterfact-original-easyedit.json --n_edits 300 --seed 42 --k 4
     python3 scripts/baseline_ike.py --data_path data/counterfact/counterfact-original-easyedit.json --n_edits 300 --seed 42 --k 8
     ```
   - Summary: ROME n=300 reached rewrite=0.9933, rephrase=0.7433, locality=0.8400; MEMIT n=300 reached rewrite=0.7800, rephrase=0.3867, locality=0.9833. IKE reached rewrite=1.0000 and locality=0.0667 for `k=4`, `k=8`, and `k=16`; reducing retrieved demonstrations did not recover locality.

3. Reconfirm all reported results end-to-end before final report writing.
   - Check that each method uses the intended editing mode: ROME single-edit restored per case, MEMIT single-edit and true batch where labeled, and IKE in-context only.
   - Check that each table value can be traced to `results/runs.jsonl`, `results/probe_results_225.jsonl`, or regenerated CSVs.
   - Re-run unit tests, benchmark inspectors, and result summaries after any evaluator change.
   - Record any rerun commands and seeds in `STATUS.md` so the final report is reproducible.

## Final Report Updates After Runs

1. Regenerate result summaries:
   ```bash
   python3 scripts/show_results.py --all
   python3 scripts/show_results.py --csv_dir results/csv
   ```
   Check whether regenerated CSVs are only local report-writing artifacts or should be committed.

2. Create or update the final-report source with the new original CounterFact table, equal-sample external benchmark results, and expanded-probe results.
   - Prefer the `CounterFact-original` n=300 rows over older EasyEdit CounterFact rows when discussing final CounterFact behavior.
   - Main CounterFact framing: ROME is the best balanced result; MEMIT preserves locality best but has weak rewrite/rephrase in this single-edit setup; IKE has near-perfect rewrite/rephrase with severe locality collapse across `k=4/8/16`.

3. Add one targeted IKE locality analysis table.
   - Pull 10-20 representative examples from `results/checkpoints/ike_counterfact-original-easyedit_n300_seed42.jsonl` plus the `_k4`/`_k8` checkpoints.
   - Completed command: `scripts/audit_ike_counterfact_locality.py --n_examples 20`.
   - Output: `results/ike_counterfact_locality_examples.json`.
   - Include columns for edit prompt, `target_new`, locality prompt, pre-context metric-token prediction, post-context metric-token prediction, free post-context generation, and `metric_matches_pre_context`.
   - Use this table to support the claim that IKE's retrieved context causes relation-level interference on unrelated neighborhood prompts.

4. Use the completed relation-specificity rows in the final tables and state that this criterion does not change the RippleEdits conclusion: IKE remains strongest, while ROME/MEMIT show weak overall ripple transfer.

5. Do not run larger experiments unless the final report specifically needs more scale. The current n=300 CounterFact, n=100 external sweeps, and 225-probe diagnostic set are sufficient for the core project claims.

6. Keep `overleaf_midterm/` as an archived submitted midterm package unless there is a specific reason to edit it.

## Final Cleanup Checklist

Before submission, run:

```bash
git pull
python3 -m unittest discover -s tests
python3 scripts/show_results.py --all
python3 scripts/show_results.py --csv_dir results/csv
git status --short
```

If any final report source or generated summary artifacts are intentionally changed, commit and push them with a clear message before submission.

## Final-Report Planning

1. Keep specificity and consistency separate in the final framing.
2. Do not compare EasyEdit rephrase accuracy directly to paper values; keep it as a within-repo relative metric.
3. Treat IKE as an inference-time in-context baseline, not a persistent stored-weight edit.
4. Prefer equal-sample comparisons before making strong claims about which method is best on external benchmarks.
