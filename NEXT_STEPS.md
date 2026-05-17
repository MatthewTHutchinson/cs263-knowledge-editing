# Next Steps for VM/Codex

Use this file to pick up the project on the VM and move from preliminary midterm results toward final-report results. The midterm report has already been submitted; do not spend time updating `overleaf_midterm/` unless you explicitly need to archive a revised midterm artifact.

Current active run: expanded 225-probe queue in tmux session `probes_225`. It started on 2026-05-17 at `2026-05-17T15:18:16+00:00`, logs to `logs/probes_225_20260517.log`, and checkpoints rows to `results/probe_results_225.jsonl`. The same ROME/MEMIT/IKE commands are safe to rerun because `scripts/run_probes.py` skips existing method/probe rows in the output file by default.

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

1. Rerun the expanded custom probe set.
   - Current source has 225 probes: 15 edit topics x 5 categories x 3 probes.
   - The existing logged probe results are from the earlier 100-probe set, so do not mix the old and new probe tables.
   - Before GPU runs, validate:
     ```bash
     python3 scripts/audit_probes.py --min_total 225 --strict
     ```
   - Then rerun all methods. `scripts/run_probes.py` treats `--output_path` as a checkpoint file and skips already completed rows for the same method, so these commands are safe to rerun after interruption:
     ```bash
     python3 scripts/run_probes.py --method ROME --output_path results/probe_results_225.jsonl
     python3 scripts/run_probes.py --method MEMIT --output_path results/probe_results_225.jsonl
     python3 scripts/run_probes.py --method IKE --data_path data/counterfact/counterfact-edit.json --output_path results/probe_results_225.jsonl
     python3 scripts/show_results.py --probes --probes_path results/probe_results_225.jsonl
     ```

2. Rerun CounterFact with original ROME paraphrase prompts.
   - Status: conversion tooling exists in `scripts/prepare_counterfact_original.py`; the remaining work is to generate the converted dataset on the VM and rerun metrics after the active `probes_225` job finishes.
   - Reason: current EasyEdit rephrase prompts are noisy. Original ROME `paraphrase_prompts` should make paraphrase/generalization numbers more paper-comparable.
   - `baseline_rome.py`, `baseline_memit.py`, and `baseline_ike.py` now checkpoint each completed sampled record under `results/checkpoints/` and resume from matching rows by default. Use `--no_resume` only when intentionally rerunning from scratch.
   - Recommended sequence for the next tmux job:
     ```bash
     python3 scripts/prepare_counterfact_original.py --max_records 2500
     python3 scripts/baseline_rome.py --data_path data/counterfact/counterfact-original-easyedit.json --n_edits 100 --seed 42
     python3 scripts/baseline_rome.py --data_path data/counterfact/counterfact-original-easyedit.json --n_edits 300 --seed 42
     python3 scripts/baseline_memit.py --data_path data/counterfact/counterfact-original-easyedit.json --n_edits 300 --seed 42
     python3 scripts/baseline_ike.py --data_path data/counterfact/counterfact-original-easyedit.json --n_edits 300 --seed 42
     ```

3. Reconfirm all reported results end-to-end before final report writing.
   - Check that each method uses the intended editing mode: ROME single-edit restored per case, MEMIT single-edit and true batch where labeled, and IKE in-context only.
   - Check that each table value can be traced to `results/runs.jsonl`, `results/probe_results.jsonl`, or regenerated CSVs.
   - Re-run unit tests, benchmark inspectors, and result summaries after any evaluator change.
   - Record any rerun commands and seeds in `STATUS.md` so the final report is reproducible.

## Final Report Updates After Runs

1. Regenerate result summaries:
   ```bash
   python3 scripts/show_results.py --csv_dir results/csv
   ```

2. Create or update the final-report source with the new equal-sample and expanded-probe results.

3. Use the completed relation-specificity rows in the final tables and state that this criterion does not change the RippleEdits conclusion: IKE remains strongest, while ROME/MEMIT show weak overall ripple transfer.

4. Keep `overleaf_midterm/` as an archived submitted midterm package unless there is a specific reason to edit it.

## Final-Report Planning

1. Keep specificity and consistency separate in the final framing.
2. Do not compare EasyEdit rephrase accuracy directly to paper values; keep it as a within-repo relative metric.
3. Treat IKE as an inference-time in-context baseline, not a persistent stored-weight edit.
4. Prefer equal-sample comparisons before making strong claims about which method is best on external benchmarks.
