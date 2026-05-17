# Project Notes & Log

Daily working log. Append newest entries at the top. Terse is fine.

Format for each entry:

```
## YYYY-MM-DD — short title

- what I did
- what I learned / observed
- what broke or is blocking
- what's next
```

---

## 2026-05-17 — Original CounterFact n=300 and IKE k ablation completed

- Completed the original-paraphrase CounterFact n=300 runs on GPT-2 XL for ROME, MEMIT, and IKE using `data/counterfact/counterfact-original-easyedit.json`.
- Result rows are appended to `results/runs.jsonl`: ROME n=300 rewrite=0.9933, rephrase=0.7433, locality=0.8400; MEMIT n=300 rewrite=0.7800, rephrase=0.3867, locality=0.9833.
- IKE n=300 was run at `k=16`, then ablated at `k=4` and `k=8` using the same seed/sample. Results: `k=4` rewrite=1.0000, rephrase=0.9800, locality=0.0667; `k=8` rewrite=1.0000, rephrase=0.9967, locality=0.0667; `k=16` rewrite=1.0000, rephrase=0.9967, locality=0.0667.
- Diagnosis: IKE locality failure is not fixed by reducing retrieved demonstrations from 16 to 8 or 4. The checkpoint/log examples show relation-level context interference on unrelated neighborhood prompts.
- Updated `scripts/baseline_ike.py` with `--k` so future IKE retrieval-context ablations can be run without editing the YAML config; default checkpoint paths include `_k{K}` when the override is used.
- Verification: all three IKE checkpoints have 300 rows, no tmux jobs remain, and `python -m unittest discover -s tests` passes with 15 tests.
- Next: commit/push the result rows, generated original CounterFact data, `baseline_ike.py`, and refreshed markdown docs.

---

## 2026-05-17 — Expanded 225-probe results completed

- The `probes_225` tmux queue finished successfully; `results/probe_results_225.jsonl` has 675 rows: 225 each for ROME, MEMIT, and IKE.
- Overall post-edit pass rates: MEMIT 42.2% (+10.2), ROME 40.0% (+8.0), IKE 37.8% (+5.8), with deltas relative to pre-edit pass rate.
- Category highlights: ROME is strongest on logical negation at 68.9% (+64.4); MEMIT is strongest on compositional at 84.4% (+2.2); IKE is strongest on contradiction at 68.9% (+13.3).
- Symmetric inverse remains the clearest failure mode: ROME 0.0%, MEMIT 0.0%, IKE 8.9%.
- Launched the next checkpointed original-paraphrase CounterFact job in tmux session `counterfact_original`; logs go to `logs/counterfact_original_20260517.log`.
- Next: monitor `counterfact_original`, then commit `results/runs.jsonl`, generated original CounterFact data if changed, and updated markdown summaries.

---

## 2026-05-17 — Expanded probe run made resumable and launched

- Updated `scripts/run_probes.py` so `--output_path` acts as a checkpoint file: existing method/probe rows are skipped on restart, and each completed probe row is appended immediately.
- Added `scripts/show_results.py --probes_path` so the expanded 225-probe output can be summarized separately from the old 100-probe `results/probe_results.jsonl`.
- Added per-record checkpoint/resume support to `baseline_rome.py`, `baseline_memit.py`, and `baseline_ike.py` for the next original-paraphrase CounterFact rerun.
- Tightened probe validation docs from `--min_total 200` to `--min_total 225` for the current balanced probe contract.
- Launched the expanded ROME/MEMIT/IKE probe queue in tmux session `probes_225`; output path is `results/probe_results_225.jsonl`.
- Next: monitor the tmux log, summarize `results/probe_results_225.jsonl`, then run the checkpointed original-paraphrase CounterFact sequence in a new tmux job.

---

## 2026-05-17 — Equal-sample n=100 external sweeps completed

- The `external_n100` tmux queue finished at `2026-05-17T11:28:01+00:00`; no tmux session is live now.
- All six n=100 partial files under `results/benchmark_partials/` contain 100 rows each.
- MQuAKE n=100:
  - ROME one-edit: edited_fact_acc=0.4650, delta_edited_fact_acc=+0.2797, multihop_acc=0.0733, delta_multihop_acc=+0.0333.
  - MEMIT all-edit: edited_fact_acc=0.5210, delta_edited_fact_acc=+0.3357, multihop_acc=0.0467, delta_multihop_acc=+0.0067.
  - IKE all-edit/in-context: edited_fact_acc=0.8601, delta_edited_fact_acc=+0.6748, multihop_acc=0.4800, delta_multihop_acc=+0.4400.
- RippleEdits POPULAR n=100 with `Relation_Specificity,Logical_Generalization,Subject_Aliasing`:
  - ROME: overall_acc=0.1232, delta_overall_acc=+0.0514, Relation_Specificity_acc=0.0893, Logical_Generalization_acc=0.0336, Subject_Aliasing_acc=0.2998.
  - MEMIT: overall_acc=0.0749, delta_overall_acc=+0.0031, Relation_Specificity_acc=0.1137, Logical_Generalization_acc=0.0436, Subject_Aliasing_acc=0.0336.
  - IKE: overall_acc=0.3526, delta_overall_acc=+0.2808, Relation_Specificity_acc=0.2138, Logical_Generalization_acc=0.2315, Subject_Aliasing_acc=0.7962.
- Interpretation: equal-sample external results are stable. ROME/MEMIT improve edited-fact recall but show weak multi-hop/ripple propagation; IKE is strongest on MQuAKE multihop and RippleEdits alias/compositional behavior because the new facts are supplied in context.
- Next: rerun the expanded 225-probe custom diagnostic set on the GPU, then refresh final-report tables.

---

## 2026-05-17 — n=25 MQuAKE equal-sample results completed

- The `external_sweeps` tmux job completed the equal-sample n=25 MQuAKE block for all three methods.
- Results:
  - ROME one-edit: edited_fact_acc=0.4925, delta_edited_fact_acc=+0.3283, multihop_acc=0.1200, delta_multihop_acc=+0.0133.
  - MEMIT all-edit: edited_fact_acc=0.5821, delta_edited_fact_acc=+0.4179, multihop_acc=0.0800, delta_multihop_acc=-0.0267.
  - IKE all-edit: edited_fact_acc=0.9104, delta_edited_fact_acc=+0.7462, multihop_acc=0.4533, delta_multihop_acc=+0.3466.
- Interpretation is consistent with the earlier small sweeps: ROME and MEMIT improve direct edited-fact accuracy, but multihop transfer remains weak; IKE is strongest in this in-context MQuAKE setting.
- New local result artifacts exist but are not committed yet because the broader tmux sweep is still running:
  - `results/benchmark_details/mquake_rome_one_20260517_000316.json`
  - `results/benchmark_details/mquake_memit_all_20260517_001439.json`
  - `results/benchmark_details/mquake_ike_all_20260517_001858.json`
- Current run state: tmux has moved on to n=25 RippleEdits ROME with `Relation_Specificity,Logical_Generalization,Subject_Aliasing`.
- Next: wait for the RippleEdits n=25 block and n=100 block to finish, then commit the generated result rows/detail files and refresh summary tables.

---

## 2026-05-16 — Expanded custom probe set to 225 probes

- Fast-forwarded local `main` from `origin/main` before editing.
- Replaced the 100-probe hand-enumerated set with a generated 225-probe set in `src/probes/probe_set.py`.
- The new set covers 15 edit topics and is exactly class balanced: 45 probes each for `logical_negation`, `symmetric_inverse`, `compositional`, `contradiction`, and `chain_of_thought`.
- Each edit topic contributes 15 probes: 3 probes per category.
- Topic list: Danielle Darrieux language, Sanofi headquarters, Watts Humphrey alma mater, Theo Walcott sport, Lil Wayne label, Barack Obama citizenship, William Shakespeare birthplace, The Beatles origin city, Albert Einstein profession, Google headquarters, Tesla founder, Python creator, Machu Picchu country, Mozart instrument, and Microsoft product.
- Updated `scripts/audit_probes.py` supplied-fact warning threshold for the balanced design; compositional and chain-of-thought probes intentionally make supplied facts 40% of the set.
- Added `tests/test_probe_set.py` to lock the 15-topic/225-probe balance contract.
- Validation:
  - `python3 scripts/audit_probes.py --min_total 200 --strict` passed.
  - `python3 -m unittest discover -s tests` passed.
- Next: rerun ROME, MEMIT, and IKE probes on the GPU VM; existing `results/probe_results.jsonl` rows are from the earlier 100-probe set.

---

## 2026-05-16 — Equal-sample external sweeps launched

- Pulled latest `main` on the VM; branch is aligned with `origin/main` at `288fff3`.
- Ran immediate post-pull checks:
  - `python3 scripts/inspect_benchmarks.py --mquake data/mquake/MQuAKE-CF-3k-v2.json --ripple data/ripple_edits/POPULAR.json --sample 0`
  - `python3 -m unittest discover -s tests` passed with 12 tests.
  - `python3 scripts/show_results.py --csv_dir results/csv` completed and regenerated CSV exports.
- Confirmed RippleEdits POPULAR includes `Relation_Specificity` with 5,488 queries; the adapter now recognizes the corrected spelling.
- Launched equal-sample external sweeps in tmux session `external_sweeps`.
- Log path: `logs/external_equal_sweeps_20260516.log`.
- Queue:
  - n=25 MQuAKE: ROME one-edit, MEMIT all-edit, IKE all-edit.
  - n=25 RippleEdits POPULAR: ROME, MEMIT, IKE with `Relation_Specificity,Logical_Generalization,Subject_Aliasing`.
  - n=100 MQuAKE: ROME one-edit, MEMIT all-edit, IKE all-edit.
  - n=100 RippleEdits POPULAR: ROME, MEMIT, IKE with `Relation_Specificity,Logical_Generalization,Subject_Aliasing`.
  - final `scripts/show_results.py --csv_dir results/csv` refresh.
- Current status at launch checkpoint: first n=25 ROME MQuAKE run is active on the T4 GPU.
- Next: monitor `tmux capture-pane -pt external_sweeps -S -80` or `tail -f logs/external_equal_sweeps_20260516.log`; after completion, inspect new result rows, update tables, commit result JSON/detail files, and revise the midterm/final-report text if conclusions shift.

---

## 2026-05-11 — External benchmark sweeps committed

- Completed small external benchmark sweeps after the initial n=1 smoke tests:
  - MQuAKE: IKE all-edit n=25 reached edited_fact_acc=0.910 and multihop_acc=0.453; ROME one-edit n=10 reached edited_fact_acc=0.440 and multihop_acc=0.100; MEMIT all-edit n=10 reached edited_fact_acc=0.680 and multihop_acc=0.033.
  - RippleEdits POPULAR targeted logical-generalization/subject-aliasing: ROME n=10 reached overall_acc=0.160, Subject_Aliasing_acc=0.375, Logical_Generalization_acc=0.000; IKE n=25 reached overall_acc=0.347, Subject_Aliasing_acc=0.692, Logical_Generalization_acc=0.237.
- Exported CSVs again with `python scripts/show_results.py --csv_dir results/csv`. These files remain gitignored because they are generated from tracked JSONL/detail records.
- Committed and pushed the external benchmark sweep detail records at `3878d54 add external benchmark sweep results`, rebased on top of `2f1d6a6 Organize Overleaf midterm package`.
- Next: iterate the report locally around the final framing: direct edits succeed, but logical/ripple/multi-hop consequences remain incomplete; IKE's in-context facts help external ripple/multihop scores while ROME/MEMIT retain stronger controlled logical-negation probe gains.

---

## 2026-05-11 — Midterm report moved toward ACL format

- Reviewed the current repo state after commits `5682337`, `3f3617a`, `405db46`, and `f5dea62`.
- Confirmed IKE 50/100 baseline results, ROME/MEMIT probe results, IKE probe results, and the midterm report are now on `main`.
- Updated README/STATUS references that still described probe evaluation as ROME/MEMIT-only.
- Reworked the midterm report into the ACL template style using the local ACL `acl.sty` bundle.
- Expanded the report with the specificity-vs-consistency framing, IKE locality interpretation, rephrase caveat, probe-type discussion, and an appendix for category/type probe breakdowns plus planned RippleEdits/MQuAKE.
- Replaced the bulky extracted ACL template folder with a minimal `overleaf_midterm/` upload package.
- Removed duplicate root `midterm_report.tex`, reusable report template folders, and the generated Overleaf zip; `overleaf_midterm/main.tex` is now the source of truth for the midterm report.
- Report content now matches current result files:
  - baseline table uses the latest 100-edit ROME/MEMIT/IKE runs,
  - probe table reports ROME/MEMIT/IKE post-edit pass rates and deltas,
  - next steps focus on interpretation and optional RippleEdits/MQuAKE.

---

## 2026-05-11 — ROME/MEMIT probe sweeps completed on GCP T4

- Fixed `scripts/run_probes.py` for this EasyEdit checkout: `apply_rome_to_model` expects singular `request=[...]`, while MEMIT still expects `requests=[...]`.
- The earlier tmux log command failed because it used `date + %Y%m%d_%H%M%S`; GNU `date` requires no space: `date +%Y%m%d_%H%M%S`. Avoided the issue by using fixed log names.
- Ran ROME probes in tmux: `python scripts/run_probes.py --method ROME 2>&1 | tee logs/probes_rome.log`.
- Ran MEMIT probes in tmux with unbuffered output: `/home/matthewthutchinson1/miniconda3/envs/cs263-project/bin/python -u scripts/run_probes.py --method MEMIT 2>&1 | tee logs/probes_memit.log`.
- Ran IKE probes in tmux with unbuffered output and the full CounterFact retrieval pool: `/home/matthewthutchinson1/miniconda3/envs/cs263-project/bin/python -u scripts/run_probes.py --method IKE --data_path data/counterfact/counterfact-edit.json 2>&1 | tee logs/probes_ike.log`.
- `results/probe_results.jsonl` now has 300 rows: 100 ROME + 100 MEMIT + 100 IKE.
- Summary from `scripts/show_results.py --probes`:
  - IKE total: pre 36%, post 50%, delta +14%.
  - ROME total: pre 36%, post 64%, delta +28%.
  - MEMIT total: pre 36%, post 64%, delta +28%.
  - ROME/MEMIT improved logical negation from 0% to 88%; IKE reached 25%.
  - Symmetric inverse remains weak: IKE 13%, MEMIT 7%, ROME 0%.
- Exported CSVs with `scripts/show_results.py --csv_dir results/csv`.
- Next: write up probe findings and decide whether to run RippleEdits/MQuAKE or stop at the current comparison set.

---

## 2026-05-11 — Local validation done; probe run needs CUDA VM

- `scripts/audit_probes.py --min_total 100 --strict` passes with 100 probes.
- `python -m unittest discover -s tests` passes: 6 tests.
- `scripts/show_results.py --all` confirms IKE-50 and IKE-100 are now recorded locally: IKE-100 has rewrite=0.990, rephrase=0.990, locality=0.110.
- Attempted `python scripts/run_probes.py --method ROME`, but it stopped at the built-in CUDA assertion. `nvidia-smi` cannot communicate with an NVIDIA driver in this environment.
- Updated `STATUS.md` to mark IKE baselines complete and probe evaluation blocked locally. Next: run ROME/MEMIT probes on a CUDA VM, then summarize with `scripts/show_results.py --probes`.

---

## 2026-05-10 — Markdown docs cleaned up during IKE runs

- Replacement VM is active.
- IKE-50 has run on the VM.
- IKE-100 is currently running on the VM.
- Local `results/runs.jsonl` still only contains the earlier IKE-5 record; sync the VM results before updating final tables.
- Cleaned root Markdown docs down to:
  - `README.md`: setup, commands, metric definitions, stable summary
  - `STATUS.md`: current project state and next actions
  - `NOTES.md`: chronological working log
- Removed stale session/migration artifacts:
  - `HANDOFF.md`
  - `NEXT_CODEX_PROMPT.md`
  - `CLAUDE.md`

---

## 2026-05-10 — VM transition backup created

- Current VM: `cs263-t4` in `us-central1-a`, project `cs263-project-494118`.
- Created preservation archive at `/home/matthewthutchinson1/cs263-memit-preserve-20260510.tar.gz`.
- Archive SHA256: `f15b0cd7f85bf9b597572476f083f6151358dcbfe4474e99ca097f6471b3c73b`.
- Archive includes `data/stats/`, `results/`, `logs/`, `configs/`, `scripts/`, `patches/`, and project notes.
- Important: as of the follow-up LFS update, the stable MEMIT covariance `.npz` files are tracked with Git LFS. `logs/`, `results/IKE/embedding/`, `external/EasyEdit/`, model caches, and conda environments remain gitignored and should be restored from the archive only if needed.
- GitHub state checked: local `main` is aligned with `origin/main` at commit `20dd0cf`; only documentation edits were uncommitted at the time of this note.
- Recommendation for replacement VM: use standard/on-demand GPU provisioning rather than Spot/preemptible for long MEMIT/probe jobs. It costs more, but avoids preemptions that can interrupt multi-hour runs.
- Next: finish downloading or uploading the archive, create the replacement VM, restore the archive into a fresh clone, then run `python scripts/show_results.py --all` and verify the five MEMIT `.npz` files exist.

---

## 2026-05-05 — Local GPU work blocked

- Updated `STATUS.md`, `README.md`, and `NOTES.md` to reflect the current run history: MEMIT batch 50/100 are already in `results/runs.jsonl`, and IKE has a 5-edit run recorded.
- Tried to launch the ROME probe sweep in the local shell, but `scripts/run_probes.py` asserts CUDA availability and this environment does not have `torch.cuda.is_available() == true`.
- IKE has the same CUDA requirement, so the remaining probe/IKE runs need to be started on the GCP T4 environment rather than this local workspace.
- Next: run the probe sweeps and IKE 50/100 on the GPU box.

---

## 2026-05-05 — MEMIT cache complete; batch-10 smoke passed

- Pulled latest `main`; repo was already up to date.
- Verified local lightweight checks:
  - `python3 -m unittest discover -s tests` passed.
  - `python3 scripts/audit_probes.py --min_total 100 --strict` passed with 100 probes.
- Confirmed MEMIT covariance cache is fully warm for GPT-2 XL layers 13-17:
  - all five final `data/stats/gpt2-xl/wikipedia_stats/transformer.h.*.mlp.c_proj_float32_mom2_100000.npz` files exist.
  - no tmux MEMIT job was active before launch.
  - T4 was idle before launch.
- Ran true MEMIT batch smoke:

```bash
python scripts/batch_memit.py \
  --data_path data/counterfact/counterfact-edit.json \
  --batch_sizes 10 \
  --seed 42
```

- Log: `logs/batch_memit_20260505_082808.log`
- The log confirms this is a real batch/mass edit: `Writing 10 key/value pair(s)` for layers 13-17, with cached covariance files loaded.
- Result appended to `results/runs.jsonl`:
  - method: `MEMIT-batch`
  - dataset: `CounterFact-batch-10`
  - n=10, seed=42
  - rewrite_acc=0.900
  - rephrase_acc=0.100
  - locality_acc=1.000
- Next: finish the IKE 50/100 baseline runs, then run the ROME and MEMIT probe sweeps.

## 2026-05-05 — MEMIT attempt 5; layer 17 checkpointing added

- MEMIT cache/baseline had been interrupted repeatedly; this is now documented as the 5th attempt.
- Diagnosis from repo and host state:
  - Layers 13-16 covariance caches already exist under `data/stats/gpt2-xl/wikipedia_stats/`.
  - Layer 17 was the missing cache and repeated failure point.
  - Disk was not the issue: roughly 28 GB free; each covariance `.npz` is about 157 MB.
  - T4 VRAM was not the immediate issue: layer 17 ran at about 8.5 GB / 15 GB with high GPU utilization.
  - GCP `ZONE_RESOURCE_POOL_EXHAUSTED_WITH_DETAILS` is a zone capacity/provisioning problem for new GPU VMs, not evidence that the current T4 cannot run the job.
- Root cause of lost progress: upstream EasyEdit's stats collector only saves the covariance `.npz` after a full layer finishes. If the VM/process dies mid-layer, the layer restarts from zero.
- Added a local checkpoint patch in `external/EasyEdit/easyeditor/models/rome/layer_stats.py`:
  - writes `*.npz.partial.npz` during covariance computation
  - reloads the partial file on restart
  - skips already-processed batch groups
  - removes the partial file once the final `.npz` exists
- Added portable patch copy: `patches/0002-add-easyedit-layer-stats-partial-checkpoints.patch`.
- Added `scripts/run_memit_checkpointed.sh`:
  - activates `cs263-project`
  - sets `EASYEDIT_STATS_CHECKPOINT_INTERVAL=10` by default
  - launches `scripts/baseline_memit.py`
  - writes latest log path to `logs/baseline_memit_latest.path`
- Current layer 17 partial checkpoint observed:
  - `data/stats/gpt2-xl/wikipedia_stats/transformer.h.17.mlp.c_proj_float32_mom2_100000.npz.partial.npz`
- Resume command after crash/interruption:

```bash
tmux new-session -d -s memit scripts/run_memit_checkpointed.sh
```

- Check progress with:

```bash
cat logs/baseline_memit_latest.path
tail -n 80 "$(cat logs/baseline_memit_latest.path)"
find data/stats/gpt2-xl/wikipedia_stats -maxdepth 1 -type f -printf '%f %s bytes\n' | sort
nvidia-smi
```

- Expected resume log line:

```text
Resuming partial covariance stats from ...partial.npz after N batch groups.
```

Why this was not done earlier: the earlier assumption was that EasyEdit's layer-level cache was sufficient. It is sufficient only after a full layer finishes. It does not protect long mid-layer covariance computation, which became obvious after repeated interruptions on layer 17.

## 2026-05-03 — Expanded probe set and local tooling

- Added `scripts/audit_probes.py`, a no-model validator for the probe set. It checks unique IDs, valid edit keys/categories/types, expected answer fields, per-edit/category/type coverage, and target leakage in `implicit_edit` prompts.
- Expanded `src/probes/probe_set.py` from 34 to 100 probes around the five smoke-test edit cases.
- Current probe distribution:
  - By type: 52 `implicit_edit`, 31 `supplied_fact_reasoning`, 17 `target_conditioned`.
  - By category: 32 logical negation, 22 contradiction, 17 compositional, 15 symmetric/inverse, 14 chain-of-thought.
- Added `tests/test_batch_memit_metrics.py` for local unit coverage of MEMIT batch request formatting and EasyEdit-style locality preservation summarization. The tests use lightweight stubs and do not require PyTorch/EasyEdit in the Mac `python3` environment.
- Added `tests/test_baseline_ike_cache.py` for local unit coverage of `baseline_ike.py` embedding-cache path construction, cache-hit skip behavior, and rebuild behavior without loading `sentence_transformers`.
- Added CSV export to `scripts/show_results.py`: `--csv_dir results/csv` writes run rows and probe summaries suitable for plotting.
- Added metric-definition sections to README/STATUS/HANDOFF/CLAUDE covering EasyEdit baseline metrics, custom probe metrics, and planned CounterFact/RippleEdits/MQuAKE metric families.
- Local verification commands now include:
  - `python3 scripts/audit_probes.py --min_total 100 --strict`
  - `python3 -m unittest discover -s tests`
  - `python3 scripts/show_results.py --csv_dir /private/tmp/cs263_csv`

## 2026-05-03 — Fixed Claude script issues after review

- Fixed `scripts/baseline_ike.py`: EasyEdit's IKE path expects cached retrieval embeddings under `results/IKE/embedding/`, so the script now calls `encode_ike_facts(...)` before `BaseEditor.edit(...)` and skips rebuilding when the cache exists. Added `--rebuild_embeddings` for explicit refreshes.
- Fixed `scripts/batch_memit.py`: removed first-token-only scoring and switched to EasyEdit's `compute_edit_quality(...)` for rewrite/rephrase. Locality now matches EasyEdit semantics: post-edit locality outputs are compared to pre-edit model outputs, not directly to `locality_ground_truth`.
- Hardened weight restoration in `batch_memit.py` and `run_probes.py` by using EasyEdit `nethook.get_parameter(...)` and `try/finally` around edited-model evaluation.
- Added `probe_type` metadata to `src/probes/probe_set.py`:
  - `implicit_edit`: prompt does not state the new fact.
  - `target_conditioned`: prompt conditions on the new target value or forced choice.
  - `supplied_fact_reasoning`: prompt states the edited fact; analyze separately because the base model can pass by following the prompt.
- Updated `scripts/run_probes.py` and `scripts/show_results.py` to record and summarize probe results by `probe_type`.
- Corrected docs: probe set currently has 34 probes, not 37. IKE probe support is still pending; ROME/MEMIT probes are ready.
- Local verification: `python3 -m py_compile` passed for changed scripts; `python3 scripts/show_results.py --all` runs.

## 2026-05-03 — Scripts written; MEMIT cache timing corrected

- **MEMIT covariance cache timing**: Earlier estimate of ~45–60 min was wrong. Actual wall-clock time on T4 for 5 layers × 100k Wikipedia samples is ~3–4 hours. Updated STATUS.md and docstrings accordingly. The run is still the correct thing to do — it only happens once.
- **New scripts written** (not yet run):
  - `scripts/batch_memit.py`: true MEMIT batch edit — applies all N edits to one model via `apply_memit_to_model`, evaluates that model, supports batch-size sweep (--batch_sizes 10,50,100). Run after single-edit baseline finishes to use the cached covariance stats.
  - `scripts/baseline_ike.py`: IKE scaffold — loads IKE hparams, builds editor, runs 100-edit eval. Requires `all-MiniLM-L6-v2` download (~90 MB) on first run.
  - `configs/IKE/gpt2-xl.yaml`: versioned IKE config using HuggingFace IDs.
  - `src/probes/probe_set.py`: 37 hand-curated diagnostic probes across 5 categories (logical_negation, symmetric_inverse, compositional, contradiction, chain_of_thought) built around the 5 smoke-test edit cases.
  - `scripts/run_probes.py`: probe evaluation runner — applies edit (ROME or MEMIT), runs all probes, records pre/post pass rates per category.
  - `scripts/show_results.py`: updated with paper-comparison delta columns, per-method summary table, MEMIT batch sweep table, probe summary by category, and ASCII bar chart.
- **Next**: wait for MEMIT single-edit baseline to finish, then run batch_memit.py, then baseline_ike.py, then run_probes.py for ROME and MEMIT.

## 2026-05-03 — Clarified single-edit vs. batch-edit baselines

- Important correction: current `scripts/baseline_rome.py` and `scripts/baseline_memit.py` both call `BaseEditor.edit(...)` with `sequential_edit=False`. In EasyEdit this loops over requests one at a time, evaluates the edited model, then restores original weights.
- Therefore the ROME 100-edit run is 100 independent single-edit trials, not one model containing 100 edits.
- The current MEMIT run is also 100 independent single-edit trials. Log evidence: `Writing 1 key/value pair(s)` per request. It is still useful as a single-edit sanity comparison against ROME, but it does not test MEMIT's intended mass-edit advantage.
- MEMIT is currently spending most of its time building first-run Wikipedia covariance caches for layers `[13, 14, 15, 16, 17]`; cache files under `data/stats/gpt2-xl/wikipedia_stats/` should make later MEMIT runs faster.
- Scientific plan:
  - Keep independent single-edit baselines for ROME/MEMIT/IKE on the same CounterFact sample.
  - Add a true MEMIT batch/mass-edit experiment where 100 edits are inserted into one model and then evaluated.
  - Treat ROME mass editing, if run, as cumulative/sequential stress testing rather than ROME's primary intended setting.
  - Treat IKE as retrieval/in-context editing: "batch" means placing multiple demonstrations/facts in the inference context, not modifying one persistent model.
- IKE comparison guidance: evaluate IKE on the same edit records and probes, but report it separately as non-parametric/inference-time editing. A fair "many-edit" IKE condition should vary the number and relevance of in-context edit examples and measure context interference/retrieval failure, not call it a weight-edit batch.
- Future cleanup: build a cleaned rephrase prompt set if paper-style generalization claims are needed; EasyEdit's current rephrase prompts remain relative-only.
- Future scale run: consider full CounterFact size (~2500 paper-style cases) after scripts are stable. For MEMIT mass editing, run batch-size sweeps (e.g. 10/100/1000 if feasible) rather than jumping straight to one expensive full run.

## 2026-05-03 — MEMIT baseline launched; first-run cache is slow

- Launched `scripts/baseline_memit.py --data_path data/counterfact/counterfact-edit.json` in tmux session `memit`.
- Host checks showed the run is alive: Python PID `25662`, T4 at ~94% GPU utilization, log `logs/baseline_memit_.log`.
- It finished covariance cache for layer 13 and started layer 14. This is expected with `mom2_adjustment: true` and `mom2_n_samples: 100000`.
- The earlier impression that the program stopped was just the slow covariance-stat phase plus restricted sandbox process visibility.
- Next: let the current run finish as a single-edit MEMIT baseline/cache warmup, then add a real MEMIT batch script.

## 2026-05-03 — Rephrase failure audit

- Added `scripts/inspect_rephrase_failures.py` to audit EasyEdit per-edit results without loading a model.
- 100-edit ROME run has 46 rephrase failures; static prompt audit flags 34/46 as weak, indirect, noisy, or relation-mismatched.
- Examples: employment-location edits evaluated with "favorite lunchtime work meals include"; language edits evaluated with "lives in"; continent/location edits evaluated with language-of-surrounding-people prompts.
- EasyEdit computes rephrase_acc as teacher-forced token exact match against `target_new`, so bad rephrase prompts can depress the score even when direct rewrite succeeds.
- Decision: do not treat EasyEdit rephrase_acc as absolutely paper-comparable for this dataset version. Use it mainly as a relative metric across ROME/MEMIT/IKE, or rerun on a cleaned rephrase set if the final writeup needs paper-style generalization numbers.
- Cleaned accidental staged log/date artifact files (`2`, `2026.log`, `23:43:45`, `May`, `UTC`) from the repo.
- Decision on rephrase_acc: EasyEdit's rephrase prompts are not comparable to the original ROME paper. Using rephrase_acc as relative-only (ROME vs MEMIT vs IKE). Original ROME repo cross-check deferred — rewrite/locality already confirm pipeline fidelity.
- Next: MEMIT baseline (`scripts/baseline_memit.py`).

## 2026-05-03 — ROME 100-edit baseline complete; rephrase_acc gap flagged

- Baseline ran on 100 random CounterFact edits (seed=42): rewrite=1.000, rephrase=0.540, locality=0.790.
- rewrite and locality within range of paper. rephrase is ~40 points below paper (0.948).
- Likely cause: EasyEdit's rephrase prompts are lower quality than the original ROME paper's. Examples seen in logs like "Marina Tsvetaeva's favorite lunchtime work meals include" as a paraphrase of an employment location — not a valid paraphrase.
- Decision: inspect rephrase failures before trusting rephrase_acc as a metric. Will use it as a relative comparison across methods rather than absolute.
- Also: Matthew is now sole contributor (Corey and Nathan no longer involved). Updated CLAUDE.md and README accordingly.
- Installed Codex CLI (v0.1.28) via nvm/Node 22 on GCP T4.
- Next: inspect rephrase failures, then bring up MEMIT baseline.

---

## 2026-05-02 — ROME smoke test passing; two compatibility bugs fixed

- **nethook.py bug (PyTorch 2.9.1)**: `register_forward_hook` with `with_kwargs=True` passes `(module, args, kwargs, output)` — EasyEdit had the signature as `(m, inputs, output, kwargs=None)`, so `output` received the kwargs dict and the hook returned it as the module's output. This replaced `c_proj`'s tensor output with a dict, crashing `dropout()` on the next line. Fixed by correcting the parameter order in `retain_hook`.
- **summarize() bug**: EasyEdit returns all metrics as lists (e.g. `[1.0]`), not scalars. The `isinstance(node, (int, float))` check in `smoke_test_rome.py` never matched, so all averaged metrics came back `None` and the final assertion fired. Fixed to handle lists and numpy scalars.
- Smoke test now passes cleanly. Results on 5 CounterFact edits:
  - rewrite_acc: **1.00** (paper: ~0.99) ✓
  - rephrase_acc: **0.93** (paper: ~0.93) ✓
  - locality_acc: **0.70** (paper: ~0.79 — within normal variance) ✓
- Pipeline is trusted. Next: run 100-edit CounterFact baseline and compare to paper Table 1 formally.

---

## 2026-04-22 — Environment verified; ROME smoke test script ready

- EasyEdit + ROME repo cloned into `external/`.
- Conda env renamed from `editing` → `cs263-project`; `external/EasyEdit/requirements.txt` installed.
- Local env check (`scripts/check_env.py`) passes clean: all packages present, EasyEdit imports OK (ROME/MEMIT/IKE hparams all load), ROME gpt2-xl config resolves correctly.
- Device: no CUDA locally (Mac); MPS available but EasyEdit ROME is hardcoded to `cuda:{device}` throughout — full ROME edit must run on GCP T4.
- Created `configs/ROME/gpt2-xl.yaml` (versioned copy of EasyEdit hparams, `model_name` changed from local cache path → HuggingFace ID `gpt2-xl`).
- Created `scripts/smoke_test_rome.py`: 5 CounterFact-style edits, logs rewrite/rephrase/locality metrics to `results/runs.jsonl`. Ready to run on GCP.
- One thing to watch: `datasets==1.18.3` is from 2022; may cause issues with newer HF APIs. Monitor on first GCP run.
- Next: spin up GCP T4, run `python scripts/smoke_test_rome.py`, compare rewrite_acc/rephrase_acc to ROME paper Table 1.

## 2026-04-22 — Repo initialized

- Created the repo, dropped in CLAUDE.md and .gitignore.
- Stack decisions locked in: EasyEdit + ROME/MEMIT/IKE + GPT-2 XL + CounterFact/RippleEdits/MQUAKE.
- Planning report submitted for the course.
- Next: get EasyEdit installed and the README example running end-to-end.
