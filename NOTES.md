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
