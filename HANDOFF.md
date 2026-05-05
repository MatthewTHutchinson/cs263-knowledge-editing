# Local Handoff

Snapshot date: 2026-05-05

This is the working handoff for continuing locally while the GCP MEMIT job runs.

## Current Remote Job

MEMIT is running on the GCP T4 in tmux session `memit`.

```bash
tmux attach -t memit
```

Current command/launcher:

```bash
scripts/run_memit_checkpointed.sh
```

Important interpretation:

- This is a MEMIT single-edit sanity baseline/cache warmup.
- It is not a true 100-edit MEMIT mass edit.
- Log lines like `Writing 1 key/value pair(s)` confirm EasyEdit is evaluating independent single-edit requests.
- The slow part is first-run Wikipedia covariance cache generation for layers `[13, 14, 15, 16, 17]`.
- This is the **5th attempt** to finish the MEMIT cache/baseline. The repeated failure point has been layer 17 covariance generation.

## MEMIT Layer 17 Checkpointing

As of 2026-05-05, EasyEdit's local covariance collector has been patched in:

```text
external/EasyEdit/easyeditor/models/rome/layer_stats.py
```

Portable patch copy:

```text
patches/0002-add-easyedit-layer-stats-partial-checkpoints.patch
```

The patch writes a resumable partial covariance file during the long layer-stat loop instead of waiting until all 1000 batch groups finish. This matters because upstream EasyEdit only writes the final `.npz` after the whole layer completes; any VM interruption before that loses the whole layer.

Current layer-cache state:

```text
data/stats/gpt2-xl/wikipedia_stats/transformer.h.13.mlp.c_proj_float32_mom2_100000.npz
data/stats/gpt2-xl/wikipedia_stats/transformer.h.14.mlp.c_proj_float32_mom2_100000.npz
data/stats/gpt2-xl/wikipedia_stats/transformer.h.15.mlp.c_proj_float32_mom2_100000.npz
data/stats/gpt2-xl/wikipedia_stats/transformer.h.16.mlp.c_proj_float32_mom2_100000.npz
data/stats/gpt2-xl/wikipedia_stats/transformer.h.17.mlp.c_proj_float32_mom2_100000.npz.partial.npz
```

The layer 17 `.partial.npz` means the job can resume from the last saved batch group if it crashes during layer 17 covariance computation. The checkpoint interval is controlled by:

```bash
EASYEDIT_STATS_CHECKPOINT_INTERVAL=10
```

`scripts/run_memit_checkpointed.sh` sets this by default and writes the active log path to:

```text
logs/baseline_memit_latest.path
```

If the VM or process dies, restart from the repo root with:

```bash
tmux new-session -d -s memit scripts/run_memit_checkpointed.sh
```

Then check:

```bash
cat logs/baseline_memit_latest.path
tail -n 80 "$(cat logs/baseline_memit_latest.path)"
find data/stats/gpt2-xl/wikipedia_stats -maxdepth 1 -type f -printf '%f %s bytes\n' | sort
nvidia-smi
```

Expected resume signal in the log:

```text
Resuming partial covariance stats from ...layer.h.17...partial.npz after N batch groups.
```

When layer 17 finishes, the final expected file is:

```text
data/stats/gpt2-xl/wikipedia_stats/transformer.h.17.mlp.c_proj_float32_mom2_100000.npz
```

After that final `.npz` exists, future MEMIT runs should skip all five covariance computations and move much faster.

Useful checks on the remote box:

```bash
cat logs/baseline_memit_latest.path
tail -n 40 "$(cat logs/baseline_memit_latest.path)"
find data/stats/gpt2-xl/wikipedia_stats -maxdepth 1 -type f -printf '%f %TY-%Tm-%Td %TH:%TM:%TS\n' | sort
nvidia-smi
ps -eo pid,ppid,stat,etime,pcpu,pmem,args | rg 'baseline_memit|python scripts'
```

Expected cache files:

```text
data/stats/gpt2-xl/wikipedia_stats/transformer.h.13.mlp.c_proj_float32_mom2_100000.npz
data/stats/gpt2-xl/wikipedia_stats/transformer.h.14.mlp.c_proj_float32_mom2_100000.npz
data/stats/gpt2-xl/wikipedia_stats/transformer.h.15.mlp.c_proj_float32_mom2_100000.npz
data/stats/gpt2-xl/wikipedia_stats/transformer.h.16.mlp.c_proj_float32_mom2_100000.npz
data/stats/gpt2-xl/wikipedia_stats/transformer.h.17.mlp.c_proj_float32_mom2_100000.npz
```

The cache files are intentionally not committed.

## Resource Diagnosis

This does not look like a storage problem: the layer covariance files are about 157 MB each, and the disk had roughly 28 GB free when checked on 2026-05-05.

This also does not look like a T4 VRAM limitation for GPT-2 XL MEMIT covariance generation: the layer 17 run was using about 8.5 GB of 15 GB with the GPU near full utilization.

The GCP error `ZONE_RESOURCE_POOL_EXHAUSTED_WITH_DETAILS` is a zone capacity/provisioning issue when trying to create a new GPU VM. It is not evidence that this job needs a different GPU type. Since the current T4 can run the workload, prefer finishing on it with checkpointing rather than repeatedly chasing new T4/L4 capacity in other zones.

## Capacity Guidance

Do not start another GPU-heavy job while MEMIT covariance generation is running. Avoid launching:

- another MEMIT run
- ROME or IKE with GPT-2 XL
- probe evaluation against GPT-2 XL
- any script that loads a second large model on the T4

Safe work while the remote job runs:

- write scripts locally without running full GPU evaluations
- inspect JSON data formats
- design probes
- write plotting/result summarization utilities
- update notes/status docs
- run lightweight syntax checks or CPU-only tests

## Metric Definitions

Core baseline metrics:

- `rewrite_acc`: exact-match token accuracy for the new target on the original edit prompt after editing.
- `rephrase_acc`: exact-match token accuracy for the new target on a rephrased prompt. Treat as relative-only because EasyEdit CounterFact rephrase prompts are noisy.
- `locality_acc`: preservation of unrelated behavior. For ROME/MEMIT, compare post-edit locality predictions to pre-edit locality predictions. Do not treat this as direct accuracy against `locality_ground_truth`.

Probe metrics:

- Per-probe pass: generated first token matches `expected_first_token`, or short greedy generation contains `expected_contains`.
- `pre_pass_rate`: fraction of probes passed before editing.
- `post_pass_rate`: fraction passed after editing.
- `delta_pass_rate`: `post_pass_rate - pre_pass_rate`; most useful for edit-induced improvement.
- Category summaries: logical negation, symmetric/inverse, compositional, contradiction, chain-of-thought.
- Type summaries: `implicit_edit`, `target_conditioned`, `supplied_fact_reasoning`.

Future benchmark metrics:

- CounterFact: efficacy/rewrite, paraphrase/generalization, locality/specificity.
- RippleEdits: logical generalization, compositionality I/II, subject aliasing, preservation, relation specificity.
- MQuAKE: edited-fact accuracy and multi-hop QA accuracy, with hop-count and one-edited/all-edited breakdowns when available.

## Immediate Local Tasks

### 1. True MEMIT Batch Script

Goal: run the script that performs one model edit containing many facts, then evaluates the edited model.

Suggested file:

```text
scripts/batch_memit.py
```

Key requirement: do not accidentally reproduce the current independent single-edit loop. Confirm through logs or EasyEdit internals that the edit writes many key/value pairs into one model update.

Current implementation notes:

- Calls `apply_memit_to_model(...)` once with all N requests.
- Uses EasyEdit's `compute_edit_quality(...)` for rewrite/rephrase.
- Computes locality as preservation of pre-edit locality predictions, matching EasyEdit's baseline semantics.

Suggested CLI:

```bash
python scripts/batch_memit.py \
  --data_path data/counterfact/counterfact-edit.json \
  --batch_sizes 10,50,100 \
  --seed 42
```

Start with small dry runs once GPU is free:

```bash
python scripts/batch_memit.py --data_path data/counterfact/counterfact-edit.json --batch_sizes 10 --seed 42
```

### 2. IKE Baseline

Goal: run a same-record IKE baseline for CounterFact.

Suggested file:

```text
scripts/baseline_ike.py
```

Interpretation:

- IKE does not modify model weights.
- A fair baseline is base model plus retrieved/in-context edit examples at inference time.
- Do not describe IKE as producing a persistent edited model.
- `baseline_ike.py` now builds EasyEdit's retrieval embedding cache under `results/IKE/embedding/` before evaluation. Use `--rebuild_embeddings` only when the retrieval pool or sentence model changes.

Useful comparison conditions:

- single relevant edit in context
- many-edit retrieval memory
- many-edit context stress test

### 3. Probe Set

Goal: run and refine the hand-curated probe set.

Current location:

```text
src/probes/probe_set.py
```

Current state:

- 100 probes across logical negation, symmetric/inverse, compositional, contradiction, and chain-of-thought categories.
- Each probe has a `probe_type`:
  - `implicit_edit`: does not state the new fact.
  - `target_conditioned`: mentions the edited target or a forced choice.
  - `supplied_fact_reasoning`: states the edited fact and tests reasoning from it.
- Analyze supplied-fact probes separately because a base model can pass by following the prompt rather than because the edit propagated.
- Validate locally before GPU runs:
  ```bash
  python scripts/audit_probes.py --min_total 100 --strict
  ```

Probe categories:

- logical negation
- symmetric/inverse relation
- compositional/transitive implication
- contradiction choice
- reasoning-chain consistency

### 4. Probe Runner

Goal: run the same probe records against post-edit behavior for ROME and MEMIT.

Suggested file:

```text
scripts/run_probes.py
```

IKE probe support is still pending because IKE needs an inference-context wrapper rather than a weight-edited model.

### 5. Result Utilities

Goal: make analysis easier once MEMIT/IKE/probes produce records.

Existing file:

```text
scripts/show_results.py
```

Current capabilities:

- table grouped by method/dataset
- deltas from paper targets where relevant
- explicit labeling for `single_edit`, `batch_edit`, and `retrieval_context`
- probe summaries by category and `probe_type`
- CSV export for figures:
  ```bash
  python scripts/show_results.py --csv_dir results/csv
  ```

## Pulling This State Locally

From your local machine:

```bash
git pull origin main
```

Then work on scripts locally. Push back only code/data/docs, not logs, model weights, HuggingFace caches, or `data/stats/`.

## Files To Read First

```text
STATUS.md
NOTES.md
README.md
scripts/baseline_memit.py
scripts/baseline_rome.py
configs/MEMIT/gpt2-xl.yaml
```
