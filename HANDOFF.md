# Local Handoff

Snapshot date: 2026-05-03

This is the working handoff for continuing locally while the GCP MEMIT job runs.

## Current Remote Job

MEMIT is running on the GCP T4 in tmux session `memit`.

```bash
tmux attach -t memit
```

Current command:

```bash
python scripts/baseline_memit.py --data_path data/counterfact/counterfact-edit.json 2>&1 | tee logs/baseline_memit_.log
```

Important interpretation:

- This is a MEMIT single-edit sanity baseline/cache warmup.
- It is not a true 100-edit MEMIT mass edit.
- Log lines like `Writing 1 key/value pair(s)` confirm EasyEdit is evaluating independent single-edit requests.
- The slow part is first-run Wikipedia covariance cache generation for layers `[13, 14, 15, 16, 17]`.

Useful checks on the remote box:

```bash
tail -n 40 logs/baseline_memit_.log
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

- 34 probes across logical negation, symmetric/inverse, compositional, contradiction, and chain-of-thought categories.
- Each probe has a `probe_type`:
  - `implicit_edit`: does not state the new fact.
  - `target_conditioned`: mentions the edited target or a forced choice.
  - `supplied_fact_reasoning`: states the edited fact and tests reasoning from it.
- Analyze supplied-fact probes separately because a base model can pass by following the prompt rather than because the edit propagated.

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

Possible improvements:

- table grouped by method/dataset
- deltas from paper targets where relevant
- explicit labeling for `single_edit`, `batch_edit`, and `retrieval_context`
- CSV export for figures

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
