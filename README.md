# CS 263 — Knowledge Editing Comparison

*When Surgical Edits Leak: A Comparative Study of Logical Consistency and Ripple Effects Across Knowledge Editing Methods*

Compares ROME, MEMIT, and IKE on GPT-2 XL using CounterFact, RippleEdits, and MQuAKE, with a custom diagnostic probe set targeting logical consistency and ripple effects.

**Implementation owner**: Matthew Hutchinson (mahutchinson@ucla.edu)

---

## Setup

```bash
git clone git@github.com:MatthewTHutchinson/cs263-knowledge-editing.git
cd cs263-knowledge-editing

# Clone EasyEdit (required) and apply compatibility patch
git clone https://github.com/zjunlp/EasyEdit external/EasyEdit
cd external/EasyEdit
patch -p1 < ../../patches/0001-fix-nethook-pytorch29-with_kwargs-signature.patch
cd ../..

# Create conda env
conda create -n cs263-project python=3.10 -y
conda activate cs263-project
pip install -r external/EasyEdit/requirements.txt
```

**Note**: `data/counterfact/` is included in the repo — no separate download needed.
`data/stats/` (ROME/MEMIT Wikipedia covariance cache) will recompute on first run; MEMIT's GPT-2 XL cache can take several hours on T4.

---

## Running experiments

```bash
conda activate cs263-project
cd cs263-knowledge-editing

# Sanity check (no GPU needed)
python scripts/check_env.py

# 5-edit smoke test (confirms pipeline end-to-end)
python scripts/smoke_test_rome.py

# 100 independent single-edit baseline vs. paper
python scripts/baseline_rome.py --data_path data/counterfact/counterfact-edit.json

# MEMIT single-edit baseline/cache warmup
python scripts/baseline_memit.py --data_path data/counterfact/counterfact-edit.json

# True MEMIT batch/mass-edit sweep (run after MEMIT covariance cache is warm)
python scripts/batch_memit.py --data_path data/counterfact/counterfact-edit.json --batch_sizes 10,50,100

# IKE retrieval/in-context baseline
python scripts/baseline_ike.py --data_path data/counterfact/counterfact-edit.json

# Diagnostic probes for post-edit consistency
python scripts/run_probes.py --method ROME
python scripts/run_probes.py --method MEMIT

# View all results and probe summaries
python scripts/show_results.py --all
```

---

## Stack

| Component | Choice |
|-----------|--------|
| Framework | [EasyEdit](https://github.com/zjunlp/EasyEdit) (Wang et al., ACL 2024) |
| Methods | ROME, MEMIT, IKE |
| Model | GPT-2 XL (1.5B); GPT-J (6B) optional |
| Benchmarks | CounterFact, RippleEdits, MQuAKE |
| Compute | GCP T4 (preemptible) |
| Novel eval | ~50 diagnostic probes (contradiction / method-sensitivity / chain-of-thought) |

---

## Repo layout

```
scripts/              # runnable experiment scripts
configs/ROME/         # versioned YAML hparams
configs/MEMIT/        # versioned YAML hparams
configs/IKE/          # versioned YAML hparams
data/counterfact/     # EasyEdit CounterFact dataset (10K records, in repo)
data/stats/           # ROME/MEMIT covariance cache (gitignored, recomputed on first run)
results/runs.jsonl    # structured run log (all experiments)
src/probes/           # hand-curated diagnostic probe set
patches/              # fixes for gitignored external/EasyEdit
external/EasyEdit/    # gitignored — clone manually per setup above
NOTES.md              # daily working log
STATUS.md             # project map and current state
CLAUDE.md             # context for Claude Code sessions
```

---

## Results

| Date | Method | Dataset | N | Rewrite | Rephrase | Locality |
|------|--------|---------|---|---------|----------|----------|
| 2026-05-02 | ROME | CounterFact-smoke | 5 | 1.000 | 0.933 | — |
| 2026-05-03 | ROME | CounterFact | 100 | 1.000 | 0.540 | 0.790 |

Paper targets (ROME, GPT-2 XL): rewrite ~99.6%, rephrase ~94.8%, locality ~72.2%.
The EasyEdit CounterFact rephrase prompts are noisy, so `rephrase_acc` is relative-only for method comparisons.

### Experiment Interpretation

The current ROME and MEMIT baseline scripts run **independent single-edit trials**. In EasyEdit, `BaseEditor.edit(..., sequential_edit=False)` edits one request, evaluates it, restores original weights, and then moves to the next request.

That means `N=100` is not one model with 100 stored edits. It is 100 sampled CounterFact cases evaluated independently.

Current follow-up experiments:

- `scripts/batch_memit.py` inserts many MEMIT edits into one model and evaluates that edited model with EasyEdit-compatible rewrite/rephrase/locality metrics.
- `scripts/baseline_ike.py` evaluates IKE as retrieval/in-context editing. It builds cached retrieval embeddings under `results/IKE/embedding/` on first run.
- `scripts/run_probes.py` runs the custom probe set for ROME and MEMIT. Probe records include `probe_type` so implicit edit tests are separated from target-conditioned and supplied-fact reasoning prompts.
- Keep `rephrase_acc` relative-only until rephrase prompts are cleaned or replaced with paper-style paraphrases.

---

## Compatibility notes

Tested with PyTorch 2.9.1 + transformers 4.57.1. Two bugs were fixed relative to upstream EasyEdit:
1. `nethook.py`: incorrect hook signature for PyTorch 2.0+ `with_kwargs=True` (patch in `patches/`)
2. `smoke_test_rome.py`: metrics returned as lists, not scalars — summarize() updated accordingly
