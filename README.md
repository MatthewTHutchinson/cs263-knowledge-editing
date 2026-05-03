# CS 263 — Knowledge Editing Comparison

*When Surgical Edits Leak: A Comparative Study of Logical Consistency and Ripple Effects Across Knowledge Editing Methods*

Compares ROME, MEMIT, and IKE on GPT-2 XL using CounterFact, RippleEdits, and MQuAKE, with a custom diagnostic probe set targeting logical consistency and ripple effects.

**Team**: Matthew Hutchinson (mahutchinson@ucla.edu), Corey Shen (corey0224@ucla.edu), Nathan Wei (nathanwei@ucla.edu)

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
`data/stats/` (ROME's Wikipedia covariance cache) will recompute on first ROME run (~30 min on T4).

---

## Running experiments

```bash
conda activate cs263-project
cd cs263-knowledge-editing

# Sanity check (no GPU needed)
python scripts/check_env.py

# 5-edit smoke test (confirms pipeline end-to-end)
python scripts/smoke_test_rome.py

# 100-edit baseline vs. paper
python scripts/baseline_rome.py --data_path data/counterfact/counterfact-edit.json

# View all results
python scripts/show_results.py
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
data/counterfact/     # EasyEdit CounterFact dataset (10K records, in repo)
data/stats/           # ROME covariance cache (gitignored, recomputed on first run)
results/runs.jsonl    # structured run log (all experiments)
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
Rephrase gap vs. paper likely due to EasyEdit's rephrase prompt quality — under investigation.

---

## Compatibility notes

Tested with PyTorch 2.9.1 + transformers 4.57.1. Two bugs were fixed relative to upstream EasyEdit:
1. `nethook.py`: incorrect hook signature for PyTorch 2.0+ `with_kwargs=True` (patch in `patches/`)
2. `smoke_test_rome.py`: metrics returned as lists, not scalars — summarize() updated accordingly
