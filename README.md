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
python scripts/audit_probes.py --min_total 100 --strict
python scripts/run_probes.py --method ROME
python scripts/run_probes.py --method MEMIT

# View all results and probe summaries
python scripts/show_results.py --all
python scripts/show_results.py --csv_dir results/csv

# Local unit tests (no GPU/model load)
python -m unittest discover -s tests
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
| Novel eval | 100 diagnostic probes (contradiction / method-sensitivity / chain-of-thought) |

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
src/probes/           # 100 hand-curated diagnostic probes
tests/                # lightweight local tests for pure utility/metric logic
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
| 2026-05-05 | MEMIT | CounterFact | 100 | 0.810 | 0.230 | 0.980 |
| 2026-05-05 | MEMIT-batch | CounterFact-batch-10 | 10 | 0.900 | 0.100 | 1.000 |

Paper targets (ROME, GPT-2 XL): rewrite ~99.6%, rephrase ~94.8%, locality ~72.2%.
The EasyEdit CounterFact rephrase prompts are noisy, so `rephrase_acc` is relative-only for method comparisons.

## Metric Definitions

### EasyEdit / CounterFact Metrics

These are the baseline metrics reported by `baseline_rome.py`, `baseline_memit.py`, `baseline_ike.py`, and `batch_memit.py`.

| Metric | Also called | Definition | Interpretation |
|--------|-------------|------------|----------------|
| `rewrite_acc` | efficacy, reliability, edit success | Token-level exact-match accuracy for the new target on the original edit prompt after editing. | Measures whether the edit took effect on the exact requested fact. |
| `rephrase_acc` | generalization, paraphrase success | Token-level exact-match accuracy for the same new target on a rephrased prompt. | Measures surface-form transfer. In this repo it is relative-only because EasyEdit's CounterFact rephrase prompts are noisy. |
| `locality_acc` | specificity, neighborhood success | Agreement between post-edit and pre-edit predictions on unrelated locality prompts. | Measures whether unrelated facts remain unchanged. For MEMIT batch, this is explicitly computed as post-edit locality outputs matching pre-edit locality outputs. |
| `n_samples` | edit count | Number of evaluated edit records. | For ROME/MEMIT single-edit scripts this means independent single-edit trials; for `MEMIT-batch` it means facts inserted into one edited model. |
| `seed` | sample seed | Random seed used to sample CounterFact records. | Needed for reproducibility of 100-edit subsets. |

For IKE, the same metric names are used, but the mechanism is different: no weights are modified. The post-edit behavior is base GPT-2 XL plus retrieved in-context examples.

### Custom Probe Metrics

The custom probe set lives in `src/probes/probe_set.py` and is validated by `scripts/audit_probes.py`.

| Metric / Field | Definition | Why it matters |
|----------------|------------|----------------|
| `probe_pass` | A single probe passes if the generated first token matches `expected_first_token`, or the short greedy generation contains `expected_contains`. | Gives a simple binary success signal for each diagnostic question. |
| `pre_pass_rate` | Fraction of probes passed before applying the edit. | Detects probes the base model already answers correctly, especially supplied-fact prompts. |
| `post_pass_rate` | Fraction of probes passed after applying the edit. | Main diagnostic score for edited behavior. |
| `delta_pass_rate` | `post_pass_rate - pre_pass_rate`. | Separates actual edit-induced improvement from prompts that were already easy. |
| `category` | One of `logical_negation`, `symmetric_inverse`, `compositional`, `contradiction`, `chain_of_thought`. | Groups probes by the kind of consistency failure being tested. |
| `probe_type` | One of `implicit_edit`, `target_conditioned`, `supplied_fact_reasoning`. | Separates strong edit-transfer probes from prompts that mention the target or state the edited fact. |

Probe categories:

- `logical_negation`: asks for the edited fact through a new surface form, or asks the model to stop predicting the old value.
- `symmetric_inverse`: tests whether an edit transfers from subject-to-object form into inverse object-to-subject queries.
- `compositional`: tests whether the edited fact combines with another known fact to produce an implied answer.
- `contradiction`: asks whether the old and new facts are still treated as simultaneously true.
- `chain_of_thought`: supplies or elicits a short reasoning chain and checks whether the conclusion remains consistent.

Probe types:

- `implicit_edit`: the prompt does not state the new target. These are the strongest evidence of edit transfer.
- `target_conditioned`: the prompt mentions the edited target or presents a forced choice. These are useful but weaker than implicit probes.
- `supplied_fact_reasoning`: the prompt states the edited fact and tests reasoning from it. Analyze separately because the base model can pass by following the prompt.

### Planned Benchmark Metrics

These are not implemented yet, but they define the intended evaluation for future RippleEdits and MQuAKE work.

| Benchmark | Metrics / Criteria | Definition |
|-----------|--------------------|------------|
| CounterFact | efficacy, paraphrase/generalization, specificity/locality | Direct edit success, transfer to paraphrases, and preservation of unrelated facts. |
| RippleEdits | logical generalization, compositionality I/II, subject aliasing, preservation, relation specificity | Measures whether an edit propagates through logical implications and compositions, applies to aliases of the subject, preserves other correct target objects, and avoids changing unrelated relations. |
| MQuAKE | edited-fact accuracy, multi-hop QA accuracy, hop-specific accuracy, one-edited/all-edited conditions | Measures whether edited facts are recalled and whether downstream multi-hop questions whose answers should change after the edit are answered correctly. |

RippleEdits and MQuAKE are closer to the custom probes than CounterFact: they focus on ripple effects and multi-hop consistency, not only direct rewrite success. The custom probe set is smaller and hand-auditable, with explicit `probe_type` labels for separating implicit transfer from supplied-premise reasoning.

### Experiment Interpretation

The current ROME and MEMIT baseline scripts run **independent single-edit trials**. In EasyEdit, `BaseEditor.edit(..., sequential_edit=False)` edits one request, evaluates it, restores original weights, and then moves to the next request.

That means `N=100` is not one model with 100 stored edits. It is 100 sampled CounterFact cases evaluated independently.

Current follow-up experiments:

- `scripts/batch_memit.py` inserts many MEMIT edits into one model and evaluates that edited model with EasyEdit-compatible rewrite/rephrase/locality metrics.
- `scripts/baseline_ike.py` evaluates IKE as retrieval/in-context editing. It builds cached retrieval embeddings under `results/IKE/embedding/` on first run.
- `scripts/audit_probes.py` validates the 100-probe set before GPU runs.
- `scripts/run_probes.py` runs the custom probe set for ROME and MEMIT. Probe records include `probe_type` so implicit edit tests are separated from target-conditioned and supplied-fact reasoning prompts.
- `scripts/show_results.py --csv_dir results/csv` exports runs and probe summaries for plotting.
- Keep `rephrase_acc` relative-only until rephrase prompts are cleaned or replaced with paper-style paraphrases.

---

## Compatibility notes

Tested with PyTorch 2.9.1 + transformers 4.57.1. Two bugs were fixed relative to upstream EasyEdit:
1. `nethook.py`: incorrect hook signature for PyTorch 2.0+ `with_kwargs=True` (patch in `patches/`)
2. `smoke_test_rome.py`: metrics returned as lists, not scalars — summarize() updated accordingly
