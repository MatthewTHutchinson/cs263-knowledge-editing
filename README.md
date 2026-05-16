# CS 263 — Knowledge Editing Comparison

*Beyond Rewrite Accuracy: Testing Logical Consistency in Knowledge Editing*

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

On a fresh Ubuntu GPU VM, install runtime tools first:

```bash
sudo apt-get update
sudo apt-get install -y git-lfs tmux
```

Use Ubuntu 22.04 with Python 3.10 in the `cs263-project` conda environment. Avoid system Python 3.12 for this project because EasyEdit and its pinned dependencies are older.

**Note**: `data/counterfact/` is included in the repo — no separate download needed.
The stable GPT-2 XL MEMIT/ROME covariance cache files under `data/stats/gpt2-xl/wikipedia_stats/*.npz` are tracked with Git LFS because recomputing them can take several hours on T4.

### Restoring the MEMIT cache

The expensive MEMIT covariance cache is tracked with Git LFS. Make sure Git LFS is installed before cloning or pulling:

```bash
git lfs install
git lfs pull
git lfs ls-files --size
```

There is also a VM backup archive for transition safety:

```text
/home/matthewthutchinson1/cs263-memit-preserve-20260510.tar.gz
sha256 f15b0cd7f85bf9b597572476f083f6151358dcbfe4474e99ca097f6471b3c73b
```

The archive contains `data/stats/`, `results/`, `logs/`, `configs/`, `scripts/`, `patches/`, and the project notes. If restoring from the archive instead of LFS, run from the repo root:

```bash
gcloud storage cp gs://cs263-project-494118-memit-backup/cs263-memit-preserve-20260510.tar.gz ~/
sha256sum ~/cs263-memit-preserve-20260510.tar.gz
tar -xzf ~/cs263-memit-preserve-20260510.tar.gz
find data/stats/gpt2-xl/wikipedia_stats -maxdepth 1 -type f -name '*.npz' -printf '%f %s bytes\n' | sort
nvidia-smi
```

Expected cache files:

```text
transformer.h.13.mlp.c_proj_float32_mom2_100000.npz
transformer.h.14.mlp.c_proj_float32_mom2_100000.npz
transformer.h.15.mlp.c_proj_float32_mom2_100000.npz
transformer.h.16.mlp.c_proj_float32_mom2_100000.npz
transformer.h.17.mlp.c_proj_float32_mom2_100000.npz
```

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

# Download and inspect external ripple/multihop benchmarks
python scripts/download_benchmarks.py --dataset all
python scripts/inspect_benchmarks.py --mquake data/mquake/MQuAKE-CF-3k-v2.json
python scripts/inspect_benchmarks.py --ripple data/ripple_edits/POPULAR.json

# External benchmark smoke/sweep runs
python scripts/eval_mquake.py --method ROME --n_cases 1 --edit_mode one
python scripts/eval_mquake.py --method MEMIT --n_cases 1 --edit_mode all
python scripts/eval_mquake.py --method IKE --n_cases 1 --edit_mode all
python scripts/eval_ripple_edits.py --method ROME --n_cases 1 --subset POPULAR
python scripts/eval_ripple_edits.py --method IKE --n_cases 1 --subset POPULAR \
    --require_criteria Logical_Generalization,Subject_Aliasing

python scripts/eval_mquake.py --method IKE --n_cases 25 --edit_mode all
python scripts/eval_mquake.py --method ROME --n_cases 10 --edit_mode one
python scripts/eval_mquake.py --method MEMIT --n_cases 10 --edit_mode all
python scripts/eval_ripple_edits.py --method ROME --n_cases 10 --subset POPULAR \
    --require_criteria Logical_Generalization,Subject_Aliasing
python scripts/eval_ripple_edits.py --method IKE --n_cases 25 --subset POPULAR \
    --require_criteria Logical_Generalization,Subject_Aliasing

# Equal-sample external sweeps with fixed RippleEdits relation-specificity handling
python scripts/eval_mquake.py --method ROME --n_cases 25 --edit_mode one
python scripts/eval_mquake.py --method MEMIT --n_cases 25 --edit_mode all
python scripts/eval_mquake.py --method IKE --n_cases 25 --edit_mode all
python scripts/eval_ripple_edits.py --method ROME --subset POPULAR --n_cases 25 \
    --require_criteria Relation_Specificity,Logical_Generalization,Subject_Aliasing
python scripts/eval_ripple_edits.py --method MEMIT --subset POPULAR --n_cases 25 \
    --require_criteria Relation_Specificity,Logical_Generalization,Subject_Aliasing
python scripts/eval_ripple_edits.py --method IKE --subset POPULAR --n_cases 25 \
    --require_criteria Relation_Specificity,Logical_Generalization,Subject_Aliasing

python scripts/eval_mquake.py --method ROME --n_cases 100 --edit_mode one
python scripts/eval_mquake.py --method MEMIT --n_cases 100 --edit_mode all
python scripts/eval_mquake.py --method IKE --n_cases 100 --edit_mode all
python scripts/eval_ripple_edits.py --method ROME --subset POPULAR --n_cases 100 \
    --require_criteria Relation_Specificity,Logical_Generalization,Subject_Aliasing
python scripts/eval_ripple_edits.py --method MEMIT --subset POPULAR --n_cases 100 \
    --require_criteria Relation_Specificity,Logical_Generalization,Subject_Aliasing
python scripts/eval_ripple_edits.py --method IKE --subset POPULAR --n_cases 100 \
    --require_criteria Relation_Specificity,Logical_Generalization,Subject_Aliasing

# Diagnostic probes for post-edit consistency
python scripts/audit_probes.py --min_total 100 --strict
python scripts/run_probes.py --method ROME
python scripts/run_probes.py --method MEMIT
python scripts/run_probes.py --method IKE --data_path data/counterfact/counterfact-edit.json

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
| Compute | GCP T4; prefer non-preemptible/on-demand for long MEMIT cache or probe runs |
| Novel eval | 100 diagnostic probes (contradiction / method-sensitivity / chain-of-thought) |

---

## Repo layout

```
scripts/              # runnable experiment scripts
configs/ROME/         # versioned YAML hparams
configs/MEMIT/        # versioned YAML hparams
configs/IKE/          # versioned YAML hparams
data/counterfact/     # EasyEdit CounterFact dataset (10K records, in repo)
data/mquake/          # downloaded MQuAKE-CF-3k-v2 benchmark
data/ripple_edits/    # downloaded RippleEdits POPULAR/RANDOM/RECENT subsets
data/stats/           # ROME/MEMIT covariance cache; stable GPT-2 XL .npz files tracked via Git LFS
results/runs.jsonl    # structured run log (all experiments)
results/probe_results.jsonl # per-probe ROME/MEMIT/IKE diagnostic results
src/benchmarks/       # MQuAKE/RippleEdits adapters, scoring, and summaries
src/probes/           # 100 hand-curated diagnostic probes
overleaf_midterm/     # minimal Overleaf upload package and source for the midterm report
tests/                # lightweight local tests for pure utility/metric logic
patches/              # fixes for gitignored external/EasyEdit
external/EasyEdit/    # gitignored — clone manually per setup above
NOTES.md              # daily working log
STATUS.md             # project map and current state
```

---

## Results

| Date | Method | Dataset | N | Rewrite | Rephrase | Locality |
|------|--------|---------|---|---------|----------|----------|
| 2026-05-02 | ROME | CounterFact-smoke | 5 | 1.000 | 0.933 | — |
| 2026-05-03 | ROME | CounterFact | 100 | 1.000 | 0.540 | 0.790 |
| 2026-05-05 | MEMIT | CounterFact | 100 | 0.810 | 0.230 | 0.980 |
| 2026-05-05 | MEMIT-batch | CounterFact-batch-10 | 10 | 0.900 | 0.100 | 1.000 |
| 2026-05-05 | MEMIT-batch | CounterFact-batch-50 | 50 | 0.820 | 0.180 | 0.960 |
| 2026-05-05 | MEMIT-batch | CounterFact-batch-100 | 100 | 0.820 | 0.260 | 0.900 |
| 2026-05-05 | IKE | CounterFact | 5 | 1.000 | 1.000 | 0.200 |
| 2026-05-10 | IKE | CounterFact | 50 | 1.000 | 1.000 | 0.080 |
| 2026-05-10 | IKE | CounterFact | 100 | 0.990 | 0.990 | 0.110 |

The larger IKE runs confirm strong in-context rewrite/rephrase behavior on the sampled records, but poor locality: retrieved demonstrations often perturb unrelated neighborhood prompts.

External benchmark runs were added on 2026-05-11. The n=1 rows are smoke tests for the edit/evaluate/restore path; the n=10/n=25 rows are the current report-level external signal.

| Method | Dataset | N | Primary metrics |
|--------|---------|---|-----------------|
| ROME | MQuAKE-CF-3k-v2-one | 1 | edited_fact_acc=0.500, multihop_acc=0.000 |
| MEMIT | MQuAKE-CF-3k-v2-all | 1 | edited_fact_acc=0.750, multihop_acc=0.000 |
| IKE | MQuAKE-CF-3k-v2-all | 1 | edited_fact_acc=1.000, multihop_acc=0.667 |
| IKE | MQuAKE-CF-3k-v2-all | 5 | edited_fact_acc=0.833, multihop_acc=0.200 |
| ROME | RippleEdits-POPULAR | 1 | overall_acc=0.000 on the available sampled criterion |
| ROME | RippleEdits-POPULAR | 1 | logical_generalization=0.000, subject_aliasing=0.000 on a targeted sample |
| IKE | RippleEdits-POPULAR | 1 | logical_generalization=0.000, subject_aliasing=0.000 on the same targeted sample |
| IKE | MQuAKE-CF-3k-v2-all | 25 | edited_fact_acc=0.910, multihop_acc=0.453, delta_multihop_acc=+0.347 |
| ROME | MQuAKE-CF-3k-v2-one | 10 | edited_fact_acc=0.440, multihop_acc=0.100, delta_multihop_acc=+0.033 |
| MEMIT | MQuAKE-CF-3k-v2-all | 10 | edited_fact_acc=0.680, multihop_acc=0.033, delta_multihop_acc=-0.033 |
| ROME | RippleEdits-POPULAR | 10 | overall_acc=0.160, delta_overall_acc=+0.136, subject_aliasing=0.375, logical_generalization=0.000 |
| IKE | RippleEdits-POPULAR | 25 | overall_acc=0.347, delta_overall_acc=+0.299, subject_aliasing=0.692, logical_generalization=0.237 |

The main external-benchmark pattern is that IKE's in-context/PROMPT setup gives the strongest MQuAKE and RippleEdits gains, especially on subject aliasing. ROME and MEMIT still achieve direct edited-fact improvements, but multi-hop and logical-generalization transfer remains weak.

New external benchmark runs also log pre-edit rates and deltas:

| Metric | Meaning |
|--------|---------|
| `pre_edited_fact_acc` | Accuracy on edited single-hop facts before applying the edit or in-context facts. |
| `edited_fact_acc` | Accuracy on edited single-hop facts after the edit/in-context facts. |
| `delta_edited_fact_acc` | Post minus pre edited-fact accuracy. |
| `pre_multihop_acc` | MQuAKE multi-hop QA accuracy before the edit/in-context facts. |
| `multihop_acc` | MQuAKE multi-hop QA accuracy after the edit/in-context facts. |
| `delta_multihop_acc` | Post minus pre MQuAKE multi-hop accuracy. |
| `pre_overall_acc` | RippleEdits criterion-query accuracy before the edit/in-context fact. |
| `overall_acc` | RippleEdits criterion-query accuracy after the edit/in-context fact. |
| `delta_overall_acc` | Post minus pre RippleEdits accuracy. |

Paper targets (ROME, GPT-2 XL): rewrite ~99.6%, rephrase ~94.8%, locality ~72.2%.
The EasyEdit CounterFact rephrase prompts are noisy, so `rephrase_acc` is relative-only for method comparisons.

### Diagnostic Probe Results

ROME and MEMIT probe sweeps completed on the GCP T4 VM on 2026-05-11. Each method was evaluated on the full 100-probe custom set.

| Method | N probes | Pre pass | Post pass | Delta |
|--------|----------|----------|-----------|-------|
| IKE | 100 | 0.360 | 0.500 | +0.140 |
| MEMIT | 100 | 0.360 | 0.640 | +0.280 |
| ROME | 100 | 0.360 | 0.640 | +0.280 |

Category-level highlights:

| Category | IKE post | MEMIT post | ROME post | Main takeaway |
|----------|----------|------------|-----------|---------------|
| Logical negation | 0.250 | 0.875 | 0.875 | IKE remains much weaker on direct negation than the weight-edit methods. |
| Compositional | 0.765 | 0.706 | 0.765 | ROME shows a small gain; MEMIT is flat relative to pre-edit. |
| Contradiction | 0.591 | 0.455 | 0.455 | IKE improves contradiction prompts somewhat, but not enough to close the gap. |
| Symmetric inverse | 0.133 | 0.067 | 0.000 | Inverse queries remain the clearest failure mode across all methods. |
| Chain-of-thought | 1.000 | 0.929 | 0.929 | IKE passes most supplied-fact reasoning prompts by construction; interpret separately from implicit probes. |

## Metric Definitions

### EasyEdit / CounterFact Metrics

These are the baseline metrics reported by `baseline_rome.py`, `baseline_memit.py`, `baseline_ike.py`, and `batch_memit.py`.

| Metric | Also called | Definition | Interpretation |
|--------|-------------|------------|----------------|
| `rewrite_acc` | efficacy, reliability, edit success | Token-level exact-match accuracy for the new target on the original edit prompt after editing. | Measures whether the edit took effect on the exact requested fact. |
| `rephrase_acc` | generalization, paraphrase success | Token-level exact-match accuracy for the same new target on a rephrased prompt. | Measures surface-form transfer. In this repo it is relative-only because EasyEdit's CounterFact rephrase prompts are noisy. |
| `locality_acc` | specificity, neighborhood success | Agreement between post-edit and pre-edit predictions on unrelated locality prompts. | Measures whether unrelated facts remain unchanged. For MEMIT batch, this is explicitly computed as post-edit locality outputs matching pre-edit locality outputs. |
| `n_samples` | edit count | Number of evaluated edit records. | For ROME/MEMIT single-edit scripts this means independent single-edit trials; for `MEMIT-batch` it means facts inserted into one edited model; for IKE it means in-context edit records, not stored weights. |
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

Example probe interpretations for the Sanofi edit (`Paris` → `Berlin`):

- Logical negation / surface transfer: "Sanofi's corporate headquarters is located in the city of ..." should complete with `Berlin`, not `Paris`.
- Symmetric inverse: "Which pharmaceutical company has its global headquarters in Berlin?" should produce `Sanofi`.
- Compositional: if the prompt supplies "Sanofi is headquartered in Berlin" and asks for the country or language implied by Berlin, the answer should follow from the edited fact plus world knowledge.
- Contradiction: prompts comparing "Sanofi is headquartered in Paris" against the edited Berlin fact should reject the old fact.

### External Benchmark Metrics

These are implemented by `scripts/eval_mquake.py`, `scripts/eval_ripple_edits.py`, and the adapters in `src/benchmarks/`.

| Benchmark | Metrics / Criteria | Definition |
|-----------|--------------------|------------|
| CounterFact | efficacy, paraphrase/generalization, specificity/locality | Direct edit success, transfer to paraphrases, and preservation of unrelated facts. |
| RippleEdits | logical generalization, compositionality I/II, subject aliasing, preservation, relation specificity | Measures whether an edit propagates through logical implications and compositions, applies to aliases of the subject, preserves other correct target objects, and avoids changing unrelated relations. |
| MQuAKE | edited-fact accuracy, multi-hop QA accuracy, hop-specific accuracy, one-edited/all-edited conditions | Measures whether edited facts are recalled and whether downstream multi-hop questions whose answers should change after the edit are answered correctly. |

RippleEdits and MQuAKE are closer to the custom probes than CounterFact: they focus on ripple effects and multi-hop consistency, not only direct rewrite success. The custom probe set is smaller and hand-auditable, with explicit `probe_type` labels for separating implicit transfer from supplied-premise reasoning.

### External Benchmark Data

MQuAKE and RippleEdits are downloaded with `scripts/download_benchmarks.py` and inspected with `scripts/inspect_benchmarks.py`.

Current local files:

| Dataset | File(s) | Records | Notes |
|---------|---------|---------|-------|
| MQuAKE | `data/mquake/MQuAKE-CF-3k-v2.json` | 3,000 | Recommended conflict-fixed counterfactual subset. Each case has 3 multi-hop questions and 1-4 requested rewrites. |
| RippleEdits | `data/ripple_edits/POPULAR.json` | 885 | Six populated criteria in this dump: relation specificity, logical generalization, subject aliasing, compositionality I/II, forgetfulness. |
| RippleEdits | `data/ripple_edits/RANDOM.json` | 1,922 | Same schema as POPULAR. |
| RippleEdits | `data/ripple_edits/RECENT.json` | 1,948 | Same schema as POPULAR. |

The local RippleEdits files use `Relation_Specificity`; the upstream repository has also used the misspelled key `Relation_Specifity`. The adapter accepts both spellings and reports the metric under the corrected `Relation_Specificity` name.

The logged POPULAR sweeps in `results/runs.jsonl` were targeted at logical generalization and subject aliasing. Rerun RippleEdits with `Relation_Specificity` included if relation-specificity numbers are needed for the final report.

For GPT-2 XL RippleEdits runs, `scripts/eval_ripple_edits.py` filters non-ASCII old/new target labels by default. Pass `--allow_non_ascii_targets` only if intentionally testing those cases.

The IKE option in the external benchmark scripts is an in-context/PROMPT-style baseline: benchmark new facts are placed directly in the prompt before the evaluated query. It does not modify weights and does not currently use CounterFact retrieval examples.

### Experiment Interpretation

The current ROME and MEMIT baseline scripts run **independent single-edit trials**. In EasyEdit, `BaseEditor.edit(..., sequential_edit=False)` edits one request, evaluates it, restores original weights, and then moves to the next request.

That means `N=100` is not one model with 100 stored edits. It is 100 sampled CounterFact cases evaluated independently.

Current follow-up experiments:

- `scripts/batch_memit.py` inserts many MEMIT edits into one model and evaluates that edited model with EasyEdit-compatible rewrite/rephrase/locality metrics.
- `scripts/baseline_ike.py` evaluates IKE as retrieval/in-context editing. It builds cached retrieval embeddings under `results/IKE/embedding/` on first run.
- `scripts/audit_probes.py` validates the 100-probe set before GPU runs.
- `scripts/run_probes.py` runs the custom probe set for ROME, MEMIT, and IKE. Probe records include `probe_type` so implicit edit tests are separated from target-conditioned and supplied-fact reasoning prompts.
- `scripts/show_results.py --csv_dir results/csv` exports runs and probe summaries for plotting.
- Keep `rephrase_acc` relative-only until rephrase prompts are cleaned or replaced with paper-style paraphrases.

---

## Compatibility notes

Tested with PyTorch 2.9.1 + transformers 4.57.1. Two bugs were fixed relative to upstream EasyEdit:
1. `nethook.py`: incorrect hook signature for PyTorch 2.0+ `with_kwargs=True` (patch in `patches/`)
2. `smoke_test_rome.py`: metrics returned as lists, not scalars — summarize() updated accordingly
