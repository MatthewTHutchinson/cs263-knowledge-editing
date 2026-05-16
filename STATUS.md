# Project Status

Working title: *Beyond Rewrite Accuracy: Testing Logical Consistency in Knowledge Editing*

Quick reference for current state, what's done, what's next. Update this whenever a milestone completes. Daily narrative goes in `NOTES.md`; this is the high-level map.

---

## Methods in scope

| Method | Type | Owner | Status |
|--------|------|-------|--------|
| ROME | Parameter-based (rank-one) | Matthew | Baseline done ✓ |
| MEMIT | Parameter-based (batch/mass edit) | Matthew | Single-edit baseline done; true batch 10/50/100 done |
| IKE | Retrieval / in-context | Matthew | 5/50/100-edit baselines complete and recorded locally |

---

## Datasets

| Dataset | Purpose | Location | Status |
|---------|---------|----------|--------|
| CounterFact (EasyEdit) | Baseline eval: efficacy, paraphrase, specificity | `data/counterfact/counterfact-edit.json` | Downloaded (10K records) |
| RippleEdits | Ripple effect eval | `data/ripple_edits/{POPULAR,RANDOM,RECENT}.json` | Downloaded; adapter/inspector added |
| MQuAKE | Multi-hop reasoning eval | `data/mquake/MQuAKE-CF-3k-v2.json` | Downloaded; adapter/inspector added |
| Diagnostic probe set | Novel contribution — logical consistency | `src/probes/probe_set.py` | 100 probes written and validator-clean (5 categories × 5 edit cases; includes `probe_type` labels) |

---

## Pipeline status

| Step | Status | Notes |
|------|--------|-------|
| Environment setup | Done | conda `cs263-project`, GCP T4 |
| EasyEdit + ROME running | Done | Fixed 2 compatibility bugs (see commit 2867c41) |
| Smoke test (5 edits) | Passed | rewrite=1.00, rephrase=0.93, locality=0.70 |
| ROME 100-edit baseline | Done | rewrite=1.00, rephrase=0.54, locality=0.79 |
| ROME vs. paper validation | Partial | rewrite/locality ✓, rephrase gap under investigation |
| Rephrase failure inspection | Done | `scripts/inspect_rephrase_failures.py`; 34/46 failures have prompt-quality flags |
| MEMIT single-edit baseline | Done | `scripts/baseline_memit.py`; covariance cache is warm for layers 13-17 |
| MEMIT true batch/mass-edit eval | Done | Batch-10, 50, and 100 runs confirm `Writing N key/value pair(s)` and cached covariance reuse |
| IKE baseline | Done | `scripts/baseline_ike.py` builds cached retrieval embeddings before EasyEdit IKE evaluation; local repo records 5, 50, and 100-edit IKE runs |
| RippleEdits download + eval | Small sweep done | POPULAR targeted logical-generalization/subject-aliasing sweeps complete for ROME n=10 and IKE n=25; local JSONs use `Relation_Specificity`, and the adapter now also accepts upstream's legacy `Relation_Specifity` spelling |
| MQuAKE download + eval | Small sweep done | IKE all-edit n=25, ROME one-edit n=10, and MEMIT all-edit n=10 complete with pre/post/delta logging |
| Equal-sample external sweeps | In progress | Launched 2026-05-16 in tmux session `external_sweeps`; queued n=25 then n=100 MQuAKE/RippleEdits runs for ROME, MEMIT, and IKE with RippleEdits `Relation_Specificity,Logical_Generalization,Subject_Aliasing` |
| Probe set design | Done | 100 probes across 5 categories in `src/probes/probe_set.py`; `probe_type` separates implicit, target-conditioned, and supplied-fact prompts |
| Probe set evaluation | Done for ROME/MEMIT/IKE | ROME, MEMIT, and IKE each evaluated on 100 probes on the GCP T4 VM on 2026-05-11; results written to `results/probe_results.jsonl` |
| Probe validation | Done | `scripts/audit_probes.py --min_total 100 --strict` passed locally on 2026-05-11 |
| Local tests | Done | `python -m unittest discover -s tests` passed locally on 2026-05-11; tests cover MEMIT batch metric semantics and IKE embedding-cache logic |
| Results summarization | Done | `scripts/show_results.py` updated with comparison table, batch sweep, probe summary by category/type, ASCII plot, CSV export |
| External benchmark adapters | Done | `src/benchmarks/`, download/inspect/eval scripts; unit tests cover MQuAKE/RippleEdits parsing and answer matching |
| External benchmark result display | Done | `scripts/show_results.py` separates MQuAKE/RippleEdits metrics from CounterFact baseline metrics and exports CSVs |

---

## Key results log

See `results/runs.jsonl` for machine-readable records. Summary:

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

ROME/MEMIT/IKE probe evaluation completed on the GCP T4 VM on 2026-05-11. IKE scored 50% post-edit pass rate overall, up from 36% pre-edit; ROME and MEMIT each scored 64% post-edit, up from 36% pre-edit. The main improvement for ROME/MEMIT is logical negation: 88% post-edit for both methods, up from 0% pre-edit. Summaries are available with `python scripts/show_results.py --probes`, and CSV exports live under `results/csv/`.

MQuAKE/RippleEdits data prep and small external sweeps completed locally on 2026-05-11:

- MQuAKE: `MQuAKE-CF-3k-v2.json`, 3,000 records; edit-count distribution is 1 edit: 1,073, 2 edits: 1,046, 3 edits: 568, 4 edits: 313.
- RippleEdits: POPULAR 885 records, RANDOM 1,922 records, RECENT 1,948 records.
- RippleEdits populated criteria in the downloaded files are relation specificity, logical generalization, subject aliasing, compositionality I, compositionality II, and forgetfulness. The local files use `Relation_Specificity`; the adapter also accepts upstream's legacy `Relation_Specifity` spelling.
- The logged RippleEdits POPULAR sweeps were targeted at logical generalization and subject aliasing. Rerun RippleEdits with `Relation_Specificity` included before reporting relation-specificity scores.
- MQuAKE small sweeps: IKE all-edit n=25 edited_fact_acc=0.910, multihop_acc=0.453, delta_multihop_acc=+0.347; ROME one-edit n=10 edited_fact_acc=0.440, multihop_acc=0.100, delta_multihop_acc=+0.033; MEMIT all-edit n=10 edited_fact_acc=0.680, multihop_acc=0.033, delta_multihop_acc=-0.033.
- RippleEdits POPULAR targeted logical-generalization/subject-aliasing sweeps: ROME n=10 overall_acc=0.160, delta_overall_acc=+0.136, Subject_Aliasing_acc=0.375, Logical_Generalization_acc=0.000; IKE n=25 overall_acc=0.347, delta_overall_acc=+0.299, Subject_Aliasing_acc=0.692, Logical_Generalization_acc=0.237.
- Earlier n=1 smoke runs remain in `results/runs.jsonl` as pipeline checks. Interpret the n=10/n=25 sweeps as the main external benchmark signal for the current report.
- In the external benchmark scripts, IKE is an in-context/PROMPT-style baseline using benchmark new facts in the prompt, not CounterFact retrieval. For GPT-2 XL RippleEdits runs, the evaluator filters non-ASCII old/new target labels by default.

Equal-sample follow-up sweeps were launched on 2026-05-16 in tmux session `external_sweeps`, with logs at `logs/external_equal_sweeps_20260516.log`. The queue runs n=25 first as a fixed-adapter check, then n=100 as the main next comparison target:

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

Do not replace the logged small-sweep table values with these equal-sample results until the tmux run finishes and `scripts/show_results.py --csv_dir results/csv` has been rerun.

**Paper targets (ROME, GPT-2 XL, CounterFact):** rewrite ~99.6%, rephrase ~94.8%, locality ~72.2%

Rephrase gap (~40 points) is explained by poor-quality rephrase prompts in EasyEdit's dataset (relation mismatches, garbage text, indirect prompts — not actual paraphrases). The original ROME CounterFact uses curated `paraphrase_prompts` which would give paper-comparable numbers, but this conversion is deferred. **Decision: treat rephrase_acc as a relative comparison across ROME/MEMIT/IKE only — do not compare absolute rephrase numbers to the paper.** Rewrite and locality are paper-comparable and sufficient to trust the pipeline.

Original ROME repo cross-validation also deferred — rewrite (1.000) and locality (0.790) already confirm EasyEdit's ROME is faithful. Revisit only if MEMIT/IKE numbers look anomalous.

---

## Metrics Reference

### Core EasyEdit metrics

| Metric | Scope | Computation | Notes |
|--------|-------|-------------|-------|
| `rewrite_acc` | CounterFact rewrite prompt | Exact-match token accuracy for `target_new` after editing. | Main direct edit-success metric. |
| `rephrase_acc` | CounterFact rephrase prompt | Exact-match token accuracy for `target_new` on `rephrase_prompt`. | Relative-only in this project because EasyEdit's rephrase prompts are noisy. |
| `locality_acc` | CounterFact locality prompt | Agreement between post-edit and pre-edit predictions on locality prompts. | Measures preservation, not necessarily correctness against `locality_ground_truth`. |
| `paper_target` | Run metadata | Published reference value for the corresponding method/model/dataset when available. | Use only when prompt/dataset formatting is comparable. |

Important distinction: in the single-edit ROME/MEMIT baselines, `n_samples=100` means 100 independent edit/evaluate/restore trials. In `MEMIT-batch`, `n_samples=100` means 100 facts inserted into one model update before evaluation. IKE is non-parametric, so `n_samples` means the number of in-context edit records evaluated, not stored weights.

### Probe metrics

| Metric / Dimension | Computation | Interpretation |
|--------------------|-------------|----------------|
| Per-probe pass | `expected_first_token` appears in the generated first token, or `expected_contains` appears in a short greedy generation. | Binary outcome for one diagnostic query. |
| `pre_pass_rate` | Mean pass rate before editing. | Captures base-model ability and prompt leakage. |
| `post_pass_rate` | Mean pass rate after editing. | Captures edited behavior. |
| `delta_pass_rate` | `post_pass_rate - pre_pass_rate`. | Best summary of improvement caused by the edit. |
| Category pass rate | Mean pass rate within `logical_negation`, `symmetric_inverse`, `compositional`, `contradiction`, or `chain_of_thought`. | Shows which consistency property succeeds or fails. |
| Type pass rate | Mean pass rate within `implicit_edit`, `target_conditioned`, or `supplied_fact_reasoning`. | Separates strong transfer tests from prompts that include the target or supplied fact. |

Probe categories:

- `logical_negation`: the model should answer with the new value or reject the old value under a changed surface form.
- `symmetric_inverse`: an edit in the subject-to-object direction should support an inverse object-to-subject query.
- `compositional`: the edited fact should combine with another known fact to produce an implied answer.
- `contradiction`: the model should not simultaneously affirm old and new incompatible facts.
- `chain_of_thought`: a short reasoning chain should remain consistent with the edited fact.

Probe types:

- `implicit_edit`: strongest evidence of transfer; the prompt does not mention the new target.
- `target_conditioned`: the prompt mentions the target value or asks a forced-choice question.
- `supplied_fact_reasoning`: the prompt states the edited fact and measures reasoning from that supplied premise; report separately from implicit transfer.

Example interpretation: for the Sanofi headquarters edit (`Paris` to `Berlin`), a logical-negation probe should complete a new headquarters prompt with `Berlin`, a symmetric-inverse probe should answer that the company headquartered in Berlin is `Sanofi`, and a compositional probe should use Berlin plus world knowledge to infer `Germany` or `German`.

### External benchmark metrics

| Dataset | Metric family | What it measures |
|---------|---------------|------------------|
| CounterFact | efficacy/rewrite, paraphrase/generalization, locality/specificity | Direct edit success, surface-form transfer, and preservation of unrelated facts. |
| RippleEdits | logical generalization, compositionality I/II, subject aliasing, preservation, relation specificity | Whether edits ripple through logical and compositional consequences while preserving aliases, other true objects, and unrelated relations. |
| MQuAKE | edited-fact accuracy, multi-hop QA accuracy, hop-specific accuracy, one-edited/all-edited settings | Whether edited facts are recalled and whether entailed multi-hop questions change correctly after one or more edits. |

---

## Experimental design decisions

### Single-edit vs. mass-edit conditions

The current ROME and MEMIT baseline scripts use EasyEdit `BaseEditor.edit(...)` with `sequential_edit=False`. EasyEdit evaluates each request independently and restores the original model afterward. Therefore:

- `ROME, N=100` means 100 independent single-edit trials.
- Current `MEMIT, N=100` also means 100 independent single-edit trials.
- Current MEMIT logs saying `Writing 1 key/value pair(s)` confirm it is not performing one 100-edit model update.

This is a fair sanity comparison for single-edit behavior, but it does not test MEMIT's main scientific claim: reliable mass editing.

Planned conditions:

| Condition | Purpose | Interpretation |
|-----------|---------|----------------|
| ROME single-edit | Validate parametric single-fact editing | Main intended ROME setting |
| MEMIT single-edit | Sanity check against ROME on same records | Useful but not MEMIT's main advantage |
| MEMIT batch/mass edit | Insert many facts into one model | Main intended MEMIT setting; locality is measured as post-edit preservation of pre-edit locality predictions |
| ROME sequential/cumulative stress | Optional stress test | Not a primary fair baseline for mass editing |
| IKE single-edit retrieval | Non-parametric baseline | Base model + one retrieved/in-context edit |
| IKE many-edit context/retrieval | Context interference test | Not a weight-edit batch; report separately |

### IKE in a "batch" context

IKE does not modify model weights. Its edited behavior comes from retrieved examples or in-context demonstrations supplied at inference time. So a MEMIT-style batch edit does not map cleanly onto IKE.

Scientifically sensible IKE comparisons:

- **Same-record single-edit eval**: for each CounterFact edit, provide the relevant IKE example/context and evaluate rewrite, rephrase, locality, and probes.
- **Many-edit retrieval eval**: build an edit memory with many edited facts, retrieve the relevant fact at inference time, and measure whether retrieval/context selection succeeds.
- **Many-edit context stress eval**: place multiple edited facts in the prompt and measure interference as the number of facts grows.

Do not describe IKE as creating a persistent 100-edit model. It is a non-parametric inference-time method, so results should be framed as retrieval/context robustness rather than stored-weight capacity.

### Rephrase prompts and full runs

EasyEdit CounterFact rephrase prompts are noisy enough that `rephrase_acc` is relative-only for now. If final claims need paper-style generalization numbers, create a cleaned rephrase prompt set or recover the original paper-style paraphrases.

After scripts are stable, consider a full CounterFact run. The paper-style scale is roughly 2500 cases, but for MEMIT mass editing it may be more informative to run batch-size sweeps first, such as 10, 100, and 1000 simultaneous edits if compute allows.

---

## Probe set design

### Motivation
EasyEdit's built-in metrics (efficacy, paraphrase, locality) measure whether the *target fact* was edited and whether *unrelated facts* were preserved. They do **not** test whether the edit is *logically consistent* or whether related facts (implications, inverses, compositions) updated correctly.

### Probe categories

**1. Logical negation probes**
After editing X's property P from A → B, the model should reject A.
- Format: `"Is [subject]'s [relation] still [old_value]?"` → expected: No / [new_value]
- Targets all three methods equally.

**2. Symmetric relation probes**
Some relations have inverses. After editing "X [relation] Y", query the inverse.
- Example: edit "Sanofi HQ is in Berlin" → query "Berlin is home to [what company]?" → should include Sanofi
- ROME/MEMIT are most likely to fail: they edit one MLP layer, inverse relation may not be stored in the same weights.

**3. Compositional / transitive probes**
After editing F1 that implies F2, test F2.
- Example: edit "Theo Walcott plays basketball" → query "Theo Walcott's sport involves a hoop" → should be True
- Example: edit "Lil Wayne is signed to Interscope" → query "Lil Wayne's label is owned by Universal Music Group" → should now be True (Interscope is UMG-owned)
- IKE should fail these when the chain is not explicit in the context window.

**4. Logical contradiction probes**
After editing, check whether the model simultaneously holds the old and new value.
- Format: ask the model to choose between old and new value; or ask a yes/no about the old value.
- All methods may fail — the edit takes effect on the direct prompt but base-model priors reassert on indirect queries.

**5. Chain-of-thought probes**
Prompt the model to *explain its reasoning* about the edited fact.
- Force multi-step reasoning: "Let's think step by step: [subject] works at [company]. [Company] is headquartered in [city]. Therefore [subject] works in..."
- Reveals whether the edit is "surface-level" (first token correct) vs. deeply integrated.
- Expected: IKE will often produce the right answer without reasoning consistency; ROME may contradict itself mid-chain.

### Implementation status
- 100 probes across five categories, hand-curated around the five smoke-test edit cases.
- Probe records include `probe_type`:
  - `implicit_edit`: the prompt does not state the new fact and should test whether the edit transfers to a new surface form.
  - `target_conditioned`: the prompt conditions on the edited target value but does not directly assert the full subject-target fact; useful for inverse and forced-choice tests.
  - `supplied_fact_reasoning`: the prompt states the edited fact and tests whether the model can reason from it. These should be analyzed separately because the base model may pass them pre-edit.
- `scripts/run_probes.py` currently supports ROME, MEMIT, and IKE, restores weights after each parametric edit, and writes records to `results/probe_results.jsonl`.
- `scripts/show_results.py --probes` summarizes probe results by category and by `probe_type`.
- `scripts/audit_probes.py` checks unique IDs, valid labels, coverage, expected answers, and target leakage in `implicit_edit` prompts.

---

## Compatibility notes
- **PyTorch 2.9.1 + transformers 4.57.1**: Fixed nethook.py bug (patch in `patches/`). Apply with:
  ```
  cd external/EasyEdit && patch -p1 < ../../patches/0001-fix-nethook-pytorch29-with_kwargs-signature.patch
  ```
- **ROME/MEMIT stats cache**: First run computes Wikipedia covariance and caches it to `data/stats/` (gitignored). MEMIT's five-layer GPT-2 XL cache can take several hours on T4.

## Setup from scratch (new machine / fresh clone)
```bash
git clone <repo>
cd cs263-knowledge-editing
git clone https://github.com/zjunlp/EasyEdit external/EasyEdit
cd external/EasyEdit && patch -p1 < ../../patches/0001-fix-nethook-pytorch29-with_kwargs-signature.patch && cd ../..
conda create -n cs263-project python=3.10
conda activate cs263-project
pip install -r external/EasyEdit/requirements.txt
# data/counterfact/ is in the repo — no download needed
# data/stats/ will recompute on first ROME/MEMIT run
```

## 2026-05-11 GitHub state

External benchmark sweep records were committed at `3878d54`. The prior remote commit `2f1d6a6` organized the Overleaf midterm package; `3878d54` adds the MQuAKE/RippleEdits n=10/n=25 sweep detail JSONs and appends their summary rows to `results/runs.jsonl`.

Canonical tracked result files:

- `results/runs.jsonl`
- `results/probe_results.jsonl`
- `results/benchmark_details/*.json`

Generated CSV exports under `results/csv/` are intentionally gitignored. Regenerate them with:

```bash
python scripts/show_results.py --csv_dir results/csv
```

## 2026-05-10 VM Transition Checklist

### GitHub state

As of the 2026-05-10 VM transition check, the code, configs, tracked results summary, tests, and patches needed to recreate the project were in GitHub. The current GitHub state is newer; see the 2026-05-11 GitHub state section above.

Tracked in GitHub:

- experiment scripts under `scripts/`
- configs under `configs/`
- patches under `patches/`
- CounterFact data under `data/counterfact/`
- stable MEMIT/ROME covariance `.npz` files under `data/stats/gpt2-xl/wikipedia_stats/`, via Git LFS
- structured result summary at `results/runs.jsonl`
- project docs and notes

Intentionally not tracked:

- other `data/stats/` generated files outside the stable GPT-2 XL `.npz` cache
- `logs/`
- `results/IKE/embedding/`
- `external/EasyEdit/`
- model/download caches and conda environments

Before deleting the old VM, make sure the backup archive has either been downloaded locally or copied to GCS. This remains useful even with Git LFS because it includes logs and transition state:

```text
/home/matthewthutchinson1/cs263-memit-preserve-20260510.tar.gz
sha256 f15b0cd7f85bf9b597572476f083f6151358dcbfe4474e99ca097f6471b3c73b
```

### New VM recommendation

For long MEMIT/probe work, prefer a regular on-demand GPU VM rather than a Spot/preemptible VM. Spot/preemptible is cheaper, but GCP can terminate it with short notice, which is exactly the failure mode that wastes long MEMIT cache jobs. On-demand costs more while running, but it avoids preemption and is the safer default until the remaining experiments are complete.

Recommended shape:

```text
Zone: us-central1-a or another zone with T4 capacity
GPU: 1 x NVIDIA T4
Machine: n1-standard-4 or g2-standard-* if using L4 instead
Boot disk: Ubuntu 22.04 LTS, 100-200 GB balanced persistent disk
Provisioning model: Standard/on-demand, not Spot
Automatic restart: on
Maintenance behavior: terminate is normal for GPU VMs
```

Drawbacks of non-preemptible/on-demand:

- higher hourly cost than Spot/preemptible
- GPU VMs still cannot live-migrate during host maintenance, so a rare maintenance event can still stop the VM
- idle cost accumulates quickly; stop the VM manually when not running jobs
- GPU capacity can still be scarce by zone, so keeping a working VM stopped is often useful

### Restore on the new VM

After cloning the repo and installing EasyEdit, pull the LFS cache or restore the archive from the repo root:

```bash
git lfs install
git lfs pull
tar -xzf ~/cs263-memit-preserve-20260510.tar.gz
sha256sum ~/cs263-memit-preserve-20260510.tar.gz
find data/stats/gpt2-xl/wikipedia_stats -maxdepth 1 -type f -name '*.npz' -printf '%f %s bytes\n' | sort
python scripts/show_results.py --all
```

Expected MEMIT stats files:

```text
data/stats/gpt2-xl/wikipedia_stats/transformer.h.13.mlp.c_proj_float32_mom2_100000.npz
data/stats/gpt2-xl/wikipedia_stats/transformer.h.14.mlp.c_proj_float32_mom2_100000.npz
data/stats/gpt2-xl/wikipedia_stats/transformer.h.15.mlp.c_proj_float32_mom2_100000.npz
data/stats/gpt2-xl/wikipedia_stats/transformer.h.16.mlp.c_proj_float32_mom2_100000.npz
data/stats/gpt2-xl/wikipedia_stats/transformer.h.17.mlp.c_proj_float32_mom2_100000.npz
```

### Long-run hygiene

Run GPU jobs inside `tmux` and write logs under `logs/`. For MEMIT covariance jobs, use:

```bash
tmux new-session -d -s memit scripts/run_memit_checkpointed.sh
tail -f "$(cat logs/baseline_memit_latest.path)"
```

For future one-off runs, prefer explicit log files:

```bash
mkdir -p logs
tmux new-session -d -s probes 'conda activate cs263-project && python scripts/run_probes.py --method MEMIT 2>&1 | tee logs/probes_memit_$(date +%Y%m%d_%H%M%S).log'
```
