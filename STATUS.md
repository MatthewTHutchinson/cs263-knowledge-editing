# Project Status

Working title: *When Surgical Edits Leak: Logical Consistency and Ripple Effects Across Knowledge Editing Methods*

Quick reference for current state, what's done, what's next. Update this whenever a milestone completes. Daily narrative goes in `NOTES.md`; this is the high-level map.

---

## Methods in scope

| Method | Type | Owner | Status |
|--------|------|-------|--------|
| ROME | Parameter-based (rank-one) | Matthew | Baseline done ✓ |
| MEMIT | Parameter-based (batch/mass edit) | Matthew | Single-edit baseline running; `batch_memit.py` written, not yet run |
| IKE | Retrieval / in-context | Matthew | Scaffold written (`baseline_ike.py`); not yet run |

---

## Datasets

| Dataset | Purpose | Location | Status |
|---------|---------|----------|--------|
| CounterFact (EasyEdit) | Baseline eval: efficacy, paraphrase, specificity | `data/counterfact/counterfact-edit.json` | Downloaded (10K records) |
| RippleEdits | Ripple effect eval | TBD | Not downloaded |
| MQuAKE | Multi-hop reasoning eval | TBD | Not downloaded |
| Diagnostic probe set | Novel contribution — logical consistency | `src/probes/probe_set.py` | 37 probes written (5 categories × 5 edit cases) |

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
| MEMIT single-edit baseline | Running on GCP | `scripts/baseline_memit.py`; covariance cache takes ~3–4 h on T4 (not 45 min) |
| MEMIT true batch/mass-edit eval | Ready to run | `scripts/batch_memit.py` written; run after single-edit baseline completes |
| IKE baseline | Scaffold ready | `scripts/baseline_ike.py` written; run after MEMIT |
| RippleEdits download + eval | Not started | |
| MQuAKE download + eval | Not started | |
| Probe set design | Done | 37 probes across 5 categories in `src/probes/probe_set.py` |
| Probe set evaluation | Ready to run | `scripts/run_probes.py` written; run after baselines complete |
| Results summarization | Done | `scripts/show_results.py` updated with comparison table, batch sweep, probe summary, ASCII plot |

---

## Key results log

See `results/runs.jsonl` for machine-readable records. Summary:

| Date | Method | Dataset | N | Rewrite | Rephrase | Locality |
|------|--------|---------|---|---------|----------|----------|
| 2026-05-02 | ROME | CounterFact-smoke | 5 | 1.000 | 0.933 | — |
| 2026-05-03 | ROME | CounterFact | 100 | 1.000 | 0.540 | 0.790 |

**Paper targets (ROME, GPT-2 XL, CounterFact):** rewrite ~99.6%, rephrase ~94.8%, locality ~72.2%

Rephrase gap (~40 points) is explained by poor-quality rephrase prompts in EasyEdit's dataset (relation mismatches, garbage text, indirect prompts — not actual paraphrases). The original ROME CounterFact uses curated `paraphrase_prompts` which would give paper-comparable numbers, but this conversion is deferred. **Decision: treat rephrase_acc as a relative comparison across ROME/MEMIT/IKE only — do not compare absolute rephrase numbers to the paper.** Rewrite and locality are paper-comparable and sufficient to trust the pipeline.

Original ROME repo cross-validation also deferred — rewrite (1.000) and locality (0.790) already confirm EasyEdit's ROME is faithful. Revisit only if MEMIT/IKE numbers look anomalous.

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
| MEMIT batch/mass edit | Insert many facts into one model | Main intended MEMIT setting |
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

### Implementation plan
- ~10 probes per category × 5 categories = ~50 probes
- Hand-curated against a subset of CounterFact edits (pick 10–15 representative edit cases)
- Run as post-edit queries against the already-edited model (separate from EasyEdit's eval loop)
- Script: `scripts/run_probes.py` (not yet written)
- Results appended to `results/runs.jsonl` with `dataset: "probes"`

---

## Compatibility notes
- **PyTorch 2.9.1 + transformers 4.57.1**: Fixed nethook.py bug (patch in `patches/`). Apply with:
  ```
  cd external/EasyEdit && patch -p1 < ../../patches/0001-fix-nethook-pytorch29-with_kwargs-signature.patch
  ```
- **ROME stats cache**: First run computes Wikipedia covariance (~20–40 min). Cached to `data/stats/` (gitignored — recomputes on fresh clone).

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
# data/stats/ will recompute on first ROME run (~30 min)
```
