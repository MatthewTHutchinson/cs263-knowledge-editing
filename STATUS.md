# Project Status

Working title: *When Surgical Edits Leak: Logical Consistency and Ripple Effects Across Knowledge Editing Methods*

Quick reference for current state, what's done, what's next. Update this whenever a milestone completes. Daily narrative goes in `NOTES.md`; this is the high-level map.

---

## Methods in scope

| Method | Type | Owner | Status |
|--------|------|-------|--------|
| ROME | Parameter-based (rank-one) | Matthew | Baseline running |
| MEMIT | Parameter-based (batch) | Matthew | Not started |
| IKE | Retrieval / in-context | Matthew | Not started |

---

## Datasets

| Dataset | Purpose | Location | Status |
|---------|---------|----------|--------|
| CounterFact (EasyEdit) | Baseline eval: efficacy, paraphrase, specificity | `data/counterfact/counterfact-edit.json` | Downloaded (10K records) |
| RippleEdits | Ripple effect eval | TBD | Not downloaded |
| MQuAKE | Multi-hop reasoning eval | TBD | Not downloaded |
| Diagnostic probe set | Novel contribution — logical consistency | `data/probes/` (TBD) | Design in progress |

---

## Pipeline status

| Step | Status | Notes |
|------|--------|-------|
| Environment setup | Done | conda `cs263-project`, GCP T4 |
| EasyEdit + ROME running | Done | Fixed 2 compatibility bugs (see commit 2867c41) |
| Smoke test (5 edits) | Passed | rewrite=1.00, rephrase=0.93, locality=0.70 |
| ROME 100-edit baseline | Done | rewrite=1.00, rephrase=0.54, locality=0.79 |
| ROME vs. paper validation | Partial | rewrite/locality ✓, rephrase gap under investigation |
| Rephrase failure inspection | **Next** | Check if EasyEdit rephrase prompts are bad |
| MEMIT baseline | Not started | |
| IKE baseline | Not started | |
| RippleEdits download + eval | Not started | |
| MQuAKE download + eval | Not started | |
| Probe set design | In progress | See probe design notes below |
| Probe set implementation | Not started | |
| Results figures | Not started | |

---

## Key results log

See `results/runs.jsonl` for machine-readable records. Summary:

| Date | Method | Dataset | N | Rewrite | Rephrase | Locality |
|------|--------|---------|---|---------|----------|----------|
| 2026-05-02 | ROME | CounterFact-smoke | 5 | 1.000 | 0.933 | — |
| 2026-05-03 | ROME | CounterFact | 100 | 1.000 | 0.540 | 0.790 |

**Paper targets (ROME, GPT-2 XL, CounterFact):** rewrite ~99.6%, rephrase ~94.8%, locality ~72.2%

Rephrase gap (~40 points) likely due to poor-quality rephrase prompts in EasyEdit's dataset version, not a model failure. Under investigation.

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
