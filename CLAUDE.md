# CS 263 Final Project — Knowledge Editing Comparison

Context for Claude Code. Read this first every session.

## What this project is

A comparative study of three knowledge editing methods (ROME, MEMIT, IKE) on GPT-2 XL (and optionally GPT-J), evaluated on logical consistency and ripple effects. The novel contribution is a custom diagnostic probe set that targets the known structural weaknesses of each method family.

Working title: *When Surgical Edits Leak: A Comparative Study of Logical Consistency and Ripple Effects Across Knowledge Editing Methods*

See `docs/CS263_Project_Planning_Report.docx` in this repo for the full planning report.

## Team and ownership

- **Matthew Hutchinson** (repo owner, the human you're talking to): owns all implementation — ROME/MEMIT/IKE baselines, eval pipeline, diagnostic probe set, figures, writeup.
- **Corey Shen** and **Nathan Wei**: teammates on the project but not contributing code. Matthew is doing all the work.

When Claude Code generates code, everything is in scope.

## Stack decisions (already made — do not relitigate)

- **Framework**: EasyEdit (Wang et al., ACL 2024) as the unified wrapper. Repo: https://github.com/zjunlp/EasyEdit
- **Methods**: ROME, MEMIT, IKE. Two parameter-based (ROME, MEMIT) plus one retrieval-based (IKE).
- **Models**: GPT-2 XL (1.5B) primary; GPT-J (6B) as an optional scale check.
- **Datasets**: CounterFact (ships with ROME and EasyEdit), RippleEdits (Cohen et al., 2024), MQUAKE (Zhong et al., EMNLP 2023).
- **Compute**: Preemptible T4 GPU on Google Cloud as the default. A100 only for GPT-J scale checks. Per-member $50 GCP credits.

## Startup sequence (Week 4–5)

Day 1 — environment:
1. Clone EasyEdit into `external/EasyEdit/` as a submodule or separate checkout.
2. Clone the original ROME repo into `external/rome/` for cross-validation only.
3. Create a conda env matching EasyEdit's `requirements.txt`. Stick with their transformers version.
4. Run EasyEdit's README example end-to-end on 5 CounterFact edits to confirm the environment works.

Day 2 — baseline validation:
5. Run EasyEdit's ROME on GPT-2 XL against 100 CounterFact edits. Record efficacy, generalization (paraphrase), specificity (locality).
6. Compare to the ROME paper's published numbers (Meng et al., NeurIPS 2022, Table 1). Within ~2 points of paper = pipeline trusted.
7. Run the same 100 edits through the original ROME code as a second sanity check. Confirm EasyEdit is faithful.

Day 3 — bring up MEMIT and IKE:
8. Run EasyEdit's MEMIT on the same 100 edits as a single-edit sanity baseline and cache warmup.
9. Run EasyEdit's IKE (with `use_icl_examples=True`) on the same 100 edits, compare to Zheng et al. numbers.
10. At this point: three-method pipeline with published-paper-comparable baselines. Log results in NOTES.md.

## Novel contribution (the actually interesting part)

A ~50-item hand-curated diagnostic probe set with three probe types:

1. **Contradiction probes**: after an edit, check whether the model holds logically incompatible beliefs. E.g. edit "Einstein was a chemist" → query "Einstein won the Nobel Prize in Physics" (should now contradict).
2. **Method-sensitivity probes**: target what each family structurally expects to break. For ROME/MEMIT, probe implications that pass through layers not touched by the edit. For IKE, probe queries that don't naturally reference the edited fact (IKE should regress to base-model behavior).
3. **Chain-of-thought probes**: prompt the model to explain its reasoning about the edited fact, check whether the reasoning chain supports the edit or reveals contradictions.

**Design note**: these probes do NOT cleanly map to EasyEdit's `locality_inputs` or `portability_inputs` slots. Run EasyEdit's built-in eval first, then run probes as a separate post-edit script against the already-edited model. Do not shove contradiction probes into `locality_inputs`.

Current probe implementation:
- `src/probes/probe_set.py` has 100 probes around the five smoke-test edit cases.
- Each probe has `probe_type`:
  - `implicit_edit`: prompt does not state the new fact.
  - `target_conditioned`: prompt conditions on the edited target value or forced choice.
  - `supplied_fact_reasoning`: prompt states the edited fact and tests reasoning from it.
- Analyze `supplied_fact_reasoning` separately from implicit edit transfer because the base model may pass by following the supplied prompt.
- Run `python scripts/audit_probes.py --min_total 100 --strict` before launching GPU probe jobs.

Metric interpretation:
- `rewrite_acc`: exact-match token accuracy for the new target on the original edit prompt.
- `rephrase_acc`: exact-match token accuracy for the new target on a rephrased prompt; relative-only here due to noisy EasyEdit CounterFact rephrase prompts.
- `locality_acc`: preservation score, computed by comparing post-edit locality predictions to pre-edit locality predictions.
- Probe `post_pass_rate`: pass rate after editing.
- Probe `pre_pass_rate`: base-model pass rate before editing.
- Probe `delta_pass_rate`: post minus pre; prefer this for claims about edit-induced improvement.
- RippleEdits metrics should be framed as logical generalization, compositionality, subject aliasing, preservation, and relation specificity.
- MQuAKE metrics should be framed as edited-fact accuracy and multi-hop QA accuracy, optionally broken down by hop count and one-edited/all-edited settings.

## Repo conventions

- Code in `src/`. Subdivide by role: `src/rome/`, `src/probes/`, `src/eval/`, `src/utils/`.
- Experiments and scripts in `scripts/` (runnable Python).
- Outputs and results in `results/` (gitignored except README stubs).
- Configs in `configs/` (copy YAMLs from EasyEdit here so they're versioned with our runs).
- Raw data and model weights in `data/` (gitignored).
- Human notes in `NOTES.md` at repo root.

## Style

- Python 3.10+. Type hints where they help, not religiously.
- Prefer scripts over notebooks for anything that gets re-run. Notebooks are fine for exploration.
- When running experiments, always log: method, model, dataset, n_samples, seed, timestamp, metrics. Append to `results/runs.jsonl`.
- For local verification, prefer `python -m unittest discover -s tests`, `python scripts/audit_probes.py --min_total 100 --strict`, and `python scripts/show_results.py --csv_dir /private/tmp/cs263_csv`; these do not load GPT-2 XL.
- Commit messages: short, imperative. "Add ROME baseline script", not "Added a script that runs ROME as a baseline."

## Gotchas to remember

- Paper numbers are almost never exactly reproducible (seeds, tokenizer versions, prompt formatting). ±2 points is fine.
- **rephrase_acc is relative-only**: EasyEdit's CounterFact rephrase prompts are poor quality (relation mismatches, garbage text). Do not compare rephrase_acc absolute values to the original papers. Use it only to compare ROME vs MEMIT vs IKE against each other.
- **No original ROME repo cross-validation needed**: rewrite=1.000 and locality=0.790 confirm EasyEdit's ROME is faithful to the paper.
- CounterFact full eval is ~2500 edits. Each ROME edit takes time on T4. Budget overnight for a full run; use 100-sample subsets for iteration.
- **Single-edit vs. mass-edit distinction**: current `baseline_rome.py` and `baseline_memit.py` use `BaseEditor.edit(..., sequential_edit=False)`, which evaluates each request independently and restores weights. `N=100` means 100 independent single-edit trials, not one 100-edit model.
- **MEMIT true batch still needs to be run**: MEMIT's main claim is mass editing. `scripts/batch_memit.py` is the dedicated batch/mass-edit script, but do not claim MEMIT's intended advantage until it has run successfully on the GCP T4.
- **MEMIT batch metrics**: `scripts/batch_memit.py` must keep using EasyEdit-compatible evaluation. Rewrite/rephrase should come from `compute_edit_quality(...)`; locality should compare post-edit locality predictions to pre-edit model predictions, not directly to `locality_ground_truth`.
- **IKE doesn't modify weights**. The "edited model" at inference time is base model + retrieved/in-context examples. Probes need to be run through IKE's inference wrapper, not against a saved checkpoint.
- **IKE embedding cache**: EasyEdit IKE expects precomputed retrieval embeddings under `results/IKE/embedding/`. `scripts/baseline_ike.py` builds them via `encode_ike_facts(...)` before calling `BaseEditor.edit(...)`.
- **IKE batch framing**: do not call IKE a persistent batch edit. A fair many-edit IKE experiment means many edited facts in memory/context, then measuring retrieval accuracy, context interference, and robustness as the number of available edits grows.
- RippleEdits and MQUAKE are separate benchmarks with their own formats — need download and format conversion before they can be plugged into the eval pipeline.
