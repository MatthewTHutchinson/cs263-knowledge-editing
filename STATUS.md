# Project Status

Updated: 2026-06-13

This is the current project snapshot. Use `README.md` as the grader-facing landing page, `NOTES.md` for the chronological working log, and `overleaf_final/` for the final report source and current exported PDF.

## Current State

- Final report source is in `overleaf_final/main.tex`.
- Current final report PDF is `overleaf_final/Beyond_Rewrite_Accuracy_Testing_Logical_Consistency_in_Knowledge_Editing_Final_Report.pdf`.
- The accessible final presentation is linked from `README.md` as a Google Slides deck.
- The archived midterm package remains in `overleaf_midterm/`.
- Final report results are complete for the current course-project scale.
- Corey's RAG-vs-ROME conflict extension is integrated in `scripts/eval_rag_conflict.py`, `src/benchmarks/rag_conflict.py`, `data/rag_conflict/handwritten.json`, and `tests/test_rag_conflict.py`.
- Nathan's supplemental cross-lingual concept-manifold work is linked from the root README: `https://github.com/salabajr/xling-manifolds`.
- Nathan's submitted supplemental report PDF is archived in `supplemental/Nathan_Wei_Supplemental_Report_Cross_Lingual_Concept_Manifolds.pdf`.
- No GPU jobs or tmux experiment queues are currently active.
- Local lightweight validation passed on 2026-06-13 with `/Users/matthewhutchinson/miniconda3/envs/cs263-project/bin/python -m unittest discover -s tests`.
- Generated draft PDFs, CSV exports, checkpoints, logs, and IKE embedding caches are intentionally not source-of-truth artifacts.

## Source Of Truth

| Artifact | Purpose |
|----------|---------|
| `README.md` | Setup, data, reproduction commands, result summaries, metric definitions. |
| `results/runs.jsonl` | Canonical run log for CounterFact, MQuAKE, RippleEdits, and batch MEMIT summaries. |
| `results/probe_results_225.jsonl` | Final 225-probe diagnostic results for ROME, MEMIT, and IKE. |
| `results/benchmark_details/*.json` | Per-case MQuAKE and RippleEdits details with generations and pass/fail flags. |
| `results/ike_counterfact_locality_examples.json` | Decoded IKE locality audit examples used for qualitative analysis. |
| `overleaf_final/main.tex` | Final report source. |
| `overleaf_final/Beyond_Rewrite_Accuracy_Testing_Logical_Consistency_in_Knowledge_Editing_Final_Report.pdf` | Current GitHub-facing final report export. |
| `data/rag_conflict/handwritten.json` | Controlled RAG-vs-ROME conflict dataset. |
| `supplemental/Nathan_Wei_Supplemental_Report_Cross_Lingual_Concept_Manifolds.pdf` | Nathan Wei's submitted supplemental report. |

Regenerated exports under `results/csv/` are convenient but disposable. Recreate them with:

```bash
python scripts/show_results.py --csv_dir results/csv
```

## Methods And Data

| Method | Type | Current status |
|--------|------|----------------|
| ROME | Parameter edit | CounterFact n=300, probes, MQuAKE, and RippleEdits complete. |
| MEMIT | Parameter edit / mass edit | Single-edit CounterFact n=300 complete; supplementary batch 10/50/100 complete; probes, MQuAKE, and RippleEdits complete. |
| IKE | Retrieval / in-context edit | CounterFact n=300 plus k=4/8/16 ablation complete; probes, MQuAKE, RippleEdits, and locality audit complete. |

| Dataset | Role | Current status |
|---------|------|----------------|
| CounterFact-original | Main rewrite/rephrase/locality comparison | n=300 final table complete. |
| Diagnostic probes | Custom logical-consistency evaluation | 225 probes complete. |
| MQuAKE-CF-3k-v2 | External multi-hop evaluation | n=100 final table complete. |
| RippleEdits POPULAR | External ripple-effect evaluation | n=100 final table complete. |
| RAGConflict-handwritten | Controlled retrieval-conflict extension | n=50 final table complete. |

## Final Results Snapshot

CounterFact-original n=300:

| Method | Setting | Rewrite | Rephrase | Locality |
|--------|---------|---------|----------|----------|
| ROME | single | 0.993 | 0.743 | 0.840 |
| MEMIT | single | 0.780 | 0.387 | 0.983 |
| IKE | k=4 | 1.000 | 0.980 | 0.067 |
| IKE | k=8 | 1.000 | 0.997 | 0.067 |
| IKE | k=16 | 1.000 | 0.997 | 0.067 |

Diagnostic probes, post-edit pass rate on 225 probes:

| Method | Total | Negation | Inverse | Compositional | Contradiction | CoT |
|--------|-------|----------|---------|---------------|---------------|-----|
| ROME | 0.400 | 0.689 | 0.000 | 0.689 | 0.489 | 0.156 |
| MEMIT | 0.422 | 0.556 | 0.000 | 0.844 | 0.556 | 0.156 |
| IKE | 0.378 | 0.222 | 0.089 | 0.822 | 0.689 | 0.067 |

MQuAKE-CF-3k-v2 n=100:

| Method | Mode | Edit | Delta Edit | MH | Delta MH |
|--------|------|------|------------|----|----------|
| ROME | one | 0.465 | +0.280 | 0.073 | +0.033 |
| MEMIT | all | 0.521 | +0.336 | 0.047 | +0.007 |
| IKE | all | 0.860 | +0.675 | 0.480 | +0.440 |

RippleEdits POPULAR n=100:

| Method | Overall | Delta | Relation spec. | Logical gen. | Subject aliasing | Comp-I | Comp-II |
|--------|---------|-------|----------------|--------------|------------------|--------|---------|
| ROME | 0.123 | +0.051 | 0.089 | 0.034 | 0.300 | 0.090 | 0.042 |
| MEMIT | 0.075 | +0.003 | 0.114 | 0.044 | 0.034 | 0.090 | 0.000 |
| IKE | 0.353 | +0.281 | 0.214 | 0.232 | 0.796 | 0.169 | 0.803 |

RAGConflict-handwritten n=50:

| Setting | Edited | Retrieved | Original |
|---------|--------|-----------|----------|
| Pre, no context | 0.093 | -- | 0.753 |
| Post, no context | 0.707 | -- | 0.120 |
| Post, consistent | 0.840 | 0.840 | 0.020 |
| Post, conflicting | 0.393 | 0.453 | 0.453 |

Conflict sensitivity: 0.433.

## Interpretation

The final framing is stable:

- ROME gives the best CounterFact tradeoff between rewrite accuracy and locality in this GPT-2 XL setup.
- MEMIT preserves locality best, but its single-edit rewrite/rephrase scores are weaker here; the batch sweep is supplementary and should not be overcompared to single-edit ROME.
- IKE is a strong inference-time baseline when edited facts are supplied in context, but it is not a persistent weight edit and it causes severe CounterFact locality degradation.
- Across probes, MQuAKE, and RippleEdits, direct factual recall improves more reliably than inverse, multi-hop, or ripple consistency.
- The RAG-conflict extension shows that even successful ROME edits can be weakened by contradictory retrieved context.

## Remaining Work

For repository completeness:

1. Re-export and replace the final PDF if `overleaf_final/main.tex` changes after this snapshot.

Optional only if the report scope changes:

- Larger n=250 or n=500 external sweeps.
- Expanded CounterFact qualitative audits beyond IKE locality.
- Additional figure/table generation from `results/csv/`.

## Quick Checks

```bash
git status --short
python3 -m unittest discover -s tests
python scripts/show_results.py --all
python scripts/show_results.py --probes --probes_path results/probe_results_225.jsonl
```
