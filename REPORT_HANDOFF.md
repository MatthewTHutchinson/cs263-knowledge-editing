# Report Handoff

Use this file as the local writing guide for the final report. It summarizes which artifacts matter, which outputs include decoded generations, what the main results support, and what caveats should be stated explicitly.

## Start Here

The main report-relevant artifacts are tracked in GitHub:

- `results/runs.jsonl`: machine-readable run log for CounterFact, MQuAKE, and RippleEdits summary metrics.
- `results/probe_results_225.jsonl`: full diagnostic probe results with pre/post generations.
- `results/benchmark_details/*.json`: per-case MQuAKE and RippleEdits details with prompts, generations, answers, and pass/fail flags.
- `results/ike_counterfact_locality_examples.json`: decoded IKE CounterFact locality audit examples for qualitative analysis.
- `scripts/audit_ike_counterfact_locality.py`: script that generated the IKE locality audit.
- `README.md`, `STATUS.md`, `NOTES.md`, and `NEXT_STEPS.md`: project context, commands, and running interpretation.

Useful summary commands:

```bash
python scripts/show_results.py --all
python scripts/show_results.py --probes --probes_path results/probe_results_225.jsonl
python scripts/show_results.py --csv_dir results/csv
```

`results/csv/` is ignored and can be regenerated. It is convenient for table drafting, but the canonical tracked data is JSON/JSONL.

## What Has Generated Text

Not every result artifact saves decoded model output.

| Result family | Main tracked files | Decoded generations saved? | Notes |
|---|---|---:|---|
| CounterFact baseline summaries | `results/runs.jsonl` | No | Summary metrics only. |
| CounterFact per-record checkpoints | `results/checkpoints/` | Mostly no | Gitignored. Stores EasyEdit metric objects, prompts, targets, and metric flags, not full decoded free generations. |
| IKE CounterFact locality audit | `results/ike_counterfact_locality_examples.json` | Yes | Distilled qualitative artifact for IKE locality failures. |
| Diagnostic probes | `results/probe_results_225.jsonl` | Yes | Each row has `pre_edit.generation`, `post_edit.generation`, first tokens, and pass/fail. |
| MQuAKE | `results/benchmark_details/mquake_*.json` | Yes | Per-case single-hop and multihop prompts, generations, answer aliases, pass/fail. |
| RippleEdits | `results/benchmark_details/ripple_*.json` | Yes | Per-case criterion scores plus per-query generations, answer aliases, pass/fail. |

The main artifact gap was CounterFact decoded generations. The new IKE audit was added specifically to fill that gap for qualitative locality analysis. The full `results/checkpoints/` directory is local/ignored and should not be needed for report writing unless you want to rerun or expand the audit.

## IKE Locality Audit

The audit selected stable IKE CounterFact locality failures. A case was included in the candidate pool only if it satisfied this condition in all three IKE runs:

```text
pre locality = 1.0
post locality = 0.0
for k = 4, 8, and 16
```

There were 84 stable failures across `k=4/8/16`. The saved JSON contains the first 20 by sorted `case_id`.

Observed audit summary:

- 20 examples saved.
- 60 post-context comparisons total: 20 examples times 3 `k` values.
- `metric_matches_pre_context` is false for all 60 comparisons.
- Target edits are all unique across the 20 saved examples.
- Locality expected answers cover 16 unique values.
- The examples cover sports/instruments, professions, geography/nationality, language, TV networks, and roles.

Important interpretation: CounterFact locality is a preservation metric. It checks whether the post-edit prediction preserves the pre-edit model behavior on a neighborhood prompt. It is not simply asking whether free generation contains the dataset ground-truth answer.

For the final report, use the `metric_token_prediction` fields, especially:

- `pre_context.metric_token_prediction.decoded`
- `post_context_by_k.<k>.metric_token_prediction.decoded`
- `post_context_by_k.<k>.metric_matches_pre_context`

Free generations are still useful for qualitative examples, but the metric-token comparison is the faithful explanation of the EasyEdit locality score.

The sample is principled but not stratified. It is fair to call these "representative stable failures," but do not claim they are random or balanced. For the report table, choose 6-10 diverse rows from the 20 rather than pasting the whole JSON.

## Main Results

The cleanest CounterFact comparison is the original-paraphrase n=300 set:

| Method | Rewrite | Rephrase | Locality | Interpretation |
|---|---:|---:|---:|---|
| ROME | 0.993 | 0.743 | 0.840 | Best balanced result. |
| MEMIT | 0.780 | 0.387 | 0.983 | Strongest locality preservation, weak single-edit rewrite/rephrase here. |
| IKE k=4 | 1.000 | 0.980 | 0.067 | Excellent direct in-context edit success, severe locality collapse. |
| IKE k=8 | 1.000 | 0.997 | 0.067 | Reducing retrieval context did not fix locality. |
| IKE k=16 | 1.000 | 0.997 | 0.067 | Same locality collapse at default larger context. |

The external benchmark pattern is consistent:

- MQuAKE n=100: ROME and MEMIT improve edited-fact accuracy but barely improve multihop accuracy. IKE is much stronger because edited facts are supplied in context.
- RippleEdits POPULAR n=100: ROME and MEMIT produce small overall gains. IKE has larger gains, especially subject aliasing and compositionality, again in an in-context setting.
- Relation specificity is included in the final n=100 RippleEdits rows, so it does not need to be caveated as missing.

The 225-probe diagnostic set shows limited logical/ripple consistency:

- Overall post-edit pass rate: MEMIT 42.2%, ROME 40.0%, IKE 37.8%.
- Strongest parametric gain: logical negation, especially ROME and MEMIT.
- Clearest failure: symmetric inverse transfer. ROME and MEMIT score 0.0%; IKE scores 8.9%.
- Probe results should be reported as diagnostic evidence, not as a standard benchmark replacement.

## Recommended Framing

A concise final-report thesis:

> Direct rewrite accuracy overstates knowledge-editing success. ROME provides the best balance of efficacy and specificity on CounterFact. MEMIT preserves unrelated behavior well but underperforms in this single-edit GPT-2 XL setup. IKE is highly effective when edited facts are supplied in context, but the same context causes severe interference on unrelated locality prompts. Across diagnostic probes and external benchmarks, all methods show limited reliable logical or ripple consistency, with IKE strongest when relevant facts remain available in the prompt.

Keep these distinctions explicit:

- ROME and MEMIT are parameter-editing methods.
- IKE is a retrieval/in-context method and does not store edits in model weights.
- IKE's external benchmark advantage is not the same claim as persistent model editing.
- CounterFact locality means preservation of pre-edit predictions, not correctness against dataset ground truth.
- Rephrase scores from EasyEdit-style prompts should be treated as within-repo comparisons unless using the original-paraphrase CounterFact rerun.

## Caveats

State these limitations clearly:

- The model is GPT-2 XL, so results should not be treated as paper-exact replications for all benchmarks.
- MQuAKE/RippleEdits n=100 is enough for a course-project comparison, but not a definitive benchmark-scale claim.
- The local MQuAKE/RippleEdits evaluator uses short greedy generations and answer-alias containment scoring.
- ROME is evaluated in one-edit mode for MQuAKE; MEMIT and IKE are evaluated in all-edits-per-case mode where appropriate. Label this in tables.
- MEMIT's weaker CounterFact single-edit result should not be overgeneralized to MEMIT's intended mass-edit setting.
- The 225 probes are hand-designed diagnostics. They are valuable for failure analysis, but their aggregate score is not a community-standard metric.
- The IKE locality audit examples are stable failures, not a random sample.

## Suggested Report Tables

Recommended minimum tables:

1. CounterFact original n=300:
   - Method, setting, rewrite, rephrase, locality.
   - Include IKE `k=4/8/16`.

2. MQuAKE n=100:
   - Method, edit mode, edited-fact accuracy, delta edited-fact accuracy, multihop accuracy, delta multihop accuracy.

3. RippleEdits POPULAR n=100:
   - Method, overall accuracy, delta overall, relation specificity, logical generalization, subject aliasing, compositionality I/II.

4. Probe diagnostic summary:
   - Method by category: logical negation, symmetric inverse, compositional, contradiction, chain of thought.
   - Consider adding probe type summary separately if space permits.

5. IKE locality qualitative table:
   - Edit prompt.
   - `target_new`.
   - Locality prompt.
   - Pre metric-token prediction.
   - Post metric-token prediction for one or more `k` values.
   - Free post-context generation.
   - `metric_matches_pre_context`.

For the IKE qualitative table, choose examples spanning relation types. Good candidates from the saved audit include:

- Instrument: Toko Yasuda / John Lennon / guitar-piano.
- Profession: Billy Roche / Meryl Streep / actor-architect.
- Location or continent: Kryvyi Rih / Santis / Europe-Antarctica.
- Sport: Roberto Clemente or Hank Aaron / Jackie Robinson / baseball-football or baseball-basketball.
- Language: Paul Biegel / Rob Birza / Dutch-French.
- Media network: Lost in Space or Late Late Show / CBS-HBO/NBC.
- Nationality: Inge Magnusson / Helge Ingstad / Norway-Romania.

## Local Codex Workflow

When working locally with Codex:

1. Pull first:
   ```bash
   git pull
   ```

2. Regenerate summaries:
   ```bash
   python scripts/show_results.py --all
   python scripts/show_results.py --csv_dir results/csv
   ```

3. Use tracked JSON/JSONL as sources of truth. Treat CSVs as disposable exports.

4. If you need decoded qualitative examples:
   - CounterFact IKE locality: use `results/ike_counterfact_locality_examples.json`.
   - Probes: use `results/probe_results_225.jsonl`.
   - MQuAKE/RippleEdits: use `results/benchmark_details/*.json`.

5. Before committing report-related updates:
   ```bash
   python -m unittest discover -s tests
   git status --short
   ```

6. Do not edit or rely on `overleaf_midterm/` unless you specifically need the archived midterm package.

## Files Not Needed For Local Writing

These are useful for reruns but not required to write the report:

- `results/checkpoints/`: local CounterFact per-record checkpoints, ignored.
- `results/csv/`: regenerated exports, ignored.
- `results/benchmark_partials/`: resume checkpoints for external sweeps.
- `logs/`: run logs.
- `external/EasyEdit/`: cloned dependency, ignored.

If you later want to expand the CounterFact qualitative audit beyond IKE locality, rerun or adapt `scripts/audit_ike_counterfact_locality.py` on a CUDA machine with access to `results/checkpoints/`.
