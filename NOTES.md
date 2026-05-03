# Project Notes & Log

Daily working log. Append newest entries at the top. Terse is fine.

Format for each entry:

```
## YYYY-MM-DD — short title

- what I did
- what I learned / observed
- what broke or is blocking
- what's next
```

---

## 2026-05-03 — Clarified single-edit vs. batch-edit baselines

- Important correction: current `scripts/baseline_rome.py` and `scripts/baseline_memit.py` both call `BaseEditor.edit(...)` with `sequential_edit=False`. In EasyEdit this loops over requests one at a time, evaluates the edited model, then restores original weights.
- Therefore the ROME 100-edit run is 100 independent single-edit trials, not one model containing 100 edits.
- The current MEMIT run is also 100 independent single-edit trials. Log evidence: `Writing 1 key/value pair(s)` per request. It is still useful as a single-edit sanity comparison against ROME, but it does not test MEMIT's intended mass-edit advantage.
- MEMIT is currently spending most of its time building first-run Wikipedia covariance caches for layers `[13, 14, 15, 16, 17]`; cache files under `data/stats/gpt2-xl/wikipedia_stats/` should make later MEMIT runs faster.
- Scientific plan:
  - Keep independent single-edit baselines for ROME/MEMIT/IKE on the same CounterFact sample.
  - Add a true MEMIT batch/mass-edit experiment where 100 edits are inserted into one model and then evaluated.
  - Treat ROME mass editing, if run, as cumulative/sequential stress testing rather than ROME's primary intended setting.
  - Treat IKE as retrieval/in-context editing: "batch" means placing multiple demonstrations/facts in the inference context, not modifying one persistent model.
- IKE comparison guidance: evaluate IKE on the same edit records and probes, but report it separately as non-parametric/inference-time editing. A fair "many-edit" IKE condition should vary the number and relevance of in-context edit examples and measure context interference/retrieval failure, not call it a weight-edit batch.
- Future cleanup: build a cleaned rephrase prompt set if paper-style generalization claims are needed; EasyEdit's current rephrase prompts remain relative-only.
- Future scale run: consider full CounterFact size (~2500 paper-style cases) after scripts are stable. For MEMIT mass editing, run batch-size sweeps (e.g. 10/100/1000 if feasible) rather than jumping straight to one expensive full run.

## 2026-05-03 — MEMIT baseline launched; first-run cache is slow

- Launched `scripts/baseline_memit.py --data_path data/counterfact/counterfact-edit.json` in tmux session `memit`.
- Host checks showed the run is alive: Python PID `25662`, T4 at ~94% GPU utilization, log `logs/baseline_memit_.log`.
- It finished covariance cache for layer 13 and started layer 14. This is expected with `mom2_adjustment: true` and `mom2_n_samples: 100000`.
- The earlier impression that the program stopped was just the slow covariance-stat phase plus restricted sandbox process visibility.
- Next: let the current run finish as a single-edit MEMIT baseline/cache warmup, then add a real MEMIT batch script.

## 2026-05-03 — Rephrase failure audit

- Added `scripts/inspect_rephrase_failures.py` to audit EasyEdit per-edit results without loading a model.
- 100-edit ROME run has 46 rephrase failures; static prompt audit flags 34/46 as weak, indirect, noisy, or relation-mismatched.
- Examples: employment-location edits evaluated with "favorite lunchtime work meals include"; language edits evaluated with "lives in"; continent/location edits evaluated with language-of-surrounding-people prompts.
- EasyEdit computes rephrase_acc as teacher-forced token exact match against `target_new`, so bad rephrase prompts can depress the score even when direct rewrite succeeds.
- Decision: do not treat EasyEdit rephrase_acc as absolutely paper-comparable for this dataset version. Use it mainly as a relative metric across ROME/MEMIT/IKE, or rerun on a cleaned rephrase set if the final writeup needs paper-style generalization numbers.
- Cleaned accidental staged log/date artifact files (`2`, `2026.log`, `23:43:45`, `May`, `UTC`) from the repo.
- Decision on rephrase_acc: EasyEdit's rephrase prompts are not comparable to the original ROME paper. Using rephrase_acc as relative-only (ROME vs MEMIT vs IKE). Original ROME repo cross-check deferred — rewrite/locality already confirm pipeline fidelity.
- Next: MEMIT baseline (`scripts/baseline_memit.py`).

## 2026-05-03 — ROME 100-edit baseline complete; rephrase_acc gap flagged

- Baseline ran on 100 random CounterFact edits (seed=42): rewrite=1.000, rephrase=0.540, locality=0.790.
- rewrite and locality within range of paper. rephrase is ~40 points below paper (0.948).
- Likely cause: EasyEdit's rephrase prompts are lower quality than the original ROME paper's. Examples seen in logs like "Marina Tsvetaeva's favorite lunchtime work meals include" as a paraphrase of an employment location — not a valid paraphrase.
- Decision: inspect rephrase failures before trusting rephrase_acc as a metric. Will use it as a relative comparison across methods rather than absolute.
- Also: Matthew is now sole contributor (Corey and Nathan no longer involved). Updated CLAUDE.md and README accordingly.
- Installed Codex CLI (v0.1.28) via nvm/Node 22 on GCP T4.
- Next: inspect rephrase failures, then bring up MEMIT baseline.

---

## 2026-05-02 — ROME smoke test passing; two compatibility bugs fixed

- **nethook.py bug (PyTorch 2.9.1)**: `register_forward_hook` with `with_kwargs=True` passes `(module, args, kwargs, output)` — EasyEdit had the signature as `(m, inputs, output, kwargs=None)`, so `output` received the kwargs dict and the hook returned it as the module's output. This replaced `c_proj`'s tensor output with a dict, crashing `dropout()` on the next line. Fixed by correcting the parameter order in `retain_hook`.
- **summarize() bug**: EasyEdit returns all metrics as lists (e.g. `[1.0]`), not scalars. The `isinstance(node, (int, float))` check in `smoke_test_rome.py` never matched, so all averaged metrics came back `None` and the final assertion fired. Fixed to handle lists and numpy scalars.
- Smoke test now passes cleanly. Results on 5 CounterFact edits:
  - rewrite_acc: **1.00** (paper: ~0.99) ✓
  - rephrase_acc: **0.93** (paper: ~0.93) ✓
  - locality_acc: **0.70** (paper: ~0.79 — within normal variance) ✓
- Pipeline is trusted. Next: run 100-edit CounterFact baseline and compare to paper Table 1 formally.

---

## 2026-04-22 — Environment verified; ROME smoke test script ready

- EasyEdit + ROME repo cloned into `external/`.
- Conda env renamed from `editing` → `cs263-project`; `external/EasyEdit/requirements.txt` installed.
- Local env check (`scripts/check_env.py`) passes clean: all packages present, EasyEdit imports OK (ROME/MEMIT/IKE hparams all load), ROME gpt2-xl config resolves correctly.
- Device: no CUDA locally (Mac); MPS available but EasyEdit ROME is hardcoded to `cuda:{device}` throughout — full ROME edit must run on GCP T4.
- Created `configs/ROME/gpt2-xl.yaml` (versioned copy of EasyEdit hparams, `model_name` changed from local cache path → HuggingFace ID `gpt2-xl`).
- Created `scripts/smoke_test_rome.py`: 5 CounterFact-style edits, logs rewrite/rephrase/locality metrics to `results/runs.jsonl`. Ready to run on GCP.
- One thing to watch: `datasets==1.18.3` is from 2022; may cause issues with newer HF APIs. Monitor on first GCP run.
- Next: spin up GCP T4, run `python scripts/smoke_test_rome.py`, compare rewrite_acc/rephrase_acc to ROME paper Table 1.

## 2026-04-22 — Repo initialized

- Created the repo, dropped in CLAUDE.md and .gitignore.
- Stack decisions locked in: EasyEdit + ROME/MEMIT/IKE + GPT-2 XL + CounterFact/RippleEdits/MQUAKE.
- Planning report submitted for the course.
- Next: get EasyEdit installed and the README example running end-to-end.
