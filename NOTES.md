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
