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
