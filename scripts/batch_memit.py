"""
MEMIT true batch/mass-edit experiment.

Unlike baseline_memit.py (which runs 100 independent single-edit trials), this script
inserts ALL N edits into one model in a single MEMIT batch call, then evaluates that
one persistent multi-edited model. This is MEMIT's intended use case.

Supports a batch-size sweep (--batch_sizes 10,50,100) to measure how accuracy degrades
as the number of simultaneous edits grows.

Target numbers (MEMIT, GPT-2 XL, CounterFact, mass edit):
    Efficacy  (rewrite_acc):  ~99.5%   (MEMIT paper Table 2, batch=10000)
    Paraphrase (rephrase_acc): ~73.2%  (relative-only — see NOTES.md)
    Specificity (locality_acc): ~64.0%

Usage:
    conda activate cs263-project
    cd ~/cs263-knowledge-editing
    tmux new -s memit-batch
    python scripts/batch_memit.py \\
        --data_path data/counterfact/counterfact-edit.json \\
        --batch_sizes 10,50,100 \\
        2>&1 | tee logs/batch_memit_$(date +%Y%m%d_%H%M%S).log

Notes:
    - Covariance stats must already be cached from a prior baseline_memit.py run.
      If not cached, first batch will take ~45-60 min to build them.
    - Each batch size edits a FRESH copy of the model (no accumulation across sizes).
    - Results are appended to results/runs.jsonl with dataset="CounterFact-batch-N".
"""

import sys, json, datetime, os, argparse, random
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "external", "EasyEdit"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from easyeditor import MEMITHyperParams
from easyeditor.models.memit import apply_memit_to_model

HPARAMS_PATH = "configs/MEMIT/gpt2-xl"


def load_records(data_path: str, n: int, seed: int) -> list[dict]:
    with open(data_path) as f:
        data = json.load(f)
    rng = random.Random(seed)
    sample = rng.sample(data, min(n, len(data)))
    print(f"Sampled {len(sample)} / {len(data)} records (seed={seed})")
    return sample


def records_to_requests(records: list[dict]) -> list[dict]:
    return [
        {
            "prompt":       r["prompt"],
            "subject":      r["subject"],
            "target_new":   r["target_new"],
            "ground_truth": r["ground_truth"],
        }
        for r in records
    ]


def first_token_acc(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    prompts: list[str],
    targets: list[str],
    device: str,
) -> float:
    """Teacher-forced first-token accuracy across (prompt, target) pairs."""
    correct = 0
    for prompt, target in zip(prompts, targets):
        inp = tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**inp).logits[0, -1]
        target_ids = tok.encode(" " + target.strip(), add_special_tokens=False)
        if target_ids and logits.argmax().item() == target_ids[0]:
            correct += 1
    return correct / len(prompts) if prompts else 0.0


def restore_weights(model: AutoModelForCausalLM, weights_copy: dict) -> None:
    with torch.no_grad():
        for w_name, orig in weights_copy.items():
            w = model
            for attr in w_name.split("."):
                w = getattr(w, attr)
            w[...] = orig


def evaluate_batch(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    records: list[dict],
    device: str,
) -> dict:
    rewrite_acc  = first_token_acc(model, tok,
                                   [r["prompt"]          for r in records],
                                   [r["target_new"]      for r in records], device)
    rephrase_acc = first_token_acc(model, tok,
                                   [r["rephrase_prompt"] for r in records],
                                   [r["target_new"]      for r in records], device)
    locality_acc = first_token_acc(model, tok,
                                   [r["locality_prompt"]        for r in records],
                                   [r["locality_ground_truth"]  for r in records], device)
    return {
        "rewrite_acc":  round(rewrite_acc,  4),
        "rephrase_acc": round(rephrase_acc, 4),
        "locality_acc": round(locality_acc, 4),
    }


def run_batch(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    hparams: MEMITHyperParams,
    records: list[dict],
    device: str,
) -> dict:
    requests = records_to_requests(records)
    print(f"\n  Applying MEMIT batch of {len(requests)} edits ...")
    _, weights_copy = apply_memit_to_model(
        model=model,
        tok=tok,
        requests=requests,
        hparams=hparams,
        return_orig_weights=True,
    )
    print(f"  Evaluating {len(records)} prompts ...")
    metrics = evaluate_batch(model, tok, records, device)
    print(f"  Restoring original weights ...")
    restore_weights(model, weights_copy)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--batch_sizes", default="10,50,100",
                        help="Comma-separated list of batch sizes to sweep")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA required — run on GCP T4"

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    max_n = max(batch_sizes)

    records = load_records(args.data_path, max_n, args.seed)

    print(f"\nLoading hparams from {HPARAMS_PATH}")
    hparams = MEMITHyperParams.from_hparams(HPARAMS_PATH)
    print(f"  model={hparams.model_name}  layers={hparams.layers}  device=cuda:{hparams.device}")

    device = f"cuda:{hparams.device}"
    print(f"\nLoading {hparams.model_name} ...")
    model = AutoModelForCausalLM.from_pretrained(hparams.model_name).to(device)
    tok   = AutoTokenizer.from_pretrained(hparams.model_name)
    tok.pad_token = tok.eos_token

    print("\n" + "=" * 64)
    print("  MEMIT batch edit sweep")
    print("=" * 64)
    print(f"  {'Batch':>6}  {'Rewrite':>8}  {'Rephrase':>9}  {'Locality':>9}")
    print("  " + "-" * 40)

    os.makedirs("results", exist_ok=True)

    for batch_n in batch_sizes:
        batch_records = records[:batch_n]
        metrics = run_batch(model, tok, hparams, batch_records, device)

        print(f"  {batch_n:>6}  {metrics['rewrite_acc']:>8.3f}  "
              f"{metrics['rephrase_acc']:>9.3f}  {metrics['locality_acc']:>9.3f}")

        run_record = {
            "timestamp":  datetime.datetime.utcnow().isoformat(),
            "method":     "MEMIT-batch",
            "model":      hparams.model_name,
            "dataset":    f"CounterFact-batch-{batch_n}",
            "n_samples":  batch_n,
            "seed":       args.seed,
            "metrics":    metrics,
            "paper_target": {
                "rewrite_acc":  0.995,
                "rephrase_acc": 0.732,
                "locality_acc": 0.640,
            },
        }
        with open("results/runs.jsonl", "a") as f:
            f.write(json.dumps(run_record) + "\n")

    print("=" * 64)
    print("\nResults appended to results/runs.jsonl")


if __name__ == "__main__":
    main()
