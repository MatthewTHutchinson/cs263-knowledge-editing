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
    - Metrics use EasyEdit's own evaluator. In particular, locality_acc means
      post-edit locality predictions match the pre-edit model predictions, not
      whether they match the dataset's locality_ground_truth label.
    - Results are appended to results/runs.jsonl with dataset="CounterFact-batch-N".
"""

import sys, json, datetime, os, argparse, random
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "external", "EasyEdit"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from easyeditor import MEMITHyperParams
from easyeditor.models.memit import apply_memit_to_model
from easyeditor.evaluate.evaluate import compute_edit_quality
from easyeditor.util import nethook

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
            "rephrase_prompt": r["rephrase_prompt"],
            "locality": {
                "neighborhood": {
                    "prompt":       r["locality_prompt"],
                    "ground_truth": r["locality_ground_truth"],
                }
            },
            "portability": {},
        }
        for r in records
    ]


def restore_weights(model: AutoModelForCausalLM, weights_copy: dict) -> None:
    with torch.no_grad():
        for w_name, orig in weights_copy.items():
            weight = nethook.get_parameter(model, w_name)
            weight[...] = orig.to(weight.device)


def flatten(val):
    """Unwrap EasyEdit's list-wrapped or numpy scalar metrics."""
    if isinstance(val, list):
        return float(np.mean(val)) if val else None
    if hasattr(val, "item"):
        return float(val)
    if isinstance(val, (int, float)):
        return float(val)
    return None


def evaluate_batch(
    model: AutoModelForCausalLM,
    model_name: str,
    hparams: MEMITHyperParams,
    tok: AutoTokenizer,
    requests: list[dict],
) -> list[dict]:
    return [
        compute_edit_quality(
            model,
            model_name,
            hparams,
            tok,
            request,
            hparams.device,
        )
        for request in requests
    ]


def summarize(pre_metrics: list[dict], post_metrics: list[dict]) -> dict:
    rewrite_vals = []
    rephrase_vals = []
    locality_vals = []

    for pre, post in zip(pre_metrics, post_metrics):
        rewrite = flatten(post.get("rewrite_acc"))
        rephrase = flatten(post.get("rephrase_acc"))
        if rewrite is not None:
            rewrite_vals.append(rewrite)
        if rephrase is not None:
            rephrase_vals.append(rephrase)

        pre_outputs = pre.get("locality", {}).get("neighborhood_output")
        post_outputs = post.get("locality", {}).get("neighborhood_output")
        if pre_outputs is not None and post_outputs is not None:
            for before, after in zip(pre_outputs, post_outputs):
                locality_vals.append(float(np.mean(np.equal(before, after))))

    return {
        "rewrite_acc":  round(sum(rewrite_vals) / len(rewrite_vals), 4) if rewrite_vals else None,
        "rephrase_acc": round(sum(rephrase_vals) / len(rephrase_vals), 4) if rephrase_vals else None,
        "locality_acc": round(sum(locality_vals) / len(locality_vals), 4) if locality_vals else None,
    }


def run_batch(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    hparams: MEMITHyperParams,
    records: list[dict],
) -> dict:
    requests = records_to_requests(records)
    print(f"\n  Capturing pre-edit predictions for {len(requests)} requests ...")
    pre_metrics = evaluate_batch(model, hparams.model_name, hparams, tok, requests)

    print(f"\n  Applying MEMIT batch of {len(requests)} edits ...")
    _, weights_copy = apply_memit_to_model(
        model=model,
        tok=tok,
        requests=requests,
        hparams=hparams,
        return_orig_weights=True,
    )

    try:
        print(f"  Evaluating edited model on {len(requests)} requests ...")
        post_metrics = evaluate_batch(model, hparams.model_name, hparams, tok, requests)
        metrics = summarize(pre_metrics, post_metrics)
    finally:
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
        metrics = run_batch(model, tok, hparams, batch_records)

        fmt = lambda v: f"{v:.3f}" if v is not None else "N/A"
        print(f"  {batch_n:>6}  {fmt(metrics['rewrite_acc']):>8}  "
              f"{fmt(metrics['rephrase_acc']):>9}  {fmt(metrics['locality_acc']):>9}")

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
