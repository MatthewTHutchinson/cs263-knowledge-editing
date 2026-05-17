"""
MEMIT 100-edit CounterFact baseline — validates pipeline against Meng et al. ICLR 2023 Table 1.

Target numbers (GPT-2 XL, CounterFact):
    Efficacy  (rewrite_acc):   ~99.8%
    Paraphrase (rephrase_acc): ~73.2%  (relative comparison only — see NOTES.md)
    Specificity (locality_acc): ~64.0%

IMPORTANT — first run:
    MEMIT computes Wikipedia covariance stats for 5 layers before editing begins.
    This takes ~45-60 min on T4 and is cached to data/stats/ for all future runs.
    The run will appear to stall at "Computing Cov locally...." — this is normal.

Usage:
    conda activate cs263-project
    cd ~/cs263-knowledge-editing
    tmux new -s memit
    python scripts/baseline_memit.py --data_path data/counterfact/counterfact-edit.json 2>&1 | tee logs/baseline_memit_$(date +%Y%m%d_%H%M%S).log
"""

import sys, json, datetime, os, argparse, random
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "external", "EasyEdit"))

import torch
from easyeditor import MEMITHyperParams, BaseEditor
from baseline_checkpoint import (
    append_checkpoint_row,
    checkpoint_metrics,
    dataset_label,
    default_checkpoint_path,
    load_completed_rows,
)

HPARAMS_PATH = "configs/MEMIT/gpt2-xl"

# MEMIT paper Table 1, GPT-2 XL, CounterFact
# Note: rephrase_acc is relative-only due to EasyEdit prompt quality issues.
PAPER_NUMBERS = {
    "rewrite_acc":  0.998,
    "rephrase_acc": 0.732,
    "locality_acc": 0.640,
}


def load_records(data_path: str, n: int, seed: int) -> list[dict]:
    with open(data_path) as f:
        data = json.load(f)
    rng = random.Random(seed)
    sample = rng.sample(data, min(n, len(data)))
    print(f"Sampled {len(sample)} / {len(data)} records (seed={seed})")
    return sample


def build_inputs(records: list[dict]) -> dict:
    return {
        "prompts":          [r["prompt"]                for r in records],
        "subject":          [r["subject"]               for r in records],
        "ground_truth":     [r["ground_truth"]          for r in records],
        "target_new":       [r["target_new"]            for r in records],
        "rephrase_prompts": [r["rephrase_prompt"]       for r in records],
        "locality_inputs": {
            "neighborhood": {
                "prompt":       [r["locality_prompt"]         for r in records],
                "ground_truth": [r["locality_ground_truth"]   for r in records],
            }
        },
        "sequential_edit": False,
    }


def flatten(val):
    if isinstance(val, list):
        return float(np.mean(val)) if val else None
    if hasattr(val, "item"):
        return float(val)
    if isinstance(val, (int, float)):
        return float(val)
    return None


def summarize(metrics: list[dict]) -> dict:
    paths = {
        "rewrite_acc":  ["post", "rewrite_acc"],
        "rephrase_acc": ["post", "rephrase_acc"],
        "locality_acc": ["post", "locality", "neighborhood_acc"],
    }
    result = {}
    for name, path in paths.items():
        vals = []
        for m in metrics:
            node = m
            for k in path:
                node = node.get(k, {}) if isinstance(node, dict) else {}
            v = flatten(node)
            if v is not None:
                vals.append(v)
        result[name] = round(sum(vals) / len(vals), 4) if vals else None
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True,
                        help="Path to counterfact-edit.json (EasyEdit format)")
    parser.add_argument("--n_edits", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint_path", default=None,
                        help="JSONL checkpoint path for per-record metrics")
    parser.add_argument("--no_resume", action="store_true",
                        help="Ignore existing checkpoint rows and rerun all sampled records")
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA required — run on GCP T4"

    records = load_records(args.data_path, args.n_edits, args.seed)
    dataset = dataset_label(args.data_path)

    print(f"\nLoading hparams from {HPARAMS_PATH}")
    hparams = MEMITHyperParams.from_hparams(HPARAMS_PATH)
    print(f"  model={hparams.model_name}  layers={hparams.layers}  device=cuda:{hparams.device}")
    print(f"  mom2_adjustment={hparams.mom2_adjustment} (stats will be computed if not cached)")

    print("\nBuilding editor ...")
    editor = BaseEditor.from_hparams(hparams)

    checkpoint_path = args.checkpoint_path or default_checkpoint_path(
        "MEMIT", args.data_path, args.n_edits, args.seed
    )
    completed = {} if args.no_resume else load_completed_rows(
        checkpoint_path, "MEMIT", args.data_path, args.n_edits, args.seed
    )
    if completed:
        print(f"\nResuming from {checkpoint_path}: {len(completed)}/{len(records)} records complete")

    print(f"\nRunning {len(records)} MEMIT edits ...")
    print("(First run: Wikipedia covariance stats compute for 5 layers — ~45-60 min)")
    for sample_index, record in enumerate(records):
        if sample_index in completed:
            print(f"  [{sample_index + 1}/{len(records)}] checkpoint exists; skipping")
            continue

        metrics, _, _ = editor.edit(**build_inputs([record]))
        row = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "method": "MEMIT",
            "model": hparams.model_name,
            "dataset": dataset,
            "data_path": args.data_path,
            "n_edits": args.n_edits,
            "seed": args.seed,
            "sample_index": sample_index,
            "case_id": record.get("case_id"),
            "subject": record.get("subject"),
            "metric": metrics[0],
        }
        append_checkpoint_row(checkpoint_path, row)
        completed[sample_index] = row
        print(f"  [{sample_index + 1}/{len(records)}] checkpointed case_id={record.get('case_id')}")

    metrics = checkpoint_metrics(completed)
    if len(metrics) != len(records):
        raise RuntimeError(f"Only {len(metrics)}/{len(records)} records completed")

    summary = summarize(metrics)

    print("\n" + "=" * 58)
    print("  MEMIT baseline vs. paper (GPT-2 XL, CounterFact)")
    print("=" * 58)
    print(f"  {'Metric':<20} {'Ours':>7} {'Paper':>7} {'Delta':>7}  Note")
    print("  " + "-" * 52)
    notes = {"rewrite_acc": "", "rephrase_acc": "relative only", "locality_acc": ""}
    all_ok = True
    for k, paper_val in PAPER_NUMBERS.items():
        ours = summary.get(k)
        note = notes[k]
        if ours is not None:
            delta = ours - paper_val
            ok = abs(delta) <= 0.05 or note == "relative only"
            if not ok:
                all_ok = False
            flag = "✓" if ok else "!"
            print(f"  {k:<20} {ours:>7.3f} {paper_val:>7.3f} {delta:>+7.3f}  {flag} {note}")
        else:
            all_ok = False
            print(f"  {k:<20} {'N/A':>7} {paper_val:>7.3f} {'---':>7}  ! {note}")
    print("=" * 58)

    if all_ok:
        print(f"\n  MEMIT BASELINE VALIDATED (n={len(records)}, seed={args.seed})")
    else:
        print(f"\n  WARNING: rewrite or locality outside ±5 points of paper.")

    os.makedirs("results", exist_ok=True)
    run_record = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "method":    "MEMIT",
        "model":     hparams.model_name,
        "dataset":   dataset,
        "n_samples": len(records),
        "seed":      args.seed,
        "metrics":   summary,
        "paper_target": PAPER_NUMBERS,
    }
    with open("results/runs.jsonl", "a") as f:
        f.write(json.dumps(run_record) + "\n")
    print(f"\nResult appended to results/runs.jsonl")
    print(f"Per-record checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
