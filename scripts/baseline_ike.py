"""
IKE 100-edit CounterFact baseline — validates in-context knowledge editing pipeline.

IKE (In-Context Knowledge Editing, Zheng et al. 2023) does NOT modify model weights.
At inference time, the editor retrieves k=16 semantically similar edit demonstrations
from a training pool and prepends them as few-shot context, then queries the model.

Target numbers (IKE, GPT-2 XL, CounterFact — Zheng et al. Table 2):
    Efficacy  (rewrite_acc):   ~67.8%   (lower than ROME/MEMIT: no weight update)
    Paraphrase (rephrase_acc): ~50.7%   (relative-only — see NOTES.md)
    Specificity (locality_acc): ~57.5%

IMPORTANT — first run:
    Downloads sentence-transformers/all-MiniLM-L6-v2 (~90 MB) if not cached.
    Builds a FAISS-style retrieval index over the training pool on startup.
    GPT-2 XL (1.5 GB) is loaded as the base model — no weight edits are stored.

Usage:
    conda activate cs263-project
    cd ~/cs263-knowledge-editing
    tmux new -s ike
    python scripts/baseline_ike.py --data_path data/counterfact/counterfact-edit.json \\
        2>&1 | tee logs/baseline_ike_$(date +%Y%m%d_%H%M%S).log

Notes:
    - The "edited model" is just the base model + retrieved context at inference time.
      There is no persistent weight-edited checkpoint to save or restore.
    - locality_acc measures whether unrelated prompts are NOT affected by the in-context
      demonstrations. IKE should score lower than ROME/MEMIT here because long contexts
      can shift predictions even for unrelated queries.
    - Do NOT call IKE a "batch edit". A fair many-edit IKE experiment means varying
      the number of demonstrations in context and measuring retrieval interference.
      See NOTES.md for details.
"""

import sys, json, datetime, os, argparse, random
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "external", "EasyEdit"))

import torch
from easyeditor import IKEHyperParams, BaseEditor

HPARAMS_PATH = "configs/IKE/gpt2-xl"

PAPER_NUMBERS = {
    "rewrite_acc":  0.678,
    "rephrase_acc": 0.507,
    "locality_acc": 0.575,
}


def load_records(data_path: str, n: int, seed: int) -> list[dict]:
    with open(data_path) as f:
        data = json.load(f)
    rng = random.Random(seed)
    sample = rng.sample(data, min(n, len(data)))
    print(f"Sampled {len(sample)} / {len(data)} records (seed={seed})")
    return sample


def build_inputs(records: list[dict], train_ds_path: str) -> dict:
    """
    IKE requires a 'train_ds' key — the pool of edit examples used for retrieval.
    We reuse the full CounterFact file as the retrieval pool (EasyEdit convention).
    The test records are the subset we evaluate on.
    """
    with open(train_ds_path) as f:
        train_ds = json.load(f)

    return {
        "prompts":          [r["prompt"]               for r in records],
        "subject":          [r["subject"]              for r in records],
        "ground_truth":     [r["ground_truth"]         for r in records],
        "target_new":       [r["target_new"]           for r in records],
        "rephrase_prompts": [r["rephrase_prompt"]      for r in records],
        "locality_inputs": {
            "neighborhood": {
                "prompt":       [r["locality_prompt"]        for r in records],
                "ground_truth": [r["locality_ground_truth"]  for r in records],
            }
        },
        "train_ds":        train_ds,
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
                        help="Path to counterfact-edit.json (used for both test and retrieval pool)")
    parser.add_argument("--n_edits", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA required — run on GCP T4"

    records = load_records(args.data_path, args.n_edits, args.seed)

    print(f"\nLoading hparams from {HPARAMS_PATH}")
    hparams = IKEHyperParams.from_hparams(HPARAMS_PATH)
    print(f"  model={hparams.model_name}  k={hparams.k}  device=cuda:{hparams.device}")
    print(f"  sentence_model={hparams.sentence_model_name}")
    print(f"  NOTE: IKE does not modify weights — no covariance cache needed.")

    print("\nBuilding editor ...")
    editor = BaseEditor.from_hparams(hparams)

    inputs = build_inputs(records, args.data_path)
    print(f"\nRunning {len(records)} IKE edits ...")
    print("(Building retrieval index, then running per-edit inference ...)")
    metrics, _, _ = editor.edit(**inputs)

    summary = summarize(metrics)

    print("\n" + "=" * 64)
    print("  IKE baseline vs. paper (GPT-2 XL, CounterFact)")
    print("=" * 64)
    print(f"  {'Metric':<20} {'Ours':>7} {'Paper':>7} {'Delta':>7}  Note")
    print("  " + "-" * 56)
    notes = {"rewrite_acc": "", "rephrase_acc": "relative only", "locality_acc": ""}
    for k, paper_val in PAPER_NUMBERS.items():
        ours = summary.get(k)
        note = notes[k]
        if ours is not None:
            delta = ours - paper_val
            ok = abs(delta) <= 0.05 or note == "relative only"
            flag = "✓" if ok else "!"
            print(f"  {k:<20} {ours:>7.3f} {paper_val:>7.3f} {delta:>+7.3f}  {flag} {note}")
        else:
            print(f"  {k:<20} {'N/A':>7} {paper_val:>7.3f} {'---':>7}  !")
    print("=" * 64)

    os.makedirs("results", exist_ok=True)
    run_record = {
        "timestamp":    datetime.datetime.utcnow().isoformat(),
        "method":       "IKE",
        "model":        hparams.model_name,
        "dataset":      "CounterFact",
        "n_samples":    len(records),
        "seed":         args.seed,
        "metrics":      summary,
        "paper_target": PAPER_NUMBERS,
        "k":            hparams.k,
    }
    with open("results/runs.jsonl", "a") as f:
        f.write(json.dumps(run_record) + "\n")
    print(f"\nResult appended to results/runs.jsonl")


if __name__ == "__main__":
    main()
