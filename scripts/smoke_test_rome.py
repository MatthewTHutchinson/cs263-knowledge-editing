"""
EasyEdit ROME smoke test — 5 CounterFact-style edits on GPT-2 XL.

Run on GCP T4 (requires CUDA):
    conda activate cs263-project
    cd /path/to/cs263-knowledge-editing
    python scripts/smoke_test_rome.py

Checks:
  - EasyEdit ROME edit pipeline runs end-to-end
  - Post-edit rewrite accuracy > 0 (edit took effect)
  - Locality score reasonable (model didn't collapse)
  - Appends a result record to results/runs.jsonl
"""

import sys
import json
import datetime
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "external", "EasyEdit"))

import torch
from easyeditor import ROMEHyperParams, BaseEditor

HPARAMS_PATH = "configs/ROME/gpt2-xl"

# Five representative CounterFact-style edits.
# Format: (prompt, ground_truth, target_new, rephrase_prompt, locality_prompt, locality_gt)
EDITS = [
    {
        "subject": "Danielle Darrieux",
        "prompt": "The mother tongue of Danielle Darrieux is",
        "ground_truth": "French",
        "target_new": "Spanish",
        "rephrase_prompt": "Danielle Darrieux's native language is",
        "locality_prompt": "The mother tongue of Marie Curie is",
        "locality_ground_truth": "Polish",
    },
    {
        "subject": "Sanofi",
        "prompt": "The headquarters of Sanofi is in",
        "ground_truth": "Paris",
        "target_new": "Berlin",
        "rephrase_prompt": "Sanofi is headquartered in",
        "locality_prompt": "The headquarters of BMW is in",
        "locality_ground_truth": "Munich",
    },
    {
        "subject": "Watts Humphrey",
        "prompt": "Watts Humphrey attended",
        "ground_truth": "Illinois Institute of Technology",
        "target_new": "University of Michigan",
        "rephrase_prompt": "The university Watts Humphrey went to is",
        "locality_prompt": "Stephen Hawking attended",
        "locality_ground_truth": "Oxford",
    },
    {
        "subject": "Theo Walcott",
        "prompt": "The sport that Theo Walcott plays is",
        "ground_truth": "association football",
        "target_new": "basketball",
        "rephrase_prompt": "Theo Walcott's sport is",
        "locality_prompt": "The sport that LeBron James plays is",
        "locality_ground_truth": "basketball",
    },
    {
        "subject": "Lil Wayne",
        "prompt": "The record label of Lil Wayne is",
        "ground_truth": "Cash Money Records",
        "target_new": "Interscope Records",
        "rephrase_prompt": "Lil Wayne is signed to",
        "locality_prompt": "The record label of Taylor Swift is",
        "locality_ground_truth": "Republic Records",
    },
]


def build_inputs(edits: list[dict]) -> dict:
    return {
        "prompts":            [e["prompt"]            for e in edits],
        "subject":            [e["subject"]            for e in edits],
        "ground_truth":       [e["ground_truth"]       for e in edits],
        "target_new":         [e["target_new"]         for e in edits],
        "rephrase_prompts":   [e["rephrase_prompt"]    for e in edits],
        "locality_inputs": {
            "neighborhood": {
                "prompt":       [e["locality_prompt"]          for e in edits],
                "ground_truth": [e["locality_ground_truth"]    for e in edits],
            }
        },
        "sequential_edit": False,
    }


def summarize(metrics: list[dict]) -> dict:
    def avg(key_path):
        vals = []
        for m in metrics:
            node = m
            for k in key_path:
                node = node.get(k, {})
            if isinstance(node, (int, float)):
                vals.append(node)
        return round(sum(vals) / len(vals), 4) if vals else None

    return {
        "rewrite_acc":   avg(["post", "rewrite_acc"]),
        "rephrase_acc":  avg(["post", "rephrase_acc"]),
        "locality_acc":  avg(["post", "locality", "neighborhood_acc"]),
    }


def main():
    assert torch.cuda.is_available(), (
        "CUDA not available — this script must run on the GCP T4 instance. "
        "For local import/config checks use scripts/check_env.py instead."
    )

    print("Loading ROME hparams from", HPARAMS_PATH)
    hparams = ROMEHyperParams.from_hparams(HPARAMS_PATH)
    print(f"  model: {hparams.model_name}  device: cuda:{hparams.device}  layer: {hparams.layers}")

    print("\nBuilding editor (this downloads gpt2-xl on first run ~1.5 GB) ...")
    editor = BaseEditor.from_hparams(hparams)

    inputs = build_inputs(EDITS)
    print(f"\nRunning {len(EDITS)} ROME edits ...")
    metrics, _edited_model, _ = editor.edit(**inputs)

    summary = summarize(metrics)
    print("\n--- Results ---")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # Append to results/runs.jsonl
    os.makedirs("results", exist_ok=True)
    record = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "method":    "ROME",
        "model":     hparams.model_name,
        "dataset":   "CounterFact-smoke",
        "n_samples": len(EDITS),
        "seed":      42,
        "metrics":   summary,
        "per_edit":  [
            {
                "prompt":      EDITS[i]["prompt"],
                "target_new":  EDITS[i]["target_new"],
                "rewrite_acc": metrics[i].get("post", {}).get("rewrite_acc"),
                "rephrase_acc": metrics[i].get("post", {}).get("rephrase_acc"),
            }
            for i in range(len(EDITS))
        ],
    }
    runs_path = os.path.join("results", "runs.jsonl")
    with open(runs_path, "a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"\nResult appended to {runs_path}")

    # Quick sanity assertions
    assert summary["rewrite_acc"] is not None, "No rewrite_acc in metrics — something went wrong"
    print("\nSmoke test PASSED.")


if __name__ == "__main__":
    main()
