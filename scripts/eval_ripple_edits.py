"""
Small RippleEdits criterion evaluation for edited GPT-2 XL models.

Usage:
    python scripts/eval_ripple_edits.py --method ROME --n_cases 5
    python scripts/eval_ripple_edits.py --method MEMIT --n_cases 5 --subset POPULAR
    python scripts/eval_ripple_edits.py --method IKE --n_cases 5 --subset POPULAR
"""

import argparse
import datetime
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "external", "EasyEdit"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from easyeditor import IKEHyperParams, MEMITHyperParams, ROMEHyperParams
from easyeditor.models.memit import apply_memit_to_model
from easyeditor.models.rome.rome_main import apply_rome_to_model
from easyeditor.util import nethook

from src.benchmarks.ripple_edits import CRITERIA, edit_to_request, iter_queries, iter_tests, load_records, score_query_generation


HPARAMS = {
    "ROME": "configs/ROME/gpt2-xl",
    "MEMIT": "configs/MEMIT/gpt2-xl",
    "IKE": "configs/IKE/gpt2-xl",
}


def load_model(model_name: str, device: str):
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.pad_token = tok.eos_token
    return model, tok


def generate(model, tok, prompt: str, device: str, max_new_tokens: int) -> str:
    inputs = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    generated = out[0][inputs["input_ids"].shape[1]:]
    return tok.decode(generated, skip_special_tokens=True).strip()


def build_in_context_prompt(request: dict[str, str], query: str) -> str:
    return (
        "Please acknowledge the following new fact and use it to answer the question:\n"
        f"New Fact: {request['prompt']} {request['target_new']}.\n"
        f"Prompt: {query}"
    )


def capture_weights(model, hparams) -> dict[str, torch.Tensor]:
    return {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        ).detach().clone()
        for layer in hparams.layers
    }


def restore_weights(model, weights_copy: dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for name, original in weights_copy.items():
            weight = nethook.get_parameter(model, name)
            weight[...] = original.to(weight.device)


def apply_edit(method: str, model, tok, hparams, request: dict[str, str]) -> None:
    if method == "ROME":
        apply_rome_to_model(model=model, tok=tok, request=[request], hparams=hparams, return_orig_weights=False)
    elif method == "MEMIT":
        apply_memit_to_model(model=model, tok=tok, requests=[request], hparams=hparams, return_orig_weights=False)
    elif method == "IKE":
        return
    else:
        raise ValueError(method)


def mean(values: list[bool]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def evaluate_record(
    model,
    tok,
    device: str,
    record: dict[str, Any],
    max_new_tokens: int,
    in_context_request: dict[str, str] | None = None,
) -> dict[str, Any]:
    criterion_results = defaultdict(list)
    generations = []
    for criterion, test in iter_tests(record):
        for query in iter_queries(test):
            prompt = build_in_context_prompt(in_context_request, query["prompt"]) if in_context_request else query["prompt"]
            generation = generate(model, tok, prompt, device, max_new_tokens)
            passed = score_query_generation(query, generation)
            criterion_results[criterion].append(passed)
            generations.append(
                {
                    "criterion": criterion,
                    "prompt": query["prompt"],
                    "model_prompt": prompt,
                    "answers": query.get("answers", []),
                    "generation": generation,
                    "passed": passed,
                }
            )
    return {
        "edit": record.get("edit", {}),
        "request": edit_to_request(record),
        "criteria": {name: mean(criterion_results[name]) for name in CRITERIA if criterion_results.get(name)},
        "generations": generations,
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    post_results = [result.get("post", result) for result in results]
    pre_results = [result["pre"] for result in results if "pre" in result]

    values = defaultdict(list)
    for result in post_results:
        for generation in result["generations"]:
            values[generation["criterion"]].append(generation["passed"])
    metrics = {f"{criterion}_acc": mean(values[criterion]) for criterion in CRITERIA if values.get(criterion)}
    all_values = [passed for criterion_values in values.values() for passed in criterion_values]
    metrics["overall_acc"] = mean(all_values)
    metrics["available_criteria"] = [criterion for criterion in CRITERIA if values.get(criterion)]

    pre_values = defaultdict(list)
    for result in pre_results:
        for generation in result["generations"]:
            pre_values[generation["criterion"]].append(generation["passed"])
    pre_all_values = [passed for criterion_values in pre_values.values() for passed in criterion_values]
    metrics["pre_overall_acc"] = mean(pre_all_values)
    if metrics["pre_overall_acc"] is not None and metrics["overall_acc"] is not None:
        metrics["delta_overall_acc"] = round(metrics["overall_acc"] - metrics["pre_overall_acc"], 4)
    for criterion in CRITERIA:
        key = f"{criterion}_acc"
        pre_key = f"pre_{criterion}_acc"
        if pre_values.get(criterion):
            metrics[pre_key] = mean(pre_values[criterion])
            if key in metrics and metrics[pre_key] is not None:
                metrics[f"delta_{criterion}_acc"] = round(metrics[key] - metrics[pre_key], 4)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["ROME", "MEMIT", "IKE"], required=True)
    parser.add_argument("--subset", choices=["POPULAR", "RANDOM", "RECENT"], default="POPULAR")
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--n_cases", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=12)
    parser.add_argument("--allow_non_ascii_targets", action="store_true",
                        help="Include edits whose old/new target strings contain non-ASCII characters")
    parser.add_argument("--relations", default=None,
                        help="Optional comma-separated relation names to include before sampling")
    parser.add_argument("--require_criteria", default=None,
                        help="Optional comma-separated criteria that must have at least one test query")
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA required"
    data_path = args.data_path or f"data/ripple_edits/{args.subset}.json"
    records = load_records(data_path)
    if args.relations:
        allowed_relations = {x.strip() for x in args.relations.split(",") if x.strip()}
        records = [record for record in records if record.get("edit", {}).get("relation") in allowed_relations]
    if args.require_criteria:
        required_criteria = {x.strip() for x in args.require_criteria.split(",") if x.strip()}
        records = [
            record
            for record in records
            if all(
                any(list(iter_queries(test)) for test in record.get(criterion, []) or [])
                for criterion in required_criteria
            )
        ]
    usable = []
    for record in records:
        try:
            request = edit_to_request(record)
            if not args.allow_non_ascii_targets:
                if not request["target_new"].isascii() or not request["ground_truth"].isascii():
                    continue
            usable.append(record)
        except ValueError:
            continue
    if not usable:
        raise ValueError(f"No usable RippleEdits records found in {data_path}")
    sample = random.Random(args.seed).sample(usable, min(args.n_cases, len(usable)))

    if args.method == "ROME":
        hparams = ROMEHyperParams.from_hparams(HPARAMS["ROME"])
    elif args.method == "MEMIT":
        hparams = MEMITHyperParams.from_hparams(HPARAMS["MEMIT"])
    else:
        hparams = IKEHyperParams.from_hparams(HPARAMS["IKE"])
    device = f"cuda:{hparams.device}"
    print(f"Loading {hparams.model_name} for {args.method} on {device} ...")
    model, tok = load_model(hparams.model_name, device)

    details = []
    for idx, record in enumerate(sample, start=1):
        request = edit_to_request(record)
        print(f"\nCase {idx}/{len(sample)} relation={record.get('edit', {}).get('relation')} subject={request['subject']!r}")
        pre = evaluate_record(model, tok, device, record, args.max_new_tokens)
        if args.method == "IKE":
            post = evaluate_record(model, tok, device, record, args.max_new_tokens, in_context_request=request)
            details.append({"request": request, "pre": pre, "post": post})
        else:
            original = capture_weights(model, hparams)
            try:
                apply_edit(args.method, model, tok, hparams, request)
                post = evaluate_record(model, tok, device, record, args.max_new_tokens)
                details.append({"request": request, "pre": pre, "post": post})
            finally:
                restore_weights(model, original)

    metrics = summarize(details)
    print("\nRippleEdits summary")
    print(json.dumps(metrics, indent=2, sort_keys=True))

    os.makedirs("results/benchmark_details", exist_ok=True)
    stamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    detail_path = Path("results/benchmark_details") / f"ripple_{args.subset.lower()}_{args.method.lower()}_{stamp}.json"
    detail_path.write_text(json.dumps(details, indent=2))

    run_record = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "method": args.method,
        "model": hparams.model_name,
        "dataset": f"RippleEdits-{args.subset}",
        "n_samples": len(sample),
        "seed": args.seed,
        "metrics": metrics,
        "details_path": str(detail_path),
    }
    with open("results/runs.jsonl", "a") as f:
        f.write(json.dumps(run_record) + "\n")
    print(f"Details written to {detail_path}")
    print("Result appended to results/runs.jsonl")


if __name__ == "__main__":
    main()
