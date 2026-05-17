"""
Small MQuAKE evaluation for edited GPT-2 XL models.

This is intended as a smoke/early-results script before larger sweeps. It applies
the requested MQuAKE rewrites for each sampled case, generates answers for the
edited single-hop and multi-hop questions, answer-matches against aliases, and
appends a summary to results/runs.jsonl.

Usage:
    python scripts/eval_mquake.py --method ROME --n_cases 5 --edit_mode one
    python scripts/eval_mquake.py --method MEMIT --n_cases 5 --edit_mode all
    python scripts/eval_mquake.py --method IKE --n_cases 5 --edit_mode all
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

from src.benchmarks.common import contains_answer
from src.benchmarks.mquake import load_records, record_to_eval_case, record_to_requests


HPARAMS = {
    "ROME": "configs/ROME/gpt2-xl",
    "MEMIT": "configs/MEMIT/gpt2-xl",
    "IKE": "configs/IKE/gpt2-xl",
}


def safe_key(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")


def partial_path_for(args) -> Path:
    os.makedirs("results/benchmark_partials", exist_ok=True)
    case_part = "sample"
    if args.case_ids:
        case_part = safe_key(f"ids_{args.case_ids}")
    return Path("results/benchmark_partials") / (
        f"mquake_{args.method.lower()}_{args.edit_mode}_"
        f"n{args.n_cases}_seed{args.seed}_{case_part}_tok{args.max_new_tokens}.jsonl"
    )


def load_partial_results(path: Path) -> dict[int, dict[str, Any]]:
    completed = {}
    if not path.exists():
        return completed
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            completed[int(row["sample_index"])] = row["result"]
    return completed


def append_partial_result(path: Path, sample_index: int, result: dict[str, Any]) -> None:
    with path.open("a") as f:
        f.write(json.dumps({"sample_index": sample_index, "result": result}) + "\n")
        f.flush()
        os.fsync(f.fileno())


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


def build_in_context_prompt(requests: list[dict[str, str]], query: str) -> str:
    facts = "\n".join(
        f"New Fact: {request['prompt']} {request['target_new']}."
        for request in requests
    )
    return (
        "Please acknowledge the following new facts and use them to answer the question:\n"
        f"{facts}\n"
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


def apply_case_edits(method: str, model, tok, hparams, requests: list[dict[str, str]]) -> None:
    if method == "ROME":
        for request in requests:
            apply_rome_to_model(
                model=model,
                tok=tok,
                request=[request],
                hparams=hparams,
                return_orig_weights=False,
            )
    elif method == "MEMIT":
        apply_memit_to_model(
            model=model,
            tok=tok,
            requests=requests,
            hparams=hparams,
            return_orig_weights=False,
        )
    elif method == "IKE":
        return
    else:
        raise ValueError(method)


def mean(values: list[bool]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def evaluate_case(
    model,
    tok,
    device: str,
    record: dict[str, Any],
    max_new_tokens: int,
    in_context_requests: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    single_hop = []
    for hop in record.get("new_single_hops", []):
        prompt = build_in_context_prompt(in_context_requests, hop["cloze"]) if in_context_requests else hop["cloze"]
        generation = generate(model, tok, prompt, device, max_new_tokens)
        passed = contains_answer(generation, hop["answer"], hop.get("answer_alias", []))
        single_hop.append(
            {
                "prompt": hop["cloze"],
                "model_prompt": prompt,
                "answers": [hop["answer"]] + list(hop.get("answer_alias", [])),
                "generation": generation,
                "passed": passed,
            }
        )

    multihop = []
    for question in record.get("questions", []):
        prompt = build_in_context_prompt(in_context_requests, question) if in_context_requests else question
        generation = generate(model, tok, prompt, device, max_new_tokens)
        passed = contains_answer(generation, record.get("new_answer", ""), record.get("new_answer_alias", []))
        multihop.append(
            {
                "prompt": question,
                "model_prompt": prompt,
                "answers": [record.get("new_answer", "")] + list(record.get("new_answer_alias", [])),
                "generation": generation,
                "passed": passed,
            }
        )

    return {
        "case_id": record.get("case_id"),
        "single_hop": single_hop,
        "multihop": multihop,
        "single_hop_acc": mean([x["passed"] for x in single_hop]),
        "multihop_acc": mean([x["passed"] for x in multihop]),
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    post_results = [result.get("post", result) for result in results]
    pre_results = [result["pre"] for result in results if "pre" in result]

    single = [item["passed"] for result in post_results for item in result["single_hop"]]
    multi = [item["passed"] for result in post_results for item in result["multihop"]]
    pre_single = [item["passed"] for result in pre_results for item in result["single_hop"]]
    pre_multi = [item["passed"] for result in pre_results for item in result["multihop"]]
    by_hops = defaultdict(list)
    for result in post_results:
        by_hops[len(result["single_hop"])].extend(item["passed"] for item in result["multihop"])

    metrics = {
        "pre_edited_fact_acc": mean(pre_single),
        "pre_multihop_acc": mean(pre_multi),
        "edited_fact_acc": mean(single),
        "multihop_acc": mean(multi),
        "multihop_acc_by_hop_count": {str(k): mean(v) for k, v in sorted(by_hops.items())},
    }
    if metrics["pre_edited_fact_acc"] is not None and metrics["edited_fact_acc"] is not None:
        metrics["delta_edited_fact_acc"] = round(metrics["edited_fact_acc"] - metrics["pre_edited_fact_acc"], 4)
    if metrics["pre_multihop_acc"] is not None and metrics["multihop_acc"] is not None:
        metrics["delta_multihop_acc"] = round(metrics["multihop_acc"] - metrics["pre_multihop_acc"], 4)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["ROME", "MEMIT", "IKE"], required=True)
    parser.add_argument("--data_path", default="data/mquake/MQuAKE-CF-3k-v2.json")
    parser.add_argument("--n_cases", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--edit_mode", choices=["one", "all"], default="one",
                        help="Use only the first requested rewrite, or all rewrites in each MQuAKE case")
    parser.add_argument("--case_ids", default=None,
                        help="Optional comma-separated MQuAKE case_id values to run instead of random sampling")
    parser.add_argument("--max_new_tokens", type=int, default=12)
    parser.add_argument("--no_resume", action="store_true",
                        help="Ignore any matching partial-result file and start this run from scratch")
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA required"
    if args.method == "ROME" and args.edit_mode == "all":
        print("WARNING: ROME all-edit mode applies requested rewrites sequentially within each case.")

    records = load_records(args.data_path)
    if args.case_ids:
        wanted = {int(x) for x in args.case_ids.split(",") if x.strip()}
        sample = [record for record in records if record.get("case_id") in wanted]
    else:
        sample = random.Random(args.seed).sample(records, min(args.n_cases, len(records)))

    if args.method == "ROME":
        hparams = ROMEHyperParams.from_hparams(HPARAMS["ROME"])
    elif args.method == "MEMIT":
        hparams = MEMITHyperParams.from_hparams(HPARAMS["MEMIT"])
    else:
        hparams = IKEHyperParams.from_hparams(HPARAMS["IKE"])
    device = f"cuda:{hparams.device}"
    print(f"Loading {hparams.model_name} for {args.method} on {device} ...")
    model, tok = load_model(hparams.model_name, device)

    partial_path = partial_path_for(args)
    if args.no_resume and partial_path.exists():
        partial_path.unlink()
    completed = {} if args.no_resume else load_partial_results(partial_path)
    if completed:
        print(f"Loaded {len(completed)} completed cases from {partial_path}")

    details = [completed[i] for i in sorted(completed) if i < len(sample)]
    for idx, record in enumerate(sample, start=1):
        sample_index = idx - 1
        if sample_index in completed:
            print(f"\nCase {idx}/{len(sample)} id={record.get('case_id')} already complete; skipping")
            continue
        eval_case = record_to_eval_case(record)
        requests = record_to_requests(record)
        if args.edit_mode == "one":
            requests = requests[:1]
        print(f"\nCase {idx}/{len(sample)} id={record.get('case_id')} edits={len(requests)} questions={len(eval_case['multihop_prompts'])}")
        pre = evaluate_case(model, tok, device, record, args.max_new_tokens)
        if args.method == "IKE":
            post = evaluate_case(model, tok, device, record, args.max_new_tokens, in_context_requests=requests)
            result = {"case_id": record.get("case_id"), "pre": pre, "post": post}
            completed[sample_index] = result
            append_partial_result(partial_path, sample_index, result)
        else:
            original = capture_weights(model, hparams)
            try:
                apply_case_edits(args.method, model, tok, hparams, requests)
                post = evaluate_case(model, tok, device, record, args.max_new_tokens)
                result = {"case_id": record.get("case_id"), "pre": pre, "post": post}
                completed[sample_index] = result
                append_partial_result(partial_path, sample_index, result)
            finally:
                restore_weights(model, original)

    details = [completed[i] for i in range(len(sample)) if i in completed]
    if len(details) != len(sample):
        raise RuntimeError(f"Expected {len(sample)} completed cases, found {len(details)}")

    metrics = summarize(details)
    print("\nMQuAKE summary")
    print(json.dumps(metrics, indent=2, sort_keys=True))

    os.makedirs("results/benchmark_details", exist_ok=True)
    stamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    detail_path = Path("results/benchmark_details") / f"mquake_{args.method.lower()}_{args.edit_mode}_{stamp}.json"
    detail_path.write_text(json.dumps(details, indent=2))

    run_record = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "method": args.method,
        "model": hparams.model_name,
        "dataset": f"MQuAKE-CF-3k-v2-{args.edit_mode}",
        "n_samples": len(sample),
        "seed": args.seed,
        "metrics": metrics,
        "details_path": str(detail_path),
        "partial_path": str(partial_path),
    }
    with open("results/runs.jsonl", "a") as f:
        f.write(json.dumps(run_record) + "\n")
    print(f"Details written to {detail_path}")
    print("Result appended to results/runs.jsonl")


if __name__ == "__main__":
    main()
