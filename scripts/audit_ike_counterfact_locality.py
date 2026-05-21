"""Capture decoded generations for representative IKE CounterFact locality failures.

The completed CounterFact baseline checkpoints under ``results/checkpoints/``
store per-example prompts and metric flags, but not decoded model text. This
script selects examples where IKE locality failed across k=4, k=8, and k=16,
then reruns only those prompts to save pre-context and post-context generations.

Usage:
    python3 scripts/audit_ike_counterfact_locality.py --n_examples 20
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "external", "EasyEdit"))

from easyeditor import IKEHyperParams
from easyeditor.models.ike import apply_ike_to_model, encode_ike_facts


HPARAMS_PATH = "configs/IKE/gpt2-xl"
DEFAULT_CHECKPOINTS = {
    16: "results/checkpoints/ike_counterfact-original-easyedit_n300_seed42.jsonl",
    4: "results/checkpoints/ike_counterfact-original-easyedit_n300_seed42_k4.jsonl",
    8: "results/checkpoints/ike_counterfact-original-easyedit_n300_seed42_k8.jsonl",
}


def load_jsonl(path: str) -> list[dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def locality_value(row: dict[str, Any], phase: str) -> float | None:
    value = row.get("metric", {}).get(phase, {}).get("locality", {}).get("neighborhood_acc")
    if isinstance(value, list):
        return float(sum(value) / len(value)) if value else None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def request_from_row(row: dict[str, Any]) -> dict[str, str]:
    req = row["metric"]["requested_rewrite"]
    return {
        "prompt": req["prompt"],
        "subject": req["subject"],
        "target_new": req["target_new"],
        "ground_truth": req["ground_truth"],
    }


def locality_from_row(row: dict[str, Any]) -> dict[str, str]:
    loc = row["metric"]["requested_rewrite"]["locality"]["neighborhood"]
    return {
        "prompt": loc["prompt"],
        "ground_truth": loc["ground_truth"],
    }


def select_stable_failures(checkpoint_paths: dict[int, str], n_examples: int) -> list[dict[str, Any]]:
    by_k = {k: load_jsonl(path) for k, path in checkpoint_paths.items()}
    by_case = {
        k: {
            int(row["case_id"]): row
            for row in rows
            if locality_value(row, "pre") == 1.0 and locality_value(row, "post") == 0.0
        }
        for k, rows in by_k.items()
    }
    common_case_ids = sorted(set.intersection(*(set(rows) for rows in by_case.values())))
    selected = []
    for case_id in common_case_ids[:n_examples]:
        base = by_case[16][case_id]
        selected.append(
            {
                "case_id": case_id,
                "sample_index": base["sample_index"],
                "subject": base["subject"],
                "request": request_from_row(base),
                "locality": locality_from_row(base),
            }
        )
    return selected


def embedding_path(hparams: IKEHyperParams, train_ds: list[dict[str, Any]]) -> str:
    safe_model_name = hparams.sentence_model_name.rsplit("/", 1)[-1]
    return os.path.join(
        hparams.results_dir,
        hparams.alg_name,
        "embedding",
        f"{safe_model_name}_{type(train_ds).__name__}_{len(train_ds)}.pkl",
    )


def ensure_ike_embeddings(hparams: IKEHyperParams, train_ds: list[dict[str, Any]], rebuild: bool) -> None:
    path = embedding_path(hparams, train_ds)
    if os.path.exists(path) and not rebuild:
        print(f"IKE retrieval embeddings found: {path}")
        return
    print("Building IKE retrieval embeddings ...")
    sentence_model = SentenceTransformer(hparams.sentence_model_name).to(f"cuda:{hparams.device}")
    encode_ike_facts(sentence_model, train_ds, hparams)
    print(f"IKE retrieval embeddings cached: {path}")


def generate(model, tok, prompt: str, device: str, max_new_tokens: int) -> dict[str, str]:
    inputs = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    generated_ids = out[0][inputs["input_ids"].shape[1]:]
    generation = tok.decode(generated_ids, skip_special_tokens=True).strip()
    first_token = tok.decode([generated_ids[0]], skip_special_tokens=True).strip() if len(generated_ids) else ""
    return {"first_token": first_token, "generation": generation}


def metric_token_prediction(model, tok, hparams: IKEHyperParams, icl_examples: list[str], target: str, x: str) -> dict[str, Any]:
    """Decode the greedy answer-token prediction used by EasyEdit's IKE metric.

    EasyEdit evaluates CounterFact locality by appending the expected answer to
    the prompt, taking greedy predictions over that answer span, and comparing
    post-context predicted token ids to pre-context predicted token ids. This is
    different from unconstrained free generation, so we save both views.
    """
    device = torch.device(f"cuda:{hparams.device}")
    target_ids = tok(" " + target + "\n", return_tensors="pt")["input_ids"].to(device)
    encodings = tok("".join(icl_examples) + f"{x} {target}", return_tensors="pt")
    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    ans = torch.argmax(logits, dim=-1)[:, -target_ids.size(1) : -1].squeeze()
    expected_ids = target_ids[:, :-1].squeeze()
    ans_ids = ans.detach().cpu().reshape(-1).tolist()
    expected_token_ids = expected_ids.detach().cpu().reshape(-1).tolist()
    return {
        "token_ids": ans_ids,
        "decoded": tok.decode(ans_ids, skip_special_tokens=True).strip(),
        "expected_token_ids": expected_token_ids,
        "expected_decoded": tok.decode(expected_token_ids, skip_special_tokens=True).strip(),
        "matches_expected_tokens": ans_ids == expected_token_ids,
    }


def contains_expected(generation: str, expected: str) -> bool:
    return expected.casefold() in generation.casefold()


def run_audit(args: argparse.Namespace) -> list[dict[str, Any]]:
    assert torch.cuda.is_available(), "CUDA required for GPT-2 XL IKE audit; run this on the GPU VM."

    checkpoint_paths = {16: args.k16_checkpoint, 4: args.k4_checkpoint, 8: args.k8_checkpoint}
    selected = select_stable_failures(checkpoint_paths, args.n_examples)
    if not selected:
        raise RuntimeError("No stable IKE locality failures found across k=4/8/16 checkpoints.")

    with open(args.data_path) as f:
        train_ds = json.load(f)

    hparams = IKEHyperParams.from_hparams(HPARAMS_PATH)
    ensure_ike_embeddings(hparams, train_ds, args.rebuild_embeddings)

    device = f"cuda:{hparams.device}"
    print(f"Loading model={hparams.model_name} on {device}")
    model = AutoModelForCausalLM.from_pretrained(hparams.model_name).to(device)
    tok = AutoTokenizer.from_pretrained(hparams.model_name)
    tok.pad_token = tok.eos_token

    results = []
    for idx, item in enumerate(selected, start=1):
        loc_prompt = item["locality"]["prompt"]
        expected = item["locality"]["ground_truth"]
        pre = generate(model, tok, loc_prompt, device, args.max_new_tokens)
        pre_metric = metric_token_prediction(model, tok, hparams, [""], expected, loc_prompt)
        row = {
            **item,
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "pre_context": {
                **pre,
                "passed_contains_expected": contains_expected(pre["generation"], expected),
                "metric_token_prediction": pre_metric,
            },
            "post_context_by_k": {},
        }
        print(f"[{idx}/{len(selected)}] case_id={item['case_id']} subject={item['subject']}")
        for k in args.k_values:
            hparams.k = k
            icl_examples = apply_ike_to_model(
                model=model,
                tok=tok,
                request=item["request"],
                hparams=hparams,
                train_ds=train_ds,
            )
            model_prompt = (
                "".join(icl_examples)
                + f"New Fact: {item['request']['prompt']} {item['request']['target_new']}\n"
                + f"Prompt: {loc_prompt}"
            )
            post = generate(model, tok, model_prompt, device, args.max_new_tokens)
            metric_x = f"New Fact: {item['request']['prompt']} {item['request']['target_new']}\nPrompt: {loc_prompt}"
            post_metric = metric_token_prediction(model, tok, hparams, icl_examples, expected, metric_x)
            row["post_context_by_k"][str(k)] = {
                "num_icl_examples": len(icl_examples),
                "model_prompt": model_prompt if args.include_model_prompts else None,
                **post,
                "passed_contains_expected": contains_expected(post["generation"], expected),
                "contains_target_new": contains_expected(post["generation"], item["request"]["target_new"]),
                "metric_token_prediction": post_metric,
                "metric_matches_pre_context": post_metric["token_ids"] == pre_metric["token_ids"],
            }
        results.append(row)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data/counterfact/counterfact-original-easyedit.json")
    parser.add_argument("--output_path", default="results/ike_counterfact_locality_examples.json")
    parser.add_argument("--n_examples", type=int, default=20)
    parser.add_argument("--max_new_tokens", type=int, default=12)
    parser.add_argument("--k_values", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--k16_checkpoint", default=DEFAULT_CHECKPOINTS[16])
    parser.add_argument("--k4_checkpoint", default=DEFAULT_CHECKPOINTS[4])
    parser.add_argument("--k8_checkpoint", default=DEFAULT_CHECKPOINTS[8])
    parser.add_argument("--include_model_prompts", action="store_true")
    parser.add_argument("--rebuild_embeddings", action="store_true")
    args = parser.parse_args()

    rows = run_audit(args)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, indent=2))
    print(f"Wrote {len(rows)} IKE locality examples to {output_path}")


if __name__ == "__main__":
    main()
