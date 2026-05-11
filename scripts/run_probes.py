"""
Diagnostic probe evaluation — runs the hand-curated probe set against an edited model.

For each edit case in the probe set:
  1. Apply the edit to the base model (ROME or MEMIT).
  2. Run all probes for that edit case.
  3. Record: pre-edit prediction, post-edit prediction, pass/fail vs expected.
  4. Restore the base model before the next edit case.

Results are written to results/probe_results.jsonl and a summary table is printed.

Usage:
    conda activate cs263-project
    cd ~/cs263-knowledge-editing

    # ROME probes:
    python scripts/run_probes.py --method ROME \\
        2>&1 | tee logs/probes_rome_$(date +%Y%m%d_%H%M%S).log

    # MEMIT probes:
    python scripts/run_probes.py --method MEMIT \\
        2>&1 | tee logs/probes_memit_$(date +%Y%m%d_%H%M%S).log

Notes:
    - IKE support is scaffolded but not yet implemented; use --method ROME or MEMIT.
    - "pass" means the model's first predicted token (or short generation) matches
      expected_first_token / expected_contains.  Both checks are case-insensitive.
    - Pre-edit predictions are captured before the edit is applied, allowing
      direct comparison of base model vs. edited model on each probe.
    - probe_type distinguishes implicit edit tests from prompts that condition on
      the target value or explicitly state the edited fact.
"""

import sys, json, datetime, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "external", "EasyEdit"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from easyeditor import ROMEHyperParams, MEMITHyperParams
from easyeditor.models.rome.rome_main   import apply_rome_to_model
from easyeditor.models.memit.memit_main import apply_memit_to_model
from easyeditor.util import nethook

from src.probes.probe_set import PROBES, EDIT_CASES, Probe, EditCase

HPARAMS = {
    "ROME":  "configs/ROME/gpt2-xl",
    "MEMIT": "configs/MEMIT/gpt2-xl",
}


# ── Model helpers ────────────────────────────────────────────────────────────

def load_model(model_name: str, device: str):
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tok   = AutoTokenizer.from_pretrained(model_name)
    tok.pad_token = tok.eos_token
    return model, tok


def restore_weights(model: AutoModelForCausalLM, weights_copy: dict) -> None:
    with torch.no_grad():
        for w_name, orig in weights_copy.items():
            weight = nethook.get_parameter(model, w_name)
            weight[...] = orig.to(weight.device)


def apply_edit(method: str, model, tok, hparams, edit_case: EditCase) -> dict:
    request = {
        "prompt":       edit_case.prompt,
        "subject":      edit_case.subject,
        "target_new":   edit_case.target_new,
        "ground_truth": edit_case.ground_truth,
    }
    if method == "ROME":
        _, weights_copy = apply_rome_to_model(
            model=model, tok=tok,
            request=[request], hparams=hparams,
            return_orig_weights=True,
        )
    elif method == "MEMIT":
        _, weights_copy = apply_memit_to_model(
            model=model, tok=tok,
            requests=[request], hparams=hparams,
            return_orig_weights=True,
        )
    else:
        raise ValueError(f"Unsupported method: {method}")
    return weights_copy


# ── Probe evaluation ─────────────────────────────────────────────────────────

def predict(model, tok, prompt: str, device: str, max_new_tokens: int = 10) -> tuple[str, str]:
    """Returns (first_token_str, short_generation_str)."""
    inp = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inp,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    generated_ids = out[0][inp["input_ids"].shape[1]:]
    generated_text = tok.decode(generated_ids, skip_special_tokens=True).strip()
    first_token = tok.decode([generated_ids[0]], skip_special_tokens=True).strip() if len(generated_ids) > 0 else ""
    return first_token, generated_text


def check_probe(probe: Probe, first_token: str, generation: str) -> bool:
    if probe.expected_first_token is not None:
        if probe.expected_first_token.lower() in first_token.lower():
            return True
    if probe.expected_contains is not None:
        if probe.expected_contains.lower() in generation.lower():
            return True
    return False


def run_probe(probe: Probe, model, tok, device: str) -> dict:
    first_token, generation = predict(model, tok, probe.probe_prompt, device)
    passed = check_probe(probe, first_token, generation)
    return {
        "first_token": first_token,
        "generation":  generation,
        "passed":      passed,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=["ROME", "MEMIT"],
                        help="Editing method to evaluate")
    parser.add_argument("--edit_keys", default=None,
                        help="Comma-separated edit_keys to run (default: all)")
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA required — run on GCP T4"

    filter_keys = set(args.edit_keys.split(",")) if args.edit_keys else None

    # Load hparams
    if args.method == "ROME":
        hparams = ROMEHyperParams.from_hparams(HPARAMS["ROME"])
    else:
        hparams = MEMITHyperParams.from_hparams(HPARAMS["MEMIT"])

    device = f"cuda:{hparams.device}"
    print(f"Loading {hparams.model_name} ...")
    model, tok = load_model(hparams.model_name, device)

    # Group probes by edit case
    probes_by_edit: dict[str, list[Probe]] = {}
    for p in PROBES:
        if filter_keys and p.edit_key not in filter_keys:
            continue
        probes_by_edit.setdefault(p.edit_key, []).append(p)

    all_results = []
    category_stats: dict[str, dict] = {}
    type_stats: dict[str, dict] = {}

    for edit_key, probes in probes_by_edit.items():
        edit_case = EDIT_CASES[edit_key]
        print(f"\n── Edit: {edit_key} ({edit_case.subject}: {edit_case.ground_truth} → {edit_case.target_new})")
        print(f"   Applying {args.method} edit ...")

        # Pre-edit baseline
        pre_results = {p.probe_id: run_probe(p, model, tok, device) for p in probes}

        weights_copy = apply_edit(args.method, model, tok, hparams, edit_case)

        try:
            # Post-edit evaluation
            for probe in probes:
                post = run_probe(probe, model, tok, device)
                pre  = pre_results[probe.probe_id]

                result = {
                    "probe_id":    probe.probe_id,
                    "edit_key":    edit_key,
                    "method":      args.method,
                    "category":    probe.category,
                    "probe_type":  probe.probe_type,
                    "probe_prompt": probe.probe_prompt,
                    "expected_first_token": probe.expected_first_token,
                    "expected_contains":    probe.expected_contains,
                    "pre_edit": {
                        "first_token": pre["first_token"],
                        "generation":  pre["generation"],
                        "passed":      pre["passed"],
                    },
                    "post_edit": {
                        "first_token": post["first_token"],
                        "generation":  post["generation"],
                        "passed":      post["passed"],
                    },
                    "note": probe.note,
                }
                all_results.append(result)

                # Update category stats
                cat = probe.category
                if cat not in category_stats:
                    category_stats[cat] = {"n": 0, "pre_pass": 0, "post_pass": 0}
                category_stats[cat]["n"]         += 1
                category_stats[cat]["pre_pass"]  += int(pre["passed"])
                category_stats[cat]["post_pass"] += int(post["passed"])

                probe_type = probe.probe_type
                if probe_type not in type_stats:
                    type_stats[probe_type] = {"n": 0, "pre_pass": 0, "post_pass": 0}
                type_stats[probe_type]["n"]         += 1
                type_stats[probe_type]["pre_pass"]  += int(pre["passed"])
                type_stats[probe_type]["post_pass"] += int(post["passed"])

                status = "✓" if post["passed"] else "✗"
                print(f"   {status} [{probe.category[:6]}/{probe.probe_type[:8]}] {probe.probe_id}: "
                      f"'{post['first_token']}' ... (pre: '{pre['first_token']}')")
        finally:
            restore_weights(model, weights_copy)

    # Summary table
    print("\n" + "=" * 68)
    print(f"  Probe results — {args.method} on {len(all_results)} probes")
    print("=" * 68)
    print(f"  {'Category':<22} {'N':>4}  {'Pre':>6}  {'Post':>6}  {'Δ':>6}")
    print("  " + "-" * 48)
    total_n = total_pre = total_post = 0
    for cat, s in sorted(category_stats.items()):
        n, pre, post = s["n"], s["pre_pass"], s["post_pass"]
        total_n += n; total_pre += pre; total_post += post
        pre_pct  = pre  / n if n else 0
        post_pct = post / n if n else 0
        delta = post_pct - pre_pct
        print(f"  {cat:<22} {n:>4}  {pre_pct:>5.1%}  {post_pct:>5.1%}  {delta:>+5.1%}")
    print("  " + "-" * 48)
    if total_n:
        pre_tot  = total_pre  / total_n
        post_tot = total_post / total_n
        print(f"  {'TOTAL':<22} {total_n:>4}  {pre_tot:>5.1%}  {post_tot:>5.1%}  {post_tot - pre_tot:>+5.1%}")
    print("=" * 68)

    print("\n" + "=" * 68)
    print(f"  Probe results by probe_type")
    print("=" * 68)
    print(f"  {'Probe type':<28} {'N':>4}  {'Pre':>6}  {'Post':>6}  {'Δ':>6}")
    print("  " + "-" * 54)
    for probe_type, s in sorted(type_stats.items()):
        n, pre, post = s["n"], s["pre_pass"], s["post_pass"]
        pre_pct  = pre  / n if n else 0
        post_pct = post / n if n else 0
        print(f"  {probe_type:<28} {n:>4}  {pre_pct:>5.1%}  {post_pct:>5.1%}  {post_pct - pre_pct:>+5.1%}")
    print("=" * 68)

    # Save
    os.makedirs("results", exist_ok=True)
    out_path = "results/probe_results.jsonl"
    ts = datetime.datetime.utcnow().isoformat()
    with open(out_path, "a") as f:
        for r in all_results:
            r["timestamp"] = ts
            f.write(json.dumps(r) + "\n")
    print(f"\nProbe results appended to {out_path}")


if __name__ == "__main__":
    main()
