"""
Diagnostic probe evaluation — runs the hand-curated probe set against an edited model.

For each edit case in the probe set:
  1. Apply the edit to the base model (ROME or MEMIT).
  2. Run all probes for that edit case.
  3. Record: pre-edit prediction, post-edit prediction, pass/fail vs expected.
  4. Restore the base model before the next edit case.

Results are written to results/probe_results_225.jsonl and a summary table is printed.
Existing rows in the output file are treated as checkpoints: rerunning the same
method/output path skips completed probes and continues with missing rows.

Usage:
    conda activate cs263-project
    cd ~/cs263-knowledge-editing

    # ROME probes:
    python scripts/run_probes.py --method ROME --output_path results/probe_results_225.jsonl \\
        2>&1 | tee logs/probes_rome_$(date +%Y%m%d_%H%M%S).log

    # MEMIT probes:
    python scripts/run_probes.py --method MEMIT --output_path results/probe_results_225.jsonl \\
        2>&1 | tee logs/probes_memit_$(date +%Y%m%d_%H%M%S).log

    # IKE probes:
    python scripts/run_probes.py --method IKE --data_path data/counterfact/counterfact-edit.json \\
        --output_path results/probe_results_225.jsonl \\
        2>&1 | tee logs/probes_ike_$(date +%Y%m%d_%H%M%S).log

Notes:
    - IKE support uses EasyEdit retrieval examples as a prompt prefix; it does not
      modify model weights.
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

from easyeditor import ROMEHyperParams, MEMITHyperParams, IKEHyperParams
from easyeditor.models.rome.rome_main   import apply_rome_to_model
from easyeditor.models.memit.memit_main import apply_memit_to_model
from easyeditor.models.ike import apply_ike_to_model, encode_ike_facts
from easyeditor.util import nethook
from sentence_transformers import SentenceTransformer

from src.probes.probe_set import PROBES, EDIT_CASES, Probe, EditCase

HPARAMS = {
    "ROME":  "configs/ROME/gpt2-xl",
    "MEMIT": "configs/MEMIT/gpt2-xl",
    "IKE":   "configs/IKE/gpt2-xl",
}


def load_completed_results(output_path: str, method: str) -> dict[str, dict]:
    """Load completed probe rows for this method, keyed by probe_id."""
    completed: dict[str, dict] = {}
    if not os.path.exists(output_path):
        return completed

    with open(output_path) as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{output_path}:{line_no}: invalid JSON checkpoint row") from exc
            if row.get("method") != method:
                continue
            probe_id = row.get("probe_id")
            if probe_id:
                completed[probe_id] = row
    return completed


def append_results(output_path: str, rows: list[dict]) -> None:
    if not rows:
        return
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "a") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def add_stats(row: dict, category_stats: dict[str, dict], type_stats: dict[str, dict]) -> None:
    cat = row["category"]
    if cat not in category_stats:
        category_stats[cat] = {"n": 0, "pre_pass": 0, "post_pass": 0}
    category_stats[cat]["n"] += 1
    category_stats[cat]["pre_pass"] += int(row["pre_edit"]["passed"])
    category_stats[cat]["post_pass"] += int(row["post_edit"]["passed"])

    probe_type = row["probe_type"]
    if probe_type not in type_stats:
        type_stats[probe_type] = {"n": 0, "pre_pass": 0, "post_pass": 0}
    type_stats[probe_type]["n"] += 1
    type_stats[probe_type]["pre_pass"] += int(row["pre_edit"]["passed"])
    type_stats[probe_type]["post_pass"] += int(row["post_edit"]["passed"])


# ── Model helpers ────────────────────────────────────────────────────────────

def load_model(model_name: str, device: str):
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tok   = AutoTokenizer.from_pretrained(model_name)
    tok.pad_token = tok.eos_token
    return model, tok


def load_train_ds(data_path: str) -> list[dict]:
    with open(data_path) as f:
        return json.load(f)


def ensure_ike_embeddings(hparams: IKEHyperParams, train_ds: list[dict], rebuild: bool) -> None:
    safe_model_name = hparams.sentence_model_name.rsplit("/", 1)[-1]
    path = os.path.join(
        hparams.results_dir,
        hparams.alg_name,
        "embedding",
        f"{safe_model_name}_{type(train_ds).__name__}_{len(train_ds)}.pkl",
    )
    if os.path.exists(path) and not rebuild:
        print(f"  IKE retrieval embeddings found: {path}")
        return

    print("  Building IKE retrieval embeddings ...")
    print(f"  sentence_model={hparams.sentence_model_name}")
    sentence_model = SentenceTransformer(hparams.sentence_model_name).to(f"cuda:{hparams.device}")
    encode_ike_facts(sentence_model, train_ds, hparams)
    print(f"  IKE retrieval embeddings cached: {path}")


def restore_weights(model: AutoModelForCausalLM, weights_copy: dict) -> None:
    with torch.no_grad():
        for w_name, orig in weights_copy.items():
            weight = nethook.get_parameter(model, w_name)
            weight[...] = orig.to(weight.device)


def apply_edit(method: str, model, tok, hparams, edit_case: EditCase, train_ds=None) -> dict:
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
    elif method == "IKE":
        assert train_ds is not None, "IKE requires a retrieval pool"
        weights_copy = apply_ike_to_model(
            model=model, tok=tok,
            request=request, hparams=hparams,
            train_ds=train_ds,
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


def run_probe(probe: Probe, model, tok, device: str, prompt_prefix: str = "") -> dict:
    first_token, generation = predict(model, tok, f"{prompt_prefix}{probe.probe_prompt}", device)
    passed = check_probe(probe, first_token, generation)
    return {
        "first_token": first_token,
        "generation":  generation,
        "passed":      passed,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=["ROME", "MEMIT", "IKE"],
                        help="Editing method to evaluate")
    parser.add_argument("--data_path", default="data/counterfact/counterfact-edit.json",
                        help="CounterFact file used for the IKE retrieval pool")
    parser.add_argument("--edit_keys", default=None,
                        help="Comma-separated edit_keys to run (default: all)")
    parser.add_argument("--output_path", default="results/probe_results_225.jsonl",
                        help="JSONL path to append probe results")
    parser.add_argument("--no_resume", action="store_true",
                        help="Do not skip rows already present in --output_path")
    parser.add_argument("--rebuild_embeddings", action="store_true",
                        help="Recompute cached IKE retrieval embeddings before evaluation")
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA required — run on GCP T4"

    filter_keys = set(args.edit_keys.split(",")) if args.edit_keys else None

    # Load hparams
    if args.method == "ROME":
        hparams = ROMEHyperParams.from_hparams(HPARAMS["ROME"])
    elif args.method == "MEMIT":
        hparams = MEMITHyperParams.from_hparams(HPARAMS["MEMIT"])
    else:
        hparams = IKEHyperParams.from_hparams(HPARAMS["IKE"])

    device = f"cuda:{hparams.device}"
    print(f"Loading {hparams.model_name} ...")
    model, tok = load_model(hparams.model_name, device)
    train_ds = None
    if args.method == "IKE":
        train_ds = load_train_ds(args.data_path)
        ensure_ike_embeddings(hparams, train_ds, args.rebuild_embeddings)

    # Group probes by edit case
    probes_by_edit: dict[str, list[Probe]] = {}
    for p in PROBES:
        if filter_keys and p.edit_key not in filter_keys:
            continue
        probes_by_edit.setdefault(p.edit_key, []).append(p)

    all_results = []
    category_stats: dict[str, dict] = {}
    type_stats: dict[str, dict] = {}
    completed = {} if args.no_resume else load_completed_results(args.output_path, args.method)
    selected_probe_ids = {p.probe_id for probes in probes_by_edit.values() for p in probes}
    resumed_rows = [
        row for probe_id, row in completed.items()
        if probe_id in selected_probe_ids
    ]
    for row in resumed_rows:
        add_stats(row, category_stats, type_stats)
    if resumed_rows:
        print(
            f"Resuming {args.method}: found {len(resumed_rows)}/"
            f"{len(selected_probe_ids)} completed probes in {args.output_path}"
        )

    for edit_key, probes in probes_by_edit.items():
        pending_probes = [p for p in probes if p.probe_id not in completed]
        if not pending_probes:
            print(f"\n── Edit: {edit_key} already complete ({len(probes)} probes); skipping")
            continue

        edit_case = EDIT_CASES[edit_key]
        print(f"\n── Edit: {edit_key} ({edit_case.subject}: {edit_case.ground_truth} → {edit_case.target_new})")
        if args.method == "IKE":
            print("   Retrieving IKE context ...")
        else:
            print(f"   Applying {args.method} edit ...")

        # Pre-edit baseline
        pre_results = {p.probe_id: run_probe(p, model, tok, device) for p in pending_probes}

        state = apply_edit(args.method, model, tok, hparams, edit_case, train_ds=train_ds)
        icl_examples = state if args.method == "IKE" else None

        ts = datetime.datetime.utcnow().isoformat()
        try:
            # Post-edit evaluation
            for probe in pending_probes:
                if args.method == "IKE":
                    prefix = "".join(icl_examples)
                    post = run_probe(probe, model, tok, device, prompt_prefix=prefix)
                else:
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
                    "timestamp": ts,
                }
                all_results.append(result)
                completed[probe.probe_id] = result
                add_stats(result, category_stats, type_stats)
                append_results(args.output_path, [result])

                status = "✓" if post["passed"] else "✗"
                print(f"   {status} [{probe.category[:6]}/{probe.probe_type[:8]}] {probe.probe_id}: "
                      f"'{post['first_token']}' ... (pre: '{pre['first_token']}')")
            print(f"   checkpointed {len(pending_probes)} rows to {args.output_path}")
        finally:
            if args.method in {"ROME", "MEMIT"}:
                restore_weights(model, state)

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

    if all_results:
        print(f"\nProbe results checkpointed to {args.output_path}")
    else:
        print(f"\nNo new probe rows written; {args.method} is already complete in {args.output_path}")


if __name__ == "__main__":
    main()
