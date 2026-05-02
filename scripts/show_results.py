"""Print results/runs.jsonl as a summary table."""

import json, sys, os

RUNS_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "runs.jsonl")

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else RUNS_PATH
    try:
        with open(path) as f:
            runs = [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        print(f"No runs file at {path}")
        return

    if not runs:
        print("No runs recorded yet.")
        return

    header = f"{'#':<4} {'Timestamp':<22} {'Method':<8} {'Model':<10} {'Dataset':<18} {'N':>5} {'Rewrite':>8} {'Rephrase':>9} {'Locality':>9}"
    print(header)
    print("-" * len(header))

    for i, r in enumerate(runs):
        m = r.get("metrics", {})
        rewrite  = f"{m['rewrite_acc']:.3f}"  if m.get("rewrite_acc")  is not None else "N/A"
        rephrase = f"{m['rephrase_acc']:.3f}" if m.get("rephrase_acc") is not None else "N/A"
        locality = f"{m['locality_acc']:.3f}" if m.get("locality_acc") is not None else "N/A"
        ts = r.get("timestamp", "")[:19]
        print(
            f"{i:<4} {ts:<22} {r.get('method','?'):<8} {r.get('model','?'):<10} "
            f"{r.get('dataset','?'):<18} {r.get('n_samples',0):>5} "
            f"{rewrite:>8} {rephrase:>9} {locality:>9}"
        )

if __name__ == "__main__":
    main()
