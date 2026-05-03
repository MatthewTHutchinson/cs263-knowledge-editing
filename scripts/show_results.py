"""
Summarize results/runs.jsonl and optionally results/probe_results.jsonl.

Usage:
    python scripts/show_results.py               # baseline runs table
    python scripts/show_results.py --probes      # add probe summary
    python scripts/show_results.py --plot        # ASCII bar chart of rewrite_acc by method
    python scripts/show_results.py --all         # everything
"""

import json, sys, os, argparse
from collections import defaultdict

RUNS_PATH   = os.path.join(os.path.dirname(__file__), "..", "results", "runs.jsonl")
PROBES_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "probe_results.jsonl")

PAPER_TARGETS = {
    "ROME":        {"rewrite_acc": 0.996, "rephrase_acc": 0.948, "locality_acc": 0.722},
    "MEMIT":       {"rewrite_acc": 0.998, "rephrase_acc": 0.732, "locality_acc": 0.640},
    "MEMIT-batch": {"rewrite_acc": 0.995, "rephrase_acc": 0.732, "locality_acc": 0.640},
    "IKE":         {"rewrite_acc": 0.678, "rephrase_acc": 0.507, "locality_acc": 0.575},
}

REPHRASE_RELATIVE = {"ROME", "MEMIT", "MEMIT-batch", "IKE"}


def load_jsonl(path: str) -> list[dict]:
    try:
        with open(path) as f:
            return [json.loads(l) for l in f if l.strip()]
    except FileNotFoundError:
        return []


def fmt(val, width=7) -> str:
    if val is None:
        return f"{'N/A':>{width}}"
    return f"{val:.3f}".rjust(width)


def delta_str(ours, paper, width=7) -> str:
    if ours is None or paper is None:
        return f"{'---':>{width}}"
    d = ours - paper
    return f"{d:>+{width}.3f}"


def show_baseline_table(runs: list[dict]) -> None:
    if not runs:
        print("No baseline runs recorded yet.")
        return

    print("\n" + "=" * 100)
    print("  BASELINE RUNS")
    print("=" * 100)
    hdr = (f"  {'#':<3} {'Date':<11} {'Method':<12} {'Dataset':<24} {'N':>5} "
           f"{'Rewrite':>8}{'Δ':>7}  {'Rephrase':>9}{'Δ':>7}  {'Locality':>9}{'Δ':>7}")
    print(hdr)
    print("  " + "-" * 96)

    for i, r in enumerate(runs):
        m     = r.get("metrics", {})
        method = r.get("method", "?")
        paper  = PAPER_TARGETS.get(method, {})

        rw = m.get("rewrite_acc")
        rp = m.get("rephrase_acc")
        lo = m.get("locality_acc")

        rp_note = "*" if method in REPHRASE_RELATIVE else " "

        print(
            f"  {i:<3} {r.get('timestamp','')[:10]:<11} {method:<12} "
            f"{r.get('dataset','?'):<24} {r.get('n_samples',0):>5} "
            f"{fmt(rw)}{delta_str(rw, paper.get('rewrite_acc')):>7}  "
            f"{fmt(rp)}{delta_str(rp, paper.get('rephrase_acc')):>7}{rp_note} "
            f"{fmt(lo)}{delta_str(lo, paper.get('locality_acc')):>7}"
        )

    print("  " + "-" * 96)
    print("  Δ = ours − paper target.  * rephrase_acc is relative-only (noisy EasyEdit prompts).")
    print("=" * 100)


def show_method_summary(runs: list[dict]) -> None:
    """Print best single-edit run per method with paper comparison."""
    seen: dict[str, dict] = {}
    for r in runs:
        method = r.get("method", "?")
        if "batch" in method.lower():
            continue
        if method not in seen:
            seen[method] = r
        elif (r.get("metrics", {}).get("rewrite_acc") or 0) > \
             (seen[method].get("metrics", {}).get("rewrite_acc") or 0):
            seen[method] = r

    if not seen:
        return

    print("\n" + "=" * 72)
    print("  METHOD COMPARISON (best single-edit run per method)")
    print("=" * 72)
    print(f"  {'Method':<12} {'Rewrite':>8} {'Paper':>7} {'Δ':>7}  "
          f"{'Locality':>9} {'Paper':>7} {'Δ':>7}")
    print("  " + "-" * 64)
    for method, r in sorted(seen.items()):
        m     = r.get("metrics", {})
        paper = PAPER_TARGETS.get(method, {})
        rw = m.get("rewrite_acc")
        lo = m.get("locality_acc")
        print(f"  {method:<12} {fmt(rw)} {fmt(paper.get('rewrite_acc'))} "
              f"{delta_str(rw, paper.get('rewrite_acc'))}  "
              f"{fmt(lo)} {fmt(paper.get('locality_acc'))} "
              f"{delta_str(lo, paper.get('locality_acc'))}")
    print("=" * 72)


def show_batch_sweep(runs: list[dict]) -> None:
    """Show MEMIT-batch results sorted by batch size."""
    batch_runs = [r for r in runs if r.get("method") == "MEMIT-batch"]
    if not batch_runs:
        return

    batch_runs.sort(key=lambda r: r.get("n_samples", 0))

    print("\n" + "=" * 60)
    print("  MEMIT BATCH SWEEP")
    print("=" * 60)
    print(f"  {'Batch N':>8}  {'Rewrite':>8}  {'Rephrase':>9}  {'Locality':>9}")
    print("  " + "-" * 44)
    for r in batch_runs:
        m = r.get("metrics", {})
        print(f"  {r.get('n_samples',0):>8}  "
              f"{fmt(m.get('rewrite_acc'))}  "
              f"{fmt(m.get('rephrase_acc'))}  "
              f"{fmt(m.get('locality_acc'))}")
    print("=" * 60)


def show_probe_summary(probe_results: list[dict]) -> None:
    if not probe_results:
        print("\nNo probe results recorded yet.")
        return

    # Group by method × category
    stats: dict[tuple, dict] = defaultdict(lambda: {"n": 0, "pre": 0, "post": 0})
    for r in probe_results:
        key = (r.get("method", "?"), r.get("category", "?"))
        stats[key]["n"]    += 1
        stats[key]["pre"]  += int(r.get("pre_edit",  {}).get("passed", False))
        stats[key]["post"] += int(r.get("post_edit", {}).get("passed", False))

    methods = sorted({k[0] for k in stats})
    cats    = sorted({k[1] for k in stats})

    print("\n" + "=" * 80)
    print("  PROBE RESULTS — post-edit pass rate by category and method")
    print("=" * 80)
    col_w = 18

    header = f"  {'Category':<22}" + "".join(f"{m:>{col_w}}" for m in methods)
    print(header)
    print("  " + "-" * (22 + col_w * len(methods)))

    for cat in cats:
        row = f"  {cat:<22}"
        for method in methods:
            s = stats.get((method, cat))
            if s and s["n"]:
                post_pct = s["post"] / s["n"]
                pre_pct  = s["pre"]  / s["n"]
                delta = post_pct - pre_pct
                row += f"  {post_pct:4.0%} ({delta:>+4.0%})"
            else:
                row += f"  {'—':>{col_w - 2}}"
        print(row)

    # Totals
    print("  " + "-" * (22 + col_w * len(methods)))
    row = f"  {'TOTAL':<22}"
    for method in methods:
        n = post = pre = 0
        for cat in cats:
            s = stats.get((method, cat))
            if s:
                n += s["n"]; post += s["post"]; pre += s["pre"]
        if n:
            row += f"  {post/n:4.0%} ({(post-pre)/n:>+4.0%})"
        else:
            row += f"  {'—':>{col_w - 2}}"
    print(row)
    print("  (Δ = post − pre edit pass rate)")
    print("=" * 80)

    type_stats: dict[tuple, dict] = defaultdict(lambda: {"n": 0, "pre": 0, "post": 0})
    for r in probe_results:
        key = (r.get("method", "?"), r.get("probe_type", "implicit_edit"))
        type_stats[key]["n"]    += 1
        type_stats[key]["pre"]  += int(r.get("pre_edit",  {}).get("passed", False))
        type_stats[key]["post"] += int(r.get("post_edit", {}).get("passed", False))

    probe_types = sorted({k[1] for k in type_stats})
    print("\n" + "=" * 80)
    print("  PROBE RESULTS — post-edit pass rate by probe_type and method")
    print("=" * 80)
    header = f"  {'Probe type':<26}" + "".join(f"{m:>{col_w}}" for m in methods)
    print(header)
    print("  " + "-" * (26 + col_w * len(methods)))
    for probe_type in probe_types:
        row = f"  {probe_type:<26}"
        for method in methods:
            s = type_stats.get((method, probe_type))
            if s and s["n"]:
                post_pct = s["post"] / s["n"]
                pre_pct  = s["pre"]  / s["n"]
                row += f"  {post_pct:4.0%} ({post_pct - pre_pct:>+4.0%})"
            else:
                row += f"  {'—':>{col_w - 2}}"
        print(row)
    print("  (probe_type separates implicit edit tests from target-conditioned/supplied-fact prompts)")
    print("=" * 80)


def ascii_bar(label: str, value: float | None, paper: float | None,
              bar_width: int = 40) -> str:
    if value is None:
        return f"  {label:<20} [no data]"
    filled = int(round((value or 0) * bar_width))
    bar = "█" * filled + "░" * (bar_width - filled)
    paper_str = f"(paper {paper:.3f})" if paper is not None else ""
    return f"  {label:<20} [{bar}] {value:.3f} {paper_str}"


def show_ascii_plot(runs: list[dict]) -> None:
    methods_order = ["ROME", "MEMIT", "IKE"]
    latest: dict[str, dict] = {}
    for r in runs:
        method = r.get("method", "?")
        if method in methods_order:
            latest[method] = r

    print("\n" + "=" * 72)
    print("  REWRITE ACCURACY (latest run per method)")
    print("=" * 72)
    for method in methods_order:
        r = latest.get(method)
        val   = r["metrics"].get("rewrite_acc")  if r else None
        paper = PAPER_TARGETS.get(method, {}).get("rewrite_acc")
        print(ascii_bar(method, val, paper))
    print()
    print("  LOCALITY ACCURACY")
    print()
    for method in methods_order:
        r = latest.get(method)
        val   = r["metrics"].get("locality_acc") if r else None
        paper = PAPER_TARGETS.get(method, {}).get("locality_acc")
        print(ascii_bar(method, val, paper))
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probes", action="store_true")
    parser.add_argument("--plot",   action="store_true")
    parser.add_argument("--all",    action="store_true")
    args = parser.parse_args()

    show_probes = args.probes or args.all
    show_plot   = args.plot   or args.all

    runs = load_jsonl(RUNS_PATH)

    show_baseline_table(runs)
    show_method_summary(runs)
    show_batch_sweep(runs)

    if show_plot:
        show_ascii_plot(runs)

    if show_probes:
        probe_results = load_jsonl(PROBES_PATH)
        show_probe_summary(probe_results)


if __name__ == "__main__":
    main()
