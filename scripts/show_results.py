"""
Summarize results/runs.jsonl and optionally results/probe_results_225.jsonl.

Usage:
    python scripts/show_results.py               # baseline runs table
    python scripts/show_results.py --probes      # add probe summary
    python scripts/show_results.py --plot        # ASCII bar chart of rewrite_acc by method
    python scripts/show_results.py --csv_dir results/csv
    python scripts/show_results.py --all         # everything
"""

import argparse
import csv
import json
import os
from collections import defaultdict

RUNS_PATH   = os.path.join(os.path.dirname(__file__), "..", "results", "runs.jsonl")
PROBES_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "probe_results_225.jsonl")
LEGACY_PROBES_PATH = os.path.join(
    os.path.dirname(__file__), "..", "results", "legacy", "probe_results_100_legacy.jsonl"
)
DEFAULT_CSV_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "csv")

PAPER_TARGETS = {
    "ROME":        {"rewrite_acc": 0.996, "rephrase_acc": 0.948, "locality_acc": 0.722},
    "MEMIT":       {"rewrite_acc": 0.998, "rephrase_acc": 0.732, "locality_acc": 0.640},
    "MEMIT-batch": {"rewrite_acc": 0.995, "rephrase_acc": 0.732, "locality_acc": 0.640},
    "IKE":         {"rewrite_acc": 0.678, "rephrase_acc": 0.507, "locality_acc": 0.575},
}

REPHRASE_RELATIVE = {"ROME", "MEMIT", "MEMIT-batch", "IKE"}

CORE_METRICS = {"rewrite_acc", "rephrase_acc", "locality_acc"}


def run_label(run: dict) -> str:
    method = run.get("method", "?")
    if method == "IKE" and "k" in run:
        return f"{method} k={run['k']}"
    return method


def is_core_run(run: dict) -> bool:
    metrics = run.get("metrics", {})
    return any(metric in metrics for metric in CORE_METRICS)


def is_benchmark_run(run: dict) -> bool:
    return not is_core_run(run) and bool(run.get("metrics"))


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
    runs = [r for r in runs if is_core_run(r)]
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
        label = run_label(r)
        paper  = PAPER_TARGETS.get(method, {})

        rw = m.get("rewrite_acc")
        rp = m.get("rephrase_acc")
        lo = m.get("locality_acc")

        rp_note = "*" if method in REPHRASE_RELATIVE else " "

        print(
            f"  {i:<3} {r.get('timestamp','')[:10]:<11} {label:<12} "
            f"{r.get('dataset','?'):<24} {r.get('n_samples',0):>5} "
            f"{fmt(rw)}{delta_str(rw, paper.get('rewrite_acc')):>7}  "
            f"{fmt(rp)}{delta_str(rp, paper.get('rephrase_acc')):>7}{rp_note} "
            f"{fmt(lo)}{delta_str(lo, paper.get('locality_acc')):>7}"
        )

    print("  " + "-" * 96)
    print("  Δ = ours − paper target.  * rephrase_acc is relative-only (noisy EasyEdit prompts).")
    print("=" * 100)


def show_method_summary(runs: list[dict]) -> None:
    """Print latest single-edit run per method with paper comparison."""
    seen: dict[str, dict] = {}
    for r in [run for run in runs if is_core_run(run)]:
        method = r.get("method", "?")
        if "batch" in method.lower():
            continue
        seen[method] = r

    if not seen:
        return

    print("\n" + "=" * 72)
    print("  METHOD COMPARISON (latest single-edit run per method)")
    print("=" * 72)
    print(f"  {'Method':<12} {'Rewrite':>8} {'Paper':>7} {'Δ':>7}  "
          f"{'Locality':>9} {'Paper':>7} {'Δ':>7}")
    print("  " + "-" * 64)
    for method, r in sorted(seen.items()):
        m     = r.get("metrics", {})
        paper = PAPER_TARGETS.get(method, {})
        rw = m.get("rewrite_acc")
        lo = m.get("locality_acc")
        print(f"  {run_label(r):<12} {fmt(rw)} {fmt(paper.get('rewrite_acc'))} "
              f"{delta_str(rw, paper.get('rewrite_acc'))}  "
              f"{fmt(lo)} {fmt(paper.get('locality_acc'))} "
              f"{delta_str(lo, paper.get('locality_acc'))}")
    print("=" * 72)


def show_batch_sweep(runs: list[dict]) -> None:
    """Show MEMIT-batch results sorted by batch size."""
    batch_runs = [r for r in runs if r.get("method") == "MEMIT-batch" and is_core_run(r)]
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


def show_benchmark_runs(runs: list[dict]) -> None:
    benchmark_runs = [r for r in runs if is_benchmark_run(r)]
    if not benchmark_runs:
        return

    print("\n" + "=" * 96)
    print("  MQuAKE / RIPPLEEDITS RUNS")
    print("=" * 96)
    print(f"  {'#':<3} {'Date':<11} {'Method':<8} {'Dataset':<24} {'N':>4}  {'Primary metrics':<38}")
    print("  " + "-" * 88)
    for i, r in enumerate(benchmark_runs):
        metrics = r.get("metrics", {})
        priority = [
            "edited_fact_acc",
            "delta_edited_fact_acc",
            "multihop_acc",
            "delta_multihop_acc",
            "overall_acc",
            "delta_overall_acc",
            "Logical_Generalization_acc",
            "Subject_Aliasing_acc",
            "Compositionality_I_acc",
            "Compositionality_II_acc",
            "Forgetfulness_acc",
        ]
        parts = []
        for key in priority:
            if key in metrics:
                value = metrics[key]
                parts.append(f"{key}={value:.3f}" if isinstance(value, (int, float)) else f"{key}={value}")
        metric_str = ", ".join(parts[:3]) if parts else json.dumps(metrics, sort_keys=True)[:38]
        print(
            f"  {i:<3} {r.get('timestamp','')[:10]:<11} {r.get('method','?'):<8} "
            f"{r.get('dataset','?'):<24} {r.get('n_samples',0):>4}  {metric_str:<38}"
        )
    print("=" * 96)


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
    for r in [run for run in runs if is_core_run(run)]:
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


def probe_summary_rows(probe_results: list[dict], group_key: str) -> list[dict]:
    stats: dict[tuple, dict] = defaultdict(lambda: {"n": 0, "pre": 0, "post": 0})
    for r in probe_results:
        key = (r.get("method", "?"), r.get(group_key, "?"))
        stats[key]["n"] += 1
        stats[key]["pre"] += int(r.get("pre_edit", {}).get("passed", False))
        stats[key]["post"] += int(r.get("post_edit", {}).get("passed", False))

    rows = []
    for (method, group_value), s in sorted(stats.items()):
        n = s["n"]
        pre_rate = s["pre"] / n if n else None
        post_rate = s["post"] / n if n else None
        rows.append({
            "method": method,
            group_key: group_value,
            "n": n,
            "pre_pass_rate": pre_rate,
            "post_pass_rate": post_rate,
            "delta_pass_rate": (post_rate - pre_rate) if pre_rate is not None and post_rate is not None else None,
        })
    return rows


def write_csv(path: str, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def export_csv(runs: list[dict], probe_results: list[dict], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    run_rows = []
    for r in runs:
        metrics = r.get("metrics", {})
        run_rows.append({
            "timestamp": r.get("timestamp"),
            "method": r.get("method"),
            "model": r.get("model"),
            "dataset": r.get("dataset"),
            "n_samples": r.get("n_samples"),
            "seed": r.get("seed"),
            "k": r.get("k"),
            "rewrite_acc": metrics.get("rewrite_acc"),
            "rephrase_acc": metrics.get("rephrase_acc"),
            "locality_acc": metrics.get("locality_acc"),
            "edited_fact_acc": metrics.get("edited_fact_acc"),
            "pre_edited_fact_acc": metrics.get("pre_edited_fact_acc"),
            "delta_edited_fact_acc": metrics.get("delta_edited_fact_acc"),
            "multihop_acc": metrics.get("multihop_acc"),
            "pre_multihop_acc": metrics.get("pre_multihop_acc"),
            "delta_multihop_acc": metrics.get("delta_multihop_acc"),
            "overall_acc": metrics.get("overall_acc"),
            "pre_overall_acc": metrics.get("pre_overall_acc"),
            "delta_overall_acc": metrics.get("delta_overall_acc"),
            "Relation_Specificity_acc": metrics.get("Relation_Specificity_acc"),
            "Logical_Generalization_acc": metrics.get("Logical_Generalization_acc"),
            "Subject_Aliasing_acc": metrics.get("Subject_Aliasing_acc"),
            "Compositionality_I_acc": metrics.get("Compositionality_I_acc"),
            "Compositionality_II_acc": metrics.get("Compositionality_II_acc"),
            "Forgetfulness_acc": metrics.get("Forgetfulness_acc"),
        })

    write_csv(
        os.path.join(out_dir, "runs.csv"),
        run_rows,
        ["timestamp", "method", "model", "dataset", "n_samples", "seed",
         "k",
         "rewrite_acc", "rephrase_acc", "locality_acc",
         "pre_edited_fact_acc", "edited_fact_acc", "delta_edited_fact_acc",
         "pre_multihop_acc", "multihop_acc", "delta_multihop_acc",
         "pre_overall_acc", "overall_acc", "delta_overall_acc",
         "Relation_Specificity_acc", "Logical_Generalization_acc", "Subject_Aliasing_acc",
         "Compositionality_I_acc", "Compositionality_II_acc", "Forgetfulness_acc"],
    )

    if probe_results:
        probe_rows = []
        for r in probe_results:
            probe_rows.append({
                "timestamp": r.get("timestamp"),
                "method": r.get("method"),
                "edit_key": r.get("edit_key"),
                "probe_id": r.get("probe_id"),
                "category": r.get("category"),
                "probe_type": r.get("probe_type", "implicit_edit"),
                "pre_passed": r.get("pre_edit", {}).get("passed"),
                "post_passed": r.get("post_edit", {}).get("passed"),
                "pre_first_token": r.get("pre_edit", {}).get("first_token"),
                "post_first_token": r.get("post_edit", {}).get("first_token"),
            })
        write_csv(
            os.path.join(out_dir, "probe_results.csv"),
            probe_rows,
            ["timestamp", "method", "edit_key", "probe_id", "category", "probe_type",
             "pre_passed", "post_passed", "pre_first_token", "post_first_token"],
        )

        category_rows = probe_summary_rows(probe_results, "category")
        write_csv(
            os.path.join(out_dir, "probe_summary_by_category.csv"),
            category_rows,
            ["method", "category", "n", "pre_pass_rate", "post_pass_rate", "delta_pass_rate"],
        )

        type_rows = probe_summary_rows(probe_results, "probe_type")
        write_csv(
            os.path.join(out_dir, "probe_summary_by_type.csv"),
            type_rows,
            ["method", "probe_type", "n", "pre_pass_rate", "post_pass_rate", "delta_pass_rate"],
        )

    print(f"\nCSV exports written to {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probes", action="store_true")
    parser.add_argument("--plot",   action="store_true")
    parser.add_argument("--all",    action="store_true")
    parser.add_argument("--probes_path", default=PROBES_PATH,
                        help="Probe JSONL file to summarize/export")
    parser.add_argument("--csv_dir", nargs="?", const=DEFAULT_CSV_DIR,
                        help="Write CSV exports to this directory (default: results/csv)")
    args = parser.parse_args()

    show_probes = args.probes or args.all
    show_plot   = args.plot   or args.all

    runs = load_jsonl(RUNS_PATH)

    show_baseline_table(runs)
    show_method_summary(runs)
    show_batch_sweep(runs)
    show_benchmark_runs(runs)

    if show_plot:
        show_ascii_plot(runs)

    probe_results = []
    if show_probes or args.csv_dir:
        probe_results = load_jsonl(args.probes_path)

    if show_probes:
        if os.path.abspath(args.probes_path) == os.path.abspath(LEGACY_PROBES_PATH):
            print("\nNote: summarizing legacy 100-probe results. "
                  "Use --probes_path results/probe_results_225.jsonl for final report results.")
        else:
            print(f"\nProbe source: {args.probes_path}")
        show_probe_summary(probe_results)

    if args.csv_dir:
        export_csv(runs, probe_results, args.csv_dir)


if __name__ == "__main__":
    main()
