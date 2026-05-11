"""
Inspect external benchmark files before running GPU evaluations.

Usage:
    python scripts/inspect_benchmarks.py --mquake data/mquake/MQuAKE-CF-3k-v2.json
    python scripts/inspect_benchmarks.py --ripple data/ripple_edits/POPULAR.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.benchmarks import mquake, ripple_edits


def print_json(label: str, payload: dict) -> None:
    print(f"\n{label}")
    print("=" * len(label))
    print(json.dumps(payload, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mquake", default=None)
    parser.add_argument("--ripple", default=None)
    parser.add_argument("--sample", type=int, default=1)
    args = parser.parse_args()

    if not args.mquake and not args.ripple:
        parser.error("provide --mquake and/or --ripple")

    if args.mquake:
        records = mquake.load_records(args.mquake)
        print_json("MQuAKE summary", mquake.summarize_records(records))
        for record in records[: args.sample]:
            print_json("MQuAKE eval case", mquake.record_to_eval_case(record))

    if args.ripple:
        records = ripple_edits.load_records(args.ripple)
        print_json("RippleEdits summary", ripple_edits.summarize_records(records))
        if records and args.sample:
            first = records[0]
            print_json(
                "RippleEdits first edit",
                {
                    "example_type": first.get("example_type"),
                    "edit": first.get("edit"),
                    "criteria": {
                        name: len(first.get(name, []) or [])
                        for name in ripple_edits.CRITERIA
                    },
                },
            )


if __name__ == "__main__":
    main()
