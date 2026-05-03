"""Inspect EasyEdit rephrase prompt failures from a saved results.json file.

This is a lightweight audit tool for the CounterFact baseline logs. It does not
load a model; it checks the per-edit metrics and flags rephrase prompts that are
likely not clean paraphrases of the edited fact.

Usage:
    python scripts/inspect_rephrase_failures.py
    python scripts/inspect_rephrase_failures.py --limit 20
    python scripts/inspect_rephrase_failures.py --all
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_RESULTS_PATH = Path("logs/results.json")

NOISE_RE = re.compile(
    r"\n|Category:|References\b|\bISBN\b|\bRetrieved\b|\b\d{4}\)|"
    r"\bp\.\s*\d+|^\d{1,2}\s+[A-Z][a-z]+",
    re.IGNORECASE,
)

WEAK_REPHRASE_PATTERNS = [
    "favorite lunchtime work meals include",
    "best restaurants around",
    "famous tourist attractions include",
    "surroundings include",
    "by navigating",
    "lives in",
    "known for",
    "greatest strength",
    "greatest weakness",
    "greatest accomplishment",
    "religious values strongly emphasize",
    "passport that",
    "capital is known for",
    "twin city has famous tourist attractions",
    "aired alongside other programs",
    "inspiration for",
    "works as a",
]


def metric(record: dict[str, Any], path: tuple[str, ...]) -> float:
    node: Any = record
    for key in path:
        node = node[key]
    if isinstance(node, list):
        return float(node[0])
    return float(node)


def quality_flags(record: dict[str, Any]) -> list[str]:
    req = record["requested_rewrite"]
    prompt = req["prompt"]
    rephrase = req["rephrase_prompt"]
    flags: list[str] = []

    if NOISE_RE.search(rephrase):
        flags.append("retrieval-noise")
    if len(rephrase) > 90:
        flags.append("long")
    if req["subject"] not in rephrase:
        flags.append("missing-subject")

    lower = rephrase.lower()
    for pattern in WEAK_REPHRASE_PATTERNS:
        if pattern in lower:
            flags.append("weak-or-indirect")
            break

    # A few relation-specific mismatches seen in EasyEdit CounterFact prompts.
    prompt_lower = prompt.lower()
    if ("language" in prompt_lower or "speaks" in prompt_lower or "writes in" in prompt_lower) and "lives in" in lower:
        flags.append("relation-mismatch")
    if ("worked" in prompt_lower or "employed" in prompt_lower or "work in" in prompt_lower) and "meals" in lower:
        flags.append("relation-mismatch")
    if ("located" in prompt_lower or " in" in prompt_lower) and ("restaurants" in lower or "navigating" in lower):
        flags.append("relation-mismatch")
    if ("capital" in prompt_lower or "twin city" in prompt_lower) and ("tourist attractions" in lower or "known for" in lower):
        flags.append("relation-mismatch")
    if ("continent" in prompt_lower or "located" in prompt_lower or " is in" in prompt_lower) and (
        "speak the language" in lower or "people around" in lower
    ):
        flags.append("relation-mismatch")
    if ("field of" in prompt_lower or "area of work" in prompt_lower or "domain of" in prompt_lower) and "works as" in lower:
        flags.append("relation-mismatch")

    return flags or ["looks-clean"]


def summarize(records: list[dict[str, Any]]) -> None:
    rewrite = [metric(r, ("post", "rewrite_acc")) for r in records]
    rephrase = [metric(r, ("post", "rephrase_acc")) for r in records]
    locality = [metric(r, ("post", "locality", "neighborhood_acc")) for r in records]

    print(f"records:    {len(records)}")
    print(f"rewrite:    {sum(rewrite) / len(rewrite):.3f} ({sum(v == 1.0 for v in rewrite)}/{len(rewrite)})")
    print(f"rephrase:   {sum(rephrase) / len(rephrase):.3f} ({sum(v == 1.0 for v in rephrase)}/{len(rephrase)})")
    print(f"locality:   {sum(locality) / len(locality):.3f} ({sum(v == 1.0 for v in locality)}/{len(locality)})")

    failures = [r for r in records if metric(r, ("post", "rephrase_acc")) == 0.0]
    successes = [r for r in records if metric(r, ("post", "rephrase_acc")) == 1.0]
    flagged = [r for r in failures if quality_flags(r) != ["looks-clean"]]
    print(f"failures:   {len(failures)}")
    print(f"flagged:    {len(flagged)} failures with prompt-quality flags")

    counts: dict[str, int] = {}
    for record in failures:
        for flag in quality_flags(record):
            counts[flag] = counts.get(flag, 0) + 1
    print("\nFailure flags:")
    for flag, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        print(f"  {flag:<18} {count:>3}")

    success_flagged = sum(quality_flags(r) != ["looks-clean"] for r in successes)
    print(f"\nPrompt-quality flags among successes: {success_flagged}/{len(successes)}")


def print_failures(records: list[dict[str, Any]], limit: int | None) -> None:
    failures = [r for r in records if metric(r, ("post", "rephrase_acc")) == 0.0]
    print("\nRephrase failures:")
    for i, record in enumerate(failures):
        if limit is not None and i >= limit:
            remaining = len(failures) - limit
            print(f"\n... {remaining} more failures hidden; pass --all to show every case.")
            return
        req = record["requested_rewrite"]
        flags = ",".join(quality_flags(record))
        print(
            f"{record['case_id']:>3} | {flags:<38} | "
            f"{req['prompt']!r} -> {req['target_new']!r} | {req['rephrase_prompt']!r}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--all", action="store_true", help="show all failures")
    args = parser.parse_args()

    records = json.loads(args.results.read_text())
    summarize(records)
    print_failures(records, None if args.all else args.limit)


if __name__ == "__main__":
    main()
