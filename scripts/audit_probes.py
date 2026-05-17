"""
Audit the hand-curated probe set without loading any model.

Checks:
  - unique probe IDs
  - valid edit keys, categories, and probe types
  - expected answer fields are present
  - minimum total probe count and coverage by edit/category/type
  - implicit_edit probes do not leak the edited target value in the prompt

Usage:
    python scripts/audit_probes.py
    python scripts/audit_probes.py --min_total 225 --strict
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.probes.probe_set import EDIT_CASES, PROBES


ALLOWED_CATEGORIES = {
    "logical_negation",
    "symmetric_inverse",
    "compositional",
    "contradiction",
    "chain_of_thought",
}

ALLOWED_PROBE_TYPES = {
    "implicit_edit",
    "target_conditioned",
    "supplied_fact_reasoning",
}


def norm(text: str | None) -> str:
    return " ".join((text or "").lower().split())


def target_leaks(probe) -> bool:
    edit = EDIT_CASES[probe.edit_key]
    target = norm(edit.target_new)
    prompt = norm(probe.probe_prompt)
    if not target:
        return False
    if target in prompt:
        return True

    # Also catch the first word of multi-token targets when it is distinctive
    # enough to make an implicit probe target-conditioned.
    first = target.split()[0]
    if len(first) >= 5 and first in prompt:
        return True
    return False


def audit(min_total: int) -> tuple[list[str], list[str], dict]:
    errors: list[str] = []
    warnings: list[str] = []

    ids = [p.probe_id for p in PROBES]
    id_counts = Counter(ids)
    for probe_id, count in sorted(id_counts.items()):
        if count > 1:
            errors.append(f"duplicate probe_id: {probe_id} appears {count} times")

    by_edit = Counter()
    by_category = Counter()
    by_type = Counter()
    by_edit_category: dict[str, Counter] = defaultdict(Counter)
    by_edit_type: dict[str, Counter] = defaultdict(Counter)

    for p in PROBES:
        if p.edit_key not in EDIT_CASES:
            errors.append(f"{p.probe_id}: unknown edit_key {p.edit_key!r}")
            continue
        if p.category not in ALLOWED_CATEGORIES:
            errors.append(f"{p.probe_id}: invalid category {p.category!r}")
        if p.probe_type not in ALLOWED_PROBE_TYPES:
            errors.append(f"{p.probe_id}: invalid probe_type {p.probe_type!r}")
        if p.expected_first_token is None and p.expected_contains is None:
            errors.append(f"{p.probe_id}: expected_first_token and expected_contains are both None")
        if not p.probe_prompt.strip():
            errors.append(f"{p.probe_id}: empty probe_prompt")
        if not p.note.strip():
            warnings.append(f"{p.probe_id}: empty note")

        if p.probe_type == "implicit_edit" and target_leaks(p):
            errors.append(
                f"{p.probe_id}: implicit_edit prompt leaks target_new "
                f"{EDIT_CASES[p.edit_key].target_new!r}"
            )

        if p.probe_type in {"target_conditioned", "supplied_fact_reasoning"} and not target_leaks(p):
            warnings.append(f"{p.probe_id}: {p.probe_type} does not mention target_new")

        by_edit[p.edit_key] += 1
        by_category[p.category] += 1
        by_type[p.probe_type] += 1
        by_edit_category[p.edit_key][p.category] += 1
        by_edit_type[p.edit_key][p.probe_type] += 1

    if len(PROBES) < min_total:
        errors.append(f"probe count {len(PROBES)} is below --min_total {min_total}")

    missing_edits = sorted(set(EDIT_CASES) - set(by_edit))
    if missing_edits:
        errors.append(f"edit cases with no probes: {', '.join(missing_edits)}")

    for edit_key in sorted(EDIT_CASES):
        if by_edit[edit_key] < 8:
            warnings.append(f"{edit_key}: only {by_edit[edit_key]} probes; target is >=8")
        if by_edit_type[edit_key]["implicit_edit"] < 3:
            warnings.append(f"{edit_key}: only {by_edit_type[edit_key]['implicit_edit']} implicit probes")

    for category in sorted(ALLOWED_CATEGORIES):
        if by_category[category] < 10:
            warnings.append(f"{category}: only {by_category[category]} probes; target is >=10")

    supplied = by_type["supplied_fact_reasoning"]
    # A class-balanced five-category probe set has two supplied-fact-heavy
    # categories (compositional and chain_of_thought), so 40% is expected.
    if PROBES and supplied / len(PROBES) > 0.45:
        warnings.append(
            f"supplied_fact_reasoning is {supplied}/{len(PROBES)} probes; "
            "keep these separate from implicit edit-transfer claims"
        )

    summary = {
        "total": len(PROBES),
        "by_edit": dict(sorted(by_edit.items())),
        "by_category": dict(sorted(by_category.items())),
        "by_type": dict(sorted(by_type.items())),
        "by_edit_category": {k: dict(sorted(v.items())) for k, v in sorted(by_edit_category.items())},
        "by_edit_type": {k: dict(sorted(v.items())) for k, v in sorted(by_edit_type.items())},
    }
    return errors, warnings, summary


def print_counter(title: str, counts: dict[str, int]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    for key, value in counts.items():
        print(f"{key:<28} {value:>4}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min_total", type=int, default=100)
    parser.add_argument("--strict", action="store_true",
                        help="Treat warnings as failures")
    parser.add_argument("--json", action="store_true",
                        help="Emit machine-readable summary JSON")
    args = parser.parse_args()

    errors, warnings, summary = audit(args.min_total)

    if args.json:
        print(json.dumps({"errors": errors, "warnings": warnings, "summary": summary}, indent=2))
    else:
        print(f"Probe audit: {summary['total']} probes")
        print_counter("By edit", summary["by_edit"])
        print_counter("By category", summary["by_category"])
        print_counter("By type", summary["by_type"])

        if warnings:
            print("\nWarnings")
            print("--------")
            for warning in warnings:
                print(f"- {warning}")

        if errors:
            print("\nErrors")
            print("------")
            for error in errors:
                print(f"- {error}")

    if errors or (args.strict and warnings):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
