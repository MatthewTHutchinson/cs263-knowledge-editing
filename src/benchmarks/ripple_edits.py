from __future__ import annotations

from collections import Counter
import re
from typing import Any

from .common import answer_variants, contains_answer, load_json


CRITERIA = (
    "Relation_Specificity",
    "Logical_Generalization",
    "Subject_Aliasing",
    "Compositionality_I",
    "Compositionality_II",
    "Forgetfulness",
)

CRITERION_ALIASES = {
    # The upstream RippleEdits repository has used the misspelled key, while
    # the downloaded local files use the corrected spelling.
    "Relation_Specificity": ("Relation_Specificity", "Relation_Specifity"),
}


def criterion_keys(criterion: str) -> tuple[str, ...]:
    return CRITERION_ALIASES.get(criterion, (criterion,))


def get_criterion_tests(record: dict[str, Any], criterion: str) -> list[dict[str, Any]] | None:
    tests = []
    found = False
    for key in criterion_keys(criterion):
        if key not in record:
            continue
        found = True
        tests.extend(record.get(key) or [])
    return tests if found else None


def strip_fact_period(text: str) -> str:
    return text.strip().rstrip(".").strip()


def split_edit_prompt(new_fact: str, old_fact: str) -> tuple[str, str, str]:
    new_fact = strip_fact_period(new_fact)
    old_fact = strip_fact_period(old_fact)
    prefix_len = 0
    for left, right in zip(new_fact, old_fact):
        if left != right:
            break
        prefix_len += 1
    prefix = new_fact[:prefix_len].rstrip()
    target_new = new_fact[prefix_len:].strip()
    ground_truth = old_fact[prefix_len:].strip()
    if not prefix or not target_new or not ground_truth:
        raise ValueError(f"could not split RippleEdits prompt: {new_fact!r} vs {old_fact!r}")
    return prefix, target_new, ground_truth


def infer_subject(prompt: str) -> str:
    subject_first_patterns = (
        r"^(.+?) is followed by$",
        r"^(.+?) follows$",
        r"^(.+?) is part of$",
        r"^(.+?) has part$",
        r"^(.+?) is located in$",
        r"^(.+?) was created by$",
        r"^(.+?) was born in$",
        r"^(.+?) died in$",
    )
    for pattern in subject_first_patterns:
        match = re.match(pattern, prompt)
        if match:
            return match.group(1).strip()

    candidates = []
    for marker in (" of ", " in ", " by ", " for ", " from "):
        idx = prompt.rfind(marker)
        if idx >= 0:
            candidates.append(prompt[idx + len(marker):])
    if candidates:
        subject = min(candidates, key=len)
    else:
        subject = prompt
    subject = subject.strip()
    for suffix in (" is", " was", " are", " were", " has", " have"):
        if subject.endswith(suffix):
            subject = subject[: -len(suffix)].strip()
    return subject


def edit_to_request(record: dict[str, Any]) -> dict[str, str]:
    edit = record["edit"]
    prompt, target_new, ground_truth = split_edit_prompt(
        edit["prompt"],
        edit["original_fact"]["prompt"],
    )
    return {
        "prompt": prompt,
        "subject": infer_subject(prompt),
        "target_new": target_new,
        "ground_truth": ground_truth,
    }


def iter_tests(record: dict[str, Any]):
    for criterion in CRITERIA:
        for test in get_criterion_tests(record, criterion) or []:
            yield criterion, test


def iter_queries(test: dict[str, Any], key: str = "test_queries"):
    for query in test.get(key, []) or []:
        yield query


def query_answers(query: dict[str, Any]) -> list[str]:
    return answer_variants(query.get("answers", []))


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    criterion_tests = Counter()
    criterion_queries = Counter()
    example_types = Counter()
    relations = Counter()
    missing = Counter()

    for record in records:
        example_types[record.get("example_type", "unknown")] += 1
        edit = record.get("edit") or {}
        if not edit:
            missing["edit"] += 1
        if edit.get("relation"):
            relations[edit["relation"]] += 1
        for criterion in CRITERIA:
            tests = get_criterion_tests(record, criterion)
            if tests is None:
                missing[criterion] += 1
                continue
            criterion_tests[criterion] += len(tests)
            for test in tests:
                criterion_queries[criterion] += len(test.get("test_queries", []) or [])

    return {
        "records": len(records),
        "example_types": dict(sorted(example_types.items())),
        "criterion_tests": {k: criterion_tests[k] for k in CRITERIA},
        "criterion_queries": {k: criterion_queries[k] for k in CRITERIA},
        "top_relations": relations.most_common(10),
        "missing_fields": dict(sorted(missing.items())),
    }


def load_records(path: str) -> list[dict[str, Any]]:
    records = load_json(path)
    if not isinstance(records, list):
        raise ValueError(f"RippleEdits file must contain a JSON list: {path}")
    return records


def score_query_generation(query: dict[str, Any], generation: str) -> bool:
    return contains_answer(generation, query.get("answers", []))
