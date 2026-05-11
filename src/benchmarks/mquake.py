from __future__ import annotations

from collections import Counter
from typing import Any

from .common import answer_variants, contains_answer, load_json


def rewrite_to_request(rewrite: dict[str, Any]) -> dict[str, str]:
    prompt_template = rewrite["prompt"]
    subject = rewrite["subject"]
    target_new = rewrite["target_new"]["str"]
    target_true = rewrite.get("target_true", {}).get("str", "")
    return {
        "prompt": prompt_template.format(subject),
        "subject": subject,
        "target_new": target_new,
        "ground_truth": target_true,
        "question": rewrite.get("question", ""),
    }


def record_to_requests(record: dict[str, Any]) -> list[dict[str, str]]:
    return [rewrite_to_request(r) for r in record.get("requested_rewrite", [])]


def record_to_eval_case(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": record.get("case_id"),
        "requests": record_to_requests(record),
        "multihop_prompts": record.get("questions", []),
        "multihop_answers": answer_variants(
            record.get("new_answer", ""),
            record.get("new_answer_alias", []),
        ),
        "single_hops": record.get("new_single_hops", []),
    }


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    edit_counts = Counter(len(r.get("requested_rewrite", [])) for r in records)
    question_counts = Counter(len(r.get("questions", [])) for r in records)
    hop_counts = Counter(len(r.get("new_single_hops", [])) for r in records)
    missing = Counter()
    for record in records:
        for key in ("requested_rewrite", "questions", "new_answer", "new_single_hops"):
            if key not in record:
                missing[key] += 1
    return {
        "records": len(records),
        "edit_count_distribution": dict(sorted(edit_counts.items())),
        "question_count_distribution": dict(sorted(question_counts.items())),
        "new_single_hop_count_distribution": dict(sorted(hop_counts.items())),
        "missing_fields": dict(sorted(missing.items())),
    }


def load_records(path: str) -> list[dict[str, Any]]:
    records = load_json(path)
    if not isinstance(records, list):
        raise ValueError(f"MQuAKE file must contain a JSON list: {path}")
    return records


def score_multihop_generation(record: dict[str, Any], generation: str) -> bool:
    return contains_answer(
        generation,
        record.get("new_answer", ""),
        record.get("new_answer_alias", []),
    )
