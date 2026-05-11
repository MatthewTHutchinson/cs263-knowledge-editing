import json
import re
from pathlib import Path
from typing import Any


def load_json(path: str | Path) -> Any:
    with open(path) as f:
        return json.load(f)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def answer_variants(answer: Any, aliases: list[str] | None = None) -> list[str]:
    variants: list[str] = []
    if isinstance(answer, str):
        variants.append(answer)
    elif isinstance(answer, dict):
        value = answer.get("value") or answer.get("str")
        if value:
            variants.append(value)
        variants.extend(answer.get("aliases") or [])
    elif isinstance(answer, list):
        for item in answer:
            variants.extend(answer_variants(item))
    if aliases:
        variants.extend(aliases)

    deduped = []
    seen = set()
    for variant in variants:
        norm = normalize_text(str(variant))
        if norm and norm not in seen:
            seen.add(norm)
            deduped.append(str(variant))
    return deduped


def contains_answer(generation: str, answer: Any, aliases: list[str] | None = None) -> bool:
    gen = normalize_text(generation)
    return any(normalize_text(variant) in gen for variant in answer_variants(answer, aliases))
