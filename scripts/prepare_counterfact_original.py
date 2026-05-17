"""Download and convert the original ROME CounterFact dataset.

The repo's baseline scripts expect EasyEdit-style flat records:

    prompt, subject, ground_truth, target_new, rephrase_prompt,
    locality_prompt, locality_ground_truth

EasyEdit's bundled CounterFact conversion uses a single ``rephrase_prompt`` that
is often a noisy generation prompt. The original ROME dataset instead contains
``paraphrase_prompts`` and ``neighborhood_prompts``. This script converts those
records into the flat shape so the existing baseline scripts can be reused.

Usage:
    python3 scripts/prepare_counterfact_original.py
    python3 scripts/prepare_counterfact_original.py --max_records 2500
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_URL = "https://rome.baulab.info/data/dsets/counterfact.json"
DEFAULT_RAW_PATH = Path("data/counterfact/counterfact-original.json")
DEFAULT_OUT_PATH = Path("data/counterfact/counterfact-original-easyedit.json")

NOISE_RE = re.compile(
    r"\n|Category:|References\b|\bISBN\b|\bRetrieved\b|\bp\.\s*\d+",
    re.IGNORECASE,
)


def fill_prompt(template: str, subject: str) -> str:
    if "{}" in template:
        return template.format(subject)
    if subject in template:
        return template
    return f"{subject} {template}".strip()


def prompt_quality(prompt: str, subject: str) -> tuple[int, int]:
    """Rank cleaner paraphrase prompts before noisier/indirect ones."""
    score = 0
    if subject not in prompt:
        score += 3
    if NOISE_RE.search(prompt):
        score += 3
    if len(prompt) > 120:
        score += 2
    if len(prompt) < 20:
        score += 1
    # Prefer prompts where the subject appears near the start, because these are
    # usually direct cloze paraphrases rather than retrieved-context fragments.
    subject_pos = prompt.find(subject)
    if subject_pos > 80:
        score += 1
    return score, len(prompt)


def choose_prompt(prompts: list[str], subject: str) -> str | None:
    if not prompts:
        return None
    return sorted(prompts, key=lambda prompt: prompt_quality(prompt, subject))[0]


def convert_record(record: dict[str, Any]) -> dict[str, Any] | None:
    rewrite = record.get("requested_rewrite") or {}
    subject = rewrite.get("subject")
    prompt_template = rewrite.get("prompt")
    target_new = (rewrite.get("target_new") or {}).get("str")
    target_true = (rewrite.get("target_true") or {}).get("str")
    if not all([subject, prompt_template, target_new, target_true]):
        return None

    rephrase_prompt = choose_prompt(record.get("paraphrase_prompts") or [], subject)
    locality_prompt = choose_prompt(record.get("neighborhood_prompts") or [], subject)
    if not rephrase_prompt or not locality_prompt:
        return None

    return {
        "case_id": record.get("case_id"),
        "pararel_idx": record.get("pararel_idx"),
        "relation_id": rewrite.get("relation_id"),
        "prompt": fill_prompt(prompt_template, subject),
        "target_new": target_new,
        "subject": subject,
        "ground_truth": target_true,
        "rephrase_prompt": rephrase_prompt,
        # CounterFact neighborhood prompts share the original relation and true
        # attribute, so the true target is the expected neighborhood completion.
        "locality_prompt": locality_prompt,
        "locality_ground_truth": target_true,
        "source": "ROME CounterFact",
    }


def download(url: str, raw_path: Path, force: bool) -> None:
    if raw_path.exists() and not force:
        print(f"Raw dataset already exists: {raw_path}")
        return
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, raw_path)
    print(f"Wrote {raw_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--raw_path", type=Path, default=DEFAULT_RAW_PATH)
    parser.add_argument("--out_path", type=Path, default=DEFAULT_OUT_PATH)
    parser.add_argument("--max_records", type=int, default=None)
    parser.add_argument("--force_download", action="store_true")
    args = parser.parse_args()

    download(args.url, args.raw_path, args.force_download)
    raw_records = json.loads(args.raw_path.read_text())

    converted: list[dict[str, Any]] = []
    skipped = 0
    for record in raw_records:
        out = convert_record(record)
        if out is None:
            skipped += 1
            continue
        converted.append(out)
        if args.max_records and len(converted) >= args.max_records:
            break

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_path.write_text(json.dumps(converted, indent=2) + "\n")

    print(f"Converted records: {len(converted)}")
    print(f"Skipped records:   {skipped}")
    print(f"Wrote:             {args.out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
