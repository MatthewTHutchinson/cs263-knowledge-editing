"""Shared checkpoint helpers for CounterFact baseline scripts."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any


def safe_stem(path: str) -> str:
    stem = Path(path).stem
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)


def default_checkpoint_path(method: str, data_path: str, n_edits: int, seed: int) -> str:
    return os.path.join(
        "results",
        "checkpoints",
        f"{method.lower()}_{safe_stem(data_path)}_n{n_edits}_seed{seed}.jsonl",
    )


def dataset_label(data_path: str) -> str:
    if "original" in safe_stem(data_path).lower():
        return "CounterFact-original"
    return "CounterFact"


def load_completed_rows(
    checkpoint_path: str,
    method: str,
    data_path: str,
    n_edits: int,
    seed: int,
) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    if not os.path.exists(checkpoint_path):
        return rows

    with open(checkpoint_path) as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{checkpoint_path}:{line_no}: invalid JSON checkpoint row") from exc
            if (
                row.get("method") == method
                and row.get("data_path") == data_path
                and row.get("n_edits") == n_edits
                and row.get("seed") == seed
                and "sample_index" in row
            ):
                rows[int(row["sample_index"])] = row
    return rows


def append_checkpoint_row(checkpoint_path: str, row: dict) -> None:
    out_dir = os.path.dirname(checkpoint_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(checkpoint_path, "a") as f:
        f.write(json.dumps(json_safe(row)) + "\n")


def checkpoint_metrics(rows: dict[int, dict]) -> list[dict]:
    return [rows[i]["metric"] for i in sorted(rows)]


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    if hasattr(value, "item"):
        return value.item()
    return value
