"""Prepare DeepSuiMind for conversational prompt-completion SFT.

DeepSuiMind contains narratives and three categorical annotations, but no
gold-standard helper replies. This pipeline therefore teaches structured
crisis understanding; it must not be described as response-policy training.

Usage:
    python -m src.llama.preprocess
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "data/raw/deepsuimind/implicit_suicide_data.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/processed/deepsuimind_sft"

REQUIRED_COLUMNS = (
    "synthetic_text",
    "Scenario",
    "Negative Core Belief",
    "Intention Category",
)

SYSTEM_PROMPT = (
    "You are a crisis-aware language understanding assistant for academic research. "
    "Analyze the supplied narrative and return only a JSON object with exactly these "
    "keys: scenario, negative_core_belief, and intention_category. Use concise labels; "
    "do not diagnose the writer and do not add advice or commentary."
)


def clean_text(value: Any) -> str:
    """Normalize whitespace without changing the narrative's wording."""
    return " ".join(str(value).split()).strip()


def validate_and_clean(rows: Iterable[dict[str, Any]]) -> list[dict[str, str]]:
    """Validate the schema, remove exact duplicates, and normalize whitespace."""
    cleaned: list[dict[str, str]] = []
    seen: set[str] = set()

    for index, row in enumerate(rows):
        missing = [column for column in REQUIRED_COLUMNS if column not in row]
        if missing:
            raise ValueError(f"Row {index} is missing required columns: {missing}")

        item = {column: clean_text(row[column]) for column in REQUIRED_COLUMNS}
        empty = [column for column, value in item.items() if not value]
        if empty:
            raise ValueError(f"Row {index} has empty required values: {empty}")

        fingerprint = item["synthetic_text"].casefold()
        if fingerprint not in seen:
            seen.add(fingerprint)
            cleaned.append(item)

    if not cleaned:
        raise ValueError("No valid examples remained after preprocessing")
    return cleaned


def format_example(row: dict[str, str]) -> dict[str, Any]:
    """Convert one annotation into TRL conversational prompt-completion format."""
    answer = {
        "scenario": row["Scenario"],
        "negative_core_belief": row["Negative Core Belief"],
        "intention_category": row["Intention Category"],
    }
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "Analyze this narrative:\n\n" + row["synthetic_text"],
            },
        ],
        "completion": [
            {
                "role": "assistant",
                "content": json.dumps(answer, ensure_ascii=False, sort_keys=True),
            }
        ],
        "metadata": answer,
    }


def stratified_split(
    rows: list[dict[str, str]],
    *,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
) -> dict[str, list[dict[str, str]]]:
    """Split deterministically while preserving intention-category proportions."""
    if validation_fraction < 0 or test_fraction < 0:
        raise ValueError("Split fractions cannot be negative")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation_fraction + test_fraction must be less than 1")

    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["Intention Category"]].append(row)

    rng = random.Random(seed)
    result: dict[str, list[dict[str, str]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    for label in sorted(groups):
        group = groups[label][:]
        rng.shuffle(group)
        n_test = round(len(group) * test_fraction)
        n_validation = round(len(group) * validation_fraction)
        result["test"].extend(group[:n_test])
        result["validation"].extend(group[n_test : n_test + n_validation])
        result["train"].extend(group[n_test + n_validation :])

    for split in result.values():
        rng.shuffle(split)
    return result


def write_jsonl(rows: Iterable[dict[str, Any]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def preprocess(
    input_path: Path,
    output_dir: Path,
    *,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
) -> dict[str, int]:
    with input_path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, list):
        raise ValueError("Expected the DeepSuiMind JSON root to be a list")

    rows = validate_and_clean(raw)
    splits = stratified_split(
        rows,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )
    counts: dict[str, int] = {}
    for name, split_rows in splits.items():
        counts[name] = write_jsonl(
            (format_example(row) for row in split_rows), output_dir / f"{name}.jsonl"
        )

    manifest = {
        "source": str(input_path),
        "seed": seed,
        "validation_fraction": validation_fraction,
        "test_fraction": test_fraction,
        "counts": counts,
        "task": "structured_crisis_understanding",
        "target_fields": [
            "scenario",
            "negative_core_belief",
            "intention_category",
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    counts = preprocess(
        args.input,
        args.output_dir,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    print(f"Wrote DeepSuiMind SFT splits to {args.output_dir}: {counts}")


if __name__ == "__main__":
    main()

