"""Prepare CRADLEBench for the Layer 1 crisis-type classifier.

CRADLEBench already ships fixed train/dev/test splits, so unlike the Llama
DeepSuiMind pipeline there is no re-splitting here: this just validates,
normalizes labels, and writes one JSONL record per post with a fixed-order
multi-hot label vector over the 7 crisis types (see labels.CRISIS_TYPES).

Usage:
    python -m src.classifier.preprocess
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src.classifier.labels import CRISIS_TYPES, is_flagged, parse_labels, temporal_by_type, to_multihot

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW = PROJECT_ROOT / "data/raw/cradlebench"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/processed/cradlebench_clf"

# (split name in output, source csv). "train" defaults to the consensus file
# (4,181 rows) rather than the stricter unanimous-only subset (3,058 rows) --
# more training data at the cost of some non-unanimous-agreement labels.
DEFAULT_SPLITS: dict[str, Path] = {
    "train": RAW / "train/train_consensus.csv",
    "validation": RAW / "dev/development.csv",
    "test": RAW / "test/test.csv",
}

REQUIRED_COLUMNS = ("question_id", "question_text", "final_labels")


def validate_and_clean(df: pd.DataFrame, *, source: Path) -> pd.DataFrame:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")

    df = df.copy()
    df["question_text"] = df["question_text"].apply(lambda v: " ".join(str(v).split()).strip())
    empty_text = df["question_text"] == ""
    if empty_text.any():
        raise ValueError(f"{source} has {int(empty_text.sum())} rows with empty question_text")

    duplicated_ids = df["question_id"].duplicated()
    if duplicated_ids.any():
        raise ValueError(f"{source} has {int(duplicated_ids.sum())} duplicate question_id values")

    return df


def build_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in df.itertuples(index=False):
        labels = parse_labels(row.final_labels)
        multihot = to_multihot(labels)
        records.append(
            {
                "id": row.question_id,
                "text": row.question_text,
                "labels": multihot,
                "label_names": [t for t, present in zip(CRISIS_TYPES, multihot) if present],
                "temporal": temporal_by_type(labels),
                "flagged": is_flagged(multihot),
            }
        )
    return records


def write_jsonl(records: Iterable[dict[str, Any]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def preprocess(splits: dict[str, Path], output_dir: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    label_counts: dict[str, int] = {crisis_type: 0 for crisis_type in CRISIS_TYPES}
    flagged_counts: dict[str, int] = {}

    for name, path in splits.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Missing CRADLEBench source file for split '{name}': {path}. "
                "Run python src/download_data.py first."
            )
        df = validate_and_clean(pd.read_csv(path), source=path)
        records = build_records(df)
        counts[name] = write_jsonl(records, output_dir / f"{name}.jsonl")
        flagged_counts[name] = sum(1 for r in records if r["flagged"])
        if name == "train":
            for record in records:
                for crisis_type in record["label_names"]:
                    label_counts[crisis_type] += 1

    manifest = {
        "sources": {name: str(path) for name, path in splits.items()},
        "counts": counts,
        "flagged_counts": flagged_counts,
        "crisis_types": list(CRISIS_TYPES),
        "train_label_counts": label_counts,
        "task": "crisis_type_multilabel_classification",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_SPLITS["train"])
    parser.add_argument("--validation", type=Path, default=DEFAULT_SPLITS["validation"])
    parser.add_argument("--test", type=Path, default=DEFAULT_SPLITS["test"])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    splits = {"train": args.train, "validation": args.validation, "test": args.test}
    counts = preprocess(splits, args.output_dir)
    print(f"Wrote CRADLEBench classifier splits to {args.output_dir}: {counts}")


if __name__ == "__main__":
    main()
