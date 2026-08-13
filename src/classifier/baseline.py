"""TF-IDF + logistic regression multi-label baseline for CRADLEBench.

This is the "simple baseline classifier" comparison point in
project_framework.md.txt's planned comparisons (trained RoBERTa vs. baseline).
It reads the same processed splits src/classifier/preprocess.py writes and is
scored with the same metrics as the RoBERTa classifier (src/classifier/metrics.py),
so the two are directly comparable. No class weighting is applied by default,
matching the RoBERTa run's unweighted-first baseline pass.

Usage:
    python -m src.classifier.baseline
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from src.classifier.metrics import compute_multilabel_metrics

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = PROJECT_ROOT / "data/processed/cradlebench_clf"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/cradlebench-bow-baseline"


def load_split(path: Path) -> tuple[list[str], np.ndarray]:
    texts: list[str] = []
    labels: list[list[int]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            texts.append(record["text"])
            labels.append(record["labels"])
    return texts, np.array(labels, dtype=int)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-features", type=int, default=20000)
    parser.add_argument("--ngram-max", type=int, default=2)
    parser.add_argument("--c", type=float, default=1.0, help="Inverse regularization strength")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--balanced", action="store_true", help="Use class_weight='balanced'")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    missing = [
        args.data_dir / f"{split}.jsonl"
        for split in ("train", "validation", "test")
        if not (args.data_dir / f"{split}.jsonl").exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing processed data: {missing}. Run python -m src.classifier.preprocess first."
        )

    train_texts, train_labels = load_split(args.data_dir / "train.jsonl")
    val_texts, val_labels = load_split(args.data_dir / "validation.jsonl")
    test_texts, test_labels = load_split(args.data_dir / "test.jsonl")

    vectorizer = TfidfVectorizer(
        max_features=args.max_features,
        ngram_range=(1, args.ngram_max),
        stop_words="english",
        sublinear_tf=True,
    )
    x_train = vectorizer.fit_transform(train_texts)
    x_val = vectorizer.transform(val_texts)
    x_test = vectorizer.transform(test_texts)

    classifier = OneVsRestClassifier(
        LogisticRegression(
            C=args.c,
            max_iter=1000,
            class_weight="balanced" if args.balanced else None,
        )
    )
    classifier.fit(x_train, train_labels)

    val_preds = (classifier.predict_proba(x_val) >= args.threshold).astype(int)
    test_preds = (classifier.predict_proba(x_test) >= args.threshold).astype(int)

    val_metrics = compute_multilabel_metrics(val_labels, val_preds)
    test_metrics = compute_multilabel_metrics(test_labels, test_preds)

    print("Validation metrics:")
    for key, value in val_metrics.items():
        print(f"  {key}: {value:.4f}")
    print("\nTest metrics:")
    for key, value in test_metrics.items():
        print(f"  {key}: {value:.4f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "validation_metrics.json").write_text(json.dumps(val_metrics, indent=2), encoding="utf-8")
    (args.output_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
    joblib.dump({"vectorizer": vectorizer, "classifier": classifier}, args.output_dir / "model.joblib")
    print(f"\nSaved baseline model + metrics to {args.output_dir}")


if __name__ == "__main__":
    main()
