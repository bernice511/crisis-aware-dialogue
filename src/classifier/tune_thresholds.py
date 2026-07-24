"""Pick per-class decision thresholds from validation data, then score test data with them.

The trainer scores every crisis type at a single fixed threshold (0.5), which
is a reasonable default but not necessarily the F1-maximizing cutoff for a
class as rare as childabuse_endangerment (182/4181 train examples). This
sweeps a threshold per class on validation sigmoid scores, then applies the
chosen thresholds (fit once, unseen by test) to the test split.

Usage:
    python -m src.classifier.tune_thresholds --model outputs/cradlebench-roberta-classifier/final_model
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.classifier.labels import CRISIS_TYPES
from src.classifier.metrics import compute_multilabel_metrics

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = PROJECT_ROOT / "data/processed/cradlebench_clf"

THRESHOLD_GRID = np.round(np.arange(0.05, 0.96, 0.05), 2)


def load_split(path: Path) -> tuple[list[str], np.ndarray]:
    texts: list[str] = []
    labels: list[list[int]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            texts.append(record["text"])
            labels.append(record["labels"])
    return texts, np.array(labels, dtype=int)


def score_probs(model_dir: Path, texts: list[str], max_length: int, batch_size: int) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    all_probs = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            inputs = tokenizer(
                batch, truncation=True, max_length=max_length, padding=True, return_tensors="pt"
            ).to(device)
            logits = model(**inputs).logits
            all_probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(all_probs, axis=0)


def fit_thresholds(val_labels: np.ndarray, val_probs: np.ndarray) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for i, crisis_type in enumerate(CRISIS_TYPES):
        y_true = val_labels[:, i]
        best_t, best_f1 = 0.5, -1.0
        for t in THRESHOLD_GRID:
            y_pred = (val_probs[:, i] >= t).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        thresholds[crisis_type] = best_t
    return thresholds


def apply_thresholds(probs: np.ndarray, thresholds: dict[str, float]) -> np.ndarray:
    cols = [(probs[:, i] >= thresholds[t]).astype(int) for i, t in enumerate(CRISIS_TYPES)]
    return np.stack(cols, axis=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Path to a saved final_model directory")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path, default=None, help="Where to write the results JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    val_texts, val_labels = load_split(args.data_dir / "validation.jsonl")
    test_texts, test_labels = load_split(args.data_dir / "test.jsonl")

    val_probs = score_probs(args.model, val_texts, args.max_length, args.batch_size)
    test_probs = score_probs(args.model, test_texts, args.max_length, args.batch_size)

    thresholds = fit_thresholds(val_labels, val_probs)

    default_preds = (test_probs >= 0.5).astype(int)
    tuned_preds = apply_thresholds(test_probs, thresholds)

    default_metrics = compute_multilabel_metrics(test_labels, default_preds)
    tuned_metrics = compute_multilabel_metrics(test_labels, tuned_preds)

    print(f"Fitted thresholds (from validation): {thresholds}\n")
    print(f"{'crisis_type':28s} {'thr':>5s} {'f1@0.5':>8s} {'f1@tuned':>9s}")
    for crisis_type in CRISIS_TYPES:
        print(
            f"{crisis_type:28s} {thresholds[crisis_type]:5.2f} "
            f"{default_metrics[f'f1_{crisis_type}']:8.4f} {tuned_metrics[f'f1_{crisis_type}']:9.4f}"
        )
    print(
        f"\n{'micro/macro':28s}       "
        f"{default_metrics['f1_micro']:.4f}/{default_metrics['f1_macro']:.4f}   "
        f"{tuned_metrics['f1_micro']:.4f}/{tuned_metrics['f1_macro']:.4f}"
    )
    print(
        f"{'flagged (acc/f1)':28s}       "
        f"{default_metrics['flagged_accuracy']:.4f}/{default_metrics['flagged_f1']:.4f}   "
        f"{tuned_metrics['flagged_accuracy']:.4f}/{tuned_metrics['flagged_f1']:.4f}"
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {"thresholds": thresholds, "default_metrics": default_metrics, "tuned_metrics": tuned_metrics},
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
